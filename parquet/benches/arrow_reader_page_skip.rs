// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Reading a `RowSelection` that skips many whole pages.
//!
//! Row groups of 1,048,576 rows are written with 512-row pages, so a column chunk has
//! 2,048 pages. The selections range from a single short run (most pages skipped) to
//! runs shorter than a page (no page can be skipped whole). Each case is read with the
//! offset index, and the needle case also without it.

use std::hint;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use bytes::Bytes;
use criterion::*;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection,
    RowSelectionPolicy, RowSelector,
};
use parquet::file::metadata::PageIndexPolicy;
use parquet::file::properties::WriterProperties;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const ROW_GROUP_ROWS: usize = 1 << 20;
const ROW_GROUPS: usize = 2;
const ROWS: usize = ROW_GROUP_ROWS * ROW_GROUPS;
const PAGE_ROWS: usize = 512;

fn write_file() -> Bytes {
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(ROW_GROUP_ROWS))
        .set_data_page_row_count_limit(PAGE_ROWS)
        .set_write_batch_size(PAGE_ROWS)
        .build();
    let ids: ArrayRef = Arc::new(Int64Array::from_iter_values(0..ROWS as i64));
    let names: ArrayRef = Arc::new(StringArray::from_iter_values(
        (0..ROWS).map(|i| format!("name-{}", i % 1000)),
    ));
    let batch = RecordBatch::try_from_iter([("id", ids), ("name", names)]).unwrap();
    let mut out = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut out, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    Bytes::from(out)
}

/// Alternating runs, starting with a skip, whose lengths come from `next_len`.
fn runs(mut next_len: impl FnMut(bool) -> usize) -> RowSelection {
    let mut selectors = vec![];
    let mut remaining = ROWS;
    let mut skip = true;
    while remaining > 0 {
        let len = next_len(skip).clamp(1, remaining);
        selectors.push(RowSelector {
            row_count: len,
            skip,
        });
        remaining -= len;
        skip = !skip;
    }
    selectors.into()
}

fn selections() -> Vec<(&'static str, RowSelection)> {
    let needle = RowSelection::from(vec![
        RowSelector::skip(ROW_GROUP_ROWS + 300_001),
        RowSelector::select(8),
        RowSelector::skip(ROW_GROUP_ROWS - 300_009),
    ]);
    let sparse = runs(|skip| if skip { 65_535 } else { 1 });
    let mut rng = StdRng::seed_from_u64(42);
    let random = runs(|skip| {
        if skip {
            rng.random_range(1..20_000)
        } else {
            rng.random_range(1..64)
        }
    });
    let short = runs(|skip| if skip { 300 } else { 100 });
    vec![
        ("needle", needle),
        ("sparse", sparse),
        ("random", random),
        ("short_runs", short),
    ]
}

fn read(bytes: &Bytes, metadata: &ArrowReaderMetadata, selection: &RowSelection) -> usize {
    ParquetRecordBatchReaderBuilder::new_with_metadata(bytes.clone(), metadata.clone())
        .with_batch_size(8192)
        .with_row_selection(selection.clone())
        .with_row_selection_policy(RowSelectionPolicy::Selectors)
        .build()
        .unwrap()
        .map(|batch| batch.unwrap().num_rows())
        .sum()
}

fn page_skip_benchmark(c: &mut Criterion) {
    let bytes = write_file();
    let load = |policy| {
        let options = ArrowReaderOptions::new().with_page_index_policy(policy);
        ArrowReaderMetadata::load(&bytes, options).unwrap()
    };
    let with_index = load(PageIndexPolicy::Required);
    let without_index = load(PageIndexPolicy::Skip);

    let mut group = c.benchmark_group("arrow_reader_page_skip");
    for (name, selection) in selections() {
        group.bench_function(format!("page_index/{name}"), |b| {
            b.iter(|| hint::black_box(read(&bytes, &with_index, &selection)))
        });
        if name == "needle" {
            group.bench_function(format!("no_page_index/{name}"), |b| {
                b.iter(|| hint::black_box(read(&bytes, &without_index, &selection)))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, page_skip_benchmark);
criterion_main!(benches);
