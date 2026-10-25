# ADR 003: Global IDF across shards

## Status

Accepted

## Context

BM25 document frequency and average document length are properties of the whole corpus. If each shard used only its own counts, the same term would get a different score on each shard and the merged ranking would not match `BM25Index`.

## Decision

A query has two phases. Every ready shard reports `n_docs`, `total_tokens`, and document frequency for the query terms. The coordinator sums those values, checks that every shard is on the same generation, and sends that one statistic set back with the search. Each shard scores with it. The coordinator merges by descending score and ascending `doc_id`.

## Consequences

One extra round trip is required before scoring. A shard that was not part of the generation cannot be mixed in. Local IDF is not used for distributed queries.
