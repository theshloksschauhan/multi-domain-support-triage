# ADR 001: Inverted index beside full-scan BM25

## Status

Accepted

## Context

Ticket retrieval originally scored every chunk with `BM25Index`. That full scan is the local-mode behavior and the parity oracle. A shard needs a posting list so it does not read every document for every term, while still returning the same ranking.

## Decision

Keep `BM25Index` as the default local backend. Use `InvertedIndex` inside each shard. Document frequency still comes from the set of tokens on a document, and the score still comes from `term_freq`. The inverted index accepts optional global statistics so a shard can score with corpus-wide values instead of its own local counts.

## Consequences

Local mode does not change. Shard search can be compared directly with a full scan of the same documents. The inverted index is not a second scoring formula.

## Complexity

Building the posting lists reads every token once, so build time is linear in the number of tokens. A query walks the posting lists of its own terms and scores those documents. The coordinator merges the top hits from each partition, which is linear in the number of partitions after those hits have come back.
