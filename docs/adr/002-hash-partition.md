# ADR 002: Hash partitioning instead of company partitioning

## Status

Accepted

## Context

The corpus has three companies, but chunk counts are uneven and a ticket with company `None` must search every company. Putting one company on one shard would leave that shard as a single point of failure for those tickets and would make shard sizes follow the corpus layout.

## Decision

Assign each document by SHA-256 of its exact `doc_id` string, including Windows backslashes. The first eight digest bytes are an integer, then modulo the configured shard count. Slot order is the order of shards in the config file. Changing the count or the order requires a full rebuild. The index generation includes that shard order and the sorted document ids with token counts.

## Consequences

Company filters still apply at query time on every shard. Partitions stay disjoint. A rebuilt cluster with a different shard list does not reuse an old generation.
