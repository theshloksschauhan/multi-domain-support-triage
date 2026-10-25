# ADR 005: One SQLite file per shard

## Status

Accepted

## Context

The shard process must survive a restart without a new database server and without extra packages. The search itself stays on the in-memory inverted index; the database is the durable copy of one partition.

## Decision

Each shard owns one SQLite file in WAL mode. Index builds stage rows, then commit them into the live tables only when the document count matches. A crash during a batch leaves the previous generation queryable. Queries do not scan SQLite per term. There is no replication and no shared database between shards.

## Consequences

Restarting a process reloads the last committed generation. Changing shard count still requires a rebuild. Database files, WAL files, and shared-memory files stay out of git.
