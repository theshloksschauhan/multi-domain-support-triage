# ADR 004: Fail closed instead of partial results

## Status

Accepted

## Context

A missing shard does not mean that the missing documents are irrelevant. Returning the hits from the shards that answered would look like a successful search with weaker evidence, and the triage rules could then reply.

## Decision

Distributed mode never falls back to the local full scan. If a shard is down, times out, returns malformed JSON, disagrees on generation, or reports a duplicate document id, the query raises `RetrievalUnavailable`. The pipeline still writes one CSV row per ticket. Non-risk, non-invalid tickets escalate with a retrieval-unavailable justification. Invalid tickets keep the out-of-scope reply. Risk tickets keep the risk escalation. An empty failed result is not passed to the reply path.

## Consequences

Availability of a reply depends on every configured shard. The CSV schema stays the same. Operators can see the failure in the log and, when requested, in the diagnostics JSONL file.
