# Triage labels

Skills like `triage`, `to-issues`, and `qa` apply these labels to GitHub Issues.

## Label vocabulary

| Role | GitHub label | Applied when |
|------|-------------|-------------|
| `needs-triage` | `needs-triage` | New issue, not yet evaluated by a maintainer |
| `needs-info` | `needs-info` | Issue requires more details from the reporter before it can be triaged |
| `ready-for-agent` | `ready-for-agent` | Issue is fully specified — an AFK agent can implement it with no human context |
| `ready-for-human` | `ready-for-human` | Issue is triaged but requires human implementation (too complex, needs design, etc.) |
| `wontfix` | `wontfix` | Issue will not be actioned — closed with explanation |

## State machine

```
                   ┌──────────────┐
                   │ needs-triage │
                   └──────┬───────┘
                          │ maintainer evaluates
              ┌───────────┼───────────┐
              ▼           ▼           ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────┐
    │ needs-info  │  │ready-for-   │  │ wontfix │
    └──────┬──────┘  │  agent      │  └─────────┘
           │         └─────────────┘
           │ reporter responds
           ▼
    ┌──────────────┐
    │ needs-triage │  (re-evaluate)
    └──────────────┘
```

## Usage notes

- Always apply exactly ONE of `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, or `wontfix` at a time.
- `ready-for-human` is a fallback — prefer `ready-for-agent` when the spec is complete enough.
- Bug/feature type labels (e.g. `bug`, `enhancement`) may co-exist with triage labels — they are orthogonal.
