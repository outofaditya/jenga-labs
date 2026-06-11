---
description: Run the deep file-by-file checkpoint pass after an atom is complete
---

You are now running the `/checkpoint` command. Refer to `.claude/guidelines.md` for the full protocol.

## Scope

Loop through every source file in the repository, ordered shortest-first by total line count. Exclude `.venv/`, `.git/`, `__pycache__/`, `.claude/`, `tasks/`, and any `*.pem` files.

For each file, perform the five steps below — even if the file has already been visited earlier in this checkpoint run.

## Per-file steps

1. **Re-read fresh** — open the file without leaning on any prior mental model. Re-derive intent from scratch.
2. **Dead code hunt** — unused imports, unreachable branches, helpers with one trivial call site, leftover scaffolding, commented-out code, vestigial flags.
3. **Optimization hunt** — tight-loop allocations, redundant work, algorithmically suboptimal paths, expensive operations inside callbacks, missing caching, unnecessary `await`s.
4. **Compression pass** — combine related lines, eliminate single-use temporaries, replace verbose patterns with idiomatic Python equivalents, fold one-line helpers back into call sites.
5. **/verify sweep** — run the full 7-step `/verify` protocol on the resulting file before moving to the next.

## Output format

Group results per file under a level-3 markdown header showing the file path, then a 5-bullet block matching the verify status-tag layout. Use a single trailing summary line.

```
### chain.py

- [PASS]  Re-read              — pure block primitives, no surprises
- [FIXED] Dead code            — dropped unused `_legacy_pack` helper
- [FIXED] Optimization         — folded txs_hash empty case into precomputed const
- [FIXED] Compression          — merged 4-line struct call into single literal
- [PASS]  /verify              — clean pass, perf(chain): tighten primitives

### node.py

- [PASS]  Re-read              — community boot + handlers
- [PASS]  Dead code            — none
- [FIXED] Optimization         — moved SERVER_PUBLIC_KEY decode to module load
- [PASS]  Compression          — already tight
- [PASS]  /verify              — clean pass, perf(node): cache server pubkey

### Checkpoint complete — 2 files reviewed, 2 files modified.
```

Status tags follow the same definitions as `/verify`:
- `PASS` — no change needed.
- `FIXED` — issue found and fixed inline.
- `SKIP` — not applicable.
- `FAIL` — blocking issue; halt the checkpoint and explain.

No prose anywhere except inline result clauses and the final summary line.
