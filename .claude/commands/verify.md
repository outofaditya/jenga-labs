---
description: Run the 7-step verify protocol on the just-written code
---

You are now running the `/verify` command. Refer to `.claude/guidelines.md` for the full protocol. Execute the seven steps below in order. After each step, emit exactly one bullet using the status-tag format.

## Steps

1. **Correctness** — re-read the just-written code, mentally trace the happy path, confirm types and the contract.
2. **Optimization sweep** — scan for any obvious perf wins: precomputation, hot-loop allocations, redundant syscalls, unnecessary copies. Apply or note.
3. **Comment hygiene** — every comment lowercase, no full sentences, explains *why* not *what*. Remove obvious comments.
4. **Ascending-length sort** — verify imports / definitions / decorative literals follow ascending character-length. Re-sort if needed.
5. **README check** — if the change affects public surface (CLI args, file layout, run instructions, dependencies), update `README.md` accordingly.
6. **Format + lint** — run `ruff format .` then `ruff check --fix .` on the repo.
7. **Commit + push** — stage relevant files, write a conventional commit message (`<type>(<scope>): <message>`), commit, push to origin.

## Output format

Output only the bullet block. No prose introduction, no closing summary, no extra commentary — unless a step status is `FAIL`, in which case stop the flow and briefly explain the blocking issue after the bullets.

Use this exact bullet layout (status tag in brackets first, step name padded to 20 characters, em-dash, then a one-clause result):

```
- [PASS]  Correctness          — happy path traced, types match
- [FIXED] Optimization         — moved hex decode to module load
- [PASS]  Comment hygiene      — all lowercase, no descriptive noise
- [FIXED] Sort order           — imports + 2 dict literals re-sorted
- [SKIP]  README check         — no public surface changed
- [PASS]  Format + lint        — ruff clean, no diff
- [PASS]  Commit + push        — perf(chain): cache header bytes
```

Status tags:
- `PASS` — step passed with no change.
- `FIXED` — issue found and fixed inline as part of this verify run.
- `SKIP` — not applicable to this run.
- `FAIL` — blocking issue; stop and explain.
