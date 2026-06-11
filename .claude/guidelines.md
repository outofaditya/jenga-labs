# Blockchain Labs — Guidelines

Sole reference for code style, commit conventions, doc upkeep, and the two protocol commands.

---

## Communication style
- **Small chunks.** No walls of text. Break concepts across messages when needed.
- **Concepts before code.** Always cover the concept in plain language with a worked example first. Then implement. Then verify. Then commit. Repeat.
- **Validate before moving on.** After each chunk, user paraphrases or confirms; assistant validates with corrections in a single sentence where possible.
- **No assumed prior knowledge.** Explain from scratch on request.
- **No fluff.** Skip filler text. State results and next steps directly.

---

## Code style

### Hard rules
- **Production-grade.** No experimental scaffolding, no half-finished branches, no commented-out code blocks.
- **Highly optimized.** Performance is graded. Pick the data structure and algorithm with the lowest realistic complexity. Hot paths get hand-tuned (precomputation, amortization, multiprocessing for CPU-bound, asyncio for I/O-bound).
- **Highly compressed.** Fewest lines that still read cleanly. No accidental duplication; no helper that wraps a one-line call. No over-abstraction.
- **Zero dead code.** Unreferenced names, unreached branches, kept-for-history blocks — all removed.
- **Lowercase comments only.** No capitalized first letters, no full-sentence punctuation. Comments explain *why*, not *what*. Default: write none.
- **Ruff-formatted and ruff-clean.** `ruff format` handles formatting (black-compatible); `ruff check` handles linting. No suppressions unless commented with a one-line reason.

### Ascending character-length sort
Apply everywhere the order does not change semantics. Shortest line goes first.

Applies to:
- Import statements within a group.
- Top-level definitions (constants, functions, classes) within a file.
- Dataclass / class attribute declarations when order is decorative (i.e., payload field order is dictated by wire spec — that overrides sort).
- Items in list/tuple/set/dict literals where order is decorative (NOT where iteration order is load-bearing).
- Keyword-argument groups in function calls.

Does NOT apply to (semantics override):
- Positional function parameters.
- Wire-format field ordering in payload dataclasses (spec wins).
- Lists where order has meaning (e.g., `MEMBER_KEYS` is registration order).
- Dict keys where iteration order affects behavior.

### Other conventions
- **Constants ALL_CAPS at module top.** Precompute expensive operations (hex→bytes, etc.) once at load time. Never inside hot paths.
- **Type-annotated dataclasses for IPv8 payloads.** Let IPv8 infer wire format from annotations.
- **Helpers prefixed with `_`** (`_server_peer`, `_sign`) for class-internal use.
- **Byte-exact serialization.** For headers and protocol-defined blobs, explicit `struct.pack(">QIQ", ...)`. Never let Python pick widths or endianness.
- **No `print` in library code.** Top-level entry points print user-facing output; everything else uses `logging` at appropriate level.

---

## Git
- **Conventional commits.** `<type>(<scope>): <message>`.
- Types: `feat`, `refactor`, `perf`, `docs`, `fix`, `chore`.
- Scope = file or module name (e.g. `miner`, `client`, `signer`, `node`, `chain`).
- Message: lowercase, imperative, minimal. No trailing period.
- **No commas in commit messages.** Rephrase or split into separate commits.
- **No description body.** Subject line only.
- **No co-authored-by or claude watermark.** Subject line only — nothing else.
- One logical change per commit.
- **Push after every commit** as part of the `/verify` flow.

## Branching
- Working on `main` directly for this course.

---

## Documents
Working docs live under `.claude/` at the repo root and are gitignored (local-only):
- **`.claude/plan.md`** — atoms across all assignments. Each atom contains: goal, concept anchor, numbered tasks, test, commit shape.
- **`.claude/guidelines.md`** — this file. Stable; updated only when a new convention is agreed.
- **`.claude/commands/verify.md`** — the `/verify` slash command implementation.
- **`.claude/commands/checkpoint.md`** — the `/checkpoint` slash command implementation.

`README.md` lives in the repo root and is the only user-facing document. It must:
- Read like a polished big-tech project README: clear, detailed, no meta-commentary.
- Never mention `.claude/`, planning materials, the verify/checkpoint workflow, AI assistance, or "in progress" status tags.
- Use **Title Case** for every heading.
- Use Title Case for proper nouns and product names in body text (Python, IPv8, SHA-256, TU Delft, GitHub, UDP, etc.). Regular descriptive prose stays sentence case.

`plan.md` is updated at atom boundaries (start: mark in-progress; end: mark complete with notes if anything diverged from the spec).

---

## Output format for `/verify` and `/checkpoint`
Both commands emit results as fixed-width-aligned bullets:

```
- [STATUS] Step name          — short result detail
```

- `STATUS` is one of: `PASS`, `FIXED`, `SKIP`, `FAIL`.
  - `PASS` — step passed with no change.
  - `FIXED` — issue found and fixed inline.
  - `SKIP` — not applicable to this run (e.g., README check with no public surface change).
  - `FAIL` — blocking issue; stop the flow and explain.
- Step name is padded to a consistent column width.
- Result detail follows an em-dash, kept to one short clause.
- No prose summary unless a `FAIL` occurred.

---

## `/verify` command
Runs **after every task** of every atom. Seven steps, in order:

1. **Correctness** — re-read the just-written code, trace through the happy path mentally, confirm types and contract.
2. **Optimization sweep** — scan for any obvious perf wins: precomputation, hot-loop allocations, redundant syscalls, unnecessary copies. Apply or note.
3. **Comment hygiene** — every comment lowercase, no full sentences, explains *why* not *what*. Remove obvious comments. Default to none.
4. **Ascending-length sort** — verify imports / definitions / decorative literals follow ascending character-length.
5. **README check** — if the change affects public surface (CLI args, file layout, run instructions, dependencies), update `README.md`.
6. **Format + lint** — run `ruff format .` then `ruff check --fix .` on the repo.
7. **Commit + push** — stage relevant files, write a conventional commit message, commit, push to origin.

## `/checkpoint` command
Runs **only after an atom is complete**. File-by-file deep pass:

For each file in the repo (one at a time, even if visited before in this atom):

1. **Re-read the file fresh.** Build a mental model from scratch.
2. **Dead code hunt.** Unused imports, unreachable branches, helpers with one call site that's a one-liner, leftover scaffolding.
3. **Optimization hunt.** Tight-loop allocations, redundant work, algorithmically suboptimal paths, expensive operations inside callbacks, missing caching.
4. **Compression pass.** Combine related lines, eliminate temporary variables that are used once, replace verbose patterns with idiomatic Python equivalents.
5. **Run `/verify`** on the resulting file.

Output uses the same bullet format, grouped per file under a `### <file path>` header. Closes with a single `### Checkpoint complete — N files reviewed, M files modified.` line.
