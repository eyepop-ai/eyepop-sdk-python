# eyepop-sdk-python

Official EyePop.ai Python SDK (`eyepop` on PyPI) — async/sync client for the Worker (CV inference)
and Data (datasets / VLM / evaluation) APIs. Runs on `uv`, Python ≥ 3.12.

## Commands
`task --list-all` for tasks. Run `task check` before done (lint + typecheck-changed + test).

## Gotchas
- Typecheck gate is **basedpyright**, not mypy — despite a `[tool.mypy]` block + mypy in dev deps. Fixing
  types against mypy won't match CI.
- Integration tests hit **real EyePop endpoints** and are skipped by default (`addopts = --ignore=tests/integration`,
  and each needs `EYEPOP_API_KEY`). They run nightly or manually (`uv run pytest tests/integration/ --timeout=300`).
- Version is git-derived (`setuptools_scm`) — building from a worktree/shallow clone misversions; publish asserts
  `eyepop.__version__ == release tag` with `fetch-depth: 0`.
- Auth env is layered: `EYEPOP_API_KEY` (or `EYEPOP_ACCESS_TOKEN`); `EYEPOP_ACCOUNT_ID` required for the Data API;
  `EYEPOP_URL` defaults to **production** — override for staging. `EYEPOP_SECRET_KEY` is deprecated. In the
  composable-pop API, `model=`/`modelUuid=` are deprecated (use `ability=`) and `set_pop()` takes only `Pop` objects.

## See also
- `.claude/skills/eyepop-sdk` — SDK usage patterns, auth/vault, composable Pop, examples
