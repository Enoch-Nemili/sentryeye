# Contributing to SentryEye

Issues and pull requests are welcome.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.check_env        # versions + Apple GPU (MPS) availability
```

## Before opening a pull request

```bash
pip install ruff
ruff check app
```

If your change touches the gate or the cascade, re-run the evaluation and paste the before/after numbers into the PR:

```bash
python -m app.eval_gate   /path/to/TAD/test --out data/test_features.csv
python -m app.eval_system /path/to/TAD/test
```

## Conventions

- One branch and one pull request per change; reference issues with `Closes #N`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, ...).
- Report results with train/test discipline; don't tune on the test split.
- Security issues: see [SECURITY.md](SECURITY.md).
