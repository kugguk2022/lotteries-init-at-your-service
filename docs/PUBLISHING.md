# Publishing LottoBench

LottoBench is an alpha distribution published through PyPI Trusted Publishing. No PyPI password or
long-lived API token belongs in GitHub Secrets.

## One-time setup

1. Create protected GitHub environments named `testpypi` and `pypi`; require reviewer approval for
   `pypi`.
2. Configure pending trusted publishers on TestPyPI and PyPI:
   - project: `lottobench`
   - owner: `kugguk2022`
   - repository: `lotteries-init-at-your-service`
   - workflows: `publish-testpypi.yml` and `publish-pypi.yml`
   - matching environments: `testpypi` and `pypi`

Name availability is secured only by the first accepted upload.

## Rehearse on TestPyPI

Run `publish-testpypi` manually, then install in a clean environment:

```bash
python -m venv .venv-testpypi
.venv-testpypi/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  lottobench==0.1.0a1
lottobench games
```

## Publish

After CI and the TestPyPI rehearsal pass, create a GitHub release tagged `v0.1.0a1`. The production
workflow rejects a tag that does not exactly match `project.version`, builds fresh artifacts, and
waits for approval in the protected `pypi` environment.

PyPI versions are immutable. Publish fixes under a new version such as `0.1.0a2`.
