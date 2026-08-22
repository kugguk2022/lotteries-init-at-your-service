# Publishing `lotteries-core`

The distribution is Alpha. Releases are built by GitHub Actions and published through PyPI Trusted
Publishing; no PyPI password or long-lived API token belongs in GitHub Secrets.

## One-time setup

1. Create and verify accounts on both TestPyPI and PyPI. They are separate services.
2. In GitHub, create protected environments named `testpypi` and `pypi`. Require reviewer approval
   for `pypi`; optionally restrict it to version tags.
3. On TestPyPI, create a pending trusted publisher with:
   - PyPI project name: `lotteries-core`
   - GitHub owner: `kugguk2022`
   - repository: `lotteries-init-at-your-service`
   - workflow: `publish-testpypi.yml`
   - environment: `testpypi`
4. On PyPI, create the equivalent pending publisher using workflow `publish-pypi.yml` and environment
   `pypi`.

The distribution name currently appears unclaimed, but availability is only secured when the first
upload succeeds.

## First Alpha rehearsal

1. Ensure CI is green at the release commit.
2. Run the `publish-testpypi` workflow manually from GitHub Actions.
3. Install from TestPyPI in a fresh environment while resolving dependencies from normal PyPI:

   ```bash
   python -m venv .venv-testpypi
   .venv-testpypi/bin/python -m pip install \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     lotteries-core==0.1.0a1
   ```

4. Run the installed smoke test or reproduce the README example.

## Publish to real PyPI

Create and publish a GitHub release whose tag matches the immutable package version, for example
`v0.1.0a1`. The `publish-pypi` workflow builds fresh distributions from that tag and waits at the
protected `pypi` environment before OIDC publication.

PyPI versions cannot be replaced. Fixes require a new version such as `0.1.0a2`; never try to reuse
an uploaded version number.
