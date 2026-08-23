"""Failure modes a retrieval attempt can end in.

Separated from the adapters so callers can catch them without importing an adapter (and therefore
without importing that adapter's optional dependencies).
"""

from __future__ import annotations


class FetchError(RuntimeError):
    """Every configured source failed. The message lists what each one did."""


class ContentTypeError(RuntimeError):
    """A source answered, but with something that is not a text payload.

    Usually an error page or a captcha served with HTTP 200, which is exactly the case that would
    otherwise be normalized into plausible-looking nonsense.
    """


class NormalizationError(ValueError):
    """A payload was retrieved but could not be read as draw history."""
