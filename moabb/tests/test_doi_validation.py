"""DOI validation for MOABB dataset metadata.

Treats the DOI in each dataset class (``__init__``) as ground truth,
then validates that:

1. All DOIs (class, METADATA, docstring) have valid format.
2. Docstring DOIs are tracked in METADATA.
3. Every DOI resolves via doi.org content negotiation (works for
   CrossRef, DataCite, and Medra registries).
4. When class DOI and metadata DOI differ, they share at least one author.

Run all (including network tests)::

    python -m pytest moabb/tests/test_doi_validation.py --timeout=300 -v

Run only offline checks::

    python -m pytest moabb/tests/test_doi_validation.py -k "not network" -v
"""

import re
import time

import httpx
import pytest

from moabb.datasets.metadata.schema import DatasetMetadata
from moabb.datasets.utils import dataset_list


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SKIP_CLASSES = {"FakeDataset", "FakeVirtualRealityDataset"}

# Non-standard identifiers (not resolvable as DOIs)
_NON_DOI_PREFIXES = ("hal-", "tel-", "arXiv:")

_REQUEST_DELAY = 0.15  # polite rate-limit

# ---------------------------------------------------------------------------
# DOI helpers
# ---------------------------------------------------------------------------

_DOI_URL_PREFIXES = ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/")
_DOI_RE = re.compile(r"^10\.\d{4,}/")


def _normalize_doi(value: str | None) -> str | None:
    """Strip URL prefix from a DOI, returning the bare ``10.xxxx/…`` form."""
    if not value:
        return None
    for prefix in _DOI_URL_PREFIXES:
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _is_doi(value: str) -> bool:
    """Return True if *value* looks like a DOI (``10.xxxx/…``)."""
    return bool(value and _DOI_RE.match(_normalize_doi(value) or ""))


def _extract_docstring_dois(cls) -> list[str]:
    """Return deduplicated DOIs found anywhere in *cls.__doc__*."""
    doc = getattr(cls, "__doc__", "") or ""
    raw = re.findall(r"10\.\d{4,}/[^\s\]\">]+", doc)
    cleaned = []
    for d in raw:
        d = d.rstrip(".,;:)")
        d = d.rstrip("`")
        if ">`_" in d:
            d = d[: d.index(">")]
        d = d.rstrip("`_>")
        if d.endswith("/abstract"):
            d = d[: -len("/abstract")]
        cleaned.append(d)
    return list(dict.fromkeys(cleaned))


# ---------------------------------------------------------------------------
# DOI collection per dataset
# ---------------------------------------------------------------------------


def _collect_dois(cls) -> dict[str, str | None]:
    """Return ``{source_label: doi}`` for all DOI sources in *cls*.

    Sources:
        ``init.doi``                  — ground-truth DOI from ``__init__``
        ``metadata.doi``              — ``METADATA.documentation.doi``
        ``metadata.associated_paper`` — ``METADATA.documentation.associated_paper_doi``
        ``docstring.<N>``             — DOIs extracted from the class docstring
    """
    result: dict[str, str | None] = {}

    # 1. Ground-truth DOI from __init__
    try:
        instance = cls()
        result["init.doi"] = _normalize_doi(getattr(instance, "doi", None))
    except Exception:
        result["init.doi"] = None

    # 2. METADATA documentation DOIs
    meta = getattr(cls, "METADATA", None)
    if isinstance(meta, DatasetMetadata):
        doc = getattr(meta, "documentation", None)
        if doc:
            result["metadata.doi"] = _normalize_doi(getattr(doc, "doi", None))
            result["metadata.associated_paper"] = _normalize_doi(
                getattr(doc, "associated_paper_doi", None)
            )

    # 3. Docstring DOIs
    for i, doi in enumerate(_extract_docstring_dois(cls)):
        result[f"docstring.{i}"] = doi

    return result


# ---------------------------------------------------------------------------
# DOI resolution via doi.org content negotiation
# Uses citeproc+json — works for CrossRef, DataCite, and Medra DOIs.
# ---------------------------------------------------------------------------


def _resolve_doi(doi: str) -> dict | None:
    """Resolve *doi* via doi.org content negotiation.

    Returns ``{title, authors, year}`` or ``None`` on failure.
    """
    try:
        time.sleep(_REQUEST_DELAY)
        r = httpx.get(
            f"https://doi.org/{doi}",
            headers={"Accept": "application/citeproc+json"},
            follow_redirects=True,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        title = data.get("title")
        authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in data.get("author", [])
        ]
        year = None
        issued = data.get("issued", {}).get("date-parts", [[None]])
        if issued and issued[0] and issued[0][0]:
            year = issued[0][0]
        return {"title": title, "authors": authors, "year": year, "doi": doi}
    except Exception:
        return None


def _extract_surnames(authors: list[str]) -> set[str]:
    """Extract lowercase surnames from author name strings."""
    out = set()
    for a in authors or []:
        a = a.strip()
        if not a:
            continue
        if ", " in a:
            # "Last, First" format (DataCite)
            out.add(a.split(",")[0].strip().lower())
        else:
            # "First Last" format (CrossRef)
            parts = a.split()
            if parts:
                out.add(parts[-1].strip(".").lower())
    return out


# ---------------------------------------------------------------------------
# Build parametrized dataset list
# ---------------------------------------------------------------------------

_REAL_DATASETS = [
    cls
    for cls in dataset_list
    if cls.__name__ not in _SKIP_CLASSES
    and isinstance(getattr(cls, "METADATA", None), DatasetMetadata)
]


# ---------------------------------------------------------------------------
# OFFLINE TESTS — no network required
# ---------------------------------------------------------------------------


class TestDOIFormat:
    """Every DOI string should have valid ``10.xxxx/…`` format."""

    @pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=lambda c: c.__name__)
    def test_all_dois_have_valid_format(self, dataset_class):
        dois = _collect_dois(dataset_class)
        invalid = []
        for source, doi in dois.items():
            if doi is None:
                continue
            ok = _is_doi(doi) or any(doi.startswith(p) for p in _NON_DOI_PREFIXES)
            if not ok:
                invalid.append(f"  {source} = {doi!r}")
        if not any(v for v in dois.values()):
            pytest.skip("No DOIs found")
        assert (
            not invalid
        ), f"{dataset_class.__name__}: invalid DOI format:\n" + "\n".join(invalid)


class TestDOIConsistency:
    """Docstring DOIs must be tracked in METADATA or class DOI."""

    @pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=lambda c: c.__name__)
    def test_docstring_dois_tracked_in_metadata(self, dataset_class):
        """Every DOI in the docstring should appear in METADATA or init."""
        dois = _collect_dois(dataset_class)

        # All known DOIs from non-docstring sources
        known = set()
        for key in ("metadata.doi", "metadata.associated_paper", "init.doi"):
            val = dois.get(key)
            if val and _is_doi(val):
                known.add(val)

        # Also check data_url for embedded DOIs
        meta = getattr(dataset_class, "METADATA", None)
        if meta:
            doc = getattr(meta, "documentation", None)
            if doc:
                data_url = getattr(doc, "data_url", None)
                if isinstance(data_url, str):
                    m = re.search(r"10\.\d{4,}/[^\s]+", data_url)
                    if m:
                        known.add(m.group().rstrip(".,;:)"))

        doc_dois = [
            dois[k] for k in sorted(dois) if k.startswith("docstring.") and dois[k]
        ]
        if not doc_dois:
            pytest.skip("No DOIs in docstring")

        # Data-repo prefixes are fine in docstrings without metadata tracking
        _DATA_REPO_PREFIXES = (
            "10.5281/zenodo.",
            "10.7910/DVN/",
            "10.6084/m9.figshare.",
            "10.6094/",
            "10.5524/",
            "10.34973/",
            "10.18115/",
            "10.48550/arXiv.",
        )

        untracked = []
        for d in doc_dois:
            if d in known:
                continue
            if any(d.startswith(p) for p in _DATA_REPO_PREFIXES):
                continue
            if any(d in k or k in d for k in known):
                continue
            untracked.append(d)

        assert not untracked, (
            f"{dataset_class.__name__}: docstring DOIs not tracked in metadata: "
            f"{untracked}\n  Known: {known}"
        )


# ---------------------------------------------------------------------------
# NETWORK TESTS — resolve DOIs via doi.org content negotiation
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestDOIResolution:
    """Every DOI should resolve via doi.org content negotiation."""

    @pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=lambda c: c.__name__)
    def test_all_dois_resolve(self, dataset_class):
        """All unique DOIs from class, METADATA, and docstring must resolve."""
        dois = _collect_dois(dataset_class)
        unique_dois = sorted({d for d in dois.values() if d and _is_doi(d)})
        if not unique_dois:
            pytest.skip("No DOIs to resolve")

        failures = []
        for doi in unique_dois:
            result = _resolve_doi(doi)
            if result is None:
                failures.append(doi)
            elif not result.get("title"):
                failures.append(f"{doi} (resolved but no title)")

        assert (
            not failures
        ), f"{dataset_class.__name__}: DOIs failed to resolve: {failures}"

    @pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=lambda c: c.__name__)
    def test_class_doi_resolves(self, dataset_class):
        """The class DOI (ground truth) must resolve to a real publication."""
        dois = _collect_dois(dataset_class)
        init_doi = dois.get("init.doi")

        if not init_doi or not _is_doi(init_doi):
            pytest.skip("No class DOI")

        result = _resolve_doi(init_doi)
        assert (
            result is not None
        ), f"{dataset_class.__name__}: class DOI {init_doi!r} failed to resolve"
        assert result.get("title"), (
            f"{dataset_class.__name__}: class DOI {init_doi!r} resolved but has "
            f"no title"
        )

    @pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=lambda c: c.__name__)
    def test_class_and_metadata_dois_share_authors(self, dataset_class):
        """Class DOI must share authors with at least one metadata DOI.

        Compares ``init.doi`` against ``metadata.doi`` and
        ``metadata.associated_paper`` — at least one must have
        overlapping author surnames.
        """
        dois = _collect_dois(dataset_class)
        init_doi = dois.get("init.doi")
        if not init_doi or not _is_doi(init_doi):
            pytest.skip("No class DOI")

        # If init.doi exactly matches any metadata DOI, it's consistent
        all_meta = {
            dois.get(k)
            for k in ("metadata.doi", "metadata.associated_paper")
            if dois.get(k)
        }
        if init_doi in all_meta:
            pytest.skip("Class DOI matches a metadata DOI — consistent")

        # Collect metadata DOIs that differ from init for author comparison
        meta_dois = {
            k: v
            for k, v in [
                ("metadata.doi", dois.get("metadata.doi")),
                ("metadata.associated_paper", dois.get("metadata.associated_paper")),
            ]
            if v and _is_doi(v) and v != init_doi
        }
        if not meta_dois:
            pytest.skip("No differing metadata DOIs to compare")

        init_result = _resolve_doi(init_doi)
        if init_result is None:
            pytest.skip(f"Could not resolve init DOI {init_doi!r}")

        init_authors = _extract_surnames(init_result.get("authors"))
        report_lines = [
            f"  class DOI {init_doi}: {init_result.get('title')}",
            f"    authors: {init_result.get('authors')}",
        ]

        for key, meta_doi in meta_dois.items():
            meta_result = _resolve_doi(meta_doi)
            if meta_result is None:
                continue
            meta_authors = _extract_surnames(meta_result.get("authors"))
            if init_authors & meta_authors:
                return  # found overlap — test passes
            report_lines.append(f"  {key} {meta_doi}: {meta_result.get('title')}")
            report_lines.append(f"    authors: {meta_result.get('authors')}")

        pytest.fail(
            f"{dataset_class.__name__}: class DOI shares no authors with "
            f"any metadata DOI.\n" + "\n".join(report_lines)
        )
