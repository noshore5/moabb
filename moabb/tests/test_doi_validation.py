"""Systematic DOI validation for MOABB dataset metadata.

Uses the CrossRef API (via habanero) and DataCite API to verify that
every DOI in dataset METADATA and docstrings resolves to a real
publication and that the referenced paper matches the dataset description.

Run with:
    python -m pytest moabb/tests/test_doi_validation.py -x --timeout=300 -v

Or just the quick offline checks:
    python -m pytest moabb/tests/test_doi_validation.py -k "not network" -v
"""

import json
import re
import time
import urllib.request

import pytest

from moabb.datasets.metadata.schema import DatasetMetadata
from moabb.datasets.utils import dataset_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# DOIs that are not resolvable via CrossRef (HAL, arXiv IDs, etc.)
_NON_CROSSREF_PREFIXES = (
    "hal-",
    "tel-",
    "arXiv:",
)

# DOI prefixes registered with DataCite (not CrossRef)
_DATACITE_PREFIXES = (
    "10.5281/zenodo.",  # Zenodo
    "10.6084/m9.figshare.",  # Figshare
    "10.48550/arXiv.",  # arXiv
    "10.7910/DVN/",  # Harvard Dataverse
    "10.6094/",  # FreiDok (Freiburg)
    "10.5524/",  # GigaDB
    "10.34973/",  # Radboud Data Repository
    "10.18115/",  # ERPSS
    "10.35376/",  # University of Valladolid
    "10.3217/",  # TU Graz conference proceedings
)

# Test / fake datasets to skip
_SKIP_CLASSES = {"FakeDataset", "FakeVirtualRealityDataset"}

# Rate-limit: be polite to APIs
_REQUEST_DELAY = 0.15  # seconds between requests


def _is_doi(value: str) -> bool:
    """Check if a string looks like a DOI (starts with 10.xxxx/)."""
    if not value:
        return False
    return bool(re.match(r"^10\.\d{4,}/", value))


def _extract_docstring_dois(cls) -> list[str]:
    """Extract all DOI-like strings from a class docstring."""
    doc = getattr(cls, "__doc__", "") or ""
    raw = re.findall(r"10\.\d{4,}/[^\s\]\">]+", doc)
    # Clean trailing punctuation and RST artifacts
    cleaned = []
    for d in raw:
        d = d.rstrip(".,;:)")
        d = d.rstrip("`")
        if d.endswith(">`_"):
            d = d[: d.index(">")]
        d = d.rstrip("`_>")
        if d.endswith("/abstract"):
            d = d[: -len("/abstract")]
        cleaned.append(d)
    return list(dict.fromkeys(cleaned))  # deduplicate, preserve order


def _get_metadata_dois(meta: DatasetMetadata) -> dict[str, str | None]:
    """Extract all DOI fields from a DatasetMetadata object."""
    result = {}
    doc = getattr(meta, "documentation", None)
    if doc:
        result["documentation.doi"] = getattr(doc, "doi", None)
        result["documentation.associated_paper_doi"] = getattr(
            doc, "associated_paper_doi", None
        )
    return result


def _is_datacite_doi(doi: str) -> bool:
    """Check if a DOI is registered with DataCite rather than CrossRef."""
    return any(doi.startswith(p) for p in _DATACITE_PREFIXES)


def _resolve_doi_crossref(doi: str) -> dict | None:
    """Resolve a DOI via CrossRef and return title/authors/year, or None."""
    try:
        from habanero import Crossref

        cr = Crossref(mailto="moabb-test@example.com")
        time.sleep(_REQUEST_DELAY)
        r = cr.works(ids=doi)
        msg = r["message"]
        title = msg.get("title", [None])[0]
        authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in msg.get("author", [])
        ]
        year = None
        all_years = set()
        for key in ("published-print", "published-online", "created"):
            if key in msg:
                parts = msg[key].get("date-parts", [[None]])
                if parts and parts[0] and parts[0][0]:
                    all_years.add(parts[0][0])
                    if year is None:
                        year = parts[0][0]
        return {"title": title, "authors": authors, "year": year, "all_years": all_years}
    except Exception:
        return None


def _resolve_doi_datacite(doi: str) -> dict | None:
    """Resolve a DOI via DataCite API and return title/authors/year, or None."""
    try:
        time.sleep(_REQUEST_DELAY)
        url = f"https://api.datacite.org/dois/{doi}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        attrs = data.get("data", {}).get("attributes", {})
        title = (attrs.get("titles") or [{}])[0].get("title")
        authors = [
            c.get("name", "")
            or f"{c.get('givenName', '')} {c.get('familyName', '')}".strip()
            for c in attrs.get("creators", [])
        ]
        year = attrs.get("publicationYear")
        return {"title": title, "authors": authors, "year": year}
    except Exception:
        return None


def _resolve_doi(doi: str) -> dict | None:
    """Resolve a DOI via the appropriate registry (CrossRef or DataCite)."""
    if _is_datacite_doi(doi):
        return _resolve_doi_datacite(doi)
    return _resolve_doi_crossref(doi)


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
# OFFLINE TESTS (no network needed)
# ---------------------------------------------------------------------------


class TestDOIFormat:
    """Validate that DOI strings in metadata have correct format."""

    @pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=lambda c: c.__name__)
    def test_metadata_doi_is_valid_format(self, dataset_class):
        """documentation.doi should be a valid DOI or None."""
        meta = dataset_class.METADATA
        doc = getattr(meta, "documentation", None)
        if doc is None:
            pytest.skip("No documentation section")
        doi = getattr(doc, "doi", None)
        if doi is None:
            pytest.skip("No DOI set")
        is_valid = _is_doi(doi) or any(doi.startswith(p) for p in _NON_CROSSREF_PREFIXES)
        assert is_valid, (
            f"{dataset_class.__name__}: documentation.doi={doi!r} "
            f"does not look like a valid DOI (expected 10.xxxx/...) "
            f"or recognized identifier (hal-/tel-/arXiv:)"
        )

    @pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=lambda c: c.__name__)
    def test_associated_doi_is_valid_format(self, dataset_class):
        """associated_paper_doi should be a valid DOI, HAL ID, or None."""
        meta = dataset_class.METADATA
        doc = getattr(meta, "documentation", None)
        if doc is None:
            pytest.skip("No documentation section")
        assoc = getattr(doc, "associated_paper_doi", None)
        if assoc is None:
            pytest.skip("No associated DOI set")
        is_valid = _is_doi(assoc) or any(
            assoc.startswith(p) for p in _NON_CROSSREF_PREFIXES
        )
        assert is_valid, (
            f"{dataset_class.__name__}: associated_paper_doi={assoc!r} "
            f"is not a valid DOI or recognized ID"
        )


class TestDOIConsistency:
    """Check that DOIs in metadata match docstring references."""

    @pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=lambda c: c.__name__)
    def test_metadata_doi_appears_in_docstring(self, dataset_class):
        """The primary DOI in metadata should also appear in the docstring."""
        meta = dataset_class.METADATA
        doc = getattr(meta, "documentation", None)
        if doc is None:
            pytest.skip("No documentation section")
        doi = getattr(doc, "doi", None)
        if doi is None:
            pytest.skip("No DOI set")
        if not _is_doi(doi):
            pytest.skip(f"DOI {doi!r} is not a standard DOI format")

        docstring = getattr(dataset_class, "__doc__", "") or ""
        # Skip if docstring has no references section at all
        if "doi" not in docstring.lower() and "10." not in docstring:
            pytest.skip("Docstring has no DOI references")

        docstring_dois = _extract_docstring_dois(dataset_class)
        assert doi in docstring_dois, (
            f"{dataset_class.__name__}: metadata DOI {doi!r} "
            f"not found in docstring DOIs: {docstring_dois}"
        )

    @pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=lambda c: c.__name__)
    def test_docstring_dois_are_known_in_metadata(self, dataset_class):
        """All DOIs in docstring should be either the primary DOI,
        associated DOI, or a recognized data repository DOI."""
        meta = dataset_class.METADATA
        doc_meta = getattr(meta, "documentation", None)

        known_dois = set()
        if doc_meta:
            for field in ("doi", "associated_paper_doi", "data_url", "repository"):
                val = getattr(doc_meta, field, None)
                if val and _is_doi(val):
                    known_dois.add(val)
                elif isinstance(val, str) and "doi.org/" in val:
                    # Extract DOI from URL
                    m = re.search(r"10\.\d{4,}/[^\s]+", val)
                    if m:
                        known_dois.add(m.group().rstrip(".,;:)"))

        # External links may contain DOIs too
        ext = getattr(meta, "external_links", None)
        if isinstance(ext, dict):
            for v in ext.values():
                if isinstance(v, str) and _is_doi(v):
                    known_dois.add(v)

        docstring_dois = _extract_docstring_dois(dataset_class)
        if not docstring_dois:
            pytest.skip("No DOIs in docstring")

        # Data-repository DOI prefixes that are fine to appear in docstrings
        _DATA_REPO_PREFIXES = (
            "10.5281/zenodo.",  # Zenodo
            "10.7910/DVN/",  # Harvard Dataverse
            "10.6084/m9.figshare.",  # Figshare
            "10.6094/",  # FreiDok (Freiburg)
            "10.5524/",  # GigaDB
            "10.34973/",  # Radboud Data Repository
            "10.18115/",  # ERPSS
            "10.48550/arXiv.",  # arXiv
        )

        unknown = []
        for d in docstring_dois:
            if d in known_dois:
                continue
            if any(d.startswith(p) for p in _DATA_REPO_PREFIXES):
                continue
            # Check if it's a substring match (different URL forms)
            if any(d in k or k in d for k in known_dois):
                continue
            unknown.append(d)

        if unknown:
            pytest.fail(
                f"{dataset_class.__name__}: docstring contains DOIs not tracked "
                f"in metadata: {unknown}\n"
                f"  Known metadata DOIs: {known_dois}"
            )


# ---------------------------------------------------------------------------
# NETWORK TESTS (require CrossRef API)
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestDOIResolution:
    """Verify that every DOI in metadata resolves via CrossRef."""

    @pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=lambda c: c.__name__)
    def test_primary_doi_resolves(self, dataset_class):
        """documentation.doi should resolve to a real publication."""
        meta = dataset_class.METADATA
        doc = getattr(meta, "documentation", None)
        if doc is None:
            pytest.skip("No documentation section")
        doi = getattr(doc, "doi", None)
        if doi is None:
            pytest.skip("No DOI set")
        if not _is_doi(doi):
            pytest.skip(f"Not a standard DOI: {doi!r}")

        result = _resolve_doi(doi)
        assert (
            result is not None
        ), f"{dataset_class.__name__}: DOI {doi!r} failed to resolve via CrossRef"
        assert result[
            "title"
        ], f"{dataset_class.__name__}: DOI {doi!r} resolved but has no title"

    @pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=lambda c: c.__name__)
    def test_primary_doi_year_matches_metadata(self, dataset_class):
        """Publication year from CrossRef should match metadata."""
        meta = dataset_class.METADATA
        doc = getattr(meta, "documentation", None)
        if doc is None:
            pytest.skip("No documentation section")
        doi = getattr(doc, "doi", None)
        meta_year = getattr(doc, "publication_year", None)
        if doi is None or meta_year is None:
            pytest.skip("No DOI or publication_year")
        if not _is_doi(doi):
            pytest.skip(f"Not a standard DOI: {doi!r}")

        result = _resolve_doi(doi)
        if result is None:
            pytest.skip(f"DOI {doi!r} did not resolve")

        cr_year = result["year"]
        if cr_year is None:
            pytest.skip("Registry returned no year")

        # DataCite records (Zenodo, figshare, etc.) can have updated
        # publication years when new versions are uploaded.  Only flag
        # mismatches for journal/conference DOIs (CrossRef).
        if _is_datacite_doi(doi):
            pytest.skip("Skipping year check for DataCite DOI (versions change dates)")

        # For journal papers, CrossRef may return the print date which
        # can differ from the online-first date.  Accept if metadata year
        # matches any of: published-print, published-online, or created year.
        cr_years = result.get("all_years", set())
        cr_years.add(cr_year)

        assert meta_year in cr_years, (
            f"{dataset_class.__name__}: metadata publication_year={meta_year} "
            f"but CrossRef says year(s)={sorted(cr_years)} for DOI {doi!r}\n"
            f"  CrossRef title: {result['title']}"
        )

    @pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=lambda c: c.__name__)
    def test_primary_doi_authors_overlap_metadata(self, dataset_class):
        """At least one CrossRef author should appear in metadata investigators."""
        meta = dataset_class.METADATA
        doc = getattr(meta, "documentation", None)
        if doc is None:
            pytest.skip("No documentation section")
        doi = getattr(doc, "doi", None)
        investigators = getattr(doc, "investigators", None)
        if doi is None or not investigators:
            pytest.skip("No DOI or investigators")
        if not _is_doi(doi):
            pytest.skip(f"Not a standard DOI: {doi!r}")

        result = _resolve_doi(doi)
        if result is None:
            pytest.skip(f"DOI {doi!r} did not resolve")

        cr_authors = result["authors"]
        if not cr_authors:
            pytest.skip("CrossRef returned no authors")

        # Extract family names from both sources, handling:
        #   CrossRef: "Given Family" format
        #   DataCite: "Family, Given" format
        #   Metadata: "G. Family" or "Given Family" or "Family, G." format
        def _extract_surnames(names):
            surnames = set()
            for name in names:
                name = name.strip()
                if not name:
                    continue
                if ", " in name:
                    # "Last, First" format — take the part before the comma
                    surnames.add(name.split(",")[0].strip().lower())
                else:
                    # "First Last" or "F. Last" — take the last token
                    parts = name.split()
                    if parts:
                        surnames.add(parts[-1].strip(".").lower())
            return surnames

        cr_surnames = _extract_surnames(cr_authors)
        meta_surnames = _extract_surnames(investigators)

        overlap = cr_surnames & meta_surnames
        assert overlap, (
            f"{dataset_class.__name__}: no author overlap between "
            f"CrossRef authors {cr_authors} and "
            f"metadata investigators {investigators} "
            f"for DOI {doi!r}\n"
            f"  CrossRef title: {result['title']}"
        )


@pytest.mark.network
class TestDocstringDOIContent:
    """Verify that DOIs referenced in docstrings resolve to related papers."""

    @pytest.mark.parametrize("dataset_class", _REAL_DATASETS, ids=lambda c: c.__name__)
    def test_docstring_reference_dois_resolve(self, dataset_class):
        """Every DOI in docstring References section should resolve."""
        doc = getattr(dataset_class, "__doc__", "") or ""
        # Only check DOIs from the References section
        ref_section = ""
        in_refs = False
        for line in doc.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("references") or stripped == "----------":
                in_refs = True
                continue
            if in_refs:
                if (
                    stripped
                    and not stripped.startswith("..")
                    and not stripped.startswith("http")
                    and not stripped.startswith("10.")
                    and not stripped[0:1].isdigit()
                    and stripped[0:1].isalpha()
                    and not stripped.startswith("-")
                ):
                    # New section header
                    break
                ref_section += line + "\n"

        if not ref_section:
            pytest.skip("No References section in docstring")

        dois = re.findall(r"10\.\d{4,}/[^\s\]\">]+", ref_section)
        dois = [d.rstrip(".,;:)`_>") for d in dois]
        dois = list(dict.fromkeys(dois))

        if not dois:
            pytest.skip("No DOIs in References section")

        failures = []
        for doi in dois:
            if not _is_doi(doi):
                continue
            result = _resolve_doi(doi)
            if result is None:
                failures.append(doi)

        assert not failures, (
            f"{dataset_class.__name__}: these docstring DOIs failed to resolve "
            f"via CrossRef: {failures}"
        )
