#!/usr/bin/env python3
"""
Test Script: Component 1.2 - Resolver

Validates that provider resolution is serial and Semantic Scholar calls
are proactively throttled before the request is sent.

Usage:
    python scripts/test_resolver.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.headers = {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self):
        self.timestamps = []

    def get(self, url, timeout, headers=None):
        self.timestamps.append(time.monotonic())
        return FakeResponse(
            200,
            {
                "paperId": "paper-1",
                "title": "Test Paper",
                "abstract": "",
                "authors": [],
                "year": 2024,
                "venue": "",
                "externalIds": {},
                "openAccessPdf": {},
            },
        )


def test_resolve_by_arxiv_is_serial():
    from agents import resolver

    calls = []
    originals = (
        resolver.query_arxiv,
        resolver.query_semantic_scholar,
        resolver.query_openalex_by_arxiv,
    )

    try:
        def fake_arxiv(arxiv_id):
            calls.append(f"arxiv:{arxiv_id}")
            return None

        def fake_s2(identifier):
            calls.append(f"s2:{identifier}")
            return None

        def fake_openalex(arxiv_id):
            calls.append(f"openalex:{arxiv_id}")
            return resolver.ResolvedPaper(
                source="openalex",
                confidence=0.95,
                arxiv_id=arxiv_id,
                title="Recovered From OpenAlex",
            )

        resolver.query_arxiv = fake_arxiv
        resolver.query_semantic_scholar = fake_s2
        resolver.query_openalex_by_arxiv = fake_openalex

        result = resolver.resolve_by_arxiv("1706.03762")

        assert result is not None, "expected a result"
        assert result.source == "openalex", "expected OpenAlex fallback"
        assert calls == [
            "arxiv:1706.03762",
            "openalex:1706.03762",
        ], f"unexpected provider order: {calls}"
        print("PASS: arXiv providers run serially in order")
        return True
    finally:
        (
            resolver.query_arxiv,
            resolver.query_semantic_scholar,
            resolver.query_openalex_by_arxiv,
        ) = originals


def test_semantic_scholar_throttle_is_proactive():
    from agents import resolver

    fake_session = FakeSession()
    original_get_session = resolver.get_session
    original_interval = resolver.SEMANTIC_SCHOLAR_MIN_INTERVAL
    original_last_call = resolver._s2_last_call_time

    try:
        resolver.get_session = lambda: fake_session
        resolver.SEMANTIC_SCHOLAR_MIN_INTERVAL = 0.05
        resolver._s2_last_call_time = 0.0

        resolver.query_semantic_scholar("ARXIV:1706.03762")
        resolver.query_semantic_scholar("ARXIV:1706.03762")

        assert len(fake_session.timestamps) == 2, "expected two S2 requests"
        delta = fake_session.timestamps[1] - fake_session.timestamps[0]
        assert delta >= 0.045, f"request gap too small: {delta:.4f}s"
        print(f"PASS: Semantic Scholar calls are spaced by {delta:.4f}s")
        return True
    finally:
        resolver.get_session = original_get_session
        resolver.SEMANTIC_SCHOLAR_MIN_INTERVAL = original_interval
        resolver._s2_last_call_time = original_last_call


def test_openalex_result_is_enriched():
    from agents import resolver

    payload = {
        "title": "Playing Atari with Deep Reinforcement Learning",
        "doi": "https://doi.org/10.48550/arXiv.1312.5602",
        "publication_year": 2013,
        "ids": {
            "openalex": "https://openalex.org/W123",
            "doi": "https://doi.org/10.48550/arXiv.1312.5602",
            "semantic_scholar": "https://api.semanticscholar.org/CorpusID:123456",
        },
        "authorships": [
            {
                "author": {"display_name": "Volodymyr Mnih"},
                "institutions": [{"display_name": "DeepMind"}],
            }
        ],
        "primary_location": {
            "landing_page_url": "https://arxiv.org/abs/1312.5602",
            "pdf_url": "https://arxiv.org/pdf/1312.5602.pdf",
            "source": {"display_name": "arXiv"},
        },
    }

    paper = resolver.openalex_to_paper(payload, confidence=0.91)

    assert paper.arxiv_id == "1312.5602", paper.arxiv_id
    assert paper.doi == "10.48550/arXiv.1312.5602", paper.doi
    assert paper.year == 2013, paper.year
    assert paper.venue == "arXiv", paper.venue
    assert paper.open_access_pdf == "https://arxiv.org/pdf/1312.5602.pdf", paper.open_access_pdf
    assert paper.s2_paper_id == "CorpusID:123456", paper.s2_paper_id
    assert paper.authors and paper.authors[0]["last"] == "Mnih", paper.authors
    print("PASS: OpenAlex results are enriched with IDs and metadata")
    return True


def main() -> int:
    tests = [
        test_resolve_by_arxiv_is_serial,
        test_semantic_scholar_throttle_is_proactive,
        test_openalex_result_is_enriched,
    ]

    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:
            failures += 1
            print(f"FAIL: {test.__name__}: {exc}")

    if failures:
        print(f"{failures} test(s) failed")
        return 1

    print("All resolver tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
