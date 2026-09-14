"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import feedparser
import pytest
import requests

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in new_entries]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


@pytest.mark.parametrize("statuses, expected_waits, failure", [
    ([429, 503, 200], [300, 600], None),
    ([500, 502, 504, 200], [300, 600, 1200], None),
    ([requests.ConnectionError("connection reset"), 200], [300], None),
    ([requests.Timeout("read timed out"), 200], [300], None),
    ([503] * 7, [300, 600, 1200, 1800, 1800, 1800], 503),
    ([400], [], 400),
    ([401], [], 401),
    ([404], [], 404),
])
def test_arxiv_api_retries(config, mock_feedparser, monkeypatch, statuses, expected_waits, failure):
    mock_feedparser.entries = [next(
        entry for entry in mock_feedparser.entries
        if entry.get("arxiv_announce_type", "new") == "new"
    )]
    paper_id = mock_feedparser.entries[0].id.removeprefix("oai:arXiv.org:")
    atom = f"""<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"
        xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
        <opensearch:totalResults>1</opensearch:totalResults>
        <entry><id>https://arxiv.org/abs/{paper_id}</id>
        <title>Recovered paper</title><summary>Abstract</summary>
        <published>2026-09-11T00:00:00Z</published>
        <updated>2026-09-11T00:00:00Z</updated>
        <author><name>Author</name></author>
        <arxiv:primary_category term="cs.AI"/><category term="cs.AI"/>
        <link href="https://arxiv.org/abs/{paper_id}" rel="alternate"/>
        </entry></feed>""".encode()
    responses = iter(statuses)
    requested_urls = []
    waits = []

    def get_response(session, url, **kwargs):
        requested_urls.append(url)
        status = next(responses)
        if isinstance(status, Exception):
            raise status
        response = requests.Response()
        response.status_code = status
        response._content = atom if status == 200 else b"unavailable"
        return response

    monkeypatch.setattr(requests.Session, "get", get_response)
    monkeypatch.setattr(arxiv_retriever, "sleep", waits.append)
    monkeypatch.setattr(arxiv_retriever.arxiv.time, "sleep", lambda _: None)
    retriever = ArxivRetriever(config)
    if failure:
        with pytest.raises(arxiv_retriever.arxiv.HTTPError) as caught:
            retriever._retrieve_raw_papers()
        assert caught.value.status == failure
    else:
        papers = retriever._retrieve_raw_papers()
        assert [paper.get_short_id() for paper in papers] == [paper_id]
    assert len(requested_urls) == len(statuses)
    assert len(set(requested_urls)) == 1
    assert [wait for wait in waits if wait > 10] == expected_waits


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
