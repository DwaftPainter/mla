"""Exercise the production downloader without any network requests."""
from unittest.mock import MagicMock

import pytest
import requests

from src.data import data_download as downloader


def response(chunks):
    result = MagicMock()
    result.__enter__.return_value = result
    result.iter_content.return_value = chunks
    return result


def test_download_and_skip_existing(tmp_path, monkeypatch):
    dest = tmp_path / "raw" / "nucle.sgml"
    get = MagicMock(return_value=response([b"<DOC>", b"", b"</DOC>"]))
    monkeypatch.setattr(downloader.requests, "get", get)
    downloader.DownloadData.fetch_data("https://example.test/nucle", dest)
    assert dest.read_bytes() == b"<DOC></DOC>"
    downloader.DownloadData.fetch_data("https://example.test/nucle", dest)
    assert get.call_count == 1


def test_partial_download_is_removed_before_retry(tmp_path, monkeypatch):
    def broken_stream():
        yield b"partial"
        raise requests.ConnectionError("disconnected")

    get = MagicMock(side_effect=[response(broken_stream()), response([b"complete"])])
    sleep = MagicMock()
    monkeypatch.setattr(downloader.requests, "get", get)
    monkeypatch.setattr(downloader.time, "sleep", sleep)
    dest = tmp_path / "nucle.sgml"
    downloader.DownloadData.fetch_data("https://example.test/nucle", dest)
    assert dest.read_bytes() == b"complete"
    assert get.call_count == 2
    sleep.assert_called_once_with(downloader.RETRY_BACKOFF_SECONDS)


def test_download_failure_does_not_leave_partial_file(tmp_path, monkeypatch):
    get = MagicMock(side_effect=requests.Timeout("timeout"))
    monkeypatch.setattr(downloader.requests, "get", get)
    monkeypatch.setattr(downloader.time, "sleep", MagicMock())
    dest = tmp_path / "nucle.sgml"
    with pytest.raises(RuntimeError, match="Could not download"):
        downloader.DownloadData.fetch_data("https://example.test/nucle", dest)
    assert get.call_count == downloader.MAX_RETRIES
    assert not dest.exists()
