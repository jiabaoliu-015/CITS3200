import io
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.progress import Progress

from adeval import downloaders


def setup_progress():
    progress = Progress(console=Console(file=io.StringIO()))
    task = progress.add_task("archive", total=None, detail="Queued")
    return progress, task


def spec_for(filename="mini.zip", estimate=0.001):
    return SimpleNamespace(
        filename=filename,
        approx_size_gb=estimate,
        target_subdir=Path("data"),
        skip_levels=0,
    )


def test_observes_growth_before_download_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(downloaders, "PART_POLL_INTERVAL", 0.01)
    progress, task = setup_progress()
    observed = threading.Event()
    update = progress.update

    def record(task_id, **kwargs):
        update(task_id, **kwargs)
        if kwargs.get("completed") == 12345:
            observed.set()

    monkeypatch.setattr(progress, "update", record)

    def fetch(spec, output_dir):
        partial = output_dir / (spec.filename + ".part")
        partial.write_bytes(b"x" * 12345)
        assert observed.wait(2), "No intermediate .part size was displayed"
        assert "Downloading" in progress.tasks[task].fields["detail"]
        partial.write_bytes(b"x" * 20000)
        destination = output_dir / spec.filename
        partial.rename(destination)
        return destination

    result = downloaders._download_monitored(
        fetch, spec_for(), tmp_path, progress, task
    )
    assert result.stat().st_size == 20000
    assert progress.tasks[task].completed == 20000
    assert progress.tasks[task].fields["detail"] == "20.0 kB | Complete"
    assert not any(t.name == "nuplan-part-progress" for t in threading.enumerate())


def test_cached_archive_uses_actual_size(tmp_path):
    progress, task = setup_progress()
    archive = tmp_path / "mini.zip"
    archive.write_bytes(b"cached")
    result = downloaders._download_monitored(
        lambda **kwargs: archive, spec_for(), tmp_path, progress, task
    )
    assert result == archive
    assert progress.tasks[task].total == 6
    assert progress.tasks[task].finished


@pytest.mark.parametrize("error", [RuntimeError("network failed"), KeyboardInterrupt()])
def test_failure_keeps_partial_bytes_and_stops_observer(tmp_path, error):
    progress, task = setup_progress()

    def fetch(spec, output_dir):
        (output_dir / (spec.filename + ".part")).write_bytes(b"partial")
        raise error

    with pytest.raises(type(error)) as caught:
        downloaders._download_monitored(fetch, spec_for(), tmp_path, progress, task)
    assert caught.value is error
    assert "7 bytes" in progress.tasks[task].fields["detail"]
    assert "Failed / interrupted" in progress.tasks[task].fields["detail"]
    assert not progress.tasks[task].finished
    assert not any(t.name == "nuplan-part-progress" for t in threading.enumerate())


def test_exceeding_estimate_does_not_claim_completion(tmp_path):
    progress, task = setup_progress()
    (tmp_path / "mini.zip.part").write_bytes(b"x" * 2000)

    def fetch(**kwargs):
        detail = progress.tasks[task].fields["detail"]
        assert "2.0 kB / ~1.0 kB (~99.0%)" in detail
        assert not progress.tasks[task].finished
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        downloaders._download_monitored(
            fetch, spec_for(estimate=0.000001), tmp_path, progress, task
        )


def test_unknown_total_and_no_growth_message(tmp_path, monkeypatch):
    progress, task = setup_progress()
    times = iter([100.0, 106.0])
    monkeypatch.setattr(downloaders, "monotonic", lambda: next(times))

    def fetch(**kwargs):
        assert progress.tasks[task].total is None
        assert "no size change for 6s" in progress.tasks[task].fields["detail"]
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        downloaders._download_monitored(
            fetch, spec_for(estimate=0), tmp_path, progress, task
        )


def test_stat_failure_does_not_replace_download_error(tmp_path):
    progress, task = setup_progress()
    error = RuntimeError("original network failure")

    def fetch(**kwargs):
        assert "size unavailable" in progress.tasks[task].fields["detail"]
        raise error

    with (
        patch.object(Path, "stat", side_effect=PermissionError("unreadable")),
        pytest.raises(RuntimeError) as caught,
    ):
        downloaders._download_monitored(fetch, spec_for(), tmp_path, progress, task)
    assert caught.value is error


def test_upstream_parallel_download_and_extraction_hooks(tmp_path, monkeypatch):
    from py123d.parser.nuplan import nuplan_download as upstream

    output = io.StringIO()
    monkeypatch.setattr(downloaders, "console", Console(file=output, width=160))
    specs = [spec_for("mini.zip"), spec_for("maps.zip")]
    zip_dir = tmp_path / "zip"
    zip_dir.mkdir()
    extracted = []

    def fetch(spec, output_dir):
        destination = output_dir / spec.filename
        destination.write_bytes(b"archive")
        return destination

    def extract(**kwargs):
        extracted.append(kwargs["archive_path"].name)

    monkeypatch.setattr(upstream, "_download_archive", fetch)
    monkeypatch.setattr(upstream, "_extract_nuplan_archive", extract)
    adapter = downloaders.NuplanProgressDownloader(max_workers=2)
    adapter._fetch_and_extract(specs, zip_dir, tmp_path / "extracted")
    assert extracted == ["mini.zip", "maps.zip"]
    assert "2/2 archives" in output.getvalue()
    assert output.getvalue().count("Complete") == 2
    assert upstream._download_archive is fetch
    assert upstream._extract_nuplan_archive is extract
    assert not list(zip_dir.iterdir())
