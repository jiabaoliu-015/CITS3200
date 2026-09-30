"""Progress adapters for py123d's streaming downloaders.

Count materialized AV2 files and monitor nuPlan archive bytes on disk.
The scoped hooks preserve upstream selection, concurrency and error handling.
"""

from pathlib import Path
from threading import Event, Thread
from time import monotonic
from unittest.mock import patch

from rich.filesize import decimal
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from adeval.console import console

PART_POLL_INTERVAL = 1.0


def _download_monitored(download_archive, spec, output_dir, progress, task):
    """Observe .part sizes without changing upstream downloads or reading payloads.

    Catalog sizes are estimates, so only a successful downloader return marks
    completion. Stop the observer before publishing the final task state.
    """
    destination = Path(output_dir) / spec.filename
    partial = destination.with_suffix(destination.suffix + ".part")
    estimated_bytes = max(0, spec.approx_size_gb * 1_000_000_000)
    stopped = Event()
    observed_bytes = 0
    last_change = monotonic()

    def sample(state="Downloading"):
        nonlocal observed_bytes, last_change
        readable = True
        try:
            # Prefer the completed archive after the upstream atomic rename.
            try:
                size = destination.stat().st_size
            except FileNotFoundError:
                try:
                    size = partial.stat().st_size
                except FileNotFoundError:
                    # The .part file can be renamed between the two stat calls.
                    try:
                        size = destination.stat().st_size
                    except FileNotFoundError:
                        size = observed_bytes
            if size != observed_bytes:
                observed_bytes = size
                last_change = monotonic()
        except OSError:
            # A display-only filesystem error must not abort a download.
            readable = False

        detail = decimal(observed_bytes)
        total = estimated_bytes or None
        completed = (
            min(observed_bytes, estimated_bytes * 0.99) if total else observed_bytes
        )
        if state == "Complete":
            total = max(observed_bytes, 1)
            completed = total
            detail += " | Complete"
        else:
            if estimated_bytes:
                percent = min(99.0, observed_bytes / estimated_bytes * 100)
                detail += f" / ~{decimal(estimated_bytes)} (~{percent:.1f}%)"
            detail += f" | {state}"
            if not readable:
                detail += " | size unavailable"
            elif state == "Downloading":
                idle_seconds = int(monotonic() - last_change)
                if idle_seconds >= 5:
                    detail += f" | no size change for {idle_seconds}s"
        progress.update(task, total=total, completed=completed, detail=detail)

    def watch():
        while not stopped.wait(PART_POLL_INTERVAL):
            sample()

    sample()
    observer = Thread(target=watch, name="nuplan-part-progress", daemon=True)
    observer.start()
    state = "Failed / interrupted"
    try:
        result = download_archive(spec=spec, output_dir=output_dir)
        state = "Complete"
        return result
    finally:
        stopped.set()
        observer.join()
        sample(state)


def download_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed:.0f}/{task.total:.0f} files"),
        console=console,
    )


# Import dataset dependencies only when Hydra requests the relevant adapter.
def __getattr__(name):
    if name == "Av2ProgressDownloader":
        from py123d.parser.av2 import av2_download as upstream

        class Av2ProgressDownloader(upstream.Av2Downloader):
            def download(self):
                console.print("AV2: listing download files...")
                with download_progress() as progress:
                    task = progress.add_task("AV2 download", total=0, visible=False)
                    list_keys = upstream.list_log_object_keys
                    download_one = upstream._download_one_object

                    def listed(*args, **kwargs):
                        keys = list_keys(*args, **kwargs)
                        progress.update(task, total=progress.tasks[0].total + len(keys))
                        return keys

                    def downloaded(*args, **kwargs):
                        progress.update(task, visible=True)
                        result = download_one(*args, **kwargs)
                        progress.advance(task)
                        return result

                    with (
                        patch.object(upstream, "list_log_object_keys", listed),
                        patch.object(upstream, "_download_one_object", downloaded),
                    ):
                        super().download()

        return Av2ProgressDownloader

    if name == "NuplanProgressDownloader":
        from py123d.parser.nuplan import nuplan_download as upstream

        class NuplanProgressDownloader(upstream.NuplanDownloader):
            def _fetch_and_extract(self, archives, zip_dir, extract_dir):
                with Progress(
                    SpinnerColumn(),
                    TextColumn("{task.description}", markup=False),
                    BarColumn(),
                    TextColumn("{task.fields[detail]}", markup=False),
                    console=console,
                ) as progress:
                    archive_tasks = {
                        spec.filename: progress.add_task(
                            spec.filename, total=None, detail="Queued"
                        )
                        for spec in archives
                    }
                    extract = progress.add_task(
                        "nuPlan extract (archives)",
                        total=len(archives),
                        detail=f"0/{len(archives)} archives",
                    )
                    console.print(
                        "nuPlan: checking archive/.part sizes every second. "
                        "Totals and percentages are estimates; completion is confirmed "
                        "by the downloader."
                    )
                    download_archive = upstream._download_archive
                    extract_archive = upstream._extract_nuplan_archive

                    def downloaded(spec, output_dir):
                        return _download_monitored(
                            download_archive,
                            spec,
                            output_dir,
                            progress,
                            archive_tasks[spec.filename],
                        )

                    def extracted(*args, **kwargs):
                        result = extract_archive(*args, **kwargs)
                        progress.advance(extract)
                        progress.update(
                            extract,
                            detail=f"{progress.tasks[extract].completed:.0f}/{len(archives)} archives",
                        )
                        return result

                    with (
                        patch.object(upstream, "_download_archive", downloaded),
                        patch.object(upstream, "_extract_nuplan_archive", extracted),
                    ):
                        super()._fetch_and_extract(archives, zip_dir, extract_dir)

        return NuplanProgressDownloader

    raise AttributeError(name)
