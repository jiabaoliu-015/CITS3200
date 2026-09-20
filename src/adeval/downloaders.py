"""Progress adapters for py123d's streaming downloaders.

Count successfully materialized files (including cached files), not elapsed time.
The scoped hooks preserve upstream selection, concurrency and error handling.
"""

from unittest.mock import patch

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from adeval.console import console


def download_progress():
    return Progress(
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
                with download_progress() as progress:
                    download = progress.add_task("nuPlan download (archives)", total=len(archives))
                    extract = progress.add_task("nuPlan extract (archives)", total=len(archives))
                    download_archive = upstream._download_archive
                    extract_archive = upstream._extract_nuplan_archive

                    def downloaded(*args, **kwargs):
                        result = download_archive(*args, **kwargs)
                        progress.advance(download)
                        return result

                    def extracted(*args, **kwargs):
                        result = extract_archive(*args, **kwargs)
                        progress.advance(extract)
                        return result

                    with (
                        patch.object(upstream, "_download_archive", downloaded),
                        patch.object(upstream, "_extract_nuplan_archive", extracted),
                    ):
                        super()._fetch_and_extract(archives, zip_dir, extract_dir)

        return NuplanProgressDownloader

    raise AttributeError(name)
