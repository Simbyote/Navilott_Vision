"""
dataset_loader.py

Dataset Loading Utility

Purpose:
    Extracts a zipped dataset archive into a working directory and returns
    the list of track subdirectories in a format compatible with the
    SAMPLE_DIRS convention used across all pipeline stage test blocks.

    Extraction is to disk rather than in-memory because test blocks write
    many debug images per frame. Keeping the extracted layout on disk avoids
    holding the entire dataset in memory across hundreds of frames and lets
    the existing test code operate unchanged.

    The archive is expected to contain one or more track directories at its
    top level, each holding sequential frame images:

        dataset.zip
        ├── trackT3/
        │   ├── frame_0001.jpg
        │   └── ...
        ├── trackT4/
        │   └── ...
        └── trackT5/
            └── ...

    After extraction the layout on disk mirrors this structure under dest_dir,
    and the returned list of paths is a drop-in replacement for SAMPLE_DIRS
"""

import os
import zipfile
import urllib.request

# ==============================================================================
# Fetches Remote Datasets
# ==============================================================================
def fetch_dataset(
    url: str,
    zip_path: str,
    dest_dir: str,
    track_prefix: str = "track",
) -> list[str]:
    """
    Downloads the dataset zip from url if zip_path doesn't exist locally,
    then delegates to extract_dataset().
    """
    if not os.path.isfile(zip_path):
        print(f"[dataset_loader] Downloading dataset from {url} ...")
        os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        print(f"[dataset_loader] Saved to {zip_path}")
    return extract_dataset(zip_path=zip_path, dest_dir=dest_dir, track_prefix=track_prefix)

# ==============================================================================
# Extracts Local Datasets
# ==============================================================================
def extract_dataset(
    zip_path: str,
    dest_dir: str,
    track_prefix: str = "track",
    force: bool = False,
) -> list[str]:
    """
    Purpose:
        Extracts a dataset zip archive and returns a sorted list of track
        subdirectory paths suitable for use as SAMPLE_DIRS

    Inputs:
        zip_path: str
            Path to the zip archive to extract

        dest_dir: str
            Root directory into which the archive is extracted
            Created if it does not exist
            Each top-level zip entry becomes a subdirectory here

        track_prefix: str (default "track")
            Only top-level directories whose name starts with this prefix
            are included in the returned list
            Set to "" to return all top-level directories

        force: bool (default False)
            When False, skips extraction if dest_dir already contains
            entries matching track_prefix (idempotent re-runs)
            When True, always re-extracts, overwriting existing files

    Outputs:
        Returns : list[str]
            Sorted list of absolute paths to extracted track directories
            Compatible with the SAMPLE_DIRS convention used in test blocks

    Raises:
        FileNotFoundError: if zip_path does not exist
        ValueError: if the archive contains no matching track dirs
        zipfile.BadZipFile: if zip_path is not a valid zip archive
    """
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"extract_dataset: archive not found: {zip_path}")

    os.makedirs(dest_dir, exist_ok=True)
    dest_dir = os.path.abspath(dest_dir)

    # Idempotency check:skip extraction if matching dirs already exist
    if not force:
        existing = _find_track_dirs(dest_dir, track_prefix)
        if existing:
            print(f"[dataset_loader] {len(existing)} track dir(s) already present in "
                  f"{dest_dir}, skipping extraction (pass force=True to override)")
            return existing

    print(f"[dataset_loader] Extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        _safe_extract(zf, dest_dir)

    track_dirs = _find_track_dirs(dest_dir, track_prefix)
    if not track_dirs:
        raise ValueError(
            f"extract_dataset: no directories matching prefix '{track_prefix}' "
            f"found in {dest_dir} after extraction"
        )

    print(f"[dataset_loader] Ready: {track_dirs}")
    return track_dirs

# =============================================================================
# Helper Functions
# =============================================================================
def _find_track_dirs(root: str, prefix: str) -> list[str]:
    """
    Purpose:
        Returns a sorted list of absolute paths to immediate subdirectories of
        root whose names start with prefix
    """
    entries = []
    try:
        for name in os.listdir(root):
            full = os.path.join(root, name)
            if os.path.isdir(full) and name.startswith(prefix):
                entries.append(full)
    except OSError:
        pass
    return sorted(entries)

def _safe_extract(zf: zipfile.ZipFile, dest: str) -> None:
    """
    Extracts all members of zf into dest while rejecting path traversal
    entries (names containing '..' or starting with '/').

    zipfile.ZipFile.extractall() does not guard against crafted archives
    that use '../' to write outside the destination tree.
    """
    dest = os.path.realpath(dest)
    for member in zf.infolist():
        # Normalize the member path
        member_path = os.path.realpath(os.path.join(dest, member.filename))

        # Reject any entry that would land outside dest
        if not member_path.startswith(dest + os.sep) and member_path != dest:
            print(f"[dataset_loader][WARN] Skipping unsafe path: {member.filename}")
            continue

        zf.extract(member, dest)