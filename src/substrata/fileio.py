"""Safe file writing: atomic replacement and per-path locks.

Every writer that rewrites a user file (annotation CSVs, camera XML/JSON, project
YAML, crops) goes through :func:`atomic_write`: the new content is written to a
uniquely named temporary file next to the destination and moved into place with
``os.replace``, so an interrupted write never leaves a truncated file. The
temporary file gets the destination's permission bits (or, for a new file,
``0o666`` less the process umask, as ``open()`` would give it) - not the ``0600``
of ``tempfile.mkstemp`` - so shared project folders stay readable by the group.

:func:`path_lock` serialises read-modify-write edits of one file within a process
(e.g. the threads of a viewer server editing the same annotation CSV).

:func:`imread` reads photos with OpenCV in their stored pixel grid.

Standard library only at import time (OpenCV is imported on first use), so any
module can import it.
"""

from __future__ import annotations

import contextlib
import os
import stat
import tempfile
import threading
from typing import IO, Any, Dict, Iterator, Optional, Union

__all__ = ["atomic_write", "atomic_write_text", "atomic_write_bytes", "path_lock", "imread"]


def _umask() -> int:
    """The process umask, read without changing it where the OS allows."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError, IndexError):
        pass
    mask = os.umask(0o022)  # (not thread-safe, but only reached off Linux)
    os.umask(mask)
    return mask


def _target_mode(path: str) -> int:
    """Permission bits for ``path``: the existing file's, else 0o666 & ~umask."""
    try:
        return stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        return 0o666 & ~_umask()


@contextlib.contextmanager
def atomic_write(
    path: str, mode: str = "w", **open_kwargs
) -> Iterator[Union[IO[str], IO[bytes]]]:
    """Open a temporary file that replaces ``path`` when the block ends.

    On an exception inside the block the temporary file is removed and ``path``
    is left untouched.

    Args:
        path: Destination file.
        mode: ``"w"`` (text) or ``"wb"`` (binary).
        **open_kwargs: Passed to ``open`` (e.g. ``newline=""``, ``encoding``).

    Yields:
        The open temporary file.
    """
    if mode not in ("w", "wb"):
        raise ValueError(f"atomic_write: mode must be 'w' or 'wb', not {mode!r}")
    dest = os.path.abspath(path)
    folder = os.path.dirname(dest) or "."
    fd, tmp = tempfile.mkstemp(
        prefix="." + os.path.basename(dest) + ".", suffix=".tmp", dir=folder
    )
    try:
        with os.fdopen(fd, mode, **open_kwargs) as fh:
            yield fh
        os.chmod(tmp, _target_mode(dest))
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def atomic_write_text(
    path: str, text: str, encoding: str = "utf-8", newline: Optional[str] = None
) -> str:
    """Write ``text`` to ``path`` atomically (see :func:`atomic_write`).

    Returns:
        The absolute path written.
    """
    with atomic_write(path, "w", encoding=encoding, newline=newline) as fh:
        fh.write(text)
    return os.path.abspath(path)


def atomic_write_bytes(path: str, data: bytes) -> str:
    """Write ``data`` to ``path`` atomically (see :func:`atomic_write`).

    Returns:
        The absolute path written.
    """
    with atomic_write(path, "wb") as fh:
        fh.write(data)
    return os.path.abspath(path)


_LOCKS: Dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def path_lock(path: str) -> threading.RLock:
    """A re-entrant lock for one file (by real path), shared within the process.

    Use it around a read-modify-write of the file, so two threads editing the
    same file cannot lose each other's changes::

        with path_lock(csv_path):
            rows = read(csv_path)
            ...
            write(csv_path, rows)

    It does not lock against other processes.
    """
    key = os.path.realpath(path)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.RLock()
        return lock


def imread(path: str, flags: Optional[int] = None) -> Any:
    """Read an image with OpenCV in its stored pixel grid (EXIF orientation ignored).

    ``cv2.imread`` turns a full-size read the way the EXIF Orientation tag says,
    but not a reduced one (``IMREAD_REDUCED_*``), and PIL, Metashape's camera
    calibration and every projection in substrata use the pixels as stored. So
    all OpenCV reads go through here: pixel ``(x, y)`` of the returned array is
    pixel ``(x, y)`` of the sensor whatever the flags and the camera's
    orientation tag.

    Args:
        path: Image file.
        flags: ``cv2.IMREAD_*`` flags (default ``IMREAD_COLOR``: BGR).

    Returns:
        The image array, or None if it cannot be read (as ``cv2.imread``).
    """
    import cv2

    flags = cv2.IMREAD_COLOR if flags is None else int(flags)
    return cv2.imread(path, flags | cv2.IMREAD_IGNORE_ORIENTATION)
