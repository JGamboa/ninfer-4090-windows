"""Windows-compatible positional file I/O and page-cache-hint shims.

POSIX (Linux/WSL) has ``os.pread``/``os.pwrite``/``os.posix_fadvise``/``os.fdatasync``
and reports its page size through ``os.sysconf("SC_PAGE_SIZE")``; Windows has none of
these. This module gives the nine POSIX-only call sites in ``file_io.py``, ``reader.py``,
``writer.py`` and ``sources/safetensors.py`` a single portable entry point with the same
observable behavior on both platforms. ``os.link`` already works unchanged on NTFS and is
not wrapped here.
"""

from __future__ import annotations

import mmap
import os
import sys
import threading

WINDOWS = sys.platform == "win32"

# OR into any os.open(..., os.O_RDONLY) call that will be read with os.read/pread: without
# it, Windows opens the descriptor in text mode, which silently translates bytes (CRLF,
# ctrl-Z-as-EOF) and produces short reads of binary artifact/GGUF/safetensors data. A no-op
# on POSIX, where os.O_BINARY does not exist.
BINARY_FLAG = getattr(os, "O_BINARY", 0)


def _fsync(fd: int) -> None:
    os.fsync(fd)


if WINDOWS:
    _locks_guard = threading.Lock()
    _fd_locks: dict[int, threading.Lock] = {}

    def _lock_for(fd: int) -> threading.Lock:
        with _locks_guard:
            lock = _fd_locks.get(fd)
            if lock is None:
                lock = threading.Lock()
                _fd_locks[fd] = lock
            return lock

    def pread(fd: int, count: int, offset: int) -> bytes:
        """Read *count* bytes at *offset* without disturbing the file position."""
        with _lock_for(fd):
            saved = os.lseek(fd, 0, os.SEEK_CUR)
            try:
                os.lseek(fd, offset, os.SEEK_SET)
                return os.read(fd, count)
            finally:
                os.lseek(fd, saved, os.SEEK_SET)

    def pwrite(fd: int, data: bytes, offset: int) -> int:
        """Write *data* at *offset* without disturbing the file position."""
        with _lock_for(fd):
            saved = os.lseek(fd, 0, os.SEEK_CUR)
            try:
                os.lseek(fd, offset, os.SEEK_SET)
                return os.write(fd, data)
            finally:
                os.lseek(fd, saved, os.SEEK_SET)

    def fadvise_dontneed(fd: int, offset: int = 0, count: int | None = None) -> None:
        """No-op: Windows has no page-cache-eviction advisory syscall."""
        return None

    def fsync_data(fd: int) -> None:
        # No fdatasync equivalent; a full fsync is the closest durability guarantee.
        _fsync(fd)

    def page_bytes() -> int:
        return mmap.ALLOCATIONGRANULARITY

else:

    def pread(fd: int, count: int, offset: int) -> bytes:
        return os.pread(fd, count, offset)

    def pwrite(fd: int, data: bytes, offset: int) -> int:
        return os.pwrite(fd, data, offset)

    def fadvise_dontneed(fd: int, offset: int = 0, count: int | None = None) -> None:
        if count:
            os.posix_fadvise(fd, offset, count, os.POSIX_FADV_DONTNEED)
        else:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def fsync_data(fd: int) -> None:
        if hasattr(os, "fdatasync"):
            os.fdatasync(fd)
        else:
            _fsync(fd)

    def page_bytes() -> int:
        return os.sysconf("SC_PAGE_SIZE")


__all__ = [
    "WINDOWS",
    "BINARY_FLAG",
    "pread",
    "pwrite",
    "fadvise_dontneed",
    "fsync_data",
    "page_bytes",
]
