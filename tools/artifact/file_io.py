"""Keep large offline transfers from retaining whole artifacts in the page cache."""

from __future__ import annotations

from . import os_compat

IO_CHUNK_BYTES = 8 * 1024 * 1024
WRITEBACK_BYTES = 64 * 1024 * 1024
_PAGE_BYTES = os_compat.page_bytes()


def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
    if count is None:
        os_compat.fadvise_dontneed(fd)
    elif count > 0:
        begin = offset // _PAGE_BYTES * _PAGE_BYTES
        end = (offset + count + _PAGE_BYTES - 1) // _PAGE_BYTES * _PAGE_BYTES
        os_compat.fadvise_dontneed(fd, begin, end - begin)


class Writeback:
    """Bound dirty output across all open shards; release clean pages after writeback."""

    def __init__(self) -> None:
        self._bytes = 0
        self._fds: set[int] = set()

    def written(self, fd: int, count: int) -> None:
        self._fds.add(fd)
        self._bytes += count
        if self._bytes >= WRITEBACK_BYTES:
            self.flush()

    def flush(self) -> None:
        for fd in self._fds:
            os_compat.fsync_data(fd)
            discard_cached_pages(fd)
        self._fds.clear()
        self._bytes = 0
