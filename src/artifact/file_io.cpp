#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <limits>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <string>
#include <system_error>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

#ifdef _WIN32

// Same exception type as the POSIX branch so callers see one error contract on every platform.
[[noreturn]] void fail(const std::filesystem::path& path, const char* operation,
                       DWORD error = ::GetLastError()) {
    throw ArtifactError(path.string() + ": " + operation + ": " +
                        std::system_category().message(static_cast<int>(error)) + " (" +
                        std::to_string(error) + ")");
}

// A per-request event keeps GetOverlappedResult from waiting on the shared file handle, which
// is signalled by any completion on that handle and is therefore unreliable under concurrency.
class OverlappedEvent {
public:
    OverlappedEvent() : event_(::CreateEventW(nullptr, TRUE, FALSE, nullptr)) {}
    ~OverlappedEvent() {
        if (event_ != nullptr) { ::CloseHandle(event_); }
    }
    OverlappedEvent(const OverlappedEvent&)            = delete;
    OverlappedEvent& operator=(const OverlappedEvent&) = delete;

    [[nodiscard]] HANDLE get() const noexcept { return event_; }

private:
    HANDLE event_;
};

#else

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

#endif

} // namespace

#ifdef _WIN32

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    HANDLE opened = ::CreateFileW(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (opened == INVALID_HANDLE_VALUE) { fail(path_, "CreateFileW"); }
    handle_ = opened;

    // Mirrors the S_ISREG check of the POSIX branch: reject pipes, character devices and the like.
    if (::GetFileType(opened) != FILE_TYPE_DISK) {
        ::CloseHandle(opened);
        handle_ = nullptr;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }

    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(opened, &size)) {
        const DWORD error = ::GetLastError();
        ::CloseHandle(opened);
        handle_ = nullptr;
        fail(path_, "GetFileSizeEx", error);
    }
    bytes_ = static_cast<std::uint64_t>(size.QuadPart);
}

InputFile::~InputFile() {
    if (direct_handle_ != nullptr) { ::CloseHandle(static_cast<HANDLE>(direct_handle_)); }
    if (handle_ != nullptr) { ::CloseHandle(static_cast<HANDLE>(handle_)); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    const HANDLE handle = static_cast<HANDLE>(handle_);
    while (!destination.empty()) {
        const auto count =
            static_cast<DWORD>(std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024));
        // handle_ was opened without FILE_FLAG_OVERLAPPED, so this ReadFile blocks until the
        // read completes; the OVERLAPPED offset still selects the start position (a pread
        // equivalent), so concurrent callers do not disturb each other through the file pointer.
        OVERLAPPED overlapped{};
        overlapped.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
        overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32U);

        DWORD read = 0;
        if (!::ReadFile(handle, destination.data(), count, &read, &overlapped)) {
            const DWORD error = ::GetLastError();
            if (error != ERROR_HANDLE_EOF) { fail(path_, "ReadFile", error); }
            read = 0;
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += read;
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment) {
        throw ArtifactError(path_.string() + ": unaligned direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_handle_ == nullptr) {
        // FILE_FLAG_NO_BUFFERING is the O_DIRECT equivalent: offsets, lengths and buffer
        // addresses must be sector aligned, which kPayloadAlignment (4096) guarantees.
        HANDLE opened = ::CreateFileW(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                      OPEN_EXISTING,
                                      FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
                                          FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
                                      nullptr);
        if (opened == INVALID_HANDLE_VALUE) { fail(path_, "CreateFileW direct"); }
        direct_handle_ = opened;
    }
    const HANDLE handle = static_cast<HANDLE>(direct_handle_);

    // The direct handle is overlapped, so each ReadFile either completes immediately or must be
    // waited on through GetOverlappedResult. Requests are chunked at 1 GiB so every byte count
    // fits the DWORD nNumberOfBytesToRead parameter. A short final block (or EOF) ends the loop.
    std::size_t total = 0;
    while (total < destination.size()) {
        constexpr std::size_t max_read = std::size_t{1} << 30;
        const auto amount = static_cast<DWORD>(std::min(max_read, destination.size() - total));
        const std::uint64_t block_offset = offset + total;

        OverlappedEvent event;
        if (event.get() == nullptr) { fail(path_, "CreateEventW"); }
        OVERLAPPED overlapped{};
        overlapped.Offset     = static_cast<DWORD>(block_offset & 0xffffffffULL);
        overlapped.OffsetHigh = static_cast<DWORD>(block_offset >> 32U);
        overlapped.hEvent     = event.get();

        DWORD bytes = 0;
        if (!::ReadFile(handle, destination.data() + total, amount, &bytes, &overlapped)) {
            DWORD error = ::GetLastError();
            if (error == ERROR_IO_PENDING) {
                error = ::GetOverlappedResult(handle, &overlapped, &bytes, TRUE) ? ERROR_SUCCESS
                                                                                 : ::GetLastError();
            }
            if (error == ERROR_HANDLE_EOF) { break; }
            if (error != ERROR_SUCCESS) { fail(path_, "direct ReadFile", error); }
        }
        total += bytes;
        if (bytes != amount) { break; }
    }
    return total;
}

#else

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }

    struct stat status {};

    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
}

InputFile::~InputFile() {
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read  = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    ssize_t read;
    do {
        read = ::pread(direct_fd_, destination.data(), destination.size(), file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
}

#endif

} // namespace ninfer::artifact
