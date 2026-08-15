//===- driver_util.h - tuning driver internal helpers (no public API) -------===//
//
// kernscope tools/tune todo 28. Small host-side utilities shared by the
// driver pipeline and the per-family adapter TUs: deterministic hashing,
// subprocess execution, file IO, timestamping. Internal to kernscope_tune_driver
// (never installed).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_TUNE_DRIVER_UTIL_H
#define KERNSCOPE_TUNE_DRIVER_UTIL_H

#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace kernscope::tune::driver_detail {

// FNV-1a 64-bit (deterministic, no external hash lib — libs/db has none and
// this must stay dependency-free).
inline std::uint64_t fnv1a64(std::string_view s) noexcept {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  for (const unsigned char c : s) {
    h ^= c;
    h *= 0x100000001b3ULL;
  }
  return h;
}

inline std::string to_hex(std::uint64_t v) {
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(v));
  return buf;
}

inline std::string iso_timestamp_utc() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[24];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour,
                tm.tm_min, tm.tm_sec);
  return buf;
}

inline std::string read_file(const std::string &path) {
  std::ifstream fh(path, std::ios::binary);
  std::ostringstream ss;
  ss << fh.rdbuf();
  return ss.str();
}

inline bool write_file(const std::string &path, const std::string &content) {
  std::ofstream fh(path, std::ios::binary | std::ios::trunc);
  if (!fh)
    return false;
  fh << content;
  return static_cast<bool>(fh);
}

// Run `cmd` with cwd `dir`, redirecting stdout+stderr into `dir/cmd.log`.
// Returns the child exit code, or -1 on spawn/non-normal exit.
inline int run_command(const std::string &dir, const std::string &cmd) {
  const std::string full = "cd " + dir + " && " + cmd + " > cmd.log 2>&1";
  const int status = std::system(full.c_str());
  if (status == -1)
    return -1;
  if (!WIFEXITED(status))
    return -1;
  return WEXITSTATUS(status);
}

// Run `cmd` with cwd `dir` and CAPTURE stdout (a pipe read back after the
// child exits); stderr is merged into the returned stdout. Returns the
// child exit code, or -1 on spawn/non-normal exit. Used by the todo-54
// verifier gate, which needs the kernscope-verify --json verdict document
// (the JSON goes to stdout, diagnostics to stderr).
inline int run_command_capture(const std::string &dir, const std::string &cmd,
                               std::string *stdout_out) {
  const std::string full = "cd " + dir + " && " + cmd + " 2>&1";
  FILE *pipe = ::popen(full.c_str(), "r");
  if (!pipe)
    return -1;
  std::string out;
  char buf[4096];
  std::size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), pipe)) > 0)
    out.append(buf, n);
  const int status = ::pclose(pipe);
  if (status == -1)
    return -1;
  if (!WIFEXITED(status))
    return -1;
  if (stdout_out)
    *stdout_out = std::move(out);
  return WEXITSTATUS(status);
}

// A default scratch dir under the system temp dir: deterministic prefix +
// pid so parallel tests never collide.
inline std::string default_workdir(const std::string &family,
                                   const std::string &tag) {
  std::error_code ec;
  const auto tmp = std::filesystem::temp_directory_path(ec);
  const std::string base = ec ? "/tmp" : tmp.string();
  return base + "/kernscope-tune-" + family + "-" + tag + "-" +
         std::to_string(::getpid());
}

inline bool ensure_dir(const std::string &path) {
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  return !ec;
}

// Last `n` lines of a command log (for the evidence `reason` on failures).
inline std::string log_tail(const std::string &log, std::size_t n = 12) {
  std::vector<std::string> lines;
  std::size_t pos = 0;
  while (pos < log.size()) {
    const std::size_t nl = log.find('\n', pos);
    lines.push_back(log.substr(pos, nl == std::string::npos ? std::string::npos
                                                            : nl - pos));
    if (nl == std::string::npos)
      break;
    pos = nl + 1;
  }
  if (lines.size() <= n)
    return log;
  std::string out = "... (truncated)\n";
  for (std::size_t i = lines.size() - n; i < lines.size(); ++i)
    out += lines[i] + "\n";
  return out;
}

} // namespace kernscope::tune::driver_detail

#endif // KERNSCOPE_TUNE_DRIVER_UTIL_H
