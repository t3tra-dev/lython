#include "Common/Instrumentation.h"

#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <sys/resource.h>

namespace py {
namespace {

bool perfEnabled() {
  static const bool enabled = [] {
    auto value = llvm::sys::Process::GetEnv("LYTHON_PERF");
    if (!value)
      return false;
    llvm::StringRef text(*value);
    return text == "1" || text.equals_insensitive("true") ||
           text.equals_insensitive("yes") || text.equals_insensitive("on");
  }();
  return enabled;
}

std::uint64_t timevalMicros(const timeval &value) {
  return static_cast<std::uint64_t>(value.tv_sec) * 1000000ULL +
         static_cast<std::uint64_t>(value.tv_usec);
}

#if defined(__linux__)
long perfEventOpen(perf_event_attr *attr, pid_t pid, int cpu, int groupFd,
                   unsigned long flags) {
  return syscall(__NR_perf_event_open, attr, pid, cpu, groupFd, flags);
}

class HardwareCounter {
public:
  explicit HardwareCounter(std::uint64_t config) {
    perf_event_attr attr = {};
    attr.type = PERF_TYPE_HARDWARE;
    attr.size = sizeof(attr);
    attr.config = config;
    attr.disabled = 1;
    attr.exclude_kernel = 0;
    attr.exclude_hv = 1;
    fd = static_cast<int>(perfEventOpen(&attr, /*pid=*/0, /*cpu=*/-1,
                                        /*groupFd=*/-1, /*flags=*/0));
  }

  ~HardwareCounter() {
    if (fd >= 0)
      close(fd);
  }

  void start() {
    if (fd < 0)
      return;
    ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
  }

  std::optional<std::uint64_t> stop() {
    if (fd < 0)
      return std::nullopt;
    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    std::uint64_t value = 0;
    if (read(fd, &value, sizeof(value)) != static_cast<ssize_t>(sizeof(value)))
      return std::nullopt;
    return value;
  }

private:
  int fd = -1;
};
#endif

} // namespace

struct PerfScope::State {
  explicit State(llvm::StringRef phase) : phase(phase.str()) {
    wallStart = Clock::now();
    getrusage(RUSAGE_SELF, &usageStart);
#if defined(__linux__)
    instructions.emplace(PERF_COUNT_HW_INSTRUCTIONS);
    cycles.emplace(PERF_COUNT_HW_CPU_CYCLES);
    instructions->start();
    cycles->start();
#endif
  }

  ~State() {
#if defined(__linux__)
    std::optional<std::uint64_t> instructionCount = instructions->stop();
    std::optional<std::uint64_t> cycleCount = cycles->stop();
#else
    std::optional<std::uint64_t> instructionCount;
    std::optional<std::uint64_t> cycleCount;
#endif
    rusage usageEnd = {};
    getrusage(RUSAGE_SELF, &usageEnd);
    auto wallEnd = Clock::now();
    auto wallUs = std::chrono::duration_cast<std::chrono::microseconds>(
                      wallEnd - wallStart)
                      .count();
    std::uint64_t userUs =
        timevalMicros(usageEnd.ru_utime) - timevalMicros(usageStart.ru_utime);
    std::uint64_t sysUs =
        timevalMicros(usageEnd.ru_stime) - timevalMicros(usageStart.ru_stime);

    llvm::errs()
        << "[LYTHON_PERF] phase=" << phase << " wall_us=" << wallUs
        << " user_us=" << userUs << " sys_us=" << sysUs
        << " minor_faults=" << (usageEnd.ru_minflt - usageStart.ru_minflt)
        << " major_faults=" << (usageEnd.ru_majflt - usageStart.ru_majflt)
        << " voluntary_csw=" << (usageEnd.ru_nvcsw - usageStart.ru_nvcsw)
        << " involuntary_csw=" << (usageEnd.ru_nivcsw - usageStart.ru_nivcsw)
        << " maxrss=" << usageEnd.ru_maxrss;
    if (instructionCount)
      llvm::errs() << " instructions=" << *instructionCount;
    else
      llvm::errs() << " instructions=unavailable";
    if (cycleCount)
      llvm::errs() << " cycles=" << *cycleCount;
    else
      llvm::errs() << " cycles=unavailable";
    llvm::errs() << "\n";
  }

  using Clock = std::chrono::steady_clock;

  std::string phase;
  Clock::time_point wallStart;
  rusage usageStart = {};
#if defined(__linux__)
  std::optional<HardwareCounter> instructions;
  std::optional<HardwareCounter> cycles;
#endif
};

PerfScope::PerfScope(llvm::StringRef phase) {
  if (perfEnabled())
    state = std::make_unique<State>(phase);
}

PerfScope::~PerfScope() = default;

} // namespace py
