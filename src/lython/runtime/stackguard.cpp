// Native stack overflow diagnostics. Deeply recursive Python programs hit the
// native stack guard page and die with a bare SIGSEGV/SIGBUS; this handler
// classifies faults inside the stack guard region and reports them in
// CPython's RecursionError vocabulary instead.

#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <pthread.h>
#include <unistd.h>

extern "C" void LyRt_InstallStackGuard();

namespace {

std::uintptr_t mainStackLimit = 0;

bool isStackOverflowAddress(std::uintptr_t address) {
  if (!mainStackLimit)
    return false;
  // The guard region sits directly below the lowest valid stack address.
  // Frames are small enough that an overflowing store lands within a few
  // pages below the limit; allow a generous window without swallowing
  // unrelated faults.
  constexpr std::uintptr_t kPage = 4096;
  constexpr std::uintptr_t kBelow = 64 * kPage;
  constexpr std::uintptr_t kAbove = kPage;
  return address >= mainStackLimit - kBelow &&
         address < mainStackLimit + kAbove;
}

void stackGuardHandler(int signalNumber, siginfo_t *info, void *) {
  if (info &&
      isStackOverflowAddress(reinterpret_cast<std::uintptr_t>(info->si_addr))) {
    constexpr char kMessage[] =
        "RecursionError: maximum recursion depth exceeded "
        "(native stack overflow)\n";
    (void)!write(STDERR_FILENO, kMessage, sizeof(kMessage) - 1);
    _exit(1);
  }
  // Not a stack overflow: fall back to the default disposition so crash
  // reporting (stack traces, core dumps) keeps working.
  signal(signalNumber, SIG_DFL);
  raise(signalNumber);
}

struct StackGuardInstaller {
  StackGuardInstaller() { LyRt_InstallStackGuard(); }
};

StackGuardInstaller installer;

} // namespace

extern "C" void LyRt_InstallStackGuard() {
  static bool installed = false;
  if (installed)
    return;
  installed = true;

#if defined(__APPLE__)
  pthread_t self = pthread_self();
  std::uintptr_t stackTop =
      reinterpret_cast<std::uintptr_t>(pthread_get_stackaddr_np(self));
  std::size_t stackSize = pthread_get_stacksize_np(self);
  mainStackLimit = stackTop - stackSize;
#elif defined(__linux__)
  pthread_attr_t attr;
  if (pthread_getattr_np(pthread_self(), &attr) == 0) {
    void *stackBase = nullptr;
    std::size_t stackSize = 0;
    if (pthread_attr_getstack(&attr, &stackBase, &stackSize) == 0)
      mainStackLimit = reinterpret_cast<std::uintptr_t>(stackBase);
    pthread_attr_destroy(&attr);
  }
#endif
  if (!mainStackLimit)
    return;

  // The handler must run on an alternate stack: the faulting thread has no
  // usable stack left by definition.
  static stack_t alternateStack = {};
  std::size_t alternateSize = 4 * static_cast<std::size_t>(SIGSTKSZ);
  alternateStack.ss_sp = std::malloc(alternateSize);
  if (!alternateStack.ss_sp)
    return;
  alternateStack.ss_size = alternateSize;
  alternateStack.ss_flags = 0;
  if (sigaltstack(&alternateStack, nullptr) != 0)
    return;

  struct sigaction action;
  std::memset(&action, 0, sizeof(action));
  action.sa_sigaction = stackGuardHandler;
  action.sa_flags = SA_SIGINFO | SA_ONSTACK;
  sigemptyset(&action.sa_mask);
  sigaction(SIGSEGV, &action, nullptr);
  sigaction(SIGBUS, &action, nullptr);
}
