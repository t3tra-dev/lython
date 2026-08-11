"""Time access and conversions.

CPython implements all of `time` in C (Modules/timemodule.c). Lython splits it
the way CPython splits io: the clock and calendar natives live in the `_time`
manifest (runtime/modules/_time.mlir) and this file is the thin public layer
that owns `struct_time` and the calls that return one. The split exists
because `struct_time` is a structseq -- a tuple subclass with named fields --
and a manifest cannot declare a class with named int fields yet. CPython has
no `_time` module; importing it directly is a Lython-only spelling.

Deviations from CPython:
  - `struct_time` is a plain class with the nine `tm_*` attributes plus
    `tm_gmtoff`. It does NOT index, unpack, compare, or iterate like the
    9-tuple CPython's structseq is, and `tm_zone` is absent (the zone name is
    a `char *` in `struct tm` with no owner). `time.strftime` and
    `time.mktime` therefore take a struct_time, never a bare tuple.
  - `localtime()` / `gmtime()` cost ten localtime_r/gmtime_r calls, one per
    field: the native layer hands fields across as scalars. The ten reads are
    of the SAME `seconds` argument, so the result is internally consistent;
    `localtime()` with no argument reads the clock once, here, and passes that
    fixed value down.
  - `strftime(format, t)` requires the struct_time argument; CPython defaults
    it to `localtime()`. Formatting is libc strftime's in the process locale,
    and a result longer than 1023 bytes comes back as ''.
  - `sleep()` does not retry on EINTR (PEP 475); an interrupted sleep raises
    InterruptedError.
  - `perf_counter` is `monotonic` (see the manifest docstring).
  - `strptime`, `asctime`, `ctime`, `tzset`, `process_time`, `thread_time`,
    `get_clock_info`, `clock_gettime`, and the `timezone`/`altzone`/`daylight`/
    `tzname` module attributes are not ported. `timezone`-style offsets are
    reachable through `struct_time.tm_gmtoff`.
"""

import _time
from _time import (time, time_ns, monotonic, monotonic_ns, perf_counter,
                  perf_counter_ns, sleep)

__all__ = [
    "time", "time_ns", "monotonic", "monotonic_ns", "perf_counter",
    "perf_counter_ns", "sleep", "struct_time", "localtime", "gmtime",
    "strftime", "mktime",
]


class struct_time:
    """The broken-down time: CPython's time.struct_time, as a plain class.

    Fields carry CPython's struct_time conventions, not libc's: tm_year is the
    full year, tm_mon is 1-12, tm_mday 1-31, tm_wday 0-6 with Monday 0, and
    tm_yday 1-366. tm_gmtoff is the seconds east of UTC.
    """

    def __init__(self, seconds: int, utc: int) -> None:
        self.tm_sec: int = _time.field(seconds, utc, 0)
        self.tm_min: int = _time.field(seconds, utc, 1)
        self.tm_hour: int = _time.field(seconds, utc, 2)
        self.tm_mday: int = _time.field(seconds, utc, 3)
        # libc's tm_mon is 0-based and tm_year counts from 1900.
        self.tm_mon: int = _time.field(seconds, utc, 4) + 1
        self.tm_year: int = _time.field(seconds, utc, 5) + 1900
        # libc's tm_wday is 0=Sunday; CPython's is 0=Monday.
        self.tm_wday: int = (_time.field(seconds, utc, 6) + 6) % 7
        # libc's tm_yday is 0-based.
        self.tm_yday: int = _time.field(seconds, utc, 7) + 1
        self.tm_isdst: int = _time.field(seconds, utc, 8)
        self.tm_gmtoff: int = _time.field(seconds, utc, 9)


# CPython's default is `seconds=None`. An Optional parameter is not accepted at
# a call site here yet ("static type ... is not callable"), and a module-level
# constant is not visible in a default expression, so "now" is spelled inline as
# the one int no clock can produce. It used to be ANY negative value, which made
# every pre-1970 timestamp mean "now": `gmtime(-1)` returned the current year
# instead of 1969.
def localtime(seconds: int = -9223372036854775808) -> struct_time:
    """Convert seconds since the Epoch to a struct_time in local time.

    Omitting `seconds` means "now", as `seconds=None` does in CPython.
    """
    when = seconds
    if when == -9223372036854775808:
        when = time_ns() // 1000000000
    return struct_time(when, 0)


def gmtime(seconds: int = -9223372036854775808) -> struct_time:
    """Convert seconds since the Epoch to a struct_time in UTC.

    Omitting `seconds` means "now"; see localtime.
    """
    when = seconds
    if when == -9223372036854775808:
        when = time_ns() // 1000000000
    return struct_time(when, 1)


def strftime(format: str, t: struct_time) -> str:
    """Format a struct_time through the platform's strftime."""
    return _time.strftime(format, t.tm_sec, t.tm_min, t.tm_hour, t.tm_mday,
                          t.tm_mon - 1, t.tm_year - 1900,
                          (t.tm_wday + 1) % 7, t.tm_yday - 1, t.tm_isdst)


def mktime(t: struct_time) -> int:
    """Seconds since the Epoch for a struct_time read as LOCAL time.

    CPython returns a float; the platform's mktime has one-second resolution,
    so this returns the int it actually produces.
    """
    return _time.mktime(t.tm_sec, t.tm_min, t.tm_hour, t.tm_mday,
                        t.tm_mon - 1, t.tm_year - 1900,
                        (t.tm_wday + 1) % 7, t.tm_yday - 1, t.tm_isdst)
