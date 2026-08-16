# time's deterministic surface. The clocks can only be checked for the
# properties that do not depend on when the case runs (ordering, sign,
# elapsed-at-least); the calendar is pinned exactly, against fixed epoch
# seconds read in UTC so no timezone enters.
#
# strftime's directives are all numeric: %a / %b would depend on LC_TIME.
import time

# --- clocks -----------------------------------------------------------------
print(time.time() > 1700000000.0)
print(time.time_ns() > 1700000000000000000)
print(time.time_ns() // 1000000000 > 1700000000)

first = time.monotonic_ns()
second = time.monotonic_ns()
print(second >= first)
print(time.monotonic() > 0.0)
print(time.perf_counter() > 0.0)
print(time.perf_counter_ns() > 0)

before = time.monotonic_ns()
time.sleep(0.01)
after = time.monotonic_ns()
print(after - before >= 10000000)

# --- the epoch itself -------------------------------------------------------
t = time.gmtime(0)
print(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
print(t.tm_wday, t.tm_yday, t.tm_isdst, t.tm_gmtoff)
print(time.strftime("%Y-%m-%dT%H:%M:%S", t))

# --- a second whose broken-down form is memorable ---------------------------
t = time.gmtime(1000000000)
print(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
print(t.tm_wday, t.tm_yday)
print(time.strftime("%Y-%m-%d %H:%M:%S", t))
print(time.strftime("%j %w %y %H", t))

# --- a leap year's first and last day ---------------------------------------
t = time.gmtime(1577836800)
print(t.tm_year, t.tm_mon, t.tm_mday, t.tm_wday, t.tm_yday)
print(time.strftime("%Y-%m-%dT%H:%M:%S", t))
t = time.gmtime(1609372800)
print(t.tm_year, t.tm_mon, t.tm_mday, t.tm_wday, t.tm_yday)
t = time.gmtime(1583020800)
print(t.tm_year, t.tm_mon, t.tm_mday, t.tm_yday)

# --- the last second before an hour rolls over ------------------------------
t = time.gmtime(1234567890)
print(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec, t.tm_wday)
print(time.strftime("%Y%m%d%H%M%S", t))

# --- localtime is at least self-consistent ---------------------------------
now = time.localtime()
print(now.tm_year >= 2026)
print(1 <= now.tm_mon and now.tm_mon <= 12)
print(1 <= now.tm_mday and now.tm_mday <= 31)
print(0 <= now.tm_hour and now.tm_hour <= 23)
print(0 <= now.tm_wday and now.tm_wday <= 6)
print(1 <= now.tm_yday and now.tm_yday <= 366)

# --- strftime with no struct_time means NOW, as it does in CPython ---------
# It used to require the argument, recorded as a deviation in time.py's
# docstring, and `time.strftime("%Y")` -- the shape a log line is written in --
# was "call arguments do not match the Callable contract". Only the properties
# that do not depend on when the case runs can be pinned, so this checks the
# width and that the default agrees with an explicit localtime().
stamp = time.strftime("%Y-%m-%d")
print(len(stamp) == 10, stamp[4] == "-", stamp[7] == "-")
print(int(stamp[0:4]) >= 2026)
print(time.strftime("%Y") == time.strftime("%Y", time.localtime()))
print(len(time.strftime("%H:%M:%S")) == 8)

# --- a negative sleep is a ValueError, with CPython's message --------------
try:
    time.sleep(-1.0)
except ValueError as exc:
    print(exc)
