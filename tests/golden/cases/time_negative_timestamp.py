# Why execution: `gmtime(-1)` returned the CURRENT year. "now" was spelled as
# any negative value, so every pre-1970 timestamp silently meant now. The year
# is the assertion, and it has to come from running it.
import time


def main() -> None:
    print(time.gmtime(-1).tm_year, time.gmtime(-1).tm_mday)
    print(time.gmtime(0).tm_year, time.gmtime(0).tm_mon)
    print(time.gmtime(1000000000).tm_year)
    # ⛔ Not the YEAR: which side of the epoch -1 lands on is the machine's
    # time zone (1969 west of UTC, 1970 east of it), and this golden ran on a
    # UTC+9 box and then failed in CI at UTC. What the defect printed was the
    # CURRENT year, so "before 1971" is the assertion that catches it and does
    # not name a zone.
    print(time.localtime(-1).tm_year < 1971)
    print(time.gmtime().tm_year > 2000)


main()
