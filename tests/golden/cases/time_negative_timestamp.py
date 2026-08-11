# Why execution: `gmtime(-1)` returned the CURRENT year. "now" was spelled as
# any negative value, so every pre-1970 timestamp silently meant now. The year
# is the assertion, and it has to come from running it.
import time


def main() -> None:
    print(time.gmtime(-1).tm_year, time.gmtime(-1).tm_mday)
    print(time.gmtime(0).tm_year, time.gmtime(0).tm_mon)
    print(time.gmtime(1000000000).tm_year)
    print(time.localtime(-1).tm_year)
    print(time.gmtime().tm_year > 2000)


main()
