# What: a host function whose stub declares a `number` takes an int, and a
# result the statement discards is not checked -- node's `setTimeout` returns
# a Timeout object where the (browser) stub declares a number.
#
# Why run: the timer fires after main returns, from the host's loop.
from js import setTimeout


def later() -> None:
    print("timer fired")


setTimeout(later, 10)
print("main done")
