# `sys.exit("bye")` prints the message to stderr and exits 1; sys.exit took
# only an int and refused the call.
import sys

print("before")
sys.exit("bye")
