# What: user-visible exceptions from Template.substitute are catchable
# with CPython messages - ValueError for an invalid placeholder,
# KeyError for a missing mapping key - and execution continues.
from string import Template

bad = Template("abc $ x")
try:
    bad.substitute({"who": "y"})
    print("NO RAISE")
except ValueError as e:
    print("caught:", e)
t = Template("$who")
try:
    t.substitute({"other": "y"})
except KeyError as e:
    print("caught:", e)
print("after")
