# What: an invalid $-placeholder makes Template.substitute raise CPython's
# ValueError with the 1-based line/col of the offending position.
from string import Template

t = Template("abc $ x")
t.substitute({"who": "y"})
