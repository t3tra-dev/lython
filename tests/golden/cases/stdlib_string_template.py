# What: Template safe_substitute keeps unknown/invalid placeholders intact,
# is_valid/get_identifiers scan like the default CPython pattern, and a
# missing substitute key raises the dict's KeyError.
from string import Template

t2 = Template("na$me")
print(t2.substitute({"me": "X"}))
t3 = Template("$who ${w x")
print(t3.safe_substitute({"who": "t"}))
t4 = Template("$who $$")
print(t4.is_valid())
t5 = Template("$ ")
print(t5.is_valid())
t6 = Template("$who x ${who} $other")
print(t6.get_identifiers())
t7 = Template("$who")
try:
    t7.substitute({"other": "y"})
except KeyError as e:
    print("KeyError:", e)
print("done")
