# What: names bound by a tuple unpack inside a loop are locals of the scope
# around it, and they hold the LAST iteration's values. Running it is what
# shows the binding reached the read and carried the right element to each
# name -- a walk that binds them in the wrong order still resolves.
def last_setting(lines: "list[str]") -> str:
    for line in lines:
        key, _, value = line.partition("=")
    return key + " is " + value


for entry in ["a:1", "b:2"]:
    name, count = entry.split(":")
print(name, count)

if len(name) == 1:
    left, right = 10, "ten"
else:
    left, right = 20, "twenty"
print(left, right)

print(last_setting(["x=1", "y=2"]))
