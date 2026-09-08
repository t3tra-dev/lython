# A function that hands an element to its caller on one exit and takes another
# exit besides was refused:
#
#     def find(names: "list[str]", want: str) -> str:
#         for name in names:
#             if name == want:
#                 return name
#         return "<none>"
#     # owned resource from builtin.unrealized_conversion_cast result 0
#     # reaches function exit without release, transfer, or owned return
#
# -- a linear search. The release placer for owned locals was handed no
# deallocators, and they are the only thing that lets it read an owned return as
# a transfer, so it bailed at the first `return` and placed no release anywhere.
#
# Why execution: what changed is WHERE a release is written, and the two exits
# of each function below spend the element's reference differently. Only running
# both, repeatedly, shows each spent it exactly once -- a placement that is one
# short leaks and one long double-frees, and neither is visible in the IR.
#
# ⭐ The element is read through every container that mints a token for it: a
# list under a loop, a list under a subscript guard, a dict key. `magnitude`
# returns a DERIVED value on one arm and the element itself on the other, which
# is the pair the walk has to keep apart.
#
# ⛔ Reading an element of a container built in the SAME frame and returning it
# is still refused (`released owned resource ... is used by function return`) --
# a different defect, measured at tests/probe/wb_return_an_element_of_a_local_container.py.


def find(names: "list[str]", want: str) -> str:
    for name in names:
        if name == want:
            return name
    return "<none>"


def magnitude(xs: "list[int]") -> int:
    if len(xs) == 0:
        return 0
    head = xs[0]
    if head < 0:
        return -head
    return head


def first_key(table: "dict[str, int]") -> str:
    for key in table:
        return key
    return "<empty>"


names = ["ann", "bob", "cid"]
print(find(names, "bob"))
print(find(names, "zed"))
print(len(find(names, "bob")))

print(magnitude([7, 8]))
print(magnitude([-9]))
print(magnitude([]))

print(first_key({"host": 1}))

hits = 0
for _ in range(50):
    if find(names, "cid") == "cid":
        hits += 1
print(hits)
print(names)
