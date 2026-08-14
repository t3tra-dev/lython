# A dict literal with a NON-STATIC key, built inside nested loops. Execution is
# needed because the defect was a refusal that came from the verifier running
# out of states, not from anything wrong with the program -- so the only way to
# pin the repair is to run it and read the sums. The trip counts matter: the
# state explosion grew with them, and `range(3, 6)` past the immortal small-int
# cache {0, 1, 2} is where the twin defect in July showed an over-release the
# cap had been hiding.
#
# The static-key spelling is here too. It always compiled, took the other
# lowering, and is what a repair that moved the stamping to the wrong path
# would break.

probe = 0
for i in range(3, 6):
    for j in range(2):
        d = {i: 1}
        for k in d:
            probe += k
print(probe)

payload = 0
for i in range(3, 6):
    for j in range(2):
        e = {"k": i}
        payload += e["k"]
print(payload)

# Both key kinds in one literal: one non-static key sends the whole thing down
# the probe path, so the static entry is charged there too.
mixed = 0
for i in range(3, 6):
    m = {"a": 1, i: 2}
    mixed += len(m)
print(mixed)
