import generic_import_lib
from generic_import_lib import first, pair_up, count_matches, label_of, Tagger
from generic_import_lib import describe

# One imported generic, two ground instantiations.
print(first([10, 20, 30]))
print(first(["a", "b"]))

# Qualified call reaches the same registration as the bare name.
print(generic_import_lib.first([7, 8]))

# Two independent type parameters.
print(pair_up(1, "x"))
print(pair_up("y", 2))

# The type parameter appears in a local annotation inside the body.
print(count_matches([1, 2, 2, 3, 2], 2))
print(count_matches(["a", "b", "a"], "a"))

# Module-level alias, imported by its alias name.
print(label_of("z"))

# Imported module-level helper reading its own module's globals.
print(describe("ab"))
print(generic_import_lib.describe("abcd"))

# Imported class whose method reads its own module's globals.
print(Tagger("q").tagged())
