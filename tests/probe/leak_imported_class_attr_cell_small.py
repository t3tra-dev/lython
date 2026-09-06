# WHAT: an imported class's attribute cell REBOUND in a loop. The cell is
# module-lifetime storage, so the value it held before each store has to be
# released; the string is sized past the probe floor. The container attribute
# beside it is not rebound -- a growing list is growth, not a leak -- so only
# the scalar slot is under test.
from leak_imported_class_attr_lib import Slot

i = 0
while i < 300:
    Slot.tag = "z" * 4096
    Slot.hits = i
    i += 1
print("done", Slot.hits)
