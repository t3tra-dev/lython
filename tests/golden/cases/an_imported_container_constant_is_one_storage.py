# What: a module-level CONTAINER constant in an imported module. A constant
# travels as a literal TYPE and a container has no literal spelling, so it
# resolved from nowhere -- "module 'm' has no attribute 'NAMES'" from the
# importer and "unresolved name 'NAMES'" from a function in its own module --
# while the same three lines in the MAIN module compile, because there the
# assignment IS a module global cell.
#
# It is one cell now, and only running it says so: the list the importer reads
# and the list the module's own function appends to have to be the SAME list,
# and the initializer has to have run before the main module's first statement.
# The scalar beside it still travels as a literal, which is what makes the
# container case about storage rather than about scope.
import a_module_of_container_constants as constants
from a_module_of_container_constants import PAIR, TABLE

print("read", constants.NAMES, TABLE, PAIR, constants.COUNT)
print("annotated", sorted(constants.TAGS), constants.EMPTY)
print("from its own body", constants.count(), constants.first())

print("appended there", constants.add("c"), constants.NAMES)
constants.NAMES.append("d")
print("appended here", constants.count(), constants.NAMES)
