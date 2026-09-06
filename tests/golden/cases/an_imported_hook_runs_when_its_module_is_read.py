# What: `__init_subclass__` for classes declared in an IMPORTED module. The
# hook is emitted at the class statement's position in module flow, and an
# imported module's body does not run -- so a library whose base declares one
# registered NOTHING, silently, which is the one thing that pattern exists to
# do. The same classes in the main module have run their hooks since the day
# the hook was implemented.
#
# The order is the whole evidence and only running it shows it: the library's
# own subclass has to be registered before the importer's, the middle class
# has to get its PARENT's hook while declaring one of its own, and the
# class-header keyword has to reach it across the file.
from a_module_of_registered_classes import Alpha, Middle, Registry


class Beta(Registry, tag="b"):
    pass


class Leaf(Middle, tag="l"):
    pass


print(Registry.seen)
