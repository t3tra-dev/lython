# WHAT: a library function reading, through its base class, class attributes
# the program's subclasses override -- a field (`limit: int | None`) set to
# None and to a number, and a plain one (`label`) -- the way unittest reads
# `self.maxDiff` of the program's TestCase.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the library's bodies are
# emitted before the program's classes exist, and the read was refused
# ("'limit' is overridden by a subclass of ..."); the dispatcher now waits
# for them. Which class answered each read is seen only in the strings.
import a_library_reads_an_attribute_the_program_overrides_lib as lib


class Unlimited(lib.Report):
    limit = None


class Short(lib.Report):
    limit = 3
    label = "short"


print(lib.render_all([lib.Report(), Unlimited(), Short()], "abcdef"))
