# WHAT: a function reads `__name__` and `__file__`, and gets those of the
# module it is written in: `__main__` here, the imported module's own in a
# function of that module and at that module's top level.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answer is a value the
# program prints, and which module's it is depends on where the body is
# written, not on the module that calls it.
#
# ⛔ Both names were bound only in the main module's own scope, so a function
# that read either was "unresolved name" -- and an imported module's top-level
# binding read the MAIN module's, silently.
import a_module_that_reports_its_name as reporter


def mine() -> str:
    return __name__ + " " + __file__.rsplit("/", 1)[-1]


print(mine())
print(reporter.report())
print(reporter.OWN_NAME, reporter.OWN_FILE)
print(__name__ == "__main__")
