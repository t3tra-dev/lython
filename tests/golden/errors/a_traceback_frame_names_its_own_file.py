# WHAT: a method of another module, written into its caller here, raises; the
# traceback names THIS file for the caller's frame and the helper's for the
# method's, as CPython does.
#
# WHY THIS IS RUN: the frames exist only in what the running program prints. The
# inlined body is emitted under the helper's file name, and the caller's frame
# borrowed it: both frames read "a_method_of_another_module_raises.py", the
# first against this file's line number.
import a_method_of_another_module_raises as helper

table = helper.Table()
print(table.get("present"))
