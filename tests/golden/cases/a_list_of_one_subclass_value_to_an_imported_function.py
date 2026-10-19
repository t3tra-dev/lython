# WHAT: a list display of ONE subclass of another module's class, passed
# where that module's function declares a list of the base's class values;
# and of two, and through the imported name.
#
# WHY THIS IS RUN: the classes the function reads are what the list holds at
# run time. A one-element display kept its own `list[type[Mine]]` -- the
# base's class value was not seen to accept a subclass's across modules --
# and the call was refused in the lowering; two elements joined to the base.
import a_list_of_one_subclass_value_to_an_imported_function_lib as lib
from a_list_of_one_subclass_value_to_an_imported_function_lib import main


class Mine(lib.TestCase):
    pass


class Other(lib.TestCase):
    pass


lib.main([Mine])
lib.main([Mine, Other])
main([Other])
