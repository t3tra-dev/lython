# WHAT: A base class and a function taking a list of its subclasses. Run
#   alone it prints nothing.
# KIND: helper module of ut_bug_imported_function_list_of_one_subclass_value
class TestCase:
    pass


def main(classes: list[type[TestCase]]) -> None:
    print([c.__name__ for c in classes])
