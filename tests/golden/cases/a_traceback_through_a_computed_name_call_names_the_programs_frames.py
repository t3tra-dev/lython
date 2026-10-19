# A test method called by a computed name from an imported runner --
# `getattr(self, self.name)()`, as unittest calls one -- shows in a traceback
# as its own frame, in its own file: the frame named the runner's file
# against the test's line, and the function the compiler writes to choose the
# method by name showed as a frame of its own (`__lygetattr$5`). A filter by
# file name then dropped the test's frame as the runner's.
# Why run it: the frames are what the run records as the exception passes.
from a_traceback_through_a_computed_name_call_names_the_programs_frames_lib import Case


class Mine(Case):
    def test_compare(self) -> None:
        self.same(1 + 1, 3)

    def test_fail(self) -> None:
        self.fail("direct")

    def test_helper(self) -> None:
        helper()


def helper() -> None:
    raise AssertionError("from a helper")


for name in ["test_compare", "test_fail", "test_helper"]:
    Mine(name).run()
