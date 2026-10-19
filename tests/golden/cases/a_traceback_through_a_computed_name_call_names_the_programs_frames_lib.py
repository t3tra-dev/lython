import sys
import traceback


class Case:
    failureException: type[BaseException] = AssertionError

    def __init__(self, name: str) -> None:
        self.name = name

    def same[T](self, first: T, second: T) -> None:
        if first != second:
            self.fail("differ")

    def fail(self, msg: str) -> None:
        raise self.failureException(msg)

    def run(self) -> None:
        try:
            getattr(self, self.name)()
        except AssertionError as e:
            frames = [entry.split("/")[-1].split("\n")[0].replace(
                          "a_traceback_through_a_computed_name_call_names_the_programs_frames", "main")
                      for entry in traceback.format_tb(sys.exc_info()[2])]
            print(self.name, repr(e), frames)
