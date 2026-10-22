# warnings.warn attributes a warning to the frame `stacklevel` calls up, prints
# it once per place under the default filter, and the filters, catch_warnings
# and unittest's assertWarns / assertWarnsRegex behave as CPython's do.
import io
import sys
import unittest
import warnings


def warn_here(text: str) -> None:
    warnings.warn(text, UserWarning)


def warn_at_caller(text: str) -> None:
    warnings.warn(text, UserWarning, stacklevel=2)


def call_warn_at_caller(text: str) -> int:
    warn_at_caller(text)  # the line stacklevel=2 names
    return 0


def lineno_of(marker: str) -> int:
    with open(__file__) as f:
        for number, line in enumerate(f.read().splitlines(), 1):
            if line.endswith("# " + marker):
                return number
    return 0


class WarnTest(unittest.TestCase):

    def test_record_names_this_line(self) -> None:
        with warnings.catch_warnings(record=True) as log:
            warnings.simplefilter("always")
            warnings.warn("here")  # record-line
        assert log is not None
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0].lineno, lineno_of("record-line"))
        self.assertEqual(log[0].filename, __file__)
        self.assertIs(log[0].category, UserWarning)
        self.assertEqual(str(log[0].message), "here")

    def test_stacklevel_names_the_caller(self) -> None:
        with warnings.catch_warnings(record=True) as log:
            warnings.simplefilter("always")
            call_warn_at_caller("up")
        assert log is not None
        self.assertEqual(log[0].lineno,
                         lineno_of("the line stacklevel=2 names"))

    def test_stacklevel_past_the_bottom_is_sys(self) -> None:
        with warnings.catch_warnings(record=True) as log:
            warnings.simplefilter("always")
            warnings.warn("deep", stacklevel=1000)
        assert log is not None
        self.assertEqual(log[0].filename, "<sys>")
        self.assertEqual(log[0].lineno, 0)

    def test_default_filter_prints_once_per_place(self) -> None:
        with warnings.catch_warnings(record=True) as log:
            warnings.simplefilter("default")
            for _ in range(3):
                warn_here("same")
            warn_here("other")
        assert log is not None
        self.assertEqual([str(m.message) for m in log], ["same", "other"])

    def test_ignore_and_error(self) -> None:
        with warnings.catch_warnings(record=True) as log:
            warnings.simplefilter("ignore")
            warn_here("dropped")
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertRaises(RuntimeWarning):
                warnings.warn("raised", RuntimeWarning)
            warn_here("still dropped")
        assert log is not None
        self.assertEqual(log, [])

    def test_filterwarnings_matches_the_message(self) -> None:
        with warnings.catch_warnings(record=True) as log:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", message="noisy")
            warn_here("noisy thing")
            warn_here("quiet thing")
        assert log is not None
        self.assertEqual([str(m.message) for m in log], ["quiet thing"])

    def test_catch_warnings_restores_the_filters(self) -> None:
        before = list(warnings.filters)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertNotEqual(list(warnings.filters), before)
        self.assertEqual(list(warnings.filters), before)

    def test_formatwarning(self) -> None:
        text = warnings.formatwarning(UserWarning("msg"), UserWarning,
                                      "file.py", 3, "  code()  ")
        self.assertEqual(text, "file.py:3: UserWarning: msg\n  code()\n")

    def test_printed_to_stderr_with_the_source_line(self) -> None:
        saved = sys.stderr
        captured = io.StringIO()
        sys.stderr = captured
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn("printed")  # printed-line
        finally:
            sys.stderr = saved
        self.assertEqual(captured.getvalue(),
                         "%s:%d: UserWarning: printed\n"
                         "  warnings.warn(\"printed\")  # printed-line\n"
                         % (__file__, lineno_of("printed-line")))

    def test_showwarning_to_a_file(self) -> None:
        out = io.StringIO()
        warnings.showwarning(UserWarning("shown"), UserWarning, "f.py", 7,
                             out, "x = 1")
        self.assertEqual(out.getvalue(), "f.py:7: UserWarning: shown\n  x = 1\n")


class AssertWarnsTest(unittest.TestCase):

    def test_assert_warns(self) -> None:
        with self.assertWarns(DeprecationWarning) as cm:
            warnings.warn("old", DeprecationWarning)  # assert-warns-line
        self.assertEqual(str(cm.warning), "old")
        self.assertEqual(cm.lineno, lineno_of("assert-warns-line"))
        self.assertEqual(cm.filename, __file__)

    def test_assert_warns_fails_when_nothing_warns(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            with self.assertWarns(UserWarning):
                pass
        self.assertEqual(str(cm.exception), "UserWarning not triggered")

    def test_assert_warns_regex(self) -> None:
        with self.assertWarnsRegex(UserWarning, r"o+ps"):
            warn_here("whooops")
        with self.assertRaises(AssertionError) as cm:
            with self.assertWarnsRegex(UserWarning, "nope"):
                warn_here("whooops")
        self.assertEqual(str(cm.exception), '"nope" does not match "whooops"')

    def test_assert_warns_twice_in_one_place(self) -> None:
        # assertWarns clears the registries, so a place that already warned
        # under the default filter warns again.
        for _ in range(2):
            with self.assertWarns(UserWarning):
                warn_here("again")


if __name__ == "__main__":
    unittest.main()
