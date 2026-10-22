# logging attributes a record to the caller of the logging call, formats it
# with the three styles and printf conversions as CPython does, walks the
# logger hierarchy, and unittest's assertLogs / assertNoLogs capture it.
import io
import logging
import unittest


def lineno_of(marker: str) -> int:
    with open(__file__) as f:
        for number, line in enumerate(f.read().splitlines(), 1):
            if line.endswith("# " + marker):
                return number
    return 0


def capture(name: str, fmt: str, style: str = "%") -> io.StringIO:
    out = io.StringIO()
    handler = logging.StreamHandler(out)
    handler.setFormatter(logging.Formatter(fmt, style=style))
    logger = logging.getLogger(name)
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return out


def emit_from_helper(logger: logging.Logger) -> None:
    logger.info("from helper")  # helper-line


class RecordTest(unittest.TestCase):

    def test_caller_is_the_logging_call(self) -> None:
        out = capture("t.caller", "%(funcName)s:%(lineno)d:%(module)s:%(message)s")
        logging.getLogger("t.caller").info("here")  # caller-line
        self.assertEqual(out.getvalue(),
                         "test_caller_is_the_logging_call:%d:test_logging:here\n"
                         % lineno_of("caller-line"))

    def test_stacklevel(self) -> None:
        out = capture("t.stack", "%(funcName)s:%(lineno)d")
        emit_from_helper(logging.getLogger("t.stack"))
        self.assertEqual(out.getvalue(),
                         "emit_from_helper:%d\n" % lineno_of("helper-line"))

    def test_printf_conversions(self) -> None:
        out = capture("t.printf", "%(message)s|%(levelname)-8s|%(levelno)03d")
        logging.getLogger("t.printf").warning(
            "%s %r %d %5.2f %-4d| %x %o %e %%", "s", "r", 7, 3.14159, 42,
            255, 8, 12345.678)
        self.assertEqual(out.getvalue(),
                         "s 'r' 7  3.14 42  | ff 10 1.234568e+04 %|WARNING |030\n")

    def test_brace_and_dollar_styles(self) -> None:
        brace = capture("t.brace", "{levelname}:{name}:{message!r}", style="{")
        logging.getLogger("t.brace").error("x")
        dollar = capture("t.dollar", "$levelname ${name} $message", style="$")
        logging.getLogger("t.dollar").error("y")
        self.assertEqual(brace.getvalue(), "ERROR:t.brace:'x'\n")
        self.assertEqual(dollar.getvalue(), "ERROR t.dollar y\n")

    def test_extra(self) -> None:
        out = capture("t.extra", "[%(user)s] %(message)s")
        logging.getLogger("t.extra").info("hi", extra={"user": "ann"})
        self.assertEqual(out.getvalue(), "[ann] hi\n")
        with self.assertRaises(KeyError):
            logging.getLogger("t.extra").info("x", extra={"message": "no"})

    def test_exception_appends_the_traceback(self) -> None:
        out = capture("t.exc", "%(levelname)s %(message)s")
        try:
            raise ValueError("bad value")
        except ValueError:
            logging.getLogger("t.exc").exception("caught")
        lines = out.getvalue().splitlines()
        self.assertEqual(lines[0], "ERROR caught")
        self.assertEqual(lines[1], "Traceback (most recent call last):")
        self.assertEqual(lines[-1], "ValueError: bad value")


class HierarchyTest(unittest.TestCase):

    def test_parents(self) -> None:
        leaf = logging.getLogger("h.mid.leaf")
        mid = logging.getLogger("h.mid")
        top = logging.getLogger("h")
        self.assertIs(leaf.parent, mid)
        self.assertIs(mid.parent, top)
        self.assertIs(top.parent, logging.getLogger())
        self.assertIs(logging.getLogger("h.mid"), mid)
        self.assertIs(top.getChild("mid.leaf"), leaf)

    def test_effective_level_and_propagation(self) -> None:
        out = capture("p", "%(name)s %(message)s")
        child = logging.getLogger("p.child")
        child.setLevel(logging.NOTSET)
        self.assertEqual(child.getEffectiveLevel(), logging.DEBUG)
        child.debug("up")
        logging.getLogger("p").setLevel(logging.ERROR)
        child.warning("dropped")
        self.assertEqual(out.getvalue(), "p.child up\n")

    def test_filter(self) -> None:
        out = capture("f", "%(name)s")
        logging.getLogger("f").handlers[0].addFilter(logging.Filter("f.a"))
        logging.getLogger("f.a").info("x")
        logging.getLogger("f.a.b").info("x")
        logging.getLogger("f.ab").info("x")
        logging.getLogger("f.b").info("x")
        self.assertEqual(out.getvalue(), "f.a\nf.a.b\n")

    def test_levels(self) -> None:
        self.assertEqual(logging.getLevelName(logging.WARNING), "WARNING")
        self.assertEqual(logging.getLevelName(15), "Level 15")
        logging.addLevelName(15, "VERBOSE")
        self.assertEqual(logging.getLevelName(15), "VERBOSE")
        logger = logging.getLogger("lv")
        logger.setLevel("ERROR")
        self.assertFalse(logger.isEnabledFor(logging.WARNING))
        self.assertTrue(logger.isEnabledFor(logging.CRITICAL))
        self.assertEqual(repr(logger), "<Logger lv (ERROR)>")

    def test_disable(self) -> None:
        out = capture("d", "%(message)s")
        logging.disable(logging.WARNING)
        logging.getLogger("d").warning("muted")
        logging.disable(logging.NOTSET)
        logging.getLogger("d").warning("heard")
        self.assertEqual(out.getvalue(), "heard\n")


class AssertLogsTest(unittest.TestCase):

    def test_assert_logs(self) -> None:
        with self.assertLogs("foo", level="INFO") as cm:
            logging.getLogger("foo").info("first message")
            logging.getLogger("foo.bar").error("second message")
            logging.getLogger("foo").debug("not captured")
        self.assertEqual(cm.output, ["INFO:foo:first message",
                                     "ERROR:foo.bar:second message"])
        self.assertEqual(cm.records[1].levelname, "ERROR")

    def test_assert_logs_fails_without_logs(self) -> None:
        with self.assertRaises(AssertionError) as cm:
            with self.assertLogs("quiet", level=logging.WARNING):
                logging.getLogger("quiet").info("too low")
        self.assertEqual(str(cm.exception),
                         "no logs of level WARNING or higher triggered on quiet")

    def test_assert_logs_restores_the_logger(self) -> None:
        logger = logging.getLogger("restored")
        handler = logging.NullHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
        with self.assertLogs(logger, level="DEBUG"):
            logger.debug("captured")
        self.assertEqual(logger.handlers, [handler])
        self.assertEqual(logger.level, logging.ERROR)
        self.assertTrue(logger.propagate)

    def test_assert_no_logs(self) -> None:
        with self.assertNoLogs("silent"):
            logging.getLogger("silent").debug("below INFO")
        with self.assertRaises(AssertionError) as cm:
            with self.assertNoLogs("silent"):
                logging.getLogger("silent").warning("loud")
        self.assertEqual(str(cm.exception),
                         "Unexpected logs found: ['WARNING:silent:loud']")


if __name__ == "__main__":
    unittest.main()
