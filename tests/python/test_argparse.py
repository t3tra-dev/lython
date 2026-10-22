# argparse parses what CPython's parses, reports the same errors, and formats
# usage and help byte for byte as CPython does when its output is not a
# terminal.
import argparse
import io
import os
import sys
import unittest


def parser_for_errors() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="e", exit_on_error=False)
    p.add_argument("--n", type=int)
    p.add_argument("--mode", choices=["a", "b"])
    p.add_argument("pos")
    return p


def error_of(p: argparse.ArgumentParser, argv: list[str]) -> str:
    try:
        p.parse_args(argv)
    except argparse.ArgumentError as e:
        return str(e)
    return ""


class ParseTest(unittest.TestCase):

    def setUp(self) -> None:
        os.environ["COLUMNS"] = "80"

    def test_positionals_options_and_types(self) -> None:
        p = argparse.ArgumentParser(prog="tool")
        p.add_argument("integers", type=int, nargs="+")
        p.add_argument("--sum", dest="accumulate", action="store_const",
                       const="sum", default="max")
        p.add_argument("-v", "--verbose", action="count", default=0)
        p.add_argument("--name", default="world")
        args = p.parse_args(["1", "2", "3", "--sum", "-vv", "--name=you"])
        self.assertEqual(repr(args),
                         "Namespace(integers=[1, 2, 3], accumulate='sum', "
                         "verbose=2, name='you')")
        self.assertEqual(repr(args.integers), "[1, 2, 3]")
        self.assertEqual(repr(args.verbose), "2")

    def test_nargs_append_extend_and_boolean(self) -> None:
        q = argparse.ArgumentParser(prog="q")
        q.add_argument("--add", action="append", type=int)
        q.add_argument("--two", nargs=2, metavar=("A", "B"))
        q.add_argument("--opt", nargs="?", const="C", default="D")
        q.add_argument("--flag", action=argparse.BooleanOptionalAction)
        q.add_argument("--ext", action="extend", nargs="+")
        q.add_argument("rest", nargs="*")
        args = q.parse_args(["--add", "1", "--add", "2", "--two", "x", "y",
                             "--opt", "--no-flag", "--ext", "e1", "e2",
                             "--ext", "e3", "--", "-r", "s"])
        self.assertEqual(repr(args),
                         "Namespace(add=[1, 2], two=['x', 'y'], opt='C', "
                         "flag=False, ext=['e1', 'e2', 'e3'], rest=['-r', 's'])")
        self.assertEqual(repr(q.parse_args(["--opt=v", "--flag"])),
                         "Namespace(add=None, two=None, opt='v', flag=True, "
                         "ext=None, rest=[])")

    def test_abbreviation_and_known_args(self) -> None:
        r = argparse.ArgumentParser(prog="r")
        r.add_argument("--verbose", action="store_true")
        r.add_argument("--value", type=float, default=1.5)
        ns, extra = r.parse_known_args(["--verb", "--valu", "2.5",
                                        "--unknown", "x"])
        self.assertEqual(repr(ns), "Namespace(verbose=True, value=2.5)")
        self.assertEqual(extra, ["--unknown", "x"])

    def test_errors(self) -> None:
        p = parser_for_errors()
        self.assertEqual(error_of(p, ["x", "--n", "zz"]),
                         "argument --n: invalid int value: 'zz'")
        self.assertEqual(error_of(p, ["x", "--mode", "c"]),
                         "argument --mode: invalid choice: 'c' "
                         "(choose from 'a', 'b')")
        self.assertEqual(error_of(p, []),
                         "the following arguments are required: pos")
        self.assertEqual(error_of(p, ["x", "--n"]),
                         "argument --n: expected one argument")
        self.assertEqual(error_of(p, ["x", "y"]), "unrecognized arguments: y")

    def test_error_exits_with_usage_on_stderr(self) -> None:
        p = argparse.ArgumentParser(prog="ex")
        p.add_argument("--n", type=int, required=True)
        saved = sys.stderr
        captured = io.StringIO()
        sys.stderr = captured
        try:
            with self.assertRaises(SystemExit):
                p.parse_args([])
        finally:
            sys.stderr = saved
        self.assertEqual(captured.getvalue(),
                         "usage: ex [-h] --n N\n"
                         "ex: error: the following arguments are required: --n\n")

    def test_mutually_exclusive(self) -> None:
        m = argparse.ArgumentParser(prog="m", exit_on_error=False)
        g = m.add_mutually_exclusive_group(required=True)
        g.add_argument("--yes", action="store_true")
        g.add_argument("--no", action="store_true")
        self.assertEqual(error_of(m, ["--yes", "--no"]),
                         "argument --no: not allowed with argument --yes")
        self.assertEqual(error_of(m, []),
                         "one of the arguments --yes --no is required")
        self.assertEqual(m.format_usage(), "usage: m [-h] (--yes | --no)\n")

    def test_subparsers(self) -> None:
        s = argparse.ArgumentParser(prog="git")
        s.add_argument("--debug", action="store_true")
        subs = s.add_subparsers(dest="command", title="commands",
                                help="sub-command help")
        c = subs.add_parser("commit", help="record changes")
        c.add_argument("-m", "--message", required=True)
        pu = subs.add_parser("push", aliases=["p"], help="update remote")
        pu.add_argument("remote", nargs="?", default="origin")
        self.assertEqual(repr(s.parse_args(["--debug", "commit", "-m", "msg"])),
                         "Namespace(debug=True, command='commit', message='msg')")
        self.assertEqual(repr(s.parse_args(["p", "upstream"])),
                         "Namespace(debug=False, command='p', remote='upstream')")
        self.assertEqual(s.format_help(),
                         "usage: git [-h] [--debug] {commit,push,p} ...\n"
                         "\n"
                         "options:\n"
                         "  -h, --help       show this help message and exit\n"
                         "  --debug\n"
                         "\n"
                         "commands:\n"
                         "  {commit,push,p}  sub-command help\n"
                         "    commit         record changes\n"
                         "    push (p)       update remote\n")


class HelpTest(unittest.TestCase):

    def setUp(self) -> None:
        os.environ["COLUMNS"] = "80"

    def test_help_with_groups_and_defaults(self) -> None:
        f = argparse.ArgumentParser(
            prog="fmt", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="A tool that does quite a lot of things, described at "
                        "enough length that the description has to wrap onto a "
                        "second line.",
            epilog="See the manual.")
        grp = f.add_argument_group("output", "Options that control output.")
        grp.add_argument("--width", type=int, default=80,
                         help="the width to wrap at")
        f.add_argument("inputs", nargs="+", help="input files")
        self.assertEqual(f.format_help(),
                         "usage: fmt [-h] [--width WIDTH] inputs [inputs ...]\n"
                         "\n"
                         "A tool that does quite a lot of things, described at "
                         "enough length that the\n"
                         "description has to wrap onto a second line.\n"
                         "\n"
                         "positional arguments:\n"
                         "  inputs         input files\n"
                         "\n"
                         "options:\n"
                         "  -h, --help     show this help message and exit\n"
                         "\n"
                         "output:\n"
                         "  Options that control output.\n"
                         "\n"
                         "  --width WIDTH  the width to wrap at (default: 80)\n"
                         "\n"
                         "See the manual.\n")

    def test_long_usage_wraps(self) -> None:
        q = argparse.ArgumentParser(prog="q")
        q.add_argument("--add", action="append", type=int)
        q.add_argument("--many", nargs="*")
        q.add_argument("--flag", action=argparse.BooleanOptionalAction)
        q.add_argument("--ext", action="extend", nargs="+")
        q.add_argument("--another-option", metavar="VALUE")
        q.add_argument("rest", nargs="*")
        self.assertEqual(q.format_usage(),
                         "usage: q [-h] [--add ADD] [--many [MANY ...]] "
                         "[--flag | --no-flag]\n"
                         "         [--ext EXT [EXT ...]] "
                         "[--another-option VALUE]\n"
                         "         [rest ...]\n")

    def test_help_percent_expansion(self) -> None:
        p = argparse.ArgumentParser(prog="pct")
        p.add_argument("--n", type=int, default=3,
                       help="count (default: %(default)s, type %(type)s)")
        self.assertTrue("count (default: 3, type int)" in p.format_help())


class NamespaceTest(unittest.TestCase):

    def test_namespace(self) -> None:
        ns = argparse.Namespace(a=1, b="x")
        self.assertEqual(repr(ns), "Namespace(a=1, b='x')")
        self.assertEqual(ns, argparse.Namespace(a=1, b="x"))
        self.assertTrue("a" in ns)
        self.assertEqual(repr(ns.a), "1")
        with self.assertRaises(AttributeError):
            print(ns.missing)


if __name__ == "__main__":
    unittest.main()
