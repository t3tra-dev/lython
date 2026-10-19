# WHAT: A module that defines two tuple aliases and functions annotated with
#   them. Run alone it prints nothing.
# KIND: helper module of ut_bug_type_alias_of_an_imported_module
import types

Info = tuple[type[BaseException], BaseException, types.TracebackType]
Pair = tuple[int, str]


def first_name(info: Info) -> str:
    return info[0].__name__


def pair_text(p: Pair) -> str:
    return str(p[0]) + p[1]
