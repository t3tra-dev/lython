# WHAT: A base class whose method calls a function private to this module
#   and a method subclasses override. Run alone it prints nothing.
# KIND: helper module of ut_bug_inherited_method_loses_its_module_globals
def _decorate(text: str) -> str:
    return "<" + text + ">"


class Base:
    def name(self) -> str:
        return "base"

    def run(self) -> str:
        return _decorate(self.name())
