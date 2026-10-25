# A base class whose methods call the hooks a subclass overrides, for
# tests/python/test_a_method_reads_its_modules_names.py. Each method has a local
# spelled like one of that module's globals.


class Template:
    def render(self) -> str:
        out = "base"
        return self.hook() + "/" + out

    def hook(self) -> str:
        return ""

    def total(self) -> int:
        tag = 10
        return self.measure() + tag

    def measure(self) -> int:
        return 0
