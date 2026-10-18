# WHAT: a class in an imported module whose methods are decorated -- by its
# own module's decorators -- is that class in the importer: the decorators
# ran when the module was imported, a call runs what they returned on the
# instance it was made on, and a subclass here inherits and overrides them.
#
# WHY THIS IS RUN: what a decorated method is was decided by running its
# decorator, and which `helper` its body reaches is the instance's class at
# run time.
from a_decorated_method_in_an_imported_class_lib import Base, registry


class Sub(Base):
    def helper(self) -> str:
        return "sub-helper:" + self.tag

    def plain(self) -> None:
        print("sub plain")
        super().plain()


print(registry)
b = Base("b")
b.run()
b.setup()
s = Sub("s")
s.run()
s.plain()
for name in ["run", "setup", "plain"]:
    getattr(s, name)()
print(len(registry))
