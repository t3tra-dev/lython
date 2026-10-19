# An imported module's docstring, and its functions' and classes', read
# through the module, through the names `from ... import` bound, and through
# a subclass written in the importer (whose docstring is its own: None). All
# were refused; `lib.__doc__` was "module has no attribute '__doc__' that
# resolves statically".
# Why run it: the texts are the imported module's, and only the run shows
# which module's each one is.
import a_modules_docstring_is_its_own_lib as lib
from a_modules_docstring_is_its_own_lib import Base, helper


class Mine(Base):
    def run(self) -> None:
        pass


print(repr(lib.__doc__))
print(repr(lib.helper.__doc__), repr(helper.__doc__), repr(lib.undocumented.__doc__))
print(repr(Base.__doc__), repr(lib.Base.run.__doc__), repr(Mine.__doc__))
objects: list[Base] = [Base(), Mine()]
print([o.run.__doc__ for o in objects], [o.__doc__ for o in objects])
