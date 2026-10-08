# WHAT: a field a base declares (`name: str`) and a subclass answers without
# storing it -- a class attribute, a property -- reads the subclass's answer,
# through the subclass and through the base. Both used to read the empty
# field slot: a segfault where CPython prints the attribute.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the slot is empty only at
# run time; the failure was a crash of the compiled program.

class HasName:
    name: str

class Robot(HasName):
    name = "R2"

def show(h: HasName) -> str:
    return h.name
print(show(Robot()))
print(Robot().name)


class Labelled:
    name: str

class Droid(Labelled):
    @property
    def name(self) -> str:
        return "R2"

def label_of(h: Labelled) -> str:
    return h.name
print(label_of(Droid()))
