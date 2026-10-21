# WHAT: a class this module declares below another module's class, as a value,
# stored into that module's `type[Base] | None` field -- through its
# constructor and by assignment -- and read back.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the value was written into
# the union's lanes without its tag, so the field read back as None -- and as
# whatever the old tag said where it had held a class -- with nothing refused.
import a_class_value_stored_into_another_modules_field_lib as lib


class Sub(lib.Base):
    pass


print(lib.Prog(Sub).runner_name(), lib.Prog().runner_name())
p = lib.Prog()
p.runner = Sub
print(p.runner is None, p.runner_name())
q = lib.Prog(lib.Base)
q.runner = Sub
print(q.runner_name())
