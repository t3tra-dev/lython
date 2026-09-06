# What: a class attribute whose TYPE is a class from another file. Whether an
# attribute gets a storage slot was decided by the name's shape -- a dotted
# contract is manifest, and only a handful of manifest containers are slotted
# -- so an imported class's `mod.Big` fell out and the read went to the
# constant channel, which has no arm for an instance. It did not refuse: the
# lowering pipeline failed. The same class written in one file is read fine.
#
# The subclass redeclaring the attribute is here because that read goes through
# the dispatcher, which reads through the CLASS -- so it needs the slot too.
from a_module_of_operator_classes import Big


class Holder:
    default: Big = Big(3)

    def show(self) -> int:
        return self.default.n


class Sub(Holder):
    default: Big = Big(9)


holders: list[Holder] = [Holder(), Sub()]
print("through a method", [h.show() for h in holders])
print("through the class", Holder.default.n, Sub.default.n)
print("compared", Sub.default > Holder.default)
