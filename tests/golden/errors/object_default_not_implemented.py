# What: a user class inherits ALL of builtins.object's declared methods through
#   its protocol-table base, but only six of the eleven have a default behind
#   them. The other five are refused at the call site -- located, naming the
#   class and listing what IS inherited -- rather than left to surface from the
#   lowering as "runtime manifest has no C.__setattr__ method", which is the
#   same points-away-from-the-defect wording the contract audit was about.
#   (Attribute assignment itself works: `c.v = 2`. Only the dunder spelling of
#   it is unimplemented.)
class C:
    def __init__(self) -> None:
        self.v: int = 1


c = C()
c.__setattr__("v", 2)
print(c.v)
