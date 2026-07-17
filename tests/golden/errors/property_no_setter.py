class C:
    @property
    def x(self) -> int:
        return 1


c = C()
c.x = 5
