# Helper for an_imported_float_constant.
RATIO = 1.5
TINY = 0.001
BIG = 1e20
WHOLE = 3.0
NEG_F = -2.25
NEG_I = -7
LIMIT = 4
NAME = "app"
FLAG = False


class Settings:
    ratio = RATIO
    limit = LIMIT

    def summary(self) -> str:
        return str(self.ratio) + "/" + str(self.limit)
