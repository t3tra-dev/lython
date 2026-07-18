class BoomError(Exception):
    pass


class LoudError(BoomError):
    def __init__(self, where: str) -> None:
        super().__init__("boom at " + where)


raise LoudError("startup")
