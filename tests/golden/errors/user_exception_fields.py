class E(Exception):
    def __init__(self, code: int) -> None:
        self.code = code


raise E(3)
