class TestCase:
    failureException: type[BaseException] = AssertionError
    longMessage: bool = True
    maxDiff: int | None = 640

    def describe(self) -> str:
        return (self.failureException.__name__ + " " + str(self.longMessage)
                + " " + str(self.maxDiff))
