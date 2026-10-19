class TestCase:
    pass


def main(classes: list[type[TestCase]]) -> None:
    print([c.__name__ for c in classes])
