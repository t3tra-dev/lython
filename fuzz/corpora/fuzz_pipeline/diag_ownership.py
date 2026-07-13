class Resource:
    def __init__(self, name: str) -> None:
        self.name = name


def use(flag: int) -> str:
    r = Resource("res" + str(flag))
    try:
        if flag < 0:
            raise ValueError("bad " + r.name)
        return r.name
    except ValueError as err:
        return "caught " + str(err)
    finally:
        print("cleanup")


print(use(-1))
