def cleanup(s: str) -> str:
    try:
        return s + "!"
    finally:
        print("cleanup done")


def guard(x: int) -> None:
    try:
        if x < 0:
            raise ValueError("negative")
        print("positive")
    except ValueError:
        print("caught negative")


print(cleanup("hello"))
guard(5)
guard(-1)
