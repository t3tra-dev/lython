def risky(n: int) -> str:
    try:
        if n < 0:
            raise ValueError("bad " + str(n))
        return str(n)
    except ValueError as err:
        return "caught " + str(err)


print(risky(-1))
print(risky(5))
