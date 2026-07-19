def probe(x: int) -> None:
    try:
        if x == 0:
            raise TypeError("t-error")
        raise ValueError("v-error")
    except (TypeError, ValueError) as e:
        print("caught:", e)
probe(0)
probe(1)
