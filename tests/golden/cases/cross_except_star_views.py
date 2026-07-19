class AppError(Exception):
    pass


class NetError(AppError):
    pass


def run() -> None:
    try:
        raise ExceptionGroup(
            "mixed",
            [NetError("timeout", 30), ValueError("bad", 1, 2), AppError("boom")],
        )
    except* NetError as eg:
        print("net:", repr(eg), len(eg.exceptions))
    except* AppError as eg:
        print("app:", repr(eg), len(eg.exceptions))
    except* ValueError as eg:
        print("val:", repr(eg), len(eg.exceptions))
    counts = {"app": 0, "net": 0, "val": 0}
    counts["app"] = counts["app"] + 1
    counts["net"] = counts["net"] + 1
    counts["val"] = counts["val"] + 1
    total = 0
    for k in counts.keys():
        print(k, counts[k])
    for n in counts.values():
        total += n
    print("total:", total)
    try:
        raise AppError("solo", 1, 2, 3)
    except AppError as e:
        print("args:", e.args)


run()
