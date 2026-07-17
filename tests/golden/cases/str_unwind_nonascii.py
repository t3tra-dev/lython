# Cross-track (Wave 0 integration): owned adaptive-width strings (latin-1,
# UCS-2, and UCS-4 payloads) are built on paths that unwind through
# raise/except/finally. Compiling at all proves the cleanup: the
# affine-ownership verifier accepts no leaked Own on the exception edges.

def shout(tag: str, flag: int) -> str:
    label = "métier-" + tag
    if flag > 0:
        raise ValueError("mauvais drapeau: " + tag)
    return label


def guard(flag: int) -> None:
    kept = "गार्ड🛡" + str(flag)
    try:
        piece = "célébration…" + shout("Ω" + str(flag), flag)
        print(piece)
    except ValueError as err:
        print("attrapé: " + str(err))
    finally:
        print("fin " + kept)


def loop_unwind() -> None:
    for i in range(3):
        try:
            piece = "π" + str(i) + "…😀"
            shout(piece, i)
            print("ok " + piece)
        except ValueError as err:
            print("saute " + str(i) + ": " + str(err))


guard(0)
guard(1)
loop_unwind()
