_MAX = 80
_PLACE = 12
_MIN = 5
_DIFF = _MAX - (_PLACE + _MIN + _PLACE + _MIN)
_HALF = _DIFF // 2
_REM = -_DIFF % 7
_SHIFT = 1 << 4
_MASK = _SHIFT - 1 | 32
_PREFIX = "un" + "ittest"
_RULE = "=" * 5


def show() -> str:
    return " ".join([str(_DIFF), str(_HALF), str(_REM), str(_SHIFT), str(_MASK),
                     _PREFIX, _RULE])
