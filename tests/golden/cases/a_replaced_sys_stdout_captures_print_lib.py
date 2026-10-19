import io
import sys


def run_captured(n: int) -> str:
    buffer = io.StringIO()
    saved = sys.stdout
    sys.stdout = buffer
    try:
        for i in range(n):
            print("line", i)
    finally:
        sys.stdout = saved
    return buffer.getvalue()
