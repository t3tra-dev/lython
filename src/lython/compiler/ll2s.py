import os


def ll2s(ll_file: str) -> None:
    """
    This function is used to convert the low level code to the high level code.
    """
    try:
        os.system(f"llc {ll_file}.ll -o {ll_file}.s")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
