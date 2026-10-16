# Helper for a_traceback_frame_names_its_own_file: a method that raises, so the
# importer's call inlines it. Run on its own it only defines the class.
class Table:
    def __init__(self) -> None:
        self.n = 0

    def get(self, key: str) -> str:
        if self.n == 0:
            raise KeyError(key)
        return key
