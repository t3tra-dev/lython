# Helper for leak_imported_class_attr_{small,big}: the class whose attribute
# cells the loop rebinds.
class Slot:
    tag: str = ""
    hits: int = 0
