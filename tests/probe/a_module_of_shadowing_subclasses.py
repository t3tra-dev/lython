# Helper for wb_an_imported_class_attribute_a_subclass_redeclares.
WIDTH = 80


class Shape:
    kind = "shape"

    def describe(self) -> str:
        return self.kind


class Square(Shape):
    kind = "square"
