# Helper for wb_an_imported_class_attribute_built_by_an_expression.
W = 20


class C:
    lit = 5
    expr = W // 4

    def read_lit(self) -> int:
        return self.lit

    def read_expr(self) -> int:
        return self.expr
