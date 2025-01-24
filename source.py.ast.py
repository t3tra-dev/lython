from ast import Call, Constant, Expr, Load, Module, Name, keyword

Module(
    body=[
        Expr(
            value=Call(
                func=Name(id="print", ctx=Load()),
                args=[Constant(value="Hello, world!")],
                keywords=[],
            )
        ),
        Expr(
            value=Call(
                func=Name(id="print", ctx=Load()),
                args=[Constant(value="hoge"), Constant(value="huga")],
                keywords=[
                    keyword(arg="sep", value=Constant(value="")),
                    keyword(arg="end", value=Constant(value="\n")),
                    keyword(arg="file", value=Constant(value=None)),
                    keyword(arg="flush", value=Constant(value=False)),
                ],
            )
        ),
    ],
    type_ignores=[],
)
