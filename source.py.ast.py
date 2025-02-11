from ast import (
    Assign,
    Call,
    Constant,
    Dict,
    Expr,
    List,
    Load,
    Module,
    Name,
    Store,
    Subscript,
)

Module(
    body=[
        Expr(
            value=Call(
                func=Name(id="print", ctx=Load()),
                args=[Constant(value="Hello, world!")],
                keywords=[],
            )
        ),
        Assign(
            targets=[Name(id="listvar", ctx=Store())],
            value=List(
                elts=[
                    Constant(value=0),
                    Constant(value=1),
                    Constant(value=2),
                    Constant(value="a"),
                    Constant(value="b"),
                    Constant(value="c"),
                ],
                ctx=Load(),
            ),
        ),
        Assign(
            targets=[Name(id="dictvar", ctx=Store())],
            value=Dict(
                keys=[
                    Constant(value="key1"),
                    Constant(value="key2"),
                    Constant(value="key3"),
                ],
                values=[Constant(value=1), Constant(value=2), Constant(value=3)],
            ),
        ),
        Expr(
            value=Call(
                func=Name(id="print", ctx=Load()),
                args=[
                    Subscript(
                        value=Name(id="listvar", ctx=Load()),
                        slice=Constant(value=3),
                        ctx=Load(),
                    )
                ],
                keywords=[],
            )
        ),
        Expr(
            value=Call(
                func=Name(id="print", ctx=Load()),
                args=[
                    Call(
                        func=Name(id="str", ctx=Load()),
                        args=[
                            Subscript(
                                value=Name(id="dictvar", ctx=Load()),
                                slice=Constant(value="key1"),
                                ctx=Load(),
                            )
                        ],
                        keywords=[],
                    )
                ],
                keywords=[],
            )
        ),
    ],
    type_ignores=[],
)
