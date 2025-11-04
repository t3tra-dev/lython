from __future__ import annotations

from typing import Callable

from lython.mlir import ir
from lython.mlir.dialects import _lython_ops_gen as py_ops
from lython.mlir.dialects import arith as arith_ops
from lython.mlir.dialects import func as func_ops


def _py(ctx: ir.Context, text: str) -> ir.Type:
    return ir.Type.parse(text, ctx)


def build_module(ctx: ir.Context) -> ir.Module:
    with ir.Location.unknown(ctx):
        module = ir.Module.create()

        # Precompute frequently used types.
        py_int = _py(ctx, "!py.int")
        py_object = _py(ctx, "!py.object")
        py_str = _py(ctx, "!py.str")
        tuple_int = _py(ctx, "!py.tuple<!py.int>")
        tuple_empty = _py(ctx, "!py.tuple<>")
        dict_str_object = _py(ctx, "!py.dict<!py.str, !py.object>")
        py_none = _py(ctx, "!py.none")
        py_func_sig_unary = _py(ctx, "!py.funcsig<[!py.int] -> [!py.int]>")
        py_func_type_unary = _py(ctx, "!py.func<!py.funcsig<[!py.int] -> [!py.int]>>")
        prim_func_type_i32 = _py(ctx, "!py.prim.func<(i32, i32) -> (i32)>")
        py_class_demo = _py(ctx, '!py.class<"DemoCallable">')
        i32 = ir.IntegerType.get_signless(32)

        # ------------------------------------------------------------------
        # py.func / py.return : identity function.
        # ------------------------------------------------------------------
        with ir.InsertionPoint(module.body):
            id_func = py_ops.FuncOp(
                "py_id",
                ir.TypeAttr.get(py_func_sig_unary),
                arg_names=ir.ArrayAttr.get([ir.StringAttr.get("x", ctx)]),  # type: ignore
            )
        with (
            ir.Location.unknown(ctx),
            ir.InsertionPoint(id_func.regions[0].blocks.append(py_int)),
        ):
            py_ops.ReturnOp([id_func.regions[0].blocks[0].arguments[0]])

        # ------------------------------------------------------------------
        # py.tuple.create / py.tuple.empty / py.call.vector.
        # ------------------------------------------------------------------
        with ir.InsertionPoint(module.body):
            use_id = py_ops.FuncOp(
                "py_use_id",
                ir.TypeAttr.get(py_func_sig_unary),
                arg_names=ir.ArrayAttr.get([ir.StringAttr.get("value", ctx)]),  # type: ignore
            )

        block = use_id.regions[0].blocks.append(py_int)
        with ir.Location.unknown(ctx), ir.InsertionPoint(block):
            fn_object = py_ops.FuncObjectOp(
                py_func_type_unary,
                ir.FlatSymbolRefAttr.get("py_id", ctx),
            ).result
            posargs = py_ops.TupleCreateOp(tuple_int, [block.arguments[0]]).result
            empty_names = py_ops.TupleEmptyOp(tuple_empty).result
            empty_values = py_ops.TupleEmptyOp(tuple_empty).result
            call_result = py_ops.CallVectorOp(
                [py_int], fn_object, posargs, empty_names, empty_values
            ).results_[0]
            py_ops.ReturnOp([call_result])

        # ------------------------------------------------------------------
        # py.call with kwargs=None / kwargs=dict
        # ------------------------------------------------------------------
        def build_py_call_user(name: str, kwargs_builder: Callable[[], ir.Value]):
            with ir.InsertionPoint(module.body):
                func = py_ops.FuncOp(
                    name,
                    ir.TypeAttr.get(py_func_sig_unary),
                    arg_names=ir.ArrayAttr.get([ir.StringAttr.get("value", ctx)]),  # type: ignore
                )
            block_local = func.regions[0].blocks.append(py_int)
            with ir.Location.unknown(ctx), ir.InsertionPoint(block_local):
                fn_obj = py_ops.FuncObjectOp(
                    py_func_type_unary,
                    ir.FlatSymbolRefAttr.get("py_id", ctx),
                ).result
                posargs_local = py_ops.TupleCreateOp(
                    tuple_int, [block_local.arguments[0]]
                ).result
                kwargs_value = kwargs_builder()
                call = py_ops.CallOp(
                    [py_int], fn_obj, posargs_local, kwargs_value
                ).results_[0]
                py_ops.ReturnOp([call])

        build_py_call_user("py_use_call_none", lambda: py_ops.NoneOp(py_none).result)
        build_py_call_user(
            "py_use_call_dict", lambda: py_ops.DictEmptyOp(dict_str_object).result
        )

        # ------------------------------------------------------------------
        # py.make_function, dict.insert, str.constant, upcast
        # ------------------------------------------------------------------
        with ir.InsertionPoint(module.body):
            make_fn_user = py_ops.FuncOp(
                "py_use_make_function",
                ir.TypeAttr.get(py_func_sig_unary),
                arg_names=ir.ArrayAttr.get([ir.StringAttr.get("value", ctx)]),  # type: ignore
            )

        block = make_fn_user.regions[0].blocks.append(py_int)
        with ir.Location.unknown(ctx), ir.InsertionPoint(block):
            defaults = py_ops.TupleEmptyOp(tuple_empty).result
            closure = py_ops.TupleEmptyOp(tuple_empty).result
            none_value = py_ops.NoneOp(py_none).result
            upcast_none = py_ops.UpcastOp(py_object, none_value).result
            kwdefaults_val = py_ops.DictInsertOp(
                dict_str_object,
                py_ops.DictEmptyOp(dict_str_object).result,
                py_ops.StrConstantOp(py_str, ir.StringAttr.get("default", ctx)).result,
                upcast_none,
            ).result
            annotations = py_ops.DictInsertOp(
                dict_str_object,
                py_ops.DictEmptyOp(dict_str_object).result,
                py_ops.StrConstantOp(py_str, ir.StringAttr.get("returns", ctx)).result,
                upcast_none,
            ).result
            module_name = py_ops.StrConstantOp(
                py_str, ir.StringAttr.get("demo_module", ctx)
            ).result
            made_func = py_ops.MakeFunctionOp(
                py_func_type_unary,
                ir.FlatSymbolRefAttr.get("py_id", ctx),
                defaults=defaults,
                kwdefaults=kwdefaults_val,
                closure=closure,
                annotations=annotations,
                module=module_name,
            ).result
            posargs = py_ops.TupleCreateOp(tuple_int, [block.arguments[0]]).result
            empty_kwargs = py_ops.DictEmptyOp(dict_str_object).result
            call_made = py_ops.CallOp(
                [py_int], made_func, posargs, empty_kwargs
            ).results_[0]
            py_ops.ReturnOp([call_made])

        # ------------------------------------------------------------------
        # py.num.add verification.
        # ------------------------------------------------------------------
        with ir.InsertionPoint(module.body):
            add_func = py_ops.FuncOp(
                "py_add",
                ir.TypeAttr.get(
                    _py(ctx, "!py.funcsig<[!py.int, !py.int] -> [!py.int]>")
                ),
                arg_names=ir.ArrayAttr.get(  # type: ignore
                    [ir.StringAttr.get("lhs", ctx), ir.StringAttr.get("rhs", ctx)]
                ),
            )
        add_block = add_func.regions[0].blocks.append(py_int, py_int)
        with ir.Location.unknown(ctx), ir.InsertionPoint(add_block):
            num_add = ir.Operation.create(
                "py.num.add",
                results=[py_int],
                operands=[add_block.arguments[0], add_block.arguments[1]],
            )
            summed = num_add.result
            py_ops.ReturnOp([summed])

        # ------------------------------------------------------------------
        # py.class / py.func nested symbols.
        # ------------------------------------------------------------------
        with ir.InsertionPoint(module.body), ir.Location.unknown(ctx):
            class_op = py_ops.ClassOp("DemoCallable")
        class_block = class_op.body.blocks.append()
        method_sig = _py(
            ctx, '!py.funcsig<[!py.class<"DemoCallable">, !py.int] -> [!py.int]>'
        )
        with ir.Location.unknown(ctx), ir.InsertionPoint(class_block):
            method = py_ops.FuncOp(
                "__call__",
                ir.TypeAttr.get(method_sig),
                arg_names=ir.ArrayAttr.get(  # type: ignore
                    [ir.StringAttr.get("self", ctx), ir.StringAttr.get("value", ctx)]
                ),
            )
        with (
            ir.Location.unknown(ctx),
            ir.InsertionPoint(method.regions[0].blocks.append(py_class_demo, py_int)),
        ):
            py_ops.ReturnOp([method.regions[0].blocks[0].arguments[1]])

        # ------------------------------------------------------------------
        # func.func + py.make_native + py.native_call
        # ------------------------------------------------------------------
        func_type_i32x2 = ir.FunctionType.get([i32, i32], [i32])
        with ir.InsertionPoint(module.body):
            native_add = func_ops.FuncOp("native_add", func_type_i32x2)
        native_entry = native_add.add_entry_block()
        with ir.InsertionPoint(native_entry):
            added = arith_ops.AddIOp(
                native_entry.arguments[0], native_entry.arguments[1]
            )
            func_ops.ReturnOp([added.result])

        with ir.InsertionPoint(module.body):
            invoke_native = func_ops.FuncOp("invoke_native", func_type_i32x2)
        inv_entry = invoke_native.add_entry_block()
        with ir.InsertionPoint(inv_entry):
            native_handle = py_ops.MakeNativeOp(
                prim_func_type_i32, ir.FlatSymbolRefAttr.get("native_add", ctx)
            ).result
            native_call = py_ops.NativeCallOp(
                [i32], native_handle, list(inv_entry.arguments)
            ).results_[0]
            func_ops.ReturnOp([native_call])

        # ------------------------------------------------------------------
        # py.cast.from_prim / py.cast.to_prim / py.upcast / dict.insert path.
        # ------------------------------------------------------------------
        with ir.InsertionPoint(module.body):
            cast_bridge = func_ops.FuncOp(
                "cast_bridge", ir.FunctionType.get([i32], [i32])
            )
        cast_entry = cast_bridge.add_entry_block()
        with ir.InsertionPoint(cast_entry):
            boxed = py_ops.CastFromPrimOp(py_int, cast_entry.arguments[0]).result
            upcasted = py_ops.UpcastOp(py_object, boxed).result
            dict_with_obj = py_ops.DictInsertOp(
                dict_str_object,
                py_ops.DictEmptyOp(dict_str_object).result,
                py_ops.StrConstantOp(py_str, ir.StringAttr.get("boxed", ctx)).result,
                upcasted,
            ).result
            _ = dict_with_obj
            unboxed = py_ops.CastToPrimOp(
                i32, boxed, ir.StringAttr.get("exact", ctx)
            ).result
            func_ops.ReturnOp([unboxed])

        return module


def main() -> None:
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = build_module(ctx)

    is_valid = module.operation.verify()
    print("Module is valid" if is_valid else "Module is invalid")
    print(module)


if __name__ == "__main__":
    main()
