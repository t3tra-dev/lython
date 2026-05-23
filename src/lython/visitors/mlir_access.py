from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import cast

from ..mlir import ir


def _attribute(obj: object | None, name: str) -> object | None:
    if obj is None:
        return None
    try:
        return object.__getattribute__(obj, name)
    except AttributeError:
        return None


def value_owner(value: ir.Value) -> object | None:
    return _attribute(value, "owner")


def value_operation(value: ir.Value) -> ir.Operation | None:
    owner = value_owner(value)
    if owner is None:
        return None
    opview = _attribute(owner, "opview")
    if opview is not None:
        operation = _attribute(opview, "operation")
        if operation is not None:
            return cast(ir.Operation, operation)
    if _attribute(owner, "operands") is None and _attribute(owner, "name") is None:
        return None
    return cast(ir.Operation, owner)


def block_owner(block: ir.Block | None) -> object | None:
    return _attribute(block, "owner")


def owner_parent(owner: object | None) -> object | None:
    parent = _attribute(owner, "owner")
    if parent is not None:
        return parent
    return _attribute(owner, "parent")


def op_name(op: ir.Operation) -> str:
    name = _attribute(op, "name")
    return "" if name is None else str(name)


def op_operands(op: ir.Operation) -> list[ir.Value]:
    operands = _attribute(op, "operands")
    if not isinstance(operands, Iterable):
        return []
    values = cast(Iterable[object], operands)
    return [cast(ir.Value, operand) for operand in values]


def op_attributes(op: ir.Operation) -> MutableMapping[str, object]:
    attributes = _attribute(op, "attributes")
    if isinstance(attributes, MutableMapping):
        return cast(MutableMapping[str, object], attributes)
    return {}


def _int_value(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    try:
        return int(cast(int | str, value))
    except (TypeError, ValueError):
        return None


def op_operand_segment_sizes(op: ir.Operation) -> list[int]:
    segment_attr = op_attributes(op).get("operandSegmentSizes")
    if not isinstance(segment_attr, Iterable):
        return []
    sizes: list[int] = []
    for value in cast(Iterable[object], segment_attr):
        size = _int_value(value)
        if size is None:
            return []
        sizes.append(size)
    return sizes


def value_argument_index(value: ir.Value) -> int | None:
    direct_arg = _attribute(value, "arg_number")
    if direct_arg is not None:
        return _int_value(direct_arg)

    owner = value_owner(value)
    arguments = _attribute(owner, "arguments")
    if isinstance(arguments, Iterable):
        try:
            return list(cast(Iterable[object], arguments)).index(value)
        except ValueError:
            return None

    owner_arg = _attribute(owner, "arg_number")
    if owner_arg is None:
        return None
    return _int_value(owner_arg)
