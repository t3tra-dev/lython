# OPEN. An IMPORTED class attribute whose initializer is an EXPRESSION cannot be
# read at all:
#
#     # a_module_of_computed_attributes.py
#     W = 20
#     class C:
#         lit = 5
#         expr = W // 4
#
#     import a_module_of_computed_attributes as m
#     print(m.C().read_expr())      # class m.C has no field 'expr'
#
# The literal one beside it (`lit = 5`) reads fine, and the same class in the
# MAIN module reads both -- so the boundary is the defect and the shape is not.
#
# ⛔ WHY. A main-module class attribute whose widened type has cell storage
# becomes SLOT-backed, and a slot's initializer is any expression. An imported
# class keeps its attributes on the CONSTANT channel, which records the AST
# shape (`kind = "binop"`, with a "ref" to W inside it) and can materialize
# only `constant.int` / `.str` / `.float` / `.bool` / `.none`. Nothing folds
# the binop, so the read falls through to a FIELD lookup and reports a field
# that was never declared.
#
# ⛔ AND SLOTTING IMPORTED CLASSES IS NOT THE CHEAP FIX. The slot emits a
# module-level store, and a runtime-internal lib module may not run
# module-level code -- the note in EmitterClasses.cpp records
# `stackguard_support.py` failing to build for exactly that when `_fields_`
# was slotted. Any repair has to distinguish a user module from a lib one, or
# fold the constant instead.
#
# Measured 2026-09-04. The dispatcher half of the same boundary IS fixed
# (cases/a_class_attribute_a_subclass_redeclares_across_modules).
import a_module_of_computed_attributes as m

print(m.C().read_lit())
print(m.C().read_expr())
