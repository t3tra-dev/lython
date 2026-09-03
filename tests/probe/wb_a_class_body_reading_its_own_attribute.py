# OPEN. A class attribute defined in terms of an EARLIER attribute of the same
# class body is refused, and the refusal arrives from the lowering:
#
#     class C:
#         n = 3
#         m = n            # unsupported static class attribute expression
#     class D:
#         n: int = 3
#         m: int = n       # emit error: unresolved name 'n'
#
# CPython prints 3 for both. The module-level spelling of the same question --
# `N = 3` outside the class, `m = N` inside it -- was the twin of this and IS
# fixed (cases/a_class_attribute_from_a_module_constant): a plain module
# assignment now binds a symbol, so `inferExpr` can type the attribute and the
# attribute gets slot storage. The class-body name has no such binding to make.
#
# ⛔ WHY IT IS NOT THE SAME ONE-LINE REPAIR. Typing it is the easy half: bind
# each attribute's name as `collectStaticClassAssignments` walks the body in
# order and `inferExpr` resolves the later ones. EMITTING it is the half that
# needs a mechanism. `emitClassAttrInitializers` emits the initializer with
# `emitExprExpected`, and a bare `n` there is not a local, not a module global
# and not a builtin -- it is a slot on the class being defined, which only
# `C.n` reaches. So the initializer has to be walked with every reference to an
# earlier attribute rewritten to that attribute read, and there is no AST
# rewriter in the emitter to walk it with (`grep -rn 'cloneNode\|rewriteNames'
# src/lython/emitter` finds nothing; `synth::` builds nodes, it does not
# transform them).
#
# ⛔ AND THE REWRITE IS NOT UNCONDITIONAL. `C.n` reads the attribute's FINAL
# value; CPython's class body reads the value at that point. The two differ
# for a body that rebinds -- `n = 1; m = n; n = 2` prints m == 1 -- so the
# rewrite needs the same bound-once restriction `collectModuleGlobals` uses,
# or the shape has to be refused where it would disagree.
#
# Measured 2026-09-03: `n = 3` / `m = n` dies in the lowering, `n: int = 3` /
# `m: int = n` at emit. Reading the attribute through the class (`m = C.n`) is
# not available either -- the class does not exist yet inside its own body.
class C:
    n = 3
    m = n


print(C.n, C.m)
