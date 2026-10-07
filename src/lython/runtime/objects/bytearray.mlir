// `bytearray` -- CPython's Objects/bytearrayobject.c.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
//
// The contract only: no bytearray method has a runtime implementation yet.

module attributes {
  ly.typing.manifest
} {
  py.class @bytearray attributes {base_names = ["MutableSequence"],
                                 ly.typing.base_args = [[!py.contract<"builtins.int">]],
                                 ly.typing.final} {}
}
