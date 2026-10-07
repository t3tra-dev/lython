// `memoryview` -- CPython's Objects/memoryobject.c.
//
// Signature source (1:1 correspondence target):
//   https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
//
// The contract only: no memoryview method has a runtime implementation yet.

module attributes {
  ly.typing.manifest
} {
  py.class @memoryview attributes {base_names = ["Sequence"],
                                  ly.typing.base_args = [[!py.contract<"builtins.int">]]} {}
}
