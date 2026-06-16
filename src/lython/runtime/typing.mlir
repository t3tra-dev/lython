// The abstract Protocol/iteration tower.
//
// This manifest is the source of truth for the protocol hierarchy: the
// dialect (TableGen) keeps only primitive runtime representations; higher
// type contracts are constructed here and expanded by the frontend protocol
// oracle.
//
// Conventions:
//   - @Protocol is the root of all static type contracts.
//   - ly.typing.protocol marks a py.class as a Protocol contract. Concrete
//     builtin bindings below intentionally omit this marker: they conform to
//     contracts but are not contracts themselves.
//   - ly.typing.params lists a contract's generic parameters; the type
//     variable occurrences in method signatures and base arguments are spelled
//     !py.class<"$T">.
//   - base_names holds the bare base symbols (resolvable, MRO-checked by the
//     py.class verifier); ly.typing.base_args carries the parallel generic
//     instantiation argument lists.
//   - Method declarations are bodyless py.callable.func symbols whose first
//     positional parameter is the nominal self. Overloaded methods use unique
//     MLIR symbol names plus ly.typing.method_name to record the Python name.
//   - !py.object is used as this manifest's erased Any/object parameter.
//   - Concrete builtin types (list, str, tuple, dict, range) are declared as
//     subclasses of the tower; the frontend binds their dialect types to
//     these symbols and substitutes their element types for the parameters.
module {
  py.class @object attributes {ly.typing.abstract} {}

  py.class @Protocol attributes {
      base_names = ["object"], ly.typing.abstract, ly.typing.protocol} {}

  // Callable is represented by !py.protocol<"Callable", [args] -> [results]>.
  // The contract itself owns the callable shape; there is no separate function
  // or function-signature type in the dialect.
  py.class @Callable attributes {
      base_names = ["Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[]], ly.typing.protocol} {}

  // Runtime library placeholders used by stdlib contracts below.
  py.class @AbstractEventLoop attributes {base_names = ["object"], ly.typing.abstract} {}
  py.class @Context attributes {base_names = ["object"], ly.typing.abstract} {}
  py.class @FrameType attributes {base_names = ["object"], ly.typing.abstract} {}
  py.class @GenericAlias attributes {base_names = ["object"], ly.typing.abstract} {}
  py.class @TextIO attributes {base_names = ["object"], ly.typing.abstract} {}

  // Runtime-checkable support protocols from typeshed's typing.pyi.
  py.class @SupportsInt attributes {
      base_names = ["Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[]], ly.typing.protocol}
  {
    py.callable.func @__int__ : !py.protocol<"Callable", [!py.class<"SupportsInt">] -> [!py.int]> attributes {nothrow} {}
  }

  py.class @SupportsFloat attributes {
      base_names = ["Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[]], ly.typing.protocol}
  {
    py.callable.func @__float__ : !py.protocol<"Callable", [!py.class<"SupportsFloat">] -> [!py.float]> attributes {nothrow} {}
  }

  py.class @SupportsComplex attributes {
      base_names = ["Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[]], ly.typing.protocol}
  {
    py.callable.func @__complex__ : !py.protocol<"Callable", [!py.class<"SupportsComplex">] -> [!py.class<"complex">]> attributes {nothrow} {}
  }

  py.class @SupportsBytes attributes {
      base_names = ["Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[]], ly.typing.protocol}
  {
    py.callable.func @__bytes__ : !py.protocol<"Callable", [!py.class<"SupportsBytes">] -> [!py.class<"bytes">]> attributes {nothrow} {}
  }

  py.class @SupportsIndex attributes {
      base_names = ["Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[]], ly.typing.protocol}
  {
    py.callable.func @__index__ : !py.protocol<"Callable", [!py.class<"SupportsIndex">] -> [!py.int]> attributes {nothrow} {}
  }

  py.class @SupportsAbs attributes {
      base_names = ["Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[]], ly.typing.params = ["T"],
      ly.typing.param_variance = ["covariant"],
      ly.typing.protocol}
  {
    py.callable.func @__abs__ : !py.protocol<"Callable", [!py.class<"SupportsAbs">] -> [!py.class<"$T">]> attributes {nothrow} {}
  }

  py.class @SupportsRound attributes {
      base_names = ["Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[]], ly.typing.params = ["T"],
      ly.typing.param_variance = ["covariant"],
      ly.typing.protocol}
  {
    py.callable.func @__round__.none : !py.protocol<"Callable", [!py.class<"SupportsRound">] -> [!py.int]> attributes {ly.typing.method_name = "__round__", nothrow} {}
    py.callable.func @__round__.ndigits : !py.protocol<"Callable", [!py.class<"SupportsRound">, !py.int] -> [!py.class<"$T">]> attributes {ly.typing.method_name = "__round__", nothrow} {}
  }

  py.class @Hashable attributes {
      base_names = ["Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[]], ly.typing.protocol}
  {
    py.callable.func @__hash__ : !py.protocol<"Callable", [!py.class<"Hashable">] -> [!py.int]> attributes {nothrow} {}
  }

  py.class @Iterable attributes {
      base_names = ["Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[]], ly.typing.params = ["T"],
      ly.typing.param_variance = ["covariant"],
      ly.typing.protocol}
  {
    py.callable.func @__iter__ : !py.protocol<"Callable", [!py.class<"Iterable">] -> [!py.protocol<"Iterator", [!py.class<"$T">]>]> attributes {nothrow} {}
  }

  py.class @Iterator attributes {
      base_names = ["Iterable", "Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[!py.class<"$T">], []],
      ly.typing.params = ["T"], ly.typing.param_variance = ["covariant"],
      ly.typing.protocol}
  {
    py.callable.func @__iter__ : !py.protocol<"Callable", [!py.class<"Iterator">] -> [!py.protocol<"Iterator", [!py.class<"$T">]>]> attributes {nothrow} {}
    py.callable.func @__next__ : !py.protocol<"Callable", [!py.class<"Iterator">] -> [!py.class<"$T">]> attributes {maythrow} {}
  }

  py.class @Container attributes {
      base_names = ["Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[]], ly.typing.params = ["T"],
      ly.typing.param_variance = ["contravariant"],
      ly.typing.param_defaults = [!py.object],
      ly.typing.protocol}
  {
    py.callable.func @__contains__ : !py.protocol<"Callable", [!py.class<"Container">, !py.class<"$T">] -> [!py.bool]> attributes {nothrow} {}
  }

  py.class @Sized attributes {
      base_names = ["Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[]], ly.typing.protocol}
  {
    py.callable.func @__len__ : !py.protocol<"Callable", [!py.class<"Sized">] -> [!py.int]> attributes {nothrow} {}
  }

  py.class @Reversible attributes {
      base_names = ["Iterable", "Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[!py.class<"$T">], []],
      ly.typing.params = ["T"], ly.typing.param_variance = ["covariant"],
      ly.typing.protocol}
  {
    py.callable.func @__reversed__ : !py.protocol<"Callable", [!py.class<"Reversible">] -> [!py.protocol<"Iterator", [!py.class<"$T">]>]> attributes {nothrow} {}
  }

  py.class @Collection attributes {
      base_names = ["Sized", "Iterable", "Container", "Protocol"],
      ly.typing.abstract,
      ly.typing.base_args = [[], [!py.class<"$T">], [!py.object], []],
      ly.typing.params = ["T"], ly.typing.param_variance = ["covariant"],
      ly.typing.protocol}
  {
    py.callable.func @__len__ : !py.protocol<"Callable", [!py.class<"Collection">] -> [!py.int]> attributes {nothrow} {}
  }

  py.class @Sequence attributes {
      base_names = ["Reversible", "Collection", "Protocol"],
      ly.typing.abstract,
      ly.typing.base_args = [[!py.class<"$T">], [!py.class<"$T">], []],
      ly.typing.params = ["T"], ly.typing.param_variance = ["covariant"],
      ly.typing.protocol}
  {
    py.callable.func @__getitem__.int : !py.protocol<"Callable", [!py.class<"Sequence">, !py.int] -> [!py.class<"$T">]> attributes {ly.typing.method_name = "__getitem__", maythrow} {}
    py.callable.func @__contains__ : !py.protocol<"Callable", [!py.class<"Sequence">, !py.object] -> [!py.bool]> attributes {nothrow} {}
    py.callable.func @__iter__ : !py.protocol<"Callable", [!py.class<"Sequence">] -> [!py.protocol<"Iterator", [!py.class<"$T">]>]> attributes {nothrow} {}
    py.callable.func @__reversed__ : !py.protocol<"Callable", [!py.class<"Sequence">] -> [!py.protocol<"Iterator", [!py.class<"$T">]>]> attributes {nothrow} {}
    py.callable.func @count : !py.protocol<"Callable", [!py.class<"Sequence">, !py.object] -> [!py.int]> attributes {nothrow} {}
  }

  py.class @MutableSequence attributes {
      base_names = ["Sequence", "Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[!py.class<"$T">], []],
      ly.typing.params = ["T"], ly.typing.param_variance = ["invariant"],
      ly.typing.protocol}
  {
    py.callable.func @insert : !py.protocol<"Callable", [!py.class<"MutableSequence">, !py.int, !py.class<"$T">] -> [!py.none]> attributes {maythrow} {}
    py.callable.func @__setitem__.int : !py.protocol<"Callable", [!py.class<"MutableSequence">, !py.int, !py.class<"$T">] -> [!py.none]> attributes {ly.typing.method_name = "__setitem__", maythrow} {}
    py.callable.func @__delitem__.int : !py.protocol<"Callable", [!py.class<"MutableSequence">, !py.int] -> [!py.none]> attributes {ly.typing.method_name = "__delitem__", maythrow} {}
    py.callable.func @append : !py.protocol<"Callable", [!py.class<"MutableSequence">, !py.class<"$T">] -> [!py.none]> attributes {maythrow} {}
    py.callable.func @clear : !py.protocol<"Callable", [!py.class<"MutableSequence">] -> [!py.none]> attributes {nothrow} {}
    py.callable.func @reverse : !py.protocol<"Callable", [!py.class<"MutableSequence">] -> [!py.none]> attributes {nothrow} {}
    py.callable.func @remove : !py.protocol<"Callable", [!py.class<"MutableSequence">, !py.class<"$T">] -> [!py.none]> attributes {maythrow} {}
  }

  py.class @AbstractSet attributes {
      base_names = ["Collection", "Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[!py.class<"$T">], []],
      ly.typing.params = ["T"], ly.typing.param_variance = ["covariant"],
      ly.typing.protocol}
  {
    py.callable.func @__contains__ : !py.protocol<"Callable", [!py.class<"AbstractSet">, !py.object] -> [!py.bool]> attributes {nothrow} {}
    py.callable.func @_hash : !py.protocol<"Callable", [!py.class<"AbstractSet">] -> [!py.int]> attributes {nothrow} {}
  }

  py.class @MutableSet attributes {
      base_names = ["AbstractSet", "Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[!py.class<"$T">], []],
      ly.typing.params = ["T"], ly.typing.param_variance = ["invariant"],
      ly.typing.protocol}
  {
    py.callable.func @add : !py.protocol<"Callable", [!py.class<"MutableSet">, !py.class<"$T">] -> [!py.none]> attributes {maythrow} {}
    py.callable.func @discard : !py.protocol<"Callable", [!py.class<"MutableSet">, !py.class<"$T">] -> [!py.none]> attributes {maythrow} {}
  }

  py.class @Mapping attributes {
      base_names = ["Collection", "Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[!py.class<"$K">], []],
      ly.typing.params = ["K", "V"],
      ly.typing.param_variance = ["invariant", "covariant"],
      ly.typing.protocol}
  {
    py.callable.func @__getitem__ : !py.protocol<"Callable", [!py.class<"Mapping">, !py.class<"$K">] -> [!py.class<"$V">]> attributes {maythrow} {}
    py.callable.func @get.none : !py.protocol<"Callable", [!py.class<"Mapping">, !py.class<"$K">] -> [!py.union<!py.class<"$V">, !py.none>]> attributes {ly.typing.method_name = "get", nothrow} {}
    py.callable.func @get.default : !py.protocol<"Callable", [!py.class<"Mapping">, !py.class<"$K">, !py.class<"$D">] -> [!py.union<!py.class<"$V">, !py.class<"$D">>]> attributes {ly.typing.method_name = "get", nothrow} {}
    py.callable.func @__contains__ : !py.protocol<"Callable", [!py.class<"Mapping">, !py.object] -> [!py.bool]> attributes {nothrow} {}
  }

  py.class @MutableMapping attributes {
      base_names = ["Mapping", "Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[!py.class<"$K">, !py.class<"$V">], []],
      ly.typing.params = ["K", "V"],
      ly.typing.param_variance = ["invariant", "invariant"],
      ly.typing.protocol}
  {
    py.callable.func @__setitem__ : !py.protocol<"Callable", [!py.class<"MutableMapping">, !py.class<"$K">, !py.class<"$V">] -> [!py.none]> attributes {maythrow} {}
    py.callable.func @__delitem__ : !py.protocol<"Callable", [!py.class<"MutableMapping">, !py.class<"$K">] -> [!py.none]> attributes {maythrow} {}
  }

  py.class @Generator attributes {
      base_names = ["Iterator", "Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[!py.class<"$Y">], []],
      ly.typing.params = ["Y", "S", "R"],
      ly.typing.param_variance = ["covariant", "contravariant", "covariant"],
      ly.typing.param_defaults = [!py.none, !py.none], ly.typing.protocol}
  {
    py.callable.func @__next__ : !py.protocol<"Callable", [!py.class<"Generator">] -> [!py.class<"$Y">]> attributes {maythrow} {}
    py.callable.func @send : !py.protocol<"Callable", [!py.class<"Generator">, !py.class<"$S">] -> [!py.class<"$Y">]> attributes {maythrow} {}
    py.callable.func @throw.type : !py.protocol<"Callable", [!py.class<"Generator">, !py.type<!py.exception>] -> [!py.class<"$Y">]> attributes {ly.typing.method_name = "throw", maythrow} {}
    py.callable.func @throw.type.val : !py.protocol<"Callable", [!py.class<"Generator">, !py.type<!py.exception>, !py.union<!py.exception, !py.object>] -> [!py.class<"$Y">]> attributes {ly.typing.method_name = "throw", maythrow} {}
    py.callable.func @throw.type.val.tb : !py.protocol<"Callable", [!py.class<"Generator">, !py.type<!py.exception>, !py.union<!py.exception, !py.object>, !py.union<!py.traceback, !py.none>] -> [!py.class<"$Y">]> attributes {ly.typing.method_name = "throw", maythrow} {}
    py.callable.func @throw.instance : !py.protocol<"Callable", [!py.class<"Generator">, !py.exception] -> [!py.class<"$Y">]> attributes {ly.typing.method_name = "throw", maythrow} {}
    py.callable.func @throw.instance.val : !py.protocol<"Callable", [!py.class<"Generator">, !py.exception, !py.none] -> [!py.class<"$Y">]> attributes {ly.typing.method_name = "throw", maythrow} {}
    py.callable.func @throw.instance.val.tb : !py.protocol<"Callable", [!py.class<"Generator">, !py.exception, !py.none, !py.union<!py.traceback, !py.none>] -> [!py.class<"$Y">]> attributes {ly.typing.method_name = "throw", maythrow} {}
    py.callable.func @close : !py.protocol<"Callable", [!py.class<"Generator">] -> [!py.none]> attributes {maythrow} {}
    py.callable.func @__iter__ : !py.protocol<"Callable", [!py.class<"Generator">] -> [!py.protocol<"Generator", [!py.class<"$Y">, !py.class<"$S">, !py.class<"$R">]>]> attributes {nothrow} {}
  }

  py.class @Awaitable attributes {
      base_names = ["Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[]], ly.typing.params = ["T"],
      ly.typing.param_variance = ["covariant"],
      ly.typing.protocol}
  {
    py.callable.func @__await__ : !py.protocol<"Callable", [!py.class<"Awaitable">] -> [!py.protocol<"Generator", [!py.object, !py.object, !py.class<"$T">]>]> attributes {maythrow} {}
  }

  py.class @Coroutine attributes {
      base_names = ["Awaitable", "Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[!py.class<"$R">], []],
      ly.typing.params = ["Y", "S", "R"],
      ly.typing.param_variance = ["covariant", "contravariant", "covariant"],
      ly.typing.short_arg_positions = [2],
      ly.typing.short_arg_defaults = [!py.object, !py.object],
      ly.typing.protocol}
  {
    py.callable.func @send : !py.protocol<"Callable", [!py.class<"Coroutine">, !py.class<"$S">] -> [!py.class<"$Y">]> attributes {maythrow} {}
    py.callable.func @throw.type : !py.protocol<"Callable", [!py.class<"Coroutine">, !py.type<!py.exception>] -> [!py.class<"$Y">]> attributes {ly.typing.method_name = "throw", maythrow} {}
    py.callable.func @throw.type.val : !py.protocol<"Callable", [!py.class<"Coroutine">, !py.type<!py.exception>, !py.union<!py.exception, !py.object>] -> [!py.class<"$Y">]> attributes {ly.typing.method_name = "throw", maythrow} {}
    py.callable.func @throw.type.val.tb : !py.protocol<"Callable", [!py.class<"Coroutine">, !py.type<!py.exception>, !py.union<!py.exception, !py.object>, !py.union<!py.traceback, !py.none>] -> [!py.class<"$Y">]> attributes {ly.typing.method_name = "throw", maythrow} {}
    py.callable.func @throw.instance : !py.protocol<"Callable", [!py.class<"Coroutine">, !py.exception] -> [!py.class<"$Y">]> attributes {ly.typing.method_name = "throw", maythrow} {}
    py.callable.func @throw.instance.val : !py.protocol<"Callable", [!py.class<"Coroutine">, !py.exception, !py.none] -> [!py.class<"$Y">]> attributes {ly.typing.method_name = "throw", maythrow} {}
    py.callable.func @throw.instance.val.tb : !py.protocol<"Callable", [!py.class<"Coroutine">, !py.exception, !py.none, !py.union<!py.traceback, !py.none>] -> [!py.class<"$Y">]> attributes {ly.typing.method_name = "throw", maythrow} {}
    py.callable.func @close : !py.protocol<"Callable", [!py.class<"Coroutine">] -> [!py.none]> attributes {maythrow} {}
  }

  // _asyncio.Future/Task contracts. These are Awaitable runtime objects, not
  // native coroutine descriptors.
  py.class @Future attributes {
      base_names = ["Awaitable", "Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[!py.class<"$T">], []],
      ly.typing.params = ["T"], ly.typing.param_variance = ["invariant"],
      ly.typing.protocol}
  {
    py.callable.func @__init__ : !py.protocol<"Callable", [!py.class<"Future">], kwonly = [!py.union<!py.class<"AbstractEventLoop">, !py.none>], kw_names = ["loop"], kw_defaults = [true] -> [!py.none]> attributes {kwonly_names = ["loop"], maythrow} {}
    py.callable.func @__class_getitem__ : !py.protocol<"Callable", [!py.type<!py.class<"Future">>, !py.object], posonly = 2 -> [!py.class<"GenericAlias">]> attributes {ly.typing.classmethod, nothrow} {}
    py.callable.func @__del__ : !py.protocol<"Callable", [!py.class<"Future">] -> [!py.none]> attributes {nothrow} {}
    py.callable.func @get_loop : !py.protocol<"Callable", [!py.class<"Future">] -> [!py.class<"AbstractEventLoop">]> attributes {maythrow} {}
    py.callable.func @add_done_callback : !py.protocol<"Callable", [!py.class<"Future">, !py.protocol<"Callable", [!py.self] -> [!py.object]>], posonly = 2, kwonly = [!py.union<!py.class<"Context">, !py.none>], kw_names = ["context"], kw_defaults = [true] -> [!py.none]> attributes {kwonly_names = ["context"], maythrow} {}
    py.callable.func @cancel : !py.protocol<"Callable", [!py.class<"Future">, !py.union<!py.object, !py.none>], arg_defaults = [false, true] -> [!py.bool]> attributes {maythrow} {}
    py.callable.func @cancelled : !py.protocol<"Callable", [!py.class<"Future">] -> [!py.bool]> attributes {nothrow} {}
    py.callable.func @done : !py.protocol<"Callable", [!py.class<"Future">] -> [!py.bool]> attributes {nothrow} {}
    py.callable.func @result : !py.protocol<"Callable", [!py.class<"Future">] -> [!py.class<"$T">]> attributes {maythrow} {}
    py.callable.func @exception : !py.protocol<"Callable", [!py.class<"Future">] -> [!py.union<!py.exception, !py.none>]> attributes {maythrow} {}
    py.callable.func @remove_done_callback : !py.protocol<"Callable", [!py.class<"Future">, !py.protocol<"Callable", [!py.self] -> [!py.object]>], posonly = 2 -> [!py.int]> attributes {maythrow} {}
    py.callable.func @set_result : !py.protocol<"Callable", [!py.class<"Future">, !py.class<"$T">], posonly = 2 -> [!py.none]> attributes {maythrow} {}
    py.callable.func @set_exception : !py.protocol<"Callable", [!py.class<"Future">, !py.union<!py.type<!py.exception>, !py.exception>], posonly = 2 -> [!py.none]> attributes {maythrow} {}
    py.callable.func @__iter__ : !py.protocol<"Callable", [!py.class<"Future">] -> [!py.protocol<"Generator", [!py.object, !py.none, !py.class<"$T">]>]> attributes {maythrow} {}
    py.callable.func @__await__ : !py.protocol<"Callable", [!py.class<"Future">] -> [!py.protocol<"Generator", [!py.object, !py.none, !py.class<"$T">]>]> attributes {maythrow} {}
  }

  py.class @Task attributes {
      base_names = ["Future", "Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[!py.class<"$T">], []],
      ly.typing.params = ["T"], ly.typing.param_variance = ["covariant"],
      ly.typing.protocol}
  {
    py.callable.func @__init__ : !py.protocol<"Callable", [!py.class<"Task">, !py.protocol<"Coroutine", [!py.object, !py.object, !py.class<"$T">]>], posonly = 2, kwonly = [!py.union<!py.class<"AbstractEventLoop">, !py.none>, !py.union<!py.str, !py.none>, !py.union<!py.class<"Context">, !py.none>, !py.bool], kw_names = ["loop", "name", "context", "eager_start"], kw_defaults = [true, true, true, true] -> [!py.none]> attributes {kwonly_names = ["loop", "name", "context", "eager_start"], maythrow} {}
    py.callable.func @__class_getitem__ : !py.protocol<"Callable", [!py.type<!py.class<"Task">>, !py.object], posonly = 2 -> [!py.class<"GenericAlias">]> attributes {ly.typing.classmethod, nothrow} {}
    py.callable.func @get_coro : !py.protocol<"Callable", [!py.class<"Task">] -> [!py.union<!py.protocol<"Coroutine", [!py.object, !py.object, !py.class<"$T">]>, !py.none>]> attributes {nothrow} {}
    py.callable.func @get_name : !py.protocol<"Callable", [!py.class<"Task">] -> [!py.str]> attributes {nothrow} {}
    py.callable.func @set_name : !py.protocol<"Callable", [!py.class<"Task">, !py.object], posonly = 2 -> [!py.none]> attributes {nothrow} {}
    py.callable.func @get_context : !py.protocol<"Callable", [!py.class<"Task">] -> [!py.class<"Context">]> attributes {nothrow} {}
    py.callable.func @get_stack : !py.protocol<"Callable", [!py.class<"Task">], kwonly = [!py.union<!py.int, !py.none>], kw_names = ["limit"], kw_defaults = [true] -> [!py.list<!py.class<"FrameType">>]> attributes {kwonly_names = ["limit"], maythrow} {}
    py.callable.func @print_stack : !py.protocol<"Callable", [!py.class<"Task">], kwonly = [!py.union<!py.int, !py.none>, !py.union<!py.class<"TextIO">, !py.none>], kw_names = ["limit", "file"], kw_defaults = [true, true] -> [!py.none]> attributes {kwonly_names = ["limit", "file"], maythrow} {}
    py.callable.func @cancelling : !py.protocol<"Callable", [!py.class<"Task">] -> [!py.int]> attributes {nothrow} {}
    py.callable.func @uncancel : !py.protocol<"Callable", [!py.class<"Task">] -> [!py.int]> attributes {nothrow} {}
  }

  py.class @AsyncIterable attributes {
      base_names = ["Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[]], ly.typing.params = ["T"],
      ly.typing.param_variance = ["covariant"],
      ly.typing.protocol}
  {
    py.callable.func @__aiter__ : !py.protocol<"Callable", [!py.class<"AsyncIterable">] -> [!py.protocol<"AsyncIterator", [!py.class<"$T">]>]> attributes {nothrow} {}
  }

  py.class @AsyncIterator attributes {
      base_names = ["AsyncIterable", "Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[!py.class<"$T">], []],
      ly.typing.params = ["T"], ly.typing.param_variance = ["covariant"],
      ly.typing.protocol}
  {
    py.callable.func @__anext__ : !py.protocol<"Callable", [!py.class<"AsyncIterator">] -> [!py.protocol<"Awaitable", [!py.class<"$T">]>]> attributes {maythrow} {}
    py.callable.func @__aiter__ : !py.protocol<"Callable", [!py.class<"AsyncIterator">] -> [!py.protocol<"AsyncIterator", [!py.class<"$T">]>]> attributes {nothrow} {}
  }

  py.class @AsyncGenerator attributes {
      base_names = ["AsyncIterator", "Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[!py.class<"$Y">], []],
      ly.typing.params = ["Y", "S"],
      ly.typing.param_variance = ["covariant", "contravariant"],
      ly.typing.param_defaults = [!py.none], ly.typing.protocol}
  {
    py.callable.func @__anext__ : !py.protocol<"Callable", [!py.class<"AsyncGenerator">] -> [!py.protocol<"Coroutine", [!py.object, !py.object, !py.class<"$Y">]>]> attributes {maythrow} {}
    py.callable.func @asend : !py.protocol<"Callable", [!py.class<"AsyncGenerator">, !py.class<"$S">] -> [!py.protocol<"Coroutine", [!py.object, !py.object, !py.class<"$Y">]>]> attributes {maythrow} {}
    py.callable.func @athrow.type : !py.protocol<"Callable", [!py.class<"AsyncGenerator">, !py.type<!py.exception>] -> [!py.protocol<"Coroutine", [!py.object, !py.object, !py.class<"$Y">]>]> attributes {ly.typing.method_name = "athrow", maythrow} {}
    py.callable.func @athrow.type.val : !py.protocol<"Callable", [!py.class<"AsyncGenerator">, !py.type<!py.exception>, !py.union<!py.exception, !py.object>] -> [!py.protocol<"Coroutine", [!py.object, !py.object, !py.class<"$Y">]>]> attributes {ly.typing.method_name = "athrow", maythrow} {}
    py.callable.func @athrow.type.val.tb : !py.protocol<"Callable", [!py.class<"AsyncGenerator">, !py.type<!py.exception>, !py.union<!py.exception, !py.object>, !py.union<!py.traceback, !py.none>] -> [!py.protocol<"Coroutine", [!py.object, !py.object, !py.class<"$Y">]>]> attributes {ly.typing.method_name = "athrow", maythrow} {}
    py.callable.func @athrow.instance : !py.protocol<"Callable", [!py.class<"AsyncGenerator">, !py.exception] -> [!py.protocol<"Coroutine", [!py.object, !py.object, !py.class<"$Y">]>]> attributes {ly.typing.method_name = "athrow", maythrow} {}
    py.callable.func @athrow.instance.val : !py.protocol<"Callable", [!py.class<"AsyncGenerator">, !py.exception, !py.none] -> [!py.protocol<"Coroutine", [!py.object, !py.object, !py.class<"$Y">]>]> attributes {ly.typing.method_name = "athrow", maythrow} {}
    py.callable.func @athrow.instance.val.tb : !py.protocol<"Callable", [!py.class<"AsyncGenerator">, !py.exception, !py.none, !py.union<!py.traceback, !py.none>] -> [!py.protocol<"Coroutine", [!py.object, !py.object, !py.class<"$Y">]>]> attributes {ly.typing.method_name = "athrow", maythrow} {}
    py.callable.func @aclose : !py.protocol<"Callable", [!py.class<"AsyncGenerator">] -> [!py.protocol<"Coroutine", [!py.object, !py.object, !py.none]>]> attributes {maythrow} {}
  }

  py.class @ContextManager attributes {
      base_names = ["Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[]], ly.typing.params = ["T", "ExitT"],
      ly.typing.param_variance = ["covariant", "covariant"],
      ly.typing.param_defaults = [!py.union<!py.bool, !py.none>],
      ly.typing.protocol}
  {
    py.callable.func @__enter__ : !py.protocol<"Callable", [!py.class<"ContextManager">] -> [!py.class<"$T">]> attributes {maythrow} {}
    py.callable.func @__exit__ : !py.protocol<"Callable", [!py.class<"ContextManager">, !py.union<!py.type<!py.exception>, !py.none>, !py.union<!py.exception, !py.none>, !py.union<!py.traceback, !py.none>] -> [!py.class<"$ExitT">]> attributes {maythrow} {}
  }

  py.class @AsyncContextManager attributes {
      base_names = ["Protocol"], ly.typing.abstract,
      ly.typing.base_args = [[]], ly.typing.params = ["T", "ExitT"],
      ly.typing.param_variance = ["covariant", "covariant"],
      ly.typing.param_defaults = [!py.union<!py.bool, !py.none>],
      ly.typing.protocol}
  {
    py.callable.func @__aenter__ : !py.protocol<"Callable", [!py.class<"AsyncContextManager">] -> [!py.protocol<"Awaitable", [!py.class<"$T">]>]> attributes {maythrow} {}
    py.callable.func @__aexit__ : !py.protocol<"Callable", [!py.class<"AsyncContextManager">, !py.union<!py.type<!py.exception>, !py.none>, !py.union<!py.exception, !py.none>, !py.union<!py.traceback, !py.none>] -> [!py.protocol<"Awaitable", [!py.class<"$ExitT">]>]> attributes {maythrow} {}
  }

  // Concrete builtin conformance. The frontend binds !py.list<E> to @list
  // with T := E, !py.str to @str, homogeneous !py.tuple<E> to @tuple with
  // T := E, and !py.dict<K, V> to @dict with K/V.
  py.class @list attributes {
      base_names = ["MutableSequence"],
      ly.typing.base_args = [[!py.class<"$T">]], ly.typing.params = ["T"]} {}

  py.class @tuple attributes {
      base_names = ["Sequence"],
      ly.typing.base_args = [[!py.class<"$T">]], ly.typing.params = ["T"]} {}

  py.class @int attributes {
      base_names = ["SupportsInt", "SupportsFloat", "SupportsComplex",
                    "SupportsIndex", "SupportsAbs", "SupportsRound",
                    "Hashable"],
      ly.typing.base_args = [[], [], [], [], [!py.int], [!py.int], []],
      ly.typing.final} {}

  py.class @bool attributes {base_names = ["int"], ly.typing.final} {}

  py.class @float attributes {
      base_names = ["SupportsFloat", "SupportsComplex", "SupportsAbs",
                    "SupportsRound", "Hashable"],
      ly.typing.base_args = [[], [], [!py.float], [!py.float], []],
      ly.typing.final} {}

  py.class @str attributes {
      base_names = ["Sequence", "Hashable"],
      ly.typing.base_args = [[!py.str], []]} {}

  py.class @dict attributes {
      base_names = ["MutableMapping"],
      ly.typing.base_args = [[!py.class<"$K">, !py.class<"$V">]],
      ly.typing.params = ["K", "V"]} {}

  py.class @bytes attributes {
      base_names = ["Sequence", "SupportsBytes", "Hashable"],
      ly.typing.base_args = [[!py.int], [], []],
      ly.typing.final} {}

  py.class @complex attributes {
      base_names = ["SupportsComplex", "SupportsAbs", "Hashable"],
      ly.typing.base_args = [[], [!py.float], []],
      ly.typing.final} {}

  // class range(Sequence[int]) - final.
  py.class @range attributes {
      base_names = ["Sequence", "Hashable"],
      ly.typing.base_args = [[!py.int], []],
      ly.typing.final} {}
}
