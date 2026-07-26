# What: `object()` is refused with a located, actionable diagnostic instead of
#   the manifest-lookup failure it used to produce ("runtime manifest has no
#   builtins.object.__new__", which reads as "object has no constructor").
#   The refusal names the representation conflict (class id 0 is the None
#   singleton's) and the workaround (declare a class).
sentinel = object()
print(sentinel is sentinel)
