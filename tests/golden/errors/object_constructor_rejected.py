# What: `object()` is refused with a located, actionable diagnostic instead of
#   the manifest-lookup failure it used to produce ("runtime manifest has no
#   builtins.object.__new__", which reads as "object has no constructor").
#   The refusal says what is missing (a bare object has no runtime allocation
#   of its own -- the class-number conflict with None it used to name is
#   gone) and the workaround (declare a class).
sentinel = object()
print(sentinel is sentinel)
