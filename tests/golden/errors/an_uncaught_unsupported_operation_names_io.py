# WHAT: io.UnsupportedOperation is named by the module CPython defines it in,
#   io, though the runtime declares it under _io.
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the traceback printer reads
#   the name off the raised object's type object at run time.
import io

raise io.UnsupportedOperation("not here")
