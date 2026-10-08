# WHAT: the last line of an uncaught exception's traceback names a class the
#   program defines as CPython does: bare for __main__ ("Boom: boom", not
#   "__main__.Boom"), where the class's own name is the qualified one.
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the traceback printer reads
#   the name off the raised object's type object at run time.
class Boom(Exception):
    pass


raise Boom("boom")
