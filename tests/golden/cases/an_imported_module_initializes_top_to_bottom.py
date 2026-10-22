# WHAT: an imported module's top-level instance, built below the class it
# instantiates, reads that class's attribute; a list built from those
# instances reads their fields.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the module's statements
# run when the program starts, and whether the attribute was written before
# the read is a fact of that run.
#
# ⛔ An imported module's globals were all written before any of its class
# attributes, so `DEFAULT_STYLE = Style()` raised "module global
# 'Style.default_format' referenced before assignment"; CPython runs the
# module top to bottom.
import a_module_of_classes_read_while_it_initializes as styles

print(styles.DEFAULT_STYLE.fmt)
print(styles.BRACE_STYLE.fmt)
print(styles.NAMES)
