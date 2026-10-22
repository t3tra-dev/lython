# The imported half of a_function_reads_its_modules_name_and_file.

# Bound at this module's top level: this module's, not the importer's.
OWN_FILE: str = __file__.rsplit("/", 1)[-1]
OWN_NAME: str = __name__


def report() -> str:
    return __name__ + " " + __file__.rsplit("/", 1)[-1]
