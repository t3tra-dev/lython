# What: the imported half of `a_def_chosen_by_a_platform_test`. It is a case of
# its own because the golden runner globs every .py here -- running it alone
# declares one constant and one function and prints nothing, which is what its
# empty expectation says.
import sys

if sys.platform == "win32":
    SEP = "\\"

    def joiner(a: str, b: str) -> str:
        return a + "\\" + b

else:
    SEP = "/"

    def joiner(a: str, b: str) -> str:
        return a + "/" + b
