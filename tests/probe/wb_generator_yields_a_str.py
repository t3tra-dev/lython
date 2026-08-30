# A generator that yields a STR is refused outright:
#
#   source generator next lowering currently supports yields whose runtime
#   value is a single lane, and '!py.contract<"builtins.str">' has 2
#
# A str is a header plus a payload descriptor, and the resume path assumes one
# lane. Every line-filtering generator is written this way -- the reduction is
# the shape `config.py` parsers use.
#
# ⛔ NOT THE SAME DEFECT as wb_generator_yields_what_it_keeps.py even though
# both are about what crosses a yield: this one is a REFUSAL in the state
# machine's lane model, that one is a refcount the lanes do carry. A two-lane
# yield has to be modelled before its ownership can be got wrong.
def lines(text: str):
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            yield stripped


for line in lines("a\n\nb\n"):
    print(line)
