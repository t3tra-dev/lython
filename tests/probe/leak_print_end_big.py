# WHAT: `print(end=...)` builds a concatenated string and writes it through
# sys.stdout.write, whose int result nothing reads. Both have to be released.
i = 0
while i < 4000:
    print("y" * 2048, end="")
    print("z" * 2048, end="\n")
    i += 1
