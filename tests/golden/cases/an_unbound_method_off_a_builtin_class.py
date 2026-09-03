# A method read off the CLASS rather than off an instance. Used as a value it
# has to become a callable of one argument; called directly it has to take its
# receiver from the first argument, with the rest staying where they are.
# A classmethod reached the same way must NOT shift: its first parameter is not
# the class.

words = ["delta", "Alpha", "charlie", "Bravo"]

print(sorted(words, key=str.lower))
print(list(map(str.upper, words)))

shout = str.upper
print(shout("quiet"), shout(words[0]))

print(str.strip("  padded  "))
print(str.strip("--edged--", "-"))
print(str.replace("banana", "an", "AN"))
print(str.count("banana", "a"), str.startswith("banana", "ban"))

print(sorted([7, 1, 30], key=int.bit_length))
print(bytes.fromhex("6c79"))

print(sorted(map(str.title, words)))
