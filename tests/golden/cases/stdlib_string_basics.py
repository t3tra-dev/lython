# What: string module constants, capwords, and Template substitute paths
# (constants/identity, $name and ${name} substitution, $$ escaping) match
# CPython 3.14.
import string
from string import Template, capwords

print(string.whitespace == " \t\n\r\v\f")
print(string.ascii_letters)
print(string.hexdigits)
print(string.octdigits)
print(string.punctuation)
print(len(string.printable))
print(capwords(" aBc  dEf "))
print(capwords("x,,y", ","))
print(string.capwords("hello world"))

t = Template("$who likes $$ and ${what}x")
print(t.template)
print(t.substitute({"who": "tim", "what": "ham"}))
print(t.safe_substitute({"who": "tim"}))
