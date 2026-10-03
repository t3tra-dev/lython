# What: the stub is checked where it is read, not believed. JSON.stringify is
# declared to return a string and returns undefined for undefined; a global
# the host does not have (node has no `document`) is JavaScript's
# ReferenceError, not an undefined to carry on with. (A result the program
# never reads is not checked, so the value is bound.)
#
# Why run: the disagreement exists only in the host's answer.
from js import JSON, document

try:
    text = JSON.stringify(None)
except TypeError as error:
    print("TypeError:", error)
try:
    print(document.title)
except RuntimeError as error:
    print("RuntimeError:", error)
