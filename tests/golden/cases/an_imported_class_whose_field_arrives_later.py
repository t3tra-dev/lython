# An imported class whose field is seeded `None` and given its value by a
# method, and whose class attribute is a module constant of its own module.
# Both walks asked the type system about the class by its SOURCE spelling --
# `Record` -- while an imported class is `<module>.Record` there, so every
# question answered nothing: the field stayed NoneType and the attribute stayed
# an unmaterializable "ref". The same file compiled as the main module.
import a_module_of_linked_records as m

root = m.Record("a")
user = m.User("u")
root.attach(user)
print(user.chain(), root.chain())

store = m.Store()
print(store.add(root), store.add(user), m.Store.capacity)
for _ in range(3):
    store.add(m.Record("x"))
print(len(store.items))
