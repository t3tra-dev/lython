class User:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return "User(name=" + self.name + ")"


class Group:
    def __init__(self, members: list[User]) -> None:
        self.members = members

    def add_member(self, user: User) -> None:
        self.members.append(user)

    def remove_member(self, user: User) -> None:
        self.members.remove(user)


g1 = Group([User("Alice"), User("Bob")])
g1.add_member(User("Charlie"))
print(g1.members)
g1.remove_member(g1.members[0])
print(g1.members)
