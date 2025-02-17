class Person {
  var name = "Alice"

  var age: Int {
    return 10
  }

  func printName() -> Void {
    print(self.name)
  }
}

let person1 = Person()

print(person1.name)
print(person1.age)
person1.printName()
