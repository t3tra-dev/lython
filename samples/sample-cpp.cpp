#include <iostream>
#include <string>

class Person {
public:
    std::string name = "Alice";

    int age() const {
        return 10;
    }

    void printName() const {
        puts(name.c_str());
    }
};

int main() {
    Person person1;

    puts(person1.name.c_str());
    printf("%d\n", person1.age());
    person1.printName();

    return 0;
}
