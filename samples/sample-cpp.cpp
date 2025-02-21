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

    std::cout << person1.name << std::endl;
    std::cout << person1.age() << std::endl;
    person1.printName();

    return 0;
}
