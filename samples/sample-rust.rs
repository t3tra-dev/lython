struct Person {
    name: String,
}

impl Person {
    fn new() -> Self {
        Person {
            name: String::from("Alice"),
        }
    }

    fn age(&self) -> i32 {
        10
    }

    fn print_name(&self) {
        println!("{}", self.name);
    }
}

fn main() {
    let person1 = Person::new();

    println!("{}", person1.name);
    println!("{}", person1.age());
    person1.print_name();
}
