#include <stdio.h>

int fib(int n) {
    if (n <= 1) {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

int main() {
    int result = fib(35);
    printf("%d\n", result);
    return 0;
}
