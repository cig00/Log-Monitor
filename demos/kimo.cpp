#include <iostream>
#include <vector>

class Kimo {
public:
    Kimo() {
        std::cout << "Kimo object created!" << std::endl;
    }

    void greet() {
        std::cout << "Hello from Kimo!" << std::endl;
    }
};

int main() {
    Kimo kimo;
    kimo.greet();
    return 0;
}
