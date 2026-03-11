#include <iostream>
#include <memory>
#include <vector>

using namespace std;

// ==========================================
// Exercise 1: Smart Pointers & Circular References
// ==========================================
// TASK:
// 1. Create a parent-child relationship using smart pointers.
// 2. The Parent should own multiple Children.
// 3. The Child needs a reference back to the Parent.
// 4. Instantiate them in testSmartPointers() and prove (via prints in destructors)
//    that there are no memory leaks when the function exits.
// SUMMARY: A shared_ptr cycle prevents reference counts from reaching 0, causing leaks.
// Always break the cycle by using std::weak_ptr for the back-reference (Child to Parent).

struct Child; // Forward declaration

struct Parent {
    std::vector<std::shared_ptr<Child>> c;
    Parent() { cout << "Parent created\n"; }
    ~Parent() { cout << "Parent destroyed\n"; }
};

struct Child {
    std::weak_ptr<Parent> p;
    // std::shared_ptr<Parent> p; would lead to memory leak
    Child() { cout << "Child created\n"; }
    ~Child() { cout << "Child destroyed\n"; }
};

void testSmartPointers() {
    cout << "\n--- Executing Exercise 1 ---\n";
    std::shared_ptr<Parent> p = std::make_shared<Parent>();
    std::shared_ptr<Child> c = std::make_shared<Child>();
    p->c.push_back(c);
    c->p = p;
}

//
// ==========================================
// Exercise 2: Struct Padding and Alignment
// ==========================================
// TASK:
// 1. Observe 'SloppyStruct'.
// 2. Create 'PackedStruct' with the exact same variables, but reorder them
//    to minimize memory padding.
// 3. Print the sizeof() both structs in testStructPadding() to prove your optimization.
// SUMMARY: Always order struct members from largest type size to smallest.
// This minimizes compiler-inserted padding required for hardware alignment,
// saving significant memory when allocating arrays/vectors of structs.

struct SloppyStruct {
    bool is_active;    // 1 byte
    double score;      // 8 bytes
    int id;            // 4 bytes
    char grade;        // 1 byte
};

struct PackedStruct {
    double score;      // 8 bytes
    int id;            // 4 bytes
    bool is_active;    // 1 byte
    char grade;        // 1 byte
};

void testStructPadding() {
    cout << "\n--- Executing Exercise 2 ---\n";
    SloppyStruct sloppy{false, 1.2,1,'A'};
    PackedStruct packed{1.2, 1,true,'B'};

    std::cout << sizeof(sloppy) << std::endl;
    std::cout << sizeof(packed) << std::endl;

}


// ==========================================
// Exercise 3: Inheritance & The Virtual Destructor Trap
// ==========================================
// TASK:
// 1. The current setup causes a memory leak if deleted through a Base pointer.
// 2. Fix the Base/Derived classes so memory is properly freed.
// 3. Write the instantiation and deletion logic in testInheritance().
//
// SUMMARY: Deleting a Derived object through a Base pointer causes a memory leak
// unless the Base class destructor is marked 'virtual'. Without 'virtual', the
// compiler only calls the Base destructor, leaving the Derived members in memory.
class Base {
public:
    Base() { cout << "Base allocated\n"; }
    // TODO: Fix the destructor
    virtual ~Base() { cout << "Base freed\n"; }
};

class Derived : public Base {
    int* data;
public:
    Derived() {
        data = new int[100];
        cout << "Derived allocated\n";
    }
    ~Derived() {
        delete[] data;
        cout << "Derived freed\n";
    }
};

void testInheritance() {
    cout << "\n--- Executing Exercise 3 ---\n";
    // TODO: Allocate a Derived object using a Base pointer.
    Base* b = new Derived();
    // TODO: Delete the object and verify both destructors are called.
    delete b;
}


// ==========================================
// Exercise 4: Manual Memory & The Rule of Three
// ==========================================
// TASK:
// 1. 'Buffer' manages a raw dynamic array.
// 2. It currently crashes on a double-free if copied.
// 3. Implement the copy constructor and copy assignment operator to fix it.

class Buffer {
    int* array;
    size_t size;
public:
    Buffer(size_t s) : size(s) {
        array = new int[size];
        cout << "Buffer allocated\n";
    }

    ~Buffer() {
        delete[] array;
        cout << "Buffer freed\n";
    }
    Buffer(const Buffer& other) : size(other.size){
        array = new int[other.size];
        std::copy(other.array, other.array + size, array);
    }
    Buffer& operator=(const Buffer& other) {
        if (this==&other)
            return *this;
        delete[] array;
        size = other.size;
        array = new int[other.size];

        std::copy(other.array, other.array + other.size, array);
        return *this;
    }
};

void testRuleOfThree() {
    cout << "\n--- Executing Exercise 4 ---\n";
    Buffer b1(10);
    Buffer b2 = b1;
}


// ==========================================
int main() {
    testSmartPointers();
    testStructPadding();
    testInheritance();
    testRuleOfThree();

    return 0;
}