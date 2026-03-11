#include <iostream>
#include <string>
#include <vector>

using namespace std;

// ==========================================
// Exercise 1: Object Slicing
// SUMMARY: Passing by reference (Animal&) or pointer prevents Object Slicing, keeping the derived object intact.
// This preserves the object's vptr, ensuring dynamic dispatch correctly calls the overridden Dog::speak() method.
// ==========================================
class Animal {
public:
    virtual string speak() const { return "Animal noise"; }
    virtual ~Animal() = default;
};

class Dog : public Animal {
    string breed = "Husky";
public:
    string speak() const override { return "Woof! I am a " + breed; }
};

// TODO: This function currently causes Object Slicing.
// Fix the function signature so that it correctly prints the Dog's speak().
void makeAnimalSpeak(Animal& a) {
    cout << a.speak() << endl;
}

void testObjectSlicing() {
    cout << "\n--- Executing Exercise 1 ---\n";
    Dog myDog;
    makeAnimalSpeak(myDog); // Should print "Woof! I am a Husky"
}


// ==========================================
// Exercise 2: VTable Traps & The 'override' Keyword
// SUMMARY: To safely override a virtual function, the function signatures must match exactly.
// Always append the 'override' keyword in the derived class so the compiler throws an error if you make a typo, preventing accidental name hiding.
// ==========================================
class BaseSystem {
public:
    virtual void process(float data) { cout << "Base processing float\n"; }
    virtual ~BaseSystem() = default;
};

class DerivedSystem : public BaseSystem {
public:
    // TODO: The developer meant to override process(), but made a typo in the parameter type.
    // 1. Add the modern C++ keyword that would have caught this bug at compile-time.
    // 2. Fix the parameter type so it actually overrides the Base function.
    void process(float data) override { cout << "Derived processing double\n"; }

};

void testVTableOverride() {
    cout << "\n--- Executing Exercise 2 ---\n";
    BaseSystem* sys = new DerivedSystem();
    sys->process(5.0f); // Should print "Derived processing double" (once fixed)
    delete sys;
}


// ==========================================
// Exercise 3: The Diamond Problem
// SUMMARY: The Diamond Problem causes ambiguous duplicate base classes in multiple inheritance.
// Using 'virtual public' inheritance solves this by replacing duplicate data with hidden pointers (vbptrs) that route to a single, shared instance of the base class.
// ==========================================
// TODO: The UIElement class currently has TWO copies of the 'Component' base class.
// Fix the inheritance declarations so that UIElement only has ONE shared Component.

class Component {
public:
    int id = 42;
    Component() { cout << "Component constructed\n"; }
};

class Renderable : public virtual Component {}; // Fix me
class Clickable : public virtual Component {};  // Fix me

class UIElement : public Renderable, public Clickable {
public:
    void printId() {
        // This line will not compile right now due to ambiguity!
        // Uncomment it once you fix the inheritance above.
        cout << "UIElement ID: " << id << endl;
    }
};

void testDiamondProblem() {
    cout << "\n--- Executing Exercise 3 ---\n";
    UIElement btn;
    btn.printId();
}


// ==========================================
// Exercise 4: RTTI and dynamic_cast
// SUMMARY: dynamic_cast safely downcasts base pointers to derived pointers by checking the vtable's
// Run-Time Type Information (RTTI). If the object is not of the target type, it safely returns nullptr.
// ==========================================
class Entity {
public:
    virtual ~Entity() = default; // Enables RTTI
};

class Player : public Entity {
public:
    void attack() { cout << "Player attacks!\n"; }
};

class Enemy : public Entity {
public:
    void defend() { cout << "Enemy defends!\n"; }
};

void testCasting() {
    cout << "\n--- Executing Exercise 4 ---\n";

    Entity* e1 = new Player();
    Entity* e2 = new Enemy();

    // TODO: We want to call attack() on e1, but we only have an Entity pointer.
    // 1. Safely cast e1 to a Player pointer using the correct C++ cast.
    Player* p1 = dynamic_cast<Player*>(e1);
    // TODO: Try to safely cast e2 to a Player pointer.
    Enemy* p2 = dynamic_cast<Enemy*>(e2);

    // 2. Add an if-statement to check if the cast succeeded before calling attack().
    if (p1!=nullptr && p2!=nullptr) {
        p1->attack();
        p2->defend();
    }
    // Prove that it safely fails (returns nullptr) rather than crashing.
    Enemy* wrongtype_entity = dynamic_cast<Enemy*>(e1);
    if (wrongtype_entity==nullptr) {
        std::cout << "Wrong type exiting gracefully";
    }
    delete e1;
    delete e2;
}


// ==========================================
int main() {
    testObjectSlicing();
    testVTableOverride();
    testDiamondProblem();
    testCasting();

    return 0;
}