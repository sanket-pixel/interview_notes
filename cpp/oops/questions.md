### Object-Oriented Programming & Internals Q&A

**Q1. Explain how C++ implements runtime polymorphism (dynamic dispatch) under the hood.**
C++ achieves runtime polymorphism using Virtual Tables (`vtable`) and Virtual Pointers (`vptr`).
When a class contains at least one `virtual` function, the compiler creates a static array of function pointers called the `vtable` for that specific class. Furthermore, the compiler secretly adds a pointer, the `vptr`, into the memory layout of every instantiated object of that class. This `vptr` points to the class's `vtable`.
When a virtual function is called through a base pointer, the program dereferences the object's `vptr` at runtime, finds the correct function address in the `vtable`, and executes it.

**Q2. What is the size of an empty class in C++, and what happens to its size if you add a single virtual function?**
An empty class is exactly **1 byte** in size. C++ requires every instantiated object to have a unique memory address, so the compiler inserts a dummy byte.
If you add a single virtual function (or virtual destructor) to that empty class, the size jumps to **8 bytes** (on a 64-bit system). The 1-byte dummy is removed, but the compiler adds an 8-byte `vptr` inside the object to point to the class's `vtable`.

**Q3. *[Trap]* What is "Object Slicing", and how do you prevent it?**
Object slicing occurs when a Derived class object is assigned to a Base class object *by value*. The compiler only copies the Base part of the object, completely discarding (slicing off) the Derived-specific data and overriding any polymorphic behavior.

```cpp
class Base { public: virtual void print() { cout << "Base"; } };
class Derived : public Base { public: void print() override { cout << "Derived"; } };

void process(Base b) { b.print(); } // Passed by value!

Derived d;
process(d); // Outputs "Base". The 'Derived' part was sliced off.

```

**Fix:** Always pass polymorphic objects by pointer (`Base*`) or reference (`Base&`).

**Q4. What is the "Diamond Problem" in C++, and how does virtual inheritance solve it?**
The Diamond Problem occurs with multiple inheritance. If Class B and Class C both inherit from Class A, and Class D inherits from both B and C, Class D will contain two distinct copies of Class A in memory. Calling a method from A through D creates an ambiguity error because the compiler doesn't know which A to use.

**Solution:** `virtual` inheritance (`class B : virtual public A`). This tells the compiler that the inheritance tree should only share a single, unified instance of the base Class A, regardless of how many paths lead back to it.

**Q5. *[Trap]* Can you call a virtual function from a constructor or destructor? What happens?**
Yes, you can call it, but **it will not behave polymorphically**.
When a Base constructor is running, the Derived part of the object does not exist yet. The `vptr` currently points only to the Base's `vtable`. If you call a virtual function inside the Base constructor, it will reliably call the Base's implementation of that function, never the Derived class's implementation. The same logic applies in reverse for destructors.

**Q6. Contrast `static_cast` and `dynamic_cast`. When must you use `dynamic_cast`?**

* **`static_cast`**: Performed at *compile-time*. It is used for safe, predictable conversions (like `float` to `int`, or casting a generic `void*` back to a known type). It does no runtime checks.
* **`dynamic_cast`**: Performed at *run-time*. It is strictly used to safely cast a Base pointer/reference down to a Derived pointer/reference. It checks the object's `vtable` (via RTTI - Run-Time Type Information) to ensure the object is actually of the target type. If the cast is invalid, it returns `nullptr` (for pointers) or throws an exception (for references).

**Q7. *[Trap]* Why might `dynamic_cast` fail to compile even if the class hierarchy is correct?**
`dynamic_cast` relies on Run-Time Type Information (RTTI), which is stored in the `vtable`. If your Base class does not have at least one `virtual` function (typically a virtual destructor), it does not have a `vtable`. The compiler will throw an error because it has no way to verify the object's actual type at runtime.

**Q8. What does `reinterpret_cast` do, and what are its dangers?**
`reinterpret_cast` instructs the compiler to take a sequence of bits in memory and treat it as a completely different type, without doing any actual conversion or checking. For example, casting a raw memory address (`intptr_t`) into a `MyStruct*`. It is highly dangerous, breaks type safety, and is strictly reserved for low-level systems programming (like mapping hardware registers or custom allocators).

**Q9. How do you create an interface in C++?**
C++ does not have an `interface` keyword. Interfaces are created using **Abstract Classes**. An abstract class is any class that contains at least one **pure virtual function** (a virtual function assigned to `0`).

```cpp
class ILog {
public:
    virtual ~ILog() = default;
    virtual void write(const std::string& msg) = 0; // Pure virtual function
};

```

You cannot instantiate an abstract class. Any derived class must implement `write()`, or it too becomes abstract.

**Q10. *[Trap]* What is "Name Hiding" (or Shadowing) in inheritance?**
If a Base class has an overloaded function (e.g., `void load(int)` and `void load(string)`), and a Derived class overrides *only one* of those signatures, the compiler will hide the other signature.

```cpp
class Base { 
public: 
    void process(int x); 
    void process(string s); 
};
class Derived : public Base { 
public: 
    void process(int x); // Hides process(string)
};

```

Calling `derived_obj.process("test")` will cause a compile error. To fix this, use the `using` keyword in the derived class: `using Base::process;`.

**Q11. Why is the `override` keyword considered mandatory in modern C++?**
Without `override`, if you make a slight typo in a derived class's method signature (e.g., `void update(float)` instead of `void update(double)`), the compiler won't override the base function; it will silently create a brand new, separate function.
Appending `override` (`void update(double) override;`) forces the compiler to verify that a virtual function with that exact signature actually exists in the base class. If it doesn't, it throws a compile error, preventing subtle runtime bugs.

**Q12. What is the difference between Compile-Time and Run-Time polymorphism?**

* **Compile-Time (Static):** The function to be invoked is determined during compilation. This includes Function Overloading and Templates. It is extremely fast because there is zero overhead at runtime.
* **Run-Time (Dynamic):** The function to be invoked is determined while the program is executing. This is achieved via virtual functions and inheritance. It incurs a slight performance penalty due to the `vtable` pointer lookup.
