### Memory Management & Object Lifecycle Q&A

**Q1. What is RAII, and how does it guarantee resource safety in C++?**
RAII stands for Resource Acquisition Is Initialization. It is a programming idiom where resource management (allocating memory, opening files, locking mutexes) is tied to the lifespan of a local stack object. You acquire the resource in the object's constructor and release it in the destructor. Because C++ guarantees that destructors for stack-allocated objects are called when they go out of scope (even if an exception is thrown or a premature `return` occurs), RAII prevents resource leaks deterministically.

**Q2. Explain the difference in memory overhead between `std::unique_ptr` and `std::shared_ptr`.**
A `std::unique_ptr` has zero overhead compared to a raw pointer. It is exactly the size of a raw pointer (typically 8 bytes on a 64-bit system), and all ownership semantics are resolved by the compiler at compile-time.
A `std::shared_ptr` has significant overhead. It contains two pointers under the hood: one pointing to the actual object, and another pointing to a dynamically allocated "Control Block" on the heap. This control block stores the reference count, the weak count, and custom deleters, requiring atomic operations for thread-safe reference counting.

**Q3. How does `std::weak_ptr` solve the circular reference problem?**
A circular reference occurs when two objects managed by `std::shared_ptr` point to each other. Their reference counts will never drop to zero, causing a memory leak. A `std::weak_ptr` observes an object managed by a `shared_ptr` but does not increment its primary reference count.

```cpp
struct B; // Forward declaration
struct A { std::shared_ptr<B> b_ptr; };
struct B { std::weak_ptr<A> a_ptr; }; // weak_ptr breaks the cycle

int main() {
    auto a = std::make_shared<A>();
    auto b = std::make_shared<B>();
    a->b_ptr = b;
    b->a_ptr = a; 
    // Both will be safely destroyed when main ends.
}
```

**Q4. *[Trap]* Is it safe to create a `std::shared_ptr` from `this` inside a class method?**
No, it is highly dangerous. If you do `std::shared_ptr<MyClass>(this)`, it creates a brand new control block. If another `shared_ptr` already owns this object, you now have two independent control blocks for the same memory, leading to a double-free crash. The correct way is to inherit from `std::enable_shared_from_this<MyClass>` and call `shared_from_this()`, which safely reuses the existing control block.

**Q5. Why does the size of a `struct` often exceed the sum of the sizes of its individual members?**
This is due to memory padding and alignment. CPUs read memory in word-sized chunks (e.g., 4 or 8 bytes). To ensure variables start at addresses that align with these chunk boundaries (for performance reasons), the compiler inserts invisible, unused bytes called "padding" between the struct members.

**Q6. How can you minimize the size of a `struct` without changing the types of its members?**
You can minimize wasted padding space by ordering the members in descending order of their size. Place 8-byte types (pointers, `double`, `int64_t`) first, followed by 4-byte types (`int`, `float`), down to 1-byte types (`char`, `bool`) at the end.

```cpp
// Bad packing (size is likely 24 bytes due to padding)
struct Bad {
    char a;      // 1 byte + 7 padding
    double b;    // 8 bytes
    int c;       // 4 bytes + 4 padding
};

// Good packing (size is exactly 16 bytes)
struct Good {
    double b;    // 8 bytes
    int c;       // 4 bytes
    char a;      // 1 byte + 3 padding at the end
};
```

**Q7. What is placement `new`, and when would you use it?**
Placement `new` is a variation of the `new` operator that does not allocate memory from the OS. Instead, you pass it a pre-allocated memory address, and it simply constructs an object at that exact location. It is used in performance-critical applications (like custom memory pools or object arenas) to avoid the overhead of dynamic heap allocation.

```cpp
char buffer[sizeof(MyClass)]; // Pre-allocated memory
MyClass* obj = new (buffer) MyClass(); // Constructs obj inside buffer

```

**Q8. *[Trap]* How do you properly destroy an object created with placement `new`?**
You cannot use `delete obj;` because standard `delete` will attempt to free the memory back to the heap, which will crash since the memory might be on the stack or part of a custom pool. You must manually call the destructor: `obj->~MyClass();`.

**Q9. Why must a base class with virtual functions have a virtual destructor?**
If a base class destructor is not virtual, deleting a derived class object through a base class pointer will result in undefined behavior—typically, only the base class destructor is called. This leaves the derived class members undestroyed, causing a memory leak.

```cpp
class Base { public: ~Base() {} }; // Missing virtual!
class Derived : public Base { std::vector<int> data; };

Base* b = new Derived();
delete b; // LEAK! Derived's vector is never destroyed.

```

**Q10. *[Trap]* Can you call a virtual function from a constructor or destructor?**
Yes, but it will not exhibit polymorphic behavior. During the execution of a base class constructor, the derived class does not exist yet. The virtual table (`vtable`) points to the base class's functions. Calling a virtual function inside a base constructor will only call the base version of that function, never the derived version.

**Q11. What is the difference between `delete` and `delete[]`? What happens if you mix them up?**
`delete` is used to destroy a single object created with `new`. `delete[]` is used to destroy an array of objects created with `new[]`. If you allocate an array like `int* arr = new int[10];` and use `delete arr;`, it results in undefined behavior. Typically, it will only call the destructor for the first element in the array, leading to memory leaks for the remaining elements.

**Q12. What happens if an exception is thrown inside a constructor?**
If a constructor throws an exception, the object is considered "never fully constructed." Its destructor will **not** be called. However, any fully constructed sub-objects or member variables will have their destructors called. This is why using RAII members (like `std::unique_ptr` or `std::vector`) inside a class is crucial; they will clean themselves up even if the parent constructor throws.

**Q13. What is a dangling pointer, and why is `nullptr` not a complete fix?**
A dangling pointer is a pointer that points to memory that has already been freed or has gone out of scope. Setting a pointer to `nullptr` right after calling `delete` prevents *that specific pointer* from being a dangling pointer. However, if there are *other* pointers in your program pointing to that exact same memory address, those other pointers are now dangling, and setting the original pointer to `nullptr` does nothing to protect them.

---

**Would you like to move on to the pure coding practice questions for Memory Management, or would you prefer to switch to the next major subfolder/topic in your repository?**