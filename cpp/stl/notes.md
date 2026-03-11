### Pillar 1: Containers (Where the Data Lives)

Containers are just data structures that manage memory for you. The absolute most important thing an interviewer wants to know is if you understand **Memory Layout and Cache Locality**. Modern CPUs are incredibly fast, but fetching memory from RAM is slow. The CPU grabs chunks of memory into its ultra-fast L1/L2 cache.

* **Sequence Containers (Linear data):**
* **`std::vector`:** The king. It is a single, contiguous block of memory. Because the data sits right next to each other, the CPU pulls the whole chunk into its cache at once. It is blindingly fast to iterate over. **Default to this 95% of the time.**
* **`std::list`:** A doubly-linked list. Every element is a separately allocated node somewhere random on the heap. This causes "cache misses" because the CPU has to constantly jump around RAM to find the next node. You only use this if you need massive amounts of insertions/deletions in the *middle* of the data.
* **`std::deque`:** A "double-ended queue." It looks like a vector, but under the hood, it's an array of pointers pointing to fixed-size memory blocks. It allows fast insertions at *both* the front and the back (unlike a vector, where `push_front` is O(N)).


* **Associative Containers (Key-Value or Node-based):**
* **`std::map` / `std::set`:** Implemented as Red-Black Trees. They keep your data perfectly sorted at all times. The trade-off is that every element is a separate heap allocation (bad cache locality) and lookups are O(log N).
* **`std::unordered_map` / `std::unordered_set`:** Implemented as Hash Tables. No sorting, but lookups are O(1).


* **Container Adapters (Wrappers):**
* These aren't new data structures; they just wrap existing ones to restrict how you use them.
* **`std::stack`:** LIFO (Last In, First Out). Usually wraps a `deque`.
* **`std::queue`:** FIFO (First In, First Out). Usually wraps a `deque`.
* **`std::priority_queue`:** A Heap. It ensures the "maximum" or "minimum" element is always at the top. Usually wraps a `vector`.



---

### Pillar 2: Iterators (The Bridge)

Iterators are the glue that connects Containers to Algorithms. An iterator is basically a safe, abstracted pointer.

* **Why they exist:** An algorithm like `std::count` doesn't know what a `std::vector` or a `std::set` is. It only knows that it has been given a `begin()` iterator and an `end()` iterator, and it can use `++` to move forward.
* **The Big Interview Trap (Iterator Invalidation):** You must know exactly when an iterator becomes a dangling pointer. If you have an iterator pointing to element #3 in a vector, and the vector exceeds its capacity and moves its entire memory block to a new address, your iterator is now pointing to dead memory.

---

### Pillar 3: Algorithms (The Logic)

The `<algorithm>` header contains dozens of pre-written, highly optimized functions. Interviewers look for candidates who use these instead of writing raw `for` loops.

* **Sorting & Searching:** `std::sort` (which is an Introsort: QuickSort + HeapSort + InsertionSort), `std::binary_search`.
* **Querying:** `std::find`, `std::count_if`, `std::any_of`.
* **Modifying:** `std::transform` (like `map` in Python/JS), `std::accumulate` (like `reduce`).
* **The Erase-Remove Idiom:** A classic C++ quirk. Because algorithms only know about iterators, `std::remove` cannot actually delete elements from a vector (it can't resize the vector's memory). It just shifts the "garbage" to the end of the vector. You have to combine it with the vector's actual `erase()` method to physically shrink the container. (Note: C++20 finally fixed this with `std::erase`, but the idiom is still widely tested).

---

### Pillar 4: Functors & Lambdas (Customizing the Logic)

Algorithms are great, but sometimes you need them to do something specific. For example, how do you tell `std::sort` to sort a list of custom `Employee` objects by their `salary`?

You pass a callable object.

* In the old days, we wrote **Functors** (a `struct` that overloads the `operator()`).
* In modern C++, we use **Lambdas** (`[]() {}`). Lambdas are just "inline functors." They allow you to write custom sorting logic, filtering conditions, or transformations directly inside the algorithm call, while safely capturing local variables from the surrounding scope.

---

This is the entire mental model of the STL. You pick a **Container** based on your memory and cache needs, you use **Iterators** to point to the data, you apply an **Algorithm** to do the heavy lifting, and you inject a **Lambda** to customize the algorithm's behavior.

