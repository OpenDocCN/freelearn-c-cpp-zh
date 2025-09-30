# Concurrency

In the previous chapter, we discussed how `std::shared_ptr<T>` implements reference-counting memory management, so that an object's lifetime can be cooperatively controlled by stakeholders who might be otherwise unaware of each other--for example, the stakeholders might live in different threads. In C++ before C++11, this would have posed a stumbling block right away: if one stakeholder decrements the reference count while, simultaneously, a stakeholder in another thread is in the process of decrementing the reference count, then don't we have a *data race* and therefore undefined behavior?

In C++ before C++11, the answer was generally "yes." (In fact, C++ before C++11 didn't have a standard concept of "threads," so another reasonable answer might have been that the question itself was irrelevant.) In C++ as of 2011, though, we have a standard memory model that accounts for concepts such as "threading" and "thread-safety," and so the question is meaningful and the answer is categorically "No!" Accesses to the reference count of `std::shared_ptr` are guaranteed not to race with each other; and in this chapter we'll show you how you can implement similarly thread-safe constructions using the tools the standard library provides.

In this chapter we'll cover the following topics:

*   The difference between `volatile T` and `std::atomic<T>`
*   `std::mutex`, `std::lock_guard<M>`, and `std::unique_lock<M>`
*   `std::recursive_mutex` and `std::shared_mutex`
*   `std::condition_variable` and `std::condition_variable_any`
*   `std::promise<T>` and `std::future<T>`
*   `std::thread` and `std::async`
*   The dangers of `std::async`, and how to build a thread pool to replace it

# The problem with volatile

If you've been living under a rock for the past ten years--or if you're coming from old-style C--you might ask, "What's wrong with the `volatile` keyword? When I want to make sure some access really hits memory, I make sure it's done `volatile`."

The official semantics of `volatile` are that volatile accesses are evaluated strictly according to the rules of the abstract machine, which means, more or less, that the compiler is not allowed to reorder them or combine multiple accesses into one. For example, the compiler cannot assume that the value of `x` remains the same between these two loads; it must generate machine code that performs two loads, one on either side of the store to `y`:

[PRE0]

If `x` were not volatile, then the compiler would be perfectly within its rights to reorder the code like this:

[PRE1]

The compiler could do this (if `x` weren't volatile) because the write to a `bool` variable `y` cannot possibly affect the value of the `int` variable `x`. However, since `x` is volatile, this reordering optimization is not allowed.

[PRE2]

So that's what `volatile` does. But why can't we use `volatile` to make our programs thread-safe? In essence, the problem with `volatile` is that it's too old. The keyword has been in C++ ever since we split off from C, and it was in C since the original standard in 1989\. Back then, there was very little concern about multithreading, and compilers were simpler, which meant that some potentially problematic optimizations had not yet been dreamt of. By the late 1990s and early 2000s, when C++'s lack of a thread-aware memory model started to become a real concern, it was too late to make `volatile` do everything that was required for thread-safe memory access, because every vendor had already implemented `volatile` and documented exactly what it did. Changing the rules at that point would have broken a lot of people's code--and the code that would have broken would have been low-level hardware interface code, the kind of code you really don't want bugs in.

Here are a couple of examples of the kind of guarantee we need in order to get thread-safe memory accesses:

[PRE3]

Suppose `thread_A`, `thread_B`, and `thread_C` are all running concurrently in different threads. How could this code go wrong? Well, `thread_B` is checking that `x` always holds exactly either zero or `0x42'00000042`. On a 32-bit computer, however, it may not be possible to make that guarantee; the compiler might have to implement the assignment in `thread_A` as a pair of assignments "set the upper half of `x` to 42; set the lower half of `x` to 42." If the test in `thread_B` happens to run at the right (wrong) time, it could end up seeing `x` as `0x42'00000000`. Making `x` volatile will not help with this one; in fact, nothing will, because our 32-bit hardware simply doesn't support this operation! It would be nice for the compiler to detect that we're trying to get an atomic 64-bit assignment, and give us a compile-time error if it knows our goal is impossible. In other words, `volatile` accesses are not guaranteed to be *atomic*. In practice, they often are atomic--and so are non-volatile accesses, but they aren't *guaranteed* to be, and sometimes you have to go down to the machine code level to figure out whether you're getting the code you expected. We'd like a way to guarantee that an access will be atomic (or if that's impossible, we'd like a compiler error).

Now consider `thread_C`. It's checking that *if* the value of `y` is visibly true, *then* the value of `x` must already be set to its final value. In other words, it's checking that the write to `x` "happened before" the write to `y`. This is definitely true from the point of view of `thread_A`, at least if `x` and `y` are both volatile, because we have seen that the compiler is not allowed to reorder volatile accesses. However, the same is not necessarily true from the point of view of `thread_C`! If `thread_C` is running on a different physical CPU, with its own data cache, then it may become aware of the updated values of `x` and `y` at different times, depending on when it refreshes their respective cache lines. We would like a way to say that when the compiler loads from `y`, it must also ensure that its entire cache is up-to-date--that it will never read a "stale" value for `x`. However, on some processor architectures, that requires special instructions, or additional memory-barrier logic. The compiler doesn't generate those instructions for "old-style" volatile accesses, because threading wasn't a concern when `volatile` was invented; and the compiler can't be *made* to generate those instructions for volatile accesses, because that would unnecessarily slow down or maybe even break, all the existing low-level code that uses old-style `volatile` for its old-style meaning. So we're left with the problem that even though volatile accesses happen in sequential order from the point of view of their own thread, they may well appear in a different order from the point of view of another thread. In other words, `volatile` accesses are not guaranteed to be *sequentially consistent*. We'd like a way to guarantee that an access will be sequentially consistent with respect to other accesses.

The solution to both of our problems was added to C++ in 2011\. That solution is `std::atomic`.

# Using std::atomic<T> for thread-safe accesses

In C++11 and later, the `<atomic>` header contains the definition of class template `std::atomic<T>`. There are two different ways you can think about `std::atomic`: you can think of it as a class template just like `std::vector`, with overloaded operators that just happen to implement thread-safe operations; or you can think of it as a magical built-in family of types whose names just happen to contain angle brackets. The latter way of thinking about it is actually pretty useful, because it suggests--correctly--that `std::atomic` is partly built into the compiler, and so the compiler will usually generate optimal code for atomic operations. The latter also suggests a way in which `atomic` is different from `vector`: with `std::vector<T>`, the `T` can be pretty much anything you like. With `std::atomic<T>`, the `T` is *can* be anything you like, but in practice it is a bad idea to use any `T` that doesn't belong to a small set of *atomic-friendly* types. More on this topic in a moment.

The *atomic-friendly* types are the integral types (at least, those no bigger than a machine register) and the pointer types. Generally speaking, on common platforms, you'll find that operations on `std::atomic` objects of these types will do exactly what you want:

[PRE4]

`std::atomic<T>` overloads its assignment operator to perform atomic, thread-safe assignment; and likewise its `++`, `--`, `+=`, and `-=` operators; and for integral types, also the `&=`, `|=`, and `^=` operators.

It's important to bear in mind the difference between *objects* of type `std::atomic<T>` (which conceptually live "out there" in memory) and short-lived *values* of type `T` (which conceptually live "right here," close at hand; for example, in CPU registers). So, for example, there is no copy-assignment operator for `std::atomic<int>`:

[PRE5]

There's no copy-assignment operator (nor move-assignment operator) because it wouldn't have a clear meaning: Does the programmer mean that the computer should load the value of `b` into a register and then store the value of that register into `a`? That sounds like two different atomic operations, not one operation! Or the programmer might mean that the computer should copy the value from `b` to `a` in a single atomic operation; but that involves touching two different memory locations in a single atomic operation, which is not within the capabilities of most computer hardware. So instead, C++ requires that you write out explicitly what you mean: a single atomic load from object `b` into a register (represented in C++ by a non-atomic stack variable), and then a single atomic store into object `a`:

[PRE6]

`std::atomic<T>` provides the member functions `.load()` and `.store(v)` for the benefit of programmers who like to see what they're doing at every step. Using them is optional:

[PRE7]

In fact, by using these member functions, you *could* write the assignment in a single line of code as `b.store(a.load())`; but I advise strongly against doing that. Writing both function calls on one line of code does *not* mean that they'll happen "closer together" in time, and *certainly* doesn't mean they'll happen "atomically" (as we've just seen, that's impossible on most hardware), but writing both function calls on one line of code might very well *deceive you into thinking* that the calls happen "together."

Dealing with threaded code is hard enough when you're doing only one thing at a time. If you start getting clever and doing several things at once, in a single line of code, the potential for bugs skyrockets. Stick to a single atomic operation per source line; you'll find that it clarifies your thinking process and incidentally makes your code easier to read.

# Doing complicated operations atomically

You may have noticed that the operators `*=`, `/=`, `%=`, `<<=`, and `>>=` were omitted from the list of overloaded operators in the preceding section. These operators were deleted by `std::atomic<int>` and all the rest of the integral atomic types because they were perceived as being difficult to provide efficiently on any real hardware. However, even among the operations that were included in `std::atomic<int>`, most of them require a slightly expensive implementation trick.

Let's suppose that our hardware doesn't have an "atomic multiply" instruction, but we'd still like to implement `operator*=`). How would we do it? The trick is to use a primitive atomic operation known as "compare and swap," or in C++ "compare-exchange."

[PRE8]

The meaning of `a.compare_exchange_weak(expected, desired)` is that the processor should look at `a`; and *if* its value is currently `expected`, then set its value to `desired`; otherwise don't. The function call returns `true` if `a` was set to `desired` and `false` otherwise.

But there's one more thing it does, too. Notice that every time through the preceding loop, we're loading the value of `a` into `expected`; but the compare-exchange function is also loading the value of `a` in order to compare it with `expected`. The second time we go through the loop, we'd prefer not to load `a` a second time; we'd prefer simply to set `expected` to the value that the compare-exchange function saw. Fortunately, `a.compare_exchange_weak(expected, desired)` anticipates this desire of ours, and preemptively--if it would return `false`--updates `expected` to the value it saw. That is, whenever we use `compare_exchange_weak`, we must provide a modifiable value for `expected` because the function takes it by reference.

Therefore, we should really write our example like this:

[PRE9]

The `desired` variable isn't really necessary except if it helps to clarify the code.

The dirty little secret of `std::atomic` is that most of the compound assignment operations are implemented as compare-exchange loops just like this. On RISC processors, this is practically always the case. On x86 processors, this is the case only if you want to use the return value of the operator, as in `x = (a += b)`.

When the atomic variable `a` isn't being modified very frequently by other threads, there's no harm in doing a compare-exchange loop. But when `a` is being frequently modified--when it is highly *contended*--then we might see the loop being taken several times before it succeeds. In an absolutely pathological case, we might even see starvation of the looping thread; it might just keep looping forever, until the contention died down. However, notice that every time our compare-exchange returns `false` and we loop around again, it is because the value of `a` in memory has changed; which means that some other thread must have made a little bit of progress. Compare-exchange loops by themselves will never cause the program to enter a state where *nobody* is making progress (a state known technically as "livelock").

The previous paragraph probably sounds scarier than it ought to. There's generally no need to worry about this pathological behavior, since it manifests itself only under really high contention and even then doesn't really cause any terrible problem. The real takeaway you should take from this section is how you can use a compare-exchange loop to implement complicated, non-built-in "atomic" operations on `atomic<T>` objects. Just remember the order of the parameters to `a.compare_exchange_weak(expected, desired)` by remembering what it does to `a`: "if `a` has the expected value, give it the desired value."

# Big atomics

The compiler will recognize and generate optimal code for `std::atomic<T>` when `T` is an integral type (including `bool`), or when `T` is a pointer type such as `void *`. But what if `T` is a bigger type, such as `int[100]`? In that case, the compiler will generally call out to a routine in the C++ runtime library which will perform the assignment under a *mutex*. (We'll look at mutexes in a moment.) Since the assignment is being performed out in a library which doesn't know how to copy arbitrary user-defined types, the C++17 standard restricts `std::atomic<T>` to work only with types that are *trivially copyable*, which is to say they can be copied safely using `memcpy`. So, if you wanted `std::atomic<std::string>`, tough luck--you'll have to write that one yourself.

The other catch when using big (trivially copyable) types with `std::atomic` is that the relevant C++ runtime routines often live in a different place from the rest of the C++ standard library. On some platforms, you'd be required to add `-latomic` to your linker command line. But this is only a problem if you actually do use big types with `std::atomic`, and as you really shouldn't, there's generally no reason to worry.

Now let's look at how you'd write that atomic string class!

# Taking turns with std::mutex

Suppose we want to write a class type that behaves basically like `std::atomic<std::string>` would, if it existed. That is, we'd like to make it support atomic, thread-safe loads and stores, so that if two threads are accessing the `std::string` concurrently, neither one will ever observe it in a "halfway assigned" state, the way we observed a "halfway assigned" `int64_t` in the code sample in the previous section "The problem with volatile."

The best way to write this class is to use a standard library type called `std::mutex`. The name "mutex" is so common in technical circles that these days it basically just stands for itself, but originally its name is derived from "*mut*ual *ex*clusion." This is because a mutex acts as a way to ensure that only one thread is allowed into a particular section of code (or set of sections of code) at once--that is, to ensure that the possibilities "thread A is executing this code" and "thread B is executing this code" are *mutually exclusive* possibilities.

At the start of such a critical section, to indicate that we don't want to be disturbed by any other thread, we *take a lock* on the associated mutex. When we leave the critical section, we *release the lock*. The library takes care of making sure that no two threads can hold locks on the same mutex at the same time. Specifically, this means that if thread B comes in while thread A is already holding the lock, thread B must *wait* until thread A leaves the critical section and releases the lock. As long as thread A holds the lock, thread B's progress is *blocked*; therefore this phenomenon is referred to as either *waiting* or *blocking*, interchangeably.

"Taking a lock on a mutex" is often shortened to "locking the mutex," and "releasing the lock" shortened to "unlocking the mutex."

Sometimes (albeit rarely) it can be useful to test whether a mutex is currently locked. For this purpose `std::mutex` exposes not only the member functions `.lock()` and `.unlock()` but also the member function `.try_lock()`, which returns `true` if it was able to acquire a lock on the mutex (in which case the mutex will be locked) and `false` if the mutex was already locked by some thread.

In some languages, like Java, each object carries with it its own mutex; this is how Java implements its `synchronized` blocks, for example. In C++, a mutex is its own object type; when you want to use a mutex to control a section of code, you need to think about the lifetime semantics of the mutex object itself. Where can you put the mutex so that there will be just a single mutex object that is visible to everyone who wants to use it? Sometimes, if there is just one critical section that needs protection, you can put the mutex in a function-scoped static variable:

[PRE10]

The `static` keyword here is very important! If we had omitted it, then `m` would have been a plain old stack variable, and each thread that entered `log` would have received its own distinct copy of `m`. That wouldn't have helped us with our goal, because the library merely ensures that no two threads have a lock on the *same* mutex object at once. If each thread is locking and unlocking its own distinct mutex object, then the library has nothing to do; none of the mutexes are being *contended*.

If we want to make sure that two different functions are mutually exclusive with each other, such that only one thread is allowed in either `log1` or `log2` at any given time, we must put the mutex object somewhere that can be seen by both critical sections:

[PRE11]

Generally, if you find yourself needing to do this, you should try to eliminate the global variable by creating a class type and making the mutex object a member variable of that class, like this:

[PRE12]

Now messages printed by one `Logger` may interleave with messages printed by another `Logger`, but concurrent accesses to the same `Logger` object will take locks on the same `m_mtx`, which means they will block each other and nicely take turns entering the critical functions `log1` and `log2`, one at a time.

# "Taking locks" the right way

Recall from [Chapter 6](part0093.html#2OM4A0-2fdac365b8984feebddfbb9250eaf20d), *Smart Pointers*, that one of the major problems of programs written in C and "old-style" C++ is the presence of pointer bugs--memory leaks, double-frees, and heap corruption--and that the way we eliminate those bugs from "new-style" C++ programs is via the use of RAII types such as `std::unique_ptr<T>`. Multi-threaded programming with raw mutexes have failure modes that are analogous to the failure modes of heap programming with raw pointers:

*   **Lock leaks***:* You might take a lock on a particular mutex, and accidentally forget to write the code that frees it.
*   **Lock leaks**: You might have written that code, but due to an early return or an exception being thrown, the code never runs and the mutex remains locked!
*   **Use-outside-of-lock**: Because a raw mutex is just another variable, it is physically disassociated from the variables it "guards." You might accidentally access one of those variables without taking the lock first.
*   **Deadlock**: Suppose thread A takes a lock on mutex 1 and thread B takes a lock on mutex 2\. Then, thread A attempts to acquire a lock on mutex 2 (and blocks); and while thread A is still blocked, thread B attempts to acquire a lock on mutex 1 (and blocks). Now both threads are blocked, and will never make progress again.

This is not an exhaustive list of concurrency pitfalls; for example, we've already briefly mentioned "livelock" in connection with `std::atomic<T>`. For a thorough treatment of concurrency bugs and how to avoid them, consult a book on multithreaded or concurrent programming.

The C++ standard library has some tools that help us eliminate these bugs from our multithreaded programs. Unlike the situation with memory management, the standard library's solutions in this case are not 100 percent guaranteed to fix your issues--multithreading is much harder than single-threaded programming, and in fact a good rule of thumb is *not to do it* if you can help it. But if you must do concurrent programming, the standard library can help somewhat.

Just as in [Chapter 6](part0093.html#2OM4A0-2fdac365b8984feebddfbb9250eaf20d), *Smart Pointers,* we can eliminate bugs related to "lock leaks" by the conscientious use of RAII. You might have noticed that I have been consistently using the phrase "take a lock on the mutex" instead of "lock the mutex"; now we'll see why. In the phrase "lock the mutex," "lock" is a *verb*; this phrasing corresponds exactly to the C++ code `mtx.lock()`. But in the phrase "take a lock *on* the mutex," "lock" is a *noun*. Let's invent a type that reifies the idea of "lock"; that is, that turns it into a noun (an RAII class) instead of a verb (a method on a non-RAII class):

[PRE13]

As suggested by the name, `std::unique_lock<M>` is a "unique ownership" RAII class, similar in spirit to `std::unique_ptr<T>`. If you stick to using the noun `unique_ptr` instead of the verbs `new` and `delete`, you'll never forget to free a pointer; and if you stick to using the noun `unique_lock` instead of the verbs `lock` and `unlock`, you'll never forget to release a mutex lock.

`std::unique_lock<M>` does expose the member functions `.lock()` and `.unlock()`, but generally you will not need to use those. They can be useful if you need to acquire or release a lock in the middle of a block of code, far away from the natural point of destruction of the `unique_lock` object. We will also see in the next section a function that takes as a parameter a locked `unique_lock`, which the function unlocks and re-locks as part of its functionality.

Notice that because `unique_lock` is movable, it must have a "null" or "empty" state, just like `unique_ptr`. In most cases, you won't need to move your locks around; you'll just unconditionally take the lock at the start of some scope, and unconditionally release it at the end of the scope. For this use-case, there's `std::lock_guard<M>`. `lock_guard` is much like `unique_lock`, but it is not movable, nor does it have the `.lock()` and `.unlock()` member functions. Therefore, it doesn't need to carry around an `m_locked` member, and its destructor can unconditionally unlock the mutex the object has been guarding, without any extra tests.

In both cases (`unique_lock` and `lock_guard`), the class template is parameterized on the kind of mutex being locked. (We'll look at a couple more kinds of mutexes in a minute, but almost invariably, you'll want to use `std::mutex`.) C++17 has a new language feature called *class template argument deduction* that, in most cases, allows you to elide the template parameter: to write simply `std::unique_lock` instead of `std::unique_lock<std::mutex>`, for example. This is one of the very few cases where I would personally recommend relying on class template argument deduction, because writing out the parameter type `std::mutex` really adds so little information for your reader.

Let's see some examples of `std::lock_guard`, with and without class template argument deduction:

[PRE14]

Before we can see similarly practical examples of `std::unique_lock`, we'll have to explain a good reason to use `std::unique_lock` in the first place.

# Always associate a mutex with its controlled data

Consider the following sketch of a thread-safe `StreamingAverage` class. There is a bug here; can you find it?

[PRE15]

The bug is the line `A`, which writes to `this->m_count` in the producer thread, races with line `D`, which reads from `this->m_count` in the consumer thread. Line `A` correctly takes a lock on `this->m_mtx` before writing, but line `D` fails to take a similar lock, which means that it will happily barge in and attempt to read `m_count` even while line `A` is writing to it.

Lines `B` and `C` look superficially similar, which is probably how the bug originally crept in. Line `C` doesn't need to take a lock; why should line `D` have to? Well, line `C` is called only from the consumer thread, which is the same thread that writes to `m_last_average` on line `B`. Since lines `B` and `C` are executed only by the single consumer thread, they can't both be executed simultaneously--at least as long as the rest of the program conforms to the comments! (Let's assume the code comments are correct. This is often sadly untrue in practice, but for the sake of this example let's assume it.)

We have a recipe for confusion here: Locking `m_mtx` is required when touching `m_sum` or `m_count`, but it is not required when touching `m_last_average`. If this class becomes even more complicated, it might even have several mutexes involved (although at that point, it would clearly be violating the Single Responsibility Principle and would probably benefit from refactoring into smaller components). Therefore, a very good practice when dealing with mutexes is to place the mutex in the tightest possible relationship to the variables it "guards." One way to do this is simply via careful naming:

[PRE16]

A better way is via a nested struct definition:

[PRE17]

The hope above is that when the programmer is forced to write `this->m_guarded_sc.sum`, it reminds him to make sure he's already gotten a lock on `this->m_guarded_sc.mtx`. We could use the GNU extension of "anonymous struct members" to avoid retyping `m_guarded_sc` all over our code; but this would defeat the purpose of this approach, which is to make sure that every place the data is accessed *must* use the word "guarded," reminding the programmer to take that lock on `this->m_guarded_sc.mtx`.

An even more bulletproof, but somewhat inflexible, approach is to place the mutex in a class that allows access to its private members only when the mutex is locked, by returning an RAII handle. The handle-returning class would look more or less like this:

[PRE18]

And our `StreamingAverage` class could use it like this:

[PRE19]

This pattern is included in Facebook's Folly library under the name `folly::Synchronized<T>`, and many more variations on it are available in Ansel Sermersheim and Barbara Geller's "libGuarded" template library.

Notice the use of `std::unique_lock<std::mutex>` in the `Handle` class! We're using `unique_lock` here, not `lock_guard`, because we want the ability to pass this lock around, return it from functions, and so on--so it needs to be movable. This is the main reason you'd reach into your toolbox for `unique_lock`.

Do be aware that this pattern does not solve all lock-related bugs--it solves only the simplest "forget to lock the mutex" cases--and it might encourage programming patterns that lead to more concurrency bugs of other types. For example, consider the following rewrite of `StreamingAverage::get_current_average`:

[PRE20]

Because of the two calls to `m_sc.lock()`, there is a gap between the read of `m_sum` and the read of `m_count`. If the producer thread calls `add_value` during this gap, we will compute an incorrect average (too low by a factor of `1 / m_count`). And if we try to "fix" this bug by taking a lock around the entire computation, we'll find ourselves in deadlock:

[PRE21]

The line marked `LOCK 1` causes the mutex to become locked; then, on the line marked `LOCK 2`, we try to lock the mutex again. The general rule with mutexes is, if you're trying to lock a mutex and it's already locked, then you must *block* and wait for it to become unlocked. So our thread blocks and waits for the mutex to unlock--which will never happen, since the lock is being held by our own thread!

This problem (deadlock with oneself) should generally be dealt with by careful programming--that is, you should try not to take locks you already hold! But if taking locks this way is unavoidably part of your design, then the standard library has your back, so let's talk about `recursive_mutex`.

# Special-purpose mutex types

Recall that `std::lock_guard<M>` and `std::unique_lock<M>` are parameterized on the mutex type. So far we've seen only `std::mutex`. However, the standard library does contain a few other mutex types which can be useful in special circumstances.

`std::recursive_mutex` is like `std::mutex`, but remembers *which* thread has locked it. If that particular thread tries to lock it a second time, the recursive mutex will merely increment an internal reference count of "how many times I've been locked." If some other thread tries to lock the recursive mutex, that thread will block until the original thread has unlocked the mutex the appropriate number of times.

`std::timed_mutex` is like `std::mutex`, but is aware of the passage of time. It has as member functions not only the usual `.try_lock()`, but also `.try_lock_for()` and `.try_lock_until()`, which interact with the standard `<chrono>` library. Here's an example of `try_lock_for`:

[PRE22]

And here's an example of `try_lock_until`:

[PRE23]

Incidentally, the `count_ms` function being used here is just a little lambda that factors out some of the usual `<chrono>` boilerplate:

[PRE24]

In both of the preceding examples, pay attention to our use of `std::atomic<bool>` to synchronize threads `A` and `B`. We simply initialize the atomic variable to `false`, and then loop until it becomes `true`. The body of the polling loop is a call to `std::this_thread::sleep_for`, which is a sufficient hint to the compiler that the value of the atomic variable might change. Be careful never to write a polling loop that does not contain a sleep, because in that case the compiler is within its rights to collapse all the consecutive loads of `ready` down into a single load and a (necessarily infinite) loop.

`std::recursive_timed_mutex` is like you took `recursive_mutex` and `timed_mutex` and smushed them together; it provides the "counting" semantics of `recursive_mutex`, *plus* the `try_lock_for` and `try_lock_until` methods of `timed_mutex`.

`std::shared_mutex` is perhaps poorly named. It implements behavior that in most concurrency textbooks would be called a *read-write lock* (also known as a *rwlock* or *readers-writer lock*). The defining characteristic of a read-write lock, or `shared_mutex`, is that it can be "locked" in two different ways. You can take a normal exclusive ("write") lock by calling `sm.lock()`, or you can take a non-exclusive ("read") lock by calling `sm.lock_shared()`. Many different threads are allowed to take read locks at the same time; but if *anybody* is reading, then *nobody* can be writing; and if *anybody* is writing, then *nobody* can be doing anything else (neither reading nor writing). These happen to be fundamentally the same rules that define "race conditions" in the C++ memory model: two threads reading from the same object simultaneously is fine, as long as no thread is writing to it at the same time. What `std::shared_mutex` adds to the mix is safety: it ensures that if anyone *does* try to write (at least if they play nice and take a write lock on the `std::shared_mutex` first), they'll block until all the readers have exited and it's safe to write.

`std::unique_lock<std::shared_mutex>` is the noun corresponding to an exclusive ("write") lock on a `std::shared_mutex`. As you might expect, the standard library also provides `std::shared_lock<std::shared_mutex>` to reify the idea of a non-exclusive ("read") lock on a `std::shared_mutex`.

# Upgrading a read-write lock

Suppose you have a read lock on a `shared_mutex` (that is to say, you have a `std::shared_lock<std::shared_mutex> lk` such that `lk.owns_lock()`), and you want to get a write lock. Can you "upgrade" your lock?

No, you can't. Consider what would happen if threads `A` and `B` both hold read locks, and simultaneously attempt to upgrade to write locks without first releasing their read locks. Neither one would be able to acquire a write lock, and so they'd deadlock with each other.

There *are* third-party libraries that attempt to solve this problem, such as `boost::thread::upgrade_lock`, which works with `boost::thread::shared_mutex`; but they are outside the scope of this book. The standard solution is that if you hold a read lock and want a write lock, you must release your read lock and then go stand in line for a write lock with everyone else:

[PRE25]

# Downgrading a read-write lock

Suppose you have an exclusive write lock on a `shared_mutex` and you want to get a read lock. Can you "downgrade" your lock?

In principle the answer is yes, it should be possible to downgrade a write lock to a read lock; but in standard C++17 the answer is no, you can't do it directly. As in the upgrade case, you can use `boost::thread::shared_mutex`. The standard solution is that if you hold a write lock and want a read lock, you must release your write lock and then go stand in line for a read lock with everyone else:

[PRE26]

As you can see from these examples, C++17's `std::shared_mutex` is a bit half-baked at the moment. If your architectural design calls for a read-write lock, I strongly recommend using something like `boost::thread::shared_mutex`, which comes "batteries included."

You may have noticed that since new readers can come in while a read lock is held, but new writers cannot, it is conceivable and even likely for a prospective writer thread to be "starved" by a steady stream of prospective readers, unless the implementation goes out of its way to provide a strong "no starvation" guarantee. `boost::thread::shared_mutex` provides such a guarantee (at least, it avoids starvation if the underlying operating system's scheduler does). The standard wording for `std::shared_mutex` provides no such guarantee, although any implementation that allowed starvation in practice would be considered a pretty poor one. In practice you'll find that your standard library vendor's implementation of `shared_mutex` is pretty close to the Boost one, except for the missing upgrade/downgrade functionality.

# Waiting for a condition

In the section titled "Special-purpose mutex types," we launched a task in a separate thread and then needed to wait until a certain bit of initialization was done before continuing. We used a polling loop around a `std::atomic<bool>` in that case. But there are better ways to wait!

The problem with our 50-millisecond polling loop is that it *never* spends the right amount of time asleep. Sometimes our thread will wake up, but the condition it's waiting for hasn't been satisfied, so it'll go back to sleep--that means we didn't sleep long enough the first time. Sometimes our thread will wake up and see that the condition it's waiting for *has* been satisfied, sometime in the past 50 milliseconds, but we don't know how long ago--that means we've *overslept* by about 25 milliseconds on average. Whatever happens, the chance that we slept *just precisely the right amount of time* is slim to none.

So, if we don't want to waste time, the right thing to do is to avoid polling loops. The standard library provides a way to wait just the *right* amount of time; it's called `std::condition_variable`.

Given a variable `cv` of type `std::condition_variable`, our thread can "wait on" `cv` by calling `cv.wait(lk)`; that puts our thread to sleep. Calling `cv.notify_one()` or `cv.notify_all()` wakes up one, or all of, the threads currently waiting on `cv`. However, this is not the only way that those threads might wake up! It's possible that an interrupt from outside (such as a POSIX signal) might jar your thread awake without anybody's having called `notify_one`. This phenomenon is called a *spurious wakeup*. The usual way to guard against spurious wakeups is to check your condition when you wake up. For example, if you're waiting for some input to arrive in a buffer `b`, then when you wake up, you ought to check `b.empty()` and, if it's empty, go back to waiting.

By definition, some other thread is going to be putting that data into `b`; so when you read `b.empty()`, you'd better do it under some kind of mutex. Which means the first thing you'll do when you wake up is take a lock on that mutex, and the last thing you'll do when you go back to sleep is release your lock on that mutex. (In fact, you need to release your lock on that mutex atomically with the going-to-sleep operation, so that nobody can slip in, modify `b`, and call `cv.notify_one()` before you've managed to get to sleep.) This chain of logic leads us to the reason that `cv.wait(lk)` takes that parameter `lk`--it's a `std::unique_lock<std::mutex>` that will be released upon going to sleep and regained upon awaking!

Here's an example of waiting for some condition to be satisfied. First the simple but wasteful polling loop on a `std::atomic` variable:

[PRE27]

And now the preferable and more efficient `condition_variable` implementation:

[PRE28]

If we're waiting to read from a structure protected by a read-write lock (that is, a `std::shared_mutex`), then we don't want to pass in a `std::unique_lock<std::mutex>`; we want to pass in a `std::shared_lock<std::shared_mutex>`. We can do this, if (and sadly only if) we plan ahead and define our condition variable to be of type `std::condition_variable_any` instead of `std::condition_variable`. In practice, there is unlikely to be any performance difference between `std::condition_variable_any` and `std::condition_variable`, which means you should choose between them based on your program's needs, or, if either one would serve, then based on the clarity of the resulting code. Generally this means saving four characters and using `std::condition_variable`. However, notice that because of the layer of insulating abstraction provided by `std::shared_lock`, the actual code for waiting on `cv` under a read-write lock is almost identical to the code for waiting on `cv` under a plain old mutex. Here is the read-write lock version:

[PRE29]

This is perfectly correct code, and as efficient as it can be. However, manually fiddling with mutex locks and condition variables is almost as dangerous to one's health as fiddling with raw mutexes or raw pointers. We can do better! The better solution is the subject of our next section.

# Promises about futures

If you haven't encountered concurrent programming topics before, the last few sections probably got progressively more and more challenging. Mutexes are pretty simple to understand because they model a familiar idea from daily life: getting exclusive access to some resource by putting a lock on it. Read-write locks (`shared_mutex`) aren't much harder to understand. However, we then took a significant jump upward in esotericism with condition variables--which are hard to grasp partly because they seem to model not a noun (like "padlock") but a sort of prepositional verb phrase: "sleep until, but also, wake." Their opaque name doesn't help much either.

Now we continue our journey into concurrent programming with a topic that may be unfamiliar even if you've taken an undergraduate course in concurrent programming, but is well worth the learning: *promises* and *futures*.

In C++11, the types `std::promise<T>` and `std::future<T>` always appear in pairs. Someone coming from the Go language might think of a promise-future pair as a sort of *channel*, in that if one thread shoves a value (of type `T`) into the "promise" side of the pair, that value will eventually emerge at the "future" side (which is typically in a different thread by then). However, promise-future pairs are also like unstable wormholes: as soon as you've shoved a single value through the wormhole, it collapses.

We might say that a promise-future pair is like a directed, portable, one-shot wormhole. It's "directed" because you're allowed to shove data into only the "promise" side and retrieve data only via the "future" side. It's "portable" because if you own one end of the wormhole, you can move that end around and even move it between threads; you won't break the tunnel between the two ends. And it's "one-shot" because once you've shoved one piece of data into the "promise" end, you can't shove any more.

Another metaphor for the pair is suggested by their names: A `std::future<T>` is not actually a value of type `T`, but it is in some sense a *future* value--it will, at some point in the future, give you access to a `T`, but "not yet." (In this way, it is also something like a thread-safe `optional<T>`.) Meanwhile, a `std::promise<T>` object is like an unfulfilled promise, or an I-O-U. The holder of the promise object *promises* to put a value of type `T` into it at some point; if he doesn't ever put in a value, then he's "broken his promise."

Generally speaking, you use a promise-future pair by first creating a `std::promise<T>`, where `T` is the type of data you're planning to send through it; then creating the wormhole's "future" end by calling `p.get_future()`. When you're ready to fulfill the promise, you call `p.set_value(v)`. Meanwhile, in some other thread, when you're ready to retrieve the value, you call `f.get()`. If a thread calls `f.get()` before the promise has been fulfilled, that thread will block until the promise is fulfilled and the value is ready to retrieve. On the other hand, when the promise-holding thread calls `p.set_value(v)`, if nobody's waiting, that's fine; `set_value` will just record the value `v` in memory so that it's ready and waiting whenever anyone *does* ask for it via `f.get()`.

Let's see `promise` and `future` in action!

[PRE30]

(For the definition of `count_ms`, see the previous section, *Special-purpose mutex types*.)

One nice detail about the standard library's `std::promise` is that it has a specialization for `void`. The idea of `std::future<void>` might seem a little silly at first--what good is a wormhole if the only data type you can shove through it is a type with no values? But in fact `future<void>` is extremely useful, whenever we don't care so much about the *value* that was received as about the fact that some signal was received at all. For example, we can use `std::future<void>` to implement yet a third version of our "wait for thread B to launch" code:

[PRE31]

Compare this version to the code samples from the section titled "Waiting for a condition." This version is much cleaner! There's practically no cruft, no boilerplate at all. The "signal B's readiness" and "wait for B's readiness" operations both take only a single line of code. So this is definitely the preferred way to signal between a single pair of threads, as far as syntactic cleanliness is concerned. For yet a fourth way to signal from one thread to a group of threads, see this chapter's subsection titled "Identifying individual threads and the current thread."

There *is* a price to pay for `std::future`, though. The price is dynamic memory allocation. You see, `promise` and `future` both need access to a shared storage location, so that when you store `42` in the promise side, you'll be able to pull it out from the future side. (That shared storage location also holds the mutex and condition variable required for synchronizing between the threads. The mutex and condition variable haven't disappeared from our code; they've just moved down a layer of abstraction so that we don't have to worry about them.) So, `promise` and `future` both act as a sort of "handle" to this shared state; but they're both movable types, so neither of them can actually hold the shared state as a member. They need to allocate the shared state on the heap, and hold pointers to it; and since the shared state isn't supposed to be freed until *both* handles are destroyed, we're talking about shared ownership via something like `shared_ptr` (see [Chapter 6](part0093.html#2OM4A0-2fdac365b8984feebddfbb9250eaf20d), *Smart Pointers*). Schematically, `promise` and `future` look like this:

![](img/00023.jpeg)

The shared state in this diagram will be allocated with `operator new`, unless you use a special "allocator-aware" version of the constructor `std::promise`. To use `std::promise` and `std::future` with an allocator of your choice, you'd write the following:

[PRE32]

`std::allocator_arg` is defined in the `<memory>` header. See [Chapter 8](part0129.html#3R0OI0-2fdac365b8984feebddfbb9250eaf20d), *Allocators*, for the details of `MyAllocator`.

# Packaging up tasks for later

Another thing to notice about the preceding diagram is that the shared state doesn't just contain an `optional<T>`; it actually contains a `variant<T, exception_ptr>` (for `variant` and `optional`, see [Chapter 5](part0074.html#26I9K0-2fdac365b8984feebddfbb9250eaf20d), *Vocabulary Types*). This implies that not only can you shove data of type `T` through the wormhole; you can also shove *exceptions* through. This is particularly convenient and symmetrical because it allows `std::future<T>` to represent all the possible outcomes of calling a function with the signature `T()`. Maybe it returns a `T`; maybe it throws an exception; and of course maybe it never returns at all. Similarly, a call to `f.get()` may return a `T`; or throw an exception; or (if the promise-holding thread loops forever) might never return at all. In order to shove an exception through the wormhole, you'd use the method `p.set_exception(ex)`, where `ex` is an object of type `std::exception_ptr` such as might be returned from `std::current_exception()` inside a catch handler.

Let's take a function of signature `T()` and package it up in a future of type `std::future<T>`:

[PRE33]

This class superficially resembles the standard library type `std::packaged_task<R(A...)>`; the difference is that the standard library type takes arguments, and uses an extra layer of indirection to make sure that it can hold even move-only functor types. Back in [Chapter 5](part0074.html#26I9K0-2fdac365b8984feebddfbb9250eaf20d), *Vocabulary Types*, we showed you some workarounds for the fact that `std::function` can't hold move-only function types; fortunately those workarounds are not needed when dealing with `std::packaged_task`. On the other hand, you'll probably never have to deal with `std::packaged_task` in your life. It's interesting mainly as an example of how to compose promises, futures, and functions together into user-friendly class types with externally very simple interfaces. Consider for a moment: The `simple_packaged_task` class above uses type-erasure in `std::function`, and then has the `std::promise` member, which is implemented in terms of `std::shared_ptr`, which does reference counting; and the shared state pointed to by that reference-counted pointer holds a mutex and a condition variable. That's quite a lot of ideas and techniques packed into a very small volume! And yet the interface to `simple_packaged_task` is indeed simple: construct it with a function or lambda of some kind, then call `pt.get_future()` to get a future that you can `f.get()`; and meanwhile call `pt()` (probably from some other thread) to actually execute the stored function and shove the result through the wormhole into `f.get()`.

If the stored function throws an exception, then `packaged_task` will catch that exception (in the promise-holding thread) and shove it into the wormhole. Then, whenever the other thread calls `f.get()` (or maybe it already called it and it's blocked inside `f.get()` right now), `f.get()` will throw that exception out into the future-holding thread. In other words, by using promises and futures, we can actually "teleport" exceptions across threads. The exact mechanism of this teleportation, `std::exception_ptr`, is unfortunately outside the scope of this book. If you do library programming in a codebase that uses a lot of exceptions, it is definitely worth becoming familiar with `std::exception_ptr`.

# The future of futures

As with `std::shared_mutex`, the standard library's own version of `std::future` is only half-baked. A much more complete and useful version of `future` is coming, perhaps in C++20, and there are very many third-party libraries that incorporate the best features of the upcoming version. The best of these libraries include `boost::future` and Facebook's `folly::Future`.

The major problem with `std::future` is that it requires "touching down" in a thread after each step of a potentially multi-step computation. Consider this pathological usage of `std::future`:

[PRE34]

Notice the line marked `DANGER`: each of the three thread bodies has the same bug, which is that they fail to catch and `.set_exception()` when an exception is thrown. The solution is a `try...catch` block, just like we used in our `simple_packaged_task` in the preceding section; but since that would get tedious to write out every time, the standard library provides a neat wrapper function called `std::async()`, which takes care of creating a promise-future pair and spawning a new thread. Using `std::async()`, we have this much cleaner-looking code:

[PRE35]

However, this code is cleaner only in its aesthetics; it's equally horrifically bad for the performance and robustness of your codebase. This is *bad* code!

Every time you see a `.get()` in that code, you should think, "What a waste of a context switch!" And every time you see a thread being spawned (whether explicitly or via `async`), you should think, "What a possibility for the operating system to run out of kernel threads and for my program to start throwing unexpected exceptions from the constructor of `std::thread`!" Instead of either of the preceding codes, we'd prefer to write something like this, in a style that might look familiar to JavaScript programmers:

[PRE36]

Here, there are no calls to `.get()` except at the very end, when we have nothing to do but wait for the final answer; and there is one fewer thread spawned. Instead, before `f1` finishes its task, we attach a "continuation" to it, so that when `f1` does finish, the promise-holding thread can immediately segue right into working on the continuation task (if original task of `f1` threw an exception, we won't enter this continuation at all. The library should provide a symmetrical method, `f1.on_error(continuation)`, to deal with the exceptional codepath).

Something like this is already available in Boost; and Facebook's Folly library contains a particularly robust and fully featured implementation even better than Boost's. While we wait for C++20 to improve the situation, my advice is to use Folly if you can afford the cognitive overhead of integrating it into your build system. The single advantage of `std::future` is that it's standard; you'll be able to use it on just about any platform without needing to worry about downloads, include paths, or licensing terms.

# Speaking of threads...

Throughout this entire chapter, we've been using the word "thread" without ever defining exactly what we mean by it; and you've probably noticed that many of our multithreaded code examples have used the class type `std::thread` and the namespace `std::this_thread` without much explanation. We've been focusing on *how* to synchronize behavior between different threads of execution, but so far we have glossed over exactly *who* is doing the executing!

To put it another way: When execution reaches the expression `mtx.lock()`, where `mtx` is a locked mutex, the semantics of `std::mutex` say that the current thread of execution should block and wait. While that thread is blocked, what is happening? Our C++ program is still "in charge" of what's going on, but clearly *this particular C++ code* is no longer executing; so who *is* executing? The answer is: another thread. We specify the existence of other threads, and the code we want them to execute, by using the standard library class `std::thread`, defined in the `<thread>` header.

To spawn a new thread of execution, simply construct an object of type `std::thread`, and pass a single argument to the constructor: a lambda or function that tells you what code you want to run in the new thread. (Technically, you are allowed to pass multiple arguments; all arguments after the first will be passed along to the first argument as *its* function parameters, after undergoing `reference_wrapper` decay as described in [Chapter 5](part0074.html#26I9K0-2fdac365b8984feebddfbb9250eaf20d), *Vocabulary Types*. As of C++11, lambdas have made the extra arguments to the `thread` constructor unnecessary and even error-prone; I recommend avoiding them.)

The new thread will immediately start running; if you want it to "start up paused," you'll have to build that functionality yourself using one of the synchronization tricks shown in the section titled "Waiting for a condition," or the alternative trick shown in "Identifying individual threads and the current thread."

The new thread will run through the code it's given, and when it gets to the end of the lambda or function you provided to it, it will "become joinable." This idea is very similar to what happens with `std::future` when it "becomes ready": the thread has completed its computation and is ready to deliver the result of that computation to you. Just as with `std::future<void>`, the result of that computation is "valueless"; but the very fact that the computation *has finished* is valuable nonetheless--no pun intended!

Unlike `std::future<void>`, though, it is not permitted to destroy a `std::thread` object without fetching that valueless result. By default, if you destroy any new thread without dealing with its result, the destructor will call `std::terminate`, which is to say, it will bluntly kill your program. The way to avoid this fate is to indicate to the thread that you see and acknowledge its completion--"Good job, thread, well done!"--by calling the member function `t.join()`. Alternatively, if you do not expect the thread to finish (for example if it is a background thread running an infinite loop) or don't care about its result (for example if it represents some short-lived "fire and forget" task), you can dismiss it to the background--"Go away, thread, I don't want to hear from you again!"--via `t.detach()`.

Here are some complete examples of how to use `std::thread`:

[PRE37]

# Identifying individual threads and the current thread

Objects of type `std::thread`, like every other type described in this chapter, do not support `operator==`. You can't directly ask "Are these two thread objects the same?" This also means that you can't use `std::thread` objects as the keys in an associative container such as `std::map` or `std::unordered_map`. However, you *can* ask about equality indirectly, via a feature called *thread-ids*.

The member function `t.get_id()` returns a unique identifier of type `std::thread::id`, which, although it is technically a class type, behaves an awful lot like an integer type. You can compare thread-ids using operators `<` and `==`; and you can use thread-ids as keys in associative containers. Another valuable feature of thread-id objects is that they can be *copied*, unlike `std::thread` objects themselves, which are move-only. Remember, each `std::thread` object represents an actual thread of execution; if you could copy `thread` objects, you would be "copying" threads of execution, which doesn't make a whole lot of sense--and would certainly lead to some interesting bugs!

The third valuable feature of `std::thread::id` is that it is possible to get the thread-id of the *current* thread, or even of the main thread. From within a thread, there is no way to say "Please give me the `std::thread` object that manages this thread." (This would be a trick analogous to `std::enable_shared_from_this<T>` from [Chapter 6](part0093.html#2OM4A0-2fdac365b8984feebddfbb9250eaf20d), *Smart Pointers*; but as we've seen, such a trick requires support from the part of the library that creates managed resources--which in this case would be the constructor of `std::thread`.) And the main thread, the thread in which `main` begins execution, doesn't have a corresponding `std::thread` object at all. But it still has a thread-id!

Finally, thread-ids are convertible in some implementation-defined manner to a string representation, which is guaranteed to be unique--that is, `to_string(id1) == to_string(id2)` if and only if `id1 == id2`. Unfortunately this string representation is exposed only via the stream operator (see [Chapter 9](part0144.html#49AH00-2fdac365b8984feebddfbb9250eaf20d), *Iostreams*); if you want to use the syntax `to_string(id1)` you need to write a simple wrapper function:

[PRE38]

You can get the thread-id of the current thread (including of the main thread, if that happens to be your current thread) by calling the free function `std::this_thread::get_id()`. Look carefully at the syntax! `std::thread` is the name of a class, but `std::this_thread` is the name of a *namespace*. In this namespace live some free functions (unassociated with any C++ class instance) that manipulate the current thread. `get_id()` is one of those functions. Its name was chosen to be reminiscent of `std::thread::get_id()`, but in fact it is a completely different function: `thread::get_id()` is a member function and `this_thread::get_id()` is a free function.

Using two thread-ids, you can find out, for example, which of an existing list of threads represents your current thread:

[PRE39]

What you cannot do, ever, is go the other direction; you cannot reconstruct the `std::thread` object corresponding to a given `std::thread::id`. Because if you could, you'd have two different objects in your program representing that thread of execution: the original `std::thread` wherever it is, and the one you just reconstructed from its thread-id. And you can never have two `std::thread` objects controlling the same thread.

The two other free functions in the `std::this_thread` namespace are `std::this_thread::sleep_for(duration)`, which you've seen me use extensively in this chapter, and `std::this_thread::yield()`, which is basically the same thing as `sleep_for(0ms)`: it tells the runtime that it would be a good idea to context-switch to a different thread right now, but doesn't connote any *particular* time delay on the current thread.

# Thread exhaustion and std::async

In this chapter's section *The future of futures*, we introduced `std::async`, which is a simple wrapper around a thread constructor with the result captured into a `std::future`. Its implementation looks more or less like this:

[PRE40]

Notice the commented-out lines indicating a special behavior "on destruction" of the `std::future` returned from `std::async`. This is a strange and awkward behavior of the standard library's `std::async` implementation, and a good reason to avoid or reimplement `std::async` in your own code: The futures returned from `std::async` have destructors that call `.join()` on their underlying threads! This means that their destructors can block, and that the task certainly will not be "executing in the background" as you might naturally expect. If you call `std::async` and don't assign the returned future to a variable, the return value will be destroyed right then and there, which means ironically that a line containing nothing but a call to `std::async` will actually execute the specified function *synchronously:*

[PRE41]

The original reason for this limitation seems to have been a concern that if `std::async` launched background threads in the usual way, it would lead to people overusing `std::async` and possibly introducing dangling-reference bugs, as in this example:

[PRE42]

If we didn't wait for the result of this future, the function `test()` might return to its caller before the new thread got a chance to run; then, when the new thread did finally run and attempt to increment `i`, it would be accessing a stack variable that no longer existed. So, rather than run the risk of people writing such buggy code, the Standards Committee decided that `std::async` should return futures with special, "magic" destructors that join their threads automatically.

Anyway, overuse of `std::async` is problematic for other reasons as well. The biggest reason is that on all popular operating systems, `std::thread` represents a *kernel thread*--a thread whose scheduling is under the control of the OS kernel. Because the OS has only finite resources to track these threads, the number of threads available to any one process is fairly limited: often only a few tens of thousands. If you're using `std::async` as your thread manager, spawning a new `std::thread` every time you have another task that might benefit from concurrency, you'll quickly find yourself running out of kernel threads. When this happens, the constructor of `std::thread` will start throwing exceptions of type `std::system_error`, often with the text `Resource temporarily unavailable`.

# Building your own thread pool

If you use `std::async` to spawn a thread every time you have a new task, you risk exhausting the kernel's number of available threads for your process. A better way to run tasks concurrently is to use a *thread pool*--a small number of "worker threads" whose sole job is to run tasks as they are provided by the programmer. If there are more tasks than workers, the excess tasks are placed in a *work queue*. Whenever a worker finishes a task, it checks the work queue for new tasks.

This is a well-known idea, but has not yet been taken up into the standard library as of C++17\. However, you can combine the ideas shown in this chapter to create your own production-quality thread pool. I'll walk through a simple one here; it's not "production quality" in terms of performance, but it *is* properly thread-safe and correct in all its functionality. Some performance tweaks will be discussed at the end of the walkthrough.

We'll start with the member data. Notice that we are using the rule that all the data controlled by a mutex should be located together under a single visual namespace; in this case, a nested struct definition. We're also going to use `std::packaged_task<void()>` as our move-only function type; if your codebase already has a move-only function type, you'll probably want to use that instead. If you don't already have a move-only function type, consider adopting Folly's `folly::Function` or Denis Blank's `fu2::unique_function`:

[PRE43]

The `work_queue` variable will hold tasks as they come in to us. The member variable `m_state.aborting` will be set to `true` when it's time for all the workers to stop working and "come home to rest." `m_workers` holds the worker threads themselves; and `m_state.mtx` and `m_cv` are just for synchronization. (The workers will spend much of their time asleep when there's no work to do. When a new task comes in and we need to wake up some worker, we'll notify `m_cv`.)

The constructor of `ThreadPool` spawns worker threads and populates the `m_workers` vector. Each worker thread will be running the member function `this->worker_loop()`, which we'll see in a minute:

[PRE44]

As promised, the destructor sets `m_state.aborting` to `true` and then waits for all of the worker threads to notice the change and terminate. Notice that when we touch `m_state.aborting`, it's only under a lock on `m_state.mtx`; we are following good hygiene in order to avoid bugs!

[PRE45]

Now let's see how we enqueue tasks into the work queue. (We have not yet seen how workers grab tasks out; we'll see that happening in the `worker_loop` member function.) It's very straightforward; we just have to make sure that we access `m_state` only under the mutex lock, and that once we have enqueued the task, we call `m_cv.notify_one()` so that some worker will wake up to handle the task:

[PRE46]

At last, here is the worker loop. This is the member function that each worker runs:

[PRE47]

Notice the inevitable loop around `m_cv.wait(lk)`, and notice that we hygienically access `m_state` only under the mutex lock. Also notice that when we actually call out to perform `task`, we release the mutex lock first; this ensures that we are not holding the lock for a very long time while the user's task executes. If we *were* to hold the lock for a long time, then no other worker would be able to get in and grab its next task--we'd effectively reduce the concurrency of our pool. Also, if we were to hold the lock during `task`, and if `task` itself tried to enqueue a new task on this pool (which requires taking the lock itself), then `task` would deadlock and our whole program would freeze up. This is a special case of the more general rule never to call a user-provided callback while holding a mutex lock: that's generally a recipe for deadlock.

Finally, let's round out our `ThreadPool` class by implementing a safe version of `async`. Our version will allow calling `tp.async(f)` for any `f` that is callable without arguments, and just like `std::async`, we'll return a `std::future` via which our caller can retrieve the result of `f` once it's ready. Unlike the futures returned from `std::async`, our futures will be safe to drop on the floor: If the caller decides that he doesn't want to wait for the result after all, the task will remain enqueued and will eventually be executed, and the result will simply be ignored:

[PRE48]

We can use our `ThreadPool` class to write code like this function, which creates 60,000 tasks:

[PRE49]

We could try to do the same with `std::async`, but we'd likely run into thread exhaustion when we tried to create 60,000 kernel threads. The preceding example uses only four kernel threads, as indicated by the parameter to the `ThreadPool` constructor.

When you run this code, you'll see at least the numbers 0 through 42 printed to standard output, in some order. We know that 42 must be printed because the function definitely waits for `futures[42]` to be ready before it exits, and all the previous numbers must be printed because their tasks were placed in the work queue ahead of task number 42\. The numbers 43 through 59,999 might or might not be printed, depending on the scheduler; because as soon as task 42 is completed, we exit `test` and thus destroy the thread pool. The thread pool's destructor, as we've seen, notifies all of its workers to stop working and come home after they complete their current tasks. So it is likely that we'll see a few more numbers printed, but then all the workers will come home and the remaining tasks will be dropped on the floor.

Of course if you wanted the destructor of `ThreadPool` to block until all enqueued tasks were completed, you could do that, by changing the code of the destructor. However, typically when you're destroying a thread pool, it's because your program (such as a web server) is exiting, and that's because you've received a signal such as the user pressing *Ctrl* + *C*. In that situation, you *probably* want to exit as soon as you can, as opposed to trying to clear the queue. Personally, I'd prefer to add a member function `tp.wait_for_all_enqueued_tasks()`, so that the user of the thread pool could decide whether they want to block or just drop everything on the floor.

# Improving our thread pool's performance

The biggest performance bottleneck in our `ThreadPool` is that every worker thread is vying for the same mutex, `this->m_state.mtx`. The reason they're all contending that mutex is because that is the mutex that guards `this->m_state.work_queue`, and every worker needs to touch that queue in order to find out its next job. So one way to reduce contention and speed up our program is to find a way of distributing work to our workers that doesn't involve a single central work queue.

The simplest solution is to give each worker its own "to-do list"; that is, to replace our single `std::queue<Task>` with a whole `std::vector<std::queue<Task>>`, with one entry for each worker thread. Of course then we'd also need a `std::vector<std::mutex>` so that we had one mutex for each work queue. The `enqueue_task` function distributes tasks to the work queues in a round-robin fashion (using atomic increments of a `std::atomic<int>` counter to deal with simultaneous enqueues).

You could alternatively use a `thread_local` counter per enqueuing thread, if you are fortunate enough to work on a platform that supports C++11's `thread_local` keyword. On x86-64 POSIX platforms, access to a `thread_local` variable is approximately as fast as access to a plain old global variable; all the complication of setting up thread-local variables happens under the hood and only when you spawn a new thread. However, because that complication *does* exist and needs runtime support, many platforms do not yet support the `thread_local` storage class specifier. (On those that do, `thread_local int x` is basically the same thing as `static int x`, except that when your code accesses `x` by name, the actual memory address of `x` will vary depending on `std::this_thread::get_id()`. In principle, there is a whole array of `x` somewhere behind the scenes, indexed by thread-id and populated by the C++ runtime as threads are created and destroyed.)

The next significant performance improvement to our `ThreadPool` would be "work-stealing": now that each worker has its own to-do list, it might happen by chance or malice that one worker becomes overworked while all the other workers lie idle. In this case, we want the idle workers to scan the queues of the busy workers and "steal" tasks if possible. This re-introduces lock contention among the workers, but only when an inequitable assignment of tasks has already produced inefficiency--inefficiency which we are hoping to *correct* via work-stealing.

Implementing separate work queues and work-stealing is left as an exercise for the reader; but I hope that after seeing how simple the basic `ThreadPool` turned out, you won't be too daunted by the idea of modifying it to include those extra features.

Of course, there also exists professionally written thread-pool classes. Boost.Asio contains one, for example, and Asio is on track to be brought into the standard perhaps in C++20\. Using Boost.Asio, our `ThreadPool` class would look like this:

[PRE50]

An explanation of Boost.Asio is, of course, far outside the scope of this book.

Any time you use a thread pool, be careful that the tasks you enqueue never block indefinitely on conditions controlled by other tasks in the same thread pool. A classic example would be a task A that waits on a condition variable, expecting that some later task B will notify the condition variable. If you make a `ThreadPool` of size 4 and enqueue four copies of task A followed by four copies of task B, you'll find that task B never runs--the four worker threads in your pool are all occupied by the four copies of task A, which are all asleep waiting for a signal that will never come! "Handling" this scenario is tantamount to writing your own user-space threading library; if you don't want to get into that business, then the only sane answer is to be careful that the scenario cannot arise in the first place.

# Summary

Multithreading is a difficult and subtle subject, with many pitfalls that are obvious only in hindsight. In this chapter we have learned:

`volatile`, while useful for dealing directly with hardware, is insufficient for thread-safety. `std::atomic<T>` for scalar `T` (up to the size of a machine register) is the right way to access shared data without races and without locks. The most important primitive atomic operation is compare-and-swap, which in C++ is spelled `compare_exchange_weak`.

To force threads to take turns accessing shared non-atomic data, we use `std::mutex`. Always lock mutexes via an RAII class such as `std::unique_lock<M>`. Remember that although C++17 class template argument deduction allows us to omit the `<M>` from these templates' names, that is just a syntactic convenience; they remain template classes.

Always clearly indicate which data is controlled by each mutex in your program. One good way to do this is with a nested struct definition.

`std::condition_variable` allows us to "sleep until" some condition is satisfied. If the condition can be satisfied only once, such as a thread becoming "ready," then you probably want to use a promise-future pair instead of a condition variable. If the condition can be satisfied over and over again, consider whether your problem can be rephrased in terms of the *work queue* pattern.

`std::thread` reifies the idea of a thread of execution. The "current thread" is not directly manipulable as a `std::thread` object, but a limited set of operations are available as free functions in the `std::this_thread` namespace. The most important of these operations are `sleep_for` and `get_id`. Each `std::thread` must always be joined or detached before it can be destroyed. Detaching is useful only for background threads that you will never need to shut down cleanly.

The standard function `std::async` takes a function or lambda for execution on some other thread, and returns a `std::future` that becomes ready when the function is done executing. While `std::async` itself is fatally flawed (destructors that `join`; kernel thread exhaustion) and thus should not be used in production code, the general idea of dealing with concurrency via futures is a good one. Prefer to use an implementation of promises and futures that supports the `.then` method. Folly's implementation is the best.

*Multithreading is a difficult and subtle subject, with many pitfalls that are* *obvious only in hindsight.*