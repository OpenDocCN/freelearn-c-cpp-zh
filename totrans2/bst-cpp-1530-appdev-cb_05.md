# Chapter 5. Multithreading

In this chapter we will cover:

*   Creating an execution thread
*   Syncing access to a common resource
*   Fast access to a common resource using atomics
*   Creating a work_queue class
*   Multiple-readers-single-writer lock
*   Creating variables that are unique per thread
*   Interrupting a thread
*   Manipulating a group of threads

# Introduction

In this chapter we'll take care of threads and all of the stuff connected with them. Basic knowledge of multithreading is encouraged.

**Multithreading** means that multiple execution threads exist within a single process. Threads may share process resources and have their own resources. Those execution threads may run independently on different CPUs, leading to faster and more responsive programs.

The `Boost.Thread` library provides uniformity across operating system interfaces for working with threads. It is not a header-only library, so all of the examples from this chapter will need to link against the `libboost_thread` and `libboost_system` libraries.

# Creating an execution thread

On modern multi-core compilers, to achieve maximal performance (or just to provide a good user experience), programs usually must use multiple execution threads. Here is a motivating example in which we need to create and fill a big file in a thread that draws the user interface:

[PRE0]

## Getting ready

This recipe will require knowledge of the `boost::bind` library.

## How to do it...

Starting an execution thread was never so easy:

[PRE1]

## How it works...

The `boost::thread` variable accepts a functional object that can be called without parameters (we provided one using `boost::bind`) and creates a separate execution thread. That functional object will be copied into a constructed execution thread and will be run there.

![How it works...](img/4880OS_05_01.jpg)

### Note

In all of the recipes with the `Boost.Thread` library, we'll be using Version 4 (defined `BOOST_THREAD_VERSION to 4`) of threads by default and pointing out some important differences between `Boost.Thread` versions.

After that, we call the `detach()` function, which will do the following:

*   The execution thread will be detached from the `boost::thread` variable but will continue its execution
*   The `boost::thread` variable will hold a `Not-A-Thread` state

Note that without a call to `detach()`, the destructor of `boost::thread` will notice that it still holds a thread and will call `std::terminate`, which will terminate our program.

Default constructed threads will also have a `Not-A-Thread` state, and they won't create a separate execution thread.

## There's more...

What if we want to make sure that a file was created and written before doing some other job? In that case we need to join a thread using the following:

[PRE2]

After the thread is joined, the `boost::thread` variable will hold a `Not-A-Thread` state and its destructor won't call `std::terminate`.

### Note

Remember that the thread must be joined or detached before its destructor is called. Otherwise, your program will terminate!

Beware that `std::terminate()` is called when any exception that is not of type `boost::thread_interrupted` leaves the boundary of the functional object and is passed to the `boost::thread` constructor.

The `boost::thread` class was accepted as a part of the C++11 standard and you can find it in the `<thread>` header in the `std::` namespace. By default, with `BOOST_THREAD_VERSION=2`, the destructor of `boost::thread` will call `detach()`, which won't lead to `std::terminate`. But doing so will break compatibility with `std::thread`, and some day, when your project is moving to the C++ standard library threads or when `BOOST_THREAD_VERSION=2` is no longer supported this will give you a lot of surprises. Version 4 of `Boost.Thread` is more explicit and strong, which is usually preferable in C++ language.

There is a very helpful wrapper that works as a RAII wrapper around the thread and allows you to emulate the `BOOST_THREAD_VERSION=2` behavior; it is called `boost::scoped_thread<T>`, where `T` can be one of the following classes:

*   `boost::interrupt_and_join_if_joinable`: To interrupt and join thread at destruction
*   `boost::join_if_joinable`: To join a thread at destruction
*   `boost::detach`: To detach a thread at destruction

Here is a small example:

[PRE3]

### Note

We added additional parentheses around `(boost::thread(&some_func))` so that the compiler won't interpret it as a function declaration instead of a variable construction.

There is no big difference between the Boost and C++11 STL versions of the `thread` class; however, `boost::thread` is available on the C++03 compilers, so its usage is more versatile.

## See also

*   All of the recipes in this chapter will be using `Boost.Thread`; you may continue reading to get more information about them
*   The official documentation has a full list of the `boost::thread` methods and remarks about their availability in the C++11 STL implementation; it can be found at [http://www.boost.org/doc/libs/1_53_0/doc/html/thread.html](http://www.boost.org/doc/libs/1_53_0/doc/html/thread.html)
*   The *Interrupting a thread* recipe will give you an idea of what the `boost::interrupt_and_join_if``_joinable` class does

# Syncing access to a common resource

Now that we know how to start execution threads, we want to have access to some common resources from different threads:

[PRE4]

This `'Oops!'` is not written there accidentally. For some people it will be a surprise, but there is a big chance that `shared_i` won't be equal to 0:

[PRE5]

### Note

Modern compilers and processors have a huge number of different, tricky optimizations that can break the preceding code. We won't discuss them here, but there is a useful link in the *See also* section to a document that briefly describes them.

And it will get even worse in cases when a common resource has some non-trivial classes; segmentation faults and memory leaks may (and will) occur.

We need to change the code so that only one thread modifies the `shared_i` variable at a single moment of time and so that all of the processor and compiler optimizations that inflict multithreaded code are bypassed.

## Getting ready

Basic knowledge of threads is recommended for this recipe.

## How to do it...

Let's see how we can fix the previous example and make `shared_i` equal at the end of the run:

1.  First of all we'll need to create a mutex:

    [PRE6]

2.  Put all the operations that modify or get data from the `shared_i` variable between the following:

    [PRE7]

    And the following:

    [PRE8]

This is what it will look like:

[PRE9]

## How it works...

The `boost::mutex` class takes care of all of the synchronization stuff. When a thread tries to lock it via the `boost::lock_guard<boost::mutex>` variable and there is no other thread holding a lock, it will successfully acquire unique access to the section of code until the lock is unlocked or destroyed. If some other thread already holds a lock, the thread that tried to acquire the lock will wait until another thread unlocks the lock. All the locking/unlocking operations imply specific instructions so that the changes made in a **critical section** will be visible to all threads. Also, you no longer need to *make sure that modified values of resources are visible to all cores and are not just modified in the processor's register* and *force the processor and compiler to not reorder the instructions*.

The `boost::lock_guard` class is a very simple RAII class that stores a reference to the mutex and calls `lock()` in the single-parameter constructor and `unlock()` in the destructor. Note the curly bracket usage in the preceding example; the `lock` variable is constructed inside them so that, on reaching the `critical section` closing bracket, the destructor for the `lock` variable will be called and the mutex will be unlocked. Even if some exception occurs in the critical section, the mutex will be correctly unlocked.

![How it works...](img/4880OS_05_02.jpg)

### Note

If you have some resources that are used from different threads, usually all the code that uses them must be treated as a critical section and secured by a mutex.

## There's more...

Locking a mutex is potentially a very slow operation, which may stop your code for a long time, until some other thread releases a lock. Try to make critical sections as small as possible and try to have less of them in your code.

Let's take a look at how some operating systems (OS) handle locking on a multicore CPU. When `thread #1`, running on CPU1, tries to lock a mutex that is already locked by another thread, `thread #1` is stopped by the OS till the lock is released. The stopped thread does not *eat* processor resources, so the OS will still execute other threads on CPU1\. Now we have some threads running on CPU1; some other thread releases the lock, and now the OS has to resume execution of a `thread #1`. So it will resume its execution on a currently free CPU, for example, CPU2\. This will result in CPU cache misses, and code will be running slightly slower after the mutex is released. This is another reason to reduce the number of critical sections. However, things are not so bad because a good OS will try to resume the thread on the same CPU that it was using before.

Do not attempt to lock a `boost::mutex` variable twice in the same thread; it will lead to a **deadlock**. If locking a mutex multiple times from a single thread is required, use `boost::recursive_mutex` instead of the `<boost/thread/recursive_mutex.hpp>` header. Locking it multiple times won't lead to a deadlock. The `boost::recursive_mutex` will release the lock only after `unlock()` is called once for each `lock()` call. Avoid using `boost::recursive_mutex`; it is slower than `boost::mutex` and usually indicates bad code flow design.

The `boost::mutex`, `boost::recursive_mutex`, and `boost::lock_guard` classes were accepted to the C++11 standard, and you may find them in the `<mutex>` header in the `std::` namespace. No big difference between Boost and STL versions exists; a Boost version may have some extensions (which are marked in the official documentation as *EXTENSION*) and provide better portability because they can be used even on C++03 compilers.

## See also

*   The next recipe will give you some ideas on how to make this example much faster (and shorter).
*   Read the first recipe from this chapter to get more information about the `boost::thread` class. The official documentation for `Boost.Thread` may help you too; it can be found at [http://www.boost.org/doc/libs/1_53_0/doc/html/thread.html](http://www.boost.org/doc/libs/1_53_0/doc/html/thread.html).
*   For more information about why the first example will fail and how multiprocessors work with common resources, see *Memory Barriers: a Hardware View for Software Hackers* at [http://www.rdrop.com/users/paulmck/scalability/paper/whymb.2010.07.23a.pdf](http://www.rdrop.com/users/paulmck/scalability/paper/whymb.2010.07.23a.pdf).

# Fast access to common resource using atomics

In the previous recipe, we saw how to safely access a common resource from different threads. But in that recipe, we were doing two system calls (in locking and unlocking the mutex) just to get the value from an integer:

[PRE10]

This looks lame! And slow! Can we make the code from the previous recipe better?

## Getting ready

Reading the first recipe is all you need to start with this. Or just some basic knowledge of multithreading.

## How to do it...

Let's see how to improve our previous example:

1.  We will need different headers:

    [PRE11]

2.  Changing the type of `shared_i` is required (as it is no longer needed in the mutex):

    [PRE12]

3.  Remove all the `boost::lock_guard` variables:

    [PRE13]

    And that's it! Now it works.

    [PRE14]

## How it works...

Processors provide specific atomic operations that cannot be interfered with by other processors or processor cores. These operations appear to occur instantaneously for a system. `Boost.Atomic` provides classes that wrap around system-specific atomic operations and provide a uniform and portable interface to work with them.

In other words, it is safe to use the `boost::atomic<>` variables from different threads simultaneously. Each operation on the atomic variable will be seen by the system as a single transaction. Series of operations on the atomic variables will be treated by the system as a series of transactions:

[PRE15]

![How it works...](img/4880OS_05_03.jpg)

## There's more...

The `Boost.Atomic` library can work only with POD types; otherwise, its behavior is undefined. Some platforms/processors do not provide atomic operations for some types, so `Boost.Atomic` will emulate atomic behavior using `boost::mutex`. The atomic type won't use `boost::mutex` if the type-specific macro is set to `2`:

[PRE16]

The `boost::atomic<T>::is_lock_free` member function depends on runtime, so it is not good for compile-time checks but may provide a more readable syntax when the runtime check is enough:

[PRE17]

Atomics work much faster than mutexes. If we compare the execution time of a recipe that uses mutexes (0:00.08 seconds) and the execution time of the preceding example in this recipe (0:00.02 seconds), we'll see the difference (tested on 3,00,000 iterations).

The C++11 compilers should have all the atomic classes, typedefs, and macros in the `<atomic>` header in the `std::` namespace. Compiler-specific implementations of `std::atomic` may work faster than the Boost's version, if the compiler correctly supports the C++11 memory model and atomic operations are not a compiler barrier for it any more.

## See also

*   The official documentation may give you many more examples and some theoretical information on the topic; it can be found at [http://www.boost.org/doc/libs/1_53_0/doc/html/atomic.html](http://www.boost.org/doc/libs/1_53_0/doc/html/atomic.html)
*   For more information about how atomics work, see *Memory Barriers: a Hardware View for Software Hackers* at [http://www.rdrop.com/users/paulmck/scalability/paper/whymb.2010.07.23a.pdf](http://www.rdrop.com/users/paulmck/scalability/paper/whymb.2010.07.23a.pdf)

# Creating a work_queue class

Let's call the functional object that takes no arguments (a task, in short).

[PRE18]

And now, imagine a situation where we have threads that post tasks and threads that execute posted tasks. We need to design a class that can be safely used by both types of thread. This class must have methods for getting a task (or blocking and waiting for a task until it is posted by another thread), checking and getting a task if we have one (returning an empty task if no tasks remain), and a method to post tasks.

## Getting ready

Make sure that you feel comfortable with `boost::thread` or `std::thread` and know some basics of mutexes.

## How to do it...

The classes that we are going to implement will be close in functionality to `std::queue<task_t>` and will also have thread synchronization. Let's start:

1.  We'll need the following headers and members:

    [PRE19]

2.  A function for putting a task in the queue will look like this:

    [PRE20]

3.  A non-blocking function for getting a pushed task or an empty task (if no tasks remain):

    [PRE21]

4.  Blocking function for getting a pushed task or for blocking while the task is pushed by another thread:

    [PRE22]

    And this is how a `work_queue` class may be used:

    [PRE23]

## How it works...

In this example, we will see a new RAII class `boost::unique_lock`. It is just a `boost::lock_guard` class with additional functionality; for example, it has methods for explicit unlocking and locking mutexes.

Going back to our `work_queue` class, let's start with the `pop_task()` function. In the beginning, we are acquiring a lock and checking for available tasks. If there is a task, we return it; otherwise, `cond_.wait(lock)` is called. This method will unlock the lock and pause the execution thread until till some other thread notifies the current thread.

Now, let's take a look at the `push_task` method. In it we also acquire a lock, push a task to `tasks_.queue`, unlock the lock, and call `cond_notify_one()`, which will wake up the thread (if any) waiting in `cond_wait(lock)`. So, after that, if some thread was waiting on a conditional variable in a `pop_task()` method, the thread will continue its execution, call `lock.lock()` deep inside `cond_wait(lock)`, and check `tasks_empty()` in the while loop. Because we just added a task in `tasks_`, we'll get out from the `while` loop, unlock the mutex (the `lock` variable will go out of scope), and return a task.

![How it works...](img/4880OS_05_04.jpg)

### Note

It is highly recommended that you check conditions in a loop, not just in an `if` statement. The `if` statement will lead to an error if `thread #1` pops a task after it is pushed by `thread #2` but `thread #3` is notified by `thread #2` before it (`thread #3`) starts waiting.

## There's more...

Note that we explicitly unlocked the mutex before calling `notify_one()`. Without unlocking, our example would still work.

But, in that case, the thread that has woken up may be blocked once more during an attempt to call `lock.lock()` deep inside `cond_wait(lock)`, which leads to more context switches and worse performance.

With `tests_tasks_count` set to `3000000` and without explicit unlocking, this example runs for 7 seconds:

[PRE24]

With explicit unlocking, this example runs for 5 seconds:

[PRE25]

You may also notify all the threads waiting on a specific conditional variable using `cond_notify_all()`.

The C++11 standard has `std::condition_variable` declared in the `<condition_variable>` header and `std::unique_lock` declared in the `<mutex>` header. Use the Boost version if you need portable behavior, use C++03 compiler, or just use some of the Boost's extensions.

## See also

*   The first three recipes in this chapter provide a lot of useful information about `Boost.Thread`
*   The official documentation may give you many more examples and some theoretical information on the topic; it can be found at [http://www.boost.org/doc/libs/1_53_0/doc/html/thread.html](http://www.boost.org/doc/libs/1_53_0/doc/html/thread.html)

# Multiple-readers-single-writer lock

Imagine that we are developing some online services. We have a map of registered users with some properties for each user. This set is accessed by many threads, but it is very rarely modified. All operations with the following set are done in a thread-safe manner:

[PRE26]

But any operation will acquire a unique lock on the `mutex_` variable, so even getting resources will result in waiting on a locked mutex; therefore, this class will become a bottleneck very soon.

Can we fix it?

## How to do it...

Replace `boost::unique_locks` with `boost::shared_lock` for methods that do not modify data:

[PRE27]

## How it works...

We can allow getting the data from multiple threads simultaneously if those threads do not modify it. We need to uniquely own the mutex only if we are going to modify the data in it; in all other situations simultaneous access to it is allowed. And that is what `boost::shared_mutex` was designed for. It allows shared locking (read locking), which allows multiple simultaneous access to resources.

When we do try to unique lock a resource that is shared locked, operations will be blocked until there are no read locks remaining and only after that resource is unique locked, forcing new shared locks to wait until the unique lock is released.

Some readers may be seeing the mutable keyword for the first time. This keyword can be applied to non-static and non-constant class members. The mutable data member can be modified in the constant member functions.

## There's more...

When you do need only unique locks, do not use `boost::shared_mutex` because it is slightly slower than a usual `boost::mutex` class. However, in other cases, it may give a big performance gain. For example, with four reading threads, shared mutex will work almost four times faster than `boost::mutex`.

Unfortunately, shared mutexes are not the part of the C++11 standard.

## See also

*   There is also a `boost::upgrade_mutex` class, which may be useful for cases when a shared lock needs promotion to unique lock. See the `Boost.Thread` documentation at [http://www.boost.org/doc/libs/1_53_0/doc/html/thread.html](http://www.boost.org/doc/libs/1_53_0/doc/html/thread.html) for more information.
*   For more information about the mutable keyword see [http://herbsutter.com/2013/01/01/video-you-dont-know-const-and-mutable/](http://herbsutter.com/2013/01/01/video-you-dont-know-const-and-mutable/).

# Creating variables that are unique per thread

Let's take a glance at the recipe *Creating a* *work_queue class*. Each task there can be executed in one of many threads and we do not know which one. Imagine that we want to send the results of an executed task using some connection.

[PRE28]

We have the following solutions:

*   Open a new connection when we need to send the data (which is slow)
*   Have a single connection for all the threads and wrap them in mutex (which is also slow)
*   Have a pool of connections, get a connection from it in a thread-safe manner and use it (a lot of coding is required, but this solution is fast)
*   Have a single connection per thread (fast and simple to implement)

So, how can we implement the last solution?

## Getting ready

Basic knowledge of threads is required.

## How to do it...

It is time to make a thread local variable:

[PRE29]

Using a thread-specific resource was never so easy:

[PRE30]

## How it works...

The `boost::thread_specific_ptr` variable holds a separate pointer for each thread. Initially, this pointer is equal to `NULL`; that is why we check for `!p` and open a connection if it is `NULL`.

So, when we enter `get_connection()` from the thread that has already initiated the pointer, `!p` will return the value `false` and we'll return the already opened connection. `delete` for the pointer will be called when the thread is exiting, so we do not need to worry about memory leaks.

## There's more...

You may provide your own cleanup function that will be called instead of `delete` at thread exit. A cleanup function must have the `void (*cleanup_function)(T*)` signature and will be passed during the `boost::thread_specific_ptr` construction.

C++11 has a special keyword, `thread_local`, to declare variables with thread local storage duration. C++11 has no `thread_specific_ptr` class, but you may use `thread_local boost::scoped_ptr<T>` or `thread_local std::unique_ptr<T>` to achieve the same behavior on compilers that support `thread_local`.

## See also

*   The `Boost.Thread` documentation gives a lot of good examples on different cases; it can be found at [http://www.boost.org/doc/libs/1_53_0/doc/html/thread.html](http://www.boost.org/doc/libs/1_53_0/doc/html/thread.html)
*   Reading this topic at [http://stackoverflow.com/questions/13106049/c11-gcc-4-8-thread-local-performance-penalty.html](http://stackoverflow.com/questions/13106049/c11-gcc-4-8-thread-local-performance-penalty.html) and about the `GCC__thread` keyword at [http://gcc.gnu.org/onlinedocs/gcc-3.3.1/gcc/Thread-Local.html](http://gcc.gnu.org/onlinedocs/gcc-3.3.1/gcc/Thread-Local.html) may give you some ideas about how `thread_local` is implemented in compilers and how fast it is

# Interrupting a thread

Sometimes, we need to kill a thread that ate too many resources or that is just executing for too long. For example, some parser works in a thread (and actively uses `Boost.Thread`), but we already have the required amount of data from it, so parsing can be stopped. All we have is:

[PRE31]

How can we do it?

## Getting ready

Almost nothing is required for this recipe. You only need to have at least basic knowledge of threads.

## How to do it...

We can stop a thread by interrupting it:

[PRE32]

## How it works...

`Boost.Thread` provides some predefined interruption points in which the thread is checked for being interrupted via the `interrupt()` call. If the thread was interrupted, the exception `boost::thread_interrupted` is thrown.

`boost::thread_interrupted` is not derived from `std::exception`!

## There's more...

As we know from the first recipe, if a function passed into a thread won't catch an exception and the exception will leave function bounds, the application will terminate. `boost::thread_interrupted` is the only exception to that rule; it may leave function bounds and does not `std::terminate()` application; instead, it stops the execution thread.

We may also add interruption points at any point. All we need is to call `boost::this_thread::interruption_point()`:

[PRE33]

If interruptions are not required for a project, defining `BOOST_THREAD_DONT_PROVIDE_INTERRUPTIONS` gives a small performance boost and totally disables thread interruptions.

C++11 has no thread interruptions but you can partially emulate them using atomic operations:

*   Create an atomic Boolean variable
*   Check the atomic variable in the thread and throw some exception if it has changed
*   Do not forget to catch that exception in the function passed to the thread (otherwise your application will terminate)

However, this won't help you if the code is waiting somewhere in a conditional variable or in a sleep method.

## See also

*   The official documentation for `Boost.Thread` provides a list of predefined interruption points at [http://www.boost.org/doc/libs/1_53_0/doc/html/thread/thread_management.html#thread.thread_management.tutorial.interruption.html](http://www.boost.org/doc/libs/1_53_0/doc/html/thread/thread_management.html#thread.thread_management.tutorial.interruption.html)
*   As an exercise, see the other recipes from this chapter and think of where additional interruption points would improve the code
*   Reading other parts of the `Boost.Thread` documentation may be useful; go to [http://www.boost.org/doc/libs/1_53_0/doc/html/thread.html](http://www.boost.org/doc/libs/1_53_0/doc/html/thread.html)

# Manipulating a group of threads

Those readers who were trying to repeat all the examples by themselves or those who were experimenting with threads must already be bored with writing the following code to launch threads:

[PRE34]

Maybe there is a better way to do this?

## Getting ready

Basic knowledge of threads will be more than enough for this recipe.

## How to do it...

We may manipulate a group of threads using the `boost::thread_group` class.

1.  Construct a `boost::thread_group` variable:

    [PRE35]

2.  Create threads into the preceding variable:

    [PRE36]

3.  Now you may call functions for all the threads inside `boost::thread_group`:

    [PRE37]

## How it works...

The `boost::thread_group` variable just holds all the threads constructed or moved to it and may send some calls to all the threads.

## There's more...

C++11 has no `thread_group` class; it's Boost specific.

## See also

*   The official documentation of `Boost.Thread` may surprise you with a lot of other useful classes that were not described in this chapter; go to [http://www.boost.org/doc/libs/1_53_0/doc/html/thread.html](http://www.boost.org/doc/libs/1_53_0/doc/html/thread.html)