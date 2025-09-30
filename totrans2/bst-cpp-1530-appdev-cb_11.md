# Chapter 11. Working with the System

In this chapter we will cover:

*   Listing files in a directory
*   Erasing and creating files and directories
*   Passing data quickly from one process to another
*   Syncing interprocess communications
*   Using pointers in shared memory
*   The fastest way to read files
*   Coroutines – saving the state and postponing the execution

# Introduction

Each operating system has many system calls doing almost the same things in slightly different ways. Those calls differ in performance and differ from one operating system to another. Boost provides portable and safe wrappers around those calls. Knowledge of those wrappers is essential for writing good programs.

This chapter is devoted to working with the operating system. We have seen how to deal with network communications and signals in [Chapter 6](ch06.html "Chapter 6. Manipulating Tasks"), *Manipulating Tasks*. In this chapter, we'll take a closer look at the filesystem and creating and deleting files. We'll see how data can be passed between different system processes, how to read files at maximum speed, and how to perform other tricks.

# Listing files in a directory

There are STL functions and classes to read and write data to files. But there are no functions to list files in a directory, to get the type of a file, or to get access rights for a file.

Let's see how such iniquities can be fixed using Boost. We'll be creating a program that lists names, write accesses, and types of files in the current directory.

## Getting ready

Some basics of C++ would be more than enough to use this recipe.

This recipe requires linking against the `boost_system` and `boost_filesystem` libraries.

## How to do it...

This recipe and the next one are about portable wrappers for working with a filesystem:

1.  We need to include the following two headers:

    [PRE0]

2.  Now we need to specify a directory:

    [PRE1]

3.  After specifying the directory, loop through its content:

    [PRE2]

4.  The next step is getting the file info:

    [PRE3]

5.  Now output the file info:

    [PRE4]

6.  The final step would be to output the filename:

    [PRE5]

That's it. Now, if we run the program, it will output something like this:

[PRE6]

## How it works...

Functions and classes of `Boost.Filesystem` just wrap around system-specific functions to work with files.

Note the usage of `/` in step 2\. POSIX systems use a slash to specify paths; Windows, by default, uses backslashes. However, Windows understands forward slashes too, so `./` will work on all of the popular operating systems, and it means "the current directory".

Take a look at step 3, where we are default constructing the `boost::filesystem::directory_iterator` class. It works just as a `std::istream_iterator` class, which acts as an `end` iterator when default constructed.

Step 4 is a tricky one, not because this function is hard to understand, but because lots of conversions are happening. Dereferencing the `begin` iterator returns `boost::filesystem::directory_entry`, which is implicitly converted to `boost::filesystem::path`, which is used as a parameter for the `boost::filesystem::status` function. Actually, we can do much better:

[PRE7]

### Tip

Read the reference documentation carefully to avoid unrequired implicit conversions.

Step 5 is obvious, so we are moving to step 6 where implicit conversion to the path happens again. A better solution would be the following:

[PRE8]

Here, `begin->path()` returns a const reference to the `boost::filesystem::path` variable that is contained inside `boost::filesystem::directory_entry`.

## There's more...

Unfortunately, `Boost.Filesystem` is not a part of C++11, but it is proposed for inclusion in the next C++ standard. `Boost.Filesystem` currently misses support for rvalue references, but still remains one of the simplest and most portable libraries to work with a filesystem.

## See also

*   The *Erasing and creating files and directories* recipe will show another example of the usage of `Boost.Filesystem`.
*   Read Boost's official documentation for `Boost.Filesystem` to get more info about its abilities; it is available at the following link:

    [http://www.boost.org/doc/libs/1_53_0/libs/filesystem/doc/index.htm](http://www.boost.org/doc/libs/1_53_0/libs/filesystem/doc/index.htm).

*   The `Boost.Filesystem` library is proposed for inclusion in C++1y. The draft is available at [http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3399.html](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3399.h).

# Erasing and creating files and directories

Let's consider the following lines of code:

[PRE9]

In these lines, we attempt to write something to `file.txt` in the `dir/subdir` directory. This attempt will fail if there is no such directory. The ability to work with filesystems is necessary for write a good working code.

In this recipe we'll construct a directory and a subdirectory, write some data to a file, and try to create `symlink`, and if the symbolic link's creation fails, erase the created file. We will also avoid using exceptions as a mechanism of error reporting, preferring some form of return codes.

Let's see how that can be done in an elegant way using Boost.

## Getting ready

Basic knowledge of C++ and the `std::ofstream` class is required for this recipe. `Boost.Filesystem` is not a header-only library, so code in this recipe requires linking against the `boost_system` and `boost_filesystem` libraries.

## How to do it...

We continue to deal with portable wrappers for a filesystem, and in this recipe we'll see how to modify the directory content:

1.  As always, we'll need to include some headers:

    [PRE10]

2.  Now we need a variable to store errors (if any):

    [PRE11]

3.  We will also create directories, if required, as follows:

    [PRE12]

4.  Then we will write data to the file:

    [PRE13]

5.  We need to attempt to create `symlink`:

    [PRE14]

6.  Then we need to check that the file is accessible through `symlink`:

    [PRE15]

7.  Or remove the created file, if `symlink` creation failed:

    [PRE16]

## How it works...

We saw `boost::system::error_code` in action in almost all of the recipes in , *Manipulating Tasks*. It can store information about errors and is widely used throughout the Boost libraries.

### Note

If you do not provide an instance of `boost::system::error_code` to the `Boost.Filesystem` functions, the code will compile well, but when an error occurs, an exception will be thrown. Usually a `boost::filesystem::filesystem_error` exception is thrown unless you are having trouble with allocating memory.

Take a careful look at step 3\. We used the `boost::filesystem::create_directories` function, not `boost::filesystem::create_directory`, because the latter cannot create subdirectories.

The remaining steps are trivial to understand and should not cause any trouble.

## There's more...

The `boost::system::error_code` class is a part of C++11 and can be found in the `<system_error>` header in the `std::` namespace. The classes of `Boost.Filesystem` are not a part of C++11, but they are proposed for inclusion in C++1y, which will probably be ready in 2014.

Finally, a small recommendation for those who are going to use `Boost.Filesystem`; when the errors occurring during filesystem operations are routine, use `boost::system::error_codes`. Otherwise, catching exceptions is preferable and more reliable.

## See also

*   The *Listing files in a directory* recipe also contains information about `Boost.Filesystem`. Read Boost's official documentation to get more information and examples at [http://www.boost.org/doc/libs/1_53_0/libs/filesystem/doc/index.htm](http://www.boost.org/doc/libs/1_53_0/libs/filesystem/doc/index.htm).

# Passing data quickly from one process to another

Sometimes we write programs that will communicate with each other a lot. When programs are run on different machines, using sockets is the most common technique for communication. But if multiple processes run on a single machine, we can do much better!

Let's take a look at how to make a single memory fragment available from different processes using the `Boost.Interprocess` library.

## Getting ready

Basic knowledge of C++ is required for this recipe. Knowledge of atomic variables is also required (take a look at the *See also* section for more information about atomics). Some platforms require linking against the runtime library.

## How to do it...

In this example we'll be sharing a single atomic variable between processes, making it increment when a new process starts and decrement when the process terminates:

1.  We'll need to include the following header for interprocess communications:

    [PRE17]

2.  Following the header, `typedef` and a check will help us make sure that atomics are usable for this example:

    [PRE18]

3.  Create or get a shared segment of memory:

    [PRE19]

4.  Get or construct an `atomic` variable:

    [PRE20]

5.  Work with the `atomic` variable in the usual way:

    [PRE21]

6.  Destroy the `atomic` variable:

    [PRE22]

That's all! Now if we run multiple instances of this program simultaneously, we'll see that each new instance increments its index value.

## How it works...

The main idea of this recipe is to get a segment of memory that is visible to all processes, and place some data in it. Let's take a look at step 3, where we retrieve such a segment of memory. Here, `shm-cache` is the name of the segment (different segments differ in name); you can give any names you like to the segments. The first parameter is `boost::interprocess::open_or_create`, which says that `boost::interprocess::managed_shared_memory` will open an existing segment with the name `shm-cache`, or it will construct it. The last parameter is the size of the segment.

### Note

The size of the segment must be big enough to fit the `Boost.Interprocess` library-specific data in it. That's why we used `1024` and not `sizeof(atomic_t)`. But it does not really matter, because the operating system will round this value to the nearest larger supported value, which is usually equal to or larger than 4 kilobytes.

Step 4 is a tricky one as we are doing multiple tasks at the same time here. In part `2` of this step, we will find or construct a variable with the name `shm-counter` in the segment. In part `3` of step 4, we will provide a parameter, which will be used for the initialization of a variable if it has not been found in step 2\. This parameter will be used only if the variable is not found and must be constructed, otherwise it is ignored. Take a closer look at the second line (part `1`). See the call to the dereference operator `*`. We are doing it because `segment.find_or_construct<atomic_t>` returns a pointer to `atomic_t`, and working with bare pointers in C++ is a bad style.

### Note

Note that we are using atomic variables in shared memory! This is required, because two or more processes can simultaneously work with the same `shm-counter` atomic variable.

You must be very careful when working with objects in shared memory; do not forget to destroy them! In step 6, we are destroying the object and segment using their names.

## There's more...

Take a closer look at step 2 where we are checking for `BOOST_ATOMIC_INT_LOCK_FREE != 2`. We are checking that `atomic_t` won't use mutexes. This is very important, because usually, mutexes won't work in shared memory. So if `BOOST_ATOMIC_INT_LOCK_FREE` is not equal to `2`, we'll get an undefined behavior.

Unfortunately, C++11 has no interprocess classes, and as far as I know, `Boost.Interprocess` is not proposed for inclusion in C++1y.

### Note

Once a managed segment is created, it cannot increase in size! Make sure that you are creating segments big enough for your needs, or take a look at the *See also* section for information about increasing managed segments.

Shared memory is the fastest way for processes to communicate, and works for processes that can share memory. That usually means that the processes must run on the same host or on a **symmetric multiprocessing** (**SMP**) cluster.

## See also

*   The *Syncing interprocess communications* recipe will tell you more about shared memory, interprocess communications, and syncing access to resources in shared memory
*   See the *Fast access to common resource using atomics* recipe in Chapter 5, Multithreading for more information about atomics
*   Boost's official documentation for `Boost.Interprocess` may also help; it is available at [http://www.boost.org/doc/libs/1_53_0/doc/html/interprocess.html](http://www.boost.org/doc/libs/1_53_0/doc/html/interprocess.html)
*   How to increase managed segments is described at [http://www.boost.org/doc/libs/1_53_0/doc/html/interprocess/managed_memory_segments.html#interprocess.managed_memory_segments.managed_memory_segment_advanced_features.growin](http://www.boost.org/doc/libs/1_53_0/doc/html/interprocess/managed_memory_segments.html#interprocess.managed_memory_segments.managed_memory_segment_advanced_features.growin)[g_managed_memory](http://g_managed_memory)

# Syncing interprocess communications

In the previous recipe, we saw how to create shared memory and how to place some objects in it. Now it's time to do something useful. Let's take an example from the *Creating a work_queue class* recipe in [Chapter 5](ch05.html "Chapter 5. Multithreading"), *Multithreading*, and make it work for multiple processes. At the end of this example, we'll get a class that can store different tasks and pass them between processes.

## Getting ready

This recipe uses techniques from the previous one. You will also need to read the *Creating a work_queue class* recipe in [Chapter 5](ch05.html "Chapter 5. Multithreading"), *Multithreading*, and get its main idea. The example requires linking against the runtime library on some platforms.

## How to do it...

It is considered that spawning separate subprocesses instead of threads makes a program more reliable, because termination of a subprocess won't terminate the main process. We won't argue with that assumption here, and just see how data sharing between processes can be implemented.

1.  A lot of headers are required for this recipe:

    [PRE23]

2.  Now we need to define our structure, `task_structure`, which will be used to store tasks:

    [PRE24]

3.  Let's start writing the `work_queue` class:

    [PRE25]

4.  Write the members of `work_queue` as follows:

    [PRE26]

5.  Initialization of members should look like the following:

    [PRE27]

6.  We need to make some minor changes to the member functions of `work_queue`, such as using `scoped_lock_t` instead of the original unique locks:

    [PRE28]

## How it works...

In this recipe, we are doing almost exactly the same things as in the *Creating a work_queue class* recipe in [Chapter 5](ch05.html "Chapter 5. Multithreading"), *Multithreading*, but when we allocate the data in shared memory, additional care must be taken when doing memory allocations or using synchronization primitives.

Take additional care when storing shared memory objects that have pointers or references as member fields. We'll see how to cope with pointers in the next recipe.

Take a look at step 2\. We did not use `boost::function` as a task type because it has pointers in it, so it won't work in shared memory.

Step 3 is interesting because of `allocator_t`. It is a type of allocator that all containers must use to allocate elements. It is a stateful allocator, which means that it will be copied along with the container. Also, it cannot be default constructed.

If memory is not allocated from the shared memory segment, it won't be available to other processes; that's why a specific allocator for containers is required.

Step 4 is pretty trivial, except that we have only references to `tasks_`, `mutex_`, and `cond_`. This is done because objects themselves are constructed in the shared memory. So, `work_queue` can only store references to them.

In step 5 we are initializing members. This code will be familiar to you; we were doing exactly the same things in the previous recipe. Note that we are providing an instance of allocator to `tasks_` while constructing it. That's because `allocator_t` cannot be constructed by the container itself.

### Note

Shared memory is not destructed at the exit event of a process, so we can run the program once, post the tasks to a work queue, stop the program, start some other program, and get tasks stored by the first instance of the program. Shared memory will be destroyed only at restart, or if you explicitly call `segment.deallocate("work-queue");`.

## There's more...

As was mentioned in the previous recipe, C++11 has no classes from `Boost.Interprocess`. Moreover, you must not use C++11 or C++03 containers in shared memory segments. Some of those containers may work, but that behavior is not portable.

If you look inside some of the `<boost/interprocess/containers/*.hpp>` headers, you'll find that they just use containers from the `Boost.Containers` library:

[PRE29]

Containers of `Boost.Interprocess` have all of the benefits of the `Boost.Containers` library, including rvalue references and their emulation on older compilers.

`Boost.Interprocess` is the fastest solution for communication between processes that are running on the same machine.

## See also

*   The *Using pointers in shared memory* recipe
*   Read [Chapter 5](ch05.html "Chapter 5. Multithreading"), *Multithreading*, for more information about synchronization primitives and multithreading
*   Refer to Boost's official documentation for the `Boost.Interprocess` library for more examples and information; it is available at the following link:

    [http://www.boost.org/doc/libs/1_53_0/doc/html/interprocess.html](http://www.boost.org/doc/libs/1_53_0/doc/html/interprocess.html)

# Using pointers in shared memory

It is hard to imagine writing some C++ core classes without pointers. Pointers and references are everywhere in C++, and they do not work in shared memory! So if we have a structure like this in shared memory and assign the address of some integer variable in shared memory to `pointer_`, we won't get the correct address in the other process that will attempt to use `pointer_` from that instance of `with_pointer`:

[PRE30]

How can we fix that?

## Getting ready

The previous recipe is required for understanding this one. The example requires linking against the runtime system library on some platforms.

## How to do it...

Fixing it is very simple; we need only to replace the pointer with `offset_ptr<>`:

[PRE31]

Now we are free to use it as a normal pointer:

[PRE32]

## How it works...

We cannot use pointers in shared memory because when a piece of shared memory is mapped into the address space of a process, its address is valid only for that process. When we are getting the address of a variable, it is just a local address for that process; other processes will map shared memory to a different base address, and as a result the variable address will differ.

![How it works...](img/4880OS_11_01.jpg)

So how can we work with an address that is always changing? There is a trick! As the pointer and structure are in the same shared memory segment, the distance between them does not change. The idea behind `boost::interprocess::offset_ptr` is to remember that distance, and on dereference, add the distance value to the process-dependent address of the `offset_ptr` variable.

The offset pointer imitates the behavior of pointers, so it is a drop-in replacement that can be applied fast.

### Tip

Do not place classes that may have pointers or references into shared memory!

## There's more...

An offset pointer works slightly slower than the usual pointer because on each dereference, it is required to compute the address. But this difference is not usually sufficient to bother you.

C++11 has no offset pointers.

## See also

*   Boost's official documentation contains many examples and more advanced `Boost.Interprocess` features; it is available at [http://www.boost.org/doc/libs/1_53_0/doc/html/interprocess.html](http://www.boost.org/doc/libs/1_53_0/doc/html/interprocess.html)
*   The *fastest way to read files* recipe contains information about some nontraditional usage of the `Boost.Interprocess` library

# The fastest way to read files

All around the Internet, people are asking "What is the fastest way to read files?". Let's make our task for this recipe even harder: "What is the fastest and most portable way to read binary files?"

## Getting ready

Basic knowledge of C++ and the `std::fstream` containers is required for this recipe.

## How to do it...

The technique from this recipe is widely used by applications critical to input and output performance.

1.  We'll need to include two headers from the `Boost.Interprocess` library:

    [PRE33]

2.  Now we need to open a file:

    [PRE34]

3.  The main part of this recipe is mapping all of the files to memory:

    [PRE35]

4.  Getting a pointer to the data in the file:

    [PRE36]

That's it! Now we can work with a file just as with normal memory:

[PRE37]

## How it works...

All popular operating systems have the ability to map a file to processes' address space. After such mapping is done, the process can work with those addresses just as with normal memory. The operating system will take care of all of the file operations, such as caching and read-ahead.

Why is it faster than traditional read/writes? That's because in most cases read/write is implemented as memory mapping and copying data to a user-specified buffer. So read usually does more work.

Just as in the case of STL, we must provide an open mode when opening a file. See step 2 where we provided the `boost::interprocess::read_only` mode.

See step 3 where we mapped a whole file at once. This operation is actually really fast, because the OS does not read data from the disk, but waits for the requests to be a part of the mapped region. After a part of the mapped region was requested, the OS loads that part of the file from the disk. As we can see, memory mapping operations are lazy, and the size of the mapped region does not affect performance.

### Note

However, a 32-bit OS cannot memory-map large files, so you'll need to map them in pieces. POSIX (Linux) operating systems require `_FILE_OFFSET_BITS=64` to be defined for the whole project to work with large files on a 32-bit platform. Otherwise, the OS won't be able to map parts of the file that are beyond 4 GB.

Now it's time to measure the performance:

[PRE38]

Just as expected, memory-mapped files are slightly faster than traditional reads. We can also see that pure C methods have the same performance as that of the C++ `std::ifstream` class, so please do not use functions related to `FILE*` in C++. They are just for C, not for C++!

For optimal performance of `std::ifstream`, do not forget to open files in binary mode and read data by blocks:

[PRE39]

## There's more...

Unfortunately, classes for memory mapping files are not part of C++11, and it looks like they won't be in C++14 either.

Writing to memory-mapped regions is also a very fast operation. The OS will cache the writes and won't flush modifications to the disc immediately. There is a difference between the OS and the `std::ofstream` data caching. If the `std::ofstream` data is cached by an application and it terminates, the cached data can be lost. When data is cached by the OS, termination of the application won't lead to data loss. Power failures and system crashes lead to data loss in both cases.

If multiple processes map a single file, and one of the processes modifies the mapped region, the changes are immediately visible to the other processes.

## See also

*   The `Boost`.`Interprocess` library contains a lot of useful features to work with the system; not all of them are covered in this book. You can read more about this great library at the official site:

    [http://www.boost.org/doc/libs/1_53_0/doc/html/interprocess.html](http://www.boost.org/doc/libs/1_53_0/doc/html/interproces)

# Coroutines – saving the state and postponing the execution

Nowadays, plenty of embedded devices still have only a single core. Developers write for those devices, trying to squeeze maximum performance out of them. Using `Boost.Threads` or some other thread library for such devices is not effective; the OS will be forced to schedule threads for execution, manage resources, and so on, as the hardware cannot run them in parallel.

So how can we make a program switch to the execution of a subprogram while waiting for some resource in the main part?

## Getting ready

Basic knowledge of C++ and templates is required for this recipe. Reading some recipes about `Boost.Function` may also help.

## How to do it...

This recipe is about coroutines, subroutines that allow multiple entry points. Multiple entry points give us an ability to suspend and resume the execution of a program at certain locations, switching to/from other subprograms.

1.  The `Boost.Coroutine` library will take care of almost everything. We just need to include its header:

    [PRE40]

2.  Make a coroutine type with the required signature:

    [PRE41]

3.  Make a coroutine:

    [PRE42]

4.  Now we can execute the subprogram while waiting for an event in the main program:

    [PRE43]

5.  The coroutine method should look like this:

    [PRE44]

## How it works...

At step 2, we are describing the signature of our subprogram using the function signature `std::string& (std::size_t)` as a template parameter. This means that the subprogram accepts `std::size_t` and returns a reference to a string.

Step 3 is interesting because of the `coroutine_task` signature. Note that this signature is common for all coroutine tasks. `caller` is the variable that will be used to get parameters from the caller and to return the result of the execution back.

Step 3 requires additional care because the constructor of `corout_t` will automatically start the coroutine execution. That's why we call `caller(result)` at the beginning of the coroutine task (it returns us to the `main` method).

When we call `coroutine(10)` in step 4, we are causing a coroutine program to execute. Execution will jump to step 5 right after the first `caller(result)` method, where we'll get a value `10` from `caller.get()` and will continue our execution until `caller(result)`. After that, execution will return to step 4, right after the `coroutine(10)` call. Next, a call to `coroutine(10)` or `coroutine(300)` will continue the execution of the subprogram from the place right after the second `caller(result)` method at step 5.

![How it works...](img/4880OS_11_02.jpg)

Take a look at `std::string& s = coroutine.get()` in step 4\. Here, we'll be getting a reference to the `std::string` result from the beginning of `coroutine_task` described in step 5\. We can even modify it, and `coroutine_task` will see the modified value. Let me describe the main difference between coroutines and threads. When a coroutine is executed, the main task does nothing. When the main task is executed, the coroutine task does nothing. You have no such guarantee with threads. With coroutines, you explicitly specify when to start a subtask and when to finish it. In a single core environment, threads can switch at any moment of time; you cannot control that behavior.

### Note

Do not use thread's local storage and do not call `boost::coroutines::coroutine<>::operator()` from inside the same coroutine; do not call `boost::coroutines::coroutine<>::get()` when a coroutine task is finished. These operations lead to undefined behavior.

## There's more...

While switching threads, the OS does a lot of work, so it is not a very fast operation. However, with coroutines, you have full control over switching tasks; moreover, you do not need to do any OS-specific internal kernel work. Switching coroutines is much faster than switching threads, however, it's not as fast as calling `boost::function`.

The `Boost.Coroutine` library will take care of calling a destructor for variables in a coroutine task, so there's no need to worry about leaks.

### Note

Coroutines use the `boost::coroutines::detail::forced_unwind` exception to free resources that are not derived from `std::exception`. You must take care not to catch that exception in coroutine tasks.

C++11 has no coroutines. But coroutines use features of C++11 when possible, and even emulate rvalue references on C++03 compilers. You cannot copy `boost::coroutines::coroutine<>`, but you can move them using `Boost.Move`.

## See also

*   Boost's official documentation contains more examples, performance notes, restrictions, and use cases for the `Boost.Coroutines` library; it is available at the following link:

    [http://www.boost.org/doc/libs/1_53_0/libs/coroutine/doc/html/index.htm](http://www.boost.org/doc/libs/1_53_0/libs/coroutine/doc/html/index.htm)

*   Take a look at recipes from [Chapter 3](ch03.html "Chapter 3. Managing Resources"), *Managing Resources*, and [Chapter 5](ch05.html "Chapter 5. Multithreading"), *Multithreading*, to get the difference between the `Boost.Coroutine`, `Boost.Thread`, and `Boost.Function` libraries