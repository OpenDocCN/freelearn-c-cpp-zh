# Chapter 6. Boost Your Code Using Multithreading

In this chapter, we will gain skills about:

*   How to run multiple parts of your program in parallel
*   How to protect memory access to avoid data race
*   How to incorporate those functionalities into Gravitris

At the end of this chapter, you will be able to use all the power offered by the CPU of the computer, by paralyzing your code in a smart way. But first, let's describe the theory.

# What is multithreading?

In computer science, a software can be seen as a stream with a start and exit point. Each software starts its life with the `main()` function in C/C++. This is the entry point of your program. Until this point, you are able to do whatever you want; including creating new routine streams, cloning the entire software, and starting another program. The common point with all these examples is that another stream is created and has its own life, but they are not equivalent.

## The fork() function

This functionality is pretty simple. Calling `fork()` will duplicate your entire running process to a new one. The new process that is created is totally separated from its parent (new PID, new memory area as the exact copy of its parent), and will start just after the `fork()` call. The return value of the `fork()` function is the only difference between the two executions.

Following is an example of the `fork()` function:

[PRE0]

As you can see, it is very simple to use, but there are also some limitations with this use. The most important one concerns the sharing of memory. Because each process has its own memory area, you are not able to share some variables between them. A solution to this is to use files as sockets, pipes, and so on. Moreover, if the parent process dies, the child will still continue its own life without paying attention to its parent.

So this solution is interesting only when you don't want to share anything between your different executions, even their states.

## The exec() family functions

The `exec()` family functions (`execl()`, `execlp()`, `execle()`, `execv()`, `execvp()`, `execvpe()`) will replace the entire running program with another one. When paired with `fork()`, these functions become very powerful. Following is an example of these functions:

[PRE1]

This little code snippet will create two different processes as previously mentioned. Then, the child process will be replaced by an instance of Gravitris. As a call of any of the `exec()` family functions replace the entire running stream with a new one, all the code under the `exec` call will not be executed, except if an error occurs.

# Thread functionality

Now, we will speak about threads. The threads' functionalities are very close to the fork ones, but with some important differences. A thread will create a new stream to your running process. Its starting point is a function that is specified as a parameter. A thread will also be executed in the same context as its parent. The main implication is that the memory is the same, but it's not the only one. If the parent process dies, all its threads will die too.

These two points can be a problem if you don't know how to deal with them. Let's take an example of the concurrent memory access.

Let's say that you have a global variable in your program named `var`. The main process will then create a thread. This thread will then write into `var` and at the same time, the main process can write in it too. This will result in an undefined behavior. There are different solutions to avoid this behavior and the common one is to lock the access to this variable with a mutex.

To put it simply, a mutex is a token. We can try to take (lock) it or release it (unlock). If more than one process wants to lock it at the same time, the first one will effectively lock it and the second process will be waiting until the unlock function is called on the mutex by the first one. To sum up, if you want to access to a shared variable by more than one thread, you have to create a mutex for it. Then, each time you want to access it, lock the mutex, access the variable, and finally unlock the mutex. With this solution, you are sure that you don't make any data corrupt.

The second problem concerns the synchronization of the end of the execution of your thread with the main process. In fact, there is a simple solution for this. At the end of the main stream, you need to wait until the end of all the running threads. The stream will be blocked as long as any threads remain alive and consequently will not die.

Here is an example of usage of a thread's functionality:

[PRE2]

Now that the theory has been explained, let's explain what is the motivation to use multithreading.

## Why do we need to use the thread functionality?

Nowadays, computers in general have a CPU that is able to deal with several threads at the same time. Most of the time 4-12 calculation units are present in a CPU. Each of these units are able to do a task independently from the others.

Let's pretend that your CPU has only four calculation units.

If you take the example of our previous games, all the work was done in a single thread. So only one core is used over the four present. This is a shame, because all the work is done by only one component, and the others are simply not used. We can make it better by splitting our code into several parts. Each of these parts will be executed into a different thread, and the job will be shared between them. Then, the different threads will be executed into a different core (with a maximum of four in our case). So the work is now done in parallel.

Creating several threads offers you the possibility to exploit all the power offered by the computer, allowing you to spend more time on some functionalities such as artificial intelligence.

Another way of usage is when you use some blocking functions such as waiting for a message from the network, playing music, and so on. The problem here is that the running process will be in wait for something, and can't continue its execution. To deal with this, you can simply create a thread and delegate a job to it. This is exactly how `sf::Music` works. There is an internal thread that is used to play music. This is the reason why our games do not freeze when we play a sound or music. Each time a thread is created for this task, it appears transparent to the user. Now that the theory has been explained, let's use it in practice.

## Using threads

In [Chapter 4](ch04.html "Chapter 4. Playing with Physics"), *Playing with Physics*, we have introduced physics to our game. For this functionality, we have created two game loops: one for logic and another one for physics. Until now, the executions of the physics loop and the other one were made in the same process. Now, it's time to separate their execution into distinct threads.

We will need to create a thread, and protect our variables using a `Mutex` class. There are two options:

*   Using object from the standard library
*   Using object from the SFML library

Here is a table that summarizes the functionalities needed and the conversion from a standard C++ library to SFML.

The `thread` class:

| Library | Header | Class | Start | Wait |
| --- | --- | --- | --- | --- |
| C++ | `<thread>` | `std::thread` | Directly after construction | `::join()` |
| SFML | `<SFML/System.hpp>` | `sf::Thread` | `::launch()` | `::wait()` |

The `mutex` class:

| Library | Header | Class | Lock | Unlock |
| --- | --- | --- | --- | --- |
| C++ | `<mutex>` | `std::mutex` | `::lock()` | `::unlock()` |
| SFML | `<SFML/System.hpp>` | `sf::Mutex` | `::lock()` | `::unlock()` |

There is a third class that can be used. It automatically calls `mutex::lock()` on construction and `mutex::unlock()` on destruction, in respect of the RAII idiom. This class is called a lock or guard. Its use is simple, construct it with mutex as a parameter and it will automatically lock/unlock it. Following table explains the details of this class:

| Library | Header | Class | Constructor |
| --- | --- | --- | --- |
| C++ | `<mutex>` | `std::lock_guard` | `std::lock_guard(std::mutex&)` |
| SFML | `<SFML/System.hpp>` | `sf::Lock` | `sf::Lock(sf::Mutex&)` |

As you can see both libraries offer the same functionalities. The API changed a bit for the `thread` class, but nothing really important.

In this book, I will use the SFML library. There is no real reason for this choice, except that it allows me to show you a bit more of the SFML possibilities.

Now that the class has been introduced, let's get back to the previous example to apply our new skills as follows:

[PRE3]

There are several parts in this simple example. The first part initializes the global variables. Then, we create a function named `f()` that prints **"Hello world"** and then prints another message. In the `main()` function, we create a thread attached to the `f()` function, we launch it, and print the value of `i`. Each time, we protect the access of the shared variable with a mutex (the two different approaches are used).

The print message from the `f()` function is unpredictable. It could be **"The value of i is 1 from f()"** or **"The value of i is 2 from f()"**. We are not able to say which one of the `f()` or `main()` prints will be made first, so we don't know the value that will be printed. The only point that we are sure of is that there is no concurrent access to `i` and the thread will be ended before the `main()` function, thanks to the `thread.wait()` call.

Now that the class that we needed have been explained and shown, let's modify our games to use them.

# Adding multithreading to our games

We will now modify our Gravitris to paralyze the physics calculations from the rest of the program. We will need to change only two files: `Game.hpp` and `Game.cpp`.

In the header file, we will not only need to add the required header, but also change the prototype of the `update_physics()` function and finally add some attributes to the class. So here are the different steps to follow:

1.  Add `#include <SFML/System.hpp>`, this will allow us to have access to all the classes needed.
2.  Then, change the following code snippet:

    [PRE4]

    to:

    [PRE5]

    The reason is that a thread is not able to pass any parameters to its wrapped function so we will use another solution: member variables.

3.  Add the following variables into the `Game` class as private:

    [PRE6]

    All these variables will be used by the physics thread, and the `_mutex` variable will ensure that no concurrent access to one of those variables is made. We will also need to protect the access to the `_world` variable for the same reasons.

4.  Now that the header contains all the requirements, let's turn to the implementation.
5.  First of all, we will not only need to update our constructor to initialize the `_physicsThread` and `_isRunning` variables, but also protect the access to `_world`.

    [PRE7]

6.  In the constructor, we will not only initialize the new member variables, but also protect our `_world` variable used in one of the callbacks. This lock is important to be sure that no data race occurs randomly during the execution.
7.  Now that the constructor has been updated, we need to change the `run()` function. The goal is to run the physics thread. There are not a lot of changes to make. See it by yourself:

    [PRE8]

8.  Now that the main game loop has been updated, we need to make a small change in the `update()` method to protect the member `_world` variable.

    [PRE9]

9.  As you can see there is only one modification. We just need to protect the access to the `_world` variable, that's it. Now, we need to change the `updatePhysics()` function. This one will be changed a lot as shown in the following code snippet:

    [PRE10]

    We need to change the signature of this function because we are not able to give it some parameters through the thread. So we add an internal clock for this function, with its own loop. The rest of the function follows the logic developed in the `update()` method. Of course, we also use the mutex to protect the access to all the variables used. Now, the physics is able to be updated independently from the rest of the game.

10.  There are now little changes to be made in other functions where `_world` is used such as `initGame()` and `render()`. Each time, we will need to lock the access of this variable using the mutex.
11.  The changes are as follows concerning the `initGame()` function:

    [PRE11]

12.  And now take a look at the `render()` function after it is updated:

    [PRE12]

13.  As you can see, the changes made were really minimalistic, but required to avoid any race conditions.

Now that all the changes have been made in the code, you should be able to compile the project and test it. The graphical result will stay unchanged, but the usage of the different cores of your CPU has changed. Now, the project uses two threads instead of only one. The first one used for the physics and another one for the rest of the game.

# Summary

In this chapter, we covered the use of multithreading and applied it to our existing Gravitris project. We have learned the reason for this, the different possible uses, and the protection of the shared variables.

In our actual game, multithreading is a bit overkill, but in a bigger one for instance with hundreds of players, networking, and real-time strategies; it becomes a *must have*.

In the next chapter, we will build an entire new game and introduce new things such as the isometric view, component system, path finding, and more.