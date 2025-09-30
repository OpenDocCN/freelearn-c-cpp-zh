# Chapter 6. Manipulating Tasks

In this chapter we will cover:

*   Registering a task for processing an arbitrary datatype
*   Making timers and processing timer events as tasks
*   Network communication as a task
*   Accepting incoming connections
*   Executing different tasks in parallel
*   Conveyor tasks processing
*   Making a nonblocking barrier
*   Storing an exception and making a task from it
*   Getting and processing system signals as tasks

# Introduction

This chapter is all about tasks. We'll be calling the functional object a task (because it is shorter and better reflects what it shall do). The main idea of this chapter is that we can split all the processing, computations, and interactions into **functors** (tasks) and process each of those tasks almost independently. Moreover, we may not block on some slow operations (such as receiving data from a socket or waiting for a time-out), but instead provide a callback task and continue working with other tasks. Once the OS finishes the slow operation, our callback will be executed.

## Before you start

This chapter requires at least a basic knowledge of the first, third, and fifth chapters.

# Registering a task for processing an arbitrary datatype

First of all, let's take care of the class that will hold all the tasks and provide methods for their execution. We were already doing something like this in the *Creating a work_queue class* recipe, but some of the following problems were not addressed:

*   A task may throw an exception that leads a call to `std::terminate`
*   An interrupted thread may not notice interruption but will finish its task and interrupt only during the next task (which is not what we wanted; we wanted to interrupt the previous task)
*   Our `work_queue` class was only storing and returning tasks, but we need to add methods for executing existing tasks
*   We need a way to stop processing the tasks

## Getting ready

This recipe requires linking with the `libboost_system` library. Knowledge of `Boost.Bind` and basic knowledge of `Boost.Thread` is also required.

## How to do it...

We'll be using `boost::io_service` instead of `work_queue` from the previous chapter. There is a reason for doing this, and we'll see it in the following recipes.

1.  Let's start with the structure that wraps around a user task:

    [PRE0]

2.  For ease of use, we'll create a function that produces `task_wrapped` from the user's functor:

    [PRE1]

3.  Now we are ready to write the `tasks_processor` class:

    [PRE2]

4.  Now we will add the `push_task` method:

    [PRE3]

5.  Let's finish this class by adding the member functions for starting and stopping a task's execution loop:

    [PRE4]

    It is time to test our class. For that, we'll create a testing function:

    [PRE5]

    The `main` function might look like this:

    [PRE6]

## How it works...

The `boost::io_service` variable can store and execute tasks posted to it. But we may not post a user's tasks to it directly because they may throw or receive an interruption addressed to other tasks. That is why we wrap a user's task in the `detail::task_wrapped` structure. It resets all the previous interruptions by calling:

[PRE7]

And this executes the task within the `try{}catch()` block making sure that no exception will leave the `operator()` bounds.

The `boost::io_service::run()` method will be getting ready tasks from the queue and executing them one by one. This loop is stopped via a call to `boost::io_service::stop()`. The `boost::io_service` class will return from the `run()` function if there are no more tasks left, so we force it to continue execution using an instance of `boost::asio::io_service::work`.

### Note

The **iostream** classes and variables such as `std::cerr` and `std::cout` are not thread safe. In real projects, additional synchronization must be used to get readable output. For simplicity, we do not do that here.

## There's more...

The C++11 STL library has no `io_service`; however, it (and a large part of the `Boost.Asio` library) is proposed as a **Technical Report** (**TR**) as an addition to C++.

## See also

*   The following recipes will show you why we chose `boost::io_service` instead of our handwritten code
*   You may consider the `Boost.Asio` documentation to get some examples, tutorials, and class references at [http://www.boost.org/doc/libs/1_53_0/doc/html/boost_asio.html](http://www.boost.org/doc/libs/1_53_0/doc/html/boost_asio.html)
*   You may also read the *Boost.Asio C++ Network Programming* book, which gives a smoother introduction to `Boost.Asio` and covers some details that are not covered in this book

# Making timers and processing timer events as tasks

It is a common task to check something at specified intervals; for example, we need to check some session for an activity once every 5 seconds. There are two popular solutions to such a problem: creating a thread or sleeping for 5 seconds. This is a very lame solution that consumes a lot of system resources and scales badly. We could instead use system specific APIs for manipulating timers asynchronously. This is a better solution, but it requires a lot of work and is not very portable (until you write many wrappers for different platforms). It also makes you work with OS APIs that are not always very nice.

## Getting ready

You must know how to use `Boost.Bind` and `Boost.SmartPtr`. See the first recipe of this chapter to get information about the `boost::asio::io_service` and `task_queue` classes. Link this recipe with the `libboost_system` library.

This recipe is a tricky one, so get ready!

## How to do it...

This recipe is based on the code from the previous recipe. We just modify the `tasks_processor` class by adding new methods to run a task at some specified time.

1.  Let's add a method to our `tasks_processor` class for running a task at some time:

    [PRE8]

2.  We add a method to our `task_queue` class for running a task after the required time duration passes:

    [PRE9]

3.  It's time to take care of the `detail::make_timer_task` function:

    [PRE10]

4.  And the final step will be writing a `timer_task` structure:

    [PRE11]

## How it works...

That's how it all works; the user provides a timeout and a functor to the `run_after` function. In it, a `detail::timer_task` object is constructed that stores a user provided functor and creates a shared pointer to `boost::asio::deadline_timer`. The constructed `detail::timer_task` object is pushed as a functor that must be called when the timer is triggered. The `detail::timer_task::operator()` method accepts `boost::system::error_code`, which will contain the description of any error that occurred while waiting. If no error is occurred, we call the user's functor that is wrapped to catch exceptions (we re-use the `detail::task_wrapped` structure from the first recipe). The following diagram illustrates this:

![How it works...](img/4880OS_06_01.jpg)

Note that we wrapped `boost::asio::deadline_timer` in `boost::shared_ptr` and passed the whole `timer_task` functor (including `shared_ptr`) in `timer_->async_wait(*this)`. This is done because `boost::asio::deadline_timer` must not be destroyed until it is triggered, and storing the `timer_task` functor in `io_service` guarantees this.

### Note

In short, when a specified amount of time has passed, `boost::asio::deadline_timer` will push the user's task to the `boost::asio::io_service queue` class for execution.

## There's more...

Some platforms have no APIs to implement timers in a good way, so the `Boost.Asio` library emulates the behavior of the asynchronous timer using an additional execution thread per `io_service`. Anyways, `Boost.Asio` is one of the most portable and effective libraries to deal with timers.

## See also

*   Reading the first recipe from this chapter will teach you the basics of `boost::asio::io_service`. The following recipes will provide you with more examples of `io_service` usage and will show you how to deal with network communications, signals, and other features using `Boost.Asio`.
*   You may consider the `Boost.Asio` documentation to get some examples, tutorials, and class references at [http://www.boost.org/doc/libs/1_53_0/doc/htm](http://www.boost.org/doc/libs/1_53_0/doc/htm)[l/boost_asio.html](http://l/boost_asio.html).

# Network communication as a task

Receiving or sending data by network is a slow operation. While packets are received by the machine, and while the OS verifies them and copies the data to the user-specified buffer, multiple seconds may pass. And we may be able to do a lot of work instead of waiting. Let's modify our `tasks_processor` class so that it will be capable of sending and receiving data in an asynchronous manner. In nontechnical terms, we ask it to "receive at least N bytes from the remote host and after that is done, call our functor. And by the way, do not block on this call". Those readers who know about `libev`, `libevent`, or `Node.js` will find a lot of familiar things in this recipe.

## Getting ready

The previous and first recipes from this chapter are required to adopt this material more easily. Knowledge of `boost::bind`, `boost::shared_ptr`, and placeholders are required to get through it. Also, information on linking this recipe with the `libboost_system` library is required.

## How to do it...

Let's extend the code from the previous recipe by adding methods to create connections. A connection would be represented by a `tcp_connection_ptr` class, which must be constructed using only `tasks_processor` (As an analogy, `tasks_processor` is a factory for constructing such connections).

1.  We need a method in `tasks_processor` to create sockets to endpoints (we will be calling them connections):

    [PRE12]

2.  We'll need a lot of header files included as follows:

    [PRE13]

3.  The class `tcp_connection_ptr` is required to manage connections. It owns the socket and manages its lifetime. It's just a thin wrapper around `boost::shared_ptr<boost::asio::ip::tcp::socket>` that hides `Boost.Asio` from the user.

    [PRE14]

4.  The `tcp_connection_ptr` class will need methods for reading data:

    [PRE15]

5.  Methods for writing data are also required:

    [PRE16]

6.  We will also add a method to shutdown the connection:

    [PRE17]

    Now the library user can use the preceding class like this to send the data:

    [PRE18]

    Users may also use it like this to receive data:

    [PRE19]

    And this is how a library user may handle the received data:

    [PRE20]

## How it works...

All the interesting things happen in the `async_*` function's call. Just as in the case of timers, asynchronous calls return immediately without executing a function. They only tell the `boost::asio::io_service` class to execute the callback task after some operation (for example, reading data from the socket) finishes. `io_service` will execute our function in one of the threads that called the `io_service::run()` method.

The following diagram illustrates this:

![How it works...](img/4880OS_06_02.jpg)

Now, let's examine this step-by-step.

The `tcp_connection_ptr` class holds a shared pointer to `boost::asio::ip::tcp::socket`, which is a `Boost.Asio` wrapper around native sockets. We do not want to give a user the ability to use this wrapper directly because it has synchronous methods whose usage we are trying to avoid.

The first constructor accepts a pointer to the socket (and will be used in our next recipe). This constructor won't be used by the user because the `boost::asio::ip::tcp::socket` constructor requires a reference to `boost::asio::io_service`, which is hidden inside `tasks_processor`.

### Note

Of course, some users of our library could be smart enough to create an instance of `boost::asio::io_service`, initialize sockets, and push tasks to that instance. Moving the `Boost.Asio` library's contents into the source file and implementing the **Pimpl idiom** will help you to protect users from shooting their own feet, but we won't implement it here for simplicity. Another way to do things is to declare the `tasks_processor` class as a friend to `tcp_connection_ptr` and make the `tcp_connection_ptr` constructors private.

The second constructor accepts a remote endpoint and a reference to `io_service`. There you may see how the socket is connected to an endpoint using the `socket_->connect(endpoint)` method. Also, this constructor should not be used by the user; the user should use `tasks_processor::create_connection` instead.

Special care should be taken while using the `async_write` and `async_read` functions. Socket and buffer must not be destructed until the asynchronous operation is completed; that is why we bind `shared_ptr` to the functional object when calling the `async_*` functions:

[PRE21]

Binding the shared pointer to the functional object, which will be called at the end of the asynchronous operation, guarantees that at least one instance of `boost::shared_ptr` to the connection and data exists. This means that both connection and data won't be destroyed until the functional object destructor is called.

### Note

`Boost.Asio` may copy functors and that is why we used a `boost::shared_ptr<std::string>` class instead of passing the `std::string` class by value (which would invalidate `boost::asio::buffer(*data)` and lead to a segmentation fault).

## There's more...

Take a closer look at the `finsh_socket_auth_task` function. It checks for `err != boost::asio::error::eof`. This is done because the end of a data input is treated as an error; however, this may also mean that the end host closed the socket, which is not always bad (in our example, we treat it as a nonerror behavior).

`Boost.Asio` is not a part of C++11, but it is proposed for inclusion in C++, and we may see it (or at least some parts of it) included in the next TR.

## See also

*   See the official documentation to `Boost.Asio` for more examples, tutorials, and full references at [http://www.boost.org/doc/libs/1_53_0/doc/html/boost_asio.html](http://www.boost.org/doc/libs/1_53_0/doc/html/boost_asio.html), as well as an example of how to use the UDP and ICMP protocols. For readers familiar with the BSD socket API, the [http://www.boost.org/doc/libs/1_53_0/doc/html/boost_asio/overview/networking/bsd_sockets.html](http://www.boost.org/doc/libs/1_53_0/doc/html/boost_asio/overview/networking/bsd_sockets.html) page provides information about what a BSD call looks like in `Boost.Asio`.
*   Read the *Recording the parameters of function* and *Binding a value as a function parameter* recipes from [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*, for more information about `Boost.Bind`. The *Reference counting of pointers to classes used across methods* recipe from [Chapter 3](ch03.html "Chapter 3. Managing Resources"), *Managing Resources*, will give you more information about what the `boost::shared_ptr` class does.
*   You may also read the book *Boost.Asio C++ Network Programming*, *Packt Publishing*, which describes `Boost.Asio` in more detail.

# Accepting incoming connections

A server side working with a network usually looks like a sequence where we first get data, then process it, and then send the result. Imagine that we are creating some kind of authorization server that will process a huge number of requests per second. In that case, we will need to receive and send data asynchronously and process tasks in multiple threads.

In this recipe, we'll see how to extend our `tasks_processor` class to accept and process incoming connections, and in the next recipe, we'll see how to make it multithreaded.

## Getting ready

This recipe requires a good knowledge of `boost::asio::io_service` basics as described in the first and third recipes of this chapter. Some knowledge of network communications will be of help to you. Knowledge of `boost::bind, boost::function`, `boost::shared_ptr`, and information from at least the two previous recipes is also required. Don't forget to link this example with `libboost_system`.

## How to do it...

Just as in the previous recipes, we'll be adding new methods to our `tasks_processor` class.

1.  First of all, we need to add a function that starts listening on a specified port:

    [PRE22]

2.  We will also add a `std::map` variable that holds all the listeners:

    [PRE23]

3.  And a function to stop the listener:

    [PRE24]

4.  Now we need to take care of the `detail::tcp_listener` class itself. It must have an acceptor:

    [PRE25]

5.  And a function that will be called on a successful accept:

    [PRE26]

6.  This is what a function for starting an accept will look like:

    [PRE27]

7.  A function to stop accepting is written like this:

    [PRE28]

8.  And that is our wrapper function that will be called on a successful accept:

    [PRE29]

## How it works...

The function `add_listener` just checks that we have no listeners on the specified port already, constructs a new `detail::tcp_listener`, and adds it to the `listeners_` list.

When we construct `boost::asio::ip::tcp::acceptor` specifying the endpoint (see step 5), it opens a socket at the specified address.

Calling `async_accept(socket, handler)` for `boost::asio::ip::tcp::acceptor` makes a call to our handler when the incoming connection is accepted. When a new connection comes in, `acceptor_` binds this connection to a socket and pushes the ready task to execute the handler in `task_queue` (in `boost::asio::io_service`). As we understood from the previous recipe, all the `async_*` calls return immediately and `async_accept` is not a special case, so it won't call the handler directly. Let's take a closer look at our handler:

[PRE30]

We need an instance of the current class to be alive when an accepting operation occurs, so we provide a `boost::shared_ptr` variable as a second parameter for `boost::bind` (we do it via `this->shared_from_this()` call). We also need to keep the socket alive, so we provide it as a third parameter. The last parameter is a placeholder (such as `_1` and `_2` for `boost::bind`) that says where the `async_accept` function should put the `error` variable into your method.

Now let's take a closer look at our `handle_accept` method. Calling the `push_task()` method is required to restart accepting our `acceptor_`. After that, we will check for errors and if there are no errors, we will bind the user-provided handler to `tcp_connection_ptr`, make an instance of `task_wrapped` from it (required for correctly handling exceptions and interruption points), and execute it.

Now let's take a look at the `remove_listener()` method. On call, it will find a listener in the list and call `stop()` for it. Inside `stop()`, we will call `close()` for an acceptor, return to the `remove_listener` method, and erase the shared pointer to `tcp_listener` from the map of listeners. After that, shared pointers to `tcp_listener` remain only in one accept task.

When we call `stop()` for an acceptor, all of its asynchronous operations will be canceled and handlers will be called. If we take a look at the `handle_accept` method in the last step, we'll see that in case of an error (or stopped acceptor), no more accepting tasks will be added.

After all the handlers are called, no shared pointer to the acceptor remains and a destructor for `tcp_connection` will be called.

## There's more...

We did not use all the features of the `boost::asio::ip::tcp::acceptor` class. It can bind to a specific IPv6 or IPv4 address, if we provide a specific `boost::asio::ip::tcp::endpoint`. You may also get a native socket via the `native_handle()` method and use some OS-specific calls to tune the behavior. You may set up some options for `acceptor_` by calling `set_option`. For example, this is how you may force an acceptor to reuse the address:

[PRE31]

### Note

Reusing the address provides an ability to restart the server quickly after it was terminated without correct shutdown. After the server was terminated, a socket may be opened for some time and you won't be able to start the server on the same address without the `reuse_address` option.

## See also

*   Starting this chapter from the beginning is a good idea to get much more information about `Boost.Asio`.
*   See the official documentation of `Boost.Asio` for more examples, tutorials, and a complete reference at [http://www.boost.org/doc/libs/1_53_0/doc/html/boost_asio.html](http://www.boost.org/doc/libs/1_53_0/doc/html/boost_asio.html).
*   Read the *Reordering the parameters of function* and *Binding a value as a function parameter* recipes from [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*, for more information about `Boost.Bind`.
*   The *Reference counting of pointers to classes used across methods* recipe in [Chapter 3](ch03.html "Chapter 3. Managing Resources"), *Managing Resources*, will give you more information about what `boost::shared_ptr` does.

# Executing different tasks in parallel

Now it is time to make our `tasks_queue` process tasks in multiple threads. How hard could this be?

## Getting ready

You will need to read the first recipe from this chapter. Some knowledge of multithreading is also required, especially reading the *Manipulating a group of threads* recipe in [Chapter 5](ch05.html "Chapter 5. Multithreading"), *Multithreading*.

## How to do it...

All we need to do is to add the `start_multiple` method to our `tasks_queue` class:

[PRE32]

And now we are able to do much more work, as illustrated in the following diagram:

![How to do it...](img/4880OS_06_03.jpg)

## How it works...

The `boost::asio::io_service::run` method is thread safe. Almost all the methods of `Boost.Asio` are thread safe, so all we need to do is run the `boost::asio::io_service::run` method from different threads.

### Note

If you are executing tasks that modify a common resource, you will need to add mutexes around that resource.

See the call to `boost::thread::hardware_concurrency()`? It returns the number of threads that can be run concurrently. But it is just a hint and may sometimes return a `0` value, which is why we are calling the `std::max` function for it. This ensures that `threads_count` will store at least the value `1`.

### Note

We wrapped `std::max` in parenthesis because some popular compilers define the `min()` and `max()` macros, so we need additional tricks to work-around this.

## There's more...

The `boost::thread::hardware_concurrency()` function is a part of C++11; you will find it in the `<thread>` header of the `std::` namespace. However, not all the `boost::asio` classes are part of C++11 (but they are proposed for inclusion, so we may see them in the next Technical Report (TR) for C++).

## See also

*   See the `Boost.Asio` documentation for more examples and information about different classes at [http://www.boost.org/doc/libs/1_53_0/doc/html/boost_asio.html](http://www.boost.org/doc/libs/1_53_0/doc/html/boost_asio.html)
*   See the `Boost.Thread` documentation for information about `boost::thread_group` and `boost::threads` at [http://www.boost.org/doc/libs/1_53_0/doc/html/thread.html](http://www.boost.org/doc/libs/1_53_0/doc/html/thread.html)
*   Recipes from [Chapter 5](ch05.html "Chapter 5. Multithreading"), *Multithreading*, (especially the last recipe called *Manipulating a group of threads*) will give you information about `Boost.Thread` usage
*   The *Binding a value as a function parameter* recipe will help you to understand the `boost::``bind` function better

# Conveyor tasks processing

Sometimes there is a requirement to process tasks within a specified time interval. Compared to previous recipes, where we were trying to process tasks in the order of their appearance in the queue, this is a big difference.

Consider an example where we are writing a program that connects two subsystems, one of which produces data packets and the other writes modified data to the disk (something like this can be seen in video cameras, sound recorders, and other devices). We need to process data packets one by one, smoothly with the least jitter, and in multiple threads.

Our previous `tasks_queue` was bad at processing tasks in a specified order:

[PRE33]

So how can we solve this?

## Getting ready

Basic knowledge of `boost::asio::io_service` is required for this recipe; read at least the first recipe from this chapter. The *Creating a work_queue class* recipe from [Chapter 5](ch05.html "Chapter 5. Multithreading"), *Multithreading*, is required for understanding this example. Code must be linked against the `boost_thread` library.

## How to do it...

This recipe is based on the code of the `work_queue` class from the *Creating a work_queue class* recipe of [Chapter 5](ch05.html "Chapter 5. Multithreading"), *Multithreading*. We'll make some modifications and will be using a few instances of that class.

1.  Let's start by creating separate queues for data decoding, data compressing, and data sending:

    [PRE34]

2.  Now it is time to refactor the operator `<<` and split it into multiple functions:

    [PRE35]

3.  Our `work_queue` class from [Chapter 5](ch05.html "Chapter 5. Multithreading"), *Multithreading*, had no `stop()` function. Let's add it:

    [PRE36]

    Now the `work_queue` class can be stopped. The `pop_task()` method will return empty tasks if `work_queue` is stopped and no further tasks remain in the `tasks_` variable.

4.  After doing all that is shown in step 3, we can write the code like this:

    [PRE37]

5.  That is all! Now we only need to start the conveyor:

    [PRE38]

6.  The conveyor can be stopped like this:

    [PRE39]

## How it works...

The trick is to split the processing of a single data packet into some equally small subtasks and process them one by one in different `work_queues`. In this example, we can split the data process into data decoding, data compression, and data send.

The processing of six packets, ideally, would look like this:

| Time | Receiving | Decoding | Compressing | Sending |
| --- | --- | --- | --- | --- |
| `Tick 1:` | `packet #1` |   |   |   |
| `Tick 2:` | `packet #2` | `packet #1` |   |   |
| `Tick 3:` | `packet #3` | `packet #2` | `packet #1` |   |
| `Tick 4:` | `packet #4` | `packet #3` | `packet #2` | `packet #1` |
| `Tick 5:` | `packet #5` | `packet #4` | `packet #3` | `packet #2` |
| `Tick 6:` | `packet #6` | `packet #5` | `packet #4` | `packet #3` |
| `Tick 7:` |   | `packet #6` | `packet #5` | `packet #4` |
| `Tick 8:` |   |   | `packet #6` | `packet #5` |
| `Tick 9:` |   |   |   | `packet #6` |

However, our world is not ideal, so some tasks may finish faster than others. For example, receiving may go faster than decoding and in that case, the decoding queue will be holding a set of tasks to be done. We did not use `io_service` in our example because it does not guarantee that posted tasks will be executed in order of their posting.

## There's more...

All the tools used to create a conveyor in this example are available in C++11, so nothing would stop you creating the same things without Boost on a C++11 compatible compiler. However, Boost will make your code more portable, and usable on C++03 compilers.

## See also

*   This technique is well known and used by processor developers. See [http://en.wikipedia.org/wiki/Instruction_pipeline](http://en.wikipedia.org/wiki/Instruction_pipeline). Here you will find a brief description of all the characteristics of the conveyor.
*   The *Creating a work_queue* *class* recipe from [Chapter 5](ch05.html "Chapter 5. Multithreading"), *Multithreading*, and the *Binding a value as a function parameter* recipe from [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*, will give you more information about methods used in this recipe.

# Making a nonblocking barrier

In multithreaded programming, there is an abstraction called **barrier** . It stops execution threads that reach it until the requested number of threads are not blocked on it. After that, all the threads are released and they continue with their execution. Consider the following example of where it can be used.

We want to process different parts of the data in different threads and then send the data:

[PRE40]

The `data_barrier.wait()` method blocks until all the threads fill the data. After that, all the threads are released; the thread with the index `0` will compute data to be sent using `compute_send_data(data)`, while others are again waiting at the barrier as shown in the following diagram:

![Making a nonblocking barrier](img/4880OS_06_04.jpg)

Looks lame, isn't it?

## Getting ready

This recipe requires knowledge of the first recipe of this chapter. Knowledge of `Boost.Bind` and `Boost.Thread` is also required. Code from this recipe requires linking against the `boost_thread` and `boost_system` libraries.

## How to do it...

We do not need to block at all! Let's take a closer look at the example. All we need to do is to post four `fill_data` tasks and make the last finished task call `compute_send_data(data)`.

1.  We'll need the `tasks_processor` class from the first recipe; no changes to it are needed.
2.  Instead of a barrier, we'll be using the atomic variable:

    [PRE41]

3.  Our new runner function will look like this:

    [PRE42]

4.  Only the main function will change slightly, as follows:

    [PRE43]

## How it works...

We don't block as no threads will be waiting for resources. Instead of blocking, we count the tasks that finished filling the data. This is done by the `counter atomic` variable. The last remaining task will have a `counter` variable equal to `data_t::static_size`. It will only need to compute and send the data.

After that, we check for the exit condition (1000 iterations are done), and post the new data by filling tasks to the queue.

## There's more...

Is this solution better? Well, first of all, it scales better:

![There's more...](img/4880OS_06_05.jpg)

This method can also be more effective for situations where a program does a lot of different work. Because no threads are waiting in barriers, free threads may do other work while one of the threads computes and sends the data.

All the tools used for this example are available in C++11 (you'll only need to replace `io_service` inside `tasks_processor` with `work_queue` from [Chapter 5](ch05.html "Chapter 5. Multithreading"), *Multithreading*).

## See also

*   The official documentation for `Boost.Asio` may give you more information about `io_service` usage at [http://www.boost.org/doc/libs/1_53_0/doc/html/boost_asio.html](http://www.boost.org/doc/libs/1_53_0/doc/html/boost_asio.html)
*   See all the `Boost.Function` related recipes from [Chapter 3](ch03.html "Chapter 3. Managing Resources"), *Managing Resources*, and the official documentation at [http://www.boost.org/doc/libs/1_53_0/doc/html/function.html](http://www.boost.org/doc/libs/1_53_0/doc/html/function.html) for getting an idea of how tasks work
*   See the recipes from [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*, related to `Boost.Bind` to get more information about what the `boost::bind` function does, or see the official documentation at [http://www.boost.org/doc/libs/1_53_0/libs/bind/bind.html](http://www.boost.org/doc/libs/1_53_0/libs/bind/bind.html)

# Storing an exception and making a task from it

Processing exceptions is not always trivial and may take a lot of time. Consider the situation where an exception must be serialized and sent by the network. This may take milliseconds and a few thousand lines of code. After the exception is caught is not always the best time and place to process it.

So, can we store exceptions and delay their processing?

## Getting ready

This recipe requires knowledge of `boost::asio::io_service`, which was described in the first recipe of this chapter. Knowledge of `Boost.Bind` is also required.

## How to do it...

All we need is to have the ability to store exceptions and pass them between threads just like a normal variable.

1.  Let's start with the function that processes exceptions. In our case, it will only be outputting the exception information to the console:

    [PRE44]

2.  Now we will write some functions to demonstrate how exceptions work:

    [PRE45]

3.  Now, if we run the example like this:

    [PRE46]

    We'll get the following output:

    [PRE47]

## How it works...

The `Boost.Exception` library provides an ability to store and rethrow exceptions. The `boost::current_exception()` method must be called from inside the `catch()` block, and it returns an object of the type `boost::exception_ptr`. So in `func_test1()`, the `boost::bad_lexical_cast` exception will be thrown, which will be returned by `boost::current_exception()`, and a task (a functional object) will be created from that exception and the `process_exception` function's pointer.

The `process_exception` function will re-throw the exception (the only way to restore the exception type from `boost::exception_ptr` is to rethrow it using `boost::rethrow_exception(exc)` and then catch it by specifying the exception type).

In `func_test2`, we are throwing a `std::logic_error` exception using the `BOOST_THROW_EXCEPTION` macro. This macro does a lot of useful work: it checks that our exception is derived from `std::exception` and adds information to our exception about the source filename, function name, and the number of the line of code where the exception was thrown. So when an exception is re-thrown and caught by `catch(...)`, `boost::current_exception_diagnostic_information()`, we will be able to output much more information about it.

## There's more...

Usually, `exception_ptr` is used to pass exceptions between threads. For example:

[PRE48]

The `boost::exception_ptr` class may allocate memory through heap multiple times, uses atomics, and implements some of the operations by rethrowing and catching exceptions. Try not to use it without an actual need.

C++11 has adopted `boost::current_exception`, `boost::rethrow_exception`, and `boost::exception_ptr`. You will find them in the `<exception>` header of the `std::` namespace. However, the `BOOST_THROW_EXCEPTION` and `boost::current_exception_diagnostic_information()` methods are not in C++11, so you'll need to realize them on your own (or just use the Boost versions).

## See also

*   The official documentation for `Boost.Exception` contains a lot of useful information about implementation and restrictions at [http://www.boost.org/doc/libs/1_53_0/libs/exception/doc/boost-exception.html](http://www.boost.org/doc/libs/1_53_0/libs/exception/doc/boost-exception.html). You may also find some information that is not covered in this recipe (for example, how to add additional information to an already thrown exception).
*   The first recipe from this chapter will give you information about the `tasks_processor` class. Recipes *Binding a value as a function parameter* from [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*, and *Converting strings to numbers* from [Chapter 2](ch02.html "Chapter 2. Converting Data"), *Converting Data*, will help you with `Boost.Bind` and `Boost.LexicalCast`.

# Getting and processing system signals as tasks

When writing some server applications (especially for Linux OS), catching and processing signals is required. Usually, all the signal handlers are set up at server start and do not change during the application's execution.

The goal of this recipe is to make our `tasks_processor` class capable of processing signals.

## Getting ready

We will need code from the first recipe of this chapter. Good knowledge of `Boost.Bind` and `Boost.Function` is also required.

## How to do it...

This recipe is similar to previous ones; we have some signal handlers, functions to register them, and some support code.

1.  Let's start with including the following headers:

    [PRE49]

2.  Now we add a member for signals processing to the `tasks_processor` class:

    [PRE50]

3.  The function that will be called upon signal capture is as follows:

    [PRE51]

4.  Do not forget to initialize the `signals_` member in the `tasks_processor` constructor:

    [PRE52]

5.  And now we need a function for registering the signals handler:

    [PRE53]

    That's all. Now we are ready to process signals. Following is a test program:

    [PRE54]

    This will give the following output:

    [PRE55]

## How it works...

Nothing is difficult here (compared to some previous recipes from this chapter). The `register_signals_handler` function adds the signal numbers that will be processed. It is done via a call to the `boost::asio::signal_set::add` function for each element of the `signals_to_wait` vector (we do it using `std::for_each` and some magic of `boost::bind`).

Next, the instruction makes `signals_ member` wait for the signal and calls the `tasks_processor::handle_signals` member function for `this` on the signal capture. The `tasks_processor::handle_signals` function checks for errors and if there is no error, it creates a functional object by referring to `users_signal_handler_` and the signal number. This functional object will be wrapped in the `task_wrapped` structure (that handles all the exceptions) and executed.

After that, we make `signals_ member` wait for a signal again.

## There's more...

When a thread-safe dynamic adding and removing of signals is required, we may modify this example to look like `detail::timer_task` from the *Making timers and processing timer events as tasks* recipe of this chapter. When multiple `boost::asio::signal_set` objects are registered as waiting on the same signals, a handler from each of `signal_set` will be called on a single signal.

C++ has been capable of processing signals for a long time using the `signal` function from the `<csignal>` header. However, it is incapable of using functional objects (which is a huge disadvantage).

## See also

*   The *Binding a value as a function parameter* and *Reordering the parameters of function* recipes from [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*, provide a lot of information about `boost::bind`. The official documentation may also help: [http://www.boost.org/doc/libs/1_53_0/libs/bind/bind.html](http://www.boost.org/doc/libs/1_53_0/libs/bind/bind.html)
*   The *Storing any functional object in a variable* recipe (on `Boost.Function`) from [Chapter 3](ch03.html "Chapter 3. Managing Resources"), *Managing Resources*, provides information about `boost::function`.
*   See the official `Boost.Asio` documentation has more information and examples on `boost::asio::signal_set` and other features of this great library at [http://www.boost.org/doc/libs/1_53_0/doc/html/boost_asio.html](http://www.boost.org/doc/libs/1_53_0/doc/html/boost_asio.html).