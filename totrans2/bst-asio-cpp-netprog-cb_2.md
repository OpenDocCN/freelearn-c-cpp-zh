# Chapter 2. I/O Operations

In this chapter, we will cover the following recipes:

*   Using fixed length I/O buffers
*   Using extensible stream-oriented I/O buffers
*   Writing to a TCP socket synchronously
*   Reading from a TCP socket synchronously
*   Writing to a TCP socket asynchronously
*   Reading from a TCP socket asynchronously
*   Canceling asynchronous operations
*   Shutting down and closing a socket

# Introduction

I/O operations are the key operations in the networking infrastructure of any distributed application. They are directly involved in the process of data exchange. Input operations are used to receive data from remote applications, whereas output operations allow sending data to them.

In this chapter, we will see several recipes that show how to perform I/O operations and other operations related to them. In addition to this, we'll see how to use some classes provided by Boost.Asio, which are used in conjunction with I/O operations.

The following is the short summary and introduction to the topics discussed in this chapter.

## I/O buffers

Network programming is all about organizing inter-process communication over a computer network. **Communication** in this context implies exchanging data between two or more processes. From the perspective of a process that participates in such communication, the process performs I/O operations, sending data to and receiving it from other participating processes.

Like any other type of I/O, the network I/O involves using memory buffers, which are contiguous blocks of memory allocated in the process's address space used to store the data. When doing any sort of input operation (for example, reading some data from a file, a pipe, or a remote computer over the network), the data arrives at the process and must be stored somewhere in its address space so that it is available for further processing. That is, when the buffer comes in handy. Before performing an input operation, the buffer is allocated and then used as a data destination point during the operation. When the input operation is completed, the buffer contains input data, which can be processed by the application. Likewise, before performing the output operation, the data must be prepared and put into an output buffer, which is then used in the output operation, where it plays the role of the data source.

Apparently, the buffers are essential ingredients of any application that performs any type of I/O, including the network I/O. That's why it is critical for the developer who develops a distributed application to know how to allocate and prepare the I/O buffers to use them in the I/O operations.

## Synchronous and asynchronous I/O operations

Boost.Asio supports two types of I/O operations: synchronous and asynchronous. Synchronous operations block the thread of execution invoking them and unblock only when the operation is finished. Hence, the name of this type of operation: synchronous.

The second type is an asynchronous operation. When an asynchronous operation is initiated, it is associated with a callback function or functor, which is invoked by the Boost.Asio library when the operation is finished. These types of I/O operations provide great flexibility, but may significantly complicate the code. The initiation of the operation is simple and doesn't block the thread of execution, which allows us to use the thread to run other tasks, while the asynchronous operation is being run in the background.

The Boost.Asio library is implemented as a framework, which exploits an **inversion of control** approach. After one or more asynchronous operations are initiated, the application hands over one of its threads of execution to the library, and the latter uses this thread to run the event loop and invoke the callbacks provided by the application to notify it about the completion of the previously initiated asynchronous operation. The results of asynchronous operations are passed to the callback as arguments.

## Additional operations

In addition to this, we are going to consider such operations as canceling asynchronous operations, shutting down, and closing a socket.

The ability to cancel a previously initiated asynchronous operation is very important. It allows the application to state that the previously initiated operation is not relevant anymore, which may save the application's resources (both CPU and memory), that otherwise (in case, the operation would continue its execution even after it was known that nobody is interested in it anymore) would be unavoidably wasted.

Shutting down the socket is useful if there is a need for one part of the distributed application to inform the other part that the whole message has been sent, when the application layer protocol does not provide us with other means to indicate the message boundary.

As with any other operating system resource, a socket should be returned back to the operating system when it is not needed anymore by the application. A closing operation allows us to do so.

# Using fixed length I/O buffers

Fixed length I/O buffers are usually used with I/O operations and play the role of either a data source or destination when the size of the message to be sent or received is known. For example, this can be a constant array of chars allocated on a stack, which contain a string that represents the request to be sent to the server. Or, this can be a writable buffer allocated in the free memory, which is used as a data destination point, when reading data from a socket.

In this recipe, we'll see how to represent fixed length buffers so that they can be used with Boost.Asio I/O operations.

## How to do it…

In Boost.Asio, a fixed length buffer is represented by one of the two classes: `asio::mutable_buffer` or `asio::const_buffer`. Both these classes represent a contiguous block of memory that is specified by the address of the first byte of the block and its size in bytes. As the names of these classes suggest, `asio::mutable_buffer` represents a writable buffer, whereas `asio::const_buffer` represents a read-only one.

However, neither the `asio::mutable_buffer` nor `asio::const_buffer` classes are used in Boost.Asio I/O functions and methods directly. Instead, the `MutableBufferSequence` and `ConstBufferSequence` concepts are introduced.

The `MutableBufferSequence` concept specifies an object that represents a collection of the `asio::mutable_buffer` objects. Correspondingly, the `ConstBufferSequence` concept specifies an object that represents a collection of the `asio::const_buffer` objects. Boost.Asio functions and methods that perform I/O operations accept objects that satisfy the requirements of either the `MutableBufferSequence` or `ConstBufferSequence` concept as their arguments that represent buffers.

### Note

A complete specification of the `MutableBufferSequence` and `ConstBufferSequence` concepts are available in the Boost.Asio documentation section, which can be found at the following links:

*   Refer to [http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/MutableBufferSequence.html](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/MutableBufferSequence.html) for `MutableBufferSequence`
*   Refer to [http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/ConstBufferSequence.html](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/ConstBufferSequence.html) for `ConstBufferSequence`

Although in most use cases, a single buffer is involved in a single I/O operation, in some specific circumstances (for example, in a memory-constrained environment), a developer may want to use a composite buffer that comprises multiple smaller simple buffers distributed over the process's address space. Boost.Asio I/O functions and methods are designed to work with composite buffers that are represented as a collection of buffers that fulfill the requirements of either the `MutableBufferSequence` or `ConstBufferSequence` concept.

For instance, an object of the `std::vector<asio::mutable_buffer>` class satisfies the requirements of the `MutableBufferSequence` concept, and therefore, it can be used to represent a composite buffer in I/O-related functions and methods.

So, now we know that if we have a buffer that is represented as an object of the `asio::mutable_buffer` or `asio::const_buffer` class, we still can't use it with I/O-related functions or methods provided by Boost.Asio. The buffer must be represented as an object, satisfying the requirements of either the `MutableBufferSequence` or `ConstBufferSequence` concept, respectively. To do this, we for example could create a collection of buffer objects consisting of a single buffer by instantiating an object of the `std::vector<asio::mutable_buffer>` class and placing our buffer object into it. Now that the buffer is part of the collection, satisfying the `MutableBufferSequence` requirements can be used in I/O operations.

However, although this method is fine to create composite buffers consisting of two or more simple buffers, it looks overly complex when it comes to such simple tasks as representing a single simple buffer so that it can be used with Boost.Asio I/O functions or methods. Fortunately, Boost.Asio provides us with a way to simplify the usage of single buffers with I/O-related functions and methods.

The `asio::buffer()` free function has 28 overloads that accept a variety of representations of a buffer and return an object of either the `asio::mutable_buffers_1` or `asio::const_buffers_1` classes. If the buffer argument passed to the `asio::buffer()` function is a read-only type, the function returns an object of the `asio::const_buffers_1` class; otherwise, an object of the `asio::mutable_buffers_1` class is returned.

The `asio::mutable_buffers_1` and `asio::const_buffers_1` classes are *adapters* of the `asio::mutable_buffer` and `asio::const_buffer` classes, respectively. They provide an interface and behavior that satisfy the requirements of the `MutableBufferSequence` and `ConstBufferSequence` concepts, which allows us to pass these adapters as arguments to Boost.Asio I/O functions and methods.

Let's consider two algorithms and corresponding code samples that describe how to prepare a memory buffer that can be used with Boost.Asio I/O operations. The first algorithm deals with buffers intended to be used for an output operation and the second one is used for an input operation.

### Preparing a buffer for an output operation

The following algorithm and corresponding code sample describes how to prepare a buffer that can be used with the Boost.Asio socket's method that performs an output operation such as `asio::ip::tcp::socket::send()` or the `asio::write()`free function:

1.  Allocate a buffer. Note that this step does not involve any functionality or data types from Boost.Asio.
2.  Fill the buffer with the data that is to be used as the output.
3.  Represent the buffer as an object that satisfies the `ConstBufferSequence` concept's requirements.
4.  The buffer is ready to be used with Boost.Asio output methods and functions.

Let's say we want to send a string `Hello` to the remote application. Before we send the data using Boost.Asio, we need to properly represent the buffer. This is how we do this in the following code:

[PRE0]

### Preparing a buffer for an input operation

The following algorithm and corresponding code sample describes how to prepare the buffer that can be used with the Boost.Asios socket's method that performs an input operation such as `asio::ip::tcp::socket::receive()` or the `asio::read()`free function:

1.  Allocate a buffer. The size of the buffer must be big enough to fit the block of data to be received. Note that this step does not involve any functionalities or data types from Boost.Asio.
2.  Represent the buffer using an object that satisfies the `MutableBufferSequence` concept's requirements.
3.  The buffer is ready to be used with Boost.Asio input methods and functions.

Let's say we want to receive a block of data from the server. To do this, we first need to prepare a buffer where the data will be stored. This is how we do this in the following code:

[PRE1]

## How it works…

Both the samples look quite simple and straightforward; however, they contain some subtleties, which are important to understand so that we can properly use buffers with Boost.Asio. In this section, we'll see how each sample works in detail.

### Preparing a buffer for an output operation

Let's consider the first code sample that demonstrates how to prepare a buffer that can be used with Boost.Asio output methods and functions. The `main()`entry point function starts with instantiating the object of the `std::string` class. Because we want to send a string of text, `std::string` is a good candidate to store this kind of data. In the next line, the string object is assigned a value of `Hello`. This is where the buffer is allocated and filled with data. This line implements steps 1 and 2 of the algorithm.

Next, before the buffer can be used with Boost.Asio I/O methods and functions, it must be properly represented. To better understand why this is needed, let's take a look at one of the Boost.Asio output functions. Here is the declaration of the `send()`method of the Boost.Asio class that represents a TCP socket:

[PRE2]

As we can see, this is a template method, and it accepts an object that satisfies the requirements of the `ConstBufferSeqenece` concept as its argument that represents the buffer. A suitable object is a composite object that represents a collection of objects of the `asio::const_buffer` class and provides a typical collection interface that supports an iteration over its elements. For example, an object of the `std::vector<asio::const_buffer>` class is suitable for being used as the argument of the `send()` method, but objects of the `std::string` or `asio::const_bufer` class are not.

In order to use our `std::string` object with the `send()`method of the class that represents a TCP socket, we can do something like this:

[PRE3]

The object named `buffer_sequence` in the preceding snippet satisfies the `ConstBufferSequence` concept's requirements, and therefore, it can be used as an argument for the `send()` method of the socket object. However, this approach is very complex. Instead, we use the `asio::buffer()`function provided by Boost.Asio to obtain *adaptor* objects, which we can directly use in I/O operations:

[PRE4]

After the adaptor object is instantiated, it can be used with Boost.Asio output operations to represent the output buffer.

### Preparing a buffer for an input operation

The second code sample is very similar to the first one. The main difference is that the buffer is allocated but is not filled with data because its purpose is different. This time, the buffer is intended to receive the data from a remote application during the input operation.

With an output buffer, an input buffer must be properly represented so that it can be used with Boost.Asio I/O methods and functions. However, in this case, the buffer must be represented as an object that meets the requirements of the `MutableBufferSequence` concept. Contrary to `ConstBufferSequence`, this concept represents the collection of *mutable* buffers, that is, those that can be written to. Here, we use the `buffer()` function, which helps us create the required representation of the buffer. The object of the `mutable_buffers_1` adaptor class represents a single mutable buffer and meets the `MutableBufferSequence` concept's requirements.

In the first step, the buffer is allocated. In this case, the buffer is an array of chars allocated in the free memory. In the next step, the adaptor object is instantiated that can be used with both the input and output operations.

### Note

**Buffer ownership**

It's important to note that neither the classes that represent the buffers nor the adaptor classes provided by Boost.Asio that we've considered (namely, `asio::mutable_buffer`, `asio::const_buffer`, `asio::mutable_buffers_1`, and `asio::const_buffers_1`) take ownership of the underlying raw buffer. These classes only provide the interface to the buffer and don't control its lifetime.

## See also

*   The *Writing to a TCP socket synchronously* recipe demonstrates how to write data to the socket from a fixed-length buffer
*   The *Reading from a TCP socket synchronously* recipe demonstrates how to read data from the socket to a fixed-length buffer
*   The *Using composite buffers for scatter/gather operations* recipe in [Chapter 6](ch06.html "Chapter 6. Other Topics"), *Other Topics*, provides more information on composite buffers and demonstrates how to use them

# Using extensible stream-oriented I/O buffers

Extensible buffers are those buffers that dynamically increase their size when new data is written to them. They are usually used to read data from sockets when the size of the incoming message is unknown.

Some application layer protocols do not define the exact size of the message. Instead, the boundary of the message is represented by a specific sequence of symbols at the end of the message itself or by a transport protocol service message **end of file** (**EOF**) issued by the sender after it finishes sending the message.

For example, according to the HTTP protocol, the header section of the request and response messages don't have a fixed length and its boundary is represented by a sequence of four ASCII symbols, `<CR><LF><CR><LF>`, which is part of the message. In such cases, dynamically extensible buffers and functions that can work with them, which are provided by the Boost.Asio library, are very useful.

In this recipe, we will see how to instantiate extensible buffers and how to read and write data to and from them. To see how these buffers can be used with I/O-related methods and functions provided by Boost.Asio, refer to the corresponding recipes dedicated to I/O operations listed in the *See also* section.

## How to do it…

Extensible stream-oriented buffers are represented in Boost.Asio with the `asio::streambuf` class, which is a `typedef` for `asio::basic_streambuf`:

[PRE5]

The `asio::basic_streambuf<>` class is inherited from `std::streambuf`, which means that it can be used as a stream buffer for STL stream classes. In addition to this, several I/O functions provided by Boost.Asio deal with buffers that are represented as objects of this class.

We can work with an object of the `asio::streambuf` class just like we would work with any stream buffer class that is inherited from the `std::streambuf` class. For example, we can assign this object to a stream (for example, `std::istream`, `std::ostream`, or `std::iostream`, depending on our needs), and then, use stream's `operator<<()` and `operator>>()` operators to write and read data to and from the stream.

Let's consider a sample application in which an object of `asio::streambuf` is instantiated, some data is written to it, and then the data is read back from the buffer to an object of the `std::string` class:

[PRE6]

Note that this sample does not contain any network I/O operations because it focuses on the `asio::streambuf` class itself and its operations rather than on how to use this class with I/O operations.

## How it works…

The `main()` application entry point function begins with instantiating an object of the `asio::streambuf` class named `buf`. Next, the output stream object of the `std::ostream` class is instantiated. The `buf` object is used as a *stream buffer* for the output stream.

In the next line, the `Message1\nMessage2` sample data string is written to the output stream object, which in turn redirects the data to the `buf` stream buffer.

Usually, in a typical client or server application, the data will be written to the `buf` stream buffer by the Boost.Asio input function such as `asio::read()`, which accepts a stream buffer object as an argument and reads data from the socket to that buffer.

Now, we want to read the data back from the stream buffer. To do this, we allocate an input stream and pass the `buf` object as a stream buffer argument to its constructor. After this, we allocate a string object named `message1`, and then, use the `std::getline` function to read part of the string currently stored in the `buf` stream buffer until the delimiter symbol, `\n`.

As a result, the `string1` object contains the `Message1` string and the `buf` stream buffer contains the rest of the initial string after the delimiter symbol, that is, `Message2`.

## See also

*   The *Reading from a TCP socket asynchronously* recipe demonstrates how to read data from the socket to an extensible stream-oriented buffer

# Writing to a TCP socket synchronously

Writing to a TCP socket is an output operation that is used to send data to the remote application connected to this socket. Synchronous writing is the simplest way to send the data using a socket provided by Boost.Asio. The methods and functions that perform synchronous writing to the socket block the thread of execution and do not return until the data (at least some amount of data) is written to the socket or an error occurs.

In this recipe, we will see how to write data to a TCP socket synchronously.

## How to do it…

The most basic way to write to the socket provided by the Boost.Asio library is to use the `write_some()` method of the `asio::ip::tcp::socket` class. Here is the declaration of one of the method's overloads:

[PRE7]

This method accepts an object that represents a composite buffer as an argument, and as its name suggests, writes *some* amount of data from the buffer to the socket. If the method succeeds, the return value indicates the number of bytes written. The point to emphasize here is that the method may *not* send all the data provided to it through the `buffers` argument. The method only guarantees that at least one byte will be written if an error does not occur. This means that, in a general case, in order to write all the data from the buffer to the socket, we may need to call this method several times.

The following algorithm describes the steps required to synchronously write data to a TCP socket in a distributed application:

1.  In a client application, allocate, open, and connect an active TCP socket. In a server application, obtain a connected active TCP socket by accepting a connection request using an acceptor socket.
2.  Allocate the buffer and fill it with data that is to be written to the socket.
3.  In a loop, call the socket's `write_some()` method as many times as it is needed to send all the data available in the buffer.

The following code sample demonstrates a client application, which operates according to the algorithm:

[PRE8]

Although in the presented code sample, writing to the socket is performed in the context of an application that acts as a client, the same approach can be used to write to the socket in a server application.

## How it works…

The `main()`application entry point function is quite simple. It allocates a socket, opens, and synchronously connects it to a remote application. Then, the `writeToSocket()` function is called and the socket object is passed to it as an argument. In addition to this, the `main()`function contains a `try-catch` block intended to catch and handle exceptions that may be thrown by Boost.Asio methods and functions.

The interesting part in the sample is the `writeToSocket()`function that performs synchronous writing to the socket. It accepts a reference to the socket object as an argument. Its precondition is that the socket passed to it is already connected; otherwise, the function fails.

The function begins with allocating and filling the buffer. In this sample, we use an ASCII string as data that is to be written to the socket, and, therefore, we allocate an object of the `std::string` class and assign it a value of `Hello`, which we will use as a dummy message that will be written to the socket.

Then, the variable named `total_bytes_written` is defined and its value is set to `0`. This variable is used as a counter that stores the count of bytes already written to the socket.

Next, the loop is run in which the socket's `write_some()` method is called. Except for the degenerate case when the buffer is empty (that is, the `buf.length()` method returns a value of `0`), at least one iteration of the loop is executed and the `write_some()` method is called at least once. Let's take a closer look at the loop:

[PRE9]

The termination condition evaluates to `true` when the value of the `total_bytes_written` variable is equal to the size of the buffer, that is, when all the bytes available in the buffer have been written to the socket. In each iteration of the loop, the value of the `total_bytes_written` variable is increased by the value returned by the `write_some()` method, which is equal to the number of bytes written during this method call.

Each time the `write_some()` method is called, the argument passed to it is adjusted. The start byte of the buffer is shifted by the value of `total_bytes_written` as compared to the original buffer (because the previous bytes have already been sent by preceding calls to the `write_some()` method) and the size of the buffer is decreased by the same value, correspondingly.

After the loop terminates, all the data from the buffer is written to the socket and the `writeToSocket()` function returns.

It's worth noting that the amount of bytes written to the socket during a single call to the `write_some()` method depends on several factors. In the general case, it is not known to the developer; and therefore, it should not be accounted for. A demonstrated solution is independent of this value and calls the `write_some()` method as many times as needed to write all the data available in the buffer to the socket.

### Alternative – the send() method

The `asio::ip::tcp::socket` class contains another method to synchronously write data to the socket named `send()`. There are three overloads of this method. One of them is equivalent to the `write_some()` method, as described earlier. It has exactly the same signature and provides exactly the same functionality. These methods are synonyms in a sense.

The second overload accepts one additional argument as compared to the `write_some()` method. Let's take a look at it:

[PRE10]

This additional argument is named `flags`. It can be used to specify a bit mask, representing flags that control the operation. Because these flags are used quite rarely, we won't consider them in this book. Refer to the Boost.Asio documentation to find out more information on this topic.

The third overload is equivalent to the second one, but it doesn't throw exceptions in case of a failure. Instead, the error information is returned by means of an additional method's output argument of the `boost::system::error_code` type.

## There's more...

Writing to a socket using the socket's `write_some()` method seems very complex for such a simple operation. Even if we want to send a small message that consists of several bytes, we must use a loop, a variable to keep track of how many bytes have already been written, and properly construct a buffer for each iteration of the loop. This approach is error-prone and makes the code more difficult to understand.

Fortunately, Boost.Asio provides a free function, which simplifies writing to a socket. This function is called `asio::write()`. Let's take a look at one of its overloads:

[PRE11]

This function accepts two arguments. The first of them named `s` is a reference to an object that satisfies the requirements of the `SyncWriteStream` concept. For a complete list of the requirements, refer to the corresponding Boost.Asio documentation section at [http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/SyncWriteStream.html](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/SyncWriteStream.html). The object of the `asio::ip::tcp::socket` class that represents a TCP socket satisfies these requirements and, therefore, can be used as the first argument of the function. The second argument named `buffers` represents the buffer (simple or composite) and contains data that is to be written to the socket.

In contrast to the socket object's `write_some()` method, which writes *some* amount of data from the buffer to the socket, the `asio::write()` function writes all the data available in the buffer. This simplifies writing to the socket and makes the code shorter and cleaner.

This is how our `writeToSocket()` function from a previous sample would look like if we used the `asio::write()` function instead of the socket object's `write_some()` method to write data to the socket:

[PRE12]

The `asio::write()` function is implemented in a similar way as the original `writeToSocket()` function is implemented by means of several calls to the socket object's `write_some()` method in a loop.

### Note

Note that the `asio::write()` function has seven more overloads on the top of the one we just considered. Some of them may be very useful in specific cases. Refer to the Boost.Asio documentation to find out more about this function at [http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/write.html](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/write.html).

## See also

*   The *Implementing a synchronous TCP client* recipe in [Chapter 3](ch03.html "Chapter 3. Implementing Client Applications"), *Implementing Client Applications*, demonstrates how to implement a synchronous TCP client that performs synchronous writing to send request messages to the server
*   The *Implementing a synchronous iterative TCP server* recipe in [Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*, demonstrates how to implement a synchronous TCP server that performs synchronous writing to send response messages to the client

# Reading from a TCP socket synchronously

Reading from a TCP socket is an input operation that is used to receive data sent by the remote application connected to this socket. Synchronous reading is the simplest way to receive the data using a socket provided by Boost.Asio. The methods and functions that perform synchronous reading from the socket blocks the thread of execution and doesn't return until the data (at least some amount of data) is read from the socket or an error occurs.

In this recipe, we will see how to read data from a TCP socket synchronously.

## How to do it…

The most basic way to read data from the socket provided by the Boost.Asio library is the `read_some()` method of the `asio::ip::tcp::socket` class. Let's take a look at one of the method's overloads:

[PRE13]

This method accepts an object that represents a writable buffer (single or composite) as an argument, and as its name suggests, reads *some* amount of data from the socket to the buffer. If the method succeeds, the return value indicates the number of bytes read. It's important to note that there is no way to control how many bytes the method will read. The method only guarantees that at least one byte will be read if an error does not occur. This means that, in a general case, in order to read a certain amount of data from the socket, we may need to call the method several times.

The following algorithm describes the steps required to synchronously read data from a TCP socket in a distributed application:

1.  In a client application, allocate, open, and connect an active TCP socket. In a server application, obtain a connected active TCP socket by accepting a connection request using an acceptor socket.
2.  Allocate the buffer of a sufficient size to fit in the expected message to be read.
3.  In a loop, call the socket's `read_some()` method as many times as it is needed to read the message.

The following code sample demonstrates a client application, which operates according to the algorithm:

[PRE14]

Although in the presented code sample, reading from a socket is performed in the context of an application that acts as a client, the same approach can be used to read data from the socket in a server application.

## How it works…

The `main()`application entry point function is quite simple. First, it allocates a TCP socket, opens, and synchronously connects it to a remote application. Then, the `readFromSocket()` function is called and the socket object is passed to it as an argument. In addition to this, the `main()` function contains a `try-catch` block intended to catch and handle exceptions that may be thrown by Boost.Asio methods and functions.

The interesting part in the sample is the `readFromSocket()` function that performs synchronous reading from the socket. It accepts a reference to the socket object as an input argument. Its precondition is that the socket passed to it as an argument must be connected; otherwise, the function fails.

The function begins with allocating a buffer named `buf`. The size of the buffer is chosen to be 7 bytes. This is because in our sample, we expect to receive exactly a 7 bytes long message from a remote application.

Then, a variable named `total_bytes_read` is defined and its value is set to `0`. This variable is used as a counter that keeps the count of the total number of bytes read from the socket.

Next, the loop is run in which the socket's `read_some()` method is called. Let's take a closer look at the loop:

[PRE15]

The termination condition evaluates to `true` when the value of the `total_bytes_read` variable is equal to the size of the expected message, that is, when the whole message has been read from the socket. In each iteration of the loop, the value of the `total_bytes_read` variable is increased by the value returned by the `read_some()` method, which is equal to the number of bytes read during this method call.

Each time the `read_some()` method is called, the input buffer passed to it is adjusted. The start byte of the buffer is shifted by the value of `total_bytes_read` as compared to the original buffer (because the preceding part of the buffer has already been filled with data read from the socket during preceding calls to the `read_some()` method) and the size of the buffer is decreased by the same value, correspondingly.

After the loop terminates, all the data expected to be read from the socket is now in the buffer.

The `readFromSocket()` function ends with instantiating an object of the `std::string` class from the received buffer and returning it to the caller.

It's worth noting that the amount of bytes read from the socket during a single call to the `read_some()` method depends on several factors. In a general case, it is not known to the developer; and, therefore, it should not be accounted for. The proposed solution is independent of this value and calls the `read_some()` method as many times as needed to read all the data from the socket.

### Alternative – the receive() method

The `asio::ip::tcp::socket` class contains another method to read data from the socket synchronously called `receive()`. There are three overloads of this method. One of them is equivalent to the `read_some()` method, as described earlier. It has exactly the same signature and provides exactly the same functionality. These methods are synonyms in a sense.

The second overload accepts one additional argument as compared to the `read_some()` method. Let's take a look at it:

[PRE16]

This additional argument is named `flags`. It can be used to specify a bit mask, representing flags that control the operation. Because these flags are rarely used, we won't consider them in this book. Refer to the Boost.Asio documentation to find out more about this topic.

The third overload is equivalent to the second one, but it doesn't throw exceptions in case of a failure. Instead, the error information is returned by means of an additional output argument of the `boost::system::error_code` type.

## There's more...

Reading from a socket using the socket's `read_some()` method seems very complex for such a simple operation. This approach requires us to use a loop, a variable to keep track of how many bytes have already been read, and properly construct a buffer for each iteration of the loop. This approach is error-prone and makes the code more difficult to understand and maintain.

Fortunately, Boost.Asio provides a family of free functions that simplify synchronous reading of data from a socket in different contexts. There are three such functions, each having several overloads, that provide a rich functionality that facilitates reading data from a socket.

### The asio::read() function

The `asio::read()` function is the simplest one out of the three. Let's take a look at the declaration of one of its overloads:

[PRE17]

This function accepts two arguments. The first of them named `s` is a reference to an object that satisfies the requirements of the `SyncReadStream` concept. For a complete list of the requirements, refer to the corresponding Boost.Asio documentation section available at [http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/SyncReadStream.html](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/SyncReadStream.html). The object of the `asio::ip::tcp::socket` class that represents a TCP socket satisfies these requirements and, therefore, can be used as the first argument of the function. The second argument named `buffers` represents a buffer (simple or composite) to which the data will be read from the socket.

In contrast to the socket's `read_some()` method, which reads *some* amount of data from the socket to the buffer, the `asio::read()` function, during a single call, reads data from the socket until the buffer passed to it as an argument is filled or an error occurs. This simplifies reading from the socket and makes the code shorter and cleaner.

This is how our `readFromSocket()` function from the previous sample would look like if we used the `asio::read()` function instead of the socket object's `read_some()` method to read data from the socket:

[PRE18]

In the preceding sample, a call to the `asio::read()` function will block the thread of execution until exactly 7 bytes are read or an error occurs. The benefits of this approach over the socket's `read_some()` method are obvious.

### Note

The `asio::read()` function has several overloads, which provide flexibility in specific contexts. Refer to the corresponding section of the Boost.Asio documentation to find out more about this function at [http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/read.html](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/read.html).

### The asio::read_until() function

The `asio::read_until()` function provides a way to read data from a socket until a specified pattern is encountered in the data. There are eight overloads of this function. Let's consider one of them:

[PRE19]

This function accepts three arguments. The first of them named `s` is a reference to an object that satisfies the requirements of the `SyncReadStream` concept. For a complete list of the requirements, refer to the corresponding Boost.Asio documentation section at [http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/SyncReadStream.html](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/SyncReadStream.html). The object of the `asio::ip::tcp::socket` class that represents a TCP socket satisfies these requirements and, therefore, can be used as the first argument of the function.

The second argument named `b` represents a stream-oriented extensible buffer in which the data will be read. The last argument named `delim` specifies a delimiter character.

The `asio::read_until()` function will read data from the `s` socket to the buffer `b` until it encounters a character specified by the `delim` argument in the read portion of the data. When the specified character is encountered, the function returns.

It's important to note that the `asio::read_until()` function is implemented so that it reads the data from the socket by blocks of variable sizes (internally it uses the socket's `read_some()` method to read the data). When the function returns, the buffer `b` may contain some symbols after the delimiter symbol. This may happen if the remote application sends some more data after the delimiter symbol (for example, it may send two messages in a row, each having a delimiter symbol in the end). In other words, when the `asio::read_until()` function returns successfully, it is guaranteed that the buffer `b` contains at least one delimiter symbol but may contain more. It is the developer's responsibility to parse the data in the buffer and handle the situation when it contains data after the delimiter symbol.

This is how we will implement our `readFromSocket()` function if we want to read all the data from a socket until a specific symbol is encountered. Let's assume the message delimiter to be a new line ASCII symbol, `\n`:

[PRE20]

This example is quite simple and straightforward. Because `buf` may contain more symbols after the delimiter symbol, we use the `std::getline()` function to extract the messages of interest before the delimiter symbol and put them into the `message` string object, which is then returned to the caller.

### Note

The `read_until()` function has several overloads, which provide more sophisticated ways to specify termination conditions, such as string delimiters, regular expressions, or functors. Refer to the corresponding Boost.Asio documentation section to find out more about this topic at [http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/read_until.html](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/read_until.html).

### The asio::read_at() function

The `asio::read_at()` function provides a way to read data from a socket, starting at a particular offset. Because this function is rarely used, it is beyond the scope of this book. Refer to the corresponding Boost.Asio documentation section for more details about this function and its overloads at [http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/read_at.html](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/read_at.html).

The `asio::read()`, `asio::read_until()`, and `asio::read_at()` functions are implemented in a similar way to how the original `readFromSocket()` function in our sample is implemented by means of several calls to the socket object's `read_some()` method in a loop until the termination condition is satisfied or an error occurs.

## See also

*   The *Using extensible stream-oriented I/O buffers* recipe demonstrates how to write and read data to and from the `asio::streambuf` buffer
*   The *Implementing a synchronous TCP client* recipe in [Chapter 3](ch03.html "Chapter 3. Implementing Client Applications"), *Implementing Client Applications*, demonstrates how to implement a synchronous TCP client that performs synchronous reading from a socket to receive response messages sent by the server
*   The *Implementing a synchronous iterative TCP server* recipe in [Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*, demonstrates how to implement a synchronous TCP server that performs synchronous reading to receive request messages from the client

# Writing to a TCP socket asynchronously

Asynchronous writing is a flexible and efficient way to send data to a remote application. In this recipe, we will see how to write data to a TCP socket asynchronously.

## How to do it…

The most basic tool used to asynchronously write data to the socket provided by the Boost.Asio library is the `async_write_some()` method of the `asio::ip::tcp::socket` class. Let's take a look at one of the method's overloads:

[PRE21]

This method initiates the write operation and returns immediately. It accepts an object that represents a buffer that contains the data to be written to the socket as its first argument. The second argument is a callback, which will be called by Boost.Asio when an initiated operation is completed. This argument can be a function pointer, functor, or any other object that satisfies the requirements of the `WriteHandler` concept. The complete list of the requirements can be found in the corresponding section of the Boost.Asio documentation at [http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/WriteHandler.html](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/WriteHandler.html).

The callback should have the following signature:

[PRE22]

Here, `ec` is an argument that indicates an error code if one occurs, and the `bytes_transferred` argument indicates how many bytes have been written to the socket during the corresponding asynchronous operation.

As the `async_write_some()` method's name suggests, it initiates an operation that is intended to write *some* amount of data from the buffer to the socket. This method guarantees that at least one byte will be written during the corresponding asynchronous operation if an error does not occur. This means that, in a general case, in order to write all the data available in the buffer to the socket, we may need to perform this asynchronous operation several times.

Now that we know how the key method works, let's see how to implement an application that performs asynchronous writing to the socket.

The following algorithm describes the steps required to perform and implement an application, which writes data to a TCP socket asynchronously. Note that this algorithm provides a *possible* way to implement such an application. Boost.Asio is quite flexible and allows us to organize and structure the application by writing data to a socket asynchronously in many different ways:

1.  Define a data structure that contains a pointer to a socket object, a buffer, and a variable used as a counter of bytes written.
2.  Define a callback function that will be called when the asynchronous writing operation is completed.
3.  In a client application, allocate and open an active TCP socket and connect it to a remote application. In a server application, obtain a connected active TCP socket by accepting a connection request.
4.  Allocate a buffer and fill it with data that is to be written to the socket.
5.  Initiate an asynchronous writing operation by calling the socket's `async_write_some()` method. Specify a function defined in step 2 as a callback.
6.  Call the `run()` method on an object of the `asio::io_service` class.
7.  In a callback, increase the counter of bytes written. If the number of bytes written is less than the total amount of bytes to be written, initiate a new asynchronous writing operation to write the next portion of the data.

Let's implement a sample client application that performs asynchronous writing in accordance with the preceding algorithm.

We begin with adding the `include` and `using` directives:

[PRE23]

Next, according to step 1 of the algorithm, we define a data structure that contains a pointer to the socket object, a buffer that contains data to be written, and a counter variable that contains the number of bytes already written:

[PRE24]

In step 2, we define a callback function, which will be called when the asynchronous operation is completed:

[PRE25]

Let's skip step 3 for now and implement steps 4 and 5 in a separate function. Let's call this function `writeToSocket()`:

[PRE26]

Now, we come back to step 3 and implement it in the `main()`application entry point function:

[PRE27]

## How it works…

Now, let's track the application's execution path to better understand how it works.

The application is run by a single thread, in the context of which the application's `main()` entry point function is called. Note that Boost.Asio may create additional threads for some internal operations, but it guarantees that no application code is executed in the context of those threads.

The `main()` function allocates, opens, and synchronously connects a socket to a remote application and then calls the `writeToSocket()` function by passing a pointer to the socket object. This function initiates an asynchronous write operation and returns. We'll consider this function in a moment. The `main()` function continues with calling the `run()` method on the object of the `asio::io_service` class, where Boost.Asio *captures* the thread of execution and uses it to call the callback functions associated with asynchronous operations when they get completed.

The `asio::os_service::run()` method blocks, as long as, at least one pending asynchronous operation. When the last callback of the last pending asynchronous operation is completed, this method returns.

Now, let's come back to the `writeToSocket()` function and analyze its behavior. It begins with allocating an instance of the `Session` data structure in the free memory. Then, it allocates and fills the buffer with the data to be written to the socket. After this, a pointer to the socket object and the buffer are stored in the `Session` object. Because the socket's `async_write_some()` method may not write all the data to the socket in one go, we may need to initiate another asynchronous write operation in a callback function. That's why we need the `Session` object and we allocate it in the free memory and not on the stack; it must *live* until the callback function is called.

Finally, we initiate the asynchronous operation, calling the socket object's `async_write_some()` method. The invocation of this method is somewhat complex, and, therefore, let's consider this in more detail:

[PRE28]

The first argument is a buffer that contains data to be written to the socket. Because the operation is asynchronous, this buffer may be accessed by Boost.Asio at any moment between operation initiation and when the callback is called. This means that the buffer must stay intact and must be available until the callback is called. We guarantee this by storing the buffer in a `Session` object, which in turn is stored in the free memory.

The second argument is a callback that is to be invoked when the asynchronous operation is completed. Boost.Asio defines a callback as a *concept*, which can be a function or a functor, that accepts two arguments. The first argument of the callback specifies an error that occurs while the operation is being executed, if any. The second argument specifies the number of bytes written by the operation.

Because we want to pass an additional argument to our callback function, a pointer to the corresponding `Session` object, which acts as a context for the operation, we use the `std::bind()` function to construct a function object to which we attach a pointer to the `Session` object as the third argument. The function object is then passed as a callback argument to the socket object's `async_write_some()` method.

Because it is asynchronous, the `async_write_some()` method doesn't block the thread of execution. It initiates the writing operation and returns.

The actual writing operation is executed behind the scenes by the Boost.Asio library and underlying operating system, and when the operation is complete or an error occurs, the callback is invoked.

When invoked, the callback function named, literally, `callback` in our sample application begins with checking whether the operation succeeded or an error occurred. In the latter case, the error information is output to the standard output stream and the function returns. Otherwise, the counter of the total written bytes is increased by the number of bytes written as a result of an operation. Then, we check whether the total number of bytes written to the socket is equal to the size of the buffer. If these values are equal, this means that all the data has been written to the socket and there is no more work to do. The callback function returns. However, if there is still data in the buffer that is to be written, a new asynchronous write operation is initiated:

[PRE29]

Note how the beginning of the buffer is shifted by the number of bytes already written, and how the size of the buffer is decreased by the same value, correspondingly.

As a callback, we specify the same `callback()` function using the `std::bind()` function to attach an additional argument—the `Session` object, just like we did when we initiated the first asynchronous operation.

The cycles of initiation of an asynchronous writing operation and consequent callback invocation repeat until all the data from the buffer is written to the socket or an error occurs.

When the `callback` function returns without initiating a new asynchronous operation, the `asio::io_service::run()` method, called in the `main()` function, unblocks the thread of execution and returns. The `main()` function returns as well. This is when the application exits.

## There's more...

Although the `async_write_some()` method described in the previous sample allows asynchronously writing data to the socket, the solution based on it is somewhat complex and error-prone. Fortunately, Boost.Asio provides a more convenient way to asynchronously write data to a socket using the free function `asio::async_write()`. Let's consider one of its overloads:

[PRE30]

This function is very similar to the socket's `async_write_some()` method. Its first argument is an object that satisfies the requirements of the `AsyncWriteStream` concept. For the complete list of the requirements, refer to the corresponding Boost.Asio documentation section at [http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/AsyncWriteStream.html](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/AsyncWriteStream.html). The object of the `asio::ip::tcp::socket` class satisfies these requirements and, therefore, can be used with this function.

The second and the third arguments of the `asio::async_write()` function are similar to the first and second arguments of the `async_write_some()` method of a TCP socket object described in the previous sample. These arguments are buffers that contain data that is to be written and functions or objects that represent a callback, which will be called when the operation is completed.

In contrast to the socket's `async_write_some()` method, which initiates the operation that writes *some* amount of data from the buffer to the socket, the `asio::async_write()` function initiates the operation, which writes all the data available in the buffer. In this case, the callback is called only when all the data available in the buffer is written to the socket or when an error occurs. This simplifies writing to the socket and makes the code shorter and cleaner.

If we change our previous sample so that it uses the `asio::async_write()` function instead of the socket object's `async_write_some()` method to write data to the socket asynchronously, our application becomes significantly simpler.

Firstly, we don't need to keep track of the number of bytes written to the socket, so therefore, the `Session` structure becomes smaller:

[PRE31]

Secondly, we know that when the callback function is invoked, it means that either all the data from the buffer has been written to the socket or an error has occurred. This makes the callback function much simpler:

[PRE32]

The `asio::async_write()` function is implemented by means of zero or more calls to the socket object's `async_write_some()` method. This is similar to how the `writeToSocket()` function in our initial sample is implemented.

### Note

Note that the `asio::async_write()` function has three more overloads, providing additional functionalities. Some of them may be very useful in specific circumstances. Refer to the Boost.Asio documentation to find out more about this function at [http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/async_write.html](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/async_write.html).

## See also

*   The *Writing to a TCP socket synchronously* recipe describes how to write data to a TCP socket synchronously
*   The *Implementing an asynchronous TCP client* recipe in [Chapter 3](ch03.html "Chapter 3. Implementing Client Applications"), *Implementing Client Applications*, demonstrates how to implement an asynchronous TCP client that performs asynchronous writing to a TCP socket to send request messages to the server
*   The *Implementing an asynchronous TCP server* recipe in [Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*, demonstrates how to implement an asynchronous TCP server that performs asynchronous writing to a TCP socket to send response messages to the client

# Reading from a TCP socket asynchronously

Asynchronous reading is a flexible and efficient way to receive data from a remote application. In this recipe, we will see how to read data from a TCP socket asynchronously.

## How to do it…

The most basic tool used to asynchronously read data from a TCP socket provided by the Boost.Asio library is the `async_read_some()` method of the `asio::ip::tcp::socket` class. Here is one of the method's overloads:

[PRE33]

This method initiates an asynchronous read operation and returns immediately. It accepts an object that represents a mutable buffer as its first argument to which the data will be read from the socket. The second argument is a callback that is called by Boost.Asio when the operation is completed. This argument can be a function pointer, a functor, or any other object that satisfies the requirements of the `ReadHandler` concept. The complete list of the requirements can be found in the corresponding section of the Boost.Asio documentation at [http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/ReadHandler.html](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/ReadHandler.html).

The callback should have the following signature:

[PRE34]

Here, `ec` is an argument that notifies an error code if one occurs, and the `bytes_transferred` argument indicates how many bytes have been read from the socket during the corresponding asynchronous operation.

As the `async_read_some()` method's name suggests, it initiates an operation that is intended to read *some* amount of data from the socket to the buffer. This method guarantees that at least one byte will be read during the corresponding asynchronous operation if an error does not occur. This means that, in a general case, in order to read all the data from the socket, we may need to perform this asynchronous operation several times.

Now that we know how the key method works, let's see how to implement an application that performs asynchronous reading from the socket.

The following algorithm describes the steps required to implement an application, which reads data from a socket asynchronously. Note that this algorithm provides a *possible* way to implement such an application. Boost.Asio is quite flexible and allows us to organize and structure the application by reading data from a socket asynchronously in different ways:

1.  Define a data structure that contains a pointer to a socket object, a buffer, a variable that defines the size of the buffer, and a variable used as a counter of bytes read.
2.  Define a callback function that will be called when an asynchronous reading operation is completed.
3.  In a client application, allocate and open an active TCP socket, and then, connect it to a remote application. In a server application, obtain a connected active TCP socket by accepting a connection request.
4.  Allocate a buffer big enough for the expected message to fit in.
5.  Initiate an asynchronous reading operation by calling the socket's `async_read_some()` method, specifying a function defined in step 2 as a callback.
6.  Call the `run()` method on an object of the `asio::io_service` class.
7.  In a callback, increase the counter of bytes read. If the number of bytes read is less than the total amount of bytes to be read (that is, the size of an expected message), initiate a new asynchronous reading operation to read the next portion of data.

Let's implement a sample client application which will perform asynchronous reading in accordance with the preceding algorithm.

We begin with adding the `include` and `using` directives:

[PRE35]

Next, according to step 1, we define a data structure that contains a pointer to the socket object named `sock`, a pointer to the buffer named `buf`, a variable named `buf_size` that contains the size of the buffer, and a `total_bytes_read` variable that contains the number of bytes already read:

[PRE36]

In step 2, we define a callback function, which will be called when asynchronous operation is completed:

[PRE37]

Let's skip step 3 for now and implement steps 4 and 5 in a separate function. Let's name this function `readFromSocket()`:

[PRE38]

Now, we come back to step 3 and implement it in the application's `main()` entry point function:

[PRE39]

## How it works…

Now, let's track the application's execution path to better understand how it works.

The application is run by a single thread; in the context of which the application's `main()` entry point function is called. Note that Boost.Asio may create additional threads for some internal operations, but it guarantees that no application code is called in the context of those threads.

The `main()` function begins with allocating, opening, and connecting a socket to a remote application. Then, it calls the `readFromSocket()` function and passes a pointer to the socket object as an argument. The `readFromSocket()` function initiates an asynchronous reading operation and returns. We'll consider this function in a moment. The `main()` function continues with calling the `run()` method on the object of the `asio::io_service` class, where Boost.Asio *captures* the thread of execution and uses it to call the callback functions associated with asynchronous operations when they get completed.

The `asio::io_service::run()` method blocks as long as there is at least one pending asynchronous operation. When the last callback of the last pending operation is completed, this method returns.

Now, let's come back to the `readFromSocket()` function and analyze its behavior. It begins with allocating an instance of the `Session` data structure in the free memory. Then, it allocates a buffer and stores a pointer to it in a previously allocated instance of the `Session` data structure. A pointer to the socket object and the size of the buffer are stored in the `Session` data structure as well. Because the socket's `async_read_some()` method may not read all the data in one go, we may need to initiate another asynchronous reading operation in the callback function. This is why we need the `Session` data structure and why we allocate it in the free memory and not on a stack. This structure and all the objects that reside in it must *live* at least until the callback is invoked.

Finally, we initiate the asynchronous operation, calling the socket object's `async_read_some()` method. The invocation of this method is somewhat complex; therefore, let's take a look at it in more detail:

[PRE40]

The first argument is the buffer to which the data will be read. Because the operation is asynchronous, this buffer may be accessed by Boost.Asio at any moment between the operation initiation and when the callback is invoked. This means that the buffer must stay intact and be available until the callback is invoked. We guarantee this by allocating the buffer in the free memory and storing it in the `Session` data structure, which in turn is allocated in the free memory as well.

The second argument is a callback that is to be invoked when the asynchronous operation is completed. Boost.Asio defines a callback as a concept, which can be a function or a functor, that accepts two arguments. The first argument of the callback specifies an error that occurs while the operation is being executed, if any. The second argument specifies the number of bytes read by the operation.

Because we want to pass an additional argument to our callback function, a pointer to the corresponding `Session` object, which serves as a context for the operation—we use the `std::bind()` function to construct a function object to which we attach a pointer to the `Session` object as the third argument. The function object is then passed as a callback argument to the socket object's `async_write_some()` method.

Because it is asynchronous, the `async_write_some()` method doesn't block the thread of execution. It initiates the reading operation and returns.

The actual reading operation is executed behind the scenes by the Boost.Asio library and underlying operating system, and when the operation is completed or an error occurs, the callback is invoked.

When invoked, the callback function named, literally, `callback` in our sample application begins with checking whether the operation succeeded or an error occurred. In the latter case, the error information is output to the standard output stream and the function returns. Otherwise, the counter of the total read bytes is increased by the number of bytes read as a result of the operation. Then, we check whether the total number of bytes read from the socket is equal to the size of the buffer. If these values are equal, it means that the buffer is full and there is no more work to do. The callback function returns. However, if there is still some space in the buffer, we need to continue with reading; therefore, we initiate a new asynchronous reading operation:

[PRE41]

Note that the beginning of the buffer is shifted by the number of bytes already read and the size of the buffer is decreased by the same value, respectively.

As a callback, we specify the same `callback` function using the `std::bind()` function to attach an additional argument—the `Session` object.

The cycles of initiation of an asynchronous reading operation and consequent callback invocation repeat until the buffer is full or an error occurs.

When the `callback` function returns without initiating a new asynchronous operation, the `asio::io_service::run()` method, called in the `main()` function, unblocks the thread of execution and returns. The `main()` function returns as well. This is when the application exits.

## There's more...

Although the `async_read_some()` method, as described in the previous sample, allows asynchronously reading data from the socket, the solution based on it is somewhat complex and error-prone. Fortunately, Boost.Asio provides a more convenient way to asynchronously read data from a socket: the free function `asio::async_read()`. Let's consider one of its overloads:

[PRE42]

This function is very similar to the socket's `async_read_some()` method. Its first argument is an object that satisfies the requirements of the `AsyncReadStream` concept. For the complete list of the requirements, refer to the corresponding Boost.Asio documentation section at [http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/AsyncReadStream.html](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/AsyncReadStream.html). The object of the `asio::ip::tcp::socket` class satisfies these requirements and, therefore, can be used with this function.

The second and third arguments of the `asio::async_read()` function are similar to the first and second arguments of the `async_read_some()` method of a TCP socket object described in the previous sample. These arguments are buffers used as data destination points and functions or objects that represent a callback, which will be called when the operation is completed.

In contrast to the socket's `async_read_some()` method, which initiates the operation, that reads *some* amount of data from the socket to the buffer, the `asio::async_read()` function initiates the operation that reads the data from the socket until the buffer passed to it as an argument is full. In this case, the callback is called when the amount of data read is equal to the size of the provided buffer or when an error occurs. This simplifies reading from the socket and makes the code shorter and cleaner.

If we change our previous sample so that it uses the `asio::async_read()` function instead of the socket object's `async_read_some()` method to read data from the socket asynchronously, our application becomes significantly simpler.

Firstly, we don't need to keep track of the number of bytes read from the socket; therefore, the `Session` structure becomes smaller:

[PRE43]

Secondly, we know that when the callback function is invoked, it means that either an expected amount of data has been read from the socket or an error has occurred. This makes the callback function much simpler:

[PRE44]

The `asio::async_read()` function is implemented by means of zero or more calls to the socket object's `async_read_some()` method. This is similar to how the `readFromSocket()` function in our initial sample is implemented.

### Note

Note that the `asio::async_read()` function has three more overloads, providing additional functionalities. Some of them may be very useful in specific circumstances. Refer to the Boost.Asio documentation to find out about this at [http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/async_read.html](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/async_read.html).

## See also

*   The *Reading from a TCP socket synchronously* recipe describes how to read data from a TCP socket synchronously
*   The *Implementing an asynchronous TCP client* recipe in [Chapter 3](ch03.html "Chapter 3. Implementing Client Applications"), *Implementing Client Applications*, demonstrates how to implement an asynchronous TCP client that performs asynchronous reading from a TCP socket to receive response messages sent by the server
*   The *Implementing an asynchronous TCP server* recipe in [Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*, demonstrates how to implement an asynchronous TCP server that performs asynchronous reading from a TCP socket to receive request messages from the client

# Canceling asynchronous operations

Sometimes, after an asynchronous operation has been initiated and has not yet completed, the conditions in the application may change so that the initiated operation becomes irrelevant or outdated and nobody is interested in the completion of the operation.

In addition to this, if an initiated asynchronous operation is a reaction to a user command, the user may change their mind while the operation is being executed. The user may want to discard the previous issued command and may want to issue a different one or decide to exit from the application.

Consider a situation where a user types a website address in a typical web browser's address bar and hits the *Enter* key. The browser immediately initiates a DNS name resolution operation. When the DNS name is resolved and the corresponding IP address is obtained, it initiates the connection operation to connect to the corresponding web server. When a connection is established, the browser initiates an asynchronous write operation to send a request to the server. Finally, when the request is sent, the browser starts waiting for the response message. Depending on the responsiveness of the server application, the volume of the data transmitted over the network, the state of the network, and other factors, all these operations may take a substantial amount of time. And the user while waiting for the requested web page to be loaded, may change their mind, and before the page gets loaded, the user may type another website address in the address bar and hit *Enter*.

Another (extreme) situation is where a client application sends a request to the server application and starts waiting for the response message, but the server application while processing the client's request, gets into a deadlock due to bugs in it. In this case, the user would have to wait forever for the response message and would never get it.

In both the cases, the user of the client application would benefit from having the ability to cancel the operation they initiated before it completes. In general, it is a good practice to provide the user with the ability to cancel an operation that may take a noticeable amount of time. Because the network communication operations fall into a class of operations that may last for unpredictably long periods of time, it is important to support the cancelation of operations in distributed applications that communicate over the network.

One of the benefits of asynchronous operations provided by the Boost.Asio library is that they can be canceled at any moment after the initiation. In this recipe, we'll see how to cancel asynchronous operations.

## How to do it…

The following algorithm provides the steps required to initiate and cancel asynchronous operations with Boost.Asio:

1.  If the application is intended to run on Windows XP or Windows Server 2003, define flags that enable asynchronous operation canceling on these versions of Windows.
2.  Allocate and open a TCP or UDP socket. It may be an active or passive (acceptor) socket in the client or server application.
3.  Define a callback function or functor for an asynchronous operation. If needed, in this callback, implement a branch of code that handles the situation when the operation has been canceled.
4.  Initiate one or more asynchronous operations and specify a function or an object defined in step 4 as a callback.
5.  Spawn an additional thread and use it to run the Boost.Asio event loop.
6.  Call the `cancel()` method on the socket object to cancel all the outstanding asynchronous operations associated with this socket.

Let's consider the implementation of the client application designed in accordance with the presented algorithm in which an asynchronous *connection* operation is first initiated and then canceled.

According to step 1, to compile and run our code on Windows XP or Windows Server 2003, we need to define some flags that control the behavior of the Boost.Asio library with regard to which mechanisms of the underlying OS to exploit.

By default, when it is compiled for Windows, Boost.Asio uses the I/O completion port framework to run operations asynchronously. On Windows XP and Windows Server 2003, this framework has some issues and limitations with regard to the cancelation of an operation. Therefore, Boost.Asio requires developers to explicitly notify that they want to enable the asynchronous operation canceling functionality despite of the known issues, when targeting the application in versions of Windows in question. To do this, the `BOOST_ASIO_ENABLE_CANCELIO` macro must be defined before Boost.Asio headers are included. Otherwise, if this macro is not defined, while the source code of the application contains calls to asynchronous operations, cancelation methods and functions, the compilation will always fail.

In other words, it is mandatory to define the `BOOST_ASIO_ENABLE_CANCELIO` macro, when targeting Windows XP or Windows Server 2003, and the application needs to cancel asynchronous operations.

To get rid of issues and limitations imposed by the usage of the I/O completion port framework on Windows XP and Windows Server 2003, we can prevent Boost.Asio from using this framework by defining another macro named `BOOST_ASIO_DISABLE_IOCP` before including Boost.Asio headers. With this macro defined, Boost.Asio doesn't use the I/O completion port framework on Windows; and therefore, problems related to asynchronous operations canceling disappear. However, the benefits of scalability and efficiency of the I/O completion ports framework disappear too.

Note that the mentioned issues and limitations related to asynchronous operation canceling do not exist on Windows Vista and Windows Server 2008 and later. Therefore, when targeting these versions of Windows, canceling works fine, and there is no need to disable the I/O completion port framework usage unless there is another reason to do so. Refer to the `asio::ip::tcp::cancel()` method's documentation section for more details on this issue at [http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/basic_stream_socket/cancel/overload1.html](http://www.boost.org/doc/libs/1_58_0/doc/html/boost_asio/reference/basic_stream_socket/cancel/overload1.html).

In our sample, we will consider how to construct a cross-platform application that, when targeted at Windows during compilation, can be run on any Windows version, starting from Windows XP or Windows Server 2003\. Therefore, we define both the `BOOST_ASIO_DISABLE_IOCP` and `BOOST_ASIO_ENABLE_CANCELIO` macros.

To determine the target operating system at compile time, we use the `Boost.Predef` library. This library provides us with macro definitions that allow us to identify parameters of the environment in which the code is compiled as the target operating system family and its version, processor architecture, compiler, and many others. Refer to the Boost.Asio documentation section for more details on this library at [http://www.boost.org/doc/libs/1_58_0/libs/predef/doc/html/index.html](http://www.boost.org/doc/libs/1_58_0/libs/predef/doc/html/index.html).

To use the `Boost.Predef` library, we include the following header file:

[PRE45]

Then, we check whether the code is being compiled for Windows XP or Windows Server 2003, and if it is, we define the `BOOST_ASIO_DISABLE_IOCP` and `BOOST_ASIO_ENABLE_CANCELIO` macros:

[PRE46]

Next, we include the common Boost.Asio header and standard library `<thread>` header. We will need the latter because we'll spawn additional threads in our application. In addition to this, we specify a `using` directive to make the names of Boost.Asio classes and functions shorter and more convenient to use:

[PRE47]

Then, we define the application's `main()` entry point function, which contains all the functionalities of the application:

[PRE48]

## How it works…

Now, let's analyze how the application works.

Our sample client application consists of a single function, which is the application's `main()` entry point function. This function begins with allocating and opening a TCP socket according to step 2 of the algorithm.

Next, the asynchronous connection operation is initiated on the socket. The callback provided to the method is implemented as a lambda function. This corresponds to steps 3 and 4 of the algorithm. Note how the fact that the operation was canceled is determined in the callback function. When an asynchronous operation is canceled, the callback is invoked and its argument that specifies the error code contains an OS dependent error code defined in Boost.Asio as `asio::error::operation_aborted`.

Then, we spawn a thread named `worker_thread`, which will be used to run the Boost.Asio event loop. In the context of this thread, the callback function will be invoked by the library. The entry point function of the `worker_thread` thread is quite simple. It contains a `try-catch` block and a call to the `asio::io_service` object's `run()` method. This corresponds to step 5 of the algorithm.

After the worker thread is spawned, the main thread is put to sleep for 2 seconds. This is to allow the connection operation to progress a bit and emulate what could be a delay between the two commands issued by the user in the real application; for example, a web browser.

According to the last step 6 of the algorithm, we call the socket object's `cancel()` method to cancel the initiated connection operation. At this point, if the operation has not yet finished, it will be canceled and the corresponding callback will be invoked with an argument that specifies the error code containing the `asio::error::operation_aborted` value to notify that the operation was canceled. However, if the operation has already finished, calling the `cancel()` method has no effect.

When the callback function returns, the worker thread exits the event loop because there are no more outstanding asynchronous operations to be executed. As a result, the thread exits its entry point function. This leads to the main thread running to its completion as well. Eventually, the application exits.

## There's more...

In the previous sample, we considered the canceling of an asynchronous connection operation associated with an active TCP socket. However, any operation associated with both the TCP and UDP sockets can be canceled in a similar way. The `cancel()` method should be called on the corresponding socket object after the operation has been initiated.

In addition to this, the `async_resolve()` method of the `asio::ip::tcp::resolver` or `asio::ip::udp::resolver` class used to asynchronously resolve a DNS name can be canceled by calling the resolver object's `cancel()` method.

All asynchronous operations initiated by the corresponding free functions provided by Boost.Asio can be canceled as well by calling the `cancel()` method on an object that was passed to the free function as the first argument. This object can represent either a socket (active or passive) or a resolver.

## See also

*   The *Implementing an asynchronous TCP client* recipe in [Chapter 3](ch03.html "Chapter 3. Implementing Client Applications"), *Implementing Client Applications*, demonstrates how to construct a more complex client application that supports the asynchronous operation cancelation functionality
*   [Chapter 1](ch01.html "Chapter 1. The Basics"), *The Basics*, contains recipes that demonstrate how to synchronously connect a socket and resolve a DNS name

# Shutting down and closing a socket

In some distributed applications that communicate over the TCP protocol, there is a need to transfer messages that do not have a fixed size and specific byte sequence, marking its boundary. This means that the receiving side, while reading the message from the socket, cannot determine where the message ends by analyzing the message itself with either its size or its content.

One approach to solve this problem is to structure each message in such a way that it consists of a logical header section and a logical body section. The header section has a fixed size and structure and specifies the size of the body section. This allows the receiving side to first read and parse the header, find out the size of the message body, and then properly read the rest of the message.

This approach is quite simple and is widely used. However, it brings some redundancy and additional computation overhead, which may be unacceptable in some circumstances.

Another approach can be applied when an application uses a separate socket for each message sent to its peer, which is a quite popular practice. The idea of this approach is to **shut down** the send part of the socket by the message sender after the message is written to the socket. This results in a special service message being sent to the receiver, informing the receiver that the message is over and the sender will not send anything else using the current connection.

The second approach provides many more benefits than the first one and, because it is part of the TCP protocol software, it is readily available to the developer for usage.

Another operation on a socket, that is, **closing** may seem similar to shutting down, but it is actually very different from it. Closing a socket assumes returning the socket and all the other resources associated with it back to the operating system. Just like memory, a process or a thread, a file handle or a mutex, a socket is a resource of an operating system. And like any other resource, a socket should be returned back to the operating system after it has been allocated, used, and is not needed by the application anymore. Otherwise, a resource leak may occur, which may eventually lead to the exhaustion of the resource and to the application's fault or instability of the whole operating system.

Serious issues that may occur when sockets are not closed make closing a very important operation.

The main difference between shutting down and closing a TCP socket is that closing interrupts the connection if one is established and, eventually, deallocates the socket and returns it back to the operating system, while shutting down only disables writing, reading, or both the operations on the socket and sends a service message to the peer application notifying about this fact. Shutting down a socket never results in deallocating the socket.

In this recipe, we'll see how to shut down and close a TCP socket.

## How to do it…

Here, we'll consider a distributed application that consists of two parts: a client and a server to better understand how a socket shut down operation can be used to make an application layer protocol more efficient and clear when the communication between parts of distributed applications is based on binary messages of random sizes.

For simplicity, all operations in both the client and server applications are synchronous.

### The client application

The purpose of the client application is to allocate the socket and connect it to the server application. After the connection is established, the application should prepare and send a request message notifying its boundary by shutting down the socket after writing the message to it.

After the request is sent, the client application should read the response. The size of the response is unknown; therefore, the reading should be performed until the server closes its socket to notify the response boundary.

We begin the client application by specifying the `include` and `using` directives:

[PRE49]

Next, we define a function that accepts a reference to the socket object connected to the server and performs the communication with the server using this socket. Let's name this function `communicate()`:

[PRE50]

Finally, we define an application's `main()` entry point function. This function allocates and connects the socket, and then calls the `communicate()` function defined in the previous step:

[PRE51]

### The server application

The server application is intended to allocate an acceptor socket and passively wait for a connection request. When the connection request arrives, it should accept it and read the data from the socket connected to the client until the client application shuts down the socket on its side. Having received the request message, the server application should send the response message notifying its boundary by shutting down the socket.

We begin the client application by specifying `include` and `using` directives:

[PRE52]

Next, we define a function that accepts a reference to the socket object connected to the client application and performs the communication with the client using this socket. Let's name this function `processRequest()`:

[PRE53]

Finally, we define the application's `main()` entry point function. This function allocates an acceptor socket and waits for the incoming connection requests. When the connection request arrives, it obtains an active socket that is connected to the client application and calls the `processRequest()` function defined in the previous step by passing a connected socket object to it:

[PRE54]

### Closing a socket

In order to close an allocated socket, the `close()` method should be called on the corresponding object of the `asio::ip::tcp::socket` class. However, usually, there is no need to do it explicitly because the destructor of the socket object closes the socket if one was not closed explicitly.

## How it works…

The server application is first started. In its `main()` entry point function, an acceptor socket is allocated, opened, bound to port `3333`, and starts waiting for the incoming connection request from the client.

Then, the client application is started. In its `main()` entry point function, an active socket is allocated, opened, and connected to the server. After the connection is established, the `communicate()` function is called. In this function, all the interesting things take place.

The client application writes a request message to the socket and then calls the socket's `shutdown()` method, passing an `asio::socket_base::shutdown_send` constant as an argument. This call shuts down the send part of the socket. At this point, writing to the socket is disabled, and there is no way to restore the socket state to make it writable again:

[PRE55]

Shutting down the socket in the client application is seen in the server application as a protocol service message that arrives to the server, notifying the fact that the peer application has shut down the socket. Boost.Asio delivers this message to the application code by means of an error code returned by the `asio::read()` function. The Boost.Asio library defines this code as `asio::error::eof`. The server application uses this error code to find out when the client finishes sending the request message.

When the server application receives a full request message, the server and client exchange their roles. Now, the server writes the data, namely, the response message to the socket on its side, and the client application reads this message on its side. When the server finishes writing the response message to the socket, it shuts down the send part of its socket to imply that the whole message has been sent to its peer.

Meanwhile, the client application is blocked in the `asio::read()` function and reads the response sent by the server until the function returns with the error code equal to `asio::error::eof`, which implies that the server has finished sending the response message. When the `asio::read()` function returns with this error code, the client *knows* that it has read the whole response message, and it can then start processing it:

[PRE56]

Note that after the client has shut down its socket's send part, it can still read data from the socket because the receive part of the socket stays open independently from the send part.

## See also

*   The *Writing to a TCP socket synchronously* recipe demonstrates how to write data to a TCP socket synchronously
*   The *Reading from a TCP socket synchronously* recipe demonstrates how to read data from a TCP socket synchronously
*   The *Implementing the HTTP client application* and *Implementing the HTTP server application* recipes in [Chapter 5](ch05.html "Chapter 5. HTTP and SSL/TLS"), *HTTP and SSL/TLS*, demonstrate how a socket shut down is used in the implementation of the HTTP protocol