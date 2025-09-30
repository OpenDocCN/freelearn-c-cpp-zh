# Socket Programming Tips and Pitfalls

This chapter builds on all the knowledge you've obtained throughout this book.

**Socket programming** can be complicated. There are many pitfalls to avoid and subtle programming techniques to implement. In this chapter, we consider some nuanced details of network programming that are essential for writing robust programs.

The following topics are covered in this chapter:

*   Error handling and error descriptions
*   TCP handshakes and orderly release
*   Timeout on `connect()`
*   Preventing TCP deadlocks
*   TCP flow control
*   Avoiding address-in-use errors
*   Preventing `SIGPIPE` crashes
*   Multiplexing limitations of `select()`

# Technical requirements

Any modern C compiler can compile the example programs from this chapter. We recommend **MinGW** on Windows and **GCC** on Linux and macOS. See [Appendix B](47da8507-709b-44a6-9399-b18ce6afd8c9.xhtml), *Setting Up Your C Compiler On Windows*, [Appendix C](221eebc0-0bb1-4661-a5aa-eafed9fcba7e.xhtml), *Setting Up Your C Compiler On Linux*, and [Appendix D](632db68e-0911-4238-a2be-bd1aa5296120.xhtml), *Setting Up Your C Compiler On macOS*, for compiler setup.

The code for this book can be found at: [https://github.com/codeplea/Hands-On-Network-Programming-with-C](https://github.com/codeplea/Hands-On-Network-Programming-with-C).

From the command line, you can download the code for this chapter with the following command:

```cpp
git clone https://github.com/codeplea/Hands-On-Network-Programming-with-C
cd Hands-On-Network-Programming-with-C/chap13
```

Each example program in this chapter runs on Windows, Linux, and macOS. When compiling on Windows, each example program will require linking to the **Winsock** library. This can be accomplished by passing the `-lws2_32` option to `gcc`.

All of the example programs in this chapter require the same header files and C macros that we developed in [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs*. For brevity, we put these statements in a separate header file, `chap13.h`. For an explanation of these statements, please refer to [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs*.

The first part of `chap13.h` includes the needed networking headers for each platform. The code for that is as follows:

```cpp
/*chap13.h*/

#if defined(_WIN32)
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>

#endif
```

We also define some macros to make writing portable code easier, and we include the additional headers that our programs need:

```cpp
/*chap13.h continued*/

#if defined(_WIN32)
#define ISVALIDSOCKET(s) ((s) != INVALID_SOCKET)
#define CLOSESOCKET(s) closesocket(s)
#define GETSOCKETERRNO() (WSAGetLastError())

#else
#define ISVALIDSOCKET(s) ((s) >= 0)
#define CLOSESOCKET(s) close(s)
#define SOCKET int
#define GETSOCKETERRNO() (errno)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
```

That concludes `chap13.h`.

# Error handling

**Error handling** can be a problematic topic in C as it does not "hold the programmer's hand". Any memory or resources allocated must be manually released, and this can be tricky to get exactly right in every situation.

When a networked program encounters an error or unexpected situation, the normal program flow is interrupted. This is made doubly difficult when designing a multiplexed system that handles many connections concurrently.

The example programs in the book take a shortcut to error handling. Almost all of them simply terminate after an error is detected. While this is sometimes a valid strategy in real-world programs, real-world programs usually need more complicated error recovery.

Sometimes, you can get away with merely having your client program terminate after encountering an error. This behavior is often the correct response for simple command-line utilities. At other times, you may need to have your program automatically try again.

Event-driven programming can provide the technique needed to simplify this logic a bit. Mainly, your program is structured so that a data structure is allocated to store information about each connection. Your program uses a main loop that checks for events, such as a readable or writable socket, and then handles those events. When structuring your program in this way, it is often easier to flag a connection as needing an action, rather than calling a function to process that action immediately.

With careful design, errors can be handled as a simple matter of course, instead of as exceptions to the normal program flow.

Ultimately, error handling is a very specialized process, and careful care needs to be taken to consider application requirements. What's appropriate for one system is not necessarily correct for another.

In any case, a robust program design dictates that you carefully consider how to handle errors. Many programmers focus only on the happy path. That is, they take care to design the program flow based on the assumption that everything goes correctly. For robust programs, this is a mistake. It is equally important to consider the program flow in cases where everything goes wrong.

Throughout the rest of this chapter, we touch on places where network programming can go wrong. Network programming can be subtle, and many of these failure modes are surprising. However, with proper consideration, they are all capable of being handled.

Before diving into all the weird ways a connection can fail, let's first focus on making error logging a bit easier. In this book, so far, we've been dealing with numeric error codes. It is often more useful to obtain a text description of an error. We look at a method for this next.

# Obtaining error descriptions

In [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs*,  we developed the `GETSOCKETERRNO()` macro as a cross-platform way to obtain the error code after a failed system call.

The `GETSOCKETERRNO()` macro is repeated here for your convenience:

```cpp
#if defined(_WIN32)
#define GETSOCKETERRNO() (WSAGetLastError())
#else
#define GETSOCKETERRNO() (errno)
#endif
```

The preceding code has served us well throughout this book. It has the advantage of being short and simple.

In a real-world program, you may want to display a text-based error message in addition to the error code. Windows and Unix-based systems both provide functions for this purpose.

We can build a simple function to return the last error message as a C string. The code for this function is the following:

```cpp
/*error_text.c excerpt*/

const char *get_error_text() {

#if defined(_WIN32)

    static char message[256] = {0};
    FormatMessage(
        FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_IGNORE_INSERTS,
        0, WSAGetLastError(), 0, message, 256, 0);
    char *nl = strrchr(message, '\n');
    if (nl) *nl = 0;
    return message;

#else
    return strerror(errno);
#endif

}
```

The preceding function formats the error as text using `FormatMessage()` on Windows and `strerror()` on other operating systems.

Unix-based systems provide the `strerror()` function. This function takes the error code as its only parameter, and it returns a pointer to an error message string.

Getting an error code description on Windows is a bit more involved. We use the `FormatMessage()` function to obtain the text description. This function has many options, but the parameters used in the preceding code snippet work well for our purposes. Note that Windows error descriptions are generally returned ending with a newline. Our function uses `strrchr()` to find the last line-feed character and truncate the description at that point.

This chapter's code includes a program called `error_text.c` that demonstrates this method. This program calls the `socket()` function with invalid parameters, and then uses `get_error_text()` to display the error message:

```cpp
/*error_text.c excerpt*/

    printf("Calling socket() with invalid parameters.\n");
    socket(0, 0, 0);
    printf("Last error was: %s\n", get_error_text());
```

Note that error codes and descriptions vary greatly between operating systems. The next two screenshots show the error message displayed by this program on both Windows and Linux.

The following screenshot shows `error_text` running on Windows:

![](img/29559741-e392-4d19-8382-f2028b1a5ee8.png)

The next screenshot shows `error_text` running on an Ubuntu Linux desktop:

![](img/447be4e8-cc7a-43db-8f1f-dc28fcb5816d.png)

As you can see from the preceding two screenshots, different operating systems don't often report errors in the same way.

Now that we have a better way to investigate errors, let's move on to consider some ways that **TCP sockets** can fail.

# TCP socket tips

The **Transmission Control Protocol** (**TCP**) is a fantastic protocol, and TCP sockets provide a beautiful abstraction. They present discrete packets on an unreliable network as a reliable, continuous stream of data. To the programmer, sending and receiving data from a peer anywhere in the world is made nearly as easy as reading and writing to a file.

TCP works very well to hide network shortcomings. When a flaky network drops a few packets, TCP faithfully sorts out the mess and retransmits as needed. The application using TCP receives the data in perfect order. The application doesn't even know there was a network problem, and it certainly doesn't need to address the problem.

With this abstraction, like all abstractions, comes some inherent risk. TCP tries very hard to make networks look reliable. It usually succeeds, but sometimes, abstractions leak. What happens if your network cable is cut? What happens if the application you are connected to crashes? TCP isn't magic. It can't fix these problems.

Of course, it's evident that the abstraction must break when faced with severe problems such as a total network outage. However, sometimes, more subtle problems can arise from details thought to be abstracted away. For example, what happens when you try to send a lot of data, but the peer that you are connected to isn't reading it? (Answer: the data gets backed up.)

In this section, we look at TCP in a little more detail. We're especially interested in the behavior of TCP sockets in these edge cases.

A TCP connection lifespan can be divided into three distinct phases. They are as follows:

*   The setup phase
*   The data-transfer phase
*   The tear-down phase

Problems can arise in each step.

In the setup phase, we have to consider what happens if the target system doesn't respond. By default, `connect()` sometimes waits a long time, attempting to establish a TCP connection. Sometimes, that is what you want, but often it isn't.

For the data-transfer phase, we must be careful to prevent deadlocks. An awareness of TCP congestion control mechanisms can also help us to prevent degenerate cases where our connection becomes slow or uses a lot more bandwidth than necessary.

Finally, knowing the details of the tear-down phase helps us to ensure that we haven't lost data at the end of a connection. Details of how sockets are terminated can also cause operating systems to hold on to half-dead connections long after they've disconnected. These lingering sockets can prevent new programs from binding to their local ports.

Let's begin with some information about the three-way handshake that establishes TCP connections and how to timeout a `connect()` call.

# Timeout on connect()

Usually, when we call `connect()` on a TCP socket, `connect()` blocks until the connection is established.

The following diagram illustrates the TCP three-way handshake that establishes a typical TCP connection and how it relates to a standard, blocking `connect()` call:

![](img/dda38f05-7a0b-45f7-8dd0-83a423efd7e8.png)

The standard TCP three-way handshake consists of three parts. First, the **Client** sends a **Synchronize (SYN)** message to the **Server**. Then the **Server** responds with an **SYN Message** of its own, combined with an **Acknowledged** (**ACK**) message of the **Client**'s **SYN Message**. The **Client** then responds with an acknowledgment of the **Server**'s **SYN Message**. The connection is then open and ready for data transmission.

When the `connect()` function is called on the **Client** side, the first **SYN Message** is sent, and the `connect()` function blocks until the **SYN+ACK Message** is received from the **Server**. After the **SYN+ACK Message** is received, `connect()` enqueues the final **ACK Message** and returns.

This means that `connect()` blocks for at least one round-trip network time. That is, it blocks from the time that its **SYN Message** is sent to the time that the **SYN+ACK Message** is received. While one round-trip network time is the best-case scenario, in the worst case, it could block for much longer. Consider what happens when an overloaded **Server** receives an **SYN Message**. The **Server** could take some time to reply with the **SYN+ACK Message**.

If `connect()` cannot establish a connection successfully (that is, **SYN+ACK Message** is never received), then the `connect()` call eventually times out. This timeout period is controlled by the operating system. The exact timeout period varies, but 20 seconds is about typical.

There is no standard way to extend the timeout period of `connect()`, but you can always call `connect()` again if you want to keep trying.

There are a few ways to make `connect()` timeout early. One way is to use multiple processes and kill the child process if it doesn't connect in time. Another way is to use `SIGALARM` in Unix-based systems.

A cross-platform `connect()` timeout can be achieved by using `select()`. Recall from [Chapter 3](eb2e080f-fad2-4e14-b4c1-9a6c124df77c.xhtml), *An In-Depth Overview of TCP Connections*, that `select()` allows us to wait on a socket operation with a specified timeout.

`select()` also has the additional benefit of allowing your program to do useful work while waiting for the TCP connection to be established. That is, `select()` can be used to wait on multiple `connect()` calls, and other socket events besides. It can work well for a client that needs to connect to several servers in parallel.

Using `select()` to timeout a `connect()` call involves a few steps. They are as follows:

1.  Set the socket to non-blocking operation. This is done using `fcntl(O_NONBLOCK)` on Unix-based systems and `ioctlsocket(FIONBIO)` on Windows.
2.  Call `connect()`. This call returns immediately, provided that *step 1* was successful.
3.  Check the return code from `connect()`. A return value of zero indicates that the connection was successful, which probably indicates that non-blocking mode was set incorrectly. A non-zero return value from `connect()` means we should check the error code (that is, `WSAGetLastError()` on Windows and `errno` on other platforms). An error code of `EINPROGRESS` (`WSAEWOULDBLOCK` on Windows) indicates that the TCP connection is in progress. Any other value indicates an actual error.
4.  Set up and call `select()` with the desired timeout.
5.  Set the socket back to blocking mode.
6.  Check to see whether the socket connected successfully.

*Step 1*, setting the socket to non-blocking mode, can be accomplished with the following code:

```cpp
#if defined(_WIN32)
    unsigned long nonblock = 1;
    ioctlsocket(socket_peer, FIONBIO, &nonblock);
#else
    int flags;
    flags = fcntl(socket_peer, F_GETFL, 0);
    fcntl(socket_peer, F_SETFL, flags | O_NONBLOCK);
#endif
```

The preceding code works a bit differently depending on whether it is running on Windows. On Windows, the `ioctlsocket()` function is used with the `FIONBIO` flag to indicate non-blocking socket operation. On non-Windows systems, the `fcntl()` function is used to set the `O_NONBLOCK` flag for the same purpose.

In *step 2* and *step 3*, the call to `connect()` is done normally. The only difference is that you should expect an error code of `EINPROGRESS` on Unix-based systems and `WSAEWOULDBLOCK` on Windows.

In *step 4*, the setup for `select()` is straightforward. The `select()` function is used in the same way as described in previous chapters. For your convenience, the following code shows one way to use `select()` for this purpose:

```cpp
fd_set set;
FD_ZERO(&set);
FD_SET(socket_peer, &set);

struct timeval timeout;
timeout.tv_sec = 5; timeout.tv_usec = 0;
select(socket_peer+1, 0, &set, 0, &timeout);
```

Notice in the preceding code that we set a timeout of five seconds. Therefore, this `select()` call returns after either the connection is established, the connection has an error, or `5` seconds have elapsed.

In *step 5*, setting the socket back to non-blocking mode is accomplished with the following code:

```cpp
#if defined(_WIN32)
    nonblock = 0;
    ioctlsocket(socket_peer, FIONBIO, &nonblock);
#else
    fcntl(socket_peer, F_SETFL, flags);
#endif
```

In *step 6*, we are looking to see whether the call to `select()` timed out, returned early from an error, or returned early because our socket has successfully connected.

Surprisingly, there is no easy, robust, cross-platform way to check whether the socket is connected at this point. My advice is to simply assume that any socket marked by `select()` as writable has connected successfully. Just try to use the socket. Most TCP client programs will want to call `send()` after connecting, anyway. The return value from this first `send()` call indicates whether you have a problem.

If you really do want to try and determine the socket state without resorting to `send()`, you should be aware of some differences in how `select()` signals in this situation. On Unix-based systems, `select()` signals a socket as writable once the connection is established. If an error has occurred, `select()` signals the socket as both writable and readable. However, if the socket has connected successfully and data has arrived from the remote peer, this also produces both the readable and writable situation. In that case, the `getsockopt()` function can be used to determine whether an error has occurred. On Windows, `select()` marks a socket as excepted if an error occurred.

Please refer to `connect_timeout.c` in this chapter's code repository for a working example of the `connect()` timeout method using `select()`. An additional example, `connect_blocking.c`, is also included for comparison.

Once a new connection is established, our concern moves to preventing data-transfer problems. In the worst case, our program could get deadlocked with its peer, preventing any data transfer. We'll consider this in more detail next.

# TCP flow control and avoiding deadlock

When designing application protocols and writing network code, we need to be careful to prevent a **deadlock** state. A deadlock is when both sides on a connection are waiting for the other side to do something. The worst-case scenario is when both sides end up waiting indefinitely.

A trivial example of a deadlock is if both the client and server call `recv()` immediately after the connection is established. In that case, both sides wait forever for data that is never going to come.

A less obvious deadlock situation can happen if both parties try to send data at the same time. Before we can consider this situation, we must first understand a few more details of how TCP connections operate.

When data is sent over a TCP connection, this data is broken up into segments. A few segments are sent immediately, but additional segments aren't sent over the network until the first few segments are acknowledged as being received by the connected peer. This is part of TCP's **flow-control** scheme, and it helps to prevent a sender from transmitting data faster than a receiver can handle.

Consider the following diagram:

![](img/d90cf010-182f-48da-8dc8-f225e1c62572.png)

In the preceding diagram, the **Client** sends three TCP segments of data to the **Server**. The **Client** has additional **DATA** ready to send, but it must wait until the already-sent data is acknowledged. Once the **ACK Message** is received, the **Client** resumes sending its remaining **DATA**.

This is the TCP flow-control mechanism that ensures that the sender isn't transmitting faster than the receiver can handle.

Now, keeping in mind that a TCP socket can send only a limited amount of data before requiring acknowledgment of receipt, imagine what happens if both parties to a TCP connection try to send a bunch of data at the same time. In this case, both parties send the first few TCP segments. They both then wait until their peer acknowledges receipt before sending more. However, if neither party is reading data, then neither party acknowledges receiving data. This is a deadlock state. Both parties are stuck waiting forever.

Many application protocols prevent this problem by design. These protocols naturally alternate between sending and receiving data. For example, in HTTP, the client sends a request, and then the server sends a reply. The server only starts sending data after the client has finished sending.

However, TCP is a full-duplex protocol. Applications that do need to send data in both directions simultaneously should take advantage of TCP's ability to do so.

As a motivating example, imagine implementing a file-transfer program where both peers to a TCP connection are sending large parts of a file at the same time. How do we prevent the deadlock condition?

The solution to this is straightforward. Both sides should alternate calls to `send()` with calls to `recv()`. The liberal use of `select()` will help us do this efficiently.

Recall that `select()` indicates which sockets are ready to be read from and which sockets are ready to be written to. The `send()` function should be called only when you know that a socket is ready to be written to. Otherwise, you risk that `send()` may block. In the worst case, `send()` will block indefinitely.

Thus, one procedure to send a large amount of data is as follows:

1.  Call `send()` with your remaining data.
2.  The return value of `send()` indicates how many bytes were actually consumed by `send()`. If fewer bytes were sent than you intended, then your next call to `send()` should be used to transmit the remainder.
3.  Call `select()` with your socket in both the read and write sets.
4.  If `select()` indicates that the socket is ready to be read from, call `recv()` on it and handle the received data as needed.
5.  If `select()` indicates that the socket is ready to write to again, go to *step 1* and call `send()` with the remaining data to be sent.

The important point is that calls to `send()` are interspersed with calls to `recv()`. In this way, we can be sure that no data is lost, and this deadlock condition does not occur.

This method also neatly extends to applications with many open sockets. Each socket is added to the `select()` call, and ready sockets are serviced as needed. Your application will need to keep track of which data is remaining to be sent for each connection.

It should also be noted that setting sockets to a non-blocking mode can simplify your program's logic in some cases. Even with non-blocking sockets, `select()` can still be used as a central blocking point to wait for socket events.

Two files are included with this chapter's code repository that can help to demonstrate the deadlock state and how `select()` can be used to prevent it. The first file, `server_ignore.c`, implements a simple TCP server that accepts connections and then ignores them. The second file, `big_send.c`, initiates a TCP connection and then attempts to send lots of data. By using the `big_send` program to connect to the `server_ignore` program, you can investigate the blocking behavior of `send()` for yourself.

Deadlocks represent only one way a TCP connection can unexpectedly fail. While deadlocks can be very difficult to diagnose, they are preventable with careful programming. Besides the risk for deadlock, TCP also presents other data transfer pitfalls. Let's consider another common performance problem next.

# Congestion control

As we've just seen, TCP implements flow control to prevent a sender from overwhelming a receiver. This flow control works by allowing only a limited number of TCP segments to be sent before requiring an acknowledgment of receipt.

TCP also implements **congestion control** methods as part of a network **congestion-avoidance** scheme. While flow control is vital to prevent overwhelming the receiver, congestion control is essential to prevent overwhelming the network.

One way TCP congestion control works is by allowing only a limited amount of data to be sent before pausing to wait for an acknowledgment of receipt. This data limit is decreased when network congestion is detected. In this way, TCP doesn't try putting more data over the network than the network can handle.

Another way TCP implements congestion control is through the **TCP slow start algorithm**. This method provides a way for TCP to ramp-up a connection to its full potential, instead of immediately dumping a lot of data on the network all at once.

It works like this—when a new TCP connection is established, only a minimal amount of data is allowed to be sent unacknowledged. When this data is acknowledged, the limit is increased. Each time a new acknowledgment is received, the limit is increased further until packet loss happens or the limit reaches the desired maximum.

The following diagram shows a TCP slow-start in action:

![](img/41e61db5-bfc4-4436-8ef0-36829b4f3203.png)

In the preceding diagram, you can see that the **Client** starts by sending only a little data. Once that data is acknowledged, the **Client** is willing to send a larger amount of data before requiring another acknowledgment. Once that acknowledgment is received, the **Client** increases its limit again, and so on.

The slow-start algorithm can cause problems for short-lived connections. In practice, if a connection needs to send only a small amount of data, that connection won't ever reach its full potential. This has caused many protocols to be designed around keeping connections open for longer. For example, it used to be common for an HTTP connection to transmit only one resource. Now it is far more common for an HTTP connection to be held open for additional resources, one after another. This connection reuse avoids the overhead of the TCP three-way handshake and slow start.

In addition to avoiding congestion, TCP also provides methods to increase bandwidth efficiency.

# The Nagle algorithm

One technique used by TCP increase efficiency is the **Nagle algorithm**. The Nagle algorithm works to make the sender pool small amounts of data together until it has enough to justify sending.

Consider sending just one byte of data over a TCP connection. Each TCP segment uses 20 bytes to transmit TCP bookkeeping. An additional 20 bytes are needed for the IPv4 header. So, this 1 byte of application data becomes 41 bytes on the network. That's an overhead of 4,000%, and we are not even counting the overhead from lower layers (for example, the Ethernet frame overhead) yet!

The Nagle algorithm states that only one small, unacknowledged TCP segment may be outstanding at any given time. A small segment is considered any segment less than the **Maximum Segment Size** (**MSS**).

Let's see how this applies to a program doing small writes. Consider the following code called on a connected, but otherwise idle, TCP socket:

```cpp
send(my_socket, "a", 1, 0);
send(my_socket, "b", 1, 0);
```

After the first `send()` call, the `a` data is packed into a TCP message and sent off, along with its 40 bytes of TCP and IPv4 overhead.

The second `send()` call returns immediately, but the `b` data isn't actually sent immediately. The Nagle algorithm causes the `b` data to be queued-up by the operating system. It won't be sent until either the first TCP message is acknowledged or `send()` is called again with enough additional data to fill up an entire max-size TCP segment.

For both `a` and `b` to be received by the recipient, it will take the duration of one round-trip network time, plus an additional one-way network time.

We can easily get this 1.5 round-trip network time down to 0.5 round-trip network time by just using the following code:

```cpp
send(my_socket, "ab", 2, 0);
```

For this reason, you should always prefer doing one large write to `send()` instead of many small ones, whenever possible. Doing one large write allows `ab` to be sent in the same TCP message, thereby sidestepping the Nagle algorithm altogether.

In some applications, you really do need to send a small packet followed by another small packet immediately after. For example, in a real-time multiplayer video game, you can't queue up player commands; they must be sent continuously. In these cases, it makes sense to disable the Nagle algorithm for reduced latency, at the expense of decreased bandwidth efficiency.

Disabling the Nagle algorithm can be done using the `setsockopt()` function. The following code shows this method in action:

```cpp
int yes = 1;
if (setsockopt(my_socket, IPPROTO_TCP, TCP_NODELAY,
        (void*)&yes, sizeof(yes)) < 0) {
    fprintf(stderr, "setsockopt() failed. (%d)\n", GETSOCKETERRNO());
}
```

Be sure to consider all your options before disabling Nagle. When faced with a poorly performing network program, some programmers will disable the Nagle algorithm as a first step. In reality, the decision to disable the Nagle algorithm should be approached cautiously. Disabling the Nagle algorithm in real-time applications often makes sense. Disabling it in other contexts rarely does.

For example, imagine that you've implemented an HTTP client. It seems a bit sluggish, and so you try disabling the Nagle algorithm. You do that and find that it runs much faster now. However, by disabling the Nagle algorithm, you've increased network overhead. You could have gotten the same improvement by simply pooling together your `send()` calls.

If you're implementing a real-time algorithm that does need to send small time-critical packets, using `TCP_NODELAY` may still not be the right method for you. TCP can introduce delays in many other ways. For example, if one TCP packet is lost over the network, no further data can be delivered until that packet is retransmitted. This can have the effect of delaying many packets because of one hold-up.

Many real-time applications prefer using UDP over TCP. Each UDP packet is entirely independent of any other packets sent before or after. Of course, the trade-off is that there is a lesser guarantee of reliable delivery; messages may arrive in a different order than they were set, and some messages may arrive twice. Nonetheless, many applications can tolerate this. Real-time video streaming, for example, can use UDP, where each packet stores a very short, time-stamped part of the video. If a packet is lost, there is no need to retransmit; the video stutters for a moment and resumes when the next packet arrives. Packets received late, or out of order, are safely ignored.

Although the Nagle algorithm often works well to improve network utilization, not understanding how it works can lead to problems. In addition to the Nagle algorithm, TCP implements many other methods to limit the needless waste of network resources. Sometimes, these other methods work poorly with one another. The **delayed ACK** is one such method that can work badly with the Nagle algorithm.

# Delayed acknowledgment

We've seen that many client-server protocols work by having the client send a request and then the server send a response. We've also seen that when a TCP peer reads data off the network, it sends an acknowledgment to let the sender know that the data was received successfully.

A typical client-server interchange might, therefore, look like the following:

![](img/99e6e139-9d54-45e9-b163-fb05f28217f7.png)

In the preceding diagram, the **Client** first sends a request to the **Server**. The **Server** reads this request, and a TCP **ACK Message** is sent back to the **Client**. The **Server** then processes the request data and replies with its response.

Some TCP stacks implement a **delayed acknowledgment** method to reduce network congestion. This technique works by delaying the acknowledgment of received data. The hope is that the receiver is going to send a response very soon anyway, and that the acknowledgment can piggyback on this response. When it works, which is often, it conserves bandwidth.

If the receiver doesn't send a reply, the acknowledgment is sent after a short delay; 200 milliseconds is typical.

If the server from before implements delayed acknowledgment, the client-server interchange might look like the following:

![](img/34dd12d7-990d-43dd-896b-57572f2754b3.png)

This is when delayed acknowledgment works well.

Now, consider combining the Nagle algorithm with delayed acknowledgment. If the client transmits its request in two small messages, then the sending channel is blocked for not only the round-trip time. It is also blocked for the additional acknowledgment delay time.

This is illustrated in the following diagram:

![](img/9dbdd387-cc7c-4913-9c8b-226f395e6a51.png)

In the preceding diagram, we see that the **Client** sent the first part of its request in a small packet. The Nagle algorithm prevents it from sending the second part of its request until it receives an acknowledgment from the **Server**. Meanwhile, the **Server** receives the request, but it delays acknowledgment in the hope that it can piggyback the **ACK Message** on the reply. The **Server** processes the first part of the request and sees that it doesn't have the full request yet, so it cannot send a reply. After the delay period elapses, the **Server** does eventually send an **ACK Message**. The **Client** receives this **ACK Message** and sends the rest of the reply. The **Server** replies with its response.

In this degenerate case, the interaction of the Nagle algorithm and the delayed acknowledgment technique cased the **Client**-**Server** interaction to take two full round-trip network times plus the delayed acknowledgment time (which could itself be many round-trip times).

Some programmers jump in these situations to disable the Nagle algorithm. Sometimes that is needed, but often it is the wrong solution.

In our example, merely passing larger data buffers to `send()` completely solves the degenerate interaction. Passing the entire request to `send()` in one call reduces the transaction time from two round-trips plus delay to one round-trip and no delay.

My advice is to prefer calling `send()` with one large write instead of multiple small writes, whenever possible. Of course, if you're implementing a real-time application with TCP, then you can't pool `send()` calls. In that case, disabling the Nagle algorithm can be the correct call.

For the sake of completeness, it should be noted that a delayed ACK can usually be disabled. This is done by passing `TCP_QUICKACK` to `setsockopt()` on systems that support it. Again, this is not usually needed.

Now that we've reviewed a few hidden problems that can crop up with active TCP connections, it's time to move on to connection teardown.

# Connection tear-down

The way a TCP connection transitions from an established connection to a closed one is nuanced. Let's consider this in more detail.

TCP connections are **full-duplex**. This means that the data being sent is independent of the data being received. Data is sent and received simultaneously. This also implies that the connection must be closed by both sides before it is truly disconnected.

To close a TCP connection, each side sends a **Finish** (**FIN**) message and receives an ACK message from their peer.

The exact tear-down process, from the perspective of each peer, depends on whether it sent a FIN first, or received a FIN first. There are three basic connection tear-down cases. They are as follows:

1.  You initiate the tear-down by sending the first FIN message
2.  You receive a FIN message from your connected peer
3.  You and your peer send FIN messages simultaneously

In case 3, where both sides send a FIN message simultaneously, each side thinks that it is in case 1\. That is, each side thinks that it has sent the first FIN message, and each side tears down its socket as in case 1\. In practice, this is pretty rare, but certainly possible.

When a TCP socket is open for full-duplex communication, it is said to be in the `ESTABLISHED` state. The closing initiator sends a FIN message to its peer. The peer replies with an ACK. At this point, the connection is only half closed. The initiator can no longer send data, but it can still receive data. The peer has the option to continue to send more data to the closing initiator. When the peer is ready to finish closing the connection, it sends its own FIN message. The initiator then responds with the final ACK message, and the connection is fully closed.

The TCP connection state transitions on the initiator are `ESTABLISHED`, `FIN-WAIT-1`, `FIN-WAIT-2`, `TIME-WAIT`, and `CLOSED`. The TCP connection state transitions on the receiving peer are `ESTABLISHED`, `CLOSE-WAIT`, `LAST-ACK`, and `CLOSED`.

The following diagram illustrations the normal TCP four-way closing handshake:

![](img/b0d1267f-37e2-4887-be22-f59535f917c2.png)

It is sometimes possible for the **Peer** to combine its **ACK Message** and **FIN Message** into one message. In that case, the connection can be torn down with only three messages, instead of four.

In the case where both sides initiate the tear-down simultaneously, both sides follow the state transition of the **Initiator**. The messages sent and received are the same.

Networks are inherently unreliable, so there is a chance that the final **ACK Message** sent by the **Initiator** will be lost. In this case, the **Peer**, having not received an **ACK Message**, resends its **FIN Message**. If the **Initiator** had completely **CLOSED** its socket after sending the final **ACK Message**, then it would be impossible to reply to this resent **FIN Message**. For this reason, the **Initiator** enters a `TIME-WAIT` state after sending the last **ACK Message**. During this `TIME-WAIT` state, it responds to any retransmitted **FIN Message** from the **Peer** with an **ACK Message**. After a delay, the **Initiator** leaves the `TIME-WAIT` state and fully closes its socket.

The `TIME-WAIT` delay is usually on the order of one minute, but it could be configured for much longer.

In this book, we've used only the `close()` function (`closesocket()` on Windows) to disconnect a socket. This function, although simple to use, has the disadvantage of always fully closing a socket. That is, no data can be sent or received on a socket called with `close()`. The TCP teardown handshake does allow for data to be received after a FIN message has been sent. Let's next consider how to do this programmatically.

# The shutdown() function

As we've just seen, TCP connections are torn down in two steps. The first one side sends a FIN message, and then the other side does. However, each side is allowed to continue to send data until it has sent its own FIN message.

We've used the `close()` function (`closesocket()` on Windows) to disconnect sockets because of its simplicity. The `close()` function, however, closes both sides of a socket. If you use `close()` in your application, and the remote peer tries to send more data, it will cause an error. Your system will then transmit a **Reset** (**RST**) message to indicate to the peer that the connection was not closed in an orderly manner.

If you want to close your sending channel, but still leave the option for receiving more data, you should use the `shutdown()` function instead. The `shutdown()` function takes two parameters. The first parameter is a socket, and the second is an `int` indicating how to shut down the socket.

In theory, `shutdown()` supports three options—closing the sending side of a connection, closing the receiving side, and closing both sides. However, the TCP protocol itself doesn't reflect these options, and it is rarely useful to use `shutdown()` for closing the receiving side.

There is a small portability issue about `shutdown()` functions parameters. Under windows, you want to call it with `SD_SEND`. On other systems, you should use `SHUT_WR`. Both values are defined as `1`, so you can also call it that way.

The code to shutdown the sending channel of a socket in a cross-platform manner is as follows:

```cpp
if (shutdown(my_socket, 1) /* 1 = SHUT_WR, SD_SEND */) {
    fprintf(stderr, "shutdown() failed. (%d)\n", GETSOCKETERRNO());
}
```

This use of `shutdown()` causes the TCP FIN message to be transmitted after the transmission queued is emptied.

You may wonder, if you're receiving data from a peer, and `recv()` returns `0`, how you know whether your peer has called `shutdown()` or `close()`? Unfortunately, you can't know, except by prior agreement. If they have used `shutdown()` only to close their sending data channel, then they are still receptive to additional data. If they instead used `close()`, additional data will trigger an error state.

Although half-closed connections have their uses, it is often easier to use an application protocol that clearly indicates the end of the transaction. For example, consider the HTTP protocol covered in [Chapter 6](de3d2e9b-b94e-47d1-872c-c2ecb34c4026.xhtml), *Building a Simple Web Client*. With the HTTP protocol, the client indicates the end of its request with a blank line. The server knows it has the full request when it sees this blank line. The server then specifies how much data it will be sending with the `Content-Length` header. Once the client has received that much data, it knows that it hasn't missed anything. The client can then call `close()` and be confident that the server won't be sending additional data.

In many applications, knowing whether the shutdown was orderly isn't always useful. Consider the chat room program (`tcp_serve_chat.c`) from [Chapter 3](eb2e080f-fad2-4e14-b4c1-9a6c124df77c.xhtml), *An In-Depth Overview of TCP Connections*. This program has no real application protocol. That program simply sends messages from one client to every other client. When a client decides to disconnect, it isn't important that it continues to receive data from the server. Guaranteeing an orderly TCP release would provide no benefit.

So, when is `shutdown()` useful? Basically, a TCP orderly release is useful when the application protocol doesn't have a way to signal that it has finished sending data, and your application isn't tolerant of missed data. In that case, `shutdown()` is a useful signal.

Please note that if you're using threading or forking, there are additional differences to the behavior of `close()` versus `shutdown()` that must be considered. When `shutdown()` is called, it always affects the socket. The `close()` function, by contrast, has no effect if additional processes also hold handles to the socket.

Finally, note that `close()` must still eventually be called on a socket closed with `shutdown()` in order to release associated system resources.

Another issue that comes up around the TCP tear-down procedure is the long delay for the side that initiated the close to remain in the `TIME-WAIT` state. This can sometimes cause problems for TCP servers. Let's look at that next.

# Preventing address-in-use errors

If you do TCP server programming for very long, you will eventually run into the following scenario—your TCP server has one or more open connections, and then you terminate it (or it crashes). You restart the server, but the call to `bind()` fails with an `EADDRINUSE` (`WSAEADDRINUSE` on Windows) error.

When this happens, you can wait a few moments, try it again, and it works. What's going on here?

Essentially, when an application initiates a TCP socket close (or causes the disconnection by crashing), that socket goes into the `TIME-WAIT` state. The operating system continues to keep track of this socket for some time, potentially minutes.

An example program, `server_noreuse.c`, is included in this chapter's code repo. You can reproduce this address-in-use problem by running it, accepting a connection, and then terminating `server_noreuse`. To reproduce the problem, it is vital that the server is the one to terminate the open connection, not the client.

If you immediately start `server_noreuse` again, you will see the `bind()` error.

The following screenshot shows this on a Linux desktop:

![](img/41a33bfe-17d9-4f2f-83fc-c8343bc0c0a0.png)

You can use the `netstat` command to see these half-dead connections that are preventing our server from starting. The following command shows which connections are stuck in the `TIME-WAIT` state on Linux:

```cpp
netstat -na | grep TIME
```

As long as one of these connections is hanging on, it prevents any new process from calling `bind()` on the same local port and address.

This failure of the `bind()` call can be prevented by setting the `SO_REUSEADDR` flag on the server socket before calling `bind()`.

The following code demonstrates this:

```cpp
int yes = 1;
if (setsockopt(my_socket, SOL_SOCKET, SO_REUSEADDR,
        (void*)&yes, sizeof(yes)) < 0) {
    fprintf(stderr, "setsockopt() failed. (%d)\n", GETSOCKETERRNO());
}
```

Once the `SO_REUSEADDR` flag is set, `bind()` succeeds even if a few `TIME-WAIT` connections are still hanging on to the same local port and address.

An example program, `server_reuse.c`, is included to demonstrate this technique.

I suggest that you always use `SO_REUSEADDR` for TCP servers because there are few downsides. The only real drawback is that using `SO_REUSEADDR` allows your program to bind to a specific interface even if another program has already bound to the wildcard address. Usually, this isn't a problem, but it is something to keep in mind.

You may sometimes see programs that attempt to fix this issue by killing sockets in the `TIME-WAIT` state. This can be accomplished by setting the socket linger option. This is dangerous! The `TIME-WAIT` state is essential to TCP's reliability, and interfering with it can lead to severe problems.

Why is this address-in-use only a problem for servers and not clients? Because the problem manifests itself when calling `bind()`. Client programs don't usually call `bind()`. If they do, this can also be a problem on the client-side.

While we are still on the topic of disconnected sockets, what happens when you try to send data to a peer that has already called `close()`? Let's consider that next.

# Sending to a disconnected peer

There are three basic ways a TCP connection can fail. They are as follows:

*   A network outage
*   The peer application crashes
*   The peer's system crashes

A network outage prevents data from reaching your peer. In this case, TCP tries to retransmit data. If connectivity is re-established, TCP simply picks back up where it left off. Otherwise, the connection eventually times out. This timeout can be on the order of 10 minutes.

The second way a TCP connection can fail is if the connected peer application crashes. In this case, the peer's operating system sends a FIN message. This case is indistinguishable from the peer calling `close()` on their end. If your application continues to send data after having received the FIN message, the peer's system will send an RST message to indicate an error.

Finally, a connection could fail because the peer's whole system has crashed. In this case, it won't be able to send a FIN message. This case looks the same as a network outage, and the TCP connection would eventually timeout. However, consider what happens if the crashed system reboots before the connection times out. In that case, the rebooted system will eventually receive a TCP message from the original connection. The rebooted system will not recognize the TCP connection and will send an RST message in response to indicate an error condition.

To reiterate, if you use `send()` on a socket that your peer thinks is closed, that peer will respond with an RST message. This state is easily detected by the return value of `recv()`.

A more serious issue to consider is what happens when `send()` is called on a socket that has already received an RST message from its peer. On Unix-based systems, the default is to send a `SIGPIPE` signal to your program. If you don't handle this signal, the operating system will terminate your program.

It is therefore essential for TCP servers to either handle or disable the `SIGPIPE` signal. Failure to handle this case means that a rude client could kill your server.

Signals are complicated. If you're already using signals in your program, you may want to handle `SIGPIPE`. Otherwise, I recommend you just disable it by setting the `SIGPIPE` handler to `SIG_IGN`.

The following code disables `SIGPIPE` on Unix-based systems:

```cpp
#if !defined(_WIN32)
#include <signal.h>
#endif

#if !defined(_WIN32)
signal(SIGPIPE, SIG_IGN);
#endif
```

As an alternative, you can use `MSG_NOSIGNAL` with `send()` as shown in the following code:

```cpp
send(my_socket, buffer, sizeof(buffer), MSG_NOSIGNAL);
```

If the signal is ignored or `MSG_NOSIGNAL` is used, `send()` returns `-1` and sets `errno` to `EPIPE`.

On Windows, attempting to call `send()` on a closed socket generally results in `WSAGetLastError()` returning `WSAECONNRESET`.

An example program, `server_crash.c`, is included in this chapter's code repository. This program accepts TCP connections on port `8080`. It then waits for the client to disconnect, and then attempts two sends to that disconnected client. This program is useful as a tool to explore the return values, error codes, and function behavior in different scenarios.

# Socket's local address

When implementing servers, for both TCP and UDP, it is important to bind the listening socket to a local address and port. If the socket isn't bound, then clients can't know where to connect.

It is also possible to use `bind()` on the client side to associate a socket with a particular address and port. It is sometimes useful to use `bind()` in this manner on machines that have multiple network interfaces. The use of `bind()` can allow the selection of which network address to use for the outgoing connection.

Sometimes, `bind()` is used to set the local port for an outgoing connection. This is usually a bad idea for a few reasons. First, it very seldom serves any purpose. The port number presented to the connected server is likely to be different because of network address translation regardless. Binding to a local port also invites the error of selecting a port that is already in use. Usually, the operating system takes care of selecting a free port. This use of `bind()` also raises the issue with `TIME-WAIT`, which would prevent a new connection from being established after a closed one without a substantial delay.

We have used `bind()` in this book mostly for binding servers to a particular port number. It can also be used to associate servers to a particular address. If a server has multiple network interfaces, it may be the case that you only care to listen on connections at one address. In this case, `bind()` can easily be used to limit the connections to that address. It can also be used to limit connections to the local machine by binding sockets to `127.0.0.1`. This can be an important security measure for some applications.

We have employed the `select()` function for many purposes—timing out `connect()`, signaling when data is available, and preventing `send()` from blocking. However, `select()` is only suitable for monitoring a limited number of sockets. Let's look at this limitation, and how to circumvent it, next.

# Multiplexing with a large number of sockets

We've used `select()` in this book to multiplex between open sockets. The `select()` function is great because it is available on many platforms. However, if you have a large number of open sockets, you can quickly run into the limitations of `select()`.

There is a maximum number of sockets you can pass to `select()`. This number is available through the `FD_SETSIZE` macro.

This chapter's code repository includes a program, `setsize.c`, which prints the value of `FD_SETSIZE`.

The following screenshot shows this program being compiled and run on Windows 10:

![](img/3507d567-9997-464f-8647-8c1ced30825c.png)

The preceding screenshot shows `FD_SETSIZE` is `64` on this system. Although Windows's default size for `FD_SETSIZE` is quite low, it is common to see higher values on other systems. The default value of `FD_SETSIZE` on Linux is `1024`.

On Windows, it is possible to increase `FD_SETSIZE` easily. You only need to define `FD_SETSIZE` yourself before including the `winsock2.h` header. For example, the following code increases `FD_SETSIZE` to `1024` on Windows:

```cpp
#ifndef FD_SETSIZE
#define FD_SETSIZE 1024
#endif
#include <winsock2.h>
```

This works because Winsock uses `FD_SETSIZE` to build the `fd_set` type.

On Linux, this trick does not work. Linux defines `fd_set` as a bitmask, and it is not possible to increase its size without recompiling the kernel.

There are possible workarounds to effectively cheat `select()` into accepting socket descriptors larger than `1023` on Linux. One trick that usually works is to allocate an array of `fd_set` variables. Setting a socket is then done like this:

```cpp
FD_SET(s % FD_SETSIZE, &set_array[s / FD_SETSIZE])
```

However, if you have to resort to a hack such as the preceding code, you may be better off avoiding `select()` and using a different multiplexing technique. The `poll()` function, for example, provides the functionality of `select()` without a limit on the number of file descriptors it can handle.

# Summary

We covered a lot of ground in this chapter. First, we reviewed error-handling methods, and then we implemented a function to obtain text descriptions for error codes.

We then jumped right into the hard details of TCP sockets. We saw how TCP sockets hide much complexity, and how it is sometimes necessary to understand that hidden state to get good application performance. We saw a method for an early timeout on a TCP `connect()` call, and we looked at how to terminate a connection with an orderly release.

We then took a closer look at the `bind()` function and how its usefulness differs between servers and clients. Finally, we discussed how the `select()` function limits the total number of sockets your program can handle, and how to work around it.

So far, this book has been focused mainly on network code as it would pertain to personal computers and servers. In the next chapter, [Chapter 14](c8466f85-a6e3-4d33-beb7-0a9f38d35062.xhtml), *Web Programming for the Internet of Things*, we move our focus to the extending of internet access to everyday objects—that is, the **Internet of Things**.

# Questions

Try these questions to test your knowledge acquired from this chapter:

1.  Is it ever acceptable to just terminate a program if a network error is detected?
2.  Which system functions are used to convert error codes into text descriptions?
3.  How long does it take for a call to `connect()` to complete on a TCP socket?
4.  What happens if you call `send()` on a disconnected TCP socket?
5.  How can you ensure that the next call to `send()` won't block?
6.  What happens if both peers to a TCP connection try to send a large amount of data simultaneously?
7.  Can you improve application performance by disabling the Nagle algorithm?
8.  How many connections can `select()` handle?

The answers to these questions are found in [Appendix A](bd8b8f52-52cb-4d34-b01b-e907564bfece.xhtml), *Answers to Questions*.