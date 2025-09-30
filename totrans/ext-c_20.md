# Chapter 20

# Socket Programming

In the previous chapter, we discussed single-host IPC and gave an introduction to socket programming. In this chapter, we want to complete our introduction and address socket programming in depth using a real client-server example: the calculator project.

The order of topics in this chapter might seem a bit unusual, but the purpose is to give you a better understanding about various types of sockets and how they behave in a real project. As part of this chapter, we discuss the following topics:

*   Firstly, we give a review on what we explained in the previous chapter. Note that this review is just a short recap, and it is a must for you to read the second part of the previous chapter dedicated to socket programming.
*   As part of the recap we discuss various types of sockets, stream and datagram sequences, and some other topics that are essential for our continuation of our calculator example.
*   The client-server example, the calculator project, is described and fully analyzed. This prepares us to continue with various components in the example and to present C code.
*   As a critical component of the example, a serializer/deserializer library is developed. This library is going to represent the main protocol used between a calculator client and its server.
*   It is crucial to understand that a calculator client and a calculator server must be able to communicate over any type of socket. Therefore, we present various types of sockets integrated within the example and as the starting point, **Unix domain sockets** (**UDS**) are introduced.
*   We show in our example how they are used to establish a client-server connection in a single-host setup.
*   To continue with other types of sockets, we discuss network sockets. We present how TCP and UDP sockets can be integrated within the calculator project.

Let's begin the chapter with a summary of what we know about sockets and socket programming in general. It is highly recommended that you familiarize yourself with the second half of the previous chapter before delving into this chapter, as we assume some pre-existing knowledge here.

# Socket programming review

In this section, we are going to discuss what sockets are, what their various types are, and generally what it means if we say that we are doing socket programming. This is going to be a short review, but it is essential to build this basis so that we can continue into deeper discussion in subsequent sections.

If you remember from the previous chapters, we have two categories of IPC techniques to be used by two or more processes to communicate and share data. The first category contains *pull-based* techniques that require an accessible *medium* (such as a shared memory or a regular file) to store data to and retrieve data from. The second category contains *push-based* techniques. These techniques require a *channel* to be established and the channel should be accessible by all processes. The main difference between these categories is regarding the way that data is retrieved from a medium in pull-based techniques, or a channel in push-based techniques.

To put it simply, in pull-based techniques, the data should be pulled or read from the medium, but in push-based techniques the data is pushed or delivered to the reader process automatically. In pull-based techniques, since the processes pull data from a shared medium, it is prone to race conditions if a number of them can write to that medium.

To be more exact about push-based techniques, the data is always delivered to a buffer in the kernel and that buffer is accessible to the receiver process through using a descriptor (file or socket).

Then the receiver process can either block until some new data is available on that descriptor or it can *poll* the descriptor to see if the kernel has received some new data on that descriptor and if not, continue to some other work. The former approach is *blocking I/O* and the latter is *non-blocking I/O* or *asynchronous I/O*. In this chapter, all push-based techniques use the blocking approach.

We know that socket programming is a special type of IPC that belongs to the second category. Therefore, all socket-based IPCs are push-based. But the main characteristic that distinguishes socket programming from other push-based IPC techniques is the fact that in socket programming we use *sockets*. Sockets are special objects in Unix-like operating systems, even in Microsoft Windows which is not Unix-like, that represent *two-way channels*.

In other words, a single socket object can be used to both read from and write to the same channel. This way, two processes located at two sides of the same channel can have *two-way communication*.

In the previous chapter, we saw that sockets are represented by socket descriptors, just like files that are represented by file descriptors. While socket descriptors and file descriptors are similar in certain ways such as I/O operation and being *poll-able*, they are in fact different. A single socket descriptor always represents a channel, but a file descriptor can represent a medium such as a regular file, or a channel like a POSIX pipe. Therefore, certain operations related to files such as seek are not supported for socket descriptors, and even for a file descriptor when it represents a channel.

Socket-based communication can be *connection-oriented* or *connection-less*. In connection-oriented communication, the channel represents a *stream* of bytes being transmitted between two specific processes, while in connection-less communication, *datagrams* can be transmitted along the channel and there is no specific connection between two processes. A number of processes can use the same channel for sharing states or transmitting data.

Therefore, we have two types of channels: *stream channels* and *datagram channels*.In a program, every stream channel is represented by a *stream socket* and every datagram channel is represented by a *datagram socket*. When setting up a channel, we have to decide if it should be either stream or datagram. We shortly see that our calculator example can support both channels.

Sockets have various types. Each type of socket exists for a certain usage and a certain situation. Generally, we have two types of socket: UDS and network sockets. As you may know and as we've explained in the previous chapter, UDS can be used whenever all the processes willing to participate in an IPC are located on the same machine. In other words, UDS can be used only in single-host deployments.

In contrast, network sockets can be used in almost any deployment no matter how processes are deployed and where they are located. They can be all on the same machine, or they can be distributed throughout a network. In case of having a single-host deployment, UDS are preferred because they are faster, and they have less overhead in comparison to network sockets. As part of our calculator example, we provide the support for both UDS and network sockets.

UDS and network sockets can represent both stream and datagram channels. Therefore, we have four varieties: UDS over a stream channel, UDS over a datagram channel, network socket over a stream channel, and finally network socket over a datagram channel. All these four variations are covered by our example.

A network socket offering a stream channel is usually a TCP socket. That's because, most of the time, we are using TCP as the transport protocol for such a socket. Likewise, a network socket offering a datagram channel is usually a UDP socket. That's because, most of the time, we are using UDP as the transport protocol for such a socket. Note that UDS socket offering either stream or datagram channels don't have any specific names because there is no underlying transport protocol.

In order to write actual C code for the different types of sockets and channels, it is better to do it when you are working on a real example. That's basically why we have taken this unusual approach. This way, you'll notice the common parts between various types of sockets and the channels, and we can extract them as units of code that can be reused again. In the next section, we are going to discuss the calculator project and its internal structure.

# Calculator project

We are dedicating a separate section to explain the purpose of the calculator project. It is a lengthy example and thus it will be helpful to have a firm grounding before diving into it. The project should help you to achieve the following goals:

*   Observe a fully functional example that has a number of simple and well-defined functionalities.
*   Extract common parts among the various types of sockets and channels and have them as some reusable libraries. This reduces the amount of code we write significantly, and from a learning point of view, it shows you the boundaries that are common between various types of sockets and channels.
*   Maintain communication using a well-defined application protocol. Ordinary socket programming examples lack this very important feature. They generally address very simple, and usually one-time, communication scenarios between a client and its server.
*   Work on an example that has all the ingredients required for a fully functional client-server program such as an application protocol, supporting various types of channels, having serializer/deserializer, and so on, giving you a different perspective regarding socket programming.

With all that being said, we are going to present this project as our main example in this chapter. We do it step by step, and I will guide you through the various steps that culminate in a complete and working project.

The first step is to come up with a relatively simple and complete application protocol. This protocol is going to be used between the clients and the server. As we explained before, without a well-defined application protocol, the two parties cannot communicate. They can be connected and transmit data because that's the functionality that the socket programming offers, but they cannot understand each other.

That's why we have to dedicate a bit of time to understand the application protocol used in the calculator project. Before talking about the application protocol, let's present the source hierarchy that can be seen in the project code base. Then, we can find the application protocol and the associated serializer/deserializer library much easier in the project code base.

## Source hierarchy

From a programmer's point of view, the POSIX socket programming API treats all the stream channels the same no matter whether the associated socket object is a UDS or a network socket. If you remember from the previous chapter, for stream channels, we had certain sequences for the listener-side and for the connector-side, and these sequences remain the same for different types of stream sockets.

Therefore, if you are going to support various types of sockets, together with various types of channels, it is better to extract the common part and write it once. That's exactly the approach that we take regarding the calculator project and that's what you see in the source code. Therefore, it is expected to see various libraries in the project and some of them contain the common code that is reused by other parts of the code.

Now, it's time to delve into the code base. First of all, the source code of the project can be found here: https://github.com/PacktPublishing/Extreme-C/tree/master/ch20-socket-programming. If you open the link and have a look at the code base, you see there are a number of directories that contain multiple source files. Obviously, we cannot demonstrate all of them because this would take too long, but we are going to explain important parts of the code. You are encouraged to look at the code and go through it, then try to build and run it; this will give you an idea of how the example has been developed.

Note that all the code relating to the examples of UDS, UDP sockets, and TCP sockets has been put in a single hierarchy. Next, we are going to explain the source hierarchy and the directories you find as part of the code base.

If you go to the root of the example and use the `tree` command to show the files and directories, you will find something similar to *Shell Box 20-1*.

The following shell box demonstrates how to clone the book's GitHub repository and how to navigate to the root of the example:

```cpp
$ git clone https://github.com/PacktPublishing/Extreme-C
Cloning into 'Extreme-C'...
...
Resolving deltas: 100% (458/458), done.
$ cd Extreme-C/ch20-socket-programming
$ tree
.
├── CMakeLists.txt
├── calcser
...
├── calcsvc
...
├── client
│   ├── CMakeLists.txt
│   ├── clicore
...
│   ├── tcp
│   │   ├── CMakeLists.txt
│   │   └── main.c
│   ├── udp
│   │   ├── CMakeLists.txt
│   │   └── main.c
│   └── Unix
│       ├── CMakeLists.txt
│       ├── datagram
│       │   ├── CMakeLists.txt
│       │   └── main.c
│       └── stream
│           ├── CMakeLists.txt
│           └── main.c
├── server
│   ├── CMakeLists.txt
│   ├── srvcore
...
│   ├── tcp
│   │   ├── CMakeLists.txt
│   │   └── main.c
│   ├── udp
│   │   ├── CMakeLists.txt
│   │   └── main.c
│   └── Unix
│       ├── CMakeLists.txt
│       ├── datagram
│       │   ├── CMakeLists.txt
│       │   └── main.c
│       └── stream
│           ├── CMakeLists.txt
│           └── main.c
└── types.h
18 directories, 49 files
$
```

Shell Box 20-1: Cloning the calculator project's code base and listing the files and directories

As you can see in the listing of files and directories, the calculator project is made up of a number of parts, some of them being libraries, and each of them having its own dedicated directory. Next, we explain these directories:

*   `/calcser`: This is the serializer/deserializer library. It contains the serialization/deserialization-related source files. This library dictates the application protocol that is defined between a calculator client and a calculator server. This library is eventually built into a static library file named `libcalcser.a`.
*   `/calcsvc`: This library contains the sources for the calculation service. The *calculation service* is different from the server process. This service library contains the core functionality of the calculator and it is agnostic regarding being behind a server process and can be used individually as a separate standalone C library. This library eventually gets built into a static library file named `libcalcsvc.a`.
*   `/server/srvcore`: This library contains the sources that are common between the stream and the datagram server processes, regardless of the socket type. Therefore, all calculator server processes, whether using UDS or network sockets, and whether operating on stream or datagram channels, can rely on this common part. The final output of this library is a static library file named `libsrvcore.a`.
*   `/server/unix/stream`: This directory contains the sources for a server program using stream channels behind a UDS. The final build result of this directory is an executable file named `unix_stream_calc_server`. This is one of the possible output executables in this project that we can use to bring up a calculator server, this one listening on a UDS to receive stream connections.
*   `/server/unix/datagram`: This directory contains the sources for a server program using datagram channels behind a UDS. The final build result of this directory is an executable file named `unix_datagram_calc_server`. This is one of the possible output executables in this project that we can use to bring up a calculator server, this one listening on a UDS to receive datagram messages.
*   `/server/tcp`: This directory contains the sources for a server program using stream channels behind a TCP network socket. The final build result of this directory is an executable file named `tcp_calc_server`. This is one of the possible output executables in this project that we can use to bring up a calculator server, this one listening on a TCP socket to receive stream connections.
*   `/server/udp`: This directory contains the sources for a server program using datagram channels behind a UDP network socket. The final build result of this directory is an executable file named `udp_calc_server`. This is one of the possible output executables in this project that we can use to bring up a calculator server, this one listening on a UDP socket to receive datagram messages.
*   `/client/clicore`: This library contains the sources that are common between the stream and the datagram client processes, regardless of the socket type. Therefore, all calculator client processes, no matter whether they are using UDS or network sockets, and no matter operating on stream or datagram channels, can rely on this common part. It would be built into a static library file named `libclicore.a`.
*   `/client/unix/stream`: This directory contains the sources for a client program using stream channels behind a UDS. The final build result of this directory is an executable file named `unix_stream_calc_client`. This is one of the possible output executables in this project that we can use to start a calculator client, this one connecting to a UDS endpoint and establishing a stream connection.
*   `/client/unix/datagram`: This directory contains the sources for a client program using datagram channels behind a UDS. The final build result of this directory is an executable file named `unix_datagram_calc_client`. This is one of the possible output executables in this project that we can use to start a calculator client, this one connecting to a UDS endpoint and sending some datagram messages.
*   `/client/tcp`: This directory contains the sources for a client program using stream channels behind a TCP socket. The final build result of this directory is an executable file named `tcp_calc_client`. This is one of the possible output executables in this project that we can use to start a calculator client, this one connecting to a TCP socket endpoint and establishing a stream connection.
*   `/client/udp`: This directory contains the sources for a client program using datagram channels behind a UDP socket. The final build result of this directory is an executable file named `udp_calc_client`. This is one of the possible output executables in this project that we can use to start a calculator client, this one connecting to a UDP socket endpoint and sending some datagram messages.

## Build the project

Now that we have gone through all the directories in the project, we need to show how to build it. The project uses CMake, and you should have it installed before moving on to build the project.

In order to build the project, run the following commands in the chapter's root directory:

```cpp
$ mkdir -p build
$ cd build
$ cmake ..
...
$ make
...
$
```

Shell Box 20-2: The commands to build the calculator project

## Run the project

There is nothing like running a project to see for yourself how it works. Therefore, before delving into technical details, I want you to bring up a calculator server, and then a calculator client, and finally see how they talk to each other.

Before running the processes, you need to have two separate Terminals (or shells) in order to enter two separate commands. In the first Terminal, in order to run a stream server listening on UDS, type the following command.

Note that you need to be in the `build` directory before entering the following command. The `build` directory was made as part of the previous section, *Build the Project*:

```cpp
$ ./server/unix/stream/unix_stream_calc_server
```

Shell Box 20-3: Running a stream server listening on a UDS

Ensure the server is running. In the second Terminal, run the stream client built for using UDS:

```cpp
$ ./client/unix/stream/unix_stream_calc_client
? (type quit to exit) 3++4
The req(0) is sent.
req(0) > status: OK, result: 7.000000
? (type quit to exit) mem
The req(1) is sent.
req(1) > status: OK, result: 7.000000
? (type quit to exit) 5++4
The req(2) is sent.
req(2) > status: OK, result: 16.000000
? (type quit to exit) quit
Bye.
$
```

Shell Box 20-4: Running the calculator client and sending some requests

As you see in the preceding shell box, the client process has its own command line. It receives some commands from the user, turns them into some requests according to the application protocol, and sends them to the server for further processing. Then, it waits for the response and, as soon as it receives it, prints the result. Note that this command line is part of the common code written for all clients and therefore, no matter the channel type or socket type the client is using, you always see the client command line.

Now, it's time to jump into the details of the application protocol and see how request and response messages look like.

## Application protocol

Any two processes willing to communicate must obey an application protocol. This protocol can be custom, like the calculator project, or it can be one of the well-known protocols like HTTP. We call our protocol the *calculator protocol*.

The calculator protocol is a variable-length protocol. In other words, every message has its own length and every message should be separated from the next one using a delimiter. There is only one type of request message and one type of response message. The protocol is also textual. It means that we use only alphanumerical characters together with a few other characters as valid characters in request and response messages. In other words, the calculator messages are human-readable.

The request message has four fields: *request ID*, *method*, *first operand*, and *second operand*. Every request has a unique ID and the server uses this ID to relate a response to its corresponding request.

The method is an operation that can be performed by the calculator service. Next, you can see the `calcser/calc_proto_req.h` header file. This file describes the calculator protocol's request message:

```cpp
#ifndef CALC_PROTO_REQ_H
#define CALC_PROTO_REQ_H
#include <stdint.h>
typedef enum {
  NONE,
  GETMEM, RESMEM,
  ADD, ADDM,
  SUB, SUBM,
  MUL, MULM,
  DIV
} method_t;
struct calc_proto_req_t {
  int32_t id;
  method_t method;
  double operand1;
  double operand2;
};
method_t str_to_method(const char*);
const char* method_to_str(method_t);
#endif
```

Code Box 20-1 [calcser/calc_proto_req.h]: Definition of the calculator request object

As you can see, we have nine methods defined as part of our protocol. As a good calculator, our calculator has an internal memory, and because of that we have memory operations associated with addition, subtraction, and multiplication.

For example, the `ADD` method simply adds two float numbers, but the `ADDM` method is a variation of the `ADD` method that adds those two numbers together with the value stored in the internal memory, and finally it updates the value in the memory for further use. It is just like when you use the memory button on your desktop calculator. You can find that button marked as +M.

We also have a special method for reading and resetting the calculator's internal memory. The division method cannot be performed on the internal memory, so we don't have any other variation.

Suppose that the client wants to create a request with ID `1000`, using the `ADD` method, and with `1.5` and `5.6` as the operands. In C, it needs to create an object from the `calc_proto_req_t` type (the structure declared in the preceding header as part of *Code Box 20-1*) and fill it with the desired values. Next, you can see how to do it:

```cpp
struct calc_proto_req_t req;
req.id = 1000;
req.method = ADD;
req.operand1 = 1.5;
req.operand2 = 5.6;
```

Code Box 20-2: Creating a calculator request object in C

As we explained in the previous chapter, the `req` object in the preceding code box needs to be serialized to a request message before being sent to the server. In other words, we need to serialize the preceding *request object* to the equivalent *request message*. The serializer in the calculator project, according to our application protocol, serializes the `req` object as follows:

```cpp
1000#ADD#1.5#5.6$
```

Code Box 20-3: The serialized message equivalent to the req object defined in Code Box 20-2

As you can see, the `#` character is used as the *field delimiter*, and the `$` character is used as the *message separator*. In addition, each request message has exactly four fields. A *deserializer* object on the other side of the channel uses these facts to parse the incoming bytes and revive the request object again.

Conversely, the server process needs to serialize the response object while replying to a request. A calculator response object has three fields: *request ID*, *status*, and *result*. The request ID determines the corresponding request. Every request has a unique ID and this way, the server specifies the request that it wants to respond to.

The `calcser/calc_proto_resp.h` header file describes what a calculator response should look like, and you can see it in the following code box:

```cpp
#ifndef CALC_PROTO_RESP_H
#define CALC_PROTO_RESP_H
#include <stdint.h>
#define STATUS_OK              0
#define STATUS_INVALID_REQUEST 1
#define STATUS_INVALID_METHOD  2
#define STATUS_INVALID_OPERAND 3
#define STATUS_DIV_BY_ZERO     4
#define STATUS_INTERNAL_ERROR  20
typedef int status_t;
struct calc_proto_resp_t {
  int32_t req_id;
  status_t status;
  double result;
};
#endif
```

Code Box 20-4 [calcser/calc_proto_resp.h]: Definition of the calculator response object

Similarly, in order to create a *response object* for the preceding request object, `req`, mentioned in *Code Box 20-2*, the server process should do this:

```cpp
struct calc_proto_resp_t resp;
resp.req_id = 1000;
resp.status = STATUS_OK;
resp.result = 7.1;
```

Code Box 20-5: Creating a response object for the request object req defined as part of Code Box 20-2

The preceding response object is serialized as follows:

```cpp
1000#0#7.1$
```

Code Box 20-6: The serialized response message equivalent to the resp object created in the Code Box 20-5

Again, we use `#` as the field delimiter and `$` as the message separator. Note that the status is numerical, and it indicates the success or failure of the request. In the case of failure, it is a non-zero number, and its meaning is described in the response header file, or to be exact, in the calculator protocol.

Now, it is time to talk a bit more about the serialization/deserialization library and what its internals look like.

## Serialization/deserialization library

In the previous section, we described how the request and response messages look like. In this section, we are going to talk a bit more about the serializer and deserializer algorithms used in the calculator project. We are going to use the `serializer` class, with `calc_proto_ser_t` as its attribute structure, for providing the serialization and deserialization functionalities.

As said before, these functionalities are provided to other parts of the project as a static library named `libcalcser.a`. Here, you can see the public API of the `serializer` class found in `calcser/calc_proto_ser.h`:

```cpp
#ifndef CALC_PROTO_SER_H
#define CALC_PROTO_SER_H
#include <types.h>
#include "calc_proto_req.h"
#include "calc_proto_resp.h"
#define ERROR_INVALID_REQUEST          101
#define ERROR_INVALID_REQUEST_ID       102
#define ERROR_INVALID_REQUEST_METHOD   103
#define ERROR_INVALID_REQUEST_OPERAND1 104
#define ERROR_INVALID_REQUEST_OPERAND2 105
#define ERROR_INVALID_RESPONSE         201
#define ERROR_INVALID_RESPONSE_REQ_ID  202
#define ERROR_INVALID_RESPONSE_STATUS  203
#define ERROR_INVALID_RESPONSE_RESULT  204
#define ERROR_UNKNOWN  220
struct buffer_t {
  char* data;
  int len;
};
struct calc_proto_ser_t;
typedef void (*req_cb_t)(
        void* owner_obj,
        struct calc_proto_req_t);
typedef void (*resp_cb_t)(
        void* owner_obj,
        struct calc_proto_resp_t);
typedef void (*error_cb_t)(
        void* owner_obj,
        const int req_id,
        const int error_code);
struct calc_proto_ser_t* calc_proto_ser_new();
void calc_proto_ser_delete(
        struct calc_proto_ser_t* ser);
void calc_proto_ser_ctor(
        struct calc_proto_ser_t* ser,
        void* owner_obj,
        int ring_buffer_size);
void calc_proto_ser_dtor(
        struct calc_proto_ser_t* ser);
void* calc_proto_ser_get_context(
        struct calc_proto_ser_t* ser);
void calc_proto_ser_set_req_callback(
        struct calc_proto_ser_t* ser,
        req_cb_t cb);
void calc_proto_ser_set_resp_callback(
        struct calc_proto_ser_t* ser,
        resp_cb_t cb);
void calc_proto_ser_set_error_callback(
        struct calc_proto_ser_t* ser,
        error_cb_t cb);
void calc_proto_ser_server_deserialize(
        struct calc_proto_ser_t* ser,
        struct buffer_t buffer,
        bool_t* req_found);
struct buffer_t calc_proto_ser_server_serialize(
        struct calc_proto_ser_t* ser,
        const struct calc_proto_resp_t* resp);
void calc_proto_ser_client_deserialize(
        struct calc_proto_ser_t* ser,
        struct buffer_t buffer,
        bool_t* resp_found);
struct buffer_t calc_proto_ser_client_serialize(
        struct calc_proto_ser_t* ser,
        const struct calc_proto_req_t* req);
#endif
```

Code Box 20-7 [calcser/calc_proto_ser.h]: The public interface of the Serializer class

Apart from the constructor and destructor functions required for creating and destroying a serializer object, we have a pair of functions that should be used by the server process, and another pair of functions that should be used by the client process.

On the client side, we serialize the request object and we deserialize the response message. Meanwhile on the server side, we deserialize the request message and we serialize the response object.

In addition to serialization and deserialization functions, we have three *callback functions*:

*   A callback for receiving a request object that has been deserialized from the underlying channel
*   A callback for receiving a response object that has been deserialized from the underlying channel
*   A callback for receiving the error when a serialization or a deserialization has failed

These callbacks are used by client and server processes to receive incoming requests and responses and also the errors that are found during serialization and deserialization of a message.

Now, let's have a deeper look at serialization/deserialization functions for the server side.

### Server-side serializer/deserializer functions

We have two functions for the server process to serialize a response object and deserialize a request message. We begin with the response serialization function.

The following code box contains the code for the response serialization function `calc_proto_ser_server_serialize`:

```cpp
struct buffer_t calc_proto_ser_server_serialize(
    struct calc_proto_ser_t* ser,
    const struct calc_proto_resp_t* resp) {
  struct buffer_t buff;
  char resp_result_str[64];
  _serialize_double(resp_result_str, resp->result);
  buff.data = (char*)malloc(64 * sizeof(char));
  sprintf(buff.data, "%d%c%d%c%s%c", resp->req_id,
          FIELD_DELIMITER, (int)resp->status, FIELD_DELIMITER,
      resp_result_str, MESSAGE_DELIMITER);
  buff.len = strlen(buff.data);
  return buff;
}
```

Code Box 20-8 [calcser/calc_proto_ser.c]: The server-side response serializer function

As you can see, `resp` is a pointer to a response object that needs to be serialized. This function returns a `buffer_t object`, which is declared as follows as part of the `calc_proto_ser.h` header file:

```cpp
struct buffer_t {
  char* data;
  int len;
};
```

Code Box 20-9 [calcser/calc_proto_ser.h]: The definition of buffer_t

The serializer code is simple and it consists mainly of a `sprintf` statement that creates the response string message. Now, let's look at the request deserializer function. Deserialization is usually more difficult to implement, and if you go to the code base and follow the function calls, you see how complicated it can be.

*Code Box 20-9* contains the request deserialization function:

```cpp
void calc_proto_ser_server_deserialize(
    struct calc_proto_ser_t* ser,
    struct buffer_t buff,
    bool_t* req_found) {
  if (req_found) {
    *req_found = FALSE;
  }
  _deserialize(ser, buff, _parse_req_and_notify,
          ERROR_INVALID_REQUEST, req_found);
}
```

Code Box 20-9 [calcser/calc_proto_ser.c]: The server-side request deserialization function

The preceding function seems to be simple, but in fact it uses the `_deserialize` and `_parse_req_and_notify` private functions. These functions are defined in the `calc_proto_ser.c` file, which contains the actual implementation of the `Serializer` class.

It would be intense and beyond the scope of this book to bring in and discuss the code we have for the mentioned private functions, but to give you an idea, especially for when you want to read the source code, the deserializer uses a *ring buffer* with a fixed length and tries to find `$` as the message separator.

Whenever it finds `$`, it calls the function pointer, which in this case points to the `_parse_req_and_notify` function (the third argument passed in the `_deserialize` function). The `_parse_req_and_notify` function tries to extract the fields and resurrect the request object. Then, it notifies the registered *observer*, in this case the server object that is waiting for a request through the callback functions, to proceed with the request object.

Now, let's look at the functions used by the client side.

### Client-side serializer/deserializer functions

Just as for the server side, we have two functions on the client side. One for serializing the request object, and the other one meant to deserialize the incoming response.

We begin with the request serializer. You can see the definition in *Code Box 20-10*:

```cpp
struct buffer_t calc_proto_ser_client_serialize(
    struct calc_proto_ser_t* ser,
    const struct calc_proto_req_t* req) {
  struct buffer_t buff;
  char req_op1_str[64];
  char req_op2_str[64];
  _serialize_double(req_op1_str, req->operand1);
  _serialize_double(req_op2_str, req->operand2);
  buff.data = (char*)malloc(64 * sizeof(char));
  sprintf(buff.data, "%d%c%s%c%s%c%s%c", req->id, FIELD_DELIMITER,
          method_to_str(req->method), FIELD_DELIMITER,
          req_op1_str, FIELD_DELIMITER, req_op2_str,
          MESSAGE_DELIMITER);
  buff.len = strlen(buff.data);
  return buff;
}
```

Code Box 20-10 [calcser/calc_proto_ser.c]: The client-side request serialization function

As you can see, it accepts a request object and returns a `buffer` object, totally similar to the response serializer on the server side. It even uses the same technique; a `sprintf` statement for creating the request message.

*Code Box 20-11* contains the response deserializer function:

```cpp
void calc_proto_ser_client_deserialize(
    struct calc_proto_ser_t* ser,
    struct buffer_t buff, bool_t* resp_found) {
  if (resp_found) {
    *resp_found = FALSE;
  }
  _deserialize(ser, buff, _parse_resp_and_notify,
          ERROR_INVALID_RESPONSE, resp_found);
}
```

Code Box 20-11 [calcser/calc_proto_ser.c]: The client-side response deserialization function

As you can see, the same mechanism is employed, and some similar private functions have been used. It is highly recommended to read these sources carefully, in order to get a better understanding of how the various parts of the code have been put together to have the maximum reuse of the existing parts.

We won't go any deeper than this into the `Serializer` class; it's up to you to dig into the code and finds out how it works.

Now that we have the serializer library, we can proceed and write our client and server programs. Having a library that serializes objects and deserializes messages based on an agreed application protocol is a vital step in writing multi-process software. Note that it doesn't matter if the deployment is single-host or contains multiple hosts; the processes should be able to understand each other, and proper application protocols should have been defined.

Before jumping to code regarding socket programming, we have to explain one more thing: the calculator service. It is at the heart of the server process and it does the actual calculation.

## Calculator service

The calculator service is the core logic of our example. Note that this logic should work independently of the underlying IPC mechanism. The upcoming code shows the declaration of the calculator service class.

As you can see, it is designed in such a way that it can be used even in a very simple program, with just a `main` function, such that it doesn't even do any IPC at all:

```cpp
#ifndef CALC_SERVICE_H
#define CALC_SERVICE_H
#include <types.h>
static const int CALC_SVC_OK = 0;
static const int CALC_SVC_ERROR_DIV_BY_ZERO = -1;
struct calc_service_t;
struct calc_service_t* calc_service_new();
void calc_service_delete(struct calc_service_t*);
void calc_service_ctor(struct calc_service_t*);
void calc_service_dtor(struct calc_service_t*);
void calc_service_reset_mem(struct calc_service_t*);
double calc_service_get_mem(struct calc_service_t*);
double calc_service_add(struct calc_service_t*, double, double b,
    bool_t mem);
double calc_service_sub(struct calc_service_t*, double, double b,
    bool_t mem);
double calc_service_mul(struct calc_service_t*, double, double b,
    bool_t mem);
int calc_service_div(struct calc_service_t*, double,
        double, double*);
#endif
```

Code Box 20-12 [calcsvc/calc_service.h]: The public interface of the calculator service class

As you can see, the preceding class even has its own error types. The input arguments are pure C types, and it is in no way dependent on IPC-related or serialization-related classes or types. Since it is isolated as a standalone logic, we compile it into an independent static library named `libcalcsvc.a`.

Every server process must use the calculator service objects in order to do the actual calculations. These objects are usually called the *service objects*. Because of this, the final server program must get linked against this library.

An important note before we go further: if, for a specific client, the calculations don't need a specific context, then having just one service object is enough. In other words, if a service for a client doesn't require us to remember any state from the previous requests of that client, then we can use a *singleton* service object. We call this a *stateless service object*.

Conversely, if handling the current request demands knowing something from the previous requests, then for every client, we need to have a specific service object. This is the case regarding our calculator project. As you know, the calculator has an internal memory that is unique for each client. Therefore, we cannot use the same object for two clients. These objects are known as *stateful service objects*.

To summarize what we said above, for every client, we have to create a new service object. This way, every client has its own calculator with its own dedicated internal memory. Calculator service objects are stateful and they need to load some state (the value of the internal memory).

Now, we are in a good position to move forward and talk about various types of sockets, with examples given in the context of the calculator project.

# Unix domain sockets

From the previous chapter, we know that if we are going to establish a connection between two processes on the same machine, UDS are one of the best options. In this chapter, we expanded our discussion and talked a bit more about push-based IPC techniques, as well as stream and datagram channels. Now it's time to gather our knowledge from previous and current chapters and see UDS in action.

In this section, we have four subsections dedicated to processes being on the listener side or the connector side and operating on a stream or a datagram channel. All of these processes are using UDS. We go through the steps they should take to establish the channel, based on the sequences we discussed in the previous chapter. As the first process, we start with the listener process operating on a stream channel. This would be the *stream server*.

## UDS stream server

If you remember from the previous chapter, we had a number of sequences for listener and connector sides in a transport communication. A server stands in the position of a listener. Therefore, it should follow the listener sequence. More specifically, since we are talking about stream channels in this section, it should follow a stream listener sequence.

As part of that sequence, the server needs to create a socket object first. In our calculator project, the stream server process willing to receive connections over a UDS must follow the same sequence.

The following piece of code is located in the main function of the calculator server program, and as can be seen in *Code Box 20-13*, the process firstly creates a `socket` object:

```cpp
int server_sd = socket(AF_UNIX, SOCK_STREAM, 0);
if (server_sd == -1) {
  fprintf(stderr, "Could not create socket: %s\n", strerror(errno));
  exit(1);
}
```

Code Box 20-13 [server/unix/stream/main.c]: Creating a stream UDS object

As you can see, the `socket` function is used to create a socket object. This function is included from `<sys/socket.h>`, which is a POSIX header. Note that this is just a socket object, and yet it is not determined whether this is going to be a client socket or a server socket. Only the subsequent function calls determine this.

As we explained in the previous chapter, every socket object has three attributes. These attributes are determined by the three arguments passed to the `socket` function. These arguments specify the address family, the type, and the protocol used on that socket object respectively.

According to the stream listener sequence and especially regarding the UDS after creating the socket object, the server program must bind it to a *socket file*. Therefore, the next step is to bind the socket to a socket file. *Code Box 20-14* has been used in the calculator project to bind the socket object to a file located at a predetermined path specified by the `sock_file` character array:

```cpp
struct sockaddr_un addr;
memset(&addr, 0, sizeof(addr));
addr.sun_family = AF_UNIX;
strncpy(addr.sun_path, sock_file, sizeof(addr.sun_path) - 1);
int result = bind(server_sd, (struct sockaddr*)&addr, sizeof(addr));
if (result == -1) {
  close(server_sd);
  fprintf(stderr, "Could not bind the address: %s\n", strerror(errno));
  exit(1);
}
```

Code Box 20-14 [server/unix/stream/main.c]: Binding a stream UDS object to a socket file specified by the sock_file char array

The preceding code has two steps. The first step is to create an instance, named `addr`, of the type `struct sockaddr_un` and then initialize it by pointing it to a socket file. In the second step, the `addr` object is passed to the `bind` function in order to let it know which socket file should be *bound* to the socket object. The `bind` function call succeeds only if there is no other socket object bound to the same socket file. Therefore, with UDS, two socket objects, probably being in different processes, cannot be bound to the same socket file.

**Note**:

In Linux, UDS can be bound to *abstract socket addresses*. They are useful mainly when there is no filesystem mounted to be used for having a socket file. A string starting with a null character, `\0`, can be used to initialize the address structure, `addr` in the preceding code box, and then the provided name is bound to the socket object inside the kernel. The provided name should be unique in the system and no other socket object should be bound to it.

On a further note about the socket file path, the length of the path cannot exceed 104 bytes on most Unix systems. However, in Linux systems, this length is 108 bytes. Note that the string variable keeping the socket file path always include an extra null character at the end as a `char` array in C. Therefore, effectively, 103 and 107 bytes can be used as part of the socket file path depending on the operating system.

If the `bind` function returns `0`, it means that the binding has been successful, and you can proceed with configuring the size of the *backlog*; the next step in the stream listener sequence after binding the endpoint.

The following code shows how the backlog is configured for the stream calculator server listening on a UDS:

```cpp
result = listen(server_sd, 10);
if (result == -1) {
  close(server_sd);
  fprintf(stderr, "Could not set the backlog: %s\n", strerror(errno));
  exit(1);
}
```

Code Box 20-15 [server/unix/stream/main.c]: Configuring the size of the backlog for a bound stream socket

The `listen` function configures the size of the backlog for an already bound socket. As we have explained in the previous chapter, when a busy server process cannot accept any more incoming clients, a certain number of these clients can wait in the backlog until the server program can process them. This is an essential step in preparing a stream socket before accepting the clients.

According to what we have in the stream listener sequence, after having the stream socket bound and having its backlog size configured, we can start accepting new clients. *Code Box 20-16* shows how new clients can be accepted:

```cpp
while (1) {
  int client_sd = accept(server_sd, NULL, NULL);
  if (client_sd == -1) {
    close(server_sd);
    fprintf(stderr, "Could not accept the client: %s\n",
        strerror(errno));
    exit(1);
  }
  ...
}
```

Code Box 20-16 [server/unix/stream/main.c]: Accepting new clients on a stream listener socket

The magic is the `accept` function, which returns a new socket object whenever a new client is received. The returned socket object refers to the underlying stream channel between the server and the accepted client. Note that every client has its own stream channel, and hence its own socket descriptor.

Note that if the stream listener socket is blocking (which it is by default), the `accept` function would block the execution until a new client is received. In other words, if there is no incoming client, the thread calling the `accept` function is blocked behind it.

Now, it's time to see the above steps together in just one place. The following code box shows the stream server from the calculator project, which listens on a UDS:

```cpp
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <stream_server_core.h>
int main(int argc, char** argv) {
  char sock_file[] = "/tmp/calc_svc.sock";
  // ----------- 1\. Create socket object ------------------
  int server_sd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_sd == -1) {
    fprintf(stderr, "Could not create socket: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 2\. Bind the socket file ------------------
  // Delete the previously created socket file if it exists.
  unlink(sock_file);
  // Prepare the address
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, sock_file, sizeof(addr.sun_path) - 1);
  int result = bind(server_sd,
 (struct sockaddr*)&addr, sizeof(addr));
  if (result == -1) {
    close(server_sd);
    fprintf(stderr, "Could not bind the address: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 3\. Prepare backlog ------------------
  result = listen(server_sd, 10);
  if (result == -1) {
    close(server_sd);
    fprintf(stderr, "Could not set the backlog: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 4\. Start accepting clients ---------
  accept_forever(server_sd);
  return 0;
}
```

Code Box 20-17 [server/unix/stream/main.c]: The main function of the stream calculator service listening on a UDS endpoint

It should be easy to find the code blocks that perform the aforementioned steps in initializing a server socket. The only thing that is missing is the client-accepting code. The actual code for accepting new clients is put in a separate function that is called `accept_forever`. Note that this function is blocking and blocks the main thread until the server stops.

In the following code box, you can see the definition of the `accept_forever` function. The function is part of the server common library located in the `srvcore` directory. This function should be there because its definition remains the same for other stream sockets such as TCP sockets. Therefore, we can reuse the existing logic instead of writing it again:

```cpp
void accept_forever(int server_sd) {
  while (1) {
    int client_sd = accept(server_sd, NULL, NULL);
    if (client_sd == -1) {
      close(server_sd);
      fprintf(stderr, "Could not accept the client: %s\n",
              strerror(errno));
      exit(1);
    }
    pthread_t client_handler_thread;
    int* arg = (int *)malloc(sizeof(int));
    *arg = client_sd;
    int result = pthread_create(&client_handler_thread, NULL,
            &client_handler, arg);
    if (result) {
      close(client_sd);
      close(server_sd);
      free(arg);
      fprintf(stderr, "Could not start the client handler thread.\n");
      exit(1);
    }
  }
}
```

Code Box 20-18 [server/srvcore/stream_server_core.c]: The function accepting new clients on a stream socket listening on a UDS endpoint

As you can see in the preceding code box, upon accepting a new client, we spawn a new thread that is in charge of handling the client. This effectively entails reading bytes from the client's channel, passing the read bytes into the deserializer, and producing proper responses if a request has been detected.

Creating a new thread for every client is usually the pattern for every server process that operates on a blocking stream channel, no matter what the type of socket is. Therefore, in such use cases, multithreading and all the surrounding topics become enormously important.

**Note**:

Regarding non-blocking stream channels, a different approach known as *event loop* is usually used.

When you have the socket object of a client, you can use it for reading from the client, as well writing to the client. If we follow the path that we've taken so far in the `srvcore` library, the next step is to look into the companion function of a client's thread; `client_handler`. The function can be found next to the `accept_forever` in the code base. Next, you can see the code box containing the function's definition:

```cpp
void* client_handler(void *arg) {
  struct client_context_t context;
  context.addr = (struct client_addr_t*)
      malloc(sizeof(struct client_addr_t));
  context.addr->sd = *((int*)arg);
  free((int*)arg);
 context.ser = calc_proto_ser_new();
  calc_proto_ser_ctor(context.ser, &context, 256);
  calc_proto_ser_set_req_callback(context.ser, request_callback);
  calc_proto_ser_set_error_callback(context.ser, error_callback);
  context.svc = calc_service_new();
  calc_service_ctor(context.svc);
  context.write_resp = &stream_write_resp;
  int ret;
  char buffer[128];
  while (1) {
    int ret = read(context.addr->sd, buffer, 128);
    if (ret == 0 || ret == -1) {
      break;
    }
    struct buffer_t buf;
    buf.data = buffer; buf.len = ret;
    calc_proto_ser_server_deserialize(context.ser, buf, NULL);
  }
  calc_service_dtor(context.svc);
  calc_service_delete(context.svc);
  calc_proto_ser_dtor(context.ser);
  calc_proto_ser_delete(context.ser);
  free(context.addr);
  return NULL;
}
```

Code Box 20-19 [server/srvcore/stream_server_core.c]: The companion function of the client-handling thread

There are many details regarding the preceding code, but there are a few important ones that I want to mention. As you see, we are using the `read` function to read chunks from the client. If you remember, the `read` function accepts a file descriptor but here we are passing a socket descriptor. This shows, despite the differences between file descriptors and socket descriptors, regarding I/O functions, we can use the same API.

In the preceding code, we read chunks of bytes from the input and we pass them to the deserializer by calling the `calc_proto_ser_server_deserialize` function. It is possible to call this function three or four times before having a request fully deserialized. This is highly dependent on the chunk size that you read from the input and the length of the messages transmitting on the channel.

On a further note, every client has its own serializer object. This is also true for the calculator service object. These objects are created and destroyed as part of the same thread.

And as the last note about the preceding code box, we are using a function to write responses back to the client. The function is `stream_write_response` and it is meant to be used on a stream socket. This function can be found in the same file as the preceding code boxes. Next, you can see the definition of this function:

```cpp
void stream_write_resp(
        struct client_context_t* context,
        struct calc_proto_resp_t* resp) {
  struct buffer_t buf =
      calc_proto_ser_server_serialize(context->ser, resp);
  if (buf.len == 0) {
    close(context->addr->sd);
    fprintf(stderr, "Internal error while serializing response\n");
    exit(1);
  }
  int ret = write(context->addr->sd, buf.data, buf.len);
  free(buf.data);
  if (ret == -1) {
    fprintf(stderr, "Could not write to client: %s\n",
            strerror(errno));
    close(context->addr->sd);
    exit(1);
  } else if (ret < buf.len) {
    fprintf(stderr, "WARN: Less bytes were written!\n");
    exit(1);
  }
}
```

Code Box 20-20 [server/srvcore/stream_server_core.c]: The function used for writing the responses back to the client

As you see in the preceding code, we are using the `write` function to write a message back to the client. As we know, the `write` function can accept file descriptors, but it seems socket descriptors can also be used. So, it clearly shows that the POSIX I/O API works for both file descriptors and socket descriptors.

The above statement is also true about the `close` function. As you can see, we have used it to terminate a connection. It is enough to pass the socket descriptor while we know that it works for file descriptors as well.

Now that we have gone through some of the most important parts of the UDS stream server and we have an idea of how it operates, it is time to move on and discuss the UDS stream client. For sure, there are plenty places in the code that we haven't discussed but you should dedicate time and go through them.

## UDS stream client

Like the server program described in the previous section, the client also needs to create a socket object first. Remember that we need to follow the stream connector sequence now. It uses the same piece of code as server does, with exactly the same arguments, to indicate that it needs a UDS. After that, it needs to connect to the server process by specifying a UDS endpoint, similarly to how the server did. When the stream channel is established, the client process can use the opened socket descriptor to read from and write to the channel.

Next, you can see the `main` function of the stream client connecting to a UDS endpoint:

```cpp
int main(int argc, char** argv) {
  char sock_file[] = "/tmp/calc_svc.sock";
  // ----------- 1\. Create socket object ------------------
  int conn_sd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (conn_sd == -1) {
    fprintf(stderr, "Could not create socket: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 2\. Connect to server ---------------------
  // Prepare the address
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, sock_file, sizeof(addr.sun_path) - 1);
  int result = connect(conn_sd,
 (struct sockaddr*)&addr, sizeof(addr));
  if (result == -1) {
    close(conn_sd);
    fprintf(stderr, "Could no connect: %s\n", strerror(errno));
    exit(1);
  }
 stream_client_loop(conn_sd);
  return 0;
}
```

Code Box 20-21 [client/unix/stream/main.c]: The main function of the stream client connecting to a UDS endpoint

As you can see, the first part of the code is very similar to the server code but afterward, the client calls `connect` instead of `bind`. Note that the address preparation code is exactly the same as that of the server.

When `connect` returns successfully, it has already associated the `conn_sd` socket descriptor to the opened channel. Therefore, from now on, `conn_sd` can be used to communicate with the server. We pass it to the `stream_client_loop` function, which brings up the client's command line and does the rest of the actions performed by the client. It is a blocking function that runs the client until it quits.

Note that the client also uses `read` and `write` functions to transmit messages back and forth from and to the server. *Code Box 20-22* contains the definition of the `stream_client_loop` function, which is part of the client common library that is used by all stream clients, regardless of the socket type and is shared between UDS and TCP sockets. As you see, it uses the `write` function to send a serialized request message to the server:

```cpp
void stream_client_loop(int conn_sd) {
  struct context_t context;
  context.sd = conn_sd;
  context.ser = calc_proto_ser_new();
  calc_proto_ser_ctor(context.ser, &context, 128);
  calc_proto_ser_set_resp_callback(context.ser, on_response);
  calc_proto_ser_set_error_callback(context.ser, on_error);
  pthread_t reader_thread;
 pthread_create(&reader_thread, NULL,
stream_response_reader, &context);
  char buf[128];
  printf("? (type quit to exit) ");
  while (1) {
    scanf("%s", buf);
    int brk = 0, cnt = 0;
    struct calc_proto_req_t req;
    parse_client_input(buf, &req, &brk, &cnt);
    if (brk) {
      break;
    }
    if (cnt) {
      continue;
    }
    struct buffer_t ser_req =
        calc_proto_ser_client_serialize(context.ser, &req);
    int ret = write(context.sd, ser_req.data, ser_req.len);
    if (ret == -1) {
      fprintf(stderr, "Error while writing! %s\n",
              strerror(errno));
      break;
    }
    if (ret < ser_req.len) {
      fprintf(stderr, "Wrote less than anticipated!\n");
      break;
    }
    printf("The req(%d) is sent.\n", req.id);
  }
  shutdown(conn_sd, SHUT_RD);
  calc_proto_ser_dtor(context.ser);
  calc_proto_ser_delete(context.ser);
  pthread_join(reader_thread, NULL);
  printf("Bye.\n");
}
```

Code Box 20-22 [client/clicore/stream_client_core.c]: The function executing a stream client

As you can see in the preceding code, every client process has only one serializer object and it makes sense. This is opposite to the server process, where every client had a separate serializer object.

More than that, the client process spawns a separate thread for reading the responses from the server side. That's because reading from the server process is a blocking task and it should be done in a separate flow of execution.

As part of the main thread, we have the client's command line, which receives inputs from a user through the Terminal. As you see, the main thread joins the reader thread upon exiting and it waits for its completion.

On a further note regarding the preceding code, the client process uses the same I/O API for reading from and writing to the stream channel. Like we said before, the `read` and `write` functions are used and the usage of the `write` function can be seen in *Code Box 20-22*.

In the following section, we talk about datagram channels but still using the UDS for that purpose. We start with the datagram server first.

## UDS datagram server

If you remember from the previous chapter, datagram processes had their own listener and connector sequences regarding transport transmission. Now it's time to demonstrate how a datagram server can be developed based on UDS.

According to the datagram listener sequence, the process needs to create a socket object first. The following code box demonstrates that:

```cpp
int server_sd = socket(AF_UNIX, SOCK_DGRAM, 0);
if (server_sd == -1) {
  fprintf(stderr, "Could not create socket: %s\n",
          strerror(errno));
  exit(1);
}
```

Code Box 20-23 [server/unix/datagram/main.c]: Creating a UDS object meant to operate on a datagram channel

You see that we have used `SOCK_DGRAM` instead of `SOCK_STREAM`. This means that the socket object is going to operate on a datagram channel. The other two arguments remain the same.

As the second step in the datagram listener sequence, we need to bind the socket to a UDS endpoint. As we said before, this is a socket file. This step is exactly the same as for the stream server, and therefore we don't bother to demonstrate it below and you can see it in *Code Box 20-14*.

For a datagram listener process, these steps were the only ones to be performed, and there is no backlog associated to a datagram socket to be configured. More than that, there is no client-accepting phase because we can't have stream connections on some dedicated 1-to-1 channels.

Next, you can see the `main` function of the datagram server listening on a UDS endpoint, as part of the calculator project:

```cpp
int main(int argc, char** argv) {
  char sock_file[] = "/tmp/calc_svc.sock";
  // ----------- 1\. Create socket object ------------------
  int server_sd = socket(AF_UNIX, SOCK_DGRAM, 0);
  if (server_sd == -1) {
    fprintf(stderr, "Could not create socket: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 2\. Bind the socket file ------------------
  // Delete the previously created socket file if it exists.
  unlink(sock_file);
  // Prepare the address
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, sock_file, sizeof(addr.sun_path) - 1);
  int result = bind(server_sd,
          (struct sockaddr*)&addr, sizeof(addr));
  if (result == -1) {
    close(server_sd);
    fprintf(stderr, "Could not bind the address: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 3\. Start serving requests ---------
  serve_forever(server_sd);
  return 0;
}
```

Code Box 20-24 [server/unix/datagram/main.c]: The main function of the datagram server listening on a UDS endpoint

As you know, datagram channels are connection-less, and they don't operate like stream channels. In other words, there cannot be a dedicated 1-to-1 connection between two processes. Therefore, the processes can only transmit datagrams along the channel. A client process can only send some individual and independent datagrams and likewise, the server process can only receive datagrams and send back some other datagrams as responses.

So, the crucial thing about a datagram channel is that the request and response messages should be fit into a single datagram. Otherwise, they cannot be split between two datagrams and the server or client cannot handle the message. Fortunately, our messages in the calculator project are mostly short enough to be fit into a single datagram.

The size of a datagram is highly dependent on the underlying channel. For example, regarding datagram UDS this is quite flexible because it happens through the kernel, but regarding UDP sockets, you are bound to the configuration of the network. Regar[ding the UDS the following link can give you a better idea of how to set the correct size: https://stackov](https://stackoverflow.com/questions/21856517/whats-the-practical-limit-on-the-size-of-single-packet-transmitted-over-domain)erflow.com/questions/21856517/whats-the-practical-limit-on-the-size-of-single-packet-transmitted-over-domain.

Another difference that we can mention regarding datagram and stream sockets is the I/O API that is used to transmit data along them. While the `read` and `write` functions can still be used for datagram sockets just like the stream sockets, we use other functions for reading from and sending to a datagram channel. The `recvfrom` and `sendto` functions are usually used.

That's because in stream sockets the channel is dedicated, and when you write to a channel both ends are determined. Regarding datagram sockets, we have only one channel that is being used by many parties. Therefore, we can lose track of the process owning a specific datagram. These functions can keep track of and send the datagram back to the desired process.

Next, you can find the definition for the `serve_forever` function used in *Code Box 20-24* at the end of the `main` function. This function belongs to the server common library and is specific to datagram servers, regardless of the socket type. You can clearly see how the `recvfrom` function has been used:

```cpp
void serve_forever(int server_sd) {
  char buffer[64];
  while (1) {
    struct sockaddr* sockaddr = sockaddr_new();
    socklen_t socklen = sockaddr_sizeof();
    int read_nr_bytes = recvfrom(server_sd, buffer,
 sizeof(buffer), 0, sockaddr, &socklen);
    if (read_nr_bytes == -1) {
      close(server_sd);
      fprintf(stderr, "Could not read from datagram socket: %s\n",
              strerror(errno));
      exit(1);
    }
    struct client_context_t context;
    context.addr = (struct client_addr_t*)
 malloc(sizeof(struct client_addr_t));
    context.addr->server_sd = server_sd;
    context.addr->sockaddr = sockaddr;
    context.addr->socklen = socklen;
    context.ser = calc_proto_ser_new();
    calc_proto_ser_ctor(context.ser, &context, 256);
    calc_proto_ser_set_req_callback(context.ser, request_callback);
    calc_proto_ser_set_error_callback(context.ser, error_callback);
    context.svc = calc_service_new();
    calc_service_ctor(context.svc);
    context.write_resp = &datagram_write_resp;
    bool_t req_found = FALSE;
    struct buffer_t buf;
    buf.data = buffer;
    buf.len = read_nr_bytes;
    calc_proto_ser_server_deserialize(context.ser, buf, &req_found);
    if (!req_found) {
      struct calc_proto_resp_t resp;
      resp.req_id = -1;
      resp.status = ERROR_INVALID_RESPONSE;
      resp.result = 0.0;
      context.write_resp(&context, &resp);
    }
    calc_service_dtor(context.svc);
    calc_service_delete(context.svc);
    calc_proto_ser_dtor(context.ser);
    calc_proto_ser_delete(context.ser);
    free(context.addr->sockaddr);
    free(context.addr);
  }
}
```

Code Box 20-25 [server/srvcore/datagram_server_core.c]: The function handling the datagrams found in the server common library, and dedicated to the datagram servers

As you see in the preceding code box, the datagram server is a single-threaded program and there is no multithreading around it. More than that, it operates on every datagram individually and independently. It receives a datagram, deserializes its content and creates the request object, handles the request through the service object, serializes the response object and puts it in a new datagram, and sends it back to the process owning the original datagram. It does the same cycle over and over again for every incoming datagram.

Note that every datagram has its own serializer object and its own service object. We could design this in a way that we had only one serializer and one service object for all the datagrams. This might be something interesting for you to think about with regard to how it is possible and why that might not be possible for the calculator project. This is a debatable discussion and you might receive different opinions from various people.

Note that in *Code Box 20-25*, we store the client address of a datagram upon receiving it. Later, we can use this address to write directly back to that client. It is worth having a look at how we write back the datagram to the sender client. Just like the stream server, we are using a function for this purpose. *Code Box 20-26* shows the definition of the `datagram_write_resp` function. The function is in the datagram servers' common library next to the `serve_forever` function:

```cpp
void datagram_write_resp(struct client_context_t* context,
        struct calc_proto_resp_t* resp) {
  struct buffer_t buf =
      calc_proto_ser_server_serialize(context->ser, resp);
  if (buf.len == 0) {
    close(context->addr->server_sd);
    fprintf(stderr, "Internal error while serializing object.\n");
    exit(1);
  }
  int ret = sendto(context->addr->server_sd, buf.data, buf.len,
 0, context->addr->sockaddr, context->addr->socklen);
  free(buf.data);
  if (ret == -1) {
    fprintf(stderr, "Could not write to client: %s\n",
            strerror(errno));
    close(context->addr->server_sd);
    exit(1);
  } else if (ret < buf.len) {
    fprintf(stderr, "WARN: Less bytes were written!\n");
    close(context->addr->server_sd);
    exit(1);
  }
}
```

Code Box 20-26 [server/srvcore/datagram_server_core.c]: The function writing datagrams back to the clients

You can see that we use the sorted client address and we pass it to the `sendto` function together with the serialized response message. The rest is taken care of by the operating system and the datagram is sent back directly to the sender client.

Now that we know enough about the datagram server and how the socket should be used, let's look at the datagram client, which is using the same type of socket.

## UDS datagram client

From a technical point of view, stream clients and datagram clients are very similar. It means that you should see almost the same overall structure but with some differences regarding transmitting datagrams instead of operating on a stream channel.

But there is a big difference between them, and this is quite unique and specific to datagram clients connecting to UDS endpoints.

The difference is that the datagram client is required to bind a socket file, just like the server program, in order to receive the datagrams directed at it. This is not true for datagram clients using network sockets, as you will see shortly. Note that the client should bind a different socket file, and not the server's socket file.

The main reason behind this difference is the fact that the server program needs an address to send the response back to, and if the datagram client doesn't bind a socket file, there is no endpoint bound to the client socket file. But regarding network sockets, a client always has a corresponding socket descriptor that is bound to an IP address and a port, so this problem cannot occur.

If we put aside this difference, we can see how similar the code is. In *Code Box 20-26* you can see the `main` function of the datagram calculator client:

```cpp
int main(int argc, char** argv) {
  char server_sock_file[] = "/tmp/calc_svc.sock";
  char client_sock_file[] = "/tmp/calc_cli.sock";
  // ----------- 1\. Create socket object ------------------
  int conn_sd = socket(AF_UNIX, SOCK_DGRAM, 0);
  if (conn_sd == -1) {
    fprintf(stderr, "Could not create socket: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 2\. Bind the client socket file ------------
  // Delete the previously created socket file if it exists.
  unlink(client_sock_file);
  // Prepare the client address
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, client_sock_file,
          sizeof(addr.sun_path) - 1);
  int result = bind(conn_sd,
          (struct sockaddr*)&addr, sizeof(addr));
  if (result == -1) {
    close(conn_sd);
    fprintf(stderr, "Could not bind the client address: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 3\. Connect to server --------------------
  // Prepare the server address
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, server_sock_file,
          sizeof(addr.sun_path) - 1);
  result = connect(conn_sd,
          (struct sockaddr*)&addr, sizeof(addr));
  if (result == -1) {
    close(conn_sd);
    fprintf(stderr, "Could no connect: %s\n", strerror(errno));
    exit(1);
  }
  datagram_client_loop(conn_sd);
  return 0;
}
```

Code Box 20-26 [server/srvcore/datagram_server_core.c]: The function writing datagrams back to the clients

As we explained earlier, and as can be seen in the code, the client is required to bind a socket file. And of course, we have to call a different function to start the client loop, at the end of the `main` function. The datagram client calls the `datagram_client_loop` function.

If you look at the function `datagram_client_loop`, you still see many similarities between the stream client and the datagram client. Despite the small differences, a big difference is using the `recvfrom` and `sendto` functions instead of the `read` and `write` functions. The same explanation given for these functions as part of the previous section, still holds true for the datagram client.

Now it's time to talk about network sockets. As you will see, the `main` function in the client and server programs is the only code that changes when moving from UDS to network sockets.

# Network sockets

The other socket address family that is widely used is `AF_INET`. It simply refers to any channel established on top of a network connection. Unlike the UDS stream and datagram sockets, which have no protocol name assigned to them, there are two well-known protocols on top of network sockets. TCP sockets establish a stream channel between every two processes, and UDP sockets establish a datagram channel that can be used by a number of processes.

In the following sections, we are going to explain how to develop programs using TCP and UDP sockets and see real some examples as part of the calculator project.

## TCP server

A program using a TCP socket to listen and accept a number of clients, in other words a TCP server, is different from a stream server listening on a UDS endpoint in two ways: firstly, it specifies a different address family, `AF_INET` instead of `AF_UNIX`, when calling the `socket` function. And secondly, it uses a different structure for the socket address required for binding.

Despite these two differences, everything else would be the same for a TCP socket in terms of I/O operation. We should note that a TCP socket is a stream socket, therefore the code written for a stream socket using UDS should work for a TCP socket as well.

If we go back to the calculator project, we expect to see the differences just in the `main` functions where we create the socket object and bind it to an endpoint. Other than that, the rest of the code should remain unchanged. In fact, this is what we actually see. The following code box contains the `main` function of the TCP calculator server:

```cpp
int main(int argc, char** argv) {
  // ----------- 1\. Create socket object ------------------
  int server_sd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_sd == -1) {
    fprintf(stderr, "Could not create socket: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 2\. Bind the socket file ------------------
  // Prepare the address
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(6666);
  ...
  // ----------- 3\. Prepare backlog ------------------
  ...
  // ----------- 4\. Start accepting clients ---------
  accept_forever(server_sd);
  return 0;
}
```

Code Box 20-27 [server/tcp/main.c]: The main function of the TCP calculator client

If you compare the preceding code with the `main` function seen in *Code Box 20-17*, you will notice the differences we explained earlier. Instead of using the `sockaddr_un` structure, we are using the `sockaddr_in` structure for the bound endpoint address. The `listen` function is used the same, and even the same `accept_forever` function has been called to handle the incoming connections.

As a final note, regarding I/O operations on a TCP socket, since a TCP socket is a stream socket, it inherits all the properties from a stream socket; therefore, it can be used just like any other stream socket. In other words, the same `read`, `write`, and `close` functions can be used.

Let's now talk about the TCP client.

## TCP client

Again, everything should be very similar to the stream client operating on a UDS. The differences mentioned in the previous section are still true for a TCP socket on a connector side. The changes are again limited to the `main` function.

Next, you can see the `main` function of the TCP calculator client:

```cpp
int main(int argc, char** argv) {
  // ----------- 1\. Create socket object ------------------
  int conn_sd = socket(AF_INET, SOCK_STREAM, 0);
  if (conn_sd == -1) {
    fprintf(stderr, "Could not create socket: %s\n",
            strerror(errno));
    exit(1);
  }
  // ------------ 2\. Connect to server-- ------------------
  // Find the IP address behind the hostname
  ...
  // Prepare the address
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr = *((struct in_addr*)host_entry->h_addr);
  addr.sin_port = htons(6666);
  ...
  stream_client_loop(conn_sd);
  return 0;
}
```

Code Box 20-27 [server/tcp/main.c]: The main function of the TCP calculator server

The changes are very similar to the ones we saw for the TCP server program. A different address family and a different socket address structure have been used. Apart from that, the rest of the code is the same, and we therefore do not need to discuss the TCP client in detail.

Since TCP sockets are stream sockets, we can use the same common code for handling the new clients. You can see this by calling the `stream_client_loop` function, which is part of the client common library in the calculator project. Now, you should get the idea of why we extracted two common libraries, one for the client programs and one for the server programs, in order to write less code. When we can use the same code for two different scenarios, it is always best to extract it as a library and reuse it in the scenarios.

Let's look at UDP server and client programs; we will see that they are more or less similar to what we saw regarding TCP programs.

## UDP server

UDP sockets are network sockets. Other than that, they are datagram sockets. Therefore, we expect to observe a high degree of similarity between the code we wrote for the TCP server together with the code we wrote for the datagram server operating on a UDS.

In addition, the main difference between a UDP socket and a TCP socket, regardless of being used in a client or server program, is the fact that the socket type is `SOCK_DGRAM` for the UDP socket. The address family remains the same, because both of them are network sockets. The following code box contains the main function of the calculator UDP server:

```cpp
int main(int argc, char** argv) {
  // ----------- 1\. Create socket object ------------------
  int server_sd = socket(AF_INET, SOCK_DGRAM, 0);
  if (server_sd == -1) {
    fprintf(stderr, "Could not create socket: %s\n",
            strerror(errno));
    exit(1);
  }
  // ----------- 2\. Bind the socket file ------------------
  // Prepare the address
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(9999);
  ...
  // ----------- 3\. Start serving requests ---------
  serve_forever(server_sd);
  return 0;
}
```

Code Box 20-28 [server/udp/main.c]: The main function of the UDP calculator server

Note that UDP sockets are datagram sockets. Therefore, all the code written for datagram sockets operating on UDS is still valid for them. For instance, we have to use the `recvfrom` and `sendto` functions to work with UDP sockets. So, as you can see, we have used the same `serve_forever` function to serve incoming datagrams. This function is part of the server common library meant to contain the datagram-related code.

We've said enough regarding the UDP server's code. Let's see what the UDP client's code looks like.

## The UDP client

UDP client code is very similar to the TCP client code, but it uses a different socket type and it calls a different function for handling the incoming messages, which is the same function that the datagram client based on UDS used. You can see the following `main` function:

```cpp
int main(int argc, char** argv) {
  // ----------- 1\. Create socket object ------------------
  int conn_sd = socket(AF_INET, SOCK_DGRAM, 0);
  if (conn_sd == -1) {
    fprintf(stderr, "Could not create socket: %s\n",
            strerror(errno));
    exit(1);
  }
  // ------------ 2\. Connect to server-- ------------------
  ...
  // Prepare the address
  ...
  datagram_client_loop(conn_sd);
  return 0;
}
```

Code Box 20-28 [client/udp/main.c]: The main function of the UDP calculator client

That was the final concept for this chapter. In this chapter, we went through the various well-known socket types and together with that, we showed how the listener and connector sequences for both stream and datagram channels can be implemented in C.

There are many things in the calculator project that we didn't even talk about. Therefore, it is highly recommended to go through the code, find those places, and try to read and understand it. Having a fully working example can help you to examine the concepts in real applications.

# Summary

In this chapter, we went through the following topics:

*   We introduced various types of communications, channels, mediums, and sockets as part of our review of IPC techniques.
*   We explored a calculator project by describing its application protocol and the serialization algorithm that it uses.
*   We demonstrated how UDS can be used to establish a client-server connection, and we showed how they are used in the calculator project.
*   We discussed the stream and datagram channels established using Unix domain sockets, separately.
*   We demonstrated how TCP and UDP sockets can be used to make a client-server IPC channel, and we used them in the calculator example.

The next chapter is about integrating of C with other programming languages. By doing so, we can have a C library loaded and used in another programming language like Java. As part of the next chapter, we cover integration with C++, Java, Python, and Golang.