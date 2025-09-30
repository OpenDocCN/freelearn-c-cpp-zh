# Building a Simple Web Server

This chapter builds on the previous one by looking at the HTTP protocol from the server's perspective. In it, we will build a simple web server. This web server will work using the HTTP protocol, and you will be able to connect to it with any standard web browser. Although it won't be full-featured, it will be suitable for serving a few static files locally. It will be able to handle a few simultaneous connections from multiple clients at once.

The following topics are covered in this chapter:

*   Accepting and buffering multiple connections
*   Parsing an HTTP request line
*   Formatting an HTTP response
*   Serving a file
*   Security considerations

# Technical requirements

The example programs from this chapter can be compiled with any modern C compiler. We recommend MinGW on Windows and GCC on Linux and macOS; see [Appendices B](47da8507-709b-44a6-9399-b18ce6afd8c9.xhtml), *Setting Up Your C Compiler on Windows*, [Appendices C](221eebc0-0bb1-4661-a5aa-eafed9fcba7e.xhtml), *Setting Up Your C Compiler on Linux*, and [Appendices D](632db68e-0911-4238-a2be-bd1aa5296120.xhtml), *Setting Up Your C Compiler on macOS,* for compiler setup.

The code for this book can be found at [https://github.com/codeplea/Hands-On-Network-Programming-with-C](https://github.com/codeplea/Hands-On-Network-Programming-with-C).

From the command line, you can download the code for this chapter with the following command:

```cpp
git clone https://github.com/codeplea/Hands-On-Network-Programming-with-C
cd Hands-On-Network-Programming-with-C/chap07
```

Each example program in this chapter runs on Windows, Linux, and macOS. While compiling on Windows, each example program requires linking to the **Winsock** library. This can be accomplished by passing the `-lws2_32` option to `gcc`.

We provide the exact commands needed to compile each example as they are introduced.

All of the example programs in this chapter require the same header files and C macros that we developed in [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs*. For brevity, we put these statements in a separate header file, `chap07.h`, which we can include in each program. For an explanation of these statements, please refer to [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs*.

The content of `chap07.h` is as follows:

```cpp
/*chap07.h*/

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

#endif

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

# The HTTP server

In this chapter, we are going to implement an HTTP web server that can serve static files from a local directory. HTTP is a text-based client-server protocol that uses the **Transmission Control Protocol** (**TCP**).

When implementing our HTTP server, we need to support multiple, simultaneous connections from many clients at once. Each received **HTTP Request** needs to be parsed, and our server needs to reply with the proper **HTTP Response**. This **HTTP Response** should include the requested file if possible.

Consider the HTTP transaction illustrated in the following diagram:

![](img/5d678947-2cad-44df-a4a4-5e78fd50fb52.png)

In the preceding diagram, the client is requesting `/document.htm` from the server. The server finds `/document.htm` and returns it to the client.

Our HTTP server is somewhat simplified, and we only need to look at the first line of the HTTP request. This first line is called the **request line**. Our server only supports `GET` type requests, so it needs to first check that the request line starts with `GET`. It then parses out the requested resource, `/document.htm` in the preceding example.

A more full-featured HTTP server would look at several other HTTP headers. It would look at the `Host` header to determine which site it is hosting. Our server only supports hosting one site, so this header is not meaningful for us.

A production server would also look at headers such as Accept-Encoding and Accept-Language, which could inform a proper response format. Our server just ignores these, and it instead serves files in only the most straightforward way.

The internet can sometimes be a hostile environment. A production-grade web server needs to include security in layers. It should be absolutely meticulous about file access and resource allocation. In the interest of clear explanation and brevity, the server we develop in this chapter is not security-hardened, and it should not be used on the public internet for this reason.

# The server architecture

An HTTP server is a complicated program. It must handle multiple simultaneous connections, parse a complex text-based protocol, handle malformed requests with the proper errors, and serve files. The example we develop in this chapter is greatly simplified from a production-ready server, but it is still a few hundred lines of code. We benefit from breaking the program down into separate functions and data structures.

At the global level, our program stores a linked list of data structures. This linked list contains one separate data structure for each connected client. This data structure stores information about each client such as their address, their socket, and their data received so far. We implement many helper functions that work on this global linked list. These functions are used to add new clients, drop clients, wait on client data, look up clients by their socket (as sockets are returned by `select()`), serve files to clients, and send error messages to clients.

Our server's main loop can then be simplified. It waits for new connections or new data. When new data is received, it checks whether the data consists of a complete HTTP request. If a complete HTTP request is received, the server attempts to send the requested resource. If the HTTP request is malformed or the resource cannot be found, then the server sends an error message to the connected client instead.

Most of the server complexity lies in handling multiple connections, parsing the HTTP request, and handling error conditions.

The server is also responsible for telling the client the content type of each resource it sends. There are a few ways to accomplish this; let's consider them next.

# Content types

It is the HTTP server's job to tell its client the type of content being sent. This is done by the `Content-Type` header. The value of the `Content-Type` header should be a valid media type (formerly known as the **MIME type**) registered with the **Internet Assigned Numbers Authority** (**IANA**). See the *Further reading* section of this chapter for a link to the IANA list of media types.

There are a few ways to determine the media type of a file. If you're on a Unix-based system, such as Linux or macOS, then your operating system already provides a utility for this.

Try the following command on Linux or macOS (replace `example.txt` with a real filename):

```cpp
file --mime-type example.txt
```

The following screenshot shows its usage:

![](img/43a83200-c274-422a-9851-d014c8b9614b.png)

As you can see in the preceding screenshot, the `file` utility told us the media type of `index.html` is `text/html`. It also said the media type of `smile.png` is `image/png`, and the media type of `test.txt` is `text/plain`.

Our web server just uses the file's extension to determine the media type.

Common file extensions and their media type are listed in the following table:

| **Extension** | **Media Type** |
| --- | --- |
| `.css` | `text/css` |
| `.csv` | `text/csv` |
| `.gif` | `image/gif` |
| `.htm` | `text/html` |
| `.html` | `text/html` |
| `.ico` | `image/x-icon` |
| `.jpeg` | `image/jpeg` |
| `.jpg` | `image/jpeg` |
| `.js` | `application/javascript` |
| `.json` | `application/json` |
| `.png` | `image/png` |
| `.pdf` | `application/pdf` |
| `.svg` | `image/svg+xml` |
| `.txt` | `text/plain` |

If a file's media type is unknown, then our server should use `application/octet-stream` as a default. This indicates that the browser should treat the content as an unknown binary blob.

Let's continue by writing code to get `Content-Type` from a filename.

# Returning Content-Type from a filename

Our server code uses a series of `if` statements to determine the proper media type based only on the requested file's extension. This isn't a perfect solution, but it is a common one, and it works for our purposes.

The code to determine a file's media type is as follows:

```cpp
/*web_server.c except*/

const char *get_content_type(const char* path) {
    const char *last_dot = strrchr(path, '.');
    if (last_dot) {
        if (strcmp(last_dot, ".css") == 0) return "text/css";
        if (strcmp(last_dot, ".csv") == 0) return "text/csv";
        if (strcmp(last_dot, ".gif") == 0) return "image/gif";
        if (strcmp(last_dot, ".htm") == 0) return "text/html";
        if (strcmp(last_dot, ".html") == 0) return "text/html";
        if (strcmp(last_dot, ".ico") == 0) return "image/x-icon";
        if (strcmp(last_dot, ".jpeg") == 0) return "image/jpeg";
        if (strcmp(last_dot, ".jpg") == 0) return "image/jpeg";
        if (strcmp(last_dot, ".js") == 0) return "application/javascript";
        if (strcmp(last_dot, ".json") == 0) return "application/json";
        if (strcmp(last_dot, ".png") == 0) return "image/png";
        if (strcmp(last_dot, ".pdf") == 0) return "application/pdf";
        if (strcmp(last_dot, ".svg") == 0) return "image/svg+xml";
        if (strcmp(last_dot, ".txt") == 0) return "text/plain";
    }

    return "application/octet-stream";
}
```

The `get_content_type()` function works by matching the filename extension to a list of known extensions. This is done by using the `strrchr()` function to find the last dot (`.`) in the filename. If a dot is found, then `strcmp()` is used to check for a match on each extension. When a match is found, the proper media type is returned. Otherwise, the default of `application/octet-stream` is returned instead.

Let's continue building helper functions for our server.

# Creating the server socket

Before entertaining the exciting parts of the HTTP server, such as message parsing, let's get the basics out of the way. Our HTTP server, like all servers, needs to create a listening socket to accept new connections. We define a function, `create_socket()`, for this purpose. This function begins by using `getaddrinfo()` to find the listening address:

```cpp
/*web_server.c except*/

SOCKET create_socket(const char* host, const char *port) {
    printf("Configuring local address...\n");
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    struct addrinfo *bind_address;
    getaddrinfo(host, port, &hints, &bind_address);
```

`create_socket()` then continues with creating a socket using `socket()`, binding that socket to the listening address with `bind()`, and having the socket enter a listening state with `listen()`. The following code calls these functions while detecting error conditions:

```cpp
/*web_server.c except*/

    printf("Creating socket...\n");
    SOCKET socket_listen;
    socket_listen = socket(bind_address->ai_family,
            bind_address->ai_socktype, bind_address->ai_protocol);
    if (!ISVALIDSOCKET(socket_listen)) {
        fprintf(stderr, "socket() failed. (%d)\n", GETSOCKETERRNO());
        exit(1);
    }

    printf("Binding socket to local address...\n");
    if (bind(socket_listen,
                bind_address->ai_addr, bind_address->ai_addrlen)) {
        fprintf(stderr, "bind() failed. (%d)\n", GETSOCKETERRNO());
        exit(1);
    }
    freeaddrinfo(bind_address);

    printf("Listening...\n");
    if (listen(socket_listen, 10) < 0) {
        fprintf(stderr, "listen() failed. (%d)\n", GETSOCKETERRNO());
        exit(1);
    }

    return socket_listen;
}
```

The preceding code should be very familiar to you if you're working through this book in order. If not, please refer to [Chapter 3](eb2e080f-fad2-4e14-b4c1-9a6c124df77c.xhtml), *An In-Depth Overview of TCP Connections*, for information about setting up TCP servers.

# Multiple connections buffering

One important hurdle to overcome, when implementing any server software, is accepting and parsing requests from multiple clients simultaneously.

Consider a client that sends only the beginning of an HTTP request, followed by a delay, and then the remainder of the HTTP request. In this case, we cannot respond to that client until the entire HTTP request is received. However, at the same time, we do not wish to delay servicing other connected clients while waiting. For this reason, we need to buffer up received data for each client separately. Only once we've received an entire HTTP request from a client can we respond to that client.

It is useful to define a C `struct` to store information on each connected client. Our program uses the following:

```cpp
/*web_server.c except*/

#define MAX_REQUEST_SIZE 2047

struct client_info {
    socklen_t address_length;
    struct sockaddr_storage address;
    SOCKET socket;
    char request[MAX_REQUEST_SIZE + 1];
    int received;
    struct client_info *next;
};
```

This `struct` allows us to store information about each connected client. A client's address is stored in the `address` field, the address length in `address_length`, and the socket in the `socket` field. All of the data received from the client so far is stored in the `request` array; `received` indicates the number of bytes stored in that array. The `next` field is a pointer that allows us to store `client_info` structures in a linked list.

To simplify our code, we store the root of our linked list in a global variable, `clients`. The declaration is as follows:

```cpp
/*web_server.c except*/

static struct client_info *clients = 0;
```

Declaring `clients` as a global variable helps to keep our code slightly shorter and clearer. However, if you require the code to be re-entrant (for example, if you want multiple servers running simultaneously), you will want to avoid the global state. This can be done by passing around the linked list root as a separate argument to each function call. This chapter's code repository includes an example of this alternative technique in the `web_server2.c` file.

It is useful to define a number of helper functions that work on the `client_info` data structure and the `clients` linked list. We implement the following helper functions for these purposes:

*   `get_client()` takes a `SOCKET` variable and searches our linked list for the corresponding `client_info` data structure.
*   `drop_client()` closes the connection to a client and removes it from the `clients` linked list.
*   `get_client_address()` returns a client's IP address as a string (character array).
*   `wait_on_clients()` uses the `select()` function to wait until either a client has data available or a new client is attempting to connect.
*   `send_400()` and `send_404()` are used to handle HTTP error conditions.
*   `serve_resource()` attempts to transfer a file to a connected client.

Let's now implement these functions one at a time.

# get_client()

Our `get_client()` function accepts a `SOCKET` and searches through the linked list of connected clients to return the relevant `client_info` for that `SOCKET`. If no matching `client_info` is found in the linked list, then a new `client_info` is allocated and added to the linked list. Therefore, `get_client()` serves two purposes—it can find an existing `client_info`, or it can create a new `client_info`.

`get_client()` takes a `SOCKET` as its input and return a `client_info` structure. The following code is the first part of the `get_client()` function:

```cpp
/*web_server.c except*/

struct client_info *get_client(SOCKET s) {
    struct client_info *ci = clients;

    while(ci) {
        if (ci->socket == s)
            break;
        ci = ci->next;
    }

    if (ci) return ci;
```

In the preceding code, we created the `get_client()` function and implemented our linked list search functionality. First, the linked list root, `clients`, is saved into a temporary variable, `ci`. If `ci->socket` is the socket we are searching for, then the loop breaks and `ci` is returned. If the `client_info` structure for the given socket isn't found, then the code continues on and must create a new `client_info` structure. The following code achieves this:

```cpp
/*web_server.c except*/

    struct client_info *n =
        (struct client_info*) calloc(1, sizeof(struct client_info));

    if (!n) {
        fprintf(stderr, "Out of memory.\n");
        exit(1);
    }

    n->address_length = sizeof(n->address);
    n->next = clients;
    clients = n;
    return n;
}
```

In the preceding code, the `calloc()` function is used to allocate memory for a new `client_info` structure. The `calloc()` function also zeroes-out the data structure, which is useful in this case. The code then checks that the memory allocation succeeded, and it prints an error message if it fails.

The code then sets `n->address_length` to the proper size. This allows us to use `accept()` directly on the `client_info` address later, as `accept()` requires the maximum address length as an input.

The `n->next` field is set to the current global linked list root, and the global linked list root, `clients`, is set to `n`. This accomplishes the task of adding in the new data structure at the beginning of the linked list.

The `get_client()` function ends by returning the newly allocated `client_info` structure, `n`.

# drop_client()

The `drop_client()` function searches through our linked list of clients and removes a given client.

The entire function is given in the following code:

```cpp
/*web_server.c except*/

void drop_client(struct client_info *client) {
    CLOSESOCKET(client->socket);

    struct client_info **p = &clients;

    while(*p) {
        if (*p == client) {
            *p = client->next;
            free(client);
            return;
        }
        p = &(*p)->next;
    }

    fprintf(stderr, "drop_client not found.\n");
    exit(1);
}
```

As you can see in the preceding code, `CLOSESOCKET()` is first used to close and clean up the client's connection.

The function then declares a pointer-to-pointer variable, `p`, and sets it to `clients`. This pointer-to-pointer variable is useful because we can use it to change the value of `clients` directly. Indeed, if the client to be removed is the first element in the linked list, then `clients` needs to be updated, so that `clients` points to the second element in the list.

The code uses a `while` loop to walk through the linked list. Once it finds that `*p == client`, `*p`, is set to `client->next`, which effectively removes the client from the linked list, the allocated memory is then freed and the function returns.

Although `drop_client()` is a simple function, it is handy as it can be called in several circumstances. It must be called when we finish sending a client a resource, and it also must be called when we finish sending a client an error message.

# get_client_address()

It is useful to have a helper function that converts a given client's IP address into text. This function is given in the following code snippet:

```cpp
/*web_server.c except*/

const char *get_client_address(struct client_info *ci) {
    static char address_buffer[100];
    getnameinfo((struct sockaddr*)&ci->address,
            ci->address_length,
            address_buffer, sizeof(address_buffer), 0, 0,
            NI_NUMERICHOST);
    return address_buffer;
}
```

`get_client_address()` is a simple function. It first allocates a `char` array to store the IP address in. This `char` array is declared `static`, which ensures that its memory is available after the function returns. This means that we don't need to worry about having the caller `free()` the memory. The downside to this method is that `get_client_address()` has a global state and is not re-entrant-safe. See `web_server2.c` for an alternative version that is re-entrant-safe.

After a `char` buffer is available, the code simply uses `getnameinfo()` to convert the binary IP address into a text address; `getnameinfo()` was covered in detail in previous chapters, but [Chapter 5](3d80e3b8-07d3-49f4-b60f-b006a17f7213.xhtml), *Hostname Resolution and DNS*, has a particularly detailed explanation.

# wait_on_clients()

Our server is capable of handling many simultaneous connections. This means that our server must have a way to wait for data from multiple clients at once. We define a function, `wait_on_clients()`, which blocks until an existing client sends data, or a new client attempts to connect. This function uses `select()` as described in previous chapters. [Chapter 3](eb2e080f-fad2-4e14-b4c1-9a6c124df77c.xhtml), *An In-Depth Overview of TCP Connections*, has a detailed explanation of `select()`.

The `wait_on_clients()` function is defined as follows:

```cpp
/*web_server.c except*/

fd_set wait_on_clients(SOCKET server) {
    fd_set reads;
    FD_ZERO(&reads);
    FD_SET(server, &reads);
    SOCKET max_socket = server;

    struct client_info *ci = clients;

    while(ci) {
        FD_SET(ci->socket, &reads);
        if (ci->socket > max_socket)
            max_socket = ci->socket;
        ci = ci->next;
    }

    if (select(max_socket+1, &reads, 0, 0, 0) < 0) {
        fprintf(stderr, "select() failed. (%d)\n", GETSOCKETERRNO());
        exit(1);
    }

    return reads;
}
```

In the preceding code, first a new `fd_set` is declared and zeroed-out. The server socket is then added to the `fd_set` first. Then the code loops through the linked list of connected clients and adds the socket for each one in turn. A variable, `max_socket`, is maintained throughout this process to store the maximum socket number as required by `select()`.

After all the sockets are added to `fd_set reads`, the code calls `select()`, and `select()` returns when one or more of the sockets in `reads` is ready.

The `wait_on_clients()` function returns `reads` so that the caller can see which socket is ready.

# send_400()

In the case where a client sends an HTTP request which our server does not understand, it is helpful to send a code `400` error. Because errors of this type can arise in several situations, we wrap this functionality in the `send_400()` function. The entire function follows:

```cpp
/*web_server.c except*/

void send_400(struct client_info *client) {
    const char *c400 = "HTTP/1.1 400 Bad Request\r\n"
        "Connection: close\r\n"
        "Content-Length: 11\r\n\r\nBad Request";
    send(client->socket, c400, strlen(c400), 0);
    drop_client(client);
}
```

The `send_400()` function first declares a text array with the entire HTTP response hard-coded. This text is sent using the `send()` function, and then the client is dropped by calling the `drop_client()` function we defined earlier.

# send_404()

In addition to the `400 Bad Request` error, our server also needs to handle the case where a requested resource is not found. In this case, a `404 Not Found` error should be returned. We define a helper function to return this error as follows:

```cpp
/*web_server.c except*/

void send_404(struct client_info *client) {
    const char *c404 = "HTTP/1.1 404 Not Found\r\n"
        "Connection: close\r\n"
        "Content-Length: 9\r\n\r\nNot Found";
    send(client->socket, c404, strlen(c404), 0);
    drop_client(client);
}
```

The `send_404()` function works exactly like the `send_400()` function defined previously.

# serve_resource()

The `serve_resource()` function sends a connected client a requested resource. Our server expects all hosted files to be in a subdirectory called `public`. Ideally, our server should not allow access to any files outside of this `public` directory. However, as we shall see, enforcing this restriction may be more difficult than it first appears.

Our `serve_resource()` function takes as arguments a connected client and a requested resource path. The function begins as follows:

```cpp
/*web_server.c except*/

void serve_resource(struct client_info *client, const char *path) {

    printf("serve_resource %s %s\n", get_client_address(client), path);
```

The connected client's IP address and the requested path are printed to aid in debugging. In a production server, you would also want to print additional information. Most production servers log the date, time, request method, the client's user-agent string, and the response code as a minimum.

Our function then normalizes the requested path. There are a few things to check for. First, if the path is `/`, then we need to serve a default file. There is a tradition of serving a file called `index` in that case, and, indeed, this is what our code does.

We also check that the path isn't too long. Once we ensure that the path is below a maximum length, we can use fixed-size arrays to store it without worrying about buffer overflows.

Our code also checks that the path doesn't contain two consecutive dots—`..`. In file paths, two dots indicate a reference to a parent directory. However, for security reasons, we want to allow access only into our `public` directory. We do not want to provide access to any parent directory. If we allowed paths with `..`, then a malicious client could send `GET /../web_server.c HTTP/1.1` and gain access to our server source code!

The following code is used to redirect root requests and to prevent long or obviously malicious requests:

```cpp
/*web_server.c except*/

    if (strcmp(path, "/") == 0) path = "/index.html";

    if (strlen(path) > 100) {
        send_400(client);
        return;
    }

    if (strstr(path, "..")) {
        send_404(client);
        return;
    }
```

Our code now needs to convert the path to refer to files in the `public` directory. This is done with the `sprintf()` function. First, a text-array is reserved, `full_path`, and then `sprintf()` is used to store the full path into it. We are able to reserve a fixed allocation for `full_path`, as our earlier code ensured that `path` does not exceed `100` characters in length.

The code to set `full_path` is as follows:

```cpp
/*web_server.c except*/

    char full_path[128];
    sprintf(full_path, "public%s", path);
```

It is important to note that the directory separator differs between Windows and other operating systems. While Unix-based systems use a slash (`/`), Windows instead uses a backslash (`\`) as its standard. Many Windows functions handle the conversion automatically, but the difference is sometimes important. For our simple server, the slash conversion isn't an absolute requirement. However, we include it anyway as a good practice.

The following code converts slashes to backslashes on Windows:

```cpp
/*web_server.c except*/

#if defined(_WIN32)
    char *p = full_path;
    while (*p) {
        if (*p == '/') *p = '\\';
        ++p;
    }
#endif
```

The preceding code works by stepping through the `full_path` text array and detecting slash characters. When a slash is found, it is simply overwritten with a backslash. Note that the C code `'\\'` is equivalent to only one backslash. This is because the backslash has special meaning in C, and therefore the first backslash is used to escape the second backslash.

At this point, our server can check whether the requested resource actually exists. This is done by using the `fopen()` function. If `fopen()` fails, for any reason, then our server assumes that the file does not exist. The following code sends a `404` error if the requested resource isn't available:

```cpp
/*web_server.c except*/

    FILE *fp = fopen(full_path, "rb");

    if (!fp) {
        send_404(client);
        return;
    }
```

If `fopen()` succeeds, then we can use `fseek()` and `ftell()` to determine the requested file's size. This is important, as we need to use the file's size in the `Content-Length` header. The following code finds the file size and stores it in the `cl` variable:

```cpp
/*web_server.c except*/

    fseek(fp, 0L, SEEK_END);
    size_t cl = ftell(fp);
    rewind(fp);
```

Once the file size is known, we also want to get the file's type. This is used in the `Content-Type` header. We already defined a function, `get_content_type()`, which makes this task easy. The content type is store in the variable `ct` by the following code:

```cpp
/*web_server.c except*/

    const char *ct = get_content_type(full_path);
```

Once the file has been located and we have its length and type, the server can begin sending the HTTP response. We first reserve a temporary buffer to store header fields in:

```cpp
/*web_server.c except*/

#define BSIZE 1024
    char buffer[BSIZE];
```

Once the buffer is reserved, the server prints relevant headers into it and then sends those headers to the client. This is done using `sprintf()` and then `send()` in turn. The following code sends the HTTP response header:

```cpp
/*web_server.c except*/

    sprintf(buffer, "HTTP/1.1 200 OK\r\n");
    send(client->socket, buffer, strlen(buffer), 0);

    sprintf(buffer, "Connection: close\r\n");
    send(client->socket, buffer, strlen(buffer), 0);

    sprintf(buffer, "Content-Length: %u\r\n", cl);
    send(client->socket, buffer, strlen(buffer), 0);

    sprintf(buffer, "Content-Type: %s\r\n", ct);
    send(client->socket, buffer, strlen(buffer), 0);

    sprintf(buffer, "\r\n");
    send(client->socket, buffer, strlen(buffer), 0);
```

Note that the last `send()` statement sends `\r\n`. This has the effect of transmitting a blank line. This blank line is used by the client to delineate the HTTP header from the beginning of the HTTP body.

The server can now send the actual file content. This is done by calling `fread()` repeatedly until the entire file is sent:

```cpp
/*web_server.c except*/

    int r = fread(buffer, 1, BSIZE, fp);
    while (r) {
        send(client->socket, buffer, r, 0);
        r = fread(buffer, 1, BSIZE, fp);
    }
```

In the preceding code, `fread()` is used to read enough data to fill `buffer`. This buffer is then transmitted to the client using `send()`. These steps are looped until `fread()` returns `0`; this indicates that the entire file has been read.

Note that `send()` may block on large files. In a truly robust, production-ready server, you would need to handle this case. It could be done by using `select()` to determine when each socket is ready to read. Another common method is to use `fork()` or similar APIs to create separate threads/processes for each connected client. For simplicity, our server accepts the limitation that `send()` blocks on large files. Please refer to [Chapter 13](11c5bb82-e55f-4977-bf7f-5dbe791fde92.xhtml), *Socket Programming Tips and Pitfalls*, for more information about the blocking behavior of `send()`.

The function can finish by closing the file handle and using `drop_client()` to disconnect the client:

```cpp
/*web_server.c except*/

    fclose(fp);
    drop_client(client);
}
```

That concludes the `serve_resource()` function.

Keep in mind that while `serve_resource()` attempts to limit access to only the `public` directory, it is not adequate in doing so, and `serve_resource()` should not be used in production code without carefully considering additional access loopholes. We discuss more security concerns later in this chapter.

With these helper functions out of the way, implementing our main server loop is a much easier task. We begin that next.

# The main loop

With our many helper functions out of the way, we can now finish `web_server.c`. Remember to first `#include chap07.h` and also add in all of the types and functions we've defined so far—`struct client_info`, `get_content_type()`, `create_socket()`, `get_client()`, `drop_client()`, `get_client_address()`, `wait_on_clients()`, `send_400()`, `send_404()`, and `serve_resource()`.

We can then begin the `main()` function. It starts by initializing Winsock on Windows:

```cpp
/*web_server.c except*/

int main() {

#if defined(_WIN32)
    WSADATA d;
    if (WSAStartup(MAKEWORD(2, 2), &d)) {
        fprintf(stderr, "Failed to initialize.\n");
        return 1;
    }
#endif
```

We then use our earlier function, `create_socket()`, to create the listening socket. Our server listens on port `8080`, but feel free to change it. On Unix-based systems, listening on low port numbers is reserved for privileged accounts. For security reasons, our web server should be running with unprivileged accounts only. This is why we use `8080` as our port number instead of the HTTP standard port, `80`.

The code to create the server socket is as follows:

```cpp
/*web_server.c except*/

    SOCKET server = create_socket(0, "8080");
```

If you want to accept connections from only the local system, and not outside systems, use the following code instead:

```cpp
/*web_server.c except*/

    SOCKET server = create_socket("127.0.0.1", "8080");
```

We then begin an endless loop that waits on clients. We call `wait_on_clients()` to wait until a new client connects or an old client sends new data:

```cpp
/*web_server.c except*/

    while(1) {

        fd_set reads;
        reads = wait_on_clients(server);
```

The `server` then detects whether a new `client` has connected. This case is indicated by `server` being set in `fd_set reads`. We use the `FD_ISSET()` macro to detect this condition:

```cpp
/*web_server.c except*/

        if (FD_ISSET(server, &reads)) {
            struct client_info *client = get_client(-1);

            client->socket = accept(server,
                    (struct sockaddr*) &(client->address),
                    &(client->address_length));

            if (!ISVALIDSOCKET(client->socket)) {
                fprintf(stderr, "accept() failed. (%d)\n",
                        GETSOCKETERRNO());
                return 1;
            }

            printf("New connection from %s.\n",
                    get_client_address(client));
        }
```

Once a new client connection has been detected, `get_client()` is called with the argument `-1`; `-1` is not a valid socket specifier, so `get_client()` creates a new `struct client_info`. This `struct client_info` is assigned to the `client` variable.

The `accept()` socket function is used to accept the new connection and place the connected clients address information into the respective `client` fields. The new socket returned by `accept()` is stored in `client->socket`.

The client's address is printed using a call to `get_client_address()`. This is helpful for debugging.

Our server must then handle the case where an already connected client is sending data. This is a bit more complicated. We first walk through the linked list of clients and use `FD_ISSET()` on each client to determine which clients have data available. Recall that the linked list root is stored in the `clients` global variable.

We begin our linked list walk with the following:

```cpp
/*web_server.c except*/

        struct client_info *client = clients;
        while(client) {
            struct client_info *next = client->next;

            if (FD_ISSET(client->socket, &reads)) {
```

We then check that we have memory available to store more received data for `client`. If the client's buffer is already completely full, then we send a `400` error. The following code checks for this condition:

```cpp
/*web_server.c except*/

                if (MAX_REQUEST_SIZE == client->received) {
                    send_400(client);
                    continue;
                }
```

Knowing that we have at least some memory left to store received data, we can use `recv()` to store the client's data. The following code uses `recv()` to write new data into the client's buffer while being careful to not overflow that buffer:

```cpp
/*web_server.c except*/

                int r = recv(client->socket,
                        client->request + client->received,
                        MAX_REQUEST_SIZE - client->received, 0);
```

A client that disconnects unexpectedly causes `recv()` to return a non-positive number. In this case, we need to use `drop_client()` to clean up our memory allocated for that client:

```cpp
/*web_server.c except*/

                if (r < 1) {
                    printf("Unexpected disconnect from %s.\n",
                            get_client_address(client));
                    drop_client(client);
```

If the received data was written successfully, our server adds a null terminating character to the end of that client's data buffer. This allows us to use `strstr()` to search the buffer, as the null terminator tells `strstr()` when to stop.

Recall that the HTTP header and body is delineated by a blank line. Therefore, if `strstr()` finds a blank line (`\r\n\r\n`), we know that the HTTP header has been received and we can begin to parse it. The following code detects whether the HTTP header has been received:

```cpp
/*web_server.c except*/

                } else {
                    client->received += r;
                    client->request[client->received] = 0;

                    char *q = strstr(client->request, "\r\n\r\n");
                    if (q) {
```

Our server only handles `GET` requests. We also enforce that any valid path should start with a slash character; `strncmp()` is used to detect these two conditions in the following code:

```cpp
/*web_server.c except*/

                        if (strncmp("GET /", client->request, 5)) {
                            send_400(client);
                        } else {
                            char *path = client->request + 4;
                            char *end_path = strstr(path, " ");
                            if (!end_path) {
                                send_400(client);
                            } else {
                                *end_path = 0;
                                serve_resource(client, path);
                            }
                        }
                    } //if (q)
```

In the preceding code, a proper `GET` request causes the execution of the `else` branch. Here, we set the `path` variable to the beginning of the request path, which is starting at the fifth character of the HTTP request (because C arrays start at zero, the fifth character is located at `client->request + 4`).

The end of the requested path is indicated by finding the next space character. If found, we just call our `serve_resource()` function to fulfil the client's request.

Our server is basically functional at this point. We only need to finish our loops and close out the `main()` function. The following code accomplishes this:

```cpp
/*web_server.c except*/

                }
            }

            client = next;
        }

    } //while(1)

    printf("\nClosing socket...\n");
    CLOSESOCKET(server);

#if defined(_WIN32)
    WSACleanup();
#endif

    printf("Finished.\n");
    return 0;
}
```

Note that our server doesn't actually have a way to break from its infinite loop. It simply listens to connections forever. As an exercise, you may want to add in functionality that allows the server to shut down cleanly. This was omitted only to keep the code simpler. It may also be useful to drop all connected clients with this line of code—`while(clients) drop_client(clients);`

That concludes the code for `web_server.c`. I recommend you download `web_server.c` from this book's code repository and try it out.

You can compile and run `web_server.c` on Linux and macOS with the following commands:

```cpp
gcc web_server.c -o web_server
./web_server
```

On Windows, the command to compile and run using MinGW is as follows:

```cpp
gcc web_server.c -o web_server.exe -lws2_32
web_server.exe
```

The following screenshot shows the server being compiled and run on macOS:

![](img/a3be806f-1a15-46dd-bc3d-e7e99eacab87.png)

If you connect to the server using a standard web browser, you should see something such as the following screenshot:

![](img/177aacbc-25f5-4d57-a3af-7ffafa9f3ccd.png)

You can also drop different files into the `public` folder and play around with creating more complicated websites.

An alternative source file, `web_server2.c`, is also provided in this chapter's code repository. It behaves exactly like the code we developed, but it avoids having global state (at the expense of a little added verbosity). This may make `web_server2.c` more suitable for integration into more significant projects and continued development.

Although the web server we developed certainly works, it does have a number of shortcomings. Please don't deploy this server (or any other network code) in the wild without very carefully considering these shortcomings, some of which we address next.

# Security and robustness

One of the most important rules, when developing networked code, is that your program should never trust the connected peer. Your code should never assume that the connected peer sends data in a particular format. This is especially vital for server code that may communicate with multiple clients at once.

If your code doesn't carefully check for errors and unexpected conditions, then it will be vulnerable to exploits.

Consider the following code which receives data into a buffer until a **space** character is found:

```cpp
char buffer[1028] = {0};
char *p = buffer;

while (!strstr(p, " "))
    p += recv(client, p, 1028, 0);
```

The preceding code works simply. It reserves 1,028 bytes of buffer space and then uses `recv()` to write received data into that space. The `p` pointer is updated on each read to indicate where the next data should be written. The code then loops until the `strstr()` function detects a space character.

That code could be useful to read data from a client until an HTTP verb is detected. For example, it could receive data until `GET` is received, at which point the server can begin to process a `GET` request.

One problem with the preceding code is that `recv()` could write past the end of the allocated space for `buffer`. This is because `1028` is passed to `recv()`, even if some data has already been written. If a network client can cause your code to write past the end of a buffer, then that client may be able to completely compromise your server. This is because both data and executable code are stored in your server's memory. A malicious client may be able to write executable code past the `buffer` array and cause your program to execute it. Even if the malicious code isn't executed, the client could still overwrite other important data in your server's memory.

The preceding code can be fixed by passing to `recv()` only the amount of buffer space remaining:

```cpp
char buffer[1028] = {0};
char *p = buffer;

while (!strstr(p, " "))
    p += recv(client, p, 1028 - (p - buffer), 0);
```

In this case, `recv()` is not be able to write more than 1,028 bytes total into `buffer`. You may think that the memory errors are resolved, but you would still be wrong. Consider a client that sends 1,028 bytes, but no space characters. Your code then calls `strstr()` looking for a space character. Considering that `buffer` is completely full now, `strstr()` cannot find a space character or a null terminating character! In that case, `strstr()` continues to read past the end of `buffer` into unallocated memory.

So, you fix this issue by only allowing `recv()` to write 1,027 bytes total. This reserves one byte to remain as the null terminating character:

```cpp
char buffer[1028] = {0};
char *p = buffer;

while (!strstr(p, " "))
    p += recv(client, p, 1027 - (p - buffer), 0);
```

Now your code won't write or read past the array bounds for `buffer`, but the code is still very broken. Consider a client that sends 1,027 characters. Or consider a client that sends a single null character. In either case, the preceding code continues to loop forever, thus locking up your server and preventing other clients from being served.

Hopefully, the previous examples illustrate the care needed to implement a server in C. Indeed, it's easy to create bugs in any programming language, but in C special care needs to be taken to avoid memory errors.

Another issue with server software is that the server wants to allow access to some files on the system, but not others. A malicious client could send an HTTP request that tries to download arbitrary files from your server system. For example, if an HTTP request such as `GET /../secret_file.c HTTP/1.1` was sent to a naive HTTP server, that server may send the `secret_file.c` to the connected client, even though it exists outside of the `public` directory!

Our code in `web_server.c` detects the most obvious attempts at this by searching for requests containing `..` and denying those requests.

A robust server should use operating systems features to detect that requested files exist as actual files in the permitted directory. Unfortunately, there is no cross-platform way to do this, and the platform-dependent options are somewhat complicated.

Please understand that these are not purely theoretical concerns, but actual exploitable bugs. For example, if you run our `web_server.c` program on Windows and a client sends the request `GET /this_will_be_funny/PRN HTTP/1.1`, what do you suppose happens?

The `this_will_be_funny` directory doesn't exist, and the `PRN` file  certainly doesn't exist in that non-existent directory. These facts may lead you to think that the server simply returns a `404 Not Found` error, as expected. However, that's not what happens. Under Windows, `PRN` is a special filename. When your server calls `fopen()` on this special name, Windows doesn't look for a file, but rather it connects to a *printer* interface! Other special names include `COM1` (connects to serial port 1) and `LPT1` (connects to parallel port 1), although there are others. Even if these filenames have an extension, such as `PRN.txt`, Windows still redirects instead of looking for a file.

One generally applicable piece of security advice is this—run your networked programs under non-privileged accounts that have access to only the minimum resources needed to function. In other words, if you are going to run a networked server, create a new account to run it under. Give that account read access to only the files that server needs to serve. This is not a substitute for writing secure code, but rather running as a non-privilege user creates one final barrier. It is advice you should apply even when running hardened, industry-tested server software.

Hopefully, the previous examples illustrate that programming is complicated, and safe network programming in C can be difficult. It is best approached with care. Oftentimes, it is not possible to know that you have all the loopholes covered. Operating systems don't always have adequate documentation. Operating system APIs often behave in non-obvious and non-intuitive ways. Be careful.

# Open source servers

The code developed in this chapter is suitable for use in trusted applications on trusted networks. For example, if you are developing a video game, it can be very useful to make it serve a web page that displays debugging information. This doesn't have to be a security concern, as it can limit connections to the local machine.

If you must deploy a web server on the internet, I suggest you consider using a free and open source implementation that's already available. The web servers Nginx and Apache, for example, are highly performant, cross-platform, secure, written in C, and completely free. They are also well-documented and easy to find support for.

If you want to expose your program to the internet, you can communicate to a web server using either CGI or FastCGI. With CGI, the web server handles the HTTP request. When a request comes in, it runs your program and returns your program's output in the HTTP response body.

Alternatively, many web servers (such as Nginx or Apache) work as a reverse proxy. This essentially puts the web server between your code and the internet. The web server accepts and forwards HTTP messages to your HTTP server. This can have the effect of slightly shielding your code from attackers.

# Summary

In this chapter, we worked through implementing an HTTP server in C from scratch. That's no small feat! Although the text-based nature of HTTP makes parsing HTTP requests simple, we needed to spend a lot of effort to ensure that multiple clients could be served simultaneously. We accomplished this by buffering received data for each client separately. Each client's state information was organized into a linked list.

Another difficulty was ensuring the safe handling of received data and detecting errors. We learned that a programmer must be very careful when handling network data to avoid creating security risks. We also saw that even very subtle issues, such as Windows's special filenames, can potentially create dangerous security holes for networked server applications.

In the next chapter, [Chapter 8](47e209f2-0231-418c-baef-82db74df8c29.xhtml), *Making Your Program Send Email*, we move on from HTTP and consider the primary protocol associated with email—**Simple Mail Transfer Protocol** (**SMTP**).

# Questions

Try these questions to test your knowledge from this chapter:

1.  How does an HTTP client indicate that it has finished sending the HTTP request?
2.  How does an HTTP client know what type of content the HTTP server is sending?
3.  How can an HTTP server identify a file's media type?
4.  How can you tell whether a file exists on the filesystem and is readable by your program? Is `fopen(filename, "r") != 0` a good test?

The answers to these questions can be found in [Appendix A](bd8b8f52-52cb-4d34-b01b-e907564bfece.xhtml), *Answers to Questions*.

# Further reading

For more information about HTTP and HTML, please refer to the following:

*   **RFC 7230**: *Hypertext Transfer Protocol (HTTP/1.1): Message Syntax and Routing* ([https://tools.ietf.org/html/rfc7230](https://tools.ietf.org/html/rfc7230))
*   **RFC 7231**: *Hypertext Transfer Protocol (HTTP/1.1): Semantics and Content* ([https://tools.ietf.org/html/rfc7231](https://tools.ietf.org/html/rfc7231))
*   *Media Types* ([https://www.iana.org/assignments/media-types/media-types.xhtml](https://www.iana.org/assignments/media-types/media-types.xhtml))