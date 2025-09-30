# Building a Simple Web Client

**Hypertext Transfer Protocol** (**HTTP**) is the application protocol that powers the World Wide Web (WWW). Whenever you fire up your web browser to do an internet search, browse Wikipedia, or make a post on social media, you are using HTTP. Many mobile apps also use HTTP behind the scenes. It's safe to say that HTTP is one of the most widely used protocols on the internet.

In this chapter, we will look at the HTTP message format. We will then implement a C program, which can request and receive web pages.

The following topics are covered in this chapter:

*   The HTTP message format
*   HTTP request types
*   Common HTTP headers
*   HTTP response code
*   HTTP message parsing
*   Implementing an HTTP client
*   Encoding form data (`POST`)
*   HTTP file uploads

# Technical requirements

The example programs from this chapter can be compiled with any modern C compiler. We recommend MinGW on Windows and GCC on Linux and macOS. See appendices B, C, and D for compiler setup.

The code for this book can be found at [https://github.com/codeplea/Hands-On-Network-Programming-with-C](https://github.com/codeplea/Hands-On-Network-Programming-with-C).

From the command line, you can download the code for this chapter with the following command:

```cpp
git clone https://github.com/codeplea/Hands-On-Network-Programming-with-C
cd Hands-On-Network-Programming-with-C/chap06
```

Each example program in this chapter runs on Windows, Linux, and macOS. When compiling on Windows, each example program requires linking with the Winsock library. This is accomplished by passing the `-lws2_32` option to `gcc`.

We provide the exact commands needed to compile each example as they are introduced.

All of the example programs in this chapter require the same header files and C macros that we developed in [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs*. For brevity, we put these statements in a separate header file, `chap06.h`, which we can include in each program. For an explanation of these statements, please refer to [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs*.

The content of `chap06.h` is as follows:

```cpp
//chap06.h//

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
#include <clock.h>
```

# The HTTP protocol

HTTP is a text-based client-server protocol that runs over TCP. Plain HTTP runs over TCP port `80`.

It should be noted that plain HTTP is mostly deprecated for security reasons. Today, sites should use HTTPS, the secure version of HTTP. HTTPS secures HTTP by merely running the HTTP protocol through a **Transport Layer Security** (**TLS**) layer. Therefore, everything we cover in this chapter regarding HTTP also applies to HTTPS. See [Chapter 9](47ba170d-42d9-4e38-b5d8-89503e005708.xhtml), *Loading Secure Web Pages with HTTPS and OpenSSL*, for more information about HTTPS.

HTTP works by first having the web client send an HTTP request to the web server. Then, the web server responds with an HTTP response. Generally, the HTTP request indicates which resource the client is interested in, and the HTTP response delivers the requested resource.

Visually, the transaction is illustrated in the following graphic:

![](img/25330adb-dd3b-4d9e-8cc8-f3c2f5320216.png)

The preceding graphic illustrates a **GET** request. A **GET** request is used when the **Web Client** simply wants the **Web Server** to send it a document, image, file, web page, and so on. **GET** requests are the most common. They are what your browser sends to a **Web** **Server** while loading up a web page or downloading a file.

There are a few other request types that are also worth mentioning.

# HTTP request types

Although `GET` requests are the most common, there are, perhaps, three request types that are commonly used. The three common HTTP request types are as follows:

*   `GET` is used when the client wants to download a resource.
*   `HEAD` is just like `GET`, except that the client only wants information about the resource instead of the resource itself. For example, if the client only wants to know the size of a hosted file, it could send a `HEAD` request.
*   `POST` is used when the client needs to send information to the server. Your web browser typically uses a `POST` request when you submit an online form, for example. A `POST` request will typically cause a web server to change its state somehow. A web server could send an email, update a database, or change a file in response to a `POST` request.

In addition to `GET`, `HEAD`, and `POST`, there are a few more HTTP request types that are rarely used. They are as follows:

*   `PUT` is used to send a document to the web server. `PUT` is not commonly used. `POST` is almost universally used to change the web server state.
*   `DELETE` is used to request that the web server should delete a document or resource. Again, in practice, `DELETE` is rarely used. `POST` is commonly used to communicate web server updates of all types.
*   `TRACE` is used to request diagnostic information from web proxies. Most web requests don't go through a proxy, and many web proxies don't fully support `TRACE`. Therefore, it's rare to need to use `TRACE`.
*   `CONNECT` is sometimes used to initiate an HTTP connection through a proxy server.
*   `OPTIONS` is used to ask which HTTP request types are supported by the server for a given resource. A typical web server that implements `OPTIONS` may respond with something similar to `Allow: OPTIONS, GET, HEAD, POST`. Many common web servers don't support `OPTIONS`.

If you send a request that the web server doesn't support, then the server should respond with a `400 Bad Request` code.

Now that we've seen the types of HTTP requests, let's look at the request format in more detail.

# HTTP request format

If you open your web browser and navigate to `http://www.example.com/page1.htm`, your browser will need to send an HTTP request to the web server at `www.example.com`. That HTTP request may look like this:

```cpp
GET /page1.htm HTTP/1.1
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
Accept-Language: en-US
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8
Accept-Encoding: gzip, deflate
Host: example.com
Connection: Keep-Alive
```

As you can see, the browser sends a `GET` request by default. This `GET` request is asking the server for the document `/page1.htm`. A `GET` request consists of HTTP headers only. There is no HTTP body because the client isn't sending data to the server. The client is only requesting data from the server. In contrast, a `POST` request would contain an HTTP body.

The first line of an HTTP request is called the **request line**. The request line consists of three parts – the request type, the document path, and the protocol version. Each part is separated by a space. In the preceding example, the request line is `GET /page1.htm HTTP/1.1`. We can see that the request type is `GET`, the document path is `/page1.htm`, and the protocol version is `HTTP/1.1`.

When dealing with text-based network protocols, it is always important to be explicit about line endings. This is because different operating systems have standardized on different line-ending conventions. Each line of an HTTP message ends with a carriage return, followed by a newline character. In C, this looks like `\r\n`. In practice, some web servers may tolerate other line endings. You should ensure that your clients always send a proper `\r\n` line ending for maximum compatibility.

After the request line, there are various HTTP header fields. Each header field consists of its name followed by a colon, and then its value. Consider the `User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36` line. This `User-Agent` line is telling the web server what software is contacting it. Some web servers will serve different documents to different user agents. For example, it is common for some websites to serve full documents to search engine spiders while serving paywalls to actual visitors. The server generally uses the user-agent HTTP header field to determine which is which. At the same time, there is a long history of web clients lying in the user-agent field. I suggest you take the high road in your applications and clearly identify your application with a unique user-agent value.

The only header field that is actually required is `Host`. The `Host` field tells the web server which web host the client is requesting the resource from. This is important because one web server may be hosting many different websites. The request line tells the web server that the `/page1.htm` document is wanted, but it doesn't specify which server that page is on. The `Host` field fills this role.

The `Connection: Keep-Alive` line tells the web server that the HTTP client would like to issue additional requests after the current request finishes. If the client had sent `Connection: Close` instead, that would indicate that the client intended to close the TCP connection after the HTTP response was received.

The web client must send a blank line after the HTTP request header. This blank line is how the web server knows that the HTTP request is finished. Without this blank line, the web server wouldn't know whether any additional header fields were still going to being sent. In C, the blank line looks like this: `\r\n\r\n`.

Let's now consider what the web server would send in reply to an HTTP request.

# HTTP response format

Like the HTTP request, the HTTP response also consists of a header part and a body part. Also similar to the HTTP request, the body part is optional. Most HTTP responses do have a body part, though.

The server at `www.example.com` could respond to our HTTP request with the following reply:

```cpp
HTTP/1.1 200 OK
Cache-Control: max-age=604800
Content-Type: text/html; charset=UTF-8
Date: Fri, 14 Dec 2018 16:46:09 GMT
Etag: "1541025663+gzip"
Expires: Fri, 21 Dec 2018 16:46:09 GMT
Last-Modified: Fri, 09 Aug 2013 23:54:35 GMT
Server: ECS (ord/5730)
Vary: Accept-Encoding
X-Cache: HIT
Content-Length: 1270

<!doctype html>
<html>
<head>
    <title>Example Domain</title>
...
```

The first line of an HTTP response is the **status line**. The status line consists of the protocol version, the response code, and the response code description. In the preceding example, we can see that the protocol version is `HTTP/1.1`, the response code is `200`, and the response code description is `OK`. `200 OK` is the typical response code to an HTTP `GET` request when everything goes ok. If the server couldn't find the resource the client has requested, it might respond with a `404 Page Not Found` response code instead.

Many of the HTTP response headers are used to assist with caching. The `Date`, `Etag`, `Expires`, and `Last-Modified` fields can all be used by the client to cache documents.

The `Content-Type` field tells the client what type of resource it is sending. In the preceding example, it is an HTML web page, which is specified with `text/html`. HTTP can be used to send all types of resources, such as images, software, and videos. Each resource type has a specific `Content-Type`, which tells the client how to interpret the resource.

The `Content-Length` field specifies the size of the HTTP response body in bytes. In this case, we see that the requested resource is `1270` bytes long. There are a few ways to determine the body length, but the `Content-Length` field is the simplest. We will look at other ways in the *Response Body Length* section later in this chapter.

The HTTP response header section is delineated from the HTTP response body by a blank line. After this blank line, the HTTP body follows. Note that the HTTP body is not necessarily text-based. For example, if the client requested an image, then the HTTP body would likely be binary data. Also consider that, if the HTTP body is text-based, such as an HTML web page, it is free to use its own line-ending convention. It doesn't have to use the `\r\n` line ending required by HTTP.

If the client had sent a HEAD request type instead of `GET`, then the server would respond with exactly the same HTTP headers as before, but it would not include the HTTP body.

With the HTTP response format defined, let's look at some of the most common HTTP response types.

# HTTP response codes

There are many different types of HTTP response codes.

If the request was successful, then the server responds with a code in the 200 range:

*   `200 OK`: The client's request is successful, and the server sends the requested resource

If the resource has moved, the server can respond with a code in the 300 range. These codes are commonly used to redirect traffic from an unencrypted connection to an encrypted one, or to redirect traffic from a `www` subdomain to a bare one. They are also used if a website has undergone restructuring, but wants to keep incoming links working. The common 300 range codes are as follows:

*   `301 Moved Permanently`: The requested resource has moved to a new location. This location is indicated by the server in the `Location` header field. All future requests for this resource should use this new location.
*   `307 Moved Temporarily`: The requested resource has moved to a new location. This location is indicated by the server in the `Location` header field. This move may not be permanent, so future requests should still use the original location.

Errors are indicated by 400 or 500 range response codes. Some common ones are as follows:

*   `400 Bad Request`: The server doesn't understand/support the client's request
*   `401 Unauthorized`: The client isn't authorized for the requested resource
*   `403 Forbidden`: The client is forbidden to access the requested resource
*   `500 Internal Server Error`: The server encountered an error while trying to fulfill the client's request

In addition to a response type, the HTTP server must also be able to unambiguously communicate the length of the response body.

# Response body length

The HTTP body response length can be determined a few different ways. The simplest is if the HTTP server includes a `Content-Length` header line in its response. In that case, the server simply states the body length directly.

If the server would like to begin sending data before the body's length is known, then it can't use the `Content-Length` header line. In this case, the server can send a `Transfer-Encoding: chunked` header line. This header line indicates to the client that the response body will be sent in separate chunks. Each chunk begins with its chunk length, encoded in base-16 (hexadecimal), followed by a newline, and then the chunk data. The entire HTTP body ends with a zero-length chunk.

Let's consider an HTTP response example that uses chunked encoding:

```cpp
HTTP/1.1 200 OK
Content-Type: text/plain; charset=ascii
Transfer-Encoding: chunked

44
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eius
37
mod tempor incididunt ut labore et dolore magna aliqua.
0
```

In the preceding example, we see the HTTP body starts with `44` followed by a newline. This `44` should be interpreted as hexadecimal. We can use the built-in C `strtol()` function to interpret hexadecimal numbers.

Hexadecimal numbers are commonly written with a `0x` prefix to disambiguate them from decimal. We identify them with this prefix here, but keep in mind that the HTTP protocol does not add this prefix.

The `0x44` hexadecimal number is equal to 68 in decimal. After the `44` and newline, we see 68 characters that are part of the requested resource. After the 68 character chunk, the server sends a newline.

The server then sent `37`. `0x37` is 55 in decimal. After a newline, 55 characters are sent as chunk data. The server then sends a zero-length chunk to indicate that the response is finished.

The client should interpret the complete HTTP response after it has decoded the chunking as `Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua`.

There are a few other ways to indicate the HTTP response body length besides `Content-Length` and `Transfer-Encoding: chunked`. However, the server is limited to those two unless the client explicitly states support for additional encoding types in the HTTP request.

You may sometimes see a server that simply closes the TCP connection when it has finished transmitting a resource. That was a common way to indicate resource size in `HTTP/1.0`. However, this method shouldn't be used with `HTTP/1.1`. The issue with using a closed connection to indicate response length is that it's ambiguous as to why the connection was closed. It could be because all data has been sent, or it could be because of some other reason. Consider what would happen if a network cable is unplugged while data is being sent.

Now that we've seen the basics of HTTP requests and responses, let's look at how web resources are identified.

# What's in a URL

**Uniform Resource Locators** (**URL**), also known as web addresses, provide a convenient way to specify particular web resources. You can navigate to a URL by typing it into your web browser's address bar. Alternately, if you're browsing a web page and click on a link, that link is indicated with a URL.

Consider the `http://www.example.com:80/res/page1.php?user=bob#account` URL. Visually, the URL can be broken down like this:

![](img/afe37f05-570a-41e7-a535-037e9092b533.png)

The URL can indicate the protocol, the host, the port number, the document path, and hash. However, the host is the only required part. The other parts can be implied.

We can parse the example URL from the preceding diagram:

*   **http://**: The part before the first **://** indicates the protocol. In this example, the protocol is **http**, but it could be a different protocol such as `ftp://` or `https://`. If the protocol is omitted, the application will generally make an assumption. For example, your web browser would assume the protocol to be **http**.
*   **www.example.com**: This specifies the hostname. It is used to resolve an IP address that the HTTP client can connect to. This hostname must also appear in the HTTP request `Host` header field. This is required since multiple hostnames can resolve to the same IP address. This part can also be an IP address instead of a name. IPv4 addresses are used directly (`http://192.168.50.1/`), but IPv6 addresses should be put inside square brackets (`http://[::1]/`).
*   **:80**: The port number can be specified explicitly by using a colon after the hostname. If the port number is not specified, then the client uses the default port number for the given protocol. The default port number for **http** is **80**, and the default port number for **https** is **443**. Non-standard port numbers are common for testing and development.
*   **/res/page1.php?user/bob**: This specifies the document path. The HTTP server usually makes a distinction between the part before and after the question mark, but the HTTP client should not assign significance to this. The part after the question mark is often called the query string.
*   **#account**: This is called the hash. The hash specifies a position within a document, and the hash is not sent to the HTTP server. It instead allows a browser to scroll to a particular part of a document after the entire document is received from the HTTP server.

Now that we have a basic understanding of URLs, let's write code to parse them.

# Parsing a URL

We will write a C function to parse a given URL.

The function takes as input a URL, and it returns as output the hostname, the port number, and the document path. To avoid needing to do manual memory management, the outputs are returned as pointers to specific parts of the input URL. The input URL is modified with terminating null pointers as required.

Our function begins by printing the input URL. This is useful for debugging. The code for that is as follows:

```cpp
/*web_get.c excerpt*/

void parse_url(char *url, char **hostname, char **port, char** path) {
    printf("URL: %s\n", url);
```

The function then attempts to find `://` in the URL. If found, it reads in the first part of the URL as a protocol. Our program only supports HTTP. If the given protocol is not HTTP, then an error is returned. The code for parsing the protocol is as follows:

```cpp
/*web_get.c excerpt*/

    char *p;
    p = strstr(url, "://");

    char *protocol = 0;
    if (p) {
        protocol = url;
        *p = 0;
        p += 3;
    } else {
        p = url;
    }

    if (protocol) {
        if (strcmp(protocol, "http")) {
            fprintf(stderr,
                    "Unknown protocol '%s'. Only 'http' is supported.\n",
                    protocol);
            exit(1);
        }
    }
```

In the preceding code, a character pointer, `p`, is declared. `protocol` is also declared and set to `0` to indicate that no protocol has been found. `strstr()` is called to search for `://` in the URL. If it is not found, then `protocol` is left at `0`, and `p` is set to point back to the beginning of the URL. However, if `://` is found, then `protocol` is set to the beginning of the URL, which contains the protocol. `p` is set to one after `://`, which should be where the hostname begins.

If `protocol` was set, the code then checks that it points to the text `http`.

At this point in the code, `p` points to the beginning of the hostname. The code can save the hostname into the return variable, `hostname`. The code must then scan for the end of the hostname by looking for the first colon, slash, or hash. The code for this is as follows:

```cpp
/*web_get.c excerpt*/

    *hostname = p;
    while (*p && *p != ':' && *p != '/' && *p != '#') ++p;
```

Once `p` has advanced to the end of the hostname, we must check whether a port number was found. A port number starts with a colon. If a port number is found, our code returns it in the `port` variable; otherwise, a default port number of `80` is returned. The code to check for a port number is as follows:

```cpp
/*web_get.c excerpt*/

    *port = "80";
    if (*p == ':') {
        *p++ = 0;
        *port = p;
    }
    while (*p && *p != '/' && *p != '#') ++p;
```

After the port number, `p` points to the document path. The function returns this part of the URL in the `path` variable. Note that our function omits the first `/` in the path. This is for simplicity because it allows us to avoid allocating any memory. All document paths start with `/`, so the function caller can easily prepend that when the HTTP request is constructed.

The code to set the `path` variable is as follows:

```cpp
/*web_get.c excerpt*/

    *path = p;
    if (*p == '/') {
        *path = p + 1;
    }
    *p = 0;
```

The code then attempts to find a hash, if it exists. If it does exist, it is overwritten with a terminating null character. This is because the hash is never sent to the web server and is ignored by our HTTP client.

The code that advances to the hash is as follows:

```cpp
/*web_get.c excerpt*/

    while (*p && *p != '#') ++p;
    if (*p == '#') *p = 0;
```

Our function has now parsed out the hostname, port number, and document path. It then prints out these values for debugging purposes and returns. The final code for the `parse_url()` function is as follows:

```cpp
/*web_get.c excerpt*/

    printf("hostname: %s\n", *hostname);
    printf("port: %s\n", *port);
    printf("path: %s\n", *path);
}
```

Now that we have code to parse a URL, we are one step closer to building an entire HTTP client.

# Implementing a web client

We will now implement an HTTP web client. This client takes as input a URL. It then attempts to connect to the host and retrieve the resource given by the URL. The program displays the HTTP headers that are sent and received, and it attempts to parse out the requested resource content from the HTTP response.

Our program begins by including the chapter header, `chap06.h`:

```cpp
/*web_get.c*/

#include "chap06.h"
```

We then define a constant, `TIMEOUT`. Later in our program, if an HTTP response is taking more than `TIMEOUT` seconds to complete, then our program abandons the request. You can define `TIMEOUT` as you like, but we give it a value of five seconds here:

```cpp
/*web_get.c continued*/

#define TIMEOUT 5.0
```

Now, please include the entire `parse_url()` function as given in the previous section. Our client needs `parse_url()` to find the hostname, port number, and document path from a given URL.

Another helper function is used to format and send the HTTP request. We call it `send_request()`, and its code is given next:

```cpp
/*web_get.c continued*/

void send_request(SOCKET s, char *hostname, char *port, char *path) {
    char buffer[2048];

    sprintf(buffer, "GET /%s HTTP/1.1\r\n", path);
    sprintf(buffer + strlen(buffer), "Host: %s:%s\r\n", hostname, port);
    sprintf(buffer + strlen(buffer), "Connection: close\r\n");
    sprintf(buffer + strlen(buffer), "User-Agent: honpwc web_get 1.0\r\n");
    sprintf(buffer + strlen(buffer), "\r\n");

    send(s, buffer, strlen(buffer), 0);
    printf("Sent Headers:\n%s", buffer);
}
```

`send_request()` works by first defining a character buffer in which to store the HTTP request. It then uses the `sprintf()` function to write to the buffer until the HTTP request is complete. The HTTP request ends with a blank line. This blank line tells the server that the entire request header has been received.

Once the request is formatted into `buffer`, `buffer` is sent over an open socket using `send()`. `buffer` is also printed to the console for debugging purposes. We define one more helper function for our web client. This function, `connect_to_host()`, takes in a hostname and port number and attempts to establish a new TCP socket connection to it.

In the first part of `connect_to_host()`, `getaddrinfo()` is used to resolve the hostname. `getnameinfo()` is then used to print out the server IP address for debugging purposes. The code for this is as follows:

```cpp
/*web_get.c continued*/

SOCKET connect_to_host(char *hostname, char *port) {
    printf("Configuring remote address...\n");
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo *peer_address;
    if (getaddrinfo(hostname, port, &hints, &peer_address)) {
        fprintf(stderr, "getaddrinfo() failed. (%d)\n", GETSOCKETERRNO());
        exit(1);
    }

    printf("Remote address is: ");
    char address_buffer[100];
    char service_buffer[100];
    getnameinfo(peer_address->ai_addr, peer_address->ai_addrlen,
            address_buffer, sizeof(address_buffer),
            service_buffer, sizeof(service_buffer),
            NI_NUMERICHOST);
    printf("%s %s\n", address_buffer, service_buffer);
```

In the second part of `connect_to_host()`, a new socket is created with `socket()`, and a TCP connection is established with `connect()`. If everything goes well, the function returns the created socket. The code for the second half of `connect_to_host()` is as follows:

```cpp
/*web_get.c continued*/

    printf("Creating socket...\n");
    SOCKET server;
    server = socket(peer_address->ai_family,
            peer_address->ai_socktype, peer_address->ai_protocol);
    if (!ISVALIDSOCKET(server)) {
        fprintf(stderr, "socket() failed. (%d)\n", GETSOCKETERRNO());
        exit(1);
    }

    printf("Connecting...\n");
    if (connect(server,
                peer_address->ai_addr, peer_address->ai_addrlen)) {
        fprintf(stderr, "connect() failed. (%d)\n", GETSOCKETERRNO());
        exit(1);
    }
    freeaddrinfo(peer_address);

    printf("Connected.\n\n");

    return server;
}
```

If you've been working through this book from the beginning, the code in `connect_to_host()` should be very familiar by now. If it's not, please refer to the previous chapters for a more detailed explanation of `getaddrinfo()`, `socket()`, and `connect()`. [Chapter 3](eb2e080f-fad2-4e14-b4c1-9a6c124df77c.xhtml), *An In-Depth Overview of TCP Connections*, should be particularly helpful.

With our helper functions out of the way, we can now begin to define the `main()` function. We begin `main()` with the following code:

```cpp
/*web_get.c continued*/

int main(int argc, char *argv[]) {

#if defined(_WIN32)
    WSADATA d;
    if (WSAStartup(MAKEWORD(2, 2), &d)) {
        fprintf(stderr, "Failed to initialize.\n");
        return 1;
    }
#endif

    if (argc < 2) {
        fprintf(stderr, "usage: web_get url\n");
        return 1;
    }
    char *url = argv[1];
```

In the preceding code, Winsock is initialized, if needed, and the program's arguments are checked. If a URL is given as an argument, it is stored in the `url` variable.

We can then parse the URL into its hostname, port, and path parts with the following code:

```cpp
/*web_get.c continued*/

    char *hostname, *port, *path;
    parse_url(url, &hostname, &port, &path);
```

The program then continues by establishing a connection to the target server and sending the HTTP request. This is made easy by using the two helper functions we defined previously, `connect_to_host()` and `send_request()`. The code for this is as follows:

```cpp
/*web_get.c continued*/

    SOCKET server = connect_to_host(hostname, port);
    send_request(server, hostname, port, path);
```

One feature of our web client is that it times out if a request takes too long to complete. In order to know how much time has elapsed, we need to record the start time. This is done using a call to the built-in `clock()` function. We store the start time in the `start_time` variable with the following:

```cpp
/*web_get.c continued*/

    const clock_t start_time = clock();
```

It is now necessary to define some more variables that can be used for bookkeeping while receiving and parsing the HTTP response. The requisite variables are as follows:

```cpp
/*web_get.c continued*/

#define RESPONSE_SIZE 8192
    char response[RESPONSE_SIZE+1];
    char *p = response, *q;
    char *end = response + RESPONSE_SIZE;
    char *body = 0;

    enum {length, chunked, connection};
    int encoding = 0;
    int remaining = 0;
```

In the preceding code, `RESPONSE_SIZE` is the maximum size of the HTTP response we reserve memory for. Our program is unable to parse HTTP responses bigger than this. If you extend this limit, it may be useful to use `malloc()` to reserve memory on the heap instead of the stack.

`response` is a character array that holds the entire HTTP response. `p` is a `char` pointer that keeps track of how far we have written into `response` so far. `q` is an additional `char` pointer that is used later. We define `end` as a `char` pointer, which points to the end of the `response` buffer. `end` is useful to ensure that we don't attempt to write past the end of our reserved memory.

The `body` pointer is used to remember the beginning of the HTTP response body once received.

If you recall, the HTTP response body length can be determined by a few different methods. We define an enumeration to list the method types, and we define the `encoding` variable to store the actual method used. Finally, the `remaining` variable is used to record how many bytes are still needed to finish the HTTP body or body chunk.

We then start a loop to receive and process the HTTP response. This loop first checks that it hasn't taken too much time and that we still have buffer space left to store the received data. The first part of this loop is as follows:

```cpp
/*web_get.c continued*/

    while(1) {

        if ((clock() - start_time) / CLOCKS_PER_SEC > TIMEOUT) {
            fprintf(stderr, "timeout after %.2f seconds\n", TIMEOUT);
            return 1;
        }

        if (p == end) {
            fprintf(stderr, "out of buffer space\n");
            return 1;
        }
```

We then include the code to receive data over the TCP socket. Our code uses `select()` with a short timeout. This allows us to periodically check that the request hasn't timed out. You may recall from previous chapters that `select()` involves creating `fd_set` and `timeval` structures. The following code creates these objects and calls `select()`:

```cpp
/*web_get.c continued*/

        fd_set reads;
        FD_ZERO(&reads);
        FD_SET(server, &reads);

        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 200000;

        if (select(server+1, &reads, 0, 0, &timeout) < 0) {
            fprintf(stderr, "select() failed. (%d)\n", GETSOCKETERRNO());
            return 1;
        }
```

`select()` returns when either the timeout has elapsed, or new data is available to be read from the socket. Our code needs to use `FD_ISSET()` to determine whether new data is available to be read. If so, we read the data into the buffer at the `p` pointer.

Alternatively, when attempting to read new data, we may find that the socket was closed by the web server. If this is the case, we check whether we were expecting a closed connection to indicate the end of the transmission. That is the case if `encoding == connection`. If so, we print the HTTP body data that was received.

The code for reading in new data and detecting a closed connection is as follows:

```cpp
/*web_get.c continued*/

        if (FD_ISSET(server, &reads)) {
            int bytes_received = recv(server, p, end - p, 0);
            if (bytes_received < 1) {
                if (encoding == connection && body) {
                    printf("%.*s", (int)(end - body), body);
                }

                printf("\nConnection closed by peer.\n");
                break;
            }

            /*printf("Received (%d bytes): '%.*s'",
                    bytes_received, bytes_received, p);*/

            p += bytes_received;
            *p = 0;
```

Note that, in the preceding code, the `p` pointer is advanced to point to the end of received data. `*p` is set to zero, so our received data always ends with a null terminator. This allows us to use standard functions on the data that expect a null-terminated string. For example, we use the built-in `strstr()` function to search through the received data, and `strstr()` expects the input string to be null-terminated.

Next, if the HTTP body hasn't already been found, our code searches through the received data for a blank line that indicates the end of the HTTP header. A blank line is encoded by two consecutive line endings. HTTP defines a line ending as `\r\n`, so our code detects a blank line by searching for `\r\n\r\n`.

The following code finds the end of the HTTP header (which is the beginning of the HTTP body) using `strstr()`, and updates the `body` pointer to point to the beginning of the HTTP body:

```cpp
/*web_get.c continued*/

            if (!body && (body = strstr(response, "\r\n\r\n"))) {
                *body = 0;
                body += 4;
```

It may be useful to print the HTTP header for debugging. This can be done with the following code:

```cpp
/*web_get.c continued*/

                printf("Received Headers:\n%s\n", response);
```

Now that the headers have been received, we need to determine whether the HTTP server is using `Content-Length` or `Transfer-Encoding: chunked` to indicate body length. If it doesn't send either, then we assume that the entire HTTP body has been received once the connection is closed.

If the `Content-Length` is found using `strstr()`, we set `encoding = length` and store the body length in the `remaining` variable. The actual length is read from the HTTP header using the `strtol()` function.

If `Content-Length` is not found, then the code searches for `Transfer-Encoding: chunked`. If found, we set `encoding = chunked`. `remaining` is set to `0` to indicate that we haven't read in a chunk length yet.

If neither `Content-Length` or `Transfer-Encoding: chunked` is found, then `encoding = connection` is set to indicate that we consider the HTTP body received when the connection is closed.

The code for determining which body length method is used is as follows:

```cpp
/*web_get.c continued*/

                q = strstr(response, "\nContent-Length: ");
                if (q) {
                    encoding = length;
                    q = strchr(q, ' ');
                    q += 1;
                    remaining = strtol(q, 0, 10);

                } else {
                    q = strstr(response, "\nTransfer-Encoding: chunked");
                    if (q) {
                        encoding = chunked;
                        remaining = 0;
                    } else {
                        encoding = connection;
                    }
                }
                printf("\nReceived Body:\n");
            }
```

The preceding code could be made more robust by doing case-insensitive searching, or by allowing for some flexibility in spacing. However, it should work with most web servers as is, and we're going to continue to keep it simple.

If the HTTP body start has been identified, and `encoding == length`, then the program simply needs to wait until `remaining` bytes have been received. The following code checks for this:

```cpp
/*web_get.c continued*/

            if (body) {
                if (encoding == length) {
                    if (p - body >= remaining) {
                        printf("%.*s", remaining, body);
                        break;
                    }
```

In the preceding code, once `remaining` bytes of the HTTP body have been received, it prints the received body and breaks from the `while` loop.

If `Transfer-Encoding: chunked` is used, then the receiving logic is a bit more complicated. The following code handles this:

```cpp
/*web_get.c continued*/

                } else if (encoding == chunked) {
                    do {
                        if (remaining == 0) {
                            if ((q = strstr(body, "\r\n"))) {
                                remaining = strtol(body, 0, 16);
                                if (!remaining) goto finish;
                                body = q + 2;
                            } else {
                                break;
                            }
                        }
                        if (remaining && p - body >= remaining) {
                            printf("%.*s", remaining, body);
                            body += remaining + 2;
                            remaining = 0;
                        }
                    } while (!remaining);
                }
            } //if (body)
```

In the preceding code, the `remaining` variable is used to indicate whether a chunk length or chunk data is expected next. When `remaining == 0`, the program is waiting to receive a new chunk length. Each chunk length ends with a newline; therefore, if a newline is found with `strstr()`, we know that the entire chunk length has been received. In this case, the chunk length is read using `strtol()`, which interprets the hexadecimal chunk length. `remaining` is set to the expected chunk length. A chunked message is terminated by a zero-length chunk, so if `0` was read, the code uses `goto finish` to break out of the main loop.

If the `remaining` variable is non-zero, then the program checks whether at least `remaining` bytes of data have been received. If so, that chunk is printed, and the `body` pointer is advanced to the end of the current chunk. This logic loops until it finds the terminating zero-length chunk or runs out of data.

At this point, we've shown all of the logic to parse the HTTP response body. We only need to end our loops, close the socket, and the program is finished. Here is the final code for `web_get.c`:

```cpp
/*web_get.c continued*/

        } //if FDSET
    } //end while(1)
finish:

    printf("\nClosing socket...\n");
    CLOSESOCKET(server);

#if defined(_WIN32)
    WSACleanup();
#endif

    printf("Finished.\n");
    return 0;
}
```

You can compile and run `web_get.c` on Linux and macOS with the following commands:

```cpp
gcc web_get.c -o web_get
./web_get http://example.com/
```

On Windows, the command to compile and run using MinGW is as follows:

```cpp
gcc web_get.c -o web_get.exe -lws2_32
web_get.exe http://example.com/
```

Try running `web_get` with different URLs and study the outputs. You may find the HTTP response headers interesting.

The following screenshot shows what happens when we run `web_get` on `http://example.com/`:

![](img/74f1a912-e88f-48ac-a49b-831750b1f4d6.png)

`web_get` only supports the `GET` queries. `POST` queries are also common and useful. Let's now look at HTTP `POST` requests.

# HTTP POST requests

An HTTP `POST` request sends data from the web client to the web server. Unlike an HTTP `GET` request, a `POST` request includes a body containing data (although this body could be zero-length).

The `POST` body format can vary, and it should be identified by a `Content-Type` header. Many modern, web-based APIs expect a `POST` body to be JSON encoded.

Consider the following HTTP `POST` request:

```cpp
POST /orders HTTP/1.1
Host: example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:64.0)
Content-Type: application/json
Content-Length: 56
Connection: close

{"symbol":"VOO","qty":"10","side":"buy","type":"market"}
```

In the preceding example, you can see that the HTTP `POST` request is similar to an HTTP `GET` request. Notable differences are as follows: the request starts with `POST` instead of `GET`; a `Content-Type` header field is included; a `Content-Length` header field is present; and an HTTP message body is included. In that example, the HTTP message body is in JSON format, as specified by the `Content-Type` header.

# Encoding form data

If you encounter a form on a website, such as a login form, that form likely uses a `POST` request to transmit its data to the web server. A standard HTML form encodes the data it sends in a format called **URL encoding**, also called **percent encoding**. When URL encoded form data is submitted in an HTTP `POST` request, it uses the `Content-Type: application/x-www-form-urlencoded` header.

Consider the following HTML for a submittable form:

```cpp
<form method="post" action="/submission.php">

  <label for="name">Name:</label>
  <input name="name" type="text"><br>

  <label for="comment">Comment:</label>
  <input name="comment" type="text"><br>

  <input type="submit" value="submit">

</form>
```

In your web browser, the preceding HTML may render as shown in the following screenshot:

![](img/ca257bf2-56b9-45fd-8dca-7541497ca0ee.png)

When this form is submitted, its data is encoded in an HTTP request such as the following:

```cpp
POST /submission.php HTTP/1.1
Host: 127.0.0.1:8080
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7)
Accept-Language: en-US
Accept-Encoding: gzip, deflate
Content-Type: application/x-www-form-urlencoded
Content-Length: 31
Connection: keep-alive

name=Alice&comment=Well+Done%21
```

In the preceding HTTP request, you can see that `Content-Type: application/x-www-form-urlencoded` is used. In this format, each form field and value is paired by an equal sign, and multiple form fields are chained together by ampersands.

Special characters in form field names or values must be encoded. Notice that `Well Done!` was encoded as `Well+Done%21`. Spaces are encoded with plus symbols, and special characters are encoded by a percent sign followed by their two-digit hexadecimal value (thus, the exclamation point was encoded as `%21`). A percent sign itself would be encoded as `%25`.

# File uploads

When an HTML form includes file uploads, the browser uses a different content type. In this case, `Content-Type: multipart/form-data` is used. When `Content-Type: multipart/form-data` is used, a boundary specifier is included. This boundary is a special delimiter, set by the sender, which separates parts of the submitted form data.

Consider the following HTML form:

```cpp
<form method="post" enctype="multipart/form-data" action="/submit.php">
    <input name="name" type="text"><br>
    <input name="comment" type="text"><br>
    <input name="file" type="file"><br>
    <input type="submit" value="submit">
</form>
```

If the user navigates to a web page bearing the HTML form from the preceding code, and enters the name `Alice`, the comment `Well Done!`, and selects a file to upload called `upload.txt`, then the following HTTP `POST` request could be sent by the browser:

```cpp
POST /submit.php HTTP/1.1
Host: example.com
Content-Type: multipart/form-data; boundary=-----------233121195710604
Content-Length: 1727

-------------233121195710604
Content-Disposition: form-data; name="name"

Alice
-------------233121195710604
Content-Disposition: form-data; name="comment"

Well Done!
-------------233121195710604
Content-Disposition: form-data; name="file"; filename="upload.txt"
Content-Type: text/plain

Hello.... <truncated>
```

As you can see, when using `multipart/form-data`, each section of data is separated by a boundary. This boundary is what allows the receiver to delineate between separate fields or uploaded files. It is important that this boundary is chosen so that it does not appear in any submitted field or uploaded file!

# Summary

HTTP is the protocol that powers the modern internet. It is behind every web page, every link click, every graphic loaded, and every form submitted. In this chapter, we saw that HTTP is a text-based protocol that runs over a TCP connection. We learned the HTTP formats for both client requests and server responses.

In this chapter, we also implemented a simple HTTP client in C. This client had a few non-trivial tasks – parsing a URL, formatting a `GET` request HTTP header, waiting for a response, and parsing the received data out of the HTTP response. In particular, we looked at handling two different methods of parsing out the HTTP body. The first, and easiest, method was `Content-Length`, where the entire body length is explicitly specified. The second method was chunked encoding, where the body is sent as separate chunks, which our program had to delineate between.

We also briefly looked at the `POST` requests and the content formats associated with them.

In the next chapter, [Chapter 7](f352830e-089c-4369-b7a2-18a896e1c5d5.xhtml), *Building a Simple Web Server*, we will develop the counterpart to our HTTP client—an HTTP server.

# Questions

Try these questions to test your knowledge from this chapter:

1.  Does HTTP use TCP or UDP?
2.  What types of resources can be sent over HTTP?
3.  What are the common HTTP request types?
4.  What HTTP request type is typically used to send data from the server to the client?
5.  What HTTP request type is typically used to send data from the client to the server?
6.  What are the two common methods used to determine an HTTP response body length?
7.  How is the HTTP request body formatted for a `POST`-type HTTP request?

The answers to these questions can be found in [Appendix A](bd8b8f52-52cb-4d34-b01b-e907564bfece.xhtml), *Answers to Questions*.

# Further reading

For more information about HTTP and HTML, please refer to the following resources:

*   **RFC 7230**: *Hypertext Transfer Protocol (HTTP/1.1): Message Syntax and Routing* ([https://tools.ietf.org/html/rfc7230](https://tools.ietf.org/html/rfc7230))
*   **RFC 7231**: *Hypertext Transfer Protocol (HTTP/1.1): Semantics and Content* ([https://tools.ietf.org/html/rfc7231](https://tools.ietf.org/html/rfc7231))
*   **RFC 1866**: *Hypertext Markup Language – 2.0* ([https://tools.ietf.org/html/rfc1866](https://tools.ietf.org/html/rfc1866))
*   **RFC 3986**: *Uniform Resource Identifier (URI): Generic Syntax* ([https://tools.ietf.org/html/rfc3986](https://tools.ietf.org/html/rfc3986))