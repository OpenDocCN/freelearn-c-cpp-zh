# Implementing a Secure Web Server

In this chapter, we will build a simple HTTPS server program. This serves as the counterpart to the HTTPS client we worked on in the previous chapter.

HTTPS is powered by **Transport Layer Security** (**TLS**). HTTPS servers, unlike HTTPS clients, are expected to identify themselves with certificates. We'll cover how to listen for HTTPS connections, provide certificates, and send an HTTP response over TLS.

The following topics are covered in this chapter:

*   HTTPS overview
*   HTTPS certificates
*   HTTPS server setup with OpenSSL
*   Accepting HTTPS connections
*   Common problems
*   OpenSSL alternatives
*   Direct TLS termination alternatives

# Technical requirements

This chapter continues where [Chapter 9](47ba170d-42d9-4e38-b5d8-89503e005708.xhtml), *Loading Secure Web Pages with HTTPS and OpenSSL*, left off. This chapter continues to use the OpenSSL library. It is imperative that you have the OpenSSL library installed and that you know the basics of programming with OpenSSL. Refer to [Chapter 9](47ba170d-42d9-4e38-b5d8-89503e005708.xhtml), *Loading Secure Web Pages with HTTPS and OpenSSL*, for basic information about OpenSSL.

The example programs from this chapter can be compiled using any modern C compiler. We recommend **MinGW** for Windows and **GCC** for Linux and macOS. You also need to have the OpenSSL library installed. See [Appendix B](47da8507-709b-44a6-9399-b18ce6afd8c9.xhtml), *Setting Up Your C Compiler on Windows;* [Appendix C](221eebc0-0bb1-4661-a5aa-eafed9fcba7e.xhtml), *Setting Up Your C Compiler on Linux;* and [Appendix D](632db68e-0911-4238-a2be-bd1aa5296120.xhtml), *Setting Up Your C Compiler on macOS*, for compiler setup and OpenSSL installation.

The code for this book can be found at [https://github.com/codeplea/Hands-On-Network-Programming-with-C](https://github.com/codeplea/Hands-On-Network-Programming-with-C).

From the command line, you can download the code for this chapter by using the following command:

```cpp
git clone https://github.com/codeplea/Hands-On-Network-Programming-with-C
cd Hands-On-Network-Programming-with-C/chap10
```

Each example program in this chapter runs on Windows, Linux, and macOS. When compiling on Windows, each example program requires linking with the **Winsock** library. This is accomplished by passing the `-lws2_32` option to `gcc`.

Each example also needs to be linked against the OpenSSL libraries, `libssl.a` and `libcrypto.a`. This is accomplished by passing  `-lssl -lcrypto` to GCC.

We provide the exact commands needed to compile each example as it is introduced.

All of the example programs in this chapter require the same header files and C macros that we developed in [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs*. For brevity, we put these statements in a separate header file, `chap10.h`, which we can include in each program. For an explanation of these statements, please refer to [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs*.

The content of `chap10.h` begins by including the required networking header files. The code for this is as follows:

```cpp
/*chap10.h*/

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
```

We also define some macros to assist with writing portable code:

```cpp
/*chap10.h continued*/

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
```

Finally, `chap10.h` includes the additional headers required by this chapter's programs:

```cpp
/*chap10.h continued*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include <openssl/crypto.h>
#include <openssl/x509.h>
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
```

# HTTPS and OpenSSL summary

We begin with a quick review of the HTTPS protocol, as covered in [Chapter 9](47ba170d-42d9-4e38-b5d8-89503e005708.xhtml), *Loading Secure Web Pages with HTTPS and OpenSSL*. However, we do recommend that you work through [Chapter 9](47ba170d-42d9-4e38-b5d8-89503e005708.xhtml), *Loading Secure Web Pages with HTTPS and OpenSSL*, before beginning this chapter.

HTTPS uses TLS to add security to HTTP. You will recall from [Chapter 6](de3d2e9b-b94e-47d1-872c-c2ecb34c4026.xhtml), *Building a Simple Web Client*, and [Chapter 7](f352830e-089c-4369-b7a2-18a896e1c5d5.xhtml), *Building a Simple Web Server*, that HTTP is a text-based protocol that works over TCP on port `80`. The TLS protocol can be used to add security to any TCP-based protocol. Specifically, TLS is used to provide security for HTTPS. So in a nutshell, HTTPS is simply HTTP with TLS. The default HTTPS port is `443`.

OpenSSL is a popular open source library that provides functionality for TLS/SSL and HTTPS. We use it in this book to provide the methods needed to implement HTTPS clients and servers.

Generally, HTTPS connections are first made using TCP sockets. Once the TCP connection is established, OpenSSL is used to negotiate a TLS connection over the open TCP connection. From that point forward, OpenSSL functions are used to send and receive data over the TLS connection.

An important part of communication security is being able to trust that the connection is to the intended party. No amount of data encryption helps if you have connected to an impostor. TLS uses certificates to prevent against connecting to impostors and man-in-the-middle attacks.

We now need to understand certificates in more detail before we can proceed with our HTTPS server.

# Certificates

**Certificates** are an important part of the TLS protocol. Although certificates can be used on both the client and server side, HTTPS generally only uses server certificates. These certificates identify to the client that they are connected to a trusted server.

Without certificates, a client wouldn't be able to tell whether it was connected to the intended server or an impostor server.

Certificates work using a **chain-of-trust** model. Each HTTPS client has a few certificate authorities that they explicitly trust, and certificate authorities offer services where they digitally sign certificates. This service is usually done for a small fee, and usually only after some simple verification of the requester.

When an HTTPS client sees a certificate signed by an authority it trusts, then it also trusts that certificate. Indeed, these chains of trust can run deep. For example, most certificate authorities also allow resellers. In these cases, the certificate authority signs an intermediate certificate to be used by the reseller. The reseller then uses this intermediate certificate to sign new certificates. Clients trust the intermediate certificates because they are signed by trusted certificate authorities, and clients trust the certificates signed by resellers because they trust the intermediate certificates.

Certificate authorities commonly offer two types of validation. **Domain validation** is where a signed certificate is issued after simply verifying that the certificate recipient can be reached at the given domain. This is usually done by having the certificate requester temporarily modify a DNS record, or reply to an email sent to their *whois* contact.

**Let's Encrypt** is a relatively new certificate issuer that issues certificates for free. They do this using an automated model. **Domain validation** is done by having the certificate requester serve a small file over HTTP or HTTPS.

Domain validation is the most common type of validation. An HTTPS server using domain validation assures an HTTPS client that they are connected to the domain that they think they are. It implies that their connection wasn't silently hijacked or otherwise intercepted.

Certificate authorities also offer **Extended Validation** (**EV**) certificates. EV certificates are only issued after the authority verifies the recipient's identity. This is usually done using public records and a phone call.

For public-facing HTTPS applications, it is important that you obtain a certificate from a recognized certificate authority. However, this can sometimes be tedious, and it is often much more convenient to obtain a **self-signed** certificate for development and testing purposes. Let's do that now.

# Self-signed certificates with OpenSSL

Certificates signed by recognized authorities are essential to establish the chain of trust needed for public websites. However, it is much easier to obtain a self-signed certificate for testing or development.

It is also acceptable to use a self-signed certificate for certain private applications where the client can be deployed with a copy of the certificate, and trust only that certificate. This is called **certificate pinning**. Indeed, when used properly, certificate pinning can be more secure than using a certificate authority. However, it is not appropriate for public-facing websites.

We require a certificate to test our HTTPS server. We use a self-signed certificate because they are the easiest to obtain. The downside to this method is that web browsers won't trust our server. We can get around this by clicking through a few warnings in the web browser.

OpenSSL provides tools to make self-signing certificates very easy.

The basic command to self-sign a certificate is as follows:

```cpp
openssl req -x509 -newkey rsa:2048 -nodes -sha256 -keyout key.pem \
-out cert.pem -days 365
```

OpenSSL asks questions about what to put on the certificate, including the subject, your name, company, location, and so on. You can use the defaults on all of these as this doesn't matter for our testing purposes.

The preceding command places the new certificate in `cert.pem` and the key for it in `key.pem`. Our HTTPS server needs both files. `cert.pem` is the certificate that gets sent to the connected client, and `key.pem` provides our server with the encryption key that proves that it owns the certificate. Keeping this key secret is imperative.

Here is a screenshot showing the generation of a new self-signed certificate:

![](img/33777266-b70f-441a-861a-c06e3133a2df.png)

You can also use OpenSSL to view a certificate. The following command does this:

```cpp
openssl x509 -text -noout -in cert.pem
```

If you're on Windows using MSYS, you may get garbled line endings from the previous command. If so, try using `unix2dos` to fix it, as shown by the following command:

```cpp
openssl x509 -text -noout -in cert.pem | unix2dos
```

Here is what a typical self-signed certificate looks like:

![](img/4a937bae-cd76-4e13-9638-84b14a0172ed.png)

Now that we have a usable certificate, we are ready to begin our HTTPS server programming.

# HTTPS server with OpenSSL

Let's go over some basics of using the OpenSSL library in server applications before beginning a concrete example.

Before OpenSSL can be used, it must be initialized. The following code initializes the OpenSSL library, loads the requisite encryption algorithms, and loads useful error strings:

```cpp
SSL_library_init();
OpenSSL_add_all_algorithms();
SSL_load_error_strings();
```

Refer to the previous [Chapter 9](47ba170d-42d9-4e38-b5d8-89503e005708.xhtml), *Loading Secure Web Pages with HTTPS and OpenSSL,* for more information.

Our server also needs to create an SSL context object. This object works as a sort of factory from which we can create TLS/SSL connections.

The following code creates the `SSL_CTX` object:

```cpp
SSL_CTX *ctx = SSL_CTX_new(TLS_server_method());
if (!ctx) {
    fprintf(stderr, "SSL_CTX_new() failed.\n");
    return 1;
}
```

If you're using an older version of OpenSSL, you may need to replace `TLS_server_method()` with `TLSv1_2_server_method()` in the preceding code. However, a better solution is to upgrade to a newer OpenSSL version.

After the `SSL_CTX` object is created, we can set it to use our self-signed certificate and key. The following code does this:

```cpp
if (!SSL_CTX_use_certificate_file(ctx, "cert.pem" , SSL_FILETYPE_PEM)
|| !SSL_CTX_use_PrivateKey_file(ctx, "key.pem", SSL_FILETYPE_PEM)) {
    fprintf(stderr, "SSL_CTX_use_certificate_file() failed.\n");
    ERR_print_errors_fp(stderr);
    return 1;
}
```

That concludes the minimal OpenSSL setup needed for an HTTPS server.

The server should then listen for incoming TCP connections. This was covered in detail in [Chapter 3](eb2e080f-fad2-4e14-b4c1-9a6c124df77c.xhtml), *An In-Depth Overview of TCP Connections*.

After a new TCP connection is established, we use the socket returned by `accept()` to create our TLS/SSL socket.

First, a new `SSL` object is created using our SSL context from earlier. The following code demonstrates this:

```cpp
SSL *ssl = SSL_new(ctx);
if (!ctx) {
    fprintf(stderr, "SSL_new() failed.\n");
    return 1;
}
```

The `SSL` object is then linked to our open TCP socket using `SSL_set_fd()`. The `SSL_accept()` function is called to establish the TLS/SSL connection. The following code demonstrates this:

```cpp
SSL_set_fd(ssl, socket_client);
if (SSL_accept(ssl) <= 0) {
    fprintf(stderr, "SSL_accept() failed.\n");
    ERR_print_errors_fp(stderr);
    return 1;
}

printf ("SSL connection using %s\n", SSL_get_cipher(ssl));
```

You may notice that this code is very similar to the HTTPS client code from [Chapter 9](47ba170d-42d9-4e38-b5d8-89503e005708.xhtml), L*oading Secure Web Pages with HTTPS and OpenSSL*. The only real differences are in the setup of the SSL context object.

Once the TLS connection is established, data can be sent and received using `SSL_write()` and `SSL_read()`. These functions replace the `send()` and `recv()` functions used with TCP sockets.

When the connection is finished, it is important to free resources, as shown by the following code: 

```cpp
SSL_shutdown(ssl);
CLOSESOCKET(socket_client);
SSL_free(ssl);
```

When your program is finished accepting new connections, you should also free the SSL context object. The following code shows this:

```cpp
SSL_CTX_free(ctx);
```

With an understanding of the basics out of the way, let's solidify our knowledge by implementing a simple example program.

# Time server example

In this chapter, we develop a simple time server that displays the time to an HTTPS client. This program is an adaptation of `time_server.c` from [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs*, which served the time over plain HTTP. Our program begins by including the chapter header, defining `main()`, and initializing Winsock on Windows. The code for this is as follows:

```cpp
/*tls_time_server.c*/

#include "chap10.h"

int main() {

#if defined(_WIN32)
    WSADATA d;
    if (WSAStartup(MAKEWORD(2, 2), &d)) {
        fprintf(stderr, "Failed to initialize.\n");
        return 1;
    }
#endif
```

The OpenSSL library is then initialized with the following code:

```cpp
/*tls_time_server.c continued*/

    SSL_library_init();
    OpenSSL_add_all_algorithms();
    SSL_load_error_strings();
```

An SSL context object must be created for our server. This is done using a call to `SSL_CTX_new()`. The following code shows this call:

```cpp
/*tls_time_server.c continued*/

    SSL_CTX *ctx = SSL_CTX_new(TLS_server_method());
    if (!ctx) {
        fprintf(stderr, "SSL_CTX_new() failed.\n");
        return 1;
    }
```

If you're using an older version of OpenSSL, you may need to replace `TLS_server_method()` with `TLSv1_2_server_method()` in the preceding code. However, you should probably upgrade to a newer OpenSSL version instead.

Once the SSL context has been created, we can associate our server's certificate with it. The following code sets the SSL context to use our certificate:

```cpp
/*tls_time_server.c continued*/

    if (!SSL_CTX_use_certificate_file(ctx, "cert.pem" , SSL_FILETYPE_PEM)
    || !SSL_CTX_use_PrivateKey_file(ctx, "key.pem", SSL_FILETYPE_PEM)) {
        fprintf(stderr, "SSL_CTX_use_certificate_file() failed.\n");
        ERR_print_errors_fp(stderr);
        return 1;
    }
```

Make sure that you've generated a proper certificate and key. Refer to the *Self-signed certificate with OpenSSL* section from earlier in this chapter.

Once the SSL context is configured with the proper certificate, our program creates a listening TCP socket in the normal way. It begins with a call to `getaddrinfo()` and `socket()`, as shown in the following code:

```cpp
/*tls_time_server.c continued*/

    printf("Configuring local address...\n");
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    struct addrinfo *bind_address;
    getaddrinfo(0, "8080", &hints, &bind_address);

    printf("Creating socket...\n");
    SOCKET socket_listen;
    socket_listen = socket(bind_address->ai_family,
            bind_address->ai_socktype, bind_address->ai_protocol);
    if (!ISVALIDSOCKET(socket_listen)) {
        fprintf(stderr, "socket() failed. (%d)\n", GETSOCKETERRNO());
        return 1;
    }
```

The socket created by the preceding code is bound to the listening address with `bind()`. The `listen()` function is used to set the socket in a listening state. The following code demonstrates this:

```cpp
  /*tls_time_server.c continued*/

    printf("Binding socket to local address...\n");
    if (bind(socket_listen,
                bind_address->ai_addr, bind_address->ai_addrlen)) {
        fprintf(stderr, "bind() failed. (%d)\n", GETSOCKETERRNO());
        return 1;
    }
    freeaddrinfo(bind_address);

    printf("Listening...\n");
    if (listen(socket_listen, 10) < 0) {
        fprintf(stderr, "listen() failed. (%d)\n", GETSOCKETERRNO());
        return 1;
    }
```

If the preceding code isn't familiar, please refer to [Chapter 3](eb2e080f-fad2-4e14-b4c1-9a6c124df77c.xhtml), *An In-Depth Overview of TCP Connections*.

Note that the preceding code sets the listening port number to `8080`. The standard port for HTTPS is `443`. It is often more convenient to test with a high port number, since low port numbers require special privileges on some operating systems.

Our server uses a `while` loop to accept multiple connections. Note that this isn't true multiplexing, as only one connection is handled at a time. However, it proves convenient for testing purposes to be able to handle multiple connections serially. Our self-signed certificate causes mainstream browsers to reject our connection on the first try. The connection only succeeds after an exception is added. By having our code loop, it makes adding this exception easier.

Our `while` loop begins by using `accept()` to wait for new connections. This is done by the following code:

```cpp
/*tls_time_server.c continued*/

 while (1) {

        printf("Waiting for connection...\n");
        struct sockaddr_storage client_address;
        socklen_t client_len = sizeof(client_address);
        SOCKET socket_client = accept(socket_listen,
                (struct sockaddr*) &client_address, &client_len);
        if (!ISVALIDSOCKET(socket_client)) {
            fprintf(stderr, "accept() failed. (%d)\n", GETSOCKETERRNO());
            return 1;
        }
```

Once the connection is accepted, we use `getnameinfo()` to print out the client's address. This is sometimes useful for debugging purposes. The following code does this:

```cpp
/*tls_time_server.c continued*/

        printf("Client is connected... ");
        char address_buffer[100];
        getnameinfo((struct sockaddr*)&client_address,
                client_len, address_buffer, sizeof(address_buffer), 0, 0,
                NI_NUMERICHOST);
        printf("%s\n", address_buffer);
```

Once the TCP connection is established, an `SSL` object needs to be created. This is done with a call to `SSL_new()`, as shown by the following code:

```cpp
/*tls_time_server.c continued*/

        SSL *ssl = SSL_new(ctx);
        if (!ctx) {
            fprintf(stderr, "SSL_new() failed.\n");
            return 1;
        }
```

The SSL object is associated with the open socket by a call to `SSL_set_fd()`. Then, a TLS/SSL connection can be initialized with a call to `SSL_accept()`. The following code shows this:

```cpp
/*tls_time_server.c continued*/

        SSL_set_fd(ssl, socket_client);
        if (SSL_accept(ssl) <= 0) {
            fprintf(stderr, "SSL_accept() failed.\n");
            ERR_print_errors_fp(stderr);

            SSL_shutdown(ssl);
            CLOSESOCKET(socket_client);
            SSL_free(ssl);

            continue;
        }

        printf ("SSL connection using %s\n", SSL_get_cipher(ssl));
```

In the preceding code, the call to the `SSL_accept()` function can fail for many reasons. For example, if the connected client doesn't trust our certificate, or the client and server can't agree on a cipher suite, then the call to `SSL_accept()` fails. When it fails, we just clean up the allocated resources and use `continue` to repeat our listening loop.

Once the TCP and TLS/SSL connections are fully open, we use `SSL_read()` to receive the client's request. Our program ignores the content of this request. This is because our program only serves the time. It doesn't matter what the client has asked for—our server responds with the time.

The following code uses `SSL_read()` to wait on and read the client's request:

```cpp
/*tls_time_server.c continued*/

        printf("Reading request...\n");
        char request[1024];
        int bytes_received = SSL_read(ssl, request, 1024);
        printf("Received %d bytes.\n", bytes_received);
```

The following code uses `SSL_write()` to transmit the HTTP headers to the client:

```cpp
/*tls_time_server.c continued*/

        printf("Sending response...\n");
        const char *response =
            "HTTP/1.1 200 OK\r\n"
            "Connection: close\r\n"
            "Content-Type: text/plain\r\n\r\n"
            "Local time is: ";
        int bytes_sent = SSL_write(ssl, response, strlen(response));
        printf("Sent %d of %d bytes.\n", bytes_sent, (int)strlen(response));
```

The `time()` and `ctime()` functions are then used to format the current time. Once the time is formatted in `time_msg`, it is also sent to the client using `SSL_write()`. The following code shows this:

```cpp
/*tls_time_server.c continued*/

        time_t timer;
        time(&timer);
        char *time_msg = ctime(&timer);
        bytes_sent = SSL_write(ssl, time_msg, strlen(time_msg));
        printf("Sent %d of %d bytes.\n", bytes_sent, (int)strlen(time_msg));
```

Finally, after the data is transmitted to the client, the connection is closed, and the loop repeats. The following code shows this:

```cpp
/*tls_time_server.c continued*/

        printf("Closing connection...\n");
        SSL_shutdown(ssl);
        CLOSESOCKET(socket_client);
        SSL_free(ssl);
    }
```

If the loop terminates, it would be useful to close the listening socket and clean up the SSL context, as demonstrated by the following code:

```cpp
/*tls_time_server.c continued*/

    printf("Closing listening socket...\n");
    CLOSESOCKET(socket_listen);
    SSL_CTX_free(ctx);
```

Finally, Winsock should be cleaned up if necessary:

```cpp
/*tls_time_server.c continued*/

#if defined(_WIN32)
    WSACleanup();
#endif

    printf("Finished.\n");

    return 0;
}
```

That concludes `tls_time_server.c`.

You can compile and run the program using the following commands on macOS or Linux:

```cpp
gcc tls_time_server.c -o tls_time_server -lssl -lcrypto
./tls_time_server
```

On Windows, compiling and running the program is done with the following commands:

```cpp
gcc tls_time_server.c -o tls_time_server.exe -lssl -lcrypto -lws2_32
tls_time_server
```

If you have linker errors, please be sure that the OpenSSL library is installed correctly. You may find it helpful to attempt to compile `openssl_version.c` from [Chapter 9](47ba170d-42d9-4e38-b5d8-89503e005708.xhtml), *Loading Secure Web Pages with HTTPS and OpenSSL*.

The following screenshot shows what running `tls_time_server` might look like:

![](img/bb118df5-9e0b-4848-8b77-0b8834503033.png)

You can connect to the time server by navigating your web browser to `https://127.0.0.1:8080`. Upon the first connection, your browser will reject the self-signed certificate. The following screenshot shows what this rejection looks like in Firefox:

![](img/fdfb5821-6712-48de-85c3-4b37b22f8771.png)

To access the time server, you need to add an exception in the browser. The method for this is different in each browser, but generally, there is an Advanced button that leads to an option to either add a certificate exception or otherwise proceed with the insecure connection.

Once the browser connection is established, you will be able to see the current time as given by our `tls_time_server` program:

![](img/f10317c3-ad58-4cd4-86b5-a4c0ad3a66da.png)

The `tls_time_server` program proved useful to show how a TLS/SSL server can be set up without getting bogged down in the details of actualizing a complete HTTPS server. However, this chapter's code repository also includes a more substantial HTTPS server.

# A full HTTPS server

Included in this chapter's code repository is `https_server.c`. This program is a modification of `web_server.c` from [Chapter 7](f352830e-089c-4369-b7a2-18a896e1c5d5.xhtml), *Building a Simple Web Server*. It can be used to serve a simple static website over HTTPS.

In the `https_server.c` program, the basic TLS/SSL connection is set up and established the same way as shown in `tls_time_server.c`. Once the secure connection is established, the connection is simply treated as HTTP.

`https_server` is compiled using the same technique as for `tls_time_server`. The following screenshot shows how to compile and run `https_server`:

![](img/868a232a-215d-409a-8c9e-8c00f442ab5b.png)

Once `https_server` is running, you can connect to it by navigating your web browser to `https://127.0.0.1:8080`. You will likely need to add a security exception when connecting for the first time. The code is set up to serve the static pages from the `chap07` directory.

The following screenshot was taken of a web browser connected to `https_server`:

![](img/75eb6b72-dd04-495d-9bce-edcb588a7883.png)

This chapter's example programs illustrate the basics of HTTPS servers. However, implementing a genuinely robust HTTPS server does involve additional challenges. Let's consider some of these now.

# HTTPS server challenges

This chapter should serve only as an introduction to TLS/SSL server programming. There is much more to learn about secure network programming. Before deploying a secure HTTPS server with OpenSSL, it is essential to review all the OpenSSL documentation carefully. Many OpenSSL functions have edge cases that were ignored in the illustrative code for this chapter.

Multiplexing can also be complicated with OpenSSL. In typical TCP servers, we have been using the `select()` function to indicate when data is available to be read. The `select()` function works directly on the TCP socket. Using `select()` on a server secured with TLS/SSL can be tricky. This is because `select()` indicates when data is available at the TCP level. This usually, but not always, indicates that data is available to be read with `SSL_read()`. It is important that you carefully consult the OpenSSL documentation for `SSL_read()` if you are going to use it with `select()`. The example program in this chapter ignores these possibilities for reasons of simplicity.

There are also alternatives to OpenSSL. Let's consider some of them now.

# OpenSSL alternatives

Although **OpenSSL** is one of the oldest and most widely deployed libraries implementing TLS, many alternative libraries have sprung up in recent years. Some of these alternatives aim to offer better features, performance, or quality control compared to OpenSSL.

The following table contains a number of alternative open source TLS libraries:

| **TLS Library** | **Website** |
| cryptlib | [https://www.cryptlib.com/](https://www.cryptlib.com/) |
| GnuTLS | [https://www.gnutls.org/](https://www.gnutls.org/) |
| LibreSSL | [https://www.libressl.org/](https://www.libressl.org/) |
| mbed TLS | [https://tls.mbed.org/](https://tls.mbed.org/) |
| Network Security Services | [https://developer.mozilla.org/en-US/docs/Mozilla/Projects/NSS](https://developer.mozilla.org/en-US/docs/Mozilla/Projects/NSS) |
| s2n | [https://github.com/awslabs/s2n](https://github.com/awslabs/s2n) |
| wolfSSL | [https://www.wolfssl.com/](https://www.wolfssl.com/) |

There are also alternatives to doing TLS termination directly in your application, and this can simplify program design. Let's consider this next.

# Alternatives to TLS

Getting everything right when implementing an HTTPS server can prove difficult, and missing even a minor detail can compromise security entirely.

As an alternative to using TLS directly in our server itself, it is sometimes a better idea to use a **reverse proxy server**. A reverse proxy server can be configured to accept secure connections from clients and then forward these connections as plain HTTP to your program.

**Nginx** and **Apache** are two popular open source servers that can work well as HTTPS reverse proxies.

This setup is illustrated by the following diagram:

![](img/3102f763-8b15-4623-bdff-b12ea8e8bd88.png)

A reverse proxy server configured in this way is also called a **TLS termination proxy**.

An even better alternative may be to create your program using the **CGI** or **FastCGI** standard. In this case, your program communicates directly with a standard web server. The web server handles all of the HTTPS and HTTP details. This can greatly simplify program design, and, in some cases, reduce maintenance costs too.

If you do use an off-the-shelf HTTPS server, it is still important to use caution. It can be easy to inadvertently compromise security through misconfiguration.

# Summary

In this chapter, we considered the HTTPS protocol from the server's perspective. We covered how certificates work, and we showed the method for generating a self-signed certificate with OpenSSL.

Once we had a certificate, we learned how to use the OpenSSL library to listen for TLS/SSL connections. We used this knowledge to implement a simple server that displays the current time over HTTPS.

We also discussed some of the pitfalls and complexity of implementing HTTPS servers. Many applications may benefit from side-stepping the implementation of HTTPS and relying on a reverse proxy instead.

In the next chapter, [Chapter 11](c9d0a1dc-878b-4961-825e-65688fac08ae.xhtml), *Establishing SSH Connections with libssh*, we will look at another secure protocol, **Secure Shell** (**SSH**).

# Questions

Try these questions to test the knowledge you have acquired from this chapter:

1.  How does a client decide whether it should trust a server's certificate?
2.  What is the main issue with self-signed certificates?
3.  What can cause `SSL_accept()` to fail?
4.  Can `select()` be used to multiplex connections for HTTPS servers?

The answers to these questions can be found in [Appendix A](bd8b8f52-52cb-4d34-b01b-e907564bfece.xhtml), *Answers to Questions*.

# Further reading

For more information about HTTPS and OpenSSL, please refer to the following:

*   OpenSSL documentation ([https://www.openssl.org/docs/](https://www.openssl.org/docs/))
*   **RFC 5246**: *The **Transport Layer Security** (**TLS**) Protocol Version 1.2* ([https://tools.ietf.org/html/rfc5246](https://tools.ietf.org/html/rfc5246))
*   *Let's Encrypt* ([https://letsencrypt.org/](https://letsencrypt.org/))