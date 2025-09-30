# Example Programs

This book's code repository, located at [https://github.com/codeplea/hands-on-network-programming-with-c](https://github.com/codeplea/hands-on-network-programming-with-c),  includes 44 example programs. These programs are explained in detail throughout the book.

# Code license

The example programs provided in this book's code repository are released under the MIT license, the text of which follows:

```cpp
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to
whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

```

# Code included with this book

The following is a list of the 44 example programs included with this book, by chapter.

# Chapter 1 – Introducing Networks and Protocols

This chapter includes the following example programs:

*   `win_init.c`: Example code to initialize Winsock (Windows only)
*   `win_list.c`: Lists all local IP addresses (Windows only)
*   `unix_list.c`: Lists all local IP addresses (Linux and macOS only)

# Chapter 2 – Getting to Grips with Socket APIs

This chapter includes the following example programs:

*   `sock_init.c`: Example program to include all necessary headers and initialize
*   `time_console.c`: Prints the current date and time to the console
*   `time_server.c`: Serves a web page giving the current date and time
*   `time_server_ipv6.c`: The same as the preceding code but listens for IPv6 connections
*   `time_server_dual.c`: The same as the preceding code but listens for IPv6/IPv4 dual-stack connections

# Chapter 3 – An In-Depth Overview of TCP Connections

This chapter includes the following example programs:

*   `tcp_client.c`: Establishes a TCP connection and sends/receives data from the console.
*   `tcp_serve_toupper.c`: A TCP server servicing multiple connections using `select()`. Echoes received data back to the client all in uppercase.

*   `tcp_serve_toupper_fork.c`: The same as the preceding code but uses `fork()` instead of `select()`. (Linux and macOS only.)
*   `tcp_serve_chat.c`: A TCP server that relays received data to every other connected client.

# Chapter 4 – Establishing UDP Connections

This chapter includes the following example programs:

*   `udp_client.c`: Sends/receives UDP data from the console
*   `udp_recvfrom.c`: Uses `recvfrom()` to receive one UDP datagram
*   `udp_sendto.c`: Uses `sendto()` to send one UDP datagram
*   `udp_serve_toupper.c`: Listens for UDP data and echoes it back to the sender all in uppercase
*   `udp_serve_toupper_simple.c`: The same as the preceding code but doesn't use `select()`

# Chapter 5 – Hostname Resolution and DNS

This chapter includes the following example programs:

*   `lookup.c`: Uses `getaddrinfo()` to look up addresses for a given hostname
*   `dns_query.c`: Encodes and sends a UDP DNS query, then listens for and decodes the response

# Chapter 6 – Building a Simple Web Client

This chapter includes the following example program:

*   `web_get.c`: A minimal HTTP client to download a web resource from a given URL

# Chapter 7 – Building a Simple Web Server

This chapter includes the following example programs:

*   `web_server.c`: A minimal web server capable of serving a static website
*   `web_server2.c`: A minimal web server (no globals)

# Chapter 8 – Making Your Program Send Email

This chapter includes the following example program:

*   `smtp_send.c`: A simple program to transmit an email

# Chapter 9 – Loading Secure Web Pages with HTTPS and OpenSSL

The examples in this chapter use OpenSSL. Be sure to link to the OpenSSL libraries when compiling (`-lssl -lcrypto`):

*   `openssl_version.c`: A program to report the installed OpenSSL version
*   `https_simple.c`: A minimal program that requests a web page using HTTPS
*   `https_get.c`: The HTTP client of [Chapter 6](de3d2e9b-b94e-47d1-872c-c2ecb34c4026.xhtml), *Building a Simple Web Client*, modified to use HTTPS
*   `tls_client.c`: The TCP client program of [Chapter 3](eb2e080f-fad2-4e14-b4c1-9a6c124df77c.xhtml), *An In-Depth Overview of TCP Connections*, modified to use TLS
*   `tls_get_cert.c`: Prints a certificate from a TLS server

# Chapter 10 – Implementing a Secure Web Server

The examples in this chapter use OpenSSL. Be sure to link to the OpenSSL libraries when compiling (`-lssl -lcrypto`):

*   `tls_time_server.c`: The time server of [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs*, modified to use HTTPS
*   `https_server.c`: The web server of [Chapter 7](f352830e-089c-4369-b7a2-18a896e1c5d5.xhtml), *Building a Simple Web Server*, modified to use HTTPS

# Chapter 11 – Establishing SSH Connections with libssh

The examples in this chapter use `libssh`. Be sure to link to the `libssh` libraries when compiling (`-lssh`):

*   `ssh_version.c`: A program to report the `libssh` version
*   `ssh_connect.c`: A minimal client that establishes an SSH connection
*   `ssh_auth.c`: A client that attempts SSH client authentication using a password
*   `ssh_command.c`: A client that executes a single remote command over SSH
*   `ssh_download.c`: A client that downloads a file over SSH/SCP

# Chapter 12 – Network Monitoring and Security

This chapter doesn't include any example programs.

# Chapter 13 – Socket Programming Tips and Pitfalls

This chapter includes the following example programs:

*   `connect_timeout.c`: Shows how to time out a `connect()` call early.
*   `connect_blocking.c`: For comparison with `connect_timeout.c`.
*   `server_reuse.c`: Demonstrates the use of `SO_REUSEADDR`.
*   `server_noreuse.c`: For comparison with `server_reuse.c`.
*   `server_crash.c`: This server purposefully writes to a TCP socket after the client disconnects.
*   `error_text.c`: Shows how to obtain error code descriptions.
*   `big_send.c:` A TCP client that sends lots of data after connecting. Used to show the blocking behavior of `send()`.
*   `server_ignore.c`: A TCP server that accepts connections, then simply ignores them. Used to show the blocking behavior of `send()`.
*   `setsize.c`: Shows the maximum number of sockets `select()` can handle.

# Chapter 14 – Web Programming for the Internet of Things

This chapter doesn't include any example programs.