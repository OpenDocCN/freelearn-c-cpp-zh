# Answers to Questions

# Chapter 1, Introducing Networks and Protocols

1.  **What are the key differences between IPv4 and IPv6?**

IPv4 only supports 4 billion unique addresses, and because they were allocated inefficiently, we are now running out. IPv6 supports 3.4 x 10^(38) possible addresses. IPv6 provides many other improvements, but this is the one that affects our network programming directly.

2.  **Are the IP addresses given by the `ipconfig` and `ifconfig` commands the same IP addresses that a remote web server sees if you connect to it?**

Sometimes, these addresses will match, but not always. If you're on a private IPv4 network, then your router likely performs network address translation. The remote web server then sees the translated address.
If you have a publicly routable IPv4 or IPv6 address, then the address seen by the remote web server will match those reported by `ipconfig` and `ifconfig`.

3.  **What is the IPv4 loopback address?**

The IPv4 loopback address is `127.0.0.1`, and it allows networked programs to communicate with each other while executing on the same machine.

4.  **What is the IPv6 loopback address?**

The IPv6 loopback address is `::1`. It works in the same way as the IPv4 loopback address.

5.  **How are domain names (for example, `example.com`) resolved into IP addresses?**

DNS is used to resolve domain names into IP addresses. This protocol is covered in detail in [Chapter 5](3d80e3b8-07d3-49f4-b60f-b006a17f7213.xhtml), *Hostname Resolution and DNS*.

6.  **How can you find your public IP address?**

The easiest way is to visit a website that reports it for you.

7.  **How does an operating system know which application is responsible for an incoming packet?**

Each IP packet has a local address, remote address, local port number, remote port number, and protocol type. These five attributes are memorized by the operating system to determine which application should handle any given incoming packet.

# Chapter 2, Getting to Grips with Socket APIs

1.  **What is a socket?**

A socket is an abstraction that represents one endpoint of a communication link between systems.

2.  **What is a connectionless protocol? What is a connection-oriented protocol?**

A connection-oriented protocol sends data packets in the context of a larger stream of data. A connectionless protocol sends each packet of data independently of any before or after it.

3.  **Is UDP a connectionless or connection-oriented protocol?**

UDP is considered a connectionless protocol. Each message is sent independently of any before or after it.

4.  **Is TCP a connectionless or connection-oriented protocol?**

TCP is considered a connection-oriented protocol. Data is sent and received in order as a stream.

5.  **What types of applications generally benefit from using the UDP protocol?**

UDP applications benefit from better real-time performance while sacrificing reliability. They are also able to take advantage of IP multicasting.

6.  **What types of applications generally benefit from using the TCP protocol?**

Applications that need a reliable stream of data transfer benefit from the TCP protocol.

7.  **Does TCP guarantee that data will be transmitted successfully?**

TCP makes some guarantees about reliability, but nothing can truly guarantee that data is transmitted successfully. For example, if someone unplugs your modem, no protocol can overcome that.

8.  **What are some of the main differences between Berkeley sockets and Winsock sockets?**

The header files are different. Sockets themselves are represented as signed versus unsigned `ints`. When `socket()` or `accept()` calls fail, the return values are different. Berkeley sockets are also standard file descriptions. This isn't always true with Winsock. Error codes are different and retrieved in a different way. There are additional differences, but these are the main ones that affect our programs.

9.  **What does the `bind()` function do? **

The `bind()` function associates a socket with a particular local network address and port number. Its usage is almost always required for the server, and it's usually not required for the client.

10.  **What does the `accept()` function do?**

The `accept()` function will block until a new TCP client has connected. It then returns the socket for this new connection.

11.  **In a TCP connection, does the client or the server send application data first?**

Either the client or the server can send data first. They can even send data simultaneously. In practice, many client-server protocols (such as HTTP) work by having the client send a request first and then having the server send a response.

# Chapter 3, An In-Depth Overview of TCP Connections

1.  **How can we tell if the next call to `recv()` will block?**

We use the `select()` function to indicate which sockets are ready to be read from without blocking.

2.  **How can you ensure that `select()` doesn't block for longer than a specified tim**e?

You can pass `select()` a timeout parameter.

3.  **When we used our `tcp_client` program to connect to a web server, why did we need to send a blank line before the web server responded?**

HTTP, the web server's protocol, expects a blank line to indicate the end of the request. Without this blank line, it wouldn't know if the client was going to keep sending additional request headers.

4.  **Does `send()` ever block?**

Yes. You can use `select()` to determine when a socket is ready to be written to without blocking. Alternatively, sockets can be put into non-blocking mode. See [Chapter 13](11c5bb82-e55f-4977-bf7f-5dbe791fde92.xhtml), *Socket Programming Tips **and Pitfalls*, for more information.

5.  **How can we tell if a socket has been disconnected by our peer?**

The return value of `recv()` can indicate if a socket has been disconnected.

6.  **Is data received by `recv()` always the same size as data sent with `send()`?**

No. TCP is a stream protocol. There is no way to tell if the data returned from one `recv()` call was sent with one or many calls to `send()`.

7.  Consider this code:

```cpp
recv(socket_peer, buffer, 4096, 0);
printf(buffer);
```

What is wrong with it?
Also see what is wrong with this code:

```cpp
recv(socket_peer, buffer, 4096, 0);
printf("%s", buffer);
```

The data returned by `recv()` is not null terminated! Both of the preceding code excerpts will likely cause `printf()` to read past the end of the data returned by `recv()`. Additionally, in the first code example the data received could contain format specifiers (for example `%d`), which would cause additional memory access violations.

# Chapter 4, Establishing UDP Connections

1.  **How do `sendto()` and `recvfrom()` differ from `send()` and `recv()`?**

The `send()` and `recv()` functions are useful after calling `connect()`. They only work with the one remote address that was passed to `connect()`. The `sendto()` and `recvfrom()` functions can be used with multiple remote addresses.

2.  **Can `send()` and `recv()` be used on UDP sockets?**

Yes. The `connect()` function should be called first in that case. However, the `sendto()` and `recvfrom()` functions are often more useful for UDP sockets.

3.  **What does `connect()` do in the case of a UDP socket?**

The `connect()` function associates the socket with a remote address.

4.  **What makes multiplexing with UDP easier than with TCP?**

One UDP socket can talk to multiple remote peers. For TCP, one socket is needed for each peer.

5.  **What are the downsides to UDP when compared to TCP?**

UDP does not attempt to fix many of the errors that TCP does. For example, TCP ensures that data arrives in the same order it was sent, TCP tries to avoid causing network congestion, and TCP attempts to resend lost packets. UDP does none of this.

6.  **Can the same program use UDP and TCP?**

Yes. It just needs to create sockets for both.

# Chapter 5, Hostname Resolution and DNS

1.  **Which function fills in an address needed for socket programming in a portable and protocol-independent way?**

`getaddrinfo()` is the function to use for this.

2.  **Which socket programming function can be used to convert an IP address back into a name?**

`getnameinfo()` can be used to convert addresses back to names.

3.  **A DNS query converts a name to an address, and a reverse DNS query converts an address back into a name. If you run a DNS query on a name, and then a reverse DNS query on the resulting address, do you always get back the name you started with?**

Sometimes, you will get the same name back but not always. This is because the forward and reverse lookups use independent records. It's also possible to have many names point to one address, but that one address can only have one record that points back to a single name.

4.  **What are the DNS record types used to return IPv4 and IPv6 addresses for a name?**

The `A` record type returns an IPv4 address, and the `AAAA` record type returns an IPv6 address.

5.  **Which DNS record type stores special information about email servers?**

The `MX` record type is used to return email server information.

6.  **Does `getaddrinfo()` always return immediately? Or can it block?**

If `getaddrinfo()` is doing name lookups, it will often block. In the worst-case scenario, many UDP messages would need to be sent to various DNS servers, so this can be a noticeable delay. This is one reason why DNS caching is important.

If you are simply using `getaddrinfo()` to convert from a text IP address, then it shouldn't block.

7.  **What happens when a DNS response is too large to fit into a single UDP packet?**

The DNS response will have the `TC` bit set in its header. This indicates that the message was truncated. The query should be resent using TCP.

# Chapter 6, Building a Simple Web Client

1.  **Does HTTP use TCP or UDP?**

HTTP runs over TCP port `80`.

2.  **What types of resources can be sent over HTTP?**

HTTP can be used to transfer essentially any computer file. It's commonly used for web pages (HTML) and the associated files (such as styles, scripts, fonts, and images).

3.  **What are the common HTTP request types?**

`GET`, `POST`, and `HEAD` are the most common HTTP request types.

4.  **What HTTP request type is typically used to send data from the server to the client?**

`GET` is the usual request type for a client to request a resource from the server.

5.  **What HTTP request type is typically used to send data from the client to the server?**

`POST` is used when the client needs to send data to the server.

6.  What are the two common methods used to determine an HTTP response body length?

The HTTP body length is commonly determined by the `Content-Length` header or by using `Transfer-Encoding: chunked`.

7.  **How is the HTTP request body formatted for a `POST`-type HTTP request?**

This is determined by the application. The client should set the `Content-Type` header to specify which format it is using. `application/x-www-form-urlencoded` and `application/json` are common values.

# Chapter 7, Building a Simple Web Server

1.  **How does an HTTP client indicate that it has finished sending the HTTP request?**

The HTTP request should end with a blank line.

2.  **How does an HTTP client know what type of content the HTTP server is sending?**

The HTTP server should identify the content with a `Content-Type` header.

3.  **How can an HTTP server identify a file's media type?**

A common method of identifying a file's media type is just to look at the file extension. The server is free to use other methods though. When sending dynamic pages or data from a database, there will be no file and therefore no file extension. In this case, the server must know the media type from its context.

4.  **How can you tell whether a file exists on the filesystem and is readable by your program? Is `fopen(filename, "r") != 0` a good test?**

This is not a trivial problem. A robust program will need to consider system specific APIs carefully. Windows uses special filenames that will trip up a program that relies only on `fopen()` to check for a file's existence.

# Chapter 8, Making Your Program Send Email

1.  **What port does SMTP operate on?**

SMTP does mail transmission over TCP port `25`. Many providers use alternative ports for mail submission.

2.  **How do you determine which SMTP server receives mail for a given domain?**

The mail servers responsible for receiving mail for a given domain are given by MX-type DNS records.

3.  **How do you determine which SMTP server sends mail for a given provider?**

It's not possible to determine that in the general case. In any case, several servers could be responsible. Sometimes these servers will be listed under a TXT-type DNS record using SPF, but that is certainly not universal.

4.  **Why won't an SMTP server relay mail without authentication?**

Open relay SMTP servers are targeted by spammers. SMTP servers require authentication to prevent abuse.

5.  **How are binary files sent as email attachments when SMTP is a text-based protocol?**

Binary files must be re-encoded as plain text. The most common method is with `Content-Transfer-Encoding: base64`.

# Chapter 9, Loading Secure Web Pages with HTTPS and OpenSSL

1.  **What port does HTTPS typically operate on?**

HTTPS connects over TCP port `443`.

2.  **How many keys does symmetric encryption use?**

Symmetric encryption uses one key. Data is encrypted and decrypted with the same key.

3.  **How many keys does asymmetric encryption use?**

Asymmetric encryption use two different, but mathematically related, keys. Data is encrypted with one and decrypted with the other.

4.  **Does TLS use symmetric or asymmetric encryption?**

TLS use both symmetric and asymmetric encryption algorithms to function.

5.  **What is the difference between SSL and TLS?**

TLS is the successor to SSL. SSL is now deprecated.

6.  **What purpose do certificates fulfill?**

Certificates allow a server or client to verify their identity.

# Chapter 10, Implementing a Secure Web Server

1.  **How does a client decide whether it should trust a server's certificate?**

There are various ways a client can trust a server's certificate. The chain-of-trust model is the most common. In this model, the client explicitly trusts an authority. The client then implicitly trusts any certificates it encounters that are signed by this trusted authority.

2.  **What is the main issue with self-signed certificates?**

Self-signed certificates aren't signed by a trusted certificate authority. Web browsers won't know to trust self-signed certificates unless the user adds a special exception.

3.  **What can cause `SSL_accept()` to fail?**

`SSL_accept()` fails if the client doesn't trust the server's certificate or if the client and server can't agree on a mutually supported protocol version and cipher suite.

4.  **Can `select()` be used to multiplex connections for HTTPS servers?**

Yes, but be aware that `select()` works on the underlying TCP connection layer, not on the TLS layer. Therefore, when `select()` indicates that a socket has data waiting, it does not necessarily mean that there is new TLS data ready.

# Chapter 11, Establishing SSH Connections with libssh

1.  **What is a significant downside of using Telnet?**

Essentially, Telnet provides no security features. Passwords are sent as plaintext.

2.  **Which port does SSH typically run on?**

SSH's official port is TCP port `22`. In practice, it is common to run SSH on arbitrary ports in an attempt to hide from attackers. With a properly secured server, these attackers are a nuisance rather than a legitimate threat.

3.  **Why is it essential that the client authenticates the SSH server?**

If the client doesn't verify the SSH server's identity, then it could be tricked into sending credentials to an impostor.

4.  **How is the server typically authenticated?**

SSH servers typically use certificates to identity themselves. This is similar to how servers are authenticated when using HTTPS.

5.  **How is the SSH client typically authenticated?**

It is still common for clients to authenticate with a password. The downside to this method is that if a client is somehow tricked into connecting to an impostor server, then their password will be compromised. SSH provides alternate methods, including authenticating clients using certificates, that aren't susceptible to replay attacks.

# Chapter 12, Network Monitoring and Security

1.  **Which tool would you use to test the reachability of a target system?**

The `ping` tool is useful to test reachability.

2.  **Which tool lists the routers to a destination system?**

The `traceroute` (`tracert` on Windows) tool will show the network path to a target system.

3.  **What are raw sockets used for?**

Raw sockets allow the programmer to specify directly what goes into a network packet. They provide lower-level access than TCP and UDP sockets, and can be used to implement additional protocols, such as ICMP.

4.  **Which tools list the open TCP sockets on your system?**

The netstat tool can be used to show open connections on your local system.

5.  **What is one of the biggest concerns with security for networked C programs?**

When programming networked applications in C, special care must be given to memory safety. Even a small mistake could allow an attacker to compromise your program.

# Chapter 13, Socket Programming Tips and Pitfalls

1.  **Is it ever acceptable just to terminate a program if a network error is detected?**

Yes. For some applications terminating on error is the right call. For more substantial applications, the ability to retry and continue on may be needed.

2.  **Which system functions are used to convert error codes into text descriptions?**

You can use `FormatMessage()` on Windows and `strerror()` on other platforms to obtain error messages.

3.  **How long does it take for a call to `connect()` to complete on a TCP socket?**

A call to `connect()` typically blocks for at least one network time round trip while the TCP three-way handshake is being completed.

4.  **What happens if you call `send()` on a disconnected TCP socket?**

On Unix-based systems, your program can receive a `SIGPIPE` signal. It is important to plan for that. Otherwise, `send()` returns `-1`.

5.  **How can you ensure that the next call to `send()` won't block?**

Either use `select()` to make sure the socket is ready for more data or use non-blocking sockets.

6.  **What happens if both peers to a TCP connection try to send a large amount of data simultaneously?**

If both sides to a TCP connection are calling `send()`, but not `recv()`, then they can be trapped in a deadlocked state. It is important to intersperse calls to `send()` with calls to `recv()`. The use of `select()` can help inform your program about what to do next.

7.  **Can you improve application performance by disabling the Nagle algorithm?**

It depends on what your application is doing. For real-time applications using TCP, disabling the Nagle algorithm is often a good trade-off for decreasing latency at the expense of bandwidth efficiency. For other applications, disabling it can decrease throughput, increase network congestion, and even increase latency.

8.  **How many connections can `select()` handle?**

It depends on your platform. It is defined in the `FD_SETSIZE` macro, which is easily increased on Windows but not on other platforms. Typically, the upper limit is around 1,024 sockets.

# Chapter 14, Web Programming for the Internet of Things

1.  **What are the drawbacks to using Wi-Fi connectivity?**

Wi-Fi can be difficult for end user setup. It's also not available everywhere.

2.  **What are the drawbacks to using Ethernet connectivity?**

Many devices aren't used in areas where wiring has been run.

3.  **What are the drawbacks to using cellular connectivity?**

Cellular connectivity is expensive. It can also have increased latency and larger power requirements when compared to other methods.

4.  **What are some advantages to using a single-board computer with embedded Linux? What are the drawbacks?**

Having access to a full operating system, such as Linux, can simplify software development. However, **Single-Board Computers** (**SBCs**) are relatively expensive and offer few board-level connectivity options and peripherals when compared to microcontrollers. They also require lots of power, relatively speaking.

5.  **What are some advantages to using a microcontroller in your IoT device?**

Many IoT devices will need to use a microcontroller to provide their basic functionality anyway. Microcontrollers are cheap, offer a wide range of peripherals, are able to meet real-time performance constraints, and can run on very little power.

6.  **Is the use of HTTPS always appropriate in IoT device communication?**

HTTPS is a decent way to secure IoT communication for most applications; however, it has a lot of processing and bandwidth overhead. Each application is unique, and the security scheme used should be chosen based on your exact needs.