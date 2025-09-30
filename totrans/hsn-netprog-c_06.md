# Hostname Resolution and DNS

Hostname resolution is a vital part of network programming. It allows us to use simple names, such as `www.example.com`, instead of tedious addresses such as `::ffff:192.168.212.115`. The mechanism that allows us to resolve hostnames into IP addresses and IP addresses into hostnames is the **Domain Name System** (**DNS**).

In this chapter, we begin by covering the built-in `getaddrinfo()` and `getnameinfo()` socket functions in more depth. Later, we will build a program that does DNS queries using **User Datagram Protocol** (**UDP**) from scratch.

We will cover the following topics in this chapter:

*   How DNS works
*   Common DNS record types
*   The `getaddrinfo()` and `getnameinfo()` functions
*   DNS query data structures
*   DNS UDP protocol
*   DNS TCP fallback
*   Implementing a DNS query program

# Technical requirements

The example programs from this chapter can be compiled with any modern C compiler. We recommend MinGW on Windows and GCC on Linux and macOS. See Appendices B, C, and D for compiler setup.

The code for this book can be found at [https://github.com/codeplea/Hands-On-Network-Programming-with-C](https://github.com/codeplea/Hands-On-Network-Programming-with-C).

From the command line, you can download the code for this chapter with the following command:

```cpp
git clone https://github.com/codeplea/Hands-On-Network-Programming-with-C
cd Hands-On-Network-Programming-with-C/chap05
```

Each example program in this chapter runs on Windows, Linux, and macOS. When compiling on Windows, each example program should be linked with the Winsock library. This can be accomplished by passing the `-lws2_32` option to `gcc`.

We'll provide the exact commands needed to compile each example as they are introduced.

All of the example programs in this chapter require the same header files and C macros which we developed in [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs.* For brevity, we put these statements in a separate header file, `chap05.h`, which we can include in each program. For an explanation of these statements, please refer to [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs*.

The contents of `chap05.h` are as follows:

```cpp
/*chap05.h*/

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

With the `chap05.h` header in place, writing portable network programs is much easier. Let's continue now with an explanation of how DNS works, and then we will move on to the actual example programs.

# How hostname resolution works

The DNS is used to assign names to computers and systems connected to the internet. Similar to how a phone book can be used to link a phone number to a name, the DNS allows us to link a hostname to an IP address.

When your program needs to connect to a remote computer, such as `www.example.com`, it first needs to find the IP address for `www.example.com`. In this book so far, we have been using the built-in `getaddrinfo()` function for this purpose. When you call `getaddrinfo()`, your operating system goes through a number of steps to resolve the domain name.

First, your operating system checks whether it already knows the IP address for `www.example.com`. If you have used that hostname recently, the OS is allowed to remember it in a local cache for a time. This time is referred to as **time-to-live** (**TTL**) and is set by the DNS server responsible for the hostname.

If the hostname is not found in the local cache, then your operating system will need to query a DNS server. This DNS server is usually provided by your **Internet Service Provider** (**ISP**), but there are also many publicly-available DNS servers. When the DNS server receives a query, it also checks its local cache. This is useful because numerous systems may be relying on one DNS server. If a DNS server receives 1,000 requests for `gmail.com` in one minute, it only needs to resolve the hostname the first time. For the other 999 requests, it can simply return the memorized answer from its local cache.

If the DNS server doesn't have the requested DNS record in its cache, then it needs to query other DNS servers until it connects directly to the DNS server responsible for the target system. Here's an example query resolution broken down step-wise:

Client A's DNS server is trying to resolve `www.example.com` as follows:

1.  It first connects to a root DNS server and asks for `www.example.com`.
2.  The root DNS server directs it to ask the `.com` server.
3.  Our client's DNS server then connects to the server responsible for `.com` and asks for `www.example.com`.
4.  The `.com` DNS server gives our server the address of another server – the `example.com` DNS server.
5.  Our DNS server finally connects to that server and asks about the record for `www.example.com`.
6.  The `example.com` server then shares the address of `www.example.com`.
7.  Our client's DNS server relays it back to our client.

The following diagram illustrates this visually:

![](img/3885e0b3-c552-4775-926d-4cc361781b2a.png)

In this example, you can see that resolving `www.example.com` involved sending eight messages. It's possible that the lookup could have taken even longer. This is why it's imperative for DNS servers to implement caching. Let's assume that **Client B** tries the same query shortly after **Client A**; it's likely that the DNS server would have that value cached:

![](img/ea2e3e19-987a-4c09-80af-f27720d6ad66.png)

Of course, if a program on **Client A** wants to resolve `www.example.com` again, it's likely that it won't have to contact the DNS server at all – the operating system running on **Client A** should have cached the result.

On Windows, you can show your local DNS cache with the following command:

```cpp
ipconfig /displaydns
```

On Linux or macOS, the command to show your local DNS varies depending on your exact system setup.

One downside of setting a large TTL value for a domain record is that you have to wait at least that long to be sure that all clients are using the new record and not just retrieving the old cached record.

In addition to DNS records that link a hostname to an IP address, there are other DNS record types for various purposes. We'll review some of these in the next section.

# DNS record types

The DNS has five main types of records—`A`, `AAAA`, `MX`, `TXT`, `CNAME`, and `*` (`ALL`/`ANY`).

As we have learned, the DNS's primary purpose is to translate hostnames into IP addresses. This is done with two record types – type `A` and type `AAAA`. These records work in the same way, but `A` records return an IPv4 address, while `AAAA` records return an IPv6 address.

The `MX` record type is used to return mail server information. For example, if you wanted to send an email to `larry@example.com`, then the `MX` record(s) for `example.com` would indicate which mail server(s) receives emails for that domain.

`TXT` records can be used to store arbitrary information for a hostname. In practice, these are sometimes set to prove ownership of a domain name or to publish email sending guidelines. The **Sender Policy Framework** (**SPF**) standard uses `TXT` records to declare which systems are allowed to send mail for a given domain name. You can read more about SPF at [http://www.openspf.org/](http://www.openspf.org/).

`CNAME` records can be used to provide an alias for a given name. For example, many websites are accessible at both their root domain name, for example, `example.com`, and at the `www` subdomain. If `example.com` and `www.example.com` should point to the same address, then an `A` and an `AAAA` record can be added for `example.com`, while a `CNAME` record can be added for `www.example.com` pointing to `example.com`. Note that DNS clients don't query for `CNAME` records directly; instead, a client would ask for the `A` or `AAAA` record for `www.example.com` and the DNS server would reply with the `CNAME` record pointing to `example.com`. The DNS client would then continue the query using `example.com`.

When doing a DNS query, there is also a pseudo-record type called `*` or `ALL` or `ANY`. If this record is requested from a DNS server, then the DNS server returns all known record types for the current query. Note that a DNS server is allowed to respond with only the records in its cache, and this query is not guaranteed (or even likely) to actually get all of the records for the requested domain.

When sending a DNS query, each record type has an associated type ID. The IDs for the records discussed so far are as follows:

| **Record Type** | **Type ID (decimal)** | **Description** |
| `A` | 1 | IPv4 address record |
| `AAAA` | 28 | IPv6 address record |
| `MX` | 15 | Mail exchange record |
| `TXT` | 16 | Text record |
| `CNAME` | 5 | Canonical name |
| `*` | 255 | All cached records |

There are many other record types in use. Please see the *Further reading* section at the end of this chapter for more information.

It should be noted that one hostname may be associated with multiple records of the same type. For example, `example.com` could have several `A` records, each with a different IPv4 address. This is useful if multiple servers can provide the same service.

One other aspect of the DNS protocol worth mentioning is security. Let's look at that now.

# DNS security

While most web traffic and email are encrypted today, DNS is still widely used in an unsecured manner. Protocols do exist to provide security for DNS, but they are not widely adopted yet. This will hopefully change in the near future.

**Domain Name System Security Extensions** (**DNSSEC**) are DNS extensions that provide data authentication. This authentication allows a DNS client to know that a given DNS reply is authentic, but it does not protect against eavesdropping.

**DNS over HTTPS** (**DoH**) is a protocol that provides name resolution over HTTPS. HTTPS provides strong security guarantees, including resistance to interception. We cover HTTPS in [Chapter 9](47ba170d-42d9-4e38-b5d8-89503e005708.xhtml), *Loading Secure Web Pages with HTTPS and OpenSSL*, and [Chapter 10](f57ffaa2-3eba-45cf-914b-bb6aef36174f.xhtml), *Implementing a Secure Web Server*.

What are the implications of using insecure DNS? First, if DNS is not authenticated, then it could allow an attacker to lie about a domain name's IP address. This could trick a victim into connecting to a server that they think is `example.com`, but, in reality, is a malicious server controlled by the attacker at a different IP address. If the user is connecting via a secure protocol, such as HTTPS, then this attack will fail. HTTPS provides additional authentication to prove server identity. However, if the user connects with an insecure protocol, such as HTTP, then the DNS attack could successfully trick the victim into connecting to the wrong server.

If DNS is authenticated, then these hijacking attacks are prevented. However, without encryption, DNS queries are still susceptible to eavesdropping. This could potentially give an eavesdropper insight into which websites you're visiting and other servers that you are connecting to (for example, which email server you use). This doesn't let an attacker know what you are doing on each website. For example, if you do a DNS query for `example.com`, an attacker would know that you planned to visit `example.com`, but the attacker would not be able to determine which resources you requested from `example.com` – assuming that you use a secure protocol (for example, HTTPS) to retrieve those resources. An attacker with the ability to eavesdrop would be able to see that you established a connection to the IP address of `example.com` in any case, so them knowing that you performed a DNS lookup beforehand is not much extra information.

With some security discussion out of the way, let's look at how to do actual DNS lookups. Winsock and Berkeley sockets provide a simple function to do address lookup, called `getaddrinfo()`, which we've used in the previous chapters of this book. We will start with this in the next section.

# Name/address translation functions

It is common for networked programs to need to translate text-based representatives of an address or hostname into an address structure required by the socket programming API. The common function we've been using is `getaddrinfo()`. It is a useful function because it is highly portable (available on Windows, Linux, and macOS), and it works for both IPv4 and IPv6 addresses.

It is also common to need to convert a binary address back into a text format. We use `getnameinfo()` for this.

# Using getaddrinfo()

Although we've been using `getaddrinfo()` in previous chapters, we'll discuss it in more detail here.

The declaration for `getaddrinfo()` is shown in the following code:

```cpp
int getaddrinfo(const char *node,
                const char *service,
                const struct addrinfo *hints,
                struct addrinfo **res);
```

The preceding code snippet is explained as follows:

*   `node` specifies a hostname or address as a string. Valid examples could be `example.com`, `192.168.1.1`, or `::1`.
*   `service` specifies a service or port number as a string. Valid examples could be `http` or `80`. Alternately, the null pointer can be passed in for `service`, in which case, the resulting address is set to port `0`.

*   `hints` is a pointer to a `struct addrinfo`, which specifies options for selecting the address. The `addrinfo` structure has the following fields:

```cpp
struct addrinfo {
    int              ai_flags;
    int              ai_family;
    int              ai_socktype;
    int              ai_protocol;
    socklen_t        ai_addrlen;
    struct sockaddr *ai_addr;
    char            *ai_canonname;
    struct addrinfo *ai_next;
};
```

You should not assume that the fields are stored in the order listed in the previous code, or that additional fields aren't present. There is some variation between operating systems.

The call to `getaddrinfo()` looks at only four fields in `*hints`. The rest of the structure should be zeroed-out. The relevant fields are `ai_family`, `ai_socktype`, `ai_protocol`, and `ai_flags`:

*   `ai_family` specifies the desired address family. It can be `AF_INET` for IPv4, `AF_INET6` for IPv6, or `AF_UNSPEC` for any address family. `AF_UNSPEC` is defined as `0`.
*   `ai_socktype` could be `SOCK_STREAM` for TCP (see [Chapter 3](eb2e080f-fad2-4e14-b4c1-9a6c124df77c.xhtml), *An In-Depth Overview of TCP Connections*), or `SOCK_DGRAM` for UDP (see [Chapter 4](05a32725-5c72-41e4-92aa-2425bf75282e.xhtml), *Establishing UDP Connections*). Setting `ai_socktype` to `0` indicates that the address could be used for either.
*   `ai_protocol` should be left to `0`. Strictly speaking, TCP isn't the only streaming protocol supported by the socket interface, and UDP isn't the only datagram protocol supported. `ai_protocol` is used to disambiguate, but it's not needed for our purposes.
*   `ai_flags` specifies additional options about how `getaddrinfo()` should work. Multiple flags can be used by bitwise OR-ing them together. In C, the bitwise OR operator uses the pipe symbol, `|`. So, bitwise OR-ing two flags together would use the `(flag_one | flag_two)` code.

Common flags you may use for the `ai_flags` field are:

*   `AI_NUMERICHOST` can be used to prevent name lookups. In this case, `getaddrinfo()` would expect `node` to be an address such as `127.0.0.1` and not a hostname such as `example.com`. `AI_NUMERICHOST` can be useful because it prevents `getaddrinfo()` from doing a DNS record lookup, which can be slow.
*   `AI_NUMERICSERV` can be used to only accept port numbers for the `service` argument. If used, this flag causes `getaddrinof()` to reject service names.
*   `AI_ALL` can be used to request both an IPv4 and IPv6 address. The declaration for `AI_ALL` seems to be missing on some Windows setups. It can be defined as `0x0100` on those platforms.
*   `AI_ADDRCONFIG` forces `getaddrinfo()` to only return addresses that match the family type of a configured interface on the local machine. For example, if your machine is IPv4 only, then using `AI_ADDRCONFIG | AI_ALL` prevents IPv6 addresses from being returned. It is usually a good idea to use this flag if you plan to connect a socket to the address returned by `getaddrinfo()`.
*   `AI_PASSIVE` can be used with `node = 0` to request the *wildcard address*. This is the local address that accepts connections on any of the host's network addresses. It is used on servers with `bind()`. If `node` is not `0`, then `AI_PASSIVE` has no effect. See [Chapter 3](eb2e080f-fad2-4e14-b4c1-9a6c124df77c.xhtml), *An In-Depth Overview of TCP Connections* for example usage.

All other fields in `hints` should be set to `0`. You can also pass in `0` for the `hints` argument, but different operating systems implement different defaults in that case.

The final parameter to `getaddrinfo()`, `res`, is a pointer to a pointer to `struct addrinfo` and returns the address(es) found by `getaddrinfo()`.

If the call to `getaddrinfo()` succeeds, then its return value is `0`. In this case, you should call `freeaddrinfo()` on `*res` when you've finished using the address. Here is an example of using `getaddrinfo()` to find the address(es) for `example.com`:

```cpp
struct addrinfo hints;
memset(&hints, 0, sizeof(hints));
hints.ai_flags = AI_ALL;
struct addrinfo *peer_address;
if (getaddrinfo("example.com", 0, &hints, &peer_address)) {
    fprintf(stderr, "getaddrinfo() failed. (%d)\n", GETSOCKETERRNO());
    return 1;
}
```

Note that we first zero out `hints` using a call to `memset()`. We then set the `AI_ALL` flag, which specifies that we want both IPv4 and IPv6 addresses returned. This even returns addresses that we don't have a network adapter for. If you only want addresses that your machine can practically connect to, then use `AI_ALL | AI_ADDRCONFIG` for the `ai_flags` field. We can leave the other fields of `hints` as their defaults.

We then declare a pointer to hold the return address list: `struct addrinfo *peer_address`.

If the call to `getaddrinfo()` succeeds, then `peer_address` holds the first address result. The next result, if any, is in `peer_address->ai_next`.

We can loop through all the returned addresses with the following code:

```cpp
struct addrinfo *address = peer_address;
do {
    /* Work with address... */
} while ((address = address->ai_next));
```

When we've finished using `peer_address`, we should free it with the following code:

```cpp
freeaddrinfo(peer_address);
```

Now that we can convert a text address or name into an `addrinfo` structure, it is useful to see how to convert the `addrinfo` structure back into a text format. Let's look at that now.

# Using getnameinfo()

`getnameinfo()` can be used to convert an `addrinfo` structure back into a text format. It works with both IPv4 and IPv6\. It also, optionally, converts a port number into a text format number or service name.

The declaration for `getnameinfo()` can be seen in the following code:

```cpp
int getnameinfo(const struct sockaddr *addr, socklen_t addrlen,
        char *host, socklen_t hostlen,
        char *serv, socklen_t servlen, int flags);
```

The first two parameters are passed in from the `ai_addr` and `ai_addrlen` fields of `struct addrinfo`.

The next two parameters, `host` and `hostlen`, specify a character buffer and buffer length to store the hostname or IP address text.

The following two parameters, `serv` and `servlen`, specify the buffer and length to store the service name.

If you don't need both the hostname and the service, you can optionally pass in only one of either `host` or `serv`.

Flags can be a bitwise OR combination of the following:

*   `NI_NAMEREQD` requires `getnameinfo()` to return a hostname and not an address. By default, `getnameinfo()` tries to return a hostname but returns an address if it can't. `NI_NAMEREQD` will cause an error to be returned if the hostname cannot be determined.
*   `NI_DGRAM` specifies that the service is UDP-based rather than TCP-based. This is only important for ports that have different standard services for UDP versus TCP. This flag is ignored if `NI_NUMERICSERV` is set.
*   `NI_NUMERICHOST` requests that `getnameinfo()` returns the IP address and not a hostname.
*   `NI_NUMERICSERV` requests that `getnameinfo()` returns the port number and not a service name.

For example, we can use `getnameinfo()` as follows:

```cpp
char host[100];
char serv[100];
getnameinfo(address->ai_addr, address->ai_addrlen,
        host, sizeof(host),
        serv, sizeof(serv),
        0);

printf("%s %s\n", host, serv);
```

In the preceding code, `getnameinfo()` attempts to perform a reverse DNS lookup. This works like the DNS queries we've done in this chapter so far, but backward. A DNS query asks w*hich IP address does this hostname point to?* A reverse DNS query asks instead, w*hich* *hostname does this IP address point to?* Keep in mind that this is not a one-to-one relationship. Many hostnames can point to one IP address, but an IP address can store a DNS record for only one hostname. In fact, reverse DNS records are not even set for many IP addresses.

If `address` is a `struct addrinfo` with the address for `example.com` port `80` (`http`), then the preceding code might print as follows:

```cpp
example.com http
```

If the code prints something different for you, it's probably working as intended. It is dependent on which address is in `address` and how the reverse DNS is set up for that IP address. Try it with a different address.

If, instead of the hostname, we would like the IP address, we can modify our code to the following:

```cpp
char host[100];
char serv[100];
getnameinfo(address->ai_addr, address->ai_addrlen,
        host, sizeof(host),
        serv, sizeof(serv),
        NI_NUMERICHOST | NI_NUMERICSERV);

printf("%s %s\n", host, serv);
```

In the case of the previous code, it might print the following:

```cpp
93.184.216.34 80
```

Using `NI_NUMERICHOST` generally runs much faster too, as it doesn't require `getnameinfo()` to send off any reverse DNS queries.

# Alternative functions

Two widely-available functions that replicate `getaddrinfo()` are `gethostbyname()` and `getservbyname()`. The `gethostbyname()` function is obsolete and has been removed from the newer POSIX standards. Furthermore, I recommend against using these functions in new code, because they introduce an IPv4 dependency. It's very possible to use `getaddrinfo()` in such a way that your program does not need to be aware of IPv4 versus IPv6, but still supports both.

# IP lookup example program

To demonstrate the `getaddrinfo()` and `getnameinfo()` functions, we will implement a short program. This program takes a name or IP address for its only argument. It then uses `getaddrinfo()` to resolve that name or that IP address into an address structure, and the program prints that IP address using `getnameinfo()` for the text conversion. If multiple addresses are associated with a name, it prints each of them. It also indicates any errors.

To begin with, we need to include our required header for this chapter. We also define `AI_ALL` for systems that are missing it. The code for this is as follows:

```cpp
/*lookup.c*/

#include "chap05.h"

#ifndef AI_ALL
#define AI_ALL 0x0100
#endi
```

We can then begin the `main()` function and check that the user passed in a hostname to lookup. If the user doesn't pass in a hostname, we print a helpful reminder. The code for this is as follows:

```cpp
/*lookup.c continued*/

int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Usage:\n\tlookup hostname\n");
        printf("Example:\n\tlookup example.com\n");
        exit(0);
    }
```

We need the following code to initialize Winsock on Windows platforms:

```cpp
/*lookup.c continued*/

#if defined(_WIN32)
    WSADATA d;
    if (WSAStartup(MAKEWORD(2, 2), &d)) {
        fprintf(stderr, "Failed to initialize.\n");
        return 1;
    }
#endif
```

We can then call `getaddrinfo()` to convert the hostname or address into a `struct addrinfo`. The code for that is as follows:

```cpp
/*lookup.c continued*/

    printf("Resolving hostname '%s'\n", argv[1]);
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_flags = AI_ALL;
    struct addrinfo *peer_address;
    if (getaddrinfo(argv[1], 0, &hints, &peer_address)) {
        fprintf(stderr, "getaddrinfo() failed. (%d)\n", GETSOCKETERRNO());
        return 1;
    }
```

The previous code first prints the hostname or address that is passed in as the first command-line argument. This argument is stored in `argv[1]`. We then set `hints.ai_flags = AI_ALL` to specify that we want all available addresses of any type, including both IPv4 and IPv6 addresses.

`getaddrinfo()` is then called with `argv[1]`. We pass `0` in for the service argument because we don't care about the port number. We are only trying to resolve an address. If `argv[1]` contains a name, such as `example.com`, then our operating system performs a DNS query (assuming the hostname isn't already in the local cache). If `argv[1]` contains an address such as `192.168.1.1`, then `getaddrinfo()` simply fills in the resulting `struct addrinfo` as needed.

If the user passed in an invalid address or a hostname for which no record could be found, then `getaddrinfo()` returns a non-zero value. In that case, our previous code prints out the error.

Now that `peer_address` holds the desired address(es), we can use `getnameinfo()` to convert them to text. The following code does that:

```cpp
/*lookup.c continued*/

    printf("Remote address is:\n");
    struct addrinfo *address = peer_address;
    do {
        char address_buffer[100];
        getnameinfo(address->ai_addr, address->ai_addrlen,
                address_buffer, sizeof(address_buffer),
                0, 0,
                NI_NUMERICHOST);
        printf("\t%s\n", address_buffer);
    } while ((address = address->ai_next));
```

This code works by first storing `peer_address` in a new variable, `address`. We then enter a loop. `address_buffer[]` is declared to store the text address, and we call `getnameinfo()` to fill in that address. The last parameter to `getnameinfo()`, `NI_NUMERICHOST`, indicates that we want it to put the IP address into `address_buffer` and not a hostname. The address buffer can then simply be printed out with `printf()`.

If `getaddrinfo()` returned multiple addresses, then the next address is pointed to by `address->ai_next`. We assign `address->ai_next` to `address` and loop if it is non-zero. This is a simple example of walking through a linked list.

After we've printed our address, we should use `freeaddrinfo()` to free the memory allocated by `getaddrinfo()`. We should also call the Winsock cleanup function on Windows. We can do both with the following code:

```cpp
/*lookup.c continued*/

    freeaddrinfo(peer_address);

#if defined(_WIN32)
    WSACleanup();
#endif

    return 0;
}
```

That concludes our `lookup` program.

You can compile and run `lookup.c` on Linux and macOS by using the following command:

```cpp
gcc lookup.c -o lookup
./lookup example.com
```

Compiling and running on Windows with MinGW is done in the following way:

```cpp
gcc lookup.c -o lookup.exe -lws2_32
lookup.exe example.com
```

The following screenshot is an example of using `lookup` to print the IP addresses for `example.com`:

![](img/6c0777cf-fe92-47b0-9b88-2c0f2737ef5e.png)

Although `getaddrinfo()` makes performing DNS lookups easy, it is useful to know what happens behind the scenes. We will now look at the DNS protocol in more detail.

# The DNS protocol

When a client wants to resolve a hostname into an IP address, it sends a DNS query to a DNS server. This is typically done over UDP using port `53`. The DNS server then performs the lookup, if possible, and returns an answer. The following diagram illustrates this transaction:

![](img/b1727003-67ae-41ac-86c5-afc0e8977d7a.png)

If the query (or, more commonly, the answer) is too large to fit into one UDP packet, then the query can be performed over TCP instead of UDP. In this case, the size of the query is sent over TCP as a 16-bit value, and then the query itself is sent. This is called **TCP fallback** or **DNS transport over TCP**. However, UDP works for most cases, and UDP is how DNS is used the vast majority of the time.

It's also important to note that the client must know the IP address of at least one DNS server. If the client doesn't know of any DNS servers, then it has a sort of chicken-and-egg problem. DNS servers are usually provided by your ISP.

The actual UDP data format is simple and follows the same basic format for both the query and the answer.

# DNS message format

The following illustration describes the DNS message format:

![](img/debc3c62-a462-47e0-af20-6d25f771e40d.png)

Every DNS message follows that format, although a query would leave the **Answer**, **Authority**, and **Additional** sections blank. A DNS response commonly doesn't use **Authority** or **Additional**. We won't concern ourselves with the **Authority** or **Additional** sections, as they are not needed for typical DNS queries.

# DNS message header format

The header is exactly 12 bytes long and is exactly the same for a DNS query or DNS response. The **Header Format** is illustrated graphically in the following diagram:

![](img/346de5c8-e0a1-4694-bee3-2ffd68761f09.png)

As the preceding diagram illustrates, the DNS message header contains 13 fields—**ID**, **QR**, **OPCODE**, **AA**, **TC**, **RD**, **RA**, **Z**, **RCODE**, **QDCOUNT**, **ANCOUNT**, **NSCOUNT**, and **ARCOUNT**:

*   **ID** is any 16-bit value that is used to identify the DNS message. The client is allowed to put any 16 bits into the DNS query, and the DNS server copies those same 16 bits into the DNS response **ID**. This is useful to allow the client to match up which response is in reply to which query, in cases where the client is sending multiple queries.
*   **QR** is a 1-bit field. It is set to `0` to indicate a DNS query or `1` to indicate a DNS response.
*   **Opcode** is a 4-bit field, which specifies the type of query. `0` indicates a standard query. `1` indicates a reverse query to resolve an IP address into a name. `2` indicates a server status request. Other values (`3` through `15`) are reserved.
*   **AA** indicates an authoritative answer.
*   **TC** indicates that the message was truncated. In this case, it should be re-sent using TCP.
*   **RD** should be set if recursion is desired. We leave this bit set to indicate that we want the DNS server to contact additional servers until it can complete our request.
*   **RA** indicates in a response whether the DNS server supports recursion.
*   **Z** is unused and should be set to `0`.

*   **RCODE** is set in a DNS response to indicate the error condition. Its possible values are as follows:

| **RCODE** | **Description** |
| `0` | No error |
| `1` | Format error |
| `2` | Server failure |
| `3` | Name error |
| `4` | Not implemented |
| `5` | Refused |

Please see **RFC 1035**: *DOMAIN NAMES – IMPLEMENTATION AND SPECIFICATION,* in the *Further reading* section of this chapter for more information on the meaning of these values.

*   **QDCOUNT**, **ANCOUNT**, **NSCOUNT**, and **ARCOUNT** indicate the number of records in their corresponding sections. **QDCOUNT** indicates the number of questions in a DNS query. It is interesting that **QDCOUNT** is a 16-bit value capable of storing large numbers, and yet no real-world DNS server allows more than one question per message. **ANCOUNT** indicates the number of answers, and it is common for a DNS server to return multiple answers in one message.

# Question format

The DNS **Question Format** consists of a name followed by two 16-bit values—`QTYPE` and `QCLASS`. This **Question Format** is illustrated as follows:

![](img/d793dee6-5fa1-4bb2-947d-eae5221dd568.png)

**QTYPE** indicates the record type we are asking for, and **QCLASS** should be set to `1` to indicate the internet.

The name field involves a special encoding. First, a hostname should be broken up into its individual labels. For example, `www.example.com` would be broken up into `www`, `example`, and `com`. Then, each label should be prepended with 1 byte, indicating the label length. Finally, the entire name ends with a 0 byte.

Visually, the name `www.example.com` is encoded as follows:

![](img/982010ad-4fe5-4e83-b2e5-37d2798aa631.png)

If the **QTYPE** and **QCLASS** fields were appended to the preceding name example, then it could make up an entire DNS question.

A DNS response is sometimes required to repeat the same name multiple times. In this case, a DNS server may encode a pointer to an earlier name instead of sending the same name multiple times. A pointer is indicated by a 16-bit value with the two most significant bits set. The lower 14 bits indicate the pointer value. This 14-bit value specifies the location of the name as an offset from the beginning of the message. Having the two most significant bits reserved has an additional side-effect of limiting labels to 63 characters. A longer name would require setting both high bits in the label length specifier, but if both high bits are set, it indicates a pointer and not a label length!

The answer format is similar to the question format but with a few more fields. Let's look at that next.

# Answer format

The DNS answer format consists of the same three fields that questions have; namely, a name followed by a 16-bit `TYPE` and a 16-bit `CLASS`. The answer format then has a 32-bit `TTL` field. This field specifies how many seconds the answer is allowed to be cached for. `TTL` is followed by a 16-bit length specifier, `RDLENGTH`, followed by data. The data is `RDLENGTH` long, and the data's interpretation is dependent upon the type specified by `TYPE`.

Visually, the answer format is shown in the following diagram:

![](img/d78755d4-2ce9-463f-9bc6-f7f5b8a48a5b.png)

Keep in mind that most DNS servers use name pointers in their answer names. This is because the DNS response will have already included the relevant name in the question section. The answer can simply point back to that, rather than encoding the entire name a second (or third) time.

Whenever sending binary data over the network, the issue of byte order becomes relevant. Let's consider this now.

# Endianness

The term **endianness** refers to the order in which individual bytes are stored in memory or sent over a network.

Whenever we read a multi-byte number from a DNS message, we should be aware that it's in big-endian format (or so-called network byte order). The computer you're using likely uses little-endian format, although we are careful to write our code in an endian-independent manner throughout this book. We accomplish this by avoiding the conversion of multiple bytes directly to integers, and instead we interpret bytes one at a time.

For example, consider a message with a single 8-bit value, such as `0x05`. We know that the value of that message is `5`. Bytes are sent atomically over a network link, so we also know that anyone receiving our message can unambiguously interpret that message as `5`.

The issue of endianness comes into play when we need more than one byte to store our number. Imagine that we want to send the number `999`. This number is too big to fit into 1 byte, so we have to break it up into 2 bytes—a 16-bit value. Because `999 = (3 * 2⁸) + 231`, we know that the high-order byte stores `3` while the low-order byte stores `231`. In hexadecimal, the number `999` is `0x03E7`. The question is whether to send the high-order or the low-order byte over the network first.

Network byte order, which is used by the DNS protocol, specifies that the high-order byte is sent first. Therefore, the number `999` is sent over the network as a `0x03` followed by `0xE7`.

See the *Further reading* section of this chapter for more information.

Let's now look at encoding an entire DNS query.

# A simple DNS query

To perform a simple DNS query, we would put an arbitrary number into `ID`, set the `RD` bit to `1`, and set `QDCOUNT` to `1`. We would then add a question after the header. That data would be sent as a UDP packet to port `53` of a DNS server.

A hand-constructed DNS for `example.com` in C is as follows:

```cpp
char dns_query[] = {0xAB, 0xCD,                           /* ID */
                    0x01, 0x00,                           /* Recursion */
                    0x00, 0x01,                           /* QDCOUNT */
                    0x00, 0x00,                           /* ANCOUNT */
                    0x00, 0x00,                           /* NSCOUNT */
                    0x00, 0x00,                           /* ARCOUNT */
                    7, 'e', 'x', 'a', 'm', 'p', 'l', 'e', /* label */
                    3, 'c', 'o', 'm',                     /* label */
                    0,                                    /* End of name */
                    0x00, 0x01,                           /* QTYPE = A */
                    0x00, 0x01                            /* QCLASS */
                    };
```

This data could be sent as is to a DNS server over port `53`.

The DNS server, if successful, sends a UDP packet back as a response. This packet has `ID` set to match our query. `QR` is set to indicate a response. `QDCOUNT` is set to `1`, and our original question is included. `ANCOUNT` is some small positive integer that indicates the number of answers included in the message.

In the next section, we'll implement a program to send and receive DNS messages.

# A DNS query program

We will now implement a utility to send DNS queries to a DNS server and receive the DNS response.

This should not normally be needed in the field. It is, however, a good opportunity to better understand the DNS protocol and to get experience of sending binary UDP packets.

We begin with a function to print a name from a DNS message.

# Printing a DNS message name

DNS encodes names in a particular way. Normally, each label is indicated by its length, followed by its text. A number of labels can be repeated, and then the name is terminated with a single 0 byte.

If a length has its two highest bits set (that is, `0xc0`), then it and the next byte should be interpreted as a pointer instead.

We must also be aware at all times that the DNS response from the DNS server could be ill-formed or corrupted. We must try to write our program in such a way that it won't crash if it receives a bad message. This is easier said than done.

The declaration for our name-printing function looks like this:

```cpp
/*dns_query.c*/

const unsigned char *print_name(const unsigned char *msg,
        const unsigned char *p, const unsigned char *end);
```

We take `msg` to be a pointer to the message's beginning, `p` to be a pointer to the name to print, and `end` to be a pointer to one past the end of the message. `end` is required so that we can check that we're not reading past the end of the received message. `msg` is required for the same reason, but also so that we can interpret name pointers.

Inside the `print_name` function, our code checks that a proper name is even possible. Because a name should consist of at least a length and some text, we can return an error if `p` is already within two characters of the end. The code for that check is as follows:

```cpp
/*dns_query.c*/

    if (p + 2 > end) {
        fprintf(stderr, "End of message.\n"); exit(1);}
```

We then check to see if `p` points to a name pointer. If it does, we interpret the pointer and call `print_name` recursively to print the name that is pointed to. The code for this is as follows:

```cpp
/*dns_query.c*/

    if ((*p & 0xC0) == 0xC0) {
        const int k = ((*p & 0x3F) << 8) + p[1];
        p += 2;
        printf(" (pointer %d) ", k);
        print_name(msg, msg+k, end);
        return p;
```

Note that `0xC0` in binary is `0b11000000`. We use `(*p & 0xC0) == 0xC0` to check for a name pointer. In that case, we take the lower 6 bits of `*p` and all 8 bits of `p[1]` to indicate the pointer. We know that `p[1]` is still within the message because of our earlier check that `p` was at least 2 bytes from the end. Knowing the name pointer, we can pass a new value of `p` to `print_name()`.

If the name is not a pointer, we simply print it one label at a time. The code for printing the name is as follows:

```cpp
/*dns_query.c*/

    } else {
        const int len = *p++;
        if (p + len + 1 > end) {
            fprintf(stderr, "End of message.\n"); exit(1);}

        printf("%.*s", len, p);
        p += len;
        if (*p) {
            printf(".");
            return print_name(msg, p, end);
        } else {
            return p+1;
        }
    }
```

In the preceding code, `*p` is read into `len` to store the length of the current label. We are careful to check that reading `len + 1` bytes doesn't put us past the end of the buffer. We can then print the next `len` characters to the console. If the next byte isn't `0`, then the name is continued, and we should print a dot to separate the labels. We call `print_name()` recursively to print the next label of the name.

If the next byte is `0`, then that means the name is finished and we return.

That concludes the `print_name()` function.

Now we will devise a function that prints an entire DNS message.

# Printing a DNS message

Using `print_name()` that we have just defined, we can now construct a function to print an entire DNS message to the screen. DNS messages share the same format for both the request and the response, so our function is able to print either.

The declaration for our function is as follows:

```cpp
/*dns_query.c*/

void print_dns_message(const char *message, int msg_length);
```

`print_dns_message()` takes a pointer to the start of the message, and an `int` data type indicates the message's length.

Inside `print_dns_message()`, we first check that the message is long enough to be a valid DNS message. Recall that the DNS header is 12 bytes long. If a DNS message is less than 12 bytes, we can easily reject it as an invalid message. This also ensures that we can read at least the header without worrying about reading past the end of the received data.

The code for checking the DNS message length is as follows:

```cpp
/*dns_query.c*/

    if (msg_length < 12) {
        fprintf(stderr, "Message is too short to be valid.\n");
        exit(1);
    }
```

We then copy the `message` pointer into a new variable, `msg`. We define `msg` as an `unsigned char` pointer, which makes certain calculations easier to work with.

```cpp
/*dns_query.c*/

    const unsigned char *msg = (const unsigned char *)message;
```

If you want to print out the entire raw DNS message, you can do that with the following code:

```cpp
/*dns_query.c*/

    int i;
    for (i = 0; i < msg_length; ++i) {
        unsigned char r = msg[i];
        printf("%02d:   %02X  %03d  '%c'\n", i, r, r, r);
    }
    printf("\n");
```

Be aware that running the preceding code will print out many lines. This can be annoying, so I would recommend using it only if you are curious about seeing the raw DNS message.

The message ID can be printed very easily. Recall that the message ID is simply the first two bytes of the message. The following code prints it in a nice hexadecimal format:

```cpp
/*dns_query.c*/

    printf("ID = %0X %0X\n", msg[0], msg[1]);
```

Next, we get the `QR` bit from the message header. This bit is the most significant bit of `msg[2]`. We use the bitmask `0x80` to see whether it is set. If it is, we know that the message is a response; otherwise, it's a query. The following code reads `QR` and prints a corresponding message:

```cpp
/*dns_query.c*/

    const int qr = (msg[2] & 0x80) >> 7;
    printf("QR = %d %s\n", qr, qr ? "response" : "query");
```

The `OPCODE`, `AA`, `TC`, and `RD` fields are read in much the same way as `QR`. The code for printing them is as follows:

```cpp
/*dns_query.c*/

    const int opcode = (msg[2] & 0x78) >> 3;
    printf("OPCODE = %d ", opcode);
    switch(opcode) {
        case 0: printf("standard\n"); break;
        case 1: printf("reverse\n"); break;
        case 2: printf("status\n"); break;
        default: printf("?\n"); break;
    }

    const int aa = (msg[2] & 0x04) >> 2;
    printf("AA = %d %s\n", aa, aa ? "authoritative" : "");

    const int tc = (msg[2] & 0x02) >> 1;
    printf("TC = %d %s\n", tc, tc ? "message truncated" : "");

    const int rd = (msg[2] & 0x01);
    printf("RD = %d %s\n", rd, rd ? "recursion desired" : "");
```

Finally, we can read in `RCODE` for response-type messages. Since `RCODE` can have several different values, we use a `switch` statement to print them. Here is the code for that:

```cpp
/*dns_query.c*/

    if (qr) {
        const int rcode = msg[3] & 0x07;
        printf("RCODE = %d ", rcode);
        switch(rcode) {
            case 0: printf("success\n"); break;
            case 1: printf("format error\n"); break;
            case 2: printf("server failure\n"); break;
            case 3: printf("name error\n"); break;
            case 4: printf("not implemented\n"); break;
            case 5: printf("refused\n"); break;
            default: printf("?\n"); break;
        }
        if (rcode != 0) return;
    }
```

The next four fields in the header are the question count, the answer count, the name server count, and the additional count. We can read and print them in the following code:

```cpp
/*dns_query.c*/

    const int qdcount = (msg[4] << 8) + msg[5];
    const int ancount = (msg[6] << 8) + msg[7];
    const int nscount = (msg[8] << 8) + msg[9];
    const int arcount = (msg[10] << 8) + msg[11];

    printf("QDCOUNT = %d\n", qdcount);
    printf("ANCOUNT = %d\n", ancount);
    printf("NSCOUNT = %d\n", nscount);
    printf("ARCOUNT = %d\n", arcount);
```

That concludes reading the DNS message header (the first 12 bytes).

Before reading the rest of the message, we define two new variables as follows:

```cpp
/*dns_query.c*/

    const unsigned char *p = msg + 12;
    const unsigned char *end = msg + msg_length;
```

In the preceding code, the `p` variable is used to walk through the message. We set the `end` variable to one past the end of the message. This is to help us detect whether we're about to read past the end of the message – a situation we certainly wish to avoid!

We read and print each question in the DNS message with the following code:

```cpp
/*dns_query.c*/

    if (qdcount) {
        int i;
        for (i = 0; i < qdcount; ++i) {
            if (p >= end) {
                fprintf(stderr, "End of message.\n"); exit(1);}

            printf("Query %2d\n", i + 1);
            printf("  name: ");

            p = print_name(msg, p, end); printf("\n");

            if (p + 4 > end) {
                fprintf(stderr, "End of message.\n"); exit(1);}

            const int type = (p[0] << 8) + p[1];
            printf("  type: %d\n", type);
            p += 2;

            const int qclass = (p[0] << 8) + p[1];
            printf(" class: %d\n", qclass);
            p += 2;
        }
    }
```

Although no real-world DNS server will accept a message with multiple questions, the DNS RFC does clearly define the format to encode multiple questions. For that reason, we make our code loop through each question using a `for` loop. First, the `print_name()` function, which we defined earlier, is called to print the question name. We then read in and print out the question type and class.

Printing the answer, authority, and additional sections is slightly more difficult than the question section. These sections start the same way as the question section – with a name, a type, and a class. The code for reading the name, type, and class is as follows:

```cpp
/*dns_query.c*/

    if (ancount || nscount || arcount) {
        int i;
        for (i = 0; i < ancount + nscount + arcount; ++i) {
            if (p >= end) {
                fprintf(stderr, "End of message.\n"); exit(1);}

            printf("Answer %2d\n", i + 1);
            printf("  name: ");

            p = print_name(msg, p, end); printf("\n");

            if (p + 10 > end) {
                fprintf(stderr, "End of message.\n"); exit(1);}

            const int type = (p[0] << 8) + p[1];
            printf("  type: %d\n", type);
            p += 2;

            const int qclass = (p[0] << 8) + p[1];
            printf(" class: %d\n", qclass);
            p += 2;
```

Note that, in the preceding code, we stored the class in a variable called `qclass`. This is to be nice to our C++ friends, who are not allowed to use `class` as a variable name.

We then expect to find a 16-bit TTL field, and a 16-bit data length field. The TTL field tells us how many seconds we are allowed to cache an answer for. The data length field tells us how many bytes of additional data are included for the answer. We read TTL and the data length in the following code:

```cpp
/*dns_query.c*/

            const unsigned int ttl = (p[0] << 24) + (p[1] << 16) +
                (p[2] << 8) + p[3];
            printf("   ttl: %u\n", ttl);
            p += 4;

            const int rdlen = (p[0] << 8) + p[1];
            printf(" rdlen: %d\n", rdlen);
            p += 2;
```

Before we can read in the data of the `rdlen` length, we should check that we won't read past the end of the message. The following code achieves that:

```cpp
/*dns_query.c*/

            if (p + rdlen > end) {
                fprintf(stderr, "End of message.\n"); exit(1);}
```

We can then try to interpret the answer data. Each record type stores different data. We need to write code to display each type. For our purposes, we limit this to the `A`, `MX`, `AAAA`, `TXT`, and `CNAME` records. The code to print each type is as follows:

```cpp
/*dns_query.c*/

            if (rdlen == 4 && type == 1) { /* A Record */
                printf("Address ");
                printf("%d.%d.%d.%d\n", p[0], p[1], p[2], p[3]);

            } else if (type == 15 && rdlen > 3) { /* MX Record */
                const int preference = (p[0] << 8) + p[1];
                printf("  pref: %d\n", preference);
                printf("MX: ");
                print_name(msg, p+2, end); printf("\n");

            } else if (rdlen == 16 && type == 28) { /* AAAA Record */
                printf("Address ");
                int j;
                for (j = 0; j < rdlen; j+=2) {
                    printf("%02x%02x", p[j], p[j+1]);
                    if (j + 2 < rdlen) printf(":");
                }
                printf("\n");

            } else if (type == 16) { /* TXT Record */
                printf("TXT: '%.*s'\n", rdlen-1, p+1);

            } else if (type == 5) { /* CNAME Record */
                printf("CNAME: ");
                print_name(msg, p, end); printf("\n");
            }
            p += rdlen;
```

We can then finish the loop. We check that all the data was read and print a message if it wasn't. If our program is correct, and if the DNS message is properly formatted, we should have read all the data with nothing left over. The following code checks this:

```cpp
/*dns_query.c*/

        }
    }

    if (p != end) {
        printf("There is some unread data left over.\n");
    }

    printf("\n");
```

That concludes the `print_dns_message()` function.

We can now define our `main()` function to create the DNS query, send it to a DNS server, and await a response.

# Sending the query

We start `main()` with the following code:

```cpp
/*dns_query.c*/

int main(int argc, char *argv[]) {

    if (argc < 3) {
        printf("Usage:\n\tdns_query hostname type\n");
        printf("Example:\n\tdns_query example.com aaaa\n");
        exit(0);
    }

    if (strlen(argv[1]) > 255) {
        fprintf(stderr, "Hostname too long.");
        exit(1);
    }
```

The preceding code checks that the user passed in a hostname and record type to query. If they didn't, it prints a helpful message. It also checks that the hostname isn't more than 255 characters long. Hostnames longer than that aren't allowed by the DNS standard, and checking it now ensures that we don't need to allocate too much memory.

We then try to interpret the record type requested by the user. We support the following options – `a`, `aaaa`, `txt`, `mx`, and `any`. The code to read in those types and store their corresponding DNS integer value is as follows:

```cpp
/*dns_query.c*/

    unsigned char type;
    if (strcmp(argv[2], "a") == 0) {
        type = 1;
    } else if (strcmp(argv[2], "mx") == 0) {
        type = 15;
    } else if (strcmp(argv[2], "txt") == 0) {
        type = 16;
    } else if (strcmp(argv[2], "aaaa") == 0) {
        type = 28;
    } else if (strcmp(argv[2], "any") == 0) {
        type = 255;
    } else {
        fprintf(stderr, "Unknown type '%s'. Use a, aaaa, txt, mx, or any.",
                argv[2]);
        exit(1);
    }
```

Like all of our previous programs, we need to initialize Winsock. The code for that is as follows:

```cpp
/*dns_query.c*/

#if defined(_WIN32)
    WSADATA d;
    if (WSAStartup(MAKEWORD(2, 2), &d)) {
        fprintf(stderr, "Failed to initialize.\n");
        return 1;
    }
#endif
```

Our program connects to `8.8.8.8`, which is a public DNS server run by Google. Refer to [Chapter 1](e3e07fa7-ff23-4871-b897-c0d4551e6422.xhtml),  *An Introduction to Networks and Protocols*, *Domain Names*, for a list of additional public DNS servers you can use.

Recall that we are connecting on UDP port `53`. We use `getaddrinfo()` to set up the required structures for our socket with the following code:

```cpp
/*dns_query.c*/

    printf("Configuring remote address...\n");
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_DGRAM;
    struct addrinfo *peer_address;
    if (getaddrinfo("8.8.8.8", "53", &hints, &peer_address)) {
        fprintf(stderr, "getaddrinfo() failed. (%d)\n", GETSOCKETERRNO());
        return 1;
    }
```

We then create our socket using the data returned from `getaddrinfo()`. The following code does that:

```cpp
/*dns_query.c*/

    printf("Creating socket...\n");
    SOCKET socket_peer;
    socket_peer = socket(peer_address->ai_family,
            peer_address->ai_socktype, peer_address->ai_protocol);
    if (!ISVALIDSOCKET(socket_peer)) {
        fprintf(stderr, "socket() failed. (%d)\n", GETSOCKETERRNO());
        return 1;
    }
```

Our program then constructs the data for the DNS query message. The first 12 bytes compose the header and are known at compile time. We can store them with the following code:

```cpp
/*dns_query.c*/

    char query[1024] = {0xAB, 0xCD, /* ID */
                        0x01, 0x00, /* Set recursion */
                        0x00, 0x01, /* QDCOUNT */
                        0x00, 0x00, /* ANCOUNT */
                        0x00, 0x00, /* NSCOUNT */
                        0x00, 0x00 /* ARCOUNT */};
```

The preceding code sets our query's ID to `0xABCD`, sets a recursion request, and indicates that we are attaching `1` question. As mentioned earlier, `1` is the only number of questions supported by real-world DNS servers.

We then need to encode the user's desired hostname into the query. The following code does that:

```cpp
/*dns_query.c*/

    char *p = query + 12;
    char *h = argv[1];

    while(*h) {
        char *len = p;
        p++;
        if (h != argv[1]) ++h;

        while(*h && *h != '.') *p++ = *h++;
        *len = p - len - 1;
    }

    *p++ = 0;
```

The preceding code first sets a new pointer, `p`, to the end of the query header. We will be adding to the query starting at `p`. We also define a pointer, `h`, which we use to loop through the hostname.

We can loop while `*h != 0` because `*h` is equal to zero when we've finished reading the hostname. Inside the loop, we use the `len` variable to store the position of the label beginning. The value in this position needs to be set to indicate the length of the upcoming label. We then copy characters from `*h` to `*p` until we find a dot or the end of the hostname. If either is found, the code sets `*len` equal to the label length. The code then loops into the next label.

Finally, outside the loop, we add a terminating 0 byte to finish the name section of the question.

We then add the question type and question class to the query with the following code:

```cpp
/*dns_query.c*/

    *p++ = 0x00; *p++ = type; /* QTYPE */
    *p++ = 0x00; *p++ = 0x01; /* QCLASS */
```

We can then calculate the query size by comparing `p` to the query beginning. The code for figuring the total query size is as follows:

```cpp
/*dns_query.c*/

    const int query_size = p - query;
```

Now, with the query message formed, and its length known, we can use `sendto()` to transmit the DNS query to the DNS server. The code for sending the query is as follows:

```cpp
/*dns_query.c*/

    int bytes_sent = sendto(socket_peer,
            query, query_size,
            0,
            peer_address->ai_addr, peer_address->ai_addrlen);
    printf("Sent %d bytes.\n", bytes_sent);
```

For debugging purposes, we can also display the query we sent with the following code:

```cpp
/*dns_query.c*/

    print_dns_message(query, query_size);
```

The preceding code is useful to see whether we've made any mistakes in encoding our query.

Now that the query has been sent, we await a DNS response message using `recvfrom()`. In a practical program, you may want to use `select()` here to time out. It could also be wise to listen for additional messages in the case that an invalid message is received first.

The code to receive and display the DNS response is as follows:

```cpp
/*dns_query.c*/

    char read[1024];
    int bytes_received = recvfrom(socket_peer,
            read, 1024, 0, 0, 0);

    printf("Received %d bytes.\n", bytes_received);

    print_dns_message(read, bytes_received);
    printf("\n");
```

We can finish our program by freeing the address(es) from `getaddrinfo()` and cleaning up Winsock. The code to complete the `main()` function is as follows:

```cpp
/*dns_query.c*/

    freeaddrinfo(peer_address);
    CLOSESOCKET(socket_peer);

#if defined(_WIN32)
    WSACleanup();
#endif

    return 0;
}
```

That concludes the `dns_query` program.

You can compile and run `dns_query.c` on Linux and macOS by running the following command:

```cpp
gcc dns_query.c -o dns_query
./dns_query example.com a
```

Compiling and running on Windows with MinGW is done by using the following command:

```cpp
gcc dns_query.c -o dns_query.exe -lws2_32
dns_query.exe example.com a
```

Try running `dns_query` with different domain names and different record types. In particular, try it with `mx` and `txt` records. If you're brave, try running it with the `any` record type. You may find the results interesting.

The following screenshot is an example of using `dns_query` to query the `A` record of `example.com`:

![](img/969470fe-66c2-45b8-bc85-cfc29b63d33b.png)

The next screenshot shows `dns_query` querying the `mx` record of `gmail.com`:

![](img/fc7393f1-2aa1-42c4-877a-2f78463f726e.png)

Note that UDP is not always reliable. If our DNS query is lost in transit, then `dns_query` hangs while waiting forever for a reply that never comes. This could be fixed by using the `select()` function to time out and retry.

# Summary

This chapter was all about hostnames and DNS queries. We covered how the DNS works, and we learned that resolving a hostname can involve many UDP packets being sent over the network.

We looked at `getaddrinfo()` in more depth and showed why it is usually the preferred way to do a hostname lookup. We also looked at its sister function, `getnameinfo()`, which is capable of converting an address to text or even doing a reverse DNS query.

Finally, we implemented a program that sent DNS queries from scratch. This program was a good learning experience to better understand the DNS protocol, and it gave us a chance to gain experience in implementing a binary protocol. When implementing a binary protocol, we had to pay special attention to byte order. For the simple DNS message format, this was achieved by carefully interpreting bytes one at a time.

Now that we've worked with a binary protocol, DNS, we will move on to text-based protocols in the next few chapters. In the next chapter, we will learn about HTTP, the protocol used to request and retrieve web pages.

# Questions

Try these questions to test your knowledge of this chapter:

1.  Which function fills in an address needed for socket programming in a portable and protocol-independent way?
2.  Which socket programming function can be used to convert an IP address back into a name?
3.  A DNS query converts a name into an address, and a reverse DNS query converts an address back into a name. If you run a DNS query on a name, and then a reverse DNS query on the resulting address, do you always get back the name you started with?
4.  What are the DNS record types used to return IPv4 and IPv6 addresses for a name?
5.  Which DNS record type stores special information about email servers?
6.  Does `getaddrinfo()` always return immediately? or can it block?
7.  What happens when a DNS response is too large to fit into a single UDP packet?

The answers are in [Appendix A](bd8b8f52-52cb-4d34-b01b-e907564bfece.xhtml), *Answers to Questions*.

# Further reading

For more information about DNS, please refer to:

*   **RFC 1034**: *DOMAIN NAMES – CONCEPTS AND FACILITIES* *(*[https://tools.ietf.org/html/rfc1034](https://tools.ietf.org/html/rfc1034))
*   **RFC 1035**: *DOMAIN NAMES – IMPLEMENTATION AND SPECIFICATION* *(*[https://tools.ietf.org/html/rfc1035](https://tools.ietf.org/html/rfc1035))
*   **RFC 3596**: *DNS Extensions to Support IP Version 6 (*[https://tools.ietf.org/html/rfc3596](https://tools.ietf.org/html/rfc3596))