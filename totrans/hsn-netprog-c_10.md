# Making Your Program Send Email

In this chapter, we will consider the protocol responsible for delivering email on the internet. This protocol is called the **Simple Mail Transfer Protocol** (**SMTP**).

Following an exposition of the inner workings of email transfer, we will build a simple SMTP client capable of sending short emails.

The following topics are covered in this chapter:

*   How SMTP servers work
*   Determining which mail server is responsible for a given domain
*   Using SMTP
*   Email encoding
*   Spam-blocking and email-sending pitfalls
*   SPF, DKIM, and DMARC

# Technical requirements

The example programs from this chapter can be compiled with any modern C compiler. We recommend MinGW on Windows and GCC on Linux and macOS. See [Appendix B](47da8507-709b-44a6-9399-b18ce6afd8c9.xhtml), *Setting Up Your C Compiler on Windows*, [Appendix C](221eebc0-0bb1-4661-a5aa-eafed9fcba7e.xhtml), *Setting Up Your C Compiler on Linux*, and [Appendix D](632db68e-0911-4238-a2be-bd1aa5296120.xhtml), *Setting Up Your C Compiler on macOS*, for compiler setup.

The code for this book can be found at [https://github.com/codeplea/Hands-On-Network-Programming-with-C](https://github.com/codeplea/Hands-On-Network-Programming-with-C).

From the command line, you can download the code for this chapter with the following command:

```cpp
git clone https://github.com/codeplea/Hands-On-Network-Programming-with-C
cd Hands-On-Network-Programming-with-C/chap08
```

Each example program in this chapter runs on Windows, Linux, and macOS. When compiling on Windows, each example program requires linking with the Winsock library. This can be accomplished by passing the `-lws2_32` option to `gcc`.

We provide the exact commands needed to compile each example as they are introduced.

All of the example programs in this chapter require the same header files and C macros that we developed in [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs*. For brevity, we put these statements in a separate header file, `chap08.h`, which we can include in each program. For a detailed explanation of these statements, please refer to [Chapter 2](https://cdp.packtpub.com/hands_on_network_programming_with_c/wp-admin/post.php?post=31&action=edit#post_25), *Getting to Grips with Socket APIs*.

The first part of `chap08.h` includes the needed networking headers for each platform. The code for this is as follows:

```cpp
/*chap08.h*/

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

We also define some macros to make writing portable code easier, and we'll include the additional headers that our programs require:

```cpp
/*chap08.h continued*/

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

That concludes `chap08.h`.

# Email servers

SMTP is the protocol responsible for delivering emails between servers. It is a text-based protocol operating on TCP port `25`.

Not all emails need to be delivered between systems. For example, imagine you have a Gmail account. If you compose and send an email to your friend who also has a Gmail account, then SMTP is not necessarily used. In this case, Gmail only needs to copy your email into their inbox (or do equivalent database updates).

On the other hand, consider a case where you send an email to your friend's Yahoo! account. If the email is sent from your Gmail account, then it's clear that the Gmail and Yahoo! servers must communicate. In that case, your email is transmitted from the Gmail server to the Yahoo! server using SMTP.

This connection is illustrated in the following diagram:

![](img/cc2d0147-cd00-430a-889b-01c9c8ab33eb.png)

Retrieving your email from your mail service provider is a different issue than delivering email between service providers. Webmail is very popular now for sending and receiving mail from your mail provider. Webmail providers allow mailbox access through a web browser. Web browsers communicate using HTTP (or HTTPS).

Let's consider the full path of an email from **Alice** to **Bob**. In this example, Alice has **Gmail** as her mail provider, and Bob has **Yahoo!** as his mail provider. Both Alice and Bob access their mailbox using a standard web browser. The path an email takes from Bob to Alice is illustrated by the following diagram:

![](img/fc5f08f6-9756-47cd-9963-4871398ede99.png)

As you can see in the preceding diagram, **SMTP** is only used when delivering the mail between mail providers. This usage of SMTP is called **mail transmission**.

In fact, the email in this example could take other paths. Let's consider that Alice uses a desktop email client instead of webmail. Gmail still supports desktop email clients, and these clients offer many good features, even if they are falling out of fashion. A typical desktop client connects to a mail provider using either: **Internet Message Access Protocol** (**IMAP**) or **Post Office Protocol** (**POP**) and SMTP. In this case, SMTP is used by Alice to deliver her mail to her mail provider (Gmail). This usage of SMTP is called **mail submission**.

The Gmail provider then uses SMTP again to deliver the email to the **Yahoo!** mail server. This is illustrated in the following diagram:

![](img/f654b33e-4fdf-4fdc-afb2-6f523c1bb1e9.png)

In the preceding diagram, the Gmail server would be considered an **SMTP relay**. In general, an SMTP server should only relay mail for trusted users. If an SMTP server relayed all mail, it would become quickly overwhelmed by spammers.

Many mail providers have a set of mail servers used to accept incoming mail and a separate set of mail servers used to accept outgoing mail from their users.

It is important to understand that SMTP is used to send mail. SMTP is not used to retrieve mail from a server. IMAP and POP are the common protocols used by desktop mail programs to retrieve mail from a server.

It is not necessary for Alice to send her mail through her provider's SMTP server. Instead, she could send it directly to Bob's mail provider as illustrated in the following diagram:

![](img/5dd6d1fb-1504-416b-91d0-7f29d8d9da93.png)

In practice, one usually delegates the delivery responsibility to their mail provider. This has several advantages; namely, the mail provider can attempt a redelivery if the destination mail server isn't available. Other advantages are discussed later in this chapter.

The program we develop in this chapter is used to deliver mail directly to the recipient's email provider. It is not useful to deliver mail to a relay server because we are not going to implement authentication techniques. Generally, SMTP servers do not relay mail without authenticating that the sender has an account with them.

The SMTP protocol we describe in this chapter is unsecured and not encrypted. This is convenient for explanation and learning purposes, but in the real world, you may want to secure your email transfer.

# SMTP security

We describe unsecured SMTP in this chapter. In real-world use, SMTP should be secured if both communicating servers support it. Not all do.

Securing SMTP is done by having SMTP connections start out as plaintext on port `25`. The SMTP client then issues a `STARTTLS` command to upgrade to a secure, encrypted connection. This secure connection works by merely running the SMTP commands through a TLS layer; therefore, everything we cover in this chapter is applicable to secure SMTP too. See [Chapter 9](47ba170d-42d9-4e38-b5d8-89503e005708.xhtml), *Loading **Secure Web Pages with HTTPS and OpenSSL,* for more information about TLS.

Mail transmission between servers is always done on port `25`.

Many desktop email clients use TCP ports `465` or `587` for SMTP mail submission. **Internet Service Providers (ISPs)** prefer these alternative ports for mail submission, and it allows them to block port `25` altogether. This is generally justified as a spam prevention technique.

Next, let's see how to determine which mail server receives mail for a given email address.

# Finding an email server

Consider the email address `bob@example.com`. In this case, `bob` identifies the user, while `example.com` identifies the domain name of the service provider. These parts are delineated by the `@` symbol.

One domain name can potentially use multiple mail servers. Likewise, one mail server can provide service for many domain names. For this reason, identifying the mail server or servers responsible for receiving mail for `bob@example.com` isn't as easy as connecting to `example.com`. Instead, the mail server must be identified by performing a DNS lookup for an MX record.

DNS was covered in depth back in [Chapter 5](3d80e3b8-07d3-49f4-b60f-b006a17f7213.xhtml), *Hostname Resolution and DNS*. The program we developed in that chapter can be used to query MX records.

Otherwise, most operating systems provide a command-line tool for DNS lookup. Windows provides `nslookup`, while Linux and macOS provide `dig`.

On Windows, we can find the mail servers responsible for accepting mail `@gmail.com` using the following command:

```cpp
nslookup -type=mx gmail.com
```

This lookup is shown in the following screenshot:

![](img/6f4e6420-bf5b-491a-9c56-2a09ec4f19d7.png)

On Linux or macOS, an MX record lookup for a `@gmail.com` account is done with the following command:

```cpp
dig mx gmail.com
```

The use of `dig` is shown in the following screenshot:

![](img/cc1f9572-2088-4dbf-87a6-cf0a0750a8c0.png)

As you can see in the preceding two screenshots, Gmail uses five mail servers. When multiple MX records are found, mail should be delivered to the server having the lowest MX preference first. If mail delivery fails to that server, then an attempt should be made to deliver to the server having the next lowest preference, and so on. At the time of this writing, Gmail's primary mail server, having a preference of `5`, is `gmail-smtp-in.l.google.com`. That is the SMTP server you would connect to in order to send mail to an `@gmail.com` address.

It is also possible for MX records to have the same preference. Yahoo! uses mail servers having the same preference. The following screenshot shows the MX records for `yahoo.com`:

![](img/36cd8937-4fc7-4057-8168-7f2586deb7ce.png)

In the preceding screenshot, we see that Yahoo! uses three mail servers. Each server has a preference of `1`. This means that mail can be delivered to any one of them with no special preference. If mail delivery fails to the first chosen server, then another server should be chosen at random to retry.

Programmatically getting the MX record in a cross-platform manner can be difficult. Please see [Chapter 5](3d80e3b8-07d3-49f4-b60f-b006a17f7213.xhtml), *Hostname Resolution and DNS*, where this topic was covered in some depth. The SMTP client we develop in this present chapter assumes that the mail server is known in advance.

Now that we know which server to connect to, let's consider the SMTP protocol itself in more detail.

# SMTP dialog

SMTP is a text-based TCP protocol that works on port `25`. SMTP works in a lock-step, one-at-a-time dialog, with the client sending commands and the server sending responses for each command.

In a typical session, the dialog goes as follows:

1.  The client first establishes a connection to the SMTP server.
2.  The server initiates with a greeting. This greeting indicates that the server is ready to receive commands.
3.  The client then issues its own greeting.
4.  The server responds.
5.  The client sends a command indicating who the mail is from.
6.  The server responds to indicate that the sender is accepted.
7.  The client issues another command, which specifies the mail's recipient.
8.  The server responds indicating the recipient is accepted.
9.  The client then issues a `DATA` command.
10.  The server responds asking the client to proceed.
11.  The client transfers the email.

The protocol is very simple. In the following example SMTP session, `mail.example.net` is the client, and the server is `mail.example.com` (`C` and `S` indicate whether the client or server is sending, respectively):

```cpp
S: 220 mail.example.com SMTP server ready
C: HELO mail.example.net
S: 250 Hello mail.example.net [192.0.2.67]
C: MAIL FROM:<alice@example.net>
S: 250 OK
C: RCPT TO:<bob@example.com>
S: 250 Accepted
C: DATA
S: 354 Enter message, ending with "." on a line by itself
C: Subject: Re: The Cake
C: Date: Fri, 03 May 2019 02:31:20 +0000
C:
C: Do NOT forget to bring the cake!
C: .
S: 250 OK
C: QUIT
S: 221 closing connection
```

Everything the server sends is in reply to the client's commands, except for the first line. The first line is simply in response to the client connecting.

You may notice that each of the client's commands start with a four-letter word. Each one of the server's responses starts with a three-digit code.

The common client commands we use are as follows:

*   `HELO` is used for the client to identify itself to the server.
*   `MAIL` is used to specify who is sending the mail.
*   `RCPT` is used to specify a recipient.
*   `DATA` is used to initiate the transfer of the actual email. This email should include both headers and a body.
*   `QUIT` is used to end the session.

The server response codes used in a successful email transfer are the following:

*   `220`: The service is ready
*   `250`: The requested command was accepted and completed successfully
*   `354`: Start sending the message
*   `221`: The connection is closing

Error codes vary between providers, but they are generally in the 500 range.

SMTP servers can also send replies spanning multiple lines. In this case, the very last line begins with the response code followed by a space. All preceding lines begin with the response code followed by a dash. The following example illustrates a multiline response after attempting to deliver to a mailbox that does not exist:

```cpp
C: MAIL FROM:<alice@example.net>
S: 250 OK
C: RCPT TO:<not-a-real-user@example.com>
S: 550-The account you tried to deliver to does not
S: 550-exist. Please double-check the recipient's
S: 550 address for typos and try again.
```

Note that some servers validate that the recipient address is valid before replying to the `RCPT` command, but many servers only validate the recipient address after the client sends the email using the `DATA` command.

Although that explains the basics of the protocol used to send mail, we still must consider the format of the email itself. This is covered next.

# The format of an email

If we make an analogy to physical mail, the SMTP commands `MAIL FROM` and `RCPT TO` address the envelope. Those commands give the SMTP server information on how the mail is to be delivered. In this analogy, the `DATA` command would be the letter inside the envelope. As it's common to address a physical letter inside an envelope, it's also common to repeat the delivery information in the email, even though it was already sent to the SMTP server through the `MAIL` and `RCPT` commands.

A simple email may look like the following:

```cpp
From: Alice Doe <alice@example.net>
To: Bob Doe <bob@example.com>
Subject: Re: The Cake
Date: Fri, 03 May 2019 02:31:20 +0000

Hi Bob,

Do NOT forget to bring the cake!

Best,
Alice
```

The entire email is transmitted to an SMTP server following the `DATA` command. A single period on an otherwise blank line is transmitted to indicate the end of the email. If the email contains any line beginning with a period, the SMTP client should replace it with two consecutive periods. This prevents the client from indicating the email is over prematurely. The SMTP server knows that any line beginning with two periods should be replaced with a single period.

The email itself can be divided into two parts—the header and the body. The two parts are delineated by the first blank line.

The header part consists of various headers that indicate properties of the email. `From`, `To`, `Subject`, and `Date` are the most common headers.

The body part of the email is simply the message being sent.

With a basic understanding of the email format, we are now ready to begin writing a simple C program to send emails.

# A simple SMTP client program

With a basic understanding of both SMTP and the email format, we are ready to program a simple email client. Our client takes as inputs: the destination email server, the recipient's address, the sender's address, the email subject line, and the email body text.

Our program begins by including the necessary headers with the following statements:

```cpp
/*smtp_send.c*/

#include "chap08.h"
#include <ctype.h>
#include <stdarg.h>
```

We also define the following two constants to make buffer allocation and checking easier:

```cpp
/*smtp_send.c continued*/

#define MAXINPUT 512
#define MAXRESPONSE 1024
```

Our program needs to prompt the user for input several times. This is required to get the email server's hostname, the recipient's address, and so on. C provides the `gets()` function for this purpose but `gets()` is deprecated in the latest C standard. Therefore, we implement our own function.

The following function, `get_input()`, prompts for user input:

```cpp
/*smtp_send.c continued*/

void get_input(const char *prompt, char *buffer)
{
    printf("%s", prompt);

    buffer[0] = 0;
    fgets(buffer, MAXINPUT, stdin);
    const int read = strlen(buffer);
    if (read > 0)
        buffer[read-1] = 0;
}
```

The `get_input()` function uses `fgets()` to read from `stdin`. The buffer passed to `get_input()` is assumed to be `MAXINPUT` bytes, which we defined at the top of the file.

The `fgets()` function does not remove a newline character from the received input; therefore, we overwrite the last character inputted with a terminating null character.

It is also very helpful to have a function that can send formatted strings directly over the network. We implement a function called `send_format()` for this purpose. It takes a socket, a formatting string, and the additional arguments to send. You can think of `send_format()` as being very similar to `printf()`. The difference is that `send_format()` delivers the formatted text over the network instead of printing to the screen.

The code for `send_format()` is as follows:

```cpp
/*smtp_send.c continued*/

void send_format(SOCKET server, const char *text, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, text);
    vsprintf(buffer, text, args);
    va_end(args);

    send(server, buffer, strlen(buffer), 0);

    printf("C: %s", buffer);
}
```

The preceding code works by first reserving a buffer. `vsprintf()` is then used to format the text into that buffer. It is up to the caller to ensure that the formatted output doesn't exceed the reserved buffer space. We are assuming for this program that the user is trusted, but in a production program, you would want to add checks to prevent a buffer overflow here.

After the output text is formatted into `buffer`, it is sent using `send()`. We also print the sent text to the screen. A `C:` is printed preceding it to indicate that the text was sent by us, the client.

One of the trickier parts of our SMTP client is parsing the SMTP server responses. This is important because the SMTP client must not issue a second command until a response is received for the first command. If the SMTP client sends a new command before the server is ready, then the server will likely terminate the connection.

Recall that each SMTP response starts with a three-digit code. We want to parse out this code to check for errors. Each SMTP response is usually followed by text that we ignore. SMTP responses are typically only one line long, but they can sometimes span multiple lines. In this case, each line up to the penultimate line contains a dash character, `-`, directly following the three-digit response code.

To illustrate how multiline responses work, consider the following two responses as equivalent:

```cpp
/*response 1*/

250 Message received!
/*response 2*/

250-Message
250 received!
```

It is important that our program recognizes multiline responses; it must not mistakenly treat a single multiline response as separate responses.

We implement a function called `parse_response()` for this purpose. It takes in a null-terminated response string and returns the parsed response code. If no code is found or the response isn't complete, then `0` is returned instead. The code for this function is as follows:

```cpp
/*smtp_send.c continued*/

int parse_response(const char *response) {
    const char *k = response;
    if (!k[0] || !k[1] || !k[2]) return 0;
    for (; k[3]; ++k) {
        if (k == response || k[-1] == '\n') {
            if (isdigit(k[0]) && isdigit(k[1]) && isdigit(k[2])) {
                if (k[3] != '-') {
                    if (strstr(k, "\r\n")) {
                        return strtol(k, 0, 10);
                    }
                }
            }
        }
    }
    return 0;
}
```

The `parse_response()` function begins by checking for a null terminator in the first three characters of the response. If a null is found there, then the function can return immediately because `response` isn't long enough to constitute a valid SMTP response.

It then loops through the `response` input string. The loop goes until a null-terminating character is found three characters out. Each loop, `isdigit()` is used to see whether the current character and the next two characters are all digits. If so, the fourth character, `k[3]`, is checked. If `k[3]` is a dash, then the response continues onto the next line. However, if `k[3]` isn't a dash, then `k[0]` represents the beginning of the last line of the SMTP response. In this case, the code checks if the line ending has been received; `strstr()` is used for this purpose. It the line ending was received, the code uses `strtol()` to convert the response code to an integer.

If the code loops through `response()` without returning, then `0` is returned, and the client needs to wait for more input from the SMTP server.

With `parse_response()` out of the way, it is useful to have a function that waits until a particular response code is received over the network. We implement a function called `wait_on_response()` for this purpose, which begins as follows:

```cpp
/*smtp_send.c continued*/

void wait_on_response(SOCKET server, int expecting) {
    char response[MAXRESPONSE+1];
    char *p = response;
    char *end = response + MAXRESPONSE;

    int code = 0;
```

In the preceding code, a `response` buffer variable is reserved for storing the SMTP server's response. A pointer, `p`, is set to the beginning of this buffer; `p` will be incremented to point to the end of the received data, but it starts at `response` since no data has been received yet. An `end` pointer variable is set to the end of the buffer, which is useful to ensure we do not attempt to write past the buffer end.

Finally, we set `code = 0` to indicate that no response code has been received yet.

The `wait_on_response()` function then continues with a loop as follows:

```cpp
/*smtp_send.c continued*/

    do {
        int bytes_received = recv(server, p, end - p, 0);
        if (bytes_received < 1) {
            fprintf(stderr, "Connection dropped.\n");
            exit(1);
        }

        p += bytes_received;
        *p = 0;

        if (p == end) {
            fprintf(stderr, "Server response too large:\n");
            fprintf(stderr, "%s", response);
            exit(1);
        }

        code = parse_response(response);

    } while (code == 0);
```

The beginning of the preceding loop uses `recv()` to receive data from the SMTP server. The received data is written at point `p` in the `response` array. We are careful to use `end` to make sure received data isn't written past the end of `response`.

After `recv()` returns, `p` is incremented to the end of the received data, and a null terminating character is set. A check for `p == end` ensures that we haven't written to the end of the response buffer.

Our function from earlier, `parse_response()`, is used to check whether a full SMTP response has been received. If so, then `code` is set to that response. If not, then `code` is equal to `0`, and the loop continues to receive additional data.

After the loop terminates, the `wait_on_response()` function checks that the received SMTP response code is as expected. If so, the received data is printed to the screen, and the function returns. The code for this is as follows:

```cpp
/*smtp_send.c continued*/

    if (code != expecting) {
        fprintf(stderr, "Error from server:\n");
        fprintf(stderr, "%s", response);
        exit(1);
    }

    printf("S: %s", response);
}
```

That concludes the `wait_on_response()` function. This function proves very useful, and it is needed after every command sent to the SMTP server.

We also define a function called `connect_to_host()`, which attempts to open a TCP connection to a given hostname and port number. This function is extremely similar to the code we've used in the previous chapters.

First `getaddrinfo()` is used to resolve the hostname and `getnameinfo()` is then used to print the server IP address. The following code achieves those two purposes:

```cpp
/*smtp_send.c continued*/

SOCKET connect_to_host(const char *hostname, const char *port) {
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

A socket is then created with `socket()`, as shown in the following code:

```cpp
/*smtp_send.c continued*/

    printf("Creating socket...\n");
    SOCKET server;
    server = socket(peer_address->ai_family,
            peer_address->ai_socktype, peer_address->ai_protocol);
    if (!ISVALIDSOCKET(server)) {
        fprintf(stderr, "socket() failed. (%d)\n", GETSOCKETERRNO());
        exit(1);
    }
```

Once the socket has been created, `connect()` is used to establish the connection. The following code shows the use of `connect()` and the end of the `connect_to_host()` function:

```cpp
/*smtp_send.c continued*/

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

Don't forget to call `freeaddrinfo()` to free the memory allocated for the server address, as shown by the preceding code.

Finally, with those helper functions out of the way, we can begin on `main()`. The following code defines `main()` and initializes Winsock if required:

```cpp
/*smtp_send.c continued*/

int main() {

#if defined(_WIN32)
    WSADATA d;
    if (WSAStartup(MAKEWORD(2, 2), &d)) {
        fprintf(stderr, "Failed to initialize.\n");
        return 1;
    }
#endif
```

See [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs*, for more information about initializing Winsock and establishing connections.

Our program can proceed by prompting the user for an SMTP hostname. This hostname is stored in `hostname`, and our `connect_to_host()` function is used to open a connection. The following code shows this:

```cpp
    /*smtp_send.c continued*/

    char hostname[MAXINPUT];
    get_input("mail server: ", hostname);

    printf("Connecting to host: %s:25\n", hostname);

    SOCKET server = connect_to_host(hostname, "25");
```

After the connection is established, our SMTP client must not issue any commands until the server responds with a `220` code. We use `wait_on_response()` to wait for this with the following code:

```cpp
/*smtp_send.c continued*/

    wait_on_response(server, 220);
```

Once the server is ready to receive commands, we must issue the `HELO` command. The following code sends the `HELO` command and waits for a `250` response code:

```cpp
/*smtp_send.c continued*/

    send_format(server, "HELO HONPWC\r\n");
    wait_on_response(server, 250);
```

`HELO` should be followed by the SMTP client's hostname; however, since we are probably running this client from a development machine, it's likely we don't have a hostname setup. For this reason, we simply send `HONPWC`, although any arbitrary string can be used. If you are running this client from a server, then you should change the `HONPWC` string to a domain that points to your server.

Also, note the line ending used in the preceding code. The line ending used by SMTP is a carriage return character followed by a line feed character. In C, this is represented by `"\r\n"`.

Our program then prompts the user for the sending and receiving addresses and issues the appropriate SMTP commands. This is done using `get_input()` to prompt the user, `send_format()` to issue the SMTP commands, and `wait_on_response()` to receive the SMTP server's response:

```cpp
/*smtp_send.c continued*/

    char sender[MAXINPUT];
    get_input("from: ", sender);
    send_format(server, "MAIL FROM:<%s>\r\n", sender);
    wait_on_response(server, 250);

    char recipient[MAXINPUT];
    get_input("to: ", recipient);
    send_format(server, "RCPT TO:<%s>\r\n", recipient);
    wait_on_response(server, 250);
```

After the sender and receiver are specified, the next step in the SMTP is to issue the `DATA` command. The `DATA` command instructs the server to listen for the actual email. It is issued by the following code:

```cpp
/*smtp_send.c continued*/

    send_format(server, "DATA\r\n");
    wait_on_response(server, 354);
```

Our client program then prompts the user for an email subject line. After the subject line is specified, it can send the email headers: `From`, `To`, and `Subject`. The following code does this:

```cpp
/*smtp_send.c continued*/

    char subject[MAXINPUT];
    get_input("subject: ", subject);

    send_format(server, "From:<%s>\r\n", sender);
    send_format(server, "To:<%s>\r\n", recipient);
    send_format(server, "Subject:%s\r\n", subject);
```

It is also useful to add a date header. Emails use a special format for dates. We can make use of the `strftime()` function to format the date properly. The following code formats the date into the proper email header:

```cpp
/*smtp_send.c continued*/

    time_t timer;
    time(&timer);

    struct tm *timeinfo;
    timeinfo = gmtime(&timer);

    char date[128];
    strftime(date, 128, "%a, %d %b %Y %H:%M:%S +0000", timeinfo);

    send_format(server, "Date:%s\r\n", date);
```

In the preceding code, the `time()` function is used to get the current date and time, and `gmtime()` is used to convert it into a `timeinfo` struct. Then, `strftime()` is called to format the data and time into a temporary buffer, `date`. This formatted string is then transmitted to the SMTP server as an email header.

After the email headers are sent, the email body is delineated by a blank line. The following code sends this blank line:

```cpp
/*smtp_send.c continued*/

    send_format(server, "\r\n");
```

We can then prompt the user for the body of the email using `get_input()`. The body is transmitted one line at a time. When the user finishes their email, they should enter a single period on a line by itself. This indicates both to our client and the SMTP server that the email is finished.

The following code sends user input to the server until a single period is inputted:

```cpp
/*smtp_send.c continued*/

    printf("Enter your email text, end with \".\" on a line by itself.\n");

    while (1) {
        char body[MAXINPUT];
        get_input("> ", body);
        send_format(server, "%s\r\n", body);
        if (strcmp(body, ".") == 0) {
            break;
        }
    }
```

If the mail was accepted by the SMTP server, it sends a `250` response code. Our client then issues the `QUIT` command and checks for a `221` response code. The `221` response code indicates that the connection is terminating as shown in the following code:

```cpp
/*smtp_send.c continued*/

    wait_on_response(server, 250);

    send_format(server, "QUIT\r\n");
    wait_on_response(server, 221);
```

Our SMTP client concludes by closing the socket, cleaning up Winsock (if required), and exiting as shown here:

```cpp
/*smtp_send.c continued*/

    printf("\nClosing socket...\n");
    CLOSESOCKET(server);

#if defined(_WIN32)
    WSACleanup();
#endif

    printf("Finished.\n");
    return 0;
}
```

That concludes `smtp_send.c`.

You can compile and run `smtp_send.c` on Windows with MinGW using the following:

```cpp
gcc smtp_send.c -o smtp_send.exe -lws2_32
smtp_send.exe
```

On Linux or macOS, compiling and running `smtp_send.c` is done by the following:

```cpp
gcc smtp_send.c -o smtp_send
./smtp_send
```

The following screenshot shows the sending of a simple email using `smtp_send.c`:

![](img/18ae9fc0-07ef-4066-afca-6be70e0c0e26.png)

If you're doing a lot of testing, you may find it tedious to type in the email each time. In that case, you can automate it by putting your text in a file and using the `cat` utility to read it into `smtp_send`. For example, you may have the `email.txt` file as follows:

```cpp
/*email.txt*/

mail-server.example.net
bob@example.com
alice@example.net
Re: The Cake
Hi Alice,

What about the cake then?

Bob
.
```

With the program input stored in `email.txt`, you can send an email using the following command:

```cpp
cat email.txt | ./smtp_send
```

Hopefully, you can send some test emails with `smtp_send`. There are, however, a few obstacles you may run into. Your ISP may block outgoing emails from your connection, and many email servers do not accept mail from residential IP address blocks. See the *Spam-blocking pitfalls* section later in the chapter for more information.

Although `smtp_send` is useful for sending simple text-based messages, you may be wondering how to add formatting to your email. Perhaps you want to send files as attachments. The next section addresses these issues.

# Enhanced emails

The emails we've been looking at so far have been only simple text. Modern email usage often demands fancier formatted emails.

We can control the content type of an email using the `Content-Type` header. This is very similar to the content type header used by HTTP, which we covered in [Chapter 7](f352830e-089c-4369-b7a2-18a896e1c5d5.xhtml), *Building a Simple Web Server*.

If the content type header is missing, a content type of `text/plain` is assumed by default. Therefore, the `Content-Type` header in the following email is redundant:

```cpp
From: Alice Doe <alice@example.net>
To: Bob Doe <bob@example.com>
Subject: Re: The Cake
Date: Fri, 03 May 2019 02:31:20 +0000
Content-Type: text/plain

Hi Bob,

Do NOT forget to bring the cake!

Best,
Alice
```

If you want formatting support in your email, which is common today, you should use a `text/html` content type. In the following email, HTML is used to add emphasis:

```cpp
From: Alice Doe <alice@example.net>
To: Bob Doe <bob@example.com>
Subject: Re: The Cake
Date: Fri, 03 May 2019 02:31:20 +0000
Content-Type: text/html

Hi Bob,<br>
<br>
Do <strong>NOT</strong> forget to bring the cake!<br>
<br>
Best,<br>
Alice<br>
```

Not all email clients support HTML emails. For this reason, it may be useful to encode your message as both plaintext and HTML. The following email uses this technique:

```cpp
From: Alice Doe <alice@example.net>
To: Bob Doe <bob@example.com>
Subject: Re: The Cake
Date: Fri, 03 May 2019 02:31:20 +0000
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="SEPARATOR"

This is a message with multiple parts in MIME format.
--SEPARATOR
Content-Type: text/plain

Hi Bob,

Do NOT forget to bring the cake!

Best,
Alice
--SEPARATOR
Content-Type: text/html

Hi Bob,<br>
<br>
Do <strong>NOT</strong> forget to bring the cake!<br>
<br>
Best,<br>
Alice<br>
--SEPARATOR--
```

The preceding email example uses two headers to indicate that it's a multipart message. The first one, `MIME-Version: 1.0`, indicates which version of **Multipurpose Internet Mail Extensions** (**MIME**) we're using. MIME is used for all emails that aren't simply plaintext.

The second header, `Content-Type: multipart/alternative; boundary="SEPARATOR"`, specifies that we're sending a multipart message. It also specifies a special boundary character sequence that delineates the parts of the email. In our example, `SEPARATOR` is used as the boundary. It is important that the boundary not appear in the email text or attachments. In practice, boundary specifiers are often long randomly generated strings.

Once the boundary has been set, each part of the email begins with `--SEPARATOR` on a line by itself. The email ends with `--SEPARATOR--`. Note that each part of the message gets its own header section, specific to only that part. These header sections are used to specify the content type for each part.

It is also often useful to attach files to email, which we will cover now.

# Email file attachments

If a multipart email is being sent, a part can be designated as an attachment using the `Content-Disposition` header. See the following example:

```cpp
From: Alice Doe <alice@example.net>
To: Bob Doe <bob@example.com>
Subject: Re: The Cake
Date: Fri, 03 May 2019 02:31:20 +0000
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="SEPARATOR"

This is a message with multiple parts in MIME format.
--SEPARATOR
Content-Type: text/plain

Hi Bob,

Please see the attached text file.

Best,
Alice
--SEPARATOR
Content-Disposition: attachment; filename=my_file.txt;
  modification-date="Fri, 03 May 2019 02:26:51 +0000";
Content-Type: application/octet-stream
Content-Transfer-Encoding: base64

VGhpcyBpcyBhIHNpbXBsZSB0ZXh0IG1lc3NhZ2Uu
--SEPARATOR--
```

The preceding example includes a file called `my_file.txt`. SMTP is a purely text-based protocol; therefore, any attachments that may include binary data need to be encoded into a text format. Base64 encoding is commonly used for this purpose. In this example, the header, `Content-Transfer-Encoding: base64`, specifies that we are going to use Base64 encoding.

The content of `my_file.txt` is `This is a simple text message`. That sentence encodes to Base64 as `VGhpcyBpcyBhIHNpbXBsZSB0ZXh0IG1lc3NhZ2Uu` as seen in the preceding code.

# Spam-blocking pitfalls

It can be much harder to send emails today than it was in the past. Spam has become a major problem, and every provider is taking actions to curb it. Unfortunately, many of these actions can also make it much more difficult to send legitimate emails.

Many residential ISPs don't allow outgoing connections on port `25`. If your residential provider blocks port `25`, then you won't be able to establish an SMTP connection. In this case, you may consider renting a virtual private server to run this chapter's code.

Even if your ISP does allow outgoing connections on port `25`, many SMTP servers won't accept mail from a residential IP address. Of the servers that do, many will send those emails straight into a spam folder.

For example, if you attempt to deliver an email to Gmail, you may get a response similar to the following:

```cpp
550-5.7.1 [192.0.2.67] The IP you're using to send mail is not authorized
550-5.7.1 to send email directly to our servers. Please use the SMTP
550-5.7.1 relay at your service provider instead. Learn more at
550 5.7.1  https://support.google.com/mail/?p=NotAuthorizedError
```

Another spam-blocking measure that may trip you up is **Sender Policy Framework** (**SPF**). SPF works by listing which servers can send mail for a given domain. If a sending server isn't on the SPF list, then receiving SMTP servers reject their mail.

**DomainKeys Identified Mail** (**DKIM**) is a measure to authenticate email using digital signatures. Many popular email providers are more likely to treat non-DKIM mail as spam. DKIM signing is very complicated and out of scope for this book.

**Domain-based Message Authentication*,* Reporting, and Conformance** (**DMARC**) is a technique used for domains to publish whether SPF and/or DKIM is required of mail originating from them, among other things.

Most commercial email servers use SPF, DKIM, and DMARC. If you're sending email without these, your email will likely be treated as spam. If you're sending email in opposition to these, your email will either be rejected outright or labeled as spam.

Finally, many popular providers assign a reputation to sending domains and servers. This puts potential emails senders in a catch-22 situation. Email can't be delivered without building a reputation, but a reputation can't be built without successfully delivering lots of emails. As spam continues to be a major problem, we may soon come to a time where only big-name, trusted SMTP servers can operate with each other. Let's hope it doesn't come to that!

If your program needs to send email reliably, you should likely consider using an email service provider. One option is to allow an SMTP relay to do the final delivery. A potentially easier option is to use a mail sending service that operates an HTTP API.

# Summary

In this chapter, we looked at how email is delivered over the internet. SMTP, the protocol responsible for email delivery, was studied in some depth. We then constructed a simple program to send short emails using SMTP.

We looked at the email format too. We saw how MIME could be used to send multipart emails with file attachments.

We also saw how sending emails over the modern internet is full of pitfalls. Many of these stems from attempts to block spam. Techniques used by providers, such as blocking residential IP addresses, SPF, DKIM, DMARC, and IP address reputation monitoring, may make it difficult for our simple program to deliver email reliably.

In the next chapter, [Chapter 9](47ba170d-42d9-4e38-b5d8-89503e005708.xhtml), *Loading Secure Web Pages with HTTPS and OpenSSL*, we look at secure web connections using HTTPS.

# Questions

Try these questions to test your knowledge from this chapter:

1.  What port does SMTP operate on?
2.  How do you determine which SMTP server receives mail for a given domain?
3.  How do you determine which SMTP server sends mail for a given provider?
4.  Why won't an SMTP server relay mail without authentication?
5.  How are binary files sent as email attachments when SMTP is a text-based protocol?

The answers can be found in [Appendix A](bd8b8f52-52cb-4d34-b01b-e907564bfece.xhtml), *Answers to Questions*.

# Further reading

For more information about SMTP and email formats, please refer to the following links:

*   **RFC 821**: *Simple Mail Transfer Protocol* ([https://tools.ietf.org/html/rfc821](https://tools.ietf.org/html/rfc821))
*   **RFC 2822**: *Internet* *Message Format* ([https://tools.ietf.org/html/rfc2822](https://tools.ietf.org/html/rfc2822))