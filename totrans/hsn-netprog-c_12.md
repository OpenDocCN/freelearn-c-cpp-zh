# Loading Secure Web Pages with HTTPS and OpenSSL

In this chapter, we'll learn how to initiate secure connections to web servers using **Hypertext Transfer Protocol Secure** (**HTTPS**). HTTPS provides several benefits over HTTP. HTTPS gives an authentication method to identify servers and detect impostors. It also protects the privacy of all transmitted data and prevents interceptors from tampering with or forging transmitted data.

In HTTPS, communication is secured using **Transport Layer Security** (**TLS**). In this chapter, we'll learn how to use the popular OpenSSL library to provide TLS functionality.

The following topics are covered in this chapter:

*   Background information about HTTPS
*   Types of encryption ciphers
*   How servers are authenticated
*   Basic OpenSSL usage
*   Creating a simple HTTPS client

# Technical requirements

The example programs from this chapter can be compiled using any modern C compiler. We recommend MinGW for Windows and GCC for Linux and macOS. You also need to have the OpenSSL library installed. See [Appendices B](47da8507-709b-44a6-9399-b18ce6afd8c9.xhtml), *Setting Up Your C Compiler on Windows*, [Appendices C](221eebc0-0bb1-4661-a5aa-eafed9fcba7e.xhtml), *Setting Up Your C Compiler on Linux*, and [Appendices D](632db68e-0911-4238-a2be-bd1aa5296120.xhtml), *Setting Up Your C Compiler on macOS,* for compiler setup and OpenSSL installation.

The code for this book can be found at [https://github.com/codeplea/Hands-On-Network-Programming-with-C](https://github.com/codeplea/Hands-On-Network-Programming-with-C).

From the command line, you can download the code for this chapter with the following command:

```cpp
git clone https://github.com/codeplea/Hands-On-Network-Programming-with-C
cd Hands-On-Network-Programming-with-C/chap09
```

Each example program in this chapter runs on Windows, Linux, and macOS. While compiling on Windows, each example program requires linking with the Winsock library. This can be accomplished by passing the `-lws2_32` option to `gcc`.

Each example also needs to be linked against the OpenSSL libraries, `libssl.a` and `libcrypto.a`. This is accomplished by passing `-lssl -lcrypto` to `gcc`.

We provide the exact commands needed to compile each example as it is introduced.

All of the example programs in this chapter require the same header files and C macros that we developed in [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs*. For brevity, we put these statements in a separate header file, `chap09.h`, which we can include in each program. For an explanation of these statements, please refer to [Chapter 2](4f41c930-c4b4-47e5-b9ef-f8faf21fa96b.xhtml), *Getting to Grips with Socket APIs*.

The content of `chap09.h` begins by including the necessary networking header files. The code for this follows:

```cpp
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

We also define some macros to assist with writing portable code like so:

```cpp
/*chap09.h continued*/

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

Finally, `chap09.h` includes some additional headers, including the headers for the OpenSSL library. This is shown in the following code:

```cpp
/*chap09.h continued*/

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

# HTTPS overview

HTTPS provides security to HTTP. We covered HTTP in [Chapter 6](de3d2e9b-b94e-47d1-872c-c2ecb34c4026.xhtml), *Building a Simple Web Client*. HTTPS secures HTTP by using TLS over TCP on port `443`. TLS is a protocol that can provide security to any TCP connection.

TLS is the successor to **Secure Socket Layer** (**SSL**), an earlier protocol also used by HTTPS. TLS and SSL are compatible, and most of the information in this chapter also applies to SSL. Generally, establishing an HTTPS connection involves the client and server negotiating which protocol to use. The ideal outcome is that the client and server agree on the most secure, mutually supported protocol and cipher.

When we talk about protocol security, we are generally looking for the following three things:

*   **Authentication**: We need a way to prevent impostors from posing as legitimate communication partners. TLS provides peer authentication methods for this reason.
*   **Encryption**: TLS uses encryption to obfuscate transmitted data. This prevents an eavesdropper from correctly interpreting intercepted data.
*   **Integrity**: TLS also ensures that received data has not been tampered with or otherwise forged.

**HTTP** is most commonly used to transmit web pages. The text on a web page is first encoded as **Hypertext Markup Language** (**HTML**). **HTML** provides formatting, layout, and styling to web pages. **HTTP** is then used to transmit the **HTML**, and **HTTP** itself is transmitted over a **TCP** connection.

Visually, an **HTTP** session is encapsulated like the following:

![](img/f6ee2b1c-e5bf-49e0-bdf8-c5cb6b83cdfe.png)

**TLS** works inside TCP to provide a secure communication channel. HTTPS is then basically the same as the **HTTP** protocol, but it is sent inside a **TLS** channel.

Visually, HTTPS is encapsulated in the following manner:

![](img/691de8da-d33f-47a0-9d2f-a9b21f640a45.png)

Of course, the same abstraction still applies if **HTTP** is used to transmit an image, video, or other data instead of **HTML**.

Do keep in mind that these abstractions are accurate at the conceptual level, but some details transcend layers. For example, some HTTPS headers are used to refer to security parameters of how **TLS** should be applied. In general, though, it is reasonable to think of **TLS** as securing the **TCP** connection used by HTTPS.

Although **TLS** is most commonly used for HTTPS security, **TLS** is also used to secure many other **TCP**-based protocols. The email protocol, SMTP, which we covered in [Chapter 8](47e209f2-0231-418c-baef-82db74df8c29.xhtml), *Making Your Program Send Email*, is also commonly secured by **TLS**.

Before going into further detail about using **TLS**, it is useful to understand some necessary background information on encryption.

# Encryption basics

**Encryption** is a method of encoding data so that only authorized parties can access it. Encryption does not prevent interception or interference, but it denies the original data to a would-be attacker.

Encryption algorithms are called **ciphers**. An encryption cipher takes unencrypted data as input, referred to as **plaintext**. The cipher produces encrypted data, called **ciphertext**, as its output. The act of converting plaintext into ciphertext is called encryption, and the act of reversing it back is called **decryption**.

Modern ciphers use keys to control the encryption and decryption of data. Keys are typically relatively short, pseudo-random data sequences. Ciphertext encrypted with a given key cannot be decrypted without the proper key.

Broadly, there are two categories of ciphers—symmetric and asymmetric. A symmetric cipher uses the same key for both encryption and decryption, while an asymmetric cipher uses two different keys.

# Symmetric ciphers

The following diagram illustrates a symmetric cipher:

![](img/ca2c2000-2b21-4918-8248-b0cf74dae2a8.png)

In the preceding diagram, the plaintext **Hello!** is encrypted using a symmetric cipher. A secret key is used with the cipher to produce a ciphertext. This ciphertext can then be transmitted over an insecure channel, and eavesdroppers cannot decipher it without knowledge of the secret key. The privileged receiver of the ciphertext uses the decryption algorithm and secret key to convert it back into plaintext.

Some symmetric ciphers in general use (not just for TLS) are the following:

*   **American Encryption Standard** (**AES**), also known as **Rijndael**
*   Camellia
*   **Data Encryption Standard** (**DES**)
*   Triple DES
*   **International Data Encryption Algorithm** (**IDEA**)
*   QUAD
*   RC4
*   Salsa20, Chacha20
*   **Tiny Encryption Algorithm** (**TEA**)

One issue with symmetric encryption is that the same key must be known to both the sender and receiver. Generating and transmitting this key securely poses a problem. How can the key be sent between parties if they don't already have a secure communication channel? If they do already have a secure communication channel, why is encryption needed in the first place?

Key exchange algorithms attempt to address these problems. Key exchange algorithms work by allowing both communicating parties to generate the same secret key. In general, the parties first agree on a public, non-secret key. Then, each party generates its own secret key and combines it with the public key. These combined keys are exchanged. Each party then adds its own secret to the combined keys to arrive at a combined, secret key. This combined, secret key is then known to both parties, but not derivable by an eavesdropper.

The most common key exchange algorithm in use today is the **Diffie-Hellman key exchange algorithm**.

While key exchange algorithms are resistant against eavesdroppers, they are not resilient to interception. In the case of interception, an attacker could stand in the middle of a key exchange, while posing as each corresponding party. This is called a **man-in-the-middle attack**.

Asymmetric ciphers can be used to address some of these problems.

# Asymmetric ciphers

**Asymmetric encryption**, also known as **public-key encryption**, attempts to solve the key exchange and authentication problems of symmetric encryption. With asymmetric encryption, two keys are used. One key can encrypt data, while the other key can decrypt it. These keys are generated together and related mathematically. However, deriving one key from the other after the fact is not possible.

The following diagram shows an asymmetric cipher:

![](img/0ed9f4d0-7f5d-4e39-9a6c-3e3e5a476bbc.png)

Establishing a secure communication channel with asymmetric encryption is easier. Each party can generate its own asymmetric encryption keys. The encryption key can then be transmitted without the worry of interception, while the decryption key is kept private. In this scheme, these keys are referred to as the **Public Key** and the **Private Key**, respectively.

The **Rivest-Shamir-Adleman** (**RSA**) cipher is one of the first public-key ciphers and is widely used today. Newer **elliptic-curve cryptography** (**ECC**) algorithms promise greater efficiency and are quickly gaining market share. 

Asymmetric ciphers are also used to implement digital signatures. A digital signature is used to verify the authenticity of data. Digital signatures are created by using a private key to generate a signature for a document. The public key can then be used to verify that the document was signed by the holder of the private key.

The following diagram illustrates a digital signing and verifying process:

![](img/9657cca3-eed9-47bb-a65a-2ad088b23853.png)

TLS uses a combination of these methods to achieve security.

# How TLS uses ciphers

Digital signatures are essential in TLS; they are used to authenticate servers. Without digital signatures, a TLS client wouldn't be able to differentiate between an authentic server and an impostor.

TLS can also use digital signatures to authenticate the client, although this is much less common in practice. Most web applications either don't care about client authentication, or use other simpler methods, such as passwords.

In theory, asymmetric encryption could be used to protect an entire communication channel. However, in practice, modern asymmetric ciphers are inefficient and only able to protect small amounts of data. For this reason, symmetric ciphers are preferred whenever possible. TLS uses asymmetric ciphers only to authenticate the server. TLS uses a key exchange algorithm and symmetric ciphers to protect the actual communication.

Vulnerabilities are found for encryption algorithms all the time. It is, therefore, imperative that TLS connections are able to select the best algorithms that are mutually supported by both parties. This is done using **cipher suites**. A cipher suite is a list of algorithms, generally a **key exchange algorithm**, a **bulk encryption algorithm**, and a **message authentication algorithm** (**MAC**).

When a TLS connection is first established, the TLS client sends a list of preferred **cipher suites** to the server. The TLS server will select one of these cipher suites to be used for the connection. If the server doesn't support any of the cipher suites given by the client, then a secure TLS connection cannot be established.

With some background information about security out of the way, we can discuss TLS in more detail.

# The TLS protocol

After a TCP connection is established, the TLS handshake is initiated by the client. The client sends a number of specifications to the server, including which versions of SSL/TLS it is running, which cipher suites it supports, and which compression methods it would like to use.

The server selects the highest mutually supported version of SSL/TLS to use. It also chooses a cipher suite and compression method from the choices given by the client.

If the client and server do not support any cipher suite in common, then no TLS connection can be established. This is not uncommon when using very old browsers with newer servers.

After the basic setup is done, the server sends the client its certificate. This is used by the client to verify that it's connected to a legitimate server. We'll discuss more on certificates in the next section.

Once the client has verified that the server really is who it claims to be, a key exchange is initiated. After key exchange completes, both the client and server have a shared secret key. All further communication is encrypted using this key and their chosen symmetric cipher.

Certificates are used to verify server identities with digital signatures. Let's explore how they work next.

# Certificates

Each HTTPS server uses one or more certificates to verify their identity. This certificate must either be trusted by the client itself or trusted by a third party that the client trusts. In common usages, such as web browsers, it's not really possible to list all trusted certificates. For this reason, it's most common to validate certificates by verifying that a trusted third party trusts them. This trust is proven using digital signatures.

For example, a popular digital certificate authority is DigiCert Inc. Suppose that you trust DigiCert Inc. and have stored a certificate from them locally; you then connect to a website, `example.com`. You may not trust `example.com`, because you haven't seen their certificate before. However, `example.com` shows you that their certificate has been digitally signed by the DigiCert Inc. certificate you do trust. Therefore, you trust `example.com` website's certificate too.

In practice, certificate chains can be several layers deep. As long as you can verify digital signatures back to a certificate you trust, you are able to validate the whole chain.

This method is the common one used by HTTPS to authenticate servers. It does have some issues; namely, you must entirely trust the certificate authority. This is because certificate authorities could theoretically issue a certificate to an impostor, in which case you would be forced to trust the impostor. Certificate authorities are careful to avoid this, as it would ruin their reputation.

The most popular certificate authorities at the time of this writing are the following:

*   IdenTrust
*   Comodo
*   DigiCert
*   GoDaddy
*   GlobalSign

The five preceding certificate authorities are responsible for over 90% of the HTTPS certificates found on the web.

It is also possible to self-sign a certificate. In this case, no certificate authority is used. In these cases, a client needs to somehow reliably obtain and verify a copy of your certificate before it can be trusted.

Certificates are usually matched to domain names, but it's also possible for them to identify other information, such as company names, address, and so on.

In the next chapter, [Chapter 10](f57ffaa2-3eba-45cf-914b-bb6aef36174f.xhtml), *Implementing a Secure Web Server*, certificates are covered in more detail.

It's common today for one server to host many different domains, and each domain requires its own certificate. Let's now consider how these servers know which certificate to send.

# Server name identification

Many servers host multiple domains. Certificates are tied to domains; therefore, TLS must provide a method for the client to specify which domain it's connecting to. You may recall that the HTTP *Host* header servers this purpose. The problem is that the TLS connection should be established before the HTTP data is sent. Therefore, the server must decide which certificate to transmit before the HTTP *Host* header is received.

This is accomplished using **Server Name Indication** (**SNI**). SNI is a technique that, when used by TLS, requires the client to indicate to the server which domain it is attempting to connect to. The server can then find a matching certificate to use for the TLS connection.

SNI is relatively new, and older browsers and servers do not support it. Before SNI was popular, servers had two choices—they could either host only one domain per IP address, or they could send certificates for all hosted domains for each connection.

It should be noted that SNI involves sending the unencrypted domain name over the network. This means an eavesdropper can see which host the client is connecting to, even though they wouldn't know which resources the client is requesting from that host. Newer protocols, such as **encrypted server name identification** (**ESNI**), address this problem but are not widely deployed yet.

With a basic understanding of the TLS protocol, we're ready to look at the most popular library that implements it—OpenSSL.

# OpenSSL

**OpenSSL** is a widely used open source library that provides SSL and TLS services to applications. We use it in this chapter for secure connections required by HTTPS.

OpenSSL can be challenging to install. Refer to [Appendices B](47da8507-709b-44a6-9399-b18ce6afd8c9.xhtml), *Setting Up Your C Compiler on Windows*, [Appendices C](221eebc0-0bb1-4661-a5aa-eafed9fcba7e.xhtml), *Setting Up* *Your* *C Compiler on Linux*, and [Appendices D](632db68e-0911-4238-a2be-bd1aa5296120.xhtml), *Setting Up Your C Compiler on macOS,* for more information.

You can check whether you have the OpenSSL command-line tools installed by running the following command:

```cpp
openssl version
```

The following screenshot shows this on Ubuntu Linux:

![](img/29d33ad8-8b0d-491f-9ad6-4f850bfd74fa.png)

You'll also need to ensure that you have the OpenSSL library installed. The following program can be used to test this. If it compiles and runs successfully, then you do have the OpenSSL library installed and working:

```cpp
/*openssl_version.c*/

#include <openssl/ssl.h>

int main(int argc, char *argv[]) {
    printf("OpenSSL version: %s\n", OpenSSL_version(SSLEAY_VERSION));
    return 0;
}
```

If you're using an older version of OpenSSL, you may need to replace the `OpenSSL_version()` function call with `SSLeay_version()` instead. However, a better solution is to just upgrade to a newer OpenSSL version.

The preceding `openssl_version.c` program is compiled on macOS and Linux using the following command:

```cpp
gcc openssl_version.c -o openssl_version -lcrypto
./openssl_version
```

The following screenshot shows compiling and running `openssl_version.c`:

![](img/627f1ba2-5ed2-4fa7-99b3-a8215a27a85d.png)

On Windows, `openssl_version` can be compiled using MinGW and the following commands:

```cpp
gcc openssl_version.c -o openssl_version.exe -lcrypto
openssl_version.exe
```

Once the OpenSSL library is installed and usable, we are ready to begin using encrypted sockets.

# Encrypted sockets with OpenSSL

The TLS provided by OpenSSL can be applied to any TCP socket.

Before using OpenSSL in your program, it is important to initialize it. The following code does this:

```cpp
SSL_library_init();
OpenSSL_add_all_algorithms();
SSL_load_error_strings();
```

In the preceding code, the call to `SSL_library_init()` is required to initialize the OpenSSL library. The second call (to `OpenSSL_add_all_algorithms()`) causes OpenSSL to load all available algorithms. Alternately, you could load only the algorithms you know are needed. For our purposes, it is easy to load them all. The third call, `SSL_load_error_strings()`, causes OpenSSL to load error strings. This call isn't strictly needed, but it is handy to have easily readable error messages when something goes wrong.

Once OpenSSL is initialized, we are ready to create an SSL context. This is done by calling the `SSL_CTX_new()` function, which returns an `SSL_CTX` object. You can think of this object as a sort of factory for SSL/TLS connections. It holds the initial settings that you want to use for your connections. Most programs only need to create one `SSL_CTX` object, and they can reuse it for all their connections.

The following code creates the SSL context:

```cpp
SSL_CTX *ctx = SSL_CTX_new(TLS_client_method());
if (!ctx) {
    fprintf(stderr, "SSL_CTX_new() failed.\n");
    return 1;
}
```

The `SSL_CTX_new()` function takes one argument. We use `TLS_client_method()`, which indicates that we want general-purpose, version-flexible TLS methods available. Our client automatically negotiates the best mutually supported algorithm with the server upon connecting.

If you're using an older version of OpenSSL, you may need to replace `TLS_client_method()` with `TLSv1_2_client_method()` in the preceding code. However, a much better solution is to upgrade to a newer OpenSSL version.

To secure a TCP connection, you must first have a TCP connection. This TCP connection should be established in the normal way. The following pseudo-code shows this:

```cpp
getaddrinfo(hostname, port, hints, address);
socket = socket(address, type, protocol);
connect(socket, address, type);
```

For more information on setting up a TCP connection, refer to [Chapter 3](eb2e080f-fad2-4e14-b4c1-9a6c124df77c.xhtml), *An In-Depth Overview of TCP Connections*.

Once `connect()` has returned successfully, and a TCP connection is established, you can use the following code to initiate a TLS connection:

```cpp
SSL *ssl = SSL_new(ctx);
if (!ctx) {
    fprintf(stderr, "SSL_new() failed.\n");
    return 1;
}

if (!SSL_set_tlsext_host_name(ssl, hostname)) {
    fprintf(stderr, "SSL_set_tlsext_host_name() failed.\n");
    ERR_print_errors_fp(stderr);
    return 1;
}

SSL_set_fd(ssl, socket);
if (SSL_connect(ssl) == -1) {
    fprintf(stderr, "SSL_connect() failed.\n");
    ERR_print_errors_fp(stderr);
    return 1;
}
```

In the preceding code, `SSL_new()` is used to create an `SSL` object. This object is used to track the new SSL/TLS connection.

We then use `SSL_set_tlsext_host_name()` to set the domain for the server we are trying to connect to. This allows OpenSSL to use SNI. This call is optional, but without it, the server does not know which certificate to send in the case that it hosts more than one site.

Finally, we call `SSL_set_fd()` and `SSL_connect()` to initiate the new TLS/SSL connection on our existing TCP socket.

It is possible to see which cipher the TLS connection is using. The following code shows this:

```cpp
printf ("SSL/TLS using %s\n", SSL_get_cipher(ssl));
```

Once the TLS connection is established, data can be sent and received using `SSL_write()` and `SSL_read()`, respectively. These functions are used in a nearly identical manner as the standard socket `send()` and `recv()` functions.

The following example shows transmitting a simple message over a TLS connection:

```cpp
char *data = "Hello World!";
int bytes_sent = SSL_write(ssl, data, strlen(data));
```

Receiving data, done with `SSL_read()`, is shown in the following example:

```cpp
char read[4096];
int bytes_received = SSL_read(ssl, read, 4096);
printf("Received: %.*s\n", bytes_received, read);
```

When the connection is finished, it's important to free the used resources by calling `SSL_shutdown()` and `SSL_free(ssl)`. This is shown in the following code:

```cpp
SSL_shutdown(ssl);
CLOSESOCKET(socket);
SSL_free(ssl);
```

When you're done with an SSL context, you should also call `SSL_CTX_free()`. In our case, it looks like this:

```cpp
SSL_CTX_free(ctx);
```

If your program requires authentication of the connected peer, it is important to look at the certificates sent during the TLS initialization. Let's consider that next.

# Certificates

Once the TLS connection is established, we can use the `SSL_get_peer_certificate()` function to get the server's certificate. It's also easy to print the certificate subject and issuer, as shown in the following code:

```cpp
    X509 *cert = SSL_get_peer_certificate(ssl);
    if (!cert) {
        fprintf(stderr, "SSL_get_peer_certificate() failed.\n");
        return 1;
    }

    char *tmp;
    if (tmp = X509_NAME_oneline(X509_get_subject_name(cert), 0, 0)) {
        printf("subject: %s\n", tmp);
        OPENSSL_free(tmp);
    }

    if (tmp = X509_NAME_oneline(X509_get_issuer_name(cert), 0, 0)) {
        printf("issuer: %s\n", tmp);
        OPENSSL_free(tmp);
    }

    X509_free(cert);
```

OpenSSL automatically verifies the certificate during the TLS/SSL handshake. You can get the verification results using the `SSL_get_verify_result()` function. Its usage is shown in the following code:

```cpp
long vp = SSL_get_verify_result(ssl);
if (vp == X509_V_OK) {
    printf("Certificates verified successfully.\n");
} else {
    printf("Could not verify certificates: %ld\n", vp);
}
```

If `SSL_get_verify_result()` returns `X509_V_OK`, then the certificate chain was verified by OpenSSL and the connection can be trusted. If `SSL_get_verify_result()` does not return `X509_V_OK`, then HTTPS authentication has failed, and the connection should be abandoned.

In order for OpenSSL to successfully verify the certificate, we must tell it which certificate authorities we trust. This can be done by using the `SSL_CTX_load_verify_locations()` function. It must be passed the filename that stores all of the trusted root certificates. Assuming your trusted certificates are in `trusted.pem`, the following code sets this up:

```cpp
if (!SSL_CTX_load_verify_locations(ctx, "trusted.pem", 0)) {
    fprintf(stderr, "SSL_CTX_load_verify_locations() failed.\n");
    ERR_print_errors_fp(stderr);
    return 1;
}
```

Deciding which root certificates to trust isn't easy. Each operating system provides a list of trusted certificates, but there is no general, easy way to import these lists. Using the operating system's default list is also not always appropriate for every application. For these reasons, certificate verification has been omitted from the examples in this chapter. However, it is absolutely critical that it be implemented appropriately in order for your HTTPS connections to be secure.

In addition to validating the certificate's signatures, it is also important to validate that the certificate is actually valid for the particular server you're connected with! Newer versions of OpenSSL provide functions to help with this, but with older versions of OpenSSL, you're on your own. Consult the OpenSSL documentation for more information.

We've now covered enough background information about TLS and OpenSSL that we're ready to tackle a concrete example program.

# A simple HTTPS client

To bring the concepts of this chapter together, we build a simple HTTPS client. This client can connect to a given HTTPS web server and request the root document `/`. 

Our program begins by including the needed chapter header, defining `main()`, and initializing Winsock as shown here:

```cpp
/*https_simple.c*/

#include "chap09.h"

int main(int argc, char *argv[]) {

#if defined(_WIN32)
    WSADATA d;
    if (WSAStartup(MAKEWORD(2, 2), &d)) {
        fprintf(stderr, "Failed to initialize.\n");
        return 1;
    }
#endif
```

We then initialize the OpenSSL library with the following code:

```cpp
   /*https_simple.c continued*/

    SSL_library_init();
    OpenSSL_add_all_algorithms();
    SSL_load_error_strings();
```

The `SSL_load_error_strings()` function call is optional, but it's very useful if we run into problems.

We can also create an OpenSSL context. This is done by calling `SSL_CTX_new()` as shown by the following code:

```cpp
   /*https_simple.c continued*/ 

    SSL_CTX *ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) {
        fprintf(stderr, "SSL_CTX_new() failed.\n");
        return 1;
    }
```

If we were going to do certificate verification, this would be a good place to include the `SSL_CTX_load_verify_locations()` function call, as explained in the *Certificates* section of this chapter. We're omitting certification verification in this example for simplicity, but it is important to include it in real-world applications.

Our program then checks that a hostname and a port number was passed in on the command line like so:

```cpp
/*https_simple.c continued*/

    if (argc < 3) {
        fprintf(stderr, "usage: https_simple hostname port\n");
        return 1;
    }

    char *hostname = argv[1];
    char *port = argv[2];
```

The standard HTTPS port number is `443`. Our program lets the user specify any port number, which can be useful for testing.

We then configure the remote address for the socket connection. This code uses the same technique that we've been using since [Chapter 3](eb2e080f-fad2-4e14-b4c1-9a6c124df77c.xhtml), *An In-Depth Overview of TCP connections*. The code for this is as follows:

```cpp
/*https_simple.c continued*/

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

We continue to create the socket using a call to `socket()`, and we connect it using the `connect()` function as follows:

```cpp
/*https_simple.c continued*/

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
```

At this point, a TCP connection has been established. If we didn't need encryption, we could communicate over it directly. However, we are going to use OpenSSL to initiate a TLS/SSL connection over our TCP connection. The following code creates a new SSL object, sets the hostname for SNI, and initiates the TLS/SSL handshake:

```cpp
/*https_simple.c continued*/

    SSL *ssl = SSL_new(ctx);
    if (!ctx) {
        fprintf(stderr, "SSL_new() failed.\n");
        return 1;
    }

    if (!SSL_set_tlsext_host_name(ssl, hostname)) {
        fprintf(stderr, "SSL_set_tlsext_host_name() failed.\n");
        ERR_print_errors_fp(stderr);
        return 1;
    }

    SSL_set_fd(ssl, server);
    if (SSL_connect(ssl) == -1) {
        fprintf(stderr, "SSL_connect() failed.\n");
        ERR_print_errors_fp(stderr);
        return 1;
    }
```

The preceding code is explained in the *Encrypted Sockets with OpenSSL* section earlier in this chapter.

The call to `SSL_set_tlsext_host_name()` is optional, but useful if you may be connecting to a server that hosts multiple domains. Without this call, the server wouldn't know which certificates are relevant to this connection.

It is sometimes useful to know which cipher suite the client and server agreed upon. We can print the selected cipher suite with the following:

```cpp
/*https_simple.c continued*/

    printf("SSL/TLS using %s\n", SSL_get_cipher(ssl));
```

It is also useful to see the server's certificate. The following code prints the server's certificate:

```cpp
/*https_simple.c continued*/

    X509 *cert = SSL_get_peer_certificate(ssl);
    if (!cert) {
        fprintf(stderr, "SSL_get_peer_certificate() failed.\n");
        return 1;
    }

    char *tmp;
    if ((tmp = X509_NAME_oneline(X509_get_subject_name(cert), 0, 0))) {
        printf("subject: %s\n", tmp);
        OPENSSL_free(tmp);
    }

    if ((tmp = X509_NAME_oneline(X509_get_issuer_name(cert), 0, 0))) {
        printf("issuer: %s\n", tmp);
        OPENSSL_free(tmp);
    }

    X509_free(cert);
```

The certificate *subject* should match the domain we're connecting to. The issuer should be a certificate authority that we trust. Note that the preceding code does *not* validate the certificate. Refer to the *Certificates* section in this chapter for more information.

We can then send our HTTPS request. This request is the same as if we were using plain HTTP. We begin by formatting the request into a buffer and then sending it over the encrypted connection using `SSL_write()`. The following code shows this:

```cpp
/*https_simple.c continued*/

    char buffer[2048];

    sprintf(buffer, "GET / HTTP/1.1\r\n");
    sprintf(buffer + strlen(buffer), "Host: %s:%s\r\n", hostname, port);
    sprintf(buffer + strlen(buffer), "Connection: close\r\n");
    sprintf(buffer + strlen(buffer), "User-Agent: https_simple\r\n");
    sprintf(buffer + strlen(buffer), "\r\n");

    SSL_write(ssl, buffer, strlen(buffer));
    printf("Sent Headers:\n%s", buffer);
```

For more information about the HTTP protocol, please refer back to [Chapter 6](de3d2e9b-b94e-47d1-872c-c2ecb34c4026.xhtml), *Building a Simple Web Client*.

Our client now simply waits for data from the server until the connection is closed. This is accomplished by using `SSL_read()` in a loop. The following code receives the HTTPS response:

```cpp
/*https_simple.c continued*/

    while(1) {
        int bytes_received = SSL_read(ssl, buffer, sizeof(buffer));
            if (bytes_received < 1) {
                printf("\nConnection closed by peer.\n");
                break;
            }

            printf("Received (%d bytes): '%.*s'\n",
                    bytes_received, bytes_received, buffer);
    }
```

The preceding code also prints any data received over the HTTPS connection. Note that it does not parse the received HTTP data, which would be more complicated. See the `https_get.c` program in this chapter's code repository for a more advanced program that does parse the HTTP response.

Our simple client is almost done. We only have to shut down the TLS/SSL connection, close the socket, and clean up. This is done by the following code:

```cpp
/*https_simple.c continued*/

    printf("\nClosing socket...\n");
    SSL_shutdown(ssl);
    CLOSESOCKET(server);
    SSL_free(ssl);
    SSL_CTX_free(ctx);

#if defined(_WIN32)
    WSACleanup();
#endif

    printf("Finished.\n");
    return 0;
}
```

That concludes `https_simple.c`.

You should be able to compile and run it on Windows using MinGW and the following commands:

```cpp
gcc https_simple.c -o https_simple.exe -lssl -lcrypto -lws2_32
https_simple example.org 443
```

If you're using certain older versions of OpenSSL, you may also need an additional linker option—`-lgdi32`.

Compiling and executing on macOS and Linux can be done using the following:

```cpp
gcc https_simple.c -o https_simple -lssl -lcrypto
./https_simple example.org 443
```

If you run into linker errors, you should check that your OpenSSL library is properly installed. You may find it helpful to attempt compiling the `openssl_version.c` program first.

The following screenshot shows successfully compiling `https_simple.c` and using it to connect to `example.org`:

![](img/3b7332ca-c781-4e1f-85c3-0318b7b59384.png)

The `https_simple` program should serve as an elementary example of the techniques of connecting as an HTTPS client. These same techniques can be applied to any TCP connection.

It's easy to apply these techniques to some of the programs developed earlier in the book, such as `tcp_client` and `web_get`.

It is also worth mentioning that, while TLS works only with TCP connections, **Datagram Transport Layer** **Security** (**DTLS**) aims to provide many of the same guarantees for **User Datagram Protocol** (**UDP**) datagrams. OpenSSL provides support for DTLS too.

# Other examples

A few other examples are included in this chapter's repository. They are the following:

*   `tls_client.c`: This is `tcp_client.c` from [Chapter 3](eb2e080f-fad2-4e14-b4c1-9a6c124df77c.xhtml), *An In-Depth Overview of TCP Connections*, but it has been modified to make TLS connections.
*   `https_get.c`: This is the `web_get.c` program from [Chapter 6](de3d2e9b-b94e-47d1-872c-c2ecb34c4026.xhtml), *Building a Simple Web Client*, but it has been modified for HTTPS. You can think of it as the extended version of `https_simple.c`.
*   `tls_get_cert.c`: This is like `https_simple.c`, but it simply prints the connected server's certificate and exits.

Keep in mind that none of the examples in this chapter perform certification verification. This is an important step that must be added before using these programs in the field.

# Summary

In this chapter, we learned about the features that HTTPS provides over HTTP, such as authentication and encryption. We saw that HTTPS is really just HTTP over a TLS connection and that TLS can be applied to any TCP connection.

We also learned about basic encryption concepts. We saw how asymmetric ciphers use two keys, and how this allows for digital signatures. The very basics of certificates were covered, and we explored some of the difficulties with verifying them.

Finally, we worked through a concrete example that established a TLS connection to an HTTPS server.

This chapter was all about HTTPS clients, but in the next chapter, we focus on how HTTPS servers work.

# Questions

Try these questions to test your knowledge from this chapter:

1.  What port does HTTPS typically operate on?
2.  How many keys does symmetric encryption use?
3.  How many keys does asymmetric encryption use?
4.  Does TLS use symmetric or asymmetric encryption?
5.  What is the difference between SSL and TLS?
6.  What purpose do certificates fill?

Answers are in [Appendix A](bd8b8f52-52cb-4d34-b01b-e907564bfece.xhtml), *Answers to Questions*.

# Further reading

For more information about HTTPS and OpenSSL, please refer to the following:

*   OpenSSL documentation ([https://www.openssl.org/docs/](https://www.openssl.org/docs/))
*   **RFC 5246**: *The **Transport Layer Security** (**TLS**) Protocol* ([https://tools.ietf.org/html/rfc5246](https://tools.ietf.org/html/rfc5246))