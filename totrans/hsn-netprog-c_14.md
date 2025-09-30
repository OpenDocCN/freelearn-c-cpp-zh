# Establishing SSH Connections with libssh

This chapter is all about programming with the **Secure Shell** (**SSH**) protocol. SSH is a secure network protocol used to authenticate with remote servers, grant command-line access, and securely transfer files.

SSH is widely used for the configuration and management of remote servers. Oftentimes, web servers aren't connected to monitors or keyboards. For many of these servers, SSH provides the only method of command-line access and administration.

The following topics are covered in this chapter:

*   SSH protocol overview
*   The `libssh` library
*   Establishing a connection
*   SSH authentication methods
*   Executing a remote command
*   File transfers

# Technical requirements

The example programs from this chapter can be compiled using any modern C compiler. We recommend **MinGW** for Windows and **GCC** for Linux and macOS. You also need to have the `libssh` library installed. See [Appendix B](47da8507-709b-44a6-9399-b18ce6afd8c9.xhtml), *Setting Up Your C Compiler on Windows*, [Appendix C](221eebc0-0bb1-4661-a5aa-eafed9fcba7e.xhtml), *Setting Up Your C Compiler on Linux*, and [Appendix D](632db68e-0911-4238-a2be-bd1aa5296120.xhtml), *Setting Up Your C Compiler on macOS*, for compiler setup and `libssh` installation.

The code for this book can be found at [https://github.com/codeplea/Hands-On-Network-Programming-with-C](https://github.com/codeplea/Hands-On-Network-Programming-with-C).

From the command-line, you can download the code for this chapter by using the following command:

```cpp
git clone https://github.com/codeplea/Hands-On-Network-Programming-with-C
cd Hands-On-Network-Programming-with-C/chap11
```

Each example program in this chapter runs on Windows, Linux, and macOS.

Each example needs to be linked against the `libssh` library. This is accomplished by passing the `-lssh` option to `gcc`.

We provide the exact commands needed to compile each example as it is introduced.

For brevity, we use a standard header file with each example program in this chapter. This header includes the other needed headers in one place. Its contents are as follows:

```cpp
/*chap11.h*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libssh/libssh.h>
```

# The SSH protocol

Most servers providing a service (such as websites and emails) over the modern internet aren't attached to keyboards or monitors. Even when servers do have local input/output hardware, remote access is often much more convenient.

Various protocols have been used to provide remote command-line access to servers. One of the first such protocols was **Telnet**. With Telnet, a client remotely connects to a server using plaintext over TCP port `23`. The server provides more-or-less direct access to the operating system command-line through this **Transmission Control Protocol** (**TCP**) connection. The client sends plaintext commands to the server, and the server executes these commands. The command-line output is sent back from the server to the client.

Telnet has a major security shortcoming: it does not encrypt any data sent over the network. Even user passwords are sent as plaintext when using Telnet. This means that any network eavesdropper could obtain user credentials!

The SSH protocol has now largely replaced Telnet. The SSH protocol works over TCP using port `22`. SSH uses strong encryption to protect against eavesdropping.

SSH allows clients to verify servers' identities using **public-key authentication**. Without public-key authentication of the server, an impostor could masquerade as a legitimate server and attempt to trick a client into connecting. Once connected, the client would send its credentials to the impostor server.

SSH also provides many methods for client authentication with severs. These include sending a password or using public-key authentication. We look at these methods in detail later.

SSH is a complicated protocol. So, instead of attempting to implement it ourselves, we use an existing library to provide the needed functionality.

# libssh

`libssh` is a widely used open source C library implementing the SSH protocol. It allows us to remotely execute commands and transfer files using the SSH protocol.

`libssh` is structured in a way that abstracts network connections. We won't need to bother with the low-level networking APIs we've been using so far. The `libssh` library handles hostname resolution and creation of the needed TCP sockets for us.

# Testing out libssh

Before continuing with this chapter, it is essential that you have the `libssh` library installed and available. Please refer to [Appendix B](47da8507-709b-44a6-9399-b18ce6afd8c9.xhtml), *Setting Up Your C Compiler on Windows*, [Appendix C](221eebc0-0bb1-4661-a5aa-eafed9fcba7e.xhtml), *Setting Up Your C Compiler on Linux*, and [Appendix D](632db68e-0911-4238-a2be-bd1aa5296120.xhtml), *Setting Up Your C Compiler on macOS*, for `libssh` installation.

Our first program using `libssh` is designed to ensure that it's installed correctly. This program merely prints the `libssh` library version. The program is as follows:

```cpp
/*ssh_version.c*/

#include "chap11.h"

int main()
{
    printf("libssh version: %s\n", ssh_version(0));
    return 0;
}
```

You can compile and run `ssh_version.c` with the following commands on Windows using MinGW:

```cpp
gcc ssh_version.c -o ssh_version.exe -lssh
ssh_version
```

On Linux and macOS, the commands to compile and run `ssh_version.c` are as follows:

```cpp
gcc ssh_version.c -o ssh_version -lssh
./ssh_version
```

The following screenshot shows `ssh_version.c` being successfully compiled and run on Linux:

![](img/c1ff83a9-7bef-4aa7-83e9-bba24cedcf8e.png)

If you receive an error message about `libssh.h` not being found, you should check that you have the `libssh` library headers in your compiler's `include` directory search path. If you see an error message about an undefined reference to `ssh_version`, then please check that you didn't forget to pass the `-lssh` option to your compiler.

The next step to understanding `libssh` is to establish an actual SSH connection.

# Establishing a connection

Now that we've ensured that `libssh` is correctly installed, it's time to attempt an actual SSH connection.

You'll need to have access to an SSH server before continuing. OpenSSH is a popular server that is available for Linux, macOS, and Windows 10\. It works well for testing but be sure to understand the security implementations before installing it on your device. Refer to your operating system's documentation for more information.

If you would rather test with a remote system, Linux **Virtual Private Servers** (**VPS**) running OpenSSH are available from many providers. They typically cost only a few dollars a month.

Let's continue by implementing a program that uses `libssh` to open an SSH connection.

We structure the rest of the programs in this chapter to take the SSH server's hostname and port number as command-line arguments. Our program starts with the following code, which checks these arguments:

```cpp
/*ssh_connect.c*/

#include "chap11.h"

int main(int argc, char *argv[])
{
    const char *hostname = 0;
    int port = 22;
    if (argc < 2) {
        fprintf(stderr, "Usage: ssh_connect hostname port\n");
        return 1;
    }
    hostname = argv[1];
    if (argc > 2) port = atol(argv[2]);
```

In the preceding code, `argc` is checked to see whether at least the hostname was passed in as a command-line argument. If it wasn't, a usage message is displayed instead. Otherwise, the server's hostname is stored in the `hostname` variable. If a port number was passed in, it is stored in the `port` variable. Otherwise, the default port `22` is stored instead.

SSH often provides complete and total access to a server. For this reason, some internet criminals randomly scan IP addresses for SSH connections. When they successfully establish a connection, they attempt to guess login credentials, and if successful, they take control of the server. These attacks aren't successful against properly secured servers, but they are a common nuisance, nonetheless. Using SSH on a port other than the default (`22`) often avoids these automated attacks. This is one reason why we want to ensure our programs work well with non-default port numbers.

Once our program has obtained the hostname and connection port number, we continue by creating an SSH session object. This is done with a call to `ssh_new()` as shown in the following code:

```cpp
/*ssh_connect.c continued*/

    ssh_session ssh = ssh_new();
    if (!ssh) {
        fprintf(stderr, "ssh_new() failed.\n");
        return 1;
    }
```

The preceding code creates a new SSH session object and stores it in the `ssh` variable.

Once the SSH session is created, we need to specify some options before completing the connection. The `ssh_options_set()` function is used to set options. The following code shows setting the remote hostname and port:

```cpp
/*ssh_connect.c continued*/

    ssh_options_set(ssh, SSH_OPTIONS_HOST, hostname);
    ssh_options_set(ssh, SSH_OPTIONS_PORT, &port);
```

`libssh` includes useful debugging tools. By setting the `SSH_OPTIONS_LOG_VERBOSITY` option, we tell `libssh` to print almost everything it does. The following code causes `libssh` to log a lot of information about which actions it takes:

```cpp
/*ssh_connect.c continued*/

    int verbosity = SSH_LOG_PROTOCOL;
    ssh_options_set(ssh, SSH_OPTIONS_LOG_VERBOSITY, &verbosity);
```

This logging is useful to see, but it can also be distracting. I recommend you try it once and then disable it unless you run into problems. The rest of this chapter's examples won't use it.

We can now use `ssh_connect()` to initiate the SSH connection. The following code shows this:

```cpp
/*ssh_connect.c continued*/

    int ret = ssh_connect(ssh);
    if (ret != SSH_OK) {
        fprintf(stderr, "ssh_connect() failed.\n%s\n", ssh_get_error(ssh));
        return -1;
    }
```

Note that `ssh_connect()` returns `SSH_OK` on success. On failure, we use the `ssh_get_error()` function to detail what went wrong.

Next, our code prints out that the connection was successful:

```cpp
/*ssh_connect.c continued*/

    printf("Connected to %s on port %d.\n", hostname, port);
```

The SSH protocol allows servers to send a message to clients upon connecting. This message is called the **banner**. It is typically used to identify the server or provide short access rules. We can print the banner using the following code:

```cpp
/*ssh_connect.c continued*/

    printf("Banner:\n%s\n", ssh_get_serverbanner(ssh));
```

This is as far as our `ssh_connect.c` example goes. Our program simply disconnects and frees the SSH session before terminating. The following code concludes `ssh_connect.c`:

```cpp
/*ssh_connect.c continued*/

    ssh_disconnect(ssh);
    ssh_free(ssh);

    return 0;
}
```

You can compile `ssh_connect.c` with the following command on Windows using MinGW:

```cpp
gcc ssh_connect.c -o ssh_connect.exe -lssh
```

On Linux and macOS, the command to compile `ssh_connect.c` is as follows:

```cpp
gcc ssh_connect.c -o ssh_connect -lssh
```

The following screenshot shows `ssh_connect.c` being successfully compiled and run on Linux:

![](img/9e3681f9-10ad-44da-a8fd-a740fb004421.png)

In the preceding screenshot, you can see that `ssh_connect` was able to connect to the OpenSSH server running locally.

Now that we've established a connection, let's continue by authenticating with the server.

# SSH authentication

SSH provides authentication methods for both the server (host) and the client (user). It should be obvious why the server must authenticate the client. The server wants to only provide access to authorized users. Otherwise, anyone could take over the server.

However, the client also needs to authenticate the server. If the client fails to authenticate the server properly, then the client could be tricked into sending its password to an impostor!

In SSH, servers are authenticated using public key encryption. Conceptually, this is very similar to how HTTPS provides server authentication. However, SSH doesn't typically rely on certificate authorities. Instead, when using SSH, most clients simply keep a list of the public keys (or hashes of the public keys) that they trust. How the clients obtain this list in the first place varies. Generally, if a client connects to a server under trusted circumstances, then it can trust that public key in the future too.

`libssh` implements features to remember trusted servers' public keys. In this way, once a server has been connected to and trusted once, `libssh` remembers that it's trusted in the future.

Some SSH deployments also use other methods to validate SSH hosts' public keys. For example, a **Secure Shell fingerprint** (**SSHFP**) record is a type of DNS record used to validate SSH public keys. Its use requires secure DNS access.

Regardless of how you decide to trust (or not trust) a server's public key, you'll need to obtain the server's public key in the first place. Let's look at how `libssh` provides access to the server authentication functionality.

# Server authentication

Once the SSH session is established, we can get the server's public key using the `ssh_get_server_publickey()` function. The following code illustrates this function call:

```cpp
/*ssh_auth.c excerpt*/

    ssh_key key;
    if (ssh_get_server_publickey(ssh, &key) != SSH_OK) {
        fprintf(stderr, "ssh_get_server_publickey() failed.\n%s\n",
                ssh_get_error(ssh));
        return -1;
    }
```

It is often useful to obtain and display a hash of the server's SSH public key. Users can look at hashes and compare these to known keys. The `libssh` library provides the `ssh_get_publickey_hash()` function for this purpose.

The following code prints out an `SHA1` hash of the public key obtained earlier:

```cpp
/*ssh_auth.c excerpt*/

    unsigned char *hash;
    size_t hash_len;
    if (ssh_get_publickey_hash(key, SSH_PUBLICKEY_HASH_SHA1,
                &hash, &hash_len) != SSH_OK) {
        fprintf(stderr, "ssh_get_publickey_hash() failed.\n%s\n",
                ssh_get_error(ssh));
        return -1;
    }

    printf("Host public key hash:\n");
    ssh_print_hash(SSH_PUBLICKEY_HASH_SHA1, hash, hash_len);
```

`libssh` prints `SHA1` hashes using Base64\. It also prepends the hash type first. For example, the preceding code might print the following:

```cpp
Host public key hash:
SHA1:E348CMNeCGGec/bQqEX7aocDTfI
```

When you've finished with the public key and hash, free their resources with the following code:

```cpp
/*ssh_auth.c excerpt*/

    ssh_clean_pubkey_hash(&hash);
    ssh_key_free(key);
```

`libssh` provides the `ssh_session_is_known_server()` function to determine whether a server's public key is known. The following code shows an example of using this code:

```cpp
/*ssh_auth.c excerpt*/

    enum ssh_known_hosts_e known = ssh_session_is_known_server(ssh);
    switch (known) {
        case SSH_KNOWN_HOSTS_OK: printf("Host Known.\n"); break;

        case SSH_KNOWN_HOSTS_CHANGED: printf("Host Changed.\n"); break;
        case SSH_KNOWN_HOSTS_OTHER: printf("Host Other.\n"); break;
        case SSH_KNOWN_HOSTS_UNKNOWN: printf("Host Unknown.\n"); break;
        case SSH_KNOWN_HOSTS_NOT_FOUND: printf("No host file.\n"); break;

        case SSH_KNOWN_HOSTS_ERROR:
            printf("Host error. %s\n", ssh_get_error(ssh)); return 1;

        default: printf("Error. Known: %d\n", known); return 1;
    }
```

If the server's public key is known (previously trusted), then `ssh_session_is_known_server()` returns `SSH_KNOWN_HOSTS_OK`. Otherwise, `ssh_session_is_known_server()` can return other values with various meanings.

`SSH_KNOWN_HOSTS_UNKNOWN` indicates that the server is unknown. In this case, the user should verify the server's hash.

`SSH_KNOWN_HOSTS_NOT_FOUND` means that `libssh` didn't find a hosts file, and one is created automatically. This should generally be treated in the same way as `SSH_KNOWN_HOSTS_UNKNOWN`.

`SSH_KNOWN_HOSTS_CHANGED` indicates that the server is returning a different key than was previously known, while `SSH_KNOWN_HOSTS_OTHER` indicates that the server is returning a different type of key than was previously used. Either of these could indicate a potential attack! In a real-world application, you should be more explicit about notifying the user of these risks.

If the user has verified that a host is to be trusted, use `ssh_session_update_known_hosts()` to allow `libssh` to save the servers public key hash. This allows `ssh_session_is_known_server()` to return `SSH_KNOWN_HOSTS_OK` for the next connection.

The following code illustrates prompting the user to trust a connection and using `ssh_session_update_known_hosts()`:

```cpp
/*ssh_auth.c excerpt*/

    if (known == SSH_KNOWN_HOSTS_CHANGED ||
            known == SSH_KNOWN_HOSTS_OTHER ||
            known == SSH_KNOWN_HOSTS_UNKNOWN ||
            known == SSH_KNOWN_HOSTS_NOT_FOUND) {
        printf("Do you want to accept and remember this host? Y/N\n");
        char answer[10];
        fgets(answer, sizeof(answer), stdin);
        if (answer[0] != 'Y' && answer[0] != 'y') {
            return 0;
        }

        ssh_session_update_known_hosts(ssh);
    }
```

Please see `ssh_auth.c` in this chapter's code repository for a working example. Consult the `libssh` documentation for more information.

After the client has authenticated the server, the server needs to authenticate the client.

# Client authentication

SSH offers several methods of client authentication. These methods include the following:

*   **No authentication**: This allows any user to connect
*   **Password authentication**: This requires the user to provide a username and password
*   **Public key**: This uses public key encryption methods to authenticate
*   **Keyboard-interactive**: This authenticates by having the user answer several prompts
*   **Generic Security Service Application Program Interface** (**GSS-API**): This allows authentication through a variety of other services

Password authentication is the most common method, but it does have some drawbacks. If an impostor server tricks a user into sending their password, then that user's password is effectively compromised. Public key user authentication doesn't suffer from this attack to the same degree. With public key authentication, the server issues a unique challenge for each authentication attempt. This prevents a malicious impostor server from replaying a previous authentication to the legitimate server.

Once public key authentication is set up, `libssh` makes using it very simple. In many cases, it's as easy as calling the `ssh_userauth_publickey_auto()` function. However, setting up public key authentication in the first place can be a tedious process.

Although public key authentication is more secure, password authentication is still in common use. Password authentication is also more straightforward and easier to test. For these reasons, we continue the examples in this chapter by using password authentication.

Regardless of the user authentication method, the SSH server must know what user you are trying to authenticate as. The `libssh` library lets us provide this information using the `ssh_set_options()` function that we saw earlier. It should be called before using `ssh_connect()`. To set the user, call `ssh_options_set()` with `SSH_OPTIONS_USER` as shown in the following code:

```cpp
ssh_options_set(ssh, SSH_OPTIONS_USER, "alice");
```

After the SSH session has been established, a password can be provided with the `ssh_userauth_password()` function. The following code prompts for a password and sends it to the connected SSH server:

```cpp
/*ssh_auth.c excerpt*/

    printf("Password: ");
    char password[128];
    fgets(password, sizeof(password), stdin);
    password[strlen(password)-1] = 0;

    if (ssh_userauth_password(ssh, 0, password) != SSH_AUTH_SUCCESS) {
        fprintf(stderr, "ssh_userauth_password() failed.\n%s\n",
                ssh_get_error(ssh));
        return 0;
    } else {
        printf("Authentication successful!\n");
    }
```

The preceding code uses the `fgets()` function to obtain the password from the user. The `fgets()` function always includes the newline character with the input, which we don't want. The `password[strlen(password)-1] = 0` code effectively shortens the password by one character, thus removing the newline character.

Note that using `fgets()` causes the entered password to display on the screen. This isn't secure, and it would be an improvement to hide the password while it's being entered. Unfortunately, there isn't a cross-platform way to do this. If you're using Linux, consider using the `getpass()` function in place of `fgets()`.

See `ssh_auth.c` in this chapter's code repository for a working example of authenticating with a server using user password authentication.

You can compile and run `ssh_auth.c` with the following commands on Windows using MinGW:

```cpp
gcc ssh_auth.c -o ssh_auth.exe -lssh
ssh_auth example.com 22 alice
```

On Linux and macOS, the commands to compile and run `ssh_auth.c` are as follows:

```cpp
gcc ssh_auth.c -o ssh_auth -lssh
./ssh_auth example.com 22 alice
```

The following screenshot shows compiling `ssh_auth` and using it to connect to a locally running SSH server on Linux:

![](img/3e14a7e9-b149-4d92-a352-3f3cb3a250fd.png)

In the preceding screenshot, `ssh_auth` was used to successfully authenticate with the locally running SSH server. The `ssh_auth` program used password authentication with the username `alice` and the password `password123`. Needless to say, you need to change the username and password as appropriate for your SSH server. Authentication will be successful only if you use the username and password for an actual user account on the server you connect to.

After authenticating, we're ready to run a command over SSH.

# Executing a remote command

The SSH protocol works using channels. After we've established an SSH connection, a channel must be opened to do any real work. The advantage is that many channels can be opened over one connection. This potentially allows an application to do multiple things (seemingly) simultaneously.

After the SSH session is open and the user is authenticated, a channel can be opened. A new channel is opened by calling the `ssh_channel_new()` function. The following code illustrates this:

```cpp
/*ssh_command.c excerpt*/

    ssh_channel channel = ssh_channel_new(ssh);
    if (!channel) {
        fprintf(stderr, "ssh_channel_new() failed.\n");
        return 0;
    }
```

The SSH protocol implements many types of channels. The **session** channel type is used for executing remote commands and transferring files. With `libssh`, we can request a session channel by using the `ssh_channel_open_session()` function. The following code shows calling `ssh_channel_open_session()`:

```cpp
/*ssh_command.c excerpt*/

    if (ssh_channel_open_session(channel) != SSH_OK) {
        fprintf(stderr, "ssh_channel_open_session() failed.\n");
        return 0;
    }
```

Once the session channel is open, we can issue a command to run with the `ssh_channel_request_exec()` function. The following code uses `fgets()` to prompt the user for a command and `ssh_channel_request_exec()` to send the command to the remote host:

```cpp
/*ssh_command.c excerpt*/

    printf("Remote command to execute: ");
    char command[128];
    fgets(command, sizeof(command), stdin);
    command[strlen(command)-1] = 0;

    if (ssh_channel_request_exec(channel, command) != SSH_OK) {
        fprintf(stderr, "ssh_channel_open_session() failed.\n");
        return 1;
    }
```

Once the command has been sent, our program uses `ssh_channel_read()` to receive the command output. The following code loops until the entire output is read:

```cpp
/*ssh_command.c excerpt*/

    char output[1024];
    int bytes_received;
    while ((bytes_received =
                ssh_channel_read(channel, output, sizeof(output), 0))) {
        if (bytes_received < 0) {
            fprintf(stderr, "ssh_channel_read() failed.\n");
            return 1;
        }
        printf("%.*s", bytes_received, output);
    }
```

The preceding code first allocates a buffer, `output`, to hold the received data from the command's output. The `ssh_channel_read()` function returns the number of bytes read, but it returns `0` when the read is complete or a negative number for an error. Our code loops while `ssh_channel_read()` returns data.

After the entire output from the command has been received, the client should send an **end-of-file** (**EOF**) over the channel, close the channel, and free the channel resources. The following code shows this:

```cpp
/*ssh_command.c excerpt*/

    ssh_channel_send_eof(channel);
    ssh_channel_close(channel);
    ssh_channel_free(channel);
```

If your program is also done with the SSH session, be sure to call `ssh_disconnect()` and `ssh_free()` as well.

The `ssh_command.c` program included in this chapter's code repository is a simple utility that connects to a remote SSH host and executes a single command.

You can compile `ssh_command.c` with the following command on Windows using MinGW:

```cpp
gcc ssh_command.c -o ssh_command.exe -lssh
```

On Linux and macOS, the command to compile `ssh_command.c` is as follows:

```cpp
gcc ssh_command.c -o ssh_command -lssh
```

The following screenshot shows `ssh_command.c` being compiled and run on Linux:

![](img/63f94273-810d-45e6-bf34-82c3f88985a8.png)

The preceding screenshot shows connecting to the local OpenSSH server and executing the `ls -l` command. The `ssh_command` code faithfully prints the output of that command (which is a file listing for the user's home directory).

The `libssh` library function `ssh_channel_request_exec()` is useful to execute a single command. However, SSH also supports methods for opening a fully interactive remote shell. Generally, a session channel is opened as shown previously. Then the `libssh` library function `ssh_channel_request_pty()` is called to initialize a remote shell. The `libssh` library provides many functions to send and receive data this way. Please refer to the `libssh` documentation for more information.

Now that you're able to execute a remote command and receive its output, it may also be useful to transfer files. Let's consider that next.

# Downloading a file

The **Secure Copy Protocol** (**SCP**) provides a method to transfer files. It supports both uploading and downloading files.

`libssh` makes using SCP easy. This chapter's code repository contains an example, `ssh_download.c`, which shows the basic method for downloading a file over SCP with `libssh`.

After the SSH session is started and the user is authenticated, `ssh_download.c` prompts the user for the remote filename using the following code:

```cpp
/*ssh_download.c excerpt*/

    printf("Remote file to download: ");
    char filename[128];
    fgets(filename, sizeof(filename), stdin);
    filename[strlen(filename)-1] = 0;
```

A new SCP session is initialized by calling the `libssh` library function `ssh_scp_new()`, as shown in the following code:

```cpp
/*ssh_download.c excerpt*/

    ssh_scp scp = ssh_scp_new(ssh, SSH_SCP_READ, filename);
    if (!scp) {
        fprintf(stderr, "ssh_scp_new() failed.\n%s\n",
                ssh_get_error(ssh));
        return 1;
    }
```

In the preceding code, `SSH_SCP_READ` is passed to `ssh_scp_new()`. This specifies that we are going to use the new SCP session for downloading files. The `SSH_SCP_WRITE` option would be used to upload files. The `libssh` library also provides the `SSH_SCP_RECURSIVE` option to assist with uploading or downloading entire directory trees.

After the SCP session is created successfully, `ssh_scp_init()` must be called to initialize the SCP channel. The following code shows this:

```cpp
/*ssh_download.c excerpt*/

    if (ssh_scp_init(scp) != SSH_OK) {
        fprintf(stderr, "ssh_scp_init() failed.\n%s\n",
                ssh_get_error(ssh));
        return 1;
    }
```

`ssh_scp_pull_request()` must be called to begin the file download. This function returns `SSH_SCP_REQUEST_NEWFILE` to indicate that the remote host is going to begin sending a new file. The following code shows this:

```cpp
/*ssh_download.c excerpt*/

    if (ssh_scp_pull_request(scp) != SSH_SCP_REQUEST_NEWFILE) {
        fprintf(stderr, "ssh_scp_pull_request() failed.\n%s\n",
                ssh_get_error(ssh));
        return 1;
    }
```

`libssh` provides some methods we can use to retrieve the remote filename, file size, and permissions. The following code retrieves these values and prints them to the console:

```cpp
/*ssh_download.c excerpt*/

    int fsize = ssh_scp_request_get_size(scp);
    char *fname = strdup(ssh_scp_request_get_filename(scp));
    int fpermission = ssh_scp_request_get_permissions(scp);

    printf("Downloading file %s (%d bytes, permissions 0%o\n",
            fname, fsize, fpermission);
    free(fname);
```

Once the file size is known, we can allocate space to store it in memory. This is done using `malloc()` as shown in the following code:

```cpp
/*ssh_download.c excerpt*/

    char *buffer = malloc(fsize);
    if (!buffer) {
        fprintf(stderr, "malloc() failed.\n");
        return 1;
    }
```

Our program then accepts the new file request with `ssh_scp_accept_request()` and downloads the file with `ssh_scp_read()`. The following code shows this:

```cpp
/*ssh_download.c excerpt*/

    ssh_scp_accept_request(scp);
    if (ssh_scp_read(scp, buffer, fsize) == SSH_ERROR) {
        fprintf(stderr, "ssh_scp_read() failed.\n%s\n",
                ssh_get_error(ssh));
        return 1;
    }
```

The downloaded file can be printed to the screen with a simple call to `printf()`. When we're finished with the file data, it's important to free the allocated space too. The following code prints out the file's contents and frees the allocated memory:

```cpp
/*ssh_download.c excerpt*/

    printf("Received %s:\n", filename);
    printf("%.*s\n", fsize, buffer);
    free(buffer);
```

An additional call to `ssh_scp_pull_request()` should return `SSH_SCP_REQUEST_EOF`. This indicates that we received the entire file from the remote host. The following code checks for the end-of-file request from the remote host:

```cpp
/*ssh_download.c excerpt*/

    if (ssh_scp_pull_request(scp) != SSH_SCP_REQUEST_EOF) {
        fprintf(stderr, "ssh_scp_pull_request() unexpected.\n%s\n",
                ssh_get_error(ssh));
        return 1;
    }
```

The preceding code is simplified a bit. The remote host could also return other values which aren't necessarily errors. For example, if `ssh_scp_pull_request()` returns `SSH_SCP_REQUEST_WARNING`, then the remote host has sent a warning. This warning can be read by calling `ssh_scp_request_get_warning()`, but, in any case, `ssh_scp_pull_request()` should be called again.

After the file is received, `ssh_scp_close()` and `ssh_scp_free()` should be used to free resources as shown in the following code excerpt:

```cpp
/*ssh_download.c excerpt*/

    ssh_scp_close(scp);
    ssh_scp_free(scp);
```

After your program is done with the SSH session, don't forget to call `ssh_disconnect()` and `ssh_free()` as well.

The entire file-downloading example is included with this chapter's code as `ssh_download.c`.

You can compile `ssh_download.c` with the following command on Windows using MinGW:

```cpp
gcc ssh_download.c -o ssh_download.exe -lssh
```

On Linux and macOS, the command to compile `ssh_download.c` is as follows:

```cpp
gcc ssh_download.c -o ssh_download -lssh
```

The following screenshot shows `ssh_download.c` being successfully compiled and used on Linux to download a file:

![](img/45e07451-d9e8-43ce-8c4a-0f2c22ce53d9.png)

As you can see from the preceding screenshot, downloading a file using SSH and SCP is pretty straightforward. This can be a useful way to transfer data between computers securely.

# Summary

This chapter provided a cursory overview of the SSH protocol and how to use it with `libssh`. We learned a lot about authentication with the SSH protocol, and how the server and client must both authenticate for security. Once the connection was established, we implemented a simple program to execute a command on a remote host. We also saw how `libssh` makes downloading a file using SCP very easy.

SSH provides a secure communication channel, which effectively denies eavesdroppers the meaning of intercepted communication.

In the next chapter, [Chapter 12](1d616e6f-d234-4269-8507-f007ffc1b7d0.xhtml), *Network Monitoring and Security*, we continue with the security theme by looking at tools that can effectively eavesdrop on non-secure communication channels.

# Questions

Try these questions to test your knowledge from this chapter:

1.  What is a significant downside of using Telnet?
2.  Which port does SSH typically run on?
3.  Why is it essential that the client authenticates the SSH server?
4.  How is the server typically authenticated?
5.  How is the SSH client typically authenticated?

The answers to this questions can be found in [Appendix A](bd8b8f52-52cb-4d34-b01b-e907564bfece.xhtml), *Answers to Questions*.

# Further reading

For more information about Telnet, SSH, and `libssh`, please refer to the following:

*   The SSH library [https://www.libssh.org](https://www.libssh.org)
*   **RFC 15**: *Network Subsystem for Time Sharing Hosts* (Telnet) ([https://tools.ietf.org/html/rfc15](https://tools.ietf.org/html/rfc15))
*   **RFC 855**: *Telnet Option Specifications* ([https://tools.ietf.org/html/rfc855](https://tools.ietf.org/html/rfc855))
*   **RFC 4250**: *The **Secure Shell** (**SSH**) Protocol Assigned Numbers* ([https://tools.ietf.org/html/rfc4250](https://tools.ietf.org/html/rfc4250))
*   **RFC 4251**: *The **Secure Shell** (**SSH**) Protocol Architecture* ([https://tools.ietf.org/html/rfc4251](https://tools.ietf.org/html/rfc4251))
*   **RFC 4252**: *The **Secure Shell** (**SSH**) Authentication Protocol* ([https://tools.ietf.org/html/rfc4252](https://tools.ietf.org/html/rfc4252))
*   **RFC 4253**: *The **Secure Shell** (**SSH**) Transport Layer Protocol* ([https://tools.ietf.org/html/rfc4253](https://tools.ietf.org/html/rfc4253))
*   **RFC 4254**: *The **Secure Shell** (**SSH**) Connection Protocol* ([https://tools.ietf.org/html/rfc4254](https://tools.ietf.org/html/rfc4254))
*   **RFC 4255**: *Using DNS to Securely Publish **Secure Shell** (**SSH**) Key Fingerprints* ([https://tools.ietf.org/html/rfc4255](https://tools.ietf.org/html/rfc4255))
*   **RFC 4256**: *Generic Message Exchange Authentication for the **Secure Shell** Protocol (**SSH**)* ([https://tools.ietf.org/html/rfc4256](https://tools.ietf.org/html/rfc4256))