# Networking and Interprocess Communication

Processes run individually and work independently in their respective address spaces. However, they sometimes need to communicate with each other to pass on information. For processes to cooperate, they need to be able to communicate with each other as well as synchronize their actions. Here are the types of communication that take place between processes:

*   **Synchronous communication**: Such communication doesn't allow the process to continue with any other work until the communication is over
*   **Asynchronous communication**: In this communication, the process can continue doing other tasks, and so it supports multitasking and results in better efficiency
*   **Remote Procedure Call** (**RPC**): This is a protocol that uses client service techniques for communication where the client cannot do anything, that is, it is suspended until it gets a response from the server

These communications can be unidirectional or bidirectional. To enable any form of communication between processes, the following popular **interprocess communication** (**IPC**) mechanisms are used: pipes, FIFOs (named pipes), sockets, message queues, and shared memory. Pipes and FIFO enable unidirectional communication, whereas sockets, message queues, and shared memory enable bidirectional communication. 

In this chapter, we will learn how to make the following recipes so that we can establish communication between processes:

*   Communicating between processes using pipes
*   Communicating between processes using FIFO
*   Communicating between the client and server using socket programming
*   Communicating between processes using a UDP socket
*   Passing a message from one process to another using the message queue
*   Communicating between processes using shared memory

Let's begin with the first recipe!

# Communicating between processes using pipes

In this recipe, we will learn how to write data into a pipe from its writing end and then how to read that data from its reading end. This can happen in two ways:

*   One process, both writing and reading from the pipe
*   One process writing and another process reading from the pipe

Before we begin with the recipes, let's quickly review the functions, structures, and terms that are used in successful interprocess communication. 

# Creating and to connecting processes

The most commonly used functions and terms for communication between processes are `pipe`, `mkfifo`, `write`, `read`, `perror`, and `fork.`

# pipe()

A pipe is used for connecting two processes. The output from one process can be sent as an input to another process. The flow is unidirectional, that is, one process can write to the pipe and another process can read from the pipe. Writing and reading are done in an area of main memory, which is also known as a virtual file. Pipes have a **First in First out** (**FIFO**) or a queue structure, that is, what is written first will be read first.

A process should not try to read from the pipe before something is written into it, otherwise it will suspend until something is written into the pipe.

Here is its syntax:

```cpp
int pipe(int arr[2]);
```

Here, `arr[0]` is the file descriptor for the read end of the pipe, and `arr[1]` is the file descriptor for the write end of the pipe.

The function returns `0` on success and `-1` on error.

# mkfifo()

This function creates a new FIFO special file. Here is its syntax:

```cpp
int mkfifo(const char *filename, mode_t permission);
```

Here, `filename` represents the filename, along with its complete path, and `permission` represents the permission bits of the new FIFO file. The default permissions are read and write permission for the owner, group, and others, that is, (0666).

The function returns `0` on successful completion; otherwise, it returns `-1`.

# write()

This function is used for writing into the specified file or pipe whose descriptor is supplied. Here is its syntax:

```cpp
write(int fp, const void *buf, size_t n);
```

It writes the *n* number of bytes into the file that's being pointed to by the file pointer, `fp`, from the buffer, `buf`.

# read()

This function reads from the specified file or pipe whose descriptor is supplied in the method. Here is its syntax:

```cpp
read(int fp, void *buf, size_t n);
```

It tries to read up to *n* number of bytes from a file that's being pointed to by a descriptor, `fp`. The bytes that are read are then assigned to the buffer, `buf`.

# perror()

This displays an error message indicating the error that might have occurred while invoking a function or system call. The error message is displayed to `stderr`, that is, the standard error output stream. This is basically the console. 

Here is its syntax:

```cpp
void perror ( const char * str );
```

The error message that is displayed is optionally preceded by the message that's represented by `str`.

# fork()

This is used for creating a new process. The newly created process is called the child process, and it runs concurrently with the parent process. After executing the `fork` function, the execution of the program continues and the instruction following the `fork` function is executed by the parent as well as the child process. If the system call is successful, it will return a process ID of the child process and returns a `0` to the newly created child process. The function returns a negative value if the child process is not created.

Now, let's start with the first recipe for enabling communication between processes using pipes. 

# One process, both writing and reading from the pipe

Here, we will learn how writing and reading from the pipe are done by a single process.

# How to do it…

1.  Define an array of size `2` and pass it as an argument to the `pipe` function.
2.  Invoke the `write` function and write your chosen string into the pipe through the `write` end of the array. Repeat the procedure for the second message.
3.  Invoke the `read` function to read the first message from the pipe. Invoke the `read` function again to read the second message.

The `readwritepipe.c` program for writing into the pipe and reading from it thereafter is as follows:

```cpp
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#define max 50

int main()
{
    char str[max];
    int pp[2];

    if (pipe(pp) < 0)
        exit(1);
    printf("Enter first message to write into pipe: ");
    gets(str);
    write(pp[1], str, max);
    printf("Enter second message to write into pipe: ");
    gets(str);
    write(pp[1], str, max);
    printf("Messages read from the pipe are as follows:\n");
    read(pp[0], str, max);
    printf("%s\n", str);
    read(pp[0], str, max);
    printf("%s\n", str);
    return 0;
}
```

Let's go behind the scenes.

# How it works...

We defined a macro, `max`, of size of `50`, a string, `str`, of size `max`, and an array, `pp`, with size `2` . We will invoke the `pipe` function to connect two processes and pass the `pp` array to it. The index location, `pp[0]`, will get the file descriptor for the reading end of the pipe and `pp[1]` will get the file descriptor for the write end of the pipe. The program will exit if the `pipe` function does not execute successfully.

You will be prompted to enter the first message to be written into the pipe. The text that's entered by you will be assigned to the string variable, `str`. Invoke the `write` function and the string in `str` will be written into the pipe, `pp`. Repeat the procedure for the second message. The second text that's entered by you will also be written into the pipe.

Obviously, the second text will be written behind the first text in the pipe. Now, invoke the `read` function to read from the pipe. The text that was entered first in the pipe will be read and assigned to the string variable, `str`, and is consequently displayed on the screen. Again, invoke the `read` function and the second text message in the pipe will be read from its read end and assigned to the string variable, `str`, and then displayed on the screen.

Let's use GCC to compile the `readwritepipe.c` program, as follows:

```cpp
$ gcc readwritepipe.c -o readwritepipe
```

If you get no errors or warnings, this means that the `readwritepipe.c` program has been compiled into an executable file, `readwritepipe.exe`. Let's run this executable file:

```cpp
$ ./readwritepipe
Enter the first message to write into pipe: This is the first message for the pipe
Enter the second message to write into pipe: Second message for the pipe
Messages read from the pipe are as follows:
This is the first message for the pipe
Second message for the pipe
```

In the preceding program, the main thread does the job of writing and reading from the pipe. But what if we want one process to write into the pipe and another process to read from the pipe?  Let's find out how we can make that happen.

# One process writing into the pipe and another process reading from the pipe

In this recipe, we will use the fork system call to create a child process. Then, we will write into the pipe using the child process and read from the pipe through the parent process, thereby establishing communication between two processes.

# How to do it…

1.  Define an array of size `2`.
2.  Invoke the `pipe` function to connect the two processes and pass the array we defined previously to it.
3.  Invoke the `fork` function to create a new child process.
4.  Enter the message that is going to be written into the pipe. Invoke the `write` function using the newly created child process.
5.  The parent process invokes the `read` function to read the text that's been written into the pipe.

The `pipedemo.c` program for writing into the pipe through a child process and reading from the pipe through the parent process is as follows:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define max  50

int main()
{
    char wstr[max];
    char rstr[max];
    int pp[2];
    pid_t p;
    if(pipe(pp) < 0)
    {
        perror("pipe");
    } 
    p = fork();
    if(p >= 0)
    {
        if(p == 0)
        {
            printf ("Enter the string : ");
            gets(wstr);
            write (pp[1] , wstr , strlen(wstr));
            exit(0);
        }
        else
        {
            read (pp[0] , rstr , sizeof(rstr));
            printf("Entered message : %s\n " , rstr);
            exit(0);
        }
    }
    else
    {
        perror("fork");
        exit(2);
    }        
    return 0;
}
```

Let's go behind the scenes.

# How it works...

Define a macro `max`, of size `50` and two string variables, `wstr` and `rstr`, of size `max`. The `wstr` string will be used for writing into the pipe and `rstr` will be used for reading from the pipe. Define an array, `pp`, of size `2`, which will be used for storing the file descriptors of the read and write ends of the pipe. Define a variable, `p`, of the `pid_t` data type, which will be used for storing a process ID.

We will invoke the `pipe` function to connect the two processes and pass the `pp` array to it. The index location `pp[0]` will get the file descriptor for the reading end of the pipe, while `pp[1]` will get the file descriptor for the write end of the pipe. The program will exit if the `pipe` function does not execute successfully.

Then, we will invoke the `fork` function to create a new child process. You will be prompted to enter the message to be written into the pipe. The text you enter will be assigned to the string variable `wstr`. When we invoke the `write` function using the newly created child process, the string in the `wstr` variable will be written into the pipe, `pp`. Thereafter, the parent process will invoke the `read` function to read the text that's been written into the pipe. The text that's read from the pipe will be assigned to the string variable `rstr` and will consequently be displayed on the screen.

Let's use GCC to compile the `pipedemo.c` program, as follows:

```cpp
$ gcc pipedemo.c -o pipedemo
```

If you get no errors or warnings, this means that the `pipedemo.c` program has been compiled into an executable file, `pipedemo.exe`. Let's run this executable file:

```cpp
$ ./pipedemo
Enter the string : This is a message from the pipe
Entered message : This is a message from the pipe
```

Voila! We've successfully communicated between processes using pipes. Now, let's move on to the next recipe!

# Communicating between processes using FIFO

In this recipe, we will learn how two processes communicate using a named pipe, also known as FIFO. This recipe is divided into the following two parts:

*   Demonstrating how data is written into a FIFO
*   Demonstrating how data is read from a FIFO

The functions and terms we learned in the previous recipe will also be applicable here.

# Writing data into a FIFO

As the name suggests, we will learn how data is written into a FIFO in this recipe.

# How to do it…

1.  Invoke the `mkfifo` function to create a new FIFO special file.
2.  Open the FIFO special file in write-only mode by invoking the `open` function.
3.  Enter the text to be written into the FIFO special file.
4.  Close the FIFO special file.

The `writefifo.c` program for writing into a FIFO is as follows:

```cpp
#include <stdio.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int main()
{
    int fw;
    char str[255];
    mkfifo("FIFOPipe", 0666);
    fw = open("FIFOPipe", O_WRONLY);
    printf("Enter text: ");
    gets(str);
    write(fw,str, sizeof(str));
    close(fw);
    return 0;
}
```

Let's go behind the scenes.

# How it works...

Let's assume we have defined a string called `str` of size `255`. We will invoke the `mkfifo` function to create a new FIFO special file. We will create the FIFO special file with the name `FIFOPipe` with read and write permissions for owner, group, and others.

We will open this FIFO special file in write-only mode by invoking the `open` function. Then, we will assign the file descriptor of the opened FIFO special file to the `fw` variable. You will be prompted to enter the text that is going to be written into the file. The text you enter will be assigned to the `str` variable, which in turn will be written into the special FIFO file when you invoke the `write` function. Finally, close the FIFO special file.

Let's use GCC to compile the `writefifo.c` program, as follows:

```cpp
$ gcc writefifo.c -o writefifo
```

If you get no errors or warnings, this means that the `writefifo.c` program has compiled into an executable file, `writefifo.exe`. Let's run this executable file:

```cpp
$ ./writefifo
Enter text: This is a named pipe demo example called FIFO
```

If your program does not prompts for the string that means it is waiting for the other end of the FIFO to open. That is, you need to run the next recipe, *Reading data from a FIFO,* on the second Terminal screen. Please press *Alt+F2* on Cygwin to open the next terminal screeen.

Now, let's check out the other part of this recipe.

# Reading data from a FIFO

In this recipe, we will see how we can read data from a FIFO.

# How to do it…

1.  Open the FIFO special file in read-only mode by invoking the `open` function.
2.  Read the text from the FIFO special file using the `read` function.
3.  Close the FIFO special file.

The `readfifo.c` program for reading from the named pipe (FIFO) is as follows:

```cpp
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#define BUFFSIZE 255

int main()
{
    int fr;
    char str[BUFFSIZE];
    fr = open("FIFOPipe", O_RDONLY);
    read(fr, str, BUFFSIZE);
    printf("Read from the FIFO Pipe: %s\n", str);
    close(fr);
    return 0;
}
```

Let's go behind the scenes.

# How it works...

We will start by defining a macro called `BUFFSIZE` of size `255` and a string called `str` of size `BUFFSIZE`, that is, 255 characters. We will open the FIFO special file named `FIFOPipe` in read-only mode by invoking the `open` function. The file descriptor of the opened FIFO special file will be assigned to the `fr` variable.

Using the `read` function, the text from the FIFO special file will be read and assigned to the `str` string variable. The text that's read from the FIFO special file will then be displayed on the screen. Finally, the FIFO special file will be closed.

Now, press *Alt + F2* to open a second Terminal window. In the second Terminal window, let's use GCC to compile the `readfifo.c` program, as follows: 

```cpp
$ gcc readfifo.c -o readfifo
```

If you get no errors or warnings, this means that the `readfifo.c` program has compiled into an executable file, `readfifo.exe`. Let's run this executable file:

```cpp
$ ./readfifo
Read from the FIFO Pipe: This is a named pipe demo example called FIFO
```

The moment you run the `readfifo.exe` file, you will find, that on the previous Terminal screen where `writefifo.c` program was run will prompt you to enter a string. The moment you enter a string on that Terminal and press *Enter* key, you will get the output from the `readfifo.c` program.

Voila! We've successfully communicated between processes using a FIFO. Now, let's move on to the next recipe!

# Communicating between the client and server using socket programming

In this recipe, we will learn how data from the server process is sent to the client process. This recipe is divided into the following parts:

*   Sending data to the client
*   Reading data that's been sent from the server

Before we begin with the recipes, let's quickly review the functions, structures, and terms that are used in successful client-server communication.

# Client-server model

Different models are used for IPC, but the most popular one is the client-server model. In this model, whenever the client needs some information, it connects to another process called the server. But before establishing the connection, the client needs to know whether the server already exists, and it should know the address of the server.

On the other hand, the server is meant to serve the needs of the client and does not need to know the address of the client prior to the connection. To establish a connection, a basic construct called a socket is required, and both the connecting processes must establish their own sockets. The client and the server need to follow certain procedures to establish their sockets.

To establish a socket on the client side, a socket is created with the `socket` function system call. Thereafter, that socket is connected to the server's address using the `connect` function system call, followed by sending and receiving data by invoking the `read` function and `write` function system calls.

To establish a socket on the server side, again, a socket is created with the `socket` function system call and then the socket is bonded to an address using the `bind` function system call. Thereafter, the `listen` function system call is invoked to listen for the connections. Finally, the connection is accepted by invoking the `accept` function system call.

# struct sockaddr_in structure

This structure references the socket's elements that are used for keeping addresses. The following are the built-in members of this structure:

```cpp
struct sockaddr_in {
 short int sin_family;
 unsigned short int sin_port;
 struct in_addr sin_addr;
 unsigned char sin_zero[8];
};
```

Here, we have the following:

*   `sin_family`: Represents an address family. The valid options are `AF_INET`, `AF_UNIX`, `AF_NS`, and `AF_IMPLINK`. In most applications, the address family that's used is `AF_INET`.
*   `sin_port`: Represents the 16-bit service port number.
*   `sin_addr`: Represents a 32-bit IP address.
*   `sin_zero`: This is not used and is usually set to `NULL`.

`struct in_addr` comprise one member, as follows:

```cpp

struct in_addr {
     unsigned long s_addr; 
};
```

Here, `s_addr` is used to represent the address in network byte order.

# socket()

This function creates an endpoint for communication. To establish communication, every process needs a socket at the end of the communication line. Also, the two communicating processes must have the same socket type and both should be in the same domain. Here is the syntax for creating a socket:

```cpp
int socket(int domain, int type, int protocol);
```

Here, `domain` represents the communication domain in which a socket is to be created. Basically, the `address family` or `protocol family` is specified, which will be used in the communication.

A few of the popular `address family` are listed as follows:                     

*   `AF_LOCAL`: This is used for local communication.
*   `AF_INET`: This is used for IPv4 internet protocols.
*   `AF_INET6`: This is used for IPv6 internet protocols.
*   `AF_IPX`: This is used for protocols that use standard **IPX** (short for **Internetwork Packet Exchange**) socket addressing. 
*   `AF_PACKET`: This is used for packet interface.
*   `type`: Represents the type of socket to be created. The following are the popular socket types:
*   `SOCK_STREAM`: Stream sockets communicate as a continuous stream of characters using a **Transmission Control Protocol (TCP)**. TCP is a reliable stream-oriented protocol. So, the `SOCK_STREAM` type provides reliable, bidirectional, and connection-based byte streams.
*   `SOCK_DGRAM`: Datagram sockets read the entire messages at once using a **User Datagram Protocol (UDP)**. UDP is an unreliable, connectionless, and message-oriented protocol. These messages are of a fixed maximum length.
*   `SOCK_SEQPACKET`: Provides reliable, bidirectional, and connection-based transmission paths for datagrams.
*   `protocol`: Represents the protocol to be used with the socket. A `0` value is specified so that you can use the default protocol that's suitable for the requested socket type.

You can replace the `AF_` prefix in the preceding list with `PF_` for `protocol family`.

On successful execution, the `socket` function returns a file descriptor that can be used to manage sockets.

# memset()

This is used to fill a block of memory with the specified value. Here is its syntax:

```cpp
void *memset(void *ptr, int v, size_t n);
```

Here, `ptr` points at the memory address to be filled, `v` is the value to be filled in the memory block, and `n` is the number of bytes to be filled, starting at the location of the pointer.

# htons()

This is used to convert the unsigned short integer from host to network byte order.

# bind()

A socket that is created with the `socket` function remains in the assigned address family. To enable the socket to receive connections, an address needs to be assigned to it. The `bind` function assigns the address to the specified socket. Here is its syntax:

```cpp
   int bind(int fdsock, const struct sockaddr *structaddr, socklen_t lenaddr);
```

Here, `fdsock` represents the file descriptor of the socket, `structaddr` represents the `sockaddr` structure that contains the address to be assigned to the socket, and `lenaddr` represents the size of the address structure that's pointed to by `structaddr`.

# listen()

It listens for connections on a socket in order to accept incoming connection requests. Here is its syntax:

```cpp
int listen(int sockfd, int lenque);
```

Here, `sockfd` represents the file descriptor of the socket, and `lenque` represents the maximum length of the queue of pending connections for the given socket. An error will be generated if the queue is full.

If the function is successful it returns zero, otherwise it returns `-1`.

# accept()

It accepts a new connection on the listening socket, that is, the first connection from the queue of pending connections is picked up. Actually, a new socket is created with the same socket type protocol and address family as the specified socket, and a new file descriptor is allocated for that socket. Here is its syntax:

```cpp
int accept(int socket, struct sockaddr *address, socklen_t *len);
```

Here, we need to address the following:

*   `socket`: Represents the file descriptor of the socket waiting for the new connection. This is the socket that is created when the `socket` function is bound to an address with the `bind` function, and has invoked the `listen` function successfully.
*   `address`: The address of the connecting socket is returned through this parameter. It is a pointer to a `sockaddr` structure, through which the address of the connecting socket is returned.
*   `len`: Represents the length of the supplied `sockaddr` structure. When returned, this parameter contains the length of the address returned in bytes.

# send()

This is used for sending the specified message to another socket. The socket needs to be in a connected state before you can invoke this function. Here is its syntax:

```cpp
       ssize_t send(int fdsock, const void *buf, size_t length, int flags);
```

Here, `fdsock` represents the file descriptor of the socket through which a message is to be sent, `buf` points to the buffer that contains the message to be sent, `length` represents the length of the message to be sent in bytes, and `flags` specifies the type of message to be sent. Usually, its value is kept at `0`.

# connect()

This initiates a connection on a socket. Here is its syntax:

```cpp
int connect(int fdsock, const struct sockaddr *addr,  socklen_t len);
```

Here, `fdsock` represents the file descriptor of the socket onto which the connection is desired, `addr` represents the structure that contains the address of the socket, and `len` represents the size of the structure `addr` that contains the address.

# recv()

This is used to receive a message from the connected socket. The socket may be in connection mode or connectionless mode. Here is its syntax:

```cpp
ssize_t recv(int fdsock, void *buf, size_t len, int flags);
```

Here, `fdsock` represents the file descriptor of the socket from which the message has to be fetched, `buf` represents the buffer where the message that is received is stored, `len` specifies the length in bytes of the buffer that's pointed to by the `buf` argument, and `flags` specifies the type of message being received. Usually, its value is kept at `0`.

We can now begin with the first part of this recipe – how to send data to the client.

# Sending data to the client 

In this part of the recipe, we will learn how a server sends desired data to the client.

# How to do it…

1.  Define a variable of type `sockaddr_in`.
2.  Invoke the `socket` function to create a socket. The port number that's specified for the socket is `2000`.
3.  Call the `bind` function to assign an IP address to it.
4.  Invoke the `listen` function.
5.  Invoke the `accept` function.
6.  Invoke the `send` function to send the message that was entered by the user to the socket.
7.  The socket at the client end will receive the message.

The server program, `serverprog.c`, for sending a message to the client is as follows:

```cpp
#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>

int main(){
    int serverSocket, toSend;
    char str[255];
    struct sockaddr_in server_Address;
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    server_Address.sin_family = AF_INET;
    server_Address.sin_port = htons(2000);
    server_Address.sin_addr.s_addr = inet_addr("127.0.0.1");
    memset(server_Address.sin_zero, '\0', sizeof 
    server_Address.sin_zero); 
    bind(serverSocket, (struct sockaddr *) &server_Address, 
    sizeof(server_Address));
    if(listen(serverSocket,5)==-1)
    {
        printf("Not able to listen\n");
        return -1;
    }
    printf("Enter text to send to the client: ");
    gets(str);
    toSend = accept(serverSocket, (struct sockaddr *) NULL, NULL);
    send(toSend,str, strlen(str),0);
    return 0;
}
```

Let's go behind the scenes.

# How it works...

We will start by defining a string of size `255`, and a `server_Address` variable of type `sockaddr_in`. This structure references the socket's elements. Then, we will invoke the `socket` function to create a socket by the name of `serverSocket`. A socket is an endpoint for communication. The address family that's supplied for the socket is `AF_INET`, and the socket type selected is the stream socket type, since the communication that we want is of a continuous stream of characters.

The address family that's specified for the socket is `AF_INET`, and is used for IPv4 internet protocols. The port number that's specified for the socket is `2000`. Using the `htons` function, the short integer `2000` is converted into the network byte order before being applied as a port number. The fourth parameter, `sin_zero`, of the `server_Address` structure is set to `NULL` by invoking the `memset` function.

To enable the created `serverSocket` to receive connections, call the `bind` function to assign an address to it. Using the `sin_addr` member of the `server_Address` structure, a 32-bit IP address will be applied to the socket. Because we are working on the local machine, the localhost address `127.0.0.1` will be assigned to the socket. Now, the socket can receive the connections. We will invoke the `listen` function to enable the `serverSocket` to accept incoming connection requests. The maximum pending connections that the socket can have is 5.

You will be prompted to enter the text that is to be sent to the client. The text you enter will be assigned to the `str` string variable. By invoking the `accept` function, we will enable the `serverSocket` to accept a new connection.

The address of the connection socket will be returned through the structure of type `sockaddr_in`. The socket that is returned and that is ready to accept a connection is named `toSend`. We will invoke the `send` function to send the message that's entered by you. The socket at the client end will receive the message.

Let's use GCC to compile the `serverprog.c` program, as follows:

```cpp
$ gcc serverprog.c -o serverprog
```

If you get no errors or warnings, this means that the `serverprog.c` program has compiled into an executable file, `serverprog.exe`. Let's run this executable file:

```cpp
$ ./serverprog
Enter text to send to the client: thanks and good bye
```

Now, let's look at the other part of this recipe.

# Reading data that's been sent from the server

In this part of the recipe, we will learn how data that's been sent from the server is received and displayed on the screen.

# How to do it…

1.  Define a variable of type `sockaddr_i`.
2.  Invoke the `socket` function to create a socket. The port number that's specified for the socket is `2000`.
3.  Invoke the `connect` function to initiate a connection to the socket.
4.  Because we are working on the local machine, the localhost address `127.0.0.1` is assigned to the socket.
5.  Invoke the `recv` function to receive the message from the connected socket. The message that's read from the socket is then displayed on the screen.

The client program, `clientprog.c`, for reading a message that's sent from the server is as follows:

```cpp
#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>

int main(){
    int clientSocket;
    char str[255];
    struct sockaddr_in client_Address;
    socklen_t address_size;
    clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    client _Address.sin_family = AF_INET;
    client _Address.sin_port = htons(2000);
    client _Address.sin_addr.s_addr = inet_addr("127.0.0.1");
    memset(client _Address.sin_zero, '\0', sizeof client_Address.sin_zero); 
    address_size = sizeof server_Address;
    connect(clientSocket, (struct sockaddr *) &client_Address, address_size);
    recv(clientSocket, str, 255, 0);
    printf("Data received from server: %s", str);  
    return 0;
}
```

Let's go behind the scenes.

# How it works...

So, we have defined a string of size `255` and a variable called `client_Address` of type `sockaddr_in`. We will invoke the `socket` function to create a socket by the name of `clientSocket`.

The address family that's supplied for the socket is `AF_INET` and is used for IPv4 internet protocols, and the socket type that's selected is stream socket type. The port number that's specified for the socket is `2000`. By using the `htons` function, the short integer `2000` is converted into the network byte order before being applied as a port number.

We will set the fourth parameter, `sin_zero`, of the `client_Address` structure to `NULL` by invoking the `memset` function. We will initiate the connection to the `clientSocket` by invoking the connect function. By using the `sin_addr` member of the `client_Address` structure, a 32-bit IP address is applied to the socket. Because we are working on the local machine,  the localhost address `127.0.0.1` is assigned to the socket. Finally, we will invoke the `recv` function to receive the message from the connected `clientSocket`. The message that's read from the socket will be assigned to the `str` string variable, which will then be displayed on the screen.

Now, press *Alt + F2* to open a second Terminal window. Here, let's use GCC to compile the `clientprog.c` program, as follows:                    

```cpp
$ gcc clientprog.c -o clientprog
```

If you get no errors or warnings, this means that the `clientprog.c` program has compiled into an executable file, `clientprog.exe`. Let's run this executable file:

```cpp
$ ./clientprog
Data received from server: thanks and good bye
```

Voila! We've successfully communicated between the client and server using socket programming. Now, let's move on to the next recipe!

# Communicating between processes using a UDP socket

In this recipe, we will learn how two-way communication is implemented between a client and a server using a UDP socket. This recipe is divided into the following parts:

*   Awaiting a message from the client and sending a reply using a UDP socket
*   Sending a message to the server and receiving the reply from the server using the UDP socket

Before we begin with these recipes, let's quickly review the functions, structures, and terms that are used in successful interprocess communication using a UDP socket.

# Using a UDP socket for server-client communication

In the case of communication with UDP, the client does not establish a connection with the server but simply sends a datagram. The server does not have to accept a connection; it simply waits for datagrams to be sent from the client. Every datagram contains the address of the sender, enabling the server to identify the client on the basis of where the datagram is sent from.

For communication, the UDP server first creates a UDP socket and binds it to the server address. Then, the server waits until the datagram packet arrives from the client. Once it has arrived, the server processes the datagram packet and sends a reply to the client. This procedure keeps on repeating.

On the other hand, the UDP client, for communication, creates a UDP socket, sends a message to the server, and waits for the server's response. The client will keep repeating the procedure if they want to send more messages to the server, otherwise the socket descriptor will close.

# bzero()

This places *n* zero-valued bytes in the specified area. Here it its syntax:

```cpp
void bzero(void *r, size_t n);
```

Here, `r` is the area that's pointed to by `r` and `n` is the n number of zero values bytes that are placed in the area that was pointed to by `r`.

# INADDR_ANY

This is an IP address that is used when we don't want to bind a socket to any specific IP. Basically, while implementing communication, we need to bind our socket to an IP address. When we don't know the IP address of our machine, we can use the special IP address `INADDR_ANY`. It allows our server to receive packets that have been targeted by any of the interfaces.

# sendto()

This is used to send a message on the specified socket. The message can be sent in connection mode as well as in connectionless mode. In the case of connectionless mode, the message is sent to the specified address. Here it its syntax:

```cpp
ssize_t sendto(int fdsock, const void *buff, size_t len, int flags, const struct sockaddr *recv_addr, socklen_t recv_len);
```

Here, we need to address the following:

*   `fdsock`: Specifies the file descriptor of the socket.
*   `buff`: Points to a buffer that contains the message to be sent.
*   `len`: Specifies the length of the message in bytes.
*   `flags`: Specifies the type of the message that is being transmitted. Usually, its value is kept as 0.
*   `recv_addr`: Points to the `sockaddr` structure that contains the receiver's address. The length and format of the address depends on the address family that's been assigned to the socket.
*   `recv_len`: Specifies the length of the `sockaddr` structure that's pointed to by the `recv_addr` argument.

On successful execution, the function returns the number of bytes sent, otherwise it returns `-1`.

# recvfrom()

This is used to receive a message from a connection-mode or connectionless-mode socket. Here it its syntax:

```cpp
ssize_t recvfrom(int fdsock, void *buffer, size_t length, int flags, struct sockaddr *address, socklen_t *address_len);
```

Here, we need to address the following:

*   `fdsock`: Represents the file descriptor of the socket.
*   `buffer`: Represents the buffer where the message is stored.
*   `length`: Represents the number of bytes of the buffer that are pointed to by the `buffer` parameter.
*   `flags`: Represents the type of message that's received.
*   `address`: Represents the `sockaddr` structure in which the sending address is stored. The length and format of the address depend on the address family of the socket.
*   `address_len`: Represents the length of the `sockaddr` structure that's pointed to by the address parameter.

The function returns the length of the message that's written to the buffer, which is pointed to by the buffer argument.

Now, we can begin with the first part of this recipe: preparing a server to wait for and reply to a message from the client using a UDP socket.

# Await a message from the client and sending a reply using a UDP socket 

In this part of the recipe, we will learn how a server waits for the message from the client and how, on receiving a message from the client, it replies to the client. 

# How to do it…

1.  Define two variables of type `sockaddr_in`. Invoke the `bzero` function to initialize the structure.
2.  Invoke the `socket` function to create a socket. The address family that's supplied for the socket is `AF_INET`, and the socket type that's selected is datagram type.

3.  Initialize the members of the `sockaddr_in` structure to configure the socket. The port number that's specified for the socket is `2000`. Use `INADDR_ANY`, a special IP address, to assign an IP address to the socket.
4.  Call the `bind` function to assign the address to it.
5.  Call the `recvfrom` function to receive the message from the UDP socket, that is, from the client machine. A null character, `\0`, is added to the message that's read from the client machine and is displayed on the screen. Enter the reply that is to be sent to the client.
6.  Invoke the `sendto` function to send the reply to the client.

The server program, `udps.c`, for waiting for a message from the client and sending a reply to it using a UDP socket is as follows:

```cpp
#include <stdio.h>
#include <strings.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include<netinet/in.h>
#include <stdlib.h> 

int main()
{   
    char msgReceived[255];
    char msgforclient[255];
    int UDPSocket, len;
    struct sockaddr_in server_Address, client_Address;
    bzero(&server_Address, sizeof(server_Address));
    printf("Waiting for the message from the client\n");
    if ( (UDPSocket = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { 
        perror("Socket could not be created"); 
        exit(1); 
    }      
    server_Address.sin_addr.s_addr = htonl(INADDR_ANY);
    server_Address.sin_port = htons(2000);
    server_Address.sin_family = AF_INET; 
    if ( bind(UDPSocket, (const struct sockaddr *)&server_Address, 
    sizeof(server_Address)) < 0 ) 
    { 
        perror("Binding could not be done"); 
        exit(1); 
    } 
    len = sizeof(client_Address);
    int n = recvfrom(UDPSocket, msgReceived, sizeof(msgReceived),  0, 
    (struct sockaddr*)&client_Address,&len);
    msgReceived[n] = '\0';
    printf("Message received from the client: ");
    puts(msgReceived);
    printf("Enter the reply to be sent to the client: ");
    gets(msgforclient);
    sendto(UDPSocket, msgforclient, 255, 0, (struct 
    sockaddr*)&client_Address, sizeof(client_Address));
    printf("Reply to the client sent \n");
}
```

Let's go behind the scenes.

# How it works...

We start by defining two strings by the names of `msgReceived` and `msgforclient`, both of which are of size `255`. These two strings will be used to receive the message from and send a message to the client, respectively. Then, we will define two variables, `server_Address` and `client_Address`, of type `sockaddr_in`. These structures will referenc the socket's elements and store the server's and client's addresses, respectively. We will invoke the `bzero` function to initialize the `server_Address` structure, that is, zeros will be filled in for all of the members of the `server_Address` structure.

The server, as expected, waits for the datagram from the client. So, the following text message is displayed on the screen: `Waiting for the message from the client`. We invoke the `socket` function to create a socket by the name of `UDPSocket`. The address family that's supplied for the socket is `AF_INET`, and the socket type that's selected is datagram. The members of the `server_Address` structure are initialized to configure the socket.

Using the `sin_family ` member, the address family that's specified for the socket is `AF_INET`, which is used for IPv4 internet protocols. The port number that's specified for the socket is `2000`. Using the `htons` function, the short integer `2000` is converted into the network byte order before being applied as a port number. Then, we use a special IP address, `INADDR_ANY`, to assign an IP address to the socket. Using the `htonl` function, the `INADDR_ANY` will be converted into the network byte order before being applied as the address to the socket.

To enable the created socket, `UDPSocket`, to receive connections, we will call the `bind` function to assign the address to it. We will call the `recvfrom` function to receive the message from the UDP socket, that is, from the client machine. The message that's read from the client machine is assigned to the `msgReceived` string, which is supplied in the `recvfrom` function. A null character, `\0`, is added to the `msgReceived` string and is displayed on the screen. Thereafter, you will be prompted to enter the reply to be sent to the client. The reply that's entered is assigned to `msgforclient`. By invoking the `sendto` function, the reply is sent to the client. After sending the message, the following message is displayed to the screen: `Reply to the client sent`.

Now, let's look at the other part of this recipe.

# Sending a message to the server and receiving the reply from the server using the UDP socket

As the name suggests, in this recipe we will show you how the client sends a message to the server and then receives a reply from the server using the UDP socket.

# How to do it…

1.  Execute the first three steps from the previous part of this recipe. Assign the localhost IP address, `127.0.0.1`, as the address to the socket.
2.  Enter the message to be sent to the server. Invoke the `sendto` function to send the message to the server.
3.  Invoke the `recvfrom` function to get the message from the server. The message that's received from the server is then displayed on the screen.
4.  Close the descriptor of the socket.

The client program, `udpc.c`, to send a message to the server and to receive the reply using a UDP socket is as follows:

```cpp
#include <stdio.h>
#include <strings.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include<netinet/in.h>
#include<unistd.h>
#include<stdlib.h>

int main()
{   
    char msgReceived[255];
    char msgforserver[255];
    int UDPSocket, n;
    struct sockaddr_in client_Address;    
    printf("Enter the message to send to the server: ");
    gets(msgforserver);
    bzero(&client_Address, sizeof(client_Address));
    client_Address.sin_addr.s_addr = inet_addr("127.0.0.1");
    client_Address.sin_port = htons(2000);
    client_Address.sin_family = AF_INET;     
    if ( (UDPSocket = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { 
        perror("Socket could not be created"); 
        exit(1); 
    } 
    if(connect(UDPSocket, (struct sockaddr *)&client_Address, 
    sizeof(client_Address)) < 0)
    {
        printf("\n Error : Connect Failed \n");
        exit(0);
    } 
    sendto(UDPSocket, msgforserver, 255, 0, (struct sockaddr*)NULL, 
    sizeof(client_Address));
    printf("Message to the server sent. \n");
    recvfrom(UDPSocket, msgReceived, sizeof(msgReceived), 0, (struct 
    sockaddr*)NULL, NULL);
    printf("Received from the server: ");
    puts(msgReceived);
    close(UDPSocket);
}
```

Now, let's go behind the scenes.

# How it works...

In the first part of this recipe, we have already defined two strings by the names of `msgReceived` and `msgforclient`, both of which are of size `255`. We have also defined two variables, `server_Address` and `client_Address`, of type `sockaddr_in`.

Now, you will be prompted to enter a message that is to be sent to the server. The message you enter will be assigned to the `msgforserver` string. Then, we will invoke the `bzero` function to initialize the `client_Address` structure, that is, zeros will be filled in for all the members of the `client_Address` structure.

Next, we will initialize the members of the `client_Address` structure to configure the socket. Using the `sin_family ` member, the address family that's specified for the socket is `AF_INET`, which is used for IPv4 internet protocols. The port number that's specified for the socket is `2000`. By using the `htons` function, the short integer, `2000`, is converted into the network byte order before being applied as a port number. Then, we will assign the localhost IP address, `127.0.0.1`, as the address to the socket. We will invoke the `inet_addr` function on the localhost address to convert the string containing the address in standard IPv4 dotted decimal notation into an integer value (suitable to be used as an internet address) before is it applied to the `sin_addr` member of the `client_Address` structure.

We will invoke the `socket` function to create a socket by the name of `UDPSocket`. The address family that's supplied for the socket is `AF_INET`, and the socket type that's selected is datagram.

Next, we will invoke the `sendto` function to send the message that's been assigned to the `msgforserver` string to the server. Similarly, we will invoke the `recvfrom` function to get the message from the server. The message that's received from the server is assigned to the `msgReceived` string, which is then displayed on the screen. Finally, the descriptor of the socket is closed.

Let's use GCC to compile the `udps.c` program, as follows:

```cpp
$ gcc udps.c -o udps
```

If you get no errors or warnings, this means that the `udps.c` program has compiled into an executable file, `udps.exe`. Let's run this executable file:

```cpp
$ ./udps
Waiting for the message from the client
```

Now, press *Alt + F2* to open a second Terminal window. Here, let's use GCC again to compile the `udpc.c` program, as follows:

```cpp
$ gcc udpc.c -o udpc
```

If you get no errors or warnings, this means that the `udpc.c` program has compiled into an executable file, `udpc.exe`. Let's run this executable file:

```cpp
$ ./udpc
Enter the message to send to the server: Will it rain today?
Message to the server sent.
```

The output on the server will give us the following output:

```cpp
Message received from the client: Will it rain today?
Enter the reply to be sent to the client: It might
Reply to the client sent
```

Once the reply is sent from the server, on the client window, you will get the following output:

```cpp
Received from the server: It might
```

To run the recipes that demonstrate IPC using shared memory and message queue, we need to run Cygserver. If you are running these programs on Linux, then you can skip this section. Let's see how Cygserver is run.

# Running Cygserver

Before executing the command to run the Cygwin server, we need to configure Cygserver and install it as a service. To do so, you need to run the `cygserver.conf` script on the Terminal. The following is the output you get by running the script:

```cpp
$ ./bin/cygserver-config
Generating /etc/cygserver.conf file
Warning: The following function requires administrator privileges!
Do you want to install cygserver as service? yes

The service has been installed under LocalSystem account.
To start it, call `net start cygserver' or `cygrunsrv -S cygserver'.

Further configuration options are available by editing the configuration
file /etc/cygserver.conf. Please read the inline information in that
file carefully. The best option for the start is to just leave it alone.

Basic Cygserver configuration finished. Have fun!
```

Now, Cygserver will have been configured and installed as a service. The next step is to run the server. To run Cygserver, you need to use the following command:

```cpp
$ net start cygserver
The CYGWIN cygserver service is starting.
The CYGWIN cygserver service was started successfully.
```

Now that Cygserver is running, we can make a recipe to demonstrate IPC using shared memory and message queues.

# Passing a message from one process to another using the message queue

In this recipe, we will learn how communication between two processes is established using the message queue. This recipe is divided into the following parts:

*   Writing a message into the message queue
*   Reading a message from the message queue

Before we begin with these recipes, let's quickly review the functions, structures, and terms that are used in successful interprocess communication using shared memory and message queues.

# Functions used in IPC using shared memory and message queues

The most commonly used functions and terms for IPC using shared memory and message queues are `ftok`, `shmget`, `shmat`, `shmdt`, `shmctl`, `msgget`, `msgrcv`, and `msgsnd`.

# ftok()

This generates an IPC key on the basis of the supplied filename and ID. The filename can be provided along with its complete path. The filename must refer to an existing file. Here is the syntax:

```cpp
key_t ftok(const char *filename, int id);
```

The `ftok` function will generate the same key value if the same filename (with same path) and the same ID is supplied. Upon successful completion, `ftok` will return a key, otherwise it will return `-1`.

# shmget()

This allocates a shared memory segment and returns the shared memory identifier that's associated with the key. Here is its syntax:

```cpp
int shmget(key_t key, size_t size, int shmflg);
```

Here, we need to address the following:

*   `key`: This is (usually) the value that is returned by invoking the `ftok` function. You can also set the value of the key as `IPC_PRIVATE` if you don't want the shared memory to be accessed by other processes.
*   `size`: Represents the size of the desired shared memory segment.
*   `shmflg`: This can be any of the following constants:
    *   `IPC_CREAT`: This creates a new segment if no shared memory identifier exists for the specified key. If this flag is not used, the function returns the shared memory segment associated with the key. 
    *   `IPC_EXCL`: This makes the `shmget` function fail if the segment already exists with the specified key.

On successful execution, the function returns the shared memory identifier in the form of a non-negative integer, otherwise it returns `-1`.

# shmat()

This is used to attach a shared memory segment to the given address space. That is, the shared memory identifier that's received by invoking the `shmgt` function needs to be associated with the address space of a process. Here is its syntax:

```cpp
void *shmat(int shidtfr, const void *addr, int flag);
```

Here, we need to address the following:

*   `shidtfr`: Represents the memory identifier of the shared memory segment.
*   `addr`: Represents the address space where the segment needs to be attached. If `shmaddr` is a null pointer, the segment is attached at the first available address or selected by the system.
*   `flag`: This is attached as a read-only memory if the flag is `SHM_RDONLY`; otherwise, it is readable and writable.

If successfully executed, the function attaches the shared memory segment and returns the segment's start address, otherwise it returns `-1`.

# shmdt()

This detaches the shared memory segment. Here is its syntax:

```cpp
int shmdt(const void *addr);
```

Here, `addr` represents the address at which the shared memory segment is located.

# shmctl()

This is used for performing certain control operations on the specified shared memory segment. Here is its syntax:

```cpp
int shmctl(int shidtr, int cmd, struct shmid_ds *buf);
```

Here, we have to address the following:

*   `shidtr`: Represents the identifier of the shared memory segment.
*   `cmd`: This can have any of the following constants:
    *   `IPC_STAT`: This copies the content of the `shmid_ds` data structure associated with the shared memory segment represented by `shidtr` into the structure that's pointed to by `buf`
    *   `IPC_SET`: This writes the content of the structure that's pointed to by `buf` into the `shmid_ds` data structure, which is associated with the memory segment that's represented by `shidtr`
    *   `IPC_RMID`: This removes the shared memory identifier that's specified by `shidtr` from the system and destroys the shared memory segment and `shmid_ds` data structure associated with it
*   `buf`: This is a pointer to a `shmid_ds` structure.

If successfully executed, the function returns `0`, otherwise it returns `-1`.

# msgget()

This is used for creating a new message queue, and for accessing an existing queue that is related to the specified key. If this is executed successfully, the function returns the identifier of the message queue:

```cpp
       int msgget(key_t key, int flag);
```

Here, we have to address the following:

*   `key`: This is a unique key value that is retrieved by invoking the `ftok` function.
*   `flag`: This can be any of the following constants:
    *   `IPC_CREAT`: Creates the message queue if it doesn't already exist and returns the message queue identifier for the newly created message queue. If the message queue already exists with the supplied key value, it returns its identifier.
    *   `IPC_EXCL`: If both `IPC_CREAT` and `IPC_EXCL` are specified and the message queue does not exist, then it is created. However, if it already exists, then the function will fail.

# msgrcv()

This is used for reading a message from a specified message queue whose identifier is supplied. Here is its syntax:

```cpp
int msgrcv(int msqid, void *msgstruc, int msgsize, long typemsg, int flag);
```

Here, we have to address the following:

*   `msqid`: Represents the message queue identifier of the queue from which the message needs to be read.
*   `msgstruc`: This is the user-defined structure into which the read message is placed. The user-defined structure must contain two members. One is usually named `mtype`, which must be of type long int that specifies the type of the message, and the second is usually called `mesg`, which should be of `char` type to store the message.
*   `msgsize`: Represents the size of text to be read from the message queue in terms of bytes. If the message that is read is larger than `msgsize`, then it will be truncated to `msgsize` bytes.
*   `typemsg`: Specifies which message on the queue needs to be received:
    *   If `typemsg` is `0`, the first message on the queue is received
    *   If `typemsg` is greater than `0`, the first message whose `mtype` field is equal to `typemsg` is received
    *   If `typemsg` is less than `0`, a message whose `mtype` field is less than or equal to `typemsg` is received

*   `flag`: Determines the action to be taken if the desired message is not found in the queue. It keeps its value of `0` if you don't want to specify the `flag`. The `flag` can have any of the following values:
    *   `IPC_NOWAIT`: This makes the `msgrcv` function fail if there is no desired message in the queue, that is, it will not make the caller wait for the appropriate message on the queue. If `flag` is not set to `IPC_NOWAIT, it` will make the caller wait for an appropriate message on the queue instead of failing the function.
    *   `MSG_NOERROR`: This allows you to receive text that is larger than the size that's specified in the `msgsize` argument. It simply truncates the text and receives it. If this `flag` is not set, on receiving the larger text, the function will not receive it and will fail the function.

If the function is executed successfully, the function returns the number of bytes that were actually placed into the text field of the structure that is pointed to by `msgstruc`. On failure, the function returns a value of `-1`.

# msgsnd()

This is used for sending or delivering a message to the queue. Here is its syntax:                                                   

```cpp
 int msgsnd ( int msqid, struct msgbuf *msgstruc, int msgsize, int flag );
```

Here, we have to address the following:

*   `msqid`: Represents the queue identifier of the message that we want to send. The queue identifier is usually retrieved by invoking the `msgget` function.
*   `msgstruc`: This is a pointer to the user-defined structure. It is the `mesg` member that contains the message that we want to send to the queue.
*   `msgsize`: Represents the size of the message in bytes.
*   `flag`: Determines the action to be taken on the message. If the `flag` value is set to `IPC_NOWAIT` and if the message queue is full, the message will not be written to the queue, and the control is returned to the calling process. But if `flag` is not set and the message queue is full, then the calling process will suspend until a space becomes available in the queue. Usually, the value of `flag` is set to `0`. 

If this is executed successfully, the function returns `0`, otherwise it returns `-1`. 

We will now begin with the first part of this recipe: writing a message into the queue.

# Writing a message into the message queue

In this part of the recipe, we will learn how a server writes a desired message into the message queue.

# How to do it…

1.  Generate an IPC key by invoking the `ftok` function. A filename and ID are supplied while creating the IPC key.
2.  Invoke the `msgget` function to create a new message queue. The message queue is associated with the IPC key that was created in step 1.
3.  Define a structure with two members, `mtype` and `mesg`. Set the value of the `mtype` member to 1.
4.  Enter the message that's going to be added to the message queue. The string that's entered is assigned to the `mesg` member of the structure that we defined in step 3.
5.  Invoke the `msgsnd` function to send the entered message into the message queue.

The `messageqsend.c` program for writing the message to the message queue is as follows:

```cpp
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MSGSIZE     255

struct msgstruc {
    long mtype;
    char mesg[MSGSIZE];
};

int main()
{
    int msqid, msglen;
    key_t key;
    struct msgstruc msgbuf;
    system("touch messagefile");
    if ((key = ftok("messagefile", 'a')) == -1) {
        perror("ftok");
        exit(1);
    } 
    if ((msqid = msgget(key, 0666 | IPC_CREAT)) == -1) {
        perror("msgget");
        exit(1);
    }
    msgbuf.mtype = 1;
    printf("Enter a message to add to message queue : ");
    scanf("%s",msgbuf.mesg);
    msglen = strlen(msgbuf.mesg);
    if (msgsnd(msqid, &msgbuf, msglen, IPC_NOWAIT) < 0)
        perror("msgsnd");
    printf("The message sent is %s\n", msgbuf.mesg);
    return 0;
}
```

Let's go behind the scenes.

# How it works...

We will start by generating an IPC key by invoking the `ftok` function. The filename and ID are supplied while creating the IPC key are `messagefile` and `a`, respectively. The generated key is assigned to the key variable. Thereafter, we will invoke the `msgget` function to create a new message queue. The message queue is associated with the IPC key we just created using the `ftok` function.

Next, we will define a structure by the name of `msgstruc` with two members, `mtype` and `mesg`. The `mtype` member helps in determining the sequence number of the message that is going to be sent or received from the message queue. The `mesg` member contains the message that is going to be read or written into the message queue. We will define a variable called `msgbuf` of the `msgstruc` structure type. The value of the `mtype` member is set to `1`.

You will be prompted to enter the message that is going to be added to the message queue. The string you enter is assigned to the `mesg` member of the `msgbuf` structure. The `msgsnd` function is invoked to send the message you entered into the message queue. Once the message is written into the message queue, a text message is displayed on the screen as confirmation.

Now, let's move on to the other part of this recipe.

# Reading a message from the message queue

In this part of the recipe, we will learn how the message that was written into the message queue is read and displayed on the screen.

# How to do it…

1.  Invoke the `ftok` function to generate an IPC key. The filename and ID are supplied while creating the IPC key. These must be the same as what were applied while generating the key for writing the message in the message queue.
2.  Invoke the `msgget` function to access the message queue that is associated with the IPC key. The message queue that's associated with this key already contains a message that we wrote through the previous program.
3.  Define a structure with two members, `mtype` and `mesg`. 
4.  Invoke the `msgrcv` function to read the message from the associated message queue. The structure that was defined in Step 3 is passed to this function.
5.  The read message is then displayed on the screen.

The `messageqrecv.c` program for reading a message from the message queue is as follows:

```cpp
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <stdio.h>
#include <stdlib.h>
#define MSGSIZE     255

struct msgstruc {
    long mtype;
    char mesg[MSGSIZE];
};

int main()
{
    int msqid;
    key_t key;
    struct msgstruc rcvbuffer;

    if ((key = ftok("messagefile", 'a')) == -1) {
        perror("ftok");
        exit(1);
    }
    if ((msqid = msgget(key, 0666)) < 0)
    {
        perror("msgget");
        exit(1);
    }
    if (msgrcv(msqid, &rcvbuffer, MSGSIZE, 1, 0) < 0)
    {
        perror("msgrcv");
        exit(1);
    }
    printf("The message received is %s\n", rcvbuffer.mesg);
    return 0;
}
```

Let's go behind the scenes.

# How it works...

First, we will invoke the `ftok` function to generate an IPC key. The filename and ID that are supplied while creating the IPC key are `messagefile` and `a`, respectively. These filenames and ID must be the same as the ones that were applied while generating the key for writing the message in the message queue. The generated key is assigned to the key variable.

Thereafter, we will invoke the `msgget` function to access the message queue that is associated with the IPC key. The identifier of the accessed message queue is assigned to the `msqid` variable. The message queue that's associated with this key already contains the message that we wrote in the previous program.

Then, we will define a structure by the name `msgstruc` with two members, `mtype` and `mesg`. The `mtype` member is for determining the sequence number of the message to be read from the message queue. The `mesg` member will be used for storing the message that is read from the message queue. We will then define a variable called `rcvbuffer` of the `msgstruc` structure type. We will invoke the `msgrcv` function to read the message from the associated message queue.

The message identifier, `msqid`, is passed to the function, along with the `rcvbuffer` – the structure whose `mesg` member will store the read message. After successful execution of the `msgrcv` function, the `mesg` member of the `rcvbuffer` containing the message from the message queue will be displayed on screen.

Let's use GCC to compile the `messageqsend.c` program, as follows:

```cpp
$ gcc messageqsend.c -o messageqsend
```

If you get no errors or warnings, this means that the `messageqsend.c` program has compiled into an executable file, `messageqsend.exe`. Let's run this executable file:

```cpp
$ ./messageqsend
Enter a message to add to message queue : GoodBye
The message sent is GoodBye
```

Now, press *Alt + F2* to open a second Terminal screen. On this screen, you can compile and run the script for reading the message from the message queue.

Let's use GCC to compile the `messageqrecv.c` program, as follows:

```cpp
$ gcc messageqrecv.c -o messageqrecv
```

If you get no errors or warnings, this means that the `messageqrecv.c` program has compiled into an executable file, `messageqrecv.exe`. Let's run this executable file:

```cpp
$ ./messageqrecv
The message received is GoodBye
```

Voila! We've successfully passed a message from one process to another using the message queue. Let's move on to the next recipe!

# Communicating between processes using shared memory

In this recipe, we will learn how communication between two processes is established using shared memory. This recipe is divided into the following parts:

*   Writing a message into shared memory
*   Reading a message from shared memory

We will start with the first one, that is, *Writing a message into shared memory*. The functions we learned in the previous recipe will also be applicable here.

# Writing a message into shared memory

In this part of this recipe, we will learn how a message is written into shared memory.

# How to do it…

1.  Invoke the `ftok` function to generate an IPC key by supplying a filename and an ID.
2.  Invoke the `shmget` function to allocate a shared memory segment that is associated with the key that was generated in step 1.
3.  The size that's specified for the desired memory segment is `1024`. Create a new memory segment with read and write permissions.
4.  Attach the shared memory segment to the first available address in the system.
5.  Enter a string that is then assigned to the shared memory segment.
6.  The attached memory segment will be detached from the address space.

The `writememory.c` program for writing data into the shared memory is as follows:

```cpp
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
    char *str;
    int shmid;

    key_t key = ftok("sharedmem",'a');
    if ((shmid = shmget(key, 1024,0666|IPC_CREAT)) < 0) {
        perror("shmget");
        exit(1);
    }
    if ((str = shmat(shmid, NULL, 0)) == (char *) -1) {
        perror("shmat");
        exit(1);
    }
    printf("Enter the string to be written in memory : ");
    gets(str);
    printf("String written in memory: %s\n",str);
    shmdt(str);
    return 0;
}
```

Let's go behind the scenes.

# How it works...

By invoking the `ftok` function, we generate an IPC key with the filename `sharedmem` (you can change this) and an ID of `a`. The generated key is assigned to the key variable. Thereafter, invoke the `shmget` function to allocate a shared memory segment that is associated with the supplied key generated using the `ftok` function.

The size that's specified for the desired memory segment is `1024`. Create a new memory segment with read and write permissions and assign the shared memory identifier to the `shmid` variable. Then, attach the shared memory segment to the first available address in the system.

Once the memory segment is attached to the address space, the segment's start address is assigned to the `str` variable. You will be asked to enter a string. The string you enter will be assigned to the shared memory segment through the `str` variable. Finally, the attached memory segment is detached from the address space.

Let's move on to the next part of this recipe, *Reading a message from shared memory*.

# Reading a message from shared memory

In this part of the recipe, we will learn how the message that was written into shared memory is read and displayed on screen.

# How to do it…

1.  Invoke the `ftok` function to generate an IPC key. The filename and ID that are supplied should be the same as those in the program for writing content into shared memory.
2.  Invoke the `shmget` function to allocate a shared memory segment. The size that's specified for the allocated memory segment is `1024` and is associated with the IPC key that was generated in step 1\. Create the memory segment with read and write permissions.
3.  Attach the shared memory segment to the first available address in the system. 
4.  The content from the shared memory segment is read and displayed on screen.
5.  The attached memory segment is detached from the address space.
6.  The shared memory identifier is removed from the system, followed by destroying the shared memory segment.

The `readmemory.c` program for reading data from shared memory is as follows:

```cpp
#include <stdio.h> 
#include <sys/ipc.h> 
#include <sys/shm.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
    int shmid;
    char * str;
    key_t key = ftok("sharedmem",'a');
    if ((shmid = shmget(key, 1024,0666|IPC_CREAT)) < 0) {
        perror("shmget");
        exit(1);
    }
    if ((str = shmat(shmid, NULL, 0)) == (char *) -1) {
        perror("shmat");
        exit(1);
    }
    printf("Data read from memory: %s\n",str);
    shmdt(str);                
    shmctl(shmid,IPC_RMID,NULL);
    return 0;
}
```

Let's go behind the scenes.

# How it works...

We will invoke the `ftok` function to generate an IPC key. The filename and ID that are supplied for generating the key are `sharedmem` (any name) and `a`, respectively. The generated key is assigned to the `key` variable. Thereafter, we will invoke the `shmget` function to allocate a shared memory segment. The size that's specified for the allocated memory segment is `1024` and is associated with the IPC key that was generated earlier.

We will create the new memory segment with read and write permissions and assign the fetched shared memory identifier to the `shmid` variable. The shared memory segment is then attached to the first available address in the system. This is done so that we can access the text that was written in the shared memory segment through the previous program.

So, after the memory segment is attached to the address space, the segment's start address is assigned to the `str` variable. Now, we can read the content that's been written in the shared memory through the previous program in the current program. The content from the shared memory segment is read through the `str` string and displayed on screen.

Thereafter, the attached memory segment is detached from the address space. Finally, the shared memory identifier `shmid` is removed from the system and the shared memory segment is destroyed.

Let's use GCC to compile the `writememory.c` program, as follows:

```cpp
$ gcc writememory.c -o writememory
```

If you get no errors or warnings, this means that the `writememory.c` program has compiled into an executable file, `writememory.exe`. Let's run this executable file:

```cpp
$ ./writememory
Enter the string to be written in memory : Today it might rain
String written in memory: Today it might rain
```

Now, press *Alt + F2* to open a second Terminal window. In this window, let's use GCC to compile the `readmemory.c` program, as follows:

```cpp
$ gcc readmemory.c -o readmemory
```

If you get no errors or warnings, this means that the `readmemory.c` program has compiled into an executable file, `readmemory.exe`. Let's run this executable file:

```cpp
$ ./readmemory
 Data read from memory: Today it might rain
```

Voila! We've successfully communicated between processes using shared memory.