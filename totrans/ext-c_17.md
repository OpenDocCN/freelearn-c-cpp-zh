# Chapter 17

# Process Execution

We are now ready to talk about the software systems consisting of more than one process in their overall architecture. These systems are usually called multi-process or multiple-process systems. This chapter, together with the next chapter, is trying to cover the concepts of multi-processing and conduct a pros-and-cons analysis in order to compare it with multithreading, which we covered in *Chapter 15*, *Thread Execution*, and *Chapter 16*, *Thread Synchronization*.

In this chapter, our focus is the available APIs and techniques to start a new process and how process execution actually happens, and in the next chapter, we'll go through concurrent environments consisting of more than one process. We are going to explain how various states can be shared among a number of processes and what common ways of accessing shared state in a multi-processing environment are.

A proportion of this chapter is based on comparing multi-processing and multithreading environments. In addition, we briefly talk about single-host multi-processing systems and distributed multi-processing systems.

# Process execution APIs

Every program is executed as a process. Before we have a process, we just have an executable binary file that contains some memory segments and probably lots of machine-level instructions. Conversely, every process is an individual instance of a program being executed. Therefore, a single compiled program (or an executable binary file) can be executed multiple times through different processes. In fact, that's why our focus is on the processes in this chapter, rather than upon the programs themselves.

In two previous chapters, we talked about threads in single-process software, but to follow our objective in this chapter, we are going to be talking about software with multiple processes. But first, we need to know how, and by using which API, a new process can be spawned.

Note that our main focus is on executing processes in Unix-like operating systems since all of them follow the Unix onion architecture and expose very well-known and similar APIs. Other operating systems can have their own ways for executing processes, but since most of them, more or less, follow the Unix onion architecture, we expect to see similar methods for process execution.

In a Unix-like operating system, there are not many ways to execute a process at the system call level. If you remember the *kernel ring* from *Chapter 11*, *System Calls and Kernel*, it is the most inner ring after the *hardware ring*, and it provides the *system call interface* to outer rings, *shell*, and *user*, in order to let them execute various kernel-specific functionalities. Two of these exposed system calls are dedicated to process creation and process execution; respectively, `fork` and `exec` (`execve` in Linux though). In *process creation*, we spawn a new process, but in *process execution* we use an existing process as the host, and we replace it with a new program; therefore, no new process is spawned in process execution.

As a result of using these systems calls, a program is always executed as a new process, but this process is not always spawned! The `fork` system call spawns a new process while the `exec` system call replaces the caller (the host) process with a new one. We talk about the differences between the `fork` and `exec` system calls later. Before that, let's see how these system calls are exposed to the outer rings.

As we explained in *Chapter 10*, *Unix – History and Architecture*, we have two standards for Unix-like operating systems, specifically about the interface they should expose from their shell ring. These standards are **Single Unix Specification** (**SUS**) and **POSIX**. For more information regarding these standards, along with their similarities and differences, please refer to *Chapter 10*, *Unix – History and Architecture*.

The interface that should be exposed from the shell ring is thoroughly specified in the POSIX interface, and indeed, there are parts in the standard that deal with process execution and process management.

Therefore, we would expect to find headers and functions for process creation and process execution within POSIX. Such functions do indeed exist, and we find them in different headers that provide the desired functionality. Following is a list of POSIX functions responsible for process creation and process execution:

*   The function `fork` that can be found in the `unistd.h` header file is responsible for process creation.
*   The `posix_spawn` and `posix_spawnp` functions that can be found in the `spawn.h` header file. These functions are responsible for process creation.
*   The group of `exec*` functions, for example, `execl` and `execlp`, that can be found in the `unistd.h` header file. These functions are responsible for process execution.

Note that the preceding functions should not be mistaken for the `fork` and `exec` system calls. These functions are part of the POSIX interface exposed from the shell ring while the system calls are exposed from the kernel ring. While most Unix-like operating systems are POSIX-compliant, we can have a non-Unix-like system that is also POSIX-compliant. Then, the preceding functions exist in that system, but the underlying mechanism for spawning a process can be different at the system call level.

A tangible example is using Cygwin or MinGW to make Microsoft Windows POSIX-compliant. By installing these programs, you can write and compile standard C programs that are using the POSIX interface, and Microsoft Windows becomes partially POSIX-compliant, but there are no `fork` or `exec` system calls in Microsoft Windows! This is in fact very confusing and very important at the same time, and you should know that the shell ring does not necessarily expose the same interface that is exposed by the kernel ring.

**Note**:

You can find the implementation details of the `fork` function in Cygwin here: https://github.com/openunix/cygwin/blob/master/winsup/cygwin/fork.cc. Note that it doesn't call the `fork` system call that usually exists in Unix-like kernels; instead, it includes headers from the Win32 API and calls functions that are well-known functions regarding process creation and process management.

According to the POSIX standard, the C standard library is not the only thing that is exposed from the shell ring on a Unix-like system. When using a Terminal, there are prewritten shell utility programs that are used to provide a complex usage of the C standard API. About the process creation, whenever the user enters a command in the Terminal, a new process is created.

Even a simple `ls` or `sed` command spawns a new process that might only last less than a second. You should know that these utility programs are mostly written in C language and they are consuming the same exact POSIX interface which you would have been using when writing your own programs.

Shell scripts are also executed in a separate process but in a slightly different fashion. We will discuss them in future sections on how a process is executed within a Unix-like system.

Process creation happens in the kernel, especially in monolithic kernels. Whenever a user process spawns a new process or even a new thread, the request is received by the system call interface, and it gets passed down to the kernel ring. There, a new *task* is created for the incoming request, either a process or a thread.

Monolithic kernels like Linux or FreeBSD keep track of the tasks (process and threads) within their kernel, and this makes it reasonable to have processes being created in the kernel itself.

Note that whenever a new task is created within the kernel, it is placed in the queue of the *task scheduler unit* and it might take a bit of time for it to obtain the CPU and begin execution.

In order to create a new process, a parent process is needed. That's why every process has a parent. In fact, each process can have only one parent. The chain of parents and grandparents goes back to the first user process, which is usually called *init*, and the kernel process is its parent.

It is the ancestor to all other processes within a Unix-like system and exists until the system shuts down. Regularly, the init process becomes the parent of all *orphan processes* that have had their parent processes terminated, so that no process can be left without a parent process.

This parent-child relationship ends up in a big process tree. This tree can be examined by the command utility *pstree*. We are going to show how to use this utility in future examples.

Now, we know the API that can execute a new process, and we need to give some real C examples on how these methods actually work. We start with the fork API, which eventually calls the `fork` system call.

## Process creation

As we mentioned in the previous section, the fork API can be used to spawn a new process. We also explained that a new process can only be created as a child of a running process. Here, we see a few examples of how a process can fork a new child using the fork API.

In order to spawn a new child process, a parent process needs to call the `fork` function. The declaration of the `fork` function can be included from the `unistd.h` header file which is part of the POSIX headers.

When the `fork` function is called, an exact copy of the caller process (which is called the parent process) is created, and both processes continue to run concurrently starting from the very next instruction after the `fork` invocation statement. Note that the child (or forked) process inherits many things from the parent process including all the memory segments together with their content. Therefore, it has access to the same variables in the Data, Stack, and Heap segments, and also the program instructions found in the Text segment. We talk about other inherited things in the upcoming paragraphs, after talking about the example.

Since we have two different processes now, the `fork` function returns twice; once in the parent process and another time in the child process. In addition, the `fork` function returns different values to each process. It returns 0 to the child process, and it returns the PID of the forked (or child) process to the parent process. *Example 17.1* shows how `fork` works in one of its simplest usages:

```cpp
#include <stdio.h>
#include <unistd.h>
int main(int argc, char** argv) {
  printf("This is the parent process with process ID: %d\n",
          getpid());
  printf("Before calling fork() ...\n");
  pid_t ret = fork();
  if (ret) {
    printf("The child process is spawned with PID: %d\n", ret);
  } else {
    printf("This is the child process with PID: %d\n", getpid());
  }
  printf("Type CTRL+C to exit ...\n");
  while (1);
  return 0;
}
```

Code Box 17-1 [ExtremeC_examples_chapter17_1.c]: Create a child process using the fork API

In the preceding code box, we have used `printf` to print out some logs in order to track the activity of the processes. As you see, we have invoked the `fork` function in order to spawn a new process. As is apparent, it doesn't accept any argument, and therefore, its usage is very easy and straightforward.

Upon calling the `fork` function, a new process is forked (or cloned) from the caller process, which is now the parent process, and after that, they continue to work concurrently as two different processes.

Surely, the call to the `fork` function will cause further invocations on the system call level, and only then, the responsible logic in the kernel can create a new forked process.

Just before the `return` statement, we have used an infinite loop to keep both processes running and prevent them from exiting. Note that the processes should reach this infinite loop eventually because they have exactly the same instructions in their Text segments.

We want to keep the processes running intentionally in order to be able to see them in the list of processes shown by the `pstree` and `top` commands. Before that, we need to compile the preceding code and see how the new process is forked, as shown in *Shell Box 17-1*:

```cpp
$ gcc ExtremeC_examples_chapter17_1.c -o ex17_1.out
$ ./ex17_1.out
This is the parent process with process ID: 10852
Before calling fork() …
The child process is spawned with PID: 10853
This is the child process with PID: 10853
Type CTRL+C to exit ...
$
```

Shell Box 17-1: Building and running example 17.1

As you can see, the parent process prints its PID, and that is `10852`. Note that the PID is going to change in each run. After forking the child process, the parent process prints the PID returned by the `fork` function, and it is `10853`.

On the next line, the child process prints its PID, which is again `10853` and it is in accordance with what the parent has received from the `fork` function. And finally, both processes enter the infinite loop, giving us some time to observe them in the probing utilities.

As you see in *Shell Box 17-1*, the forked process inherits the same `stdout` file descriptor and the same terminal from its parent. Therefore, it can print to the same output that its parent writes to. A forked process inherits all the open file descriptors at the time of the `fork` function call from its parent process.

In addition, there are also other inherited attributes, which can be found in `fork`'s manual pages. The `fork`'s manual page for Linux can be found here: http://man7.org/linux/man-pages/man2/fork.2.html.

If you open the link and look through the attributes, you are going to see that there are attributes that are shared between the parent and forked processes, and there are other attributes that are different and specific to each process, for example, PID, parent PID, threads, and so on.

The parent-child relationship between processes can be easily seen using a utility program like `pstree`. Every process has a parent process, and all of the processes contribute to building a big tree. Remember that each process has exactly one parent, and a single process cannot have two parents.

While the processes in the preceding example are stuck within their infinite loops, we can use the `pstree` utility command to see the list of all processes within the system displayed as a tree. The following is the output of the `pstree` usage in a Linux machine. Note that the `pstree` command is installed on Linux systems by default, but it might need to be installed in other Unix-like operating systems:

```cpp
$ pstree -p
systemd(1)─┬─accounts-daemon(877)─┬─{accounts-daemon}(960)
           │                      └─{accounts-daemon}(997)
...
...
...
           ├─systemd-logind(819)
           ├─systemd-network(673)
           ├─systemd-resolve(701)
           ├─systemd-timesyn(500)───{systemd-timesyn}(550)
           ├─systemd-udevd(446)
           └─tmux: server(2083)─┬─bash(2084)───pstree(13559)
                                └─bash(2337)───ex17_1.out(10852)───ex17_1.out(10853)
$
```

Shell Box 17-2: Use pstree to find the processes spawned as part of example 17.1

As can be seen in the last line of *Shell Box 17-2*, we have two processes with PIDs `10852` and `10853` that are in the parent-child relationship. Note that process `10852` has a parent with PID `2337`, which is a *bash* process.

It's interesting to note that on the line before the last line, we can see the `pstree` process itself as the child of the bash process with PID `2084`. Both of the bash processes belong to the same *tmux* terminal emulator with PID `2083`.

In Linux, the very first process is the *scheduler* process, which is part of the kernel image, and it has the PID 0\. The next process, which is usually called *init*, has the PID 1, and it is the first user process which is created by the scheduler process. It exists from system startup until its shutdown. All other user processes are directly or indirectly the children of the `init` process. The processes which lose their parent processes become orphan processes, and they become abducted by the init process as its direct children.

However, in the newer versions of almost all famous distributions of Linux, the init process has been replaced by the *systemd daemon*, and that's why you see `systemd(1)` on the first line in *Shell Box 17-2*. The following link is a great source to read more about the differences between `init` and `systemd` and why Linux distro developers have made such a decision: https://www.tecmint.com/systemd-replaces-init-in-linux.

When using the fork API, the parent and forked processes are executed concurrently. This means that we should be able to detect some behaviors of concurrent systems.

The best-known behavior that can be observed is some interleavings. If you are not familiar with this term or you have not heard it before, it is strongly recommended to have a read of *Chapter 13*, *Concurrency*, and *Chapter 14*, *Synchronization*.

The following example, *example 17.2*, shows how the parent and forked processes can have non-deterministic interleavings. We are going to print some strings and observe how some various interleavings can happen in two successive runs:

```cpp
#include <stdio.h>
#include <unistd.h>
int main(int argc, char** argv) {
  pid_t ret = fork();
  if (ret) {
    for (size_t i = 0; i < 5; i++) {
      printf("AAA\n");
      usleep(1);
    }
  } else {
    for (size_t i = 0; i < 5; i++) {
      printf("BBBBBB\n");
      usleep(1);
    }
  }
  return 0;
}
```

Code Box 17-2 [ExtremeC_examples_chapter17_2.c]: Two processes that print some lines to the standard output

The preceding code is very similar to the code we wrote for *example 17.1*. It creates a forked process, and after that, the parent and forked processes print some lines of text to the standard output. The parent process prints `AAA` 5 times, and the forked process prints `BBBBBB` five times. The following is the output of the two consecutive runs of the same compiled executable:

```cpp
$ gcc ExtremeC_examples_chapter17_2.c -o ex17_2.out
$ ./ex17_2.out
AAA
AAA
AAA
AAA
AAA
BBBBBB
BBBBBB
BBBBBB
BBBBBB
BBBBBB
$ ./ex17_2.out
AAA
AAA
BBBBBB
AAA
AAA
BBBBBB
BBBBBB
BBBBBB
AAA
BBBBBB
$
```

Shell Box 17-3: Output of two successive runs of example 17.2

It is clear from the preceding output that we have different interleavings. This means we can be potentially suffering from a race condition here if we define our invariant constraint according to what we see in the standard output. This would eventually lead to all the issues we faced while writing multithreaded code, and we need to use similar methods to overcome these issues. In the next chapter, we will discuss such solutions in greater detail.

In the following section, we are going to talk about process execution and how it can be achieved using `exec*` functions.

## Process execution

Another way to execute a new process is by using the family of `exec*` functions. This group of functions takes a different approach to execute a new process in comparison to the fork API. The philosophy behind `exec*` functions is to create a simple base process first and then, at some point, load the target executable and replace it as a new *process image* with the base process. A process image is the loaded version of a executable that has its memory segments allocated, and it is ready to be executed. In the future sections, we will discuss the different steps of loading an executable, and we will explain process images in greater depth.

Therefore, while using the `exec*` functions, no new process is created, and a process substitution happens. This is the most important difference between `fork` and `exec*` functions. Instead of forking a new process, the base process is totally substituted with a new set of memory segments and code instructions.

*Code Box 17-3*, containing *example 17.3*, shows how the `execvp` function, one of the functions in the family of `exec*` functions, is used to start an echo process. The `execvp` function is one of the functions in the group of `exec*` functions that inherits the environment variable `PATH` from the parent process and searches for the executables as the parent process did:

```cpp
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
int main(int argc, char** argv) {
  char *args[] = {"echo", "Hello", "World!", 0};
  execvp("echo", args);
  printf("execvp() failed. Error: %s\n", strerror(errno));
  return 0;
}
```

Code Box 17-3 [ExtremeC_examples_chapter17_3.c]: Demonstration of how execvp works

As you see in the preceding code box, we have invoked the function `execvp`. As we explained before, the `execvp` function inherits the environment variable `PATH`, together with the way it looks for the existing executables, from the base process. It accepts two arguments; the first is the name of the executable file or the script which should be loaded and executed, and the second is the list of arguments that should be passed to the executable.

Note that we are passing `echo` and not an absolute path. Therefore, `execvp` should locate the `echo` executable first. These executable files can be anywhere in a Unix-like operating system, from `/usr/bin` to `/usr/local/bin` or even other places. The absolute location of the `echo` can be found by going through all directory paths found in the `PATH` environment variable.

The `exec*` functions can execute a range of executable files. Following is a list of some file formats that can be executed by `exec*` functions:

*   ELF executable files
*   Script files with a *shebang* line indicating the *interpreter* of the script
*   Traditional `a.out` format binary files
*   ELF FDPIC executable files

After finding the `echo` executable file, the `execvp` does the rest. It calls the `exec` (`execve` in Linux) system call with a prepared set of arguments and subsequently, the kernel prepares a process image from the found executable file. When everything is ready, the kernel replaces the current process image with the prepared one, and the base process is gone forever. Now, the control returns to the new process, and it becomes executing from its `main` function, just like a normal execution.

As a result of this process, the `printf` statement after the `execvp` function call statement cannot be executed if the `execvp` has been successful, because now we have a whole new process with new memory segments and new instructions. If the `execvp` statement wasn't successful, then the `printf` should have been executed, which is a sign for the failure of `execvp` function call.

Like we said before, we have a group of `exec*` functions, and the `execvp` function is only one of them. While all of them behave similarly, they have slight differences. Next, you can find a comparison of these functions:

*   `execl(const char* path, const char* arg0, ..., NULL)`: Accepts an absolute path to the executable file and a series of arguments that should be passed to the new process. They must end with a null string, `0` or `NULL`. If we wanted to rewrite *example 17.3* using `execl`, we would use `execl("/usr/bin/echo", "echo", "Hello", "World", NULL)`.
*   `execlp(const char* file, const char* arg0, ..., NULL)`: Accepts a relative path as its first argument, but since it has access to the `PATH` environment variable, it can locate the executable file easily. Then, it accepts a series of arguments that should be passed to the new process. They must end with a null string, `0` or `NULL`. If we wanted to rewrite *example 17.3* using `execlp`, we would use `execlp("echo", "echo," "Hello," "World," NULL)`.
*   `excele(const char* path, const char* arg0, ..., NULL, const char* env0, ..., NULL)`: Accepts an absolute path to the executable file as its first argument. Then, it accepts a series of arguments that should be passed to the new process followed by a null string. Following that, it accepts a series of strings representing the environment variables. They must also end with a null string. If we wanted to rewrite *example 17.3* using `execle`, we would use `execle("/usr/bin/echo", "echo", "Hello", "World", NULL, "A=1", "B=2", NULL)`. Note that in this call we have passed two new environment variables, `A` and `B`, to the new process.
*   `execv(const char* path, const char* args[])`: Accepts an absolute path to the executable file and an array of the arguments that should be passed to the new process. The last element in the array must be a null string, `0` or `NULL`. If we wanted to rewrite *example 17.3* using `execl`, we would use `execl("/usr/bin/echo", args)` in which `args` is declared like this: `char* args[] = {"echo", "Hello", "World", NULL}`.
*   `execvp(const char* file, const char* args[])`: It accepts a relative path as its first argument, but since it has access to the `PATH` environment variable, it can locate the executable file easily. Then, it accepts an array of the arguments that should be passed to the new process. The last element in the array must be a null string, `0` or `NULL`. This is the function that we used in *example 17.3*.

When `exec*` functions are successful, the previous process is gone, and a new process is created instead. Therefore, there isn't a second process at all. For this reason, we cannot demonstrate interleavings as we did for the `fork` API. In the next section, we compare the `fork` API and the `exec*` functions for executing a new program.

## Comparing process creation and process execution

Based on our discussion and the given examples in previous sections, we can make the following comparison between the two methods used for executing a new program:

*   A successful invocation of the `fork` function results in two separate processes; a parent process that has called the `fork` function and a forked (or child) process. But a successful invocation of any `exec*` function results in having the caller process substituted by a new process image and therefore no new process is created.
*   Calling the `fork` function duplicates all memory contents of the parent process, and the forked process sees the same memory contents and variables. But calling the `exec*` functions destroys the memory layout of the base process and creates a new layout based on the loaded executable.
*   A forked process has access to certain attributes of the parent process, for example, open file descriptors but using `exec*` functions. The new process doesn't know anything about it, and it doesn't inherit anything from the base process.
*   In both APIs, we end up with a new process that has only one main thread. The threads in the parent process are not forked using the fork API.
*   The `exec*` API can be used to run scripts and external executable files, but the `fork` API can be used only to create a new process that is actually the same C program.

In the next section, we'll talk about the steps that most kernels take to load and execute a new process. These steps and their details vary from one kernel to another, but we try to cover the general steps taken by most known kernels to execute a process.

# Process execution steps

To have a process executed from an executable file, the user space and the kernel space take some general steps in most operating systems. As we noted in the previous section, executable files are mostly executable object files, for example, ELF, Mach, or script files that need an interpreter to execute them.

From the user ring's point of view, a system call like `exec` should be invoked. Note that we don't explain the `fork` system call here because it is not actually an execution. It is more of a cloning operation of the currently running process.

When the user space invokes the `exec` system call, a new request for the execution of the executable file is created within the kernel. The kernel tries to find a handler for the specified executable file based on its type and according to that handler, it uses a *loader program* to load the contents of the executable file.

Note that for the script files, the executable binary of the interpreter program that is usually specified in the *shebang line* on the first line of the script. The loader program has the following duties in order to execute a process:

*   It checks the execution context and the permissions of the user that has requested the execution.
*   It allocates the memory for the new process from the main memory.
*   It copies the binary contents of the executable file into the allocated memory. This mostly involves the Data, and Text segments.
*   It allocates a memory region for the Stack segment and prepares the initial memory mappings.
*   The main thread and its Stack memory region are created.
*   It copies the command-line arguments as a *stack frame* on top of the Stack region of the main thread.
*   It initializes the vital registers that are needed for the execution.
*   It executes the first instruction of the program entry point.

In the case of script files, the path to the script files is copied as the command-line argument of the interpreter process. The preceding general steps are taken by most kernels, but the implementation details can vary greatly from a kernel to another.

For more information on a specific operating system, you need to go to its documentation or simply search for it on Google. The following articles from LWN are a great start for those seeking more details about the pro[cess execution in Linux: https:/](https://lwn.net/Articles/631631/)/lwn.[net/Articles/631631/ and https:/](https://lwn.net/Articles/630727/)/lwn.net/Articles/630727/.

In the next section, we'll start to talk about concurrency-related topics. We prepare the ground for the next chapter, which is going to talk about multi-processing-specific synchronization techniques in great depth. We start here by discussing shared states, which can be used in multi-process software systems.

# Shared states

As with threads, we can have some shared states between processes. The only difference is that the threads are able to access the same memory space owned by their owner process, but processes cannot have that luxury. Therefore, other mechanisms should be employed to share a state among a number of processes.

In this section, we are going to discuss these techniques and as part of this chapter, we focus on some of them that function as storage. In the first section, we will be discussing different techniques and trying to group them based on their nature.

## Sharing techniques

If you look at the ways you can share a state (a variable or an array) between two processes, it turns out that it can be done in a limited number of ways. Theoretically, there are two main categories of sharing a state between a number of processes, but in a real computer system, each of these categories has some subcategories.

You either have to put a state in a "place" that can be accessed by a number of processes, or you must have your state *sent* or *transferred* as a message, signal, or event to other processes. Similarly, you either have to *pull* or *retrieve* an existing state from a "place," or *receive* it as a message, signal, or event. The first approach needs storage or a *medium* like a memory buffer or a filesystem, and the second approach requires you to have a messaging mechanism or a *channel* in place between the processes.

As an example for the first approach, we can have a shared memory region as a medium with an array inside that can be accessed by a number of processes to read and modify the array. As an example for the second approach, we can have a computer network as the channel to allow some messages to be transmitted between a number of processes located on different hosts in that network.

Our current discussion on how to share states between some processes is not in fact limited to just processes; it can be applied to threads as well. Threads can also have signaling between themselves to share a state or propagate an event.

In different terminology, the techniques found in the first group that requires a *medium* such as storage to share states are called *pull-based* techniques. That's because the processes that want to read states have to pull them from storage.

The techniques in the second group that require a *channel* to transmit states are called *push-based* techniques. That's because the states are pushed (or delivered) through the channel to the receiving process and it doesn't need to pull them from a medium. We will be using these terms from now on to refer to these techniques.

The variety in push-based techniques has led to various distributed architectures in the modern software industry. The pull-based techniques are considered to be legacy in comparison to push-based techniques, and you can see it in many enterprise applications where a single central database is used to share various states throughout the entire system.

However, the push-based approach is gaining momentum these days and has led to techniques such as *event sourcing* and a number of other similar distributed approaches used for keeping all parts of a big software system consistent with each other without having all data stored in a central place.

Between the two approaches discussed, we are particularly interested in the first approach throughout this chapter. We will focus more upon the second approach in *Chapter 19*, *Single-Host IPC and Sockets*, and *Chapter 20*, *Socket Programming*. In those chapters we are going to introduce the various channels available to transmit messages between processes as part of **Inter-Process Communication** (**IPC**) techniques. Only then will we be able to explore the various push-based techniques and give some real examples for the observed concurrency issues and the control mechanisms that can be employed.

The following is a list of pull-based techniques that are supported by the POSIX standard and can be used widely in all POSIX-compliant operating systems:

*   **Shared memory**: This is simply a region in the main memory that is shared and accessible to a number of processes, and they can use it to store variables and arrays just like an ordinary memory block. A shared memory object is not a file on disk, but it is the actual memory. It can exist as a standalone object in the operating system even when there is no process using it. Shared memory objects can be removed whether by a process when not needed anymore or by rebooting the system. Therefore, in terms of surviving reboots, shared memory objects can be thought of as temporary objects.
*   **Filesystem**: Processes can use files to share states. This technique is one of the oldest techniques to share some states throughout a software system among a number of processes. Eventually, difficulties with synchronizing access to the shared files, together with many other valid reasons, have led to the invention of **Database Management Systems** (**DBMSes**), but still, the shared files are being used in certain use cases.
*   **Network services**: Once available to all processes, processes can use network storage or a network service to store and retrieve a shared state. In this scenario, the processes do not know exactly what is going on behind the scenes. They just use a network service through a well-defined API that allows them to perform certain operations on a shared state. As some examples, we can name **Network Filesystems** (**NFS**) or DBMSes. They offer network services that allow maintaining states through a well-defined model and a set of companion operations. To give a more specific example, we can mention *Relational DBMSes,* which allow you to store your states in a relational model through using SQL commands.

In the following subsections, we will be discussing each of the above methods found as part of the POSIX interface. We start with POSIX shared memory, and we show how it can lead to familiar data races known from *Chapter 16*, *Thread Synchronization*.

## POSIX shared memory

Supported by POSIX standard, shared memory is one of the widely used techniques to share a piece of information among a number of processes. Unlike threads that can access the same memory space, processes do not have this power and access to the memory of other processes is prohibited by the operating system. Therefore, we need a mechanism in order to share a portion of memory between two processes, and shared memory is exactly that technique.

In the following examples, we go through the details of creating and using a shared memory object, and we start our discussion by creating a shared memory region. The following code shows how to create and populate a shared memory object within a POSIX-compliant system:

```cpp
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#define SH_SIZE 16
int main(int argc, char** argv) {
  int shm_fd = shm_open("/shm0", O_CREAT | O_RDWR, 0600);
  if (shm_fd < 0) {
    fprintf(stderr, "ERROR: Failed to create shared memory: %s\n",
        strerror(errno));
    return 1;
  }
  fprintf(stdout, "Shared memory is created with fd: %d\n",
          shm_fd);
  if (ftruncate(shm_fd, SH_SIZE * sizeof(char)) < 0) {
    fprintf(stderr, "ERROR: Truncation failed: %s\n",
            strerror(errno));
    return 1;
  }
  fprintf(stdout, "The memory region is truncated.\n");
  void* map = mmap(0, SH_SIZE, PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (map == MAP_FAILED) {
    fprintf(stderr, "ERROR: Mapping failed: %s\n",
            strerror(errno));
    return 1;
  }
  char* ptr = (char*)map;
  ptr[0] = 'A';
  ptr[1] = 'B';
  ptr[2] = 'C';
  ptr[3] = '\n';
  ptr[4] = '\0';
  while(1);
  fprintf(stdout, "Data is written to the shared memory.\n");
  if (munmap(ptr, SH_SIZE) < 0) {
    fprintf(stderr, "ERROR: Unmapping failed: %s\n",
            strerror(errno));
    return 1;
  }
  if (close(shm_fd) < 0) {
    fprintf(stderr, "ERROR: Closing shared memory failed: %s\n",
        strerror(errno));
    return 1;
  }
  return 0;
}
```

Code Box 17-4 [ExtremeC_examples_chapter17_4.c]: Creating and writing to a POSIX shared memory object

The preceding code creates a shared memory object named `/shm0` with 16 bytes in it. Then it populates the shared memory with the literal `ABC\n` and finally, it quits by *unmapping* the shared memory region. Note that the shared memory object remains in place even when the process quits. Future processes can open and read the same shared memory object over and over again. A shared memory object is destructed either by rebooting the system or by getting *unlinked* (removed) by a process.

**Note**:

In FreeBSD, the names of the shared memory objects should start with `/`. This is not mandatory in Linux or macOS, but we did the same for both of them to remain compatible with FreeBSD.

In the preceding code, we firstly open a shared memory object using the `shm_open` function. It accepts a name and the modes that the shared memory object should be created with. `O_CREAT` and `O_RDWR` mean that the shared memory should be created, and it can be used for both reading and writing.

Note the creation won't fail if the shared memory object already exists. The last argument indicates the permissions of the shared memory object. `0600` means that it is available for reading and write operations performed by the processes that are initiated only by the owner of the shared memory object.

On the following lines, we define the size of the shared memory region by truncating it using `ftruncate` function. Note that this is a necessary step if you're about to create a new shared memory object. For the preceding shared memory object, we have defined 16 bytes to be allocated and then truncated.

As we proceed, we map the shared memory object to a region accessible by the process using the `mmap` function. As a result of this, we have a pointer to the mapped memory and that can be used to access the shared memory region behind. This is also a necessary step that makes the shared memory accessible to our C program.

The function `mmap` is usually used to map a file or a shared memory region (originally allocated from the kernel's memory space) to an address space that is accessible to the caller process. Then, the mapped address space can be accessed as a regular memory region using ordinary pointers.

As you can see, the region is mapped as a writable region indicated by `PROT_WRITE` and as a shared region among processes indicated by the `MAP_SHARED` argument. `MAP_SHARED` simply means any changes to the mapped area will be visible to other processes mapping the same region.

Instead of `MAP_SHARED`, we could have `MAP_PRIVATE`; this means that the changes to the mapped region are not propagated to other processes and are, rather, private to the mapper process. This usage is not common unless you want to use the shared memory inside a process only.

After mapping the shared memory region, the preceding code writes a null-terminated string `ABC\n` into the shared memory. Note the new line feed character at the end of the string. As the final steps, the process unmaps the shared memory region by calling the `munmap` function and then it closes the file descriptor assigned to the shared memory object.

**Note**:

Every operating system offers a different way to create an *unnamed* or *anonymous shared memory* object. In FreeBSD, it is enough to pass `SHM_ANON` as the path of the shared memory object to the `shm_open` function. In Linux, one can create an anonymous file using a `memfd_create` function instead of creating a shared memory object and use the returned file descriptor to create a mapped region. An anonymous shared memory is private to the owner process and cannot be used to share states among a number of processes.

The preceding code can be compiled on macOS, FreeBSD, and Linux systems. In Linux systems, shared memory objects can be seen inside the directory `/dev/shm`. Note that this directory doesn't have a regular filesystem and those you see are not files on a disk device. Instead, `/dev/shm` uses the `shmfs` filesystem. It is meant to expose the temporary objects created inside the memory through a mounted directory, and it is only available in Linux.

Let's compile and run *example 17.4* in Linux and examine the contents of the `/dev/shm` directory. In Linux, it is mandatory to link the final binary with the `rt` library in order to use shared memory facilities, and that's why you see the option `-lrt` in the following shell box:

```cpp
$ ls /dev/shm
$ gcc ExtremeC_examples_chapter17_4.c -lrt -o ex17_4.out
$ ./ex17_4.out
Shared memory is created with fd: 3
The memory region is truncated.
Data is written to the shared memory.
$ ls /dev/shm
shm0
$
```

Shell Box 17-4: Building and running example 17.4 and checking if the shared memory object is created

As you can see on the first line, there are no shared memory objects in the `/dev/shm` directory. On the second line, we build *example 17.4*, and on the third line, we execute the produced executable file. Then we check `/dev/shm`, and we see that we've got a new shared memory object, `shm0`, there.

The output of the program also confirms the creation of the shared memory object. Another important thing about the preceding shell box is the file descriptor `3`, which is assigned to the shared memory object.

For every file you open, a new file descriptor is opened in each process. This file is not necessarily on disk, and it can be a shared memory object, standard output, and so on. In each process, file descriptors start from 0 and go up to a maximum allowed number.

Note that in each process, the file descriptors `0`, `1`, and `2` are preassigned to the `stdout`, `stdin`, and `stderr` streams, respectively. These file descriptors are opened for every new process before having its `main` function run. That's basically why the shared memory object in the preceding example gets `3` as its file descriptor.

**Note**:

On macOS systems, you can use the `pics` utility to check active IPC objects in the system. It can show you the active message queues and shared memories. It shows you the active semaphores as well.

The `/dev/shm` directory has another interesting property. You can use the `cat` utility to see the contents of shared memory objects, but again this is only available in Linux. Let's use it on our created `shm0` object. As you see in the following shell box, the contents of the shared memory object are displayed. It is the string `ABC` plus a new line feed character `\n`:

```cpp
$ cat /dev/shm/shm0
ABC
$
```

Shell Box 17-5 Using the cat program to see the content of the shared memory object created as part of example 17.4

As we explained before, a shared memory object exists as long as it is being used by at least one process. Even if one of the processes has already asked the operating system to delete (or *unlink*) the shared memory, it won't be actually deleted until the last process has used it. Even when there is no process unlinking a shared memory object, it would be deleted when a reboot happens. Shared memory objects cannot survive reboots, and the processes should create them again in order to use them for communication.

The following example shows how a process can open and read from an already existing shared memory object and how it can unlink it finally. *Example 17.5* reads from the shared memory object created in *example 17.4*. Therefore, it can be considered as complementary to what we did in *example 17.4*:

```cpp
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#define SH_SIZE 16
int main(int argc, char** argv) {
  int shm_fd = shm_open("/shm0", O_RDONLY, 0600);
  if (shm_fd < 0) {
    fprintf(stderr, "ERROR: Failed to open shared memory: %s\n",
        strerror(errno));
    return 1;
  }
  fprintf(stdout, "Shared memory is opened with fd: %d\n", shm_fd);
  void* map = mmap(0, SH_SIZE, PROT_READ, MAP_SHARED, shm_fd, 0);
  if (map == MAP_FAILED) {
    fprintf(stderr, "ERROR: Mapping failed: %s\n",
            strerror(errno));
    return 1;
  }
  char* ptr = (char*)map;
  fprintf(stdout, "The contents of shared memory object: %s\n",
          ptr);
  if (munmap(ptr, SH_SIZE) < 0) {
    fprintf(stderr, "ERROR: Unmapping failed: %s\n",
            strerror(errno));
    return 1;
  }
  if (close(shm_fd) < 0) {
    fprintf(stderr, "ERROR: Closing shared memory fd filed: %s\n",
        strerror(errno));
    return 1;
  }
  if (shm_unlink("/shm0") < 0) {
    fprintf(stderr, "ERROR: Unlinking shared memory failed: %s\n",
        strerror(errno));
    return 1;
  }
  return 0;
}
```

Code Box 17-5 [ExtremeC_examples_chapter17_5.c]: Reading from the shared memory object created as part of example 17.4

As the first statement in the `main` function, we have opened an existing shared memory object named `/shm0`. If there is no such shared memory object, we will generate an error. As you can see, we have opened the shared memory object as read-only, meaning that we are not going to write anything to the shared memory.

On the following lines, we map the shared memory region. Again, we have indicated that the mapped region is read-only by passing the `PROT_READ` argument. After that, we finally get a pointer to the shared memory region, and we use it to print its contents. When we're done with the shared memory, we unmap the region. Following this, the assigned file descriptor is closed, and lastly the shared memory object is registered for removal by unlinking it through using the `shm_unlink` function.

After this point, when all other processes that are using the same shared memory are done with it, the shared memory object gets removed from the system. Note that the shared memory object exists as long as there is a process using it.

The following is the output of running the preceding code. Note the contents of `/dev/shm` before and after running *example 17.5*:

```cpp
$ ls /dev/shm
shm0
$ gcc ExtremeC_examples_chapter17_5.c -lrt -o ex17_5.out
$ ./ex17_5.out
Shared memory is opened with fd: 3
The contents of the shared memory object: ABC
$ ls /dev/shm
$
```

Shell Box 17-6: Reading from the shared memory object created in example 17.4 and finally removing it

### Data race example using shared memory

Now, it's time to demonstrate a data race using the combination of the fork API and shared memory. It would be analogous to the examples given in *Chapter 15*, *Thread Execution*, to demonstrate a data race among a number of threads.

In *example 17.6*, we have a counter variable that is placed inside a shared memory region. The example forks a child process out of the main running process, and both of them try to increment the shared counter. The final output shows a clear data race over the shared counter:

```cpp
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#define SH_SIZE 4
// Shared file descriptor used to refer to the
// shared memory object
int shared_fd = -1;
// The pointer to the shared counter
int32_t* counter = NULL;
void init_shared_resource() {
  // Open the shared memory object
  shared_fd = shm_open("/shm0", O_CREAT | O_RDWR, 0600);
  if (shared_fd < 0) {
    fprintf(stderr, "ERROR: Failed to create shared memory: %s\n",
        strerror(errno));
    exit(1);
  }
  fprintf(stdout, "Shared memory is created with fd: %d\n",
          shared_fd);
}
void shutdown_shared_resource() {
  if (shm_unlink("/shm0") < 0) {
    fprintf(stderr, "ERROR: Unlinking shared memory failed: %s\n",
        strerror(errno));
    exit(1);
  }
}
void inc_counter() {
  usleep(1);
  int32_t temp = *counter;
  usleep(1);
  temp++;
  usleep(1);
  *counter = temp;
  usleep(1);
}
int main(int argc, char** argv) {
  // Parent process needs to initialize the shared resource
  init_shared_resource();
  // Allocate and truncate the shared memory region
  if (ftruncate(shared_fd, SH_SIZE * sizeof(char)) < 0) {
    fprintf(stderr, "ERROR: Truncation failed: %s\n",
            strerror(errno));
    return 1;
  }
  fprintf(stdout, "The memory region is truncated.\n");
  // Map the shared memory and initialize the counter
  void* map = mmap(0, SH_SIZE, PROT_WRITE,
          MAP_SHARED, shared_fd, 0);
  if (map == MAP_FAILED) {
    fprintf(stderr, "ERROR: Mapping failed: %s\n",
            strerror(errno));
    return 1;
  }
  counter = (int32_t*)map;
  *counter = 0;
  // Fork a new process
  pid_t pid = fork();
  if (pid) { // The parent process
    // Increment the counter
    inc_counter();
    fprintf(stdout, "The parent process sees the counter as %d.\n",
        *counter);
    // Wait for the child process to exit
    int status = -1;
    wait(&status);
    fprintf(stdout, "The child process finished with status %d.\n",
        status);
  } else { // The child process
    // Incrmenet the counter
    inc_counter();
    fprintf(stdout, "The child process sees the counter as %d.\n",
        *counter);
  }
  // Both processes should unmap shared memory region and close
  // its file descriptor
  if (munmap(counter, SH_SIZE) < 0) {
    fprintf(stderr, "ERROR: Unmapping failed: %s\n",
            strerror(errno));
    return 1;
  }
  if (close(shared_fd) < 0) {
    fprintf(stderr, "ERROR: Closing shared memory fd filed: %s\n",
        strerror(errno));
    return 1;
  }
  // Only parent process needs to shutdown the shared resource
  if (pid) {
    shutdown_shared_resource();
  }
  return 0;
}
```

Code Box 17-6 [ExtremeC_examples_chapter17_6.c]: Demonstration of a data race using a POSIX shared memory and the fork API

There are three functions in the preceding code other than the `main` function. The function `init_shared_resource` creates the shared memory object. The reason that I've named this function `init_shared_resource` instead of `init_shared_memory` is the fact that we could use another pull-based technique in the preceding example and having a general name for this function allows the `main` function to remain unchanged in the future examples.

The function `shutdown_shared_resource` destructs the shared memory and unlinks it. In addition, the function `inc_counter` increments the shared counter by 1.

The `main` function truncates and maps the shared memory region just like we did in *example 17.4*. After having the shared memory region mapped, the forking logic beings. By calling the `fork` function, a new process is spawned, and both processes (the forked process and the forking process) try to increment the counter by calling the `inc_counter` function.

When the parent process writes to the shared counter, it waits for the child process to finish, and only after that, it tries to unmap, close, and unlink the shared memory object. Note that the unmapping and the closure of the file descriptor happen in both processes, but only the parent process unlinks the shared memory object.

As you can see as part of *Code Box 17-6*, we have used some unusual `usleep` calls in the `inc_counter` function. The reason is to force the scheduler to take back the CPU core from one process and give it to another process. Without these `usleep` function calls, the CPU core is not usually transferred between the processes, and you cannot see the effect of different interleavings very often.

One of the reasons for such an effect is having a small number of instructions in each process. If the number of instructions per process increases significantly, one can see the non-deterministic behavior of interleavings even without sleep calls. As an example, having a loop in each process that counts for 10,000 times and increments the shared counter in each iteration is very likely to reveal the data race. You can try this yourself.

As the final note about the preceding code, the parent process creates and opens the shared memory object and assigns a file descriptor to it before forking the child process. The forked process doesn't open the shared memory object, but it can use the same file descriptor. The fact that all open file descriptors are inherited from the parent process helped the child process to continue and use the file descriptor, referring to the same shared memory object.

The following in *Shell Box 17-7* is the output of running *example 17.6* for a number of times. As you can see, we have a clear data race over the shared counter. There are moments when the parent or the child process updates the counter without obtaining the latest modified value, and this results in printing `1` by both processes:

```cpp
$ gcc ExtremeC_examples_chapter17_6 -o ex17_6.out
$ ./ex17_6.out
Shared memory is created with fd: 3
The memory region is truncated.
The parent process sees the counter as 1.
The child process sees the counter as 2.
The child process finished with status 0.
$ ./ex17_6
...
...
...
$ ./ex17_6.out
Shared memory is created with fd: 3
The memory region is truncated.
The parent process sees the counter as 1.
The child process sees the counter as 1.
The child process finished with status 0.
$
```

Shell Box 17-7: Running example 17.6 and demonstration of the data race happening over the shared counter

In this section, we showed how to create and use shared memory. We also demonstrated a data race example and the way concurrent processes behave while accessing a shared memory region. In the following section, we're going to talk about the filesystem as another widely used pull-based method to share a state among a number of processes.

## File system

POSIX exposes a similar API for working with files in a filesystem. As long as the file descriptors are involved and they are used to refer to various system objects, the same API as that introduced for working with shared memory can be used.

We use file descriptors to refer to actual files in a filesystem like **ext4**, together with shared memory, pipes, and so on; therefore, the same semantic for opening, reading, writing, mapping them to a local memory region, and so on can be employed. Therefore, we'd expect to see similar discussion and perhaps similar C code regarding the filesystem as we had for the shared memory. We see this in *example 17.7*.

**Note**:

We usually map file descriptors. There are some exceptional cases, however, where *socket descriptors* can be mapped. Socket descriptors are similar to file descriptors but are used for network or Unix sockets. This link provides an interesting use case for mapping the kernel buffer behind a TCP socket which is referred to as a *zero-copy receive mechanism*: https://lwn.net/Articles/752188/.

Note that it's correct that the API employed for using the filesystem is very similar to the one we used for shared memory, but it doesn't mean that their implementation is similar as well. In fact, a file object in a filesystem backed by a hard disk is fundamentally different from a shared memory object. Let's briefly discuss some differences:

*   A shared memory object is basically in the memory space of the kernel process while a file in a filesystem is located on a disk. At most, such a file has some allocated buffers for reading and writing operations.
*   The states written to shared memory are wiped out by rebooting the system, but the states written to a shared file, if it is backed by a hard disk or permanent storage, can be retained after the reboot.
*   Generally, accessing shared memory is far faster than accessing the filesystem.

The following code is the same data race example that we gave for the shared memory in the previous section. Since the API used for the filesystem is pretty similar to the API we used for the shared memory, we only need to change two functions from *example 17.6*; `init_shared_resource` and `shutdown_shared_resource`. The rest will be the same. This is a great achievement that is accomplished by using the same POSIX API operating on the file descriptors. Let's get into the code:

```cpp
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#define SH_SIZE 4
// The shared file descriptor used to refer to the shared file
int shared_fd = -1;
// The pointer to the shared counter
int32_t* counter = NULL;
void init_shared_resource() {
  // Open the file
  shared_fd = open("data.bin", O_CREAT | O_RDWR, 0600);
  if (shared_fd < 0) {
    fprintf(stderr, "ERROR: Failed to create the file: %s\n",
        strerror(errno));
    exit(1);
  }
  fprintf(stdout, "File is created and opened with fd: %d\n",
          shared_fd);
}
void shutdown_shared_resource() {
  if (remove("data.bin") < 0) {
    fprintf(stderr, "ERROR: Removing the file failed: %s\n",
        strerror(errno));
    exit(1);
  }
}
void inc_counter() {
  ... As exmaple 17.6 ...
}
int main(int argc, char** argv) {
  ... As exmaple 17.6 ...
}
```

Code Box 17-7 [ExtremeC_examples_chapter17_7.c]: Demonstration of a data race using regular files and the fork API

As you see, the majority of the preceding code is obtained from *example 17.6*. The rest is a substitute for using the `open` and `remove` functions instead of the `shm_open` and `shm_unlink` functions.

Note that the file `data.bin` is created in the current directory since we've not given an absolute path to the `open` function. Running the preceding code also produces the same data race over the shared counter. It can be examined similarly to our approach for *example 17.6*.

So far, we have seen that we can use shared memory and shared files to store a state and access it from a number of processes concurrently. Now, it's time to talk about multithreading and multi-processing in a greater sense and compare them thoroughly.

# Multithreading versus multi-processing

After discussing multithreading and multi-processing in *Chapter 14*, *Synchronization*, together with concepts we have covered throughout the recent chapters, we are in a good position to compare them and give a high-level description of situations in which each of the approaches should be employed. Suppose that we are going to design a piece of software that aims to process a number of input requests concurrently. We discuss this in the context of three different situations. Let's start with the first one.

## Multithreading

The first situation is when you can write a piece of software that has only one process, and all the requests go into the same process. All the logic should be written as part of the same process, and as a result, you get a fat process that does everything in your system. Since this is single-process software, if you want to handle many requests concurrently, you need to do it in a multithreaded way by creating threads to handle multiple requests. Further, it can be a better design decision to go for a *thread pool* that has a limited number of threads.

There are the following considerations regarding concurrency and synchronization which should be taken care of. Note that we don't talk about using event loops or asynchronous I/O in this situation, while it can still be a valid alternative to multithreading.

If the number of requests increases significantly, the limited number of threads within the thread pool should be increased to overcome the demand. This literally means upgrading the hardware and resources on the machine on which the main process is running. This is called *scaling up* or *vertical scaling*. It means that you upgrade the hardware you have on a single machine to be able to respond to more requests. In addition to the possible downtime that clients experience while upgrading to the new hardware (though it can be prevented), the upgrade is costly, and you have to do another scale up when the number of requests grows again.

If processing the requests ends up in manipulating a shared state or a data store, synchronization techniques can be implemented easily, by knowing the fact that threads have access to the same memory space. Of course, this is needed whether they have a shared data structure that should be maintained or they have access to remote data storage that is not transactional.

All the threads are running on the same machine, and thus they can use all the techniques used for sharing a state that we explained so far, used by both threads and processes. This is a great feature and mitigates a lot of pain when it comes to thread synchronization.

Let's talk about the next situation, when we can have more than one process but all of them are on the same machine.

## Single-host multi-processing

In this situation, we write a piece of software that has multiple processes, but all are deployed on a single machine. All of these processes can be either single-threaded, or they can have a thread pool inside that allows each of them to handle more than one request at a time.

When the number of requests increases, one can create new processes instead of creating more threads. This is usually called *scaling out* or *horizontal scaling*. When you have only one single machine, however, you must scale it up, or in other words, you must upgrade its hardware. This can cause the same issues we mentioned for the scaling up of a multithreaded program in the previous subsection.

When it comes to concurrency, the processes are being executed in a concurrent environment. They can only use the multi-processing ways of sharing a state or synchronizing the processes. Surely, it is not as convenient as writing multithreaded code. In addition, processes can use both pull-based or push-based techniques to share states.

Multi-processing on a single machine is not very effective, and it seems multithreading is more convenient when it comes to the effort of coding.

The next subsection talks about the distributed multi-processing environment, which is the best design to create modern software.

## Distributed multi-processing

In the final situation, we have written a program that is run as multiple processes, running on multiple hosts, all connected to each other through a network, and on a single host we can have more than one process running. The following features can be seen in such a deployment.

When faced with significant growth in the number of requests, this system can be scaled out without limits. This is a great feature that enables you to use commodity hardware when you face such high peaks. Using the clusters of commodity hardware instead of powerful servers was one of the ideas that enabled Google to run its *Page Rank* and *Map* *Reduce* algorithms on a cluster of machines.

The techniques discussed in this chapter barely help because they have an important prerequisite: that all the processes are running within the same machine. Therefore, a completely different set of algorithms and techniques should be employed to make the processes synchronized and make shared states available to all processes within the system. *Latency*, *fault tolerance*, *availability*, *data consistency*, and many more factors should be studied and tuned regarding such a distributed system.

Processes on different hosts use network sockets to communicate in a push-based manner, but the processes on the same host may use local IPC techniques, for example, message queues, shared memory, pipes, and so on, to transfer messages and share state.

As a final word in this section, in the modern software industry, we prefer scaling out rather than scaling up. This will give rise to many new ideas and technologies for data storage, synchronization, message passing, and so on. It can even have an impact on the hardware design to make it suitable for horizontal scaling.

# Summary

In this chapter, we explored multi-processing systems and the various techniques that can be used to share a state among a number of processes. The following topics were covered in this chapter:

*   We introduced the POSIX APIs used for process execution. We explained how the `fork` API and `exec*` functions work.
*   We explained the steps that a kernel takes to execute a process.
*   We discussed the ways that a state can be shared among a number of processes.
*   We introduced the pull-based and push-based techniques as the two top-level categories for all other available techniques.
*   Shared memory and shared files on a filesystem are among common techniques to share a state in a pull-based manner.
*   We explained the differences and similarities of multithreading and multi-processing deployments and the concepts of vertical and horizontal scaling in a distributed software system.

In the next chapter, we are going to talk about concurrency in single-host multi-processing environments. It will consist of discussions about concurrency issues and the ways to synchronize a number of processes in order to protect a shared resource. The topics are very similar to the ones you encountered in *Chapter 16*, *Thread Synchronization*, but their focus is on the processes rather than the threads.