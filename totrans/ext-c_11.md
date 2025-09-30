# Chapter 11

# System Calls and Kernels

In the previous chapter, we discussed the history of Unix and its onion-like architecture. We also introduced and talked about the POSIX and SUS standards governing the shell ring in Unix, before explaining how the C standard library is there to provide common functionalities exposed by a Unix-compliant system.

In this chapter, we are going to continue our discussion of the *system call interface* and the Unix *kernel*. This will give us a complete insight into how a Unix system works.

After reading this chapter, you will be able to analyze the system calls a program invokes, you will be able to explain how the process lives and evolves inside the Unix environment, and you will also be able to use system calls directly or through libc. We'll also talk about Unix kernel development and show you how you can add a new system call to the Linux kernel and how it can be invoked from the shell ring.

In the last part of this chapter, we will talk about *monolithic* kernels and *microkernels* and how they differ. We will introduce the Linux kernel as a monolithic kernel, and we will write a *kernel module* for it that can be loaded and unloaded dynamically.

Let's start this chapter by talking about system calls.

# System calls

In the previous chapter, we briefly explained what a system call is. In this section, we want to take a deeper look and explain the mechanism that is used behind system calls to transfer the execution from a user process to the kernel process.

However, before we do that, we need to explain a bit more about both the kernel space and the user space, because this will be beneficial in our understanding of how the system calls work behind the scenes. We will also write a simple system call to gain some ideas about kernel development.

What we're about to do is crucial if you want to be able to write a new system call when you're going to add a new functionality into the kernel that wasn't there before. It also gives you a better understanding of the kernel space and how it differs from the user space because, in reality, they are very different.

## System calls under the microscope

As we discussed in the previous chapter, a separation happens when moving from the shell ring into the kernel ring. You find that whatever resides in the first two rings, the user application and the shell, belongs to the user space. Likewise, what ever appears in the kernel ring or the hardware ring belongs to the kernel space.

There is one rule about this separation, and that is nothing in the two most inner rings – kernel and hardware – can be accessed directly by the user space. In other words, no processes in the user space can access the hardware, internal kernel data structures, and algorithms directly. Instead, they should be accessed via system calls.

That said you may think that it seems a little contradictory to whatever you know and have experienced about Unix-like operating systems, such as Linux. If you don't see the issue, let me explain it to you. It seems to be a contradiction because, for instance, when a program reads some bytes from a network socket, it is not the program that actually reads those bytes from the network adapter. It is the kernel that reads the bytes and copies them to the user space, and then the program can pick them up and use them.

We can clarify this by going through all the steps taken from the user space to the kernel space and vice versa in an example. When you want to read a file from a hard disk drive, you write a program in the user application ring. Your program uses a libc I/O function called `fread` (or another similar function) and will eventually be running as a process in the user space. When the program makes a call to the `fread` function, the implementation behind libc gets triggered.

So far, everything is still in the user process. Then, the `fread` implementation eventually invokes a system call, while `fread` is receiving an already opened *file descriptor* as the first argument, the address of a buffer allocated in the process's memory, which is in the user space, as the second argument, and the length of the buffer as the third argument.

When the system call is triggered by the libc implementation, the kernel gets control of execution on behalf of the user process. It receives the arguments from the user space and keeps them in the kernel space. Then, it is the kernel that reads from the file by accessing the filesystem unit inside the kernel (as can be seen in *Figure 10-5* in the previous chapter).

When the `read` operation is complete in the kernel ring, the read data will be copied to the buffer in the user space, as specified by the second augment when calling the `fread` function, and the system call leaves and returns the control of execution to the user process. Meanwhile, the user process usually waits while the system call is busy with the operation. In this case, the system call is blocking.

There are some important things to note about this scenario:

*   We only have one kernel that performs all the logic behind system calls.
*   If the system call is *blocking*, when that system call is in progress, the caller user process has to wait while the system call is busy and has not finished. Conversely, if the system call is *non-blocking*, the system call returns very quickly, but the user process has to make extra system calls to check if the results are available.
*   Arguments together with input and output data will be copied from/to user space. Since the actual values are copied, system calls are supposed to be designed in such a way that they accept tiny variables and pointers as input arguments.
*   The kernel has full privileged access to all resources of a system. Therefore, there should be a mechanism to check if the user process is able to make such a system call. In this scenario, if the user is not the owner of the file, `fread` should fail with an error about the lack of required permissions.
*   A similar separation exists between the memory dedicated to the user space and the kernel space. User processes can only access the user space memory. Multiple transfers might be required in order to fulfil a certain system call.

Before we move onto the next section, I want to ask you a question. How does a system call transfer the control of execution to the kernel? Take a minute to think about that, because in the next section we're going to work on the answer to it.

## Bypassing standard C – calling a system call directly

Before answering the raised question, let's go through an example that bypasses the standard C library and calls a system call directly. In other words, the program calls a system call without going through the shell ring. As we have noted before, this is considered an anti-pattern, but when certain system calls are not exposed through libc, a user application can call the system calls directly.

In every Unix system, there is a specific method for invoking system calls directly. For example, in Linux, there is a function called `syscall` located in the `<sys/syscall.h>` header file that can be used for this purpose.

The following code box, *example 11.1*, is a different Hello World example that does not use libc to print to the standard output. In other words, the example does not use the `printf` function that can be found as part of shell ring and the POSIX standard. Instead, it invokes a specific system call directly, hence the code is only compilable on Linux machines, not other Unix systems. In other words, the code is not portable between various Unix flavors:

```cpp
// We need to have this to be able to use non-POSIX stuff
#define _GNU_SOURCE
#include <unistd.h>
// This is not part of POSIX!
#include <sys/syscall.h>
int main(int argc, char** argv) {
  char message[20] = "Hello World!\n";
  // Invokes the 'write' system call that writes
  // some bytes into the standard output.
  syscall(__NR_write, 1, message, 13);
  return 0;
}
```

Code Box 11-1 [ExtremeC_examples_chapter11_1.c]: A different Hello World example that invokes the write system call directly

As the first statement in the preceding code box, we have to define `_GNU_SOURCE` to indicate that we are going to use parts of the **GNU C Library** (**glibc**) that are not part of POSIX, or SUS standards. This breaks the portability of the program, and because of that, you may not be able to compile your code on another Unix machine. In the second `include` statement, we include one of the glibc-specific header files that doesn't exist in other POSIX systems using implementations other than glibc as their main libc backbone.

In the `main` function, we make a system call by calling the `syscall` function. First of all, we have to specify the system call by passing a number. This is an integer that refers to a specific system call. Every system call has its own unique specific *system call number* in Linux.

In the example code, the `__R_write` constant has been passed instead of the system call number, and we don't know its exact numerical value. After looking it up in the `unistd.h` header file, apparently 64 is the number of the `write` system call.

After passing the system call number, we should pass the arguments that are required for the system call.

Note that, despite the fact that the preceding code is very simple, and it just contains a simple function call, you should know that `syscall` is not an ordinary function. It is an assembly procedure that fills some proper CPU registers and actually transfers the control of execution from the user space to the kernel space. We will talk about this shortly.

For `write`, we need to pass three arguments: the file descriptor, which here is `1` to refer to the standard output; the second is the *pointer to a buffer* allocated in the user space; and finally, the *length of bytes* that should be copied from the buffer.

The following is the output of *example 11.1*, compiled and run in Ubuntu 18.04.1 using `gcc`:

```cpp
$ gcc ExtremeC_examples_chapter11_1.c -o ex11_1.out
$ ./ex11_1.out
Hello World!
$
```

Shell Box 11-1: The output of example 11.1

Now it's time to use `strace`, introduced in the previous chapter, to see the actual system calls that *example 11.1* has invoked. The output of `strace`, shown as follows, demonstrates that the program has invoked the desired system call:

```cpp
$ strace ./ex11_1.out
execve("./ex11_1.out", ["./ex11_1.out"], 0x7ffcb94306b0 /* 22 vars */) = 0
brk(NULL)                               = 0x55ebc30fb000
access("/etc/ld.so.nohwcap", F_OK)      = -1 ENOENT (No such file or directory)
access("/etc/ld.so.preload", R_OK)      = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
...
...
arch_prctl(ARCH_SET_FS, 0x7f24aa5624c0) = 0
mprotect(0x7f24aa339000, 16384, PROT_READ) = 0
mprotect(0x55ebc1e04000, 4096, PROT_READ) = 0
mprotect(0x7f24aa56a000, 4096, PROT_READ) = 0
munmap(0x7f24aa563000, 26144)           = 0
write(1, "Hello World!\n", 13Hello World!
)          = 13
exit_group(0)                           = ?
+++ exited with 0 +++
$
```

Shell Box 11-2: The output of strace while running example 11.1

As you can see as a bold in *Shell Box 11-2*, the system call has been recorded by `strace`. Look at the return value, which is `13`. It means that the system call has successfully written 13 bytes into the given file, the standard output in this case.

**Note**:

A user application should never try to use system calls directly. There are usually steps that should be taken before and after calling the system call. Libc implementations do these steps. When you're not going to use libc, you have to do these steps yourself, and you must know that these steps vary from one Unix system to another.

## Inside the syscall function

However, what happens inside the `syscall` function? Note that the current discussion is only applicable to glibc and not to the rest of the libc implementations. Firstly, we need to find `syscall` in glibc. Here is the link to the `syscall` [definition: https://github.com/lattera/glibc/blob/master/sysdeps/unix/sysv/linux/x86](https://github.com/lattera/glibc/blob/master/sysdeps/unix/sysv/linux/x86_64/syscall.S)_64/syscall.S.

If you open the preceding link in a browser, you will see that this function is written in assembly language.

**Note**:

Assembly language can be used together with C statements in a C source file. In fact, this is one of the great features of C that makes it suitable for writing an operating system. For the `syscall` function, we have a declaration written in C, but the definition is in assembly.

Here is the source code you find as part of `syscall.S`:

```cpp
/* Copyright (C) 2001-2018 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
...
   <http://www.gnu.org/licenses/>.  */
#include <sysdep.h>
/* Please consult the file sysdeps/unix/sysv/linux/x86-64/sysdep.h for
   more information about the value -4095 used below.  */
/* Usage: long syscall (syscall_number, arg1, arg2, arg3, arg4, arg5, arg6)
   We need to do some arg shifting, the syscall_number will be in
   rax.  */
	.text
ENTRY (syscall)
    movq %rdi, %rax            /* Syscall number -> rax.  */
    movq %rsi, %rdi            /* shift arg1 - arg5\.  */
    movq %rdx, %rsi
    movq %rcx, %rdx
    movq %r8, %r10
    movq %r9, %r8
    movq 8(%rsp),%r9           /* arg6 is on the stack.  */
    syscall                    /* Do the system call.  */
    cmpq $-4095, %rax          /* Check %rax for error.  */
    jae SYSCALL_ERROR_LABEL    /* Jump to error handler if error.  */
    ret                        /* Return to caller.  */
PSEUDO_END (syscall)
```

Code Box 11-2: The definition of the syscall function in glibc

These instructions are short and simple despite the fact that making a system call in this way seems to be more complex. The usage comment explains that a system call in glibc can be provided up to six arguments in each invocation.

What this means is that if the underlying kernel supports system calls with more than six arguments, glibc cannot provide certain kernel functionalities, and it should be changed to support them. Fortunately, six arguments have been fine in most cases, and for system calls that need more than six arguments, we can pass pointers to structure variables allocated in the user space memory.

In the preceding code box, after the `movq` instructions, the assembly code calls the `syscall` subroutine. It simply generates an *interrupt*, which allows a specific part of the kernel, which is waiting for such interrupts, to wake up and handle the interrupt.

As you can see on the first line of the `syscall` procedure, the system call number is moved to the `%rax` register. On the following lines, we are copying other arguments into the different registers. When the system call interrupt is fired, the kernel's interrupt handler unit picks up the call and gathers the system call number and the arguments. Then it searches its *system call table* to find the appropriate function that should be invoked on the kernel side.

An interesting point is that, by the time the interrupt handler is being executed in the CPU, the user code that has initiated the system call has left the CPU, and the kernel is doing the job. This is the main mechanism behind system calls. When you initiate a system call, the CPU changes its mode, and the kernel instructions are fetched into the CPU and the user space application is no longer being executed. That's basically why we say that the kernel performs the logic behind the system call on behalf of the user application.

In the next section, we're going to give an example of this by writing a system call that prints a hello message. It can be considered a progressive version of *example 11.1* that accepts an input string and returns a greeting string.

## Add a system call to Linux

In this section, we are going to add a new system call to the system call table of an existing Unix-like kernel. This may be the first time that most of you reading this book have written C code that is supposed to be run within the kernel space. All of the past examples that we wrote in previous chapters, and almost all of the codes that we will write in future chapters, run in the user space.

In fact, most of the programs we write are meant to be running in the user space. In fact, this is what we call *C programming* or *C development*. However, if we are going to write a C program that is supposed to run in the kernel space, we use a different name; we call it *kernel development*.

We are going through the next example, *example 11.2*, but before that we need to explore the kernel environment to see how it is different from the user space.

### Kernel development

This section will be beneficial to those of you who are seeking to be a kernel developer or a security researcher in the field of operating systems. In the first part, before jumping to the system call itself, we want to explain the differences between the kernel development and the ordinary C development.

The development of kernels is different from the development of ordinary C programs in a number of ways. Before looking at the differences, one thing we should note is that C development usually takes place in the user space.

In the following list, we have provided six of the key differences between the development efforts happening in the kernel and the user space:

*   There is only one kernel process that runs everything. This simply means that if your code causes a crash in the kernel, you probably need to reboot the machine and let the kernel become initialized again. So, with the kernel process, the development cost is very high, and you cannot try various solutions without rebooting the machine, which you can do very easily for user space programs while working on them. Upon a crash in the kernel, a *kernel crash dump* is generated, which can be used to diagnose the cause.
*   In the kernel ring, there is no C standard library like glibc! In other words, this is a realm in which SUS and POSIX standards are no longer valid. So, you cannot include any libc header files, such as `stdio.h` or `string.h`. In this case, you have a dedicated set of functions that should be used for various operations. These functions are usually located in *kernel headers* and can be different from one Unix flavor to another because there is no standardization in this field.

    As an example, if you are doing kernel development in Linux, you may use `printk` to write a message into the kernel's *message buffer*. However, in FreeBSD, you need to use the `printf` family of functions, which are different from the libc `printf` functions. You will find these `printf` functions in the `<sys/system.h>` header file in a FreeBSD system. The equivalent function while doing XNU kernel development is `os_log`. Note that XNU is the kernel of macOS.

*   You can read or modify files in the kernel, but not using libc functions. Each Unix kernel has its own method of accessing files inside the kernel ring. This is the same for all functionalities exposed through libc.
*   You have full access to the physical memory and many other services in the kernel ring. So, writing secure and reliable code is very important.
*   There is no system call mechanism in the kernel. System calls are the main user space mechanism to enable user processes to communicate with the kernel ring. So, once you're in the kernel, there is no need for it.
*   The kernel process is created by copying the kernel image into the physical memory, performed by the *boot loader*. You cannot add a new system call without having to create the kernel image from scratch and reload it again by rebooting the system. In kernels that support *kernel modules*, you can easily add or remove a module when the kernel is up and running, but you cannot do the same with system calls.

As you can see with the points we've just listed, kernel development takes place in a different flow compared to the ordinary C development. Testing written logic is not an easy task, and buggy code can cause a system crash.

In the next section, we will do our first kernel development by adding a new system call. We're doing this not because it's common to add a system call when you want to introduce a new functionality into the kernel, but we're going to give it a try in order to get familiar with kernel development.

### Writing a Hello World system call for Linux

In this section, we're going to write a new system call for Linux. There are many great sources on the internet that explain how to add a system call to an existing Linux kernel, but the following forum post, *Adding a Hello World System Call to Linux Kernel* – available at https://medium.com/anubhav-shrimal/adding-a-hello-world-system-call-to-linux-kernel-dad32875872 – was used as the basis to build my own system call in Linux.

*Example 11.2* is an advanced version of *example 11.1* that uses a different and custom system call, which we are going to write in this section. The new system call receives four arguments. The first two are for the input name and the second two are for the greeting string output. Our system call accepts a name using its first two arguments, one `char` pointer addressing an already allocated buffer in the user space and one integer as the buffer's length, and returns the greeting string using its second two arguments, a pointer that is different from the input buffer and is again allocated in the user space and another integer as its length.

**WARNING**:

Please don't perform this experiment in a Linux installation that is supposed to be used for work or home usage purposes. Run the following commands on an experimental machine, which is strongly recommended to be a virtual machine. You can easily create virtual machines by using emulator applications such as VirtualBox or VMware.

The following instructions have the potential to corrupt your system and make you lose part, if not all, of your data if they are used inappropriately or in the wrong order. Always consider some backup solutions to make a copy of your data if you're going to run the following commands on a none-experimental machine.

First of all, we need to download the latest source code of the Linux kernel. We will use the Linux GitHub repository to clone its source code and then we will pick a specific release. Version 5.3 was released on 15 September 2019, and so we're going to use this version for this example.

**Note**:

Linux is a kernel. It means that it can only be installed in the kernel ring in a Unix-like operating system, but a *Linux distribution* is a different thing. A Linux distribution has a specific version of the Linux kernel in its kernel ring and a specific version of GNU libc and Bash (or GNU shell) in its shell ring.

Each Linux distribution is usually shipped with a complete list of user applications in its external rings. So, we can say a Linux distribution is a complete operating system. Note that, *Linux distribution*, *Linux distro*, and *Linux flavor* all refer to the same thing.

In this example, I'm using the Ubuntu 18.04.1 Linux distribution on a 64-bit machine.

Before we start, it's vital to make sure that the prerequisite packages are installed by running the following commands:

```cpp
$ sudo apt-get update
$ sudo apt-get install -y build-essential autoconf libncurses5-dev libssl-dev bison flex libelf-dev git
...
...
$
```

Shell Box 11-3: Installing the prerequisite packages required for example 11.2

Some notes about the preceding instructions: `apt` is the main package manager in Debian-based Linux distributions, while `sudo` is a utility program that we use to run a command in *superuser* mode. It is available on almost every Unix-like operating system.

The next step is to clone the Linux GitHub repository. We also need to check out the release 5.3 after cloning the repository. The version can be checked out by using the release tag name, as you can see in the following commands:

```cpp
$ git clone https://github.com/torvalds/linux
$ cd linux
$ git checkout v5.3
$
```

Shell Box 11-4: Cloning the Linux kernel and checking out version 5.3

Now, if you look at the files in the root directory, you will see lots of files and directories that combined build up the Linux kernel code base:

```cpp
$ ls
total 760K
drwxrwxr-x  33 kamran kamran 4.0K Jan 28  2018 arch
drwxrwxr-x   3 kamran kamran 4.0K Oct 16 22:11 block
drwxrwxr-x   2 kamran kamran 4.0K Oct 16 22:11 certs
...
drwxrwxr-x 125 kamran kamran  12K Oct 16 22:11 Documentation
drwxrwxr-x 132 kamran kamran 4.0K Oct 16 22:11 drivers
-rw-rw-r--   1 kamran kamran 3.4K Oct 16 22:11 dropped.txt
drwxrwxr-x   2 kamran kamran 4.0K Jan 28  2018 firmare
drwxrwxr-x  75 kamraln kamran 4.0K Oct 16 22:11 fs
drwxrwxr-x  27 kamran kamran 4.0K Jan 28  2018 include
...
-rw-rw-r--   1 kamran kamran  287 Jan 28  2018 Kconfig
drwxrwxr-x  17 kamran kamran 4.0K Oct 16 22:11 kernel
drwxrwxr-x  13 kamran kamran  12K Oct 16 22:11 lib
-rw-rw-r--   1 kamran kamran 429K Oct 16 22:11 MAINTAINERS
-rw-rw-r--   1 kamran kamran  61K Oct 16 22:11 Makefile
drwxrwxr-x   3 kamran kamran 4.0K Oct 16 22:11 mm
drwxrwxr-x  69 kamran kamran 4.0K Jan 28  2018 net
-rw-rw-r--   1 kamran kamran  722 Jan 28  2018 README
drwxrwxr-x  28 kamran kamran 4.0K Jan 28  2018 samples
drwxrwxr-x  14 kamran kamran 4.0K Oct 16 22:11 scripts
...
drwxrwxr-x   4 kamran kamran 4.0K Jan 28  2018 virt
drwxrwxr-x   5 kamran kamran 4.0K Oct 16 22:11 zfs
$
```

Shell Box 11-5: The content of the Linux kernel code base

As you can see, there are directories that might seem familiar: `fs`, `mm`, `net`, `arch`, and so on. I should point out that we are not going to give more details on each of these directories as it can vary massively from a kernel to another, but one common feature is that all kernels follow almost the same internal structure.

Now that we have the kernel source, we should begin to add our new Hello World system call. However, before we do that, we need to choose a unique numerical identifier for our system call; in this case, I give it the name `hello_world`, and I choose `999` as its number.

Firstly, we need to add the system call function declaration to the end of the `include/linux/syscalls.h` header file. After this modification, the file should look like this:

```cpp
/*
 * syscalls.h - Linux syscall interfaces (non-arch-specific)
 *
 * Copyright (c) 2004 Randy Dunlap
 * Copyright (c) 2004 Open Source Development Labs
 *
 * This file is released under the GPLv2.
 * See the file COPYING for more details.
 */
#ifndef _LINUX_SYSCALLS_H
#define _LINUX_SYSCALLS_H
struct epoll_event;
struct iattr;
struct inode;
...
asmlinkage long sys_statx(int dfd, const char __user *path, unsigned flags,
                          unsigned mask, struct statx __user *buffer);
asmlinkage long sys_hello_world(const char __user *str,
 const size_t str_len,
 char __user *buf,
 size_t buf_len);
#endif
```

Code Box 11-3 [include/linux/syscalls.h]: Declaration of the new Hello World system call

The description at the top says that this is a header file that contains the Linux `syscall` interfaces, which are not *architecture specific*. This means that on all architectures, Linux exposes the same set of system calls.

At the end of the file, we have declared our system call function, which accepts four arguments. As we have explained before, the first two arguments are the input string and its length, and the second two arguments are the output string and its length.

Note that input arguments are `const`, but the output arguments are not. Additionally, the `__user` identifier means that the pointers are pointing to memory addresses within the user space. As you can see, every system call has an integer value being returned as part of its function signature, which will actually be its execution result. The range of returned values and their meanings is different from one system call to another. In the case of our system call, `0` means success and any other number is a failure.

We now need to define our system call. To do this, we must first create a folder named `hello_world` in the root directory, which we accomplish using the following commands:

```cpp
$ mkdir hello_world
$ cd hello_world
$
```

Shell Box 11-6: Creating the hello_world directory

Next, we create a file named `sys_hello_world.c` inside the `hello_world` directory. The contents of this file should be as follows:

```cpp
#include <linux/kernel.h>   // For printk
#include <linux/string.h>   // For strcpy, strcat, strlen
#include <linux/slab.h>     // For kmalloc, kfree
#include <linux/uaccess.h>  // For copy_from_user, copy_to_user
#include <linux/syscalls.h> // For SYSCALL_DEFINE4
// Definition of the system call
SYSCALL_DEFINE4(hello_world,
          const char __user *, str,    // Input name
          const unsigned int, str_len, // Length of input name
          char __user *, buf,          // Output buffer
          unsigned int, buf_len) {     // Length of output buffer
  // The kernel stack variable supposed to keep the content
  // of the input buffer
  char name[64];
  // The kernel stack variable supposed to keep the final
  // output message.
  char message[96];
  printk("System call fired!\n");
  if (str_len >= 64) {
    printk("Too long input string.\n");
    return -1;
  }
  // Copy data from user space into kernel space
  if (copy_from_user(name, str, str_len)) {
    printk("Copy from user space failed.\n");
    return -2;
  }
  // Build up the final message
  strcpy(message, "Hello ");
  strcat(message, name);
  strcat(message, "!");
  // Check if the final message can be fit into the output binary
  if (strlen(message) >= (buf_len - 1)) {
    printk("Too small output buffer.\n");
    return -3;
  }
  // Copy back the message from the kernel space to the user space
  if (copy_to_user(buf, message, strlen(message) + 1)) {
    printk("Copy to user space failed.\n");
    return -4;
  }
  // Print the sent message into the kernel log
  printk("Message: %s\n", message);
  return 0;
}
```

Code Box 11-4: The definition of the Hello World system call

In the *Code Box 11-4*, we have used the `SYSCALL_DEFINE4` macro to define our function definition, with the `DEFINE4` suffix simply meaning that it accepts four arguments.

At the beginning of the function body, we have declared two-character arrays on the top of the kernel Stack. Much like ordinary processes, the kernel process has an address space that contains a Stack. After we've achieved that, we copy the data from the user space into the kernel space. Following that, we create the greeting message by concatenating some strings. This string is still in the kernel memory. Finally, we copy back the message to the user space and make it available for the caller process.

In the case of errors, appropriate error numbers are returned in order to let the caller process know about the result of the system call.

The next step to make our system call work is to update one more table. There is only one system call table for both x86 and x64 architectures, and the newly added system calls should be added to this table to become exposed.

Only after this step the system calls are available in x86 and x64 machines. To add the system call to the table, we need to add `hello_word` and its function name, `sys_hello_world`.

To do this, open the `arch/x86/entry/syscalls/syscall_64.tbl` file and add the following line to the end of the file:

```cpp
999      64     hello_world             __x64_sys_hello_world
```

Code Box 11-5: Adding the newly added Hello World system call to the system call table

After the modification, the file should look like this:

```cpp
$ cat arch/x86/entry/syscalls/syscall_64.tbl
...
...
546     x32     preadv2                 __x32_compat_sys_preadv64v2
547     x32     pwritev2                __x32_compat_sys_pwritev64v2
999      64     hello_world             __x64_sys_hello_world
$
```

Shell Box 11-7: Hello World system call added to the system call table

Note the `__x64_` prefix in the name of the system call. This is an indication that the system call is only exposed in x64 systems.

The Linux kernel uses the Make build system to compile all the source files and build the final kernel image. Moving on, you must make a file named `Makefile` in the `hello_world` directory. Its content, which is a single line of text, should be the following:

```cpp
obj-y := sys_hello_world.o
```

Code Box 11-6: Makefile of the Hello World system call

Then, you need to add `hello_world` directory to the main `Makefile` in the root directory. Change to the kernel's root directory, open the `Makefile` file, and find the following line:

```cpp
core-y  += kernel/certs/mm/fs/ipc/security/crypto/block/
```

Code Box 11-7: The target line that should be modified in the root Makefile

Add `hello_world/` to this list. All of these directories are simply the directories that should be built as part of the kernel.

We need to add the directory of the Hello World system call in order to include it in the build process and have it included in the final kernel image. The line should look like the following code after the modification:

```cpp
core-y  += kernel/certs/mm/fs/hello_world/ipc/security/crypto/block/
```

Code Box 11-8: The target line after modification

The next step is to build the kernel.

### Building the kernel

To build the kernel, we must first go back to the kernel's root directory because before we start to build the kernel, you need to provide a configuration. A configuration has a list of features and units that should be built as part of the build process.

The following command tries to make the target configuration based on the current Linux kernel's configuration. It uses the existing values in your kernel and asks you about confirmation if a newer configuration value exists in the kernel we are trying to build. If it does, you can simply accept all newer versions by just pressing the `Enter` key:

```cpp
$ make localmodconfig
...
...
#
# configuration written to .config
#
$
```

Shell Box 11-8: Creating a kernel configuration based on the current running kernel

Now you can start the build process. Since the Linux kernel contains a lot of source files, the build can take hours to complete. Therefore, we need to run the compilations in parallel.

If you're using a virtual machine, please configure your machine to have more than one core in order to have an effective boost in the build process:

```cpp
$ make -j4
SYSHDR  arch/x86/include/generated/asm/unistd_32_ia32.h
SYSTBL  arch/x86/include/generated/asm/syscalls_32.h
HOSTCC  scripts/basic/bin2c
SYSHDR  arch/x86/include/generated/asm/unistd_64_x32.h
...
...
UPD     include/generated/compile.h
CC      init/main.o
CC      hello_world/sys_hello_world.o
CC      arch/x86/crypto/crc32c-intel_glue.o
...
...
LD [M]  net/netfilter/x_tables.ko
LD [M]  net/netfilter/xt_tcpudp.ko
LD [M]  net/sched/sch_fq_codel.ko
LD [M]  sound/ac97_bus.ko
LD [M]  sound/core/snd-pcm.ko
LD [M]  sound/core/snd.ko
LD [M]  sound/core/snd-timer.ko
LD [M]  sound/pci/ac97/snd-ac97-codec.ko
LD [M]  sound/pci/snd-intel8x0.ko
LD [M]  sound/soundcore.ko
$
```

Shell Box 11-9: Output of the kernel build. Please note the line indicating the compilation of the Hello World system call

**Note**:

Make sure that you have installed the prerequisite packages introduced in the very first part of this section; otherwise, you will get compilation errors.

As you can see, the build process has started with four jobs trying to compile C files in parallel. You need to wait for it to complete. When it's finished, you can easily install the new kernel and reboot the machine:

```cpp
$ sudo make modules_install install
INSTALL arch/x86/crypto/aes-x86_64.ko
INSTALL arch/x86/crypto/aesni-intel.ko
INSTALL arch/x86/crypto/crc32-pclmul.ko
INSTALL arch/x86/crypto/crct10dif-pclmul.ko
...
...
run-parts: executing /et/knel/postinst.d/initam-tools 5.3.0+ /boot/vmlinuz-5.3.0+
update-iniras: Generating /boot/initrd.img-5.3.0+
run-parts: executing /etc/keneostinst.d/unattende-urades 5.3.0+ /boot/vmlinuz-5.3.0+
...
...
Found initrd image: /boot/initrd.img-4.15.0-36-generic
Found linux image: /boot/vmlinuz-4.15.0-29-generic
Found initrd image: /boot/initrd.img-4.15.0-29-generic
done.  
$
```

Shell Box 11-10: Creating and installing the new kernel image

As you can see, a new kernel image for the version 5.3.0 has been created and installed. Now we 're ready to reboot the system. Don't forget to check the current kernel's version before rebooting if you don't know it. In my case, my version is `4.15.0-36-generic`. I've used the following commands to find it out:

```cpp
$ uname -r
4.15.0-36-generic $
```

Shell Box 11-11: Checking the version of the currently installed kernel

Now, reboot the system using the following command:

```cpp
$ sudo reboot
```

Shell Box 11-12: Rebooting the system

While the system is booting up, the new kernel image will be picked up and used. Note that boot loaders won't pick up the older kernels; therefore, if you've had a kernel with version above 5.3, you are going to need to load the built kernel image manually. This link can help you with that: https://askubuntu.com/questions/82140/how-can-i-boot-with-an-older-kernel-version.

When the operating system boot is complete, you should have the new kernel running. Check the version. It must look like this:

```cpp
$ uname -r
5.3.0+
$
```

Shell Box 11-13: Checking the kernel version after the reboot.

If everything has gone well, the new kernel should be in place. Now we can continue to write a C program that invokes our newly added Hello World system call. It will be very similar to *example 11.1*, that called the `write` system call. You can find *example 11.2* next:

```cpp
// We need to have this to be able to use non-POSIX stuff
#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
// This is not part of POSIX!
#include <sys/syscall.h>
int main(int argc, char** argv) {
  char str[20] = "Kam";
  char message[64] = "";
  // Call the hello world system call
  int ret_val = syscall(999, str, 4, message, 64);
  if (ret_val < 0) {
    printf("[ERR] Ret val: %d\n", ret_val);
    return 1;
  }
  printf("Message: %s\n", message);
  return 0;
}
```

Code Box 11-9 [ExtremeC_examples_chapter11_2.c]: Example 11.2 invoking the newly added Hello World system call

As you can see, we have invoked the system call with the number `999`. We pass `Kam` as the input, and we expect to receive `Hello Kam!` as the greeting message. The program waits for the result and prints the message buffer that is filled by the system call in the kernel space.

In the following code, we build and run the example:

```cpp
$ gcc ExtremeC_examples_chapter11_2.c -o ex11_2.out
$ ./ex11_2.out
Message: Hello Kam!
$
```

Shell Box 11-14: Compiling and running example 11.2

After running the example, and if you look at the kernel logs using the `dmesg` command, you will see the generated logs using `printk`:

```cpp
$ dmesg
...
...
[  112.273783] System call fired!
[  112.273786] Message: Hello Kam!
$
```

Shell Box 11-15: Using dmesg to see the logs generated by the Hello World system call

If you run *example 11.2* with `strace`, you can see that it actually calls system call `999`. You can see it in the line starting with `syscall_0x3e7(...)`. Note that `0x3e7` is the hexadecimal value for 999:

```cpp
$ strace ./ex11_2.out
...
...
mprotect(0x557266020000, 4096, PROT_READ) = 0
mprotect(0x7f8dd6d2d000, 4096, PROT_READ) = 0
munmap(0x7f8dd6d26000, 27048)           = 0
syscall_0x3e7(0x7fffe7d2af30, 0x4, 0x7fffe7d2af50, 0x40, 0x7f8dd6b01d80, 0x7fffe7d2b088) = 0
fstat(1, {st_mode=S_IFCHR|0620, st_rdev=makedev(136, 0), ...}) = 0
brk(NULL)                               = 0x5572674f2000
brk(0x557267513000)
...
...
exit_group(0)                           = ?
+++ exited with 0 +++
$
```

Shell Box 11-16: Monitoring the system calls made by example 11.2

In *Shell Box 11-16*, you can see that `syscall_0x3e7` has been called and `0` has been returned. If you change the code in *example 11.2* to pass a name with more than 64 bytes, you will receive an error. Let's change the example and run it again:

```cpp
int main(int argc, char** argv) {
  char name[84] = "A very very long message! It is really hard to produce a big string!";
  char message[64] = "";
  ...
  return 0;
}
```

Code Box 11-10: Passing a long message (more than 64 bytes) to our Hello World system call

Let's compile and run it again:

```cpp
$ gcc ExtremeC_examples_chapter11_2.c -o ex11_2.out
$ ./ex11_2.out
[ERR] Ret val: -1
$
```

Shell Box 11-17: Compiling and running example 11.2 after the modification

As you see, the system call returns `-1` based on the logic we have written for it. Running with `strace` also shows that system call has returned `-1`:

```cpp
$ strace ./ex11_2.out
...
...
munmap(0x7f1a900a5000, 27048)           = 0
syscall_0x3e7(0x7ffdf74e10f0, 0x54, 0x7ffdf74e1110, 0x40, 0x7f1a8fe80d80, 0x7ffdf74e1248) = -1 (errno 1)
fstat(1, {st_mode=S_IFCHR|0620, st_rdev=makedev(136, 0), ...}) = 0
brk(NULL)                               = 0x5646802e2000
...
...
exit_group(1)                           = ?
+++ exited with 1 +++
$
```

Shell Box 11-18: Monitoring the system calls made by example 11.2 after the modification

In the next section, we talk about the approaches we can take in designing kernels. As part of our discussion, we introduce the kernel modules and explore how they are used in kernel development.

# Unix kernels

In this section, we are going to talk about the architectures that Unix kernels have been developed with throughout the last 30 years. Before talking about the different types of kernels, and there are not very many, we should know that there is no standardization about the way a kernel should be designed.

The best practices that we have obtained are based on our experiences over the years, and they have led us to a high-level picture of the internal units in a Unix kernel, which results in illustrations such as *Figure 10-5* in the previous chapter. Therefore, each kernel is somewhat different in comparison to another. The main thing that all of them have in common is that they should expose their functionalities through a system call interface. However, every kernel has its own way of handling system calls.

This variety and the debates around it have made it one of the hottest computer architecture-related topics of the 1990s, with large groups of people taking part in these debates – the *Tanenbaum-Torvalds* debate being considered the most famous one.

We are not going to go into the details of these debates, but we want to talk a bit about the two major dominant architectures for designing a Unix kernel: *monolithic* and *microkernel*. There are still other architectures, such as *hybrid kernels*, *nanokernels*, and *exokernels*, all of which have their own specific usages.

We, however, are going to focus on monolithic kernels and microkernels by creating a comparison so that we can learn about their characteristics.

## Monolithic kernels versus microkernels

In the previous chapter where we looked at Unix architecture, we described the kernel as a single process containing many units, but in reality, we were actually talking about a monolithic kernel.

A monolithic kernel is made up of one kernel process with one address space that contains multiple smaller units within the same process. Microkernels take the opposite approach. A microkernel is a minimal kernel process that tries to push out services such as filesystem, device drivers, and process management to the user space in order to make the kernel process smaller and thinner.

Both of these architectures have advantages and disadvantages, and as a result, they've been the topic of one of the most famous debates in the history of operating systems. It goes back to 1992, just after the release of the first version of Linux. A debate was started on *Usenet* by a post written by **Andrew S. Tanenbaum**. The debate is known as the Tanenbaum-Torvalds debate. You can read more at https://en.wikipedia.org/wiki/Tanenbaum–Torvalds_debate.

That post was the starting point for a flame war between the Linux creator **Linus Torvalds** and Tanenbaum and a bunch of other enthusiasts, who later became the first Linux developers. They were debating the nature of monolithic kernels and microkernels. Many different aspects of kernel design and the influence of hardware architecture on kernel design were discussed as part of this flame war.

Further discussion of the debates and topics described would be lengthy and complex and therefore beyond the scope of this book, but we want to compare these two approaches and let you get familiar with the advantages and disadvantages of each approach.

The following is a list of differences between monolithic kernels and microkernels:

*   A monolithic kernel is made up of a single process containing all the services provided by the kernel. Most early Unix kernels were developed like this, and it is considered to be an old approach. Microkernels are different because they have separate processes for every service the kernel offers.
*   A monolithic kernel process resides in the kernel space, whereas the *server processes* in a microkernel are usually in the user space. Server processes are those processes that provide the kernel's functionalities, such as memory management, filesystem, and so on. Microkernels are different in that they let server processes be in the user space. This means some operating systems are more microkernel-like than the others.
*   Monolithic kernels are usually faster. That's because all kernel services are performed inside the kernel process, but microkernels need to do some *message passing* between the user space and the kernel space, hence more system calls and context switches.
*   In a monolithic kernel, all device drivers are loaded into the kernel. Therefore, device drivers written by third-party vendors will be run as a part of the kernel. Any flaw in any device driver or any other unit inside the kernel may lead to a kernel crash. This is not the case with microkernels because all of the device drivers and many other units are run in the user space, which we could hypothesize as the reason why monolithic kernels are not used in mission-critical projects.
*   In monolithic kernels, injecting a small piece of malicious code is enough to compromise the whole kernel, and subsequently the whole system. However, this can't happen easily in a microkernel because many server processes are in the user space, and only a minimal set of critical functionalities are concentrated in the kernel space.
*   In a monolithic kernel, even a simple change to the kernel source needs the whole kernel to be compiled again, and a new kernel image should be generated. Loading the new image also requires the machine to be rebooted. But changes in a microkernel can lead to a compilation of only a specific server process, and probably loading the new functionality without rebooting the system. In monolithic kernels, a similar functionality can be obtained to some extent using kernel modules.

MINIX is one of the best-known examples of microkernels. It was written by Andrew S. Tanenbaum and was initiated as an educational operating system. Linus Torvalds used MINIX as his development environment to write his own kernel, called Linux, in 1991 for the 80386 microprocessor.

As Linux has been the biggest and most successful defender of monolithic kernels for nearly 30 years, we're going to talk more about Linux in the next section.

## Linux

You've already been introduced to the Linux kernel in the previous section of this chapter, when we were developing a new system call for it. In this section, we want to focus a bit more on the fact that Linux is monolithic and that every kernel functionality is inside the kernel.

However, there should be a way to add a new functionality to the kernel without needing it to be recompiled. New functionalities cannot be added to the kernel as new system calls simply because, as you saw, by adding a new system call, many fundamental files need to be changed, and this means we need to recompile the kernel in order to have the new functionalities.

The new approach is different. In this technique, kernel modules are written and plugged into the kernel dynamically, which we will discuss in the first section, before moving on to writing a kernel module for Linux.

## Kernel modules

Monolithic kernels are usually equipped with another facility that enables kernel developers to hot-plug new functionalities into an up-and-running kernel. These pluggable units are called kernel modules. These are not the same as server processes in microkernels.

Unlike server processes in a microkernel, which are in fact separate processes using IPC techniques to communicate with each other, kernel modules are *kernel object files* that are already compiled and can be loaded dynamically into the kernel process. These kernel object files can either become statically built as part of the kernel image or become loaded dynamically when the kernel is up and running.

Note that the kernel object files are twin concepts to the ordinary object files produced in C development.

It's worth noting again that if the kernel module does something bad inside the kernel, a *kernel crash* can happen.

The way you communicate with kernel modules is different from system calls, and they cannot be used by calling a function or using a given API. Generally, there are three ways to communicate with a kernel module in Linux and some similar operating systems:

*   **Device files in the /dev directory**: Kernel modules are mainly developed to be used by device drivers, and that's why devices are the most common way to communicate with kernel modules. As we explained in the previous chapter, devices are accessible as device files located in the `/dev` directory. You can read from and write to these files and, using them, you can send and receive data to/from the modules.
*   **Entries in procfs**: Entries in the `/proc` directory can be used to read meta-information about a specific kernel module. These files can also be used to pass meta-information or control commands to a kernel module. We shortly demonstrate the usage of procfs in the next example, *example 11.3*, as part of the following section.
*   **Entries in sysfs**: This is another filesystem in Linux that allows scripts and users to control user processes and other kernel-related units, such as kernel modules. It can be considered as a new version of procfs.

In fact, the best way to see a kernel module is to write one, which is what we are going to do in the next section, where we write a Hello World kernel module for Linux. Note that kernel modules are not limited to Linux; monolithic kernels such as FreeBSD also benefit from the kernel module mechanism.

### Adding a kernel module to Linux

In this section, we are going to write a new kernel module for Linux. This is the Hello World kernel module, which creates an entry in procfs. Then, using this entry, we read the greeting string.

In this section, you will become familiar with writing a kernel module, compiling it, loading it into the kernel, unloading it from the kernel, and reading data from a procfs entry. The main purpose of this example is to get your hands dirty with writing a kernel module and, as a result more development can be done by yourself.

**Note**:

Kernel modules are compiled into kernel object files that can be loaded directly into the kernel at run-time. There is no need to reboot the system after loading the kernel module object file as long as it doesn't do something bad in the kernel that leads to a kernel crash. That's also true for unloading the kernel module.

The first step is to create a directory that is supposed to contain all files related to the kernel module. We name it `ex11_3` since this is the third example in this chapter:

```cpp
$ mkdir ex11_3
$ cd ex11_3
$
```

Shell Box 11-19: Making the root directory for example 11.3

Then, create a file named `hwkm.c`, which is just an acronym made up of the first letters of "Hello World Kernel Module," with the following content:

```cpp
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/proc_fs.h>
// The structure pointing to the proc file
struct proc_dir_entry *proc_file;
// The read callback function
ssize_t proc_file_read(struct file *file, char __user *ubuf, size_t count, loff_t *ppos) {
  int copied = 0;
  if (*ppos > 0) {
    return 0;
  }
  copied = sprintf(ubuf, "Hello World From Kernel Module!\n");
  *ppos = copied;
  return copied;
}
static const struct file_operations proc_file_fops = {
 .owner = THIS_MODULE,
 .read  = proc_file_read
};
// The module initialization callback
static int __init hwkm_init(void) {
  proc_file = proc_create("hwkm", 0, NULL, &proc_file_fops);
  if (!proc_file) {
    return -ENOMEM;
  }
  printk("Hello World module is loaded.\n");
  return 0;
}
// The module exit callback
static void __exit hkwm_exit(void) {
  proc_remove(proc_file);
  printk("Goodbye World!\n");
}
// Defining module callbacks
module_init(hwkm_init);
module_exit(hkwm_exit);
```

Code Box 11-11 [ex11_3/hwkm.c]: The Hello World kernel module

Using the two last statements in *Code Box 11-11*, we have registered the module's initialization and exit callbacks. These functions are called when the module is being loaded and unloaded respectively. The initialization callback is the first code to be executed.

As you can see inside the `hwkm_init` function, it creates a file named `hwkm` inside the `/proc` directory. There is also an exit callback. Inside the `hwkm_exit` function, it removes the `hwkm` file from the `/proc` path. The `/proc/hwkm` file is the contact point for the user space to be able to communicate with the kernel module.

The `proc_file_read` function is the read callback function. This function is called when the user space tries to read the `/proc/hwkm` file. As you will soon see, we use the `cat` utility program to read the file. It simply copies the `Hello World From Kernel Module!` string to the user space.

Note that at this stage, the code written inside a kernel module has total access to almost anything inside the kernel, and it can leak out any kind of information to the user space. This is a major security issue, and further reading about the best practices for writing a secure kernel module should be undertaken.

To compile the preceding code, we need to use an appropriate compiler, including possibly linking it with the appropriate libraries. In order to make life easier, we create a file named `Makefile` that will trigger the necessary build tools in order to build the kernel module.

The following code box shows the content of the `Makefile`:

```cpp
obj-m += hwkm.o
all:
    make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
clean:
    make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
```

Code Box 11-12: Makefile of the Hello World kernel module

Then, we can run the `make` command. The following shell box demonstrates this:

```cpp
$ make
make -C /lib/modules/54.318.0+/build M=/home/kamran/extreme_c/ch11/codes/ex11_3 modules
make[1]: Entering directory '/home/kamran/linux'
  CC [M]  /home/kamran/extreme_c/ch11/codes/ex11_3/hwkm.o
  Building modules, stage 2.
  MODPOST 1 modules
WARNING: modpost: missing MODULE_LICENSE() in /home/kamran/extreme_c/ch11/codes/ex11_3/hwkm.o
see include/linux/module.h for more information
  CC      /home/kamran/extreme_c/ch11/codes/ex11_3/hwkm.mod.o
  LD [M]  /home/kamran/extreme_c/ch11/codes/ex11_3/hwkm.ko
make[1]: Leaving directory '/home/kamran/linux'
$
```

Shell Box 11-20: Building the Hello World kernel module

As you can see, the compiler compiles the code and produces an object file. Then, it continues by linking the object file with other libraries to create a `.ko` file. Now, if you look at the generated files, you find a file named `hwkm.ko`.

Notice the `.ko` extension, which simply means that the output file is a kernel object file. It is something like a shared library that can be dynamically loaded into the kernel and become running.

Please note that in *Shell Box 11-20*, the build process has produced a warning message. It says that the module has no license associated with it. It is a highly recommended practice to generate licensed modules when developing or deploying kernel modules in test and production environments.

The following shell box shows the list of files that can be found after building the kernel module:

```cpp
$ ls -l
total 556
-rw-rw-r-- 1 kamran kamran    154 Oct 19 00:36 Makefile
-rw-rw-r-- 1 kamran kamran      0 Oct 19 08:15 Module.symvers
-rw-rw-r-- 1 kamran kamran   1104 Oct 19 08:05 hwkm.c
-rw-rw-r-- 1 kamran kamran 272280 Oct 19 08:15 hwkm.ko
-rw-rw-r-- 1 kamran kamran    596 Oct 19 08:15 hwkm.mod.c
-rw-rw-r-- 1 kamran kamran 104488 Oct 19 08:15 hwkm.mod.o
-rw-rw-r-- 1 kamran kamran 169272 Oct 19 08:15 hwkm.o
-rw-rw-r-- 1 kamran kamran     54 Oct 19 08:15 modules.order
$
```

Shell Box 11-21: List of existing files after building the Hello World kernel module

**Note**:

We have used module build tools from Linux kernel version 5.3.0 You might get a compilation error if you compile this example using a kernel version below 3.10.

To load the `hwkm` kernel module, we use the `insmod` command in Linux, which simply loads and installs the kernel module, as we have done in the following shell box:

```cpp
$ sudo insmod hwkm.ko
$
```

Shell Box 11-22: Loading and installing the Hello World kernel module

Now, if you look at the kernel logs, you will see the lines that are produced by the initializer function. Just use the `dmesg` command to see the latest kernel logs, which is what we have done next:

```cpp
$ dmesg
...
...
[ 7411.519575] Hello World module is loaded.
$
```

Shell Box 11-23: Checking the kernel log messages after installing the kernel module

Now, the module has been loaded, and the `/proc/hwkm` file should have been created. We can read it now by using the `cat` command:

```cpp
$ cat /proc/hwkm
Hello World From Kernel Module!
$ cat /proc/hwkm
Hello World From Kernel Module!
$
```

Shell Box 11-24: Reading the/proc/hwkm file using cat

As you can see in the preceding shell box, we have read the file twice, and both times, it returns the same `Hello World From Kernel Module!` string. Note that the string is copied into the user space by the kernel module, and the `cat` program has just printed it to the standard output.

When it comes to unloading the module, we can use the `rmmod` command in Linux, as we have done next:

```cpp
$ sudo rmmod hwkm
$
```

Shell Box 11-25: Unloading the Hello World kernel module

Now that the module has been unloaded, look at the kernel logs again to see the goodbye message:

```cpp
$ dmesg
...
...
[ 7411.519575] Hello World module is loaded.
[ 7648.950639] Goodbye World!
$
```

Shell Box 11-26: Checking the kernel log messages after unloading the kernel module

As you saw in the preceding example, kernel modules are very handy when it comes to writing kernel codes.

To finish off this chapter, I believe it would be helpful to give you a list of the features that we have seen so far regarding kernel modules:

*   Kernel modules can be loaded and unloaded without needing to reboot the machine.
*   When loaded, they become part of the kernel and can access any unit or structure within the kernel. This can be thought of as a vulnerability, but a Linux kernel can be protected against installing unwanted modules.
*   In the case of kernel modules, you only need to compile their source code. But for system calls, you have to compile the whole kernel, which can easily take an hour of your time.

Finally, kernel modules can be handy when you are going to develop a code that needs to be run within the kernel behind a system call. The logic that is going to be exposed using a system call can be loaded into the kernel using a kernel module first, and after being developed and tested properly, it can go behind a real system call.

Developing system calls from scratch can be a tedious job because you have to reboot your machine countless times. Having the logic firstly written and tested as part of a kernel module can ease the pain of kernel development. Note that if your code is trying to cause a kernel crash, it doesn't matter if it is in a kernel module or behind a system call; it causes a kernel crash and you must reboot your machine.

In this section, we talked about various types of kernels. We also showed how a kernel module can be used within a monolithic kernel to have transient kernel logic by loading and unloading it dynamically.

# Summary

We've now completed our two-chapter discussion about Unix. In this chapter, we learned about the following:

*   What a system call is and how it exposes a certain functionality
*   What happens behind the invocation of a system call
*   How a certain system call can be invoked from C code directly
*   How to add a new system call to an existing Unix-like kernel (Linux) and how to recompile the kernel
*   What a monolithic kernel is and how it differs from a microkernel
*   How kernel modules work within a monolithic kernel and how to write a new kernel module for Linux

In the following chapter, we're going to talk about the C standards and the most recent version of C, C18\. You will become familiar with the new features introduced as part of it.