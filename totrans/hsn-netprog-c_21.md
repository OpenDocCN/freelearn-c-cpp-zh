# Setting Up Your C Compiler on Linux

**Linux** is an excellent choice for C programming. It has arguably the easiest setup and the best support for C programming out of the three operating systems covered in this book.

Using Linux also allows you to take the ethical high road and feel good about supporting free software.

One issue with describing the setup for Linux is that there are many Linux distributions with different software. In this appendix, we will provide the commands needed to set up on systems using the `apt` package manager, such as **Debian Linux** and **Ubuntu Linux**. If you are using a different Linux distribution, you will need to find the commands relevant to your system. Refer to your distribution's documentation for help.

Before diving right in, take a moment to make sure your package list is up to date. This is done with the following command:

```cpp
sudo apt-get update
```

With `apt` ready to go, setup is easy. Let's get started.

# Installing GCC

The first step is to get the C compiler, `gcc`, installed.

Assuming your system uses `apt` as its package manager, try the following commands to install `gcc` and prepare your system for C programming:

```cpp
sudo apt-get install build-essential
```

Once the `install` command completes, you should be able to run the following command to find the version of `gcc` that was installed:

```cpp
gcc --version
```

# Installing Git

You will need to have the Git version control software installed to download this book's code.

Assuming your system uses the `apt` package manager, you can install Git with the following command:

```cpp
sudo apt-get install git
```

Check whether Git has installed successfully by means of the following command:

```cpp
git --version
```

# Installing OpenSSL

**OpenSSL** can be tricky. You can try your distribution's package manager with the following commands:

```cpp
sudo apt-get install openssl libssl-dev
```

The problem is that your distribution may have an old version of OpenSSL. If that is the case, you should obtain the OpenSSL libraries directly from [https://www.openssl.org/source/](https://www.openssl.org/source/). You will, of course, need to build OpenSSL before it can be used. Building OpenSSL is not easy, but instructions are provided in the `INSTALL` file included with the OpenSSL source code. Note that its build system requires that you have **Perl** installed.

# Installing libssh

You can try installing `libssh` from your package manager with the following command:

```cpp
sudo apt-get install libssh-dev
```

The problem is that the code in this book is not compatible with older versions of `libssh`. Therefore, I recommend you build `libssh` yourself.

You can obtain the latest `libssh` library from [https://www.libssh.org/](https://www.libssh.org/). If you are proficient in installing C libraries, feel free to give it a go. Otherwise, read on for the step-by-step instructions.

Before beginning, be sure that you've first installed the OpenSSL libraries successfully. These are required by the `libssh` library.

We will also need CMake installed in order to build `libssh`. You can obtain CMake from [https://cmake.org/](https://cmake.org/). You can also get it from your distro's packaging tool with the following command:

```cpp
sudo apt-get install cmake
```

Finally, the `zlib` library is also required by `libssh`. You can install the `zlib` library using this command:

```cpp
sudo apt-get install zlib1g-dev
```

Once you have CMake, the `zlib` library, and the OpenSSL library installed, locate the version of `libssh` you would like from [https://www.libssh.org/](https://www.libssh.org/). Version 0.8.7 is the latest at the time of writing. You can download and extract the `libssh` source code with the following commands:

```cpp
 wget https://www.libssh.org/files/0.8/libssh-0.8.7.tar.xz
 tar xvf libssh-0.8.7.tar.xz
 cd libssh-0.8.7
```

I recommend that you take a look at the installation instructions included with `libssh`. You can use `less` to view them. Press the *Q* key to quit `less`:

```cpp
less INSTALL
```

Once you've familiarized yourself with the build instructions, you can try building `libssh` with these commands:

```cpp
mkdir build
cd build
cmake ..
make
```

The final step is to install the library with the following command:

```cpp
sudo make install
```