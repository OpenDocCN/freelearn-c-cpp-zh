# Setting Up Your C Compiler on Windows

Microsoft Windows is one of the most popular desktop operating systems.

Before beginning, I highly recommend that you install **7-Zip** from [https://www.7-zip.org](https://www.7-zip.org)/. 7-Zip will allow you to extract the various compression archive formats that library source code is distributed in.

Let's continue and get MinGW, OpenSSL, and `libssh` set up on Windows 10.

# Installing MinGW GCC

MinGW is a port of GCC to Windows. It is the compiler we recommend for this book.

You can obtain MinGW from [http://www.mingw.org/](http://www.mingw.org/). Find the download link on that page and download and run the **MinGW Installation Manager** (**mingw-get**).

The MinGW Installation Manager is a GUI tool for installing MinGW. It's shown in the following screenshot:

![](img/84679923-d996-454e-914b-7077bac444c6.png)

Click Install. Then, click Continue. Wait while some files download, and then click Continue once again.

At this point, the tool will give you a list of packages that you can install. You need to mark mingw32-base-bin, msys-base-bin, and mingw32-gcc-g++-bin for installation. This is shown in the following screenshot:

![](img/0c4584f0-3672-41e6-912d-ca2c5e8311da.png)

You will also want to select the mingw32-libz-dev package. It is listed under the MinGW Libraries section. The following screenshot shows this selection:

![](img/f80d717a-c480-4811-b918-af1b220e155a.png)

The `g++` and `libz` packages we've selected are required for building `libssh` later.

When you're ready to proceed, click Installation from the menu and select Apply Changes.

A new dialog will show the changes to be made. The following screenshot shows what this dialog may look like:

![](img/8e98f21f-e4d0-442c-9707-f5d06334bdcc.png)

Click the Apply button to download and install the packages. Once the installation is complete, you can close the MinGW Installation Manager.

To be able to use MinGW from the command line easily, you will need to add MinGW to your `PATH`.

The steps for adding MinGW to your `PATH` are as follows:

1.  Open the System control panel (Windows key + *Pause*/*Break*).
2.  Select Advanced system settings:

![](img/e776b32e-ffa4-4c58-a8f4-878ebfc15151.png)

3.  From the System Properties window, navigate to the Advanced tab and click the Environment Variables... button:

![](img/c57bf130-4e24-406a-90f1-4733d447a8da.png)

4.  From this screen, find the `PATH` variable under System variables. Select it and press **Edit...**.
5.  Click New and type in the MinGW path—`C:\mingw\bin`, as shown in the following screenshot:

![](img/9f990d51-0b4b-4e7a-ad35-55e58858f68e.png)

6.  Click OK to save your changes.

Once MinGW has been added to your `PATH`, you can open a new command window and enter `gcc --version` to ensure that `gcc` has been installed correctly. This is shown in the following screenshot:

![](img/0558634c-758f-49ab-ac91-9ad356da1c56.png)

# Installing Git

You will need to have the `git` version control software installed to download this book's code.

`git` is available from [https://git-scm.com/download](https://git-scm.com/download). A handy GUI-based installer is provided, and you shouldn't have any issues getting it working. When installing, be sure to check the option for adding `git` to your `PATH`. This is shown in the following screenshot:

![](img/f0b7a8df-4d1e-45fa-bc23-fb303a42bf62.png)

After `git` has finished installing, you can test it by opening a new command window and entering `git --version`:

![](img/5ad52e29-8376-4e11-8232-5d6ff30c9744.png)

# Installing OpenSSL

The OpenSSL library can be tricky to get going on Windows.

If you are brave, you can obtain the OpenSSL library source code directly from [https://www.openssl.org/source/](https://www.openssl.org/source/). You will, of course, need to build OpenSSL before it can be used. Building OpenSSL is not easy, but instructions are provided in the `INSTALL` and `NOTES.WIN` files included with the OpenSSL source code.

An easier alternative is to install prebuilt OpenSSL binaries. You can find a list of prebuilt OpenSSL binaries from the OpenSSL wiki at [https://wiki.openssl.org/index.php/Binaries](https://wiki.openssl.org/index.php/Binaries). You will need to locate binaries that match your operating system and compiler. Installing them will be a matter of copying the relevant files to the MinGW `include`, `lib`, and `bin` directories.

The following screenshot shows a binary OpenSSL distribution. The `include` and `lib` folders should be copied over to `c:\mingw\` and merged with the existing folders, while `openssl.exe` and the two DLL files need to be placed in `c:\mingw\bin\`:

![](img/f9238f39-e6e9-4b9c-a17a-1e9b05038b65.png)

You can try building `openssl_version.c` from [Chapter 9](47ba170d-42d9-4e38-b5d8-89503e005708.xhtml), *Loading Secure Web Pages with HTTPS and OpenSSL*, to test that everything is installed correctly. It should look like the following:

![](img/b798e570-c39f-4c2f-a3cd-62b920b3e749.png)

# Installing libssh

You can obtain the latest `libssh` library from [https://www.libssh.org/](https://www.libssh.org/). If you are proficient in installing C libraries, feel free to give it a go. Otherwise, read on for step-by-step instructions.

Before beginning, be sure that you've first installed the OpenSSL libraries successfully. These are required by the `libssh` library.

We will need CMake installed in order to build `libssh`. You can obtain CMake from [https://cmake.org/](https://cmake.org/). They provide a nice GUI installer, and you shouldn't run into any difficulties. Make sure you select the option to add CMake to your `PATH` during installation:

![](img/e80c6ca3-2751-432e-a3b5-afdf77b81058.png)

Once you have the CMake tool and the OpenSSL libraries installed, navigate to the `libssh` website to download the `libssh` source code. At the time of writing, Version 0.8.7 is the latest, and it is available from [https://www.libssh.org/files/0.8/](https://www.libssh.org/files/0.8/). Download and extract the `libssh` source code.

Take a look at the included `INSTALL` file.

Now, open a command window in the `libssh` source code directory. Create a new `build` folder with the following commands:

```cpp
mkdir build cd build
```

Keep this command window open. We'll do the build here in a minute.

Start CMake 3.14.3 (**cmake-gui**) from the start menu or desktop shortcut.

You need to set the source code and build locations using the Browse Source... and Browse Build... buttons. This is shown in the following screenshot:

![](img/c4cf41a7-8d2d-4576-8b81-ab20ddd20bde.png)

Then, click Configure.

On the next screen, select MinGW Makefiles as the generator for this project. Click Finish.

![](img/349233d1-58f1-492d-85fa-76bec0dd6ae1.png)

It may take a moment to process.

From the configuration options, make the following changes:

1.  Uncheck WITH_NACL
2.  Uncheck WITH_GSSAPI
3.  Change `CMAKE_INSTALL_PREFIX` to `c:\mingw`

Then, click Configure again. It will take a moment. If everything worked, click Generate.

You should now be able to build `libssh`.

Go back to your command window in the build directory. Use the following command to complete the build:

```cpp
mingw32-make
```

After the build completes, use the following command to copy the files over to your MinGW installation:

```cpp
mingw32-make install
```

You can try building `ssh_version.c` from [Chapter 11](c9d0a1dc-878b-4961-825e-65688fac08ae.xhtml), *Establishing SSH Connections with libssh*, to test that everything is installed correctly. It should look like the following:

![](img/cb20c15a-dc94-4992-9eb1-11dffc687062.png)

# Alternatives

In this book, we recommend free software whenever possible. This is important for user freedom, and this is one reason we recommend GCC throughout the book.

In addition, to MinGW GCC, the Clang C compiler is also open source and excellent quality. The code in this book was also tested to run successfully using Clang on Windows.

Command-line tools such as GCC and Clang are often easier to integrate into the complicated workflows required for larger projects. These open source tools also provide better standards compliance than Microsoft's compilers.

That said, the code in this book also works with Microsoft's compilers. The code was tested for both Microsoft Visual Studio 2015 and Microsoft Visual Studio 2017.