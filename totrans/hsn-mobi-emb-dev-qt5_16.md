# Cross Compiling and Remote Debugging

Since there is a good chance of using a Linux system on an embedded device, we will go through the steps needed to set up a cross compiler on Linux. Mobile phone platforms have their own ways of development, which will also be discussed. You will learn to compile cross-platforms for a different device and debug remotely via a network or USB connection. We will go through various mobile platforms.

We will cover the following topics in this section:

*   Cross compiling
*   Connecting to a remote device
*   Debugging remotely

# Cross compiling

Cross compiling is a method for building applications and libraries on a host machine for a different architecture than what is running on the host machine. When you build for a phone using an Android or iOS SDKs, you are cross compiling.

One easy way to do this is to use Qt for Device Creation, or Qt's Boot to Qt commercial tools. It is available for evaluation or purchase.

You do not have to build any of the tools and device system images yourself. I used Boot to Qt for my Raspberry Pi. This made set up a lot faster and easier. There are also more traditional ways of building for different devices, and they would be about the same, except for the target machine.

If you are on Windows, cross compiling can be a bit more tricky. You either install MinGW or Cygwin to build your own cross compiler, install Windows Subsystem for Linux, or install a prebuilt cross `toolchain`, for example, from Sysprogs.

# Traditional cross tools

There are many ways to get a cross compiler. Device makers can release a cross `toolchain` with their software stack. Of course, if you are building your own hardware, or just want to create your own cross `toolchain`, there are other options. You can download a prebuilt cross `toolchain` for your device's architecture, or you can build it yourself. If you do end up compiling the `toolchain`, you will need a fast and robust machine with a lot of disk space, as it will take quite a long time to finish and use a lot of filesystem—easily 50 GB if you build the entire system.

# DIY toolchain

There are also projects for which you can or must (if there is no supplied `toolchain`) build your own `toolchain`. The following are some of the more well known cross tools:

*   **Buildroot**: [https://buildroot.org/](https://buildroot.org/)
*   **Crosstool-NG**: [http://crosstool-ng.github.io/](http://crosstool-ng.github.io/)
*   **OpenEmbedded**: [http://www.openembedded.org](http://www.openembedded.org)
*   **Yocto**: [https://www.yoctoproject.org/](https://www.yoctoproject.org/)
*   **Ångström**: [http://wp.angstrom-distribution.org/](http://wp.angstrom-distribution.org/)

BitBake is used by OpenEmbedded, Yocto, and Ångström (as well as Boot to Qt), so it might be easiest to start out with one of those. You could say it is *Buildroot 2.0*, as it is the second incarnation of the original Buildroot. It is a completely different construction though. Buildroot is simpler and has no concept of packages, and thus, upgrading the system can be more difficult.

I will describe building a `toolchain` with BitBake in [Chapter 15](590553c7-965b-4002-bbfe-fd61e30ce5a8.xhtml), *Building a Linux System.* Essentially it is very similar to building a system image; in fact, it has to build the `toolchain` before it can build the system image.

# Buildroot

Buildroot is a tool that helps build complete systems. It can build the cross `toolchain` or use an external one. It traditionally uses an ncurses interface for configuration, much like the Linux kernel. It also has a new ncurses configurator, but also a Qt-based one. Let's use that!

In the directory where you unpacked Buildroot, run the following command:

```cpp
make xconfig
```

Bah! It uses Qt 4\. If you don't want to install Qt 4, you can always use `make menuconfig` or `make nconfig`.

Here is what the Qt interface looks like:

![](img/7a3b33a4-4924-4da7-abb3-96a917070e28.png)

By default, Buildroot will create a system based on BusyBox, instead of glibc.

Once you have configured your system, save the configuration and close the configurator. Then run `make`, sit back, and let it build. It will place files into a directory called `output/`, under which your system image is in a directory named image.

# Crosstool-NG

Crosstool-NG is meant for building toolchains, not system images. You can use the `toolchain` built with crosstools to build a system, although you would have to do it manually.

Crosstool-NG is similar to Buildroot, in that it uses ncurses to configure the `toolchain` to be built. Once you unpack it, you need to run the following `bootstrap` script:

```cpp
./bootstrap
```

To install it, you would call configure with the following `--prefix` argument:

```cpp
./configure --prefix=/path/to/output
```

You can also run it locally as follows:

```cpp
./configure --enable-local
```

It will tell you any packages that are missing to install. On my Ubuntu Linux, I had to install `flex`, `lzip`, `help2man`, `libtool-bin`, and `ncurses-dev`.

Then run `make` and `make install` it you configured with a prefix.

You will need to add `/path/to/output/bin` into your `$PATH`

`export PATH=$PATH:/path/to/output/bin`.

Now you can run the following configuration:

```cpp
./ct-ng menuconfig
```

![](img/09fb9dce-d527-44f9-a869-f34254b3f006.png)

Then run `make`, which will build the cross `toolchain`.

# Prebuilt tools

There are companies that make the following prebuilt cross tool chains for various devices and architectures:

*   **Code Sourcery**: [http://www.codesourcery.com/](http://www.codesourcery.com/)
*   **Bootlin**: [https://toolchains.bootlin.com/toolchains.html](https://toolchains.bootlin.com/toolchains.html)
*   **Linaro, Debian, Fedora**: download from package manager
*   **Boot to Qt**: [https://doc.qt.io/QtForDeviceCreation/qtb2-index.html](https://doc.qt.io/QtForDeviceCreation/qtb2-index.html)
*   **Sysprogs**: [http://gnutoolchains.com/](http://gnutoolchains.com/)

These are a few of the better ones. I have experienced the majority of these and used them at one time or another. Each comes with it's own instructions on how to install and use. Linaro, Debian, and Fedora all make ARM cross compilers. This is a book on Qt development, so I will describe Qt Company's offering—Boot to Qt.

# Boot to Qt

Qt Company's Boot to Qt product comes complete with development tools and a prebuilt operating system image that you write to a micro SD card or flash to run on the device. They support the following other devices besides the Raspberry Pi:

*   Boundary Devices i.MX6 Boards
*   Intel NUC
*   NVIDIA Jetson TX2
*   NXP i.MX 8QMax LPDDR4
*   Raspberry Pi 3
*   Toradex Apalis iMX6 and iMX8
*   Toradex Colibri iMX6, iMX6ULL, and iMX7
*   WaRP7

I picked the RPI, as I already have a model 3 lying around with a touch screen.

When you run the system image, you boot into a Qt app that serves as a launcher for example apps. It also sets up Qt Creator to be able to run cross compiled apps on the device. You can run it on the device by hitting the Run button in Qt Creator.

Boot to Qt is a really fast and easy way to get a prototype up and running on a touch screen with a relatively small system. The Qt Company is currently working on getting Qt working well on smaller devices, such as microcontrollers.

You can run the Boot to Qt `toolchain` directly; you simply have to source the environment file. In the case of Raspberry Pi and Boot to Qt, it's called `environment-setup-cortexa7hf-neon-vfpv4-poky-linux-gnueabi`. You can also call the qmake of `toolchain` directly and run it on your profile `/path/to/x86_64-pokysdk-linux/usr/bin/qmake myApp.pro`.

A third option here is to just use Qt Creator and pick the Raspberry Pi as the target.

If you use Windows, there are a few options you can use to get a cross compiler `toolchain`.

# Cross toolchains on Windows

There are a few ways you can cross compile on Windows, and we can briefly go through them. They are as follows, but there are undoubtedly others that are not covered here:

*   Sysrogs provides prebuilt cross `toolchain` for use on Windows.
*   Windows Subsystem for Linux.

# Sysprogs

Sysprogs is a company that makes cross tool chains for targeting Linux devices that runs on Windows. Their `toolchain` can be downloaded from [http://gnutoolchains.com/](http://gnutoolchains.com/)

1.  Once installed, start a Qt 5.12.1 (MinGW 7.3.0 64-bit) console terminal
2.  You need to add the `toolchain` to your path as follows:

```cpp
set PATH=C:\SysGCC\raspberry\bin;%PATH%
```

3.  Add the `PATH` to Qt's `mingw` as follows:

```cpp
set PATH=C:\Qt\Tools\mingw730_64\bin;%PATH%
```

You will also have to build OpenGL and other requirements for Qt.

Configure Qt to cross compile as follows:

```cpp
..\qtbase/configure -opengl es2 -device linux-rasp-pi-g++ -device-option CROSS_COMPILE=C:\SysGCC\raspberry\bin\arm-linux-gnueabihf- -sysroot C:\SysGCC\raspberry\arm-linux-gnueabihf\sysroot -prefix /usr/local/qt5pi -opensource -confirm-license -nomake examples -make libs -v -platform win32-g++
```

# Windows Subsystem for Linux

You can install Windows Subsystem for Linux to install a cross compiler, which can be downloaded from [https://docs.microsoft.com/en-us/windows/wsl/install-win10.](https://docs.microsoft.com/en-us/windows/wsl/install-win10)

You can then pick the required Linux distribution—Ubuntu, OpenSUSE, or Debian. Once this is installed, you can use the built-in package manager to install the `toolchain` for Linux.

# Mobile platform-specific tools

Both iOS and Android have prebuilt cross tools and SDKs that are available to download. You will need either of these if you are going to use Qt on the mobile platforms, as Qt Creator depends on the native platform build tools.

# iOS

Xcode is the IDE beast you want to download, and it only runs on macOS X. You can get it from the App store on your desktop if you do not already have it. You will need to register as an iOS developer. From there, you can select the iOS build tools to download and set up. It's fairly automatic once you start the download.

You can also use these tools from the command line, but you need to install the command line tools from within Xcode. For Sierra, you can simply type the `gcc` command in the terminal. In that case, the system will open a dialog asking you if you want to install the command line tools. Alternatively, you can install it by running `xcode-select --install`.

I don't know of any tools for embedded systems that you can use with Xcode, unless you count the iWatch or iTV SDKs. Both of these you can download through Xcode.

You could use Darwin, of course, since it is open source and based on **Berkeley Software Distribution** (**BSD**). You could also use BSD. This is far from being able to run an Apple operating system on arbitrary embedded hardware, so your choices are limited.

# Android

Android has Android Studio for its IDE development package and is available for macOS X, Windows, and Linux systems.

Like Xcode, Android Studio has command line tools as well, which you install through the SDK manager or the `sdkmanager` command.

`~/Android/Sdk/tools/bin/sdkmanager --list` will list all packages available. If you wanted to download the `adb` and `fastboot` commands, you could do the following:

`~/Android/Sdk/tools/bin/sdkmanager install "platform-tools"`

Android has catchy code names for their different versions, which is completely different from their API level. You should stick with the API level when installing Android SDKs. I have an Android phone that runs Android version 8.0.0, which has the code name Oreo. I would need to install an SDK for the API level 26 or 27\. If I wanted to install the SDK, I might do the following:

```cpp
~/Android/Sdk/tools/bin/sdkmanager install "platforms;android-26"
```

For working in Qt, you also need to install the Android NDK. I have NDK version 10.4.0, or r10e, and Qt Creator works with that just fine. I had issues with running a later version of the NDK. Your mileage may vary, as they say.

# QNX

QNX is a commercial UNIX-like operating system, which is currently owned by Blackberry. It is not open source, but I thought I would mention it here, as Qt runs on QNX, and is being used commercially in the market.

# Connecting to a remote device

This is a book about Qt development and I will stick to Qt Creator. The method is nearly the same to connect to any device, with a few minor differences. You can also connect via **Secure Shell** (**SSH**) and friends with a terminal. I often use both methods, as each has its own advantages and disadvantages.

# Qt Creator

I remember when what is now called Qt Creator was first released for internal testing at Nokia. At that time, it was called Workbench. It was basically a good text editor. Since that time, it has gained heaps of awesome features and it is my go to IDE for Qt-based projects.

Qt Creator is a multi-platform IDE, it runs on macOS X, Windows, and Linux. It can connect to Android, iOS, or generic Linux devices. You can even get SDKs for devices such as UBports (Open Source Ubuntu Phone) or Jolla Phones.

To configure your device, in Qt Creator navigate to Tools | Options... | Devices | Devices.

# Generic Linux

A generic Linux device could be a custom embedded Linux device or even a Raspberry Pi. It should be running an SSH server. Since I used an RPI, I will demonstrate with that.

The following is the devices tab showing connection details for a Raspberry Pi:

![](img/acc696a5-6f66-4d2e-baa6-fff925e52696.png)

As you can see here, the most important item is probably Host name. Make sure the IP address in the Host name configuration matches the actual IP of the device. Other devices may have a direct USB connection instead of using the regular network.

# Android

You will need Android SDK and NDK installed.

Android is a device that uses a direct USB connection, so it will be easier copying the application binary files when it runs the application:

![](img/0b655d7d-223b-4a0c-86e3-243ac6d4bd8d.png)

Qt Creator more or less configures this connection automatically.

# iOS

Make sure your device is seen by Xcode first, then Qt Creator will automatically pick it up and use it.

It would look similar to this image:

![](img/89ed76f8-f521-4bce-8035-e363aedea0f5.png)

Notice the little green LED-like icon? Ya, all good to go!

# Bare metal

If your device does not run an SSH server, you can connect with it using `gdb/gdbserver` or a hardware server. You will need to enable the plugin first. In Qt Creator, navigate to **Help** | **About Plugins** | **Device Support**, and then select BareMetal. The bare metal connection uses OpenOCD that you can get from `http://openocd.org`. OpenOCD is not some new anxiety disorder, but an on-chip debugger that runs through a JTAG interface. Qt Creator also has support for the ST-LINK debugger. Both use JTAG connectors. There are USB JTAG connectors as well as the traditional JTAG interface, which do not require any device drivers to get connected.

Writing this section brought back memories of when Trolltech got the Trolltech Greenphone up and running, as well as working on some other devices, like the OpenMoko phone. Good times!

Now that we have a connected device, we can start debugging.

# Debugging remotely

Developing software is hard. All software has bugs. Some bugs are more painful than others. The worst kind are probably when you have a random crash that requires a specific sequence of events to trigger that reside on a read-only filesystem remote device that was built in release mode. Been there. Done that. Even got a t-shirt. (I have many Trolltech and Nokia t-shirts left over from days gone by.)

Remote debugging traditionally involves running the gdbserver command on the device. On very small machines where there isn't enough RAM to run gdb directly, running gdbserver on the remote device is probably the only way to use gdb. Let's put on some groove salad and get cracking!

# gdbserver

You may want to experience remote debugging without a UI, or something weird like that. This will get you started. The `gdbserver` command needs to be running on the remote device, and there needs to be either a serial or TCP connection.

On the *remote* device, run the following command:

```cpp
gdbserver host:1234 <target> <app args>
```

Using the `host` argument will start `gdbserver` running on port `1234`. You could also attach the debugger on a running application by running the following command:

```cpp
gdbserver host:1234 --attach <pid>
```

`pid` is the process ID of the already running application you are trying to debug, which you could get through running the command `ps`, or top, or similar.

On the *host* device, run the following command:

```cpp
gdb target remote <host ip/name>:1234
```

You will then `issue` commands on the host device through the console that is running `gdb`.

If you run into a crash, after it happens you can type `bt` to get a backtrace listing. If you have a crash memory dump, or core dump as it's called, on the remote, `gdbserver` does not support debugging core memory dumps remotely. You will have to run `gdb` itself on the remote in order to do this.

Using `gdb` via the command line might be fun to some, but I prefer a UI, since it is easier to remember things to be done. Having a GUI that can do remote debugging can help if you are not very familiar with running `gdb` commands, as this can be a daunting task. Qt Creator can do remote debugging, so let's move on to debugging with Qt Creator.

# Qt Creator

Qt Creator uses `gdbserver` on the device, so it is essentially just a UI interface. You will need to have Python scripting support for `gdbserver` on the device; otherwise, you will see a message Selected build of GDB does not support Python scripting, and it will not work.

For the most part, debugging with Qt Creator works out-of-the-box for Android, iOS, and any supported Boot to Qt device.

Load any project in Qt Creator and it can handle C++ debugging, as well as debugging into Qt Quick projects. Make sure the correct settings are configured in the Run Settings page down where it says Debugger settings within Qt Creator to enable `qml` debugging and/or C++ debugging if you need it.

Add the following to your project and rebuild:

```cpp
CONFIG+=debug qml_debug
```

Add this to the application startup arguments `-qmljsdebugger=port:<port>, host:<ip>`.

To interrupt the execution of the app, click on the icon whose tooltip says 'Interrupt GDB for "yourapp"'. You can then inspect the value of variables and step though the code.

Set a breakpoint somewhere—right-click on the line in question and select Set Breakpoint on line .

Press *F5* to start the application build (if needed). Once successfully built, it will be transferred and executed on the device and the remote debugging service is started. It will, of course, stop execution on your breakpoint, if you have one set. To continue normal execution, press *F5* until you hit that painful crash, and then you can inspect that wonderful backtrace! From here, you can hopefully gather enough clues to fix it.

Other key commands supported by Qt Creator by default are as follows:

*   *F5*: start / continue execution
*   *F9*: toggle breakpoint
*   *F10*: step over
*   *Ctrl* + *F10*: run to current line
*   *F11*: step into
*   *Shift* + *F11*: step out

Let's try it out. Load the source code for this chapter.

To toggle a breakpoint on the current line in the Qt Creator editor, press *F9* on Linux and Windows, or *F8* on macOS as follows:

![](img/84ec6236-d9de-44d0-925a-7cd43c4d946a.png)

Now start the debugger by pressing *F5* to run the app in the debugger. It will stop execution on our line as follows:

![](img/3cda3273-f7a4-434a-ade2-c1f34afe8813.png)

See that little yellow arrow? It informs us that the execution has stopped on this line, before the statement has been executed.

You will be able to see the following values for the variables:

![](img/37d3bd2f-15bc-497d-912b-d62bcbfe5e15.png)

As you can see, the breakpoint stopped execution before `QString` `b` has been initialized, so the value is `""`. If you push *F10* or step over, the `QString` `b` gets initialized and you can see the new value as follows:

![](img/9521bc99-abd2-4ab8-8367-8bac3e2ec6d1.png)

You will notice from the following screenshot that the execution line gets moved to the next statement as well:

![](img/681dd2a8-e4c7-4156-8b00-f1b79aaa78f9.png)

You can also edit the breakpoint by right-clicking on the breakpoint in the editor and selecting Edit Breakpoint. Let's set a breakpoint on Line 20 in the for loop as follows:

![](img/a0aba595-2164-4246-897d-b69b5d1ab132.png)

Right-click and select Edit Breakpoint to open the Edit Breakpoint Properties dialog as follows:

![](img/ce1db2e2-1e79-4137-aaf8-f78275a29dfd.png)

Edit the Condition field and add `i == 15` as follows and click OK:

![](img/edb54855-33c5-4510-a91c-1a0a26fe9d6c.png)

Run the app in the debugger by clicking *F5*. Click on the Strings button. When it hits the breakpoint, you can see it stopped when i contains the value 15:

![](img/746ee8de-bda8-4c15-a82e-52ab03c6e52d.png)

You could then step into, or step over.

Let's now look at a crash bug, which is a divide by zero crash when you push the crash button.

Set a breakpoint at line *31*. Run the debugger, and it will stop just before the crash. Now step over. You should see a dialog popup as follows:

![](img/64eb0602-9f5a-4c92-a5d0-32e74c5e48d7.png)

Oh my. Now that is ugly.

In the stack view shown in the following screenshot, you can see where the program has crashed:

![](img/d12dc805-5795-4da9-83d0-15f05fd99896.png)

Yep, it is right where I put it! Bad things happen when you divide by zero in C++.

# Summary

Debugging is a powerful process and often required to fix bugs. You can either run debuggers such as `gdb` from the command line, or you should be able to connect the `gdb` to a debugger server running on a remote device. Running a GUI-based debugger is much more fun. You should be able to debug your Qt app running on your mobile or embedded device from Qt Creator over a remote connection.

The next step is to deploy your application. We will explore various ways to deploy your application on a few different mobile and embedded platforms.