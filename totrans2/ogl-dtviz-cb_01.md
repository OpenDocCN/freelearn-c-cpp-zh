# Chapter 1. Getting Started with OpenGL

In this chapter, we will cover the following topics:

*   Setting up a Windows-based development platform
*   Setting up a Mac-based development platform
*   Setting up a Linux-based development platform
*   Installing the GLFW library in Windows
*   Installing the GLFW library in Mac OS X and Linux
*   Creating your first OpenGL application with GLFW
*   Compiling and running your first OpenGL application in Windows
*   Compiling and running your first OpenGL application in Mac OS X or Linux

# Introduction

OpenGL is an ideal multiplatform, cross-language, and hardware-accelerated graphics rendering interface that is well suited to visualize large 2D and 3D datasets in many fields. In fact, OpenGL has become the industry standard to create stunning graphics, most notably in gaming applications and numerous professional tools for 3D modeling. As we collect more and more data in fields ranging from biomedical imaging to wearable computing (especially with the evolution of Big Data), a high-performance platform for data visualization is becoming an essential component of many future applications. Indeed, the visualization of massive datasets is becoming an increasingly challenging problem for developers, scientists, and engineers in many fields. Therefore, OpenGL can provide a unified solution for the creation of impressive, stunning visuals in many real-time applications.

The APIs of OpenGL encapsulate the complexity of hardware interactions while allowing users to have low-level control over the process. From a sophisticated multiserver setup to a mobile device, OpenGL libraries provide developers with an easy-to-use interface for high-performance graphics rendering. The increasing availability and capability of graphics hardware and mass storage devices, coupled with their decreasing cost, further motivate the development of interactive OpenGL-based data visualization tools.

Modern computers come with dedicated **Graphics Processing Units** (**GPUs**), highly customized pieces of hardware designed to accelerate graphics rendering. GPUs can also be used to accelerate general-purpose, highly parallelizable computational tasks. By leveraging hardware and OpenGL, we can produce highly interactive and aesthetically pleasing results.

This chapter introduces the essential tools to develop OpenGL-based data visualization applications and provides a step-by-step tutorial on how to set up the environment for our first demo application. In addition, this chapter outlines the steps to set up a popular tool called CMake, which is a cross-platform software that automates the process of generating standard build files (for example, makefiles in Linux that define the compilation parameters and commands) with simple configuration files. The CMake tool will be used to compile additional libraries in the future, including the GLFW (OpenGL FrameWork) library introduced later in this chapter. Briefly, the GLFW library is an open source, multiplatform library that allows users to create and manage windows with OpenGL contexts as well as handle inputs from peripheral devices such as the mouse and keyboard. By default, OpenGL itself does not support other peripherals; thus, the GLFW library is used to fill in the gap. We hope that this detailed tutorial will be especially useful for beginners who are interested in exploring OpenGL for data visualization but have little or no prior experience. However, we will assume that you are familiar with the C/C++ programming language.

# Setting up a Windows-based development platform

There are various development tools available to create applications in the Windows environment. In this book, we will focus on creating OpenGL applications using Visual C++ from Microsoft Visual Studio 2013, given its extensive documentation and support.

## Installing Visual Studio 2013

In this section, we outline the steps to install Visual Studio 2013.

## Getting ready

We assume that you have already installed Windows 7.0 or higher. For optimal performance, we recommend that you get a dedicated graphics card, such as NVIDIA GeForce graphics cards, and have at least 10 GB of free disk space as well as 4 GB of RAM on your computer. Download and install the latest driver for your graphics card.

## How to do it...

To install Microsoft Visual Studio 2013 for free, download the Express 2013 version for Windows Desktop from Microsoft's official website (refer to [https://www.visualstudio.com/en-us/downloads/](https://www.visualstudio.com/en-us/downloads/)). Once you have downloaded the installer executable, we can start the process. By default, we will assume that programs are installed in the following path:

![How to do it...](img/9727OS_01_01.jpg)

To verify the installation, click on the **Launch** button at the end of the installation, and it will execute the VS Express 2013 for Desktop application for the first time.

## Installing CMake in Windows

In this section, we outline the steps to install CMake, which is a popular tool that automates the process of creating standard build files for Visual Studio (among other tools).

## Getting ready

To obtain the CMake tool (CMake 3.2.1), you can download the executable (`cmake-3.2.1-win32-x86.exe`) from [http://www.cmake.org/download/](http://www.cmake.org/download/).

## How to do it…

The installation wizard will guide you through the process (select **Add CMake to the system PATH for all users** when prompted for installation options). To verify the installation, run CMake(`cmake-gui`).

![How to do it…](img/9727OS_01_02.jpg)

At this point, you should have both Visual Studio 2013 and CMake successfully installed on your machine and be ready to compile/install the GLFW library to create your first OpenGL application.

# Setting up a Mac-based development platform

One important advantage of using OpenGL is the possibility of cross-compiling the same source code on different platforms. If you are planning to develop your application on a Mac platform, you can easily set up your machine for development using the upcoming steps. We assume that you have either Mac OS X 10.9 or higher installed. OpenGL updates are integrated into the system updates for Mac OS X through the graphics driver.

## Installing Xcode and command-line tools

The Xcode development software from Apple provides developers with a comprehensive set of tools, which include an IDE, OpenGL headers, compilers, and debugging tools, to create native Mac applications. To simplify the process, we will compile our code using the command-line interface that shares most of the common features in Linux.

## Getting ready

If you are using Mac OS X 10.9 or higher, you can download Xcode through the App Store shipped with Mac OS. Full installation support and instructions are available on the Apple Developer website ([https://developer.apple.com/xcode/](https://developer.apple.com/xcode/)).

## How to do it...

We can install the command-line tools in Xcode through the following steps:

1.  Search for the keyword `Terminal` in **Spotlight** and run **Terminal**.![How to do it...](img/9727OS_01_03.jpg)
2.  Execute the following command in the terminal:

    [PRE0]

    Note that if you have previously installed the command-line tools, an error stating "command-line are already installed" will appear. In this case, simply skip to step 4 to verify the installation.

3.  Click on the **Install** button to directly install the command-line tools. This will install basic compiling tools such as **gcc** and **make** for application development purposes (note that CMake needs to be installed separately).
4.  Finally, enter `gcc --version` to verify the installation.![How to do it...](img/9727OS_01_04.jpg)

## See also

If you encounter the **command not found** error or other similar issues, make sure that the command-line tools are installed successfully. Apple provides an extensive set of documentation, and more information on installing Xcode can be found at [https://developer.apple.com/xcode](https://developer.apple.com/xcode).

## Installing MacPorts and CMake

In this section, we outline the steps to install MacPorts, which greatly simplifies the subsequent setup steps, and CMake for Mac.

## Getting ready

Similar to the Windows installation, you can download the binary distribution of **CMake** from [http://www.cmake.org/cmake/resources/software.html](http://www.cmake.org/cmake/resources/software.html) and manually configure the command-line options. However, to simplify the installation and automate the configuration process, we highly recommend that you use MacPorts.

## How to do it...

To install MacPorts, follow these steps:

1.  Download the MacPorts package installer for the corresponding version of Mac OS X ([https://guide.macports.org/#installing.macports](https://guide.macports.org/#installing.macports)):

    *   Mac OS X 10.10 Yosemite: [https://distfiles.macports.org/MacPorts/MacPorts-2.3.3-10.10-Yosemite.pkg](https://distfiles.macports.org/MacPorts/MacPorts-2.3.3-10.10-Yosemite.pkg)
    *   Mac OS X 10.9 Mavericks: [https://distfiles.macports.org/MacPorts/MacPorts-2.3.3-10.9-Mavericks.pkg](https://distfiles.macports.org/MacPorts/MacPorts-2.3.3-10.9-Mavericks.pkg)

2.  Double-click on the package installer and follow the onscreen instructions.![How to do it...](img/9727OS_01_05.jpg)
3.  Verify the installation in the terminal by typing in `port version`, which returns the version of MacPorts currently installed (`Version: 2.3.3` in the preceding package).

To install **CMake** on Mac, follow these steps:

1.  Open the **Terminal** application.
2.  Execute the following command:

    [PRE1]

To verify the installation, enter `cmake –version` to show the current version installed and enter `cmake-gui` to explore the GUI.

![How to do it...](img/9727OS_01_06.jpg)

At this point, your Mac is configured for OpenGL development and is ready to compile your first OpenGL application. For those who have been more accustomed to GUIs, using the command-line interface in Mac can initially be an overwhelming experience. However, in the long run, it is a rewarding learning experience due to its overall simplicity. Command-line tools and interfaces are often more time-invariant compared to constantly evolving GUIs. At the end of the day, you can just copy and paste the same command lines, thereby saving precious time needed to consult new documentation every time a GUI changes.

# Setting up a Linux-based development platform

To prepare your development environment on the Linux platform, we can utilize the powerful Debian Package Management system. The `apt-get` or `aptitude` program automatically retrieves the precompiled packages from the server and also resolves and installs all dependent packages that are required. If you are using non-Debian based platform, such as Fedora, you can find the equivalents by searching for the keywords of each packages listed in this recipe.

## Getting ready

We assume that you have successfully installed all updates and latest graphics drivers associated with your graphics hardware. Ubuntu 12.04 or higher has support for third-party proprietary NVIDIA and AMD graphics drivers, and more information can be found at [https://help.ubuntu.com/community/BinaryDriverHowto](https://help.ubuntu.com/community/BinaryDriverHowto).

## How to do it…

Use the following steps to install all development tools and the associated dependencies:

1.  Open a terminal.
2.  Enter the update command:

    [PRE2]

3.  Enter the install command and enter `y` for all prompts:

    [PRE3]

4.  Verify the results:

    [PRE4]

    If successful, this command should return the current version of `gcc` installed.

## How it works…

In summary, the `apt-get update` command automatically updates the local database in the Debian Package Management system. This ensures that the latest packages are retrieved and installed in the process. The `apt-get` system also provides other package management features, such as package removal (uninstall), dependency retrieval, as well as package upgrades. These advanced functions are outside the scope of this book, but more information can be found at [https://wiki.debian.org/apt-get](https://wiki.debian.org/apt-get).

The preceding commands install a number of packages to your machine. Here, we will briefly explain the purpose of each package.

The `build-essential` package, as the name itself suggests, encapsulates the essential packages, namely gcc and g++, that are required to compile C and C++ source code in Linux. Additionally, it will download header files and resolve all dependencies in the process.

The `cmake-gui` package is the CMake program described earlier in the chapter. Instead of downloading CMake directly from the website and compiling from the source, it retrieves the latest supported version that had been compiled, tested, and released by the Ubuntu community. One advantage of using the Debian Package Management system is the stability and ease of updating in the future. However, for users who are looking for the cutting-edge version, apt-get based systems would be a few versions behind.

The `xorg-dev` and `libglu1-mesa-dev` packages are the development files required to compile the GLFW library. These packages include header files and libraries required by other programs. If you choose to use the precompiled binary version of GLFW, you may be able to skip some of the packages. However, we highly recommend that you follow the steps for the purpose of this tutorial.

## See also

For more information, most of the steps described are documented and explained in depth in this online documentation: [https://help.ubuntu.com/community/UsingTheTerminal](https://help.ubuntu.com/community/UsingTheTerminal).

# Installing the GLFW library in Windows

There are two ways to install the GLFW library in Windows, both of which will be discussed in this section. The first approach involves compiling the GLFW source code directly with CMake for full control. However, to simplify the process, we suggest that you download the precompiled binary distribution.

## Getting ready

We assume that you have successfully installed both Visual Studio 2013 and CMake, as described in the earlier section. For completeness, we will demonstrate how to install GLFW using CMake.

## How to do it...

To use the precompiled binary package for GLFW, follow these steps:

1.  Create the `C:/Program Files (x86)/glfw-3.0.4` directory. Grant the necessary permissions when prompted.
2.  Download the `glfw-3.0.4.bin.WIN32.zip` package from [http://sourceforge.net/projects/glfw/files/glfw/3.0.4/glfw-3.0.4.bin.WIN32.zip](http://sourceforge.net/projects/glfw/files/glfw/3.0.4/glfw-3.0.4.bin.WIN32.zip) and unzip the package.
3.  Copy all the extracted content inside the `glfw-3.0.4.bin.WIN32` folder (for example, include `lib-msvc2012`) into the `C:/Program Files (x86)/glfw-3.0.4` directory. Grant permissions when prompted.
4.  Rename the `lib-msvc2012` folder to `lib`inside the `C:/Program Files (x86)/glfw-3.0.4` directory. Grant permissions when prompted.

Alternatively, to compile the source files directly, follow these procedures:

1.  Download the source package from [http://sourceforge.net/projects/glfw/files/glfw/3.0.4/glfw-3.0.4.zip](http://sourceforge.net/projects/glfw/files/glfw/3.0.4/glfw-3.0.4.zip) and unzip the package on the desktop. Create a new folder called `build` inside the extracted `glfw-3.0.4` folder to store the binaries.and open `cmake-gui`.
2.  Select `glfw-3.0.4` (from the desktop) as the source directory and `glfw-3.0.4/build` as the build directory. The screenshot is shown as follows:![How to do it...](img/9727OS_01_07.jpg)
3.  Click on **Generate** and select **Visual Studio 12 2013** in the prompt.![How to do it...](img/9727OS_01_08.jpg)
4.  Click on **Generate** again.![How to do it...](img/9727OS_01_09.jpg)
5.  Open the `build` directory and double-click on **GLFW.sln** to open Visual Studio.
6.  In Visual Studio, click Build Solution (press *F7*).
7.  Copy **build/src/Debug/glfw3.lib** to **C:/Program Files (x86)/glfw-3.0.4/lib**.
8.  Copy the `include` directory (inside `glfw-3.0.4/include`) to **C:/Program Files (x86)/glfw-3.0.4/**.

After this step, we should have the `include` (`glfw3.h`) and `library` (`glfw3.lib`) files inside the `C:/Program Files (x86)/glfw-3.0.4` directory, as shown in the setup procedure using precompiled binaries.

# Installing the GLFW library in Mac OS X and Linux

The installation procedures for Mac and Linux are essentially identical using the command-line interface. To simplify the process, we recommend that you use MacPorts for Mac users.

## Getting ready

We assume that you have successfully installed the basic development tools, including CMake, as described in the earlier section. For maximum flexibility, we can compile the library directly from the source code (refer to [http://www.glfw.org/docs/latest/compile.html](http://www.glfw.org/docs/latest/compile.html) and [http://www.glfw.org/download.html](http://www.glfw.org/download.html)).

## How to do it...

For Mac users, enter the following command in a terminal to install GLFW using MacPorts:

[PRE5]

For Linux users (or Mac users who would like to practice using the command-line tools), here are the steps to compile and install the GLFW source package directly with the command-line interface:

1.  Create a new folder called `opengl_dev` and change the current directory to the new path:

    [PRE6]

2.  Obtain a copy of the GLFW source package (`glfw-3.0.4`) from the official repository: [http://sourceforge.net/projects/glfw/files/glfw/3.0.4/glfw-3.0.4.tar.gz](http://sourceforge.net/projects/glfw/files/glfw/3.0.4/glfw-3.0.4.tar.gz).
3.  Extract the package.

    [PRE7]

4.  Perform the compilation and installation:

    [PRE8]

## How it works...

The first set of commands create a new working directory to store the new files retrieved using the `wget` command, which downloads a copy of the GLFW library to the current directory. The `tar xzvf` command extracts the compressed packages and creates a new folder with all the contents.

Then, the `cmake` command automatically generates the necessary build files that are needed for the compilation process to the current `build` directory. This process also checks for missing dependencies and verifies the versioning of the applications.

The `make` command then takes all instructions from the Makefile script that is generated automatically and compiles the source code into libraries.

The `sudo make install` command installs the library header files as well as the static or shared libraries onto your machine. As this command requires writing to the root directory, the `sudo` command is needed to grant such permissions. By default, the files will be copied to the `/usr/local` directory. In the rest of the book, we will assume that the installations follow these default paths.

For advanced users, we can optimize the compilation by configuring the packages with the CMake GUI (`cmake-gui`).

![How it works...](img/9727OS_01_10.jpg)

For example, you can enable the `BUILD_SHARED_LIBS` option if you are planning to compile the GLFW library as a shared library. In this book, we will not explore the full functionality of the GLFW library, but these options can be useful to developers who are looking for further customizations. Additionally, you can customize the installation prefix (`CMAKE_INSTALL_PREFIX`) if you would like to install the library files at a separate location.

# Creating your first OpenGL application with GLFW

Now that you have successfully configured your development platform and installed the GLFW library, we will provide a tutorial on how to create your first OpenGL-based application.

## Getting ready

At this point, you should already have all the pre requisite tools ready regardless of which operating system you may have, so we will immediately jump into building your first OpenGL application using these tools.

## How to do it...

The following code outlines the basic steps to create a simple OpenGL program that utilizes the GLFW library and draws a rotating triangle:

1.  Create an empty file, and then include the header file for the GLFW library and standard C++ libraries:

    [PRE9]

2.  Initialize GLFW and create a GLFW window object (640 x 480):

    [PRE10]

3.  Define a loop that terminates when the window is closed:

    [PRE11]

4.  Set up the viewport (using the width and height of the window) and clear the screen color buffer:

    [PRE12]

5.  Set up the camera matrix. Note that further details on the camera model will be discussed in [Chapter 3](ch03.html "Chapter 3. Interactive 3D Data Visualization"), *Interactive 3D Data Visualization*:

    [PRE13]

6.  Draw a rotating triangle and set a different color (red, green, and blue channels) for each vertex (*x*, *y*, and *z*) of the triangle. The first line rotates the triangle over time:

    [PRE14]

7.  Swap the front and back buffers (GLFW uses double buffering) to update the screen and process all pending events:

    [PRE15]

8.  Release the memory and terminate the GLFW library. Then, exit the application:

    [PRE16]

9.  Save the file as `main.cpp` using the text editor of your choice.

## How it works...

By including the GLFW library header, `glfw3.h`, we automatically import all necessary files from the OpenGL library. Most importantly, GLFW automatically determines the platform and thus allows you to write portable source code seamlessly.

In the main function, we must first initialize the GLFW library with the **glfwInit** function in the main thread. This is required before any GLFW functions can be used. Before a program exits, GLFW should be terminated to release any allocated resources.

Then, the **glfwCreateWindow** function creates a window and its associated context, and it also returns a pointer to the `GLFWwindow` object. Here, we can define the width, height, title, and other properties for the window. After the window is created, we then call the **glfwMakeContextCurrent** function to switch the context and make sure that the context of the specified window is current on the calling thread.

At this point, we are ready to render our graphics element on the window. The **while** loop provides a mechanism to redraw our graphics as long as the window remains open. OpenGL requires an explicit setup on the camera parameters; further details will be discussed in the upcoming chapters. In the future, we can provide different parameters to simulate perspective and also handle more complicated issues (such as anti-aliasing). For now, we have set up a simple scene to render a basic primitive shape (namely a triangle) and fixed the color for the vertices. Users can modify the parameters in the **glColor3f** and **glVertex3f** functions to change the color as well as the position of the vertices.

This example demonstrates the basics required to create graphics using OpenGL. Despite the simplicity of the sample code, it provides a nice introductory framework on how you can create high-performance graphics rendering applications with graphics hardware using OpenGL and GLFW.

# Compiling and running your first OpenGL application in Windows

There are several ways to set up an OpenGL project. Here, we create a sample project using Visual Studio 2013 or higher and provide a complete walkthrough for the first-time configuration of the OpenGL and GLFW libraries. These same steps can be incorporated into your own projects in the future.

## Getting ready

Assuming that you have both Visual Studio 2013 and GLFW (version 3.0.4) installed successfully on your environment, we will start our project from scratch.

## How to do it...

In Visual Studio 2013, use the following steps to create a new project and compile the source code:

1.  Open Visual Studio 2013 (VS Express 2013 for desktop).
2.  Create a new Win32 Console Application and name it as `Tutorial1`.![How to do it...](img/9727OS_01_11.jpg)
3.  Check the **Empty project** option, and click on **Finish**.![How to do it...](img/9727OS_01_12.jpg)
4.  Right-click on **Source Files**, and add a new C++ source file (**Add** | **New Item**) called **main.cpp**.![How to do it...](img/9727OS_01_13.jpg)
5.  Copy and paste the source code from the previous section into the **main.cpp** and save it.
6.  Open **Project Properties** (*Alt* + *F7*).
7.  Add the `include` path of the GLFW library, **C:\Program Files (x86)\glfw-3.0.4\include**, by navigating to **Configuration Properties** | **C/C++** | **General** | **Additional Include Directories**.![How to do it...](img/9727OS_01_14.jpg)

    ### Tip

    **Downloading the example code**

    You can download the example code files from your account at [http://www.packtpub.com](http://www.packtpub.com) for all the Packt Publishing books you have purchased. If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files e-mailed directly to you.

8.  Add the GLFW library path, **C:\Program Files (x86)\glfw-3.0.4\lib**, by navigating to **Configuration Properties** | **Linker** | **General** | **Additional Library Directories**.![How to do it...](img/9727OS_01_15.jpg)
9.  Add the GLFW and OpenGL libraries (`glu32.lib`, `glfw3.lib` and `opengl32.lib`) by navigating to **Configuration Properties** | **Linker** | **Input** | **Additional Dependencies**.![How to do it...](img/9727OS_01_16.jpg)
10.  Build **Solution** (press *F7*).
11.  Run the program (press *F5*).

Here is your first OpenGL application showing a rotating triangle that is running natively on your graphics hardware. Although we have only defined the color of the vertices to be red, green, and blue, the graphics engine interpolates the intermediate results and all calculations are performed using the graphics hardware. The screenshot is shown as follows:

![How to do it...](img/9727OS_01_17.jpg)

# Compiling and running your first OpenGL application in Mac OS X or Linux

Setting up a Linux or Mac machine is made much simpler with the command-line interface. We assume that you have all the components that were discussed earlier ready, and all default paths are used as recommended.

## Getting ready

We will start by compiling the sample code described previously. You can download the complete code package from the official website of Packt Publishing [https://www.packtpub.com](https://www.packtpub.com). We assume that all files are saved to a top-level directory called `code` and the `main.cpp` file is saved inside the `/code/Tutorial1` subdirectory.

## How to do it...

1.  Open a terminal or an equivalent command-line interface.
2.  Change the current directory to the working directory:

    [PRE17]

3.  Enter the following command to compile the program:

    [PRE18]

4.  Run the program:

    [PRE19]

Here is your first OpenGL application that runs natively on your graphics hardware and displays a rotating triangle. Although we have defined the color of only three vertices to be red, green, and blue, the graphics engine interpolates the intermediate results and all calculations are performed using the graphics hardware.

![How to do it...](img/9727OS_01_18.jpg)

To further simplify the process, we have provided a compile script in the sample code. You can execute the script by simply typing the following commands in a terminal:

[PRE20]

You may notice that the OpenGL code is platform-independent. One of the most powerful features of the GLFW library is that it handles the windows management and other platform-dependent functions behind the scene. Therefore, the same source code (`main.cpp`) can be shared and compiled on multiple platforms without the need for any changes.