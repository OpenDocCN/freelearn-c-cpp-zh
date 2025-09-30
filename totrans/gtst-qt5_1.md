# Introducing Qt 5

Qt gives developers a great toolbox with which to create fantastic and practical applications with minimal stress, as you will soon discover. In this chapter, we will introduce Qt and describe how to set it up on a machine. By the end of the chapter, you should be able to do the following:

*   Install Qt
*   Write a simple program in Qt
*   Compile and run a Qt program

The objectives have been kept simple and straightforward. So let's get started!

# Installing Qt on Linux

The Ubuntu operating system makes it reasonably easy to install Qt 5\. Issue the following commands to set up your box:

```cpp
sudo apt-get install qt5-default
```

After the installation, Qt programs will be compiled and run from the command line. In [Chapter 6](bfdfd852-205f-4c4b-bb41-d798fdc865f7.xhtml), *Connecting Qt with Databases*, we will illustrate how to connect to the database using Qt. Issue the following command to ensure that the relevant libraries are installed for Qt to work with. The database that will'll connect to is MySQL:

```cpp
sudo apt-get install libqt5sql5-mysql
```

# Installing Qt on macOS

There are a variety of ways to get Qt installed on a Mac. To begin the process of installing Qt 5 on your Mac, you need to get Xcode installed on your machine. Issue the following commands on the Terminal:

```cpp
xcode-select --install
```

If you get the following output, then you are ready for the next series of steps:

```cpp
xcode-select: error: command line tools are already installed, use "Software Update" to install updates
```

*HomeBrew* is a package management software tool that allows you to easily install Unix tools that don't come shipped with the macOS.

If you don't already have it on your machine, you can install it by issuing the following command in a Terminal:

```cpp
 /user/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

After that, you should issue yet another set of commands to get Qt installed via the Terminal:

```cpp
curl -O https://raw.githubusercontent.com/Homebrew/homebrew-core/fdfc724dd532345f5c6cdf47dc43e99654e6a5fd/Formula/qt5.rb

brew install ./qt5.rb
```

In a few chapters' time, we will be working with the MySql database. To configure Qt 5 with MySql, issue the following command:

```cpp
brew install ./qt5 --with-mysql
```

This command should take a while to complete and, assuming nothing goes wrong, you are ready to write Qt programs.

# Installation on Windows

For readers using Windows, installation remains simple, albeit a little less straightforward. We can start by heading over to [http://download.qt.io](http://download.qt.io).

Select `official_releases/`, then `online_installers/`, and opt to download `qt-unified-windows-x86-online.exe`.

Run the program and opt to create an account. Click through to select the installation folder and don't forget to select the MinGW 5.3.0 32 bit option as the compiler when selecting the components that need to be installed.

Most of the commands in this book should run in this IDE.

# What is Qt?

Now that we have set up our boxes to start development, let's put together a hello world example. First, however, let's take a brief detour.

Qt is a toolkit for creating **Graphical User Interfaces** (**GUI**), as well as cross-platform applications. GUI applications are programs that employ the use of the mouse to issue commands to the computer for execution. Though Qt can, in some cases, be used without necessarily making use of this, therein lies its utility.

The difficulty in trying to produce the same look, feel, and functionality across multiple operating systems is one big hurdle you have to deal with when writing GUI applications. Qt completely does away with this impediment by providing a means to write code only once and ensuring that it runs on most operating systems without requiring much or any change.

Qt makes use of some modules. These modules group related functionalities together. The following lists some modules and what they do:

*   `QtCore`: As the name implies, these modules contains core and important classes for the Qt framework. These include containers, events, and thread management, among others.
*   `QtWidgets` and `QtGui`: This module contains classes for calling widgets. Widgets are the components that make up the majority of a graphical interface. These include buttons, textboxes, and labels.
*   `QtWebkit`: This module makes it possible to use web pages and apps within a Qt application.
*   `QtNetwork`: This module provides classes to connect to and communicate with network resources.
*   `QtXML`: For parsing XML documents, this module contains useful classes.
*   `QtSQL`: This module is feature-rich with classes and drivers that allow for connecting to databases, including My SQL, PostgreSQL, and SQLite.

# Hello world in Qt

In this section, we will put together a very simple hello world program. The program will show a simple button within a window. Create a file called `hello.cpp` in a newly created folder called `hello_world`. Open the file and insert the code:

```cpp
#include <QApplication>
#include <QLabel>
int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   QLabel label("Hello world !");
   label.show();
   return app.exec();
}
```

This looks like a regular C++ program, with the exception of unfamiliar classes being used.

Like any regular program, the `int main()` function is the entry point of our application.

An instance of the `QApplication` class is created, called `app`, and the arguments passed to the `main()` function. The `app` object is required because it sets off the `Event` loop that continues to run until we close the application. Without the `QApplication` object, you cannot really create a Qt GUI application.

However, it is possible to use certain aspects of Qt without the need to create an instance of `QApplication`.

Also, the constructor for `QApplication` requires that we pass the `argc` and `argv` to it.

We instantiate an object of the `QLabel` class, `label`. We pass the `"Hello World!"` string to its constructor. A `QLabel` represents what we call a widget, which is a term used to describe visual elements on the screen. Labels are used to hold text for display.

By default, created widgets are hidden. To display them, a call to the `show()` function has to be made.

To start the `Event` loop, the `app.exec()` line is executed. This passes control of the application to Qt.

The `return` keyword will pass an integer back to the operating system, indicating the state of the application when it was closed or exited.

To compile and run our program, navigate to the folder where `hello.cpp` is stored. Type the following command in the Terminal:

```cpp
% qmake -project
```

This will create the `hello_world.pro` file. The name `hello_world` is the name of the folder where `hello.cpp` is located. The generated file should change, depending on the location you stored the `hello.cpp` file.

Open the `hello_world.pro` file with any text editor of your choice. The following lines deserve some explanation:

```cpp
TEMPLATE = app
```

The value, `app`, here means that the final output of the project will be an application. Alternatively, it could be a library or sub-directory:

```cpp
TARGET = hello_world
```

The name, `hello_world`, here is the name of the application or (library) that will be executed:

```cpp
SOURCES += hello.cpp
```

Since `hello.cpp` is the only source file in our project, it is added to the `SOURCES` variable.

We need to generate a `Makefile` that will detail the steps needed to compile our hello world program. The benefit of this autogenerated `Makefile` is that it takes away the need for us to know the various nuances involved in compiling the program on the different operating systems.

While in the same project directory, issue the following command:

```cpp
% qmake
```

This generates a `Makefile` in the directory.

Now, issue the following command to compile the program:

```cpp
% make
```

The following error will be produced (along with further information) as the output from running the `make` command:

```cpp
#include <QApplication>
        ^~~~~~~~~~~~
```

Earlier on, we mentioned that various components and classes are packaged into modules. The `QApplication` is being utilized in our application, but the correct module has not been included. During compilation, this omission results in an error.

To fix this issue, open the `hello_world.pro` file and insert the following lines after the line:

```cpp
INCLUDEPATH += .
QT += widgets
```

This will add the `QtWidget` module, along with the `QtCore` modules, to the compiled program. With the correct module added, run the `make` command again on the command line:

```cpp
% make
```

A `hello_world` file will be generated in the same folder. Run this file from the command line as follows:

```cpp
% ./hello_world
```

On a macOS, the full path to the executable will be specified with the following path from the command line:

```cpp
./hello_world.app/Contents/MacOS/hello_world
```

This should produce the following output:

![](img/2c834541-f8a2-4f5c-b1e7-2c8f97c37a3e.png)

Well, there is our first GUI program. It displays the Hello world ! in a label. To close the application, click on the Close button of the window.

Let's add a dash of **Qt Style Sheet** (**QSS**) to give our label a little effect!

Modify the `hello.cpp` file as follows:

```cpp
#include <QApplication>
#include <QLabel>
int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   QLabel label("Hello world !");
   label.setStyleSheet("QLabel:hover { color: rgb(60, 179, 113)}");
   label.show();
   return app.exec();
}
```

The only change here is `label.setStyleSheet("QLabel:hover { color: rgb(60, 179, 113)}");`.

A QSS rule is passed as an argument to the `setStyleSheet` method on the `label` object. The rule sets every label within our application to show the color green when the cursor hovers over it.

Run the following commands to recompile the application and run it:

```cpp
% make
% ./hello_world
```

The program should appear as in the following screenshot. The label turns green when the mouse is placed over it:

![](img/e2178602-669c-4509-90db-06cb03263baa.png)

# Summary

This chapter laid the foundation for getting to know Qt and what it can be used for. Installing of Qt on macOS and Linux was outlined. A small hello world application was written and compiled, all from the command line, without any need for an IDE. This meant that we were also introduced to the various steps that lead to the final program.

Finally, the hello world application was modified to employ QSS in a bid to show what other things can be done to a widget.

In [Chapter 2](a0d84833-24c7-4f5d-933b-c4d99fe82034.xhtml), *Creating Widgets and Layouts*, we will explore more widgets in Qt and how to organize and group them.