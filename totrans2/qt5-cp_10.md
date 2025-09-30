# Chapter 10. Don't Panic When You Encounter These Issues

During application development, you may get stuck with some issues. Qt is amazing, as always, since Qt Creator has an excellent **Debug** mode that can save you time when debugging. You'll learn how to debug either Qt/C++ or Qt Quick/QML applications. The following topics will be covered in this chapter:

*   Commonly encountered issues
*   Debugging Qt applications
*   Debugging Qt Quick applications
*   Useful resources

# Commonly encountered issues

Errors, or more appropriately, unexpected results, are definitely unavoidable during application development. Besides, there could also be compiler errors, or even application crashes. Please don't panic when you encounter these kinds of issues. To ease your pain and help you locate the problem, we have collected some commonly encountered and reproducible unexpected results and categorized them, as shown in the next sections.

## C++ syntax mistakes

For programming beginners, or developers who are not familiar with C and C++, the syntax of C++ is not easy to remember. If there are any syntax mistakes, the compiler will abort with error messages. In fact, the editor will display tildes below problematic statements, as shown here:

![C++ syntax mistakes](img/4615OS_10_01.jpg)

Among all C++ syntax mistakes, the most common one is a missing semicolon (;). C++ needs a semicolon to mark the end of a statement. Therefore, line 7 and line 8 are equivalent to the following line:

[PRE0]

This, in C++, is obviously written incorrectly. Not only will the editor highlight the error, the compiler will also give you a thorough error message. In this case, it'll display the following message:

`C:\Users\Symeon\OneDrive\Book_Dev\4615OS\4615OS_07\project\Weather_Demo\main.cpp:8: error: C2146: syntax error : missing ';' before identifier 'w'`

As you can tell, the compiler won't tell you that you should add a semicolon at the end of line 7\. Instead, it reads `missing;` before the `w` identifier, which is in line 8\. Anyway, in most cases the C++ syntax errors can be detected by the compiler, while most of them will first be detected by the editor. Thanks to the highlighting feature of Qt Creator, these types of mistakes should be avoided effectively.

It's recommended as a good habit that you add a semicolon before you press *Enter*. This is because in some cases the syntax may seem correct for compilers and Qt Creator, but it's definitely wrongly coded and will cause unexpected behavior.

## Pointer and memory

Anyone familiar with C and its wild pointers understands how easy it is to make a mistake regarding memory management. As we mentioned before, Qt has a superior memory management mechanism, which will release its child objects once the parent is deleted. This, unfortunately, may lead to a crash if the developer explicitly uses `delete` to release a child object.

The primary reason behind this is that `delete` is not a thread-safe operation. It may cause a double delete, resulting in a segment fault. Therefore, to release memory in a thread-safe way, we use the `deleteLater()` function defined in the `QObject` class, which means that this method is available for all classes inherited from `QObject`. As stated in the documentation, `deleteLater()` will schedule the object for deletion but the deletion won't happen immediately.

### Note

It's completely safe to call `deleteLater()` multiple times. Once the first deferred deletion is completed, any pending deletions are removed from the event queue. There won't be any double deletes.

There is another class dealing with memory management in Qt, `QObjectCleanupHandler`. This class watches the lifetime of multiple QObjects. You can treat it as a simple Qt garbage collector. For instance, there are a lot of `QTcpSocket` objects that need to be watched and deleted properly. These kinds of cases are not uncommon, especially for networking programs. An easy trick is to add all these objects to `QObjectCleanupHandler`. The following piece of code is a simple demonstration that adds `QObject` to `QObjectCleanupHandler ch`:

[PRE1]

Adding the `t` object to `ch` won't change the parent object of `t` from this to `&ch`. `QObjectCleanupHandler` is more like `QList` in this way. If `t` is deleted somewhere else, it'll get removed from the list of `ch` automatically. If there is no object left, the `isEmpty()` function will return `true`. All objects in `QObjectCleanupHandler` will be deleted when it's destroyed. You can also explicitly call `clear()` to delete all objects in `QObjectCleanupHandler` manually.

## Incompatible shared libraries

This type of errors are the so-called DLL Hell, which we discussed in the previous chapter. It results from incompatible shared libraries, which may lead to strange behavior or crashes.

In most cases, Qt libraries are backwards compatible, which means that you may replace all DLLs with newer ones and not need to recompile executables. Some certain modules or APIs may be deprecated and be deleted from a later version of Qt. For example, the `QGLWidget` class is replaced by a newly introduced `QOpenGLWidget` class in Qt 5.4\. `QGLWidget` is still provided for now though.

In the reverse direction, things are getting pretty bad. If your application calls an API that is introduced since, for example, Qt 5.4, the application definitely will malfunction with an older version of Qt, such as Qt 5.2.

The following is a simple program that makes use of `QSysInfo`, which is introduced in Qt 5.4\. The `main.cpp` file of this simple `incompat_demo` project is shown here:

[PRE2]

`QSysInfo::currentCpuArchitecture()` returns the architecture of the CPU that the application is running on as a `QString` object. If the version of Qt is high enough (greater than or equal to 5.4), it'll run as expected, as shown in the following screenshot:

![Incompatible shared libraries](img/4615OS_10_02.jpg)

As you can see, it says we're running this application on a 64-bit x86 CPU machine. However, if we put the compiled executable with DLLs from Qt 5.2, it'll give an error as shown here and crash:

![Incompatible shared libraries](img/4615OS_10_03.jpg)

This situation is rare, of course. However, if this happens, you'll get an idea about what goes wrong. From the error dialog, we can see the error is because of the missing `QSysInfo::currentCpuArchitecture` line in the dynamic link library.

Another DLL Hell is more complex and may be ignored by beginners. All libraries must be built by the same compiler. You can't use the MSVC libraries with GCC, which holds true for other compilers, such as ICC and Clang. Different compiler versions might cause incompatibility as well. You probably don't want to use a library compiled by GCC 4.3 in your development environment where the GCC version is 4.9\. However, libraries compiled by GCC 4.9.1 should be compatible with those compiled by GCC 4.9.2.

In addition to compilers, different architectures are often incompatible. For example, 64-bit libraries won't work on 32-bit platforms. Similarly, x86 libraries and binaries can't be used on the non-x86 devices, such as ARM and MIPS.

## Doesn't run on Android!

Qt was ported to Android not too long ago. Hence, there is a possibility that it runs well on a desktop PC but not on Android. On one hand, Android hardware varies, not even speaking of thousands of customized ROMs. Therefore, it is reasonable that some Android devices may encounter compatibility issues. On the other hand, the Qt application running on Android is a native C++ application with a Java wrapper, while binary executables are naturally more vulnerable to compatibility issues than scripts.

Anyway, here's the recipe:

1.  Try to run your application on another Android handset or virtual Android device.
2.  If it still doesn't work, it can be a potential bug of Qt on Android. We'll talk about how to report a bug to Qt at the end of this chapter.

# Debugging Qt applications

To debug any Qt application, you need to ensure that you have installed the debug symbols of the Qt libraries. On Windows, they are installed together with release version DLLs. Meanwhile, on Linux, you may need to install debug symbols by the distribution's package manager.

Some developers tend to use a function similar to `printf` to debug the application. Qt provides four global functions, which are shown in the following table, to print out debug, warnings, and error text:

| Function | Usage |
| --- | --- |
| `qDebug()` | This function is used for writing custom debug output. |
| `qWarning()` | This function is used for reporting warnings and recoverable errors. |
| `qCritical()` | This function is used for writing critical error messages and reporting system errors. |
| `qFatal()` | This function is used for printing fatal error messages shortly before exiting. |

Normally, you can just use a C-style method similar to `printf`.

[PRE3]

However, in most cases, we'll include the `<QtDebug>` header file so that we can use the stream operator (`<<`) as a more convenient way.

[PRE4]

The most powerful place of these functions is that they can output the contents of some complex classes', `QList` and `QMap`. It's noted that these complex data types can only be printed through a stream operator (`<<`).

Both `qDebug()` and `qWarning()` are debugging tools, which mean that they can be disabled at compile time by defining `QT_NO_DEBUG_OUTPUT` and `QT_NO_WARNING_OUTPUT`, respectively.

In addition to these functions, Qt also provides the `QObject::dumpObjectTree()` and `QObject::dumpObjectInfo()` functions which are often useful, especially when an application looks strange. `QObject::dumpObjectTree()` dumps information about signal connections, which is really useful if you think there may be a problem in signal slot connections. Meanwhile, the latter dumps a tree of children to the debug output. Don't forget to build the application in **Debug** mode, otherwise neither of them will print anything.

Apart from these useful debugging functions, Qt Creator has offered an intuitive way to debug your application. Ensure that you've installed Microsoft **Console debugger** (**CDB**) if you're using an MSVC compiler. In other cases, the GDB debugger is bundled in a MinGW version.

### Note

CDB is now a part of **Windows Driver Kit** (**WDK**); visit [http://msdn.microsoft.com/en-us/windows/hardware/hh852365](http://msdn.microsoft.com/en-us/windows/hardware/hh852365) to download it. Don't forget to check Debugging Tools for Windows during the installation.

Consider `Fancy_Clock` from [Chapter 2](ch02.xhtml "Chapter 2. Building a Beautiful Cross-platform Clock"), *Building a Beautiful Cross-platform Clock*, as an example. In the `MainWindow::setColour()` function, move the cursor to line 97, which is `switch (i) {`. Then, navigate to **Debug** | **Toggle Breakpoint** or just press *F9* on the keyboard. This will add a breakpoint on line 97, which will add a breakpoint marker (a red pause icon in front of a line number) as shown here:

![Debugging Qt applications](img/4615OS_10_04.jpg)

Now click on the **Start Debugging** button on the pane, which has a bug on it, or navigate to **Debug** | **Start Debugging** | **Start Debugging** on the menu bar, or press *F5* on the keyboard. This will recompile the application, if needed, and start it in **Debug** mode. At the same time, Qt Creator will automatically switch to **Debug** mode.

![Debugging Qt applications](img/4615OS_10_05.jpg)

The application is interrupted because of the breakpoint we set. You can see a yellow arrow indicating which line the application is currently on, as shown in the preceding screenshot. By default, on the right pane, you can see **Locals and Expressions** where all the local variables along with their values and types are shown. To change the default settings, navigate to **Window** | **Views**, and then choose what to display or hide.

The panes in the **Debug** mode are marked in blue text in this screenshot:

![Debugging Qt applications](img/4615OS_10_06.jpg)

Briefly said, you can monitor the variables in **Locals** and expressions in **Expressions**. **Stack** displays the current stack and all breakpoints can be managed in the **Breakpoints** pane.

On the bottom pane, there are a series of buttons to control the debugging process. The first six buttons are **Continue**, **Stop Debugger**, **Step Over**, **Step Into**, **Step Out**, and **Restart the debugging session**, respectively. **Step Over** is to execute a line of code as a whole. **Step Into** will step into a function or a subfunction, while **Step Out** can leave the current function or subfunction.

**Breakpoints** plays a crucial role in debugging, as you can tell whether a breakpoint represents a position or set of positions in the code that interrupts the application from being debugged and grants you control. Once it is interrupted, you can examine the state of the program or continue the execution, either line-by-line or continuously. Qt Creator shows breakpoints in the **Breakpoints** view, which is located at the lower-right-hand side by default. You can add or delete breakpoints in the **Breakpoints** view. To add a breakpoint, right-click on the **Breakpoints** view and select **Add Breakpoint…**; there will be an **Add Breakpoint** dialog as shown here:

![Debugging Qt applications](img/4615OS_10_07.jpg)

In the **Breakpoint type** field, select the location in the program code where you want the application to be interrupted. Other options are dependent on the selected type.

To move the breakpoint, simply drag the breakpoint marker and drop it on the destination. It's not an often needed function, though.

There're many ways to delete a breakpoint.

*   By clicking on the breakpoint marker in the editor, moving the cursor to the corresponding line, and navigating to **Debug** | **Toggle Breakpoint**, or by pressing *F9*
*   By right-clicking on the breakpoint in the **Breakpoints** view and selecting **Delete Breakpoint**
*   By selecting the breakpoint in the **Breakpoints** view and pressing the *Delete* button on the keyboard

The most powerful place is the previously introduced **Locals and Expressions** view. Every time the program stops under the control of the debugger, it retrieves information and displays it in the **Locals and Expressions** view. The **Locals** pane shows function parameters and local variables. There is a comprehensive display of data belonging to Qt's basic objects. In this case, when the program is interrupted in `MainWindow::setColour()`, there is a pointer whose **Value** is `"MainWindow"`. Instead of just memory address of this pointer, it can show you all the data and children that belong to this object:

![Debugging Qt applications](img/4615OS_10_08.jpg)

As you can see from preceding screenshot, this is a `MainWindow` instance, which is inherited from `QMainWindow`. It has three children items: `_layout`, `qt_rubberband`, and `centralWidget`. It's noted that only slot functions are displayed in `[methods]`. Now you'll understand why the **Locals** pane is the most important and commonly used view in the **Debug** mode.

On the other hand, the **Expressions** pane is even more powerful and can compute the values of arithmetic expressions or function calls. Right-click on the **Locals and Expressions** view and select **Add New Expression Evaluator…** in the context menu.

Note that the context menu entries are available only when the program is interrupted. In this case, `Fancy_Clock` is interrupted in the `MainWindow::setColour()` function where the local variable, `i`, can be used to perform some arithmetic operations. For example, we fill `i * 5` in the **New Evaluated Expression** pop-up dialog.

![Debugging Qt applications](img/4615OS_10_09.jpg)

In addition to arithmetic operations, you can call a function to evaluate the return value. However, this function must be accessible to the debugger, which means it's either compiled into the executable or can be invoked from a library.

The expression value will be re-evaluated after each step. After you click on the **OK** button, the expression `i * 5`, is shown in the **Expressions** pane as shown here:

![Debugging Qt applications](img/4615OS_10_10.jpg)

The value of `i` is now `3`. Therefore, the expression `i * 5` is evaluated as `15`.

> *"Expression evaluators are powerful, but slow down debugger operation significantly. It is advisable not to use them excessively, and to remove unneeded expression evaluators as soon as possible."*

Even if functions used in the expressions have side effects, they will be called each time the current frame changes. After all, the expression evaluator is powerful but bad for debugging speed.

# Debugging Qt Quick applications

We will use the `Weather_QML` project from [Chapter 7](ch07.xhtml "Chapter 7. Parsing JSON and XML Documents to Use Online APIs"), *Parsing JSON and XML Documents to Use Online APIs*, as a demonstration program to show how to debug a Qt Quick application.

First, we need to ensure that QML debugging is enabled. Open the `Weather_QML` project in Qt Creator. Then, perform the following steps:

1.  Switch to the **Projects** mode.
2.  Expand the **qmake** step in **Build Steps**.
3.  Check **Enable QML debugging** if it's not checked.

    ### Tip

    Debugging QML will open a socket at a well-known port, which poses a security risk. Anyone on your network could connect to the debugging application and execute any JavaScript function. Therefore, you have to make sure there are appropriate firewall rules.

The same procedure is used to start QML debugging, which is to navigate to **Debug** | **Start Debugging** | **Start Debugging**, or click the **Debug** button, or just press *F5*. It may trigger a **Windows Security Alert**, shown in the following screenshot. Don't forget to click on the **Allow access** button.

![Debugging Qt Quick applications](img/4615OS_10_11.jpg)

Once the application starts running, it behaves and performs as usual. However, you can perform some useful tasks in debugging mode. You can see all the elements and their properties in the **Locals** pane as we did for the Qt/C++ applications.

In addition to just watching these variables, you can change them temporarily and see the changes at runtime immediately. To change a value, you can either directly change it in the **Locals** pane or change it in **QML/JS Console**.

For example, to change the `title` property of `ApplicationWindow`, perform the following steps:

1.  Expand **ApplicationWindow** | **Properties** in the **Locals** pane.
2.  Double-click on the `title` entry.
3.  Change the value from `Weather QML` to `Yahoo! Weather`.
4.  Press the *Enter* or *Return* key on the keyboard to confirm.

Alternatively, you can change it in **QML/JS Console**. There is no need to expand `ApplicationWindow`; just click on `ApplicationWindow` in the **Locals** pane. You'll notice **Context** on the **QML/JS Console** panel will become `ApplicationWindow`, as shown in the following screenshot. Then, just input the `title="Yahoo! Weather"` command to change the title.

![Debugging Qt Quick applications](img/4615OS_10_12.jpg)

You'll notice the title in the application window is changed to **Yahoo! Weather** immediately, as shown here:

![Debugging Qt Quick applications](img/4615OS_10_13.jpg)

Meanwhile, the source code is left intact. This feature is really handy when you want to test a better value for a property. Instead of changing it in the code and rerunning, you can change and test it on the fly. In fact, you can also execute the JavaScript expressions in **QML/JS Console**, not just change their values.

# Useful resources

Still getting stuck with an issue? In addition to online search engines, there are two online forums that could also be useful for you. The first one is the forum in the Qt Project, whose URL is [http://qt-project.org/forums](http://qt-project.org/forums). The other one is maintained by a community site, Qt Centre, and its URL is [http://www.qtcentre.org/forum.php](http://www.qtcentre.org/forum.php).

In most cases, you should be able to find similar or even identical problems on these websites. If not, you can post a new thread asking for help. Describe the problem as thoroughly as possible so that other users can get an idea of what's going wrong.

There is a possibility that you did everything correctly but still might be getting unexpected results, compiler errors, or crashes. In this case, it may be a Qt bug. If you believe that you've encountered a Qt bug, you are encouraged to report it. It's easy to report a bug since Qt has a bug tracker, whose URL is [https://bugreports.qt.io](https://bugreports.qt.io).

### Tip

The quality of the bug report dramatically impacts how soon the bug will be fixed.

To produce a high-quality bug report, here is a simple step-by-step manual:

1.  Visit the Qt bug tracker website.
2.  Log in. If it's your first time, you need to create a new account. Remember to supply a valid e-mail address as this is the only way for the Qt developers to contact you.
3.  Use the **Search** field on the upper-right side to find any similar, or even identical bugs.
4.  If you find one, you can leave a comment with any additional information that you have. Besides, you can click on **Vote** to vote for that bug. Lastly, you could add yourself as a watcher if you want to track the progress.
5.  If not, click on **Create New Issues** and fill in the fields.

You should enter a brief descriptive text in **Summary**. This is not only for a higher chance to get it fixed, but also good for other people searching for existing bugs. For other fields, you're always encouraged to provide as much information as you can.

# Summary

After having a read through this chapter, you can sort out the majority of Qt-based issues on your own. We started off with a few commonly encountered problems, followed by how to debug Qt and Qt Quick applications. At the end, there were a few useful links to help you crack down on the varied issues and errors. If you encounter any problem with a particular Qt bug, don't panic, just go to the bug tracker and report it.