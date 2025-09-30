# Chapter 1. Introduction to Qt

> *In this chapter, you will learn what Qt is and how it evolved. We will pay special attention to the differences between Qt's major versions 4 and 5\. Finally, you will learn to decide on which of the available Qt licensing schemes to choose for our projects.*

# The cross-platform programming

Qt is an application programming framework that is used to develop cross-platform applications. What this means is that software written for one platform can be ported and executed on another platform with little or no effort. This is obtained by limiting the application source code to a set of calls to routines and libraries available to all the supported platforms, and by delegating all tasks that may differ between platforms (such as drawing on the screen and accessing system data or hardware) to Qt. This effectively creates a layered environment (as shown in the following figure), where Qt hides all platform-dependent aspects from the application code:

![The cross-platform programming](img/8874OS_01_01.jpg)

Of course, at times we need to use some functionality that Qt doesn't provide. In such situations, it is important to use conditional compilation like the one used in the following code:

[PRE0]

### Tip

**Downloading the example code**

You can download the example code files from your account at [http://www.packtpub.com](http://www.packtpub.com) for all the Packt Publishing books you have purchased. If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register there to have the files e-mailed directly to you.

## *What just happened?*

Before the code is compiled, it is first fed to a preprocessor that may change the final text that is going to be sent to a compiler. When it encounters a `#ifdef` directive, it checks for the existence of a label that will follow (such as `Q_OS_WIN32`), and only includes a block of code in compilation if the label is defined. Qt makes sure to provide proper definitions for each system and compiler so that we can use them in such situations.

### Tip

You can find a list of all such macros in the Qt reference manual under the term "QtGlobal".

## Qt Platform Abstraction

Qt itself is separated into two layers. One is the core Qt functionality that is implemented in a standard C++ language, which is essentially platform-independent. The other is a set of small plugins that implement a so-called **Qt Platform Abstraction** (**QPA**) that contains all the platform-specific code related to creating windows, drawing on surfaces, using fonts, and so on. Therefore, porting Qt to a new platform in practice boils down to implementing the QPA plugin for it, provided the platform uses one of the supported standard C++ compilers. Because of this, providing basic support for a new platform is work that can possibly be done in a matter of hours.

## Supported platforms

The framework is available for a number of platforms, ranging from classical desktop environments through embedded systems to mobile phones. The following table lists down all the platforms and compiler families that Qt supports at the time of writing. It is possible that when you are reading this, a couple more rows could have been added to this table:

| Platform | QPA plugins | Supported compilers |
| --- | --- | --- |
| Linux | XCB (X11) and Wayland | GCC, LLVM (clang), and ICC |
| Windows XP, Vista, 7, 8, and 10 | Windows | MinGW, MSVC, and ICC |
| Mac OS X | Cocoa | LLVM (clang) and GCC |
| Linux Embedded | DirectFB, EGLFS, KMS, and Wayland | GCC |
| Windows Embedded | Windows | MSVC |
| Android | Android | GCC |
| iOS | iOS | LLVM (clang) and GCC |
| Unix | XCB (X11) | GCC |
| RTOS (QNX, VxWorks, and INTEGRITY) | qnx | qcc, dcc, and GCC |
| BlackBerry 10 | qnx | qcc |
| Windows 8 (WinRT) | winrt | MSVC |
| Maemo, MeeGo, and Sailfish OS | XCB (X11) | GCC |
| Google Native Client (unsupported) | pepper | GCC |

# A journey through time

The development of Qt was started in 1991 by two Norwegians—Eirik Chambe-Eng and Haavard Nord, who were looking to create a cross-platform GUI programming toolkit. The first commercial client of Trolltech (the company that created the Qt toolkit) was the European Space Agency. The commercial use of Qt helped Trolltech sustain further development. At that time, Qt was available for two platforms—Unix/X11 and Windows; however, developing with Qt for Windows required buying a proprietary license, which was a significant drawback in porting the existing Unix/Qt applications.

A major step forward was the release of Qt Version 3.0 in 2001, which saw the initial support for Mac as well as an option to use Qt for Unix and Mac under a liberal GPL license. Still, Qt for Windows was only available under a paid license. Nevertheless, at that time, Qt had support for all the important players in the market—Windows, Mac, and Unix desktops, with Trolltech's mainstream product and Qt for embedded Linux.

In 2005, Qt 4.0 was released, which was a real breakthrough for a number of reasons. First, the Qt API was completely redesigned, which made it cleaner and more coherent. Unfortunately, at the same time, it made the existing Qt-based code incompatible with 4.0, and many applications needed to be rewritten from scratch or required much effort to be adapted to the new API. It was a difficult decision, but from the time perspective, we can see it was worth it. Difficulties caused by changes in the API were well countered by the fact that Qt for Windows was finally released under GPL. Many optimizations were introduced that made Qt significantly faster. Lastly, Qt, which was a single library until now, was divided into a number of modules:

![A journey through time](img/8874OS_01_02.jpg)

This allowed programmers to only link to the functionality that they used in their applications, reducing the memory footprint and dependencies of their software.

In 2008, Trolltech was sold to Nokia, which at that time was looking for a software framework to help it expand and replace its Symbian platform in the future. The Qt community became divided, some people were thrilled, others worried after seeing Qt's development get transferred to Nokia. Either way, new funds were pumped into Qt, speeding up its progress and opening it for mobile platforms—Symbian and then Maemo and MeeGo.

For Nokia, Qt was not considered a product of its own, but rather a tool. Therefore, they decided to introduce Qt to more developers by adding a very liberal LGPL license that allowed the usage of the framework for both open and closed source development.

Bringing Qt to new platforms and less powerful hardware required a new approach to create user interfaces and to make them more lightweight, fluid, and eye candy. Nokia engineers working on Qt came up with a new declarative language to develop such interfaces—the **Qt Modeling Language** (**QML**) and a Qt runtime for it called Qt Quick.

The latter became the primary focus of the further development of Qt, practically stalling all nonmobile-related work, channeling all efforts to make Qt Quick faster, easier, and more widespread. Qt 4 was already in the market for 7 years and it became obvious that another major version of Qt had to be released. It was decided to bring more engineers to Qt by allowing anyone to contribute to the project.

Nokia did not manage to finish working on Qt 5.0\. As a result of an unexpected turn over of Nokia toward different technology in 2011, the Qt division was sold in mid-2012 to the Finnish company Digia that managed to complete the effort and release Qt 5.0 in December of the same year.

# New in Qt 5

The API of Qt 5 does not differ much from that of Qt 4\. Therefore, Qt 5 is almost completely source compatible with its predecessor, which means that we only need a minimal effort to port the existing applications to Qt 5\. This section gives a brief introduction to the major changes between versions 4 and 5 of Qt. If you are already familiar with Qt 4, this can serve as a small compendium of what you need to pay attention to if you want to use the features of Qt 5 to their fullest extent.

## Restructured codebase

The biggest change compared to the previous major release of Qt and the one that is immediately visible when we try to build an older application against Qt 5 is that the whole framework was refactored into a different set of modules. Because it expanded over time and became harder to maintain and update for the growing number of platforms that it supported, a decision was made to split the framework into much smaller modules contained in two module groups—Qt Essentials and Qt Add-ons. A major decision relating to the split was that each module could now have its own independent release schedule.

### Qt Essentials

The Essentials group contains modules that are mandatory to implement for every supported platform. This implies that if you are implementing your system using modules from this group only, you can be sure that it can be easily ported to any other platform that Qt supports. Some of the modules are explained as follows:

*   The QtCore module contains the most basic Qt functionality that all other modules rely on. It provides support for event processing, meta-objects, data I/O, text processing, and threading. It also brings a number of frameworks such as the animation framework, the State Machine framework, and the plugin framework.
*   The Qt GUI module provides basic cross-platform support to build user interfaces. It is much smaller compared with the same module from Qt 4, as the support for widgets and printing has been moved to separate modules. Qt GUI contains classes that are used to manipulate windows that can be rendered using either the raster engine (by specifying `QSurface::RasterSurface` as the surface type) or OpenGL (`QSurface::OpenGLSurface`). Qt supports desktop OpenGL as well as OpenGL ES 1.1 and 2.0.
*   The Qt Network module brings support for IPv4 and IPv6 networking using TCP and UDP as well as by controlling the device connectivity state. Compared to Qt 4, this module improves IPv6 support, adds support for opaque SSL keys (such as hardware key devices) and UDP multicast, and assembles MIME multipart messages to be sent over HTTP. It also extends support for DNS lookups.
*   Qt Multimedia allows programmers to access audio and video hardware (including cameras and FM radio) to record and play multimedia content.
*   Qt SQL brings a framework that is used to manipulate SQL databases in an abstract way.
*   Qt WebKit is a port of the WebKit 2 web browser engine to Qt. It provides classes to display and manipulate web content and integrates with your desktop application.
*   Qt Widgets extends the GUI module with the ability to create a user interface using widgets, such as buttons, edit boxes, labels, data views, dialog boxes, menus, and toolbars that are arranged using a special layout engine. It also contains the implementation of an object-oriented 2D graphics canvas called **Graphics View**. When porting Qt 4 applications to Qt 5, it is a good idea to start by enabling support of the widgets module (by adding *QT += widgets* to the project file) and then work your way down from here.
*   Qt Quick is an extension of Qt GUI, which provides means to create lightweight fluid user interfaces using QML. It is described in more detail later in this chapter as well as in [Chapter 9](ch09.html "Chapter 9. Qt Quick Basics"), *Qt Quick Basics*.

### Tip

There are also other modules in this group, but we will not focus on them in this book. If you want to learn more about them, you can look them up in the Qt reference manual.

### Qt Add-ons

This group contains modules that are optional for any platform. This means that if a particular functionality is not available on some platform or there is nobody willing to spend time working on this functionality for a platform, it will not prevent Qt from supporting this platform.

Some of the most important modules are QtConcurrent for parallel processing, Qt Script that allows us to use JavaScript in C++ applications, Qt3D that provides high-level OpenGL building blocks, and Qt XML Patterns that helps us to access XML data. Many others are also available, but we will not cover them here.

## Qt Quick 2.0

The largest upgrade to Qt functionality-wise is Qt Quick 2.0\. In Qt 4, the framework was implemented on top of Graphics View. This proved to be too slow when used with low-end hardware even with OpenGL ES acceleration enabled. This is because of the way Graphics View renders its content—it iterates all the items in sequence, calculates and sets its transformation matrix, paints the item, recalculates and resets the matrix for the next item, paints it, and so on. Since an item can contain any generic content drawn in an arbitrary order, it requires frequent changes to the GL pipeline, causing major slowdowns.

The new version of Qt Quick instead uses a scene-graph approach. It describes the whole scene as a graph of attributes and well-known operations. To paint the scene, information about the current state of the graph is gathered and the scene is rendered in a more optimal way. For example, it can first draw triangle strips from all items, then render fonts from all items, and so on. Furthermore, since the state of each item is represented by a subgraph, changes to each item can be tracked and it can be decided whether the visual representation of a particular item needs to be updated or not.

The old `QDeclarativeItem` class was replaced by `QQuickItem`, which has no ties to the Graphics View architecture. There is no routine available where you can directly paint the item, but there is a `QQuickPaintedItem` class available that aids in porting old code by rendering content based on `QPainter` to a texture and then rendering that texture using a scene-graph. Such items are, however, significantly slower than those directly using the graph approach, so if performance is important, they should be avoided.

Qt Quick plays an important role in Qt 5 and it is very useful to create games. We will cover this technology in detail in [Chapters 9](ch09.html "Chapter 9. Qt Quick Basics"), *Qt Quick Basics* and [Chapter 10](ch10.html "Chapter 10. Qt Quick"), *Qt Quick*.

## Meta-objects

In Qt 4, adding signals and slots to a class required the presence of a meta-object (that is, an instance of a class that describes another class) for that class. This was done by subclassing `QObject`, adding the `Q_OBJECT` macro to it, and declaring signals and slots in special scopes of the class. In Qt 5, this is still possible and advised in many situations, but we now have new interesting possibilities.

It is now acceptable to connect a signal to any compatible member function of a class or any callable entity, such as a standalone function or function object (`functor`). A side-effect is a compile-time compatibility check of the signal and the slot (as opposed to the runtime check of the "old" syntax).

## C++11 support

In August 2011, ISO approved a new standard for C++, commonly referred to as C++11\. It provides a number of optimizations and makes it easier for programmers to create effective code. While you could use C++11 together with Qt 4, it didn't provide any dedicated support for it. This has changed with Qt 5, which is now aware of C++11 and supports many of the constructs introduced by the new version of the language. In this book, we will sometimes use C++11 features in our code. Some compilers have C++11 support enabled by default, in others, you need to enable it. Don't worry if your compiler doesn't support C++11\. Each time we use such features, I will make you aware of it.

# Choosing the right license

Qt is available under two different licensing schemes—you can choose between a commercial license and an open source one. We will discuss both here to make it easier for you to choose. If you have any doubts regarding whether a particular licensing scheme applies to your use case, better consult a professional lawyer.

## An open source license

The advantage of open source licenses is that we don't have to pay anyone to use Qt; however, the downside is that there are some limitations imposed on how it can be used.

When choosing the open source edition, we have to decide between GPL 3.0 and LGPL 2.1 or 3\. Since LGPL is more liberal, in this chapter we will focus on it. Choosing LGPL allows you to use Qt to implement systems that are either open source or closed source—you don't have to reveal the sources of your application to anyone if you don't want to.

However, there are a number of restrictions you need to be aware of:

*   Any modifications that you make to Qt itself need to be made public, for example, by distributing source code patches alongside your application binary.
*   LGPL requires that users of your application must be able to replace Qt libraries that you provide them with other libraries with the same functionality (for example, a different version of Qt). This usually means that you have to dynamically link your application against Qt so that the user can simply replace Qt libraries with his own. You should be aware that such substitutions can decrease the security of your system, thus, if you need it to be very secure, open source might not be the option for you.
*   LGPL is incompatible with a number of licenses, especially proprietary ones, so it is possible that you won't be able to use Qt with some commercial components.

The open source edition of Qt can be downloaded directly from [http://www.qt.io](http://www.qt.io).

## A commercial license

All these restrictions are lifted if you decide to buy a commercial license for Qt. This allows you to keep the entire source code a secret, including any changes you may want to incorporate in Qt. You can freely link your application statically against Qt, which means fewer dependencies, a smaller deployment bundle size, and a faster startup. It also increases the security of your application, as end users cannot inject their own code into the application by replacing a dynamically loaded library with their own.

### Note

To buy a commercial license, go to [http://qt.io/buy](http://qt.io/buy).

# Summary

In this chapter, you learned about the architecture of Qt. We saw how it evolved over time and we had a brief overview of what it looks like now. Qt is a complex framework and we will not manage to cover it all, as some parts of its functionality are more important for game programming than others that you can learn on your own in case you ever need them. Now that you know what Qt is, we can proceed with the next chapter where you will learn how to install Qt on your development machine.