# Preface

As a leading cross-platform toolkit for all significant desktop, mobile, and embedded platforms, Qt is becoming more popular by the day. This book will help you learn the nitty-gritty of Qt and will equip you with the necessary toolsets to build apps and games. This book is designed as a beginner's guide to take programmers that are new to Qt from the basics, such as objects, core classes, widgets, and so on, and new features in version 5.4, to a level where they can create a custom application with best practices when it comes to programming with Qt.

With a brief introduction on how to create an application and prepare a working environment for both desktop and mobile platforms, we will dive deeper into the basics of creating graphical interfaces and Qt's core concepts of data processing and display before you try to create a game. As you progress through the chapters, you'll learn to enrich your games by implementing network connectivity and employing scripting. Delve into Qt Quick, OpenGL, and various other tools to add game logic, design animation, add game physics, and build astonishing UIs for games. Toward the end of this book, you'll learn to exploit mobile device features, such as accelerators and sensors, to build engaging user experiences.

# What this book covers

[Chapter 1](ch01.html "Chapter 1. Introduction to Qt"), *Introduction to Qt*, will familiarize you with the standard behavior that is required when creating cross-platform applications as well as show you a bit of history of Qt and how it evolved over time with an emphasis on the most recent architectural changes in Qt*.*

[Chapter 2](ch02.html "Chapter 2. Installation"), *Installation*, will guide you through the process of installing a Qt binary release for desktop platforms, setting up the bundled IDE, and looking at various configuration options related to cross-platform programming.

[Chapter 3](ch03.html "Chapter 3. Qt GUI Programming"), *Qt GUI Programming*, will show you how to create classic user interfaces with the Qt Widgets module. It will also familiarize you with the process of compiling applications using Qt.

[Chapter 4](ch04.html "Chapter 4. Qt Core Essentials"), *Qt Core Essentials*, will familiarize you with the concepts related to data processing and display in Qt—file handling in different formats, Unicode text handling and displaying user-visible strings in different languages, and regular expression matching.

[Chapter 5](ch05.html "Chapter 5. Graphics with Qt"), *Graphics with Qt*, describes the whole mechanism related to creating and using graphics in Qt in 2D and 3D. It also presents multimedia capabilities for audio and video (capturing, processing, and output)

[Chapter 6](ch06.html "Chapter 6. Graphics View"), *Graphics View*, will familiarize you with 2D-object-oriented graphics in Qt. You will learn how to use built-in items to compose the final results as well as create your own items supplementing what is already available and possibly animate them.

[Chapter 7](ch07.html "Chapter 7. Networking"), *Networking*, will demonstrate the IP networking technologies that are available in Qt. It will teach you how to connect to TCP servers, implement a reliable server using TCP, and implement an unreliable server using UDP.

[Chapter 8](ch08.html "Chapter 8. Scripting"), *Scripting*, shows you the benefits of scripting in applications. It will teach you how to employ a scripting engine for a game by using JavaScript. It will also suggest some alternatives to JavaScript for scripting that can be easily integrated with Qt.

[Chapter 9](ch09.html "Chapter 9. Qt Quick Basics"), *Qt Quick Basics*, will teach you to program resolution-independent fluid user interfaces using a QML declarative engine and Qt Quick 2 scene graph environment. In addition, you will learn how to implement new graphical items in your scenes.

[Chapter 10](ch10.html "Chapter 10. Qt Quick"), *Qt Quick*, will show you how to bring dynamics to various aspects of a UI. You will see how to create fancy graphics and animations in Qt Quick by using the particle engine, GLSL shaders and built-in animation, and state machine capabilities, and you will learn how to use these techniques in games.

*Chapter 11*, *Miscellaneous and Advanced Concepts*, covers the important aspects of Qt programming that didn't make it into the other chapters but may be important for game programming. This chapter is available online at the link [https://www.packtpub.com/sites/default/files/downloads/Advanced_Concepts.pdf](https://www.packtpub.com/sites/default/files/downloads/Advanced_Concepts.pdf).

# What you need for this book

All you need for this book is a Windows machine with the latest version of Qt installed. The examples presented in this book are based on Qt 5.4.

Qt can be downloaded from [http://www.qt.io/download-open-source/](http://www.qt.io/download-open-source/).

# Who this book is for

The expected readers of this book will be application and UI developers/programmers who have basic/intermediate functional knowledge of C++. The target audience also includes C++ programmers. No previous experience with Qt is required for you to read this book. Developers with up to a year of Qt experience will also benefit from the topics covered in this book.

# Sections

In this book, you will find several headings that appear frequently (Time for action, What just happened?, Pop quiz, and Have a go hero).

To give clear instructions on how to complete a procedure or task, we use these sections as follows:

# Time for action – heading

1.  Action 1
2.  Action 2
3.  Action 3

Instructions often need some extra explanation to ensure they make sense, so they are followed with these sections:

## *What just happened?*

This section explains the working of the tasks or instructions that you have just completed.

You will also find some other learning aids in the book, for example:

## Pop quiz – heading

These are short multiple-choice questions intended to help you test your own understanding.

## Have a go hero – heading

These are practical challenges that give you ideas to experiment with what you have learned.

# Conventions

You will also find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "This API is centered on `QNetworkAccessManager`, which handles the complete communication between your game and the Internet."

A block of code is set as follows:

[PRE0]

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

[PRE1]

Any command-line input or output is written as follows:

[PRE2]

**New terms** and **important words** are shown in bold. Words that you see on the screen, in menus or dialog boxes for example, appear in the text like this: "On the **Select Destination Location** screen, click on **Next** to accept the default destination."

### Note

Warnings or important notes appear in a box like this.

### Tip

Tips and tricks appear like this.

# Reader feedback

Feedback from our readers is always welcome. Let us know what you think about this book—what you liked or disliked. Reader feedback is important for us as it helps us develop titles that you will really get the most out of.

To send us general feedback, simply e-mail `<[feedback@packtpub.com](mailto:feedback@packtpub.com)>`, and mention the book's title in the subject of your message.

If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, see our author guide at [www.packtpub.com/authors](http://www.packtpub.com/authors).

# Customer support

Now that you are the proud owner of a Packt book, we have a number of things to help you to get the most from your purchase.

## Downloading the example code

You can download the example code files from your account at [http://www.packtpub.com](http://www.packtpub.com) for all the Packt Publishing books you have purchased. If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files e-mailed directly to you.

## Downloading the color images of this book

We also provide you with a PDF file that has color images of the screenshots/diagrams used in this book. The color images will help you better understand the changes in the output. You can download this file from [https://www.packtpub.com/sites/default/files/downloads/GameProgrammingUsingQt_ColoredImages.pdf](https://www.packtpub.com/sites/default/files/downloads/GameProgrammingUsingQt_ColoredImages.pdf).

## Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books—maybe a mistake in the text or the code—we would be grateful if you could report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the **Errata Submission Form** link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded to our website or added to any list of existing errata under the Errata section of that title.

To view the previously submitted errata, go to [https://www.packtpub.com/books/content/support](https://www.packtpub.com/books/content/support) and enter the name of the book in the search field. The required information will appear under the **Errata** section.

## Piracy

Piracy of copyrighted material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works in any form on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at `<[copyright@packtpub.com](mailto:copyright@packtpub.com)>` with a link to the suspected pirated material.

We appreciate your help in protecting our authors and our ability to bring you valuable content.

## Questions

If you have a problem with any aspect of this book, you can contact us at `<[questions@packtpub.com](mailto:questions@packtpub.com)>`, and we will do our best to address the problem.