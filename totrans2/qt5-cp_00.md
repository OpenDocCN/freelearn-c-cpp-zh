# Preface

Qt has been developed as a cross-platform framework and has been provided free to the public for years. It's mainly used to build GUI applications. It also provides thousands of APIs for easier development.

Qt 5, the latest major version of Qt, has once again proven to be the most popular cross-platform toolkit. With all these platform-independent classes and functions, you only need to code once, and then you can make it run everywhere.

In addition to the traditional and powerful C++, Qt Quick 2, which is more mature, can help web developers to develop dynamic and reliable applications, since QML is very similar to JavaScript.

# What this book covers

[Chapter 1](ch01.xhtml "Chapter 1. Creating Your First Qt Application"), *Creating Your First Qt Application*, takes you through the fundamental concepts of Qt, such as signals and slots, and helps you create your first Qt and Qt Quick applications.

[Chapter 2](ch02.xhtml "Chapter 2. Building a Beautiful Cross-platform Clock"), *Building a Beautiful Cross-platform Clock*, teaches you how to read and write configurations and handle cross-platform development.

[Chapter 3](ch03.xhtml "Chapter 3. Cooking an RSS Reader with Qt Quick"), *Cooking an RSS Reader with Qt Quick*, demonstrates how to develop a stylish RSS Reader in QML, which is a script language quite similar to JavaScript.

[Chapter 4](ch04.xhtml "Chapter 4. Controlling Camera and Taking Photos"), *Controlling Camera and Taking Photos*, shows you how to access camera devices through the Qt APIs and make use of the status and menu bars.

[Chapter 5](ch05.xhtml "Chapter 5. Extending Paint Applications with Plugins"), *Extending Paint Applications with Plugins*, teaches you how to make applications extendable and write plugins, by using the Paint application as as an example.

[Chapter 6](ch06.xhtml "Chapter 6. Getting Wired and Managing Downloads"), *Getting Wired and Managing Downloads*, shows you how to utilize Qt's network module using the progress bar, as well as learning about threaded programming in Qt.

[Chapter 7](ch07.xhtml "Chapter 7. Parsing JSON and XML Documents to Use Online APIs"), *Parsing JSON and XML Documents to Use Online APIs*, teaches you how to parse JSON and XML documents in both Qt/C++ and Qt Quick/QML, which is essential to obtain data from online APIs.

[Chapter 8](ch08.xhtml "Chapter 8. Enabling Your Qt Application to Support Other Languages"), *Enabling Your Qt Application to Support Other Languages*, demonstrates how to make internationalized applications, translate strings using Qt Linguist, and then load translation files dynamically.

[Chapter 9](ch09.xhtml "Chapter 9. Deploying Applications on Other Devices"), *Deploying Applications on Other Devices*, shows you how to package and make your applications redistributable on Windows, Linux, and Android.

[Chapter 10](ch10.xhtml "Chapter 10. Don't Panic When You Encounter These Issues"), *Don't Panic When You Encounter These Issues*, gives you some solutions and advice for common issues during Qt and Qt Quick application development and shows you how to debug Qt and Qt Quick applications.

# What you need for this book

Qt is cross-platform, which means you can use it on almost all operating systems, including Windows, Linux, BSD, and Mac OS X. The hardware requirements are listed as follows:

*   A computer (PC or Macintosh)
*   A webcam or a connected camera device
*   Available Internet connection

An Android phone or tablet is not required, but is recommended so that you can test applications on a real Android device.

All the software mentioned in this book, including Qt itself, is free of charge and can be downloaded from the Internet.

# Who this book is for

If you are a programmer looking for a truly cross-platform GUI framework to help you save time by side-stepping issues involving incompatibility between different platforms and building applications using Qt 5 for multiple targets, this book is most certainly intended for you. It is assumed that you have basic programming experience of C++.

# Conventions

In this book, you will find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "The UI files are under the `Forms` directory."

A block of code is set as follows:

[PRE0]

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

[PRE1]

Any command-line input or output is written as follows:

[PRE2]

**New terms** and **important words** are shown in bold. Words that you see on the screen, for example, in menus or dialog boxes, appear in the text like this: " Navigate to **File** | **New File** or **Project**."

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

## Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books—maybe a mistake in the text or the code—we would be grateful if you could report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the **Errata Submission Form** link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded to our website or added to any list of existing errata under the Errata section of that title.

To view the previously submitted errata, go to [https://www.packtpub.com/books/content/support](https://www.packtpub.com/books/content/support) and enter the name of the book in the search field. The required information will appear under the **Errata** section.

## Piracy

Piracy of copyrighted material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works in any form on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at `<[copyright@packtpub.com](mailto:copyright@packtpub.com)>` with a link to the suspected pirated material.

We appreciate your help in protecting our authors and our ability to bring you valuable content.

## Questions

If you have a problem with any aspect of this book, you can contact us at `<[questions@packtpub.com](mailto:questions@packtpub.com)>`, and we will do our best to address the problem.