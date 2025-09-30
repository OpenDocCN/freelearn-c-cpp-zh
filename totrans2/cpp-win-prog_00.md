# Preface

Application development has gained massive popularity because of the immense impact it has on various sectors. In this booming market, it has become critical to have the right set of tools to enable developers to build practical, user-friendly, and efficient applications. This book is focused on the use and implementation of Small Windows, which is a C++ object-oriented class library that eases the development of interactive Windows applications.

# What this book covers

[Chapter 1](ch01.html "Chapter 1. Introduction"), *Introduction*, gives an introduction to Small Windows, which is a class library that encapsulates a part of the Win32 API.

[Chapter 2](ch02.html "Chapter 2. Hello, Small World!"), *Hello, Small World!*, starts by building a (very) small application—the Small Windows version of the famous Hello World program. Then, we will continue with a (still rather small) application that handles circles in a window. The user can add and move circles, change their colors, and save and load circles.

[Chapter 3](ch03.html "Chapter 3. Building a Tetris Application"), *Building a Tetris Application*, explores a version of the classic Tetris game. Seven different kinds of figure are falling down the screen and the user’s task is to move or rotate them so that as many rows as possible can be completely filled and removed.

[Chapter 4](ch04.html "Chapter 4. Working with Shapes and Figures"), *Working with Shapes and Figures*, teaches you how to build a drawing program, which can be regarded as a more advanced version of the circle application. It is possible to create and remove figures as well as mark and drag figures.

[Chapter 5](ch05.html "Chapter 5. The Figure Hierarchy"), *The Figure Hierarchy*, continues to build the drawing program. We can define a class hierarchy with lines, arrows, rectangles, and ellipses.

[Chapter 6](ch06.html "Chapter 6. Building a Word Processor"), *Building a Word Processor*, describes a word processor capable of formatting individual characters.

[Chapter 7](ch07.html "Chapter 7. Keyboard Input and Character Calculation"), *Keyboard Input and Character Calculation*, discusses how the word processor handles many keyboard input combinations and calculates the size and position of each individual character.

[Chapter 8](ch08.html "Chapter 8. Building a Spreadsheet Application"), *Building a Spreadsheet Application*, talks about the final application, which is a spreadsheet program capable of calculating formulas with the four rules of arithmetic. It is also possible to cut and paste blocks of cells.

[Chapter 9](ch09.html "Chapter 9. Formula Interpretation"), *Formula Interpretation*, explains that when the user inputs a formula, we need to interpret it. The process is divided into scanning and parsing, which we will look into in this chapter.

[Chapter 10](ch10.html "Chapter 10. The Framework"), *The Framework*, describes the most central part of Small Windows. This chapter begins the description of Small Windows. The Application class handles the message loop of the application and the registration of Windows classes. The Window class handles basic window functionality.

[Chapter 11](ch11.html "Chapter 11. The Document"), *The Document*, talks about the document-based Window subclasses, that is, the Document class that provides basic document functionality, such as menus and accelerators, and the Standard Document framework, which provides a document-based framework.

[Chapter 12](ch12.html "Chapter 12. The Auxiliary Classes"), *The Auxiliary Classes*, explores a set of small auxiliary classes handling points and sizes, rectangles, colors and fonts, dynamic lists, and tree structures.

[Chapter 13](ch13.html "Chapter 13. The Registry, Clipboard, Standard Dialogs, and Print Preview"), *The Registry, Clipboard, Standard Dialogs, and Print Preview*, explains the implementation of the registry and clipboard, the standard dialogs to save and load files, choosing color or font, or printing a document. The chapter also explains the implementation a class for print previewing.

[Chapter 14](ch14.html "Chapter 14. Dialogs, Controls, and Page Setup"), *Dialogs, Controls, and Print Setup*, describes the possibility to design custom dialogs with controls such as push buttons, check boxes, radio buttons, list boxes, combo boxes, and text field. The input of a text field can be converted to any type. Finally, the Print Setup dialog is a custom dialog annotated with suitable controls.

# What you need for this book

First of all, you need to download Visual Studio on your computer. I suggest you download and install Express for Desktop, which is free, and is available at [https://www.visualstudio.com/en-us/products/visual-studio-express-vs.aspx](https://www.visualstudio.com/en-us/products/visual-studio-express-vs.aspx).

Then, there are two ways to install Small Windows:

1.  If you want to follow the chapter structure of this book you can download it from [https://github.com/PacktPublishing/Cpp-Windows-Programming](https://github.com/PacktPublishing/Cpp-Windows-Programming). It is made up by a set of Visual Studio projects holding the applications of this book.
2.  If you want all the code in one Visual Studio solution you can download the C++ Windows Programming solution in the Cpp Windows Programming file.
3.  If you want to write code with Small Windows on your own, you can download the Empty project in the Empty Project file. It is an application holding only the Small Windows source code with a very simple application. You can change the name of the project and add your own application-specific code.

# Who this book is for

This book is for application developers who want a head-first approach into Windows programming. It will teach you how to develop an object-oriented class library in C++ and enhanced applications in Windows. Basic knowledge of C++ and the object-oriented framework is assumed to get the most out of this book.

# Conventions

In this book, you will find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "The first part of a Small Windows application is the `MainWindow` function."

A block of code is set as follows:

[PRE0]

**New terms** and **important words** are shown in bold. Words that you see on the screen, for example, in menus or dialog boxes, appear in the text like this: "For instance, often, the **Open** item in the **File** menu is annotated with the text **Ctrl+O**."

### Note

Warnings or important notes appear in a box like this.

### Tip

Tips and tricks appear like this.

# Reader feedback

Feedback from our readers is always welcome. Let us know what you think about this book-what you liked or disliked. Reader feedback is important for us as it helps us develop titles that you will really get the most out of. To send us general feedback, simply e-mail feedback@packtpub.com, and mention the book's title in the subject of your message. If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, see our author guide at [www.packtpub.com/authors](http://www.packtpub.com/authors).

# Customer support

Now that you are the proud owner of a Packt book, we have a number of things to help you to get the most from your purchase.

## Downloading the example code

You can download the example code files for this book from your account at [http://www.packtpub.com](http://www.packtpub.com). If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files e-mailed directly to you.

You can download the code files by following these steps:

1.  Log in or register to our website using your e-mail address and password.
2.  Hover the mouse pointer on the **SUPPORT** tab at the top.
3.  Click on **Code Downloads & Errata**.
4.  Enter the name of the book in the **Search** box.
5.  Select the book for which you're looking to download the code files.
6.  Choose from the drop-down menu where you purchased this book from.
7.  Click on **Code Download**.

Once the file is downloaded, please make sure that you unzip or extract the folder using the latest version of:

*   WinRAR / 7-Zip for Windows
*   Zipeg / iZip / UnRarX for Mac
*   7-Zip / PeaZip for Linux

The code bundle for the book is also hosted on GitHub at [https://github.com/PacktPublishing/Cpp-Windows-Programming](https://github.com/PacktPublishing/Cpp-Windows-Programming). We also have other code bundles from our rich catalog of books and videos available at [https://github.com/PacktPublishing/](https://github.com/PacktPublishing/). Check them out!

## Downloading the color images of this book

We also provide you with a PDF file that has color images of the screenshots/diagrams used in this book. The color images will help you better understand the changes in the output. You can download this file from [https://www.packtpub.com/sites/default/files/downloads/CppWindowsProgramming_ColorImages.pdf](https://www.packtpub.com/sites/default/files/downloads/CppWindowsProgramming_ColorImages.pdf).

## Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books-maybe a mistake in the text or the code-we would be grateful if you could report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the **Errata Submission Form** link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded to our website or added to any list of existing errata under the Errata section of that title.

To view the previously submitted errata, go to [https://www.packtpub.com/books/content/support](https://www.packtpub.com/books/content/support) and enter the name of the book in the search field. The required information will appear under the **Errata** section.

## Piracy

Piracy of copyrighted material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works in any form on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at copyright@packtpub.com with a link to the suspected pirated material.

We appreciate your help in protecting our authors and our ability to bring you valuable content.

## Questions

If you have a problem with any aspect of this book, you can contact us at questions@packtpub.com, and we will do our best to address the problem.