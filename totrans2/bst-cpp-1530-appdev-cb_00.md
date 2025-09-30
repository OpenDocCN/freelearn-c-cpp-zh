# Preface

A few years ago one of my friends was looking for a book about the Boost libraries. I asked him "Why don't you read the documentation?". His answer was, "I do not know much and I do not know where to start. Boost is huge; I have no time to read all about it."

Well, that was a good hint but such a book would be of interest only to beginners. Professionals would find nothing interesting in it unless I added some C++11 stuff and compared the existing Boost libraries with the new C++ standard.

I could also add answers to common questions that arise in Boost mailing lists but are hard to find or not covered by the documentation. Spice it up with performance notes and we'd get a book that would be interesting to almost everyone.

This book will take you through a number of clear, practical recipes that will help you to take advantage of some readily available solutions.

*Boost C++ Application Development Cookbook* starts out teaching the basics of the Boost libraries that are now mostly part of C++11 and leave no chance for memory leaks. Managing resources will become a piece of cake. We'll see what kind of work can be done at compile time and what Boost containers can do. Do you think multithreading is a burden? Not with Boost. Do you think writing portable and fast servers is impossible? You'll be surprised! Compilers and operating systems differ too much? Not with Boost. From manipulating images to graphs, directories, timers, files, and strings – everyone will find an interesting topic.

You will learn everything needed for the development of high-quality, fast, and portable applications. Write a program once and you can use it on Linux, Windows, Mac OS, and Android operating systems.

# What this book covers

[Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*, covers some recipes for everyday use. We'll see how to get configuration options from different sources and what can be cooked up using some of the datatypes introduced by Boost library authors.

[Chapter 2](ch02.html "Chapter 2. Converting Data"), *Converting Data*, explains how to convert strings, numbers, and user-defined types to each other, how to safely cast polymorphic types, and how to write small and large parsers right in C++ source files.

[Chapter 3](ch03.html "Chapter 3. Managing Resources"), *Managing Resources*, provides guidance to easily managing resources and how to use a datatype capable of storing any functional objects, functions, and lambda expressions. After reading this chapter your code will become more reliable and memory leaks will become history.

[Chapter 4](ch04.html "Chapter 4. Compile-time Tricks"), *Compile-time Tricks*, walks you through some basic examples on how Boost libraries can be used in compile-time checking, for tuning algorithms and in other metaprogramming tasks.

[Chapter 5](ch05.html "Chapter 5. Multithreading"), *Multithreading*, discusses threads and everything connected with them.

[Chapter 6](ch06.html "Chapter 6. Manipulating Tasks"), *Manipulating Tasks*, explains that we can split all of the processing, computations, and interactions to functors (tasks) and process each of those tasks almost independently. Moreover, we need not block on some slow operations such as receiving data from socket or waiting for timeout, but instead provide a callback task and continue processing other tasks.

[Chapter 7](ch07.html "Chapter 7. Manipulating Strings"), *Manipulating Strings*, covers different aspects of changing, searching, and representing strings. We'll see how some common string-related tasks can easily be done using Boost libraries.

[Chapter 8](ch08.html "Chapter 8. Metaprogramming"), *Metaprogramming*, is devoted to some cool and hard-to-understand metaprogramming methods. Those methods are not for everyday use, but they will be a real help for development of generic libraries.

[Chapter 9](ch09.html "Chapter 9. Containers"), *Containers*, covers Boost containers and everything directly connected to them. This chapter provides information about Boost classes that can be used in everyday programming and that will make your code much faster and development of new applications easier.

[Chapter 10](ch10.html "Chapter 10. Gathering Platform and Compiler Information"), *Gathering Platform and Compiler Information*, provides different helper macros used to detect compiler, platform, and Boost features. Those macros are widely used across Boost libraries and are essential for writing portable code that is able to work with any compiler flags.

[Chapter 11](ch11.html "Chapter 11. Working with the System"), *Working with the System*, takes a closer look at the filesystem and at creating and deleting files. We'll see how data can be passed between different system processes, how to read files with maximum speed, and how to do other tricks.

[Chapter 12](ch12.html "Chapter 12. Scratching the Tip of the Iceberg"), *Scratching the Tip of the Iceberg*, is devoted to some of those big libraries, giving the basics to start with. Some of the Boost libraries are small and meant for everyday use, others require a separate book to describe all of their features.

# What you need for this book

To run the examples in this book, the following software will be required:

*   **C++ compiler**: Any modern, popular C++ compiler will be suitable
*   **IDE**: QtCreator is recommended as an IDE
*   **Boost**: You should have a full build of Boost 1.53
*   **Miscellaneous tools**: Graphviz (any version) and libpng (latest version)

Note that if you are using Linux, all of the required software except Boost can be found in the repository.

# Who this book is for

This book is great for developers who are new to Boost, and who are looking to improve their knowledge of Boost and see some undocumented details or tricks. It's assumed that you will have some experience in C++ already, as well as being familiar with the basics of STL. A few chapters will require some previous knowledge of multithreading and networking. You are expected to have at least one good C++ compiler and compiled version of Boost (1.53.0 or later is recommended), which will be used during the exercises within this book.

# Conventions

In this book, you will find a number of styles of text that distinguish between different kinds of information. Here are some examples of these styles, and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "It means that you can catch almost all Boost exceptions using `catch (const std::exception& e)`."

A block of code is set as follows:

[PRE0]

**New terms** and **important words** are shown in bold.

### Note

Warnings or important notes appear in a box like this.

### Tip

Tips and tricks appear like this.

# Reader feedback

Feedback from our readers is always welcome. Let us know what you think about this book—what you liked or may have disliked. Reader feedback is important for us to develop titles that you really get the most out of.

To send us general feedback, simply send an e-mail to `<[feedback@packtpub.com](mailto:feedback@packtpub.com)>`, and mention the book title via the subject of your message.

If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, see our author guide on [www.packtpub.com/authors](http://www.packtpub.com/authors).

# Customer support

Now that you are the proud owner of a Packt book, we have a number of things to help you to get the most from your purchase.

## Downloading the example code

You can download the example code files for all Packt books you have purchased from your account at [http://www.packtpub.com](http://www.packtpub.com). If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files e-mailed directly to you.

## Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books—maybe a mistake in the text or the code—we would be grateful if you would report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the **errata submission form** link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded on our website, or added to any list of existing errata, under the Errata section of that title. Any existing errata can be viewed by selecting your title from [http://www.packtpub.com/support](http://www.packtpub.com/support).

## Piracy

Piracy of copyright material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works, in any form, on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at `<[copyright@packtpub.com](mailto:copyright@packtpub.com)>` with a link to the suspected pirated material.

We appreciate your help in protecting our authors, and our ability to bring you valuable content.

## Questions

You can contact us at `<[questions@packtpub.com](mailto:questions@packtpub.com)>` if you are having a problem with any aspect of the book, and we will do our best to address it.