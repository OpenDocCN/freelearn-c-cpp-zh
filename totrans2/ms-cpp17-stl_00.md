# Preface

The C++ language has a long history, dating back to the 1980s. Recently it has undergone a renaissance, with major new features being introduced in 2011 and 2014\. At press time, the C++17 standard is just around the corner.

C++11 practically doubled the size of the standard library, adding such headers as `<tuple>`, `<type_traits>`, and `<regex>`. C++17 doubles the library again, with additions such as `<optional>`, `<any>`, and `<filesystem>`. A programmer who’s been spending time writing code instead of watching the standardization process might fairly feel that the standard library has gotten away from him--that there’s so many new things in the library that he'll never be able to master the whole thing, or even to sort the wheat from the chaff. After all, who wants to spend a month reading technical documentation on `std::locale` and `std::ratio`, just to find out that they aren't useful in your daily work?

In this book, I'll teach you the most important features of the C++17 standard library. In the interest of brevity, I omit some parts, such as the aforementioned `<type_traits>`; but we'll cover the entire modern STL (every standard container and every standard algorithm), plus such important topics as smart pointers, random numbers, regular expressions, and the new-in-C++17 `<filesystem>` library.

I'll teach by example. You'll learn to build your own iterator type; your own memory allocator using `std::pmr::memory_resource`; your own thread pool using `std::future`.

I'll teach concepts beyond what you'd find in a reference manual. You'll learn the difference between monomorphic, polymorphic, and generic algorithms ([Chapter 1](part0021.html#K0RQ0-2fdac365b8984feebddfbb9250eaf20d), *Classical Polymorphism and Generic Programming*); what it means for `std::string` or `std::any` to be termed a "vocabulary type" ([Chapter 5](part0074.html#26I9K0-2fdac365b8984feebddfbb9250eaf20d), *Vocabulary Types*); and what we might expect from future C++ standards in 2020 and beyond.

I assume that you are already reasonably familiar with the core language of C++11; for example, that you already understand how to write class and function templates, the difference between lvalue and rvalue references, and so on.

# What this book covers

[Chapter 1](part0021.html#K0RQ0-2fdac365b8984feebddfbb9250eaf20d), *Classical Polymorphism and Generic Programming*, covers classical polymorphism (virtual member functions) and generic programming (templates).

[Chapter 2](part0026.html#OPEK0-2fdac365b8984feebddfbb9250eaf20d), *Iterators and Ranges*, explains the concept of *iterator* as a generalization of pointer, and the utility of half-open ranges expressed as a pair of iterators.

[Chapter 3](part0036.html#12AK80-2fdac365b8984feebddfbb9250eaf20d), *The Iterator-Pair Algorithms*, explores the vast variety of standard generic algorithms that operate on ranges expressed as iterator-pairs.

[Chapter 4](part0052.html#1HIT80-2fdac365b8984feebddfbb9250eaf20d), *The Container Zoo*, explores the almost equally vast variety of standard container class templates, and which containers are suitable for which jobs.

[Chapter 5](part0074.html#26I9K0-2fdac365b8984feebddfbb9250eaf20d), *Vocabulary Types*, walks you through algebraic types such as `std::optional`. and ABI-friendly type-erased types such as `std::function`.

[Chapter 6](part0093.html#2OM4A0-2fdac365b8984feebddfbb9250eaf20d), *Smart Pointers*, teaches the purpose and use of smart pointers.

[Chapter 7](part0108.html#36VSO0-2fdac365b8984feebddfbb9250eaf20d), *Concurrency*, covers atomics, mutexes, condition variables, threads, futures, and promises.

[Chapter 8](part0129.html#3R0OI0-2fdac365b8984feebddfbb9250eaf20d), *Allocators*, explains the new features of C++17's `<memory_resource>` header.

[Chapter 9](part0144.html#49AH00-2fdac365b8984feebddfbb9250eaf20d), *Iostreams*, explores the evolution of the C++ I/O model, from `<unistd.h>` to `<stdio.h>` to `<iostream>`.

[Chapter 10](part0161.html#4PHAI0-2fdac365b8984feebddfbb9250eaf20d), *Regular Expressions*, teaches regular expressions in C++.

[Chapter 11](part0174.html#55U1S0-2fdac365b8984feebddfbb9250eaf20d), *Random Numbers*, walks you through C++'s support for pseudo-random number generation.

[Chapter 12](part0188.html#5J99O0-2fdac365b8984feebddfbb9250eaf20d), *Filesystem*, covers the new-in-C++17 `<filesystem>` library.

# What you need for this book

As this book is not a reference manual, it might be useful for you to have a reference manual, such as cppreference ([en.cppreference.com/w/cpp](https://en.cppreference.com/w/cpp)), at your side to clarify any confusing points. It will definitely help to have a C++17 compiler handy. At press time, there are several more or less feature-complete C++17 implementations, including GCC, Clang, and Microsoft Visual Studio. You can run them locally or via many free online compiler services, such as Wandbox (wandbox.org), Godbolt (gcc.godbolt.org), and Rextester (rextester.com).

# Who this book is for

This book is for developers who would like to master the C++17 STL and make full use of its components. Prior C++ knowledge is assumed.

# Conventions

In this book, you will find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "The `buffer()` function accepts arguments of type `int`."

A block of code is set as follows:

[PRE0]

**New terms** and **important words** are shown in bold.

Warnings or important notes appear like this.

Tips and notes appear like this.

# Reader feedback

Feedback from our readers is always welcome. Let us know what you think about this book--what you liked or disliked. Reader feedback is important for us as it helps us develop titles that you will really get the most out of. To send us general feedback, simply e-mail `feedback@packtpub.com`, and mention the book's title in the subject of your message. If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, see our author guide at [www.packtpub.com/authors](http://www.packtpub.com/authors).

# Customer support

Now that you are the proud owner of a Packt book, we have a number of things to help you to get the most from your purchase.

# Downloading the example code

You can download the example code files for this book from your account at [http://www.packtpub.com](http://www.packtpub.com). If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files e-mailed directly to you. You can download the code files by following these steps:

1.  Log in or register to our website using your e-mail address and password.
2.  Hover the mouse pointer on the SUPPORT tab at the top.
3.  Click on Code Downloads & Errata.
4.  Enter the name of the book in the Search box.
5.  Select the book for which you're looking to download the code files.
6.  Choose from the drop-down menu where you purchased this book from.
7.  Click on Code Download.

Once the file is downloaded, please make sure that you unzip or extract the folder using the latest version of:

*   WinRAR / 7-Zip for Windows
*   Zipeg / iZip / UnRarX for Mac
*   7-Zip / PeaZip for Linux

The code bundle for the book is also hosted on GitHub at [https://github.com/PacktPublishing/Mastering-the-Cpp17-STL](https://github.com/PacktPublishing/Mastering-the-Cpp17-STL). We also have other code bundles from our rich catalog of books and videos available at [https://github.com/PacktPublishing/](https://github.com/PacktPublishing/). Check them out!

# Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books--maybe a mistake in the text or the code--we would be grateful if you could report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the Errata Submission Form link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded to our website or added to any list of existing errata under the Errata section of that title. To view the previously submitted errata, go to [https://www.packtpub.com/books/content/support](https://www.packtpub.com/books/content/support) and enter the name of the book in the search field. The required information will appear under the Errata section.

# Piracy

Piracy of copyrighted material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works in any form on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy. Please contact us at `copyright@packtpub.com` with a link to the suspected pirated material. We appreciate your help in protecting our authors and our ability to bring you valuable content.

# Questions

If you have a problem with any aspect of this book, you can contact us at `questions@packtpub.com`, and we will do our best to address the problem.