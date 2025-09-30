# Preface

In today's information-centric globalized world, telecommunications have become an essential part of our lives. They penetrate and play crucial roles in almost every aspect of our day-to-day activities, from personal to professional. Sometimes, a failure to communicate information correctly and on time may lead to significant loss of material assets or even casualties.

Therefore, it is very important to provide the highest level of reliability when it comes to developing telecommunication software. However, it can be a really challenging task due to the inherent complexity of the domain and accidental complexity of the low-level tools provided by modern operating systems.

The Boost.Asio library is aimed at reducing accidental complexity by introducing type systems and exploiting object-oriented methods, and decreasing the development time by providing high degrees of reusability. In addition to this, because the library is cross-platform, the applications implemented with it can be built on multiple platforms, which enhances software qualities even more, while decreasing its costs.

This book contains more than 30 recipes—step-by-step solutions to various tasks that often (and not so often) arise in the area of network programming. All recipes take advantage of facilities provided by the Boost.Asio library, demonstrating best practices of applying the library to execute typical tasks and solve different problems.

# **What this book covers**

[Chapter 1](ch01.html "Chapter 1. The Basics"), *The Basics*, introduces you to basic classes provided by the Boost.Asio library and demonstrates how to execute basic operations, such as resolving a DNS name, connecting a socket, accepting a connection, and so on.

[Chapter 2](ch02.html "Chapter 2. I/O Operations"), *I/O Operations*, demonstrates how to perform individual network I/O operations, both synchronous and asynchronous.

[Chapter 3](ch03.html "Chapter 3. Implementing Client Applications"), *Implementing Client Applications*, contains recipes that demonstrate how to implement different types of client applications.

[Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*, contains recipes that demonstrate how to implement different types of server applications.

[Chapter 5](ch05.html "Chapter 5. HTTP and SSL/TLS"), *HTTP and SSL/TLS*, covers more advanced topics on the HTTP and SSL/TLS protocol implementation.

[Chapter 6](ch06.html "Chapter 6. Other Topics"), *Other Topics*, includes recipes that discuss less popular but still quite important topics, such as timers, socket options, composite buffers, and others.

# What you need for this book

To compile samples presented in this book, you will need Visual Studio 2012+ on Windows or GCC 4.7+ on Unix platforms.

# Who this book is for

If you want to enhance your C++ network programming skills using the Boost.Asio library and understand the theory behind the development of distributed applications, this book is just what you need. The prerequisite for this book is to have a basic knowledge of C++11\. To get the most from the book and comprehend advanced topics, you will need some background experience in multithreading.

# Sections

In this book, you will find several headings that appear frequently (Getting ready, How to do it, How it works, There's more, and See also).

To give clear instructions on how to complete a recipe, we use these sections as follows:

## Getting ready

This section tells you what to expect in the recipe, and describes how to set up any software or any preliminary settings required for the recipe.

## How to do it…

This section contains the steps required to follow the recipe.

## How it works…

This section usually consists of a detailed explanation of what happened in the previous section.

## There's more…

This section consists of additional information about the recipe in order to make the reader more knowledgeable about the recipe.

## See also

This section provides helpful links to other useful information for the recipe.

# Conventions

In this book, you will find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "In Boost.Asio a passive socket is represented by the `asio::ip::tcp::acceptor` class."

A block of code is set as follows:

[PRE0]

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

[PRE1]

**New terms** and **important words** are shown in bold.

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