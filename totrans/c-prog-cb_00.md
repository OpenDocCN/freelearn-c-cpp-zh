# Preface

In this book, we will explore all the important elements of C such as strings, arrays, one-and two-dimensional arrays, functions, pointers, file handling, threads, interprocess communication, and database handling. Through the cookbook approach, you will find solutions to different problems that you will usually come across while making applications. By the end of the book, you will have sufficient knowledge to use some low- as well as high-level features of the C language and the ability to apply them for making real-time applications.

# Who this book is for

This book is meant for basic to intermediate programmers and developers who want to make complex and real-time applications in C. This book can be of great use to trainers, teachers, and software developers who get stuck while making applications with arrays, pointers, functions, structures, files, databases, and interprocess communication and wish to see a running examples to find the way out of the problem.

# What this book covers

[Chapter 1](3fd2bfb0-da67-467d-9986-c66c9bdcf736.xhtml), *Working with Arrays*, explains how to do some complex but essential operations with arrays. You will learn how to insert an element in an array, multiply two matrices, find the common elements in two arrays, and also find the difference between two sets or arrays. Also, you will learn how to find the unique elements in an array, how to know whether a matrix is a sparse matrix or not, and how to merge two sorted arrays into one array.

[Chapter 2](d3f17a83-f91b-4831-81b4-107b3eb19092.xhtml), *Managing Strings*, covers manipulating strings at the character level. You will learn how to work out whether the given string is a palindrome or not, how to find the first repetitive character in a string, and how to count each character in a string. You will also learn how to count the number of vowels and consonants in a string and the procedure of converting the vowels in a sentence into uppercase.

[Chapter 3](caff231f-af25-4be5-a3b1-c4db3d65939d.xhtml), *Exploring Functions*, covers functions, which play a major role in breaking down a big application into small, independent, manageable modules. In this chapter, you will learn how to make a function that finds whether the supplied argument is an Armstrong number. You will also learn how a function returns an array, and we make a function that finds the **greatest common divisor** (**GCD**) of two numbers using recursion. You will also learn how to make a function that converts a binary number into hexadecimal and how to make a function that determines whether the supplied number is a palindrome or not.

[Chapter 4](e7568e63-9577-4ec3-b3d0-bc4d4fe9b71d.xhtml), *Deep Dive into Pointers*, explains how to use pointers to access content from specific memory locations. You will learn how to reverse a string using pointers, how to find the largest value in an array using pointers, and how to sort a singly linked list. Besides this, the chapter also explains how to find the transpose of a matrix and how to access a structure using pointers.

[Chapter 5](947d4b9d-0441-43af-86c1-270d385a9fd0.xhtml), *File Handling*, explains that file handling is very important for storing data for later use. In this chapter, you will learn how to read a text file and convert all the characters after full stops into uppercase. You will also learn how to display the content of a random file in reverse order and how to count the number of vowels in a file. This chapter will also explain how to replace a word in a file with another word and how to keep your file secure from unauthorized access. You will also learn how a file is encrypted.

[Chapter 6](6ea0edad-4e5e-402b-8a6d-6cb85954125c.xhtml), *Implementing Concurrency*, covers concurrency, which is implemented to increase the efficiency of the CPU. In this chapter, you will learn how to do a task using a single thread. Also, you will learn how to do multiple tasks with multiple threads and the technique of sharing data on two threads using a `mutex`. Besides this, you will learn how to recognize situations in which a deadlock can arise and how it can be avoided.

[Chapter 7](06fdb2b6-c319-464a-be81-6ea7358164b9.xhtml), *Networking and Interprocess Communication*, focuses on explaining how to establish communication between processes. You will learn how to communicate between processes using pipes, establish communication between processes using FIFO, and how communication is established between the client and server using socket programming. You will also learn how to do interprocess communication using the UDP socket, how a message is passed from one process to another using the message queue, and how two processes communicate using shared memory.

[Chapter 8](afdb55d7-2322-4dce-ad9d-ff737f8c3b4b.xhtml), *Using MySQL Database*, explains that no real-time application is possible without storing information in a database. The information in a database needs to be managed. In this chapter, you will learn how to display all the built-in tables in a default MySQL database. You will learn how to store information in a MySQL database and how to search for information in database tables. You will also learn how to update information in database tables and how to delete data from the database when it's no longer required.

*Appendix A,* explains how to create sequential and random files step by step. Most of the recipes in [Chapter 5](947d4b9d-0441-43af-86c1-270d385a9fd0.xhtml), *File Handling,* are about reading content from a file, and those recipes assume that the file already exists. This chapter explains how to create a sequential file and enter some text in it. You will also learn how to read content from a sequential file. Besides this, you will learn how to create a random file and enter some content in it, and how to read content from a random file and display it on the screen. Finally, you will also learn how to decrypt the content of an encrypted file.

*Appendix B,* explains how to install Cygwin.

*Appendix C,* explains how to install MySQL Community Server.

# To get the most out of this book

You must have some preliminary knowledge of C programming. You will find it beneficial to have some prior basic knowledge of arrays, strings, functions, file handling, threads, and interprocess communication.

In addition, you must have some knowledge of basic SQL commands to handle databases. 

# Download the example code files

You can download the example code files for this book from your account at [www.packt.com](http://www.packt.com). If you purchased this book elsewhere, you can visit [www.packt.com/support](http://www.packt.com/support) and register to have the files emailed directly to you.

You can download the code files by following these steps:

1.  Log in or register at [www.packtpub.com](http://www.packtpub.com/support).
2.  Select the SUPPORT tab.
3.  Click on Code Downloads & Errata.
4.  Enter the name of the book in the Search box and follow the onscreen instructions.

Once the file is downloaded, please make sure that you unzip or extract the folder using the latest version of:

*   WinRAR/7-Zip for Windows
*   Zipeg/iZip/UnRarX for Mac
*   7-Zip/PeaZip for Linux

The code bundle for the book is also hosted on GitHub at [https://github.com/PacktPublishing/C-Programming-Cookbook](https://github.com/PacktPublishing/C-Programming-Cookbook). We also have other code bundles from our rich catalog of books and videos available at **[https://github.com/PacktPublishing/](https://github.com/PacktPublishing/)**. Check them out!

# Download the color images

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. You can download it here: [https://www.packtpub.com/sites/default/files/downloads/9781789617450_ColorImages.pdf](https://www.packtpub.com/sites/default/files/downloads/9781789617450_ColorImages.pdf).

# Conventions used

There are a number of text conventions used throughout this book.

`CodeInText`: Indicates code words in text, database table names, folder names, filenames, file extensions, path names, dummy URLs, user input, and Twitter handles. Here is an example: "In the figure, `1000` represents the memory address of the `i` variable."

A block of code is set as follows:

```cpp
 for(i=0;i<2;i++)
  {
    for(j=0;j<4;j++)
    {
      matR[i][j]=0;
      for(k=0;k<3;k++)
      {
        matR[i][j]=matR[i][j]+matA[i][k]*matB[k][j];
      }
    }
  }
```

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

```cpp
printf("How many elements are there? ");
scanf("%d", &n);
```

Any command-line input or output is written as follows:

```cpp
D:\CBook>reversestring
Enter a string: manish
Reverse string is hsinam
```

**Bold**: Indicates a new term, an important word, or words that you see onscreen. For example, words in menus or dialog boxes appear in the text like this. Here is an example: "Simply click the Next button to continue."

Warnings or important notes appear like this.

Tips and tricks appear like this.

# Sections

In this book, you will find several headings that appear frequently (*How to do it* and *How it works*).

To receive clear instructions on how to complete a recipe, use these sections as follows:

# How to do it…

This section contains the steps required to follow the recipe.

# How it works…

This section consists of a detailed explanation of the steps followed in the previous section.

# There's more…

This section, when present, consists of additional information about the recipe in order to enhance your knowledge about the recipe.

# See also

This section, when present, provides helpful links to other useful information for the recipe.

# Get in touch

Feedback from our readers is always welcome.

**General feedback**: Email `feedback@packtpub.com` and mention the book title in the subject of your message. If you have questions about any aspect of this book, please email us at `questions@packtpub.com`.

**Errata**: Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you have found a mistake in this book, we would be grateful if you would report this to us. Please visit [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), select your book, click on the Errata Submission Form link, and enter the details.

**Piracy**: If you come across any illegal copies of our works in any form on the internet, we would be grateful if you would provide us with the location address or website name. Please contact us at `copyright@packtpub.com` with a link to the material.

**If you are interested in becoming an author**: If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, please visit [authors.packtpub.com](http://authors.packtpub.com/).

# Reviews

Once you have read and used this book, why not leave a review on the site that you purchased it from? Potential readers can then see and use your unbiased opinion to make purchase decisions, we at Packt can understand what you think about our products, and our authors can see your feedback on their book. Thank you!

For more information about Packt, please visit [packtpub.com](https://www.packtpub.com/).