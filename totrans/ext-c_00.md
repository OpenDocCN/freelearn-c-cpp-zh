# Preface

In this modern era, we observe mind-blowing technologies on a regular basis and experience luxury and pleasure far beyond what could have been imagined even a few decades ago. We find ourselves on the cusp of autonomous cars becoming a reality on our streets. Advances in physics and other branches of science are changing the way we perceive reality itself. We read news about researchers taking baby steps in quantum computation, rumors about blockchain technology and cryptocurrencies, and plans to colonize other planets. Unbelievably, such a diversity of breakthroughs find their roots in just a few core technologies. This book is about one of those technologies: C.

I started programming with C++ when I was studying in my first year of high school. There, I joined a 2D soccer simulation team for juniors. Soon after C++, I got introduced to Linux and C. I must admit that I didn't know much about the importance of C and Unix in those years, but over time, as I gained more experience in using them through various projects, and as I learned about them through my education, I came to see their critical role and position. The more I knew about C, the bigger my respect for it grew. Finally, I decided to be an expert in this programming language that had so captured my interest. I also decided to be an advocate to spread the knowledge and make people aware of the importance of C. This book is a result of that ambition.

Despite the mistaken belief that C is a dead language, and despite the general ignorance that C receives from tech people, the TIOBE index found at [https://www.tiobe.com/tiobe-index](https://www.tiobe.com/tiobe-index/c) demonstrates otherwise. C is, in fact, one of the most popular programming languages of the past 15 years, together with Java, and has gained popularity in recent years.

I come to this book following many years of experience in development and design using C, C++, Golang, Java, and Python on various platforms including various BSD Unix flavors, Linux, and Microsoft Windows. The primary aim of this book is to increase the skill of its audience to the next level; to enable them to take the next step in their use of C, and practically apply it in a way that comes through hard-earned experience. This won't be an easy journey for us and that's why we've called this book *Extreme C*. This journey is the core concern of this book, and we will not be entering into the debate of C versus other programming languages. This book tries to remain practical, but still we have to deal with a significant quantity of hardcore theory that is relevant to practical application. The book is full of examples that aim to prepare you to deal with the things you will encounter within a real system.

It is indeed a great honor to be able to address such a weighty topic. Words won't suffice, so I'll only say that it has been an incredible joy to have the opportunity to write about a topic that is so close to my heart. And I owe this pleasure and astonishment to Andrew Waldron, who let me take on this book as my first experience in book writing.

As part of this, I want to send my special regards and best thanks to Ian Hough, the Development Editor, who was with me chapter by chapter on this journey, to Aliakbar Abbasi for his tireless peer review feedback, and to Kishor Rit, Gaurav Gavas, Veronica Pais, and many more valuable people who have put their best efforts into preparing and publishing this book.

With that said, I invite you to be my companion on this long journey. I hope that the reading of this book will prove to be transformative, helping you to see C in a new light, and to become a better programmer in the process.

# Who this book is for

This book is written for an audience that has a minimum level of knowledge regarding C and C++ development. Junior and intermediate C/C++ engineers are the main audience that can get the most from this book and leverage their expertise and knowledge. Hopefully, after reading this book, they can gain a raise in their position and become senior engineers. In addition, after reading this book, their expertise would be a better match for more relevant job opportunities that are challenging and usually well paid. Some topics can still be useful to senior C/C++ engineers, but it is expected that most of the topics are known to them and only some extra details can still be useful.

The other audience that can benefit from reading this book is students and researchers. Students of bachelor's, master's, or PhD degrees studying in any branch of science or engineering such as computer science, software engineering, artificial intelligence, **Internet of Things** (**IoT**), astronomy, particle physics, and cosmology, as well as all researchers in these fields, can use the book to increase the level of their knowledge about C/C++, Unix-like operating systems, and the relevant programming skills. This book would be good for engineers and scientists working on complex, multithreaded, or even multi-process systems performing remote device controlling, simulations, big data processing, machine learning, deep learning, and so on.

# What this book covers

This book has 7 parts. In each of these 7 parts, we cover some particular aspects of C programming. The first part focuses upon how to build a C project, the second part focuses on memory, the third on object orientation, and the fourth primarily looks at Unix and its relationship to C. The fifth part then discusses concurrency, the sixth covers inter-process communication, and finally the seventh part of the book is about testing and maintenance. Below is a summary of each of the 23 chapters found in this book.

*Chapter 1*, *Essential Features*: This chapter is about the essential features found in C that have a profound effect on the way we use C. We will be using these features often throughout the book. The main topics are preprocessing and directives to define macros, variable and function pointers, function call mechanisms, and structures.

*Chapter 2*, *Compilation and Linking*: As part of this chapter, we discuss how to build a C project. The compilation pipeline is studied in great detail, both in terms of the pipeline as a whole and in terms of the individual pipeline components.

*Chapter 3*, *Object Files*: This chapter looks at the products of a C project after having built it using the compilation pipeline. We introduce object files and their various types. We also take a look inside these object files and see what information can be extracted.

*Chapter 4*, *Process Memory Structure*: In this chapter, we explore a process's memory layout. We see what segments can be found in this memory layout and what static and dynamic memory layouts mean.

*Chapter 5*, *Stack and Heap*: As part of this chapter, we discuss the Stack and Heap segments specifically. We talk about the Stack and Heap variables and how their lifetime is managed in C. We discuss some best practice regarding Heap variables and the way that they should be managed.

*Chapter 6*, *OOP and Encapsulation*: This is the first chapter in a group of four chapters discussing object orientation in C. As part of this chapter, we go through the theory behind object orientation and we give important definitions to the terms often used in the literature.

*Chapter 7*, *Composition and Aggregation*: This chapter focuses upon composition and a special form of it: aggregation. We discuss the differences between composition and aggregation and give examples to demonstrate these differences.

*Chapter 8*, *Inheritance and Polymorphism*: Inheritance is one of the most important topics in **object-oriented programming** (**OOP**). In this chapter, we show how an inheritance relationship can be established between two classes and how it can be done in C. Polymorphism is another big topic that is discussed as part of this chapter.

*Chapter 9*, *Abstraction and OOP in C++*: As the final chapter in the third part of the book, we talk about abstraction. We discuss abstract data types and how they can be implemented in C. We discuss the internals of C++ and we demonstrate how object-oriented concepts are implemented in C++.

*Chapter 10*, *Unix – History and Architecture*: You cannot talk about C and forget about Unix. In this chapter, we describe why they are strongly bound to each other, and how Unix and C have helped one another to survive thus far. The architecture of Unix is also studied, and we see how a program uses the functionalities exposed by the operating system.

*Chapter 11*, *System Calls and Kernel*: In this chapter, we focus on the kernel ring in the Unix architecture. We discuss system calls in greater detail and we add a new system call to Linux. We also talk about various types of kernels, and we write a new simple kernel module for Linux to demonstrate how kernel modules work.

*Chapter 12*, *The Most Recent C*: As part of this chapter, we take a look at the most recent version of C standard, C18\. We see how it is different from the previous version, C11\. We also demonstrate some of the newly added features in comparison to C99.

*Chapter 13*, *Concurrency*: This is the first chapter of the fifth part of the book, and it is regarding concurrency. This chapter mainly talks about concurrent environments and their various properties such as interleavings. We explain why these systems are non-determinant and how this property can lead to concurrency issues such as race conditions.

*Chapter 14*, *Synchronization*: In this chapter, we continue our discussion regarding concurrent environments, and we discuss the various types of issues that we can expect to observe in a concurrent system. Race conditions, data races, and deadlocks are among the issues that we discuss. We also talk about the techniques that we can utilize to overcome these issues. Semaphores, mutexes, and condition variables are discussed in this chapter.

*Chapter 15*, *Thread Execution*: As part of this chapter, we demonstrate how a number of threads can be executed and how they can be managed. We also give real C examples about the concurrency issues discussed in the previous chapter.

*Chapter 16*, *Thread Synchronization*: In this chapter, we look at the techniques that we can use to synchronize a number of threads. Semaphores, mutexes, and condition variables are among the notable topics that are discussed and demonstrated in this chapter.

*Chapter 17*, *Process Execution*: This chapter talks about the ways that we can create or spawn a new process. We also discuss push-based and pull-based techniques for sharing state among a number of processes. We also demonstrate the concurrency issues discussed in *Chapter 14*, *Synchronization* using real C examples.

*Chapter 18*, *Process Synchronization*: This chapter mainly deals with available mechanisms to synchronize a number of processes residing on the same machine. Process-shared semaphores, process-shared mutexes, and process-shared condition variables are among the techniques discussed in this chapter.

*Chapter 19*, *Single-Host IPC and Sockets*: In this chapter, we mainly discuss push-based **interprocess communication** (**IPC**) techniques. Our focus is on the techniques available to processes residing on the same machine. We also introduce socket programming, and the required background to establish channels between processes residing on different nodes in a network.

*Chapter 20*, *Socket Programming*: As part of this chapter, we discuss socket programming through code examples. We drive our discussion by bringing up an example that is going to support various types of sockets. Unix domain sockets, TCP, and UDP sockets operating on either a stream or a datagram channel are discussed.

*Chapter 21*, *Integration with Other Languages*: In this chapter, we demonstrate how a C library, built as a shared object file, can be loaded and used in programs written with C++, Java, Python, and Golang.

*Chapter 22*, *Unit Testing and Debugging*: This chapter is dedicated to testing and debugging. For the testing half, we explain various levels of testing, but we focus on unit testing in C. We also introduce CMocka and Google Test as two available libraries to write test suites in C. For the debugging half, we go through various available tools that can be used for debugging different types of bugs.

*Chapter 23*, *Build Systems*: In the final chapter of the book, we discuss build systems and build script generators. Make, Ninja, and Bazel are the build systems that we explain as part of this chapter. CMake is also the sole build script generator that we discuss in this chapter.

# To get the most out of this book

As we have explained before, this book requires you to have a minimum level of knowledge and skill regarding computer programming. The minimum requirements are listed below:

*   General knowledge of computer architecture: You should know about memory, CPU, peripheral devices and their characteristics, and how a program interacts with these elements in a computer system.
*   General knowledge of programming: You should know what an algorithm is, how its execution can be traced, what source code is, what binary numbers are, and how their related arithmetic works.
*   Familiarity with using the *Terminal* and the basic *shell commands* in a Unix-like operating system such as Linux or macOS.
*   Intermediate knowledge about programming topics such as conditional statements, different kinds of loops, structures or classes in at least one programming language, pointers in C or C++, functions, and so on.
*   Basic knowledge about OOP: This is not mandatory because we will explain OOP in detail, but it can help you to have a better understanding while reading the chapters in third part of the book, *Object Orientation*.

In addition, it is strongly recommended to download the code repository and follow the commands given in the shell boxes. Please use a platform with Linux or macOS installed. Other POSIX-compliant operating systems can still be used.

## Download the example code files

You can download the example code files for this book from your account at [www.packt.com/](http://www.packt.com/). If you purchased this book elsewhere, you can visit [www.packtpub.com/support](https://www.packtpub.com/support) and register to have the files emailed directly to you.

You can download the code files by following these steps:

1.  Log in or register at [http://www.packt.com](http://www.packt.com).
2.  Select the **Support** tab.
3.  Click on **Code Downloads**.

1.  Enter the name of the book in the **Search** box and follow the on-screen instructions.

Once the file is downloaded, please make sure that you unzip or extract the folder using the latest version of:

*   WinRAR / 7-Zip for Windows
*   Zipeg / iZip / UnRarX for Mac
*   7-Zip / PeaZip for Linux

The code bundle for the book is also hosted on GitHub at [https://github.com/PacktPublishing/Extreme-C](https://github.com/PacktPublishing/Extreme-C). In case there's an update to the code, it will be updated on the existing GitHub repository.

We also have other code bundles from our rich catalog of books and videos available at [https://github.com/PacktPublishing/](https://github.com/PacktPublishing/). Check them out!

## Conventions used

In this book, we have used code boxes and shell boxes. Code boxes contain a piece of either C code or pseudo-code. If the content of a code box is brought from a code file, the name of the code file is shown beneath the box. Below, you can see an example of a code box:

```cpp
#include <stdio.h>
#include <unistd.h>
int main(int argc, char** argv) {
  printf("This is the parent process with process ID: %d\n",
          getpid());
  printf("Before calling fork() ...\n");
  pid_t ret = fork();
  if (ret) {
    printf("The child process is spawned with PID: %d\n", ret);
  } else {
    printf("This is the child process with PID: %d\n", getpid());
  }
  printf("Type CTRL+C to exit ...\n");
  while (1);
  return 0;
}
```

Code Box 17-1 [ExtremeC_examples_chapter17_1.c]: Creating a child process using fork API

As you can see, the above code can be found in the `ExtremeC_examples_chapter17_1.c` file, as part of the code bundle of the book, in the directory of *Chapter 17, Process Execution*. You can get the code bundle from GitHub at [https://github.com/PacktPublishing/Extreme-C](https://github.com/PacktPublishing/Extreme-C).

If a code box doesn't have an associated filename, then it contains pseudo-code or C code that cannot be found in the code bundle. An example is given below:

```cpp
Task P {
    1\. num = 5
    2\. num++
    3\. num = num – 2
    4\. x = 10
    5\. num = num + x
}
```

Code Box 13-1: A simple task with 5 instructions

There can sometimes be some lines shown in bold font within code boxes. These lines are usually the lines of code that are discussed before or after the code box. They are in bold font in order to help you find them more easily.

Shell boxes are used to show the output of the Terminal while running a number of shell commands. The commands are usually in bold font and the output is in the normal font. An example is shown below:

```cpp
$ ls /dev/shm
shm0
$ gcc ExtremeC_examples_chapter17_5.c -lrt -o ex17_5.out
$ ./ex17_5.out
Shared memory is opened with fd: 3
The contents of the shared memory object: ABC
$ ls /dev/shm
$
```

Shell Box 17-6: Reading from the shared memory object created in example 17.4 and finally removing it

The commands start either with `$` or `#`. The commands starting with `$` should be run with a normal user, and the commands starting with `#` should be run with a super user.

The working directory of a shell box is usually the chapter directory found in the code bundle. In cases when a particular directory should be chosen as the working directory, we give you the necessary information regarding that.

**Bold**: Indicates a new term, an important word. Words that you see on the screen, for example, in menus or dialog boxes, also appear in the text like this. For example: "Select **System info** from the **Administration** panel."

Warnings or important notes appear like this.

Tips and tricks appear like this.

# Get in touch

Feedback from our readers is always welcome.

**General feedback**: If you have questions about any aspect of this book, mention the book title in the subject of your message and email us at `customercare@packtpub.com`.

**Errata**: Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you have found a mistake in this book we would be grateful if you would report this to us.

Please visit, [www.packtpub.com/support/errata](https://www.packtpub.com/support/errata), selecting your book, clicking on the Errata Submission Form link, and entering the details.

**Piracy**: If you come across any illegal copies of our works in any form on the Internet, we would be grateful if you would provide us with the location address or website name. Please contact us at `copyright@packt.com` with a link to the material.

**If you are interested in becoming an author**: If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, please visit [authors.packtpub.com](http://authors.packtpub.com/).

## Reviews

Please leave a review. Once you have read and used this book, why not leave a review on the site that you purchased it from? Potential readers can then see and use your unbiased opinion to make purchase decisions, we at Packt can understand what you think about our products, and our authors can see your feedback on their book. Thank you!

For more information about Packt, please visit packt.com.