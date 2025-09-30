# Chapter 10

# Unix – History and Architecture

You might have asked yourself why there should be a chapter about Unix in the middle of a book about expert-level C. If you have not, I invite you to ask yourself, how can these two topics, C and Unix, be related in such a way that there's a need for two dedicated chapters (this and the next chapter) in the middle of a book that should talk about C?

The answer is simple: if you think they are unrelated, then you are making a big mistake. The relationship between the two is simple; Unix is the first operating system that is implemented with a fairly high-level programming language, C, which is designed for this purpose, and C got its fame and power from Unix. Of course, our statement about C being a high-level programming language is not true anymore, and C is no longer considered to be so high-level.

Back in the 1970s and 1980s, if the Unix engineers at Bell Labs had decided to use another programming language, instead of C, to develop a new version of Unix, then we would be talking about that language today, and this book wouldn't be *Extreme C* anymore. Let's pause for a minute to read this quote from Dennis M. Ritchie, one of the pioneers of C, about the effect of Unix on the success of C:

> "Doubtless, the success of Unix itself was the most important factor; it made the language available to hundreds of thousands of people. Conversely, of course, Unix's use of C and its consequent portability to a wide variety of machines was important in the system's success."

> - Dennis M. Ritchie – The Development of the C Language

Available at [https://www.bell-labs.com/usr/dmr/www/chist.html](https://www.bell-labs.com/usr/dmr/www/chist.html).

As part of this chapter, we cover the following topics:

*   We briefly talk about the history of Unix and how the invention of C happened.
*   We explain how C has been developed based on B and BCPL.
*   We discuss the Unix onion architecture and how it was designed based on the Unix philosophy.
*   We describe the user application layer together with shell ring and how the programs consume the API exposed by the shell ring. The SUS and POSIX standards are explained as part of this section.
*   We discuss the kernel layer and what features and functionalities should be present in a Unix kernel.
*   We talk about the Unix devices and how they can be used in a Unix system.

Let's start the chapter by talking about the Unix history.

# Unix history

In this section, we are going to give a bit of history about Unix. This is not a history book, so we're going to keep it short and straight to the point, but the goal here is to gain some hints of history in order to develop a basis for having Unix side by side with C forever in your minds.

## Multics OS and Unix

Even before having Unix, we had the Multics OS. It was a joint project launched in 1964 as a cooperative project led by MIT, General Electric, and Bell Labs. Multics OS was a huge success because it could introduce the world to a real working and secure operating system. Multics was installed everywhere from universities to government sites. Fast-forward to 2019, and every operating system today is borrowing some ideas from Multics indirectly through Unix.

In 1969, because of the various reasons that we will talk about shortly, some people at Bell Labs, especially the pioneers of Unix, such as Ken Thompson and Dennis Ritchie, gave up on Multics and, subsequently, Bell Labs quit the Multics project. But this was not the end for Bell Labs; they had designed their simpler and more efficient operating system, which was called Unix.

You can read more about Multics and its history her[e: https://multicians.org/history.](https://multicians.org/history.html)html, where you can get a breakdown of the history of Multics.

The following link: [https://www.quora.com/Why-did-Unix-succeed-and-not-Multics](https://www.quora.com/Why-did-Unix-succeed-and-not-Multics), is also a good one that explains why Unix continued to live while Multics became discontinued.

It is worthwhile to compare the Multics and Unix operating systems. In the following list, you will see similarities and differences found while comparing Multics and Unix:

*   **Both follow the onion architecture as their internal structure**. We mean that they both have more or less the same rings in their onion architecture, especially kernel and shell rings. Therefore, programmers could write their own programs on top of the shell ring. Also, Unix and Multics expose a list of utility programs such as `ls` and `pwd`. In the following sections, we will explain the various rings found in the Unix architecture.
*   **Multics needed expensive resources and machines to be able to work**. It was not possible to install it on ordinary commodity machines, and that was one of the main drawbacks that let Unix thrive and finally made Multics obsolete after about 30 years.
*   **Multics was complex by design**. This was the reason behind the frustration of Bell Labs employees and, as we said earlier, the reason why they left the project. But Unix tried to remain simple. In the first version, it was not even multitasking or multi-user!

You can read more about Unix and Multics online, and follow the events that happened in that era. Both were successful projects, but Unix has been able to thrive and survive to this day.

It is worth sharing that Bell Labs has been working on a new distributed operating system called *Plan 9*, which is based on the Unix project. You can read more about it at Wik[ipedia: https://en.wikipedia.org/wiki/Plan_9_from_B](https://en.wikipedia.org/wiki/Plan_9_from_Bell_Labs)ell_Labs.

![](img/B11046_10_01.png)

Figure 10-1: Plan 9 from Bell Labs (from Wikipedia)

I suppose that it is enough for us to know that Unix was a simplification of the ideas and innovations that Multics presented; it was not something new, and so, I can quit talking about Unix and Multics history at this point.

So far, there are no traces of C in the history because it has not been invented yet. The first versions of Unix were purely written using assembly language. Only in 1973 was Unix version 4 written using C.

Now, we are getting close to discussing C itself, but before that, we must talk about BCPL and B because they have been the gateway to C.

## BCPL and B

BCPL was created by Martin Richards as a programming language invented for the purpose of writing compilers. The people from Bell Labs were introduced to the language when they were working as part of the Multics project. After quitting the Multics project, Bell Labs first started to write Unix using assembly programming language. That's because, back then, it was an anti-pattern to develop an operating system using a programming language other than assembly!

For instance, it was strange that the people at the Multics project were using PL/1 to develop Multics but, by doing that, they showed that operating systems could be successfully written using a higher-level programming language other than assembly. So, because of that, Multics became the main inspiration for using another language for developing Unix.

The attempt to write operating system modules using a programming language other than assembly remained with Ken Thompson and Dennis Ritchie at Bell Labs. They tried to use BCPL, but it turned out that they needed to apply some modifications to the language to be able to use it in minicomputers such as the DEC PDP-7\. These changes led to the B programming language.

We are going to avoid going too deep into the properties of the B language here, but you can read more about it and the way it was developed in the following links:

*   The B Programmin[g Language, at https://en.wikipedia.org/wiki/B_(progra](https://en.wikipedia.org/wiki/B_(programming_language))mming_language)
*   *The Development of the* [*C Language*, at https://www.bell-labs.com/usr/dmr](https://www.bell-labs.com/usr/dmr/www/chist.html)/www/chist.html

Dennis Ritchie authored the latter article himself, and it is a good way to explain the development of the C programming language while still sharing valuable information about B and its characteristics.

B also had its shortcomings in terms of being a system programming language. B was typeless, which meant that it was only possible to work with a *word* (not a byte) in each operation. This made it hard to use the language on machines with a different word length.

This is why, over time, further modifications were made to the language until it led to developing the **NB** (**New B**) language, and later it derived the structures from the B language. These structures were typeless in B, but they became typed in C. Finally, in 1973, the fourth version of Unix could be developed using C, in which there were still many assembly codes.

In the next section, we talk about the differences between B and C, and why C is a top-notch modern system programming language for writing an operating system.

## The way to C

I do not think we can find anyone better than Dennis Ritchie himself to explain why C was invented after the difficulties met with B. In this section, we're going to list the causes that made Dennis Ritchie, Ken Thompson, and others create a new programming language instead of using B for writing Unix.

Following is the list of flaws found in B, which led to the creation of C:

*   **B could only work with words in memory**: Every single operation should have been performed in terms of words. Back then, having a programming language that was able to work with bytes was a dream. This was because of the available hardware at the time, which was addressing the memory in a word-based scheme.
*   **B was typeless**: The more accurate statement we could say is that B was a single-type language. All variables were from the same type: word. So, if you had a string with 20 characters (21 plus the null character at the end), you had to divide it up by words and store it in more than one variable. For example, if a word was 4 bytes, you would have 6 variables to store 21 characters of the string.
*   **Being typeless meant that multiple byte-oriented algorithms, such as string manipulation algorithms, were not efficiently written with B**: This was because B was using the memory words not bytes, and they could not be used efficiently to manage multi-byte data types such as integers and strings.
*   **B was not supporting floating-point operations**: At the time, these were becoming increasingly available on the new hardware, but there was no support for that in the B language.
*   Through the availability of machines such as PDP-1, which were able to address memory on a byte basis, B showed that it could be inefficient in addressing bytes of memory: This became even clearer with B pointers, which could only address the words in the memory, and not the bytes. In other words, for a program wanting to access a specific byte or a byte range in the memory, more computations had to be done to calculate the corresponding word indexes.

The difficulties with B, particularly its slow development and execution on machines that were available at the time, forced Dennis Ritchie to produce a new language. This new language was called NB, or New B at first, but it eventually turned out to be C.

This newly developed language, C, tried to cover the difficulties and flaws of B and became a *de facto* programming language for system development, instead of the assembly language. In less than 10 years, newer versions of Unix were completely written in C, and all newer operating systems that were based on Unix got tied with C and its crucial presence in the system.

As you can see, C was not born as an ordinary programming language, but instead, it was designed by having a complete set of requirements in mind and, nowadays, it has no competitor. You may consider languages such as Java, Python, and Ruby to be higher-level languages, but they cannot be considered as direct competitors as they are different and serve different purposes. For instance, you cannot write a device driver or a kernel module with Java or Python, and they themselves have been built on top of a layer written in C.

Unlike many programming languages, C is standardized by ISO, and if it is required to have a certain feature in the future, then the standard can be modified to support the new feature.

In the next section, we'll discuss Unix architecture. This is a fundamental concept in understanding how a program evolves within the Unix environment.

# Unix architecture

In this section, we are going to explore the philosophy that the Unix creators had in mind and what they were expecting it to be when they created the architecture.

As we've explained in the previous section, the people involved in Unix from Bell Labs were working for the Multics project. Multics was a big project, the proposed architecture was complex, and it was tuned to be used on expensive hardware. But we should remember that despite all the difficulties, Multics had big goals. The ideas behind the Multics project revolutionized the way we were thinking about the operating systems.

Despite the challenges and difficulties discussed previously the ideas presented in the project were successful because Multics managed to live for around 40 years, until the year 2000\. Not only that, but the project created a huge revenue stream for its owner company.

People such as Ken Thompson and his colleagues brought ideas into Unix even though Unix was, initially, supposed to be simple. Both Multics and Unix tried to bring in similar architecture, but they had two vastly different fates. Multics, since the turn of the century, has started to be forgotten, while Unix, and the operating system families based on it such as BSD, have been growing since then.

We're going to move on to talk about the Unix philosophy. It is simply a set of high-level requirements that the design of Unix is based on. After that, we're going to talk about the Unix multi-ring, onion-like architecture and the role of each ring in the overall behavior of the system.

## Philosophy

The philosophy of Unix has been explained several times by its founders. As such, a thorough breakdown of the entire topic is beyond the scope of this book. What we will do is summarize all of the main viewpoints.

Before we do that, I've listed below some great external literature that could help you on the subject of Unix philosophy:

*   Wikipedia, *Unix philosophy*: [https://en.wikipedia.org/wiki/Unix_philosophy](https://en.wikipedia.org/wiki/Unix_philosophy )
*   *The Unix Philosophy: A Brief Introduction*: [http://www.linfo.org/unix_philosophy.html](http://www.linfo.org/unix_philosophy.html)
*   Eric Steven Raymond, *The Art of Unix Programming*: [https://homepage.cs.uri.edu/~thenry/resources/unix_art/ch01s06.html](https://homepage.cs.uri.edu/~thenry/resources/unix_art/ch01s06.html)

Equally, in the following link, you'll see a quite angry opposite view to the Unix philosophy. I've included this because it's always great to know both sides since, intrinsically, nothing is perfect:

*   *The Colla*[*pse of UNIX Philosophy*: https://kukuruku.co/post/the-collapse](https://kukuruku.co/post/the-collapse-of-the-unix-philosophy/)-of-the-unix-philosophy/

To summarize these viewpoints, I've grouped the key Unix philosophies as follows:

*   **Unix is mainly designed and developed to be used by programmers and not ordinary end users**: Therefore, many considerations addressing user interface and user experience requirements are not part of the Unix architecture.
*   **A Unix system is made up of many small and simple programs**: Each of them is designed to perform a small and simple task. There are lots of examples of these small and simple programs, such as `ls`, `mkdir`, `ifconfig`, `grep`, and `sed`.
*   **A complex task can be performed by executing a sequence of these small and simple programs in a chain**: It means that essentially more than one program is involved in a big and complex task and that, together, each of the programs could be executed multiple times in order to accomplish the task. A good example of this is to use shell scripts instead of writing a program from scratch. Note that shell scripts are often portable between Unix systems, and Unix encourages programmers to break down their big and complex programs into small and simple programs.
*   **Each small and simple program should be able to pass its output as the input of another program, and this chain should continue**: This way, we can use small programs in a chain that has the potential to perform complex tasks. In this chain, each program can be considered as a transformer that receives the output of the previous program, transforms it based on its logic, and passes it to the next program in the chain. A particularly good example of this is *piping* between Unix commands, which is denoted by a vertical bar, such as `ls -l | grep a.out`.
*   **Unix is very text-oriented**: All configurations are text files, and it has a textual command line. Shell scripts are also text files that use simple grammar to write an algorithm that executes other Unix shell programs.
*   **Unix suggests choosing simplicity over perfection**: For example, if a simple solution is working in most cases, don't design a complicated solution that only works marginally better.
*   **Programs written for a certain Unix-compliant operating system should be easily usable in other Unix systems**: This is mainly satisfied by having a single code base that can be built and executed on various Unix-compliant systems.

The points we've just listed have been extracted and interpreted by different people, but in general, they've been agreed upon as the main principles driving Unix philosophy and, as a result, have shaped the design of Unix.

If you have had experience with a Unix-like operating system, for example, Linux, then you'll be able to align your experience with the preceding statements. As we explained in the previous section regarding the history of Unix, it was supposed to be a simpler version of Multics, with the experiences the Unix founders had with Multics leading them to the preceding philosophies.

But back to the topic of C. You may be asking how C has been contributing to the preceding philosophy? Well, almost all of the essential things reflected in the preceding statements are written in C. In other words, the abovementioned small and simple programs that propel much of Unix are all written in C.

It's often better to show rather than simply tell. So, let's look at an example. The C source code for the ls program in NetBSD [can be found here: http://cvsweb.netbsd.org/bsdweb.cgi/~checkout~](http://cvsweb.netbsd.org/bsdweb.cgi/~checkout~/src/bin/ls/ls.c?rev=1.67)/src/bin/ls/ls.c?rev=1.67\. As you should know, the ls program lists the contents of a directory and does nothing more than that, and this simple logic has been written in C as you can see in the link. But this is not the only contribution of C in Unix. We will explain more about this in future sections while talking about the C Standard Library.

## Unix onion

Now, it is time to explore the Unix architecture. An *onion model*, as we briefly mentioned before, can describe the Unix overall architecture. It is onion-like because it consists of a few *rings*, each of which acts as a wrapper for internal rings.

*Figure 10-2* demonstrates the proposed famous onion model for the Unix architecture:

![](img/B11046_10_02.png)

Figure 10-2: The onion model of Unix architecture

The model looks quite simple at first glance. However, to understand it fully requires you to write a few programs in Unix. Only after that can you understand what each ring is really doing. We're going to try and explain the model as simply as possible in order to develop an initial foundation before moving forward with writing real examples.

Let's explain the onion model from the innermost ring.

At the core of the preceding model is **Hardware**. As we know, the main task of an operating system is to allow the user to interact with and use the hardware. That's why hardware is at the core of the model in *Figure 10-2*. This simply shows us that one of the main goals of Unix is to make hardware available to the programs willing to have access to it. Everything that we've read about the Unix philosophy in the previous section focuses upon delivering these services in the best possible manner.

The ring around the hardware is the **Kernel**. The kernel is the most important part of an operating system. This is because it is the closest layer to the hardware, and it acts as a wrapper to expose the functionalities of the attached hardware directly. Because of this direct access, the kernel has the highest privilege to use the whole available resources in a system. This unlimited access to everything is the best justification for having other rings in the architecture that don't have that unlimited access. In fact, this was behind the separation between the *kernel space* and the *user space*. We discuss this in further detail in this chapter and the following one.

Note that writing the kernel takes most of the effort needed when writing a new Unix-like operating system, and, as you can see, its ring is drawn thicker than the other rings. There are many different units inside a Unix kernel, and each of them are playing a vital role in the Unix ecosystem. Later on in this chapter, we will explain more about the internal structure of a Unix kernel.

The next ring is called **Shell**. It is simply a shell around the kernel that allows user applications to interact with the kernel and use its many services. Take note that the shell ring alone brings mainly most of the requirements addressed by the Unix philosophy that we explained in the previous section. We will elaborate more on this in the upcoming paragraphs.

The shell ring consists of many small programs, which, together, form a set of tools that allows a user or an application to use the kernel services. It also contains a set of libraries, all written in C, which will allow a programmer to write a new application for Unix.

Based on the libraries found in **Simple Unix Specification** (**SUS**), the shell ring must expose a standard and a precisely defined interface for programmers. Such standardizations will make Unix programs portable, or at least compilable, on various Unix implementations. We will reveal some shocking secrets about this ring in the following sections!

Finally, the outermost ring, **User Applications**, consists of all of the actual applications written to be used on Unix systems, such as database services, web services, mail services, web browsers, worksheet programs, and word editor programs.

These applications should use the APIs and tools provided by the shell ring instead of accessing the kernel directly (via *system calls*, which we will discuss shortly) to accomplish their tasks. This is done because of the portability principle in the Unix philosophy. Note that in our current context, by the term *user*, we usually mean the user applications, and not necessarily the people working with these applications.

Being restricted to use just the shell ring also helps these applications to be compliable on various Unix-like operating systems that are not true Unix-compliant operating systems. The best example is the various Linux distributions, which are just Unix-like. We like to have big pieces of software available on both Unix-compliant and Unix-like operating systems with a single code base. As we progress, you find out more about the differences between Unix-like and Unix-compliant systems.

One general theme in the Unix onion is the fact that the inner rings should provide some interface for the outer rings in order to let them use their services. In fact, these interfaces between the rings are more important than the rings themselves. For example, we are more interested in knowing how to use the existing kernel services rather than just digging down the kernel, which is different from one Unix implementation to another.

The same could be said of the shell ring and the interface it exposes to the user applications. In fact, these interfaces are our main subject focus across these two chapters while looking at Unix. In the following sections, we're going to talk about each ring individually and discuss its exposed interface in some detail.

# Shell interface to user applications

A *human user* either uses a Terminal or a specific GUI program such as a web browser to use the functionalities available on a Unix system. Both are referred to as user applications, or just simply applications or programs, that allow the hardware to be used through the shell ring. Memory, CPU, network adapter, and hard drives are typical examples of hardware that are usually used by most Unix programs through the API provided by the shell ring. The API provided is one of the topics that we are going to talk about.

From a developer's perspective, there is not much difference between an application and a program. But from a human user's perspective, an application is a program that has a means such as a **Graphical User interface** (**GUI**) or **Command-Line Interface** (**CLI**) to interact with the user, but a program is a piece of software running on a machine that has no UI, such as a running service. This book does not distinguish between programs and applications, and we use the terms interchangeably.

There is a wide range of programs that have been developed for Unix in C. Database services, web servers, mail servers, games, office applications, and many others are among various types of programs that exist in a Unix environment. There is one common feature among these applications, and that is that their code is portable on most Unix and Unix-like operating systems with some slight changes. But how is that possible? How can you write a program in C that can be built on various versions of Unix and through various types of hardware?

The answer is simple: all Unix systems expose the same **Application Programming Interface** (**API**) from their shell ring. A piece of C source code that is only using the exposed standard interface can be built and run across all Unix systems.

But what exactly do we mean by exposing an API? An API, as we have explained before, is a bunch of header files that contain a set of declarations. In Unix, these headers, and the declared functions in them, are the same throughout all Unix systems, but the implementation of those functions, in other words the static and dynamic libraries written for each UNIX-compliant system, can be unique and different from others.

Note that we are looking at Unix as a standard and not an operating system. There are systems that are built fully compatible with the Unix standard, and we call them *Unix-compliant systems*, such as BSD Unix, while there are systems that partly comply with the Unix standard, and which are called *Unix-like systems*, such as Linux.

The same API is being exposed from the shell ring in more or less all Unix systems. As an example, the `printf` function must always be declared in the `stdio.h` header file, as specified by the Unix standard. Whenever you want to print something to the standard output in a Unix-compliant system, you should use `printf` or `fprintf` from the `stdio.h` header file.

In fact, `stdio.h` is not part of C even though all C books explain this header and the declared functions in it. It's part of the C standard library specified in the SUS standard. A C program written for Unix is not aware of the actual implementation of a specific function, such as `printf` or `fopen`. In other words, the shell ring is seen as a black box by the programs in the outer ring.

The various APIs exposed by the shell ring are collected under the SUS standard. This standard is maintained by **The Open Group** consortium and has had multiple iterations since the creation of Unix. The most recent version is SUS version 4, which goes back to 2008\. However, the most recent version has itself some revisions in 2013, 2016, and finally in 2018.

The following link will take you to the document explaini[ng the exposed interfaces in SUS version](http://www.unix.org/version4/GS5_APIs.pdf) 4: [http://www.unix.org/version4/GS5_APIs.pdf](http://www.unix.org/version4/GS5_APIs.pdf). As you can see in the link, there are different kinds of APIs that are exposed by the shell ring. Some of these are mandatory, and some others are optional. The following is the list of APIs found in SUS v4:

*   **System interfaces**: This is a list of all functions that should be usable by any C program. SUS v4 has 1,191 functions that need to be implemented by a Unix system. The table also describes the fact that a specific function either is mandatory or optional for a specific version of C. Take note that the version we are interested in is C99.
*   **Header interfaces**: This is a list of header files that can be available in an SUS v4-compatible Unix system. In this version of SUS, there are 82 header files that can be accessible to all C programs. If you go through the list, you will find many famous header files, such as `stdio.h`, `stdlib.h`, `math.h`, and `string.h`. Based on the Unix version and the C version used, some of them are mandatory, while others are optional. The optional headers could be missing in a Unix system, but mandatory header files certainly exist somewhere in the filesystem.
*   **Utility interfaces**: This is a list of utility programs, or command-line programs, that should be available in a SUS v4-compatible Unix system. If you go through the tables, you will see lots of commands that are familiar to you, for example, `mkdir`, `ls`, `cp`, `df`, `bc`, and many more, which make up to 160 utility programs. Note that these are usually programs that must have already been written by a specific Unix vendor before shipping as part of its installation bundle.

    These utility programs are mostly used in Terminals or in shell scripts and are not often called by another C program. These utility programs usually use the same system interfaces that are exposed to an ordinary C program written in the user application ring.

    As an example, the following is a link to the `mkdir` utility program's source code written for macOS High Sierra 10.13.6, which is a Berkeley Software Distribution (BSD) - based Unix system. The source code is published on the Apple Open Source website, macOS High Sierra (10.13.6), and is available at [https://opensource.apple.com/source/file_cmds/file_cmds-272/mkdir/mkdir.c](https://opensource.apple.com/source/file_cmds/file_cmds-272/mkdir/mkdir.c).

    If you open the link and go through the source, you see that it is using `mkdir` and `umask` functions declared as part of the system interfaces.

*   **Scripting interface**: This interface is a language that is used to write *shell scripts*. It is mainly used for writing automated tasks that are using utility programs. This interface is usually denoted as a *shell scripting language* or a *shell command language*.
*   **XCURSES interfaces**: XCURSES is a set of interfaces that allow a C program to interact with the user in a minimalistic text-based GUI.

    In the following screenshot, you can see an example of the GUI that has been written using `ncurses` that is an implementation for XCURSES.

    In SUS v4, there are 379 functions located in 3 headers, together with 4 utility programs, which make up the XCURSES interface.

    Many programs today are still using XCURSES to interact with the user through a better interface. It's worth noting that, by using XCURSES-based interfaces, you don't need to have a graphics engine. Likewise, it is usable over remote connections such as **Secure Shell** (**SSH**) as well.

![](img/B11046_10_03.png)

Figure 10-3: A config menu based on ncurses (Wikipedia)

As you can see, SUS doesn't talk about the filesystem hierarchy and the place where header files should be found. It only states which headers should be available and present in the system. A widely used convention for the path of standard header files says that these headers should be found in either `/usr/include` or `/usr/local/include`, but it is still up to the operating system and the user to make the final decision. These are the default paths for the header files. However, systems can be configured to use other paths instead of default ones.

If we put system interfaces and header interfaces together with the implementation of the exposed functions, which are different in each Unix flavor (or implementation), then we get the **C Standard Library** or **libc**. In other words, libc is a set of functions placed in specific header files, all according to SUS, together with the static and shared libraries containing the implementation of the exposed functions.

The definition of libc is entangled tightly with standardizations of Unix systems. Every C program being developed in a Unix system uses libc for communicating further down to the kernel and the hardware levels.

It's important to remember that not all operating systems are Unix fully compatible systems. Microsoft Windows and operating systems using the Linux kernel, for example, Android, are examples of that. These operating systems are not Unix-compliant systems, but they can be Unix-like systems. We have used the terms Unix-compliant and Unix-like across earlier chapters without explaining their true meanings, but now we are going to define them carefully.

A Unix-compliant system is fully compliant to SUS standards, but this isn't true of a Unix-like system that is only partially compliant with the standard. What this means is that the Unix-like systems are only conforming to a specific subset of SUS standards and not all of them. This means, theoretically, that the programs developed for a Unix-compliant system are supposed to be portable to other Unix-compatible systems, but may not be ported to a Unix-like operating system. This is especially the case regarding the programs being ported from Linux to, or to Linux from, other Unix-compliant systems.

Having lots of Unix-like operating systems developed, especially after the birth of Linux, became the basis for giving this subset of SUS standards a specific name. They called it the **Portable Operating System Interface** (**POSIX**). We can say that POSIX is a subset of SUS standards that Unix-like systems chose to comply with.

In the following link, you can find all of the diffe[rent interfaces that should be exposed in a POSI](http://pubs.opengroup.org/onlinepubs/9699919799/)X system: [http://pubs.opengroup.org/onlinepubs/9699919799/](http://pubs.opengroup.org/onlinepubs/9699919799/).

As you can see in the link, there are similar interfaces in POSIX, just like there are in SUS. The standards are remarkably similar, but POSIX has enabled Unix standards to be applicable to a broader range of operating systems.

Unix-like operating systems, such as most Linux distributions, are POSIX-compliant from the beginning. That's why if you've worked with Ubuntu, you can work with FreeBSD Unix in the same manner.

However, that's not true for some operating systems such as Microsoft Windows. Microsoft Windows cannot be considered as POSIX-compliant, but further tools can be installed to make it a POSIX operating system, for example, *Cygwin*¸ a POSIX-compatible environment that runs natively on the Windows operating system.

This again shows that POSIX compatibility is about having a standard shell ring and not the kernel.

On a slight tangent, it was quite the story when Microsoft Windows became POSIX-compliant in the 1990s. However, over time, that support became deprecated.

Both SUS and POSIX standards dictate the interfaces. They both state what should be available, but they don't talk about how it should become available. Each Unix system has its own implementation of POSIX or SUS implementation. These implementations are then put in libc libraries that are part of the shell ring. In other words, in a Unix system, the shell ring contains a libc implementation that is exposed in a standard way. Subsequently, the shell ring will pass the request further down to the kernel ring to be processed.

# Kernel interface to shell ring

In the previous section, we explained that the shell ring in a Unix system exposes the interfaces defined in the SUS or POSIX standard. There are mainly two ways to invoke a certain logic in the shell ring, either through the libc or using shell utility programs. A user application should either get linked with libc libraries to execute shell routines, or it should execute an existing utility program that's available in the system.

Note that the existing utility programs are themselves using the libc libraries. Therefore, we can generalize and state that all shell routines can be found in libc libraries. This gives even more importance to standard C libraries. If you want to create a new Unix system from scratch, you must write your own libc after having the kernel up and ready.

If you have followed the flow of this book and have read the previous chapters, you'll see that pieces of the puzzle are coming together. We needed to have a compilation pipeline and a linking mechanism to be able to design an operating system that exposes an interface and has been implemented using a set of library files. At this point, you are able to see that every feature of C is acting in favor of having Unix. The more you understand about the relationship between C and Unix, the more you find them tied together.

Now that the relationship between user applications and the shell ring is clear, we need to see how the shell ring (or libc) communicates with the kernel ring. Before we go any further, note that, in this section, we are not going to explain what a kernel is. Instead, we are going to look at it as a black box exposing certain functionalities.

The main mechanism that libc (or the functions in shell ring) uses to consume a kernel functionality is through using *system calls*. To explain this mechanism, we need to have an example to follow down the rings of the onion model in order to find the place where system calls are used to do certain things.

We also need to choose a real libc implementation, so we can track down the sources and find the system calls. We choose FreeBSD for further investigations. FreeBSD is a Unix-like operating system branched from the BSD [Unix.](https://github.com/freebsd/freebsd0)

 [**Note**:

The Git repository of FreeBSD can be found here: https://github.com/freebsd/freebsd. This repository contains the sources for FreeBSD's kernel and shell rings. The sources for FreeBSD libc can be found in the `lib/libc` directory.](https://github.com/freebsd/freebsd0) 

Let's start with the following example. *Example 10.1* is a simple program that just waits for one second. Likewise, the program is considered to be in the application ring, which means it is a user application, even though it is remarkably simple.

So, let's first look at the source code of *example 10.1*:

```cpp
#include <unistd.h>
int main(int argc, char** argv) {
  sleep(1);
  return 0;
}
```

Code Box 10-1 [ExtremeC_examples_chapter10_1.c]: Example 10.1 calling the sleep function included from the shell ring

As you can see, the code includes the `unistd.h` header file and calls the `sleep` function, both of which are part of the SUS exposed interfaces. But then what happens next, especially in the `sleep` function? As a C programmer, you may have never asked yourself this question before, but knowing it can enhance your understanding of a Unix system.

We have always used functions such as `sleep`, `printf`, and `malloc`, without knowing how they work internally, but now we want to take a leap of faith and discover the mechanism that libc uses to communicate with the kernel.

We know that system calls, or *syscalls* for short, are being triggered by the codes written in a libc implementation. In fact, this is the way that kernel routines are triggered. In SUS, and subsequently in POSIX-compatible systems, there is a program that is used to trace system calls when a program is running.

We are almost certain that a program that doesn't call system calls literally cannot do anything. So, as a result, we know that every program that we write has to use system calls through calling the libc functions.

Let's compile the preceding example and find out the system calls that it triggers. We can start this process by running:

```cpp
$ cc ExtremeC_examples_chapter10_1.c -lc -o ex10_1.out
$ truss ./ex10_1.out
...                                           
$
```

Shell Box 10-1: Building and running example 10.1 using truss to trace the system calls that it invokes

As you can see in *Shell Box 10-1*, we have used a utility program called `truss`. The following text is an excerpt from the FreeBSD's manual page for `truss`:

> "The truss utility traces the system calls called by the specified process or program. The output is to the specified output file or standard error by default. It does this by stopping and restarting the process being monitored via ptrace(2)."

As the description implies, `truss` is a program for seeing all system calls that a program has invoked during the execution. Utilities similar to truss are available in most Unix-like systems. For instance, `strace` can be used in Linux systems.

The following shell box shows the output of `truss` being used to monitor the system calls invoked by the preceding example:

```cpp
$ truss ./ex10_1.out
mmap(0x0,32768,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0x0) = 34366160896 (0x800620000)
issetugid()                                      = 0 (0x0)
lstat("/etc",{ mode=drwxr-xr-x ,inode=3129984,size=2560,blksize=32768 }) = 0 (0x0)
lstat("/etc/libmap.conf",{ mode=-rw-r--r-- ,inode=3129991,size=109,blksize=32768 }) = 0 (0x0)                                                     openat(AT_FDCWD,"/etc/libmap.conf",O_RDONLY|O_CLOEXEC,00) = 3 (0x3)
fstat(3,{ mode=-rw-r--r-- ,inode=3129991,size=109,blksize=32768 }) = 0 (0x0)
...
openat(AT_FDCWD,"/var/run/ld-elf.so.hints",O_RDONLY|O_CLOEXEC,00) = 3 (0x3)                                                                       read(3,"Ehnt\^A\0\0\0\M^@\0\0\0Q\0\0\0\0"...,128) = 128 (0x80)
fstat(3,{ mode=-r--r--r-- ,inode=7705382,size=209,blksize=32768 }) = 0 (0x0)
lseek(3,0x80,SEEK_SET)                           = 128 (0x80)                                                                                     read(3,"/lib:/usr/lib:/usr/lib/compat:/u"...,81) = 81 (0x51)
close(3)                                         = 0 (0x0)
access("/lib/libc.so.7",F_OK)                    = 0 (0x0)
openat(AT_FDCWD,"/lib/libc.so.7",O_RDONLY|O_CLOEXEC|O_VERIFY,00) = 3 (0x3)
...
sigprocmask(SIG_BLOCK,{ SIGHUP|SIGINT|SIGQUIT|SIGKILL|SIGPIPE|SIGALRM|SIGTERM|SIGURG|SIGSTOP|SIGTSTP|SIGCONT|SIGCHLD|SIGTTIN|SIGTTOU|SIGIO|SIGXCPU|SIGXFSZ|SIGVTALRM|SIGPROF|SIGWINCH|SIGINFO|SIGUSR1|SIGUSR2 },{ }) = 0 (0x0)sigprocmask(SIG_SETMASK,{ },0x0)                 = 0 (0x0)
sigprocmask(SIG_BLOCK,{ SIGHUP|SIGINT|SIGQUIT|SIGKILL|SIGPIPE|SIGALRM|SIGTERM|SIGURG|SIGSTOP|SIGTSTP|SIGCONT|SIGCHLD|SIGTTIN|SIGTTOU|SIGIO|SIGXCPU|SIGXFSZ|SIGVTALRM|SIGPROF|SIGWINCH|SIGINFO|SIGUSR1|SIGUSR2 },{ }) = 0 (0x0)sigprocmask(SIG_SETMASK,{ },0x0)                 = 0 (0x0)
nanosleep({ 1.000000000 })                       = 0 (0x0)
sigprocmask(SIG_BLOCK,{ SIGHUP|SIGINT|SIGQUIT|SIGKILL|SIGPIPE|SIGALRM|SIGTERM|SIGURG|SIGSTOP|SIGTSTP|SIGCONT|SIGCHLD|SIGTTIN|SIGTTOU|SIGIO|SIGXCPU|SIGXFSZ|SIGVTALRM|SIGPROF|SIGWINCH|SIGINFO|SIGUSR1|SIGUSR2 },{ }) = 0 (0x0)
...
sigprocmask(SIG_SETMASK,{ },0x0)                 = 0 (0x0)
exit(0x0)
process exit, rval = 0                                                   $
```

Shell Box 10-2: Output of truss showing the system calls invoked by example 10.1

As you can see in the preceding output, there are many system calls initiated by our simple example, with some of them being about loading shared object libraries, especially when initializing the process. The first system call shown in bold opens the `libc.so.7` shared object library file. This shared object library contains the actual implementation of FreeBSD's libc.

In the same shell box, you can see that the program is calling the `nanosleep` system call. The value passed to this system call is 1000000000 nanoseconds, which is equivalent to 1 second.

System calls are like function calls. Note that each system call has a dedicated and predetermined constant number, and subsequently, together with that, it has a specific name, and a list of arguments. Each system call also performs a specific task. In this case, `nanosleep` makes the calling thread sleep for the specified number of nanoseconds.

More information regarding the system calls can be found in the FreeBSD *system calls manual*. The following shell box shows the page dedicated to the `nanosleep` system call in the manual:

```cpp
$ man nanosleep
NANOSLEEP(2)              FreeBSD System Calls Manual             NANOSLEEP(2)
NAME
     nanosleep - high resolution sleep
LIBRARY      Standard C Library (libc, -lc)
SYNOPSIS      #include <time.h>
     Int
     clock_nanosleep(clockid_t clock_id, int flags,
         const struct timespec *rqtp, struct timespec *rmtp);
     int
     nanosleep(const struct timespec *rqtp, struct timespec *rmtp);
DESCRIPTION
     If the TIMER_ABSTIME flag is not set in the flags argument, then
     clock_nanosleep() suspends execution of the calling thread until either
     the time interval specified by the rqtp argument has elapsed, or a signal
     is delivered to the calling process and its action is to invoke a signal-
     catching function or to terminate the process.  The clock used to measure
     the time is specified by the clock_id argument
...
...
$
```

Shell Box 10-3: The manual page dedicated to the nanosleep system call

The preceding manual page describes the following:

*   `nanosleep` is a system call.
*   The system call is accessible through calling the `nanosleep` and `clock_nanosleep` functions from the shell ring defined in `time.h`. Note that we used the `sleep` function from `unitsd.h`. We could also use the preceding two functions from `time.h`. It's also worth noting that both header files and all of the preceding functions, together with the functions actually used, are part of SUS and POSIX.
*   If you want to be able to call these functions, you need to link your executable against libc by passing the `-lc` option to your linker. This might be specific to FreeBSD only.
*   This manual page doesn't talk about the system call itself, but it talks about the standard C API, which is exposed from the shell ring. These manuals are written for application developers and, as such, they won't discuss the systems calls and kernel internals often. Instead, they focus on the APIs exposed from the shell ring.

Now, let's find the place in libc where the system call is actually invoked. We will be using FreeBSD sources on GitHub. The commit hash we are using is `bf78455d496` from the master branch. In order to clone and use the proper commit from the repository, run the following commands:

```cpp
$ git clone https://github.com/freebsd/freebsd
...
$ cd freebsd
$ git reset --hard bf78455d496
...
$
```

Shell Box 10-4: Cloning the FreeBSD project and going to a specific commit

It is also possibl[e to navigate the FreeBSD project on the GitHub website](https://github.com/freebsd/freebsd/tree/bf78455d496) itself using the following link: [https://github.com/freebsd/freebsd/tree/bf78455d496](https://github.com/freebsd/freebsd/tree/bf78455d496). No matter what method you use to navigate the project, you should be able to find the following line of codes.

If you go into the `lib/libc` directory and do a `grep` for `sys_nanosleep`, you will find the following file entries:

```cpp
$ cd lib/libc
$ grep sys_nanosleep . -R
./include/libc_private.h:int		__sys_nanosleep(const struct timespec *, struct timespec *);
./sys/Symbol.map:	__sys_nanosleep;
./sys/nanosleep.c:__weak_reference(__sys_nanosleep, __nanosleep);
./sys/interposing_table.c:	SLOT(nanosleep, __sys_nanosleep), 
$
```

Shell Box 10-5: Finding the entries related to the nanosleep system call in FreeBSD libc files

As you can see in the `lib/libc/sys/interposing_table.c` file, the `nanosleep` function is mapped to the `__sys_nanosleep` function. Therefore, any function call targeted at `nanosleep` will cause `__sys_nanosleep` to be invoked.

The functions starting with `__sys` are actual system call functions in FreeBSD convention. Note that this is part of the libc implementation, and the used naming convention and other implementation-related configurations are highly specific to FreeBSD.

Having said all of that there's also another interesting point in the preceding shell box. The `lib/libc/include/libc_private.h` file contains the private and internal function declarations required for the wrapper functions around the system calls.

So far, we have seen how shell rings route the function calls made to libc to the inner rings by using system calls. But why do we need system calls in the first place? Why is it called a system call and not a function call? When looking at an ordinary function in a user application or libc, how is it different from a system call in the kernel ring? In *Chapter 11*, *System Calls and Kernels*, we will discuss this further by giving a more concrete definition of a system call.

The next section is about the kernel ring and its internal units, which are common in kernels used by most Unix-compliant and Unix-like systems.

# Kernel

The main purpose of the kernel ring is to manage the hardware attached to a system and expose its functionalities as system calls. The following diagram shows how a specific hardware functionality is exposed through different rings before a user application can finally use it:

![](img/B11046_10_04.png)

Figure 10-4: Function calls and system calls made between various Unix rings in order to expose a hardware functionality

The preceding diagram shows a summary of what we have explained so far. In this section, we are going to focus on the kernel itself and see what the kernel is. A kernel is a process that, like any other processes that we know, executes a sequence of instructions. But a **kernel process** is fundamentally different from an ordinary process, which we know as a **user process**.

The following list compares a kernel process and a user process. Note that our comparison is biased to a monolithic kernel such as Linux. We will explain the different types of kernels in the next chapter.

*   A kernel process is the first thing that is loaded and executed, but user processes need to have the kernel process loaded and running before being spawned.
*   We only have one kernel process, but we can have many user processes working at the same time.
*   The kernel process is created by copying a kernel image into the main memory by the boot loader, but a user process is created using the `exec` or `fork` system calls. These system calls exist in most Unix systems.
*   The kernel process handles and executes system calls, but a user process invokes the system call and waits for its execution handled by the kernel process. This means that, when a user process demands the execution of a system call, the flow of execution is transferred to the kernel process and it is the kernel itself that executes the system call's logic on behalf of the user process. We will clarify this in the second part of our look into Unix, *Chapter 11*, *System Calls and Kernel*.
*   The kernel process sees the physical memory and all of the attached hardware in a *privileged* mode, but a user process sees the virtual memory, which is mapped to a portion of physical memory, where the user process doesn't know anything about the physical memory layout. Likewise, the user process has controlled and supervised access to resources and hardware. We can say that the user process is being executed in a sandbox simulated by the operating system. This also implies that a user process cannot see the memory of another user process.

As it is understood from the preceding comparison, we have two different execution modes in an operating system's runtime. One of them is dedicated to the kernel process, and the other is dedicated to the user processes.

The former execution mode is called *kernel land* or *kernel space*, and the latter is called *user land* or *user space*. Calling system calls by user processes is a way to bring these two lands together. Basically, we invented the system calls because we needed to isolate the kernel space and the user space from each other. Kernel space has the most privileged access to the system resources, and the user space has the least privileged and supervised access.

The internal structure of a typical Unix kernel can be discerned by the tasks a kernel performs. In fact, managing the hardware is not the only task that a kernel performs. The following is the list of a Unix kernel's responsibilities. Note that we have included the hardware management tasks as well in the following list:

*   **Process management**: User processes are created by the kernel via a system call. Allocating memory for a new process and loading its instructions are some of the operations, among all of the operations, that should be performed before running a process.
*   **Inter-Process Communication** (**IPC**): User processes on the same machine can use different methods for exchanging data among them. Some of these methods are shared memories, pipes, and Unix domain sockets. These methods should be facilitated by the kernel, and some of them need the kernel to control the exchange of data. We will explain these methods in *Chapter 19*, *Single Host IPC and Sockets*, while talking about IPC techniques.
*   **Scheduling**: Unix has always been known as a multi-tasking operating system. The kernel manages access to CPU cores and tries to balance access to them. Scheduling is a name given to the task that shares the CPU time among many processes based on their priority and importance. We will explain more about multi-tasking, multithreading, and multi-processing in the following chapters.
*   **Memory management**: Without doubt, this is one of the key tasks of a kernel. The kernel is the only process that sees the whole physical memory and has superuser access to it. So, the task of breaking it into allocatable pages, assigning new pages to the processes in case of Heap allocation, freeing the memory, and many more memory-related tasks besides, should be performed and managed by the kernel.
*   **System startup**: Once the kernel image is loaded into the main memory and the kernel process is started, it should initialize the user space. This is usually done by creating the first user process with the **Process Identifier** (**PID**) 1\. In some Unix systems such as Linux, this process is called *init*. After having this process started, further services and daemons will be started by it.
*   **Device management**: Apart from the CPU and memory, the kernel should be able to manage hardware through an abstraction made over all of them. A *device* is a real or virtual hardware attached to a Unix system. A typical Unix system uses the `/dev` path to store mapped device files. All attached hard drives, network adapters, USB devices, etc. are mapped to files found in the `/dev` path. These device files can be used by user processes to communicate with these devices.

The following diagram shows the most common internal structure of a Unix kernel based on the preceding list:

![](img/fig10-5.png)

Figure 10-5: Internal structure of different rings in the Unix architecture

The preceding figure is a detailed illustration of Unix rings. It clearly shows that, in the shell ring, we have three parts that are exposed to the user applications. It also shows a detailed inner structure of the kernel ring.

At the top in the kernel ring, we have the system call interface. As is clear in the figure, all of the preceding units that are in the user space must communicate with the bottom units only through the system call interface. This interface is like a gate or a barrier between user and kernel spaces.

There are various units in the kernel such as the **Memory Management unit** (**MMU**) that manages the available physical memory. The **Process Management unit** creates processes in the user space and allocates resources for them. It also makes IPC available to processes. The diagram also shows the **Character** and **Block devices** that are mediated by the **Device Drivers** to expose the various I/O functionalities. We explain character and block devices in the following section. The **File System** unit is an essential part of the kernel, which is an abstraction over the block and character devices and lets the processes and the kernel itself use the same shared file hierarchy.

In the next section, we are going to talk about hardware.

# Hardware

The final purpose of every operating system is to allow the user and the applications to be able to use and interact with hardware. Unix also aims to provide access to the attached hardware in an abstract and transparent way, using the same set of utility programs and commands in all existing and future platforms.

By having this transparency and abstraction, Unix abstracts all of the different hardware to be a number of devices attached to a system. So, the term *device* is centric in Unix, and every connected piece of hardware is considered to be a device connected to the Unix system.

The hardware attached to a computer can be categorized into two different categories: *mandatory* and *peripheral*. The CPU and main memory are mandatory devices attached to a Unix system. All other hardware such as the hard drive, network adapter, mouse, monitor, graphics card, and Wi-Fi adapter, are peripheral devices.

A Unix machine cannot work without mandatory hardware, but you can have a Unix machine that doesn't have a hard drive or a network adapter. Note that having a filesystem, which is essential for a Unix kernel to operate, doesn't necessarily require a hard disk!

A Unix kernel completely hides the CPU and physical memory. They are managed directly by the kernel, and no access is allowed to be made from the user space. The **Memory Management** and **Scheduler** units in a Unix kernel are responsible for managing the physical memory and the CPU, respectively.

This is not the case with other peripheral devices connected to a Unix system. They are exposed through a mechanism called *device files*. You can see these files in a Unix system as part of the `/dev` path.

The following is the list of files that can be found on an ordinary Linux machine:

```cpp
$ ls -l /dev
total 0
crw-r--r--  1 root   root     10, 235 Oct 14 16:55 autofs
drwxr-xr-x  2 root   root         280 Oct 14 16:55 block
drwxr-xr-x  2 root   root          80 Oct 14 16:55 bsg
crw-rw----  1 root   disk     10, 234 Oct 14 16:55 btrfs-control
drwxr-xr-x  3 root   root          60 Oct 14 17:02 bus
lrwxrwxrwx  1 root   root           3 Oct 14 16:55 cdrom -> sr0
drwxr-xr-x  2 root   root        3500 Oct 14 16:55 char
crw-------  1 root   root      5,   1 Oct 14 16:55 console
lrwxrwxrwx  1 root   root          11 Oct 14 16:55 core -> /proc/kcore
crw-------  1 root   root     10,  59 Oct 14 16:55 cpu_dma_latency
crw-------  1 root   root     10, 203 Oct 14 16:55 cuse
drwxr-xr-x  6 root   root         120 Oct 14 16:55 disk
drwxr-xr-x  3 root   root          80 Oct 14 16:55 dri
lrwxrwxrwx  1 root   root           3 Oct 14 16:55 dvd -> sr0
crw-------  1 root   root     10,  61 Oct 14 16:55 ecryptfs
crw-rw----  1 root   video    29,   0 Oct 14 16:55 fb0
lrwxrwxrwx  1 root   root          13 Oct 14 16:55 fd -> /proc/self/fd
crw-rw-rw-  1 root   root      1,   7 Oct 14 16:55 full
crw-rw-rw-  1 root   root     10, 229 Oct 14 16:55 fuse
crw-------  1 root   root    245,   0 Oct 14 16:55 hidraw0
crw-------  1 root   root     10, 228 Oct 14 16:55 hpet
drwxr-xr-x  2 root   root           0 Oct 14 16:55 hugepages
crw-------  1 root   root     10, 183 Oct 14 16:55 hwrng
crw-------  1 root   root     89,   0 Oct 14 16:55 i2c-0
...
crw-rw-r--  1 root   root     10,  62 Oct 14 16:55 rfkill
lrwxrwxrwx  1 root   root           4 Oct 14 16:55 rtc -> rtc0
crw-------  1 root   root    249,   0 Oct 14 16:55 rtc0
brw-rw----  1 root   disk      8,   0 Oct 14 16:55 sda
brw-rw----  1 root   disk      8,   1 Oct 14 16:55 sda1
brw-rw----  1 root   disk      8,   2 Oct 14 16:55 sda2
crw-rw----+ 1 root   cdrom    21,   0 Oct 14 16:55 sg0
crw-rw----  1 root   disk     21,   1 Oct 14 16:55 sg1
drwxrwxrwt  2 root   root          40 Oct 14 16:55 shm
crw-------  1 root   root     10, 231 Oct 14 16:55 snapshot
drwxr-xr-x  3 root   root         180 Oct 14 16:55 snd
brw-rw----+ 1 root   cdrom    11,   0 Oct 14 16:55 sr0
lrwxrwxrwx  1 root   root          15 Oct 14 16:55 stderr -> /proc/self/fd/2
lrwxrwxrwx  1 root   root          15 Oct 14 16:55 stdin -> /proc/self/fd/0
lrwxrwxrwx  1 root   root          15 Oct 14 16:55 stdout -> /proc/self/fd/1
crw-rw-rw-  1 root   tty       5,   0 Oct 14 16:55 tty
crw--w----  1 root   tty       4,   0 Oct 14 16:55 tty0
...
$
```

Shell Box 10-6: Listing the content of /dev on a Linux machine

As you can see, it is quite a list of devices attached to the machine. But of course, not all of them are physical. The abstraction over the hardware devices in Unix has given it the ability to have *virtual devices*.

For example, you can have a virtual network adapter that has no physical counterpart, but is able to perform additional operations on the network data. This is one of the ways that VPNs are being used in Unix-based environments. The physical network adapter brings the real network functionality, and the virtual network adapter gives the ability to transmit the data through a secure tunnel.

As is clear from the preceding output, each device has its own file in the `/dev` directory. The lines starting with `c` and `b` are device files representing character devices and block devices, respectively. Character devices are supposed to deliver and consume data byte by byte. Examples of such devices are serial ports, and parallel ports. Block devices are supposed to deliver and consume chunks of data that have more than one byte. Hard disks, network adapters, cameras, and so on are examples of block devices. In the preceding shell box, the lines starting with 'l' are symbolic links to other devices, and the lines starting with d represent directories that may contain other device files.

User processes use these device files in order to access the corresponding hardware. These files can be written or can be read in order to send or receive data to and from the device.

In this book, we won't go deeper than this, but if you are curious to know more about devices and device drivers, you should read more around this subject. In the next chapter, *Chapter 11*, *System Calls and Kernels*, we continue our talk about system calls in greater detail, and we will add a new system call to an existing Unix kernel.

# Summary

In this chapter, we started to discuss Unix and how it is interrelated with C. Even in non-Unix operating systems, you see some traces of a similar design to Unix systems.

As part of this chapter, we went through the history of the early 1970s and explained how Unix appeared from Multics and how C was derived from the B programming language. After that, we talked about the Unix architecture, an onion-like architecture that consists of four layers: user applications, the shell, the kernel, and hardware.

We briefly went over the various layers in the Unix onion model and provided detailed explanations of the shell layer. We introduced the C standard library and how it is used through POSIX and SUS standards to give programmers the ability to write programs that can be built on various Unix systems.

In the second part of our look into Unix, *Chapter 11*, *System Calls and Kernels*, we will continue our discussion about Unix and its architecture, and we will provide explanations of the kernel and the system call interface surrounding it in greater depth.