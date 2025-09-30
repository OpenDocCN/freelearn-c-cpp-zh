# Preface

LLVM is one of the very hot topics in recent times. It is an open source project with an ever-increasing number of contributors. Every programmer comes across a compiler at some point or the other while programming. Simply speaking, a compiler converts a high-level language to machine-executable code. However, what goes on under the hood is a lot of complex algorithms at work. So, to get started with compiler, LLVM will be the simplest infrastructure to study. Written in object-oriented C++, modular in design, and with concepts that are very easy to map to theory, LLVM proves to be attractive for experienced compiler programmers and for novice students who are willing to learn.

As authors, we maintain that simple solutions frequently work better and are easier to grasp than complex solutions. Throughout the book we will look at various topics that will help you enhance your skills and drive you to learn more.

We also believe that this book will be helpful for people not directly involved in compiler development as knowledge of compiler development will help them write code optimally.

# What this book covers

[Chapter 1](part0014_split_000.html#DB7S1-684481f6e3394b1e8596d1aa3001290f "Chapter 1. Playing with LLVM"), *Playing with LLVM*, introduces you to the modular design of LLVM and LLVM Intermediate Representation. In this chapter, we also look into some of the tools that LLVM provides.

[Chapter 2](part0018_split_000.html#H5A41-684481f6e3394b1e8596d1aa3001290f "Chapter 2. Building LLVM IR"), *Building LLVM IR*, introduces you to some basic function calls provided by the LLVM infrastructure to build LLVM IR. This chapter demonstrates building of modules, functions, basic blocks, condition statements, and loops using LLVM APIs.

[Chapter 3](part0028_split_000.html#QMFO1-684481f6e3394b1e8596d1aa3001290f "Chapter 3. Advanced LLVM IR"), *Advanced LLVM IR*, introduces you to some advanced IR paradigms. This chapter explains advanced IR to the readers and shows how LLVM function calls can be used to emit them in the IR.

[Chapter 4](part0035_split_000.html#11C3M1-684481f6e3394b1e8596d1aa3001290f "Chapter 4. Basic IR Transformations"), *Basic IR Transformations*, deals with basic transformation optimizations at the IR level using the LLVM optimizer tool opt and the LLVM Pass infrastructure. You will learn how to use the information of one pass in another and then look into Instruction Simplification and Instruction Combining Passes.

[Chapter 5](part0041_split_000.html#173722-684481f6e3394b1e8596d1aa3001290f "Chapter 5. Advanced IR Block Transformations"), *Advanced IR Block Transformations*, deals with optimizations at block level on IR. We will discuss various optimizations such as Loop Optimizations, Scalar Evolution, Vectorization, and so on, followed by the summary of this chapter.

[Chapter 6](part0046_split_000.html#1BRPS2-684481f6e3394b1e8596d1aa3001290f "Chapter 6. IR to Selection DAG phase"), *IR to Selection DAG phase*, takes you on a journey through the abstract infrastructure of a target-independent code generator. We explore how LLVM IR is converted to Selection DAG and various phases thereafter. It also introduces you to instruction selection, scheduling, register allocation, and so on.

[Chapter 7](part0054_split_000.html#1JFUC2-684481f6e3394b1e8596d1aa3001290f "Chapter 7. Generating Code for Target Architecture"), *Generating Code for Target Architecture*, introduces the readers to the tablegen concept. It shows how target architecture specifications such as register sets, instruction sets, calling conventions, and so on can be represented using tablegen, and how the output of tablegen can be used to emit code for a given architecture. This chapter can be used by readers as a reference for bootstrapping a target machine code generator.

# What you need for this book

All you need to work through most of the examples covered in this book is a Linux machine, preferably Ubuntu. You will also need a simple text or code editor, Internet access, and a browser. We recommend installing the meld tool to compare two files; it works well on the Linux platform.

# Who this book is for

This book is intended for those who already know some of the concepts concerning compilers and want to quickly become familiar with LLVM's infrastructure and the rich set of libraries that it provides. Compiler programmers, who are familiar with concepts of compilers and want to indulge in understanding, exploring, and using the LLVM infrastructure in a meaningful way in their work, will find this book useful.

This book is also for programmers who are not directly involved in compiler projects but are often involved in development phases where they write thousands of lines of code. With knowledge of how compilers work, they will be able to code in an optimal way and improve performance with clean code.

# Conventions

In this book, you will find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "The LLVM `Pass Manager` uses the explicitly mentioned dependency information."

A block of code is set as follows:

[PRE0]

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

[PRE1]

Any command-line input or output is written as follows:

[PRE2]

**New terms** and **important words** are shown in bold. Words that you see on the screen, for example, in menus or dialog boxes, appear in the text like this: "Clicking the **Next** button moves you to the next screen."

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