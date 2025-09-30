# Preface

A programmer might have come across compilers at some or the other point when programming. Simply speaking, a compiler converts a human-readable, high-level language into machine-executable code. But have you ever wondered what goes on under the hood? A compiler does lot of processing before emitting optimized machine code. Lots of complex algorithms are involved in writing a good compiler.

This book travels through all the phases of compilation: frontend processing, code optimization, code emission, and so on. And to make this journey easy, LLVM is the simplest compiler infrastructure to study. It's a modular, layered compiler infrastructure where every phase is dished out as a separate recipe. Written in object-oriented C++, LLVM gives programmers a simple interface and lots of APIs to write their own compiler.

As authors, we maintain that simple solutions frequently work better than complex solutions; throughout this book, we'll look at a variety of recipes that will help develop your skills, make you consider all the compiling options, and understand that there is more to simply compiling code than meets the eye.

We also believe that programmers who are not involved in compiler development will benefit from this book, as knowledge of compiler implementation will help them code optimally next time they write code.

We hope you will find the recipes in this book delicious, and after tasting all the recipes, you will be able to prepare your own dish of compilers. Feeling hungry? Let's jump into the recipes!

# What this book covers

[Chapter 1](part0015.xhtml#aid-E9OE1 "Chapter 1. LLVM Design and Use"), *LLVM Design and Use*, introduces the modular world of LLVM infrastructure, where you learn how to download and install LLVM and Clang. In this chapter, we play with some examples to get accustomed to the workings of LLVM. We also see some examples of various frontends.

[Chapter 2](part0028.xhtml#aid-QMFO1 "Chapter 2. Steps in Writing a Frontend"), *Steps in Writing a Frontend*, explains the steps to write a frontend for a language. We will write a bare-metal toy compiler frontend for a basic toy language. We will also see how a frontend language can be converted into the LLVM intermediate representation (IR).

[Chapter 3](part0041.xhtml#aid-173721 "Chapter 3. Extending the Frontend and Adding JIT Support"), *Extending the Frontend and Adding JIT Support*, explores the more advanced features of the toy language and the addition of JIT support to the frontend. We implement some powerful features of a language that are found in most modern programming languages.

[Chapter 4](part0047.xhtml#aid-1CQAE1 "Chapter 4. Preparing Optimizations"), *Preparing Optimizations*, takes a look at the pass infrastructure of the LLVM IR. We explore various optimization levels, and the optimization techniques kicking at each level. We also see a step-by-step approach to writing our own LLVM pass.

[Chapter 5](part0056.xhtml#aid-1LCVG1 "Chapter 5. Implementing Optimizations"), *Implementing Optimizations*, demonstrates how we can implement various common optimization passes on LLVM IR. We also explore some vectorization techniques that are not yet present in the LLVM open source code.

[Chapter 6](part0065.xhtml#aid-1TVKI1 "Chapter 6. Target-independent Code Generator"), *Target-independent Code Generator*, takes us on a journey through the abstract infrastructure of a target-independent code generator. We explore how LLVM IR is converted to Selection DAGs, which are further processed to emit target machine code.

[Chapter 7](part0079.xhtml#aid-2BASE1 "Chapter 7. Optimizing the Machine Code"), *Optimizing the Machine Code*, examines how Selection DAGs are optimized and how target registers are allocated to variables. This chapter also describes various optimization techniques on Selection DAGs as well as various register allocation techniques.

[Chapter 8](part0087.xhtml#aid-2IV0U1 "Chapter 8. Writing an LLVM Backend"), *Writing an LLVM Backend*, takes us on a journey of describing a target architecture. This chapter covers how to describe registers, instruction sets, calling conventions, encoding, subtarget features, and so on.

[Chapter 9](part0098.xhtml#aid-2TEN41 "Chapter 9. Using LLVM for Various Useful Projects"), *Using LLVM for Various Useful Projects*, explores various other projects where LLVM IR infrastructure can be used. Remember that LLVM is not just a compiler; it is a compiler infrastructure. This chapter explores various projects that can be applied to a code snippet to get useful information from it.

# What you need for this book

All you need to work through most of the examples covered in this book is a Linux machine, preferably Ubuntu. You will also need a simple text or code editor, Internet access, and a browser. We recommend installing the meld tool for comparison of two files; it works well on the Linux platform.

# Who this book is for

The book is for compiler programmers who are familiar with concepts of compilers and want to indulge in understanding, exploring, and using LLVM infrastructure in a meaningful way in their work.

This book is also for programmers who are not directly involved in compiler projects but are often involved in development phases where they write thousands of lines of code. With knowledge of how compilers work, they will be able to code in an optimal way and improve performance with clean code.

# Sections

In this book, you will find several headings that appear frequently (Getting ready, How to do it, How it works, There's more, and See also).

To give clear instructions on how to complete a recipe, we use these sections.

## Getting ready

This section tells you what to expect in the recipe, and describes how to set up any software or any preliminary settings required for the recipe.

## How to do it…

This section contains the steps required to follow the recipe.

## How it works…

This section usually consists of a detailed explanation of what happened in the previous section.

## There's more…

This section consists of additional information about the recipe in order to make you more knowledgeable about the recipe.

## See also

This section provides helpful links to other useful information for the recipe.

# Conventions

In this book, you will find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "We can include other contexts through the use of the `include` directive."

A block of code is set as follows:

[PRE0]

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

[PRE1]

Any command-line input or output is written as follows:

[PRE2]

**New terms** and **important words** are shown in bold. Words that you see on the screen, for example, in menus or dialog boxes, appear in the text like this: "Clicking on the **Next** button moves you to the next screen."

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

We also provide you with a PDF file that has color images of the screenshots/diagrams used in this book. The color images will help you better understand the changes in the output. You can download this file from: [https://www.packtpub.com/sites/default/files/downloads/5981OS_ColorImages.pdf](https://www.packtpub.com/sites/default/files/downloads/5981OS_ColorImages.pdf).

## Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books—maybe a mistake in the text or the code—we would be grateful if you could report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the **Errata Submission Form** link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded to our website or added to any list of existing errata under the Errata section of that title.

To view the previously submitted errata, go to [https://www.packtpub.com/books/content/support](https://www.packtpub.com/books/content/support) and enter the name of the book in the search field. The required information will appear under the **Errata** section.

## Piracy

Piracy of copyrighted material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works in any form on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at `<[copyright@packtpub.com](mailto:copyright@packtpub.com)>` with a link to the suspected pirated material.

We appreciate your help in protecting our authors and our ability to bring you valuable content.

## Questions

If you have a problem with any aspect of this book, you can contact us at `<[questions@packtpub.com](mailto:questions@packtpub.com)>`, and we will do our best to address the problem.