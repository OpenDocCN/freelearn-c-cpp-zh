# Preface

C++ does not add any bloat, extends maintainability, and offers many advantages over different programming languages, thus making it a good choice for embedded development. Do you want to build standalone or networked embedded systems and make them safety-critical and memory-safe? In this book, you will learn exactly how to do this. You will learn how C++ works and compares to other languages used for embedded development, how to create advanced GUIs for embedded devices in order to design an attractive and functional UI, and how to integrate proven strategies into your design for optimum hardware performance.

This book will take you through various embedded systems hardware boards so that you can choose the best one for your project. You will learn how to tackle complex architectural problems by fully embracing the proven programming patterns presented in the book.

# Who this book is for

If you want to start developing effective embedded programs in C++, then this book is for you. Good knowledge of C++ language constructs is required to understand the topics covered in the book. No knowledge of embedded systems is assumed.

# What this book covers

[Chapter 1](0ff8cac9-3155-45e1-af05-7005fc419dd6.xhtml), *What Are Embedded Systems?* makes you familiar with what an embedded system entails. By looking at the various categories and examples of embedded systems in each category, a good overview of what is meant with the term *embedded* and the wide variety within that term should be formed. It explores the wide range of historic and currently available microcontrollers and system-on-chip solutions you can find in existing systems as well as new designs.

[Chapter 2](cae3bf4a-2936-42b4-a33e-569e693bfcc8.xhtml), *C++ as an Embedded Language*, explains why C++ is actually as nimble as C and similar languages. Not only is C++ generally at least as fast as C, there is no additional bloat, and it offers many advantages with code paradigms and maintainability.

[Chapter 3](47e0b6fb-cb68-43c3-9453-2dc7575b1a46.xhtml), *Developing for Embedded Linux and Similar Systems*, explains how to develop for Linux-based embedded systems and kin on SBCs and manage the differences between Linux-based and PC-based development.

[Chapter 4](bb67db6a-7c71-4519-80c3-7cd571cddfc0.xhtml), *Resource-Restricted Embedded Systems*, deals with planning for and using limited resources efficiently. We will take a look at how to select the right MCU for a new project and add peripherals and deal with Ethernet and serial interface requirements in a project. We will also look at an example AVR project, how to develop for other MCU architectures, and whether to use an RTOS.

[Chapter 5](886aecf2-8926-4aec-8045-a07ae2cdde84.xhtml), *Example – Soil Humidity Monitor with Wi-Fi*, explains how to create a Wi-Fi-enabled soil humidity monitor with actuator options for a pump or similar. Using the built-in web server, you can use its browser-based UI for monitoring and control, or integrate it into a larger system using its REST API.

[Chapter 6](7d5d654f-a027-4825-ab9e-92c369b576a8.xhtml), *Testing OS-Based Applications*, looks at how to develop and test embedded OS-based applications. You will learn how to install and use a cross-compilation toolchain, do remote debugging using GDB, and write a build system.

[Chapter 7](d8237285-fcb7-4bbc-84f3-e45568598865.xhtml), *Testing Resource-Restricted Platforms*, shows how to effective develop for MCU-based targets. You will also see how to implement an integration environment that allows us to debug MCU-based applications from the comfort of a desktop OS and the tools it provides.

[Chapter 8](4416b2de-d86a-4001-863d-b167635a0e10.xhtml), *Example – Linux-Based Infotainment System*, explains how you can fairly easily construct an SBC-based infotainment system, using voice-to-text to construct a voice-driven UI. We will also look at how we can extend it to add even more functionality.

[Chapter 9](d165297b-8be7-44f5-90b5-53b3bcb51d3b.xhtml), *Example – Building Monitoring and Control*, shows how a building-wide monitoring and management system is developed, what its components looks like, and what lessons are learned during its development.

[Chapter 10](47e0b6fb-cb68-43c3-9453-2dc7575b1a46.xhtml), *Developing Embedded Systems with Qt*, looks at the myriad of ways in which the Qt framework can be used to develop for embedded systems. We will look at how it compares with other frameworks and how Qt is optimized for these embedded platforms, before working through an example of a QML-based GUI that can be added to the previously-created infotainment system.

[Chapter 11](c90e29ad-2e13-4838-a9c2-885209717513.xhtml), *Developing for Hybrid SoC/FPGA Systems*, teaches you how to communicate with the FPGA side of a hybrid FPGA/SoC system and helps you understand how a variety of algorithms are implemented in FPGA and used on the SoC side. You will also learn how to implement a basic oscilloscope on a hybrid FPGA/SoC system. 

[Appendix](ddead19d-4726-49ec-b780-34689efdd0b7.xhtml), *Best Practices*, runs through a number of common issues and pitfalls that are likely to occur while working on an embedded software design.

# To get the most out of this book

A working knowledge of Raspberry Pi is required. You will need C++ compiler, the GCC ARM Linux (cross-) toolchain, the AVR toolchain, the Sming framework, Valgrind, the Qt framework, and the Lattice Diamond IDE.

# Download the example code files

You can download the example code files for this book from your account at [www.packtpub.com](http://www.packtpub.com). If you purchased this book elsewhere, you can visit [www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files emailed directly to you.

You can download the code files by following these steps:

1.  Log in or register at [www.packtpub.com](http://www.packtpub.com/support).
2.  Select the SUPPORT tab.
3.  Click on Code Downloads & Errata.
4.  Enter the name of the book in the Search box and follow the onscreen instructions.

Once the file is downloaded, please make sure that you unzip or extract the folder using the latest version of:

*   WinRAR/7-Zip for Windows
*   Zipeg/iZip/UnRarX for Mac
*   7-Zip/PeaZip for Linux

The code bundle for the book is also hosted on GitHub at [https://github.com/PacktPublishing/Hands-On-Embedded-Programming-with-CPP-17](https://github.com/PacktPublishing/Hands-On-Embedded-Programming-with-CPP-17). In case there's an update to the code, it will be updated on the existing GitHub repository.

We also have other code bundles from our rich catalog of books and videos available at **[https://github.com/PacktPublishing/](https://github.com/PacktPublishing/)**. Check them out!

# Conventions used

There are a number of text conventions used throughout this book.

`CodeInText`: Indicates code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles. Here is an example: "The C++ class itself is implemented in C as a `struct` containing the class variables."

A block of code is set as follows:

```cpp
class B : public A { 
   // Private members. 

public: 
   // Additional public members. 
}; 
```

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

```cpp
class B : public A { 
   // Private members. 

public: 
   // Additional public members. 
}; 
```

Any command-line input or output is written as follows:

```cpp
sudo usermod -a -G gpio user
sudo usermod -a -G i2c user
```

**Bold**: Indicates a new term, an important word, or words that you see onscreen. For example, words in menus or dialog boxes appear in the text like this. Here is an example: "Compared to MCUs, SoCs are not as resource-limited, usually running a full **operating system** (**OS**) such as a Linux-derived OS, VxWorks, or QNX."

Warnings or important notes appear like this.

Tips and tricks appear like this.

# Get in touch

Feedback from our readers is always welcome.

**General feedback**: Email `feedback@packtpub.com` and mention the book title in the subject of your message. If you have questions about any aspect of this book, please email us at `questions@packtpub.com`.

**Errata**: Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you have found a mistake in this book, we would be grateful if you would report this to us. Please visit [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the Errata Submission Form link, and entering the details.

**Piracy**: If you come across any illegal copies of our works in any form on the Internet, we would be grateful if you would provide us with the location address or website name. Please contact us at `copyright@packtpub.com` with a link to the material.

**If you are interested in becoming an author**: If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, please visit [authors.packtpub.com](http://authors.packtpub.com/).

# Reviews

Please leave a review. Once you have read and used this book, why not leave a review on the site that you purchased it from? Potential readers can then see and use your unbiased opinion to make purchase decisions, we at Packt can understand what you think about our products, and our authors can see your feedback on their book. Thank you!

For more information about Packt, please visit [packtpub.com](https://www.packtpub.com/).