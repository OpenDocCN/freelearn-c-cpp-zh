# Preface

Qt is everywhere these days. From your typical home computer to cloud servers, mobile phones, machine automation, coffee machines, medical devices, and embedded devices of all kinds, even in some of the classiest automobiles. It might even be somewhere in space, too! Even watches run Qt these days.

The **Internet of Things** (**IoT**) and home automation are big buzzwords that Qt is also a part of. As I like to say, there's no IoT without sensors! Would I write a book on Qt development without mentioning sensors? No. So, we will also dive into Qt Sensors as well.

There's no shortage of target devices these days, that's for sure. But in this book, we will only specifically target mobile phone platforms and Raspberry Pi to demonstrate some of the embedded features of Qt. Because Qt is cross-platform, you should still come away with knowledge on how to target coffee makers, too!

One consideration to make is which UI framework is going to be used. Using OpenGL is a viable option for mobile devices, especially if there is hardware support. I deliberately skip discussing OpenGL, as that is quite complicated and would be a complete book in itself. It is an option, however, and is available through the use of Qt frameworks.

# Who this book is for

This book is aimed at developers who are interested in developing cross-platform applications with Qt for use on mobile and embedded devices. Readers should have prior knowledge of C++ and be familiar, running commands from a command-line interface.

# What this book covers

[Chapter 1](a10f6140-aba5-4bda-8662-a951d5daa779.xhtml), *Standard Qt Widgets*, covers standard UI elements and dynamic layouts to teach the reader how to handle orientation changes.

[Chapter 2](ab2105ed-232b-4c99-8fd8-4ca295f6f5f9.xhtml), *Fluid UI with Qt Quick*, outlines standard QML elements, charts, data visualization, and animation.

[Chapter 3](ae64f3a5-9f62-442b-b665-58bdb3f3b06d.xhtml), *Graphical and Special Effects*, looks at QML particles and graphical effects.

[Chapter 4](b9dbf233-aa2a-4f5c-ae51-dfa7218b3f94.xhtml), *Input and Touch*, teaches the readers how to create and use virtual keyboards, and touch and voice input.

[Chapter 5](997cf699-f3e3-4906-8168-57d081f22b76.xhtml), *Qt Network for Communication*, talks the reader through network operations, sockets, and sessions.

[Chapter 6](b5f09986-608e-4109-9f27-1444d086852f.xhtml), *Connectivity with Qt Bluetooth LE*, goes over Bluetooth LE device discovery, setting up a service, and manipulating characteristics.

[Chapter 7](0a6e358d-e771-458e-b68f-380149f259a0.xhtml), *Machines Talking*, discusses sensors, WebSockets, and MQTT automation.

[Chapter 8](ce267d05-6d92-471e-945b-4e9b8ed091ec.xhtml), *Where Am I? Location and Positioning*, looks at GPS location, positioning, and maps.

[Chapter 9](03c9a3bc-4baf-4494-a9b2-2f86bd1dcdff.xhtml), *Sounds and Visions – Qt Multimedia*, covers 3D audio, FM radio, and recording and playing video.

[Chapter 10](0b426c47-30a5-499e-96c8-33eb539b38f8.xhtml), *Remote Databases with Qt SQL*, outlines, remote use of SQLite and MySQL databases.

[Chapter 11](6c506043-2533-4026-bd92-a1848b6c6d87.xhtml), *Enabling In-App Purchases with Qt Purchasing*, discusses adding in-app purchases to your apps.

[Chapter 12](143c9219-edf3-4886-aadb-41d91691b2f5.xhtml), *Cross Compiling and Remote Debugging*, looks at cross-compiling, and connecting to and debugging on a remote device.

[Chapter 13](d64b8be7-517b-4a07-951b-eeecc1793754.xhtml), *Deploying to Mobile and Embedded*, examines the setting up of an application and completion of app store procedures.

[Chapter 14](04b4eb0e-2f09-4205-9d2f-ac17ff6a958d.xhtml), *Universal Platform for Mobiles and Embedded Devices*, looks at running Qt apps in a web browser.

[Chapter 15](590553c7-965b-4002-bbfe-fd61e30ce5a8.xhtml), *Building a Linux System*, covers setting up and building a complete Linux embedded system.

# To get the most out of this book

This book assumes you have used C++, are familiar with QML, can use Git to download source code, and can type commands into a command-line interface. You should also be accustomed to using GDB debugger.

It also assumes you have a mobile or embedded device, such as a phone or Raspberry Pi.

# Download the example code files

You can download the example code files for this book from your account at [www.packt.com](http://www.packt.com). If you purchased this book elsewhere, you can visit [www.packt.com/support](http://www.packt.com/support) and register to have the files emailed directly to you.

You can download the code files by following these steps:

1.  Log in or register at [www.packt.com](http://www.packt.com).
2.  Select the SUPPORT tab.
3.  Click on Code Downloads & Errata.
4.  Enter the name of the book in the Search box and follow the onscreen instructions.

Once the file is downloaded, please make sure that you unzip or extract the folder using the latest version of:

*   WinRAR/7-Zip for Windows
*   Zipeg/iZip/UnRarX for Mac
*   7-Zip/PeaZip for Linux

The code bundle for the book is also hosted on GitHub at [https://github.com/PacktPublishing/Hands-On-Mobile-and-Embedded-Development-with-Qt-5/tree/master](https://github.com/PacktPublishing/Hands-On-Mobile-and-Embedded-Development-with-Qt-5/tree/master). In case there's an update to the code, it will be updated on the existing GitHub repository.

We also have other code bundles from our rich catalog of books and videos available at **[https://github.com/PacktPublishing/](https://github.com/PacktPublishing/)**. Check them out!

# Download the color images

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. You can download it here: [https://www.packtpub.com/sites/default/files/downloads/9781789614817_ColorImages.pdf](https://www.packtpub.com/sites/default/files/downloads/9781789614817_ColorImages.pdf).

# Conventions used

There are a number of text conventions used throughout this book.

`CodeInText`: Indicates code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles. Here is an example: "On *iOS*, you need to edit the `plist.info` file."

A block of code is set as follows:

```cpp
if (!QTouchScreen::devices().isEmpty()) {
   qApp->setStyleSheet("QButton {padding: 10px;}");
}
```

**Bold**: Indicates a new term, an important word, or words that you see onscreen. For example, words in menus or dialog boxes appear in the text like this. Here is an example: "Select the Projects icon on the left side of Qt Creator, then pick which target platform you want like Qt 5.12.0 for iOS"

Warnings or important notes appear like this.

Tips and tricks appear like this.

# Get in touch

Feedback from our readers is always welcome.

**General feedback**: If you have questions about any aspect of this book, mention the book title in the subject of your message and email us at `customercare@packtpub.com`.

**Errata**: Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you have found a mistake in this book, we would be grateful if you would report this to us. Please visit [www.packt.com/submit-errata](http://www.packt.com/submit-errata), selecting your book, clicking on the Errata Submission Form link, and entering the details.

**Piracy**: If you come across any illegal copies of our works in any form on the Internet, we would be grateful if you would provide us with the location address or website name. Please contact us at `copyright@packt.com` with a link to the material.

**If you are interested in becoming an author**: If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, please visit [authors.packtpub.com](http://authors.packtpub.com/).

# Reviews

Please leave a review. Once you have read and used this book, why not leave a review on the site that you purchased it from? Potential readers can then see and use your unbiased opinion to make purchase decisions, we at Packt can understand what you think about our products, and our authors can see your feedback on their book. Thank you!

For more information about Packt, please visit [packt.com](http://www.packt.com/).