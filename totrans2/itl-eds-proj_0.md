# Preface

*Intel Edison Projects* is meant for beginners who want to get to grips with the Intel Edison and explore its features. Intel Edison is an embedded computing platform, which allows us to explore areas of IoT, embedded systems, and robotics.

This book takes you through various concepts and each chapter has a project that can be performed by you. It covers multiple topics, including sensor data acquisition and pushing it to the cloud to control devices over the Internet, as well as topics ranging from image processing to both autonomous and manual robotics.

In every chapter, the book first covers some theoretical aspects of the topic, which include some small chunks of code and a minimal hardware  setup. The rest of the chapter is dedicated to the practical aspects of the project.

The projects discussed in this book wherever possible require only minimal hardware, and the projects in each chapter are included to make sure that you understand the basics.

# What this book covers

[Chapter 1](c225d705-919a-4442-adc8-7b22d33437fc.xhtml), *Setting up Intel Edison*, covers the initial steps of setting up the Intel Edison, flashing it, and setting up the environment for development.

[Chapter 2](bada9944-ec60-4e8f-8d88-0085dd1c8210.xhtml), *Weather Station (IoT)*, introduces you to IoT and uses a simple case of a weather station where we use temperature, smoke level, and sound level and push data to the cloud to visualize it.

[Chapter 3](3bd53219-a287-4d8f-9a58-5a06c5b14062.xhtml), *Intel Edison and IoT (Home Automation)*, covers a case for home automation, where we are controlling electrical load using the Intel Edison.

[Chapter 4](3fa86b30-3d51-4628-a827-db7b3e31f3e7.xhtml), *Intel Edison and Security System*, covers voice and image processing for the Intel Edison.

[Chapter 5](45fccd6a-a75e-465d-89dc-dad31f528ac1.xhtml), *Autonomous Robotics with Intel Edison*, explores the field of robotics, where we develop a line-following robot using the Intel Edison and related algorithms.

[Chapter 6](56a788c0-bef4-43e2-91cb-02b2d981c5c0.xhtml), *Manual Robotics with Intel Edison*, explores UGVs and also guides you through the process of developing controller software.

# What you need for this book

The mandatory prerequisites for this book are the Intel Edison with Windows/Linux/Mac OS. The software requirements are as follows:

*   Arduino IDE
*   Visual Studio
*   FileZilla
*   Notepad++
*   PuTTY
*   Intel XDK

# Who this book is for

If you are a hobbyist, robot engineer, IoT enthusiast, programmer, or developer who wants to create autonomous projects with the Intel Edison, then this book is for you. Prior programming knowledge would be beneficial.

# Conventions

In this book, you will find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "We can include other contexts through the use of the `include` directive."

A block of code is set as follows:

[PRE0]

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

[PRE1]

Any command-line input or output is written as follows:

[PRE2]

**New terms** and **important words** are shown in bold. Words that you see on the screen, for example, in menus or dialog boxes, appear in the text like this: "After you click on OK, the tool will automatically unzip the file."

Warnings or important notes appear in a box like this.

Tips and tricks appear like this.

# Reader feedback

Feedback from our readers is always welcome. Let us know what you think about this book-what you liked or disliked. Reader feedback is important for us as it helps us develop titles that you will really get the most out of.

To send us general feedback, simply e-mail `feedback@packtpub.com`, and mention the book's title in the subject of your message.

If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, see our author guide at [www.packtpub.com/authors](http://www.packtpub.com/authors).

# Customer support

Now that you are the proud owner of a Packt book, we have a number of things to help you to get the most from your purchase.

# Downloading the example code

You can download the example code files for this book from your account at [http://www.packtpub.com](http://www.packtpub.com). If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files e-mailed directly to you.

You can download the code files by following these steps:

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

The code bundle for the book is also hosted on GitHub at [https://github.com/PacktPublishing/Intel-Edison-Projects](https://github.com/PacktPublishing/Intel-Edison-Projects). We also have other code bundles from our rich catalog of books and videos available at [https://github.com/PacktPublishing/](https://github.com/PacktPublishing/). Check them out!

# Downloading the color images of this book

We also provide you with a PDF file that has color images of the screenshots/diagrams used in this book. The color images will help you better understand the changes in the output. You can download this file from [https://www.packtpub.com/sites/default/files/downloads/IntelEdisonProjects_ColorImages.pdf](https://www.packtpub.com/sites/default/files/downloads/IntelEdisonProjects_ColorImages.pdf).

# Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books-maybe a mistake in the text or the code-we would be grateful if you could report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the Errata Submission Form link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded to our website or added to any list of existing errata under the Errata section of that title.

To view the previously submitted errata, go to [https://www.packtpub.com/books/content/support](https://www.packtpub.com/books/content/support) and enter the name of the book in the search field. The required information will appear under the Errata section.

# Piracy

Piracy of copyrighted material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works in any form on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at `copyright@packtpub.com` with a link to the suspected pirated material.

We appreciate your help in protecting our authors and our ability to bring you valuable content.

# Questions

If you have a problem with any aspect of this book, you can contact us at `questions@packtpub.com`, and we will do our best to address the problem.