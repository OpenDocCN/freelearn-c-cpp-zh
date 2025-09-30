# Preface

Creating games in C++ is a complicated process requiring a lot of time and dedication to achieve results. A good foundation of reusable classes can speed up development time and allow focus to be on creating a great game rather than struggling with low-level code. This book aims to show an approach to creating a reusable framework that could be used for any game, whether 2D or 3D.

# What this book covers

[Chapter 1](ch01.html "Chapter 1. Getting Started with SDL"), *Getting started with SDL*, covers setting up SDL in Visual C++ 2010 express and then moves onto the basics of SDL including creating a window and listening for quit events.

[Chapter 2](ch02.html "Chapter 2. Drawing in SDL"), *Drawing in SDL*, covers the development of some core drawing classes to help simplify SDL rendering. The `SDL_image` extension is also introduced to allow the loading of a variety of different image file types.

[Chapter 3](ch03.html "Chapter 3. Working with Game Objects"), *Working with Game Objects*, gives a basic introduction to inheritance and polymorphism along with the development of a reusable `GameObject` class that will be used throughout the rest of the book.

[Chapter 4](ch04.html "Chapter 4. Exploring Movement and Input Handling"), *Exploring Movement and Input Handling*, gives a detailed look at handling events in SDL. Joystick, keyboard, and mouse input are all covered with the development of reusable classes.

[Chapter 5](ch05.html "Chapter 5. Handling Game States"), *Handling Game States*, covers the design and implementation of a finite state machine to manage game states. Implementing and moving between different states is covered in detail.

[Chapter 6](ch06.html "Chapter 6. Data-driven Design"), *Data-driven Design*, covers the use of TinyXML to load states. A class to parse states is developed along with examples for different states.

[Chapter 7](ch07.html "Chapter 7. Creating and Displaying Tile Maps"), *Creating and Displaying Tile Maps*, brings together everything from the previous chapters to allow the creation of levels using the Tiled map editor. A level parsing class is created to load maps from an XML file.

[Chapter 8](ch08.html "Chapter 8. Creating Alien Attack"), *Creating Alien Attack*, covers the creation of a 2D side scrolling shooter, utilizing everything learned in the previous chapters.

[Chapter 9](ch09.html "Chapter 9. Creating Conan the Caveman"), *Creating Conan the Caveman*, covers the creation of a second game, altering the code from Alien Attack, showing that the framework is flexible enough to be used for any 2D game genre.

# What you need for this book

To use this book you will need the following software:

*   Visual C++ 2010 Express
*   Tiled map editor
*   TinyXML
*   zlib library

# Who this book is for

This book is aimed at beginner/intermediate C++ programmers who want to take their existing skills and apply them to creating games in C++. This is not a beginner's book and you are expected to know the basics of C++, including inheritance, polymorphism, and class design.

# Conventions

In this book, you will find a number of styles of text that distinguish between different kinds of information. Here are some examples of these styles, and an explanation of their meaning.

Code words in text are shown as follows: "We can include other contexts through the use of the `include` directive."

A block of code is set as follows:

[PRE0]

**New terms** and **important words** are shown in bold. Words that you see on the screen, in menus or dialog boxes for example, appear in the text like this: "Right-click on the project and choose **Build**.".

### Note

Warnings or important notes appear in a box like this.

### Tip

Tips and tricks appear like this.

# Reader feedback

Feedback from our readers is always welcome. Let us know what you think about this book—what you liked or may have disliked. Reader feedback is important for us to develop titles that you really get the most out of.

To send us general feedback, simply send an e-mail to `<[feedback@packtpub.com](mailto:feedback@packtpub.com)>`, and mention the book title via the subject of your message.

If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, see our author guide on [www.packtpub.com/authors](http://www.packtpub.com/authors).

# Customer support

Now that you are the proud owner of a Packt book, we have a number of things to help you to get the most from your purchase.

## Downloading the example code

You can download the example code files for all Packt books you have purchased from your account at [http://www.packtpub.com](http://www.packtpub.com). If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files e-mailed directly to you.

## Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books—maybe a mistake in the text or the code—we would be grateful if you would report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the **errata submission form** link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded on our website, or added to any list of existing errata, under the Errata section of that title. Any existing errata can be viewed by selecting your title from [http://www.packtpub.com/support](http://www.packtpub.com/support).

## Piracy

Piracy of copyright material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works, in any form, on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at `<[copyright@packtpub.com](mailto:copyright@packtpub.com)>` with a link to the suspected pirated material.

We appreciate your help in protecting our authors, and our ability to bring you valuable content.

## Questions

You can contact us at `<[questions@packtpub.com](mailto:questions@packtpub.com)>` if you are having a problem with any aspect of the book, and we will do our best to address it.