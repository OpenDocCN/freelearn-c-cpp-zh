# Preface

Game development is one of the most interesting career choices to date. Apart from the many other fields that are incorporated in this process, it's also a realm where pure imagination comes to life. Even during the times when one may think that there's nothing new under the sun, ground-breaking ideas are still cemented in this medium, both as revolutionary milestones and exciting adventures that will make us feel child-like excitement yet again.

Getting started with game programming is easier now than ever before! Documentation and tutorials aside, there even exist enthusiasts out there who actually put together libraries of code that can be used to eliminate the redundant or difficult parts of building different types of applications. As it so happens, one of these libraries is titled "Simple and Fast Multimedia Library", and it is the focal point of this publication.

Throughout the course of this book, three projects are built from scratch, with each one having increased complexity compared to its preceding project. We will start with a basic clone of the classical arcade game—*Snake*, which introduces the basics of SFML and some of the framework that is going to persist until the very end. As difficult subjects are addressed, we will begin to cobble the second project together, turning it into a side-scrolling platformer. The remaining chapters of this book focus on building and polishing an online RPG-style game that can be played with your friends! No detail of any of these projects will remain undiscussed, as you will be guided through the entire process of planning and implementing every single aspect of these projects.

If the vast array of features that need to be worked on hasn't scared you away yet, congratulations! You are about to embark on a journey of tremendous proportions. So don't let the odds intimidate you. We hope to see you at the finish line!

# What this book covers

[Chapter 1](ch01.html "Chapter 1. It's Alive! It's Alive! – Setup and First Program"), *It's Alive! It's Alive! – Setup and First Program*, covers the fundamentals that are necessary in order to build basic SFML applications.

[Chapter 2](ch02.html "Chapter 2. Give It Some Structure – Building the Game Framework"), *Give It Some Structure – Building the Game Framework*, introduces a better framework for the applications that will be used throughout the book. It also covers the basics of timing in video games.

[Chapter 3](ch03.html "Chapter 3. Get Your Hands Dirty – What You Need to Know"), *Get Your Hands Dirty – What You Need to Know*, helps solidify all the information from the previous chapters by finishing our first game project.

[Chapter 4](ch04.html "Chapter 4. Grab That Joystick – Input and Event Management"), *Grab That Joystick – Input and Event Management*, elaborates on the process of obtaining a window event and peripheral information as well as using it in an automated way.

[Chapter 5](ch05.html "Chapter 5. Can I Pause This? – Application States"), *Can I Pause This? – Application States*, addresses the issue of state switching and blending using a state machine.

[Chapter 6](ch06.html "Chapter 6. Set It in Motion! – Animating and Moving around Your World"), *Set It in Motion! – Animating and Moving around Your World*, deals with the issues of screen scrolling and resource management as well as the usage and animation of sprite sheets.

[Chapter 7](ch07.html "Chapter 7. Rediscovering Fire – Common Game Design Elements"), *Rediscovering Fire – Common Game Design Elements*, wraps up the second project of the book by dealing with entity management, tile-maps, and collision.

[Chapter 8](ch08.html "Chapter 8. The More You Know – Common Game Programming Patterns"), *The More You Know – Common Game Programming Patterns*, introduces the third project of the book by covering the fundamentals of a few common programming patterns, including the entity component system.

[Chapter 9](ch09.html "Chapter 9. A Breath of Fresh Air – Entity Component System Continued"), *A Breath of Fresh Air – Entity Component System Continued*, focuses on building common game functionality by breaking it down to its components and systems.

[Chapter 10](ch10.html "Chapter 10. Can I Click This? – GUI Fundamentals"), *Can I Click This? – GUI Fundamentals*, breaks down how a graphical user interface can be implemented using the fundamental data types.

[Chapter 11](ch11.html "Chapter 11. Don't Touch the Red Button! – Implementing the GUI"), *Don't Touch the Red Button! – Implementing the GUI*, picks up where the previous chapter left off and wraps up the implementation of a GUI system. We also discuss three basic element types.

[Chapter 12](ch12.html "Chapter 12. Can You Hear Me Now? – Sound and Music"), *Can You Hear Me Now? – Sound and Music*, livens up the third project of the book by bringing entity sounds and music to the table.

[Chapter 13](ch13.html "Chapter 13. We Have Contact! – Networking Basics"), *We Have Contact! – Networking Basics*, covers all the basics that are required in order to implement networking in our final project.

[Chapter 14](ch14.html "Chapter 14. Come Play with Us! – Multiplayer Subtleties"), *Come Play with Us! – Multiplayer Subtleties*, transforms the final project of the book into a multiplayer RPG-style death match with the application of a client-server network model as well as a combat system.

# What you need for this book

Given that this book covers the SFML library, it's necessary to have it downloaded and set up. [Chapter 1](ch01.html "Chapter 1. It's Alive! It's Alive! – Setup and First Program"), *It's Alive! It's Alive! – Setup and First Program* covers this process step by step.

Additionally, a compiler or an IDE that supports *C++11* is needed in order to compile the code that we're about to write. The code for the book has been written on and compiled with the *Microsoft Visual Studio 2013* IDE on a system that runs *Windows 7*.

# Who this book is for

This book is intended for game development enthusiasts who have at least a decent knowledge of the C++ programming language and an optional background in game design.

# Conventions

In this book, you will find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "We can include other contexts through the use of the `include` directive."

A block of code is set as follows:

[PRE0]

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

[PRE1]

**New terms** and **important words** are shown in bold. Words that you see on the screen, for example, in menus or dialog boxes, appear in the text like this: "Navigate to the **VC++ Directories** underneath **Configuration Properties** by right clicking on our project and selecting **Properties**."

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