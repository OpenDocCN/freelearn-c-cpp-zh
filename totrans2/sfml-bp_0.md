# Preface

Throughout this book, I'll try to share my knowledge on how to make video games and share them with you. Five different projects will be covered, which include many techniques and ways to resolve quite commons problems involved in game development.

The technologies used are the C++ programming language (2011 standard) and the SFML library (version 2.2).

Many aspects of game programming are developed over the different chapters and give you all the keys in hand to build every kind of game you want in 2D, with the only limit of your imagination.

# What this book covers

[Chapter 1](ch01.html "Chapter 1. Preparing the Environment"), *Preparing the Environment*, helps you install everything needed for this book, and build a small application using SFML to test whether everything is fine.

[Chapter 2](ch02.html "Chapter 2. General Game Architecture, User Inputs, and Resource Management"), *General Game Architecture, User Inputs, and Resource Management*, explains general game architectures, managing user inputs and finally, how to keep track of external resources.

[Chapter 3](ch03.html "Chapter 3. Making an Entire 2D Game"), *Making an Entire 2D Game*, helps you build Asteroid and Tetris clones, learning entity models and board management.

[Chapter 4](ch04.html "Chapter 4. Playing with Physics"), *Playing with Physics*, provides a description of physical engines. It also covers the usage of Box2D paired with SFML, and turns our Tetris into a new game, Gravitris.

[Chapter 5](ch05.html "Chapter 5. Playing with User Interfaces"), *Playing with User Interfaces*, helps you create and use a game user interface. It introductes you to SFGUI and adding them to our Gravitris game.

[Chapter 6](ch06.html "Chapter 6. Boost Your Code Using Multithreading"), *Boost Your Code Using Multithreading*, introduces multithreading and adapts our game to use it.

[Chapter 7](ch07.html "Chapter 7. Building a Real-time Tower Defense Game from Scratch – Part 1"), *Building a Real-time Tower Defense Game from Scratch – Part 1*, helps you create animations, a generic tile map system (isometric hexagonal tiles), and an entity system. Finally, you will create all the game logic.

[Chapter 8](ch08.html "Chapter 8. Build a Real-time Tower Defense Game from Scratch – Part 2, Networking"), *Build a Real-time Tower Defense Game from Scratch – Part 2, Networking*, introduces network architectures and networking. It helps you create a custom communication protocol, and modify our game to allow multiplayer matches over the network. Then, we finally add a save/load option to our game using Sqlite3 through an ORM.

# What you need for this book

To be able to build the projects covered throughout this book, you are assumed to have knowledge of the C++ language with its basic features, and also parts of the standard template library, such as strings, streams, and containers. It's important to keep in mind that game development is not an easy task, so if you don't have the prerequisites, it can get frustrating. So, don't hesitate to read some books or tutorials on C++ before starting with this one.

# Who this book is for

This book is for developers who know the basics of the SFML library and its capabilities for 2D game development. Minimal experience with C++ is required.

# Conventions

In this book, you will find a number of styles of text that distinguish between different kinds of information. Here are some examples of these styles, and an explanation of their meaning.

Code words in text, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "We also add the point calculation to this class with the `addLines()` function."

A block of code is set as follows:

[PRE0]

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

[PRE1]

Any command-line input or output is written as follows:

[PRE2]

**New terms** and **important words** are shown in bold. Words that you see on the screen, in menus or dialog boxes for example, appear in the text like this: "We will also use this class to display the **Game Over** message if it's needed".

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

You can download the example code files from your account at [http://www.packtpub.com](http://www.packtpub.com) for all the Packt Publishing books you have purchased. If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files e-mailed directly to you.

## Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books—maybe a mistake in the text or the code—we would be grateful if you could report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the **Errata Submission Form** link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded to our website or added to any list of existing errata under the Errata section of that title.

To view the previously submitted errata, go to [https://www.packtpub.com/books/content/support](https://www.packtpub.com/books/content/support) and enter the name of the book in the search field. The required information will appear under the **Errata** section.

## Piracy

Piracy of copyright material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works, in any form, on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at `<[copyright@packtpub.com](mailto:copyright@packtpub.com)>` with a link to the suspected pirated material.

We appreciate your help in protecting our authors, and our ability to bring you valuable content.

## Questions

You can contact us at `<[questions@packtpub.com](mailto:questions@packtpub.com)>` if you are having a problem with any aspect of the book, and we will do our best to address it.