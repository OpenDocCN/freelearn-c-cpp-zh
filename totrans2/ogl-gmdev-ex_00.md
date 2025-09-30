# Preface

Welcome to *OpenGL Game Development Blueprints*! We are excited that you chose this book as your guide to both OpenGL and game development. This section will provide you with a brief preview of each chapter, followed by the technologies that are required to complete the work that is presented in the book. Finally, we will discuss the target audience for this book so that you will know whether this book is right for you.

# What this book covers

[Chapter 1](ch01.html "Chapter 1. Building the Foundation"), *Building the Foundation*, guides you through creating the code framework for the game. Games use a particular structure that is known as the *game loop*. By the end of this chapter, you will understand and have created the game loop for the game as well as initialized the required OpenGL elements.

[Chapter 2](ch02.html "Chapter 2. Your Point of View"), *Your Point of View*, introduces you to the first project in the book—creating a 2D platform game. The first step in this project will be to define the type of view that is required by OpenGL, and render the background of the game.

[Chapter 3](ch03.html "Chapter 3. A Matter of Character"), *A Matter of Character*, covers the creation of sprites that move on the screen. 2D frame-based animations are the core of any 2D game, and you will learn how to create simple graphics and render them to the screen.

[Chapter 4](ch04.html "Chapter 4. Control Freak"), *Control Freak*, teaches you how to build an input system that will allow you to control the main character and other aspects of the game. You will also create a basic user interface that allows you to start the game and navigate to various options.

[Chapter 5](ch05.html "Chapter 5. Hit and Run"), *Hit and Run*, covers collision detection. You will learn how to stop the character from falling through the ground, how to land on objects, and how to detect whether enemies have hit you or have been hit by player weapons. By the end of this chapter, you will be able to play the game for the first time.

[Chapter 6](ch06.html "Chapter 6. Polishing the Silver"), *Polishing the Silver*, covers the topics that make a game presentable (but are often overlooked by novice developers). You will learn how to implement a scoring system, game over and game won scenarios, and simple level progression. This chapter will conclude the 2D project of the book.

[Chapter 7](ch07.html "Chapter 7. Audio Adrenaline"), *Audio Adrenaline*, guides you through implementing sound effects and music in the game. We will provide links to some audio files that you can use in your game.

[Chapter 8](ch08.html "Chapter 8. Expanding Your Horizons"), *Expanding Your Horizons*, will start the second project of the book—a 3D first-person space shooter. At the end of this chapter you will have created a new project, starting the framework for a 3D game.

[Chapter 9](ch09.html "Chapter 9. Super Models"), *Super Models*, introduces you to the concepts of 3D art and modeling, and then guides you through the process of loading 3D models into the game environment. Although you will be able try your hand at creating a 3D model, the resources that are required for the game will be provided online.

[Chapter 10](ch10.html "Chapter 10. Expanding Space"), *Expanding Space*, expands on many of the concepts that were covered in the 2D segment of the book and applies them to a 3D world. Movement and collision detection are revamped to take this new dimension into consideration. An input scheme to move in 3D space is implemented. By the end of this chapter, you will be able to control a 3D model in 3D space.

[Chapter 11](ch11.html "Chapter 11. Heads Up"), *Heads Up*, guides you through creating a 2D user interface on top of the 3D world. You will create a menu system to start and end the game, as well as a heads-up-display (HUD) that shows the score and stats in game. By the end of this chapter, you will have created a playable 3D shooter game.

[Chapter 12](ch12.html "Chapter 12. Conquer the Universe"), *Conquer the Universe*, introduces you to some of the more advanced concepts that were beyond the scope of the book, and it gives you some direction to advance your skills.

# What you need for this book

Each chapter in the book will have exercises that you will need to code. Each exercise is a building block toward creating your first game using OpenGL. It is vitally important that you actually write the code. In our experience, you can't learn any kind of computer programming without actually writing code. Don't just read the book, do the book!

The first chapter of the book will go through the details of setting up a development environment so that you can code the examples in the book. In general, you will need the following:

*   **A Windows-based personal computer**: You could use a Mac, but the examples used in the book are based on a Windows 10 operating system.
*   **A copy of Visual Studio**: We will show you how to obtain and install this for free in chapter one, or you can go to [http://www.visualstudio.com/downloads/download-visual-studio-vs](http://www.visualstudio.com/downloads/download-visual-studio-vs) right now. Again, you could use another development tool and compiler, but you are on your own to set it up.
*   **A 2D image editor program**: We recommend GIMP, which you can download for free at [http://www.gimp.org/](http://www.gimp.org/).
*   **A 3D modeling program**: We recommend Blender, which you can download for free at [http://www.blender.org/](http://www.blender.org/).
*   **An Internet connection**: You could complete the exercises without this, but an Internet connection is very useful for looking up additional resources.
*   Some free time and dedication!

That's it! The good news is that as long as you have a personal computer, the technology and tools that are used to create games using OpenGL are completely free!

# Who this book is for

If you are reading this book, it is pretty obvious that you are interested in game development. You have either heard of OpenGL or perhaps even used it, and you want to learn more. Finally, you are already a programmer in some computer language or you want to be.

Does this sound like you? Read on!

This book assumes that you have some familiarity with computer programming in the C++ computer language. If you have programmed in some other language, such as C#, Java, JavaScript, or PHP, then you are pretty familiar with the constructs of the C++ language. Nevertheless, if have never programmed in C++ then you may need to brush up on your skills. You can try *Microsoft Visual C++ Windows Applications by Example*, also published by *Packt Publishing*. If you feel comfortable with programming in general, but have not coded in C++, you can look at the free online C++ tutorials at [http://www.cplusplus.com/doc/tutorial/](http://www.cplusplus.com/doc/tutorial/).

We don't assume that you have any knowledge of OpenGL—that is what this book is going to give you. We start by explaining the basic concepts of OpenGL and move through more advanced concepts by example. As you learn, you will also code, providing you with the opportunity to put what you have learned into practice. This book won't make you an OpenGL expert overnight, but it will give you the foundation to understand and use OpenGL. At the end of this book, we will give you some pointers to other resources that will allow you to learn even more about OpenGL.

We also don't assume that you have any experience developing games. This book is rather unique in that it provides you with a primer to learn OpenGL and a primer to learn game development. There are many books out there that teach OpenGL, but most do so within a more academic or theoretical framework. We felt that it was better to teach you OpenGL while you were using it to create an actual game. Actually, you will code two games: one in 2D, and one in 3D. Two for the price of one!

# Conventions

In this book, you will find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "As usual, change the middle line in update to call `drawQuad`."

A block of code is set as follows:

[PRE0]

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

[PRE1]

**New terms** and **important words** are shown in bold. Words that you see on the screen, for example, in menus or dialog boxes, appear in the text like this: "For the **Configuration** drop-down box, make sure you select **All Configurations**."

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