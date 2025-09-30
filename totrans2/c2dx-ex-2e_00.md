# Preface

Cocos2d-x combines the benefits of using one the most popular and test-proven 2D game frameworks out there with the power and portability of C++. So, you get the best deal possible. Not only is the framework built to be easy to use and quick to implement, it also allows your code to target more than one system.

The book will show you how to use the framework to quickly implement your ideas, and let Cocos2d-x help you with the translation of all that OpenGL gobbledygook, leaving you with the fun part: making sprites jump around and hit each other!

There are seven examples of games in this book with two of them being physics-based, using Box2D and one using the Lua bindings and the new Cocos Code IDE. With each example, you'll learn more about the framework and the magical lines that can quickly add particle effects, animations, sounds, UI elements, and all sorts of wonderful things to your games.

Not only this, but you will also learn how to target both iOS and Android devices, and multiple screen sizes.

# What this book covers

[Chapter 1](part0016_split_000.html#page "Chapter 1. Installing Cocos2d-x"), *Installing Cocos2d-x*, guides you through the download and installation of the Cocos2d-x framework. It also examines the ins and outs of a basic Cocos2d-x application and deployment to both iOS and Android devices.

[Chapter 2](part0026_split_000.html#page "Chapter 2. You Plus C++ Plus Cocos2d-x"), *You Plus C++ Plus Cocos2d-x*, explains the main elements in the framework. It also covers the differences in syntax when developing in C++, and the differences in memory management when developing with Cocos2d-x.

[Chapter 3](part0034_split_000.html#page "Chapter 3. Your First Game – Air Hockey"), *Your First Game – Air Hockey*, kick-starts our game development tutorials by using Cocos2d-x to build an air hockey game. You will learn how to load the images for your sprites, display text, manage touches, and add sounds to your game.

[Chapter 4](part0053_split_000.html#page "Chapter 4. Fun with Sprites – Sky Defense"), *Fun with Sprites – Sky Defense*, demonstrates the power of actions in Cocos2d-x, and shows how an entire game could be built with them. It also introduces the concept of sprite sheets and the steps to build a universal application targeting different screen resolutions.

[Chapter 5](part0072_split_000.html#page "Chapter 5. On the Line – Rocket Through"), *On the Line – Rocket Through*, adds two new elements to our game development toolbox: how to draw primitives, such as lines, curves, and circles, and how to use particle systems to improve the look of our game with special effects.

[Chapter 6](part0087_split_000.html#page "Chapter 6. Quick and Easy Sprite – Victorian Rush Hour"), *Quick and Easy Sprite – Victorian Rush Hour*, shows how you can use Cocos2d-x to quickly implement game ideas for further testing and development by rapidly building game prototypes with placeholder sprites. In the game example used for this chapter, you'll also learn how to build a side-scrolling platform game.

[Chapter 7](part0105_split_000.html#page "Chapter 7. Adding the Looks – Victorian Rush Hour"), *Adding the Looks – Victorian Rush Hour*, continues with the project from the previous chapter adding the final touches to the game including a menu and a playable tutorial.

[Chapter 8](part0117_split_000.html#page "Chapter 8. Getting Physical – Box2D"), *Getting Physical – Box2D*, introduces the popular Box2D API for a physics simulation, guiding you through the process of using Box2D in the development of a pool game. You learn how to create bodies and manage the way they interact with each other.

[Chapter 9](part0126_split_000.html#page "Chapter 9. On the Level – Eskimo"), *On the Level – Eskimo*, teaches you how to load external data for game levels, how to store game-related data locally as well as structure your games with multiple scenes. We use a second Box2D game to illustrate these topics, plus a couple of new concepts, such as using the event dispatcher to structure your games better.

[Chapter 10](part0144_split_000.html#page "Chapter 10. Introducing Lua!"), *Introducing Lua!*, will guide you in the development of a multiplatform match-three game using Lua and the new Cocos Code IDE. You will see how similar the calls are between the C++ version and its Lua bindings and how easy it is to develop a game in Lua.

[Appendix A](part0157_split_000.html#page "Appendix A. Vector Calculations with Cocos2d-x"), *Vector Calculations with Cocos2d-x*, covers some of the math concepts used in [Chapter 5](part0072_split_000.html#page "Chapter 5. On the Line – Rocket Through"), *On the Line – Rocket Through*, in a little more detail.

[Appendix B](part0159_split_000.html#page "Appendix B. Pop Quiz Answers"), *Pop Quiz Answers*, provides answers to the pop quiz available in some chapters.

# What you need for this book

In order to run the games developed in this book, you will need Xcode for iOS devices, and Eclipse for Android, as well as the Cocos Code IDE for the Lua game. Although the tutorials describe the development process using Xcode in each chapter of the book, you will see how to import the code in Eclipse and develop and deploy from there.

# Who this book is for

You have a passion for games. You may have used Cocos2d already (the Objective-C version of the framework) and are eager to learn its C++ port. Or, you know a little bit of some other C-based language, such as Java, PHP, or Objective-C and you want to learn how to develop 2D games in C++. You may also be a C++ developer already and want to know what all the hoopla about Cocos2d-x is. If you fit any of these scenarios, welcome aboard!

# Sections

In this book, you will find several headings that appear frequently (Time for action, What just happened?, Pop quiz, and Have a go hero).

To give clear instructions on how to complete a procedure or task, we use these sections as follows:

# Time for action – heading

1.  Action 1
2.  Action 2
3.  Action 3

Instructions often need some extra explanation to ensure they make sense, so they are followed with these sections:

## *What just happened?*

This section explains the working of the tasks or instructions that you have just completed.

You will also find some other learning aids in the book, for example:

## Pop quiz – heading

These are short multiple-choice questions intended to help you test your own understanding.

## Have a go hero – heading

These are practical challenges that give you ideas to experiment with what you have learned.

# Conventions

You will also find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "For background music volume, you must use `setBackgroundMusicVolume`."

A block of code is set as follows:

[PRE0]

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

[PRE1]

Any command-line input or output is written as follows:

[PRE2]

**New terms** and **important words** are shown in bold. Words that you see on the screen, in menus or dialog boxes for example, appear in the text like this: "In the dialog box select **cocos2d-x** under the **iOS** menu and choose the **cocos2dx** template."

### Note

Warnings or important notes appear in a box like this.

### Tip

Tips and tricks appear like this.

# Reader feedback

Feedback from our readers is always welcome. Let us know what you think about this book—what you liked or may have disliked. Reader feedback is important for us to develop titles that you really get the most out of.

To send us general feedback, simply send an e-mail to `<[feedback@packtpub.com](mailto:feedback@packtpub.com)>`, and mention the book title through the subject of your message.

If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, see our author guide on [www.packtpub.com/authors](http://www.packtpub.com/authors).

# Customer support

Now that you are the proud owner of a Packt book, we have a number of things to help you to get the most from your purchase.

## Downloading the example code

You can download the example code files for all Packt books you have purchased from your account at [http://www.packtpub.com](http://www.packtpub.com). If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files e-mailed directly to you.

## Downloading the color images of this book

We also provide you with a PDF file that has color images of the screenshots/diagrams used in this book. The color images will help you better understand the changes in the output. You can download this file from [https://www.packtpub.com/sites/default/files/downloads/8852OS_GraphicsBundle.pdf](https://www.packtpub.com/sites/default/files/downloads/8852OS_GraphicsBundle.pdf).

## Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books—maybe a mistake in the text or the code—we would be grateful if you could report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the **Errata** **Submission** **Form** link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded to our website or added to any list of existing errata under the Errata section of that title.

To view the previously submitted errata, go to [https://www.packtpub.com/books/content/support](https://www.packtpub.com/books/content/support) and enter the name of the book in the search field. The required information will appear under the **Errata** section.

## Piracy

Piracy of copyrighted material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works in any form on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at `<[copyright@packtpub.com](mailto:copyright@packtpub.com)>` with a link to the suspected pirated material.

We appreciate your help in protecting our authors and our ability to bring you valuable content.

## Questions

If you have a problem with any aspect of this book, you can contact us at `<[questions@packtpub.com](mailto:questions@packtpub.com)>`, and we will do our best to address the problem.