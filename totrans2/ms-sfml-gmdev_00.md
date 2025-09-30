# Preface

Designing a game from scratch can be one of the most difficult journeys to embark on. With the amount of work that goes into it, it’s would not be farfetched to compare game development to building a car. It combines many different areas of expertise that would, otherwise, not be overlapping, meaning the mastermind behind it has to, often enough, also act as the *Jack of all trades*. Not many other types of projects can claim that they found a way to combine advanced lighting calculations, accurate physics simulations, and the inner-workings of a fully fledged, stable, and self-sustaining economy-model into something cohesive. These are just some of those *trades* one has to pick up, and in the fast-growing world of gaming, new ones are constantly popping into existence. Among all of the noise, some patterns slowly begin to emerge as time passes by. With several different generations having access to gaming now, and a couple of them not knowing the world without it, certain expectations begin to form within the zeitgeist. The breath-taking inventions and technical demos of yesterday have become the common-place features of today, and the beacons of light shining onto tomorrow. Keeping up with these features and not being left behind in the dark is what makes a good game developer today, and that’s where we come in. Although it won’t teach you everything, this book will do a solid job at giving you an edge by not only expanding your repertoire of techniques and ideas, but also setting a clear goal into the future, that’s always going to keep progressing into something bigger and better.

As the first two chapters fly by, you will learn about setting up a basic, yet powerful RPG-style game, built on top of flexible architectures used in today’s games. That same game will then be given extra graphical oomph, as we cover building an efficient particle system, capable of easy expansion and many different graphical options. Subsequently, you will be brought up to speed to the practicalities and benefits of creating custom tools, such as a map editor, for modifying and managing the assets of your game. Usage of SFML’s shaders will also be touched on, right before we embark on a journey of cutting SFML out completely in [Chapter 7](ch07.html "Chapter 7.  One Step Forward, One Level Down - OpenGL Basics"), *One Step Forward, One Level Down – Integrating OpenGL*, by using raw OpenGL and rendering something on screen all by ourselves. This is followed by us exploring and implementing advanced lighting techniques, such as normal and specular maps, to really give the game scene a graphical kick with dynamic lights. Of course, there can be no light that doesn’t cast a shadow, which is why [Chapter 9](ch09.html "Chapter 9.  The Speed of Dark - Lighting and Shadows"), *The Speed of Dark – Lighting and Shadows*, covers and implements shadow-mapping in 3D, allowing us to have realistic, three-dimensional shadows. This is all topped off by making final optimizations to the game that will not only make it run as fast as possible, but also provide you with all of the tools and skills necessary to keep making improvements into the future.

While this book aims to inspire you to be the *Jack of all trades*, it will also make you a master of some by enabling your games to look and run as good as they possibly can. There is a long road ahead of us, so make sure you pack your ambition, and hopefully we shall see each other again at the finish line. Good luck!

# What this book covers

[Chapter 1](ch01.html "Chapter 1. Under the Hood - Setting up the Backend"), *Under the Hood – Setting up the Backend*, covers the usage of several underlying architectures that will power our game.

[Chapter 2](ch02.html "Chapter 2.  Its Game Time! - Designing the Project"),*It’s Game Time! – Designing the Project*, partakes in actually building and running the game project of the book, using the architectures set up in [Chapter 1](ch01.html "Chapter 1. Under the Hood - Setting up the Backend"), *Under the Hood – Setting up the Backend*.

[Chapter 3](ch03.html "Chapter 3.  Make It Rain! - Building a Particle System"), *Make It Rain! – Building a Particle System*, deals with the complexities of implementing an efficient and expansive particle system.

[Chapter 4](ch04.html "Chapter 4. Have Thy Gear Ready - Building Game Tools"), *H**ave Thy Gear Ready – Building Game Tools*, gets the ball rolling on building custom game tools by setting up their backend.

[Chapter 5](ch05.html "Chapter 5. Filling the Tool Belt - a few More Gadgets"), *Filling* *the Tool Belt – A few More Gadgets*, finishes implementing the map editor that can be used to place, edit, and otherwise manipulate map tiles, entities, and particle emitters.

[Chapter 6](ch06.html "Chapter 6. Adding Some Finishing Touches - Using Shaders")*, **Adding Some Finishing Touches – Using Shaders*, explains and uses the newly re-architected renderer that allows for easy use of shaders, by implementing a day/night cycle in our game.

[Chapter 7](ch07.html "Chapter 7.  One Step Forward, One Level Down - OpenGL Basics"), *One Step Forward, One Level Down – OpenGL Basics*, descends into the depths of technicalities of using raw OpenGL, guiding us through rendering basic shapes, texturing them, and creating the means of movement around the world.

[Chapter 8](ch08.html "Chapter 8.  Let There Be Light - An Introduction to Advanced Lighting"), *Let There Be Light – An Introduction to Advanced Lighting*, introduces and applies the concepts of lighting up our game world in three-dimensions, using normal maps to add the illusion of extra details, and adding specular highlights to create shining surfaces.

[Chapter 9](ch09.html "Chapter 9.  The Speed of Dark - Lighting and Shadows"), *The Speed of Dark – Lighting and Shadows*, expands on the lighting engine by implementing dynamic, three-dimensional, point-light shadows being cast in all directions at once.

[Chapter 10](ch10.html "Chapter 10. A Chapter You Shouldnt Skip - Final Optimizations"), *A Chapter You Shouldn’t Skip – Final Optimizations*, wraps up the book by making our game run many times faster, and providing you with the tools of taking it even further.

# What you need for this book

 First and foremost, a compiler that supports new C++ standards is required. The actual SFML library is also needed, as it powers the game we’re building. Chapters 7 and up require the newest versions of the GLEW and GLM libraries as well. Any other individual tools that may be used throughout the course of this book have been mentioned in the individual chapters they’re used in.

# Who this book is for

This book is for beginning game developers, who already have some basic knowledge of SFML, intermediate skills in modern C++, and have already built a game or two on their own, no matter how simple. Knowledge in modern OpenGL is not required, but may be a plus.

# Conventions

In this book, you will find a number of styles of text that distinguish between different kinds of information. Here are some examples of these styles, and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "We can include other contexts through the use of the include directive. "

A block of code is set as follows:

[PRE0]

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

[PRE1]

**New terms** and **important words** are shown in bold. Words that you see on the screen, in menus or dialog boxes for example, appear in the text like this: "clicking the **Next** button moves you to the next screen".

### Note

Warnings or important notes appear in a box like this.

### Tip

Tips and tricks appear like this.

# Reader feedback

Feedback from our readers is always welcome. Let us know what you think about this book-what you liked or disliked. Reader feedback is important for us as it helps us develop titles that you will really get the most out of. To send us general feedback, simply e-mail `feedback@packtpub.com`, and mention the book's title in the subject of your message. If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, see our author guide at [www.packtpub.com/authors](http://www.packtpub.com/authors).

# Customer support

Now that you are the proud owner of a Packt book, we have a number of things to help you to get the most from your purchase.

## Downloading the example code

You can download the example code files for this book from your account at [http://www.packtpub.com](http://www.packtpub.com). If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files e-mailed directly to you.

You can download the code files by following these steps:

1.  Log in or register to our website using your e-mail address and password.
2.  Hover the mouse pointer on the **SUPPORT** tab at the top.
3.  Click on **Code Downloads & Errata**.
4.  Enter the name of the book in the **Search** box.
5.  Select the book for which you're looking to download the code files.
6.  Choose from the drop-down menu where you purchased this book from.
7.  Click on **Code Download**.

Once the file is downloaded, please make sure that you unzip or extract the folder using the latest version of:

*   WinRAR / 7-Zip for Windows
*   Zipeg / iZip / UnRarX for Mac
*   7-Zip / PeaZip for Linux

The code bundle for the book is also hosted on GitHub at [https://github.com/PacktPublishing/Mastering-SFML-Game-Development](https://github.com/PacktPublishing/Mastering-SFML-Game-Development). We also have other code bundles from our rich catalog of books and videos available at [https://github.com/PacktPublishing/](https://github.com/PacktPublishing/). Check them out!

## Downloading the color images of this book

We also provide you with a PDF file that has color images of the screenshots/diagrams used in this book. The color images will help you better understand the changes in the output. You can download this file from [https://www.packtpub.com/sites/default/files/downloads/MasteringSFMLGameDevelopment_ColorImages.pdf](https://www.packtpub.com/sites/default/files/downloads/MasteringSFMLGameDevelopment_ColorImages.pdf).

## Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books-maybe a mistake in the text or the code-we would be grateful if you could report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the **Errata Submission Form** link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded to our website or added to any list of existing errata under the Errata section of that title.

To view the previously submitted errata, go to [https://www.packtpub.com/books/content/support](https://www.packtpub.com/books/content/support) and enter the name of the book in the search field. The required information will appear under the **Errata** section.

## Piracy

Piracy of copyrighted material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works in any form on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at `copyright@packtpub.com` with a link to the suspected pirated material.

We appreciate your help in protecting our authors and our ability to bring you valuable content.

## Questions

If you have a problem with any aspect of this book, you can contact us at `questions@packtpub.com`, and we will do our best to address the problem.