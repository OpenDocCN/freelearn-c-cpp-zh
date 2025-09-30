# Preface

This book is an introduction to the Godot game engine and its new version, 3.0\. Godot 3.0 has a large number of new features and capabilities that make it a strong alternative to more expensive commercial game engines. For beginners, it offers a friendly way to learn game development techniques. For more experienced developers, Godot is a powerful, customizable tool for bringing visions to life.

This book will have a project-based approach. It consists of five projects that will help developers achieve a sound understanding of how to use the Godot engine to build games.

# Who this book is for

This book is for anyone who wants to learn how to make games using a modern game engine. New users and experienced developers alike will find it a helpful resource. Some programming experience is recommended.

# What this book covers

This book is a project-based introduction to using the Godot game engine. Each of the five game projects builds on the concepts learned in the previous project.

[Chapter 1](part0022.html#KVCC0-5809b3bef8d2453086d97dfad17b2ee2), *Introduction*, introduces the concept of game engines in general and Godot specifically, including how to download Godot and install it on your computer.

[Chapter 2](part0036.html#12AK80-5809b3bef8d2453086d97dfad17b2ee2), *Coin Dash*, deals with a small game that demonstrates how to create scenes and work with Godot's node architecture.

[Chapter 3](part0079.html#2BASE0-5809b3bef8d2453086d97dfad17b2ee2), *Escape the Maze*, entails a project based on a top-down maze game that will show how to use Godot's powerful inheritance features and nodes for tile maps and sprite animation.

[Chapter 4](part0109.html#37UDA0-5809b3bef8d2453086d97dfad17b2ee2), *Space Rocks*, demonstrates working with physics bodies to create an *Asteroids*-style space game.

[Chapter 5](part0148.html#4D4J80-5809b3bef8d2453086d97dfad17b2ee2), *Jungle Jump*, involves a side-scrolling platform game in the spirit of *Super Mario Brothers. *You'll learn about kinematic bodies, animation states, and parallax backgrounds.

[Chapter 6](part0199.html#5TOVU0-5809b3bef8d2453086d97dfad17b2ee2), *3D Minigolf*, extends the previous concepts into three dimensions. You'll work with meshes, lighting, and camera control.

[Chapter 7](part0243.html#77NOM0-5809b3bef8d2453086d97dfad17b2ee2), *Additional Topics*, covers even more topics to explore once you've mastered the material in the previous chapters.

# To get the most out of this book

To best understand the example code in this book, you should have a general knowledge of programming, preferably with a modern, dynamically-typed language, such as Python or JavaScript. If you're new to programming entirely, you may wish to review a beginner Python tutorial before diving into the game projects in this book.

Godot will run on any relatively modern PC running Windows, macOS, or Linux operating systems. Your video card must support OpenGL ES 3.0.

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

The code bundle for the book is also hosted on GitHub at **[https://github.com/PacktPublishing/Godot-Game-Engine-Projects/issues](https://github.com/PacktPublishing/Godot-Game-Engine-Projects)**. In case there's an update to the code, it will be updated on the existing GitHub repository.

We also have other code bundles from our rich catalog of books and videos available at **[https://github.com/PacktPublishing/](https://github.com/PacktPublishing/)**. Check them out!

# Download the color images

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. You can download it here: [https://www.packtpub.com/sites/default/files/downloads/GodotEngineGameDevelopmentProjects_ColorImages.pdf](https://www.packtpub.com/sites/default/files/downloads/GodotEngineGameDevelopmentProjects_ColorImages.pdf).

# Conventions used

There are a number of text conventions used throughout this book.

`CodeInText`: Indicates code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles. Here is an example: "Mount the downloaded `WebStorm-10*.dmg` disk image file as another disk in your system."

A block of code is set as follows:

```cpp
extends Area2D

export (int) var speed
var velocity = Vector2()
var screensize = Vector2(480, 720)
```

Any command-line input or output is written as follows:

```cpp
adb install dodge.apk
```

**Bold**: Indicates a new term, an important word, or words that you see onscreen. For example, words in menus or dialog boxes appear in the text like this. Here is an example: "The main portion of the editor window is the Viewport."

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