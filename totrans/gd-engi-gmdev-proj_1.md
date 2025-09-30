# Introduction

Whether it's your desired career or a recreational hobby, game development is a fun and rewarding endeavor. There never been a better time to get started in game development. Modern programming languages and tools have made it easier than ever to build high- quality games and distribute them to the world. If you're reading this book, then you've set your feet on the path to making the game of your dreams.

This book is an introduction to the Godot game engine and its new 3.0 version, which was released in early 2018\. Godot 3.0 has a large number of new features and capabilities that make it a strong alternative to expensive commercial game engines. For beginners, it offers a friendly way to learn fundamental game development techniques. For more experienced developers, Godot is a powerful, customizable, and *open* tool for bringing your visions to life.

This book takes a project-based approach that will introduce you to the fundamentals of the engine. It consists of five games that are designed to help you achieve a sound understanding of game development concepts and how they're applied in Godot. Along the way, you will learn how Godot works and absorb important techniques that you can apply to your projects.

# General advice

This section contains some general advice to readers, based on the author's experience as a teacher and lecturer. Keep these tips in mind as you work through the book, especially if you're very new to programming.

Try to follow the projects in the book in order. Later chapters may build on topics that were introduced in earlier chapters, where they are explained in more detail. When you encounter something that you don't remember, go back and review that topic in the earlier chapter. No one is timing you, and there's no prize for finishing the book quickly.

There is a lot of material to absorb here. Don't feel discouraged if you don't get it at first. The goal is not to become an expert in game development overnight—that's just not possible. Repetition is the key to learning complex topics; the more you work with Godot's features, the more familiar and *easy* they will start to seem. Try looking back at [Chapter 2](f24a8958-bb32-413a-97ae-12c9e7001c2c.xhtml), *Coin Dash*, when you finish [Chapter 7](3d5cd7c5-b53a-4731-88f9-fab128f609a1.xhtml), *Additional Topics*. You'll be surprised at how much more you'll understand compared to the first time you read it.

If you're using the PDF version of this book, resist the temptation to copy and paste the code. Typing the code yourself will engage more of your brain. It's similar to how taking notes during a lecture helps you learn better than just listening, even if you never read the notes. If you're a slow typist, it will also help you work on your typing speed. In a nutshell: you're a programmer, so get used to typing code!

One of the biggest mistakes that new game developers make is taking on a bigger project than they can handle. It is very important to keep the scope of your project as small as possible when starting out. You will be much more successful (and learn more) if you finish two or three small games than if you have a large, incomplete project that has grown beyond your ability to manage.

You'll notice that the five games in this book follow this strategy very strictly. They are all small in scope, both for practical reasons—to fit reasonably into book-sized lessons—but also to remain focused on teaching you the basics. As you build them, you will likely find yourself thinking of additional features and gameplay elements right away. *What if the spaceship had upgrades?* *What if the character could do wall jumps?*

Ideas are great, but if you haven't finished the basic project yet, write them down and save them for later. Don't let yourself be sidetracked by one *cool idea* after another. Developers call this *feature creep*, and it's a trap that has led to many an unfinished game. Don't fall victim to it.

Finally, don't forget to take a break now and again. You shouldn't try and power through the whole book in just a few sittings. After each new concept, and especially after each chapter, give yourself time to absorb the new information before you dive into the next one. You'll find that you not only retain more information, but you'll probably enjoy the process more.

# What is a game engine?

Game development is complex and involves a wide variety of knowledge and skills. In order to build a modern game, you need a great deal of underlying technology before you can make the actual game itself. Imagine that you had to build your own computer and write your own operating system before you could even start programming. Game development would be a lot like that if you truly had to start from scratch and build *everything* you needed.

In addition, there are a number of common needs that every game has. For example, no matter what the game is, it's going to need to draw things on the screen. If the code to do that has already been written, it makes more sense to reuse it than to create it all over again for every game. That's where game frameworks and engines come in.

A **game framework** is a set of libraries with helper code that assists in building the foundational parts of a game. It doesn't necessarily provide all the pieces, and you may still have to write a great deal of code to tie everything together. Because of this, building a game with a game framework can take more time than one built with a full game engine.

A **game engine** is a collection of tools and technologies designed to ease the process of game-making by removing the need to *reinvent the wheel* for each new game project. It provides a framework of commonly needed functionality that often needs a significant investment in time to develop.

Here is a list of some of the main features a game engine will provide:

*   **Rendering (2D and 3D)**: Rendering is the process of displaying your game on the player's screen. A good rendering pipeline must take into account modern GPU support, high-resolution displays, and effects like lighting, perspective, and viewports, while maintaining a very high frame rate.
*   **Physics**: While a very common requirement, building a robust and accurate physics engine is a monumental task. Most games require some sort of collision detection and response system, and many need physics simulation, but few developers want to take on the task of writing one, especially if they have ever tried to do so.
*   **Platform support**: In today's market, most developers want to be able to release their games on multiple platforms, such as consoles, mobile, PC, and/or the web. A game engine provides a unified exporting process to publish on multiple platforms without needing to rewrite game code or support multiple versions.
*   **Common development environment**: By using the same unified interface to make multiple games, you don't have to re learn a new workflow every time you start a new project.

In addition, there will be tools to assist with features such as networking, easing the process of working with images and sound, animations, debugging, level creation, and many more. Often, game engines will include the ability to import content from other tools such as those used to create animations or 3D models.

Using a game engine allows the developer to focus on building their game, rather than creating all of the underlying framework needed to make it work. For small or independent developers, this can mean the difference between releasing a game after one year of development instead of three, or even never at all.

There are dozens of popular game engines on the market today, such as Unity, Unreal Engine, and GameMaker Studio, just to name a few. An important fact to be aware of is that the majority of popular game engines are commercial products. They may or may not require any financial investment to get started, but they will require some kind of licensing and/or royalty payments if your game makes money. Whatever engine you choose, you need to carefully read the user agreement and make sure you understand what you are and are not allowed to with the engine, and what hidden costs, if any, you may be responsible for.

On the other hand, there are some engines which are non-commercial and *open source*, such as the Godot game engine, which is what this book is all about.

# What is Godot?

Godot is a fully featured modern game engine, providing all of the features described in the previous section and more. It is also completely free and open source, released under the very permissive MIT license. This means there are no fees, no hidden costs, and no royalties to pay on your game's revenue. Everything you make with Godot 100% belongs to you, which is not the case with many commercial game engines that require an ongoing contractual relationship. For many developers, this is very appealing.

If you're not familiar with the concept of open source, community-driven development, this may seem strange to you. However, much like the Linux kernel, Firefox browser, and many other very well-known pieces of software, Godot is not developed by a company as a commercial product. Instead, a dedicated community of passionate developers donate their time and expertise to building the engine, testing and fixing bugs, producing documentation, and more.

As a game developer, the benefits of using Godot are vast. Because it is unencumbered by commercial licensing, you have complete control over exactly how and where your game is distributed. Many commercial game engines restrict the types of projects you can make, or require a much more expensive license to build games in certain categories, such as gambling.

Godot's open source nature also means there is a level of transparency that doesn't exist with commercial game engines. For example, if you find that a particular engine feature doesn't quite meet your needs, you are free to modify the engine itself and add the new features you need, no permission required. This can also be very helpful when debugging a large project, because you have full access to the engine's internal workings.

It also means that you can directly contribute to Godot's future. See [Chapter 7](3d5cd7c5-b53a-4731-88f9-fab128f609a1.xhtml), *Additional Topics*, for more information about how you can get involved with Godot development.

# Downloading Godot

You can download the latest version of Godot by visiting [https://godotengine.org/](https://godotengine.org/) and clicking Download. This book is written for version 3.0\. If the version you download has another number at the end (like 3.0.3), that's fine—this just means that it includes updates to version 3.0 that fix bugs or other issues.

A version 3.1 release is currently in development and may have been released by the time you read this book. This version may or may not include changes that are incompatible with the code in this book. Check the GitHub repository for this book for information and errata: [https://github.com/PacktPublishing/Godot-Game-Engine-Projects](https://github.com/PacktPublishing/Godot-Game-Engine-Projects)

On the download page, there are a few options that bear explaining. First, 32-bit versus 64-bit: this option depends on your operating system and your computer's processor. If you're not sure, you should choose the 64-bit version. You will also see a *Mono Version*. This is a version specially built to be used with the C# programming language. Don't download this one unless you plan to use C# with Godot. At the time of writing, C# support is still experimental, and is not recommended for beginners.

Double-click on the file you downloaded to unzip it, and you'll have the Godot application. Optionally, you can drag it to your `Programs` or `Applications` folder, if you have one. Double-click the application to launch it and you'll see Godot's Project Manager window.

# Alternate installation methods

There are a few other ways to get Godot on your computer besides downloading it from the Godot website. Note that there is no difference in functionality when installed this way. The following are merely alternatives for downloading the application:

*   **Steam**: If you have an account on Steam, you can install Godot via the Steam desktop application. Search for Godot in the Steam store and follow the instructions to install it. You can launch Godot from the Steam application and it will even track your *playtime*.

*   **Package Managers**: If you're using one of the following operating system package managers, you can install Godot via its normal install process. See the documentation for your package manager for details. Godot is available in these package managers:
*   Homebrew (macOS)
*   Scoop (Windows)
*   Snap (Linux)

# Overview of the Godot UI

Like most game engines, Godot has a unified development environment. This means that you use the same interface to work on all of the aspects of your game—code, visuals, audio, and so on. This section is an introduction to the interface and its parts. Take note of the terminology used here; it will be used throughout this book when referring to actions you'll take in the editor window.

# Project Manager

The Project Manager is the first window you'll see when you open Godot:

![](img/45f9d5c9-0319-410f-a5e9-139a53c59215.png)

In this window, you can see a list of your existing Godot projects. You can choose an existing project and click Run to play the game or click Edit to work on it in the Godot Editor (refer to the following screenshot). You can also create a new project by clicking New Project:

![](img/57d0c336-86a3-4684-9f60-48c37b6f571f.png)

Here, you can give the project a name and create a folder to store it in. Always try to choose a name that describes the project. Also keep in mind that different operating systems handle capitalization and spaces in filenames differently. It's a good idea to stick to lowercase and use underscores, `_`, instead of spaces for maximum compatibility.

Note the warning message—in Godot, each project is stored as a separate folder on the computer. All the files that the project uses are in this folder. Nothing outside of this project folder will be accessible in the game, so you need to put any images, sounds, models, or other data into the project folder. This makes it convenient to share Godot projects; you only need to zip the project folder and you can be confident that another Godot user will be able to open it and not be missing any necessary data.

# Choosing filenames

When you're naming your new project, there are a few simple rules you should try and follow that may save you some trouble in the future. Give your project a name that describes what it is—*Wizard Battle Arena* is a much better project name than *Game #2*. In the future, you'll never be able to remember which game #2 was, so be as descriptive as possible.

You should also think about how you name your project folder and the files in it. Some operating systems are *case-sensitive* and distinguish between `My_Game` and `my_game`, while others do not. This can lead to problems if you move your project from one computer to another. For this reason, many programmers develop a standardized naming scheme for their projects, for example: *No spaces in filenames, use "_" between words*. Regardless of what naming scheme you adopt, the most important thing is to be consistent.

Once you've created the project folder, the Create & Edit button will open the new project in the Editor window.

Try it now: create a project called `test_project`.

If you're using a version of the Windows operating system, you'll also see a console window open when you run Godot. In this window, you can see warnings and errors produced by the engine and/or your project. This window doesn't appear under macOS or Linux, but you can see the console output if you launch the application from the command line using a Terminal program.

# Editor window

The following is a screenshot of the main Godot editor window. This is where you will spend most of your time when building projects in Godot. The editor interface is divided into several sections, each offering different functionality. The specific terminology for each section is described as follows:

![](img/9e898853-13db-4ced-b2b0-c13c3370671c.png)

Godot Editor Window

The main portion of the editor window is the Viewport. This is where you'll see parts of your game as you're working on them.

In the upper-left corner is the Main menus, where you can save and load files, edit project settings, and get help.

In the center at the top is a list of the Workspaces you can switch between when working on different parts of your game. You can switch between 2D and 3D mode, as well Script mode, where you can edit your game's code. The AssetLib is a place where you can download add-ons and example projects. See [Chapter 7](3d5cd7c5-b53a-4731-88f9-fab128f609a1.xhtml), *Additional Topics*, for more information on using the AssetLib. Refer to the following screenshot:

![](img/ba56f0bf-bc0b-4186-902c-adfd03d61c50.png)

The following screenshot shows the Workspaces buttons on the toolbar. The icons in the toolbar will change based on what kind of object you are editing. So will the items in the Bottom panel, which will open various smaller windows for accessing specific information such as debugging, audio settings, and more:

![](img/bcdc0fb3-a9b8-4d25-ba25-4f9ac171d7d7.png)

The buttons in the upper-right Playtest area are for launching the game and interacting with it when it's running:

![](img/ccb6753f-8b55-4419-a88b-e2b2df28535d.png)

Finally, on the left and right sides are the Docksyou can use to view and select game items and set their properties. The left-hand dock contains the FileSystem tab:

![](img/c0fdd007-c791-49f2-9ef5-84462553b1bf.png)

All of the files inside the project folder are shown here, and you can click on folders to open them and see what they contain. All resources in your project will be located relative to `res://`, which is the project's root folder. For example, a file path might look like this: `res://player/Player.tscn`.

In the right-hand dock, you can see several tabs. The Scene tab shows the current scene you are working on in the Viewport. In the Inspector tab below it, you can see and adjust the properties of any object you select. Refer to the following screenshot:

![](img/548db3d8-c0e8-4234-8879-ef634bd4e58d.png)

Selecting the Import tab and clicking on a file in the FileSystem tab lets you adjust how Godot imports resources like textures, meshes, and sounds, as shown in the following screenshot:

![](img/944b8294-9eaf-4926-8a84-081175e8cdaa.png)

As you work through the game projects in this book, you'll learn about the functionality of these items and become familiar with navigating the editor interface. However, there are a few other concepts you need to know about before getting started.

# About nodes and scenes

**Nodes** are the basic building blocks for creating games in Godot. A node is an object that can represent a variety of specialized game functions. A given type of node might display graphics, play an animation, or represent a 3D model of an object. The node also contains a collection of properties, allowing you to customize its behavior. Which nodes you add to your project depends on what functionality you need. It's a modular system designed to give you flexibility in building your game objects.

In your project, the nodes you add are organized into a *tree* structure. In a tree, nodes are added as *children* of other nodes. A particular node can have any number of children, but only one *parent* node. When a group of nodes are collected into a tree, it is called a **scene**, and the tree is referred to as the **scene tree**:

![](img/3b876fac-ec8b-4b81-b33b-bec54b1319b7.png)

Scenes in Godot are typically used to create and organize the various game objects in your project. You might have a player scene that contains all the nodes and scripts that make the player's character work. Then, you might create another scene that defines the game's map: the obstacles and items that the player must navigate through. You can then combine these various scenes into the final game using *instancing*, which you'll learn about later.

While nodes come with a variety of properties and functions, any node's behavior and capabilities can also be extended by attaching a *script* to the node. This allows you to write code that makes the node do more than it can in its default state. For example, you can add a Sprite node to your scene to display an image, but if you want that image to move or disappear when clicked, you'll need to add a script to create that behavior.

# Scripting in Godot

At the time of writing, Godot provides three official languages for scripting nodes: GDScript, VisualScript, and C#. GDScript is the dedicated built-in language, providing the tightest integration with the engine, and is the most straightforward to use. VisualScript is still very new and in the *testing* stage, and should be avoided until you have a good understanding of Godot's workings. For most projects, C# is best reserved for those portions of the game where there is a specific performance need; most Godot projects will not need this level of additional performance. For those that do, Godot gives the flexibility to use a combination of GDScript and C# where you need them.

In addition to the three supported scripting languages, Godot itself is written in C++ and you can get even more performance and control by extending the engine's functionality directly. See [Chapter 7](3d5cd7c5-b53a-4731-88f9-fab128f609a1.xhtml), *Additional Topics*, for information on using other languages and extending the engine.

All of the games in this book use GDScript. For the majority of projects, GDScript is the best choice of language. It is very tightly integrated with Godot's **Application Programming Interface** (**API**), and is designed for rapid development.

# About GDScript

GDScript's syntax is very closely modeled on the Python language. If you are familiar with Python already, you will find GDScript very familiar. If you are comfortable with another dynamic language, such as JavaScript, you should find it relatively easy to learn. Python is very often recommended as a good beginner language, and GDScript shares that user-friendliness.

This book assumes you have at least *some* programming experience already. If you've never coded before, you may find it a little more difficult. Learning a game engine is a large task on its own; learning to code at the same time means you've taken on a major challenge. If you find yourself struggling with the code in this book, you may find that working through an introductory Python lesson will help you grasp the basics.

Like Python, GDScript is a *dynamically typed* language, meaning you do not need to declare a variable's type when creating it, and it uses *whitespace* (indentation) to denote code blocks. Overall, the result of using GDScript for your game's logic is that you write less code, which means faster development and fewer mistakes to fix.

To give you an idea of what GDScript looks like, here is a small script that causes a sprite to move from left to right across the screen at a given speed:

```cpp
extends Sprite

var speed = 200

func _ready():
    position = Vector2(100, 100)

func _process(delta):
    position.x += speed * delta
    if position.x > 500:
        position.x = 0
```

Don't worry if this doesn't make sense to you yet. In the following chapters, you'll be writing lots of code, which will be accompanied by explanations of how it all works.

# Summary

In this chapter, you were introduced to the concept of a game engine in general and to Godot in particular. Most importantly, you downloaded Godot and launched it!

You learned some important vocabulary that will be used throughout this book when referring to various parts of the Godot editor window. You also learned about the concepts of nodes and scenes, which are the fundamental building blocks of Godot. 

You also received some advice on how to approach the projects in this book and game development in general. If you ever find yourself getting frustrated as you are working through this book, go back and reread the *General advice* section. There's a lot to learn, and it's okay if it doesn't all make sense the first time. You'll make five different games over the course of this book, and each one will help you understand things a little bit more.

You're ready to move on to [Chapter 2](a56e3c2d-5d7f-41d6-98c4-c1d95e17fc31.xhtml), *Coin Dash*, where you'll start building your first game in Godot.