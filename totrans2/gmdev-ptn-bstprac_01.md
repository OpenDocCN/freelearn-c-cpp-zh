# Introduction to Design Patterns

You've learned how to program, and you've probably created some simple games at this point, but now you want to start building something larger. Perhaps you have tried building an interesting project but you felt like the code was hacked together. Maybe you worked with a team of programmers and you couldn't see eye-to-eye on how to solve problems. Maybe your code didn't integrate well, or features were constantly being added that didn't fit with your original design. Maybe there wasn't a design to begin with. When building larger game projects, it's important that you break apart your problems, focus on writing quality code, and spend your time solving problems unique to your game, as opposed to common programming problems that already have a solution. The old advice *don't reinvent the wheel* applies to programming as well. One could say that instead of just being someone that writes code, you now need to think like a game developer or software engineer.

Knowing how to program is very similar to knowing a language. It's one thing to use a language to make conversation, but it's quite different if you're trying to create a novel or write poetry. In much the same way as when programmers are writing code in their game projects, you'll need to pick the right parts of the language to use at the best time. To organize your code well, as well as to solve problems that arise time and time again, you'll need to have certain tools. These tools, design patterns, are exactly what this book is about.

# Chapter overview

Over the course of this chapter, we will be discussing the idea of design patterns as well as the thought processes to be going through when deciding to use them. We will also be setting up our project and installing everything necessary to work with the Mach5 engine, which was written by one of the authors.

# Your objective

This chapter will be split into a number of topics. It will contain a simple step-by-step process from beginning to end. Here is the outline of our tasks:

*   What are design patterns?
*   Why you should plan for change
*   Separating the what and how
*   An introduction to interfaces
*   The advantages of compartmentalizing code
*   The problems with using design patterns in games
*   Project setup

# What are design patterns

Famously documented in the book *Design Patterns: Elements of Reusable Object-Oriented Software* by *Erich Gamma*, *John Vlissides*, *Ralph Johnson*, and *Richard Helm*, also known as the **Gang of Four** (**GoF** for short), design patterns are solutions for common programming problems. More than that, they are solutions that were designed and redesigned as developers tried to get more flexibility and reuse from their code. You don't need to have read the Gang of Four's book in order to understand this book, but after finishing you may wish to read or reread that book to gain additional insights.

As the Gang of Four title suggests, design patterns are reusable, meaning the implemented solution can be reused in the same project, or a completely new one. As programmers, we want to be as efficient as possible. We don't want to spend time writing the same code over and over, and we shouldn't want to spend time solving a problem that already has an answer. An important programming principle to follow is the **DRY** principle, **Don't Repeat Yourself**. By using and reusing design patterns, we can prevent issues or silly mistakes that would cause problems down the road. In addition, design patterns can improve the readability of your code not only by breaking apart sections that you would have put together, but also by using solutions that other developers are (hopefully) familiar with.

When you understand and use design patterns, you can shorten the length of a discussion with another developer. It is much easier to tell another programmer that they should implement a factory than to go into a lengthy discussion involving diagrams and a whiteboard. In the best-case scenario, you both know about design patters well enough that there doesn't need to be a discussion, because the solution would be obvious.

Although design patterns are important, they aren't just a library that we can just plug into our game. Rather, they are a level above libraries. They are methods for solving common problems, but the details of implementing them is always going to be unique to your project. However, once you have a good working knowledge of patterns, implementing them is easy and will feel natural. You can apply them when first designing your project, using them like a blueprint or starting point. You can also use them to rework old code if you notice that it's becoming jumbled (something we refer to as spaghetti code). Either way, it is worth studying patterns so your code quality will improve and your programming *toolbox* will grow larger.

With this *toolbox*, the number of ways to solve a problem is limited only by your imagination. It can sometimes be difficult to think of the *best* solution right off the bat. It can be difficult to know the *best* place or *best* pattern to use in a given situation. Unfortunately, when implemented in the wrong place, design patterns can create many problems, such as needlessly increasing the complexity of your project with little gain. As I mentioned before, software design is similar to writing poetry in that they are both an art. There will be advantages and disadvantages to the choices you make.

That means in order to use patterns effectively, you first need to know what problem you are trying to solve in your project. Then you must know the design patterns well enough to understand which one will help you. Finally, you'll need to know the specific pattern you are using well enough so you can adapt it to your project and your situation. The goal of this book is to provide you with this in-depth knowledge so you can always use the correct tool for the job.

There are many design patterns out there, including the foundational patterns from the Gang of Four book, architectural patterns, and many more. We will only be touching on the ones that we feel are best used for game development. We feel it is better to supply you with deep knowledge of a select few patterns than to give you a primer on every possible pattern out there. If you're interested in learning more about all of the ones out there, feel free to visit [https://en.wikipedia.org/wiki/Software_design_pattern](https://en.wikipedia.org/wiki/Software_design_pattern).

# Why you should plan for change

In my many years doing game development, one thing that has always been constant is that a project never ends up 100% the same as it was imagined in the pre-production phase. Features are added and removed at a moment's notice and things that you think are pivotal to the game experience will get replaced with something completely different. Many people can be involved in game development, such as producers, game directors, designers, quality assurance, or even marketing, so we can never tell who, where, or when those changes will be made to the project.

Since we never can tell what will be changed, it's a good practice to always write your code so it can be easily modified. This will involve planning ahead much more than you might be used to, and typically involves either a drawn flowchart, some form of pseudocode, or possibly both. However, this planning will get you much further much faster than jumping straight to coding.

Sometimes you may be starting a project from scratch; other times you may be joining a game team and using an existing framework. Either way, it is important to start coding with a plan. Writing code is called software engineering for a reason. The structure of code is often likened to building or architecting. However, let's think in smaller terms for now. Let's say you want to build some furniture from IKEA.

When you buy your furniture, you receive it unassembled with an instruction manual. If you were to start building it without following the instructions, it is possible you wouldn't finish it at all. Even if you did eventually finish, you may have assembled things out of order, causing much more work. It's much better to have a blueprint that shows you every step along the way.

Unfortunately, building a game is not exactly like following an instruction manual for furniture. In the case of games and software of any kind, the requirements from the client might constantly change. Our *client* might be the producer that has an updated timeline for us to follow. It might be our designer that just thought of a new feature we *must* have. It might even be our play testers. If they don't think the game is fun, we shouldn't just keep moving along making a bad game. We need to stop, think about our design, and try something new.

Having a plan and knowing a project will change seem to be in opposition to each other. How can we have a plan if we don't know what our end product will be like? The answer is to plan for that change. That means writing code in such a way that making changes to the design is fast and easy. We want to write code so that changing the starting location on the second level doesn't force us to edit code and rebuild all the configurations of the project. Instead, it should be as simple as changing a text file, or better yet, letting a designer control everything from a tool.

Writing code like this takes work and planning. We need to think about the design of the code and make sure it can handle change. Oftentimes this planning will involve other programmers. If you are working on a team, it helps if everyone can understand the goal of each class and how it connects with every other class. It is important to have some standards in place so others can start on or continue with the project without you there.

# Understanding UML class diagrams

Software developers have their own form of blueprints as well, but they look different from what you may be used to. In order to create them, developers use a format called **Unified Markup Language**, or **UML** for short. This simple diagramming style was primarily created by Jim Rumbaugh, Grady Booch, and Ivar Jacobson and has become a standard in software development due to the fact that it works with any programming language. We will be using them when we need to display details or concepts to you via diagrams.

Design patterns are usually best explained through the use of class diagrams, as you're able to give a demonstration of the idea while remaining abstracted. Let's consider the following class:

[PRE0]

Converted to UML, it would look something like this:

![](img/00005.jpeg)

Basic UML diagrams consist of three boxes that represent classes and the data that they contain. The top box is the name of the class. Going down, you'll see the properties or variables the class will have (also referred to as the data members) and then in the bottom box you'll see the functions that it will have. A plus symbol (**+**) to the left of the property means that it is going to be public, while a minus symbol (**-**) means it'll be private. For functions, you'll see that whatever is to the right of the colon symbol (**:**) is the return type of the function. It can also include parentheses, which will show the input parameters for the functions. Some functions don't need them, so we don't need to place them. Also, note in this case I did add `void` as the return type for both functions, but that is optional.

# Relationships between classes

Of course, that class was fairly simple. In most programs, we also have multiple classes and they can relate to each other in different ways. Here's a more in-depth example, showing the relationships between classes.

# Inheritance

First of all, we have inheritance, which shows the IS-A relationship between classes.

[PRE1]

When an object inherits from another object, it has all of the methods and fields that are contained in the parent class, while also adding their own content and features. In this instance, we have a special `FlyingEnemy`, which has the ability to fly in addition to all of the functionality of the `Enemy` class.

In UML, this is normally shown by a solid line with a hollow arrow and looks like the following:

![](img/00006.jpeg)

# Aggregation

The next idea is aggregation, which is designated by the HAS-A relationship. This is when a single class contains a collection of instances of other classes that are obtained from somewhere else in your program. These are considered to have a weak HAS-A relationship as they can exist outside of the confines of the class.

In this case, I created a new class called `CombatEncounter` which can have an unlimited number of enemies that can be added to it. However when using aggregation, those enemies will exist before the `CombatEncounter` starts; and when it finishes, they will also still exist. Through code it would look something like this:

[PRE2]

Inside of UML, it would look like this:

![](img/00007.jpeg)

# Composition

When using composition, this is a strong HAS-A relationship, and this is when a class contains one or more instances of another class. Unlike aggregation, these instances are not created on their own but, instead, are created in the constructor of the class and then destroyed by its destructor. Put into layman's terms, they can't exist separately from the whole.

In this case, we have created some new properties for the `Enemy` class, adding in combat skills that it can use, as in the Pokémon series. In this case, for every one enemy, there are four skills that the enemy will be able to have:

[PRE3]

The line in the diagram looks similar to aggregation, aside from the fact that the diamond is filled in:

![](img/00008.jpeg)

# Implements

Finally, we have implements, which we will talk about in the *Introduction to interfaces* section.

The advantage to this form of communication is that the ideas presented will work the same way no matter what programming language you're using and without a specific implementation. That's not to say that a specific implementation isn't valuable, which is why we will also include the implementation for problems in code as well.

There's a lot more information out there about UML and there are various different formats that different people like to use. A nice guide that I found that may be of interest can be found at [https://cppcodetips.wordpress.com/2013/12/23/uml-class-diagram-explained-with-c-samples/](https://en.wikipedia.org/wiki/Software_design_pattern).

# Separating the why and the how

When creating games, we have many different systems that need to be juggled around in order to provide the entire game experience. We need to have objects that are drawn to the screen, need to have realistic physics, react when they hit each other, animate, have gameplay behavior and, on top of all that, we then need to make sure that it runs well 60 times every second.

# Understanding the separation of concerns

Each of these different aspects is a problem of its own, and trying to solve all of these issues at once would be quite a headache. One of the most important concepts to learn as a developer is the idea of compartmentalizing problems, and breaking them apart into simpler and simpler pieces until they're all manageable. In computer science, there is a design principle known as the separation of concerns which deals with this issue. In this aspect, a concern would be something that will change the code of a program. Keeping this in mind, we would separate each of these concerns into their own distinct sections, with as little overlap in functionality as possible. Alternatively, we can make it so that each section solves a separate concern.

Now when we mention concerns, they are a distinct feature or a distinct section. Keeping that in mind, it can either be something as high level as an entire class or as low level as a function. By breaking apart these concerns into self-contained pieces that can work entirely on their own, we gain some distinct advantages. By separating each system and making it so they do not depend on each other, we can alter or extend any part of our project with minimal hassle. This concept creates the basis for almost every single design pattern that we'll be discussing.

By using this separation effectively, we can create code that is flexible, modular, and easy to understand. It'll also allow us to build the project in a much more iterative way because each class and function has its own clearly defined purpose. We won't have to worry nearly as much about adding new features that would break previously written code because the dependencies are on the existing functional classes, and never the other way around. This means we can to easily expand the game with things like **Downloadable Content** (**DLC**). This might include new game types, additional players, or new enemies with their own unique artificial intelligence. Finally, we can take things we've already written and decouple them from the engine so we can use them for future projects, saving time and development costs.

# An Introduction to interfaces

One of the main features of using design patterns is the idea of always programming to an interface and not to an implementation. In other words, the top of any class hierarchy should have an abstract class or an interface.

# Polymorphism refresher

In Hollywood, lots of actors and actresses take on many different roles when filming movies. They can be the hero of a story, a villain, or anything else as they inhabit a role. No matter what role they've gotten, when they are being filmed they are acting even if what they do specifically can be quite different. This kind of behavior acts similarly to the idea of polymorphism.

Polymorphism is one of the three pillars of an object-oriented language (along with encapsulation and inheritance). It comes from the words *poly* meaning many and *morph* meaning change.

Polymorphism is a way to call different specific class functions in an inheritance hierarchy, even though our code only uses a single type. That single type, the base class reference, will be changed many ways depending on the derived type. Continuing with the Hollywood example, we can tell an actor to act out a role and, based on what they've been cast in, they will do something different.

By using the `virtual` keyword on a base class function and overriding that function in a derived class, we can gain the ability to call that derived class function from a base class reference. While it may seem a bit complex at first, this will seem clearer with an example. For instance, if we have the following class:

[PRE4]

I could create a derived class with its own method, without modifying the base class in any way. In addition, we have the ability to replace or override a method within a derived class without affecting the base class. Let's say I wanted to change this function:

[PRE5]

Since a derived class can be used anywhere a base class is needed, we can refer to derived classes using a base class pointer or an array of pointers and call the correct function at runtime. Let's have a look at the following code:

[PRE6]

The following is the output of the preceding code:

[PRE7]

As you can see, even though we have an array of base class pointers, the correct derived class function is called. If the functions weren't marked as virtual, or if the derived classes didn't override the correct functions, polymorphism wouldn't work.

# Understanding interfaces

An interface implements no functions, but simply declares the methods that the class will support. Then, all of the derived classes will do the implementation. In this way, the developer will have more freedom to implement the functions to fit each instance, while having things work correctly due to the nature of using an object-oriented language.

Interfaces may contain only static final variables, and they may contain only abstract methods, which means that they cannot be implemented within the class. However, we can have interfaces that inherit from other interfaces. When creating theses classes, we can implement whatever number of interfaces we want to. This allows us to make classes become even more polymorphic but, by doing so, we are agreeing that we will implement each of the functions defined in the interface. Because a class that implements an interface extends from that base class, we would say that it has an IS-A relationship with that type.

Now, interfaces have one disadvantage, and that's the fact that they tend to require a lot of coding to implement each of the different versions as needed, but we will talk about ways to adjust and/or fix this issue over the course of this book.

In C++, there isn't an official concept of interfaces, but you can simulate the behavior of interfaces by creating an abstract class.

Here's a simple example of an interface, and an implementation of it:

[PRE8]

And here's how it looks in UML:

![](img/00009.jpeg)

# The advantages of compartmentalizing code

One important difference between procedural programming (think C-style) and object-oriented programming is the ability to encapsulate or compartmentalize code. Oftentimes we think of this as just data hiding: making variables private. In a C-style program, the functions and data are separate, but it is hard to reuse any one function because it might depend on other functions or other pieces of data in the program. In object-oriented programming, we are allowed to group the data and function together into reusable pieces. That means we can (hopefully) take a class or module and place it in a new project. This also means that since the data is private, a variable can be easily changed as long as the interface or public methods don't change. These concepts of encapsulation are important, but they aren't showing us all of the power that this provides us.

The goal of writing object-oriented code is to create objects that are responsible for themselves. Using a lot of if/else or switch statements within your code can be a symptom of bad design. For example, if I have three classes that need to read data from a text file, I have the choice of using a switch statement to read the data differently for each class type, or passing the text file to a class method and letting the class read the data itself. This is even more powerful when combined with the power of inheritance and polymorphism.

By making the classes responsible for themselves, the classes can change without breaking other code, and the other code can change without breaking the classes. We can all imagine how fragile the code would be if a game was written entirely in the main function. Anything that is added or removed is likely to break other code. Anytime a new member joined the team, they would need to understand absolutely every line and every variable in the game before they could be trusted to write anything.

By separating code into functions or classes, we are making the code easier to read, test, debug, and maintain. Anyone joining the team would of course need to understand some pieces of the code, but it might not be necessary to understand all of graphics if they are working on game logic or file loading.

Design patterns are solutions to common programming problems flexible enough to handle change. They do this by compartmentalizing sections of code. This isn't by accident. For the purposes of this book, the definition of good design is encapsulated, flexible, reusable code. So it should come as no surprise that these solutions are organized into classes or groups of classes that encapsulate the changing sections of your code.

# The structure of the Mach5 engine

Throughout this book, we will be using design patterns to solve common game programming problems. The best way to do this is by example, and so we will be examining how these problems arise and implementing the solutions using the Mach5 engine, a 2D game engine designed in C++ by *Matt Casanova*. By looking at the entire source code for a game, we will be able to see how many of the patterns work together to create powerful and easy-to-use systems.

However, before we can dive into the patterns, we should spend a little time explaining the structure of the engine. You don't need to understand every line of source code, but it is important to understand some of the core engine components and how they are used. This way we can better understand the problems we will be facing and how the solution fits together.

![](img/00010.jpeg)

While looking at the diagram, it may seem a little confusing at first, so let's examine each piece of the engine separately.

# Mach5 core engines and systems

The meaning of engine is getting a little blurred these days. Often when people talk of engines they think of entire game creation tools such as Unreal or Unity. While these are engines, the term didn't always require a tool. Game engines such as Id Software's Quake Engine or Valve Corporation's Source engine existed independently of tools, although the latter did have tools including the Hammer Editor for creating levels.

The term engine is also used to refer to components within the larger code base. This includes things like a rendering engine, audio engine, or physics engine. Even these can be created completely separate from a larger code base. Orge 3D is an open source 3D graphics engine, while the Havok Physics engine is proprietary software created by the Havok company and used in many games.

So, when we talk about the engines or systems of the Mach5 engine, we are simply referring to groups of related code for performing a specific task.

# The app

The `M5App` or application layer is a class responsible for interfacing with the operating system. Since we are trying to write clean, reusable code, it is important that we don't mix our game code with any operating system function calls. If we did this, our game would be difficult to port to another system. The `M5App` class is created in WinMain and responsible for creating and destroying every other system. Anytime our game needs to interact with the OS, including changing resolution, switching to full screen, or getting input from a device, we will use the `M5App` class. In our case, the operating system that we will be using will be Windows.

# The StageManager

The `M5StageManager` class is responsible for controlling the logic of each stage. We consider things such as the main menu, credits screen, options menu, loading screen, and playable levels to be stages. They contain behaviors that control the flow of the game. Examples of stage behavior include reading game object data from files, spawning units after specific time intervals, or switching between menus and levels.

`StageManager` is certainly not a standardized name. In other engines, this section of code may be called the game logic engine; however, most of our game logic will be separated into components so this name doesn't fit. No matter what it is called, this class will control which objects need to be created for the current stage, as well as when to switch to the next stage or quit the game altogether.

Even though this uses the name *manager* instead of *engine*, it serves as one of the core systems of the game. This class controls the main game loop and manages the collection of user stages. In order to make a game, users must derive at least one class from the base `M5Stage` class and overload the virtual functions to implement their game logic.

# The ObjectManager

The `M5ObjectManager` is responsible for creating, destroying, updating, and searching for game objects. A game object is anything visible or invisible in the game. This could include the player, bullets, enemies, and triggers--the invisible regions in a game that cause events when collided with. The derived `M5Stage` classes will use the `M5ObjectManager` to create the appropriate objects for the stage. They can also search for specific game objects to update game logic. For example, a stage may search for a player object. If one doesn't exist, the manager will switch to the game over stage.

As seen in the previous diagram, our game will use components. This means the `M5ObjectManager` will be responsible for creating those as well.

# The graphics engine

This book isn't about creating a graphics engine but we do need one to draw to the screen. Similar to how the `M5App` class encapsulates important OS function calls, our `M5Gfx` class encapsulates our graphics API. We want to make sure there is a clear separation between any API calls and our game logic. This is important so we can port our game to another system. For example, we may want to develop our game for PC, XBox One, and PlayStation 4\. This will mean supporting multiple graphics APIs since a single API isn't available for all platforms. If our game logic contains API code, then those files will need to be modified for every platform.

We won't be going deep into the details of how to implement a full graphics engine, but we give an overview of how graphics works. Think of this as a primer to the world of graphics engines.

This class allows us manipulate and draw textures, as well as control the game camera and find the visible extents of the world. `M5Gfx` also manages two arrays of graphics components, one for world space and one for screen space. The most common use of the screen space components is for creating **User Interface** (**UI**) elements such as buttons.

# Tools and utilities

Besides the core engines and systems for a game, every engine should provide some basics tools and support code. The Mach5 engine includes a few categories for tools:

*   **Debug Tools**: This includes debug asserts, message windows, and creating a debug console
*   **Random**: Helper functions to create random `int` or `float` from min/max values
*   **Math**: This includes 2D vectors and 4 x 4 matrices, as well some more general math helper functions
*   **FileIO**: Support for reading and writing `.ini` files

# The problems with using design patterns in games

Unfortunately, there are also some issues that may come into play from using design patterns exactly as described. It's often said that the fastest executing code is the code that is never called, and using design patterns will typically require you to add more code to your project than what you would have done otherwise. This will have a performance cost as well, as there will likely need to be more calculations done whenever you're using a part of your engine.

For instance, using some principles will cause some classes that you write to become extremely bloated with extra code. Design patterns are another form of complexity to add to your project. If the problem itself is simple, it can be a much better idea to focus on the simpler solutions before going straight into implementing a design pattern just because you have heard of it.

Sometimes it's better to follow the simple rule of **K.I.S.S**. and remember that it is the knowledge of the pattern that holds the most important value, not using the pattern itself.

# Setting up the project

Now that we've gotten a good understanding of why we would want to use design patterns, let's get set up the game engine that we will be using over the course of the book: the Mach5 engine. Now in order to get started, we will need to download the engine as well as the software needed to run the project. Perform the following steps:

1.  Open up your web browser of choice and visit the following website: [https://beta.visualstudio.com/downloads/](https://beta.visualstudio.com/downloads/). Once there, move to the Visual Studio Community version on the left and then click on the Free download option, as shown in the following screenshot:

![](img/00011.jpeg)

2.  If you get a window asking what to do with the file, go ahead and open it or save and then open it by clicking on the Run button:

![](img/00012.jpeg)

3.  From there, wait until the installer pops up, then select Custom, and then click on Next to start downloading the program:

![](img/00013.jpeg)

4.  Now once you get to the Features section, uncheck whatever is selected and then open up the Programming Languages tab and check Visual C++. You may go ahead and remove the other options, as we will not be using them. Then go ahead and click on the Next button, then Install, and allow it to make changes to your computer:

![](img/00014.jpeg)

You may need to wait a while at this point, so go ahead and get yourself a coffee, and once it's finished you'll need to restart your computer. After that, go ahead and continue with the project.

5.  Once you've finished installing, you next need to actually install the engine itself. With that in mind, go over to [https://github.com/mattCasanova/Mach5](https://beta.visualstudio.com/downloads/) and from there, click on the Clone or download section and then click on Download ZIP:

![](img/00015.jpeg)

6.  Once you're finished with the download, go ahead and unzip the file to a folder of your choice; then open up the `Mach5-master\EngineTest` folder, double-click on the `EngineTest.sln` file, and start up Visual Studio.
7.  You may get a login screen asking you to log in; go ahead and sign up or press the Not now, maybe later option on the bottom of the screen. You can then pick a color theme; then click Start Visual Studio.
8.  Upon starting, you may get a security warning asking if you'd still like to open this project. This is displayed from any Visual Studio solution that wasn't made on your machine, so it wants to make sure that you know where it came from, but in this case the project is perfectly safe. Go ahead and uncheck the Ask me for every project in this solution option and then select OK:

![](img/00016.gif)

9.  Once it's finished loading, you should finally see the Visual Studio interface, which should look like this:

![](img/00017.jpeg)

Visual Studio is a very powerful tool and, for developers, it can be quite useful to learn all of the functionality that it has. We'll be discussing the features as we use them, but this book shouldn't be considered the end-all book on Visual Studio.

If you're interested in learning more about the Visual Studio interface, check out: [https://msdn.microsoft.com/en-us/library/jj620919.aspxa](https://msdn.microsoft.com/en-us/library/jj620919.aspxa).

10.  The engine is built to work on 32-bit processors, so change the x64 dropdown to x86 and then click the Play button or press *F5*. It will then ask if you wish to rebuild the project. Go ahead and say Yes. If all goes well, you should eventually see a debug window and a gameplay window as well. After a few seconds, it should transition to a simple default project:

![](img/00018.jpeg)

You can play around by using the *W*, *A*, and *D* keys to move the character around and the *Spacebar* to shoot bullets at enemies. Once you're finished playing around, go ahead and hit the *Esc* key to go to the menu, and then click on the Quit button to leave the project and go back to the editor!

# Summary

And there we have it! In this first chapter, you've learned some fundamentals about design patterns and also got the Mach5 engine running on your computer.

Specifically, we learned that design patterns are solutions for common programming problems. There are a lot of reasons why we should use them, but in order to use patterns effectively, you first need to know what problem you are trying to solve and which ones can help you in that instance, which is what this book intends to teach you.

We learned how game development is always changing and how important it is to have a plan, as well as an architecture that can support those changes. With that in mind, we learned about various aspects of coding that will be used in the creation of our architecture.

We dived into learning about the separation of concerns principle and how important it is for us to separate the what and how; making it so they do not depend on each other allows us to alter or extend any part of our project with minimal hassle. Afterwards, we explored what interfaces were and how they are useful in giving us a foundation we can build on. Later, we dived into the Mach5 engine, saw an example of how compartmentalized code worked, and the advantages of it. We also saw how using design patterns in games can be a great thing, as well as the problems that they have.

Finally, we downloaded the Mach5 engine ourselves and made sure that it worked correctly. Moving on, in the next chapter, we will tackle our first design pattern, the Singleton, and see how it can be useful to us!