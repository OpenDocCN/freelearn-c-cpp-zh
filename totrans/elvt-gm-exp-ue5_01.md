# 1

# Introduction to Unreal Engine

Welcome to *Game Development Projects with Unreal Engine Second Edition*. If this is the first time you’re using **Unreal Engine 5** (**UE5**), this book will support you in getting started with one of the most in-demand game engines on the market. You will discover how to build up your game development skills and how to express yourself by creating video games. If you’ve already tried using UE5, this book will help you develop your knowledge and skills further so that you can build games more easily and effectively.

A game engine is a software application that allows you to produce video games from the ground up. Their feature sets vary significantly but usually allow you to import multimedia files, such as 3D models, images, audio, and video, and manipulate those files through the use of programming, where you can use programming languages such as C++, Python, and Lua, among others.

UE5 uses two main programming languages, C++ and Blueprint, with the latter being a visual scripting language that allows you to do most of what C++ also allows. Although we will be teaching a bit of Blueprint in this book, we will mostly focus on C++, and hence expect you to have a basic understanding of the language, including topics such as **variables**, **functions**, **classes**, **inheritance**, and **polymorphism**. We will remind you about these topics throughout this book where appropriate.

Examples of popular video games made with Unreal Engine 4, the previous Unreal Engine version that UE5 is heavily based on, include *Fortnite*, *Final Fantasy VII Remake*, *Borderlands 3*, *Star Wars: Jedi Fallen Order*, *Gears 5*, and *Sea of Thieves*, among many others. All of these have a very high level of visual fidelity, are well-known, and have (or had) millions of players.

The following link specifies some of the great games that have been made with Unreal Engine 5: [https://youtu.be/kT4iWCxu5hA](https://youtu.be/kT4iWCxu5hA). This showcase will show you the variety of games that UE5 allows you to make, both in terms of visuals and gameplay style.

If you’d like to make games such as the ones shown in the video one day or contribute to them in any way, then you’ve taken your first step in that direction.

This chapter will be an introduction to the Unreal Engine editor. You will learn about the editor’s interface; how to add, remove, and manipulate objects in a level; how to use Unreal Engine’s Blueprint visual scripting language; and how to use materials in combination with meshes.

By the end of this chapter, you will be able to navigate the Unreal Engine editor, create Actors, manipulate them inside the level, and create materials. Let’s start this chapter by learning how to create a new UE5 project in this first exercise.

Note

Before you continue this chapter, make sure you have installed all the necessary software mentioned in the *Preface*.

# Technical requirements

The code files for this chapter can be found here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition)

# Exercise 1.01 – creating an Unreal Engine 5 project

In this first exercise, we will learn how to create a new UE5 project. UE5 has predefined project templates that allow you to implement a basic setup for your project. We’ll be using the **Third Person** template project in this exercise.

Follow these steps to complete this exercise:

1.  After installing Unreal Engine version 5.0, launch the editor by clicking the **Launch** button next to the version icon.
2.  After you’ve done that, you’ll be greeted with the engine’s **Projects** window, which will show you the existing projects that you can open and work on. It will also give you the option to create a new project. Because we have no projects yet, the **Recent Projects** section will be empty. To create a new project, choose a **Project Category** option, which in our case will be **Games**. Then, click **Next**.
3.  After that, you’ll see the **Project Templates** window. This window will show all the available project templates in Unreal Engine. When creating a new project, instead of having that project start empty, you have the option to add some assets and code out of the box, which you can then modify to your liking. There are several project templates available for different types of games, but we’ll want to go with the **Third Person** project template in this case.
4.  Select that template and click the **Next** button, which should take you to the **Project Settings** window.

In this window, you can choose a few options related to your project:

*   **Blueprint or C++**: Here, you can choose whether you want to be able to add C++ classes. The default option is **Blueprint**, but in our case, we’ll want to select the **C++** option.
*   **Quality**: Here, you can choose whether you want your project to have high-quality graphics or high performance. Set this option to **Maximum Quality**.
*   **Raytracing**: Here, you can choose whether you want raytracing enabled or disabled. Raytracing is a novel graphics rendering technique that allows you to render objects by simulating the path of light (using light rays) over a digital environment. Although this technique is rather costly in terms of performance, it also provides much more realistic graphics, especially when it comes to lighting. Set it to **disabled**.
*   **Target Platforms**: Here, you can choose the main platforms you’ll want this project to run on. Set this option to **Desktop/Console**.
*   **Starter Content**: Here, you can choose whether you want this project to come with an additional set of basic assets. Set this option to **With Starter Content**.
*   **Location and Name**: At the bottom of the window, you’ll be able to choose the location where your project will be stored on your computer and its name.

1.  Once you’ve made sure that all the options have been set to their intended values, click the **Create Project** button. This will cause your project to be created according to the parameters you set. It may take a few minutes for it to be ready. With that, you have created your first UE5 project!

Now, let’s learn about some of the basics of UE5.

# Getting to know Unreal Engine

In this section, you will be introduced to the Unreal Engine editor, which is a fundamental topic for becoming familiar with UE5.

Once your project has been generated, you should see the Unreal Engine editor open automatically. This screen is likely the one that you will see the most when working with Unreal Engine, so you must get accustomed to it.

Let’s break down what we can see in the editor window:

![Figure 1.1 – The Unreal Engine editor divided into its main windows ](img/Figure_1.01_B18531.jpg)

Figure 1.1 – The Unreal Engine editor divided into its main windows

Let’s look at these windows in more detail:

1.  **Viewport**: At the very center of the screen, you can see the **Viewport** window. This will show you the content of the current level and will allow you to navigate through your level as well as add, move, remove, and edit objects inside it. It also contains several different parameters regarding visual filters, object filters (which objects you can see), and the lighting on your level.
2.  **Outliner**: At the top-right corner of the screen, you’ll see the **Outliner** window. This allows you to quickly list and manipulate the objects that are on your level. The **Viewport** and **Outliner** windows work hand in hand in allowing you to manage your level, where the former will show you what it looks like and the latter will help you manage and organize it. The **Outliner** window allows you to organize the objects in your level in directories by showing you the *objects* in your level.
3.  **Details**: At the far right of the screen, below **Outliner**, you’ll be able to see the **Details** panel, which allows you to edit the properties of an object that you have selected in your level. Since no objects have been selected in the preceding screenshot, it is empty. However, if you select any object in your level by *left-clicking* on it, its properties should appear in this window, as shown in the following screenshot:

![Figure 1.2 – The Details tab ](img/Figure_1.02_B18531.jpg)

Figure 1.2 – The Details tab

1.  **Toolbar**: At the top of the screen, you’ll see the **Toolbar** area, where you’ll be able to save your current level, add objects to your level, and play your level, among other things.

Note

We will only be using some of the buttons from these toolbars, namely, the **Save Current**, **Settings**, **Add**, and **Play** buttons.

1.  **Content Drawer**: One of the windows that you’ll be using very frequently is the **Content Drawer** window. This window lets you quickly access the **Context Browser** window. You can also open it by using *Ctrl* + *Space*. The **Content Browser** window will let you browse and manipulate all the files and assets located inside your project’s folder. As mentioned at the start of this chapter, Unreal Engine allows you to import several types of multimedia files, and **Content Browser** is the window that will allow you to browse and edit them in their respective sub-editors. Whenever you create an Unreal Engine project, it will always generate a **Content** folder. This folder will be the **root directory** of the **Content Browser** window, meaning you can only browse files inside that folder. You can see the directory you’re currently browsing inside the **Content Browser** window by looking at the top of it, which, in our case, is **Content** | **ThirdPersonCPP**.

![Figure 1.3 – The Content Browser window shown in Unreal Editor interface](img/Figure_1.03_B18531.jpg)

Figure 1.3 – The Content Browser window shown in Unreal Editor interface

If you click the icon to the left of the **Filters** button, at the very left of the **Content Browser** window, you will be able to see the directory hierarchy of the **Content** folder. This directory view allows you to select, expand, and collapse individual directories in the **Content** folder of your project:

![Figure 1.4 – The Content Browser window’s directory view ](img/Figure_1.04_B18531.jpg)

Figure 1.4 – The Content Browser window’s directory view

Note

The terms **Content Drawer** and **Content Browser** are interchangeable.

Now that we have learned about the main windows of the Unreal Engine editor, let’s look at how to manage those windows (hiding and showing their tabs).

# Exploring editor windows

As we’ve seen, the Unreal Engine editor is comprised of many windows, all of which are resizable, movable, and have a corresponding tab on top of them. You can *click and hold* a window’s tab and drag it to move it somewhere else. You can hide tab labels by *right-clicking* them and selecting the **Hide Tabs** option:

![Figure 1.5 – How to hide a tab](img/Figure_1.05_B18531.jpg)

Figure 1.5 – How to hide a tab

If the tab labels have been hidden, you can get them to reappear by clicking the *blue triangle* in the top-left corner of that window, as shown in the following screenshot:

![Figure 1.6 – The blue triangle that allows you to show a window’s tab](img/Figure_1.06_B18531.jpg)

Figure 1.6 – The blue triangle that allows you to show a window’s tab

You can also dock the windows to the sidebar to hide them while also having them easily available:

![Figure 1.7 – Docking a window to the sidebar ](img/Figure_1.07_B18531.jpg)

Figure 1.7 – Docking a window to the sidebar

After that, to show or hide them, you simply have to click them:

![Figure 1.8 – Showing a window docked to the sidebar ](img/Figure_1.08_B18531.jpg)

Figure 1.8 – Showing a window docked to the sidebar

When it comes to the windows that are docked to the lower bar, such as the **Content Drawer** window, you can undock them from the lower bar into the editor by clicking **Dock in Layout** in the top-right corner:

![Figure 1.9 – Undocking a window from the lower bar in the editor ](img/Figure_1.09_B18531.jpg)

Figure 1.9 – Undocking a window from the lower bar in the editor

Keep in mind that you can browse and open all the windows available in the editor, including the ones that were just mentioned, by clicking the **Window** button in the top-left corner of the editor.

Another very important thing you should know is how to play your level from inside the editor (also known as **PIE**). At the right edge of the **Toolbar** window, you’ll see the green **Play** button. If you click it, you’ll start playing the currently open level inside the editor:

![Figure 1.10 – The green play button, alongside other game playback buttons ](img/Figure_1.10_B18531.jpg)

Figure 1.10 – The green play button, alongside other game playback buttons

Once you hit **Play**, you’ll be able to control the player character in the level by using the *W*, *A*, *S*, and *D* keys to move the player character, the *Spacebar* to jump, and moving your mouse to rotate the camera:

![Figure 1.11 – The level being played inside the editor](img/Figure_1.11_B18531.jpg)

Figure 1.11 – The level being played inside the editor

Then, you can press the *Shift* + *Esc* keys to stop playing the level.

Now that we’ve gotten accustomed to some of the editor’s windows, let’s take a deeper look at the **Viewport** window’s navigation.

# Viewport navigation

In the previous section, we mentioned that the **Viewport** window allows you to visualize your level, as well as manipulate the objects inside it. Because this is a very important window for you to use and has a lot of functionality, we’re going to learn more about it in this section.

Before we start learning about the **Viewport** window, let’s quickly get to know **levels**. In UE5, levels represent a **collection of objects**, as well as their locations and properties. The **Viewport** window will always show you the contents of the currently selected level, which in this case was already made and generated alongside the **Third Person** template project. In this level, you can see four wall objects, one ground object, a set of stairs, and some other elevated objects, as well as the player character, which is represented by the UE5 mannequin. You can create multiple levels and switch between them by opening them via the **Content Browser** window.

To manipulate and navigate the currently selected level, you must use the **Viewport** window. If you press and hold the *left mouse button* inside the window, you’ll be able to rotate the camera horizontally by moving the mouse *left* and *right*, and move the camera forward and backward by moving the mouse *forward* and *backward*. You can achieve similar results by holding the *right mouse button*, except the camera will rotate vertically when you move the mouse *forward* and *backward*, which allows you to rotate the camera both horizontally and vertically.

Additionally, you can move around the level by clicking and holding the **Viewport** window with the *right mouse button* (the *left mouse button* works too, but using it for movement is not as useful due to there not being as much freedom when rotating the camera) and using the *W* and *S* keys to move forward and backward, the *A* and *D* keys to move sideways, and the *E* and *Q* keys to move up and down.

If you look at the top-right corner of the **Viewport** window, you will see a small camera icon with a number next to it, which will allow you to change the speed at which the camera moves in the **Viewport** window.

Another thing you can do in the **Viewport** window is change its visualization settings. You can change the type of visualization in the **Viewport** window by clicking the button that currently says **Lit**, which will show you all the options available for different lighting and other types of visualization filters.

If you click on the **Perspective** button, you’ll have the option to switch between seeing your level from a perspective view, as well as from an orthographic view, the latter of which may help you build your levels faster.

Now that we’ve learned how to navigate the viewport, let’s learn how to manipulate objects, also known as Actors, in your level.

# Manipulating Actors

In Unreal Engine, all the objects that can be placed in a level are referred to as Actors. In a movie, an actor would be a human playing a character, but in UE5, every single object you see in your level, including walls, floors, weapons, and characters, is an Actor.

Every Actor must have what’s called a **Transform** property, which is a collection of three things:

*   `Vector` property signifying the position of that Actor in the level in the *X*, *Y,* and *Z*-axis. A vector is simply a tuple with three floating-point numbers – one for the location of the point on each axis.
*   `Rotator` property signifying the rotation of that Actor along the *X*, *Y,* and *Z*-axis. A rotator is also a tuple with three floating-point numbers – one for the angle of rotation on each axis.
*   `Vector` property signifying the scale (that is, the size) of that Actor in the level in the *X*, *Y,* and *Z*-axis. This is also a collection of three floating-point numbers – one for the scale value on each axis.

Actors can be moved, rotated, and scaled in a level, which will modify their **Transform** property accordingly. To do this, select any object in your level by *left-clicking* on it. You should see the **Move** tool appear:

![Figure 1.12 – The Move tool, which allows you to move an Actor in the level  ](img/Figure_1.12_B18531.jpg)

Figure 1.12 – The Move tool, which allows you to move an Actor in the level

The **Move** tool is a three-axis gizmo that allows you to move an object in any of the axes simultaneously. The red arrow of the **Move** tool (pointing to the left in the preceding screenshot) represents the *X*-axis, the green arrow (pointing to the right in the preceding screenshot) represents the *Y*-axis, and the blue arrow (pointing up in the preceding screenshot) represents the *Z*-axis. If you *click and hold* any of these arrows and then drag them around the level, you will move your Actor along that axis in the level. If you click the handles that connect two arrows, you will move the Actor along both those axes simultaneously, and if you click the white sphere at the intersection of all the arrows, you will move the Actor freely along all three axes:

![Figure 1.13 – An Actor being moved on the Z-axis using the Move tool ](img/Figure_1.13_B18531.jpg)

Figure 1.13 – An Actor being moved on the Z-axis using the Move tool

The **Move** tool allows you to move an Actor around the level, but if you want to rotate or scale an Actor, you’ll need to use the **Rotate** and **Scale** tools, respectively. You can switch between the **Move**, **Rotate**, and **Scale** tools by pressing the *W*, *E*, and *R* keys, respectively. Press *E* to switch to the **Rotate** tool:

![Figure 1.14 – The Rotate tool, which allows you to rotate an Actor ](img/Figure_1.14_B18531.jpg)

Figure 1.14 – The Rotate tool, which allows you to rotate an Actor

The **Rotate** tool, as expected, allows you to rotate an Actor in your level. You can *click and hold* any of the arcs to rotate the Actor around its associated axis. The red arc (top left in the preceding screenshot) will rotate the Actor around the *X*-axis, the green arc (top right in the preceding screenshot) will rotate the Actor around the *Y*-axis, and the blue arc (lower center in the preceding screenshot) will rotate the Actor around the *Z*-axis:

![Figure 1.15 – A cube before and after being rotated 30 degrees around the Y-axis  ](img/Figure_1.15_B18531.jpg)

Figure 1.15 – A cube before and after being rotated 30 degrees around the Y-axis

Keep in mind that an object’s rotation around the *X*-axis is usually designated as **Roll**, its rotation around the *Y*-axis is usually designated as **Pitch**, and its rotation around the *Z*-axis is usually designated as **Yaw**.

Lastly, we have the **Scale** tool. Press *R* to switch to it:

![Figure 1.16 – The Scale tool ](img/Figure_1.16_B18531.jpg)

Figure 1.16 – The Scale tool

The **Scale** tool allows you to increase and decrease the scale (size) of an Actor in the *X*, *Y*, and *Z* axes, where the red handle (left in the preceding screenshot) will scale the Actor on the *X*-axis, the green handle (right in the preceding screenshot) will scale the Actor on the *Y*-axis, and the blue handle (top in the preceding screenshot) will scale the Actor on the *Z*-axis:

![Figure 1.17 – A Cube Actor before and after being scaled on all three axes  ](img/Figure_1.17_B18531.jpg)

Figure 1.17 – A Cube Actor before and after being scaled on all three axes

You can also toggle between the **Move**, **Rotate**, and **Scale** tools by clicking the following icons at the top of the **Viewport** window:

![Figure 1.18 – The Move, Rotate, and Scale tool icons ](img/Figure_1.18_B18531.jpg)

Figure 1.18 – The Move, Rotate, and Scale tool icons

Additionally, you can change the increments with which you move, rotate, and scale your objects through the grid snapping options to the right of the **Move**, **Rotate**, and **Scale** tool icons. By clicking the buttons highlighted in blue, you’ll be able to disable snapping altogether, and by pressing the buttons showing the current snapping increments, you’ll be able to change those increments:

![Figure 1.19 – The grid-snapping icons for moving, rotating, and scaling ](img/Figure_1.19_B18531.jpg)

Figure 1.19 – The grid-snapping icons for moving, rotating, and scaling

Now that you know how to manipulate Actors already present in your level, let’s learn how to add and remove Actors to and from our level.

## Exercise 1.02 – adding and removing Actors

In this exercise, we will be adding and removing Actors from our level.

When it comes to adding Actors to your level, there are two main ways in which you can do so: by dragging assets from the **Content Browser** window or by dragging the default assets from the **Modes** window’s **Place Mode**.

Follow these steps to complete this exercise:

1.  Go to the `ThirdPersonCharacter` Actor. If you drag that asset to your level using the *left mouse button*, you will be able to add an instance of that Actor to it. It will be placed wherever you let go of the *left mouse button*:

![Figure 1.20 – Dragging an instance of the ThirdPersonCharacter Actor to our level ](img/Figure_1.20_B18531.jpg)

Figure 1.20 – Dragging an instance of the ThirdPersonCharacter Actor to our level

1.  Similarly, drag an Actor to your level by using the **Add** button in the **Toolbar** window (the cube with the green *+*):

![Figure 1.21 – Dragging a Cylinder Actor to our level ](img/Figure_1.21_B18531.jpg)

Figure 1.21 – Dragging a Cylinder Actor to our level

1.  To delete an Actor, simply select the Actor and press the *Delete* key. You can also *right-click* on an Actor to look at the many other options available to you regarding that Actor.

Note

Although we won’t be covering this topic in this book, one of the ways developers can populate their levels with simple boxes and geometry, for prototyping purposes, is BSP Brushes. These can be quickly molded into the desired shape as you build your levels. To find out more about BSP Brushes, go to [https://docs.unrealengine.com/en-US/Engine/Actors/Brushes](https://docs.unrealengine.com/en-US/Engine/Actors/Brushes).

And with this, we have concluded this exercise and learned how to add and remove Actors to and from our level.

Now that we know how to navigate the **Viewport** window, let’s learn about Blueprint Actors.

# Understanding Blueprint Actors

In UE5, the word Blueprint can be used to refer to two different things: UE5’s visual scripting language or a specific type of asset, also referred to as a Blueprint class or Blueprint asset.

As we’ve mentioned previously, an Actor is an object that can be placed in a level. This object can either be an instance of a C++ class or an instance of a Blueprint class, both of which must inherit from the Actor class (either directly or indirectly). So, what is the difference between a C++ class and a Blueprint class, you may ask? There are a few:

*   If you add programming logic to your C++ class, you’ll have access to more advanced engine functionality than you would if you were to create a Blueprint class.
*   In a Blueprint class, you can easily view and edit visual components of that class, such as a 3D mesh or a Trigger Box Collision, as well as modify properties defined in the C++ class that are exposed to the editor, which makes managing those properties much easier.
*   In a Blueprint class, you can easily reference other assets in your project, whereas in C++, you can also do so but less simply and less flexibly.
*   Programming logic that runs on Blueprint visual scripting is slower in terms of performance than that of a C++ class.
*   It’s simple to have more than one person work on a C++ class simultaneously without conflicts in a source version platform, whereas with a Blueprint class, which is interpreted as a binary file instead of a text file, conflicts will occur in your source version platform if two different people edit the same Blueprint class.

Note

If you don’t know what a source version platform is, this is how several developers can work on the same project and have it updated with the work done by other developers. In these platforms, different people can usually edit the same file simultaneously, so long as they edit different parts of that file, and still receive updates that other programmers made without them affecting your work on that same file. One of the most popular source version platforms is GitHub.

Keep in mind that Blueprint classes can inherit either from a C++ class or from another Blueprint class.

Lastly, before we create our first Blueprint class, another important thing you should know is that you can write programming logic in a C++ class and then create a Blueprint class that inherits from that class, but can also access its properties and methods if you specify that in the C++ class. You can have a Blueprint class edit properties defined in the C++ class, as well as call and override functions using the Blueprint scripting language. We will be doing some of these things in this book.

Now that you know a bit more about Blueprint classes, let’s create our own.

## Exercise 1.03 – creating Blueprint Actors

In this short exercise, we will learn how to create a new Blueprint Actor.

Follow these steps to complete this exercise:

1.  Go to the **ThirdPersonCPP** | **Blueprints** directory inside the **Content Browser** window and *right-click* inside it. The following window should pop up:

![Figure 1.22 – The options window inside the Content Browser window ](img/Figure_1.22_B18531.jpg)

Figure 1.22 – The options window inside the Content Browser window

This options menu contains the types of assets that you can create in UE5 (Blueprints are simply a type of asset, along with other types of assets, such as **Level**, **Material**, and **Sound**).

1.  Click on the **Blueprint Class** icon to create a new Blueprint class. When you do, you will be given the option to choose the C++ or Blueprint class that you want to inherit from:

![Figure 1.23 – The Pick Parent Class window that pops up when you create a new Blueprint class ](img/Figure_1.23_B18531.jpg)

Figure 1.23 – The Pick Parent Class window that pops up when you create a new Blueprint class

1.  Select the first class from this window – that is, the `Actor` class. After this, the text of the new Blueprint class will be automatically selected so that you can easily name it what you want. Name this Blueprint class `TestActor` and press the *Enter* key to accept this name.

After following these steps, you will have created your Blueprint class and completed this exercise. Once you’ve created this asset, double-click on it with the *left mouse button* to open the Blueprint editor. We will learn more about this in the next section.

# Exploring the Blueprint editor

The Blueprint editor is a sub-editor within the Unreal Engine editor specifically for Blueprint classes. Here, you can edit the properties and logic for your Blueprint classes, or those of their parent classes, as well as their visual appearance.

When you open an Actor Blueprint class, you should see the Blueprint editor. This window will allow you to edit your Blueprint classes in UE5\. Let’s learn about the windows that you’re currently seeing:

![Figure 1.24 – The Blueprint editor window is broken down into five parts ](img/Figure_1.24_B18531.jpg)

Figure 1.24 – The Blueprint editor window is broken down into five parts

Let’s look at these windows in more detail:

1.  **Viewport**: Front and center in the editor, you have the **Viewport** window. This window, similar to the **Level Viewport** window that we already learned about, will allow you to visualize your Actor and edit its components. Every Actor can have several Actor Components, some of which have a visual representation, such as Mesh Components and Collision Components. We’ll talk about Actor Components in more depth later in this book.

Technically, this center window contains three tabs, only one of which is the **Viewport** window, but we’ll be talking about the other important tab, **Event Graph**, after we tackle this editor’s interface. The third tab is the **Construction Script** window, which will not be covered in this book.

1.  **Components**: At the top left of the editor, you have the **Components** window. As mentioned previously, Actors can have several Actor Components, and this window is the one that will allow you to add and remove those Actor Components in your Blueprint class, as well as access the Actor Components defined in the C++ classes it inherits from.
2.  **My Blueprint**: At the bottom left of the editor, you have the **My Blueprint** window. This will allow you to browse, add, and remove variables and functions defined in both this Blueprint class and the C++ class it inherits from. Keep in mind that Blueprints have a special kind of function, called an **event**, which is used to represent an event that happened in the game. You should see three of them in this window: **BeginPlay**, **ActorBeginOverlap**, and **Tick**. We’ll be talking about these shortly.
3.  **Details**: At the right of the editor, you have the **Details** window. Similar to the editor’s **Details** window, this window will show you the properties of the currently selected Actor Component, function, variable, event, or any other individual element of this Blueprint class. If you currently have no elements selected, this window will be empty.
4.  **Toolbar**: At the top center of the editor, you have the **Toolbar** window. This window will allow you to compile the code you wrote in this Blueprint class, save it, locate it in the **Content Browser** window, and access this class’s settings, among other things.

You can see the parent class of a Blueprint class by looking at the top-right corner of the Blueprint editor. If you click the name of the parent class, you’ll be taken to either the corresponding Blueprint class, through the Unreal Engine editor, or the C++ class, through Visual Studio.

Additionally, you can change a Blueprint class’s parent class by clicking on the **File** tab at the top left of the Blueprint editor and selecting the **Reparent Blueprint** option, which will allow you to specify the new parent class of this Blueprint class.

Now that we’ve learned about the basics of the Blueprint editor, let’s look at its **Event Graph**.

# Exploring the Event Graph window

The **Event Graph** window is where you’ll be writing all of your Blueprint visual scripting code, creating your variables and functions, and accessing other variables and functions declared in this class’s parent class.

If you select the **Event Graph** tab, which you should be able to see to the right of the **Viewport** tab, you will be shown the **Event Graph** window instead of the **Viewport** window. On clicking the **Event Graph** tab, you will see the following window:

![Figure 1.25 – The Event Graph window, showing three disabled events ](img/Figure_1.25_B18531.jpg)

Figure 1.25 – The Event Graph window, showing three disabled events

You can navigate the **Event Graph** window by holding the *right mouse button* and dragging inside the graph, you can zoom in and out by scrolling the *mouse wheel*, and you can select nodes from the graph by either clicking the *left mouse button* or by clicking and holding to select an area of nodes.

You can also *right-click* inside the **Event Graph** window to access the Blueprint’s **Actions** menu, which allows you to access the actions you can perform in the **Event Graph** window, including getting and setting variables, calling functions or events, and many others.

The way scripting works in Blueprint is by connecting nodes using pins. There are several types of nodes, such as variables, functions, and events. You can connect these nodes through pins, of which there are two types:

*   **Execution pins**: These will dictate the order in which the nodes will be executed. If you want node 1 to be executed first and then node 2, you can link the output execution pin of node 1 to the input execution pin of node 2, as shown in the following screenshot:

![Figure 1.26 – Blueprint execution pins ](img/Figure_1.26_B18531.jpg)

Figure 1.26 – Blueprint execution pins

*   **Variable pins**: These work as parameters (also known as input pins), at the left of the node, and return values (also known as output pins), at the right of the node, representing a value of a certain type (integer, float, Boolean, and others):

![Figure 1.27 – The Get Scalar Parameter Value node ](img/Figure_1.27_B18531.jpg)

Figure 1.27 – The Get Scalar Parameter Value node

Let’s understand this better by completing an exercise.

## Exercise 1.04 – creating Blueprint variables

In this exercise, we will learn how to create Blueprint variables by creating a new variable of the **Boolean** type.

In Blueprint, variables work similarly to the ones you would use in C++. You can create them, get their value, and set them.

Follow these steps to complete this exercise:

1.  To create a new Blueprint variable, head to the **My Blueprint** window and click the **+** button in the **Variables** category:

![Figure 1.28 – The + button in the Variables category ](img/Figure_1.28_B18531.jpg)

Figure 1.28 – The + button in the Variables category

1.  After that, you’ll automatically be allowed to name your new variable. Name this new variable `MyVar`:

![Figure 1.29 – Naming the new variable MyVar ](img/Figure_1.29_B18531.jpg)

Figure 1.29 – Naming the new variable MyVar

1.  Compile your Blueprint by clicking the **Compile** button on the left-hand side of the **Toolbar** window:

![Figure 1.30 – The Compile button ](img/Figure_1.30_B18531.jpg)

Figure 1.30 – The Compile button

1.  Now, if you look at the **Details** window, you should see the following:

![Figure 1.31 – The MyVar variable settings in the Details window ](img/Figure_1.31_B18531.jpg)

Figure 1.31 – The MyVar variable settings in the Details window

1.  Here, you’ll be able to edit all the settings related to this variable, with the most important ones being **Variable Name**, **Variable Type**, and **Default Value** at the end of the settings. You can change the values of Boolean variables by clicking the gray box to their right:

![Figure 1.32 – The variable types available from the Variable Type drop-down menu](img/Figure_1.32_B18531.jpg)

Figure 1.32 – The variable types available from the Variable Type drop-down menu

1.  You can also drag a getter or setter for a variable inside the **My Blueprint** tab into the **Event Graph** window:

![Figure 1.33 – Dragging the MyVar variable into the Event Graph window ](img/Figure_1.33_B18531.jpg)

Figure 1.33 – Dragging the MyVar variable into the Event Graph window

Getters are nodes that contain the current value of a variable, while setters are nodes that allow you to change the value of a variable.

1.  To allow a variable to be editable in each of the instances of this Blueprint class, you can click the eye icon to the right of that variable inside the **My Blueprint** window:

![Figure 1.34 – Clicking the eye icon to expose a variable and allow it to be instance-editable ](img/Figure_1.34_B18531.jpg)

Figure 1.34 – Clicking the eye icon to expose a variable and allow it to be instance-editable

1.  Then, you can drag an instance of this class to your level, select that instance, and see the option to change that variable’s value in the **Details** window of the editor:

![Figure 1.35 – The exposed MyVar variable that can be edited through the Details panel of that object ](img/Figure_1.35_B18531.jpg)

Figure 1.35 – The exposed MyVar variable that can be edited through the Details panel of that object

And with that, you know how to create Blueprint variables. Now, let’s learn how to create Blueprint functions.

## Exercise 1.05 – creating Blueprint functions

In this exercise, we will create our first Blueprint function. In Blueprint, functions and events are relatively similar, with the only difference being that an event will only have an output pin, usually because it gets called from outside of the Blueprint class:

![Figure 1.36 – An event (left), a pure function call that doesn’t need execution pins (middle), and a normal function call (right) ](img/Figure_1.36_B18531.jpg)

Figure 1.36 – An event (left), a pure function call that doesn’t need execution pins (middle), and a normal function call (right)

Follow these steps to complete this exercise:

1.  Click the **+** button inside the **Functions** category of the **My Blueprint** window:

![Figure 1.37 – The + Function button being hovered over, which will create a new function ](img/Figure_1.37_B18531.jpg)

Figure 1.37 – The + Function button being hovered over, which will create a new function

1.  Name the new function `MyFunc`.

Compile your Blueprint by clicking the **Compile** button in the **Toolbar** window.

1.  Now, if you look at the **Details** window, you should see the following:

![Figure 1.38 – The Details panel for the MyFunc function ](img/Figure_1.38_B18531.jpg)

Figure 1.38 – The Details panel for the MyFunc function

Here, you can edit all the settings related to this function, with the most important ones being **Inputs** and **Outputs**. These will allow you to specify the variables that this function must receive and will return.

Lastly, you can edit what this function does by *clicking* it inside the `false` every time it is called:

![Figure 1.39 – The MyFunc function ](img/Figure_1.39_B18531.jpg)

Figure 1.39 – The MyFunc function

1.  To save the modifications we made to this Blueprint class, click the **Save** button next to the **Compile** button on the toolbar. Alternatively, you can have it so that the Blueprint automatically saves every time you compile it successfully by selecting that option.

Now, you know how to create Blueprint functions. Next, we will look at the **Multiply** Blueprint node we’ll be making use of in this chapter’s remaining exercises and activities.

# Understanding the Multiply node

Blueprints contain many more nodes that are not related to variables or functions. One such example is arithmetic nodes (that is, adding, subtracting, multiplying, and so on). If you search for `Multiply` in the Blueprint **Actions** menu, you’ll find the **Multiply** node:

![Figure 1.40 – The multiply node ](img/Figure_1.40_B18531.jpg)

Figure 1.40 – The multiply node

This node allows you to input two or more parameters, which can be of many types (for example, integer, float, vector, and so on; you can add more by clicking the **+** icon to the right of the **Add pin** text) and output the result of multiplying all of them. We will be using this node later, in this chapter’s activity.

# Exploring the BeginPlay and Tick events

Now, let’s look at two of the most important events in UE5: **BeginPlay** and **Tick**.

As mentioned previously, events will usually be called from outside the Blueprint class. In the case of the **BeginPlay** event, this event gets called either when an instance of this Blueprint class is placed in the level and the level starts being played, or when an instance of this Blueprint class is spawned dynamically while the game is being played. You can think of the **BeginPlay** event as the first event that will be called on an instance of this Blueprint, which you can use for initialization.

The other important event to know about in UE5 is the **Tick** event. As you may know, games run at a certain frame rate, with the most frequent being either 30 **frames per second** (**FPS**) or 60 FPS. This means that the game will render an updated image of the game 30 or 60 times every second. The **Tick** event will get called every time the game does this, which means that if the game is running at 30 FPS, the **Tick** event will get called 30 times every second.

Go to your Blueprint class’s **Event Graph** window and delete the three grayed-out events by selecting all of them and clicking the *Delete* key, which should cause the **Event Graph** window to become empty. After that, *right-click* inside the **Event Graph** window, type in **BeginPlay**, and select the **Event BeginPlay** node by either clicking the *Enter* key or by clicking on that option in the Blueprint **Actions** menu. This should cause that event to be added to the **Event Graph** window:

![Figure 1.41 – The BeginPlay event being added to the Event Graph window through the Blueprint Actions menu](img/Figure_1.41_B18531.jpg)

Figure 1.41 – The BeginPlay event being added to the Event Graph window through the Blueprint Actions menu

*Right-click* inside the **Event Graph** window, type **Tick**, and select the **Event Tick** node. This should cause that event to be added to the **Event Graph** window:

![Figure 1.42 – The Tick event ](img/Figure_1.42_B18531.jpg)

Figure 1.42 – The Tick event

Unlike the **BeginPlay** event, the **Tick** event will be called with a parameter, **DeltaTime**. This parameter is a float that indicates the amount of time that has passed since the last frame was rendered. If your game is running at 30 FPS, this means that the interval between each of the frames being rendered (the delta time) is going to be, on average, 1/30 seconds, which is around 0.033 seconds (33.33 milliseconds). If frame 1 is rendered and then frame 2 is rendered 0.2 seconds after that, then frame 2’s delta time will be 0.2 seconds. If frame 3 gets rendered 0.1 seconds after frame 2, frame 3’s delta time will be 0.1 seconds, and so forth.

But why is the **DeltaTime** parameter so important? Let’s take a look at the following scenario: you have a Blueprint class that increases its position on the *Z*-axis by 1 unit every time a frame is rendered using the **Tick** event. However, you are faced with a problem: there’s the possibility that players will run your game at different frame rates, such as 30 FPS and 60 FPS. The players that are running the game at 60 FPS will cause the **Tick** event to be called twice as much as the players that are running the game at 30 FPS, and the Blueprint class will end up moving twice as fast because of that. This is where the delta time comes into play: because the game that’s running at 60 FPS will have the **Tick** event called with a lower delta time value (the interval between the frames being rendered is much smaller), you can use that value to change the position on the *Z*-axis. Although the **Tick** event is being called twice as much on the game running at 60 FPS, its delta time is half the value, so it all balances out. This will cause two players playing the game with different frame rates to have the same result.

Note

In this book, the **Tick** event is used a few times for demonstration purposes. However, because of its performance hit, you should be mindful when using it. If you use the **Tick** event for something that doesn’t need to be done every single frame, there’s probably a better or more efficient way of doing it.

Note

If you want a Blueprint that is using the delta time to move, you can make it move faster or slower by multiplying the delta time by the number of units you want it to move per second (for example, if you want a Blueprint to move 3 units per second on the *Z*-axis, you can tell it to move `3 * DeltaTime` units every frame).

Now, let’s complete an exercise where we will work with Blueprint nodes and pins.

## Exercise 1.06 – offsetting the TestActor class on the Z-axis

In this exercise, you’ll be using the `BeginPlay` event to offset (move) the `TestActor` class on the *Z*-axis when the game starts being played.

Follow these steps to complete this exercise:

1.  Open the `TestActor` Blueprint class.
2.  Using the `Event BeginPlay` node to the graph, if it’s not already there.
3.  Add the `AddActorWorldOffset` function and connect the `BeginPlay` event’s output execution pin to this function’s input execution pin. This function is responsible for moving an Actor in the intended axes (*X*, *Y*, and *Z*) and it receives the following parameters:
    *   `Target`: The Actor that this function should be called on, which will be the Actor calling this function. The default behavior is to call this function on the Actor calling this function, which is exactly what we want and is shown using the `self` property.
    *   `DeltaLocation`: The amount that we want to offset this Actor by in each of the three axes: *X*, *Y*, and *Z*.
    *   We won’t be getting into the other two parameters, `Sweep` and `Teleport`, so you can leave them as-is. They are both Boolean types and should be left set to `false`:

![Figure 1.43 – The BeginPlay event calling the AddActorWorldOffset function ](img/Figure_1.43_B18531.jpg)

Figure 1.43 – The BeginPlay event calling the AddActorWorldOffset function

1.  Split the `Delta Location` input pin, which will cause this `Vector` property to be split into three float properties. You can do this to any variable type that is comprised of one or more subtypes (you wouldn’t be able to do this to the float type because it’s not comprised of any variable subtypes) by *right-clicking* on them and selecting **Split Struct Pin**:

![Figure 1.44 – The Delta Location parameter being split from a vector into three floats ](img/Figure_1.44_B18531.jpg)

Figure 1.44 – The Delta Location parameter being split from a vector into three floats

1.  Set the *Z* property of `100` units by using the *left mouse button*, typing that number, and then pressing the *Enter* key. This will cause our `100` units when the game starts.
2.  Add a cube shape to your **TestActor** using the **Components** window so that we can see our Actor. You can do this by clicking the **+ Add** button, typing **Cube**, and then selecting the first option under the **Basic Shapes** section:

![Figure 1.45 – Adding a cube shape ](img/Figure_1.45_B18531.jpg)

Figure 1.45 – Adding a cube shape

1.  Compile and save your Blueprint class by clicking the **Compile** button.
2.  Go back to the level’s **Viewport** window and place an instance of your **TestActor** Blueprint class inside the level, if you haven’t done so already:

![Figure 1.46 – Adding an instance of TestActor to the level ](img/Figure_1.46_B18531.jpg)

Figure 1.46 – Adding an instance of TestActor to the level

1.  When you play the level, you should notice that the **TestActor** class we added to the level is in a more elevated position:

![Figure 1.47 – TestActor increasing its position on the Z-axis when the game starts ](img/Figure_1.47_B18531.jpg)

Figure 1.47 – TestActor increasing its position on the Z-axis when the game starts

1.  After making these modifications, save the changes that you’ve made to our level by either pressing *Ctrl* + *S* or clicking the **Save Current** button in the **Toolbar** window.

In this exercise, you learned how to create your first Actor Blueprint class with Blueprint scripting logic.

Note

Both the `TestActor` Blueprint asset and the `Map` asset, along with the final result of this exercise, can be found here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition).

Now, let’s learn a bit more about the `ThirdPersonCharacter` Blueprint class.

# The ThirdPersonCharacter Blueprint class

Let’s take a look at the `ThirdPersonCharacter` Blueprint class, which is the Blueprint that represents the character that the player controls, and look at the Actor Components that it contains.

Go to the **ThirdPersonCPP** | **Blueprints** directory inside the **Content Browser** window and open the **ThirdPersonCharacter** asset:

![Figure 1.48 – The ThirdPersonCharacter Blueprint class ](img/Figure_1.48_B18531.jpg)

Figure 1.48 – The ThirdPersonCharacter Blueprint class

Previously, when we introduced the **Components** window inside the Blueprint editor, we mentioned **Actor Components**.

Actor Components are entities that must live inside an Actor and allow you to spread the logic of your Actor into several different Actor Components. In this Blueprint, we can see that there are four visually represented Actor Components:

*   A Skeletal Mesh Component, which shows the UE5 mannequin
*   A Camera Component, which shows where the player will be able to see the game from
*   An Arrow Component, which allows us to see where the character is facing (this is mainly used for development purposes, not while the game is being played)
*   A Capsule Component, which specifies the collision range of this character

If you look at the **Components** window, you’ll see a few more Actor Components than the ones we can see in the **Viewport** window. This is because some Actor Components don’t have a visual representation and are purely made up of C++ or Blueprint code. We’ll look at Actor Components in more depth in the next chapter and [*Chapter 7*](B18531_07.xhtml#_idTextAnchor154)*, Working with UE5 Utilities*.

If you take a look at this Blueprint class’s **Event Graph** window, you’ll see that it’s essentially empty, similar to the one we saw with our **TestActor** Blueprint class, despite it having a bit of logic associated with it. This is because that logic is defined in the C++ class, not in this Blueprint class. We’ll look at how to do this in the next chapter.

To explain this Blueprint class’s Skeletal Mesh Component, we should first talk about meshes and materials.

# Exploring the usage of meshes and materials

For a computer to visually represent a 3D object, it needs two things: a 3D mesh and a material. 3D meshes allow us to specify the shape of an object and its size, while a material allows us to specify its color and visual tones, among other things. We’ll dive deeper into both of these in the following sections and see how UE5 allows us to work with them.

## Meshes

3D meshes allow you to specify the size and shape of an object, like this mesh representing a monkey’s head:

![Figure 1.49 – A 3D mesh of a monkey’s head ](img/Figure_1.49_B18531.jpg)

Figure 1.49 – A 3D mesh of a monkey’s head

Meshes are comprised of several vertices, edges, and faces. Vertices are simply 3D coordinates with *X*, *Y,* and *Z* positions; an edge is a connection (that is, a line) between two vertices; and a face is a connection of three or more edges. The preceding screenshot shows the individual vertices, edges, and faces of the mesh, where each face is colored between white and black, depending on how much light is reflecting off the face. Nowadays, video games can render meshes with thousands of vertices in such a way that you can’t tell the individual vertices apart because there are so many of them so close together.

## Materials

Materials, on the other hand, allow you to specify how a mesh is going to be represented. They allow you to specify a mesh’s color, draw a texture on its surface, or even manipulate its vertices.

Creating meshes is something that, at the time of writing this book, is not properly supported in UE5 and should be done in another piece of software, such as Blender or Autodesk Maya, so we won’t be going into this in great detail here. We will, however, learn how to create materials for existing meshes.

In UE5, you can add meshes through Mesh Components, which inherit from the Actor Component class. There are several types of Mesh Components, but the two most important ones are Static Mesh Components, for meshes that don’t have animations (for example, cubes, static level geometry), and Skeletal Mesh Components, for meshes that have animations (for example, character meshes that play movement animations). As we saw earlier, the **ThirdPersonCharacter** Blueprint class contains a Skeletal Mesh Component because it’s used to represent a character mesh that plays movement animations. In the next chapter, we’ll learn how to import assets such as meshes into our UE5 project.

Now, let’s learn how to manipulate materials in UE5.

# Manipulating materials in UE5

In this section, we’ll learn how materials work in UE5\. As mentioned previously, materials are what specify the visual aspects of a certain object, including its color and how it reacts to light. To learn more about them, follow these steps:

1.  Go back to your **Level Viewport** window and select the **Cube** object shown in the following screenshot:

![Figure 1.50 – The Cube object, next to the text that says Third Person on the floor ](img/Figure_1.50_B18531.jpg)

Figure 1.50 – The Cube object, next to the text that says Third Person on the floor

1.  Take a look at the **Details** window, where you’ll be able to see both the mesh and material associated with this object’s **Static Mesh** Component:

![Figure 1.51 – The Static Mesh and Materials (Element 0) properties of the Cube object’s Static Mesh Component ](img/Figure_1.51_B18531.jpg)

Figure 1.51 – The Static Mesh and Materials (Element 0) properties of the Cube object’s Static Mesh Component

Note

Keep in mind that meshes can have more than one material, but must have at least one.

1.  Click the *looking glass* icon next to the **Materials** property to be taken to that material’s location in the **Content Browser** window. This icon works with any reference to any asset inside the editor, so you can do the same thing with the asset referenced as the cube object’s **Static Mesh**:

![Figure 1.52 – The looking glass icon (left), which takes you to that asset’s location in the Content Browser (right) ](img/Figure_1.52_B18531.jpg)

Figure 1.52 – The looking glass icon (left), which takes you to that asset’s location in the Content Browser (right)

1.  *Double-click* the asset with the *left mouse button* to open its properties. Because this material is a child of another material, we must select its parent material. In this material’s **Details panel**, you’ll find the **Parent** property. Click the *looking glass* icon to select it in the **Context Browser** window:

![Figure 1.53 – The Parent property ](img/Figure_1.53_B18531.jpg)

Figure 1.53 – The Parent property

1.  After selecting that asset, double-click it with the *left mouse button* to open it in the **Material** Editor. Let’s break down the windows present in the **Material** Editor:

![Figure 1.54 – The Material Editor window broken down into five parts ](img/Figure_1.54_B18531.jpg)

Figure 1.54 – The Material Editor window broken down into five parts

Let’s look at these windows in more detail:

1.  **Graph**: Front and center in the editor, you have the **Graph** window. Similar to the Blueprint editor’s **Event Graph** window, the **Material** Editor’s graph is also node-based, where you’ll also find nodes connected by pins, although here, you won’t find execution pins, only input and output pins.
2.  **Palette**: At the right edge of the screen, you’ll see the **Palette** window, where you can search for all the nodes that you can add to the **Graph** window. You can also do this the same way as in the Blueprint editor’s **Event Graph** window by *right-clicking* inside the **Graph** window and typing in the node you wish to add.
3.  **Viewport**: At the top-left corner of the screen, you’ll see the **Viewport** window. Here, you can preview the result of your material and how it will appear on some basic shapes such as spheres, cubes, and planes.
4.  **Details**: At the bottom-left corner of the screen, you’ll see the **Details** window where, similar to the Blueprint editor, you’ll see the details of either this **Material** asset or those of the currently selected node in the **Graph** window.
5.  **Toolbar**: At the top edge of the screen, you’ll see the **Toolbar** window, where you’ll be able to apply and save the changes you’ve made to your material, as well as perform several actions related to the **Graph** window.

In every single Material Editor inside UE5, you’ll find a node with the name of that **Material** asset, where you’ll be able to specify several parameters related to it by plugging that node’s pins into other nodes.

In this case, you can see that there’s a node called `0.7` being plugged into the `0.7`. You can create constant nodes of a single number, a 2 vector (for example, `(1,` `0.5)`), a 3 vector (for example, `(1,` `0.5,` `4)`), and a 4 vector (for example, `(1,0.5,` `4,` `0)`). To create these nodes, you can click the **Graph** window with the *left mouse button* while holding the *1*, *2*, *3*, or *4* number keys, respectively.

Materials have several input parameters, so let’s go through some of the most important ones:

*   `BaseColor`: This parameter is simply the color of the material. Generally, constants or texture samples are used to connect to this pin, to either have an object be a certain color or to map to a certain texture.
*   `Metallic`: This parameter will dictate how much your object will look like a metal surface. You can do this by connecting a constant single number node that ranges from 0 (not metallic) to 1 (very metallic).
*   `Specular`: This parameter will dictate how much your object will reflect light. You can do this by connecting a constant single number node that ranges from 0 (doesn’t reflect any light) to 1 (reflects all the light). If your object is already very metallic, you will see little to no difference.
*   `Roughness`: This parameter will dictate how much the light that your object reflects will be scattered (the more the light scatters, the less clear this object will reflect what’s around it). You can do this by connecting a constant single number node that ranges from 0 (the object essentially becomes a mirror) to 1 (the reflection on this object is blurry and unclear).

Note

To learn more about `material` inputs like the ones shown previously, go to [https://docs.unrealengine.com/en-US/Engine/Rendering/Materials/MaterialInputs](https://docs.unrealengine.com/en-US/Engine/Rendering/Materials/MaterialInputs).

UE5 also allows you to import images (`.jpeg`, `.png`) as `Texture` assets, which can then be referenced in a material using `Texture Sample` nodes:

![Figure 1.55 – The Texture Sample node, which allows you to specify a texture and use it or its color channels as pins ](img/Figure_1.55_B18531.jpg)

Figure 1.55 – The Texture Sample node, which allows you to specify a texture and use it or its color channels as pins

Note

We will learn how to import files into UE5 in the next chapter.

To create a new **Material** asset, *right-click* on the directory inside the **Content Browser** window where you want to create the new asset, which will allow you to choose which asset to create, and then select **Material**.

With that, you know how to create and manipulate materials in UE5.

Now, let’s jump into this chapter’s activity, which is the first activity in this book.

# Activity 1.01 – propelling TestActor on the Z-axis indefinitely

In this activity, you will use the `TestActor` to move it on the *Z*-axis indefinitely, instead of doing this only once when the game starts.

Follow these steps to complete this activity:

1.  Open the `TestActor` Blueprint class.
2.  Add the **Event Tick** node to the Blueprint’s **Event Graph** window.
3.  Add the **AddActorWorldOffset** function, split its **DeltaLocation** pin, and connect the **Tick** event’s output execution pin to this function’s input execution pin, similar to what we did in *Exercise 1.01 – creating an Unreal Engine 5 project*.
4.  Add a `Float Multiplication` node to the **Event Graph** window.
5.  Connect the **Tick** event’s **Delta Seconds** output pin to the first input pin of the **Float Multiplication** node.
6.  Create a new variable of the `float` type, call it `VerticalSpeed`, and set its default value to `25`.
7.  Add a getter to the `VerticalSpeed` variable in the `Float Multiplication` node. After that, connect the `Float Multiplication` node’s output pin to the **Delta Location Z** pin of the **AddActorWorldOffset** function.
8.  Delete the **BeginPlay** event and the **AddActorWorldOffset** function connected to it, both of which we created in *Exercise 1.01 – creating an Unreal Engine 5 project*.
9.  Delete the existing instance of our actor in the level and drag in a new one.
10.  Play the level. You will notice our `TestActor` rising from the ground and up into the air over time:

![Figure 1.56 – TestActor propelling itself vertically ](img/Figure_1.56_B18531.jpg)

Figure 1.56 – TestActor propelling itself vertically

And with those steps completed, we have concluded this activity – the first of many in this book. We’ve consolidated adding and removing nodes to and from the Blueprint editor’s **Event Graph** window, as well as using the **Tick** event and its **DeltaSeconds** property to create game logic that maintains consistency across different frame rates.

Note

The solution to this activity can be found on GitHub here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions).

The `TestActor` Blueprint asset can be found here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition).

# Summary

By completing this chapter, you have taken the first step in your game development journey by learning about UE5\. You now know how to navigate the Unreal Engine editor, manipulate the Actors inside a level, create Actors, use the Blueprint scripting language, and how 3D objects are represented in UE5.

Hopefully, you realize that there’s a whole world of possibilities ahead of you and that the sky is the limit in terms of the things you can create using this game development tool.

In the next chapter, you will recreate the project template that was automatically generated in this chapter from scratch. You will learn how to create C++ classes and then create Blueprint classes that can manipulate properties declared in their parent class. You will also learn how to import character meshes and animations into Unreal Engine 5, as well as become familiar with other animation-related assets such as *Animation Blueprints*.