# Preface

Immerse yourself in Unreal game projects with this book, written by four highly experienced industry professionals with many years of combined experience with Unreal Engine. *Elevating Game Experiences with Unreal Engine 5* will walk you through the latest version of Unreal Engine by helping you get hands-on with the game creation projects.The book starts with an introduction to the Unreal Editor and key concepts such as actors, blueprints, animations, inheritance, and player input. You’ll then move on to the first of three projects – building a dodgeball game, where you’ll learn the concepts of line traces, collisions, projectiles, user interface, and sound effects. You’ll also discover how to combine these concepts to showcase your new skills. The second project, a side-scroller game, will help you implement concepts such as animation blending, enemy AI, spawning objects, and collectibles. And finally, you’ll cover the key concepts in creating a multiplayer environment as you work on the third project – an FPS game. By the end of the Unreal Engine book, you’ll have a broad understanding of how to use the tools that the game engine provides to start building your own games.

# Who this book is for

This book is for game developers looking to get started with using Unreal Engine 5 for their game development projects. Anyone who has used Unreal Engine before and wants to consolidate, improve, and apply their skills will find this book useful. To better grasp the concepts explained in this book, prior knowledge of C++ basics (such as variables, functions, classes, polymorphism, and pointers) is required. For full compatibility with the IDE used in this book, a Windows system is recommended.

# What this book covers

[*Chapter 1*](B18531_01.xhtml#_idTextAnchor016)*, Introduction to Unreal Engine*, explores the Unreal Engine editor. You will be introduced to the editor’s interface, see how to manipulate actors in a level, understand the basics of the blueprint visual scripting language, and discover how to create material assets that can then be used by meshes.

[*Chapter 2*](B18531_02.xhtml#_idTextAnchor043)*, Working with Unreal Engine*, introduces Unreal Engine game fundamentals, along with how to create a C++ project and set up the Content Folder of projects. You’ll also be introduced to the topic of animations.

[*Chapter 3*](B18531_03.xhtml#_idTextAnchor090)*, Character Class Components and Blueprint Setup*, introduces you to the Unreal Character class, along with the concept of object inheritance and how to work with input mappings.

[*Chapter 4*](B18531_04.xhtml#_idTextAnchor099)*, Getting Started with Player Input*, introduces the topic of player input. You will learn how to associate a key press or a touch input with an in-game action, such as jumping or moving, through the use of action mappings and axis mappings.

[*Chapter 5*](B18531_05.xhtml#_idTextAnchor113)*, Query with Line Traces*, starts a new project called Dodgeball. In this chapter, you will learn about the concept of line traces and the various ways in which they can be used in games.

[*Chapter 6*](B18531_06.xhtml#_idTextAnchor134)*, Setting Up Collision Objects*, explores the topic of object collision. You will learn about collision components, collision events, and physics simulation. You will also study the topic of timers, the projectile movement component, and physical materials.

[*Chapter 7*](B18531_07.xhtml#_idTextAnchor154)*, Working with UE5 Utilities*, teaches you how to implement some useful utilities available in Unreal Engine, including actor components, interfaces, and blueprint function libraries, which will help keep your projects well structured and approachable for other people that join your team.

[*Chapter 8*](B18531_08.xhtml#_idTextAnchor168)*, Creating User Interfaces with UMG*, explores the topic of game UI. You will learn how to make menus and HUDs using Unreal Engine’s UI system, UMG, as well as how to display the player character’s health using a progress bar.

[*Chapter 9*](B18531_09.xhtml#_idTextAnchor183)*, Adding Audio-Visual Elements*, introduces the topic of sounds and particle effects in Unreal Engine. You will learn how to import sound files to the project and use them as both 2D and 3D sounds, as well as how to add existing particle systems to the game. Lastly, a new level will be made that uses all the game mechanics built in the last few chapters to conclude the Dodgeball project.

[*Chapter 10*](B18531_10.xhtml#_idTextAnchor199)*, Creating the SuperSideScroller Game*, discusses in detail the goals of the SuperSideScroller project and covers an overview of how animation works in Unreal Engine 5 through examples of manipulating the default Mannequin Skeleton.

[*Chapter 11*](B18531_11.xhtml#_idTextAnchor222)*, Working with Blend Space 1D, Key Bindings, and State Machines*, teaches you how to use Blend Space 1D, Animation State Machines, and the Enhanced Input System in Unreal Engine 5 to create working movement-based animation logic for the player character.

[*Chapter 12*](B18531_12.xhtml#_idTextAnchor247)*, Animation Blending and Montages*, discusses further animation concepts in Unreal Engine 5 such as Animation Blending and Animation Montages to allow for concurrent animation to occur when the player character moves and throws the projectile.

[*Chapter 13*](B18531_13.xhtml#_idTextAnchor268)*, Creating and Adding the Enemy Artificial Intelligence*, teaches you how to use AI Controller, Blackboards, and Behavior Trees in Unreal Engine 5 to create simple AI logic for an enemy that the player can face.

[*Chapter 14*](B18531_14.xhtml#_idTextAnchor298)*, Spawning the Player Projectile*, teaches you how to spawn and destroy game objects, and uses additional animation-based concepts such as Anim Notifies and Anim Notify states to spawn the player projectile during the throwing animation.

[*Chapter 15*](B18531_15.xhtml#_idTextAnchor322)*, Exploring Collectibles, Power-Ups, and Pickups*, introduces you to UI concepts of UMG in Unreal Engine 5 and puts your knowledge to the test in creating additional collectibles and power-ups for the player.

[*Chapter 16*](B18531_16.xhtml#_idTextAnchor345)*, Getting Started with Multiplayer Basics*, introduces you to multiplayer and how the server/client architecture works, as well as covering concepts such as connections, ownership, roles, and variable replication. It also covers 2D Blend Spaces to create an animation grid for 2D movement and the Transform Modify Bone control to change the transform of a bone at runtime.

[*Chapter 17*](B18531_17.xhtml#_idTextAnchor386)*, Using Remote Procedure Calls*, teaches you how remote procedure calls work, the different types, and important considerations when using them. It also shows how to expose enumerations to the editor and how to use array index wrapping to cycle between an array in both directions.

[*Chapter 18*](B18531_18.xhtml#_idTextAnchor404)*, Using Gameplay Framework Classes in Multiplayer*, explains how to use the most important classes in the Gameplay Framework in multiplayer. It also explains more about Game Mode, Player State, Game State, and some useful engine built-in functionality.

## To get the most out of this book

![](img/B18531_Preface_Table.jpg)

To access the files of the Unreal Engine GitHub repository linked throughout this book, make sure to follow these instructions:

[https://www.unrealengine.com/en-US/ue-on-github](https://www.unrealengine.com/en-US/ue-on-github)

If you get an error 404 on a link from this book to the Unreal Engine documentation, it means that it hasn't been updated yet to 5.0\. You should pick the previous engine version from the dropdown on the top left corner of the page.

## Installing Visual Studio

Because we’ll be using C++ while working with Unreal Engine 5, we’ll need an **Integrated Development Environment** (**IDE**) that easily works alongside the engine. Visual Studio Community is the best IDE you have available for this purpose on Windows. If you’re using macOS or Linux, you’ll have to use another IDE, such as Visual Studio Code, Qt Creator, or Xcode (available exclusively on macOS).

The instructions given in this book are specific to Visual Studio Community on Windows, so if you are using a different OS and/or IDE, then you will need to do your research on how to set these up for use in your working environment. In this section, you’ll be taken through the installation of Visual Studio, so that you can easily edit UE5’s C++ files:

1.  Go to the Visual Studio download web page at [https://visualstudio.microsoft.com/downloads](https://visualstudio.microsoft.com/downloads). The recommended Visual Studio Community version for the Unreal Engine 5 version we’ll be using in this book (5.0.3) is Visual Studio Community 2022\. Be sure to download that version.
2.  When you do, open the executable file that you just downloaded. It should eventually take you to a window where you’ll be able to pick the modules of your Visual Studio installation. There, you’ll have to tick the **Game Development with C++** module and then click the **Install** button in the bottom-right corner of the window. After you click that button, Visual Studio will start downloading and installing. When the installation is complete, it may ask you to reboot your PC. After rebooting your PC, Visual Studio should be installed and ready for use.
3.  When you run Visual Studio for the first time, you may see a few windows, the first one of which is the login window. If you have a Microsoft Outlook/Hotmail account, you should use that account to log in, otherwise, you can skip login by clicking **Not now, maybe later**.

Note

If you don’t input an email address, you will only have 30 days to use Visual Studio before it locks out and you have to input an email address to continue using it.

1.  After that, you will be asked to choose a color scheme. The **Dark** theme is the most popular and the one we will be using in this section.

Finally, you can select the **Start Visual Studio** option. When you do so, however, you can close it again. We will be taking a deeper look at how to use Visual Studio in [*Chapter 2*](B18531_02.xhtml#_idTextAnchor043)*, Working with Unreal Engine* of this book.

## Epic Games Launcher

To access Unreal Engine 5, you’ll need to download the Epic Games Launcher, available at this link: [https://store.epicgames.com/en-US/download](https://store.epicgames.com/en-US/download).

Before you do so, be sure to check its hardware requirements at this link: [https://docs.unrealengine.com/5.0/en-US/hardware-and-software-specifications-for-unreal-engine/](https://docs.unrealengine.com/5.0/en-US/hardware-and-software-specifications-for-unreal-engine/).

This link will allow you to download the Epic Games Launcher for Windows and macOS. If you use Linux, you’ll have to download the Unreal Engine source code and compile it from the source – [https://docs.unrealengine.com/5.0/en-US/downloading-unreal-engine-source-code/](https://docs.unrealengine.com/5.0/en-US/downloading-unreal-engine-source-code/):

1.  Click the `.msi` file will be downloaded to your computer. Open this `.msi` file when it finishes downloading, which will prompt you to install the Epic Games Launcher. Follow the installation instructions and then launch the Epic Games Launcher. When you do so, you should be greeted with a login screen.
2.  If you already have an account, you can simply log in with your existing credentials. If you don’t, you’ll have to sign up for an Epic Games account by clicking the **Sign Up** text at the bottom.

Once you log in with your account, you should be greeted by the **Home** tab. From there, you’ll want to go to the **Unreal Engine** tab by clicking the text that says **Unreal Engine**.

1.  When you’ve done that, you’ll be greeted with the **Store** tab. The Epic Games Launcher is not only the place from which you install and launch Unreal Engine 5, but it’s also a game store. Press the **Unreal Engine** tab on the left side of the launcher.
2.  You will now find several sub-tabs at the top of the Epic Games Launcher, the first of which is the **News** sub-tab. This acts as a hub for Unreal Engine resources. From this page, you’ll be able to access the following:
    *   The **News** page, on which you’ll be able to take a look at all the latest Unreal Engine news
    *   The **YouTube** channel, on which you’ll be able to watch dozens of tutorials and live streams that go into detail about several different Unreal Engine topics
    *   The **Q&A** page, on which you’ll be able to see, ask, and answer questions posed and answered by the Unreal Engine community
    *   The **Forums** page, on which you’ll be able to access the Unreal Engine forums
    *   The **Roadmap** page, on which you’ll be able to access the Unreal Engine roadmap, including features delivered in past versions of the engine, as well as features that are currently in development for future versions
3.  The **Samples** tab will allow you to access several project samples that you can use to learn how to use Unreal Engine 5\.
4.  To the right of the **Samples** tab is the **Marketplace** tab. This tab shows you several assets and code plugins made by members of the Unreal Engine community. Here, you’ll be able to find 3D assets, music, levels, and code plugins that will help you advance and accelerate the development of your game.
5.  To the right of the **Marketplace** tab, we have the **Library** tab. Here, you’ll be able to browse and manage all your Unreal Engine version installations, your Unreal Engine projects, and your **Marketplace** asset vault. Because we have none of these things yet, these sections are all empty. Let’s change that.
6.  Click the yellow plus sign to the right of the **ENGINE VERSIONS** text. This should make a new icon show up, where you’ll be able to choose your desired Unreal Engine version.
7.  Throughout this book, we’ll be using version *5.0* of Unreal Engine. After you’ve selected that version, click the **Install** button:

![](img/Figure_Preface_1.01_B18531.jpg)

Figure Preface 1.1 – The icon that allows you to install Unreal Engine 5.0

1.  After you’ve done this, you’ll be able to choose the installation directory for this Unreal Engine version, which will be of your choosing, and you should then click the **Install** button again.

Note

If you are having issues installing the 5.0 version, make sure to install it on your D drive, with the shortest path possible (that is, don’t try to install it too many folders deep, and make sure those folders have short names).

1.  This will result in the installation of Unreal Engine 5.0 starting. When the installation is done, you can launch the editor by clicking the **Launch** button of the version icon:

![](img/Figure_Preface_1.02_B18531.jpg)

Figure Preface 1.2 – The version icon once installation has finished

If you are using the digital version of this book, we advise you to type the code yourself or access the code from the book’s GitHub repository (a link is available in the next section). Doing so will help you avoid any potential errors related to the copying and pasting of code.

# Download the example code files

You can download the example code files for this book from GitHub at [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition). If there’s an update to the code, it will be updated in the GitHub repository.

We also have other code bundles from our rich catalog of books and videos available at [https://github.com/PacktPublishing/](https://github.com/PacktPublishing/). Check them out!

You can download videos for Chapters 1, 3, 4-9, and 16-18 which will help to understand the chapters better. Here is the link for it: [https://packt.link/1GnAS](https://packt.link/1GnAS)

# Download the color images

We also provide a PDF file that has color images of the screenshots and diagrams used in this book. You can download it here: [https://packt.link/iAmVj](https://packt.link/iAmVj).

# Conventions used

There are a number of text conventions used throughout this book.

Code in text: Indicates code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles. Here is an example: ”These lines of code represent the declarations of the `Tick()` and `BeginPlay()` functions that are included in every Actor-based class by default.”

A block of code is set as follows:

```cpp
// Called when the game starts or when spawned
void APlayerProjectile::BeginPlay()
{
  Super::BeginPlay();
}
// Called every frame
void APlayerProjectile::Tick(float DeltaTime)
{
  Super::Tick(DeltaTime);
}
```

Bold: Indicates a new term, an important word, or words that you see onscreen. For instance, words in menus or dialog boxes appear in bold. Here is an example: “From the **Open Level** dialog box, navigate to **/ThirdPersonCPP/Maps** to find **SideScrollerExampleMap**.”

Tips or Important Notes

Appear like this.

# Get in touch

Feedback from our readers is always welcome.

**General feedback**: If you have questions about any aspect of this book, email us at [customercare@packtpub.com](mailto:customercare@packtpub.com) and mention the book title in the subject of your message.

**Errata**: Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you have found a mistake in this book, we would be grateful if you would report this to us. Please visit [www.packtpub.com/support/errata](http://www.packtpub.com/support/errata) and fill in the form.

**Piracy**: If you come across any illegal copies of our works in any form on the internet, we would be grateful if you would provide us with the location address or website name. Please contact us at [copyright@packt.com](mailto:copyright@packt.com) with a link to the material.

If you are interested in becoming an author: If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, please visit [authors.packtpub.com](http://authors.packtpub.com)

# Share Your Thoughts

Once you’ve read *Elevating Game Experiences with Unreal Engine 5*, we’d love to hear your thoughts! Please [click here to go straight to the Amazon review page](https://packt.link/r/1803239867) for this book and share your feedback.

Your review is important to us and the tech community and will help us make sure we’re delivering excellent quality content.