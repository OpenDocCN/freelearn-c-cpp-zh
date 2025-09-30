# Preface

So, you want to program your own games using Unreal Engine 4 (UE4). You have a great number of reasons to do so:

*   UE4 is powerful: UE4 provides some of the most state-of-the-art, beautiful, realistic lighting and physics effects, of the kind used by AAA Studios.
*   UE4 is device-agnostic: Code written for UE4 will work on Windows desktop machines, Mac desktop machines, Android devices, and iOS devices (at the time of writing this book—even more devices may be supported in the future).

So, you can use UE4 to write the main parts of your game once, and after that, deploy to iOS and Android Marketplaces without a hitch. (Of course, there will be a few hitches: iOS and Android in app purchases will have to be programmed separately.)

# What is a game engine anyway?

A game engine is analogous to a car engine: the game engine is what drives the game. You will tell the engine what you want, and (using C++ code and the UE4 editor) the engine will be responsible for actually making that happen.

You will build your game around the UE4 game engine, similar to how the body and wheels are built around an actual car engine. When you ship a game with UE4, you are basically customizing the UE4 engine and retrofitting it with your own game's graphics, sounds, and code.

# What will using UE4 cost me?

The answer, in short, is $19 and 5 percent of sales.

"What?" you say. $19?

That's right. For only $19, you get full access to a world class AAA Engine, complete with a source. This is a great bargain, considering the fact that other engines can cost anywhere from $500 to $1,000 for just a single license.

# Why don't I just program my own engine and save the 5 percent?

Take it from me, if you want to create games within a reasonable time frame and you don't have a large team of dedicated engine programmers to help you, you'll want to focus your efforts on what you sell (your game).

Not having to focus on programming a game engine gives you the freedom to think only about how to make the actual game. Not having to maintain and bug-fix your own engine is a load off your mind too.

# A game's overview – the Play-Reward-Growth loop

I want to show you this diagram now because it contains a core concept that many novice developers might miss when writing their first games. A game can be complete with sound effects, graphics, realistic physics, and yet, still not feel like a game. Why is that?

![A game's overview – the Play-Reward-Growth loop](img/00002.jpeg)

Starting at the top of the loop, Play actions committed during the game (such as defeating a monster) result in rewards for the player (such as gold or experience). These rewards, in turn, can be used for in-game Growth (such as stats increases or new worlds to explore). This Growth then drives the gameplay in new and interesting ways. For example, a new weapon can change the basic mechanics of fighting, new spells let you take on groups of monsters with a completely different approach, or new modes of transportation can let you reach areas that were previously inaccessible.

This is the basic core loop that creates interesting gameplay. The key is that Play must result in some kind of Reward—think of glittering gold pieces popping out of nasty baddies. For rewards to have a point, it must result in some kind of Growth in the gameplay. Think about how many new locations were unlocked with the hook shot in *The Legend of Zelda*.

A game that is only Play (without Rewards or Growth) won't feel like a game: it will feel only like a really basic prototype of a game. For example, imagine a flight simulator with just an open world and no goals or objectives as well as without the ability to upgrade your plane or weapons. It wouldn't be much of a game.

A game with only Play and Rewards (but no Growth) will feel primitive and simple. The rewards will not satisfy the player if they cannot be used for anything.

A game with only Play and Growth (without Rewards) will just be seen as a mindless increasing challenge, without giving the player a sense of gratification for his achievements.

A game with all three elements will keep the player engaged with an entertaining Play. The Play has a rewarding result (loot drops and story progression), which results in the Growth of the game world. Keeping this loop in mind while you are devising your game will really help you to design a complete game.

### Tip

A prototype is the proof of concept of a game. Say, you want to create your own unique version of *Blackjack*. The first thing you might do is program a prototype to show how the game will be played.

## Monetization

Something you need to think about early in your game's development is your monetization strategy. How will your game make money? If you are trying to start a company, you have to think of what will be your sources of revenue from early on.

Are you going to try to make money from the purchase price, such as *Jamestown*, *The Banner Saga*, *Castle Crashers*, or *Crypt of the Necrodancer*? Or, will you focus on distributing a free game with in-app purchases, such as *Clash of Clans*, *Candy Crush Saga*, or *Subway Surfers*?

A class of games for mobile devices (for example, builder games on iOS) make lots of money by allowing the user to pay in order to skip Play and jump straight to the rewards and Growth parts of the loop. The pull to do this can be very powerful; many people spend hundreds of dollars on a single game.

## Why C++

UE4 is programmed in C++. To write code for UE4, you must know C++.

C++ is a common choice for game programmers because it offers very good performance combined with object-oriented programming features. It's a very powerful and flexible language.

# What this book covers

[Chapter 1](part0018_split_000.html#H5A42-dd4a3f777fc247568443d5ffb917736d "Chapter 1. Coding with C++"), *Coding with C++*, talks about getting up and running with your first C++ program.

[Chapter 2](part0022_split_000.html#KVCC2-dd4a3f777fc247568443d5ffb917736d "Chapter 2. Variables and Memory"), *Variables and Memory*, talks about how to create, read, and write variables from computer memory.

[Chapter 3](part0024_split_000.html#MSDG1-dd4a3f777fc247568443d5ffb917736d "Chapter 3. If, Else, and Switch"), *If, Else, and Switch*, talks about branching the code: that is, allowing different sections of the code to execute, depending on program conditions.

[Chapter 4](part0029_split_000.html#RL0A2-dd4a3f777fc247568443d5ffb917736d "Chapter 4. Looping"), *Looping*, discusses how we repeat a specific section of code as many times as needed.

[Chapter 5](part0034_split_000.html#10DJ41-dd4a3f777fc247568443d5ffb917736d "Chapter 5. Functions and Macros"), *Functions and Macros*, talks about functions, which are bundles of code that can get called any number of times, as often you wish.

[Chapter 6](part0043_split_000.html#190862-dd4a3f777fc247568443d5ffb917736d "Chapter 6. Objects, Classes, and Inheritance"), *Objects, Classes, and Inheritance*, talks about class definitions and instantiating some objects based on a class definition.

[Chapter 7](part0051_split_000.html#1GKCM1-dd4a3f777fc247568443d5ffb917736d "Chapter 7. Dynamic Memory Allocation"), *Dynamic Memory Allocation*, discusses heap-allocated objects as well as low-level C and C++ style arrays.

[Chapter 8](part0056_split_000.html#1LCVG1-dd4a3f777fc247568443d5ffb917736d "Chapter 8. Actors and Pawns"), *Actors and Pawns*, is the first chapter where we actually delve into UE4 code. We begin by creating a game world to put actors in, and derive an `Avatar` class from a customized actor.

[Chapter 9](part0066_split_000.html#1UU541-dd4a3f777fc247568443d5ffb917736d "Chapter 9. Templates and Commonly Used Containers"), *Templates and Commonly Used Containers*, explores UE4 and the C++ STL family of collections of data, called containers. Often, a programming problem can be simplified many times by selecting the right type of container.

[Chapter 10](part0072_split_000.html#24L8G2-dd4a3f777fc247568443d5ffb917736d "Chapter 10. Inventory System and Pickup Items"), *Inventory System and Pickup Items*, discusses the creation of an inventory system with the ability to pick up new items.

[Chapter 11](part0076_split_000.html#28FAO1-dd4a3f777fc247568443d5ffb917736d "Chapter 11. Monsters"), *Monsters*, teaches how to create monsters that give chase to the player and attack it with weapons.

[Chapter 12](part0080_split_000.html#2C9D02-dd4a3f777fc247568443d5ffb917736d "Chapter 12. Spell Book"), *Spell Book*, teaches how to create and cast spells in our game.

# What you need for this book

To work with this text, you will need two programs. The first is your integrated development environment, or IDE. The second piece of software is, of course, the Unreal Engine itself.

If you are using Microsoft Windows, then you will need Microsoft Visual Studio 2013 Express Edition for Windows Desktop. If you are using a Mac, then you will need Xcode. Unreal Engine can be downloaded from [https://www.unrealengine.com/](https://www.unrealengine.com/).

# Who this book is for

This book is for anyone who wants to write an Unreal Engine application. The text begins by telling you how to compile and run your first C++ application, followed by chapters that describe the rules of the C++ programming language. After the introductory C++ chapters, you can start to build your own game application in C++.

# Conventions

In this book, you will find a number of styles of text that distinguish between different kinds of information. Here are some examples of these styles, and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "The `variableType` is going to tell you what type of data we are going to store in our variable. The `variableName` is the symbol we'll use to read or write that piece of memory".

A block of code is set as follows:

[PRE0]

**New terms** and **important words** are shown in bold. Text that appears on the screen appears like this: From the **File** menu, select **New Project...**

### Note

Extra information that is relevant, but kind of a side note, appears in boxes like this.

### Tip

Tips and tricks appear like this.

# Reader feedback

Feedback from our readers is always welcome. Let us know what you think about this book—what you liked or may have disliked. Reader feedback is important for us to develop titles that you really get the most out of.

To send us general feedback, simply send an e-mail to `<[feedback@packtpub.com](mailto:feedback@packtpub.com)>`, and mention the book title via the subject of your message.

If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, see our author guide on [www.packtpub.com/authors](http://www.packtpub.com/authors).

# Customer support

Now that you are the proud owner of a Packt book, we have a number of things to help you to get the most from your purchase.

## Downloading the example code

You can download the example code files for all Packt books you have purchased from your account at [http://www.packtpub.com](http://www.packtpub.com). If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files e-mailed directly to you.

## Downloading the color images of this book

We also provide you with a PDF file that has color images of the screenshots/diagrams used in this book. The color images will help you better understand the changes in the output. You can download this file from [https://www.packtpub.com/sites/default/files/downloads/6572OT_ColoredImages.pdf](https://www.packtpub.com/sites/default/files/downloads/6572OT_ColoredImages.pdf).

## Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books—maybe a mistake in the text or the code—we would be grateful if you would report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the **errata** **submission** **form** link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded on our website, or added to any list of existing errata, under the Errata section of that title. Any existing errata can be viewed by selecting your title from [http://www.packtpub.com/support](http://www.packtpub.com/support).

## Piracy

Piracy of copyright material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works, in any form, on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at `<[copyright@packtpub.com](mailto:copyright@packtpub.com)>` with a link to the suspected pirated material.

We appreciate your help in protecting our authors, and our ability to bring you valuable content.

## Questions

You can contact us at `<[questions@packtpub.com](mailto:questions@packtpub.com)>` if you are having a problem with any aspect of the book, and we will do our best to address it.