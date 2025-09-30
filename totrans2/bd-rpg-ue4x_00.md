# Preface

Now that Unreal Engine 4 has become one of the most cutting-edge game engines in the world, developers both AAA and Indie alike are looking for the best ways of creating games of any genre using the engine. Upon Unreal's first release, it was known as a great first-person shooter game engine, but with the success of games such as WB's *Mortal Kombat*, Chair Entertainment's *Shadow Complex*, and Epic Games' *Gears of War*, along with highly anticipated upcoming games such as Capcom's *Street Fighter 5*, Comcept's *Mighty No. 9*, and Square Enix's *Final Fantasy VII Remake*, Unreal has proven itself to be one of the greatest engines to use when creating virtually any genre of game. This book will lay the foundations of creating a turn-based RPG in Unreal Engine 4.

# What this book covers

[Chapter 1](ch01.html "Chapter 1. Getting Started with RPG Design in Unreal"), *Getting Started with RPG Design in Unreal*, alerts the reader to the various preparation steps required to make an RPG before jumping into Unreal. In order to avoid potential obstacles to progress, the example content is provided and briefly introduced.

[Chapter 2](ch02.html "Chapter 2. Scripting and Data in Unreal"), *Scripting and Data in Unreal*, walks the reader through using C++ to program gameplay elements in Unreal, creating Blueprint graphs, and working with custom game data in Unreal.

[Chapter 3](ch03.html "Chapter 3. Exploration and Combat"), *Exploration and Combat*, walks the reader through creating a character that runs around the game world, defining character data and party members, defining enemy encounters, and creating a basic combat engine.

[Chapter 4](ch04.html "Chapter 4. Pause Menu Framework"), *Pause Menu Framework*, covers how to create a pause menu with inventory and equipment submenus.

[Chapter 5](ch05.html "Chapter 5. Bridging Character Statistics"), *Bridging Character Statistics*, covers how to keep track of the player's stats within the menu system.

[Chapter 6](ch06.html "Chapter 6. NPCs and Dialog"), *NPCs and Dialog*, covers adding interactive NPCs and dialogue to the game world. The reader will learn how to use Blueprints to define what happens when an object or NPC is interacted with, including using a set of custom Blueprint nodes to create dialogue trees.

[Chapter 7](ch07.html "Chapter 7. Gold, Items, and a Shop"), *Gold, Items, and a Shop*, covers adding interactive NPCs and objects to the game world. The reader will learn how to use Blueprint to define what happens when an object or NPC is interacted with, including using a set of custom Blueprint nodes to create dialogue trees. The user will also be creating items that can be bought in a shop using the gold dropped by enemies.

[Chapter 8](ch08.html "Chapter 8. Inventory Population and Item Use"), *Inventory Population and Item Use*, covers populating an inventory screen with items and using the items when not in combat.

[Chapter 9](ch09.html "Chapter 9. Equipment"), *Equipment*, covers the creation of equipment and equipping weapons and armor from an equipment screen.

[Chapter 10](ch10.html "Chapter 10. Leveling, Abilities, and Saving Progress"), *Leveling, Abilities, and Saving Progress*, covers adding abilities to the game, keeping track of experience for each party member, awarding experience to party members after combat, defining leveling and stat updates for a character class, and saving and loading player progress.

# What you need for this book

The required software: all chapters require Unreal Engine 4 version 4.12 or above along with either Visual Studio 2015 Enterprise/Community or above or XCode 7.0 or above.

The required OS: Windows 7 64-bit or above, or Mac OS X 10.9.2.

The required hardware: Quad-core 2.5 GHz or faster, 8 GB of RAM, and NVidia GeForce 470 GTX or AMD Radeon 6870 HD or above.

# Who this book is for

If you are new to Unreal Engine and always wanted to script an RPG, you are this book's target reader. The lessons assume that you understand the conventions of RPG games and have some awareness of the basics of using the Unreal editor to build levels. By the end of this book, you will be able to build upon core RPG framework elements to create your own game experience.

# Conventions

In this book, you will find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "We can include other contexts through the use of the `include` directive."

A block of code is set as follows:

[PRE0]

Any command-line input or output is written as follows:

[PRE1]

**New terms** and **important words** are shown in bold. Words that you see on the screen, for example, in menus or dialog boxes, appear in the text like this: "Compile and save the Blueprint and then press **Play**."

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

## Downloading the color images of this book

We also provide you with a PDF file that has color images of the screenshots/diagrams used in this book. The color images will help you better understand the changes in the output. You can download this file from [http://www.packtpub.com/sites/default/files/downloads/BuildingAnRPGWithUnreal_ColorImages.pdf](http://www.packtpub.com/sites/default/files/downloads/BuildingAnRPGWithUnreal_ColorImages.pdf).

## Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books—maybe a mistake in the text or the code—we would be grateful if you could report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the **Errata Submission Form** link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded to our website or added to any list of existing errata under the Errata section of that title.

To view the previously submitted errata, go to [https://www.packtpub.com/books/content/support](https://www.packtpub.com/books/content/support) and enter the name of the book in the search field. The required information will appear under the **Errata** section.

## Piracy

Piracy of copyrighted material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works in any form on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at `<[copyright@packtpub.com](mailto:copyright@packtpub.com)>` with a link to the suspected pirated material.

We appreciate your help in protecting our authors and our ability to bring you valuable content.

## Questions

If you have a problem with any aspect of this book, you can contact us at `<[questions@packtpub.com](mailto:questions@packtpub.com)>`, and we will do our best to address the problem.