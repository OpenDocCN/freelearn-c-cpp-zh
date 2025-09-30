# Preface

*Game Development with Godot and Blender is a comprehensive introduction for those who are new to building 3D models and games, allowing you to leverage the abilities of these two technologies to create dynamic, interactive, and engaging games.*

*This book will start by focusing on what low-poly modeling actually is, before diving into using Blender to create, rig, and animate our models. We will also make sure that these assets are game-ready, making it easy for you to import them into Godot and use your assets effectively and efficiently. Then, in Godot, you will use the game engine to design scenes, work with light and shadows, and transform your 3D models into interactive, controllable assets.*

*By the end of the book, you will have a seamless workflow between Blender and Godot that is specifically geared towards game development and will have created a point-and-click adventure game following our instructions and guidance. Beyond this point, you should be able to take these newly acquired skills and create your own 3D games from conception to completion!*

# Who this book is for

*This book is for game developers who are looking to make the transition from 2D to 3D games. You should have a basic understanding of Godot, and be able to navigate the UI, understand the Inspector panel, create scenes, add scripts to game objects, and so on. Previous experience with Blender is helpful but not required.*

# What this book covers

[*Chapter 1*](B17473_01.xhtml#_idTextAnchor013), *Creating Low-Poly Models*, covers the creation of low-poly models in Blender. You’ll also look at how to utilize modifiers to expedite the process.

[*Chapter 2*](B17473_02.xhtml#_idTextAnchor032), *Building Materials and Shaders*, shows you how to create and assign different materials to your models, and understand where shaders come into play.

[*Chapter 3*](B17473_03.xhtml#_idTextAnchor042), *Adding and Creating Textures*, teaches you how to prepare your models for texturing. Applying third-party textures and creating your own are also covered in this chapter.

[*Chapter 4*](B17473_04.xhtml#_idTextAnchor060), *Adjusting Cameras and Lights*, presents different light types and how to capture a shot of your scene. You’ll be revisiting some of these notions in the Godot context later in [*Chapter 10*](B17473_10.xhtml#_idTextAnchor165), *Making Things Look Better with Lights and Shadows*.

[*Chapter 5*](B17473_05.xhtml#_idTextAnchor075), *Setting up Animation and Rigging*, discusses the notion of animation and whether doing it in Godot or Blender is the right choice. Once we settle the matter in Blender’s favor, you’ll rig and animate a simple model.

[*Chapter 6*](B17473_06.xhtml#_idTextAnchor092), *Exporting Blender Assets*, tackles a most crucial and often ignored topic: exporting your models from Blender. You’ll be specifically shown a format that is the most suitable for Godot Engine.

[*Chapter 7*](B17473_07.xhtml#_idTextAnchor112), *Importing Blender Assets into Godot*, conveniently shows how to import your models into Godot. The transition between different applications is not always smooth, so you’ll also be presented with shortcomings and workarounds.

[*Chapter 8*](B17473_08.xhtml#_idTextAnchor129), *Adding Sound Assets*, investigates the use of sound in Godot Engine. You’ll partake in a short exercise to play a sound file after discovering different types of audio files the engine supports.

[*Chapter 9*](B17473_09.xhtml#_idTextAnchor146), *Designing the Level*, will be the beginning of a series of exercises for building a point-and-click adventure game. To kick off the effort, you’ll be designing the level with the models that come within the GitHub repository.

[*Chapter 10*](B17473_10.xhtml#_idTextAnchor165), *Making Things Look Better with Lights and Shadows*, presents different light types you can deploy in your level to enhance the look and feel of the game. To complement the scene further, you’ll also discover the use of global illumination and post-processing effects.

[*Chapter 11*](B17473_11.xhtml#_idTextAnchor186), *Creating the User Interface*, discusses the necessity of user interfaces. Then, you’ll utilize a bunch of Godot UI components to compose a piece of note. Last but not least, you’ll investigate why creating themes in Godot might be a time-saver.

[*Chapter 12*](B17473_12.xhtml#_idTextAnchor206), *Interacting with the World through Camera and Character Controllers*, presents different camera types and settings on different gaming platforms. After attaining a basic view into the game world, you’ll continue with detecting user input, which is essential for the type of game you are building. To finish off, you’ll use this information to move a game character to their designated spot.

[*Chapter 13*](B17473_13.xhtml#_idTextAnchor230), *Finishing with Sound and Animation*, finishes the core mechanics of our little game. To that end, you’ll be adding sound effects and animations to certain game objects. Also, you’ll create a simple animation in Godot and create the necessary conditions for the player to meet in order to trigger this animation. Once all the in-game requirements are finished, you’ll load a new level for the player.

[*Chapter 14*](B17473_14.xhtml#_idTextAnchor255), *Conclusion*, shows how to export your game to Windows, so you can share it with the world. You’ll finish this chapter and the book off by getting to know what else Godot can offer to you.

# To get the most out of this book

*You will need the Windows versions of Blender 2.93 and Godot 3.4.4 installed on your computer. All the visual examples and code samples have been tested for these versions. If you have newer or older versions installed, you might notice discrepancies.*

![](img/B17473_Preface_Table.jpg)

*Knowing how to use GitHub at a basic level might help. Alternatively, you can download the whole repository and work with your local copy.*

**If you are using the digital version of this book, we advise you to type the code yourself or access the code from the book’s GitHub repository (a link is available in the next section). Doing so will help you avoid any potential errors related to the copying and pasting of code.**

# Download the example code files

You can download the example code files for this book from GitHub at [https://github.com/PacktPublishing/Game-Development-with-Blender-and-Godot](https://github.com/PacktPublishing/Game-Development-with-Blender-and-Godot). If there’s an update to the code, it will be updated in the GitHub repository.

We also have other code bundles from our rich catalog of books and videos available at [https://github.com/PacktPublishing/](https://github.com/PacktPublishing/). Check them out!

# Download the color images

We also provide a PDF file that has color images of the screenshots and diagrams used in this book. You can download it here: [https://packt.link/0KyZi](https://packt.link/0KyZi).

# Conventions used

There are a number of text conventions used throughout this book.

`Code in text`: Indicates code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles. Here is an example: “If you increase the radius to `10.0`, something interesting will happen.”

A block of code is set as follows:

```cpp
extends AudioStreamPlayer
func _unhandled_key_input(event: InputEventKey) -> void:
    if event.is_pressed() and event.scancode == KEY_SPACE:
        stream_paused = false
    else:
        stream_paused = true
```

**Bold**: Indicates a new term, an important word, or words that you see onscreen. For instance, words in menus or dialog boxes appear in **bold**. Here is an example: “When you applied the **Solidify** modifier, you must have seen that there are so many other modifiers.”

Tips or important notes

Appear like this.

# Get in touch

Feedback from our readers is always welcome.

**General feedback**: If you have questions about any aspect of this book, email us at [customercare@packtpub.com](mailto:customercare@packtpub.com) and mention the book title in the subject of your message.

**Errata**: Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you have found a mistake in this book, we would be grateful if you would report this to us. Please visit [www.packtpub.com/support/errata](http://www.packtpub.com/support/errata) and fill in the form.

**Piracy**: If you come across any illegal copies of our works in any form on the internet, we would be grateful if you would provide us with the location address or website name. Please contact us at [copyright@packt.com](mailto:copyright@packt.com) with a link to the material.

**If you are interested in becoming an author**: If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, please visit [authors.packtpub.com](http://authors.packtpub.com).

# Part 1: 3D Assets with Blender

This part of the book provides you a detailed look into how to create models, textures, and animation in Blender. By the end of this part, you’ll be able to create game-ready assets.

In this part, we cover the following chapters:

*   [*Chapter 1*](B17473_01.xhtml#_idTextAnchor013)*, Creating Low-Poly Models*
*   [*Chapter 2*](B17473_02.xhtml#_idTextAnchor032)*, Building Materials and Shaders*
*   [*Chapter 3*](B17473_03.xhtml#_idTextAnchor042)*, Adding and Creating Textures*
*   [*Chapter 4*](B17473_04.xhtml#_idTextAnchor060)*, Adjusting Cameras and Lights*
*   [*Chapter 5*](B17473_05.xhtml#_idTextAnchor075)*, Setting Up Animation and Rigging*