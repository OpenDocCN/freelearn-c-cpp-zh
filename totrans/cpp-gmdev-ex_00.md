# Preface

Computer graphics programming is considered to be one of the hardest subjects to cover, as it involves complex mathematics, programming, and graphics concepts that are intimidating to the average developer. Also, with alternative game engines available, such as Unity and Unreal, it is important to understand graphics programming, as it is a lot easier to make 2D or 3D games using these more sophisticated game engines. These engines also use some rendering APIs, such as OpenGL, Vulkan, Direct3D, and Metal, to draw objects in a scene, and the graphics engine in a game engine constitutes more than 50% of it. Therefore, it is imperative to have some knowledge about graphics programming and graphics APIs.

The objective of this book is to break down this complex subject into bite-sized chunks to make it easy to understand. So, we will start with the basic concepts that are required to understand the math, programming, and graphics basics. 

In the next section of the book, we will create a 2D game, initially with **Simple and Fast Multimedia Library** (**SFML**), which covers the basics that are required to create any game, and with which you can make any game with the utmost ease, without worrying about how a game object is drawn. We will be using SFML just to draw our game objects.

In the next part of the book, we will see how game objects get presented onscreen using OpenGL. OpenGL is a high-level Graphics API that enables us to get something rendered to a scene quickly. A simple sprite created in SFML goes through a lot of steps before actually getting drawn on the screen. We will see how a simple image gets loaded and gets displayed on the screen and what steps are required to do so. But that is just the start. We will see how to add 3D physics to the game and develop a physics-based game from the ground up. Finally, we will add some lighting to make the scene a little more interesting.

With that knowledge of OpenGL, we will dive further into graphics programming and see how Vulkan works. Vulkan is the successor to OpenGL and is a low-level, verbose graphics API. OpenGL is a high-level graphics API that hides a lot of inner workings. With Vulkan, you have complete access to the GPU, and with the Vulkan graphics API, we will learn how to render our game objects.

# Who this book is for

This book is targeted at game developers keen to learn game development with C++ and graphics programming using OpenGL or the Vulkan graphics API. This book is also for those looking to update their existing knowledge of those subjects. Some prior knowledge of C++ programming is assumed.

# What this book covers

[Chapter 1](3592c1fa-996c-4629-bd98-a9f6733447fd.xhtml)*,** C++ Concepts,* covers the basics of C++ programming, which are essential to understand and code the chapters in this book.

[Chapter 2](ee788533-687d-4231-91a4-cb1de9ca01dd.xhtml)*, Mathematics and Graphics Concepts,* In this chapter we cover the basic topics of maths such as vector calculations and knowledge on matrices. These are essential for graphics programming and basic physics programming. We then move on to the basics of graphics programming, starting with how a bunch of vertices is sent to the graphics pipeline and how they are converted into shapes and rendered on the screen.

[Chapter 3](7ae25a2f-fcf6-4501-a5f3-e5b7fb6e27c3.xhtml), *Setting Up Your Game,* introduces the SFML framework, its uses, and its limitations. It also covers creating a Visual Studio project and adding SFML to it, creating a basic window with the basic framework of the game to initialize, update, render, and close it. We will also learn how to draw different shapes and learn how to add a textured sprite to the scene and add keyboard input*.*

[Chapter 4](6ee4094f-a6f2-4dbe-9326-6ae2b2f33fd2.xhtml), *Creating Your Game*, covers the creation of the character class and adding functionality to a character to make them move and jump. We will also create the enemy class to populate enemies for the game. We will add a rockets class so the player can spawn rockets when they fire. Finally, we will add collision detection to detect collisions between two sprites.

[Chapter 5](e3fc199a-6496-42db-9cb6-eb668c5fe9d8.xhtml)*, Finalizing Your Game*, covers finishing the game and adding some polishing touches by adding scoring, text, and audio. We'll also add some animation to the player character to make the game more lively.

[Chapter 6](df91d837-a1b9-4723-a779-792d2d6358fa.xhtml)*, Getting Started with OpenGL*, looks at what OpenGL is, its advantages, and its shortcomings. We'll integrate OpenGL into the Visual Studio project and use GLFW to create a window. We'll create a basic game loop and render our first 3D object using Vertex and fragment shaders.

[Chapter 7](ccdc775e-ac6b-41aa-b861-1eae379feb04.xhtml)*, Building on the Game Objects*, covers adding textures to an object. We'll include the Bullet Physics library in the project to add physics to the scene. We will see how to integrate physics with our 3D OpenGL rendered object. Finally, we will create a basic level to test out the physics.

[Chapter 8](ed5ec7d6-9257-48c4-9f66-3a2aca68eeeb.xhtml)*, Enhancing Your Game with Collision, Loop, and Lighting*, covers adding a game-over condition and finishing the game loop. We will add some finishing touches to the game by adding some basic 3D lighting and text to display the score and game-over condition.

[Chapter 9](91c1d4b1-9d80-472d-9fb6-1b34cffb21b6.xhtml)*, Getting Started with Vulkan*, looks at the need for Vulkan and how it is different from OpenGL. We'll look at the advantages and disadvantages of Vulkan, integrate it into the Visual Studio project, and add GLFW to create a window for the project. We will create an app and a Vulkan instance, and add validation layers to check whether Vulkan is running as required. We'll also get physical device properties and create a logical device.

[Chapter 10](58a38eaf-b67b-4425-b5d6-80efaf4970ad.xhtml), *Preparing the Clear Screen,* covers the creation of a window surface to which we can render the scene. We also need to create a swap chain so that we can ping-pong between the front buffer and back buffer, and create image views and frame buffers to which the views are attached. We will create the draw command buffer to record and submit graphics commands and create a renderpass clear screen. 

[Chapter 11](ebc6fd68-325b-439e-9e11-8e01f818dd9b.xhtml), *Creating Object Resources,* covers creating the resources required to draw the geometry. This includes adding a mesh class that has all the geometry information, including vertex and index data. We'll create object buffers to store the vertex, index, and uniform buffers. We'll also create DescriptorSetLayout and Descriptor Sets, and finally, we'll create shaders and convert them to SPIR-V binary format.

[Chapter 12](dc30df72-df2e-4bb9-a598-a481ecb595f3.xhtml), *Drawing Vulkan Objects,* covers creating the graphics pipeline, in which we set the vertices and enable viewports, multisampling, and depth and stencil testing. We'll also create an object class, which will help create the object buffers, descriptor set, and graphics pipeline for the object. We will create a camera class to view the world through, and then finally render the object. At the end of the chapter, we will also see how to synchronize information being sent.

# To get the most out of this book

The book is designed to be read from the start, chapter by chapter. If you have prior knowledge of the contents of a chapter, then please feel free to skip ahead instead.

It is good to have some prior programming experience with C++, but if not, then there is a chapter on C++ programming, which covers the basics. No prior knowledge of graphics programming is assumed.

To run OpenGL and Vulkan projects, make sure your hardware supports the current version of the API. The book uses OpenGL 4.5 and Vulkan 1.1\. Most GPU vendors support OpenGL and Vulkan, but for a full list of supported GPUs, please refer to the GPU manufacturer or to the wiki, at [https://en.wikipedia.org/wiki/Vulkan_(API)](https://en.wikipedia.org/wiki/Vulkan_(API)).

# Download the example code files

You can download the example code files for this book from your account at [www.packt.com](http://www.packt.com). If you purchased this book elsewhere, you can visit [www.packtpub.com/support](https://www.packtpub.com/support) and register to have the files emailed directly to you.

You can download the code files by following these steps:

1.  Log in or register at [www.packt.com](http://www.packt.com).
2.  Select the Support tab.
3.  Click on Code Downloads.
4.  Enter the name of the book in the Search box and follow the onscreen instructions.

Once the file is downloaded, please make sure that you unzip or extract the folder using the latest version of:

*   WinRAR/7-Zip for Windows
*   Zipeg/iZip/UnRarX for Mac
*   7-Zip/PeaZip for Linux

The code bundle for the book is also hosted on GitHub at [https://github.com/PacktPublishing/CPP-Game-Development-By-Example](https://github.com/PacktPublishing/CPP-Game-Development-By-Example). In case there's an update to the code, it will be updated on the existing GitHub repository.

We also have other code bundles from our rich catalog of books and videos available at **[https://github.com/PacktPublishing/](https://github.com/PacktPublishing/)**. Check them out!

# Download the color images

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. You can download it here:

[https://www.packtpub.com/sites/default/files/downloads/9781789535303_ColorImages.pdf](https://www.packtpub.com/sites/default/files/downloads/9781789535303_ColorImages.pdf)[.](https://www.packtpub.com/sites/default/files/downloads/9781789535303_ColorImages.pdf)

# Conventions used

There are a number of text conventions used throughout this book.

`CodeInText`: Indicates code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles. Here is an example: "Here, the printing of `Hello, World` is tasked to the `main` function."

A block of code is set as follows:

```cpp
#include <iostream>
// Program prints out "Hello, World" to screen
int main()
{
std::cout<< "Hello, World."<<std::endl;
return 0;
} 
```

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

```cpp
int main() {
    //init game objects
        while (window.isOpen()) {
            // Handle Keyboard events
            // Update Game Objects in the scene
    window.clear(sf::Color::Red);
    // Render Game Objects
    window.display();
        }
        return 0;
    }
```

**Bold**: Indicates a new term, an important word, or words that you see onscreen. For example, words in menus or dialog boxes appear in the text like this. Here is an example: "In Input and under Linker, type the following `.lib` files."

Warnings or important notes appear like this.

Tips and tricks appear like this.

# Get in touch

Feedback from our readers is always welcome.

**General feedback**: If you have questions about any aspect of this book, mention the book title in the subject of your message and email us at `customercare@packtpub.com`.

**Errata**: Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you have found a mistake in this book, we would be grateful if you would report this to us. Please visit [www.packtpub.com/support/errata](https://www.packtpub.com/support/errata), selecting your book, clicking on the Errata Submission Form link, and entering the details.

**Piracy**: If you come across any illegal copies of our works in any form on the Internet, we would be grateful if you would provide us with the location address or website name. Please contact us at `copyright@packt.com` with a link to the material.

**If you are interested in becoming an author**: If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, please visit [authors.packtpub.com](http://authors.packtpub.com/).

# Reviews

Please leave a review. Once you have read and used this book, why not leave a review on the site that you purchased it from? Potential readers can then see and use your unbiased opinion to make purchase decisions, we at Packt can understand what you think about our products, and our authors can see your feedback on their book. Thank you!

For more information about Packt, please visit [packt.com](http://www.packt.com/).