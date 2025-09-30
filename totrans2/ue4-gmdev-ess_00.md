# Preface

The purpose of *Unreal Engine 4 Game Development Essentials* is to teach people interested in using Unreal Engine how to create video games. You will learn what Unreal Engine is and how to download and use it. From there, we will go through the collection of tools available in Unreal Engine 4 including Materials, Blueprints, Matinee, UMG, C++, and more.

# What this book covers

[Chapter 1](ch01.html "Chapter 1. Introduction to Unreal Engine 4"), *Introduction to Unreal Engine 4*, is where we begin our journey on *Unreal Engine 4 Game Development Essentials*. In this chapter, the reader will learn how and where to download Unreal Engine as well as the difference between the source version and launcher version. After the Engine's installation (or compilation, if it was the source version) we will get comfortable with the user interface of Unreal Engine. We will also learn about the basics of Content Browser, BSP, and how to change the splash screen and the icons for your game.

[Chapter 2](ch02.html "Chapter 2. Importing Assets"), *Importing Assets*, teaches how to import your custom FBX assets into Unreal Engine once we get the Engine up and running. You will learn about collisions, materials, and the level of detail.

[Chapter 3](ch03.html "Chapter 3. Materials"), *Materials*, teaches you about the Material editor and some common nodes used to create shaders for your assets. After learning the basics of Material, we will create an example material function that can change the intensity of a normal map.

[Chapter 4](ch04.html "Chapter 4. Post Process"), *Post Process*, continues to post-processing after teaching you about materials. In this chapter, you will learn how to override the default post process settings. After that, we will learn how to add our own post process volume and learn a simple but very powerful feature called LUT. After that, we will create a special material that can be used with post process, and this material will have the ability to highlight user-defined objects in the world.

[Chapter 5](ch05.html "Chapter 5. Lights"), *Lights*, gets us halfway through our *Unreal Engine 4 Game Development Essentials* journey, and this chapter will introduce you to the lighting system. We start of by covering the basics, such as placing lights and going through the common settings. You will then learn more about the Lightmass Global Illumination system, including how to properly prepare a UV channel for your asset to be used with Lightmass. By the end of this chapter, you will learn how to build your scene with Lightmass as well as Lightmass settings.

[Chapter 6](ch06.html "Chapter 6. Blueprints"), *Blueprints*, teaches you what Blueprints are and about the various types of Blueprints that are available in the Engine. Blueprints are Unreal Engine's number one tool that allows artists and designers to quickly prototype their game (or even make one!). You will also learn about the different graph types, such as event graph, function graph, macro graph, and so on, and how to spawn a Blueprint dynamically at runtime.

[Chapter 7](ch07.html "Chapter 7. Matinee"), *Matinee*, looks at the cinematic side of Unreal Engine 4 and the tool associated with it, called Matinee. You will learn what Matinee is, how to create one, and get familiar with the UI. After the basics, we will learn how to manipulate objects in Matinee as well as create a very basic cutscene, which we will trigger using Blueprints.

[Chapter 8](ch08.html "Chapter 8. Unreal Motion Graphics"), *Unreal Motion Graphics*, teaches you to create a basic HUD that shows the health of the player. Unreal Motion Graphics (UMG) is the UI authoring tool in Unreal Engine. UMG is used to create Player HUD, Main Menu, Pause Menu, and so on. You will also learn how to create 3D widgets, which can be placed in the world or attached to an actor class.

[Chapter 9](ch09.html "Chapter 9. Particles"), *Particles*, looks at the extremely powerful and robust tool called cascade particle editor and creates a particle system, as no game is good without good visual effects. We then combine this with simple Blueprint scripting to create randomly bursting particles.

[Chapter 10](ch10.html "Chapter 10. Introduction to Unreal C++"), *Introduction to Unreal C++*, goes over C++ as we draw close to the end of our *Unreal Engine 4 Game Development Essentials* journey. In this chapter, you will learn how to get Visual Studio 2015 Community Edition and learn the basics of C++ by inspecting the Third Person Template character class. We will then extend this class to add support for health and the health regeneration system. You will also learn how to expose variables and functions to Blueprint Editor.

[Chapter 11](ch11.html "Chapter 11. Packaging Project"), *Packaging Project*, brings us to the end of our *Unreal Engine 4 Game Development Essentials* journey. In this final chapter, we will recap all the things we've done, including a few tips, and finally, you will learn how to create a release version of your game.

# What you need for this book

Unreal Engine 4.9 or higher

# Who this book is for

This book is aimed at anyone who is interested in learning game development using Unreal Engine 4\. If you are passionate about developing games and want to know about the essentials of Unreal Engine 4 and its tools, then this book will get you up and running quickly. Unreal Engine 4 will be your next step towards creating next gen video games for all platforms, including mobile and consoles.

# Conventions

In this book, you will find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "This adds or removes a path (it can be a virtual package path such as `\Game\MyContent\` or an absolute path such as `C:\My Contents`) for the engine to monitor new content."

A block of code is set as follows:

[PRE0]

**New terms** and **important words** are shown in bold. Words that you see on the screen, for example, in menus or dialog boxes, appear in the text like this: "Once you log in, you can download the launcher by clicking on the big orange **Download** button under **Get Unreal Engine**."

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

## Downloading the color images of this book

We also provide you with a PDF file that has color images of the screenshots/diagrams used in this book. The color images will help you better understand the changes in the output. You can download this file from [http://www.packtpub.com/sites/default/files/downloads/UnrealEngine4GameDevelopment Essentials_ColorImages.pdf](http://www.packtpub.com/sites/default/files/downloads/UnrealEngine4GameDevelopment Essentials_ColorImages.pdf).

## Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books—maybe a mistake in the text or the code—we would be grateful if you could report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the **Errata Submission Form** link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded to our website or added to any list of existing errata under the Errata section of that title.

To view the previously submitted errata, go to [https://www.packtpub.com/books/content/support](https://www.packtpub.com/books/content/support) and enter the name of the book in the search field. The required information will appear under the **Errata** section.

## Piracy

Piracy of copyrighted material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works in any form on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at `<[copyright@packtpub.com](mailto:copyright@packtpub.com)>` with a link to the suspected pirated material.

We appreciate your help in protecting our authors and our ability to bring you valuable content.

## Questions

If you have a problem with any aspect of this book, you can contact us at `<[questions@packtpub.com](mailto:questions@packtpub.com)>`, and we will do our best to address the problem.