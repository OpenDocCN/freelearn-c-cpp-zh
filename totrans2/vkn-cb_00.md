# Preface

Computer graphics have a very long and interesting history. Many APIs or custom approaches to the generation of 2D or 3D images have come and gone. A landmark in this history was the invention of OpenGL, one of the first graphics libraries, which allowed us to create real‑time, high-performance 3D graphics, and which was available for everyone on multiple operating systems. It is still developed and widely used even today. And this year we can celebrate its 25^(th) birthday!

But many things have changed since OpenGL was created. The graphics hardware industry is evolving very quickly. And recently, to accommodate these changes, a new approach to 3D graphics rendering was presented. It took the form of a low‑level access to the graphics hardware. OpenGL was designed as a high-level API, which allows users to easily render images on screen. But this high‑level approach, convenient for users, is difficult for graphics drivers to handle. This is one of the main reasons for restricting the hardware to show its full potential. The new approach tries to overcome these struggles–it gives users much more control over the hardware, but also many more responsibilities. This way application developers can release the full potential of the graphics hardware, because the drivers no longer block them. Low‑level access allows drivers to be much smaller, much thinner. But these benefits come at the expense of much more work that needs to done by the developers.

The first evangelist of the new approach to graphics rendering was a Mantle API designed by AMD. When it proved that low‑level access can give considerable performance benefits, other companies started working on their own graphics libraries. One of the most notable representatives of the new trend were Metal API, designed by Apple, and DirectX 12, developed by Microsoft.

But all of the above libraries were developed with specific operating systems and/or hardware in mind. There was no open and multiplatform standard such as OpenGL. Until last year. Year 2016 saw the release of the Vulkan API, developed by Khronos consortium, which maintains the OpenGL library. Vulkan also represents the new approach, a low‑level access to the graphics hardware, but unlike the other libraries it is available for everyone on multiple operating systems and hardware platforms–from high‑performance desktop computers with Windows or Linux operating systems, to mobile devices with Android OS. And as it is still being very new, there are few resources teaching developers how to use it. This book tries to fill this gap.

# What this book covers

[Chapter 1](d10e8284-6122-4d0a-8f86-ab0bc0bba47e.xhtml), *Instance and Devices*, shows how to get started with the Vulkan API. This chapter explains where to download the Vulkan SDK from, how to connect with the Vulkan Loader library, how to select the physical device on which operations will be performed, and how to prepare and create a logical device.

[Chapter 2](45eb1180-672a-4745-bd85-f13c7bb658b7.xhtml), *Image Presentation*, describes how to display Vulkan‑generated images on screen. It explains what a swapchain is and what parameters are required to create it, so we can use it for rendering and see the results of our work.

[Chapter 3](fc38e0ae-51aa-4f6f-8fb3-551861273018.xhtml), *Command Buffers and Synchronization*, is about recording various operations into command buffers and submitting them to queues, where they are processed by the hardware. Also, various synchronization mechanisms are presented in this chapter.

[Chapter 4](f1332ca0-b5a2-49bd-ac41-e37068e31042.xhtml), *Resources and Memory*, presents two basic and most important resource types, images and buffers, which allow us to store data. We explain how to create them, how to prepare memory for these resources, and, also, how to upload data to them from our application (CPU).

[Chapter 5](fe2cb528-9d22-49db-a05b-372bce2f87ee.xhtml), *Descriptor Sets*, explains how to provide created resource to shaders. We explain how to prepare resources so they can be used inside shaders and how to set up descriptor sets, which form the interface between the application and the shaders.

[Chapter 6](2de4339d-8912-440a-89a6-fd1f84961448.xhtml), *Render Passes and Framebuffers*, shows how to organize drawing operations into sets of separate steps called subpasses, which are organized into render passes. In this chapter we also show how to prepare descriptions of attachments (render targets) used during drawing and how to create framebuffers, which bind specific resources according to these descriptions.

[Chapter 7](97217f0d-bed7-4ae1-a543-b4d599f299cf.xhtml), *Shaders*, describes the specifics of programming all available graphics and compute shader stages. This chapter presents how to implement shader programs using GLSL programming language and how to convert them into SPIR‑V assemblies – the only form core Vulkan API accepts.

[Chapter 8](5744ea05-b18a-4f84-a1df-250b549dfea5.xhtml), *Graphics and Compute Pipelines*, presents the process of creating two available pipeline types. They are used to set up all the parameters graphics hardware needs to properly process drawing commands or computational work.

[Chapter 9](0a69f5b5-142e-422b-aa66-5cb09a6467b3.xhtml), *Command Recording and Drawing*, is about recording all the operations needed to successfully draw 3D models or dispatch computational work. Also, various optimization techniques are presented in this chapter, which can help increase the performance of the application.

[Chapter 10](1b6b28e0-2101-47a4-8551-c30eb9bfb573.xhtml), *Helper Recipes*, shows convenient set of tools no 3D rendering application can do without. It is shown how to load textures and 3D models from files and how to manipulate the geometry inside shaders.

[Chapter 11](45108a92-6d49-4759-9495-3f1166e69128.xhtml), *Lighting*, presents commonly used lighting techniques from simple diffuse and specular lighting calculations to normal mapping and shadow mapping techniques.

[Chapter 12](82938796-18a3-413b-b05d-47816d70e49a.xhtml), *Advanced Rendering Techniques*, explains how to implement impressive graphics techniques, which can be found in many popular 3D applications such as games and benchmarks.

# What you need for this book

This book explains various aspects of the Vulkan graphics API, which is open and multiplatform. It is available on Microsoft Windows (version 7 and newer) or Linux (preferably Ubuntu 16.04 or newer) systems. (Vulkan is also supported on Android devices with the 7.0+ / Nougat version of the operating system, but the code samples available with this book weren’t designed to be executed on the Android OS.)

To execute sample programs or to develop our own applications, apart from Windows 7+ or Linux operating systems, graphics hardware and drivers that support Vulkan API are also required. Refer to 3D graphics vendors’ sites and/or support to check which hardware is capable of running Vulkan‑enabled software.

When using the Windows operating system, code samples can be compiled using the Visual Studio Community 2015 IDE (or newer), which is free and available for everyone. To generate a solution for the Visual Studio IDE the CMAKE 3.0 or newer is required.

On Linux systems, compilation is performed using a combination of the CMAKE 3.0 and the make tool. But the samples can also be compiled using other tools such as QtCreator.

# Who this book is for

This book is ideal for developers who know C/C++ languages, have some basic familiarity with graphics programming, and now want to take advantage of the new Vulkan API in the process of building next generation computer graphics. Some basic familiarity with Vulkan would be useful to follow the recipes. OpenGL developers who want to take advantage of the Vulkan API will also find this book useful.

# Sections

In this book, you will find several headings that appear frequently (Getting ready, How to do it, How it works, There's more, and See also).

To give clear instructions on how to complete a recipe, we use these sections as follows:

# Getting ready

This section tells you what to expect in the recipe, and describes how to set up any software or any preliminary settings required for the recipe.

# How to do it…

This section contains the steps required to follow the recipe.

# How it works…

This section usually consists of a detailed explanation of what happened in the previous section.

# There's more…

This section consists of additional information about the recipe in order to make the reader more knowledgeable about the recipe.

# See also

This section provides helpful links to other useful information for the recipe.

# Conventions

In this book, you will find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "Just assign the names of the layers you want to activate to the `VK_INSTANCE_LAYERS` environment variable"

A block of code is set as follows:

[PRE0]

Any command-line input or output is written as follows:

[PRE1]

**New terms** and **important words** are shown in bold. Words that you see on the screen, for example, in menus or dialog boxes, appear in the text like this: "Select System info from the Administration panel."

Warnings or important notes appear in a box like this.

Tips and tricks appear like this.

# Reader feedback

Feedback from our readers is always welcome. Let us know what you think about this book-what you liked or disliked. Reader feedback is important for us as it helps us develop titles that you will really get the most out of.

To send us general feedback, simply e-mail `feedback@packtpub.com`, and mention the book's title in the subject of your message.

If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, see our author guide at [www.packtpub.com/authors](http://www.packtpub.com/authors) .

# Customer support

Now that you are the proud owner of a Packt book, we have a number of things to help you to get the most from your purchase.

# Downloading the example code

You can download the example code files for this book from your account at [h t t p ://w w w . p a c k t p u b . c o m](http://www.packtpub.com). If you purchased this book elsewhere, you can visit [h t t p ://w w w . p a c k t p u b . c o m /s u p p o r t](http://www.packtpub.com/support) and register to have the files e-mailed directly to you.

You can download the code files by following these steps:

1.  Log in or register to our website using your e-mail address and password.
2.  Hover the mouse pointer on the SUPPORT tab at the top.
3.  Click on Code Downloads & Errata.
4.  Enter the name of the book in the Search box.
5.  Select the book for which you're looking to download the code files.
6.  Choose from the drop-down menu where you purchased this book from.
7.  Click on Code Download.

You can also download the code files by clicking on the Code Files button on the book's webpage at the Packt Publishing website. This page can be accessed by entering the book's name in the Search box. Please note that you need to be logged in to your Packt account.

Once the file is downloaded, please make sure that you unzip or extract the folder using the latest version of:

*   WinRAR / 7-Zip for Windows
*   Zipeg / iZip / UnRarX for Mac
*   7-Zip / PeaZip for Linux

The code bundle for the book is also hosted on GitHub at **[h t t p s ://g i t h u b . c o m /P a c k t P u b l i s h i n g /V u l k a n - C o o k b o o k](https://github.com/PacktPublishing/Vulkan-Cookbook)** . We also have other code bundles from our rich catalog of books and videos available at **[h t t p s ://g i t h u b . c o m /P a c k t P u b l i s h i n g /](https://github.com/PacktPublishing/)**. Check them out!

# Downloading the color images of this book

We also provide you with a PDF file that has color images of the screenshots/diagrams used in this book. The color images will help you better understand the changes in the output. You can download this file from [h t t p s ://w w w . p a c k t p u b . c o m /s i t e s /d e f a u l t /f i l e s /d o w n l o a d s /V u l k a n C o o k b o o k _ C o l o r I m a g e s . p d f](https://www.packtpub.com/sites/default/files/downloads/VulkanCookbook_ColorImages.pdf).

# Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books-maybe a mistake in the text or the code-we would be grateful if you could report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [h t t p ://w w w . p a c k t p u b . c o m /s u b m i t - e r r a t a](http://www.packtpub.com/submit-errata) , selecting your book, clicking on the Errata Submission Form link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded to our website or added to any list of existing errata under the Errata section of that title.

To view the previously submitted errata, go to [h t t p s ://w w w . p a c k t p u b . c o m /b o o k s /c o n t e n t /s u p p o r t](https://www.packtpub.com/books/content/support) and enter the name of the book in the search field. The required information will appear under the Errata section.

# Piracy

Piracy of copyrighted material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works in any form on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at `copyright@packtpub.com` with a link to the suspected pirated material.

We appreciate your help in protecting our authors and our ability to bring you valuable content.

# Questions

If you have a problem with any aspect of this book, you can contact us at `questions@packtpub.com`, and we will do our best to address the problem.