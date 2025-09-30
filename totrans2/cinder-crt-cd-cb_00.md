# Preface

Cinder is one of the most exciting frameworks available for creative coding. It is developed in C++ for increased performance and allows for the fast creation of visually complex and interactive applications. The big advantage of Cinder is that it can target many platforms such as Mac, Windows, and iOS with the exact same code.

*Cinder Creative Coding Cookbook* will show you how to develop interactive and visually dynamic applications using simple-to-follow recipes.

You will learn how to use multimedia content, draw generative graphics in 2D and 3D, and animate them in compelling ways.

Beginning with creating simple projects with Cinder, you will use multimedia, create animations, and interact with the user.

From animation with particles to using video, audio, and images, the reader will gain a broad knowledge of creating creative applications using Cinder.

With recipes that include drawing in 3D, image processing, and sensing and tracking in real-time from camera input, this book will teach you how to develop interactive applications that can be run on a desktop computer, mobile device, or be a part of an interactive installation.

This book will give you the necessary knowledge to start creating projects with Cinder that use animations and advanced visuals.

# What this book covers

[Chapter 1](ch01.html "Chapter 1. Getting Started"), *Getting Started*, teaches you the fundamentals of creating applications using Cinder.

[Chapter 2](ch02.html "Chapter 2. Preparing for Development"), *Preparing for Development*, introduces several simple recipes that can be very useful during the development process.

[Chapter 3](ch03.html "Chapter 3. Using Image Processing Techniques"), *Using Image Processing Techniques*, consists of examples of using image processing techniques implemented in Cinder and using third-party libraries.

[Chapter 4](ch04.html "Chapter 4. Using Multimedia Content"), *Using Multimedia Content*, teaches us how to load, manipulate, display, save, and share videos, graphics, and mesh data.

[Chapter 5](ch05.html "Chapter 5. Building Particle Systems"), *Building Particle Systems*, explains how to create and animate particles using popular and versatile physics algorithms.

[Chapter 6](ch06.html "Chapter 6. Rendering and Texturing Particle Systems"), *Rendering and Texturing Particle Systems*, teaches us how to render and apply textures to our particles in order to make them more appealing.

[Chapter 7](ch07.html "Chapter 7. Using 2D Graphics"), *Using 2D Graphics*, is about how to work and draw with 2D graphics using the OpenGL and built-in Cinder tools.

[Chapter 8](ch08.html "Chapter 8. Using 3D Graphics"), *Using 3D Graphics*, goes through the basics of creating graphics in 3D using OpenGL and some useful wrappers that Cinder includes in some advanced OpenGL features.

[Chapter 9](ch09.html "Chapter 9. Adding Animation"), *Adding Animation*, presents the techniques of animating 2D and 3D objects. We will also introduce Cinder's features in this field such as Timeline and math functions.

[Chapter 10](ch10.html "Chapter 10. Interacting with the User"), *Interacting with the User*, creates the graphical objects that react to the user using both mouse and touch interaction. It also teaches us how to create simple graphical interfaces that have their own events for greater flexibility, and integrate with the popular physics library Bullet Physics.

[Chapter 11](ch11.html "Chapter 11. Sensing and Tracking Input from the Camera"), *Sensing and Tracking Input from the Camera*, explains how to receive and process data from input devices such as a camera or a Microsoft Kinect sensor.

[Chapter 12](ch12.html "Chapter 12. Using Audio Input and Output"), *Using Audio Input and Output*, is about generating sound with the examples, where sound is generated on object's collision in physics simulation. We will present examples of visualizing sound with audio reactive animations.

*Appendix*, *Integrating with Bullet Physics,* will help us learn how to integrate Bullet Physics library with Cinder.

This chapter is available as a downloadable file at: [http://www.packtpub.com/sites/default/files/downloads/Integrating_with_Bullet_Physics.pdf](http://www.packtpub.com/sites/default/files/downloads/Integrating_with_Bullet_Physics.pdf)

# What you need for this book

Mac OS X or Windows operating system. Mac users will need XCode, which is available free from Apple and iOS SDK, if they wish to use iOS recipes. Windows users will need Visual C++ 2010\. Express Edition is available for free. Windows users will also need Windows Platform SDK installed. While writing this book the latest release of Cinder was 0.8.4.

# Who this book is for

This book is for C++ developers who want to start or already began using Cinder for building creative applications. This book is easy to follow for developers who use other creative coding frameworks and want to try Cinder.

The reader is expected to have basic knowledge of C++ programming language.

# Conventions

In this book, you will find a number of styles of text that distinguish between different kinds of information. Here are some examples of these styles, and an explanation of their meaning.

Code words in text are shown as follows: "We can include other contexts through the use of the `include` directive."

A block of code is set as follows:

[PRE0]

Any command-line input or output is written as follows:

[PRE1]

**New terms** and **important words** are shown in bold. Words that you see on the screen, in menus or dialog boxes for example, appear in the text like this: "clicking the **Next** button moves you to the next screen".

### Note

Warnings or important notes appear in a box like this.

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

## Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books—maybe a mistake in the text or the code—we would be grateful if you would report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the **errata** **submission** **form** link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded on our website, or added to any list of existing errata, under the Errata section of that title. Any existing errata can be viewed by selecting your title from [http://www.packtpub.com/support](http://www.packtpub.com/support).

## Piracy

Piracy of copyright material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works, in any form, on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at `<[copyright@packtpub.com](mailto:copyright@packtpub.com)>` with a link to the suspected pirated material.

We appreciate your help in protecting our authors, and our ability to bring you valuable content.

## Questions

You can contact us at `<[questions@packtpub.com](mailto:questions@packtpub.com)>` if you are having a problem with any aspect of the book, and we will do our best to address it.