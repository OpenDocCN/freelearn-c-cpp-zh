# Preface

OpenGL is a multiplatform, cross-language, and hardware-accelerated application programming interface for the high-performance rendering of 2D and 3D graphics. An emerging use of OpenGL is the development of real-time, high-performance data visualization applications in fields ranging from medical imaging, simulation or modeling in architecture and engineering, to cutting-edge mobile/wearable computing. Indeed, data visualization has become increasingly challenging using conventional approaches without graphics hardware acceleration as datasets become larger and more complex, especially with the evolution of big data. From a mobile device to a sophisticated high-performance computing cluster, the OpenGL libraries provide developers with an easy-to-use interface to create stunning visuals in 3D in real time for a wide range of interactive applications.

This book contains a series of hands-on recipes that are tailored to both beginners who have very little experience with OpenGL and more advanced users who would like to explore state-of-the-art techniques. We begin with a basic introduction to OpenGL in chapters 1 to 3 by demonstrating how to set up the environment in Windows, Mac OS X, and Linux and learning how to render basic 2D datasets with primitives, as well as more complex 3D volumetric datasets interactively. This part requires only OpenGL 2.0 or higher so that even readers with older graphics hardware can experiment with the code. In chapters 4 to 6, we transition to more advanced techniques (which requires OpenGL 3.2 or higher), such as texture mapping for image/video processing, point cloud rendering of depth sensor data from 3D range-sensing cameras, and stereoscopic 3D rendering. Finally, in chapters 7 to 9, we conclude this book by introducing the use of OpenGL ES 3.0 on the increasingly powerful mobile (Android-based) computing platform and the development of highly interactive, augmented reality applications on mobile devices.

Each recipe in this book gives readers a set of standard functions that can be imported to an existing project and can form the basis for the creation of a diverse array of real-time, interactive data visualization applications. This book also utilizes a set of popular open-source libraries, such as GLFW, GLM, Assimp, and OpenCV, to simplify application development and extend the capabilities of OpenGL by enabling OpenGL context management and 3D model loading, as well as image/video processing using state-of-the-art computer vision algorithms.

# What this book covers

[Chapter 1](ch01.html "Chapter 1. Getting Started with OpenGL"), *Getting Started with OpenGL*, introduces the essential development tools required to create OpenGL-based data visualization applications and provides a step-by-step tutorial on how to set up the environment for our first OpenGL demo application in Windows, Mac OS X, and Linux.

[Chapter 2](ch02.html "Chapter 2. OpenGL Primitives and 2D Data Visualization"), *OpenGL Primitives and 2D Data Visualization*, focuses on the use of OpenGL 2.0 primitives, such as points, lines, and triangles, to enable the basic 2D visualization of data, including time series such as an electrocardiogram (ECG).

[Chapter 3](ch03.html "Chapter 3. Interactive 3D Data Visualization"), *Interactive 3D Data Visualization*, builds upon the fundamental concepts discussed previously and extends the demos to incorporate more sophisticated OpenGL features for 3D rendering.

[Chapter 4](ch04.html "Chapter 4. Rendering 2D Images and Videos with Texture Mapping"), *Rendering 2D Images and Videos with Texture Mapping*, introduces OpenGL techniques to visualize another important class of datasets—those involving images or videos. Such datasets are commonly encountered in many fields, including medical imaging applications.

[Chapter 5](ch05.html "Chapter 5. Rendering of Point Cloud Data for 3D Range-sensing Cameras"), *Rendering of Point Cloud Data for 3D Range-sensing Cameras*, introduces the techniques used to visualize another interesting and emerging class of data—depth information from 3D range sensing cameras.

[Chapter 6](ch06.html "Chapter 6. Rendering Stereoscopic 3D Models using OpenGL"), *Rendering Stereoscopic 3D Models using OpenGL*, demonstrates how to visualize data with stunning stereoscopic 3D technology using OpenGL. OpenGL does not provide any mechanism to load, save, or manipulate 3D models. Thus, to support this, we will integrate a new library named Assimp into our code.

[Chapter 7](ch07.html "Chapter 7. An Introduction to Real-time Graphics Rendering on a Mobile Platform using OpenGL ES 3.0"), *An Introduction to Real-time Graphics Rendering on a Mobile Platform using OpenGL ES 3.0*, transitions to an increasingly powerful and ubiquitous computing platform by demonstrating how to set up the Android development environment and create the first Android-based application on the latest mobile devices, from smartphones to tablets, using OpenGL for Embedded Systems (OpenGL ES).

[Chapter 8](ch08.html "Chapter 8. Interactive Real-time Data Visualization on Mobile Devices"), *Interactive Real-time Data Visualization on Mobile Devices*, demonstrates how to visualize data interactively by using built-in motion sensors called Inertial Measurement Units (IMUs) and the multitouch interface found on mobile devices.

[Chapter 9](ch09.html "Chapter 9. Augmented Reality-based Visualization on Mobile or Wearable Platforms"), *Augmented Reality-based Visualization on Mobile or Wearable Platforms*, introduces the fundamental building blocks required to create your first AR-based application on a commodity Android-based mobile device: OpenCV for computer vision, OpenGL for graphics rendering, as well as Android's sensor framework for interaction.

# What you need for this book

This book supports a wide range of platforms and open source libraries, ranging from Windows, Mac OS X, or Linux-based desktop applications to portable Android-based mobile applications. You will need a basic understanding of C/C++ programming and background in basic linear algebra for geometric models.

The following are the requirements for chapters 1 to 3:

*   **OpenGL version**: 2.0 or higher (easy to test on legacy graphics hardware).
*   **Platforms**: Windows, Mac OS X, or Linux.
*   **Libraries**: GLFW for OpenGL Windows/context management and handling user inputs. No additional libraries are needed, which makes it very easy to integrate into existing projects.
*   **Development tools**: Windows Visual Studio or Xcode, CMake, and gcc.

The following are the requirements for chapters 4 to 6:

*   **OpenGL version**: 3.2 or higher.
*   **Platforms**: Windows, Mac OS X, or Linux.
*   **Libraries**: Assimp for 3D model loading, SOIL for image and texture loading, GLEW for runtime OpenGL extension support, GLM for matrix operations, and OpenCV for image processing
*   **Development tools**: Windows Visual Studio or Xcode, CMake, and gcc.

The following are the requirements for chapters 7 to 9:

*   **OpenGL version**: OpenGL ES 3.0
*   **Platforms**: Linux or Mac OS X for development, and Android OS 4.3 and higher (API 18 and higher) for deployment
*   **Libraries**: OpenCV for Android and GLM
*   **Development tools**: Android SDK, Android NDK, and Apache Ant in Mac OS X or Linux

For more information, keep in mind that the code in this book was built and tested with the following libraries and development tools in all supported platforms:

*   OpenCV 2.4.9 ([http://opencv.org/downloads.html](http://opencv.org/downloads.html))
*   OpenCV 3.0.0 for Android ([http://opencv.org/downloads.html](http://opencv.org/downloads.html))
*   SOIL ([http://www.lonesock.net/soil.html](http://www.lonesock.net/soil.html))
*   GLEW 1.12.0 ([http://glew.sourceforge.net/](http://glew.sourceforge.net/))
*   GLFW 3.0.4 ([http://www.glfw.org/download.html](http://www.glfw.org/download.html))
*   GLM 0.9.5.4 ([http://glm.g-truc.net/0.9.5/index.html](http://glm.g-truc.net/0.9.5/index.html))
*   Assimp 3.0 ([http://assimp.sourceforge.net/main_downloads.html](http://assimp.sourceforge.net/main_downloads.html))
*   Android SDK r24.3.3 ([https://developer.android.com/sdk/index.html](https://developer.android.com/sdk/index.html))
*   Android NDK r10e ([https://developer.android.com/ndk/downloads/index.html](https://developer.android.com/ndk/downloads/index.html))
*   Windows Visual Studio 2013 ([https://www.visualstudio.com/en-us/downloads/download-visual-studio-vs.aspx](https://www.visualstudio.com/en-us/downloads/download-visual-studio-vs.aspx))
*   CMake 3.2.1 ([http://www.cmake.org/download/](http://www.cmake.org/download/))

# Who this book is for

This book is aimed at anyone interested in creating impressive data visualization tools using modern graphics hardware. Whether you are a developer, engineer, or scientist, if you are interested in exploring the power of OpenGL for data visualization, this book is for you. While familiarity with C/C++ is recommended, no previous experience with OpenGL is assumed.

# Sections

In this book, you will find several headings that appear frequently (Getting ready, How to do it, How it works, There's more, and See also).

To give clear instructions on how to complete a recipe, we use these sections as follows:

## Getting ready

This section tells you what to expect in the recipe, and describes how to set up any software or any preliminary settings required for the recipe.

## How to do it...

This section contains the steps required to follow the recipe.

## How it works...

This section usually consists of a detailed explanation of what happened in the previous section.

## There's more...

This section consists of additional information about the recipe in order to make the reader more knowledgeable about the recipe.

## See also

This section provides helpful links to other useful information for the recipe.

# Conventions

In this book, you will find a number of text styles that distinguish between different kinds of information. Here are some examples of these styles and an explanation of their meaning.

Code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles are shown as follows: "We assume that all files are saved to a top-level directory called `code` and the `main.cpp` file is saved inside the `/code/Tutorial1` subdirectory."

A block of code is set as follows:

[PRE0]

Any command-line input or output is written as follows:

[PRE1]

**New terms** and **important words** are shown in bold. Words that you see on the screen, for example, in menus or dialog boxes, appear in the text like this: "Check the **Empty project** option, and click on **Finish**."

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

We also provide you with a PDF file that has color images of the screenshots/diagrams used in this book. The color images will help you better understand the changes in the output. You can download this file from: [https://www.packtpub.com/sites/default/files/downloads/9727OS.pdf](https://www.packtpub.com/sites/default/files/downloads/9727OS.pdf).

## Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books—maybe a mistake in the text or the code—we would be grateful if you could report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the **Errata** **Submission** **Form** link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded to our website or added to any list of existing errata under the Errata section of that title.

To view the previously submitted errata, go to [https://www.packtpub.com/books/content/support](https://www.packtpub.com/books/content/support) and enter the name of the book in the search field. The required information will appear under the **Errata** section.

## Piracy

Piracy of copyrighted material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works in any form on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at `<[copyright@packtpub.com](mailto:copyright@packtpub.com)>` with a link to the suspected pirated material.

We appreciate your help in protecting our authors and our ability to bring you valuable content.

## Questions

If you have a problem with any aspect of this book, you can contact us at `<[questions@packtpub.com](mailto:questions@packtpub.com)>`, and we will do our best to address the problem.