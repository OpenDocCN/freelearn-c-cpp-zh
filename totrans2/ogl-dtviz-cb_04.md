# Chapter 4. Rendering 2D Images and Videos with Texture Mapping

In this chapter, we will cover the following topics:

*   Getting started with modern OpenGL (3.2 or higher)
*   Setting up the GLEW, GLM, SOIL, and OpenCV libraries in Windows
*   Setting up the GLEW, GLM, SOIL, and OpenCV libraries in Mac OS X/Linux
*   Creating your first vertex and fragment shader using GLSL
*   Rendering 2D images with texture mapping
*   Real-time video rendering with filters

# Introduction

In this chapter, we will introduce OpenGL techniques to visualize another important class of datasets: those involving images or videos. Such datasets are commonly encountered in many fields, including medical imaging applications. To enable the rendering of images, we will discuss fundamental OpenGL concepts for texture mapping and transition to more advanced techniques that require newer versions of OpenGL (OpenGL 3.2 or higher). To simplify our tasks, we will also employ several additional libraries, including **OpenGL Extension Wrangler Library** (**GLEW**) for runtime OpenGL extension support, **Simple OpenGL Image Loader** (**SOIL**) to load different image formats, **OpenGL Mathematics** (**GLM**) for vector and matrix manipulation, as well as **OpenCV** for image/video processing. To get started, we will first introduce the features of modern OpenGL 3.2 and higher.

# Getting started with modern OpenGL (3.2 or higher)

Continuous evolution of OpenGL APIs has led to the emergence of a modern standard. One of the biggest changes happened in 2008 with OpenGL version 3.0, in which a new context creation mechanism was introduced and most of the older functions, such as Begin/End primitive specifications, were marked as deprecated. The removal of these older standard features also implies a more flexible yet more powerful way of handling the graphics pipeline. In OpenGL 3.2 or higher, a core and a compatible profile were defined to differentiate the deprecated APIs from the current features. These profiles provide clear definitions for various features (core profile) while enabling backward compatibility (compatibility profile). In OpenGL 4.x, support for the latest graphics hardware that runs Direct3D 11 is provided, and a detailed comparison between OpenGL 3.x and OpenGL 4.x is available at [http://www.g-truc.net/post-0269.html](http://www.g-truc.net/post-0269.html).

## Getting ready

Starting from this chapter, we need compatible graphics cards with OpenGL 3.2 (or higher) support. Most graphics cards released before 2008 will most likely not be supported. For example, NVIDIA GeForce 100, 200, 300 series and higher support the OpenGL 3 standard. You are encouraged to consult the technical specifications of your graphics cards to confirm the compatibility (refer to [https://developer.nvidia.com/opengl-driver](https://developer.nvidia.com/opengl-driver)).

## How to do it...

To enable OpenGL 3.2 support, we need to incorporate the following lines of code at the beginning of every program for initialization:

[PRE0]

## How it works...

The `glfwWindowHint` function defines a set of constraints for the creation of the GLFW windows context (refer to [Chapter 1](ch01.html "Chapter 1. Getting Started with OpenGL"), *Getting Started with OpenGL*). The first two lines of code here define the current version of OpenGL that will be used (3.2 in this case). The third line enables forward compatibility, while the last line specifies that the core profile will be used.

## See also

Detailed explanation of the differences between various OpenGL versions can be found at [http://www.opengl.org/wiki/History_of_OpenGL](http://www.opengl.org/wiki/History_of_OpenGL).

# Setting up the GLEW, GLM, SOIL, and OpenCV libraries in Windows

In this section, we will provide step-by-step instructions to set up several popular libraries that will be used extensively in this chapter (and in subsequent chapters), including the GLEW, GLM, SOIL, and OpenCV libraries:

*   The GLEW library is an open-source OpenGL extension library.
*   The GLM library is a header-only C++ library that provides an easy-to-use set of common mathematical operations. It is built on the GLSL specifications and as it is a header-only library, it does not require tedious compilation steps.
*   The SOIL library is a simple C library that is used to load images in a variety of common formats (such as BMP, PNG, JPG, TGA, TIFF, and HDR) in OpenGL textures.
*   The OpenCV library is a very powerful open source computer vision library that we will use to simplify image and video processing in this chapter.

## Getting ready

We will first need to download the prerequisite libraries from the following websites:

*   **GLEW** (glew-1.10.0): [http://sourceforge.net/projects/glew/files/glew/1.10.0/glew-1.10.0-win32.zip](http://sourceforge.net/projects/glew/files/glew/1.10.0/glew-1.10.0-win32.zip)
*   **GLM** (glm-0.9.5.4): [http://sourceforge.net/projects/ogl-math/files/glm-0.9.5.4/glm-0.9.5.4.zip](http://sourceforge.net/projects/ogl-math/files/glm-0.9.5.4/glm-0.9.5.4.zip)
*   **SOIL**: [http://www.lonesock.net/files/soil.zip](http://www.lonesock.net/files/soil.zip)
*   **OpenCV** (opencv-2.4.9): [http://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.9/opencv-2.4.9.exe](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.9/opencv-2.4.9.exe)

## How to do it...

To use the precompiled package from GLEW, follow these steps:

1.  Unzip the package.
2.  Copy the directory to `C:/Program Files (x86)`.
3.  Ensure that the `glew32.dll` file (`C:\Program Files (x86)\glew-1.10.0\bin\Release\Win32`) can be found at runtime by placing it either in the same folder as the executable or including the directory in the Windows system `PATH` environment variable (Navigate to **Control Panel** | **System and Security** | **System** | **Advanced Systems Settings** | **Environment Variables**).

To use the header-only GLM library, follow these steps:

1.  Unzip the package.
2.  Copy the directory to `C:/Program Files (x86)`.
3.  Include the desired header files in your source code. Here is an example:

    [PRE1]

To use the SOIL library, follow these steps:

1.  Unzip the package.
2.  Copy the directory to `C:/Program Files (x86)`.
3.  Generate the `SOIL.lib` file by opening the Visual Studio solution file (`C:\Program Files (x86)\Simple OpenGL Image Library\projects\VC9\SOIL.sln`) and compiling the project files. Copy this file from `C:\Program Files (x86)\Simple OpenGL Image Library\projects\VC9\Debug to C:\Program Files (x86)\Simple OpenGL Image Library\lib`.
4.  Include the header file in your source code:

    [PRE2]

Finally, to install OpenCV, we recommend that you use prebuilt binaries to simplify the process:

1.  Download the prebuilt binaries from [http://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.9/opencv-2.4.9.exe](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.9/opencv-2.4.9.exe) and extract the package.
2.  Copy the directory (the `opencv` folder) to `C:\Program Files (x86)`.
3.  Add this to the system `PATH` environment variable (Navigate to **Control Panel** | **System and Security** | **System** | **Advanced Systems Settings** | **Environment Variables**) – `C:\Program Files (x86)\opencv\build\x86\vc12\bin`.
4.  Include the desired header files in your source code:

    [PRE3]

Now, we generate our Microsoft Visual Studio Solution files (the build environment) using `CMake`. We create the `CMakeList.txt` file within each project directory, which lists all the libraries and dependencies for the project. The following is a sample `CMakeList.txt` file for our first demo application:

[PRE4]

As you can see in the `CMakeList.txt` file, the various dependencies, including the OpenCV, SOIL, GLFW, and GLEW libraries, are all included.

Finally, we run the `CMake` program to generate the Microsoft Visual Studio Solution for the project (refer to [Chapter 1](ch01.html "Chapter 1. Getting Started with OpenGL"), *Getting Started with OpenGL* for details). Note that the output path for the binary must match the project folder due to dependencies of the shader programs. The following is a screenshot of the `CMake` window after generating the first sample project called `code_simple`:

![How to do it...](img/9727OS_04_01.jpg)

We will repeat this step for each project we create, and the corresponding Microsoft Visual Studio Solution file will be generated accordingly (for example, `code_simple.sln` in this case). To compile the code, open `code_simple.sln` with Microsoft Visual Studio 2013 and build the project using the Build (press *F7*) function as usual. Make sure that you set main as the start up project (by right-clicking on the *main* project in the **Solution Explorer** and left-clicking on the **Set as StartUp Project** option) before running the program, as shown follows:

![How to do it...](img/9727OS_04_03.jpg)

## See also

Further documentation on each of the libraries installed can be found here:

*   **GLEW**: [http://glew.sourceforge.net/](http://glew.sourceforge.net/)
*   **GLM**: [http://glm.g-truc.net/0.9.5/index.html](http://glm.g-truc.net/0.9.5/index.html)
*   **SOIL**: [http://www.lonesock.net/soil.html](http://www.lonesock.net/soil.html)
*   **OpenCV**: [http://opencv.org/](http://opencv.org/)

# Setting up the GLEW, GLM, SOIL, and OpenCV libraries in Mac OS X/Linux

In this section, we will outline the steps required to set up the same libraries in Mac OS X and Linux.

## Getting ready

We will first need to download the prerequisite libraries from the following websites:

1.  **GLEW** (glew-1.10.0): [https://sourceforge.net/projects/glew/files/glew/1.10.0/glew-1.10.0.tgz](https://sourceforge.net/projects/glew/files/glew/1.10.0/glew-1.10.0.tgz)
2.  **GLM** (glm-0.9.5.4): [http://sourceforge.net/projects/ogl-math/files/glm-0.9.5.4/glm-0.9.5.4.zip](http://sourceforge.net/projects/ogl-math/files/glm-0.9.5.4/glm-0.9.5.4.zip)
3.  **SOIL**: [http://www.lonesock.net/files/soil.zip](http://www.lonesock.net/files/soil.zip)
4.  **OpenCV** (opencv-2.4.9): [http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.9/opencv-2.4.9.zip](http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.9/opencv-2.4.9.zip)

To simplify the installation process for Mac OS X or Ubuntu users, the use of MacPorts in Mac OS X or the `apt-get` command in Linux (as described in [Chapter 1](ch01.html "Chapter 1. Getting Started with OpenGL"), *Getting Started with OpenGL*) is highly recommended.

The following section assumes that the download directory is `~/opengl_dev` (refer to [Chapter 1](ch01.html "Chapter 1. Getting Started with OpenGL"), *Getting Started with OpenGL*).

## How to do it...

There are two methods to install the prerequisite libraries. The first method uses precompiled binaries. These binary files are fetched from remote repository servers and the version updates of the library are controlled externally. An important advantage of this method is that it simplifies the installation, especially in terms of resolving dependencies. However, in a release environment, it is recommended that you disable the automatic updates and thus protect the binary from version skewing. The second method requires users to download and compile the source code directly with various customizations. This method is recommended for users who would like to control the installation process (such as the paths), and it also provides more flexibility in terms of tracking and fixing bugs.

For beginners or developers who are looking for rapid prototyping, we recommend that you use the first method as it will simplify the workflow and have short-term maintenance. On an Ubuntu or Debian system, we can install the various libraries using the `apt-get` command. To install all the prerequisite libraries and dependencies on Ubuntu, simply run the following commands in the terminal:

[PRE5]

Similarly, on Mac OS X, we can install GLEW, OpenCV, and GLM with MacPorts through command lines in the terminal:

[PRE6]

However, the SOIL library is not currently supported by MacPorts, and thus, the installation has to be completed manually, as described in the following section.

For advanced users, we can install the latest packages by directly compiling from the source, and the upcoming steps are common among Mac OS as well as other Linux OS.

To compile the GLEW package, follow these steps:

1.  Extract the `glew-1.10.0.tgz` package:

    [PRE7]

2.  Install GLEW in `/usr/include/GL` and `/usr/lib`:

    [PRE8]

To set up the header-only GLM library, follow these steps:

1.  Extract the unzip `glm-0.9.5.4.zip` package:

    [PRE9]

2.  Copy the header-only GLM library directory (`~/opengl_dev/glm/glm`) to `/usr/include/glm`:

    [PRE10]

To set up the SOIL library, follow these steps:

1.  Extract the unzip `soil.zip` package:

    [PRE11]

2.  Edit `makefile` (inside the `projects/makefile` directory) and add `-arch x86_64` and `-arch i386` to `CXXFLAGS` to ensure proper support:

    [PRE12]

3.  Compile the source code library:

    [PRE13]

To set up the OpenCV library, follow these steps:

1.  Extract the `opencv-2.4.9.zip` package:

    [PRE14]

2.  Build the OpenCV library using `CMake`:

    [PRE15]

3.  Configure the library path:

    [PRE16]

4.  With the development environment fully configured, we can now create the compilation script (`Makefile`) within each project folder:

    [PRE17]

To compile the code, we simply run the `make` command in the project directory and it generates the executable (`main`) automatically.

## See also

Further documentation on each of the libraries installed can be found here:

*   **GLEW**: [http://glew.sourceforge.net/](http://glew.sourceforge.net/)
*   **GLM**: [http://glm.g-truc.net/0.9.5/index.html](http://glm.g-truc.net/0.9.5/index.html)
*   **SOIL**: [http://www.lonesock.net/soil.html](http://www.lonesock.net/soil.html)
*   **OpenCV**: [http://opencv.org/](http://opencv.org/)
*   **MacPorts**: [http://www.macports.org/](http://www.macports.org/)

# Creating your first vertex and fragment shader using GLSL

Before we can render images using OpenGL, we need to first understand the basics of the GLSL. In particular, the concept of shader programs is essential in GLSL. Shaders are simply programs that run on graphics processors (GPUs), and a set of shaders is compiled and linked to form a program. This concept emerges as a result of the increasing complexity of various common processing tasks in modern graphics hardware, such as vertex and fragment processing, which necessitates greater programmability of specialized processors. Accordingly, the vertex and fragment shader are two important types of shaders that we will cover here, and they run on the vertex processor and fragment processor, respectively. A simplified diagram illustrating the overall processing pipeline is shown as follows:

![Creating your first vertex and fragment shader using GLSL](img/9727OS_04_02.jpg)

The main purpose of the vertex shader is to perform the processing of a stream of vertex data. An important processing task involves the transformation of the position of each vertex from the 3D virtual space to a 2D coordinate for display on the screen. Vertex shaders can also manipulate the color and texture coordinates. Therefore, vertex shaders serve as an important component of the OpenGL pipeline to control movement, lighting, and color.

A fragment shader is primarily designed to compute the final color of an individual pixel (fragment). Oftentimes, we implement various image post-processing techniques, such as blurring or sharpening, at this stage; the end results are stored in the framebuffer, which will be displayed on screen.

For readers interested in understanding the rest of the pipeline, a detailed summary of these stages, such as the clipping, rasterization, and tessellation, can be found at [https://www.opengl.org/wiki/Rendering_Pipeline_Overview](https://www.opengl.org/wiki/Rendering_Pipeline_Overview). Additionally, a detailed documentation of GLSL can be found at [https://www.opengl.org/registry/doc/GLSLangSpec.4.40.pdf](https://www.opengl.org/registry/doc/GLSLangSpec.4.40.pdf).

## Getting ready

At this point, we should have all the prerequisite libraries, such as GLEW, GLM, and SOIL. With GLFW configured for the OpenGL core profile, we are now ready to implement the first simple example code, which takes advantage of the modern OpenGL pipeline.

## How to do it...

To keep the code simple, we will divide the program into two components: the main program (`main.cpp`) and shader programs (`shader.cpp`, `shader.hpp`, `simple.vert`, and `simple.frag`). The main program performs the essential tasks to set up the simple demo, while the shader programs perform the specialized processing in the modern OpenGL pipeline. The complete sample code can be found in the `code_simple` folder.

First, let's take a look at the shader programs. We will create two extremely simple vertex and fragment shader programs (specified inside the `simple.vert` and `simple.frag` files) that are compiled and loaded by the program at runtime.

For the `simple.vert` file, enter the following lines of code:

[PRE18]

For the `simple.frag` file, enter the following lines of code:

[PRE19]

First, let's define a function to compile and load the shader programs (`simple.frag` and `simple.vert`) called `LoadShaders` inside `shader.hpp`:

[PRE20]

Next, we will create the `shader.cpp` file to implement the `LoadShaders` function and two helper functions to handle file I/O (`readSourceFile`) and the compilation of the shaders (`CompileShader`):

1.  Include prerequisite libraries and the `shader.hpp` header file:

    [PRE21]

2.  Implement the `readSourceFile` function as follows:

    [PRE22]

3.  Implement the `CompileShader` function as follows:

    [PRE23]

4.  Now, let's implement the `LoadShaders` function. First, create the shader ID and read the shader code from two files specified by `vertex_file_path` and `fragment_file_path`:

    [PRE24]

5.  Compile the vertex shader and fragment shader programs:

    [PRE25]

6.  Link the programs together, check for errors, and clean up:

    [PRE26]

Finally, let's put everything together with the `main.cpp` file:

1.  Include prerequisite libraries and the shader program header file inside the common folder:

    [PRE27]

2.  Create a global variable for the GLFW window:

    [PRE28]

3.  Start the main program with the initialization of the GLFW library:

    [PRE29]

4.  Set up the GLFW window:

    [PRE30]

5.  Create the GLFW window object and make the context of the specified window current on the calling thread:

    [PRE31]

6.  Initialize the GLEW library and include support for experimental drivers:

    [PRE32]

7.  Set up the shader programs:

    [PRE33]

8.  Set up Vertex Buffer Object (and color buffer) and copy the vertex data to it:

    [PRE34]

9.  Specify the layout of the vertex data:

    [PRE35]

10.  Run the drawing program:

    [PRE36]

11.  Clean up and exit the program:

    [PRE37]

Now we have created the first GLSL program by defining custom shaders:

![How to do it...](img/9727OS_04_04.jpg)

## How it works...

As there are multiple components in this implementation, we will highlight the key features inside each component separately, organized in the same order as the previous section using the same file name for simplicity.

Inside `simple.vert`, we defined a simple vertex shader. In the first simple implementation, the vertex shader simply passes information forward to the rest of the rendering pipeline. First, we need to define the GLSL version that corresponds to the OpenGL 3.2 support, which is 1.50 (`#version 150`). The vertex shader takes two parameters: the position of the vertex (`in vec3 position`) and the color (`in vec3 color_in`). Note that only the color is defined explicitly in an output variable (`out vec3 color`) as `gl_Position` is a built-in variable. In general, variable names with the prefix `gl` should not be used inside shader programs in OpenGL as these are reserved for built-in variables. Notice that the final position, `gl_Position`, is expressed in homogeneous coordinates.

Inside `simple.frag`, we defined the fragment shader, which again passes the color information forward to the output framebuffer. Notice that the final output (`color_out`) is expressed in the RGBA format, where A is the alpha value (transparency).

Next, in `shader.cpp`, we created a framework to compile and link shader programs. The workflow shares some similarity with conventional code compilation in C/C++. Briefly, there are six major steps:

1.  Create a shader object (`glCreateShader`).
2.  Read and set the shader source code (`glShaderSource`).
3.  Compile (`glCompileShader`).
4.  Create the final program ID (`glCreateProgram`).
5.  Attach a shader to the program ID (`glAttachShader`).
6.  Link everything together (`glLinkProgram`).

Finally, in `main.cpp`, we set up a demo to illustrate the use of the compiled shader program. As described in the *Getting Started with Modern OpenGL* section of this chapter, we need to use the `glfwWindowHint` function to properly create the GLFW window context in OpenGL 3.2\. An interesting aspect to point out about this demo is that even though we defined only six vertices (three vertices for each of the two triangles drawn using the `glDrawArrays` function) and their corresponding colors, the final result is an interpolated color gradient.

# Rendering 2D images with texture mapping

Now that we have introduced the basics of GLSL using a simple example, we will incorporate further complexity to provide a complete framework that enables users to modify any part of the rendering pipeline in the future.

The code in this framework is divided into smaller modules to handle the shader programs (`shader.cpp` and `shader.hpp`), texture mapping (`texture.cpp` and `texture.hpp`), and user inputs (`controls.hpp` and `controls.hpp`). First, we will reuse the mechanism to load shader programs in OpenGL introduced previously and incorporate new shader programs for our purpose. Next, we will introduce the steps required for texture mapping. Finally, we will describe the main program, which integrates all the logical pieces and prepares the final demo. In this section, we will show how we can load an image and convert it into a texture object to be rendered in OpenGL. With this framework in mind, we will further demonstrate how to render a video in the next section.

## Getting ready

To avoid redundancy here, we will refer readers to the previous section for part of this demo (in particular, `shader.cpp` and `shader.hpp`).

## How to do it...

First, we aggregate all the common libraries used in our program into the `common.h` header file. The `common.h` file is then included in `shader.hpp`, `controls.hpp`, `texture.hpp`, and `main.cpp`:

[PRE38]

We previously implemented a mechanism to load a fragment and vertex shader program from files, and we will reuse the code here (`shader.cpp` and `shader.hpp`). However, we will modify the actual vertex and shader programs as follows.

For the vertex shader (`transform.vert`), we will implement the following:

[PRE39]

For the fragment shader (`texture.frag`), we will implement the following:

[PRE40]

For the texture objects, in `texture.cpp`, we provide a mechanism to load images or video stream into the texture memory. We also take advantage of the SOIL library for simple image loading and the OpenCV library for more advanced video stream handling and filtering (refer to the next section).

In `texture.cpp`, we will implement the following:

1.  Include the `texture.hpp` header and SOIL library header for simple image loading:

    [PRE41]

2.  Define the initialization of texture objects and set up all parameters:

    [PRE42]

3.  Define the routine to update the texture memory:

    [PRE43]

4.  Finally, implement the texture-loading mechanism for images. The function takes the image path and automatically converts the image into various compatible formats for the texture object:

    [PRE44]

On the controller front, we capture the arrow keys and modify the camera model parameter in real time. This allows us to change the position and orientation of the camera as well as the angle of view. In `controls.cpp`, we implement the following:

1.  Include the GLM library header and the `controls.hpp` header for the projection matrix and view matrix computations:

    [PRE45]

2.  Define global variables (camera parameters as well as view and projection matrices) to be updated after each frame:

    [PRE46]

3.  Create helper functions to return the most updated view matrix and projection matrix:

    [PRE47]

4.  Compute the view matrix and projection matrix based on the user input:

    [PRE48]

In `main.cpp`, we will use the various previously defined functions to complete the implementation:

1.  Include the GLFW and GLM libraries as well as our helper functions, which are stored in separate files inside a folder called the `common` folder:

    [PRE49]

2.  Define all global variables for the setup:

    [PRE50]

3.  Define the keyboard `callback` function:

    [PRE51]

4.  Initialize the GLFW library with the OpenGL core profile enabled:

    [PRE52]

5.  Set up the GLFW windows and keyboard input handlers:

    [PRE53]

6.  Set a black background and enable alpha blending for various visual effects:

    [PRE54]

7.  Load the vertex shader and fragment shader:

    [PRE55]

8.  Load an image file into the texture object using the SOIL library:

    [PRE56]

9.  Get the locations of the specific variables in the shader programs:

    [PRE57]

10.  Define our **Vertex Array Objects** (**VAO**):

    [PRE58]

11.  Define our VAO for vertices and UV mapping:

    [PRE59]

12.  Use the shader program and bind all texture units and attribute buffers:

    [PRE60]

13.  In the main loop, clear the screen and depth buffers:

    [PRE61]

14.  Compute the transforms and store the information in the shader variables:

    [PRE62]

15.  Draw the elements and flush the screen:

    [PRE63]

16.  Finally, define the conditions to exit the `main` loop and clear all the memory to exit the program gracefully:

    [PRE64]

## How it works...

To demonstrate the use of the framework for data visualization, we will apply it to the visualization of a histology slide (an H&E cross-section of a skin sample), as shown in the following screenshot:

![How it works...](img/9727OS_04_05.jpg)

An important difference between this demo and the previous one is that here, we actually load an image into the texture memory (`texture.cpp`). To facilitate this task, we use the SOIL library call (`SOIL_load_image`) to load the histology image in the RGBA format (`GL_RGBA`) and the `glTexImage2D` function call to generate a texture image that can be read by shaders.

Another important difference is that we can now dynamically recompute the view (`g_view_matrix`) and projection (`g_projection_matrix`) matrices to enable an interactive and interesting visualization of an image in the 3D space. Note that the GLM library header is included to facilitate the matrix computations. Using the keyboard inputs (up, down, left, and right) defined in `controls.cpp` with the GLFW library calls, we can zoom in and out of the slide as well as adjust the view angle, which gives an interesting perspective of the histology image in the 3D virtual space. Here is a screenshot of the image viewed with a different perspective:

![How it works...](img/9727OS_04_06.jpg)

Yet another unique feature of the current OpenGL-based framework is illustrated by the following screenshot, which is generated with a new image filter implemented into the fragment shader that highlights the edges in the image. This shows the endless possibilities for the real-time interactive visualization and processing of 2D images using OpenGL rendering pipeline without compromising on CPU performance. The filter implemented here will be discussed in the next section.

![How it works...](img/9727OS_04_07.jpg)

# Real-time video rendering with filters

The GLSL shader provides a simple way to perform highly parallelized processing. On top of the texture mapping shown previously, we will demonstrate how to implement a simple video filter that postprocesses the end results of the buffer frame using the fragment shader. To illustrate this technique, we implement the Sobel Filter along with a heat map rendered using the OpenGL pipeline. The heat map function that was previously implemented in [Chapter 3](ch03.html "Chapter 3. Interactive 3D Data Visualization"), *Interactive 3D Data Visualization*, will now be directly ported to GLSL with very minor changes.

The Sobel operator is a simple image processing technique frequently used in computer vision algorithms such as edge detection. This operator can be defined as a convolution operation with a 3 x 3 kernel, shown as follows:

![Real-time video rendering with filters](img/9727OS_04_14.jpg)

![Real-time video rendering with filters](img/9727OS_04_16.jpg) and ![Real-time video rendering with filters](img/9727OS_04_17.jpg) are results of the horizontal and vertical derivatives of an image, respectively, from the convolution operation of image *I* at the pixel location *(x, y)*.

We can also perform a sum of squares operation to approximate the gradient magnitude of the image:

![Real-time video rendering with filters](img/9727OS_04_18.jpg)

## Getting ready

This demo builds on top of the previous section, where an image was rendered. In this section, we will demonstrate the rendering of an image sequence or a video with the use of OpenCV library calls to handle videos. Inside `common.h`, we will add the following lines to include the OpenCV libraries:

[PRE65]

## How to do it...

Now, let's complete the implementation as follows:

1.  First, modify `main.cpp` to enable video processing using OpenCV. Essentially, instead of loading an image, feed the individual frames of a video into the same pipeline:

    [PRE66]

2.  Then, add the `update` function in the `main` loop to update the texture in every frame:

    [PRE67]

3.  Next, modify the fragment shader and rename it `texture_sobel.frag` (from `texture.frag`). In the `main` function, we will outline the overall processing (process the texture buffers with the Sobel filter and heat map renderer):

    [PRE68]

4.  Now, implement the Sobel filter algorithm that takes the neighboring pixels to compute the result:

    [PRE69]

5.  Define the helper function that computes the brightness value:

    [PRE70]

6.  Create a helper function for the per-pixel operator operations:

    [PRE71]

7.  Lastly, define the heat map renderer prototype and implement the algorithm for better visualization of the range of values:

    [PRE72]

## How it works...

This demo effectively opens up the possibility of rendering any image sequence with real-time processing using the OpenGL pipeline at the fragment shading stage. The following screenshot is an example that illustrates the use of this powerful OpenGL framework to display one frame of a video (showing the authors of the book) without the Sobel filter enabled:

![How it works...](img/9727OS_04_08.jpg)

Now, with the Sobel filter and heat map rendering enabled, we see an interesting way to visualize the world using real-time OpenGL texture mapping and processing using custom shaders:

![How it works...](img/9727OS_04_09.jpg)

Further fine-tuning of the threshold parameters and converting the result into grayscale (in the `texture_sobel.frag` file) leads to an aesthetically interesting output:

[PRE73]

![How it works...](img/9727OS_04_10.jpg)

In addition, we can blend these results with the original video feed to create filtered effects in real time by modifying the main function in the shader program (`texture_sobel.frag`):

[PRE74]

![How it works...](img/9727OS_04_11.jpg)

To illustrate the use of the exact same program to visualize imaging datasets, here is an example that shows a volumetric dataset of a human finger imaged with **Optical Coherence Tomography** (**OCT**), simply by changing the input video's filename:

![How it works...](img/9727OS_04_12.jpg)

This screenshot represents one of 256 cross-sectional images of the nail bed in this volumetric OCT dataset (which is exported in a movie file format).

Here is another example that shows a volumetric dataset of a scar specimen imaged with **Polarization-Sensitive Optical Coherence Tomography** (**PS-OCT**), which provides label-free, intrinsic contrast to the scar region:

![How it works...](img/9727OS_04_13.jpg)

In this case, the volumetric PS-OCT dataset was rendered with the ImageJ 3D Viewer and converted into a movie file. The colors denote the **Degree of Polarization** (**DOP**), which is a measure of the randomness of the polarization states of light (a low DOP in yellow/green and a high DOP in blue), in the skin. The scar region is characterized by a high DOP compared to the normal skin.

As we have demonstrated here, this program can be easily adopted (by changing the input video source) to display many types of datasets, such as endoscopy videos or other volumetric imaging datasets. The utility of OpenGL becomes apparent in demanding applications that require real-time processing of very large datasets.