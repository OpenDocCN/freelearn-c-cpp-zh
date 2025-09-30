# Chapter 7. An Introduction to Real-time Graphics Rendering on a Mobile Platform using OpenGL ES 3.0

In this chapter, we will cover the following topics:

*   Setting up the Android SDK
*   Setting up the **Android Native Development Kit** (**NDK**)
*   Developing a basic framework to integrate the Android NDK
*   Creating your first Android application with OpenGL ES 3.0

# Introduction

In this chapter, we will transition to an increasingly powerful and ubiquitous computing platform by demonstrating how to visualize data on the latest mobile devices, from smart phones to tablets, using **OpenGL for Embedded Systems** (**OpenGL ES**). As mobile devices become more ubiquitous and with their increasing computing capability, we now have an unprecedented opportunity to develop novel interactive data visualization tools using high-performance graphics hardware directly integrated into modern mobile devices.

OpenGL ES plays an important role in standardizing the 2D and 3D graphics APIs to allow the large-scale deployment of mobile applications on embedded systems with various hardware settings. Among the various mobile platforms (predominantly Google Android, Apple iOS, and Microsoft Windows Phone), the Android mobile operating system is currently one of the most popular ones. Therefore, in this chapter, we will focus primarily on the development of an Android-based application (API 18 and higher) using OpenGL ES 3.0, which provides a newer version of GLSL support (including full support for integer and 32-bit floating point operations) and enhanced texture rendering support. Nevertheless, OpenGL ES 3.0 is also supported on other mobile platforms, such as Apple iOS and Microsoft Phone.

Here, we will first introduce how to set up the Android development platform, including the SDK that provides the essential tools to build mobile applications, and the NDK, which enables the use of native-code languages (C/C++) for high-performance scientific computing and simulations by exploiting direct hardware acceleration. We will provide a script to simplify the process of deploying your first Android-based application on your mobile device.

# Setting up the Android SDK

The Google Android OS website provides a standalone package for Android application development called the **Android SDK**. It contains all the necessary compilation and debugging tools to develop an Android application (except native code support, which is provided by the Android NDK). The upcoming steps explain the installation procedure in Mac OS X or, similarly, in Linux, with minor modifications to the script and binary packages required.

## How to do it...

To install the Android SDK, follow these steps:

1.  Download the standalone package from the Android Developers website at [http://dl.google.com/android/android-sdk_r24.3.3-macosx.zip](http://dl.google.com/android/android-sdk_r24.3.3-macosx.zip).
2.  Create a new directory called `3rd_party/android` and move the setup file into this folder:

    [PRE0]

3.  Unzip the package:

    [PRE1]

4.  Execute the Android SDK Manager:

    [PRE2]

5.  Select **Android 4.3.1 (API 18)** from the list of packages in addition to the default options. Deselect **Android M (API22, MBC preview)** and **Android 5.1.1 (API 22)**. Press the **Install 9 packages...** button on the **Android SDK Manager** screen, as shown here:![How to do it...](img/9727OS_07_01.jpg)
6.  Select **Accept** **License** and click on the **Install** button:![How to do it...](img/9727OS_07_02.jpg)
7.  To verify the installation, type the following command into the terminal:

    [PRE3]

8.  This is an example that illustrates the successful installation of the Android 4.3.1 platform:

    [PRE4]

9.  Finally, we will install Apache Ant to automate the software build process for Android application development. We can easily obtain the Apache Ant package by using MacPort with the command line or from its official website at [http://ant.apache.org/](http://ant.apache.org/):

    [PRE5]

## See also

To install the Android SDK in Linux or Windows, download the corresponding installation files and follow the instructions on the Android developer website at [https://developer.android.com/sdk/index.html](https://developer.android.com/sdk/index.html).

The setup procedures to set up the Android SDK in Linux are essentially identical using the command-line interface, except that a different standalone package should be downloaded using this link: [http://dl.google.com/android/android-sdk_r24.3.3-linux.tgz](http://dl.google.com/android/android-sdk_r24.3.3-linux.tgz).

In addition, for Windows users, the standalone package can be obtained using this link: [http://dl.google.com/android/installer_r24.3.3-windows.exe](http://dl.google.com/android/installer_r24.3.3-windows.exe).

To verify that your mobile phone has proper OpenGL ES 3.0 support, consult the Android documentation on how to check the OpenGL ES version at runtime: [http://developer.android.com/guide/topics/graphics/opengl.html#version-check](http://developer.android.com/guide/topics/graphics/opengl.html#version-check).

# Setting up the Android Native Development Kit (NDK)

The Android NDK environment is essential for native-code language development. Here, we will outline the setup steps for the Mac OS X platform again.

## How to do it...

To install the Android NDK, follow these steps:

1.  Download the NDK installation package from the Android developer website at [http://dl.google.com/android/ndk/android-ndk-r10e-darwin-x86_64.bin](http://dl.google.com/android/ndk/android-ndk-r10e-darwin-x86_64.bin).
2.  Move the setup file into the same installation folder:

    [PRE6]

3.  Set the permission of the file to be an executable:

    [PRE7]

4.  Run the NDK installation package:

    [PRE8]

5.  The installation process is fully automated and the following output confirms the successful installation of the Android NDK:

    [PRE9]

## See also

To install the Android NDK on Linux or Windows, download the corresponding installation files and follow the instructions on the Android developer website at [https://developer.android.com/tools/sdk/ndk/index.html](https://developer.android.com/tools/sdk/ndk/index.html).

# Developing a basic framework to integrate the Android NDK

Now that we have successfully installed the Android SDK and NDK, we will demonstrate how to develop a basic framework to integrate native C/C++ code into a Java-based Android application. Here, we describe the general mechanism to create high-performance code for deployment on mobile devices using OpenGL ES 3.0.

OpenGL ES 3.0 supports both Java and C/C++ interfaces. Depending on the specific requirements of the application, you may choose to implement the solution in Java due to its flexibility and portability. For high-performance computing and applications that require a high memory bandwidth, it is preferable that you use the NDK for fine-grain optimization and memory management. In addition, we can port our existing libraries, such as OpenCV with Android NDK, using static library linking. The cross-platform compilation capability opens up many possibilities for real-time image and signal processing on a mobile platform with minimal development effort.

Here, we introduce a basic framework that consists of three classes: `GL3JNIActivity`, `GL3JNIView`, and `GL3JNIActivity`. We show a simplified class diagram in the following figure, illustrating the relationship between the classes. The native code (C/C++) is implemented separately and will be described in detail in the next section:

![Developing a basic framework to integrate the Android NDK](img/9727OS_07_03.jpg)

## How to do it...

First, we will create the core Java source files that are essential to an Android application. These files serve as a wrapper for our OpenGL ES 3.0 native code:

1.  In the project directory, create a folder named `src/com/android/gl3jni` with the following command:

    [PRE10]

2.  Create the first class, `GL3JNIActivity`, in the Java source file, `GL3JNIActivity.java`, within the new folder, `src/com/android/gl3jni/`:

    [PRE11]

3.  Next, implement the `GL3JNIView` class, which handles the OpenGL rendering setup in the `GL3JNIView.java` source file inside `src/com/android/gl3jni/`:

    [PRE12]

4.  Finally, create the `GL3JNILib` class to handle native library loading and calling in `GL3JNILib.java` inside `src/com/android/gl3jni`:

    [PRE13]

5.  Now, in the project directory of the project, add the `AndroidManifest.xml` file, which contains all the essential information about your application on the Android system:

    [PRE14]

6.  In the `res/values/` directory, add the `strings.xml` file, which saves our application's name:

    [PRE15]

## How it works...

The following class diagram illustrates the core functions and relationships between the classes. Similar to all other Android applications with a user interface, we define the **Activity** class, which handles the core interactions. The implementation of `GL3JNIActivity` is straightforward. It captures the events from the Android application (for example, `onPause` and `onResume`) and also creates an instance of the `GL3JNIView` class, which handles graphics rendering. Instead of adding UI elements, such as textboxes or labels, we create a surface based on `GLSurfaceView`, which handles hardware-accelerated OpenGL rendering:

![How it works...](img/9727OS_07_04.jpg)

The `GL3JNIView` class is a subclass of the `GLSurfaceView` class, which provides a dedicated surface for OpenGL rendering. We choose the RGB8 color mode, a 16-bit depth buffer, and no stencil with the `setEGLConfigChooser` function and ensure that the environment is set up for OpenGL ES 3.0 by using the `setEGLContextClientVersion` function. The `setRenderer` function then registers the custom `Renderer` class, which is responsible for the actual OpenGL rendering.

The `Renderer` class implements the key event functions—`onDrawFrame`, `onSurfaceChanged`, and `onSurfaceCreated`—in the rendering loop. These functions connect to the native implementation (C/C++) portion of the code that is handled by the `GL3JNILib` class.

Finally, the `GL3JNILib` class creates the interface to communicate with the native code functions. First, it loads the native library named `gl3jni`, which contains the actual OpenGL ES 3.0 implementation. The function prototypes, `step` and `init`, are used to interface with the native code, which will be defined separately in the next section. Note that we can also pass in the canvas width and height values to the native functions as parameters.

The `AndroidManifest.xml` and `strings.xml` files are the configuration files required by the Android application, and they must be stored in the root directory of the project in the XML format. The `AndroidManifest.xml` file defines all the essential information including the name of the Java package and the declaration of permission requirements (for example, file read/write access), as well as the minimum version of the Android API that the application requires.

## See also

For further information on Android application development, the Android Developers website provides detailed documentation on the API at [http://developer.android.com/guide/index.html](http://developer.android.com/guide/index.html).

For further information on using OpenGL ES within an Android application, the Android programming guide describes the programming workflow in detail and provides useful examples at [http://developer.android.com/training/graphics/opengl/environment.html](http://developer.android.com/training/graphics/opengl/environment.html).

# Creating your first Android application with OpenGL ES 3.0

In this section, we will complete our implementation with native code in C/C++ to create the first Android application with OpenGL ES 3.0\. As illustrated in the simplified class diagram, the Java code only provides the basic interface on the mobile device. Now, on the C/C++ side, we implement all the functionalities previously defined on the Java side and also include all the required libraries from OpenGL ES 3.0 (inside the `main_simple.cpp` file). The `main_simple.cpp` file also defines the key interface between the C/C++ and Java side by using the **Java Native Interface** (**JNI**):

![Creating your first Android application with OpenGL ES 3.0](img/9727OS_07_05.jpg)

## Getting ready

We assume that you have installed all the prerequisite tools from the Android SDK and NDK in addition to setting up the basic framework introduced in the previous section. Also, you should review the basics of shader programming, introduced in earlier chapters, before you proceed.

## How to do it...

Here, we describe the implementation of the OpenGL ES 3.0 native code to complete the demo application:

1.  In the project directory, create a folder named `jni` by using the following command:

    [PRE16]

2.  Create a file named `main_simple.cpp` and store it inside the `jni` directory.
3.  Include all necessary header files for JNI and OpenGL ES 3.0:

    [PRE17]

4.  Include the logging header and define the macros to show the debug messages:

    [PRE18]

5.  Declare the shader program variables for our demo application:

    [PRE19]

6.  Define the shader program code for the vertex shader and the fragment shader:

    [PRE20]

7.  Implement the error call handlers for OpenGL ES, using the Android log:

    [PRE21]

8.  Implement the vertex or fragment program-loading mechanisms. The warning and error messages are redirected to the Android log output:

    [PRE22]

9.  Implement the shader program creation mechanism. The function also attaches and links the shader program:

    [PRE23]

10.  Create a function to handle the initialization. This function is a helper function that handles requests from the Java side:

    [PRE24]

11.  Set up the rendering function that draws a triangle on the screen with red, green, and blue vertices:

    [PRE25]

12.  Define the JNI prototypes that connect to the Java side. These calls are the interfaces to communicate between the Java code and the C/C++ native code:

    [PRE26]

13.  Set up the internal function calls with the helper functions:

    [PRE27]

14.  Now that we have completed the implementation of the native code, we must compile the code and link it to the Android application. To compile the code, create a `build` file that is similar to a `Makefile`, called `Android.mk`, in the `jni` folder:

    [PRE28]

15.  In addition, we must create an `Application.mk` file that provides information about the build type, such as the **Application Binary Interface** (**ABI**). The `Application.mk` file must be stored inside the `jni` directory:

    [PRE29]

16.  At this point, we should have the following list of files in the root directory:

    [PRE30]

To compile the native source code and deploy our application on a mobile phone, run the following `build` script in the terminal, which is shown as follows:

1.  Set up our environment variables for the SDK and the NDK. (Note that the following relative paths assume that the SDK and NDK are installed 3 levels outside the current directory, where the `compile.sh` and `install.sh` scripts are executed in the code package. These paths should be modified to match your code directory structure as necessary.):

    [PRE31]

2.  Initialize the project with the android `update` command for the first-time compilation. This will generate all the necessary files (such as the `build.xml` file) for later steps:

    [PRE32]

3.  Compile the JNI native code with the `build` command:

    [PRE33]

4.  Run the `build` command. Apache Ant takes the `build.xml` script and builds the **Android Application Package** (**APK**) file that is ready for deployment:

    [PRE34]

5.  Install the Android application by using the **Android Debug Bridge** (**adb**) command:

    [PRE35]

For this command to work, before connecting the mobile device through the USB port, ensure that the USB Debugging mode is enabled and accept any prompts for security-related warnings. On most devices, you can find this option by navigating to **Settings** | **Applications** | **Development** or **Settings** | **Developer**. However, on Android 4.2 or higher, this option is hidden by default and must be enabled by navigating to **Settings** | **About Phone** (or **About Tablet**) and tapping **Build Number** multiple times. For further details, follow the instructions provided on the official Android Developer website at [http://developer.android.com/tools/device.html](http://developer.android.com/tools/device.html). Here is a sample screenshot of an Android phone with the USB debugging mode successfully configured:

![How to do it...](img/9727OS_07_06.jpg)

After the application is installed, we can execute the application as we normally do with any other Android application by opening it directly using the application icon on the phone, as shown here:

![How to do it...](img/9727OS_07_07.jpg)

A screenshot after launching the application is shown next. Note that the CPU monitor has been enabled to show the CPU utilization. This is not enabled by default but can be found in **Developer Options**. The application supports both the portrait and landscape modes and the graphics automatically scale to the window size upon changing the frame buffer size:

![How to do it...](img/9727OS_07_08.jpg)

Here is another screenshot of the landscape mode:

![How to do it...](img/9727OS_07_09.jpg)

## How it works...

This chapter demonstrates the portability of our approach in previous chapters. Essentially, the native code developed in this chapter resembles what we covered in previous chapters. In particular, the shader program's creation and loading mechanism is virtually identical, except that we have used a predefined string (`static char[]`) to simplify the complexity of loading files in Android. However, there are some subtle differences. Here, we will list the differences and new features.

In the fragment program and vertex program, we need to add the `#version 300 es` directive to ensure that the shader code can access the new features, such as uniform blocks and the full support of integer and floating point operations. For example, OpenGL ES 3.0 replaces the attribute and varying qualifiers with the **in** and **out** keywords. This standardization allows much faster code development of OpenGL on various platforms.

The other notable difference is that we have replaced the GLFW library completely with the EGL library, which comes as a standard library in Android, for context management. All event handling, such as Windows management and user inputs, are now handled through the Android API and the native code is only responsible for graphics rendering.

The Android log and error reporting system is now accessible through the Android `adb` program. The interaction is similar to a terminal output, and we can see the log in real time with the following command:

[PRE36]

For example, our application reports the OpenGL ES version, as well as the extensions supported by the mobile device in the log. With the preceding command, we can extract the following information:

[PRE37]

The real-time log data is very useful for debugging and can allow developers to quickly analyze the problem.

One common question is how the Java and C/C++ elements communicate with each other. The JNI syntax is rather puzzling to understand in the first place, but we can decode it by carefully analyzing the following code snippet:

[PRE38]

The `JNIEXPORT` and `JNICALL` tags allow the functions to be located in the shared library at runtime. The class name is specified by `com_android_gl3jni_GL3JNILib` (`com.android.gl3jni.GL3JNILib`), and `init` is the method name of the Java native function. As we can see, the period in the class name is replaced by an underscore. In addition, we have two additional parameters, namely the width and height of the frame buffer. More parameters can be simply appended to the end of the parameters' list in the function, as required.

In terms of backward compatibility, we can see that OpenGL 4.3 is a complete superset of OpenGL ES 3.0\. In OpenGL 3.1 and higher, we can see that the embedded system version of OpenGL and the standard Desktop version of OpenGL are slowly converging, which reduces the underlying complexity in maintaining various versions of OpenGL in the application life cycle.

## See also

A detailed description of the Android OS architecture is beyond the scope of this book. However, you are encouraged to consult the official developer workflow guide at [http://developer.android.com/tools/workflow/index.html](http://developer.android.com/tools/workflow/index.html).

Further information on the OpenGL ES Shading Language can be found at [https://www.khronos.org/registry/gles/specs/3.0/GLSL_ES_Specification_3.00.3.pdf](https://www.khronos.org/registry/gles/specs/3.0/GLSL_ES_Specification_3.00.3.pdf).