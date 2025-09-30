# Chapter 1. Getting Started with Cocos2d-x

In this chapter, we're going to install Cocos2d-x and set up the development environment. The following topics will be covered in this chapter:

*   Setting up our Android environment
*   Installing Cocos2d-x
*   Using the Cocos command
*   Building the project using Xcode
*   Building the project using Eclipse
*   Implementing multi-resolution support
*   Preparing your original game

# Introduction

Cocos2d-x is an open source, cross-platform game engine, which is free and mature. It can publish games for mobile devices and desktops, including iPhone, iPad, Android, Kindle, Windows, and Mac. Cocos2d-x is written in C++, so it can build on any platform. Cocos2d-x is open source written in C++, so we can feel free to read the game framework. Cocos2d-x is not a black box, and this proves to be a big advantage for us when we use it. Cocos2d-x version 3, which supports C++11, was only recently released. It also supports 3D and has an improved rendering performance. This book focuses on using version 3.4, which is the latest version of Cocos2d-x that was available at the time of writing this book. This book also focuses on iOS and Android development, and we'll be using Mac because we need it to develop iOS applications. This chapter explains how to set up Cocos2d-x.

# Setting up our Android Environment

## Getting ready

We begin by setting up our Android environment. If you wish to build only on iOS, you can skip this step. To follow this recipe, you will need some files.

The following list provides the prerequisites that need to be downloaded to set up Android:

*   Eclipse ADT (Android Developer Tools) with the Android SDK:

    [https://dl.google.com/android/adt/adt-bundle-mac-x86_64-20140702.zip](https://dl.google.com/android/adt/adt-bundle-mac-x86_64-20140702.zip)

    Eclipse ADT includes the Android SDK and Eclipse IDE. This is the Android development tool that is used to develop Android applications. Android Studio is an Android development IDE, but it is not supported to build NDK. The official site states that a version of Android Studio that supports NDK will be released soon. That's why we use Eclipse in this book.

*   Android NDK (Native Development Kit):

    [https://dl.google.com/android/ndk/android-ndk-r10c-darwin-x86_64.bin](https://dl.google.com/android/ndk/android-ndk-r10c-darwin-x86_64.bin)

    The NDK is required to build an Android application. You have to use NDK r10c. This is because compiling and linking errors may occur when using NDK r9 or an earlier version.

*   Apache ANT:

    You can download Apache ANT from [http://ant.apache.org/bindownload.cgi](http://ant.apache.org/bindownload.cgi)

    This is a java library that aids in building software. At the time of writing this book, version 1.9.4 was the latest stable version available.

## How to do it...

1.  You begin by installing Eclipse ADT with the Android SDK, and then continue to unzip the zip file to any working directory you are aware of. I recommend that you unzip it to the `Documents` folder (`~/adt-bundle-mac-x86_64-20140702`). ADT includes Android SDK and Eclipse. The SDK and Eclipse folders are located under the ADT folder. We call the SDK folder path that is located under the ADT folder `ANDROID_SDK_ROOT`. You have to remember it because you will use it the next recipe. Now, you can launch Eclipse from `~/adt-bundle-mac-x86_64-20140702/eclipse/Eclipse.app`.
2.  The next step is to update Android SDK:

    *   Open Eclipse from the `eclipse` folder located in ADT.
    *   Go to **Window** | **Android SDK Manager**.
    *   After opening **Android SDK Manager**, check **Tools** and the latest Android SDK (`API21`), `Android 2.3.3(API10)`, and any other SDK if necessary, as shown in the following screenshot:![How to do it...](img/B0561_01_1.jpg)
    *   Click on **Install packages...**.
    *   Select each license and click on **Accept,** as shown in the following screenshot:![How to do it...](img/B0561_01_2.jpg)
    *   After you accept all licenses, you will see that the **Install** button is enabled. Click on it.
    *   You have to wait for a long time to update and install the SDKs.

3.  Installing NDK:

    Open the terminal window and change the directory to the path from which you downloaded the package. Change the permission on the downloaded package and execute the package. For example:

    [PRE0]

    Finally, you move the `NDK` folder to the `Documents` folder. We call the installation path for NDK `NDK_ROOT`. `NDK_ROOT` is the address of the folder that contains the files, it helps the Cocos2dx engine to locate the native files of Android. You have to remember `NDK_ROOT` because you will use it in the next recipe.

4.  Installing Apache ANT:

    Unzip the file to the `Documents` folder. That's all. We call `ANT_ROOT` the installation path for ANT. You have to remember `ANT_ROOT`, as we'll be using it in the next recipe.

5.  Installing Java:

    By entering the following command in the terminal, you can automatically install Java (if you haven't installed it earlier):

    [PRE1]

    After installing it, you can check that it was successfully installed by entering the command again.

## How it works...

Let's take a look at what we did throughout the recipe:

*   Installing Eclipse: You can use Eclipse as an editor for Cocos2d-x
*   Installing ADT: You can develop Android applications on Eclipse
*   Installing NDK: You can build a C++ source code for Java
*   Installing ANT: You can use command line tools for Cocos2d-x

Now you've finished setting up the Android development environment. At this point, you know how to install them and their path. In the next recipe, you will use them to build and execute Android applications. This will be very useful when you want to debug Android applications.

# Installing Cocos2d-x

## Getting ready

To follow this recipe, you need to download the zip file from the official site of Cocos2d-x ([http://www.cocos2d-x.org/download](http://www.cocos2d-x.org/download)).

At the time of writing this book, version 3.4 was the latest stable version that was available. This version will be used throughout this book.

## How to do it...

1.  Unzip your file to any folder. This time, we will install the user's home directory. For example, if the user name is `syuhari`, then the install path is `/Users/syuhari/cocos2d-x-3.4`. In this book, we call it `COCOS_ROOT`.
2.  The following steps will guide you through the process of setting up Cocos2d-x:

    *   Open the terminal
    *   Change the directory in terminal to `COCOS_ROOT`, using the following command:

        [PRE2]

    *   Run `setup.py`, using the following command:

        [PRE3]

    *   The terminal will ask you for `NDK_ROOT`. Enter into `NDK_ROOT` path.
    *   The terminal will then ask you for `ANDROID_SDK_ROOT`. Enter the `ANDROID_SDK_ROOT` path.
    *   Finally, the terminal will ask you for `ANT_ROOT`. Enter the `ANT_ROOT` path.
    *   After the execution of the `setup.py` command, you need to execute the following command to add the system variables:

        [PRE4]

        ### Tip

        Open the `.bash_profile` file, and you will find that `setup.py` shows how to set each path in your system. You can view the `.bash_profile` file using the `cat` command:

        [PRE5]

3.  We now verify whether Cocos2d-x can be installed:

    *   Open the terminal and run the `cocos` command without parameters:

        [PRE6]

    *   If you can see a window like the following screenshot, you have successfully completed the Cocos2d-x install process:

    ![How to do it...](img/B0561_01_4.jpg)

## How it works...

Let's take a look at what we did throughout the above recipe. You can install Cocos2d-x by just unzipping it. You know `setup.py` is only setting up the `cocos` command and the path for Android build in the environment. Installing Cocos2d-x is very easy and simple. If you want to install a different version of Cocos2d-x, you can do that too. To do so, you need to follow the same steps that are given in this recipe, but they will be for a different version.

## There's more...

Setting up the Android environment is a bit tough. If you recently started to develop Cocos2d-x, you can skip the settings part of Android. and you can do it when you run on Android. In this case, you don't have to install Android SDK, NDK, and Apache ANT. Also, when you run `setup.py`, you only press *Enter* without entering a path for each question.

# Using the Cocos command

The next step is using the `cocos` command. It is a cross-platform tool with which you can create a new project, build it, run it, and deploy it. The `cocos` command works for all Cocos2d-x supported platforms and you don't need to use an IDE if you don't want to. In this recipe, we take a look at this command and explain how to use it.

## How to do it...

1.  You can use the `cocos` command help by executing it with the `--help` parameter, as follows:

    [PRE7]

2.  We then move on to generating our new project:

    First, we create a new Cocos2d-x project with the `cocos new` command, as shown here:

    [PRE8]

    The result of this command is shown the following screenshot:

    ![How to do it...](img/B0561_01_5.jpg)

    Behind the `new` parameter is the project name. The other parameters that are mentioned denote the following:

    *   `MyGame` is the name of your project.
    *   `-p` is the package name for Android. This is the application ID in the Google Play store. So, you should use the reverse domain name as the unique name.
    *   `-l` is the programming language used for the project. You should use `cpp` because we will use C++ in this book.
    *   `-d` is the location in which to generate the new project. This time, we generate it in the user's documents directory.

    You can look up these variables using the following command:

    [PRE9]

    Congratulations, you can generate your new project. The next step is to build and run using the `cocos` command.

3.  Compiling the project:

    If you want to build and run for iOS, you need to execute the following command:

    [PRE10]

    The parameters that are mentioned are explained as follows:

    *   `-s` is the directory of the project. This could be an absolute path or a relative path.
    *   `-p` denotes which platform to run on. If you want to run on Android you use `-p android`. The available options are IOS, Android, Win32, Mac, and Linux.
    *   You can run `cocos run –help` for more detailed information.

    The result of this command is shown in the following screenshot:

    ![How to do it...](img/B0561_01_6.jpg)
4.  You can now build and run iOS applications on cocos2d-x. However, you have to wait for a long time if this is your first time building an iOS application. It takes a long time to build a Cocos2d-x library, depending on if it was a clean build or a first build.![How to do it...](img/B0561_01_7.jpg)

## How it works...

The `cocos` command can create a new project and build it. You should use the `cocos` command if you want to create a new project. Of course, you can build using Xcode or Eclipse. You can easily develop and debug using these tools.

## There's more...

The cocos `run` command has other parameters. They are the following:

*   `--portrait` will set the project as a portrait. This command has no argument.
*   `--ios-bundleid` will set the bundle ID for the iOS project. However, it is not difficult to set it later.

The `cocos` command also includes some other commands, which are as follows:

*   The `compile` command: This command is used to build a project. The following patterns are useful parameters. You can see all parameters and options if you execute the `cocos compile [–h]` command:

    [PRE11]

*   The `deploy` command: This command only takes effect when the target platform is Android. It will re-install the specified project to the android device or simulator:

    [PRE12]

    ### Tip

    The `run` command continues to compile and deploy commands.

# Building the project using Xcode

## Getting ready

Before building the project using Xcode, you require Xcode with an iOS developer account to test it on a physical device. However, you can also test it on an iOS simulator. If you did not install Xcode, you can get it from the Mac App Store. Once you have installed it, get it activated.

## How to do it...

1.  Open your project from Xcode:

    You can open your project by double-clicking on the file placed at: `~/Documents/MyGame/proj.ios_mac/MyGame.xcodeproj`:

    ![How to do it...](img/B0561_01_8.jpg)
2.  Build and Run using Xcode:

    You should select an iOS simulator or real device on which you want to run your project.

## How it works...

If this is your first time building, it will take a long time but continue to build with confidence as it's the first time. You can develop your game faster if you develop and debug it using Xcode rather than Eclipse.

# Building the project using Eclipse

## Getting ready

You must finish the first recipe before you begin this step. If you have not finished it yet, you will need to install Eclipse.

## How to do it...

1.  Setting up `NDK_ROOT`:

    *   Open the preference of Eclipse
    *   Open **C++** | **Build** | **Environment**![How to do it...](img/B0561_01_9.jpg)

2.  Click on **Add** and set the new variable, the name is `NDK_ROOT`, and the value is `NDK_ROOT` path:![How to do it...](img/B0561_01_10.jpg)
3.  Importing your project into Eclipse:

    *   Open the file and click on **Import**
    *   Go to **Android** | **Existing Android Code into Workspace**
    *   Click on **Next**

    ![How to do it...](img/B0561_01_11.jpg)
4.  Import the project into Eclipse at `~/Documents/MyGame/proj.android`:![How to do it...](img/B0561_01_12.jpg)
5.  Importing the Cocos2d-x library into Eclipse:

    *   Perform the same steps from Step 3 to Step 4.
    *   Import the project `cocos2d lib` at `~/Documents/MyGame/cocos2d/cocos/platform/android/java`, using the following command:

        [PRE13]

    ![How to do it...](img/B0561_01_13.jpg)
6.  Build and Run:

    *   Click on the `Run` icon
    *   The first time, Eclipse will ask you to select a way to run your application. Select **Android Application** and click on **OK**, as shown in the following screenshot:![How to do it...](img/B0561_01_14.jpg)
    *   If you connected to the Android device on your Mac, you can run your game on your real device or an emulator. The following screenshot shows that it is running on Nexus5:![How to do it...](img/B0561_01_15.jpg)

7.  If you added `cpp` files into your project, you have to modify the `Android.mk` file at `~/Documents/MyGame/proj.android/jni/Android.mk`. This file is needed to build the NDK. This fix is required to add files.
8.  The original `Android.mk` would look as follows:

    [PRE14]

9.  If you added the `TitleScene.cpp` file, you have to modify it as shown in the following code:

    [PRE15]

The preceding example shows an instance of when you add the `TitleScene.cpp` file. However, if you are also adding other files, you need to add all the added files.

## How it works...

You get lots of errors when importing your project into Eclipse, but don't panic. After importing the Cocos2d-x library, errors soon disappear. This allows us to set the path of the NDK, Eclipse could compile C++. After you have modified the C++ codes, run your project in Eclipse. Eclipse automatically compiles C++ codes, Java codes, and then runs.

It is a tedious task to fix `Android.mk` again to add the C++ files. The following code is the original `Android.mk`:

[PRE16]

The following code is the customized `Android.mk` that adds C++ files automatically:

[PRE17]

The first line of the code gets C++ files to the `Classes` directory into the `CPP_FILES` variable. The second and third lines add C++ files into the `LOCAL_C_INCLUDES` variable. By doing so, C++ files will be automatically compiled in the NDK. If you need to compile a file other than the extension `.cpp` file, you will need to add it manually.

## There's more...

If you want to manually build C++ in NDK, you can use the following command:

[PRE18]

This script is located in `~/Documents/MyGame/proj.android` . It uses `ANDROID_SDK_ROOT` and `NDK_ROOT` in it. If you want to see its options, run `./build_native.py –help`.

# Implementing multi-resolution support

You may notice a difference in screen appearance on different devices. In some previous recipes, there is an iOS's screenshot and a Nexus 5's screenshot. It shows different image sizes. This image is `HelloWorld.png` located at `MyGame/Resources`. It is 480 x 320 pixels. In this recipe, we explain how to maintain the same size regardless of screen size.

## How to do it…

Open `AppDelegate.cpp` through Xcode, and modify the `AppDelegate::applicationDidFinishLaunching()` method by adding the code after the `director->setAnimationInterval(1.0/60.0);` line, as shown in the following code:

[PRE19]

In this book, we design the game with a screen size of iPhone's 3.5 inch screen. So, we set this screen size to the design resolution size by using the `setDesignResolutionSize` method. The last parameter is resolution policy. The following screenshot is the Nexus 5's screenshot after implementing multi-resolution:

![How to do it…](img/B0561_01_16.jpg)

The following screenshot is the iPhone 5 simulator's screenshot. You now know that both screenshots have the same appearance:

![How to do it…](img/B0561_01_17.jpg)

## How it works…

The resolution policy has `EXACT_FIT`, `NO_BORDER`, `SHOW_ALL`, `FIXED_HEIGHT,` and `FIXED_WIDTH`. These are explained as follows:

*   `EXACT_FIT`: The entire application is visible in the specified area without trying to preserve the original aspect ratio.
*   `NO_BORDER`: The entire application fills the specified area, without distortion but possibly with some cropping, while maintaining the original aspect ratio of the application.
*   `SHOW_ALL`: The entire application is visible in the specified area without distortion, while maintaining the internal the aspect ratio of the application. Borders can appear on two sides of the application.
*   `FIXED_HEIGHT`: The application takes the height of the design resolution size and modifies the width of the internal canvas so that it fits the aspect ratio of the device. No distortion will occur, however, you must make sure your application works on different aspect ratios.
*   `FIXED_WIDTH`: The application takes the width of the design resolution size and modifies the height of the internal canvas so that it fits the aspect ratio of the device. No distortion will occur, however, you must make sure your application works on different aspect ratios.

By implementing multi-resolution, regardless of screen size, you will maintain the image on the screen.

# Preparing your original game

In the next chapter, we will start the original game. You know there are a lot of comments and codes in `HelloWorldScene.cpp` and the `HelloWorldScene.h` file. That's why we will remove unnecessary codes in the template project and get started with the original game right away.

## How to do it…

1.  Open `HelloWorldScene.h` and remove the `menuCloseCallback` method and unnecessary comments. Now `HelloWorldScene.h` should look like the following code:

    [PRE20]

2.  The next step is to open `HelloWorldScene.cpp` and remove unnecessary comments, codes, and methods. Now `HelloWorldScene.cpp` should look like the following code:

    [PRE21]

3.  The next step is to remove unnecessary images in `resources`. Remove `CloseNormal.png`, `CloseSelected.png` and `HelloWorld.png` from the `Resources` folder in Xcode:![How to do it…](img/B0561_01_18.jpg)
4.  Finally, if you are developing only iOS and Android applications, you don't need files for other platforms such as Linux, Windows, and Windows Phone. You should remove these files.

    Before removing platform files, it should look like the following screenshot:

    ![How to do it…](img/B0561_01_19.jpg)

    After removing platform files, it should look like the following screenshot:

    ![How to do it…](img/B0561_01_20.jpg)

## How it works…

With this recipe, you can get the simplest project ready before removing unnecessary comments, codes, and methods. Removing unnecessary platform codes and resources is important for reducing the size of your application. If you start building your original game from scratch, you will need to follow this recipe or chances are, you may get a black screen if you build and run this project. In the next chapter, you can start coding within this simple project.