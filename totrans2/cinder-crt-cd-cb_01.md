# Chapter 1. Getting Started

In this chapter we will cover:

*   Creating a project for a basic application
*   Creating a project for a screensaver application
*   Creating a project for an iOS touch application
*   Understanding the basic structure of an application
*   Responding to mouse input
*   Responding to key input
*   Responding to touch input
*   Accessing the files dropped onto the application window
*   Adjusting a scene after resizing the window
*   Using resources on Windows
*   Using resources on OSX and iOS
*   Using assets

# Introduction

In this chapter we'll learn the fundamentals of creating applications using Cinder.

We'll start by creating different types of applications on the different platforms that Cinder supports using a powerful tool called TinderBox.

We'll cover the basic structure of an application and see how to respond to user input events.

Finally, we will learn how to use resources on Windows and Mac.

# Creating a project for a basic application

In this recipe, we'll learn how to create a project for a basic desktop application for Windows and Mac OSX.

## Getting ready

Projects can be created using a powerful tool called TinderBox. TinderBox comes bundled in your Cinder download and contains templates for creating projects for different applications for both Microsoft Visual C++ 2010 and OSX Xcode.

To find Tinderbox, go to your Cinder folder, inside which you will find a folder named `tools` with, TinderBox application in it.

![Getting ready](img/8703OS_1_1.jpg)

The first time you open TinderBox, you'll be asked to specify the folder where you installed Cinder. You'll need to do this only the first time you open TinderBox. If you need to redefine the location of Cinder installation, you can do so by selecting the **File** menu and then **Preferences** on Windows or selecting the **TinderBox** menu and then **Preferences** on OS X.

## How to do it…

We'll use TinderBox, a utility tool that comes bundled with Cinder that allows for the easy creation of projects. Perform the following steps to create a project for a basic application:

1.  Open TinderBox and choose your project's location. In the main **TinderBox** window select **BasicApp** as **Target** and **OpenGL** as **Template**, as shown in the following screenshot:![How to do it…](img/8703OS_1_2.jpg)
2.  Choose your project's location. The **Naming Prefix** and **Project Name** fields will default to the project's name, as shown in the following screenshot:![How to do it…](img/8703OS_1_3.jpg)
3.  Select the compilers you want to use for your project, either Microsoft Visual C++ 2010 and/or OS X Xcode.![How to do it…](img/8703OS_1_4.jpg)
4.  Click on the **Create** button and TinderBox will show you the folder where your new project is located. TinderBox will remain open; you can close it now.

## How it works...

TinderBox will create the selected projects for the chosen platforms (Visual C++ 2010 and OS X Xcode) and create references to the compiled Cinder library. It will also create the application's class as a subclass of `ci::app::AppBasic`. It will also create some sample code with a basic example to help you get started.

## There's more...

Your project name and naming prefix will be, by default, the name of the folder in which the project is being created. You can edit this if you want, but always make sure both **Project Name** and **Naming Prefix** fields do not have spaces as you might get errors.

The naming prefix will be used to name your application's class by adding the `App` suffix. For example, if you set your **Naming Prefix** field as `MyCinderTest`, your application's class will be `MyCinderTestApp`.

# Creating a project for a screensaver application

In this recipe, we will learn how to create a project for a desktop screensaver for both Windows and Mac OS X.

## Getting ready

To get ready with TinderBox, please refer to the *Getting ready* section of the previous *Creating a project for a basic application* recipe.

## How to do it…

We'll use TinderBox, a utility tool that comes bundled with Cinder that allows easy creation of projects. Perform the following steps to create a project for a screensaver application:

1.  Open TinderBox and choose your project's location. In the main **TinderBox** window select **Screensaver** as **Target** and **OpenGL** as **Template**, as shown in the following screenshot:![How to do it…](img/8703OS_1_5.jpg)
2.  Select the compilers you want to create a project to, either Microsoft Visual C++ 2010 and/or OS X Xcode.
3.  Click on **Create** and TinderBox will direct you to the folder where your project was created.

## How it works...

TinderBox will create both a project for you and link it against the compiled Cinder library. It will also create the application's class and make it a subclass of `ci::app::AppScreenSaver`, which is the class with all the basic functionality for a screensaver application. It will also create some sample code with a basic example to help you get started.

# Creating a project for an iOS touch application

In this recipe, we'll learn how to create a project for an application that runs on iOS devices such as iPhone and iPad.

## Getting ready

To get ready with TinderBox, please refer to the *Getting ready* section of the *Creating a project for a basic application* recipe.

Please note that the iOS touch application will only work on iOS devices such as iPhones and iPads, and that the projects created with TinderBox will be for OSX Xcode only.

## How to do it…

We'll use TinderBox, a utility tool that comes bundled with Cinder that allows easy creation of projects. Perform the following steps to create a project for an iOS touch application:

1.  Open TinderBox and choose your project's location. In the main **TinderBox** window select **Cocoa Touch** as **Target** and **Simple** as **Template**, as shown in the following screenshot:![How to do it…](img/8703OS_1_6.jpg)
2.  Select the compilers you want to create a project to, either Microsoft Visual C++ 2010 and/or OS X Xcode.
3.  Click on **Create** and TinderBox will direct you to the folder where your project was created.

## How it works...

TinderBox will create an OS X Xcode project and create the references to link against the compiled Cinder library. It will also create the application's class as a subclass of `ci::app::AppCocoaTouch`, which is the class with all the basic functionality for a screensaver application. It will also create some sample code with a basic example to help you get started.

This application is built on top of Apple's Cocoa Touch framework to create iOS applications.

# Understanding the basic structure of an application

Your application's class can have several methods that will be called at different points during the execution of the program. The following table lists these methods:

| Method | Usage |
| --- | --- |
| `prepareSettings` | This method is called once at the very beginning of the application, before creating the renderer. Here, we may define several parameters of the application before the application gets initialized, such as the frame rate or the size of the window. If none are specified, the application will initialize with default values. |
| `setup` | This method is called once at the beginning of the application lifecycle. Here, you initialize all members and prepare the application for running. |
| `update` | This method is called in a loop during the application's runtime before the `draw` method. It is used to animate and update the states of the application's components. Even though you may update them during the `draw` method, it is recommended you keep the update and drawing routines separate as a matter of organization. |
| `draw` | This method is called in a loop during the application's runtime after the update. All drawing code should be placed here. |
| `shutdown` | This method is called just before the application exits. Use it to do any necessary cleanup such as freeing memory and allocated resources or shutting down hardware devices. |

To execute our code, we must overwrite these methods with our own code.

## Getting ready

It is not mandatory to override all of the preceding methods; you can use the ones that your application requires specifically. For example, if you do not want to do any drawing, you may omit the `draw` method.

In this recipe and for the sake of learning, we will implement all of them.

Declare the following methods in your class declaration:

[PRE0]

## How to do it…

We will implement several methods that make up the basic structure of an application. Perform the following steps to do so:

1.  Implement the `prepareSettings` method. Here we can define, for example, the size of the window, its title, and the frame rate:

    [PRE1]

2.  Implement the `setup` method. Here we should initialize all members of the application's class. For example, to initialize capturing from a webcam we would declare the following members:

    [PRE2]

3.  Implement the `update` method. As an example, we will print the current frame count to the console:

    [PRE3]

4.  Implement the `draw` method with all the drawing commands. Here we clear the background with black and draw a red circle:

    [PRE4]

5.  Implement the `shutdown` method. This method should take code for doing cleanup, for example, to shut down threads or save the state of your application.
6.  Here's a sample code for saving some parameters in an XML format:

    [PRE5]

## How it works...

Our application's superclass implements the preceding methods as virtual empty methods.

When the application runs, these methods are called, calling our own code we implemented or the parent class' empty method if we didn't.

In step 1 we defined several application parameters in the `prepareSettings` method. It is not recommended to use the `setup` method to initialize these parameters, as it means that the renderer has to be initialized with the default values and then readjusted during the setup. The result is extra initialization time.

## There's more...

There are other callbacks that respond to user input such as mouse and keyboard events, resizing of the window, and dragging files onto the application window. These are described in more detail in the *Responding to mouse input*, *Responding to key input*, *Responding to touch input*, *Accessing files dragged on the application window*, and *Adjusting a scene after resizing the window* recipes.

## See also

To learn how to create a basic app with TinderBox, read the *Creating a project for a basic application* recipe.

# Responding to mouse input

An application can respond to mouse interaction through several event handlers that are called depending on the action being performed.

The existing handlers that respond to mouse interaction are listed in the following table:

| Method | Usage |
| --- | --- |
| `mouseDown` | This is called when the user presses a mouse button |
| `mouseUp` | This is called when the user releases a mouse button |
| `mouseWheel` | This is called when the user rotates the mouse wheel |
| `mouseMove` | This is called when the mouse is moved without any button pressed |
| `mouseDrag` | This is called when the mouse is moved with any button pressed |

It is not mandatory to implement all of the preceding methods; you can implement only the ones required by your application.

## Getting ready

Implement the necessary event handlers according to the mouse events you need to respond to. For example, to create an application that responds to all available mouse events, you must implement the following code inside your main class declaration:

[PRE6]

The `MouseEvent` object passed as a parameter contains information about the mouse event.

## How to do it…

We will learn how to work with the `ci::app::MouseEvent` class to respond to mouse events. Perform the following steps to do so:

1.  To get the position where the event has happened, in terms of screen coordinates, we can type in the following line of code:

    [PRE7]

    Or we can get the separate x and y coordinates by calling the `getX` and `getY` methods:

    [PRE8]

2.  The `MouseEvent` object also lets us know which mouse button triggered the event by calling the `isLeft`, `isMiddle`, or `isRight` methods. They return a `bool` value indicating if it was the left, middle, or right button, respectively.

    [PRE9]

3.  To know if the event was triggered by pressing a mouse button, we can call the `isLeftDown`, `isRightDown`, and `isMiddleDown` methods that return `true` depending on whether the left, right, or middle buttons of the mouse were pressed.

    [PRE10]

4.  The `getWheelIncrement` method returns a `float` value with the movement increment of the mouse wheel.

    [PRE11]

5.  It is also possible to know if a special key was being pressed during the event. The `isShiftDown` method returns `true` if the *Shift* key was pressed, the `isAltDown` method returns `true` if the *Alt* key was pressed, `isControlDown` returns `true` if the *control* key was pressed, and `isMetaDown` returns `true` if the Windows key was pressed on Windows or the *option* key was pressed on OS X, `isAccelDown` returns `true` if the *Ctrl* key was pressed on Windows or the *command* key was pressed on OS X.

## How it works

A Cinder application responds internally to the system's native mouse events. It then creates a `ci::app::MouseEvent` object using the native information and calls the necessary mouse event handlers of our application's class.

## There's more...

It is also possible to access the native modifier mask by calling the `getNativeModifiers` method. These are platform-specific values that Cinder uses internally and may be of use for advanced uses.

# Responding to key input

A Cinder application can respond to key events through several callbacks.

The available callbacks that get called by keyboard interaction are listed in the following table:

| Method | Usage |
| --- | --- |
| `keyDown` | This is called when the user first presses a key and called repeatedly if a key is kept pressed. |
| `keyUp` | This is called when a key is released. |

Both these methods receive a `ci::app::KeyEvent` object as a parameter with information about the event such as the key code being pressed or if any special key (such as *Shift* or *control*) is being pressed.

It is not mandatory to implement all of the preceding key event handlers; you can implement only the ones that your application requires.

## Getting ready

Implement the necessary event handlers according to what key events you need to respond to. For example, to create an application that responds to both key down and key up events, you must declare the following methods:

[PRE12]

The `ci::app::KeyEvent` parameter contains information about the key event.

## How to do it…

We will learn how to work with the `ci::app::KeyEvent` class to learn how to understand key events. Perform the following steps to do so:

1.  To get the ASCII code of the character that triggered the key event, you can type in the following line of code:

    [PRE13]

2.  To respond to special keys that do not map to the ASCII character table, we must call the `getCode` method that retrieves an `int` value that can be mapped to a character table in the `ci::app::KeyEvent` class. To test, for example, if the key event was triggered by the *Esc* key you can type in the following line of code:

    [PRE14]

    `escPressed` will be `true` if the escape key triggered the event, or `false` otherwise.

3.  The `ci::app::KeyEvent` parameter also has information about modifier keys that were pressed during the event. The `isShiftDown` method returns `true` if the *Shift* key was pressed, `isAltDown` returns `true` if the *Alt* key was pressed, `isControlDown` returns `true` if the *control* key was pressed, `isMetaDown` returns `true` if the Windows key was pressed on Windows or the *command* key was pressed on OS X, and `isAccelDown` returns `true` if the *Ctrl* key was pressed on Windows or the *command* key was pressed on OS X.

## How it works…

A Cinder application responds internally to the system's native key events. When receiving a native key event, it creates a `ci::app::KeyEvent` object based on the native information and calls the correspondent callback on our application's class.

## There's more...

It is also possible to access the native key code by calling the `getNativeKeyCode` method. This method returns an `int` value with the native, platform-specific code of the key. It can be important for advanced uses.

# Responding to touch input

A Cinder application can receive several touch events.

The available touch event handlers that get called by touch interaction are listed in the following table:

| Method | Usage |
| --- | --- |
| `touchesBegan` | This is called when new touches are detected |
| `touchesMoved` | This is called when existing touches move |
| `touchesEnded` | This is called when existing touches are removed |

All of the preceding methods receive a `ci::app::TouchEvent` object as a parameter with a `std::vector` of `ci::app::TouchEvent::Touch` objects with information about each touch detected. Since many devices can detect and respond to several touches simultaneously, it is possible and common for a touch event to contain several touches.

It is not mandatory to implement all of the preceding event handlers; you can use the ones your application requires specifically.

Cinder applications can respond to touch events on any touch-enabled device running Windows 7, OS X, or iOS.

## Getting ready

Implement the necessary touch event handlers according to the touch events you want to respond to. For example, to respond to all available touch events (touches added, touches moved, and touches removed), you would need to declare and implement the following methods:

[PRE15]

## How to do it…

We will learn how to work with the `ci::app::TouchEvent` class to understand touch events. Perform the following steps to do so:

1.  To access the list of touches, you can type in the following line of code:

    [PRE16]

    Iterate through the container to access each individual element.

    [PRE17]

2.  You can get the position of a touch by calling the `getPos` method that returns a `Vec2f` value with its position or using the `getX` and `getY` methods to receive the x and y coordinates separately, for example:

    [PRE18]

3.  The `getId` method returns a `uint32_t` value with a unique ID for the `touch` object. This ID is persistent throughout the lifecycle of the touch, which means you can use it to keep track of a specific touch as you access it on the different touch events.

    For example, to make an application where we draw lines using our fingers, we can create `std::map` that associates each line, in the form of a `ci::PolyLine<Vec2f>` object, with a `uint32_t` key with the unique ID of a touch.

    We need to include the file with `std::map` and `PolyLine` to our project by adding the following code snippet to the beginning of the source file:

    [PRE19]

4.  We can now declare the container:

    [PRE20]

5.  In the `touchesBegan` method we create a new line for each detected touch and map it to the unique ID of each touch:

    [PRE21]

6.  In the `touchesMoved` method, we add the position of each touch to its corresponding line:

    [PRE22]

7.  In the `touchesEnded` method, we remove the line that corresponds to a touch being removed:

    [PRE23]

8.  Finally, the lines can be drawn. Here we clear the background with black and draw the lines with in white. The following is the implementation of the `draw` method:

    [PRE24]

    The following is a screenshot of our app running after drawing some lines:

    ![How to do it…](img/8703OS_1_7.jpg)

## How it works…

A Cinder application responds internally to the system calls for any touch event. It will then create a `ci::app::TouchEvent` object with information about the event and call the corresponding event handler in our application's class. The way to respond to touch events becomes uniform across the Windows and Mac platforms.

The `ci::app::TouchEvent` class contains only one accessor method that returns a `const` reference to a `std::vector<TouchEvent::Touch>` container. This container has one `ci::app::TouchEvent::Touch` object for each detected touch and contains information about the touch.

The `ci::app::TouchEvent::Touch` object contains information about the touch including position and previous position, unique ID, the time stamp, and a pointer to the native event object which maps to `UITouch` on Cocoa Touch and `TOUCHPOINT` on Windows 7.

## There's more...

At any time, it is also possible to get a container with all active touches by calling the `getActiveTouches` method. It returns a `const` reference to a `std::vector<TouchEvent::Touch>` container. It offers flexibility when working with touch applications as it can be accessed outside the touch event methods.

For example, if you want to draw a solid red circle around each active touch, you can add the following code snippet to your `draw` method:

[PRE25]

# Accessing files dropped onto the application window

Cinder applications can respond to files dropped onto the application window through the callback, `fileDrop` . This method takes a `ci::app::FileDropEvent` object as a parameter with information about the event.

## Getting ready

Your application must implement a `fileDrop` method which takes a `ci::app::FileDropEvent` object as a parameter.

Add the following method to the application's class declaration:

[PRE26]

## How to do it…

We will learn how to work with the `ci::app::FileDropEvent` object to work with file drop events. Perform the following steps to do so:

1.  In the method implementation you can use the `ci::app::FileDropEvent` parameter to access the list of files dropped onto the application by calling the `getFiles` method. This method returns a `conststd::vector` container with `fs::path` objects:

    [PRE27]

2.  The position where the files were dropped onto the window can be accessed through the following callback methods:

    *   To get a `ci::Vec2i` object with the position of the files dropped, type in the following line of code:

        [PRE28]

    *   To get the x and y coordinates separately, you can use the `getX` and `getY` methods, for example:

        [PRE29]

3.  You can find the number of dropped files by using the `getNumFiles` method:

    [PRE30]

4.  To access a specific file, if you already know its index, you can use the `getFile` method and pass the index as a parameter.

    For example, to access the file with an index of `2`, you can use the following line of code:

    [PRE31]

## How it works…

A Cinder application will respond to the system's native event for file drops. It will then create a `ci::app::FileDropEvent` object with information about the event and call the `fileDrop` callback in our application. This way Cinder creates a uniform way of responding to file drop events across the Windows and OS X platforms.

## There's more…

Cinder uses `ci::fs::path` objects to define paths. These are `typedef` instances of `boost::filesystem::path` objects and allow for much greater flexibility when working with paths. To learn more about the `fs::path` objects, please refer to the `boost::filesystem` library reference, available at [http://www.boost.org/doc/libs/1_50_0/libs/filesystem/doc/index.htm](http://www.boost.org/doc/libs/1_50_0/libs/filesystem/doc/index.htm).

# Adjusting a scene after resizing the window

Cinder applications can respond to resizing the window by implementing the resize event. This method takes a `ci::app::ResizeEvent` parameter with information about the event.

## Getting ready

If your application doesn't have a `resize` method, implement one. In the application's class declaration, add the following line of code:

[PRE32]

In the method's implementation, you can use the `ResizeEvent` parameter to find information about the window's new size and format.

## How to do it…

We will learn how to work with the `ci::app::ResizeEvent` parameter to respond to window resize events. Perform the following steps to do so:

1.  To find the new size of the window, you can use the `getSize` method which returns a `ci::Vec2iwith` object, the window's width as the x component, and the height as the y component.

    [PRE33]

    The `getWidth and getHeight` methods both return `int` values with the window's width and height respectively, for example:

    [PRE34]

2.  The `getAspectRatio` method returns a `float` value with the aspect ratio of the window, which is the ratio between its width and height:

    [PRE35]

3.  Any element on screen that needs adjusting must use the new window size to recalculate its properties. For example, to have a rectangle that is drawn at the center of the window with a 20 pixel margin on all sides, we must first declare a `ci::Rectf` object in the class declaration:

    [PRE36]

    In the setup we set its properties so that it has a 20 pixel margin on all sides from the window:

    [PRE37]

4.  To draw the rectangle with a red color, add the following code snippet to the `draw` method:

    [PRE38]

5.  In the `resize` method, we must recalculate the rectangle properties so that it resizes itself to maintain the 20 pixel margin on all sides of the window:

    [PRE39]

6.  Run the application and resize the window. The rectangle will maintain its relative size and position according to the window size.![How to do it…](img/8703OS_1_8.jpg)

## How it works…

A Cinder application responds internally to the system's window resize events. It will then create the `ci::app::ResizeEvent` object and call the `resize` method on our application's class. This way Cinder creates a uniform way of dealing with resize events across the Windows and Mac platforms.

# Using resources on Windows

It is common for Windows applications to use external files either to load images, play audio or video, or to load or save settings on XML files.

Resources are external files to your application that are embedded in the application's executable file. Resource files are hidden from the user to avoid alterations.

## Getting ready

Resources should be stored in a folder named `resources` in your project folder. If this folder does not exist, create it.

Resources on Windows must be referenced in a file called `Resources.rc`. This file should be placed next to the Visual C++ solution in the `vc10` folder. If this file does not exist, you must create it as an empty file. If the `resources.rs` file is not included already in your project solution, you must add it by right-clicking on the **Resources** filter and choosing **Add** and then **ExistingItem**. Navigate to the file and select it. As a convention, this file should be kept in the same folder as the project solution.

## How to do it…

We will use Visual C++ 2010 to add resources to our applications on Windows. Perform the following steps to do so:

1.  Open the Visual C++ solution and open the `resources.h` file inside the **Header Files** filter.
2.  Add the `#pragma once` macro to your file to prevent it from being included more than once in your project and include the `CinderResources.h` file.

    [PRE40]

3.  On Windows, each resource must have a unique ID number. As a convention, the IDs are defined as sequential numbers starting from 128, but you can use other IDs if it suits you better. Make sure to never use the same ID twice. You must also define a type string. The type string is used to identify resources of the same type, for example, the string `IMAGE` may be used when declaring image resources, `VIDEO` for declaring video resources, and so on.
4.  To simplify writing multiplatform code, Cinder has a macro for declaring resources that can be used on both Windows and Mac.

    For example, to declare the resource of an image file named `image.png`, we would type in the following line of code:

    [PRE41]

    The first parameter of the `CINDER_RESOURCE` macro is the relative path to the folder where the resource file is, in this case the default `resources` folder.

    The second parameter is the name of the file, and after that comes the unique ID of this resource, and finally its type string.

5.  Now we need to add our `resources` macro to the `resources.rs` file, as follows:

    [PRE42]

6.  This resource is now ready to be used in our application. To load this image into `ci::gl::Texture` we simply include the `Texture.h` file in our application's source code:

    [PRE43]

7.  We can now declare the texture:

    [PRE44]

8.  In the setup, we create the texture by loading the resource:

    [PRE45]

9.  The texture is now ready to be drawn on screen. To draw the image at position (20, 20), we will type in the following line of code inside the `draw` method:

    [PRE46]

## How it works...

The `resources.rc` file is used by a resource compiler to embed resources into the executable file as binary data.

## There's more...

Cinder allows writing code to use resources that is coherent across all supported platforms, but the way resources are handled on Windows and OS X/iOS is slightly different. To learn how to use resources on a Mac, please read the *Using resources on iOS and OS X* recipe.

# Using resources on iOS and OS X

It is common for Windows applications to use external files either to load images, play audio or video, or to load or save settings on XML files.

Resources are external files to your application that are included in the applications bundle. Resource files are hidden from the user to avoid alterations.

Cinder allows writing code to use resources that is equal when writing Windows or Mac applications, but the way resources are handled is slightly different. To learn how to use resources on Windows, please read the *Using resources on Windows* recipe.

## Getting ready

Resources should be stored in a folder named `resources` in your `project` folder. If this folder does not exist, create it.

## How to do it…

We will use Xcode to add resources to our application on iOS and OS X. Perform the following steps to do so:

1.  Place any resource file you wish to use in the `resources` folder.
2.  Add these files to your project by right-clicking on the **Resources** filter in your Xcode project and selecting **Add** and then **ExistingFiles**, navigate to the `resources` folder, and select the resource files you wish to add.
3.  To load a resource in your code, you use the `loadResource` method and pass the name of the resource file. For example, to load an image named `image.png`, you should first create the `gl::Texture` member in the class declaration:

    [PRE47]

4.  In the `setup` method, we initialize the texture with the following resource:

    [PRE48]

5.  The texture is now ready to be drawn in the window. To draw it at position (20, 20), type in the following line of code inside the `draw` method:

    [PRE49]

## How it works...

On iOS and OS X, applications are actually folders that contain all the necessary files to run the application, such as the Unix executable file, the frameworks used, and the resources. You can access the content of these folders by clicking on any Mac application and selecting **Show Package Contents**.

When you add resources to the `resources` folder in your Xcode project, these files are copied during the build stage to the `resources` folder of your application bundle.

## There's more...

You can also load resources using the same `loadResource` method that is used in Windows applications. This is very useful when writing cross-platform applications so that no changes are necessary in your code.

You should create the `resource` macro in the `Resources.h` file, and add the unique resource ID and its type string. For example, to load the image `image.png`, you can type in the following code snippet:

[PRE50]

And this is what the `Resources.rc` file should look like:

[PRE51]

Using the preceding example to load an image, the only difference is that we would load the texture with the following line of code:

[PRE52]

The resource unique ID and type string will be ignored in Mac applications, but adding them allows creating code that is cross-platform.

# Using assets

In this recipe, we will learn how we can load and use assets.

## Getting ready

As an example for this recipe, we will load and display an asset image.

Place an image file inside the `assets` folder in your project directory and name it `image.png`.

Include the following files at the top of your source code:

[PRE53]

Also add the following useful `using` statements:

[PRE54]

## How to do it…

As an example, we will learn how we can load and display an image asset. Perform the following steps to do so:

1.  Declare a `ci::gl::Texture` object:

    [PRE55]

2.  In the `setup` method let's load the image asset. We will use a `try/catch` block in if it is not possible to load the asset.

    [PRE56]

3.  In the `draw` method we will draw the texture. We will use an `if` statement to check if the texture has been successfully initialized:

    [PRE57]

## How it works…

The first application uses an asset Cinder, which will try to find its default `assets` folder. It will begin by searching the executable or application bundle folder, depending on the platform, and continue searching its parent's folder up to five levels. This is done to accommodate for different project setups.

## There's more…

You can add an additional `assets` folder using the `addAssetDirectory` method, which takes a `ci::fs::path` object as a parameter. Every time Cinder searches for an asset, it will first look in its default `asset` folder and then in every folder the user may have added.

You can also create subfolders inside the `assets` folder, for example, if our image was inside a subfolder named `My Images`, we would type in the following code snippet in the `setup` method:

[PRE58]

It is also possible to know the path where a specific folder lies. To do this, use the `getAssetPath` method, which takes a `ci::fs::path` object as a parameter with the name of the file.