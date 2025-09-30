# Chapter 2. Preparing for Development

In this chapter, we will cover:

*   Setting up a GUI for tweaking parameters
*   Saving and loading configurations
*   Making a snapshot of the current parameter state
*   Using MayaCamUI
*   Using 3D space guides
*   Communicating with other software
*   Preparing your application for iOS

# Introduction

In this chapter, we will introduce several simple recipes that can be very useful during the development process.

# Setting up a GUI for tweaking parameters

**Graphical User Interface** (**GUI**) is often required for controlling and tuning your Cinder application. In many cases, you spend more time tweaking the application parameters to achieve the desired result than writing the code. It is true especially when you are working on some generative graphics.

Cinder provides a convenient and easy-to-use GUI via the `InterfaceGl` class.

![Setting up a GUI for tweaking parameters](img/8703OS_02_01.jpg)

## Getting ready

To make the `InterfaceGl` class available in your Cinder application, all you have to do is include one header file.

[PRE0]

## How to do it…

Follow the steps given here to add a GUI to your Cinder application.

1.  Let's start with preparing different types of variables within our main class, which we will be manipulating using the GUI.

    [PRE1]

2.  Next, declare the `InterfaceGl` class member like this:

    [PRE2]

3.  Now we move to the `setup` method and initialize our GUI window passing `"Parameters"` as the window caption and size to the `InterfaceGl` constructor:

    [PRE3]

4.  And now we can add and configure controls for our variables:

    [PRE4]

    Take a look at the `addParam` method and its parameters. The first parameter is just the field caption. The second parameter is a pointer to the variable where the value is stored. There are a bunch of supported variable types, such as `bool`, `float`, `double`, `int`, `Vec3f`, `Quatf`, `Color`, `ColorA`, and `std::string`.

    The possible variables types and their interface representations are tabulated in the following table:

    | Type | Representation |
    | --- | --- |
    | `std:string` | ![How to do it…](img/8703OS_02_02.jpg) |
    | `Numerical: int, float, double` | ![How to do it…](img/8703OS_02_03.jpg) |
    | `bool` | ![How to do it…](img/8703OS_02_04.jpg) |
    | `ci::Vec3f` | ![How to do it…](img/8703OS_02_05.jpg) |
    | `ci::Quatf` | ![How to do it…](img/8703OS_02_06.jpg) |
    | `ci::Color` | ![How to do it…](img/8703OS_02_07.jpg) |
    | `ci::ColorA` | ![How to do it…](img/8703OS_02_08.jpg) |
    | Enumerated parameter | ![How to do it…](img/8703OS_02_09.jpg) |

    The third parameter defines the control options. In the following table, you can find some commonly used options and their short explanations:

    | Name | Explanation |
    | --- | --- |
    | `min` | The minimum possible value of a numeric variable |
    | `max` | The maximum possible value of a numeric variable |
    | `step` | Defines the number of significant digits printed after the period for floating point variables |
    | `key` | Keyboard shortcut for calling button callback |
    | `keyIncr` | Keyboard shortcut for incrementing the value |
    | `keyDecr` | Keyboard shortcut for decrementing the value |
    | `readonly` | Setting the value to `true` makes a variable read-only in GUI |
    | `precision` | Defines the number of significant digits printed after the period for floating point variables |

    ### Tip

    You can find the complete documentation of the available options on the AntTweakBar page at the following address: [http://anttweakbar.sourceforge.net/doc/tools:anttweakbar:varparamsyntax](http://anttweakbar.sourceforge.net/doc/tools:anttweakbar:varparamsyntax).

5.  The last thing to do is invoke the `InterfaceGl::draw()` method. We will do this at the end of the `draw` method in our main class by typing the following code line:

    [PRE5]

## How it works...

In the `setup` method we will set up the GUI window and then add controls, setting up a name in the first parameter of the `addParam` method. In a second parameter, we are pointing to the variable we want to link the GUI element to. Whenever we change values through the GUI, the linked variable will be updated.

## There's more...

There are a few more options for `InterfaceGl`, if you need more control over built-in GUI mechanism, please refer to the *AntTweakBar* documentation which you can find on the project page mentioned in the *See also* section of this recipe.

### Buttons

You can also add buttons to the InterfaceGl (CIT) panel with callbacks to some functions. For example:

[PRE6]

Clicking on the **Start** button in the GUI fires the `start` method of the `MainApp` class.

### Panel position

A convenient way to control the position of the GUI panel is through the usage of the *AntTweekBar* facility. You have to include an additional header file:

[PRE7]

And now you can change the position of the GUI panel with this code line:

[PRE8]

In this case, `Parameters` is the GUI panel name and the `position` option takes x and y as values.

## See also

There are some good looking GUI libraries available as CinderBlocks. Cinder has an extensions system called blocks. The idea behind CinderBlocks is to provide easy-to-use integration with many third-party libraries. You can find how to add examples of CinderBlocks to your project in the *Communicating with other software* recipe.

### SimpleGUI

An alternative GUI developed by *Marcin Ignac* as a CinderBlock can be found at [https://github.com/vorg/MowaLibs/tree/master/SimpleGUI](https://github.com/vorg/MowaLibs/tree/master/SimpleGUI).

### ciUI

You can check out an alternative user interface developed by *Reza Ali* as a CinderBlock at [http://www.syedrezaali.com/blog/?p=2366](http://www.syedrezaali.com/blog/?p=2366).

### AntTweakBar

`InterfaceGl` in Cinder is built on top of *AntTweakBar*; you can find its documentation at [http://www.antisphere.com/Wiki/tools:anttweakbar](http://www.antisphere.com/Wiki/tools:anttweakbar).

# Saving and loading configurations

Many applications that you will develop operate on input parameters set by the user. For example, it could be the color or position of some graphical elements or parameters used to set up communication with other applications. Reading configurations from external files is necessary for your applications. We will use a built-in Cinder support for reading and writing XML files to implement the configuration persistence mechanism.

## Getting ready

Create two configurable variables in the main class: the IP address and the port of the host we are communicating with.

[PRE9]

## How to do it...

Now we will implement the `loadConfig` and `saveConfig` methods and use them to load the configuration on application startup and save the changes while closing.

1.  Include the two following additional headers:

    [PRE10]

2.  We will prepare two methods for loading and saving the XML configuration file.

    [PRE11]

3.  Now in the `setup` method, inside our main class, we will put:

    [PRE12]

4.  After this we will implement the `shutdown` method as follows:

    [PRE13]

5.  And don't forget to declare the `shutdown` method in the main class:

    [PRE14]

## How it works...

The first two methods, `loadConfig` and `saveConfig`, are essential. The `loadConfig` method tries to open the `config.xml` file and find the `general` node. Inside the `general` node should be the `hostIP` and `hostPort` nodes. The values of these nodes will be assigned to corresponding variables in our application: `mHostIP` and `mHostPort`.

The `shutdown` method is automatically triggered by Cinder just before the application closes, so our configuration values will be stored in the XML file when we quit the application. Finally, our configuration XML file looks like this:

[PRE15]

You can see clearly that the nodes are referring to application variables.

## See also

You can write your own configuration loader and saver or use the existing CinderBlock.

### Cinder-Config

Cinder-Config is a small CinderBlock for creating configuration files along with `InterfaceGl`.

[https://github.com/dawidgorny/Cinder-Config](https://github.com/dawidgorny/Cinder-Config)

# Making a snapshot of the current parameter state

We will implement a simple but useful mechanism for saving and loading the parameters' states. The code used in the examples will be based on the previous recipes.

## Getting ready

Let's say we have a variable that we are changing frequently. In this case, it will be the color of some element we are drawing and the main class will have the following member variable:

[PRE16]

## How to do it...

We will use a built-in XML parser and the `fileDrop` event handler.

1.  We have to include the following additional headers:

    [PRE17]

2.  First, we implement two methods for loading and saving parameters:

    [PRE18]

3.  Now we declare a class member. It will be the flag to trigger snapshot creation:

    [PRE19]

4.  Assign a value to it value inside the `setup` method:

    [PRE20]

5.  At the end of the `draw` method we put the following code, just before the `params::InterfaceGl::draw();` line:

    [PRE21]

6.  We want to make a button in our `InterfaceGl` window:

    [PRE22]

    As you can see we don't have the `makeSnapshotClick` method yet. It is simple to implement:

    [PRE23]

7.  The last step will be adding the following method for *drag-and-drop* support:

    [PRE24]

## How it works...

We have two methods for loading and storing the `mColor` values in an XML file. These methods are `loadParameters` and `saveParameters`.

The code we put inside the `draw` method needs some explanation. We are waiting for the `mMakeSnapshot` method to be set to `true` and then we are creating a timestamp to avoid overwriting previous snapshots.The next two lines store the chosen values by invoking the `saveParameters` method and save a current window view as a PNG file using the `writeImage` function. Please notice that we have put that code before invoking `InterfaceGl::draw`, so we save the window view without the GUI.

A nice thing we have here is the *drag-and-drop* feature for loading snapshot files. It's implemented in the `fileDrop` method; Cinder invokes this method every time files are dropped to your application window. First, we get a path to the dropped file; in the case of multiple files, we are taking only one. Then we invoke the `loadParameters` method with the dropped file path as an argument.

# Using MayaCamUI

We are going to add to your 3D scene a navigation facility known to us since we modelled a 3D software. Using `MayaCamUI`, you can do this with just a few lines of code.

## Getting ready

We need to have some 3D objects in our scene. You can use some primitives provided by Cinder, for example:

[PRE25]

A color cube is a cube with a different color on each face, so it is easy to determine the orientation.

![Getting ready](img/8703OS_02_10.jpg)

## How to do it...

Perform the following steps to create camera navigation:

1.  We need the `MayaCam.h` header file:

    [PRE26]

2.  We also need some member declarations in the main class:

    [PRE27]

3.  Inside the `setup` method, we are going to set up the camera's initial state:

    [PRE28]

4.  Now we have to implement three methods:

    [PRE29]

5.  Apply camera matrices before your 3D drawing stuff inside the `draw` method:

    [PRE30]

## How it works...

Inside the `setup` method, we set the initial camera settings. While the window is resizing, we have to update the aspect ratio of our camera, so we put the code for this in the `resize` method. This method is automatically invoked by Cinder each time the window of our application is resized. We catch mouse events inside the `mouseDown` and `mouseDrag` methods. You can click and drag your mouse for tumbling, right-click for zooming, and use the middle button for panning. Now you have interaction similar to a common 3D modeling software in your own application.

# Using 3D space guides

We will try to use built-in Cinder methods to visualize some basic information about the scene we are working on. It should make working with 3D space more comfortable.

## Getting ready

We will need the `MayaCamUI` navigation that we have implemented in the previous recipe.

## How to do it...

We will draw some objects that will help to visualize and find the orientation of a 3D scene.

1.  We will add another camera besides `MayaCamUI`. Let's start by adding member declarations to the main class:

    [PRE31]

2.  Then we will set the initial values inside the `setup` method:

    [PRE32]

3.  We have to update the aspect ratio of `mSceneCamera` inside the `resize` method:

    [PRE33]

4.  Now we will implement the `keyDown` method that will switch between two cameras by pressing the *1* or *2* keys on the keyboard:

    [PRE34]

5.  Another method we are going to use is `drawGrid`, which looks like this:

    [PRE35]

6.  After that, we can implement our main drawing routine, so here is the whole `draw` method:

    [PRE36]

## How it works...

We have two cameras; `mSceneCam` is for final rendering and `mMayaCam` is for the preview of objects in our scene. You can switch between them by pressing the *1* or *2* keys. The default camera is `MayaCam`.

![How it works...](img/8703OS_02_11.jpg)

In the previous screenshot, you can see the whole scene set up with the elements, such as the origin of the coordinate system, the construction grid that lets you keep orientation in 3D space easily, and the `mSceneCam` frustum and vector visualization between two points in 3D space. You can navigate through this space using `MayaCamUI`.

If you press the *2* key, you will switch to the view of `mSceneCam`, so you will see only your 3D objects without guides as shown in the following screenshot:

![How it works...](img/8703OS_02_12.jpg)

# Communicating with other software

We will implement an example communication between two Cinder applications written in Cinder to illustrate how we can send and receive signals. Each of these two applications can be replaced by a non-Cinder application very easily.

We are going to use the **Open Sound Control** (**OSC**) messaging format, which is dedicated for communication between wide ranges of multimedia devices over the network. OSC uses UDP protocol, providing flexibility and performance. Each message consists of URL-like addresses and arguments of integer, float, or string type. The popularity of OSC makes it a great tool for connecting different environments or applications developed with different technologies over the network or even on the local machine.

## Getting ready

While downloading the Cinder package we are also downloading four primary blocks. One of them is the `osc` block located in the `blocks` directory. First, we will add a new group to our XCode project root and name it `Blocks`, and after that we will drag the `osc` folder inside the `Blocks` group. Be sure the **Create groups for any added folders** options and **MainApp** in the **Add to targets** section are checked.

![Getting ready](img/8703OS_02_13.jpg)

We only need to include an `src` from the `osc` folders, so we will delete references to the `lib` and `samples` folders from our project tree. The final project structure should look like the following screenshot:

![Getting ready](img/8703OS_02_14.jpg)

Now we have to add a path to the `OSC` library file as another linker flag's position in your project's build settings:

[PRE37]

### Tip

**CINDER_PATH** should be set as a user-defined setting in the build settings of your project and it should be the path to Cinder root directory.

## How to do it...

First we will cover instructions for the *sender*, and then for the *listener*.

### Sender

We will implement an application that sends OSC messages.

1.  We have to include an additional header file:

    [PRE38]

2.  After that we can use the `osc::Sender` class, so let's declare the needed properties in the main class:

    [PRE39]

3.  Now we have to set up our sender inside the `setup` method:

    [PRE40]

4.  Set the default value for `mObjectPosition` to be the center of the window:

    [PRE41]

5.  We can now implement the `mouseDrag` method, which includes two major operations—updating the object position according to the mouse position and sending the position information via OSC.

    [PRE42]

6.  The last thing we need to do is to draw a method just to visualize the position of the object:

    [PRE43]

### Listener

We will implement an application that receives `OSC` messages.

1.  We have to include an additional header file:

    [PRE44]

2.  After that we can use the `osc::Listener` class, so let's declare the required properties in the main class:

    [PRE45]

3.  Now we have to set up our listener object inside the `setup` method, passing the port number for listening as a parameter:

    [PRE46]

4.  And the default value for `mObjectPosition` to be the center of the window:

    [PRE47]

5.  Inside the `update` method, we will be listening for the incoming `OSC` messages:

    [PRE48]

6.  Our `draw` method will be almost the same as the sender version, but instead of stroked circle we will draw a filled circle:

    [PRE49]

# How it works...

We have implemented the sender application that sends the position of the mouse via OSC protocol. Those messages, with the address `/obj/position`, can be received by any non-Cinder OSC application implemented in many other frameworks and programming languages. The first argument in the message is the x axis position of the mouse and the second argument is the y axis position. Both are of the `float` type.

![How it works...](img/8703OS_02_15.jpg)

In our case, the application that receives messages is another Cinder application that draws a filled circle at exactly the same position where you point it in the sender application window.

![How it works...](img/8703OS_02_16.jpg)

# There's more...

That was just a short example of the possibilities that OSC offers. This simple communication method can be applied even in very complex projects. OSC works great when several devices are working as independent units. But at some point, data coming from them is processed; for example, frames coming from the camera can be processed by the computer vision software and results sent over the network to another machine projecting the visualization. Implementation on top of the UDP protocol gives not only performance, because of the fact that transmitting data is faster than using TCP, but also implementation is much simpler without a connection handshake.

## Broadcast

You can send OSC messages to all the hosts on your network by setting a broadcast address as a destination host: `255.255.255.255`. For example, in case of subnets, you can use `192.168.1.255` .

### Tip

If you have problems with compilation under Mac OS X 10.7 because of a linker error, try to set **Inline Methods Hidden** to **No** in your project's build settings.

# See also

You can find more information about OSC implementations by checking out the following links.

## OSC in Flash

To support receiving and sending OSC messages in your ActionScript 3.0 code you can use the following library: [http://bubblebird.at/tuioflash/](http://bubblebird.at/tuioflash/)

## OSC in Processing

To support **OSC** protocol in your **Processing** sketch you can use following library: [http://www.sojamo.de/libraries/oscP5/](http://www.sojamo.de/libraries/oscP5/)

## OSC in openFrameworks

To support receiving and sending OSC messages in your `openFrameworks` project, you can use the `ofxOsc` add-on: [http://ofxaddons.com/repos/112](http://ofxaddons.com/repos/112)

## OpenSoundControl Protocol

You can find more information about OSC protocol and related tools at its official site: [http://opensoundcontrol.org/](http://opensoundcontrol.org/).

# Preparing your application for iOS

The big benefit of using Cinder is the resulting multiplatform code. In most cases, your application can be compiled on Windows, Mac OS X, and iOS without significant modifications.

## Getting ready

If you want to run your applications on iOS devices, you will need to register as an Apple Developer and purchase the iOS Developer Program.

## How to do it...

After registering yourself as an Apple Developer or purchasing the iOS Developer Program, you can create an initial XCode project for iOS using Tinderbox.

1.  After running Tinderbox you have to set **Target** to **Cocoa Touch**.![How to do it...](img/8703OS_02_18.jpg)
2.  It will generate a project structure for you, supporting iOS events that are specific for multitouch screens.

    We can use events for multiple touches and for easy access to accelerometer data. The main difference between touch and mouse events is that there can be more than one active touch points while there is only one mouse cursor. Because of that, each touch session has an ID that can be read from `TouchEvent` object.

    | Method | Describe |
    | --- | --- |
    | `touchesBegan( TouchEvent event )` | Beginning of a multitouch sequence |
    | `touchesMoved( TouchEvent event )` | Drags during a multitouch sequence |
    | `touchesEnded( TouchEvent event )` | The end of a multitouch sequence |
    | `getActiveTouches()` | Returns all active touches |
    | `accelerated( AccelEvent event )` | Vector 3D of the acceleration direction |

## See also

I recommend you take a look at the sample projects included in the Cinder package: `MultiTouchBasic` and `iPhoneAccelerometer`.

### Apple Developer Center

You can find more information about the iOS Developer Program here: [https://developer.apple.com/](https://developer.apple.com/)