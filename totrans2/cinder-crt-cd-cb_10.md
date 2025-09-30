# Chapter 10. Interacting with the User

In this chapter we will learn how to receive and respond to input from the user. The following recipes will be covered in the chapter:

*   Creating an interactive object that responds to the mouse
*   Adding mouse events to our interactive object
*   Creating a slider
*   Creating a responsive text box
*   Dragging, scaling, and rotating objects using multi-touch

# Introduction

In this chapter we will create graphical objects that react to the user using both mouse and touch interaction. We will learn how to create simple graphical interfaces that have their own events for greater flexibility.

# Creating an interactive object that responds to the mouse

In this recipe, we will create an `InteractiveObject` class for making graphical objects that interact with the mouse cursor and executes the following actions:

| Action | Description |
| --- | --- |
| Pressed | The user pressed the mouse button while over the object. |
| Pressed outside | The user pressed the mouse button while outside the object. |
| Released | The mouse button is released after being pressed over the object and is still over the object. |
| Released outside | The mouse button is released outside the object. |
| Rolled over | The cursor moves over the object. |
| Rolled out | The cursor moves out of the object. |
| Dragged | The cursor is dragged while being over the object and after having pressed the object. |

For each of the previous actions, a virtual method will be called, and it would change the color of the object been drawn.

This object can be used as a base class to create interactive objects with more interesting graphics, such as textures.

## Getting ready

Create and add the following files to your project:

*   `InteractiveObject.h`
*   `InteractiveObject.cpp`

In the source file with your application class, include the `InteractiveObject.h` file and add the following `using` statements:

[PRE0]

## How to do it…

We will create an `InteractiveObject` class and make it responsive to mouse events.

1.  Move to the file `InteractiveObject.h` and add the `#pragma once` directive and include the following files:

    [PRE1]

2.  Declare the class `InteractiveObject`:

    [PRE2]

3.  Move on to the `InteractiveObject.cpp` file, and let's begin by including the `InteractiveObject.h` file and adding the following `using` statements:

    [PRE3]

4.  Let's begin by implementing `constructor` and `destructor`.

    [PRE4]

5.  In the `InteractiveObject::draw` method we will draw the rectangle using the appropriate colors:

    [PRE5]

6.  In the `pressed`, `released`, `rolledOver`, `rolledOut`, and `dragged` methods we will simply output to the console on which the action just happened:

    [PRE6]

7.  In the mouse event handlers we will check if the cursor is inside the object and update the `mPressed` and `mOver` variables accordingly. Every time the action is detected, we will also call the correspondent method.

    [PRE7]

8.  With our `InteractiveObject` class ready, let's move to our application's class source file. Let's begin by declaring an `InteractiveObject` object.

    [PRE8]

9.  In the `setup` method we will initialize `mObject`.

    [PRE9]

10.  We will need to declare the mouse event handlers.

    [PRE10]

11.  In the implementation of the previous methods we will simply call the corresponding method of `mObject`.

    [PRE11]

12.  In the implementation of the `draw` method, we will clear the background with black and call the `draw` method of `mObject`.

    [PRE12]

13.  Now build and run the application. Use the mouse to interact with the object. Whenever you press, release, or roll over or out of the object, a message will be sent to the console indicating the behavior.

## How it works…

The `InteractiveObject` class is to be used as a base class for interactive objects. The methods `pressed`, `released`, `rolledOver`, `rolledOut`, and `dragged` are specifically designed to be overridden.

The mouse handlers of `InteractiveObject` call the previous methods whenever an action is detected. By overriding the methods, it is possible to implement specific behavior.

The virtual `destructor` is declared so that extending classes can have their own `destructor`.

# Adding mouse events to our interactive object

In this recipe, we will continue with the previous recipe, *Creating an interactive object that responds to the mouse* and add the mouse events to our `InteractiveObject` class so that other objects can register and receive notifications whenever a mouse event occurs.

## Getting ready

Grab the code from the recipe *Creating an interactive object that responds to the mouse* and add it to your project, as we will continue on from what was made earlier.

## How to do it…

We will make our `InteractiveObject` class and send its own events whenever it interacts with the cursor.

1.  Let's create a class to use as an argument when sending events. Add the following code in the file `InteractiveObject.h` right before the `InteractiveObject` class declaration:

    [PRE13]

2.  In the `InteractiveObject` class, we will need to declare a member to manage the registered objects using the `ci::CallbakcMgr` class. Declare the following code as a protected member:

    [PRE14]

3.  Now we will need to add a method so that other objects can register to receive events. Since the method will use a template, we will declare and implement it in the `InteraciveObject.h` file. Add the following member method:

    [PRE15]

4.  Let's also create a method so that objects can unregister from receiving further events. Declare the following method:

    [PRE16]

5.  Let's implement the `removeListener` method. Add the following code in the `InteractiveObject.cpp` file:

    [PRE17]

6.  Modify the methods `mouseDown`, `mouseUp`, `mouseDrag`, and `mouseMove` so that `mEvents` gets called whenever an event occurs. The implementation of these methods should be as follows:

    [PRE18]

7.  With our `InteractiveObject` class ready, we need to register our application class to receive its events. In your application class declaration add the following method:

    [PRE19]

8.  Let's implement the `receivedEvent` method. We will check what type of event has been received and print a message to the console.

    [PRE20]

9.  All that is left is to register for the events. In the `setup` method add the following code after `mObject` has been initialized:

    [PRE21]

10.  Now build and run the application and use the mouse to interact with the rectangle on the window. Whenever a mouse event occurs on `mObject`, our method, `receivedEvent`, will be called.

## How it works…

We are using the template class `ci::CallbakMgr` to manage our event listeners. This class takes a template with the signature of the methods that can be registered. In our previous code, we declared `mEvents` to be of type `ci::CallbakcMgr<void(InteractiveObjectEvent)>;` it means that only methods that return `void` and receive `InteractiveObejctEvent` as a parameter can be registered.

The template method `registerEvent` will take an object pointer and method pointer. These are bound to `std::function` using `std::bind1st` and added to `mEvents`. The method will return `ci::CallbackId` with the identification of the listener. The `ci::CallbackId` can be used to unregister listeners.

## There's more…

The `InteractiveObject` class is very useful for creating user interfaces. If we want to create a `Button` class using three textures (for displaying when pressed, over, and idle), we can do so as follows:

1.  Include the `InteractiveObject.h` and `cinder/gl/texture.h` files:

    [PRE22]

2.  Declare the following class:

    [PRE23]

# Creating a slider

In this recipe we will learn how to create a slider UI element by extending the `InteractiveObject` class mentioned in the *Creating an interactive object that responds to the mouse* recipe of this chapter.

![Creating a slider](img/8703OS_10_01.jpg)

## Getting ready

Please refer to the *Creating an interactive object that responds to the mouse* recipe to find the `InteractiveObject` class headers and source code.

## How to do it…

We will create a `Slider` class and show you how to use it.

1.  Add a new header file named `Slider.h` to your project:

    [PRE24]

2.  Inside the source file of your main application class, include the previously created header file:

    [PRE25]

3.  Add the new properties to your main class:

    [PRE26]

4.  Inside the `setup` method do the initialization of the `slider` objects:

    [PRE27]

5.  Add the following code for drawing sliders inside your `draw` method:

    [PRE28]

## How it works…

We created the `Slider` class by inheriting and overriding the `InteractiveObject` methods and properties. In step 1, we extended it with methods for controlling the position and dimensions of the `slider` object. The methods `getValue` and `setValue` can be used to retrieve or set the actual state of `slider`, which can vary from `0` to `1`.

In step 4, you can find the initialization of example sliders by setting the initial position, size, and value just after creating the `Slider` object. We are drawing example sliders along with captions and information about current state.

## See also

*   The recipe *Creating interactive object that responds to the mouse.*
*   The recipe *Dragging scaling, and rotating objects using multi-touch*.

# Creating a responsive text box

In this recipe we will learn how to create a text box that responds to the user's keystrokes. It will be active when pressed over by the mouse and inactive when the mouse is released outside the box.

## Getting ready

Grab the following files from the recipe *Creating an interactive object that responds to the mouse* and add them to your project:

*   `InteractiveObject.h`
*   `InteractiveObject.cpp`

Create and add the following files to your project:

*   `InteractiveTextBox.h`
*   `InteractiveTextBox.cpp`

## How to do it…

We will create an `InteractiveTextBox` class that inherits from `InteractiveObject` and adds text functionality.

1.  Go to the file `InteractiveTextBox.h` and add the `#pragma once` macro and include the necessary files.

    [PRE29]

2.  Now declare the `InteractiveTextBox` class, making it a subclass of `InteractiveObject` with the following members and methods:

    [PRE30]

3.  Now go to `InteractiveTextBox.cpp` and include the `InteractiveTextBox.h` file and add the following `using` statements:

    [PRE31]

4.  Now let's implement the constructor by initializing the parent class and setting up the internal `ci::TextBox`.

    [PRE32]

5.  In the `InteractiveTextBox::draw` method we will set the background color of `mTextBox` depending if it is active or not. We will also render `mTextBox` into `ci::gl::Texture` and draw it.

    [PRE33]

6.  Now let's implement the overridden methods `pressed` and `releasedOutside` to define the value of `mActive`.

    [PRE34]

7.  Finally, we need to implement the `keyPressed` method.

    If `mActive` is false this method will simply return. Otherwise, we will remove the last letter of `mText` if the key released was the *Backspace* key, or, add the corresponding letter if any other key was pressed.

    [PRE35]

    [PRE36]

8.  Now move to your application class source file and include the following file and the `using` statements:

    [PRE37]

9.  In your application class declare the following member:

    [PRE38]

10.  Let's initialize `mTextBox` in the `setup` method:

    [PRE39]

11.  In the `draw` method we will clear the background with black, enable `AlphaBlending`, and draw our `mTextBox`:

    [PRE40]

12.  We now need to declare the following mouse event handlers:

    [PRE41]

13.  And implement them by calling the respective mouse event handler of `mTextBox`:

    [PRE42]

14.  Now we just need to do the same with the key released event handler. Start by declaring it:

    [PRE43]

15.  And in it's implementation we will call the `keyUp` method of `mTextBox`.

    [PRE44]

16.  Now build and run the application. You will see a white textbox with the phrase **Write some text**. Press the text box and write some text. Click outside the text box to set the textbox as inactive.

## How it works…

Internally, our `InteractiveTextBox` uses a `ci::TextBox` object. This class manages the text inside a box with a specified width and height. We take advantage of that and update the text according to the keys that the user presses.

# Dragging, scaling, and rotating objects using multi-touch

In this recipe, we will learn how to create objects responsible to multi-touch gestures, such as dragging, scaling, or rotating by extending the `InteractiveObject` class mentioned in the *Creating an interactive object that responds to the mouse* recipe of this chapter. We are going to build an iOS application that uses iOS device multi-touch capabilities.

![Dragging, scaling, and rotating objects using multi-touch](img/8703OS_10_02.jpg)

## Getting ready

Please refer to the *Creating an interactive object that responds to the mouse* recipe to find the `InteractiveObject` class headers and source code and *Creating a project for an iOS touch application recipe from* [Chapter 1](ch01.html "Chapter 1. Getting Started").

## How to do it…

We will create an iPhone application with sample objects that can be dragged, scaled, or rotated.

1.  Add a new header file named `TouchInteractiveObject.h` to your project:

    [PRE45]

2.  Add a new source file named `TouchInteractiveObject.cpp` to your project and include the previously created header file by adding the following code line:

    [PRE46]

3.  Implement the constructor of `TouchInteractiveObject`:

    [PRE47]

4.  Implement the handlers for touch events:

    [PRE48]

    [PRE49]

5.  Now, implement the basic `draw` method for `TouchInteractiveObjects`:

    [PRE50]

6.  Here is the class, which inherits all the features of `TouchInteractiveObject`, but overrides the `draw` method and, in this case, we want our interactive object to be a circle. Add the following class definition to your main source file:

    [PRE51]

7.  Now take a look at your main application class file. Include the necessary header files:

    [PRE52]

8.  Add the `typedef` declaration:

    [PRE53]

9.  Add members to your application class to handle the objects:

    [PRE54]

10.  Inside the `setup` method initialize the objects:

    [PRE55]

11.  The `draw` method is simple and looks as follows:

    [PRE56]

12.  As you can see in the `setup` method we are using the function `getRandPos`, which returns a random position in screen boundaries with some margin:

    [PRE57]

## How it works…

We created the `TouchInteractiveObject` class by inheriting and overriding the `InteractiveObject` methods and properties. We also extended it with methods for controlling position and dimensions.

In step 3, we are initializing properties and registering callbacks for touch events. The next step is to implement these callbacks. On the `touchesBegan` event, we are checking if the object is touched by any of the new touches, but all the calculations of movements and gestures happen during `touchesMoved` event.

In step 6, you can see how simple it is to change the appearance and keep all the interactive capabilities of `TouchInteractiveObject` by overriding the `draw` method.

## There is more…

You can notice an issue that you are dragging multiple objects while they are overlapping. To solve that problem, we will add a simple object activation manager.

1.  Add a new class definition to your Cinder application:

    [PRE58]

2.  Add a new member to your application's main class:

    [PRE59]

3.  At the end of the `setup` method initialize `mObjMgr`, which is the object's manager, and add the previously initialized interactive objects:

    [PRE60]

4.  Add the `update` method to your main class as follows:

    [PRE61]

5.  Add two new methods to the `TouchInteractiveObject` class:

    [PRE62]