# Chapter 5. Playing with User Interfaces

In the previous chapters, we have learned how to build some simple games. This chapter will show you how to improve those games by adding a user interface to them. This chapter will cover two different possibilities of user interface:

*   Creating your own objects
*   Using a library that already exists–**Simple and Fast Graphical User Interface** (**SFGUI**)

By the end of this chapter, you should be able to create simple to complex interfaces to communicate with the player.

# What is a GUI?

A **Graphical User Interface** (**GUI**) is a mechanism that allows the user to visually interact with a software through graphical objects such as icons, text, buttons, and so on. Internally, a GUI handles some events and binds them to functions, mostly called callbacks. These functions define the reaction of the program.

There are a lot of different common objects that are always present in a GUI, such as buttons, windows, labels, and layouts. I don't think I need to explain to you what a button, window, or label is, but I will explain to you in short what a layout is.

A layout is an invisible object that manages the arrangements of the graphical objects on the screen. Simply put, its goal is to take care of the size and the position of the objects by managing a part of them. It's like a table that makes sure none of these objects are on top of the others, and which adapts their size to fill the screen as proportionately as possible.

## Creating a GUI from scratch

Now that the GUI terms have been introduced, we will think about how to build it one by one using SFML. This GUI will be added to the Gravitris project, and the result will be similar to the following two screenshots:

![Creating a GUI from scratch](img/8477OS_05_01.jpg)

These show you the starting menu of the game and the pause menu during the game.

To build this GUI, only four different objects have been used: `TextButton`, `Label`, `Frame`, and `VLayout`. We will now see how to structure our code to be as flexible as possible to be able to extend this GUI in future if needed.

## Class hierarchy

As already said, we will need to build different components for the GUI. Each one has its own characteristics and features that can be slightly different from the others. Following are some characteristics of these components:

*   `TextButton`: This class will represent a button that can trigger an "on click" event when clicked on. Graphically, it's a box with text inside it.
*   `Label`: This accepts simple text that can be displayed on the screen.
*   `Frame`: This class is an invisible container that will contain some object through a layout. This object will also be attached to an SFML window and will fill the entire window. This class can also process events (like catching the resize of the window, the click of the *Esc* key, and so on).
*   `Vlayout`: This class's functionality has already been explained–it displays objects vertically. This class has to be able to adjust the positions of all the objects attached to it.

Because we want to build a GUI reusable and it needs to be as flexible as possible, we need to think bigger than our 4 classes to build it. For example, we should be able to easily add a container, switch to a horizontal layout or grid layout, make use of sprite buttons and so on. Basically, we need a hierarchy that allows the addition of new components easily. Here is a possible solution:

![Class hierarchy](img/8477OS_05_02.jpg)

### Note

In this hierarchy, each green box represents an external class of the GUI.

In the GUI system, each component is a `Widget`. This class is the base of all the other components and defines the common methods to interact with them. We also define some virtual classes, such as `Button`, `Container`, and `Layout`. Each of these classes adapts the `Widget` class and adds the possibility of growing our system without too much effort. For example, adding an `HLayout` class will be made possible by extending it from `Layout`. Other examples include some specific buttons such as `RadioButton` and `CheckBox`, which use the `Button` class.

In this hierarchy, the `Frame` class extends the `ActionTarget` class. The idea is to be able to use the bind methods of `ActionTarget` to catch some events such as when working in some window and the *Esc* key is pressed.

Now that the hierarchy has been shown to you, we will continue with the implementation of the different classes. Let's start from the base: the `Widget` class.

### The Widget class

As already explained, this class is the common trunk of all the other GUI components. It provides some common methods with default behaviors that can be customized or improved on. A `Widget` class not only has a position and can be moved, but also has the ability to be displayed on screen. Take a look at its header source:

[PRE0]

This first class is simple. We define a construct and a virtual destructor. The virtual destructor is very important because of the polymorphism usage inside the GUI logic. Then we define some getters and setters on the internal variables. A widget can also be attached to another one that is contained in it so we keep a reference to it for updating purposes. Now take a look at the implementation for a better understanding:

[PRE1]

Up to this point, nothing should surprise you. We only defined some getters/setters and coded the default behavior for event handling.

Now have a look at the following function:

[PRE2]

This function, unlike the others we saw, is important. Its goal is to propagate the update request through the GUI tree. For example, from a button with a change in its size due to a text change, to its layout, to the container. By doing this, we are sure that each component will be updated without further efforts.

### The Label class

Now that the `Widget` class has been introduced, let's build our first widget, a label. This is the simplest widget that we can build. So we will learn the logic of GUI through it. The result will be as follows:

![The Label class](img/8477OS_05_03.jpg)

For doing this we will run the following code:

[PRE3]

As you can see this class is nothing other than a box around `sf::Text`. It defines some methods taken from the `sf::Text` API with the exact same behavior. It also implements the requirements of `Widget` class such as the `getSize()` and `draw()` methods. Now let's have a look at the implementation:

[PRE4]

The constructor initializes the text from a parameter, sets the default font taken from the `Configuration` class, and sets a color.

[PRE5]

These two functions forward their jobs to `sf::Text` and request for an update because of the possible change of size.

[PRE6]

SFML already provides a function to get the size of a `sf::Text` parameter, so we use it and convert the result into the excepted one as shown by the following code snippet:

[PRE7]

This function is simple, but we need to understand it. Each widget has its own position, but is relative to the parent. So when we display the object, we need to update the `sf::RenderStates` parameter by translating the transform matrix by the relative position, and then draw all the stuff needed. It's simple, but important.

### The Button class

Now, we will build another `Widget` class that is very useful: the `Button` class. This class will be a virtual one because we want to be able to build several button classes. But there are common functions shared by all the button classes, such as the "on click" event. So, the goal of this class is to group them. Take a look to the header of this class:

[PRE8]

As usual, we declare the constructor and the destructor. We also declare an `onClick` attribute, which is an `std::function` that will be triggered when the button is pushed. This is our callback. The callback type is kept as `typedef` and we also declare a default empty function for convenience. Now, take a look at the implementation:

[PRE9]

With the help of the following code snippet, we declare an empty function that will be used as the default for the `onClick` attribute. This function does nothing:

[PRE10]

We build the constructor that forwards its parameter to its parent class and also sets the `onClick` value to the default empty function previously defined to avoid undefined performance when the callback is not initialized by the user, as shown in the following code snippet:

[PRE11]

This function is the heart of our class. It manages the events by triggering some callbacks when some criteria are satisfied. Let's take a look at it step by step:

1.  If the event received as the parameter is a click, we have to check whether it happens in the button area. If so, we trigger our `onClick` function.
2.  On the other hand, if the event is caused by moving the pointer, we verify if the mouse pointer is hovering over the button. If so, we set the status value to `Hover`, and here is the trick:
3.  If this flag was newly defined to `Hover`, then we call the `onMouseEntered()` method, which can be customized.
4.  If the flag was previously defined to `Hover` but is not set to it anymore, it's because the mouse left the area of the button, so we call another method: `onMouseLeft()`.

### Note

The value returned by the `processEvent()` method will stop the propagation of the event on the GUI if it's set to `true`. Returning false will continue the propagation of the event, so it's also possible to use an event without stopping its propagation; on the mouse moving away, for example. But in this case, we simply can't click on multiple widget objects at the same time, so we stop if needed.

I hope the logic of the `processEvent()` function is clear, because our GUI logic is based on it.

Following two functions are the default empty behavior of the button with a mouse move event. Of course, we will customize them in the specialized `Button` classes:

[PRE12]

### The TextButton class

This class will extend our previously defined `Button` class. The result will be a rectangle on the screen with text inside it, just as shown in the following screenshot:

![The TextButton class](img/8477OS_05_04.jpg)

Now take a look at the implementation. Remember that our `Button` class extends from `sf::Drawable`:

[PRE13]

This class extends the `Button` class and adds a rectangle shape and a label to it. It also implements the `onMouseEntered()` and `onMouseLeft()` functions. These two functions will change the color of the button, making them a bit lighter:

[PRE14]

The constructor initializes the different colors and the initial text:

[PRE15]

All these functions set the different attributes by forwarding the job. It also calls the `updateShape()` method to update the container:

[PRE16]

The following function updates the shape by resizing it using the size from the internal label and adding some padding to it:

[PRE17]

This method has the same logic as Label. It moves `sf::RenderStates` to the position of the button and draws all the different `sf::Drawable` parameters:

[PRE18]

These two functions change the color of the button when the cursor is hovering over it and reset the initial color when the cursor leaves it. This is useful for the user, because he knows which button will be clicked easily.

As you can see, implementation of a `TextButton` is pretty short, all thanks to the changes made in the parent classes, `Button` and `Widget`.

### The Container class

This class is another type of `Widget` and will be abstract. A `Container` class is a `Widget` class that will store other widgets through a `Layout` class. The purpose of this class is to group all the common operations between the different possible `Container` classes, even as in our case, we only implement a `Frame` container.

[PRE19]

As usual, we define the constructor and destructor. We also add accessors to the internal `Layout` class. We will also implement the `draw()` method and the event processing. Now take a look at the implementation in the following code snippet:

[PRE20]

The destructor deletes the internal `Layout` class, but only if the parent of the `Layout` class is the current container. This avoids double free corruption and respects the RAII idiom:

[PRE21]

The previous function sets the layout of the container and deletes it from the memory if needed. Then it takes ownership of the new layout and updates the internal pointer to it.

[PRE22]

The three previous functions do the usual job, just as with the other `Widgets`:

[PRE23]

These two previous functions process for the events. Because a `Layout` class doesn't have any event to deal with, it forwards the job to all the internal `Widget` classes. If an event is processed by a `Widget` class, we stop the propagation, because logically no other widget should be able to deal with it.

### The Frame class

Now that the basic container has been constructed, let's extend it with a special one. The following `Widget` class will be attached to `sf::RenderWindow` and will be the main widget. It will manage the render target and the events by itself. Take a look at its header:

[PRE24]

As you can see, this class is a bit more complex than the previous `Widget`. It extends the `Container` class to be able to attach a `Layout` class to it. Moreover, it also extends the `ActionTarget` class, but as protected. This is an important point. In fact, we want to allow the user to bind/unbind events, but we don't want to allow them to cast the `Frame` to an `ActionTarget`, so we hide it to the user and rewrite all the methods of the `ActionTarget` class. This is why there is a protected keyword.

The class will also be able to extract events from its parent windows; this explains why we need to keep a reference to it, as seen here:

[PRE25]

All these methods are simple and don't require a lot of explanation. You simply initialize all the attributes with the constructor and forward the job to the attributes stored inside the class for the others, as done here:

[PRE26]

These two overload functions are exposed to the user. It forwards the job to the override functions inherited from `Widget` by constructing the missing ones or the already known arguments.

[PRE27]

On the other hand, these two functions process to the event management of the `ActionTarget` and `Container` bases of the class, but also take in charge the polling event from its parent window. In this case, all event management will be automatic.

The `Frame` class is now over. As you can see, it's not a complex task, thanks to our hierarchical tree and because we reused code here.

### The Layout class

Now that all the widgets that will be rendered on the screen are building, let's build the class that will be in charge of their arrangement:

[PRE28]

As you can see, the abstract class is very simple. The only new feature is the ability to set spacing. We don't have any `add(Widget*)` method, for example. The reason is that the argument will be slightly different depending on the kind of `Layout` used. For example, we just need a `Widget` class as argument for the layout with a single column or line, but the situation is completely different for a grid. We need two other integers that represent the cell in which the widget can be placed. So, no common API is designed here. As you will see, the implementation of this class is also very simple and doesn't require any explanation. It follows the logic of the `Widget` class we previously created.

[PRE29]

### The VLayout class

This `Layout` class will be more complex than the previous ones. This one contains the full implementation of a vertical layout, which automatically adjusts its size and the alignment of all its internal objects:

[PRE30]

The class will implement all the requirements from the widget and will also add the features to add widgets in it. So there are some functions to implement. To keep a trace of the widgets attached to the `Layout` class, we will internally store them in a container. The choice of the `std::vector` class makes sense here because of the random access of the elements for the `at()` method and the great number access through the container. So the only reason for the choice is performance, since an `std::list` will also be able to do the same job. Now, let's have a look at the implementation:

[PRE31]

The destructor will free the memory from the objects attached to the `Layout` class, with the same criteria as the ones explained in the `Container` class:

[PRE32]

These two previous functions add the possibility to add and get access to the widget stored by the class instance. The `add()` method additionally takes ownership of the added object:

[PRE33]

This method calculates the total size of the layout, taking into account the spacing. Because our class will display all the objects in a single column, the height will be their total size and the width the maximal of all the objects. The spacing has to be taken into account each time.

[PRE34]

These two previous methods forward the job to all the stored widget , but we stop the propagation when it's needed.

[PRE35]

This method is the most important for this class. It resets the different positions of all the objects by calculating it based on all the other widgets. The final result will be a column of widgets centered vertically and horizontally.

[PRE36]

This last function asks each `Widget` to render itself by forwarding the parameter. This time, we don't need to translate states because the position of the layout is the same as its parent.

The entire class has now been built and explained. It's now time for the user to use them and add a menu to our game.

# Adding a menu to the game

Now that we have all the pieces in place to build a basic menu, let's do it with our fresh GUI. We will build two of them. The main, game-opening one and the pause menu. This will show you the different usage possibilities of our actual GUI.

If you have understood what we have done until now well, you would have noticed that the base component of our GUI is `Frame`. All the other widgets will be displayed on the top of it. Here is a schema that summarizes the GUI tree hierarchy:

![Adding a menu to the game](img/8477OS_05_05.jpg)

Each color represents a different type of component. The trunk is **sf::RenderWindow** and then we have a **Frame** attached to it with its **Layout**. And finally we have some different **Widget**. Now that the usage has been explained, let's create our main menu.

## Building the main menu

To build the main menu, we will need to add an attribute to the `Game` class. Let's call it `_mainMenu`.

[PRE37]

We then create an `enum` function with different possibilities of values in order to know the currently displayed status:

[PRE38]

Now let's create a function to initialize the menu:

[PRE39]

This function will store the entire GUI construction, except from the constructor that is calling. Now that we have all that we need in the header file, let's move on to the implementation of all this stuff.

First of all, we need to update the constructor by adding in the initialization of `_mainMenu` and `_status`. It should look like this:

[PRE40]

Now we need to implement the `initGui()` function as follows:

[PRE41]

Let's discuss this function step by step:

1.  We create a `Vlayout` class and set its spacing.
2.  We create a button with `New Game` as its label.
3.  We set the `onClick` callback function that initializes the game.
4.  We add the button to the layout.
5.  With the same logic, we create two other buttons with different callbacks.
6.  Then we set the layout to the `_mainMenu` parameter.
7.  And we finally add an event directly to the frame that will handle the *Esc* key. This key is defined in the `GuiInputs enum` contained in the `Configuration` class, which was constructed as `PlayerInputs`.

Now that our menu is created, we need to make some little changes in the existing `run()`, `processEvents()`, and `render()` methods. Let's start with `run()`. The modification is negligible. In fact, we just have to add a condition for the call of the update methods, adding verification on the `_status` variable. The new line is now as follows:

[PRE42]

The next function is `processEvents()`, which will require a little more modification, but not too much. In fact, we need to call `_mainMenu::processEvent(const f::Event&)` and `_mainMenu::processEvents()`, but only when the game is in `StatusMainMenu` mode. The new method is now as follows:

[PRE43]

As you can see, the modification is not too complicated, and easily understandable.

And now, the last change in the `render()` method. The logic is the same, a switch on the `_status` value.

[PRE44]

As you can see, we have been able to add a menu to our game without too much effort. The result should be like the figure shown here:

![Building the main menu](img/8477OS_05_06.jpg)

Now, let's build the second menu.

## Building the pause menu

The pause menu will be constructed just like the previous one, so I will skip the constructor part and directly move on to the `initGui()` function:

[PRE45]

The logic is exactly the same as the one used for the previous menu, but here we use a `Label` and a `TextButton` class. The callback of the button will also change the `_status` value. Here, again, we catch the *Esc* key. The result is to leave this menu. In the `processEvents()`, we only need to add one line to the first switch:

[PRE46]

And add another line to the second switch:

[PRE47]

And that's it. We are done with this function.

The next step is the `render()` function. Here again it will be very quick. We add a case in the switch statement as follows:

[PRE48]

The request to draw `_world` means to set the current game state in the background on the menu. This is useless, but pretty cool, so why not?

The final result is the second screenshot shown at the beginning of this chapter. Have a look at what appears on my screen:

![Building the pause menu](img/8477OS_05_07.jpg)

## Building the configuration menu

This menu will in fact be implemented in the second part (by using SFGUI), but we need a way to exit the configuration menu. So we simply have to create a `_configurationMenu` as the two others and bind the `Escape` event to set the status to the main menu. The code in the `initGui()` to add is shown as follows:

[PRE49]

I'm sure you are now able to update the `processEvents()` and `render()` functions by yourself using your new skills.

That's all concerning our home-made GUI. Of course, you can improve it as you wish. That's one of its advantages.

### Tip

If you are interested in making improvements, take a look to the external library made regrouping all our custom game framework at [http://github.com/Krozark/SFML-utils/](http://github.com/Krozark/SFML-utils/).

The next step is to use an already made GUI with more complex widgets. But keep in mind that if you only need to show menus like those presented here, this GUI is enough.

# Using SFGUI

SFGUI is an open source library that implements a complete GUI system based on the top of SFML. Its goal is to provide a rich set of widgets and to be easily customizable and extensible. It also uses modern C++, so it's easy to use in any SFML project without too much effort.

The following screenshot shows the SFGUI in action with the test example provided with the source:

![Using SFGUI](img/8477OS_05_08.jpg)

## Installing SFGUI

The first step is to download the source code. You will find it on the official website of the library: [http://sfgui.sfml-dev.de/](http://sfgui.sfml-dev.de/). The current version is 0.2.3 (Feb 20, 2014). You will need to build SFGUI by yourself, but as usual, it comes with the `cmake` file to help with the build. That is perfect, because we already know how to use it.

Sometimes, you could have a problem like the one shown in the following screenshot during the build step:

![Installing SFGUI](img/8477OS_05_09.jpg)

In this case, you have to set the `CMAKE_MODULE_PATH` variable to `/path/to/SFML/cmake/Modules` using the `add entry` parameter. This should fix the problem.

### Note

For other similar problems, take a look at this page: [http://sfgui.sfml-dev.de/p/faq#findsfml](http://sfgui.sfml-dev.de/p/faq#findsfml). It should be helpful.

Now that SFGUI is configured, you need to build it and finally install it exactly as SFML and Box2D. You should now be pretty familiar with this.

## Using the features of SFGUI

I will not go too deep into the usage of SFGUI in this book. The goal is to show you that you don't always need to reinvent the wheel when a good one already exists.

SFGUI use a lot of C++11 features, such as `shared_pointers`, `std::functions`, and some others that have already been covered in this book, and uses the RAII idiom as well. As you already know how to work with these features, you will not be lost when it comes to using SFGUI optimally.

First of all, to use SFGUI objects, you must instantiate one object before all the others: `sfg::SFGUI`. This class holds all the information needed for the rendering. Except from this point, the library can be used pretty much like ours. So let's try it.

## Building the starting level

We will add a menu to our game that will allow us to choose the starting level. The goal of this section is to add a simple form that takes a number as parameter and sets it as the starting level of the game. The final result will look like this:

![Building the starting level](img/8477OS_05_10.jpg)

Before starting with SFGUI, we need to make an update to our `Stats` class. In fact, this class doesn't allow us to start at a specific level, so we need to add that functionality. This will be done by adding a new attribute to it as follows:

[PRE50]

We will also need a new method:

[PRE51]

That's it for the header. Now we need to initialize `_initialLvl` to `0` by default. And then change the calculation of the current level in the `addLines()` function. To do this, go to the following line:

[PRE52]

Change the preceding line to the following:

[PRE53]

And finally, we will need to update or implement the assessors on the current level as follows:

[PRE54]

And that's it for the update on this class. Now let's go back to SFGUI.

We will use only three different visual objects to build the needed form: label, text input, and button. But we will also use a layout and a desktop, which is the equivalent of our `Frame` class. All the initialization will be done in the `initGui()` function, just as before.

We also need to add two new attributes to our game:

[PRE55]

The reason for adding `_sfgui` was previously explained. We add `_sfDesktop` for the exact same reason we add `Frame` to contain the objects.

Now take a look at the code needed to create the form:

[PRE56]

Okay, a lot of new features here, so I will explain them step by step:

1.  First of all, we create the different components needed for this form.
2.  Then we set the callback of the button on a press event. This callback does a lot of things:

    *   We get back the text entered by the user
    *   We convert this text to an integer using `std::stringstream`
    *   We check the validity of the input
    *   If the input is not valid, we display an error message
    *   On the other hand, if it is valid, we reset the game, set the starting level, and start the game

3.  Until all the objects are created, we add them into a layout one by one.
4.  We change the size of the layout and center it on the window.
5.  Finally, we attach the layout to the desktop.

As all the object are created and stored into `std::shared_` we don't need to keep a trace of them. SFGUI does it for us.

Now that the form is created, we have the same challenges as with our GUI: events and rendering. Good news, the logic is the same! However, we do have to code the `processEvents()` and `render()` functions again.

In the `processEvents()` method, we only need to complete the first switch as shown in the following code snippet:

[PRE57]

As you can see, the logic is the same as our GUI, so the reasoning is clear.

And finally, the rendering. Here, again, the switch has to be completed by using the following code snippet:

[PRE58]

The new thing is the `Update()` call. This is for animations. Since in our case, we don't have any animation, we can put `0` as the parameter. It would be good practice to add this in the `Game::update()` function, but it's okay for our needs–and it also avoids changes.

You should now be able to use this new form in the configuration menu.

Of course, in this example, I have just shown you a little piece of SFGUI. It packs in many more features, and if you are interested, I would suggest you to take a look at the documentation and the examples given with the library. It's very interesting.

# Summary

Congratulations, you have now finished this chapter and have gained the ability to communicate with your player in a good way. You are now able to create some buttons, use labels, and add callbacks to some event triggers set off by the user. You also know the basics to create your own GUI and how to use SFGUI.

In the next chapter, we will learn how to use the full power of the CPU by using more than one thread, and see its implications in game programming.