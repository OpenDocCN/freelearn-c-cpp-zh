# Chapter 2. Hello, Small World!

This chapter introduces Small Windows by presenting the following two small applications:

*   The first application writes "Hello, Small Windows!" in a window
*   The second application handles circles of different colors in a document window

# Hello, Small Windows!

In *The C Programming Language* by Brian Kernighan and Dennis Richie, the hello-world example was introduced. It was a small program that wrote "hello, world" on the screen. In this section, we shall write a similar program for Small Windows.

In regular C++, the execution of the application starts with the `main` function. In Small Windows, however, `main` is hidden in the framework and has been replaced by `MainWindow`, whose task is to define the application name and create the main window object. The following `argumentList` parameter corresponds to `argc` and `argv` in main. The `commandShow` parameter forwards the system's request regarding the window's appearance:

**MainWindow.cpp**

[PRE0]

In C++, there are to two character types, `char` and `wchar_t`, where `char` holds a regular character of 1 byte and `wchar_t` holds a wide character of larger size, usually 2 bytes. There is also the `string` class, which holds a string of `char` values, and the `wstring` class, which holds a string of `wchar_t` values.

However, in Windows, there is also the generic character type `TCHAR`, which is `char` or `wchar_t`, depending on system settings. There is also the `String` class, which holds a string of `TCHAR` values. Moreover, `TEXT` is a macro that translates a character value to `TCHAR` and a text value to an array of `TCHAR` values.

To sum it up, the following table shows character types and string classes:

| **Regular character** | **Wide character** | **Generic character** |
| char | wchar_t | TCHAR |
| string | wstring | String |

In the applications of this book, we always use the `TCHAR` type, the `String` class, and the `TEXT` macro. The only exception to that rule is clipboard handling in [Chapter 13](ch13.html "Chapter 13. The Registry, Clipboard, Standard Dialogs, and Print Preview"), *The Registry, Clipboard, Standard Dialogs, and Print Preview*.

Our version of the hello-world program writes "Hello, Small Windows!" in the center of the client area. The client area of the window is that part of the window where it is possible to draw graphical objects. In the following window, the client area is the white area:

![Hello, Small Windows!](img/image_02_001.jpg)

The `HelloWindow` class extends the Small Windows `Window` class. It holds a constructor and the `Draw` method. The constructor calls the `Window` constructor with suitable information regarding the appearance of the window. The `Draw` method is called every time the client area of the window needs to be redrawn:

**HelloWindow.h**

[PRE1]

The constructor of `HelloWindow` calls the constructor of `Window` with the following parameters:

*   The first parameter of the `HelloWindow` constructor is the coordinate system. `LogicalWithScroll` indicates that each logical unit is one hundredth of a millimeter, regardless of the physical resolution of the screen. The current scroll bar settings are taken into consideration.
*   The second parameter of the `Window` constructor is the preferred size of the window. It indicates that a default size should be used.
*   The third parameter is a pointer to the parent window. It is null since the window has no parent window.
*   The fourth and fifth parameters set the window's style, in this case overlapped windows.
*   The last parameter is `windowShow`, given by the surrounding system to `MainWindow`, which decides the window's initial appearance (minimized, normal, or maximized).
*   Finally, the constructor sets the header of the window by calling the `Window` class's `SetHeader` method.

**HelloWindow.cpp**

[PRE2]

The `OnDraw` method is called every time the client area of the window needs to be redrawn. It obtains the size of the client area and draws the text in its center with black text on a white background. The `SystemFont` parameter will make the text appear in the default system font.

The Small Windows `Color` class holds the constants `Black` and `White`. The `Point` class holds a two-dimensional point. The `Size` class holds `width` and `height`. The `Rect` class holds a rectangle; more specifically, it holds the four corners of a rectangle:

[PRE3]

# The circle application

In this section, we look into a simple circle application. As the name implies, it enables the user to handle circles in a graphical application. The user can add a new circle by pressing the left mouse button. The user can also move an existing circle by dragging it. Moreover, the user can change the color of a circle as well as save and open the document:

![The circle application](img/image_02_002.jpg)

## The main window

As we will see throughout this book, the `MainWindow` function always does the same thing: it sets the application name and creates the main window of the application. The name is used by the **Save** and **Open** standard dialogs, the **About** menu item, and the registry.

The difference between the main window and other windows of the application is that, when the user closes the main window, the application exits. Moreover, when the user selects the **Exit** menu item, the main window is closed, and its destructor is called:

**MainWindow.cpp**

[PRE4]

## The CircleDocument class

The `CircleDocument` class extends the Small Windows `StandardDocument` class, which, in turn, extends the `Document` and `Window` classes. In fact, the `StandardDocument` class constitutes a framework, that is, a base class with a set of virtual methods with functionality that we can override and further specify.

The `OnMouseDown` and `OnMouseUp` methods are overridden from the `Window` class and are called when the user presses or releases one of the mouse buttons. The `OnMouseMove` method is called when the user moves the mouse. The `OnDraw` method is also overridden from the `Window` class and is called every time the window needs to be redrawn.

The `ClearDocument`, `ReadDocumentFromStream`, and `WriteDocumentToStream` methods are overridden from the `Standard­Document` class and are called when the user creates a new file, opens a file, or saves a file:

**CircleDocument.h**

[PRE5]

The `DEFINE_BOOL_LISTENER` and `DEFINE_VOID_LISTENER` macros define **listeners** which are methods without parameters that are called when the user selects a menu item. The only difference between the macros is the return type of the defined methods: `bool` or `void`.

In the applications of this book, we use the common standard whereby listeners called in response to user actions are prefixed with `On`, for instance, `OnRed`, as shown in the following code snippet. The methods that decide whether the menu item should be enabled are suffixed with `Enable`, and the methods that decide whether the menu item should be marked with a check mark or a radio button are suffixed with `Check` or `Radio`.

In the following application, we define menu items for the red, green, and blue colors. We also define a menu item for the color standard dialog:

[PRE6]

When the user has chosen one of the colors, red, green, or blue, its corresponding menu item is checked with a radio button. The `RedRadio`, `GreenRadio`, and `BlueRadio` parameters are called before the menu items become visible and return a Boolean value indicating whether the menu item should be marked with a radio button:

[PRE7]

The circle radius is always 500 units, which corresponds to 5 mm:

[PRE8]

The `circleList` field holds the circles, where the topmost circle is located at the beginning of the list. The `nextColor` field holds the color of the next circle to be added by the user. It is initialized to minus 0ne to indicate that no circle is being moved at the beginning. The `moveIndex` and `movePoint` fields are used by the `OnMouseDown` and `OnMouseMove` methods to keep track of the circle being moved by the user:

[PRE9]

In the `StandardDocument` constructor call, the first two parameters are `LogicalWithScroll` and `USLetterPortrait`. They indicate that the logical size is hundredths of millimeters and that the client area holds the logical size of a US letter: *215.9*279.4 millimeters (8.5*11 inches)*. If the window is resized so that the client area becomes smaller than a US letter, scroll bars are added to the window.

The third parameter sets the file information used by the standard save and open dialogs; the text description is set to `Circle Files` and the file suffix is set to `cle`. The `nullptr` parameter indicates that the window does not have a parent window. The `OverlappedWindow` constant parameter indicates that the window should overlap other windows, and the `windowShow` parameter is the window's initial appearance passed on from the surrounding system by the `MainWindow` class:

**CircleDocument.cpp**

[PRE10]

The `StandardDocument` class adds the standard **File**, **Edit**, and **Help** menus to the window menu bar. The **File** menu holds the **New**, **Open**, **Save**, **Save As**, **Page Setup**, **Print Preview**, and **Exit** items. **Page Setup** and **Print Preview** are optional. The seventh parameter of the `StandardDocument` constructor (the default value is `false`) indicates their presence. The **Edit** menu holds the **Cut**, **Copy**, **Paste**, and **Delete** items. They are disabled by default; we will not use them in this application. The **Help** menu holds the **About** item, and the application name set in `MainWindow` is used to display a message box with a standard message **Circle, version 1.0**.

We add the standard **File** and **Edit** menus to the menu bar. Then we add the **Color** menu, which is the application-specific menu of this application. Finally, we add the standard **Help** menu and set the menu bar of the document.

The **Color** menu holds the menu items used to set the circle colors. The `OnRed`, `OnGreen`, and `OnBlue` methods are called when the user selects the menu item, and the `RedRadio`, `GreenRadio`, and `BlueRadio` methods are called before the user selects the **Color** menu in order to decide if the items should be marked with a radio button. The `OnColorDialog` method opens a standard color dialog.

In the `&Red\tCtrl+R` text in the following code snippet, the **ampersand** (**&**) indicates that the menu item has a **mnemonic**; that is, the letter R will be underlined and it is possible to select the menu item by pressing **R** after the menu has been opened. The **tabulator character** (**\t**) indicates that the second part of the text defines an **accelerator**; that is, the text `Ctrl+R` will occur right-justified in the menu item and the item can be selected by pressing Ctrl+R:

[PRE11]

The `false` parameter to `StandardFileMenu` indicates that we do not want to include the file menu items.

[PRE12]

The `AddItem` method in the `Menu` class also takes two more parameters for enabling the menu item and setting a checkbox. However, we do not use them in this application. Therefore, we send null pointers:

[PRE13]

Finally, we read the current color (the color of the next circle to be added) from the registry; red is the default color in case there is no color stored in the registry:

[PRE14]

The destructor saves the current color in the registry. In this application, we do not need to perform the destructor's normal tasks such as deallocating memory or closing files:

[PRE15]

The `ClearDocument` method is called when the user selects the **New** menu item. In this case, we just clear the circle list. Every other action, such as redrawing the window or changing its title, is taken care of by the `StandardDocument` class:

[PRE16]

The `WriteDocumentToStream` method is called by the `StandardDocument` class when the user saves a file (by selecting **Save** or **Save As**). It writes the number of circles (the size of the circle list) to the output stream and calls the `WriteCircle` method for each circle in order to write their states to the stream:

[PRE17]

The `ReadDocumentFromStream` method is called by the `StandardDocument` method when the user opens a file by selecting the **Open** menu item. It reads the number of circles (the size of the circle list) and for each circle it creates a new object of the `Circle` class, calls the `ReadCircle` method in order to read the state of the circle, and adds the circle object to the `circleList` method:

[PRE18]

The `OnMouseDown` method is called when the user presses one of the mouse buttons. First we need to check that they have pressed the left mouse button. If they have, we loop through the circle list and call the `IsClick` method for each circle in order to decide whether they have clicked on a circle. Note that the topmost circle is located at the beginning of the list; therefore, we loop from the beginning of the list. If we find a clicked circle, we break the loop.

If the user has clicked on a circle, we store its index `moveIndex` and the current mouse position in `movePoint`. Both values are needed by that `OnMouseMove` method that will be called when the user moves the mouse:

[PRE19]

However, if the user has not clicked on a circle, we add a new circle. A circle is defined by its center position (`mousePoint`), radius (`CircleRadius`), and color (`nextColor`).

An invalidated area is a part of the client area that needs to be redrawn. Remember that in Windows, we normally do not draw figures directly. Instead, we call the `Invalidate` method to tell the system that an area needs to be redrawn and force the actual redrawing by calling the `UpdateWindow` method, which eventually results in a call to the `OnDraw` method. The invalidated area is always a rectangle. The `Invalidate` method has a second parameter (the default value is `true`) indicating that the invalidated area should be cleared.

Technically, it is painted in the window's client color, which in this case is white. In this way, the previous location of the circle is cleared and the circle is drawn at its new location.

The `SetDirty` method tells the framework that the document has been altered (the document has become *dirty*), which causes the **Save** menu item to be enabled and the user to be warned if he/she tries to close the window without saving it:

[PRE20]

The `OnMouseMove` method is called every time the user moves the mouse with at least one mouse button pressed. We first need to check whether the user is pressing the left mouse button and is clicking on a circle (whether the `moveIndex` method does not equal `-1`). If the user is, we calculate the distance from the previous mouse event (`OnMouseDown` or `OnMouseMove`) by comparing the previous and the current mouse position using the `mousePoint` method. We update the circle position, invalidate both the old and new area, forcing a redrawing of the invalidated areas with the `UpdateWindow` method, and set the dirty flag:

[PRE21]

Strictly speaking, the `OnMouseUp` method could be excluded since the `moveIndex` method is set to minus one in the `OnMouseDown` method, which is always called before the `OnMouseMove` method. However, it has been included for the sake of completeness:

[PRE22]

The `OnDraw` method is called every time the window needs to be (partly or completely) redrawn. The call can be initialized by the system as a response to an event (for instance, the window has been resized) or by an earlier call to the `UpdateWindow` method. The `Graphics` reference parameter has been created by the framework and can be considered as a toolbox for drawing lines, painting areas, and writing text. However, in this application, we do not write text.

We iterate through the circle list and, for each circle, call the `Draw` method. Note that we do not care about which circles are to be physically redrawn. We simple redraw all circles. However, only the circles located in an area that has been invalidated by a previous call to the `Invalidate` method will be physically redrawn.

The `Draw` method has a second parameter indicating the draw mode, which can be `Paint` or `Print`. The `Paint` method indicates that the `OnDraw` method is called by the `OnPaint` method in the `Window` class and that the painting is performed in the window's client area. The `Print` method indicates that the `OnDraw` method is called by the `OnPrint` method and that the painting is sent to a printer. However, in this application, we do not use that parameter:

[PRE23]

The `RedRadio`, `GreenRadio`, and `BlueRadio` methods are called before the menu items are shown, and the items will be marked with a radio button if they return `true`. The `Red`, `Green`, and `Blue` constants are defined in the `Color` class:

[PRE24]

The `OnRed`, `OnGreen`, and `OnBlue` methods are called when the user selects the corresponding menu item. They all set the `nextColor` field to an appropriate value:

[PRE25]

The `OnColorDialog` method is called when the user selects the **Color** dialog menu item and displays the standard color dialog. If the user chooses a new color, the `nextcolor` method will be given the chosen color value:

[PRE26]

## The Circle class

`Circle` is a class holding the information about a single circle. The default constructor is used when reading a circle from a file. The second constructor is used when creating a new circle. The `IsClick` method returns `true` if the given point is located inside the circle (to check whether the user has clicked in the circle), the `Area` method returns the circle's surrounding rectangle (for invalidation), and the `Draw` method is called to redraw the circle:

**Circle.h**

[PRE27]

As mentioned in the previous section, a circle is defined by its center position (`center`), radius (`radius`), and color (`color`):

[PRE28]

The default constructor does not need to initialize the fields since it is called when the user opens a file and the values are read from the file. The second constructor, however, initializes the center point, radius, and color of the circle:

**Circle.cpp**

[PRE29]

The `WriteCircle` method writes the color, center point, and radius to the stream. Since `radius` is a regular integer, we simply use the C standard function `write`, while `Color` and `Point` have their own methods to write their values to a stream. In the `ReadCircle` method, we read the color, center point, and radius from the stream in a similar manner:

[PRE30]

The `IsClick` method uses Pythagoras' theorem to calculate the distance between the given point and the circle's center point and returns `true` if the point is located inside the circle (if the distance is less than or equal to the circle radius):

![The Circle class](img/image_02_003.jpg)

[PRE31]

The top-left corner of the resulting rectangle is the center point minus the radius and the bottom-right corner is the center point plus the radius:

[PRE32]

We use the `FillEllipse` method (there is no `FillCircle` method) of the Small Windows `Graphics` class to draw the circle. The circle's border is always black, while its interior color is given by the `color` field:

[PRE33]

# Summary

In this chapter, you looked into two applications in Small Windows: a simple hello-world application and a slightly more advanced circle application, which introduced the framework. You also looked into menus, circle drawing, and mouse handling.

In [Chapter 3](ch03.html "Chapter 3. Building a Tetris Application"), *Building a Tetris Application*, we will develop a classic Tetris game.