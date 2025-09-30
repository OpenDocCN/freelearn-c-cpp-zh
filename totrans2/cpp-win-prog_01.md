# Chapter 1. Introduction

The purpose of this book is to learn how to develop applications in Windows. In order to do so, I have developed Small Windows, which is a C++ object-oriented class library for graphical applications in Windows.

The idea is to guide you into Windows programming by introducing increasingly more advanced applications written in C++ with Small Windows, thereby hiding the technical details of the **Windows 32-bit Applications Programming Interface** (**Win32 API**), which is the underlying library for Windows development. With this approach, we can focus on the business logic without struggling with the underlying technical details. If you are interested in knowing how the Win32 API works, the second part of this book gives a detailed description of how Small Windows is implemented.

This book is made up of two parts, where the first part describes the applications developed in C++ with Small Windows. While some books have many examples, this book only includes six examples, among which the last four are rather advanced: the Tetris game, a drawing program, a word processor, and a spreadsheet program. Note that this book is not only a tutorial about Windows programming, but also a tutorial about how to develop object-oriented graphical applications.

The second part holds a detailed description of the implementation of Small Windows in the Win32 API. Note that the Win32 API is not introduced until the second part. Some of you may be satisfied with the high level aspects of Small Windows and only want to study application-specific problems, while others may want to read the second part in order to understand how the classes, methods, and macros of Small Windows are implemented in the Win32 API.

Naturally, I am aware of the existence of modern object-oriented class libraries for Windows. However, the purpose of those libraries is to make it easier for the developer by hiding the details of the architecture, which also prevents the developer from using the Windows architecture to its full extent. Even though the Win32 API has been around for a while, I regard it as the best way to develop professional Windows applications and to understand the Windows architecture.

All source code is given in this book; it is also available as a Visual Studio solution.

# The library

This section gives an introduction to Small Windows. The first part of a Small Windows application is the `MainWindow` function. It corresponds to `main` in regular C++. Its task is to set the name of the application and create the main window of the application.

In this book we talk about **definitions** and **declarations**. A declaration is just a notification for the compiler, while a definition is what defines the feature. Below is the declaration of the `MainWindow` function. Its definition is left to the user of Small Windows.

[PRE0]

Simply put, in Windows the application does not take any initiative; rather, it waits for messages and reacts when it receives them. Informally speaking, *you do not call Windows, Windows calls you*.

The most central part of Small Windows is the `Application` class. In Windows, each event generates a message that is sent to the window that has input focus at the moment. The `Application` class implements the `RunMessageLoop` method, which makes sure that each message is sent to the correct window. It also closes the application when a special quit message is sent.

The creation of a window takes place in two steps. In the first step, the `RegisterWindowClasses` method sets features such as style, color, and appearance. Note that Windows classes are not C++ classes:

[PRE1]

The next step is to create an individual window, which is done by the `Window` class. All `virtual` methods are empty and are intended to be overridden by sub classes shown as follows:

[PRE2]

A window can be visible or invisible, enabled or disabled. When a window is enabled, it accepts mouse, touch, and keyboard input:

[PRE3]

The `OnMove` and the `OnSize` methods are called when the window is moved or resized. The `OnHelp` method is called when the user presses the *F1* key or the **Help** button in a message box:

[PRE4]

The **client area** is the part of the window that it is possible to paint in. Informally, the client area is the window minus its frame. The contents of the client area can be zoomed. The default zoom factor is 1.0:

[PRE5]

The **timer** can be set to an interval in milliseconds. The `OnTimer` method is called on every interval. It is possible to set up several timers, as long as they have different identity numbers:

[PRE6]

The `OnMouseDown`, `OnMouseUp`, and `OnDoubleClick` methods are called when the user presses, releases, or double-clicks on a mouse button. The `OnMouseMove` method is called when the user moves the mouse with at least one button pressed. The `OnMouseWheel` method is called when the user moves the mouse wheel with one click:

[PRE7]

The `OnTouchDown`, `OnTouchMove`, and `OnTouchDown` methods work in the same way as the mouse methods. However, as the user can touch several points at the same time, the methods takes lists of points rather than an individual point:

[PRE8]

The `OnKeyDown` and `OnKeyUp` methods are called when the user presses or releases a key. If the user presses a graphical key (a key with an ASCII value between 32 and 127, inclusive), the `OnChar` method is called in between:

[PRE9]

The `Invalidate` method marks a part of the client area (or the whole client area) to be repainted; the area becomes **invalidated**. The area is cleared before the painting if `clear` is `true`. The `UpdateWindow` method forces a repainting of the invalidated area. It causes the `OnPaint` method to be called eventually:

[PRE10]

The `OnPaint` method is called when some part of the client area needs to be repainted and the `OnPrint` method is called when it is sent to a printer. Their default behavior is to call the `OnDraw` method with `Paint` or `Print` as the value of the `drawMode` parameter:

[PRE11]

The `OnClose` method closes the window if `TryClose` returns `true`. The `OnDestroy` method is called when the window is being closed:

[PRE12]

The following methods inspect and modify the size and position of the window. Note that we cannot set the size of the client area; it can only be set indirectly by resizing the window:

[PRE13]

In the word processor and spreadsheet programs in this book, we handle text and need to calculate the size of individual characters. The following methods calculate the width of a character with a given font. They also calculate the height, ascent, and average character width of a font:

[PRE14]

The ascent line separates the upper and lower part of a letter, shown as follows:

![The library](img/B05475_01_01.jpg)

Finally, the `MessageBox` method displays a simple message box in the window:

[PRE15]

The `Window` class also uses the `Graphics` class responsible for drawing text and geometrical objects in the window. A reference to a `Graphics` object is sent to the `OnPaint`, `OnPrint`, and `OnDraw` methods in the `Window` class. It can be used to draw lines, rectangles, and ellipses and to write text:

[PRE16]

The `Document` class extends the `Window` class with some functionality common to document-based applications. The scroll thumbs are automatically set to reflect the visible part of the document. The mouse wheel moves the vertical scroll bar one line-height for each click. The height of a line is set by the constructor. The code snippet for it is shown as follows:

[PRE17]

The **dirty flag** is `true` when the user has made a change in the document and it needs to be saved. In `Document`, the dirty flag is set manually, but in the following `StandardDocument` subclass it is handled by the framework:

[PRE18]

The **caret** is the blinking marker that indicates to the user where they should input the next character. The keyboard can be set (with the Insert key) to insert or overwrite mode. The caret is often a thin vertical bar in insert mode and a block with the width of an average character in overwrite mode.

The caret can be set or cleared. For instance, in the word processor, the caret is visible when the user writes text and invisible when the user marks text. When the window gains focus, the caret becomes visible if it has earlier been set. When the window loses focus, the caret becomes invisible, regardless of whether it has earlier been set:

[PRE19]

A document may hold a menu bar, which is set by the `SetMenuBar` method:

[PRE20]

The `OnDropFiles` method is called when the user drops one or several files in the window. Their paths are stored in the path list:

[PRE21]

The keyboard mode of a document can be set to **insert** or **overwrite** as follows:

[PRE22]

The `OnHorizontalScroll` and `OnVerticalScroll` methods are called when the user scrolls the bar by clicking on the scroll bar arrows or the scroll bar fields, or dragging the scroll thumbs. The code snippet for it is shown as follows:

[PRE23]

There is a large set of methods for inspecting or changing scroll bar settings. The size of a line or page is set by the constructor:

[PRE24]

The `Menu` class handles the menu bar, a menu, a menu item, or a menu item separator (a horizontal bar) in the document. The `selection` listener is called when the user selects the menu item. The `enable`, `check`, and `radio` listeners are called (unless they are null) when the item is about to become visible. If they return `true`, the item is enabled or annotated with a check box or radio button:

[PRE25]

An **accelerator** is a shortcut command. For instance, often, the **Open** item in the **File** menu is annotated with the text **Ctrl+O**. This means that you can obtain the same result by pressing the ***Ctrl*** key and the ***O*** key at the same time, just as if you selected the **Open** menu item. In both cases, the Open dialog is displayed.

The `Accelerator` class holds only the `TextToAccelerator` method. It interprets the menu item text and adds the accelerator, if present, to the accelerator set:

[PRE26]

The `StandardDocument` class extends the `Document` class and sets up a framework that takes care of all traditional tasks, such as load and save, and cut, copy, and paste, in a document-based application:

[PRE27]

The `StandardDocument` class comes equipped with the common **File**, **Edit**, and **Help** menus. The **File** menu can optionally (if the `print` parameter is `true`) be equipped with menu items for printing and print previewing:

[PRE28]

The `ClearDocument` method is called when the user selects the **New** menu item; its task is to clear the document. The `WriteDocumentToStream` method is called when the user selects the **Save** or **Save As** menu item and the `ReadDocumentFromStream` method is called when the user selects the **Open** menu item:

[PRE29]

The `CopyAscii`, `CopyUnicode`, and `CopyGeneric` methods are called when the user selects the **Cut** or **Copy** menu item and the corresponding `ready` method returns `true`. The code snippet for it is shown as follows:

[PRE30]

In the same way, the `PasteAscii`, `PasteUnicode`, and `PasteGeneric` methods are called when the user selects the **Paste** menu item and the corresponding `ready` method returns `true`:

[PRE31]

The `OnDropFile` method checks the path list and accepts the drop if exactly one file has the suffix of the document type of the application (set by the constructor):

[PRE32]

In Small Windows, we do not care about the pixel size. Instead, we use **logical units** that stay the same, regardless of the physical resolution of the screen. We can choose from the following three coordinate systems:

*   `LogicalWithScroll`: A logical unit is one hundredth of a millimeter, with the current scroll bar settings taken into account. The drawing program and word processor use this system.
*   `LogicalWithoutScroll`: A logical unit is one hundredth of a millimeter also in this case, but the current scroll bar settings are ignored. The spreadsheet program uses this system.
*   `PreviewCoordinate`: The client area of the window is set to a fixed logical size when the window is created. This means that the size of the logical units changes when the user changes the window size. The Tetris game and the `PreviewDocument` class uses this system.

Besides the `StandardDocument` class, there is also the `PrintPreviewDocument`, which class that also extends the `Document` class. It displays one of the pages of a standard document. It is possible for the user to change the page by using the arrow keys and the ***Page Up*** and ***Page Down*** keys or by using the vertical scroll bar:

[PRE33]

There are also the simple auxiliary classes:

*   `Point`: It holds a two-dimensional point (x and y)
*   `Size`: It holds two-dimensional width and height
*   `Rect`: It holds the four corners of a rectangle
*   `DynamicList`: It holds a dynamic list
*   `Tree`: It holds a tree structure
*   `InfoList`: It holds a list of generic information that can be transformed into a memory block

The `Registry` class holds an interface to the **Windows Registry**, the database in the Windows system that we can use to store values in between the execution of our applications. The `Clipboard` class holds an interface to the **Windows Clipboard**, an area in Windows intended for short-term data storage that we can use to store information cut, copied, and pasted between applications.

The `Dialog` class is designed for customized dialogs. The `Control` class is the root class for the controls of the dialog. The `CheckBox`, `RadioButton`, `PushButton`, `ListBox`, and `ComboBox` classes are classes for the specific controls. The `TextField` class holds a text field that can be translated to different types by the `Converter` class. Finally, the `PageSetupDialog` class extends the `Dialog` class and implements a dialog with controls and converters.

# Summary

This chapter has given an introduction to Small Windows. In [Chapter 2](ch02.html "Chapter 2. Hello, Small World!"), *Hello, Small World*, we will start to develop applications with Small Windows.