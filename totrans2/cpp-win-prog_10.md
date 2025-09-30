# Chapter 10. The Framework

The remaining chapters of this book explain the details of the Small Windows implementation. This chapter covers the following topics:

*   An overview of the classes of Small Windows
*   An example of the Hello World application, which we covered at the beginning of this book, written in the Win32 API7
*   The `MainWindow` and `WinMain` functions
*   The implementation of the main classes of Small Windows: `Application`, `Window`, and `Graphics`

# An overview of Small Windows

Here is a short description of the classes of Small Windows:

| **Chapter** | **Class** | **Description** |
| 10 | `Application` | This is the `main` class of Small Windows. It manages the message loop and registration of Windows classes. |
| 10 | `Window` | This the root `Window` class. It creates individual windows and provides basic window functionality, such as mouse, touch, and keyboard input, drawing, zooming, timer, focus, size, and coordinate systems. |
| 10 | `Graphics` | This is the class for drawing lines, rectangles, ellipses, and text in the client area of the window. |
| 11 | `Document` extends `Window` | This extends the window with document functionality, such as scrolling, caret handling, and drop files. |
| 11 | `Menu` | This handles menu bars, menus, menu items, and the menu separator. |
| 11 | `Accelerator` | This extracts accelerator information from the menu item texts. |
| 11 | `StandardDocument` extends `Document` | This provides a document-based framework with the common **File**, **Edit**, and **Help** menu items. |
| 12 | `Size``Point``Rect` | These are auxiliary classes that handle a two-dimensional point (x and y), size (width and height), or the four corners of a rectangle. |
| 12 | `Font` | This wraps the `LOGFONT` structure, which holds information about the font's name, size, and whether it is bold or italic. |
| 12 | `Cursor` | This sets the cursor and provides a set of standard cursors. |
| 12 | `DynamicList` template | This is a list of dynamic size and a set of callback methods. |
| 12 | `Tree` template | This is a tree structure where each node has a (possibly empty) list of child nodes. |
| 12 | `InfoList` | This is a list of generic information, which can be transformed to and from a memory buffer. |
| 13 | `Registry` | This provides an interface against the Windows registry. |
| 13 | `Clipboard` | This provides an interface against the Windows clipboard. |
| 13 | `StandardDialog` | This displays the standard dialogs for saving and opening files, choosing a font or color, and printing. |
| 13 | `PreviewDocument` extends `Document` | This sets up a document whose logical size is fixed regardless of its physical size. |
| 14 | `Dialog` extends `Window` | This provides a modal dialog. The controls below are added to the dialog. |
| 14 | `Control` abstract | This is the base class for dialog controls. |
| 14 | `ButtonControl` extends `Control` | This is the base class for button controls. |
| 14 | `GroupBox`, `PushButton`, `CheckBox`, `RadioButton` extends `ButtonControl` | These are classes for group boxes, push buttons, checkboxes, and radio buttons. |
| 14 | `ListControl` extends `Control` | This is the base class for list controls. |
| 14 | `ListBox`, `MultipleListBox` extends `ListControl` | These are classes for single and multiple list boxes. |
| 14 | `ComboBox` extends `Control` | This is the class for a combo (drop-down) box. |
| 14 | `Label` extends `Control` | This is the class for a simple label, often used as a prompt for `TextField`. |
| 14 | `TextField` template extends `Control` | This is a class for an editable field, where a converter may convert between a string and any type. |
| 14 | `Converter` template | This is a converter class that can be specified by any type. |
| 14 | `PageSetupDialog` extends `Dialog` | This is a dialog for page setup settings, such as margins, headers, and footer text. |
| 14 | `PageSetupInfo` | This has page setup information, which we saw previously. |

# "Hello" window for the Win32 API

First of all, let's take a look at the Hello application from the first chapter of this book. The following code snippet is the same application written directly with the Win32 API, without Small Windows. Note that the code is written in C rather than C++ as the Win32 API is a C function library rather than a C++ class library. As you can see, the code is a lot more complicated compared to the application in the first chapter.

Do not worry if it looks complicated. Its purpose is actually to demonstrate the complexity of the Win32 API; we'll discuss the details in this and the following chapters.

**MainWindow.c**

[PRE0]

The `WinMain` method is called when the application starts to execute. It corresponds to `main` in Standard C.

[PRE1]

First, we need to register the `Windows` class for our window. Note that `Windows` classes are not C++ classes:

[PRE2]

The style of the `Windows` class will be redrawn when the window size is changed in the horizontal and vertical direction:

[PRE3]

The icon of the window is the standard application icon, the cursor is the standard arrow cursor, and the background of the client area is white.

[PRE4]

The `WindowProc` function is a callback function called every time the window receives a message:

[PRE5]

The name of the `Windows` class is `window`, which is used in the `CreateWindowEx` call here:

[PRE6]

The `CreateWindowEx` method creates a window with the default position and size. Note that we can create many windows with the same `Windows` class:

[PRE7]

The `GetMessage` method waits for the next message, which is translated and dispatched to the window with an input focus. The `GetMessage` method returns `true` for all messages except the quit message, which is eventually sent when the user closes the window:

[PRE8]

When painting the client area, we need to create a paint structure and a device context, which is created by `BeginPaint`:

[PRE9]

Since we want to use logical units (hundreds of a millimeters), we need to set the device context by calling `SetWindowExtEx` and `SetViewportExtEx`:

[PRE10]

Since we also want to take scroll movements into consideration, we also call `SetWindowOrgEx`:

[PRE11]

Also, as we want to take scroll movements into consideration, we call `SetWindowOrgEx` to set to logical origin of the client area:

[PRE12]

We need to set a `LOGFONT` structure to create the 12-point boldface `Times New Roman` font:

[PRE13]

Since we work with logical units that are hundreds of millimeters, one typographical point is 1 inch divided by 72 and 1 inch is 25.4 millimeters. We multiply the font size by 2,540 and divide it by 72:

[PRE14]

When we use the font to write text in the client area, we need to create the font indirectly and add it as a graphical object. We also need to save the previous object in order to restore it later:

[PRE15]

The text color is black and its background color is white. `RGB` is a macro that transforms the red, green, and blue parts of the color into a `COLORREF` value:

[PRE16]

Finally, `DrawText` draws the text in the middle of the client area:

[PRE17]

Since fonts are system resources, we need to restore the previous font object and delete the new font object. We also need to restore the paint structure:

[PRE18]

Since we have handled the `WM_PAINT` message, we return zero.

[PRE19]

For all messages other than `WM_PAINT`, we call `DefWindowProc` to handle the message:

[PRE20]

# The MainWindow function

In regular C and C++, the execution of the application starts with the `main` function. In Small Windows, however, `main` has been replaced by `MainWindow`. `MainWindow` is implemented by the user of Small Windows for each project. Its task is to define the application name and create the main window object.

**MainWindow.h**

[PRE21]

# The WinMain function

In the Win32 API, `WinMain` is the function equivalent to `main`. Each application must include the definition of the `WinMain` function. In order for Small Windows to work, `WinMain` is implemented as a part of Small Windows, while `MainWindow` has to be implemented by the user of Small Windows for each project. To sum it up, here are the three kinds of main functions:

| **Regular C/C++** | **Win32 API** | **Small Windows** |
| main | WinMain | MainWindow |

The `WinMain` function is called by the Windows system and takes the following parameters:

*   `instanceHandle`: This holds the handle of the application
*   `prevInstanceHandle`: This is present due to backward compatibility but is always `null`
*   `commandLine`: This is a null-terminated character (`char`, not `TCHAR`) array holding the arguments for the application, separated by spaces
*   `commandShow`: This holds the preferred appearance of the main window

**WinMain.cpp**

[PRE22]

The `WinMain` function performs the following tasks:

*   It divides the space-separated words of the command line into a `String` list by calling `GenerateArgumentList`. Refer to [Chapter 12](ch12.html "Chapter 12. The Auxiliary Classes"), *Auxiliary Classes*, for the definitions of `CharPtrToGenericString` and `Split`.
*   It instantiates an `Application` object.
*   It calls the `MainWindow` function, which creates the main window of the application and sets its name.
*   It calls the `RunMessageLoop` method of `Application`, which continues to handle Windows messages until the quit message is sent.

[PRE23]

# The Application class

The `Application` class handles the message loop of the application. The message loop waits for the next message from the Windows system and sends it to the right window. The `Application` class also defines the `Windows` classes (which are not C++ classes) for the `Window`, `Document`, `StandardDocument`, and `Dialog` C++ classes. The fields of the classes are static since `Application` is not intended to be instantiated.

From this point in Small Windows, every part of the Small Windows implementation is included in the `SmallWindows` namespace. A namespace is a C++ feature that encapsulates classes and functions. The declaration of `MainWindow`, we saw earlier, is not included in the `Smallwindows` namespace since the C++ language rules stipulate that it cannot be included in a namespace. The `WinMain` definition is also not included in the namespace, since it needs to be placed outside the namespace to be called by the Windows system.

**Application.h**

[PRE24]

The `RegisterWindowClasses` method defines the Windows classes for the `Window`, `Document`, `StandardDocument`, and `Dialog` C++ classes. The `RunMessageLoop` method runs the message loop of the Windows message system. It waits for the next message and sends it to the right window. When a special quit message is received it breaks the message loop, which leads to the termination of the `Application` class:

[PRE25]

In Windows, each application holds a **handle** to the application instance. Handles are common in the Win32 API, and are used to access objects of the Windows system. They are similar to pointers but provide identification without revealing any location information.

The instance handle (of the `HINSTANCE` type) is used when creating windows in the constructor of the following `Window` class and when displaying standard dialogs in the Standard Dialogs section in [Chapter 14](ch14.html "Chapter 14. Dialogs, Controls, and Page Setup"), *Dialogs, Controls, and Page Setup*:

[PRE26]

The application name is set by each application and is referred to by the standard **File**, **Help**, and **About** menus, the **Open** and **Save** dialogs, and the registry:

[PRE27]

The pointer to the main window of the application is referenced when the user closes a window. If it is the main window, the application exits. Moreover, when the user selects the **Exit** menu item, the main window is closed before the application exits:

[PRE28]

**Application.cpp**

[PRE29]

## The Win32 API Windows classes

The `Windows` classes are registered in `Application`. A Windows class needs to be registered only once. After it has been registered, more than one window can be created for each `Windows` class. Again, note that windows classes are not C++ classes. Each `Windows` class is stored by its name: `lpszClassName`. The `lpfnWndProc` field defines the freestanding function that receives the window messages from the message loop. Each window allows double-clicks as well as horizontal and vertical redraw styles, which means that the `WM_PAINT` message is sent to the window and the `OnPaint` method is called each time the user changes the size of the window. Moreover, each window has the standard application icon in its top-right corner and the standard arrow cursor. The client area is white, except for the dialog, where the client area is light gray:

[PRE30]

## The message loop

The `RunMessageLoop` method holds the classic Windows message loop. There are two cases: if the main window pointer points at an object of the `Window` class, we just need to handle the message queue with the Win32 API functions `GetMessage`, `TranslateMessage`, and `DispatchMessage` without caring about accelerators. However, if it points at an object of `Document` or any of its subclasses, the message loop becomes more complicated because we need to take accelerators into consideration:

[PRE31]

If the main window pointer points at an object of `Document` or any of its subclasses, we set up a buffer for the accelerator table defined in `Document`, which we use in the message loop. The Win32 API `TranslateAccelerator` function looks up the accelerator and decides whether a key stroke message should be treated as the menu item associated with the accelerator:

[PRE32]

The `TranslateAccelerator` method wants an array of ACCEL structures, so we convert the accelerator set to an array:

[PRE33]

When the accelerator array is used, it is deleted:

[PRE34]

When the message loop is finished, we return the last massage:

[PRE35]

# The Window class

The `Window` class is the root class of the document classes; it handles basic window functionality such as the timer, input focus, coordinate transformation, window size and position, text metrics, and the message box as well as mouse, keyboard, and touch screen input. Moreover, `Window` defines enumerations for window styles and appearances, buttons, icons, and coordinate systems.

**Window.h**

[PRE36]

There is large set of window styles. The window may be equipped with a border, a thick frame, scroll bars, or minimize and maximize boxes:

[PRE37]

The window can be displayed in minimized, maximized, or normal mode:

[PRE38]

A mouse may hold the left, middle, and right button. The mouse wheel can be rolled upwards or downwards:

[PRE39]

There are four kinds of coordinate system as follows:

*   `LogicalWithScroll`: In this, each unit is one hundredth of a millimeter, regardless of the physical screen resolution, with the current scroll bar settings taken into consideration
*   `LogicalWithoutScroll`: This is the same as `LogicalWithScroll`, except that the scroll bars settings are ignored
*   `PreviewCoordinate`: In this, the window client area always holds a specific logical size, which means that the size of the logical units is changed when the size of the window is changed

[PRE40]

The message box comes equipped with a set of button combinations, icons, and answers. Note that the answer corresponding to the **OK** button is named `OkAnswer` in the `Answer` enumeration in order to avoid name clashes with the `OK` button in the `ButtonGroup` enumeration:

[PRE41]

The default definitions of `OnPaint` and `OnPrint` both call `OnDraw`. In order to distinguish the two cases, the `OnDraw` parameter has the value `Paint` or `Print`:

[PRE42]

The first `Window` constructor is public and intended to be used when a window is created directly. The `pageSize` field refers to the size of the window client area. The constructor also takes a pointer to the window's parent window (which is `null` if there is no parent window), the window's basic style and extended style, and its initial appearance, position, and size. If the position or size is zero, the window is located or dimensioned in accordance with the system's default settings.

Note the difference between the document and windows sizes in `PreviewCoordinate`: the document size is the size of the client area in units defined by the window's coordinate system, while the size and position of the window are given in the coordinate system of the parent window or in device units if there is no parent window. Moreover, the document size refers to the size of the client area while the window size refers to the size of the whole window:

[PRE43]

The second constructor is protected and intended to be called by subclasses' constructors. The difference when compared to the first constructor is that is takes the name of the `window` class as its first parameter. As defined by the `Application` class, the class name can be `Window`, `Document`, `StandardDocument`, or `Dialog`:

[PRE44]

A **device context** is used when painting the client area, when transforming between logical and device units, and when calculating the size of text. It is a connection to the client area of a window or to a printer. However, since it comes with a set of functions for drawing text of graphical objects, it can also be considered as a toolbox for drawing. However, before it is used, it needs to be prepared and adjusted in accordance with the current coordinate system:

[PRE45]

The destructor destroys the window and exits the application if the window is the application's main window:

[PRE46]

The window can be visible or invisible; it can also be enabled in such a way that it catches mouse, touch, and keyboard inputs:

[PRE47]

The `OnSize` and `OnMove` methods are called when the user changes the size of the window or moves it. The size and position are given in logical coordinates. The `OnHelp` method is called when the user presses the *F1* key of the *Help* button in a message box. The methods are intended to be overridden by subclasses, and their default behavior is to do nothing:

[PRE48]

The `WindowHandle` method returns the Win32 API window handle, which is used by standard dialog functions. The `ParentWindowPtr` method returns the pointer to the parent window, which is `null`, meaning that there is no parent window. The `SetHeader` method sets the title of the window, which is visible in the upper border:

[PRE49]

The client area of the window is zoomed in accordance with the zoom factor; 1.0 corresponds to the normal size:

[PRE50]

Several timers can be set or dropped as long as the values of the `timerId` parameter differ. The `OnTimer` method is called in accordance with the intervals in milliseconds; its default behavior is to do nothing.

[PRE51]

The `SetFocus` method sets the input focus to this window. The input focus directs the keyboard input and clipboard to the window. However, the mouse pointer may be aiming at another window. The window previously holding the input focus loses the focus; only one window can hold the focus at a given time. The `HasFocus` method returns `true` if the window has input focus.

[PRE52]

The `OnGainFocus` and `OnLoseFocus` methods are called when the window gains or loses input focus. They are intended to be overridden by subclasses, and their default behavior is to do nothing.

[PRE53]

In Windows, a mouse is regarded as holding three buttons, even if it does not do so physically. The mouse buttons can be pressed or released and the mouse can be moved. The `OnMouseDown`, `OnMouseUp`, and `OnMouseMove` methods are called when the user presses or releases one of the mouse buttons or moves the mouse with at least one button pressed. The user may press the ***Shift*** or ***Ctrl*** key at the same time, in which case `shiftPressed` or `controlPressed` is `true`:

[PRE54]

The user can also double-click a mouse button, in which case `OnDoubleClick` is called. What constitutes a double-click is decided by the Windows system and can be set in the Control Panel. When the user single-clicks a button, `OnMouseDown` is called, followed by `OnMouseMove` in the case of potential mouse movements, and finally `OnMouseUp`. However, in the case of a double-click, `OnMouseDown` is not called, its call is replaced by `OnDoubleClick`:

[PRE55]

The `OnMouseWheel` method is called when the user rolls the mouse wheel one step upwards or downwards:

[PRE56]

The `OnTouchDown`, `OnTouchMove`, and `OnTouchUp` methods are called when the user touches the screen. Unlike mouse clicks, the user can touch the screen at several locations at the same time. Therefore, the parameter is a list of points rather than a single point. The methods are intended to be overridden by subclasses. Their default behavior is to simulate a mouse click for each touch point with no button and with neither the ***Shift*** nor the ***Ctrl*** key pressed:

[PRE57]

The `OnKeyDown` and `OnKeyUp` methods are called when the user presses and releases a key. If the key is a graphical character (with ASCII number between 32 and 127, inclusive), `OnChar` is called in between. The `OnKeyDown` and `OnKeyUp` methods return `bool`; the idea is that the methods return `true` if the key was used. If not, they return `false` and the caller method is free to use the key to, for instance, control scroll movements:

[PRE58]

The `OnPaint` method is called when the client area of the window needs to be redrawn, partly or completely, and `OnPrint` is called when the user selects the **Print** menu item. In both cases, the default definition calls `OnDraw`, which performs the actual drawing; `drawMode` is `Paint` when called by `OnPaint` and `Print` when called by `OnPrint`. The idea is that we let `OnPaint` and `OnPrint` perform actions specific to painting and printing and call `OnDraw` for the common drawing. The `Graphics` class is described in the next section:

[PRE59]

The `Invalidate` method invalidates the client area, partly or completely; that is, it prepares the area to be redrawn by `OnPaint` or `OnDraw`. If `clear` is `true`, the area is first cleared (painted by the window client color). The `UpdateWindow` method forces a repainting of the invalidated parts of the client area:

[PRE60]

The `OnClose` method is called when the user tries to close the window; its default behavior is to call `TryClose`. If `TryClose` returns `true` (which it does in its default definition), the window is closed. If that happens, `OnDestroy` is called, whose default behavior is to do nothing:

[PRE61]

The following method transforms a `Point`, `Rectangle`, or `Size` object between device units and logical units. They are protected since they are intended to be called by subclasses only:

[PRE62]

The following method gets or sets the size and position of the window and the client area in device units:

[PRE63]

The following method gets or sets the logical size and position of the window and the client area, in logical units, in accordance with the coordinate system of the window:

[PRE64]

The `CreateTextMetric` method initializes and returns a Win32 API `TEXTMETRIC` structure, which is then used by the text metric methods in order to calculate the logical size of text. It is private since it in intended to be called only by the `Window` methods:

[PRE65]

The following method calculates and returns the width, height, ascent, or average width of a character or text with the given font, in logical units:

[PRE66]

The `MessageBox` method displays a message box with a message, caption, a set of buttons, an icon, and on optional **Help** button:

[PRE67]

The `pageSize` field holds the window client's logical size in the `PreviewCoordinate` coordinate system, which is used when transforming coordinates between logical and device coordinates. In the `LogicalWithScroll` and `LogicalWithoutScroll` coordinate systems, `pageSize` holds the logical size of the document, which does not necessarily equal the logical size of the client area, and is not changed when the window is resized. It is protected since it is also used by the `Document` and `StandardDocument` subclasses in the next chapter:

[PRE68]

In the previous section, there was a handle to the application instance. `windowHandle` is a handle of type `HWND` to a Win32 API window; `parentPtr` is a pointer to the parent window, which is `null` if there is no parent window:

[PRE69]

The coordinate system chosen for the window is stored in `system`. The `zoom` field holds the zooming factor of the window, where 1.0 is the default:

[PRE70]

The `WindowProc` method is called each time the window receives a message. It is a friend of `Window`, since it needs access to its private members:

[PRE71]

Finally, `WindowMap` maps the `HWND` handles to the `Window` pointers, which are used in `WindowProc` as follows:

[PRE72]

**Window.cpp**

[PRE73]

## Initialization

The first constructor simply calls the second constructor with the class name `window`:

[PRE74]

The second constructor initializes the `parentPtr`, `system`, and `pageSize` fields:

[PRE75]

If the window is a child window (the parent pointer is not `null`), its coordinates are converted to the coordinate system of its parent window:

[PRE76]

The Win32 API window creation process is divided into two steps. First, a Windows class needs to be registered, which was done in the `Application` constructor earlier. Then, the `Windows` class name is used in the call to the Win32 API `CreateWindowEx` function, which returns a handle to the window. If the size or position is zero, default values are used:

[PRE77]

In order for `WindowProc` to be able to receive messages and identify the recipient window, the handle is stored in `WindowMap`:

[PRE78]

The Win32 API functions `ShowWindow` and `RegisterTouchWindow` are called to make the window visible in accordance with the `windowShow` parameter and to make the window receptive to touch movements:

[PRE79]

The destructor calls `OnDestroy` and erases the window from `windowMap`. If the window has a parent window, it receives an input focus:

[PRE80]

If the window is the application's main window, the Win32 API `PostQuitMessage` function is called. It posts a quit message, which is eventually caught by `RunMessageLoop` in the `Application` class that terminates the execution. Finally, the window is destroyed:

[PRE81]

## Header and visibility

The `ShowWindow` and `EnableWindow` methods call the Win32 API functions `ShowWindow` and `EnableWindow` with the window handle as their first parameter:

[PRE82]

Note that the second parameter of `EnableWindow` is a value of the Win32 API type `BOOL`, which is not necessarily the same type as the C++ type `bool`. Therefore, since `enable` holds the type `bool` we need to convert it to `BOOL`:

[PRE83]

The `SetHeader` method sets the title of the window by calling the Win32 API function `SetWindowText`. As `headerText` is a `String` object and `SetWindowText` wants a C string (a zero-terminated char pointer) as parameter, we need to call the `c_str` function:

[PRE84]

The `SetTimer` and `DropTimer` methods turn the timer with the given identity on and off by calling the Win32 API functions `SetTimer` and `KillTimer`. The interval in the `SetTimer` call is given in milliseconds:

[PRE85]

The `SetFocus` method sets the focus by calling the corresponding Win32 API function `SetFocus`. The `HasFocus` method returns `true` if the window has the input focus by calling the `GetFocus` Win32 API function, which returns the handle to the window, holding the input focus that is compared to the window's handle:

[PRE86]

## The touch screen

The default behavior of `OnTouchDown`, `OnTouchMove`, and `OnTouchUp` is to call the corresponding mouse input method for each touch point, with no button and neither the ***Shift*** nor the ***Ctrl*** key pressed:

[PRE87]

With a modern screen, the user can touch the screen in ways similar to mouse clicks. However, the user can touch the screen at several locations at once, and its positions are stored in a point list. The `OnTouch` method is an auxiliary method calling `OnTouchDown`, `OnTouchMove`, and `OnTouchUp` when the user touches the screen. It creates a list of points in logical coordinates:

[PRE88]

If the touch identity does not equal the first value in the input array, we have a touch down event; if it does, we have a touch move event:

[PRE89]

## Invalidation and window updates

When the window's client area needs to be (partly or completely) repainted, one of the `Invalidate` methods is called. The `Invalidate` methods call the Win32 API function `InvalicateRect`, which posts a message that results in a call to `OnPaint` when `UpdateWindow` is called. The `clear` parameter indicates whether the invalidated area should be cleared (repainted with the window client area's color) before it is redrawn, which normally is the case. Similar to the `EnableWindow` method we saw earlier, we need to convert `clear` from type `bool` to `BOOL`:

[PRE90]

The `Invalidate` method transforms the area from logical to device coordinates before the call to the Win32 API function `InvalidateRect` and stores the size in a `RECT` structure:

[PRE91]

The `UpdateWindow` method calls the Win32 API function `UpdateWindow`, which eventually results in a call to `OnPaint`:

[PRE92]

## Preparing the device context

When painting the windows's client area, we need a device context, which we need to prepare in accordance with the coordinate system in order to paint with logical coordinates. The Win32 API function `SetMapMode` sets the mapping mode of the logical coordinate system. `MISOTROPIC` forces that the *x* and *y* axis to have the same unit length (resulting in non-elliptic circles) that is suitable for the `LogicalWithScroll` and `LogicalWithoutScroll` systems, while `MANISOTROPIC` allows different unit lengths that are suitable for the `PreviewCoordinate` system. We establish a mapping between the logical and device systems by calling the Win32 API functions `SetWindowExtEx`, which takes the logical size of the client area, and `SetViewportExtEx`, which takes its physical (device) size.

In the case of the `PreviewCoordinate` system, we simply match the logical size (`pageSize`) of the client area to its device size (`clientDeviceRect`), given by the Win32 API function `GetClientRect`, resulting in the client area always having the same logical size, regardless of its physical size:

[PRE93]

In the case of the logical coordinate system, we need to find the ratio between logical coordinates (hundreds of millimeters) and device coordinates (pixels). In other words, we need to establish the logical size of a pixel. We can find the number of pixels on the screen by calling the Win32 API function `GetDeviceCaps` with `HORZSIZE` and `VERTSIZE`, and the size of the screen in millimeters with `HORZRES` and `VERTRES`. We multiply the logical size by 100, since we have hundreds of millimeters as our logical unit. We also need to take into account the zooming factor of the window, which we do by multiplying the physical size by `zoom`.

Note that it's only in the `PreviewCoordinate` system that the client area always has the same logical size. In the other systems, the logical size changes when the size of the window is changed. The logical units are always the same in `LogicalWithScroll` and `LogicalWithoutScroll`: hundreds of millimeters:

[PRE94]

In the case of the `LogicalWithScroll` logical coordinate system, we also need to adjust the origin of the window in accordance with the current scroll settings by calling the Win32 API function `SetWindowOrg`:

[PRE95]

## Unit transformation

The `DeviceToLogical` method transforms the device coordinates of a point, rectangle, or size to logical coordinates by preparing the device context and then calling the Win32 API function `DPtoLP` (Device Point to Logical Point). Note that we establish the device context by calling the Win32 API function `GetDC` and we need to return it by calling `ReleaseDC`. Also, note that we need to convert the `Point` object to a `POINT` structure and back again, since `DPtoLP` takes a pointer to a `POINT`:

[PRE96]

When transforming a rectangle, we use the point method to transform its top-left and bottom-right corners. When transforming a size, we create a rectangle, call the rectangle method, and convert the rectangle to a size:

[PRE97]

The `LogicalToDevice` method transforms the point, rectangle, or size from logical to device coordinates calling the Win32 API function `LPtoDP` (Logical Point to Device Point) in the same manner as the earlier methods. The only difference is that they call `LPtoDP` instead of `DPtoLP`:

[PRE98]

## Window size and position

The `GetWindowDevicePosition`, `SetWindowDevicePosition`, `GetWindowDeviceSize`, `SetWindowDeviceSize`, and `GetClientDeviceSize` methods call the corresponding Win32 API functions `GetWindowRect`, `GetClientRect`, and `SetWindowPos`:

[PRE99]

The `GetWindowPosition`, `SetWindowPosition`, `GetWindowSize`, `SetWindowSize`, and `GetClientSize` methods call the corresponding device methods together with `LogicalToDevice` or `DeviceToLogical`:

[PRE100]

## Text metrics

Given a font, `CreateTextMetric` creates a metric structure holding the height, ascent line, and average width of a character of the font. The `CreateFontIndirect` and `SelectObject` methods prepare the font for `GetTextExtentPoint`:

[PRE101]

Note that `CreateFontIndirect` must be matched by `DeleteObject` and the first call to `SelectObject` must be matched by a second call to `SelectObject` to reinstall the original object:

[PRE102]

Also, note that the device context received from `GetDC` must be released with `ReleaseDC`:

[PRE103]

The `GetCharacterHeight`, `GetCharacterAscent`, and `GetCharacterAverageWidth` methods call `CreateTextMetric` and return the relevant information:

[PRE104]

The `GetCharacterWidth` method calls `GetTextExtentPoint` to establish the width of a character of the given font. Since the font height is given in typographical points (1 point = 1/72 of an inch = 1/72 * 25.4 mm ≈≈ 0.35 mm) and needs to be given in millimeters, we call `PointsToLogical`. Similar to what we did earlier in `CreateTextMetric`, `CreateFontIndirect` and `SelectObject` prepare the font for `GetTextExtentPoint`:

[PRE105]

## Closing the window

When the user tries to close the window, the `Window` object (`this`) is deleted if `TryClose` returns `true`:

[PRE106]

## The MessageBox method

The `MessageBox` method displays a message box holding a caption, a message, a combination of buttons (**OK**, **OK-Cancel**, **Retry-Cancel**, **Yes-No**, **Yes-No-Cancel**, **Cancel-Try-Continue**, or **Abort-Retry-Ignore**), an optional icon (**Information**, **Stop**, **Warning**, or **Question**), and an optional H**elp** button. It returns the answer **OK Answer** (since OK is already taken by the `ButtonGroup` enumeration), **Cancel**, **Yes**, **No**, **Retry**, **Continue**, **Abort**, or **Ignore**:

[PRE107]

When a window is created by calling `CreateWindowEx` in the `Window` class constructor, the name of a `Windows` class that has earlier been given by the `Application` class constructor is enclosed. When the class is registered, a freestanding function is also given. For the `Window` class, the function is `WindowProc`, which is thereby called every time the window receives a message.

The `wordParam` and `longParam` parameters (`WPARAM` and `LPARAM` are both 4 bytes) hold message-specific information, which may be divided into low and high words (2 bytes) with the `LOWORD` and `HIWORD` macros:

[PRE108]

First we need to find the `Window` object associated with the window handle by looking up the handle in the static field `WindowMap`:

[PRE109]

When receiving the `WSETFOCUS`, `WKILLFOCUS`, and `WTIMER` messages, the corresponding methods in `Window` are simply called. When the messages have been handled, they do not need to be further processed; therefore, zero is returned:

[PRE110]

The identity of the timer (the `timerId` parameter in `SetTimer` and `DropTimer`) is stored in `wordParam`:

[PRE111]

When receiving the `WMOVE` and `WSIZE` messages, the `Point` value stored in `longParam` is given in device units that need to be transformed into logical units by calling `DeviceToLogical` in the calls to `OnMove` and `OnSize` in `Window`:

[PRE112]

If the user presses the ***F1*** key or the **Help** button in a message box, the `WM_HELP` message is sent. We call `OnHelp` in `Window`:

[PRE113]

When handling mouse or keyboard input messages, it is useful to decide whether the user simultaneously presses the ***Shift*** or ***Ctrl*** key. This can be established by calling the Win32 API function, `GetKeyState`, which returns an integer value less than zero if the key is pressed when called with `VK_SHIFT` or `VK_CONTROL`:

[PRE114]

If `OnKeyDown` returns `true`, the key message has been processed and we return zero. If it returns `false`, the Win32 API function `DefWindowProc`, as shown here, will be called, which further processes the message:

[PRE115]

If the pressed key is a graphical character (ASCII numbers between 32 and 127, inclusive), `OnChar` is called:

[PRE116]

All mouse input points stored in `longParam` are given in device coordinates, which need to be transformed into logical coordinates by `DeviceToLogical`. The mouse-down message is normally followed by the corresponding mouse-up message. Unfortunately, that is not the case if the user presses the mouse button in one window and releases it in another window, in which case the mouse-up message is sent to the other window. However, the problem can be solved by the Win32 API function, `SetCapture`, which makes sure that every mouse message is sent to the window until `ReleaseCapture` is called:

[PRE117]

When the user moves the mouse, they may at the same time press a combination of buttons, stored in `buttonMask`:

[PRE118]

Note that `ReleaseCapture` is called at the end of the mouse-up methods in order to release the mouse message from the window and make it possible for mouse messages to be sent to other windows:

[PRE119]

When a touch message is sent, `OnTouch` is called, which needs the position of the window in device units:

[PRE120]

When creating a device context in response to a paint message, we use the Win32 API functions `BeginPaint` and `EndPaint` instead of `GetDC` and `ReleaseDC` to handle the device context. However, the device context still needs to be prepared for the window's coordinate system, which is accomplished by `PrepareDeviceContext`:

[PRE121]

When the user tries to close the window by clicking on the close box in the top-right corner, `OnClose` is called. It calls `TryClose` and closes the window if `TryClose` returns true:

[PRE122]

If we reach this point, the Win32 API function `DefWindowProc` is called, which performs the default message handling:

[PRE123]

# The Graphics class

The `Graphics` class is a wrapper class for a device context. It also provides functionality for drawing lines, rectangles, and ellipses; writing text; saving and restoring graphic states; setting the origin of the device context; and clipping the painting area. The constructor is private since `Graphics` objects are intended to be created internally by Small Windows only.

**Graphics.h**

[PRE124]

When drawing a line, it can be solid, dashed, dotted, dashed and dotted, as well as dashed and double-dotted:

[PRE125]

The `Save` method saves the current state of the `Graphics` object and `Restore` restores it:

[PRE126]

The `SetOrigin` method sets the origin of the coordinate system and `IntersectClip` restricts the area to be painted:

[PRE127]

The following methods draw lines, rectangles, and ellipses, and write text:

[PRE128]

The `GetDeviceContextHandle` method returns the device context wrapped by the `Graphics` object:

[PRE129]

The `windowPtr` field holds a pointer to the window about which client area is to be drawn, and `deviceContextHandle` holds the handle to the device context, of type `HDC`:

[PRE130]

The `WindowProc` and `DialogProc` functions are friends of the `Graphics` class, since they need access to its private members. This is the same for the `PrintDialog` methods of the `StandardDialog` class:

[PRE131]

**Graphics.cpp**

[PRE132]

The constructor initializes the window pointer and device context:

[PRE133]

Sometimes, it is desirable to save the current state of the `Graphics` object with `Save`, which returns an identity number that can be used to restore the `Graphics` object with `Restore`:

[PRE134]

The default origin (x = 0 and y = 0) of the coordinate system is the top-left corner of the window client area. This can be changed with `SetOrigin`, which takes the new origin in logical units. The win32 API function `SetWindowOrgEx` sets the new origin:

[PRE135]

The part of the client area to be painted can be restricted with `IntersectClip`, resulting in the area outside the given rectangle not being affected. The Win32 API function `IntersectClip` sets the restricted area:

[PRE136]

It is possible to draw lines, rectangles, and ellipses using a pen, which is obtained by the Win32 API functions `CreatePen` and `SelectObject`. Note that we save the previous object in order to restore it later:

[PRE137]

By the way, the technique of moving the pen to the start point and then drawing the line to the end point with `MoveToEx` and `LineTo` is called **Turtle** **graphics**, referring to a turtle moving over the client area with the pen up or down:

[PRE138]

Similar to `CreateTextMetrics` and `GetCharacterWidth` in `Window`, we need to select the previous object and restore the pen:

[PRE139]

When drawing a rectangle, we need a solid pen and a hollow brush, which we create with the Win32 API function `CreateBrushIndirect` with a `LOGBRUSH` structure parameter:

[PRE140]

When filling a rectangle, we also need a solid brush, which we create with the Win32 API function `CreateSolidBrush`:

[PRE141]

The `DrawEllipse` and `FillEllipse` methods are similar to `DrawRectangle` and `FillRectangle`. The only difference is that they call the Win32 API function `Ellipse` instead of `Rectangle`:

[PRE142]

When drawing text, we first need to check whether the font is given in typographical points and needs to be transformed into logical units (if `pointToMeters` is true), which is the case in the `LogicalWithScroll` and `LogicalWithoutScroll` coordinates systems. However, in the `PreviewCoordinate` system, the size of the text is already given in logical units and should not be transformed. Moreover, before we write the text, we need to create and select a font object and set the text and background colors. The Win32 `DrawText` function centers the text within the given rectangle:

[PRE143]

# Summary

In this chapter, we looked into the core of Small Windows: the `MainWindow` function and the `Application`, `Window`, and `Graphics` classes. In [Chapter 11](ch11.html "Chapter 11. The Document"), *The Document*, we look into the document classes of Small Windows: `Document`, `Menu`, `Accelerator`, and `StandardDocument`.