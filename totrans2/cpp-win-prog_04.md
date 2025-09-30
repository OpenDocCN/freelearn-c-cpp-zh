# Chapter 4. Working with Shapes and Figures

In this chapter, we develop a program capable of drawing lines, arrows, rectangles, and ellipses. The application can be viewed as a more advanced version of the circle application. Similar to the circle application, we have a list of figures and we catch the user's mouse actions. However, there are four different kinds of figures: lines, arrows, rectangles, and ellipses. They are defined in a class hierarchy that is similar to but more advanced than the hierarchy in the Tetris game. Moreover, we also introduce cut, copy, paste, cursor control, and registry handling:

![Working with Shapes and Figures](img/image_04_001.jpg)

The user can add new figures, move one or several figures, modify figures by grabbing their endpoints, mark and unmark figures by pressing the mouse button and the *Ctrl* key, and mark several figures by enclosing them by a rectangle. When a figure is marked, it becomes annotated with small black squares. The user can modify the shape of a figure by grabbing one of the squares. The user can also move a figure by grabbing some other part of the figure.

# The MainWindow function

The `MainWindow` function in this application is very similar to that in [Chapter 3](ch03.html "Chapter 3. Building a Tetris Application"), *Building a Tetris Application*; it sets the application name and creates the main document window:

[PRE0]

# The DrawDocument class

The `DrawDocument` class extends the `StandardDocument` framework, similar to the circle application. It catches the mouse events, overrides the file methods, implements cut, copy, and paste, as well as cursor handling:

**DrawDocument.h**

[PRE1]

Similar to the circle application, we catch mouse action with the `OnMouseDown`, `OnMouseMove`, and `OnMouseUp` methods. However, in this application, we also catch double-clicks with the `OnDoubleClick` method. When the user double-clicks on a figure, it takes individual actions:

[PRE2]

The `OnDraw` method is called when the window's client area needs to be redrawn. It draws the figures, and the rectangle enclosing the figures, if the user is in the process of marking figures with a rectangle:

[PRE3]

The `ClearDocument` method is called when the user selects the **New** menu item, the `ReadDocumentFromStream` method is called when they select the **Open** menu item, and the `WriteDocumentToStream` method is called when they select the **Save** or **Save As** menu item:

[PRE4]

Each figure has an integer identity value that is written by the `WriteDocumentToStream` method and read by the `ReadDocumentFromStream` method to decide which figure has to be created. Given the identity value, the `CreateFigure` method creates the new figure:

[PRE5]

In this application, we introduce functionality for cut, copy, and paste. The `CopyGeneric` method is called when the user selects the **Cut** or **Copy** menu item in the **Edit** menu and the `PasteGeneric` method is called when the user selects the **Paste** menu item. In the `StandardDocument` framework, there are methods for cutting, copying, and pasting ASCII and Unicode text as well. However, we do not use them in this application:

[PRE6]

The `CopyEnable` method returns `true` if information is ready to be copied. In that case, the **Cut**, **Copy**, and **Delete** menu items are enabled. In this application, we do not override the `PasteEnable` method, since the `StandardDocument` framework looks up whether there is a memory buffer in the global clipboard suitable to paste. The `OnDelete` method is called when the user selects the **Delete** menu item:

[PRE7]

Similar to the circle application, we have a set of listeners, even though the set is larger in this case. Each listener is added to the menus in the constructor. Unlike the circle application, we also use enable methods: methods that are called before the menu item becomes visible. If the methods return `false`, the menu items become disabled and grayed. If the menu item is connected to an accelerator, the accelerator also becomes disabled. We place the **Modify**, **Color**, and **Fill** items in the **Modify** menu, and the **Line**, **Arrow,** **Rectangle**, and **Ellipse** items in the **Add** menu:

[PRE8]

In this application, we also introduce cursor control. The `UpdateCursor` method sets the cursor to an appropriate appearance depending on whether the user is creating, modifying, or moving figures:

[PRE9]

One central point of this application is its mode: the `applicationMode` method keeps track of the actions when the user presses the left mouse button. It holds the following modes:

*   `Idle`: The application waits for input from the user. This is always the mode as long as the user does not press the left mouse button. However, when the user presses the mouse button, until they release it, the `applicationMode` method holds one value. The user presses the *Ctrl* key and clicks on an already marked figure. The figure becomes unmarked, nothing more happens.
*   `ModifySingle`: The user grabs one single figure that is being modified (if the user clicks on one of its endpoints) or moved (if the user clicks on any other part of the figure).
*   `ModifyRectangle`: The user has clicked on the client area without hitting a figure, resulting in a rectangle being drawn. When the user releases the mouse button, every figure completely enclosed by the rectangle is marked.
*   `MoveMultiple`: The user presses the *Ctr*l key and clicks on an unmarked figure. It is not possible to modify more than one figure at the same time.

Note that the `applicationMode` method is relevant only as long as the user presses the left mouse button. As soon as they release the mouse button, the `applicationMode` method is always `Idle`:

[PRE10]

When the `applicationMode` method holds the `Idle` mode, the application waits for further input from the user. The `actionMode` field defines the next action, which may hold the following values:

*   `Modify`: When the user presses the mouse button, the `applicationMode` method is set to the `ModifySingle` mode if they click on a figure, the `MoveMultiple` mode if they click on an unmarked figure while pressing the *Ctrl* key, the `Idle` mode if the figure is already marked, or the `ModifyRectangle` mode if they click on the client area without hitting a figure.
*   `Add`: When the user presses the left mouse button, a new figure is created at the location, regardless of whether there already is a figure at the location. The value of the `addFigureId` method decides which kind of figure should be added; it can hold any of the values `LineId`, `ArrowId`, `RectangleId`, or `EllipseId`.

[PRE11]

Later in the chapter, we will encounter expressions such as **in Modify mode** and **in Add mode**, which refer to the value of the `actionMode` variable: `Modify` or `Add`.

The `nextColor` and `nextFill` fields hold the figure's color and fill status (in the case of a rectangle or ellipse), respectively, of the next figure to be added:

[PRE12]

Similar to the circle application, when the user adds or modifies a figure, we need to store the previous mouse position in the `prevMousePoint` method in order to keep track of the distance the mouse has been moved since the last mouse action:

[PRE13]

When the `applicationMode` method holds the `ModifySingle` value, the figure being modified is always placed at the beginning of the figure pointer list (`figurePtrList[0]`) in order for it to appear on top of the figures. When the `applicationMode` method holds the `ModifyRectangle` mode, the `insideRectangle` method keeps track of the rectangle enclosing the figures:

[PRE14]

The `static DrawFormat` constant is used to identify data to be cut, copied, or pasted in the global clipboard. It is arbitrarily set to 1000:

[PRE15]

As the user adds and removes figures from the drawing, the figures are dynamically created and deleted; their addresses are stored in the `figurePtrList` list. The `DynamicList` class is a Small Windows class that is a more advanced version of the C++ standard classes `list` and `vector`.

The values of the figure list are pointers to the `DrawFigure` class, which is the root class of the figure hierarchy used in this application (described in [Chapter 5](ch05.html "Chapter 5. The Figure Hierarchy"), *The Figure Hierarchy*). Unlike the circle and Tetris applications in the previous chapters, we do not store the figure objects directly in the list, but rather their pointers. This is necessary, since we use class hierarchy holds with pure virtual methods, which makes the `DrawWindow` class abstract and not possible to store directly in the list. It is also necessary in order to take advantage of dynamic binding of the class hierarchy:

[PRE16]

## The application modes

This section holds a further description of the `applicationMode` field. It is closely connected to the mouse input cycle. When the user is not pressing the left mouse button, the `applicationMode` method is always in the `Idle` mode. When the user presses the left mouse button in modify mode, they can choose to press the *Ctrl* key at the same time:

*   If they do not press the *Ctrl* key, the `applicationMode` method is set to the `ModifySingle` mode if they hit a figure. That figure becomes marked and other figures become unmarked.
*   If they do press the *Ctrl* key, the `applicationMode` method is set to the `MoveMultiple` mode if they hit a figure that is not marked and to the `Idle` mode if it is marked. The figure becomes marked if it is unmarked and unmarked if it is marked. The rest of the figures are unaffected.
*   If they do not hit a figure, the `applicationMode` method is set to the `ModifyRectangle` mode regardless of whether they pressed the *Ctrl* key and the inside rectangle (`insideRectangle`) is being initialized. All figures become unmarked. All figures that are completely enclosed by the rectangle when the user releases the left button are marked.

When the user moves the mouse with the left button pressed in modify mode, there are four possible values of the `applicationMode` method to consider:

*   `Idle`: We do nothing.
*   `ModifySingle`: We call the `Modify` method on the single figure. This may result in the single hit figure being modified or moved, depending on where the user hit the figure.
*   `MoveMultiple`: We call the `Move` method on all marked figures. This always results in the marked figures being moved, not modified.
*   `ModifyRectangle`: We modify the inside rectangle.

Finally, when the user releases the left mouse button, we again look into the four modes of the `applicationMode` method:

*   `Idle`, `ModifySingle`, or `MoveMultiple`: We do nothing since everything has already been done when the user moved the mouse. The marked figures have been moved or modified.
*   `ModifyRectangle`: We mark all figures completely enclosed by the rectangle.

## The DynamicList class

In this chapter, we use a subset of the methods of the auxiliary `DynamicList` class. It holds a set of methods that take callback functions, that is, functions that are sent as parameters to methods and called by the methods:

[PRE17]

`IfFuncPtr` and `DoFuncPtr` are pointers to callback functions. The difference between them is that the `IfFuncPtr` pointer is intended for methods that only inspect the values of the list. Therefore, the `value` parameter is constant. The `DoFuncPtr` pointer is intended for methods that modify the values. Consequently, the `value` parameter is not constant:

[PRE18]

The `AnyOf` method takes the `ifFuncPtr` pointer and applies it to each value of the array. The methods return `true` if at least one of the values satisfies the `ifFunctPtr` pointer (if the `ifFuncPtr` pointer returns `true` for the value). The `ifVoidPtr` parameter is sent as the second parameter to the `ifFuncPtr` pointer:

[PRE19]

The `FirstOf` method also returns `true` if at least one value satisfies the `ifFuncPtr` pointer. In that case, the first satisfied value is copied to the `value` parameter:

[PRE20]

The `Apply` method calls the `doFunctPtr` pointer to every value of the list. The `ApplyIf` method calls the `doFuncPtr` pointer to all values that satisfy the `ifFuncPtr` pointer:

[PRE21]

The `CopyIf` method copies the values satisfying the `ifFuncPtr` pointer into the `copyArray` method. The `RemoveIf` method removes every value satisfying the `ifFuncPtr` pointer:

[PRE22]

The `ApplyRemoveIf` method calls the `doFuncPtr` pointer and then removes every value satisfying the `ifFuncPtr` pointer, which comes in handy when we want to deallocate and remove pointers from the list:

[PRE23]

## Initialization

The constructor of the `DrawDocument` class is similar to the constructor of the `CircleDocument` class. We use the `LogicalWithScroll` coordinate system with US letter size. The file description `Draw Files` and the suffix `drw` are used to filter drawing files in the open and save dialogs. The null pointer indicates that the document does not have a parent window, and the `false` parameter indicates that the **Print** and **Print Preview** items in the **File** menu are omitted. Finally, the initiation lists holding the `DrawFormat` parameter indicates the format used to identify data to be copied and pasted. In this case, we use the same format for both copying and pasting:

**DrawDocument.cpp**

[PRE24]

Since we extend the `StandardDocument` framework, the window has a standard menu bar with the **File** menu holding **New**, **Open**, **Save**, **Save As**, and **Exit** (the **Print** and **Print Preview** items are omitted due to the `false` parameter in the constructor call) items, the **Edit** menu holding **Cut**, **Copy**, **Paste**, and **Delete**, and the **Help** items, and **About**.

We also add two application-specific menus: **Format** and **Add**. The **Format** menu holds the menu items **Modify**, **Color**, and **Fill**. Similar to the circle application, we mark the menu items with mnemonics and accelerators. However, we also use the enable parameters; the `ModifyEnable`, `ColorEnable`, and `FillEnable` methods are called before the menu items become visible. If they return `false`, the menu item is disabled and grayed:

[PRE25]

The **Add** menu holds one item for each kind of figure to be added:

[PRE26]

Finally, we read values from the **Windows Registry**, which is a database in the Windows system that we can use to store values between the executions of our applications. The Small Windows auxiliary classes `Color`, `Font`, `Point`, `Size`, and `Rect` have their own registry methods. The Small Windows `Registry` class holds static methods for reading and writing text as well as numerical and integer values:

[PRE27]

The destructor writes the values to the registry. In this application, it is not necessary to provide any common destructor actions such as deallocating memory or closing files:

[PRE28]

## Mouse input

`IsFigureMarked`, `IsFigureClicked`, and `UnmarkFigure` are callback functions that are called by the `DynamicList` methods `AnyOf`, `FirstOf`, `CopyIf`, `ApplyIf`, and `ApplyRemoveIf`. These methods take the pointer to a figure and an optional void pointer that holds additional information.

The `IsFigureMarked` function returns `true` if the figure is marked, the `IsFigureClicked` function returns `true` if the mouse point given in the `voidPtr` pointer hits the figure, and the `IsFigureClicked` function unmarks the figure if it is marked. As you can see, the `IsFigureMarked` function is defined as a lambda function, while the `IsFigureClicked` function is defined as a regular function.

There is no rational reason for this, other than that I would like to demonstrate both ways to define functions:

[PRE29]

In the `OnMouseDown` method, we first check that the user presses the left mouse button. If so, we save the mouse position in the `prevMousePoint` field so that we can calculate the distance the figure has moved in subsequent calls to the `OnMouseMove` method:

[PRE30]

As mentioned earlier, the mouse click will result in different actions depending on the value of the `actionMode` method. In case of the `Modify` method, we call the `FirstOf` parameter on the figure pointer list to extract the first clicked figure. The figures can overlap, and the click may hit more than one figure. In that case, we want the topmost figure located at the beginning of the list. The `FirstOf` method returns `true` if there is at least one clicked figure, which is copied into the `topClickedFigurePtr` reference parameter. The address of the `mousePoint` method is given as the second parameter to the `FirstOf` method and is, in turn, given to the `IsFigureClicked` function as its second parameter:

[PRE31]

We have two cases to consider, depending on whether the user presses the *Ctrl* key. If they do so, the figure will be marked if it is unmarked and vice versa, and other marked figures will remain marked.

However, in the other case, when the user does not press the *Ctrl* key, the figure becomes marked regardless of whether it is already marked, all other marked figures become unmarked, and the application is set to the `ModifySingle` mode. The figures are removed from the list and inserted at the beginning (front) in order to appear on top of the drawing:

[PRE32]

If the user presses the *Ctrl* key, we have another two cases. If the clicked figure is already marked, we unmark it and set the `applicationMode` method to the `Idle` mode. If the clicked figure is not already marked, we mark it and set the `applicationMode` method to the `MoveMultiple` mode. In this way, we have at least one marked figure to be moved in the `OnMouseMove` method when the user moves the mouse. Note that if the user presses the *Ctrl* key, one or several figures can be moved but not modified. It would be illogical to modify more than one figure at the same time:

[PRE33]

If the user hits a point where no figure is located (the `figurePtrList.FirstOf` method returns `false`), we unmark all marked figures, initialize the `insideRectangle` method, and set the `applicationMode` method to the `ModifyRectangle` mode.

[PRE34]

All the aforementioned cases in this method takes place when the `actionMode` method is `Modify`. However, it can also be `Add`, in which case a new figure will be added to the drawing. We use the `addFigureId` method to decide which kind of figure to add when calling the `CreateFigure` method. We set the dirty flag, since we have added a figure and the document has been modified. Finally, we add the address of the new figure to the beginning of the figure list (so that it appears on top) and set the `applicationMode` method to the `ModifySingle` mode:

[PRE35]

Depending on the action and modes, the window and cursor may need to be updated:

[PRE36]

The `MoveMarkFigure` method is a callback function that is called by the `Apply` method on `figurePtrList` in the `OnMouseMove` method. It moves the figure that is marked. The address of the moving distance is given in the `voidPtr` parameter:

[PRE37]

In the `OnMouseMove` method, we start by calculating the distance since the previous call to the `OnMouseDown` or `OnMouseMove` method. We also set the `prevMousePoint` method to the mouse position:

[PRE38]

Depending on the `applicationMode` method, we perform different tasks. In case of the `Modify` method on a single figure, we call the `MoveOrModify` method on that figure. The figure is placed at the beginning of the figure pointer list (`figurePtrList[0]`), since we placed it there in the `OnMouseDown` method. The idea is that the figure itself, depending on where the user clicked, decides whether it is moved or modified. The state of the figure is set when the user clicks on it, and depends on whether they click on any of the endpoints of the figure:

[PRE39]

In case of multiple movements, we move every marked figure the distance since the last mouse message. Note that we do not modify the figures in the multiple cases as we do in the single case:

[PRE40]

In the rectangle case, we set its bottom-right corner and redraw it:

[PRE41]

The `IsFigureInside` and `MarkFigure` methods are callback functions that are called by the `DynamicList` methods `CopyIf`, `RemoveIf`, and `Apply` on `figurePtrList` in the `OnMouseUp` method. The `IsFigureInside` method returns `true` if the figure is located inside the given rectangle, while the `MarkFigure` method simply marks the figure:

[PRE42]

In the `OnMouseUp` method, we only need to take the `ModifyRectangle` case into consideration. We need to decide which figures are totally enclosed by the rectangle. In order for them to appear on top of the drawing, we first call the `CopyIf` method on the `figurePtrList` list to temporarily copy the figures located completely inside the rectangle to the `insideList` list.

Then we remove the figures from the `figurePtrList` list and insert them from the `insideList` list at the beginning of the `figurePtrList` list. This makes them appear at the top of the drawing. Finally, we mark the figure inside the rectangle by calling `Apply` on the `insideList` list:

[PRE43]

After the user has released the left mouse button, the application holds the `Idle` mode, which it always holds as long as the user does not press the left mouse button:

[PRE44]

The `OnDoubleClick` method is called when the user double-clicks on the mouse button. The difference between a double-click and two consecutive clicks is decided by the Windows system, and can be adjusted in the Windows control panel. In case of a double-click, the `OnMouseDown` and `OnMouseUp` methods are called before the `OnDoubleClick` method. We extract the topmost clicked figure, if any, and call the `DoubleClick` method. The result depends on the type of figure: the head of an arrow is reversed, a rectangle or ellipse is filled if unfilled and vice versa, and a line is not affected at all:

[PRE45]

## Painting

In Small Windows, there are three general painting methods: `OnPaint`, `OnPrint`, and `OnDraw`. The Windows system indirectly calls the `OnPaint` and `OnPrint` methods for painting a window or printing a paper, respectively. Their default behavior is to call the `OnDraw` method. Remember that we do not take any initiatives to paint the window, we just wait for the right message. The idea is that in cases when we need to distinguish between painting and printing, we override the `OnPaint` and `OnPrint` methods, and when we do not need that distinction, we override the `OnDraw` method instead.

In the word processor, which is discussed later in this book, we will look into the difference between painting and printing. However, in this application, we just override the `OnDraw` method. As mentioned in [Chapter 3](ch03.html "Chapter 3. Building a Tetris Application"), *Building a Tetris Application*, the `Graphics` class reference is created by the framework and can be considered a toolbox equipped with pens and brushes. In this case, we just call the `DrawFigure` method for each figure with the `Graphics` reference as a parameter. In case of the `ModifyRectangle` mode, we also draw the rectangle:

[PRE46]

## The File menu

Thanks to the framework in the `StandardDocument` class, the file management is quite easy. The `ClearDocument` method is called when the user selects the **New** menu item, we just delete the figures and clear the figure list:

[PRE47]

The `WriteDocumentToStream` method is called when the user selects the **Save** or **Save As** menu item. It first writes the size of the figure list, and for each figure it writes its identity number (which is necessary when reading the figure in the `ReadDocumentFromStream` method shown as follows), and then writes the figure itself by calling its `WriteFigureToStream` method:

[PRE48]

The `ReadDocumentFromStream` method is called when the user selects the **Open** menu item. It starts by reading the number of figures in the figure list. We need to read the identity number for the next figure and call the `CreateFigure` method to receive a pointer to the created figure. Then we just call the `ReadFigureFromStream` method for the figure and add the figure's address to the figure pointer list:

[PRE49]

The `CreateFigure` method is called by the `ReadFigureFromStream` and `ReadFigureFromClipboard` method and creates a figure of the given type:

[PRE50]

## Cut, copy, and paste

Similar to the aforementioned file management case, the framework also takes care of the details of cut, copy, and paste. First, we do need to decide when the cut and copy menu items and accelerators will be enabled. In `Modify` mode, it is enough that at least one figure is marked. We use the `DynamicList` method `AnyOf` to decide whether at least one figure is marked. In `Add` mode, cut or copy is never allowed. We do not need to override the `CutEnable` method, since its default behavior in the `StandardDocument` framework is to call the `CopyEnable` method:

[PRE51]

There is a `PasteEnable` method in the `StandardDocument` framework. However, in this application we do not need to override it, since the framework decides when to enable pasting or, more specifically, when there is data on the global clipboard with the format code given in the `StandardDocument` constructor, in this case the `DrawFormat` field. The global clipboard is a Windows resource intended for short-term storing of information that has been copied.

The `CopyGeneric` method takes a list of characters that are intended to be filled with application-specific information. We save the number of marked figures, and for each marked figure, we write its identity number and call the `WriteFigureToClipboard` method, which writes the figure-specific information to the `infoList` parameter:

[PRE52]

The `PasteGeneric` method pastes the figures in a way similar to the aforementioned the `ReadDocumentFromStream` method:

[PRE53]

There is a `DeleteEnable` method in the `StandardDocument` framework, which we do not need to override since its default behavior is to call the `CopyEnable` method. The `OnDelete` method goes through the figure list, invalidating and deleting the marked figures. We use the `DynamicList` method `ApplyRemoveIf` to remove and delete marked figures.

We cannot simply use the `ApplyIf` and `RemoveIf` methods to deallocate and remove the figures, since it would result in memory errors (dangling pointers):

[PRE54]

## The Modify menu

The **Modify** menu item is quite easy to handle. It is enabled in case the application is in the `Idle` mode, which it is in when the user does not press the left mouse button. The radio button is also present if the `actionMode` method is `Modify`, and the menu item listener just sets the `actionMode` method to `Modify`:

[PRE55]

For the **Color** and **Fill** menu items, there are enable methods that are rather easy and listeners that are a little bit more complicated. It is possible to change the color in `Modify` mode if at least one figure is marked. In `Add` mode, it is always possible to change the color:

[PRE56]

The `SetFigureColor` method is a callback function that is called by the `ApplyIf` method on the `figurePtrList` list in the `OnColor` method:

[PRE57]

The `OnColor` method is called when the user selects the **Color** menu item. In `Modify` mode, we extract the marked figures and choose the color of the topmost of them. We know that at least one figure is marked, otherwise the preceding `ColorEnable` method would return `false` and the **Color** menu item would be disabled. If the `ColorDialog` call returns `true`, we set the new color of all marked figures by calling the `ApplyIf` method on the `figurePtrList` list:

[PRE58]

If the `actionMode` method is `Add`, we just display a color dialog to set the next color:

[PRE59]

The `IsFigureMarkedAndFilled` method is a callback function that is called by the `AnyOf` method on the `figurePtrList` list in the `FillCheck` method. The **Fill** menu item is checked with a radio mark if at least one figure is marked and filled:

[PRE60]

The `IsFigureMarkedAndFillable` method is a callback function that is called by the `AnyOf` method on the `figurePtrList` list in the `FillEnable` method. The **Fill** menu item is enabled if at least one fillable figure (rectangle or ellipse) is marked, or if the user is about to add a rectangle or ellipse:

[PRE61]

In order to test whether the figure type of the next figure to be added is fillable, we create and delete such a figure:

[PRE62]

The `InverseFill` method is a callback function that is called by the `AnyOf` method on the `figurePtrList` list in the `OnFill` method, which is called when the user selects the **Fill** menu item. The `OnFill` method inverts the fill status of all marked figures in `Modify` mode. In `Add` mode, it just inverts the value of `nextFill`, indicating that the next figure to be added will have the inverted fill status:

[PRE63]

## The Add menu

The listeners for the items of the `Add` menu are rather straightforward. The enable methods are simple, for the menu item to be enabled it is enough if the `applicationMode` method is in the `Idle` mode:

[PRE64]

The radio methods return `true` in `Add` mode if the figure to be added matches the figure of the radio method:

[PRE65]

Finally, the methods responding to the menu item and accelerator selections sets the `actionMode` to `Add` and the figure to be added:

[PRE66]

## The cursor

The `Set` method in the `Cursor` class sets the cursor to an appropriate value. If the application mode is `Idle` mode, we wait for the user to press the mouse button. In that case, we use the well-known arrow cursor image. If the user is in the process of enclosing figures with a rectangle, we use the cross-hair. If the user is in the process of moving several figures, we use the cursor with four arrows (size all). Finally, if they are in the process of modifying a single figure, the figure (whose address is located in the `figurePtrList[0]` list) itself is deciding which cursor to use:

[PRE67]

# Summary

In this chapter, you started the development of a drawing program capable of drawing lines, arrows, rectangles, and ellipses. In [Chapter 5](ch05.html "Chapter 5. The Figure Hierarchy"), *The Figure Hierarchy*, we will look into the figure hierarchy.