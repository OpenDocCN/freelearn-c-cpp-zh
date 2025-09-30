# Chapter 5. The Figure Hierarchy

This chapter introduces the figure classes of the drawing program. Each figure is responsible for deciding whether it is hit by a mouse click or if it is enclosed by a rectangle. It is also responsible for moving or modifying, as well as drawing and communicating with a file stream and the clipboard.

The drawing figure hierarchy is made up of the `Draw`, `LineFigure`, `ArrowFigure`, `RectangleFigure`, and `EllipseFigure` classes, as shown in the following image:

![The Figure Hierarchy](img/image_05_001.jpg)

# The DrawFigure class

The `Draw` class is the root class of the hierarchy and is mostly made up of virtual and pure virtual methods intended to be overridden by the subclasses.

The difference between a virtual method and a pure virtual method is that the virtual method has a body and it may be overridden by a subclass. If the subclass overrides the method, its version of the method is called.

If the subclass does not override the method, the method of the base class is called instead. A pure virtual method does not usually have a body, and a class holding at least one pure virtual method becomes abstract. The subclass can either override all the pure virtual methods of its base class or become abstract itself:

**Draw.h**

[PRE0]

Each figure has its own identity number, returned by the `GetId` method:

[PRE1]

The `IsClick` method returns `True` if the mouse point hits the figure, and the `IsInside` method returns `True` if the figure is completely enclosed by the area. The `DoubleClick` method gives the figure a possibility to perform a figure-specific action:

[PRE2]

The `Modify` and `Move` methods simply move the figure. However, the `Modify` method performs figure-specific actions defined by the `IsClick` method. If the user clicked on one of the figure endpoints, it will be modified, and if they clicked on any other part of the figure, it will be moved:

[PRE3]

The `Invalidate` method invalidates the figure by calling the `Area` method, which returns the area occupied by the figure. The `Draw` method draws the figure with the given `Graphics` class's reference:

[PRE4]

The `IsFillable`, `IsFilled`, and `Fill` methods are only overridden by the `Rectangle` and `Ellipse` methods:

[PRE5]

The `WriteFigureToStream` and `ReadFigureFromStream` methods are called when the user opens or saves a document. They write or read the information of the figure to and from the streams:

[PRE6]

The `WriteFigureToClipboard` and `ReadFigureFromClipboard` methods are called when the user copies or pastes figures. They write information to a character list and read information to a character buffer:

[PRE7]

The `color` and `marked` fields have their own get and set methods:

[PRE8]

The `GetCursor` method returns the correct cursor for the figure:

[PRE9]

The `MarkRadius` method is the size of the small squares showing that the figure is marked:

[PRE10]

The `windowPtr` pointer is used when invalidating the figure:

[PRE11]

Each figure, regardless of its type, has a color and is marked or unmarked:

[PRE12]

**Draw.cpp**

[PRE13]

The `MarkRadius` parameter is set to 100 * 100 units, which is 1 * 1 millimeters:

[PRE14]

When a figure is created, it is always unmarked.

[PRE15]

We redraw when the user toggles the figure's marked state. You may notice the different order in the `if...else` statements. The reason is that when we mark a figure, it becomes larger; that is why we first set the `marked` parameter to `True` and then invalidate the figure to catch its area including its markings. On the other hand, when we unmark a figure it becomes smaller; that is why we first invalidate it to catch its area, including the markings, and then set the `marked` parameter to `False`.

[PRE16]

The color is the only field written or read in file handling and in communication with the clipboard. The subclasses of the `DrawFigure` class call these methods and then write and read figure-specific information. The `WriteFigureToStream` and `ReadFigureFromStream` methods return the Boolean value of the stream to indicate whether the file operation succeeded.

[PRE17]

# The LineFigure class

A line is drawn between two points, represented by the `firstPoint` field to the `lastPoint` field in the `LineFigure` class, as shown in the following image:

![The LineFigure class](img/image_05_002.jpg)

The `header` file overrides some of the methods of its `DrawFigure` base class. The `DoubleClick` method does nothing. As I see it, there is no really meaningful response to a double-click on a line. However, we still need to override the `DoubleClick` method, since it is a pure virtual method in the `DrawFigure` base class. If we do not override it, the `LineFigure` class will be abstract.

**LineFigure.h**

[PRE18]

**LineFigure.cpp**

[PRE19]

The `SetFirstPoint` method is called when the line is created and sets both the first and last points.

[PRE20]

The `IsClick` method has two cases: the user has to hit either one of the endpoints or the line itself. We define two squares (`firstSquare` and `lastSquare`) covering the endpoints, and test whether the mouse hits one of them. If not, we test whether the mouse hits the line itself by calling the `IsPointInLine` method.

[PRE21]

The `IsPointInLine` method checks whether the point is located on the line, with some tolerance. We use trigonometric functions to calculate the position of the point relative to the line. However, if the line is completely vertical and the points have the same x coordinate, we have a special case.

Applying the trigonometric functions would result in division by zero. Instead, we create a small rectangle surrounding the line and check if the point is located in the rectangle, as shown in the following image:

![The LineFigure class](img/image_05_003.jpg)

[PRE22]

If the line is not vertical, we start by creating an enclosing rectangle and test if the mouse point is in it. If it is, we let the leftmost point of the `firstPoint` and `lastPoint` fields equal to the `minPoint` field and the rightmost point equal to the `maxPoint` field. Then we calculate the width (`lineWidth`) and height (`lineHeight`) of the enclosing rectangle, as well as the distance between the `minPoint` and `mousePoint` fields in x and y directions (`diffWidth` and `diffHeight`), as shown in the following image:

![The LineFigure class](img/image_05_004.jpg)

Due to uniformity, the following equation is true if the mouse point hits the line:

![The LineFigure class](img/image_05_005.jpg)

This implies that:

![The LineFigure class](img/image_05_006.jpg)

And this also implies that:

![The LineFigure class](img/image_05_007.jpg)

Let us allow for a small tolerance; let us say that the user is allowed to miss the line by a millimeter (100 units). This changes the last equation to the following:

![The LineFigure class](img/image_05_008.jpg)

[PRE23]

The `IsInside` method is easier than the `IsClick` method. We just check whether both endpoints are enclosed by the given rectangle.

[PRE24]

In the `Modify` mode, we move one of the endpoints or the line depending on the value of the `lineMode` parameter set by the `IsClick` method. If the user has hit the first point, we move it. If they have hit the last point, or if the line is in the process of being created, we move the last point. If they have hit the line, we move the line. That is, we move both the first and last points.

[PRE25]

The `Move` method is also easy; we just move the two endpoints.

[PRE26]

In the `Draw` method, we draw the line and, if the line is marked, its two endpoints are always black.

[PRE27]

The area occupied by the line is a rectangle with the endpoints as corners. If the line is marked, the mark radius is added.

[PRE28]

If the line is being modified, the `Crosshair` cursor is returned. If it is being moved, the size-all cursor (four arrows in the compass directions) is returned. If none of these cases apply, then we just return the normal arrow cursor.

[PRE29]

The `WriteFigureToStream`, `ReadFigureFromStream`, `WriteFigureToClipboard`, and `ReadFigureFromClipboard` methods write and read the first and last endpoints of the line after calling the corresponding methods in the `DrawFigure` class.

[PRE30]

# The ArrowFigure class

The `ArrowFigure` is a subclass of the `LineFigure` class and reuses the `firstPoint` and `lastPoint` fields and some of its functionality. The endpoints of the arrowhead are stored in the `leftPoint` and `rightPoint` fields, as shown in the following image. The lengths of the sides are defined by the `ArrowLength` constant to 500 units, which is 5 millimeters.

![The ArrowFigure class](img/image_05_010.jpg)

The `ArrowFigure` class overrides some of the methods of the `LineFigure` class. Mostly, it calls the methods of the `LineFigure` class and then adds functionality of its own.

**ArrowFigure.h**

[PRE31]

The constructors let the `LineFigure` constructors initialize the arrow's endpoints, and then call the `CalculateArrowHead` method to calculate the endpoints of the arrowhead.

**ArrowFigure.cpp**

[PRE32]

The `IsClick` method returns `True` if the user clicks on the line or any part of the arrowhead.

[PRE33]

The `IsInside` method returns `True` if all the endpoints of the line and arrowhead are inside the area.

[PRE34]

The `Modify` method modifies the line and recalculates the arrowhead.

[PRE35]

The `Move` method moves the line and the arrowhead.

[PRE36]

When the user double-clicks on the arrow, its head and tail are swapped.

[PRE37]

The `Area` method calculates the minimum and maximum of the line's and arrowhead's endpoints and returns an area with its top-left and bottom-right corners. If the arrow is marked, the mark radius is added to the area.

[PRE38]

The `Draw` method draws the line and the arrowhead. If the arrow is marked, the arrow's endpoints are also marked with squares.

[PRE39]

The `WriteFigureToStream`, `ReadFigureFromStream`, `WriteFigureToClipboard`, and `ReadFigureFromClipboard` methods let the `LineFigure` class write and read the line's endpoints. Then it writes and reads the arrowhead's endpoints.

[PRE40]

The `CalculateArrowHead` method is a private auxiliary method that calculates the endpoints of the arrowhead. We will use the following relations to calculate the `leftPoint` and `rightPoint` fields.

![The ArrowFigure class](img/image_05_011.jpg)

The calculation is performed in three steps; first we calculate `alpha` and `beta`. See the following illustration for the definition of the angles:

![The ArrowFigure class](img/image_05_012.jpg)

Then we calculate `leftAngle` and `rightAngle` and use their values to calculate the value of `leftPoint` and `rightPoint`. The angle between the line and the arrowhead parts is 45 degrees, which is equivialent to Π/4 radians. So, in order to determine the angles for the arrowhead parts, we simply subtract Π/4 from `beta` and add Π/4 to `beta`:

![The ArrowFigure class](img/image_05_013.jpg)

Then we use the following formulas to finally determine `leftPoint` and `rightPoint`:

![The ArrowFigure class](img/image_05_014.jpg)

The trigonometric functions are available in the C standard library. However, we need to define our value for Π. The `atan2` function calculates the tangent value for the quota of `height` and `width` and takes into consideration the possibility that `width` might be zero.

![The ArrowFigure class](img/image_05_018.jpg)

[PRE41]

# The RectangleFigure class

The `RectangleFigure` class holds a rectangle, which can be filled or unfilled. The user can modify it by grabbing one of its four corners. The `DrawRectangle` class overrides most of the methods of the `DrawFigure` class.

One difference compared to the line and arrow cases is that a rectangle is two-dimensional and can be filled or unfilled. The `Fillable` method returns `True` and the `IsFilled` and `Fill` methods are overridden. When the user double-clicks on a rectangle it will be toggled between the filled and unfilled states.

**RectangleFigure.h**

[PRE42]

**RectangleFigure.cpp**

[PRE43]

When the user clicks on the rectangle, they may hit one of its four corners, the borders of the rectangle, or (if it is filled) its interior. First, we check the corners and then the rectangle itself. If it is filled, we just test whether the mouse point is enclosed in the rectangle. If the rectangle is unfilled, we test whether any of its four borders has been hit by constructing a slightly smaller rectangle and a slightly larger one. If the mouse position is included in the larger rectangle, but not in the smaller one, the user has hit one of the rectangle borders.

![The RectangleFigure class](img/image_05_019.jpg)

[PRE44]

The `IsInside` method returns `true` if the top-left and bottom-right corners are enclosed by the rectangle area.

[PRE45]

The `DoubleClick` method fills the rectangle if it is unfilled and vice versa.

[PRE46]

The `Modify` method modifies or moves the rectangle in accordance with the setting of the `rectangleMode` parameter in the `IsClick` method.

[PRE47]

The `Move` method moves the rectangle's corners.

[PRE48]

The area of the rectangle is simply that of the rectangle. However, if it is marked, we increase it in order to include the corner squares.

![The RectangleFigure class](img/image_05_020.jpg)

[PRE49]

The `Draw` method draws or fills the rectangle. It also fills the squares if it is marked.

[PRE50]

The cursor of the rectangle is the size-all cursor (arrows in the four compass directions) when the figure is being moved. It is a cursor with arrows in accordance with the grabbed corner while being modified: north-west and south-east arrows in the case of the top-left or bottom-right corner, and north-east and south-west arrows in the case of the top-right or bottom-left corner.

[PRE51]

The `WriteFigureToStream`, `ReadFigureFromStream`, `WriteFigureToClipboard`, and `ReadFigureFromClipboard` methods call the corresponding methods in the `DrawFigure` class. Then they write and read the four corners of the rectangle, and whether it is filled or not.

[PRE52]

# The EllipseFigure class

The `EllipseFigure` class is a subclass of the `RectangleFigure` class. The ellipse can be moved or reshaped by the horizontal or vertical corners. Most of the methods from the `RectangleFigure` class are not overridden by the `Ellipse` class.

**Ellipse.h**

[PRE53]

**Ellipse.cpp**

[PRE54]

Just as in the rectangle case, the `IsClick` method first decides if the user has clicked on one of the four endpoints; however, the positions are different compared to the rectangle corners.

![The EllipseFigure class](img/image_05_021.jpg)

[PRE55]

If the user has not clicked on one of the modifying positions, we have to decide if the user has clicked on the ellipse itself. It is rather easy if the ellipse is not filled. We create an elliptic region by using the Win32 API function `CreateEllipticRgn` and test if the mouse position is in it. If the ellipse is not filled, we create two regions, one slightly smaller and one slightly larger. If the mouse position is included in the larger region, but not in the smaller one, we have a hit.

![The EllipseFigure class](img/image_05_022.jpg)

[PRE56]

The `Modify` method moves the corner in accordance with the setting of the `ellipseMode` parameter in the `IsClick` method.

[PRE57]

The `Draw` method fills or draws the ellipse, and the four squares if the ellipse is marked.

[PRE58]

Finally, when it comes to the cursor, we have the following five different cases:

*   When the ellipse is being created, the crosshair is returned
*   When the user grabs the left or right endpoint of the ellipse, the west-east (left-right) arrow is returned
*   When the user grabs the top or bottom endpoint, the top-bottom (up-down) arrow is returned
*   When the user moves the ellipse, the size-all arrow (four arrows that point left, right, up, and down) is returned
*   Finally, when the user neither moves nor modifies the ellipse, the normal arrow cursor is returned

[PRE59]

# Summary

In this chapter, you studied the figure class hierarchy for the drawing program of [Chapter 4](ch04.html "Chapter 4. Working with Shapes and Figures"), *Working with Shapes and Figures*. You covered the following topics:

*   Testing whether the figure has been hit by a mouse click or if it is enclosed by a rectangle
*   Modification and movement of the figure
*   Drawing the figure and calculating the area of the figure
*   Writing and reading the figure to and from a file stream or the clipboard
*   Cursor handling with different cursors depending on the current state of figure

In [Chapter 6](ch06.html "Chapter 6. Building a Word Processor"), *Building a Word Processor*, you will start developing a word processor.