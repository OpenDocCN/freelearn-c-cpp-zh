# Chapter 6. Graphics View

> *Widgets are great for designing graphical user interfaces. However, you will run into problems if you wish to animate multiple widgets at the same time by constantly moving them around in the application. For these situations, or in general for frequently transforming 2D graphics, Qt offers you Graphics View. In this chapter, you will learn the basics of the Graphics View architecture and its items. You also will learn how to combine widgets with Graphics View items. Once you have acquired a basic understanding, we are next going to develop a simple jump-and-run game illustrating how to animate the items. Finally, we'll look into some possibilities for optimizing Graphics View's performance.*

# Graphics View architecture

Three components form the core of Graphics View: an instance of `QGraphicsView`, which is referred to as **view**; an instance of `QGraphicsScene`, which is referred to as **scene**; and usually multiple instances of `QGraphicsItem`, which are referred to as **items**. The usual workflow is to first create a couple of items, then add them to a scene, and finally set that scene on a view.

In the following section, we will be discussing all three parts of the Graphics View architecture one after the other, beginning with the items, followed by the scene, and concluding with the view.

![Graphics View architecture](img/8874OS_06_01.jpg)

An illustration of Graphics View components

However, because it is not possible to deal with one component as entirely separate from the others, you need to get the big picture up front. This will help you to better understand the description of the three single parts. And do not worry if you do not understand all the details on their first occurrence. Be patient, work through the three parts, and all issues will hopefully become clear in the end.

Think of the items as Post-it notes. You take a note and write a message on it, paint an image on it, both write and paint on it or, quite possibly, just leave it blank. This is equivalent to creating an item with a defined paint function, whether it is a default one or you have customized it. Since the items do not have a predetermined size, you define a bounding rectangle inside which all the painting of the item is done. As with a note, which does not care where it is positioned or from which angle it is being looked at, the item always draws its content as if it were in an untransformed state, where a length unit corresponds to 1 pixel. The item exists in its own coordinate system. Although you can apply various transformations to the item, such as rotating and scaling, it's not the job of the item's paint function to take that into account; that's the scene's job.

What is the scene, then? Well, think of it as a larger sheet of paper onto which you attach your smaller Post-its, that is, the notes. On the scene, you can freely move the items around while applying funny transformations to them. It is the scene's responsibility to correctly display the items' position and any transformations applied to them. The scene further informs the items about any events that affect them and it has—as with the items—a bounding rectangle within which the items can be positioned.

Last but not least, let's turn our attention to the view. Think of the view as an inspection window or a person who holds the paper with the notes in their hands. You can watch the paper as a whole or you can only look at specific parts. And as a person can rotate and shear the paper with their hands, so the view can rotate and shear the scene and do a lot more transformations with it.

### Note

You may look at the preceding diagram and be worried about all the items being outside the view. Aren't they wasting GPU render time? Don't you need to take care of them by adding a so-called "view frustum culling" mechanism (to detect which item not to draw/render because it is not visible)? Well, the short answer is "no" because Qt is already taking care of this.

## Items

So, let's look at the items. The most fundamental characteristic of items in Graphics View is their object-oriented approach. All items in the scene must inherit `QGraphicsItem`, which is an abstract class with—amongst numerous other public functions—two pure virtual functions called `boundingRect()` and `paint()`. Because of this simple and clear fact, there are principles which apply to each item.

### Parent child relationship

The constructor of `QGraphicsItem` takes a pointer to another item that is set as the item's parent. If the pointer is `0`, the item has no parent. This gives you the opportunity to organize items in a tree structure similar to the `QObject` object even though the `QGraphicsItem` element does not inherit from the `QObject` object. You can change the relationship of items at any given time by calling the `setParentItem()` function. It takes the new parent as an argument. If you want to remove a child item from its parent, simply call the `setParentItem(0)` function on the child. The following code illustrates both possibilities for creating a relationship between items. (Please note that this code will not compile since `QGraphicsItem` is an abstract class. Here, it is just for the purpose of illustration, but it will work with a real item class.)

[PRE0]

First we create an item called `parentItem`, and since we do not use the constructor's argument, the item has no parent or child. Next, we create another item called `firstChildItem` and pass a pointer to the `parentItem` item as an argument. Thus, it has the `parentItem` item as its parent, and the `parentItem` item now has the `firstChildItem` item as its child. Next we create a third item called `secondChildItem`, but since we do not pass anything to its constructor, it has no parent at this point. In the next line, however, we change that by calling the `setParentItem()` function. Now it is also a child of the `parentItem` item.

### Tip

You can always check whether an item has a parent using the `parentItem()` function and check the returned `QGraphicsItem` pointer against `0`, which means that the item does not have a parent. To figure out if there are any children, call the `childItems()` function on the item. A `QList` method with the `QGraphicsItem` pointers to all child items is returned.

![Parent child relationship](img/8874OS_06_21.jpg)

The parent-child relationship

The benefit of this parent-child relationship is that specific actions performed on a parent item also affect associated child items. For example, when you delete a parent item, all child items will also be deleted. For that reason, it is sufficient to delete the `parentItem` item in the preceding code. The destructors of the `firstChildItem` and `secondChildItem` items are called implicitly. The same applies when you add or remove a parent item from a scene. All child items will then get added or removed as well. The same applies when you hide a parent item or when you move a parent item. In both cases, the child items will behave the same way the parent does. Think of the earlier example of Post-it notes; they would behave the same. If you have a note with other notes attached to it, they will also move when you move the parent note.

### Tip

If you are not sure whether a function call on the parent item is propagated to its child items, you can always have a look at the sources. You will find them in your Qt installation if you checked the option to also install the sources at the time of installation. You can also find them online at [https://github.com/qtproject/qtbase](https://github.com/qtproject/qtbase).

Even if there isn't a meaningful comment, you can spot the relevant code easily. Just look for a `children` variable addressed through the d-pointer. Inside the destructor of the `QGraphicsItem` item, the relevant code fragment is as follows:

[PRE1]

### Appearance

You are probably wondering what a `QGraphicsItem` item looks like. Well, since it is an abstract class (and unfortunately the paint function is a pure virtual one), it does not look like anything. You will have to do all the painting yourself. Luckily, since the paint function of the `QGraphicsItem` item offers you a technique you already know, the `QPainter` pointer, this is not very difficult.

Don't panic! You don't have to draw all items yourself though. Qt offers a lot of standard shaped items you can use just out-of-the-box. You'll find them discussed in an upcoming section titled *Standard items*. However, since we need to draw a custom item once in a while, we go through this process.

# Time for action – creating a black, rectangular item

As a first approach, let's create an item that paints a black rectangle:

[PRE2]

## *What just happened?*

First, we subclass `QGraphicItem` and call the new class `BlackRectangle`. The class' constructor accepts a pointer to a `QGraphicItem` item. This pointer is then passed to the constructor of the `QGraphicItem` item. We do not have to worry about it; `QGraphicItem` will take care of it and establish the parent-child relationship for our item, among other things. Next, the virtual destructor makes sure that it gets called even if the class is getting deleted through a base class pointer. This is a crucial point, as you will learn later when we talk about the scene.

Next, we define the `boundingRect()` function of our item, where we return a rectangle 75 pixels wide and 25 pixels high. This returned rectangle is the canvas for the `paint` method and simultaneously the promise to the scene that the item will only paint in this area. The scene relies on the correctness of that information, so you should strictly obey that promise. Otherwise, the scene will become cluttered up with relics of your drawing!

Lastly, we do the actual painting from `QPainter` in conjunction with a `QWidget` item. There is nothing different here except that the painter is already initialized with the appropriate values given to us through the first argument. Even if it is not needed, I would suggest that the painter be kept in the same state at the end of the function as it was in the beginning. If you follow that advice, and if you only use custom items in the scene, you can later optimize the render speed enormously. This especially applies to scenes with many items. But let us go back to what we were actually doing. We have taken the painter and called the `fillRect()` function, which does not touch the painter's internal state. As arguments, we used the `boundingRect()` function, which defines the area to fill, and the `Qt::black` parameter, which defines the fill color. Thus, by only filling the bounding rectangle of the item, we obeyed the bounding rectangle promise.

In our example, we have not used the two other arguments of the `paint` function. To suppress the compiler warnings about unused variables, we used Qt's `Q_UNUSED` macro.

# Time for action – reacting to an item's selection state

The assigned pointer to a `QStyleOptionGraphicsItem` item might become handy if you want to alter the appearance of the item related to its state. For example, say you want to fill the rectangle with red when it gets selected. To do so, you only have to type this:

[PRE3]

## *What just happened?*

The `state` variable is a bitmask holding the possible states of the item. You can check its value against the values of the `QStyle::StateFlag` parameter by using bitwise operators. In the preceding case, the `state` variable is checked against the `State_Selected` parameter. If this flag is set, the rectangle is painted red.

### Tip

The type of state is `QFlags<StateFlag>`. So, instead of using the bitwise operator to test if a flag is set, you can use the convenient function `testFlag()`. Used with the preceding example it would be as follows:

[PRE4]

The most important states you can use with items are described in the following table:

| State | Description |
| --- | --- |
| `State_Enabled` | Indicates that the item is enabled. If the item is disabled, you may want to draw it as grayed out. |
| `State_HasFocus` | Indicates that the item has the input focus. To receive this state, the item needs to have the `ItemIsFocusable` flag set. |
| `State_MouseOver` | Indicates that the cursor is currently hovering over the item. To receive this state the item needs to have the `acceptHoverEvents` variable set to `true`. |
| `State_Selected` | Indicates that the item is selected. To receive this state, the item needs to have the `ItemIsSelectable` flag set. The normal behavior would be to draw a dashed line around the item as a selection marker. |

Besides the state, `QStyleOptionGraphicsItem` offers much more information about the currently used style, such as the palette and the font used, accessible through the `QStyleOptionGraphicsItem::palette` and `QStyleOptionGraphicsItem::fontMetrics` parameters, respectively. If you aim for style-aware items, have a deeper look at this class in the documentation.

# Time for action – making the item's size definable

Let's push the example of the black rectangle a step further. So far, `BlackRectangle` draws a fixed rectangle of size 75 x 25 pixels. It would be nice if one could define this size, so let us add the ability to define the size of the rectangle. Remember, only painting the rectangle larger does not help here because then you would break the promise regarding the bounding rectangle. So we need also to change the bounding rectangle as follows:

[PRE5]

## *What just happened?*

Since the destructor and the `paint` function are unchanged, they are omitted. What exactly have we done here? First, we introduced a private member called `m_rect` to save the current rectangle's value. In the initialization list, we set `m_rect` to a default value of `QRectF(0, 0, 75, 25)` like we hard-coded it in the first example. Since the bounding rectangle should be the same as `m_rect`, we altered `boundingRect()` to return `m_rect`. The same value is returned by the getter function `rect()`. For now it seems redundant to have two functions returning the same value, but as soon as you draw a border around the rectangle, you need to return a different bounding rectangle. It needs to be increased by the used pen's width. Therefore, we leave this redundancy in place in order to make further improvements easier. The last new part is the setter function, which is pretty standard. We check if the value has changed, and if not we exit the function. Otherwise, we set a new value, but this has to happen after the `prepareGeometryChange()` call. This call is important to inform the scene about the coming geometry change. Then, the scene will ask the item to redraw itself. We do not need to handle that part.

## Have a go hero – customizing the item

As an exercise, you can try to add an option to change the background color. You can also create a new item that allows you to set an image. If doing so, keep in mind that you have to change the item's bounding rectangle according to the size of the image.

## Standard items

As you have seen, creating your own item involves some work, but overall it is not that difficult. A big advantage is that you can use `QPainter` to draw the item, the same technique you use to paint widgets. So there is nothing new you need to learn. Indeed, even if it is easy to draw filled rectangles or any other shape, it is a lot of work to subclass `QGraphicsItem` each time you need to create an item that does such basic tasks. And that's the reason why Qt comes with the following standard items that make your life as a developer much easier:

| Standard item | Description |
| --- | --- |
| `QGraphicsLineItem` | Draws a simple line. You can define the line with `setLine(const QLineF&)`. |
| `QGraphicsRectItem` | Draws a rectangle. You can define the rectangle's geometry with `setRect(const QRectF&)`. |
| `QGraphicsEllipseItem` | Draws an ellipse. You can define the rectangle within which the ellipse is being drawn with `setRect(const QRectF&)`. Additionally, you can define whether only a segment of the ellipse should be drawn by calling `setStartAngle(int)` and `setSpanAngle(int)`. The arguments of both functions are in 16ths of a degree. |
| `QGraphicsPolygonItem` | Draws a polygon. You can define the polygon with `setPolygon(const QPolygonF&)`. |
| `QGraphicsPathItem` | Draws a path. You can define the path with `setPath(const QPainterPath&)`. |
| `QGraphicsSimpleTextItem` | Draws a simple text path. You can define the text with `setText(const QString&)` and the font with `setFont(const QFont&)`. This item is only for drawing *plain* text without any modification. |
| `QGraphicsTextItem` | Draws text. Unlike `QGraphicsSimpleTextItem`, this item can display HTML or render a `QTextDocument` element. You can set HTML with `setHtml(const QString&)` and the document with `setDocument(QTextDocument*)`. `QGraphicsTextItem` can even interact with the displayed text so that text editing or URL opening is possible. |
| `QGraphicsPixmapItem` | Draws a pixmap. You can define the pixmap with `setPixmap(const QPixmap&)`. |

Since the drawing of these items is done by a `QPainter` pointer you can also define which pen and which brush should be used. The pen is set with `setPen(const QPen&)` and the brush with `setBrush(const QBrush&)`. These two functions, however, do not exist for `QGraphicsTextItem` and `QGraphicsPixmapItem`. To define the appearance of a `QGraphicsTextItem` item you have to use `setDefaultTextColor()` or HTML tags supported by Qt. Note that pixmaps usually do not have a pen or a brush.

### Tip

Use `QGraphicsSimpleTextItem` wherever possible and try to avoid `QGraphicsTextItem` if it is not absolutely necessary. The reason is that `QGraphicsTextItem` lugs a `QTextDocument` object around and it is, besides being a subclass of `QGraphicsItem`, also a subclass of `QObject`. This is definitely too much overhead and has too high a performance cost for displaying simple text.

A word on how you set up items. Instead of writing two expressions, one for the initialization of the item and one for setting up its key information such as the rectangle for a `QGraphicsRextItem` item or the pixmap for a `QGraphicsPixmapItem`, almost all standard items offer you the option to pass that key information as a first argument to their constructors—besides the optional last argument for setting the item's parent. Say you would have written the following code:

[PRE6]

You can now simply write this:

[PRE7]

You can even just write this:

[PRE8]

This is very convenient, but keep in mind that compact code may be harder to maintain than code that sets all variables through setter methods.

## Coordinate system of the items

A last but very important note on the used coordinate system. Altogether, Graphics View deals with three different but connected coordinate systems. There is the item's coordinate system, the scene's coordinate system, and the view's coordinate system. All three coordinate systems differ from the Cartesian coordinate systems regarding the *y* axis: in Graphics View, like in `QPainter` pointer's coordinate system, the *y* axis is orientated and measured from the origin to the bottom. This means that a point below the origin has a positive *y* value. For now, we only care about the item's coordinate system. Since Graphics View is for 2D graphics, we have an *x* coordinate and a *y* coordinate with the origin at (0, 0). All points, lines, rectangles, and so on are specified in the item's own coordinate system. This applies to almost all occasions where you deal with values representing coordinates within the `QGraphicsItem` class or its derived classes. If you define, for example, the rectangle of a `QGraphicsRectItem` item, you use item coordinates. If an item receives a mouse press event, `QGraphicsSceneMouseEvent::pos()` is expressed in item coordinates. But there are some easy-to-identify exceptions to this statement. The return value of `scenePos()` and `sceneBoundingRect()` is expressed in scene coordinates. Pretty obvious, isn't it? The one thing that is a little bit tricky to identify is the returned `QPointF` pointer of `pos()`. The coordinates of this point are expressed in the item's parent coordinate system. This can be either the parent item's coordinate system or, more likely, the scene's coordinate system when the item does not have a parent item.

For a better understanding of `pos()` and the involved coordinate systems, think of Post-it notes again. If you put a note on a larger sheet of paper and then had to determine its exact position, how would you do it? Probably somewhat like this: "The note's upper left corner is positioned 3 cm to the right and 5 cm to the bottom from the paper's top left edge". In the Graphics View world, this would correspond to a parentless item whose `pos()` function returns a position in scene coordinates since the item's origin is directly pinned to the scene. On the other hand, say you put a note A on top of a (larger) note B, which is already pinned on a paper, and you have to determine A's position; how would you describe it this time? Probably by saying that note A is placed on top of note B or "2 cm to the right and 1 cm to the bottom from the top-left edge of note B". You most likely wouldn't use the underlying paper as a reference since it is not the next point of reference. This is because, if you move note B, A's position regarding the paper will change whereas A's relative position to B still remains unchanged. To switch back to Graphics View, the equivalent situation is an item that has a parent item. In this case, the `pos()` function's returned value is expressed in the coordinate system of its parent. So `setPos()` and `pos()` specify the position of the item's origin in relation to the next (higher) point of reference. This could be the scene or the item's parent item.

Keep in mind, however, that changing an item's position does not affect the item's internal coordinate system.

# Time for action – creating items with different origins

Let's have a closer look at these three items defined by the following code snippet:

[PRE9]

## *What just happened?*

All three items are rectangles with a side length of 20 pixels. The difference between them is the position of their coordinate origin points. `itemA` has its origin in the center of the rectangle, `itemB` has its origin in the top-left corner of the rectangle, and `itemC` has its origin outside the drawn rectangle. In the following diagram, you see the origin points marked as red dots.

![What just happened?](img/8874OS_06_11.jpg)

So what's the deal with these origin points? On the one hand, the origin point is used to create a relation between the item's coordinate system and the scene's coordinate system. As you will see later in more detail, if you set the position of the item on the scene, the position on the scene is the origin of the item. You can say scene *(x, y) = item(0, 0)*. On the other hand, the origin point is used as a center point for all transformations you can use with items, such as scaling, rotating, or adding a freely definable transformation matrix of `QTransform` type. As an additional feature, you always have the option to combine a new transformation with the already applied ones or to replace the old transformation(s) with a new one.

# Time for action – rotating an item

As an example, let's rotate `itemB` and `itemC` by 45 degrees counter-clockwise. For `itemB`, the function call would look like this:

[PRE10]

The `setRotation()` function accepts `qreal` as the argument value, so you can set very precise values. The function interprets the number as degrees for a clockwise rotation around the *z* coordinate. If you set a negative value, a counter-clockwise rotation is performed. Even if it does not make much sense, you can rotate an item by 450 degrees, which would result in a rotation of 90 degrees. Here is what the two items would look like after the rotation by 45 degrees counter-clockwise:

![Time for action – rotating an item](img/8874OS_06_14.jpg)

## *What just happened?*

As you can see, the rotation has its center in the item's origin point. Now you could run into the problem that you want to rotate the rectangle of `itemC` around its center point. In such a situation, you can use `setTransformOriginPoint()`. For the described problem, the relevant code would look like this:

[PRE11]

Let us take this opportunity to recapitulate the item's coordinate system. The item's origin point is in (0, 0). In the constructor of `QGraphicsRectItem`, you define that the rectangle should have its top-left corner at (10, 10). And since you gave the rectangle a width and a height of 20 pixels, its bottom-right corner is at (30, 30). This makes (20, 20) the center of the rectangle. After setting the transformation's origin point to (20, 20), you rotate the item around that point 45 degrees counter-clockwise. You will see the result in the following image, where the transformation's origin point is marked with a cross.

![What just happened?](img/8874OS_06_15.jpg)

Even if you "change" the item's origin point by such a transformation, this does not affect the item's position on the scene. First, the scene positions the untransformed item with respect to its origin point and only then are all transformations applied to the item.

## Have a go hero – applying multiple transformations

To understand the concept of transformations and their origin point, go ahead and try it yourself. Apply `rotate()` and `scale()` sequentially to an item. Also, change the point of origin and see how the item will react. As a second step, use `QTransform` in conjunction with `setTransform()` to add a custom transformation to an item.

## Scenes

Let us take a look at how we can improvise the scene.

### Adding items to the scene

At this point, you should have a basic understanding of items. The next question is what to do with them. As described earlier, you put the items on a `QGraphicsScene` method. This is done by calling `addItem(QGraphicsItem *item)`. Did you notice the type of the argument? It's a pointer to a `QGraphicsItem` method. Since all items on the scene must inherit `QGraphicsItem`, you can use this function with any item, be it a `QGraphicsRectItem` item or any custom item. If you have a look at the documentation of `QGraphicsScene`, you will notice that all functions returning items or dealing with them expect pointers to a `QGraphicsItem` item. This universal usability is a huge advantage of the object-orientated approach in Graphics View.

### Tip

If you have a pointer of the type `QGraphicsItem` pointing to an instance of a `QGraphicsRectItem` and you want to use a function of `QGraphicsRectItem`, use `qgraphicsitem_cast<>()` to cast the pointer. This is because it is safer and faster than using `static_cast<>()` or `dynamic_cast<>()`.

[PRE12]

Please note that if you want to use `qgraphicsitem_cast<>()` with your own custom item, you have to make sure that `QGraphicsItem::type()` is reimplemented and that it returns a unique type for a particular item. To ensure a unique type, use `QGraphicsItem::UserType + x` as a return value where you count up `x` for every custom item you create.

# Time for action – adding an item to a scene

Let's have a first try and add an item to the scene:

[PRE13]

## *What just happened?*

Nothing complicated here. You create a scene, create an item of type `QGraphicsRectItem`, define the geometry of the item's rectangle, and then set the item to the scene by calling `addItem()`. Pretty straightforward. But what you do not see here is what this implies for the scene. The scene is now responsible for the added item! First of all, the ownership of the item is transferred to the scene. For you, this means that you do not have to worry about freeing the item's memory because deleting the scene also deletes all items associated with the scene. Now remember what we said about the destructor of a custom item: it must be virtual! `QGraphicsScene` operates with pointers to `QGraphicsItem`. Thus, when it deletes the assigned items, it does that by calling `delete` on the base class pointer. If you have not declared the destructor of the derived class virtual, it will not be executed, which may cause memory leaks. Therefore, form habit of declaring the destructor virtual.

Transferring the ownership of the item to the scene also means that an item can only be added to one single scene. If the item was previously already added to another scene, it gets removed from there before it will be added to the new scene. The following code will demonstrate that:

[PRE14]

After creating two scenes and one item, we add the item `item` to the scene `firstScene`. Then, with the debug message, we print out the number of associated items with that `firstScene` scene. For this, we call `items()` on the scene, which returns a `QList` list with pointers to all items of the scene. Calling `count()` on that list tells us the size of the list, which is equivalent to the number of added items. As you can see after adding the item on `secondScene`, the `firstScene` item count returns `0`. Before `item` was added to `secondScene`, it was first removed from `firstScene`.

### Tip

If you want to remove an item from a scene without setting it directly to another scene or without deleting it, you can call `removeItem()`, which takes a pointer for the item that should be removed. Be aware, however, that now it is your responsibility to delete the item in order to free the allocated memory!

## Interacting with items on the scene

When it takes ownership of an item, the scene also has to take care of a lot of other stuff. The scene has to make sure that events get delivered to the right items. If you click on a scene (to be precise, you click on a view that propagates the event to the scene), the scene receives the mouse press event and it then becomes the scene's responsibility to determine which item was meant by the click. In order to be able to do that, the scene always needs to know where all the items are. Therefore, the scene keeps track of the items in a Binary Space Partitioning tree.

You can benefit from this knowledge too! If you want to know which item is shown at a certain position, call `itemAt()` with `QPointF` as an argument. You will receive the topmost item at that position. If you want all items that are located at this position, say in cases where multiple items are on top of each other, call an overloaded function of `items()` (which takes a `QPointF` pointer as an argument). It will return a list of all items that the bounding rectangle contains that point. The `items()` function also accepts `QRectF`, `QPolygonF`, and `QPainterPath` as arguments if you need all visible items of an area. With the second argument of the type `Qt::ItemSelectionMode`, you can alter the mode for how the items in the area will be determined. The following table shows the different modes:

| Mode | Meaning |
| --- | --- |
| `Qt::ContainsItemShape` | The item's shape must be completely inside the selection area. |
| `Qt::IntersectsItemShape` | Similar to `Qt::ContainsItemShape` but also returns items whose shapes intersect with the selection area. |
| `Qt::ContainsItemBoundingRect` | The item's bounding rectangle must be completely inside the selection area. |
| `Qt::IntersectsItemBoundingRect` | Similar to `Qt::ContainsItemBoundingRect` but also returns items whose bounding rectangles intersect with the selection area. |

The scene's responsibility for delivering events does not only apply to mouse events; it also applies to key events and all other sorts of events. The events that are passed to the items are subclasses of `QGraphicsSceneEvent`. Thus, an item does not get a `QMouseEvent` event like widgets; it gets a `QGraphicsSceneMouseEvent` event. In general, these scene events behave like normal events, but instead of say a `globalPos()` function you have `scenePos()`.

The scene also handles the selection of items. To be selectable, an item must have the `QGraphicsItem::ItemIsSelectable` flag turned on. You can do that by calling `QGraphicsItem::setFlag()` with the flag and `true` as arguments. Besides that, there are different ways to select items. There is the item's `QGraphicsItem::setSelected()` function, which takes a `bool` value to toggle the selection state on or off, or you can call `QGraphicsScene::setSelectionArea()` on the scene, which takes a `QPainterPath` parameter as argument, in which case all items get selected. With the mouse, you can click on an item to select or deselect it or—if the view's rubber-band selection mode is enabled—you can select multiple items with that rubber band.

### Note

For activating the rubber band selection for the view, call `setDragMode` `(QGraphicsView::RubberBandDrag)` on the view. Then you can press the left mouse button and, while holding it down, move the mouse to define the selection area. The selection rectangle is then defined by the point of the first mouse press and the current mouse position.

With the scene's `QGraphicsScene::selectedItems()` function, you can query the actual selected items. The function returns a `QList` list holding `QGraphicsItem` pointers to selected items. For example, calling `QList::count()` on that list would give you the number of selected items. To clear the selection, call `QGraphicsScene::clearSelection()`. To query the selection state of an item, use `QGraphicsItem::isSelected()`, which returns `true` if the item is selected and `false` otherwise. If you write a customized `paint` function, do not forget to alter the item's appearance to indicate that it is selected. Otherwise, the user cannot know this. The determination inside the paint function is done by `QStyle::State_Selected`, as shown earlier.

![Interacting with items on the scene](img/8874OS_06_10.jpg)

The standard items show a dashed rectangle around a selected item.

The item's handling of focus is done in a similar way. To be focusable an item must have the `QGraphicsItem::ItemIsFocusable` flag enabled. Then, the item can be focused by a mouse click, through the item's `QGraphicsItem::setFocus()` function, or through the scene's `QGraphicsScene::setFocusItem()` function, which expects a pointer to the item you like to focus as a parameter. To determine if an item has focus, you again have two possibilities. One is that you can call `QGraphicsItem::hasFocus()` on an item, which returns `true` if the item has focus or `false` otherwise. Alternatively, you can get the actual focused item by calling the scene's `QGraphicsScene::focusItem()` method. On the other hand, if you call the item's `QGraphicsItem::focusItem()` function, the focused item is returned if the item itself or any descendant item has focus; otherwise, `0` is returned. To remove focus, call `clearFocus()` on the focused item or click somewhere in the scene's background or on an item that cannot get focus.

### Tip

If you want a click on the scene's background not to cause the focused item to lose its focus, set the scene's `stickyFocus` property to `true`.

## Rendering

It is also the scene's responsibility to render itself with all the assigned items.

# Time for action – rendering the scene's content to an image

Let's try to render a scene to an image. In order to do that, we take the following code snippet from our first example where we tried to put items on a scene:

[PRE15]

The only change we make here is that we set a brush resulting in a green-filled rectangle with a red border, which was defined through `setBrush()` and `setPen()`. You can also define the thickness of the stroke by passing a `QPen` object with the corresponding arguments. To render the scene, you only need to call `render()`, which takes a pointer to a `QPainter` pointer. This way, the scene can render its contents to any paint device the painter is pointing to. For us, a simple PNG file will do the job.

[PRE16]

![Time for action – rendering the scene's content to an image](img/8874OS_06_13.jpg)

Result of the rendering

## *What just happened?*

First you determined the rectangle of the scene with `sceneRect()`. Since this returns a `QRectF` parameter and `QImage` can only handle `QRect`, you transformed it on-the-fly by calling `toAlignedRect()`. The difference between the `toRect()` function and `toAlignedRect()` is that the former rounds to the nearest integer, which may result in a smaller rectangle whereas the latter expands to the smallest possible rectangle containing the original `QRectF` parameter. Then, you created a `QImage` file with the size of the aligned scene's rectangle. Because the image is created with uninitialized data, you need to call `fill()` with `Qt::transparent` to receive a transparent image. You can assign any color you like as an argument both a value of `Qt::GlobalColor` enumeration and an ordinary `QColor` object; `QColor(0, 0, 255)` would result in a blue background. Next, you create a `QPainter` object which points to the image. This painter object is now used in the scene's `render()` function to draw the scene. After that, all you have to do is to save the image to a place of your choice. The file name (which can also contain an absolute path such as `/path/to/image.png`) is given by the first argument whereas the second argument determines the format of the image. Here, we set the file name to `scene.png` and choose the PNG format. Since we haven't specified a path, the image will be saved in the application's current directory.

## Have a go hero – rendering only specific parts of a scene

This example draws the whole scene. Of course, you can also render only specific parts of the scene by using the other arguments of `render()`. We will not go into this here but you may want to try it as an exercise.

## Coordinate system of the scene

What is left is a look at the coordinate system of the scene. Like the items, the scene lives in its own coordinate system with the origin at (0, 0). Now when you add an item via `addItem()`, the item is positioned at the scene's (0, 0) coordinate. If you want to move the item to another position on the scene, call `setPos()` on the item.

[PRE17]

After creating the scene and the item, you add the item to the scene by calling `addItem()`. At this stage, the scene's origin and the item's origin are stacked on top of each other at (0, 0). By calling `setPos()`, you move the item 50 pixels right and down. Now the item's origin is at (50, 50) in scene coordinates. If you need to know the position of the bottom-right corner of the item's rectangle in scene coordinates, you have to do a quick calculation. In the item's coordinate system, the bottom right corner is at (10, 10). The item's origin point is (0, 0) in the item's coordinate system, which corresponds to the point (50, 50) in the scene's coordinate system. So you just have to take (50, 50) and add (10,10) to get (60, 60) as the scene's coordinates for the bottom-right corner of the item. This is an easy calculation, but it quickly gets complicated when you rotate, scale, and/or shear the item. Because of this, you should use one of the convenience functions provided by `QGraphicsItem`:

| Function | Description |
| --- | --- |
| `mapToScene(const QPoint &point)` | Maps the point `point` that is in the item's coordinate system to the corresponding point in the scene's coordinate system. |
| `mapFromScene(const QPoint &point)` | Maps the point `point` that is in the scene's coordinate system to the corresponding point in the item's coordinate system. This function is the reverse function to `mapToScene()`. |
| `mapToParent(const QPoint &point)` | Maps the point `point` that is in the item's coordinate system to the corresponding point in the coordinate system of the item's parent. If the item does not have a parent, this function behaves like `mapToScene()`; thus, it returns the corresponding point in the scene's coordinate system. |
| `mapFromParent(const QPoint &point)` | Maps the point `point` that is in the coordinate system of the item's parent to the corresponding point in the item's own coordinate system. This function is the reverse function to `mapToParent()`. |
| `mapToItem(const QGraphicsItem *item, const QPointF &point)` | Maps the point `point` that is in the item's own coordinate system to the corresponding point in the coordinate system of the item `item`. |
| `mapFromItem(const QGraphicsItem *item, const QPointF &point)` | Maps the point `point` which is in the coordinate system of the item `item` to the corresponding point in the item's own coordinate system. This function is the reverse function to `mapToItem()`. |

What is great about these functions is that they are not only available for `QPointF`. The same functions are also available for `QRectF`, `QPolygonF`, and `QPainterPath`. Not to mention that these are of course convenience functions: If you call these functions with two numbers of the type `qreal`, the numbers get interpreted as the *x* and *y* coordinates of a `QPointF` pointer; if you call the functions with four numbers, the numbers get interpreted as the *x* and *y* coordinates and the width and the height of a `QRectF` parameter.

Since the positioning of the items is done by the items themselves, it is possible that an item independently moves around. Do not worry; the scene will get notified about any item position change. And not only the scene! Remember the parent-child relationship of items and that parents delete their child items when they get destroyed themselves? It's the same with `setPos()`. If you move a parent, all child items get moved as well. This can be very useful if you have a bunch of items that should stay together. Instead of moving all items by themselves, you only have to move one item. Since transformations that you apply on a parent also affect the children, this might not be the best solution for grouping together equal items that should be independently transformable but also transformable altogether. The solution for such a case is `QGraphicsItemGroup`. It behaves like a parent in a parent-child relationship. The `QGraphicsItemGroup` is an invisible parent item so that you can alter the child items separately through their transformation functions or all together by invoking the transformation functions of `QGraphicsItemGroup`.

# Time for action – transforming parent items and child items

Have a look at the following code:

[PRE18]

## *What just happened?*

After creating a scene, we create four rectangle items that are arranged in a 2 x 2 matrix. This is done with the calls of the `moveBy()` function, which interprets the first argument as a shift to the right or left when negative and the second argument as a shift to the bottom or top when negative. Then we create a new `QGraphicsItemGroup` item which, since it subclasses `QGraphicsItem`, is a regular item and can be used as such. By calling `addToGroup()`, we add the items that we want to position inside that group. If you'd like to remove an item from the group later on, simply call `removeFromGroup()` and pass the respective item. The `rectD` parameter is added to the group in a different way. By calling `setGroup()` on `rectD`, it gets assigned to `group`; this behavior is comparable to `setParent()`. If you want to check whether an item is assigned to a group, just call `group()` on it. It will return a pointer to the group or `0` if the item is not in a group. After adding the group to the scene, and thus also the items, we rotate the whole group by 70 degrees clockwise. Afterward, all items are separately rotated 25 degrees counter-clockwise around their top left corner. This will result in the following appearance:

![What just happened?](img/8874OS_06_09.jpg)

Here you see the initial state after moving the items, then after rotating the group by 70 degrees, and then after rotating each item by -25 degrees

If we were to rotate the items more, they would overlap each other. But which item would overlap which? This is defined by the item's *z* value; you can define the value by using `QGraphicsItem::setZValue()` otherwise it is `0`. Based on that, the items get stacked. Items with a higher *z* value are displayed on top of items with lower *z* values. If items have the same *z* value, the order of insertion decides the placement: items added later overlap those added earlier. Also, negative values are possible.

## Have a go hero – playing with the z value

Take the item group from the example as a starting point and apply various transformations to it as well as different *z* values for the item. You will be astonished at what crazy geometrical figures you can create with these four items. Coding really is fun!

For the sake of completeness, a word on the scene's bounding rectangle is required (set through `setSceneRect()`). Just as the offset of an item's bounding rectangle affects its position on the scene, the offset of the scene's bounding rectangle affects the scene's position on the view. More importantly, however, the bounding rectangle is used for various internal computations, such as the calculation of the view's scroll bar value and position. Even if you do not have to set the scene's bounding rectangle, it is recommended that you do. This applies especially when your scene holds a lot of items. If you do not set a bounding rectangle, the scene calculates this itself by going through all the items, retrieving their positions and their bounding rectangles as well as their transformations to figure out the maximum occupied space. This calculation is done by the function `itemsBoundingRect()`. As you may imagine, this becomes increasingly resource-intensive the more items a scene has. Furthermore, if you do not set the scene's rectangle, the scene checks on each item's update if the item is still in the scene's rectangle. Otherwise, it enlarges the rectangle to hold the item inside the bounding rectangle. The downside to is that it will never adjust by shirking; it will only enlarge. Thus, when you move an item to the outside and then to the inside again, you will mess up the scroll bars.

### Tip

If you do not want to calculate the size of your scene yourself, you can add all items to the scene and then call `setSceneRect()` with `itemsBoundingRect()` as an argument. With this, you stop the scene from checking and updating the maximum bounding rectangle on item updates.

## View

With `QGraphicsView`, we are back in the world of widgets. Since `QGraphicsView` inherits `QWidget`, you can use the view like any other widget and place it into layouts for creating neat graphical user interfaces. For the Graphics View architecture, `QGraphicsView` provides an inspection window on a scene. With the view, you can display the whole scene or only part of it, and by using a transformation matrix you can manipulate the scene's coordinate system. Internally, the view uses `QGraphicsScene::render()` to visualize the scene. By default, the view uses a `QWidget` element as a painting device. Since `QGraphicsView` inherits `QAbstractScrollArea`, the widget is set as its viewport. Therefore, when the rendered scene exceeds the view's geometry, scroll bars are automatically shown.

### Note

Instead of using the default `QWidget` element as the viewport widget, you can set your own widget by calling `setViewport()` with the custom one as an argument. The view will then take ownership of the assigned widget, which is accessible by `viewport()`. This also gives you the opportunity to use OpenGL for rendering. Simply call `setViewport(new QGLWidget)`.

# Time for action – putting it all together!

Before we go on, however, and after talking a lot about items and scenes, let's see how the view, the scene, and the items all work together:

[PRE19]

Build and run this example and you will see following image in the middle of the view:

![Time for action – putting it all together!](img/8874OS_06_06.jpg)

## *What just happened?*

What have we done here? On top, we included the needed headers and then wrote a normal main function and created a `QApplication` elment. Its event loop is started in the return statement on the bottom. In between, we created a scene and added the first item to it by calling `addEllipse()`. This function is one of the many convenience functions of Qt and is, in our case, equivalent to the following code:

[PRE20]

We thus have put a circle with a radius of 50 pixels in the scene. The origins of the circle and of the scene are stacked on top of each other. Next, by calling `addLine()`, we add a blue line that goes through the center point of the circle, parallel to the scene's bottom line. The first two arguments are the *x* and *y* coordinates of the line's starting point and the second two arguments the *x* and *y* coordinates of the end point. With `addRect()`, we add a square with a 25-pixel side at the top-left corner of the scene. This time, however, we fetch the pointer, which is then returned by these functions. This is because we want to move the rectangle to the center of the scene. In order to do that, we use `setPos()` and need to do some arithmetic. Why? Because of the relationship between the scene's and the item's coordinate systems. By simply calling `item->setPos(scene.sceneRect().center())`, the origin of the item (which is (0, 0) in the item's coordinates and thus the rectangle's top left corner) would be in the middle of the scene, not the red square itself. Thus we need to shift the rectangle back by half of its width and height. This is done by subtracting its center point from the scene's center point. As you probably have already guessed, `QRectF::center()` returns the center point of a rectangle as a `QPointF` pointer. Lastly, we create a view and declare that it should display the scene by calling `setScene()` with the scene as an argument. Then we show the view. That's all you need to do to show a scene with items.

Two things you will probably notice if you have a look at the result are that the drawing looks pixelated and that it stays in the center of the view when you resize the view. The solution for the first problem you should already know from what you learned in the previous chapter. You have to turn on antialiasing. For the view, you do that with this line of code

[PRE21]

With `setRenderHint()`, you can set all hints you know from `QPainter` to the view. Before the view renders the scene on its viewport widget, it initializes the internally used `QPainter` element with these hints. With the antialiasing flag turned on, the painting is done much more smoothly. Unfortunately, the line is also painted antialiased (even though we do not want this since now the line looks washy). To prevent the line from getting drawn antialiased, you have to override the `paint()` function of the item and explicitly turn off antialiasing. However, you might want to have a line with aliasing somewhere, so there is another small and easy solution for that problem without the need for reimplementing the `paint` function. All you have to do is to shift the position by half of the pen's width. For that, write the following code:

[PRE22]

By calling `pen()`, you get the pen that is used to draw the line. Then you determine its width by calling `widthF()` and dividing it by 2\. Then just move the line whereby the `moveBy()` function behaves as if we had called the following:

[PRE23]

To be pixel-perfect, you might need to alter the length of the line.

The second "problem" was that the scene is always visualized in the center of the view, which is the default behavior of the view. You can change this setting with `setAlignment()`, which accepts `Qt::Alignment` flags as arguments. So, calling `view.setAlignment(Qt::AlignBottom | Qt::AlignRight)`; would result in the scene staying in the lower-right corner of the view.

## Showing specific areas of the scene

As soon as the scene's bounding rectangle exceeds the viewport's size, the view will show scroll bars. Besides using them with the mouse to navigate to a specific item or point on the scene, you can also access them by code. Since the view inherits `QAbstractScrollArea`, you can use all its functions for accessing the scroll bars. `horizontalScrollBar()` and `verticalScrollBar()` return a pointer to `QScrollBar`, and thus you can query their range with `minimum()` and `maximum()`. By invoking `value()` and `setValue()`, you get and can set the current value, which results in scrolling the scene.

But normally, you do not need to control free scrolling inside the view from your source code. The normal task would be to scroll to a specific item. In order to do that, you do not need to do any calculations yourself; the view offers a pretty simple way to do that for you: `centerOn()`. With `centerOn()`, the view ensures that the item, which you have passed as an argument, is centered on the view unless it is too close to the scene's border or even outside. Then, the view tries to move it as far as possible on the center. The `centerOn()` function does not only take a `QGraphicsItem` item as argument; you can also center on a `QPointF` pointer or as a convenience on an *x* and *y* coordinate.

If you do not care where an item is shown, you can simply call `ensureVisible()` with the item as an argument. Then the view scrolls the scene as little as possible so that the item's center remains or becomes visible. As a second and third argument, you can define a horizontal and vertical margin, which are both the minimum space between the item's bounding rectangle and the view's border. Both values have 50 pixels as their default value. Beside a `QGraphicsItem` item, you can also ensure the visibility of a `QRectF` element (of course, there is also the convenience function taking four `qreal` elements).

### Tip

If you like to ensure the entire visibility of an item (since `ensureVisible(item)` only takes the item's center into account) use `ensureVisible(item->boundingRect())`. Alternatively, you can use `ensureVisible(item)`, but then you have to set the margins at least to the item's half width or half height respectively.

`centerOn()` and `ensureVisible()` only scroll the scene but do not change its transformation state. If you absolutely want to ensure the visibility of an item or a rectangle that exceeds the size of the view, you have to transform the scene as well. With this task, again the view will help you. By calling `fitInView()` with `QGraphicsItem` or a `QRectF` element as argument, the view will scroll and scale the scene so that it fits in the viewport size. As a second argument, you can control how the scaling is done. You have the following options:

| Value | Description |
| --- | --- |
| `Qt::IgnoreAspectRatio` | The scaling is done absolutely freely regardless of the item's or rectangle's aspect ratio. |
| `Qt::KeepAspectRatio` | The item's or rectangle's aspect ratio is taken into account while trying to expand as far as possible while respecting the viewport's size. |
| `Qt::KeepAspectRatioByExpanding` | The item's or rectangle's aspect ratio is taken into account, but the view tries to fill the whole viewport's size with the smallest overlap. |

The `fitInView()` function does not only scale larger items down to fit the viewport, it also enlarges items to fill the whole viewport. The following picture illustrates the different scaling options for an item that is enlarged:

![Showing specific areas of the scene](img/8874OS_06_02.jpg)

The circle on the left is the original item. Then, from left to right it is `Qt::IgnoreAspectRatio`, `Qt::KeepAspectRatio`, and `Qt::KeepAspectRatioByExpanding`.

## Transforming the scene

In the view, you can transform the scene as you like. Besides the normal convenience functions, such as `rotate()`, `scale()`, `shear()`, and `translate()`, you can also apply a free definable `QTransform` parameter via `setTransform()`, where you also can decide if the transformation should be combined with existing ones or if it should replace them. As an example of probably the most used transformation on a view, let us have a look how you can scale and move the scene inside the view.

# Time for action – creating an item where transformations can easily be seen

First we set up a playground. To do this, we subclass a `QGraphicsRectItem` item and customize its paint function as follows:

[PRE24]

## *What just happened?*

By using the `Q_UNUSED` macro, we simply suppress compiler warnings about unused variables. The macro expands to `(void)x;`, which does nothing. Then we cache the current pen for putting it back at the end of the function. This gives `painter` back unchanged. Of course, we could have called `save()` and `restore()` on the painter, but these functions save a lot of other properties we do not want to change, so simply saving and restoring the pen is much faster. Next, we draw four red rectangles at the corners of the bounding rectangle (`r`) by calling `fillRect()`, which does not change the painter state. Then we set a 1-pixel thick and solid black pen—because this changes the pen's state, we have saved the old pen—and draw the bounding rectangle, the diagonals, and a centered rectangle, which is a quarter of the size of the bounding rectangle. This will give us the following item, which shows the transformations better than with a black-filled rectangle:

![What just happened?](img/8874OS_06_16.jpg)

# Time for action – implementing the ability to scale the scene

Let's do the scaling first. We add the item to a scene and put that scene on a custom view we have subclassed from `QGraphicsView`. In our customized view, we only need to reimplement `wheelEvent()` as we want to scale the view by using the mouse's scroll wheel.

[PRE25]

## *What just happened?*

The `factor` parameter for the zooming can be freely defined. You can also create a getter and setter method for it. For us, 1.1 will do the work. With `event->angleDelta()`, you get the distance of the mouse's wheel rotation as a `QPoint` pointer. Since we only care about vertical scrolling, just the *y* axis is relevant for us. In our example, we also do not care about how far the wheel was turned because, normally, every step is delivered separately to `wheelEvent()`. But if you should need it, it's in eighths of a degree, and since a mouse works in general steps of 15 degrees, the value should be 120 or -120, depending on whether you move the wheel forward or backward. On a forward wheel move, if `y()` is greater than zero, we zoom in by using the built-in `scale()` function. It takes the scale factor for the *x* and the *y* coordinates. Otherwise, if the wheel was moved backwards, we zoom out. That's all there is to it. When you try this example, you will notice that, while zooming, the view zooms in and out on the center of the view, which is the default behavior for the view. You can change this behavior with `setTransformationAnchor()`. `QGraphicsView::AnchorViewCenter` is, as described, the default behavior. With `QGraphicsView::NoAnchor`, the zoom center is in the top-left corner of the view, and the value you probably want to use is `QGraphicsView::AnchorUnderMouse`. With that option, the point under the mouse builds the center of the zooming and thus stays at the same position inside the view.

# Time for action – implementing the ability to move the scene

Next it would be good to move the scene around without the need of using the scroll bars. Let us add the functionality for pressing and holding the left mouse button. First, we add two private members to the view: the `m_pressed` parameter of type `bool` and the `m_lastMousePos` element of type `QPoint`. Then, we reimplement the `mousePressEvent()` and `mouseReleaseEvent()` functions as follows:

[PRE26]

## *What just happened?*

Within `mousePressEvent()`, we check whether the left mouse button was pressed. If it was `true`, we then set `m_pressed` to `true` and save the current mouse position in `m_lastMousePos`. Then we pass the event to the base class event handler. Within `mouseReleaseEvent()`, we set `m_pressed` to `false` if it was the left button; then we pass the event to the base class implementation. We do not need to alter `m_pressPoint` here. With `mouseMoveEvent()`, we can then react on the value of those two variables:

[PRE27]

If `m_pressed` is `false`—this means the left button wasn't pressed and held—we will be exiting the function while passing the event to the base class implementation. This is, by the way, important for getting unhandled events propagated to the scene correctly. If the button has been pressed, we first calculate the difference (`diff`) between the point where the mouse was pressed and the current position. Thus we know how far the mouse was moved. Now we simply move the scroll bars by that value. For the horizontal scroll bar, the pointer to it is received by calling `horizontalScrollBar()`. The encapsulation in an `if` clause is just a paranoid safety check to ensure that the pointer is not null. Normally, this should never happen. Through that pointer, we set a new value by adding the old value, received by `value()`, to the moved distance, `diff.x()`. We then do the same for the vertical scroll bar. Last, we save the current mouse position to `m_lastMousePos`. That's all. Now you can move the scene around while holding the left mouse button down. The downside of this method is that the left mouse click does not reach the scene and, therefore, features such as item selection do not work. If you need that or a similar functionality on the scene, check for a keyboard modifier too. For example, if the *Shift* key must also be pressed to move the scene, additionally check the events `modifiers()` for whether `Qt::ShiftModifier` is set to activate the mouse-moving functionality:

[PRE28]

# Time for action – taking the zoom level into account

As a last detail, I would like to mention that you can draw an item differently depending on its scale. To do that, the level of detail can be used. You use the passed pointer to `QStyleOptionGraphicsItem` of the item's `paint` function and call `levelOfDetailFromTransform()` with the painter's world transformation. We change the paint function of the `ScaleItem` item to the following:

[PRE29]

## *What just happened?*

The `detail` parameter now contains the maximum width of unity square, which was mapped to the painter coordinate system via the painter's world transformation matrix. Based on that value, we set the fill color of the border rectangles to yellow or red. The expression `detail >= 5` will become `true` if the rectangle is displayed at least five times as large as in a normal state. The level of detail is helpful when you want to draw more detail on an item only if it is visible. By using the level of detail, you can control when a possibly resource-intensive drawing should be performed. It makes sense, for example, to make difficult drawings only when you can see them.

When you zoom into the scene, the diagonal lines as well as the rectangle lines get zoomed. But you may want to leave the stroke the same regardless of the zoom level. Here Qt also has an easy approach to offer. In the paint function of the item we used earlier for exemplifying the zoom functionality, locate the following line of code:

[PRE30]

Replace it with the following lines:

[PRE31]

The important part is to make the painter cosmetic. Now, regardless of the zoom or any other transformation, the pen's width stays the same. This can be very helpful for drawing outlined shapes.

## Questions you should keep in mind

Whenever you are going to use the Graphics View architecture, ask yourself these questions: Which standard items are suited for my specific needs? Am I reinventing the wheel over and over again? Do I need `QGraphicsTextItem` or is `QGraphicsSimpleTextItem` good enough? Do I need the items to inherit `QObject` or will plain items not suffice? (We will cover this topic in the next section.) Could I group items together for the sake of cleaner and leaner code? Is the parent-child relationship sufficient or do I need to use a `QGraphicsItemGroup` element?

Now you really know most of the functions of the Graphics View framework. With this knowledge, you can already do a lot of cool stuff. But for a game, it is still too static. We will change that next!

# The jumping elephant or how to animate the scene

By now, you should have a good understanding about the items, the scene, and the view. With your knowledge of how to create items, standard and custom ones, of how to position them on the scene, and of how to set up the view to show the scene, you can make pretty awesome things. You even can zoom and move the scene with the mouse. That's surely good, but for a game, one crucial point is still missing: you have to animate the items. Instead of going through all possibilities for how to animate a scene, let us develop a simple jump-and-run game where we recap parts of the previous topics and learn how to animate items on a screen. So let's meet Benjamin, the elephant:

![The jumping elephant or how to animate the scene](img/8874OS_06_03.jpg)

## The game play

The goal of the game is for Benjamin to collect the coins that are placed all over the game field. Besides walking right and left, Benjamin can, of course, also jump. In the following screenshot, you see what this minimalistic game should look like in the end:

![The game play](img/8874OS_06_12.jpg)

## The player item

Let's now look at how we can mobilize Benjamin.

# Time for action – creating an item for Benjamin

First we need a custom item class for Benjamin. We call the class `Player` and choose `QGraphicsPixmapItem` as the base class because Benjamin is a PNG image. In the item's `Player` class, we further create a property of integer type and call it `m_direction`. Its value signifies in which direction Benjamin walks—left or right—or if he stands still. Of course, we use a getter and setter function for this property. Since the header file is simple, let's have a look at the implementation right away (you will find the whole source code at the end of this book):

[PRE32]

In the constructor, we set `m_direction` to `0`, which means that Benjamin isn't moving at all. If `m_direction` is `1`, Benjamin moves right, and if the value is `-1`, he moves left. In the body of the constructor, we set the image for the item by calling `setPixmap()`. The image of Benjamin is stored in the Qt Resource System; thus, we access it through `QPixmap(":/elephant")` with `elephant` as the given alias for the actual image of Benjamin. Last, we set the point of origin for all transformations we are going to apply to the center of the item. This equals the center of the image.

[PRE33]

The `direction()` function is a standard getter function for `m_direction` returning its value. The next function of this class is much more important:

[PRE34]

## *What just happened?*

With `addDirection()`, one "sets" the direction of Benjamin's movement. "Set" is put in quotes because you do not set `m_direction` to the passed value; instead, you add the passed value to `m_direction`. This is done in the second line after we have ensured the correctness of `m_direction`. For that, we use `qBound()`, which returns a value that is bound by the first and last argument. The argument in the middle is the actual value that we want to get bound. So the possible values for `m_direction` are restricted to -1, 0, and 1\. If the property `direction` is 0, the player item does not move and the function exits.

If you haven't already done so earlier, you might wonder by now why not simply set the value? Why that addition? Well, it is because of how we will use this function: Benjamin is moved by the left and right arrow key. If the right key is pressed, 1 is added; if it gets released, -1 is added. Think of it as an impulse to the right (1) and to the left (-1). The first accelerates the player and the second slows him down. The same applies for the left key, but only the other way around. As we do not allow multiple acceleration, we limit the value of `m_direction` to 1 and -1\. The addition of the value rather than setting it is now necessary because of the following situation: A user presses and holds the right key, and the value of `m_direction` is therefore 1\. Now, without releasing the right key, he also presses and holds the left key. Therefore, the value of `m_direction` is getting decreased by one; the value is now 0 and Benjamin stops. But remember, both keys are still being pressed. What happens when the left key is released? How would you know in this situation in which direction Benjamin should move? To achieve that, you would have to find out an additional bit of information: whether the right key is still pressed down or not. That seems too much trouble and overhead. In our implementation, when the left key is released, 1 is added and the value of `m_direction` becomes 1, making Benjamin move right. Voilà! All without any concern about what the state of the other button might be.

Lastly, we check in which direction Benjamin is moving. If he is moving left, we need to flip his image so that Benjamin looks to the left, the direction in which he is moving. Therefore, we apply a `QTransform` matrix, which flips the image vertically. If he is moving towards the right, we restore the normal state by assigning an empty `QTransform` object, which is an identity matrix.

So we now have our item of class `Player` for the game's character, which shows the image of Benjamin. The item also stores the current moving direction, and based on that information, the image is flipped vertically if needed.

## The playing field

To understand the following code, it might be good to know the composition of the environment in which our elephant will be walking and jumping. Overall, we have a view fixed in size holding a scene which is exactly as big as the view. We do not take size changes into account since they would complicate the example too much, and when you develop a game for a mobile device, you know the available size up front.

All animations inside the playing field are done by moving the items, not the scene. So we have to distinguish between the view's, or rather the scene's width and the width of the elephant's virtual "world" in which he can move. The width of this virtual world is defined by `m_fieldWidth` and has no (direct) correlation with the scene. Within the range of `m_fieldWidth`, which is 500 pixels in the example, Benjamin or the graphics item can be moved from the minimum *x* coordinate, defined by `m_minX`, to the maximum *x* coordinate, defined by `m_maxX`. We keep track of his actual *x* position with the variable `m_realPos`. Next, the minimum *y* coordinate the item is allowed to have is defined by `m_groundLevel`. For `m_maxX` and `m_groundLevel`, we have to take into account that the position of the item is determined by its top-left corner. Lastly, what is left is the view, which has a fixed size defined by the scene's bounding rectangle size, which is not as wide as `m_fieldWidth`. So the scene (and the view) follows the elephant while he walks through his virtual world of the length `m_fieldWidth`. Have a look at the picture to see the variables in their graphical representation:

![The playing field](img/8874OS_06_04.jpg)

## The scene

Since we will have to do some work on the scene, we subclass `QGraphicsScene` and name the new class `MyScene`. There we implement one part of the game logic. This is convenient since `QGraphicsScene` inherits `QObject` and thus we can use Qt's signal and slot mechanism. Also, for the next code of the scene, we only go through the implementation of the functions. For more information on the header, have a look at the sources bundled with this book.

# Time for action – making Benjamin move

The first thing we want to do is to make our elephant movable. In order to achieve that, we use a `QTimer` parameter called `m_timer`, which is a private member of `MyScene`. In the constructor we set up the timer with the following code:

[PRE35]

First we define that the timer emits a timeout signal every 30 milliseconds. Then we connect that signal to the scene's slot called `movePlayer()`, but we do not start the timer yet. This is done by the arrow keys in a way we have already discussed when the `m_direction` variable of the class `Player` was introduced. Here is the implementation of what was described there:

[PRE36]

### Note

As a small side note, whenever code snippets in the following code passages are irrelevant for the actual detail, I am going to skip the code but will indicate missing code with `//...` so that you know it is not the entire code. We will cover the skipped parts later when it is more appropriate.

## *What just happened?*

In the key press event handler, we first check if the key event was triggered because of an auto repeat. If this is the case, we exit the function because we only want to react on the first real key press event. We also do not call the base class implementation of that event handler since no item on the scene needs to get a key press event. If you do have items that could and should receive events, do not forget to forward them while reimplementing event handlers at the scene.

### Note

If you press and hold a key down, Qt will continuously deliver the key press event. To determine if it was the first real key press or an auto-generated event, use `QKeyEvent::isAutoRepeat()`. It returns `true` if the event was automatically generated. There is no easy way to turn off the auto repeat since it is platform-dependent and you have to use the platform API for that.

As soon as we know that the event was not delivered by an auto repeat, we react to the different key presses. If the left key was pressed, we decrease the direction property of the player item by one; if the right key was pressed, we increase it by one. The `m_player` element is our instance of the player item. After calling `addDirection()`, we call `checkTimer()` in both cases:

[PRE37]

This function first checks whether the player moves. If not, the timer is stopped because nothing has to be updated when our elephant stands still. Otherwise, the timer gets started, but only if it isn't already running. This we check by calling `isActive()` on the timer.

When the user presses the right key, for example at the beginning of the game, `checkTimer()` will start `m_timer`. Since its time out signal was connected to `movePlayer()`, the slot will be called every 30 milliseconds till the key is released. Since the `move()` function is a bit longer, let's go through it step-by-step:

[PRE38]

First, we cache the player's current direction in a local variable to avoid multiple calls of `direction()`. Then we check whether the player is moving at all. If they aren't, we exit the function because there is nothing to animate.

[PRE39]

Next we calculate the shift the player item should get and store it in `dx`. The distance the player should move every 30 milliseconds is defined by the member variable `m_velocity`, expressed in pixels. You can create setter and getter functions for that variable if you like. For us, the default value of 4 pixels will do the job. Multiplied by the direction (which could only be 1 or -1 at this point), we get a shift of the player by 4 pixels to the right or to the left. Based on this shift, we calculate the new *x* position of the player and store it in `newPos`. Next, we check whether that new position is inside the range of `m_minX` and `m_maxX`, two member variables that are already calculated and set up properly at this point. Next, if the new position is not equal to the actual position, which is stored in `m_realPos`, we proceed by assigning the new position as the current one. Otherwise, we exit the function since there is nothing to move.

[PRE40]

The next question to tackle is whether the view should always move when the elephant is moving, which means that the elephant would always stay say in the middle of the view. No, he shouldn't stay at a specific point inside the view. Rather, the view should be fixed when the elephant is moving. Only if he reaches the borders should the view follow. The "non-movable" center is defined by `leftBorder` and `rightBorder`, which are related to the item's position; thus we must subtract the item's width from the `rightBorder` element. If we don't take the item's width into account, the right side of a player with a width of more than 150 pixels will disappear before the scrolling takes place. Please note that the values for `leftBorder` and `rightBorder` are randomly chosen. You can alter them as you like. Here we decided to set the border at 150 pixels. Of course, you can create a setter and getter for these parameters too:

[PRE41]

Ok, so what have we done here? Here we have calculated whether only the elephant moves or the view as well so that the elephant does not walk out of the screen. The `if` clause applies when the elephant is moving towards the right. For a better understanding, let's begin at the end of this scope. There is a situation where we do not move the elephant but simply add the shift `dx` to a variable named `m_skippedMoving`. What does that mean? It means that the virtual "world" is moving but the elephant inside the view is not. This is the case when the elephant moves too far to the borders. In other words, you move the view with the elephant above the virtual world by `dx` to the left. Let's take a look at the following figure:

![What just happened?](img/8874OS_06_05.jpg)

The `m_skippedMoving` element is the difference between the view's *x* coordinate and the virtual world's *x* coordinate. So the `if` clause `m_realPos - m_skippedMoving < rightBorder` reads: *If the position of the elephant in "view coordinates", calculated by* `m_realPos – m_skippedMoving` *, is smaller than* `rightBorder` *, then move the elephant by calling* `moveBy()` *since he is allowed to walk till* `rightBorder` *.* `m_realPos - m_skippedMoving` *is the same as* `m_player->pos().x() + dx` *.*

Lastly, let's turn to the first clause: `m_realPos > m_fieldWidth - (width() - rightBorder)`. This returns `true` when the actual position is behind the `rightBorder` element but the fictional world is moved to its maximum left. Then we also have to move the elephant so that he can reach `m_maxX`. The expression `width() - rightBorder` calculates the width between `rightBorder` and the scene's right border.

The same considerations and calculations apply for moving to the left, the other branch.

So far, we have accomplished two things. First, with a `QTimer` object, we trigger a slot that moves an item; thus, we have animated the scene. Second, we have determined the elephant's position in the virtual world. You might wonder why we have done this. To be able to do parallax scrolling!

## Parallax scrolling

Parallax scrolling is a trick to add an illusion of depth to the background of the game. This illusion occurs when the background has different layers which move at different speeds. The nearest background must move faster than the ones farther away. In our case, we have these four backgrounds ordered from the most distant to the nearest:

![Parallax scrolling](img/8874OS_06_17.jpg)

The sky

![Parallax scrolling](img/8874OS_06_18.jpg)

The trees

![Parallax scrolling](img/8874OS_06_07.jpg)

The grass

![Parallax scrolling](img/8874OS_06_08.jpg)

The ground

# Time for action – moving the background

Now the question is how to move them at different speeds. The solution is quite simple: the slowest one, the sky, is the smallest image. The fastest background, the ground and the grass, are the largest images. Now when we have a look at the end of the `movePlayer()` function's slot we see this:

[PRE42]

## *What just happened?*

What are we doing here? At the beginning, the sky's left border is the same as the view's left border, both at point (0, 0). At the end, when Benjamin has walked to the maximum right, the sky's right border should be the same as the view's right border. So the distance we have to move the sky over time is the sky's width (`m_sky->boundingRect().width()`) minus the width of the view (`width()`). The shift of the sky depends on the position of the player: If he is far to the left, the sky isn't shifted, if the player is far to the right, the sky is maximally shifted. We thus have to multiply the sky's maximum shift value with a factor based on the current position of the player. The relation to the player's position is the reason why this is handled in the `movePlayer()` function. The factor we have to calculate has to be between 0 and 1\. So we get the minimum shift (0 * shift, which equals 0) and the maximum shift (1 * shift, which equals shift). This factor we name `ff`. The calculation reads: *If we subtract the width of the view (* `width()` *) from the virtual field's width* `m_fieldWidth` *, we have the area where the player isn't moved by (* `m_player->moveBy()` *) because in that range only the background should move.*

How often the moving of the player was skipped is saved in `m_skippedMoving`. So by dividing `m_skippedMoving` through `m_fieldWidth – width()`, we get the needed factor. It is 0 when the player is to the far left and 1 if they are to the far right. Then we simply have to multiply `ff` with the maximum shift of the sky. To avoid the backgrounds from being moved too far, we ensure through `qMin()` that the factor is always lesser than, or equal to, 1.0.

The same calculation is used for the other background items. The calculation also explains why a smaller image is moving slower. It's because the overlap of the smaller image is less than that of the larger one. And since the backgrounds are moved in the same time period, the larger has to move faster.

## Have a go hero – adding new background layers

Try to add additional background layers to the game following the preceding example. As an idea, you can add a barn behind the trees or let an airplane fly through the sky.

## QObject and items

The `QGraphicsItem` item and all standard items introduced so far don't inherit `QObject` and thus can't have slots or emit signals; they also don't benefit from the `QObject` property system. But we can make them use `QObject`!

# Time for action – using properties, signals, and slots with items

So let's alter the `Player` class to use `QObject`:

[PRE43]

All you have to do is to add `QObject` as a base class and add the `Q_OBJECT` macro. Now you can use signals and slots with items too. Be aware that `QObject` must be the first base class of an item.

### Tip

If you want an item that inherits from `QObject` and `QGraphicsItem`, you can directly inherit `QGraphicsObject`. Moreover, this class defines and emits some useful signals such as `xChanged()` when the *x* coordinate of the item has changed or `scaleChanged()` when the item is scaled.

### Note

A word of warning: Only use `QObject` with items if you really need its capabilities. `QObject` adds a lot of overhead to the item, which will have a noticeable impact on performance when you have many items. So use it wisely and not only because you can.

Let us go back to our player item. After adding `QObject`, we define a property called `m_jumpFactor` with a getter, a setter, and a change signal. We need that property to make Benjamin jump, as we will see later on. In the header file, we define the property as follows:

[PRE44]

The getter function `jumpFactor()` simply returns the private member `m_jumpFactor`, which is used to store the actual position. The implementation of the setter looks like this:

[PRE45]

It is important to check if `pos` would change the current value of `m_jumpFactor`. If this is not the case, exit the function because, otherwise, a change signal will be emitted even if nothing has changed. Otherwise, we set `m_jumpFactor` to `pos` and emit the signal that informs about the chance.

## Property animations

The new `jumpFactor` property we use immediately with a `QPropertyAnimation` element, a second way to animate items.

# Time for action – using animations to move items smoothly

In order to use it, we add a new private member called `m_animation` of type `QPropertyAnimation` and initialize it in the constructor of `Player`:

[PRE46]

## *What just happened?*

For the instance of `QPropertyAnimation` created here, we define the item as parent; thus, the animation will get deleted when the scene deletes the item and we don't have to worry about freeing the used memory. Then we define the target of the animation—our `Player` class—and the property that should be animated—`jumpFactor`, in this case. Then we define the start and the end value of that property, and in addition to that we also define a value in between by setting `setKeyValueAt()`. The first argument of type `qreal` defines time inside the animation, where 0 is the beginning and 1 the end, and the second argument defines the value that the animation should have at this time. So your `jumpFactor` element will get animated from 0 to 1 and back to 0 in 800 milliseconds. This was defined by `setDuration()`. Finally, we define how the interpolation between the start and end value should be done and call `setEasingCurve()` with `QEasingCurve::OutInQuad` as an argument. Qt defines up to 41 different easing curves for linear, quadratic, cubic, quartic, quintic, sinusoidal, exponential, circular, elastic, back easing, and bounce functions. These are too many to describe here. Instead, have a look at the documentation. Simply search for `QEasingCurve::Type`. In our case, `QEasingCurve::OutInQuad` makes sure that the jump speed of Benjamin looks like an actual jump: fast in the beginning, slow at the top, and fast at the end again. We start this animation with the jump function:

[PRE47]

We only start the animation by calling `start()` when the animation isn't running. Therefore, we check the animation's state to see if it is stopped. Other states could be `Paused` or `Running`. We want this jump action to be activated whenever the player presses the Space key on their keyboard. Therefore, we expand the switch statement inside the key press event handler by using this code:

[PRE48]

Now the property gets animated but Benjamin will still not jump yet. Therefore, we connect the `jumpFactorChange()` signal to a slot of the scene that handles the jump:

[PRE49]

Inside that function, we calculate the *y* coordinate of the player item to respect the ground level defined by `m_groundLevel`. This is done by subtracting the item's height from the ground level's value since the item's origin point is the top-left corner. Then we subtract the maximum jump height, defined by `m_jumpHeight`, which is multiplied by the actual jump factor. Since the factor is in range from 0 to 1, the new *y* coordinate stays inside the allowed jump height. Then we alter the player item's *y* position by calling `setPos()`, leaving the *x* coordinate the same. Et voilà, Benjamin is jumping!

## Have a go hero – letting the scene handle Benjamin's jump

Of course, we could have done the property animation inside the scene's class without the need to extend `Player` by `QObject`. But this should be an example of how to do it. So try to put the logic of making Benjamin jump to the scene's class. This is, however, more consistent as we already move Benjamin left and right there. Or, also consistent, do it the other way around; move Benjamin's movement to the left and right also to the `Player` class.

# Time for action – keeping multiple animations in sync

If you have a look at how the coins (their class being called `Coin`) are created, you see similar structures. They inherit from `QObject` and `QGraphicsEllipseItem` and define two properties: opacity of type `qreal` and `rect` of type `QRect`. This is done only by the following code:

[PRE50]

No function or slot was added because we simply used built-in functions of `QGraphicsItem` and "redeclared" them as properties. Then, these two properties are animated by two `QPropertyAnimation` objects. One fades the coin out, while the other scales the coin in. To ensure that both animations get started at the same time, we use `QParallelAnimationGroup` as follows:

[PRE51]

## *What just happened?*

After setting up each property animation, we add them to the group animation by calling `addAnimation()` on the group while passing a pointer to the animation we would like to add. Then, when we start the group, `QParallelAnimationGroup` makes sure that all assigned animations start at the same time.

The animations are set up for when the coin explodes. You may want to have a look at the `explode()` function of Coin in the sources. A coin should explode when Benjamin touches the coin.

### Tip

If you want to play animations one after the other you can use `QSequentialAnimationGroup`.

## Item collision detection

Whether the player item collides with a coin is checked by the scene's `checkColliding()` function, which is called after the player item has moved (`movePlayer()`) or after Benjamin jumped (`jumpPlayer()`).

# Time for action – making the coins explode

The implementation of `checkColliding()` looks like this:

[PRE52]

## *What just happened?*

First we call the scene's `QGraphicsScene::collidingItems()` function, which takes the item for which colliding items should be detected as a first argument. With the second, optional argument, you could define how the collision should be detected. The type of that argument is `Qt::ItemSelectionMode`, which was explained earlier. In our case, a list of all the items that collide with `m_player` will be returned. So we loop through that list and check whether the current item is a `Coin` object. This is done by trying to cast the pointer to `Coin.` If it is successful, we explode the coin by calling `explode()`. Calling the `explode()` function multiple times is no problem since it will not allow more than one explosion. This is important since `checkColliding()` will be called after each movement of the player. So the first time the player hits a coin, the coin will explode, but this takes time. During this explosion, the player will most likely be moved again and thus collides with the coin once more. In such a case, `explode()` may be called for a second, third, xth time.

The `collidingItems()` function will always return the background items as well since the player item is above all of them most of the time. To avoid the continuous check if they actually are coins, we use a trick. In the used `BackgroundItem` class for the background items, implement the `QGraphicsItem` item's virtual `shape()` function as follows:

[PRE53]

Since the collision detection is done with the item's shape, the background items can't collide with any other item since their shape is permanently empty. `QPainterPath` itself is a class holding information about graphical shapes. For more information—since we do not need anything special for our game—have a look at the documentation. The class is pretty straightforward.

Had we done the jumping logic inside `Player`, we could have implemented the item collision detection from within the item itself. `QGraphicsItem` also offers a `collidingItems()` function that checks against colliding items with itself. So `scene->collidingItems(item)` is equivalent to `item->collidingItems()`.

If you are only interested in whether a item collides with another item, you can call `collidesWithItem()` on the item passing the other item as an argument.

## Setting up the playing field

The last function we have to discuss is the scene's `initPlayField()` function where all is set up. Here we initialize the sky, trees, ground, and player item. Since there is nothing special, we skip that and look directly at how the coins get initialized:

[PRE54]

In total, we are adding 25 coins. First we calculate the width between `m_minX` and `m_maxX`. That is the space where Benjamin can move. To make it a little bit smaller, we only take 94 percent of that width. Then we set up an invisible item with the size of the virtual world called `m_coins`. This item should be the parent to all coins. Then, in the `for` loop we create a coin and randomly set its *x* and *y* position, ensuring that Benjamin can reach them by calculating the modulo of the available width and of the maximal jump height. After all 25 coins are added, we place the parent item holding all coins on the scene. Since most coins are outside the actual view's rectangle, we also need to move the coins while Benjamin is moving. Therefore, `m_coins` must behave like any other background. For this, we simply add the following code:

[PRE55]

We add the preceding code to the `movePlayer()` function where we also move the sky by the same pattern.

## Have a go hero – extending the game

That is it. This is our little game. Of course, there is much room to improve and extend it. For example, you can add some barricades Benjamin has to jump over. Then, you would have to check if the player item collides with such a barricade item when moving forward, and if so, refuse movement. You have learned all the necessary techniques you need for that task, so try to implement some additional features to deepen your knowledge.

## A third way of animation

Besides `QTimer` and `QPropertyAnimation`, there is a third way to animate the scene. The scene provides a slot called `advance()`. If you call that slot, the scene will forward that call to all items it holds by calling `advance()` on each one. The scene does that twice. First, all item `advance()` functions are called with `0` as an argument. This means that the items are about to advance. Then in the second round, all items are called passing 1 to the item's `advance()` function. In that phase each item should advance, whatever that means; maybe moving, maybe a color change, and so on. The scene's slot advance is typically called by a `QTimeLine` element; with this, you can define how many times during a specific period of time the timeline should be triggered.

[PRE56]

This timeline will emit the signal `frameChanged()` every 5 seconds for 10 times. All you have to do is to connect that signal to the scene's `advance()` slot and the scene will advance 10 times during 50 seconds. However, since all items receive two calls for each advance, this may not be the best animation solution for scenes with a lot of items where only a few should advance.

# Widgets inside Graphics View

In order to show a neat feature of Graphics View, have a look at the following code snippet, which adds a widget to the scene:

[PRE57]

First we create a `QSpinBox` and a `QGraphicsProxyWidget` element, which act as containers for widgets and indirectly inherit `QGraphicsItem.` Then we add the spin box to the the proxy widget by calling `addWidget()`. The ownership of the spin box isn't transferred, but when `QGraphicsProxyWidget` gets deleted, it calls `delete` on all assigned widgets. We thus do not have to worry about that ourselves. The widget you add should be parentless and must not be shown elsewhere. After setting the widget to the proxy, you can treat the proxy widget like any other item. Next, we add it to the scene and apply a transformation for demonstration. As a result we get this:

![Widgets inside Graphics View](img/8874OS_06_19.jpg)

A rotated and scaled spin box on a scene

Since it is a regular item, you can even animate it, for example, with a property animation. Nevertheless, be aware that, originally, Graphics View wasn't designed for holding widgets. So when you add a lot of widgets to the scene, you will quickly notice performance issues, but in most situations it should be fast enough.

If you want to arrange some widgets in a layout, you can use `QGraphicsAnchorLayout`, `QGraphicsGridLayout`, or `QGraphicsLinearLayout`. Create all widgets, create a layout of your choice, add the widgets to that layout, and set the layout to a `QGraphicsWidget` element, which is the base class for all widgets and is easily spoken the `QWidget` equivalent for Graphics View by calling `setLayout()`:

[PRE58]

The scene's `addWidget()` function is a convenience function and behaves in the first usage for `QLineEdit`, as shown in the following code snippet:

[PRE59]

The item with the layout will look like this:

![Widgets inside Graphics View](img/8874OS_06_20.jpg)

# Optimization

Let us now take a look at some of the optimizations we can perform to speed up the scene.

## A binary space partition tree

The scene constantly keeps record of the position of the item in its internal binary space partition tree. Thus, on every move of an item, the scene has to update the tree, an operation that can become quite time-and memory-consuming. This is especially true of scenes with a large number of animated items. On the other hand, the tree enables you to find an item (for example, with `items()` or `itemAt()`) incredibly fast even if you have thousands of items.

So when you do not need any positional information about the items—this also includes collision detection—you can disable the index function by calling `setItemIndexMethod(QGraphicsScene::NoIndex)`. Be aware, however, that a call to `items()` or `itemAt()` results in a loop through all items in order to do the collision detection, which can cause performance problems for scenes with many items. If you cannot relinquish the tree in total, you still can adjust the depth of the tree with `setBspTreeDepth()`, taking the depth as an argument. By default, the scene will guess a reasonable value after it takes several parameters, such as the size and the number of items, into account.

## Caching the item's paint function

If you have items with a time-consuming paint function, you can change the item's cache mode. By default, no rendering is cached. With `setCacheMode()`, you can set the mode to either `ItemCoordinateCache` or to `DeviceCoordinateCache`. The former renders the the item in a cache of a given `QSize` element. The size of that cache can be controlled with the second argument of `setCacheMode()`. So the quality depends on how much space you assign. The cache is then used for every subsequent paint call. The cache is even used for applying transformations. If the quality deteriorates too much, just adjust the resolution by calling `setCacheMode()` again, but with a larger `QSize` element. `DeviceCoordinateCache`, on the other hand, does not cache the item on an item base but rather on a device level. This is therefore optimal for items that do not get transformed all the time, because every new transformation will cause a new caching. Moving the item, however, does not end in a new cache. If you use this cache mode, you do not have to define a resolution with the second argument. The caching is always performed at maximum quality.

## Optimizing the view

Since we are talking about the item's paint function, let's touch on something related. At the beginning, when we discussed the item's appearance and made a black rectangle item, I told you to return the painter as you get. If you have followed this advice, you can call `setOptimizationFlag(DontSavePainterState, true)` on the view. By default, the view ensures that the painter state is saved before calling the item's paint function and that the state gets restored afterward. This will end up saving and restoring the painter state say 50 times if you have a scene with 50 items. If you prevent automatic saving and restoring, keep in mind that now the standard items will alter the painter state. So if you use both standard and custom items, either stay with the default behavior or set `DontSavePainterState`, but then set up the pen and brush with a default value in each item's paint function.

Another flag that can be used with `setOptimizationFlag()` is `DontAdjustForAntialiasing`. By default, the view adjusts the painting area of each item by 2 pixels in all directions. This is useful because when one paints antialiased, one easily draws outside the bounding rectangle. Enable that optimization if you do not paint antialiased or if you are sure your painting will stay inside the bounding rectangle. If you enable this flag and spot painting artifacts on the view, you haven't respected the item's bounding rectangle!

As a further optimization, you can define how the view should update its viewport when the scene changes. You can set the different modes with `setViewportUpdateMode()`. By default (`QGraphicsView::MinimalViewportUpdate`), the view tries to determinate only those areas which need an update and repaints only these. However, sometimes it is more time-consuming to find all the areas that need a redraw than to just paint the entire viewport. This applies if you have many small updates. Then, `QGraphicsView::FullViewportUpdate` is the better choice since it simply repaints the whole viewport. A kind of combination of the last two modes is `QGraphicsView::BoundingRectViewportUpdate`. In this mode, Qt detects all areas that need a redraw and then it redraws a rectangle of the viewport that covers all areas affected by the change. If the optimal update mode changes over time, you can tell Qt to determine the best mode by using `QGraphicsView::SmartViewportUpdate`. The view then tries to find the best update mode.

As a last optimization, you can take advantage of OpenGL. Instead of using the default viewport based on `QWidget`, advise Graphics View to use an OpenGL widget. This way, you can use all the power that comes with OpenGL.

[PRE60]

Unfortunately, you have to do a little more than just putting in this line, but that goes beyond the topic and scope of this chapter. You can, however, find more information about OpenGL and Graphics View in Qt's documentation example under "Boxes" as well as in Rødal's Qt Quarterly article–issue 26–which can be found online at [http://doc.qt.digia.com/qq/qq26-openglcanvas.html](http://doc.qt.digia.com/qq/qq26-openglcanvas.html).

### Note

A general note on optimization: Unfortunately I can't say that you have to do this or that to optimize Graphics View as it highly depends on your system and view/scene. What I can tell you, however, is how to proceed. Once you have finished your game based on Graphics View, measure the performance of your game using a profiler. Make an optimization you think may pay or simply guess and then profile your game again. If the results are better, keep the change; otherwise, reject it. This sounds simple and is the only way optimization can be done. There are no hidden tricks or deeper knowledge. With time, however, your forecasting will get better.

## Pop quiz – mastering Graphics View

After studying this chapter, you should be able to answer these questions as they are important when it comes to designing the components of a game based on Graphics View:

Q1\. What standard items does Qt offer?

Q2\. How is the coordinate system of an item related to the coordinate system of the scene? Next, how is the coordinate system of the scene related to the coordinate system of the view?

Q3\. How can one extend items to use properties as well as signals and slots?

Q4\. How can one create realistic movements with the help of animations?

Q5\. How can Graphics View's performance be improved?

# Summary

In the first part of this chapter, you have learned how the Graphics View architecture works. First, we had a look at the items. There you learned how to create your own items by using `QPainter` and which kinds of standard item Qt has to offer. Later on, we also discussed how to transform these items and what the point of origin for that transformation has to do with it. Next we went through the coordinate system of the items, the scene, and the view. We also saw how these three parts work together, for example. how to put items on a scene. Lastly, we learned how to scale and move the scene inside the view. At the same time, you read about advanced topics, such as taking the zoom level into account when painting an item.

In the second part you, deepened your knowledge about items, about the scene, and about the view. While developing the game, you became familiar with different approaches on how to animate items, and you were taught how to detect collisions. As an advanced topic, you were introduced to parallax scrolling.

After having completed the entire chapter, you should now know almost everything about Graphics View. You are able to create complete custom items, you can alter or extend standard items, and with the information about the level of detail you even have the power to alter an item's appearance, depending on its zoom level. You can transform items and the scene, and you can animate items and, thus, the entire scene.

Furthermore, as you have seen while developing the game, your skills are good enough to develop a jump-and-run game with parallax scrolling as it is used in highly professional games. To keep your game fluid and highly responsive, finally we saw some tricks on how to get the most out of Graphics View.

In order to build a bridge to the world of widgets, you also learned how to incorporate items based on `QWidget` into Graphics View. With that knowledge, you can create modern, widget-based user interfaces.