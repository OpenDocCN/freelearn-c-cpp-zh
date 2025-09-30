# Custom 2D Graphics with Graphics View

Widgets are great for designing graphical user interfaces, but they are not convenient if you want to use multiple objects with custom painting and behavior together, such as in a 2D game. You will also run into problems if you wish to animate multiple widgets at the same time, by constantly moving them around in the application. For these situations, or generally for frequently transforming 2D graphics, Qt offers you Graphics View. In this chapter, you will learn the basics of the Graphics View architecture and its items. You will also learn how to combine widgets with Graphics View items.

The main topics covered in this chapter are as follows:

*   Graphics View architecture
*   Coordinate systems
*   Standard graphics items
*   Pens and brushes
*   Useful features of Graphics View
*   Creating custom items
*   Event handling
*   Embedding widgets in the view
*   Optimizations

# Graphics View architecture

The Graphics View Framework is part of the Qt Widgets module and provides a higher level of abstraction useful for custom 2D graphics. It uses software rendering by default, but it is very optimized and extremely convenient to use. Three components form the core of Graphics View, as shown:

*   An instance of `QGraphicsView`, which is referred to as **View**
*   An instance of `QGraphicsScene`, which is referred to as **Scene**
*   Instances of `QGraphicsItem`, which are referred to as **Items**

The usual workflow is to first create a couple of items, add them to a scene, and then show that scene on a view:

![](img/6eb777be-ab33-4d5e-811b-a337b1ab917f.png)

After that, you can manipulate items from the code and add new items, while the user also has the ability to interact with visible items.

Think of the **items** as Post-it notes. You take a note and write a message on it, paint an image on it, both write and paint on it, or, quite possibly, just leave it blank. Qt provides a lot of item classes, all of which inherit `QGraphicsItem`. You can also create your own item classes. Each class must provide an implementation of the `paint()` function, which performs painting of the current item, and the `boundingRect()` function, which must return the boundary of the area the `paint()` function paints on.

What is the **scene**, then? Well, think of it as a larger sheet of paper on to which you attach your smaller Post-its, that is, the notes. On the scene, you can freely move the items around while applying funny transformations to them. It is the scene's responsibility to correctly display the items' position and any transformations applied to them. The scene further informs the items about any events that affect them.

Last, but not least, let's turn our attention to the **view**. Think of the view as an inspection window or a person who holds the paper with the notes in their hands. You can see the paper as a whole, or you can only look at specific parts. Also, as a person can rotate and shear the paper with their hands, so the view can rotate and shear the scene's content and do a lot more transformations with it. `QGraphicsView` is a widget, so you can use the view like any other widget and place it into layouts for creating neat graphical user interfaces.

You might have looked at the preceding diagram and worried about all the items being outside the view. Aren't they wasting CPU time? Don't you need to take care of them by adding a so-called *view frustum culling* mechanism (to detect which item not to draw/render because it is not visible)? Well, the short answer is "no", because Qt is already taking care of this.

# Time for action – Creating a project with a Graphics View

Let's put all these components together in a minimalistic project. From the Welcome screen, click on the New Project button and select Qt Widgets Application again. Name the project `graphics_view_demo`, select the correct kit, uncheck the Generate form checkbox, and finish the wizard. We don't actually need the `MainWindow` class that was generated for us, so let's delete it from the project. In the project tree, locate `mainwindow.h` and select Remove file in the context menu. Enable the Delete file permanently checkbox and click on OK. This will result in deleting the `mainwindow.h` file from the disk and removing its name from the `graphics_view_demo.pro` file. If the file was open in Qt Creator, it will suggest that you close it. Repeat the process for `mainwindow.cpp`.

Open the `main.cpp` file, remove `#include "mainwindow.h"`, and write the following code:

```cpp
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QGraphicsScene scene;
    QGraphicsRectItem *rectItem = 
        new QGraphicsRectItem(QRectF(0, 0, 100, 50));
    scene.addItem(rectItem);
    QGraphicsEllipseItem *circleItem =
        new QGraphicsEllipseItem(QRect(0, 50, 25, 25));
    scene.addItem(circleItem);
    QGraphicsSimpleTextItem *textItem =
        new QGraphicsSimpleTextItem(QObject::tr("Demo"));
    scene.addItem(textItem);
    QGraphicsView view(&scene);
    view.show();
    return a.exec();
}
```

When you run the project, you should get the following result:

![](img/8bcc98a5-4f2e-4bc6-b1fd-c80d34c92bde.png)

# What just happened?

Our new project is so simple that all its code is located in the `main()` function. Let's examine the code. First, we create a `QApplication` object, as in any Qt Widgets project. Next, we create a scene object and three instances of different item classes. The constructor of each item class accepts an argument that defines the content of the item:

*   The `QGraphicsRectItem` constructor receives a `QRectF` object that contains the coordinates of the rectangle
*   The `QGraphicsEllipseItem` constructor, similarly, receives a `QRectF` object that defines the bounding rectangle of the circle
*   The `QGraphicsSimpleTextItem` constructor receives the text to display

`QRectF` is basically a helpful struct with four fields that allow us to specify four coordinates of the rectangle's boundaries (left, top, width, and height). Qt also offers `QPointF` that contains *x* and *y* coordinates of a point, `QLineF` that contains *x* and *y* coordinates of two ends of a line, and `QPolygonF` that contains a vector of points. The `F` letter stands for "floating point" and indicates that these classes hold real numbers. They are widely used in Graphics View, as it always works with floating point coordinates. The corresponding classes without `F` (`QPoint`, `QRect`, and so on) store integer coordinates and are more useful when working with widgets.

After creating each item, we use the `QGraphicsScene::addItem` function to add the item to the scene. Finally, we create a `QGraphicsView` object and pass the pointer to the scene to its constructor. The `show()` method will make the view visible, as it does for any `QWidget`. The program ends with an `a.exec()` call, necessary to start the event loop and keep the application alive.

The scene takes ownership of the items, so they will be automatically deleted along with the scene. This also means that an item can only be added to one single scene. If the item was previously added to another scene, it gets removed from there before it is added to the new scene.

If you want to remove an item from a scene without setting it directly to another scene or without deleting it, you can call `scene.removeItem(rectItem)`. Be aware, however, that now it is your responsibility to delete `rectItem` to free the allocated memory!

Examine the resulting window and compare it to the coordinates of the rectangles in the code (the `QRectF` constructor we use accepts four arguments in the following order: left, top, width, height). You should be able to see that all three elements are positioned in a single coordinate system, where the *x* axis points to the right and the *y* axis points down. We didn't specify any coordinates for the text item, so it's displayed at the **origin** point (that is, the point with zero coordinates), next to the top-left corner of the rectangle. However, that (0, 0) point does not correspond to the top-left corner of the window. In fact, if you resize the window, you'll note that the origin has shifted relative to the window, because the view tries to display the scene's content as centered.

# Coordinate systems

To use Graphics View correctly, you need to understand how the coordinate systems in this framework work. We'll go through all the levels of hierarchy and see how we can change the positioning of items and the whole scene, on each level. We will provide examples of the code that you can paste into our demo project and examine the effect.

# The item's coordinate system

Each item has its own coordinate system. In our example of Post-it notes, the content of each note is defined relative to the top-left corner of the note. No matter how you move or rotate the item, these coordinates remain the same. The coordinates of a drawn object can usually be passed to the constructor of the class, like we did in our demo project, or to a special setter function (for example, `rectItem->setRect(0, 10, 20, 25)`). These are coordinates in the item's coordinate system.

Some classes, such as `QGraphicsSimpleTextItem`, do not provide the ability to change the coordinates of the content, so they're always positioned at the origin of the item's coordinate system. This is not a problem at all; as we'll see next, there are ways to change the visible position of the content.

If you try to create your own graphics item class (we'll get to it later in this chapter), you'll need to implement the `paint()` and `boundingRect()` functions, and they always operate in the item's coordinate system. That's right, when you're painting the content, you can just pretend that your item will never be moved or transformed. When that actually happens, Qt will take care of transforming paint operations for you. Additionally, coordinates in any events the item receives (for example, coordinates of a mouse button click) are expressed in the item's coordinate system.

# The scene's coordinate system

Any item can be moved in the scene using the `setPos()` function. Try to call `textItem->setPos(50, 50)` and verify that the text was moved in the scene. Technically, this operation changes the **transformation** between the item's coordinate system and the scene's coordinate system. A convenience function called `moveBy()` allows you to shift the position by specified amounts.

An item can also be rotated with `setRotation()` and scaled with `setScale()`. Try calling `textItem->setRotation(20)` to see this in action. If you need a more advanced transformation, such as shearing, or you want to perform multiple translations in a particular order, you can create a `QTransform` object, apply the required transformations, and use the `setTransform()` function of the item.

The `setRotation()` function accepts `qreal` as the argument value, which is usually a typedef for `double`. The function interprets the number as degrees for a clockwise rotation around the *z* coordinate. If you set a negative value, a counter-clockwise rotation is performed. Even if it does not make much sense, you can rotate an item by 450 degrees, which will result in a rotation of 90 degrees.

# The viewport's coordinate system

The view consists of the **viewport** and two scrollbars. The viewport is a subwidget that actually contains the content of the scene. The view performs conversion from the scene coordinates to the viewport coordinates based on multiple parameters.

First, the view needs to know the bounding rectangle of everything we could want to see in the scene. It's called the **scene rect** and is measured in the scene's coordinate system. By default, the scene rect is the bounding rectangle of all items that were added at the scene since it was created. This is usually fine, but if you move or delete an item, that bounding rectangle will not shrink (because of performance reasons), so it may result in showing a lot of unwanted empty space. Luckily, in such cases, you can set the scene rect manually using the `setSceneRect` function of the scene or view.

The difference between `QGraphicsScene::setSceneRect` and `QGraphicsView::setSceneRect` is usually marginal, since you will typically have one view per scene. However, it is possible to have multiple views for a single scene. In this case, `QGraphicsScene::setSceneRect` sets the scene rect for all views, and `QGraphicsView::setSceneRect` allows you to override the scene rect for each view.

If the area corresponding to the scene rect is small enough to fit in the viewport, the view will align the content according to the view's `alignment` property. As we saw earlier, it positions the content at the center by default. For example, calling `view.setAlignment(Qt::AlignTop | Qt::AlignLeft)` will result in the scene staying in the upper-left corner of the view.

If the scene rect area is too large to fit in the viewport, the horizontal or vertical scrollbars appear by default. They can be used to scroll the view and see any point inside the scene rect (but not beyond it). The presence of scrollbars can also be configured using the  `horizontalScrollBarPolicy` and the `verticalScrollBarPolicy` properties of the view.

Try to call `scene.setSceneRect(0, 20, 100, 100)` and see how the view acts when resizing the window. If the window is too small, the top part of the scene will no longer be visible. If the window is large enough and the view has the default alignment, the top part of the scene will be visible, but only the defined scene rect will be centered, with no regard to the items outside of it.

The view provides the ability to transform the entire scene. For example, you can call `view.scale(5, 5)` to make everything five times larger, `view.rotate(20)` to rotate the scene as a whole, or `view.shear(1, 0)` to shear it. As with items, you can apply a more complex transformation using `setTransform()`.

You may note that Graphics View (and Qt Widgets in general) uses a **left-handed** coordinate system by default, where *x* axis points right and *y* axis points down. However, OpenGL and science-related applications usually use a **right-handed** or standard coordinate system, where *y* axis points up. If you need to change the direction of the *y* axis, the simplest solution is to transform the view by calling `view.scale(1, -1)`.

# Origin point of the transformation

In our next example, we will create a cross at (0, 0) point and add a rectangle to the scene:

![](img/29c10c5a-6843-4284-88d8-08444454723b.png)

You can do it with the following code:

```cpp
scene.addLine(-100, 0, 100, 0);
scene.addLine(0, -100, 0, 100);
QGraphicsRectItem* rectItem = scene.addRect(50, 50, 50, 50);
```

In this code, we use the `addLine()` and `addRect()` convenience functions. This is the same as creating a `QGraphicsLineItem` or `QGraphicsRectItem` and adding it to the scene manually.

Now, imagine that you want to rotate the rectangle by 45 degrees to produce the following result:

![](img/b9f23d56-4ce9-44ac-9718-77d4cfb7be27.png)

A straightforward attempt to do it will use the `setRotation()` method:

```cpp
QGraphicsRectItem* rectItem = scene.addRect(50, 50, 50, 50);
rectItem->setRotation(45);
```

However, if you try to do that, you will get the following result:

![](img/de15be54-8700-4b6f-beb0-8a9516a4baf2.png)

# What just happened?

Most transformations depend on the **origin** point of the coordinate system. For rotation and scaling, the origin point is the only point that remains in place. In the preceding example, we used a rectangle with the top-left corner at (50, 50) and the size of (50, 50). These coordinates are in the item's coordinate system. Since we originally didn't move the item, the item's coordinate system was the same as the scene's coordinate system, and the origin point is the same as the scene's origin (it's the point marked with the cross). The applied rotation uses (0, 0) as the center of rotation, thus providing an unexpected result.

There are multiple ways to overcome this problem. The first way is to change the transform's origin point:

```cpp
QGraphicsRectItem* rectItem = scene.addRect(50, 50, 50, 50);
rectItem->setTransformOriginPoint(75, 75);
rectItem->setRotation(45);
```

This code produces the rotation we want, because it changes the origin point used by the `setRotation()` and `setScale()` functions. Note that the item's coordinate system was not translated, and (75, 75) point continues to be the center of the rectangle in the item's coordinates.

However, this solution has its limitations. If you use `setTransform()` instead of `setRotation()`, you will get the unwanted result again:

```cpp
QGraphicsRectItem* rectItem = scene.addRect(50, 50, 50, 50);
rectItem->setTransformOriginPoint(75, 75);
QTransform transform;
transform.rotate(45);
rectItem->setTransform(transform);
```

Another solution is to set up the rectangle in such a way that its center is in the origin of the item's coordinate system:

```cpp
QGraphicsRectItem* rectItem = scene.addRect(-25, -25, 50, 50);
rectItem->setPos(75, 75);
```

This code uses completely different rectangle coordinates, but the result is exactly the same as in our first example. However, now, (75, 75) point in the scene's coordinates corresponds to (0, 0) point in the item's coordinates, so all transformations will use it as the origin:

```cpp
QGraphicsRectItem* rectItem = scene.addRect(-25, -25, 50, 50);
rectItem->setPos(75, 75);
rectItem->setRotation(45);
```

This example shows that it is usually more convenient to set up the items so that their origin point corresponds to their actual location.

# Have a go hero – Applying multiple transformations

To understand the concept of transformations and their origin point, try to apply `rotate()` and `scale()` to an item. Also, change the point of origin and see how the item will react. As a second step, use `QTransform` in conjunction with `setTransform()` to apply multiple transformations to an item in a specific order.

# Parent–child relationship between items

Imagine that you need to create a graphics item that contains multiple geometric primitives, for example, a circle inside a rectangle. You can create both items and add them to the scene individually, but this solution is inconvenient. First, when you need to remove that combination from the scene, you would need to manually delete both items. However, more importantly, when you need to move or transform the combination, you will need to calculate positions and complex transformations for each graphics item.

Fortunately, graphics items do not have to be a flat list of items added directly into the scene. Items can be added into any other items, forming a parent–child relationship very similar to the relationship of `QObject` that we observed in the last chapter:

![](img/a5c4acc7-403c-40cc-a181-ea784a629bfa.png)

Adding an item as a child of another item has the following consequences:

*   When the parent item is added to the scene, the child item automatically becomes part of that scene, so there is no need to call `QGraphicsScene::addItem()` for it.
*   When the parent item is deleted, its children are also deleted.
*   When the parent item is hidden using the `hide()` or `setVisible(false)` functions, the child items will also be hidden.
*   Most importantly, the child's coordinate system is derived from the parent's coordinate system instead of the scene's. This means that when the parent is moved or transformed, all children are also affected. The child's position and transformations are relative to the parent's coordinate system.

You can always check whether an item has a parent using the `parentItem()` function, and check the returned `QGraphicsItem` pointer against `nullptr`, which means that the item does not have a parent. To figure out whether there are any children, call the `childItems()` function on the item. A `QList` method with the `QGraphicsItem` pointers to all child items is returned.

For a better understanding of `pos()` and the involved coordinate systems, think of post-it notes again. If you put a note on a larger sheet of paper and then had to determine its exact position, how would you do it? Probably like this: "The note's upper-left corner is positioned 3 cm to the right and 5 cm to the bottom from the paper's top-left edge". In the Graphics View world, this will correspond to a parentless item whose `pos()` function returns a position in the scene coordinates, since the item's origin is directly pinned to the scene. On the other hand, say you put a note A on top of a (larger) note B, which is already pinned on a paper, and you have to determine A's position; how would you describe it this time? Probably by saying that note A is placed on top of note B or "2 cm to the right and 1 cm to the bottom from the top-left edge of note B". You most likely wouldn't use the underlying paper as a reference since it is not the next point of reference. This is because if you move note B, A's position regarding the paper will change, whereas A's relative position to B still remains unchanged. To switch back to Graphics View, the equivalent situation is an item that has a parent item. In this case, the `pos()` function's returned value is expressed in the coordinate system of its parent. So, `setPos()` and `pos()` specify the position of the item's origin in relation to the next (higher) point of reference. This can be the scene or the item's parent item.

Keep in mind, however, that changing an item's position does not affect the item's internal coordinate system.

For widgets, the child always occupies a subarea of its direct parent. For graphics items, such a rule does not apply by default. A child item can be displayed outside the bounding rectangle or visible content of the parent. In fact, a common situation is when the parent item does not have any visual content by itself and only serves as a container for a set of primitives belonging to one object.

# Time for action – Using child items

Let's try to make an item containing multiple children. We want to create a rectangle with a filled circle in each corner and be able to move and rotate it as a whole, like this:

![](img/94aa7f68-0fd5-43ec-bd98-c60c8da3a39b.png)

First, you need to create a function that creates a single complex rectangle, by using the following code:

```cpp
QGraphicsRectItem *createComplexItem(
    qreal width, qreal height, qreal radius) 
{
    QRectF rect(-width / 2, -height / 2, width, height);
    QGraphicsRectItem *parent = new QGraphicsRectItem(rect);
    QRectF circleBoundary(-radius, -radius, 2 * radius, 2 * radius);
    for(int i = 0; i < 4; i++) {
        QGraphicsEllipseItem *child =
            new QGraphicsEllipseItem(circleBoundary, parent);
        child->setBrush(Qt::black);
        QPointF pos;
        switch(i) {
        case 0:
            pos = rect.topLeft();
            break;
        case 1:
            pos = rect.bottomLeft();
            break;
        case 2:
            pos = rect.topRight();
            break;
        case 3:
            pos = rect.bottomRight();
            break;
        }
        child->setPos(pos);
    }
    return parent;
}
```

We start with creating a `QRectF` variable that contains the rectangle coordinates in the item's coordinate system. Following the tip we provided earlier, we create a rectangle centered at the origin point. Next, we create a rectangle graphics item called `parent`, as usual. The `circleBoundary` rectangle is set to contain the boundary rect of a single circle (again, the center is at the origin point). When we create a new `QGraphicsEllipseItem` for each corner, we pass `parent` to the constructor, so the new circle item is automatically added as a child of the rectangle item.

To set up a child circle, we first use the `setBrush()` function that enables filling of the circle. This function expects a `QBrush` object that allows you to specify an advanced filling style, but in our simple case, we use an implicit conversion from the `Qt::GlobalColor` enum to `QBrush`. You will learn more about brushes later in this chapter.

Next, we select a different corner of the rectangle for each circle and call `setPos()` to move the circle to that corner. Finally, we return the parent item to the caller.

You can use this function as follows:

```cpp
QGraphicsRectItem *item1 = createComplexItem(100, 60, 8);
scene.addItem(item1);

QGraphicsRectItem *item2 = createComplexItem(100, 60, 8);
scene.addItem(item2);
item2->setPos(200, 0);
item2->setRotation(20);
```

Note that when you call `setPos()`, the circles are moved along with the parent item, but the `pos()` values of the circles do not change. This is the consequence of the fact that `pos()` means the position relative to the parent item (or the scene's origin, if there is no parent item). When the rectangle is rotated, circles rotate with it, as if they were fixed to the corners. If the circles weren't children of the rectangle, positioning them properly, in this case, would be a more challenging task.

# Have a go hero – Implementing the custom rectangle as a class

In this example, we avoided creating a class for our custom rectangle to keep the code as simple as possible. Following the principles of object-oriented programming, subclassing `QGraphicsRectItem` and creating children items in the constructor of the new class is a good idea. Doing this doesn't require anything you don't already know. For example, when subclassing `QGraphicsRectItem`, you don't need to implement any virtual functions, because they are all properly implemented in the base classes.

# Conversions between coordinate systems

If an item is simply moved using `setPos()`, conversion from the item's coordinates to the scene coordinates is as simple as `sceneCoord = itemCoord + item->pos()`. However, this conversion quickly becomes very complex when you use transformations and parent–child relationships, so you should always use dedicated functions to perform such conversions. `QGraphicsItem` provides the following functions:

| **Function** | **Description** |
|  `mapToScene(`
`  const QPointF &point)` | Maps the point `point` that is in the item's coordinate system to the corresponding point in the scene's coordinate system. |
| `scenePos()` | Maps the item's origin point to the scene's coordinate system. This is the same as `mapToScene(0, 0)`. |
| `sceneBoundingRect()` | Returns the item's bounding rectangle in the scene's coordinate system. |
| `mapFromScene(` `  const QPointF &point)` | Maps the point `point` that is in the scene's coordinate system to the corresponding point in the item's coordinate system. This function is the reverse function to `mapToScene()`. |
| `mapToParent(` `  const QPointF &point)` | Maps the point `point` that is in the item's coordinate system to the corresponding point in the coordinate system of the item's parent. If the item does not have a parent, this function behaves like `mapToScene()`; thus, it returns the corresponding point in the scene's coordinate system. |
| `mapFromParent(` `  const QPointF &point)` | Maps the point `point` that is in the coordinate system of the item's parent to the corresponding point in the item's own coordinate system. This function is the reverse function to `mapToParent()`. |
| `mapToItem(` `  const QGraphicsItem *item,`
` const QPointF &point)` | Maps the point `point` that is in the item's own coordinate system to the corresponding point in the coordinate system of the item `item`. |
| `mapFromItem(` `  const QGraphicsItem *item,`
` const QPointF  &point)` | Maps the point `point` that is in the coordinate system of the item `item` to the corresponding point in the item's own coordinate system. This function is the reverse function to `mapToItem()`. |

What is great about these functions is that they are not only available for `QPointF`. The same functions are also available for `QRectF`, `QPolygonF`, and `QPainterPath`, not to mention that there are some convenience functions:

*   If you call these functions with two numbers of the `qreal` type, the numbers are interpreted as the *x* and *y* coordinates of a `QPointF` pointer
*   If you call the functions with four numbers, the numbers are interpreted as the *x* and *y* coordinates and the width and height of a `QRectF` parameter

The `QGraphicsView` class also contains a set of `mapToScene()` functions that map coordinates from the viewport's coordinate system to the scene coordinates and `mapFromScene()` functions that map the scene coordinates to the viewport coordinates.

# Overview of functionality

You should now have some understanding of Graphics View's architecture and transformation mechanics. We will now describe some easy-to-use functionality that you'll probably need when creating a Graphics View application.

# Standard items

In order to effectively use the framework, you need to know what graphics item classes it provides. It's important to identify the classes you can use to construct the desirable picture and resort to creating a custom item class, only if there is no suitable item or you need better performance. Qt comes with the following standard items that make your life as a developer much easier:

| **Standard item** | **Description** |
| `QGraphicsLineItem` | Draws a line. You can define the line with `setLine(const QLineF&)`. |
| `QGraphicsRectItem` | Draws a rectangle. You can define the rectangle's geometry with `setRect(const QRectF&)`. |
| `QGraphicsEllipseItem` | Draws an ellipse or an ellipse segment. You can define the rectangle within which the ellipse is being drawn with `setRect(const QRectF&)`. Additionally, you can define whether only a segment of the ellipse should be drawn by calling `setStartAngle(int)` and `setSpanAngle(int)`. The arguments of both functions are in sixteenths of a degree. |
| `QGraphicsPolygonItem` | Draws a polygon. You can define the polygon with `setPolygon(const QPolygonF&)`. |
| `QGraphicsPathItem` | Draws a path, that is, a set of various geometric primitives. You can define the path with `setPath(const QPainterPath&)`. |
| `QGraphicsSimpleTextItem` | Draws plain text. You can define the text with `setText(const QString&)` and the font with `setFont(const QFont&)`. This item doesn't support rich formatting. |
| `QGraphicsTextItem` | Draws formatted text. Unlike `QGraphicsSimpleTextItem`, this item can display HTML stored in a `QTextDocument` object. You can set HTML with `setHtml(const QString&)` and the document with `setDocument(QTextDocument*)`. `QGraphicsTextItem` can even interact with the displayed text so that text editing or URL opening is possible. |
| `QGraphicsPixmapItem` | Draws a pixmap (a raster image). You can define the pixmap with `setPixmap(const QPixmap&)`. It's possible to load pixmaps from local files or resources, similar to icons (refer to [Chapter 3](ebffc011-752f-4dbe-a383-0917a002841d.xhtml), *Qt GUI Programming*, for more information about resources). |
| `QGraphicsProxyWidget` | Draws an arbitrary `QWidget` and allows you to interact with it. You can set the widget with `setWidget(QWidget*)`. |

As we already saw, you can usually pass the content of the item to the constructor instead of calling a setter method such as `setRect()`. However, keep in mind that compact code may be harder to maintain than code that sets all the variables through setter methods.

For most items, you can also define which pen and which brush should be used. The pen is set with `setPen()` and the brush with `setBrush()` (we've already used it for the child circles in the previous example). These two functions, however, do not exist for `QGraphicsTextItem`. To define the appearance of a `QGraphicsTextItem` item, you have to use `setDefaultTextColor()` or HTML tags supported by Qt. `QGraphicsPixmapItem` has no similar methods, as the concepts of pen and brush cannot be applied to pixmaps.

Use `QGraphicsSimpleTextItem` wherever possible and try to avoid `QGraphicsTextItem,` if it is not absolutely necessary. The reason is that `QGraphicsTextItem` is a subclass of `QObject` and uses `QTextDocument`, which is basically an HTML engine (although quite limited). This is way heavier than an average graphics item and is definitely too much overhead for displaying simple text.

It is generally easier to use standard items than to implement them from scratch. Whenever you will use Graphics View, ask yourself these questions: Which standard items are suited for my specific needs? Am I re-inventing the wheel over and over again? However, from time to time, you need to create custom graphics items, and we'll cover this topic later in this chapter.

# Anti-aliasing

If you look at the result of the previous screenshot, you can probably note that the drawing looks pixelated. This happens because each pixel in a line is completely black, and all the surrounding pixels are completely white. The physical display's resolution is limited, but a technique called **anti-aliasing** allows you to produce more smooth images with the same resolution. When drawing a line with anti-aliasing, some pixels will be more or less blacker than others, depending on how the line crosses the pixel grid.

You can easily enable anti-aliasing in Graphics View using the following code:

```cpp
view.setRenderHint(QPainter::Antialiasing);
```

With the anti-aliasing flag turned on, the painting is done much more smoothly:

![](img/8e604129-e7ec-41fd-83f2-046bf4eb0d4c.png)

However, lines in the rectangle on the left now look thicker. This happens because we used lines with integer coordinates and 1 pixel width. Such a line is located exactly on the border between two rows of pixels, and when anti-aliased, both adjacent rows of pixels will be partially painted. This can be fixed by adding 0.5 to all coordinates:

```cpp
QRectF rect(-width / 2, -height / 2, width, height);
rect.translate(0.5, 0.5);
QGraphicsRectItem *parent = new QGraphicsRectItem(rect);
```

Now the line is positioned right in the middle of a pixel row, so it only occupies a single row:

![](img/fee58494-72a0-401e-b359-2751ce199ed7.png)

Another solution is to implement a custom item class and disable anti-aliasing when painting a horizontal or vertical line.

`QGraphicsView` also supports the `QPainter::TextAntialiasing` flag that enables anti-aliasing when drawing text, and the `QPainter::SmoothPixmapTransform` flag that enables smooth pixmap transformation. Note the anti-aliasing and smoothing impact performance of your application, so use them only when needed.

# Pens and brushes

The pen and brush are two attributes that define how different drawing operations are performed. The pen (represented by the `QPen` class) defines the outline, and the brush  (represented by the `QBrush` class) fills the drawn shapes. Each of them is really a set of parameters. The most simple one is the color defined, either as a predefined global color enumeration value (such as `Qt::red` or `Qt::transparent`), or an instance of the `QColor` class. The effective color is made up of four attributes: three color components (red, green, and blue) and an optional alpha channel value that determines the transparency of the color (the larger the value, the more opaque the color). By default, all components are expressed as 8-bit values (0 to 255) but can also be expressed as real values representing a percentage of the maximum saturation of the component; for example, 0.6 corresponds to 153 (0.6⋅255). For convenience, one of the `QColor` constructors accepts hexadecimal color codes used in HTML (with `#0000FF` being an opaque blue color) or even bare color names (for example, `blue`) from a predefined list of colors returned by a static function—`QColor::colorNames()`. Once a color object is defined using RGB components, it can be queried using different color spaces (for example, CMYK or HSV). Also, a set of static methods are available that act as constructors for colors expressed in different color spaces.

For example, to construct a clear magenta color any of the following expressions can be used:

*   `QColor("magenta")`
*   `QColor("#FF00FF")`
*   `QColor(255, 0, 255)`
*   `QColor::fromRgbF(1, 0, 1)`
*   `QColor::fromHsv(300, 255, 255)`
*   `QColor::fromCmyk(0, 255, 0, 0)`
*   `Qt::magenta`

Apart from the color, `QBrush` has two additional ways of expressing the fill of a shape. You can use `QBrush::setTexture()` to set a pixmap that will be used as a stamp or `QBrush::setGradient()` to make the brush use a gradient to do the filling. For example, to use a gradient that goes diagonally and starts as yellow in the top-left corner of the shape, becomes red in the middle of the shape, and ends as magenta at the bottom-right corner of the shape, the following code can be used:

```cpp
QLinearGradient gradient(0, 0, width, height);
gradient.setColorAt(0,   Qt::yellow);
gradient.setColorAt(0.5, Qt::red);
gradient.setColorAt(1.0, Qt::magenta);
QBrush brush = gradient; 
```

When used with drawing a rectangle, this code will give the following output:

![](img/507836c9-026d-43d4-a40e-2906b785ae8f.png)

Qt can handle linear (`QLinearGradient`), radial (`QRadialGradient`), and conical (`QConicalGradient`) gradients. Qt provides a Gradients example (shown in the following screenshot) where you can see different gradients in action:

![](img/d0639b60-4645-4d39-81b8-cbdc0c722790.png)

As for the pen, its main attribute is its width (expressed in pixels), which determines the thickness of the shape outline. A pen can, of course, have a color set but, in addition to that, you can use any brush as a pen. The result of such an operation is that you can draw thick outlines of shapes using gradients or textures.

There are three more important properties for a pen. The first is the pen style, set using `QPen::setStyle()`. It determines whether lines drawn by the pen are continuous or divided in some way (dashes, dots, and so on). You can see the available line styles here:

![](img/0f854294-e767-4d1f-a34a-ee44fb5137fa.png)

The second attribute is the cap style, which can be flat, square, or round. The third attribute—the join style—is important for polyline outlines and dictates how different segments of the polyline are connected. You can make the joins sharp (with `Qt::MiterJoin` or `Qt::SvgMiterJoin`), round (`Qt::RoundJoin`), or a hybrid of the two (`Qt::BevelJoin`). You can see the different pen attribute configurations (including different join and cap styles) in action by launching the Path Stroking example shown in the following screenshot:

![](img/d6804e9a-8cc2-44e2-9be5-2cbaf2cb9ff5.png)

# Item selection

The scene supports the ability of selecting items, similar to how you select files in a file manager. To be selectable, an item must have the `QGraphicsItem::ItemIsSelectable` flag turned on. Try to add `parent->setFlag(QGraphicsItem::ItemIsSelectable, true)` to the `createComplexItem()` function we created earlier. Now, if you run the application and click on a rectangle, it is selected, which is indicated by dashed lines:

![](img/8e77e8b5-c994-4b3d-b06b-932f690e8da7.png)

You can use the *Ctrl* button to select multiple items at once. Alternatively, you can call `view.setDragMode(QGraphicsView::RubberBandDrag)` to activate the rubber band selection for the view.

Another useful drag mode of the Graphics View is `ScrollHandDrag`. It allows you to scroll the view by dragging the scene with the left mouse button, without the need to use scrollbars.

Besides that, there are different ways to select items programmatically. There is the item's `QGraphicsItem::setSelected()` function, which takes a `bool` value to toggle the selection state on or off, or you can call `QGraphicsScene::setSelectionArea()` on the scene, which takes a `QPainterPath` parameter as an argument, in which case all items within the area are selected.

With the scene's `QGraphicsScene::selectedItems()` function, you can query the actual selected items. The function returns a `QList` holding `QGraphicsItem` pointers to the selected items. For example, calling `QList::count()` on that list will give you the number of selected items. To clear the selection, call `QGraphicsScene::clearSelection()`. To query the selection state of an item, use `QGraphicsItem::isSelected()`, which returns `true` if the item is selected and `false` otherwise.

Another interesting flag of `GraphicsItem` is `ItemIsMovable`. It enables you to drag the item within the scene by holding it with the left mouse button, effectively changing the `pos()` of the item. Try to add `parent->setFlag(QGraphicsItem::ItemIsMovable, true)` to our `createComplexItem` function and drag around the rectangles.

# Keyboard focus in graphics scene

The scene implements the concept of focus that works similar to keyboard focus in widgets. Only one item can have focus at a time. When the scene receives a keyboard event, it is dispatched to the focus item.

To be focusable, an item must have the `QGraphicsItem::ItemIsFocusable` flag enabled:

```cpp
item1->setFlag(QGraphicsItem::ItemIsFocusable, true);
item2->setFlag(QGraphicsItem::ItemIsFocusable, true);
```

Then, an item can be focused by a mouse click. You can also change the focused item from the code:

```cpp
item1->setFocus();
```

Another way to set the focus is to use the scene's `QGraphicsScene::setFocusItem()` function, which expects a pointer to the item you like to focus as a parameter. Every time an item gains focus, the previously focused item (if any) will automatically lose focus.

To determine whether an item has focus, you again have two possibilities. One is that you can call `QGraphicsItem::hasFocus()` on an item, which returns `true` if the item has focus or `false` otherwise. Alternatively, you can get the actual focused item by calling the scene's `QGraphicsScene::focusItem()` method. On the other hand, if you call the item's `QGraphicsItem::focusItem()` function, the focused item is returned if the item itself or any descendant item has focus; otherwise, `nullptr` is returned. To remove focus, call `clearFocus()` on the focused item or click somewhere in the scene's background or on an item that cannot get focus.

If you want a click on the scene's background not to cause the focused item to lose its focus, set the scene's `stickyFocus` property to `true`.

# Painter paths

If you want to create a graphics item that consists of multiple geometric primitives, creating multiple `QGraphicsItem` objects seems to be tedious. Fortunately, Qt provides a  `QGraphicsPathItem` class that allows you to specify a number of primitives in a `QPainterPath` object. `QPainterPath` allows you to "record" multiple painting instructions (including filling, outlining, and clipping), and then efficiently reuse them multiple times.

# Time for action – Adding path items to the scene

Let's paint a few objects consisting of a large number of lines:

```cpp
static const int SIZE = 100;
static const int MARGIN = 10;
static const int FIGURE_COUNT = 5;
static const int LINE_COUNT = 500;
for(int figureNum = 0; figureNum < FIGURE_COUNT; ++figureNum) {
    QPainterPath path;
    path.moveTo(0, 0);
    for(int i = 0; i < LINE_COUNT; ++i) {
        path.lineTo(qrand() % SIZE, qrand() % SIZE);
    }
    QGraphicsPathItem *item = scene.addPath(path);
    item->setPos(figureNum * (SIZE + MARGIN), 0);
}
```

For each item, we first create a `QPainterPath` and set the current position to (0, 0). Then, we use the `qrand()` function to generate random numbers, apply the modulus operator (`%`) to produce a number from 0 to `SIZE` (excluding `SIZE`), and feed them to the `lineTo()` function that strokes a line from the current position to the given position and sets it as the new current position. Next, we use the `addPath()` convenience function that creates a `QGraphicsPathItem` object and adds it to the scene. Finally, we use `setPos()` to move each item to a different position in the scene. The result looks like this:

![](img/4ad6b1a5-f2dd-4d74-bec6-c3136354f36e.png)`QPainterPath` allows you to use practically every paint operation Qt supports. For example, `QGraphicsPathItem` is the only standard item able to draw Bezier curves in the scene, as `QPainterPath` supports them. Refer to the documentation of `QPainterPath` for more information.

Using painter paths in this example is very efficient, because we avoided creating thousands of individual line objects on the heap. However, putting a large part of a scene in a single item may reduce the performance. When parts of the scene are separate graphics items, Qt can efficiently determine which items are not visible and skip drawing them.

# Z-order of items

Have you wondered what happens when multiple items are painted in the same area of the scene? Let's try to do this:

```cpp
QGraphicsEllipseItem *item1 = scene.addEllipse(0, 0, 100, 50);
item1->setBrush(Qt::red);
QGraphicsEllipseItem *item2 = scene.addEllipse(50, 0, 100, 50);
item2->setBrush(Qt::green);
QGraphicsEllipseItem *item3 = scene.addEllipse(0, 25, 100, 50);
item3->setBrush(Qt::blue);
QGraphicsEllipseItem *item4 = scene.addEllipse(50, 25, 100, 50);
item4->setBrush(Qt::gray);
```

By default, items are painted in the order they were added, so the last item will be displayed in front of the others:

![](img/58e6925e-58dc-4a7e-bd50-c84426f473c5.png)

However, you can change the **z-order** by calling the `setZValue()` function:

```cpp
item2->setZValue(1);
```

The second item is now displayed in front of the others:

![](img/39ea5b10-7e79-4c56-ab83-06a045cf001b.png)

Items with a higher *z* value are displayed on top of the items with lower *z* values. The default *z* value is 0\. Negative values are also possible. If items have the same *z* value, the order of insertion decides the placement, and items added later overlap those added earlier.

Ability to change the z-order of items is very important when developing 2D games. Any scene typically consists of a number of layers that must be painted in a specific order. You can set a *z* value for each item based on the layer this item belongs to.

The parent–child relationship between items also has an impact on the z-order. Children are displayed on top of their parent. Additionally, if an item is displayed in front of another item, the children of the former are also displayed in front of the children of the latter.

# Ignoring transformations

If you try to zoom in on our custom rectangles scene (for example, by calling `view.scale(4, 4)`) , you will note that everything is scaled proportionally, as you would expect. However, there are situations where you don't want some elements to be affected by scale or other transformations. Qt provides multiple ways to deal with it.

If you want lines to always have the same width, regardless of the zoom, you need to make the pen cosmetic:

```cpp
QPen pen = parent->pen();
pen.setCosmetic(true);
parent->setPen(pen);
```

Now, the rectangles will always have lines with one-pixel width, regardless of the view's scale (anti-aliasing can still blur them, though). It's also possible to have cosmetic pens with any width, but using them in Graphics View is not recommended.

Another common situation where you don't want transformation to apply is displaying text. Rotating and shearing text usually makes it unreadable, so you'd usually want to make it horizontal and untransformed. Let's try to add some text to our project and look at how we can solve this problem.

# Time for action – Adding text to a custom rectangle

Let's add a number to each of the corner circles:

```cpp
child->setPos(pos);
QGraphicsSimpleTextItem *text =
    new QGraphicsSimpleTextItem(QString::number(i), child);
text->setBrush(Qt::green);
text->setPos(-text->boundingRect().width() / 2,
             -text->boundingRect().height() / 2);
```

The `QString::number(i)` function returns the string representation of number `i`. The text item is a child of the circle item, so its position is relative to the circle's origin point (in our case, its center). As we saw earlier, the text is displayed to the top-left of the item's origin, so if we want to center the text within the circle, we need to shift it up and right by half of the item's size. Now the text is positioned and rotated along with its parent circle:

![](img/cce0ca8a-314d-4c25-8641-d16a88cdd8bd.png)

However, we don't want the text to be rotated, so we need to enable the `ItemIgnoresTransformations` flag for the text item:

```cpp
text->setFlag(QGraphicsItem::ItemIgnoresTransformations);
```

This flag makes the item ignore any transformations of its parent items or the view. However, the origin of its coordinate system is still defined by the position of `pos()` in the parent's coordinate system. So, the text item will still follow the circle, but it will no longer be scaled or rotated:

![](img/ed297d98-80a2-48cd-ae0d-b721952e66eb.png)

However, now we hit another problem: the text is no longer properly centered in the circle. It will become more apparent if you scale the view again. Why did that happen? With the `ItemIgnoresTransformations` flag, our `text->setPos(...)` statement is no longer correct. Indeed, `pos()` uses coordinates in the parent's coordinate system, but we used the result of `boundingRect()`, which uses the item's coordinate system. These two coordinate systems were the same before, but with the `ItemIgnoresTransformations` flag enabled, they are now different.

To elaborate on this problem, let's see what happens with the coordinates (we will consider only *x* coordinate, since *y* behaves the same). Let's say that our text item's width is eight pixels, so the `pos()` we set has `x = -4`. When no transformations are applied, this `pos()` results in shifting the text to the left by four pixels. If the `ItemIgnoresTransformations` flag is disabled and the view is scaled by 2, the text is shifted by eight pixels relative to the circle's center, but the size of the text itself is now 16 pixels, so it's still centered. If the `ItemIgnoresTransformations` flag is enabled, the text is still shifted to the left by eight pixels relative to the circle's center (because `pos()` operates in the parent item's coordinate system, and the circle is scaled), but the width of the item is now 8, because it ignores the scale and so it's no longer centered. When the view is rotated, the result is even more incorrect, because `setPos()` will shift the item in the direction that depends on the rotation. Since the text item itself is not rotated, we always want to shift it up and left.

This problem would go away if the item were already centered around its origin. Unfortunately, `QGraphicsSimpleTextItem` can't do this. Now, if it were  `QGraphicsRectItem`, doing this would be easy, but nothing stops us from adding a rectangle that ignores transformations and then adding text inside that rectangle! Let's do this:

```cpp
QGraphicsSimpleTextItem *text =
        new QGraphicsSimpleTextItem(QString::number(i));
QRectF textRect = text->boundingRect();
textRect.translate(-textRect.center());
QGraphicsRectItem *rectItem = new QGraphicsRectItem(textRect, child);
rectItem->setPen(QPen(Qt::green));
rectItem->setFlag(QGraphicsItem::ItemIgnoresTransformations);
text->setParentItem(rectItem);
text->setPos(textRect.topLeft());
text->setBrush(Qt::green);
```

In this code, we first create a text item, but don't set its parent. Next, we get the bounding rect of the item that will tell us how much space the text needs. Then, we shift the rect so that its center is at the origin point (0, 0). Now we can create a rect item for this rectangle, set the circle as its parent, and disable transformations for the rect item. Finally, we set the rect item as the parent of the text item and change the position of the text item to place it inside the rectangle.

The rectangle is now properly positioned at the center of the circle, and the text item always follows the rectangle, as children usually do:

![](img/1c50bcd2-4d81-4f14-9218-4cc2d2ad76d4.png)

Since we didn't originally want the rectangle, we may want to hide it. We can't use `rectItem->hide()` in this case, because that would also result in hiding its child item (the text). The solution is to disable the painting of the rectangle by calling `rectItem->setPen(Qt::NoPen)`.

An alternative solution to this problem is to translate the text item's coordinate system instead of using `setPos()`. `QGraphicsItem` doesn't have a dedicated function for translation, so we will need to use `setTransform`:

```cpp
QTransform transform;
transform.translate(-text->boundingRect().width() / 2,
                    -text->boundingRect().height() / 2);
text->setTransform(transform);
```

Contrary to what you would expect, `ItemIgnoresTransformations` doesn't cause the item to ignore its own transformations, and this code will position the text correctly without needing an additional rectangle item.

# Finding items by position

If you want to know which item is shown at a certain position, you can use the `QGraphicsScene::itemAt()` function that takes the position in the scene's coordinate system (either a `QPointF` or two `qreal` numbers) and the device transformation object (`QTransform`) that can be obtained using the `QGraphicsView::transform()` function. The function returns the topmost item at the specified position or a null pointer if no item was found. The device transformation only matters if your scene contains items that ignore transformations. If you have no such items, you can use the default-constructed `QTransform` value:

```cpp
QGraphicsItem *foundItem = scene.itemAt(scenePos, QTransform());
```

If your scene contains items that ignore transformations, it may be more convenient to use the `QGraphicsView::itemAt()` function that automatically takes the device transform into account. Note that this function expects the position to be in the viewport's coordinate system.

If you want all items that are located at some position, say in cases where multiple items are on top of each other, or if you need to search for items in some area, use the  `QGraphicsScene::items()` function. It will return a list of items defined by the specified arguments. This function has a number of overloads that allow you to specify a single point, a rectangle, a polygon, or a painter path. The `deviceTransform` argument works in the same way as for the `QGraphicsScene::itemAt()` function discussed earlier. The `mode` argument allows you to alter how the items in the area will be determined. The following table shows the different modes:

| **Mode** | **Meaning** |
| `Qt::ContainsItemBoundingRect` | The item's bounding rectangle must be completely inside the selection area. |
| `Qt::IntersectsItemBoundingRect` | Similar to `Qt::ContainsItemBoundingRect` but also returns items whose bounding rectangles intersect with the selection area. |
| `Qt::ContainsItemShape` | The item's shape must be completely inside the selection area. The shape may describe the item's boundaries more precisely than the bounding rectangle, but this operation is more computationally intensive. |
| `Qt::IntersectsItemShape` | Similar to `Qt::ContainsItemShape` but also returns items whose shapes intersect with the selection area. |

The `items()` function sorts items according to their stacking order. The `order` argument allows you to choose the order in which the results will be returned.  `Qt::DescendingOrder` (default) will place the topmost item at the beginning, and  `Qt::AscendingOrder` will result in a reversed order.

The view also provides a similar `QGraphicsView::items()` function that operates in viewport coordinates.

# Showing specific areas of the scene

As soon as the scene's bounding rectangle exceeds the viewport's size, the view will show scroll bars. Besides using them with the mouse to navigate to a specific item or point on the scene, you can also access them by code. Since the view inherits `QAbstractScrollArea`, you can use all its functions for accessing the scroll bars; `horizontalScrollBar()` and `verticalScrollBar()` return a pointer to `QScrollBar`, and thus you can query their range with `minimum()` and `maximum()`. By invoking `value()` and `setValue()`, you get and can set the current value, which results in scrolling the scene.

However, normally, you do not need to control free scrolling inside the view from your source code. The normal task would be to scroll to a specific item. In order to do that, you do not need to do any calculations yourself; the view offers a pretty simple way to do that for you—`centerOn()`. With `centerOn()`, the view ensures that the item, which you have passed as an argument, is centered on the view unless it is too close to the scene's border or even outside. Then, the view tries to move it as far as possible on the center. The `centerOn()` function does not only take a `QGraphicsItem` item as argument; you can also center on a `QPointF` pointer or as a convenience on an *x* and *y* coordinate.

If you do not care where an item is shown, you can simply call `ensureVisible()` with the item as an argument. Then, the view scrolls the scene as little as possible so that the item's center remains or becomes visible. As a second and third argument, you can define a horizontal and vertical margin, which are both the minimum space between the item's bounding rectangle and the view's border. Both values have 50 pixels as their default value. Besides a `QGraphicsItem` item, you can also ensure the visibility of a `QRectF` element (of course, there is also the convenience function taking four `qreal` elements).

If you need to ensure the entire visibility of an item, use `ensureVisible(item->boundingRect())` (since `ensureVisible(item)` only takes the item's center into account).

`centerOn()` and `ensureVisible()` only scroll the scene but do not change its transformation state. If you absolutely want to ensure the visibility of an item or a rectangle that exceeds the size of the view, you have to transform the scene as well. With this task, again the view will help you. By calling `fitInView()` with `QGraphicsItem` or a `QRectF` element as an argument, the view will scroll and scale the scene so that it fits in the viewport size.

As a second argument, you can control how the scaling is done. You have the following options:

| **Value** | **Description** |
| `Qt::IgnoreAspectRatio` | The scaling is done absolutely freely regardless of the item's or rectangle's aspect ratio. |
| `Qt::KeepAspectRatio` | The item's or rectangle's aspect ratio is taken into account while trying to expand as far as possible while respecting the viewport's size. |
| `Qt::KeepAspectRatioByExpanding` | The item's or rectangle's aspect ratio is taken into account, but the view tries to fill the whole viewport's size with the smallest overlap. |

The `fitInView()` function does not only scale larger items down to fit the viewport, it also enlarges items to fill the whole viewport. The following diagram illustrates the different scaling options for an item that is enlarged (the circle on the left is the original item, and the black rectangle is the viewport):

![](img/a698c68a-d4b2-49a1-acaa-0a27d1936f2c.png)

# Saving a scene to an image file

We've only displayed our scene in the view so far, but it is also possible to render it to an image, a printer, or any other object Qt can use for painting. Let's save our scene to a PNG file:

```cpp
QRect rect = scene.sceneRect().toAlignedRect();
QImage image(rect.size(), QImage::Format_ARGB32);
image.fill(Qt::transparent);
QPainter painter(&image);
scene.render(&painter);
image.save("scene.png");
```

# What just happened?

First, you determined the rectangle of the scene with `sceneRect()`. Since this returns a `QRectF` parameter and `QImage` can only handle `QRect`, you transformed it on the fly by calling `toAlignedRect()`. The difference between the `toRect()` function and `toAlignedRect()` is that the former rounds to the nearest integer, which may result in a smaller rectangle, whereas the latter expands to the smallest possible rectangle containing the original `QRectF` parameter.

Then, you created a `QImage` file with the size of the aligned scene's rectangle. As the image is created with uninitialized data, you need to call `fill()` with `Qt::transparent` to receive a transparent background. You can assign any color you like as an argument both as a value of `Qt::GlobalColor` enumeration and an ordinary `QColor` object; `QColor(0, 0, 255)` will result in a blue background. Next, you create a `QPainter` object that points to the image. This painter object is then used in the scene's `render()` function to draw the scene. After that, all you have to do is use the `save()` function to save the image to a place of your choice. The format of the output file is determined by its extension. Qt supports a variety of formats, and Qt plugins can add support for new formats. Since we haven't specified a path, the image will be saved in the application's working directory (which is usually the build directory, unless you changed it using the Projects pane of Qt Creator). You can also specify an absolute path, such as `/path/to/image.png`.

Of course, you'll need to construct a path that's valid on the current system instead of hard-coding it in the sources. For example, you can use the `QFileDialog::getSaveFileName()` function to ask the user for a path.

# Have a go hero – Rendering only specific parts of a scene

This example draws the whole scene. Of course, you can also render only specific parts of the scene using the other arguments of `render()`. We will not go into this here, but you may want to try it as an exercise.

# Custom items

As we already saw, Graphics View provides a lot of useful functionality that covers most typical use cases. However, the real power of Qt is its extensibility, and Graphics View allows us to create custom subclasses of `QGraphicsItem` to implement items that are tailored for your application. You may want to implement a custom item class when you need to do the following:

*   Paint something that is not possible or difficult to do with standard item classes
*   Implement some logic related to the item, for example, add your own methods
*   Handle events in individual items

In our next small project, we will create an item that can draw a graph of the sine function `sin(x)` and implement some event handling.

# Time for action – Creating a sine graph project

Use Qt Creator to create a new Qt Widgets project and name it `sine_graph`. On the Class Information page of the wizard, select `QWidget` as the base class and input `View` as the class name. Uncheck the Generate form checkbox and finish the wizard.

We want the `View` class to be the graphics view, so you need to change the base class to `QGraphicsView` (the wizard doesn't suggest such an option). For this, edit the class declaration to look like `class View : public QGraphicsView ...` and the constructor implementation to look like `View::View(QWidget *parent) : QGraphicsView(parent) ...`.

Next, edit the `View` constructor to enable anti-aliasing and set a new graphics scene for our view:

```cpp
setRenderHint(QPainter::Antialiasing);
setScene(new QGraphicsScene);
```

The view doesn't delete the associated scene on destruction (because you may have multiple views for the same scene), so you should delete the scene manually in the destructor:

```cpp
delete scene();
```

You can try to run the application and check that it displays an empty view.

# Time for action – Creating a graphics item class

Ask Qt Creator to add a new C++ class to the project. Input `SineItem` as the class name, leave <Custom> in the Base class drop-down list, and input `QGraphicsItem` in the field below it. Finish the wizard and open the created `sineitem.h` file.

Set the text cursor inside `QGraphicsItem` in the class declaration and press *Alt* + *Enter*. At first; Qt Creator will suggest that you Add `#include <QGraphicsItem>`. Confirm that and press *Alt* + *Enter* on `QGraphicsItem` again. Now, Qt Creator should suggest that you select Insert Virtual Functions of Base Classes. When you select this option, a special dialog will appear:

![](img/22622853-aad8-4b25-99fe-2631b66ffda4.png)

The function list contains all virtual functions of the base class. The pure virtual functions (which must be implemented if you want to create objects of the class) are enabled by default. Check that everything is set as in the preceding screenshot, and then click on OK. This convenient operation adds declaration and implementation of the selected virtual functions to the source files of our class. You can write them manually instead, if you want.

Let's edit `sineitem.cpp` to implement the two pure virtual functions. First of all, a couple of constants at the top of the file:

```cpp
static const float DX = 1;
static const float MAX_X = 50;
```

In our graph, *x* will vary from 0 to `MAX_X`, and `DX` will be the difference between the two consequent points of the graph. As you may know, `sin(x)` can have values from -1 to 1\. This information is enough to implement the `boundingRect()` function:

```cpp
QRectF SineItem::boundingRect() const
{
    return QRectF(0, -1, MAX_X, 2);
}
```

This function simply returns the same rectangle every time. In this rectangle, *x* changes from 0 to `MAX_X`, and *y* changes from -1 to 1\. This returned rectangle is a promise to the scene that the item will only paint in this area. The scene relies on the correctness of that information, so you should strictly obey that promise. Otherwise, the scene will become cluttered up with relics of your drawing!

Now, implement the `paint()` function, as follows:

```cpp
void SineItem::paint(QPainter *painter, 
    const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    QPen pen;
    pen.setCosmetic(true);
    painter->setPen(pen);
    const int steps = qRound(MAX_X / DX);
    QPointF previousPoint(0, sin(0));
    for(int i = 1; i < steps; ++i) {
        const float x = DX * i;
        QPointF point(x, sin(x));
        painter->drawLine(previousPoint, point);
        previousPoint = point;
    }
    Q_UNUSED(option)
    Q_UNUSED(widget)
}
```

Add `#include <QtMath>` to the top section of the file to make math functions available.

# What just happened?

When the view needs to display the scene, it calls the `paint()` function of each visible item and provides three arguments: a `QPainter` pointer that should be used for painting, a `QStyleOptionGraphicsItem` pointer that contains painting-related parameters for this item, and an optional `QWidget` pointer that may point to the currently painted widget. In the implementation of the function, we start with setting a cosmetic pen in the `painter` so that the line width of our graph is always 1\. Next, we calculate the number of points in the graph and save it to the `steps` variable. Then, we create a variable to store the previous point of the graph and initialize it with the position of the first point of the graph (corresponding to `x = 0`). Next, we iterate through points, calculate *x* and *y* for each point, and then use the `painter` object to draw a line from the previous point to the current point. After this, we update the value of the `previousPoint` variable. We use the `Q_UNUSED()` macro to suppress compiler warnings about unused arguments and to indicate that we, intentionally, didn't use them.

Edit the constructor of our `View` class to create an instance of our new item:

```cpp
SineItem *item = new SineItem();
scene()->addItem(item);
```

The application should display the sine graph now, but it is very small:

![](img/23ef1e22-8726-480b-baaa-2cb0b21a05f3.png)

We should add a way for users to scale our view using the mouse wheel. However, before we get to this, you need to learn a little more about event handling.

# Events

Any GUI application needs to react to the input events. We are already familiar with the signals and slots mechanism in `QObject`-based classes. However, `QObject` is not exactly a lightweight class. Signals and slots are powerful and convenient for connecting parts of the application, but invoking a signal for processing each keyboard press or mouse move will be too inefficient. To process such events, Qt has a special system that uses the `QEvent` class.

The dispatcher of the events is the **event loop**. Almost any Qt application uses the main event loop that is started by calling `QCoreApplication::exec` at the end of the `main()` function. While the application is running, the control flow is either in your code (that is, in the implementation of any function in the project) or in the event loop. When the operating system or a component of the application asks the event loop to process an event, it determines the receiver and calls a virtual function that corresponds to the event type. A `QEvent` object containing information about the event is passed to that function. The virtual function has a choice to **accept** or **ignore** the event. If the event was not accepted, the event is **propagated** to the parent object in the hierarchy (for example, from a widget to its parent widget, and from a graphics item to the parent item). You can subclass a Qt class and reimplement a virtual function to add custom events processing.

The following table shows the most useful events:

| **Event types** | **Description** |
| `QEvent::KeyPress`, `QEvent::KeyRelease` | A keyboard button was pressed or released. |
| `QEvent::MouseButtonPress`, `QEvent::MouseButtonRelease`, `QEvent::MouseButtonDblClick` | The mouse buttons were pressed or released. |
| `QEvent::Wheel` | The mouse wheel was rolled. |
| `QEvent::Enter` | The mouse cursor entered the object's boundaries. |
| `QEvent::MouseMove` | The mouse cursor was moved. |
| `QEvent::Leave` | The mouse cursor left the object's boundaries. |
| `QEvent::Resize` | The widget was resized (for example, because the user resized the window or the layout changed). |
| `QEvent::Close` | The user attempted to close the widget's window. |
| `QEvent::ContextMenu` | The user requested a context menu (the exact action depends on the operating system's way to open the context menu). |
| `QEvent::Paint` | The widget needs to be repainted. |
| `QEvent::DragEnter`, `QEvent::DragLeave`, `QEvent::DragMove`, `QEvent::Drop` | The user performs a drag and drop action. |
| `QEvent::TouchBegin`, `QEvent::TouchUpdate`, `QEvent::TouchEnd`, `QEvent::TouchCancel` | A touchscreen or a trackpad reported an event. |

Each event type has a corresponding class that inherits `QEvent` (for example, `QMouseEvent`). Many event types have the dedicated virtual function, for example, `QWidget::mousePressEvent` and `QGraphicsItem::mousePressEvent`. More exotic events must be processed by re-implementing the `QWidget::event` (or `QGraphicsItem::sceneEvent`) function that receives all events, and using `event->type()` to check the event type.

Events dispatched in the graphics scene have special types (for example, `QEvent::GraphicsSceneMousePress`) and special classes (for example, `QGraphicsSceneMouseEvent`) because they have an extended set of information about the event. In particular, mouse events contain information about the coordinates in the item's and the scene's coordinate systems.

# Time for action – Implementing the ability to scale the scene

Let's allow the user to scale the scene using the mouse wheel on the view. Switch to the `view.h` file and add a declaration and an implementation of the `wheelEvent()` virtual function using the same method we just used in the `SineItem` class. Write the following code in the `view.cpp` file:

```cpp
void View::wheelEvent(QWheelEvent *event)
{
    QGraphicsView::wheelEvent(event);
    if (event->isAccepted()) {
        return;
    }
    const qreal factor = 1.1;
    if (event->angleDelta().y() > 0) {
        scale(factor, factor);
    } else {
        scale(1 / factor, 1 / factor);
    }
    event->accept();
}
```

If you run the application now, you can scale the sine graph using the mouse wheel.

# What just happened?

When an event occurs, Qt calls the corresponding virtual function in the widget in which the event occurred. In our case, whenever the user uses the mouse wheel on our view, the `wheelEvent()` virtual function will be called, and the `event` argument will hold information about the event.

In our implementation, we start with calling the base class's implementation. It is very important to do this whenever you reimplement a virtual function, unless you want the default behavior to be completely disabled. In our case, `QGraphicsView::wheelEvent()` will pass the event to the scene, and if we forget to call this function, neither the scene nor any of its items will receive any wheel events, which can be very much unwanted in some cases.

After the default implementation is complete, we use the `isAccepted()` function to check whether an event was accepted by the scene or any items. The event will be rejected by default, but if we later add some item that can process wheel events (for example, a text document with its own scrollbar), it will receive and accept the event. In that case, we don't want to perform any other action based on this event, as it's usually desirable that any event is only processed (and accepted) in one location.

In some cases, you may want your custom implementation to take priority over the default one. In that case, move the call to the default implementation to the end of the function body. When you want to prevent a particular event from being dispatched to the scene, use an early `return` to prevent the default implementation from executing.

The `factor` parameter for the zooming can be freely defined. You can also create a getter and setter method for it. For us, 1.1 will do the work. With `event->angleDelta()`, you get the distance of the mouse's wheel rotation as a `QPoint` pointer. Since we only care about vertical scrolling, just the *y* axis is relevant for us. In our example, we also do not care about how far the wheel was turned because, normally, every step is delivered separately to `wheelEvent()`. However, if you should need it, it's in eighths of a degree, and since most mouses work in general steps of 15 degrees, the value should be 120 or -120, depending on whether you move the wheel forward or backward. On a forward wheel move, if `y()` is greater than zero, we zoom in using the already familiar `scale()` function. Otherwise, if the wheel was moved backward, we zoom out. Finally, we accept the event, indicating that the user's input was understood, and there is no need to propagate the event to parent widgets (although the view currently doesn't have a parent). That's all there is to it.

When you try this example, you will note that, while zooming, the view zooms in and out on the center of the view, which is the default behavior for the view. You can change this behavior with `setTransformationAnchor()`. `QGraphicsView::AnchorViewCenter` is, as described, the default behavior. With `QGraphicsView::NoAnchor`, the zoom center is in the top-left corner of the view, and the value you probably want to use is `QGraphicsView::AnchorUnderMouse`. With that option, the point under the mouse builds the center of the zooming and thus stays at the same position inside the view.

# Time for action – Taking the zoom level into account

Our graph currently contains points with integer *x* values because we set `DX = 1`. This is exactly what we want for the default level of zoom, but once the view is zoomed in, it becomes apparent that the graph's line is not smooth. We need to change `DX` based on the current zoom level. We can do this by adding the following code to the beginning of the `paint()` function():

```cpp
const qreal detail = QStyleOptionGraphicsItem::levelOfDetailFromTransform(
    painter->worldTransform());
const qreal dx = 1 / detail;
```

Delete the `DX` constant and replace `DX` with `dx` in the rest of the code. Now, when you scale the view, the graph's line keeps being smooth because the number of points increases dynamically. The `levelOfDetailFromTransform` helper function examines the value of the painter's transformation (which is a combination of all transformations applied to the item) and returns the **level of detail**. If the item is zoomed in 2:1, the level of detail is 2, and if the item is zoomed out 1:2, the level of detail is 0.5.

# Time for action – Reacting to an item's selection state

Standard items, when selected, change appearance (for example, the outline usually becomes dashed). When we're creating a custom item, we need to implement this feature manually. Let's make our item selectable in the `View` constructor:

```cpp
SineItem *item = new SineItem();
item->setFlag(QGraphicsItem::ItemIsSelectable);
```

Now, let's make the graph line green when the item is selected:

```cpp
if (option->state & QStyle::State_Selected) {
    pen.setColor(Qt::green);
}
painter->setPen(pen);
```

# What just happened?

The `state` variable is a bitmask holding the possible states of the item. You can check its value against the values of the `QStyle::StateFlag` parameter using bitwise operators. In the preceding case, the `state` variable is checked against the `State_Selected` parameter. If this flag is set, we use green color for the pen.

The type of state is `QFlags<StateFlag>`. So, instead of using the bitwise operator to test whether a flag is set, you can use the convenient function `testFlag()`.

Used with the preceding example, it would be as follows:

```cpp
if (option->state.testFlag(QStyle::State_Selected)) {  
```

The most important states you can use with items are described in the following table:

| **State** | **Description** |
| `State_Enabled` | Indicates that the item is enabled. If the item is disabled, you may want to draw it as grayed out. |
| `State_HasFocus` | Indicates that the item has the input focus. To receive this state, the item needs to have the `ItemIsFocusable` flag set. |
| `State_MouseOver` | Indicates that the cursor is currently hovering over the item. To receive this state, the item needs to have the `acceptHoverEvents` variable set to `true`. |
| `State_Selected` | Indicates that the item is selected. To receive this state, the item needs to have the `ItemIsSelectable` flag set. The normal behavior would be to draw a dashed line around the item as a selection marker. |

Besides the state, `QStyleOptionGraphicsItem` offers much more information about the currently used style, such as the palette and the font used, accessible through the `QStyleOptionGraphicsItem::palette` and `QStyleOptionGraphicsItem::fontMetrics` parameters, respectively. If you aim for style-aware items, take a deeper look at this class in the documentation.

# Time for action – Event handling in a custom item

Items, like widgets, can receive events in virtual functions. If you click on a scene (to be precise, you click on a view that propagates the event to the scene), the scene receives the mouse press event, and it then becomes the scene's responsibility to determine which item was meant by the click.

Let's override the `SineItem::mousePressEvent` function that is called when the user presses a mouse button inside the item:

```cpp
void SineItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() & Qt::LeftButton) {
        float x = event->pos().x();
        QPointF point(x, sin(x));
        static const float r = 0.3;
        QGraphicsEllipseItem *ellipse =
                new QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r, this);
        ellipse->setPen(Qt::NoPen);
        ellipse->setBrush(QBrush(Qt::red));
        ellipse->setPos(point);
        event->accept();
    } else {
        event->ignore();
    }
}
```

When a mouse press event occurs, this function is called and the passed `event` object contains information about the event. In our case, we check whether the left mouse button was pressed and use the `event->pos()` function that returns coordinates of the clicked point *in the item's coordinate system*. In this example, we ignored the *y* coordinate and used the *x* coordinate to find the corresponding point on our graph. Then, we simply created a child circle item that shows that point. We `accept` the event if we did understand the action performed and `ignore` it if we don't know what it means so that it can be passed to another item. You can run the application and click on the graph to see these circles. Note that when you click outside of the graph's bounding rect, the scene doesn't dispatch the event to our item, and its `mousePressEvent()` function is not called.

The `event` object also contains the `button()` function that returns the button that was pressed, and the `scenePos()` function that returns the clicked point in the scene's coordinate system. The scene's responsibility for delivering events does not only apply to mouse events, but also to key events and all other sorts of events.

# Time for action – Implementing the ability to create and delete elements with mouse

Let's allow the users to create new instances of our sine item when they click on the view with the left mouse button and delete the items if they use the right mouse button. Reimplement the `View::mousePressEvent` virtual function, as follows:

```cpp
void View::mousePressEvent(QMouseEvent *event)
{
    QGraphicsView::mousePressEvent(event);
    if (event->isAccepted()) {
        return;
    }
    switch (event->button()) {
        case Qt::LeftButton: {
            SineItem *item = new SineItem();
            item->setPos(mapToScene(event->pos()));
            scene()->addItem(item);
            event->accept();
            break;
        }
        case Qt::RightButton: {
            QGraphicsItem *item = itemAt(event->pos());
            if (item) {
                delete item;
            }
            event->accept();
            break;
        }
        default:
            break;
    }
}
```

Here, we first check whether the event was accepted by the scene or any of its items. If not, we determine which button was pressed. For the left button, we create a new item and place it in the corresponding point of the scene. For the right button, we search for an item at that position and delete it. In both cases, we accept the event. When you run the application, you will note that if the user clicks on an existing item, a new circle will be added, and if the user clicks outside of any items, a new sine item will be added. That's because we properly set and read the `accepted` property of the event.

You may note that the scene jumps within the view when we add a new item. This is caused by changes of the scene rect. To prevent this, you can set a constant rect using `setSceneRect()` or change the alignment using `setAlignment(Qt::AlignTop | Qt::AlignLeft)` in the view's constructor.

# Time for action – Changing the item's size

Our custom graphics item always displays the graph for *x* values between 0 and 50\. It would be neat to make this a configurable setting. Declare a private `float m_maxX` field in the `SineItem` class, remove the `MAX_X` constant, and replace its uses with `m_maxX` in the rest of the code. As always, you must set the initial value of the field in the constructor, or bad things can happen. Finally, implement a getter and a setter for it, as shown:

```cpp
float SineItem::maxX()
{
    return m_maxX;
}

void SineItem::setMaxX(float value)
{
    if (m_maxX == value) {
        return;
    }
    prepareGeometryChange();
    m_maxX = value;
}
```

The only non-trivial part here is the `prepareGeometryChange()` call. This method is inherited from `QGraphicsItem` and notifies the scene that our `boundingRect()` function will return a different value on the next update. The scene caches bounding rectangles of the items, so if you don't call `prepareGeometryChange()`, the change of the bounding rectangle may not take effect. This action also schedules an update for our item.

When the bounding rect does not change but the actual content of the item changes, you need to call `update()` on the item to notify the scene that it should repaint the item.

# Have a go hero – Extending the item's functionality

The abilities of `SineItem` are still pretty limited. As an exercise, you can try to add an option to change the minimum *x* value of the graph or set a different pen. You can even allow the user to specify an arbitrary function pointer to replace the `sin()` function. However, keep in mind that the bounding rect of the item depends on the value range of the function, so you need to update the item's geometry accurately.

# Widgets inside Graphics View

In order to show a neat feature of Graphics View, take a look at the following code snippet, which adds a widget to the scene:

```cpp
QSpinBox *box = new QSpinBox;
QGraphicsProxyWidget *proxyItem = new QGraphicsProxyWidget;
proxyItem->setWidget(box);
scene()->addItem(proxyItem);
proxyItem->setScale(2);
proxyItem->setRotation(45); 
```

First, we create a `QSpinBox` and a `QGraphicsProxyWidget` element, which act as containers for widgets and indirectly inherit `QGraphicsItem.` Then, we add the spin box to the proxy widget by calling `addWidget()`. When `QGraphicsProxyWidget` gets deleted, it calls `delete` on all assigned widgets, so we do not have to worry about that ourselves. The widget you add should be parentless and must not be shown elsewhere. After setting the widget to the proxy, you can treat the proxy widget like any other item. Next, we add it to the scene and apply a transformation for demonstration. As a result, we get this:

![](img/2f93b81a-9072-445b-af32-a9ee80065407.png)

Be aware that, originally, Graphics View wasn't designed for holding widgets. So when you add a lot of widgets to the scene, you will quickly notice performance issues, but in most situations, it should be fast enough.

If you want to arrange some widgets in a layout, you can use `QGraphicsAnchorLayout`, `QGraphicsGridLayout`, or `QGraphicsLinearLayout`. Create all widgets, create a layout of your choice, add the widgets to that layout, and set the layout to a `QGraphicsWidget` element, which is the base class for all widgets and is, easily spoken, the `QWidget` equivalent for Graphics View by calling `setLayout()`:

```cpp
QGraphicsProxyWidget *edit = scene()->addWidget(
  new QLineEdit(tr("Some Text")));
QGraphicsProxyWidget *button = scene()->addWidget(
  new QPushButton(tr("Click me!")));
QGraphicsLinearLayout *layout = new QGraphicsLinearLayout;
layout->addItem(edit);
layout->addItem(button);
QGraphicsWidget *graphicsWidget = new QGraphicsWidget;
graphicsWidget->setLayout(layout);
scene()->addItem(graphicsWidget); 
```

The scene's `addWidget()` function is a convenience function and behaves similar to `addRect`, as shown in the following code snippet:

```cpp
QGraphicsProxyWidget *proxy = new QGraphicsProxyWidget(0);
proxy->setWidget(new QLineEdit(QObject::tr("Some Text")));
scene()->addItem(proxy); 
```

The item with the layout will look like this:

![](img/f01f4b1c-7c36-4056-82c4-79869bbcb137.png)

# Optimization

When adding many items to a scene or using items with complex `paint()` functions, the performance of your application may decrease. While default optimizations of Graphics View are suitable for most cases, you may need to tweak them to achieve better performance. Let's now take a look at some of the optimizations we can perform to speed up the scene.

# A binary space partition tree

The scene constantly keeps a record of the position of the item in its internal binary space partition tree. Thus, on every move of an item, the scene has to update the tree, an operation that can become quite time-consuming, and also memory consuming. This is especially true of scenes with a large number of animated items. On the other hand, the tree enables you to find an item (for example, with `items()` or `itemAt()`) incredibly quickly, even if you have thousands of items.

So when you do not need any positional information about the items—this also includes collision detection—you can disable the index function by calling `setItemIndexMethod(QGraphicsScene::NoIndex)`. Be aware, however, that a call to `items()` or `itemAt()` results in a loop through all items in order to do the collision detection, which can cause performance problems for scenes with many items. If you cannot relinquish the tree in total, you can still adjust the depth of the tree with `setBspTreeDepth()`, taking the depth as an argument. By default, the scene will guess a reasonable value after it takes several parameters, such as the size and the number of items, into account.

# Caching the item's paint function

If you have items with a time-consuming paint function, you can change the item's cache mode. By default, no rendering is cached. With `setCacheMode()`, you can set the mode to either `ItemCoordinateCache` or `DeviceCoordinateCache`. The former renders the item in a cache of a given `QSize` element. The size of that cache can be controlled with the second argument of `setCacheMode()`, so the quality depends on how much space you assign. The cache is then used for every subsequent paint call. The cache is even used for applying transformations. If the quality deteriorates too much, just adjust the resolution by calling `setCacheMode()` again, but with a larger `QSize` element. `DeviceCoordinateCache`, on the other hand, does not cache the item on an item base but on a device level. This is, therefore, optimal for items that do not get transformed all the time because every new transformation will cause a new caching. Moving the item, however, does not invalidate the cache. If you use this cache mode, you do not have to define a resolution with the second argument. The caching is always performed at maximum quality.

# Optimizing the view

Since we are talking about the item's `paint()` function, let's touch on something related. By default, the view ensures that the painter state is saved before calling the item's paint function and that the state gets restored afterward. This will end up saving and restoring the painter state, say 50 times, if you have a scene with 50 items. However, you can disable this behavior by calling `setOptimizationFlag(DontSavePainterState, true)` on the view. If you do this, it is now your responsibility to ensure that any `paint()` function that changes the state of the painter (including pen, brush, transformation, and many other properties) must restore the previous state at the end. If you prevent automatic saving and restoring, keep in mind that now the standard items will alter the painter state. So if you use both standard and custom items, either stay with the default behavior or set `DontSavePainterState`, but then set up the pen and brush with a default value in each item's paint function.

Another flag that can be used with `setOptimizationFlag()` is `DontAdjustForAntialiasing`. By default, the view adjusts the painting area of each item by two pixels in all directions. This is useful because when one paints anti-aliased, one easily draws outside the bounding rectangle. Enable that optimization if you do not paint anti-aliased or if you are sure that your painting will stay inside the bounding rectangle. If you enable this flag and spot painting artifacts on the view, you haven't respected the item's bounding rectangle!

As a further optimization, you can define how the view should update its viewport when the scene changes. You can set the different modes with `setViewportUpdateMode()`. By default (`QGraphicsView::MinimalViewportUpdate`), the view tries to determine only those areas that need an update and repaints only these. However, sometimes it is more time-consuming to find all the areas that need a redraw than to just paint the entire viewport. This applies if you have many small updates. Then, `QGraphicsView::FullViewportUpdate` is the better choice since it simply repaints the whole viewport. A kind of combination of the last two modes is `QGraphicsView::BoundingRectViewportUpdate`. In this mode, Qt detects all areas that need a redraw, and then it redraws a rectangle of the viewport that covers all areas affected by the change. If the optimal update mode changes over time, you can tell Qt to determine the best mode using `QGraphicsView::SmartViewportUpdate`. The view then tries to find the best update mode.

# OpenGL in the Graphics View

As a last optimization, you can take advantage of OpenGL. Instead of using the default viewport based on `QWidget`, advise Graphics View to use an OpenGL widget:

```cpp
QGraphicsView view;
view.setViewport(new QOpenGLWidget()); 
```

This usually improves the rendering performance. However, Graphics View wasn't designed for GPUs and can't use them effectively. There are ways to improve the situation, but that goes beyond the topic and scope of this chapter. You can find more information about OpenGL and Graphics View in the Boxes Qt example as well as in Rødal's article "Accelerate your Widgets with OpenGL", which can be found online at [https://doc.qt.io/archives/qq/qq26-openglcanvas.html](https://doc.qt.io/archives/qq/qq26-openglcanvas.html).

If you want to use a framework designed to be GPU accelerated, you should turn your attention to Qt Quick (we will start working with it in [Chapter 11](b81d9c47-58fa-49dd-931a-864c7be05840.xhtml), *Introduction to Qt Quick*). However, Qt Quick has its own limitations compared to Graphics View. This topic is elaborated in Nichols's article *Should you still be using QGraphicsView?*, available at [https://blog.qt.io/blog/2017/01/19/should-you-be-using-qgraphicsview/](https://blog.qt.io/blog/2017/01/19/should-you-be-using-qgraphicsview/). Alternatively, you can access the full power of OpenGL directly using its API and helpful Qt utilities. We will describe this approach in [Chapter 9](15aa8ec3-9e80-4f68-8e0e-f365e860f5c5.xhtml)*, OpenGL and Vulkan in Qt applications*.

Unfortunately, we can't say that you have to do this or that to optimize Graphics View as it highly depends on your system and view/scene. What we can tell you, however, is how to proceed. Once you have finished your game based on Graphics View, measure the performance of your game using a profiler. Make an optimization you think may pay or simply guess, and then profile your game again. If the results are better, keep the change, otherwise reject it. This sounds simple and is the only way optimization can be done. There are no hidden tricks or deeper knowledge. With time, however, your forecasting will get better.

# Pop quiz

Q1\. Which of the following classes is a widget class?

1.  `QGraphicsView`
2.  `QGraphicsScene`
3.  `QGraphicsItem`

Q2\. Which of the following actions does not change the graphics item's position on the screen?

1.  Scaling the view.
2.  Shearing this item's parent item.
3.  Translating this item.
4.  Rotating this item's child item.

Q3\. Which function is not mandatory to implement in a new class derived from `QGraphicsItem`?

1.  `boundingRect()`
2.  `shape()`
3.  `paint()`

Q4\. Which item class should be used to display a raster image in the Graphics View?

1.  `QGraphicsRectItem`
2.  `QGraphicsWidget`
3.  `QGraphicsPixmapItem`

# Summary

In this chapter, you learned how the Graphics View architecture works. We went through the building blocks of the framework (items, scene, and view). Next, you learned how their coordinate systems are related and how to use them to get the picture you want. Later on, we described the most useful and frequently needed features of Graphics View. Next, we covered creating custom items and handling input events. In order to build a bridge to the world of widgets, you also learned how to incorporate items based on `QWidget` into Graphics View. Finally, we discussed ways to optimize the scene.

Now, you really know most of the functions of the Graphics View framework. With this knowledge, you can already do a lot of cool stuff. However, for a game, it is still too static. In the next chapter, we will go through the process of creating a complete game and learn to use the Animation framework.