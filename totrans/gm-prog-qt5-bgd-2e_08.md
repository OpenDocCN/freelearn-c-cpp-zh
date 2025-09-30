# Custom Widgets

We have so far been using only ready-made widgets for the user interface, which resulted in the crude approach of using buttons for a tic-tac-toe game. In this chapter, you will learn about much of what Qt has to offer with regard to custom widgets. This will let you implement your own painting and event handling, incorporating content that is entirely customized.

The main topics covered in this chapter are as follows:

*   Working with `QPainter`
*   Creating custom widgets
*   Image handling
*   Implementing a chess game

# Raster and vector graphics

When it comes to graphics, Qt splits this domain into two separate parts. One of them is raster graphics (used by widgets and the Graphics View, for example). This part focuses on using high-level operations (such as drawing lines or filling rectangles) to manipulate colors of a grid of points that can be visualized on different devices, such as images, printers, or the display of your computer device. The other is vector graphics, which involves manipulating vertices, triangles, and textures. This is tailored for maximum speed of processing and display, using hardware acceleration provided by modern graphics cards.

Qt abstracts graphics using the concept of a surface (represented by the `QSurface` class) that it draws on. The type of the surface determines which drawing operations can be performed on the surface: surfaces that support software rendering and raster graphics have the `RasterSurface` type, and surfaces that support the OpenGL interface have the `OpenGLSurface` type. In this chapter, you will deepen your knowledge of Qt's raster painting system. We will come back to the topic of OpenGL in the next chapter.

`QSurface` objects can have other types, but they are needed less often. `RasterGLSurface` is intended for internal Qt use. `OpenVGSurface` supports OpenVG (a hardware accelerated 2D vector graphics API) and is useful on embedded devices that support OpenVG but lack OpenGL support. Qt 5.10 introduces `VulkanSurface`, which supports Vulkan graphics API.

# Raster painting

When we talk about GUI frameworks, raster painting is usually associated with drawing on widgets. However, since Qt is something more than a GUI toolkit, the scope of raster painting that it offers is much broader.

In general, Qt's drawing architecture consists of three parts. The most important part is the device the drawing takes place on, represented by the `QPaintDevice` class. Qt provides a number of paint device subclasses, such as `QWidget` or `QImage` and `QPrinter` or `QPdfWriter`. You can see that the approach for drawing on a widget and printing on a printer is quite the same. The difference is in the second component of the architecture—the paint engine (`QPaintEngine`). The engine is responsible for performing the actual paint operations on a particular paint device. Different paint engines are used to draw on images and to print on printers. This is completely hidden from you, as a developer, so you really don't need to worry about it.

For you, the most important piece is the third component—`QPainter`—which is an adapter for the whole painting framework. It contains a set of high-level operations that can be invoked on the paint device. Behind the scenes, the whole work is delegated to an appropriate paint engine. While talking about painting, we will be focusing solely on the painter object, as any painting code can be invoked on any of the target devices only by using a painter initialized on a different paint device. This effectively makes painting in Qt device agnostic, as in the following example:

```cpp
void doSomePainting(QPainter *painter) {
    painter->drawLine(QPoint(0,0), QPoint(100, 40));
} 
```

The same code can be executed on a painter working on any possible `QPaintDevice` class, be it a widget, an image, or an OpenGL context (through the use of `QOpenGLPaintDevice`). We've already seen `QPainter` in action in [Chapter 4](33efb525-a584-4f9a-afaa-fe389d4a0400.xhtml), *Custom 2D Graphics with Graphics View*, when we created a custom graphics item. Now, let's learn more about this important class.

The `QPainter` class has a rich API. The most important methods in this class can be divided into three groups:

*   Setters and getters for attributes of the painter
*   Methods, with names starting with `draw` and `fill`, that perform drawing operations on the device
*   Methods that allow manipulating the coordinate system of the painter

# Painter attributes

Let's start with the attributes. The three most important ones are the pen, brush, and font. The pen holds properties of the outline drawn by the painter, and the brush determines how it will fill shapes. We've already described pens and brushes in [Chapter 4](33efb525-a584-4f9a-afaa-fe389d4a0400.xhtml), *Custom 2D Graphics with Graphics View*, so you should already understand how to work with them.

The `font` attribute is an instance of the `QFont` class. It contains a large number of methods for controlling font parameters such as font family, style (italic or oblique), font weight, and font size (either in points or device-dependent pixels). All the parameters are self-explanatory, so we will not discuss them here in detail. It is important to note that `QFont` can use any font installed on the system. In case more control over fonts is required or a font that is not installed in the system needs to be used, you can take advantage of the `QFontDatabase` class. It provides information about the available fonts (such as whether a particular font is scalable or bitmap or what writing systems it supports) and allows adding new fonts into the registry by loading their definitions directly from files.

An important class, when it comes to fonts, is the `QFontMetrics` class. It allows calculating how much space is needed to paint particular text using a font or calculates text eliding. The most common use case is to check how much space to allocate for a particular user-visible string; consider this example:

```cpp
QFontMetrics fm = painter.fontMetrics();
QRect rect = fm.boundingRect("Game Programming using Qt"); 
```

This is especially useful when trying to determine `sizeHint` for a widget.

# Coordinate systems

The next important aspect of the painter is its coordinate system. The painter in fact has two coordinate systems. One is its own logical coordinate system that operates on real numbers, and the other is the physical coordinate system of the device the painter operates on. Each operation on the logical coordinate system is mapped to physical coordinates in the device and applied there. Let's start with explaining the logical coordinate system first, and then we'll see how this relates to physical coordinates.

The painter represents an infinite cartesian canvas, with the horizontal axis pointing right and the vertical axis pointing down by default. The system can be modified by applying affine transformations to it—translating, rotating, scaling, and shearing. This way, you can draw an analog clock face that marks each hour with a line by executing a loop that rotates the coordinate system by 30 degrees for each hour and draws a line that is vertical in the newly-obtained coordinate system. Another example is when you wish to draw a simple plot with an *x* axis going right and a *y* axis going up. To obtain the proper coordinate system, you would scale the coordinate system by −1 in the vertical direction, effectively reversing the direction of the vertical axis.

What we described here modifies the world transformation matrix for the painter represented by an instance of the `QTransform` class. You can always query the current state of the matrix by calling `transform()` on the painter, and you can set a new matrix by calling `setTransform()`. `QTransform` has methods such as `scale()`, `rotate()`, and `translate()` that modify the matrix, but `QPainter` has equivalent methods for manipulating the world matrix directly. In most cases, using these would be preferable.

Each painting operation is expressed in logical coordinates, goes through the world transformation matrix, and reaches the second stage of coordinate manipulation, which is the view matrix. The painter has the concept of `viewport()` and `window()` rectangles. The `viewport` rectangle represents the physical coordinates of an arbitrary rectangle, while the `window` rectangle expresses the same rectangle but in logical coordinates. Mapping one to another gives a transformation that needs to be applied to each drawn primitive to calculate the area of the physical device that is to be painted.

By default, the two rectangles are identical to the rectangle of the underlying device (thus, no `window`–`viewport` mapping is done). Such transformation is useful if you wish to perform painting operations using measurement units other than the pixels of the target device. For example, if you want to express coordinates using percentages of the width and height of the target device, you would set both the window width and height to `100`. Then, to draw a line starting at 20% of the width and 10% of the height and ending at 70% of the width and 30% of the height, you would tell the painter to draw the line between `(20, 10)` and `(70, 30)`. If you wanted those percentages to apply not to the whole area of an image but to its left half, you would set the viewport rectangle only to the left half of the image.

Setting the `window` and `viewport` rectangles only defines coordinate mapping; it does not prevent drawing operations from painting outside the `viewport` rectangle. If you want such behavior, you have to enable **clipping** in the painter and define the clipping region or path.

# Drawing operations

Once you have the painter properly set, you can start issuing painting operations. `QPainter` has a rich set of operations for drawing different kinds of primitives. All of these operations have the `draw` prefix in their names, followed by the name of the primitive that is to be drawn. Thus, operations such as `drawLine`, `drawRoundedRect`, and `drawText` are available with a number of overloads that usually allow us to express coordinates using different data types. These may be pure values (either integer or real), Qt's classes, such as `QPoint` and `QRect`, or their floating point equivalents—`QPointF` and `QRectF`. Each operation is performed using current painter settings (font, pen, and brush).

Refer to the documentation of the `QPainter` class for the list of all drawing operations.

Before you start drawing, you have to tell the painter which device you wish to draw on. This is done using the `begin()` and `end()` methods. The former accepts a pointer to a `QPaintDevice` instance and initializes the drawing infrastructure, and the latter marks the drawing as complete. Usually, we don't have to use these methods directly, as the constructor of `QPainter` calls `begin()` for us, and the destructor invokes `end()`.

Thus, the typical workflow is to instantiate a painter object, pass it to the device, then do the drawing by calling the `set` and `draw` methods, and finally let the painter be destroyed by going out of scope, as follows:

```cpp
{
    QPainter painter(this); // paint on the current object
    QPen pen(Qt::red);
    pen.setWidth(2);
    painter.setPen(pen);
    painter.setBrush(Qt::yellow);
    painter.drawRect(0, 0, 100, 50);
} 
```

We will cover more methods from the `draw` family in the following sections of this chapter.

# Creating a custom widget

It is time to actually get something onto the screen by painting on a widget. A widget is repainted as a result of receiving a paint event, which is handled by reimplementing the `paintEvent()` virtual method. This method accepts a pointer to the event object of the `QPaintEvent` type that contains various bits of information about the repaint request. Remember that you can only paint on the widget from within that widget's `paintEvent()` call.

# Time for action – Custom-painted widgets

Let's immediately put our new skills in to practice! Start by creating a new Qt Widgets Application in Qt Creator, choosing `QWidget` as the base class, and ensuring that the Generate Form box is unchecked. The name of our widget class will be `Widget`.

Switch to the header file for the newly created class, add a protected section to the class, and type `void paintEvent` in that section. Then, press *Ctrl* + *Space* on your keyboard and Creator will suggest the parameters for the method. You should end up with the following code:

```cpp
protected:
    void paintEvent(QPaintEvent *); 
```

Creator will leave the cursor positioned right before the semicolon. Pressing *Alt* + *Enter* will open the refactoring menu, letting you add the definition in the implementation file. The standard code for a paint event is one that instantiates a painter on the widget, as shown:

```cpp
void Widget::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
} 
```

If you run this code, the widget will remain blank. Now we can start adding the actual painting code there:

```cpp
void Widget::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    QPen pen(Qt::black);
    pen.setWidth(4);
    painter.setPen(pen);
    QRect r = rect().adjusted(10, 10, -10, -10);
    painter.drawRoundedRect(r, 20, 10);
} 
```

Build and run the code, and you'll obtain the following output:

![](img/486d6364-a374-484c-b234-088fe96a195d.png)

# What just happened?

First, we set a four pixels wide black pen for the painter. Then, we called `rect()` to retrieve the geometry rectangle of the widget. By calling `adjusted()`, we receive a new rectangle with its coordinates (in the left, top, right, and bottom order) modified by the given arguments, effectively giving us a rectangle with a 10 pixel margin on each side.

Qt usually offers two methods that allow us to work with modified data. Calling `adjusted()` returns a new object with its attributes modified, while if we had called `adjust()`, the modification would have been done in place. Pay special attention to which method you use to avoid unexpected results. It's best to always check the return value for a method—whether it returns a copy or void.

Finally, we call `drawRoundedRect()`, which paints a rectangle with its corners rounded by the number of pixels (in the *x*, *y* order) given as the second and third argument. If you look closely, you will note that the rectangle has nasty jagged rounded parts. This is caused by the effect of aliasing, where a logical line is approximated using the limited resolution of the screen; due to this, a pixel is either fully drawn or not drawn at all. As we learned in [Chapter 4](33efb525-a584-4f9a-afaa-fe389d4a0400.xhtml), *Custom 2D Graphics with Graphics View*, Qt offers a mechanism called anti-aliasing to counter this effect using intermediate pixel colors where appropriate. You can enable this mechanism by setting a proper render hint on the painter before you draw the rounded rectangle, as shown:

```cpp
void Widget::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    // ...
} 
```

Now you'll get the following output:

![](img/458ab448-fa77-4ff5-9e7e-74d82e6facfb.png)

Of course, this has a negative impact on performance, so use anti-aliasing only where the aliasing effect is noticeable.

# Time for action – Transforming the viewport

Let's extend our code so that all future operations focus only on drawing within the border boundaries after the border is drawn. Use the `window` and `viewport` transformation, as follows:

```cpp
void Widget::paintEvent(QPaintEvent *) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    QPen pen(Qt::black);
    pen.setWidth(4);
    painter.setPen(pen);
    QRect r = rect().adjusted(10, 10, -10, -10);
    painter.drawRoundedRect(r, 20, 10);
    painter.save();
    r.adjust(2, 2, -2, -2);
    painter.setViewport(r);
    r.moveTo(0, -r.height() / 2);
    painter.setWindow(r);
    drawChart(&painter, r);
    painter.restore();
} 
```

Also, create a protected method called `drawChart()`:

```cpp
void Widget::drawChart(QPainter *painter, const QRect &rect) {
    painter->setPen(Qt::red);
    painter->drawLine(0, 0, rect.width(), 0);
} 
```

Let's take a look at our output:

![](img/d2936d80-e129-4603-8666-8988912ce80e.png)

# What just happened?

The first thing we did in the newly added code is call `painter.save()`. This call stores all parameters of the painter in an internal stack. We can then modify the painter state (by changing its attributes, applying transformations, and so on) and then, if at any point we want to go back to the saved state, it is enough to call `painter.restore()` to undo all the modifications in one go.

The `save()` and `restore()` methods can be called as many times as needed. States are stored in a stack, so you can save multiple times in a row and then restore to undo each change. Just remember to always pair a call to `save()` with a similar call to `restore()`, or the internal painter state will get corrupted. Each call to `restore()` will revert the painter to the last saved state.

After the state is saved, we modify the rectangle again by adjusting for the width of the border. Then, we set the new rectangle as the viewport, informing the painter about the physical range of coordinates to operate on. Then, we move the rectangle by half its height and set that as the painter window. This effectively puts the origin of the painter at half the height of the widget. Then, the `drawChart()` method is called, whereby a red line is drawn on the *x* axis of the new coordinate system.

# Time for action – Drawing an oscillogram

Let's further extend our widget to become a simple oscillogram renderer. For that, we have to make the widget remember a set of values and draw them as a series of lines.

Let's start by adding a `QVector<quint16>` member variable that holds a list of unsigned 16-bit integer values. We will also add slots for adding values to the list and for clearing the list, as shown:

```cpp
class Widget : public QWidget
{
    // ...
public slots:
    void addPoint(unsigned yVal) {
        m_points << qMax(0u, yVal);
        update();
    }
    void clear() {
        m_points.clear();
        update();
    }
protected:
    // ...
    QVector<quint16> m_points;
}; 
```

Note that each modification of the list invokes a method called `update()`. This schedules a paint event so that our widget can be redrawn with the new values.

Drawing code is also easy; we just iterate over the list and draw symmetric blue lines based on the values from the list. Since the lines are vertical, they don't suffer from aliasing and so we can disable this render hint, as shown:

```cpp
void Widget::drawChart(QPainter *painter, const QRect &rect) {
    painter->setPen(Qt::red);
    painter->drawLine(0, 0, rect.width(), 0);
    painter->save();
    painter->setRenderHint(QPainter::Antialiasing, false);
    painter->setPen(Qt::blue);
    for(int i = 0; i < m_points.size(); ++i) {
        painter->drawLine(i, -m_points.at(i), i, m_points.at(i));
    }
    painter->restore();
} 
```

To see the result, let's fill the widget with data in the `main()` function:

```cpp
for(int i = 0; i < 450; ++i) {
    w.addPoint(qrand() % 120);
}
```

This loop takes a random number between `0` and `119` and adds it as a point to the widget. A sample result from running such code can be seen in the following screenshot:

![](img/8bbcf228-49c4-4bfe-b36a-d5c7e563224b.png)

If you scale down the window, you will note that the oscillogram extends past the boundaries of the rounded rectangle. Remember about clipping? You can use it now to constrain the drawing by adding a simple `painter.setClipRect(r)` call just before you call `drawChart()`.

So far, the custom widget was not interactive at all. Although the widget content could be manipulated from within the source code (say by adding new points to the plot), the widget was deaf to any user actions (apart from resizing the widget, which caused a repaint). In Qt, any interaction between the user and the widget is done by delivering events to the widget. Such a family of events is generally called input events and contains events such as keyboard events and different forms of pointing-device events—mouse, tablet, and touch events.

In a typical mouse event flow, a widget first receives a mouse press event, then a number of mouse move events (when the user moves the mouse around while the mouse button is kept pressed), and finally, a mouse release event. The widget can also receive an additional mouse double-click event in addition to these events. It is important to remember that by default, mouse move events are only delivered if a mouse button is pressed when the mouse is moved. To receive mouse move events when no button is pressed, a widget needs to activate a feature called **mouse tracking**.

# Time for action – Making oscillograms selectable

It's time to make our oscillogram widget interactive. We will teach it to add a couple of lines of code to it that let the user select part of the plot. Let's start with storage for the selection. We'll need two integer variables that can be accessed via read-only properties; therefore, add the following two properties to the class:

```cpp
Q_PROPERTY(int selectionStart READ selectionStart
                              NOTIFY selectionChanged)
Q_PROPERTY(int selectionEnd   READ selectionEnd
                              NOTIFY selectionChanged)
```

Next, you need to create corresponding private fields (you can initialize them both to −1), getters, and signals.

The user can change the selection by dragging the mouse cursor over the plot. When the user presses the mouse button over some place in the plot, we'll mark that place as the start of the selection. Dragging the mouse will determine the end of the selection. The scheme for naming events is similar to the paint event; therefore, we need to declare and implement the following two protected methods:

```cpp
void Widget::mousePressEvent(QMouseEvent *mouseEvent) {
    m_selectionStart = m_selectionEnd = mouseEvent->pos().x() - 12;
    emit selectionChanged();
    update();
}
void Widget::mouseMoveEvent(QMouseEvent *mouseEvent) {
    m_selectionEnd = mouseEvent->pos().x() - 12;
    emit selectionChanged();
    update();
} 
```

The structure of both event handlers is similar. We update the needed values, taking into consideration the left padding (12 pixels) of the plot, similar to what we do while drawing. Then, a signal is emitted and `update()` is called to schedule a repaint of the widget.

What remains is to introduce changes to the drawing code. We suggest that you add a `drawSelection()` method similar to `drawChart()`, but that it is called from the paint event handler immediately before `drawChart()`, as shown:

```cpp
void Widget::drawSelection(QPainter *painter, const QRect &rect) {
    if(m_selectionStart < 0) {
        return;
    }
    painter->save();
    painter->setPen(Qt::NoPen);
    painter->setBrush(palette().highlight());
    QRect selectionRect = rect;
    selectionRect.setLeft(m_selectionStart);
    selectionRect.setRight(m_selectionEnd);
    painter->drawRect(selectionRect);
    painter->restore();
} 
```

First, we check whether there is any selection to be drawn at all. Then, we save the painter state and adjust the pen and brush of the painter. The pen is set to `Qt::NoPen`, which means the painter should not draw any outline. To determine the brush, we use `palette()`; this returns an object of the `QPalette` type holding basic colors for a widget. One of the colors held in the object is the color of the highlight often used for marking selections. If you use an entry from the palette instead of manually specifying a color, you gain an advantage because when the user of the class modifies the palette, this modification is taken into account by our widget code.

You can use other colors from the palette in the widget for other things we draw in the widget. You can even define your own `QPalette` object in the constructor of the widget to provide default colors for it.

Finally, we adjust the rectangle to be drawn and issue the drawing call.

When you run this program, you will note that the selection color doesn't contrast very well with the plot itself. To overcome this, a common approach is to draw the "selected" content with a different (often inverted) color. This can easily be applied in this situation by modifying the `drawChart()` code slightly:

```cpp
for(int i = 0; i < m_points.size(); ++i) {
    if(m_selectionStart <= i && m_selectionEnd >=i) {
        painter->setPen(Qt::white);
    } else {
        painter->setPen(Qt::blue);
    }
    painter->drawLine(i, -m_points.at(i), i, m_points.at(i));
} 
```

Now you see the following output:

![](img/365e08df-895e-4841-8e76-eb33818a422d.png)

# Have a go hero – Reacting only to the left mouse button

As an exercise, you can modify the event handling code so that it only changes the selection if the mouse event was triggered by the left mouse button. To see which button triggered the mouse press event, you can use the `QMouseEvent::button()` method, which returns `Qt::LeftButton` for the left button, `Qt::RightButton` for the right, and so on.

# Touch events

Handling touch events is different. For any such event, you receive a call to the `touchEvent()` virtual method. The parameter of such a call is an object that can retrieve a list of points currently touched by the user with additional information regarding the history of user interaction (whether the touch was just initiated or the point was pressed earlier and moved) and what force is applied to the point by the user. Note that this is a low-level framework that allows you to precisely follow the history of touch interaction. If you are more interested in higher-level gesture recognition (pan, pinch, and swipe), there is a separate family of events available for it.

Handling gestures is a two-step procedure. First, you need to activate gesture recognition on your widget by calling `grabGesture()` and passing in the type of gesture you want to handle. A good place for such code is the widget constructor.

Then, your widget will start receiving gesture events. There are no dedicated handlers for gesture events but, fortunately, all events for an object flow through its `event()` method, which we can reimplement. Here's some example code that handles pan gestures:

```cpp
bool Widget::event(QEvent *e) {
  if(e->type() == QEvent::Gesture) {
    QGestureEvent *gestureEvent = static_cast<QGestureEvent*>(e);
    QGesture *pan  = gestureEvent->gesture(Qt::PanGesture);
    if(pan) {
      handlePanGesture(static_cast<QPanGesture*>(pan));
    }
  }
  return QWidget::event(e);
} 
```

First, a check for the event type is made; if it matches the expected value, the event object is cast to `QGestureEvent`. Then, the event is asked whether `Qt::PanGesture` was recognized. Finally, a `handlePanGesture` method is called. You can implement such a method to handle your pan gestures.

# Working with images

Qt has two classes for handling images. The first one is `QImage`, more tailored toward direct pixel manipulation. You can check the size of the image or check and modify the color of each pixel. You can convert the image into a different internal representation (say from 8-bit color map to full 32-bit color with a premultiplied alpha channel). This type, however, is not that fit for rendering. For that, we have a different class called `QPixmap`. The difference between the two classes is that `QImage` is always kept in the application memory, while `QPixmap` can only be a handle to a resource that may reside in the graphics card memory or on a remote *X* server. Its main advantage over `QImage` is that it can be rendered very quickly at the cost of the inability to access pixel data. You can freely convert between the two types, but bear in mind that on some platforms, this might be an expensive operation. Always consider which class serves your particular situation better. If you intend to crop the image, tint it with some color, or paint over it, `QImage` is a better choice, but if you just want to render a bunch of icons, it's best to keep them as `QPixmap` instances.

# Loading

Loading images is very easy. Both `QPixmap` and `QImage` have constructors that simply accept a path to a file containing the image. Qt accesses image data through plugins that implement reading and writing operations for different image formats. Without going into the details of plugins, it is enough to say that the default Qt installation supports reading the following image types:

| **Type** | Description |
| --- | --- |
| BMP | Windows Bitmap |
| GIF | Graphics Interchange Format |
| JPG/JPEG | Joint Photography Experts Group |
| PNG | Portable Network Graphics |
| PPM/PBM/PGM | Portable anymap |
| XBM | X Bitmap |
| XPM | X Pixmap |

As you can see, the most popular image formats are available. The list can be further extended by installing additional plugins.

You can ask Qt for a list of supported image types by calling a static method, `QImageReader::supportedImageFormats()`, which returns a list of formats that can be read by Qt. For a list of writable formats, call `QImageWriter::supportedImageFormats()`.

An image can also be loaded directly from an existing memory buffer. This can be done in two ways. The first one is to use the `loadFromData()` method (it exists in both `QPixmap` and `QImage`), which behaves the same as when loading an image from a file—you pass it a data buffer and the size of the buffer and based on that, the loader determines the image type by inspecting the header data and loads the picture into `QImage` or `QPixmap`. The second situation is when you don't have images stored in a "filetype" such as JPEG or PNG; rather, you have raw pixel data itself. In such a situation, `QImage` offers a constructor that takes a pointer to a block of data together with the size of the image and format of the data. The format is not a file format such as the ones listed earlier but a memory layout for data representing a single pixel.

The most popular format is `QImage::Format_ARGB32`, which means that each pixel is represented by 32-bits (4 bytes) of data divided equally between alpha, red, green, and blue channels—8-bits per channel. Another popular format is `QImage::Format_ARGB32_Premultiplied`, where values for the red, green, and blue channels are stored after being multiplied by the value of the alpha channel, which often results in faster rendering. You can change the internal data representation using a call to `convertToFormat()`. For example, the following code converts a true-color image to 256 colors, where color for each pixel is represented by an index in a color table:

```cpp
QImage trueColor("image.png");
QImage indexed = trueColor.convertToFormat(QImage::Format_Indexed8); 
```

The color table itself is a vector of color definitions that can be fetched using `colorTable()` and replaced using `setColorTable()`. For example, you can convert an indexed image to grayscale by adjusting its color table, as follows:

```cpp
QImage indexed = ...;
QVector<QRgb> colorTable = indexed.colorTable();
for(QRgb &item: colorTable) {
    int gray = qGray(item);
    item = qRgb(gray, gray, gray);
}
indexed.setColorTable(colorTable); 
```

However, there is a much cleaner solution to this task. You can convert any image to the `Format_Grayscale8` format:

```cpp
QImage grayImage = coloredImage.convertToFormat(QImage::Format_Grayscale8);
```

This format uses 8 bits per pixel and doesn't have a color table, so it can only store grayscale images.

# Modifying

There are two ways to modify image pixel data. The first one works only for `QImage` and involves direct manipulation of pixels using the `setPixel()` call, which takes the pixel coordinates and color to be set for that pixel. The second one works for both `QImage` and `QPixmap` and makes use of the fact that both these classes are subclasses of `QPaintDevice`. Therefore, you can open `QPainter` on such objects and use its drawing API. Here's an example of obtaining a pixmap with a blue rectangle and red circle painted over it:

```cpp
QPixmap px(256, 256);
px.fill(Qt::transparent);
QPainter painter(&px);
painter.setPen(Qt::NoPen);
painter.setBrush(Qt::blue);
QRect r = px.rect().adjusted(10, 10, -10, -10);
painter.drawRect(r);
painter.setBrush(Qt::red);
painter.drawEllipse(r); 
```

First, we create a 256 x 256 pixmap and fill it with transparent color. Then, we open a painter on it and invoke a series of calls that draws a blue rectangle and red circle.

`QImage` also offers a number of methods for transforming the image, including `scaled()`, `mirrored()`, `transformed()`, and `copy()`. Their API is intuitive, so we won't discuss it here.

# Painting

Painting images in its basic form is as simple as calling `drawImage()` or `drawPixmap()` from the `QPainter` API. There are different variants of the two methods, but, basically, all of them allow one to specify which portion of a given image or pixmap is to be drawn and where. It is worth noting that painting pixmaps is preferred to painting images, as an image has to first be converted into a pixmap before it can be drawn.

If you have a lot of pixmaps to draw, a class called `QPixmapCache` may come in handy. It provides an application-wide cache for pixmaps. Using it, you can speed up pixmap loading while introducing a cap on memory usage.

Finally, if you just want to show a pixmap as a separate widget, you can use `QLabel`. This widget is usually used for displaying text, but you can configure it to show a pixmap instead with the `setPixmap()` function. By default, the pixmap is displayed without scaling. When the label is larger than the pixmap, it's position is determined by the label's alignment that you can change with the `setAlignment()` function. You can also call `setScaledContents(true)` to stretch the pixmap to the whole size of the label.

# Painting text

Drawing text using `QPainter` deserves a separate explanation, not because it is complicated, but because Qt offers much flexibility in this regard. In general, painting text takes place by calling `QPainter::drawText()` or `QPainter::drawStaticText()`. Let's focus on the former first, which allows the drawing of generic text.

The most basic call to paint some text is a variant of this method, which takes *x* and *y* coordinates and the text to draw:

```cpp
painter.drawText(10, 20, "Drawing some text at (10, 20)"); 
```

The preceding call draws the given text at position 10 horizontally and places the baseline of the text at position 20 vertically. The text is drawn using the painter's current font and pen. The coordinates can alternatively be passed as `QPoint` instances, instead of being given *x* and *y* values separately. The problem with this method is that it allows little control over how the text is drawn. A much more flexible variant is one that lets us give a set of flags and expresses the position of the text as a rectangle instead of a point. The flags can specify the alignment of the text within the given rectangle or instruct the rendering engine about wrapping and clipping the text. You can see the result of giving a different combination of flags to the call in the following diagram:

![](img/7734f14d-1980-4743-b108-123f2fb12aa8.png)

In order to obtain each of the preceding results, run code similar to the following:

```cpp
painter.drawText(rect, Qt::AlignLeft | Qt::TextShowMnemonic, "&ABC"); 
```

You can see that unless you set the `Qt::TextDontClip` flag, the text is clipped to the given rectangle; setting `Qt::TextWordWrap` enables line wrapping, and `Qt::TextSingleLine` makes the engine ignore any newline characters encountered.

# Static text

Qt has to perform a number of calculations when laying out the text, and this has to be done each time the text is rendered. This will be a waste of time if the text and its attributes have not changed since the last time. To avoid the need to recalculate the layout, the concept of static text was introduced.

To use it, instantiate `QStaticText` and initialize it with the text you want to render along with any options you might want it to have (kept as the `QTextOption` instance). Then, store the object somewhere, and whenever you want the text to be rendered, just call `QPainter::drawStaticText()`, passing the static text object to it. If the layout of the text has not changed since the previous time the text was drawn, it will not be recalculated, resulting in improved performance. Here's an example of a custom widget that simply draws text using the static text approach:

```cpp
class TextWidget : public QWidget {
public:
    TextWidget(QWidget *parent = nullptr) : QWidget(parent) {}
    void setText(const QString &txt) {
        m_staticText.setText(txt);
        update();
    }
protected:
    void paintEvent(QPaintEvent *) {
        QPainter painter(this);
        painter.drawStaticText(0, 0, m_staticText);
    }
private:
    QStaticText m_staticText;
}; 
```

# Optimizing widget painting

As an exercise, we will modify our oscillogram widget so that it only rerenders the part of its data that is required.

# Time for action – Optimizing oscillogram drawing

The first step is to modify the paint event handling code to fetch information about the region that needs updating and pass it to the method drawing the chart. The changed parts of the code have been highlighted here:

```cpp
void Widget::paintEvent(QPaintEvent *event)
{
    QRect exposedRect = event->rect();
    ...
    drawSelection(&painter, r, exposedRect);
    drawChart(&painter, r, exposedRect);
    painter.restore();
} 
```

The next step is to modify `drawSelection()` to only draw the part of the selection that intersects with the exposed rectangle. Luckily, `QRect` offers a method to calculate the intersection for us:

```cpp
void Widget::drawSelection(QPainter *painter, const QRect &rect,
                           const QRect &exposedRect)
{
    // ...
    QRect selectionRect = rect;
    selectionRect.setLeft(m_selectionStart);
    selectionRect.setRight(m_selectionEnd);
    painter->drawRect(selectionRect.intersected(exposedRect));
    painter->restore();
} 
```

Finally, `drawChart` needs to be adjusted to omit the values outside the exposed rectangle:

```cpp
void Widget::drawChart(QPainter *painter, const QRect &rect,
                       const QRect &exposedRect)
{
    painter->setPen(Qt::red);
    painter->drawLine(exposedRect.left(), 0, exposedRect.width(), 0);
    painter->save();
    painter->setRenderHint(QPainter::Antialiasing, false);
    const int lastPoint = qMin(m_points.size(),
                               exposedRect.right() + 1);
    for(int i = exposedRect.left(); i < lastPoint; ++i) {
      if(m_selectionStart <= i && m_selectionEnd >=i) {
        painter->setPen(Qt::white);
      } else
      painter->setPen(Qt::blue);
      painter->drawLine(i, -m_points.at(i), i, m_points.at(i));
    }
    painter->restore();
    Q_UNUSED(rect)
} 
```

# What just happened?

By implementing these changes, we have effectively reduced the painted area to the rectangle received with the event. In this particular situation, we will not save much time as drawing the plot is not that time-consuming; in many situations, however, you will be able to save a lot of time using this approach. For example, if we were to plot a very detailed aerial map of a game world, it would be very expensive to replot the whole map if only a small part of it were modified. We can easily reduce the number of calculations and drawing calls by taking advantage of the information about the exposed area.

Making use of the exposed rectangle is already a good step toward efficiency, but we can go a step further. The current approach requires that we redraw each and every line of the plot within the exposed rectangle, which still takes some time. Instead, we can paint those lines only once into a pixmap, and then whenever the widget needs repainting, tell Qt to render part of the pixmap to the widget.

# Have a go hero – Caching the oscillogram in a pixmap

Now, it should be very easy for you to implement this approach for our example widget. The main difference is that each change to the plot contents should not result in a call to `update()` but in a call that will rerender the pixmap and then call `update()`. The `paintEvent` method then becomes simply this:

```cpp
void Widget::paintEvent(QPaintEvent *event)
{
    QRect exposedRect = event->rect();
    QPainter painter(this);
    painter.drawPixmap(exposedRect, m_pixmap, exposedRect);
} 
```

You'll also need to rerender the pixmap when the widget is resized. This can be done from within the `resizeEvent()` virtual function.

While it is useful to master the available approaches to optimization, it's always important to check whether they actually make your application faster. There are often cases where the straightforward approach is more optimal than a clever optimization. In the preceding example, resizing the widget (and subsequently resizing the pixmap) can trigger a potentially expensive memory allocation. Use this optimization only if direct painting on the widget is even more expensive.

# Implementing a chess game

At this point, you are ready to employ your newly gained skills in rendering graphics with Qt to create a game that uses widgets with custom graphics. The hero of today will be chess and other chess-like games.

# Time for action – Developing the game architecture

Create a new Qt Widgets Application project. After the project infrastructure is ready, choose New File or Project from the File menu and choose to create a C++ Class. Call the new class `ChessBoard` and set `QObject` as its base class. Repeat the process to create a `ChessAlgorithm` class derived from `QObject` and another one called `ChessView`, but choose `QWidget` as the base class this time. You should end up with a file named `main.cpp` and four classes:

*   `MainWindow` will be our main window class that contains a `ChessView`
*   `ChessView` will be the widget that displays our chess board
*   `ChessAlgorithm` will contain the game logic
*   `ChessBoard` will hold the state of the chess board and provide it to `ChessView` and `ChessAlgorithm`

Now, navigate to the header file for `ChessAlgorithm` and add the following methods to the class:

```cpp
public:
    ChessBoard* board() const;
public slots:
    virtual void newGame();
signals:
    void boardChanged(ChessBoard*);
protected:
    virtual void setupBoard();
    void setBoard(ChessBoard *board); 
```

Also, add a private `m_board` field of the `ChessBoard*` type. Remember to either include `chessboard.h` or forward-declare the `ChessBoard` class. Implement `board()` as a simple getter method for `m_board`. The `setBoard()` method will be a protected setter for `m_board`:

```cpp
void ChessAlgorithm::setBoard(ChessBoard *board)
{
    if(board == m_board) {
        return;
    }
    delete m_board;
    m_board = board;
    emit boardChanged(m_board);
} 
```

Next, let's provide a base implementation for `setupBoard()` to create a default chess board with eight ranks and eight columns:

```cpp
void ChessAlgorithm::setupBoard()
{
    setBoard(new ChessBoard(8, 8, this));
} 
```

The natural place to prepare the board is in a function executed when a new game is started:

```cpp
void ChessAlgorithm::newGame()
{
    setupBoard();
} 
```

The last addition to this class for now is to extend the provided constructor to initialize `m_board` to a null pointer.

In the last method shown, we instantiated a `ChessBoard` object, so let's focus on that class now. First, extend the constructor to accept two additional integer parameters besides the regular parent argument. Store their values in private `m_ranks` and `m_columns` fields (remember to declare the fields themselves in the class header file).

In the header file, just under the `Q_OBJECT` macro, add the following two lines as property definitions:

```cpp
  Q_PROPERTY(int ranks READ ranks NOTIFY ranksChanged)
  Q_PROPERTY(int columns READ columns NOTIFY columnsChanged) 
```

Declare signals and implement getter methods to cooperate with those definitions. Also, add two protected methods:

```cpp
protected:
    void setRanks(int newRanks);
    void setColumns(int newColumns); 
```

These will be setters for the rank and column properties, but we don't want to expose them to the outside world, so we will give them `protected` access scope.

Put the following code into the `setRanks()` method body:

```cpp
void ChessBoard::setRanks(int newRanks)
{
    if(ranks() == newRanks) {
        return;
    }
    m_ranks = newRanks;
    emit ranksChanged(m_ranks);
} 
```

Next, in a similar way, you can implement `setColumns()`.

The last class we will deal with now is our custom widget, `ChessView`. For now, we will provide only a rudimentary implementation for one method, but we will expand it later as our implementation grows. Add a public `setBoard(ChessBoard *)` method with the following body:

```cpp
void ChessView::setBoard(ChessBoard *board)
{
    if(m_board == board) {
        return;
    }
    if(m_board) {
        // disconnect all signal-slot connections between m_board and this
        m_board->disconnect(this);
    }
    m_board = board;
    // connect signals (to be done later)
    updateGeometry();
} 
```

Now, let's declare the `m_board` member. As we are not the owners of the board object (the algorithm class is responsible for managing it), we will use the `QPointer` class, which tracks the lifetime of `QObject` and sets itself to null once the object is destroyed:

```cpp
private:
    QPointer<ChessBoard> m_board; 
```

`QPointer` initializes its value to null, so we don't have to do it ourselves in the constructor. For completeness, let's provide a getter method for the board:

```cpp
ChessBoard *ChessView::board() const {
    return m_board;
} 
```

# What just happened?

In the last exercise, we defined the base architecture for our solution. We can see that there are three classes involved: `ChessView` acting as the user interface, `ChessAlgorithm` for driving the actual game, and `ChessBoard` as a data structure shared between the view and the engine. The algorithm will be responsible for setting up the board (through `setupBoard()`), making moves, checking win conditions, and so on. The view will be rendering the current state of the board and will signal user interaction to the underlying logic.

Most of the code is self-explanatory. You can see in the `ChessView::setBoard()` method that we are disconnecting all signals from an old board object, attaching the new one (we will come back to connecting the signals later when we have already defined them), and finally telling the widget to update its size and redraw itself with the new board.

# Time for action – Implementing the game board class

Now we will focus on our data structure. Add a new private member to `ChessBoard`, a vector of characters that will contain information about pieces on the board:

```cpp
QVector<char> m_boardData; 
```

Consider the following table that shows the piece type and the letters used for it:

| Piece type |  | White | Black |
| --- | --- | --- | --- |
| ![](img/a5eb6d64-b6b8-4344-beb4-6a0d23d08e5c.png)![](img/2dda234d-fdfe-4be0-9b36-d9ee1dd70486.png) | King | K | k |
| ![](img/a6bb3af1-bb15-410d-8de6-301393e517bf.jpg) ![](img/de3320da-fb85-4d7e-9443-1cb66365436b.png) | Queen | Q | q |
| ![](img/5e99e21e-bfda-447b-b556-9adcd9e352aa.png) ![](img/1c63c299-5f79-486c-8989-b93c9031aee0.jpg) | Rook | R | r |
| ![](img/c2a5c597-ac35-4ae2-9ecb-2c308b83fde9.jpg) ![](img/bb149e00-8633-4c11-82e6-faac23536bc4.jpg) | Bishop | B | b |
| ![](img/bb697436-b5be-4312-951d-d6eaa7fcf179.jpg) ![](img/a32f3941-3593-475f-bf94-e47f5a3837ac.jpg) | Knight | N | n |
| ![](img/4f17458d-cec3-48d8-a1ce-156d1d8ff7c4.jpg) ![](img/05af7208-dcbb-4c6b-9a0d-053293de1ada.jpg) | Pawn | P | P |

You can see that white pieces use uppercase letters and black pieces use lowercase variants of the same letters. In addition to that, we will use a space character (0x20 ASCII value) to denote that a field is empty. We will add a protected method for setting up an empty board based on the number of ranks and columns on the board and a `boardReset()` signal to inform that the position on the board has changed:

```cpp
void ChessBoard::initBoard()
{
    m_boardData.fill(' ', ranks() * columns());
    emit boardReset();
} 
```

We can update our methods for setting rank and column counts to make use of that method:

```cpp
void ChessBoard::setRanks(int newRanks)
{
    if(ranks() == newRanks) {
        return;
    }
    m_ranks = newRanks;
    initBoard();
    emit ranksChanged(m_ranks);
}

void ChessBoard::setColumns(int newColumns)
{
    if(columns() == newColumns) {
        return;
    }
    m_columns = newColumns;
    initBoard();
    emit columnsChanged(m_columns);
} 
```

The `initBoard()` method should also be called from within the constructor, so place the call there as well.

Next, we need a method to read which piece is positioned in a particular field of the board:

```cpp
char ChessBoard::data(int column, int rank) const
{
    return m_boardData.at((rank-1) * columns() + (column - 1));
} 
```

Ranks and columns have indexes starting from 1, but the data structure is indexed starting from 0; therefore, we have to subtract 1 from both the rank and column index. It is also required to have a method to modify the data for the board. Implement the following public method:

```cpp
void ChessBoard::setData(int column, int rank, char value)
{
    if(setDataInternal(column, rank, value)) {
        emit dataChanged(column, rank);
    }
} 
```

The method makes use of another one that does the actual job. However, this method should be declared with `protected` access scope. Again, we adjust for index differences:

```cpp
bool ChessBoard::setDataInternal(int column, int rank, char value)
{
    int index = (rank-1) * columns() + (column - 1);
    if(m_boardData.at(index) == value) {
        return false;
    }
    m_boardData[index] = value;
    return true;
} 
```

Since `setData()` makes use of a signal, we have to declare it as well:

```cpp
signals:
    void ranksChanged(int);
    void columnsChanged(int);
    void dataChanged(int c, int r);
    void boardReset(); 
```

The signal will be emitted every time there is a successful change to the situation on the board. We delegate the actual work to the protected method to be able to modify the board without emitting the signal.

Having defined `setData()`, we can add another method for our convenience:

```cpp
void ChessBoard::movePiece(int fromColumn, int fromRank, 
                           int toColumn, int toRank)
{
    setData(toColumn, toRank, data(fromColumn, fromRank));
    setData(fromColumn, fromRank, ' ');
} 
```

Can you guess what it does? That's right! It moves a piece from one field to another one, leaving an empty space behind.

There is still one more method worth implementing. A regular chess game contains 32 pieces, and there are variants of the game where starting positions for the pieces might be different. Setting the position of each piece through a separate call to `setData()` would be very cumbersome. Fortunately, there is a neat chess notation called the **Forsyth-Edwards Notation** (**FEN**), with which the complete state of the game can be stored as a single line of text. If you want the complete definition of the notation, you can look it up yourself. In short, we can say that the textual string lists piece placement rank by rank, starting from the last rank where each position is described by a single character interpreted as in our internal data structure (`K` for white king, `q` for black queen, and so on). Each rank description is separated by a `/` character. If there are empty fields on the board, they are not stored as spaces, but as a digit specifying the number of consecutive empty fields. Therefore, the starting position for a standard game can be written as follows:

```cpp
"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" 
```

This can be interpreted visually, as follows:

![](img/91155804-2828-48c5-ba5f-786aa686d7c9.png)

Let's write a method called `setFen()` to set up the board based on an FEN string:

```cpp
void ChessBoard::setFen(const QString &fen)
{
    int index = 0;
    int skip = 0;
    const int columnCount = columns();
    QChar ch;
    for(int rank = ranks(); rank > 0; --rank) {
        for(int column = 1; column <= columnCount; ++column) {
            if(skip > 0) {
                ch = ' ';
                skip--;
            } else {
                ch = fen.at(index++);
                if(ch.isDigit()) {
                    skip = ch.toLatin1() - '0';
                    ch = ' ';
                    skip--;
                }
            }
            setDataInternal(column, rank, ch.toLatin1());
        }
        QChar next = fen.at(index++);
        if(next != '/' && next != ' ') {
            initBoard();
            return; // fail on error
        }
    }
    emit boardReset();
} 
```

The method iterates over all fields on the board and determines whether it is currently in the middle of inserting empty fields on the board or should rather read the next character from the string. If a digit is encountered, it is converted into an integer by subtracting the ASCII value of the 0 character (that is, `'7' - '0'` = 7). After setting each rank, we require that a slash or a space be read from the string. Otherwise, we reset the board to an empty one and bail out of the method.

# What just happened?

We taught the `ChessBoard` class to store simple information about chess pieces using a one-dimensional array of characters. We also equipped it with methods that allow querying and modifying game data. We implemented a fast way of setting the current state of the game by adopting the FEN standard. The game data itself is not tied to classic chess. Although we comply with a standard notation for describing pieces, it is possible to use other letters and characters outside the well-defined set for chess pieces. This creates a versatile solution for storing information about chess-like games, such as checkers, and possibly any other custom games played on a two-dimensional board of any size with ranks and columns. The data structure we came up with is not a stupid one—it communicates with its environment by emitting signals when the state of the game is modified.

# Time for action – Understanding the ChessView class

This is a chapter about doing graphics, so it is high time we focus on displaying our chess game. Our widget currently displays nothing, and our first task will be to show a chess board with rank and column symbols and fields colored appropriately.

By default, the widget does not have any proper size defined, and we will have to fix that by implementing `sizeHint()`. However, to be able to calculate the size, we have to decide how big a single field on the board will be. Therefore, in `ChessView`, you should declare a property containing the size of the field, as shown:

```cpp
Q_PROPERTY(QSize fieldSize
           READ fieldSize WRITE setFieldSize
           NOTIFY fieldSizeChanged) 
```

To speed up coding, you can position the cursor over the property declaration, hit the *Alt* + *Enter* combination, and choose the Generate missing Q_PROPERTY members fix-up from the pop-up menu. Creator will provide minor implementations for the getter and setter for you. You can move the generated code to the implementation file by positioning the cursor over each method, hitting *Alt* + *Enter*, and choosing the Move definition to chessview.cpp file fixup. While the generated getter method is fine, the setter needs some adjusting. Modify it by adding the following highlighted code:

```cpp
void ChessView::setFieldSize(QSize arg)
{
    if (m_fieldSize == arg) {
        return;
    }
    m_fieldSize = arg;
    emit fieldSizeChanged(arg);
    updateGeometry();
} 
```

This tells our widget to recalculate its size whenever the size of the field is modified. Now we can implement `sizeHint()`:

```cpp
QSize ChessView::sizeHint() const
{
    if(!m_board) {
        return QSize(100,100);
    }
    QSize boardSize = QSize(fieldSize().width()
        * m_board->columns() + 1,
    m_fieldSize.height() * m_board->ranks() + 1);
    // 'M' is the widest letter
    int rankSize = fontMetrics().width('M') + 4;
    int columnSize = fontMetrics().height() + 4;
    return boardSize + QSize(rankSize, columnSize);
} 
```

First, we check whether we have a valid board definition and if not, return a sane size of 100 × 100 pixels. Otherwise, the method calculates the size of all the fields by multiplying the size of each of the fields by the number of columns or ranks. We add one pixel to each dimension to accommodate the right and bottom border. A chess board not only consists of fields themselves but also displays rank symbols on the left edge of the board and column numbers on the bottom edge of the board.

Since we use letters to enumerate ranks, we check the width of the widest letter using the `QFontMetrics` class. We use the same class to check how much space is required to render a line of text using the current font so that we have enough space to put column numbers. In both cases, we add 4 to the result to make a 2 pixel margin between the text and the edge of the board and another 2 pixel margin between the text and the edge of the widget.

Actually, the widest letter in the most common fonts is W, but it won't appear in our game.

It is very useful to define a helper method for returning a rectangle that contains a particular field, as shown:

```cpp
QRect ChessView::fieldRect(int column, int rank) const
{
    if(!m_board) {
        return QRect();
    }
    const QSize fs = fieldSize();
    QPoint topLeft((column - 1) * fs.width(),
                   (m_board->ranks()-rank) * fs.height());
    QRect fRect = QRect(topLeft, fs);
    // offset rect by rank symbols
    int offset = fontMetrics().width('M');    
    return fRect.translated(offset+4, 0);
} 
```

Since rank numbers decrease from the top toward the bottom of the board, we subtract the desired rank from the maximum rank there is while calculating `fRect`. Then, we calculate the horizontal offset for rank symbols, just like we did in `sizeHint()`, and translate the rectangle by that offset before returning the result.

Finally, we can move on to implementing the event handler for the paint event. Declare the `paintEvent()` method (the fixup menu available under the *Alt* + *Enter* keyboard shortcut will let you generate a stub implementation of the method) and fill it with the following code:

```cpp
void ChessView::paintEvent(QPaintEvent *)
{
    if(!m_board) {
        return;
    }
    QPainter painter(this);
    for(int r = m_board->ranks(); r > 0; --r) {
        painter.save();
        drawRank(&painter, r);
        painter.restore();
    }
    for(int c = 1; c <= m_board->columns(); ++c) {
        painter.save();
        drawColumn(&painter, c);
        painter.restore();
    }
    for(int r = 1; r <= m_board->ranks(); ++r) {
        for(int c = 1; c <= m_board->columns(); ++c) {
            painter.save();
            drawField(&painter, c, r);
            painter.restore();
        }
    }
} 
```

The handler is quite simple. First, we instantiate the `QPainter` object that operates on the widget. Then, we have three loops: the first one iterates over ranks, the second over columns, and the third over all fields. The body of each loop is very similar; there is a call to a custom draw method that accepts a pointer to the painter and index of the rank, column, or both of them, respectively. Each of the calls is surrounded by executing `save()` and `restore()` on our `QPainter` instance. What are the calls for here? The three draw methods—`drawRank()`, `drawColumn()`, and `drawField()`—will be virtual methods responsible for rendering the rank symbol, the column number, and the field background. It will be possible to subclass `ChessView` and provide custom implementations for those renderers so that it is possible to provide a different look of the chess board. Since each of these methods takes the painter instance as its parameter, overrides of these methods can alter attribute values of the painter behind our back. Calling `save()` before handing over the painter to such override stores its state on an internal stack, and calling `restore()` after returning from the override resets the painter to what was stored with `save()`. Note that the painter can still be left in an invalid state if the override calls `save()` and `restore()`  a different number of times.

Calling `save()` and `restore()` very often introduces a performance hit, so you should avoid saving and restoring painter states too often in time-critical situations. As our painting is very simple, we don't have to worry about that when painting our chess board.

Having introduced our three methods, we can start implementing them. Let's start with `drawRank` and `drawColumn`. Remember to declare them as virtual and put them in protected access scope (that's usually where Qt classes put such methods), as follows:

```cpp
void ChessView::drawRank(QPainter *painter, int rank)
{
    QRect r = fieldRect(1, rank);
    QRect rankRect = QRect(0, r.top(), r.left(), r.height())
        .adjusted(2, 0, -2, 0);
    QString rankText = QString::number(rank);
    painter->drawText(rankRect,
       Qt::AlignVCenter | Qt::AlignRight, rankText);
}

void ChessView::drawColumn(QPainter *painter, int column)
{
    QRect r = fieldRect(column, 1);
    QRect columnRect =
        QRect(r.left(), r.bottom(), r.width(), height() - r.bottom())
        .adjusted(0, 2, 0, -2);
    painter->drawText(columnRect,
        Qt::AlignHCenter | Qt::AlignTop, QChar('a' + column - 1));
} 
```

Both methods are very similar. We use `fieldRect()` to query for the left-most column and bottom-most rank, and, based on that, we calculate where rank symbols and column numbers should be placed. The call to `QRect::adjusted()` is to accommodate the 2 pixel margin around the text to be drawn. Finally, we use `drawText()` to render appropriate text. For the rank, we ask the painter to align the text to the right edge of the rectangle and to center the text vertically. In a similar way, when drawing the column, we align to the top edge and center the text horizontally.

Now we can implement the third draw method. It should also be declared protected and virtual. Place the following code in the method body:

```cpp
void ChessView::drawField(QPainter *painter, int column, int rank)
{
    QRect rect = fieldRect(column, rank);
    QColor fillColor = (column + rank) % 2 ?
        palette().color(QPalette::Light) :
        palette().color(QPalette::Mid);
    painter->setPen(palette().color(QPalette::Dark));
    painter->setBrush(fillColor);
    painter->drawRect(rect);
} 
```

In this method, we use the `QPalette` object coupled with each widget to query for `Light` (usually white) and `Mid` (darkish) color, depending on whether the field we are drawing on the chess board is considered white or black. We do that instead of hardcoding the colors to make it possible to modify colors of the tiles without subclassing simply by adjusting the palette object. Then, we use the palette again to ask for the `Dark` color and use that as a pen for our painter. When we draw a rectangle with such settings, the pen will stroke the border of the rectangle to give it a more elegant look. Note how we modify attributes of the painter in this method and do not set them back afterward. We can get away with it because of the `save()` and `restore()` calls surrounding the `drawField()` execution.

We are now ready to see the results of our work. Let's switch to the `MainWindow` class and equip it with the following two private variables:

```cpp
ChessView *m_view;
ChessAlgorithm *m_algorithm; 
```

Then, modify the constructor by adding the following highlighted code to set up the view and the game engine:

```cpp
MainWindow::MainWindow(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    m_view = new ChessView;
    m_algorithm = new ChessAlgorithm(this);
    m_algorithm->newGame();
    m_view->setBoard(m_algorithm->board());
    setCentralWidget(m_view);
    m_view->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    m_view->setFieldSize(QSize(50,50));
    layout()->setSizeConstraint(QLayout::SetFixedSize);
} 
```

Afterward, you should be able to build the project. When you run it, you should see a result similar to the one in the following screenshot:

![](img/d8889080-7000-424f-bb79-aab60a893d2b.png)

# What just happened?

In this exercise, we did two things. First, we provided a number of methods for calculating the geometry of important parts of the chess board and the size of the widget. Second, we defined three virtual methods for rendering visual primitives of a chess board. By making the methods virtual, we provided an infrastructure to let the look be customized by subclassing and overriding base implementations. Furthermore, by reading color from `QPalette`, we allowed customizing the colors of the primitives even without subclassing.

The last line of the main window constructor tells the layout of the window to force a fixed size of the window equal to what the size hint of the widget inside it reports.

# Time for action – Rendering the pieces

Now that we can see the board, it is time to put the pieces on it. We will use images for that purpose. In my case, we found a number of SVG files with chess pieces and decided to use them. SVG is a vector graphics format where all curves are defined not as a fixed set of points but as mathematic curves. Their main benefit is that they scale very well without causing an aliasing effect.

Let's equip our view with a registry of images to be used for "stamping" a particular piece type. Since each piece type is identified with char, we can use it to generate keys for a map of images. Let's put the following API into `ChessView`:

```cpp
public:
    void setPiece(char type, const QIcon &icon);
    QIcon piece(char type) const;
private:
    QMap<char, QIcon> m_pieces; 
```

For the image type, we do not use `QImage` or `QPixmap` but `QIcon`. This is because `QIcon` can store many pixmaps of different sizes and use the most appropriate one when we request an icon of a given size to be painted. It doesn't matter if we use vector images, but it does matter if you choose to use PNG or other types of image. In such cases, you can use `addFile()` to add many images to a single icon.

Going back to our registry, the implementation is very simple. We just store the icon in a map and ask the widget to repaint itself:

```cpp
void ChessView::setPiece(char type, const QIcon &icon)
{
    m_pieces.insert(type, icon);
    update();
}

QIcon ChessView::piece(char type) const
{
    return m_pieces.value(type, QIcon());
} 
```

Now we can fill the registry with actual images right after we create the view inside the `MainWindow` constructor. Note that we stored all the images in a resource file, as shown:

```cpp
m_view->setPiece('P', QIcon(":/pieces/Chess_plt45.svg")); // pawn
m_view->setPiece('K', QIcon(":/pieces/Chess_klt45.svg")); // king
m_view->setPiece('Q', QIcon(":/pieces/Chess_qlt45.svg")); // queen
m_view->setPiece('R', QIcon(":/pieces/Chess_rlt45.svg")); // rook
m_view->setPiece('N', QIcon(":/pieces/Chess_nlt45.svg")); // knight
m_view->setPiece('B', QIcon(":/pieces/Chess_blt45.svg")); // bishop

m_view->setPiece('p', QIcon(":/pieces/Chess_pdt45.svg")); // pawn
m_view->setPiece('k', QIcon(":/pieces/Chess_kdt45.svg")); // king
m_view->setPiece('q', QIcon(":/pieces/Chess_qdt45.svg")); // queen
m_view->setPiece('r', QIcon(":/pieces/Chess_rdt45.svg")); // rook
m_view->setPiece('n', QIcon(":/pieces/Chess_ndt45.svg")); // knight
m_view->setPiece('b', QIcon(":/pieces/Chess_bdt45.svg")); // bishop 
```

The next thing to do is to extend the `paintEvent()` method of the view to actually render our pieces. For that, we will introduce another protected virtual method called `drawPiece()`. We'll call it when iterating over all the ranks and columns of the board, as shown:

```cpp
void ChessView::paintEvent(QPaintEvent *)
{
    // ...
    for(int r = m_board->ranks(); r > 0; --r) {
        for(int c = 1; c <= m_board->columns(); ++c) {
            drawPiece(&painter, c, r);
        }
    }
} 
```

It is not a coincidence that we start drawing from the highest (top) rank to the lowest (bottom) one. By doing that, we allow a pseudo-3D effect; if a piece drawn extends past the area of the field, it will intersect the field from the next rank (which is possibly occupied by another piece). By drawing higher rank pieces first, we cause them to be partially covered by pieces from the lower rank, which imitates the effect of depth. By thinking ahead, we allow reimplementations of `drawPiece()` to have more freedom in what they can do.

The final step is to provide a base implementation for this method, as follows:

```cpp
void ChessView::drawPiece(QPainter *painter, int column, int rank)
{
    QRect rect = fieldRect(column, rank);
    char value = m_board->data(column, rank);
    if(value != ' ') {
        QIcon icon = piece(value);
        if(!icon.isNull()) {
            icon.paint(painter, rect, Qt::AlignCenter);
        }
    }
} 
```

The method is very simple; it queries for the rectangle of a given column and rank and then asks the `ChessBoard` instance about the piece occupying the given field. If there is a piece there, we ask the registry for the proper icon; if we get a valid one, we call its `paint()` routine to draw the piece centered in the field's rect. The image drawn will be scaled to the size of the rectangle. It is important that you only use images with a transparent background (such as PNG or SVG files and not JPEG files) so that the color of the field can be seen through the piece.

# What just happened?

To test the implementation, you can modify the algorithm to fill the board with the default piece set up by introducing the following change to the `ChessAlgorithm` class:

```cpp
void ChessAlgorithm::newGame()
{
  setupBoard();
  board()->setFen(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
  );
} 
```

Running the program should show the following result:

![](img/c6ebcd58-228f-4567-b7ef-a4a6916db4d4.png)

The modification we did in this step was very simple. First, we provided a way to tell the board what each piece type looks like. This includes not only standard chess pieces but anything that fits into char and can be set inside the `ChessBoard` class's internal data array. Second, we made an abstraction for drawing the pieces with the simplest possible base implementation: taking an icon from the registry and rendering it to the field. By making use of `QIcon`, we can add several pixmaps of different sizes to be used with different sizes of a single field. Alternatively, the icon can contain a single vector image that scales very well all by itself.

# Time for action – Making the chess game interactive

We have managed to display the chess board, but to actually play a game, we have to tell the program what moves we want to play. We can do that by adding the `QLineEdit` widget where we will input the move in algebraic form (for example, `Nf3` to move a knight to `f3`), but a more natural way is to click on a piece with the mouse cursor (or tap it with a finger) and then click again on the destination field. To obtain such functionality, the first thing to do is to teach `ChessView` to detect mouse clicks. Therefore, add the following method:

```cpp
QPoint ChessView::fieldAt(const QPoint &pt) const
{
    if(!m_board) {
        return QPoint();
    }
    const QSize fs = fieldSize();
    int offset = fontMetrics().width('M') + 4;
    // 'M' is the widest letter
    if(pt.x() < offset) {
        return QPoint();
    }
    int c = (pt.x() - offset) / fs.width();
    int r = pt.y() / fs.height();
    if(c < 0 || c >= m_board->columns() ||
       r < 0 || r >= m_board->ranks()) {
        return QPoint();
    }
    return QPoint(c + 1, m_board->ranks() - r);
    // max rank - r
} 
```

The code looks very similar to the implementation of `fieldRect()`. This is because `fieldAt()` implements its reverse operation—it transforms a point in the widget coordinate space to the column and rank index of a field the point is contained in. The index is calculated by dividing point coordinates by the size of the field. You surely remember that, in the case of columns, the fields are offset by the size of the widest letter and a margin of 4, and we have to consider that in our calculations here as well. We do two checks: first we check the horizontal point coordinate against the offset to detect whether the user clicked on the part of the widget where column symbols are displayed, and then we check whether the rank and column calculated fit the range represented in the board. Finally, we return the result as a `QPoint` value, since this is the easiest way in Qt to represent a two-dimensional value.

Now we need to find a way to make the widget notify its environment that a particular field was clicked on. We can do this through the signal-slot mechanism. Switch to the header file of `ChessView` (if you currently have `chessview.cpp` opened in Qt Creator, you can simply press the *F4* key to be transferred to the corresponding header file) and declare a `clicked(const QPoint &)` signal:

```cpp
signals:
  void clicked(const QPoint &); 
```

To detect mouse input, we have to override one of the mouse event handlers a widget has: either `mousePressEvent` or `mouseReleaseEvent`. It seems obvious that we should choose the former event; this would work, but it is not the best decision. Just think about the semantics of a mouse click: it is a complex event composed of pushing and releasing the mouse button. The actual "click" takes place after the mouse is released. Therefore, let's use `mouseReleaseEvent` as our event handler:

```cpp
void ChessView::mouseReleaseEvent(QMouseEvent *event)
{
    QPoint pt = fieldAt(event->pos());
    if(pt.isNull()) {
        return;
    }
    emit clicked(pt);
} 
```

The code is simple; we use the method we just implemented and pass it the position read from the `QMouseEvent` object. If the returned point is invalid, we quietly return from the method. Otherwise, `clicked()` is emitted with the obtained column and rank values.

We can make use of the signal now. Go to the constructor of `MainWindow` and add the following line to connect the widget's clicked signal to a custom slot:

```cpp
connect(m_view, &ChessView::clicked,
        this,   &MainWindow::viewClicked);
```

Declare the slot and implement it, as follows:

```cpp
void MainWindow::viewClicked(const QPoint &field)
{
    if(m_clickPoint.isNull()) {
        m_clickPoint = field;
    } else {
      if(field != m_clickPoint) {
        m_view->board()->movePiece(
          m_clickPoint.x(), m_clickPoint.y(),
          field.x(), field.y()
        );
      }
      m_clickPoint = QPoint();
    }
} 
```

The function uses a class member variable—`m_clickPoint`—to store the clicked field. The variable value is made invalid after a move is made. Thus, we can detect whether the click we are currently handling has "select" or "move" semantics. In the first case, we store the selection in `m_clickPoint`; in the other case, we ask the board to make a move using the helper method we implemented some time ago. Remember to declare `m_clickPoint` as a private member variable of `MainWindow`.

All should be working now. However, if you build the application, run it, and start clicking around on the chess board, you will see that nothing happens. This is because we forgot to tell the view to refresh itself when the game position on the board is changed. We have to connect the signals that the board emits to the `update()` slot of the view. Open the `setBoard()` method of the widget class and fix it, as follows:

```cpp
void ChessView::setBoard(ChessBoard *board)
{
    // ...
    m_board = board;
    // connect signals
    if(board) {
      connect(board, SIGNAL(dataChanged(int,int)),
              this,  SLOT(update()));
      connect(board, SIGNAL(boardReset()),
              this,  SLOT(update()));
    }
    updateGeometry();
} 
```

If you run the program now, moves you make will be reflected in the widget, as shown:

![](img/73c9dcbe-c7a9-4627-acbf-ba50c4752b47.png)

At this point, we might consider the visual part of the game as finished, but there is still one problem you might have spotted while testing our latest additions. When you click on the board, there is no visual hint that any piece was actually selected. Let's fix that now by introducing the ability to highlight any field on the board.

To do that, we will develop a generic system for different highlights. Begin by adding a `Highlight` class as an internal class to `ChessView`:

```cpp
class ChessView : public QWidget
    // ...
public:
    class Highlight {
    public:
        Highlight() {}
        virtual ~Highlight() {}
        virtual int type() const { return 0; }
    };
    // ...
}; 
```

It is a minimalistic interface for highlights and only exposes a method returning the type of the highlight using a virtual method. In our exercise, we will focus on just a basic type that marks a single field with a given color. Such a situation will be represented by the `FieldHighlight` class:

```cpp
class FieldHighlight : public Highlight {
public:
    enum { Type = 1 };
    FieldHighlight(int column, int rank, QColor color)
      : m_field(column, rank), m_color(color) {}
    inline int column() const { return m_field.x(); }
    inline int rank() const { return m_field.y(); }
    inline QColor color() const { return m_color; }
    int type() const { return Type; }
private:
    QPoint m_field;
    QColor m_color;
}; 
```

You can see that we provided a constructor that takes the column and rank indices and a color for the highlight and it stores them in private member variables. Also, `type()` is redefined to return `FieldHighlight::Type`, which we can use to easily identify the type of highlight. The next step is to extend `ChessView` with abilities to add and remove highlights. As the container declares a private `QList<Highlight*> m_highlights` member variable, add method declarations:

```cpp
public:
    void addHighlight(Highlight *hl);
    void removeHighlight(Highlight *hl);
    inline Highlight *highlight(int index) const {
        return m_highlights.at(index);
    }
    inline int highlightCount() const {
        return m_highlights.size();
    } 
```

Next, provide implementations for non-inline methods:

```cpp
void ChessView::addHighlight(ChessView::Highlight *hl) {
    m_highlights.append(hl); 
    update(); 
}

void ChessView::removeHighlight(ChessView::Highlight *hl) { 
    m_highlights.removeOne(hl); 
    update(); 
} 
```

Drawing the highlights is really easy; we will use yet another virtual `draw` method. Place the following call in the `paintEvent()` implementation right before the loop that is responsible for rendering pieces:

```cpp
drawHighlights(&painter); 
```

The implementation simply iterates over all the highlights and renders those it understands:

```cpp
void ChessView::drawHighlights(QPainter *painter)
{
    for(int idx = 0; idx < highlightCount(); ++idx) {
        Highlight *hl = highlight(idx);
        if(hl->type() == FieldHighlight::Type) {
            FieldHighlight *fhl = static_cast<FieldHighlight*>(hl);
            QRect rect = fieldRect(fhl->column(), fhl->rank());
            painter->fillRect(rect, fhl->color());
        }
    }
} 
```

By checking the type of the highlight, we know which class to cast the generic
pointer to. Then, we can query the object for the needed data. Finally, we use `QPainter::fillRect()` to fill the field with the given color. As `drawHighlights()` is called before the piece painting loop and after the field painting loop, the highlight will cover the background but not the piece.

That's the basic highlighting system. Let's make our `viewClicked()` slot use it:

```cpp
void MainWindow::viewClicked(const QPoint &field)
{
    if(m_clickPoint.isNull()) {
        if(m_view->board()->data(field.x(), field.y()) != ' ') {
            m_clickPoint = field;
            m_selectedField = new ChessView::FieldHighlight(
              field.x(), field.y(), QColor(255, 0, 0, 50)
            );
            m_view->addHighlight(m_selectedField);
        }
    } else {
        if(field != m_clickPoint) {
            m_view->board()->movePiece(
                m_clickPoint.x(), m_clickPoint.y(), field.x(), field.y()
            );
        };
        m_clickPoint = QPoint();
        m_view->removeHighlight(m_selectedField);
        delete m_selectedField;
        m_selectedField = nullptr;
    }
} 
```

Note how we check that a field can only be selected if it is not empty (that is, there is an existing piece occupying that field).

You should also add a `ChessView::FieldHighlight *m_selectedField` private member variable and initialize it with a null pointer in the constructor. You can now build the game, execute it, and start moving pieces around:

![](img/0895e10a-73a0-471c-8504-441296a71bf5.png)

# What just happened?

By adding a few lines of code, we managed to make the board clickable. We connected a custom slot that reads which field was clicked on and can highlight it with a semitransparent red color. Clicking on another field will move the highlighted piece there. The highlighting system we developed is very generic. We use it to highlight a single field with a solid color, but you can mark as many fields as you want with a number of different colors, for example, to show valid moves after selecting a piece. The system can easily be extended with new types of highlights; for example, you can draw arrows on the board using `QPainterPath` to have a complex hinting system (say, showing the player the suggested move).

![](img/fb34cf99-db8a-470b-bb54-6f2f027de779.png)

# Time for action – Connecting the game algorithm

It would take us too long to implement a full chess game algorithm here, so instead, we will settle for a much simpler game called Fox and Hounds. One of the players has four pawns (hounds), which can only move over black fields and the pawn can only move in a forward fashion (toward higher ranks). The other player has just a single pawn (fox), which starts from the opposite side of the board:

![](img/582240d1-7e6f-4eb1-ba8e-1c75a7efb00f.png)

It can also move only over black fields; however it can move both forward (toward higher ranks) as well as backward (toward lower ranks). Players move their pawns in turn. The goal of the fox is to reach the opposite end of the board; the goal of the hounds is to trap the fox so that it can't make a move:

![](img/5595544e-b857-4322-8efd-9ddf79941505.png)

It's time to get to work! First, we will extend the `ChessAlgorithm` class with the required interface:

```cpp
class ChessAlgorithm : public QObject
{
    Q_OBJECT
    Q_PROPERTY(Result result READ result)
    Q_PROPERTY(Player currentPlayer
               READ currentPlayer
               NOTIFY currentPlayerChanged)
public:
    enum Result { NoResult, Player1Wins, Draw, Player2Wins };
    Q_ENUM(Result)
    enum Player { NoPlayer, Player1, Player2 };
    Q_ENUM(Player)

    explicit ChessAlgorithm(QObject *parent = 0);
    ChessBoard* board() const;
    inline Result result() const {
        return m_result;
    }
    inline Player currentPlayer() const {
        return m_currentPlayer;
    }

signals:
    void boardChanged(ChessBoard*);
    void gameOver(Result);
    void currentPlayerChanged(Player);

public slots:
    virtual void newGame();
    virtual bool move(int colFrom, int rankFrom, int colTo, int rankTo);
    bool move(const QPoint &from, const QPoint &to);

protected:
    virtual void setupBoard();
    void setBoard(ChessBoard *board);
    void setResult(Result);
    void setCurrentPlayer(Player);
private:
    ChessBoard *m_board;
    Result m_result;
    Player m_currentPlayer;
}; 
```

There are two sets of members here. First, we have a number of enums, variables, signals, and methods that are related to the state of the game: which player should make their move now and what is the result of the game currently. The `Q_ENUM` macro is used to register enumerations in Qt's metatype system so that they can be used as values for properties or arguments in signals. Property declarations and getters for them don't need any extra explanation. We have also declared protected methods for setting the variables from within subclasses. Here's their suggested implementation:

```cpp
void ChessAlgorithm::setResult(Result value)
{
    if(result() == value) {
        return;
    }
    if(result() == NoResult) {
        m_result = value;
        emit gameOver(m_result);
    } else {
        m_result = value;
    }
}

void ChessAlgorithm::setCurrentPlayer(Player value)
{
    if(currentPlayer() == value) {
        return;
    }
    m_currentPlayer = value;
    emit currentPlayerChanged(m_currentPlayer);
} 
```

Remember about initializing `m_currentPlayer` and `m_result` to `NoPlayer` and `NoResult` in the constructor of the `ChessAlgorithm` class.

The second group of functions is methods that modify the state of the game: the two variants of `move()`. The virtual variant is meant to be reimplemented by the real algorithm to check whether a given move is valid in the current game state and if that is the case, to perform the actual modification of the game board. In the base class, we can simply reject all possible moves:

```cpp
bool ChessAlgorithm::move(int colFrom, int rankFrom,
   int colTo, int rankTo)
{
    Q_UNUSED(colFrom)
    Q_UNUSED(rankFrom)
    Q_UNUSED(colTo)
    Q_UNUSED(rankTo)
    return false;
} 
```

The overload is simply a convenience method that accepts two `QPoint` objects instead of four integers:

```cpp
bool ChessAlgorithm::move(const QPoint &from, const QPoint &to)
{
    return move(from.x(), from.y(), to.x(), to.y());
} 
```

The interface for the algorithm is ready now, and we can implement it for the Fox and Hounds game. Subclass `ChessAlgorithm` to create a `FoxAndHounds` class:

```cpp
class FoxAndHounds : public ChessAlgorithm
{
public:
    FoxAndHounds(QObject *parent = 0);
    void newGame();
    bool move(int colFrom, int rankFrom, int colTo, int rankTo);
}; 
```

The implementation of `newGame()` is pretty simple: we set up the board, place pieces on it, and signal that it is time for the first player to make their move:

```cpp
void FoxAndHounds::newGame()
{
    setupBoard();
    board()->setFen("3p4/8/8/8/8/8/8/P1P1P1P1 w");
     // 'w' - white to move
    m_fox = QPoint(5,8);
    setResult(NoResult);
    setCurrentPlayer(Player1);
} 
```

The algorithm for the game is quite simple. Implement `move()` as follows:

```cpp
bool FoxAndHounds::move(int colFrom, int rankFrom,
   int colTo, int rankTo)
{
    if(currentPlayer() == NoPlayer) {
        return false;
    }

    // is there a piece of the right color?
    char source = board()->data(colFrom, rankFrom);
    if(currentPlayer() == Player1 && source != 'P') return false;
    if(currentPlayer() == Player2 && source != 'p') return false;

    // both can only move one column right or left
    if(colTo != colFrom + 1 && colTo != colFrom - 1) return false;

    // do we move within the board?
    if(colTo < 1  || colTo  > board()->columns()) return false;
    if(rankTo < 1 || rankTo > board()->ranks())   return false;

    // is the destination field black?
    if((colTo + rankTo) % 2) return false;

    // is the destination field empty?
    char destination = board()->data(colTo, rankTo);
    if(destination != ' ') return false;

    // is white advancing?
    if(currentPlayer() == Player1 && rankTo <= rankFrom) return false;

    board()->movePiece(colFrom, rankFrom, colTo, rankTo);
    // make the move
    if(currentPlayer() == Player2) {
      m_fox = QPoint(colTo, rankTo); // cache fox position
    }
    // check win condition
    if(currentPlayer() == Player2 && rankTo == 1) {
        setResult(Player2Wins); // fox has escaped
    } else if(currentPlayer() == Player1 && !foxCanMove()) {
        setResult(Player1Wins); // fox can't move
    } else {
        // the other player makes the move now
        setCurrentPlayer(currentPlayer() == Player1 ? Player2 : Player1);
    }
    return true;
} 
```

Declare a protected `foxCanMove()` method and implement it using the following code:

```cpp
bool FoxAndHounds::foxCanMove() const
{
    if(emptyByOffset(-1, -1) || emptyByOffset(-1, 1) ||
       emptyByOffset( 1, -1) || emptyByOffset( 1, 1)) {
        return true;
    }
    return false;
} 
```

Then, do the same with `emptyByOffset()`:

```cpp
bool FoxAndHounds::emptyByOffset(int x, int y) const
{
    const int destCol = m_fox.x() + x;
    const int destRank = m_fox.y() + y;
    if(destCol < 1 || destRank < 1 ||
       destCol >  board()->columns() ||
       destRank > board()->ranks()) {
        return false;
    }
    return (board()->data(destCol, destRank) == ' ');
} 
```

Lastly, declare a private `QPoint m_fox` member variable.

The simplest way to test the game is to make two changes to the code. First, in the constructor of the main window class, replace `m_algorithm = new ChessAlgorithm(this)` with `m_algorithm = new FoxAndHounds(this)`. Second, modify the `viewClicked()` slot, as follows:

```cpp
void MainWindow::viewClicked(const QPoint &field)
{
    if(m_clickPoint.isNull()) {
        // ...
    } else {
        if(field != m_clickPoint) {
            m_algorithm->move(m_clickPoint, field);
        }
        // ...
    }
} 
```

You can also connect signals from the algorithm class to custom slots of the view or window to notify about the end of the game and provide a visual hint as to which player should make their move now.

# What just happened?

We created a very simplistic API for implementing chess-like games by introducing the `newGame()` and `move()` virtual methods to the algorithm class. The former method simply sets up everything. The latter uses simple checks to determine whether a particular move is valid and whether the game has ended. We use the `m_fox` member variable to track the current position of the fox to be able to quickly determine whether it has any valid moves. When the game ends, the `gameOver()` signal is emitted and the result of the game can be obtained from the algorithm. You can use the exact same framework for implementing all chess rules.

# Have a go hero – Implementing the UI around the chess board

During the exercise, we focused on developing the game board view and necessary classes to make the game actually run. However, we completely neglected the regular user interface the game might possess, such as toolbars and menus. You can try designing a set of menus and toolbars for the game. Make it possible to start a new game, save a game in progress (say by implementing a FEN serializer), load a saved game (say by leveraging the existing FEN string parser), or choose different game types that will spawn different `ChessAlgorithm` subclasses. You can also provide a settings dialog for adjusting the look of the game board. If you feel like it, you can add chess clocks or implement a simple tutorial system that will guide the player through the basics of chess using text and visual hints via the highlight system we implemented.

# Have a go hero – Connecting a UCI-compliant chess engine

If you really want to test your skills, you can implement a `ChessAlgorithm` subclass that will connect to a **Universal Chess Interface** (**UCI**) chess engine such as StockFish ([http://stockfishchess.org](http://stockfishchess.org)) and provide a challenging artificial intelligence opponent for a human player. UCI is the de facto standard for communication between a chess engine and a chess frontend. Its specification is freely available, so you can study it on your own. To talk to a UCI-compliant engine, you can use `QProcess`, which will spawn the engine as an external process and attach itself to its standard input and standard output. Then, you can send commands to the engine by writing to its standard input and read messages from the engine by reading its standard output. To get you started, here's a short snippet of code that starts the engine and attaches to its communication channels:

```cpp
class UciEngine : public QObject {
    Q_OBJECT
public:
    UciEngine(QObject *parent = 0) : QObject(parent) {
        m_uciEngine = new QProcess(this);
        m_uciEngine->setReadChannel(QProcess::StandardOutput);
        connect(m_uciEngine, SIGNAL(readyRead()), SLOT(readFromEngine()));
    }
public slots:
    void startEngine(const QString &enginePath) {
        m_uciEngine->start(enginePath);
    }
    void sendCommand(const QString &command) {
        m_uciEngine->write(command.toLatin1());
    }
private slots:
    void readFromEngine() {
        while(m_uciEngine->canReadLine()) {
            QString line = QString::fromLatin1(m_uciEngine->readLine());
            emit messageReceived(line);
        }
    }
signals:
    void messageReceived(QString);
private:
    QProcess *m_uciEngine;
}; 
```

# Pop quiz

Q1\. Which class should you use to load a JPEG image from a file and change a few pixels in it?

1.  `QImage`
2.  `QPixmap`
3.  `QIcon`

Q2\. Which function can be used to schedule a repaint of the widget?

1.  `paintEvent()`
2.  `update()`
3.  `show()`

Q3\. Which function can be used to change the color of the outline drawn by `QPainter`?

1.  `setColor()`
2.  `setBrush()`
3.  `setPen()`

# Summary

In this chapter, we learned about using raster graphics with Qt Widgets. What was presented in this chapter will let you implement custom widgets with painting and event handling. We also described how to handle image files and do some basic painting on images. This chapter concludes our overview of CPU rendering in Qt.

In the next chapter, we will switch from raster painting to accelerated vector graphics and explore Qt capabilities related to OpenGL and Vulkan.