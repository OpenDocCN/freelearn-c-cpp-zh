# Standard Qt Widgets

Qt Widgets are not the new kid on the block, but they still do have their place in applications that target mobile and embedded devices. They are well formed, predictable and have standard UI elements.

Recognizable UI elements are found in Qt Widgets and work great on laptops, which are simply mobile desktops. In this chapter, you will learn to design standard looking applications. Basic widgets such as menus, icons, and lists will be discussed with an emphasis on how to constrain the user interface to medium and small-sized displays. Topics we will discuss include how to use Qt's dynamic layouts to handle orientation changes. Classes such as `QGraphicsScene`, `QGraphicsView`, and `QGraphicsItem` will be used. Layout API such as `QVBoxLayout`, `QGridLayout`, and `QStackedLayout` will be discussed.

In this chapter we will cover:

*   Using Qt Creator and Qt Widgets to create a mobile app and run on the device
*   Differences between desktop and mobile apps including screen size, memory, gestures
*   Using Qt Widgets in dynamic layouts for easy screen size and orientation changes
*   Using `QGraphicsView` for graphical apps

# Hello mobile!

So you want to develop apps for mobile and embedded devices using Qt. Excellent choice, as Qt was made for cross-platform development. To get you started, we will run through the basic procedure of using Qt Creator to create, build and run an application. We will briefly examine different aspects to consider when creating mobile and embedded apps, such as how to use Qt Creator to add a menu. Adding a `QWidget` in the designer is not that difficult, and I will show you how.

Qt has a long history of running on mobile devices, starting with Qt Embedded, which was initially released in 2000\. Qt Embedded was the base framework for the UI Qtopia, which was initially released on the Sharp Zaurus on the SL-5000D developer edition.

These days, you can develop an application using Qt and sell it in the iOS App Store, Android Google Play store, or other Linux mobile phones. Qt apps run on TVs and you can even see them running on entertainment systems in cars and planes. It runs on medical devices as well as industrial automation machines on factory floors.

There are considerations for using Qt on mobile and embedded devices such as memory constraints and display size constraints. Mobiles have touchscreens, and embedded devices might not have screens at all. 

When you install Qt, you can use the Qt Creator IDE to edit, build and run your code. It's free and open source, so you can even customize it. I once had a patch that customized Qt Creator in a way that would allow me to print out all the keyboard commands that is was using, so I could have a quick reference sheet. Let's take a quick look at Qt Creator, that was once known as Workbench.

# Qt Creator

We are not going to go into any great detail about Qt Creator, but I thought I should mention it to demonstrate how we could go about using it to develop a cross-platform `QWidget` based application that runs on a desktop and mobile platform. Some differences between the two will be discussed. We'll then demonstrate how using dynamic layouts can help you target many different screen sizes and handle device orientation changes. You might already be familiar with Qt Creator, so we will refresh your memory.

# Basic Qt Creator procedure

The basic procedure for cross-compiling and building apps that run on a mobile device are straight forward after you get set up. The procedure that we would hypothetically follow is:

1.  File | New File or Project... | Qt Widgets Application, click the Choose... button
2.  Write some amazing code
3.  Select the Projects icon on the left side of Qt Creator, then pick which target platform you want like Qt 5.12.0 for iOS
4.  Hit *Ctrl* + *B*, or *Command* + *B* to build
5.  Hit *Ctrl* + *R*, or *Command *+ *R* to run
6.  Hit *F5*, or *Command* + *Y* to debug

For this first chapter, we will use Qt Widgets, which are UI elements that are more closely aligned to traditional desktop computer applications. They are still useful for mobile and embedded devices.

# Qt Designer

Qt Creator comes with a design tool named Qt Designer. When you create a new template application, you will see a list of files on the left. It will open your application form in Qt Designer when you click on any `.ui` file.

The source code can be found on the Git repository under the `Chapter01-a` directory, in the `cp1` branch.

Navigate to Forms | mainwindow.ui and double click on that. This will open the UI file in Qt Creators Designer. A UI file is just a text file in the form of XML, and you can edit that file directly if you choose. The following image shows how it looks when opened in Qt Designer:

![](img/1bead9f6-e9bb-419f-9340-5a7bcf8e06bb.png)

Let's start with something just about every desktop application has—a Menu. Your mobile or embedded application might even need a Menu. As you can see, there is a template Menu that the Qt app wizard has produced for us. We need to customize this to make it usable. We can add some sub-menu items to demonstrate basic Qt Creator functionality.

# Add a QMenu

Click on the application form where it says Menu to add menu items. Type in something like `Item1`, hit *Enter*. Add another menu item, as demonstrated in the following image:

![](img/82d969a7-759d-4030-b93a-c91190d4a7c1.png)

If you were to build this now, you would have an empty application with a Menu, so let's add more to demonstrate how to add widgets from the list of widgets that is on the left side of Qt Creator.

# Add QListView

Our UI form needs some content. We will build and run it for the desktop, then build and run it for the mobile simulator to compare the two. The procedure here is easy as drag and drop. 

On the left side of Qt Creator is a list of Widgets, Layouts and Spacers that you can simply drag and drop to place onto the template form and create your masterpiece Qt application. Let's get started:

1.  Drag a ListView and drop it on the form.

![](img/5406742a-752d-4b14-aef1-e23ed79bc91f.png)

2.  Select Desktop kit and build and run it by hitting the Run button. Qt Creator can build and run your application in the same step if you have made any changes to the form or source code. When you run it, the application should look similar to this image:

![](img/fe198551-16f7-47ae-8020-da8a2b260b7f.png)

That's all fine and dandy, but it is not running on anything small like a phone.

Qt Creator comes with iOS and Android simulators, which you can use to see how your application will run on a small screened device. It is not an emulator, which is to say it does not try to emulate the device hardware, but simply simulates the machine. In effect, Qt Creator is building and running the target architectures.

3.  Now select iOS Simulator kit, or `Android` from the Projects tool in green, as seen in the following image:

 ![](img/72629807-bb30-405a-ab3a-2cbacfc573d0.png)

4.  Build and run it, which will start it in the simulator.

Here is this app running on the iOS simulator:

![](img/679053a5-fd4a-4a8c-8bfa-6bf647c0eae1.png) 

There you go! You made a mobile app! Feels good, doesn't it? As you see, it looks slightly different in the simulator. 

# Going smaller, handling screen sizes

Porting applications which were developed for the desktop to run on smaller mobile devices can be a daunting task, depending on the application. Even creating new apps for mobiles, a few considerations need to be made, such as differences in screen resolution, memory constraints, and handling orientation changes. Touch screens add another fantastic way to offer touch gestures and can be challenging due to the differences in the size of a finger as point compared to a mouse pointer. Then there are sensors, GPS and networking to contemplate! 

# Screen resolution

As you can see in the previous images in the *Add QListView* section, the application paradigms are fairly different between desktop and mobile phones. When you move to an even smaller display, things start to get tricky in regards to fitting everything on the screen.

Luckily, there are Qt Widgets that can help. The C++ classes `QScrollArea`, `QStackedWidget` and `QTabbedWidget` can show content more appropriately. Delegating your on-screen widgets to different pages will allow your users the same ease-of-navigation which a desktop application allows.

There might also be an issue on mobile devices while using `QMenu`. They can be long, unruly and have a menu tree that drills down too deeply for a small screen. Here's a menu which works well on a desktop:

![](img/c1a1a823-53bb-46ab-a397-515e73f002b5.png)

On a mobile device, the last items on this menu become unreachable and unusable. We need to redesign this!

![](img/d1788e62-d176-4bd3-a9c6-07fa94a416d6.png)

Menus can be fixed by eliminating them or refactoring them to reduce their depth, or by using something like a `QStackedWidget` to present the Menu options.

Qt has support for high (**Dots Per Inch**) **DPI** displays. Newer versions of Qt automatically compensate for differences between high DPI and low DPI displays for iOS and the Wayland display server protocol. For Android, the environmental variable `QT_AUTO_SCALE_FACTOR` needs to be set to true. To test different scale factors, set `QT_SCALE_FACTOR`, which works best with an integer, typically 1 or 2.

Let's run through a few examples of widgets and how they can be better used on differing screens:

*   Widgets like `QScrollBar` can be increased in size to better accommodate a finger as pointer, or better yet be hidden and use the widget itself to scroll. The UI usually needs to be simplified.
*   Long `QListViews` can present some challenges. You can try to filter or add a search feature for such long lists to make the data more accessible and eye pleasing on smaller displays.
*   Even `QStackedWidget` or `QTabbedWidget` can be too big. Don't make the user flick left or right more than a few pages. Anything more can be cumbersome and annoying for the user to be flicking endlessly to get at content.
*   `QStyleSheets` are a good way to scale for smaller display's, allowing the developer to specify customizations to any widget. You can increase the padding and margins to make it easier for finger touch input. You can either set a style on a specific widget or apply it to the entire `QApplication` for a certain class of widget.

```cpp
qApp->setStyleSheet("QButton {padding: 10px;}");
```

or for one particular widget it would be:

```cpp
myButton->setStyleSheet("padding: 10px;");
```

Let's apply this only when there is a touch screen available on the device. It will make the button slightly bigger and easier to hit with a finger:

```cpp
if (!QTouchScreen::devices().isEmpty()) {
   qApp->setStyleSheet("QButton {padding: 10px;}");
}
```

If you set one style with a style sheet, you will most likely need to customize the other properties and sub-controls as well. Applying one style sheet removes the default style.

Of course, it is also easy to set a style sheet in Qt Designer, just right click on the target widget and select, Change styleSheet from the context menu. As seen here on the Apple Mac:

![](img/97df5f88-44e3-4546-bdda-d93a8e20977a.png)

Mobile phones and embedded devices have smaller displays, and they also have less RAM and storage.

# Memory and storage

Mobile phones and embedded devices usually have less memory than desktop machines. Especially for embedded devices both RAM and storage are limited.

The amount of storage space used can be lowered by optimizing images, compressing if needed. If different screen sizes are not used, the images can by manually resized instead of scaling at runtime.

There are also heap vs stack considerations which generally always pass arguments into functions by reference by using the `&` (ampersand) operator. You will notice this in the majority of Qt code.

Compiler optimizations can greatly effect both performance and the size of executables. In general, Qt's `qmake mkspec` build files are fairly good at using the correct optimizations.

If storage space is a critical consideration, then building Qt yourself is a good idea. Configuring Qt using the `-no-feature-*` to configure out any Qt features you might not need is a good way to reduce it's footprint. For example, if a device has one static Ethernet cable connection and does not need network bearer management, simply configure Qt using `-no-feature-bearermanagement`. If you know you are not using SQL why ship those storage using libraries? Running configure with `--list-features` argument will list all the features available.

# Orientation

Mobile devices move around (whodathunkit?) and sometimes it is better to view a particular app in landscape mode instead of portrait. On Android and iOS, responding to orientation changes are built in and occurs by default according to the users configuration. One thing you might need to do, is actually disable the orientation change. 

On **iOS**, you need to edit the `plist.info` file. For the key `UISupportedInterfaceOrientations`, you need to add the following:

```cpp
<array><string>UIInterfaceOrientationLandscapeLeft</string></array>
```

On **Android**, edit the `AndroidManifest.xml` file `android:screenOrientation="landscape"`

If a picture frame device has a custom-built operating system, it might need it's photo viewing app to respond when the user switches orientations. That's where Qt Sensors can help out. More on that later in the first section of [Chapter 7](0a6e358d-e771-458e-b68f-380149f259a0.xhtml), *Machines Talking*.

# Gestures

Touchscreen gestures are another way mobiles are different to desktops. Multi-touch screens have revolutionized the device world. `QPanGesture`, `QPinchGesture` and `QSwipeGesture` can be used to great effect on these devices, and Qt Quick has components build for this type of thing—`Flickable`, `SwipeView`, `PinchArea` and others. More on Qt Quick later.

To use `QGestures`, first create a `QList` of containing the gestures you want to handle, and call the `grabGesture` function for the target widget.

```cpp
QList<Qt::GestureType> gestures;
gestures << Qt::PanGesture;
gestures << Qt::PinchGesture;
gestures << Qt::SwipeGesture;
for (Qt::GestureType gesture : gestures)
    someWidget->grabGesture(gesture);
```

You will need to derive from and then override the widgets event loop to handle when the event happens.

```cpp
bool SomeWidget::event(QEvent *event)
{
    if (event->type() == QEvent::Gesture)
        return handleGesture(static_cast<QGestureEvent *>(event));
    return QWidget::event(event);
}
```

To do something useful with the gesture, we could handle it like this:

```cpp
if (QGesture *swipe = event->gesture(Qt::SwipeGesture)) {
    if (swipe->state() == Qt::GestureFinished) {
        switch (gesture->horizontalDirection()) {
            case QSwipeGesture::Left:
            break;
            case QSwipeGesture::Right:
            break;
            case QSwipeGesture::Up:
            break;
            case QSwipeGesture::Down:
            break;
        }
    }
}
```

Devices with sensors also have access to `QSensorGesture`, which enable motion gestures such as shake. More on that later, in [Chapter 7](0a6e358d-e771-458e-b68f-380149f259a0.xhtml), *Machines Talking*.

# Dynamic layouts

Considering that mobile phones come in all shapes and sizes, it would be ridiculous to need to provide a different package for every different screen resolution. Hence we will use dynamic layouts.

The source code can be found on the Git repository under the `Chapter01-layouts` directory, in the `cp1` branch.

Qt Widgets have support for this using classes such as `QVBoxLayout` and `QGridLayout`.

Qt Creator's designer is the easiest way to develop dynamic layouts. Let's go through how we can do that.

To set up a layout, we position a widget on the application form, and press *Command* or *Control* on the keyboard while selecting the widgets that we want to put in a layout. Here are two `QPushButtons` selected for use:

![](img/2d581605-39ce-43db-a05a-0598e7abfd38.png)

Next, click on the Horizontal Layout icon highlighted here:

![](img/d41f07bf-9db4-4e1c-9ffe-45fb8867935f.png)

You will then see the two widgets enclosed by a thin red box as in the following screenshot:

![](img/dd780f36-b362-4146-9bb6-c1851af0f4dc.png)

Now repeat this for the remaining widgets.

To make the widgets expand and resize with the form, click on the background and select Grid Layout:

![](img/5b78d085-2083-451f-9e1d-2802abe677df.png)

Save and build this, and this app will now be able to resize for orientation changes as well as being able to work on different sized screens. Notice how this looks like in portrait (vertical) orientation:

![](img/507a6a4b-2153-489e-9f71-eb2c6452e54b.png)

Also note how this same application looks in landscape (horizontal) orientation:

![](img/709eb346-27da-49b3-910a-e298d7bf7408.png)

As you can see, this application can change with orientation changes, but all the widgets are visible and usable. Using `QSpacer` will help guide the widgets and layouts positioning. They can push the widgets together, apart, or hold some to one side or another.

Of course, layouts can be used without touching Qt Designer. For example the following code:

```cpp
QPushButton *button = new QPushButton(this);
QPushButton *button2 = new QPushButton(this);
QBoxLayout *boxLayout = new QVBoxLayout;
boxLayout->addWidget(button);
boxLayout->addWidget(button2);
QHBoxLayout *horizontalLayout = new QHBoxLayout;
horizontalLayout->setLayout(boxLayout); 
```

`QLayout` and friends are the key to writing a cross-platform application that can accommodate the myriad screen resolutions and dynamically changing orientations of the target devices.

# Graphics view

`QGraphicsView`, `QGraphicScene` and `QGraphicsItem` provide a way for applications based on Qt Widgets to show 2D graphics.

The source code can be found on the Git repository under the `Chapter01-graphicsview` directory, in the `cp1` branch.

Every `QGraphicsView` needs a `QGraphicsScene`. Every `QGraphicsScene` needs one or more `QGraphicsItem`.

`QGraphicsItem` can be any of the following: 

*   `QGraphicsEllipseItem`
*   `QGraphicsLineItem`
*   `QGraphicsLineItem`
*   `QGraphicsPathItem`
*   `QGraphicsPixmapItem`
*   `QGraphicsPolygonItem`
*   `QGraphicsRectItem`
*   `QGraphicsSimpleTextItem`
*   `QGraphicsTextItem`

Qt Designer has support for adding `QGraphicsView` . You can follow these steps to do so:

1.  Drag the `QGraphicsView` to a new application form and fill the form with a `QGridLayout` like we did before.

![](img/d3ee54ba-fcb9-4e35-8eb2-23bde797938a.png)

2.  Implement a `QGraphicsScene` in the source code and add it to the `QGraphicsView`

```cpp
QGraphicsScene *gScene = new QGraphicsScene(this);
ui->graphicsView->setScene(gScene);
```

3.  Define a rectangle which will be the extent of the `Scene`. Here it is smaller than the size of the graphics view so we can go on and define some collision detection.

```cpp
gScene->setSceneRect(-50, -50, 120, 120);

```

4.  Create a red rectangle to show the bounding rectangle. To make it a red color, create a `QPen` which will be used to paint the rectangle and then add the rectangle to the `Scene`. 

```cpp
QPen pen = QPen(Qt::red);
gScene->addRect(gScene->sceneRect(), pen);
```

5.  Build and run the application. You will notice an app with a red bordered square on it.

As mentioned before, `QGraphicsView` shows `QGraphicsItems`. If we want to add some collision detection we need to subclass `QGraphicsSimpleTextItem`.

The header file for this is as follows:

```cpp
#include <QGraphicsScene>
#include <QGraphicsSimpleTextItem>
#include <QGraphicsItem>
#include <QPainter>
class TextGraphic :public QGraphicsSimpleTextItem
{
public:
    TextGraphic(const QString &text);
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    QString simpleText;
};
```

This custom class derived from `QGraphicsSimpleTextItem` will reimplement the `paint(..)` function, and use the `collidingItems(...)` function of `scene` to detect when something collides with our text object. Normally, `collidingItems` will return a `QList` of `QGraphicsItems`, but here it is just used to detect if any items are colliding.

Since this class holds only one item, it is known which item it is. If a collision is detected, the text changes. We don't need to check if the item's text is different before we change it, as the parent class's `setText(...)` does that for us.

```cpp
TextGraphic::TextGraphic(const QString &text)
 : QGraphicsSimpleTextItem(text),
      simpleText(text)
{
}

void TextGraphic::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    if (scene()->collidingItems(this).isEmpty())
        QGraphicsSimpleTextItem::setText("BOOM!");
    else
        QGraphicsSimpleTextItem::setText(simpleText);

    QGraphicsSimpleTextItem::paint(painter, option, widget);
}
```

Now create our `TextGraphic` object and add it to the `Scene` to use.

```cpp
TextGraphic *text = new TextGraphic(QStringLiteral("Qt Mobile!"));
gScene->addItem(text);
```

If you build and run this, notice the `text` object will not move if we try to drag it around. `QGraphicsItems` have a `flag` property called `QGraphicsItem::ItemIsMovable` that can be set to allow it to be moved around, either by the user or programmatically:

```cpp
text->setFlag(QGraphicsItem::ItemIsMovable);
```

When we build and run this, you can grab the `text` object and move it around. If it goes beyond our bounding rectangle, it will change text, only returning to the original text if it moves inside the red box again.

If you wanted to animate this, just throw in a timer and change the `text` object's position when the timer fires.

Even with Qt Quick's software renderer, `QGraphicsView` is still a viable solution for graphics animation. If the target device's storage space is really tight, there might not be enough space to add Qt Quick libraries. Or a legacy app might be difficult to import to Qt Quick.

# Summary

In this chapter we covered some of the issues facing mobile and embedded developers when trying to develop for smaller display devices, and how  `QStyleSheets` can be used to change the interface at runtime to adapt itself for using touchscreen inputs.

We discussed storage and memory space requirements, and the need to configure unneeded features out of Qt to make it have a smaller footprint.

We went through handling orientation changes and discussed using screen gestures such as `Pinch` and `Swipe`.

We learning how to use Qt Designer to add `QLayouts` to create dynamically resizing applications.

Finally, we discussed how to use `QGraphicsView` to utilize graphical elements such as graphical text and images.

Next, we will go through the next best thing since sliced bread for mobile and embedded development—Qt Quick and QML. Then we'll crack on with the real fancy stuff about graphical effects to spice up any interface!