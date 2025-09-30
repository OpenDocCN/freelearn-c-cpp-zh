# Fluid UI with Qt Quick

My television uses Qt. My phone uses Qt. I could buy a car that uses Qt. I can fly on a plane that uses Qt on its infotainment center. All these things use Qt Quick as their UI. Why? Because it provides faster development—no waiting around for compiling—and the syntax is easy to use, but complex enough to customize it beyond your imagination.

Qt Quick started out being developed in the Brisbane development office of Trolltech as one developer's research project. One of my jobs was to put a demo app of an early version of it onto a Nokia N800 tablet, which I had customized to run Qtopia instead of Nokia's Maemo interface. This was before Nokia purchased the Trolltech company. In my opinion, it was going to become the next generation of Qtopia, which had been renamed Qt Extended. Qtopia, by 2006, had been sold on millions of phone handsets, including 11 models of phones and 30 various handheld devices. Some parts of Qtopia were melded into Qt itself – my favorites, Qt Sensors, and Qt Bearer Management, are examples of these. This new XML-like framework became QML and Qt Quick.

Qt Quick is a really exciting technology and it seems to be taking over the world. It is used in laptops, mobile phones such as the Jolla Sailfish, and medical devices, among others things.

It allows rapid development, fluid transformations, animations, and special effects. Qt Quick allows developers to design customized animated **User Interfaces** (**UI**). Along with the related Qt Quick Controls 2 and Qt Charts APIs, anyone can create snazzy mobile and embedded apps.

In this chapter, we will design and construct an animated UI. We will also cover basic components, such as `Item`, `Rectangle`, and more advanced elements, such as `GraphicsView`. We will look at positioning items with anchors, states, animations, and transitions, and we will also cover traditional features, such as buttons, sliders, and scrollbars. Advanced components showing data in charts, such as BarChart and PieChart, will be shown.

We will be covering the following topics in this chapter:

*   Learning Qt Quick basics
*   Advanced QML elements in Qt Quick Controls
*   Elements for displaying data—Qt Data Visualization and Qt Charts
*   Basic animation with Qt Quick

# Qt Quick basics – anything goes

Qt Quick is unreal. You should be aware that, at its core, it has only a few fundamental building blocks, called components. You will undoubtedly be using these components quite often:

*   `Item`
*   `Rectangle`
*   `Text`
*   `Image`
*   `TextInput`
*   `MouseArea`

Although there are probably hundreds of components and types, these items are the most important. There are also several classes of elements for text, positioning, states, animation, transitions, and transformations. Views, paths, and data handling all have their own elements.

With those building blocks, you can create fantastic UIs that are alive with animations.

The language to write Qt Quick applications is quite easy to pick up. Let's get started.

# QML

**Qt Modeling Language** (**QML**) is the declarative programming language that Qt Quick uses. Closely aligned with JavaScript, it is the centerpiece language for Qt Quick. You can use JavaScript functions within a QML document, and Qt Quick will run it.

We use Qt Quick 2 for this book, as Qt Quick 1.0 is depreciated.

All QML documents need to have one or more `import` statements.

This is about the same as C and C++'s `#include` statement.

The most basic QML will have at least one import statement, such as this:

```cpp
import QtQuick 2.12
```

The `.12` corresponds with Qt's minor version, which is the lowest version the application will support.

If you are using properties or components that were added in a certain Qt version, you will need to specify that version.

Qt Quick applications are built with building blocks known as elements, or components. Some basic types are `Rectangle`, `Item`, and `Text`.

Input interaction is supported through `MouseArea` and other items, such as `Flickable.`

One way to start developing a Qt Quick app is by using the Qt Quick app wizard in Qt Creator. You can also grab your favorite text editor and start coding away!

Let's go though some of the following concepts that are important to be aware of as terms that make up the QML language:

*   Components, types, and elements
*   Dynamic binding
*   Signal connection

# Components

Components, also known as types or elements, are objects of code and can contain both UI and non-UI aspects.

A UI component example would be the `Text` object:

```cpp
Text {
// this is a component
}
```

Component properties can be bound to variables, other properties, and values.

# Dynamic binding

Dynamic binding is a way to set a property value, which can either be a hardcoded static value, or be bound to other dynamic property values. Here, we bind the `Text` component's `id` property to `textLabel`. We can then refer it to this element just by using its `id`:

```cpp
Text {
   id: textLabel
}
```

A component can have none, one, or a few signals that can be utilized.

# Signal connections

There are two ways signals can be handled. The easiest way is by prepending `on` and then capitalizing the first letter of the particular signal. For example, a `MouseArea` has a signal named `clicked`, which can be connected by declaring `onClicked`, and then binding this to a function with curly brackets, `{ }`, or even a single line:

```cpp
MouseArea {
    onClicked: console.log("mouse area clicked!")
}
```

You can also use the `Connections` type to target some other component's signal:

```cpp
Connections {
    target: mouseArea
    onClicked: console.log("mouse area clicked!")
}
```

The model-view paradigm is not dead with Qt Quick. There are a few elements that can show data model views.

# Model-view programming

Qt Quick's views are based on a model, which can be defined either with the `model` property or as a list of elements within the component. The view is controlled by a delegate, which is any UI element capable of showing the data.

You can refer to properties of the model data in the delegate.

For example, let's declare a `ListModel`, and fill it with two sets of data. `Component` is a generic object that can be declared, and here, I use it to contain a `Text` component that will function as the delegate. The model's data with the ID of `carModel` can be referred to in the delegate. Here, there is a binding to the `text` property of the `Text` element:

The source code can be found on the Git repository under the `Chapter02-1b` directory, in the `cp2` branch.

```cpp
ListModel {
    id: myListModel
    ListElement { carModel: "Tesla" }
    ListElement { carModel: "Ford Sync 3" }
}

Component {
    id: theDelegate
    Text {
        text: carModel
    }
}
```

We can then use this model and its delegate in different views. Qt Quick has a few different views to choose from:

*   `GridView`
*   `ListView`
*   `PathView`
*   `TreeView`

Let's look at how we can use each of these.

# GridView

The `GridView` type shows model data in a grid, much like a `GridLayout`.

The grid's layout can be contained with the following properties:

*   `flow`:
    *   `GridView.FlowLeftToRight`
    *   `GridView.FlowTopToBottom`
*   `layoutDirection`:
    *   `Qt.LeftToRight`
    *   `Qt.RightToLeft` 
*   `verticalLayoutDirection`:
    *   `GridView.TopToBottom`
    *   `GridView.BottomToTop`

The `flow` property contains the way the data is presented so it becomes wrapped to the next line or column when it is appropriate. It controls the way it overflows to the next line or column. 

The icon for the following example came from [https://icons8.com](https://icons8.com).

`FlowLeftToRight` means the flow is horizontal. Here's a pictorial representation for `FlowLeftToRight`:

![](img/487b0a67-225c-4e49-bd0c-39ac5c91d0fc.png)

For `FlowTopToBottom`, the flow is vertical; here's a representation of `FlowTopToBottom`:

![](img/a15cb40a-8dd7-44bb-ab31-55010d39c798.png)

When this example gets built and run, you can resize the window by grabbing the corner with the mouse. You will get a better idea of how the flow works.

The `layoutDirection` property indicates which direction the data will be laid out. In the following case, this is `RightToLeft`:

![](img/50f2bd00-e771-4d1d-98ad-58e9f28575d5.png)

The `verticalLayoutDirection` also indicates which direction the data is laid out, except this will be vertical. Here's the `GridView.BottomToTop` representation:

![](img/e34592be-d6ca-4bf0-9537-52fc184cb427.png)

# ListView

The QML `Listview` is a type of `Flickable` element, meaning that the user can swipe or flick left or right to progress through the difference views. `ListView` looks different from the `QListView` desktop, as the items are presented in their own page, which is accessible by flicking left or right.

The layout is handled by these properties:

*   `orientation`:
    *   `Qt.horizontal`
    *   `Qt.vertical `
*   `layoutDirection`:
    *   `Qt.LeftToRight`
    *   `Qt.RightToLeft` 
*   `verticalLayoutDirection`:
    *   `ListView.TopToBottom`
    *   `ListView.BottonToTop` 

# PathView

`PathView` shows model data in a `Path`. Its delegate is a view for displaying the model data. It could be a simple drawn line, or an image with text. This can produce a flowing wheel type of data presentation. A `Path` can be constructed by one or more of the following `path` segments:

*   `PathAngleArc`: An arc with radii and center
*   `PathArc`: An arc with radius
*   `PathCurve`: A path through a set of points
*   `PathCubic`: A path on Bézier curve
*   `PathLine`: A straight line
*   `PathQuad`: A quadratic Bézier curve

Here, we use `PathArc` to display a wheel-like item model, using our `carModel`:

The source code can be found on the Git repository under the `Chapter02-1c` directory, in the `cp2` branch.

```cpp
     PathView {
         id: pathView
         anchors.fill: parent
         anchors.margins: 30
         model: myListModel
         delegate:  Rectangle {
             id: theDelegate
             Text {                 
                 text: carModel
             }
              Image {
                source: "/icons8-sedan-64.png"
             }
         }
         path: Path {
             startX: 0; startY: 40
             PathArc { x: 0; y: 400; radiusX:5; radiusY: 5 }
         }
     }
```

You should now see something like this:

![](img/e6b1ded4-1eae-4839-ab66-7217df3d6ff3.png)

There are a couple of special `path` segments that augment and change attributes of the `path`:

*   `PathAttribute`: Allows an attribute to be specified at certain points along a path
*   `PathMove`: Moves a path to a new position

# TreeView

`TreeView` is perhaps the most recognizable of these views. It looks very similar to the desktop variety. It displays a tree structure of its model data. `TreeView` has headers, called `TableViewColumn`, which you can use to add a title as well as to specify its width. Further customization can be made using `headerDelegate`, `itemDelegate`, and `rowDelegate`.

Sorting is not implemented by default, but can be controlled by a few properties:

*   `sortIndicatorColumn`: `Int`, indicating the column to be sorted
*   `sortIndicatorVisible`: `Bool` is used to enable sorting
*   `sortIndicatorOrder`: `Enum` either `Qt.AscendingOrder` or `Qt.DescendingOrder`

# Gestures and touch

Touch gestures can be an innovative way to interact with your application. To use the `QtGesture` class in Qt, you will need to implement the handlers in C++ by overriding the `QGestureEvent` class and handling the built-in `Qt::GestureType`. In this way, the following gestures can be handled:

*   `Qt::TapGesture`
*   `Qt::TapAndHoldGesture`
*   `Qt::PanGesture`
*   `Qt::PinchGesture`
*   `Qt::SwipeGesture`
*   `Qt::CustomGesture`

The `Qt::CustomGesture` flag is a special one that can be used to invent your own custom gestures.

There is one built-in gesture item in Qt Quick— `PinchArea`.

# PinchArea

 `PinchArea` handles pinch gestures, which are commonly used on mobile phones to zoom in on an image from within Qt Quick, so you can use simple QML to implement it for any `Item`-based element.

You can use the `onPinchFinished`, `onPinchStarted`, and `onPinchUpdated` signals, or set the `pinch.target` property to the target item to handle the pinch gesture.

# MultiPointTouchArea

The `MultiPointTouchArea` is not a gesture, but rather a way to track multiple points of contact of the touchscreen. Not all touchscreens support multi-touch. Mobile phones usually support multi-touch, and some embedded devices do as well. 

To use multi-point touchscreens in QML, there is the `MultiPointTouchArea` component, which works a bit like `MouseArea`. It can operate alongside `MouseArea` by setting its `mouseEnabled` property to `true`**.** This makes the `MultiPointTouchArea` component ignore events from the mouse and only respond to touch events.

Each `MultiPointTouchArea` takes an array of `TouchPoints`. Note the use of square brackets, `[ ]`—this denotes that it is an array. You can define one or more of these to handle a certain number of `TouchPoints` or fingers. Here, we define and handle only three `TouchPoints`.

If you try this on a non-touchscreen, only one green dot will track the touch point:

The source code can be found on the Git repository under the `Chapter02-2a` directory, in the `cp2` branch.

```cpp
import QtQuick 2.12
import QtQuick.Window 2.12
Window {
    visible: true
    width: 640
    height: 480
    color: "black"
    title: "You can touch this!"

    MultiPointTouchArea {        
        anchors.fill: parent
        touchPoints: [
            TouchPoint { id: touch1 },
            TouchPoint { id: touch2 },
            TouchPoint { id: touch3 }
        ]
        Rectangle {
            width: 45; height: 45
            color: "#80c342"
            x: touch1.x
            y: touch1.y
            radius: 50
            Behavior on x  {
                 PropertyAnimation {easing.type: Easing.OutBounce; duration: 500 }
             }
            Behavior on y  {
                 PropertyAnimation {easing.type: Easing.OutBounce; duration: 500 }
             }
     }
     Rectangle {
         width: 45; height: 45
         color: "#b40000"
         x: touch2.x
         y: touch2.y
         radius: 50
            Behavior on x  {
                 PropertyAnimation {easing.type: Easing.OutBounce; duration: 500 }
             }
            Behavior on y  {
                 PropertyAnimation {easing.type: Easing.OutBounce; duration: 500 }
             }
     }
     Rectangle {
         width: 45; height: 45
         color: "#6b11d8"
         x: touch2.x
         y: touch2.y
         radius: 50
            Behavior on x  {
                 PropertyAnimation {easing.type: Easing.OutBounce; duration: 500 }
             }
            Behavior on y  {
                 PropertyAnimation {easing.type: Easing.OutBounce; duration: 500 }
             }
         }
       }
}
```

You should see this when you run it on a non-touchscreen:

![](img/12f55c55-7461-43af-afd1-543b80a5696c.png)

Notice the `PropertyAnimation`? We'll get to that soon; keep reading.

# Positioning

With the myriad of different mobile phones and embedded device sizes currently available, the dynamic positioning of elements becomes more important. You may not necessarily want things placed randomly on the screen. If you have a great-looking layout on a high-DPI iPhone, it may look completely different on a small Android device, with images covering half of the screen. Automatic layouts in QML are called positioners.

Mobile and embedded devices come with a variety of screen sizes. We can better target the size variance by using dynamic layouts. 

# Layouts

These are the positioning elements that arrange the layout of the different items that you may want to use: 

*   `Grid`: Positions items in a grid
*   `Column`: Positions items vertically
*   `Row`: Positions items horizontally
*   `Flow`: Positions items side by side with wrapping

Additionally, there are also the following items:

*   `GridLayout` 
*   `ColumnLayout`
*   `RowLayout`
*   `StackLayout`

The difference between the `Grid` and the `GridLayout` elements are that the layouts are more dynamic in terms of resizing. Layouts have attached properties, so you can easily specify aspects of the layout, such as `minimumWidth`, the number of columns, or the number of rows. The item can be made to fill itself to the grid or fixed width.

You can also use *rigid* layouts which are more like tables. Let's look at using layouts that are slight less dynamic and use static sizing.

# Rigid layouts

I use the word *rigid* because they are less dynamic than all the layout items. The cell sizes are fixed and based on a percentage of the space where they are contained. They cannot span across rows or columns to fill the next column or row. Take this code, for example.

It has no layouts at all, and, when you run it, all the elements get squished together on top of one another:

The source code can be found on the Git repository under the `Chapter02-3` directory, in the `cp2` branch.

```cpp
import QtQuick 2.12
import QtQuick.Window 2.12

Window {
    visible: true
    width: 640
    height: 480
    title: qsTr("Hello World")
    Rectangle {
        width: 35
        height: 35
        gradient: Gradient {
            GradientStop { position: 0.0; color: "green"; }
            GradientStop { position: 0.25; color: "purple"; }
            GradientStop { position: 0.5; color: "yellow"; }
            GradientStop { position: 1.0; color: "black"; }
        }
    }
    Text {
        text: "Hands-On"
        color: "purple"
        font.pointSize: 20
    }
    Text {
        text: "Mobile"
        color: "red"
        font.pointSize: 20
    }
    Text {
        text: "and Embedded"
        color: "blue"
        font.pointSize: 20
    }
}
```

As you can see in the following screenshot, all of the elements are bunched up on top of each other without positioning:

![](img/0464a11d-5686-4af5-9f0a-75419a45019b.png)

This was probably not what the design team had dreamed up. Unless, of course, they did, and then wanted to use a `PropertyAnimation` value to animate the elements moving to their proper layout positions.

What happens when we add a `Column` QML element? Examine the following code:

The source code can be found on the Git repository under the `Chapter02-3a` directory, in the `cp2` branch.

```cpp

Rectangle {
 width: 500
 height: 500
     Column {
         Rectangle {
             width: 35
             height: 35
             gradient: Gradient {
                 GradientStop { position: 0.0; color: "green"; }
                 GradientStop { position: 0.25; color: "purple"; }
                 GradientStop { position: 0.5; color: "yellow"; }
                 GradientStop { position: 1.0; color: "black"; }
             }
         }

         Text {
             text: "Hands-On"
             color: "purple"
             font.pointSize: 20
         }

         Text {
             text: "Mobile"
             color: "red"
             font.pointSize: 20
         }

         Text {
             text: "and Embedded"
             color: "blue"
             font.pointSize: 20
         }
    }
}
```

When you build this example, the layout looks like this:

![](img/299cad9c-5b13-4d19-bfc4-651581471b0c.png) 

That's more like what the designer's mock-ups look like! (I know; cheap designers.)

`Flow` is another layout item we can use.

The source code can be found on the Git repository under the `Chapter02-3b`directory, in the `cp2` branch.

```cpp
    Flow {
        anchors.fill: parent
        anchors.margins: 4
        spacing: 10
```

Now, from our preceding code, change `Column` to `Flow`, add some anchor items, and build, then run on a simulator to get a feel for how the `Flow` item works on a small screen:

![](img/897f3421-5af1-49b5-8b05-eab237eb16a7.png)

The `Flow` type will wrap its contents around if needed, and, indeed, it has wrapped here on the last `Text` element. If this were to be re-oriented to the landscape orientation or on a tablet, there would be no need to wrap, and all of these elements would be on one row at the top.

# Dynamic layout

Instead of using a `Grid` element to lay out items, there is also `GridLayout`, which can be used to customize the layout. In terms of targeting mobile and embedded devices that come with different screen sizes and device orientations, it is probably better to use `GridLayout`, `RowLayout`, and `ColumnLayout`. Using these, you will gain the ability to use its attached properties. Here is a list of attached properties you can use:

| `Layout.alignment`  | A `Qt.Alignment` value specifying alignment of item within the cell |
| `Layout.bottomMargin`  | Bottom margin of space |
| `Layout.column`  | Specifies column position |
| `Layout.columnSpan`  | How many columns to spreads out to |
| `Layout.fillHeight`  | If `true`, item fills to the height |
| `Layout.fillWidth` | If `true`, item fills to the width |
| `Layout.leftMargin` | Left margin of space |
| `Layout.margins ` | All margins of space |
| `Layout.maximumHeight ` | Maximum height of item |
| `Layout.maximumWidth ` | Maximum width of item |
| `Layout.minimumHeight` | Minimum height of item |
| `Layout.minimumWidth` | Minimum width of item |
| `Layout.preferredHeight` | Preferred height of item |
| `Layout.preferredWidth` | Preferred width of item |
| `Layout.rightMargin`  | Right margin of space |
| `Layout.row` | Specifies row position |
| `Layout.rowSpan`  | How many rows to spread out to |
| `Layout.topMargin`  | Top margin of space |

In this code, we use `GridLayout` to position the three `Text` items. The first `Text` item will span, or fill, two rows so that the second `Text` will be in the second row:

The source code can be found on the Git repository under the `Chapter02-3c` directory, in the `cp2` branch.

```cpp
    GridLayout {
        rows: 3
        columns: 2
        Text {
            text: "Hands-On"
            color: "purple"
            font.pointSize: 20
        }
        Text {
            text: "Mobile"
            color: "red"
            font.pointSize: 20
        }
         Text {
            text: "and Embedded"
            color: "blue"
            font.pointSize: 20
            Layout.fillHeight: true
         }
    }
```

Positioning is a way to get dynamically changing applications and allow them to work on various devices without having to change the code. `GridLayout` works much like a layout, but with expanded capabilities.

Let's take a look at how we can dynamically position these components using `Anchors`.

# Anchors

`Anchors` are related to positioning, and are a way to position elements relative to each other. They are a way to dynamically position UI elements and layouts.

They use the following points of contact:

*   `left`
*   `right`
*   `top`
*   `bottom`
*   `horizontalCenter`
*   `verticalCenter`

Take, for example, two images; you can put them together like a puzzle by specifying anchor positions:

```cpp
Image{ id: image1; source: "image1.png"; }
Image{ id: image2; source: "image2.png; anchors.left: image1.right; }
```

This will position the left side of `image2` at the right side of `image1`. If you were to add `anchors.top: parent.top` to `image1`, both of these items would then be positioned relative to the top of the parent position. If the parent was a top-level item, they would be placed at the top of the screen.

Anchors are a way to achieve columns, rows, and grids of components that are relative to some other component. You can anchor items diagonally and anchor them apart from each other, among other things.

For example, the `anchor` property of `Rectangle`, called `fill`, is a special term meaning top, bottom, left, and right, and is bound to its parent. This means that it will fill itself to the size of its parent.

Using `anchors.top` indicates an anchor point for the top of the element, meaning that it will be bound to the parent component's top position. For example, a `Text` component will sit above of the `Rectangle` component.

To get a component such as `Text` to be centered horizontally, we use the `anchor.horizontal` property and bind it with the `parent.horizontalCenter` positional property.

Here, we anchor the `Text` label to the top center of the `Rectangle` label, itself anchored to `fill` its parent, which is the `Window`:

```cpp
import QtQuick 2.12
import QtQuick.Window 2.12

Window {
   visible: true
   width: 500
   height: 500

   Rectangle {
     anchors.fill: parent

       Text {
           id: textLabel
           text: "Hands-On Mobile and Embedded"
           color: "purple"
           font.pointSize: 20
           anchors.top: parent.top
           anchors.horizontalCenter: parent.horizontalCenter
       }
   }
}
```

The source code can be found on the Git repository under the `Chapter02` directory, in the `cp2` branch.

The `Window` component was provided by the Qt Quick app wizard and is not visible by default, so the wizard set the `visible` property to `true` as we need see it. We will use `Window` as the parent for the `Rectangle` component. Our `Rectangle` component will provide an area for our `Text` component, which is a simple label type.

Each component has its own properties to fiddle with. By fiddling, I mean binding. For instance, the `color: "purple"` line is binding the color referenced as `"purple"` to the `color` property of the `Text` element. These bindings do not have to be static; they can be dynamically changed, and the property's value that they are bound to changes as well. This value binding will persist until the property is written with another value.

The background of this application is boring. How about we add a gradient there? Under the closing bracket for the `Text` component, but still within the `Rectangle`, add this gradient. `GradientStop` is a way to specify a color at a certain point in the gradient. The `position` property is a percent fraction point from zero to one, corresponding to where the color should start. The gradient will fill in the gap in between:

```cpp
gradient: Gradient {
    GradientStop { position: 0.0; color: "green"; }
    GradientStop { position: 0.25; color: "purple"; }
    GradientStop { position: 0.75; color: "yellow"; }
    GradientStop { position: 1.0; color: "black"; }
}
```

The source code can be found on the Git repository under the `Chapter02-1` directory, in the `cp2` branch.

As you can see, the gradient starts with the green color at the top, smoothly blends to purple, then yellow, and finishes at black:

![](img/4eb99784-f4a0-4648-b9b0-eb9410b832c1.png)

Easy peasy, lemon squeezy!

Layouts and anchors are important to be able to control the UIs. They provide an easy way to deal with differences in display size and orientation changes on hundreds of different devices with different screen sizes. You could have a QML file work on all displays, although it is recommended to use different layouts for extremely different devices. An application could work fine on a tablet, or even a phone, but try to place it on a watch or other embedded device, and you will run into trouble accessing many details that your users can use.

Qt Quick has many building blocks to create useful applications on any device. What happens when you don't want to create all the UI elements yourself? That is where Qt Quick Controls come into play.

# Qt Quick Controls 2 button, button, who's got the button?

For a while in the life of Qt Quick, there were only basic components, such as `Rectangle` and `Text`. Developers had to create their own implementations of buttons, dials, and just about every common UI item. As it matured, it also grew elements such as `Window` and even `Sensor` elements. There were always rumblings about having a set of common UI elements available. Eventually, common UI elements were released.

Spotlight on Qt Quick Controls. No more having to create your own buttons and other components, yay! And developers rejoiced!

Then, they found a better way to do things and released Qt Quick Controls 2!

Qt Quick Controls comes in two versions, Qt Quick Controls and Qt Quick Controls 2\. Qt Quick Controls (the original one) has been depreciated by Qt Quick Controls 2\. Any new use of these components should use Qt Quick Controls 2. 

You can access all kinds of common UI elements, including the following: 

*   `Buttons`
*   `Containers`
*   `Input`
*   `Menu`
*   `Radio buttons`
*   `Progress bar`
*   `Popups`

Let's examine a simple Qt Quick Controls 2 example.

An `ApplicationWindow` has attached `menuBar`, `header`, and `footer` properties that you can use to add whatever you need to them. Since an `ApplicationWindow` is not visible by default, we almost always need to add `visible: true`.

Here, we will add a traditional menu with `TextField` in the header.

The menu has an `onTriggered` signal, which is used here to run the `open()` function of `MessageDialog`:

The source code can be found on the Git repository under the `Chapter02-4` directory, in the `cp2` branch.

```cpp
import QtQuick 2.12
import QtQuick.Controls 2.3
import QtQuick.Dialogs 1.1

ApplicationWindow {
   visible: true
   title: "Mobile and Embedded"
   menuBar: MenuBar {
      Menu { title: "File"
          MenuItem { text: "Open "
              onTriggered: helloDialog.open()
          }
      }
   }
   header: TextField {
       placeholderText: "Remember the Qt 4 Dance video?"
   }
   MessageDialog {
       id: helloDialog
       title: "Hello Mobile!"
       text: "Qt for Embedded devices to rule the world!"
   }
}
```

Here's what our code would result in:

![](img/550a8d92-202c-4a7d-a8f6-93b153a008c9.png)

Oooooh – fancy!

Qt Quick Controls 2 has different styles to choose from – `Default`, `Fusion`, `Imagine`, `Material`, and `Universal`. This can be set in the C++ backend as `QQuickStyle::setStyle("Fusion");`. I presume you do have a C++ backend, right?

Views that can come in handy on mobile and embedded devices are as follows:

*   `ScrollView`
*   `StackView`
*   `SwipeView`

These can be helpful on small screens, as they provide a way to easily view and access several pages without too much hassle. A `Drawer` element is also handy and can provide a way to implement a menu or a toolbar that sticks to the side.

Buttons are awesome, and Qt Quick Controls 2 has buttons. It even has the `RoundButton` component, as well as icons for the buttons! Before Qt Quick Controls, we had to roll these up ourselves. At the same time, it is nice that we can implement these things to do what we need with little effort. And now with even less effort!

Let's put some of these to the test and expand upon our last example.

I like `SwipeView`, so let's use that, with two `Page` elements as children of `SwipeView`:

The source code can be found on the Git repository under the `Chapter02-5` directory, in the `cp2` branch.

```cpp
    SwipeView {
        id: swipeView
        anchors.fill: parent

        Page {
            id: page1
            anchors.fill: parent.fill
            header: Label {
                text: "Working"
                font.pixelSize: Qt.application.font.pixelSize * 2
                padding: 10
            }
            BusyIndicator {
                id: busyId
                anchors.centerIn: parent
                running: true;
            }
            Label {
                text: "Busy Working"
                anchors.top: busyId.bottom
                anchors.horizontalCenter: parent.horizontalCenter
            }
        }

        Page {
            id: page2
            anchors.fill: parent.fill
            header: Label {
                text: "Go Back"
                font.pixelSize: Qt.application.font.pixelSize * 2
                padding: 10
            }
            Label {
                text: "Nothing here to see. Move along, move along."
                anchors.centerIn: parent
            }
        }
 }

 PageIndicator {
     id: indicator
     count: swipeView.count
     currentIndex: swipeView.currentIndex
     anchors.bottom: swipeView.bottom
     anchors.horizontalCenter: parent.horizontalCenter
 }
```

I think that a `PageIndicator` at the bottom to indicate which page we are on gives the user some visual feedback for navigation. We tie in `PageIndicator` by binding the `count` of `SwipeView` and `currentIndex` properties to its properties of the same name. How convenient!

Instead of `PageIndicator`, we could just as easily use `TabBar`.

# Customizing

You can customize the look and feel of just about every Qt Quick Control 2 component. You can override different properties of the controls, such as `background`. In the previous example code, we customized the `Page` header. Here, we override the background to a button, add our own `Rectangle`, color it, give it a border with a contrasting color, and make it rounded at the ends by using the `radius` property. Here's how it would work:

The source code can be found on the Git repository under the `Chapter02-5` directory, in the `cp2` branch.

```cpp
                Button {
                    text: "Click to go back"
                    background: Rectangle {
                        color: "#673AB7"
                        radius: 50
                        border.color: "#4CAF50"
                        border.width: 2
                    }
                    onClicked: swipeView.currentIndex = 0
                }
```

![](img/bc0f81cf-6c19-4d18-acee-a3c87f2f2b67.png)

Customizing is easy with Qt Quick. It was built with customizing in mind. The ways are endless. Nearly all the Qt Quick Controls 2 elements have visual elements that can be customized including most of the background and content items, although not all.

These controls seem to be best on a desktop, but they can be customized to work well on mobile and embedded devices. The `ScrollBar` property of `ScrollView` can be made larger in width on touchscreens.

# Show your data – Qt Data Visualization and Qt Charts

Qt Quick has a convenient way to show data of all kinds. The two modules, Qt Data Visualization and Qt Charts, can both supply integral UI elements. They are similar, except Qt Data Visualization displays data in 3D. 

# Qt Charts

Qt Charts shows 2D graphs and uses the Graphics View framework.

It adds the following chart types:

*   Area 
*   Bar 
*   Box-and-whiskers 
*   Candlestick
*   Line: a simple line chart
*   Pie: pie slices
*   Polar: a circular line
*   Scatter: a collection of points
*   Spline: a line chart with curved points

The following example from Qt shows a few different charts that are available:

![](img/386351c0-be88-4103-830c-265c626aafb0.png)

Each graph or chart has at least one axis and can have the following types:

*   Bar axis
*   Category
*   Date-time
*   Logarithmic value
*   Value

Qt Charts requires a `QApplication` instance. If you use the Qt Creator wizard to create your app, it uses a `QGuiApplication` instance by default. You will need to replace the `QGuiApplication` instance in `main.cpp` to `QApplication`, and also change the `includes` file.

You can use grid lines, shades, and tick marks on the axis, which can also be shown in these charts.

Let's look at how to create a simple BarChart.

The source code can be found on the Git repository under the `Chapter02-6` directory, in the `cp2` branch.

```cpp
import QtCharts 2.0
ChartView {     
    title: "Australian Rain"     
    anchors.fill: parent     
    legend.alignment: Qt.AlignBottom     
    antialiasing: true     

    BarSeries {         
        id: mySeries         
        axisX: BarCategoryAxis { 
            categories: ["2015", "2016", "2017" ] 
        }         
        BarSet { label: "Adelaide"; values: [536, 821, 395] }         
        BarSet { label: "Brisbane"; values: [1076, 759, 1263] }         
        BarSet { label: "Darwin"; values: [2201, 1363, 1744] }
        BarSet { label: "Melbourne"; values: [526, 601, 401] }
        BarSet { label: "Perth"; values: [729, 674, 578] }
        BarSet { label: "Sydney"; values: [1076, 1386, 1338] }
   } 
}
```

See how nice the charts look? Have a look:

![](img/8376dc5b-21cf-4c0d-a654-25cde7f49cda.png)

# Qt Data Visualization

Qt Data Visualization is similar to Qt Charts but presents data in 3D form. It can be downloaded through Qt Creator's Maintenance Tool app. It is available for use with Qt Widget and Qt Quick. We will be working with the Qt Quick version. It uses OpenGL to present 3D graphs of data.

Since we are targeting mobile phones and embedded devices, we talk about using OpenGL ES2\. There are some features of Qt Data Visualization that do not work with OpenGl ES2, which is what you will find on mobile phones:

*   Antialiasing
*   Flat shading
*   Shadows
*   Volumetric objects that use 3D textures

Let's try using a `Bars3D` with data from the total amount of rain in certain Australian cities used in the previous example.

I set the theme to `Theme3D.ThemeQt`, which is a green-based theme. Add a few customizations such as font size to be able to see the content better on small mobile displays.

 `Bar3DSeries` will manage the visual elements such as labels for rows, columns, and the data, which here is the total rain amount for that year. `ItemModelBarDataProxy` is the proxy for displaying the data. The model data here is a `ListModel` containing `ListElements` of cities rainfall data for the previous three years. We will use the same data from the previous Qt Charts example so you can compare the differences in the way the bar charts display their data:

The source code can be found on the Git repository under the `Chapter02-7` directory, in the `cp2` branch.

```cpp
import QtQuick 2.12
import QtQuick.Window 2.12
import QtDataVisualization 1.2
Window {
    visible: true
    width: 640
    height: 480
    title: qsTr("Australian Rain")
    Bars3D {
        width: parent.width
        height: parent.height
        theme: Theme3D {
            type: Theme3D.ThemeQt
            labelBorderEnabled: true
            font.pointSize: 75
            labelBackgroundEnabled: true
        }
        Bar3DSeries {
            itemLabelFormat: "@colLabel, @rowLabel: @valueLabel"
            ItemModelBarDataProxy {
                itemModel: dataModel
                rowRole: "year"
                columnRole: "city"
                valueRole: "total"
            }
        }
    }
    ListModel {
        id: dataModel
        ListElement{ year: "2017"; city: "Adelaide"; total: "536"; }
        ListElement{ year: "2016"; city: "Adelaide"; total: "821"; }
        ListElement{ year: "2015"; city: "Adelaide"; total: "395"; }
        ListElement{ year: "2017"; city: "Brisbane"; total: "1076"; }
        ListElement{ year: "2016"; city: "Brisbane"; total: "759"; }
        ListElement{ year: "2015"; city: "Brisbane"; total: "1263"; }
        ListElement{ year: "2017"; city: "Darwin"; total: "2201"; }
        ListElement{ year: "2016"; city: "Darwin"; total: "1363"; }
        ListElement{ year: "2015"; city: "Darwin"; total: "1744"; }
        ListElement{ year: "2017"; city: "Melbourne"; total: "526"; }
        ListElement{ year: "2016"; city: "Melbourne"; total: "601"; }
        ListElement{ year: "2015"; city: "Melbourne"; total: "401"; }
        ListElement{ year: "2017"; city: "Perth"; total: "729"; }
        ListElement{ year: "2016"; city: "Perth"; total: "674"; }
        ListElement{ year: "2015"; city: "Perth"; total: "578"; }
        ListElement{ year: "2017"; city: "Sydney"; total: "1076"; }
        ListElement{ year: "2016"; city: "Sydney"; total: "1386"; }
        ListElement{ year: "2015"; city: "Sydney"; total: "1338"; }
    }
}

```

You can run this on a touchscreen device, and then move the chart around in 3D!:

![](img/70a2d639-c4f5-470f-bdd7-58ebf20a5806.png)

You can grab the graph and spin it around to see the data from different perspectives. You can zoom in and back out, as well.

The `QtDataVisualization` module also has scatter and surface graphs that show data in 3D.

# Animate it!

This is where it gets gloriously complicated. There are various types of animations:

*   `ParallelAnimation`
*   `SmoothedAnimation`
*   `PauseAnimation`
*   `SequentialAnimation`

Additionally, `PropertyAction` and `ScriptAction` can be used. A `PropertyAction` is a change to any property that happens without an animation. We learned about `ScriptAction` in the last section on *States*.

There are also animation types that operate on various values:

*   `AnchorAnimation` 
*   `ColorAnimation`
*   `NumberAnimation`
*   `OpacityAnimator`
*   `PathAnimation`
*   `ParentAnimation`
*   `PropertyAnimation`
*   `RotationAnimation`
*   `SpringAnimation`
*   `Vector3DAnimation`

A `Behavior` can be used to specify an animation for a property change.

Let's look at how some of these can be used.

# Transitions

Transitions and states are explicitly tied together. A `Transition` animation happens when a `State` change occurs.

A `State` change can handle different kinds of changes:

*   `AnchorChanges`: Changes to anchor layouts
*   `ParentChanges`: Changes to parenting (as in reparenting)
*   `PropertyChanges`: Changes to the target's properties

You can even run JavaScript on `State` changes using `StateChangeScript` and `ScriptAction`.

To define different `states`, an element has a `states` array of `State` elements that can be defined. We will add a `PropertyChanges`:

```cpp
states : [
    State {
        name: "phase1"
        PropertyChanges { target: someTarget; someproperty: "some value";}
    },
    State {
        name: "phase2"
        PropertyChanges { target: someTarget; someproperty: "some other value";}
    }
]
```

Target properties can be just about anything—`opacity`, `position`, `color`, `width`, or `height`. If an element has a changeable property, the chances are that you can animate it in a `State` change.

As I mentioned before, to run a script in a `State` change, you can define a `StateChangeScript` in the `State` element that you want it to run in. Here, we simply output some logging text:

```cpp
function phase3Script() {
    console.log("demonstrate a state running a script");
}

State {
    name: "phase3"
    StateChangeScript {
        name: "phase3Action"
        script: phase3Script()
    }
}
```

Just imagine the possibilities! We haven't even presented animations! We will go there next.

# Animation

Animation can spice up your apps in wonderful ways. Qt Quick makes it almost trivial to animate different aspects of your application. At the same time, it allows you to customize them into unique and more complicated animations.

# PropertyAnimation

`PropertyAnimation` animates an item's changeable property. Typically, this is x or y color, or it can be some other property of any item:

```cpp
Behavior on activeFocus { PropertyAnimation { target: myItem; property: color; to: "green"; } }
```

The `Behavior` specifier implies that when the `activeFocus` is on `myItem`, the `color` will change to `green`.

# NumberAnimation

`NumberAnimation` derives from `PropertyAnimation`, but only works on properties that have a `qreal` changeable value: 

```cpp
NumberAnimation { target: myOtherItem; property: "y"; to: 65; duration: 250 }
```

This will move the `myOtherItem` element's `y` position to 65 over a 250-microsecond period of time.

Some of these animation elements control how other animations are played, including `SequentialAnimation` and `ParallelAnimation`.

# SequentialAnimation

`SequentialAnimation` is an animation that runs other animation types consecutively, one after the other, like a numbered procedure:

```cpp
SequentialAnimation {
    NumberAnimation { target: myOtherItem; property: "x"; to: 35; duration: 1500 }
    NumberAnimation { target: myOtherItem; property: "y"; to: 65; duration: 1500 }
}
```

In this instance, the animation that would play first is `ColorAnimation`, and, once that is finished, it would play `NumberAnimation`. Move the `myOtherItem` element's `x` property to position `35`, and then move its `y` property to position `65`, in two steps:

![](img/f45daa0e-b695-4a4d-a3a2-42e541872458.png)

You can use either `on <property>` or `properties` to target a property.

Also available is the `when` keyword, which denotes when something can take place. It can be used with any property if it evaluates to `true` or `false`, such as `when: y > 50`. You could use it, for example, on the `running` property. 

# ParallelAnimation

`ParallelAnimation` plays all its defined animations at the same time, asynchronously:

```cpp
ParallelAnimation {
    NumberAnimation { target: myOtherItem; property: "x"; to: 35; duration: 1500 }
    NumberAnimation { target: myOtherItem; property: "y"; to: 65; duration: 1500 }
}
```

These are the same animations, but this would perform them at the same time.

It is interesting to note that this animation would move `myOtherItem` to position `35` and `65` directly from where the current position is, as if it were one step:

![](img/5dd76d12-c152-4558-bd01-f58932a46035.png)

# SpringAnimation

`SpringAnimation` animates items with a spring-like motion. It has two properties to pay attention to—`spring` and `damping`:

*   `spring`: A `qreal` value that controls how energetic the bounce is
*   `damping`:How quickly the bouncing stops
*   `mass`:Adds a weight to the bounce, so it acts as if there is gravity and weight
*   `velocity`:Specifies the maximum velocity
*   `modulus`: The value at which a value will wrap around to zero
*   `epsilon`: Amount of rounding to zero

The source code can be found on the Git repository under the `Chapter02-8` directory, in the `cp2` branch.

```cpp
import QtQuick 2.12
import QtQuick.Window 2.12
Window {
    visible: true
    width: 640
    height: 480
    color: "black"
    title: qsTr("Red Bouncy Box")
    Rectangle {
        id: redBox
        width: 50; height: 50
        color: "black"
        border.width: 4
        border.color: "red"
        Behavior on x { SpringAnimation { spring: 10; damping: 10; } }
        Behavior on y { SpringAnimation { spring: 10; damping: .1;  mass: 10 } }
    }    
    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        onClicked: animation.start()
        onPositionChanged: {
            redBox.x = mouse.x - redBox.width/2
            redBox.y = mouse.y - redBox.height/2
        }
    }
    ParallelAnimation {
        id: animation
        NumberAnimation { target: redBox; property: "x"; to: 35; duration: 1500 }
        NumberAnimation { target: redBox; property: "y"; to: 65; duration: 1500 }
    }
}
```

In this example, a red square follows the finger or mouse cursor around, bouncing up and down as it goes. When the user clicks on the app, the red square will move to position `35` and `65`. A `spring` value of `10` makes it very bouncy, but the `mass` of `10` on the `y` axis will cause it to bounce like it has more weight. The lower the `damping` value is, the more quickly it will come to rest. Here, the `damping` value is much greater on the `x` axis, so it will tend to keep bouncing up and down more than side to side.

# Easing

I should mention easing at this point. Every Qt Quick animation has an `easing` property. Easing is a way to specify the speed at which the animation progresses. The default `easing` value is `Easing.Linear`. There are 40 different `easing` properties, which are probably better seen running in an example than seen here demonstrated with graphs. 

You can see a demonstration of this at my GitHub web server by the magic of Qt for WebAssembly at

[https://lpotter.github.io/easing/easing.html](https://lpotter.github.io/easing/easing.html).

Qt for WebAssembly brings Qt apps to the web. Firefox has the fastest WebAssembly implementation at the time of writing this book. We will discuss Qt for WebAssembly in [Chapter 14](04b4eb0e-2f09-4205-9d2f-ac17ff6a958d.xhtml), *Universal Platform for Mobiles and Embedded Devices*.

# SceneGraph

Scene Graph is based on OpenGL for Qt Quick. On mobile and embedded devices, it is usually OpenGL ES2\. As I mentioned before, Scene Graph caters to manage a sizable number of graphics. OpenGL is a huge subject worthy of its own book—in fact, tons of books—about OpenGL ES2 programming. I won't go into too much detail about it here, but will just mention that OpenGL is available for mobile phones and embedded devices, depending on the hardware.

If you are planning to use Scene Graph, most of the heavy lifting will be done in C++. You should already be familiar with how to use C++ and QML together, as well as OpenGL ES2\. If not, Qt has great documentation on it. 

# Summary

Qt Quick is ready-made for using on mobile and embedded devices. From the simple building blocks of basic Qt Quick items to 3D data charts, you can write complicated animated applications using various data sets and presentations in QML.

You should now be able to use basic components such as `Rectangle` or `Text` to create Qt Quick applications that use dynamic variable bindings and signals.

We also covered how to use `anchors` to position the components visually and will be able to accept changing orientations and various screen sizes of target devices.

You are now able to use more conventional-looking components such as ready-made `Button`, `Menu` and `ProgressBar` instances, as well as more advanced graphical elements such as `PieChart` and `BarChart`.

We also examined using different animation methods available in Qt Quick, such as `ProperyAnimation` and `NumberAnimation`.

In the next chapter, we will learn about using particles and special graphical effects.