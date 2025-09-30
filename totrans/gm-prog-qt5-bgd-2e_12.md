# Customization in Qt Quick

In the previous chapter, you learned how to use controls and layouts provided by Qt Quick to build the user interface of your application. Qt contains numerous QML types that can serve as building blocks for your game, providing rich functionality and a nice appearance. However, sometimes you need to create a custom component that satisfies the needs of your game. In this chapter, we will show a couple of convenient ways to extend your QML project with custom components. By the end of this chapter, you will know how to perform custom painting on a canvas, handle various input events, and implement lazy loading for your components. We will also see how to integrate a C++ object into QML's object tree.

The main topics covered in this chapter are as listed:

*   Creating a custom component
*   Handling mouse, touch, keyboard, and gamepad events
*   Dynamic and lazy loading
*   Painting on Canvas using JavaScript

# Creating a custom QML component

We already touched the topic of custom components when we worked with the form editor in the previous chapter. Our QML files implemented reusable components with a clean interface that can be used in the rest of the application. We will now take a more low-level approach and create a new QML component directly from QML code using the basic Qt Quick building blocks. Our component will be a button with a rounded shape and a nice background. The button will hold definable text and an icon. Our component should look good for different texts and icons.

# Time for action – Creating a button component

Start by creating a new project in Qt Creator. Choose Qt Quick Application - Empty as the project template. Name the project `custom_button` and leave the rest of the options unchanged.

At this point, you should end up with a QML document containing an empty window. Let's start by creating the button frame. Edit the `main.qml` file to add a new `Rectangle` item to the window:

```cpp
import QtQuick 2.9
import QtQuick.Window 2.2

Window {
    visible: true
    width: 640
    height: 480
    title: qsTr("Hello World")

    Rectangle {
        id: button
        anchors.centerIn: parent
        border { width: 1; color: "black" }
        radius: 5
        width: 100; height: 30
        gradient: Gradient {
            GradientStop { position: 0; color: "#eeeeee" }
            GradientStop { position: 1; color: "#777777" }
        }
    }
}
```

After running the project, you should see a result similar to the following:

![](img/479fb0fc-08a6-4fe3-afdd-0705da063a8d.png)

# What just happened?

You can see that the rectangle is centered in the window using a `centerIn` anchor binding that we didn't mention before. This is one of the two special anchors that are provided for convenience, to avoid having to write too much code. Using `centerIn` is equivalent to setting both `horizontalCenter` and `verticalCenter`. The other convenience binding is `fill`, which makes one item occupy the whole area of another item (similar to setting the left, right, top, and bottom anchors to their respective anchor lines in the destination item).

Instead of setting a solid color for the button, we declared the background to be a linear gradient. We bound a `Gradient` element to the `gradient` property and defined two `GradientStop` elements as its children, where we specified two colors to blend between. `Gradient` does not inherit from `Item` and thus is not a visual Qt Quick element. Instead, it is just a QML object that serves as a data holder for the gradient definition.

The `Item` type has a property called `children` that contains a list of the visual children (`Item` instances) of an item and another property called `resources`, which contains a list of non-visual objects (such as `Gradient` or `GradientStop`) for an item. Normally, you don't need to use these properties when adding visual or non-visual objects to an item, as the item will automatically assign child objects to appropriate properties. Note that in our code, the `Gradient` object is not a child object of the `Rectangle`; it is just assigned to its `gradient` property.

# Time for action – Adding button content

The next step is to add text and an icon to the button. First, copy the icon file to the project directory. In Qt Creator, locate the `qml.qrc` resource file in the project tree. In the context menu of the resource file, select Add Existing Files and select your icon file. The file will be added to the resources and will appear in the project tree. Our example file is called `edit-undo.png`, and the corresponding resource URL is `qrc:/edit-undo.png`.

You can get the resource path or URL of a file by locating that file in the project tree and using the Copy Path or Copy URL option in its context menu.

Next, we will add the icon and the text to our button using another item type called `Row`, as shown:

```cpp
Rectangle {
    id: button
    anchors.centerIn: parent
    border { width: 1; color: "black" }
    radius: 5
    gradient: Gradient {
        GradientStop { position: 0; color: "#eeeeee" }
        GradientStop { position: 1; color: "#777777" }
    }
    width: buttonContent.width + 8
    height: buttonContent.height + 8

    Row {
        id: buttonContent
        anchors.centerIn: parent
        spacing: 4

        Image {
            id: buttonIcon
            source: "qrc:/edit-undo.png"
        }
        Text {
            id: buttonText
            text: "ButtonText"
        }
    }
}
```

You'll get the following output:

![](img/995933a6-b01b-4bb2-bd3d-14d3b60dfafc.png)

# What just happened?

`Row` is a **positioner** QML type provided by the `QtQuick` module. Its purpose is similar to the `RowLayout` type from the `QtQuick.Layouts` module. The `Row` item spreads its children in a horizontal row. It makes it possible to position a series of items without using anchors. `Row` has the `spacing` property that dictates how much space to leave between items.

The `QtQuick` module also contains the `Column` type that arranges children in a column, the `Grid` type that creates a grid of items, and the `Flow` type that positions its children side by side, wrapping as necessary.

# Time for action – Sizing the button properly

Our current panel definition still doesn't behave well when it comes to sizing the button. If the button content is very small (for example, the icon doesn't exist or the text is very short), the button will not look good. Typically, push buttons enforce a minimum size—if the content is smaller than a specified size, the button will be expanded to the minimum size allowed. Another problem is that the user might want to override the width or height of the item. In such cases, the content of the button should not overflow past the border of the button. Let's fix these two issues by replacing the `width` and `height` property bindings with the following code:

```cpp
clip: true
implicitWidth: Math.max(buttonContent.implicitWidth + 8, 80)
implicitHeight: buttonContent.implicitHeight + 8 
```

# What just happened?

The `implicitWidth` and `implicitHeight` properties can contain the desired size the item wants to have. It's a direct equivalent of `sizeHint()` from Qt Widgets. By using these two properties instead of `width` and `height` (which are bound to `implicitWidth` and
`implicitHeight` by default), we allow the user of our component to override those implicit values. When this happens and the user does not set the width or height big enough to contain the icon and text of the button, we prevent the content from crossing the boundaries of the parent item by setting the `clip` property to `true`.

Clipping can reduce performance of your game, so use it only when necessary.

# Time for action – Making the button a reusable component

So far, we have been working on a single button. Adding another button by copying the code, changing the identifiers of all components, and setting different bindings to properties is a very tedious task. Instead, we can make our button item a real component, that is, a new QML type that can be instantiated on demand as many times as required.

First, position the text cursor in the beginning of our `Rectangle` item and press *Alt* + *Enter* on the keyboard to open the refactoring menu, like in the following screenshot:

![](img/4c4e5a63-dcf9-48c9-b5f0-85aff19c09e6.png)

From the menu, choose Move Component into Separate File. In the popup, type in a name for the new type (for example, `Button`) and check `anchors.centerIn` in the Property assignments for main.qml list:

![](img/7257c266-1ae3-45cd-8d12-2fb70bf177ad.png)

Accept the dialog by clicking on the OK button.

# What just happened?

You can see that we have a new file called `Button.qml` in the project, which contains everything the button item used to have, with the exception of the `id` and `anchors.centerIn` properties. The main file was simplified to the following:

```cpp
Window {
    visible: true
    width: 640
    height: 480
    title: qsTr("Hello World")

    Button {
        id: button
        anchors.centerIn: parent
    }
}
```

`Button` has become a component—a definition of a new type of element that can be used the same way as standard QML element types. Remember that QML component names, as well as names of files representing them, need to begin with a capital letter! If you name a file `button.qml` instead of `Button.qml`, then you will not be able to use `Button` as a component name, and trying to use "button" will result in an error message. This works both ways—every QML file starting with a capital letter can be treated as a component definition.

Since we checked `anchors.centerIn` in the dialog, this property was not moved to `Button.qml`. The reason for that choice is that our button can be put anywhere, so it can't possibly know how it should be positioned. Instead, positioning of the button should be done at the location where we use the component. Now we can edit `main.qml` to put the button into a layout or use other positioning properties without having to change the component's code.

# Importing components

A component definition can be used directly by other QML files residing in the same directory as the component definition. In our example, the `main.qml` and `Button.qml` files are located in the same directory, so you can use the `Button` QML type inside `main.qml` without having to import anything.

If you need to access a component definition from a file residing elsewhere, you will have to first import the module containing the component in the file where you want to use it. The definition of a module is very simple—it is just a relative path to the *directory* containing QML files. This means that if you have a file named `Baz.qml` in a directory called `Base/Foo/Bar` and you want to use the `Baz` component from within the  `Base/Foo/Ham.qml` file, you will have to put the following import statement in `Ham.qml`:

```cpp
import "Bar" 
```

If you want to use the same component from within the `Base/Spam.qml` file, you will have to replace the import statement with this:

```cpp
import "Foo/Bar" 
```

Importing a module makes all its components available for use. You can then declare objects of types imported from a certain module.

# QML and virtual resource paths

Our project uses a Qt resource file to make our QML files embedded into the binary and ensure that they are always available to the application, even if the source directory is not present at the computer. During startup, we refer to the main QML file using the `qrc:/main.qml` URL. This means that the runtime only sees the file hierarchy in the resource file, and the actual source directory of the project is not taken into account.

The other QML file has the `qrc:/Button.qml` URL, so Qt considers them to be in the same virtual directory and everything still works. However, if you create a QML file but forget to add it to the project's resources, Qt will be unable to load that file. Even if the file is present in the same real directory as `main.qml`, Qt will only look for it in the virtual `qrc:/` directory.

It's possible to add a file to the resources with a prefix, in which case it can have an URL like `qrc:/some/prefix/Button.qml`, and the runtime will consider it to be in another virtual directory. That being said, unless you explicitly create a new prefix, you should be fine. If your QML files are arranged in subdirectories, their hierarchy will be preserved when you add them to the resource file.

# Event handlers

Qt Quick is meant to be used for creating user interfaces that are highly interactive. It offers a number of elements for taking input events from the user. In this section, we will go through them and see how you can use them effectively.

# Time for action – Making the button clickable

So far, our component only looks like a button. The next task is to make it respond to mouse input.

The `MouseArea` QML type defines a transparent rectangle that exposes a number of properties and signals related to mouse input. Commonly used signals include `clicked`, `pressed`, and `released`. Let's do a couple of exercises to see how the element can be used.

Open the `Button.qml` file and add a `MouseArea` child item to the button and use anchors to make it fill the whole area of the button. Call the element `buttonMouseArea`. Put the following code in the body of the item:

```cpp
Rectangle {
    id: button
    // ...
    Row { ... }
    MouseArea {
        id: buttonMouseArea
        anchors.fill: parent
        onClicked: button.clicked()
    }
} 
```

In addition to this, set the following declaration in the button object just after its ID is declared:

```cpp
Rectangle {
    id: button
    signal clicked()
    // ...
} 
```

To test the modification, go to the `main.qml` file and add a signal handler to the button:

```cpp
Button {
    id: button
    anchors.centerIn: parent
    onClicked: console.log("Clicked!")
}
```

Then, run the program and click on the button. You'll see your message printed to the Qt Creator's console.

# What just happened?

With the `signal clicked()` statement, we declared that the button object can emit a signal called `clicked`. With the `MouseArea` item, we defined a rectangular area (covering the whole button) that reacts to mouse events. Then, we defined `onClicked`, which is a signal handler. For every signal an object has, a script can be bound to a handler named like the signal and prefixed with "on"; hence, for the `clicked` signal, the handler is called `onClicked`, and, for `valueChanged`, it is called `onValueChanged`.

In this particular case, we have two handlers defined—one for the button where we write a simple statement to the console, and the other for the `MouseArea` element where we call the button's signal function, effectively emitting that signal.

`MouseArea` has even more features, so now let's try putting them to the right use to make our button more feature-rich.

# Time for action – Visualizing button states

Currently, there is no visual reaction to clicking on the button. In the real world, the button has some depth and when you push it and look at it from above, its contents seems to shift a little toward the right and downward. Let's mimic this behavior by making use of the pressed property `MouseArea` has, which denotes whether the mouse button is currently being pressed (note that the pressed property is different from the pressed signal that was mentioned earlier). The content of the button is represented by the `Row` element, so add the following statements inside its definition:

```cpp
Row {
    id: buttonContent
    // ...
    anchors.verticalCenterOffset: buttonMouseArea.pressed ? 1 : 0
    anchors.horizontalCenterOffset: buttonMouseArea.pressed ? 1 : 0
    // ...
} 
```

We can also make the text change color when the mouse cursor hovers over the button. For this, we have to do two things. First, let's enable receiving hover events on the `MouseArea` by setting its `hoverEnabled` property:

```cpp
hoverEnabled: true 
```

When this property is set, `MouseArea` will be setting its `containsMouse` property to `true` whenever it detects the mouse cursor over its own area. We can use this value to set the text color:

```cpp
Text {
  id: buttonText
  text: "ButtonText"
  color: buttonMouseArea.containsMouse ? "white" : "black"
} 
```

# What just happened?

In the last exercise, we learned to use some properties and signals from `MouseArea` to make the button component more interactive. However, the element is much richer in features. In particular, if hover events are enabled, you can get the current mouse position in the item's local coordinate system through the `mouseX` and `mouseY` properties that return values. The cursor position can also be reported by handling the `positionChanged` signal. Speaking of signals, most `MouseArea` signals carry a `MouseEvent` object as their argument. This argument is called `mouse` and contains useful information about the current state of the mouse, including its position and buttons currently pressed. By default, `MouseArea` only reacts to the left mouse button, but you can use the `acceptedButtons` property to select which buttons it should handle. These features are shown in the following example:

```cpp
MouseArea {
    id: buttonMouseArea
    anchors.fill: parent
    hoverEnabled: true
    acceptedButtons: Qt.LeftButton | Qt.MiddleButton | Qt.RightButton
    onClicked: {
        switch(mouse.button) {
            case Qt.LeftButton:
                console.log("Left button clicked"); break;
            case Qt.MiddleButton:
                console.log("Middle button clicked"); break;
            case Qt.RightButton:
                console.log("Right button clicked"); break;
        }
    }
    onPositionChanged: {
        console.log("Position: [" + mouse.x + "; " + mouse.y + "]");
    }
} 
```

# Time for action – Notifying the environment about button states

We have added some code to make the button look more natural by changing its visual aspects. Now, let's extend the button programming interface so that developers can use more features of the button.

The first thing we can do is make button colors definable by introducing some new properties for the button. Let's put the highlighted code at the beginning of the button component definition:

```cpp
Rectangle {
    id: button
    property color topColor: "#eeeeee"
    property color bottomColor: "#777777"
    property color textColor: "black"
    property color textPressedColor: "white"
    signal clicked() 
```

Then, we'll use the new definitions for the background gradient:

```cpp
gradient: Gradient {
    GradientStop { position: 0; color: button.topColor }
    GradientStop { position: 1; color: button.bottomColor }
} 
```

Now for the text color:

```cpp
Text {
    id: buttonText
    text: "ButtonText"
    color: buttonMouseArea.pressed ?
        button.textPressedColor : button.textColor
} 
```

As you can note, we used the `pressed` property of `MouseArea` to detect whether a mouse button is currently being pressed on the area. We can equip our button with a similar property. Add the following code to the top level `Rectangle` of the `Button` component:

```cpp
property alias pressed: buttonMouseArea.pressed 
```

# What just happened?

The first set of changes introduced four new properties defining four colors that we later used in statements defining gradient and text colors for the button. In QML, you can define new properties for objects with the `property` keyword. The keyword should be followed by the property type and property name. QML understands many property types, the most common being `int`, `real`, `string`, `font`, and `color`. Property definitions can contain an optional default value for the property, preceded with a colon. The situation is different with the pressed property definition.

You can see that for the property type, the definition contains the word `alias`. It is not a property type but an indicator that the property is really an alias to another property—each time the `pressed` property of the button is accessed, the value of the `buttonMouseArea.pressed` property is returned, and every time the property is changed, it is the mouse area's property that really gets changed. With a regular property declaration, you can provide any valid expression as the default value because the expression is bound to the property. With a property alias, it is different—the value is mandatory and has to be pointing to an existing property of the same or another object.

Consider the following two definitions:

```cpp
property int foo: someobject.prop
property alias bar: someobject.prop 
```

At first glance, they are similar as they point to the same property and therefore the values returned for the properties are the same. However, the properties are really very different, which becomes apparent if you try to modify their values:

```cpp
  foo = 7
  bar = 7 
```

The first property actually has an expression bound to it, so assigning `7` to `foo` simply releases the binding and assigns the value `7` to the `foo` property, leaving `someobject.prop` with its original value. The second statement, however, is an alias; therefore, assigning a new value applies the modification to the `someobject.prop` property the alias is really pointing to.

Speaking of properties, there is an easy way to react when a property value is modified. For each existing property, there is a handler available that is executed whenever the property value is modified. The handler name is `on` followed by the property name, then followed by the word `Changed`, all in camel case—thus, for a `foo` property, it becomes `onFooChanged` and for `topColor`, it becomes `onTopColorChanged`. To log the current press state of the button to the console, all we need to do is implement the property change handler for this property:

```cpp
Button {
    // ...

    onPressedChanged: {
        console.log("The button is currently " +
                    (pressed ? "" : "not ") + "pressed")
    }
}
```

In this example, we created a fully functional custom QML component. Our button reacts to mouse input and exposes some useful properties and signals to the user. This makes it a reusable and customizable object. In a real project, always think of the repeating parts of your UI and consider moving them into a single component to reduce code duplication.

# Touch input

`MouseArea` is the simplest of input event elements. Nowadays, more and more devices have touch capabilities and Qt Quick can handle them as well. Currently, we have three ways of handling touch input.

First of all, simple touch events are also reported as mouse events. Tapping and sliding a finger on the screen can be handled using `MouseArea`, just like mouse input.

# Time for action – Dragging an item around

Create a new Qt Quick Application - Empty project. Edit the `main.qml` file to add a circle to the window:

```cpp
Rectangle {
    id: circle
    width: 60; height: width
    radius: width / 2
    color: "red"
} 
```

Next, add a `MouseArea` to the circle and use its `drag` property to enable moving the circle by touch (or mouse):

```cpp
Rectangle {
    //...
    MouseArea {
        anchors.fill: parent
        drag.target: circle
    }
}
```

Then, you can start the application and begin moving the circle around.

# What just happened?

A circle was created by defining a rectangle with its height equal to width, making it a square and rounding the borders to half the side length. The `drag` property can be used to tell `MouseArea` to manage a given item's position using input events flowing into this `MouseArea` element. We denote the item to be dragged using the `target` subproperty. You can use other subproperties to control the axis the item is allowed to move along or constrain the move to a given area. An important thing to remember is that the item being dragged cannot be anchored for the axis on which the drag is requested; otherwise, the item will respect the anchor and not the drag. We didn't anchor our circle item at all since we want it to be draggable along both axes.

The second approach to handling touch input in Qt Quick applications is to use `PinchArea`, which is an item similar to `MouseArea`, but rather than dragging an item around, it allows you to rotate or scale it using two fingers (with a so called "pinch" gesture), as shown:

![](img/aceb744d-bcdc-470b-b636-76654417b369.png)

Be aware that `PinchArea` reacts only to touch input, so to test the example, you will need a real multitouch capable device.

# Time for action – Rotating and scaling a picture by pinching

Start a new Qt Quick Application - Empty project. Add an image file to the resources, just like we previously did in the button project. In the `main.qml` file, add an image to the window and make it centered in its parent:

```cpp
Image {
    id: image
    anchors.centerIn: parent
    source: "qrc:/wilanow.jpg"
}
```

Now, we will add a `PinchArea` element. This kind of item can be used in two ways—either by manually implementing signal handlers `onPinchStarted`, `onPinchUpdated`, and `onPinchFinished` to have total control over the functionality of the gesture, or using a simplified interface similar to the `drag` property of `MouseArea`. Since the simplified interface does exactly what we want, there is no need to handle pinch events manually. Let's add the following declaration to the file:

```cpp
PinchArea {
    anchors.fill: parent
    pinch {
        target: image
        minimumScale: 0.2
        maximumScale: 2.0
        minimumRotation: -90
        maximumRotation: 90
    }
}
```

You'll get an output similar to the following screenshot:

![](img/12e68796-9807-4696-93d9-c65d6fdf5c94.png)

# What just happened?

Our simple application loads an image and centers it in the view. Then, there is a `PinchArea` item filling the view area that is told to operate on the image object. We define the range of the scaling and rotating of the item. The rest is left to the `PinchArea` item itself. If you start interacting with the application, you will see the item rotate and scale. What really happens behind the scenes is that `PinchArea` modifies the values of the two properties each Qt Quick item has—`rotation` and `scale`.

`PinchArea` can also control the dragging of the item with `pinch.dragAxis`, just like `MouseArea` does with drag, but for simplicity, we didn't use this part of the API. Feel free to experiment with it in your own code.

# Have a go hero – Rotating and scaling with a mouse

Of course, you don't have to use `PinchArea` to rotate or scale an item. Properties controlling those aspects are regular properties that you can read and write at any time. Try replacing `PinchArea` with `MouseArea` to obtain a result similar to what we just did by modifying the scale and rotation properties as a result of receiving mouse events—when the user drags the mouse while pressing the left button, the image is scaled and when the user does the same while pressing the right button, the image is rotated.

If you manage to do the task, try replacing `MouseArea` with `PinchArea` again, but then, instead of using the `pinch` property, handle events manually to obtain the same effect (the event object is called `pinch` and has a number of properties you can play with).

A third approach to handling touch input is using the `MultiPointTouchArea` item. It provides a low-level interface to gestures by reporting each touch point separately. It can be used to create custom high-level gesture handlers similar to `PinchArea`.

# Keyboard input

So far, we've been dealing with pointer input, but user input is not just that—we can also handle keyboard input. This is quite simple and basically boils down to two easy steps.

First, you have to enable receiving keyboard events by stating that a particular item has keyboard focus:

```cpp
focus: true 
```

Then, you can start handling events by writing handlers in a similar fashion as for mouse events. However, `Item` doesn't provide its own handler for manipulating keys that is a counterpart for `keyPressEvent` and `keyReleaseEvent` of `QWidget`. Instead, adequate handlers are provided by the `Keys` attached property.

Attached properties are provided by elements that are not used as standalone elements but provide properties to other objects by getting attached to them as well. This is a way of adding support for new properties without modifying the API of the original element (it doesn't add new properties through an **is-a** relation, but rather through a **has-a** one). Each object that references an attached property gets its own copy of the attaching object that then handles the extra properties. We will come back to attached properties later in the chapter. For now, you just need to remember that in certain situations, an element can obtain additional properties that are not part of its API.

Let's go back to implementing event handlers for keyboard input. As we said earlier, each Item has a `Keys` attached property that allows us to install our own keyboard handlers. The basic two signals `Keys` adds to `Item` are `pressed` and `released`; therefore, we can implement the `onPressed` and `onReleased` handlers that have a `KeyEvent` argument providing similar information as `QKeyEvent` in the widget world. As an example, we can see an item that detects when the spacebar was pressed:

```cpp
Rectangle {
    focus: true
    color: "black"
    width: 100
    height: 100
    Keys.onPressed: {
        if(event.key === Qt.Key_Space) {
             color = "red";
        }
    }
    Keys.onReleased: {
        if(event.key === Qt.Key_Space) {
            color = "blue";
        }
    }
}
```

It might become problematic if you want to handle many different keys in the same item, as the `onPressed` handler would likely contain a giant switch section with branches for every possible key. Fortunately, `Keys` offers more properties. Most of the commonly used keys (but not letters) have their own handlers that are called when the particular key is pressed. Thus, we can easily implement an item that takes a different color depending on which key was pressed last:

```cpp
Rectangle {
    //...
    focus: true
    Keys.onSpacePressed:      color = "purple"
    Keys.onReturnPressed:     color = "navy"
    Keys.onVolumeUpPressed:   color = "blue"
    Keys.onRightPressed:      color = "green"
    Keys.onEscapePressed:     color = "yellow"
    Keys.onTabPressed:        color = "orange"
    Keys.onDigit0Pressed:     color = "red"
} 
```

Note that the `released` signal will still be emitted for every released key even if the key has its own pressed signal.

Now, consider another example:

```cpp
Item {
    id: item
    property int number: 0
    width: 200; height: width
    focus: true
 Keys.onSpacePressed: {
 number++;
 }
    Text {
        text: item.number
        anchors.centerIn: parent
    }
}
```

We would expect that when we press and hold the spacebar, we will see the text change from `0` to `1` and stay on that value until we release the key. If you run the example, you will see that instead, the number keeps incrementing as long as you hold down the key. This is because by default, the keys autorepeat—when you hold the key, the operating system keeps sending a sequence of press-release events for the key (you can verify that by adding the `console.log()` statements to the `Keys.onPressed` and `Keys.onReleased` handlers). To counter this effect, you can differentiate between autorepeat and regular events. In Qt Quick, you can do this easily, as each key event carries the appropriate information. Simply replace the handler from the last example with the following one:

```cpp
Keys.onSpacePressed: {
    if(!event.isAutoRepeat) {
        number++;
    }
}
```

The event variable we use here is the name of the parameter of the `spacePressed` signal. As we cannot declare our own names for the parameters like we can do in C++, for each signal handler, you will have to look up the name of the argument in the documentation. You can search for `Keys` in the documentation index to open the Keys QML Type page. The signal list will contain type and name of the signal's parameter, for example, `spacePressed(KeyEvent event)`.

Whenever you process an event, you should mark it as accepted to prevent it from being propagated to other elements and handled by them:

```cpp
Keys.onPressed: {
    if(event.key === Qt.Key_Space) {
        color = "blue";
        event.accepted = true;
    }
}
```

However, if you use a handler dedicated to an individual button (like `onSpacePressed`), you don't need to accept the event, as Qt will do that for you automatically.

In standard C++ applications, we usually use the *Tab* key to navigate through focusable items. With games (and fluid user interfaces in general), it is more common to use arrow keys for item navigation. Of course, we can handle this situation using the `Keys` attached property and adding `Keys.onRightPressed`, `Keys.onTabPressed`, and other signal handlers to each of our items where we want to modify the focus property of the desired item, but it would quickly clutter our code. Qt Quick comes to our help once again by providing a `KeyNavigation` attached property, which is meant to handle this specific situation and allows us to greatly simplify the needed code. Now, we can just specify which item should get into focus when a specific key is triggered:

```cpp
Row {
    spacing: 5
    Rectangle {
        id: first
        width: 50; height: width
        color: focus ? "blue" : "lightgray"
        focus: true
        KeyNavigation.right: second
    }
    Rectangle {
        id: second
        width: 50; height: width
        color: focus ? "blue" : "lightgray"
        KeyNavigation.right: third
    }
    Rectangle {
        id: third
        width: 50; height: width
        color: focus ? "blue" : "lightgray"
    }
} 
```

Note that we made the first item get into focus in the beginning by explicitly setting the `focus` property. By setting the `KeyNavigation.right` property, we instruct Qt to focus on the specified item when this item receives a right key press event. The reverse transition is added automatically—when the left key is pressed on the second item, the first item will receive focus. Besides `right`, `KeyNavigation` contains the `left`, `down`, `up`, `tab`, and `backtab` (*Shift* + *Tab*) properties.

Both the `Keys` and `KeyNavigation` attached properties have a way to define
the order in which each of the mechanisms receive the events. This is handled by the `priority` property, which can be set to either `BeforeItem` or `AfterItem`. By default, `Keys` will get the event first (`BeforeItem`), then the internal event handling can take place and finally, `KeyNavigation` will have a chance of handling the event (`AfterItem`). Note that if the key is handled by one of the mechanisms, the event is accepted and the remaining mechanisms will not receive that event.

# Have a go hero – Practicing key-event propagation

As an exercise, you can expand our last example by building a larger array of items (you can use the `Grid` element to position them) and defining a navigation system that makes use of the `KeyNavigation` attached property. Have some of the items handle events themselves using the `Keys` attached property. See what happens when the same key is handled by both mechanisms. Try influencing the behavior using the `priority` property.

When you set the `focus` property of an item to `true`, any previously used item loses focus. This becomes a problem when you try to write a reusable component that needs to set focus to its children. If you add multiple instances of such a component to a single window, their focus requests will conflict with each other. Only the last item will have focus because it was created last. To overcome this problem, Qt Quick introduces a concept of **focus scopes**. By wrapping your component into a `FocusScope` item, you gain ability to set focus to an item inside the component without influencing the global focus directly. When an instance of your component receives focus, the internal focused item will also receive focus and will be able to handle keyboard events. A good explanation of this feature is given on the Keyboard Focus in Qt Quick documentation page.

# Text input fields

Apart from the attached properties we described, Qt Quick provides built-in elements for handling keyboard input. The two most basic types are `TextInput` and `TextEdit`, which are QML equivalents of `QLineEdit` and `QTextEdit`. The former are used for single-line text input, while the latter serves as its multiline counterpart. They both offer cursor handling, undo-redo functionality, and text selections. You can validate text typed into `TextInput` by assigning a validator to the `validator` property. For example, to obtain an item where the user can input a dot-separated IP address, we can use the following declaration:

```cpp
TextInput {
    id: ipAddress
    width: 100
    validator: RegExpValidator {
        // four numbers separated by dots
        regExp: /\d+\.\d+\.\d+\.\d+/
    }
    focus: true
} 
```

The regular expression only verifies the format of the address. The user can still insert bogus numbers. You should either do a proper check before using the address or provide a more complex regular expression that will constrain the range of numbers the user can enter.

One thing to remember is that neither `TextInput` nor `TextEdit` has any visual appearance (apart from the text and cursor they contain), so if you want to give the user some visual hint as to where the item is positioned, the easiest solution is to wrap it in a styled rectangle:

```cpp
Rectangle {
  id: textInputFrame
  width: 200
  height: 40
  border { color: "black"; width: 2 }
  radius: 10
  antialiasing: true
  color: "darkGray"
}
TextInput {
  id: textInput
  anchors.fill: textInputFrame
  anchors.margins: 5
  font.pixelSize: height-2
  verticalAlignment: TextInput.AlignVCenter
  clip: true
} 
```

Note that the highlighted code—the `clip` property of `textInput`—is enabled such that by default, if the text entered in the box doesn't fit in the item, it will overflow it and remain visible outside the actual item. By enabling clipping, we explicitly say that anything that doesn't fit the item should not be drawn.

![](img/b2e5449b-d3b5-4db5-a7b5-089516765a84.png)

The `QtQuick.Controls` module provides more advanced text input controls, such as `TextField` and `TextArea`. We've already used them in our project in [Chapter 11](b81d9c47-58fa-49dd-931a-864c7be05840.xhtml), *Introduction to Qt Quick*.

# Gamepad input

Handling gamepad events is a very common task when developing a game. Fortunately, Qt provides Qt Gamepad module for this purpose. We already learned how to use it in C++. Now let's see how to do this in QML applications.

To enable Qt Gamepad module, add `QT += gamepad` to the project file. Next, import the QML module by adding the following line at the beginning of your QML file:

```cpp
import QtGamepad 1.0
```

This import allows you to declare objects of the `Gamepad` type. Add the following object inside your top-level QML object:

```cpp
 Gamepad {
     id: gamepad
     deviceId: GamepadManager.connectedGamepads.length > 0 ? 
         GamepadManager.connectedGamepads[0] : -1
 }
```

The `GamepadManager` object allows us to list identifiers of gamepads available in the system. We use the first available gamepad if any are present in the system. If you want the game to pick up a connected gamepad on the fly, use the following code snippet:

```cpp
    Connections {
        target: GamepadManager
        onGamepadConnected: gamepad.deviceId = deviceId
    }
```

# What just happened?

The preceding code simply adds a signal handler for the `gamepadConnected` signal of the `GamepadManager` object. The usual way to add a signal handler is to declare it directly in the section of the sender. However, we can't do that in this case, since  `GamepadManager` is an existing global object that is not part of our QML object tree. Thus, we use the `Connections` QML type that allows us to specify an arbitrary sender (using the `target` property) and attach a signal handler to it. You can think of `Connections` as a declarative version of `QObject::connect` calls.

The initialization is done, so we can now use the `gamepad` object to request information about the gamepad input. There are two ways to do that.

First, we can use property bindings to set properties of other objects depending on the buttons pressed on the gamepad:

```cpp
Text {
    text: gamepad.buttonStart ? "Start!" : ""
}
```

Whenever the start button is pressed or released on the gamepad, the value of the `gamepad.buttonStart` property will be set to `true` or `false`, and the QML engine will automatically update the displayed text.

The second way is to add a signal handler to detect when a property changes:

```cpp
Gamepad {
    //...
    onButtonStartChanged: {
        if (value) {
            console.log("start pressed");
        } else {
            console.log("start released");
        }
    }
}
```

The `Gamepad` QML type has a separate property and signal for each gamepad button, just like the `QGamepad` C++ class.

You can also use the `GamepadKeyNavigation` QML type to introduce gamepad support to a game that supports keyboard input:

```cpp
GamepadKeyNavigation {
    gamepad: gamepad
    active: true
    buttonStartKey: Qt.Key_S
}
```

When this object is declared in your QML file, gamepad events provided by the `gamepad` object will be automatically converted to key events. By default, `GamepadKeyNavigation` is able to emulate up, down, left, right, back, forward, and return keys when the corresponding gamepad buttons are pressed. However, you can override the default mapping or add your own mapping for other gamepad buttons. In the preceding example, we tell `GamepadKeyNavigation` that the start key on the gamepad should act as if the *S* key was pressed on the keyboard. You can now handle these events just as any regular keyboard event.

# Sensor input

Qt is reaching out to more and more platforms that are used nowadays. This includes a number of popular mobile platforms. Mobile devices are usually equipped with additional hardware, less often seen on desktop systems. Let's see how to handle sensor input in Qt so that you can use it in your games.

Most of the features discussed in this section are not usually available on desktops. If you want to play with them, you need to set up running Qt applications on a mobile device. This requires a few configuration steps that depend on your target platform. Please refer to Qt documentation for exact instructions, as they will offer complete and up-to-date information that wouldn't be possible to provide in a book. Good starting points are Getting Started with Qt for Android  and Qt for iOS documentation pages.

Access to sensors present on mobile devices is provided by the Qt Sensors module and must be imported before it can be used:

```cpp
import QtSensors 5.0
```

There are a lot of QML types you can use to interact with sensors. Have a look at this impressive list:

| **QML type** | **Description** |
| --- | --- |
| `Accelerometer` | Reports the device's linear acceleration along the *x*, *y*, and *z* axes. |
| `Altimeter` | Reports the altitude in meters relative to mean sea level. |
| `AmbientLightSensor` | Reports the intensity of the ambient light. |
| `AmbientTemperatureSensor` | Reports the temperature in degree Celsius of the current device's ambient. |
| `Compass` | Reports the azimuth of the device's top as degrees from magnetic north. |
| `DistanceSensor` | Reports distance in cm from an object to the device. |
| `Gyroscope` | Reports the device's movement around its axes in degrees per second. |
| `HolsterSensor` | Reports if the device is holstered in a specific pocket. |
| `HumiditySensor` | Reports on humidity. |
| `IRProximitySensor` | Reports the reflectance of the outgoing infrared light. The range is from 0 (zero reflection) to 1 (total reflection). |
| `LidSensor` | Reports on whether a device is closed. |
| `LightSensor` | Reports the intensity of light in lux. |
| `Magnetometer` | Reports the device's magnetic flux density along its axes. |
| `OrientationSensor` | Reports the orientation of the device. |
| `PressureSensor` | Reports the atmospheric pressure in Pascals. |
| `ProximitySensor` | Reports if an object is close to the device. Which distance is considered "close" depends on the device. |
| `RotationSensor` | Reports the three angles that define the orientation of the device in a three-dimensional space. |
| `TapSensor` | Reports if a device was tapped. |
| `TiltSensor` | Reports the angle of tilt in degrees along the device's axes. |

Unfortunately, not all sensors are supported on all platforms. Check out the Compatibility Map documentation page to see which sensors are supported on your target platforms before trying to use them.

All these types inherit the `Sensor` type and provide similar API. First, create a sensor object and activate it by setting its `active` property to `true`. When the hardware reports new values, they are assigned to the sensor's `reading` property. As with any property in QML, you can choose between using the property directly, using it in a property binding, or using the `onReadingChanged` handler to react to each new value of the property.

The type of the `reading` object corresponds to the type of the sensor. For example, if you use a tilt sensor, you'll receive a `TiltReading` object that provides suitable properties to access the angle of tilt around the *x* (`xRotation`) and *y* (`yRotation`) axes. For each sensor type, Qt provides a corresponding reading type that contains the sensor data.

All readings also have the `timestamp` property that contains the number of microseconds since some fixed point. That point can be different for different sensor objects, so you can only use it to calculate time intervals between two readings of the same sensor.

The following QML code contains a complete example of using a tilt sensor:

```cpp
import QtQuick 2.9
import QtQuick.Window 2.2
import QtSensors 5.0
Window {
    visible: true
    width: 640
    height: 480
    title: qsTr("Hello World")
    Text {
        anchors.centerIn: parent
        text: {
            if (!tiltSensor.reading) {
                return "No data";
            }
            var x = tiltSensor.reading.xRotation;
            var y = tiltSensor.reading.yRotation;
            return "X: " + Math.round(x) +
                   " Y: " + Math.round(y)
        }
    }
    TiltSensor {
        id: tiltSensor
        active: true
        onReadingChanged: {
            // process new reading
        }
    }
}
```

When this application receives a new reading, the text on screen will be automatically updated. You can also use the `onReadingChanged` handler to process new data in another way.

# Detecting device location

Some modern games require information about the player's geographic location and other related data, such as movement speed. The Qt Positioning module allows you to access this information. Let's see a basic QML example of determining the location:

```cpp
import QtQuick 2.9
import QtQuick.Window 2.2
import QtPositioning 5.0
Window {
    visible: true
    width: 640
    height: 480
    title: qsTr("Hello World")
    Text {
        anchors.centerIn: parent
        text: {
            var pos = positionSource.position;
            var coordinate = pos.coordinate;
            return "latitude: " + coordinate.latitude +
              "\nlongitude: " + coordinate.longitude;
        }
    }
    PositionSource {
        id: positionSource
        active: true
        onPositionChanged: {
            console.log("pos changed",
                        position.coordinate.latitude,
                        position.coordinate.longitude);
        }
    }
}
```

First, we import the `QtPositioning` module into scope. Next, we create a `PositionSource` object and set its `active` property to true. The `position` property of the `PositionSource` object will be automatically updated as new information is available. In addition to latitude and longitude, this property also contains information about altitude, direction and speed of travel, and accuracy of location. Since some of values may not be available, each value is accompanied with a Boolean property that indicates if the data is present. For example, if `directionValid` is true, then `direction` value was set.

There are multiple ways to determine the player's location. The `PositionSource` type has a few properties that allow you to specify the source of the data. First, the `preferredPositioningMethods`  property allows you to choose between satellite data, non-satellite data, or using both of them. The `supportedPositioningMethods` property holds information about currently available methods. You can also use the `nmeaSource` property to provide an NMEA position-specification data file which overrides any other data sources and can be used to simulate the device's location and movement which is very useful during development and testing of the game. 

# Creating advanced QML components

By now, you should be familiar with the very basics of QML and Qt Quick. Now, we can start combining what you know and fill the gaps with more information to build more advanced QML components. Our target will be to display an analog clock.

# Time for action – A simple analog clock application

Create a new Qt Quick Application - Empty project. To create a clock, we will implement a component representing the clock needle, and we will use instances of that component in the actual clock element. In addition to this, we will make the clock a reusable component; therefore, we will create it in a separate file and instantiate it from within `main.qml`:

```cpp
Window {
    visible: true
    width: 640
    height: 480
    title: qsTr("Hello World")

    Clock {
        id: clock
        anchors {
            fill: parent
            margins: 20
        }
    }
}
```

We use the `anchors` property group to expand the item to fit the whole window except for the 20-pixel margin for all four sides.

Before this code works, however, we need to add a new QML file for the `Clock` component. Locate the `qml.qrc` resource file in the project tree and select Add New... in its context menu. From the Qt category, select QML File (Qt Quick 2), input `Clock` as the name and confirm the operation. A new file called `Clock.qml` will be created and added to the resources list.

Let's start by declaring a circular clock plate:

```cpp
import QtQuick 2.9

Item {
    id: clock

    property color color: "lightgray"

    Rectangle {
        id: plate

        anchors.centerIn: parent
        width: Math.min(clock.width, clock.height)
        height: width
        radius: width / 2
        color: clock.color
        border.color: Qt.darker(color)
        border.width: 2
    }
} 
```

If you run the program now, you'll see a plain gray circle hardly resembling a clock plate:

![](img/9f396b5d-9020-41ce-a194-7a811e1c2543.png)

The next step is to add marks dividing the plate into 12 sections. We can do this by putting the following declaration inside the `plate` object:

```cpp
Repeater {
    model: 12

    Item {
        id: hourContainer

        property int hour: index
        height: plate.height / 2
        transformOrigin: Item.Bottom
        rotation: index * 30
        x: plate.width/2
        y: 0

        Rectangle {
            width: 2
            height: (hour % 3 == 0) ? plate.height * 0.1
                                    : plate.height * 0.05
            color: plate.border.color
            antialiasing: true
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.top: parent.top
            anchors.topMargin: 4
        }
    }
} 
```

Running the program should now give the following result, looking much more like a clock plate:

![](img/f9920416-1d71-4d85-a249-cca203a92d26.png)

# What just happened?

The code we just created introduces a couple of new features. Let's go through them one by one.

First of all, we used a new element called `Repeater`. It does exactly what its name says—it repeats items declared within it using a given model. For each entry in the model, it creates an instance of a component assigned to a property called `delegate` (the property name means that it contains an entity to which the caller delegates some responsibility, such as describing a component to be used as a stencil by the caller). `Item` declared in `Repeater` describes the delegate even though we cannot see it explicitly assigned to a property. This is because `delegate` is a default property of the `Repeater` type, which means anything unassigned to any property explicitly gets implicitly assigned to the default property of the type.

The `Item` type also has a default property called `data`. It holds a list of elements that gets automatically split into two "sublists"—the list of the item's children (which creates the hierarchy of `Item` instances in Qt Quick) and another list called resources, which contains all "child" elements that do not inherit from `Item`. You have direct access to all three lists, which means calling `children[2]` will return the third `Item` element declared in the item, and `data[5]` will return the sixth element declared in the `Item`, regardless of whether the given element is a visual item (that inherits `Item`) or not.

The model can be a number of things, but in our case, it is simply a number denoting how many times the delegate should be repeated. The component to be repeated is a transparent item containing a rectangle. The item has a property declared called `hour` that has the `index` variable bound to it. The latter is a property assigned by `Repeater` to each instance of the delegate component. The value it contains is the index of the instance in the `Repeater` object—since we have a model containing twelve elements, `index` will hold values within a range of `0` to `11`. The item can make use of the `index` property to customize instances created by `Repeater`. In this particular case, we use `index` to provide values for a `rotation` property and by multiplying the index by `30`, we get values starting from `0` for the first instance and ending at `330` for the last one.

The `rotation` property brings us to the second most important subject—item transformations. Each item can be transformed in a number of ways, including rotating the item and scaling it in two-dimensional space, as we already mentioned earlier. Another property called `transformOrigin` denotes the origin point around which scale and rotation are applied. By default, it points to `Item.Center`, which makes the item scale and rotate around its center, but we can change it to eight other values, such as `Item.TopLeft` for the top-left corner or `Item.Right` for the middle of the right edge of the item. In the code we crafted, we rotate each item clockwise around its bottom edge. Each item is positioned horizontally in the middle of the plate using the `plate.width / 2` expression and vertically at the top of the plate with the default width of `0` and the height of half the plate's height; thus, each item is a thin vertical line spanning from within the top to the center of the plate. Then, each item is rotated around the center of the plate (each item's bottom edge) by 30 degrees more than a previous item effectively laying items evenly on the plate.

Finally, each item has a gray `Rectangle` attached to the top edge (offset by `4`) and horizontally centered in the transparent parent. Transformations applied to an item influence the item's children similar to what we have seen in Graphics View; thus, the effective rotation of the rectangle follows that of its parent. The height of the rectangle depends on the value of `hour`, which maps to the index of the item in `Repeater`. Here, you cannot use `index` directly as it is only visible within the topmost item of the delegate. That's why we create a real property called `hour` that can be referenced from within the whole delegate item hierarchy.

If you want more control over item transformations, then we are happy to inform you that apart from rotation and scale properties, each item can be assigned an array of elements such as `Rotation`, `Scale`, and `Translate` to a property called `transform`, which are applied in order, one at a time. These types have properties for fine-grained control over the transformation. For instance, using `Rotation`, you can implement rotation over any of the three axes and around a custom origin point (instead of being limited to nine predefined origin points as when using the `rotation` property of `Item`).

# Time for action – Adding needles to the clock

The next step is to add the hour, minute, and second needles to the clock. Let's start by creating a new component called `Needle` in a file called `Needle.qml` (remember that component names and files representing them need to start with a capital letter):

```cpp
import QtQuick 2.9

Rectangle {
    id: root

    property int value: 0
    property int granularity: 60
    property alias length: root.height

    width: 2
    height: parent.height / 2
    radius: width / 2
    antialiasing: true
    anchors.bottom: parent.verticalCenter
    anchors.horizontalCenter: parent.horizontalCenter
    transformOrigin: Item.Bottom
    rotation: 360 / granularity * (value % granularity)
} 
```

`Needle` is basically a rectangle anchored to the center of its parent by its bottom edge, which is also the item's pivot. It also has the `value` and `granularity` properties driving the rotation of the item, where `value` is the current value the needle shows and `granularity` is the number of different values it can display. Also, anti-aliasing for the needle is enabled as we want the tip of the needle nicely rounded. Having such a definition, we can use the component to declare the three needles inside the clock plate object:

```cpp
Needle {
    length: plate.height * 0.3
    color: "blue"
    value: clock.hours
    granularity: 12
}
Needle {
    length: plate.height * 0.4
    color: "darkgreen"
    value: clock.minutes
    granularity: 60
}
Needle {
    width: 1
    length: plate.height * 0.45
    color: "red"
    value: clock.seconds
    granularity: 60
} 
```

The three needles make use of the `hours`, `minutes`, and `seconds` properties of clock, so these need to be declared as well:

```cpp
property int hours: 0
property int minutes: 0
property int seconds: 0 
```

By assigning different values to the properties of `Clock` in `main.qml`, you can make the clock show a different time:

```cpp
import QtQuick 2.9

Clock {
    //...
    hours: 7
    minutes: 42
    seconds: 17
} 
```

You'll get an output as shown:

![](img/349cc52a-3655-4afb-93e9-37089402cf32.png)

# What just happened?

Most `Needle` functionality is declared in the component itself, including geometry and transformations. Then, whenever we want to use the component, we declare an instance of `Needle` and optionally customize the `length` and `color` properties as well as set its `value` and `granularity` to obtain the exact functionality we need.

# Time for action – Making the clock functional

The final step in creating a clock is to make it actually show the current time. In JavaScript, we can query the current time using the `Date` object:

```cpp
var currentDate = new Date();
var hours   = currentDate.getHours();
var minutes = currentDate.getMinutes();
var seconds = currentDate.getSeconds();
```

Therefore, the first thing that comes to mind is to use the preceding code to show the current time on the clock:

```cpp
Item {
    id: clock
    property int hours:   currentDate.getHours()
    property int minutes: currentDate.getMinutes()
    property int seconds: currentDate.getSeconds()
    property date currentDate: new Date()
    // ...
} 
```

This will indeed show the current time once you start the application, but the clock will not be updating itself as the time passes. This is because `new Date()` returns an object representing one particular moment in time (the date and time at the moment when the object was instantiated). While QML usually is capable of automatically updating a property when the value of the bound expression changes, it's unable to do so in this case. Even if QML was smart enough to see that the `new Date()` property always returns a different date, it doesn't know how often we want to update the value, and updating it as frequently as possible is generally a bad idea. Thus, we need a way to manually schedule periodic execution of an action.

To obtain this effect in QML, we can use a `Timer` element that is an equivalent of `QTimer` in C++ and lets us periodically execute some code. Let's modify the code to use a timer:

```cpp
Item {
    id: clock
    //...
    property alias running: timer.running
    Timer {
        id: timer
        repeat: true
        interval: 500
        running: true
        onTriggered: clock.currentDate = new Date()
    }
    //...
} 
```

# What just happened?

By setting the `interval` property, we ask the timer to emit the `triggered` signal every 500 ms, causing our `currentDate` property to be updated with a new `Date`  object representing the current time. The clock is also given the `running` property (pointing to its equivalent in the timer) that can control whether updates should be enabled. The timer's `repeat`
property is set to `true`; otherwise, it will trigger just once.

To briefly sum up what you have learned so far, we can say that you know how to create hierarchies of objects by declaring their instances, and you also know how to program new types in separate files, making definitions available as components to be instantiated in other QML files. You can even use the `Repeater` element to declare a series of objects based on a common stencil.

# Dynamic and lazy loading of QML objects

All our previous QML projects contain an explicit declaration of the object tree. We usually create a window and place multiple specific elements into it in specific order. The QML engine creates these objects on startup and keeps them alive until the application terminates. This is a very convenient approach that allows you to save a lot of time, as you could see in our previous examples. However, sometimes you need the object tree to be more flexible—for example, if you don't know upfront which elements should be created. QML offers a few ways to create objects dynamically and to delay creating an object until you really need it. Utilizing these features can make your QML application more performant and flexible.

# Creating objects on request

The problem with predeclaring objects directly in a QML file is that you need to know upfront how many objects you will need. More often than not, you will want to dynamically add and remove objects to your scene, for example, in an alien invasion game, where, as the player progresses, new alien saucers will be entering the game screen and other saucers will be getting shot down and destroyed. Also, the player's ship will be "producing" new bullets streaking in front of the ship, eventually running out of fuel or otherwise disappearing from the game scene. By putting a good amount of effort into the problem, you will be able to use `Repeater` to obtain this effect, but there is a better tool at hand.

QML offers us another element type called `Component`, which is another way to teach the engine about a new element type by declaring its contents in QML. There are basically two approaches to doing this.

The first approach is to declare a `Component` element instance in the QML file and inline the definition of the new type directly inside the element:

```cpp
Component {
    id: circleComponent
    Item {
        //...
    }
} 
```

The other approach is to load the component definition from an existing QML file. Let's say that we have a `Circle.qml` file with the following content:

```cpp
import QtQuick 2.9
Item {
    property int diameter: 20
    property alias color: rect.color
    property alias border: rect.border

    implicitWidth: diameter
    implicitHeight: diameter

    Rectangle {
        id: rect
        width: radius
        height: radius
        radius: diameter / 2
        anchors.centerIn: parent
    }
}
```

Such code declares a component that defines a circle and exposes its `diameter`, `color`, and `border` properties. Let's see how we can create instances of this component dynamically.

QML exposes a special global object called `Qt`, which provides a set of interesting methods. One of the methods allows the caller to create a component passing the URL of an existing QML document:

```cpp
var circleComponent = Qt.createComponent("Circle.qml"); 
```

An interesting note is that `createComponent` can not only accept a local file path but also a remote URL, and if it understands the network scheme (for example, `http`), it will download the document automatically. In this case, you have to remember that it takes time to do that, so the component may not be ready immediately after calling  `createComponent`. Since the current loading status is kept in the `status` property, you can connect to the `statusChanged` signal to be notified when this happens. A typical code path looks similar to the following:

```cpp
Window {
    //...
    Component.onCompleted: {
        var circleComponent = Qt.createComponent("Circle.qml");
 if(circleComponent.status === Component.Ready) {
 addCircles(circleComponent);
 } else {
 circleComponent.statusChanged.connect(function() {
 if(circleComponent.status === Component.Ready) {
 addCircles(circleComponent);
 }
 });
 }
    }
}
```

In this example, we use the `Component.onCompleted` handler to run the code as soon as the window object is created. This handler is available in all Qt Quick items and is often used to perform initialization. You can also use any other signal handler here. For example, you can start loading the component when a button is pressed or a timer has expired.

The counterpart of the `completed()` signal of `Component` is `destruction()`. You can use the `Component.onDestruction` handler to perform actions such as saving the state of the object to persistent storage or otherwise cleaning the object up.

If the component definition is incorrect or the document cannot be retrieved, the status of the object will change to `Error`. In that case, you can make use of the `errorString()` method to see what the actual problem is:

```cpp
if(circleComponent.status === Component.Error) {
    console.warn(circleComponent.errorString());
} 
```

Once you are sure that the component is ready, you can finally start creating objects from it. For this, the component exposes a method called `createObject`. In its simplest form, it accepts an object that is to become the parent of the newly born instance (similar to widget constructors accepting a pointer to a parent widget) and returns the new object itself so that you can assign it to some variable. Then, you can start setting the object's properties:

```cpp
Window {
    //...
    ColumnLayout {
        id: layout
        anchors.fill: parent
    }
    function addCircles(circleComponent) {
        ["red", "yellow", "green"].forEach(function(color) {
 var circle = circleComponent.createObject(layout);
 circle.color = color;
 circle.Layout.alignment = Qt.AlignCenter;
        });
    }
    //...
}
```

A more complex invocation lets us do both these operations (create the object and set its properties) in a single call by passing a second parameter to `createObject`:

```cpp
var circle = circleComponent.createObject(layout,
    { diameter: 20, color: 'red' });
```

The second parameter is a JavaScript object whose properties are to be applied to the object being created. The advantage of the latter syntax is that all property values are applied to the object as one atomic operation and they won't trigger property change handlers (just like when the item is declared in a QML document) instead of a series of separate operations, each of which sets the value for a single property, possibly causing an avalanche of change handler invocations in the object.

After creation, the object becomes a first-class citizen of the scene, acting in the same way as items declared directly in the QML document. The only difference is that a dynamically created object can also be dynamically destructed by calling its `destroy()` method, which is similar to calling `delete` on C++ objects. When speaking of destroying dynamic items, we have to point out that when you assign a result of `createObject` to a variable (like `circle`, in our example) and that variable goes out of scope, the item will not be released and garbage collected as its parent still holds a reference to it, preventing it from being recycled.

We didn't mention this explicitly before, but we have already used inline component definitions earlier in this chapter when we introduced the `Repeater` element. The repeated item defined within the repeater is in fact not a real item, but a component definition that is instantiated as many times as needed by the repeater.

# Delaying item creation

Another recurring scenario is that you do know how many elements you will need, but the problem is that you cannot determine upfront what type of elements they will be. At some point during the lifetime of your application, you will learn that information and will be able to instantiate an object. Until you gain the knowledge about the given component, you will need some kind of item placeholder where you will later put the real item. You can, of course, write some code to use the `createObject()` functionality of the component, but this is cumbersome.

Fortunately, Qt Quick offers a nicer solution in the form of a `Loader` item. This item type is exactly what we described it to be—a temporary placeholder for a real item that will be loaded on demand from an existing component. You can put `Loader` in place of another item and when you need to create this item, one way is to set the URL of a component to the `source` property:

```cpp
Loader {
    id: loader
}
//...
onSomeSignal: loader.source = "Circle.qml"
```

You can also directly attach a real component to the `sourceComponent` of a `Loader`:

```cpp
Loader {
    id: loader
    sourceComponent: shouldBeLoaded ? circleComponent : undefined
} 
```

Immediately afterward, the magic begins and an instance of the component appears in the loader. If the `Loader` object has its size set explicitly (for example, by anchoring or setting the width and height), then the item will be resized to the size of the loader. If an explicit size is not set, then `Loader` will instead be resized to the size of the loaded element once the component is instantiated. In the following code, the loader has its size set explicitly, so when its item is created, it will respect the anchors and sizes declared here:

```cpp
Loader {
    anchors {
        left: parent.left
        leftMargin: 0.2 * parent.width
        right: parent.right
        verticalCenter: parent.verticalCenter
    }
    height: 250
    source: "Circle.qml"
} 
```

# Imperative painting on Canvas using JavaScript

Declaring graphical items is nice and easy, but as programmers, we're more used to writing imperative code, and some things are easier expressed as an algorithm rather than as a description of the final result to be achieved. It is easy to use QML to encode a definition of a primitive shape such as a rectangle in a compact way—all we need is to mark the origin point of the rectangle, its width, height, and optionally, a color. Writing down a declarative definition of a complex shape consisting of many control points positioned in given absolute coordinates, possibly with an outline in some parts of it, maybe accompanied by an image or two, is still possible in a language such as QML; however, this will result in a much more verbose and much less readable definition. This is a case where using an imperative approach might prove more effective. HTML (being a declarative language) already exposes a proven imperative interface for drawing different primitives called a `Canvas` that is widely used in web applications. Fortunately, Qt Quick provides us with its own implementation of a `Canvas` interface similar to the one from the web by letting us instantiate `Canvas` items. Such items can be used to draw straight and curved lines, simple and complex shapes, and graphs and graphic images. It can also add text, colors, shadows, gradients, and patterns. It can even perform low-level pixel operations. Finally, the output may be saved as an image file or serialized to a URL usable as source for an `Image` item. There are many tutorials and papers available out there on using an HTML canvas, and they can usually be easily applied to a Qt Quick canvas as well (the reference manual even includes a list of aspects you need to pay attention to when porting HTML canvas applications to a Qt Quick Canvas), so here we will just give you the very basics of imperative drawing in Qt Quick.

Consider a game where the player's health is measured by the condition of his heart—the slower the beat, the healthier the player is. We will use this kind of visualization as our exercise in practicing painting using the `Canvas` element.

# Time for action – Preparing Canvas for heartbeat visualization

Let's start with simple things by creating an empty Qt Quick project. Add the following code to the `main.qml` file:

```cpp
Window {
    //...
    Canvas {
        id: canvas

        implicitWidth: 600
        implicitHeight: 300

        onPaint: {
            var ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.strokeRect(50, 50, 100, 100);
        }
    }
}
```

When you run the project, you will see a window containing a rectangle:

![](img/0b1553a2-ae63-4fcd-ac97-abd0e997ff27.png)

# What just happened?

In the preceding code, we created a basic boilerplate code for using a canvas. First, we created a `Canvas` instance with an implicit width and height set. There, we created a handler for the `paint` signal that is emitted whenever the canvas needs to be redrawn. The code placed there retrieves a context for the canvas, which can be thought of as an equivalent to the `QPainter` instance we used when drawing on Qt widgets. We inform the canvas that we want its 2D context, which gives us a way to draw in two dimensions. A 2D context is the only context currently present for the `Canvas` element, but you still have to identify it explicitly—similar to HTML. Having the context ready, we tell it to clear the whole area of the canvas. This is different from the widget world in which when the `paintEvent` handler was called, the widget was already cleared for us and everything had to be redrawn from scratch. With `Canvas`, it is different; the previous content is kept by default so that you can draw over it if you want. Since we want to start with a clean sheet, we call `clearRect()` on the context. Finally, we use the `strokeRect()` convenience method that draws a rectangle on the canvas.

# Time for action - drawing a heartbeat

We will extend our component now and implement its main functionality—drawing a heartbeat-like diagram.

Add the following property declarations to the `canvas` object:

```cpp
property int lineWidth: 2
property var points: []
property real arg: -Math.PI 
```

Inside the `Canvas` section, add a declaration for a timer that will trigger updates of the picture:

```cpp
Timer {
    interval: 10
    repeat: true
    running: true
    onTriggered: {
        canvas.arg += Math.PI / 180;
        while(canvas.arg >= Math.PI) {
            canvas.arg -= 2 * Math.PI;
        }
    }
}
```

Then again, inside the `Canvas` section, define the handler for when the value of `arg` is modified:

```cpp
onArgChanged: {
    points.push(func(arg));
    points = points.slice(-canvas.width);
    canvas.requestPaint();
}
```

This handler uses a custom JavaScript function—`func()`. Place the implementation of the function just above the handler:

```cpp
function func(argument) {
    var a = (2 * Math.PI / 10);
    var b = 4 * Math.PI / 5;
    return Math.sin(20 * argument) * (
        Math.exp(-Math.pow(argument / a, 2)) +
        Math.exp(-Math.pow((argument - b) / a, 2)) +
        Math.exp(-Math.pow((argument + b) / a, 2))
    );
}
```

Finally, modify the `onPaint` signal handler:

```cpp
onPaint: {
    var ctx = canvas.getContext("2d");
    ctx.reset();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    var pointsToDraw = points.slice(-canvas.width);
    ctx.translate(0, canvas.height / 2);
    ctx.beginPath();
    ctx.moveTo(0, -pointsToDraw[0] * canvas.height / 2);
    for(var i = 1; i < pointsToDraw.length; i++) {
        ctx.lineTo(i, -pointsToDraw[i] * canvas.height / 2);
    }
    ctx.lineWidth = canvas.lineWidth;
    ctx.stroke();
}
```

Then, you can run the code and see a heartbeat-like diagram appear on the canvas:

![](img/e6e910ae-4a56-41bf-bad2-d87807ec3e5e.png)

# What just happened?

We added two kinds of properties to the element. By introducing `lineWidth`, we can manipulate the width of the line that visualizes the heartbeat. The `points` variable stores an array of already calculated function values. We initialize it to an empty array. The `arg` variable stores the function argument that was last evaluated. The argument of the function should be in the range from −π to +π; thus, we initialize `arg` to `-Math.PI`. Then, we add a timer that ticks in regular intervals, incrementing `arg` by 1° until it reaches +π, in which case it is reset to the initial value.

Changes to `arg` are intercepted in the handler we implement next. In there, we push a new item to the array of points. The value is calculated by the `func` function, which is quite complicated, but it is sufficient to say that it returns a value from within a range of −1 to +1\. The oldest records are removed from the array of points using `Array.slice()` so that at most, the last `canvas.width` items remain in the array. This is so that we can plot one point for each pixel of the width of the canvas, and we don't have to store any more data than required. At the end of the function, we invoke `requestPaint()`, which is an equivalent of `QWidget::update()` and schedules a repaint of the canvas.

That, in turn, calls our `onPaint` signal handler. There, after retrieving the context, we reset the canvas to its initial state and then calculate an array of points that is to be drawn again using `slice()`. Then, we prepare the canvas by translating and scaling it in the vertical axis so that the origin is moved to half of the height of the canvas (that's the reason for calling `reset()` at the beginning of the procedure—to revert this transformation). After that, `beginPath()` is called to inform the context that we are starting to build a new path. Then, the path is built segment by segment by appending lines. Each value is multiplied by `canvas.height / 2` so that values from the point array are scaled to the size of the item. The value is negated as the vertical axis of the canvas grows to the bottom, and we want positive values to be above the origin line. After that, we set the width of the pen and draw the path by calling `stroke()`.

# Time for action – Hiding properties

If we convert our heartbeat canvas to a QML component, the `points` and `arg` properties will be the public properties visible to the user of the component. However, they are really implementation details we want to hide. We should only expose properties that make sense to the user of the component, such as `lineWidth` or `color`.

Since the `Timer` object inside the `Canvas` is not exported as public property, that timer object will be unavailable from the outside, so we can attach properties to the timer instead of attaching them to the top-level `Canvas` object. However, the properties do not belong to the timer logically, so this solution will be confusing. For such cases, there is a convention that you should create an empty `QtObject` child in the top-level object and move properties into it:

```cpp
Canvas {
    id: canvas
    property int lineWidth: 2
    //...
    QtObject {
        id: d
        property var points: []
        property real arg: -Math.PI

        function func(argument) { /* ... */ }
        onArgChanged: { /* ... */ }
    }
    //...
}
```

`QtObject` is the QML representation of the `QObject` class. It is a QML type that doesn't have any particular functionality, but can hold properties. As part of the convention, we set `id` of this object to `d`. The `onArgChanged` handler is moved to the private object as well. In the `onTriggered` and `onPaint` handlers, we should now refer to the internal properties as `d.points` and `d.arg`. Consider this example:

```cpp
onTriggered: {
    d.arg += Math.PI / 180;
    while(d.arg >= Math.PI) {
        d.arg -= 2 * Math.PI;
    }
}
```

The `points` and `arg` properties are now unavailable from the outside, leading to clean public interface of our heartbeat object.

# Time for action – Making the diagram more colorful

The diagram serves its purpose, but it looks a bit dull. Add some shine to it by defining three new color properties in the canvas object—`color`, `topColor`, `bottomColor`—and setting their default values to `black`, `red`, and `blue`, respectively:

```cpp
property color color: "black"
property color topColor: "red"
property color bottomColor: "blue"
```

Then, let's make use of these properties by extending `onPaint` implementation:

```cpp
onPaint: {
    //...
    // fill:
    ctx.beginPath();
    ctx.moveTo(0, 0);
    var i;
    for(i = 0; i < pointsToDraw.length; i++) {
        ctx.lineTo(i, -pointsToDraw[i] * canvas.height/2);
    }
    ctx.lineTo(i, 0);
    var gradient = ctx.createLinearGradient(
            0, -canvas.height / 2, 0, canvas.height / 2);
    gradient.addColorStop(0.1, canvas.topColor);
    gradient.addColorStop(0.5, Qt.rgba(1, 1, 1, 0));
    gradient.addColorStop(0.9, canvas.bottomColor);
    ctx.fillStyle = gradient;
    ctx.fill();

    // stroke:
    ctx.beginPath();
    ctx.moveTo(0, -pointsToDraw[0] * canvas.height / 2);
    for(i = 1; i < pointsToDraw.length; i++) {
        ctx.lineTo(i, -pointsToDraw[i] * canvas.height / 2);
    }
    ctx.lineWidth = canvas.lineWidth;
    ctx.strokeStyle = canvas.color;
    ctx.stroke();
}
```

Upon running the preceding code snippet, you get the following output:

![](img/a9ce7759-007e-440c-9c3c-a6df4b2b1b81.png)

# What just happened?

The modifications to `onPaint` that we implemented are creating another path and using that path to fill an area using a gradient. The path is very similar to the original one, but it contains two additional points that are the first and last points drawn projected onto the horizontal axis. This ensures that the gradient fills the area properly. Note that the canvas uses imperative code for drawing; therefore, the order of drawing the fill and the stroke matters—the fill has to be drawn first so that it doesn't obscure the stroke.

# Using C++ classes as QML components

In the next exercise, we will implement a car dashboard that can be used in a racing game and will show a number of parameters such as current speed and motor revolutions per minute. The input data will be provided by a C++ object. We'll see how to include this object into the QML object tree and use property bindings to implement the dashboard.

The final result will look similar to the following:

![](img/a197c586-48c3-456d-b365-297c88d58738.png)

# Time for action – Self-updating car dashboard

We will start with the C++ part. Set up a new Qt Quick application. This will generate the main function for you that instantiates `QGuiApplication`   and  `QQmlApplicationEngine` and sets them up to load a QML document.

Use the File menu to create New file or Project and create a new Qt Designer form class. Call it `CarInfo` and choose the `Widget` template. We will use this class for modifying values of different parameters so that we may observe how they influence what the Qt Quick scene displays. In the class declaration, add the following properties:

```cpp
class CarInfo : public QWidget {
    Q_OBJECT
    Q_PROPERTY(int rpm READ rpm NOTIFY rpmChanged)
    Q_PROPERTY(int gear READ gear NOTIFY gearChanged)
    Q_PROPERTY(int speed READ speed NOTIFY speedChanged)
    Q_PROPERTY(double distance READ distance NOTIFY distanceChanged)
    //...
};
```

The properties are read-only, and the `NOTIFY` clause defines signals emitted when the respective property values change. Go ahead and implement the appropriate functions for each property. Apart from the getter, also implement a setter as a public slot. Here's an example for a property controlling the speed of the car:

```cpp
int CarInfo::speed() const {
    return m_speed;
}
void CarInfo::setSpeed(int newSpeed) {
    if(m_speed == newSpeed) {
        return;
    }
    m_speed = newSpeed;
    emit speedChanged(m_speed);
} 
```

You should be able to follow the example for the remaining properties on your own.

Since we want to use the widget to tweak property values, design the user interface for it using the form editor. It can look like this:

![](img/b5191e3b-b1a6-41fd-9a29-5e297308d7c9.png)

Make appropriate signal-slot connections in the widget so that modifying any of the widgets for a given parameter or using the setter slot directly updates all the widgets for that parameter.

Instead of adding member variables to the `CarInfo` class for properties such as `speed`, `rpm`, `distance`, or `gear`, you can operate directly on the widgets placed on the `ui` form, as shown further.

For example, a getter for the `distance` property will look like this:

```cpp
qreal CarInfo::distance() const
{
    return ui->distanceBox->value();
}
```

The setter would then be modified to the following:

```cpp
void CarInfo::setDistance(qreal newDistance)
{
    ui->distanceBox->setValue(newDistance);
}
```

You will then need to add `connect()` statements to the constructor to ensure that signals are propagated from the ui form:

```cpp
connect(ui->distanceBox, SIGNAL(valueChanged(double)),
        this,            SIGNAL(distanceChanged(double)));   
```

Next, you can test your work by running the widget. To do this, you have to alter the main function to look as follows:

```cpp
int main(int argc, char **argv) {
    QApplication app(argc, argv);
    CarInfo cinfo;
    cinfo.show();
    return app.exec();
}; 
```

Since we are using widgets, we have to replace `QGuiApplication` with  `QApplication`  and enable the widgets module by placing `QT += widgets` in the project file (remember to run `qmake` from the project's context menu afterward). Ensure that everything works as expected (that is, that moving sliders and changing spinbox values reflect the changes to widget properties) before moving on to the next step.

We will now add Qt Quick to the equation, so let's start by updating our main function to display our scene. Introduce the highlighted changes to the code:

```cpp
int main(int argc, char **argv) {
    QApplication app(argc, argv);
    CarInfo cinfo;
    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("carData", &cinfo);
    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;
    cinfo.show();
    return app.exec();
}
```

The modifications create a QML engine for our scene, export the `CarInfo` instance to the global context of the QML engine, and load and display the scene from a file located in a resource.

It is important to first export all the objects and only then load the scene. This is because we want all the names to be already resolvable when the scene is being initialized so that they can be used right away. If we reversed the order of calls, we would get a number of warnings on the console about the identities being undefined.

Finally, we can focus on the QML part. Look at the picture of the result we want to be shown at the beginning of the exercise. For the black background, we used a bitmap image created in a graphical editor (you can find the file in the materials for this book), but you can obtain a similar effect by composing three black rounded rectangles directly in Qt Quick—the two outer parts are perfect circles, and the inner module is a horizontally stretched ellipse.

If you decide to use our background file (or make your own prettier image), you should add it to the project's resources and put the following code into `main.qml`:

```cpp
import QtQuick 2.9
import QtQuick.Window 2.3

Window {
    visible: true
    width: backgroundImage.width
    height: backgroundImage.height

    Image {
        id: backgroundImage
        source: "qrc:/dashboard.png"
        Item {
            id: leftContainer
            anchors.centerIn: parent
            anchors.horizontalCenterOffset: -550
            width: 400; height: width
        }
        Item {
            id: middleContainer
            anchors.centerIn: parent
            width: 700; height: width
        }
        Item {
            id: rightContainer
            anchors.centerIn: parent
            anchors.horizontalCenterOffset: 525
            width: 400; height: width
        }
    }
}
```

What we do here is add the image to the window and create three items to serve as containers for different elements of the dashboard. The containers are all centered in the parent, and we use a `horizontalCenterOffset` property to move the two outer items sideways. The values of the offset as well as the widths are based on the background image's geometry (note that all three containers are perfect squares). If instead of using our file, you settle for creating the three parts yourself using Qt Quick items, the containers will simply be anchored to the centers of the three black items.

The dials look complicated, but in reality, they are very easy to implement, and you have already learned everything you need to design them.

Let's start with the needle. Use the context menu of the resource file to create a new QML file and call it `Needle.qml`. Open the file and place the following content:

```cpp
import QtQuick 2.9

Item {
    id: root
    property int length: parent.width * 0.4
    property color color: "white"
    property color middleColor: "red"
    property int size: 2

    Rectangle { //needle
        width: root.size
        height: length + 20
        color: root.color
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: -20
        antialiasing: true
    }

    Rectangle { //fixing
        anchors.centerIn: parent
        width: 8 + root.size
        height: width
        radius: width / 2
        color: root.color
        Rectangle { //middle dot
            anchors {
                fill: parent
                margins: parent.width * 0.25
            }
            color: root.middleColor
        }
    }
}
```

The document defines an item with four attributes—the length of the needle (defaults to 80% of the dial's radius), the color of the needle, `middleColor`, which stands for the color of the needle's fixing, and the size, which defines how wide the needle is. The code is self-explanatory. The item itself does not have any dimensions and only acts as an anchor for visual elements—the needle itself is a thin rectangle oriented vertically with a fixing 20 units from the end. The fixing is a circle of the same color as the needle with a smaller circle in the middle that uses a different fill color. The smaller radius of the inner circle is obtained by filling the outer circle with a 25% margin from each side.

As for the dials, we will put their code inline in the main file since we just have two of them and they differ a bit, so the overhead of creating a separate component with a well-designed set of properties will outweigh the benefits of having nicely encapsulated objects.

If you think about what needs to be done to have the dial displayed and working, it seems that the hardest thing is to lay out the numbers nicely on the circle, so let's start with that. Here's an implementation of a function for calculating the position along a circle circumference, based on the radius of the circle and angle (in degrees) where an item should be positioned:

```cpp
function calculatePosition(angle, radius) {
    if(radius === undefined) {
        radius = width / 2 * 0.8;
    }
    var a = angle * Math.PI / 180;
    var px = width / 2 + radius * Math.cos(a);
    var py = width / 2 + radius * Math.sin(a);
    return Qt.point(px, py);
}
```

The function converts degrees to radians and returns the desired point. The function expects the `width` property to be available that helps calculate the center of the circle and in case a radius was not given, serves as a means to calculate a feasible value for it.

With such a function available, we can use the already familiar `Repeater` element to position items where we want them. Let's put the function in `middleContainer` and declare the dial for car speed:

```cpp
Item {
    id: middleContainer
    // ...
    function calculatePosition(angle, radius) { /* ... */ }
    Repeater {
        model: 24 / 2
        Item {
            property point pt:
            middleContainer.calculatePosition(120 + index * 12 * 2)
            x: pt.x
            y: pt.y
            Label {
                anchors.centerIn: parent
                text: index * 20
            }
        }
    }
    Needle {
        anchors.centerIn: parent
        length: parent.width * 0.35
        size: 4
        rotation: 210 + (carData.speed * 12 / 10)
        color: "yellow"
    }
} 
```

You might have noted that we used an element called `Label`. We created it to avoid having to set the same property values for all the texts we use in the user interface:

```cpp
import QtQuick 2.9

Text {
    color: "white"
    font.pixelSize: 24
} 
```

The dial consists of a repeater that will create 12 elements. Each element is an item positioned using the earlier described function. The item has a label anchored to it that displays the given speed. We use `120 + index * 12 * 2` as the angle expression as we want "0" to be positioned at 120 degrees and each following item positioned 24 degrees further.

The needle is given rotation based on the value read from the `carData` object. Since the angular distance between consecutive 20 kph labels is 24 degrees, the distance for one kph is 1.2 and thus we multiply `carData.speed` by that factor. Item rotation is calculated with 0 degrees "pointing right"; therefore, we add 90 to the initial 120 degree offset of the first label to obtain starting coordinates matching those of the label system.

As you can see in the image of the final result at the beginning of this section, the speed dial contains small lines every 2 kph, with those divisible by 10 kph longer than others. We can use another `Repeater` to declare such ticks:

```cpp
Repeater {
    model: 120 - 4

    Item {
        property point pt: middleContainer.calculatePosition(
            120 + index * 1.2 * 2, middleContainer.width * 0.35
        )
        x: pt.x
        y: pt.y
        Rectangle {
            width: 2
            height: index % 5 ? 5 : 10
            color: "white"
            rotation: 210 + index * 1.2 * 2
            anchors.centerIn: parent
            antialiasing: true
        }
    }
} 
```

Finally, we can put a label for the dial:

```cpp
Text {
    anchors.centerIn: parent
    anchors.verticalCenterOffset: 40
    text: "SPEED\n[kph]"
    horizontalAlignment: Text.AlignHCenter
    color: "#aaa"
    font.pixelSize: 16
} 
```

Ensure that the label is declared before the dial needle, or give the needle a higher *z* value so that the label doesn't overpaint the needle.

Next, repeat the process on your own for the left container by creating an RPM dial reading values from the `carData.rpm` property. The dial also displays the current gear of the car's engine. Place the following code inside the `leftContainer` object definition:

```cpp
Item {
    id: gearContainer
    anchors.centerIn: parent
    anchors.horizontalCenterOffset: 10
    anchors.verticalCenterOffset: -10

    Text {
        id: gear
        property int value: carData.gear
        property var gears: [
 "R", "N",
 "1<sup>st</sup>", "2<sup>nd</sup>", "3<sup>rd</sup>",
 "4<sup>th</sup>", "5<sup>th</sup>"
 ]
 text: gears[value + 1]
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        color: "yellow"
        font.pixelSize: 32
        textFormat: Text.RichText
    }
} 
```

The only part needing explanation is highlighted. It defines an array of gear labels starting with reverse, going through neutral, and then through five forward gears. The array is then indexed with the current gear and the text for that value is applied to the label. Note that the value is incremented by 1, which means the 0th index of the array will be used when `carData.gear` is set to `1`.

We will not show how to implement the right container. You can do that easily yourself now with the use of the `Grid` positioner to lay out the labels and their values. To display the series of controls on the bottom of the right container (with texts `ABS`, `ESP`, `BRK`, and `CHECK`), you can use `Row` of `Label` instances.

Now, start the program and begin moving the sliders on the widget. See how the Qt Quick scene follows the changes.

# What just happened?

We have created a very simple `QObject` instance and exposed it as our "data model" to QML. The object has a number of properties that can receive different values. Changing a value results in emitting a signal, which in turn notifies the QML engine and causes bindings containing those properties to be reevaluated. As a result, our user interface gets updated.

# Time for action – Grouping engine properties

The data interface between the QML and C++ worlds that we created is very simple and has a small number of properties. However, as the amount of data we want to expose grows, the object can become cluttered. Of course, we can counter that effect by dividing it into multiple smaller objects, each having separate responsibilities and then exporting all those objects to QML, but that is not always desirable. In our case, we can see that `rpm` and `gear` are properties of the engine subsystem, so we can move them to a separate object; however, in reality, their values are tightly coupled with the speed of the car and to calculate the speed, we will need to know the values of those two parameters. However, the speed also depends on other factors such as the slope of the road, so putting the speed into the engine subsystem object just doesn't seem right. Fortunately, there is a nice solution to that problem.

QML has a concept called grouped properties. You already know a number of them—the `border` property of the `Rectangle` element or the `anchors` property of the `Item` element, for example. Let's see how to define such properties for our exposed object.

Create a new `QObject`-derived class and call it `CarInfoEngine`. Move the property definitions of `rpm` and `gear` to that new class along with their getters, setters, and change signals. Add the following property declaration to `CarInfo`:

```cpp
Q_PROPERTY(QObject* engine READ engine NOTIFY engineChanged) 
```

Implement the getter and the private field:

```cpp
    QObject* engine() const { return m_engine; }
private:
    CarInfoEngine *m_engine; 
```

We will not use the signal right now. However, we had to declare it; otherwise, QML would complain we were binding expressions that depend on properties that are non-notifiable:

```cpp
signals:
    void engineChanged(); 
```

Initialize `m_engine` in the constructor of `CarInfo`:

```cpp
m_engine = new CarInfoEngine(this); 
```

Next, update the code of `CarInfo` to modify properties of `m_engine` whenever respective sliders on the widget are moved. Provide a link the other way as well—if the property value is changed, update the user interface accordingly.

Update the QML document and replace `carData.gear` with `carData.engine.gear`. Do the same for `carData.rpm` and `carData.engine.rpm`. You should end up with something along the lines of the following:

```cpp
Item {
    id: leftContainer
    // ...

    Item {
        id: gearContainer
        Text {
            id: gear
            property int value: carData.engine.gear
            // ...
        }
    }
    Needle {
        anchors.centerIn: parent
        length: parent.width * 0.35
        rotation: 210 + (carData.engine.rpm * 35)
    }
} 
```

# What just happened?

Essentially, what we did is expose a property in `CarInfo` that is itself an object that exposes a set of properties. This object of the `CarInfoEngine` type is bound to the `CarInfo` instance it refers to.

# Time for action – Registering C++ class as QML type

So far, what we did was expose ourselves to QML single objects created and initialized in C++. However, we can do much more—the framework allows us to define new QML types. These can either be generic `QObject`-derived QML elements or items specialized for Qt Quick.

We will start with something simple—exposing the `CarInfo` type to QML so that instead of instantiating it in C++ and then exposing it in QML, we can directly declare the element in QML and still allow the changes made to the widget to be reflected in the scene.

To make a certain class (derived from `QObject`) instantiable in QML, all that is required is to register that class with the declarative engine using the `qmlRegisterType` template function. This function takes the class as its template parameter along a number of function arguments: the module `uri`, the major and minor version numbers, and the name of the QML type we are registering. The following call will register the `FooClass` class as the QML type `Foo`, available after importing `foo.bar.baz` in Version 1.0:

```cpp
qmlRegisterType<FooClass>("foo.bar.baz", 1, 0, "Foo"); 
```

You can place this invocation anywhere in your C++ code; just ensure that this is before you try to load a QML document that might contain declarations of `Foo` objects. A typical place to put the function call is in the program's main function. Afterward, you can start declaring objects of the `Foo` type in your documents. Just remember that you have to import the respective module first:

```cpp
import QtQuick 2.9
import foo.bar.baz 1.0

Item {
    Foo {
        id: foo
    }
} 
```

# Time for action – Making CarInfo instantiable from QML

First, we will update the QML document to create an instance of `CarInfo` present in the `CarInfo 1.0` module:

```cpp
import QtQuick 2.9
import CarInfo 1.0

Image {
    source: "dashboard.png"

    CarInfo {
        id: carData
        visible: true // make the widget visible
    }
  // ...
} 
```

As for registering `CarInfo`, it might be tempting to simply call `qmlRegisterType` on `CarInfo` and congratulate ourselves for a job well done:

```cpp
int main(int argc, char **argv) {
    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;
    // this code does not work
    qmlRegisterType<CarInfo>("CarInfo", 1, 0, "CarInfo");
    //...
} 
```

In general, this would work (yes, it is as simple as that). However, it will not work with widgets. It's not possible to include `QWidget`-based objects into a QML object tree because a `QWidget` object can only have another `QWidget` object as its parent, and QML needs to set the outer QML object as the parent. To resolve this conflict, we need to ensure that what we instantiate is not a widget. For that, we will use a proxy object that will forward our calls to the actual widget. Therefore, create a new class called `CarInfoProxy` derived from `QObject` and make it have the same properties as `CarInfo`. Consider this example:

```cpp
class CarInfoProxy : public QObject {
    Q_OBJECT
    Q_PROPERTY(QObject *engine READ engine NOTIFY engineChanged)
    Q_PROPERTY(int speed READ speed WRITE setSpeed NOTIFY speedChanged)
    // ... 
```

Declare one more property that will let us show and hide the widget on demand:

```cpp
  Q_PROPERTY(bool visible READ visible WRITE setVisible 
                          NOTIFY visibleChanged)
```

Then, we can place the widget as a member variable of the proxy so that it is created and destroyed alongside its proxy:

```cpp
private:
    CarInfo m_car; 
```

This way, the `CarInfo` widget will have `nullptr` parent, so it will be displayed as a top-level window. The QML engine will create an object of the `CarInfoProxy` class and set its parent to be another QML object, but this will not affect the parent of the widget.

Next, implement the missing interface. For simplicity, we are showing you code for some of the properties. The others are similar, so you can fill in the gaps on your own:

```cpp
public:
    CarInfoProxy(QObject *parent = nullptr) : QObject(parent) {
        connect(&m_car, &CarInfo::engineChanged,
                this, &CarInfoProxy::engineChanged);
        connect(&m_car, &CarInfo::speedChanged,
                this, &CarInfoProxy::speedChanged);
    }
    QObject *engine() const { 
        return m_car.engine(); 
    }
    bool visible() const { 
        return m_car.isVisible(); 
    }
    void setVisible(bool v) {
        if(v == visible()) return;
        m_car.setVisible(v);
        emit visibleChanged(v);
    }
    int speed() const { 
        return m_car.speed(); 
    }
    void setSpeed(int v) { 
        m_car.setSpeed(v); 
    }
signals:
    void engineChanged();
    void visibleChanged(bool);
    void speedChanged(int);
}; 
```

You can see that we reuse the `CarInfoEngine` instance from the widget instead of duplicating it in the proxy class. Finally, we can register `CarInfoProxy` as `CarInfo`:

```cpp
  qmlRegisterType<CarInfoProxy>("CarInfo", 1, 0, "CarInfo"); 
```

If you run the code now, you will see that it works—`CarInfo` has become a regular QML element. Due to this, its properties can be set and modified directly in the document, right? If you try setting the speed or the distance, it will work just fine. However, try to set a property grouped in the `engine` property:

```cpp
CarInfo {
    id: carData
    visible: true
 engine.gear: 3
}
```

QML runtime will complain with a message similar to the following one:

```cpp
Cannot assign to non-existent property "gear"
             engine.gear: 3
                    ^ 
```

This is because the runtime does not understand the `engine` property—we declared it as `QObject` and yet we are using a property this class doesn't have. To avoid this issue, we have to teach the runtime about `CarInfoEngine`.

First, let's update the property declaration macro to use `CarInfoEngine` instead of `QObject`:

```cpp
Q_PROPERTY(CarInfoEngine* engine READ engine NOTIFY engineChanged) 
```

Also, the getter function itself:

```cpp
CarInfoEngine* engine() const { 
    return m_engine; 
}
```

You should make these changes in both the `CarInfo` and `CarInfoProxy` classes. Then, we should teach the runtime about the type:

```cpp
QString msg = QStringLiteral("Objects of type CarInfoEngine cannot be created");
qmlRegisterUncreatableType<CarInfoEngine>("CarInfo", 1, 0, "CarInfoEngine", msg);
```

# What just happened?

In this exercise, we let the QML runtime know about two new elements. One of them is `CarInfo`, which is a proxy to our widget class. We told the engine that this is a full-featured class that is instantiable from QML. The other class, `CarInfoEngine`, also became known to QML; however, the difference is that every attempt to declare an object of this type in QML fails with a given warning message. There are other functions available for registering types in QML, but they are rarely used, so we will not be describing them here. If you are curious about them, type in `qmlRegister` in the Index tab of Creator's Help pane.

# Pop quiz

Q1\. Which QML type allows you to create a placeholder for an object that will be instantiated later?

1.  `Repeater`
2.  `Loader`
3.  `Component`

Q2\. Which QML type provides low-level access to individual touch events?

1.  `PinchArea`
2.  `MouseArea`
3.  `MultiPointTouchArea`

Q3\. When can you access a component defined in another QML file without an import statement?

1.  You can do that if the component is registered using the `qmlRegisterType` function
2.  You can do that if the component file is added to the project resources
3.  You can do that if the component file is in the same directory as the current file

# Summary

You are now familiar with multiple methods that can be used to extend Qt Quick with your own item types. You learned to use JavaScript to create custom visual items. You also know how to use C++ classes as non-visual QML elements fully integrated with your UI. We also discussed how to handle mouse, touch, keyboard, and gamepad events in Qt Quick applications. However, so far, despite us talking about "fluid" and "dynamic" interfaces, you haven't seen much of them. Do not worry; in the next chapter, we will focus on animations in Qt Quick as well as fancy graphics and applying what you learned in this chapter for creating nice-looking and interesting games. So, read on!