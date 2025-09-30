# Introduction to Qt Quick

In this chapter, you will be introduced to a technology called **Qt Quick** that allows us to implement resolution-independent user interfaces with lots of eye-candy, animations, and effects that can be combined with regular Qt code that implements the logic of the application. You will learn the basics of the QML declarative language that forms the foundation of Qt Quick. You will create a simple Qt Quick application and see the advantages offered by the declarative approach.

The main topics covered in this chapter are these:

*   QML basics
*   Overview of Qt modules
*   Using Qt Quick Designer
*   Utilizing Qt Quick modules
*   Property bindings and signal handling
*   Qt Quick and C++
*   States and transitions

# Declarative UI programming

Although it is technically possible to use Qt Quick by writing C++ code, the module is accompanied by a dedicated programming language called **QML** (**Qt Modeling Language**). QML is an easy to read and understand declarative language that describes the world as a hierarchy of components that interact and relate to one another. It uses a JSON-like syntax and allows us to use imperative JavaScript expressions as well as dynamic property bindings. So, what is a declarative language, anyway?

Declarative programming is one of the programming paradigms that dictates that the program describes the logic of the computation without specifying how this result should be obtained. In contrast to imperative programming, where the logic is expressed as a list of explicit steps forming an algorithm that directly modifies the intermediate program state, a declarative approach focuses on what the ultimate result of the operation should be.

# Time for action – Creating the first project

Let's create a project to better understand what QML is. In Qt Creator, select File and then New File or Project in the main menu. Choose Application in the left column and select the Qt Quick Application - Empty template. Name the project as `calculator` and go through the rest of the wizard.

Qt Creator created a sample application that displays an empty window. Let's examine the project files. The first file is the usual `main.cpp`:

```cpp
#include <QGuiApplication>
#include <QQmlApplicationEngine>

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;
    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;
    return app.exec();
}
```

This code simply creates the application object, instantiates the QML engine, and asks it to load the `main.qml` file from the resources. If an error occurs, `rootObjects()` will return an empty list, and the application will terminate. If the QML file was loaded successfully, the application enters the main event loop.

The `*.qrc` file is a resource file. The concept of resource files should be familiar to you from [Chapter 3](ebffc011-752f-4dbe-a383-0917a002841d.xhtml), *Qt GUI Programming*. Basically, it contains the list of arbitrary project files that are required for project execution. During compilation, the contents of these files are embedded into the executable. You can then retrieve the content at runtime by specifying a virtual filename, such as `qrc:/main.qml` in the preceding code. You can expand the `Resources` section of the Project tree further to see all files added to the resource file.

In the sample project, `qml.qrc` references a QML file named `main.qml`. If you don't see it in the project tree, expand `Resources`, `qml.qrc`, and then `/` sections. The `main.qml` file is the top-level QML file that is loaded into the engine. Let's take a look at it:

```cpp
import QtQuick 2.9
import QtQuick.Window 2.2

Window {
    visible: true
    width: 640
    height: 480
    title: qsTr("Hello World")
}
```

This file *declares* what objects should be created at the start of the application. As it uses some QML types provided by Qt, it contains two `import` directives at the top of the file. Each `import` directive contains the name and the version of the imported module. In this example, `import QtQuick.Window 2.2` enables us to use the `Window` QML type provided by this module.

The rest of the file is the declaration of the objects the engine should create. The `Window { ... }` construction tells QML to create a new object of the `Window` type. The code within this section assigns values to properties of this object. We explicitly assign a constant to the `visible`, `width`, and `height` properties of the window object. The `qsTr()` function is the translation function, just like `tr()` in C++ code. It returns the passed string without change by default. The `title` property will contain the result of evaluating the passed expression.

# Time for action – Editing QML

Let's add some content to our window. Edit the `main.qml` file with the following code:

```cpp
import QtQuick 2.9
import QtQuick.Window 2.2
import QtQuick.Controls 2.2
Window {
    visible: true
    width: 640
    height: 480
    title: qsTr("Hello World")
 TextField {
 text: "Edit me"
 anchors {
 top: parent.top
 left: parent.left
 }
 }
 Label {
 text: "Hello world"
 anchors {
 bottom: parent.bottom
 left: parent.left
 }
 }
}
```

When you run the project, you will see a text field and a label in the window:

![](img/7ef74b08-fdd9-4e08-aae5-da1c917026ce.png)

# What just happened?

First, we added an import statement to make the `QtQuick.Controls` module available in the current scope. If you're not sure which version to use, invoke Qt Creator's code completion and use the most recent version. Due to the new import, we can now use the `TextField` and `Label` QML types in our QML file.

Next, we declared two **children** of the top-level `Window` object. QML objects form a parent-child relationship, much like `QObject` in C++. However, you don't need to explicitly assign parents to items. Instead, you declare the object within the declaration of its parent, and QML will automatically ensure that relationship. In our example, the `TextField { ... }` part tells QML to create a new QML object of the `TextField` type. 

Since this declaration lies within the `Window { ... }` declaration, the `TextField` object will have the `Window` object as its parent. The same applies to the `Label` object. You can create multiple levels of nesting in a single file, if needed. You can use the `parent` property to access the parent item of the current item.

After declaring a new object, we assign values to its properties within its declaration. The `text` property is self-explanatory—it contains the text displayed in the UI. Note that the `TextField` object allows the user to edit the text. When the text is edited in the UI, the `text` property of the object will reflect the new value.

Finally, we assign value to the `anchors` **property group** to position the items as we like. We put the text field in the top-left corner of the window and put the label in the bottom-left corner. This step requires a more thorough explanation.

# Property groups

Before we discuss anchors, let's talk about property groups in general. This is a new concept introduced in QML. Property groups are used when there are multiple properties with a similar purpose. For example, the `Label` type has a number of properties related to the font. They can be implemented as separate properties; consider the following example:

```cpp
Label {
    // this code does not work
    fontFamily: "Helvetica"
    fontSize: 12
    fontItalic: true 
}
```

However, such repetitive code is hard to read. Luckily, font properties are implemented as a property group, so you can set them using the **group notation** syntax:

```cpp
Label {
    font {
        family: "Helvetica"
        pointSize: 12
        italic: true 
    }
}
```

This code is much cleaner! Note that there is no colon character after `font`, so you can tell that this is a property group assignment.

In addition, if you only need to set one subproperty of the group, you can use the **dot notation** syntax:

```cpp
Label {
    font.pointSize: 12
}
```

The dot notation is also used to refer to subproperties in the documentation. Note that you should prefer group notation if you need to set more than one subproperty.

That's all you need to know about property groups. Besides `font`, you can find many other property groups in some QML types, for example, `border`, `easing`, and `anchors`.

# Anchors

Anchors allow you to manage item geometry by attaching certain points of some objects to points of another object. These points are called anchor lines. The following diagram shows the anchor lines that are available for each Qt Quick item:

![](img/525087ad-048e-4dfb-a6dd-2debe2a3e6bd.png)

You can establish bindings between anchor lines to manage relative positioning of items. For each anchor line, there is a property that returns the current coordinate of that anchor line. For example, the `left` property returns the *x* coordinate of the left border of the item, and the `top` property returns the *y* coordinate of its top border. Next, each object contains the `anchors` property group that allows you to set coordinates of the anchor line for that item. For example, the `anchors.left` property can be used to request the position of the left border of the object. You can use these two kinds of properties together to specify relative positions of objects:

```cpp
anchors.top: otherObject.bottom
```

This code declares that the top anchor line of the object must be bound to the bottom anchor line of the other object. It's also possible to specify a margin for such binding through properties, such as `anchors.topMargin`.

The `anchors.fill` property is the shortcut for binding the `top`, `bottom`, `left`, and `right` anchors to the specified object's respective anchor lines. As a result, the item will have the same geometry as the other object. The following code snippet is often used to expand the item to the whole area of its parent:

```cpp
anchors.fill: parent
```

# Time for action – Positioning items relative to each other

In our previous example, we used the following code to position the label:

```cpp
anchors {
    bottom: parent.bottom
    left: parent.left
}
```

You should be able to understand this code by now. The `parent` property returns the reference to the parent QML object. In our case, it's the window. The `parent.bottom` expression returns the *y* coordinate of the parent's bottom anchor line. By assigning this expression to the `anchors.bottom` property, we ensure that the bottom anchor line of the label stays in the same position as the bottom anchor line of the window. The *x* coordinate is restricted in a similar way.

Now, let's see whether we can position the label just below the text field. In order to do that, we need to bind the `anchors.top` property of the label to the bottom anchor line of the text field. However, we have no way to access the text field from within the label yet. We can fix this by defining the `id` property of the text field:

```cpp
TextField {
    id: textField
    text: "Edit me"
    anchors {
        top: parent.top
        left: parent.left
    }
}
Label {
    text: "Hello world"
    anchors {
        top: textField.bottom
 topMargin: 20
        left: parent.left
    }
}
```

Setting an ID is similar to assigning the object to a variable. We can now use the `textField` variable to refer to our `TextField` object. The label is now positioned 20 pixels below the text field.

# QML types, components, and documents

QML introduces some new concepts that you should be familiar with. A **QML type** is a concept similar to C++ class. Any value or object in QML should have some type and should be exposed to JavaScript code in a certain way. There are two major kinds of QML types:

*   **Basic types** are types that hold a concrete value and do not refer to any other objects, for example, `string` or `point`
*   **Object types** are types that can be used to create objects with certain functionality and consistent interface

Basic QML types are similar to C++ primitive types and data structures, such as `QPoint`. Object types are closer to widget classes, such as `QLineEdit`, but they are not necessarily tied to GUI.

There are numerous QML types provided by Qt. We've already used the `Window`, `TextField`, and `Label` types in our previous examples. You can also create your own custom QML types with unique functionality and behavior. The simplest way to create a QML type is to add a new `.qml` file with a capitalized name to the project. The base file name defines the name of the created QML type. For example, the `MyTextField.qml` file will declare a new `MyTextField` QML type.

Any complete and valid QML code is called a **document**. Any valid QML file contains a document. It's also possible to load documents from any source (for example, over the network). A **component** is a document loaded into the QML engine.

# How does it work?

Qt Quick infrastructure hides most of the implementation details from the developer and allows you to keep your application code clean. Nevertheless, it's always important to understand what's going on.

The **QML engine** is a C++ class that understands QML code and executes the required actions to make it work. In particular, the QML engine is responsible for creating objects according to the requested hierarchy, assigning values to properties, and executing event handlers in response to events.

While QML language itself is quite far from JavaScript, it allows you to use any JavaScript expressions and code blocks for calculating values and handling events. This means that the QML engine must be capable of executing JavaScript. Under the hood, the implementation uses a very fast JavaScript engine, so you shouldn't usually worry about the performance of your JavaScript code.

The JavaScript code should be able to interact with QML objects, so every QML object is exposed as a JavaScript object with corresponding properties and methods. This integration uses the same mechanism that we learned in [Chapter 10](fa5baf43-2d1a-4717-8ac1-cd190ab6e440.xhtml), *Scripting*. In C++ code, you have some control over the objects embedded into the QML engine and can even create new objects. We will get back to this topic later in the chapter.

While QML is a general purpose language, Qt Quick is a QML-based module that focuses on user interfaces. It provides a two-dimensional hardware accelerated canvas that contains a hierarchy of interconnected items. Unlike Qt Widgets, Qt Quick was designed to support visual effects and animations efficiently, so you can use its powers without significant performance degradation.

Qt Quick views are not based on a web browser engine. Browsers tend to be quite heavy, especially for mobile devices. However, you can use a web engine explicitly when you need it by adding the `WebView` or `WebEngine` object to your QML files.

# Time for action – Property binding

QML is much more powerful than simple JSON. Instead of specifying an explicit value for a property, you can use an arbitrary JavaScript expression that will be automatically evaluated and assigned to the property. For example, the following code will display "ab" in the label:

```cpp
Label {
    text: "a" + "b"
    //...
}
```

You can also refer to properties of the other objects in the file. As we saw earlier, you can use the `textEdit` variable to set relative position of the label. This is one example of a property binding. If the value of the `textField.bottom` expression changes for some reason, the `anchors.top` property of the label will be automatically updated with the new value. QML allows you to use the same mechanism for every property. To make the effect more obvious, let's assign an expression to the label's text property:

```cpp
Label {
    text: "Your input: " + textField.text
    //...
}
```

Now the label's text will be changed according to this expression. When you change the text in the input field, the text of the label will be automatically updated!:

![](img/8b4e90ea-05d8-45d2-a6a1-4526614636be.png)

The property binding differs from a regular value assignment and binds the value of the property to the value of the supplied JavaScript expression. Whenever the expression's value changes, the property will reflect that change in its own value. Note that the order of statements in a QML document does not matter as you declare relations between properties.

This example shows one of the advantages of the declarative approach. We didn't have to connect signals or explicitly determine when the text should be changed. We just *declared* that the text should be influenced by the input field, and the QML engine will enforce that relation automatically.

If the expression is complex, you can replace it with a multiline block of text that works as a function:

```cpp
text: {
    var x = textField.text;
    return "(" + x + ")";
}
```

You can also declare and use a named JavaScript function within any QML object declaration:

```cpp
Label {
    function calculateText() {
        var x = textField.text;
        return "(" + x + ")";
    }
    text: calculateText()
    //...
}
```

# A limitation of automatic property updates

QML does its best to determine when the function value may change, but it is not omnipotent. For our last function, it can easily determine that the function result depends on the value of the `textField.text` property, so it will re-evaluate the binding if that value changes. However, in some cases, it can't know that a function may return a different value the next time it is called, and in such situations, the statement will not be re-evaluated. Consider the following property binding:

```cpp
Label {
    function colorByTime() {
        var d = new Date();
        var seconds = d.getSeconds();
        if(seconds < 15) return "red";
        if(seconds < 30) return "green";
        if(seconds < 45) return "blue";
        return "purple";
    }
    color: colorByTime()
    //...
}
```

The color will be set at the start of the application, but it will not work properly. QML will only call the `colorByTime()` function once when the object is initialized, and it will never call it again. This is because it has no way of knowing how often this function must be called. We will see how to overcome this in [Chapter 12](4fdfe294-c35c-476d-9656-0aefd533e491.xhtml), *Customization in Qt Quick*.

# Overview of QML types provided by Qt

Before we continue to work on our QML application, let's see what the built-in libraries are capable of. This will allow us to pick the right modules for the task. Qt provides a lot of useful QML types. In this section, we will provide an overview of the most useful modules available in Qt 5.9.

The following modules are important for building user interfaces:

*   The `QtQuick` base module provides functionality related to drawing, event handling, positioning of elements, transformations, and many other useful types
*   `QtQuick.Controls` provides basic controls for user interfaces, such as buttons and input fields
*   `QtQuick.Dialogs` contains file dialogs, color dialogs, and message boxes
*   `QtQuick.Extras` provides additional controls, such as dials, tumblers, and gauges
*   `QtQuick.Window` enables window management
*   `QtQuick.Layouts` provide layouts for automatic positioning of objects on screen
*   `UIComponents` provides tab widget, progress bar, and switch types
*   `QtWebView` allows you to add web content to the application
*   `QtWebEngine` provides more sophisticated web browser functionality

If you want to implement rich graphics, the following modules may be of help:

*   `QtCanvas3D` provides a canvas for 3D rendering
*   `Qt3D` modules provide access to real-time simulation systems supporting 2D and 3D rendering
*   `QtCharts` allows you to create sophisticated charts

*   `QtDataVisualization` can be used to build 3D visualizations of datasets
*   `QtQuick.Particles` allows you to add particle effects
*   `QtGraphicalEffects` can apply graphical effects (such as blur or shadow) to other Qt Quick objects

Qt provides a lot of functionality commonly required on mobile devices:

*   `QtBluetooth` supports basic communication with other devices over Bluetooth
*   `QtLocation` allows you to display maps and find routes
*   `QtPositioning` provides information about the current location
*   `QtNfc` allows you to utilize NFC hardware
*   `QtPurchasing` implements in-app purchases
*   `QtSensors` provides access to on-board sensors, such as accelerometer or gyroscope
*   `QtQuick.VirtualKeyboard` provides an implementation of an onscreen keyboard

Finally, there are two modules providing multimedia capabilities:

*   `QtMultimedia` provides access to audio and video playback, audio recording, camera, and radio
*   `QtAudioEngine` implements 3D positional audio playback

There are many more QML modules that we didn't mention here. You can find the full list on the All QML Modules documentation page. Note that some of the modules are not provided under LGPL license.

# Qt Quick Designer

We can use QML to easily create a hierarchy of objects. If we need a few input boxes or buttons, we can just add some blocks to the code, just like we added the `TextField` and `Label` components in the previous example, and our changes will appear in the window. However, when dealing with complex forms, it's sometimes hard to position the objects properly. Instead of trying different `anchors` and relaunching the application, you can use the visual form editor to see the changes as you make them.

# Time for action – Adding a form to the project

Locate the `qml.qrc` file in Qt Creator's project tree and invoke the Add New... option in its context menu. From Qt section, select the QtQuick UI File template. Input `Calculator` in the Component name field. The Component form name field will be automatically set to `CalculatorForm`. Finish the wizard.

Two new files will appear in our project. The `CalculatorForm.ui.qml` file is the form file that can be edited in the form editor. The `Calculator.qml` file is a regular QML file that can be edited manually to implement the behavior of the form. Each of these files introduces a new QML type. The `CalculatorForm` QML type is immediately used in the generated `Calculator.qml` file:

```cpp
import QtQuick 2.4
CalculatorForm {
}
```

Next, we need to edit the `main.qml` file to add a `Calculator` object to the window:

```cpp
import QtQuick 2.9
import QtQuick.Window 2.2
import QtQuick.Controls 2.2
Window {
    visible: true
    width: 640
    height: 480
    title: qsTr("Calculator")

 Calculator {
 anchors.fill: parent
 }
}
```

QML components are similar to C++ classes in some way. A QML component encapsulates an object tree so that you can use it without knowing about the exact content of the component. When the application is started, the `main.qml` file will be loaded into the engine, so the `Window` and `Calculator` objects will be created. The Calculator object, in turn, will contain a `CalculatorForm` object. The `CalculatorForm` object will contain the items that we add later in the form editor.

# Form editor files

When we worked with Qt Widgets form editor, you may have noted that a widget form is an XML file that is converted to a C++ class during compilation. This does not apply to Qt Quick Designer. In fact, the files produced by this form editor are completely valid QML files that are directly included in the project. However, the form editor files have a special extension (`.ui.qml`), and there are some artificial restrictions that protect you from doing bad things.

The `ui.qml` files should only contain content that is visible in the form editor. You do not need to edit these files by hand. It's not possible to call functions or execute JavaScript code from these files. Instead, you should implement any logic in a separate QML file that uses the form as a component.

If you're curious about the content of a `ui.qml` file, you can click on the Text Editor tab that is positioned on the right border of the form editor's central area.

# Form editor interface

When you open a `.ui.qml` file, Qt Creator goes to the Design mode and opens the Qt Quick Designer interface:

![](img/77e0308b-cbaf-4db9-9269-498cc96a7cf4.png)

We've highlighted the following important parts of the interface:

*   The main area (**1**) contains visualization of the document's content. You can click on the Text Editor tab at the right border of the main area to view and edit the QML code of the form without exiting the form editor. The bottom part of the main area displays list of states of the component.
*   The Library pane (**2**) shows the available QML object types and allows you to create new objects by dragging them to the navigator or to the main area. The Imports tab contains a list of available QML modules and allows you to export a module and access more QML types.
*   The Navigator pane (**3**) displays the hierarchy of the existing objects and their names. The buttons to the right of the names allow you to export an object as public property and toggle its visibility in the form editor.
*   The Connections pane (**4**) provides ability to connect signals, change property bindings, and manage public properties of the form.
*   The Properties pane (**5**) allows you to view and edit properties of the selected object.

We will now use the form editor to create a simple calculator application. Our form will contain two input boxes for operands, two radio buttons for selecting the operation, a label to display the result, and a button to reset everything to the original state.

# Time for action – Adding an import

The default object palette contains a very minimal set of types provided by the `QtQuick` module. To access a richer set of controls, we need to add an `import` directive to our document. To do this, locate the Library pane in the top-left corner of the window and go to its Imports tab. Next, click on Add Import and select QtQuick.Controls 2.2 in the drop-down list. The selected import will appear in the tab. You can click on the × button to the left of the import to remove it. Note that you cannot remove the default import.

Adding the import using the form editor will result in adding the `import QtQuick.Controls 2.2` directive to the `.ui.qml` file. You can switch the main area to the Text Editor mode to see this change.

Now you can switch back to the QML Types tab of the Library pane. The palette will contain controls provided by the imported module.

# Time for action – Adding items to the form

Locate the Text Field type in the Qt Quick - Controls 2 section of the library pane and drag it to the main area. A new text field will be created. We will also need the Radio Button, Label, and Button types from the same section. Drag them to the form and arrange them as shown:

![](img/6e977c66-f03a-4ced-b927-27ee0bc1f8b2.png)

Next, you need to select each element and edit its properties. Click on the first text field in the main area or in the navigator. The blue frame around the object in the main area will indicate that it is selected. Now you can use the property editor to view and edit properties of the selected element. First, we want to set the `id` property that will be used to refer to the object in the code. Set the `id` property of the text edits to `argument1` and `argument2`. Locate the Text property under the TextField tab in the property editor. Set it to `0` for both text fields. The changed text will be immediately displayed in the main area.

Set `id` of the radio buttons to `operationAdd` and `operationMultiply`. Set their text to `+` and `×`. Set the `checked` property of the `operationAdd` button to `true` by toggling the corresponding checkbox in the property editor.

The first label will be used to statically display the `=` sign. Set its `id` to `equalSign` and `text` to `=`. The second label will actually display the result. Set its `id` to `result`. We will take care of the `text` property later.

The button will reset the calculator to the original state. Set its `id` to `reset` and `text` to `Reset`.

You can run the application now. You will see that the controls are shown in the window, but they are not repositioned in respect to the window size. They always stay in the same positions. If you check out the text content of `CalculatorForm.ui.qml`, you will see that the form editor sets the `x` and `y` properties of each element. To make a more responsive form, we need to utilize the `anchors` property instead.

# Time for action – Editing anchors

Let's see how we can edit anchors in the form editor and see the result on the fly. Select the `argument1` text field and switch to the Layout tab in the middle part of the Properties pane. The tab contains Anchors text, followed by a set of buttons for all anchor lines of this item. You can mouse over the buttons to see their tooltips. Click on the first button, 
Anchor item to the top. A new set of controls will appear below the button, allowing you to configure this anchor.

First, you can select the target object, that is, the object containing the anchor line that will be used as the reference. Next, you can select the margin between the reference anchor line and the anchor line of the current object. To the right of the margin, there are buttons that allow you to choose which anchor line of the target to use as the reference. For example, if you choose the bottom line, our text field will retain its position relative to the bottom border of the form.

Anchor the top line of the text field to the top line of the parent and set Margin to 20\. Next, anchor the horizontal center line to parent with Margin 0\. The property editor should look like this:

![](img/94d9e4eb-eaed-45a2-a566-fa4be81e576b.png)

You can also verify the QML representation of these settings:

```cpp
TextField {
    id: a
    text: qsTr("0")
    anchors.horizontalCenter: parent.horizontalCenter
    anchors.top: parent.top
    anchors.topMargin: 20
}
```

If you drag the text field around using the mouse instead of setting the anchors, the form editor will set the `x` and `y` properties to position the element according to your actions. If you edit anchors of the item afterward, the `x` and `y` properties may remain set, but their effect will be overridden by the anchor effects.

Let's repeat this process for the `operationAdd` radio button. First, we need to adjust its horizontal position relative to the horizontal center of the form. Select the radio button, click on the ![](img/1a31d9cc-40ac-4379-965f-3173d6cb6f85.png) Anchor item to the right button, leave `parent` as the target, and click on the 
![](img/a72fda6d-5348-4cb5-9df5-0980531c8e12.png) Anchor to the horizontal center of the target button to the right of the margin input. Set margin to `10`. This will allow us to position the second radio button 10 points to the right of the horizontal center, and the space between the radio buttons will be 20.

Now, what about the top anchor? We can attach it to the parent and just set the margin that will look nice. However, ultimately, what we want is a specific vertical margin between the first text field and the first radio button. We can do this easily.

Enable the top anchor for the `operationAdd` radio button, select `argument1` in the Target drop-down list, click on the ![](img/7d6a8982-6c6a-457e-8c96-ed97c97a9cfe.png) Anchor to the bottom of the target button to the right of the margin field, and input 20 in the margin field. Now the radio button is anchored to the text field above it. Even if we change the height of the text field, the vertical margin between the elements will stay intact. You can run the application and verify that the `argument1` and `operationAdd` elements now respond to window size changes.

Now, all we need is to repeat this process for the rest of the objects. However, this is quite a tedious task. It will get even more inconvenient in a larger form. Making changes to such forms will also be cumbersome. For example, to change the order of fields, you will need to carefully edit the anchors of involved objects. While anchors are good in simple cases, it's better to use a more automated approach for large forms. Luckily, Qt Quick provides layouts for this purpose.

# Time for action – Applying layouts to the items

Before we apply layouts to objects, remove the anchors we had created. To do this, select each element and click on the buttons under Anchors text to uncheck them. The anchor properties below the buttons will disappear. The layout will now be able to position the objects.

First, import the `QtQuick.Layouts 1.3` module into the form, like we did earlier for `QtQuick.Controls`. Locate the Qt Quick - Layouts section in the palette and examine the available layouts:

*   Column Layout will arrange its children vertically
*   Row Layout will arrange its children horizontally
*   Grid Layout will arrange its children vertically and horizontally in a grid
*   Stack Layout will display only one of its children and hide the rest of them

Layouts are sensitive to the hierarchy of the objects. Let's use Navigator instead of the main area to manage our items. This will allow us to see the parent-child relationships between items more clearly. First, drag a Row Layout and drop it over the root item in the Navigator. A new `rowLayout` object will be added as a child of the root object. Next, drag the `operationAdd` and `operationMultiply` objects in the Navigator and drop them to the `rowLayout`. The radio buttons are now children of the row layout, and they are automatically positioned next to each other.

Now, drag a Column Layout to the root object. Select all other children of the root object, including `rowLayout`, in the Navigator, and drag them to the `columnLayout` object. If the items end up in wrong order, use the Move up and Move down buttons at the top part of the Navigator to arrange the items properly. You should get the following hierarchy:

![](img/dbf872b0-4e72-442e-b1be-514cce400e21.png)

The `columnLayout` object will automatically position its children, but how to position the object itself? We should use anchors to do that. Select `columnLayout`, switch to the Layout tab in the property editor and click on the ![](img/60485259-c353-44c3-85ce-97228e21b65a.png) Fill parent item button. This will automatically create 4 anchor bindings and expand `columnLayout` to fill the form.

The items are now positioned automatically, but they are bound to the left border of the window. Let's align them to the middle. Select the first text field and switch to the Layout tab. As the object is now in a layout, the anchor settings are replaced with settings the layout understands. The Alignment property defines how the item is positioned within the available space. Select `AlignHCenter` in the first drop-down list. Repeat the process for each direct child of `columnLayout`.

You can now run the application and see how it reacts to changing window size:

![](img/19f83386-3473-4113-9cb0-106471ae5b54.png)

The form is ready. Let's implement the calculations now.

# Time for action – Assigning an expression to the property

As you already saw, assigning constant text to a label is easy. However, you can also assign a dynamic expression to any property in the form editor. To do that, select the `result` label and mouse over the circle in the left part of Text property input field. When the circle turns into an arrow, click on it and select Set Binding in the menu. Input `argument1.text + argument2.text` in the binding editor and confirm the change.

If you run the application now, you will see that the `result` label will always display the concatenation of the strings the user inputs in the fields. That's because the `argument1.text` and `argument2.text` properties have the `string` type, so the `+` operation performs concatenation.

This feature is very useful if you need to apply simple bindings. However, it is not sufficient in our case, as we need to convert strings to numbers and select which arithmetic operation the user requested. Using functions in the form editor is not allowed, so we cannot implement this complex logic right here. We need to do it in the `Calculator.qml` file. This restriction will help us separate the view from the logic behind it.

# Time for action – Exposing items as properties

Children of a component are not available from outside of it by default. This means that `Calculator.qml` cannot access input fields or radio buttons of our form. To implement the logic of the calculator, we need to access these objects, so let's expose them as public properties. Select the `argument1` text field in the Navigator and click on the ![](img/49a5ea47-dcc2-497f-b3e8-3b6cf42e19da.png) Toggles whether this item is exported as an alias property of the root item button to the right of the object ID. After you click on the button, its icon will change to indicate that the item is exported. Now we can use the `argument1` public property in `Calculator.qml` to access the input field object.

Enable public properties for the `argument1`, `argument2`, `operationAdd`, `operationMultiply`, and `result` objects. The rest of the objects will remain hidden as implementation details of the form.

Now go to the `Calculator.qml` file and use the exposed properties to implement the calculator logic:

```cpp
CalculatorForm {
    result.text: {
        var value1 = parseFloat(argument1.text);
        var value2 = parseFloat(argument2.text);
        if(operationMultiply.checked) {
            return value1 * value2;
        } else {
            return value1 + value2;
        }
    }
}
```

# What just happened?

Since we exported objects as properties, we can access them by ID from outside of the form. In this code, we bind the `text` property of the `result` object to the return value of the code block that is enclosed in braces. We use `argument1.text` and `argument2.text` to access the current text of the input fields. We also use `operationMultiply.checked` to see whether the user checked the `operationMultiply` radio button. The rest is just straightforward JavaScript code.

Run the application and see how the result label automatically displays the result when the user interacts with the form.

# Time for action – Creating an event handler

Let's implement the last bit of functionality. When the user clicks on the Reset button, we should change the form's values. Go back to the form editor and right-click on the `reset` button in the Navigator or in the main area. Select Add New Signal Handler. Qt Creator will navigate to the corresponding implementation file (`Calculator.qml`) and display the Implement Signal Handler dialog. Select the `clicked` signal in the drop-down list and click on the OK button to confirm the operation. This operation will do two things:

*   The `reset` button will be automatically exported as a public property, just like we did it manually for the other controls
*   Qt Creator will create a boilerplate for the new signal handler in the `Calculator.qml` file

Let's add our implementation to the automatically generated block:

```cpp
reset.onClicked: {
    argument1.text = "0";
    argument2.text = "0";
    operationAdd.checked = true;
}
```

When the button is clicked on, this code will be executed. The text fields will be set to 0, and the `operationAdd` radio button will be checked. The `operationMultiply` radio button will be unchecked automatically.

Our calculator fully works now! We used declarative approach to implement a nicely looking and responsive application.

# Qt Quick and C++

While QML has a lot of built-in functionality available, it will almost never be enough. When you're developing a real application, it always needs some unique functionality that is not available in QML modules provided by Qt. The C++ Qt classes are much more powerful, and third-party C++ libraries are also always an option. However, the C++ world is separated from our QML application by the restrictions of QML engine. Let's break that boundary right away.

# Accessing C++ objects from QML

Let's say that we want to perform a heavy calculation in C++ and access it from our QML calculator. We will choose factorial for this project.

The QML engine is really fast, so you can most likely calculate factorials directly in JavaScript without performance problems. We just use it here as a simple example.

Our goal is to inject our C++ class into the QML engine as a JavaScript object that will be available in our QML files. We will do that exactly like we did it in [Chapter 10](fa5baf43-2d1a-4717-8ac1-cd190ab6e440.xhtml), *Scripting*. The `main` function creates a `QQmlApplicationEngine` object that inherits `QJSEngine`, so we have access to the API that is already familiar to us from that chapter. Here, we'll just show how we can apply this knowledge to our application without going into detail.

Go to the Edit mode, right-click on the project in the project tree and select Add New. Select the C++ Class template, input `AdvancedCalculator` as the class name and select QObject in the Base Class drop-down list.

Declare the invokable `factorial` function in the generated `advancedcalculator.h` file:

```cpp
Q_INVOKABLE double factorial(int argument);
```

We can implement this function using the following code:

```cpp
double AdvancedCalculator::factorial(int argument) {
    if (argument < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (argument > 180) {
      return std::numeric_limits<double>::infinity();
    }
    double r = 1.0;
    for(int i = 2; i <= argument; ++i) {
        r *= i;
    }
    return r;
}
```

We guard the implementation against too large inputs because `double` wouldn't be able to fit the resulting values anyway. We also return `NaN` on invalid inputs.

Next, we need to create an instance of this class and import it into the QML engine. We do this in the `main()`:

```cpp
engine.globalObject().setProperty("advancedCalculator",
    engine.newQObject(new AdvancedCalculator));
return app.exec();
```

Our object is now available as the `advancedCalculator` global variable. Now we need to use this variable in the QML file. Open the form editor and add the third radio button to the `rowLayout` item. Set `id` of the radio button to `operationFactorial` and text to `!`. Export this radio button as a public property so that we can access it from the outside. Next, let's adjust the `result.text` property binding in the `Calculator.qml` file:

```cpp
result.text: {
    var value1 = parseFloat(argument1.text);
    var value2 = parseFloat(argument2.text);
    if(operationMultiply.checked) {
        return value1 * value2;
 } else if (operationFactorial.checked) {
 return advancedCalculator.factorial(value1);
    } else {
        return value1 + value2;
    }
}
```

If the `operationFactorial` radio button is checked, this code will call the `factorial()` method of the `advancedCalculator` variable and return it as the result. The user will see it as text of the `result` label. When factorial operation is selected, the second text field is unused. We'll do something about that later in this chapter.

For more information about exposing C++ API to JavaScript, refer to [Chapter 10](fa5baf43-2d1a-4717-8ac1-cd190ab6e440.xhtml), *Scripting*. Most of the techniques described there apply to the QML engine as well.

We exposed a C++ object as a JavaScript object that is accessible from the QML engine. However, it is not a QML object, so you can't include it in the QML objects hierarchy or apply property bindings to properties of the object that was created this way. It's possible to create a C++ class that will work as a fully functional QML type, leading to a more powerful integration of C++ and QML. We will show that approach in [Chapter 12](4fdfe294-c35c-476d-9656-0aefd533e491.xhtml), *Customization in Qt Quick*.

There is another way to expose our `AdvancedCalculator` class to JavaScript. Instead of adding it to the global object, we can register it as a singleton object in the QML module system using the `qmlRegisterSingletonType()` function:

```cpp
qmlRegisterSingletonType("CalculatorApp", 1, 0, "AdvancedCalculator",
        [](QQmlEngine *engine, QJSEngine *scriptEngine) -> QJSValue {
    Q_UNUSED(scriptEngine);
    return engine->newQObject(new AdvancedCalculator);
});
QQmlApplicationEngine engine;
```

We pass the QML module name, major and minor versions, and the singleton name to this function. You can choose these values arbitrarily. The last argument is a callback function that will be called when this singleton object is accessed in the JS engine for the first time.

The QML code also needs to be slightly adjusted. First, import our new QML module into scope:

```cpp
import CalculatorApp 1.0
```

Now you can just access the singleton by name:

```cpp
return AdvancedCalculator.factorial(value1);
```

When this line is executed for the first time, Qt will call our C++ callback and create the singleton object. For subsequent calls, the same object will be used.

# Accessing QML objects from C++

It is also possible to create QML objects from C++ and access the existing objects living in the QML engine (for example, those declared in some QML file). However, in general, doing this thing is bad practice. If we assume the most common case, which is that the QML part of our application deals with a user interface in Qt Quick for the logic written in C++, then accessing Qt Quick objects from C++ breaks the separation between logic and the presentation layer, which is one of the major principles in GUI programming. The user interface is prone to dynamic changes, relayouting up to a complete revamp. Heavy modifications of QML documents, such as adding or removing items from the design, will then have to be followed by adjusting the application logic to cope with those changes. In addition, if we allow a single application to have multiple user interfaces (skins), it might happen that because they are so different, it is impossible to decide upon a single set of common entities with hard-coded names that can be fetched from C++ and manipulated. Even if you managed to do that, such an application could crash easily if the rules were not strictly followed in the QML part.

That said, we have to admit that there are cases when it does make sense to access QML objects from C++, and that is why we decided to familiarize you with the way to do it. One of the situations where such an approach is desired is when QML serves us as a way to quickly define a hierarchy of objects with properties of different objects linked through more or fewer complex expressions, allowing them to answer to changes taking place in the hierarchy.

The `QQmlApplicationEngine` class provides access to its top-level QML objects through the `rootObjects()` function. All nested QML objects form a parent-child hierarchy visible from C++, so you can use `QObject::findChild` or `QObject::findChildren` to access the nested objects. The most convenient way to find a specific object is to set its `objectName` property. For example, if we want to access the reset button from C++, we need to set its object name.

The form editor does not provide a way to set `objectName` for its items, so we need to use the text editor to make this change:

```cpp
Button {
    id: reset
    objectName: "buttonReset"
    //...
}
```

We can now access this button from the `main` function:

```cpp
if (engine.rootObjects().count() == 1) {
    QObject *window = engine.rootObjects()[0];
    QObject *resetButton = window->findChild<QObject*>("buttonReset");
    if (resetButton) {
        resetButton->setProperty("highlighted", true);
    }
}
```

In this code, we first access the top-level `Window` QML object. Then, we use the `findChild` method to find the object corresponding to our reset button. The `findChild()` method requires us to pass a class pointer as the template argument. Without knowing what class actually implements a given type, it is safest to simply pass `QObject*` as, once again, we know all QML objects inherit it. It is more important what gets passed as the function argument value—it is the name of the object we want returned. Note that it is not the `id` of the object but the value of the `objectName` property. When the result gets assigned to the variables, we verify whether the item has been successfully found and if that is the case, the generic `QObject` API is used to set its `highlighted` property to `true`. This property will change the appearance of the button.

The `QObject::findChild` and `QObject::findChildren` functions perform recursive search with unlimited depth. While they're easy to use, these functions may be slow if the object has many children. To improve performance, you can turn off recursive search by passing the `Qt::FindDirectChildrenOnly` flag to these functions. If the target object is not a direct child, consider calling `QObject::findChild` repeatedly to find each intermediate parent.

If you need to create a new QML object, you can use the `QQmlComponent` class for that. It accepts a QML document and allows you to create a QML object from it. The document is usually loaded from a file, but you can even provide it directly in C++ code:

```cpp
QQmlComponent component(&engine);
component.setData(
    "import QtQuick 2.6\n"
    "import QtQuick.Controls 2.2\n"
    "import QtQuick.Window 2.2\n"
    "Window { Button { text: \"C++ button\" } }", QUrl());
QObject* object = component.create();
object->setProperty("visible", true);
```

The `component.create()` function instantiates our new component and returns a pointer to it as `QObject`. In fact, any QML object derives from `QObject`. You can use Qt meta-system to manipulate the object without needing to cast it to a concrete type. The object's properties can be accessed using the `property()` and `setProperty()` functions. In this example, we set the `visible` property of the `Window` QML object to `true`. When our code is executed, a new window with a button will appear on screen.

You can also call the object's methods using the `QMetaObject::invokeMethod()` function:

```cpp
QMetaObject::invokeMethod(object, "showMaximized");
```

If you want to embed a new object into the existing QML form, you need to set *visual parent* of the new object. Let's say that we want to add a button to the calculator's form. First, you need to assign `objectName` to it in `main.qml`:

```cpp
Calculator {
    anchors.fill: parent
    objectName: "calculator"
}
```

You can now add a button to this form from C++:

```cpp
QQmlComponent component(&engine);
component.setData(
    "import QtQuick 2.6\n"
    "import QtQuick.Controls 2.2\n"
    "Button { text: \"C++ button2\" }", QUrl());
QObject *object = component.create();
QObject *calculator = window->findChild<QObject*>("calculator");
object->setProperty("parent", QVariant::fromValue(calculator));
```

In this code, we create a component and assign the main form as its `parent` property. This will make the object appear in the top-left corner of the form. Like with any other QML object, you can use the `anchors` property group to change position of the object.

When creating complex objects, it takes time for them to instantiate and at times, it is desired to not block the control flow for too long by waiting for the operation to complete. In such cases, you can create an object in the QML engine asynchronously using the `QQmlIncubator` object. This object can be used to schedule instantiation and continue the flow of the program. We can query the state of the incubator and when the object is constructed, we will be able to access it. The following code demonstrates how to use the incubator to instantiate an object and keep the application responding while waiting for the operation to complete:

```cpp
QQmlComponent component(&engine,
    QUrl::fromLocalFile("ComplexObject.qml"));
QQmlIncubator incubator;
component.create(incubator);
while(!incubator.isError() && !incubator.isReady()) {
    QCoreApplication::processEvents();
}
QObject *object = incubator.isReady() ? incubator.object() : 0;
```

# Bringing life into static user interfaces

Our user interface has been quite static until now. In this section, we will add a simple animation to our calculator. When the user selects the factorial operation, the second (unused) text field will fade out. It will fade in when another operation is selected. Let's see how QML allows us to implement that.

# Fluid user interfaces

So far, we have been looking at graphical user interfaces as a set of panels embedded one into another. This is well reflected in the world of desktop utility programs composed of windows and subwindows containing mostly static content scattered throughout a large desktop area where the user can use a mouse pointer to move around windows or adjust their size.

However, this design doesn't correspond well with modern user interfaces that often try to minimize the area they occupy (because of either a small display size like with embedded and mobile devices or to avoid obscuring the main display panel like in games), at the same time providing rich content with a lot of moving or dynamically resizing items. Such user interfaces are often called "fluid", to signify that they are not formed as a number of separate different screens but contain dynamic content and layout where one screen fluently transforms into another. The `QtQuick` module provides a runtime to create rich applications with fluid user interfaces.

# States and transitions

Qt Quick introduces a concept of **states**. Any Qt Quick object can have a predefined set of states. Each state corresponds to a certain situation in the application logic. For example, we can say that our calculator application has two states:

*   When add or multiply operations are selected, the user has to input two operands
*   When factorial operation is selected, the user has to input only one operand

States are identified by `string` names. Implicitly, any object has the base state with an empty name. To declare a new state, you need to specify the state name and a set of property values that are different in that state, compared to the base state.

Each Qt Quick object also has the `state` property. When you assign a state name to this property, the object goes to the specified state. This happens immediately by default, but it's possible to define **transitions** for the object and perform some visual effects when changing states.

Let's see how we can utilize states and transitions in our project.

# Time for action – Adding states to the form

Open the `CalculatorForm.ui.qml` file in the form editor. The bottom part of the main area contains the states editor. The base state item is always present on the left. Click on the Add a new state button on the right of the states editor. A new state will appear in the editor. It contains a text field that you can use to set the state's name. Set the name to `single_argument`.

Only one of the states can be selected at a time. When a custom state is selected, any changes in the form editor will only affect the selected state. When the base state is selected, you can edit the base state and all the changes will affect all other states unless the changed property is overridden in some state.

Select the `single_argument` state by clicking on it in the state editor. It will also be automatically selected upon creation. Next, select the `argument2` text field and set its `opacity` property to 0\. The field will become completely transparent, except for the blue outline provided by the form editor. However, this change only affects the `single_argument` state. When you switch to the base state, the text field will become visible. When you switch back to the second state, the text field will become invisible again.

You can switch to the text editor to see how this state is represented in the code:

```cpp
states: [
    State {
        name: "single_argument"
        PropertyChanges {
            target: b
            opacity: 0
        }
    }
]
```

As you can see, the state does not contain a full copy of the form. Instead, it only records the difference between this state and the base state.

Now we need to ensure that the form's state is properly updated. You just need to bind the `state` property of the form to a function that returns the current state. Switch to the `Calculator.qml` file and add the following code:

```cpp
CalculatorForm {
 state: {
 if (operationFactorial.checked) {
 return "single_argument";
 } else {
 return "";
 }
 }
    //...
}
```

As with any other property binding, the QML engine will automatically update the value of the `state` property when needed. When the user selects the factorial operation, the code block will return `"single_argument"`, and the second text field will be hidden. In other cases, the function will return an empty string that corresponds to the base state. When you run the application, you should be able to see this behavior.

# Time for action – Adding smooth transition effect

Qt Quick allows us to easily implement smooth transition between states. It will automatically detect when some property needs to be changed, and if there is a matching animation attached to the object, that animation will take over the process of applying the change. You don't even need to specify the starting and ending values of the animated property; it's all done automatically.

To add a smooth transition to our form, add the following code to the `Calculator.qml` file:

```cpp
CalculatorForm {
    //...
    transitions: Transition {
        PropertyAnimation {
            property: "opacity"
            duration: 300
        }
    }
}
```

Run the application and you will see that the text field's opacity changes gradually when the form transitions to another state.

# What just happened?

The `transitions` property holds the list of `Transition` objects for this object. It's possible to specify a different `Transition` object for each pair of states if you want to perform different animations in different cases. However, you can also use a single `Transition` object that will affect all transitions. For convenience, QML allows us to assign a single object to a property that expects a list.

A `Transition` object must contain one or multiple animations that will be applied during this transition. In this example, we added `PropertyAnimation` that allows us to animate any property of any child object of the main form. The `PropertyAnimation` QML type has properties that allow you to configure what exactly it will do. We instructed it to animate the `opacity` property and take 300 ms to perform the animation. The opacity change will be linear by default, but you can use the `easing` property to select another easing function.

As always, the Qt documentation is a great source of detailed information about available types and properties. Refer to Transition QML Type and Animation QML Type documentation pages for more information. We will also talk more about states and transitions in [Chapter 13](bf16fe2f-f507-4980-96cd-9b53b200522e.xhtml),* Animations in Qt Quick Games*.

# Have a go hero – Adding an animation of the item's position

You can make the calculator's transition even more appealing if you make the text field fly away off screen while fading out. Just use the form editor to change the text field's position in the `single_argument` state, and then attach another `PropertyAnimation` to the `Transition` object. You can play with different easing types to see which looks better for this purpose.

# Pop quiz

Q1\. Which property allows you to position a QML object relative to another object?

1.  `border`
2.  `anchors`
3.  `id`

Q2\. Which file name extension indicates that the file cannot be loaded into a QML engine?

1.  `.qml`
2.  `.ui`
3.  `.ui.qml`
4.  All of the above are valid QML files

Q3\. What is a Qt Quick transition?

1.  A change of parent-child relationships among the existing Qt Quick objects
2.  A set of properties that change when an event occurs
3.  A set of animations that play when the object's state changes

# Summary

In this chapter, you were introduced to with a declarative language called QML. The language is used to drive Qt Quick—a framework for highly dynamic and interactive content. You learned the basics of Qt Quick—how to create documents with a number of element types and how to create your own in QML, or in C++. You also learned how to bind expressions to properties to automatically reevaluate them. You saw how to expose the C++ core of your application to QML-based user interfaces. You learned to use the visual form editor and how to create animated transitions in the interface.

You also learned which QML modules are available. You were shown how to use the `QtQuick.Controls` and `QtQuick.Layouts` modules to build the application's user interface out of standard components. In the next chapter, we will see how you can make your own fully customized QML components with a unique look and feel. We will show how to implement custom graphics and event handling in QML applications.