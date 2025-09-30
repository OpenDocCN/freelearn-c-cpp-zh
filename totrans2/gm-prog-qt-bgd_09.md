# Chapter 9. Qt Quick Basics

> *In this chapter, you are going to be introduced to a technology called Qt Quick that allows us to implement resolution-independent user interfaces with lots of eye-candy, animations, and effects that can be combined with regular Qt code that implements the logic of the application. You will learn the basics of the QML declarative language that forms the foundation of Qt Quick. Using this language, you can define fancy graphics and animations, make use of particle engines, and structure your code using finite state machines. Pure QML code can be complemented with JavaScript or C++ logic in a manner similar to what you have learned in the previous chapter. By the end of this chapter, you should have enough knowledge to quickly implement fantastic 2D games with custom graphics, moving elements, and lots of visual special effects.*

# Fluid user interfaces

So far, we have been looking at graphical user interfaces as a set of panels embedded one into another. This is well-reflected in the world of desktop utility programs composed of windows and subwindows containing mostly static content scattered throughout a large desktop area where the user can use a mouse pointer to move windows around or adjust their size. However, this design doesn't correspond well with modern user interfaces that often try to minimize the area they occupy (because of either a small display size like with embedded and mobile devices or to avoid obscuring the main display panel like in games), at the same time providing rich content with a lot of moving or dynamically resizing items. Such user interfaces are often called "fluid" to signify that they are not formed as a number of separate different screens, but rather contain dynamic content and layout where one screen fluently transforms into another. Part of Qt 5 is the Qt Quick (Qt User Interface Creation Kit) module, which provides a runtime to create rich applications with fluid user interfaces. It builds upon a two-dimensional hardware accelerated canvas that contains a hierarchy of interconnected items.

# Declarative UI programming

Although it is technically possible to use Qt Quick by writing C++ code, the module is accompanied by a dedicated programming language called **QML** (**Qt Modeling Language**). QML is an easy to read and understand declarative language that describes the world as a hierarchy of components that interact and relate to one another. It uses a JSON-like syntax and allows us to use imperative JavaScript expressions as well as dynamic property bindings. So, what is a declarative language, anyway?

Declarative programming is one of the programming paradigms that dictates that the program describes the logic of the computation without specifying how this result should be obtained. In contrast to imperative programming, where the logic is expressed as a list of explicit steps forming an algorithm that directly modifies the intermediate program state, a declarative approach focuses on what the ultimate result of the operation should be.

We use QML by creating one or more QML documents where we define hierarchies of objects. Each document is composed of two sections.

You can follow every example we explain in Qt Creator by creating a new Qt Quick UI project and placing the presented code in the QML file created for you. The details about using this project type will be described in a later section of this chapter.

### Tip

If you can't see a **Qt Quick UI** project in the Creator's wizard, you have to enable a plugin called `QmlProjectManager` by choosing the **About Plugins** entry from the Creator's **Help** menu, then scrolling down to the **QtQuick** section, and making sure the **QmlProjectManager** entry is checked. If it is not, check it and restart Creator:

![Declarative UI programming](img/8874OS_09_29.jpg)

The first section contains a series of `import` statements that define the range of components that can be used in a particular document. In its simplest form, each statement consists of the `import` keyword followed by the module URI (name) and the module version to import. The following statement imports the `QtQuick` module in version 2.1:

[PRE0]

The second section contains a definition of a hierarchy of objects. Each object declaration consists of two parts. First, you have to specify the type of the object and then follow it with the detailed definition enclosed in braces. Since the detailed definition can be empty, the simplest object declaration is similar to the following:

[PRE1]

This declares an instance of the `Item` element, which is the most basic Qt Quick element and represents an abstract item of the user interface without any visual appearance.

## Element properties

Each element type in QML defines a number of properties. Values for these properties can be set as part of the detailed definition of an object. The `Item` type brings a number of properties for specifying the geometry of an item:

[PRE2]

`Item` is a very interesting and useful element, but since it is totally transparent, we will now focus on its descendant type that draws a filled rectangle. This type is called `Rectangle`. It has a number of additional properties, among them, the `color` property for specifying the fill color of the rectangle. To define a red square, we could write the following code:

[PRE3]

The problem with this code is that if we ever decide to change the size of the square, we will have to update values for the two properties separately. However, we can use the power of the declarative approach and specify one of the properties as a relation to the other properties:

[PRE4]

This is called **property** **binding**. It differs from a regular value assignment and binds the value of height to the value of width. Whenever width changes, height will reflect that change in its own value.

Note that the order of statements in the definition does not matter as you declare relations between properties. The following declaration is semantically identical to the previous one:

[PRE5]

You can bind a property not only to a value of another property, but also to any JavaScript statement that returns a value. For example, we can declare rectangle color to be dependent on the proportions between the width and the height of the element by using a ternary conditional expression operator:

[PRE6]

Whenever `width` or `height` of the object changes, the statement bound to the `color` property will be re-evaluated and if `width` of the rectangle is larger than its `height`, the rectangle will become `red`; otherwise, it will be `blue`.

Property binding statements can also contain function calls. We can extend the `color` declaration to use a different color if the rectangle is a square by using a custom function:

[PRE7]

QML does its best to determine when the function value may change, but it is not omnipotent. For our last function, it can easily determine that the function result depends on the values of the `width` and `height` properties, so it will re-evaluate the binding if either of the two values change. However, in some cases, it can't know that a function might return a different value next time it is called, and in such situations, the statement will not be re-evaluated. Consider the following function:

[PRE8]

Binding the `color` property to the result of that function will not work properly. QML will only call this function once when the object is initialized, and it will never call it again. This is because it has no way of knowing that the value of this function depends on the current time. Later, we will see how to overcome this with a bit of imperative code and a timer.

## Group properties

The `Rectangle` element allows us to define not only the fill color but also the outline size and color. This is done by using the `border.width` and `border.color` properties. You can see they have a common prefix followed by a dot. This means these properties are subproperties of a property group `border`. There are two ways to bind values to these properties. The first approach is to use the dot notation:

[PRE9]

An alternative approach, which is especially useful if you want to set a large number of subproperties in a single group, is to use brackets to enclose definitions in a group:

[PRE10]

## Object hierarchies

We said that QML is about defining object hierarchies. You do this in the simplest way possible–by putting one object declaration into another object's declaration. To create a button-like object containing a rounded frame with some text inside, we'll combine a `Rectangle` item with a `Text` item:

[PRE11]

### Note

You can use a semicolon instead of newlines to separate statements in QML in order to have a more compact object definition at the cost of decreased readability.

Running this code produces a result similar to the following diagram:

![Object hierarchies](img/8874OS_09_01.jpg)

As we can see, it does not look good–the frame is not big enough to hold the text and so it flows outside the frame. Moreover, the text is positioned incorrectly.

Unlike widgets where a child widget is clipped to its parent's geometry, Qt Quick items can be positioned outside their parents.

Since we didn't specify the *x* and *y* coordinates of the text, they are set to their default value, which is `0`. As a result, the text is pinned to the top-left corner of the frame and flows outside the right edge of the frame.

To correct this behavior, we can bind the width of the frame to the width of the text. To do this in the property binding for the rectangle width, we have to specify that we want to use the width of the text object. QML provides a pseudo-property called `id` to allow the programmer to name objects. Let's provide an ID for the `Text` element and bind the width of the outside object to the width of the text, and also do the same for the height. At the same time, let's reposition the text a little to provide padding for the four pixels between the frame and the text itself:

[PRE12]

As you can see in the following image, such code works, but it is still problematic:

![Object hierarchies](img/8874OS_09_02.jpg)

If you set empty text to the internal element, the rectangle width and height will drop to `8`, which does not look good. It will also look bad if the text is very long:

![Object hierarchies](img/8874OS_09_03.jpg)

Let's complicate matters even more and add an icon to the button by adding another child element to the rectangle. Qt Quick provides an `Image` type to display images, so let's use it to position our icon on the left side of the text:

[PRE13]

In this code, we used the `Math.max` function available in JavaScript to calculate the height of the button, and we modified definitions of the *y* properties of the internal objects to center them vertically in the button. The source property of `Image` contains the URL of a file containing the image to be shown in the item.

### Note

The URL can point not only to a local file, but also to a remote HTTP resource. In such an event, if the remote machine is reachable, the file will be fetched from the remote server automatically.

The result of the code can be seen in the following image:

![Object hierarchies](img/8874OS_09_04.jpg)

Calculating the positions of each internal element as well as the size of the button frame is becoming complicated. Fortunately, we don't have to do it since Qt Quick provides a much better way of managing item geometry by attaching certain points of some objects to points of another object. These points are called anchor lines. The following anchor lines are available to each Qt Quick item:

![Object hierarchies](img/8874OS_09_05.jpg)

You can establish bindings between anchor lines to manage relative positioning of items. Each anchor line is represented by two properties–one that can be bound to something and another to bind from. Anchors to bind to are regular properties of the object. They can serve as binding arguments for properties defined in an anchors property group. Therefore, to bind the "left" anchor of the current object to the "right" anchor of the object `otherObject`, one would write:

[PRE14]

In addition to specifying an arbitrary number of anchor bindings, we can also set margins for each of the anchors (or for all of them at once) to offset the two bound anchor lines. Using anchors, we can simplify the previous button definition:

[PRE15]

You can see the `button` ID is not used anymore. Instead, we use parent, which is a property that always points to the item's parent.

# Time for action – creating a button component

As an exercise, let's try to use what you've learned so far to create a more complete and better working button component. The button is to have a rounded shape with a nice background and should hold definable text and an icon. The button should look good for different texts and icons.

Start by creating a new project in Qt Creator. Choose **Qt Quick UI** as the project type. When asked for the component set, choose the lowest available version of Qt Quick:

![Time for action – creating a button component](img/8874OS_09_06.jpg)

At this point, you should end up with a project containing two files–one with a QML project extension, which is your project management file, and the other with the QML extension, which is your main user interface file. You can see that both files contain QML definitions. That is because Qt Creator manages Qt Quick projects using QML itself (you'll notice it imports the `QmlProject` module).

The QML document that was created for us contains a "Hello World" example code, which we can use as a starting point in our Qt Quick experiments. If you go to the **Projects** pane and look at the **Run Configuration** for the project, you will notice that it uses something called QML Scene to run your project. This configuration invokes an external application called `qmlscene` that is able to load and display an arbitrary QML document. If you run the example code, you should see a white window with some text centered in it. If you click anywhere in the window, the application will close.

Let's start by creating the button frame. Replace the `Text` item with a `Rectangle` item. You can see that the text is centered in the window by using a `centerIn` anchor binding that we didn't mention before. This is one of two special anchors that are provided for convenience to avoid having to write too much code. Using `centerIn` is equivalent to setting both `horizontalCenter` and `verticalCenter`. The other convenience binding is `fill`, which makes one item occupy the whole area of another item (similar to setting the left, right, top, and bottom anchors to their respective anchor lines in the destination item).

Let's give a basic look and feel to the button panel by setting some of its basic properties. This time, instead of setting a solid color for the button, we will declare the background to be a linear gradient. Replace the `Text` definition with the following code:

[PRE16]

After running the project, you should see a result similar to the following image:

![Time for action – creating a button component](img/8874OS_09_07.jpg)

## *What just happened?*

We bound a `Gradient` element to the gradient property and defined two `GradientStop` elements as its children, where we specified two colors to blend between. `Gradient` does not inherit from `Item` and thus is not a visual Qt Quick element. Instead, it is just an object that serves as a data holder for the gradient definition.

The `Item` type has a property called `children` that contains a list of visual children (`Item` instances) of an item and another property called `resources`, which contains a list of non-visual objects (such as `Gradient` or `GradientStop`) for an item. Normally, you don't need to use these properties when adding visual or non-visual objects to an item as the item will automatically assign child objects to appropriate properties. Note that in our code, the `Gradient` object is not a child object of the `Rectangle`; it is just assigned to its `gradient` property.

# Time for action – adding button content

The next step is to add text and an icon to the button. We will do this by using another item type called `Row`, as shown:

[PRE17]

You'll get the following output:

![Time for action – adding button content](img/8874OS_09_08.jpg)

## *What just happened?*

`Row` is one out of four positioner types (the others being `Column`, `Grid`, and `Flow`) that spreads its children in a horizontal row. It makes it possible to position a series of items without using anchors. `Row` has a spacing property that dictates how much space to leave between items.

# Time for action – sizing the button properly

Our current panel definition still doesn't behave well when it comes to sizing the button. If the button content is very small (for example, the icon doesn't exist or the text is very short), the button will not look good. Typically, push buttons enforce a minimum size–if the content is smaller than a specified size, the button will be expanded to the minimum size allowed. Another problem is that the user might want to override the width or height of the item. In such cases, the content of the button should not overflow past the border of the button. Let's fix these two issues by replacing the `width` and `height` property bindings with the following code:

[PRE18]

## *What just happened?*

The `implicitWidth` and `implicitHeight` properties can contain the desired size the item wants to have. It's a direct equivalent of `sizeHint()` from the widget world. By using these two properties instead of `width` and `height` (which are bound to `implicitWidth` and `implicitHeight` by default), we allow the user of our component to override those implicit values. When this happens and the user does not set the width or height big enough to contain the icon and text of the button, we prevent the content from crossing the boundaries of the parent item by setting the `clip` property to `true`.

# Time for action – making the button a reusable component

So far, we have been working on a single button. Adding another button by copying the code, changing the identifiers of all components, and setting different bindings to properties are very tedious tasks. Instead, we can make our button item a real component, that is, a new QML type that can be instantiated on demand as many times as required.

First, position the text cursor right before the bracket opening of the definition of the button and press *Alt* + *Enter* on the keyboard to open the refactoring menu, like in the following screenshot:

![Time for action – making the button a reusable component](img/8874OS_09_09.jpg)

From the menu, choose **Move Component into Separate File**. In the popup, type in a name for the new type (for example, `Button`) and accept the dialog by clicking on the **OK** button:

![Time for action – making the button a reusable component](img/8874OS_09_10.jpg)

## *What just happened?*

You can see that we have a new file called `Button.qml` in the project, which contains everything the button item used to have. The main file was simplified to something similar the following:

[PRE19]

`Button` has become a component–a definition of a new type of element that can be used the same way as element types imported into QML. Remember that QML component names as well as names of files representing them need to begin with a capital letter! If you name a file "button.qml" instead of "Button.qml", then you will not be able to use "Button" as a component name, and trying to use "button" instead will result in an error message. This works both ways–every QML file starting with a capital letter can be treated as a component definition. We will talk more about components later.

# Event handlers

Qt Quick is meant to be used for creating user interfaces that are highly interactive. It offers a number of elements for taking input events from the user.

## Mouse input

The simplest of all of them is `MouseArea`. It defines a transparent rectangle that exposes a number of properties and signals related to mouse input. Commonly used signals include clicked, pressed, and released. Let's do a couple of exercises to see how the element can be used.

# Time for action – making the button clickable

Thus far, our component only looks like a button. The next task is to make it respond to mouse input. As you may have guessed, this is done by using the `MouseArea` item.

Add a `MouseArea` child item to the button and use anchors to make it fill the whole area of the button. Call the element `buttonMouseArea`. Put the following code in the body of the item:

[PRE20]

In addition to this, set the following declaration in the button object just after its ID is declared:

[PRE21]

To test the modification, add the following code at the end of the button object definition, just before the closing bracket:

[PRE22]

Then, run the program and click on the button. You'll see your message printed to the Creator's console. Congratulations!

## *What just happened?*

With a signal `clicked()` statement, we declared that the button object emits a signal called clicked. With the `MouseArea` item, we defined a rectangular area (covering the whole button) that reacts to mouse events. Then, we defined `onClicked`, which is a signal handler. For every signal an object has, a script can be bound to a handler named like the signal and prefixed with "on"; hence, for the clicked signal, the handler is called `onClicked` and for `valueChanged` it is called `onValueChanged`. In this particular case, we have two handlers defined–one for the button where we write a simple statement to the console, and the other for the `MouseArea` element where we call the button's signal function effectively emitting that signal.

`MouseArea` has even more features, so now let's try putting them to the right use to make our button more feature-rich.

# Time for action – visualizing button states

Currently, there is no visual reaction to clicking on the button. In the real world, the button has some depth and when you push it and look at it from above, its contents seems to shift a little toward the right and downward. Let's mimic this behavior by making use of the pressed property `MouseArea` has, which denotes whether the mouse button is currently being pressed (note that the pressed property is different from the pressed signal that was mentioned earlier). The content of the button is represented by the `Row` element, so add the following statements inside its definition:

[PRE23]

We can also make the text change color when the mouse cursor hovers over the button. For this, we have to do two things. First, let's enable receiving hover events on the `MouseArea` by settings its `hoverEnabled` property:

[PRE24]

When this property is set, `MouseArea` will be setting its `containsMouse` property to `true` whenever it detects the mouse cursor over its own area. We can use this value to set the text color:

[PRE25]

## *What just happened?*

In the last exercise, we learned to use some properties and signals from `MouseArea` to make the button component more interactive. However, the element is much richer in features. In particular, if hover events are enabled, you can get the current mouse position in the item's local coordinate system through the `mouseX` and `mouseY` properties that return values. The cursor position can also be reported by handling the `positionChanged` signal. Speaking of signals, most `MouseArea` signals carry a `MouseEvent` object as their argument. This argument is called `mouse` and contains useful information about the current state of the mouse, including its position and buttons currently pressed:

[PRE26]

# Time for action – notifying the environment about button states

We have added some code to make the button look more natural by changing its visual aspects. Now, let's extend the button programming interface so that developers can use more features of the button.

The first thing we can do is make button colors definable by introducing some new properties for the button. Let's put the highlighted code at the beginning of the button component definition:

[PRE27]

Then, we'll use the new definitions for the background gradient:

[PRE28]

Now for the text color:

[PRE29]

Also, please notice that we used the `pressed` property of `MouseArea` to detect whether a mouse button is currently being pressed on the area. We can equip our button with a similar property. Add the following code to the `Button` component:

[PRE30]

## *What just happened?*

The first set of changes introduced four new properties defining four colors that we later used in statements defining gradient and text colors for the button. In QML, you can define new properties for objects with the `property` keyword. The keyword should be followed by the property type and property name. QML understands many property types, the most common being int, real, string, font, and color. Property definitions can contain an optional default value for the property preceded with a colon. The situation is different with the pressed property definition. You can see that for the property type, the definition contains the word alias. It is not a property type, but rather an indicator that the property is really an alias to another property–each time the pressed property of the button is accessed, the value of the `buttonMouseArea.pressed` property is returned, and every time the property is changed, it is the mouse area's property that really gets changed. With a regular property declaration, you can provide any valid expression as the default value because the expression is bound to the property. With a property alias, it is different–the value is mandatory and has to be pointing to an existing property of the same or an other object. You can treat property aliases in a similar way as references in C++.

Consider the following two definitions:

[PRE31]

At first glance, they are similar as they point to the same property and therefore, the values returned for the properties are the same. However, the properties are really very different, which becomes apparent if you try to modify their values:

[PRE32]

The first property actually has an expression bound to it, so assigning `7` to `foo` simply releases the binding and assigns the value `7` to the `foo` property, leaving `someobject.prop` with its original value. The second statement, however, acts like a C++ reference; therefore, assigning a new value applies the modification to the someobject.prop property the alias is really pointing to.

Speaking of properties, there is an easy way to react when a property value is modified. For each existing property, there is a handler available that is executed whenever the property value is modified. The handler name is `on` followed by the property name, then followed by the word `Changed`, all in camel case–thus, for a foo property, it becomes `onFooChanged` and for `topColor`, it becomes `onTopColorChanged`. To log the current press state of the button to the console, all we need to do is implement the property change handler for this property:

[PRE33]

## Touch input

As mentioned earlier, `MouseArea` is the simplest of input event elements. Nowadays, more and more devices have touch capabilities and Qt Quick can handle them, as well. Currently, we have three ways of handling touch input.

First of all, we can keep using `MouseArea` as simple touch events are also reported as mouse events; therefore, tapping and sliding a finger on the screen is supported out-of-the-box. The following exercise works on touch-capable devices, as well, when using mouse as input.

# Time for action – dragging an item around

Create a new `Qt Quick UI` project. Modify the default code by discarding the existing child items and adding a circle instead:

[PRE34]

Next, use the `drag` property of `MouseArea` to enable moving the circle by touch (or mouse):

[PRE35]

Then, you can start the application and begin moving the circle around.

## *What just happened?*

A circle was created by defining a rectangle with its height equal to width, making it a square and rounding the borders to half the side length. The `drag` property can be used to tell `MouseArea` to manage a given item's position using input events flowing into the area element. We denote the item to be dragged by using the target subproperty. You can use other subproperties to control the axis the item is allowed to move along or constrain the move to a given area. An important thing to remember is that the item being dragged cannot be anchored for the axis on which the drag is requested; otherwise, the item will respect the anchor and not the drag. We didn't anchor our circle item at all since we want it to be draggable along both axes.

The second approach to handling touch input in Qt Quick applications is to use `PinchArea`, which is an item similar to `MouseArea`, but rather than dragging an item around, it allows you to rotate or scale it using two fingers (with a so called "pinch" gesture), as shown. Be aware that `PinchArea` reacts only to touch input, so to test the example you will need a real touch capable device.

![What just happened?](img/8874OS_09_11.jpg)

# Time for action – rotating and scaling a picture by pinching

Start a new `Qt Quick UI` project. In the QML file, delete everything but the external item. Then, add an image to the UI and make it centered in its parent:

[PRE36]

Now, we will add a `PinchArea` element. This kind of item can be used in two ways–either by manually implementing signal handlers `onPinchStarted`, `onPinchUpdated`, and `onPinchFinished` to have total control over the functionality of the gesture or by using a simplified interface similar to the drag property of `MouseArea`. Since the simplified interface does exactly what we want, there is no need to handle pinch events manually. Let's add the following declaration to the file:

[PRE37]

You'll get an output similar to the following screenshot:

![Time for action – rotating and scaling a picture by pinching](img/8874OS_09_12.jpg)

## *What just happened?*

Our simple application loads an image and centers it in the view. Then, there is a `PinchArea` item filling the view area that is told to operate on the image object. We define the range of the scaling and rotating of the item. The rest is left to the `PinchArea` item itself. If you start interacting with the application, you will see the item rotate and scale. What really happens behind the scenes is that `PinchArea` modifies the values of the two properties each Qt Quick item has–`rotation` and `scale`.

### Note

`PinchArea` can also control the dragging of the item with `pinch.dragAxis`, just like `MouseArea` does with drag, but for simplicity, we didn't use this part of the API. Feel free to experiment with it in your own code.

## Have a go hero – rotating and scaling with a mouse

Of course, you don't have to use `PinchArea` to rotate or scale an item. Properties controlling those aspects are regular properties that you can read and write at any time. Try replacing `PinchArea` with `MouseArea` to obtain a result similar to what we just did by modifying the scale and rotation properties as a result of receiving mouse events–when the user drags the mouse while pressing the left button, the image is scaled and when the user does the same while pressing the right button, the image is rotated. You can control which buttons trigger mouse events by manipulating the `acceptedButtons` property (setting it to `Qt.LeftButton|Qt.RightButton` will cause both buttons to trigger events). The button that triggers the event is reported in the event object (which is called `mouse`) through its `button` property, and the list of all buttons currently pressed is available in the `button` property:

[PRE38]

If you manage to do the task, try replacing `MouseArea` with `PinchArea` again, but then instead of using the `pinch` property, handle events manually to obtain the same effect (the event object is called `pinch` and has a number of properties you can play with).

A third approach to handling touch input is by using the `MultiPointTouchArea` item. It provides a low-level interface to gestures by reporting each touch point separately. It can be used to create custom high-level gesture handlers similar to `PinchArea`.

## Keyboard input

So far, we've been dealing with pointer input, but user input is not just that–we can also handle keyboard input. This is quite simple and basically boils down to two easy steps.

First, you have to enable receiving keyboard events by stating that a particular item has keyboard focus:

[PRE39]

Then, you can start handling events by writing handlers in a similar fashion as for mouse events. However, `Item` doesn't provide its own handler for manipulating keys that is a counterpart for `keyPressEvent` and `keyReleaseEvent` of `QWidget`. Instead, adequate handlers are provided by the `Keys` attached property.

Attached properties are provided by elements that are not used as stand-alone elements but instead provide properties to other objects by getting attached to them. This is a way of adding support for new properties without modifying the API of the original element (it doesn't add new properties through an **is-a** relation, but rather through a **has-a** one). Each object that references an attached property gets its own copy of the attaching object that then handles the extra properties. We will come back to attached properties later in this chapter. For now, you just need to remember that in certain situations, an element can obtain additional properties that are not part of its API.

Let's go back to implementing event handlers for keyboard input. As we said earlier, each Item has a `Keys` attached property that allows us to install our own keyboard handlers. The basic two signals `Keys` adds to `Item` are pressed and released; therefore, we can implement the `onPressed` and `onReleased` handlers that have a `KeyEvent` argument providing similar information as `QKeyEvent` in the widget world. As an example, we can see an item that detects when a spacebar was pressed:

[PRE40]

It might become problematic if you want to handle many different keys in the same item as the `onPressed` handler would likely contain a giant switch section with branches for every possible key. Fortunately, `Keys` offers more properties. Most of the commonly used keys (but not letters) have their own handlers that are called when the particular key is pressed. Thus, we can easily implement an item that takes a different color depending on which key was pressed last:

[PRE41]

Please note that there is still a single released signal even if a key has its own pressed signal.

Now, consider another example:

[PRE42]

We would expect that when we press and hold the spacebar, we will see the text change from `0` to `1` and stay on that value until we release the key. If you run the example, you will see that instead, the number keeps incrementing as long as you hold down the key. This is because by default, the keys auto-repeat–when you hold the key, the operating system keeps sending a sequence of press-release events for the key (you can verify that by adding the `console.log()` statements to the `Keys.onPressed` and `Keys.onReleased` handlers). To counter this effect, you can either disable key repeats in your system (which will, of course, not work if someone installs your program on his or her own computer) or you can differentiate between auto-repeat and regular events. In Qt Quick, you can do this easily as each key event carries the appropriate information. Simply replace the handler from the last example with the following one:

[PRE43]

The event variable we use here is the name of the parameter of the `spacePressed` signal. As we cannot declare our own names for the parameters like we can do in C++, for each signal handler you will have to look up the name of the argument in the documentation, as shown:

![Keyboard input](img/8874OS_09_13.jpg)

In standard C++ applications, we usually use the *Tab* key to navigate through focusable items. With games (and fluid user interfaces in general), it is more common to use arrow keys for item navigation. Of course, we can handle this situation by using the `Keys` attached property and adding `Keys.onRightPressed`, `Keys.onTabPressed`, and other signal handlers to each of our items where we want to modify the focus property of the desired item, but it would quickly clutter our code. Qt Quick comes to our help once again by providing a `KeyNavigation` attached property, which is meant to handle this specific situation and allows us to greatly simplify the needed code. Now, we can just specify which item should get into focus when a specific key is triggered:

[PRE44]

Notice that we made the first item get into focus in the beginning by explicitly setting the focus property.

Both the `Keys` and `KeyNavigation` attached properties have a way to define the order in which each of the mechanisms receive the events. This is handled by the priority property, which can be set to either `BeforeItem` or `AfterItem`. By default, `Keys` will get the event first (`BeforeItem`), then the internal event handling can take place and finally, `KeyNavigation` will have a chance of handling the event (`AfterItem`). Note that if the key is handled by one of the mechanisms, the event is accepted and the remaining mechanisms will not be able to handle that event.

## Have a go hero – practicing key-event propagation

As an exercise, you can expand our last example by building a larger array of items (you can use the `Grid` element to position them) and defining a navigation system that makes use of the `KeyNavigation` attached property. Have some of the items handle events themselves using the `Keys` attached property. See what happens when the same key is handled by both mechanisms. Try influencing the behavior using the priority property.

Apart from the attached properties we described, Qt Quick provides built-in elements for handling keyboard input. The two most basic types are `TextInput` and `TextEdit`, which are QML equivalents of `QLineEdit` and `QTextEdit`. The former are used for single-line text input, while the latter serve as its multi-line counterpart. They both offer cursor handling, undo-redo functionality, and text selections. You can validate text typed into `TextInput` by assigning a validator to the `validator` property. For example, to obtain an item where the user can input a dot-separated IP address, we could use the following declaration:

[PRE45]

The regular expression only verifies the format of the address. The user can still insert bogus numbers. You should either do a proper check before using the address or provide a more complex regular expression that will constrain the range of numbers the user can enter.

One thing to remember is that neither `TextInput` nor `TextEdit` has any visual appearance (apart from the text and cursor they contain), so if you want to give the user some visual hint as to where the item is positioned, the easiest solution is to wrap it in a styled rectangle:

[PRE46]

Notice the highlighted code–the `clip` property of `textInput`–is enabled such that by default, if the text entered in the box doesn't fit in the item, it will overflow it and remain visible outside the actual item. By enabling clipping, we explicitly say that anything that doesn't fit the item should not be drawn.

![Have a go hero – practicing key-event propagation](img/8874OS_09_14.jpg)

# Using components in Qt Quick

By now, you should be familiar with the very basics of QML and Qt Quick. Now, we can start combining what you know and fill the gaps with more information to build a functional Qt Quick application. Our target is going to be to display an analog clock.

# Time for action – a simple analog clock application

Create a new `Qt Quick UI` project. To create a clock, we will implement a component representing the clock needle and we will use instances of that component in the actual clock element. In addition to this, we will make the clock a reusable component; therefore, we will create it in a separate file and instantiate it from within `main.qml`:

[PRE47]

Then, add the new QML file to the project and call it `Clock.qml`. Let's start by declaring a circular clock plate:

[PRE48]

If you run the program now, you'll see a plain gray circle hardly resembling a clock plate:

![Time for action – a simple analog clock application](img/8874OS_09_15.jpg)

The next step is to add marks dividing the plate into 12 sections. We can do this by putting the following declaration inside the `plate` object:

[PRE49]

Running the program should now give the following result, looking much more like a clock plate:

![Time for action – a simple analog clock application](img/8874OS_09_16.jpg)

## *What just happened?*

The code we just created introduces a couple of new features. Let's go through them one by one.

First of all, we used a new element called `Repeater`. It does exactly what its name says–it repeats items declared within it using a given model. For each entry in the model, it creates an instance of a component assigned to a property called `delegate` (the property name means that it contains an entity to which the caller delegates some responsibility, such as describing a component to be used as a stencil by the caller). `Item` declared in `Repeater` describes the delegate even though we cannot see it explicitly assigned to a property. This is because `delegate` is a default property of the `Repeater` type, which means anything unassigned to any property explicitly gets implicitly assigned to the default property of the type.

The `Item` type also has a default property called `data`. It holds a list of elements that gets automatically split into two "sublists"–the list of the item's children (which creates the hierarchy of `Item` instances in Qt Quick) and another list called resources, which contains all "child" elements that do not inherit from `Item`. You have direct access to all three lists which means calling `children[2]` will return the third `Item` element declared in the item, and `data[5]` will return the sixth element declared in the `Item` regardless of whether the given element is a visual item (that inherits `Item`) or not.

The model can be a number of things but in our case, it is simply a number denoting how many times the delegate should be repeated. The component to be repeated is a transparent item containing a rectangle. The item has a property declared called `hour` that has something called `index` bound to it. The latter is a property assigned by `Repeater` to each instance of the delegate component. The value it contains is the index of the instance in the `Repeater` object–since we have a model containing twelve elements, `index` will hold values within a range of `0` to `11`. The item can make use of the `index` property to customize instances created by `Repeater`. In this particular case, we use `index` to provide values for a `rotation` property and by multiplying the index by `30`, we get values starting from `0` for the first instance and ending at `330` for the last one.

The `rotation` property brings us to the second most important subject–item transformations. Each item can be transformed in a number of ways, including rotating the item and scaling it in two-dimensional space as we already mentioned earlier. Another property called `transformOrigin` denotes the origin point around which scale and rotation are applied. By default, it points to `Item.Center`, which makes the item scale and rotate around its center, but we can change it to eight other values such as `Item.TopLeft` for the top-left corner or `Item.Right` for the middle of the right edge of the item. In the code we crafted, we rotate each item clockwise around its bottom edge. Each item is positioned horizontally in the middle of the plate using the `plate.width/2` expression and vertically at the top of the plate with the default width of `0` and the height of half the plate's height; thus, each item is a thin vertical line spanning from within the top to the center of the plate. Then, each item is rotated around the center of the plate (each item's bottom edge) by 30 degrees more than a previous item effectively laying items evenly on the plate.

Finally, each item has a gray `Rectangle` attached to the top edge (offset by `4`) and horizontally centered in the transparent parent. Transformations applied to an item influence the item's children similarly to what we have seen in Graphics View; thus, the effective rotation of the rectangle follows that of its parent. The height of the rectangle depends on the value of `hour`, which maps to the index of the item in `Repeater`. Here, you cannot use `index` directly as it is only visible within the top-most item of the delegate. That's why we create a real property called `hour` that can be referenced from within the whole delegate item hierarchy.

### Note

If you want more control over item transformations, then we are happy to inform you that apart from rotation and scale properties, each item can be assigned an array of elements such as `Rotation`, `Scale`, and `Translate` to a property called `transform`, which are applied in order, one at a time. These types have properties for fine-grained control over the transformation. For instance, using `Rotation` you can implement rotation over any of the three axes and around a custom origin point (instead of being limited to nine predefined origin points as when using the `rotation` property of `Item`).

# Time for action – adding needles to the clock

The next step is to add the hour, minute, and second needles to the clock. Let's start by creating a new component called `Needle` in a file called `Needle.qml` (remember that component names and files representing them need to start with a capital letter):

[PRE50]

`Needle` is basically a rectangle anchored to the center of its parent by its bottom edge, which is also the item's pivot. It also has `value` and `granularity` properties driving the rotation of the item, where `value` is the current value the needle shows and `granularity` is the number of different values it can display. Also, anti-aliasing for the needle is enabled as we want the tip of the needle nicely rounded. Having such a definition, we can use the component to declare the three needles inside the clock plate object:

[PRE51]

The three needles make use the of `hours`, `minutes`, and `seconds` properties of clock, so these need to be declared, as well:

[PRE52]

By assigning different values to the properties of `Clock` in `main.qml`, you can make the clock show a different time:

[PRE53]

You'll get an output as shown:

![Time for action – adding needles to the clock](img/8874OS_09_18.jpg)

## *What just happened?*

Most `Needle` functionality is declared in the component itself, including geometry and transformations. Then, whenever we want to use the component, we declare an instance of `Needle` and optionally customize the `length` and `color` properties, as well as set its `value` and `granularity` to obtain the exact functionality we need.

# Time for action – making the clock functional

The final step in creating a clock is to make it actually show the current time. In JavaScript, we can query the current time using the `Date` object:

[PRE54]

Therefore, the first thing that comes to mind is to use the preceding code to show the current time on the clock:

[PRE55]

This will indeed show the current time once you start the application, but the clock will not be updating itself as the time passes. This is because `new Date()` returns an object representing one particular moment in time (the date and time at the moment when the object was instantiated). What we need instead is to have the `currentDate` property updated with a new object created as the current time changes. To obtain this effect, we can use a `Timer` element that is an equivalent of `QTimer` in C++ and lets us periodically execute some code. Let's modify the code to use a timer:

[PRE56]

## *What just happened?*

Based on the interval property, we can determine that the timer emits a `triggered` signal every 500 ms, causing `currentDate` to be updated with a new `Date` object representing the current time. The clock is also given a `running` property (pointing to its equivalent in the timer) that can control whether updates should be enabled. The timer is set to `repeat`; otherwise, it would trigger just once.

## Dynamic objects

To briefly sum up what you have learned so far, we can say that you know how to create hierarchies of objects by declaring their instances and you also know how to program new types in separate files, making definitions available as components to be instantiated in other QML files. You can even use the `Repeater` element to declare a series of objects based on a common stencil.

## Using components in detail

We promised to give you more information about components and now is the time to do so. You already know the basics of creating components in separate files. Every QML file beginning with a capital letter is treated as a component definition. This definition can be used directly by other QML files residing in the same directory as the component definition. If you need to access a component definition from a file residing elsewhere, you will have to first import the module containing the component in the file where you want to use it. The definition of a module is very simple–it is just a relative path to the directory containing QML files. The path is constructed using dots as the separator. This means that if you have a file named `Baz.qml` in a directory called `Base/Foo/Bar` and you want to use the `Baz` component from within the `Base/Foo/Ham.qml` file, you will have to put the following import statement in `Ham.qml`:

[PRE57]

If you want to use the same component from within the `Base/Spam.qml` file, you will have to replace the import statement with:

[PRE58]

Importing a module makes all its components available for use. You can then declare objects of types imported from a certain module.

## Creating objects on request

The problem with pre-declaring objects directly in a QML file is that you need to know up front how many objects you are going to need. More often than not, you will want to dynamically add and remove objects to your scene, for example, in an alien invasion game where as the player progresses, new alien saucers will be entering the game screen and other saucers will be getting shot down and destroyed. Also, the player's ship will be "producing" new bullets streaking in front of the ship, eventually running out of fuel or otherwise disappearing from the game scene. By putting a good amount of effort into the problem, you would be able to use `Repeater` to obtain this effect, but there is a better tool at hand.

QML offers us another element type called `Component`, which is another way to teach the engine about a new element type by declaring its contents in QML. There are basically two approaches to doing this.

The first approach is to declare a `Component` element instance in the QML file and inline the definition of the new type directly inside the element:

[PRE59]

Such code declares a component called `circleComponent` that defines a circle and exposes its `diameter`, `color`, and `border` properties.

The other approach is to load the component definition from an existing QML file. QML exposes a special global object called `Qt`, which provides a set of interesting methods. One of the methods allows the caller to create a component passing the URL of an existing QML document:

[PRE60]

An interesting note is that `createComponent` can not only accept a local file path but also a remote URL, and if it understands the network scheme (for example, `http`), it will download the document automatically. In this case, you have to remember that it takes time to do that, so the component may not be ready immediately after calling `createComponent`. Since the current loading status is kept in the `status` property, you can connect to the `statusChanged` signal to be notified when this happens. A typical code path looks similar to the following:

[PRE61]

If the component definition is incorrect or the document cannot be retrieved, the status of the object will change to `Error`. In that case, you can make use of the `errorString()` method to see what the actual problem is:

[PRE62]

Once you are sure the component is ready, you can finally start creating objects from it. For this, the component exposes a method called `createObject`. In its simplest form, it accepts an object that is to become the parent of the newly born instance (similar to widget constructors accepting a pointer to a parent widget) and returns the new object itself so that you can assign it to some variable:

[PRE63]

Then, you can start setting the object's properties:

[PRE64]

A more complex invocation lets us do both these operations (create the object and set its properties) in a single call by passing a second parameter to `createObject`:

[PRE65]

The second parameter is an object (created here using JSON syntax) whose properties are to be applied to the object being created. The advantage of the latter syntax is that all property values are applied to the object as one atomic operation (just like usual when the item is declared in a QML document) instead of a series of separate operations, each of which sets the value for a single property, possibly causing an avalanche of change handler invocations in the object.

After creation, the object becomes a first-class citizen of the scene, acting in the same way as items declared directly in the QML document. The only difference is that a dynamically created object can also be dynamically destructed by calling its `destroy()` method, which is an equivalent of calling `delete` on C++ objects. When speaking of destroying dynamic items, we have to point out that when you assign a result of `createObject` to a variable (like `circle`, in our example) and that variable goes out of scope, the item will not be released and garbage collected as its parent still holds a reference to it, preventing it from being recycled.

We didn't mention this explicitly before, but we have already used inline component definitions earlier in this chapter when we introduced the `Repeater` element. The repeated item defined within the repeater is in fact not a real item, but a component definition that is instantiated as many times as needed by the repeater.

## Delaying item creation

Another recurring scenario is that you do know how many elements you are going to need, but the problem is that you cannot determine up front what type of elements they are going to be. At some point during the lifetime of your application, you will learn that information and will be able to instantiate an object. Until you gain the knowledge about the given component, you will need some kind of item placeholder where you will later put the real item. You can, of course, write some code to use the `createObject()` functionality of the component, but this is cumbersome. Fortunately, Qt Quick offers a nicer solution in the form of a `Loader` item. This item type is exactly what we described it to be–a temporary placeholder for a real item that will be loaded on demand from an existing component. You can put `Loader` in place of another item and when you need to create this item, one way is to set the URL of a component to the `source` property:

[PRE66]

Immediately afterwards, the magic begins and an instance of the component appears in the loader. If the `Loader` object has its size set explicitly (for example, by anchoring or setting the width and height), then the item will be resized to the size of the loader. If an explicit size is not set, then `Loader` will instead be resized to the size of the loaded element once the component is instantiated:

[PRE67]

In the preceding situation, the loader has its size set explicitly so when its item is created, it will respect the anchors and sizes declared here.

## Accessing your item's component functionality

Each item in Qt Quick is an instantiation of some kind of component. Each object has a `Component` attached property that offers two signals informing about important moments of the object's life cycle. The first signal–`completed()`—is triggered after the object has been instantiated. If you provide a handler for the signal, you can perform some late initialization of the object after it has been fully instantiated. There are many use cases for this signal, starting with logging a message to the console:

[PRE68]

A more advanced use of this signal is to optimize performance by delaying expensive operations until the component is fully constructed:

[PRE69]

When items are created, they are added to their parent's `children` property. Thus, as items get created and destroyed, the value of that property is modified, triggering the `childrenChanged` signal. As this happens, we would like to reposition the item's children according to some algorithm. For that, we have an internal `QtObject` instance (representing a `QObject`) called `priv` where we can declare functions and properties that will not be visible outside the component definition. In there, we have a `layoutItems()` function that is called whenever the list of children is updated. This is fine if items are created or destroyed dynamically (for example, using the `Component.createObject()` function). However, as the root object is being constructed, it may have a number of child items declared directly in the document. There is no point in repositioning them over and over again as declarations are instantiated. Only when the list of objects is complete does it make sense to position the items. Therefore, we declare a Boolean property in the private object denoting whether the root item is fully constructed. Until it is, every time `layoutItems()` is called, it will exit immediately without doing any computations. When `Component.onCompleted` is called, we raise the flag and call `layoutItems()`, which computes the geometry of all child items declared statically in the document.

The other signal in the attached `Component` property is `destruction`. It is triggered right after the destruction process for the object starts when the component is still fully constructed. By handling that signal, you can perform actions such as saving the state of the object in persistent storage or otherwise cleaning the object up.

## Imperative painting

Declaring graphical items is nice and easy but as programmers, we're more used to writing imperative code, and some things are easier expressed as an algorithm rather than as a description of the final result to be achieved. It is easy to use QML to encode a definition of a primitive shape such as a rectangle in a compact way–all we need is to mark the origin point of the rectangle, its width, height, and optionally, a color. Writing down a declarative definition of a complex shape consisting of many control points positioned in given absolute coordinates, possibly with an outline in some parts of it, maybe accompanied by an image or two, is still possible in language such as QML; however, this will result in a much more verbose and much less readable definition. This is a case where using an imperative approach might prove more effective. HTML (being a declarative language) already exposes a proven imperative interface for drawing different primitives called a `Canvas` that has been used in numerous Web applications. Fortunately, Qt Quick provides us with its own implementation of a `Canvas` interface similar to the one from the Web by letting us instantiate `Canvas` items. Such items can be used to draw straight and curved lines, simple and complex shapes, and graphs and graphic images. It can also add text, colors, shadows, gradients, and patterns. It can even perform low-level pixel operations. Finally, the output may be saved as an image file or serialized to a URL usable as source for an `Image` item. There are many tutorials and papers available out there on using an HTML canvas and they can usually be easily applied to a Qt Quick canvas, as well (the reference manual even includes a list of aspects you need to pay attention to when porting HTML canvas applications to a Qt Quick canvas), so here we will just give you the very basics of imperative drawing in Qt Quick.

Consider a game where the player's health is measured by the condition of his heart–the slower the beat, the more healthy the player is. We will use this kind of visualization as our exercise in practicing painting using the `Canvas` element.

# Time for action – preparing Canvas for heartbeat visualization

Let's start with simple things by creating a Quick UI project based on the latest version of Qt Quick. Rename the QML file Creator made for us to `HeartBeat.qml`. Open the `qmlproject` file that was created with the project and change the `mainFile` property of the `Project` object to `HeartBeat.qml`. Then, you can close the `qmlproject` document and return to `HeartBeat.qml`. There, you can replace the original content with the following:

[PRE70]

When you run the project, you will see... a blank window.

## *What just happened?*

In the preceding code, we created a basic boilerplate code for using a canvas. First, we renamed the existing file to what we want our component to be called, and then we informed Creator that this document is to be executed when we run the project using `qmlscene`.

Then, we created a `Canvas` instance with an implicit width and height set. There, we created a handler for the `paint` signal that is emitted whenever the canvas needs to be redrawn. The code placed there retrieves a context for the canvas, which can be thought of as an equivalent to the `QPainter` instance we used when drawing on Qt widgets. We inform the canvas that we want its 2D context, which gives us a way to draw in two dimensions. A 2D context is the only context currently present for the `Canvas` element, but you still have to identify it explicitly–similar to in HTML. Having the context ready, we tell it to clear the whole area of the canvas. This is different to the widget world in which when the `paintEvent` handler was called, the widget was already cleared for us and everything had to be redrawn from scratch. With `Canvas`, it is different; the previous content is kept by default so that you can draw over it if you want. Since we want to start with a clean sheet, we call `clearRect()` on the context.

# Time for action – drawing a heartbeat

We will extend our component now and implement its main functionality–drawing a heartbeat-like diagram.

Add the following property declarations to `canvas`:

[PRE71]

Below, add a declaration for a timer that will drive the whole component:

[PRE72]

Then, define the handler for when the value of `arg` is modified:

[PRE73]

Then, implement `func`:

[PRE74]

Finally, modify `onPaint`:

[PRE75]

Then, you can run the code and see a heart beat-like diagram appear on the canvas:

![Time for action – drawing a heartbeat](img/8874OS_09_19.jpg)

## *What just happened?*

We added two kinds of properties to the element. By introducing `lineWidth`, we can manipulate the width of the line that visualizes the heartbeat. The `points` and `arg` variables are two helper variables that store an array of points already calculated and the function argument that was last evaluated. The function we are going to use is a periodic function that extends from `-Π` to `+Π`; thus, we initialize `arg` to `-Math.PI` and we store an empty array in points.

Then, we added a timer that ticks in regular intervals, incrementing `arg` by 1° until it reaches `+Π`, in which case it is reset to the initial value.

Changes to `arg` are intercepted in the handler we implemented next. In there, we push a new item to the array of points. The value is calculated by the function `func`, which is quite complicated, but it is sufficient to say that it returns a value from within a range of `-1` to `+1`. The array of points is then compacted using `Array.slice()` so that at most, the last canvas.width items remain in the array. This is so we can plot one point for each pixel of the width of the canvas and we don't have to store any more data than required. At the end of the function, we invoke `requestPaint()`, which is an equivalent of `QWidget::update()` and schedules a call to paint.

That, in turn, calls our `onPaint`. There, after retrieving the context, we reset the canvas to its initial state and then calculate an array of points that is to be drawn again by using `slice()`. Then, we prepare the canvas by translating and scaling it in the vertical axis so that the origin is moved to half of the height of the canvas (that's the reason for calling `reset()` at the beginning of the procedure–to revert this transformation). After that, `beginPath()` is called to inform the context that we are starting to build a new path. Then, the path is built segment by segment by appending lines. Each value is multiplied by `canvas.height/2` so that values from the point array are scaled to the size of the item. The value is negated as the vertical axis of the canvas grows to the bottom and we want positive values to be above the origin line. After that, we set the width of the pen and draw the path by calling `stroke()`.

# Time for action – making the diagram more colorful

The diagram serves its purpose, but it looks a bit dull. Add some shine to it by defining three new color properties in the canvas object–`color`, `topColor`, `bottomColor`–and setting their default values to `black`, `red`, and `blue`, respectively.

Since `points` and `arg` should not really be public properties that anyone can change behind our backs, we'll correct it now. Declare a child element of the canvas of `QtObject` and set its ID to `priv`. Move declarations of `points` and `arg` inside that object. Move the `onArgChanged` handler there, as well:

[PRE76]

Then, search through the whole code and prefix all occurrences of arg and points outside the newly declared object with `priv`, so that each of their invocations lead to the `priv` object.

Then, let's make use of the three colors we defined by extending `onPaint`:

[PRE77]

Upon running the preceding code snippet, you get the following output:

![Time for action – making the diagram more colorful](img/8874OS_09_20.jpg)

## *What just happened?*

By moving the two properties inside the `priv` object, we have effectively hidden them from the external world as child objects of an object (such as `priv` being a child of `canvas`) are not accessible from outside the QML document that defines the object. This ensures that neither `points` nor `arg` can be modified from outside the `HeartBeat.qml` document.

The modifications to `onPaint` that we implemented are creating another path and using that path to fill an area using a gradient. The path is very similar to the original one, but it contains two additional points that are the first and last point drawn projected onto the horizontal axis. This makes sure the gradient fills the area properly. Please note that the canvas uses imperative code for drawing; therefore, the order of drawing the fill and the stroke matters–the fill has to be drawn first so that it doesn't obscure the stroke.

# Qt Quick and C++

Thus far, we have been using standard Qt Quick items or creating new ones by compositing existing element types in QML. But there is a lot more you can do if you interface QML and C++ using the technologies Qt has to offer. Essentially, QML runtime does not differ much in its design from Qt Script, which you read about in the previous chapter of this book. In the following paragraphs, you will learn how to gain access to objects living in one of the environments from within the other one, as well as how to extend QML with new modules and elements.

Until now, all the example projects we did in this chapter were written with just QML and because of that, the project type we were choosing was Qt Quick UI, which let us quickly see the Qt Quick scene we modeled by interpreting it with the `qmlscene` tool. Now, we will want to add C++ to the equation and since C++ is a compiled language, we will need to do some proper compilation to get things working. Therefore, we will be using the **Qt Quick Application** template.

## Creating QML objects from C++

When you start a new project of such a type in Qt Creator, after you answer the question about the component set you would like to use (choose any of the Qt Quick 2.*x* options for a regular Qt Quick application), you will receive some boilerplate code–a `main.cpp` file containing the C++ part and `main.qml`, which contains the scene definition. Let's have a look at the latter first:

[PRE78]

The code is a little bit different than before; just look at the highlighted parts. Instead of an `Item` root object, we now have a Window together with an `import` statement for a `QtQuick.Window` module. To understand why this is the case, we will have to understand the C++ code which invokes this QML document:

[PRE79]

The source code is pretty simple. First, we instantiate an application object, just like for any other type of application. As we are not using Qt widgets, `QGuiApplication` is used instead of `QApplication`. The last line of the main function is also obvious–the application's event loop is started. Between those two lines, we can see an instance of `QQmlApplicationEngine` being created and fed with the URL of our QML document.

QML is driven by an engine implemented in `QQmlEngine` that is somewhat similar to `QScriptEngine`. `QQmlApplicationEngine` is a subclass of `QQmlEngine`, which provides a simple way of loading an application from a single QML file. This class does not create a root window to display our Qt Quick scene (QML applications don't have to necessarily be Qt Quick applications; they don't have to deal with the user interface at all), so it is the responsibility of the application to create a window if it wants to show a Qt Quick scene in it.

An alternative fit for loading Qt Quick-based user interfaces would be to use `QQuickView` or its less convenient superclass–`QQuickWindow`, which inherit `QWindow` and are able to render Qt Quick scenes.

You could then replace the `main.cpp` contents with the following code:

[PRE80]

Since `QQuickView` inherits `QWindow`, we can see that a window will be created to encompass the Qt Quick scene defined in `main.qml`. In such an event, you could replace the Window declaration with an `Item` similar to what we have seen in the earlier examples.

### Tip

If you want to combine a Qt Quick scene with a Qt widgets-based user interface, you can use `QQuickWidget` present in the `QtQuickWidgets` module (add `QT += quickwidgets` to the project file to activate the module), which is similar to `QQuickView` and has a similar API, but instead of rendering the scene to a separate window, it renders it to a widget you can then put alongside other widgets.

The last way of creating QML objects is to use `QQmlComponent`. Contrary to the previous approaches, which had a `QQmlEngine` instance embedded in the object creating the QML object, we have to use a separate engine with the component method.

`QQmlComponent` is a wrapper around a QML component definition similar to the `Component` element on the QML side. It can create instances of that component with the `create()` method using a given `QQmlEngine` instance:

[PRE81]

The object created is `QObject`, since that is the base class for all objects in QML. If the object represents a Qt Quick user interface, you can cast it to `QQuickItem` and use its methods to access Item's functionality:

[PRE82]

`QQmlComponent` is the most "classic" way of instantiating QML objects. You can even use it to create additional objects in existing views:

[PRE83]

A variation on using `QQmlComponent` is to create an object in the QML engine asynchronously using the `QQmlIncubator` object. When creating complex objects, it takes time for them to instantiate and at times, it is desired to not block the control flow for too long by waiting for the operation to complete. In such cases, an incubator object can be used to schedule instantiation and continue the flow of the program. We can query the state of the incubator and when the object is constructed, we will be able to access it. The following code demonstrates how to use the incubator to instantiate an object and process pending events while waiting for the operation to complete:

[PRE84]

## Pulling QML objects to C++

In our terminology, pulling QML objects to C++ means that by using C++ code, we would like to gain access to objects living in the QML engine (for example, those declared in some QML file). Before we do that, it is important that we stress that in general, it is bad practice to try and pull objects from the QML engine. There are a few reasons for that, but we would like to stress just two of them.

First, if we assume the most common case, which is that the QML part of our application deals with a user interface in Qt Quick for the logic written in C++, then accessing QtQuick objects from C++ breaks the separation between logic and the presentation layer, which is one of the major principles in GUI programming. The second reason is that QML documents (and Qt Quick ones in particular) are often made by different people (designers) than those who implement the application logic (programmers). The user interface is prone to dynamic changes, relayouting up to a complete revamp. Heavy modifications of QML documents, such as adding or removing items from the design, would then have to be followed by adjusting the application logic to cope with those changes. This in turn needs recompilation of the whole application, which is cumbersome. In addition, if we allow a single application to have multiple user interfaces (skins), it might happen that because they are so different, it is impossible to decide upon a single set of common entities with hardcoded names that could be fetched from C++ and manipulated. Even if you managed to do that, such an application could crash easily if the rules were not strictly followed by designers.

That said, we have to admit that there are cases when it does make sense to pull objects from QML to C++, and that is why we decided to familiarize you with the way to do it. One of the situations where such an approach is desired is when QML serves us as a way to quickly define a hierarchy of objects with properties of different objects linked through more or less complex expressions, allowing them to answer to changes taking place in the hierarchy.

For example, if you create a `Qt Quick UI` project, among the files generated, you will find a `qmlproject` file containing the project definition expressed in QML itself, such as this one:

[PRE85]

It contains project contents specified as a set of file selectors and additional properties such as the main project file or a list of directories of where to look for QML modules. It is very easy to specify such a project description in QML and after doing so and by getting a handle on the `Project` instance from C++, one can read the required information directly from the object and its properties as needed.

`Project` is considered a root object of this document. There are five ways to get access to a root object, based on how the document was actually loaded into the engine:

*   `QQmlApplicationEngine::rootObjects()` if using `QQmlApplicationEngine`
*   `QQuickView::rootObject()` if using `QQuickView`
*   `QQuickWidget::rootObject()` if using `QQuickWidget`
*   `QQmlComponent::create()` if using `QQmlComponent`
*   `QQmlIncubator::object()` if using `QQmlComponent` with `QQmlIncubator`

As we noted earlier, after retrieving an object, you can downcast it to a proper type using `qobject_cast`. Alternatively, you can start using the object through the generic `QObject` interface–accessing properties with `property()` and `setProperty()`, running functions through `QMetaObject::invokeMethod()`, and connecting to signals as usual.

The use case provided is a valid and fair situation when you want to pull a view root object or a manually created object from the QML world into C++. Now, we are going to show you how to do the same for an object from an arbitrary depth of the object tree.

QML documents define object trees. We can ask Qt to traverse a `QObject` tree and return a single object or a list of objects in the tree matching specified criteria. The same approach can be implemented for QML object trees. There are two criteria that can be used when searching. First, we can search for objects inheriting from a given class. Then, we can search for objects matching a given value of the `objectName` property defined in `QObject`. To search the tree for objects, one uses a `findChild` template method.

Consider a Qt Quick document defining a number of items:

[PRE86]

After gaining access to the root object using one of the methods described earlier, we can query the object tree for any of the colored shape items using the `objectName` values:

[PRE87]

The `findChild()` method requires us to pass a class pointer as the template argument. Without knowing what class actually implements a given type, it is safest to simply pass `QObject*` as, once again, we know all QML objects inherit this. It is more important what gets passed as the function argument value–it is the name of the object we want returned. Notice it is not `id` of the object, but the value of the `objectName` property. When the results get assigned to the variables, we verify whether items have been successfully found and if that is the case, the generic `QObject` API is used to set the width of the circle to that of the rectangle.

Let us stress this again: if you have to use this approach, limit it to the minimum. And always verify whether the returned item exists (is a non-null pointer); the QML document might change between subsequent compilations of the program, and items and their names existing in one version of the document might cease to exist in the next version.

## Pushing C++ objects to QML

A much better approach is to cross the boundary in the other direction–by exporting objects from C++ to QML. This allows C++ developers to decide what API is available for the script. The choice of which API to use is left to QML developers. Separation between the application logic and the user interface is maintained.

In the previous chapter, you learned to use Qt Script. We told you how to expose existing `QObject` instances to scripting through the use of the script engine's global object. We also discussed execution contexts, which provide layers of object visibility while calling functions. As already mentioned, QML has many similarities to that framework and in QML, a very similar approach is used to expose objects to the engine.

QML engines also use contexts to provide data scopes for the language. You can set properties on a context to make certain names resolve to given objects:

[PRE88]

From this moment, `object` is visible within `context` under the name `foo`.

Contexts can form hierarchies. On the top of the hierarchy resides a root context of the engine. Context properties are resolved from the bottom up, meaning that redefining a name in a child context shadows the name defined in the parent context. Let's see an example:

[PRE89]

We created instances of classes `A`, `B`, and `C` and assigned them to a `foo` property of different contexts forming a hierarchy of five contexts. Why five? When passing a `QQmlEngine` to a constructor of `QQmlContext`, the context created becomes a child of the engine's root context. Therefore, we have four contexts we created ourselves and an additional context that always exists in the engine:

![Pushing C++ objects to QML](img/8874OS_09_21.jpg)

Now, if we call foo from within `childContext1`, we will access object `B`, and when we call `foo` from `childContext2`, we will access `C`. If we call it from `childContext3`, then, since `foo` is not defined there, the call will propagate to `parentContext` and hence `A` will be accessed. In `rootContext`, the context foo will not be available at all.

In most cases, we will not be creating contexts ourselves and thus, the most common situation is when we will have control over just the root context since it always exists and is easily accessible. Therefore, this context will usually be used to register C++ objects. As the root engine context is an ancestor of all other contexts, an object registered there will be visible from any QML document.

So what can we do with an exported object using QML? The object itself is accessible using the identifier given to it with the `setContextProperty()`. The identifier can be treated as the ID pseudo-property declared on objects in QML documents. Features that can be accessed from QML depend on the kind of object exported.

You can export two kinds of object. First, you can export a `QVariant` value that is then converted to an equivalent QML entity. The following table lists the most commonly used basic types:

| Qt type | QML basic type |
| --- | --- |
| `bool` | `bool` |
| `unsigned int, int` | `int` |
| `double` | `double` |
| `float, qreal` | `real` |
| `QString` | `string` |
| `QUrl` | `url` |
| `QColor` | `color` |
| `QFont` | `font` |
| `QDate` | `date` |
| `QPoint, QPointF` | `point` |
| `QSize, QSizeF` | `size` |
| `QRect, QRectF` | `rect` |

It allows us to export a wide range of objects:

[PRE90]

And use them easily in QtQuick:

[PRE91]

This will give us the following output:

![Pushing C++ objects to QML](img/8874OS_09_22.jpg)

In addition to the basic types, the QML engine provides automatic type conversions between special `QVariant` cases and JavaScript types–`QVariantList` is converted to JavaScript array and `QVariantMap` to a JavaScript object. This allows for an even more versatile approach. We can group all the weather information within a single JavaScript object by taking advantage of the `QVariantMap` conversion:

[PRE92]

As a result, we get better encapsulation on the QML side:

[PRE93]

That's all fine and dandy in a world where weather conditions never change. In real life, however, one needs a way to handle situations where the data changes. We could, of course, re-export the map every time any of the values changed, but that would be very tedious.

Fortunately, the second kind of object that can be exported to QML comes to our rescue. Apart from `QVariant`, the engine can accept `QObject` instances as context property values. When exporting such an instance to QML, all the object's properties are exposed and all its slots become callable functions in the declarative environment. Handlers are made available for all the object's signals.

# Time for action – self-updating car dashboard

In the next exercise, we will implement a car dashboard that can be used in a racing game and will show a number of parameters such as current speed and motor revolutions per minute. The final result will look similar to the following image:

![Time for action – self-updating car dashboard](img/8874OS_09_23.jpg)

We will start with the C++ part. Set up a new Qt Quick Application. Choose the most recent Qt Quick version for the Qt Quick component set. This will generate a main function for you that instantiates `QGuiApplication` and `QQmlApplicationEngine` and sets them up to load a QML document.

Use the **File** menu to create **New file or Project** and create a new C++ class. Call it `CarInfo` and choose `QWidget` as its base class. Why not `QObject`, you may ask? This is because our class will also be a widget, which will be used for modifying values of different parameters so that we may observe how they influence what the Qt Quick scene displays. In the class header, declare the following properties:

[PRE94]

The properties are read-only and the `NOTIFY` clause defines signals emitted when respective property values change. Go ahead and implement the appropriate functions for each property. Apart from the getter, also implement a setter as a public slot. Here is an example for a property controlling the speed of the car:

[PRE95]

You should be able to follow the example for the remaining properties on your own.

Since we want to use the widget to tweak property values, design the user interface for it using a Qt Designer Form. It can something look like this:

![Time for action – self-updating car dashboard](img/8874OS_09_24.jpg)

Make appropriate signal-slot connections in the widget so that modifying any of the widgets for a given parameter or using the setter slot directly updates all the widgets for that parameter.

### Tip

Instead of adding member variables to the `CarInfo` class for properties such as `speed`, `rpm`, `distance`, or `gear` you can operate directly on the widgets placed on the `ui` form so that, for example, a getter for the `distance` property will look like:

[PRE96]

The setter would then be modified to:

[PRE97]

You will then need to add `connect()` statements to the constructor to be sure that signals are propagated from the ui form:

[PRE98]

Next, you can test your work by running the widget. To do this, you have to alter the main function to look as follows:

[PRE99]

Since we are using widgets, we have to replace `QGuiApplication` with `QApplication` and enable the widgets module by placing `QT += widgets` in the project file (remember to run `qmake` from the project's context menu afterwards). Make sure everything works as expected (that is, that moving sliders and changing spinbox values reflect the changes to widget properties) before moving on to the next step.

We are now going to add `QtQuick` to the equation, so let's start by updating our main function to display our scene. Introduce the highlighted changes to the code:

[PRE100]

The modifications create a view for our scene, export the `CarInfo` instance to the global context of the QML engine, and load and display the scene from a file located in a resource.

It is important to first export all the objects and only then load the scene. This is because we want all the names to be already resolvable when the scene is being initialized so that they can be used right away. If we reversed the order of calls, we would get a number of warnings on the console about the identities being undefined.

Finally, we can focus on the QML part. Look at the picture of the result we want to be shown at the beginning of the exercise. For the black background, we used a bitmap image created in a graphical editor (you can find the file in the materials for this book), but you can obtain a similar effect by composing three black rounded rectangles directly in `QtQuick`–the two outer parts are perfect circles and the inner module is a horizontally stretched ellipse.

If you decide to use our background file (or make your own prettier image), you can put the following code into `main.qml`:

[PRE101]

What we do here is make the image our root item and create three items to serve as containers for different elements of the dashboard. The containers are all centered in the parent and we use a `horizontalCenterOffset` property to move the two outer items sideways. The values of the offset, as well as the widths, are calculated by trial and error to look good (note that all three containers are perfect squares). If instead of using our file, you settle for creating the three parts yourself using Qt Quick items, the containers will simply be anchored to the centers of the three black items.

The dials look complicated, but in reality, they are very easy to implement and you have already learned everything you need to design them.

Let's start with the needle. Create a new QML document and call it `Needle.qml`. Open the file and place the following content:

[PRE102]

The document defines an item with four attributes–the length of the needle (defaults to 80% of the dial's radius), the color of the needle, `middleColor`, which stands for the color of the needle's fixing, and the size, which defines how wide the needle is. The code is self-explanatory. The item itself does not have any dimensions and onlys acts as an anchor for visual elements–the needle itself is a thin rectangle oriented vertically with a fixing 20 units from the end. The fixing is a circle of the same color as the needle with a smaller circle in the middle that uses a different fill color. The smaller radius of the inner circle is obtained by filling the outer circle with a 25% margin from each side.

As for the dials, we will put their code inline in the main file since we just have two of them and they differ a bit, so the overhead of creating a separate component with a well-designed set of properties would outweigh the benefits of having nicely encapsulated objects.

If you think about what needs to be done to have the dial displayed and working, it seems the hardest thing is to layout the numbers nicely on the circle, so let's start with that. Here is an implementation of a function for calculating the position along a circle circumference, based on the radius of the circle and angle (in degrees) where an item should be positioned:

[PRE103]

The function converts degrees to radians and returns the desired point. The function expects a width property to be available that helps to calculate the center of the circle and in case a radius was not given, serves as a means to calculate a feasible value for it.

With such a function available, we can use the already familiar `Repeater` element to position items where we want them. Let's put the function in `middleContainer` and declare the dial for car speed:

[PRE104]

You might have noticed we used an element called `Label`. We created it to avoid having to set the same property values for all the texts we use in the user interface:

[PRE105]

The dial consists of a repeater that will create 12 elements. Each element is an item positioned using the earlier described function. The item has a label anchored to it that displays the given speed. We use `120+index*12*2` as the angle expression as we want "0" to be positioned at 120 degrees and each following item positioned 24 degrees further.

The needle is given rotation based on the value read from the `carData` object. Since the angular distance between consecutive 20 kph labels is 24 degrees, the distance for one kph is 1.2 and thus we multiply `carData.speed` by that factor. Item rotation is calculated with 0 degrees "pointing right"; therefore, we add 90 to the initial 120 degree offset of the first label to obtain starting coordinates matching those of the label system.

As you can see in the image, the speed dial contains small lines every 2 kph, with those divisible by 10 kph longer than others. We can use another `Repeater` to declare such ticks:

[PRE106]

Finally, we can put a label for the dial:

[PRE107]

Make sure the label is declared before the dial needle, or give the needle a higher *z* value so that the label doesn't overpaint the needle.

Next, repeat the process on your own for the left container by creating an RPM dial reading values from `carData.rpm`. The dial also displays the current gear of the car's engine. Place the following code inside the `leftContainer` object definition:

[PRE108]

The only part needing explanation is highlighted. It defines an array of gear labels starting with reverse, going through neutral, and then through five forward gears. The array is then indexed with the current gear and the text for that value is applied to the label. Notice that the value is incremented by 1, which means the 0th index of the array will be used when `carData.gear` is set to `1`.

We will not show how to implement the right container. You can do that easily yourself now with the use of the `Grid` positioner to lay out the labels and their values. To display the series of controls on the bottom of the right container (with texts `ABS`, `ESP`, `BRK`, and `CHECK`), you can use `Row` of `Label` instances.

Now, start the program and begin moving the sliders on the widget. See how the Qt Quick scene follows the changes.

## *What just happened?*

We have created a very simple `QObject` instance and exposed it as our "data model" to QML. The object has a number of properties that can receive different values. Changing a value results in emitting a signal, which in turn notifies the QML engine and causes bindings containing those properties to be reevaluated? As a result, our user interface gets updated.

The data interface between the QML and C++ worlds that we created is very simple and has a small number of properties. But as the amount of data we want to expose grows, the object can become cluttered. Of course, we can counter that effect by dividing it into multiple smaller objects each having separate responsibilities and then exporting all those objects to QML, but that is not always desirable. In our case, we can see that rpm and gear are properties of the engine sub-system so we could move them to a separate object; however, in reality, their values are tightly coupled with the speed of the car and to calculate the speed, we will need to know the values of those two parameters. But the speed also depends on other factors such as the slope of the road, so putting the speed into the engine sub-system object just doesn't seem right. Fortunately, there is a nice solution for that problem.

# Time for action – grouping engine properties

QML has a concept called grouped properties. These are properties of an object that contain a group of "sub-properties." You already know a number of them–the border property of the `Rectangle` element or the anchors property of the `Item` element, for example. Let's see how to define such properties for our exposed object.

Create a new `QObject`-derived class and call it `CarInfoEngine`. Move the property definitions of rpm and gear to that new `class.Add` the following property declaration to `CarInfo`:

[PRE109]

Implement the getter and the private field:

[PRE110]

We are not going to use the signal right now; however, we had to declare it otherwise QML would complain we were binding expressions that depend on properties that are non-notifiable:

[PRE111]

Initialize `m_engine` in the constructor of `CarInfo`:

[PRE112]

Next, update the code of `CarInfo` to modify properties of `m_engine` whenever respective sliders on the widget are moved. Provide a link the other way, as well–if the property value is changed, update the user interface accordingly.

Update the QML document and replace `carData.gear` with `carData.engine.gear`. Do the same for `carData.rpm` and `carData.engine.rpm`. You should end up with something along the lines of:

[PRE113]

## *What just happened?*

Essentially, what we did is expose a property in `CarInfo` that is itself an object that exposes a set of properties. This object of the type `CarInfoEngine` is bound to the `CarInfo` instance it refers to.

# Extending QML

Thus far, what we did was exposing to QML single objects created and initialized in C++. But we can do much more–the framework allows us to define new QML types. These can either be generic `QObject` derived QML elements or items specialized for Qt Quick. In this section, you will learn to do both.

## Registering classes as QML elements

We will start with something simple–exposing the `CarInfo` type to QML so that instead of instantiating it in C++ and then exposing it in QML, we can directly declare the element in QML and still allow the changes made to the widget to be reflected in the scene.

To make a certain class (derived from `QObject`) instantiable in QML, all that is required is to register that class with the declarative engine using the `qmlRegisterType` template function. This function takes the class as its template parameter along a number of function arguments: the module `uri`, the major and minor version numbers, and the name of the QML type we are registering. The following call would register the class `FooClass` as the QML type `Foo`, available after importing `foo.bar.baz` in Version 1.0:

[PRE114]

You can place this invocation anywhere in your C++ code; just make sure this is before you try to load a QML document that might contain declarations of `Foo` objects. A typical place to put the function call is in the program's main function:

[PRE115]

Afterwards, you can start declaring objects of the type `Foo` in your documents. Just remember you have to import the respective module first:

[PRE116]

# Time for action – making CarInfo instantiable from QML

First, we will update the QML document to create an instance of `CarInfo` present in the CarInfo 1.0 module:

[PRE117]

As for registering `CarInfo`, it might be tempting to simply call `qmlRegisterType` on `CarInfo` and congratulate ourselves for a job well done:

[PRE118]

In general this would work (yes, it is as simple as that). However, at the time of writing, trying to instantiate any widget in a QML document as the child of some `QtQuick` item will lead to a crash (maybe at the time you are reading this text the issue will have already been resolved). To avoid this, we need to make sure that what we instantiate is not a widget. For that, we will use a proxy object that will forward our calls to the actual widget. Therefore, create a new class called `CarInfoProxy` derived from `QObject` and make it have the same properties as `CarInfo`, for example:

[PRE119]

Declare one more property that will let us show and hide the widget on demand:

[PRE120]

Then, we can place the widget as a member variable of the proxy so that it is created and destroyed alongside its proxy:

[PRE121]

Next, implement the missing interface. For simplicity, we are showing you code for some of the properties. The others are similar so you can fill in the gaps on your own:

[PRE122]

You can see that we reuse the `CarInfoEngine` instance from the widget instead of duplicating it in the proxy class. Finally, we can register `CarInfoProxy` as `CarInfo`:

[PRE123]

If you run the code now, you will see it works–`CarInfo` has become a regular QML element. Because of this, its properties can be set and modified directly in the document, right? If you try setting the speed or the distance, it will work just fine. However, as soon as you try setting any of the properties grouped in the engine property, QML runtime will start complaining with a message similar to the following one:

[PRE124]

This is because the runtime does not understand the engine property–we declared it as `QObject` and yet we are using a property this class doesn't have. To avoid this issue, we have to teach the runtime about `CarInfoEngine`.

First, let's update the property declaration macro to use `CarInfoEngine` instead of `QObject`:

[PRE125]

And the getter function itself, as well:

[PRE126]

Then, we should teach the runtime about the type:

[PRE127]

## *What just happened?*

In this exercise, we let the QML runtime know about two new elements. One of them is `CarInfo`, which is a proxy to our widget class. We told the engine this is a full-featured class that is instantiable from QML. The other class, `CarInfoEngine`, also became known to QML; however, the difference is that every attempt to declare an object of this type in QML fails with a given warning message. There are other functions available for registering types in QML but they are rarely used, so we will not be describing them here. If you are curious about them, type in qmlRegister in the Index tab of Creator's **Help** pane.

## Custom Qt Quick items

It is nice to be able to create new QML element types that can be used to provide dynamic data engines or some other type of non-visual objects; however, this chapter is about Qt Quick so it is time now to learn how to provide new types of items to Qt Quick.

The first question you should ask yourself is whether you really need a new type of item. Maybe you can achieve the same goal with already existing elements? Very often you can use vector or bitmap images to use custom shapes in your applications, or you can use Canvas to quickly draw the graphics you need directly in QML.

If you decide that you do require custom items, you will be doing that by implementing subclasses of `QQuickItem`, which is the base class for all items in Qt Quick. After creating the new type, you will always have to register it with QML using `qmlRegisterType`.

### OpenGL items

To provide very fast rendering of its scene, Qt Quick uses a mechanism called scene-graph. The graph consists of a number of nodes of well-known types, each describing a primitive shape to be drawn. The framework makes use of knowledge of each of the primitives allowed and their parameters to find the most performance-wise optimal order in which items can be rendered. Rendering itself is done using OpenGL, and all the shapes are defined in terms of OpenGL calls.

Providing new items for Qt Quick boils down to delivering a set of nodes that define the shape using terminology the graph understands. This is done by subclassing `QQuickItem` and implementing the pure virtual `updatePaintNode()` method, which is supposed to return a node that will tell the scene-graph how to render the item. The node will most likely be a describing a geometry (shape) with a material (color, texture) applied.

# Time for action – creating a regular polygon item

Let's learn about the scene-graph by delivering an item class for rendering convex regular polygons. We will draw the polygon using the OpenGL drawing mode called "triangle fan." It draws a set of triangles that all have a common vertex. Subsequent triangles are defined by the shared vertex, the vertex from the previous triangle, and the next vertex specified. Have a look at the diagram to see how to draw a hexagon as a triangle fan using 8 vertices as control points:

![Time for action – creating a regular polygon item](img/8874OS_09_25.jpg)

The same method applies for any regular polygon. The first vertex defined is always the shared vertex occupying the center of the shape. The remaining points are positioned on the circumference of a bounding circle of the shape at equal angular distances. The angle is easily calculated by dividing the full angle by the number of sides. For a hexagon, this yields 60 degrees.

Let's get down to business and the subclass `QQuickItem`. We will give it a very simple interface:

[PRE128]

Our polygon is defined by the number of sides and fill color. We also get everything we inherited from `QQuickItem`, including the width and height of the item. Besides the obvious getters and setters for the properties, we define just one method–`updatePaintNode()`, which is responsible for building the scene-graph.

Before we deal with updating graph nodes, let's deal with the easy parts first. Implement the constructor as follows:

[PRE129]

We make our polygon a hexagon by default. We also set a flag, `ItemHasContents`, which tells the scene-graph that the item is not fully transparent and should ask us how the item should be painted by calling `updatePaintNode()`. This is an early optimization to avoid having to prepare the whole infrastructure if the item would not be painting anything anyway.

The setters are also quite easy to grasp:

[PRE130]

A polygon has to have at least three sides; thus, we enforce this as a minimum, sanitizing the input value with `qMax`. After we change any of the properties that might influence the look of the item, we call `update()` to let Qt Quick know that the item needs to be rerendered. Let's tackle `updatePaintNode()` now. We'll disassemble it into smaller pieces so that it is easier for you to understand how the function works:

[PRE131]

When the function is called, it might receive a node it returned during a previous call. Be aware the graph is free to delete all the nodes when it feels like it, so you should never rely on the node being there even if you returned a valid node during the previous run of the function:

[PRE132]

The node we are going to return is a geometry node that contains information about the geometry and the material of the shape being drawn. We will be filling those variables as we go through the method:

[PRE133]

As we already mentioned, the function is called with the previously returned node as the argument but we should be prepared for the node not being there and we should create it. Thus, if that is the case, we create a new `QSGGeometryNode` and a new `QSGGeometry` for it. The geometry constructor takes a so-called attribute set as its parameter, which defines a layout for data in the geometry. Most common layouts have been predefined:

| Attribute set | Usage | First attribute | Second attribute |
| --- | --- | --- | --- |
| `Point2D` | `Solid colored shape` | `float x, y` | `-` |
| `ColoredPoint2D` | `Per-vertex color` | `float x, y` | `uchar red, green, blue, alpha` |
| `TexturedPoint2D` | `Per-vertex texture coordinate` | `float x, y` | `float tx, float ty` |

We will be defining the geometry in terms of 2D points without any additional information attached to each point; therefore, we pass `QSGGeometry::defaultAttributes_Point2D()` to construct the layout we need. As you can see in the preceding table for that layout, each attribute consists of two floating point values denoting the *x* and *y* coordinates of a point.

The second argument of the `QSGGeometry` constructor informs us about the number of vertices we will be using. The constructor will allocate as much memory as is needed to store the required number of vertices using the given attribute layout. After the geometry container is ready, we pass its ownership to the geometry node so that when the geometry node is destroyed, the memory for the geometry is freed as well. At this moment, we also mark that we are going to be rendering in the `GL_TRIANGLE_FAN` mode:

[PRE134]

The process is repeated for the material. We use `QSGFlatColorMaterial` as the whole shape is going to have one color that is set from `m_color`. Qt provides a number of predefined material types. For example, if we wanted to give each vertex a separate color, we would have used `QSGVertexColorMaterial` together with the `ColoredPoint2D` attribute layout:

[PRE135]

This piece of code deals with a situation in which `oldNode` did contain a valid pointer to a node that was already initialized. In this case, we only need to make sure the geometry can hold as many vertices as we need in case the number of sides changed since the last time the function was executed:

[PRE136]

This is repeated for the material. If the color differs, we reset it and tell the geometry node that the material needs to be updated by marking the `DirtyMaterial` flag:

[PRE137]

Finally, we can set vertex data. First, we ask the geometry object to prepare a mapping for us from the allocated memory to a `QSGGeometry::Point2D` structure, which can be used to conveniently set data for each vertex. Then, actual calculations are performed using the equation for calculating points on a circle. The radius of the circle is taken as the smaller part of the width and height of the item so that the shape is centered in the item. As you can see on the diagram at the beginning of the exercise, the last point in the array has the same coordinates as the second point in the array to close the fan into a regular polygon:

[PRE138]

At the very end, we mark the geometry as changed and return the node to the caller.

## *What just happened?*

Rendering in Qt Quick can happen in a thread different than the main thread. By implementing `updatePaintNode()`, we performed synchronization between the GUI thread and the rendering thread. The function executing the main thread is blocked. Due to this reason, it is crucial that it executes as quickly as possible and doesn't do any unnecessary calculations as this directly influences performance. This is also the only place in your code where you can safely call functions from your item (such as reading properties) as well as interact with the scene-graph (creating and updating the nodes). Try not emitting any signals nor creating any objects from within this method as they will have affinity to the rendering thread rather than the GUI thread.

Having said that, you can now register your class with QML and test it with the following QML document:

[PRE139]

This should give you a nice blue pentagon. If the shape looks aliased, you can enforce anti-aliasing on the window:

[PRE140]

## Have a go hero – creating a supporting border for RegularPolygon

What is returned by `updatePaintNode()` might not just be a single `QSGGeometryNode` but also a larger tree of `QSGNode` items. Each node can have any number of child nodes. By returning a node that has two geometry nodes as children, you can draw two separate shapes in the item:

![Have a go hero – creating a supporting border for RegularPolygon](img/8874OS_09_26.jpg)

As a challenge, extend `RegularPolygon` to draw not only the internal filled part of the polygon but also an edge that can be of a different color. You can draw the edge using the `GL_QUAD_STRIP` drawing mode. Coordinates of the points are easy to calculate–the points closer to the middle of the shape are the same points that form the shape itself. The remaining points also lie on a circumference of a circle that is slightly larger (by the width of the border). Therefore, you can use the same equations to calculate them. The `GL_QUAD_STRIP` mode renders quadrilaterals with every two vertices specified after the first four, composing a connected quadrilateral. The following diagram should give you a clear idea of what we are after:

![Have a go hero – creating a supporting border for RegularPolygon](img/8874OS_09_27.jpg)

## Painted items

Implementing items in OpenGL is quite difficult–you need to come up with an algorithm of using OpenGL primitives to draw the shape you want, and then you also need to be skilled enough with OpenGL to build a proper scene graph node tree for your item. But there is another way–you can create items by painting them with `QPainter`. This comes at a cost of performance as behind the scenes, the painter draws on an indirect surface (a frame buffer object or an image) that is then converted to OpenGL texture and rendered on a quad by the scene-graph. Even considering that performance hit, it is often much simpler to draw the item using a rich and convenient drawing API than to spend hours doing the equivalent in OpenGL or by using Canvas.

To use that approach, we will not be subclassing `QQuickItem` directly but rather `QQuickPaintedItem`, which gives us the infrastructure needed to use the painter for drawing items.

Basically, all we have to do, then, is implement the pure virtual `paint()` method that renders the item using the received painter. Let's see this put into practice and combine it with the skills we gained earlier.

# Time for action – creating an item for drawing outlined text

The goal of the current exercise is to be able to make the following QML code work:

[PRE141]

And produce the following result:

![Time for action – creating an item for drawing outlined text](img/8874OS_09_28.jpg)

Start with an empty Qt project with the `core`, `gui`, and `quick` modules activated. Create a new class and call it `OutlineTextItemBorder`. Delete the implementation file as we are going to put all code into the header file. Place the following code into the class definition:

[PRE142]

You can see that `Q_PROPERTY` macros don't have the `READ` and `WRITE` keywords we've been using thus far. This is because we are taking a shortcut right now and we let `moc` produce code that will operate on the property by directly accessing the given class member. Normally, we would recommend against such an approach as without getters, the only way to access the properties is through the generic `property()` and `setProperty()` calls. However, in this case, we are not going to be exposing this class to the public in C++ so we won't need the setters, and we implement the getters ourselves, anyway. The nice thing about the `MEMBER` keyword is that if we also provide the `NOTIFY` signal, the generated code will emit that signal when the value of the property changes, which will make property bindings in QML work as expected. The rest of the class is pretty simple–we are, in fact, providing a class for defining a pen that is going to be used for stroking text, so implementing a method that returns the actual pen seems like a good idea.

The class will provide a grouped property for our main item class. Create a class called `OutlineTextItem` and derive it from `QQuickPaintedItem`, as follows:

[PRE143]

The interface defines properties for the text to be drawn, in addition to its color, font, and the grouped property for the outline data. Again, we use `MEMBER` to avoid having to manually implement getters and setters. Unfortunately, this makes our constructor code more complicated as we still need a way to run some code when any of the properties are modified. Implement the constructor using the following code:

[PRE144]

We basically connect all the property change signals from both the object and its grouped property object to the same slot that is going to update the data for the item if any of its components are modified. We also call the same slot directly to prepare the initial state of the item. The slot can be implemented like this:

[PRE145]

At the beginning, the function resets a painter path object that serves as a backend for drawing outlined text and initializes it with the text drawn using the font set. Then, the slot calculates the bounding rect for the path using a function `shape()` that we will shortly see. Finally, it sets the calculated size as the size hint for the item and asks the item to repaint itself with the `update()` call:

[PRE146]

The `shape()` function returns a new painter path that includes both the original path and its outline created with the `QPainterPathStroker` object. This is so that the width of the stroke is correctly taken into account when calculating the bounding rectangle. We use `controlPointRect()` to calculate the bounding rectangle as it is much faster than `boundingRect()` and returns an area greater or equal to the one `boundingRect()` would, which is okay for us.

What remains is to implement the `paint()` routine itself:

[PRE147]

The code is really simple–we bail out early if there is nothing to draw. Otherwise, we set up the painter using the pen and color obtained from the item's properties. We enable anti-aliasing and calibrate the painter coordinates with that of the bounding rectangle of the item. Finally, we draw the path on the painter.

## *What just happened?*

During this exercise, we made use of the powerful API of Qt's graphical engine to complement an existing set of Qt Quick items with a simple functionality. This is otherwise very hard to achieve using predefined Qt Quick elements and even harder to implement using OpenGL. We agreed to take a small performance hit in exchange for having to write just about a hundred lines of code to have a fully working solution. Remember to register the class with QML if you want to use it in your code:

[PRE148]

# Summary

In this chapter, you have been familiarized with a declarative language called QML. The language is used to drive Qt Quick–a framework for highly dynamic and interactive content. You learned the basics of Qt Quick–how to create documents with a number of element types and how to create your own in QML or in C++. You also learned how to bind expressions to properties to automatically re-evaluate them. But so far, despite us talking about "fluid" and "dynamic" interfaces, you haven't seen much of that. Do not worry; in the next chapter, we will focus on animations in Qt Quick, as well as fancy graphics and applying what you learned in this chapter for creating nice looking and interesting games. So, read on!