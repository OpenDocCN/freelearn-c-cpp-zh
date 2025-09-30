# Animations in Qt Quick Games

In the previous two chapters, we introduced you to the basics of Qt Quick and QML. By now, you should be fluent enough with the syntax and understand the basic concepts of how Qt Quick works. In this chapter, we will show you how to make your games stand out from the crowd by introducing different kinds of animations that make your applications feel more like the real world. You will also learn to treat Qt Quick objects as separate entities programmable using state machines. A significant part of this chapter will introduce how to implement a number of important gaming concepts using Qt Quick. All this will be shown while we build a simple 2D action game using the presented concepts.

The main topics covered in this chapter are as follows:

*   Animation framework in Qt Quick
*   States and transitions in depth
*   Implementing games in Qt Quick
*   Sprite animations
*   Using state machines for animation
*   Parallax scrolling
*   Collision detection

# Animation framework in Qt Quick

In [Chapter 11](b81d9c47-58fa-49dd-931a-864c7be05840.xhtml), *Introduction to Qt Quick*, we implemented a simple animation using Qt Quick states and transitions. We will now deepen our knowledge on this topic and learn how to add some dynamics into the user interfaces we create. Thus far, books cannot contain moving pictures, so you will have to test most things we describe here yourself by running the provided Qt Quick code.

Qt Quick provides a very extensive framework for creating animations. By that, we don't mean only moving items around. We define an animation as *changing an arbitrary value over time*. So, what can we animate? Of course, we can animate item geometry. However, we can also animate rotation, scale, other numeric values, and even colors, but let's not stop here. Qt Quick also lets you animate the parent-child hierarchy of items or anchor assignments. Almost anything that can be represented by an item property can be animated.

Moreover, the changes are rarely linear—if you kick a ball in the air, it first gains height quickly because its initial speed was large. However, the ball is a physical object being pulled down by the earth's gravity, which slows the climb down until the ball stops and then starts falling down, accelerating until it hits the ground. Depending on the properties of both the ground and the ball, the object can bounce off the surface into the air again with less momentum, repeating the spring-like motion until eventually it fades away, leaving the ball on the ground. Qt Quick lets you model all that using easing curves that can be assigned to animations.

# Generic animations

Qt Quick provides a number of animation types derived from a generic `Animation` element that you will never use directly. The type exists only to provide an API common to different animation types.

Let's take a closer look at the animation framework by looking at a family of animation types derived from the most common animation type—`PropertyAnimation`. As the name implies, they provide the means to animate values of object properties. Despite the fact that you can use the `PropertyAnimation` element directly, it is usually more convenient to use one of its subclasses that specialises in dealing with the peculiarities of different data types.

The most basic property animation type is `NumberAnimation`, which lets you animate all kinds of numeric values of both integral and real numbers. The simplest way of using it is to declare an animation, tell it to animate a specific property in a specific object, and then set the length of the animation and the starting and ending value for the property:

```cpp
import QtQuick 2.9

Item {
    id: root
    width: 600; height: width
    Rectangle {
        id: rect
        color: "red"
        width: 50; height: width
    }
    NumberAnimation {
        target: rect
        property: "x"
        from: 0; to: 550
        duration: 3000
        running: true
    }
} 
```

# Time for action – Scene for an action game

Let's try something new for our new project. Select New File or Project from the File menu of Qt Creator, switch to the Other Project category and choose the Qt Quick UI Prototype template. Qt Creator will create a main QML file and a project file with the `.qmlproject` extension. This kind of project file is different than regular project files with the `.pro` extension. This is a pure QML project that does not contain any C++ code and thus does not require compilation. However, you need a QML runtime environment to run this project. Your Qt installation provides such an environment, so you can run the project from the terminal using the `qmlscene main.qml` command or just let Qt Creator handle that. Note that the Qt resources system is not used with these projects, and the QML files are loaded directly from the filesystem.

If you need to add C++ code to your project or you intend to distribute compiled binaries of the project, use the Qt Quick Application templates instead. The Qt Quick UI Prototype template, as the name implies, is only good for prototypes.

In the project directory, make a subdirectory called `images` and from the game project that we have created using Graphics View, copy `grass.png`, `sky.png`, and `trees.png`. Then, put the following code into the QML document:

```cpp
import QtQuick 2.9

Image {
    id: root
    property int dayLength: 60000 // 1 minute
    source: "images/sky.png"

    Item {
        id: sun
        x: 140
        y: root.height - 170
        Rectangle {
            id: sunVisual
            width: 40
            height: width
            radius: width / 2
            color: "yellow"
            anchors.centerIn: parent
        }
    }
    Image {
        source: "images/trees.png"
        x: -200
        anchors.bottom: parent.bottom
    }
    Image {
        source: "images/grass.png"
        anchors.bottom: parent.bottom
    }
} 
```

If you don't declare the top-level `Window` object, `qmlscene` will display the top-level Qt Quick item in a window automatically. Note that when writing a Qt Quick application driven by the `QQmlApplicationEngine` class, you need to declare the `Window` object explicitly.

When you run the project now, you will see a screen similar to this one:

![](img/065bfe56-ccd5-4b71-a0eb-349a2405fd24.png)

# What just happened?

We set up a very simple scene consisting of three images stacked up to form a landscape. Between the background layer (the sky) and the foreground (trees), we placed a yellow circle representing the sun. Since we will be moving the sun around in a moment, we anchored the center of the object to an empty item without physical dimensions so that we can set the sun's position relative to its center. We also equipped the scene with a `dayLength` property, which will hold information about the length of one day of game time. By default, we set it to 60 seconds so that things happen really quickly and we can see the animation's progress without waiting. After all things are set correctly, the length of the day can be balanced to fit our needs.

The graphical design lets us easily manipulate the sun while keeping it behind the tree line. Note how the stacking order is implicitly determined by the order of elements in the document.

# Time for action – Animating the sun's horizontal movement

The everyday cruise of the sun in the sky starts in the east and continues west to hide beneath the horizon in the evening. Let's try to replicate this horizontal movement by adding animation to our `sun` object.

Open the QML document of our last project. Inside the `root` item, add the following declaration:

```cpp
NumberAnimation {
    target: sun
    property: "x"
    from: 0
    to: root.width
    duration: dayLength
    running: true
} 
```

Running the program with such modifications will produce a run with a horizontal movement of the sun. The following image is a composition of a number of frames of the run:

![](img/37a27ca1-c6ad-46c3-9ce6-dbe6356e30c2.png)

# What just happened?

We introduced a `NumberAnimation` element that is set to animate the `x` property of the `sun` object. The animation starts at `0` and lasts until `x` reaches the `root` item's width (which is the right edge of the scene). The movement lasts for `dayLength` milliseconds. The `running` property of the animation is set to `true` to enable the animation. Since we didn't specify otherwise, the motion is linear.

You may be thinking that the animation runs in the wrong direction—"west" is on the left and "east" is on the right, yes? That's true, however, only if the observer faces north. If that were the case for our scene, we wouldn't be seeing the sun at all—at noon, it crosses the south direction.

# Composing animations

The animation we made in the last section looks OK but is not very realistic. The sun should rise in the morning, reach its peak sometime before noon, and then, sometime later, start setting toward the evening, when it should cross the horizon and hide beneath the landscape.

To achieve such an effect, we can add two more animations for the `y` property of the sun. The first animation would start right at the beginning and decrease the vertical position of the sun (remember that the vertical geometry axis points down, so decreasing the vertical position means the object goes up). The animation would be complete at one-third of the day length. We would then need a way to wait for some time and then start a second animation that would pull the object down toward the ground. Starting and stopping the animation is easy—we can either call the `start()` and `stop()` functions on the animation item or directly alter the value of the `running` property. Each `Animation` object emits `started()` and `stopped()` signals. The delay can be implemented using a timer. We can provide a signal handler for the stopped signal of the first animation to trigger a timer to start the other one like this:

```cpp
NumberAnimation {
    id: sunGoesUpAnim
    // ...
    onStopped: sunGoesDownAnimTimer.start()
}
Timer {
    id: sunGoesDownAnimTimer
    interval: dayLength / 3
    onTriggered: sunGoesDownAnim.start()
} 
```

Even ignoring any side problems this would bring (for example, how to stop the animation without starting the second one), such an approach couldn't be called "declarative", could it?

Fortunately, similar to what we had in C++, Qt Quick lets us form animation
groups that run either parallel to each other or in sequence. There are the `SequentialAnimation` and `ParallelAnimation` types where you can declare any number of child animation elements forming the group. To run two animations in parallel, we can declare the following hierarchy of elements:

```cpp
ParallelAnimation {
    id: parallelAnimationGroup
    running: true

    NumberAnimation {
        target: obj1; property: "prop1"
        from: 0; to: 100
        duration: 1500
    }
    NumberAnimation {
        target: obj2; property: "prop2"
        from: 150; to: 0
        duration: 1500
    }
} 
```

The same technique can be used to synchronize a larger group of animations, even if each component has a different duration:

```cpp
SequentialAnimation {
    id: sequentialAnimationGroup
    running: true

    ParallelAnimation {
        id: parallelAnimationGroup

        NumberAnimation {
            id: animation1
            target: obj2; property: "prop2"
            from: 150; to: 0
            duration: 1000
        }
        NumberAnimation {
            id: animation2
            target: obj1; property: "prop1"
            from: 0; to: 100
            duration: 2000
        }
    }
    PropertyAnimation {
        id: animation3
        target: obj1; property: "prop1"
        from: 100; to: 300
        duration: 1500
    }
} 
```

The group presented in the snippet consists of three animations. The first two animations are executed together as they form a parallel subgroup. One member of the group runs twice as long as the other. Only after the whole subgroup completes is the third animation started. This can be visualized using a **Unified Modeling Language** (**UML**) activity diagram where the size of each activity is proportional to the duration of that activity:

![](img/cb2e67ae-f394-456e-b89f-0d93302f1031.png)

# Time for action – Making the sun rise and set

Let's add vertical movement (animation of the `y` property) to our sun by adding a sequence of animations to the QML document. As our new animations will be running in parallel to the horizontal animation, we can enclose animations for both directions within a single `ParallelAnimation` group. It would work, but in our opinion, this will unnecessarily clutter the document. Another way of specifying parallel animations is to declare them as separate hierarchies of elements, making each animation independent of the other, and that is what we will do here.

Open our document from the last exercise, and right under the previous animation, place the following code:

```cpp
SequentialAnimation {
    running: true
    NumberAnimation {
        target: sun
        property: "y"
        from: root.height + sunVisual.height
        to: root.height - 270
        duration: dayLength / 3
    }
    PauseAnimation { duration: dayLength / 3 }
    NumberAnimation {
        target: sun
        property: "y"
        from: root.height - 270
        to: root.height + sunVisual.height
        duration: dayLength / 3
    }
} 
```

Running the program will result in the light source rising in the morning and setting in the evening. However, the trajectory of the move seems somewhat awkward:

![](img/7f99c344-bada-4e58-855f-1ad90e001ad2.png)

# What just happened?

We declared a sequential animation group consisting of three animations, each taking one-third of the day length. The first member of the group makes the sun go up. The second member, which is an instance of a new element type—`PauseAnimation`—introduces a delay equal to its duration. This, in turn, lets the third component start its work in the afternoon to pull the sun down toward the horizon.

The problem with such a declaration is that the sun moves in a horribly angular way, as can be seen in the image.

# Non-linear animations

The reason for the described problem is that our animations are linear. As we noted at the beginning of this chapter, linear animations rarely occur in nature, which usually makes their use yield a very unrealistic result.

We also said earlier that Qt Quick allows us to use easing curves to perform animations along non-linear paths. There are a large number of curves offered. Here's a diagram listing the available non-linear easing curves:

![](img/a3beebc3-9ef9-47c7-99c7-9e60bfafe57f.png)

You can use any of the curves on an element of the `PropertyAnimation` type or one derived from it (for example, `NumberAnimation`). This is done using the `easing` property group, where you can set the `type` of the curve. Different curve types may further be tweaked by setting a number of properties in the `easing` property group, such as `amplitude` (for bounce and elastic curves), `overshoot` (for back curves), or `period` (for elastic curves).

Declaring an animation along an `InOutBounce` path is very easy:

```cpp
NumberAnimation {
    target: obj
    property: prop
    from: startValue
    to: endValue
    easing.type: Easing.InOutBounce
    running: true
} 
```

# Time for action – Improving the path of the sun

The task at hand will be to improve the animation of the sun so that it behaves in a more realistic way. We will do this by adjusting the animations so that the object moves over a curved path.

In our QML document, replace the previous vertical animation with the following one:

```cpp
SequentialAnimation {
    running: true
    NumberAnimation {
        target: sun
        property: "y"
        from: root.height + sunVisual.height
        to: root.height - 270
        duration: dayLength / 2
        easing.type: Easing.OutCubic
    }
    NumberAnimation {
        target: sun
        property: "y"
        to: root.height + sunVisual.height
        duration: dayLength / 2
        easing.type: Easing.InCubic
    } 
}
```

The following picture shows how the sun will now move:

![](img/79c81a48-95e9-48f6-9481-9af3d8461be7.png)

# What just happened?

The sequence of three animations (two linear ones and a pause) was replaced by another sequence of two animations that follow a path determined by a cubic function. This makes our sun rise pretty fast and then slow down to an amount almost unnoticeable near the moment when the sun approaches noon. When the first animation is finished, the second one reverses the motion, making the sun descend very slowly and then increase its velocity as dusk approaches. As a result, the farther the sun is from the ground, the slower it seems to move. At the same time, the horizontal animation remains linear, as the speed of earth in its motion around the sun is practically constant. When we combine the horizontal and vertical animations, we get a path that looks very similar to what we can observe in the real world.

# Property value sources

From the QML perspective, `Animation` and element types derived from it are called **property value sources**. This means they can be attached to a property and generate values for it. What is important is that it allows us to use animations using a much simpler syntax. Instead of explicitly declaring the target and property of an animation, you can attach the animation to a named property of the parent object.

To do this, instead of specifying `target` and `property` for `Animation`, use the `on` keyword, followed by the name of a property name for which the animation is to be a value source. For example, to animate the `rotation` property of an object with a `NumberAnimation` object, the following code can be used:

```cpp
NumberAnimation on rotation {
    from: 0
    to: 360
    duration: 500
} 
```

It is valid to specify more than one property value source for the same property of an object.

# Time for action – Adjusting the sun's color

If you look at the sun at dusk or dawn, you will see that it is not yellow but becomes red the closer it is to the horizon. Let's teach our object representing the sun to do the same by providing a property value source for it.

Open the QML document, find the declaration for the `sunVisual` object, and extend it with the highlighted part:

```cpp
Rectangle {
    id: sunVisual
    // ...
    SequentialAnimation on color {
        ColorAnimation {
            from: "red"
            to: "yellow"
            duration: 0.2 * dayLength / 2
        }
        PauseAnimation { 
            duration: 2 * 0.8 * dayLength / 2 
        }
        ColorAnimation {
            to: "red"
            duration: 0.2 * dayLength / 2
        }
        running: true
    }
} 
```

# What just happened?

An animation was attached to the `color` property of our rectangle modeling the visual aspects of the sun. The animation consists of three parts. First, we perform a transition from red to yellow using the `ColorAnimation` object. This is an `Animation` subtype dedicated to modifying colors. Since the rectangle color is not a number, using the `NumberAnimation` object will not work, as the type cannot interpolate color values. Therefore, we either have to use the `PropertyAnimation` or the  `ColorAnimation`  object. The duration for the animation is set to 20 percent of half the day length so that the yellow color is obtained very quickly. The second component is a `PauseAnimation` object to provide a delay before the third component is executed, which gradually changes the color back to red. For the last component, we do not provide a value for the `from` property. This causes the animation to be initiated with the value of the property current to the time when the animation is executed (in this case, the sun should be yellow).

Note that we only had to specify the property name for the top-level animation. This particular element is what serves as the property value source, and all descendant animation objects "inherit" the target property from that property value source.

# Time for action – Furnishing sun animation

The animation of the sun looks almost perfect right now. We can still improve it, though. If you look into the sky in the early morning and then again at noon, you will note that the sun appears much bigger during sunrise or sunset compared to its size when it is at its zenith. We can simulate that effect by scaling the object.

In our scene document, add another sequential animation that operates on the `scale` property of the sun:

```cpp
SequentialAnimation on scale {
    NumberAnimation {
      from: 1.6; to: 0.8
      duration: dayLength / 2
      easing.type: Easing.OutCubic
    }
    NumberAnimation {
      from: 0.8; to: 1.6
      duration: dayLength / 2
      easing.type: Easing.InCubic
    }
}
```

Let's examine the result again:

![](img/483fbc98-cf8f-4e18-b982-1463b56c5260.png)

# What just happened?

In this section, we just followed the path set for an earlier declaration—the vertical movement of the stellar body influences its perceived size; therefore, it seems like a good decision to bind the two animations together. Note that instead of specifying a new property value source for the scale, we might have modified the original animation and made the scale animation parallel to the animation that operates on the `y` property:

```cpp
SequentialAnimation {
    ParallelAnimation {
        NumberAnimation {
            target: sun
            property: "y"
            from: root.height + sunVisual.height
            to: root.height - 270
            duration: dayLength / 2
            easing.type: Easing.OutCubic
        }
        NumberAnimation {
            target: sun
            property: "scale"
            from: 1.6; to: 0.8
            duration: dayLength / 2
            easing.type: Easing.OutCubic
        } 
        // ... 
    }
}
```

# Have a go hero – Animating the sun's rays

By now, you should be an animation expert. If you want to try your skills, here's a task for you. The following code can be applied to the `sun` object and will display very simple red rays emitted from the sun:

```cpp
Item { 
    id: sunRays 
    property int count: 10 
    width: sunVisual.width 
    height: width 
    anchors.centerIn: parent 
    z: -1 
    Repeater { 
        model: sunRays.count 
        Rectangle { 
            color: "red" 
            rotation: index * 360 / sunRays.count 
            anchors.fill: parent 
        }
    }
} 
```

The result is shown on the following picture:

![](img/517c197a-240c-4c29-9961-8d5f425ec593.png)

The goal is to animate the rays so that the overall effect looks good and fits the tune like style of the scene. Try different animations—rotations, size changes, and colors. Apply them to different elements—all rays at once (for example, using the `sunRays` identifier) or only particular rectangles generated by the repeater.

# Behaviors

In the previous chapter, we implemented a dashboard for a racing game where we had a number of clocks with needles. We could set values for each clock (for example, car speed) and a respective needle would immediately set itself to the given value. However, such an approach is unrealistic—in the real world, changes of a value happen over time. In our example, the car accelerates from 10 mph to 50 mph by going through 11 mph, 12 mph, and so on, until after some time it reaches the desired value. We call this the **behavior** of a value—it is essentially a model that tells how the parameter reaches its destined value. Defining such models is a perfect use case for declarative programming. Fortunately, QML exposes a `Behavior` element that lets us model behaviors of property changes in Qt Quick.

The `Behavior` elements let us associate an animation with a given property so that every time the property value is to be changed, it is done by running the given animation instead of by making an immediate change to the property value.

Consider a simple scene defined by the following code:

```cpp
import QtQuick 2.9

Item {
    width: 600; height: width
    Item {
        id: empty
        x: parent.width / 2; y: parent.height / 2
        Rectangle {
            id: rect
            width: 100; height: width
            color: "red"
            anchors.centerIn: parent
        }
    }
    MouseArea {
        anchors.fill: parent
        onClicked: { 
            empty.x = mouse.x;
            empty.y = mouse.y;
        }
    }
} 
```

This scene contains a red rectangle anchored to an empty item. Whenever the user clicks somewhere within the scene, the empty item is moved there, dragging along the rectangle. Let's see how to use the `Behavior` element to smoothly change the position of the empty item. Similar to `Animation` and other property value sources, the `Behavior` element can be used with the on-property syntax:

```cpp
Item {
    id: empty
    x: parent.width / 2; y: parent.height / 2
    Rectangle {
        id: rect
        width: 100; height: width
        color: "red"
        anchors.centerIn: parent
    }
 Behavior on x { 
 NumberAnimation { } 
 }
 Behavior on y { 
 NumberAnimation { } 
 }
} 
```

By adding the two marked declarations, we define behaviors for the `x` and `y` properties that follow animations defined by `NumberAnimation`. We do not include start or end values for the animation as these will depend on the initial and final value for the property. We also don't set the property name in the animation because by default, the property for which the behavior is defined will be used. As a result, we get a linear animation of a numerical property from the original value to the destined value over the default duration.

Using linear animations for real-world objects rarely looks good. Usually, you will get much better results if you set an easing curve for the animation so that it starts slowly and then gains speed and decelerates just before it is finished.

Animations that you set on behaviors can be as complex as you want:

```cpp
Behavior on x {
    SequentialAnimation {
        PropertyAction {
            target: rect
            property: "color"
            value: "yellow"
        }
        ParallelAnimation {
            NumberAnimation { 
                easing.type: Easing.InOutQuad
                duration: 1000
            } 
            SequentialAnimation {
                NumberAnimation {
                    target: rect
                    property: "scale"
                    from: 1.0; to: 1.5
                    duration: 500
                }
                NumberAnimation {
                    target: rect
                    property: "scale"
                    from: 1.5; to: 1.0
                    duration: 500
                }
            }
        }
        PropertyAction { 
            target: rect
            property: "color"
            value: "red" 
        }
    }
} 
```

The behavioral model declared in the last piece of code performs a sequential animation. It first changes the color of the rectangle to yellow using the `PropertyAction` element, which performs an immediate update of a property value (we will talk about this more a bit later). The color will be set back to red after the last step of the model. In the meantime, a parallel animation is performed. One of its components is a `NumberAnimation` class that executes the actual animation of the `x` property of `empty` (since the target and property of the animation are not explicitly set). The second component is a sequential animation of the `scale` property of the rectangle, which first scales the item up by 50 percent during the first half of the animation and then scales it back down in the second half of the animation.

# Time for action – Animating the car dashboard

Let's employ the knowledge we just learned to improve the car dashboard we created in the previous chapter. We will use animations to show some realism in the way the clocks update their values.

Open the dashboard project and navigate to the `main.qml` file. Find the declaration of the `Needle` object, which is responsible for visualizing the speed of the vehicle. Add the following declaration to the object:

```cpp
Behavior on rotation {
    SmoothedAnimation { 
        velocity: 50 
    }
} 
```

Repeat the process for the left clock. Set the velocity of the animation to `100`. Build and run the project. See how the needles behave when you modify the parameter values in spin boxes. Adjust the `velocity` of each animation until you get a realistic result.

# What just happened?

We have set the property value sources on needle rotations that are triggered whenever a new value for the property is requested. Instead of immediately accepting the new value, the `Behavior` element intercepts the request and starts the `SmoothedAnimation` class to gradually reach the requested value. The `SmoothedAnimation` class is an animation type that animates numeric properties. The speed of the animation is not determined by its duration; instead, a `velocity` property is set. This property dictates how fast a value is to be changed. However, the animation is using a non-linear path—it starts slowly, then accelerates to the given velocity, and, near the end of the animation, decelerates in a smooth fashion. This yields an animation that is attractive and realistic and, at the same time, is of shorter or longer duration, depending on the distance between the starting and ending values.

You can implement custom property value sources by subclassing `QQmlPropertyValueSource` and registering the class in the QML engine.

# States

When you look at real-world objects, it is often very easy to define their behavior by extracting a number of states the object may take and describing each of the states separately. A lamp can be turned either on or off. When it is "on", it is emitting light of a given color, but it is not doing that when in the "off" state. Dynamics of the object can be defined by describing what happens if the object leaves one of the states and enters another one. Considering our lamp example, if you turn the lamp on, it doesn't momentarily start emitting light with its full power, but the brightness of the light gradually increases to reach its final power after a very short period.

Qt Quick supports *state-driven* development by letting us declare states and transitions between them for items. The model fits the declarative nature of Qt Quick very well.

By default, each item has a single anonymous state, and all properties you define take values of the expressions you bind or assign to them imperatively based on different conditions. Instead of this, a set of states can be defined for the object and for each of the state properties of the object itself; in addition, the objects defined within it can be programmed with different values or expressions. Our example lamp definition could be similar to this:

```cpp
Item {
    id: lamp
    property bool lampOn: false
    width: 200
    height: 200
    Rectangle {
        id: lightsource
        anchors.fill: parent
        color: "transparent"
    }
} 
```

We could, of course, bind the `color` property of `lightsource` to `lamp.lampOn ? "yellow" : "transparent"`; instead, we can define an "on" state for the lamp and use a `PropertyChanges` element to modify the rectangle color:

```cpp
Item {
    id: lamp
    property bool lampOn: false
    // ...
    states: State {
 name: "on"
 PropertyChanges {
 target: lightsource
 color: "yellow"
 }
 }
} 
```

Each item has a `state` property that you can read to get the current state, but you can also write to it to trigger transition to a given state. By default, the `state` property is set to an empty string that represents the anonymous state. Note that with the preceding definition, the item has two states—the "on" state and the anonymous state (which is used when the lamp is off in this case). Remember that state names have to be unique as the `name` parameter is what identifies a state in Qt Quick.

To enter a state, we can, of course, use an event handler fired when the value of the `lampOn` parameter is modified:

```cpp
onLampOnChanged: state = lampOn ? "on" : "" 
```

Such imperative code works, but it can be replaced with a declarative definition in the state itself:

```cpp
State {
    name: "on"
    when: lamp.lampOn
    PropertyChanges {
        target: lightsource
        color: "yellow"
    }
} 
```

Whenever the expression bound to the `when` property evaluates to `true`, the state becomes active. If the expression becomes `false`, the object will return to the default state or will enter a state for which its own `when` property evaluates to `true`.

To define more than one custom state, it is enough to assign a list of state definitions to the `states` property:

```cpp
states: [
    State {
        name: "on"
        when: lamp.lampOn
        PropertyChanges { /*...*/ }

    },
    State {
        name: "off"
        when: !lamp.lampOn
    }
] 
```

The `PropertyChanges` element is the most often used change in a state definition, but it is not the only one. In exactly the same way that the `ParentChange` element can assign a different parent to an item and the `AnchorChange` element can update anchor definitions, it is also possible to run a script when a state is entered using the `StateChangeScript` 
element. All these element types are used by declaring their instances as children in a `State` object.

# Transitions

The second part of the state machine framework is defining how an object transits from one state to another. Similar to the `states` property, all items have a `transitions`  property, which takes a list of definitions represented by the `Transition` objects and provides information about animations that should be played when a particular transition takes place.

A transition is identified by three attributes—the source state, the destination state, and a set of animations. Both the source state name (set to the `from` property) and the target state name (set to the `to` property) can be empty, in which case they should be interpreted as "any". If a `Transition` exists that matches the current state change, its animations will be executed. A more concrete transition definition (which is one where `from` and/or `to` are explicitly set) has precedence over a more generic one.

Suppose that we want to animate the opacity of the lamp rectangle from `0` to `1` when the lamp is switched on. We can do it as an alternative to manipulating the color. Let's update the lamp definition:

```cpp
Item {
    id: lamp
    property bool lampOn: false
    Rectangle {
        id: lightsource
        anchors.fill: parent
        color: "yellow"
        opacity: 0
    }
    MouseArea {
        anchors.fill: parent
        onPressed: {
            lamp.lampOn = !lamp.lampOn;
        }
    }
    states: State {
        name: "on"
        when: lamp.lampOn
        PropertyChanges {
            target: lightsource
            opacity: 1
        }
    }
    transitions: Transition {
        NumberAnimation {
            duration: 500
            property: "opacity"
        }
    }
} 
```

The transition is triggered for any source and any target state—it will be active when the lamp goes from the anonymous to the "on" state as well as in the opposite direction. It defines a single `NumberAnimation` element that works on `opacity` property and lasts for 500 miliseconds. The animation does not define the target object; thus, it will be executed for any object that needs updating as part of the transition—in the case of the lamp, it will only be the `lightsource` object.

If more than one animation is defined in a transition, all animations will run in parallel. If you need a sequential animation, you need to explicitly use a `SequentialAnimation` element:

```cpp
Transition {
    SequentialAnimation {
        NumberAnimation { 
            target: lightsource
            property: "opacity"            
            duration: 500 
        }
        ScriptAction { 
            script: {
                console.log("Transition has ended");
            }
        }
    }
} 
```

States are a feature of all `Item` types as well as its descendent types. It is, however, possible to use states with elements not derived from the `Item` object using a `StateGroup` element, which is a self-contained functionality of states and transitions with exactly the same interface as what is described here regarding `Item` objects.

# More animation types

The animation types we discussed earlier are used for modifying values of types that can be described using physical metrics (position, sizes, colors, angles). However, there are more types available.

The first group of special animations consists of the `AnchorAnimation` and  `ParentAnimation` elements.

The `AnchorAnimation` element is useful if a state change should cause a change to defined anchors for an item. Without it, the item would immediately snap into its place. By using the `AnchorAnimation` element, we trigger all anchor changes to be gradually animated.

The `ParentAnimation` element, on the other hand, makes it possible to define animations that should be present when an item receives a new parent. This usually causes an item to be moved to a different position in the scene. By using the `ParentAnimation` element in a state transition, we can define how the item gets into its target position. The element can contain any number of child animation elements that will be run in parallel during a `ParentChange` element.

The second special group of animations is action animations—`PropertyAction`  and `ScriptAction`. These animation types are not stretched in time but perform a given one-time action.

The `PropertyAction` element is a special kind of animation that performs an immediate update of a property to a given value. It is usually used as part of a more complex animation to modify a property that is not animated. It makes sense to use it if a property needs to have a certain value during an animation.

`ScriptAction` is an element that allows the execution of an imperative piece of code during an animation (usually at its beginning or end).

# Quick game programming

Here, we will go through the process of creating a platform game using Qt Quick. It will be a game similar to Benjamin the Elephant from [Chapter 6](ad89bd0a-2ed2-49a8-8701-18449629a1ee.xhtml), *Qt Core Essentials.* The player will control a character that will be walking through the landscape and collecting coins. The coins will be randomly appearing in the world. The character can access highly placed coins by jumping.

Throughout this chapter as well as the previous one, we prepared a number of pieces that we will be reusing for this game. The layered scene that was arranged when you learned about animations will serve as our game scene. The animated sun will represent the passing of time.

We will guide you through implementing the main features of the game. At the end of the chapter, you will have a chance to test your skills by adding more game mechanics to our project.

# Game loops

Most games revolve around some kind of game loop. It is usually some kind of function that is called repeatedly, and its task is to progress the game—process input events, move objects around, calculate and execute actions, check win conditions, and so on. Such an approach is very imperative and usually results in a very complex function that needs to know everything about everybody (this kind of anti-pattern is sometimes called a **god object** pattern). In QML (which powers the Qt Quick framework), we aim to separate responsibilities and declare well-defined behaviors for particular objects. Therefore, although it is possible to set up a timer that will periodically call a game loop function, this is not the best possible approach in a declarative world.

Instead, we suggest using a natural time-flow mechanism already present in Qt Quick—one that controls the consistency of animations. Remember how we defined the sun's travel across the sky at the beginning of this chapter? Instead of setting up a timer and moving the object by a calculated number of pixels, we created an animation, defined a total running time for it, and let Qt take care of updating the object. This has the great benefit of neglecting delays in function execution. If you used a timer and some external event introduced a significant delay before the timeout function was run, the animation would start lagging behind. When Qt Quick animations are used, the framework compensates for such delays, skipping some of the frame updates to ensure that the requested animation duration is respected. Thanks to that, you will not have to take care of it all by yourself.

To overcome the second difficult aspect of a game loop—the god object anti-pattern—we suggest encapsulating the logic of each item directly in the item itself using the states and transitions framework we introduced earlier. If you define an object using a natural time flow describing all states it can enter during its lifetime and actions causing transitions between states, you will be able to just plop the object with its included behavior wherever it is needed and thus easily reuse such definitions in different games, reducing the amount of work necessary to make the object fit into the game.

# Input processing

A usual approach in games is to read input events and call functions responsible for actions associated with particular events:

```cpp
void Scene::keyEvent(QKeyEvent *event) {
    switch(event->key()) {
    case Qt::Key_Right: 
        player->goRight(); break;
    case Qt::Key_Left:  
        player->goLeft();  break;
    case Qt::Key_Space: 
        player->jump();    break;
    // ...
    }
} 
```

This, however, has its drawbacks, one of which is the need to check events at even periods of time. This might be hard and is certainly not a declarative approach.

We already know that Qt Quick handles keyboard input via the `Keys` attached property. It is possible to craft QML code similar to the one just presented, but the problem with such an approach is that the faster the player taps keys on the keyboard, the more frequently the character will move, jump, or shoot. However, it's possible to overcome this problem, as we'll see as we move on.

# Time for action – Character navigation

Create a new QML document and call it `Player.qml`. In the document, place the following declarations:

```cpp
Item {
    id: player
    y: parent.height
    focus: true

    Keys.onRightPressed: x = Math.min(x + 20, parent.width)
    Keys.onLeftPressed: x = Math.max(0, x - 20)
    Keys.onUpPressed: jump()

    function jump() { 
        jumpAnim.start();
    }

    Image {
        source: "images/elephant.png"
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
    }
    Behavior on x { 
        NumberAnimation { duration: 100 } 
    }
    SequentialAnimation on y {
        id: jumpAnim
        running: false
        NumberAnimation { 
            to: player.parent.height - 50
            easing.type: Easing.OutQuad 
        } 
        NumberAnimation { 
            to: player.parent.height
            easing.type: Easing.InQuad 
        } 
    }
} 
```

Next, open the document containing the main scene definition and declare the player character near the end of the document after all the background layers have been declared:

```cpp
Player {
    id: player
    x: 40
} 
```

# What just happened?

The player itself is an empty item with a keyboard focus that handles presses of the right, left, and up arrow keys, causing them to manipulate the `x` and `y` coordinates of the player. The `x` property has a `Behavior` element set so that the player moves smoothly within the scene. Finally, anchored to the player item is the actual visualization of the player—our elephant friend.

When the right or left arrow keys are pressed, a new position for the character will be calculated and applied. Thanks to the `Behavior` element, the item will travel gradually (during one second) to the new position. Keeping the key pressed will trigger autorepeat and the handler will be called again. In a similar fashion, when the up arrow key is pressed, it will activate a prepared sequential animation that will lift the character up by 50 pixels and then move it down again to the initial position.

This approach works, but we can do better. Let's try something different.

# Time for action – Another approach to character navigation

Replace the previous key handlers with the following code:

```cpp
Item {
    id: player
    //...
    QtObject {
        id: flags
        readonly property int speed: 100
        property int horizontal: 0
    }
    Keys.onRightPressed: { 
        recalculateDurations(); 
        flags.horizontal = 1; 
    }
    Keys.onLeftPressed: {
        if(flags.horizontal != 0) {
            return;
        }
        recalculateDurations();
        flags.horizontal = -1;
    }
    Keys.onUpPressed: jump()
    Keys.onReleased: {
        if(event.isAutoRepeat) return;
        if(event.key === Qt.Key_Right) {
            flags.horizontal = 0;
        }
        if(event.key === Qt.Key_Left && flags.horizontal < 0) {
            flags.horizontal = 0;
        }
    }

    function recalculateDurations() {
        xAnimRight.duration = (xAnimRight.to - x) * 1000 / flags.speed;
        xAnimLeft.duration = (x - xAnimLeft.to) * 1000 / flags.speed;
    }
    NumberAnimation on x {
        id: xAnimRight
        running: flags.horizontal > 0
        to: parent.width
    }
    NumberAnimation on x {
        id: xAnimLeft
        running: flags.horizontal < 0
        to: 0
    } 
}
```

# What just happened?

Instead of performing actions immediately, upon pressing a key, we are now setting flags (in a private object) for which direction the character should be moving in. In our situation, the right direction has priority over the left direction. Setting a flag triggers an animation that tries to move the character toward an edge of the scene. Releasing the button will clear the flag and stop the animation. Before the animation is started, we are calling the `recalculateDurations()` function, which checks how long the animation should last for the character to move at the desired speed.

If you want to replace keyboard-based input with something else, for example, accelerometer or custom buttons, the same principle can be applied. When using an accelerometer, you can even control the speed of the player by measuring how much the device is tilted. You can additionally store the tilt in the `flags.horizontal` parameter and make use of that variable in the `recalculateDurations()` function.

# Have a go hero – Polishing the animation

What we have done is sufficient for many applications. However, you can try controlling the movement even more. As a challenge, try modifying the system in such a way that during a jump, inertia keeps the current horizontal direction and speed of movement of the character until the end of the jump. If the player releases the right or left keys during a jump, the character will stop only after the jump is complete.

Despite trying to do everything in a declarative fashion, some actions will still require imperative code. If some action is to be executed periodically, you can use the `Timer` item to execute a function on demand. Let's go through the process of implementing such patterns together.

# Time for action – Generating coins

The goal of the game we are trying to implement is to collect coins. We will spawn coins now and then in random locations of the scene.

Create a new QML Document and call it `Coin.qml`. In the editor, enter the following code:

```cpp
Item {
    id: coin

    Rectangle {
        id: coinVisual
        color: "yellow"
        border.color: Qt.darker(color)
        border.width: 2
        width: 30; height: width
        radius: width / 2
        anchors.centerIn: parent

        transform: Rotation {
            origin.x: coinVisual.width / 2
            origin.y: coinVisual.height / 2
            axis { x: 0; y: 1; z: 0 }            
            NumberAnimation on angle {
                from: 0; to: 360
                loops: Animation.Infinite
                running: true
                duration: 1000
            }
        }
        Text {
            color: coinVisual.border.color
            anchors.centerIn: parent
            text: "1"
        }
    }
} 
```

Next, open the document where the scene is defined and enter the following code somewhere in the scene definition:

```cpp
Component {
    id: coinGenerator
    Coin {}
}

Timer {
    id: coinTimer
    interval: 1000
    repeat: true
    running: true

    onTriggered: {
        var cx = Math.floor(Math.random() * root.width);
        var cy = Math.floor(Math.random() * root.height / 3)
               + root.height / 2;
        coinGenerator.createObject(root, { x: cx, y: cy });
    }
} 
```

# What just happened?

First, we defined a new element type, `Coin`, consisting of a yellow circle with a number centered over an empty item. The rectangle has an animation applied that rotates the item around a vertical axis, resulting in a pseudo three-dimensional effect.

Next, a component able to create instances of a `Coin` element is placed in the scene. Then, a `Timer` element is declared that fires every second and spawns a new coin at a random location of the scene.

# Sprite animation

The player character as well as any other component of the game should be animated. If the component is implemented using simple Qt Quick shapes, it is quite easy to do by changing the item's properties fluently, using property animations (as we did with the `Coin` object). Things get more difficult if a component is complex enough that it is easier to draw it in a graphics program and use an image in the game instead of trying to recreate the object using Qt Quick items. Then, you need a number of images—one for every frame of animation. Images would have to keep replacing one another to make a convincing animation.

# Time for action – Implementing simple character animation

Let's try to make the player character animated in a simple way. In the materials that come with this book, you will find a number of images with different walking phases for Benjamin the Elephant. You can use them, or you can draw or download some other images to be used in place of those provided by us.

Put all images in one directory (for example, `images`) and rename them so that they follow a pattern that contains the base animation name followed by a frame number, for example, `walking_01`, `walking_02`, `walking_03`, and so on.

Next, open the `Player.qml` document and replace the image element showing `elephant.png` with the following code:

```cpp
Image {
    id: elephantImage
    property int currentFrame: 1
    property int frameCount: 7
    source: "images/walking_" + currentFrame + ".png"
    mirror: player.facingLeft

    anchors.bottom: parent.bottom
    anchors.horizontalCenter: parent.horizontalCenter
    NumberAnimation on currentFrame {
        from: 1
        to: frameCount
        loops: Animation.Infinite
        duration: elephantImage.frameCount * 40
        running: player.walking
    }
} 
```

In the root element of `Player.qml`, add the following properties:

```cpp
property bool walking: flags.horizontal !== 0
property bool facingLeft: flags.horizontal < 0 
```

Start the program and use the arrow keys to see Benjamin move.

# What just happened?

A number of images were prepared following a common naming pattern containing a number. All the images have the same size. This allows us to replace one image with another just by changing the value of the `source` property to point to a different image. To make it easier, we introduced a property called the `currentFrame` element that contains the index of the image to be displayed. We used the `currentFrame` element in a string, forming an expression bound to the `source` element of the image. To make substituting frames easy, a `NumberAnimation` element was declared to modify the values of the `currentFrame` element in a loop from `1` to the number of animation frames available (represented by the `frameCount` property) so that each frame is shown for 40 milliseconds.

The animation is playing if the `walking` property evaluates to `true` (based on the value of the `flags.horizontal` element in the player object). Finally, we use the `mirror` property of the `Image` parameter to flip the image if the character is walking left.

The preceding approach works, but it's not perfect. The complexity of the declaration following this pattern grows much faster than required when we want to make movement animation more complex (for example, if we want to introduce jumping). This is not the only problem, though. Loading images does not happen instantly. The first time a particular image is to be used, the animation can stall for a moment while the graphics get loaded, which may ruin the user experience. Lastly, it is simply messy to have a bunch of pictures here and there for every image animation.

A solution to this is to use a **sprite sheet**—a set of small images combined into a single larger image for better performance. Qt Quick supports sprite sheets through its sprite engine that handles loading sequences of sprites from a single image, animating them, and transitioning between different sprites.

In Qt Quick, a sprite sheet can be an image of any type supported by Qt that contains an image strip with all frames of the animation. Subsequent frames should form a continuous line flowing from left to right and from the top to the bottom of the image. However, they do not have to start in the top-left corner of the containing image, nor do they have to end in its bottom-right corner—a single file can contain many sprites. A sprite is defined by providing the size of a single frame in pixels and a frame count. Optionally, you can specify an offset from the top-left corner where the first frame of the sprite is to be read from. The following diagram can be helpful in visualizing the scheme:

![](img/bed79f67-221f-4f3a-a5e2-c30fb7752839.png)

QML offers a `Sprite` element type with a `source` property pointing to the URL of the container image, the `frameWidth` and `frameHeight` properties determining the size of each frame, and a `frameCount` property defining the number of frames in the sprite. Offsetting the image can be achieved by setting values of the `frameX` and `frameY`  properties. In addition to this, some additional properties are present; the most important three are `frameRate`, `frameDuration,` and `duration`. All these serve to determine the pace of the animation. If the `frameRate` element is defined, it is interpreted as a number of frames to cycle through per second. If this property is not defined, then the  `frameDuration` element kicks in and is treated as a period of time in which to display a single frame (thus, it is directly an inverse of the `frameRate` element). If this property is not defined as well, the `duration` element is used, which carries the duration of the whole animation. You can set any of these three properties, but you don't need to set more than one of them.

# Time for action – Animating characters using sprites

Let's wait no further. The task at hand is to replace the manual animation from the previous exercise with a sprite sheet animation.

Open the `Player.qml` document, remove the whole image element responsible for displaying the player character, and add the following code:

```cpp
AnimatedSprite {
    id: sprite
    source: "images/sprite.png"
    frameX: 560
    frameY: 0
    frameWidth: 80
    frameHeight: 52
    frameCount: 7
    frameRate: 10
    interpolate: true
    width: frameWidth
    height: frameHeight

    running: player.walking
    anchors.bottom: parent.bottom
    anchors.horizontalCenter: parent.horizontalCenter

    transform: Scale {
        origin.x: sprite.width / 2
        xScale: player.facingLeft ? -1 : 1
    }
} 
```

# What just happened?

We replaced the previous static image with an ever-changing source with a different item. As the `Sprite` parameter is not an `Item` element but a data definition of a sprite, we cannot use it in place of the `Image` element. Instead, we will use the `AnimatedSprite` element, which is an item that can display a single animated sprite defined inline. It even has the same set of properties as the `Sprite` parameter. We defined a sprite embedded in `images/sprite.png` with a width of `80` and a height of `52` pixels. The sprite consists of seven frames that should be displayed at a rate of 10 frames per second. The  `running` 
property is set up similar to the original `Animation` element. As the  `AnimatedSprite` 
element does not have a `mirror` property, we emulate it by applying a scale transformation that flips the item horizontally if the `player.facingLeft`  expression evaluates to `true`. Additionally, we set the `interpolate` property to  `true`, which makes the sprite engine calculate smoother transitions between frames.

The result we are left with is similar to an earlier attempt, so if these two are similar, why bother using sprites? In many situations, you want more complex animation than just a single-frame sequence. What if we want to animate the way Benjamin jumps in addition to walking? Embedding more manual animations, although possible, would explode the number of internal variables required to keep the state of the object. Fortunately, the Qt Quick sprite engine can deal with that. The `AnimatedSprite` element we used provides a subset of features of the whole framework. By substituting the item with the `SpriteSequence` element, we gain access to the full power of sprites. Whilst we're on the subject of `Sprite`, we need to tell you about one additional property of the object, a property called `to` that contains a map of probabilities of transitioning from the current sprite to another one. By stating which sprites the current one migrates to, we create a state machine with weighted transitions to other sprites as well as cycling back to the current state.

Transitioning to another sprite is triggered by setting the `goalSprite` property on the `SpriteSequence` object. This will cause the sprite engine to traverse the graph until it reaches the requested state. It is a great way to fluently switch from one animation to another by going through a number of intermediate states.

Instead of asking the sprite machine to gracefully transit to a given state, you can ask it to force an immediate change by calling the `SpriteSequence` class's `jumpTo()` method and feeding it the name of the sprite that should start playing.

The last thing that needs to be clarified is how to actually attach the sprite state machine to the `SpriteSequence` class. It is very easy—just assign an array of the `Sprite` objects to the `sprites` property.

# Time for action – Adding jumping with sprite transitions

Let's replace the `AnimatedSprite` class with the `SpriteSequence` class in the Bejamin the Elephant animation, adding a sprite to be played during the jumping phase.

Open the `Player.qml` file and replace the `AnimatedSprite` object with the following code:

```cpp
SpriteSequence {
    id: sprite
    width: 80
    height: 52
    interpolate: false
    anchors.bottom: parent.bottom
    anchors.horizontalCenter: parent.horizontalCenter
    running: true

    Sprite {
        name: "still"
        source: "images/sprite.png"
        frameCount: 1
        frameWidth: 80; frameHeight: 52
        frameDuration: 100
        to: { "still": 1, "walking": 0, "jumping": 0 }
    }
    Sprite {
        name: "walking"
        source: "images/sprite.png"
        frameX: 560; frameY: 0
        frameCount: 7
        frameWidth: 80; frameHeight: 52
        frameRate: 20
        to: { "walking": 1, "still": 0, "jumping": 0 }
    }
    Sprite {
        name: "jumping"
        source: "images/sprite.png"
        frameX: 480; frameY: 52
        frameCount: 11
        frameWidth: 80; frameHeight: 70
        frameDuration: 50
        to: { "still" : 0, "walking": 0, "jumping": 1 }
    }

    transform: Scale {
        origin.x: sprite.width / 2
        xScale: player.facingLeft ? -1 : 1
    }
}
```

Next, extend the `jumpAnim` object by adding the highlighted changes:

```cpp
SequentialAnimation {
    id: jumpAnim
    running: false
 ScriptAction { 
 script: {
 sprite.goalSprite = "jumping";
 }
 }
    NumberAnimation {
        target: player; property: "y"
        to: player.parent.height - 50
        easing.type: Easing.OutQuad
    }
    NumberAnimation {
        target: player; property: "y"
        to: player.parent.height
        easing.type: Easing.InQuad
    }
 ScriptAction {
 script: { 
 sprite.goalSprite = "";
 sprite.jumpTo("still"); 
 }
 }
} 
```

# What just happened?

The `SpriteSequence` element we have introduced has its `Item` elements-related properties set up in the same way as the `AnimatedSprite` element. Apart from that, a sprite called "still" was explicitly set as the current one. We defined a number of `Sprite` objects as children of the `SpriteSequence` element. This is equivalent to assigning those sprites to the `sprites` property of the object. The complete state machine that was declared is presented in the following diagram:

![](img/0a13e4c4-79a2-4640-90a6-dc3eb6a47665.png)

A sprite called "still" has just a single frame representing a situation when Benjamin doesn't move. The sprite keeps spinning in the same state due to the weighted transition back to the "still" state. The two remaining transitions from that state have their weights set to `0`, which means they will never trigger spontaneously, but they can be invoked by setting the `goalSprite` property to a sprite that can be reached by activating one of those transitions.

The sequential animation was extended to trigger sprite changes when the elephant lifts into the air.

# Have a go hero – Making Benjamin wiggle his tail in anticipation

To practice sprite transitions, your goal is to extend the state machine of Benjamin's `SpriteSequence` element to make him wiggle his tail when the elephant is standing still. You can find the appropriate sprite in the materials that come included with this book. The sprite field is called `wiggling.png`. Implement the functionality by making it probable that Benjamin spontaneously goes from the "still" state to "wiggling". Pay attention to ensure that the animal stops wiggling and starts walking the moment the player activates the right or left arrow keys.

# Time for action – Revisiting parallax scrolling

We already discussed the useful technique of parallax scrolling in [Chapter 6](ad89bd0a-2ed2-49a8-8701-18449629a1ee.xhtml), *Qt Core Essentials*. It gives the impression of depth for 2D games by moving multiple layers of background at a different speed depending on the assumed distance of the layer from the viewer. Let's see how easy it is to apply the same technique in Qt Quick.

We will implement parallax scrolling with a set of layers that move in the direction opposite to the one the player is moving in. Therefore, we will need a definition of the scene and a moving layer.

Create a new QML File (Qt Quick 2). Call it `ParallaxScene.qml`. The scene will encompass the whole game "level" and will expose the position of the player to the moving layers. Put the following code in the file:

```cpp
import QtQuick 2.9

Item {
    id: root
    property int currentPos
    x: -currentPos * (root.width - root.parent.width) / width
} 
```

Then, create another QML file and call it `ParallaxLayer.qml`. Make it contain the following definition:

```cpp
import QtQuick 2.9

Item {
    property real factor: 0
    x: factor > 0 ? -parent.currentPos / factor - parent.x : 0
} 
```

Now, let's use the two new element types in the main QML document. We'll take elements from the earlier scene definition and make them into different parallax layers—the sky, the trees, and the grass:

```cpp
Rectangle {
    id: view

    width: 600
    height: 380

    ParallaxScene {
        id: scene
        width: 1500; height: 380
        anchors.bottom: parent.bottom
        currentPos: player.x

        ParallaxLayer {
            factor: 7.5
            width: sky.width; height: sky.height
            anchors.bottom: parent.bottom
            Image { id: sky; source: "images/sky.png" }
            Item {
                 id: sun
                 //...
            }
        }
        ParallaxLayer {
            factor: 2.5
            width: trees.width; height: trees.height
            anchors.bottom: parent.bottom
            Image { id: trees; source: "images/trees.png" }
        }
        ParallaxLayer {
            factor: 0
            width: grass.width; height: grass.height
            anchors.bottom: parent.bottom
            Image { id: grass; source: "images/grass.png" }
        }

        Item {
            id: player
            // ...
        }
        Component {
            id: coinGenerator
            Coin {}
        }
        Timer {
            id: coinTimer
            //...
            onTriggered: {
                var cx = Math.floor(Math.random() * scene.width);
                var cy = Math.floor(Math.random() * scene.height / 3) +
                    scene.height / 2;
                coinGenerator.createObject(scene, { x: cx, y: cy});
            }
        }
    }
}
```

You can now run the game and observe the movement of background layers when the player moves around:

![](img/9fcfb42b-0234-4492-b97b-79ad5f7f5ec9.png)

# What just happened?

The `ParallaxScene` element we implemented is a moving plane. Its horizontal offset depends on the character's current position and the size of the view. The range of scroll of the scene is determined by the difference between the scene size and the view size—it says how much scrolling we have to do when the character moves from the left edge to the right edge of the scene so that it is in view all the time. If we multiply that by the distance of the character from the left edge of the scene expressed as a fraction of the scene width, we will get the needed scene offset in the view (or otherwise speaking, a projection offset of the scene).

The second type—`ParallaxLayer`—is also a moving plane. It defines a distance factor that represents the relative distance (depth) of the layer behind the foreground, which influences how fast the plane should be scrolled compared to the foreground (scene). The value of `0` means that the layer should be moving with exactly the same speed as the foreground layer. The larger the value, the slower the layer moves as compared to the character. The offset value is calculated by dividing the character's position in the scene by the factor. Since the foreground layer is also moving, we have to take it into consideration when calculating the offset for each parallax layer. Thus, we subtract the horizontal position of the scene to get the actual layer offset.

Having the layers logically defined, we can add them to the scene. Each layer has a physical representation in our case, static images containing textures of the sky, trees, and grass. Each layer is defined separately and can live its own life, containing static and animated elements that have no influence on remaining layers. For example, we put the sun object into the sky layer, so it will move along with the sky layer in addition to playing its own animations.

Finally, since we no longer have the `root` element, we modified the `coinTimer` handler to use the `scene` element instead.

# Have a go hero – Vertical parallax sliding

As an additional exercise, you may want to implement a vertical parallax sliding in addition to a horizontal one. Just make your scene bigger and have it expose the vertical scroll position in addition to the horizontal one reported by the `currentPos` element. Then, just repeat all the calculations for the `y` property of each layer and you should be done in no time. Remember that distance factors for `x` and `y` may be different.

# Collision detection

There is no built-in support for collision detection in Qt Quick, but there are three ways of providing such support. First, you can use a ready collision system available in a number of 2D physics engines such as Box2D. Secondly, you can implement a simple collision system yourself in C++. Lastly, you can do collision checking directly in JavaScript by comparing object coordinates and bounding boxes.

Our game is very simple; therefore, we will use the last approach. If we had a larger number of moving objects involved in our game, we would probably choose the second approach. The first approach is best if you have an object of non-rectangular shapes that can rotate and bounce off other objects. In this case, having a physics engine at hand becomes really useful.

# Time for action – Collecting coins

From Qt Creator's menu, access File — New File or Project. From Qt category, choose the JS File template. Call the `collisions.js` file. Put the following content into the document:

```cpp
.pragma library

function boundingBox(object1) {
    var cR = object1.childrenRect;
    var mapped = object1.mapToItem(
         object1.parent, cR.x, cR.y, cR.width, cR.height);
    return Qt.rect(mapped.x, mapped.y, mapped.width, mapped.height);
}

function intersect(object1, object2) {
    var r1 = boundingBox(object1);
    var r2 = boundingBox(object2);
    return (r1.x <= r2.x+r2.width && // r1.left <= r2.right
            r2.x <= r1.x+r1.width && // r2.left <= r1.right
            r1.y <= r2.y+r2.height && // r1.top <= r2.bottom
            r2.y <= r1.y+r1.height); // r2.top <= r1.bottom
} 
```

Create another JS File and call it `coins.js`. Enter the following:

```cpp
.import "collisions.js" as Collisions

var coins = []

coins.collisionsWith = function(player) {
    var collisions = [];
    for(var index = 0; index < coins.length; ++index) {
        var obj = this[index];
        if(Collisions.intersect(player, obj)) {
            collisions.push(obj);
        }
    }
    return collisions;
};
coins.remove = function(obj) {
    var arr = Array.isArray(obj) ? obj : [ obj ];
    var L = arr.length;
    var idx, needle;
    while(L && this.length) {
        needle = arr[--L];
        idx = this.indexOf(needle);
        if(idx !== -1) { 
             this.splice(idx, 1);
        }
    }
    return this;
};
```

Finally, open the `main.qml` file and add the following `import` statement:

```cpp
import "coins.js" as Coins
```

In the player object, define the `checkCollisions()` function:

```cpp
function checkCollisions() {
    var result = Coins.coins.collisionsWith(player);
    if(result.length === 0) return;
    result.forEach(function(coin) { coin.hit() });
    Coins.coins.remove(result) // prevent the coin from being hit again
}
```

Next, modify the `coinTimer` handler to push new coins to the list:

```cpp
Timer {
    id: coinTimer
    //...
    onTriggered: {
        var cx = Math.floor(Math.random() * scene.width);
        var cy = scene.height - 60 - Math.floor(Math.random() * 60);
        var coin = coinGenerator.createObject(scene, { x: cx, y: cy});
        Coins.coins.push(coin);
    }
}
```

Lastly, in the same player object, trigger collision detection by handling the position changes of the player:

```cpp
onXChanged: { 
    checkCollisions();
}
onYChanged: { 
    checkCollisions();
} 
```

In the `Coin.qml` file, define an animation and a `hit()` function:

```cpp
SequentialAnimation {
    id: hitAnim
    running: false
    NumberAnimation {
        target: coin
        property: "opacity"
        from: 1; to: 0
        duration: 250
    }
    ScriptAction {
        script: coin.destroy()
    }
}

function hit() {
    hitAnim.start();
} 
```

# What just happened?

The `collisions.js` file contains functions used to do collision checking. The first line of the file is a `.pragma library` statement, noting that this document only contains functions and does not contain any mutable object. This statement marks the document as a library that can be shared between documents that import it. This aids in reduced memory consumption and improved speed, as the engine doesn't have to reparse and execute the document each time it is imported.

The functions defined in the library are really simple. The first one returns a bounding rectangle of an object based on its coordinates and the size of its children. It assumes that the top-level item is empty and contains children that represent the visual aspect of the object. Children coordinates are mapped using the `mapToItem` function so that the rectangle returned is expressed in the parent item coordinates. The second function does a trivial checking of intersection between two bounding rectangles and returns `true` if they intersect and `false` otherwise.

The second document keeps a definition of an array of coins. It adds two methods to the array object. The first one—`collisionsWith`—performs a collision check between any of the items in the array and the given object using functions defined in `collisions.js`. That's why we import the library at the start of the document. The method returns another array that contains objects intersecting the `player` argument. The other method, called `remove`, takes an object or an array of objects and removes them from `coins`.

The document is not a library; therefore, each document that imports `coins.js` would get its own separate copy of the object. Thus, we need to ensure that `coins.js` is imported only once in the game so that all references to the objects defined in that document relate to the same instance of the object in our program memory.

Our main document imports `coins.js`, which creates the array for storing coin objects and makes its auxiliary functions available. This allows the defined  `checkCollisions()`  function to retrieve the list of coins colliding with the player. For each coin that collides with the player, we execute a `hit()` method; as a last step, all colliding coins are removed from the array. Since coins are stationary; collision can only occur when the player character enters an area occupied by a coin. Therefore, it is enough to trigger collision detection when the position of the player character changes—we use the  `onXChanged`  and `onYChanged` 
handlers.

As hitting a coin results in removing it from the array, we lose a reference to the object. The `hit()` method has to initiate removal of the object from the scene. A minimalistic implementation of this function would be to just call the `destroy()` function on the object, but we do more—the removal can be made smoother by running a fade-out animation on the coin. As a last step, the animation can destroy the object.

The number of objects we track on the scene is really small, and we simplify the shape of each object to a rectangle. This lets us get away with checking collisions in JavaScript. For a larger amount of moving objects, custom shapes, and handling rotations, it is much better to have a collision system based on C++. The level of complexity of such a system depends on your needs.

# Have a go hero – Extending the game

You can polish your game development skills by implementing new game mechanics in our jumping elephant game. For example, you can introduce a concept of fatigue. The more the character jumps, the more tired they get and the slower they begin to move and have to rest to regain speed. To make the game more difficult, at times moving obstacles can be generated. When the character bumps into any of them, they get more and more tired. When the fatigue exceeds a certain level, the character dies and the game ends. The heartbeat diagram we previously created can be used to represent the character's level of fatigue—the more tired the character gets, the faster their heart beats.

There are many ways these changes can be implemented, and we want to give you a level of freedom, so we will not provide a step-by-step guide on how to implement a complete game. You already know a lot about Qt Quick, and this is a good opportunity to test your skills!

# Pop quiz

Q1\. Which of the following types cannot be used with the special on-property syntax?

1.  `Animation`
2.  `Transition`
3.  `Behavior`

Q2\. Which QML type allows you to configure a sprite animation with transitions between multiple states?

1.  `SpriteSequence`
2.  `Image`
3.  `AnimatedSprite`

Q3\. Which QML type is able to prevent any instant change of the property's value and perform a gradual change of value instead?

1.  `Timer`
2.  `Behavior`
3.  `PropertyAction`

# Summary

In this chapter, we showed you how to extend your Qt Quick skills to make your applications dynamic and attractive. We went through the process of recreating and improving a game created earlier in C++ to familiarize you with concepts such as collision detection, state-driven objects, and time-based game loops. You are now familiar with all the most important concepts required to make games using Qt Quick.

In the next chapter, we will turn our attention to techniques that will make your games even more visually appealing. We'll explore the built-in graphical effects Qt Quick provides. You will also learn to extend Qt Quick with custom painted items implemented in C++. This will give you the freedom to create any visual effects you have in mind.