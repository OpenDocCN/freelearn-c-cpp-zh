# Advanced Visual Effects in Qt Quick

Sprite animations and smooth transitions are not always enough to make the game visually appealing. In this chapter, we will explore many ways to add some eye candy to your games. Qt Quick provides a decent amount of built-in visual effects that will come in handy. However, from time to time, you will want to do something that is not possible to do with standard components—something unique and specific to your game. In these cases, you don't need to limit your imagination. We will teach you to dive deep into the C++ API of Qt Quick to implement truly unique graphics effects.

The main topics covered in this chapter are these:

*   Auto-scaling user interfaces
*   Applying graphical effects to the existing items
*   Particle systems
*   OpenGL painting in Qt Quick
*   Using `QPainter` in Qt Quick

# Making the game more attractive

A game should not just be based upon an interesting idea, and it should not only work fluently on a range of devices and give entertainment to those people playing it. It should also look nice and behave nicely. Whether people are choosing from a number of similar implementations of the same game or want to spend money on another similarly priced and entertaining game, there is a good chance that they'll choose the game that looks the best—having a lot of animations, graphics, and flashy content. We already learned a number of techniques to make a game more pleasing to the eye, such as using animations or implementing parallax effect. Here, we will show you a number of other techniques that can make your Qt Quick applications more attractive.

# Auto-scaling user interfaces

The first extension you may implement is making your game auto-adjust to the device resolution it is running on. There are basically two ways to accomplish this. The first is to center the user interface in the window (or screen) and if it doesn't fit, enable scrolling. The other approach is to scale the interface to always fit the window (or screen). Which one to choose depends on a number of factors, the most important of which is whether your UI is good enough when upscaled. If the interface consists of text and non-image primitives (basically rectangles), or if it includes images but only vector ones or those with very high resolution, then it is probably fine to try and scale the user interface. Otherwise, if you use a lot of low-resolution bitmap images, you will have to choose one particular size for the UI (optionally allowing it to downscale, since the quality degradation should be less significant in this direction if you enable anti-aliasing).

Whether you choose to scale or to center and scroll, the basic approach is the same—you put your UI item in another item so that you have fine control over the UI geometry, regardless of what happens to the top-level window. Taking the centered approach is quite easy—just anchor the UI to the center of the parent. To enable scrolling, wrap the UI in the `Flickable` item and constrain its size if the size of the window is not big enough to fit the whole user interface:

```cpp
Window {
    //...
    Flickable {
        id: uiFlickable
        anchors.centerIn: parent
        contentWidth: ui.width
        contentHeight: ui.height

        width: parent.width >= contentWidth ?
               contentWidth : parent.width
        height: parent.height >= contentHeight ?
                contentHeight : parent.height

        UI {
            id: ui
        }
    }
}
```

You can put the following simple code into the `UI.qml` file to see how `Flickable` positions the UI item:

```cpp
import QtQuick 2.0
Rectangle {
    width: 300
    height: 300
    gradient: Gradient {
        GradientStop { position: 0.0; color: "lightsteelblue" }
        GradientStop { position: 1.0; color: "blue" }
    }
}
```

You should probably decorate the top-level item with a nice background if the UI item does not occupy the full area of its parent.

Scaling seems more complicated, but it is really easy with Qt Quick. Again, you have two choices—either stretch or scale. Stretching is as easy as executing the `anchors.fill: parent` command, which effectively forces the UI to recalculate the geometry of all its items, but it possibly allows us to use the space more efficiently. It is, in general, very time-consuming for the developer to provide expressions for calculating the geometry of each and every element in the user interface as the size of the view changes. This is usually not worth the effort. A simpler approach is to just scale the UI item to fit the window, which will implicitly scale the contained items. In such an event, their size can be calculated relative to the base size of the main view of the user interface. For this to work, you need to calculate the scale that is to be applied to the user interface to make it fill the whole space available. The item has a scale of 1 when its effective width equals its implicit width and its effective height equals its implicit height. If the window is larger, we want to scale up the item until it reaches the size of the window.

Therefore, the window's width divided by the item's implicit width will be the item's scale in the horizontal direction. This is shown in the following diagram:

![](img/84379d94-af36-48dd-b35d-be256d7cde63.png)

The same can be applied to the vertical direction, but if the UI has a different aspect ratio than the window, its horizontal and vertical scale factors will be different. For the UI to look nice, we have to take the lower of the two values—to only scale up as much as the direction with less space allows, leaving a gap in the other direction:

```cpp
Window {
    //...
    UI {
        id: ui
        anchors.centerIn: parent
        scale: Math.min(parent.width / width,
                        parent.height / height)
    }
}
```

Again, it may be a good idea to put some background on the window item to fill in the gaps.

What if you want to save some margin between the user interface and the window? You can, of course, take that into consideration when calculating the scale (`(window.width - 2 * margin) / width`, and so on) but there is an easier way—simply put an additional item inside the window, leaving an appropriate margin, and put the user interface item in that additional item and scale it up to the additional item's size:

```cpp
Window {
    //...
    Item {
 anchors {
 fill: parent
 margins: 10
 }
        UI {
            id: ui
            anchors.centerIn: parent
            scale: Math.min(parent.width / width,
                            parent.height / height)
        }
 }
}
```

When you scale elements a lot, you should consider enabling anti-aliasing for items that can lose quality when rendered in a size different than their native size (for example, images). This is done very easily in Qt Quick, as each `Item` instance has a property called `antialiasing` which, when enabled, will cause the rendering backend to try to reduce distortions caused by the aliasing effect. Remember that this comes at the cost of increased rendering complexity, so try to find a balance between quality and efficiency, especially on low-end hardware. You may provide an option to the user to globally enable or disable anti-aliasing for all game objects or to gradually adjust quality settings for different object types.

# Graphical effects

The basic two predefined items in Qt Quick are rectangle and image. You can use them in a variety of creative ways and make them more pleasant-looking by applying GLSL shaders. However, implementing a shader program from scratch is cumbersome and requires in-depth knowledge of the shader language. Luckily, a number of common effects are already implemented and ready to use in the form of the `QtGraphicalEffects` module.

To add a subtle black shadow to our canvas-based heartbeat element defined in the `HeartBeat.qml` file, use a code similar to the following that makes use of the `DropShadow` effect:

```cpp
import QtQuick 2.9
import QtQuick.Window 2.2
import QtGraphicalEffects 1.0

Window {
    //...
    HeartBeat {
        id: heartBeat
        anchors.centerIn: parent
        visible: false
    }
    DropShadow {
        source: heartBeat
        anchors.fill: heartBeat
        horizontalOffset: 3
        verticalOffset: 3
        radius: 8
        samples: 16
        color: "black"
    }
}
```

To apply a shadow effect, you need an existing item as the source of the effect. In our case, we are using an instance of the `HeartBeat` class centered in a top-level item. Then, the shadow effect is defined and its geometry follows that of its source using the `anchors.fill` element. Just as the `DropShadow` class renders the original item as well as the shadow, the original item can be hidden by setting its `visible` property to `false`:

![](img/86c9667d-7fc3-4b2a-8a28-d4cde532d888.png)

Most of the `DropShadow` class's properties are self-explanatory, but two properties—`radius` and `samples`—require some additional explanation. The shadow is drawn as a blurred monochromatic copy of the original item offset by a given position. The two mentioned properties control the amount of blur and its quality—the more samples used for blurring, the better the effect, but also the more demanding the computation that needs to be performed.

Speaking of blur, the plain blurring effect is also available in the graphics effects module through the `GaussianBlur` element type. To apply a blur instead of a shadow to the last example, simply replace the occurrence of the `DropShadow` class with the following code:

```cpp
GaussianBlur {
    source: heartBeat
    anchors.fill: heartBeat
    radius: 12
    samples: 20
    transparentBorder: true
}
```

This change will produce the following result:  

![](img/5747ab9d-eb99-4c4d-b9ce-2fb65a04b2de.png)

Here, you can see two earlier mentioned properties as well as a vaguely named `transparentBorder` one. Enabling this property fixes some artifacts on the edges of the blur and in general, you'll want to keep it that way.

# Have a go hero – The blur parallax scrolled game view

The `blur` property is a very nice effect that can be used in many situations. For example, you can try to implement a feature within our elephant game whereby when the user pauses the game (for example, by pressing the *P* key on the keyboard), the view gets blurred. Make the effect smooth by applying an animation to the effect's `radius` property.

Another interesting effect is `Glow`. It renders a colored and blurred copy of the source element. An example use case for games is highlighting some parts of the user interface—you can direct the user's attention to the element (for example, button or badge) by making the element flash periodically:

```cpp
Window {
    //...
    Badge {
        id: importantBadge
        anchors.centerIn: parent
    }
    Glow {
        source: importantBadge
        anchors.fill: source
        samples: 64
        color: "red"

        SequentialAnimation on radius {
            loops: Animation.Infinite
            running: true

            NumberAnimation { from: 0; to: 30; duration: 500 }
            PauseAnimation { duration: 100 }
            NumberAnimation { from: 30; to: 0; duration: 500 }
            PauseAnimation { duration: 1000 }
        }
    }
}
```

The complete module contains 20 different effects. We cannot describe each effect in detail here. Nevertheless, you can learn about it yourself. If you clone the module's source git repository (found under [https://code.qt.io/cgit/qt/qtgraphicaleffects.git/](https://code.qt.io/cgit/qt/qtgraphicaleffects.git/)), in the `tests/manual/testbed` subdirectory of the cloned repository, you will find a nice application for testing the existing effects. To run the tool, open the `testBed.qml` file with `qmlscene`:

![](img/664cba82-99f9-4434-a149-c126267ea868.png)

You can also access a complete list of effects and their short descriptions by searching for QtGraphicalEffects in the documentation index.

# Particle systems

A commonly used visual effect in games is generating a large number of small, usually short-lived, often fast-moving, fuzzy objects such as stars, sparks, fumes, dust, snow, splinters, falling leaves, or the like. Placing these as regular items within a scene would greatly degrade performance. Instead, a special engine is used, which keeps a registry of such objects and tracks (simulates) their logical attributes without having physical entities in the scene. Such objects, called **particles**, are rendered upon request in the scene using very efficient algorithms. This allows us to use a large number of particles without having a negative impact on the rest of the scene.

Qt Quick provides a particle system in the `QtQuick.Particles` import. The `ParticleSystem` element provides the core for the simulation, which uses the `Emitter` elements to spawn particles. They are then rendered according to definitions in a `ParticlePainter` element. Simulated entities can be manipulated using the `Affector` objects, which can modify the trajectory or life span of particles.

Let's start with a simple example. The following code snippet declares the simplest possible particle system:

```cpp
import QtQuick 2.0
import QtQuick.Window 2.2
import QtQuick.Particles 2.0

Window {
    visible: true
    width: 360
    height: 360
    title: qsTr("Particle system")

    ParticleSystem {
        id: particleSystem
        anchors.fill: parent

        Emitter { anchors.fill: parent }
        ImageParticle { source: "star.png" }
    }
}
```

The result can be observed in the following image:

![](img/12a485eb-2095-49a4-b081-4b863c3a28e4.png)

Let's analyze the code. After importing `QtQuick.Particles 2.0`, a `ParticleSystem`  item is instantiated that defines the domain of the particle system. We define two objects within that system. The first object is the `Emitter` and defines an area where particles will be spawned. The area is set to encompass the whole domain. The second object is an object of the `ImageParticle` type, which is a `ParticlePainter` subclass. It determines that particles should be rendered as instances of a given image. By default, the `Emitter` object spawns 10 particles per second, each of which lives for one second and then dies and is removed from the scene. In the code presented, the `Emitter` and `ImageParticle` objects are direct children of the `ParticleSystem` class; however, this doesn't have to be the case. The particle system can be explicitly specified by setting the `system` property.

# Tuning the emitter

You can control the amount of particles being emitted by setting the `emitRate` property of the emitter. Another property, called the `lifeSpan`, determines how many milliseconds it takes before a particle dies. To introduce some random behavior, you can use the `lifeSpanVariation` property to set a maximum amount of time (in milliseconds) the life span can be altered by the system (in both directions):

```cpp
Emitter {
    anchors.fill: parent
    emitRate: 350
    lifeSpan: 1500
    lifeSpanVariation: 400 // effective: 1100-1900 ms
} 
```

A possible result of this change is shown in the following picture:

![](img/c3b83796-4b1f-4c37-9b0b-974eb2622d2d.png)

Increasing the emission rate and life span of particles can lead to a situation in which a very large number of particles have to be managed (and possibly rendered). This can degrade performance; thus, an upper limit of particles that can concurrently be alive can be set through the `maximumEmitted` property.

Tweaking the life span of particles makes the system more diverse. To strengthen the effect, you can also manipulate the size of each particle through the `size` and   `sizeVariation` properties:

```cpp
Emitter {
    anchors.fill: parent
    emitRate: 50
    size: 12
    sizeVariation: 6
    endSize: 2
}
```

This will give you particles of different sizes:

![](img/d5afd80a-1d48-452c-843f-8c5d7a734b14.png)

The range of functionality presented thus far should be enough to create many nice-looking and useful particle systems. However, particles are emitted from the whole area of the emitter, which is a regular `QQuickItem` and thus is rectangular. This doesn't have to be the case, though. The `Emitter` element contains a `shape` property, which is a way to declare the area that is to be giving birth to particles. The `QtQuick.Particles` module defines three types of custom shape that can be used—`EllipseShape`, `LineShape`, and `MaskShape`. The first two are very simple, defining either an empty or filled ellipse inscribed in the item or a line crossing one of the two diagonals of the item. The `MaskShape` element is more interesting, as it makes it possible to use an image as a shape for the `Emitter` element:

```cpp
Emitter {
    anchors.fill: parent
    emitRate: 1600
    shape: MaskShape { source: "star.png" }
}
```

Particles can now only spawn within the specified area:

![](img/dfd9d5e8-772c-4de8-b31f-28d6b273d999.png)

# Rendering particles

So far, we have used a bare `ImageParticle` element to render particles. It is only one of the three `ParticlePainters` available, with the others being `ItemParticle` and   `CustomParticle`. However, before we move on to other renderers, let's focus on tweaking the `ImageParticle` element to obtain some interesting effects.

The `ImageParticle` element renders each logical particle as an image. The image can be manipulated separately for each particle by changing its color and rotation, deforming its shape, or using it as a sprite animation.

To influence the color of particles, you can use any of the large number of dedicated properties—`alpha`, `color`, `alphaVariation`, `colorVariation`, `redVariation`, `greenVariation`, and `blueVariation`. The first two properties define the base value for the respective attributes, and the remaining properties set the maximum deviation of a respective parameter from the base value. In the case of opacity, there is only one type of variation you can use, but when defining the color, you can either set different values for each of the red, green, and blue channels, or you can use the global `colorVariation`    property, which is similar to setting the same value for all three channels. Allowed values are any between the range of 0 (no deviation allowed) and 1.0 (100% in either direction).

Note that when a color is applied to an image, the respective components of the colors (red, green, blue, and alpha) are multiplied. Black color (0, 0, 0, 1) has all components set to 0 except for alpha, so applying a solid color to a black image will not have any effect. On the contrary, if your image contains white pixels (1, 1, 1, 1), they will be displayed in exactly the specified color. Transparent pixels will stay transparent because their alpha component will remain set to 0.

In our example, we can create particles with different colors using the following code:

```cpp
ImageParticle {
    source: "star_white.png"
    colorVariation: 1
}
```

The result should look like this:

![](img/b0b68155-65d2-4181-9ca6-7845053e7890.png)

The properties mentioned are stationary—the particle obeys the constant value during its whole life. The `ImageParticle` element also exposes two properties, letting you control the color of particles relative to their age. First of all, there is a property called  `entryEffect` that defines what happens with the particle at its birth and death. The default value is `Fade`, which makes particles fade in from 0 opacity at the start of their life and fades them back to 0 just before they die. You have already experienced this effect in all the earlier particle animations we demonstrated. Other values for the property are  `None` and `Scale`. The first one is obvious—there is no entry effect associated with particles. The second one scales particles from 0 at their birth and scales them back to 0 at the end of their life.

The other time-related property is `colorTable`. You can feed it with a URL of an image to be used as a one-dimensional texture determining the color of each particle over its life. At the beginning, the particle gets color-defined by the left edge of the image and then progresses right in a linear fashion. It is most common to set an image here containing a color gradient to achieve smooth transitions between colors.

The second parameter that can be altered is the rotation of a particle. Here, we can also either use properties that define constant values for rotation (`rotation` and  `rotationVariation`) specified in degrees or modify the rotation of particles in time with `rotationVelocity` and `rotationVelocityVariation`. The velocity defines the pace or rotation in degrees per second.

Particles can also be deformed. The `xVector` and `yVector` properties allow binding vectors, which define distortions in horizontal and vertical axes. We will describe how to set the vectors in the next section. Last but not least, using the `sprites` property, you can define a list of sprites that will be used to render particles. This works in a fashion similar to the `SpriteSequence` type described in the previous chapter.

# Making particles move

Apart from fading and rotating, the particle systems we have seen so far were very static. While this is useful for making star fields, it is not at all useful for explosions, sparks, or even falling snow. This is because particles are mostly about movement. Here, we will show you two aspects of making your particles fly.

The first aspect is modeling how the particles are born. By that, we mean the physical conditions of the object creating the particles. During an explosion, matter is pushed away from the epicenter with a very large force that causes air and small objects to rush outward at an extremely high speed. Fumes from a rocket engine are ejected with high velocities in the direction opposite to that of the propelled craft. A moving comet draws along a braid of dust and gases put into motion by the inertia.

All these conditions can be modeled by setting the velocity or acceleration of the particles. These two metrics are described by vectors determining the direction and amount (magnitude or length) of the given quantity. In Qt Quick, such vectors are represented by an element type called `Direction`, where the tail of the vector is attached to the object and the position of the head is calculated by the `Direction` instance. Since we have no means of setting attributes on particles because we have no objects representing them, those two attributes—`velocity` and `acceleration`—are applied to emitters spawning the particles. As you can have many emitters in a single particle system, you can set different velocities and accelerations for particles of different origins.

There are four types of direction elements representing different sources of information about the direction. First, there is `CumulativeDirection`, which acts as a container for other direction types and works like a sum of directions contained within.

Then, there is `PointDirection`, where you can specify the *x* and *y* coordinates of a point where the head of the vector should be attached. To avoid the unrealistic effect of all particles heading in the same direction, you can specify `xVariation` and `yVariation` to introduce allowed deviation from a given point:

![](img/bc1602b8-72ed-444c-8ae8-0838c464a5ac.png)

The third type is the most popular direction type—`AngleDirection`, which directly specifies the angle (in degrees clockwise from straight right) and magnitude (in pixels per second) of the vector. The angle can vary from the base by `angleVariation`, and similarly, `magnitudeVariation` can be used to introduce variation to the length of the vector:

![](img/132e4975-9919-4f09-9c4f-2d57ce0859dc.png)

The last type is similar to the previous one. The `TargetDirection` vector can be used to point the vector toward the center of a given Qt Quick item (set with the `targetItem`  property). The length of the vector is calculated by giving the `magnitude` and  `magnitudeVariation`, and both can be interpreted as pixels per second or multiples of distance between the source and target points (depending on the value of the `proportionalMagnitude` property):

![](img/166fabdf-9fc8-47fb-afd2-fb8d7a616f3c.png)

Let's get back to setting particle velocity. We can use the `AngleDirection` vector to specify that particles should be moving left, spreading at a maximum of 45 degrees:

```cpp
Emitter {
    anchors.centerIn: parent
    width: 50; height: 50
    emitRate: 50

    velocity: AngleDirection {
        angleVariation: 45
        angle: 180
        magnitude: 200
    }
}
```

This code will produce the effect shown on the following picture:

![](img/fbcd2b9f-b9bc-4333-a3ce-6b578537a00d.png)

Setting acceleration works the same way. You can even set both the initial velocity and the acceleration each particle should have. It is very easy to shoot the particles in the left direction and start pulling them down:

```cpp
Emitter {
    anchors.right: parent.right
    anchors.verticalCenter: parent.verticalCenter
    emitRate: 15
    lifeSpan: 5000

    velocity: AngleDirection {
        angle: 180
        magnitude: 200
    }
    acceleration: AngleDirection {
        angle: 90 // local left = global down
        magnitude: 100
    }
}
```

This code will produce particles moving along a single curve:

![](img/82cb84a0-7627-468f-aba5-b968d18e38a3.png)

The `Emitter` element has one more nice property that is useful in the context of moving particles. Setting the `velocityFromMovement` parameter to a value different from `0`  makes any movement of the `Emitter` element apply to the velocity of the particles. The direction of the additional vector matches the direction of the emitter's movement, and the magnitude is set to the speed of the emitter multiplied by the value set to `velocityFromMovement`. It is a great way to generate fumes ejected from a rocket engine:

```cpp
Item {
    Image {
        id: image
        source: "rocket.png"
    }
    Emitter {
        anchors.right: image.right
        anchors.verticalCenter: image.verticalCenter
        emitRate: 500
        lifeSpan: 3000
        lifeSpanVariation: 1000
        velocityFromMovement: -20

        velocity: AngleDirection {
            magnitude: 100
            angleVariation: 40
        }
    }
    NumberAnimation on x {
        ...
    }
}
```

This is how the result could look like:

![](img/70d657de-7e31-4a14-9904-c502f5db4f75.png)

The second way of addressing the behavior of particles is to influence their attributes after they are born—in any particular moment of their life. This can be done using affectors. These are items inheriting affector, which can modify some attributes of particles currently traveling though the area of the affector. One of the simplest affectors is `Age`. It can advance particles to a point in their lifetime where they only have `lifeLeft` milliseconds of their life left:

```cpp
Age { 
    once: true 
    lifeLeft: 500 
    shape: EllipseShape { fill: true }
    anchors.fill: parent 
}
```

Setting `once` to `true` makes each affector influence a given particle only once. Otherwise, each particle can have its attributes modified many times.

Another affector type is `Gravity`, which can accelerate particles in a given angle. `Friction` can slow particles down, and `Attractor` will affect the particle's position, velocity, or acceleration so that it starts traveling toward a given point. `Wander` is great for simulating snowflakes or butterflies flying in pseudo-random directions.

There are also other affector types available, but we will not go into their details here. We would like to warn you, however, against using affectors too often—they can severely degrade performance.

# Time for action – Vanishing coins spawning particles

It is now time to practice our freshly acquired skills. The task is to add a particle effect to the game we created in the previous chapter. When the player collects a coin, it will explode into a sprinkle of colorful stars.

Start by declaring a particle system as filling the game scene, along with the particle painter definition:

```cpp
ParticleSystem {
    id: coinParticles
    anchors.fill: parent // scene is the parent

    ImageParticle {
        source: "images/particle.png"
        colorVariation: 1
        rotationVariation: 180
        rotationVelocityVariation: 10
    }
} 
```

Next, modify the definition of `Coin` to include an emitter:

```cpp
Emitter {
    id: emitter
    system: coinParticles
    emitRate: 0
    lifeSpan: 1500
    lifeSpanVariation: 100
    velocity: AngleDirection {
        angleVariation: 180
        magnitude: 10
    }
    acceleration: AngleDirection {
        angle: 270
        magnitude: 30
    }
}
```

Finally, the `hit()` function has to be updated:

```cpp
function hit() {
    emitter.burst(50);
    hitAnim.start();
} 
```

Run the game and see what happens when Benjamin collects coins:

![](img/c281ab1c-c633-40a5-aa0f-bf4690a32b01.png)

# What just happened?

In this exercise, we defined a simple particle system that fills the whole scene. We defined a simple image painter for the particles where we allow particles to take on all the colors and start in all possible rotations. We used a star pixmap as our particle template.

Then, an `Emitter` object is attached to every coin. Its `emitRate` is set to `0`, which means it does not emit any particles on its own. We set a varying life span on particles and let them fly in all directions by setting their initial velocity with an angle variation of 180 degrees in both directions (giving a total of 360 degrees). By setting an acceleration, we give the particles a tendency to travel toward the top edge of the scene.

In the hit function, we call a `burst()` function on the emitter, which makes it give instant birth to a given number of particles.

# Custom OpenGL-based Qt Quick items

In [Chapter 12](4fdfe294-c35c-476d-9656-0aefd533e491.xhtml), *Customization in Qt Quick*, we learned to create new QML element types that can be used to provide dynamic data engines or some other type of non-visual objects. Now we will see how to provide new types of visual items to Qt Quick.

The first question you should ask yourself is whether you really need a new type of item. Maybe you can achieve the same goal with the already existing elements? Very often, you can use vector or bitmap images to add custom shapes to your applications, or you can use Canvas to quickly draw the graphics you need directly in QML.

If you decide that you do require custom items, you will be doing that by implementing subclasses of the `QQuickItem` C++ class, which is the base class for all items in Qt Quick. After creating the new type, you will always have to register it with QML using  `qmlRegisterType`.

# The scene graph

To provide very fast rendering of its scene, Qt Quick uses a mechanism called **scene graph**. The graph consists of a number of nodes of well-known types, each describing a primitive shape to be drawn. The framework makes use of knowledge of each of the primitives allowed and their parameters to find the most performance-wise optimal order in which items can be rendered. Rendering itself is done using OpenGL, and all the shapes are defined in terms of OpenGL calls.

Providing new items for Qt Quick boils down to delivering a set of nodes that define the shape using terminology the graph understands. This is done by subclassing  `QQuickItem` and implementing the pure virtual `updatePaintNode()` method, which is supposed to return a node that will tell the scene graph how to render the item. The node will most likely be describing a geometry (shape) with a material (color, texture) applied.

# Time for action – Creating a regular polygon item

Let's learn about the scene-graph by delivering an item class for rendering convex regular polygons. We will draw the polygon using the OpenGL drawing mode called "triangle fan". It draws a set of triangles that all have a common vertex. Subsequent triangles are defined by the shared vertex, the vertex from the previous triangle, and the next vertex specified. Take a look at the diagram to see how to draw a hexagon as a triangle fan using eight vertices as control points:

![](img/d1cd68c1-1eb0-43c8-98c4-cc8083a64281.png)

The same method applies for any regular polygon. The first vertex defined is always the shared vertex occupying the center of the shape. The remaining points are positioned on the circumference of a bounding circle of the shape at equal angular distances. The angle is easily calculated by dividing the full angle by the number of sides. For a hexagon, this yields 60 degrees.

Let's get down to business and the `QQuickItem` subclass. We will give it a very simple interface:

```cpp
class RegularPolygon : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(int vertices READ vertices WRITE setVertices 
               NOTIFY verticesChanged)
    Q_PROPERTY(QColor color READ color WRITE setColor NOTIFY colorChanged)
public:
    RegularPolygon();
    ~RegularPolygon();
    int vertices() const;
    void setVertices(int v);

    QColor color() const;
    void setColor(const QColor &c);

    QSGNode *updatePaintNode(QSGNode *, UpdatePaintNodeData *);

signals:
    void verticesChanged(int);
    void colorChanged(QColor);

private:
     int m_vertexCount;
     QColor m_color;
};
```

Our polygon is defined by the number of vertices and the fill color. We also get everything we inherited from `QQuickItem`, including the width and height of the item. Besides adding the obvious getters and setters for the properties, we override the virtual  `updatePaintNode()` method, which is responsible for building the scene-graph.

Before we deal with updating graph nodes, let's deal with the easy parts first. Implement the constructor as follows:

```cpp
RegularPolygon::RegularPolygon()
{
    setFlag(ItemHasContents, true);
    m_vertexCount = 6;
}
```

We make our polygon a hexagon by default. We also set a flag, `ItemHasContents`, which tells the scene-graph that the item is not fully transparent and should ask us how the item should be painted by calling `updatePaintNode()`. Existence of this flag allows Qt to avoid preparing the whole infrastructure if the item would not be painting anything anyway.

The setters are also quite easy to grasp:

```cpp
void RegularPolygon::setVertices(int v) {
    v = qMax(3, v);
    if(v == vertices()) return;
    m_vertexCount = v;
    emit verticesChanged(v);
    update();
}

void RegularPolygon::setColor(const QColor &c) {
    if(color() == c) return;
    m_color = c;
    emit colorChanged(c);
    update();
}
```

A polygon has to have at least three sides; thus, we enforce this as a minimum, sanitizing the input value with `qMax`. After we change any of the properties that could influence the look of the item, we call `update()` to let Qt Quick know that the item needs to be rerendered. Let's tackle `updatePaintNode()` now. We'll disassemble it into smaller pieces so that it is easier for you to understand how the function works:

```cpp
QSGNode *RegularPolygon::updatePaintNode(
    QSGNode *oldNode, QQuickItem::UpdatePaintNodeData *) { 
```

When the function is called, it may receive a node it returned during a previous call. Be aware that the graph is free to delete all the nodes when it feels like it, so you should never rely on the node being there even if you returned a valid node during the previous run of the function. Let's move on to the next part of the function:

```cpp
  QSGGeometryNode *node = nullptr;
  QSGGeometry *geometry = nullptr;
```

The node we will return is a geometry node that contains information about the geometry and the material of the shape being drawn. We will be filling those variables as we go through the method. Next, we check whether `oldNode` was provided:

```cpp
    if (!oldNode) {
        node = new QSGGeometryNode;
        geometry = new QSGGeometry(
            QSGGeometry::defaultAttributes_Point2D(), m_vertexCount + 2);
        geometry->setDrawingMode(GL_TRIANGLE_FAN);
        node->setGeometry(geometry);
        node->setFlag(QSGNode::OwnsGeometry);
```

As we already mentioned, the function is called with the previously returned node as the argument, but we should be prepared for the node not being there and we should create it. Thus, if that is the case, we create a new `QSGGeometryNode` and a new `QSGGeometry`  for it. The geometry constructor takes a so-called attribute set as its parameter, which defines a layout for data in the geometry.

Most common layouts have been predefined:

| **Attribute set** | **Usage** | **First attribute** | **Second attribute** |
| `Point2D` | Solid colored shape | `float x, y` | - |
| `ColoredPoint2D` | Per-vertex color | `float x, y` | `uchar red, green, blue, alpha` |
| `TexturedPoint2D` | Per-vertex texture coordinate | `float x, y` | `float tx, float ty` |

We will be defining the geometry in terms of 2D points without any additional information attached to each point; therefore, we pass `QSGGeometry::defaultAttributes_Point2D()`  to construct the layout we need. As you can see in the preceding table for that layout, each attribute consists of two floating point values denoting the *x* and *y* coordinates of a point.

The second argument of the `QSGGeometry` constructor informs us about the number of vertices we will be using. The constructor will allocate as much memory as is needed to store the required number of vertices using the given attribute layout. After the geometry container is ready, we pass its ownership to the geometry node so that when the geometry node is destroyed, the memory for the geometry is freed as well. At this moment, we also mark that we will be rendering in the `GL_TRIANGLE_FAN` mode. The process is repeated for the material:

```cpp
        QSGFlatColorMaterial *material = new QSGFlatColorMaterial;
        material->setColor(m_color);
        node->setMaterial(material);
        node->setFlag(QSGNode::OwnsMaterial);
```

We use `QSGFlatColorMaterial` as the whole shape will have one color that is set from `m_color`. Qt provides a number of predefined material types. For example, if we wanted to give each vertex a separate color, we would have used  `QSGVertexColorMaterial` along with the `ColoredPoint2D` attribute layout.

The next piece of code deals with a situation in which `oldNode` did contain a valid pointer to a node that was already initialized:

```cpp
    } else {
        node = static_cast<QSGGeometryNode *>(oldNode);
        geometry = node->geometry();
        geometry->allocate(m_vertexCount + 2);
    }
```

In this case, we only need to ensure that the geometry can hold as many vertices as we need in case the number of sides changed since the last time the function was executed. Next, we check the material:

```cpp
    QSGMaterial *material = node->material();
    QSGFlatColorMaterial *flatMaterial =
            static_cast<QSGFlatColorMaterial*>(material);
    if(flatMaterial->color() != m_color) {
        flatMaterial->setColor(m_color);
        node->markDirty(QSGNode::DirtyMaterial);
    }
```

If the color differs, we reset it and tell the geometry node that the material needs to be updated by marking the `DirtyMaterial` flag.

Finally, we can set vertex data:

```cpp
    QRectF bounds = boundingRect();
    QSGGeometry::Point2D *vertices = geometry->vertexDataAsPoint2D();

    // first vertex is the shared one (middle)
    QPointF center = bounds.center();

    vertices[0].set(center.x(), center.y());

    // vertices are distributed along circumference of a circle

    qreal angleStep = 360.0 / m_vertexCount;

    qreal radius = qMin(width(), height()) / 2;

    for (int i = 0; i < m_vertexCount; ++i) {
        qreal rads = angleStep * i * M_PI / 180;
        qreal x = center.x() + radius * std::cos(rads);
        qreal y = center.y() + radius * std::sin(rads);
        vertices[1 + i].set(x, y);
    }
    vertices[1 + m_vertexCount] = vertices[1];
```

First, we ask the geometry object to prepare a mapping for us from the allocated memory to a `QSGGeometry::Point2D` structure, which can be used to conveniently set data for each vertex. Then, actual calculations are performed using the equation for calculating points on a circle. The radius of the circle is taken as the smaller part of the width and height of the item so that the shape is centered in the item. As you can see in the diagram at the beginning of the exercise, the last point in the array has the same coordinates as the second point in the array to close the fan into a regular polygon.

At the very end, we mark the geometry as changed and return the node to the caller:

```cpp
  node->markDirty(QSGNode::DirtyGeometry);
  return node;
} 
```

# What just happened?

Rendering in Qt Quick can happen in a thread different that is than the main thread. Before calling the `updatePaintNode()` function, Qt performs synchronization between the GUI thread and the rendering thread to allow us safely access our item's data and other objects living in the main thread. The function executing the main thread is blocked while this function executes, so it is crucial that it executes as quickly as possible and doesn't do any unnecessary calculations as this directly influences performance. This is also the only place in your code where at the same time you can safely call functions from your item (such as reading properties) and interact with the scene-graph (creating and updating the nodes). Try not emitting any signals nor creating any objects from within this method as they will have affinity to the rendering thread rather than the GUI thread.

Having said that, you can now register your class with QML using `qmlRegisterType` and test it with the following QML document:

```cpp
Window {
    width: 600
    height: 600
    visible: true
    RegularPolygon {
        id: poly
        anchors {
            fill: parent
            bottomMargin: 20
        }
        vertices: 5
        color: "blue"
    }
}
```

This should give you a nice blue pentagon. If the shape looks aliased, you can enforce anti-aliasing by setting the surface format for the application:

```cpp
int main(int argc, char **argv) {
    QGuiApplication app(argc, argv);
    QSurfaceFormat format = QSurfaceFormat::defaultFormat();
 format.setSamples(16); // enable multisampling
 QSurfaceFormat::setDefaultFormat(format);

    qmlRegisterType<RegularPolygon>("RegularPolygon", 1, 0, "RegularPolygon");

    QQmlApplicationEngine engine;
    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    if (engine.rootObjects().isEmpty())
      return -1;
    return app.exec();
}
```

If the application produces a black screen after enabling anti-aliasing, try to lower the number of samples or disable it.

# Have a go hero – Creating a supporting border for RegularPolygon

What is returned by `updatePaintNode()` may not just be a single `QSGGeometryNode`  but also a larger tree of `QSGNode` items. Each node can have any number of child nodes. By returning a node that has two geometry nodes as children, you can draw two separate shapes in the item:

![](img/1d3adbdc-88a0-4a2d-9ebb-a2daa20dafe5.png)

As a challenge, extend `RegularPolygon` to draw not only the internal filled part of the polygon but also an edge that can be of a different color. You can draw the edge using the `GL_QUAD_STRIP` drawing mode. Coordinates of the points are easy to calculate—the points closer to the middle of the shape are the same points that form the shape itself. The remaining points also lie on a circumference of a circle that is slightly larger (by the width of the border). Therefore, you can use the same equations to calculate them.

The `GL_QUAD_STRIP` mode renders quadrilaterals with every two vertices specified after the first four, composing a connected quadrilateral. The following diagram should give you a clear idea of what we are after:

![](img/c5b793c6-e376-4ef5-9a02-fba871dc80d5.png)

# Using QPainter interface in Qt Quick

Implementing items in OpenGL is quite difficult—you need to come up with an algorithm of using OpenGL primitives to draw the shape you want, and then you also need to be skilled enough with OpenGL to build a proper scene graph node tree for your item. However, there is another way—you can create items by painting them with `QPainter`. This comes at a cost of performance as behind the scenes, the painter draws on an indirect surface (a frame buffer object or an image) that is then converted to OpenGL texture and rendered on a quad by the scene-graph. Even considering that performance hit, it is often much simpler to draw the item using a rich and convenient drawing API than to spend hours doing the equivalent in OpenGL or using Canvas.

To use that approach, we will not be subclassing `QQuickItem` directly but `QQuickPaintedItem`, which gives us the infrastructure needed to use the painter for drawing items.

Basically, all we have to do, then, is implement the pure virtual `paint()` method that renders the item using the received painter. Let's see this put into practice and combine it with the skills we gained earlier.

# Time for action – Creating an item for drawing outlined text

The goal of the current exercise is to be able to make the following QML code work:

```cpp
import QtQuick 2.9
import QtQuick.Window 2.3
import OutlineTextItem 1.0

Window {
    visible: true
    width: 800
    height: 400
    title: qsTr("Hello World")

    Rectangle {
        anchors.fill: parent
        OutlineTextItem {
            anchors.centerIn: parent
            text: "This is outlined text"
            fontFamily: "Arial"
            fontPixelSize: 64
            color: "#33ff0000"
            antialiasing: true
            border {
                color: "blue"
                width: 2
                style: Qt.DotLine
            }
        }
    }
}
```

Then, it produces the following result:

![](img/a4b0cb80-bd85-42f5-b69b-c1df9ea24fa1.png)

Start with an empty Qt Quick application project. Create a new C++ class and call it `OutlineTextItemBorder`. Place the following code into the class definition:

```cpp
class OutlineTextItemBorder : public QObject {
    Q_OBJECT
    Q_PROPERTY(int width MEMBER m_width NOTIFY widthChanged)
    Q_PROPERTY(QColor color MEMBER m_color NOTIFY colorChanged)
    Q_PROPERTY(Qt::PenStyle style MEMBER m_style NOTIFY styleChanged)
public:
    OutlineTextItemBorder(QObject *parent = 0);

    int width() const;
    QColor color() const;
    Qt::PenStyle style() const;
    QPen pen() const;
signals:
    void widthChanged(int);
    void colorChanged(QColor);
    void styleChanged(int);
private:
    int m_width;
    QColor m_color;
    Qt::PenStyle m_style;
};
```

This is a simple `QObject`-based class holding a number of properties. You can see that  `Q_PROPERTY` macros don't have the `READ` and `WRITE` keywords we've been using thus far. This is because we are taking a shortcut right now, and we let **moc** produce code that will operate on the property by directly accessing the given class member. Normally, we would recommend against such an approach as without getters; the only way to access the properties is through the generic `property()` and `setProperty()` calls. However, in this case, we will not be exposing this class to the public in C++ so we won't need the setters, and we implement the getters ourselves, anyway. The nice thing about the `MEMBER`  keyword is that if we also provide the `NOTIFY` signal, the generated code will emit that signal when the value of the property changes, which will make property bindings in QML work as expected. We also need to implement the method that returns the actual pen based on values of the properties:

```cpp
QPen OutlineTextItemBorder::pen() const {
    QPen p;
    p.setColor(m_color);
    p.setWidth(m_width);
    p.setStyle(m_style);
    return p;
}
```

The class will provide a grouped property for our main item class. Create a class called `OutlineTextItem` and derive it from `QQuickPaintedItem`, as follows:

```cpp
class OutlineTextItem : public QQuickPaintedItem
{
    Q_OBJECT
    Q_PROPERTY(QString text MEMBER m_text
                            NOTIFY textChanged)
    Q_PROPERTY(QColor color MEMBER m_color
                            NOTIFY colorChanged)
    Q_PROPERTY(OutlineTextItemBorder* border READ border
                            NOTIFY borderChanged)
    Q_PROPERTY(QString fontFamily MEMBER m_fontFamily
                            NOTIFY fontFamilyChanged)
    Q_PROPERTY(int fontPixelSize MEMBER m_fontPixelSize
                            NOTIFY fontPixelSizeChanged)
public:
    OutlineTextItem(QQuickItem *parent = 0);
    void paint(QPainter *painter);
    OutlineTextItemBorder* border() const;
    QPainterPath shape(const QPainterPath &path) const;
private slots:
    void updateItem();
signals:
    void textChanged(QString);
    void colorChanged(QColor);
    void borderChanged();
    void fontFamilyChanged(QString);
    void fontPixelSizeChanged(int);
private:
    OutlineTextItemBorder* m_border;
    QPainterPath m_path;
    QRectF m_boundingRect;
    QString m_text;
    QColor m_color;
    QString m_fontFamily;
    int m_fontPixelSize;
};
```

The interface defines properties for the text to be drawn, in addition to its color, font, and the grouped property for the outline data. Again, we use `MEMBER` to avoid having to manually implement getters and setters. Unfortunately, this makes our constructor code more complicated, as we still need a way to run some code when any of the properties are modified. Implement the constructor using the following code:

```cpp
OutlineTextItem::OutlineTextItem(QQuickItem *parent) : 
    QQuickPaintedItem(parent) 
{
    m_border = new OutlineTextItemBorder(this);
    connect(this, &OutlineTextItem::textChanged,
            this, &OutlineTextItem::updateItem);
    connect(this, &OutlineTextItem::colorChanged,
            this, &OutlineTextItem::updateItem);
    connect(this, &OutlineTextItem::fontFamilyChanged,
            this, &OutlineTextItem::updateItem);
    connect(this, &OutlineTextItem::fontPixelSizeChanged,
            this, &OutlineTextItem::updateItem);
    connect(m_border, &OutlineTextItemBorder::widthChanged,
            this, &OutlineTextItem::updateItem);
    connect(m_border, &OutlineTextItemBorder::colorChanged,
            this, &OutlineTextItem::updateItem);
    connect(m_border, &OutlineTextItemBorder::styleChanged,
            this, &OutlineTextItem::updateItem);
    updateItem();
}
```

We basically connect all the property change signals from both the object and its grouped property object to the same slot that will update the data for the item if any of its components are modified. We also call the same slot directly to prepare the initial state of the item. The slot can be implemented like this:

```cpp
void OutlineTextItem::updateItem() {
    QFont font(m_fontFamily, m_fontPixelSize);
    m_path = QPainterPath();
    m_path.addText(0, 0, font, m_text);
    m_boundingRect = borderShape(m_path).controlPointRect();
    setImplicitWidth(m_boundingRect.width());
    setImplicitHeight(m_boundingRect.height());
    update();
}
```

At the beginning, the function resets a painter path object that serves as a backend for drawing outlined text and initializes it with the text drawn using the font set. Then, the slot calculates the bounding rect for the path using the `borderShape()` function that we will shortly see. We use `controlPointRect()` to calculate the bounding rectangle as it is much faster than `boundingRect()` and returns an area greater than or equal to the one  `boundingRect()`, which is OK for us. Finally, it sets the calculated size as the size hint for the item and asks the item to repaint itself with the `update()` call. Implement the `borderShape()` function using the following code:

```cpp
QPainterPath OutlineTextItem::borderShape(const QPainterPath &path) const
{
    QPainterPathStroker pathStroker;
    pathStroker.setWidth(m_border->width());
    QPainterPath p = pathStroker.createStroke(path);
    p.addPath(path);
    return p;
}
```

The `borderShape()` function returns a new painter path that includes both the original path and its outline created with the `QPainterPathStroker` object. This is so that the width of the stroke is correctly taken into account when calculating the bounding rectangle.

What remains is to implement the `paint()` routine itself:

```cpp
void OutlineTextItem::paint(QPainter *painter) {
    if(m_text.isEmpty()) return;
    painter->setPen(m_border->pen());
    painter->setBrush(m_color);
    painter->setRenderHint(QPainter::Antialiasing, true);
    painter->translate(-m_boundingRect.topLeft());
    painter->drawPath(m_path);
}
```

The code is really simple—we bail out early if there is nothing to draw. Otherwise, we set up the painter using the pen and color obtained from the item's properties. We enable anti-aliasing and calibrate the painter coordinates with that of the bounding rectangle of the item. Finally, we draw the path on the painter.

# What just happened?

During this exercise, we made use of the powerful API of Qt's raster graphics engine to complement an existing set of Qt Quick items with a simple functionality. This is otherwise very hard to achieve using predefined Qt Quick elements and even harder to implement using OpenGL. We agreed to take a small performance hit in exchange for having to write just about a hundred lines of code to have a fully working solution. Remember to register the class with QML if you want to use it in your code:

```cpp
qmlRegisterUncreatableType<OutlineTextItemBorder>(
     "OutlineTextItem", 1, 0, "OutlineTextItemBorder", "");
qmlRegisterType<OutlineTextItem>(
     "OutlineTextItem", 1, 0, "OutlineTextItem");
```

# Pop quiz

Q1\. Which QML type can be used to enable scrolling of a large item inside a smaller viewport?

1.  `Rectangle`
2.  `Flickable`
3.  `Window`

Q2\. What is the purpose of the `Affector` QML type?

1.  `Affector` allows you to change properties of QML items during an animation
2.  `Affector` influences properties of particles spawned by a particle system
3.  `Affector` allows you to control initial properties of particles spawned by a particle system

Q3\. What happens when you use `QPainter` to draw on a Qt Quick item?

1.  Every call to the `QPainter` API is translated to an equivalent OpenGL call
2.  `QPainter` paints on an invisible buffer that is then loaded as an OpenGL texture
3.  The item painted by `QPainter` is displayed without hardware acceleration

# Summary

You are now familiar with Qt Quick's capabilities that allow you to add astonishing graphical effects to your games. You can configure particle systems and implement OpenGL painting in the Qt Quick's scene graph. You are also able to utilize the skills acquired in the first parts of the book to implement painted Qt Quick items.

Of course, Qt Quick is much richer than all this, but we had to stop somewhere. The set of skills we have hopefully passed on to you should be enough to develop many great games. However, many of the elements have more properties than we have described here. Whenever you want to extend your skills, you can check the reference manual to see whether the element type has more interesting attributes. Qt Quick is still in active development, so it's a good idea to go through the changelogs of the recent Qt versions to see the new features that could not be covered in this book.

In the next chapter, we'll turn our attention to the Qt 3D module, which is a relatively recent addition to the Qt framework. Qt 3D provides a rich QML API that will allow us to use many of the skills we learned while working with Qt Quick. However, instead of user interface and 2D graphics, you will now create games that display hardware accelerated 3D graphics. When you learn to use Qt 3D, you will be able to take your games to a completely new level!