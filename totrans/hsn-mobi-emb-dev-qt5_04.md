# Graphical and Special Effects

Qt Quick has been extended with animation and special effects through the use of particles. Particles and Qt Graphical Effects will make a **User Interface** (**UI**) come alive and stand out among the crowd.

The particle system in Qt Quick allows for a large number of images or other graphical objects to simulate highly energized and chaotic animation and effects. Simulating snow falling or explosions with fire is made easier by using a particle system. Dynamic properties of these elements animate these even more.

Using Qt Graphical Effects can help make UIs visually more appealing and make it easier for the user to differentiate between graphical components. Drop shadows, glows, and blurring make 2-dimensional objects seem more like 3-dimensional ones.

In this chapter, we will cover the following topics:

*   The universe of particles
*   Particle `painters`, `emitters`, and `affectors`
*   Graphical effects for Qt Quick

# The universe of particles

Finally! We have reached the fun part of the book where the magic happens. It's all fine and dandy using rectangles, text, and buttons, but particles add splash and zing, together with adding wisps of light to games. They can also be used to highlight and emphasize items of interest.

Particles are a type of animation that consist of numerous graphical elements, all moving in a fuzzy manner. There are four main QML components to use:

*   `ParticleSystem`: Maintains the particle animation timeline
*   `Emitters`: Radiates the particles in to the system

*   `Painters`: These components paint the particles. Here are the various components: 
    *   `ImageParticle`: A particle using an image
    *   `ItemParticle`: A particle using a QML item as delegate
    *   `CustomParticle`: A particle using a shader
*   `Affectors`: Alters the properties of a particle

To see how we manage all these items, let's take a look at the main particle manager, the `ParticleSystem`.

# ParticleSystem

The `ParticleSystem` component maintains the particle animation timeline. It is what bonds all the other elements together and acts as the center for operations. You can `pause`, `resume`, `restart`, `reset`, `start`, and `stop` the particle animation.

The `painters`, `emitters`, and `affectors` all interact with each other through the `ParticleSystem`.

Many `ParticleSystem` components can exist in your application, and each has an `Emitter` component. 

Let's dive a little more into the details about particle `painters`, `emitters`, and `affectors`.

# Particle painters, emitters, and affectors

Particles in Qt Quick are graphical elements such as images, QML items, and OpenGL shaders.

They can be made to move and flow in endless ways. 

Every particle is part of a `ParticleGroup`, which, by default, has an empty name. A `ParticleGroup` is a group of particle painters that allow for the timed animation transitions for the grouped particle painters.

The direction that particles are emitted is controlled by the `Direction` items which consist of these components: `AngleDirection`, `PointDirection`, and `TargetDirection`.

There are only a few types of particle painters you can use, but they cover just about everything you would want to use them for. Particle types available in Qt Quick are as follows:

*   `CustomParticle`: A particle based on OpenGL shader
*   `ImageParticle`: A particle based on an image file
*   `ItemParticle`: A particle based on a QML Item

`ImageParticle` is probably the most common and easiest to use and can be made from any image that QML has support for. If there are going to be numerous particles, it might be best to use small and optimized images.

Let's examine a simple `ItemParticle` animation. We will start by defining a `ParticleSystem` component with a child `ItemParticle` animation that is defined as a transparent `Rectangle` element with a small green border and a radius of 65, which means it appears as a green circle.

There are actually two type of emitters—the standard `Emitter` type, but also a special `TrailEmitter` type, which is derived from the `Emitter` item, but emits its particles from other particles instead of its bounding area.

An `Emitter` item is defined with the `SystemParticle` component bound to its `system` property. For the `velocity` property of the `Emitter` item, we use `AngleDirection`. `AngleDirection` directs the particles emitted at a certain angle. 

Angles in QML elements work in a clockwise fashion, starting at the right-hand side of an element. Here's the representation of it:

![](img/d929f1ba-f034-404c-b6bb-5486a10104d5.png)

For example, setting an `AngleDirection` of 90 would make the particles move downward.

Let's dig into a particle example:

The source code can be found on the Git repository under the `Chapter03-1` directory, in the `cp3` branch.

1.  We start by defining a `ParticleSystem`:

```cpp
    ParticleSystem {
        id: particelSystem
        anchors.fill: parent
```

2.  We add an `ItemParticle` and define the `delegate` to be a transparent `Rectangle`. We define a `radius`, which gives it rounded corners, and designate it to have a small green border:

```cpp
        ItemParticle {
            delegate: Rectangle {
                height: 30; width: 30
                id: particleSquare
                color: "transparent"
                radius: 65
                border.color: "green"
                border.width: 4
            }
        }
    }
```

3.  We define an `Emitter` and assign it to the `ParticleSystem`:

```cpp
    Emitter {
        id: particles
        system: particleSystem
        anchors { horizontalCenter: parent.horizontalCenter; }
        y: parent.height / 2
        width: 10
        height: 10
        lifeSpan: 5000
        velocityFromMovement: 60
        sizeVariation: 15
        emitRate: 50

        enabled: false
```

4.  We give the `Emitter` an `AngleDirection` `velocity` to add some variation in the direction:

```cpp
       velocity: AngleDirection {
            angle: 90
            magnitude: 150
            angleVariation: 25
            magnitudeVariation: 50
        }
    }
```

At this point, the app looks like this:

![](img/ec663039-75fa-4426-ab9c-d6cfeec363f4.png)

Let's see how the emitter would look when it is not centered:

1.  We bind the `Emitter` property, called `enabled`, to the value of `false` in order to stop the particles being constantly emitted.
2.  We then bind the `burst` property to animate a pulse of `25` particles with a mouse click like this:

```cpp
MouseArea {
    id: mousey
    anchors.fill: parent
    onClicked: {particles.burst(25) }
    hoverEnabled: true
 }
```

The properties of the `Emitter` component are attributes of the particles at the start of the animation. 

3.  We bind the `Emitter` property's `x` and `y` properties to the mouse position:

```cpp
        y: mousey.mouseY
        x: mousey.mouseX
```

4.  We can also remove the `horizontalCenter` anchor as well, unless you want the particle burst start to always be centered horizontally. 

This image shows the `Emitter` when it's not centered horizontally:

![](img/7b59dd07-93e2-4394-8f21-99a2eb099b24.png)

To influence the particles as they get beamed out into the scene, you need an `Affector`. Let's take a look at how to use an `Affector` in the next section.

# Affectors

An affector is an attribute that affects the way particles are streamed. There are a few types of `affectors` to choose from:

*   `Age`: Will terminate particles early
*   `Attractor`: Attracts particles toward a point
*   `Friction`: Slows a particle proportional to its velocity
*   `Gravity`: Applies acceleration at an angle
*   `Turbulence`: Applies noise in a fluid manner
*   `Wander`: Random particle trajectory

There are also `GroupGoal` and `SpriteGoal` `affectors`.

`Affectors` are optional but add their bling to the particles after they get emitted.

Let's examine one way to use these items.

1.  We add a `Turbulence` item as a child to the `ParticleSystem` component. The particles will now fly around randomly, like falling leaves being blown around in the wind:

```cpp
      Turbulence {
            anchors.fill: parent
            strength: 32
        }
```

2.  You can have more than one affector. Let's add some `Gravity`, as well! We will make this `Gravity` go upward. `Gravity` is kind of like giving an item some weight in a certain direction:

```cpp
        Gravity {
            anchors.fill: parent
            angle: 270
            magnitude: 4
        }
```

Here is what our example of `Turbulence` circles looks like:

![](img/26d5a457-86a1-42bd-af62-83a7f7d71995.png)

You can try the Qt for WebAssembly version here at [https://lpotter.github.io/particles/ch3-1.html](https://lpotter.github.io/particles/ch3-1.html).

We can also cause the particles to flow in a particular direction, or act in a particular shape.

# Shapes and directions

Shapes are a way that can be used to affect how Affectors act upon a certain area.

*   `EllipseShape`: Acts on ellipse shaped area
*   `LineShape`: Acts on a line
*   `MaskShape`: Acts on an image shaped area
*   `RectangleShape`: Acts on a rectangle area

Particles can have a velocity in a certain direction. There are three ways to direct particles:

*   `AngleDirection`
*   `PointDirection`
*   `TargetDirection`

From the point of emissions, `AngleDirection` has four properties—`angle`, `angleVariation`, `magnitude`, and `magnitudeVariation`. As I mentioned previously, angles are measured in degrees clockwise, starting to the right of the `Emitter` item. The `magnitude` property specifies the velocity of movement in pixels per second.

`PointDirection` will direct a `velocity` property to a certain point in the scene, or off the screen, if you like. It takes the `x`, `y`, `xVariation`, and `yVariation` properties.

With `TargetDirection`, you can instruct particles to be emitted toward a target item, or a targeted `x`, `y` point. `TargetDirection` has a new property called `proportionalMagnitude`, which makes the `magnitude` and `magnitudeVariation` properties operate as a multiple of the distance between starting point and target point per second.

Particles can be quite fun and add a sci-fi element to an application. It takes some experimentation to get them to perform as you see in your mind, as there is a great randomness to them.

Now, let's look at adding some other types of effects for graphics.

# Graphical effects for Qt Quick

When you usually think of effects such as blur, contrast, and glow, you might think of image editing software, as they tend to be applied those effects to images. Qt Graphical Effects can apply those same types of effects to QML UI components. 

If you use the Qt Quick Scene Graph software renderer, these will not be available or usable, as this does not support the effects.

Qt Graphical Effects come in a variety of types, each with various sub-effects:

*   `Blend`
*   `Color`:
    *   `BrightnessContrast`
    *   `` `ColorOverlay` ``
    *   `Colorize`
    *   `Desaturate`
    *   `GammaAdjust`
    *   `HueSaturation`
    *   `LevelAdjust`
*   `Gradients`:
    *   `ConicalGradient`
    *   `LinearGradient`
    *   `RadialGradient`
*   `Displace`
*   `DropShadows`:
    *   `DropShadow`
    *   `InnerShadow`
*   `Blurs`:
    *   `FastBlur`
    *   `GaussianBlur`
    *   `MaskedBlur`
    *   `RecursiveBlur`
*   `MotionBlurs`:
    *   `DirectionalBlur`
    *   `RadialBlur`
    *   `ZoomBlur`
*   `Glows`:
    *   `Glow`
    *   `RectangularGlow`
*   `Masks`:
    *   `OpacityMask`
    *   `ThresholdMask` 

Now, let's move on to how `DropShadow`, being one of the most useful effects, works.

# DropShadow

A `DropShadow` effect is something you can use to make things stand out and look more alive. It's usefulness is that it will give depth to otherwise flat objects.

We can add a `DropShadow` effect to a `Text` item from our last example. The `horizontalOffset` and `verticalOffset` properties characterize where the shadow will be perceived as being positioned upon the scene. The `radius` property describes the focus of the shadow, while the `samples` property determines the number of samples per pixel when blurring. 

Use the following code to add a `DropShadow `and apply it to the `Text` component:

```cpp
        Text {
            id: textLabel
            text: "Hands-On Mobile and Embedded"
            color: "purple"
            font.pointSize: 20
            anchors.top: parent.top
            anchors.horizontalCenter: parent.horizontalCenter
        }
        DropShadow {
            anchors.fill: textLabel
            horizontalOffset: 2
            verticalOffset: 2
            radius: 10
            samples: 25
            color: "white"
            source: textLabel
        }

```

The source code can be found on the Git repository under the `Chapter03-2` directory, in the `cp3` branch.

Here, you can see the letters now have a white shadow under them:

![](img/b3f7ea20-24e0-4d8a-9951-d641e54aab72.png)

It also has a `spread` property that controls the sharpness of the shadow. That is still a bit difficult to read, so let's try something else. How about a `Glow` effect?

# Glow

`Glow` is an effect that produces a diffused color around the object by using the following code:

```cpp
        Glow {
            anchors.fill: textLabel
            radius: 10
            samples: 25
            color: "lightblue"
            source: textLabel
        }
```

The effect is shown in the following screenshot. Notice the nice light blue glow:

![](img/53c15dc4-6b94-4bef-a5fa-8d82ff2f13b1.png)

That's more like it! We can even give the `Glow` effect its own shadow! Change the `DropShadow`, `anchors.fill`, and `source` properties to `glow`: 

```cpp
        DropShadow {
            anchors.fill: glow
            horizontalOffset: 5
            verticalOffset: 5
            radius: 10
            samples: 25
            color: "black"
            source: glow
        }
```

Let's make the `horizontalOffset` and `verticalOffset` properties a tad bigger too.

Our banner now looks like this:

![](img/14727e9a-b68b-402f-b9d6-db88475e7496.png)

`DropShadows` are great for making something stand out from the scene. Gradients are another effect to use.

# Gradient

`Gradients` can grab a user's attention, draw them into a UI, and connect to their emotions. Qt Graphical Effects have built-in support for three types of gradients—`Conical`, `Linear`, and `Radial`.

`RadialGradient`, or any QML gradient for that matter, is made up of a series of `GradientStop` items, which specify the color and where to start it in the gradient cycle, the number zero being at the beginning, and one being at the end point.

Here's the code for representing a `RadialGradient`:

```cpp
    Item {
        width: 250; height: 250
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.verticalCenter: parent.verticalCenter
        RadialGradient {
            anchors.fill: parent
            gradient: Gradient {
                GradientStop { position: 0.0; color: "red" }
                GradientStop { position: 0.3; color: "green" }
                GradientStop { position: 0.6; color: "purple" }
            }
        }
    }
```

The source code can be found on the Git repository under the `Chapter03-4` directory, in the `cp3` branch.

The following is the pictorial presentation of our `RadialGradient`:

![](img/b34e5629-1e88-4d45-9df8-403fb6b24f58.png)

This `RadialGradient` uses three `GradientStop` items to tell the gradient where a certain color should start from. The `position` property is a `qreal` value from `0.0` to `1.0`; although having a number larger than `1.0` will not give an error, it will simply not be drawn in the bounding item.

Using the same color-stop schemes as the `RadialGradient`, we see how the `LinearGradient` and `ConicalGradient` look.

The following is a representation of `LinearGradient`: 

![](img/35a4de21-6acd-4942-b0de-4cf07ba60432.png)

The following is a representation of `ConicalGradient`:

![](img/e953671a-516d-4c19-931c-501bca66e30a.png)

You can see the differences between each of these gradients.

# Blur

`Blur` effects can help de-emphasize or bring action to a static image. The fastest `Blur` effect would be the aptly named `FastBlur` effect, but the `GaussianBlur` effect is the highest quality, and, consequently, the slowest to render.

All the `Blur` effects have `radius`, `samples`, and `source` properties. `Radius` represents the distance of pixels that will affect the `Blur` effect, with a higher number increasing the `Blur` effect. The `samples` property represents the number of samples per pixel used when the effect is applied. A higher number means better quality, but a slower render time. `Source` is the source item that the `Blur` effect will be applied to.

`Displace` is a type of a `Blur` effect, but with more of a possible watermark-type effect. The `displacementSource` property is the item that is being interposed onto the source item. The `displacement` property is a `qreal` value between -1.0 and 1.0, with 0 meaning there is no displacement of pixels.

# Summary

Qt Quick offers graphical and special effects that are very easy to start using. Particles, in particular, are great for gaming apps. You now know how to use the `ParticleSystem` to emit an `ImageParticle` at a particular angle using `AngleDirection`. We examined how `Affectors` such as `Turbulence` will affect an `Emitter` by adding variation to the particle stream.

`Gradients`, `Glow`, and `DropShadows` are useful for bringing emphasis to certain items. The `Blur` effects are used to simulate movement action or to add your watermark to images.

In the next chapter, we delve into using something now ubiquitous on mobile phones—touch input. I will also touch upon (pun intended) using other forms of inputs, such as what to do when there is no hardware keyboard and your app is what gets booted into.