# Chapter 7. Using 2D Graphics

In this chapter, we will learn how to work and draw with 2D graphics and built-in Cinder tools.

The recipes in this chapter will cover the following:

*   Drawing 2D geometric primitives
*   Drawing arbitrary shapes with the mouse
*   Implementing a scribbler algorithm
*   Implementing 2D metaballs
*   Animating text around curves
*   Adding a blur effect
*   Implementing a force-directed graph

# Drawing 2D geometric primitives

In this recipe, we will learn how to draw the following 2D geometric shapes, as filled and stroked shapes:

*   Circle
*   Ellipse
*   Line
*   Rectangle

## Getting ready

Include the necessary header to draw in OpenGL using Cinder commands.

Add the following line of code at the top of your source file:

[PRE0]

## How to do it…

We will create several geometric primitives using Cinder's methods for drawing in 2D. Perform the following steps to do so:

1.  Let's begin by declaring member variables to keep information about the shapes we will be drawing.

    Create two `ci::Vec2f` objects to store the beginning and end of a line, a `ci::Rectf` object to draw a rectangle, a `ci::Vec2f` object to define the center of the circle, and a `float` object to define its radius. Finally, we will create `aci::Vec2f` to define the ellipse's radius and two `float` objects to define its width and height.

    Let's also declare two `ci::Color` objects to define the stroke and fill colors.

    [PRE1]

2.  In the `setup` method, let's initialize the preceding members:

    [PRE2]

3.  In the `draw` method, let's start by drawing filled shapes.

    Let's clear the background and set `mFillColor` to be the drawing color.

    [PRE3]

4.  Draw the filled shapes by calling the `ci::gl::drawSolidRect`, `ci::gl::drawSolidCircle`, and `ci::gl::drawSolidEllipse` methods.

    Add the following code snippet inside the `draw` method:

    [PRE4]

5.  To draw our shapes as stroked graphics, let's first set `mStrokeColor` as the drawing color.

    [PRE5]

6.  Let's draw our shapes again, this time using only strokes by calling the `ci::gl::drawLine`, `ci::gl::drawStrokeRect`, `ci::gl::drawStrokeCircle`, and `ci::gl::drawStrokedEllipse` methods.

    Add the following code snippet inside the `draw` method:

    [PRE6]

    This results in the following:

    ![How to do it…](img/8703OS_07_01.jpg)

## How it works…

Cinder's drawing methods use OpenGL calls internally to provide fast and easy drawing routines.

The `ci::gl::color` method sets the drawing color so that all shapes will be drawn with that color until another is set by calling `ci::gl::color` again.

## There's more…

You can also set the stroke width by calling the `glLineWidth` method and passing a `float` value as a parameter.

For example, to set the stroke to be 5 pixels wide you should write the following:

[PRE7]

# Drawing arbitrary shapes with the mouse

In this recipe, we will learn how to draw arbitrary shapes using the mouse.

We will begin a new contour every time the user presses the mouse button, and draw when the user drags the mouse.

The shape will be drawn using fill and stroke.

## Getting ready

Include the necessary files to draw and create a `ci::Shape2d` object.

Add the following code snippet at the top of your source file:

[PRE8]

## How to do it…

We will create a `ci::Shape2d` object and create vertices using mouse coordinates. Perform the following steps to do so:

1.  Declare a `ci::Shape2d` object to define our shape and two `ci::Color` objects to define the fill and stroke colors.

    [PRE9]

2.  Initialize the colors in the `setup` method.

    We'll be using black for stroke and yellow for fill.

    [PRE10]

3.  Since the drawing will be made with the mouse, it is necessary to use the `mouseDown` and `mouseDrag` events.

    Declare the necessary callback methods.

    [PRE11]

4.  In the implementation of `mouseDown` we will create a new contour by calling the `moveTo` method.

    The following code snippet shows what the method should look like:

    [PRE12]

5.  In the `mouseDrag` method we will add a line to our shape by calling the `lineTo` method.

    Its implementation should look like the following code snippet:

    [PRE13]

6.  In the `draw` method, we will first need to clear the background, then set `mFillColor` as the drawing color, and draw `mShape`.

    [PRE14]

7.  All there is left to do is to set `mStrokeColor` as the drawing color and draw `mShape` as a stroked shape.

    [PRE15]

8.  Build and run the application. Press the mouse button to begin drawing a new contour, and drag to draw.![How to do it…](img/8703OS_07_02.jpg)

## How it works…

`ci:Shape2d` is a class that defines an arbitrary shape in two dimensions allowing multiple contours.

The `ci::Shape2d::moveTo` method creates a new contour starting at the coordinate passed as a parameter. Then, the `ci::Shape2d::lineTo` method creates a straight line from the last position to the coordinate which is passed as a parameter.

The shape is internally tessellated into triangles when drawing a solid graphic.

## There's more…

It is also possible to add curves when constructing a shape using `ci::Shape2d`.

| Method | Explanation |
| --- | --- |
| `quadTo (constVec2f& p1, constVec2f& p2)` | Adds a quadratic curve from the last position to `p2`, using `p1` as a control point |
| `curveTo (constVec2f& p1, constVec2f& p2, constVec2f& p3)` | Adds a curve from the last position to `p3`, using `p1` and `p2` as control points |
| `arcTo (constVec2f& p, constVec2f& t, float radius)` | Adds an arc from the last position to `p1` using `t` as the tangent point and radius as the arc's radius |

# Implementing a scribbler algorithm

In this recipe, we are going to implement a scribbler algorithm, which is very simple to implement using Cinder but gives an interesting effect while drawing. You can read more about the concept of connecting neighbor points at [http://www.zefrank.com/scribbler/about.html](http://www.zefrank.com/scribbler/about.html). You can find an example of scribbler at [http://www.zefrank.com/scribbler/](http://www.zefrank.com/scribbler/) or [http://mrdoob.com/projects/harmony/](http://mrdoob.com/projects/harmony/).

## How to do it…

We will implement an application illustrating scribbler. Perform the following steps to do so:

1.  Include the necessary headers:

    [PRE16]

2.  Add properties to your main application class:

    [PRE17]

3.  Implement the `setup` method, as follows:

    [PRE18]

4.  Since the drawing will be made with the mouse, it is necessary to use the `mouseDown` and `mouseUp` events. Implement these methods, as follows:

    [PRE19]

5.  Finally, the implementation of drawing methods looks like the following code snippet:

    [PRE20]

## How it works…

While the left mouse button is down, we are adding a new point to our container and drawing lines connecting it with other points near it. The distance between the newly-added point and the points in its neighborhood we are looking for to draw a connection line has to be less than the value of the `mMaxDist` property. Please notice that we are clearing the drawing area only once, at the program startup at the end of the `setup` method, so we don't have to redraw all the connections to each frame, which would be very slow.

![How it works…](img/8703OS_07_03.jpg)

# Implementing 2D metaballs

In this recipe, we will learn how we can implement organic looking objects called metaballs.

## Getting ready

In this recipe, we are going to use the code base from the *Applying repulsion and attraction forces* recipe in [Chapter 5](ch05.html "Chapter 5. Building Particle Systems"), *Building Particle Systems*.

## How to do it…

We will implement the metaballs' rendering using a shader program. Perform the following steps to do so:

1.  Create a file inside the `assets` folder with a name, `passThru_vert.glsl`, and put the following code snippet inside it:

    [PRE21]

2.  Create a file inside the `assets` folder with a name, `mb_frag.glsl`, and put the following code snippet inside it:

    [PRE22]

3.  Add the necessary header files.

    [PRE23]

4.  Add a property to your application's main class, which is the `GlslProg` object for our GLSL shader program.

    [PRE24]

5.  In the `setup` method, change the values of `repulsionFactor` and `numParticle`.

    [PRE25]

6.  At the end of the `setup` method, load our GLSL shader program, as follows:

    [PRE26]

7.  The last major change is in the `draw` method, which looks like the following code snippet:

    [PRE27]

## How it works…

The most important part of this recipe is the fragment shader program mentioned in step 2\. The shader generates texture with rendered metaballs based on the positions and radius passed to the shader from our particle system. In step 7, you can find out how to pass information to the shader program. We are using `setMatricesWindow` and `setViewport` to set OpenGL for drawing.

![How it works…](img/8703OS_07_04.jpg)

## See also

*   **A Wikipedia article on metaballs**: [http://en.wikipedia.org/wiki/Metaballs](http://en.wikipedia.org/wiki/Metaballs)

# Animating text around curves

In this recipe, we will learn how we can animate text around a user-defined curve.

We will create the `Letter` and `Word` classes to manage the animation, a `ci::Path2d` object to define the curve, and a `ci::Timer` object to define the duration of the animation.

## Getting ready

Create and add the following files to your project:

*   `Word.h`
*   `Word.cpp`
*   `Letter.h`
*   `Letter.cpp`

## How to do it…

We will create a word and animate its letters along a `ci::Path2d` object. Perform the following steps to do so:

1.  In the `Letter.h` file, include the necessary to use the `text`, `ci::Vec2f`, and `ci::gl::Texture` files.

    Also add the `#pragma once` macro

    [PRE28]

2.  Declare the `Letter` class with the following members and methods:

    [PRE29]

3.  Move to the `Letter.cpp` file to implement the class.

    In the constructor, create a `ci::TextBox` object, set its parameters, and render it to texture. Also, set the width as the texture's width plus a padding value of 10:

    [PRE30]

4.  In the `draw` method, we will draw the texture and use OpenGL transformations to translate the texture to its position, and rotate according to the rotation:

    [PRE31]

5.  In the `setPos` method implementation, we will update the position and calculate its rotation so that the letter is perpendicular to its movement. We do this by calculating the arc tangent of its velocity:

    [PRE32]

6.  The `Letter` class is ready! Now move to the `Word.h` file, add the `#pragma once` macro, and include the `Letter.h` file:

    [PRE33]

7.  Declare the `Word` class with the following members and methods:

    [PRE34]

8.  Move to the `Word.cpp` file and include the `Word.h` file:

    [PRE35]

9.  In the constructor, we will iterate over each character of `text` and add a new `Letter` object.We will also calculate the total length of the text by calculating the sum of widths of all the letters:

    [PRE36]

    In the destructor, we will delete all the `Letter` objects to clean up memory used by the class:

    [PRE37]

10.  In the `update` method, we will pass a reference to the `ci::Path2d` object, the total length of the path, and the progress of the animation as a normalized value from 0.0 to 1.0.

    We will calculate the position of each individual letter along the curve taking into account the length of `Word` and the current progress:

    [PRE38]

11.  In the `draw` method, we will iterate over all letters and call the `draw` method of each letter:

    [PRE39]

12.  With the `Word` and `Letter` classes ready, it's time to move to our application's class source file. Start by including the necessary source files and adding the helpful `using` statements:

    [PRE40]

13.  Declare the following members:

    [PRE41]

14.  In the `setup` method, we will start by creating `std::string` and `ci::Font` and use them to initialize `mWord`. We will also initialize `mSeconds` with the seconds we want our animation to last for:

    [PRE42]

15.  We now need to create the curve by creating the keypoints and connecting them by calling `curveTo`:

    [PRE43]

16.  Let's calculate the length of the path by summing the distance between each point and the one next to it. Add the following code snippet inside the `setup` method:

    [PRE44]

17.  We need to check if `mTimer` is running and calculate the progress by calculating the ratio between the elapsed seconds and `mSeconds`. Add the following code snippet inside the `update` method:

    [PRE45]

18.  In the `draw` method, we will need to clear the background, enable alpha blending, draw `mWord`, and draw the path:

    [PRE46]

19.  Finally, we need to start the timer whenever the user presses any key.

    Declare the `keyUp` event handler:

    [PRE47]

20.  And the following is the implementation of the the `keyUp` event handler:

    [PRE48]

21.  Build and run the application. Press any key to begin the animation.![How to do it…](img/8703OS_07_05.jpg)

# Adding a blur effect

In this recipe, we will learn how we can apply a blur effect while drawing a texture.

## Getting ready

In this recipe, we are going to use a Gaussian blur shader provided by Geeks3D at [http://www.geeks3d.com/20100909/shader-library-gaussian-blur-post-processing-filter-in-glsl/](http://www.geeks3d.com/20100909/shader-library-gaussian-blur-post-processing-filter-in-glsl/).

## How to do it…

We will implement a sample Cinder application to illustrate the mechanism. Perform the following steps:

1.  Create a file inside the `assets` folder with the name `passThru_vert.glsl` and put the following code snippet inside it:

    [PRE49]

2.  Create a file inside the `assets` folder with the name `gaussian_v_frag.glsland` and put the following code snippet inside it:

    [PRE50]

    Create a file inside the `assets` folder with the name `gaussian_h_frag.glsl` and put the following code snippet inside it:

    [PRE51]

3.  Add the necessary headers:

    [PRE52]

4.  Add the properties to your application's main class:

    [PRE53]

5.  Implement the `setup` method, as follows:

    [PRE54]

6.  At the beginning of the `draw` method calculate the blur intensity:

    [PRE55]

7.  In the `draw` function render an image to `mFboBlur1` with a first step shader applied:

    [PRE56]

8.  In the `draw` function render a texture from `mFboBlur1` with a second step shader applied:

    [PRE57]

9.  Set `mImageBlur` to the result texture from `mFboBlur2`:

    [PRE58]

10.  At the end of the `draw` method draw a texture with the result and GUI:

    [PRE59]

## How it works…

Since a Gaussian blur shader needs to be applied twice—for the vertical and horizontal processing—we have to use **frame buffer object** (**FBO** ), a mechanism of drawing to the texture in the memory of graphic card. In step 8, we are drawing the original image from the `mImage` object and applying shader program stored in the `gaussian_v_frag.glsl` file loaded into `mGaussianVShaderobject`. At this point, everything is drawn into `mFboBlur1`. The next step is to use a texture from `mFboBlur2` and apply a shader to the second pass which you can find in step 9\. The final processed texture is stored in `mImageBlur` in step 10\. In step 7 we are calculating blur intensity.

![How it works…](img/8703OS_07_06.jpg)

# Implementing a force-directed graph

A force-directed graph is a way of drawing an aesthetic graph using simple physics such as repealing and springs. We are going to make our graph interactive so that users can drag nodes around and see how graph reorganizes itself.

## Getting ready

In this recipe we are going to use the code base from the *Creating a particle system in 2D* recipe in [Chapter 5](ch05.html "Chapter 5. Building Particle Systems"), *Building Particle Systems*. To get some details of how to draw nodes and connections between them, please refer to the *Connecting particles* recipe in [Chapter 6](ch06.html "Chapter 6. Rendering and Texturing Particle Systems"), *Rendering and Texturing Particle Systems*.

## How to do it…

We will create an interactive force-directed graph. Perform the following steps to do so:

1.  Add properties to your main application class.

    [PRE60]

2.  In the `setup` method set default values and create a graph.

    [PRE61]

3.  Implement interaction with the mouse.

    [PRE62]

4.  Inside the `update` method, calculate all forces affecting particles.

    [PRE63]

5.  In the `draw` method implement drawing particles and links between them.

    [PRE64]

6.  Inside the `Particle.cpp` source file, drawing of each particle should be implemented, as follows:

    [PRE65]

## How it works…

In step 2, in the `setup` method, we are creating our particles for each level of the graph and adding links between them. In the `update` method in step 4, we are calculating forces affecting all particles, which is repelling each particle from each other, and forces coming from the springs connecting the nodes. While repelling spreading particles, springs try to keep them at a fixed distance defined in `mLinkLength`.

![How it works…](img/8703OS_07_07.jpg)

## See also

*   **The Wikipedia article on Force-directed graph drawing**: [http://en.wikipedia.org/wiki/Force-based_algorithms_(graph_drawing)](http://en.wikipedia.org/wiki/Force-based_algorithms_(graph_drawing))