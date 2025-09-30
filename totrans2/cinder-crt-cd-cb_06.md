# Chapter 6. Rendering and Texturing Particle Systems

In this chapter we will learn about:

*   Texturing particles
*   Adding a tail to our particles
*   Creating a cloth simulation
*   Texturing a cloth simulation
*   Texturing the particle system using point sprites and shaders
*   Connecting particles

# Introduction

Continuing from [Chapter 5](ch05.html "Chapter 5. Building Particle Systems"), *Building Particle Systems*, we will learn how to render and apply textures to our particles in order to make them more appealing.

# Texturing particles

In this recipe we will render particles introduced in the previous chapter using texture loaded from the PNG file.

## Getting started

This recipe code base is an example of the recipe *Simulating particles flying on the wind* from [Chapter 5](ch05.html "Chapter 5. Building Particle Systems"), *Building Particle Systems*. We also need a texture for a single particle. You can prepare one easily with probably any graphical program. For this example, we are going to use a PNG file with transparency stored inside the `assets` folder with a name, `particle.png`. In this case it is just a radial gradient with transparency.

![Getting started](img/8703OS_06_01.jpg)

## How to do it…

We will render particles using the previously created texture.

1.  Include the necessary header files:

    [PRE0]

2.  Add a member to the application main class:

    [PRE1]

3.  Inside the `setup` method load `particleTexture`:

    [PRE2]

4.  We also have to change the particle size for this example:

    [PRE3]

5.  At the end of the `draw` method we will draw our particles as follows:

    [PRE4]

6.  Replace the `draw` method inside the `Particle.cpp` source file with the following code:

    [PRE5]

## How it works…

In step 5, we saw two important lines. One enables alpha blending and the other binds our texture stored in the `particleTexture` property. If you look at step 6, you can see we drew each particle as a rectangle and each rectangle had texture applied. It is a simple way of texturing particles and not very performance effective, but in this case, it works quite well. It is possible to change the color of drawing particles by changing the color just before invoking the `draw` method on `ParticleSystem`.

![How it works…](img/8703OS_06_02.jpg)

## See also

Look into the recipe *Texturing the particle system using Point sprites and shaders*

# Adding a tail to our particles

In this recipe, we will show you how to add a tail to the particle animation.

## Getting started

In this recipe we are going to use the code base from the recipe *Applying repulsion and attraction forces* from [Chapter 5](ch05.html "Chapter 5. Building Particle Systems"), *Building Particle Systems*.

## How to do it…

We will add a tail to the particles using different techniques.

### Drawing history

Simply replace the `draw` method with the following code:

[PRE6]

### Tail as a line

We will add a tail constructed from several lines.

1.  Add new properties to the `Particle` class inside the `Particle.h` header file:

    [PRE7]

2.  At the end of the `Particle` constructor, inside the `Particle.cpp` source file, set the default value to the `tailLength` property:

    [PRE8]

3.  At the end of the `update` method of the `Particle` class add the following code:

    [PRE9]

4.  Replace your `Particle::draw` method with the following code:

    [PRE10]

# How it works…

Now, we will explain how each technique works.

## Drawing history

The idea behind this method is very simple, instead of clearing the drawing area, we are continuously drawing semi-transparent rectangles that cover old drawing states more and more. This very simple method can give you interesting effects with particles. You can also manipulate the opacity of each rectangle by changing the alpha channel of the rectangle color, which becomes a color of the background.

![Drawing history](img/8703OS_06_03.jpg)

## Tail as a line

To draw a tail with lines, we have to store several particle positions and draw a line through these locations with variable opacity. The rule for opacity is just to draw older locations with less opacity. You can see the drawing code and alpha channel calculation in step 4

![Tail as a line](img/8703OS_06_04.jpg)

# Creating a cloth simulation

In this recipe we will learn how to simulate cloth by creating a grid of particles connected by springs.

## Getting Ready

In this recipe, we will be using the particle system described in the recipe *Creating a particle system in 2D* from [Chapter 5](ch05.html "Chapter 5. Building Particle Systems"), *Building Particle Systems*.

We will also be using the `Springs` class created in the recipe *Creating springs* from [Chapter 5](ch05.html "Chapter 5. Building Particle Systems"), *Building Particle Systems*.

So, you will need to add the following files to your project:

*   `Particle.h`
*   `ParticleSystem.h`
*   `Spring.h`
*   `Spring.cpp`

## How to do it…

We will create a grid of particles connected with springs to create a cloth simulation.

1.  Include the particle system file in your project by adding the following code on top of your source file:

    [PRE11]

2.  Add the `using` statements before the application class declaration as shown in the following code:

    [PRE12]

3.  Create an instance of a `ParticleSystem` object and member variables to store the top corners of the grid. We will also create variables to store the number of rows and lines that make up our grid. Add the following code in your application class:

    [PRE13]

4.  Before we start creating our particle grid, let's update and draw our particle system in our application's `update` and `draw` methods.

    [PRE14]

5.  In the `setup` method, let's initialize the grid corner positions and number of rows and lines. Add the following code at the top of the `setup` method:

    [PRE15]

6.  Calculate the distance between each particle on the grid.

    [PRE16]

7.  Let's create a grid of evenly spaced particles and add them to `ParticleSystem`. We'll do this by creating a nested loop where each loop index will be used to calculate the particle's position. Add the following code in the `setup` method:

    [PRE17]

8.  Now that the particles are created, we need to connect them with springs. Let's start by connecting each particle to the one directly below it. In a nested loop, we will calculate the index of the particle in `ParticleSystem` and the one below it. We then create a `Spring` class connecting both particles using their current distance as `rest` and a `strength` value of `1.0`. Add the following to the bottom of the `setup` method:

    [PRE18]

9.  We now have a static grid made out of particles and springs. Let's add some gravity by applying a constant vertical force to each particle. Add the following code at the bottom of the `update` method:

    [PRE19]

10.  To prevent the grid from falling down, we need to make the particles at the top edges static in their initial positions, defined by `mLeftCorner` and `mRightCorner`. Add the following code to the `update` method:

    [PRE20]

11.  Build and run the application; you'll see a grid of particles falling down with gravity, locked by its top corners.![How to do it…](img/8703OS_06_05.jpg)
12.  Let's add some interactivity by allowing the user to drag particles with the mouse. Declare a `Particle` pointer to store the particle being dragged.

    [PRE21]

13.  In the `setup` method initialize the particle to `NULL`.

    [PRE22]

14.  Declare the `mouseUp` and `mouseDown` methods in the application's class declaration.

    [PRE23]

15.  In the implementation of the `mouseDown` event, we iterate the overall particles and, if a particle is under the cursor, we set `mDragParticle` to point to it.

    [PRE24]

16.  In the `mouseUp` event we simply set `mDragParticle` to `NULL`.

    [PRE25]

17.  We need to check if `mDragParticle` is a valid pointer and set the particle's position to the mouse cursor. Add the following code to the `update` method:

    [PRE26]

18.  Build and run the application. Press and drag the mouse over any particle and drag it around to see how the cloth simulation reacts.

## How it works…

The cloth simulation is achieved by creating a two dimensional grid of particles and connecting them with springs. Each particle will be connected with a spring to the ones next to it and to the ones above and below it.

## There's more…

The density of the grid can be changed to accommodate the user's needs. Using a grid with more particles will generate a more precise simulation but will be slower.

Change `mNumLines` and `mNumRows` to change the number of particles that make up the grid.

# Texturing a cloth simulation

In this recipe, we will learn how to apply a texture to the cloth simulation we created in the *Creating a cloth simulation* recipe of the current chapter.

## Getting ready

We will be using the cloth simulation developed in the recipe *Creating a cloth Simulation* as the base for this recipe.

You will also need an image to use as texture; place it inside your `assets` folder. In this recipe we will name our image `texture.jpg`.

## How to do it…

We will calculate the correspondent texture coordinate to each particle in the cloth simulation and apply a texture.

1.  Include the necessary files to work with the texture and read images.

    [PRE27]

2.  Declare a `ci::gl::Texture` object in your application's class declaration.

    [PRE28]

3.  In the `setup` method load the image.

    [PRE29]

4.  We will remake the `draw` method. So we'll erase everything in it which was changed in the *Creating a cloth simulation* recipe and apply the `clear` method. Your `draw` method should be like the following:

    [PRE30]

5.  After the `clear` method call, enable the `VERTEX` and `TEXTURE COORD` arrays and bind the texture. Add the following to the `draw` method:

    [PRE31]

6.  We will now iterate over all particles and springs that make up the cloth simulation grid and draw a textured triangle strip between each row and the row next to it. Start by creating a `for` loop with `mNumRows-1` iterations and create two `std::vector<Vec2f>` containers to store vertex and texture coordinates.

    [PRE32]

7.  Inside the loop we will create a nested loop that will iterate over all lines in the cloth grid. In this loop we will calculate the index of the particles whose vertices will be drawn, calculate their correspondent texture coordinates, and add them with the positions of `textureCoords` and `vertexCoords`. Type the following code into the loop that we created in the previous step:

    [PRE33]

    Now that the `vertex` and `texture` coordinates are calculated and placed inside `vertexCoords` and `textureCoords` we will draw them. Here is the complete nested loop:

    [PRE34]

8.  Finally we need to unbind `mTexture` by adding the following:

    [PRE35]

## How it works…

We calculated the correspondent texture coordinate according to the particle's position on the grid. We then drew our texture as triangular strips formed by the particles on a row with the particles on the row next to it.

# Texturing a particle system using point sprites and shaders

In this recipe, we will learn how to apply a texture to all our particles using OpenGL point sprites and a GLSL Shader.

This method is optimized and allows for a large number of particles to be drawn at fast frame rates.

## Getting ready

We will be using the particle system developed in the recipe *Creating a particle system in 2D* from [Chapter 5](ch05.html "Chapter 5. Building Particle Systems"), *Building Particle Systems*. So we will need to add the following files to your project:

*   `Particle.h`
*   `ParticleSystem.h`

We will also be loading an image to use as texture. The image's size must be a power of two, such as 256 x 256 or 512 x 512\. Place the image inside the `assets` folder and name it `particle.png`.

## How to do it...

We will create a GLSL shader and then enable OpenGL point sprites to draw textured particles.

1.  Let's begin by creating the GLSL Shader. Create the following files:

    *   `shader.frag`
    *   `shader.vert`

    Add them to the `assets` folder.

2.  Open the file `shader.frag` in your IDE of choice and declare a `uniform sampler2D`:

    [PRE36]

3.  In the `main` function we use the texture to define the fragment color. Add the following code:

    [PRE37]

4.  Open the `shader.vert` file and create `float attribute` to store the particle's radiuses. In the `main` method we define the position, color, and point size attributes. Add the following code:

    [PRE38]

5.  Our shader is done! Let's go to our application source file and include the necessary files. Add the following code to your application source file:

    [PRE39]

6.  Declare the member variables to create a particle system and arrays to store the particle's positions and radiuses. Also declare a variable to store the number of particles.

    [PRE40]

7.  In the `setup` method, let's initialize `mNumParticles` to `1000` and allocate the arrays. We will also create the random particles.

    [PRE41]

8.  In the `update` method, we will update `mParticleSystem` and the `mPositions` and `mRadiuses` arrays. Add the following code to the `update` method:

    [PRE42]

9.  Declare the shaders and the particle's texture.

    [PRE43]

10.  Load the shaders and texture by adding the following code in the `setup` method:

    [PRE44]

11.  In the `draw` method, we will start by clearing the background with black, set the window's matrices, enable the additive blend, and bind the shader.

    [PRE45]

12.  Get the location for the `particleRadius` attribute in the `Vertex` shader. Enable vertex attribute arrays and set the pointer to `mRadiuses`.

    [PRE46]

13.  Enable point sprites and enable our shader to write to point sizes.

    [PRE47]

14.  Enable vertex arrays and set the vertex pointer to `mPositions`.

    [PRE48]

15.  Now enable and bind the texture, draw the vertex array as points, and unbind the texture.

    [PRE49]

16.  All we need to do now is disable the vertex arrays, disable the vertex attribute arrays, and unbind the shader.

    [PRE50]

17.  Build and run the application and you will see `1000` random particles with the applied texture.![How to do it...](img/8703OS_06_06.jpg)

## How it works…

Point sprites is a nice feature of OpenGL that allows for the application of an entire texture to a single point. It is extremely useful when drawing particle systems and is quite optimized, since it reduces the amount of information sent to the graphics card and performs most of the calculations on the GPU.

In the recipe we also created a GLSL shader, a high-level programming language, that allows more control over the programming pipeline, to define individual point sizes for each particle.

In the `update` method we updated the `Positions` and `Radiuses` arrays, so that if the particles are animated the arrays will represent the correct values.

## There's more…

Point sprites allow us to texturize points in 3D space. To draw the particle system in 3D do the following:

1.  Use the `Particle` class described in the *There's more…* section of the recipe *Creating a Particle system in 2D* from [Chapter 5](ch05.html "Chapter 5. Building Particle Systems"), *Building Particle Systems*.
2.  Declare and initialize `mPositions` as a `ci::Vec3f` array.
3.  In the `draw` method, indicate that the vertex pointer contains 3D information by applying the following change:

    [PRE51]

    Change the previous code line to:

    [PRE52]

4.  The vertex shader needs to adjust the point size according to the depth of the particle. The `shader.vert` file would need to read the following code:

    [PRE53]

# Connecting the dots

In this recipe we will show how to connect particles with lines and introduce another way of drawing particles.

## Getting started

This recipe's code base is an example from the recipe *Simulating particles flying on the wind* (from [Chapter 5](ch05.html "Chapter 5. Building Particle Systems"), *Building Particle Systems*), so please refer to this recipe.

## How to do it…

We will connect particles rendered as circles with lines.

1.  Change the number of particles to create inside the `setup` method:

    [PRE54]

2.  We will calculate `radius` and `mass` of each particle as follows:

    [PRE55]

3.  Replace the `draw` method inside the `Particle.cpp` source file with the following:

    [PRE56]

4.  Replace the `draw` method inside the `ParticleSystem.cpp` source file as follows:

    [PRE57]

## How it works…

The most interesting part of this example is mentioned in step 4\. We are iterating through all the points, actually through all possible pairs of the points, to connect it with a line and apply the right opacity. The opacity of the line connecting two particles is calculated from the distance between these two particles; the longer distance makes the connection line more transparent.

![How it works…](img/8703OS_06_07.jpg)

Have a look at how the particles are been drawn in step 3\. They are solid circles with a slightly bigger outer circle. The nice detail is the connection line that we are drawing between particles that stick to the edge of the outer circle, but don't cross it. We have done it in step 4, where we calculated the normalized vector of the vectors connecting two particles, then used them to move the attachment point towards that vector, multiplied by the outer circle radius.

![How it works…](img/8703OS_06_08.jpg)

# Connecting particles with spline

In this recipe we are going to learn how to connect particles with splines in 3D.

## Getting started

In this recipe we are going to use the particle's code base from the recipe *Creating a particle system*, from [Chapter 5](ch05.html "Chapter 5. Building Particle Systems"), *Building Particle Systems*. We are going to use the 3D version.

## How to do it…

We will create splines connecting particles.

1.  Include the necessary header file inside `ParticleSystem.h`:

    [PRE58]

2.  Add a new property to the `ParticleSystem` class:

    [PRE59]

3.  Implement the `computeBSpline` method for the `ParticleSystem` class:

    [PRE60]

4.  At the end of the `ParticleSystem` update method, invoke the following code:

    [PRE61]

5.  Replace the `draw` method of `ParticleSystem` with the following:

    [PRE62]

6.  Add headers to your main Cinder application class files:

    [PRE63]

7.  Add members for your `main` class:

    [PRE64]

8.  Implement the `setup` method as follows:

    [PRE65]

9.  Add members for camera navigation:

    [PRE66]

10.  Implement the `update` method as follows:

    [PRE67]

11.  Implement the `draw` method as follows:

    [PRE68]

## How it works…

**B-spline** lets us draw a very smooth curved line through some given points, in our case, particle positions. We can still apply some attraction and repulsion forces so that the line behaves quite like a spring. In Cinder, you can use B-splines in 2D and 3D space and calculate them with the `BSpline` class.

![How it works…](img/8703OS_06_09.jpg)

## See also

More details about B-spline are available at [http://en.wikipedia.org/wiki/B-spline](http://en.wikipedia.org/wiki/B-spline).