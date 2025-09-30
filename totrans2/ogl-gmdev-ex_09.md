# Chapter 9. Super Models

In the previous chapter, you created a framework to render OpenGL in 3D. At the very end of that chapter, we added a block of code that rendered a cube. In this chapter, you will learn how to create 3D objects in Open GL, first using code, and then using a 3D modeling program. In this chapter, we will cover the following:

*   **Graphics cards**: 3D graphics cards are basically small computers that are optimized to render objects in 3D. We will take a quick look at how a graphics card does what it does best.
*   **Vertices**: 3D objects are drawn by plotting points and telling OpenGL to use these points to create an object that can be rendered on the screen.
*   **Triangles**: Triangles are used to create all 3D objects. You will learn about the relationship between vertices and triangles and how they are used to create simple objects.
*   **Modeling**: Once you understand how to create simple 3D objects using code, you will also understand that you are going to need a more effective tool if you ever want to create anything complicated. This is where 3D modeling software comes in and saves the day.
*   Once you create a 3D model, you have to get the model into the game. We will create the code to load a 3D model into our game by reading the data that is created by the modeling software.

# New Space

Until now, we have been working only in a two-dimensional space. This means that we were able create game objects with height and width. This works well because our computer screens are also two-dimensional. As we move into three-dimensional space, we need the ability to add another dimension to our objects: depth. As computer screens don't physically have a third dimension in which to display pixels, this is all accomplished by mathematical wizardry!

In [Chapter 8](ch08.html "Chapter 8. Expanding Your Horizons"), *Expanding Your Horizons* we discussed several methods that have been used (and are still used) to simulate three-dimensions in a two-dimensional display:

*   Objects that are farther away can be made to appear smaller than objects that are close
*   Objects that are farther away can be made to move more slowly than objects that are close
*   Lines that are parallel can be drawn to converge toward the center as they are farther away

These three techniques have one major shortcoming: they all required the programmer to write code that makes each visual effect work. For example, the programmer has to make sure that objects that are receding from the player are constantly scaled down so that they become increasingly smaller.

In a true 3D game, the only thing that the programmer has to worry about is placing each object at the right coordinates in 3D space. A special graphics card takes care of performing all of the calculations to take care of size, speed, and parallax. This frees the programmer up from doing these calculations, but it actually adds a whole new set of requirements related to positing and rotating objects in three-dimensional space.

# A computer in a computer

The thing about what it takes for your computer to process your game. The computer must receive input from the player, interpret that input, and then apply the results to the game. Once the input is completed, the computer must handle the physics of the game: objects must be moved, collisions must happen, and explosions must ensue. Once the computer has completed updating all of the objects in the game, it must then render these results to the screen. Finally, in order to be convincing, all of this must occur at least 30 times a second and often 60 times a second!

It is truly amazing that computers can process this much information that quickly. In fact, if it were truly up to the central processing unit of your computer to accomplish this, then it wouldn't be able to keep up.

The 3D graphics card solves this problem by taking care of the rendering process so that the main CPU of your computer doesn't have to. All your CPU has to do is deliver the data and the graphics card takes care of the rest, allowing the main CPU to continue processing other things.

A modern 3D graphics card is really an entire computer system that lives on a silicon card inside your main computer. The graphics card is a computer inside your computer! The graphics card has its own input, output, and its own processor known as the graphics processing unit, or GPU. It also contains its own memory, often up to 4 gigabytes or more.

The following diagram shows you the basic structure of a graphics card and how it processes information:

![A computer in a computer](img/8199OS_09_01.jpg)

The preceding sequence that is depicted is known as the graphics pipeline. A detailed discussion of every step in the process is beyond the scope of our book, but it is good to have a basic understanding of the graphics pipeline, so here are the basics:

*   **Graphics bus**: In computer lingo, a bus is just a way to move data. Think of a bus as a freeway: the more lanes you have on our freeway, the faster the traffic can move. Inside your computer, the traffic is bits of data, and most modern graphics cards have 64 lanes (known as a 64-bit bus), which allows up to 64-bits (or 8 bytes) of data to be moved simultaneously. The graphics bus receives its data directly from the CPU.

    *   **Graphics Processing Unit**: The GPU does all the work, and as you can see, there is a lot of work to do.
    *   **Transformation**: Each vertex, represented as a point in 3D space, must be properly positioned. There are several frames of reference to deal with. For example, local coordinates may describe how far a car's tires are from its body, while global coordinates describe how far the car is from the upcoming cliff. All of the data must be transformed into a single frame of reference.
    *   **Lighting**: Each vertex must be lit. This means applying light and color to each vertex and interpolating the light and color intensity from one vertex to another. In the same way that the sun lights our world, while fluorescent tubes light our offices, the GPU uses lighting data to correctly light the world of your game.
    *   **Primitives**: These are the simple objects that are used to build more complicated objects. Similarly to a virtual Lego set, the GPU constructs everything in your game using triangles, rectangles, circles, cube, spheres, cones, and cylinders. We will learn more about this later in the chapter.
    *   **Projection**: Once the GPU has constructed a 3D model of the world, it must now create a 3D projection of the world onto 2D space (remember, your display only has two dimensions). This is similar to how the sun projects a 2D shadow of 3D objects.
    *   **Clipping**: Once the 3D scene has been projected into 2D space, some vertices will be behind other vertices, and, therefore, can't actually be seen at this time. Clipping, or removing vertices that can't be seen, removes these vertices from the data, streamlining the entire process.
    *   **Rasterization**: We now have a 2D model that mathematically represents the current image that must be displayed onto the screen. Rasterization is the process of converting this virtual image into actual pixels that must be displayed on the screen.
    *   **Shading**: This final process determines the actual color that must be applied to each pixel on the screen to correctly display the model that has been created in the earlier phases. Code can even be written to manipulate the process to create special visual effects. Code that modifies the shading process in the graphics pipeline is called a shader.

*   **Render**: Of course, the reason that we do all of this is so that we can display our game on the computer screen. The final output of the graphics pipeline is a representation of the current screen in the render buffer. Now, all the CPU has to do is swap the data in the render buffer to the actual screen buffer, and the result is the next frame in your game!

By the way, you will notice that behind the scenes (the big arrow in the preceding image) everything is supported by dedicated memory on the graphics card. All of the data is moved from the CPU to the memory of the graphics card, where it is manipulated and processed before being sent back to the CPU. This means that memory on the main computer doesn't have to be set aside to handle graphics processing.

### Tip

It is important to understand that the preceding diagram is a generic representation of the graphics pipeline. Specific hardware on various graphics cards may handle things differently, and the OpenGL and DirectX specifications are slightly different, but the preceding diagram is still the basic pipeline.

# Drawing your weapons

It's time for us to learn how to draw things in OpenGL. Whether you are drawing your weapons, an alien spacecraft, or a blade of grass, it all starts by with very simple shapes that are combined to make more complex shapes.

## Getting primitive

The most basic shapes that can be drawn in OpenGL are known as primitives. The primitives that can be drawn by OpenGL include:

*   **Points**: As the name suggests, a point renders a single point and is defined by a single vertex.
*   **Lines**: A line is rendered as a line drawn between two vertices.
*   **Triangles**: A triangle is defined by three vertices and the three lines that pass from one vertex to the other.
*   **Quads**: A quad is defined by four vertices and the four lines that pass from one vertex to the other. Technically, a quad is actually two triangles that have been joined together at the hypotenuse.

That's it, folks! Everything known to exist can be created from these four primitives. Extrapolating into 3D, there are these 3D primitives:

*   A plane is a 2D extrusion of a line (okay, I know that a plane isn't really 3D!)
*   A pyramid is a 3D representation of a quad and four triangles
*   A cube is the 3D extrusion of a quad
*   A sphere is a 3D construct based on a circle, which is created by lines (yes, lines, and the shorter each line, the more convincing the circle)
*   A cylinder is a 3D extrusion of a circle

The objects in the preceding list aren't actually defined as OpenGL primitives. However, many 3D modeling programs refer to them as primitives because they are the simplest 3D objects to create.

## Drawing primitives

In the previous chapter, we created a cube using the following code:

[PRE0]

Now, let's learn about how this code actually works:

1.  Any time that we want to draw something in OpenGL, we first start by clearing the render buffer. In other words, every frame is drawn from scratch. The `glClear` function clears the buffer so that we can start drawing to it.
2.  Before we start drawing objects, we want to tell OpenGL where to draw them. The `glTranslatef` command moves us to a certain point in 3D space from which we will start our drawing (actually, `glTranslatef` moves the camera, but the effect is the same).
3.  If we want to rotate our object, then we provide that information with the `glRotatef` function. Recall that the cube in the previous chapter slowly rotated.
4.  Just before we provide vertices to OpenGL, we need to tell OpenGL how to interpret these vertices. Are they single points? Lines? Triangles? In our case, we defined vertices for the six squares that will make the faces of our cube, so we specify `glBegin(GL_QUADS)` to let OpenGL know that we are going to be providing the vertices for each quad. There are several other possibilities that we will describe next.
5.  In OpenGL, you specify the properties for each vertex just before you define the vertex. For example, we use the `glColor3f` function to define the color for the next set of vertices that we define. Each succeeding vertex will be drawn in this specified color until we change the color with another call to `glColor3f`.
6.  Finally, we define each vertex for the quad. As a quad requires four vertices, the next four `glVertex3f` calls will define one quad. If you look closely at the code, you will notice that there are six groups of four vertex definitions (each preceded by a color definition), which all work together to create the six faces of our cube.

Now that you understand how OpenGL draws quads, let's expand your knowledge by covering the other types of primitives.

## Making your point

There is only one kind of point primitive.

### Gl_Points

The `glBegin(GL_POINTS)` function call tells OpenGL that each following vertex is to be rendered as a single point. Points can even have texture mapped onto them, and these are known as **point sprites**.

Points are actually generated as squares of pixels based on the size defined by the `GL_PROGRAM_POINT_SIZE` parameter of the `glEnable` function. The size defines the number of pixels that each side of the point takes up. The point's position is defined as the center of that square.

The point size must be greater than zero, or else an undefined behavior results. There is an implementation-defined range for point sizes, and the size given by either method is clamped to that range. Two additional OpenGL properties determine how points are rendered: `GL_POINT_SIZE_RANGE` (returns 2 floats), and `GL_POINT_SIZE_GRANULARITY`. This particular OpenGL implementation will clamp sizes to the nearest multiple of the granularity.

## Getting in line

There are three kinds of line primitives, based on different interpretations the vertex list.

### Gl_Lines

When you call `glBegin(GL_LINES)`, every pair of vertices is interpreted as a single line. Vertices 1 and 2 are considered one line. Vertices 3 and 4 are considered another line. If the user specifies an odd number of vertices, then the extra vertex is ignored.

### Gl_Line_Strip

When you call `glBegin(GL_LINES)`, the first vertex defines the start of the first line. Each vertex thereafter defines the end of the previous line and the start of the next line. This has the effect of chaining the lines together up to the last vertex in the list. Thus, if you pass *n* vertices, you will get *n-1* lines. If the user only specifies only one vertex, the drawing command is ignored.

![Gl_Line_Strip](img/8199OS_09_02.jpg)

### Gl_Line_Loop

The call `glBegin(GL_LINE_LOOP)` works almost exactly like line strips, except that the first and last vertices are joined as a line. Thus, you get n lines for *n* input vertices. If the user only specifies one vertex, the drawing command is ignored. The line between the first and last vertices happens after all of the previous lines in the sequence.

![Gl_Line_Loop](img/8199OS_09_03.jpg)

## Triangulation

A triangle is a primitive formed by three vertices. There are three kinds of triangle primitives, based again on different interpretations of the vertex stream.

### Gl_Triangles

When you call `glBegin(GL_TRIANGLES)`, every three vertices define a triangle. Vertices 1, 2, and 3 form one triangle. Vertices 4, 5, and 6 form another triangle. If there are fewer than three vertices at the end of the list, they are ignored:

[PRE1]

### Gl_Triangle_Strip

When you call `glBegin(GL_TRIANGLE_STRIP)`, the first three vertices create the first triangle. Thereafter, the next two vertices create the next triangle, creating a group of adjacent triangles. A vertex stream of n length will generate *n-2* triangles:

![Gl_Triangle_Strip](img/8199OS_09_04.jpg)

### Gl_Triangle_Fan

When you call `glBegin(GL_TRIANGLE_FAN)`, the first vertex defines the point from which all other triangles are defined. Thereafter, each group of two vertices define a new triangle with the same apex as the first one, forming a fan. A vertex stream of *n* length will generate *n-2* triangles. Any leftover vertices will be ignored:

![Gl_Triangle_Fan](img/8199OS_09_05.jpg)

## Being square

A quad is a quadrilateral, having four sides. Don't get confused and think that all quads are either squares or rectangles. Any shape with four sides is a quad. The four vertices are expected to be in the same plane and failure to do so can lead to undefined results. A quad is typically constructed as a pair of triangles, which can lead to artifacts (unanticipated glitches in the image).

### Gl_Quads

When you call `glBegin(GL_QUADS)`, each set of four vertices defines a quad. Vertices 1 to 4 form one quad, while vertices 5 to 8 form another. The vertex list must be a number of vertices divisible by 4 to work:

[PRE2]

### Gl_Quad_Strip

Similarly to triangle strips, a quad strip uses adjacent edges to form the next quad. In the case of quads, the third and fourth vertices of one quad are used as the edge of the next quad. So, vertices 1 to 4 define the first quad, while 5 to 6 extend the next quad. A vertex list of *n* length will generate *(n - 2)/2* quads:

![Gl_Quad_Strip](img/8199OS_09_06.jpg)

## Saving face

All of the primitives that we discussed are created by creating multiple shapes that are glued together, more or less. OpenGL needs to know which face of a shape is facing the camera, and this is determined by the winding order. As you can't see both the front and back of a primitive, OpenGL uses facing to decide which side must be rendered.

In general, OpenGL takes care of the winding order so that all of the shapes in a particular list have consistent facing. If you, as a coder, try to take care of facing manually, you are actually second-guessing OpenGL.

## Back to Egypt

As we have already demonstrated the code to draw a cube, let's try something even more interesting: a **pyramid**. A pyramid is constructed by four triangles with a square on the bottom. So, the simplest way to create a pyramid is to create four `GL_TRIANGLE` primitives and one `GL_QUAD` primitive:

[PRE3]

# A modeling career

When you consider the amount of code that is required to create even the most basic shapes, you might despair of ever coding a complicated 3D game! Fortunately, there are better tools available to create 3D objects. 3D modeling software allows a 3D modeler to create 3D object similar to how an artist uses drawing software to create 2D images.

The process of getting 3D objects into our game typically has three steps:

1.  Creating the 3D object in a 3D modeling tool.
2.  Exporting the model as a data file.
3.  Loading the data file into our game.

## Blending in

There are many popular tools that are used by professionals to create 3D models. Two of the most popular ones are 3D Max and Maya. However, these tools are also relatively expensive. It turns out that there is a very capable 3D modeling tool called **Blender** that is available for free. We will install Blender and then learn how to use it to create 3D models for our game.

Blender is a 3D modeling and animation suite that is perfect for beginners who want to try 3D modeling. Blender is open-source software created by Blender Organization, and it is available at no cost (although Blender Organization will be glad to accept your donations). Install Blender on your computer using the following steps:

1.  Go to [http://www.Blender.Org](http://www.Blender.Org) and hit *Enter*.
2.  Click the **Download** link at the top of the page.
3.  Download the files that are compatible with your computer. For my 64-bit Windows computer, I made the selection circled in the following screenshot:![Blending in](img/8199OS_09_07.jpg)
4.  Once Blender is downloaded, run the installer program and accept all of the default values to install Blender on your computer.

## Blender overview

Once you have installed Blender on your computer, open it up and you should see something like the following screen:

![Blender overview](img/8199OS_09_08.jpg)

Don't let the complexity of the screen scare you. Blender has a lot of features that you will learn with time, and they have tried to put many of the features right at your fingertips (well, mouse tips). They have even created a model of a cube for you so that you can get started right away.

The middle of the screen is where the action takes place. This is the 3D view. The grid gives you a reference, but is not part of the model. In the preceding screenshot, the only model is the cube.

The panels surrounding the middle offer a host of options to create and manipulate your objects. We won't have time to cover most of these, but there are many tutorials available online.

## Building your spaceship

Just like we did in the 2D portion of the book, we are going to build a simple 3D spaceship so that we can fly it around in our universe. As I am a programmer and not a modeler, it will be a ridiculously simple space ship. Let's build it out of a cylinder.

To build our space ship, we first want to get rid of the cube. Use your right mouse button to select the cube. You can tell that it is selected because it will have three arrows coming from it:

![Building your spaceship](img/8199OS_09_09.jpg)

Now press the *Delete* key on your keyboard, and the cube will disappear.

### Tip

If you are like me, you will try and try to use the left mouse button to select objects. However, Blender uses the right mouse button to select objects!

You will probably notice two other objects in the 3D View:

![Building your spaceship](img/8199OS_09_10.jpg)

The object in the preceding image represents the camera. This is not a part of your game object, but rather it represents the angle of the camera as viewed from inside Blender. You can hide this by right-clicking on it and pressing *H*.

![Building your spaceship](img/8199OS_09_11.jpg)

The object in the preceding image represents the light source. This is not a part of your game object, but rather it represents the light source that Blender is using. You can hide this by right-clicking on it and pressing *H*.

Now, let's create that cylinder. Locate the **Create** tab in the left panel and use your left mouse button to click on it:

![Building your spaceship](img/8199OS_09_12.jpg)

Next, click on the cylinder button. Blender will create a cylinder in the 3D view:

![Building your spaceship](img/8199OS_09_13.jpg)

Notice the three arrows. These indicate that the cylinder is the selected object. The arrows are used to move, size, and rotate objects, but we won't be doing any of that today.

You should also notice a circle with a concentric dashed circle inside the cylinder. This indicates the origin of the object, which is the point around which the object will move, size, and rotate.

There are many more things that we would do if we were modeling a real object. As this is a coding book and not a modeling book, we won't do those things, but here are some ideas for future study:

*   We could continue creating more and more objects and use them to build a much more complex spaceship
*   We could use textures and materials to give our spaceship a skin

## Exporting the object

In order to bring the spaceship into our game, we must first export the object into a data file that can be read into the game. There are many different formats that we could use, but for this game, we will use the `.obj` export type. To export the object, perform the following action:

1.  Click the **File** command, then click **Export**.
2.  Choose **Wavefront (.obj)** as the file type.
3.  In the next screen, select the location for your export (preferably the location of your source code for the game) and name it `ship.obj`.
4.  Click the **Export OBJ** button on the right-hand side of the screen.![Exporting the object](img/8199OS_09_14.jpg)

Congratulations! You are now one step away from bringing this object into your game.

## Getting loaded

The `.obj` file is simply a text file that stores all of the vertices and other data that is used to render this object in OpenGL. The following screenshot shows the `ship.obj` file opened in Notepad:

![Getting loaded](img/8199OS_09_15.jpg)

*   `#`: This defines a comment
*   `v`: This defines a vertex
*   `vt`: This defines a texture coordinate
*   `vn`: This defines a normal
*   `f`: This defines a face

We will now write the code to load this data into our game. Open the SpaceRacer3D project into Visual Studio. Then add the following headers:

[PRE4]

Here is what the loader is doing:

The loader accepts for parameters (one input and three output):

*   A filename.
*   A pointer to an array of vertices.
*   A pointer to an array of uvs.
*   A pointer to an array of normal vectors.
*   Three vectors (a type of array in C++) are created to hold the data that is parsed from the file. One to hold the vertices, one to hold the uvs, and one to hold the normals. A fourth vector is created to pair each vertex with a uv coordinate.
*   Three temporary vectors are created to use as input buffers as the data is read.
*   The `fbx` file is now read. The program looks for the flags that indicate what type of data is being read. For our purposes now, we are only concerned with the vertex data.
*   When each piece of data is read, it is put into the appropriate vector.
*   The vectors are returned so that they can be processed by the program.

Simple enough, eh? But, there's a lot of code because parsing is always fun! The most important data that is extracted from the model for our purposes is the array of vertices.

### Tip

We haven't discussed uvs and normal vectors because I don't want to this to be a whole book on modeling. Uvs are used to add textures to an object. as we didn't add any textures, we won't have uv data. Normal vectors tell OpenGL which side of an object is facing out. This data is used to properly render and light an object.

In the next chapter, we will use this loader to load our model into the game.

# Summary

We covered a lot of ground in this chapter. You learned how to create 3D objects in code using OpenGL. At the same time, you learned that you don't really create 3D objects in code! Instead, real games use models that have been created in special 3D modeling software, such as Blender.

Even as a coder, it is useful to learn a little bit about using software, such as Blender, but you will eventually want to find artists and modelers who really know now to use these tools to their full extent. You can even find 3D models online and integrate them into your game.

To close things out, we learned how to load 3D models into our. Spend a few days playing around with Blender and see what you can come up with, and then on to the next chapter!