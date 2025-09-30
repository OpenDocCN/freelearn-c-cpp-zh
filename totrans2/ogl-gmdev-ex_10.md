# Chapter 10. Expanding Space

Now that you know how to build your 3D world, it is time to do stuff! As we are building a space racing game, we need to be able to move our space ship around. We will also put some obstacles in the game so that we have something to race against. In this chapter, you will learn about the following topics:

*   **Placing game objects**: We will take some 3D objects, load them into our game, and place them in 3D space.
*   **Transformations**: We need to learn how to move in 3D. Moving in 2D was easy. In 3D, we have another dimension, and we will now also want to account for rotation as we move around.
*   **Point of view**: We will learn how the point of view affects how we play the game. Do you want to be in the pilot's seat or just outside the ship?
*   **Collisions**: We performed some collision detection in our 2D game. Collision detection in 3D is more complicated because we now have to consider all three spatial dimensions in our checks.

# Creation 101

Our first task is to load our world. We need a few basic components. First, we need a universe. This universe will contain stars, asteroids, and our space ship. Open up SpaceRacer3D and let's get coding!

## Preparing the project

Before we get going, we will need to move some code over from our 2D project. Copy the following files and settings from RoboRacer2D to SpaceRacer3D:

1.  Copy `Input.cpp` and `Input.h`—we will use these classes to handle user input.
2.  Copy `Sprite.cpp`, `Sprite.h`, `SOIL.h`, and `SOIL.lib`—we will use them to support the user interface in the next chapter. You may need to remove the line `#include "stdafx.h"` from `Sprite.cpp`.
3.  Copy `fmodex.dll`—we need this for audio support.
4.  Copy the settings from the project `Configuration Properties/C/C++/General/Additional Include Directories` setting—this is necessary to provide access to FMOD library:![Preparing the project](img/8199OS_10_04.jpg)
5.  Copy the settings from the project `Configuration Properties/Linker/Input/ Additional Dependencies` setting—this is necessary to provide access to the OpenGL, FMOD, and SOIL libraries:![Preparing the project](img/8199OS_10_05.jpg)
6.  Copy the settings from the project Configuration Properties/Linker/ General/Additional Library Directories setting—this is also necessary to provide access to FMOD library:![Preparing the project](img/8199OS_10_01a.jpg)

## Loading game objects

In the previous chapter, we learned how to create 3D objects in Blender and export them as `obj` files. We then added code to our project to load the `obj` data. Now, we will use that code to load some models into our game.

We are going to load four models into our game: the space ship, and three asteroids. The idea will be to race through the asteroid field. As our loader holds the model data as three arrays (vertices, uvs, and normals), we will create a model class that defines these arrays and then use this class for each model that we want to load into the game.

### The Model class header

Create a new class and header file named `Model.cpp` and `Model.h`, respectively. Open `Model.h`. First, let's get the header set up:

[PRE0]

We need to use some constants defined in `math.h`, so we need to add a preprocessor directive. Add `_USE_MATH_DEFINES` to `Configuration Properties/C/C++/Preprocessor/Preprocessor Definitions`. Also, notice that we include `LoadObj.h` because we will load the model from inside this class. Now, let's create the class:

[PRE1]

We will be using color a lot, so we are defining a `struct` to hold the `r`, `g`, and `b` values to make things more convenient. Now, for our methods we use the following code:

[PRE2]

Here is a short description of each method:

*   `Model` is the constructor. It takes a filename and a color. As our models are simple shapes, we will use color to give them some pizzazz.
*   `SetPosition` and `GetPosition` manage the object's position in world space.
*   `SetHeading` and `GetHeading` manage the direction the object is heading.
*   `SetColor` and `GetColor` manage the objects color.
*   `SetBaseRotation` and `GetBaseRotation` manage any local rotation applied to the object.
*   `SetHeadingRotation` and `GetHeadingRotation` manage the orientation of the object in world space.
*   `SetVelocity` and `GetVelocity` manage the speed of the object.

Now, for the variables, we use the following code:

[PRE3]

These are self-explanatory because they directly correspond to the methods described previously. This header is a good structure for everything that we will need to do to place objects in our world and move them around.

### Implementing the Model class

Now let's implement the class. Open `Model.cpp` and let's get started. First, we implement the header, constructor, and destructor:

[PRE4]

The constructor sets everything up. Notice that we call `LoadObj` from the constructor to actually load the object into the class. The results will be stored into member arrays `m_vertices` and `m_normals`. `m_primitive` will hold an enum telling us whether this object is defined by quads or triangles. The remaining variables are set to default values. These can be defined at any time in the game by using the appropriate `accessor` method:

[PRE5]

`Deg2Rad` is a helper function that will convert degrees to radians. As we move the ship around, we keep track of the heading angle in degrees, but we often need to use radians in OpenGL functions:

[PRE6]

The `Update` function updates the position of the object based on the object's velocity. Finally, we update `m_heading`, which will be used to orient the world camera during the render. Then update the object's position in world space:

[PRE7]

The `Render` function takes care of rendering this particular object. The setup for the world matrix will happen in the game code. Then each object in the game will be rendered.

Remember the camera? The camera is a virtual object that is used to view the scene. In our case, the camera is the ship. Wherever the ship goes, the camera will go. Whatever the ship points at, the camera will point at.

Now for the real mind-blower; OpenGL doesn't really have a camera. That is, there really isn't a camera that you move around in the scene. Instead, the camera is always located at coordinates (**0.0, 0.0, 0.0**), or the world's origin. This means that our ship will always be located at the origin. Instead of moving the ship, we will actually move the other objects in the opposite direction. When we turn the ship, we will actually rotate the world in the opposite direction.

Now look at the code for the `Render` function:

*   First, we use `glRotate` to rotate everything the object's base rotation. This comes in useful if we need to orient the object. For example, the cylinder that we modeled in the previous chapter is standing up, and it works better in the game lying on its side. You will see later that we apply a 90 degree rotation to the cylinder to achieve this.
*   Next, we have to decide whether we are going to render quads or triangles. When Blender exports a model, it exports it as either quads or triangles. The loader figures out whether a model is defined as quads or triangles and stores the result in `m_primitive`. We then use that to determine whether this particular object must be rendered using triangles or quads.
*   We use `glColor` to set the color of the object. At this point we haven't assigned any textures to our models, so color gives us a simple way to give each object a personality.

Now for the real work! We need to draw each vertex of the object in world space. To do this, we loop through each point in the vertices array, and we use `glVertex3f` to place each point.

The catch is this; the points in the vertices array are in local coordinates. If we drew every object using these points, then they would all be drawn at the origin. You will recall that we want to place each object in the game relative to the ship. So, we draw the ship at the origin, and we draw every other object in the game based on the position of the ship. We move the universe, not the ship.

![Implementing the Model class](img/8199OS_10_01.jpg)

When the ship moves, the entire coordinate system moves with it. Actually, the coordinate system stays put and the entire universe moves past it!

![Implementing the Model class](img/8199OS_10_02.jpg)

If we happen to be rendering the ship, we just draw it using its local coordinates and it is rendered at the origin. All of the other objects are drawn at a distance away from the ship based on the ships position.

Now, for the rest of the class implementation, use the following code:

[PRE8]

These methods set and retrieve the object's position. The position is changed based on the object's velocity in the `Update` method:

[PRE9]

These methods set and retrieve the object's heading. The heading is changed based on the object's heading rotations in the `Update` method. Heading is the direction that the ship is headed in, and is used to rotate the world so that the ship appears to be heading in the correct direction:

[PRE10]

These methods are used to manage the object's color:

[PRE11]

These methods are used to manage the object's velocity. The velocity is set in the game code during the input phase:

[PRE12]

These methods are used to manage the object's base rotation. The base rotation is used to rotate the object in local space:

[PRE13]

These methods are used to manage the object's heading rotation. The heading rotation is used to rotate the world around the object so that the object appears to be heading in a particular direction. Only one object (the ship) will have a heading rotation. Another way to think about this is that the heading rotation is the rotation of the camera, which in our game is attached to the ship.

### Modifying the game code

Now it's time to modify our game code so that it can load and manipulate game models. Open `SpaceRacer3D.cpp`.

We'll start by adding the appropriate headers. At the top of the code, modify the header definitions so that they look like the following code:

[PRE14]

Notice that we have added `Model.h` to load our models. We also included `Sprite.h` and `Input.h` from RoboRacer2D so that we can use those classes in our new game when necessary.

Now, we need to define some global variables to manage model loading. Just under any global variables that are already defined, add the following code:

[PRE15]

These variables defined pointers to our game objects. As the ship is kind of special, we give it its own pointer. We want to be able to have an arbitrary number of asteroids; we set up a vector (a nice dynamic array) of pointers called asteroids.

Move down to the `StartGame` function, which we use to initialize all of our game models. Modify the `StartGame` function to look like the following code:

[PRE16]

We are going to create one object for the ship and three asteroids. For each object, we first define a color, then we create a new `Model` passing the filename of the object and the color. The `Model` class will load the object file exported from Blender.

Notice that we set the ship to be the camera with the `IsCamera(true)` call. We also attach the ship as the camera for every game object using the `AttachCamera(ship)` call.

We also set a position for each object. This will set the position in world space. This way we don't end up drawing every object at the origin!

Each asteroid is put in the asteroids array using the `push.back` method.

Now, we move to the `Update` function. Modify the `Update` function so that it looks like the following code:

[PRE17]

The update simply calls the `Update` method for every object in the game. As always, the update is based on the amount of time that has passed in the game, so we pass in `p_deltaTime`.

Now on to the `Render` function. Replace the existing code with the following code:

[PRE18]

The rendering code is the real workhorse of the game. First, we set up the render call for this frame, then we call the `Render` method for each game object:

*   `GlClear`: This clears the render buffer.
*   `GlMatrixMode`: This sets the model to the model view. All translations and rotations are applied to the in the model view.
*   `glLoadIdentity()`: This resets the matrix.
*   Next, we call the `Render` method for each object in the game.
*   Finally, we call `SwapBuffers`, which actually renders the scene to the screen.

Congratulations! If you run the game now, you should see the ship and the three asteroids off in the distance. As we set the velocity of the ship to 1.0, you should also see the ship slowly moving past the asteroids. However, we don't have any way to control the ship yet because we haven't implemented any input.

![Modifying the game code](img/8199OS_10_06.jpg)

# Taking control

We now have a framework to load and render game objects. But, we don't have any way to move our ship! The good news is that we already wrote an input class for RoboRacer2D, and we can reuse that code here.

## Implementing input

Earlier in the chapter, I had you copy the `Input` class from RoboRacer2D into the source folder for SpaceRacer3D. Now, we have to simply wire it into our game code.

Open SpaceRacer3D. First, we need to include the input header. Add the following line of code to the headers:

[PRE19]

We also need to create a global pointer to manage the `Input` class. Add the following line just below the model pointers:

[PRE20]

Next, we need to create an instance of the `Input` class. Add the following line of code to the top of the `StartGame` function:

[PRE21]

Now, we have to create a function to handle our input. Add the following function just above the `Update` method:

[PRE22]

This code handles keyboard input. You will recall from RoboRacer2D that we mapped virtual commands to the following keys:

*   `CM_STOP`: This is the spacebar. We use the spacebar as a toggle to both start and stop the ship. If the ship is stopped, pressing the spacebar sets the velocity. If the ship's velocity is greater than zero, then pressing the spacebar sets the velocity to zero.
*   `CM_UP`: This is both the up arrow and the *W* key. Pressing either of these keys changes the heading rotation so that the ship moves up.
*   `CM_DOWN`: This is both the down arrow and the *S* key. Pressing either of these keys changes the heading rotation so that the ship moves down.
*   `CM_LEFT`: This is both the left arrow and the *A* key. Pressing either of these keys changes the heading rotation so that the ship moves left.
*   `CM_RIGHT`: This is both the right arrow and the *D* key. Pressing either of these keys changes the heading rotation so that the ship moves up.

Every directional command works by retrieving the current heading angle and changing the appropriate component of the heading vector by one degree. The heading angle is used by each object's `Update` method to calculate a heading vector, which is used to point the camera in the `Render` method.

Finally, we need to call `HandleInput` from the games `Update` function. Add the following line of code to the top of the `Update` method, before the object update calls. We want to handle input first and then call each object's update method:

[PRE23]

That's it! Pat yourself on the back and run the game. You can now use the keyboard to control the ship and navigate through your universe.

# Asteroid slalom

It's now time to implement the final feature of this chapter. We are going to implement a slalom race with a twist. In a typical slalom, the point is to race around each obstacle without touching it. To keep things simple, we are going to race through each asteroid. If you successfully pass through each asteroid, you win the race.

## Setting up collision detection

In order to determine whether you have passed through an asteroid, we have to implement some 3D collision detection. There are many types of collision detection, but we are going to keep it simple and implement spherical collision detection.

Spherical collision detection is a simple check to see whether the center of two 3D objects are within a certain distance of each other. As our asteroids are spheres, this will be a pretty accurate indication as to whether we have collided with one. The ship, however, is not a sphere, so this technique isn't perfect.

Let's start our collision detection coding by adding the appropriate methods to the `Model` class. Open `Model.h` and add the following methods:

[PRE24]

Here is how we will use each method:

*   `IsCollideable` is used to either get or set the `m_collideable` flag. Objects are set to collide by default. All of the objects in our game are set to collide so that we can detect if the ship has hit an asteroid. However, it is very common to have some objects in a game that you don't collide with. If you set `IsCollideable(false)`, then collision detection will be ignored.
*   `CollidedWith` is the method that performs the actual collision detection.
*   `GetCenter` is a helper function that calculates the center point of the object in world space.
*   `SetRadius` and `GetRadius` are help functions to manage the collision radius for the object.

We also need to add two variables to track the radius and collision:

[PRE25]

Now, open `Model.cpp` and add the following code to implement the collision methods.

First, we need to define the radius in the constructor. Add the following line of code to the constructor:

[PRE26]

Now add the following methods:

[PRE27]

*   `IsCollideable` and the override are used to either get whether the object can be collided with or get the state of the collision flag.
*   `GetCenter` returns the current position of the object. As we modeled all of our objects with the object origin at the center, returning the position also returns the center of the object. A more sophisticated algorithm would use the bounding size of the object to calculate the center.
*   `GetRadius` and `SetRadius` manage the radius, which is required for the collision check code.
*   `CollidedWith` is the method that performs all the work. After checking that both the current object and the target objects can collide, then the method performs the following actions:

    *   Gets the center point of the two objects
    *   Calculates the distance in 3D between the two centers
    *   Checks to see whether the distance is less than the sum of the two radii. If so, the objects have collided:

    ![Setting up collision detection](img/8199OS_10_03.jpg)

If you are astute, you will notice that this collision detection is very similar to the collision detection used in RoboRacer2D. We simply added the z dimension to the equations.

## Turning on collision

Now, we will implement the collision code in our game. Open `SpaceRacer3D.cpp` and add the following function just before the `Update` function:

[PRE28]

This method performs the following actions:

*   It defines a collision flag.
*   It iterates through all of the asteroids.
*   It checks to see whether the asteroid has collided with the ship.
*   If the asteroid has collided with the ship, we set `IsCollideable` for the asteroid to `false`. This stops the collision from occurring multiple times as the ship passes through the asteroid. For our game, we only need to collide with each asteroid once, so this is sufficient.

We need to wire the collision into the `Update` function. Add the following line to the Update method just after the `HandleInput` call:

[PRE29]

That's it. We have now implemented basic collision detection!

# Summary

We covered a lot of code in this chapter. You implemented a simple, yet effective framework to create and manage 3D objects in the game. This class included necessary features to load the model, position the model in 3D space, and check for collisions.

We also implemented input and collision detection in the game to create a modified slalom race, requiring you to navigate through each asteroid.

In the next chapter, we will implement a user interface and scoring system to make this a more complete game.