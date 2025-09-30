# Chapter 4. Playing with Physics

In the previous chapter, we built several games, including a Tetris clone. In this chapter, we will add physics into this game and turn it into a new one. By doing this, we will learn:

*   What is a physics engine
*   How to install and use the Box2D library
*   How to pair the physics engine with SFML for the display
*   How to add physics in the game

In this chapter, we will learn the magic of physics. We will also do some mathematics but relax, it's for conversion only. Now, let's go!

# A physics engine – késako?

In this chapter, we will speak about physics engine, but the first question is "what is a physics engine?" so let's explain it.

A physics engine is a software or library that is able to simulate Physics, for example, the Newton-Euler equation that describes the movement of a rigid body. A physics engine is also able to manage collisions, and some of them can deal with soft bodies and even fluids.

There are different kinds of physics engines, mainly categorized into real-time engine and non-real-time engine. The first one is mostly used in video games or simulators and the second one is used in high performance scientific simulation, in the conception of special effects in cinema and animations.

As our goal is to use the engine in a video game, let's focus on real-time-based engine. Here again, there are two important types of engines. The first one is for 2D and the other for 3D. Of course you can use a 3D engine in a 2D world, but it's preferable to use a 2D engine for an optimization purpose. There are plenty of engines, but not all of them are open source.

## 3D physics engines

For 3D games, I advise you to use the `Bullet` physics library. This was integrated in the Blender software, and was used in the creation of some commercial games and also in the making of films. This is a really good engine written in C/C++ that can deal with rigid and soft bodies, fluids, collisions, forces… and all that you need.

## 2D physics engines

As previously said, in a 2D environment, you can use a 3D physics engine; you just have to ignore the depth (Z axes). However, the most interesting thing is to use an engine optimized for the 2D environment. There are several engines like this one and the most famous ones are Box2D and Chipmunk. Both of them are really good and none of them are better than the other, but I had to make a choice, which was Box2D. I've made this choice not only because of its C++ API that allows you to use overload, but also because of the big community involved in the project.

# Physics engine comparing game engine

Do not mistake a physics engine for a game engine. A physics engine only simulates a physical world without anything else. There are no graphics, no logics, only physics simulation. On the contrary, a game engine, most of the time includes a physics engine paired with a render technology (such as OpenGL or DirectX). Some predefined logics depend on the goal of the engine (RPG, FPS, and so on) and sometimes artificial intelligence. So as you can see, a game engine is more complete than a physics engine. The two mostly known engines are Unity and Unreal engine, which are both very complete. Moreover, they are free for non-commercial usage.

So why don't we directly use a game engine? This is a good question. Sometimes, it's better to use something that is already made, instead of reinventing it. However, do we really need all the functionalities of a game engine for this project? More importantly, what do we need it for? Let's see the following:

*   A graphic output
*   Physics engine that can manage collision

Nothing else is required. So as you can see, using a game engine for this project would be like killing a fly with a bazooka. I hope that you have understood the aim of a physics engine, the differences between a game and physics engine, and the reason for the choices made for the project described in this chapter.

# Using Box2D

As previously said, Box2D is a physics engine. It has a lot of features, but the most important for the project are the following (taken from the Box2D documentation):

*   **Collision**: This functionality is very interesting as it allows our tetrimino to interact with each other

    *   Continuous collision detection
    *   Rigid bodies (convex polygons and circles)
    *   Multiple shapes per body

*   **Physics**: This functionality will allow a piece to fall down and more

    *   Continuous physics with the time of impact solver
    *   Joint limits, motors, and friction
    *   Fairly accurate reaction forces/impulses

As you can see, Box2D provides all that we need in order to build our game. There are a lot of other features usable with this engine, but they don't interest us right now so I will not describe them in detail. However, if you are interested, you can take a look at the official website for more details on the Box2D features ([http://box2d.org/about/](http://box2d.org/about/)).

It's important to note that Box2D uses meters, kilograms, seconds, and radians for the angle as units; SFML uses pixels, seconds, and degrees. So we will need to make some conversions. I will come back to this later.

## Preparing Box2D

Now that Box2D is introduced, let's install it. You will find the list of available versions on the Google code project page at [https://code.google.com/p/box2d/downloads/list](https://code.google.com/p/box2d/downloads/list). Currently, the latest stable version is 2.3\. Once you have downloaded the source code (from compressed file or using SVN), you will need to build it.

### Build

Here is the good news, Box2D uses CMake as build process so you just have to follow the exact same steps as the SFML build described in the first chapter of this book and you will successfully build Box2D. If everything is fine, you will find the example project at this place: `path/to/Box2D/build/Testbed/Testbed`. Now, let's install it.

### Install

Once you have successfully built your Box2D library, you will need to configure your system or IDE to find the Box2D library and headers. The newly built library can be found in the `/path/to/Box2D/build/Box2D/` directory and is named `libBox2D.a`. On the other hand, the headers are located in the `path/to/Box2D/Box2D/` directory. If everything is okay, you will find a `Box2D.h` file in the folder.

On Linux, the following command adds Box2D to your system without requiring any configuration:

[PRE0]

# Pairing Box2D and SFML

Now that Box2D is installed and your system is configured to find it, let's build the physics "hello world": a falling square.

It's important to note that Box2D uses meters, kilograms, seconds, and radian for angle as units; SFML uses pixels, seconds, and degrees. So we will need to make some conversions.

Converting radians to degrees or vice versa is not difficult, but pixels to meters… this is another story. In fact, there is no way to convert a pixel to meter, unless if the number of pixels per meter is fixed. This is the technique that we will use.

So let's start by creating some utility functions. We should be able to convert radians to degrees, degrees to radians, meters to pixels, and finally pixels to meters. We will also need to fix the pixel per meter value. As we don't need any class for these functions, we will define them in a namespace converter. This will result as the following code snippet:

[PRE1]

As you can see, there is no difficulty here. We start to define some constants and then the convert functions. I've chosen to make the function template to allow the use of any number type. In practice, it will mostly be `double` or `int`. The conversion functions are also declared as `constexpr` to allow the compiler to calculate the value at compile time if it's possible (for example, with constant as a parameter). It's interesting because we will use this primitive a lot.

## Box2D, how does it work?

Now that we can convert SFML unit to Box2D unit and vice versa, we can pair Box2D with SFML. But first, how exactly does Box2D work?

Box2D works a lot like a physics engine:

1.  You start by creating an empty world with some gravity.
2.  Then, you create some object patterns. Each pattern contains the shape of the object position, its type (static or dynamic), and some other characteristics such as its density, friction, and energy restitution.
3.  You ask the world to create a new object defined by the pattern.
4.  In each game loop, you have to update the physical world with a small step such as our world in the games we've already made.

Because the physics engine does not display anything on the screen, we will need to loop all the objects and display them by ourselves.

Let's start by creating a simple scene with two kinds of objects: a ground and square. The ground will be fixed and the squares will not. The square will be generated by a user event: mouse click.

This project is very simple, but the goal is to show you how to use Box2D and SFML together with a simple case study. A more complex one will come later.

We will need three functionalities for this small project to:

*   Create a shape
*   Display the world
*   Update/fill the world

Of course there is also the initialization of the world and window. Let's start with the main function:

1.  As always, we create a window for the display and we limit the FPS number to 60\. I will come back to this point with the `displayWorld` function.
2.  We create the physical world from Box2D, with gravity as a parameter.
3.  We create a container that will store all the physical objects for the memory clean purpose.
4.  We create the ground by calling the `createBox` function (explained just after).
5.  Now it is time for the minimalist `game` loop:

    *   Close event managements
    *   Create a box by detecting that the right button of the mouse is pressed

6.  Finally, we clean the memory before exiting the program:

    [PRE2]

For the moment, except the Box2D world, nothing should surprise you so let's continue with the box creation.

This function is under the `book` namespace.

[PRE3]

This function contains a lot of new functionalities. Its goal is to create a rectangle of a specific size at a predefined position. The type of this rectangle is also set by the user (dynamic or static). Here again, let's explain the function step-by-step:

1.  We create `b2BodyDef`. This object contains the definition of the body to create. So we set the position and its type. This position will be in relation to the gravity center of the object.
2.  Then, we create `b2Shape`. This is the physical shape of the object, in our case, a box. Note that the `SetAsBox()` method doesn't take the same parameter as `sf::RectangleShape`. The parameters are half the size of the box. This is why we need to divide the values by two.
3.  We create `b2FixtureDef` and initialize it. This object holds all the physical characteristics of the object such as its density, friction, restitution, and shape.
4.  Then, we properly create the object in the physical world.
5.  Now, we create the display of the object. This will be more familiar because we will only use SFML. We create a rectangle and set its position, origin, and color.
6.  As we need to associate and display SFML object to the physical object, we use a functionality of Box2D: the `SetUserData()` function. This function takes `void*` as a parameter and internally holds it. So we use it to keep track of our SFML shape.
7.  Finally, the body is returned by the function. This pointer has to be stored to clean the memory later. This is the reason for the body's container in `main()`.

Now, we have the capability to simply create a box and add it to the world. Now, let's render it to the screen. This is the goal of the `displayWorld` function:

[PRE4]

This function takes the physics world and window as a parameter. Here again, let's explain this function step-by-step:

1.  We update the physical world. If you remember, we have set the frame rate to 60\. This is why we use 1,0/60 as a parameter here. The two others are for precision only. In a good code, the time step should not be hardcoded as here. We have to use a clock to be sure that the value will always be the same. Here, it has not been the case to focus on the important part: physics. And more importantly, the physics loop should be different from the display loop as already said in [Chapter 2](ch02.html "Chapter 2. General Game Architecture, User Inputs, and Resource Management"), *General Game Architecture, User Inputs, and Resource Management*. I will come back to this point in the next section.
2.  We reset the screen, as usual.
3.  Here is the new part: we loop the body stored by the world and get back the SFML shape. We update the SFML shape with the information taken from the physical body and then render it on the screen.
4.  Finally, we render the result on the screen.

That's it. The final result should look like the following screenshot:

![Box2D, how does it work?](img/8477OS_04_01.jpg)

As you can see, it's not really difficult to pair SFML with Box2D. It's not a pain to add it. However, we have to take care of the data conversion. This is the real trap. Pay attention to the precision required (`int`, `float`, `double`) and everything should be fine.

Now that you have all the keys in hand, let's build a real game with physics.

# Adding physics to a game

Now that Box2D is introduced with a basic project, let's focus on the real one. We will modify our basic Tetris to get Gravity-Tetris alias Gravitris. The game control will be the same as in Tetris, but the game engine will not be. We will replace the board with a real physical engine.

With this project, we will reuse a lot of work previously done. As already said, the goal of some of our classes is to be reusable in any game using SFML. Here, this will be made without any difficulties as you will see. The classes concerned are those you deal with user event `Action`, `ActionMap`, `ActionTarget`—but also `Configuration` and `ResourceManager`. Because all these classes have already been explained in detail in the previous chapters, I will not waste time to explain them again in this one.

There are still some changes that will occur in the `Configuration` class, more precisely, in the enums and `initialization` methods of this class because we don't use the exact same sounds and events that were used in the Asteroid game. So we need to adjust them to our needs.

Enough with explanations, let's do it with the following code:

[PRE5]

As you can see, the changes are in the `enum`, more precisely in `Sounds` and `PlayerInputs`. We change the values into more adapted ones to this project. We still have the font and music theme. Now, take a look at the initialization methods that have changed:

[PRE6]

No real surprises here. We simply adjust the resources to our needs for the project. As you can see, the changes are really minimalistic and easily done. This is the aim of all reusable modules or classes. Here is a piece of advice, however: keep your code as modular as possible, this will allow you to change a part very easily and also to import any generic part of your project to another one easily.

## The Piece class

Now that we have the configuration class done, the next step is the `Piece` class. This class will be the most modified one. Actually, as there is too much change involved, let's build it from scratch. A piece has to be considered as an ensemble of four squares that are independent from one another. This will allow us to split a piece at runtime. Each of these squares will be a different fixture attached to the same body, the piece.

We will also need to add some force to a piece, especially to the current piece, which is controlled by the player. These forces can move the piece horizontally or can rotate it.

Finally, we will need to draw the piece on the screen.

The result will show the following code snippet:

[PRE7]

Some parts of the class don't change such as the `TetriminoTypes` and `TetriminoColors` enums. This is normal because we don't change any piece's shape or colors. The rest is still the same.

The implementation of the class, on the other side, is very different from the precedent version. Let's see it:

[PRE8]

The constructor is the most important method of this class. It initializes the physical body and adds each square to it by calling `createPart()`. Then, we set the user data to the piece itself. This will allow us to navigate through the physics to SFML and vice versa. Finally, we synchronize the physical object to the drawable by calling the `update()` function:

[PRE9]

The destructor loop on all the fixtures attached to the body, destroys all the SFML shapes and then removes the body from the world:

[PRE10]

This method adds a square to the body at a specific place. It starts by creating a physical shape as the desired box and then adds this to the body. It also creates the SFML square that will be used for the display, and it will attach this as user data to the fixture. We don't set the initial position because the constructor will do it.

[PRE11]

This method synchronizes the position and rotation of all the SFML shapes from the physical position and rotation calculated by Box2D. Because each piece is composed of several parts—fixture—we need to iterate through them and update them one by one.

[PRE12]

These two methods add some force to the object to move or rotate it. We forward the job to the Box2D library.

[PRE13]

This function draws the entire piece. However, because the piece is composed of several parts, we need to iterate on them and draw them one by one in order to display the entire piece. This is done by using the user data saved in the fixtures.

## The World class

Now that we have built our pieces, let's make a world that will be populated by them. This class will be very similar to the one previously made in the Tetris clone. But now, the game is based on physics. So we need to separate the physics and the display updates. To do this, two `update` methods will be used.

The big change is that the board is no longer a grid, but a physical world. Because of this, a lot of internal logic will be changed. Now, let's see it:

[PRE14]

We make the class non-replicable, with size as a parameter. As you can see, there are now two `update` methods. One for the physics and another one for the SFML objects. We still have some methods specific for the game such as `newPiece()`, `clearLines()`, `isGameOver()`, a new one relative to the `updateGravity()` physic, and a method to add sounds to our world. This method directly comes from the Meteor game by copying and pasting it.

Now that the class is introduced, take a look at its implementation. The following constructor initializes the physical world with a default gravity and adds some walls to it:

[PRE15]

The destructor removes all the SFML shapes attached to the bodies still present in the world:

[PRE16]

The following method synchronizes the physical bodies with the SFML objects that display it. It also removes all the sounds effects that are finished, as already explained in the previous chapter:

[PRE17]

Now, we construct a class inside the `World.cpp` file because we don't need the class anywhere else. This class will be used to query the physical world by getting all the fixtures inside an area. This will be used more, especially to detect the completed lines:

[PRE18]

The following method clears the completed lines by querying the world, especially with the made class. Then, we count the number of fixtures (squares) on each line; if this number satisfies our criteria, we delete all the fixtures and the line. However, by doing this, we could have some bodies with no fixture. So, if we remove the last fixture attached to a body, we also remove the body. Of course, we also remove all the SFML shapes corresponding to those deleted objects. Finally, for more fun, we add some sounds to the world if needed:

[PRE19]

The following function sets the gravity depending on the current level. Bigger the level, stronger is the gravity:

[PRE20]

The following function is directly taken from the Asteroid clone, and was already explained. It just adds sound to our world:

[PRE21]

This method checks if the game is over with a simple criterion, "are there any bodies out of the board?":

[PRE22]

This function updates only the physical world by forwarding the job to Box2D:

[PRE23]

Now, we create a piece and set its initial position to the top of our board. We also add a sound to alert the player about this:

[PRE24]

The `draw()` function is pretty simple. We iterate on all the bodies still alive in the world and display the SFML object attached to them:

[PRE25]

The following functions are helpful. Its aim is to create a static body that will represent a wall. All the functionalities used were already explained in the first part of this chapter, so nothing should surprise you:

[PRE26]

## The Game class

Now, we have a world that can be populated by some pieces, let's build the last important class—the `Game` class.

There is a big change in this class. If you remember, in [Chapter 2](ch02.html "Chapter 2. General Game Architecture, User Inputs, and Resource Management"), *General Game Architecture, User Inputs, and Resource Management*, I said that a game with physics should use two game loops instead of one. The reason for this is that most of the physical engine works well with a fixed time step. Moreover, this can avoid a really bad thing. Imagine that your physical engine takes 0.01 second to compute the new position of all the bodies in your world, but the delta time passed as argument to your `update` function is fewer. The result will be that your game will enter in a death state and will finally freeze.

The solution is to separate the physics from the rendering. Here, the physics will run at 60 FPS and the game at a minimum of 30 FPS. The solution presented here is not perfect because we don't separate the computation in different threads, but this will be done later, in the sixth chapter.

Take a look at the `Game` header file:

[PRE27]

No surprises here. The usual methods are present. We just duplicate the `update` function, one for logic and the other for physics.

Now, let's see the implementation. The constructor initializes `World` and binds the player inputs. It also creates the initial piece that will fall on the board:

[PRE28]

The following function has nothing new except that the two `update()` functions are called instead of one:

[PRE29]

The following function updates the logic of our game:

[PRE30]

Here is the step-by-step evaluation of the preceding code:

1.  We start by updating some time value by adding the `deltaTime` parameter to them.
2.  Then, we apply some forces to the current piece if needed.
3.  We update the world by cleaning all the complete lines and also update the score.
4.  If needed, we create a new piece that will replace the current one.

Now, take a look at the physics:

[PRE31]

This function updates all the physics, including the gravity that changes with the current level. Here again, nothing is too complicated.

The `processEvents()` and `render()` functions don't change at all, and are exactly the same as in the first Tetris.

As you can see, the `Game` class doesn't change a lot and is very similar to the one previously made. The two loops—logics and physics—are the only real changes that occur.

## The Stats class

Now, the last thing to build is the `Stats` class. However, we have already made it in the previous version of Tetris, so just copy and paste it. A little change has been made for the game over, by adding a getter and setter. That's it.

Now, you have all the keys in hand to build your new Tetris with sounds and gravity. The final result should look like the following screenshot:

![The Stats class](img/8477OS_04_02.jpg)

# Summary

Since the usage of a physics engine has its own particularities such as the units and game loop, we have learned how to deal with them. Finally, we learned how to pair Box2D with SFML, integrate our fresh knowledge to our existing Tetris project, and build a new funny game.

In the next chapter, we will learn how to add a user interface to our game in order to interact with the user easily, by creating our own game user interface or by using an existing one.