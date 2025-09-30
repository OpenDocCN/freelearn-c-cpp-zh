# Chapter 12. Conquer the Universe

Congratulations! You have come this far. If you are reading this chapter, then you have already created two games—a 2D game and a 3D game. Sure, they aren't going to sell and make you a million dollars, but you already completed more games than 90 percent of all people who try.

There is a lot more to learn, and there is no way that we can cover everything in a single book. This chapter will briefly cover a few more topics and hopefully give you at least enough information to experiment further after you are done with the book. In fact, we are going to set up a framework that will allow you to play, so we will call it the playground.

The topics that we will cover include the following:

*   **The playground**: We will begin by setting up a template that you can use over and over again as you experiment with different features. This template will also be a good starting ground for any future games that you may want to create.
*   **Texture mapping**: So far, we worked with color, not textures. It would be pretty difficult to make realistic games with only color. It turns out that we can put textures onto our models to make them more realistic. We will learn the basics of texture mapping on a simple 3D shape.
*   **Lighting**: So far, we used the default lighting that was provided by OpenGL. Most of the time, we want more control over the lighting. We will discuss the various types of lighting and how they are used.
*   **Skyboxes**: The game universe can't go on forever. We often use a device known as a skybox to surround our game world and make it look like it is larger than it really is. We will learn how to add a skybox to our space game.
*   **Physics**: In the real world, objects bounce, fall, and do other things based on the laws of physics. We will discuss how objects interact with each other and the rest of the universe.
*   **AI**: Many games have enemies or weapons seeking to destroy the player. These enemies are usually controlled by some form of **Artificial Intelligence** (**AI**). We will discuss some simple forms of AI and learn how the game can control objects in the game.
*   **Where to go from here**: Finally, I'll give you a few tips on how you can continue to improve your skills once you have completed this book. We'll talk about game engines and topics for additional study.

# A fun framework

Now, it's time to create our playground. Before we start coding, let's decide on the basic features that we want to set up:

*   Visual Studio project
*   Windows environment
*   OpenGL environment
*   Game loop

That's all we are going to do for now. The idea is to set up a basic template that you can use to start any game or experimental project. We don't want to include too much in this basic framework, so we will leave out sound, input, sprite, and model loading for now. These can be added in as they are needed.

## Setting up the Visual Studio project

Start a new blank project and name it `FunWith3D`. Make sure to add the correct libraries as we have done before in the project **Properties**, **Configuration Properties**, **Linker**, **Input**, **Additional Dependencies** property:

[PRE0]

We are going to include the SOIL library because it is so useful to load images. You will want to copy the following files over from our `SpaceRacer3D.cpp` project folder:

*   `glut.h`
*   `glut32.lib`
*   `glut32.dll`
*   `SOIL.h`
*   `SOIL.lib`

Add the following libraries to Properties, Configuration Properties, Input, and Additional Dependencies:

*   `glut32.lib`
*   `SOIL.lib`

## Setting up the Windows environment

Create a new C++ file and name it `FunWith3D.cpp`. Then add the following code:

[PRE1]

Now, open `SpaceRacer3D.cpp` from the previous project and copy the following functions:

*   `WinMain`
*   `WndProc`

These are the header files and two functions that are required for Windows to do its stuff. All of this code has been explained in previous chapters, so I'm not going to re-explain it here. In fact, you could save yourself some typing and copy this code directly from our previous project.

## Setting up the OpenGL environment

Now, it is time to set up OpenGL. Copy the following function from SpaceRacer3D and add them after the `WndProc` declaration:

*   `ReSizeGLScene`
*   `InitGL`
*   `KillGLWindow`
*   `CreateGLWindow`

## Setting up the game loop

Now, we add the function that defines our game loop. Add these functions after the OpenGL code that you just added:

[PRE2]

In order to be consistent with some other code that we have written, you need to add the following precompile directives in the project **Properties**, **Configuration Properties**, **C/C++**, **Preprocessor**, and **Preprocessor Definitions** property:

*   `_USE_MATH_DEFINES`
*   `_CRT_SECURE_NO_WARNINGS`

Congratulations! You now have a framework that you can use for any future projects and experiments. You have also just successfully reviewed the OpenGL and game code that we have been working with throughout the entire book.

You will notice that I also left the code in so that you will be able render in either 3D or 2D! All together, you now have a small yet effective start for your own game engine. I suggest that you save a copy of the folder containing this solution and project. Then, when you are ready to start a new project, you can simply copy the solution folder, give it another name, and you are ready to go. We are going to use this as the basis for any code that we write in this chapter.

### Tip

To save space and keep our little playground simple, I did not include some key features, such as input, sprites, models, and sound. If you feel that any of these are essential to your playground, then this will be your first exercise. In general, you will have to simply copy the relevant files and/or code into your project folder from the last version of SpaceRacer3D.

# Texture mapping

Until now, all of our shapes and models used color, but a whole new world awaits us when we start applying textures to our models. Adding a 2D texture to a 3D model is known as **texture mapping**, or in some cases **texture wrapping**. Let's see what it takes to add a little texture to our 3D models. We are going to start with a simple cube.

First, use your favorite image editing software to create a 256 x 256 pixel square and give it some kind of texture. I will be using the following one:

![Texture mapping](img/8199OS_12_01.jpg)

Save this texture as a bitmap (BMP). We are going to use bitmaps, as opposed to PNGs, for texture mapping because the internal data structure of a bitmap happens to coincide with the data structure that is expected by OpenGL. In other words, it is easier!

I always create a folder called resources for my images. It is also a good idea to include these as resources in the Visual Studio project (right-click on the **Resources** folder in the **Solution Explorer** panel and choose **Add Existing…**, then navigate to the image).

## Loading the texture

If you recall, we created a sprite class for our previous projects. We use the `AddTexture` method of the `Sprite` class to make a call to the SOIL library to load the image. We won't be using the `Sprite` class for these textures. The `Sprite` class has a lot of methods and properties that don't apply to texturing 3D models, so we are going to write our own texture loader for this use. Add the following code somewhere above render functions:

[PRE3]

The purpose of `LoadTexture` is to load a texture into memory, and then set it up to be a texture map for a 3D object. In order to accomplish this, we actually need to load the texture twice. First, we directly open the file and read it as a binary file into a buffer called `data`. We use the `char` datatype because we want to store the binary data as unsigned integers and `char` does a really great job of this. So, our first few lines of code:

*   Define the data array
*   Create a file handle
*   Allocate memory for the data
*   Read the file into the data buffer
*   Close the file (but not the buffer)

Now, read the image a second time, though this time we use the SOIL library to read it as an OpenGL texture and use SOIL to load the texture and assign it to the OpenGL referenced by `texture`.

Then, we perform some fancy OpenGL operations on it to set it up as a model texture:

*   `GL_BindTexture` simply tells OpenGL that we want this texture to be the current texture, to which we will apply the settings that follow.
*   `glTexImage2D` tells OpenGL how to interpret the data that we have read in. We are telling OpenGL to treat the data as a 2D texture of the type RGB or RGBA (controlled by the `colordepth` parameter), and that the data is stored as unsigned integers (thus, the `char` data type).
*   The next two functions, both calls to `glTexParameteri`, tell OpenGL how to handle the texture as it gets nearer to or farther away from the camera. They are both set up to use linear filtering to handle this level of detail.
*   Finally, we close the data buffer as it is no longer needed.

We have set the `LoadTexture` function up so that you can call it for different textures based on your needs. In our case, we are first going to set up a handle to this texture. At the top of the code, add this line to the global variables section:

[PRE4]

Next, we will place the call to load the texture in the `StartGame` function:

[PRE5]

This call tells the program:

*   The location of the file
*   The width and height of the image
*   The color depth of the image (in this case `4` = RGBA)
*   The OpenGL texture handle

## Rendering the cube

We are all set up now with a texture, but we need a model to texture. To keep things simple, we are going to use quads to create a cube and apply the marble texture to each face of the cube.

Just before we get started, we need to add three variables to track rotation. Add these lines to the global variables section:

[PRE6]

Now, create the following function just below the `LoadTexture` function:

[PRE7]

This code is very similar to the code that we used to draw a cube in a previous chapter. However, when we drew that cube, we applied color to each vertex. Now, we will apply our texture to each face. First, we set things up:

1.  The first thing that we do is use `glEnable(GL_TEXTURE_2D)` to enable 2D textures. In our initial setup, we disabled 2D textures. If we did not enable them here, then our texture would not show up!
2.  Next, we use `glLoadIdentity()` to initialize the current matrix.
3.  We call `glTranslatef(0.0f, 0.0f, -5.0f)` to move the camera back (so that we will be outside the cube).
4.  Three calls to `glRotate3f` will rotate the cube for us.
5.  Then, we use `glBindTexture(GL_TEXTURE_2D, texMarble)` to inform OpenGL that for the next draw operations we will be using the texture referenced by `texMarble`.

With this setup completed, we are ready to get drawing:

1.  We start with `glBegin(GL_QUADS)` to tell OpenGL that we will be drawing quads.
2.  Now, each call comes in a pair. First a call to `glTexCoord2f` is followed by a call to `glVertex3f`. The call to `glTexCoord2f` tells OpenGL which part of the texture to put at the location specified by `glVertex3f`. In this way, we can map any point in the texture to any point in the quad. OpenGL takes care of figuring out which parts of the texture go between vertices.
3.  When we are done drawing the cube, we issue the `glEnd()` command.
4.  The last three lines update the rotation variables.
5.  Finally, we have to make a call to `DrawTexturedCube` in the `Render3D` function:

    [PRE8]

6.  Run the program and see the cube in its textured glory!![Rendering the cube](img/8199OS_12_02.jpg)

## Mapping operations

I owe you a little more explanation as to how texture mapping works. Take a look at these four lines of code from `DrawTexturedCube`:

[PRE9]

These four lines define one quad. Each vertex consists of a texture coordinate (`glTexCoord2f`) and a vertex coordinate (`glVertex3f`). When OpenGL looks at a texture, here is what it sees:

![Mapping operations](img/8199OS_12_03.jpg)

No matter how big a texture is in pixels, in texture coordinates, the texture is exactly one unit wide and one unit tall. So, the first line of the preceding code tells OpenGL to take the point (**0,0**) of the texture (the upper-left corner) and map it to the next vertex that is defined (which is the upper-left hand corner of the quad, in this example). You will notice that the third line maps the coordinate (**1,1**) of the texture to the lower-right corner of the quad. In effect, we are stretching the texture across the face of the quad! However, OpenGL also adapts the mapping so that the texture doesn't look smeared, so this isn't exactly what happens. Instead, you will see some tiling in our case.

# Let there be light!

Until this point, we haven't worried about lighting. In fact, we just assumed that light would be there so that we could see our images. OpenGL has a light setting that lights everything equally. This setting is turned on, by default, until we tell OpenGL that we would like to handle the lighting.

Imagine what our scene would look like if there was no lighting. In fact, this is going to happen to you some day. You will have everything set up and ready to roll, you'll run the program, and you'll get a big, black, nothing! What's wrong? You forgot to turn on the lights! Just as shown in the following image:

![Let there be light!](img/8199OS_12_04.jpg)

Just like real life, if you don't have a source of light, you aren't going to see anything. OpenGL has many types of lights. One common light is **ambient** light. Ambient light appears to come from all directions at the same time, similarly to how sunlight fills up a room.

![Let there be light!](img/8199OS_12_05.jpg)

Lighting is very important in 3D games, and most games have multiple light sources to add realism to the game.

## Defining a light source

Let's take over and define our own light source. Add the following lines of code to the top of the `DrawTexturedCube` function:

[PRE10]

Run the program, then come back to see what is happening:

*   `glEnable(GL_LIGHTING)` tells OpenGL that we want to take control of the lighting now. Remember: once you enable lighting, it's up to you. In fact, if you enable lighting and don't define any lights, then you will get a completely black scene.
*   Next, we define a color for our light. In this case, we are creating a blue light.
*   Now we tell OpenGL what type of lighting we would like to use with `glLightModelfv`. In this case, we are turning on a blue, ambient light.
*   Light has to have a material to reflect from. So, we use `glEnable(GL_COLOR_MATERIAL)` to tell OpenGL to use a material that will reflect color.
*   The call to `glColorMaterial(GL_FRONT, GL_AMBIENT)` tells OpenGL that the front of this material should reflect light as if it was ambient light. Remember, ambient light comes from all directions.

Of course, you have already seen the result. Our cube is blue! Play around with different colors. We only have time to barely scratch the surface on lighting. You will also want to learn about diffuse lighting. Diffuse lights fade with distance. With a diffuse light, you not only set up the color, but you also place the light at a certain location.

# The skybox

While space may be infinite, your computer isn't so there has to be a boundary somewhere. This boundary is called the skybox.

Imagine that our spaceship is flying through space! Space is big. While we may put some planets and asteroids in our universe to give the space ship something to interact with, we certainly won't model every star. Here is what our universe looks like:

![The skybox](img/8199OS_12_06.jpg)

This is pretty empty, right? You probably already noticed this in our game, SpaceRacer3D. Of course, we could add some more objects of our own—more asteroids, add a bunch of stars—and in a real game, we would. But, there is always a limit to how many objects you can add to the game before you start having performance issues.

For the really distant objects, such as distant stars, we fake it by using 2D textures. For example, our game could use a texture of stars to imitate the stars and nebula in space, as shown in the following image:

![The skybox](img/8199OS_12_07.jpg)

Now, as a cube has six sides, what we really want is six textures. A typical skybox looks similar to the following image:

![The skybox](img/8199OS_12_08.jpg)

It doesn't take too much imagination to see how this texture can be wrapped around the cube and cover all size sides. This creates an image that covers all of the space encapsulated by the skybox and gives the illusion of being surrounded by stars and nebula, as shown in the following image:

![The skybox](img/8199OS_12_09.jpg)

The following illustration shows the skybox in relation to the texture that will be applied to it from another perspective:

![The skybox](img/8199OS_12_10.jpg)

The cube containing the ship and asteroid represents the game world. The ship and asteroid are real objects in that world. The image on the left is a texture that contains the stars.

Now, imagine the star texture being wrapped around the cube, and there is your whole universe composed of the stars, the ship, and the asteroid. The star texture wrapped around the cube is the skybox.

# Advanced topics

Unfortunately, for the last two topics, we only have time to give them an honorable mention. I included them because you are going to hear about these topics, and you need to know what these terms mean.

## Game physics

**Game physics** are the rules that define how objects interact with other objects inside the game universe. For example, in SpaceRacer3D, the ship simply passes through the asteroids. However, there could be many other outcomes:

*   The ship and asteroid could bounce off of each other (rebound)
*   The ship could be sucked into the asteroid with the force increasing as the ship got closer (gravity)
*   The asteroid could push against the ship the closer the ship got to it (reverse gravity)

Each of these effects would be programmed into the game. Each of these effects would also create a different kind of gameplay. An entire genre of games known as physics-based games simply define the laws of physics for a game universe and then let things interact to see what will happen.

## AI

**AI**, or **artificial intelligence**, is another set of rules that defines how characters or objects that are controlled by the compute behave. AI is typically applied to enemies and other **Non-player Characters** (**NPCs**) to give them a life-like appearance in the game. Some examples of AI include:

*   A mine that automatically detects that the enemy is close and blows up
*   A homing missile that locks onto a space ship and draws closer no matter how the ship navigates
*   An enemy character who detects that the player coming and hides behind a rock

AI is typically considered one of the most difficult areas of game programming. Some algorithms are quite easy (for example, the homing missile only needs the ships position to know how to track it), while others are very complex (for example, hiding behind a rock). Some games even provide an AI opponent for you to play against.

# The future

You have, indeed, come a long way. If you are reading these words, and especially if you wrote all of the code along the way, then you have achieved a great accomplishment, but there is still so much to learn. I encourage you to find other books and never stop learning. The only thing that will stop you from becoming a great game programmer is you!

# Summary

As always, we covered a lot of topics in this chapter. You learned how to map a texture onto an object, then you learned how to turn the lights on. You learned how a skybox can be used to make your world seem larger than it is. And you got just a taste of physics and AI, topics which could easily fill entire books on their own. Don't stop until you have got every piece of code in this book to work for you, and then start changing the code to different and amazing things.

Good luck!