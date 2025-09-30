# Chapter 11. Heads Up

In this chapter, we will put some finishing touches on Space Racer 3D by adding some features that you would see in almost any game. Many of these features are similar to the finishing touches that we put on our Robo Racer 2D game, though there are some special considerations now that we are working in 3D. The topics that we will cover include the following:

*   **2D in a 3D world**: So far, we learned how to render in 2D and how to render in 3D. However, there are special considerations to create 2D in a 3D world. As our user interface is typically created in 2D, we will learn how to mix the two types of rendering.
*   **Creating a heads-up-display (HUD)**: It is very typical for first-person 3D games to have a continuous status showing information that is relevant to the game. We will learn how to create a basic heads-up-display or HUD.
*   **More game state**: Just as we did in Robo Racer 2D, we will create a basic state manager to handle the various modes in our completed game.
*   **Scoring**: We need a way to keep score in our game, and we need to set up the basic win and lose conditions.
*   **Game over**: When the game is over, we'll give some credit with a 3D twist.

# Mixing things up

Now that we are rendering in 3D, it isn't immediately obvious how we will render things in 2D. This is especially true of our user interface, which must be rendered on top of the 3D-scene and does not move or rotate with the rest of the world.

The trick to creating a 2D interface in a 3D world is to first render the 3D world, then switch modes in OpenGL, and then render the 2D content. The following image represents the 3D content that we need to render:

![Mixing things up](img/8199OS_11_01.jpg)

The next image represents the 2D text that we want to render:

![Mixing things up](img/8199OS_11_02.jpg)

We want the final result to be the combination of the 3D and 2D content, as shown in the following figure:

![Mixing things up](img/8199OS_11_03.jpg)

## The saving state

State is a term that is used in many different ways in game programming. For example, we will create a state manager later in the chapter that will manage different states, or modes, in the game. Another way to define state is a set of conditions. For example, when we set things up to render in 3D, this is one set of conditions or state. When we set up things to render in 2D, this is another set of conditions or state.

The trick to being able to render in both 2D and 3D is to be able to set up one state, and then change to another state. OpenGL saves state in matrices. In order to change from one state to another, we need a way to save the current matrix, set up another matrix, and then return to the previous matrix once we are done.

## Push and pop

OpenGL provides two methods to save the current state and then retrieve it later:

*   `glPushMarix()`: This command saves the current state by placing it on the stack.
*   `glPopMatrix()`: This command retrieves the previous state by pulling it off the stack.

A **stack** is a structure that allows you to put data on the top of it (a **push**), and then later retrieve the item from the top of it (a **pop**). A stack is useful when you want to save data in order, then later retrieve it in reverse order.

Let's say that we start with an initial set of conditions called **State A**:

![Push and pop](img/8199OS_11_04.jpg)

A call to `glPushMatrix()` will put **State A** on the stack:

![Push and pop](img/8199OS_11_05.jpg)

Next, we set up the conditions for **State B**. If we want to save this state, we issue another `glPushMatrix()` call:

![Push and pop](img/8199OS_11_06.jpg)

Now we have two items on the stack, and it should also be very clear why it is called a stack! We could then define **State C**. This sequence of steps can continue on as needed, creating a render state and then pushing it to the stack. In general, we want to unload the stack in the reverse order that we loaded it in. This is known as a **FILO** stack: first in, last out.

We take things off of the stack with the `glPopMatrix()` command:

![Push and pop](img/8199OS_11_07.jpg)

The result replaces **State C**, restoring the rendering settings to **State B**:

![Push and pop](img/8199OS_11_08.jpg)

Another call to `glPopMatrix()` empties the stack and restores the rendering settings to **State A**:

![Push and pop](img/8199OS_11_09.jpg)

The model view allows 32 matrices to be put onto the stack. Each view has its own stack, so the projection view has a separate stack from the model view. Also, if you issue `glPopMatrix` and there is no matrix on the stack, you will receive an error. In other words, don't try to pop more than you have pushed!

### Tip

In order to best manage memory, you should always pop the states that you have pushed, even if you don't need to do anything with them. This frees up the memory that was being used to hold the data that was part of the state that you were saving.

## Two state rendering

We are now going to set up our code to be able to render in both 3D and 2D. Open `SpaceRacer3D.cpp`. We are going to split up the rendering into two functions: `Render3D`, and `Render2D`. Then, we are going to call these from the main `Render` function. Let's start with `Render3D`. Add the following code just above the `Render` function (you can just cut it from the `Render` function):

[PRE0]

Next, we will create two support functions to turn 2D rendering on and off. The first will be `Enable2D`. Add the following function above the `Render3D` function:

[PRE1]

`Enable2D` performs the tasks that are necessary to change the rendering mode to 2D:

*   The call to `glColor3f` sets the current drawing color to white. This takes some explanation. We will always render 3D first, then switch to 2D. If we didn't set the color to white, then all of the colors in the 2D content would be blended with the last color that was used by the 3D rendering. Setting the render color to white essentially clears the render color so that the 2D content will be rendered accurately. Setting the color to white doesn't actually mean everything will be drawn in white. It means that no additional coloring will be added to the objects that we render in 2D.
*   The `glEnable(GL_TEXTURE_2D)` call is essential if you want to render 2D textures. If this were left out, then any 2D textures would not render correctly.
*   The next four lines save the 3D projection matrix and set up the projection matrix to render in 2D. `glPushMatrix` pushes the current projection matrix to the stack. We then initialize the projection matrix with `glLoadIdentity`. Finally, we set up an orthographic projection with the call to `glOrtho`. Take a look at RoboRacer2D, and you will notice that it uses the same `glOrtho` call to set up 2D rendering!
*   The next three lines save the 3D model view matrix and initialize it for our 2D drawing. `glPushMatrix` pushes the current model view matrix to the stack. We then initialize the model view matrix with the call to `glLoadIdentity`.
*   Finally, we need to turn off checking on the depth buffer. The depth buffer check is only required for 3D rendering, and interferes with 2D rendering. `glPushAttrib` works just like `glPushMatrix`, except that it only pushes a single OpenGL attribute to the stack. In this case, we are pushing the current `GL_DEPTH_BUFFER_BIT` to the attribute stack, thus saving the current state of this bit from the previous 3D rendering. Next, we turn off depth checking with the `glDisable` call.

So, setting things up for 2D rendering involves four steps:

1.  Reset the render color and enable 2D textures.
2.  Save the 3D project matrix and set up the 2D projection matrix.
3.  Save the 3D model view matrix and initialize the 2D model view matrix.
4.  Save the 3D depth bit and turn off depth checking in 2D.

Now, we are ready to code the `Disable2D` function. Create this new function just below the `Enable2D` function that we just created:

[PRE2]

It shouldn't be too surprising that `Disable2D` performs actions in the reverse order that we performed them in `Enable2D`:

*   First, we restore depth checking by calling `glPopAttrib()`, which takes the last attribute that was pushed to the attribute stack off the stack and restores that attribute in the current render state. This will restore depth checking to the state that it was in just before we started our 2D rendering.
*   The next two lines restore the projection matrix to the 3D state it was in. Again, the call to `glPopMatrix` takes the item on the top of the stack and applies it to the current render state.
*   The next two lines pop the model view matrix.
*   The final line disables 2D textures.

Now, it is time to create our `Render2D` function. Add the following code just above the `Render3D` function:

[PRE3]

The funny thing is that we don't have any 2D content to render yet! Later in the chapter, we will fill in the rest of the content of this function. The important thing to note here is that this function will take care of enabling 2D rendering with the call to `Enable2D`. Then the code will be added to render our 2D content. Finally, we will turn off 2D rendering with the call to `Disable2D`.

Now that we have all of the necessary supporting code to render in 2D and 3D, we will modify the `Render` function:

[PRE4]

You will notice how simple this is now:

1.  First, we clear the color buffer and reset the matrix. We always do this once before each frame of rendering.
2.  Next, we render the 3D content.
3.  Then we render the 2D content.
4.  Finally, we swap the buffers, which renders all of our content to the screen.

If you run the game now, you should notice that nothing has changed. As we haven't created any 2D content to render, the 3D content will display just as it did before. Now we are ready add our 2D content. Along the way we will flesh out some additional features to make a more complete game.

# A matter of state

Before we move on to actually rendering 2D items, we need to add a state machine to our game. Just as we did in RoboRacer2D, we need to be able to handle several different game states: displaying the splash screen, loading resources, displaying the main menu, running the game, pausing the game, and game over.

### Tip

Don't let the word **state** confuse you as it is used in several different ways in computer programming. We just finished a section on render state, learning how to push and pop this state from the OpenGL stacks. Now, we are talking about game state, which you can think of as the different modes that our game is in. A framework that handles different game states is known as a **state machine**.

## Adding the state machine

Fortunately, we will be able to take some of the code directly from RoboRacer2D. Open up `RoboRacer2D.cpp`. You can do this from inside the SpaceRacer3D project by clicking **File**, then **Open**, and then browsing to `RoboRacer2D.cpp`. This will allow you to copy information from `RoboRacer2D.cpp` and paste it into SpaceRacer3D.

### Tip

Opening a file loads it into the current project, but it does not add the file to the current project. However, you want to be careful because if you make changes to the file and save them, the original source file will be modified.

Copy the `GameState` enum and then paste it at the top of `SpaceRacer3D.cpp` just after the header files:

[PRE5]

We will be copying more code from `RoboRacer2D.cpp`, so go ahead and leave it open.

Next, we need to create a global game state variable. Add the following definition in the global variables section of `SpaceRacer3D.cpp`:

[PRE6]

The `gameState` variable will store the current game state.

## Getting ready for a splash

Just as we did in RoboRacer2D, we are going to start our game with a splash screen. The splash screen will be quickly loaded before any other resources, and it will be displayed for a few seconds before moving on to loading the game assets and starting the game.

Just under the definition for `gameState`, add the following lines:

[PRE7]

These two variables will handle the splash screen timing. Our splash screen is going to be one of the many 2D assets that we load into the game. Let's go ahead and define some variables for our 2D assets. Add the following lines of code to the global variables section of `SpaceRacer3D.cpp`:

[PRE8]

You will notice that all of our 2D assets are being handled as Sprites, a class that we borrowed from RoboRacer2D.

While we are here, let's add the following two lines as well:

[PRE9]

These two variables will be used to add a timing buffer to mouse clicks. Now, let's create a function to load the splash screen. Add the following function to `SpaceRacer3D.cpp` somewhere before the `StartGame` function:

[PRE10]

This code is exactly the same as the code from RoboRacer2D. In fact, feel free to copy and paste it directly from `RoboRacer2D.cpp`.

Remember: we set up our 2D orthographic viewport to exactly replicate the settings that we had in RoboRacer2D. This allows us to use the same exact code and positions for our 2D objects. Even better, it allows us to use the `Sprite` class from RoboRacer2D without changing any of the code.

### Tip

The `LoadSplash` function loads a file from the game resource folder called `splash.png`. You can download this file and all of the other 2D resources that are used in this chapter, from the book website. You should place all of them in a folder named `resources` under the same folder as the game source code. You also have to remember to add these resources to the **Resource Files** folder in the solution by right-clicking on **Resource Files**, then choosing **Add Existing Item**, then browsing to the `resources` folder and adding all of the items in that folder.

Next, we need to modify the `StartGame` function to load the splash screen. Move to the `StartGame` function add the following code:

[PRE11]

The first thing that we do is call the `LoadSplash` function, which sets the game state to `GS_Splash`, and then loads the splash page. Next, we have to update and render the splash page. Move to the `Update` function and modify it so that it looks like this:

[PRE12]

The only real change is that we implemented part of the state machine. You will notice how we moved all of the code to run the game under the `GS_Running` game state case. Next, we added an update for the splash screen game state. We will eventually modify the `Update` function to handle all of the game states, but we have some more work to do yet.

Now, we are ready to render the splash screen. Move to the `Render2D` function and add the following line of code between the `Enable2D` and `Disable2D` calls:

[PRE13]

At this point, if you run the game, you will see a splash screen render. The game will not move beyond the splash screen because we haven't added the code to move on yet.

# Creating the user interface

We are now ready to define our user interface, which will consist of 2D screens, text, and buttons. These will all work exactly as they did in RoboRacer2D. Look at the tip in the *Getting ready for a splash* section earlier in this chapter for a reminder of how to include prebuilt 2D resources in your project.

## Defining the text system

The 2D text system is built by first creating a font framework, then creating functions to display text on the screen. Open `RoboRacer2D.cpp` and copy the following functions. Then paste them into `SpaceRacer3D.cpp`:

*   `BuildFont`
*   `KillFont`
*   `DrawText`

We are going to add some new variables to handle the data that we want to display. Add the following lines of code to the global variables section of `SpaceRacer3D.cpp`:

[PRE14]

These variables will hold the stats and scoring used by the game:

*   `score`: This is the current game score
*   `speed`: This is the current speed of the ship
*   `missionTime`: This is the number of seconds that have elapsed since starting the mission
*   `asteroidsHit`: This is the number of asteroids hit by the player
*   `maximumSpeed`: This is the maximum speed obtained by the player

`Score`, `speed`, and `missionTime` will all be displayed on the heads-up-display (HUD) while the player is piloting the ship. `Score`, `asteroidsHit`, `missionTime`, and `maximumSpeed` will be displayed as stats at the end of the game.

Let's go to `StartGame` and initialize these variables:

[PRE15]

Now, let's create the functions to render these items on the screen. Add the following two functions to the game somewhere above the `Render2D` function:

[PRE16]

These functions work exactly like their corresponding functions in RoboRacer2D. First, we use `sprintf_s` to create a character string with the text that we want to display. Next, we use `glRasterPos2f` to set the render position in 2D. Then, we use `glCallLists` to actually render the font. In the `DrawCredits` function, we use the `DrawText` helper function to render the text.

Change `CheckCollisions` to look like the code below:

[PRE17]

This code updates the score and asteroid stats.

## Defining textures

Now, it's time to load all of our textures. Add the following function to the game:

[PRE18]

There is nothing new here! We are simply loading all of our 2D assets into the game as sprites. Here are a few reminders as to how this works:

*   Each sprite is loaded from a PNG file, specifying the number of frames. As none of these sprites are animated they all have one frame.
*   We position each sprite with a 2D coordinate.
*   We set the properties—visible means that it can be seen, and active means that it can be clicked on.
*   If the object is intended to be a button, we add it to the UI system.

## Wiring in render, update, and the game loop

Now that we have finally loaded all of our 2D assets, we are ready to finish the `Render2D` function:

[PRE19]

Again, there is nothing here that you haven't seen already. We are simply implementing the full state engine.

We can also implement the full `ProcessInput` function now that we have buttons to click. Add the following lines to the `switch` statement:

[PRE20]

Yep, we've seen all this before. If you recall, the `Input` class assigns a command enum to each button that can be clicked. This code simply processes the command, if there was any, and sets the state based on which button was just clicked.

We now implement the full `Update` function to handle our new state machine:

[PRE21]

Finally, we need to modify the game loop so that it supports all of our new features. Move to the `GameLoop` function and modify it so that it looks like the following code:

[PRE22]

As always, the game loop calls the `Update` and `Render` functions. We add a special case to handle the splash screen. If we are in the `GS_Splash` game state, we then load the rest of the resources for the game and change the game state to `GS_Loading`.

Note that several of the functions referenced previously haven't been created yet! We will add support for sound, fonts, and textures as we continue.

# Summary

We covered a lot of code in this chapter. The main lesson in this chapter was learning how to render 2D and 3D at the same time. We then added code to load all of our 2D resources as sprites. We also added the ability to render text, and now we can see our score, stats, and credits.

We implemented that state machine for the game and wired that into the input, update, render, and game loop systems. This included creating states for a splash screen, loading resources, playing the game, and displaying various game screens.

You now have a complete 3D game. Sure, there is more that you can do with it. In the next and final chapter, we will learn a few new tricks, then the rest is up to you!