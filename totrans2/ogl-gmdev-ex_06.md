# Chapter 6. Polishing the Silver

I'm sure that you are as excited as I am about the progress that you have made on your game. It's almost ready to publish, right? Well, not quite! There is a lot of work that goes into polishing your game before it is ready, and that's what this chapter is all about.

Many people have a great idea for a game, and lots of enthusiastic coders, such as you, actually code their game to the point where we have reached so far. Unfortunately, this is where a lot of projects die. For some reason, many first-time game coders don't take the time to really finish their game. There are lots of things that still need to be done to make your game presentable:

*   **Game state**: We already touched on game state a little bit when you learned how to pause your game. This chapter will continue the discussion of how you use game state to manage your game at various stages of gameplay.
*   **Splash screen**: Most games display one or more screens before the game starts. These screens, known as splash screens, often display the logo and name of the studios that were involved in creating the game. A splash screen shows that you went the extra mile in polishing your game.
*   **Menu screen**: Most games start with a menu of choices for the player. We will create a simple menu that loads after our splash screen and gives the player a few options.
*   **Scoring and statistics**: You probably noticed that our game currently doesn't keep score. Although it is possible to design a game that doesn't involve scoring, most players want to know how they are doing in the game.
*   **Winning and losing**: Again, while there are certainly games out there where no one wins or loses, most games have win-or-lose conditions that signal that the game is over.
*   **Game progression**: Most games allow the player to continue playing as long as the player has achieved certain goals. Many games are broken down into a series of levels, with each level becoming a little more difficult than the previous one. You will learn how to add this type of progression to your game.
*   **Credits**: Everyone likes to get credit for their work! Just like the movies, it is traditional to include a screen that shows each person that was involved in creating the game and what their role was. I'll show you how to create a simple credits screen.

# The state of the game

Remember when we coded the pause button back in [Chapter 4](ch04.html "Chapter 4. Control Freak"), *Control Freak*? We had to add some code that told the game whether it was active or paused. In fact, we defined the following enums:

[PRE0]

These `enums` defined two game states: `GS_Running`, and `GS_Paused`. We then set the default game state to `GS_Running` in the `StartGame` function:

[PRE1]

As long as the game state is set to `GS_Running`, then the game continues to cycle through the game loop, processing updates, and rendering the scene. However, when you click the pause button, the game state is set to `GS_Paused`. When the game is paused, we no longer update the game objects (that is, the robot, pickups, and enemies), but we do continue to render the scene and process the UI (user interface) so that buttons can be clicked.

## State machines

The mechanism used to set up and control game states is known as a **state machine**. A state machine sets up separate and distinct stages (or **states**) for the game. Each state defines a certain set of rules for what is supposed to happen or not happen during each state. For example, our simple state machine has two states with the following rules, illustrated by the following matrix:

|   | GS_Running | GS_Paused |
| --- | --- | --- |
| **Input** | All input | Only UI input |
| **Objects Updating** | All objects | Only UI objects |
| **Collision Detection** | All collideables | No need to check for collisions |
| **Spawning** | All spawnables | No spawning |
| **Rendering** | All objects | All objects |

The state machine also defines the progression from one state to another. Here is a simple diagram showing the progression in our current state machine:

![State machines](img/8199OS_06_01.jpg)

This state diagram is pretty simple. If you are in the running state, then it is legal to go to the paused state. If you are in the paused state, then it is legal to go to the running state. As we will see, most games are much more complex than this!

## Why do we need a state machine?

At first glance, you may wonder why we even need a state machine. You could, for example, set up several Boolean flags (maybe one called `running` and one called `paused`), and then insert them into the code in the same way that we are using our enums.

This solution may work considering that our current game only has two states, but even then, it starts to get complicated if you choose to use Booleans. For example, to change the state from running to paused, I would always have to make sure to properly set both Booleans:

[PRE2]

When I went from the running state to the paused state, I would have to set both Booleans again:

[PRE3]

Imagine the problem if I forgot to change both Booleans and left the game in a state where it was both running and paused! Then imagine how complicated this becomes if my game has three, four, or ten states!

Using enums is not the only way to set up a state engine, but it does have immediate advantages over using Booleans:

*   Enums have a descriptive name associated with their value (for example, `GS_Paused`), whereas Booleans only have `true` and `false`.
*   Enums are already mutually exclusive. In order to make a set of Booleans mutually exclusive, I have to set one to `true` and all the others to `false`.

The next consideration as to why we need a state machine is that it simplifies the coding of the control of the game. Most games have several game states, and it is important that we are able to easily manage which code runs in which state. An example of game states that are common to most games includes:

*   Loading
*   Starting
*   Running
*   Paused
*   Ending
*   GameWon
*   GameLost
*   GameOver
*   NextLevel
*   Exiting

Of course, this is just a representative list, and each coder picks his or her own names for their game states. But I think that you get the idea: there are a lot of states that a game can be in, and that means it is important to be able to manage what happens during each state. Players tend to get angry if their character dies while the game was paused!

## Planning for state

We are going to expand our simple state machine to include several more game states. This is going to help us to better organize the processing of the game, and better define which processes should be running at any particular time.

The following table shows the game states that we are going to define for our game:

| State | Description |
| --- | --- |
| Loading | The game is loading and the Splash screen should be displayed |
| Menu | The main menu is showing |
| Running | The game is actively running |
| Paused | The game is paused |
| NextLevel | The game is loading the next level |
| GameOver | The game is over and the stats are being displayed |
| Credits | Showing the Credits screen |

Here is our state diagram machine:

|   | Splash | Loading | Menu | Running | Paused | Next | GameOver | Credits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Input** | None | None | UI | All | UI | UI | UI | UI |
| **Updating** | Splash | Splash | UI | All | UI | UI | UI | UI |
| **Collision Detection** | None | None | None | All | None | None | None | None |
| **Spawning** | None | None | None | All | None | None | None | None |
| **Rendering** | Splash | Splash | Menu | Game | Game | Game | GameOver | Credits |

Finally, here is our state diagram:

![Planning for state](img/8199OS_06_02.jpg)

It turns out that our state diagram will also double as a UI diagram. A UI diagram is a diagram of all of the screens in a program and how they interact with each other. It turns out that each time that we want to change to a different screen in our game, we are also changing to a different screen. This isn't exactly the case—when the game is paused, it doesn't launch a completely new screen. However, there is often a very close correlation between the UI diagram and the state diagram.

Looking at the state diagram, you can easily see the legal state changes versus the illegal state changes. For example, it is legal to change the state from playing to paused, but you can't change the state from playing to credits.

Having this structure in place will guide us as we implement all of the final polish features that we want to add to our game.

## Defining the new state

The first step in expanding our game state machine is adding the required `enums`. Replace the `GameState enum` code with the following code:

[PRE4]

As we implement the polish features covered in this chapter, we will implement code that uses these game states.

## Implementing the state machine

In order for our state machine to have any effect, we need to modify the code so that key decisions are made based on the game state. There are three functions that game state affects in a big way:

*   **Update**: Some game states update game objects, while other game states update only the UI or a particular sprite
*   **Render**: Different game states render different items
*   **Input**: Some game states accept all input, while other game states only process UI input

It should come as no surprise then that we will be changing the `Update`, `Render`, and `ProcessInput` functions.

First, let's modify the `Update` function. Change the `Update` function in `RoboRacer2D.cpp` to match the following code:

[PRE5]

As you can see, we are now using a `switch` statement to handle each game state. This is a whole lot more readable than using `if` statements, and it keeps the code much more structured. If we need to add another game state, we just add another `case` to the `switch` statement.

Notice that each `case` has its code to run specific to that game state. Some lines of code are duplicated (almost every state has some input), but this is a small price to pay for clarity. `GS_Running` has the most work to do, while `GS_Loading` has the least work to do. We will be adding code to each switch as we add polish features.

Now, let's give the `Render` function an upgrade. Replace the `Render` function with the following code:

[PRE6]

In this case, we have some work that needs to be done regardless of the game state. We need to clear the OpenGL buffer, and set the matrix to identity. Then we decide which items to render based on the game state, and finally, we swap the buffers.

If you look closely, `GS_Running` and `GS_Paused` render the same items. This is because the pause and render buttons are rendered over the top of the gameplay screen, so we still need to render the entire game even when we are paused. We will be adding code to each switch as we add polish features.

Finally, we need to apply our state machine to the `ProcessInput` function. As the function is so long, I am only showing the top lines of the function. Change all of the lines above the `uiTimer += p_deltaTime;` statement to the following code:

[PRE7]

First, we get the latest command. Then, depending on the game state, we perform the following actions:

*   Ignore and return if we are still in the loading state
*   Reset the command to only handle UI commands if the game state is menu, paused, next level, or game over
*   Leave the command unchanged if we are in the running game state

This is exactly what we did in the prior versions, except we only had two game states to deal with in the prior versions. Once the command is handled, we move on to the `uiTimer += p_deltaTime;` (everything after this line is unchanged from the prior versions).

# Making a splash

A splash menu adds a touch of class to your game and also does a little bragging. Typically, the splash screen shows off your company logo. In fact, many game projects have multiple studios that work on them, so there are often multiple splash screens. We will use just one!

It is important to get the splash screen up and running as soon as possible, so we will do that before we perform any other loading. Part of the function of a splash screen is to give the player something pretty to look at while the rest of the game is loading.

## Creating the splash screen

It's up to you to create a splash screen that defines your game. For convenience, we have included one in the code resource package for this chapter called `splash.png`. Make sure you copy `splash.png` into your project. The only requirement for the splash image is that it is 800 x 600 pixels, the same resolution as our game screen.

## Defining the splash screen

As with all images in this game, we will implement the splash screen as a sprite. Declare the splash sprite at the top of `RoboRacer2D.cpp`:

[PRE8]

We also want to define some timers for the splash screen:

[PRE9]

As we want to define the splash screen separately, we will create a separate function just to load it. Create the `LoadSplash` function using the following code:

[PRE10]

We are not going to make a significant change to the `StartGame` function. We are going to only load the splash screen, and defer loading the other game resources. This will get our splash screen up as soon as possible. Change the `StartGame` function so that it looks like the following code:

[PRE11]

Notice that we only load the splash resources and set a few variables here. We also set the splash timer so that it will show up for at least five seconds.

Next, modify the `GS_Splash` case in the `Update` function to look like the following code:

[PRE12]

This code updates the splash timer. When the timer exceeds our threshold, then the game state changes to `GS_Menu`. We will define the code to load the next menu.

Modify the `GS_Splash` case in the `Render` function to look like the following code:

[PRE13]

### Tip

As the splash sprite is only a static image, you may wonder why we update the splash sprite. While an update has no effect on our current code, consider a case where I wanted to implement a dynamic, animated splash screen.

## Loading our resources

If you have been paying attention, then you should realize that we removed the `LoadTextures` call from the `StartGame` function. Instead, we are going to load the textures in the `GameLoop` function. Change `GameLoop` so that it looks like the following code:

[PRE14]

If you recall, `GameLoop` is called every frame. We need `GameLoop` to be running to display our splash screen, which we have already loaded. But on the first call to `GameLoop`, we haven't loaded our other resources.

We check to see whether our game state is `GS_Splash`. If it is, we call load textures, and immediately change the game state to `GS_Loading`. If we didn't change the game state, then the game would attempt to load the textures every frame, which would be a very bad thing! This is another practical example of why we define different game states in our state machine.

### Tip

In a way, we haven't created a true splash screen. That is because our splash still depends on Windows and OpenGL initializing before the splash screen can even be loaded and rendered. True splash screens use a snippet of code that does not depend on all of this initialization so that they can load before everything else. Unfortunately, that level of detail is beyond the scope of our book. Sometimes, the splash screen will run on a separate thread so that it is independent of the startup code.

![Loading our resources](img/8199OS_06_03.jpg)

When you run the game now, you should see the splash screen display, but then nothing else happens. This is because we changed the game state to `GS_Menu` in the `Update` function, and we have not coded for that game state yet! If you want to test your splash screen, change `m_gameState = GameState::GS_Menu` to `m_gameState = GameState::GS_Running` in the `Update` function. Just don't forget to change it back before you move on.

### Tip

The ability to change your game state allows you to reroute the flow of your game. This is very useful, for example, when you are trying to code a new game state but you aren't ready to run it in the game yet. Once the new game state is coded, then you can wire it in.

# What's on the menu?

Main menus may have disappeared in many applications, but they are still alive and well in games. The main menu gives the player a chance to decide what to do once the game has loaded. We are going to create a simple menu that allows the player to start the game, display the credits, or exit the game.

## Creating the menu

Our menu will be built out of two components. First, we will load an image to use as the background. Next, we will load additional images to use as UI buttons. Together, these images will create a screen that will allow the player to navigate our game.

We will start by defining a sprite to represent the menu. Add the following line of code to the variable declarations in `RoboRacer2D.cpp`:

[PRE15]

Next, we will instantiate the menu in the `LoadTextures` function. Add the following code to `LoadTextures`:

[PRE16]

Make sure that you have downloaded the `menu.png` texture from the book website, or that you have created your own background at 800 by 600 pixels.

Now, we must modify the `Update` and `Render` functions. Modify the `GS_Menu` case in `Update` to the following code:

[PRE17]

Next, modify the `GS_Menu` case in the `Render` function:

[PRE18]

If you run the game now, the splash screen should display for five seconds, followed by the menu screen.

## Defining the menu buttons

Our next task is to add buttons to the menu screen that the player can click. These buttons will work similar to the pause and resume buttons that we have already created.

We will start by declaring variables for the buttons. Add the following declarations to the variables section in `RoboRacer2D.cpp`:

[PRE19]

These three pointers will manage the three buttons on our main menu. Next, add the following code to `LoadTextures` to instantiate the buttons:

[PRE20]

This code is mostly the same as the code that we used to instantiate the pause and resume buttons. One small difference is that we set all three buttons to be visible. Our code already enforces that these buttons will not render unless we are in the game state `GS_Menu`.

We do, however, want to set the buttons as inactive. This way the `input` class will ignore them until we want them to be activated.

As with all of our objects, we now need to wire them into the `Update` and `Render` functions. Change the `GS_Menu` case in the `Update` function to the following code:

[PRE21]

This is where we set the buttons on our menu to be active. We want to guarantee that the buttons on the menu are active when we are in the game state `GS_Menu`.

Next, change the `GS_Menu` case in the `Render` function to the following code:

[PRE22]

In order for the buttons to actually do something, we need to add the following code to the `CM_UI` case in `ProcessInput`:

[PRE23]

Notice that we change the game state if the play button or credits button are clicked (if the exit button is clicked, we simply post the quit message). Notice that we have to do a little button management, setting the buttons on the menu to be inactive once we are no longer in the `GS_Menu` game state. This is because our input class checks the input for all buttons that are active. Leaving the buttons active would mean that they could still be clicked even though they are not being displayed on the screen.

We don't have to set the buttons to be invisible. This is because changing the state will automatically stop these buttons from updating or rendering. The same is true of the menu screen. Once the game state is changed, it will not render or update. This is one of the big advantages of utilizing a state machine.

![Defining the menu buttons](img/8199OS_06_04.jpg)

If you run the program right now, the main menu will display. If you click the play button, the game will start. If you click the exit button, the game will exit. We will implement the credit screen next.

# Getting some credit

Everyone likes to get credit for their hard work! Most games will implement a credits screen that shows the name and function of each person involved in creating the game. For AAA titles, this list may be as long as a list for a movie. For smaller, independent games, this list might be three people.

## Creating the credits screen

Similarly to the main menu, the credits screen will be based on a background image and a button that can be clicked. We will also need to add text to the screen.

Let's start by declaring a pointer for our screen. Add the following declaration to the variables section of `RoboRacer2D.cpp`:

[PRE24]

Then, we will instantiate the credits screen in `LoadTextures`:

[PRE25]

Next, we wire the credits screen into `Update`:

[PRE26]

We also update `Render`:

[PRE27]

## Getting back to the main menu

We now need to add a button that allows us to get from the credits screen back to the main menu. We first declare the pointer in the variables declaration section:

[PRE28]

We then instantiate the button in `LoadTextures`:

[PRE29]

Let's add the button to `Update`:

[PRE30]

We also update `Render`:

[PRE31]

Similarly to the menu buttons, we now need to add code to the case `Input::Command::CM_UI:` case in `ProcessInput` to handle clicking on the menu button:

[PRE32]

When the menu button is clicked, we change the game state back to menu, and set the menu button to be inactive. Due to the code that we have already written, the menu screen will automatically display.

![Getting back to the main menu](img/8199OS_06_05.jpg)

# Working with fonts

Until now, we embedded any text that we needed inside of an existing texture. However, there are times when we may want to have the code decide what text to display. For example, on our credits screen, we don't want to make a graphic for each person's name who took part in creating the game.

## Creating the font

We need a way to render text directly to the screen, and this means that we also need a way to define the font that we want to use when rendering the text. First, we need to add a global variable that services as a handle to our fonts. Add the following line to the variable declarations in the code:

[PRE33]

Now, we need to add the following code to create the font:

[PRE34]

This code creates a font using three main elements.

First, we use `glGenLists` to create 96 display lists to hold each letter of our font. A display list is basically a buffer that can hold rendering data. Next, we call `CreateFont` to create a Windows font. The parameters of the `CreateFont` function specify the type of font that we want to create. Finally, we use `wglUseFontBitmaps` to assign our new font to the font handle that we created earlier.

One little twist is that we have to create a temporary `HFONT` object called `tempFont` with all the properties, then we assign `tempFont` to `newFont` and delete `tempFont`.

We will want to delete the display lists when the program closes down, so add the following utility function:

[PRE35]

This code simply uses `glDeleteLists` to delete the display lists that we created to hold our font.

## Drawing text

Now that we have a font, we need to have a function that will render text to the screen. Add the following function to the code:

[PRE36]

This code takes a string and an *x* and *y* position, and draws the text at that position. It also takes `r`, `g`, and `b` parameters to define the text color:

*   `glBindTexture`(`GL_TEXTURE_2D`, `0`): This tells OpenGL that we are going to be working with 2D textures (i.e. the fonts) `glColor3f(r, g, b)`: This sets the color of the font.
*   `glRasterPos2f`: This is used to set the current draw position on the screen.
*   `glPushAttrib(GL_LIST_BIT)`: This tells OpenGL that we are going to render using display lists.
*   `glListBase`: This sets the current start of the list. We subtract 32 because the ASCII value for a space is 32, and we don't use any characters with lower ASCII values.
*   `glCallLists`: This is used to retrieve the lists for each character in the text.
*   `glPopAttrib`: This returns the OpenGL attribute to its previous value.

Now, we are ready to draw our credits text:

[PRE37]

First, we set the position on the screen where we want to draw, then we use the `DrawText` function to actually perform the drawing. The first line adds me (a subtle indulgence), and the second line is for you!

## Wiring in the font support

We have a few more book keeping tasks to perform to get the font support to work. First, modify the `GameLoop` code, adding the highlighted line:

[PRE38]

This will create our fonts when the game starts up.

Next, fill out the `GS_Credits` case of the `m_gameState` switch in the `Render` function:

[PRE39]

This draws the credits text when the game state changes to `GS_Credits`. Congratulations! You can finally get the credit that you deserve!

# Level up!

A lot of the fun in games is trying to increase your score. Part of good game design is to make the game challenging to play, but not so challenging that the player cannot score or improve.

Most players also get better at a game as they play, so if the game difficulty does not increase, the player will eventually get bored because the player will no longer be challenged.

We will start by simply displaying the score on the screen so that the player can see how well they are doing. Then we will discuss techniques that are used to continually increase the difficulty of the game, thus steadily increasing the challenge.

## Displaying the score

We already learned how to display text on the screen when we were creating the credits screen. Now, we will use the same techniques to display the score.

If you recall, we already have a mechanism to keep track of the score. Every sprite has a value property. For pickups, we assign a positive value so that the player gains points for each pickup. For enemies, we assign a negative value so that the player loses points whenever they collide with an enemy. We store the current score in the value property of the player.

Add the following code to `RoboRacer2D.cpp` to create the `DrawScore` function:

[PRE40]

This code works just like the `DrawCredits` function that we created earlier. First, we create a character string that holds the current score and a caption, then we use `DrawText` to render the text.

We also need to wire this into the main game. Modify the `GS_Running` case of the `m_gameState` switch in the `Render` function with the highlighted line:

[PRE41]

The score will display both when the game is running and when the game is paused.

## Game progression

In order to add progression to the game, we need to have certain thresholds established. For our game, we will set three thresholds:

*   Each level will last two minutes
*   If the player receives less than five pickups during a level, the game will end, and the game over screen will be displayed
*   If the player receives five or more pickups, then the level ends and the next level screen is displayed

For each level that the player successfully completes, we will make things a little more difficult. There are many ways that we could increase the difficulty of each level:

*   Increase the spawn time for pickups
*   Decrease the speed of the robot

To keep things simple, we will only do one of these. We will increase the spawn time threshold for pickups by .25 seconds for each level. With pickups spawning less often, the player will eventually receive too few pickups, and the game will end.

## Defining game levels

Let's set up the code for level progression. We will start by defining a timer to keep track of how much time has passed. Add the following declarations to `RoboRacer2D.cpp`:

[PRE42]

We will initialize the variables in the `StartGame` function:

[PRE43]

We are setting up a timer that will run for 120 seconds, or two minutes. At the end of two minutes the level will end and the spawn time for pickups will be incremented by .25 seconds. We will also check to see whether the player has received five pickups. If not, the game will be over.

To handle the logic for the level progression, let's add a new function called `NextLevel` by adding the following code:

[PRE44]

As stated previously, we check to see whether the number of pickups that the robot has is less than the pickup threshold. If so, we change the game state to `GS_GameOver`. Otherwise, we reset the level timer, reset the pickups received counter, increment the pickup spawn timer, and set the game state back to `GS_Running`.

We still need to add some code to update the level timer and check to see whether the level is over. Add the following code to the `GS_Running` case in the `Update` function:

[PRE45]

This code updates the level timer. If the timer exceeds our threshold, then call `NextLevel` to see what happens next.

Finally, we need to add two lines of code to `CheckCollisions` to count the number of pickups received by the player. Add the following highlighted line of code to `CheckCollisions`:

[PRE46]

## Game stats

It would be nice for the player to be able to see how they did between each level. Let's add a function to display the player stats:

[PRE47]

We will now wire this into the next level screen.

## The next level screen

Now that we have the logic in place to detect the end of the level, it is time to implement our next level screen. By now, the process should be second nature, so let's try an abbreviated approach:

1.  Declare a pointer to the screen:

    [PRE48]

2.  Instantiate the sprite in `LoadTextures`:

    [PRE49]

3.  Modify the `GS_NextLevel` case in the `Update` function:

    [PRE50]

4.  Modify the `GS_NextLevel` case in the `Render` function to look like the following code::

    [PRE51]

## Continuing the game

Now, we need to add a button that allows the player to continue the game. Again, you have done this so many times, so we will use a shorthand approach:

1.  Declare a pointer for the button:

    [PRE52]

2.  Instantiate the button in `LoadTextures`:

    [PRE53]

3.  Add this code to `Update`:

    [PRE54]

4.  Add this code to `Render`:

    [PRE55]

5.  Add this code to `ProcessInput`:

    [PRE56]

Clicking the continue button simply changes the game state back to `GS_Running`. The level calculations have already occurred when `NextLevel` was called.

# Game over

As the saying goes, all good things must come to an end. If the player doesn't meet the pickup threshold, the game will end, and the game over screen will be displayed. The player can choose to replay the game or exit.

## The game over screen

Our last screen is the game over screen. By now, the process should be second nature, so let's try an abbreviated approach:

1.  Declare a pointer to the screen:

    [PRE57]

2.  Instantiate the sprite in `LoadTextures`:

    [PRE58]

3.  Change the `GS_GameOver` case in the `Update` function to look like the following code:

    [PRE59]

4.  Add the following code to `Render`:

    [PRE60]

As a bonus, we will also draw the game stats on the game over screen.

![The game over screen](img/8199OS_06_09.jpg)

## Replaying the game

We need a way to reset the game to its initial state. So, let's create a function to do this:

[PRE61]

Next, we need to add a button that allows the player to replay the game. Again, as you have done this so many times, we will use a shorthand approach:

1.  Declare a pointer for the button:

    [PRE62]

2.  Instantiate the button in `LoadTextures`:

    [PRE63]

3.  Add the following code to `Update`:

    [PRE64]

4.  Add the following code to `Render`:

    [PRE65]

5.  Add the following code to `ProcessInput`:

    [PRE66]

Notice how we are reusing the exit button in the `Update` function. Also, if the player wants to replay the game, we call the `RestartGame` function when the player clicks the replay button. This resets all of the game variables and allows the player to start all over.

![Replaying the game](img/8199OS_06_07.jpg)

# Summary

We covered a lot of ground in this chapter. The focus of the chapter is to add all of the final elements to the game that make it a truly polished game. This involves adding a lot of screens and buttons, and to manage all of this, we introduced a more advanced state machine. The state machine acts like a traffic director, routing the game to the correct routines depending on the game state.

In the next chapter, we will add sound effects and music to our game!