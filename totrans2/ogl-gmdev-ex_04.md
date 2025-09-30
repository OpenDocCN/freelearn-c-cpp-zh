# Chapter 4. Control Freak

Most games are designed to be interactive. This means that the player must have some way to control what happens during the game. In the last chapter, you wrote code that displayed the robot and moved him across the screen. Now, you will control the robot!

This chapter will explain how to implement an input system to control the game's character, and interact with the game. Topics will include:

*   **Types of input**: There are many ways to interact with your game. Typically, games written for the PC depended on the mouse and keyboard. Direct touch input has now become the standard for mobile and tablet devices, and soon every PC will also have a touch-enabled display. We will cover the most common methods to receive input in your game.
*   **Using the mouse and keyboard**: In this section, you will write code to receive input from the mouse and keyboard to control both the game and our friendly robot.
*   **Creating the user interface**: In addition to controlling our robot, we also need a way to interact with the game. You will learn how to create an onscreen interface that allows you to control the game and choose the game options.
*   **Controlling the character**: We want our robot to be able to walk, run, jump, and play! You will learn how to use the mouse and keyboard to control how your robot moves about on the screen.

# A penny for your input

It's likely that at some point in your life, you have been part of a conversation that seemed one-sided. The other party was talking and talking, and it didn't seem you could get a word in. After a while, such a conversation becomes quite boring!

The same would happen with a computer game that didn't allow any **input**. Input is a set of techniques that allows you to control the game. There are many ways to implement an input system, and we will cover them here.

## The keyboard input

The most common form of input for most computers is the keyboard. Obviously, the keyboard can be used to enter text, but the keyboard can also be used to directly control the game.

Some examples of this include the following:

*   Using the right arrow, left arrow, up arrow, and down arrow keys to control the character (we'll be using this)
*   Using the *W*, *A*, *S*, and *D* keys as to move the character (these keys almost form a cross on the keyboard, making them a good substitute to move up, left, down, and right, respectively)
*   Using certain keys to perform predefined actions, such as:

    *   Using the *Esc* key or *Q* to quit
    *   Using the Spacebar or *Enter key* to fire a projectile

These are just a few examples. In fact, there are some games that seem to use every key on the keyboard!

## Using the mouse

The mouse has been around for a long time, so it makes sense that the mouse is used in many games. The mouse can be used in several ways:

*   The left and right mouse buttons can perform specific actions.
*   The wheel can be pushed and used as a third button.
*   The mouse wheel can be used to scroll.
*   The position of the mouse pointer can be tracked and used in conjunction with any of the previous actions. We will use a combination of the left mouse button and the mouse pointer position to click onscreen buttons when we design our user interface.

## Touch

More and more devices now respond to touch. Many input systems treat touch very similarly to the mouse:

*   A single touch is equivalent to using the left mouse button
*   A single touch that is held is equivalent to using the right mouse button
*   The position of the finger can be used in the same way as the mouse pointer

However, there are many features of touch that cannot be easily equated to the mouse. For example, most touch interfaces allow several touches to be handled simultaneously. This feature is known as multitouch. This has led to many standard gestures, including:

*   The swipe or flick (moving one or more fingers quickly across the screen)
*   The pinch (moving two fingers together)
*   The zoom (moving two fingers apart)

Unfortunately, we won't be implementing touch in this game because the target device for this book is the PC.

## Other inputs

The advent of mobile devices was followed by an explosion of input techniques. Some of the more common ones include:

*   The accelerometer, which can be used to track the physical motion of the device
*   Geolocation, which can be used to detect the physical location of the device
*   The compass, which can be used to detect the orientation of the device
*   The microphone, which can be used to accept voice input

There are many other input techniques, and there is a lot of overlap. For example, most PCs have a microphone. Again, while many games in the mobile market are taking advantage of these alternative input methods, our game will be limited to the keyboard and mouse.

# Someone is listening

Now, it's time to actually write some code to implement input for our game. It turns out that some rudimentary input has already been implemented. This is because Windows is an **event driven** operating system and is already looking for input to occur. From a simplistic point of view, the main task of Windows (or any modern operating system) is to listen for **events**, and then do something based on those events.

So, whenever you hit a key on your keyboard, an event is triggered that wakes up Windows and says, "Hey, someone hit the keyboard!" Windows then passes that information to any programs that happen to be listening to keyboard events. The same occurs when you use the mouse.

## The WndProc event listener

We have already told our program that we want it to listen to events. Open `RoboRacer.cpp` and locate the `WndProc` function. `WndProc` is part of the code that was created for us when use used the **Win32 Project template** to start our game. `WndProc` is known as a **callback function**.

Here is how a callback function works:

*   First, the function name is registered with the operating system. In our case, this occurs in `CreateGLWindow`:

    [PRE0]

    This line tells our window class to register a function called `WndProc` as the event handler for our program.

*   Now, any events that are caught by Windows are passed to the `WndProc` function. The code in `WndProc` then decides which events to handle. Any events that aren't handled by `WndProc` are simply ignored by the program.

As `WndProc` was created for a typical Windows application, it contains some things that we don't need, while there are some things that we can use:

[PRE1]

The main work is done by `switch`, which handles various windows events (all prefixed by **WM**, which is an abbreviation for **Windows Message**):

*   The `WM_COMMAND` events can all be ignored. In a typical Windows application, you would create a menu and then assign various command events to be triggered when the user clicks on a command on the menu (for example, `IDM_ABOUT` to click on the **About** command). Games almost never use the standard Windows menu structure (and so, neither do we).
*   We also ignore the `WM_PAINT` event. This event is triggered whenever the window containing the program needs to be redrawn. However, we are constantly redrawing our window using OpenGL via the `Render` function, so we don't need to add code to do that here.
*   We are already handling the `WM_DESTROY` event. This event is triggered when you click the close icon (**X**) in the upper-right corner of the Windows. Our handler responds to this by posting its own message using `PostQuitMessage(0)`. This tells our program that it is time to quit.

## Handling the message queue

We discussed the Windows messaging system in [Chapter 1](ch01.html "Chapter 1. Building the Foundation"), *Building the Foundation* but this discussion warrants a recap. If you take a look at the `_wWinMain` function, you will see this block of code that sets up the main messaging loop:

[PRE2]

The relevant part of this discussion is the call to `PeekMessage`. `PeekMessage` queries the message queue. In our case, if the `WM_QUIT` message has been posted (by `PostQuitMessage`), then done is set to `true` and the `while` loop exits, ending the game. As long as `WM_QUIT` has not been posted, the `while` loop will continue and `GameLoop` will be called.

The event driven system is a great way to handle input and other actions for most programs, but it doesn't work well with games. Unlike games, most programs just sit around waiting for some kind of input to occur. For example, a word processing program waits for either a keystroke, a mouse button click, or a command to be issued. With this type of system, it makes sense to wake up the program every time an event happens so that the event can be processed.

Games, on the other hand, do not sleep! Whether or not you are pressing a button, the game is still running. Furthermore, we need to be able to control the process so that an input is only processed when we are ready for it to be handled. For example, we don't want input to interrupt our render loop.

The following diagram shows how Windows is currently rigged to handle input:

![Handling the message queue](img/8199OS_04_01.jpg)

## Handling mouse and keyboard inputs

We could expand `WndProc` to handle all of the input events. However, this is a terribly inefficient way to handle input, especially in a real-time program, such as a game. We will let Windows handle the case when the user closes the Window. For everything else, we are going to create our own input class that directly polls for input.

There are many different ways to design an input system, and I am not going to presume that this is the best system. However, our input system accomplishes two important tasks:

*   We define a consistent input interface that handles both mouse and keyboard input
*   We handle input by directly polling for mouse and keyboard events during each frame (instead of waiting for Windows to send them to us)

## Creating the Input class

Create a new class called `Input`. Then add the following code into `Input.h`:

[PRE3]

As with all of our code, let's take a close look to see how this is designed:

*   We include `Windows.h` because we want access to the Windows API virtual key constants. These are constants that have been defined to represent special keys on the keyboard and mouse.
*   We create the `Key` enum so that we can easily define values to poll the keys that we want to handle.
*   We create the `Command` enum so that we can easily map input to command actions that we want to support.
*   We define a C++ macro named `KEYDOWN`. This greatly simplifies our future code (see the next step for details).
*   The class only has one member variable, `m_command`, which will be used to hold the last action that was requested.
*   We define three member functions: the constructor, the destructor, `Update`, and `GetCommand`.

## Virtual key codes

In order to understand how our input system works, you must first understand virtual key codes. There are a lot of keys on a keyboard. In addition to letters and numbers, there are special keys, including shift, control, escape, enter, arrow keys, and function keys. Coming up with a simple way to identify each key is quite a task!

Windows uses two techniques to identify keys; for the normal keys (letters and numbers), each key is identified by the ASCII code of the value that is being tested. The following table shows the ASCII value for the keys that we use in our game:

| ASCII Value | Key |
| --- | --- |
| 87 | *W* |
| 65 | *A* |
| 83 | *S* |
| 68 | *D* |
| 81 | *Q* |

For special keys, Windows defines integer constants to make them easier to work with. These are known as virtual key codes. The following table shows the virtual key codes that we will work with in our game:

| Virtual key code | Key |
| --- | --- |
| `VK_ESC` | *Esc* |
| `VK_SPACE` | Spacebar |
| `VK_LEFT` | Left arrow |
| `VK_RIGHT` | Right arrow |
| `VK_UP` | Up arrow |
| `VK_DOWN` | Down arrow |
| `VK_RETURN` | *Enter* |
| `VK_LBUTTON` | Left mouse button |
| `VK_RBUTTON` | Right mouse button |

Notice that there are even virtual key codes for the mouse buttons!

## Querying for input

The `GetAsyncKeyState` function is used to query the system for both keyboard and mouse input. Here is an example of that command:

[PRE4]

First, we pass in a virtual key code (or ASCII value), then we do a logical and with the hex value `8000` to strip out information that we don't need. If the result of this call is `true`, then the queried key is being pressed.

It's a pretty awkward command to have to use over and over again! So, we create a C++ macro to make things simpler:

[PRE5]

`KEYDOWN` executes the `GetAsyncKeyState` command. The macro accepts a key code as a parameter, and returns `true` if that key is being pressed or `false` if that key is not being pressed.

## Implementing the Input class

All of the actual work is for our input system is done in the `Update` function, so let's implement the `Input` class. Open `Input.cpp` and enter the following code:

[PRE6]

In a nutshell, the `Update` function queries all of the keys that we want to check simultaneously, and then maps those keys to one of the command enums that we have defined in the class header. The program then calls the class `GetCommand` method to determine the current action that has to be taken.

If you are really paying attention, then you may have realized that we only store a single command result into `m_command`, yet we are querying many keys. We can get away with this for two reasons:

*   This is an infinitely simple input system with few demands
*   The computer cycles through the input at 60 frames per second, so the process of the player pressing and releasing keys is infinitely slow in comparison

Basically, the last key detected will have its command stored in `m_command`, and that's good enough for us.

Also, notice that we set the initial command to `Input::Command::STOP`. As a result, if no key is currently being held down, then the `STOP` command will be the final value of `m_command`. The result of this is that if we are not pressing keys to make our robot move, then he will stop.

## Adding input to the game loop

Now that we have an input class, we will implement it in our game. We will handle input by adding it to `Update`. This gives us total control over when and how we handle input. We will only rely on the Windows event listener to tell us if the Window has been closed (so that we can still shut the game down properly).

Open `RoboRacer.cpp` and modify the `Update` function so that it looks like the following code:

[PRE7]

Before now, our `Update` function only updated the game's sprites. If you recall, the sprite `Update` method modifies the position of the sprites. So, it makes sense to perform the input before we update the sprites. The `Update` method of the `Input` class queries the system for input, and then we run a `ProcessInput` to decide what to do.

## Processing our input

Just before we update all of our sprites, we need to process the input. Remember, the `Input` class `Update` method only queries the input and stores a command. It doesn't actually change anything. This is because the `Input` class does not have access to our sprites.

First, open `RoboRacer.cpp` and include the Input header file:

[PRE8]

We need to add a variable to point to our `Input` class. Add the following line in the variable declarations section:

[PRE9]

Then, modify `StartGame` to instantiate the `Input` class:

[PRE10]

Now, we will create a function to process the input. Add the following function to `RoboRacer.cpp`:

[PRE11]

`ProcessInput` is where the changes to our game actually take place. Although it seems like a lot of code, there are really only two things that are happening:

*   We query the input system for the latest command using `inputManager->GetCommand()`
*   Based on that command we perform the required actions

The following table shows the commands that we have defined, followed by a description of how this affects the game:

| Command | Actions |
| --- | --- |
| `CM_STOP` |  
*   Set the velocity of `player` to `0`
*   Set the background velocity to `0`

 |
| `CM_LEFT` |  
*   If `player` is currently moving right, deactivate the right sprite and make it invisible, and set the left sprite to the right sprite's position
*   Set `player` to the left sprite
*   Activate the left sprite and make it visible
*   Set the velocity of the left sprite to `-50`
*   Set the velocity of the background to `50`

 |
| `CM_RIGHT` |  
*   If `player` is currently moving left, deactivate the left sprite and make it invisible, and set the right sprite to the left sprite's position
*   Set `player` to the right sprite
*   Activate the right sprite and make it visible
*   Set the velocity of the right sprite to `50`
*   Set the velocity of the background to `-50`

 |
| `CM_UP` |  
*   Call the sprite's `Jump` method with the parameter set to `UP`

 |
| `CM_DOWN` |  
*   Call the sprite's `Jump` method with the parameter set to `DOWN`

 |
| `CM_QUIT` |  
*   Quit the game

 |

## Changes to the Sprite class

Now that the robot can jump, we need to add a new method to the `Sprite` class to give the robot the ability to jump:

First, we will add an enum to Sprite.h to track the sprite state:

[PRE12]

Next, we need a new member variable to track if an element has been clicked. Add:

[PRE13]

Now go to the constructor in Sprite.cpp and add a line to initialize the new variable:

[PRE14]

Add the following code to `Sprite.h`:

[PRE15]

Then add the following code to `Sprite.cpp`:

[PRE16]

Our robot is a little unique. When he jumps, he hovers at an elevated level until we tell him to come back down. The `Jump` method moves the robot `75` pixels higher when the player presses the up arrow, and moves him `75` pixels back down when the player presses the down arrow. However, we want to make sure that we don't allow a double-jump up or a double-jump down, so we check the current `y` position before we apply the change.

Now that we are going to use input to control our robot, we no longer need to set the initial velocity as we did in the previous chapter. Locate the following two lines of code in LoadTextures and delete them:

[PRE17]

Run the game. You should now be able to control the robot with the arrow keys, moving him left and right, up and down. Congratulations, you're a control freak!

# Graphical User Interface

It is now time to turn our attention to the graphical user interface, or GUI. The GUI allows us to control other elements of the game, such as starting or stopping the game, or setting various options.

In this section, you will learn how to create buttons on the screen that can be clicked by the mouse. We'll keep it simple by adding a single button to pause the game. While we are at it, we will learn important lessons about game state.

## Creating a button

A button is nothing more than a texture that is being displayed on the screen. However, we have to perform some special coding to detect whether or not the button is being clicked. We will add this functionality to the sprite class so that our buttons are being handled by the same class that handles other image in our game.

We will actually create two buttons: one to Pause and one to Resume. I have used a simple graphics program to create the following two buttons:

![Creating a button](img/8199OS_04_02.jpg)

I have saved these buttons as, you guessed it, `pause.png` and `resume.png` in the `resources` folder.

## Enhancing the Input class

In order to integrate UI into our existing `Input` class, we are going to have to add some additional features. We will add a dynamic array to the `Input` class to hold a list of UI elements that we need to check for input.

Start by adding the following line to the includes for `Input.h`:

[PRE18]

We need to include the `Sprite` class so that we can work with sprites in the `Input` class.

Next, we add a new command. Modify the `Command` enum so that it looks like the following list:

[PRE19]

We have added `CM_UI`, which will be set as the current command if any UI element is clicked.

Now, we define a member variable to hold the list of UI elements. Add this line of code to the member variables in `Input.h`:

[PRE20]

`m_uiElements` will be a dynamic list of pointers to our elements, while `m_uiCount` will keep track of the number of elements in the list.

The final change to `Input.h` is to add the following line in the public methods:

[PRE21]

## Adding UI elements to the list

We need to be able to add a list of elements to our `Input` class so that they can be checked during the input handling.

First, we have to allocate memory for our list of elements. Add the following lines to the `Input` constructor in `Input.cpp`:

[PRE22]

I could probably get cleverer than this, but for now, we will allocate enough memory to hold 10 UI elements. We then initialize `m_uiCount` to `0`. Now, we need to add the following method to `Input.cpp`:

[PRE23]

This method allows us to add a UI element to our list (internally, each UI element is a pointer to a sprite). We add the element to the `m_uiElements` array at the current index and then increment `m_uiCount`.

## Checking each UI element

Eventually, the Input class will contain a list of all UI elements that it is supposed to check. We will need to iterate through that list to see if any of the active elements have been clicked (if we want to ignore a particular element, we simply set its active flat to `false`).

Open `Input.cpp` and add the following code to `Update` above the existing code:

[PRE24]

This code iterates through each item in the `m_uiElements` array. If the element is active, then `CheckForClick` is called to see if this element has been clicked. If the element has been clicked, the `IsClicked` property of the element is set to `true` and `m_command` is set to `CM_UI`.

We put this code above the existing code because we want checking the UI to take priority over checking for game input. Notice in the preceding code that we exit the function if we find a UI element that has been clicked.

## Pushing your buttons

In order to see if an element has been clicked, we need to see if the left mouse button is down while the mouse pointer is inside the area bounded by the UI element.

First, open `Input.cpp` and add the following code:

[PRE25]

Here is what we are doing:

*   We first make sure that the left mouse button is down.
*   We need to store the current position of the mouse. To do this, we create a `POINT` called `cursorPosition`, then pass that by reference into `GetCursorPos`. This will set `cursorPosition` to the current mouse position in screen coordinates.
*   We actually need the mouse position in client coordinates (the actual area that we have to work with, ignoring windows borders and fluff). To get this, we pass `cursorPosition` along with a handle to the current window into `ScreenToClient`.
*   Now that we have the `cursorPosition`, want to test to see if it is inside the rectangle that bounds our UI element. We calculate the left, right, top, and bottom coordinates of the sprite.
*   Finally, we check to see if `cursorPosition` is within the boundaries of the UI element. If so, we return `true`; otherwise, we return `false`.

Ensure to add the following declaration to `Sprite.h`:

[PRE26]

## Adding our pauseButton

We now need to add the code to our game to create and monitor our pause and resume buttons.

First, we will add two variables for our two new sprites. Add the following two lines to the variable declaration block of `RoboRacer.cpp`:

[PRE27]

Then, add the following lines to `LoadTextures` (just before the `return` statement):

[PRE28]

This code sets up the pause and resume sprites exactly like we set up the other sprites in our game. Only the pause sprite is set to be active and visible.

You will notice one important addition: we add each sprite to the `Input` class with a call to `AddUiElement`. This adds the sprite to the list of UI elements that need to be checked for input.

We must also add code to the `Update` function in `RoboRacer.cpp`:

[PRE29]

Similarly, we must add code to the `Render` function in `RoboRacer.cpp` (just before the call to `SwapBuffers`):

[PRE30]

That's it! If you run the game now, you should see the new pause button in the upper-left corner. Unfortunately, it doesn't do anything yet (other than change the button from Pause to Resume. Before we can actually pause the game, we need to learn about state management.

# State management

Think about it. If we want our game to pause, then we have to set some kind of flag that tells the game that we want it to take a break. We could set up a Boolean:

[PRE31]

We would set `m_isPaused` to `true` if the game is paused, and set it to `false` if the game is running.

The problem with this approach is that there are a lot of special cases that we may run into in a real game. At any time the game might be:

*   Starting
*   Ending
*   Running
*   Paused

These are just some example of **game states**. A game state is a particular mode that requires special handling. As there can be so many states, we usually create a state manager to keep track of the state we are currently in.

## Creating a state manager

The simplest version of a state manager begins with an enum that defines all of the game states. Open `RoboRacer.cpp` and add the following code just under the include statements:

[PRE32]

Then go to the variable declarations block and add the following line:

[PRE33]

To keep things simple, we are going to define two states: running and paused. A larger game will have many more states.

Enums have a big advantage over Boolean variables. First, their purpose is generally clearer. Saying that the game state is `GS_Paused` or `GS_Running` is clearer than if we just had set a Boolean to `true` or `false`.

The other advantage is that enums can have more than two values. If we need to add another state to our game, it is as simple as adding another value to our `GameState` enum list.

Our game will start in the running state, so add the following line of code to the `StartGame` function:

[PRE34]

## Pausing the game

Think about it for a minute. What do we want to do when the game is paused? We still want to see things on the screen, so that means that we still want to make all of our Render calls. However, we don't want things to change position or animate. We also don't want to process game input, though we do need to handle UI input.

All of this should have you thinking about the update calls. We want to block updates to everything except the UI. Modify the `Update` function in `RoboRacer.cpp` so that it contains the following code:

[PRE35]

Notice that we will only process the sprite updates if the game state is `GS_Running`.

We are going to get ready to accept mouse input. First, we are going to setup a timer. Add the following code in the variable declarations of RoboRacer2d.cpp:

[PRE36]

Then add the line of code below to StartGame:

[PRE37]

The time will be used to add a small delay to mouse input. Without the delay, each click on the mouse would be registered several times instead of a single time.

We still need to handle input, but not all input. Go to the `ProcessInput` function in `RoboRacer.cpp` and make the following changes:

[PRE38]

Take a look at the second line. It sets the command to `CM_UI` if the game is paused. This means that only UI commands will be processed while the game is paused. A hack? Perhaps, but it gets the job done!

We only have two more changes to make. When the pause button is clicked, we need to change the game state to `GS_Paused`, and when the resume button is clicked, we need to change the game state to `GS_Running`. Those changes have already been made in the `CS_UI` case in the preceding code!

When you run the program now, you will see that the game pauses when you click the pause button. When you click the resume button, everything picks up again.

# Summary

Again, you have traveled far! We implemented a basic input class, then modified our sprite class to handle UI. This unified approach allows one class to handle sprites as game objects as well as sprites as part of the user interface. The same approach to see if a button has been pushed, can also be used for collision detection for a game object too. Then you learned how to create a state machine to handle the various states that the game may be in.

In the next chapter, we will learn to detect when game objects collide.