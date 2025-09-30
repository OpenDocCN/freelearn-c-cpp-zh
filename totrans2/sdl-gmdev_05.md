# Chapter 5. Handling Game States

When we first start up a game, we expect to see a splash screen showing any branding for publishers and developers, followed by a loading screen as the game does its initial setup. After this, we are usually faced with a menu screen; here, we can change settings and start the game. Starting the game leads us to another loading screen, possibly followed by a cut scene, and finally, we are in the game. When we are in the game, we can pause our play (allowing us to change any settings), exit the game, restart the level, and so on. If we fail the level, we are shown either an animation or a game over screen depending on how the game is set up. All of these different sections of a game are called *Game States*. It is very important that we make the transition between these states as easy as possible.

In this chapter we will cover:

*   Two different ways of handling states, starting with a really simple implementation and gradually building our framework implementation
*   Implementing **Finite State Machines** (**FSM**)
*   Adding states to the overall framework

# A simple way for switching states

One of the simplest ways to handle states is to load everything we want at the game's initialization stage, but only draw and update the objects specific to each state. Let's look at an example of how this could work. First, we can define a set of states we are going to use:

[PRE0]

We can then use the `Game::init` function to create the objects:

[PRE1]

Then, set our initial state:

[PRE2]

Next, we can change our `update` function to only use the things we want when in a specific state:

[PRE3]

The `render` function would do something similar. These functions could of course still loop through arrays and use polymorphism as we originally had done, but on a state-by-state basis. Changing states is as simple as changing the value of the `m_currentGameState` variable.

If you can see issues with this method, then it is very encouraging that you are starting to think in an object-oriented way. This way of updating states would be a bit of a nightmare to maintain and the scope for error is quite large. There are too many areas that need to be updated and changed to make this a viable solution for any game larger than a simple arcade game.

# Implementing finite state machines

What we really need is the ability to define our states outside the `game` class, and have the state itself take care of what it needs to load, render, and update. For this we can create what is known as an FSM. The definition of FSM, as we will use it, is a machine that can exist in a finite number of states, can exist in only one state at a time (known as the current state), and can change from one state to another (known as a transition).

## A base class for game states

Let's start our implementation by creating a base class for all of our states; create a header file called `GameState.h`:

[PRE4]

Just like our `GameObject` class, this is an abstract base class; we aren't actually putting any functionality into it, we just want all of our derived classes to follow this blueprint. The `update` and `render` functions are self-explanatory, as they will function just like the functions we created in the `Game` class. We can think of the `onEnter` and `onExit` functions as similar to other `load` and `clean` functions; we call the `onEnter` function as soon as a state is created and `onExit` once it is removed. The last function is a getter for the state ID; each state will need to define this function and return its own `staticconst` ID. The ID is used to ensure that states don't get repeated. There should be no need to change to the same state, so we check this using the state ID.

That's it for our `GameState` base class; we can now create some test states that derive from this class. We will start with a state called `MenuState`. Go ahead and create `MenuState.h` and `MenuState.cpp` in our project, open up `MenuState.h`, and start coding:

[PRE5]

We can now define these methods in our `MenuState.cpp` file. We will just display some text in the console window for now while we test our implementation; we will give this state an ID of `"MENU"`:

[PRE6]

We will now create another state called `PlayState`, create `PlayState.h` and `PlayState.cpp` in our project, and declare our methods in the header file:

[PRE7]

This header file is the same as `MenuState.h` with the only difference being `getStateID` returning this class' specific ID (`"PLAY"`). Let's define our functions:

[PRE8]

We now have two states ready for testing; we must next create our FSM so that we can handle them.

## Implementing FSM

Our FSM is going to need to handle our states in a number of ways, which include:

*   **Removing one state and adding another**: We will use this way to completely change states without leaving the option to return
*   **Adding one state without removing the previous state**: This way is useful for pause menus and so on
*   **Removing one state without adding another**: This way will be used to remove pause states or any other state that had been pushed on top of another one

Now that we have come up with the behavior we want our FSM to have, let's start creating the class. Create the `GameStateMachine.h` and `GameStateMachine.cpp` files in our project. We will start by declaring our functions in the header file:

[PRE9]

We have declared the three functions we need. The `pushState` function will add a state without removing the previous state, the `changeState` function will remove the previous state before adding another, and finally, the `popState` function will remove whichever state is currently being used without adding another. We will need a place to store these states; we will use a vector:

[PRE10]

In the `GameStateMachine.cpp` file, we can define these functions and then go through them step-by-step:

[PRE11]

This is a very straightforward function; we simply push the passed-in `pState` parameter into the `m_gameStates` array and then call its `onEnter` function:

[PRE12]

Another simple function is `popState`. We first check if there are actually any states available to remove, and if so, we call the `onExit` function of the current state and then remove it:

[PRE13]

Our third function is a little more complicated. First, we must check if there are already any states in the array, and if there are, we check whether their state ID is the same as the current one, and if it is, then we do nothing. If the state IDs do not match, then we remove the current state, add our new `pState`, and call its `onEnter` function. Next, we will add new `GameStateMachine` as a member of the `Game` class:

[PRE14]

We can then use the `Game::init` function to create our state machine and add our first state:

[PRE15]

The `Game::handleEvents` function will allow us to move between our states for now:

[PRE16]

When we press the *Enter* key, the state will change. Test the project and you should get the following output after changing states:

[PRE17]

We now have the beginnings of our FSM and can next add `update` and `render` functions to our `GameStateMachine` header file:

[PRE18]

We can define them in our `GameStateMachine.cpp` file:

[PRE19]

These functions simply check if there are any states, and if so, they update and render the current state. You will notice that we use `back()` to get the current state; this is because we have designed our FSM to always use the state at the back of the array. We use `push_back()` when adding new states so that they get pushed to the back of the array and used immediately. Our `Game` class will now use the FSM functions in place of its own `update` and `render` functions:

[PRE20]

Our FSM is now in place.

# Implementing menu states

We will now move on to creating a simple menu state with visuals and mouse handling. We will use two new screenshots for our buttons, which are available with the source code downloads:

![Implementing menu states](img/6821OT_05_01.jpg)

The following screenshot shows the exit feature:

![Implementing menu states](img/6821OT_05_02.jpg)

These are essentially sprite sheets with the three states of our button. Let's create a new class for these buttons, which we will call `MenuButton`. Go ahead and create `MenuButton.h` and `MenuButton.cpp`. We will start with the header file:

[PRE21]

By now this should look very familiar and it should feel straightforward to create new types. We will also define our button states as an enumerated type so that our code becomes more readable; put this in the header file under `private`:

[PRE22]

Open up the `MenuButton.cpp` file and we can start to flesh out our `MenuButton` class:

[PRE23]

The only thing really new in this class is the `update` function. Next, we will go through each step of this function:

*   First, we get the coordinates of the mouse pointer and store them in a pointer to a `Vector2D` object:

    [PRE24]

*   Now, check whether the mouse is over the button or not. We do this by first checking whether the mouse position is less than the position of the right-hand side of the button (*x position + width*). We then check if the mouse position is greater than the position of the left-hand side of the button (*x position*). The y-position check is essentially the same with *y position + height* and *y position* for bottom and top respectively:

    [PRE25]

*   If the previous check is true, we know that the mouse is hovering over our button; we set its frame to `MOUSE_OVER (1)`:

    [PRE26]

*   We can then check whether the mouse has been clicked; if it has, then we set the current frame to `CLICKED(2)`:

    [PRE27]

*   If the check is not true, then we know the mouse is outside the button and we set the frame to `MOUSE_OUT (0)`:

    [PRE28]

We can now test out our reusable button class. Open up our previously created `MenuState.hand`, which we will implement for real. First, we are going to need a vector of `GameObject*` to store our menu items:

[PRE29]

Inside the `MenuState.cpp` file, we can now start handling our menu items:

[PRE30]

The `onExit` and `onEnter` functions can be defined as follows:

[PRE31]

We use `TextureManager` to load our new images and then assign these textures to two buttons. The `TextureManager` class also has a new function called `clearFromTextureMap`, which takes the ID of the texture we want to remove; it is defined as follows:

[PRE32]

This function enables us to clear only the textures from the current state, not the entire texture map. This is essential when we push states and then pop them, as we do not want the popped state to clear the original state's textures.

Everything else is essentially identical to how we handle objects in the `Game` class. Run the project and we will have buttons that react to mouse events. The window will look like the following screenshot (go ahead and test it out):

![Implementing menu states](img/6821OT_05_03.jpg)

## Function pointers and callback functions

Our buttons react to rollovers and clicks but do not actually do anything yet. What we really want to achieve is the ability to create `MenuButton` and pass in the function we want it to call once it is clicked; we can achieve this through the use of function pointers. Function pointers do exactly as they say: they point to a function. We can use classic C style function pointers for the moment, as we are only going to use functions that do not take any parameters and always have a return type of `void` (therefore, we do not need to make them generic at this point).

The syntax for a function pointer is like this:

[PRE33]

We declare our function pointer as a private member in `MenuButton.h` as follows:

[PRE34]

We also add a new member variable to handle clicking better:

[PRE35]

Now we can alter the constructor to allow us to pass in our function:

[PRE36]

In our `MenuButton.cpp` file, we can now alter the constructor and initialize our pointer with the initialization list:

[PRE37]

The `update` function can now call this function:

[PRE38]

Note that this `update` function now uses the `m_bReleased` value to ensure we release the mouse button before doing the callback again; this is how we want our clicking to behave.

In our `MenuState.h` object, we can declare some functions that we will pass into the constructors of our `MenuButton` objects:

[PRE39]

We have declared these functions as static; this is because our callback functionality will only support static functions. It is a little more complicated to handle regular member functions as function pointers, so we will avoid this and stick to static functions. We can define these functions in the `MenuState.cpp` file:

[PRE40]

We can pass these functions into the constructors of our buttons:

[PRE41]

Test our project and you will see our functions printing to the console. We are now passing in the function we want our button to call once it is clicked; this functionality is great for our buttons. Let's test the exit button with some real functionality:

[PRE42]

Now clicking on our exit button will exit the game. The next step is to allow the `s_menuToPlay` function to move to `PlayState`. We first need to add a getter to the `Game.h` file to allow us to access the state machine:

[PRE43]

We can now use this to change states in `MenuState`:

[PRE44]

Go ahead and test; `PlayState` does not do anything yet, but our console output should show the movement between states.

## Implementing the temporary play state

We have created `MenuState`; next, we need to create `PlayState` so that we can visually see the change in our states. For `PlayState` we will create a player object that uses our `helicopter.png` image and follows the mouse around. We will start with the `Player.cpp` file and add the code to make the `Player` object follow the mouse position:

[PRE45]

First, we get the current mouse location; we can then get a vector that leads from the current position to the mouse position by subtracting the current position from the mouse position. We then divide the velocity by a scalar to slow us down a little and allow us to see our helicopter catch up to the mouse rather than stick to it. Our `PlayState.h` file will now need its own vector of `GameObject*`:

[PRE46]

Finally, we must update the `PlayState.cpp` implementation file to use our `Player` object:

[PRE47]

This file is very similar to the `MenuState.cpp` file, but this time we are using a `Player` object rather than the two `MenuButton` objects. There is one adjustment to our `SDLGameObject.cpp` file that will make `PlayState` look even better; we are going to flip the image file depending on the velocity of the object:

[PRE48]

We check whether the object's velocity is more than `0` (moving to the right-hand side) and flip the image accordingly. Run our game and you will now have the ability to move between `MenuState` and `PlayState` each with their own functionality and objects. The following screenshot shows our project so far:

![Implementing the temporary play state](img/6821OT_05_04.jpg)

## Pausing the game

Another very important state for our games is the pause state. Once paused, the game could have all kinds of options. Our `PauseState` class will be very similar to the `MenuState`, but with different button visuals and callbacks. Here are our two new screenshots (again available in the source code download):

![Pausing the game](img/6821OT_05_05.jpg)

The following screenshot shows the resume functionality:

![Pausing the game](img/6821OT_05_06.jpg)

Let's start by creating our `PauseState.h` file in the project:

[PRE49]

Next, create our `PauseState.cpp` file:

[PRE50]

In our `PlayState.cpp` file, we can now use our new `PauseState` class:

[PRE51]

This function listens for the *Esc* key being pressed, and once it has been pressed, it then pushes a new `PauseState` class onto the state array in FSM. Remember that `pushState` does not remove the old state; it merely stops using it and uses the new state. Once we are done with the pushed state, we remove it from the state array and the game continues to use the previous state. We remove the pause state using the resume button's callback:

[PRE52]

The main menu button takes us back to the main menu and completely removes any other states:

[PRE53]

## Creating the game over state

We are going to create one final state, `GameOverState`. To get to this state, we will use collision detection and a new `Enemy` object in the `PlayState` class. We will check whether the `Player` object has hit the `Enemy` object, and if so, we will change to our `GameOverState` class. Our Enemy object will use a new image `helicopter2.png`:

![Creating the game over state](img/6821OT_05_07.jpg)

We will make our `Enemy` object's helicopter move up and down the screen just to keep things interesting. In our `Enemy.cpp` file, we will add this functionality:

[PRE54]

We can now add an `Enemy` object to our `PlayState` class:

[PRE55]

Running the game will allow us to see our two helicopters:

![Creating the game over state](img/6821OT_05_08.jpg)

Before we cover collision detection, we are going to create our `GameOverState` class. We will be using two new images for this state, one for new `MenuButton` and one for a new type, which we will call `AnimatedGraphic`:

![Creating the game over state](img/6821OT_05_09.jpg)

The following screenshot shows the game over functionality:

![Creating the game over state](img/6821OT_05_10.jpg)

`AnimatedGraphic` is very similar to other types, so I will not go into too much detail here; however, what is important is the added value in the constructor that controls the speed of the animation, which sets the private member variable `m_animSpeed`:

[PRE56]

The `update` function will use this value to set the speed of the animation:

[PRE57]

Now that we have the `AnimatedGraphic` class, we can implement our `GameOverState` class. Create `GameOverState.h` and `GameOverState.cpp` in our project; the header file we will create should look very familiar, as given in the following code:

[PRE58]

Our implementation file is also very similar to other files already covered, so again I will only cover the parts that are different. First, we define our static variables and functions:

[PRE59]

The `onEnter` function will create three new objects along with their textures:

[PRE60]

That is pretty much it for our `GameOverState` class, but we must now create a condition that creates this state. Move to our `PlayState.h` file and we will create a new function to allow us to check for collisions:

[PRE61]

We will define this function in `PlayState.cpp`:

[PRE62]

This function checks for collisions between two `SDLGameObject` types. For the function to work, we need to add three new functions to our `SDLGameObject` class:

[PRE63]

The next chapter will deal with how this function works, but for now, it is enough to know that it does. Our `PlayState` class will now utilize this collision detection in its `update` function:

[PRE64]

We have to use a `dynamic_cast` object to cast our `GameObject*` class to an `SDLGameObject*` class. If `checkCollision` returns `true`, then we add the `GameOverState` class. The following screenshot shows the `GameOver` state:

![Creating the game over state](img/6821OT_05_11.jpg)

# Summary

This chapter has left us with something a lot more like a game than in previous chapters. We have created states for menus, pause, play, and game over with each state having its own functionality and being handled using FSM. The `Game` class now uses FSM to render and update game objects and it does not now handle objects directly, as each individual state handles its own objects. We have also created simple callback functions for our buttons using function pointers and static functions.