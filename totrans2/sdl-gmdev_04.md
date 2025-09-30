# Chapter 4. Exploring Movement and Input Handling

We have already covered drawing to the screen and how to handle objects but we have not had anything moving around very much yet. Getting input from the user and then controlling our game objects is one of the most important topics in game development. It can decide the feel and responsiveness of your game and is something that a user can really pick up on. In this chapter we will cover:

*   Cartesian coordinate systems
*   2D vectors
*   Creating variables to control the movement of a game object
*   Setting up a simple movement system
*   Setting up input handling from joysticks, keyboard, and mouse
*   Creating a fixed frame rate

# Setting up game objects for movement

In the previous chapter, we gave our objects x and y values which we could then use to pass into our drawing code. The x and y values we used can be represented using a Cartesian coordinate system.

![Setting up game objects for movement](img/6821OT_04_01.jpg)

The above figure shows a Cartesian coordinate system (flipped on the Y axis) with two coordinates. Representing them as (x,y) gives us position 1 as (3,3) and position 2 as (7,4). These values can be used to represent a position in 2D space. Imagine this figure as a zoomed in image of the top-left corner of our game window, with each of the grid squares representing one pixel of our game window. With this in mind, we can see how to use these values to draw things to the screen in the correct position. We now need a way to update these position values so that we can move our objects around. For this we will look at 2D vectors.

## What is a vector?

A **vector** can be described as an entity with a direction and a magnitude. We can use them to represent aspects of our game objects, for example, velocity and acceleration, that can be used to create movement. Taking velocity as an example, to fully represent the velocity of our objects, we need the direction in which they are travelling and also the amount (or magnitude) by which they are heading in that direction.

![What is a vector?](img/6821OT_04_02.jpg)

Let's define a couple of things about how we will use vectors:

*   We will represent a vector as v(x,y)

    We can get the length of a vector using the following equation:

    ![What is a vector?](img/6821OT_04_03.jpg)

The preceding figure shows the vector v1(3,-2) which will have a length of √(32+(-22)). We can use the x and y components of a vector to represent our object's position in 2D space. We can then use some common vector operations to move our objects. Before we move onto these operations let's create a vector class called `Vector2D` in the project. We can then look at each operation we will need and add them to the class.

[PRE0]

You can see that the `Vector2D` class is very simple at this point. We have our x and y values and a way to get and set them. We already know how to get the length of a vector, so let's create a function for this purpose:

[PRE1]

## Some common operations

Now since we have our basic class in place, we can start to gradually add some operations.

### Addition of two vectors

The first operation we will look at is the addition of two vectors. For this we simply add together the individual components of each vector.

![Addition of two vectors](img/6821OT_04_04.jpg)

Let's make use of overloaded operators to make it easy for us to add two vectors together:

[PRE2]

With these functions we can add two vectors together using the standard addition operators, for example:

[PRE3]

### Multiply by a scalar number

Another operation is to multiply a vector by a regular scalar number. For this operation we multiply each component of the vector by the scalar number:

![Multiply by a scalar number](img/6821OT_04_05.jpg)

We can again use overloaded operators to create these functions:

[PRE4]

### Subtraction of two vectors

Subtraction is very similar to addition.

![Subtraction of two vectors](img/6821OT_04_06.jpg)

Let's create some functions to do this for us:

[PRE5]

### Divide by a scalar number

By now I am sure you have noticed a pattern emerging and can guess how dividing a vector by a scalar will work, but we will cover it anyway.

![Divide by a scalar number](img/6821OT_04_07.jpg)

And our functions:

[PRE6]

### Normalizing a vector

We need another very important operation and that is the ability to normalize a vector. Normalizing a vector makes its length equal to 1\. Vectors with a length (magnitude) of 1 are known as unit vectors and are useful to represent just a direction, such as the facing direction of an object. To normalize a vector we multiply it by the inverse of its length.

![Normalizing a vector](img/6821OT_04_08.jpg)

We can create a new member function to normalize our vectors:

[PRE7]

Now that we have a few basic functions in place, let's start to use these vectors in our `SDLGameObject` class.

## Adding the Vector2D class

1.  Open up `SDLGameObject.h` and we can begin implementing the vectors. First we need to include the new `Vector2D` class.

    [PRE8]

2.  We also need to remove the previous `m_x` and `m_y` values and replace them with `Vector2D`.

    [PRE9]

3.  Now we can move to the `SDLGameObject.cpp` file and update the constructor.

    [PRE10]

4.  We now construct the `m_position` vector using the member initialization list and we must also use the `m_position` vector in our draw function.

    [PRE11]

5.  One last thing before we test is to use our vector in the `Enemy::update` function.

    [PRE12]

This function will use vector addition very soon, but for now we just add `1` to the current position to get the same behavior we already had. We can now run the game and we will see that we have implemented a very basic vector system. Go ahead and play around with the `Vector2D` functions.

## Adding velocity

We previously had to separately set the `x` and `y` values of our objects, but now that our position is a vector, we have the ability to add a new vector to it to update our movement. We will call this vector the velocity vector and we can think of it as the amount we want our object to move in a specific direction:

1.  The velocity vector can be represented as follows:![Adding velocity](img/6821OT_04_09.jpg)
2.  We can add this to our `SDLGameObject` update function as this is the way we update all derived objects. So first let's create the velocity member variable.

    [PRE13]

3.  We will construct it in the member initialization list as 0,0.

    [PRE14]

4.  And now we will move to the `SDLGameObject::update` function.

    [PRE15]

5.  We can test this out in one of our derived classes. Move to `Player.cpp` and add the following:

    [PRE16]

We set the `m_velocity` x value to 1\. This means that we will add `1` to our `m_position` x value each time the update function is called. Now we can run this to see our object move using the new velocity vector.

## Adding acceleration

Not all of our objects will move along at a constant velocity. Some games will require that we gradually increase the velocity of our object using acceleration. A car or a spaceship are good examples. No one would expect these objects to hit their top speed instantaneously. We are going to need a new vector for acceleration, so let's add this into our `SDLGameObject.h` file.

[PRE17]

Then we can add it to our `update` function.

[PRE18]

Now alter our `Player::update` function to set the acceleration rather than the velocity.

[PRE19]

After running our game you will see that the object gradually picks up speed.

# Creating fixed frames per second

Earlier in the book we put in an `SDL_Delay` function to slow everything down and ensure that our objects weren't moving too fast. We will now expand upon that by making our game run at a fixed frame rate. Fixed frames per second (FPS) is not necessarily always a good option, especially when your game includes more advanced physics. It is worth bearing this in mind when you move on from this book and start developing your own games. Fixed FPS will, however, be fine for the small 2D games, which we will work towards in this book.

With that said, let's move on to the code:

1.  Open up `main.cpp` and we will create a few constant variables.

    [PRE20]

2.  Here we define how many frames per second we want our game to run at. A frame rate of 60 frames per second is a good starting point as this is essentially synced up to the refresh rate of most modern monitors and TVs. We can then divide this by the number of milliseconds in a second, giving us the amount of time we need to delay the game between loops to keep our constant frame rate. We need another two variables at the top of our main function; these will be used in our calculations.

    [PRE21]

3.  We can now implement our fixed frame rate in our main loop.

    [PRE22]

First we get the time at the start of our loop and store it in `frameStart`. For this we use `SDL_GetTicks` which returns the amount of milliseconds since we called `SDL_Init`. We then run our game loop and store how long it took to run by subtracting the time our frame started from the current time. If it is less than the time we want a frame to take, we call `SDL_Delay` and make our loop wait for the amount of time we want it to, subtracting how long the loop already took to complete.

# Input handling

We have now got our objects moving based on velocity and acceleration, so next we must introduce some way of controlling this movement through user input. SDL supports a number of different types of user interface devices including joysticks, gamepads, mouse, and keyboard, all of which will be covered in this chapter, along with how to add them into our framework implementation.

## Creating our input handler class

We will create a class that handles all device input, whether it is from controllers, keyboard, or mouse. Let's start with a basic class and build from there. First we need a header file, `InputHandler.h`.

[PRE23]

This is our singleton `InputHandler`. So far we have an `update` function which will poll for events and update our `InputHandler` accordingly, and a clean function which will clear any devices we have initialized. As we start adding device support we will flesh this out a lot more.

## Handling joystick/gamepad input

There are tons of joysticks and gamepads out there, often with different amounts of buttons and analog sticks amongst other things. PC game developers have a lot to do when trying to support all of these different gamepads. SDL has good support for joysticks and gamepads, so we should be able to come up with a system that would not be difficult to extend for different gamepad support.

### SDL joystick events

There are a few different structures for handling joystick events in SDL. The table below lists each one and their purpose.

| SDL joystick event | Purpose |
| --- | --- |
| `SDL_JoyAxisEvent` | Axis motion information |
| `SDL_JoyButtonEvent` | Button press and release information |
| `SDL_JoyBallEvent` | Trackball event motion information |
| `SDL_JoyHatEvent` | Joystick hat position change |

The events we are most interested in are the axis motion and the button press events. Each of these events also has an enumerated type that we can check for in our event loop to ensure we are only handling the events we want to handle. The table below shows the type value for each of the above events.

| SDL joystick event | Type value |
| --- | --- |
| `SDL_JoyAxisEvent` | `SDL_JOYAXISMOTION` |
| `SDL_JoyButtonEvent` | `SDL_JOYBUTTONDOWN` or `SDL_JOYBUTTONUP` |
| `SDL_JoyBallEvent` | `SDL_JOYBALLMOTION` |
| `SDL_JoyHatEvent` | `SDL_JOYHATMOTION` |

### Note

It's a good idea to use the **Joystick Control Panel** property in Windows or **JoystickShow** on OSX to find out which button numbers you will need to use in SDL for a specific button. These applications are invaluable for finding out things about your joystick/gamepad so you can support them properly.

The code we will put in place will assume we are using a Microsoft Xbox 360 controller (which can be used on PC or OSX), as this is an extremely popular controller for PC gaming. Other controllers, such as the PS3 controller, could possibly have different values for buttons and axes. The Xbox 360 controller consists of the following:

*   Two analog sticks
*   Analog sticks press as buttons
*   Start and Select buttons
*   Four face buttons: A, B, X, and Y
*   Four triggers: two digital and two analog
*   A digital directional pad

### Initializing joysticks

1.  To use gamepads and joysticks in SDL we first need to initialize them. We are going to add a new public function to the `InputHandler` class. This function will find out how many joysticks SDL has access to and then initialize them.

    [PRE24]

2.  We will also declare some private member variables that we will need.

    [PRE25]

3.  The `SDL_Joystick*` is a pointer to the joystick we will be initializing. We won't actually need these pointers when using the joysticks, but we do need to close them after we are done, so it is helpful for us to keep a list of them for later access. We will now define our `initialiseJoysticks` function and then go through it.

    [PRE26]

4.  Let's go through this line-by-line. First we check whether the joystick subsystem has been initialized using `SDL_WasInit`. If it has not been initialized we then initialize it using `SDL_InitSubSystem`.

    [PRE27]

5.  Next is the opening of each available joystick. Before we attempt to open the objects, we use `SDL_NumJoysticks` to make sure there are some joysticks available. We can then loop through the number of joysticks, opening them in turn with `SDL_JoystickOpen`. They can then be pushed into our array for closing later.

    [PRE28]

6.  Finally, we tell SDL to start listening for joystick events by enabling `SDL_JoystickEventState`. We also set our `m_bJoysticksEnabled` member variable according to how our initialization went.

    [PRE29]

7.  So, we now have a way to initialize our joysticks. We have two other functions to define, the `update` and `clean` functions. The `clean` function will loop through our `SDL_Joystick*` array and call `SDL_JoystickClose` on each iteration.

    [PRE30]

8.  The `update` function will be called in each frame in the main game loop to update the event state. For now though it will simply listen for a quit event and call the game's `quit` function (this function simply calls `SDL_Quit()`).

    [PRE31]

9.  Now we will use this `InputHandler` in our `Game` class functions. First we call `initialiseJoysticks` in the `Game::init` function.

    [PRE32]

    And we will update it in the `Game::handleEvents` function, clearing out anything we had before:

    [PRE33]

10.  The `clean` function can also be added to our `Game::clean` function.

    [PRE34]

11.  We can now plug in a pad or joystick and run the build. If everything is working according to plan we should get the following output, with `x` being the number of joysticks you have plugged in:

    [PRE35]

12.  Ideally we want to easily use one or more controllers with no change to our code. We already have a way to load in and open as many controllers that are plugged in, but we need to know which event corresponds to which controller; we do this using some information stored in the event. Each joystick event will have a `which` variable stored within it. Using this will allow us to find out which joystick the event came from.

    [PRE36]

## Listening for and handling axis movement

We are not going to handle the analog sticks in an analog way. Instead they will be handled as digital information, that is, they are either on or off. Our controller has four axes of motion, two for the left analog stick and two for the right.

We will make the following assumptions about our controller (you can use an external application to find out the specific values for your controller):

*   Left and right movement on stick one is axis 0
*   Up and down movement on stick one is axis 1
*   Left and right movement on stick two is axis 3
*   Up and down movement on stick two is axis 4

The Xbox 360 controller uses axes 2 and 5 for the analog triggers. To handle multiple controllers with multiple axes we will create a vector of pairs of `Vector2D*`, one for each stick.

[PRE37]

We use the `Vector2D` values to set whether a stick has moved up, down, left, or right. Now when we initialize our joysticks we need to create a pair of `Vector2D*` in the `m_joystickValues` array.

[PRE38]

We need a way to grab the values we need from this array of pairs; we will declare two new functions to the `InputHandler` class:

[PRE39]

The `joy` parameter is the identifier (ID) of the joystick we want to use, and the stick is 1 for the left stick and 2 for the right stick. Let's define these functions:

[PRE40]

So we grab the x or y value based on the parameters passed to each function. The `first` and `second` values are the first or second objects of the pair in the array, with `joy` being the index of the array. We can now set these values accordingly in the event loop.

[PRE41]

That is a big function! It is, however, relatively straightforward. We first check for an `SDL_JOYAXISMOTION` event and we then find out which controller the event came from using the `which` value.

[PRE42]

From this we know which joystick the event came from and can set a value in the array accordingly; for example:

[PRE43]

First we check the axis the event came from:

[PRE44]

If the axis is 0 or 1, it is the left stick, and if it is 3 or 4, it is the right stick. We use `first` or `second` of the pair to set the left or right stick. You may also have noticed the `m_joystickDeadZone` variable. We use this to account for the sensitivity of a controller. We can set this as a constant variable in the `InputHandler` header file:

[PRE45]

The value `10000` may seem like a big value to use for a stick at rest, but the sensitivity of a controller can be very high and so requires a value as large as this. Change this value accordingly for your own controllers.

Just to solidify what we are doing here, let's look closely at one scenario.

[PRE46]

If we get to the second if statement, we know that we are dealing with a left or right movement event on the left stick due to the axis being 0\. We have already set which controller the event was from and adjusted `whichOne` to the correct value. We also want `first` of the pair to be the left stick. So if the axis is 0, we use the `first` object of the array and set its x value, as we are dealing with an x movement event. So why do we set the value to 1 or -1? We will answer this by starting to move our `Player` object.

Open up `Player.h` and we can start to use our `InputHandler` to get events. First we will declare a new private function:

[PRE47]

Now in our `Player.cpp` file we can define this function to work with the `InputHandler`.

[PRE48]

Then we can call this function in the `Player::update` function.

[PRE49]

Everything is in place now, but first let's go through how we are setting our movement.

[PRE50]

Here, we first check whether `xvalue` of the left stick is more than 0 (that it has moved). If so, we set our `Player` x velocity to be the speed we want multiplied by `xvalue` of the left stick, and we know this is either 1 or -1\. As you will know, multiplying a positive number by a negative number results in a negative number, so multiplying the speed we want by -1 will mean we are setting our x velocity to a minus value (move left). We do the same for the other stick and also the y values. Build the project and start moving your `Player` object with a gamepad. You could also plug in another controller and update the `Enemy` class to use it.

## Dealing with joystick button input

Our next step is to implement a way to handle button input from our controllers. This is actually a lot simpler than handling axes. We need to know the current state of each button so that we can check whenever one has been pressed or released. To do this, we will declare an array of Boolean values, so each controller (the first index into the array) will have an array of Boolean values, one for each button on the controller.

[PRE51]

We can grab the current button state with a function that looks up the correct button from the correct joystick.

[PRE52]

The first parameter is the index into the array (the joystick ID), and the second is the index into the buttons. Next we are going to have to initialize this array for each controller and each of its buttons. We will do this in the `initialiseJoysticks` function.

[PRE53]

We use `SDL_JoystickNumButtons` to get the number of buttons for each of our joysticks. We then push a value for each of these buttons into an array. We push `false` to start, as no buttons are pressed. This array is then pushed into `our m_buttonStates` array to be used with the `getButtonState` function. Now we must listen for button events and set the value in the array accordingly.

[PRE54]

When a button is pressed (`SDL_JOYBUTTONDOWN`) we get to know which controller it was pressed on and use this as an index into the `m_buttonStates` array. We then use the button number (`event.jbutton.button`) to set the correct button to `true`; the same applies when a button is released (`SDL_JOYBUTTONUP`). That is pretty much it for button handling. Let's test it out in our Player class.

[PRE55]

Here we are checking if button 3 has been pressed (Yellow or Y on an Xbox controller) and setting our velocity if it has. That is everything we will cover about joysticks in this book. You will realize that supporting many joysticks is very tricky and requires a lot of tweaking to ensure each one is handled correctly. However, there are ways through which you can start to have support for many joysticks; for example, through a configuration file or even by the use of inheritance for different joystick types.

## Handling mouse events

Unlike joysticks, we do not have to initialize the mouse. We can also safely assume that there will only be one mouse plugged in at a time, so we will not need to handle multiple mouse devices. We can start by looking at the available mouse events that SDL covers:

| SDL Mouse Event | Purpose |
| --- | --- |
| `SDL_MouseButtonEvent` | A button on the mouse has been pressed or released |
| `SDL_MouseMotionEvent` | The mouse has been moved |
| `SDL_MouseWheelEvent` | The mouse wheel has moved |

Just like the joystick events, each mouse event has a type value; the following table shows each of these values:

| SDL Mouse Event | Type Value |
| --- | --- |
| `SDL_MouseButtonEvent` | `SDL_MOUSEBUTTONDOWN` or `SDL_MOUSEBUTTONUP` |
| `SDL_MouseMotionEvent` | `SDL_MOUSEMOTION` |
| `SDL_MouseWheelEvent` | `SDL_MOUSEWHEEL` |

We will not implement any mouse wheel movement events as most games will not use them.

### Using mouse button events

Implementing mouse button events is as straightforward as joystick events, more so even as we have only three buttons to choose from: left, right, and middle. SDL numbers these as 0 for left, 1 for middle, and 2 for right. In our `InputHandler` header, let's declare a similar array to the joystick buttons, but this time a one-dimensional array, as we won't handle multiple mouse devices.

[PRE56]

Then in the constructor of our `InputHandler` we can push our three mouse button states (defaulted to `false`) into the array:

[PRE57]

Back in our header file, let's create an `enum` attribute to help us with the values of the mouse buttons. Put this above the class so that other files that include our `InputHandler.h` header can use it too.

[PRE58]

Now let's handle mouse events in our event loop:

[PRE59]

We also need a function to access our mouse button states. Let's add this public function to the `InputHandler` header file:

[PRE60]

That is everything we need for mouse button events. We can now test it in our `Player` class.

[PRE61]

### Handling mouse motion events

Mouse motion events are very important, especially in big 3D first or third person action titles. For our 2D games, we might want our character to follow the mouse as a way to control our objects, or we might want objects to move to where the mouse was clicked (for a strategy game perhaps). We may even just want to know where the mouse was clicked so that we can use it for menus. Fortunately for us, mouse motion events are relatively simple. We will start by creating a private `Vector2D*` in the header file to use as the position variable for our mouse:

[PRE62]

Next, we need a public accessor for this:

[PRE63]

And we can now handle this in our event loop:

[PRE64]

That is all we need for mouse motion. So let's make our `Player` function follow the mouse position to test this feature:

[PRE65]

Here we have set our velocity to a vector from the player's current position to the mouse position. You can get this vector by subtracting the desired location from the current location; we already have a vector subtract overloaded operator so this is easy for us. We also divide the vector by 100; this just dampens the speed slightly so that we can see it following rather than just sticking to the mouse position. Remove the `/` to have your object follow the mouse exactly.

## Implementing keyboard input

Our final method of input, and the simplest of the three, is keyboard input. We don't have to handle any motion events, we just want the state of each button. We aren't going to declare an array here because SDL has a built-in function that will give us an array with the state of every key; 1 being pressed and 0 not pressed.

[PRE66]

The `numkeys` parameter will return the number of keys available on the keyboard (the length of the `keystate` array). So in our `InputHandler` header we can declare a pointer to the array that will be returned from `SDL_GetKeyboardState`.

[PRE67]

When we update our event handler we can also update the state of the keys; put this at the top of our event loop.

[PRE68]

We will now need to create a simple function that checks whether a key is down or not.

[PRE69]

This function takes `SDL_SCANCODE` as a parameter. The full list of `SDL_SCANCODE` values can be found in the SDL documentation at [http://wiki.libsdl.org/moin.cgi](http://wiki.libsdl.org/moin.cgi).

We can test the keys in our `Player` class. We will use the arrow keys to move our player.

[PRE70]

We now have key handling in place. Test as many keys as you can and look up the `SDL_Scancode` for the keys you are most likely to want to use.

## Wrapping things up

We have now implemented all of the devices we are going to handle, but at the moment our event loop is in a bit of a mess. We need to break it up into more manageable chunks. We will do this with the use of a switch statement for event types and some private functions, within our `InputHandler`. First let's declare our functions in the header file:

[PRE71]

We pass in the event from the event loop into each function (apart from keys) so that we can handle them accordingly. We now need to create our switch statement in the event loop.

[PRE72]

As you can see, we now break up our event loop and call the associated function depending on the type of the event. We can now split all our previous work into these functions; for example, we can put all of our mouse button down handling code into the `onMouseButtonDown` function.

[PRE73]

The rest of the code for the `InputHandler` is available within the source code downloads.

# Summary

We have covered some complicated material in this chapter. We have looked at a small amount of vector mathematics and how we can use it to move our game objects. We've also covered the initialization and the use of multiple joysticks and axes and the use of a mouse and a keyboard. Finally, we wrapped everything up with a tidy way to handle our events.