# Chapter 2. Responding to Player Inputs

Games by their very nature are interactive. They can respond to user input, unlike movies which play out exactly the same every time. So, you need a way to detect and respond to the user's input via mouse, keyboard, or gamepad. How can we accomplish this in our game? There are two options we can use for this: **DirectInput** and **XInput**.

In this chapter, we will cover the following topics:

*   DirectInput versus XInput
*   Mouse and keyboard input
*   Using joysticks with DirectInput
*   Using joysticks with XInput

# DirectInput versus XInput

So, which of these two options should we use? The answer is possibly both. Why both, you ask? The reason is that we need to use DirectInput to support older input devices that don't support the new features of XInput. An **input device** is any device that the user uses to provide input to the game, such as a mouse, keyboard, gamepad, and steering wheel.

We could just use DirectInput, but this approach has some limitations. DirectInput can handle XInput devices, but the left and right trigger buttons on a gamepad will be treated as a single button. One trigger is treated as the positive direction and the other as the negative direction. So, the two triggers are treated together as a single axis. DirectInput also does not support XInput vibration effects, and you cannot query for headset devices. On the flip side, we could just use XInput but then people with older DirectInput devices would not be able to play our game with them.

To support these features of newer input devices, we will need to use XInput along with DirectInput. This allows people with XInput devices to take advantage of their new features, while at the same time allowing users with older DirectInput devices to still be able to play our game with them. The players will undoubtedly appreciate this. So, what is the true difference between DirectInput and XInput? XInput is geared specifically towards Xbox 360 controllers and specialized Xbox 360 controllers, such as guitars. XInput supports a maximum of four controllers, each with no more than four axes, 10 buttons, two triggers, and an eight-directional digital pad.

XInput only really supports *next generation* controllers, and it does not support keyboards or mouse-type devices. DirectInput on the other hand is for all controllers and supports controllers with up to eight axes and 128 buttons.

So, the true answer to the question of whether to use DirectInput, XInput, or both, truly depends on the game you are developing. Just be aware that Microsoft essentially forces us to use XInput if we want to support some features of Xbox 360 and similar controllers in a PC game, as discussed earlier.

# Mouse and keyboard input

Even though players can play games using gamepads and other types of controllers, mouse and keyboard input are still quite important in PC game development. Some games have too many commands to map all of them on a gamepad, for example. When we assign an in-game action to a specific button on a keyboard, mouse, or gamepad, we say that we have mapped that action to that particular button. This is also sometimes referred to as binding, because we are, in effect, binding a certain key or button to a specific in-game action.

Let's first implement our mouse and keyboard input. Start Visual Studio and open the solution we worked on in the previous chapter. We are going to add a new class that will handle user input for us. Right-click on the `SlimFramework` project in the **Solution Explorer** pane and add a new class named `UserInput.cs`. We will make this class implement the `IDisposable` interface, just like we did with our `GameWindow.cs` class in [Chapter 1](ch01.html "Chapter 1. Getting Started"), *Getting Started*. So, we need to change the class declaration from `public class UserInput` to `public class UserInput : IDisposable`.

We also need to add two `using` statements to the top of this class file. One for DirectInput and one for XInput:

[PRE0]

Now, we are ready to set up the member variables for our new user input class. We'll create a member variables section just like we did in [Chapter 1](ch01.html "Chapter 1. Getting Started"), *Getting Started*. Here is the code:

[PRE1]

We have a handful of member variables here. The first one is `m_IsDisposed`, which has the same purpose as the `m_IsDisposed` member variable that we created in our `GameWindow` class. The second variable, `m_DirectInput`, will hold our DirectInput object.

Next, we have a group of three variables. The first one, `m_Keyboard`, holds the keyboard object. The next two keep track of the current and previous state of the keyboard. So, `m_KeyboardStateCurrent` holds the keyboard state for the current frame while `m_KeyboardStateLast` holds the keyboard state from the previous frame. Why do we need both? This is necessary, for example, if you want to detect whether or not the user is holding down a key, rather than simply pressing it.

Next, we have a set of three very similar variables for our mouse object and our current and previous mouse state (`m_Mouse`, `m_MouseStateCurrent`, and `m_MouseStateLast`).

## The constructor

Now, we need to create our constructor to initialize our user input object. Here is the code to do so:

[PRE2]

The first line calls the `InitDirectInput()` method to initialize DirectInput for us. We will create this method in a second, but first we need to finish looking at the `UserInput()` method. The next two lines initialize our keyboard state variables with the empty `KeyboardState` objects. This is necessary to prevent a crash that would occur if the program tries to access these variables on the first frame (when they would be uninitialized, and therefore null, which would result in a **Null Reference** exception). This type of exception occurs when the program tries to access a variable that is null. You can't use an object before you initialize it, after all!

The last two lines do exactly the same thing, but this time for our mouse state variables.

## Initializing DirectInput

Now that our constructor is done, we need to create our `InitDirectInput()` method. It is a pretty short method, and here is the code:

[PRE3]

This method only has three lines of code at the moment. The first one creates and initializes our DirectInput object and stores it in our `m_DirectInput` member variable. The second line creates and initializes our keyboard object, storing it in our `m_Keyboard` member variable. The third line does the same thing, but for our mouse object, storing it in our `m_Mouse` member variable.

The fact that this method is short as it is, owes itself to SlimDX helping us out. If you were to write this same code in C++ and without SlimDX, it would be much longer and also a bit more cryptic. This is one of the things that makes SlimDX a great framework to work with. It takes care of some stuff behind the scenes for us, while still allowing us to leverage the full power of DirectX.

## The Update() method

Now, we are going to add an `Update()` method to our user input class. This method will be called once per frame to get the latest user input data. We will be calling this method from the `UpdateScene()` method in our `GameWindow` class. Here is the code:

[PRE4]

The first two lines of code reacquire the keyboard and mouse devices in case another application has taken control of them since the previous frame. We have to acquire the mouse and keyboard devices so that our program has access to them. As long as the device is acquired, DirectInput makes its data available to our program. Acquiring the device is not permanent however, which is why we do it at the beginning of the `UpdateScene()` method. This ensures that we have access to the keyboard and mouse devices before we try to use them in the next lines of code.

So, why is this acquisition mechanism needed? Firstly, DirectInput needs a way to tell our application if the flow of data from a device has been interrupted by the system. This would happen, for example, if the user switches to another application window using *Alt* + *Tab* and uses the same input device in that application.

The second reason this acquisition mechanism is needed is because our program can change the properties of a device. DirectInput requires us to release the device before changing its properties. This is done by calling its `Unacquire()` method. Then you would reacquire it once you've finished changing its properties. This ensures that the device is not being used when we're changing its properties as this could cause serious problems. Note that there is one exception to this rule, which is that you can change the gain of a force feedback device while it is acquired.

Back to our code. The next two lines update our keyboard state variables. First, the keyboard state that was current for the previous frame is copied from the `m_KeyboardStateCurrent` member variable into the `m_KeyboardStateLast` member variable. Then, we get the current keyboard state and store it in our `m_KeyboardStateCurrent` member variable.

The last two lines do the same thing, but with our mouse state member variables.

## The IDisposable interface

As you'll recall from earlier in this chapter, we changed the declaration of the `UserInput` class to make it implement the `IDisposable` interface. We covered this interface in [Chapter 1](ch01.html "Chapter 1. Getting Started"), *Getting Started*.

As you may recall, we must implement two methods. The `public void Dispose()` method is identical to the one we created in our `GameWindow` class. So, I will not show it here. On the other hand, the `protected void Dispose(bool)` method is different. Here is its code:

[PRE5]

As you can see, the internal structure of this method is identical to the one we created in the `GameWindow` class. It has the same `if` statements inside it. The difference is that this time, we don't have an event to unhook, and we've added code to dispose of our DirectInput, keyboard, and mouse objects in the managed resources section of this method.

So, why is each of these objects disposed of inside its own little `if` statement? The reason for this is to prevent a potential crash that would happen if one of these objects is for some reason null. So, we check to see if the object is null. If it is not, then we dispose of it. Calling dispose on an object that is null will cause a Null Reference exception.

Now, we just have a few properties to add to our user input class. They are all very simple, and they just provide access to our member variables. Here are two of these properties. Check out the downloadable code for this chapter to see all of them.

[PRE6]

With this class now finished, we just need to modify our `GameWindow` class to make use of it now.

## Updating the GameWindow class

The first thing we need to do now is add a `using` statement to the top of the `GameWindow.cs` file:

[PRE7]

This will allow us to use the `Key` enumeration to specify which keys we want to check. Next, we need to add a new member variable to our `GameWindow` class. This variable will be called `m_UserInput` and it will contain our new `UserInput` object that we just finished creating. The declaration of this member variable looks like the following code:

[PRE8]

Next, we need to modify our constructor to create and initialize our user input object. To accomplish this, we simply add the following line of code to the end of our constructor, just above the closing `}`:

[PRE9]

It is a good idea to add some member methods to our `UserInput` class to make handling user input a bit simpler for us. So, let's create a new method named `IsKeyPressed()`, which looks like the following code:

[PRE10]

This method checks if the specified key is pressed, and returns `true` if it is or `false` if it is not. As you can see from the code in this method, the `KeyboardState` object has the `IsPressed()` method that we use to see if the specified key is pressed. It also has an `IsReleased()` method for testing if a key is not pressed. In addition to these, it has `PressedKeys` and `ReleasedKeys` properties that return a list of the currently pressed keys and currently not pressed keys respectively. And lastly, it has the `AllKeys` property that gives you the states of all keys on the keyboard.

### Note

The downloadable code for this chapter contains some additional keyboard handling methods like this one. They are `IsKeyReleased()` and `IsKeyHeldDown()`.

There is now just one step left before we can see our keyboard input code in action. We need to add some code into our `UpdateScene()` method to check for some key presses. Here is the new code in the `UpdateScene()` method:

[PRE11]

This code adds some basic keyboard commands to our window. The first `if` statement checks to see if the user is holding down the *Return* key along with either the left or right *Alt* key. If this is the case, then the `if` statement calls the `ToggleFullscreen()` method.

The `else if` clause checks to see if the user is pressing the *Esc* key. If so, then we close the game window, and the program terminates.

Before we can test run the program, we need to add a single line of code into the `GameWindow` class' `protected void Dispose(bool)` method. We need to add the following line of code into the managed resources section of the function:

[PRE12]

With that done, we can now test run the program. The game window looks identical to the way it did in the figure *The game window in action* in [Chapter 1](ch01.html "Chapter 1. Getting Started"), *Getting Started*. However, you can now close it by pressing the *Esc* key.

If you press *Enter* + *Alt*, nothing will happen at the moment. As mentioned in the previous chapter, we can't toggle fullscreen mode yet since we are not using DirectX's graphics APIs yet. **Application Programming Interface** (**API**) simply refers collectively to all of the public methods and types that are made available by the API. For example, SlimDX is an API, as is DirectX.

Smaller parts of an API can sometimes be considered as APIs in their own right as well. For example, DirectX's DirectInput is an API in and of itself. DirectX is more like a collection of several different APIs for different purposes, as is SlimDX.

As you can see, keyboard input is fairly simple to implement with SlimDX. Mouse input, though we haven't really used any yet, is just as simple. Responding to mouse input is almost identical to doing so for keyboard input. Simply check the `X` and `Y` properties of the `MouseState` object to find out the mouse cursor's position. The `Z` property allows you to detect movement of the mouse's scroll wheel if it has one. If your mouse does not have a scroll wheel, then this property will simply return `0`. Note that the value of the `Z` property is a delta, or in other words it is the amount that the scroll wheel has moved since the last update. Lastly, you can use the `IsPressed()` and `IsReleased()` methods to detect if a given mouse button is pressed or released.

Note that the downloadable code for this chapter also includes mouse handling methods added into our `UserInput` class. These are `IsMouseButtonPressed()`, `IsMouseButtonReleased()`, `IsMouseButtonHeldDown()`, `MouseHasMoved()`, and `MousePosition()`, among others. The `IsMouseButtonHeld()` method can be used to implement clicking-and-dragging behavior while the `HasMouseMoved()` method returns `true` if the mouse has moved since the previous frame, or `false` otherwise.

# Using joysticks with DirectInput

Now, let's shift gears and take a look at using **joysticks**. In this book, we will use the term joystick to refer to any game controller. First, we will look at how to use joysticks with DirectInput.

## Enumerating devices

You've probably seen some games that let you choose which game controller you want to use if you have more than one attached to your PC. In this section, we are going to look at how to get the list of available devices. With SlimDX, it is actually quite easy.

The `DirectInput` object (remember that we stored it in our `m_DirectInput` member variable) has a method named `GetDevices()`. To get a list of the available controllers, we would call that method like this:

[PRE13]

To try this out, let's add a new method to our `UserInput.cs` class. This method will simply write some debug output about the available devices. Here is the code:

[PRE14]

First, we create a variable named `deviceList`, get the list of game controllers, and store it in this new variable. For the first parameter of the `GetDevices()` method, we pass in the value `DeviceClass.GameController` to tell it that we are only interested in game controllers. For the second parameter, we give it the value `DeviceEnumerationFlags.AttachedOnly` because we only want devices that are actually installed and connected to the PC.

Next, we have an `if` statement that checks to see if the list of game controllers is empty. If so, it prints a debug message to let you know that no game controllers are connected to your computer. In the `else` clause of this `if` statement, we have a `foreach` loop that iterates through the list of game controllers that we just retrieved and stored in the `deviceList` variable. Inside the `foreach` loop, we have a single line of code. This line simply writes a single line of debug output into Visual Studio's **Output** pane for each game controller in the list. The **Output** pane is generally found at the bottom of the Visual Studio window. You may have to click on the **Output** tab in the lower-left corner of the window to display it if autohide is on. You can also access it by going to the **View** menu and selecting **Output**.

By default, Visual Studio automatically displays the **Output** pane while you are running your program so that you can see your program's debug output, as shown in the following screenshot. If it does not show the **Output** pane, see the preceding paragraph for how to access it.

Next, go to the `InitDirectInput()` method and add the following line of code to the end of the function:

[PRE15]

This makes a call to our new `GetJoysticks()` method at the end of the constructor. If you run this code now, you will see a list of game controllers displayed in Visual Studio's **Output** pane. The following screenshot shows what this looks like on my system, where I have one game controller connected to the computer:

![Enumerating devices](img/7389OS_02_01.jpg)

The Output pane showing our Debug output

### Note

Your output from this code will most likely be different than mine. So, you will probably see a different list of controllers to what I have, so your output will likely differ from that shown in the preceding screenshot.

# Getting input from the joystick

This is all well and good, but we still can't get input from a joystick. So let's look at that now. First, we need to add three member variables for our joystick, just like we did for the mouse and keyboard. Here are the three new member variables we need to add to our `UserInput.cs` class:

[PRE16]

As before, we have a variable to hold our device object (in this case, a `Joystick` object), and two more variables to hold the joystick state for the current frame and for the previous frame.

Now, we need to add two lines at the bottom of our constructor to initialize the joystick state variables. As discussed earlier in this chapter, this prevents a crash from potentially happening. Add these two lines at the end of the constructor:

[PRE17]

Now, let's modify our `GetJoysticks()` method. We will simply make it use the first joystick in the returned list of controllers. Here is the new code for the `GetJoysticks()` method:

[PRE18]

As you can see, we've also added a second line inside the `if` statement. This sets the minimum and maximum possible values for each axis on our game controller. In this case, we set it to `-1,000` and `1,000`. This means when the joystick is all the way to the left, its horizontal position is `-1,000`. When it is all the way to the right, its horizontal position is `1,000`. The same is true for the vertical axis. When the joystick is centered, its position will be (0,0). It is important to know the range of possible values to make our controls work correctly. You can get the range from the `Joystick.Properties.LowerRange` and `Joystick.Properties.UpperRange` properties. Note that these properties can throw an exception in some cases depending on your game controller's drivers and your DirectX version.

Now, we need to add a couple of lines of code into our `Update()` method to get the latest joystick data. To do this, we first need to add a line at the beginning of this method to acquire the joystick. You can't use a device without acquiring it first (see the *Mouse and keyboard input* section of this chapter for information on acquisition and why we need to do it). So, we will add the following line of code to acquire the joystick for us:

[PRE19]

We are basically letting the system know that we wish to use the joystick now and get access to it. Now that we have gotten the access to the joystick, we need to add these two lines at the end of the `Update()` method:

[PRE20]

As we did with the mouse and keyboard, we do with our `Joystick` object too. We take the value of the `m_Joy1StateCurrent` member variable and copy it into the `m_Joy1StateLast` variable since this state data is now one frame old. Then we get the current joystick state and store it in the `m_Joy1StateCurrent` member variable.

Our user input class now supports the use of one joystick. You could support more by adding variables and code for the second joystick the same way we did for this first joystick. Now, let's add some test code at the end of the `Update()` method to see this in action:

[PRE21]

### Tip

If you don't have a game controller, then you won't see any of the output from the previous code. The program would still work, but there would not be any debug output since there is no game controller to get it from.

This test code is a simple group of `if` statements. The first `if` statement checks if button `0` is pressed. If so, it writes a line of debug output to show you that it has detected the button press. The second `if` statement checks if button `1` is pressed, and if so, writes a debug message saying so. And the last two `if` statements do the same for buttons `2` and `3`.

So, why are we using numbers here? The reason is because each joystick button has an index that we use to refer to it. So for example, on my gamepad, button `0` is the *A* button.

We need to add two more lines of code to our `Dispose(bool)` method now. They will go in the managed resources section of the method. Here they are:

[PRE22]

This simply checks if the `Joystick` object is null. If not, then we dispose of it.

Run the program and press the buttons on your game controller. If you press the buttons `0`, `1`, `2`, or `3`, you will see some new lines of debug output appearing on Visual Studio's **Output** pane. When one of these buttons we coded for is pressed, its message appears multiple times. This is due to the speed at which the game loop is running. It is running super fast right now since it doesn't even have to render any graphics or simulate anything yet! The downloadable code for this chapter adds more `if` statements to cover more buttons than we did here. It also has some commented out lines for displaying the current positions of the left and right joysticks, and the position of the axis that is being used to represent the triggers (these are the buttons you can press in a little bit, all the way, or not at all, and they are usually found on the shoulders of a gamepad style controller).

### Note

You can detect when the user presses the thumbstick buttons the same way you do normal buttons on the gamepad, you just need to figure out which index represents each thumbstick button. This is not normally a problem since most games let the user bind game actions to whichever buttons or axes they want. In other words, you should generally never hardcode the controls in your game as they might not be correct or desirable for some players.

We've really only scratched the surface of using joysticks with DirectInput. Spend some time exploring the various properties of the `Joystick` object that we stored in our `m_Joystick1` member variable. You'll see that it has a lot of other properties we didn't use here. The `X` and `Y` properties, for example, will usually tell you what the left joystick is doing. The `RotationX` and `RotationY` properties will usually tell you the position of the right analog stick. A joystick has two axis as you can see. If the joystick is not moved at all, it is centered, so its position reading will be in the center of the ranges of both axis. If you push the joystick all the way to the right, it will be at its maximum value on its horizontal axis. If you push it up all the way, it will be at the minimum value for its vertical axis.

### Note

You might expect the joystick's position to be (0,0) if you push it all the way up and left, but it isn't. This is because most joysticks have a circular range of movement and therefore the joystick will never be at the absolute upper-left corner of the movement range defined by its pair of axes.

The `Z` property will usually give you the value for the axis that represents the trigger buttons for gamepad style devices in most cases. If neither trigger is pressed, the value is in the middle of the range. If the left trigger is completely pressed, the `Z` property will have the maximum value for the axis, and of course if the right trigger is completely pressed, then `Z` will have the minimum value for the range of this axis. The range can vary and you can also modify stuff like this by messing with the `Properties` property of the `Joystick` object (remember that you have to release a device before you can change its properties). This range can vary from one controller to the next.

What about the Directional Pad though (often called a D-Pad for short)? How you handle these depends on how the controller reports it. Some may report the D-Pad as normal buttons, in which case it would be handled in the same way as normal buttons. Other controllers report the D-Pad as a POV (point of view) controller. In this case, you can access it using the `GetPointOfViewControllers()` method of the `JoystickState` object. It returns an `int` array. The first index represents the first POV controller on your game controller. The value of the first element of the array will change depending on which direction you are pressing. For mine, the first element has the value `0` when I press up, `9,000` when I press to the right, `18,000` when I press down, and `27,000` when I press to the left on the D-Pad.

### Note

Much of this can vary depending on the type of game controller you are using and how DirectInput sees it. So, you may have to play around with different properties in the `JoystickState` object (remember we stored ours in the `m_Joy1StateCurrent` member variable) to find what you need.

Feel free to experiment with the debug code we just added into the `Update()` method. Experimentation is a great way to learn new things. Sometimes, it's quicker than reading lots of boring documentation too! We won't fully cover DirectInput here as that could fill an entire book by itself.

The downloadable code for this chapter contains a bunch of handy joystick handling methods added to our `UserInput` class. They include `DI_LeftStickPosition()`, `DI_RightStickPosition`, and `DI_TriggersAxis()`, among others. **DI** is of course short for DirectInput. The `TriggersAxis()` method gets the current value of the axis that represents the triggers (discussed earlier). The joystick methods get the current position of the joysticks. For mine, each axis has a range of `0` to `65535`, and each joystick has two axes of course (horizontal and vertical). When the joystick is not pressed at all, its position will be in the center of both its horizontal and vertical axes.

### Note

These methods may not work quite right with some devices since different game controllers are set up differently. It should work correctly for most gamepad style controllers though.

### Tip

Remember that you should really never hardcode the controls in your game. Players will be very annoyed if the controls are screwy or don't work on their particular game controller, and they find that they can't change them because you hardcoded the controls in your game.

# Using joysticks with XInput

Once again, we first need to add some member variables for our XInput device. They look a bit different this time, but here they are:

[PRE23]

In XInput, we use the `Controller` class to represent a controller. The `Gamepad` structure stores the state of the controller. As before, we have one variable to hold our device, and two more to hold its current and previous state.

Now, we will add a very short new method named `InitXInput()`. Here is its code:

[PRE24]

This code sets up one XInput controller for us to use. We pass into its constructor the value `UserIndex.One` to indicate that this controller will be used by player 1.

We need to modify the constructor of our user input class to call this new method now. We also need to add some code to initialize our XInput joystick state variables. As mentioned earlier, this is necessary to prevent the program from crashing. Here is what the constructor looks like now with the new bits of code highlighted:

[PRE25]

Now, we must add the following code to the end of the `Update()` method in our user input class:

[PRE26]

This code is very similar to our DirectInput joystick test code. It copies the state data from the previous frame into the `m_Controller1StateLast` member variable, and then gets the current controller state and stores it in the `m_Controller1StateCurrent` variable.

The `if` statements are just like the ones we used to test our DirectInput joystick code. The first one checks if the *A* button is pressed. If so, it prints a debug message saying so in Visual Studio's **Output** pane. The second `if` statement does this for the *B* button, and the last two `if` statements do the same for the *X* and *Y* buttons.

You may have noticed that we didn't have to *acquire* the XInput controller at the beginning of the `Update()` method like we do with the mouse, keyboard, and joysticks under DirectInput. Instead, we simply set up the XInput controller in our `InitXInput()` method. You may also have noticed that we didn't need to add code in our `Dispose(bool)` method to dispose of the XInput controller object either. It doesn't even have a `Dispose()` method.

We are now ready to test our new code. You will need an XInput compatible controller to test it. If you don't have one, this code will still run, but it just won't do anything since there's no XInput controller for it to get input from.

If you have a controller that supports XInput, you may see dual output from this code because both the DirectInput and the XInput test code will be outputting debug messages to Visual Studio's **Output** pane at the same time (if both are reading input from the same controller), as shown in the following screenshot:

![Using joysticks with XInput](img/7389OS_02_02.jpg)

DirectInput and XInput both reading input from the same device

We have once again only really scratched the surface here. There is more to XInput than what we've looked at. For example, you can get the state of the left and right sticks by accessing the `LeftThumbX` and `LeftThumbY` properties for the left stick, and the `RightThumbX` and `RightThumbY` properties for the right stick. Note that the range for joystick axis values in XInput is always `-32,768` to `32,767`.

You also may have noticed that we didn't add properties to the user input class to provide access to our joystick objects. They would be just as simple as the properties we added in this chapter, so they've simply been omitted from the chapter to save space. They are, however, included in the downloadable code for this chapter. Also included are a bunch of joystick handling methods for XInput devices, including `XI_LeftStickPosition()`, `XI_RightStickPosition()`, `XI_LeftTrigger()`, and `XI_RightTrigger()`, among others. **XI** is of course short for XInput. Note that for the left and right triggers, their values are in the range of `0` to `255` depending on how much you press the trigger in. Also, in XInput the D-Pad is treated as regular buttons, so you will find button flags for all of its directions in the `GamepadButtonFlags` enumeration. This is also true for the thumbstick buttons.

Explore the various properties of the XInput `Controller` object to learn more about what you can do. Remember that we stored our `Controller` object in the `m_Controller1` member variable. Experiment with this code and see what you can discover.

Note that the downloadable code for this chapter also includes some additional test code for the keyboard and mouse input inside our `Update()` method in the `UserInput` class. This code is very similar to the joystick test code that was shown in this chapter for both DirectInput and XInput.

# Summary

In this chapter, we had a crash course in responding to user input. First we looked at the differences between DirectInput and XInput. Then we looked at how to detect and respond to mouse and keyboard input. Next, we moved on to using joysticks with DirectInput, where we first looked at how to get a list of the available game controllers that are connected to the computer. Then, for simplicity, we added code to obtain the first game controller from the list and get some input from it. We wrote test code that outputs some debug text when you press the `0`, `1`, `2`, or `3` buttons. And finally, we looked at XInput controllers. The code we implemented to get input from the XInput controller was very similar to the DirectInput code, but slightly different. And lastly, we added some code to write some debug text into Visual Studio's **Output** pane whenever you press the *A*, *B*, *X*, or *Y* buttons on the XInput controller. In the next chapter, we will learn how to draw 2D graphics on the screen and create a 2D tile-based game world.