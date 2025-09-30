# Chapter 3. Rendering 2D Graphics

One of the biggest aspects in a video game is the graphics. It's why we call them video games after all! So how do we create images on the screen? As we did with user input in the previous chapter, we have a couple of options here. They are **Direct2D** and **Direct3D** . We will focus on Direct2D in this chapter and save Direct3D for a later chapter.

In this chapter we will cover the following topics:

*   Creating a Direct2D game window class
*   Drawing a rectangle on the screen
*   Creating a 2D tile-based game world and entities

# Creating a Direct2D game window class

We are finally ready to put some graphics on the screen! The first step for us is to create a new game window class that will use Direct2D. This new game window class will derive from our original game window class, while adding the Direct2D functionality.

### Note

You'll need to download the code for this chapter as some code is omitted to save space.

Open Visual Studio and we will get started with our `Ch03` project. Add a new class to the `Ch03` project called `GameWindow2D`. We need to change its declaration to:

[PRE0]

As you can see, it inherits from the `GameWindow` class meaning that it has all of the public and protected members of the `GameWindow` class, as though we had implemented them again in this class. It also implements the `IDisposable` interface, just as the `GameWindow` class does. Also, don't forget to add a reference to SlimDX to this project if you haven't already.

We need to add some `using` statements to the top of this class file as well. They are all the same `using` statements that the `GameWindow` class has, plus one more. The new one is `SlimDX.Direct2D`. They are as follows:

[PRE1]

Next, we need to create a handful of member variables:

[PRE2]

The first variable is a `WindowRenderTarget` object. The term **render target** is used to refer to the surface we are going to draw on. In this case, it is our game window. However, this is not always the case. Games can render to other places as well. For example, rendering into a texture object is used to create various effects. One example would be a simple security camera effect. Say, we have a security camera in one room and a monitor in another room. We want the monitor to display what our security camera sees. To do this, we can render the camera's view into a texture, which can then be used to texture the screen of the monitor. Of course, this has to be re-done in every frame so that the monitor screen shows what the camera is currently seeing. This idea is useful in 2D too.

Back to our member variables, the second one is a `Factory` object that we will be using to set up our Direct2D stuff. It is used to create Direct2D resources such as `RenderTargets`. The third variable is a `PathGeometry` object that will hold the geometry for the first thing we will draw, which will be a rectangle. The last three variables are all `SolidColorBrush` objects. We use these to specify the color we want to draw something with. There is a little more to them than that, but that's all we need right now.

## The constructor

Let's turn our attention now to the constructor of our Direct2D game window class. It will do two things. Firstly, it will call the base class constructor (remember the base class is the original `GameWindow` class), and it will then get our Direct2D stuff initialized. The following is the initial code for our constructor:

[PRE3]

In the preceding code, the line starting with a colon is calling the constructor of the base class for us. This ensures that everything inherited from the base class is initialized. In the body of the constructor, the first line creates a new `Factory` object and stores it in our `m_Factory` member variable. Next, we create a `WindowRenderTargetProperties` object and store the handle of our `RenderForm` object in it. Note that `FormObject` is one of the properties defined in our `GameWindow` base class in [Chapter 1](ch01.html "Chapter 1. Getting Started"), *Getting Started*, but it is one of those properties that we haven't discussed in detail in the book. You can see it in the downloadable code for this book. Remember that the `RenderForm` object is a SlimDX object that represents a window for us to draw on. The next line saves the size of our game window in the `PixelSize` property. The `WindowRenderTargetProperties` object is basically how we specify the initial configuration for a `WindowRenderTarget` object when we create it. The last line in our constructor creates our `WindowRenderTarget` object, storing it in our `m_RenderTarget` member variable. The two parameters we pass in are our `Factory` object and the `WindowRenderTargetProperties` object we just created. A `WindowRenderTarget` object is a render target that refers to the client area of a window. We use the `WindowRenderTarget` object to draw in a window.

## Creating our rectangle

Now that our render target is set up, we are ready to draw stuff, but first we need to create something to draw! So, we will add a bit more code at the bottom of our constructor. First, we need to initialize our three `SolidColorBrush` objects. Add these three lines of code at the bottom of the constructor:

[PRE4]

This code is fairly simple. For each brush, we pass in two parameters. The first parameter is the render target we will use this brush on. The second parameter is the color of the brush, which is an **ARGB** (**Alpha Red Green Blue**) value. The first parameter we give for the color is `1.0f`. The `f` character on the end indicates that this number is of the `float` data type. We set alpha to `1.0` because we want the brush to be completely opaque. A value of `0.0` will make it completely transparent, and a value of `0.5` will be 50 percent transparent. Next, we have the red, green, and blue parameters. These are all `float` values in the range of `0.0` to `1.0` as well. As you can see for the red brush, we set the red channel to `1.0f` and the green and blue channels are both set to `0.0f`. This means we have maximum red, but no green or blue in our color.

With our `SolidColorBrush` objects set up, we now have three brushes we can draw with, but we still lack something to draw! So, let's fix that by adding some code to make our rectangle. Add this code to the end of the constructor:

[PRE5]

This code is a bit longer, but it's still fairly simple. The first line creates a new `PathGeometry` object and stores it in our `m_Geometry` member variable. The next line starts the `using` block and creates a new `GeometrySink` object that we will use to build the geometry of our rectangle. The `using` block will automatically dispose of the `GeometrySink` object for us when program execution reaches the end of the `using` block.

### Note

The `using` blocks only work with objects that implement the `IDisposable` interface.

The next four lines calculate where each edge of our rectangle will be. For example, the first line calculates the vertical position of the top edge of the rectangle. In this case, we are making the rectangle's top edge be 25 percent of the way down from the top of the screen. Then, we do the same thing for the other three sides of our rectangle. The second group of four lines of code creates four `Point` objects and initializes them using the values we just calculated. These four `Point` objects represent the corners of our rectangle. A point is also often referred to as a **vertex**. When we have more than one vertex, we call them **vertices** (pronounced as *vert-is-ces*).

The final group of code has six lines. They use the `GeometrySink` and the `Point` objects we just created to set up the geometry of our rectangle inside the `PathGeometry` object. The first line uses the `BeginFigure()` method to begin the creation of a new geometric figure. The next three lines each add one more line segment to the figure by adding another point or vertex to it. With all four vertices added, we then call the `EndFigure()` method to specify that we are done adding vertices. The last line calls the `Close()` method to specify that we are finished adding geometric figures, since we can have more than one if we want. In this case, we are only adding one geometric figure, our rectangle.

## Drawing our rectangle

Since our rectangle never changes, we don't need to add any code to our `UpdateScene()` method. We will override the base class's `UpdateScene()` method anyway, in case we need to add some code in here later, which is given as follows:

[PRE6]

As you can see, we only have one line of code in this `override` modifier of the base class's `UpdateScene()` method. It simply calls the base class version of this method. This is important because the base class's `UpdateScene()` method contains our code that gets the latest user input data each frame, as you may recall from the previous chapter.

Now, we are finally ready to write the code that will draw our rectangle on the screen! We will override the `RenderScene()` method so we can add our custom code:

[PRE7]

First, we have an `if` statement, which happens to be identical to the one we put in the base class's `RenderScene()` method. This is because we are not calling the base class's `RenderScene()` method, since the only code in it is this `if` statement. Not calling the base class version of this method will give us a slight performance boost, since we don't have the overhead of that function call. We could do the same thing with the `UpdateScene()` method as well. In this case we didn't though, because the base class version of that method has a lot more code in it. In your own projects you may want to copy and paste that code into your override of the `UpdateScene()` method.

The next line of code calls the render target's `BeginDraw()` method to tell it that we are ready to begin drawing. Then, we clear the screen on the next line by filling it with the color stored in the `ClearColor` property that is defined by our `GameWindow` base class. The last three lines draw our geometry twice. First, we draw it using the `FillGeometry()` method of our render target. This will draw our rectangle filled in with the specified brush (in this case, solid blue). Then, we draw the rectangle a second time, but this time with the `DrawGeometry()` method. This draws only the lines of our shape but doesn't fill it in, so this draws a border on our rectangle. The extra parameter on the `DrawGeometry()` method is optional and specifies the width of the lines we are drawing. We set it to `1.0f`, which means the lines will be one-pixel wide. And the last line calls the `EndDraw()` method to tell the render target that we are finished drawing.

## Cleanup

As usual, we need to clean things up after ourselves when the program closes. So, we need to add `override` of the base class's `Dispose(bool)` method, just as we did in the last chapter. We've already done this a few times, so it should be somewhat familiar and is not shown here. Check out the downloadable code for this chapter to see this code.

![Cleanup](img/7389OS_03_01.jpg)

Our blue rectangle with a red border

As you might guess, there is a lot more you can do with drawing geometry. You can draw curved line segments and draw shapes with gradient brushes too for example. You can also draw text on the screen using the render target's `DrawText()` method. But since we have limited space on these pages, we're going to look at how to draw bitmap images on the screen. These images are something that makes up the graphics of most 2D games.

# Rendering bitmaps

Instead of doing a simple demo of drawing a single bitmap on the screen, we will make a small 2D tile-based world. In 2D graphics, the term tile refers to a small bitmap image that represents one square of space in the 2D world. A **tile set** or **tile sheet** is a single bitmap file that contains numerous tiles. A single 2D graphic tile is also referred to as a **sprite** . To get started, add a new project named `TileWorld` to the `SlimFramework` solution. So far, we've directly used the game window classes we made. This time, we will see how we will do this in a real-world game project.

Add a new class file to the `TileWorld` project and name it `TileGameWindow.cs`. As you may have guessed, we will make this new class inherit from the `GameWindow` class in our `SlimFramework` project. But first, we need to add a reference to the `SlimFramework` project. We've already covered this, so go ahead and add the reference. Don't forget to add a reference to SlimDX as well. You will also need to add a reference to `System.Drawing` if there isn't one already. Also, don't forget to set `TileWorld` as the startup project.

Next, we need to add our `using` statements to the top of the `TileGameWindow.cs` file. We will need to add the following `using` statements:

[PRE8]

Next, we need to create a couple of structs and member variables. First, let's define the following **constant** at the top of this class:

[PRE9]

This constant defines the movement speed of the player. A constant is just a variable whose value can never be changed after it is initialized, so its value is always the same. Now, we need a place to store the information about our player character. We will create a structure named `Player`. Just add it right below the constant we just made with the following code:

[PRE10]

The first two member variables in this struct store the player's current location within the 2D world. The `AnimFrame` variable keeps track of the current animation frame that the player character is on, and the last variable keeps track of how long the player character has been on the current animation frame. This is used to ensure that the animation runs at about the same speed regardless of how fast your PC is.

We need to add a second struct below this one now. We will name this struct `Tile`. It stores information on a single tile. As you might guess, we will be creating a list of these structures containing one for each tile type in our game world. The following is the `Tile` struct:

[PRE11]

The first variable indicates whether this tile is solid or not. If a tile is solid, it means that the player cannot walk on it or through it. So, for example, a brick wall tile would have this set to `true`, since we don't want our players to be walking through brick walls! The last two member variables of this struct hold the coordinates of the tile's image within the tile sheet.

Next, let's turn our attention to creating the member variables for the `TileGameWindow` class. You can add these just below the structs we just created as follows:

[PRE12]

The first two member variables should be familiar from the rectangle program that we wrote at the beginning of this chapter. The `m_Player` variable holds a `Player` object. This is the first struct we created earlier. The next two variables will hold the bitmap images we will use for this program. One holds the sprites that make up the animation for our player character, and the other one will hold the tile sheet that we will use to draw the game world. The next variable is a list named `m_TileList`. We will fill this with one entry for each tile type that we have. The `m_Map` variable, as you might guess, will contain a map of our game world. And lastly, we have a `SolidColorBrush` member variable named `m_DebugBrush`.

## Initialization

Now, it's time to create the constructor and start initializing everything. First, we need to set up the render target. This is very similar to how we did it in the program for creating a rectangle, but slightly different. The following is the code:

[PRE13]

As we did in the program for creating a rectangle, we first create the factory object. After that, things differ slightly. This time we need to create two properties objects instead of one. The new one is a `RenderTargetProperties` object. We use it to set the pixel format for our render target. As you can see, we are using a 32-bit format with 8 bits for each of the four channels (blue, green, red, and alpha). Yes, this is backwards from the ARGB format we've already discussed earlier. That's OK though because our `LoadBitmap()` method will flip the ARGB format to BGRA for us. The next line of code creates a `WindowRenderTargetProperties` object, just as we did in the *Rectangle* program earlier in this chapter. We use this to specify the handle of the window we want to draw on as well as the size of the window. And lastly, we create the render target object and initialize our debug brush to be an opaque yellow brush.

So, we're done initializing stuff now, right? Well, no; not yet. We still have a few things that need to be initialized. But first, we need to create our `LoadBitmap()` method so that we can load in our graphics! The following is the code:

[PRE14]

This method is a little confusing, so I've kept the comments present in this code listing. You may have noticed that in the line with the call to the `LockBits()` method, there is a pixel format parameter, but it is different from what we saw a bit earlier in the chapter; it is `System.Drawing.Imaging.PixelFormat.Format32bppPArgb`. This is the same format we are using, but what is that `P` in there for? The `P` is short for **precalculated alpha** . This basically means that the red, green, and blue channels are automatically adjusted based on the alpha value before rendering. So, if you have the red channel at maximum and the alpha channel at 50 percent, the intensity of the red channel will be reduced by half.

There is also **straight alpha** which is less efficient than precalculated alpha. The values of the red, green, and blue channels are left alone. Their intensity is adjusted based on the value of the alpha channel during rendering. Precalculated alpha is a bit faster since it adjusts the color channels once before any rendering happens, whereas straight alpha has to adjust the color channels each time we render a new frame. And lastly, there is also an **ignore alpha** mode. In this mode, the alpha channel is completely ignored, and thus you cannot use transparent bitmaps.

We are using the precalculated alpha mode in this case and this is important. If you don't do this, the player character will have white in all of the transparent areas of the robot image, which looks rather silly. We used the `LockBits()` method to lock the memory holding the bitmap because if any other code on another thread accesses that memory while we are messing with it, this can cause crashes and other odd behavior.

Now, let's return to the constructor and initialize the player character, which will be a rather silly robot. Add the following code at the bottom of the constructor:

[PRE15]

The first line of code uses our `LoadBitmap()` method to load the robot sprite sheet and store it in the `m_PlayerSprites` member variable. The second line creates the player object to hold information about the player character. Finally, the last two lines set the starting position for the player. Note that the coordinates (0, 0) represent the upper-left corner of the screen. The robot sprite sheet is just a series of animation frames for our robot that we will display one after another in quick succession to animate the robot.

Now that the player object is initialized, we need to initialize the game world! The following is the first part of the code:

[PRE16]

The first line calls our `LoadBitmap()` method again to load in the tile sheet and store it in the `m_TileSheet` member variable. The second line creates our tile list object. This will store information for each tile type. The eight lines of code at the bottom create entries in the tile list for all of the tiles in the first row of the tile sheet. Of course, the tile sheet has more than one row of tiles in it, but I will not show the code for the other rows here, since it is very similar and would take up several pages.

We have one more thing to do to finish initializing the game world. It consists of initializing the map. The map is simply a two-dimensional array. Each element in the array represents a tile position in the game world. As such, the array is of type `int`; it is of type `int` because each element stores a numeric index in the tile list. So basically, each element in the array holds a number that tells us which type of tile is at this position in the game world. As the code that fills in this array is much too wide to fit on the page, I will show a brief example of how it is initialized here:

[PRE17]

As you can see, we are creating a new two-dimensional `int` array. In this sample code, we have a 3 x 3 world. We are using tile type `14` (a brick wall tile) to make a wall around the outer border of this small world. In the center, we have tile type `0`, which in our game demo is a grass tile. Each row of values gets its own pair of enclosing brackets (`{}`), followed by a comma. This is basically how you set up a 2D tile map. Of course you can get a lot fancier with this. For example, you can implement animated tile types in your game. These would be animated very similarly to how we will animate our robot character. Check out the downloadable code for this chapter to see the complete array initialization code, which is much larger than the earlier example.

## Rendering the game world

For clarity, we will create a couple of different render methods that will each be called from our `RenderScene()` method. Since the first thing we need to draw is the game world itself, let's create that method first. We will name this method `RenderWorld`:

[PRE18]

This code is fairly straightforward. The first line creates a `Tile` object variable. Next, we have two nested `for` loops that loop through every tile position in the game world. Inside the inner `for` loop, we get the tile type for this position on the map and look it up in the tile list. We store the result in the variable `s` so that we can use it easily afterwards. The last line renders the tile. The first parameter here is the bitmap containing the tiles. The second parameter is a rectangle specifying where we want to draw the tile on the screen. The third parameter is the opacity. We have it set to `1.0f` so that the tile is completely opaque. The third parameter is the interpolation mode. And the last parameter is another rectangle, which specifies what portion of the tile sheet we want to draw on the screen. For this, we specify the part of the tile sheet containing the tile we want to draw. For the x and y coordinates of both rectangle parameters, you may have noticed that we are multiplying by 32\. This is because each tile is 32 x 32 pixels in size. So, we have to multiply by 32 to get the position of the tile within the tile sheet correctly. The fact that our tiles are 32 x 32 pixels in size is also why both rectangles we created here specify the value `32` for their `width` and `height` parameters.

## Rendering the player character

Now that we have code to draw the world, we need to draw the player character! For this, we will create a method called `RenderPlayer()`. It's pretty short compared to the `RenderWorld()` method. The following is the code:

[PRE19]

This method contains only one line. It is very similar to the code we used to draw each tile in the `RenderWorld()` method. But this time we are using the player sprites sheet rather than the tile sheet. You may also notice that we determine which sprite to draw based on the player object's `AnimFrame` variable, which we use to keep track of which animation frame the robot is currently on.

## Rendering debug information

This is not strictly necessary, but it's a good thing to know how to do. We will create a new method called `RenderDebug()`. It will draw a yellow border on every solid tile in the game world. The following is the code:

[PRE20]

As you can see, this method looks very similar to the `RenderWorld()` method; it loops through every position in the game world just as that method does. The one major difference is that we use the `DrawRectangle()` method here rather than the `DrawBitmap()` method. Using our yellow debug brush, it draws a yellow border on any tile in the game world that is solid.

## Finishing the rendering code

Now we need to add code into the `RenderScene()` method to call these methods we just made. The following is the `RenderScene()` code:

[PRE21]

With that, our rendering code is now complete. The `if` statement at the top prevents the program from crashing when it is first starting up or shutting down. The next two lines tell the render target we are ready to begin drawing by calling the `BeginDraw()` method, and then clear the screen by calling the `Clear()` method. The next line calls our `RenderWorld()` method to draw the game world. But then, the call to the `RenderDebug()` method is preceded by `#if DEBUG` and followed by `#endif`. These are known as **preprocessor directives**. This one checks if a symbol named `DEBUG` is defined, and if so, the code inside this `if` directive will be compiled into the program. Preprocessor directives are processed by the **preprocessor,** which runs before the compiler when you compile your code. After the preprocessor has finished its job, the compiler will run. There are a bunch of other preprocessor directives besides `#if`, but they are beyond the scope of this text. When you compile your code under the `Debug` configuration, the `DEBUG` symbol is automatically defined for us, meaning our call to `RenderDebug()` will be compiled into the game. In Visual Studio, you can change the compile configuration using the drop-down list box that is just to the right of the **Start** button, which you click on to compile and run your program. Visual Studio provides `Debug` and `Release` configurations. You can also run a program by pressing the *F5* key.

The next line calls our `RenderPlayer()` method to draw the player character using the appropriate animation frame from the robot's sprite sheet. And lastly, we call the `EndDraw()` method to tell the render target that we are done rendering this frame.

## Handling user input

Now, we need to add some code into our `UpdateScene()` method to handle player input:

[PRE22]

The first line calls the base class's `UpdateScene()` method, so it can perform its stuff. The next four lines may look a bit odd though. Why do we need to find out which grid square each corner of the player sprite is in? It has to do with how our player's movement will work. Specifically, this is used by our collision detection code.

You may also notice that the first four lines of code are skewing all four corners inward by 25 percent. You can think of these four corners as our bounding box for collision detection. Shrinking the bounding box like this makes it easier for the player to enter narrow spaces that are only one block wide. Note that `TL` is short for top-left, `TR` is top-right, `BL` is bottom-left, and `BR` is bottom-right. The following is the first part of our collision detection code:

[PRE23]

This code starts with a compound `if` statement, checking whether the user is pressing the *A* key or the left arrow key. Yes, you can control our game character using either of the *W*, *A*, *S*, or *D* keys or the arrow keys if you wish to move the character using the keyboard. Next, we have another `if` statement. This `if` statement checks to see if moving the player to the left will cause a collision. If not, we move the player to the left. As you can see, we use the `PLAYER_MOVE_SPEED` constant that we created earlier in this chapter to control how much the robot moves. Obviously, we need three more of these `if` statements to handle the right, up, and down directions. As the code is very similar, I will not describe it here.

### Note

The downloadable code for this chapter also supports controlling the robot using joysticks/gamepads. It adds a member variable named `m_UseDirectInput` to the `TileGameWindow` class. Set this variable to `true` to use DirectInput for joystick/gamepad controls, or set this variable to `false` to have the program use XInput for joystick/gamepad controls. We need the `m_UseDirectInput` member variable because if we used both DirectInput and XInput at the same time for the same game controller device, this will cause the player to get moved twice per frame.

## Animating the player character

With the user input and collision detection code done, there is now only one thing left to do in `UpdateScene()`. We need to add a bit of code to animate the player character:

[PRE24]

This code is fairly simple. The first line adds `frameTime` to the player object's `LastFrameChange` variable. Remember that `frameTime` is the parameter of the `UpdateScene()` method, and it contains the amount of time that has elapsed since the previous frame. Next, we have an `if` statement that checks if the player object's `LastFrameChange` variable has a value greater than `0.1`. If this is the case, it means that it has been 1/10th of a second or more since the last time we changed the animation frame, so we will change it again. Inside the `if` statement, we reset the `LastFrameChange` variable to `0`, so we will know when to change the animation frame again. The next line increments the value of the player object's `AnimFrame` variable. And lastly, we have another `if` statement that checks if the new value of the `AnimFrame` variable is too large. If it is, we reset it to a value of `0` and the animation starts all over again.

## Running the game

We are almost ready to run the game, but don't forget that you need to add the `Dispose(bool)` method. In this program, there are only four objects it needs to dispose off. They are `m_RenderTarget`, `m_Factory`, `m_TileSheet`, and `m_DebugBrush`. They should be disposed of in the managed section of the `Dispose(bool)` method. You can see this in the downloadable code for this chapter.

With the cleanup code in place, we are ready to run the game. As you can see, you control a rather goofy robot. Note that the player sprites are in the `Robot.png` file and the tile sheet is saved in the `TileSheet.png` file. Both of these files are, of course, included with the downloadable code for this chapter. The screenshot following the explanation shows what the game window looks like with the debug overlay off.

You may have noticed that we didn't implement the fullscreen mode. This is because Direct2D unfortunately does not natively support the fullscreen mode. It is, however, possible to have the fullscreen mode in a Direct2D application. To do this, you will create a Direct3D render target and share it with Direct2D. This would then allow you to draw on it with Direct2D and also be able to use the fullscreen mode.

![Running the game](img/7389OS_03_02.jpg)

Our 2D game in action

The following screenshot shows our game with the debug overlay turned on.

![Running the game](img/7389OS_03_03.jpg)

Our 2D game in action with the debug overlay turned on

# Entities

This 2D game demo we created only has one **entity** in it (the player character). In game development, the term entity refers to an object that can interact with the game world or other objects in the game world. An entity would typically be implemented as a class. So, we would create a class to represent our player object. In this demo, our player object was very simple and had no methods in it, so we just made it a struct instead. In a real-game engine, you might have an `Entity` base class that all other entity classes would inherit from. This base class would define methods such as `Update()` and `Draw()` so that every entity has them. Each entity class would then override them to provide its own custom update and draw code.

A single level or game world can have hundreds of entities in it, so how do we manage them? One way is to create an `EntityManager` class that simply holds the collection of entities that are in the currently loaded level or world. The `EntityManager` class will have an `Update()` method and a `Draw()` method. The `Update()` method would, of course, get called once per frame by the `UpdateScene()` method of our game window class. Likewise, the `Draw()` method would be called once per frame by the `RenderScene()` method. The entity manager's `Update()` method would iterate through all of the entities and call each one's `Update()` method so that the entity can update itself. And of course, the entity manager's `Draw()` method would do the same thing, but instead it would call each entity's `Draw()` method so that the entity can draw itself.

In some games, entities are able to communicate with each other via a messaging system of sorts. A good example of this is the inputs and outputs system used in `Half-Life 2`. For example, there is a button on a wall next to a door. We will set up an output on the button that fires when the button is pressed. We will connect it to the input on the door that makes the door open. So, basically, when the output of the button fires, it activates the specified input on the door. In short, the button sends a message to the door telling it to open. The output of one object can potentially send parameters to its target input as well. The big benefit here is that many interactions between objects can be handled like this and don't need to be specifically coded as a result, but instead can simply be set up in the game's level editor.

## Component-based entities

There is another way to implement our entities as well. It implements an `Entity` class that is used to represent any possible entity. The difference is that this `Entity` class contains a collection of `Components`. A **component** is a class that represents a certain action or feature that an object in the game world can have. So, for example, you might have an **Armor** component that allows an entity to have an armor value, or a **Health** component that allows the entity to have health and the ability to take damage. This Health component would probably have a property to set the maximum health for the entity and another one that is used to get the current health value for the entity.

This is a very powerful approach because you can give any entity health (and the ability to take damage) just by adding the Health component into that entity. So, as you can see, each entity is represented by the basic `Entity` class and gets all of its features and properties from the components that are added into it. This is what makes this approach so powerful. You write the Health code once and then you can re-use it on any number of entities without having to rewrite it for each one. The component-based entities are a bit trickier to program than regular entities though. For example, we would need to add a method on the `Entity` class that lets you pass in a component type to specify which component you would like to access. It would then find the component of the specified type and return it for you to use. You would usually make your entity system such that it will not allow an entity to have more than one component of any given type as this generally wouldn't make much sense anyway. For example, giving one entity two Health components doesn't make much sense.

# Summary

In this chapter we first made a simple demo application that drew a rectangle on the screen. Then we got a bit more ambitious and built a 2D tile-based game world. In the process, we covered how to render bitmaps on the screen, basic collision detection, and reviewed some basic user input handling. We also looked at how to create a handy debug overlay. Of course, this debug overlay is pretty simple, but they can show all sorts of useful information. They are a very powerful tool when it comes to solving bugs. In the next chapter, we will look at playing music and sound effects to add more life to our 2D game world that we built in this chapter!