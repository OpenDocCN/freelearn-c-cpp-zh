# Chapter 3. Your First Game – Air Hockey

*We are going to build an Air Hockey game to introduce you to all the main aspects of building a project with Cocos2d-x. These include setting up the project's configuration, loading images, loading sounds, building a game for more than one screen resolution, and managing touch events.*

*Oh, and you will need to call a friend. This is a two player game. Go on, I'll wait here.*

By the end of this chapter, you will know:

*   How to build an iPad-only game
*   How to enable multitouch
*   How to support both retina and non-retina displays
*   How to load images and sounds
*   How to play sound effects
*   How to create sprites
*   How to extend the Cocos2d-x `Sprite` class
*   How to create labels and update them

Without further ado...let's begin.

# Game configurations

The game will have the following characteristics:

*   It must support multitouch since it's a two player game
*   It must be played on large screens since it's a two player game
*   It must support retina displays because we want to cash in on that
*   It must be played only in portrait mode because I built the art in portrait

So let's create our project!

# Time for action – creating your game project

I'll build the game first in Xcode and then show how to take the project to Eclipse, but the folder structure remains the same, so you can work with any IDE you wish and the instructions here will be the same:

1.  Open the terminal and create a new Cocos2d-x project called `AirHockey` that uses C++ as its main language. I saved mine on the desktop, so the command I had to enter looks like this:

    [PRE0]

2.  Once the project is created, navigate to its `proj.ios_mac` folder and double-click on the `AirHockey.xcodeproj` file. (For Eclipse, you can follow the same steps we did when we created the `HelloWorld` project to import the project.)
3.  Select the top item in **Project Navigator** and making sure the **iOS** target is selected, edit the information by navigating to **General** | **Deployment info**, setting the target device to **iPad** and **Device Orientation** to **Portrait** and **Upside Down**.![Time for action – creating your game project](img/00007.jpeg)
4.  Save your project changes.

## *What just happened?*

You created a Cocos2d-x project targeting iPads, and you are ready to set it up with the rest of the configurations I described earlier.

So let's do that now.

# Time for action – laying down the rules

We'll update the `RootViewController.mm` file.

1.  Go to `RootViewController.mm` inside the `ios` folder and look for the `shouldAutorotateToInterfaceOrientation` method. Change the line inside the method to read:

    [PRE1]

2.  And a few lines below in the `supportedInterfaceOrientations` method, change the line inside the conditional `to`:

    [PRE2]

## *What just happened?*

We just told `RootViewController` we want our application to play in any of the two supported portrait modes.

# Supporting retina displays

Now let's add the images to our project.

# Time for action – adding the image files

First, we download the resources for this project, and then we add them in Xcode.

1.  Go to this book's **Support** page ([www.packtpub.com/support](http://www.packtpub.com/support)) and download the `4198_03_RESOURCES.zip` file. Inside it, you should find three folders called `hd`, `sd`, and `fonts`.
2.  Go to your `Project` folder, the actual folder in your system. Drag the three folders to the `Resources` folder inside your project.
3.  Go back to Xcode. Select the `Resources` folder in your project navigation panel. Then go to **File** | **Add Files to AirHockey**.
4.  In the **File** window, navigate to the `Resources` folder and select the `sd`, `hd`, and `fonts` folders.
5.  This is very important: make sure **Create folder references for any added folders** is selected. Also make sure you selected **AirHockey** as the target. It wouldn't hurt to make sure **Copy items to destination...** is also selected.
6.  Click on **Add**.

## *What just happened?*

You added the necessary image files for your Air Hockey game. These come in two versions: one for retina displays (high definition) and one for non-retina displays (standard definition). It is very important that references are added to the actual folders, only this way Xcode will be able to have two files with the same name inside the project and still keep them apart; one in each folder. We also added the font we'll be using in the game.

Now let's tell Cocos2d-x where to look for the correct files.

# Time for action – adding retina support

This time we'll work with the class `AppDelegate.cpp`:

1.  Go to `AppDelegate.cpp` (you'll find it in the `Classes` folder). Inside the `applicationDidFinishLaunching` method, and below the `director->setAnimationInterval(1.0 / 60)` line, add the following lines:

    [PRE3]

2.  Save the file.

## *What just happened?*

An entire book could be written about this topic, although in this first example, we have a very simple implementation on how to support multiple screen sizes since we are only targeting iPads. Here we are saying: "Hey `AppDelegate`, I designed this game for a 768 x 1024 screen."

All the values for positioning and font size were chosen for that screen size. If the screen is larger, make sure you grab the files from the `hd` folder and change the scale by which you will multiply all my positioning and font sizes. If the screen has the same size I designed the game for, use the files in the `sd` folder and set the scale to 1\. (Android adds even more complexity to this, but we'll tackle that in later in the book.)

`FileUtils` will look for every file you load for your game first inside `Resources` | `sd` (or `hd`). If it doesn't find them there, it will try to find them in `Resources`. This is a good thing because files shared by both versions may be added only once to the project, inside `Resources`. That is what we'll do now with the sound files.

# Adding sound effects

This game has two files for sound effects. You will find them in the same `.zip` file you downloaded previously.

# Time for action – adding the sound files

Assuming you have the sound files from the downloaded resources, let's add them to the project.

1.  Drag both the `.wav` files to the `Resources` folder inside your `Project` folder.
2.  Then go to Xcode, select the `Resources` folder in the file navigation panel and select **File** | **Add Files to AirHockey**.
3.  Make sure the **AirHockey** target is selected.
4.  Go to `AppDelegate.cpp` again. At the top, add this `include` statement:

    [PRE4]

5.  Then below the `USING_NS_CC` macro (for `using namespace cocos2d`), add:

    [PRE5]

6.  Then just below the lines you added in the previous section, inside `applicationDidFinishLaunching`, add the following lines:

    [PRE6]

## *What just happened?*

With the `preloadEffect` method from `CocosDenshion`, you manage to preload the files as well as instantiate and initialize `SimpleAudioEngine`. This step will always take a toll on your application's processing power, so it's best to do it early on.

By now, the folder structure for your game should look like this:

![What just happened?](img/00008.jpeg)

# Extending Sprite

No, there is nothing wrong with `Sprite`. I just picked a game where we need a bit more information from some of its sprites. In this case, we want to store where a sprite is and where it will be once the current iteration of the game is completed. We will also need a helper method to get the sprite's radius.

So let's create our `GameSprite` class.

# Time for action – adding GameSprite.cpp

From here on, we'll create any new classes inside Xcode, but you could do it just as easily in Eclipse if you remember to update the `Make` file. I'll show you how to do that later in this chapter.

1.  In Xcode, select the `Classes` folder and then go to **File** | **New** | **File** and navigate to **iOS** | **Source** select **C++ File**.
2.  Call it `GameSprite` and make sure the **Also create a header file** option is selected.
3.  Select the new `GameSprite.h` interface file and replace the code there with this:

    [PRE7]

## *What just happened?*

In the interface, we declare the class to be a subclass of the public `Sprite` class.

Then we add three synthesized properties. In Cocos2d-x, these are macros to create getters and setters. You declare the type, the protected variable name, and the words that will be appended to the `get` and `set` methods. So in the first `CC_SYNTHESIZE` method, the `getNextPosition` and `setNextPosition` method will be created to deal with the `Point` value inside the `_nextPosition` protected variable.

We also add the constructor and destructor for our class, and the ubiquitous static method for instantiation. This receives as a parameter, the image filename used by the sprite. We finish off by overriding `setPosition` from `Sprite` and adding the declaration for our helper method radius.

The next step then is to implement our new class.

# Time for action – implementing GameSprite

With the header out of the way, all we need to do is implement our methods.

1.  Select the `GameSprite.cpp` file and let's start on the instantiation logic of the class:

    [PRE8]

2.  Next we need to override the `Node` method `setPosition`. We need to make sure that whenever we change the position of the sprite, the new value is also used by `_nextPosition`:

    [PRE9]

3.  And finally, we implement our new method to retrieve the radius of our sprite, which we determine to be half its texture's width:

    [PRE10]

## *What just happened?*

Things only begin happening in the static method. We create a new `GameSprite` class, then we call `initWithFile` on it. This is a `GameSprite` method inherited from its super class; it returns a Boolean value for whether that operation succeeded. The static method ends by returning an `autorelease` version of the `GameSprite` object.

The `setPosition` override makes sure `_nextPosition` receives the position information whenever the sprite is placed somewhere. And the helper `radius` method returns half of the sprite's texture width.

## Have a go hero

Change the radius method to an inline method in the interface and remove it from the implementation file.

# The actual game scene

Finally, we'll get to see all our work and have some fun with it. But first, let's delete the `HelloWorldScene` class (both header and implementation files). You'll get a few errors in the project so let's fix these.

References to the class must be changed at two lines in `AppDelegate.cpp`. Go ahead and change the references to a `GameLayer` class.

We'll create that class next.

# Time for action – coding the GameLayer interface

`GameLayer` is the main container in our game.

1.  Follow the steps to add a new file to your `Classes` folder. This is a C++ file called `GameLayer`.
2.  Select your `GameLayer.h`. Just below the first `define` preprocessor command, add:

    [PRE11]

3.  We define the width of the goals in pixels.
4.  Next, add the declarations for our sprites and our score text labels:

    [PRE12]

    We have the `GameSprite` objects for two players (the weird looking things called mallets), and the ball (called a puck). We'll store the two players in a Cocos2d-x `Vector`. We also have two text labels to display the score for each player.

5.  Declare a variable to store the screen size. We'll use this a lot for positioning:

    [PRE13]

6.  Add variables to store the score information and a method to update these scores on screen:

    [PRE14]

7.  Finally, let's add our methods:

    [PRE15]

There are constructor and destructor methods, then the `Layer init` methods, and finally the event handlers for the touch events and our loop method called `update`. These touch event handlers will be added to our class to handle when users' touches begin, when they move across the screen, and when they end.

## *What just happened?*

`GameLayer` is our game. It contains references to all the sprites we need to control and update, as well as all game data.

In the class implementation, all the logic starts inside the `init` method.

# Time for action – implementing init()

Inside `init()`, we'll build the game screen, bringing in all the sprites and labels we'll need for the game:

1.  So right after the `if` statement where we call the super `Layer::init` method, we add:

    [PRE16]

2.  We create the vector where we'll store both players, initialize the score values, and grab the screen size from the singleton, all-knowing `Director`. We'll use the screen size to position all sprites relatively. Next we will create our first sprite. It is created with an image filename, which `FileUtils` will take care of loading from the correct folder:

    [PRE17]

3.  Get into the habit of positioning sprites with relative values, and not absolute ones, so we can support more screen sizes. And say hello to the `Vec2` type definition used to create points; you'll be seeing it a lot in Cocos2d-x.
4.  We finish by adding the sprite as a child to our `GameLayer` (the court sprite does not need to be a `GameSprite`).
5.  Next we will use our spanking new `GameSprite` class, carefully positioning the objects on screen:

    [PRE18]

6.  We will create TTF labels with the `Label` class `createWithTTF` static method, passing as parameters the initial string value (`0`), and the path to the font file. We will then position and rotate the labels:

    [PRE19]

7.  Then we turn `GameLayer` into a multitouch event listener and tell the `Director` event dispatcher that `GameLayer` wishes to listen to those events. And we finish by scheduling the game's main loop as follows:

    [PRE20]

## *What just happened?*

You created the game screen for Air Hockey, with your own sprites and labels. The game screen, once all elements are added, should look like this:

![What just happened?](img/00009.jpeg)

And now we're ready to handle the player's screen touches.

# Time for action – handling multitouches

There are three methods we need to implement in this game to handle touches. Each method receives, as one of its parameters, a vector of `Touch` objects:

1.  So add our `onTouchesBegan` method:

    [PRE21]

    Each `GameSprite`, if you recall, has a `_touch` property.

    So we iterate through the touches, grab their location on screen, loop through the players in the vector, and determine if the touch lands on one of the players. If so, we store the touch inside the player's `_touch` property (from the `GameSprite` class).

    A similar process is repeated for `onTouchesMoved` and `onTouchesEnded`, so you can copy and paste the code and just replace what goes on inside the `_players` array for loop.

2.  In `TouchesMoved`, when we loop through the players, we do this:

    [PRE22]

    We check to see if the `_touch` property stored inside the player is the being moved now. If so, we update the player's position with the touch's current position, but we check to see if the new position is valid: a player cannot move outside the screen and cannot enter its opponent's court. We also update the player's vector of movement; we'll need this when we collide the player with the puck. The vector is based on the player's displacement.

3.  In `onTouchesEnded,` we add this:

    [PRE23]

We clear the `_touch` property stored inside the player if this touch is the one just ending. The player also stops moving, so its vector is set to `0`. Notice that we don't need the location of the touch anymore; so in `TouchesEnded` you can skip that bit of logic.

## *What just happened?*

When you implement logic for multitouch this is pretty much what you will have to do: store the individual touches inside either an array or individual sprites, so you can keep tracking these touches.

Now, for the heart and soul of the game—the main loop.

# Time for action – adding our main loop

This is the heart of our game—the `update` method:

1.  We will update the puck's velocity with a little friction applied to its vector (`0.98f`). We will store what its next position will be at the end of the iteration, if no collision occurred:

    [PRE24]

2.  Next comes the collision. We will check collisions with each player sprite and the ball:

    [PRE25]

    Collisions are checked through the distance between ball and players. Two conditions will flag a collision, as illustrated in the following diagram:

    ![Time for action – adding our main loop](img/00010.jpeg)
3.  If the distance between ball and player equals the sum of the radii of both sprites, or is less than the sum of the radii of both sprites, we have a collision:

    [PRE26]

4.  We use the squared radii values so we don't need to use costly square root calculations to get the values for distance. So all values in the previous conditional statement are squared, including the distances.
5.  These conditions are checked both with the player's current position and its next position, so there is less risk of the ball moving "through" the player sprite between iterations.
6.  If there is a collision, we grab the magnitudes of both the ball's vector and the player's vector, and make the force with which the ball will be pushed away. We update the ball's next position in that case, and play a nice sound effect through the `SimpleAudioEngine` singleton (don't forget to include the `SimpleAudioEngine.h` header file and declare we're using the `CocosDenshion` namespace):

    [PRE27]

7.  Next, we will check the collision between the ball and screen sides. If so, we will move the ball back to the court and play our sound effect here as well:

    [PRE28]

8.  At the top and bottom sides of the court, we check to see whether the ball has not moved through one of the goals through our previously defined `GOAL_WIDTH` property as follows:

    [PRE29]

9.  We finally update the ball information, and if the ball has passed through the goal posts (drum roll):

    [PRE30]

10.  We call our helper method to score a point and we finish the update with the placement of all the elements, now that we know where the `nextPosition` value is for each one of the elements in the game:

    [PRE31]

## *What just happened?*

We have just built the game's main loop. Whenever your gameplay depends on precise collision detection, you will undoubtedly apply a similar logic of position now, position next, collision checks, and adjustments to position next, if a collision has occurred. And we finish the game with our helper method.

All that's left to do now is update the scores.

# Time for action – updating scores

Time to type the last method in the game.

1.  We start by playing a nice effect for a goal and stopping our ball:

    [PRE32]

2.  Then we update the score for the scoring player, updating the score label in the process. And the ball moves to the court of the player against whom a point was just scored:

    [PRE33]

    The players are moved to their original position and their `_touch` properties are cleared:

    [PRE34]

## *What just happened?*

Well, guess what! You just finished your first game in Cocos2d-x. We charged forward at a brisk pace for our first game, but we managed to touch on almost every area of game development with Cocos2d-x in the process.

If you click **Run** now, you should be able to play the game. In the source code for this chapter, you should also find the complete version of the game if you run into any problems.

Time to take this to Android!

# Time for action – running the game in Android

Time to deploy the game to Android.

1.  Follow the instructions from the `HelloWorld` example to import the game into Eclipse.
2.  Navigate to the `proj.android` folder and open the `AndroidManifest.xml` file in a text editor. Then, go to the `jni` folder and open the `Android.mk` file in a text editor.
3.  In the `AndroidManifest.xml` file, edit the following line in the `activity` tag:

    [PRE35]

4.  And it's possible to target only tablets by adding these lines in the `supports-screens` tag:

    [PRE36]

5.  Although if you want to target only tablets, you might also wish to target the later versions of SDK, like this:

    [PRE37]

6.  Next, let's edit the make file, so open the `Android.mk` file and edit the lines in `LOCAL_SRC_FILES` to read:

    [PRE38]

7.  Save it and run your application (don't forget to connect an Android device, in this case, a tablet if you used the settings as explained here).

## *What just happened?*

And that's it! You can edit these files inside Eclipse as well.

When you build a Cocos2d-x project in the command line, you see a message saying that the `hellocpp` target is being renamed. But I think this is still a bug in the build script and usually correcting that in the make file and the folder structure creates a much bigger headache. So for now, stick to the strangely named `hellocpp` in `Android.mk`.

## Have a go hero

Make any changes to the code. For instance, add an extra label and then publish again from Eclipse. You may find that working with the project in this IDE is faster than Xcode.

Sadly, sooner or later, Eclipse will throw one of its infamous tantrums. A common problem that occurs if you have many projects open in your navigator is for one or many of the projects to report an error like **Cannot find the class file for java.lang.Object** or **The type java.lang.Object cannot be resolved**. Get into the habit of cleaning your project and building it as soon as you open Eclipse and keeping only active projects opened, but even that might fail you. The solution? Restart Eclipse, or better yet, delete the project from the navigator (but not from the disk!) and reimport it. Yeah, I know. Welcome to Eclipse!

# Summary

You now know how to add sprites and labels, and how to add support for two screen resolutions as well as support for multitouch. There are quite a few ways to create sprites other than by passing it an image filename, and I'll show examples of these in the games to come.

`LabelTTF` won't be used as much in this book. Generally, they are good for large chunks of text and text that is not updated too frequently; we'll use bitmap fonts from now on.

So, let's move on to the next game and animations. I promise I won't make you type as much. You should get your friend to do it for you!