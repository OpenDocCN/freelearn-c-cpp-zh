# Chapter 10. Introducing Lua!

*In our last game, we'll move to the new Cocos IDE and develop an entire game using the Lua scripting language. You'll get to know and use the Lua bindings for the Cocos2d-x API, which is not much different from what we've been using in C++; if anything, it's just a lot easier!*

This time, you'll learn how to:

*   Create and publish a project in Cocos IDE
*   Code an entire game in Lua
*   Use sprites, particles, labels, menus, and actions, but this time with the Lua bindings
*   Build a match-three game

# So what is Lua like?

At the heart of Lua (which means moon in Portuguese), you have the table. You may think of it as being similar to a JavaScript object, only it's much more than that. It plays the part of arrays, dictionaries, enumerations, structures, and classes, among other things. It makes Lua the perfect language to manage large sets of data. You write a script that handles the data, and then keep feeding it different "stuff." An inventory or shop system, an interactive children's book—these types of projects could all benefit from Lua's table-centric power, as they can be built around a fixed template with a data table at its core.

Its syntax, for those not used to a scripting language, can be a little odd, with its dos, thens, and ends. But once you get past this initial hurdle, you'll find Lua quite user-friendly. Here are some of the "oddities" in its syntax:

[PRE0]

### Note

Semicolons are optional.

A table can be turned into a template to generate instances of it, in other words, a class. Methods of the instance of the table must be accessed with a `:` notation:

[PRE1]

From inside the method, you refer to the instance of the class as `self`:

[PRE2]

Alternatively, you can call the template's method with a dot notation, passing the instance of that template to it as the first parameter:

[PRE3]

I admit, it's weird, but it can be useful at times as pretty much every method you write in Lua can be made available for other parts of your code—sort of the way static methods are used in traditional OOP languages.

## Debugging in Lua – the knights who say nil

Debugging your Lua code can be frustrating at times. But you soon learn to distinguish between the minute subtleties in Lua's runtime errors. The compiler will say something is `nil` (Lua's `null`) in about 99.9 percent of cases. It's up to you to figure out why. Here are the main culprits:

*   You are referencing an object's own property without prepending `self.` or `self:`.
*   You are calling an instance method with a dot notation, and not passing the instance as the first parameter; something like `myObject.myMethod()` instead of `myObject.myMethod(myObject)`. Use `myObject:myMethod()` instead.
*   You are referencing a variable from a place outside its scope. For example, a local variable declared inside an `if` statement is being referenced outside the conditional.
*   You forgot to return the class object at the end of your class or module/table declaration.
*   You tried to access the zero index of an array.
*   You forgot to add a few dos and thens or ends.
*   And finally, maybe you're just having one of those days. A `nil` sort of day.

The Cocos IDE will show errors in bold; the same bold it uses for global variables, which is confusing at times. But it helps nonetheless. Just make a habit of scanning your code for bold text!

### Tip

You might need to increase the heap memory inside the IDE. The quickest way to accomplish this is to find the file called `eclipse.ini` inside the Cocos IDE application folder. On a Mac, this means inside the Cocos IDE app package: right-click on the app icon, select **Show Package Contents**, and then navigate to **Contents/MacOS/eclipse.ini**.

Then find the line where you read `-Xmx256m` or `-Xmx512m` and change it to `-Xmx1024m`.

This might help in slower computers. My laptop crashed a lot while running the IDE.

# The game – Stone Age

This is a match-three game. You know, the kind of game that is making a couple of companies a gazillion dollars and making a gazillion other companies clone those games in order to earn a couple of dollars. Yes, that game!

You must match three or more gems. If you match more than three, a random gem bursts and turns into a diamond, which you collect for more points.

The game has a timer, and when time runs out, it's game over.

I used pretty much the same structure as in the previous games in this book. But I broke it into separate modules so it's easier for you to use the code as a reference.

We have a `MenuScene` and a `GameScene` item. I have pretty much all Cocos2d-x actions in one module called `GridAnimations` and most of the interactivity inside another module called `GridController`. And all object pools are kept inside a class called `ObjectPools`.

This is a grid game, so it's perfect to illustrate working with table arrays in Lua, and its main advantages over C++: it's much easier to create and memory manage dynamic lists (arrays) in Lua. This flexibility, aligned with Cocos2d-x's awesomeness, make for very rapid prototyping and development. The actual game will look like this:

![The game – Stone Age](img/00039.jpeg)

But before you import the starter project, let me show you how to create a new project inside the Cocos IDE.

# Time for action – creating or importing a project

Nothing could be simpler; since the IDE is based on Eclipse, you know most of its main functionalities already:

1.  First let's set up the IDE to use the Lua bindings. Go to **Preferences** | **Cocos** | **Lua**, and in the drop-down menu for **Lua Frameworks**, find the Cocos2d-x framework folder you downloaded:![Time for action – creating or importing a project](img/00040.jpeg)
2.  Select **File** | **New** | **Cocos Lua Project**, if that option is already available, or select **File** | **New** | **Other** | **Cocos Lua** | **Cocos Lua Project**.
3.  In the **New Cocos Project** wizard, give your project a name and click **Next**.
4.  In the next dialogue, you can choose your project's orientation and design size. And that's it. Click **Finish**.
5.  In order to import a project, click **File** | **Import** then **Cocos** | **Import Cocos Project**, and navigate to the project start folder for this chapter. The game is called `StoneAge`. (Download this chapter's source files from this book's website if you haven't done so already. There is a starter project and a final project that you can run and test.)

## *What just happened?*

You learned to create and import a project into the Cocos IDE. Since the IDE is an Eclipse-based program, the steps should be familiar to you by now.

You may also wish to change the settings for the simulator. For that, all you need to do is right-click on your project and select either **Run As...** or **Debug As...**, and then **Run** or **Debug Configurations**.

It's best to leave the default for the **Mac OSX** runtime (if you're on a Mac of course), as this is the fastest option. But if you wish to change the simulator, here is where you do it:

![What just happened?](img/00041.jpeg)

### Note

On my machine, version 3.4 of the framework threw compile errors. I had to add two fixes in order to run Stone Age. In `cocos-cocos2d-Cocos2dConstants.lua`, just before the last table is declared, I added this line:

[PRE4]

Similarly, in `cocos-ui-GuiConstants.lua`, I added `ccui.LayoutComponent = {}` before new tables are added to `LayoutComponent`, also near the end of the file.

If you run into problems, switch to version 3.3, which was much more stable for Lua development.

# Time for action – setting up our screen resolution

The old `AppDelegate` class logic now exists inside a file called `main.lua`:

1.  In the IDE, open the `main.lua` file inside the `src` folder.
2.  After the line where we set the animation interval, type the following:

    [PRE5]

3.  I designed the game for iPhone retina, and here we set the appropriate scale and asset folder for both retina and non-retina phones. Next, let's preload the sound files:

    [PRE6]

4.  And finally, let's set the ball rolling by creating and running our first scene:

    [PRE7]

## *What just happened?*

Like we've done in pretty much every game so far, we set the resolution policy and scale factor for our application and preloaded the sounds we'll be using.

The game was designed only for phones this time, and it was designed with the iPhone 4 screen in mind, and it resizes to older phones.

But don't run the game just yet. Let's create our menu scene. It has a little of everything in it and it will be a perfect introduction to the Cocos2d-x API in Lua.

# Time for action – creating a menu scene

Let's create a new file and add a menu scene to our game:

1.  Right-click on the `src` folder and select **New** | **Lua File**; call the new file `MenuScene.lua`.
2.  Let's create a class that extends a scene. We first load our own module of all the game's constants (this file already exists in the starter project):

    [PRE8]

3.  Then we build our class:

    [PRE9]

    We'll add the methods next, including the `init` method we called in the class constructor (always called `ctor`), but I wanted to stress the importance of returning the class at the end of its declaration.

4.  So moving just below the constructor, let's continue building up our scene:

    [PRE10]

    With this, we added a background and two other sprites, plus an animation of a pterodactyl flying in the background. Once again, the calls are remarkably similar to the ones in C++.

5.  Now let's add a menu with a play button (all this still inside the `init` method):

    [PRE11]

Typing the button's callback inside the same method where the callback is referenced is similar to writing a block or even a lambda function in C++.

## *What just happened?*

You created a scene in Lua with Cocos2d-x using a menu, a few sprites, and an animation. It's easy to see how similar the calls are in the Lua bindings to the original C++ ones. And with code completion inside the IDE, finding the correct methods is a breeze.

Now let's tackle the `GameScene` class.

### Note

One of the nicest features of Lua is something called **live coding**, and it's switched on by default in the Cocos IDE. To see what I mean by live coding, do this: while the game is running in the simulator, change the position of the character sprite in your code and save it. You should see the change taking effect in the simulator. This is a great way to build UI and game scenes.

# Time for action – creating our game scene

The `GameScene` class is already added to the start project and some of the code is already in place. We'll focus first on building the game's interface and listening to touches:

1.  Let's work on the `addTouchEvents` method:

    [PRE12]

2.  Once again, we register the events with the node's instance of the event dispatcher. The actual touches are handled by our `GridController` object. We'll go over those later; first, let's build the UI. Time to work on the `init` method:

    [PRE13]

    Create our special objects, one to handle user interactivity, another for animations, and our good old object pool.

3.  Next, we add a couple of nodes and our score labels:

    [PRE14]

The main difference when compared to the C++ implementation of `Label:createWithTTF` is that, in Lua, we have a configuration table for the font.

## *What just happened?*

This time, we saw how to register for touch events and how to create true type font labels. Next, we'll go over creating a typical grid for a match-three game.

# Time for action – building the gems

There are basically two types of match-three games, those in which the selection of matches takes place automatically and those in which the matches are selected by the player. *Candy Crush* is a good example of the former, and *Diamond Dash* of the latter. When building the first type of game, you must add extra logic to ensure you start the game with a grid that contains no matches. This is what we'll do now:

1.  We start with the `buildGrid` method:

    [PRE15]

    Ensure that we generate a different random series of gems each time we run the game by changing the `randomseed` value.

    The `enabled` property will stop user interactions while the grid is being altered or animated.

    The grid is a two-dimensional array of columns of gems. The magic happens in the `getVerticalUnique` and `getVerticalHorizontalUnique` methods.

2.  To ensure that none of the gems will form a three-gem-match on the first two columns, we check them vertically:

    [PRE16]

    All this code is doing is checking a column to see if any gem is forming a string of three connected gems of the same type.

3.  Then, we check both vertically and horizontally, starting with column 3:

    [PRE17]

This algorithm is doing the same thing we did previously with the columns, but it's also checking on individual rows.

## *What just happened?*

We created a grid of gems, free of any three-gem matches. Again, if we had built the sort of match-three game where the user must select clusters of matched gems to have these removed from the grid (like *Diamond Dash*), we would not have to bother with this logic at all.

Next, let's manipulate the grid with gem swaps, identification of matches, and grid collapse.

# Time for action – changing the grid with GridController

The `GridController` object initiates all grid changes since it's where we handle touches. In the game, the user can drag a gem to swap places with another, or first select the gem they want to move and then select the gem they want to swap places with in a two-touch process. Let's add the touch handling for that:

1.  In `GridController`, let's add the logic to `onTouchDown`:

    [PRE18]

    If we are displaying the game over screen, restart the scene.

2.  Next, we find the gem the user is trying to select:

    [PRE19]

    We find the gem closest to the touch position. If the user has not selected a gem yet (`selectedGem = nil`), we set the one just touched as the first gem selected. Otherwise, we determine whether the second gem selected can be used for a swap. Only gems above and below the first selected gem, or the ones to the left and right of it, can be swapped with. If that is valid, we use the second gem as the target gem.

3.  Before moving on to `onTouchMove` and `onTouchUp`, let's see how we determine which gem is being selected and which gem is a valid target gem. So let's work on the `findGemAtPosition` value. Begin by determining where in the grid container the touch landed:

    [PRE20]

4.  Here is where the magic happens. We use the `x` and `y` position of the touch inside the grid to determine the index of the gem inside the array:

    [PRE21]

    We finish by checking whether the touch is out of array bounds.

5.  And now let's see the logic to determine whether the target gem is a valid target:

    [PRE22]

    We first check to see whether the target gem is at the top, bottom, left, or right of the selected gem:

    [PRE23]

    We next use a bit of trig magic to determine whether the selected target gem is diagonal to the selected gem:

    [PRE24]

    We finish by checking whether the target gem is not the same as the previously selected gem.

6.  Now, let's move on to the `onTouchUp` event handling:

    [PRE25]

    Pretty simple! We just change the `z` layering of the selected gem, as we want to make sure that the gem is shown above the others when the swap takes place. So when we release the gem, we push it back to its original `z` level (which is what the `dropSelectedGem` method does, and we'll see how it does this soon).

7.  The `onTouchMove` event handles the option of dragging the selected gem until it swaps places with another gem:

    [PRE26]

    We run most of the same logic as we did with `onTouchDown`. We move the `selectedGem` object until a suitable target gem is identified, and then we pick the second one as the target. This is when the swap happens. Let's do that now.

8.  First, the logic that sets our selected gem:

    [PRE27]

    We start the swapping process; we have a selected gem but no target gem. We change the layering of the selected gem through `setLocalZOrder`. We also make the selected gem rotate 360 degrees.

9.  Then, we're ready to select the target gem:

    [PRE28]

It is now that we finally call our `GameScene` class and ask it to swap the gems.

## *What just happened?*

We just added the logic to handle all the user interactivity. Now, all that's left for us to do is handle the swaps, checking for matches and collapsing the grid. Let's do it!

# Time for action – swapping the gems and looking for matches

The swapping logic is found in `GameScene` in the `swapGemsToNewPosition` method:

1.  The `swapGemsToNewPosition` method makes one call to `GridAnimations` to animate the swap between the selected and target gem. Once this animation is complete, we fire a `onNewSwapComplete` method. The majority of the logic takes place in there:

    [PRE29]

2.  If we have a match, we run animations on the matched gems, otherwise we play a swap back animation and play a sound effect to represent a wrong move by the player:

    [PRE30]

    At the end of each new animation, be it the match one or the swap back one, we once again run callbacks listed at the top of the method. The most important thing these do is the call to `collapseGrid` done when the matched gems finish animating inside the `onMatchedAnimatedOut` callback:

    [PRE31]

    We end the callback by clearing the selected gems and start with a clean slate.

3.  And here, at the end of the function, we call the swap gems animation with `onNewSwapComplete` as its callback:

    [PRE32]

4.  Let's move back to `GridController` and add the `checkGridMatches` method. This is broken into three parts:

    [PRE33]

    This method starts the check by running `checkTypeMatch` on each cell.

5.  The `checkTypeMatch` method searches around the current index and looks for matches at the top, bottom, left, and right of the index:

    [PRE34]

    If any matches are found, they are added to the `matches` array.

6.  But first we need to make sure there are no duplicates listed there, so when we add a gem to the `matches` array, we check whether it has not been added already:

    [PRE35]

7.  And the simple method to look for duplicates:

    [PRE36]

## *What just happened?*

Finding matches is more than half the necessary logic for any match-three game. All you need to do is traverse the grid as effectively as you can and look for repeated patterns.

The rest of the logic concerns the grid collapse. We'll do that next and then we're ready to publish the game.

# Time for action – collapsing the grid and repeating

So the flow of the game is move pieces around, look for matches, remove those, collapse the grid and add new gems, look for matches again, and if necessary, do the whole process in a loop:

1.  This is the longest method in the game, and again, most of the logic happens inside callbacks. First we tag the gems being removed by setting their type data to `-1`. All the gems inside `matchArray` will be removed:

    [PRE37]

2.  Next, we traverse the grid's columns and rearrange the gems whose type is not equal to `-1` inside the column arrays. Essentially, we update the data here so that gems above the ones removed "fall down". The actual change will take place in the `animateCollapse` method:

    [PRE38]

3.  But now, let's code the callback of that animation called `onGridCollapseComplete`. So above the code we've entered already inside `collapseGrid`, we add the `local` function:

    [PRE39]

    First, we update the array of sprites, sorting them by the new `x` and `y` indexes of the grid.

4.  Then, we check for matches again. Remember that this callback runs after the grid collapse animation has finished, which means new gems have been added already and these may create new matches (we'll look at the logic soon):

    [PRE40]

5.  Then, if we find no more matches, we replace some random gems with diamonds if the value for combos is above 0 (meaning we had more than a 3 gem match in the last player's move):

    [PRE41]

6.  And we pick random gems for the diamonds:

    [PRE42]

    Animate the diamonds being collected, and at the end of that animation, call back `onMatchedAnimatedOut`, which will collapse the grid once more now that we had gems "burst" into diamonds:

    [PRE43]

7.  Here's the whole `collapseGrid` method:

    [PRE44]

## *What just happened?*

The `collapseGrid` method collects all the gems affected by matches or gems which exploded into diamonds. The resulting array is sent to `GridAnimations` for the proper animations to be performed.

We'll work on those next and finish our game.

# Time for action – animating matches and collapses

Now for the last bit of logic: the final animations:

1.  We'll start with the easy ones:

    [PRE45]

    This rotates a gem; we use this animation when a gem is first selected.

2.  Next is the swap animation:

    [PRE46]

    All this does is swap the places of the first selected gem and the target gem.

3.  Then, we add the animations we run for matched gems:

    [PRE47]

    This will scale down a gem to nothing, and only fire the final callback when all gems have finish scaling.

4.  Next is the collect diamonds animation:

    [PRE48]

    This moves the diamonds to where the diamond score label is.

5.  And now, finally, add the grid collapse:

    [PRE49]

    We loop through all the gems and identify the ones that have been scaled down, meaning the ones which were *removed*. We move these above the column, so they will fall down as new gems, and we pick a new type for them:

    [PRE50]

    The ones which were not removed will drop to their new positions. The way we do this is simple. We count how many gems were removed until we reached a gem which has not been removed. That count is stored in the local variable drop, which is reset to `0` with every column.

    That way, we know how many gems were removed below other gems. We use that to find the new `y` position.

6.  The `dropGemTo` new position looks like this:

    [PRE51]

Again, we only fire the final callback once all gems have collapsed. This final callback will run another check for matches, as we've seen earlier, starting the whole process again.

## *What just happened?*

That's it; we have the three main parts of a match-three game: the swap, the matches, and the collapse.

There is only one animation we haven't covered, which is already included in the code for this chapter, and that is the column drop for the intro animation when the grid is first created. But there's nothing new with that one. Feel free to review it, though.

Now, it's time to publish the game.

# Time for action – publishing the game with the Cocos IDE

In order to build and publish the game, we'll need to tell the IDE a few things. I'll show you how to publish the game for Android, but the steps are very similar for any of the other targets:

1.  First, let's tell the IDE where to find the Android SDK, NDK, and ANT, just as we did when we installed the Cocos2d-x console. In the IDE, open the **Preferences** panel. Then, under **Cocos**, enter the three paths just like we did before (remember that for ANT, you need to navigate to its `bin` folder).![Time for action – publishing the game with the Cocos IDE](img/00042.jpeg)
2.  Now, in order to build the project, you need to select the fourth button at the top of the IDE (from the left-hand side), or right-click on your project and select **Cocos Tools**. You'll have different options available depending on which stage you are at in the deployment process.![Time for action – publishing the game with the Cocos IDE](img/00043.jpeg)

    First, the IDE needs to add the native code support, and then it builds the project inside a folder called frameworks (it will contain an iOS, Mac OS, Windows, Android, and Linux version of your project just as if you had created it through the Cocos console).

3.  You can then choose to package the application into an APK or IPA, which you can transfer to your phone. Or, you can use the generated project inside Eclipse or Xcode.

## *What just happened?*

You just built your Lua game to Android, or iOS, or Windows, or Linux, or Mac OS, or all of them! Well done.

# Summary

That's it. You can now choose between C++ or Lua to build your games. The whole API can be accessed either way. So, every game created in this book can be done in either language (and yes, that includes the Box2D API.)

And this is it for the book. I hope you're not too tired to start working on your own ideas. And I hope to see your game sometime soon in an App Store near me!

# Appendix A. Vector Calculations with Cocos2d-x

This appendix will cover some of the math concepts used in [Chapter 5](part0072_split_000.html#page "Chapter 5. On the Line – Rocket Through"), *On the Line – Rocket Through*, in a little more detail.

# What are vectors?

First, let's do a quick refresh on vectors and the way you can use Cocos2d-x to deal with them.

So what is the difference between a vector and a point? At first, they seem to be the same. Consider the following point and vector:

*   Point (2, 3.5)
*   Vec2 (2, 3.5)

The following figure illustrates a point and a vector:

![What are vectors?](img/00044.jpeg)

In this figure, they each have the same value for *x* and *y*. So what's the difference?

With a vector, you always have extra information. It is as if, besides those two values for *x* and *y*, we also have the *x* and *y* of the vector's origin, which in the previous figure we can assume to be point (0, 0). So the vector is *moving* in the direction described from point (0, 0) to point (2, 3.5). The extra information we can derive then from vectors is direction and length (usually referred to as magnitude).

It's as if a vector is a person's stride. We know how long each step is, and we know the direction in which the person is walking.

In game development, vectors can be used, among other things, to describe movement (speed, direction, acceleration, friction, and so on) or the combining forces acting upon a body.

## The vector methods

There is a lot you can do with vectors, and there are many ways to create them and manipulate them. And Cocos2d-x comes bundled with helper methods that will take care of most of the calculations for you. Here are some examples:

*   You have a vector, and you want to get its angle—use `getAngle()`
*   You want the length of a vector—use `getLength()`
*   You want to subtract two vectors; for example, to reduce the amount of movement of a sprite by another vector—use `vector1 - vector2`
*   You want to add two vectors; for example, to increase the amount of movement of a sprite by another vector—use `vector1 + vector2`
*   You want to multiply a vector; for example, applying a friction value to the amount of movement of a sprite—use `vector1 * vector2`
*   You want the vector that is perpendicular to another (also known as a vector's normal)—use `getPerp()` or `getRPerp()`
*   And, most importantly for our game example, you want the dot product of two vectors—use `dot(vector1, vector2)`

Now let me show you how to use these methods in our game example.

# Using ccp helper methods

In the example of *Rocket Through*, the game we developed in [Chapter 5](part0072_split_000.html#page "Chapter 5. On the Line – Rocket Through"), *On the Line – Rocket Through*, we used vectors to describe movement, and now I want to show you the logic behind some of the methods we used to handle vector operations and what they mean.

## Rotating the rocket around a point

Let's start, as an example, with the rocket sprite moving with a vector of (5, 0):

![Rotating the rocket around a point](img/00045.jpeg)

We then draw a line from the rocket, say from point **A** to point **B**:

![Rotating the rocket around a point](img/00046.jpeg)

Now we want the rocket to rotate around point **B**. So how can we change the rocket's vector to accomplish that? With Cocos2d-x, we can use the helper point method `rotateByAngle` to rotate a point around any other point. In this case, we rotate the rocket's position point around point **B** by a certain angle.

But here's a question – in which direction should the rocket rotate?

![Rotating the rocket around a point](img/00047.jpeg)

By looking at this figure, you know that the rocket should rotate clockwise, since it's moving towards the right. But programmatically, how could we determine that, and in the easiest way possible? We can determine this by using vectors and another property derived from them: the dot product.

## Using the dot product of vectors

The dot product of two vectors describes their angular relationship. If their dot product is greater than zero, the two vectors form an angle smaller than 90 degrees. If it is less than zero, the angle is greater than 90 degrees. And if it is equal to zero, the vectors are perpendicular. Have a look at this descriptive figure:

![Using the dot product of vectors](img/00048.jpeg)

But one other way to think about this is that if the dot product is a positive value, then the vectors will "point" in the same direction. If it is a negative value, they point in opposite directions. How can we use that to help us?

A vector will always have two perpendiculars, as shown in the following figure:

![Using the dot product of vectors](img/00049.jpeg)

These perpendiculars are often called right and left, or clockwise and counterclockwise perpendiculars, and they are themselves vectors, known as normals.

Now, if we calculate the dot product between the rocket's vector and each of the perpendiculars on line **AB**, you can see that we can determine the direction the rocket should rotate in. If the dot product of the rocket and the vector's right perpendicular is a positive value, it means the rocket is moving towards the right (clockwise). If not, it means the rocket is moving towards the left (counterclockwise).

![Using the dot product of vectors](img/00050.jpeg)

The dot product is very easy to calculate. We don't even need to bother with the formula (though it's a simple one), because we can use the `d` `ot(vector1, vector2)` method.

So we have the vector for the rocket already. How do we get the vector for the normals? First, we get the vector for the **AB** line. We use another method for this – `point1 - point2`. This will subtract points **A** and **B** and return a vector representing that line.

Next, we can get the left and right perpendiculars of that line vector with the `getPerp()` and `getRPerp()` methods respectively. However, we only need to check one of these. Then we get the dot product with `dot(rocketVector, lineNormal)`.

If this is the correct normal, meaning the value for the dot product is a positive one, we can rotate the rocket to point to this normal's direction; so the rocket will be at a 90-degree angle with the line at all times as it rotates. This is easy, because we can convert the normal vector to an angle with the `getAngle()` method. All we need to do is apply that angle to the rocket.

But how fast should the rocket rotate? We'll see how to calculate that next.

## Moving from pixel-based speed to angular-based speed

When rotating the rocket, we still want to show it moving at the same speed as it was when moving in a straight line, or as close to it as possible. How do we do that?

![Moving from pixel-based speed to angular-based speed](img/00051.jpeg)

Remember that the vector is being used to update the rocket's position in every iteration. In the example I gave you, the (5, 0) vector is currently adding 5 pixels to the x position of the rocket in every iteration.

Now let's consider an angular speed. If the angular speed were 15 degrees, and we kept rotating the rocket's position by that angle, it would mean the rocket would complete a full circle in 24 iterations. Because 360 degrees of a full circle divided by 15 degrees equals 24.

But we don't have the correct angle yet; we only have the amount in pixels the rocket moves in every iteration. But math can tell us a lot here.

Math says that the length of a circle is *twice the value of Pi, multiplied by the radius of the circle*, usually written as *2πr*.

We know the radius of the circle we want the rocket to describe. It is the length of the line we drew.

![Moving from pixel-based speed to angular-based speed](img/00052.jpeg)

With that formula, we can get the length in pixels of that circle, also known as its circumference. Let's say the line has a length of 100 pixels; this would mean the circle about to be described by the rocket has a length (or circumference) of 628.3 pixels (2 * π * 100).

With the speed described in the vector (5, 0), we can determine how long it would take the rocket to complete that pixel length. We don't need this to be absolutely precise; the last iteration will most likely move beyond that total length, but it's good enough for our purposes.

![Moving from pixel-based speed to angular-based speed](img/00053.jpeg)

When we have the total number of iterations to complete the length, we can convert that to an angle. So, if the iteration value is 125, the angle would be 360 degrees divided by 125; that is, 2.88\. That would be the angle required to describe a circle in 125 iterations.

![Moving from pixel-based speed to angular-based speed](img/00054.jpeg)

Now the rocket can change from pixel-based movement to angular-based movement without much visual change.

# Appendix B. Pop Quiz Answers

# Chapter 4, Fun with Sprites – Sky Defense

## Pop quiz – sprites and actions

| Q1 | 2 |
| Q2 | 1 |
| Q3 | 3 |
| Q4 | 4 |

# Chapter 8, Getting Physical – Box2D

## Pop quiz

| Q1 | 3 |
| Q2 | 2 |
| Q3 | 1 |
| Q4 | 3 |