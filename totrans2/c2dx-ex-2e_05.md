# Chapter 5. On the Line – Rocket Through

*In our third game, Rocket Through, we'll use particle effects to spice things up a bit, and we'll use DrawNode to make our own OpenGL drawings on screen. And be advised, this game uses quite a bit of vector math, but luckily for us, Cocos2d-x comes bundled with a sweet pack of helper methods to deal with that as well.*

You will learn:

*   How to load and set up a particle system
*   How to draw primitives (lines, circles, and more) with `DrawNode`
*   How to use vector math helper methods included in Cocos2d-x

# The game – Rocket Through

In this sci-fi version of the classic Snake game engine, you control a rocket ship that must move around seven planets, collecting tiny supernovas. But here's the catch: you can only steer the rocket by rotating it around pivot points put in place through `touch` events. So the vector of movement we set for the rocket ship is at times linear and at times circular.

## The game settings

This is a universal game designed for the regular iPad and then scaled up and down to match the screen resolution of other devices. It is set to play in portrait mode and it does not support multitouches.

## Play first, work later

Download the `4198_05_START_PROJECT.zip` and `4198_05_FINAL_PROJECT.zip` files from this book's **Support** page.

You will, once again, use the **Start Project** option to work on; this way, you won't need to type logic or syntax already covered in previous chapters. The **Start Project** option contains all the resource files and all the class declarations as well as placeholders for all the methods inside the classes' implementation files. We'll go over these in a moment.

You should run the final project version to acquaint yourself with the game. By pressing and dragging your finger on the rocket ship, you draw a line. Release the touch and you create a pivot point. The ship will rotate around this pivot point until you press on the ship again to release it. Your aim is to collect the bright supernovas and avoid the planets.

![Play first, work later](img/00016.jpeg)

## The start project

If you run the **Start Project** option, you should see the basic game screen already in place. There is no need to repeat the steps we've taken in our previous tutorial to create a batch node and place all the screen sprites. We once again have a `_gameBatchNode` object and a `createGameScreen` method.

But by all means, read through the code inside the `createGameScreen` method. What is of key importance here is that each planet we create is stored inside the `_planets` vector. And we also create our `_rocket` object (the `Rocket` class) and our `_lineContainer` object (the `LineContainer` class) here. More on these soon.

In the **Start Project** option, we also have our old friend `GameSprite`, which extends `Sprite` here with an extra method to get the `radius()` method of our sprites. The `Rocket` object and all the planets are `GameSprite` objects.

## Screen settings

So if you have the **Start Project** option opened in Xcode, let's review the screen settings for this game in `AppDelegate.cpp`. Inside the `applicationDidFinishLaunching` method, you should see this:

[PRE0]

So we basically start the same way we did in the previous game. The majority of sprites in this game are circle-shaped and you may notice some distortion in different screens; you should test the same configuration but using different `ResolutionPolicies`, such as `SHOW_ALL`.

# So what are particles?

Particles or particle systems are a way to add special effects to your applications. In general terms this is achieved by the use of a large number of tiny textured sprites (particles), which are animated and run through a series of transformations. You can use these systems to create smoke, explosions, sparks, lightening, rain, snow, and other such effects.

As I mentioned in [Chapter 1](part0016_split_000.html#page "Chapter 1. Installing Cocos2d-x"), *Installing Cocos2d-x*, you should seriously consider getting yourself a program to help you design your particle systems. In this game, the particles were created in ParticleDesigner.

It's time to add them to our game!

# Time for action – creating particle systems

For particles, all we need is the XML file describing the particle system's properties.

1.  So let's go to `GameLayer.cpp`.
2.  The game initializes by calling `createGameScreen`, which is already in place, then `createParticles` and `createStarGrid`, which is also implemented. So let's go over the `createParticles` method now.
3.  Go to that method in `GameLayer.cpp` and add the following code:

    [PRE1]

## *What just happened?*

We created our first particles. ParticleDesigner exports the particle system data as a `.plist` file, which we used to create our `ParticleSystemQuad` objects. You should open one of these files in Xcode to review the number of settings used in a particle system. From Cocos2d-x you can modify any of these settings through setters inside `ParticleSystem`.

The particles we'll use in this game are as follows:

*   `_jet`: This is attached to the `_rocket` object and it will trail behind the `_rocket` object. We set the system's angle and source position parameters to match the `_rocket` sprite.
*   `_boom`: This is the particle system used when `_rocket` explodes.
*   `_comet`: This is a particle system that moves across the screen at set intervals and can collide with `_rocket`.
*   `_pickup`: This is used when a star is collected.
*   `_warp`: This marks the initial position of the rocket.
*   `_star`: This is the particle system used for the star that the rocket must collect.

The following screenshot shows these various particles:

![What just happened?](img/00017.jpeg)

All particle systems are added as children to `GameLayer`; they cannot be added to our `SpriteBatchNode` class. And you must call `stopSystem()` on each system as they're created otherwise they will start playing as soon as they are added to a node.

In order to run the system, you make a call to `resetSystem()`.

### Note

Cocos2d-x comes bundled with some common particle systems you can modify for your own needs. If you go to the `test` folder at: `tests/cpp-tests/Classes/ParticleTest`, you will see examples of these systems being used. The actual particle data files are found at: `tests/cpp-tests/Resources/Particles`.

# Creating the grid

Let's take some time now to review the grid logic in the game. This grid is created inside the `createStarGrid` method in `GameLayer.cpp`. What the method does is determine all possible spots on the screen where we can place the `_star` particle system.

We use a C++ vector list called `_grid` to store the available spots:

[PRE2]

The `createStarGrid` method divides the screen into multiple cells of 32 x 32 pixels, ignoring the areas too close to the screen borders (`gridFrame`). Then we check the distance between each cell and the planet sprites stored inside the vector `_planets`. If the cell is far enough from the planets, we store it inside the `_grid` vector as `Point`.

In the following figure, you can get an idea of the result we're after. We want all the white cells not overlapping any of the planets.

![Creating the grid](img/00018.jpeg)

We output a message to the console with `Log` stating how many cells we end up with:

[PRE3]

This `vector` list will be shuffled at each new game, so we end up with a random sequence of possible positions for our star:

[PRE4]

This way we never place a star on top of a planet or so close to it that the rocket could not reach it without colliding with the planet.

# Drawing primitives in Cocos2d-x

One of the main elements in the game is the `LineContainer.cpp` class. It is a `DrawNode` derived class that allows us to draw lines and circles on the screen.

`DrawNode` comes bundled with a list of drawing methods you can use to draw lines, points, circles, polygons, and so on.

The methods we'll use are `drawLine` and `drawDot`.

# Time for action – let's do some drawing!

Time to implement the drawing inside `LineContainer.cpp`. You will notice that this class already has most of its methods implemented, so you can save a little typing. I'll go over what these methods represent once we add the game's main update method. But basically `LineContainer` will be used to display the lines the player draws on screen to manipulate `_rocket` sprite, as well as display an energy bar that acts as a sort of timer in our game:

1.  What we need to change here is the `update` method. So this is what you need to type inside that method:

    [PRE5]

2.  We end our drawing calls by drawing the energy bar in the same `LineContainer` node:

    [PRE6]

## *What just happened?*

You just learned how to draw inside `DrawNode`. One important line in that code is the `clear()` call. It clears all the drawings in that node before we update them with their new state.

In `LineContainer`, we use a `switch` statement to determine how to draw the player's line. If the `_lineType` property is set to `LINE_NONE`, we don't draw anything (this will, in effect, clear the screen of any drawings done by the player).

If `_lineType` is `LINE_TEMP`, this means the player is currently dragging a finger away from the `_rocket` object, and we want to show a white line from the `_rocket` current position to the player's current touch position. These points are called `tip` and `pivot`, respectively.

We also draw a dot right on the `pivot` point.

[PRE7]

If `_lineType` is `LINE_DASHED`, it means the player has removed his or her finger from the screen and set a new pivot point for the `_rocket` to rotate around. We draw a white dotted line, using what is known as the Bezier linear formula to draw a series of tiny circles from the `_rocket` current position and the `pivot` point:

[PRE8]

And finally, for the energy bar, we draw a black line underneath an orange one. The orange one resizes as the value for `_energy` in `LineContainer` is reduced. The black one stays the same and it's here to show contrast. You layer your drawings through the order of your `draw` calls; so the first things drawn appear underneath the latter ones.

# The rocket sprite

Time to tackle the second object in the game: the rocket.

Once again, I already put in place the part of the logic that's old news to you. But please review the code already inside `Rocket.cpp`. We have a method to reset the rocket every time a new game starts (`reset`), and a method to show the selected state of the rocket (`select(bool flag)`) by changing its displayed texture:

[PRE9]

This will either show the rocket with a glow around it, or not.

And finally a method to check collision with the sides of the screen (`collidedWithSides`). If there is a collision, we adjust the rocket so it moves away from the screen side it collided with, and we release it from any pivot point.

What we really need to worry about here is the rocket's `update` method. And that's what we'll add next.

# Time for action – updating our rocket sprite

The game's main loop will call the rocket's `update` method in every iteration.

1.  Inside the empty `update` method in `Rocket.cpp`, add the following lines:

    [PRE10]

2.  Here we are saying, if the rocket is not rotating `(_rotationOrientation == ROTATE_NONE`), just move it according to its current `_vector`. If it is rotating, then use the Cocos2d-x helper `rotateByAngle` method to find its next position around its pivot point:![Time for action – updating our rocket sprite](img/00019.jpeg)
3.  The method will rotate any point around a pivot by a certain angle. So we rotate the rocket's updated position around its pivot (determined by the player) using a property of `Rocket` class called _`angularSpeed`; we'll see in a moment how it gets calculated.
4.  Based on whether the rocket is rotating clockwise or counterclockwise, we adjust its rotation so the rocket will be at a 90 degree angle with the line drawn between the rocket and its pivot point. Then we change the rocket's movement vector based on this rotated angle, and we wrap the value of that angle between 0 and 360.
5.  Finish up the `update` method with these lines:

    [PRE11]

6.  With these lines we determine the new target rotation of our sprite and we run an animation to rotate the rocket to its target rotation (with a bit of a spring to it).

## *What just happened?*

We just wrote the logic that will move the rocket around the screen, whether the rocket is rotating or not.

So when the player picks a pivot point for the `_rocket` sprite, this pivot point is passed to both `Rocket` and `LineContainer`. The former will use it to rotate its vector around it and the latter will use it to draw a dotted line between `_rocket` and the `pivot` point.

### Note

We can't use `Action` to rotate the sprite because the target rotation is updated too many times in our logic, and `Action` needs time to initialize and run.

So it's time to code the touch events to make all that logic fall into place.

# Time for action – handling touches

We need to implement `onTouchBegan`, `onTouchMoved`, and `onTouchEnded`.

1.  Now in `GameLayer.cpp`, inside `onTouchBegan`, add the following lines:

    [PRE12]

    When a touch begins, we only need to determine whether it's touching the ship. If it is, we set our `_drawing` property to `true`. This will indicate we have a valid point (one that began by touching the `_rocket` sprite).

2.  We clear any lines we may be currently drawing in `_lineContainer` by calling `setLineType( LINE_NONE )`, and we make sure `_rocket` will not rotate until we have a pivot point by releasing `_rocket (setRotationOrientation ( ROTATE_NONE ))`, so it will continue to move on its current linear trajectory `(_vector`).
3.  From here, we begin drawing a new line with the next `onTouchMoved` method. Inside that method, we add the following lines:

    [PRE13]

4.  We'll handle touch moved only if we are using `_drawing`, which means the player has pressed on the ship and is now dragging his or her finger across the screen.

    Once the distance between the finger and `_rocket` is greater than the _`minLineLength` distance we stipulate in game `init`, then we give a visual cue to the player by adding a glow around `_rocket (_rocket->select(true))`, and we draw the new line in `_lineContainer` by passing it the touch's current position and setting the line type to `LINE_TEMP`. If the minimum length is not reached, we don't show a line and nor do we show the player selected.

5.  Next comes `onTouchEnded`. There is logic in place already inside our `onTouchEnded` method which deals with game states. You should uncomment the calls to `resetGame` and add a new `else if` statement inside the method:

    [PRE14]

6.  If the game is paused, we change the texture in the `_pauseBtn` sprite through `Sprite->setDisplayFrame`, and we start running the game again.
7.  Now we begin handling the touch. First, we determine whether it's landing on the `Pause` button:

    [PRE15]

8.  If so, we change the game state to `kGamePaused`, change the texture on the `_pauseBtn` sprite (by retrieving another sprite frame from `SpriteFrameCache`), stop running the game (pausing it), and return from the function.
9.  We can finally do something about the rocket ship. So, continuing inside the same `if(touch != nullptr) {` conditional seen previously, add these lines:

    [PRE16]

10.  We start by deselecting the `_rocket` sprite, and then we check whether we are currently showing a temporary line in `_lineContainer`. If we are, this means we can go ahead and create our new pivot point with the player's released touch. We pass this information to `_lineContainer` with our `setPivot` method, along with the line length. The `_rocket` sprite also receives the pivot point information.

    Then, things get hairy! The `_rocket` sprite is moving at a pixel-based speed. Once `_rocket` starts rotating, it will move at an angular-based speed through `Point.rotateByAngle`. So the following lines are added to translate the `_rocket` current pixel-based speed into angular speed:

    [PRE17]

11.  It grabs the length of the circumference about to be described by `_rocket (line length * 2 * PI)` and divides it by the rocket's speed, getting in return the number of iterations needed for the rocket to complete that length. Then the 360 degrees of the circle is divided by the same number of iterations (but we do it in radians) to arrive at the fraction of the circle that the rocket must rotate at each iteration: its angular speed.
12.  What follows next is even more math, using the amazingly helpful methods from Cocos2d-x related to vector math (`Point.getRPerp`, `Point.dot`, `Point.subtract`, to name a few) some of which we've seen already in the `Rocket` class:

    [PRE18]

13.  What they do here is determine which direction the rocket should rotate to: clockwise or counterclockwise, based on its current vector of movement.
14.  The line the player just drew between `_rocket` and pivot point, which we get by subtracting (`Point.subtract`) those two points, has two perpendicular vectors: one to the right (clockwise) that you get through `Point.getRPerp` and one to the left (counterclockwise) that you get through `Point.getPerp`. We use the angle of one of these vectors as the `_rocket` target rotation so the rocket will rotate to be at 90 degrees with the line drawn in `LineContainer`. And we find the correct perpendicular through the dot product of the `_rocket` current vector and one of the perpendiculars (`Point.dot`).

## *What just happened?*

I know. A lot of math and all at once! Thankfully, Cocos2d-x made it all much easier to handle.

We just added the logic that allows the player to draw lines and set new pivot points for the `_rocket` sprite.

The player will steer the `_rocket` sprite through the planets by giving the rocket a pivot point to rotate around. And by releasing the `_rocket` from pivot points, the player will make it move in a straight line again. All that logic gets managed here in the game's touch events.

And don't worry about the math. Though understanding how to deal with vectors is a very useful tool in any game developer's toolbox, and you should definitely research the topic, there are countless games you can still build with little or no math; so cheer up!

# The game loop

It's time to create our good old ticker! The main loop will be in charge of collision detection, updating the points inside `_lineContainer`, adjusting our `_jet` particle system to our `_rocket` sprite, and a few other things.

# Time for action – adding the main loop

Let's implement our main `update` method.

1.  In `GameLayer.cpp`, inside the `update` method, add the following lines:

    [PRE19]

    We check to see if we are not currently on pause. Then, if there is a line for our ship that we need to show in `_lineContainer`, we update the line's `tip` point with the `_rocket` current position.

    We run collision checks between `_rocket` and the screen sides, update the `_rocket` sprite, and position and rotate our `_jet` particle system to align it with the `_rocket` sprite.

2.  Next we update `_comet` (its countdown, initial position, movement, and collision with `_rocket` if `_comet` is visible):

    [PRE20]

3.  Next we update `_lineContainer`, and slowly reduce the opacity of the `_rocket` sprite based on the `_energy` level in `_lineContainer`:

    [PRE21]

    This will add a visual cue for the player that time is running out as the `_rocket` sprite will slowly turn invisible.

4.  Run collision with planets:

    [PRE22]

5.  And collision with the star:

    [PRE23]

    When we collect `_star`, we activate the `_pickup` particle system on the spot where `_star` was, we fill up the player's energy level, we make the game slightly harder, and we immediately reset `_star` to its next position to be collected again.

    The score is based on the time it took the player to collect `_star`.

6.  And we keep track of this time on the last lines of `update` where we also check the energy level:

    [PRE24]

## *What just happened?*

We added the main loop to our game and finally have all the pieces talking to each other. But you probably noticed quite a few calls to methods we have not implemented yet, such as `killPlayer` and `resetStar`. We'll finish our game logic with these methods.

# Kill and reset

It's that time again! Time to kill our player and reset the game! We also need to move the `_star` sprite to a new position whenever it's picked up by the player.

# Time for action – adding our resets and kills

We need to add logic to restart our game and to move our pickup star to a new position. But first, let's kill the player!

1.  Inside the `killPlayer` method, add the following lines:

    [PRE25]

2.  Inside `resetStar`, add the following lines:

    [PRE26]

3.  And finally, our `resetGame` method:

    [PRE27]

## *What just happened?*

That's it. We're done. It took more math than most people are comfortable with. But what can I tell you, I just love messing around with vectors!

Now, let's move on to Android!

# Time for action – running the game in Android

Follow these steps to deploy the game to Android:

1.  Open the manifest file and set the `app` orientation to `portrait`.
2.  Next, open the `Android.mk` file in a text editor.
3.  Edit the lines in `LOCAL_SRC_FILES` to read:

    [PRE28]

4.  Import the game into Eclipse and build it.
5.  Save and run your application. This time, you can try out different size screens if you have the devices.

## *What just happened?*

You now have Rocket Through running in Android.

## Have a go hero

Add logic to the `resetStar` method so that the new position picked is not too close to the `_rocket` sprite. So, make the function a recurrent one until a proper position is picked.

And take the `warp` particle system, which right now does not do a whole lot, and use it as a random teleport field so that the rocket may get sucked in by a randomly placed warp and moved farther away from the target star.

# Summary

Congratulations! You now have enough information about Cocos2d-x to produce awesome 2D games. First sprites, then actions, and now particles.

Particles make everything look shiny! They are easy to implement and are a very good way to add an extra bit of animation to your game. But it's very easy to overdo it, so be careful. You don't want to give your players epileptic fits. Also, running too many particles at once could stop your game in its tracks.

In the next chapter, we'll see how to use Cocos2d-x to quickly test and develop game ideas.