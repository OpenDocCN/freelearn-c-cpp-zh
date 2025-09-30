# Chapter 5. Hit and Run

You've already come a long way since beginning the book at the first chapter! You have managed to render moving images to the screen and control their movement. You are well on your way toward creating a great game. The next step is to code the interactions between various objects in the game.

This chapter will explain how to implement collision detection. Collision detection determines how objects interact with each other when they are in the same location. Topics will include:

*   **Boundary detection**: When an object reaches the top, bottom, left, or right edge of the screen, what should happen? There are a surprising number of choices and you get to choose what to do.
*   **Collision detection**: There are various scenarios that we often need to check to determine whether two objects have hit each other. We will cover circular and rectangular collision detection algorithms. We will also discuss when each type of collision detection is appropriate to use.

# Out of bounds!

If you run our current game, you will notice that the robot will go off the screen if you allow him to continue moving to the left or right. When he reaches the edge of the screen, he will keep on moving until he is no longer visible. If you reverse his direction and make him move the same number of steps now, he will reappear on the screen.

Whenever an object reaches the edge of the screen, we often want it to do something special, such as stopping, or turning around. The code that determines when an object has reached a screen edge is known as **boundary checking**. There are many possibilities for what we can do when an object reaches a boundary:

*   Stop the object
*   Allow the object to continue past the border (and therefore, disappear)
*   Allow the object to continue past the border and reappear at the opposite border (ever played the arcade version of Asteroids?)
*   Scroll the camera and the screen along with the object (aka Mario)
*   Allow the object to rebound off the border (ever played Breakout?)

As our Robo is controlled by the player, we will simply force him to stop moving when he has reached the edge of the screen.

## Getting anchored

In order to implement boundary checking, you must first know the exact anchor point of the image. Technically, the anchor point could be anywhere, but the two most common locations are the top-left corner and the center of the image.

First, let's see what happens if we just ignore the anchor point. Open the **RoboRacer2D** project and then open `RoboRacer2D.cpp`.

Insert the following function:

[PRE0]

Here is what this code is doing for us:

*   The function accepts a sprite as its parameter
*   The function first checks to see whether the `x` position of the sprite is less than `0`, where `0` is the `x` coordinate of the far-left edge of the screen
*   The function then checks to see whether the `x` position of the sprite is greater than the screen width, where `screen_width` is the `x` coordinate of the far-right edge of the screen
*   If either check is `true`, the sprite's velocity is set to `0`, effectively stopping the sprite in its tracks

Now, add the highlighted line of code to the `Update` function right after `ProcessInput` in `RoboRacer2D.cpp`:

[PRE1]

This simply calls the `CheckBoundaries` function that we just created and passes in the `player` object.

Now, run the program. Move Robo until he reaches the far left of the screen. Then run him to the far right of the screen. Does anything seem wrong about the way we have implemented our boundary checking?

### Tip

Ignore the way the background scrolls off to the side. We'll fix this shortly.

**Problem 1**: Robo doesn't seem to hit the boundary on the left.

The following screenshot shows you what happens if you allow Robo to go to the far left of the screen. He appears to stop just before reaching the edge. Although you can't see it in the following screenshot, there is a shadow that always extends to the left edge of the robot. It is the left edge of the shadow that is being detected as the edge of the image.

It turns out that the default anchor point for images loaded by our image loading routine is, in fact, the upper-left corner.

![Getting anchored](img/8199OS_05_01.jpg)

**Problem 2**: Robo moves completely off the screen to the right.

The following screenshot shows you what occurs if you allow Robo to continue traveling to the right. Now that you understand that the anchor point is at the upper-left, you may already understand what is happening.

As the boundary checking is based on the `x` coordinate of the sprite, by the time the upper-left hand corner exceeds the screen width, the entire sprite has already moved off the screen. The grayscale image of the robot shows us where his actual position would be if we could see him:

![Getting anchored](img/8199OS_05_02.jpg)

**Problem 3**: Once Robo reaches the far left or far right of the screen, he gets stuck. Changing his direction seems to have no effect!

This problem is known as **embedding**. Here is what has happened:

*   We continued check Robo's position until his *x* coordinate exceeded a threshold.
*   Once he exceeded that threshold, we set his velocity to `0`.
*   Now that Robo's *x* coordinate exceeds that threshold, it will always exceed that threshold. Any attempt to move him in the opposite direction will trigger the boundary check, which will discover that Robo's *x* coordinate still exceeds the threshold and his velocity will be set to `0`.

The solution is to set Robo's position to the other side of threshold as soon as we discover he has crossed it. We will add this correction, but first we have to understand collision rectangles.

## Collision rectangles

Take a look at the following image of Robo. The solid rectangle represents the boundaries of the texture. The dotted rectangle represents the area that we actually want to consider for boundary and collision detection. This is known as the **collision rectangle**.

![Collision rectangles](img/8199OS_05_03.jpg)

Comparing the two rectangles, here is what we would have to do to convert the texture rectangle to be the collision rectangle:

*   Add about 34 pixels to the left texture boundary
*   Subtract about 10 pixels from the right texture boundary
*   Both the top and right boundaries require no adjustment

Let's enhance the sprite class by adding functionality to define a collision rectangle.

Open `Sprite.h` and add the following member variable:

[PRE2]

Then add the two accessor methods:

[PRE3]

The implementation for `GetCollisionRect` is a little more complex, so we will put that code into `Sprite.cpp`:

[PRE4]

Here's what we are doing:

*   `m_collision`: This will hold four offset values. These values will represent a number that must be added to the texture's bounding rectangle to get to the collision rectangle that we desire.
*   `SetCollisionRectOffset`: This accepts a `Rect` parameter that contains the four offsets—top, bottom, left, and right—that must be added to the top, bottom, left, and right of the texture boundaries to create the collision rectangle.
*   `GetCollisionRect`: This returns the collision rectangle that we can actually use when checking boundaries and checking for collisions. This is calculated by adding the width and height to the sprite's current anchor point (the top-left corner), and then adjusting it by the values in `m_collision`.

Note that `GetCollisionRect` is dynamic; it always returns the current collision rectangle based on the sprite's current position. Thus, we are returning the actual top, bottom, left, and right boundaries that need to be checked at any moment in the game.

If you look closely at the design, you should be able to see that if no collision rectangle is defined, `GetCollisionRect` will return a collision rectangle determined by the texture's rectangle. Therefore, this new design allows us to use the texture rectangle as the collision rectangle by default. On the other hand, if we want to specify our own collision rectangle, we can do so using `SetCollisionRectOffset`.

Just to be safe, we will want to initialize m_collision by adding the following lines to the constructor:

[PRE5]

Now that we have the code to support a collision rectangle, we need to define the collision rectangle for the robot's sprites. Go to the `LoadTextures` function in `RoboRacer2D.cpp` and add the following highlighted lines just before the `return true` line of code:

[PRE6]

Remember, only add the preceding code that is highlighted. The last line of the code is shown to provide context.

We are now going to rewrite our boundary detection function to take advantage of the collision rectangle. Along the way we will solve all three of the problems that we encountered in our first attempt. The current code uses the anchor point of the image, which doesn't accurately reflect the actual boundaries that we want to check. The new code will use the collision rect. Replace the `CheckBoundaries` function in RoboRacer2D with the following code:

[PRE7]

This code uses the collision rectangle defined for the sprite that is being checked. As we already discussed earlier, `GetCollisionRect` returns the top, bottom, left, and right boundaries for us based on the current position of the sprite. This greatly simplifies our code! Now, we just check to see whether the left edge of the sprite is less than zero or whether the right edge of the sprite is greater than zero, and we're done!

![Collision rectangles](img/8199OS_05_04.jpg)

## Embedding

Hurrah! Robo now successfully stops at the edge of the screen (only the right-hand side is shown in the preceding image). But boo! He still gets stuck! As we mentioned earlier, this problem is called embedding. If we zoom in, we can see what's going on:

![Embedding](img/8199OS_05_05.jpg)

The vertical line represents the edge of the screen. By the time Robo has stopped, his right edge has already exceeded the right edge of the screen, so we stop him. Unfortunately, even if we try to turn him around to go in the other direction, the `CheckBoundaries` function will check on the very next frame, before Robo has a chance to start moving back:

![Embedding](img/8199OS_05_06.jpg)

According to the boundary check, the right edge of Robo is still beyond the right edge of the screen, so once again Robo's velocity is set to zero. Robo is stopped before he can even take a step!

Here is the solution; as soon as we detect that Robo has exceeded the boundary, we set his velocity to zero and we reposition Robo to just the other side of the boundary:

![Embedding](img/8199OS_05_07.jpg)

Now, Robo will be able to move as long as he goes in the other direction.

To implement this change, we are once again going to change the `CheckBoundaries` function:

[PRE8]

The highlighted lines show the added code. Basically, we perform the following actions:

*   Calculate how far past the boundary Robo has gone
*   Adjust his position by that much so that he is now positioned right at the boundary

You'll notice that we also filled out the function to handle the top and bottom boundaries so that the boundary checking can be used for any sprite travelling in any direction.

## Fixing the background

Now that we have Robo moving the way we want him to, two new problems have cropped up for the background image:

1.  When Robo stops, the background keeps scrolling.
2.  When the background image ends at either the right or the left, it slides off the screen and we are left with a black background.

Before we continue on with collision detection, let's fix the background. First, we will add the following function to `RoboRacer2D.cpp`:

[PRE9]

This code is very similar to the boundary checking code. If the background anchor point moves far enough to the left to expose the right edge of the texture, it will be reset. If the background anchor point moves far enough to the right to expose the left edge of the texture, it will be reset.

Now, add the highlighted line of code to the `Update` function right after the call to `CheckBoundaries` in `RoboRacer2D.cpp`:

[PRE10]

The background should now run from edge to edge. Play the game and take a coffee break. You deserve it!

# Collideables

There are many times that we may want to check and see whether objects in the game have collided with each other. We may want to see whether the player has struck an obstacle or an enemy. We may have objects that the player can pick up, often called **pickups** or **powerups**.

Collectively, objects in the game that can collide with other objects are known as collideables. When we created our `Sprite` class, we actually it designed for this. Looking at the class constructor, you will notice that member variable `m_isCollideable` is set to `false`. When we write our collision detection code, we will ignore objects that have `m_isCollideable` set to `false`. If we want to allow an object to collide with other objects, we have to make sure to set `m_collideable` to `true`.

## Ready to score

To keep our design simple, we are going to create one enemy and one pickup. Running into an enemy will subtract points from the player's score, while running into the pickup will increase the player's score. We will add some additional code to the sprite class to support this feature.

First, let's add some new member variables. Declare a new variable in `Sprite.h`:

[PRE11]

Then add the following methods:

[PRE12]

With these changes, every sprite will have an intrinsic value. If the value is positive, then it is a reward. If the value is negative, then it is a penalty.

Don't forget to initialize `m_value` to zero in the `Sprite` class constructor!

## A friend indeed

Let's add the sprite for our pickup. In this case, the pickup is a can of oil to keep Robo's joints working smoothly.

Add the following sprite definitions to RoboRacer2D:

[PRE13]

Now, we will set up the sprite. Add the following code to `LoadTextures`:

[PRE14]

This code is essentially the same code that we used to create all of our sprites. One notable difference is that we use the new `SetValue` method to add a value to the sprite. This represents how many points the player will earn for the collection of this pickup.

## Time to spawn

Note that we have set the sprite as inactive and invisible. Now, we will write a function to randomly spawn the pickup. First, we need to add two more C++ headers. In `RoboRacer2D.cpp` add the following headers:

[PRE15]

We need `stdlib` for the `rand` function and `time` to give us a value to seed the random generator.

### Tip

Random numbers are generated from internal tables. In order to guarantee that a different random number is chosen each time the program is started, you first seed the random number generator with a value that is guaranteed to be different each time you start the program. As the time that the program is started will always be different, we often use time as the seed.

Next, we need a timer. Declare the following variables in `RoboRacer2D.cpp`:

[PRE16]

The threshold will be the number of seconds that we want to pass before a pickup is spawned. The timer will start and zero and count up to that number of seconds.

Let's initialize these values in the `StartGame` function. The `StartGame` function is also a great place to seed our random number generator. Add the following three lines of code to the end of `StartGame`:

[PRE17]

The first line seeds the random number generator by passing in an integer representing the current time. The next line sets a spawn threshold of `15` seconds. The third line sets the spawn timer to `0`.

Now, let's create a function to spawn our pickups. Add the following code to `RoboRacer2D.cpp`:

[PRE18]

This code does the following:

*   It checks to make sure that the pickup is not already on the screen
*   If there is no pickup, then the spawn timer is incremented
*   If the spawn timer exceeds the spawn threshold, the pickup is spawned at a random position somewhere within the width of the screen and within the vertical reach of Robo

Don't get too worried about the particular math being used. Your algorithm to position the pickup could be completely different. The key here is that a single pickup will be generated within Robo's reach.

Make sure to add a call to `SpawnPickup` in the `Update` function as well as a line to update the pickup:

[PRE19]

We also need to add a line to `Render` to render the pickup:

[PRE20]

If you run the game right now, then an oil can should be spawned about five seconds after the game starts.

### Tip

The current code has one flaw. It could potentially spawn the pickup right on top of Robo. Once we implement collision detection, the result will be that Robo immediately picks up the oil can. This will happen so quickly that you won't even see it happen. In the name of keeping it simple, we will live with this particular flaw.

# Circular collision detection

One way to detect collision is to see how far each of the objects are from each other's center. This is known as circular collision detection because it treats each object as if it is bound by a circle, and uses the radius of that circle to determine whether the objects are close enough to collide.

Take a look at the following diagram:

![Circular collision detection](img/8199OS_05_08.jpg)

The circles on the left are not colliding, while the circles on the right are colliding. For the non-colliding circles, the distance (*d*) between the center points of the two circles is greater than the sum of the two radii (*r1 + r2*). For the colliding circles, the distance (*d*) between the two centers is less than the sum of the two radii (*r1 + r2*). We can use this knowledge to test any two objects for collision based on the radii of the circles and the distance between the objects center point.

So, how do we use this information?

1.  We will know **r1** and **r2** because we set them when we create the sprite.
2.  We will calculate two legs of a right-triangle using the *x* and *y* coordinates for the center of each circle.
3.  We will calculate *d*, the distance between two center points, using a variant of the **Pythagorean Theorem**.

It will probably hurt your brain a little, but I'd like to refresh your memory one tenet of basic geometry.

## The Pythagorean Theorem

The Pythagorean Theorem allows us to find the distance between any two points in a two-dimensional space if we know the lengths of the line segments that form a right-angle between the points.

![The Pythagorean Theorem](img/8199OS_05_09.jpg)

*a² + b² = c²*

In our case, we are trying to calculate the distance (c) between the two points.

A little algebra will transform this equation to:

![The Pythagorean Theorem](img/81990S_05_11.jpg)

It is computationally expensive to perform the square root. A nice mathematical trick will actually allow us to perform our collision detection without calculating the square root.

If we were to use square roots to do this calculation, here is what that might look like:

[PRE21]

Although this would work, there is a nice little mathematical trick that allows us to accomplish this test without taking the square root. Take a look at this:

[PRE22]

It turns out that we can just keep everything in the equation at the power of 2 and the comparison still works. This is because we are only interested in the relative comparison between the distance and the sum of the radii, not the absolute mathematical values.

### Tip

If the math we presented here boggles your brain, then don't worry too much. Circular collision detection is so common that the math to detect it is generally already built into the game engine that you will use. However, I wanted you to take a little look under the hood. After all, game programming is inherently mathematical, and the more you understand the math, the better you will be at coding.

## Adding the circular collision code

Now, it's time to modify `Sprite.h` to add support for the circular collision detection. First, we need to add some member variables to hold the center point and radius. Add these two properties to `Sprite.h`:

[PRE23]

Then add the following methods declarations:

[PRE24]

These methods allow us to set and retrieve the center point and radius of the sprite. The `GetCenter` method is more than one line, so we will implement it in `Sprite.cpp`:

[PRE25]

An important point to note here is that `m_center` represents an `x` and `y` offset from the sprite's anchor point. So, to return the center point we will add `m_center` to the current position of the sprite and this will give us the current center point of the sprite exactly where it is in the game.

We now need to add the code to perform the collision detection. Add the following code to `Sprite.cpp`:

[PRE26]

As we have already explained the use of the Pythagorean Theorem, this code should actually seem a little familiar to you. Here is what we are doing:

The function accepts one sprite to compare with itself.

*   First, we check to make sure both sprites are collideable.
*   `p1` and `p2` represent the two centers.
*   `x` and `y` represent the lengths of the *a* and *b* sides of a right-angled triangle. Notice that the calculation is simply the difference between the `x` and `y` position of each sprite, respectively.
*   `r1` and `r2` are the radii of the two circles (left as a power of 2).
*   `d` is the distance between the two centers (left as a power of 2).
*   If `d` is less than or equal to the sum of the two radii, then the circles are intersecting.

## Why use circular collision detection?

As we discussed many times, textures are represented as rectangles. In fact, we will take advantage of this when we cover rectangular collision detection later in the chapter. The following figure illustrates how rectangular and circular collision detection differ (the relative sizes are exaggerated to make a point):

![Why use circular collision detection?](img/8199OS_05_10.jpg)

The sprites on the left are colliding using a rectangular bounding box. The sprites on the right are colliding using a bounding circle. In general, using a bounding circle is visually more convincing when we are dealing with rounder shapes.

### Tip

I'll admit the difference in this example is not that big. You could get away with rectangular or circular collision detection in this example. The round nature of the oil can made it a good candidate for circular collision detection. Circular collision detection is really essential if the two objects that are colliding are actually circles (that is, two balls colliding).

With the code that we developed, we need to define the center and radius for any sprites that will use circular collision detection. Add the following code to the `LoadTextures` function in `RoboRacer.cpp`:

[PRE27]

Don't get too worried about the exact values that we are using here. We are basically setting up a bounding circle for Robo and the oil can that match the preceding figure. Robo's bounding circle is set to the middle of the robot, while the oil can's circle is set to the bottom half of the texture.

## Wiring in the collision detection

We are now going to add a new function that will perform all of our collision detection. Add the following function to `RoboRacer2D.cpp`:

[PRE28]

The purpose of this code is to check to see whether the player has collided with the pickup:

*   If the call to `player->IntersectsCircle(pickup)` returns `true`, then the player has collided with the pickup
*   The pickup is deactivated and made invisible
*   The pickup's value is added to the player's value (this will be the base for scoring in a future chapter)
*   The spawn timer is reset

We have two small details left. First, you must add a call to `CheckCollisions` to the `Update` function:

[PRE29]

Secondly, you need to make the player and pickup `collideable`. Add these three lines to the bottom of `LoadTextures` just before the return statement:

[PRE30]

Now, the real fun starts! Play the game and when the oil can spawns, and use Robo to pick it up. Five seconds later another oil can spawns. The fun never ends!

# Rectangular collision detection

Now, we are going to learn how to implement rectangular collision detection. It turns out that both Robo and our enemy (a water bottle) are very rectangular, making rectangular collision detection the best choice.

## The enemy within

Let's introduce our Robo's enemy—a bottle of water to rust his gears. The code for this is included next.

Add the following sprite definition to `RoboRacer2D`:

[PRE31]

Now, we will setup the sprite. Add the following code to `LoadTextures`:

[PRE32]

This code is essentially the same code that we used to create all of our sprites. One notable difference is that we use the new `SetValue` method to add a negative value to the sprite. This is how many points the player will lose if they hit this enemy. We also make sure that we set the enemy to be collideable.

## Spawning the enemy

Just like the pickups, we need to spawn our enemies. We could use the same code as the pickups, but I thought it would be nicer if our enemies worked on a different timer.

Declare the following variables in `RoboRacer2D.cpp`:

[PRE33]

The threshold will be the amount of seconds that we want to pass before an enemy is spawned. The timer will start and zero and count up to that number of seconds.

Let's initialize these values in the `StartGame` function. Add the following two lines of code to the end of `StartGame`:

[PRE34]

We set a spawn threshold of 7 seconds, and set the spawn timer to 0.

Now, let's create a function to spawn our enemies. Add the following code to `RoboRacer2D.cpp`:

[PRE35]

This code does the following:

*   It checks to make sure that the enemy is not already on the screen
*   If there is no enemy, then the spawn timer is incremented
*   If the spawn timer exceeds the spawn threshold, the enemy is spawned at a random position somewhere within the width of the screen and within the vertical reach of Robo

Don't get too worried about the particular math being used. Your algorithm to position the enemy can be completely different. The key here is that a single enemy will be generated within Robo's path.

Make sure to add a call to `SpawnEnemy` in the `Update` function as well as a line to update the enemy:

[PRE36]

We also need to add a line to `Render` to render the enemy:

[PRE37]

If you run the game right now, then a water bottle should be spawned about seven seconds after the game starts.

## Adding the rectangular collision code

As we have mentioned several times, all sprites are essentially rectangles. Visually, if any border of these rectangles overlap, we can assume that the two sprites have collided.

We are going to add a function to our `Sprite` class that determines whether two rectangles are intersecting. Open `Sprite.h` and add the following method declaration:

[PRE38]

Now, let's add the implementation to `Sprite.cpp`:

[PRE39]

Here's how this code works:

*   This function looks really complicated, but it is really only doing a few things.
*   The function accepts a sprite parameter.
*   We set `recta` to be the collision rectangle of the sprite that called the `IntersectsRect` method and set `rectb` to be the collision rectangle of the sprite that was passed in.
*   We then test every possible combination of the position of the vertices in of `recta` to those of `rectb`. If any test is `true`, then we return `true`. Otherwise we return `false`.

The following figure illustrates some of the ways that two rectangles could interact:

![Adding the rectangular collision code](img/8199OS_05_12.jpg)

## Wiring continued

We have already wired in the collision check using `CheckCollisions`. We just need to add the following code to `CheckCollisions` to the check whether the player is colliding with an enemy:

[PRE40]

Now, the real fun starts! Play the game and when the water can enemy spawns make sure Robo avoids it! If you collide with an enemy, you will lose points (as the value of enemy is set to a negative number). Until we implement a visible score, you may want to write the score out to the console.

# Summary

I'm sure you can now understand that most games would not be possible without collision detection. Collision detection allows objects in the game to interact with each other. We used collision detection to get pickups and detect whether we ran into an enemy.

We also discussed the essential task of boundary checking. Boundary checking is a special form of collision detection that checks to see whether an object has reached the screen boundaries. Another type of boundary checking is used to manage the scene background.

In the next chapter, we will wrap up the game by adding some finishing touches, including a heads-up display!