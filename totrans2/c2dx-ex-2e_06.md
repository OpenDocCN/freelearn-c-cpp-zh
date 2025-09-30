# Chapter 6. Quick and Easy Sprite – Victorian Rush Hour

*In our fourth example of a game built with Cocos2d-x, I'll show you a simple technique for rapid prototyping. Often in game development, you want to test the core ideas of your game as soon as possible, because a game may sound fun in your head but in reality it just doesn't work. Rapid prototyping techniques allow you to test your game as early as possible in the development process as well as build up on the good ideas.*

Here's what you'll learn:

*   How to quickly create placeholder sprites
*   How to code collisions for a platform game
*   How to create varied terrain for a side-scroller

# The game – Victorian Rush Hour

In this game (Victorian Rush Hour), you control a cyclist in Victorian London trying to avoid rush-hour traffic on his way home. For reasons no one can explain, he's riding his bike on top of the buildings. As the player, it is your job to ensure he makes it.

The controls are very simple: you tap the screen to make the cyclist jump and while he's in the air, if you tap the screen again, the cyclist will open his trusty umbrella, either slowing his descent or adding a boost to his jump.

This game is of a type commonly known as a dash game or endless runner, a genre that has become increasingly popular online and on various app stores. Usually in these types of games you, the developer, have two choices: either make the terrain the main obstacle and challenge in the game, or make what's added to the terrain the main challenge (enemies, pick-ups, obstacles, and so on). With this game, I decided on the first option.

So our challenge is to create a game where the terrain is the enemy but not an unbeatable one.

# The game settings

The game is a universal application, designed for the iPad retina display but with support for other display sizes. It is played in the landscape mode and it does not support multitouch.

# Rapid prototyping with Cocos2d-x

The idea behind this is to create sprites as placeholders for your game elements as quickly as possible, so you can test your game ideas and refine them. Every game in this book was initially developed in the way I'm about to show you, with simple rectangles in place of textured sprites.

The technique shown here allows you to create rectangles of any size and of any color to be used in your game logic:

![Rapid prototyping with Cocos2d-x](img/00020.jpeg)

# Time for action – creating placeholder sprites

So let me show you how to do that:

1.  Go ahead and download the `4198_06_START_PROJECT.zip` file if you haven't done so already.
2.  When you open the project in Xcode, you will see all the classes we'll need for the game, and we'll go over them in a second. But for now, just go to `GameLayer.cpp`.
3.  Scroll down to the last `createGameScreen` method and add the following lines:

    [PRE0]

    And that's it. The sprite is created with a texture called `blank.png`. This is a 1 x 1 pixel white square you will find in the `Resources` folder. Then we set the size of the sprite's texture rectangle to 100 x 100 pixels (`setTextureRect`), and fill it with a white color (`setColor`). By resizing the texture rectangle, we in effect resize the sprite. If you run the game now, you should see a white square smack in the middle of the screen.

4.  Now delete the previous lines and replace them with these:

    [PRE1]

    This creates `_gameBatchNode` that uses as its source texture the same `blank.png` file. Now we are ready to place as many rectangles inside `_gameBatchNode` as we'd like, and set a different color for each one of them if we want. We can, in other words, build an entire test game with one tiny image. Which is what we'll proceed to do now.

5.  So, to finish up here, add these last lines:

    [PRE2]

## *What just happened?*

We just created a placeholder sprite we can use to test gameplay ideas quickly and painlessly. And we created our game's two main objects: the `Player` and `Terrain` object. These are empty shells at the moment, but we'll start working on them next. But first let's go over the different game elements.

# The Player object

This represents our cyclist. It will jump, float, and collide with the `_terrain` object. Its `x` speed is passed to the `_terrain` object causing the `Terrain` object to move, side scrolling to the left of the screen.

The `Player` object derives, once again, from a `GameSprite` class. This one has getters and setters for next position, vector of movement, and the sprite's width and height.

The `Player` interface has inline helper methods to retrieve information about its rectangle boundaries related to its current position (left, right, top, bottom), and its next position (`next_left`, `next_right`, `next_top`, `next_bottom`). These will be used in collision detection with the `_terrain` object.

# The Block object

These objects form the individual pieces of the `_terrain` object. They can take the shape of a building, or an empty gap between buildings. We'll have four different types of buildings, which later will represent four different types of textures when we finally bring in our sprite sheets. These blocks can have different widths and heights.

`Block` also derives from `GameSprite` and it also has inline helper methods to retrieve information about its boundaries, but only in relation to its current position, since `Block` doesn't technically move.

# The terrain object

This object contains the individual `Block` objects that form the landscape. It contains just enough `Block` objects to fill the screen, and as the `_terrain` object scrolls to the left, the `Block` objects that leave the screen are moved to the far right side of the `_terrain` and reused as new blocks, ensuring continuous scrolling.

The `_terrain` object is also responsible for collision checks with the `_player` object, since it has quick access to all information we'll need for collision detection; namely the list of blocks currently on the screen, their size, type, and position. Our main loop then will call on the `Terrain` object to test for collision with the `player` object.

Let's work on these main objects, starting with the `Player` object.

# Time for action – coding the player

Open up the `Player.cpp` class.

1.  The `_player` object is created through a static method that uses our `blank.png` file to texture the sprite. That method also makes a call to `initPlayer`, and this is what you should type for that method:

    [PRE3]

    The `_player` object will have its registration point at the top of the sprite. The reason behind this top center anchor point has much more to do with the way the `_player` object will be animated when floating, than with any collision logic requirements.

2.  Next comes `setFloating`:

    [PRE4]

    The `_hasFloated` property will ensure the player can only open the umbrella once while in the air. And when we set `_floating` to `true`, we give the `_player.y` vector a boost.

3.  We begin the update method of `_player` with:

    [PRE5]

    The game will increase `_maxSpeed` of the `_player` object as time goes on, making the game more difficult. These first lines make the change from the `_players` current `_speed` up to `_maxSpeed` a bit smoother and not an immediate change.

    ### Note

    Victorian Rush Hour has no levels, so it's important to figure out a way to make it incrementally harder to play, and yet not impossible. Finding that sweet spot in your logic may take some time and it's one more reason to test game ideas as soon as possible. Here we make the game harder by increasing the player's speed and the size of the gaps between buildings. These are updated inside a countdown in the main loop.

4.  Next, we update the `_player` object based on its `_state` of movement:

    [PRE6]

    We have different values for gravity and friction depending on move state.

    We also have a time limit for how long the `_player` object can be floating, and we reset that timer when the `_player` object is not floating. If the `_player` object is dying (collided with a wall), we move the `_player` object backward and downward until it leaves the screen.

5.  We finish with:

    [PRE7]

    When the player presses the screen for a jump, we shouldn't make the sprite jump immediately. Changes in state should always happen smoothly. So we have a `boolean` property in `_player` called `_jumping`. It is set to `true` when the player presses the screen and we slowly add the jump force to `_vector.y`. So the longer the player presses the screen, the higher the jump will be and a quick tap will result in a shorter jump. This is a nice feature to add to any platform game.

    We next limit the `y` speed with a terminal velocity, update the next position of the `_player` object, and update the floating timer if `_player` is floating.

## *What just happened?*

The `_player` object is updated through a series of states. Touching the screen will make changes to this `_state` property, as will the results of collision checking with `_terrain`.

Now let's work on the `Block` class.

# Time for action – coding the Block object

Once again a static method, `create`, will use `blank.png` to create our `Block` sprite. Only this time, we don't actually change the texture rectangle for `Block` inside `create`:

1.  The `Block` object is properly textured inside the `setupBlock` method:

    [PRE8]

    A `Block` object's appearance will be based on its type, width, and height.

    The `Block` sprite's registration point is set to top left. And we finally change the `Block` object's texture rectangle size here.

2.  Then we set the `Block` object's color based on type:

    [PRE9]

    `kBlockGap` means there is no building, just a gap the `_player` object must jump. We make the block invisible in that case and return from the function. So again, gaps are actually types of blocks in our logic.

In this test version, the different types of buildings are represented with different colors. Later we'll use different textures.

## *What just happened?*

The `Block` object is very simple. We just need its values for `_width` and `_height` whether it's a gap or not, so we can properly run collision detection with these objects.

## Planning the Terrain class

Before we jump to coding the `Terrain` class, we need to discuss a few things regarding randomness.

It is a very common mistake among game developers to confuse randomness with variableness, and very important to know when you need what.

A random number can be anything. 1234 is a random series of numbers. And the next time you want a random series of numbers and you once again get 1234 this will be just as random as the previous one. But not varied.

If you decide to build a random terrain, you will probably be disappointed in the result as it won't necessarily be varied. Also, remember that we need to make the terrain the key challenge of the game; but this means it can be neither too easy nor too difficult. True randomness would not allow us enough control here, or worse, we would end up with a long list of conditionals to make sure we have the correct combination of blocks, and that would result in at least one recurrent function inside our main loop, which is not a good idea.

We need instead to control the results and their variableness by applying our own patterns to them.

So we'll apply this logic of patterns to our `_terrain` object, forming a kind of pool of proper random choices. We'll use four arrays to store possible results in our decision making, and we'll shuffle three of these arrays during the game to add the "randomness" feel to our terrain.

These arrays are:

[PRE10]

This holds the information of how many buildings (`Blocks`) we have in a row, between gaps.

You can easily change the `patterns` value just by adding new values or by increasing or reducing the number of times one value appears. So here we're making a terrain with far more groupings of two buildings between gaps, than groups of three or one.

Next, consider the following lines:

[PRE11]

The preceding lines specify the widths and heights of each new building. These will be multiplied with the tile size determined for our game to get the final width and height values as you saw in `Block:setupBlock`.

We'll use a `0` value for height to mean there is no change in height from the previous building. A similar logic could be easily applied to widths.

And finally:

[PRE12]

These are building types and this array will not be shuffled unlike the three previous ones, so this is the `patterns` array of `types` we'll use throughout the game and it will loop continuously. You can make it as long as you wish.

## Building the terrain object

So every time we need to create a new block, we'll set it up based on the information contained in these arrays.

This gives us far more control over the terrain, so that we don't create impossible combinations of obstacles for the player: a common mistake in randomly-built terrain for dash games.

But at the same time, we can easily expand this logic to fit every possible need. For instance, we could apply level logic to our game by creating multiple versions of these arrays, so as the game gets harder, we begin sampling data from arrays that contain particularly hard combinations of values.

And we can still use a conditional loop to refine results even further and I'll give you at least one example of this.

The values you saw in the `patterns` arrays will be stored inside the lists called `_blockPattern`, `_blockWidths`, `_blockHeights`, and `_blockTypes`.

The `Terrain` class then takes care of building the game's terrain in three stages. First we initialize the `_terrain` object, creating among other things a pool for `Block` objects. Then we add the first blocks to the `_terrain` object until a minimum width is reached to ensure the whole screen is populated with `Blocks`. And finally we distribute the various block objects.

# Time for action – initializing our Terrain class

We'll go over these steps next:

1.  The first important method to implement is `initTerrain`:

    [PRE13]

    We have a timer to increase the width of gaps (we begin with gaps two tiles long).

    We create a pool for blocks so we don't instantiate any during the game. And `20` blocks is more than enough for what we need.

    The blocks we are currently using in the terrain will be stored inside a `_blocks` vector.

    We determine that the minimum width the `_terrain` object must have is `1.5` times the screen width. We'll keep adding blocks until the `_terrain` object reaches this minimum width. We end by shuffling the `patterns` arrays and adding the blocks.

2.  The `addBlocks` method should look like this:

    [PRE14]

    The logic inside the `while` loop will continue to add blocks until `currentWidth` of the `_terrain` object reaches `_minTerrainWidth`. Every new block we retrieve from the pool in order to reach `_minTerrainWidth` gets added to the `_blocks` vector.

3.  Blocks are distributed based on their widths:

    [PRE15]

## *What just happened?*

`Terrain` is a container of `Blocks`, and we just added the logic that will add a new `block` object to this container. Inside `addBlocks`, we call an `initBlock` method, which will use the information from our `patterns` arrays to initialize each block used in the terrain. It is this method we'll implement next.

# Time for action – initializing our Blocks object

Finally, we will discuss the method that initializes the blocks based on our `patterns` array:

1.  So inside the `Terrain` class, we start the `initBlock` method as follows:

    [PRE16]

    Begin by determining the type of building we are initializing. See how we loop through the `_blockTypes` array using the index stored in `_currentTypeIndex`. We'll use a similar logic for the other `patterns` arrays.

2.  Then, let's start building our blocks:

    [PRE17]

    The player must tap the screen to begin the game (`_startTerrain`). Until then, we show buildings with the same height (two tiles) and random width:

    ![Time for action – initializing our Blocks object](img/00021.jpeg)

    We will store `_lastBlockHeight` and `_lastBlockWidth` because the more information we have about the terrain the better we can apply our own conditions to it, as you will see in a moment.

3.  Consider that we are set to `_startTerrain`:

    [PRE18]

    In the following screenshot, you can see the different widths used for our blocks:

    ![Time for action – initializing our Blocks object](img/00022.jpeg)

    The information inside `_blockPattern` determines how many buildings we show in a row, and once a series is completed, we show a gap by setting the `boolean` value of `_showGap` to `true`. A gap's width is based on the current value of `_gapSize`, which may increase as the game gets harder and it can't be less than two times the tile width.

4.  If we are not creating a gap this time, we determine the width and height of the new block based on the current indexed values of `_blockWidths` and `_blockHeights`:

    [PRE19]

    Notice how we reshuffle the arrays once we are done iterating through them (`random_shuffle`).

    We use `_lastBlockHeight` to apply an extra condition to our terrain. We don't want the next block to be too tall in relation to the previous building, at least not in the beginning of the game, which we can determine by checking the value for `_gapSize`, which is only increased when the game gets harder.

    And if the value from `_blockHeights` is `0`, we don't change the height of the new building and use instead the same value from `_lastBlockHeight`.

5.  We finish by updating the count in the current series of buildings to determine whether we should show a gap next, or not:

    [PRE20]

## *What just happened?*

We finally got to use our `patterns` arrays and build the blocks inside the terrain. The possibilities are endless here in how much control we can have in building our blocks. But the key idea here is to make sure the game does not become ridiculously hard, and I advise you to play some more with the values to achieve even better results (don't take my choices for granted).

Before we tackle collision, let's add the logic to move and reset the terrain.

# Time for action – moving and resetting

We move the terrain inside the `move` method.

1.  The `move` method receives as a parameter the amount of movement in the `x` axis:

    [PRE21]

    The value for `xMove` comes from the `_player` speed.

    We start by updating the timer that will make the gaps wider. Then we move the terrain to the left. If after moving the terrain, a block leaves the screen, we move the block back to the end of the `_blocks` vector and reinitialize it as a new block through `initBlock`.

    We make a call to `addBlocks` just in case the reinitialized block made the total width of the terrain less than the minimum width required.

2.  Next, our `reset` method:

    [PRE22]

    The `reset` method is called whenever we restart the game. We move `_terrain` back to its starting point, and we reinitialize all the current `Block` objects currently inside the `_terrain` object. This is done because we are back to `_startTerrain = false`, which means all blocks should have the same height and a random width.

    If at the end of the reset we need more blocks to reach `_minTerrainWidth`, we add them accordingly.

## *What just happened?*

We can now move the `_terrain` object and all the blocks it contains, and we can restart the process all over again if we need to.

Once again, using the container behavior of nodes simplified our job tremendously. When you scroll the terrain, you scroll all the `Block` objects it contains.

So we are finally ready to run collision logic.

# Platform collision logic

We have in place all the information we need to check for collision through the inline methods found in `Player` and `Block`.

In this game, we'll need to check collision between the `_player` object's bottom side and the `block` object's top side, and between the `_player` object's right side and the `Block` class' left side. And we'll do that by checking the `_player` object's current position and its next position. We are looking for these conditions:

![Platform collision logic](img/00023.jpeg)

The diagram represents the conditions for bottom side collision, but the same idea applies to right side collision.

In the current position, the `_player` object must be above the top of the block or touching it. In the next position, the `_player` object must be either touching the top of the block or already overlapping it (or has moved past it altogether). This would mean a collision has occurred.

# Time for action – adding collision detection

Let's see how that translates to code:

1.  Still in `Terrain.cpp`:

    [PRE23]

    First we state that the `_player` object is currently falling with `inAir = true;` we'll let the collision check determine if this will remain true or not.

    We don't check the collision if `_player` is dying and we skip collision checks with any gap blocks.

    We check collision on the `y` axis, which here means the bottom of the `_player` and top of the block. We first need to determine if the `_player` object is within range of the block we want to check against collision. This means the center of the `_player` object must be between the left and right side of the block; otherwise, the block is too far from the `_player` object and may be ignored.

    Then we run a basic check to see if there is a collision between the `_player` object's current position and next position, using the conditions I explained earlier. If so, we fix the `_player` object's position and change its `y` vector speed to `0` and we determine that `inAir = false` after all, the `_player` object has landed.

2.  Next we check collision on the `x` axis, meaning the right side of the `_player` object with the left side of the blocks:

    [PRE24]

    Similar steps are used to determine if we have a viable block or not.

    If we do have a side collision, the `_player` state is changed to `kPlayerDying`, we reverse its `x` speed so the `_player` state will move to the left and off the screen, and we return from this method.

3.  We end by updating the `_player` object's state based on our collision results:

    [PRE25]

## *What just happened?*

We just added the collision logic to our platform game. As we did in our first game, Air Hockey, we test the player's current position for collision as well as its next position to determine if a collision occurred between the current iteration and the next one. The test simply looks for overlaps between the player's and block's boundaries.

# Adding the controls

It is fairly common in a dash game such as this to have very simple controls. Often the player must only press the screen for jumping. But we spiced things up a bit, adding a floating state.

And remember we want smooth transitions between states, so pay attention to how jumping is implemented: not by immediately applying a force to the player's vector but by simply changing a `boolean` property and letting the `_player` object's update method handle the change smoothly.

We'll handle the touch events next.

# Time for action – handling touches

Let's go back to `GameLayer.cpp` and add our game's final touches (pun intended).

1.  First we work on our `onTouchBegan` method:

    [PRE26]

    If we are not running the game and the `_player` object died, we reset the game on touch.

2.  Next, if the terrain has not started, insert the following:

    [PRE27]

    Remember that at first the buildings are all the same height and there are no gaps. Once the player presses the screen, we begin changing that through `setStartTerrain`.

3.  We finish with:

    [PRE28]

    Now we are in play, and if the `_player` object is falling, we either open or close the umbrella, whichever the case may be, through a call to `setFloating`.

    And if the `_player` object is not falling, nor dying, we make it jump with `setJumping(true)`.

4.  With touches ended, we just need to stop any jumps:

    [PRE29]

## *What just happened?*

We added the logic for the game's controls. The `_player` object will change to floating if currently falling or to jumping if currently riding on top of a building.

It's time to add our main game loop.

# Time for action – coding the main loop

Finally, it's time for the last part in our logic.

1.  Inside `GameLayer.cpp`:

    [PRE30]

    If the `_player` object is off screen, we stop the game.

2.  Now update all the elements, positions and check for collision:

    [PRE31]

3.  Move `_gameBatchNode` in relation to the `_player` object:

    [PRE32]

4.  Make the game more difficult as time goes on by increasing the `_player` object's maximum speed:

    [PRE33]

## *What just happened?*

We have our test game in place. From here, we can test our terrain patterns, our speeds, and our general gameplay to find spots where things could be improved.

We should check in particular whether the game gets too hard too fast or whether we have combinations of buildings that are just impossible to get past.

I find, for instance, that starting with larger groups of buildings, say four or five, and then slowly reducing them to two and one between gaps can make the game even more fun to play, so the patterns could be changed to reflect that idea.

# Summary

Every game has a simple idea for its gameplay at its core. But often, this idea needs a whole lot of testing and improvement before we can determine whether it's fun or not, which is why rapid prototyping is vital.

We can use Cocos2d-x to quickly test core gameplay ideas and run them in the simulator or on a device in a matter of minutes.

Also, the techniques shown here can be used to build interface elements (such as the energy bar from our previous game) as well as an entire game! If you don't believe me, check out the game *Square Ball* in an App Store near you.

Now, with all the logic for gameplay in its proper place, we can proceed to making this game look good! We'll do that in the next chapter.