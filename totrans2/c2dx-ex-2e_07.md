# Chapter 7. Adding the Looks – Victorian Rush Hour

*Now that we have our test game, it's time to make it all pretty! We'll go over the new sprite elements added to make the game look nice, and cover a new topic or two. However, by now, you should be able to understand everything in the final code of this project.*

*So you can sit back and relax a bit. This time, I won't make you type so much. Promise!*

In this chapter, you will learn:

*   How to use multiple sprites to texture a tiled terrain
*   How to use multiple containers inside `SpriteBatchNode`
*   How to create a parallax effect
*   How to add a menu to your game
*   How to build a game tutorial

# Victorian Rush Hour – the game

Download the `4198_07_START_PROJECT.zip` file from this book's **Support** page ([www.packtpub.com/support](http://www.packtpub.com/support)) and run the project in Xcode. You should be able to recognize all the work we did in the test version, and pinpoint the few extra elements. You will also see that nothing was added to the actual gameplay.

In *Victorian Rush Hour*, I wanted to make the terrain the main challenge in the game, but I also wanted to show you how easily you can add new elements to the buildings and interact with them.

You can later use the same logic to add enemies, obstacles, or pickups for the cyclist sprite. All you need to do really is extend the collision detection logic to check for the new items. You could, for instance, add umbrellas as pickups, and every time the `_player` object floated, he would be minus one umbrella.

Next, I'll list the new elements added to the game.

![Victorian Rush Hour – the game](img/00024.jpeg)

## New sprites

Quite a few sprites were added to our game:

*   There is a group of cyclists at the beginning of the game representing the traffic.
*   We add a background layer (`cityscape`) and a foreground layer (`lamp posts`) to help us with our parallax effect. The clouds in the background are also part of the effect.
*   We add chimneys to the buildings. These puff smoke as the player taps the screen.
*   And, of course, the usual stuff—score label, game logo, and a game over message.

    In the following screenshot, you can see an image of the `player` sprite and the group of cyclists:

    ![New sprites](img/00025.jpeg)

## Animations

Some of the sprites now run animation actions:

*   The `_player` sprite runs an animation showing him riding the bicycle (`_rideAnimation`).
*   I also added our old friend, the swinging animation, shown when the `_player` sprite is floating (`_floatAnimation`). This is the reason for the odd registration point on the cyclist sprite, as the swing animation looks better if the sprite's anchor point is not centered.
*   Our group of cyclists is also animated during the introduction section of the game, and is moved offscreen when the game starts (`_jamAnimate`, `_jamMove`).
*   We show a puff of smoke coming out of the chimneys whenever the player jumps. This animation is stored inside the new `Block.cpp` class and it's created through a series of actions, including a frame animation (`_puffAnimation`, `_puffSpawn`, `_puffMove`, `_puffFade`, and `_puffScale`).
*   In `GameLayer.cpp`, when the `_player` object dies, we run a few actions on a `_hat` sprite to make it rise in the air and drop down again, just to add some humor.

Now let's go over the added logic.

# Texturing our buildings with sprites

So in the test version we just coded, our game screen was divided into tiles of 128 pixels in the iPad retina screen. The width and height properties of the `Block` objects are based on this measurement. So a building two tiles wide and three tiles tall would have, in effect, 256 pixels in width and 384 pixels in height. A gap too would be measured this way, though its height is set to `0`.

The logic we use to texture the buildings will take these tiles into account.

![Texturing our buildings with sprites](img/00026.jpeg)

So let's take a look at the code to add texture to our buildings.

# Time for action – texturing the buildings

There are a few changes to the way the `initBlock` method runs now:

1.  Each block will store references to four different types of texture, representing the four types of buildings used in the game (`_tile1`, `_tile2`, `_tile3`, and `_tile4`). So we now store that information in the `initBlock` method:

    [PRE0]

2.  Each block also stores references to two types of textures for the building roof tile (`_roof1` and `_roof2`):

    [PRE1]

3.  Next, we create and distribute the various sprite tiles that form our building:

    [PRE2]

    A block comprises 20 sprites stored inside a `_wallTiles` vector and five sprites stored in a `_roofTiles` vector. So, when we initialize a `Block` object, we in effect create a building that is five tiles wide and four tiles tall. I made the decision that no building in the game would exceed this size. If you decide to change this, then here is where you would need to do it.

4.  The `initBlock` method also creates five chimney sprites and places them at the top of the building. These will be spread out later according to the building type and could be very easily turned into obstacles for our `_player` sprite. We also create the animation actions for the puffs of smoke, here inside `initBlock`.
5.  Moving on to our new `setupBlock` method, this is where the unnecessary tiles and chimneys are turned invisible and where we spread out the visible chimneys. We begin the method as follows:

    [PRE3]

6.  Then, based on building type, we give different `x` positions for the chimney sprites and determine the texture we'll use on the wall tiles:

    [PRE4]

7.  The method then proceeds to position the visible chimneys. And we finally move to texturing the building. The logic to texture the roof and wall tiles is the same; for instance, here's how the walls are tiled by changing the texture of each wall sprite through the `setDisplayFrame` method and then turning unused tiles invisible:

    [PRE5]

## *What just happened?*

When we instantiate a block in `initBlock`, we create a 5 x 4 building made out of wall tiles and roof tiles, each a sprite. And when we need to turn this building into a 3 x 2 building, or a 4 x 4 building, or whatever, we simply turn the excess tiles invisible at the end of `setupBlock`.

The texture used for the roof is picked randomly, but the one picked for the walls is based on building type (from our `patterns` array). It is also inside this `for` loop that all the tiles positioned at a point greater than the new building's width and height are turned invisible.

# Containers within containers

Before we move to the parallax effect logic, there is something I wanted to talk about related to the layering of our `_gameBatchNode` object, which you'll recall is a `SpriteBatchNode` object.

If you go to the static `create` method inside `Terrain.cpp`, you will notice that the object is still created with a reference to a `blank.png` texture:

[PRE6]

In fact, the same 1 x 1 pixel image used in the test version is now in our sprite sheet, only this time the image is transparent.

This is a bit of a hack, but necessary, because a sprite can only be placed inside a batch node if its texture source is the same used to create the batch node. But `Terrain` is just a container, it has no texture. However, by setting its `blank` texture to something contained in our sprite sheet, we can place `_terrain` inside `_gameBatchNode`.

The same thing is done with the `Block` class, which now, in the final version of the game, behaves like another textureless container. It will contain the various sprites for the wall and roof tiles as well as chimneys and puff animations as its children.

The organization of the layers inside our `_gameBatchNode` object can seem complex and at times even absurd. After all, in the same node, we have a foreground "layer" of lampposts, a middle-ground "layer" of buildings, and a background "layer" containing a cityscape. The player is also placed in the background but on top of the cityscape. Not only that, but all three layers are moved at different speeds to create our parallax effect, and all this inside the same `SpriteBatchNode`!

But the amount of code this arrangement saves us justifies any confusion we might have at times when attempting to keep the batch node organized. Now we can animate the puffs of smoke, for instance, and never worry about keeping them "attached" to their respective `chimney` sprite as the terrain scrolls to the left. The container will take care of keeping things together.

# Creating a parallax effect

Cocos2d-x has a special node called `ParallaxNode`, and one surprising thing about it is how little you get to use it! `ParallaxNode` helps create a parallax effect with finite layers, or finite scrolling, which means that you can use it if your game screen has a limit to how much it can scroll each way. Implementing `ParallaxNode` to a game screen that can scroll indefinitely, such as the one in *Victorian Rush Hour*, usually requires more effort than it takes to build your own effect.

A parallax effect is created by moving objects at different depths at different speeds. The farther a layer appears from the screen, the slower its speed should be. In a game, this usually means that the player sprite's speed is fractioned to all the layers that appear behind it, and multiplied for the layers that appear in front of the player sprite:

![Creating a parallax effect](img/00027.jpeg)

Let's add this to our game.

# Time for action – creating a parallax effect

The parallax effect in our game takes place inside the main loop:

1.  So in our `update` method, you will find the following lines of code:

    [PRE7]

    First, we move the `_background` sprite, which contains the cityscape texture repeated three times along the `x` axis, and we move it at one-fourth of the speed of the `_player` sprite.

2.  The `_background` sprite scrolls to the left, and as soon as the first cityscape texture is off the screen, we shift the entire `_background` container to the right at precisely the spot where the second cityscape texture would appear if allowed to continue. We get this value by subtracting where the sprite would be from the total width of the sprite:

    [PRE8]

    So, in effect, we only ever scroll the first texture sprite inside the container.

3.  A similar process is repeated with the `_foreground` sprite and the three lamppost sprites it contains. Only the `_foreground` sprite moves at four times the speed of the `_player` sprite. These are coded as follows:

    [PRE9]

4.  We also employ our `cloud` sprites in the parallax effect. Since they appear behind the cityscape, so even farther away from `_player`, the clouds move at an even lower rate (`0.15`):

    [PRE10]

## *What just happened?*

We just added the parallax effect in our game by simply using the player speed at different ratios at different depths. The only slightly complicated part of the logic is how to ensure the sprites scroll continuously. But the math of it is very simple. You just need to make sure the sprites align correctly.

# Adding a menu to our game

Right now, we only see the game logo on our introduction screen. We need to add buttons to start the game and also for the option to play a tutorial.

In order to do that, we'll use a special kind of `Layer` class, called `Menu`.

`Menu` is a collection of `MenuItems`. The layer is responsible for distributing its items as well as tracking touch events on all items. Items can be sprites, labels, images, and so on.

![Adding a menu to our game](img/00028.jpeg)

# Time for action – creating Menu and MenuItem

In `GameLayer.cpp`, scroll down to the `createGameScreen` method. We'll add the new logic to the end of this method.

1.  First, create the menu item for our start game button:

    [PRE11]

    We create a `MenuItemSprite` object by passing it one sprite per state of the button. When the user touches a `MenuItemSprite` object, the off state sprite is turned invisible and the on state sprite is turned visible, all inside the touch began event. If the touch is ended or cancelled, the off state is displayed once again.

    We also pass the callback function for this item; in this case, `GameLayer::StartGame`.

2.  Next, we add the tutorial button:

    [PRE12]

3.  Then it's time to create the menu:

    [PRE13]

    The `Menu` constructor can receive as many `MenuItemSprite` objects as you wish to display. These items are then distributed with one of the following calls: `alignItemsHorizontally`, `alignItemsHorizontallyWithPadding`, `alignItemsHorizontally`, `alignItemsVerticallyWithPadding`, `alignItemsInColumns`, and `alignItemsInRows`. And the items appear in the order they are passed to the `Menu` constructor.

4.  Then we need to add our callback functions:

    [PRE14]

    These are called when our menu buttons are clicked on, one method to start the game and one to show the tutorial.

## *What just happened?*

We just created our game's main menu. `Menu` can save us a lot of time handling all the interactivity logic of buttons. Though it might not be as flexible as other items in Cocos2d-x, it's still good to know it's there if we need it.

We'll tackle the tutorial section next.

# Adding a tutorial to our game

Let's face it. With the possible exception of *Air Hockey*, every game so far in this book could benefit from a tutorial, or a "how to play" section. With *Victorian Rush Hour*, I'm going to show you a quick way to implement one.

The unspoken rule of game tutorials is—make it playable. And that's what we'll attempt to do here.

We'll create a game state for our tutorial, and we'll add a `Label` object to our stage and make it invisible unless the tutorial state is on. We'll use the `Label` object to display our tutorial text, as shown in the image here:

![Adding a tutorial to our game](img/00029.jpeg)

Let's go over the steps necessary to create our game tutorial.

# Time for action – adding a tutorial

Let's move back to our `createGameScreen` method.

1.  Inside that method, add the following lines to create our `Label` object:

    [PRE15]

2.  We add four states to our enumerated list of game states. These will represent the different steps in our tutorial:

    [PRE16]

    The first tutorial state, `kGameTutorial`, acts as a separator from all other game states. So, if the value for `_state` is greater than `kGameTutorial`, we are in tutorial mode.

    Now, depending on the mode, we display a different message and we wait on a different condition to change to a new tutorial state.

3.  If you recall, our `showTutorial` method starts with a message telling the player to tap the screen to make the sprite jump:

    [PRE17]

4.  Then, at the end of the `update` method, we start adding the lines that will display the rest of our tutorial information. First, if the player sprite is in the midst of a jump and has just begun falling, we use the following:

    [PRE18]

    As you can see, we let the player know that another tap will open the umbrella and cause the sprite to float.

5.  Next, as the sprite is floating, when it reaches a certain distance from the buildings, we inform the player that another tap will close the umbrella and cause the sprite to drop. Here's the code for these instructions:

    [PRE19]

6.  After that, the tutorial will be complete and we show the message that the player may start the game:

    [PRE20]

    Whenever we change a tutorial state, we pause the game momentarily and wait for a tap. We handle the rest of our logic inside `onTouchBegan`, so we'll add that next.

7.  Inside `onTouchBegan`, in the `switch` statement, add the following cases:

    [PRE21]

## *What just happened?*

We added a tutorial to our game! As you can see, we used quite a few new states. But now we can incorporate the tutorial right into our game and have one flow smoothly into the other. All these changes can be seen in action in the final version of this project, `4198_07_FINAL_PROJECT.zip`, which you can find on this book's **Support** page.

Now, you guessed it, let's run it in Android.

# Time for action – running the game in Android

Follow these steps to deploy the game to Android:

1.  Open your project's `Android.mk` file in a text editor.
2.  Edit the lines in `LOCAL_SRC_FILES` to read:

    [PRE22]

3.  Import the game into Eclipse and wait until all classes are compiled.
4.  That's it. Save it and run your application.

## *What just happened?*

You now have *Victorian Rush Hour* running in Android.

# Summary

After we got all the gameplay details ironed out in our test game, bringing in a sprite sheet and game states seems remarkably simple and easy.

But during this stage, we can also think of new ways to improve gameplay. For instance, the realization that clouds of smoke coming out of chimneys would offer a nice visual cue to the player to identify where the buildings were, if the cyclist happened to jump too high. Or that a hat flying through the air could be funny!

Now it's time to bring physics to our games, so head on to the next chapter.