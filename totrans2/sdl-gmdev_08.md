# Chapter 8. Creating Alien Attack

The framework has come on in leaps and bounds and we are almost ready to make our first game. We are going to create a simple 2D sidescrolling shooter in the vein of the classic '80's and '90's shooter games such as R-Type or Pulstar. However, the game will not be set in space. Aliens have attacked earth and only you and your weaponized helicopter can stop them. One level of fast-paced action is available in the source code downloads and this chapter will cover the steps taken to create it. Here is a screenshot of the game we will be creating:

![Creating Alien Attack](img/6821OT_08_01.jpg)

And another slightly more hectic shot:

![Creating Alien Attack](img/6821OT_08_02.jpg)

There are still a few things that the framework must handle before we can create this game. These additions include:

*   Sound
*   Collision detection

By the end of the chapter you will have a good understanding of how this game was built using the framework and you will have the ability to continue and improve it. In this chapter, we will cover:

*   Implementing sound
*   Creating game-specific object classes
*   Shooting and detecting bullets
*   Creating different enemy types
*   Developing a game

# Using the SDL_mixer extension for sound

The SDL_mixer extension has its own Mercurial repository that can be used to grab the latest source for the extension. It is located at [http://hg.libsdl.org/SDL_mixer](http://hg.libsdl.org/SDL_mixer). The TortoiseHg application can again be used to clone the extension's Mercurial repository. Follow these steps to build the library:

1.  Open up TortoiseHg and press *CTRL*+*SHIFT*+*N* to start cloning a new repository.
2.  Type [http://hg.libsdl.org/SDL_mixer](http://hg.libsdl.org/SDL_mixer) into the source box.
3.  The **Destination** will be `C:\SDL2_mixer`.
4.  Hit **Clone** and wait for completion.
5.  Navigate to `C:\SDL2_mixer\VisualC\` and open `SDL_mixer.vcproj` in Visual Studio 2010.
6.  As long as the x64 folder outlined in [Chapter 2](ch02.html "Chapter 2. Drawing in SDL"), *Drawing in SDL* was created, the project will convert with no issues.
7.  We are going to build the library without MP3 support as we are not going to need it, and also it does not work particularly well with SDL 2.0.
8.  Add `MP3_MUSIC_DISABLED` to the **Preprocessor Definitions** in the project properties, which can be found by navigating to **C/C++** | **Preprocessor**, and build as per the `SDL_image` instructions in [Chapter 2](ch02.html "Chapter 2. Drawing in SDL"), *Drawing in SDL*.

## Creating the SoundManager class

The game created in this chapter will not need any advanced sound manipulation, meaning the `SoundManager` class is quite basic. The class has only been tested using the `.ogg` files for music and the `.wav` files for sound effects. Here is the header file:

[PRE0]

The `SoundManager` class is a singleton; this makes sense because there should only be one place that the sounds are stored and it should be accessible from anywhere in the game. Before sound can be used, `Mix_OpenAudio` must be called to set up the audio for the game. `Mix_OpenAudio` takes the following parameters:

[PRE1]

This is done in the `SoundManager`'s constructor with values that will work well for most games.

[PRE2]

The `SoundManager` class stores sounds in two different `std::map` containers:

[PRE3]

These maps store pointers to one of two different types used by `SDL_mixer` (`Mix_Chunk*` and `Mix_Music*`), keyed using strings. The `Mix_Chunk*` types are used for sound effects and the `Mix_Music*` types are of course used for music. When loading a music file or a sound effect into `SoundManager`, we pass in the type of sound we are loading as an `enum` called `sound_type`.

[PRE4]

This type is then used to decide which `std::map` to add the loaded sound to and also which `load` function to use from `SDL_mixer`. The `load` function is defined in `SoundManager.cpp`.

[PRE5]

Once a sound has been loaded it can be played using the **playSound** or **playMusic** functions:

[PRE6]

Both of these functions take the ID of the sound to be played and the amount of times that it is to be looped. Both functions are very similar.

[PRE7]

One difference between `Mix_PlayMusic` and `Mix_PlayChannel` is that the latter takes an `int` as the first parameter; this is the channel that the sound is to be played on. A value of **-1** (as seen in the preceding code) tells `SDL_mixer` to play the sound on any available channel.

Finally, when the `SoundManager` class is destroyed, it will call `Mix_CloseAudio`:

[PRE8]

And that's it for the `SoundManager` class.

# Setting up the basic game objects

The majority of the work that went into creating Alien Attack was done in the object classes, while almost everything else was already being handled by manager classes in the framework. Here are the most important changes:

## GameObject revamped

The `GameObject` base class has a lot more to it than it previously did.

[PRE9]

This class now has a lot of the member variables that used to be in `SDLGameObject`. New variables for checking whether an object is updating, doing the death animation, or is dead, have been added. Updating is set to true when an object is within the game screen after scrolling with the game level.

In place of a regular pointer to `LoaderParams` in the load function, an `std::unique_ptr` pointer is now used; this is part of the new **C++11 standard** and ensures that the pointer is deleted after going out of scope.

[PRE10]

There are two new functions that each derived object must now implement (whether it's owned or inherited):

[PRE11]

## SDLGameObject is now ShooterObject

The `SDLGameObject` class has now been renamed to `ShooterObject` and is a lot more specific to this type of game:

[PRE12]

This class has default implementations for draw and update that can be used in derived classes; they are essentially the same as the previous `SDLGameObject` class, so we will not cover them here. A new function that has been added is `doDyingAnimation`. This function is responsible for updating the animation when enemies explode and then setting them to dead so that they can be removed from the game.

[PRE13]

## Player inherits from ShooterObject

The **Player** object now inherits from the new `ShooterObject` class and implements its own update function. Some new game-specific functions and variables have been added:

[PRE14]

The `ressurect` function resets the player back to the center of the screen and temporarily makes the `Player` object invulnerable; this is visualized using `alpha` of the texture. This function is also responsible for resetting the size value of the texture which is changed in `doDyingAnimation` to accommodate for the explosion texture:

[PRE15]

Animation is a big part of the feel of the `Player` object; from flashing (when invulnerable), to rotating (when moving in a forward or backward direction). This has led to there being a separate function dedicated to handling animation:

[PRE16]

The angle and `alpha` of an object are changed using new parameters to the `drawFrame` function of `TextureManager`:

[PRE17]

Finally the `Player::update` function ties this all together while also having extra logic to handle when a level is complete:

[PRE18]

Once a level is complete and the player has flown offscreen, the `Player::update` function also tells the game to increment the current level:

[PRE19]

The `Game::setCurrentLevel` function changes the state to `BetweenLevelState`:

[PRE20]

## Lots of enemy types

A game such as Alien Attack needs a lot of enemy types to keep things interesting; each with its own behavior. Enemies should be easy to create and automatically added to the collision detection list. With this in mind, the `Enemy` class has now become a base class:

[PRE21]

All enemy types will derive from this class, but it is important that they do not override the `type` method. The reason for this will become clear once we move onto our games collision detection classes. Go ahead and take a look at the enemy types in the Alien Attack source code to see how simple they are to create.

![Lots of enemy types](img/6821OT_08_10.jpg)

## Adding a scrolling background

Scrolling backgrounds are important to 2D games like this; they help give an illusion of depth and movement. This `ScrollingBackground` class uses two destination rectangles and two source rectangles; one expands while the other contracts. Once the expanding rectangle has reached its full width, both rectangles are reset and the loop continues:

[PRE22]

# Handling bullets

Most objects in the game fire bullets and they all pretty much need to be checked for collisions against bullets as well; the bottom line—bullets are important in Alien Attack. The game has a dedicated `BulletHandler` class that handles the creation, destruction, updating, and rendering of bullets.

## Two types of bullets

There are two types of bullets in the game, `PlayerBullet` and `EnemyBullet`, both of which are handled in the same `BulletManager` class. Both of the bullet classes are declared and defined in `Bullet.h`:

[PRE23]

Bullets are very simple, they just move in one direction and at a certain speed.

## The BulletHandler class

The `BulletHandler` class uses two public functions to add bullets:

[PRE24]

The `BulletHandler` class is also a singleton. So, if an object wants to add a bullet to the game, it can do so using one of the above functions. Here is an example from the `ShotGlider` class:

[PRE25]

This will add a bullet at the current location of `ShotGlider`, with a heading vector of *V*(-10,0).

Both `add` functions are very similar; they create a new instance of `PlayerBullet` or `EnemyBullet` and then push it into the correct vector. Here are their definitions:

[PRE26]

A big advantage of having a separate place to store bullets like this, rather than have objects themselves manage their own bullets, is that there is no need to pass objects around just to get their bullets to check collisions against. This `BulletHandler` class gives us a centralized location that we can then easily pass to the collision handler.

The `update` and `draw` functions are essentially just loops that call each bullet's respective functions, however the `update` function will also destroy any bullets that have gone off the screen:

[PRE27]

# Dealing with collisions

With so many bullets flying around and having the `Enemy` objects to check collisions against, it is important that there be a separate class that does this collision checking for us. This way we know where to look if we decide we want to implement a new way of checking for collisions or optimize the current code. The `Collision.h` file contains a static method that checks for collisions between two `SDL_Rect` objects:

[PRE28]

The function makes use of a buffer, which is a value that is used to make the rectangles slightly smaller. In a game such as Alien Attack, exact collision on bounding rectangles would be slightly unfair and also not much fun. With the buffer value, more direct hits are needed before they will be registered as a collision. Here the buffer is set to `4`; this will take a fourth off of each side of the rectangle.

The `Player` class will not handle its own collisions. This requires a way to separate out the player from the rest of the `GameObject` instants when the level is loaded. The `Level` class now stores a pointer to `Player`:

[PRE29]

With a public getter and setter:

[PRE30]

The `LevelParser` instance sets this pointer when it loads in `Player` from the level file:

[PRE31]

Another addition to `Level` is that it holds a separate `std::vector` of `TileLayer*` which are tile layers that the game will check against for collisions. This value is passed in from the `.tmx` file and any `TileLayer` that needs to be checked for collisions must set `collidable` as a property in the tiled application.

![Dealing with collisions](img/6821OT_08_07.jpg)

This also requires a slight alteration in `LevelParser::parseLevel` when checking for object layers, just in case the layer does contain properties (in which case data would no longer be the first child element):

[PRE32]

The `LevelParser` instance can now add collision layers to the collision layers array in `parseTileLayer`:

[PRE33]

## Creating a CollisionManager class

The class responsible for checking and handling all of these collisions is the `CollisionManager` class. Here is its declaration:

[PRE34]

Looking at the source code you will see that these functions are pretty big, yet they are relatively simple. They loop through each object that requires a collision test, create a rectangle for each, and then pass it to the static `RectRect` function defined in `Collision.h`. If a collision occurred then it calls the `collision` function for that object. The `checkEnemyPlayerBulletCollision` and `checkPlayerEnemyCollision` functions perform an extra check to see if the object is actually of `Enemy` type:

[PRE35]

If it is not, then it does not check the collision. This is why it is important that the `Enemy` subtypes do not override the `type` function or if they do, their type must also be added to this check. This condition also checks whether the object is updating or not; if it is not, then it is offscreen and does not need to be checked against for collision.

Checking for collision against tiles requires a similar method to working out where to start drawing the tiles from, which was implemented in the `TileLayer::render` function. Here is the `checkPlayerTileCollision` definition:

[PRE36]

# Possible improvements

Alien Attack is a pretty robust game at the moment; we highly recommend looking through the source code and becoming familiar with every aspect of it. Once you have a good understanding of most of the areas of the game, it is a lot easier to see where certain areas could be enhanced. Here are some ideas that could be added to improve the game:

*   Bullets could be created at the start of a level and stored in an object pool; so rather than creating and deleting bullets all the time they can be pulled from and put back into the object pool. The main advantage of this approach is that the creation and destruction of objects can be quite expensive when it comes to performance. Eliminating this while the game is running could give a real performance boost.
*   Collision detection could be optimized further, possibly through the addition of a **Quadtree** to stop unnecessary collision checks.
*   The source code has a few areas that use string comparisons to check types. This can be a bit of a performance hog, so other options such as using `enums` as types may be a better option.

You may have noticed areas yourself that you feel you could improve upon. Working on these within the context of a game, where you can test the results, is a great learning experience.

# Summary

The framework has been successfully used to create a game—Alien Attack. Throughout this chapter, the most important parts of the game were covered, along with a short explanation of why they were designed in such a way. With the source code for this game available, there is now a great project to start practicing with.