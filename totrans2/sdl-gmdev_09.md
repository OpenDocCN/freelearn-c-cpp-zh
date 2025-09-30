# Chapter 9. Creating Conan the Caveman

In the previous chapter, the creation of Alien Attack demonstrated that the framework is now at a point where it can be used to quickly create a 2D side scrolling shooter. Other genres are also simple to make with most of the changes once again being contained within the object classes.

In this chapter, we will cover:

*   Adapting the previous code base for a new game
*   More precise tile-collision detection
*   Handling jumping
*   Possible additions to the framework

This chapter will use the framework to create a platform game, Conan the Caveman. Here is a screenshot of the finished game level:

![Creating Conan the Caveman](img/6821OT_09_01.jpg)

Here's another screenshot with more enemies:

![Creating Conan the Caveman](img/6821OT_09_02.jpg)

As with the previous chapter, this chapter is not a step-by-step guide to creating Conan the Caveman, rather it is an overview of the most important aspects of the game. The project for the game is available in the source code downloads.

# Setting up the basic game objects

In some ways this game is more complicated than Alien Attack, whereas in other ways it is simpler. This section will cover the changes that were made to the Alien Attack source code: what was altered, what was removed, and what was added.

## No more bullets or bullet collisions

Conan the Caveman does not use projectile weapons, and therefore, there is no longer a `Bullet` class and the `CollisonManager` class no longer needs to have a function that checks for collisions between them; it only checks for the `Player` and `Enemy` collisions:

[PRE0]

## Game objects and map collisions

Almost all objects will need to collide with the tile map and react accordingly. The `GameObject` class now has a private member that is a pointer to the collision layers; previously only the `Player` class had this variable:

[PRE1]

`GameObject` also now has a function to set this variable:

[PRE2]

The `Player` class would previously have this set at the end of the `LevelParser::parseLevel` function, as follows:

[PRE3]

This is no longer needed, as each `GameObject` gets their `m_pCollisionLayers` variables set on creation in the object-layer parsing:

[PRE4]

## ShooterObject is now PlatformerObject

The shooter-specific code from Alien Attack has been stripped out of `ShooterObject` and the class is renamed to `PlatformerObject`. Anything that all game objects for this game will make use of is within this class:

[PRE5]

There are some variables and functions from Alien Attack that are still useful, plus a few new functions. One of the most important additions is the `checkCollideTile` function, which takes `Vector2D` as a parameter and checks whether it causes a collision:

[PRE6]

This is quite a large function, but it is essentially the same as how Alien Attack checked for tile collisions. One difference is the y position check:

[PRE7]

This is used to ensure that we can fall off the map (or fall into a hole) without the function trying to access tiles that are not there. For example, if the object's position is outside the map, the following code would try to access tiles that do not exist and would therefore fail:

[PRE8]

The y value check prevents this.

## The Camera class

In a game such as Alien Attack, precise map-collision detection is not terribly important; it is a lot more important to have precise bullet, player, and enemy collisions. A platform game, however, needs very precise map collision requiring the need for a slightly different way of moving the map, so that no precision is lost when scrolling.

In Alien Attack, the map did not actually move; some variables were used to determine which point of the map to draw and this gave the illusion of the map scrolling. In Conan the Caveman, the map will move so that any collision detection routines are relative to the actual position of the map. For this a `Camera` class was created:

[PRE9]

This class is very simple, as it merely holds a location and updates it using the position of a target, referred to the pointer as `m_pTarget`:

[PRE10]

This could also be updated to include the y value as well, but because this is a horizontal-scrolling game, it is not needed here and so the y is returned as `0`. This camera position is used to move the map and decide which tiles to draw.

## Camera-controlled map

The `TileLayer` class now needs to know the complete size of the map rather than just one section of it; this is passed in through the constructor:

[PRE11]

`LevelParser` passes the height and width in as it creates each `TileLayer`:

[PRE12]

The `TileLayer` class uses these values to set its row and column variables:

[PRE13]

With these changes, the tile map now moves according to the position of the camera and skips any tiles that are outside the viewable area:

[PRE14]

## The Player class

The `Player` class now has to contend with jumping as well as moving, all while checking for map collisions. The `Player::update` function has undergone quite a change:

[PRE15]

As movement is such an important part of this class, there is a function that is dedicated to handling it:

[PRE16]

### Tip

Notice that x and y checking has been split into two different parts; this is extremely important to make sure that an x collision doesn't stop y movement and vice versa.

The `m_lastSafePos` variable is used to put the player back into a safe spot after they are respawned. For example, if the player was to fall off the edge of the platform in the following screenshot and therefore land on the spikes below, he would be respawned at pretty much the same place as in the screenshot:

![The Player class](img/6821OT_09_03.jpg)

Finally, the handle input function now sets Boolean variables for moving to the right-hand side and left-hand side or jumping:

[PRE17]

This is all fairly self-explanatory apart from the jumping. When the player jumps, it sets the `m_bCanJump` variable to `false`, so that on the next loop the jump will not be called again, due to the fact that jump can only happen when the `m_bCanJump` variable is `true`; (landing after the jump sets this variable back to `true`).

# Possible additions

It wouldn't be hard to improve on Conan the Caveman's gameplay; increasing enemy and trap numbers would make the game significantly more exciting to play. The game could also benefit from some height to the levels so that players could really explore the map (Metroid style). Other gameplay improvements could include moving platforms, ladders, and bosses.

# Summary

Our reusable framework has proved its worth; two games have been created with minimal code duplication.

This chapter looked at scrolling a tile map using the position of the player along with collision detection. Tile-map collision was also covered, along with the important point of splitting x and y movement for effective movement in a platform game. Conan the Caveman is a great starting point for any other 2D game such as a scrolling beat-em-up or even a merging of this chapter and the last to create a platform shooter.

I hope that by now you have a good understanding of how to use SDL2.0 along with C++ to create games and how to effectively break game code apart to create a reusable framework. This is only the start and there are many more game-programming adventures ahead. Good luck!