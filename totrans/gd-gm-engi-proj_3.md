# Escape the Maze

In the previous chapter, you learned how Godot's node system works, allowing you to build a complex scene out of smaller building blocks, each providing different functionalities for your game's objects. This process will continue as you move up to larger and more complex projects. However, sometimes you'll find yourself duplicating the same nodes and/or code in more than one different object, and this project will introduce some techniques for reducing the amount of repeated code. 

In this chapter, you'll build a game called **Escape the Maze**. In this game, you will be trying to navigate a maze to find the exit while avoiding the roaming enemies:

![](img/00042.jpeg)

You will learn about the following key topics in this project:

*   Inheritance
*   Grid-based movement
*   Spritesheet animation
*   Using TileMaps for level design
*   Transitioning between scenes

# Project setup

Create a new project and download the project assets from [https://github.com/PacktPublishing/Godot-Game-Engine-Projects/releases](https://github.com/PacktPublishing/Godot-Game-Engine-Projects/releases).

As you've seen previously, Godot, by default, includes a number of input actions mapped to various keyboard inputs. For example, you used `ui_left` and `ui_right` for arrow key movement in the first project. Often, however, you need a different input from the defaults provided, or you'd like to customize the actions' names. You might also wish to add actions for mouse or gamepad inputs. You can do this in the Project Settings window.

Click on the Input Map tab and add four new input actions (left, right, up, and down) by typing the names into the Action: box and clicking Add. Then, for each new action, click the + button to add a Key action and choose the corresponding arrow key. You can also add WASD controls, if you wish:

![](img/00043.jpeg)

This game will have a variety of objects on the screen. Some of them should detect collisions (the player against the walls, for example), while others should ignore one another (like the enemies versus coins). You can solve this by setting the objects' physics layer and physics layer mask properties. To make these layers easier to work with, Godot allows you to give the game's physics layers custom names.

Click on the General tab and find the Layer Names/2D Physics section. Name the first four layers as follows:

![](img/00044.jpeg)

You'll see how the collision layer system works with the various objects in the game later in the project.

Next, in the Display/Window section, set the Mode to viewport and the Aspect to keep. This will enable you to resize the game window while keeping the display's proportions unchanged. Refer to the following screenshot:

![](img/00045.jpeg)

Finally, in the Rendering/Quality section, set Use Pixel Snap to On. This setting is useful, especially for pixel art-styled games, as it ensures that all objects are drawn at whole-number pixel values. Note that this does not affect movement, physics, or other properties; it only applies to the rendering of objects. Refer to the following screenshot:

![](img/00046.jpeg)

# Project organization

As your projects become larger and more involved, you'll find that saving all of your scenes and scripts in the same folder becomes unwieldy. 

A common response to this by Godot beginners is to make a `scenes` folder and a `scripts` folder, and to save each type of file in the respective folder. This isn't very effective. Soon, you find yourself hunting through the `scripts` folder, looking for the script you need because it's jumbled up with all the other scripts of your game.

A more logical organization is to create a folder for each type of object. A `player` folder, for example, will hold the player's scene file, script(s), and any other resources that it needs. Organizing your project in this way is much more scalable and can be extended even further if you have a very large number of objects. For example, refer to the following screenshot:

![](img/00047.jpeg)

Throughout this project, the examples will assume that each new scene type is being saved in a folder of that type, along with its script. The `Player.tscn` and `Player.gd` files, for example, will be saved in a `player` folder.

# Inheritance

In **Object-Oriented Programming** (**OOP**), inheritance is a powerful tool. Put briefly, you can define a class that *inherits* from another class. An object created using the first class will contain all of the methods and member variables of the master class as well as its own.

Godot is strongly object-oriented, and this gives you the opportunity to use inheritance not just with objects (scripts) but also with scenes, allowing you a great deal of flexibility when designing your game's architecture. It also removes the need to duplicate code—if two objects need to share a set of methods and variables, for example, you can create a common script and let both objects inherit from it. If you make a change to that code, it will apply to both objects.

In this project, the player's character will be controlled by key events, while the mobs will wander around the maze randomly. However, both types of character need to have a number of properties and functions in common:

*   A spritesheet containing the four directional movement animations
*   An `AnimationPlayer` to play the movement animations
*   Grid-based movement (the character can only move one full *tile* at a time)
*   Collision detection (the character can't move through walls)

By using inheritance, you can create a generic `Character` scene containing the nodes that all characters need. The player and mob scenes can inherit the shared nodes from that scene. Similarly, the actual movement code (though not the controls) will be identical between player and mob, so they can both inherit from the same script to handle movement.

# Character scene

Start creating the `Character` scene by adding an `Area2D` and naming it `Character`. `Area2D` is a good choice for this type of character because its main function will be to detect overlaps—when it moves onto an item or enemy, for example.

Add the following children:

*   `Sprite`
*   `CollisionShape2D`
*   `Tween` (named `MoveTween`)
*   `AnimationPlayer`

Leave the `Sprite` without a texture, but in the Inspector, under the Animation section of the `Sprite`, set its Vframes and Hframes properties to `4` and `5`, respectively. This tells Godot to slice the texture into a 5 x 4 grid of individual images.

The spritesheets you'll use for the player and the enemy are arranged in exactly this pattern, with each row containing the animation frames for a single direction of movement:

![](img/00048.jpeg)

When a spritesheet has been sliced using the Vframes and Hframes properties, you can use the Frame property to set which individual frame to use. In the preceding player sheet, the left-facing animation would use frames 5 through 9 (counting from frame 0 in the upper-left corner). You'll use an `AnimationPlayer` to change the Frame property below. Refer to the following screenshot:

![](img/00049.jpeg)

Next, create a new `RectangleShape2D` in the collision shape's Shape. Click on the new <RectangleShape2D> and set its Extents property in the Inspector to `(16, 16)`. Note that Extents measures the distance from the center in each direction, so this results in a collision shape that is 32 by 32 pixels.

Because all the characters are drawn to the same scale, we can be confident that the same sized collision shape will work for all characters. If this isn’t the case with the art you’re using, you can skip setting the collision shape here and configure it later for the individual inherited scenes.

# Animations

Create four new animations in the `AnimationPlayer` node. Name them to match the four directions you used in the input actions (left, right, up, and down). It's important that the spelling matches here: the names of the input actions must have the same spelling and capitalization as the animation names. If you are inconsistent in naming, it will make things much more difficult when you get to the scripting stage. Take a look at the following screenshot:

![](img/00050.jpeg)

For each animation, set the Length to `1` and the Step to `0.2`. These properties are located at the bottom of the Animation panel:

![](img/00051.jpeg)

Starting with the down animation, click on the `Sprite` node and set its Frame property to `0`. Click the key icon next to the Frame property and confirm that you want to add a new track for the Frame property:

![](img/00052.jpeg)

The Frame property will automatically be incremented by one and the animation track will be advanced by one step (0.2 seconds). Click the key again until you've reached frame 4\. You should now have five keyframes on the animation track. If you drag the bar back and forth, you'll see the Frame property change as you reach each keyframe:

![](img/00053.jpeg)

If, for some reason, you find that the frames aren't correct, you can delete any of the keyframes by clicking on the dot and pressing *Delete* on your keyboard, or right-clicking on the dot and choosing Remove Selection. Remember, whatever value you set Frame to, that will be the value of the keyframe when you press the Add Keyframe button. You can also click and drag keyframes to change their order in the timeline.

Repeat the process for the other animations, using the following table to guide you on which keyframes to use for each direction:

| **Animation** | **Frames** |
| Down | `0, 1, 2, 3, 4` |
| Left | `5, 6, 7, 8, 9` |
| Right | `10, 11, 12, 13, 14` |
| Up | `15, 16, 17, 18, 19` |

As long as the spritesheet for a character follows the same 5 x 4 arrangement, this `AnimationPlayer` configuration will work, and you won't need to create separate animations for each character. In larger projects, it can be a huge time-saver to create all your spritesheet animations while following a common pattern.

# Collision detection

Because the characters are moving on a grid, they need to either move the full distance to the next tile or not at all. This means that, before moving, the character needs to check to see if the move is possible. One way to test if an adjacent square has anything in it is by using a *raycast*. **Raycasting** means extending a ray from the character's position to a given destination. If the ray encounters any object along the way, it will report that contact. By adding four rays to the character, it can *look* at the squares around it to see if they are unoccupied.

Add four `RayCast2D` nodes and set their names and **Cast To** properties as follows:

| **Name** | **Cast To** |
| RayCastRight | `(64, 0)` |
| RayCastLeft | `(-64, 0)` |
| RayCastDown | `(0, 64)` |
| RayCastUp | `(0, -64)` |

Make sure to set the Enabled property on each one (`RayCast2D` options are disabled by default). Your final node setup should look like this:

![](img/00054.jpeg)

# Character script

Now, add a script to the `Character` node (make sure you've saved the scene first, and the script will automatically be named `Character.gd`). First, define the class variables:

```cpp
extends Area2D

export (int) var speed

var tile_size = 64
var can_move = true
var facing = 'right'
var moves = {'right': Vector2(1, 0),
             'left': Vector2(-1, 0),
             'up': Vector2(0, -1),
             'down': Vector2(0, 1)}
onready var raycasts = {'right': $RayCastRight,
                        'left': $RayCastLeft,
                        'up': $RayCastUp,
                        'down': $RayCastDown}
```

`speed` will control the movement and animation speed of the character, allowing you to customize the movement speed. As you learned in [Chapter 1](part0022.html#KVCC0-5809b3bef8d2453086d97dfad17b2ee2), *Introduction*, using `export` allows you to set the value of a variable via the Inspector. Save the script and set the Speed property to `3` in the Inspector.

`can_move` is a flag that will track whether the character is allowed to move during the current frame. It will be set to `false` while the movement is underway, preventing a second movement from being started before the previous one has finished. `facing` is a string denoting the current direction of movement (again, spelled and capitalized exactly like the input actions you created at the beginning of the project). The `moves` dictionary contains vectors describing the four directions, while the `raycasts` dictionary contains references to the four raycast nodes. Note that both dictionaries' keys match the input action names.

When referencing another node during variable declaration, you must use `onready` to ensure that the variable isn't set before the referenced node is ready. You can think of it as a shortcut to writing the code in the `_ready()` function. This line:
`onready var sprite = $Sprite`
Is equivalent to writing this:
`var sprite`
`func _ready():`
`    sprite = $Sprite`

The following is the code that will execute a movement from one square to another:

```cpp
func move(dir):
    $AnimationPlayer.playback_speed = speed
    facing = dir
    if raycasts[facing].is_colliding():
        return

    can_move = false
    $AnimationPlayer.play(facing)
    $MoveTween.interpolate_property(self, "position", position,
                position + moves[facing] * tile_size,
                1.0 / speed, Tween.TRANS_SINE, Tween.EASE_IN_OUT)
    $MoveTween.start()
    return true
```

`move()` takes a direction as an argument. If the `RayCast2D` for the given direction detects a collision, the move is canceled and the function returns without executing further (note that the return value will be `null`). Otherwise, it changes `facing` to the new direction, disables additional movement with `can_move`, and starts playing the matching animation. To actually perform the movement, the `Tween` node interpolates the `position` property from its current value to its current value plus a tile-sized movement in the given direction. The duration (`1.0 / speed` seconds) is set to match the length of the animation.

Using the `Tween.TRANS_SINE` transition type results in a pleasing, smooth movement that accelerates up and then down to the final position. Feel free to try other transition types here to alter the movement style.

Finally, to enable movement again, you need to reset `can_move` when the movement has finished. Connect the `tween_completed` signal from `MoveTween` and add the following:

```cpp
func _on_MoveTween_tween_completed( object, key ):
    can_move = true
```

# Player scene

The player scene needs to contain all the same nodes we gave to `Character`. This is where you'll take advantage of the power of inheritance.

Start by making a new scene. However, instead of making a new empty scene, click on Scene | New Inherited Scene in the menu. In the Open Base Scene window, select `res://character/Character.tscn`, as shown in the following screenshot:

![](img/00055.jpeg)

Rename the root node of this new scene from `Character` to `Player` and save the new scene. Note that all the `Character` nodes are also present. If you make a change to `Character.tscn` and save it, the changes will also take effect in the `Player` scene.

Now, you need to set the Player's physics layers, so find the Collision section in the Inspector and set the Layer and Mask properties. Layer should be set to player only, while Mask should show walls, enemies, and items. Refer to the following screenshot:

![](img/00056.jpeg)

The collision layers system is a powerful tool that allows you to customize which objects can detect each other. The Layer property places the object in one or more collision layers, while the Mask property defines what layers the object can *see*. If another object is not in one of its mask layers, it will not be detected or collided with.

The only other node that needs to be changed is the `Sprite`, where you need to set the texture. Drag the player spritesheet from the `res://assets` folder and drop it in the Texture property of the `Sprite`. Go ahead and test out the animations in the `AnimationPlayer` and make sure they're showing the correct directions. If you find a problem with any of the animations, make sure you fix it in the `Character` scene, and it will automatically be fixed in the `Player` scene as well:

![](img/00057.jpeg)

Add a `Camera` node as a child of `Player` and check its Current property to On. Godot will automatically render whatever the current camera sees in the game window. This will allow you to make maps of any size, and the camera will scroll the map as the player walks around on it. Note that when you add the camera, a purplish box appears, which is centered on the player. This represents the camera's visible region, and because it's a child of the player, it follows the player's movement. If you look at the camera's properties in the Inspector, you'll see four Limit properties. These are used to stop the camera from scrolling past a certain point; the edge of your map, for example. Try adjusting them and see how the box stops following the `Player` as you drag it around the screen (make sure you're moving the `Player` node itself and not one of its children). Later, the limits will be set automatically by the level itself so that the camera won't scroll "outside" the level.

# Player script

The player's script also needs to extend the character's. Remove the attached script (`Character.gd`) by selecting the `Player` node and clicking the Clear script button:

![](img/00058.jpeg)

Now, click the button again to attach a new script. In the Attach Node Script dialog, click the folder icon next to the Inherits option and select `Character.gd`:

![](img/00059.jpeg)

Here is the player script (note that it `extends` the character script):

```cpp

extends "res://character/Character.gd"

signal moved

func _process(delta):
    if can_move:
        for dir in moves.keys():
            if Input.is_action_pressed(dir):
                if move(dir):
                    emit_signal('moved')
```

Because it inherits all the behavior from `Character.gd`, the player will also have the `move()` function. You just need to extend it with code to call `move()` based on the input events. As you've seen before, you can use the `process()` function to check the input state each frame. However, only if `can_move` allows it do you actually check the inputs and call `move()`.

Because you used the names `up`, `down`, `left`, and `right` for the input actions as well as the keys to the `moves` and `raycasts` dictionaries, you can loop through those keys and check each one as an input as well.

Recall that `move()` returns `true` if it succeeds. If it does, the player emits the `moved` signal, which you'll be able to use later with the enemies.

Run the scene and try moving the player character around the screen.

The player doesn't have a level to walk around on yet, but you can go ahead and add the code the player will need later. As the player moves around the level, it will encounter various objects and needs to respond to them. By using signals, you can add the code for this before you've even created the level. Add three more signals to the script:

```cpp
signal dead
signal grabbed_key
signal win
```

Then, connect the `area_entered` signal of the `Player` and add this code:

```cpp
func _on_Player_area_entered( area ):
    if area.is_in_group('enemies'):
        emit_signal('dead')
    if area.has_method('pickup'):
        area.pickup()
    if area.type == 'key_red':
        emit_signal('grabbed_key')
    if area.type == 'star':
        emit_signal('win')
```

Whenever the player encounters another `Area2D`, this function will run. If the object is an enemy, the player loses the game. Note the use of `has_method()`. This allows you to identify collectible objects by checking whether they have a `pickup()` method and only call the method if it exists.

# Enemy scene

Hopefully, you're seeing how inheritance works by now. You'll create the `Enemy` scene using the same procedure. Make a new scene inheriting from `Character.tscn` and name it `Enemy`. Drag the mob spritesheet, `res://assets/slime.png`, to the `Sprite`'s Texture.

In the Collision section of the Inspector, set the Layer and Mask properties. Layer should be set to enemies, while Mask should show walls and player.

As you did with the `Player`, remove the existing script and attach a new script inheriting from `Character.gd`:

```cpp
extends "res://character/Character.gd"

func _ready():
    can_move = false
    facing = moves.keys()[randi() % 4]
    yield(get_tree().create_timer(0.5), 'timeout')
    can_move = true

func _process(delta):
    if can_move:
         if not move(facing) or randi() % 10 > 5:
             facing = moves.keys()[randi() % 4] 
```

The code in the `_ready()` function serves an important purpose: because the enemies are added to the tree *below* the `TileMap` nodes, they'll be processed first. You don't want the enemies to start moving before the walls have been processed, or they could step onto a wall tile and get stuck. You need to have a small delay before they start, which also serves to give the player a moment to prepare. To do this, rather than add a `Timer` node to the scene, you can use the `create_timer()` function of the `SceneTree` to make a one-off timer, yielding execution until its timeout signal fires.

GDScript's `yield()` function provides a way to *pause* execution of a function until a later time, while allowing the rest of the game to continue running. When passed an object and a named signal, execution will resume when that object emits the given signal.

Every frame, the enemy will move if it is able to. If it runs into a wall (that is, when `move()` returns `null`), or sometimes just randomly, it changes direction. The result will be an unpredictable (and hard to dodge!) enemy movement. Remember that you can adjust the `Player` and `Enemy` speeds independently in their scenes, or change `speed` in the `Character` scene and it will affect them both.

# Optional – turn-based movement

For a different style of game, you could put the `_process()` movement code in a function called `_on_Player_moved()` instead, and connect it to the player's `moved` signal. This would make the enemies move only when the player does, giving the game more of a strategic feel, rather than one of fast-paced action.

# Creating the level

In this section, you'll create the map where all the action will take place. As the name implies, you'll probably want to make a maze-like level with lots of twists and turns.

Here is a sample level:

![](img/00060.jpeg)

The player's goal is to reach the star. Locked doors can only be opened by picking up the key. The green dots mark the spawn locations of enemies, while the red dot marks the player's start location. The coins are extra items that can be picked up along the way for bonus points. Note that the entire level is larger than the display window. The `Camera` will scroll the map as the player moves around it.

You'll use the `TileMap` node to create the map. There are several benefits to using a `TileMap` for your level design. First, they make it possible to draw the level's layout by *painting* the tiles onto a grid, which is much faster than placing individual `Sprite` nodes one by one. Secondly, they allow for much larger levels because they are optimized for drawing large numbers of tiles efficiently by batching them together and only drawing the *chunks* of the map that are visible at a given time. Finally, you can add collision shapes to individual tiles and the entire map will act as a single collider, simplifying your collision code.

Once you've completed this section, you'll be able to create as many of these maps as you wish. You can put them in order to give a progression from level to level.

# Items

First, create a new scene for the collectable objects that the player can pick up. These items will be spawned by the map when the game is run. Here is the scene tree:

![](img/00061.jpeg)

Leave the `Sprite` Texture blank. Since you're using this object for multiple items, the texture can be set in the item's script when it's created.

Set the `Pickup` Collision Layer to items and its Mask to player. You don't want the enemies collecting the coins before you get there (although that might make for a fun variation on the game where you race to get as many coins as you can before the bad guys gobble them up).

Give the `CollisionShape2D` node a rectangle shape and set its extents to `(32, 32)` (strictly speaking, you can use any shape, as the player will move all the way onto the tile and completely overlap the item anyway).

Here is the script for the `Pickup`:

```cpp
extends Area2D

var textures = {'coin': 'res://assets/coin.png',
                'key_red': 'res://assets/keyRed.png',
                'star': 'res://assets/star.png'}
var type

func _ready():
    $Tween.interpolate_property($Sprite, 'scale', Vector2(1, 1),
        Vector2(3, 3), 0.5, Tween.TRANS_QUAD, Tween.EASE_IN_OUT)
    $Tween.interpolate_property($Sprite, 'modulate',
        Color(1, 1, 1, 1), Color(1, 1, 1, 0), 0.5,
        Tween.TRANS_QUAD, Tween.EASE_IN_OUT)

func init(_type, pos):
    $Sprite.texture = load(textures[_type])
    type = _type
    position = pos

func pickup():
    $CollisionShape2D.disabled = true
    $Tween.start()
```

The `type` variable will be set when the item is created and used to determine what texture the object should use. Using `_type` as the variable name in the function argument lets you use the name without conflicting with `type`, which is already in use.

Some programming languages use the notion of *private* functions or variables, meaning they are only used locally. The `_` naming convention in GDScript is used to visually designate variables or functions that should be regarded as private. Note that they aren't actually any different from any other name; it is merely a visual indication for the programmer. 

The pickup effect using `Tween` is similar to the one you used for the coins in Coin Dash—animating the scale and opacity of `Sprite`. Connect the `tween_completed` signal of `Tween` so that the item can be deleted when the effect has finished:

```cpp
func _on_Tween_tween_completed( object, key ):
     queue_free()
```

# TileSets

In order to draw a map using a `TileMap`, it must have a `TileSet` assigned to it. The `TileSet` contains all of the individual tile textures, along with any collision shapes they may have.

Depending on how many tiles you have, it can be time-consuming to create a `TileSet`, especially the first time. For that reason, there is a pre-generated `TileSet` included in the `assets` folder titled `tileset.tres`. Feel free to use that instead, but please don't skip the following section. It contains useful information to help you understand how the `TileSet` works.

# Creating a TileSet

A `TileSet` in Godot is a type of `Resource`. Examples of other resources include Textures, Animations, and Fonts. They are containers that hold a certain type of data, and are typically saved as `.tres` files.

By default, Godot saves files in text-based formats, indicated by the `t` in `.tscn` or `.tres`, for example. Text-based files are preferred over binary formats because they are human-readable. They are also more friendly for **Version Control Systems** (**VCS**), which allow you to track file changes over the course of building your project.

To make a `TileSet`, you create a scene with a set of `Sprite` nodes containing the textures from your art assets. You can then add collisions and other properties to those `Sprite` tiles. Once you've created all the tiles, you export the scene as a `TileSet` resource, which can then be loaded by the `TileMap` node.

Here is a screenshot of the `TileSetMaker.tscn` scene, containing the tiles you'll be using to build this game's levels:

![](img/00062.jpeg)

Start by adding a `Sprite` node and setting its texture to `res://assets/sokoban_tilesheet.png`. To select a single tile, set the Region/Enabled property to On and click Texture Region at the bottom of the editor window to open the panel. Set Snap Mode to Grid Snap and the Step to 64px in both *x* and *y*. Now, when you click and drag in the texture, it will only allow you to select 64 x 64 sections of the texture:

![](img/00063.jpeg)

Give the Sprite an appropriate name (`crate_brown` or `wall_red`, for example)—this name will appear as the tile's name in the `TileSet`. Add a `StaticBody2D` as a child, and then add a `CollisionPolygon2D` to that. It is important that the collision polygon be sized properly so that it aligns with the tiles placed next to it. The easiest way to do this is to turn on grid snapping in the editor window.

Click the Use Snap button (it looks like a magnet) and then open the snap menu by clicking on the three dots next to it:

![](img/00064.jpeg)

Choose Configure Snap... and set the Grid Step to `64` by `64`:

![](img/00065.jpeg)

Now, with the `CollisionPolygon2D` selected, you can click in the four corners of the tile one by one to create a closed square (it will appear as a reddish orange):

![](img/00066.jpeg)

This tile is now complete. You can duplicate it (*Ctrl* + *D*) and make another, and you only need to change the texture region. Note that collision bodies are only needed on the wall tiles. The ground and item tiles should not have them.

When you've created all your tiles, click Scene | Convert To | TileSet and save it with an appropriate name, such as `tileset.tres`. If you come back and edit the scene again, you'll need to redo the conversion. Pay special attention to the Merge With Existing option. If this is set to On, the current scene's tiles will be *merged* with the ones already in the `tileset` file. Sometimes, this can result in changes to the tile indices and change your map in unwanted ways. Take a look at the following screenshot:

![](img/00067.jpeg)`tres` stands for text resource and is the most common format Godot stores its resource files in. Compare this with `tscn`, which is the text scene storage format.

Your `TileSet` resource is ready to use!

# TileMaps

Now, let's make a new scene for the game level. The level will be a self-contained scene, and will include the map and the player, and will handle spawning any items and enemies in the level. For the root, use a `Node2D` and name it `Level1` (later, you can duplicate this node setup to create more levels).

You can open the `Level1.tscn` file from the assets folder to see the completed level scene from this section, although you're encouraged to create your own levels.

When using `TileMap`, you will often want more than one tile object to appear in a given location. You might want to place a tree, for example, but also have a ground tile appear below it. This can be done by using `TileMap` as many times as you like to create layers of data. For your level, you'll make three layers to display the ground, which the player can walk on; the walls, which are obstacles; and the collectible items, which are markers for spawning items like coins, keys, and enemies.

Add a `TileMap` and name it `Ground`. Drag the `tileset.tres` into the Tile Set property and you'll see the tiles appear, ready to be used, on the right-hand side of the editor window:

![](img/00068.jpeg)

It's very easy to accidentally click and drag in the editor window and move your whole tile map. To prevent this, make sure you select the `Ground` node and click the Lock button: ![](img/00069.jpeg).

Duplicate this `TileMap` twice and name the new `TileMap` nodes `Walls` and `Items`. Remember that Godot draws objects in the order listed in the node tree, from top to bottom, so `Ground` should be at the top, with `Walls` and `Items` underneath it.

As you're drawing your level, be careful to note which layer you're drawing on! You should only place the item markers on the Items layer, for example, because that's where the code is going to look for objects to create. Don't place any other objects there, though, because the layer itself will be invisible during gameplay.

Finally, add an instance of the `Player` scene. Make sure the `Player` node is below the three `TileMap` nodes, so it will be drawn on top. The final scene tree should look like this:

![](img/00070.jpeg)

# Level script

Now that the level is complete, attach a script to create the level behavior. This script will first scan the `Items` map to spawn any enemies and collectibles. It will also serve to monitor for events that occur during gameplay, such as picking up a key or running into an enemy:

```cpp
extends Node2D

export (PackedScene) var Enemy
export (PackedScene) var Pickup

onready var items = $Items
var doors = []
```

The first two variables contain references to the scenes that will need to be instanced from the `Items` map. Since that particular map node will be referenced frequently, you can cache the `$Items` lookup in a variable to save some time. Finally, an array called `doors` will contain the door location(s) found on the map.

Save the script and drag the `Enemy.tscn` and `Pickup.tscn` files into their respective properties in the Inspector.

Now, add the following code for `_ready()`:

```cpp
func _ready():
    randomize()
    $Items.hide()
    set_camera_limits()
    var door_id = $Walls.tile_set.find_tile_by_name('door_red')
    for cell in $Walls.get_used_cells_by_id(door_id):
        doors.append(cell)
    spawn_items()
    $Player.connect('dead', self, 'game_over')
    $Player.connect('grabbed_key', self, '_on_Player_grabbed_key')
    $Player.connect('win', self, '_on_Player_win')
```

The function starts by ensuring that the `Items` tilemap is hidden. You don't want the player to see those tiles; they exist so the script can detect where to spawn items.

Next, the camera limits must be set, ensuring that it can't scroll past the edges of the map. You'll create a function to handle that (see the following code).

When the player finds a key, the door(s) need to be opened, so the next part searches the `Walls` map for any `door_red` tiles and stores them in an array. Note that you must first find the tile's `id` from the `TileSet`, because the cells of the `TileMap` only contain ID numbers that refer to the tile set.

More on the `spawn_items()` function follows.

Finally, the `Player` signals are all connected to functions that will process their results.

Here's how to set the camera limits to match the size of the map:

```cpp
func set_camera_limits():
    var map_size = $Ground.get_used_rect()
    var cell_size = $Ground.cell_size
    $Player/Camera2D.limit_left = map_size.position.x * cell_size.x
    $Player/Camera2D.limit_top = map_size.position.y * cell_size.y
    $Player/Camera2D.limit_right = map_size.end.x * cell_size.x
    $Player/Camera2D.limit_bottom = map_size.end.y * cell_size.y
```

`get_used_rect()` returns a `Vector2` containing the size of the `Ground` layer in cells. Multiplying this by the `cell_size` gives the total map size in pixels, which is used to set the four limit values on the `Camera` node. Setting these limits ensures you won't see any *dead* space outside the map when you move near the edge.

Now, add the `spawn_items()` function:

```cpp
func spawn_items():
    for cell in items.get_used_cells():
        var id = items.get_cellv(cell)
        var type = items.tile_set.tile_get_name(id)
        var pos = items.map_to_world(cell) + items.cell_size/2
        match type:
            'slime_spawn':
                var s = Enemy.instance()
                s.position = pos
                s.tile_size = items.cell_size
                add_child(s)
            'player_spawn':
                $Player.position = pos
                $Player.tile_size = items.cell_size
            'coin', 'key_red', 'star':
                var p = Pickup.instance()
                p.init(type, pos)
                add_child(p)
```

This function looks for the tiles in the `Items` layer, returned by `get_used_cells()`. Each cell has an `id` that maps to a name in the `TileSet` (the names that were assigned to each tile when the `TileSet` was made). If you made your own tile set, make sure you use the names that match your tiles in this function. The names used in the preceding code match the tile set that was included in the asset download.

`map_to_world()` converts the tile map position to pixel coordinates. This gives you the upper-left corner of the tile, so then you must add one half-size tile to find the center of the tile. Then, depending on what tile was found, the matching item object is instanced.

Finally, add the three functions for the player signals:

```cpp
func game_over():
    pass

func _on_Player_win():
    pass

```

```cpp
func _on_Player_grabbed_key():
    for cell in doors:
        $Walls.set_cellv(cell, -1)
```

The player signals `dead` and `win` should end the game and go to a Game Over screen (which you haven't created yet). Since you can't write the code for those functions yet, use `pass` for the time being. The key pickup signal should remove any door tiles (by setting their tile index to `-1`, which means an empty tile).

# Adding more levels

If you want to make another level, you just need to duplicate this scene tree and attach the same script to it. The easiest way to do this is to use Scene | Save As and save the level as `Level2.tscn`. Then, you can use some of the existing tiles or draw a whole new level layout.

Feel free to do this with as many levels as you like, making sure to save them all in the `levels` folder. In the next section, you'll see how to link them together so that each level will lead to the next. Don't worry if you number them incorrectly; you'll be able to put them in whatever order you like.

# Game flow

Now that you have the basic building blocks completed, you need to tie everything together. In this section, you'll create:

*   The Start and Game Over screens
*   A global script to manage persistent data

The basic flow of the game follows the following chart:

![](img/00071.jpeg)

The player is sent to the end screen whenever he/she dies, or when they reach and complete the last level. After a brief time, the end screen returns the player to the start screen so that a new game can be played.

# Start and end screens

You need two scenes for this part: a start or title screen that shows before the game (and lets the player start the game), and a game over screen to notify the player that the game has ended. 

Make a new scene and add a `Control` node named `StartScreen`. Add a Label as a child and add `res://assets/Unique.ttf` as a new `DynamicFont` with a font size of `64`. Set the Align and Valign properties to Center and the Text to `Escape the Maze!`. In the Layout menu, select Full Rect. Now, duplicate this node and set the second label's Text to Press <space>.

For this demonstration, the `StartScreen` is being kept very plain. Once you have it working, feel free to add decorations, or even an `AnimationPlayer` to make a player Sprite run across the screen.

Choose Scene | Save As to save another copy of this scene and name it `EndScreen`. Delete the second `Label` (the one that says Press <space>) and add a `Timer` node. Set the Autostart property to On, One Shot to On, and Wait Time to `3`.

The `Timer` will send the game back to the `StartScreen` after it expires.

However, before you can connect these other scenes together, you need to understand how to work with persistent data and *Autoloads*.

# Globals

It is a very common scenario in game development that you have some data that needs to persist across multiple scenes. Data that is part of a scene is lost when the scene is switched, so persistent data must reside somewhere outside the current scene.

Godot solves this problem with the use of AutoLoads. These are scripts or nodes that are automatically loaded in every scene. Because Godot does not support global variables, an autoload acts like a *Singleton.* This is a node (with attached script) that is automatically loaded in *every* scene. Common uses for AutoLoads include storing global data (score, player data, and so on), handling scene switching functions, or any other functions that need to be independent of the currently running scene.

**Singleton** is a well-known pattern in programming which describes a class that only allows for a single instance of itself, and provides direct access to its member variables and functions. In game development, it is often used for persistent data that needs to be accessible by various parts of the game.

When deciding if you need a singleton, ask yourself whether the object or data needs to *always* exist and if there will always be *only one* instance of that object.

# Global script

First, make a new script by clicking File | New in the Script window. Make sure it inherits from `Node` (this is the default), and in the `Path` field, set the name to `Global.gd`. Click Create and add the following code to the new script:

```cpp
extends Node

var levels = ['res://levels/Level1.tscn',
              'res://levels/Level2.tscn']
var current_level

var start_screen = 'res://ui/StartScreen.tscn'
var end_screen = 'res://ui/EndScreen.tscn'

func new_game():
    current_level = -1
    next_level()

func game_over():
    get_tree().change_scene(end_screen)

func next_level():
    current_level += 1
    if current_level >= Global.levels.size():
        # no more levels to load :(
        game_over()
    else:
        get_tree().change_scene(levels[current_level])
```

This script provides a number of functions you'll need.

Most of the work is done by the `change_scene()` method of the `SceneTree`. The `SceneTree` represents the foundation of the currently running scene. When a scene is loaded or a new node is added, it becomes a member of the `SceneTree`. `change_scene()` replaces the current scene with a given one. 

The `next_level()` function progresses through the list of levels you've made, which are listed in the `levels` array. If you reach the end of the list, the game ends.

To add this script as an autoload, open Project Settings and click on the AutoLoad tab. Click the .. button next to Path and select your `Global.gd` script. The node Name will automatically be set to Global (this is the name you'll use to reference the node in your scripts, as shown in the following screenshot):

![](img/00072.jpeg)

Now, you can access any of the global script's properties by using its name in any script across your whole game, for example, `Global.current_level`.

Attach the following script to the `StartScreen`:

```cpp
extends Control

func _input(event):
    if event.is_action_pressed('ui_select'):
        Global.new_game()
```

This script waits for the spacebar to be pressed and then calls the `new_game()` function of `Global`.

Add this one to `EndScreen`:

```cpp
extends Control

func _on_Timer_timeout():
    get_tree().change_scene(Global.start_screen)
```

You'll also need to connect the `timeout` signal of `Timer`. To do this, you have to create the script first, then the `Connect` button will create the new function for you.

In the `Level.gd` script, you can now fill in the remaining two functions:

```cpp
func _on_Player_win():
    Global.next_level()

func game_over():
    Global.game_over()
```

# Score

The global singleton is a great place to keep the player's score so that it will be persistent from level to level. Start by adding a `var score` variable at the top of the file, and then in `new_game()`, add `score = 0`.

Now, you need to add a point whenever a coin is collected. Go to `Pickup.gd` and add `signal coin_pickup` at the top. You can emit this signal in the `pickup()` function:

```cpp
func pickup():
    match type:
        'coin':
            emit_signal('coin_pickup', 1)
    $CollisionShape2D.disabled = true
    $Tween.start()
```

The value of `1` is included here in case you want to later change the number of points that coins are worth, or add other objects that add different point amounts. This signal will be used to update the display, so now you can create the `HUD`. 

Make a new scene with a `CanvasLayer` named `HUD` and save the scene. Add a `MarginContainer` node as a child, and under that, a `Label` named `ScoreLabel`.

Set the `MarginContainer` Layout to Top Wide and its four margin properties (found under Custom Constants) all to `20`. Add the same Custom Font properties you used before for the start and end screens, then attach a script:

```cpp
extends CanvasLayer

func _ready():
    $MarginContainer/ScoreLabel.text = str(Global.score)

func update_score(value):
    Global.score += value
    $MarginContainer/ScoreLabel.text = str(Global.score)
```

Add an instance of the `HUD` to the `Level` scene. Remember from the previous project that the `CanvasLayer` node will remain on top of the rest of the game. It will also ignore any camera movement, so the display will remain fixed in place as the player moves around the level.

Finally, in the `Level.gd` script, when you spawn a new collectible object, connect the signal to the `HUD` function:

```cpp
    'coin', 'key_red', 'star':
        var p = Pickup.instance()
        p.init(type, pos)
        add_child(p)
        p.connect('coin_pickup', $HUD, 'update_score')
```

Run the game and collect a few coins to confirm that the score is updating.

# Saving the High Score

Many games require you to save some kind of information between play sessions. This is information that you want to remain available, even when the application itself has quit. Examples include saved games, user-created content, or downloadable resource packs. For this game, you'll save a High Score value that will persist across game sessions.

# Reading and writing files

As you've seen before, Godot keeps all resources stored as files in the project folder. From code, these are accessible under the `res://` folder path. For example, `res://project.godot` will always point to the current project's configuration file, no matter where on your computer the project is actually stored.

However, the `res://` filesystem is set as read-only for safety when the project is run. It is also read-only when the project is exported. Any data that needs to be retained by the user is placed in the `user://` file path. Where this folder physically exists will vary depending on what platform the game is running on.

You can find the current platform's user-writable data folder using `OS.get_user_data_dir()`. Add a `print()` statement to the `ready()` function of one of your scripts to see what the location is on your system.

Reading and writing to files is accomplished using a `File` object. This object is used to open the file in read and/or write mode, and can also be used to test for a file's existence.

Add the following code to `Global.gd`:

```cpp
var highscore = 0
var score_file = "user://highscore.txt"

func setup():
    var f = File.new()
    if f.file_exists(score_file):
        f.open(score_file, File.READ)
        var content = f.get_as_text()
        highscore = int(content)
        f.close()
```

You first need to test whether the file exists. If it does, you can read the value, which is being stored as human-readable text, and assign it to the `highscore` variable. Binary data can also be stored in files, if needed, but text will allow you to look at the file yourself and check that everything is working.

Add the following code to check if the player has beat the previous high score:

```cpp
func game_over():
    if score > highscore:
        highscore = score
        save_score()
    get_tree().change_scene(end_screen)

func save_score():
    var f = File.new()
    f.open(score_file, File.WRITE)
    f.store_string(str(highscore))
    f.close()
```

The `save_score()` function opens the file to write the new value. Note that if the file doesn't exist, opening in `WRITE` mode will automatically create it.

Next, you need to call the `setup()` function when the game starts, so add this to `Global.gd`:

```cpp
func _ready():
    setup()
```

Finally, to display the high score, add another `Label` node to the `StartScreen` scene (you can duplicate one of the existing ones). Arrange it below the other Labels (or in whatever order you like) and name it `ScoreNotice`. Add the following to the script:

```cpp
func _ready():
    $ScoreNotice.text = "High Score: " + str(Global.highscore)
```

Run the game and check that your high score is increasing (when you beat it) and persisting when you quit and start the game again.

# Finishing touches

Now that the main functionality of the game is complete, you can add a few more features to polish it up a little bit.

# Death animation

When the enemy hits the player, you can add a small animation rather than just ending the game. The effect will spin the character around while shrinking its scale property.

Start by selecting the `AnimationPlayer` node of the `Player` and clicking the New Animation button: ![](img/00073.jpeg). Name the new animation `die`.

In this animation, you'll be animating the Sprite's Rotation Degrees and Scale properties. Find the Rotation Degrees property in the Inspector and click the key, ![](img/00074.jpeg), to add a track. Move the scrubber to the end of the animation, change Rotation Degrees to 360, and click the key again. Try playing the animation to see the character spin.

Keep in mind that while degrees are typically used for Inspector properties, when writing code most Godot functions expect angles to be measured in *radians*.

Now, do the same thing with the *Scale* property. Add a keyframe (at the beginning!) for `(1, 1)` and then another at the end with the scale set to `(0.2, 0.2)`. Try playing the animation again to see the results.

The new animation needs to be triggered when the player hits an enemy. Add the following code to the player's `_on_Player_area_entered()` function:

```cpp
if area.is_in_group('enemies'):
    area.hide()
    set_process(false)
    $CollisionShape2D.disabled = true
    $AnimationPlayer.play("die")
    yield($AnimationPlayer, 'animation_finished')
    emit_signal('dead')
```

The added code takes care of a few things that need to happen. First, hiding the enemy that was hit makes sure that it doesn't cover the player and prevent you from seeing our new animation. Next, you use `set_process(false)` to stop the `_process()` function from running so that the player can't keep moving during the animation. You also need to disable the player's collision detection so that it doesn't detect another enemy if it happens to wander by.

After starting the `die` animation, you need to let it finish before emitting the `dead` signal, so `yield` is used to wait for the signal from `AnimationPlayer`.

Try running the game and getting hit by an enemy to see the animation. If everything works fine, you'll notice something wrong on the next playthrough: the player is tiny! The animation ends with the Sprite's Scale set to `(0.2, 0.2)` and nothing is setting it back to normal size. Add the following to the Player's script so that the scale will always start at the right value:

```cpp
func _ready():
    $Sprite.scale = Vector2(1, 1)
```

# Sound effects

There are six sound effects in the `res://assets/audio` folder for you to use in the game. These audio files are in OGG format. By default, Godot sets OGG files to loop when imported. Select the OGG files in the FileSystem tab (you can use *Shift* + Click to select multiple files) and click the Import tab on the right-hand side of the editor window. Uncheck Loop and click the Reimport button:

![](img/00075.jpeg)

First, add the pickup sounds for the items. Add two `AudioStreamPlayer` nodes to the `Pickup` scene and name them `KeyPickup` and `CoinPickup`. Drag the corresponding audio file into the Stream property of each node.

You can also adjust the sound's volume via its Volume Db property, as shown in the following screenshot:

![](img/00076.jpeg)

Add the following code to the beginning of the `pickup()` function:

```cpp
match type:
    'coin':
        emit_signal('coin_pickup', 1)
        $CoinPickup.play()
    'key_red':
        $KeyPickup.play()
```

The other sound effects will be added to the `Player` scene. Add three of the `AudioStreamPlayer` and name them `Win`, `Lose`, and `Footsteps`, adding the matching sound file to each node's Stream. Update the `_on_Player_area_entered()` function as follows:

```cpp
    if area.type == 'star':
        $Win.play()
        $CollisionShape2D.disabled = true
        yield($Win, "finished")
        emit_signal('win')
```

You need to disable the collision and `yield` for the sound to finish, or else it would be instantly terminated by the next level loading. This way, the player has time to hear the sound before moving on.

To play the footsteps, add `$Footsteps.play()` after `if move(dir):` in the `_process()` function. Note: you may want to reduce the sound of the footsteps so that they don't overwhelm everything; they should be subtle background sounds. In the `Footsteps` node, set the Volume Db property to `-30`.

Finally, to play the `Lose` sound, add it to the enemy collision code here:

```cpp

if area.is_in_group('enemies'):
    area.hide()
    $CollisionShape2D.disabled = true
    set_process(false)
    $Lose.play()
    $AnimationPlayer.play("die")
    yield($Lose, 'finished')
    emit_signal('dead')
```

Note that you need to change the yield function. Since the sound is slightly longer than the animation, it will get cut off if you end it on the animation's completion. Alternatively, you could adjust the duration of the animation to match the length of the sound.

# Summary

In this project, you have learned how to take advantage of Godot's inheritance system to organize and share code between different objects in your game. This is a very powerful tool that you should keep in mind whenever you start building a new game. If you start making multiple objects that repeat the same properties and/or code, you should probably stop and think about what you're doing. Ask yourself: *can I use inheritance here to share what these objects have in common?* In a bigger game with many more objects, this can save you a large amount of time.

You saw how the `TileMap` node works and how it allows you to quickly design maps and spawn new objects. They have many uses across many game genres. As you'll see later in this book, TileMaps are also ideal for designing platform game levels as well.

You were also introduced to the *AutoLoad* feature, which allows you to create a global script that contains persistent data used across multiple scenes. You also learned how to implement grid-based movement and used the `AnimationPlayer` to work with spritesheet animations.

In the next chapter, you'll learn about Godot's powerful physics body: the `RigidBody2D`. You'll use it to create a game in a classic genre: the space shooter.