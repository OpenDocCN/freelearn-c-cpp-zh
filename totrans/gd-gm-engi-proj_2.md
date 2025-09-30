# Coin Dash

This first project will guide you through making your first Godot Engine project. You will learn how the Godot editor works, how to structure a project, and how to build a small 2D game.

Why 2D? In a nutshell, 3D games are much more complex than 2D ones, while many of the underlying game engine features you'll need to know are the same. You should stick to 2D until you have a good understanding of Godot's game development process. At that point, the jump to 3D will be much easier. You'll get an introduction to 3D in this book's fifth and final project.

Important—don't skip this chapter, even if you aren't a complete newcomer to game development. While you may already understand many of the underlying concepts, this project will introduce a number of fundamental Godot features and design paradigms that you'll need to know going forward. You'll build on these concepts as you develop more complex projects.

The game in this chapter is called **Coin Dash**. Your character must move around the screen, collecting as many coins as possible while racing against the clock. When you're finished, the game will look like this:

![](img/00015.jpeg)

# Project setup

Launch Godot and create a new project, making sure to use the `Create Folder` button to ensure that this project's files will be kept separate from other projects. You can download a Zip file of the art and sounds (collectively known as *assets*) for the game here, [https://github.com/PacktPublishing/Godot-Game-Engine-Projects/releases](https://github.com/PacktPublishing/Godot-Game-Engine-Projects/releases).

Unzip this file in your new project folder.

In this project, you will make three independent scenes: `Player`, `Coin`, and `HUD`, which will all be combined into the game's `Main` scene. In a larger project, it might be useful to make separate folders to hold each scene's assets and scripts, but for this relatively small game, you can save your scenes and scripts in the root folder, which is referred to as `res://` (**res** is short for **resource**). All resources in your project will be located relative to the `res://` folder. You can see your project folders in the FileSystem dock in the upper-left corner:

![](img/00016.jpeg)

For example, the images for the coin would be located in `res://assets/coin/`.

This game will use portrait mode, so you need to adjust the size of the game window. Click on the Project menu and select Project Settings, as shown in the following screenshot:

![](img/00017.jpeg)

Look for the Display/Window section and set Width to `480` and Height to `720`. Also in this section, set the Stretch/Mode to `2D` and the Aspect to `keep`. This will ensure that if the user resizes the game window, everything will scale appropriately and not become stretched or deformed. If you like, you can also uncheck the box for Resizable, to prevent the window from being resized entirely.

# Vectors and 2D coordinate systems

Note: This section is a very brief overview of 2D coordinate systems and does not delve very deeply into vector math. It is intended as a high-level overview of how such topics apply to game development in Godot. Vector math is an essential tool in game development, so if you need a broader understanding of the topic, see Khan Academy's Linear Algebra series ([https://www.khanacademy.org/math/linear-algebra](https://www.khanacademy.org/math/linear-algebra)).

When working in 2D, you'll be using Cartesian coordinates to identify locations in space. A particular position in 2D space is written as a pair of values, such as `(4,3)`, representing the position along the *x* and *y* axes, respectively. Any position in the 2D plane can be described in this way.

In 2D space, Godot follows the common computer graphics practice of orienting the *x* axis to the right, and the *y *axis down:

![](img/00018.jpeg)

If you're new to computer graphics or game development, it might seem odd that the positive y axis points downwards instead of upwards, as you likely learned in math class. However, this orientation is very common in computer graphics applications.

# Vectors

You can also think of the position `(4, 3)` as an *offset* from the `(0, 0)` point, or *origin*. Imagine an arrow pointing from the origin to the point:

![](img/00019.jpeg)

This arrow is a *vector*. It represents a great deal of useful information including the point's location, *(4, 3)*, its length, *m,* and its angle from the *x*-axis, *θ*. Altogether, this is a *position vector*, in other words, it describes a position in space. Vectors can also represent movement, acceleration, or any other quantity that has an *x* and a *y* component.

In Godot, vectors (`Vector2` for 2D or `Vector3` for 3D) are widely used, and you'll use them in the course of building the projects in this book.

# Pixel rendering

Vector coordinates in Godot are *floating point *numbers, not *integers*. This means a `Vector2` could have a fractional value, such as `(1.5, 1.5)`. Since objects can't be drawn at half pixels, this can cause visual problems for pixel art games where you want to ensure that all the pixels of the textures are drawn.

To address this, open Project *|* Project Settings and find the Rendering*/*Quality section in the sidebar and enable Use Pixel Snap, as shown in the following screenshot:

![](img/00020.jpeg)

If you're using 2D pixel art in your game, it's a good idea to always enable this setting when you start your project. This setting has no effect in 3D games.

# Part 1 – Player scene

The first scene you'll make defines the Player object. One of the benefits of creating a separate player scene is that you can test it independently, even before you've created the other parts of the game. This separation of game objects will become more and more helpful as your projects grow in size and complexity. Keeping individual game objects separate from each other makes them easier to troubleshoot, modify, and even replace entirely without affecting other parts of the game. It also makes your player reusable—you can drop the player scene into an entirely different game and it will work just the same.

The player scene will display your character and its animations, respond to user input by moving the character accordingly, and detect collisions with other objects in the game.

# Creating the scene

Start by clicking the Add/Create a New Node button and selecting an `Area2D`. Then, click on its name and change it to `Player`. Click Scene | Save Scene to save the scene. This is the scene's *root* or top-level node. You'll add more functionality to the `Player` by adding children to this node:

![](img/00021.gif)

Before adding any children, it's a good idea to make sure you don't accidentally move or resize them by clicking on them. Select the `Player` node and click the icon next to the lock: 

![](img/00022.jpeg)

The tooltip will say Make sure the object's children are not selectable, as shown in the preceding screenshot.

It's a good idea to always do this when creating a new scene. If a body's collision shape or sprite becomes offset or scaled, it can cause unexpected errors and be difficult to fix. With this option, the node and all of its children will always move together.

# Sprite animation

With `Area2D`, you can detect when other objects overlap or run into the player, but `Area2D` doesn't have an appearance on its own, so click on the `Player` node and add an `AnimatedSprite` node as a child. The `AnimatedSprite` will handle the appearance and animations for your player. Note that there is a warning symbol next to the node. An `AnimatedSprite` requires a `SpriteFrames` resource, which contains the animation(s) it can display. To create one, find the Frame*s* property in the Inspector and click <null> | New SpriteFrames:

![](img/00023.jpeg)

Next, in the same location, click <SpriteFrames> to open the SpriteFrames panel:

![](img/00024.gif)

On the left is a list of animations. Click the default one and rename it to `run`. Then, click the **Add** button and create a second animation named `idle` and a third named `hurt`.

In the FileSystem dock on the left, find the `run`, `idle`, and `hurt` player images and drag them into the corresponding animations:

![](img/00025.jpeg)

Each animation has a default speed setting of 5 frames per second. This is a little too slow, so click on each of the animations and set the Speed (FPS) setting to 8\. In the Inspector, check On next to the Playing property and choose an Animation to see the animations in action:

![](img/00026.jpeg)

Later, you'll write code to select between these animations, depending on what the player is doing. But first, you need to finish setting up the player's nodes.

# Collision shape

When using `Area2D`, or one of the other collision objects in Godot, it needs to have a shape defined, or it can't detect collisions. A collision shape defines the region that the object occupies and is used to detect overlaps and/or collisions. Shapes are defined by `Shape2D`, and include rectangles, circles, polygons, and other types of shapes.

For convenience, when you need to add a shape to an area or physics body, you can add a `CollisionShape2D` as a child. You then select the type of shape you want and you can edit its size in the editor. 

Add a `CollisionShape2D` as a child of `Player` (make sure you don't add it as a child of the `AnimatedSprite`). This will allow you to determine the player's *hitbox*, or the bounds of its collision area. In the Inspector, next to Shape, click <null> and choose New RectangleShape2D. Adjust the shape's size to cover the sprite:

![](img/00027.jpeg)

Be careful not to scale the shape's outline! Only use the size handles (red) to adjust the shape! Collisions will not work properly with a scaled collision shape.

You may have noticed that the collision shape is not centered on the sprite. That is because the sprites themselves are not centered vertically. We can fix this by adding a small offset to the `AnimatedSprite`. Click on the node and look for the Offset property in the Inspector. Set it to `(0, -5)`.

When you're finished, your `Player` scene should look like this:

![](img/00028.jpeg)

# Scripting the Player

Now, you're ready to add a script. Scripts allow you to add additional functionality that isn't provided by the built-in nodes. Click the `Player` node and click the **Add Script** button:

![](img/00029.jpeg)

In the Script Settings window, you can leave the default settings as they are. If you've remembered to save the scene (see the preceding screenshot), the script will automatically be named to match the scene's name. Click Create and you'll be taken to the script window. Your script will contain some default comments and hints. You can remove the comments (lines starting with `#`). Refer to the following code snippet:

```cpp
extends Area2D

# class member variables go here, for example:
# var a = 2
# var b = "textvar"

func _ready():
 # Called every time the node is added to the scene.
 # Initialization here
 pass

#func _process(delta):
# # Called every frame. Delta is time since last frame.
# # Update game logic here.
# pass
```

The first line of every script will describe what type of node it is attached to. Next, you'll define your class variables:

```cpp
extends Area2D

export (int) var speed
var velocity = Vector2()
var screensize = Vector2(480, 720)
```

Using the `export` keyword on the `speed` variable allows you to set its value in the Inspector, as well as letting the Inspector know what type of data the variable should contain. This can be very handy for values that you want to be able to adjust, just like you adjust a node's built-in properties. Click on the **`Player`** node and set the Speed property to 350, as shown in the following screenshot:

![](img/00030.jpeg)

`velocity` will contain the character's current movement speed and direction, and `screensize` will be used to set the limits of the player's movement. Later, the game's main scene will set this variable, but for now you will set it manually so you can test.

# Moving the Player

Next, you'll use the `_process()` function to define what the player will do. The `_process()` function is called on every frame, so you'll use it to update elements of your game that you expect to be changing often. You need the player to do three things:

*   Check for keyboard input
*   Move in the given direction
*   Play the appropriate animation

First, you need to check the inputs. For this game, you have four directional inputs to check (the four arrow keys). Input actions are defined in the project settings under the Input Map tab. In this tab, you can define custom events and assign different keys, mouse actions, or other inputs to them. By default, Godot has events assigned to the keyboard arrows, so you can use them for this project.

You can detect whether an input is pressed using `Input.is_action_pressed()`, which returns `true` if the key is held down and `false` if it is not. Combining the states of all four buttons will give you the resultant direction of movement. For example, if you hold `right` and `down` at the same time, the resulting velocity vector will be `(1, 1)`. In this case, since we’re adding a horizontal and a vertical movement together, the player would move *faster* than if they just moved horizontally.

You can prevent that by *normalizing* the velocity, which means setting its length to **1**, then multiplying it by the desired speed:

```cpp
func get_input():
    velocity = Vector2()
    if Input.is_action_pressed("ui_left"):
        velocity.x -= 1
    if Input.is_action_pressed("ui_right"):
        velocity.x += 1
    if Input.is_action_pressed("ui_up"):
        velocity.y -= 1
    if Input.is_action_pressed("ui_down"):
        velocity.y += 1
    if velocity.length() > 0:
        velocity = velocity.normalized() * speed
```

By grouping all of this code together in a `get_input()` function, you make it easier to change things later. For example, you could decide to change to an analog joystick or other type of controller. Call this function from `_process()` and then change the player's `position` by the resulting `velocity`. To prevent the player from leaving the screen, you can use the `clamp()` function to limit the position to a minimum and maximum value:

```cpp
func _process(delta):
    get_input()

    position += velocity * delta
    position.x = clamp(position.x, 0, screensize.x)
    position.y = clamp(position.y, 0, screensize.y)
```

Click Play the Edited Scene (*F6*) and confirm that you can move the player around the screen in all directions.

# About delta

The `_process()` function includes a parameter called `delta` that is then multiplied by the velocity. What is `delta`?

The game engine attempts to run at a consistent 60 frames per second. However, this can change due to computer slowdowns, either in Godot or from the computer itself. If the frame rate is not consistent, then it will affect the movement of your game objects. For example, consider an object set to move `10` pixels every frame. If everything is running smoothly, this will translate to moving `600` pixels in one second. However, if some of those frames take longer, then there may only have been 50 frames in that second, so the object only moved `500` pixels.

Godot, like most game engines and frameworks, solves this by passing you `delta`, which is the elapsed time since the previous frame. Most of the time, this will be around `0.016` s (or around 16 milliseconds). If you then take your desired speed (`600` px/s) and multiply by delta, you will get a movement of exactly `10`. If, however, the `delta` increased to `0.3`, then the object will be moved `18` pixels. Overall, the movement speed remains consistent and independent of the frame rate.

As a side benefit, you can express your movement in units of px/s rather than px/frame, which is easier to visualize.

# Choosing animations

Now that the player can move, you need to change which animation the `AnimatedSprite` is playing based on whether it is moving or standing still. The art for the `run` animation faces to the right, which means it should be flipped horizontally (using the Flip H property) for movement to the left. Add this to the end of your `_process()` function:

```cpp
    if velocity.length() > 0:
        $AnimatedSprite.animation = "run"
        $AnimatedSprite.flip_h = velocity.x < 0
    else:
        $AnimatedSprite.animation = "idle"
```

Note that this code takes a little shortcut. `flip_h` is a Boolean property, which means it can be `true` or `false`. A Boolean value is also the result of a comparison like `<`. Because of this, we can set the property equal to the result of the comparison. This one line is equivalent to writing it out like this:

```cpp
if velocity.x < 0:
    $AnimatedSprite.flip_h = true
else:
    $AnimatedSprite.flip_h = false     
```

Play the scene again and check that the animations are correct in each case. Make sure Playing is set to On in the `AnimatedSprite` so that the animations will play.

# Starting and Ending the Player's Movement

When the game starts, the main scene will need to inform the player that the game has begun. Add the `start()` function as follows, which the main scene will use to set the player's starting animation and position:

```cpp
func start(pos):
    set_process(true)
    position = pos
    $AnimatedSprite.animation = "idle"
```

The `die()` function will be called when the player hits an obstacle or runs out of time:

```cpp
func die():
    $AnimatedSprite.animation = "hurt"
    set_process(false)
```

Setting `set_process(false)` causes the `_process()` function to no longer be called for this node. That way, when the player has died, they can't still be moved by key input.

# Preparing for collisions

The player should detect when it hits a coin or an obstacle, but you haven't made them do so yet. That's OK, because you can use Godot's *signal* functionality to make it work. Signals are a way for nodes to send out messages that other nodes can detect and react to. Many nodes have built-in signals to alert you when a body collides, for example, or when a button is pressed. You can also define custom signals for your own purposes.

Signals are used by *connecting* them to the node(s) that you want to listen and respond to. This connection can be made in the Inspector or in the code. Later in the project, you'll learn how to connect signals in both ways.

Add the following to the top of the script (after `extends Area2D`):

```cpp
signal pickup
signal hurt
```

These define custom signals that your player will *emit* (send out) when they touch a coin or an obstacle. The touches will be detected by the `Area2D` itself. Select the `Player` node and click the Node tab next to the Inspector to see the list of signals the player can emit:

![](img/00031.jpeg)

Note your custom signals are there as well. Since the other objects will also be `Area2D` nodes, you want the `area_entered()` signal. Select it and click Connect. Click Connect on the Connecting Signal window—you don't need to change any of those settings. Godot will automatically create a new function called `_on_Player_area_entered()` in your script.

When connecting a signal, instead of having Godot create a function for you, you can also give the name of an existing function that you want to link the signal to. Toggle the Make Function switch to Off if you don't want Godot to create the function for you.

Add the following code to this new function:

```cpp
func _on_Player_area_entered( area ):
    if area.is_in_group("coins"):
        area.pickup()
        emit_signal("pickup")
    if area.is_in_group("obstacles"):
        emit_signal("hurt")
        die()
```

When another `Area2D` is detected, it will be passed in to the function (using the `area` variable). The coin object will have a `pickup()` function that defines the coin's behavior when picked up (playing an animation or sound, for example). When you create the coins and obstacles, you'll assign them to the appropriate *group* so they can be detected.

To summarize, here is the complete player script so far:

```cpp
extends Area2D

signal pickup
signal hurt

export (int) var speed
var velocity = Vector2()
var screensize = Vector2(480, 720)

func get_input():
    velocity = Vector2()
    if Input.is_action_pressed("ui_left"):
        velocity.x -= 1
    if Input.is_action_pressed("ui_right"):
        velocity.x += 1
    if Input.is_action_pressed("ui_up"):
        velocity.y -= 1
    if Input.is_action_pressed("ui_down"):
        velocity.y += 1
    if velocity.length() > 0:
        velocity = velocity.normalized() * speed

func _process(delta):
    get_input()
    position += velocity * delta
    position.x = clamp(position.x, 0, screensize.x)
    position.y = clamp(position.y, 0, screensize.y)

    if velocity.length() > 0:
        $AnimatedSprite.animation = "run"
        $AnimatedSprite.flip_h = velocity.x < 0
    else:
        $AnimatedSprite.animation = "idle"

func start(pos):
    set_process(true)
    position = pos
    $AnimatedSprite.animation = "idle"

func die():
    $AnimatedSprite.animation = "hurt"
    set_process(false)

func _on_Player_area_entered( area ):
    if area.is_in_group("coins"):
        area.pickup()
        emit_signal("pickup")
    if area.is_in_group("obstacles"):
        emit_signal("hurt")
        die()
```

# Part 2 – Coin scene

In this part, you'll make the coins for the player to collect. This will be a separate scene describing all of the properties and behavior of a single coin. Once saved, the main scene will load the coin scene and create multiple *instances* (that is, copies) of it. 

# Node setup

Click Scene | New Scene and add the following nodes. Don't forget to set the children to not be selected, like you did with the `Player` scene:

*   `Area2D` (named `Coin`)
*   `AnimatedSprite`
*   `CollisionShape2D`

Make sure to save the scene once you've added the nodes.

Set up the `AnimatedSprite` like you did in the Player scene. This time, you only have one animation: a shine/sparkle effect that makes the coin look less flat and boring. Add all the frames and set the Speed (FPS) to `12`. The images are a little too large, so set the Scale of `AnimatedSprite` to (`0.5`, `0.5`). In the `CollisionShape2D`, use a `CircleShape2D` and size it to cover the coin image. Don't forget: never use the scale handles when sizing a collision shape. The circle shape has a single handle that adjusts the circle's radius.

# Using groups

Groups provide a tagging system for nodes, allowing you to identify similar nodes. A node can belong to any number of groups. You need to ensure that all coins will be in a group called `coins` for the player script to react correctly to touching the coin. Select the `Coin` node and click the Node tab (the same tab where you found the signals) and choose Groups. Type `coins` in the box and click Add, as shown in the following screenshot:

![](img/00032.jpeg)

# Script

Next, add a script to the `Coin` node. If you choose Empty in the Template setting, Godot will create an empty script without any comments or suggestions. The code for the coin's script is much shorter than the code for the player's:

```cpp
extends Area2D

func pickup():
    queue_free()
```

The `pickup()` function is called by the player script and tells the coin what to do when it's been collected. `queue_free()` is Godot's node removal method. It safely removes the node from the tree and deletes it from memory along with all of its children. Later, you'll add a visual effect here, but for now the coin disappearing is good enough.

`queue_free()` doesn't delete the object immediately, but rather adds it to a queue to be deleted at the end of the current frame. This is safer than immediately deleting the node, because other code running in the game may still need the node to exist. By waiting until the end of the frame, Godot can be sure that all code that may access the node has completed and the node can be removed safely.

# Part 3 – Main scene

The `Main` scene is what ties all the pieces of the game together. It will manage the player, the coins, the timer, and the other pieces of the game. 

# Node setup

Create a new scene and add a node named `Main`. To add the player to the scene, click the Instance button and select your saved `Player.tscn`:

![](img/00033.gif)

Now, add the following nodes as children of **`Main`**, naming them as follows:

*   `TextureRect` (named `Background`)—for the background image
*   `Node` (named `CoinContainer`)—to hold all the coins
*   `Position2D` (named `PlayerStart`)—to mark the starting position of the `Player`
*   `Timer` (named `GameTimer`)—to track the time limit

Make sure `Background` is the first child node. Nodes are drawn in the order shown, so the background will be *behind* the player in this case. Add an image to the `Background` node by dragging the `grass.png` image from the `assets` folder into the Texture property. Change the Stretch Mode to Tile and then click Layout | Full Rect to size the frame to the size of the screen, as shown in the following screenshot:

![](img/00034.jpeg)

Set the Position of the `PlayerStart` node to (`240`, `350`).

Your scene layout should look like this:

![](img/00035.jpeg)

# Main script

Add a script to the `Main` node (use the Empty template) and add the following variables:

```cpp
extends Node

export (PackedScene) var Coin
export (int) var playtime

var level
var score
var time_left
var screensize
var playing = false
```

The `Coin` and `Playtime` properties will now appear in the Inspector when you click on `Main`. Drag `Coin.tscn` from the FileSystem panel and drop it in the `Coin` property. Set `Playtime` to `30` (this is the amount of time the game will last). The remaining variables will be used later in the code.

# Initializing

Next, add the `_ready()` function:

```cpp
func _ready():
    randomize()
    screensize = get_viewport().get_visible_rect().size
    $Player.screensize = screensize
    $Player.hide()
```

In GDScript, you can use `$` to refer to a particular node by name. This allows you to find the size of the screen and assign it to the player's `screensize` variable. `hide()` makes the player start out invisible (you'll make them appear when the game actually starts).

In the `$` notation, the node name is relative to the node running the script. For example, `$Node1/Node2` would refer to a node (`Node2`) that is the child of `Node1`, which itself is a child of the currently running script. Godot's autocomplete will suggest node names from the tree as you type. Note that if the node's name contains spaces, you must put quote marks around it, for example, `$"My Node"`.

You must use `randomize()` if you want your sequence of "random" numbers to be different every time you run the scene. Technically speaking, this selects a random *seed* for the random number generator.

# Starting a new game

Next, the `new_game()` function will initialize everything for a new game:

```cpp
func new_game():
    playing = true
    level = 1
    score = 0
    time_left = playtime
    $Player.start($PlayerStart.position)
    $Player.show()
    $GameTimer.start()
    spawn_coins()
```

In addition to setting the variables to their starting values, this function calls the Player's `start()` function to ensure it moves to the proper starting location. The game timer is started, which will count down the remaining time in the game.

You also need a function that will create a number of coins based on the current level:

```cpp
func spawn_coins():
    for i in range(4 + level):
        var c = Coin.instance()
        $CoinContainer.add_child(c)
        c.screensize = screensize
        c.position = Vector2(rand_range(0, screensize.x),
        rand_range(0, screensize.y))
```

In this function, you create a number of *instances* of the `Coin` object (in code this time, rather than by clicking the Instance a Scene button), and add it as a child of the `CoinContainer`. Whenever you instance a new node, it must be added to the tree using `add_child()`. Finally, you pick a random location for the coin to appear in. You'll call this function at the start of every level, generating more coins each time.

Eventually, you'll want `new_game()` to be called when the player clicks the start button. For now, to test if everything is working, add `new_game()` to the end of your `_ready()` function and click Play the Project (*F5*). When you are prompted to choose a main scene, choose `Main.tscn`. Now, whenever you play the project, the `Main` scene will be started.

At this point, you should see your player and five coins appear on the screen. When the player touches a coin, it disappears.

# Checking for remaining coins

The main script needs to detect whether the player has picked up all of the coins. Since the coins are all children of `CoinCointainer`, you can use `get_child_count()` on this node to find out how many coins remain. Put this in the `_process()` function so that it will be checked every frame:

```cpp
func _process(delta):
    if playing and $CoinContainer.get_child_count() == 0:
        level += 1
        time_left += 5
        spawn_coins()
```

If no more coins remain, then the player advances to the next level.

# Part 4 – User Interface

The final piece your game needs is a **user interface** (**UI**). This is an interface to display information that the player needs to see during gameplay. In games, this is also referred to as a **Heads-Up Display** (**HUD**), because the information appears as an overlay on top of the game view. You'll also use this scene to display a start button.

The HUD will display the following information:

*   Score
*   Time remaining
*   A message, such as Game Over
*   A start button

# Node setup

Create a new scene and add a `CanvasLayer` node named `HUD`. A `CanvasLayer` node allows you to draw your UI elements on a layer above the rest of the game, so that the information it displays doesn't get covered up by any game elements like the player or the coins.

Godot provides a wide variety of UI elements that may be used to create anything from indicators such as health bars to complex interfaces such as inventories. In fact, the Godot editor that you are using to make this game is built in Godot using these elements. The basic nodes for UI elements are extended from `Control`, and appear with green icons in the node list. To create your UI, you'll use various `Control` nodes to position, format, and display information. Here is what the `HUD` will look like when complete:

![](img/00036.jpeg)

# Anchors and margins

Control nodes have a position and size, but they also have properties called **anchors** and **margins**. Anchors define the origin, or the reference point, for the edges of the node, relative to the parent container. Margins represent the distance from the control node's edge to its corresponding anchor. Margins update automatically when you move or resize a control node. 

# Message label

Add a `Label` node to the scene and change its name to `MessageLabel`**.** This label will display the game's title, as well as Game Over when the game ends. This label should be centered on the game screen. You could drag it with the mouse, but to place UI elements precisely, you should use the Anchor properties.

Select View | Show Helpers to display pins that will help you see the anchor positions, then click on the Layout menu and select HCenter Wide:

![](img/00037.jpeg)

The `MessageLabel` now spans the width of the screen and is centered vertically. The Text property in the Inspector sets what text the label displays. Set it to Coin Dash! and set Align and Valign to Center.

The default font for `Label` nodes is very small, so the next step is to assign a custom font. Scroll down to the Custom Fonts section in the Inspector and select New DynamicFont, as shown in the following screenshot:

![](img/00038.jpeg)

Now, click on DynamicFont and you can adjust the font settings. From the FileSystem dock, drag the `Kenney Bold.ttf` font and drop it in the Font Dataproperty. Set Sizeto **`48`**, as shown in the following screenshot:

![](img/00039.jpeg)

# Score and time display

The top of the `HUD` will display the player's score and the time remaining on the clock. Both of these will be `Label` nodes, arranged at opposite sides of the game screen. Rather than position them separately, you'll use a `Container` node to manage their positions.

# Containers

UI containers automatically arrange the positions of their child `Control` nodes (including other `Containers`). You can use them to add padding around elements, center them, or arrange elements in rows or columns. Each type of `Container` has special properties that control how they arrange their children. You can see these properties in the Custom Constants section of the Inspector.

Remember that containers *automatically* arrange their children. If you move or resize a Control that's inside a `Container` node, you'll find it snaps back to its original position. You can manually arrange controls *or* arrange them with a container, but not both.

To manage the score and time labels, add a **`MarginContainer`** node to the **`HUD`.** Use the Layout menu to set the anchors to Top Wide. In the Custom Constants section, set Margin Right, Margin Top, and Margin Left to `10`. This will add some padding so that the text isn't against the edge of the screen.

Since the score and time labels will use the same font settings as the `MessageLabel`, it will save time if you duplicate it. Click on `MessageLabel` and press *Ctrl* + *D* (*Cmd* + *D* on macOS) twice to create two duplicate labels. Drag them both and drop them on the `MarginContainer` to make them its children. Name one `ScoreLabel` and the other `TimeLabel` and set the Text property to `0` for both. Set Align to Left for `ScoreLabel` and Right for **`TimeLabel`.**

# Updating UI via GDScript

Add a script to the `HUD` node. This script will update the UI elements when their properties need to change, updating the score text whenever a coin is collected, for example. Refer to the following code:

```cpp
extends CanvasLayer

signal start_game

func update_score(value):
    $MarginContainer/ScoreLabel.text = str(value)

func update_timer(value):
    $MarginContainer/TimeLabel.txt = str(value)
```

The `Main` scene's script will call these functions to update the display whenever there is a change in value. For the `MessageLabel`, you also need a timer to make it disappear after a brief period. Add a `Timer` node and change its name to `MessageTimer`**. **In the Inspector, set its Wait Time to `2` seconds and check the box to set One Shot to On. This ensures that, when started, the timer will only run once, rather than repeating. Add the following code:

```cpp
func show_message(text):
    $MessageLabel.text = text
    $MessageLabel.show()
    $MessageTimer.start()
```

In this function, you display the message and start the timer. To hide the message, connect the `timeout()` signal of `MessageTimer` and add this:

```cpp
func _on_MessageTimer_timeout():
    $MessageLabel.hide()
```

# Using buttons

Add a `Button` node and change its name to `StartButton`**. **This button will be displayed before the game starts, and when clicked, it will hide itself and send a signal to the `Main` scene to start the game. Set the Text property to Start and change the custom font like you did with the **`MessageLabel`.** In the Layout menu, choose Center Bottom. This will put the button at the very bottom of the screen, so move it up a little bit either by pressing the *Up* arrow key or by editing the margins and setting Top to `-150` and Bottom to `-50`. 

When a button is clicked, a signal is sent out. In the Node tab for the `StartButton`, connect the `pressed()` signal:

```cpp
func _on_StartButton_pressed():
    $StartButton.hide()
    $MessageLabel.hide()
    emit_signal("start_game")
```

The `HUD` emits the `start_game` signal to notify `Main` that it's time to start a new game.

# Game over

The final task for your UI is to react to the game ending:

```cpp
func show_game_over():
    show_message("Game Over")
    yield($MessageTimer, "timeout")
    $StartButton.show()
    $MessageLabel.text = "Coin Dash!"
    $MessageLabel.show()
```

In this function, you need the Game Over message to be displayed for two seconds and then disappear, which is what `show_message()` does. However, you also want to show the start button once the message has disappeared. The `yield()` function pauses execution of the function until the given node (`MessageTimer`) emits a given signal (`timeout`). Once the signal is received, the function continues, returning you to the initial state so that you can play again.

# Adding the HUD to Main

Now, you need to set up the communication between the `Main` scene and the `HUD`. Add an instance of the `HUD` scene to the `Main` scene. In the `Main` scene, connect the `timeout()` signal of `GameTimer` and add the following:

```cpp
func _on_GameTimer_timeout():
    time_left -= 1
    $HUD.update_timer(time_left)
    if time_left <= 0:
        game_over()
```

Every time the `GameTimer` times out (every second), the remaining time is reduced. 
Next, connect the `pickup()` and `hurt()` signals of the `Player`:

```cpp
func _on_Player_pickup():
    score += 1
    $HUD.update_score(score)

func _on_Player_hurt():
    game_over()
```

Several things need to happen when the game ends, so add the following function:

```cpp
func game_over():
    playing = false
    $GameTimer.stop()
    for coin in $CoinContainer.get_children():
        coin.queue_free()
    $HUD.show_game_over()
    $Player.die()
```

This function halts the game, and also loops through the coins and removes any that are remaining, as well as calling the HUD's `show_game_over()` function.

Finally, the `StartButton` needs to activate the `new_game()` function. Click on the `HUD` instance and select its `new_game()` signal. In the signal connection dialog, click Make Function to Off and in the Method In Node field, type `new_game`. This will connect the signal to the existing function rather than creating a new one. Take a look at the following screenshot:

![](img/00040.jpeg)

Remove `new_game()` from the `_ready()` function and add these two lines to the `new_game()` function:

```cpp
$HUD.update_score(score)
$HUD.update_timer(time_left)
```

Now, you can play the game! Confirm that all the parts are working as intended: the score, the countdown, the game ending and restarting, and so on. If you find a piece that's not working, go back and check the step where you created it, as well as the step(s) where it was connected to the rest of the game.

# Part 5 – Finishing up

You have created a working game, but it still could be made to feel a little more exciting. Game developers use the term *juice* to describe the things that make the game feel good to play. Juice can include things like sound, visual effects, or any other addition that adds to the player's enjoyment, without necessarily changing the nature of the gameplay.

In this section, you'll add some small *juicy* features to finish up the game.

# Visual effects

When you pick up the coins, they just disappear, which is not very appealing. Adding a visual effect will make it much more satisfying to collect lots of coins.

Start by adding a `Tween` node to the `Coin` scene.

# What is a tween?

A **tween** is a way to interpolate (change gradually) some value over time (from a start value to an end value) using a particular function. For example, you might choose a function that steadily changes the value or one that starts slow but ramps up in speed. Tweening is also sometimes referred to as *easing*.

When using a `Tween` node in Godot, you can assign it to alter one or more properties of a node. In this case, you're going to increase the `Scale` of the coin and also cause it to fade out using the Modulate property.

Add this line to the `_ready()` function of `Coin`:

```cpp
$Tween.interpolate_property($AnimatedSprite, 'scale',
                            $AnimatedSprite.scale,
                            $AnimatedSprite.scale * 3, 0.3,
                            Tween.TRANS_QUAD,
                            Tween.EASE_IN_OUT)
```

The `interpolate_property()` function causes the `Tween` to change a node's property. There are seven parameters:

*   The node to affect
*   The property to alter
*   The property's starting value
*   The property's ending value
*   The duration (in seconds)
*   The function to use
*   The direction

The tween should start playing when the player picks up the coin. Replace `queue_free()` in the `pickup()` function:

```cpp
func pickup():
    monitoring = false
    $Tween.start() 
```

Setting `monitoring` to `false` ensures that the `area_enter()` signal won't be emitted if the player touches the coin during the tween animation.

Finally, the coin should be deleted when the animation finishes, so connect the `Tween` node's `tween_completed()` signal:

```cpp
func _on_Tween_tween_completed(object, key):
    queue_free()
```

Now, when you run the game, you should see the coins growing larger when they're picked up. This is good, but tweens are even more effective when applied to multiple properties at once. You can add another `interpolate_property()`, this time to change the sprite's opacity. This is done by altering the `modulate` property, which is a `Color` object, and changing its alpha channel from `1` (opaque) to `0` (transparent). Refer to the following code:

```cpp
$Tween.interpolate_property($AnimatedSprite, 'modulate', 
                            Color(1, 1, 1, 1),
                            Color(1, 1, 1, 0), 0.3,
                            Tween.TRANS_QUAD,
                            Tween.EASE_IN_OUT)
```

# Sound

Sound is one of the most important but often neglected pieces of game design. Good sound design can add a huge amount of juice to your game for a very small amount of effort. Sounds can give the player feedback, connect them emotionally to the characters, or even be a part of the gameplay.

For this game, you're going to add three sound effects. In the `Main` scene, add three `AudioStreamPlayer` nodes and name them `CoinSound`, `LevelSound`, and `EndSound`. Drag each sound from the `audio` folder (you can find it under `assets` in the FileSystem dock) into the corresponding Stream property of each node.

To play a sound, you call the `play()` function on it. Add `$CoinSound.play()` to the `_on_Player_pickup()` function, `$EndSound.play()` to the `game_over()` function, and `$LevelSound.play()` to the `spawn_coins()` function.

# Powerups

There are many possibilities for objects that give the player a small advantage or powerup. In this section, you'll add a powerup item that gives the player a small time bonus when collected. It will appear occasionally for a short time, then disappear.

The new scene will be very similar to the `Coin` scene you already created, so click on your `Coin` scene and choose Scene | Save Scene As and save it as `Powerup.tscn`. Change the name of the root node to Powerup and remove the script by clicking the clear script button: ![](img/00041.jpeg). You should also disconnect the `area_entered` signal (you'll reconnect it later). In the Groups tab, remove the coins group by clicking the delete button (it looks like a trash can) and adding it to a new group called `powerups` instead.

In the `AnimatedSprite`, change the images from the coin to the powerup, which you can find in the `res://assets/pow/` folder.

Click to add a new script and copy the code from the `Coin.gd` script. Change the name of `_on_Coin_area_entered` to `_on_Powerup_area_entered` and connect the `area_entered` signal to it again. Remember, this function name will automatically be chosen by the signal connect window.

Next, add a `Timer` node named `Lifetime`. This will limit the amount of time the object remains on the screen. Set its Wait Time to `2` and both One Shot and Autostart to `On`. Connect its timeout signal so that it can be removed at the end of the time period:

```cpp
func _on_Lifetime_timeout():
    queue_free()
```

Now, go to your Main scene and add another `Timer` node called `PowerupTimer`. Set its One Shot property to On. There is also a `Powerup.wav` sound in the `audio` folder you can add with another `AudioStreamPlayer`.

Connect the `timeout` signal and add the following code to spawn a `Powerup`:

```cpp
func _on_PowerupTimer_timeout():
    var p = Powerup.instance()
    add_child(p)
    p.screensize = screensize
    p.position = Vector2(rand_range(0, screensize.x),
                         rand_range(0, screensize.y))
```

The `Powerup` scene needs to be linked by adding a variable, then dragging the scene into the property in the Inspector, as you did earlier with the `Coin` scene:

```cpp
export (PackedScene) var Powerup
```

The powerups should appear unpredictably, so the wait time of the `PowerupTimer` needs to be set whenever you begin a new level. Add this to the `_process()` function after the new coins are spawned with `spawn_coins()`:

```cpp
$PowerupTimer.wait_time = rand_range(5, 10)
$PowerupTimer.start()
```

Now that you will have powerups appearing, the last step is to give the player some bonus time when one is collected. Currently, the player script assumes anything it runs into is either a coin or an obstacle. Change the code in `Player.gd` to check for what kind of object has been hit:

```cpp
func _on_Player_area_entered( area ):
    if area.is_in_group("coins"):
        area.pickup()
        emit_signal("pickup", "coin")
    if area.is_in_group("powerups"):
        area.pickup()
        emit_signal("pickup", "powerup")
    if area.is_in_group("obstacles"):
        emit_signal("hurt")
        die()
```

Note that now you're emitting the pickup signal with an additional argument naming the type of object. The corresponding function in `Main.gd` can now be changed to accept that argument and use the `match` statement to decide what action to take:

```cpp
func _on_Player_pickup(type):
    match type:
        "coin":
            score += 1
            $CoinSound.play()
            $HUD.update_score(score)
        "powerup":
            time_left += 5
            $PowerupSound.play()
            $HUD.update_timer(time_left)
```

The `match` statement is a useful alternative to `if` statements, especially when you have a large number of possible values to test.

Try running the game and collecting the powerup. Make sure the sound plays and the timer increases by five seconds.

# Coin animation

When you created the `Coin` scene, you added an `AnimatedSprite`, but it isn't playing yet. The coin animation displays a *shimmer* effect traveling across the face of the coin. If all the coins display this at the same time, it will look too regular, so each coin needs a small random delay in its animation.

First, click on the `AnimatedSprite` and then on the *Frames* resource. Make sure Loop is set to Off and that Speed is set to `12`.

Add a `Timer` node to the `Coin` scene, and add this code to `_ready()`:

```cpp
$Timer.wait_time = rand_range(3, 8)
$Timer.start()
```

Now, connect the `timeout()` signal from the `Timer` and add this:

```cpp
func _on_Timer_timeout():
    $AnimatedSprite.frame = 0
    $AnimatedSprite.play()
```

Try running the game and watching for the coins to animate. It's a nice visual effect for a very small amount of effort. You'll notice a lot of effects like this in professional games. Though very subtle, the visual appeal makes for a much more pleasing experience.

The preceding `Powerup` object has a similar animation that you can add in the same manner.

# Obstacles

Finally, the game can be made a bit more challenging by introducing an obstacle that the player must avoid. Touching the obstacle will end the game.

Create a new scene for the cactus and add the following nodes:

*   `Area2D` (named `Cactus`)
*   `Sprite`
*   `CollisionShape2D`

Drag the cactus texture from the FileSystem dock to the Texture property of the `Sprite`. Add a `RectangleShape2D` to the collision shape and size it so that it covers the image. Remember when you added `if area.is_in_group("obstacles")` to the player script? Add the `Cactus` body to the `obstacles` group using the Node tab (next to Inspector).

Now, add a `Cactus` instance to the `Main` scene and move it to a location in the upper half of the screen (away from where the player spawns). Play the game and see what happens when you run into the cactus.

You may have spotted a problem: coins can spawn behind the cactus, making them impossible to pick up. When the coin is placed, it needs to move if it detects that it's overlapping the obstacle. Connect the coin's `area_entered()` signal and add the following:

```cpp
func _on_Coin_area_entered( area ):
    if area.is_in_group("obstacles"):
        position = Vector2(rand_range(0, screensize.x), rand_range(0, screensize.y))
```

If you've added the preceding `Powerup` object, you'll need to do the same for its `area_entered` signal.

# Summary

In this chapter, you learned the basics of Godot Engine by creating a basic 2D game. You set up the project and created multiple scenes, worked with sprites and animations, captured user input, used *signals* to communicate with events, and created a UI using **Control** nodes. The things you learned here are important skills that you'll use in any Godot project.

Before moving on to the next chapter, look through the project. Do you understand what each node is doing? Are there any bits of code that you don't understand? If so, go back and review that section of the chapter. 

Also, feel free to experiment with the game and change things around. One of the best ways to get a good feel for what different parts of the game are doing is to change them and see what happens.

In the next chapter, you'll explore more of Godot's features and learn how to use more node types by building a more complex game.