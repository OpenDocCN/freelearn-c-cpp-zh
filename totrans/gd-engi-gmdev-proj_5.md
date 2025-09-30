# Jungle Jump (Platformer)

In this chapter, you'll build a classic *platform*, style game in the tradition of *Super Mario Bros.* Platform games are a very popular genre, and understanding how they work can help you make a variety of different game styles. The physics of platformers can be deceptively complex, and you'll see how Godot's `KinematicBody2D` physics node has features to help you implement the character controller features you need for a satisfying experience. Take a look at the following screenshot:

![](img/fe628371-e6f4-4732-a422-8822890627da.png)

In this project, you will learn about:

*   Using the `KinematicBody2D` physics node
*   Combining animations and user input to produce complex character behavior
*   Creating an infinitely scrolling background using ParallaxLayers
*   Organizing your project and planning for expansion

# Project setup

Create a new project. Before you download the assets from the link that follows, you need to prepare the import settings for the game art. The art assets for this project use a *pixel art* style, which means they look best when not filtered, which is Godot's default setting for textures. **Filtering** is a method by which the pixels of an image are smoothed. It can improve the look of some art, but not pixel-based images:

![](img/2e2488db-691b-43fb-ad11-aef65d2434e0.png)

It's inconvenient to have to disable this for every image, so Godot allows you to customize the default import settings. Click on the `icon.png` file in the FileSystem dock, then click the Import tab next to the Scene tab on the right. This window allows you to change the import settings for the file you've selected. Uncheck the Filter property, then click Preset and choose Set as Default for 'Texture'. This way, all images will be imported with filtering disabled. Refer to the following screenshot:

![](img/7e223219-3c2a-40c8-972f-0641e3406563.png)

If you've already imported images, their import settings won't be updated automatically. After changing the default, you'll have to reimport any existing images. You can select multiple files in the FileSystem dock and click the Reimport button to apply the settings to many files at once.

Now, you can download the game assets from the following link and unzip them in your project folder. Godot will import all the images with the new default settings, [https://github.com/PacktPublishing/Godot-Game-Engine-Projects/releases](https://github.com/PacktPublishing/Godot-Game-Engine-Projects/releases)

Next, open Project | Project Settingsand under Rendering/Quality, set Use Pixel Snap to `On`. This will ensure that all images will be aligned properly—something that will be very important when you're designing your game's levels.

While you have the settings window open, go to the Display/Window section and change Stretch/Mode to `2d` and Aspect to `expand`. These settings will allow the user to resize the game window while preserving the image's quality. Once the project has been completed, you'll be able to see the effects of this setting.

Next, set up the collision layer names so that it will be more convenient to set up collisions between different types of objects. Go to Layer Names/2d Physics and name the first four layers like this:

![](img/4a6614f2-4f23-466f-a8bb-49b4d6fbe579.png)

Finally, add the following actions for the player controls in the Input Map tab under Project | Project Settings:

| **Action Name** | **Key(s)** |
| right | D, → |
| left | A, ← |
| jump | Space |
| crouch | S, ↓ |
| climb | W, ↑ |

# Introducing kinematic bodies

A platform game requires gravity, collisions, jumping, and other physics behavior, so you might think that `RigidBody2D` would be the perfect choice to implement the character's movement. In practice, however, you'll find that the *realistic* physics of the rigid body are not desirable for a platform character. To the player, realism is less important than responsive control and an *action* feel. As the developer, you therefore want to have precise control over the character's movements and collision response. For this reason, a kinematic body is usually the better choice for a platform character.

The `KinematicBody2D` node is designed for implementing bodies that are to be controlled directly by the user or via code. These nodes detect collisions with other bodies when moving, but are not affected by global physics properties like gravity or friction. This doesn't mean that a kinematic body can't be affected by gravity and other forces, just that you must calculate those forces and their effects in code; the engine will not move a kinematic body automatically.

When moving `KinematicBody2D`, as with `RigidBody2D`, you should not set its `position` directly. Instead, you use either the `move_and_collide()` or `move_and_slide()` methods. These methods move the body along a given vector and instantly stop if a collision is detected with another body. After `KinematicBody2D` has collided, any *collision response* must be coded manually.

# Collision response

After a collision, you may want the body to bounce, to slide along a wall, or to alter the properties of the object it hit. The way you handle collision response depends on which method you used to move the body.

# move_and_collide

When using `move_and_collide()`, the function returns a `KinematicCollision2D` object upon collision. This object contains information about the collision and the colliding body. You can use this information to determine the response. Note that the function returns `null` when the movement was completed successfully with no collision.

For example, if you want the body to bounce off of the colliding object, you could use the following script:

```cpp
extends KinematicBody2D

var velocity = Vector2(250, 250)

func _physics_process(delta):
    var collide = move_and_collide(velocity * delta)
    if collide:
        velocity = velocity.bounce(collide.normal)
```

# move_and_slide

Sliding is a very common option for collision response. Imagine a player moving along walls in a top-down game or running up and down slopes in a platformer. While it's possible to code this response yourself after using `move_and_collide()`, `move_and_slide()` provides a convenient way to implement sliding movement. When using this method, the body will automatically slide along the colliding surface. In addition, sliding collisions allow you to use methods like `is_on_floor()` to detect the orientation of the colliding surface.

Since this project will require not just moving along the ground, but also running up and down slopes, `move_and_slide()` is going to play a large role in your player's movement. You'll see how it works as you build up the player object.

# Player scene

Open a new scene and add a `KinematicBody2D` object named `Player` as the root and save the scene (don't forget to click the Make children unselectable button). When saving the `Player` scene, you should also create a new folder to contain it. This will help keep your project folder organized as you add more scenes and scripts.

As you've done in other projects, you'll include all the nodes that the player character needs to function in the `Player` scene. For this game, that means handling collisions with various game objects, including platforms, enemies, and collectibles; displaying animations for actions, such as running or jumping; and a camera to follow the player around the level.

Scripting the various animations can quickly become unmanageable, so you'll use a *finite state machine* to manage and track the player's state. See [Chapter 3](f24a8958-bb32-413a-97ae-12c9e7001c2c.xhtml), *Escape the Maze*, to review how the simplified FSM was built. You'll follow a similar pattern for this project.

# Collision Layer/Mask

A body's collision layer property sets what layer(s) the body is found on. `Player` needs to be assigned to the player layer you named in Project Settings.

The Collision/Mask property allows you to set what types of objects the body will detect. Set the Player layer to `player` and its mask to environment, enemies, and collectibles (`1`, `3`, and `4`):

![](img/b5cc9f1d-67d7-4e60-8e64-0749e8feaae9.png)

# Sprite

Add a Sprite node to `Player`. Drag the `res://assets/player_sheet.png` file from the FileSystem dock and drop it in the Texture property of the `Sprite`. The player animation is saved in the form of a sprite sheet:

![](img/91acf832-e5d4-4420-9dea-627f6b0aae6f.png)

You'll use `AnimationPlayer` to handle the animations, so in the Animation properties of `Sprite`, set Vframes to `1` and Hframes to `19`. Set Frame to `7` to begin, as this is the frame that shows the character standing still (it's the first frame of the `idle` animation):

![](img/0c780e74-c4b2-4960-851a-3ddc565a24d5.png)

# Collision shape

As with other physics bodies, `KinematicBody2D` needs a shape assigned to define its collision bounds. Add a `CollisionShape2D` object and create a new `RectangleShape2D` object inside it. When sizing the rectangle, you want it to reach the bottom of the image but not be quite as wide. In general, making the collision shape a bit smaller than the image will result in a better *feel* when playing, avoiding the experience of hitting something that looks like it wouldn't result in a collision.

You'll also need to offset the shape a small amount to make it fit. Setting Position to `(0, 5)` works well. When you're done, it should look approximately like this:

![](img/a6494b7a-3f21-4d56-a093-dae24dda9edd.png)

# Shapes

Some developers prefer a capsule shape over a rectangle shape for sidescrolling characters. A capsule is a pill-shaped collision that's rounded on both ends:

![](img/2f968bf9-92a8-468a-9309-4082c72de89e.png)

However, while this shape might seem to *cover* the sprite better, it can lead to difficulties when implementing platformer-style movement. For example, when standing too near the edge of a platform, the character may slide off due to the rounded bottom, which can be very frustrating for the player.

In some cases, depending on the complexity of your character and its interactions with other objects, you may want to add multiple shapes to the same object. You might have one shape at the character's feet to detect ground collisions, another on its body to detect damage (sometimes called a hurtbox), and yet another covering the player's front to detect contact with walls.

It's recommended that you stick to `RectangleShape2D`, as shown in the preceding screenshot, for this character. However, once you've finished the project, you should try changing the player's collision shape to `CapsuleShape2D` and observing the resulting behavior. If you like it better, feel free to use it instead.

# Animations

Add an `AnimationPlayer` node to the `Player` scene. You'll use this node to change the Frame property on `Sprite` to display the character's animations. Start by making a new animation named `idle`:

![](img/23491cf3-4dc5-40f9-a9d2-c933650555c1.png)

Set Length to `0.4` seconds and keep Step at `0.1` seconds. Change the Frame of `Sprite` to `7` and click the Add keyframe button next to the Frame property to create a new animation track, then press it again, noting that it automatically increments the Frame property:

![](img/ada536c1-e847-4e17-9661-cce98de7c9c8.png)

Continue pressing it until you have frames `7` through `10`. Finally, click the Enable/Disable looping button to enable looping and then press Play to view your animation. Your animation setup should look like this:

![](img/22e338b2-66da-43ea-8436-070672dcf5c0.png)

Now you need to repeat the process for the other animations. See the following table for a list of settings:

| **name** | **length** | **frames** | **looping** |
| `idle` | `0.4` | `7, 8, 9 ,10` | on |
| `run` | `0.5` | `13, 14, 15, 16, 17, 18` | on |
| `hurt` | `0.2` | `5, 6` | on |
| `jump_up` | `0.1` | `11` | off |
| `jump_down` | `0.1` | `12` | 0ff |

# Finishing up the scene tree

Add `Camera2D` to the `Player` scene. This node will keep the game window centered on the player as it moves around the level. You can also use it to zoom in on the player, since the pixel art is relatively small. Remember, since you set filtering off in the import settings, the player's texture will remain pixelated and blocky when zoomed in. 

To enable the camera, click the Current property to `On`, then set the Zoom property to `(0.4, 0.4)`. Values smaller than one zoom the camera in, while larger values zoom it out. 

# Player states

The player character has a wide variety of behaviors, such as jumping, running, and crouching. Coding such behaviors can become very complex and hard to manage. One solution is to use Boolean variables (`is_jumping` or `is_running`, for example), but this leads to possibly confusing states (what if `is_crouching` and `is_jumping` are both `true`?) and quickly leads to spaghetti code.

A better solution to this problem is to use a state machine to handle the player's current state and control the transitions to other states. Finite state machines were discussed in [Chapter 3](f24a8958-bb32-413a-97ae-12c9e7001c2c.xhtml), *Escape the Maze*. 

Here is a diagram of the player's states and the transitions between them:

![](img/56fdfa26-eb80-44d2-9668-008c1eb70178.png)

As you can see, state machine diagrams can become quite complex, even with a relatively small number of states.

Note that while the spritesheet contains animations for them, the CROUCH and CLIMB animations are not included here. This is to keep the number of states manageable at the beginning of the project. Later, you'll have the opportunity to add them to the player's state machine.

# Player script

Attach a new script to the `Player` node. Add the following code to create the player's state machine:

```cpp
extends KinematicBody2D
enum {IDLE, RUN, JUMP, HURT, DEAD}
var state
var anim
var new_anim

func ready():
    change_state(IDLE)

func change_state(new_state):
    state = new_state
    match state:
        IDLE:
            new_anim = 'idle'
        RUN:
            new_anim = 'run'
        HURT:
            new_anim = 'hurt'
        JUMP:
            new_anim = 'jump_up'
        DEAD:
            hide()

func _physics_process(delta):
    if new_anim != anim:
        anim = new_anim
        $AnimationPlayer.play(anim)
```

Once again, you're using `enum` to list the allowed states for the system. When you want to change the player's state, you'll call `change_state()`, for example: `change_state(IDLE)`. For now, the script only changes the animation value, but you'll add more state functionality later.

You may be asking, *why not just play the animation when the state changes? Why this new_anim business?* This is because when you call `play()` on `AnimationPlayer`, it starts the animation from the beginning. If you did that while running, for example, you'd only see the first frame of the run animation as it restarted every frame. By using the `new_anim` variable, you can let the current animation continue to play smoothly until you want it to change.

# Player movement

The player needs three controls—left, right, and jump. The combination of the current state plus which keys are pressed will trigger a state change if the transition is allowed by the state rules. Add the `get_input()` function to process the inputs and determine the result:

```cpp
extends KinematicBody2D

export (int) var run_speed
export (int) var jump_speed
export (int) var gravity

enum {IDLE, RUN, JUMP, HURT, DEAD}
var state
var anim
var new_anim
var velocity = Vector2()

func get_input():
    if state == HURT:
        return # don't allow movement during hurt state
    var right = Input.is_action_pressed('right')
    var left = Input.is_action_pressed('left')
    var jump = Input.is_action_just_pressed('jump')

    # movement occurs in all states
    velocity.x = 0
    if right:
        velocity.x += run_speed
        $Sprite.flip_h = false
    if left:
        velocity.x -= run_speed
        $Sprite.flip_h = true
    # only allow jumping when on the ground
    if jump and is_on_floor():
        change_state(JUMP)
        velocity.y = jump_speed
    # IDLE transitions to RUN when moving
    if state == IDLE and velocity.x != 0:
        change_state(RUN)
    # RUN transitions to IDLE when standing still
    if state == RUN and velocity.x == 0:
        change_state(IDLE)
    # transition to JUMP when falling off an edge
    if state in [IDLE, RUN] and !is_on_floor():
        change_state(JUMP)
```

Note that the jump check is using `is_action_just_pressed()` rather than `is_action_pressed()`. While the latter always returns `true` as long as the key is held down, the former is only `true` in the frame after the key was pressed. This means that the player must press the jump key each time they want to jump.

Now, call this function from `_physics_process()`, add the pull of gravity to the player's `velocity`, and call the `move_and_slide()` method to move the body:

```cpp
func _physics_process(delta):
    velocity.y += gravity * delta
    get_input()
    if new_anim != anim:
        anim = new_anim
        $AnimationPlayer.play(anim)
    # move the player
    velocity = move_and_slide(velocity, Vector2(0, -1))
```

The second parameter of `move_and_slide()` is a *normal* vector, indicating what surface direction the engine should consider to be the ground. In physics and geometry, a *normal* is a vector perpendicular to a surface, defining the direction a surface is facing. Using `(0, -1)`, which is a vector pointing upwards, the top of a horizontal surface will be considered as ground. Refer to the following screenshot:

![](img/abf9056e-21c3-4fc9-adac-766f03248aca.png)

After moving with `move_and_slide()`, the physics engine will use this information to set the value of the `is_on_floor()`, `is_on_wall()` and `is_on_ceiling` methods. You can use this fact to detect when the jump ends by adding this after the move:

```cpp
    if state == JUMP and is_on_floor():
        change_state(IDLE)
```

Finally, the jump will look better if the animation switches from `jump_up` to `jump_down` when falling:

```cpp
    if state == JUMP and velocity.y > 0:        new_anim = 'jump_down'Testing the moves
```

At this point, it would be a good idea to test out the movement and make sure everything is working. You can't just run the player scene though, because the player will just start falling without a surface to stand on.

Create a new scene and add a `Node` called `Main` (later, this will become your real main scene). Add an instance of the `Player`, then add a `StaticBody2D` with a rectangular `CollisionShape2D`. Stretch the collision shape horizontally so that it's wide enough to walk back and forth on (like a platform) and place it below the character:

![](img/2f2a6248-e308-4ff1-ad50-d7c4b91ce010.png)

Press Play Scene and you should see the player stop falling and run the `idle` animation when it hits the static body.

Before moving on, make sure that all the movement and animations are working correctly. Run and jump in all directions and check that the correct animations are playing whenever the state changes. If you find any problems, review the previous sections and make sure you didn't miss a step.

Later, once the level is complete, the player will be passed a spawn location. To handle this, add this function to the `Player.gd` script:

```cpp
func start(pos):
    position = pos
    show()
    change_state(IDLE)
```

# Player health

Eventually, the player is going to encounter danger, so you should add a damage system. The player will start with three *hearts* and lose one each time they are damaged. 

Add the following to the top of the script:

```cpp
signal life_changed
signal dead

var life
```

The `life_changed` signal will be emitted whenever the value of `life` changes, notifying the display to update. `dead` will be emitted when `life` reaches `0`. Add these two lines to the `start()` function:

```cpp
    life = 3
    emit_signal('life_changed', life)
```

There are two possible ways for the player to be hurt: running into a *spike* object in the environment, or being hit by an enemy. In either event, the following function can be called:

```cpp
func hurt():
    if state != HURT:
        change_state(HURT)
```

This is being nice to the player: if they're already hurt, they can't get hurt again (at least for the brief time until the *hurt* animation has stopped playing).

There are several things to do when the state changes to `HURT` in `change_state()`:

```cpp
HURT:
    new_anim = 'hurt'
    velocity.y = -200
    velocity.x = -100 * sign(velocity.x)
    life -= 1
    emit_signal('life_changed', life)
    yield(get_tree().create_timer(0.5), 'timeout')
    change_state(IDLE)
    if life <= 0:
        change_state(DEAD)
DEAD:
    emit_signal('dead')
    hide()
```

Not only does does the player lose a life, but they are also bounced up and away from the damaging object. After a short time, the state changes back to `IDLE`.

Also, input will be disabled while the player is in the `HURT` state. Add this to the beginning of `get_input()`:

```cpp
if state == HURT:
    return
```

Now, the player is ready to take damage once the rest of the game is set up.

# Collectible items

Before you start making the level, you need to create some pickups for the player to collect, since those will be part of the level as well. The `assets/sprites` folder contains sprite sheets for two types of collectibles: cherries and gems.

Rather than make separate scenes for each type of item, you can use a single scene and merely swap out the sprite sheet texture. Both objects will have the same behavior: animating in place and disappearing (that is, being collected) when contacted by the player. You can also add a `Tween` animation for the pickup (see [Chapter 1](fee8a22d-c169-454d-be5e-cf6c0bc78ddb.xhtml), *Introduction*, for an example).

# Collectible scene

Start the new scene with an `Area2D` and name it `Collectible`. An area is a good choice for these objects because you want to detect when the player contacts them (using the `body_entered` signal), but you don't need collision response from them. In the Inspector, set the Collision/Layer to collectibles (layer 4) and the Collision/Mask to player (layer 2). This will ensure that only the `Player` node will be able to collect an item while the enemies will pass right through.

Add three child nodes: `Sprite`, `CollisionShape2D`, and `AnimationPlayer`, then drag the `res://assets/cherry.png` Sprite sheet into the Sprite's Texture. Set the Vframes to `1` and Hframes to `5`. Add a rectangle shape to `CollisionShape2D` and size it appropriately.

As a general rule, you should size your objects' collision shapes so that they benefit the player. This means that enemy hitboxes should generally be a little smaller than the image while the hitboxes of beneficial items should be slightly oversized. This reduces player frustration and results in a better gameplay experience.

Add a new animation to `AnimationPlayer` (you only need one, so you can just name it `anim`). Set the Length to `1.6` seconds and the Step to `0.2` seconds.

Set the Sprite's Frame property to `0` and click the keyframe button to create the track. When you reach frame number four, start reversing the order back down to `1`. The full sequence of keyframes should be:

```cpp
0 → 1 → 2 → 3 → 4 → 3 → 2 → 1 
```

Enable looping and press the Play button. Now, you have a nicely animated cherry! Drag `res://assets/gem.png` into the texture and check that it animates as well. Finally, click the Autoplay on Load button to ensure the animation will play automatically when the scene begins. Refer to the following screenshot:

![](img/000f2f26-a3f7-44ee-854a-3f2b6b2e6391.png)

# Collectible script

The Collectible's script needs to do two things:

*   Set the start conditions (`texture` and `position`)
*   Detect when the player enters the area

For the first part, add the following code to the new script:

```cpp
extends Area2D

signal pickup

var textures = {'cherry': 'res://assets/sprites/cherry.png',
                'gem': 'res://assets/sprites/gem.png'}

func init(type, pos):
    $Sprite.texture = load(textures[type])
    position = pos
```

The `pickup` signal will be emitted when the player collects the item. In the `textures` dictionary, you have a list of the item types and their corresponding texture locations. Note that you can quickly paste those file paths by right-clicking on the file in the FileSystem dock and choosing Copy Path:

![](img/89e5faff-b388-4b81-88e3-a6c2e26ae3e4.png)

Next, you have an `init()` function that sets the `texture` and `position` to the given values. The level script will use this function to spawn all the collectibles that you add to your level map.

Finally, you need the object to detect when it's been picked up. Click on the `Area2D` and connect its `body_entered` signal. Add the following code to the created function:

```cpp
func _on_Collectible_body_entered(body):
    emit_signal('pickup')
    queue_free()
```

Emitting the signal will allow the game's script to react appropriately to the item pickup. It can add to the score, increase the player's speed, or whatever other effect you want the item to apply.

# Designing the level

It wouldn't be a platformer without jumps. For most readers, this section will take up the largest chunk of time. Once you start designing a level, you'll find it's a lot of fun to lay out all the pieces, creating challenging jumps, secret paths, and dangerous encounters.

First, you'll create a generic `Level` scene containing all the nodes and code that is common to all levels. You can then create any number of level scenes that inherit from this master level.

# TileSet configuration

In the `assets` folder you downloaded at the beginning of the project is a `tilesets` folder. It contains three ready-made `TileSet` resources using the 16x16 art for the game:

*   `tiles_world.tres`: Ground and platform tiles
*   `tiles_items.tres`: Decorative items, foreground objects, and collectibles
*   `tiles_spikes.tres`: Danger items

It is recommended that you use these tile sets to create the levels for this project. However, if you would rather make them yourself, the original art is in `res://assets/environment/layers`. See [Chapter 2](a56e3c2d-5d7f-41d6-98c4-c1d95e17fc31.xhtml), *Coin Dash*, to review how to create a `TileSet` resource.

# Base-level setup

Create a new scene and add a `Node2D` named `Level`. Save the scene in a new folder called `levels`. This is where you'll save any other levels you create, after inheriting from `Level.tscn`. The node hierarchy will be the same for all levels—only the layout will be different.

Next, add a `TileMap` and set its Cell/Size to `(16, 16)`, then duplicate it three times (press *Ctrl* + *D* to duplicate a node). These will be the layers of your level, holding different tiles and information about the layout. Name the four `TileMap` instances as follows and drag-and-drop the corresponding `TileSet` into the Tile Set property of each. Refer to the following table:

| **TileMap** | **Tile Set** |
| `World` | `tiles_world.tres` |
| `Objects` | `tiles_items.tres` |
| `Pickups` | `tiles_items.tres` |
| `Danger` | `tiles_spikes.tres` |

It's a good idea to press the Lock button on your `TileMap` nodes to prevent accidentally moving them while you're working on your map.

Next, add an instance of the `Player` scene and a `Position2D` named `PlayerSpawn`. Click the hide button on the `Player`—you'll use `show()` in the level script to make the player appear when it starts. Your scene tree should now look like this:

![](img/b630e32e-b24c-4bba-b472-2cbb11d4b4ce.png)

Attach a script to the `Level` node:

```cpp
extends Node2D

onready var pickups = $Pickups

func _ready():
    pickups.hide()
    $Player.start($PlayerSpawn.position)
```

Later, you'll be scanning the `Pickups` map to spawn collectible items in the designated locations. This map layer itself shouldn't be seen, but rather than set it as hidden in the scene tree, which is easy to forget before you run the game, you can make sure it's always hidden during gameplay by doing so in `_ready()`. Because there will be many references to the node, storing the result of `$Pickups` in the `pickups` variable will cache the result. (Remember, `$NodeName` is the same as writing `get_node("NodeName")`.)

# Designing the first level

Now, you're ready to start drawing the level! Click Scene | New Inherited Scene and choose `Level.tscn`. Name the new node `Level01` and save it (still in the `levels` folder).

Start with the `World` map and be creative. Do you like lots of jumps, or twisty tunnels to explore? Long runs, or careful upward climbs?

Before going too far in your design, experiment with jump distance. You can change the Player's `jump_speed`, `run_speed`, and `gravity` properties to alter how high and how far they can jump. Set up some different gap sizes and run the scene to try them out. Don't forget to drag the `PlayerSpawn` node to the place you want the character to start.

For example, can the player make this jump? Take a look at the following screenshot:

![](img/a9e809ba-1ebf-4170-aef8-eb8e7de47bdc.png)

How you set the player's movement properties will have a big impact on how your level should be laid out. Make sure you're happy with your settings before spending too much time on the full design.

Once you have the `World` layer set up, use the `Objects` layer to place decorations and accents like plants, rocks, and vines.

Use the `Pickups` layer to mark the locations you'll spawn collectible items at. There are two kinds: gems and cherries. The tiles that spawn them are drawn with a magenta background to make them stand out. Remember, they'll be replaced at runtime by the actual items and the tiles themselves won't be seen.

Once you have your level laid out, you can limit the horizontal scrolling of the player camera to match the size of the map (plus a 5 tile buffer on each end):

```cpp
signal score_changed
var score 

func _ready():
    score = 0
    emit_signal('score_changed', score)
    pickups.hide()
    $Player.start($PlayerSpawn.position)
    set_camera_limits()

func set_camera_limits():
    var map_size = $World.get_used_rect()
    var cell_size = $World.cell_size
    $Player/Camera2D.limit_left = (map_size.position.x - 5) * cell_size.x
    $Player/Camera2D.limit_right = (map_size.end.x + 5) * cell_size.x
```

The script also needs to scan the `Pickups` layer and look for the item markers:

```cpp
func spawn_pickups():
    for cell in pickups.get_used_cells():
        var id = pickups.get_cellv(cell)
        var type = pickups.tile_set.tile_get_name(id)
        if type in ['gem', 'cherry']:
            var c = Collectible.instance()
            var pos = pickups.map_to_world(cell)
            c.init(type, pos + pickups.cell_size/2)
            add_child(c)
            c.connect('pickup', self, '_on_Collectible_pickup')

func _on_Collectible_pickup():
    score += 1
    emit_signal('score_changed', score)

func _on_Player_dead():
    pass
```

This function uses `get_used_cells()` to get an array of the tiles that are in use on the `Pickups` map. The `TileMap` sets each tile's value to an `id` that references the individual tile object in the `TileSet`. You can then query the `TileSet` for the tile's name using `tile_set.tile_get_name()`. 

Add `spawn_pickups()` to `_ready()` and add the following at the top of the script:

```cpp
var Collectible = preload('res://items/Collectible.tscn')
```

Try running your level and you should see your gems and/or cherries appear where you placed them. Also check that they disappear when you run into them.

# Scrolling background

There are two background images in the `res://assets/environment/layers` folder: `back.png` and `middle.png`, for the far and near background, respectively. By placing these images behind the tilemap and scrolling them at different speeds relative to the camera, you can create an attractive illusion of depth in the background.

To start, add a `ParallaxBackground` node to the `Level` scene. This node works automatically along with the camera to create a scrolling effect. Drag this node to the top of the scene tree so that it will be drawn behind the rest of the nodes. Next, add a `ParallaxLayer` node as a child—`ParallaxBackground` can have any number of `ParallaxLayer` as children, allowing you to make many independently scrolling layers. Add a `Sprite` node as a child to the `ParallaxLayer` and drag the `res://assets/environment/layers/back.png` image into the Texture. Important—uncheck the box next to the Centered property of the Sprite.

The background image is a little small, so set the Sprite's Scale to `(1.5, 1.5)`.

On the `ParallaxLayer`, set the Motion/Scale to `(0.2, 1)`. This setting controls how fast the background scrolls in relation to the camera. By setting it to a low number, the background will only move a small amount as the player moves left and right.

Next, you want to be sure the image repeats if your level is very wide, so set Mirroring to `(576, 0)`. This is exactly the width of the image (`384` times `1.5`), so the image will be repeated when it has moved by that amount.

Note that this background is best for wide rather than tall levels. If you jump too high, you'll reach the top of the background image and suddenly see the grey emptiness again. You can fix this by setting the top limit of the camera. If you haven't moved it, the upper-left corner of the image will be at `(0, 0)`, so you can set the Top limit on the camera to `0`. If you've moved the `ParallaxLayer`, you can find the correct value by looking at the `y` value of the node's Position.

Now, add another `ParallaxLayer` (as a sibling of the first) for the middle background layer and give it a `Sprite` child. This time, use the `res://assets/environment/layers/middle.png` texture. This texture is much narrower than the cloud/sky image, so you'll need to do a little extra adjustment to make it repeat properly. This is because the `ParallaxBackground` needs to have images that are at least as big as the viewport area.

First, click on the texture in the FileSystem dock and select the Import tab. Change the Repeat property to Mirrored, and check `On` for Mipmaps. Press Reimport. Now, the texture can be repeated to fill the screen (and the parallax system will repeat it after that):

![](img/05f28d13-c2a1-4865-87e9-8fff4d24c5c3.png)

The image's original size is `176x368`, and it needs to be repeated horizontally. In the `Sprite` properties, click On for Region Enabled. Next, set the Rect property to `(0, 0, 880, 368)` (880 is 176 times 5, so you should now see five repetitions of the image). Move the `ParallaxLayer` so that the image overlaps the bottom half of the ocean/cloud image:

Set the `ParallaxLayer` Motion/Scale to `(0.6, 1)` and the Mirroring to `(880, 0)`. Using a higher scale factor means this layer will scroll a little faster than the cloud layer behind it, giving a satisfying effect of depth, as shown in the following screenshot:

![](img/dcaf7196-4bce-4ead-b030-97d9556707de.png)

Once you're sure everything is working, try adjusting the Scale value for both layers and see how it changes. For example, try a value of `(1.2, 1)` on the middle layer for a much different visual effect.

Your main scene's tree should now look like this:

![](img/878b5297-8c6e-4fed-b435-5dbd13a6f3c7.png)

# Dangerous objects

The Danger map layer is meant to hold the spike objects that will harm the player if they're touched. Try placing a few of them on your map where you can easily test running into them. Note that because of the way TileMaps work, colliding with *any* tile on this layer will cause damage to the player!

# About slide collisions

When a `KinematicBody2D` is moved with `move_and_slide()`, it may collide with more than one object in a given frame. For example, when running into a corner, the character may hit the wall and the floor at the same time. You can use the `get_slide_count()` method to find out how many collisions occurred, and then get information about each collision with `get_slide_collision()`.

In the case of the `Player`, you want to detect when a collision occurs against the Danger `TileMap` object. You can do this just after using `move_and_slide()` in `Player.gd`:

```cpp
    velocity = move_and_slide(velocity, Vector2(0, -1))
    if state == HURT:
        return
    for idx in range(get_slide_count()):
        var collision = get_slide_collision(idx)
        if collision.collider.name == 'Danger':
            hurt()
```

Before checking for a collision with `Danger`, you can check whether the player is already in the `HURT` state and skip checking if it is. Next, you must use `get_slide_count()` to iterate through any collisions that may have occurred. For each, you can check whether the `collider.name` is `Danger`.

Run the scene and try running into one of the spike objects. Just like you wrote in the `hurt()` function previously, you should see the player change to the `HURT` state for a brief time before returning to `IDLE`. After three hits, the player enters the `DEAD` state, which currently sets the visibility to hidden.

# Enemies

Currently, the map is very lonely, so it's time to add some enemies to liven things up.

There are many different behaviors you could create for an enemy. For this project, the enemy will walk along a platform in a straight line and reverse direction when hitting an obstacle.

# Scene setup

Start with `KinematicBody2D` with three children: `Sprite`, `AnimationPlayer`, and `CollisionShape2D`. Save the scene as `Enemy.tscn` in a new folder called `enemies`. If you decide to add more enemy types to the game, you can save them all here.

Set the body's collision layer to `enemies` and its collision masks to `environment`, `player`, and `enemies`. It's also useful to group the enemies, so click on the Node tab and add the body to a group called `enemies`.

Add the `res://assets/opossum.png` sprite sheet to the Sprite's Texture. Set Vframes to `1` and Hframes to `6`. Add a rectangular collision shape that covers most (but not all) of the image, making sure that the bottom of the collision shape is aligned with the bottom of the image's feet:

![](img/e9769599-59d2-4ea3-8959-b7aa09d935b9.png)

Add a new animation to the `AnimationPlayer` called `walk`. Set the Length to `0.6` seconds and the Step to `0.1` seconds. Turn on Looping and Autoplay.

The `walk` animation will have two tracks: one that sets the Texture property and one that changes the Frame property. Click the Add keyframe button next to Texture once to add the first track, then click the one next to Frame and repeat until you have frames `0` through `5`. Press Play and verify that the walk animation is playing correctly. The Animation panel should look like this:

![](img/29c8b6f3-b0ae-4b7c-a79c-830c57652f34.png)

# Script

Add the following script:

```cpp
extends KinematicBody2D

export (int) var speed
export (int) var gravity

var velocity = Vector2()
var facing = 1

func _physics_process(delta):
    $Sprite.flip_h = velocity.x > 0
    velocity.y += gravity * delta
    velocity.x = facing * speed

    velocity = move_and_slide(velocity, Vector2(0, -1))
    for idx in range(get_slide_count()):
        var collision = get_slide_collision(idx)
        if collision.collider.name == 'Player':
            collision.collider.hurt()
        if collision.normal.x != 0:
            facing = sign(collision.normal.x)
            velocity.y = -100

    if position.y > 1000:
        queue_free()
```

In this script, the `facing` variable tracks the direction of movement (`1` or `-1`). As with the player, when moving, you iterate through the slide collisions. If the colliding object is the `Player`, you call its `hurt()` function.

Next, you can check whether the colliding body's normal vector has an `x` component that isn't `0`. This means it points to the left or right (that is, it is a wall, crate, or other obstacle). The direction of the *normal* is used to set the new facing. Finally, giving the body a small upward velocity will make the reverse transition look more appealing.

Lastly, if, for some reason, the enemy does fall off a platform, you don't want the game to have to track it falling forever, so delete any enemy whose *y* coordinate becomes too big.

Set Speed to `50` and Gravity to `900` in the Inspector, and then create an `Enemy` in your level scene. Make sure it has an obstacle on either side, and play the scene. Check that the enemy walks back and forth between the obstacles. Try putting the player in its path and verify that the player's `hurt()` method is getting called.

# Damaging the enemy

It's not fair if the player can't strike back, so in the tradition of Super Mario Bros., jumping on top of the enemy will defeat it.

Start by adding a new animation to the `AnimationPlayer` of the `Enemy` and name it `death`. Set the Length to `0.3` seconds and the Step to `0.05`. *Don't* turn on looping for this animation.

This animation will also set the Texture and Frame. This time, drag the `res://assets/enemy-death.png` image into the Sprite's Texture before adding the keyframe for that property. As before, keyframe all the `Frame` values from `0` through `5`. Press Play to see the death animation run.

Add the following code to the Enemy's script:

```cpp
func take_damage():
    $AnimationPlayer.play('death')
    $CollisionShape2D.disabled = true
    set_physics_process(false)
```

When the `Player` hits the `Enemy` under the right conditions, it will call `take_damage()`, which plays the `death` animation. It also disables collision and movement for the duration of the animation.

When the `death` animation finishes, it's OK to remove the enemy, so connect the `animation_finished()` signal of `AnimationPlayer`. This signal is called every time an animation finishes, so you need to check that it's the correct one:

```cpp
func _on_AnimationPlayer_animation_finished(anim_name):
  if anim_name == 'death':
    queue_free()
```

To complete the process, go to the `Player.gd` script and add the following to the collision checks in the `_physics_process()` method:

```cpp
for idx in range(get_slide_count()):
    var collision = get_slide_collision(idx)
    if collision.collider.name == 'Danger':
        hurt()
    if collision.collider.is_in_group('enemies'):
        var player_feet = (position + $CollisionShape2D.shape.extents).y
        if player_feet < collision.collider.position.y:
            collision.collider.take_damage()
            velocity.y = -200
        else:
            hurt()
```

This code checks the *y* coordinate of the player's feet (that is, the bottom of its collision shape) against the enemy's *y* coordinate. If the player is higher, the enemy is hurt; otherwise, the player is.

Run the level and try jumping on the enemy to make sure all is working as expected.

# HUD

The purpose of the HUD is to display the information the player needs to know during gameplay. Collecting items will increase the player's score, so that information needs to be displayed. The player also needs to see their remaining life value, which will be displayed as a series of hearts.

# Scene setup

Create a new scene with a `MarginContainer` node. Name it `HUD` and save in the `ui` folder. Set the Layout to Top Wide. In the Custom Constants section of Inspector, set the following values:

*   Margin Right: `50`
*   Margin Top: `20`
*   Margin Left: `50`
*   Margin Bottom: `20`

Add an `HBoxContainer`. This node will contain all the UI elements and keep them aligned. It will have two children: 

*   `Label`: `ScoreLabel`
*   `HBoxContainer`: `LifeCounter`

On the `ScoreLabel`, set the Text property to `1`, and under Size Flags, set Horizontal to Fill and Expand. Add a custom `DynamicFont` using `res://assets/Kenney Thick.ttf` from the `assets` folder, with a font size of `48`. In the Custom Colors section, set the Font Color to `white` and the Font Color Shadow to `black`. Finally, under Custom Constants, set Shadow Offset X, Shadow Offset Y, and Shadow As Outline all to `5`. You should see a large white 1 with a black outline.

For the `LifeCounter`, add a `TextureRect` and name it `L1`. Drag `res://assets/heart.png` into its Textureand set Stretch Mode to `Keep Aspect Centered`. Click on the node and press *Ctrl* + *D* four times so that you have a row of five hearts:

![](img/110efd8f-ca68-4a52-b9ba-9ad22b519888.png)

When finished, your HUD should look like this:

![](img/1941405b-71ea-4c2b-bc8d-e485da4a2c77.png)

# Script

Here is the script for the `HUD`:

```cpp
extends MarginContainer

onready var life_counter = [$HBoxContainer/LifeCounter/L1,
                            $HBoxContainer/LifeCounter/L2,
                            $HBoxContainer/LifeCounter/L3,
                            $HBoxContainer/LifeCounter/L4,
                            $HBoxContainer/LifeCounter/L5]

func _on_Player_life_changed(value):
    for heart in range(life_counter.size()):
        life_counter[heart].visible = value > heart

func _on_score_changed(value):
    $HBoxContainer/ScoreLabel.text = str(value)
```

First, you make an array of references to the five heart indicators. Then, in `_on_Player_life_changed()`, which will be called when the player gets hurt or healed, you calculate how many hearts to display by setting `visible` to `false` if the number of the heart is less than the life amount.

`_on_score_changed()` is similar, changing the value of the `ScoreLabel` when called.

# Attaching the HUD

Open `Level.tscn` (the base-level scene, *not* your `Level01` scene) and add a `CanvasLayer` node. Instance the `HUD` scene as a child of this `CanvasLayer`.

Click on the `Player` node and connect its `life_changed` signal to the HUD's `_on_Player_life_changed()` method:

![](img/238d3a25-9ecc-4643-b79a-2826c566ac7f.png)

Next, do the same with the `score_changed` signal of the `Level` node, connecting it to the HUD's `_on_score_changed`.

**Alternative method: **Note that if you don't want to use the scene tree to connect the signals, or if you find the signal connection window confusing, you can accomplish the same thing in code by adding these two lines to the `_ready()` function of `Level.gd`:

```cpp
$Player.connect('life_changed', $CanvasLayer/HUD,  
                '_on_Player_life_changed')
$Player.connect('dead', self, '_on_Player_dead')
connect('score_changed', $CanvasLayer/HUD, '_on_score_changed') 
```

Run your level and verify that you gain points when collecting items and lose hearts when getting hurt.

# Title screen

The title screen is the first scene the player will see. When the player dies, the game will return to this scene and allow you to restart.

# Scene setup

Start with a `Control` node and set the Layout to Full Rect.

Add a `TextureRect`. Set its Texture to `res://assets/environment/layers/back.png`, Layout to Full Rect, and Stretch Mode to Keep Aspect Covered.

Add another `TextureRect`, this time with the Texture using `res://assets/environment/layers/middle.png` and the Stretch Mode set to Tile. Drag the width of the rectangle until it's wider than the screen and arrange it so it covers the bottom half of the screen.

Next, add two `Label` nodes (`Title` and `Message`) and set their Custom Font settings using the same options you used earlier for the score label. Set their Text properties to Jungle Jump and Press Space to Play, respectively. When you're finished, the screen should look like this:

![](img/9e1e4a3a-9f03-4141-8249-b15a3ab77852.png)

To make the title screen a bit more interesting, add an `AnimationPlayer` node and create a new animation. Name it `anim` and set it to autoplay. In this animation, you can animate the various components of the screen to make them move, appear, fade in, or any other effect you like.

Drag the Title label to a position above the top of the screen and add a keyframe. Then, drag it back (or manually type the values in Position) and set another keyframe at around `0.5` seconds. Feel free to add tracks that are animating the other nodes' properties.

For example, here is an animation that drops the title down, fades in the two textures, and then makes the message appear (note the names of the properties that are modified by each track):

![](img/eddd2404-eb55-4507-8a54-85f3a515f4c7.png)

# Main scene

Delete the extra nodes you added to your temporary `Main.tscn` (the `Player` instance and the test `StaticBody2D`). This scene will now be responsible for loading the current level. Before it can do that, however, you need an Autoload script to track the game state: variables such as `current_level` and other data that needs to be carried from scene to scene.

Add a new script called `GameState.gd` in the Script editor and add the following code:

```cpp
extends Node

var num_levels = 2
var current_level = 1

var game_scene = 'res://Main.tscn'
var title_screen = 'res://ui/TitleScreen.tscn'

func restart():
    get_tree().change_scene(title_screen)

func next_level():
    current_level += 1
    if current_level <= num_levels:
        get_tree().reload_current_scene()
```

Note that you should set `num_levels` to the number of levels you've made in the `levels` folder. Make sure to name them consistently (`Level01.tscn`, `Level02.tscn`, and so on) and then you can automatically load the next one in the sequence.

Add this script in the AutoLoad tab of Project Settings, and add this script to `Main`:

```cpp
extends Node

func _ready():
    # make sure your level numbers are 2 digits ("01", etc.)
    var level_num = str(GameState.current_level).pad_zeros(2)
    var path = 'res://levels/Level%s.tscn' % level_num
    var map = load(path).instance()
    add_child(map)
```

Now, whenever the `Main` scene is loaded, it will load the level scene corresponding to `GameState.current_level`.

The title screen needs to transition to the game scene, so attach this script to the `TitleScreen` node:

```cpp
extends Control

func _input(event):
    if event.is_action_pressed('ui_select'):
        get_tree().change_scene(GameState.game_scene)
```

You can also call the restart function when the player dies by adding it to the method in `Level.gd`:

```cpp
func _on_Player_dead():
    GameState.restart()
```

# Level transitions

Your levels now need a way to transition from one to the next. In the `res://assets/environment/layers/props.png` sprite sheet, there is an image of a door that you can use for your level's exit. Finding and walking into the door will result in the player moving to the next level.

# Door scene

Make a new scene with an `Area2D` named `Door` and save it in the `items` folder. Add a `Sprite` and use the `res://assets/environment/layers/props.png` sprite sheet along with the *Region* setting to select the door image, then attach a rectangular `CollisionShape2D`. This scene doesn't need a script, because you're just going to use the area's `body_entered` signal.

Put the door on the `collectibles` layer and set its mask to only scan the `player` layer.

Instance this door scene in your first level and put it somewhere that the player can reach. Click on the `Door` node and connect the `body_entered` signal to the `Level.gd` script where you can add this code:

```cpp
func _on_Door_body_entered(body):
    GameState.next_level()
```

Run the game and try running into the door to check that it immediately transfers to the next level.

# Finishing touches

Now that you've completed the structure of the game, you can consider some additions so that you can add more game features, more visual effects, additional enemies, or other ideas you might have. In this section, there are a few suggested features—add them as-is or adjust them to your liking.

# Sound effects

As with the previous projects, you can add audio effects and music to improve the gameplay experience. In the `res://assets/audio` folder, you'll find a number of files you can use for various game events, such as player jump, enemy hit and pickup. There are also two music files: Intro Theme for the title screen and Grasslands Theme for the level scenes.

Adding these to the game will be left to you, but here are a few tips:

*   Make sure the sound effects have Loop set to Off while the music files have it On in the Import settings tab.
*   You may find it helpful to adjust the volume of individual sounds. This can be set with the Volume Db property. Setting a negative value will reduce the sound's volume.
*   You can attach music to the master `Level.tscn` and that music will be used for all levels (set the `AudioStreamPlayer` to Autoplay).
*   You an also attach separate music to individual levels if you want to set a certain mood.

# Infinite falling

Depending on how you've designed your levels, it may be possible for the player to fall off the level entirely. Typically, you want to design things so that this isn't possible by using walls that are too high to jump, spikes at the bottom of pits, and so on. However, in case it does happen, add the following code to the player's `_physics_process()` method:

```cpp
if position.y > 1000:
    change_state(DEAD)
```

Note that if you've designed a level that extends below a `y` of `1000`, you'll need to increase the value to prevent accidental death.

# Double jump

Double-jumps are a popular platforming feature. The player gets a second, usually smaller, upwards boost if they press the jump key a second time while in the air. To implement this feature, you need to add a few things to the player script.

First, you will need two variables to track the state:

```cpp
var max_jumps = 2
var jump_count = 0
```

When entering the `JUMP` state, reset the number of jumps:

```cpp
JUMP:
    new_anim = 'jump_up'
    jump_count = 1
```

Finally, in `get_input()`, allow the jump if it meets the conditions:

```cpp
if jump and state == JUMP and jump_count < max_jumps:
    new_anim = 'jump_up'
    velocity.y = jump_speed / 1.5
    jump_count += 1
```

Note that this makes the second jump 2/3 the upward speed of the normal jump. You can adjust this according to your preferences.

# Dust particles

Dust particles at the character's feet are a low-effort effect that can add a lot of character to your player's movements. In this section, you'll add a small puff of dust to the player's feet that is emitted whenever they land on the ground. This adds a sense of weight and impact to the player's jumps.

Add a `Particles2D` node and name it `Dust`. Note the warning that a process material must be added. First, however, set the properties of the `Dust` node:

| **Property** | **Value** |
| Amount | `20` |
| Lifetime | `0.45` |
| One Shot | `On` |
| Speed Scale | `2` |
| Explosiveness | `0.7` |
| Local Coords | `Off` |
| Position | `(-2, 15)` |
| Rotation | `-90` |

Now, under Process Material, add a new `ParticlesMaterial`. Click on it and you'll see all the particle settings. Here are the ones you need for the dust effect:

| **Particle Property** | **Value** |
| Emission Shape | `Box` |
| Box Extents | `(1, 6, 1)` |
| Gravity | `(0, 0, 0)` |
| Initial Velocity | `10` |
| Velocity Random | `1` |
| Scale | `5` |
| Scale Random |  `1` |

The default particle color is white, but the dust effect will look better as a tan shade. It should also fade away so that it appears to dissipate. This can be accomplished with a `ColorRamp`. Next to Color Ramp, click on New GradientTexture. In the `GradientTexture` properties, choose a new `Gradient`.

The `Gradient` has two colors: a start color on the left and an end color on the right. These are selected by the small rectangles at the ends of the gradient. Clicking on the square on the right allows you to set the color:

![](img/0ee3193e-e971-45cf-8fc2-ce575dbc6680.png)

Set the start color to a tan shade, and set the end color to the same color, but with the alpha value set to `0` (transparent). You can test how it looks by checking the Emitting box in the Inspector. Because the node is set to One Shot, there will only be one puff of particles and you have to check the box again to emit them.

Feel free to alter the properties from what is listed here. Experimenting with `Particles2D` settings can be great fun, and often you'll stumble on to a very nice effect just by tinkering. Once you're happy with the appearance, add the following to the Player's `_physics_process()` code:

```cpp
if state == JUMP and is_on_floor():
    change_state(IDLE)
    $Dust.emitting = true # add this line
```

Run the game and every time your character lands on the ground, a small puff of dust will appear.

# Crouching state

The crouching state is useful if you have enemies or projectiles that the player needs to dodge by ducking under them. The sprite sheet contains a two-frame animation for this state:

![](img/0e3f5fc6-edf5-44c7-adb2-643b7ab74ca1.png)

Add a new animation called crouch to the player's `AnimationPlayer`. Set its Length to `0.2` and add a track for the Frame property that changes the value from `3` to `4`. Set the animation to loop.

In the player's script, add the new state to the `enum` and state change:

```cpp
enum {IDLE, RUN, JUMP, HURT, DEAD, CROUCH}
```

```cpp
CROUCH:
    new_anim = 'crouch'
```

In the `get_input()` method, you need to handle the various state transitions. When on the ground, the down input should transition to `CROUCH`. When in `CROUCH`, releasing the down input should transition to `IDLE`. Finally, if in the `CROUCH` state and left or right is pressed, the state should change to `RUN`:

```cpp
var down = Input.is_action_pressed('crouch')

if down and is_on_floor():
    change_state(CROUCH)
if !down and state == CROUCH:
    change_state(IDLE)
```

You also need to change this line:

```cpp
if state == IDLE and velocity.x != 0:
    change_state(RUN)
```

To this:

```cpp
if state in [IDLE, CROUCH] and velocity.x != 0:
    change_state(RUN)
```

That's it! Run the game and try out your new animation state.

# Climbing ladders

The player animation also includes frames for a *climbing* action, and the tileset contains ladders. Currently, the ladder tiles do nothing: in the TileSet, they do not have any collision shape assigned. That's fine, because you don't want the player to collide with the ladders; you want to be able to move up and down on them.

# Player code

Start by clicking on the player's `AnimationPlayer` and adding a new animation named `climb`. Its Length should be set to `0.4` seconds and the Frame values for the `Sprite` are `0, 1, 0, 2`. Set the animation to loop.

Now, go to `Player.gd` and add a new state, `CLIMB`, to the state enum. In addition, add two new variables to the declarations at the top:

```cpp
export (int) var climb_speed
var is_on_ladder = false
```

`is_on_ladder` will be used to tell if the player is on a ladder or not. Using this, you can decide whether the up arrow should have any effect. In the Inspector, set Climb Speed to `50`.

In `change_state()`, add a condition for the new state:

```cpp
CLIMB:
    new_anim = 'climb'
```

Next, in `_get_input()`, you need to add the `climb` input action and add the code to determine when to trigger the new state. Add the following:

```cpp
var climb = Input.is_action_pressed('climb')

if climb and state != CLIMB and is_on_ladder:
    change_state(CLIMB)
if state == CLIMB:
    if climb:
        velocity.y = -climb_speed
    elif down:
        velocity.y = climb_speed
    else:
        velocity.y = 0
        $AnimationPlayer.play("climb")
if state == CLIMB and not is_on_ladder:
    change_state(IDLE)
```

Here, you have three new conditions to check. First, if the player is not in the `CLIMB` state, but is on a ladder, then pressing up should start make the player start climbing. Next, if the player is climbing, then up and down should move them accordingly, but halt movement if no keys are pressed. Finally, if the player leaves the ladder while climbing, it will leave the `CLIMB` state. 

The one remaining issue is you need gravity to stop pulling the player downwards when climbing. Add the following condition to the gravity code in `_physics_process()`:

```cpp
if state != CLIMB:
    velocity.y += gravity * delta
```

Now, the player is ready, and you can add some ladders to your level map.

# Level code

Place a few ladder tiles somewhere on your map, then add a Ladder `Area2D` to the level scene. Give this node a `CollisionShape2D` with a rectangular shape. The best way to size the area is to use grid snapping. Turn this on via the menu and use Configure Snap... to set the grid step to `(4, 4)`:

![](img/26c24151-386c-4645-aa4e-23e8a5a89843.png)

Adjust the collision shape so that it covers the center portion of the ladder from top to bottom. If you make the shape fully as wide as the ladder, the player will still count as climbing even when hanging off the side. You may find that this looks a bit odd, so making the shape a bit smaller than the width of the ladder will prevent this.

Connect the `body_entered` and `body_exited` signals of the `Ladder` and add the following code to have them set the Player's ladder variable:

```cpp
func _on_Ladder_body_entered(body):
    if body.name == "Player":
        body.is_on_ladder = true

func _on_Ladder_body_exited(body):
    if body.name == "Player":
        body.is_on_ladder = false
```

Now you can give it a try. You should be able to walk to the ladder and climb up and down it. Note that if you are at the top of a ladder and step onto it, you'll fall to the bottom rather than climb down (although pressing up as you fall will grab the ladder). If you prefer to automatically transition to the climbing state, you can add an additional falling check in `_physics_process()`.

# Moving platforms

Make a new scene with a `KinematicBody2D` root node. Add a `Sprite` child and use the `res://assets/environment/layers/tileset.png` sprite sheet as the Texture with Region enabled so you can choose one particular tile. You probably want your platform to be wider than one tile, so duplicate the `Sprite` as many times as you like. Turn grid snapping on so that the sprites can be aligned in a row:

![](img/f98f167d-2d4f-4335-bc83-d53cfb958a72.png)

A grid setting of `(8, 8)` works well for aligning the tiles. Add a rectangular `CollisionShape2D` that covers the image:

![](img/ff85015d-8bc9-42ac-9a7c-fcef6ab09c7a.png)

Platform movement can be made very complex (following paths, changing speeds, and so on), but this example will stick with a platform that moves horizontally back and forth between two objects.

Here is the platform's script:

```cpp
extends KinematicBody2D

export (Vector2) var velocity

func _physics_process(delta):
    var collision = move_and_collide(velocity * delta)
    if collision:
        velocity = velocity.bounce(collision.normal)
```

This time, you're using `move_and_collide()` to move the kinematic body. This is a better choice since the platform shouldn't slide when it collides with another wall. Instead, it bounces off the colliding body. As long as your collision shapes are rectangular (as the `TileMap` bodies are), this method will work fine. If you have a rounded object, the bounce may send the platform off in a strange direction, in which case you should use something like the following to keep the motion horizontal:

```cpp
func _physics_process(delta):
    var collision = move_and_collide(velocity * delta)
    if collision:
        velocity.x *= -1
```

Set the *Velocity* in the Inspector to `(50, 0)`, then go to your level scene and instance one of these objects somewhere in your level. Make sure it is between two objects so that it can move back and forth between them.

Run the scene and try jumping on the moving platform. Since the Player is using `move_and_slide()`, they will automatically move along with the platform if you stand on it.

Add as many of these objects as you like to your level. They will even bounce off each other, so you can make chains of moving platforms that cover a large distance and require careful timing of the player's jumps.

# Summary

In this chapter, you learned how to use the `KinematicBody2D` node to create arcade-style physics. You also used the `AnimationPlayer` to create a variety of animations for character behavior, and made extensive use of what you learned in earlier projects to tie everything together. Hopefully, by this point, you have a good grasp of the scene system and how a Godot project is structured.

Remember the Stretch Mode and Aspect properties you set in the Project Settings at the beginning? Run the game and observe what happens when you resize the game window. These settings are the best for this style of game, but try changing the Stretch Mode to Viewport instead, then make your game window very wide or tall. Experiment with the other settings to see the effect of the different resizing options.

Once again, before moving on, take a few moments to play your game and look through its various scenes and scripts to review how you built it. Review any sections of this chapter that you found particularly tricky.

In the next chapter, you'll make the jump to 3D!