# Space Rocks

By now, you should be getting more comfortable with working in Godot; adding nodes, creating scripts, modifying properties in the Inspector, and so on. As you progress through this book, you won't be forced to rehash the basics again and again. If you find yourself stuck, or feeling like you don't quite remember how something is done, feel free to jump back to a previous project where it was explained in more detail. As you repeat the more common actions in Godot, they will start to feel more and more familiar. At the same time, each chapter will introduce you to more nodes and techniques to expand your understanding of Godot's features.

In this next project, you'll make a space shooter game similar to the arcade classic Asteroids. The player will control a ship that can rotate and move in any direction. The goal will be to avoid the floating *space rocks* and shoot them with the ship's laser. Refer to the following screenshot:

![](img/3453005f-0741-4889-a4a9-4bbd9b686608.png)

You will learn about the following key topics in this project:

*   Physics using `RigidBody2D`
*   Finite State Machines
*   Building a dynamic, scalable UI
*   Sound and music
*   Particle effects

# Project setup

Create a new project and download the project assets from [https://github.com/PacktPublishing/Godot-Game-Engine-Projects/releases](https://github.com/PacktPublishing/Godot-Game-Engine-Projects/releases).

For this project, you'll set up custom input actions using the Input Map. Using this feature, you can define custom events and assign different keys, mouse events, or other inputs to them. This allows for more flexibility in designing your game, as your code can be written to respond to the `jump` input, for example, without needing to know exactly what input the user pressed to make the event happen. This allows you to make the same code work on different devices, even if they have different hardware. In addition, since many gamers expect to be able to customize a game's inputs, this enables you to provide that option to the user as well.

To set up the inputs for this game, open Project | Project Settings and select the Input Map tab.

You'll need to create four new input actions: `rotate_left`, `rotate_right`, `thrust`, and `shoot`. Type the name of each action into the Action box and click Add. Then, for each action, click the + button and select the type of input to assign. For example, to allow the player to use both the arrow keys and the popular WASD alternative, the setup will look like this:

![](img/e5e3ffae-1a9a-4712-b868-9c454b19d6cd.png)

If you have a gamepad or other controller connected to your computer, you can also add its inputs to the actions in the same way. Note: we're only considering button-style inputs at this stage, so while you'll be able to use a d-pad for this project, using an analog joystick would require changes to the project's code.

# Rigid body physics

In game development, you often need to know when two objects in the game space intersect or come into contact. This is known as *collision detection*. When a collision is detected, you typically want something to happen. This is known as *collision response*.

Godot offers three kinds of physics bodies, grouped under the `PhysicsBody2D` object type:

*   `StaticBody2D`: A static body is one that is not moved by the physics engine. It participates in collision detection, but does not move in response to the collision. This type of body is most often used for objects that are part of the environment or do not need to have any dynamic behavior, such as walls or the ground.

*   `RigidBody2D`: This is the physics body in Godot that provides simulated physics. This means that you don't control a `RigidBody2D` directly. Instead, you apply forces to it (gravity, impulses, and so on) and Godot's built-in physics engine calculates the resultant movement, including collisions, bouncing, rotating, and other effects.

*   `KinematicBody2D`: This body type provides collision detection, but no physics. All movement must be implemented in code, and you must implement any collision response yourself. Kinematic bodies are most often used for player characters or other actors that require *arcade-style* physics rather than realistic simulation.

Understanding when to use a particular physics body type is a big part of building your game. Using the right node can simplify your development, while trying to force the wrong node to do the job can lead to frustration and poor results. As you work with each type of body, you'll come to learn their pros and cons and get a feel for when they can help build what you need.

In this project, you'll be using the `RigidBody2D` node for the player ship as well as the *space rocks* themselves. You'll learn about the other body types in later chapters.

Individual `RigidBody2D` nodes have many properties you can use to customize their behavior, such as `Mass`, `Friction`, or `Bounce`. These properties can be set in the Inspector:

![](img/4cc35f3b-a174-4d0a-960d-efeb1d86e285.png)

Rigid bodies are also affected by the world's properties, which can be set in the Project Settings under Physics | 2D. These settings apply to all bodies in the world. Refer to the following screenshot:

![](img/ef7f76ba-33f1-4c83-8317-d838b5d6e35a.png)

In most cases, you won't need to modify these settings. However, note that by default, gravity has a value of `98` and a direction of `(0, 1)` (downward). If you want to change the world gravity, you can do that here. You should also be aware of the last two properties, Default Linear Damp and Default Angular Damp. These properties control how quickly a body will lose forward speed and rotation speed, respectively. Setting them to lower values will make the world feel frictionless, while using larger values will feel like your objects are moving through mud.

`Area2D` nodes can also be used to affect rigid body physics by using the Space Override property. Custom gravity and damping values will then be applied to any bodies that enter the area.

Since this game will be taking place in outer space, gravity won't be needed, so set Default Gravity to `0`. You can leave the other settings as they are.

# Player ship

The player ship is the heart of the game. Most of the code you'll write for this project will be about making the ship work. It will be controlled in the classic Asteroids style, with left/right rotation and forward thrust. It will also detect the shoot input to allow the player to fire the laser and destroy the floating rocks.

# Body setup and physics

Create a new scene and add a `RigidBody2D` named `Player` as the root node, with `Sprite` and `CollisionShape2D` children. Add the `res://assets/player_ship.png` image to the Texture property of the `Sprite`. The ship image is quite large, so set the Scale property of the `Sprite` to `(0.5, 0.5)`and its Rotation to `90`.

The image for the ship is drawn pointing upwards. In Godot, a rotation of `0` degrees points to the right (along the *x* axis). This means you need to set the Rotation of the `Sprite` node to `90` so it will match the body's direction.

In the Shape property of `CollisionShape2D`, add a `CircleShape2D` and scale it to cover the image as closely as possible (remember not to move the rectangular size handles):

![](img/77ab9da2-81c0-42d8-88cd-bfffa8e3d585.png)

Save the scene. When working on larger-scale projects, it is recommended to organize your scenes and scripts into folders based on each game object. For example, if you make a `player` folder, you can save player-related files there. This makes it easier to find and modify your files rather than having them all together in a single folder. While this project is relatively small, it's a good habit to adopt as your projects grow in size and complexity.

# State machines

The player ship can be in a number of different states during gameplay. For example, when *alive,* the ship is visible and can be controlled by the player, but is vulnerable to being hit by rocks. On the other hand, when *invulnerable, *the ship should appear semi-transparent and immune to damage.

One way that programmers often handle situations like this is to add Boolean flag variables to the code. For example, the `invulnerable` flag is set to `true` when the player spawns, or when the `alive` flag is set to `false` when the player is dead. However, this can lead to errors and strange situations where both the `alive` and `invulnerable` flags are set to `true` at the same time. What happens when a rock hits the player in this situation? The two states are mutually exclusive, so this shouldn't be allowed to happen.

One solution to this problem is to use a **Finite State Machine** (**FSM**). When using an FSM, an entity can only be in one state at a given time. To design your FSM, you define some number of states and what events or actions can cause a transition from one state to another.

The following diagram outlines the FSM for the player ship:

![](img/1d5ed158-4b96-4c18-9a64-c488e2257e63.png)

There are four states, and the arrows indicate what transitions are allowed, as well as what event triggers the transition. By checking the current state, you can decide what the player is allowed to do. For example, in the **DEAD** state, don't allow input, or in the **INVULNERABLE** state, don't allow shooting.

Advanced FSM implementations can become quite complex, and the details are beyond the scope of this book (see the Appendix for further reading). In the purest sense, you technically won't be creating a true FSM, but for the purposes of this project, it will be sufficient to illustrate the concept and keep you from running into the Boolean flag problem.

Add a script to the `Player` node and start by creating the skeleton of the FSM implementation:

```cpp
extends RigidBody2D

enum {INIT, ALIVE, INVULNERABLE, DEAD}
var state = null
```

An `enum` (short for enumeration) is a convenient way to create a set of constants. The `enum` statement in the preceding code snippet is equivalent to the following code:

```cpp
const INIT = 0
const ALIVE = 1
const INVULNERABLE = 2
const DEAD = 3
```

You can also assign a name to an `enum`, which is useful when you have more than one collection of constants in a single script. For example:

```cpp
enum States {INIT, ALIVE}

var state = States.INIT
```

However, this isn't needed in this script, as you'll only be using the one `enum` to track the ship's states.

Next, create the `change_state` function to handle state transitions:

```cpp
func _ready():
    change_state(ALIVE)

func change_state(new_state):
    match new_state:
        INIT:
            $CollisionShape2D.disabled = true
        ALIVE:
            $CollisionShape2D.disabled = false
        INVULNERABLE:
            $CollisionShape2D.disabled = true
        DEAD:
            $CollisionShape2D.disabled = true
    state = new_state
```

Whenever you need to change the state of the player, you'll call the `change_state()` function and pass it the value of the new state. Then, by using a `match` statement, you can execute whatever code should accompany the transition to the new state. To illustrate this, the `CollisionShape2D` is being enabled/disabled by the `new_state` value. In `_ready()`, you specify the initial state—currently `ALIVE` so that you can test, but you'll change it to `INIT` later.

# Controls

Add the following variables to the script:

```cpp
export (int) var engine_power
export (int) var spin_power

var thrust = Vector2()
var rotation_dir = 0
```

`engine_power` and `spin_power` control how fast the ship can accelerate and turn. In the Inspector, set them to `500` and `15000`, respectively. `thrust` will represent the force being applied by the ship's engine: either `(0, 0)` when coasting, or a vector with the length of `engine_power` when powered on. `rotation_dir` will represent what direction the ship is turning in and apply a torque, or rotational force.

By default, the physics settings provide some *damping*, which reduces a body's velocity and spin. In space, there's no friction, so for realism there shouldn't be any damping at all. However, for an arcade-style feel, it's preferable that the ship should stop when you let go of the keys. In the Inspector, set the player's Linear/Damp to `1` and its Angular/Damp to `5`.

The next step is to detect the input and move the ship:

```cpp
func _process(delta):
    get_input()

func get_input():
    thrust = Vector2()
    if state in [DEAD, INIT]:
        return
    if Input.is_action_pressed("thrust"):
        thrust = Vector2(engine_power, 0)
    rotation_dir = 0
    if Input.is_action_pressed("rotate_right"):
        rotation_dir += 1
    if Input.is_action_pressed("rotate_left"):
        rotation_dir -= 1

func _physics_process(delta):
    set_applied_force(thrust.rotated(rotation))
    set_applied_torque(spin_power * rotation_dir)

```

The `get_input()` function captures the key actions and sets the ship's thrust on or off, and the rotation direction (`rotation_dir`) to a positive or negative value (representing clockwise or counter-clockwise rotation). This function is called every frame in `_process()`. Note that if the state is `INIT` or `DEAD`, `get_input()` will exit by using `return` before checking for key actions.

When using physics bodies, their movement and related functions should be called in `_physics_process()`. Here, you can use `set_applied_force()` to apply the engine thrust in whatever direction the ship is facing. Then, you can use `set_applied_torque()` to cause the ship to rotate.

Play the scene and you should be able to fly around freely.

# Screen wrap

Another feature of classic 2D arcade games is *screen wrap*. If the player goes off one side of the screen, they *appear* on the other side. In practice, you teleport or instantaneously change the ship's position to the opposite side. Add the following to the class variables at the top of the script:

```cpp
var screensize = Vector2() 
```

And add this to `_ready()`:

```cpp
screensize = get_viewport().get_visible_rect().size
```

Later, the game's main script will handle setting `screensize` for all of the game's objects, but for now, this will allow you to test the screen wrapping with just the player scene.

When first approaching this problem, you might think you could use the body's `position` property and, if it exceeds the bounds of the screen, set it to the opposite side. However, when using `RigidBody2D`, you can't directly set its `position`, because that would conflict with the movement that the physics engine is calculating. A common mistake is to try adding something like this to `_physics_process()`:

```cpp
func _physics_process(delta):
    if position.x > screensize.x:
        position.x = 0
    if position.x < 0:
        position.x = screensize.x
    if position.y > screensize.y:
        position.y = 0
    if position.y < 0:
        position.y = screensize.y
    set_applied_force(thrust.rotated(rotation))
    set_applied_torque(rotation_dir * spin_thrust)
```

This will fail, trapping the player on the edge of the screen (and occasionally *glitching* unpredictably at the corners). So, why doesn't this work? The Godot documentation recommends `_physics_process()` for physics-related code—it even has *physics* in the name. It makes sense at first glance that this should work correctly.

In fact, the correct way to solve this problem is *not* to use `_physics_process()`.

To quote the `RigidBody2D` docs:

"You should not change a RigidBody2D's position or linear_velocity every frame or even very often. If you need to directly affect the body's state, use _integrate_forces, which allows you to directly access the physics state."

And in the description for `_integrate_forces()`:

"(It) Allows you to read and safely modify the simulation state for the object. Use this instead of _physics_process if you need to directly change the body's position or other physics properties. (emphasis added)"

The answer is to change the physics callback to `_integrate_forces()`, which gives you access to the body's `Physics2DDirectBodyState`. This is a Godot object containing a great deal of useful information about the current physics state of the body. In the case of location, the key piece of information is the body's `Transform2D`.

A *transform* is a matrix representing one or more transformations in 2D space such as translation, rotation, and/or scaling. The translation (that is, position) information is found by accessing the `origin` property of the `Transform2D`.

Using this information, you can implement the wrap around effect by changing `_physics_process()` to `_integrate_forces()` and altering the transform's origin:

```cpp
func _integrate_forces(physics_state):
    set_applied_force(thrust.rotated(rotation))
    set_applied_torque(spin_power * rotation_dir)
    var xform = physics_state.get_transform()
    if xform.origin.x > screensize.x:
        xform.origin.x = 0
    if xform.origin.x < 0:
        xform.origin.x = screensize.x
    if xform.origin.y > screensize.y:
        xform.origin.y = 0
    if xform.origin.y < 0:
        xform.origin.y = screensize.y
    physics_state.set_transform(xform)
```

Note that the function's argument name has been changed to `physics_state` from its default: `state`. This is to prevent any possible confusion with the already existing `state` variable, which tracks what FSM state the player is currently assigned to.

Run the scene again and check that everything is working as expected. Make sure you try wrapping around in all four directions. A common mistake is to accidentally flip a greater-than or less-than sign, so check that first if you're having a problem with one or more screen edges.

# Shooting

Now, it's time to give your ship some weapons. When pressing the `shoot` action, a bullet should be spawned at the front of the ship and travel in a straight line until it exits the screen. Then, the gun isn't allowed to fire again until a small amount of time has passed.

# Bullet scene

This is the node setup for the bullet:

*   `Area2D` (named `Bullet`)
*   `Sprite`
*   `CollisionShape2D`
*   `VisibilityNotifier2D`

Use `res://assets/laser.png` from the assets folder for the texture of the `Sprite`, and a `CapsuleShape2D` for the collision shape. You'll have to set the Rotation of the `CollisionShape2D` to `90` so that it will fit correctly. You should also scale the `Sprite` down to half size (`(0.5, 0.5)`).

Add the following script to the `Bullet` node:

```cpp
extends Area2D

export (int) var speed
var velocity = Vector2()

func start(pos, dir):
    position = pos
    rotation = dir
    velocity = Vector2(speed, 0).rotated(dir)

func _process(delta):
    position += velocity * delta
```

Set the exported `speed` property to `1000`.

The `VisibilityNotifier2D` is a node that can inform you (using signals) whenever a node becomes visible/invisible. You can use this to automatically delete a bullet when it goes off screen. Connect the `screen_exited` signal of `VisibilityNotifier2D` and add this:

```cpp
func _on_VisibilityNotifier2D_screen_exited():
    queue_free()
```

Finally, connect the bullet's `body_entered` signal so that you can detect when the bullet hits a rock. The bullet doesn't need to *know* anything about rocks, just that it has hit something. When you create the rocks, you'll add them to a group called `rocks` and give them an `explode()` method:

```cpp
func _on_Bullet_body_entered( body ):
    if body.is_in_group('rocks'):
        body.explode()
        queue_free()
```

# Firing bullets

Now, you need instances of the bullet to be created whenever the player fires. However, if you make the bullet a child of the player, then it will move and rotate along with the player instead of moving independently. Instead, the bullet should be added as a child of the main scene. One way to do this would be to use `get_parent().add_child()`, since the `Main` scene will be the parent of the player when the game is running. However, this would mean you could no longer run the `Player` scene by itself like you have been doing, because `get_parent()` would produce an error. Or, if in the `Main` scene you decided to arrange things differently, making the player a child of some other node, the bullet wouldn't be added where you expect.

In general, it is a bad idea to write code that assumes a fixed tree layout. Especially try to avoid using `get_parent()` if at all possible. You may find it difficult to think this way at first, but it will result in a much more modular design and prevent some common mistakes.

Instead, the player will *give* the bullet to the main scene using a signal. In this way, the `Player` scene doesn't need to *know* anything about how the `Main` scene is set up, or even if the `Main` scene exists. Producing the bullet and handing it off is the `Player` object's only responsibility.

Add a `Position2D` node to the player and name it `Muzzle`. This will mark the *muzzle* of the gun—the location where the bullet will spawn. Set its Position to `(50, 0)` to place it directly in front of the ship.

Next, add a `Timer` node named `GunTimer`. This will provide a *cooldown* to the gun, preventing a new bullet from firing until a certain amount of time has passed. Check the One Shot and Autoplay boxes.

Add these new variables to the player's script:

```cpp
signal shoot

export (PackedScene) var Bullet
export (float) var fire_rate

var can_shoot = true
```

Drag the `Bullet.tscn` onto the new Bullet property in the Inspector, and set the Fire Rate to `0.25` (this value is in seconds).

Add this to `_ready()`:

```cpp
$GunTimer.wait_time = fire_rate
```

And this to `get_input()`:

```cpp
if Input.is_action_pressed("shoot") and can_shoot:
    shoot()
```

Now, create the `shoot()` function, which will handle creating the bullet(s):

```cpp
func shoot():
    if state == INVULNERABLE:
        return
    emit_signal("shoot", Bullet, $Muzzle.global_position, rotation)
    can_shoot = false
    $GunTimer.start()
```

When emitting the `shoot` signal, you pass the `Bullet` itself plus its starting position and direction. Then, you disable shooting with the `can_shoot` flag and start the `GunTimer`. To allow the gun to shoot again, connect the `timeout` signal of the `GunTimer`:

```cpp
func _on_GunTimer_timeout():
    can_shoot = true
```

Now, make your Main scene. Add a `Node` named `Main` and a `Sprite` named `Background`. Use `res://assets/space_background.png` as the Texture*.* Add an instance of the `Player` to the scene.

Add a script to `Main`, then connect the `Player` node's `shoot` signal, and add the following to the created function:

```cpp
func _on_Player_shoot(bullet, pos, dir):
    var b = bullet.instance()
    b.start(pos, dir)
    add_child(b)
```

Play the `Main` scene and test that you can fly and shoot.

# Rocks

The goal of the game is to destroy the floating space rocks, so, now that you can shoot, it's time to add them. Like the ship, the rocks will also be `RigidBody2D`, which will make them travel in a straight line at a steady speed unless disturbed. They'll also bounce off each other in a realistic fashion. To make things more interesting, rocks will start out large and, when you shoot them, break into multiple smaller rocks.

# Scene setup

Start a new scene by making a `RigidBody2D`, naming it `Rock`, and adding a `Sprite` using the `res://assets/rock.png` texture. Add a `CollisionShape2D`, but *don't* add a shape to it yet. Because you'll be spawning different-sized rocks, the collision shape will need to be set in the code and adjusted to the correct size.

Set the Bounce property of the `Rock` to `1` and both Linear/Damp and Angular/Damp to `0`.

# Variable size

Attach a script and define the member variables:

```cpp
extends RigidBody2D

var screensize = Vector2()
var size
var radius
var scale_factor = 0.2
```

The `Main` script will handle spawning new rocks, both at the beginning of a level as well as the smaller rocks that will appear after a large one explodes. A large rock will have a `size` of `3` and break into rocks of size `2`, and so on. The `scale_factor` is multiplied by `size` to set the sprite's scale, the collision radius, and so on. You can adjust it later to change how big each category of rock is.

All of this will be set by the `start()` method:

```cpp
func start(pos, vel, _size):
    position = pos
    size = _size
    mass = 1.5 * size
    $Sprite.scale = Vector2(1, 1) * scale_factor * size
    radius = int($Sprite.texture.get_size().x / 2 * scale_factor * size)
    var shape = CircleShape2D.new()
    shape.radius = radius
    $CollisionShape2D.shape = shape
    linear_velocity = vel
    angular_velocity = rand_range(-1.5, 1.5)
```

Here is where you calculate the correct collision shape based on the rock's `size` and add it to the `CollisionShape2D`. Note that since `size` is already in use as a class variable, you can use `_size` for the function argument.

The rocks also need to wrap around the screen, so use the same technique you used for the `Player`:

```cpp
func _integrate_forces(physics_state):
    var xform = physics_state.get_transform()
    if xform.origin.x > screensize.x + radius:
       xform.origin.x = 0 - radius
    if xform.origin.x < 0 - radius:
       xform.origin.x = screensize.x + radius
    if xform.origin.y > screensize.y + radius:
       xform.origin.y = 0 - radius
    if xform.origin.y < 0 - radius:
       xform.origin.y = screensize.y + radius
    physics_state.set_transform(xform)
```

The difference here is that including the body's `radius` results in smoother-looking teleportation. The rock will appear to fully exit the screen before entering at the opposite side. You may want to do the same thing with the player ship. Try it and see which effect you like better.

# Instancing

When new rocks are spawned, the main scene will need to pick a random start location. To do this, you could use some geometry to pick a random point along the perimeter of the screen, but instead you can take advantage of yet another Godot node type. You'll draw a path around the edge of the screen, and the script will pick a random location along the path. Add a `Path2D` node and name it `RockPath`. When you click on the `Path2D`, you will see some new buttons appear at the top of the editor:

![](img/049a9c5f-2720-4687-bf7d-d6720d98348b.png)

Select the middle one (Add Point) to draw the path by clicking to add the points shown. To make the points align, make sure Snap to grid is checked. This option is found under the Snapping Options button to the left of the `Lock` button. It appears as a series of three vertical dots. Refer to the following screenshot:

![](img/3adce6de-d266-4b9f-a639-51cae3a3a506.png)

Draw the points in the order shown in the following screenshot. After clicking the fourth point, click the Close Curve button (**5**) and your path will be complete:

![](img/da983698-1c01-432f-aa34-6b4d2efb4ceb.png)

Now that the path is defined, add a `PathFollow2D` node as a child of `RockPath` and name it `RockSpawn`. This node's purpose is to automatically follow a path as it moves, using its `set_offset()` method. The higher the offset, the further along the path it goes. Since our path is closed, it will loop around if the offset value is bigger than the path's length.

Next, add a `Node` and name it `Rocks`. This node will serve as a container to hold all the rocks. By checking its number of children, you can tell if there are any rocks remaining.

Now, add this to `Main.gd`:

```cpp
export (PackedScene) var Rock

func _ready():
    randomize()
    screensize = get_viewport().get_visible_rect().size
    $Player.screensize = screensize
    for i in range(3):
        spawn_rock(3)
```

The script starts by getting the `screensize` and passing that to the `Player`. Then, it spawns three rocks of size `3` using `spawn_rock()`, which is defined in the following code. Don't forget to drag `Rock.tscn` onto the Rock property in the Inspector:

```cpp
func spawn_rock(size, pos=null, vel=null):
    if !pos:
        $RockPath/RockSpawn.set_offset(randi())
        pos = $RockPath/RockSpawn.position
    if !vel:
        vel = Vector2(1, 0).rotated(rand_range(0, 2*PI)) * rand_range(100, 150)
    var r = Rock.instance()
    r.screensize = screensize
    r.start(pos, vel, size)
    $Rocks.add_child(r)
```

This function will serve two purposes. When called with only a size parameter, it picks a random position along the `RockPath` and a random velocity. However, if those values are also provided, it will use them instead. This will let you spawn the smaller rocks at the location of the explosion.

Run the game and you should see three rocks floating around. However, your bullets don't affect them.

# Exploding rocks

The `Bullet` is checking for bodies in the `rocks` group, so in the `Rock` scene, click on the Node tab and choose Groups. Type `rocks` and click Add:

![](img/e692448c-8b16-4c2d-996d-b09bd61a8df0.png)

Now, if you run the game and shoot a rock, you'll see an error message because the bullet is trying to call the rock's `explode()` method, which you haven't defined yet. This method needs to do three things:

*   Remove the rock
*   Play an explosion animation
*   Notify `Main` to spawn new, smaller rocks

# Explosion scene

The explosion will be a separate scene, which you can add to the `Rock` and later to the `Player`. It will contain two nodes:

*   `Sprite` (named `Explosion`)
*   `AnimationPlayer`

For the sprite's Texture, use `res://assets/explosion.png`. You'll notice that this is a sprite sheet—an image made up of 64 smaller images laid out in a grid pattern. These images are the individual frames of the animation. You'll often find animations packaged this way, and Godot's `Sprite` node supports using them as individual frames.

In the Inspector, find the sprite's Animation section. Set the Vframes and Hframes both to `8`. This will *slice* the sprite sheet into its individual images. You can verify this by changing the Frame property to different values between `0` and `63`. Make sure to set Frames back to `0` when finished:

![](img/e182f5e3-83a8-4d73-8aca-1d10c7187995.png)

The `AnimationPlayer` can be used to animate any property of any node. You'll use the `AnimationPlayer` to change the Frame property over time. Start by clicking on the node and you'll see the Animation panel open at the bottom, as shown in the following screenshot:

![](img/bd1ee949-0389-418e-811e-306d5e49c805.png)

Click the New Animation button and name it `explosion`. Set the Length to `0.64` and the Step to `0.01`. Now, click on the `Sprite` node and you'll notice that each property in the Inspector now has a key button next to it. Each time you click on the key, you create a keyframe in the current animation. The key button next to the Frame property also has a `+` symbol on it, indicating that it will automatically increment the value when you add a key frame.

Click the key and confirm that you want to create a new animation track. Note that the Frame property has incremented to `1`. Click the key button repeatedly until you have reached the final frame (`63`).

Click the Play button in the Animation panel to see the animation being played.

# Adding to Rock

In the `Rock` scene, add an instance of `Explosion` and add this line to `start()`:

```cpp
$Explosion.scale = Vector2(0.75, 0.75) * size
```

This will ensure that the explosion is scaled to match the rock's size.

Add a signal called `exploded` at the top of the script, then add the `explode()` function, which will be called when the bullet hits the rock:

```cpp
func explode():
    layers = 0
    $Sprite.hide()
    $Explosion/AnimationPlayer.play("explosion")
    emit_signal("exploded", size, radius, position, linear_velocity)
    linear_velocity = Vector2()
    angular_velocity = 0
```

The `layers` property ensures that the explosion will be drawn on top of the other sprites on the screen. Then, you will send a signal that will let `Main` know to spawn new rocks. This signal also needs to pass the necessary data so that the new rocks will have the right properties.

When the animation finishes playing, the `AnimationPlayer` will emit a signal. To connect it, you need to make the `AnimationPlayer` node visible. Right-click on the instanced Explosion and select Editable Children, then select the `AnimationPlayer` and connect its `animation_finished` signal. Make sure to select the `Rock` in the Connect to Node section. The end of the animation means it is safe to delete the rock:

```cpp
func _on_AnimationPlayer_animation_finished( name ):
    queue_free()
```

Now, test the game and check that you can see explosions when you shoot the rocks. At this point, your rock scene should look like this:

![](img/cae8a3e5-fe56-4935-9e81-36d0c6c3becb.png)

# Spawning smaller rocks

The `Rock` is emitting the signal, but it needs to be connected in `Main`. You can't use the Node tab to connect it, because the `Rock` instances are being created in code. Signals can be connected in code as well. Add this line to the end of `spawn_rock()`:

```cpp
r.connect('exploded', self, '_on_Rock_exploded')
```

This connects the rock's signal to a function in `Main` called `_on_Rock_exploded()`. Create that function, which will be called whenever a rock sends its `exploded` signal:

```cpp
func _on_Rock_exploded(size, radius, pos, vel):
    if size <= 1:
        return
    for offset in [-1, 1]:
        var dir = (pos - $Player.position).normalized().tangent() * offset
        var newpos = pos + dir * radius
        var newvel = dir * vel.length() * 1.1
        spawn_rock(size - 1, newpos, newvel)
```

In this function, two new rocks are created unless the rock that was just destroyed was the smallest size it can be. The `offset` loop variable will ensure that they spawn and travel in opposite directions (that is, one will be the negative of the other). The `dir` variable finds the vector between the player and the rock, then uses `tangent()` to find the perpendicular to that vector. This ensures that the new rocks travel away from the player:

![](img/db820982-ea51-4034-b9ac-631c5366ff93.png)

Play the game once again and check that everything is working as expected.

# UI

Creating a game UI can be very complex, or at least time-consuming. Precisely placing individual elements and ensuring they work on different-sized screens and devices is the least interesting part of game development for many programmers. Godot provides a wide variety of Control nodes to assist in this process. Learning how to use the various Control nodes will help lessen the pain of creating your game's UI.

For this game, you don't need a very complex UI. The game needs to provide the following information and interactions:

*   Start button
*   Status message (Get Ready or Game Over)
*   Score
*   Lives counter

The following is a preview of what you will be able to create:

![](img/3293fc90-2ae9-41da-8b77-176c6351824b.png)

Create a new scene, and add a `CanvasLayer` with the name `HUD` as its root node. The UI will be built on this layer by using Godot's `Control` Layout features.

# Layout

Godot's `Control` nodes include a number of specialized containers. These nodes can be nested inside each other to create the precise layout you need. For example, a `MarginContainer` will automatically add padding around its contents, while `HBoxContainer` and `VBoxContainer` organize their contents in rows or columns, respectively.

Start by adding a `MarginContainer`, which will hold the score and lives counter. Under the Layout menu, select Top Wide. Then, scroll down to the Custom Constants section and set all four margins to `20`.

Next, add an `HBoxContainer`, which will hold the score counter on the left and the lives counter on the right. Under this container, add a `Label` (name it `ScoreLabel`) and another `HBoxContainer` (name it `LivesCounter`).

Set the `ScoreLabel` Text to `0` and, under `Size Flags`, set Horizontal to Fill, Expand. Under Custom Fonts, add a `DynamicFont` like you did in [Chapter 1](fee8a22d-c169-454d-be5e-cf6c0bc78ddb.xhtml), *Introduction*, using `res://assets/kenvector_future_thin.ttf` from the `assets` folder and setting the size to `64`.

Under the `LivesCounter`, add a `TextureRect` and name it `L1`. Drag `res://assets/player_small.png` into the Texture property and set the Stretch Mode to Keep Aspect Centered. Make sure you have the `L1` node selected and press Duplicate (*Ctrl* + *D*) two times to create `L2` and `L3` (they'll be named automatically). During the game, the `HUD` will show/hide these three textures to indicate how many lives the user has left.

In a larger, more complicated UI, you could save this section as its own scene and embed it in other sections of the UI. However, this game only needs a few more pieces for its UI, so it's fine to combine them all in one scene.

As a child of the `HUD` node, add a `TextureButton` (named `StartButton`), a `Label` (named `MessageLabel`), and a `Timer` (named `MessageTimer`).

In the `res://assets` folder, there are two textures for the `StartButton`, one normal (`play_button.png`) and one to show when the mouse is hovering over it (`play_button_h.png`). Drag these to the Textures/Normal and Textures/Hover properties, respectively. In the Layout menu, choose Center.

For the `MessageLabel`, make sure you set the font first before specifying the layout, or it won't be centered properly. You can use the same settings you used for the `ScoreLabel`. After setting the font, set the layout to Full Rect.

Finally, set the One Shot property of `MessageTimer` to On and its Wait Time to `2`.

When finished, your UI's scene tree should look like this:

![](img/c2c4ec3b-cd27-40db-a688-918f1b9f602d.png)

# UI functions

You've completed the UI layout, so now let's add a script to `HUD` so you can add the functionality:

```cpp
extends CanvasLayer

signal start_game

onready var lives_counter = [$MarginContainer/HBoxContainer/LivesCounter/L1,
                             $MarginContainer/HBoxContainer/LivesCounter/L2,
                             $MarginContainer/HBoxContainer/LivesCounter/L3]
```

The `start_game` signal will be emitted when the player clicks the `StartButton`. The `lives_counter` variable is an array holding references to the three life counter images. The names are fairly long, so make sure to let the editor's autocomplete fill them in for you to avoid mistakes.

Next, you need functions to handle updating the displayed information:

```cpp
func show_message(message):
    $MessageLabel.text = message
    $MessageLabel.show()
    $MessageTimer.start()

func update_score(value):
    $MarginContainer/MarginContainer/HBoxContainer/ScoreLabel.text = str(value)

func update_lives(value):
    for item in range(3):
        lives_counter[item].visible = value > item
```

Each function will be called when a value changes to update the display.

Next, add a function to handle the `Game Over` state:

```cpp
func game_over():
    show_message("Game Over")
    yield($MessageTimer, "timeout")
    $StartButton.show()
```

Now, connect the `pressed` signal of the `StartButton` so that it can emit the signal to `Main`:

```cpp
func _on_StartButton_pressed():
    $StartButton.hide()
    emit_signal("start_game")
```

Finally, connect the `timeout` signal of `MessageTimer` so that it can hide the message:

```cpp
func _on_MessageTimer_timeout():
    $MessageLabel.hide()
    $MessageLabel.text = ''
```

# Main scene code

Now, you can add an instance of the `HUD` to the `Main` scene. Add the following variables to `Main.gd`:

```cpp
var level = 0
var score = 0
var playing = false
```

These will track the named quantities. The following code will handle starting a new game:

```cpp
func new_game():
    for rock in $Rocks.get_children():
        rock.queue_free()
    level = 0
    score = 0
    $HUD.update_score(score)
    $Player.start()
    $HUD.show_message("Get Ready!")
    yield($HUD/MessageTimer, "timeout")
    playing = true
    new_level()
```

First, you need to make sure that you remove any existing rocks that are left over from the previous game and initialize the variables. Don't worry about the `start()` function on the player; you'll add that soon.

After showing the `"Get Ready!"` message, you will use `yield` to wait for the message to disappear before actually starting the level:

```cpp
func new_level():
    level += 1
    $HUD.show_message("Wave %s" % level)
    for i in range(level):
        spawn_rock(3)
```

This function will be called every time the level changes. It announces the level number and spawns a number of rocks to match. Note—since you initialized `level` to `0,` this will set it to `1` for the first level.

To detect whether the level has ended, you continually check how many children the `Rocks` node has:

```cpp
func _process(delta):
    if playing and $Rocks.get_child_count() == 0:
        new_level()
```

Now, you need to connect the HUD's `start_game` signal (emitted when the Play button is pressed) to the `new_game()` function. Select the `HUD`, click on the Node tab, and connect the `start_game` signal. Set Make Function to Off and type `new_game` in the Method In Node field.

Next, add the following function to handle what happens when the game ends:

```cpp
func game_over():
    playing = false
    $HUD.game_over()
```

Play the game and check that pressing the Play button starts the game. Note that the `Player` is currently stuck in the `INIT` state, so you can't fly around yet—the `Player` doesn't know the game has started.

# Player code

Add a new signal and a new variable to `Player.gd`:

```cpp
signal lives_changed

var lives = 0 setget set_lives
```

The `setget` statement in GDScript allows you to specify a function that will be called whenever the value of a given variable is changed. This means that when `lives` decreases, you can emit a signal to let the `HUD` know it needs to update the display:

```cpp
func set_lives(value):
    lives = value
    emit_signal("lives_changed", lives)
```

The `start()` function is called by `Main` when a new game starts:

```cpp
func start():
    $Sprite.show()
    self.lives = 3
    change_state(ALIVE)
```

When using `setget`, if you access the variable locally (in the local script), you must put `self.` in front of the variable name. If you don't, the `setget` function will not be called.

Now, you need to connect this signal from the `Player` to the `update_lives` method in the `HUD`. In `Main`, click on the `Player` instance and find its `lives_changed` signal in the Node tab. Click Connect, and in the connection window, under Connect to Node, choose the `HUD`. For Method In Node, type `update_lives`. Make sure you have Make Function off, and click Connect, as shown in the following screenshot:

![](img/bb89e2c5-af5a-4ea2-92b7-407a9ed75cc4.png)

# Game over

In this section, you'll make the player detect when it is hit by rocks, add an invulnerability feature, and end the game when the player runs out of lives.

Add an instance of the `Explosion` to the `Player`, as well as a `Timer` node (named `InvulnerabilityTimer`). In the Inspector, set the Wait Time of `InvulnerabilityTimer` to `2` and its One Shot to On. Add this to the top of `Player.gd`:

```cpp
signal dead
```

This signal will notify the `Main` scene that the player has run out of lives and the game is over. Before that, however, you need to update the state machine to do a little more with each state:

```cpp
func change_state(new_state):
    match new_state:
        INIT:
            $CollisionShape2D.disabled = true
            $Sprite.modulate.a = 0.5
        ALIVE:
            $CollisionShape2D.disabled = false
            $Sprite.modulate.a = 1.0
        INVULNERABLE:
            $CollisionShape2D.disabled = true
            $Sprite.modulate.a = 0.5
            $InvulnerabilityTimer.start()
        DEAD:
            $CollisionShape2D.disabled = true
            $Sprite.hide()
            linear_velocity = Vector2()
            emit_signal("dead")
    state = new_state
```

The `modulate.a` property of a sprite sets its alpha channel (transparency). Setting it to `0.5` makes it semi-transparent, while `1.0` is solid. 

After entering the `INVULNERABLE` state, you start the `InvulnerabilityTimer`. Connect its `timeout` signal:

```cpp
func _on_InvulnerabilityTimer_timeout():
    change_state(ALIVE)
```

Also, connect the `animation_finished` signal from the `Explosion` animation like you did in the `Rock` scene:

```cpp
func _on_AnimationPlayer_animation_finished( name ):
    $Explosion.hide()
```

# Detecting collisions between physics bodies

When you fly around, the player ship bounces off the rocks, because both bodies are `RigidBody2D` nodes. However, if you want to make something happen when two rigid bodies collide, you need to enable contact monitoring. Select the `Player` node and in the Inspector, set Contact Monitoring to On. By default, no contacts are reported, so you must also set Contacts Reported to `1`. Now, the body will emit a signal when it contacts another body. Click on the Node tab and connect the `body_entered` signal:

```cpp
func _on_Player_body_entered( body ):
    if body.is_in_group('rocks'):
        body.explode()
        $Explosion.show()
        $Explosion/AnimationPlayer.play("explosion")
        self.lives -= 1
        if lives <= 0:
            change_state(DEAD)
        else:
            change_state(INVULNERABLE)
```

Now, go to the `Main` scene and connect the Player's `dead` signal to the `game_over()` function. Play the game and try running into a rock. Your ship should explode, become invulnerable (for two seconds), and lose one life. Check that the game ends if you get hit three times.

# Pausing the game

Many games require some sort of pause mode to allow the player to take a break in the action. In Godot, pausing is a function of the scene tree and can be set using `get_tree().paused = true`. When the `SceneTree` is paused, three things happen:

*   The physics thread stops running
*   `_process` and `_physics_process` are no longer called, so no code in those methods is run
*   `_input` and `_input_event` are also not called

When the pause mode is triggered, every node in the running game can react accordingly, based on how you've configured it. This behavior is set via the node's Pause/Mode property, which you'll find all the way at the bottom of the Inspector list.

The pause mode can be set to three values: `INHERIT` (the default value), `STOP`, and `PROCESS`. `STOP` means the node will cease processing while the tree is paused, while `PROCESS` sets the node to continue running, ignoring the paused state of the tree. Because it would be very tedious to set this property on every node in the whole game, `INHERIT` lets the node use the same pause mode as its parent.

Open the Input Map tab (in Project Settings) and create a new input action called `pause`. Choose a key you'd like to use to toggle pause mode; for example, P is a good choice.

Next, add the following function to `Main.gd` to respond to the input action:

```cpp
func _input(event):
    if event.is_action_pressed('pause'):
        if not playing:
            return
    get_tree().paused = not get_tree().paused
    if get_tree().paused:
        $HUD/MessageLabel.text = "Paused"
        $HUD/MessageLabel.show()
    else:
        $HUD/MessageLabel.text = ""
        $HUD/MessageLabel.hide()
```

If you ran the game now, you'd have a problem—all nodes are paused, including `Main`. This means that since it isn't processing `_input`, it can't detect the input again to unpause the game! To fix this, you need to set the Pause/Mode of `Main` to `PROCESS`. Now, you have the opposite problem: all the nodes below `Main` inherit this setting. This is fine for most of the nodes, but you need to set the mode to `STOP` on these three nodes: `Player`, `Rocks`, and `HUD`. 

# Enemies

Space is filled with more dangers than just rocks. In this section, you'll create an enemy spaceship that will periodically appear and shoot at the player.

# Following a path

When the enemy appears, it should follow a path across the screen. To keep it from looking too repetitive, you can create multiple paths and randomly choose one when the enemy starts.

Create a new scene and add a `Node`. Name it `EnemyPaths` and save the scene. To draw the path, add a `Path2D` node. As you saw earlier, this node allows you to draw a series of connected points. When you add the node, a new menu bar appears:

![](img/f219bd0e-6e72-4d6b-bcf1-f36c6f87d719.png)

These buttons let you draw and modify the path's points. Click the one with the + symbol to add points. Click to start the path somewhere just outside the game window (the bluish-purple rectangle), and then click a few more points to create a curve. Don't worry about making it smooth just yet:

![](img/abefc450-e9d5-4d01-8549-2d4f26e64bc7.png)

When the enemy ship follows the path, it will not look very smooth when it hits the sharp corners. To smooth the curve, click the second button in the path toolbar (its tooltip says Select Control Points). Now, if you click and drag any of the curve's points, you will add a control point that allows you to angle and curve the line. Smoothing the preceding line results in something like this:

![](img/2fc0ce46-0c81-475c-9576-628f03378275.png)

Add a few more `Path2D` nodes to the scene and draw the paths however you like. Adding loops and curves rather than straight lines will make the enemy look more dynamic (and make it harder to hit). Remember that the first point you click will be the start of the path, so make sure to place them on different sides of the screen, for variety. Here are three example paths:

![](img/669260e5-364c-4cac-a15f-00ae545ad22e.png)

Save the scene. You'll add this to the enemy's scene to give it the paths it can follow.

# Enemy scene

Create a new scene for the Enemy, using an `Area2D` as its root node. Add a `Sprite` and use `res://assets/enemy_saucer.png` as its Texture. Set the Animation/HFrames to `3` so that you can choose between the different-colored ships:

![](img/b50b6fcd-0d06-441c-887c-912a9e9b5a3f.png)

As you've done before, add a `CollisionShape2D` and give it a `CircleShape2D` scaled to cover the sprite image. Next, add an instance of the `EnemyPaths` scene and an `AnimationPlayer`. In the `AnimationPlayer`, you'll need two animations: one to make the saucer spin as it moves, and the other to create a flash effect when the saucer is hit:

*   **Rotate animation**: Add a new animation named `rotate` and set its *Length* to `3`. Add a keyframe for the `Sprite` Transform/Rotation Degrees property after setting it to `0`, then drag the play bar to the end and add a keyframe with the rotation set to `360`. Click the Loop button and the Autoplay button.

*   **Hit animation**: Add a second animation named `flash`. Set its *Length* to `0.25` and the *Step* to `0.01`. The property you'll be animating is the Sprite's Modulate (found under *Visibility*). Add a keyframe for Modulate to create the track, then move the scrubber to `0.04` and change the Modulate color to red. Move forward another `0.04` and change the color back to white. 

Repeat this process two more times so that you have three flashes in total.

Add an instance of the `Explosion` scene as you did with the other objects. Also, like you did with the rocks, connect the explosion's `AnimationPlayer` `animation_finished` signal and set it to delete the enemy when the explosion finishes:

```cpp
func _on_AnimationPlayer_animation_finished(anim_name):
    queue_free()
```

Next, add a `Timer` node called `GunTimer` that will control how often the enemy shoots at the player. Set its Wait Time to `1.5` and Autostart to `On`. Connect its `timeout` signal, but leave the code reading `pass` for now.

Finally, click on the `Area2D` and the Node tab and add it to a group called `enemies`. As with the rocks, this will give you a way to identify the object, even if there are multiple enemies on the screen at the same time.

# Moving the Enemy

Attach a script to the `Enemy` scene. To begin, you'll make the code that will select a path and move the enemy along it:

```cpp
extends Area2D

signal shoot

export (PackedScene) var Bullet
export (int) var speed = 150
export (int) var health = 3

var follow
var target = null

func _ready():
    $Sprite.frame = randi() % 3
    var path = $EnemyPaths.get_children()[randi() % $EnemyPaths.get_child_count()]
    follow = PathFollow2D.new()
    path.add_child(follow)
    follow.loop = false
```

A `PathFollow2D` node is one that can automatically move along a parent `Path2D`. By default, it is set to loop around the path, so you need to manually set the property to `false`.

The next step is to move along the path:

```cpp
func _process(delta):
    follow.offset += speed * delta
    position = follow.global_position
    if follow.unit_offset > 1:
        queue_free()
```

You can detect the end of the path when `offset` is greater than the total path length. However, it's more straightforward to use `unit_offset`, which varies from zero to one over the length of the path.

# Spawning enemies

Open the `Main` scene and add a `Timer` node called `EnemyTimer`. Set its One Shot property to `On`. Then, in `Main.gd`, add a variable to reference your enemy scene (drag it into the Inspector after saving the script):

```cpp
export (PackedScene) var Enemy
```

Add the following code to `new_level()`:

```cpp
$EnemyTimer.wait_time = rand_range(5, 10)
$EnemyTimer.start()
```

Connect the `EnemyTimer` `timeout` signal, and add the following:

```cpp
func _on_EnemyTimer_timeout():
    var e = Enemy.instance()
    add_child(e)
    e.target = $Player
    e.connect('shoot', self, '_on_Player_shoot')
    $EnemyTimer.wait_time = rand_range(20, 40)
    $EnemyTimer.start()
```

This code instances the enemy whenever the `EnemyTimer` times out. When you add shooting to the enemy, it will use the same process you used for the `Player`, so you can reuse the same bullet-spawning function, which is `_on_Player_shoot()`.

Play the game, and you should see a flying saucer appear that will fly along one of your paths.

# Enemy shooting and collisions

The enemy needs to shoot at the player as well as react when hit by the player or the player's bullets.

Open the `Bullet` scene and choose Save Scene As to save it as `EnemyBullet.tscn` (afterwards, don't forget to rename the root node as well). Remove the script by selecting the root node and clicking the Clear the script button:

![](img/726a6bf4-6045-426c-8276-d91453272618.png) 

You also need to disconnect the signal connections by clicking the Node tab and choosing Disconnect:

![](img/45e9e2f5-6089-4049-b5e5-1cd63a8b22da.png)

There is also a different texture in the `assets` folder you can use to make the enemy bullet appear distinct from the player's.

The script will be very much the same as the regular bullet. Connect the area's `body_entered` signal and the `screen_exited` signal of `VisibilityNotifier2D`:

```cpp
extends Area2D

export (int) var speed

var velocity = Vector2()

func start(_position, _direction):
    position = _position
    velocity = Vector2(speed, 0).rotated(_direction)
    rotation = _direction

func _process(delta):
    position += velocity * delta

func _on_EnemyBullet_body_entered(body):
    queue_free()

func _on_VisibilityNotifier2D_screen_exited():
    queue_free()
```

For now, the bullet won't do any damage to the player. You'll be adding a shield to the player in the next section, so you can add that at the same time.

Save the scene and drag it into the Bullet property on the `Enemy`. 

In `Enemy.gd`, add the `shoot` function:

```cpp
func shoot():
    var dir = target.global_position - global_position
    dir = dir.rotated(rand_range(-0.1, 0.1)).angle()
    emit_signal('shoot', Bullet, global_position, dir)
```

First, you must find the vector pointing to the player's position, then add a little bit of randomness to it so that the bullets don't follow exactly the same path.

For an extra challenge, you can make the enemy shoot in *pulses*, or multiple rapid shots:

```cpp
func shoot_pulse(n, delay):
    for i in range(n):
        shoot()
        yield(get_tree().create_timer(delay), 'timeout')
```

This function creates a given number of bullets with `delay` time between them. You can use this whenever the `GunTimer` triggers a shot:

```cpp
func _on_GunTimer_timeout():
    shoot_pulse(3, 0.15)
```

This will shoot a pulse of `3` bullets with `0.15` seconds between them. Tough to dodge!

Next, the enemy needs to take damage when it's hit by a shot from the player. It will flash using the animation you made, and then explode when its health reaches `0`. 

Add these functions to `Enemy.gd`:

```cpp
func take_damage(amount):
    health -= amount
    $AnimationPlayer.play('flash')
    if health <= 0:
        explode()
    yield($AnimationPlayer, 'animation_finished')
    $AnimationPlayer.play('rotate')

func explode():
    speed = 0
    $GunTimer.stop()
    $CollisionShape2D.disabled = true
    $Sprite.hide()
    $Explosion.show()
    $Explosion/AnimationPlayer.play("explosion")
    $ExplodeSound.play()
```

Also, connect the area's `body_entered` signal so the enemy will explode if the player runs into it:

```cpp
func _on_Enemy_body_entered(body):
    if body.name == 'Player':
        pass
    explode()
```

Again, you're waiting for the player shield to add the damage to the player, so leave the `pass` placeholder there for now.

Right now, the player's bullet is only detecting physics bodies because its `body_entered` signal is connected. However, the enemy is an `Area2D`, so it will not trigger that signal. To detect the enemy, you need to also connect the  `area_entered` signal:

```cpp
func _on_Bullet_area_entered(area):
    if area.is_in_group('enemies'):
        area.take_damage(1)
    queue_free()
```

Try playing the game again and you'll be doing battle with an aggressive alien opponent! Verify that all the collision combinations are being handled. Also note that the enemy's bullets can be blocked by rocks—maybe you can hide behind them for cover!

# Additional features

The structure of the game is complete. You can start the game, play it through, and when it ends, play again. In this section, you'll add some additional effects and features to the game to improve the gameplay experience. Effects is a broad term and can mean many different techniques, but in this case, you'll specifically address three things:

*   **Sound effects and music: **Audio is very often overlooked, but can be a very effective part of game design. Good sound improves the *feel* of the game. Bad or annoying sounds can create boredom or frustration. You'll add some action-packed background music, and some sound effects for several actions in the game.

*   **Particles: **Particle effects are images, usually small, that are generated in large numbers and animated by a particle system. They can be used for a countless number of impressive visual effects. Godot's particle system is quite powerful; too powerful to fully explore here, but you'll learn enough to get started experimenting with it.

*   **Player shield: **If you're finding the game too hard, especially on higher levels where there are a lot of rocks, adding a shield to the player will greatly increase your chances of survival. You can also make larger rocks do more damage to the shield than smaller ones. You'll also make a nice display bar on the HUD to show the player's remaining shield level.

# Sound/music

In the `res://assets/sounds` folder are several audio files containing different sounds in the OggVorbis format. By default, Godot sets `.ogg` files to loop when imported. In the case of `explosion.ogg`, `laser_blast.ogg`, and `levelup.ogg`, you don't want the sounds to loop, so you need to change the import settings for those files. To do this, select the file in the FileSystem dock, and then click the Import tab located next to the Scene tab on the right-hand side of the editor window. Uncheck the box next to Loop and click Reimport. Do this for each of the three sounds. Refer to the following screenshot:

![](img/a9faaf00-0080-4d33-98f4-5afbebcb43f9.png)

To play a sound, it needs to be loaded by an `AudioStreamPlayer` node. Add two of these nodes to the `Player` scene, naming them `LaserSound` and `EngineSound`. Drag the respective sound into each node's Stream property in the Inspector. To play the sound when shooting, add the following line to `shoot()` in `Player.gd`:

```cpp
$LaserSound.play()
```

Play the game and try shooting. If you find the sound a bit too loud, you can adjust the Volume Db property. Try a value of `-10`.

The engine sound works a little differently. It needs to play when the thrust is on, but if you try to just `play()` the sound in the `get_input()` function, it will restart the sound every frame as long as you have the input pressed. This doesn't sound good, so you only want to start playing the sound if it isn't already playing. Here is the relevant section from the `get_input()` function:

```cpp
if Input.is_action_pressed("thrust"):
    thrust = Vector2(engine_power, 0)
    if not $EngineSound.playing:
        $EngineSound.play()
 else:
     $EngineSound.stop()
```

Note that a problem can occur—if the player dies while holding down the thrust key, the engine sound will remain stuck on. This can be solved by adding `$EngineSound.stop()` to the `DEAD` state in `change_state()`.

In the `Main` scene, add three more `AudioStreamPlayer` nodes: `ExplodeSound`, `LevelupSound`, and `Music`. In their Stream properties, drop `explosion.ogg`, `levelup.ogg`, and `Funky-Gameplay_Looping.ogg`.

Add `$ExplodeSound.play()` as the first line of `_on_Rock_exploded()`, and add `$LevelupSound.play()` to `new_level()`.

To start/stop the music, add `$Music.play()` to `new_game()` and `$Music.stop()` to `game_over()`.

The Enemy also needs an `ExplodeSound` and a `ShootSound`. You can use the same explosion as the player, but there is an `enemy_laser.wav` sound to use for the shot.

# Particles

The player ship's thrust is a perfect use for particles, creating a streaming flame from the engine. Add a `Particles2D` node to the `Player` scene and name it `Exhaust`. You might want to zoom in on the ship image while you're doing this part.

When first created, the `Particles2D` node has a warning: *A material to process the particles is not assigned*. Particles will not be emitted until you assign a `Process Material` in the Inspector. Two types of materials are possible: `ShaderMaterial` and `ParticlesMaterial`. `ShaderMaterial` allows you to write shader code in a GLSL-like language, while `ParticlesMaterial` is configured in the Inspector. Next to Particles Material, click the down-arrow and choose New ParticlesMaterial.

You'll see a line of white dots streaming down from the center of the player ship. Your challenge now is to turn those into an exhaust flame.

There are a very large number of properties to choose from when configuring particles, especially under `ParticlesMaterial`. Before starting on that, set these properties of the `Particles2D`:

*   Amount: `25`
*   Transform/Position*: *`(-28, 0)`
*   Transform/Rotation: `180`
*   Visibility/Show Behind Parent: `On`

Now, click on the `ParticlesMaterial`. This is where you'll find the majority of the properties that affect the particles' behavior. Start with Emission Shape—change it to Box. This will reveal Box Extents, which should be set to `(1, 5, 1)`. Now, the particles are emitted over a small area instead of a single point.

Next, set Spread/Spread to `0` and Gravity/Gravity to `(0, 0, 0)`. Now, the particles aren't falling or spreading out, but they are moving very slowly.

The next property is Initial Velocity. Set Velocity to `400`. Then, scroll down to Scale and set it to `8`.

To make the size change over time, you can set a Scale Curve. Click on New CurveTexture and click on it. A new panel labeled Curve will appear. The left-hand dot represents the starting scale, and the right-hand dot represents the end. Drag the right-hand dot down until your curve looks something like this:

![](img/58feb303-62b9-4398-acc6-235920927aac.png)

Now, the particles are shrinking as they age. Click the left arrow at the top of the Inspector to go back to the previous section.

The final section to adjust is Color. To make the particles appear like a flame, the particles should start out a bright orange-yellow and shift to red while fading out. In the Color Ramp property, click on New GradientTexture. Then, in the Gradient property, choose New Gradient:

![](img/363872aa-232a-46a0-b454-55ba94fc2e84.png)

The sliders labeled 1 and 2 select the starting and ending colors, while 3 shows what color is set on the currently selected slider. Click on slider 1 and then click 3 to choose an orange color, then click on slider 2 and set it to a deep red.

Now that we can see what the particles are doing, they are lasting far too long. Go back to the `Exhaust` node and change the Lifetime to `0.1`.

Hopefully, your ship's exhaust looks somewhat like a flame. If it doesn't, feel free to adjust the `ParticlesMaterial` properties until you are happy with it.

Now that the ship's `Exhaust` is configured, it needs to be turned on/off based on the player input. Go to the player script and add `$Exhaust.emitting = false` at the beginning of `get_input()`. Then, add `$Exhaust.emitting = true` under the `if` statement that checks for thrust input.

# Enemy trail

You can also use particles to make a trail effect behind the enemy. Add a `Particles2D` to the enemy scene and set the properties as follows:

*   Amount: `20`
*   Local Coords: `Off`
*   Texture: `res://assets/corona.png`
*   Show Behind Parent: `On`

Note that the effect texture you're using is white on a black background. This image needs its blend mode changed. To do this, on the particle node, find the Material property (it is in the `CanvasItem` section). Select New CanvasItemMaterial and, in the resulting material, change the Blend Mode to `Add`.

Now, create a `ParticlesMaterial` like you did previously, and use these settings:

*   Emission Shape:

    *   Shape: Box
    *   Box Extents: (`25`, `25`, `1`)
*   Spread: `25`
*   Gravity: (0, 0, 0)

Now, create a `ScaleCurve` like you did for the player exhaust. This time, make the curve look something like the following:

![](img/2be98718-8357-43c4-a5cf-2d12d192aa25.png)

Try running the game and see how it looks. Feel free to tinker with the settings until you have something you like.

# Player shield

In this section, you'll add a shield to the player and a display element to the `HUD` showing the current shield level. 

First, add the following to the top of the `Player.gd` script:

```cpp
signal shield_changed

export (int) var max_shield
export (float) var shield_regen

var shield = 0 setget set_shield
```

The `shield` variable will work similarly to `lives`, emitting a signal to the `HUD` whenever it changes. Save the script and set `max_shield` to `100` and `shield_regen` to `5` in the Inspector.

Next, add the following function, which handles changing the shield's value:

```cpp
func set_shield(value):
    if value > max_shield:
        value = max_shield
    shield = value
    emit_signal("shield_changed", shield/max_shield)
    if shield <= 0:
        self.lives -= 1
```

Also, since some things, such as regeneration, may add to the shield's value, you need to make sure it doesn't go above the maximum allowed value. Then, when you send the `shield_changed` signal, you pass the ratio of `shield/max_shield`. This way, the HUD's display doesn't need to know anything about the actual values, just the shield's relative state.

Add this line to `start()` and to `set_lives()`:

```cpp
    self.shield = max_shield
```

Hitting a rock will damage the shield, and bigger rocks should do more damage:

```cpp
func _on_Player_body_entered( body ):
    if body.is_in_group('rocks'):
        body.explode()
        $Explosion.show()
        $Explosion/AnimationPlayer.play("explosion")
        self.shield -= body.size * 25
```

The enemy's bullets should also do damage, so make this change to `EnemyBullet.gd`:

```cpp
func _on_EnemyBullet_body_entered(body):
    if body.name == 'Player':
        body.shield -= 15
    queue_free()
```

Also, running into the enemy should damage the player, so update this in `Enemy.gd`:

```cpp
func _on_Enemy_body_entered(body):
    if body.name == 'Player':
        body.shield -= 50
        explode()
```

The last addition to the player script is to regenerate the shield each frame. Add this line to `_process()`:

```cpp
    self.shield += shield_regen * delta
```

The next step is to add the display element to the `HUD`. Rather than display the shield's value in a `Label`, you'll use a `TextureProgress` node. This is a `Control` node that is a type of `ProgressBar`: a node that displays a given value as a filled bar. The `TextureProgress` node allows you to assign a texture to be used for the bar's display.

In the existing `HBoxContainer`, add `TextureRect` and `TextureProgress`. Place them after the `ScoreLabel` and before the `LivesCounter`. Change the name of the `TextureProgress` to ShieldBar. Your node setup should look like this:

![](img/d0792479-4e87-481b-8af5-87fb0ff32efa.png)

Drag the `res://assets/shield_gold.png` texture into the *Texture* property of `TextureRect`. This will be an icon indicating what the bar is displaying.

The ShieldBar has three texture properties: Under, Over, and Progress. Progress is the texture that will be displayed as the bar's value. Drag `res://assets/barHorizontal_green_mid 200.png` into this property. The other two texture properties allow you to customize the appearance by setting an image to be drawn below or above the progress texture. Drag `res://assets/glassPanel_200.png` into the *Over* texture property.

In the *Range* section, you can set the numeric properties of the bar. Min Value and Max Value should be set to `0` and `100`, as this bar will be showing the percentage value of the shield, not its raw value. Value is the property that controls the currently displayed fill value. Change it to `75` to see the bar partly filled. Also, set its Horizontal size flags to Fill, Expand.

Now, you can update the HUD script to control the shield bar. Add these variables at the top:

```cpp
onready var ShieldBar = $MarginContainer/HBoxContainer/ShieldBar
var red_bar = preload("res://assets/barHorizontal_red_mid 200.png")
var green_bar = preload("res://assets/barHorizontal_green_mid 200.png")
var yellow_bar = preload("res://assets/barHorizontal_yellow_mid 200.png")
```

In addition to the green bar texture, you also have red and yellow bars in the `assets` folder. This will allow you to change the shield's color as the value decreases. Loading the textures in this way makes them easier to access later in the script when you want to assign the appropriate image to the `TextureProgress` node: 

```cpp
func update_shield(value):
    ShieldBar.texture_progress = green_bar
    if value < 40:
        ShieldBar.texture_progress = red_bar
    elif value < 70:
        ShieldBar.texture_progress = yellow_bar
    ShieldBar.value = value
```

Lastly, click on the `Main` scene's `Player` node and connect the `shield_changed` signal to the `update_shield()` function you just created. Run the game and verify that you can see the shield and that it is working. You may want to increase or decrease the regeneration rate to adjust it to a speed you like.

# Summary

In this chapter, you learned how to work with `RigidBody2D` nodes and learned more about how Godot's physics works. You also implemented a basic Finite State Machine—something you'll find more and more useful as your projects grow larger. You saw how `Container` nodes help organize and keep UI nodes aligned. Finally, you added some sound effects and got your first taste of advanced visual effects by using the `AnimationPlayer` and `Particles2D` nodes.

You also created a number of game objects using the standard Godot hierarchies, such as `CollisionShapes` being attached to `CollisionObjects`. At this point, some of these node configurations should be starting to look familiar to you. 

Before moving on, look through the project again. Play it. Make sure you understand what each scene is doing, and read through the scripts to review how everything connects together. 

In the next chapter, you'll learn about kinematic bodies, and use them to create a side-scrolling platform game.