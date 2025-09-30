# 3D Minigolf

The previous projects in this book have been designed in 2D space. This is intentional, in order to introduce the various features and concepts of Godot while keeping the projects' scopes limited. In this chapter, you'll venture into the 3D side of game development. For some, 3D development feels significantly more difficult to manage; for others, it is more straightforward. In either case, there is certainly an additional layer of complexity for you to understand.

If you've never worked with any kind of 3D software before, you may find yourself encountering many new concepts. This chapter will explain them as much as possible, but remember to refer to the Godot documentation whenever you need a more in-depth understanding of a particular topic.

The game you'll make in this chapter is called **Minigolf**. This will consist of a small customizable course, a ball, and an interface for aiming and shooting the ball towards the hole.

This is what you'll learn in this chapter:

*   Navigating Godot's 3D editor
*   The Spatial node and its properties
*   Importing 3D meshes and using 3D collision shapes
*   How to use 3D cameras, both stationary and moving
*   Using GridMap to place the tiles of your golf course
*   Setting up lighting and the environment
*   An introduction to PBR rendering and materials

But first, here's a brief introduction to 3D in Godot.

# Introduction to 3D

One of the strengths of Godot is its ability to handle both 2D and 3D games. While much of what you've learned earlier in this book applies equally well in 3D (nodes, scenes, signals, and so on), changing from 2D to 3D brings with it a whole new layer of complexity and capabilities. First, you'll find that there are some additional features available in the 3D editor window, and it's a good idea to familiarize yourself with how to navigate in the 3D editor window.

# Orienting in 3D space

When you click on the 3D button at the top of the editor window, you will see the 3D project view:

![](img/3cc33fd0-c7e4-46fb-840a-168374ee6f9d.png)

The first thing you should notice is the three colored lines in the center. These are the *x* (red), *y* (green), and *z* (blue) axes. The point where they meet is the origin, with coordinates of `(0, 0, 0)`.

Just as you used `Vector2(x, y)` to indicate a position in two-dimensional space, `Vector3(x, y, z)` describes a position in three dimensions along these three axes.

One issue that arises when working in 3D is that different applications use different conventions for orientation. Godot uses Y-Up orientation, so when looking at the axes, if *x* is pointing to the left/right, then *y* is up/down, and *z* is forward/back. You may find when using other popular 3D software that they use Z-Up. It's good to be aware of this, as it can lead to confusion when moving between different programs.

Another major aspect to be aware of is the unit of measure. In 2D, everything is measured in pixels, which makes sense as a natural basis for measurement when drawing on the screen. However, when working in 3D space, pixels aren't really useful. Two objects of exactly the same size will occupy different areas on the screen depending on how far away they are from the camera (more about cameras soon). For this reason, in 3D space all objects in Godot are measured in generic units. You're free to call these units whatever you like: meters, inches, or even light years, depending on the scale of your game world.

# Godot's 3D editor

Before getting started with 3D, it will be useful to briefly review how to navigate in Godot's 3D space. The camera is controlled with the mouse and keyboard:

*   Mousewheel up/down: Zoom in/out
*   Middle button + drag: Orbit the camera around the current target
*   *Shift* + middle button + drag: Pan camera up/down/left/right
*   Right-click + drag: Rotate camera in place

If you're familiar with popular 3D games such as *Minecraft*, you can press *Shift* + *F* to switch to Freelook mode. In this mode, you can use the WASD keys to *fly* around the scene while aiming with the mouse. Press *Shift* + *F* again to exit Freelook mode. 

You can also alter the camera's view by clicking on the [ Perspective ] label in the upper-left corner. Here, you can snap the camera to a particular orientation such as Top View or Front View:

![](img/ed351884-c48f-48ac-aae1-f6ca5873f778.png)

This can be especially useful on large displays when combined with the use of multiple Viewports. Click the View menu and you can split the screen into multiple views of the space, allowing you to see an object from all sides simultaneously.

Note that each of these menu options has a keyboard shortcut associated with it. You can click on Editor *|* Editor Settings *|* 3D to adjust the 3D navigation and shortcuts to your liking.

![](img/7744c74b-511b-4a3b-9d16-532a62701bbc.png)

When using multiple viewports, each can be set to a different perspective so you can see the effect of your actions from multiple directions at the same time:

![](img/6f0a2741-d6d6-4fda-9881-de32f4495678.png)

# Adding 3D objects

It's time to add your first 3D node. Just as all 2D nodes inherit from `Node2D`, which provides properties such as `position` and `rotation`, 3D nodes inherit from the `Spatial` node. Add one to the scene and you'll see the following:

![](img/dd3ead8a-ce47-49dd-8f74-b434e0249f12.png)

That colorful object you see is not the node, but rather a 3D *gizmo*. Gizmos are tools that allow you to move and rotate objects in space. The three rings control rotation, while the three arrows move (translate) the object along the three axes. Notice that the rings and arrows are color-coded to match the axis colors. The arrows move the object *along* the respective axis, while the rings rotate the object *around* a particular axis. There are also three small squares that lock one axis and allow you to move the object in a single plane.

Take a few minutes to experiment and get familiar with the gizmo. Use Undo if you find yourself getting lost.

Sometimes, gizmos get in the way. You can click on the mode icons to restrict yourself to only one type of transformation: move, rotate, or scale:

![](img/352310f8-b426-416f-8a3d-fbb189188c0c.png)

The *Q*, *W*, *E*, and *R* keys are shortcuts for these buttons, allowing for quickly changing between modes.

# Global versus Local Space

By default, the gizmo controls operate in global space. Try rotating the object. No matter how you turn it, the gizmo's movement arrows still point along the axes. Now try this: put the `Spatial` node back to its original position and orientation (or delete it and add a new one). Rotate the object around one axis, then click the Local Space Mode (T) button:

![](img/f9ccf729-ad6f-4e1b-a7b6-05fb0cfc27af.png)

Observe what happened to the gizmo arrows. They now point along the *object's* local *x*/*y*/*z* axes and not the world's. When you click and drag them, they will move the object relative to its axes. Switching back and forth between these two modes can make it much easier to place an object exactly where you want it.

# Transforms

Look at the Inspector for your `Spatial` node. Instead of a Position property, you now have Translation, as well as Rotation Degrees and Scale. As you move the object around, observe how these values change. Note that the Translation represents the object's coordinates relative to the origin:

![](img/b773736e-a608-4ef9-90fb-1c48fcbe993c.png)

You'll also notice a Transform property, which also changes as you move and rotate the object. When you change translation or rotation, you'll notice that the 12 transform quantities will change as well.

A full explanation of the math behind transforms is beyond the scope of this book, but in a nutshell, a transform is a *matrix* that describes an object's translation, rotation, and scale all at once. You briefly used the 2D equivalent in the Space Rocks game earlier in this book, but the concept is more widely applied in 3D.

# Transforms in code

When positioning a 3D node via code, you have access to its `transform` and `global_transform` properties, which are `Transform` objects. A `Transform` has two sub-properties: `origin` and `basis`. The `origin` represents the body's offset from its parent's origin or the global origin, respectively. The `basis` property contains three vectors that define a local coordinate system traveling with the object. Think of the three axis arrows in the gizmo when you are in Local Space mode.

You'll see more about how to use 3D transforms later in this section.

# Meshes

Just like `Node2D`, a `Spatial` node has no size or appearance of its own. In 2D, you added a Sprite to assign a texture to the node. In 3D, you need to add a *mesh*. A mesh is a mathematical description of a shape. It consists of a collection of points, called *vertices*. These vertices are connected by lines, called *edges,* and multiple edges (at least three) together make a *face*:

![](img/7336961e-d3d3-4a62-b20d-63b848ba37ea.png)

A cube, for example, is composed of eight vertices, twelve edges, and six faces.

If you've ever used 3D design software, this will be very familiar to you. If you haven't, and you're interested in learning about 3D modeling, Blender is a very popular open source tool for designing 3D objects. You can find many tutorials and lessons on the internet to help you get started with Blender.

# Importing meshes

Whatever modeling software you may use, you will need to export your models in a format that is readable by Godot. Wavefront (`.obj`) and Collada (`.dae`) are the most popular. Unfortunately, if you're using Blender, its Collada exporter has some flaws that make it unusable with Godot. To fix this, Godot's developers have created a Blender plugin called **Better Collada Exporter** that you can download from [https://godotengine.org/download](https://godotengine.org/download).

If your objects are in another format, such as FBX, you'll need to use a converter tool to save them as OBJ or DAE in order to use them with Godot.

A new format called GLTF is gaining in popularity and has some significant advantages over Collada. Godot already supports it, so feel free to experiment with any models you may find in this format.

# Primitives

If you don't have any models handy, or if you just need a simple model quickly, Godot has the ability to create certain 3D meshes directly. Add a `MeshInstance` node as a child of Spatial, and in the Inspector, click the Mesh property:

.![](img/952db26a-7939-4067-a644-408dbec5eda0.png)

These predefined shapes are called *primitives* and they represent a handy collection of common useful shapes. You can use these shapes for a variety of purposes, as you'll see later in this chapter. Select New CubeMesh and you'll see a plain cube appear on the screen. The cube itself is white, but it may appear bluish on your screen due to the default ambient light in the 3D editor window. You'll learn how to work with lighting later in this chapter.

# Multiple meshes

Often, you'll find yourself with an object composed of many different meshes. A character might have separate meshes for its head, torso, and limbs. If you have a great many of these types of objects, it can lead to performance issues as the engine tries to render so many meshes. As a result, `MultiMeshInstance` is designed to provide a high-performance method of grouping many meshes together into a single object. You probably don't need it yet, because it won't be necessary for this project, but keep it in mind as a tool that may come in handy later.

# Cameras

Try running the scene with your cube mesh. Where is it? In 3D, you won't see anything in the game viewport without using a `Camera`. Add one, and use the camera's gizmo to position and point it towards the cube, as in the following screenshot:

![](img/9e5a6587-8a37-4f85-8374-184e1e7b1307.png)

The pinkish-purple, pyramid-shaped object is called the camera's *fustrum*. It represents the camera's view, and can be made narrow or wide to affect the camera's *field of view*. The triangular arrow at the top of the fustrum is the camera's up direction. 

As you're moving the camera around, you can use the Preview button in the upper-right to check your aim. Preview will always show you what the selected camera can see.

As with the `Camera2D` you used earlier, a `Camera` must be set as Current for it to be active. Its other properties affect how it *sees*: field of view, projection, and near/far. The default values of these properties are good for this project, but go ahead and experiment with them to see how they affect the view of the cube. Use Undo to return everything to the default values when you're done.

# Project setup

Now that you've learned how to navigate in Godot's 3D editor, you're ready to start on the Minigolf project. As with the other projects, download the game assets from the following link and unzip them in your project folder. The unzipped `assets` folder contains images, 3D models, and the other assets you need to complete the project. You can download a Zip file of the art and sounds (collectively known as *assets*) for the game here, [https://github.com/PacktPublishing/Godot-Game-Engine-Projects/releases](https://github.com/PacktPublishing/Godot-Game-Engine-Projects/releases).

This game will use the left mouse button as an input. The Input Map does not have any default actions defined for this, so you need to add one. Open Project | Project Settings and go to the Input Map tab. Add a new action called click, then click the plus to add a Mouse Button event to it. Choose Left Button:

![](img/7dc5e9bd-2341-45ef-80fb-6056ce0cde98.png)

# Creating the course

For the first scene, add a node called `Main` to serve as your scene's root. This scene will contain the major parts of the game, starting with the course itself. Start by adding a `GridMap` node to lay out the course.

# GridMaps

`GridMap` is the 3D equivalent of the `TileMap` node you used in earlier projects. It allows you to use a collection of meshes (contained in a `MeshLibrary`) and lay them out in a grid to more quickly design an environment. Because it is 3D, you can stack the meshes in any direction, although for this project, you'll stick to the same plane.

# Making a MeshLibrary

The `res://assets` folder contains a pre-generated `MeshLibrary` for the project, containing all the necessary course parts along with collision shapes. However, if you need to change it or make your own, you'll find the procedure is very similar to how `TileSet` is created in 2D.

The scene used to create the pre-generated `MeshLibrary` can also be found in the `res://assets` folder. Its name is `course_tiles_edit1.tscn`. Feel free to open it and look at how it is set up.

Start by making a new scene, with a `Spatial` as its root. To this node, add any number of `MeshInstance`. You can find the original course meshes, exported from Blender, in the `res://assets/dae` folder.

The names you give to these nodes will be their names in the `MeshLibrary`.

Once you have added the meshes, they need static collision bodies added to them. Creating collision shapes that match a given mesh can be complicated, but Godot has a method of automatically generating them.

Select a mesh and you'll see a `Mesh` menu appear at the top of the editor window:

![](img/1bd9dccb-57f8-4e68-88b4-b23c7e31e76e.png)

Select Create Trimesh Static Body and Godot will create a `StaticBody` and add a `CollisionShape` using the mesh's data:

![](img/f09ad4f0-97eb-4e95-9d6f-197aa13a457b.png)

Do this with each of your mesh objects, and then select Scene | Convert To | MeshLibrary to save the resource.

# Drawing the course

Drag the `MeshLibrary` (`res://assets/course_tiles.tres` or the one you created) into the Theme property of `GridMap` in the Inspector. Also, check that the Cell/Size property is set to `(2, 2, 2)`:

![](img/dd63475f-7f2f-47f9-b81b-7f2ade347670.png)

Try drawing by selecting the tile piece from the list on the right and placing it by left-clicking in the editor window. You can rotate a piece around the y axis by pressing *S*. To remove a tile, use *Shift* + right-click.

For now, stick to a simple course; you can get fancy later when everything is working. Don't forget the hole!

![](img/380ed00a-aa05-40e3-9a2b-3d494debc718.png)

Now, it's time to see what this is going to look like when the game is run. Add a `Camera` to the scene. Move it up and angle it so it looks down on the course. Remember, you can use the Preview button to check what the camera sees.

Run the scene. You'll see that everything seems very dark. By default, there is minimal environmental light in the scene. To see more clearly, you need to add more light.

# WorldEnvironment

Lighting is a complex subject all on its own. Deciding where to place lights and how to set their color and intensity can dramatically affect how a scene looks. 

Godot provides three lighting nodes in 3D:

*   `OmniLight`: For light that is emitted in all directions, like from a light bulb or candle
*   `DirectionalLight`: Infinite light from a distant source, such as sunlight
*   `SpotLight`: Directional light from a single source, such as a flashlight

In addition to using individual lights, you can also set an *ambient* light using `WorldEnvironment`.

Add a `WorldEnvironment` node to the scene. In the Inspector, select New Environment in the Environment property. Everything will turn black, but don't worry, you'll fix that soon:

![](img/49ac8ea4-a0d0-4a5c-9aff-4934b98b34f6.png)

Click on New Environment and you'll see a large list of properties. The one you want is Ambient Light. Set Color to white and you'll see your scene become more brightly lit.

Keep in mind that ambient light comes from all directions equally. If your scene needs shadows or other light effects, you'll want to use one of the `Light` nodes. You'll see how light nodes work later in the chapter.

# Finishing the scene

Now that you have the course laid out, two more items remain: the *tee,* or location where the ball will start, and a way to detect when the ball has entered the hole.

Add a `Position3D` node named `Tee`. Just like `Position2D`, this node is used to mark a location in space. Place this node where you want the ball to start. Make sure you put it just above the surface so that the ball doesn't spawn inside the ground.

To detect the ball entering the hole, you can use an `Area` node. This node is directly analogous to the 2D version: it can signal when a body enters its assigned shape. Add an `Area` and give it a `CollisionShape` child.

In the child's Shape property of the `CollisionShape`, add a `SphereShape`:

![](img/08d8a699-79b7-4d9d-98be-8e81cd24e3ed.png)

To size the collision sphere, use the single radius adjustment handle:

![](img/b05ff677-f689-4699-b666-7d06a68195cf.png)

Place the `Area` just below the hole and size the collision shape so that it overlaps the bottom of the hole. Don't let it project above the top of the hole, or the ball will count as *in* when it hasn't dropped in yet.

![](img/16daab45-a042-4120-9a8a-45daa6a8394d.png)

You may find it easier to position the node if you use the Perspective button to view the hole from one direction at at time. When you've finished positioning it, change the name of the `Area` to `Hole`.

# Ball

Now, you're ready to make the ball. Since the ball needs physics—gravity, friction, collision with walls, and other physics properties—`RigidBody` will be the best choice of node. Create a new scene with a `RigidBody` named `Ball`.

RigidBody is the 3D equivalent of the `RigidBody2D` node you used in [Chapter 3](f24a8958-bb32-413a-97ae-12c9e7001c2c.xhtml), *Escape the Maze*. Its behavior and properties are very similar, and you use many of the same methods to interact with it, such as `apply_impulse()` and `_integrate_forces()`.

The shape of the ball needs to be a sphere. The basic 3D shapes such as sphere, cube, cylinder, and so on are called *primitives*. Godot can automatically make primitives using the `MeshInstance` node, so add one as a child of the body. In the Inspector, choose New SphereMesh in the Mesh property:

![](img/b343cce5-673b-4fbd-b471-4bc0d6ef65dd.png)

The default size is much too large, so click on the new sphere mesh and set its size properties, Radius to `0.15` and Height to `0.3`:

![](img/3b43a974-10a4-4565-b5ce-42dd235c396d.png)

Next, add a `CollisionShape` node to the `Ball` and give it a `SphereShape`. Size it to fit the mesh using the size handle (orange dot):

![](img/65232fee-96d9-43af-89b4-695324bd259b.png)

# Testing the ball

To test the ball, add it to the `Main` scene with the instance button. Position it somewhere above the course and hit Play. You should see the ball fall and land on the ground. You may find it helpful to add another `Camera` node positioned on the side of the course for a different view. Set the Current property on whichever camera you want to use.

You can also temporarily give the ball some motion by setting its Linear/Velocity property. Try setting it to different values and playing the scene. Remember that the *y* axis is up and that using too large a value may cause the ball to go right through the wall. Set it back to `(0, 0, 0)` when you're done.

# Improving collisions

You may have noticed when adjusting the velocity that the ball sometimes goes straight through the wall and/or bounces oddly, especially if you choose a high value. There are a few adjustments you can make to the `RigidBody` properties to improve the collision behavior at high speeds.

First, turn on **Continuous Collision Detection** (**CCD**). You'll find it listed as Continuous Cd in the Inspector. Using CCD alters the way the physics engine calculates collisions. Normally, the engine operates by first moving the object and then testing for and resolving collisions. This is fast, and works in most common situations. When using CCD, however, the engine projects the object's movement along its path and attempts to predict where the collision may occur. This is slower than the default behavior, and so not as efficient, especially when simulating many objects, but it is much more accurate. Since you only have one ball in the game, CCD is a good option because it won't introduce any noticeable performance penalty, but will greatly improve collision detection.

The ball also needs a little more action, so set the Bounce to `0.2` and the Gravity Scale to `2`.

Finally, you may also have noticed that the ball takes a long time to come to a stop. Set the Linear/Damp property to `0.5` and Angular/Damp to `0.1` so that you won't have to wait as long for the ball to stop moving.

# UI

Now that the ball is on the course, you need a way to aim and hit the ball. There are a number of possible control schemes for a game of this type. For this project, you'll use a two-step process:

1.  Aim: An arrow will appear swinging back and forth. Clicking the mouse button will set the aim direction to the arrow's.
2.  Shoot: A power bar will move up and down on the screen. Clicking the mouse will set the power and launch the ball.

# Aiming arrow

Drawing an object in 3D is not as easy as it is in 2D. In many cases, you'll have to switch to a 3D modeling program such as Blender to create your game's objects. However, in this case Godot's primitives have you covered; to make an arrow, you just need two meshes: a long, thin rectangle and a triangular prism.

Start a new scene by adding a `Spatial` node with a `MeshInstance` child. Add a new `CubeMesh`. Click on the Mesh property and set the Size property to `(0.5, 0.2, 2)`. This is the body of the arrow, but it still has one problem. If you rotate the parent, the mesh rotates around its center. Instead, you need the arrow to rotate around its end, so change the Transform/Translation of MeshInstance to `(0, 0, -1)`:

![](img/3333f4fc-fafd-4edc-87d9-9b6924fa8e75.png)

Try rotating the `Arrow` (root) node with the gizmo to confirm that the shape is now offset correctly.

To create the point of the arrow, add another `MeshInstance`, and this time choose New PrismMesh. Set its size to `(1.5, 2, 0.5)`. You now have a flat triangle shape. To place it properly at the end of the rectangle, change the mesh's Transform/Translation to `(0, 0, -3)` and its Rotation Degrees to `(-90, 0, 0)`.

Using primitives is a quick way to create placeholder objects directly in Godot without having to open up your 3D modeling software.

Finally, scale the whole arrow down by setting the root node's Transform/Scale to `(0.5, 0.5, 0.5)`:

![](img/90f7f3be-d00b-466b-ac39-4368867eac39.png)

You now have a completed arrow shape. Save it, then instance it in the `Main` scene.

# UI display

Create a new scene with a CanvasLayer called `UI`. In this scene, you'll show the power bar as well as the shot count for the player's score. Add a `MarginContainer`, `VBoxContainer`, two `Label` properties, and a `TextureProgress`. Name them as shown:

![](img/d7849604-8fff-4299-b3da-b8bd8a4db401.png)

Set the Custom Constants of `MarginContainer` all to `20`. Add the `Xolonium-Regular.ttf` font to both of the `Label` nodes and set their font sizes to `30`. Set the `Shots` label's Text to Shots: 0 and the `Label` Text to Power. Drag one of the colored bar textures from `res://assets` into the Texture/Progress of `PowerBar`. By default, `TextureProgress` bars grow from left to right, so for a vertical orientation, change the Fill Mode to Bottom to Top.

The completed UI layout should look like this:

![](img/b862af3c-62fe-4f31-944b-0c2319b10aef.png)

Instance this scene in the `Main` scene. Because it's a CanvasLayer, it will be drawn on top of the 3D camera view.

# Scripts

In this section, you'll create the scripts needed to make everything work together. The flow of the game will be as follows:

1.  Place the ball at the start (`Tee`)
2.  Angle mode: Aim the ball
3.  Power mode: Set the hit power
4.  Launch the ball
5.  Repeat from step 2 until the ball is in the hole

# UI

Add the following script to the `UI` to update the UI elements:

```cpp
extends CanvasLayer

var bar_red = preload("res://assets/bar_red.png")
var bar_green = preload("res://assets/bar_green.png")
var bar_yellow = preload("res://assets/bar_yellow.png")

func update_shots(value):
    $Margin/Container/Shots.text = 'Shots: %s' % value

func update_powerbar(value):
    $Margin/Container/PowerBar.texture_progress = bar_green
    if value > 70:
        $Margin/Container/PowerBar.texture_progress = bar_red
    elif value > 40:
        $Margin/Container/PowerBar.texture_progress = bar_yellow
    $Margin/Container/PowerBar.value = value
```

The two functions provide a way to update the UI elements when they need to display a new value. As you did in the Space Rocks game, changing the progress bar's texture based on its size gives a nice high/medium/low feel to the power level.

# Main

Next, add a script to `Main` and start with these variables:

```cpp
extends Node

var shots = 0
var state
var power = 0
var power_change = 1
var power_speed = 100
var angle_change = 1
var angle_speed = 1.1
enum {SET_ANGLE, SET_POWER, SHOOT, WIN}
```

The `enum` lists the states the game can be in, while the `power*` and `angle*` variables will be used to set their respective values and change them over time. Take a look at the following code snippet:

```cpp
func _ready():
    $Arrow.hide()
    $Ball.transform.origin = $Tee.transform.origin
    change_state(SET_ANGLE)
```

At the beginning, the ball is placed at the location of the `Tee` using both bodies' `transform.origin` properties. Then, the game is put into the `SET_ANGLE` state: 

```cpp
func change_state(new_state):
    state = new_state
    match state:
        SET_ANGLE:
            $Arrow.transform.origin = $Ball.transform.origin
            $Arrow.show()
        SET_POWER:
            pass
        SHOOT:
            $Arrow.hide()
            $Ball.shoot($Arrow.rotation.y, power)
            shots += 1
            $UI.update_shots(shots)
        WIN:
            $Ball.hide()
            $Arrow.hide()
```

The `SET_ANGLE` state places the arrow at the ball's location. Recall that you offset the arrow, so it will appear to be pointing out from the ball. When rotating the arrow, you rotate it around the *y* axis so that it remains flat (the *y* axis points upwards).

Also, note that when entering the `SHOOT` state, you call the `shoot()` function on the `Ball`. You'll add that function in the next section.

The next step is to check for user input:

```cpp
func _input(event):
    if event.is_action_pressed('click'):
        match state:
            SET_ANGLE:
                change_state(SET_POWER)
            SET_POWER:
                change_state(SHOOT)
```

The only input for the game is clicking the left mouse button. Depending on what state you're in, clicking it will transition to the next state:

```cpp
func _process(delta):
    match state:
        SET_ANGLE:
            animate_angle(delta)
        SET_POWER:
            animate_power_bar(delta)
        SHOOT:
            pass
```

In `_process()`, you determine what to animate based on the state. For now, it just calls the function that animates the property that's currently being set:

```cpp
func animate_power_bar(delta):
    power += power_speed * power_change * delta
    if power >= 100:
        power_change = -1
    if power <= 0:
        power_change = 1
    $UI.update_powerbar(power)

func animate_angle(delta):
    $Arrow.rotation.y += angle_speed * angle_change * delta
    if $Arrow.rotation.y > PI/2:
        angle_change = -1
    if $Arrow.rotation.y < -PI/2:
        angle_change = 1
```

Both of these functions are similar. They gradually change a value between two extremes, reversing direction when a limit is hit. Note that the arrow is animating over a +/- 90-degree arc.

# Ball

In the ball script, there are two functions needed. First, an impulse must be applied to the ball to launch it. Second, when the ball stops moving, it needs to notify the `Main` scene so that the player can take another shot:

```cpp
extends RigidBody

signal stopped

func shoot(angle, power):
    var force = Vector3(0, 0, -1).rotated(Vector3(0, 1, 0), angle)
    apply_impulse(Vector3(), force * power / 5)

func _integrate_forces(state):
    if state.linear_velocity.length() < 0.1:
        emit_signal("stopped")
        state.linear_velocity = Vector3()
```

As you saw in the Space Rocks game, you can use the physics state in `_integrate_forces()` to safely stop the ball if the speed has gotten too slow. Remember, due to floating point number precision, the velocity may not actually slow to `0` on its own. The ball may appear to be stopped, but its velocity may actually be something like `0.0000001` instead. Rather than wait for it to reach `0`, you can make the ball stop if its speed drops below `0.1`.

# Hole

To detect when the ball has dropped into the hole, click on the `Area` in `Main` and connect its `body_entered` signal:

```cpp
func _on_Hole_body_entered(body):
    print("Win!")
    change_state(WIN)
```

Changing to the `WIN` state will prevent the ball's `stopped` signal from allowing another shot.

# Testing it out

Try running the game. You may want to make sure you have a very easy course with a straight shot to the hole for this part. You should see the arrow rotating at the ball's position. When you click the mouse button, the arrow stops, and the power bar starts moving up and down. When you click a second time, the ball is launched.

If any of those steps don't work, don't go any further, but stop and go back to try and find what you missed.

Once everything is working, you'll notice some areas that need improvement. First, when the ball stops moving the arrow may not point in the direction you want. The reason for this is that the starting angle is always `0`, which points along the *z* axis, and then the arrow swings +/- 90 degrees from there. In the next sections, you'll have the option of improving the aiming in two ways. 

# Improving aiming – option 1

The aim could be improved by making the 180-degree swing of the arrow always begin by pointing towards the hole.

Add a variable called `hole_dir` to the `Main` script. At the start of aiming, this will be set to the angle pointing towards the hole using the following function:

```cpp
func set_start_angle():
    var hole_pos = Vector2($Hole.transform.origin.z, $Hole.transform.origin.x)
    var ball_pos = Vector2($Ball.transform.origin.z, $Ball.transform.origin.x)
    hole_dir = (ball_pos - hole_pos).angle()
    $Arrow.rotation.y = hole_dir
```

Remember that the ball's position is its center, so it's slightly above the surface, while the hole's center is somewhat below. Because of this, an arrow pointing directly between them would point at a downward angle into the ground. To prevent this and keep the arrow level, you can use only the *x* and *z* values from the `transform.origin` to produce a `Vector2`.

Now the initial arrow direction is towards the hole, so you can alter the animation to add +/-90 degrees to that angle:

```cpp
func animate_angle(delta):
    $Arrow.rotation.y += angle_speed * angle_change * delta
    if $Arrow.rotation.y > hole_dir + PI/2:
        angle_change = -1
    if $Arrow.rotation.y < hole_dir - PI/2:
        angle_change = 1
```

Lastly, change the `SET_ANGLE` state to call the function:

```cpp
SET_ANGLE:
    $Arrow.transform.origin = $Ball.transform.origin
    $Arrow.show()
    set_start_angle()
```

Try the game again. The ball should now always point in the general direction of the hole. This is better, but you still can't point in any direction you like. For that, you can try aiming option 2.

# Improving aiming – option 2

The previous solution is acceptable, but there is another possibility. Instead of the arrow bouncing back and forth, you can aim by moving the mouse side-to-side. The benefit of this option is that you're not limited to 180 degrees of motion.

To accomplish this, you can make use of a particular input event: `InputEventMouseMotion`. This event occurs when the mouse moves, and returns with it a `relative` property representing how far the mouse moved in the previous frame. You can use this value to rotate the arrow by a small amount.

First, disable the arrow animation by removing the `SET_ANGLE` portion from `_process()`. Next, add the following code to `_input()`:

```cpp
func _input(event):
    if event is InputEventMouseMotion:
        if state == SET_ANGLE:
            $Arrow.rotation.y -= event.relative.x / 150
```

This sets the arrow's rotation as you move the mouse left/right on the screen. Dividing by `150` ensures that the movement isn't too fast and that you can move a full 360 degrees if you move the mouse all the way from one side of the screen to the other. Depending on your mouse's sensitivity, you can adjust this to your preference.

# Camera improvements

Another problem, especially if you have a relatively large course, is that if your camera is placed to show the starting area near the tee, it may not show the other parts of the course well, or at all. This can make it challenging to aim when the ball is in certain places.

In this section, you'll learn two different ways to address this problem. One involves creating multiple cameras and activating whichever one is closer to the ball's position. The other solution is to create an *orbiting* camera that follows the ball and that the player can control to view the course from any angle.

# Multiple cameras

Add a second `Camera` node and position it near the hole or at the opposite end of your course, for example:

![](img/c0d98a50-5a24-457c-a9ec-a24693a1a34f.png)

Add an `Area` child to this second camera. Name it `Camera2Area` and then add a `CollisionShape`. You could use a spherical shape just as well, but for this example, choose a `BoxShape`. Note that because you've rotated the camera, the box is rotated as well. You can reverse this by setting the rotation of `CollisionShape` to the opposite value, or you can leave it rotated. Either way, adjust the size and position of the box to cover the portion of the course you want the camera to be responsible for:

![](img/3ff1b125-2957-4d17-ac88-f73febd50b34.png)

Now, connect the area's `body_entered` signal to the main script. When the ball enters the area, the signal will be emitted, and you can change the active camera:

```cpp
func _on_Cam2Area_body_entered(body):
    $Camera2.current = true
```

Play the game again and hit the ball toward the new camera area. Confirm that the camera view changes when the ball enters the area. For a large course, you can add as many cameras as you want/need and set them to activate for different sections of the course.

The drawback of this method is that the cameras are still static. Unless you've very carefully placed them in the right positions, it still may not be comfortable to aim the ball from some locations on the course.

# Orbiting camera

In many 3D games, the player can control a camera that rotates and moves as desired. Typically, the control scheme uses a combination of mouse and keyboard. The first step will be to add some new input actions:

![](img/a34c0387-810d-40db-8426-4dbd57371ac8.png)

The WASD keys will be used to orbit the camera by moving it side to side and up and down. The mouse wheel will control zooming in/out.

# Creating a gimbal

The camera movement needs to have some restrictions. For one, it should always remain level, and not be tilted side to side. Try this: take a camera and rotate it a small amount around x (red ring), then a small amount around *z* (blue ring). Now, reverse the *x* rotation and click the Preview button. Do you see how the camera is now tilted? 

The solution to this problem is to place the camera on a *gimbal—*a device designed to keep an object level during movement. You can create a gimbal using two `Spatial` nodes, which will control the camera's left/right and up/down movement respectively.

First, make sure to remove any other `Camera` nodes in the scene. If you tried the multiple camera setup from the previous section and you'd rather not delete them, you can set their Current values to Off and disconnect any `Area` signals for them.

Add a new `Spatial` node called `GimbalOut` and place it near the center of the course. Make sure not to rotate it. Give it a `Spatial` child called `GimbalIn`, and then add a `Camera` to that node. Set the Transform/Translation of Camera to `(0, 0, 10)`:

![](img/714f8cff-ee71-4622-91de-a29c0eb92640.png)

Here's how the gimbal works: the outer spatial is allowed to rotate *only* in *y*, while the inner one rotates *only* in *x*. You can try it yourself, but make sure you change to Local Space Mode (see the *Introduction to 3D* section). Remember to only move the *green* ring of the outer gimbal node and only the *red* ring of the inner one. Don't change the camera at all. Reset all the rotations to `0` once you've finished experimenting.

To control this motion in the game, attach a script to `GimbalOut` and add the following:

```cpp

extends Spatial

var cam_speed = PI/2
var zoom_speed = 0.1
var zoom = 0.5

func _input(event):
    if event.is_action_pressed('cam_zoom_in'):
        zoom -= zoom_speed
    if event.is_action_pressed('cam_zoom_out'):
        zoom += zoom_speed

func _process(delta):
    zoom = clamp(zoom, 0.1, 2)
    scale = Vector3(1, 1, 1) * zoom
    if Input.is_action_pressed('cam_left'):
        rotate_y(-cam_speed * delta)
    if Input.is_action_pressed('cam_right'):
        rotate_y(cam_speed * delta)
    if Input.is_action_pressed('cam_up'):
        $GimbalIn.rotate_x(-cam_speed * delta)
    if Input.is_action_pressed('cam_down'):
        $GimbalIn.rotate_x(cam_speed * delta)
    $GimbalIn.rotation.x = clamp($GimbalIn.rotation.x, -PI/2, -0.2)
```

As you can see, the left/right actions rotate `GimbalOut` only on the *y* axis, while the up/down actions rotate `GimbalIn` on the *x* axis. The entire gimbal system's `scale` property is used to handle zooming. It is also necessary to set some limits using `clamp()`. The rotation limit holds up/down movement between `-0.2` (almost level with the ground) to `-90` degrees (looking straight down) while the zoom limit keeps you from getting too close or too far away.

Run the game and test the camera controls. You should be able to pan in all four directions and zoom with your mouse wheel. However, the gimbal's position is still static, so you may have trouble seeing the ball properly from certain angles.

# Tracking camera

There is one final improvement to the camera: making it follow the ball. Now that you have a stable, gimbaled camera, it will work great if the gimbal is set to follow the ball's position. Add this line to the `Main` scene's `_process()` function:

```cpp
$GimbalOut.transform.origin = $Ball.transform.origin
```

Note that you shouldn't set the gimbal's transform to the ball's transform, or it will also *rotate* as the ball rolls!

Try the game now and observe how the camera tracks the ball's movement while still being able to rotate and zoom.

# Visual effects

The appearance of the ball and the other meshes in your scene have been intentionally left very plain. You can think of the flat, white ball like a blank canvas, ready to be molded and shaped the way you want it. Applying graphics to 3D models can be a very complex process, especially if you're not familiar with it. First, a bit of vocabulary:

*   **Textures**: Textures are flat, 2D images that are *wrapped* around 3D objects to give them more interesting appearances. Imagine wrapping a present: the flat paper is folded around the package, conforming to its shape. Textures can be very simple or quite complex depending on the shape they are designed to be applied to. An example of a simple one would be a small pattern of bricks that can be repeated on a large wall object.

*   **Shaders**: While textures determine *what* is drawn on an object's surface, shaders determine *how* it is drawn. Imagine that same brick wall. How would it look if it were wet? The mesh and the texture would still be the same, but the way the light reflects from it would be quite different. This is the function of shaders: to alter the appearance of an object without actually changing it. Shaders are typically written in a specialized programming language and can use a great deal of advanced math, the details of which are beyond the scope of this book. For many effects, writing your own shader is unavoidable. However, Godot provides an alternative method of creating a shader for your object that allows for a great deal of customization without diving into shader code: `ShaderMaterial`.

*   **Materials**: Godot uses a computer graphics model called **Physically Based Rendering** (**PBR**). The goal of PBR is to render the surface of objects in a way that more accurately models the way light works in the real world. These affects are applied to meshes using the `Material` property. Materials are essentially containers for textures and shaders. Rather than apply them individually, they are contained in the material, which is then added to the object. The material's properties determine how the textures and shader effects are applied. Using Godot's built-in material properties, you can simulate a wide range of realistic (or stylized) real-world physical materials, such as stone, cloth, or metal. If the built-in properties aren't enough for your purposes, you can write your own shader code to add even more effects.

You can add a PBR material to a mesh using a `SpatialMaterial`.

# SpatialMaterials

Click on the ball's `MeshInstance` and, under Material, select New SpatialMaterial, then click the new material. You will see a great number of parameters, far more than can be covered in this book. This section will focus on some of the most useful ones for making the ball look more appealing. You are encouraged to visit [http://docs.godotengine.org/en/3.0/tutorials/3d/spatial_material.html](http://docs.godotengine.org/en/3.0/tutorials/3d/spatial_material.html) for a full explanation of all the `SpatialMaterial` settings. To improve the look of the ball, try experimenting with these parameters:

*   **Albedo**: This property sets the base color of the material. Change this to make the ball whatever color you like. If you're working with an object that needs a texture to be applied, you can add it here as well. 

*   **Metallic and Roughness**: These parameters control how reflective the surface is. Both can be set to values between `0` and `1`. The Metallic value controls the *shininess*; higher values will reflect more light. The *Roughness* value applies an amount of blur to the reflection. You can simulate a wide variety of materials by adjusting these two properties. The following is a guide to how the *Roughness* and *Metallic* properties affect the appearance of an object. Keep in mind that lighting and other factors will alter the surface appearance as well. Understanding how light and reflections interact with surface properties is a big part of learning to design effective 3D objects:

![](img/6e9de086-672a-47b6-ac47-ffefad22074e.png)

*   **Normal Map**: Normal mapping is a 3D graphics technique for *faking* the appearance of bumps and dents in a surface. Modeling these in the mesh itself would result in a large increase in the number of polygons, or faces, making up the object, leading to reduced performance. Instead, a 2D texture is used that maps the pattern of light and shadow that would result from these small surface features. The lighting engine then uses that information to alter the lighting as if those details were actually there. A properly constructed normal map can add a great amount of detail to an otherwise bland-looking object.

The ball is a perfect example of a good use of normal mapping because a real golf ball has hundreds of dimples on its surface, but the sphere primitive is a smooth surface. Using a regular texture could add spots, but they would look flat and painted on. A normal map that would simulate those dimples looks like this:

![](img/53e4c234-a06d-422b-9bae-c55ed04bce25.png)

It doesn't look like much, but the pattern of red and blue contains information telling the engine which direction it should assume the surface is facing at that point and therefore which direction light should reflect from it there. Note the stretching along the top and the bottom—that's because this image is made to be wrapped around a sphere shape.

Enable the Normal Map property and drag `res://assets/ball_normal_map.png` into the *Texture* field. Try this with the *Albedo* color set to white at first, so you can best see the effect. Adjust the `Depth` parameter to increase or decrease the strength of the effect. A negative value will make the dimples look inset; something between `-1.0` and `-1.5` is a good value:

![](img/3b01f311-304d-4490-ab96-ff72a6518704.png)

Take some time to experiment with these settings and find a combination you like. Don't forget to try it in the game as well, as the ambient lighting of the `WorldEnvironment` will effect the final result.

# Environment options

When you added the WorldEnvironment, the only parameter you changed was the *Ambient Light* color. In this section, you'll learn about some of the other properties you can adjust for improved visual appeal:

*   **Background**: This parameter lets you specify what the background of the world looks like. The default value is Clear Color, which is the plain grey you see currently. Change the Mode to Sky and, in the Sky property, choose New Procedural Sky. Note that the sky is not just for background appearance—objects will reflect and absorb its ambient light. Observe how the ball's appearance changes as you change the `Energy` parameter. This setting can be used to give the impression of a day or night sky, or even that of an alien planet.

*   **Screen Space Ambient Occlusion** (**SSAO**): When enabled, this parameter works together with any ambient light to produce shadows in corners. You have two sources of ambient light now: the *Background* (sky) and the *Ambient Light* settings. Enable SSAO and you'll immediately see an improvement, making the walls of the course look much less fake and plastic. Feel free to try adjusting the various SSAO properties, but remember, a small change can make a big difference. Adjust the properties in small increments, and observe the effects before changing them further.

*   **DOF Far Blur**: *Depth of Field* adds a blur effect to objects that are above a certain distance from the camera. Try adjusting the Distance property to see the effect.

For more information about advanced usage of environmental effects, see [http://docs.godotengine.org/en/3.0/tutorials/3d/environment_and_post_processing.html](http://docs.godotengine.org/en/3.0/tutorials/3d/environment_and_post_processing.html).

# Lighting

Add a `DirectionalLight` to the scene. This type of light simulates an infinite number of parallel rays of light, so it's often used to represent sunlight or another very distant source of light that illuminates an entire area equally. The location of the node in the scene doesn't matter, only its direction, so you can position it anywhere you like. Aim it using the gizmo so that it strikes the course at an angle, then turn Shadow/Enabled to On so that you'll see shadows being cast from the walls and other objects:

![](img/dcd72a60-9d4e-4d24-a054-8ea693765621.png)

There are a number of properties available to adjust and alter the appearance of the shadows, both in the *Shadow* section, which is present for all `Light` nodes, and in the *Directional Shadow* section, which is specific to `DirectionalLight`. The default values will work for most general cases, but the one property that you should probably adjust to improve shadow appearance is *Max Distance*. Lowering this value will improve shadow appearance, but only when the camera is closer than the given distance. If your camera will mostly be close to objects, you can reduce this value. To see the effect, try setting it to just `10` and zooming in/out, then do the same with it set to `1000`.

Directional light can even be used to simulate the day/night cycle. If you attach a script to the light and slowly rotate it around one axis, you'll see the shadows change as if the sun is rising and setting.

# Summary

This chapter introduced you to the world of 3D graphics. One of Godot's great strengths is that the same tools and workflow are used in both 2D and 3D. Everything you learned about the process of creating scenes, instancing, and using signals works in the same way. For example, an interface you build with control nodes for a 2D game can be dropped into a 3D game and will work just the same.

In this chapter, you learned how to navigate in the the 3D editor to view and place nodes using gizmos.You learned about meshes and how to quickly make your own objects using Godot's primitives. You used GridMap to lay out your minigolf course. You learned about using cameras, lighting, and the world environment to design how your game will appear on screen. Finally, you got a taste of using PBR rendering via Godot's SpatialMaterial resource.

Congratulations, you've made it to the end! But with these five projects, your journey to becoming a game developer has just begun. As you become more proficient with Godot's features, you'll be able to make any game you can imagine.