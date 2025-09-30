# Chapter 3. Game Objects – More and Move

We created our first room in the Unreal Editor in [Chapter 2](ch02.html "Chapter 2. Creating Your First Level"), *Creating Your First Level*. In this chapter, we will cover some information about the structure of objects we have used to prototype the level in [Chapter 2](ch02.html "Chapter 2. Creating Your First Level"), *Creating Your First Level*. This is to ensure that you have a solid foundation in some important core concepts before moving forward. Then, we will progressively introduce various concepts to make the objects move upon a player's interaction.

In this chapter, we will cover the following topics:

*   BSP Brush
*   Static Mesh
*   Texture and Materials
*   Collision
*   Volumes
*   Blueprint

# BSP Brush

We used the BSP Box Brush in [Chapter 2](ch02.html "Chapter 2. Creating Your First Level"), *Creating Your First Level*, extensively to create the ground and the walls.

BSP Brushes are the primary building blocks for level creation in the game development. They are used for quick prototyping levels like how we have used them in [Chapter 2](ch02.html "Chapter 2. Creating Your First Level"), *Creating Your First Level*.

In Unreal, BSP Brushes come in the form of primitives (box, sphere, and so on) and also predefined/custom shapes.

## Background

BSP stands for **binary space partitioning**. The structure of a BSP tree allows spatial information to be accessed quickly for rendering, especially in 3D scenes made up of polygons. A scene is recursively divided into two, until each node of the BSP tree contains only polygons that can render in arbitrary order. A scene is rendered by traversing down the BSP tree from a given node (viewpoint).

Since a scene is divided using the BSP principle, placing objects in the level could be viewed as cutting into the BSP partitions in the scene. Geometry Brushes use **Constructive Solid Geometry** (**CSG**) technique to create polygon surfaces. CSG combines simple primitives/custom shapes using Boolean operators such as union, subtraction, and intersection to create complex shapes in the level.

So, the CSG technique is used to create surfaces of the object in the level, and rendering the level is based on processing these surfaces using the BSP tree. This relationship has resulted in Geometry Brushes being known also as BSP Brushes, but more accurately, CSG surfaces.

## Brush type

BSP Brushes can either be additive or subtractive in nature. Additive brushes are like volumes that fill up the space. Additive brushes were used for the ground and the walls in our map in [Chapter 2](ch02.html "Chapter 2. Creating Your First Level"), *Creating Your First Level*.

Subtractive brushes can be used to form hollow spaces. These were used to create a hole in the wall in which to place a door and its frame in [Chapter 2](ch02.html "Chapter 2. Creating Your First Level"), *Creating Your First Level*.

## Brush solidity

For additive brushes, there are various states it can be in: solid, semi-solid, or non-solid.

Since subtractive brushes create empty spaces, players are allowed to move freely within them. Subtractive brushes can only be solid brushes.

Refer to the following table for comparison of their properties:

| Brush solidity | Brush type | Degree of blocking | BSP cutting |
| --- | --- | --- | --- |
| Solid | Additive and subtractive | Blocks both players and projectiles | Creates BSP cuts to the surrounding world geometry |
| Semi-solid | Additive only | Blocks both players and projectiles | Does not cause BSP cuts to the surrounding world geometry |
| Non-solid | Additive only | Does not block players or projectiles | Does not cause BSP cuts to the surrounding world geometry |

# Static Mesh

Static Mesh is a geometry made up of polygons. Looking more microscopically at what a mesh is made of, it is made up of lines connecting vertices.

Static Mesh has vertices that cannot be animated. This means is that you cannot animate a part of the mesh and make that part move relative to itself. But the entire mesh can be translated, rotated, and scaled. The lamp and the door that we have added in [Chapter 2](ch02.html "Chapter 2. Creating Your First Level"), *Creating Your First Level*, are examples of Static Meshes.

A higher-resolution mesh has more polygons as compared to a lower-resolution mesh. This also implies that a higher resolution mesh has a larger number of vertices. A higher resolution mesh takes more time to render but is able to provide more details in the object.

Static Meshes are usually first created in external software programs, such as Maya or 3ds Max, and then imported into Unreal for placement in game maps.

The door, its frame, and the lamp that we added in [Chapter 2](ch02.html "Chapter 2. Creating Your First Level"), *Creating Your First Level*, are Static Meshes. Notice that these objects are not simple geometry looking objects.

# BSP Brush versus Static Mesh

In game development, many objects in the game are Static Meshes. Why is that so? Static Mesh is considered more efficient, especially for a complex object with many vertices, as they can be cached to a video memory and are drawn by the computer's graphics card. So, Static Meshes are preferred when creating objects as they have better render performance, even for complex objects. However, this does not mean that BSP Brushes do not have a role in creating games.

When BSP Brush is simple, it can still be used without causing too much serious impact to the performance. BSP Brush can be easily created in the Unreal Editor, hence it is very useful for quick prototyping by the game/level designers. Simple BSP Brushes can be created and used as temporary placeholder objects while the actual Static Mesh is being modeled by the artists. The creation of a Static Mesh takes time, even more so for a highly detailed Static Mesh. We will cover a little information about the Static Mesh creation pipeline later in this chapter, so we have an idea of the amount of work that needs to be done to get a Static Mesh into the game. So, BSP Brush is great for an early game play testing without having to wait for all Static Meshes to be created.

# Making Static Mesh movable

Let us open our saved map that we have created in [Chapter 2](ch02.html "Chapter 2. Creating Your First Level"), *Creating Your First Level*, and let us first save the level as a new `Chapter3Level`.

1.  Go to **Content Browser** | **Content** | **StarterContent** | **Props**, and search for **SM_Chair**, which is a standard Static Mesh prop. Click and drag it into our map.
2.  The chair we have in the level now is unmovable. You can quickly build and run the level to check it out. To make it movable, we need to change a couple of settings under the chair's details.
3.  First, ensure **SM_Chair** is selected, go to the **Details** tab. Go to **Transform** | **Mobility**, change it from **Static** to **Movable**. Take a look at the following screenshot, which describes how to make the chair movable:![Making Static Mesh movable](img/B03679_03_01.jpg)
4.  Next, we want the chair to be able to respond to us. Scroll a little down the **Details** tab to change the **Physics** setting for the chair. Go to **Details** | **Physics**. Make sure the checkbox for **Simulate Physics** is checked. When this checkbox is checked, the auto-link setting sets the **Collision** to be a **PhysicsActor**. The following screenshot shows the **Physics** settings of the chair:![Making Static Mesh movable](img/B03679_03_02.jpg)

Let us now build and play the level. When you walk into the chair, you will be able to push it around. Just to note, the chair is still known as Static Mesh, but it is now movable.

# Materials

In [Chapter 2](ch02.html "Chapter 2. Creating Your First Level"), *Creating Your First Level*, we selected a walnut polished material and applied it to the ground. This changed the simple dull ground into a brown polished wood floor. Using materials, we are able to change the look and feel of the objects.

The reason for a short introduction of materials here is because it is a concept that we need to have learned about before we can construct a Static Mesh. We already know that we need Static Meshes in the game and we cannot only rely on the limited selection that we have in the default map package. We will need to know how to create our own Static Meshes, and we rely heavily on Materials to give the Static Meshes their look and feel.

So, when do we apply Materials while creating our custom Static Mesh? Materials are applied to the Static Mesh during its creation process outside the editor, which we will cover in a later section of this chapter. For now, let us first learn how Materials are constructed in the editor.

## Creating a Material in Unreal

To fully understand the concept of a Material, we need to break it down into its fundamental components. How a surface looks is determined by many factors, including color, presence of print/pattern/designs, reflectivity, transparency, and many more. These factors combine together to give the surface its unique look.

In Unreal Engine, we are able to create our very own material by using the Material Editor. Based on the explanation given earlier, a Material is determined by many factors and all these factors combine together to give the Material its own look and feel.

Unreal Engine offers a base Material node that has a list of customizable factors, which we can use to design our Material. By using different values to different factors, we can come up with our very own Material. Let us take a look at what is behind the scene in a material that we have used in [Chapter 2](ch02.html "Chapter 2. Creating Your First Level"), *Creating Your First Level*.

Go to **Content Browser** | **Content** | **Starter Content** | **Materials** and double-click on **M_Brick_Clay_New**. This opens up the Material Editor. The following screenshot shows the zoomed-in version of the base Material node for the brick clay material. You might notice that **Base Color**, **Roughness**, **Normal**, and **Ambient Occlusion** have inputs to the base **M_Brick_Clay_New** material node. These inputs make the brick wall look like a brick wall.

![Creating a Material in Unreal](img/B03679_03_03.jpg)

The inputs to these nodes can take on values from various sources. Take **Base Color** for example, we can define the color using RGB values or we can take the color from the texture input. Textures are images in formats, such as `.bmp`, `.jpg`, `.png`, and so on, which we can create using tools, such as Photoshop or ZBrush.

We will talk more about the construction of the materials a little later in this book. For now, let us just keep in mind that materials are applied to the surfaces and textures are what we can use in combination, to give the materials its overall visual look.

## Materials versus Textures

Notice that I have used both Materials and Textures in the previous section. It has often caused quite a bit of confusion for a newbie in the game development. Material is what we apply to surfaces and they are made up of a combination of different textures. Materials take on the properties from the textures depending on what was specified, including color, transparency, and so on.

As explained earlier, Textures are simple images in formats such as `.tga`, `.bmp`, `.jpg`, `.png`, and so on.

## Texture/UV mapping

Now, we understand that a custom material is made up of a combination of textures and material is applied onto surfaces to give the polygon meshes its identity and realism. The next question is how do we apply these numerous textures that come with the material onto the surfaces? Do we simply slap them onto the 3D object? There must be a predictable manner in which we paint these textures onto the surfaces. The method used is called **Texture Mapping** , which was pioneered by Edwin Catmull in 1974.

Texture mapping assigns pixels from a texture image to a point on the surface of the polygon. The texture image is called a **UV texture map**. The reason we are using UV as an alternative to the XY coordinates is because we are already using XY to describe the geometric space of the object. So the UV coordinates are the texture's XY coordinates, and it is solely used to determine how to paint a 3D surface.

### How to create and use a Texture Map

We will first need to unwrap a mesh at its seams and lay it out flat in 2D. This 2D surface is then painted upon to create the texture. This painted texture (also known as **Texture Map**) will then be wrapped back around the mesh by assigning the UV coordinates of the texture on each face of the mesh. To help you better visualize, take a look at the following illustration:

![How to create and use a Texture Map](img/B03679_03_04.jpg)

Source: Wikipedia ([https://en.wikipedia.org/wiki/UV_mapping](https://en.wikipedia.org/wiki/UV_mapping))

As a result of this, shared vertices can have more than one set of UV coordinates assigned.

### Multitexturing

To create a better appearance in surfaces, we can use multiple textures to create the eventual end result desired. This layering technique allows for many different textures to be created using different combinations of textures. More importantly, it gives the artists better control of details and/or lighting on a surface.

### A special form of texture maps – Normal Maps

Normal Maps are a type of texture maps. They give the surfaces little bumps and dents. Normal Maps add the details to the surfaces without increasing the number of polygons. One very effective use of Normal Mapping is to generate Normal Maps from a high polygon 3D model and use it to texture the lower polygon model, which is also known as **baking**. We will discuss why we need the same 3D model with different number of polygons in the next section.

Normal maps are commonly stored as regular RGB images where the RGB components correspond to the X, Y, and Z coordinates, respectively, of the surface normal. The following image shows an example of a normal map taken from [http://www.bricksntiles.com/textures/](http://www.bricksntiles.com/textures/):

![A special form of texture maps – Normal Maps](img/B03679_03_05.jpg)

# Level of detail

We create objects with varying **level of details** (**LODs**) to increase the efficiency of rendering. For objects that are closer to the player, high LODs objects are rendered. Objects with higher LODs have a higher number of polygons. For objects that are far away from the player, a simpler version of the object is rendered instead.

Artists can create different LOD versions of the 3D object using automated LOD algorithms, deployed through software or manually reducing the number of vertices, normals, edges in the 3D Models, to create a lower polygon count model. When creating models of different LODs, note that we always start by creating the most detailed model with the most number of polygons first and then reduce the number accordingly to create the other LOD versions. It is much harder to work the models the other way around. Do remember to keep the UV coherent when working with objects with different LODs. Currently, different LODs need to be light mapped separately.

The following image is taken from [http://renderman.pixar.com/view/level-of-detail](http://renderman.pixar.com/view/level-of-detail) and very clearly shows the polygon count based on the distance away from the camera:

![Level of detail](img/B03679_03_06.jpg)

# Collisions

Objects in Unreal Engine have collision properties that can be modified to design the behavior of the object when it collides with another object.

In real life, collisions occur when two objects move and meet each other at a point of contact. Their individual object properties will determine what kind of collision we get, how they respond to the collision, and their path after the collision. This is what we try to achieve in the game world as well.

The following screenshot shows the collision properties available to an object in Unreal Engine 4:

![Collisions](img/B03679_03_07.jpg)

If you are still confused about the concept of collision, imagine Static Mesh to give an object its shape (how large it is, how wide it is, and so on), while the collision of the object is able to determine the behavior of this object when placed on the table—whether the object is able to fall through the table in the level or lay stationery on the table.

## Collision configuration properties

Let us go through some of the possible configurations in Unreal's **Collision** properties that we should get acquainted with.

### Simulation Generates Hit Events

When an object has the **Simulation Generates Hit Events** flag checked, an alert is raised when the object has a collision. This alert notification can be used to trigger the onset of other game actions based on this collision.

### Generate Overlap Events

The **Generate Overlap Events** flag is similar to the **Simulation Generates Hit Events** flag, but when this flag is checked, in order to generate an event, all the object needs is to have another object to overlap with it.

### Collision Presets

The **Collision Presets** property contains a few frequently used settings that have been preconfigured for you. If you wish to create your own custom collision properties, set this to **Custom**.

### Collision Enabled

The **Collision Enabled** property allows three different settings: **No Collision**, **No Physics Collision**, and **Collision Enabled**. **No Physics Collision** is selected when this object is used only for non-physical types of collision such as raycasts, sweeps, and overlaps. **Collision Enabled** is selected when physics collision is needed. No Collision is selected when absolutely no collision is wanted.

### Object Type

Objects can be categorized into several groups: **WorldStatic**, **WorldDynamic**, **Pawn**, **PhysicsBody**, **Vehicle**, **Destructible**, and **Projectile**. The type selected determines the interactions it takes on as it moves.

### Collision Responses

The **Collision Responses** option sets the property values for all **Trace** and **Object Responses** that come with it. When **Block** is selected for **Collision Responses**, all the properties under **Trace** and **Object Responses** are also set to **Block**.

#### Trace Responses

The **Trace Responses** option affects how the object interacts with traces. **Visibility** and **Camera** are the two types of traces that you can choose to block, overlap, or ignore.

#### Object Responses

The **Object Responses** option affects how this object interacts with other object types. Remember the **Object Type** selection earlier? The **Object Type** property determines the type of object, and under this category, you can configure the collision response this object has with the different types of objects.

### Collision hulls

For a collision to occur in Unreal Engine, hulls are used. To view an example of the collision hull for a Static Mesh, take a look at the light blue lines surrounding the cube in the following screenshot; it's a box collision hull:

![Collision hulls](img/B03679_03_08.jpg)

Hulls can be generated in Static Mesh Editor for static meshes. The following screenshot shows the menu options available for creating an auto-generated collision hull in Static Mesh Editor:

![Collision hulls](img/B03679_03_09.jpg)

Simple geometry objects can be combined and overlapped to form a simple hull. A simple hull/bounding box reduces the amount of calculation it needs during a collision. So for complex objects, a generalized bounding box can be used to encompass the object. When creating static mesh that has a complex shape, not a simple geometry type of object, you will need to refer to the *Static Mesh creation pipeline* section later on in the chapter to learn how to create a suitable collision bounding box for it.

## Interactions

When designing collisions, you will also need to decide what kind of interactions the object has and what it will interact with.

To block means they will collide, and to overlap can mean that no collision will occur. When a block or an overlap happens, it is possible to flag the event so that other actions resulting from this interaction can be taken. This is to allow customized events, which you can have in game.

Note that for a block to actually occur, both objects must be set to **Block** and they must be set so that they block the right type of objects too. If one is set to block and the other to overlap, the overlap will occur but not the block. Block and overlap can happen when objects are moving at a high speed, but events can only be triggered on either overlap or block, not both. You can also set the blocking to ignore a particular type of object, for example, **Pawn**, which is the player.

# Static Mesh creation pipeline

Static Mesh creation pipeline is done outside of the editor using 3D modeling tools such as Autodesk's Maya and 3D's Max. Unreal Engine 4 is compatible to import the FBX 2013 version of the files.

This creation pipeline is used mainly by the artists to create game objects for the project.

The actual steps and naming convention when importing Static Mesh into the editor are well documented on the Unreal 4 documentation website. You may refer to [https://docs.unrealengine.com/latest/INT/Engine/Content/FBX/StaticMeshes/index.html](https://docs.unrealengine.com/latest/INT/Engine/Content/FBX/StaticMeshes/index.html) for more details.

# Introducing volumes

Volumes are invisible areas that are created to help the game developers perform a certain function. They are used in conjunction with the objects in the level to perform a specific purpose. Volumes are commonly used to set boundaries that are intended to prevent players from gaining access to trigger events in the game, or use the Lightmass Importance Volume to change how light is calculated within an area in the map as in [Chapter 2](ch02.html "Chapter 2. Creating Your First Level"), *Creating Your First Level*.

Here's a list of the different types of volumes that can be customized and used in Unreal Engine 4\. But feel free to quickly browse through each of the volumes here for now, and revisit them later when we start learning how to use them later in the book. For this chapter, you may focus your attention first on the Trigger Volume, as we will be using that in the later examples of this chapter.

## Blocking Volume

The Blocking Volume can be used to prevent players/characters/game objects from entering a certain area of the map. It is quite similar to collision hull which we have described earlier and can be used in place of Static Mesh collision hull, as they are simpler in shapes (block shapes), hence easier to calculate the response of the collision. These volumes also have the ability to detect which objects overlap with themselves quickly.

An example of the usage of the Blocking Volume is to prevent the player from walking across a row of low bushes. In this case, since the bushes are rather irregularly shaped but are roughly forming a straight line, like a hedge, an invisible Blocking Volume would be a very good way of preventing the player from crossing the bushes.

The following screenshot shows the properties for the Blocking Volume. We can change the shape and size of the volume under **Brush Settings**. Collision events and triggers other events using Blueprint. This is pretty much the basic configuration we will get for all other volumes too.

![Blocking Volume](img/B03679_03_10.jpg)

## Camera Blocking Volume

The Camera Blocking Volume works in the same way as the Blocking Volume but it is used specifically to block cameras. It is useful when you want to limit the player from exploring with the camera beyond a certain range.

## Trigger Volume

The Trigger Volume is probably one of the most used volumes. This is also the volume which we would be using to create events for the game level that we have been working on. As the name implies, upon entering this volume, we can trigger events, and via Blueprint, we can create a variety of events for our game, such as moving an elevator or spawning NPCs.

## Nav Mesh Bounds Volume

The Nav Mesh Bounds Volume is used to indicate the space in which NPCs are able to freely navigate around. NPCs could be enemies in the game who need some sort of path finding method to get around the level on their own. This Nav Mesh Bounds Volume will set up the area in the game that they are able to walk through. This is important as there could be obstacles such as bridges that they will need to use to in order get across to the other side (instead of walking straight into the river and possibly drowning).

## Physics Volume

The Physics Volume is used to create areas in which the physics properties of the player/objects in the level change. An example of this would be altering the gravity within a space ship only when it reaches the orbit. When the gravity is changed in these areas, the player starts to move slower and float in the space ship. We can then turn this volume off when the ship comes back to earth. The following screenshot shows the additional settings we get from the Physics Volume:

![Physics Volume](img/B03679_03_11.jpg)

### Pain Causing Volume

The Pain Causing Volume is a very specialized volume used to create damage to the players upon entry. It is a "milder" version of the Kill Z Volume. Reduction of health and the amount of damage per second are customizable, according to your game needs. The following screenshot shows the properties you can adjust to control how much pain to inflict on the player:

![Pain Causing Volume](img/B03679_03_12.jpg)

### Kill Z Volume

We kill the player when it enters the Kill Z Volume. This is a very drastic volume that kills the player immediately. An example of its usage is to kill the player immediately when the player falls off a high building. The following screenshot shows the properties of Kill Z Volume to determine the point at which the player is killed:

![Kill Z Volume](img/B03679_03_13.jpg)

### Level Streaming Volume

The Level Streaming Volume is used to display the levels when you are within the volume. It generally fills the entire space where you want the level to be loaded. The reason we need to stream levels is to give players an illusion that we have a large open game level, when in fact the level is broken up into chunks for more efficient rendering. The following screenshot shows the properties that can be configured for the Level Streaming Volume:

![Level Streaming Volume](img/B03679_03_14.jpg)

### Cull Distance Volume

The Cull Distance Volume allows objects to be culled in the volume. The definition of cull is to select from a group. The Cull Distance Volume is used to select objects in the volume that need to disappear (or not rendered) based on the distance away from the camera. Tiny objects that are far away from the camera cannot be seen visibly. These objects can be culled if the camera is too far away from those objects. Using the Cull Distance Volume, you would be able to decide upon the distance and size of objects, which you want to cull within a fixed space. This can greatly improve performance of your game when used effectively.

This might seem very similar to the idea of occlusion. Occlusion is implemented by selecting object by object, when it is not rendered on screen. These are normally used for larger objects in the scene. Cull Distance Volume can be used over a large area of space and using conditions to specify whether or not the objects are rendered.

The following screenshot shows the configuration settings that are available to the Cull Distance Volume:

![Cull Distance Volume](img/B03679_03_15.jpg)

### Audio Volume

The Audio Volume is used to mimic real ambient sound changes when one transits from one place to another, especially when transiting to and from very different environments, such as walking into a clock shop from a busy street, or walking in and out of a restaurant with a live band playing in the background.

The volume is placed surrounding the boundaries of one of the areas creating an artificial border dividing the spaces into interior and exterior. With this artificially created boundary and settings that come with this Audio Volume, sound artists are able to configure how sounds are played during this transition.

### PostProcess Volume

The PostProcess Volume affects the overall scene using post-processing techniques. Post-processing effects include Bloom effects, Anti-Aliasing, and Depth of Field.

### Lightmass Importance Volume

We have used Lightmass Importance Volume in [Chapter 2](ch02.html "Chapter 2. Creating Your First Level"), *Creating Your First Level*, to focus the light on the section of the map that has the objects in. The size of the volume should encompass your entire level.

# Introducing Blueprint

The Unreal Editor offers the ability to create custom events for game levels through a visual scripting system. Before Unreal Engine 4, it was known as the **Kismet system**. In Unreal Engine 4, this system was revamped with more features and capabilities. The improved system was launched with the new name of Blueprint.

There are several types of Blueprint: Class Blueprint, Data-Only Blueprint, and Level Blueprint. These are more or less equivalent to what we used to know as Kismet, which is now known as Level Blueprint.

Why do I need Blueprint? The simple answer is that through Blueprint, we are able to control gameplay without having to dive into the actual coding. This makes it convenient for non-programmers to design and modify the gameplay. So, it mainly benefits the game designers/artists who can configure the game through the Blueprint editor.

So, how can we use Blueprint and what can I use Blueprint for? Blueprint is just like coding with an interface. You can select, drag, and drop function nodes into the editor, and link them up logically to evoke the desired response to specified scenarios in your game. For programmers, they will be able to pick it up pretty quickly, since Blueprint is in fact coding but through a visual interface.

For the benefit of everyone who is new to Unreal Engine 4 and maybe programming as well, we will go through a basic example of how Level Blueprint works here and use that as an example to go through some basic programming concepts at the same time.

What will we be using Blueprint for? Blueprint has the capabilities to prototype, implement, or modify virtually any gameplay element. The gameplay elements affect how game objects are spawned, what gets spawned, where they are spawned, and under what conditions they are spawned. The game objects can include lights, camera, player's input, triggers, meshes, and character models. Blueprint can control properties of these game objects dynamically to create countless gameplay scenarios. The examples of usage include altering the color of the lights when you enter a room in the game, triggering the door to shut behind you after entering the room and playing the sound effect of the door closing shut, spawning weapons randomly among three possible locations in the map, and so on.

In this chapter, we will focus on Level Blueprint first, since it is the most commonly used form of Blueprint.

## Level Blueprint

Level Blueprint is a type of Blueprint that has influence over what happens in the level. Events that are created in this Blueprint affect what happens in the level, and are made specific to the situation by specifying the particular object it targets.

Feel free to jump to the next section first where we will go through a Blueprint example, so that we are able to understand Level Blueprint a little better.

The following screenshot shows a blank Level Blueprint. The most used window is **Event Graph**, which is in the center. Using different node types in **Event Graph** and linking it up appropriately creates a responsive interaction within the game. The nodes come with variables, values, and other similar properties used in programming to control the game events graphically (without writing a single line of script or code).

![Level Blueprint](img/B03679_03_16.jpg)

# Using the Trigger Volume to turn on/off light

We are now ready to use what we have learned to construct the next room for our game. We will duplicate the first room we have created in order to create our second room.

1.  Open the level that we created in [Chapter 2](ch02.html "Chapter 2. Creating Your First Level"), *Creating Your First Level*, (`Chapter2_Level`) and save it as a new level called `Chapter3_Level`.
2.  Select all the walls, the floor, the door, and the door frame.
3.  Hold down *Alt* + *Shift* and drag to duplicate the room.
4.  Place the duplicated room with the duplicated door aligned to the wall of the first room. Refer to the following screenshot to see how the walls are aligned from a **Top** view perspective:![Using the Trigger Volume to turn on/off light](img/B03679_03_17.jpg)
5.  Delete the back wall of the first room to link both the rooms.
6.  Delete all the doors to allow easy access to the second room.
7.  Move the standing lamp and chair to the side. Take a look the following screenshot to understand how the rooms look at this point:![Using the Trigger Volume to turn on/off light](img/B03679_03_18.jpg)
8.  Rebuild the lights. The following screenshot shows the room correctly illuminated after building the lights:![Using the Trigger Volume to turn on/off light](img/B03679_03_19.jpg)
9.  Now, let us focus on working on the second room. We will create a narrower walkway using the second room that we have just created.
10.  Move the sidewalls closer to each other—about 30 cm from the previous sidewall towards the center. Refer to the next two screenshots for the **Top** and **Perspective** views after moving the sidewalls:![Using the Trigger Volume to turn on/off light](img/B03679_03_20.jpg)![Using the Trigger Volume to turn on/off light](img/B03679_03_21.jpg)
11.  Note that LightMass Importance Volume is not encompassing the entire level now. Increase the size of the volume to cover the whole level. Take a look at the following screenshot to see how to extend the size of the volume correctly:![Using the Trigger Volume to turn on/off light](img/B03679_03_22.jpg)
12.  Go to **Content Browser** | **Props**. Click and drop **SM_Lamp_Wall** into the level. Rotate the lamp if necessary so that it lies nicely on the side wall.
13.  Go to **Modes** | **Lights**. Click and drop a Point Light into the second room. Place it just above the light source on the wall light, which we added in the previous step. Take a look at the following screenshot to see the placement of the lamp and Point Light that we have just added:![Using the Trigger Volume to turn on/off light](img/B03679_03_23.jpg)
14.  Adjust the Point Light settings: Intensity = 1700.0\. This is approximately the light intensity coming off a light bulb. The following screenshot shows the settings for the Point Light:![Using the Trigger Volume to turn on/off light](img/B03679_03_24.jpg)
15.  Next, go to **Light Color** and adjust the color of the light to **#FF9084FF**, to adjust the mood of the level.![Using the Trigger Volume to turn on/off light](img/B03679_03_25.jpg)
16.  Now, let us rename the Point Light to `WalkwayLight` and the **Wall Lamp prop** to `WallLamp`.
17.  Select the Point Light and right-click to display the contextual menu. Go to **Attach To** and select **WallLamp**. This attaches the light to the prop so that when we move the prop, the light moves together. The following screenshot shows that **WalkwayLight** is linked to **WallLamp**:![Using the Trigger Volume to turn on/off light](img/B03679_03_26.jpg)
18.  Now, let us create a Trigger Volume. Go to **Modes** | **Volumes**. Click and drag the Trigger Volume into the level.
19.  Resize the volume to cover the entrance of the door dividing the two rooms. Refer to the next two screenshots on how to position the volume (**Perspective** view and **Top** view). Make sure that the volume covers the entire space of the door.![Using the Trigger Volume to turn on/off light](img/B03679_03_27.jpg)![Using the Trigger Volume to turn on/off light](img/B03679_03_28.jpg)
20.  Rename **Trigger Volume** to `WalkwayLightTrigger`.
21.  In order to use the Trigger Volume to turn the light on and off, we need to figure out which property from the Point Light controls this feature. Click on the Point Light (**WalkwayLight**) to display the properties of the light. Scroll down to **Rendering** and uncheck the property box for **Visible**. Notice that the light is now turned off. We want to keep the light turned off until we trigger it.
22.  So, the next step is to link the sequence of events up. This is done via **Level Blueprint**. We will need to trigger this change in property using the Trigger Volume, which we have created and turn the light back on.
23.  With the Point Light still selected, go to the top ribbon and select **Blueprints** | **Open Level Blueprint**. This opens up the **Level Blueprint** window. Make sure that the Point Light (**WalkwayLight**) is still selected as shown in the following screenshot:![Using the Trigger Volume to turn on/off light](img/B03679_03_29.jpg)
24.  Right-click in the **Event Graph** of the **Level Blueprint** window to display what actions can be added to the **Level Blueprint**.
25.  Due to Level Blueprint's ability to guide what actions are possible, we can simply select **Add Reference to WalkwayLight**. This creates the **WalkwayLight** actor in **Level Blueprint**. The following screenshot shows the **WalkwayLight** actor correctly added in **Blueprint**:![Using the Trigger Volume to turn on/off light](img/B03679_03_30.jpg)
26.  You can keep the **Level Blueprint** window open, and go to the Trigger Volume we have created the in the level.
27.  Select the Trigger Volume (**WalkwayLightTrigger**), right-click and select **Add Event** and then **OnActorBeginOverlap**. The following screenshot shows how to add **OnActorBeginOverlap** in **Level Blueprint**:![Using the Trigger Volume to turn on/off light](img/B03679_03_31.jpg)
28.  To control a variable in the Point Light, we will click and drag on the tiny blue circle on the **WalkwayLight** node added. This creates a blue line originating from the tiny blue circle. This also opens up a menu, where we can see what action can be done to the Point Light. Enter `visi` into the search bar to display the options. Click on **Set Visibility**. The following screenshot shows how to add the **Set Visibility** function to the Point Light (**WalkwayLight**):![Using the Trigger Volume to turn on/off light](img/B03679_03_32.jpg)
29.  Check the **New Visiblity** checkbox in the **Set Visiblity** function. The following screenshot shows the configuration we want:![Using the Trigger Volume to turn on/off light](img/B03679_03_33.jpg)
30.  Now, we are ready to link the **OnActorBeginOverlap** event to the **Set Visibility** function. Click and drag the white triangular box from **OnActorBeginOverlap** and drop it on the white triangular box at the **Set Visibility** function. The following screenshot shows the event correctly linked up:![Using the Trigger Volume to turn on/off light](img/B03679_03_34.jpg)
31.  Now, build the level and play. Walk through the door from the first room to the second room. The light should be triggered on.

But what happens when you walk back into the first room? The light remained turned on and nothing happens when you walk back into the second room. In the next example, we will go through how you can toggle the light on and off as you walk in and out the room. It is an alternative way to implement the control of the light and I shall leave it as optional for you to try it out.

# Using Trigger Volume to toggle light on/off (optional)

The following steps can be used to trigger volume to toggle lights on or off:

1.  We need to replace the **Set Visibility** node in **Event Graph**. Click and drag the blue dot from Point Light (**WalkwayLight**) and drop it onto any blank space. This opens up the contextual menu. The following screenshot shows the contextual menu to place a new node from **WalkwayLight**:![Using Trigger Volume to toggle light on/off (optional)](img/B03679_03_35.jpg)
2.  Select **Toggle Visibility**. This creates an additional new node in **Event Graph**; we will need to rewire the links as per the following screenshot in order to link **OnActorBeginOverlap** to **Toggle Visibility**:![Using Trigger Volume to toggle light on/off (optional)](img/B03679_03_36.jpg)
3.  The last step is to delete the **Set Visiblity** node and we are ready to toggle the light on and off as we move in and out of the room. The following screenshot shows the final **Event Graph** we want. Compile and play the level to see how you can toggle the light on and off.![Using Trigger Volume to toggle light on/off (optional)](img/B03679_03_37.jpg)

# Summary

We have covered a number of very important concepts about the objects that we use to populate our game world in Unreal Engine 4\. We have broken one of the most common types of game object, Static Mesh, into its most fundamental components in order to understand its construction. We have also compared two types of game objects (Static Meshes and BSP), how they are different, and why they have their spot in the game. This will help you decide what kind of objects need to be created and how they will be created for your game level.

The chapter also briefly introduced textures and materials, how they are created, and applied onto the meshes. We will go into more details about Materials in the next chapter. So you might want to read [Chapter 4](ch04.html "Chapter 4. Material and Light"), *Material and Light*, first before creating/applying materials to your newly created game objects. To help you optimize your game, this chapter also covered the mesh creation pipeline and the concept of LOD. For interactions to take place, we also needed to learn how objects interact and collide with one another in Unreal, what object properties are configurable to allow different physics interaction.

This chapter also covered our first introduction to Blueprint, the graphical scripting of Unreal Engine4\. Through a simple Blueprint example, we learned how to turn on and off lights for our level using one of the many useful volumes that are in Unreal, Trigger Volume. In the next chapter, we will continue to build on the level we have created with more exciting materials and lights.