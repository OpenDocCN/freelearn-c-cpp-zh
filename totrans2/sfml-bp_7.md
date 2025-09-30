# Chapter 7. Building a Real-time Tower Defense Game from Scratch – Part 1

Now that you have all the basic tools, it's time for us to build something new. What about a mix of a **Real Time Strategy** (**RTS**) and a tower defense? And what about making it a multiplayer game? You like these ideas? Great! This is exactly what we will start building.

As this project is much more consequent than all the others, it will be split in two parts. The first one will focus on the game mechanism and logic, and the second on the multiplayer layer. So, in this chapter we will do the following:

*   Create animations
*   Build and use a generic map system with tile model and dynamic loading
*   Build an entity system
*   Make the game's logic

This project will reuse a lot of the components made previously, such as `ActionTarget`, `ResourceManager`, our GUI, and the game loop. To allow you to reuse these components easily for future projects, they have been gathered into a single framework (`SFML-utils`) that has been separated from the code in this book. This framework is available on the GitHub website at [https://github.com/Krozark/SFML-utils](https://github.com/Krozark/SFML-utils), due to which these components have been moved from the book namespace to `SFML-utils`. Moreover, the map and entity systems that will be explained in this chapter are also part of this framework.

The final result of this chapter will look as follows:

![Building a Real-time Tower Defense Game from Scratch – Part 1](img/8477OS_07_01.jpg)

# The goal of the game

First of all, let's explain our goal. As we said previously, we will build a new game that will be a mix of a real-time strategy game and tower defense.

The idea is that each team starts with some money/gold and a main building named GQ. When a team loses all its GQ, it loses the game. The money can be spent to build other buildings with different abilities, or to upgrade them. For example, some of the buildings will spawn warriors who will attack the enemies; other buildings will only defend the surrounding area. There is also a restriction concerning the area where new buildings can be made. In fact, you can only place a new building around your team's existing buildings. This keeps you from placing a big tower in the center of the enemy camp at the start of the game. It's also important to notice that once a building is built, you don't control its behavior just as you don't control the different warriors spawn by it.

Also, each time an enemy is destroyed, some gold is added to your gold stock, allowing you to build more towers, thus increasing your power to defeat your enemies.

Now that the game has been introduced, let's list our needs:

*   **Resources and event management**: These two features have been created previously, so we will just reuse them.
*   **GUI**: This feature has also been developed already in [Chapter 5](ch05.html "Chapter 5. Playing with User Interfaces"), *Playing with User Interfaces*. We will reuse it as is.
*   **Animation**: In SFML, there is no class to manage animated sprites in SFML, but for our game, we will need this functionality. So we will build it and add it to our framework.
*   **Tile map**: This functionality is very important and has to be as flexible as possible to allow us to reuse it in many other projects.
*   **Entity manager**: If you remember, this was introduced in [Chapter 3](ch03.html "Chapter 3. Making an Entire 2D Game"), *Making an Entire 2D Game*. Now it's time for us to really see it. This system will avoid a complex inheritance tree.

As you can see, this project is a bit more challenging than the previous one due its complexity, but it will also be much more interesting.

# Building animations

In all our previous games, all the different entities displayed on the screen were static; at least they were not animated. For a more attractive game, the simplest thing to do is add some animations and different entities on the player. For us, this will be applied on the different buildings and warriors.

As we use a sprite-based game and not real-time animation based on bone movement, we need some textures with the animations that are already prepared. So, our textures will look as shown in the following figure:

![Building animations](img/8477OS_07_02.jpg)

### Note

Note that the green grid is not a part of the image and is only shown here for information; the background is transparent in reality.

This type of texture is called a sprite sheet. In this example, the image can be split in two lines of four columns. Each line represents a direction of movement, namely left and right. Each cell of these lines represents a step of the future animation.

The aim of the work for this part is to be able to display a sprite using this sheet as an animation frame.

We will follow the design of the SFML by building two classes. The first one will store the animations and the second one will be used to display works such as `sf::Texture` and `sf::Sprite`. These two classes are named as `Animation` and `AnimatedSprite`.

## The Animation class

The `Animation` class only stores all the required data, for example, the textures and the different frames.

As this class is a kind of resource, we will use it through our `ResourceManager` class.

Here is the header of the class:

[PRE0]

As you can see, this class is nothing but a container for a texture and some rectangles. To simplify the usage of this class, some helper functions have been created, namely `addFramesLines()` and `addFramesColumn()`. Each of these functions add a complete line or column to the internal `_frames` list. The implementation of this class is also very simple and is as follows:

[PRE1]

The three `addFrameXXX()` functions allow us to add frames to our animation. The last two ones are some shortcuts to add an entire line or column. The rest of the methods allow us to access to the internal data.

Nothing more is required by our frame container. It's now time to build the `AnimatedSprite` class.

## The AnimatedSprite class

The `AnimatedSprite` class is in charge of the animation displayed on the screen. Due to this, it will keep a reference to an `Animation` class and will change the sub-rectangle of the texture periodically, just like `sf::Sprite`. We will also copy the `sf::Music`/`sf::Sound` API concerning the play/pause/stop ability. An `AnimatedSprite` instance should also be able to display on the screen and be transformable, due to which the class will inherit from `sf::Drawable` and `sf::Transformable`. We will also add a callback that will be triggered when the animation is complete. It could be interesting for the future.

The header looks as follows:

[PRE2]

As you can see, this class is bigger than the previous one. Its main functionality is to store an array of four vertices that will represent a frame taken from the associated animation. We also need some other information, such as the time between two frames, if the animation is a loop. This is why we need so many little functions. Now, let's see how all these are implemented:

[PRE3]

The constructor only initializes all the different attributes to their correct values:

[PRE4]

This function changes the current texture for a new one only if they are different, and resets the frame to the first one of the new animation. Note that at least one frame has to be stored in the new animation received as a parameter.

[PRE5]

All these functions are simple getters and setters. They allow us to manage basic elements of the `AnimatedSprite` class, as depicted in the previous code snippet.

[PRE6]

This function changes the current frame to a new one taken from the internal `Animation` class.

[PRE7]

This function changes the color mask of the displayed image. To do this, we set the color of each internal vertex to the new color received as a parameter:

[PRE8]

This function is the main one. Its job is to change from the current frame to the next one when the time limit is reached. Once we reach the last frame of the animation, you can do the following:

*   Reset the animation from the first one, depending of the `_loop` value
*   Reset the animation from the first one if the `_repeat` value authorizes us to do it
*   In all other cases, we trigger the event "on finish" by calling the internal callback

Now, take a look at the function that updates the frame's skin:

[PRE9]

This function is also an important one. Its aims is to update the attributes of the different vertices to those taken from the internal `Animation` class, namely the position and texture coordinates:

[PRE10]

The final function of this class manages the display. Because we inherit from `sf::Transformable`, we need to take into account the possible transformation. Then, we set the texture we used and finally draw the internal vertices array.

## A usage example

Now that we have the requisite classes to display an animation, let's build a little usage example.

Now, here's the implementation:

[PRE11]

For a better understanding of this code snippet, I've written some comments in the code.

This short program displays an animation on the screen. You can also change its position by moving it using the arrows on your keyboard. The animation will also change depending on the direction of movement.

Now that the first point of this chapter has been explained, let's continue to the second one, building a map.

# Building a generic Tile Map

For our project, we need something that will manage the map. In fact, the map is nothing but a big grid. The cells can be of any shape (square, hexagonal, and so on). The only restriction is that all the cells of a single map should have the same geometry.

Moreover, each cell can contain several objects, possibly of different types. For example, a cell can contain some background texture for the ground, a tree, and a bird. Because SFML doesn't use a `z` buffer with sprites (also called a depth buffer), we need to simulate it by hand. This is called the Painter's Algorithm. Its principle is very simple; draw everything but by depth order, starting with the most distant. It's how a tradition art painter would paint.

All this information brings us to the following structure:

*   A `Map` class must be of a specific geometry and must contain any number of layers sorted by their `z` buffer.
*   A `Layer` contains only a specific type. It also has a `z` buffer and stores a list of content sorted by their positions.
*   The `CONTENT` and `GEOMETRY` classes are template parameters but they need to have a specific API.

Here is the flowchart representing the class hierarchy of the previously explained structure:

![Building a generic Tile Map](img/8477OS_07_04.jpg)

Following is the explanation of the flowchart:

*   The `CONTENT` template class can be any class that inherits from `sf::Drawable` and `sf::Transformable`.
*   The `GEOMETRY` class is a new one that we will learn about shortly. It only defines the geometric shape and some helper functions to manipulate coordinates.
*   The `VLayer` class defines a common class for all the different types of layers.
*   The `Layer` class is just a container of a specific type with a depth variable that defines its draw order for the painter algorithm.
*   The `VMap` class defines a common API for the entire Map. It also contains a list of `VLayer` that is displayed using the painter algorithm.
*   The `Map` class inherits from `VMap` and is of a specific geometry.

## The Geometry class as an isometric hexagon

For our project, I made the choice of an isometric view with the tile as a hexagon. An isometric view is really simple to obtain but needs to be understood well. Following are the steps we need to follow:

1.  First, view your tile from the top view:![The Geometry class as an isometric hexagon](img/8477OS_07_05.jpg)
2.  Then, rotate it 45 degrees clockwise:![The Geometry class as an isometric hexagon](img/8477OS_07_06.jpg)
3.  Finally, divide its height by 2:![The Geometry class as an isometric hexagon](img/8477OS_07_07.jpg)
4.  You now have a nice isometric view. Now, let's take a look at the hexagon:![The Geometry class as an isometric hexagon](img/8477OS_07_08.jpg)

As you know, we need to calculate the coordinates of each of the edges using trigonometry, especially the Pythagoras theorem. This is without taking into account the rotation and the height resize. We need to follow two steps to find the right coordinates:

1.  Calculate the coordinates from the rotated shape (adding 45 degrees).
2.  Divide the total height value by two. By doing this, you will finally be able to build `sf::Shape`:

    [PRE12]

3.  The major part of the `GEOMETRY` class has been made. What remains is only a conversion from world to pixel coordinates, and the reverse. If you are interested in doing this, take a look at the class implementation in the `SFML-utils/src/SFML-utils/map/HexaIso.cpp` file.

Now that the main geometry has been defined, let's construct a `Tile<GEOMETRY>` class on it. This class will simply encapsulate `sf::Shape` , which is initialized by the geometry, and with the different requirements to be able to be use a `COMPONENT` parameter for the map. As this class is not very important, I will not explain it through this book, but you can take a look at its implementation in the `SFML-utils/include/SFML-utils/map/Tile.tpl` file.

## VLayer and Layer classes

The aim of a layer is to manage any number of components at the same depth. To do this, each layer contains its depth and a container of components. It also has the ability to resort the container to respect the painter algorithm. The `VLayer` class is an interface that only defines the API of the layer, allowing the map to store any kind of layer, thanks to polymorphism.

Here is the header of the `Layer` class:

[PRE13]

As mentioned previously, this class will not only store a container of its `template` class argument, but also its depth (`z`) and an is static Boolean member contained in the `Vlayer` class to optimize the display. The idea under this argument is that if the content within the layer doesn't move at all, it doesn't need to repaint the scene each time. The result is stored in an internal `sf::RenderTexture` parameter and will be refreshed only when the scene moves. For example, the ground never moves nor is it animated. So we can display it on a big texture and display this texture on the screen. This texture will be refreshed when the view is moved/resized.

To take this idea further, we only need to display content that appears on the screen. We don't need do draw something out of the screen. That's why we have the `viewport` attribute of the `draw()` method.

All other functions manage the content of the layer. Now, take a look at its implementation:

[PRE14]

This function adds new content to the layer, sort it if requested, and finally, return a reference to the new object:

[PRE15]

This function returns all the different objects to the same place. This is useful to pick up objects, for example, to pick objects under the cursor:

[PRE16]

This is the reverse function of `add()`. Using its address, it removes a component from the container:

[PRE17]

This function sorts all the content with respect to the painter algorithm order:

[PRE18]

This function is much more complicated than what we expect because of some optimizations. Let's explain it step by step:

*   First, we separate two cases. In the case of a static map we do as follows:

    *   Check if the view port has changed
    *   Resize the internal texture if needed
    *   Reset the textures

*   Draw each object with a position inside the view port into the `textureDisplay` the texture for the `RenderTarget` argument.
*   Draw each object with a position inside the view port into the `RenderTarget` argument if the layer contains dynamic objects (not static).

As you can see, the `draw()` function uses a naive algorithm in the case of dynamic content and optimizes the statics. To give you an idea of the benefits, with a layer of 10000 objects, the FPS was approximately 20\. With position optimization, it reaches 400, and with static optimization, 2,000\. So, I think the complexity of this function is justified by the enormous performance benefits.

Now that the `layer` class has been exposed to you, let's continue with the `map` class.

## VMap and Map classes

A map is a container of `VLayer`. It will implement the usual `add()`/`remove()` functions. This class can also be constructed from a file (described in the *Dynamic board loading* section) and handle unit conversion (coordinate to pixel and vice versa).

Internally, a `VMap` class store has the following layers:

[PRE19]

There are only two interesting functions in this class. The others are simply shortcuts, so I will not explain the entire class. Let us see the concerned functions:

[PRE20]

This function sorts the different layers by their `z` buffer with respect to the Painter's Algorithm. In fact, this function is simple but very important. We need to call it each time a layer is added to the map.

[PRE21]

The function draws each layer by calling its draw method; but first, we adjust the screen view port by adding a little delta on each of its borders. This is done to display all the tiles that appear on the screen, even partially (when its position is out on the screen).

## Dynamic board loading

Now that the map structure is done, we need a way to load it. For this, I've chosen the `JSON` format. There are two reasons for this choice:

*   It can be read by humans
*   The format is not verbose, so the final file is quite small even for big map

We will need some information to construct a map. This includes the following:

*   The map's geometry
*   The size of each tile (cell)
*   Define the layers as per the following:

    *   The `z` buffer
    *   If it is static or dynamic
    *   The content type

Depending on the content type of the layer, some other information to build this content could be specified. Most often, this extra information could be as follows:

*   Texture
*   Coordinates
*   Size

So, the `JSON` file will look as follows:

[PRE22]

As you can see, the different datasets are present to create a map with the isometric hexagon geometry with two layers. The first layer contains the grid with the ground texture and the second one contains some sprite for decoration.

To use this file, we need a `JSON` parser. You can use any existing one, build yours, or take the one built with this project. Next, we need a way to create an entire map from a file or update its content from a file. In the second case, the geometry will be ignored because we can't change the value of a template at runtime.

So, we will add a static method to the `VMap` class to create a new `Map`, and add another method to update its content. The signature will be as follows:

[PRE23]

The `loadFromJson()` function has to be virtual and implemented in the `Map` class because of the `GEOMETRY` parameter required by the `Tile` class. The `createMapFromFile()` function will be used internationally. Let's see its implementation:

[PRE24]

The goal of this function is pretty simple; construct the appropriate map depending on the geometry parameter and forward it the rest of the job.

[PRE25]

For a better understanding, the previous function was explained with raw comments. It's aimed at building layers and filling them with the data picked from the `JSON` file.

Now that we are able to build a map and fill it from a file, the last thing we need to do is display it on the screen. This will be done with the `MapViewer` class.

## The MapViewer class

This class encapsulates a `Map` class and manages some events such as mouse movement, moving the view, zoom, and so on. This is a really simple class with nothing new. This is why I will not go into details about anything but the `draw()` method (because of the view port). If you are interested in the full implementation, take a look at the `SFML-utils/src/SFML-utils/map/MapViewer.cpp` file.

So here is the draw method:

[PRE26]

As usual, we receive `sf::RenderTarget` and `sf::RenderStates` as parameters. However, here we don't want to interact with the current view of the target, so we make a backup of it and attach our local view to the rendered target. Then, we call the draw method of the internal map, forwarding the target, and states but adding the view port. This parameter is very important because it's used by our layers for optimization. So, we need to build a view port with the size of the rendered target, and thanks to SFML, it's very simple. We convert the top-left coordinate to the world coordinate, relative to our view. The result is in the top-left coordinate of the displayed area. Now, we only need the size. Here again, SFML provides use all the need: `sf::View::getSize()`. With this information, we are now able to build the correct view port and pass it to the map `draw()` function.

Once the rendering is complete, we restore the initial view back to the rendered target.

## A usage example

We now have all the requirements to load and display a map to the screen. The following code snippet shows you the minimal steps:

[PRE27]

The different steps of this function are as follows:

1.  Creating a window
2.  Creating a map from a file
3.  Process the events and quit if requests
4.  Update the viewer
5.  Display the viewer on the screen

The result will be as follows:

![A usage example](img/8477OS_07_09.jpg)

Now that the map is done, we need to fill it with some entities.

# Building an entity system

First of all, what is an entity system?

An **entity system** is a design pattern that focuses on data. Instead of creating a complex hierarchical tree of all possible entities, the idea is to build a system that allows us to add components to an entity at runtime. These components could be anything such as health points, artificial intelligence, skin, weapon, and everything but data.

However, if none of the entities and components hold functionalities, where are they stored? The answer is in the systems. Each system manages at least one component, and all the logic is inside these systems. Moreover, it is not possible to build an entity directly. You have to create or update it using an entity manager. It will be in charge of a set of entities, managing their components, creation, and destruction.

The structure is represented by the following chart:

![Building an entity system](img/8477OS_07_10.jpg)

There are many ways to implement such a structure. My choice was to use template and polymorphism.

## Use of the entity system

Without going much into the internal structure, we create a new component with this system as a structure, with no method except a constructor/destructor, and inherit from `Component` as follows:

[PRE28]

The inheritance is important to have a common base class between all the components. The same idea is used to create `System`:

[PRE29]

The reason for the inheritance is to have a common parent and API (the `update` function). Finally, to create an entity, you will have to do the following:

[PRE30]

If we continue this example, when an entity has no `hp`, we have to remove it from the board. This part of the logic is implemented inside the `SysHp::update()` function:

[PRE31]

This `SysHp::update()` function is used to create a specific functionality. Its aim is to remove all the entities with `hp` under or equal to zero. To do this, we initialize `ComponentHandler<CompHp>` using the `CompHp::Handle` shortcut (defined in the `Component` class). Then we create our query on the world. In our case, we need to get all the entities with `CompHp` attached to them. The multiple criteria query is also possible for more complex systems.

Once we have our view, we iterate on it. Each iteration gives us access to `Entity` and updates the handler values to the entity components. So, creating access to the `hp` handler is equivalent to the following:

[PRE32]

Then, we check the `_hp` value and remove the entity if needed.

It's important to note that the entity will actually be removed only when the `EntityManager::update()` function is called to keep data consistent inside the system loops.

Now that the `SysHp` parameter has been completed, we need to register it to `SystemManager` that is linked to `EntityManager`:

[PRE33]

We have now built an entity manager, a component, a system, and an entity. Putting them all together will result in the following code:

[PRE34]

This little code will create an entity and system manager. Then, we create 10 entities and add them to the `CompHp` component. Finally, we enter the game loop.

As mentioned previously, don't detail the implementation of the entity system; focus on its usage. If you are interested in the implementation, which is a bit complex, take a look at the files in the `SFML-utils/include/SFML-utils/es` directory. This is header only library.

## Advantages of the entity system approach

With a component system, each entity is represented as a single unique integer (its ID). These components are nothing but data. So, this is really simple to create a serialization function that saves the entire world. Database saving is made very simple with this approach but it's not the only point.

To create a flying car with a classic hierarchical tree, you have to inherit it from two different classes, namely car and flying vehicle. Each of these classes could inherit from the other. In fact, when the number of entities become large, the hierarchical tree is too much. For the same example, create an entity with the entity system, attach it to some wheels and wings. That's it! I agree that creating an entity system can be difficult, but its usage simplifies a lot the game's complexity.

# Building the game logic

We now have all the requirements to start our game: resource management, events management, GUI, animations, map, and the entity system. It's time for us to group them into a single project.

First, we need to create our entities. Thanks to the entity system previously described, we only need to build some components and their systems. We can build many of them, but the main components for the project are as follows:

| Components | Entities |
| --- | --- |
| Skin | Animation |
| Health points | Current health |
| Maximum health |
| Team | Identifier for the team |
| Build area | The authorized range around the entity |
| Movement | Speed |
| Destination |
| Artificial intelligence for warriors | Delta time |
| Damage |
| Length of hit |

The interesting ones are artificial intelligence (to damage) and movement. The others are pretty naive. Of course, you can create your own component in addition/replacement of those proposed.

## Building our components

We know all the data needed by our components, so let's build the two interesting components, namely the `walker AI` and the `warrior AI`:

[PRE35]

This component handles the speed and destination. The destination can be updated by anything (for example, when an enemy is detected at proximity):

[PRE36]

This component stores the aggressiveness of an entity, with its damaged, attack speed and area of aggressively.

As we will use this component in the system section, I will also explain the `CompSkin` component. This component stores an `AnimatedSkin` and different possible `Animation` that could be applied to it:

[PRE37]

Now that the components have been built, take a look at the systems.

## Creating the different systems

We need as many systems as the number of components. The skin system simply calls the update function on the animation. We have already built the related system for the health. For the team component, we don't need any system because this component is used only by artificial intelligence. The two systems left are more complex.

Let's start with the movement:

[PRE38]

Notice that the `Level` class has not yet been introduced. This class regroups an `EntityManager` and a `SystemManager` classes and gives us access to some functions concerning the map geometry, without having to know it. I will explain it later. In our case, we will need some information about the distance between the actual position of the component and its destination. This is why we need to keep a reference to the level.

Here's the implementation of the walker system:

[PRE39]

This system doesn't just move the entity but also makes different things. The position is stored inside the `CompSkin` component, so we need to iterate on the entities by getting the `CompAIWalker` and `CompSkin` components attached to them. Then, we calculate the position of the entity in the world coordinate and check if a move is needed. If we need to move, we calculate the vector corresponding to the total displacement (direction). This vector gives us the direction that the entity needs to follow. Then, we calculate the distance between the end point and the current position. Depending on the speed, we change the current position to the new one.

Once the movement is complete, we also change the current animation to the one matching the movement direction taken by the entity.

Now, let's take an interest in the `Warrior AI`:

[PRE40]

This system requires three components, namely `CompSkin` (for position), `CompTeam` (for detect enemy), and `CompAIWarrior`. The first thing to do is update the delta time. Then, we check if we have some enemies to defeat. Next, we search for an enemy who is closer (I won't detail this part because you can put your own algorithm). If an enemy is found, we check the distance between us and the enemy. If we can shoot the enemy, we do so and reset the delta time to avoid hitting each frame. We also trigger some events (for example, to create sound) and add gold to the team if we just kill the enemy. We also set the destination of the `CompAIWarrior` to the current position (to stay fighting) if we can, or move closer to the next enemy.

We now have all the components and systems to manage them. So, we will continue with the game architecture.

## The level class

As usual, we split the game into several parts. The `level` class represents a map. This class stores all the entities, systems, viewers, maps, sounds, and so on. As previously explained, it also implements an abstraction layer above the map geometry.

In fact, a level is a very simple object; it is just the glue between others. It registers all the systems, constructs the map, initializes a `MapViewer`, events, and regroups all the different update calls into one method. This class also offers users the ability to create new entities, by creating them through the internal `EntityManager`, and adding them to a map layer. The map is always synchronized with the `EntityManager` while doing this.

If you are interested in this implementation, take a look at the `SFML-book/07_2D_iso_game/src/SFML-Book/Level.cpp` file.

## The game class

Now, the `game` class! You should be familiar with this class by now. Its global behavior hasn't changed and still contains the same functionalities (`update()`, `processEvents()`, and `render()`).

The big change here is that the game class will initialize a `Level` and `Team`. One of these will be the one controlled by the player, and the GUI depends on it. This is the reason that the GUI for this project was attached to a team instead of the entire game. I won't say that it's the best way, but it's the simplest and allows us to jump from one team to another.

If you are interested in this implementation, take a look at the `SFML-book/07_2D_iso_game/src/SFML-Book/Game.cpp` file.

## The Team GUI class

This class handles different information and is the interface between the game and the player. It should allow the player to build some entities and interact with them.

The following screen shows you the **Build** menu. This menu shows the player the different entities that can be created and the current gold amount:

![The Team GUI class](img/8477OS_07_11.jpg)

Of course, we can complete this menu a lot, but this is the minimum information required by our game. Using our previously made GUI will facilitate this task a lot.

Once an entity is selected, we just have to place it into the game keeping in mind the following criteria:

*   The amount of gold
*   The build area

After this, everything will run easily. Don't hesitate to make some helper functions that create different entities by adding some components with specific values.

# Summary

In this chapter, we covered different things, such as creating animations. This class allowed us to display animated characters on screen. Then, we built a `Map` class that was filled with some entities. We also learned how to use an entity system by creating some components and systems to build our game logic. Finally, we put all the accumulated knowledge together to build a complete game with some artificial intelligence, a user interface, sounds, and animations.

With all this knowledge, you are now able to build any kind of game based on a tile system without too much effort.

In the next chapter, we will turn this game in a multiplayer one by using networking.