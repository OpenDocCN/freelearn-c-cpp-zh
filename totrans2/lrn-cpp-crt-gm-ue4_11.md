# Chapter 11. Monsters

We'll add in a bunch of opponents for the player.

What I've done in this chapter is added a landscape to the example. The player will walk along the path sculpted out for him and then he will encounter an army. There is an NPC before he reaches the army that will offer advice.

![Monsters](img/00160.jpeg)

The scene: starting to look like a game

# Landscape

We haven't covered how to sculpt the landscape in this book yet, but we'll do that here. First, you must have a landscape to work with. Start a new file by navigating to **File** | **New**. You can choose an empty level or a level with a sky. I chose the one without the sky in this example.

To create a landscape, we have to work from the **Modes** panel. Make sure that the **Modes** panel is displayed by navigating to **Window** | **Modes**:

![Landscape](img/00161.jpeg)

Displaying the modes panel

A landscape can be created in three steps, which are shown in the following screenshot, followed by the corresponding steps:

![Landscape](img/00162.jpeg)

1.  Click on the landscape icon (the picture of the mountains) in the **Modes** panel.
2.  Click on the **Manage** button.
3.  Next, click on the **Create** button in the lower right-hand corner of the screen.

You should now have a landscape to work with. It will appear as a gray, tiled area in the main window:

![Landscape](img/00163.jpeg)

The first thing you will want to do with your landscape scene is add some color to it. What's a landscape without colors? Right-click anywhere on your gray, tiled landscape object. In the **Details** panel at the right, you will see that it is populated with information, as shown in the following screenshot:

![Landscape](img/00164.jpeg)

Scroll down until you see the **Landscape Material** property. You can select the **M_Ground_Grass** material for a realistic-looking ground.

Next, add a light to the scene. You should probably use a directional light so that all of the ground has some light on it.

## Sculpting the landscape

A flat landscape can be boring. We will at least add some curves and hills to the place. To do so, click on the **Sculpt** button in the **Modes** panel:

![Sculpting the landscape](img/00165.jpeg)

To change the landscape, click on the Sculpt button

The strength and size of your brush are determined by the **Brush Size** and **Tool Strength** parameters in the **Modes** window.

Click on your landscape and drag the mouse to change the height of the turf. Once you're happy with what you've got, click on the **Play** button to try it out. The resultant output can be seen in the following screenshot:

![Sculpting the landscape](img/00166.jpeg)

Play around with your landscape and create a scene. What I did was lower the landscape around a flat ground plane, so the player has a well-defined flat area to walk on, as shown in the following screenshot:

![Sculpting the landscape](img/00167.jpeg)

Feel free to do whatever you like with your landscape. You can use what I'm doing here as inspiration, if you like. I will recommend that you import assets from **ContentExamples** or from **StrategyGame** in order to use them inside your game. To do this, refer to the *Importing assets* section in [Chapter 10](part0072_split_000.html#24L8G2-dd4a3f777fc247568443d5ffb917736d "Chapter 10. Inventory System and Pickup Items"), *Inventory System and Pickup Items*. When you're done importing assets, we can proceed to bring monsters into your world.

# Monsters

We'll start programming monsters in the same way we programmed NPCs and `PickupItem`. First, we'll write a base class (by deriving from character) to represent the `Monster` class. Then, we'll derive a bunch of blueprints for each monster type. Every monster will have a couple of properties in common that determine its behavior. These are the common properties:

*   A `float` variable for speed.
*   A `float` variable for the `HitPoints` value (I usually use floats for HP, so we can easily model HP leeching effects such as walking through a pool of lava).
*   An `int32` variable for the experience gained in defeating the monster.
*   A `UClass` function for the loot dropped by the monster.
*   A `float` variable for `BaseAttackDamage` done on each attack.
*   A `float` variable for `AttackTimeout`, which is the amount of time for which the monster rests between attacking.
*   Two `USphereComponents` object: One of them is `SightSphere`—how far he can see. The other is `AttackRangeSphere`, which is how far his attack reaches. The `AttackRangeSphere` object is always smaller than `SightSphere`.

Derive from the `Character` class to create your class for `Monster`. You can do this in UE4 by going to **File** | **Add Code To Project...** and then selecting the **Character** option from the menu for your base class.

Fill out the `Monster` class with the base properties. Make sure that you declare `UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = MonsterProperties)` so that the properties of the monsters can be changed in the blueprints:

[PRE0]

You will need some bare minimum code in your `Monster` constructor to get the monster's properties initialized. Use the following code in the `Monster.cpp` file:

[PRE1]

Compile and run the code. Open Unreal Editor and derive a blueprint based on your `Monster` class (call it `BP_Monster`). Now we can start configuring your monster's `Monster` properties.

For the skeletal mesh, we won't use the `HeroTPP` model for the monster because we need the monster to be able to do melee attacks and the `HeroTPP` model does not come with a melee attack. However, some of the models in the **Mixamo Animation Pack** file have melee attack animations. So download the **Mixamo Animation Pack** file from the UE4 marketplace (free).

![Monsters](img/00168.jpeg)

Inside the pack are some pretty gross models that I'd avoid, but others are quite good

Next, you should add the **Mixamo Animation Pack** file to your project, as shown in the following screenshot:

![Monsters](img/00169.jpeg)

Now, create a blueprint called `BP_Monster` based on your `Monster` class. Edit the blueprint's class properties and select **Mixamo_Adam** (it is actually typed as **Maximo_Adam** in the current issue of the package) as the skeletal mesh. Also, select **MixamoAnimBP_Adam** as the animation blueprint.

![Monsters](img/00170.jpeg)

Select the Maximo_Adam Skeletal Mesh and MixamoAnimBP_Adam for Anim Blueprint Generated Class

We will modify the animation blueprint to correctly incorporate the melee attack animation later.

While you're editing your `BP_Monster` blueprint, change the sizes of the `SightSphere` and `AttackRangeSphere` objects to values that make sense to you. I made my monster's `AttackRangeSphere` object just big enough to be about an arm's reach (60 units) and his `SightSphere` object to be 25 times bigger than that (about 1,500 units).

Remember that the monster will start moving towards the player once the player enters the monster's `SightSphere`, and the monster will start attacking the player once the player is inside the monster's `AttackRangeSphere` object.

![Monsters](img/00171.jpeg)

Mixamo Adam with his AttackRangeSphere object highlighted in orange

Place a few of your **BP_Monster** instances inside your game; compile and run. Without any code to drive the `Monster` character to move, your monsters should just stand there idly.

## Basic monster intelligence

In our game, we will add only a basic intelligence to the `Monster` characters. The monsters will know how to do two basic things:

*   Track the player and follow him
*   Attack the player

The monster will not do anything else. You can have the monster taunt the player when the player is first seen as well, but we'll leave that as an exercise for you.

### Moving the monster – steering behavior

Monsters in very basic games don't usually have complex motion behaviors. Usually they just walk towards the target and attack it. We'll program that type of monster in this game, but mind you, you can get more interesting play with monsters that position themselves advantageously on the terrain to perform ranged attacks and so on. We're not going to program that here, but it's something to think about.

In order to get the `Monster` character to move towards the player, we need to dynamically update the direction of the `Monster` character moving in each frame. To update the direction that the monster is facing, we write code in the `Monster::Tick()` method.

The `Tick` function runs in every frame of the game. The signature of the `Tick` function is:

[PRE2]

You need to add this function's prototype to your `AMonster` class in your `Monster.h` file. If we override `Tick`, we can place our own custom behavior that the `Monster` character should do in each frame. Here's some basic code that will move the monster toward the player during each frame:

[PRE3]

For `AddMovementInput` to work, you must have a controller selected under the **AIController Class** panel in your blueprint, as shown in the following screenshot:

![Moving the monster – steering behavior](img/00172.jpeg)

If you have selected `None`, calls to `AddMovementInput` won't have any effect. To prevent this, select either the `AIController` class or the `PlayerController` class as your **AIController Class**.

The preceding code is very simple. It comprises the most basic form of enemy intelligence: simply move toward the player by an incrementally small amount in each frame.

![Moving the monster – steering behavior](img/00173.jpeg)

Our not-so-intelligent army of monsters chasing the player

The result in a series of frames will be that the monster tracks and follows the player around the level. To understand how this works, you must remember that the `Tick` function is called on average about 60 times per second. What this means is that in each frame, the monster moves a tiny bit closer to the player. Since the monster moves in very small steps, his action looks smooth and continuous (while in reality, he is making small jumps and leaps in each frame).

![Moving the monster – steering behavior](img/00174.jpeg)

Discrete nature of tracking: a monster's motion over three superimposed frames

### Tip

The reason why the monster moves about 60 times a second is because of a hardware constraint. The refresh rate of a typical monitor is 60 Hz, so it acts as a practical limiter on how many updates per second are useful. Updating at a frame rate faster than the refresh rate is possible, but it is not necessarily useful for games since you will only see a new picture once every 1/60 of a second on most hardware. Some advanced physics modeling simulations do almost 1,000 updates a second, but arguably, you don't need that kind of resolution for a game and you should reserve the extra CPU time for something that the player will enjoy instead, such as better AI algorithms. Some newer hardware boasts of a refresh rate up to 120 Hz (look up gaming monitors, but don't tell your parents I asked you to blow all your money on one).

### The discrete nature of monster motion

Computer games are discrete in nature. In the preceding screenshot of superimposed sequences of frames, the player is seen as moving straight up the screen, in tiny steps. The motion of the monster is also in small steps. In each frame, the monster takes one small discrete step towards the player. The monster is following an apparently curved path as he moves directly toward where the player is in each frame.

To move the monster toward the player, we first have to get the player's position. Since the player is accessible in a global function, `UGameplayStatics::GetPlayerPawn`, we simply retrieve our pointer to the player using this function. Next we find the vector pointing from the `Monster` (`GetActorLocation()`) function that points to the player (`avatar->GetActorLocation()`). We need to find the vector that points from the monster to the avatar. To do this, you have to subtract the location of the monster from the location of the avatar, as shown in the following screenshot:

![The discrete nature of monster motion](img/00175.jpeg)

It's a simple math rule to remember but often easy to get wrong. To get the right vector, always subtract the source (the starting point) vector from the target (the terminal point) vector. In our system, we have to subtract the `Monster` vector from the `Avatar` vector. This works because subtracting the `Monster` vector from the system moves the `Monster` vector to the origin and the `Avatar` vector will be to the lower left-hand side of the `Monster` vector:

![The discrete nature of monster motion](img/00176.jpeg)

Subtracting the Monster vector from the system moves the Monster vector to (0,0)

Be sure to try out your code. At this point, the monsters will be running toward your player and crowding around him. With the preceding code that is outlined, they won't attack; they'll just follow him around, as shown in the following screenshot:

![The discrete nature of monster motion](img/00177.jpeg)

### Monster SightSphere

Right now, the monsters are not paying attention to the `SightSphere` component. That is, wherever the player is in the world, the monsters will move toward him in the current setup. We want to change that now.

To do so, all we have to do is have `Monster` respect the `SightSphere` restriction. If the player is inside the monster's `SightSphere` object, the monster will give chase. Otherwise, the monsters will be oblivious to the player's location and not chase the player.

Checking to see if an object is inside a sphere is simple. In the following screenshot, the point **p** is inside the sphere if the distance **d** between **p** and the centroid **c** is less than the sphere radius, **r**:

![Monster SightSphere](img/00178.jpeg)

P is inside the sphere when d is less than r

So, in our code, the preceding screenshot translates to the following code:

[PRE4]

The preceding code adds additional intelligence to the `Monster` character. The `Monster` character can now stop chasing the player if the player is outside the monster's `SightSphere` object. This is how the result will look:

![Monster SightSphere](img/00179.jpeg)

A good thing to do here will be to wrap up the distance comparison into a simple inline function. We can provide these two inline member functions in the `Monster` header as follows:

[PRE5]

These functions return the value `true` when the passed parameter, `d`, is inside the spheres in question.

### Tip

An `inline` function means that the function is more like a macro than a function. Macros are copied and pasted to the calling location, while functions are jumped to by C++ and executed at their location. Inline functions are good because they give good performance while keeping the code easy to read and they are reusable.

# Monster attacks on the player

There are a few different types of attacks that monsters can do. Depending on the type of the `Monster` character, a monster's attack might be melee (close range) or ranged (projectile weapon).

The `Monster` character will attack the player whenever the player is in his `AttackRangeSphere`. If the player is out of the range of the monster's `AttackRangeSphere` but the player is in the `SightSphere` object of the monster, then the monster will move closer to the player until the player is in the monster's `AttackRangeSphere`.

## Melee attacks

The dictionary definition of *melee* is a confused mass of people. A melee attack is one that is done at a close range. Picture a bunch of *zerglings* battling it out with a bunch of *ultralisks* (if you're a *StarCraft* player, you'll know that both zerglings and ultralisks are melee units). Melee attacks are basically close range, hand-to-hand combat. To do a melee attack, you need a melee attack animation that turns on when the monster begins his melee attack. To do this, you need to edit the animation blueprint in *Persona*, UE4's animation editor.

### Tip

Zak Parrish's *Persona* series is an excellent place to get started with in order to program animations in blueprints: [https://www.youtube.com/watch?v=AqYmC2wn7Cg&list=PL6VDVOqa_mdNW6JEu9UAS_s40OCD_u6yp&index=8](https://www.youtube.com/watch?v=AqYmC2wn7Cg&list=PL6VDVOqa_mdNW6JEu9UAS_s40OCD_u6yp&index=8).

For now, we will just program the melee attack and then worry about modifying the animation in blueprints later.

### Defining a melee weapon

There are going to be three parts to defining our melee weapon. The first part is the C++ code that represents it. The second is the model, and the third is to connect the code and model together using a UE4 blueprint.

#### Coding for a melee weapon in C++

We will define a new class, `AMeleeWeapon` (derived from `AActor`), to represent hand-held combat weapons. I will attach a couple of blueprint-editable properties to the `AMeleeWeapon` class, and the `AMeleeWeapon` class will look as shown in the following code:

[PRE6]

Notice how I used a bounding box for `ProxBox`, and not a bounding sphere. This is because swords and axes will be better approximated by boxes rather than spheres. There are two member functions, `Rest()` and `Swing()`, which let `MeleeWeapon` know what state the actor is in (resting or swinging). There's also a `TArray<AActor*> ThingsHit` property inside this class that keeps track of the actors hit by this melee weapon on each swing. We are programming it so that the weapon can only hit each thing once per swing.

The `AMeleeWeapon.cpp` file will contain just a basic constructor and some simple code to send damages to `OtherActor` when our sword hits him. We will also implement the `Rest()` and `Swing()` functions to clear the list of things hit. The `MeleeWeapon.cpp` file has the following code:

[PRE7]

#### Downloading a sword

To complete this exercise, we need a sword to put into the model's hand. I added a sword to the project called *Kilic* from [http://tf3dm.com/3d-model/sword-95782.html](http://tf3dm.com/3d-model/sword-95782.html) by Kaan Gülhan. The following is a list of other places where you will get free models:

*   [http://www.turbosquid.com/](http://www.turbosquid.com/)
*   [http://tf3dm.com/](http://tf3dm.com/)
*   [http://archive3d.net/](http://archive3d.net/)
*   [http://www.3dtotal.com/](http://www.3dtotal.com/)

### Tip

**Secret tip**

It might appear at first on [TurboSquid.com](http://TurboSquid.com) that there are no free models. In fact, the secret is that you have to search in the price range $0-$0 to find them. $0 means free.

![Downloading a sword](img/00180.jpeg)

TurboSquid's search for free swords

I had to edit the *kilic* sword mesh slightly to fix the initial sizing and rotation. You can import any mesh in the **Filmbox** (**FBX**) format into your game. The kilic sword model is in the sample code package for [Chapter 11](part0076_split_000.html#28FAO1-dd4a3f777fc247568443d5ffb917736d "Chapter 11. Monsters"), *Monsters*.

To import your sword into the UE4 editor, right-click on any folder you want to add the model to. Navigate to **New Asset** | **Import to** | **Game** | **Models...**, and from the file explorer that pops up, select the new asset you want to import. If a **Models** folder doesn't exist, you can create one by simply right-clicking on the tree view at the left and selecting **New Folder** in the pane on the left-hand side of the **Content Browser** tab. I selected the `kilic.fbx` asset from my desktop.

![Downloading a sword](img/00181.jpeg)

Importing to your project

#### Creating a blueprint for your melee weapon

Inside the UE4 editor, create a blueprint based off of `AMeleeWeapon` called `BP_MeleeSword`. Configure `BP_MeleeSword` to use the *kilic* blade model (or any blade model you choose), as shown in the following screenshot:

![Creating a blueprint for your melee weapon](img/00182.jpeg)

The `ProxBox` class will determine whether something was hit by the weapon, so we will modify the `ProxBox` class such that it just encloses the blade of the sword, as shown in the following screenshot:

![Creating a blueprint for your melee weapon](img/00183.jpeg)

Also, under the **Collision Presets** panel, it is important to select the **NoCollision** option for the mesh (not **BlockAll**). This is illustrated in the following screenshot:

![Creating a blueprint for your melee weapon](img/00184.jpeg)

If you select **BlockAll**, then the game engine will automatically resolve all the interpenetration between the sword and the characters by pushing away things that the sword touches whenever it is swung. The result is that your characters will appear to go flying whenever a sword is swung.

## Sockets

A socket in UE4 is a receptacle on one skeletal mesh for another `Actor`. You can place a socket anywhere on a skeletal mesh body. After you have correctly placed the socket, you can attach another `Actor` to this socket in UE4 code.

For example, if we want to put a sword in our monster's hand, we'd just have to create a socket in our monster's hand. We can attach a helmet to the player by creating a socket on his head.

### Creating a skeletal mesh socket in the monster's hand

To attach a socket to the monster's hand, we have to edit the skeletal mesh that the monster is using. Since we used the `Mixamo_Adam` skeletal mesh for the monster, we have to open and edit this skeletal mesh.

To do so, double-click on the **Mixamo_Adam** skeletal mesh in the **Content Browser** tab (this will appear as the T-pose) to open the skeletal mesh editor. If you don't see **Mixamo Adam** in your **Content Browser** tab, make sure that you have imported the **Mixamo Animation Pack** file into your project from the Unreal Launcher app.

![Creating a skeletal mesh socket in the monster's hand](img/00185.jpeg)

Edit the Maximo_Adam mesh by double-clicking on the Maximo_Adam skeletal mesh object

Click on **Skeleton** at the top-right corner of the screen. Scroll down the tree of bones in the left-hand side panel until you find the **RightHand** bone. We will attach a socket to this bone. Right-click on the **RightHand** bone and select **Add Socket**, as shown in the following screenshot:

![Creating a skeletal mesh socket in the monster's hand](img/00186.jpeg)

You can leave the default name (**RightHandSocket**) or rename the socket if you like, as shown in the following screenshot:

![Creating a skeletal mesh socket in the monster's hand](img/00187.jpeg)

Next, we need to add a sword to the actor's hand.

### Attaching the sword to the model

With the Adam skeletal mesh open, find the **RightHandSocket** option in the tree view. Since Adam swings with his right hand, you should attach the sword to his right hand. Drag and drop your sword model into the **RightHandSocket** option. You should see Adam grip the sword in the image of the model at the right-hand side of the following screenshot:

![Attaching the sword to the model](img/00188.jpeg)

Now, click on **RightHandSocket** and zoom in on Adam's hand. We need to adjust the positioning of the socket in the preview so that the sword fits in it correctly. Use the move and rotate manipulators to line the sword up so that it fits in his hand correctly.

![Attaching the sword to the model](img/00189.jpeg)

Positioning the socket in the right hand so that the sword rests correctly

### Tip

**A real-world tip**

If you have several sword models that you want to switch in and out of the same **RightHandSocket**, you will need to ensure quite a bit of uniformity (lack of anomalies) between the different swords that are supposed to go in that same socket.

You can preview your animations with the sword in the hand by going to the **Animation** tab in the top-right corner of the screen.

![Attaching the sword to the model](img/00190.jpeg)

Equipping the model with a sword

However, if you launch your game, Adam won't be holding a sword. That's because adding the sword to the socket in *Persona* is for preview purposes only.

### Code to equip the player with a sword

To equip your player with a sword from the code and permanently bind it to the actor, instantiate an `AMeleeWeapon` instance and attach it to `RightHandSocket` after the monster instance is initialized. We do this in `PostInitializeComponents()` since in this function the `Mesh` object will have been fully initialized already.

In the `Monster.h` file, add a hook to select the **Blueprint** class name (`UClass`) of a melee weapon to use. Also add a hook for a variable to actually store the `MeleeWeapon` instance using the following code:

[PRE8]

Now, select the `BP_MeleeSword` blueprint in your monster's blueprint class.

In the C++ code, you need to instantiate the weapon. To do so, we need to declare and implement a `PostInitializeComponents` function for the `Monster` class. In `Monster.h`, add a prototype declaration:

[PRE9]

`PostInitializeComponents` runs after the monster object's constructor has completed and all the components of the object are initialized (including the blueprint construction). So it is the perfect time to check whether the monster has a `MeleeWeapon` blueprint attached to it or not and to instantiate this weapon if it does. The following code is added to instantiate the weapon in the `Monster.cpp` implementation of `AMonster::PostInitializeComponents()`:

[PRE10]

The monsters will now start with swords in hand if `BPMeleeWeapon` is selected for that monster's blueprint.

![Code to equip the player with a sword](img/00191.jpeg)

Monsters holding weapons

### Triggering the attack animation

By default, there is no connection between our C++ `Monster` class and triggering the attack animation; in other words, the `MixamoAnimBP_Adam` class has no way of knowing when the monster is in the attack state.

Therefore, we need to update the animation blueprint of the Adam skeleton (`MixamoAnimBP_Adam`) to include a query in the `Monster` class variable listing and check whether the monster is in an attacking state. We haven't worked with animation blueprints (or blueprints in general) in this book before, but follow it step by step and you should see it come together.

### Tip

I will introduce blueprints terminology gently here, but I will encourage you to have a look at Zak Parrish's tutorial series at [https://www.youtube.com/playlist?list=PLZlv_N0_O1gbYMYfhhdzfW1tUV4jU0YxH](https://www.youtube.com/playlist?list=PLZlv_N0_O1gbYMYfhhdzfW1tUV4jU0YxH) for your first introduction to blueprints.

#### Blueprint basics

A UE4 blueprint is a visual realization of the code (not to be confused with how sometimes people say that a C++ class is a metaphorical blueprint of a class instance). In UE4 blueprints, instead of actually writing code, you drag and drop elements onto a graph and connect them to achieve desired play. By connecting the right nodes to the right elements, you can program anything you want in your game.

### Tip

This book does not encourage the use of blueprints since we are trying to encourage you to write your own code instead. Animations, however, are best worked with blueprints, because that is what artists and designers will know.

Let's start writing a sample blueprint to get a feel how they work. First, click on the blueprint menu bar at the top and select **Open Level Blueprint**, as shown in the following screenshot:

![Blueprint basics](img/00192.jpeg)

The **Level Blueprint** option executes automatically when you begin the level. Once you open this window, you should see a blank slate to create your gameplay on, as shown here:

![Blueprint basics](img/00193.jpeg)

Right-click anywhere on the graph paper. Start typing `begin` and click on the **Event Begin Play** option from the drop-down list that appears. Ensure that the **Context Sensitive** checkbox is checked, as shown in the following screenshot:

![Blueprint basics](img/00194.jpeg)

Immediately after you click on the **Event Begin Play** option, a red box will appear on your screen. It has a single white pin at the right-hand side. This is called the execution pin, as shown here:

![Blueprint basics](img/00195.jpeg)

The first thing that you need to know about animation blueprints is the white pin execution path (the white line). If you've seen a blueprint graph before, you must have noticed a white line going through the graph, as shown in the following diagram:

![Blueprint basics](img/00196.jpeg)

The white pin execution path is pretty much equivalent to having lines of code lined up and run one after the other. The white line determines which nodes will get executed and in what order. If a node does not have a white execution pin attached to it, then that node will not get executed at all.

Drag off the white execution pin from **Event Begin Play**. Start by typing `draw debug box` in the **Executable actions** dialog. Select the first thing that pops up (**f** **Draw Debug Box**), as shown here:

![Blueprint basics](img/00197.jpeg)

Fill in some details for how you want the box to look. Here, I selected the color blue for the box, the center of the box at (0, 0, 100), the size of the box to be (200, 200, 200), and a duration of 180 seconds (be sure to enter a duration that is long enough to see the result), as shown in the following screenshot:

![Blueprint basics](img/00198.jpeg)

Now click on the **Play** button to realize the graph. Remember that you have to find the world's origin to see the debug box.

Find the world's origin by placing a golden egg at (0, 0, (some z value)), as shown in the following screenshot:

![Blueprint basics](img/00199.jpeg)

This is how the box will look in the level:

![Blueprint basics](img/00200.jpeg)

A debug box rendered at the origin

#### Modifying the animation blueprint for Mixamo Adam

To integrate our attack animation, we have to modify the blueprint. Under **Content Browser**, open up `MixamoAnimBP_Adam`.

The first thing you'll notice is that the graph has two sections: a top section and a bottom section. The top section is marked "**Basic Character movement**...," while the bottom section says "**Mixamo Example Character Animation**...." Basic character movement is in charge of the walking and running movements of the model. We will be working in the **Mixamo Example Character Animation with Attack and Jump** section, which is responsible for the attack animation. We will be working in the latter section of the graph, shown in the following screenshot:

![Modifying the animation blueprint for Mixamo Adam](img/00201.jpeg)

When you first open the graph, it starts out by zooming in on a section near the bottom. To scroll up, right-click the mouse and drag it upwards. You can also zoom out using the mouse wheel or by holding down the *Alt* key and the right mouse button while moving the mouse up.

Before proceeding, you might want to duplicate the **MixamoAnimBP_Adam** resource so that you don't damage the original, in case you need to go back and change something later. This allows you to easily go back and correct things if you find that you've made a mistake in one of your modifications, without having to reinstall a fresh copy of the whole animation package into your project.

![Modifying the animation blueprint for Mixamo Adam](img/00202.jpeg)

Making a duplicate of the MixamoAnimBP_Adam resource to avoid damaging the original asset

### Tip

When assets are added to a project from the Unreal Launcher, a copy of the original asset is made, so you can modify **MixamoAnimBP_Adam** in your project now and get a fresh copy of the original assets in a new project later.

We're going to do only a few things to make Adam swing the sword when he is attacking. Let's do it in order.

1.  Deleting the node that says **Attacking?**:![Modifying the animation blueprint for Mixamo Adam](img/00203.jpeg)
2.  Rearrange the nodes, as follows, with the **Enable Attack** node by itself at the bottom:![Modifying the animation blueprint for Mixamo Adam](img/00204.jpeg)
3.  Next we're going to handle the monster that this animation is animating. Scroll up the graph a bit and drag the blue dot marked as **Return Value** in the **Try Get Pawn Owner** dialog. Drop it into your graph, and when the pop-up menu appears, select **Cast to Monster** (ensure that **Context Sensitive** is checked, or the **Cast to Monster** option will not appear). The **Try Get Pawn Owner** option gets the `Monster` instance that owns the animation, which is just the `AMonster` class object, as shown in the following screenshot:![Modifying the animation blueprint for Mixamo Adam](img/00205.jpeg)
4.  Click on **+** in the **Sequence** dialog and drag another execution pin from the **Sequence** group to the **Cast to Monster** node instance, as shown in the following screenshot. This ensures that the **Cast to Monster** instance actually gets executed.![Modifying the animation blueprint for Mixamo Adam](img/00206.jpeg)
5.  The next step is to pull out the pin from the **As Monster** terminal of the **Cast to Monster** node and look for the **Is in Attack Range Of Player** property:![Modifying the animation blueprint for Mixamo Adam](img/00207.jpeg)
6.  Take the white execution pin from the **Cast to Monster** node at the left-hand side and drop it into the **Is in Attack Range Of Player** node at the right-hand side:![Modifying the animation blueprint for Mixamo Adam](img/00208.jpeg)

    This ensures a transfer of control from the **Cast to Monster** operation to the **Is in Attack Range Of Player** node.

7.  Pull the white and red pins over to the **SET** node, as shown here:![Modifying the animation blueprint for Mixamo Adam](img/00209.jpeg)

### Tip

The equivalent pseudocode of the preceding blueprint is something similar to the following:

[PRE11]

Test your animation. The monster should swing only when he is within the player's range.

#### Code to swing the sword

We want to add an animation notify event when the sword is swung. First, declare and add a blueprint callable C++ function to your `Monster` class:

[PRE12]

The `BlueprintCallable` statement means that it will be possible to call this function from blueprints. In other words, `SwordSwung()` will be a C++ function that we can invoke from a blueprints node, as shown here:

[PRE13]

Next open the **Mixamo_Adam_Sword_Slash** animation by double-clicking on it from your **Content Browser** (it should be in **MixamoAnimPack/Mixamo_Adam/Anims/Mixamo_Adam_Sword_Slash**). Scrub the animation to the point where Adam starts swinging his sword. Right-click on the animation bar and select **New Notify** under **Add Notify...**, as shown in the following screenshot:

![Code to swing the sword](img/00210.jpeg)

Name the notification `SwordSwung`:

![Code to swing the sword](img/00211.jpeg)

The notification name should appear in your animation's timeline, shown as follows:

![Code to swing the sword](img/00212.jpeg)

Save the animation and then open up your version of **MixamoAnimBP_Adam** again. Underneath the **SET** group of nodes, create the following graph:

![Code to swing the sword](img/00213.jpeg)

The **AnimNotify_SwordSwung** node appears when you right-click in the graph (with **Context Sensitive** turned on) and start typing `SwordSwung`. The **Cast To Monster** node is again fed in from the **Try Get Pawn Owner** node as in step 2 of the *Modifying the animation blueprint for Mixamo Adam* section. Finally, **Sword Swung** is our blueprint-callable C++ function in the `AMonster` class.

If you start the game now, your monsters will execute their attack animation whenever they actually attack. When the sword's bounding box comes in contact with you, you should see your HP bar go down a bit (recall that the HP bar was added at the end of [Chapter 8](part0056_split_000.html#1LCVG1-dd4a3f777fc247568443d5ffb917736d "Chapter 8. Actors and Pawns"), *Actors and Pawns*, as an exercise).

![Code to swing the sword](img/00214.jpeg)

Monsters attacking the player

## Projectile or ranged attacks

Ranged attacks usually involve a projectile of some sort. Projectiles are things such as bullets, but they can also include things such as lightning magic attacks or fireball attacks. To program a projectile attack, you should spawn a new object and only apply the damage to the player if the projectile reaches the player.

To implement a basic bullet in UE4, we should derive a new object type. I derived a `ABullet` class from the `AActor` class, as shown in the following code:

[PRE14]

The `ABullet` class has a couple of important members in it, as follows:

*   A `float` variable for the damage that a bullet does on contact
*   A `Mesh` variable for the body of the bullet
*   A `ProxSphere` variable to detect when the bullet finally hits something
*   A function to be run when `Prox` near an object is detected

The constructor for the `ABullet` class should have the initialization of the `Mesh` and `ProxSphere` variables. In the constructor, we set `RootComponent` to being the `Mesh` variable and then attach the `ProxSphere` variable to the `Mesh` variable. The `ProxSphere` variable will be used for collision checking, and collision checking for the `Mesh` variable should be turned off, as shown in the following code:

[PRE15]

We initialized the `Damage` variable to `1` in the constructor, but this can be changed in the UE4 editor once we create a blueprint out of the `ABullet` class. Next, the `ABullet::Prox_Implementation()` function should deal damages to the actor hit if we collide with the other actor's `RootComponent`, using the following code:

[PRE16]

### Bullet physics

To make bullets fly through the level, you can use UE4's physics engine.

Create a blueprint based on the `ABullet` class. I selected **Shape_Sphere** for the mesh. The bullet's mesh should not have collision physics enabled; instead we'll enable physics on the bullet's bounding sphere.

Configuring the bullet to behave properly is mildly tricky, so we'll cover this in four steps, as follows:

1.  Select **[ROOT] ProxSphere** in the **Components** tab. The `ProxSphere` variable should be the root component and should appear at the top of the hierarchy.
2.  In the **Details** tab, check both **Simulate Physics** and **Simulation Generates Hit Events**.
3.  From the **Collision Presets** dropdown, select **Custom…**.
4.  Check the **Collision Responses** boxes as shown; check **Block** for most types (**WorldStatic**, **WorldDynamic**, and so on) and check **Overlap** only for **Pawn**:![Bullet physics](img/00215.jpeg)

The **Simulate Physics** checkbox makes the `ProxSphere` property experience gravity and the impulse forces exerted on it. An impulse is a momentary thrust of force, which we'll use to drive the shot of the bullet. If you do not check the **Simulation Generate Hit Events** checkbox, then the ball will drop on the floor. What **BlockAll Collision Preset** does is ensure that the ball can't pass through anything.

If you drag and drop a couple of these `BP_Bullet` objects from the **Content Browser** tab directly into the world now, they will simply fall to the floor. You can kick them once they are on the the floor. The following screenshot shows the ball object on the floor:

![Bullet physics](img/00216.jpeg)

However, we don't want our bullets falling on the floor. We want them to be shot. So let's put our bullets in the `Monster` class.

### Adding bullets to the monster class

Add a member to the `Monster` class that receives a blueprint instance reference. That's what the `UClass` object type is for. Also, add a blueprint configurable float property to adjust the force that shoots the bullet, as shown in the following code:

[PRE17]

Compile and run the C++ project and open your `BP_Monster` blueprint. You can now select a blueprint class under `BPBullet`, as shown in the following screenshot:

![Adding bullets to the monster class](img/00217.jpeg)

Once you've selected a blueprint class type to instantiate when the monster shoots, you have to program the monster to shoot when the player is in his range.

Where does the monster shoot from? Actually, he should shoot from a bone. If you're not familiar with the terminology, bones are just reference points in the model mesh. A model mesh is usually made up of many "bones." To see some bones, open up the **Mixamo_Adam** mesh by double-clicking on the asset in the **Content Browser** tab, as shown in the following screenshot:

![Adding bullets to the monster class](img/00218.jpeg)

Go to the **Skeleton** tab and you will see all the monster's bones in a tree view list in the left-hand side. What we want to do is select a bone from which bullets will be emitted. Here I've selected the `LeftHand` option.

### Tip

An artist will normally insert an additional bone into the model mesh to emit the particle, which is likely to be on the tip of the nozzle of a gun.

Working from the base model mesh, we can get the `Mesh` bone's location and have the monster emit the `Bullet` instances from that bone in the code.

The complete monster `Tick` and `Attack` functions can be obtained using the following code:

[PRE18]

The `AMonster::Attack` function is relatively simple. Of course, we first need to add a prototype declaration in the `Monster.h` file in order to write our function in the `.cpp` file:

[PRE19]

In `Monster.cpp`, we implement the `Attack` function, as follows:

[PRE20]

We leave the code that implements the melee attack as it is. Assuming that the monster is not holding a melee weapon, we then check whether the `BPBullet` member is set. If the `BPBullet` member is set, it means that the monster will create and fire an instance of the `BPBullet` blueprinted class.

Pay special attention to the following line:

[PRE21]

This is how we add a new actor to the world. The `SpawnActor()` function puts an instance of `UCLASS` that you pass, at `spawnLoc`, with some initial orientation.

After we spawn the bullet, we call the `AddImpulse()` function on its `ProxSphere` variable to rocket it forward.

## Player knockback

To add a knockback to the player, I added a member variable to the `Avatar` class called `knockback`. A knockback happens whenever the avatar gets hurt:

[PRE22]

In order to figure out the direction to knock the player back when he gets hit, we need to add some code to `AAvatar::TakeDamage`. Compute the direction vector from the attacker towards the player and store this vector in the `knockback` variable:

[PRE23]

In `AAvatar::Tick`, we apply the knockback to the avatar's position:

[PRE24]

Since the knockback vector reduces in size with each frame, it becomes weaker over time, unless the knockback vector gets renewed with another hit.

# Summary

In this chapter, we explored how to instantiate monsters on the screen that run after the player and attack him. In the next chapter, we will give the player the ability to defend himself by allowing him to cast spells that damage the monsters.