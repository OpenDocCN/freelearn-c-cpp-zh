# Chapter 12. Spell Book

The player does not yet have a means to defend himself. We will equip the player with a very useful and interesting way, of doing so called magic spells. Magic spells will be used by the player to affect monsters nearby.

Practically, spells will be a combination of a particle system with an area of effect represented by a bounding volume. The bounding volume is checked for contained actors in each frame. When an actor is within the bounding volume of a spell, then that actor is affected by that spell.

The following is a screenshot of the blizzard and force field spells, with their bounding volumes highlighted in orange:

![Spell Book](img/00219.jpeg)

Visualization of the blizzard spell can be seen at the right, with a long, box-shaped bounding volume. Visualization of the force field spell, with a spherical bounding volume, for pushing monsters away, is shown in the following screenshot:

![Spell Book](img/00220.jpeg)

In each frame, the bounding volume is checked for contained actors. Any actor contained in the spell's bounding volume is going to be affected by that spell for that frame only. If the actor moves outside the spell's bounding volume, the actor will no longer be affected by that spell. Remember, the spell's particle system is a visualization only; the particles themselves are not what will affect game actors. The `PickupItem` class we created in [Chapter 8](part0056_split_000.html#1LCVG1-dd4a3f777fc247568443d5ffb917736d "Chapter 8. Actors and Pawns"), *Actors and Pawns* can be used to allow the player to pick up items representing the spells. We will extend the `PickupItem` class and attach the blueprint of a spell to cast each `PickupItem`. Clicking on a spell's widget from the HUD will cast it. The interface will look something like this:

![Spell Book](img/00221.jpeg)

Items the player has picked up, including four different spells

We'll begin the chapter by describing how to create our own particle systems. We'll then move on to wrap up the particle emitter into a `Spell` class, and write a `CastSpell()` function for the avatar to be able to actually cast spells.

# The particle systems

First, we need a place to put all our snazzy effects. In your **Content Browser** tab, right-click on the **Game** root and create a new folder called **ParticleSystems**. Right-click on that new folder, and select **New Asset** | **Particle System**, as shown in the following screenshot:

![The particle systems](img/00222.jpeg)

### Tip

See this Unreal Engine 4 Particle Systems guide for information on how unreal particle emitters work: [https://www.youtube.com/watch?v=OXK2Xbd7D9w&index=1&list=PLZlv_N0_O1gYDLyB3LVfjYIcbBe8NqR8t](https://www.youtube.com/watch?v=OXK2Xbd7D9w&index=1&list=PLZlv_N0_O1gYDLyB3LVfjYIcbBe8NqR8t).

Double-click on the **NewParticleSystem** icon that appears, as shown in the following screenshot:

![The particle systems](img/00223.jpeg)

You will be in Cascade, the particle editor. A description of the environment is shown in the following screenshot:

![The particle systems](img/00224.jpeg)

There are several different panes here, each of which shows different information. They are as follows:

*   At the top left is the **Viewport** pane. This shows you an animation of the current emitter as its currently working.
*   At the right is the **Emitters** pane. Inside it, you can see a single object called **Particle Emitter** (you can have more than one emitter in your particle system, but we don't want that now). The listing of modules of **Particle Emitter** appears listed under it. From the preceding screenshot, we have the **Required**, **Spawn**, **Lifetime**, **Initial Size**, **Initial Velocity**, and **Color Over Life** modules.

## Changing particle properties

The default particle emitter emits crosshair-like shapes. We want to change that to something more interesting. Click on the yellow **Required** box under **Emitters** panel, then under **Material** in the **Details** panel, type `particles`. A list of all the available particle materials will pop up. Choose **m_flare_01** option to create our first particle system, as shown in the following screenshot:

![Changing particle properties](img/00225.jpeg)

Now, let's change the behavior of the particle system. Click on the **Color Over Life** entry under the **Emitters** pane. The **Details** pane at the bottom shows the information about the different parameters, as shown in the following screenshot:

![Changing particle properties](img/00226.jpeg)

In the **Details** pane of **Color Over Life** entry, I increased **X**, but not **Y** and not **Z**. This gives the particle system a reddish glow. (**X** is red, **Y** is green, and **Z** is blue).

Instead of editing the raw numbers, however, you can actually change the particle color more visually. If you click on the greenish zigzag button beside the **Color Over Life** entry, you will see the graph for **Color Over Life** displayed in the **Curve Editor** tab, as shown in the following screenshot:

![Changing particle properties](img/00227.jpeg)

We can now change the **Color Over Life** parameters. The graph in the **Curve Editor** tab displays the emitted color versus the amount of time the particle has been alive. You can adjust the values by dragging the points around. Pressing *Ctrl* + left mouse button adds a new point to a line:

![Changing particle properties](img/00228.jpeg)

Ctrl + click adds points to lines.

You can play around with the particle emitter settings to create your own spell visualizations.

## Settings for the blizzard spell

At this point, we should rename our particle system, from **NewParticle System** to something more descriptive. Let's rename it **P_Blizzard**. You can rename your particle system by simply clicking on it and pressing *F2*.

![Settings for the blizzard spell](img/00229.jpeg)

Press *F2* on an object in the Content Browser to rename it

![Settings for the blizzard spell](img/00230.jpeg)

We will tweak some of the settings to get a blizzard particle effect spell. Perform the following steps:

1.  Under the **Emitters** tab, click on the **Required** box. In the **Details** pane, change the **Emitter** material to **m_flare_01** as shown:![Settings for the blizzard spell](img/00231.jpeg)
2.  Under the **Spawn** module, change the spawn rate to 200\. This increases the density of the visualization, as shown:![Settings for the blizzard spell](img/00232.jpeg)
3.  Under the **Lifetime** module, increase the **Max** property from 1.0 to 2.0\. This introduces some variation to the length of time a particle will live, with some of the emitted particles living longer than others.![Settings for the blizzard spell](img/00233.jpeg)
4.  Under the **Initial Size** module, change the **Min** property size to 12.5 in **X**, **Y**, and **Z**:![Settings for the blizzard spell](img/00234.jpeg)
5.  Under the **Initial Velocity** module, change the **Min**/**Max** values to the values shown:![Settings for the blizzard spell](img/00235.jpeg)
6.  The reason we're having the blizzard blow in +X is because the player's forward direction starts out in +X. Since the spell will come from the player's hands, we want the spell to point in the same direction as the player.
7.  Under the **Color Over Life** menu, change the blue (**Z**) value to 100.0\. You will see an instant change to a blue glow:![Settings for the blizzard spell](img/00236.jpeg)

    Now it's starting to look magical!

8.  Right-click on the blackish area below the **Color Over Life** module. Choose **Location** | **Initial Location**:![Settings for the blizzard spell](img/00237.jpeg)
9.  Enter values under **Start Location** | **Distribution** as shown below:![Settings for the blizzard spell](img/00238.jpeg)
10.  You should have a blizzard that looks like this:![Settings for the blizzard spell](img/00239.jpeg)
11.  Move the camera to a position you like, then click on the **Thumbnail** option in the top menu bar. This will generate a thumbnail icon for your particle system in the **Content Browser** tab.![Settings for the blizzard spell](img/00240.jpeg)

    Clicking Thumbnail at the top menu bar will generate a mini icon for your particle system

# Spell class actor

The `Spell` class will ultimately do damage to all the monsters. Towards that end, we need to contain both a particle system and a bounding box inside the `Spell` class actor. When a `Spell` class is cast by the avatar, the `Spell` object will be instantiated into the level and start `Tick()` functioning. On every `Tick()` of the `Spell` object, any monster contained inside the spell's bounding volume will be affected by that `Spell`.

The `Spell` class should look something like the following code:

[PRE0]

There are only three functions we need to worry about implementing, namely the `ASpell::ASpell()` constructor, the `ASpell::SetCaster()` function, and the `ASpell::Tick()` function.

Open the `Spell.cpp` file. Add a line to include the `Monster.h` file, so we can access the definition of `Monster` objects inside the `Spell.cpp` file, as shown in the following line of code:

[PRE1]

First, the constructor, which sets up the spell and initializes all components is shown in the following code:

[PRE2]

Of particular importance is the last line here, `PrimaryActorTick.bCanEverTick = true`. If you don't set that, your `Spell` objects won't ever have `Tick()` called.

Next, we have the `SetCaster()` method. This is called so that the person who casts the spell is known to the `Spell` object. We can ensure that the caster can't hurt himself with his own spells by using the following code:

[PRE3]

Finally, we have the `ASpell::Tick()` method, which actually deals damage to all contained actors, as shown in the following code:

[PRE4]

The `ASpell::Tick()` function does a number of things, as follows:

*   Gets all actors overlapping `ProxBox`. Any actor that is not the caster gets damaged if the component overlapped is the root component of that object. The reason we have to check for overlapping with the root component is because if we don't, the spell might overlap the monster's `SightSphere`, which means we will get hits from very far away, which we don't want.
*   Notices that if we had another class of thing that should get damaged, we would have to attempt a cast to each object type specifically. Each class type might have a different type of bounding volume that should be collided with, other types might not even have `CapsuleComponent` (they might have `ProxBox` or `ProxSphere`).
*   Increases the amount of time the spell has been alive for. If the spell exceeds the duration it is allotted to be cast for, it is removed from the level.

Now, let's focus on how the player can acquire spells, by creating an individual `PickupItem` for each spell object that the player can pick up.

## Blueprinting our spells

Compile and run your C++ project with the `Spell` class that we just added. We need to create blueprints for each of the spells we want to be able to cast. In the **Class Viewer** tab, start to type `Spell`, and you should see your `Spell` class appear. Right-click on **Spell**, and create a blueprint called **BP_Spell_Blizzard**, and then double-click to open it, as shown in the following screenshot:

![Blueprinting our spells](img/00241.jpeg)

Inside the spell's properties, choose the **P_Blizzard** spell for the particle emitter, as shown in the following screenshot:

![Blueprinting our spells](img/00242.jpeg)

Scroll down until you reach the **Spell** category, and update the **Damage Per Second** and **Duration** parameters to values you like. Here, the blizzard spell will last 3.0 seconds, and do 16.0 damage total per second. After three seconds, the blizzard will disappear.

![Blueprinting our spells](img/00243.jpeg)

After you have configured the **Default** properties, switch over to the **Components** tab to make some further modifications. Click on and change the shape of `ProxBox` so that its shape makes sense. The box should wrap the most intense part of the particle system, but don't get carried away in expanding its size. The `ProxBox` object shouldn't be too big, because then your blizzard spell would affect things that aren't even being touched by the blizzard. As shown in the following screenshot, a couple of outliers are ok.

![Blueprinting our spells](img/00244.jpeg)

Your blizzard spell is now blueprinted and ready to be used by the player.

## Picking up spells

Recall that we previously programmed our inventory to display the number of pickup items the player has when the user presses *I*. We want to do more than that, however.

![Picking up spells](img/00245.jpeg)

Items displayed when the user presses *I*

To allow the player to pick up spells, we'll modify the `PickupItem` class to include a slot for a blueprint of the spell the player casts by using the following code:

[PRE5]

Once you've added the `UClass* Spell` property to the `APickupItem` class, recompile and rerun your C++ project. Now, you can proceed to make blueprints of `PickupItem` instances for your `Spell` objects.

### Creating blueprints for PickupItems that cast spells

Create a **PickupItem** blueprint called **BP_Pickup_Spell_Blizzard**. Double-click on it to edit its properties, as shown in the following screenshot:

![Creating blueprints for PickupItems that cast spells](img/00246.jpeg)

I set the blizzard item's pickup properties as follows:

The name of the item is **Blizzard Spell**, and five are in each package. I took a screenshot of the blizzard particle system and imported it to the project, so the **Icon** is selected as that image. Under spell, I selected **BP_Spell_Blizzard** as the name of the spell to be cast (not **BP_Pickup_Spell_Blizzard**), as shown in the following screenshot:

![Creating blueprints for PickupItems that cast spells](img/00247.jpeg)

I selected a blue sphere for the `Mesh` class of the `PickupItem` class. For **Icon**, I took a screenshot of the blizzard spell in the particle viewer preview, saved it to disk, and imported that image to the project (see the images folder in the **Content Browser** tab of the sample project).

![Creating blueprints for PickupItems that cast spells](img/00248.jpeg)

Place a few of these `PickupItem` in your level. If we pick them up, we will have some blizzard spells in our inventory.

![Creating blueprints for PickupItems that cast spells](img/00249.jpeg)

Left: Blizzard spell pickup items in game world. Right: Blizzard spell pickup item in inventory.

Now we need to activate the blizzard. Since we already attached the left mouse click in [Chapter 10](part0072_split_000.html#24L8G2-dd4a3f777fc247568443d5ffb917736d "Chapter 10. Inventory System and Pickup Items"), *Inventory System and Pickup Items* to dragging the icons around, let's attach the right mouse click to casting the spell.

# Attaching right mouse click to cast spell

The right mouse click will have to go through quite a few function calls before calling the avatar's `CastSpell` method. The call graph would look something like the following screenshot:

![Attaching right mouse click to cast spell](img/00250.jpeg)

A few things happen between right click and spell cast. They are as follows:

*   As we saw before, all user mouse and keyboard interactions are routed through the `Avatar` object. When the `Avatar` object detects a right-click, it will pass the click event to `HUD` through `AAvatar::MouseRightClicked()`.
*   Recall from [Chapter 10](part0072_split_000.html#24L8G2-dd4a3f777fc247568443d5ffb917736d "Chapter 10. Inventory System and Pickup Items"), *Inventory System and Pickup Items* where we used a `struct Widget` class to keep track of the items the player had picked up. `struct Widget` only had three members:

    [PRE6]

    We will need to add an extra property for `struct Widget` class to remember the spell it casts.

    The `HUD` will determine if the click event was inside `Widget` in `AMyHUD::MouseRightClicked()`.

*   If the click was on the `Widget` that casts a spell, the `HUD` then calls the avatar back with the request to cast that spell, by calling `AAvatar::CastSpell()`.

## Writing the avatar's CastSpell function

We will implement the preceding call graph in reverse. We will start by writing the function that actually casts spells in the game, `AAvatar::CastSpell()`, as shown in the following code:

[PRE7]

You might find that actually calling a spell is remarkably simple. There are two basic steps to casting the spell:

*   Instantiate the spell object using the world object's `SpawnActor` function
*   Attach it to the avatar

Once the `Spell` object is instantiated, its `Tick()` function will run each frame when that spell is in the level. On each `Tick()`, the `Spell` object will automatically feel out monsters within the level and damage them. A lot happens with each line of code mentioned previously, so let's discuss each line separately.

### Instantiating the spell – GetWorld()->SpawnActor()

To create the `Spell` object from the blueprint, we need to call the `SpawnActor()` function from the `World` object. The `SpawnActor()` function can take any blueprint and instantiate it within the level. Fortunately, the `Avatar` object (and indeed any `Actor` object) can get a handle to the `World` object at any time by simply calling the `GetWorld()` member function.

The line of code that brings the `Spell` object into the level is as follows:

[PRE8]

There are a couple of things to note about the preceding line of code:

*   `bpSpell` must be the blueprint of a `Spell` object to create. The `<ASpell>` object in angle brackets indicates that expectation.
*   The new `Spell` object starts out at the origin (0, 0, 0), and with no additional rotation applied to it. This is because we will attach the `Spell` object to the `Avatar` object, which will supply translation and direction components for the `Spell` object.

### if(spell)

We always test if the call to `SpawnActor<ASpell>()` succeeds by checking `if( spell )`. If the blueprint passed to the `CastSpell` object is not actually a blueprint based on the `ASpell` class, then the `SpawnActor()` function returns a `NULL` pointer instead of a `Spell` object. If that happens, we print an error message to the screen indicating that something went wrong during spell casting.

### spell->SetCaster(this)

When instantiating, if the spell does succeed, we attach the spell to the `Avatar` object by calling `spell->SetCaster( this )`. Remember, in the context of programming within the `Avatar` class, the `this` method is a reference to the `Avatar` object.

Now, how do we actually connect spell casting from UI inputs, to call `AAvatar::CastSpell()` function in the first place? We need to do some `HUD` programming again.

## Writing AMyHUD::MouseRightClicked()

The spell cast commands will ultimately come from the HUD. We need to write a C++ function that will walk through all the HUD widgets and test to see if a click is on any one of them. If the click is on a `widget` object, then that `widget` object should respond by casting its spell, if it has one assigned.

We have to extend our `Widget` object to have a variable to hold the blueprint of the spell to cast. Add a member to your `struct Widget` object by using the following code:

[PRE9]

Now, recall that our `PickupItem` had the blueprint of the spell it casts attached to it previously. However, when the `PickupItem` class is picked up from the level by the player, then the `PickupItem` class is destroyed.

[PRE10]

So, we need to retain the information of what spell each `PickupItem` casts. We can do that when that `PickupItem` is first picked up.

Inside the `AAvatar` class, add an extra map to remember the blueprint of the spell that an item casts, by item name:

[PRE11]

Now in `AAvatar::Pickup()`, remember the class of spell the `PickupItem` class instantiates with the following line of code:

[PRE12]

Now, in `AAvatar::ToggleInventory()`, we can have the `Widget` object that displays on the screen. Remember what spell it is supposed to cast by looking up the `Spells` map.

Find the line where we create the widget, and just under it, add assignment of the `bpSpell` objects that the `Widget` casts:

[PRE13]

Add the following function to `AMyHUD`, which we will set to run whenever the right mouse button is clicked on the icon:

[PRE14]

This is very similar to our left mouse click function. We simply check the click position against all the widgets. If any `Widget` was hit by the right-click, and that `Widget` has a `Spell` object associated with it, then a spell will be cast by calling the avatar's `CastSpell()` method.

### Activating right mouse button clicks

To connect this HUD function to run, we need to attach an event handler to the mouse right-click. We can do so by going to **Settings** | **Project Settings**, and from the dialog that pops up, adding an **Input** option for **Right Mouse Button**, as shown in the following screenshot:

![Activating right mouse button clicks](img/00251.jpeg)

Declare a function in `Avatar.h`/`Avatar.cpp` called `MouseRightClicked()` with the following code:

[PRE15]

Then, in `AAvatar::SetupPlayerInputComponent()`, we should attach `MouseClickedRMB` event to that `MouseRightClicked()` function:

[PRE16]

We have finally hooked up spell casting. Try it out, the gameplay is pretty cool, as shown in the following screenshot:

![Activating right mouse button clicks](img/00252.jpeg)

# Creating other spells

By playing around with particle systems, you can create a variety of different spells that do different effects.

## The fire spell

You can easily create a fire variant of our blizzard spell by changing the color of the particle system to red:

![The fire spell](img/00253.jpeg)

The out val of the color changed to red

## Exercises

Try the following exercises:

1.  **Lightning spell**: Create a lightning spell by using the beam particle. Follow Zak's tutorial for an example of how beams are created and shot in a direction, at [https://www.youtube.com/watch?v=ywd3lFOuMV8&list=PLZlv_N0_O1gYDLyB3LVfjYIcbBe8NqR8t&index=7](https://www.youtube.com/watch?v=ywd3lFOuMV8&list=PLZlv_N0_O1gYDLyB3LVfjYIcbBe8NqR8t&index=7).
2.  **Forcefield spell**: A forcefield will deflect attacks. It is essential for any player. Suggested implementation: Derive a subclass of `ASpell` called `ASpellForceField`. Add a bounding sphere to the class, and use that in the `ASpellForceField::Tick()` function to push the monsters out.

What's next? I would highly recommend that you expand on our little game here. Here are some ideas for expansion:

*   Create more environments, expand the terrain, add in more houses and buildings
*   Add quests that come from NPCs
*   Define more melee weapons such as, swords
*   Define armor for the player, such as shields
*   Add shops that sell weapons to the player
*   Add more monster types
*   Implement loot drops for monsters

You have literally thousands of hours of work ahead of you. If you happen to be a solo programmer, form working relationships with other souls. You cannot survive in the game marketplace on your own.

It's dangerous to go alone—Take a friend.

# Summary

This concludes this chapter. You have come a long way. From not knowing anything about C++ programming at all, to hopefully being able to string together a basic game program in UE4.