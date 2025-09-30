# Chapter 9. Particles

Particles in Unreal Engine 4 are created using cascade particle editor, which is a powerful and robust editor that allows artists to create visual effects. Cascade editor lets you add and edit various modules that make up the final effect. The primary job of the particle editor is to control the behavior of the particle system itself whereas the look and feel is often controlled by the material.

In this chapter you will learn about the cascade particle editor and create a simple particle system.

# Cascade particle editor

To access cascade particle editor, you need to create a **Particle System** in **Content Browser** by right-clicking on the **Content Browser** and selecting **Particle System**. When you select it, a new **Particle System** will be created and it prompts you to rename it. Give it a name and double-click on it to open cascade particle editor.

Once you open it you will see a window like this:

![Cascade particle editor](img/B03950_09_01.jpg)

Cascade Editor User Interface

Cascade particle editor consists of five primary areas and they are:

*   **Toolbar**: This contains visualization and navigation tools
*   **Viewport**: This shows the current particle system
*   **Details**: This lets you edit the current particle system, emitter, or modules
*   **Emitters**: These are the actual particle emitters and contain modules that are associated with the emitter
*   **Curve Editor**: This is the editor that lets you modify properties in either relative or absolute time

## Toolbar

Toolbar contains various buttons. Let's take a quick look at them:

*   **Save**: This saves the particle system
*   **Find in CB**: This locates the current particle system in Content Browser
*   **Restart Sim**: This restarts (resets) the current simulation
*   **Restart Level**: This is the same as Restart Sim but will also update all the instances placed in level
*   **Thumbnail**: This saves the viewport view as a thumbnail for Content Browser
*   **Bounds**: This enables or disables rendering of particle bounds
*   **Origin Axis**: This displays the origin axis in viewport
*   **Regen LOD**: Clicking on this generates the lowest LOD duplicating the highest LOD
*   **Regen LOD**: Clicking on this generates the lowest LOD using values based on the highest LOD
*   **Lowest LOD**: This switches to the lowest LOD
*   **Lower LOD**: This switches to the next lowest LOD
*   **Add LOD**: This adds a new LOD before the current LOD
*   **Add LOD**: This adds a new LOD after the current LOD
*   **Higher LOD**: This selects a higher LOD
*   **Highest LOD**: This selects the highest LOD
*   **Delete LOD**: This deletes the current LOD

LODs are ways to update the particle effects to use efficient screen space depending on player distance. Based on the effect, there can be modules in a particle system that can be too small to render if the player is far away. Imagine fire embers. If the player is far away, the particle system will still process and calculate these effects which we don't need. This is where we use LODs. **Level of Detail** (**LODs**) can turn off specific modules or even shut down the emitter based on player distance.

## Viewport

Viewport shows you the real-time changes made to the particle system as well as other information's, such as total particle count, bounds, and so on. On the top left corner, you can click on the **View** button to switch between various view modes, such as **Unlit**, **Texture Density**, **Wireframe mode**, and so on.

### Navigation

Using the following mouse buttons you can navigate inside the viewport:

*   **Left Mouse Button**: This moves the camera around the particle system.
*   **Middle Mouse Button**: This pans the camera.
*   **Right Mouse Button**: This rotates the camera.
*   **Alt + Left Mouse Button**: This orbits the particle system.
*   **Alt + Right Mouse Button**: This dollies the camera forward and backward from a particle system.
*   **F**: This focus on the particle system.
*   **L + Left Mouse**: This rotates the light and only affects particles using **Lit** material. **Unlit** materials have no effect.

Inside the **Viewport**, you can play/pause the particle simulation as well as adjust the simulation speed. You can access these settings under the **Time** option in **Viewport**.

![Navigation](img/B03950_09_02.jpg)

## Details

**The Details** panel is populated by the currently selected module or emitter. The main properties of the particle system can be accessed by selecting nothing in the **Emitters** panel or by right-clicking on the **Emitter** list and navigating to **Particle System** | **Select Particle System**.

![Details](img/B03950_09_03.jpg)

## Emitter

The **Emitter** panel is the heart of the particle system, and contains a horizontal arrangement of all the emitters. In each emitter column, you can add different modules to change the look and feel of the particles. You can add as many emitters as you want and each emitter will handle different aspects of the final effect.

![Emitter](img/B03950_09_04.jpg)

An **Emitter** contains three primary areas, and they are as follows:

*   On top of the emitter block are the primary properties of the emitter, such as name, type, and so on. You can double-click on the gray area to collapse or expand the emitter column.
*   Below that, you can define the type of emitter. If you leave it blank (as in the preceding screenshot), then particles are simulated on the CPU.
*   Finally, you can add modules to define how particles look.

### Emitter types

Cascade editor has four different emitter types, and they are as follows:

*   **Beam Type**: When using this type, the particle will output beams connecting two points. This means you have to define a source point (for example, the emitter itself) and a target point (for example, an actor).
*   **GPU Sprite**: Using this type lets you simulate particles on the GPU. Using this emitter lets you simulate and render thousands of particles efficiently.
*   **Mesh Type**: When using this, the particle will use actual **Static Mesh** instances for particles. This is pretty useful for simulating destruction effects (for example, debris).
*   **Ribbon**: This type indicates that the particle should be like a trail. This means, all particles (in order of their birth) are connected to each other to form ribbons.

## Curve editor

This is the standard curve editor that lets the user adjust any values that need to change during the particle's lifetime or across the life of an emitter. To learn more about curve editor, you can visit the official documentation available at [https://docs.unrealengine.com/latest/INT/Engine/UI/CurveEditor/index.html](https://docs.unrealengine.com/latest/INT/Engine/UI/CurveEditor/index.html).

# Creating a simple particle system

To create a particle system:

1.  Right-click on **Content Browser**.
2.  Select **Particle** from the resulting context menu.
3.  A new particle system asset will be created in **Content Browser** and prompts you to rename it.
4.  For this example, let's call it **MyExampleParticleSystem**.
5.  Now, double-click on it to open the **Particle** editor.

By default, Unreal creates a default emitter for you to work with. This emitter contains six modules, and they are:

*   **Required**: This contains all the properties required by the emitter, such as the material used to render, how long the emitter should run before looping, can this emitter loop, and so on. You cannot delete this module.
*   **Spawn**: This module contains the properties that determine how the particles are spawned. For example, how many particles to spawn per second. You cannot delete this module.
*   **Lifetime**: This is the lifetime of the spawned particles.
*   **Initial Size**: This sets the initial size of particles when spawning. To modify the size after spawning, use **Size by Life** or **Size by Speed**.
*   **Initial Velocity**: This sets the initial velocity (speed) of particles when spawning. To modify the velocity after spawning, use **Velocity/Life**.
*   **Color over Life**: This sets the color of a particle over its lifetime.

For this example, we will modify the existing emitter and make it a GPU particle system that looks like sparks. We will also add collisions so that our particles collide with the world.

## Creating a simple material

Before we start working with particles, we need to create a simple material that we can apply to the particles. To create a new material:

1.  Right-click on **Content Browser** and select **Material**. Feel free to name it anything.
2.  Open **Material** editor and change **Blend Mode** to **Translucent**. This is required because GPU particle collision will not work on opaque materials.
3.  Then, change **Shading Model** to **Unlit**. This is because we don't want the sparks to be affected by any kind of light since they are emissive.
4.  Finally, create a graph like this:![Creating a simple material](img/B03950_09_05.jpg)

### Note

Note that the circular gradient texture in the **Texture Sample** node comes with the Engine itself. It's called **Greyscale**.

Now that we have our material, it's time to customize our particle system:

1.  Select the **Required** module and under the **Emitters** group, apply our material created in the previous step.
2.  Right-click on the black area below the emitter and select **New GPU Sprites** under **Type Data**. This will make our emitter simulate particles on GPU.![Creating a simple material](img/B03950_09_06.jpg)
3.  Select the **Spawn** module and under the **Spawn** group, set **Rate** to **0**. This is because instead of spawning a certain amount of particles per second, we want to burst hundreds of them in one frame.
4.  Under the **Burst** group, add a new entry in **Burst List** and set **Count** to **100** and **Count Low** to **10**. This will select a random value between **100** and **10** and will spawn that many particles.

    The final **Spawn** settings will look like this:

    ![Creating a simple material](img/B03950_09_07.jpg)
5.  After adjusting the **Spawn** settings, we set the **Lifetime** of the particles to **0.4** and **3.0**, so each spawned particles' lifetime is between **0.4** and **3.0**. Now that we have particles spawning, it's time to adjust their size. To do so, select the **Initial Size** module and set **Max** to **1.0**, **10.0**, **0.0** and **Min** to **0.5**, **8.0**, **0.0**.![Creating a simple material](img/B03950_09_08.jpg)

    ### Note

    Note that since GPU sprites are 2D, you can ignore the **Z** value. That's why we set them to **0.0**.

6.  After that, select the **Initial Velocity** module and set **Max** to **100.0**, **200.0**, **200.0** and **Min** to **-100.0**, **-10.0**, **100.0**.
7.  Now, if you drag and drop this particle into the world, you will see the particles bursting into the air.

    ### Note

    Note that if you see nothing happening, make sure **Real-Time** is turned on for the editor (*Ctrl*+*R*).

![Creating a simple material](img/B03950_09_09.jpg)

## Adding gravity

In order to make things a bit more real, we will simulate gravity on these particles. Go back to your particle editor and follow these steps:

1.  Right-click on the module area.
2.  Select **Const Acceleration** from the **Acceleration** menu. This module will add the given acceleration to the existing acceleration of particles and updates the current and base velocity.![Adding gravity](img/B03950_09_10.jpg)
3.  For the **Acceleration** value, use **0.0**, **0.0**, **-450.0**. A negative value of **Z** (that is, **-450**) will make the particles go down as if they are affected by gravity.

### Note

Note that the default gravity value is **-980.0**. You can try this value as well.

Now, if you look at the particle in world, you can see them going down as if they are affected by gravity.

![Adding gravity](img/B03950_09_11.jpg)

## Applying the color over life module

Now that we have something like sparks, let's apply some color to it. Select the Color Over Life module and apply the settings shown here:

![Applying the color over life module](img/B03950_09_12.jpg)

**Color Over Life** is a curve value. It means you can define what color to apply at a certain point in the lifetime of particle. The **0.0** value is the beginning and **1.0** is the end. In the preceding screenshot, you can see I have applied a bright reddish orange color (**50.0**, **20.0**, **8.0**) when the particle is spawning (**In Val** = **0.0**) and bright white color at the end (**In Val** = **1.0**).

## Adding collision module

To complete this effect, we will add a **Collision** module so that our particles will collide with the world. To add the **Collision** module, go through the following steps:

1.  Right-click on the modules area and select **Collision** from the **Collision** menu.
2.  Select the **Collision** module.
3.  Set the **Resilience** value to **0.25**. This will make the collided particles less bouncy. Higher resilience means more bouncy particles.
4.  Set **Friction** to **0.2**. This will make the particles stick to the ground. A higher friction value (**1.0**) will not let the particle move after colliding, whereas lower values make the particle slide along the surface.

Now, if you simulate or play the game with this particle in the world, you can see it bursting and colliding with the world but it's very unrealistic. You can easily notice that every second this particle keeps repeating. So to prevent this, follow these steps:

1.  Open the particle editor.
2.  Select the **Required** module.
3.  Under the **Duration** settings, set **Emitter Loops** to **1**. By default, this is set to **0**, which means it will loop forever.![Adding collision module](img/B03950_09_13.jpg)

# Playing particle in Blueprints

Now that our particle effect is ready, let's play it using Blueprints:

1.  Right-click on **Content Browser**.
2.  Select the **Blueprint** class.
3.  From the resulting window, select **Actor**.
4.  Double-click on the **Blueprint** to open the editor.
5.  Select your bursting particles in **Content Browser**.
6.  Open the **Blueprint** editor and add a new **Particle System Component** (if you select the particle in **Content Browser**, it will automatically set that particle as the template for the **Particle System Component**).
7.  Go to the **Event Graph** tab.
8.  Right-click anywhere on the graph and select **Add Custom Event…** from the **Add Event** category.![Playing particle in Blueprints](img/B03950_09_14.jpg)
9.  Rename that **Custom Event** with any name you like. For this example, I renamed it **ActivateParticle**.
10.  Create a graph like this:![Playing particle in Blueprints](img/B03950_09_15.jpg)

This Blueprint will first execute **ActivateParticle** when the game begins and when the event is executed, it randomly selects a time (in seconds) between **0.2** and **2**. When the time runs out, it activates the particle and calls this event again.

Now, if you drag and drop this particle into the world and start playing, you will see the particles randomly bursting:

![Playing particle in Blueprints](img/B03950_09_16.jpg)

# Summary

From here, you can extend this particle and add some lights to make it look even more real. Note that the **Light** module cannot be used with GPU particles so you need to create another emitter and add a light module there. Since you learned about the GPU particle data type, you can add more and more emitters that use other data types, such as beam type, mesh type, ribbon type, and so on. From what you learned in this chapter and other chapters, you can create a Blueprint that includes a light mesh that emits this spark particle effect when it receives damage.

In the next chapter, we will dive into the world of C++.