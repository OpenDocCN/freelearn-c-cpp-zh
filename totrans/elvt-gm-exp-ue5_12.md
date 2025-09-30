# 12

# Animation Blending and Montages

In the previous chapter, you were able to bring the player character to life by implementing movement animations in a Blend Space and using that Blend Space in an Animation Blueprint to drive the animations based on the player’s speed. You were then able to implement functionality in C++ based on player input to allow the character to sprint. Lastly, you took advantage of the Animation State Machine built-in Animation Blueprints to drive the character’s movement state and jumping states to allow fluid transitions between walking and jumping.

With the player character’s Animation Blueprint and State Machine working, it’s time to introduce Animation Montages and Anim Slots by implementing the character’s `Throw` animation. In this chapter, you will learn more about animation blending, see how Unreal Engine handles the blending of multiple animations by creating an Animation Montage, and work with a new Anim Slot for the player’s throwing animation. From there, you will use the Anim Slot in the player’s Animation Blueprint by implementing new functions such as `Save Cached Pose` and `Layered blend per bone` so that the player can correctly blend the movement animations you handled in the previous chapter with the new throwing animation you will implement in this chapter.

In this chapter, you’ll learn about the following topics:

*   How to use Anim Slots to create layered animation blending for the player character
*   Creating an Animation Montage for the character’s `Throw` animation
*   Using the `Layered blend per bone node` within the Animation Blueprint to blend together the upper body `Throw` animation and the lower body movement animations of the character

By the end of this chapter, you will be able to use the `Animation Montage` tool to create a unique throwing animation using the `Throw` animation sequence you imported in [*Chapter 10*](B18531_10.xhtml#_idTextAnchor199), *Creating the SuperSideScroller Game*. With this montage, you will create and use Anim Slots that will allow you to blend animations in the Animation Blueprint for the player character. You will also get to know how to use blending nodes to effectively blend the movement and throwing animations of the character.

After finalizing the player character animation, you will create the required class and assets for the enemy AI and learn more about Materials and Material Instances, which will give this enemy a unique visual color so that it can be differentiated in-game. Finally, the enemy will be ready for [*Chapter 13*](B18531_13.xhtml#_idTextAnchor268), *Creating and Adding the Enemy Artificial Intelligence*, where you will begin to create the AI behavior logic.

# Technical requirements

For this chapter, you will need Unreal Engine 5 installed

Let’s start by learning about what Animation Montages and Anim Slots are and how they can be used for character animation.

The project for this chapter can be found in the Chapter12 folder of the code bundle for this book, which can be downloaded here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition).

# Animation blending, Anim Slots, and Animation Montages

Animation blending is the process of transitioning between multiple animations on a skeletal mesh as seamlessly as possible. You are already familiar with the techniques of animation blending because you created a `Blend` `Spaces` asset for the player character in [*Chapter 11*](B18531_11.xhtml#_idTextAnchor222), *Working with Blend Space 1D, Key Bindings, and State Machines*. In this Blend Space, the character smoothly blends between the `Idle`, `Walking`, and `Running` animations. You will now extend this knowledge by exploring and implementing new additive techniques to combine the movement animations of the character with a throwing animation. Through the use of an Anim Slot, you will send the throwing animation to a set of upper body bones, and its children’s bones, to allow movement and throwing animations to apply at the same time without negatively impacting the other. But first, let’s talk more about Animation Montages.

Animation Montages are very powerful assets that allow you to combine multiple animations and split these combined animations into what are called **Sections**. Sections can then be played back individually, in a specific sequence, or even looped.

Animation Montages are also useful because you can control animations through montages from Blueprints or C++; this means you can call logic, update variables, replicate data, and so on, based on the animation section being played, or if any Notifies are called within the montage. In C++, there is the `UAnimInstance` object, which you can use to call functions such as `UAnimInstance::Montage_Play`, which allows you to access and play montages from C++.

Note

This method will be used in [*Chapter 14*](B18531_14.xhtml#_idTextAnchor298), *Spawning the Player Projectile*, when you will begin to add polish to the game. More information about how animations and Notifies are handled by Unreal Engine 5 in C++ can be found at [https://docs.unrealengine.com/en-US/API/Runtime/Engine/Animation/AnimNotifies/UAnimNotifyState/index.xhtml](https://docs.unrealengine.com/en-US/API/Runtime/Engine/Animation/AnimNotifies/UAnimNotifyState/index.xhtml). You will learn more about Notifies in the first exercise of this chapter, and you will code your own notify state in [*Chapter 14*](B18531_14.xhtml#_idTextAnchor298), *Spawning the Player Projectile*.

The following figure shows the Persona editor for Animation Montages. However, this will be broken down even further in *Exercise 12.01*, *Setting up the Animation Montage*:

![Figure 12.1 – The Persona editor, which opens when editing an Animation Montage ](img/Figure_12.01_B18531.jpg)

Figure 12.1 – The Persona editor, which opens when editing an Animation Montage

Just like in Animation Sequences, Animation Montages allow Notifies to be triggered along the timeline of a section of an animation, which can then trigger sounds, particle effects, and events. Event Notifies will allow us to call logic from Blueprint or C++. Epic Games provides an example in their documentation of a weapon reload Animation Montage that is split between animations for `reload start`, `reload loop`, and `reload complete`. By splitting these animations and applying Notifies for sounds and events, developers have complete control over how long the `reload loop` animation will play based on internal variables, and control over any additional sounds or effects to play during the course of the animation.

Lastly, Animation Montages support **Anim Slots**. Anim Slots allow you to categorize an animation, or a set of animations, that can later be referenced in Animation Blueprints to allow unique blending behavior based on the slot. This means that you can define an Anim Slot that can later be used in Animation Blueprints to allow animations using this slot to blend on top of the base movement animations in any way you want; in our case, only affecting the upper body of the player character and not the lower body.

Let’s begin by creating the Animation Montage for the player character’s `Throw` animation in the first exercise.

## Exercise 12.01 – Setting up the Animation Montage

One of the first things you need to do for the player character is to set up the Anim Slot that will separately categorize this animation as an upper-body animation. You will use this Anim Slot in conjunction with blending functions in the Animation Blueprint to allow the player character to throw a projectile, while still correctly animating the lower body while moving and jumping.

By the end of this exercise, the player character will be able to play the `Throw` animation only with their upper body, while their lower body will still use the movement animation that you defined in the previous chapter.

Let’s begin by creating the Animation Montage for the character, throwing and setting up the Anim Slot there:

1.  First, navigate to the `/MainCharacter/Animation` directory, which is where all of the animation assets are located.
2.  Now, right-click in the content drawer and hover over the **Animation** option from the available drop-down menu.
3.  Then, left-click to select the **Animation Montage** option from the additional drop-down menu that appears.
4.  Just as with creating other animation-based assets, such as `Blend Spaces` or `Animation Blueprints`, Unreal Engine will ask you to assign a `Skeleton` object for this Animation Montage. In this case, select `MainCharacter_Skeleton`.
5.  Name the new Animation Montage `AM_Throw`. Now, double-left-click to open the montage.

When you open the `Animation Montage` asset, you are presented with a similar editor layout as you would when opening an Animation Sequence. There is a **Preview** window that shows the main character skeleton in the default T pose, but once you add animations to this montage, the skeleton will update to reflect those changes.

With this exercise complete, you have successfully created an `Throw` animation and Anim Slot you need in order to blend the `Throw` animation with the existing character movement animations.

# Animation Montages

Have a look at the following figure:

![Figure 12.2 – The Preview window alongside the Montage and Sections areas ](img/Figure_12.02_B18531.jpg)

Figure 12.2 – The Preview window alongside the Montage and Sections areas

Underneath the **Preview** window, you have the main montage timeline, in addition to other sections. Let’s evaluate these sections from top to bottom:

*   **Montage**: The **Montage** section is a collection of animations that can have one or more animations. You can also right-click on any point in the timeline to create a section.
*   **Montage Sections**: Sections allow you to compartmentalize the different parts of the montage into their own self-contained section, which allows you to set the order of how the individual animation sequences are played and whether a section should loop.

For the purposes of the `Throw` montage, you do not need to use this feature since you will only be using one animation in this montage:

*   **Timing**: The **Timing** section gives you a preview of the montage and the sequential order of the varied aspects of the montage. The playback order of **Notifies**, the **Montage** section, and other elements will be visually displayed here to give you a quick preview of how the montage will work.
*   `Play Sound` or `Play Particle Effect`, allow you to play a sound or particle at a specific time in the animation. You will use these Notifies later on in this project when you implement the throwing projectile:

![Figure 12.3 – The Timing and Notifies areas ](img/Figure_12.03_B18531.jpg)

Figure 12.3 – The Timing and Notifies areas

Now that you are familiar with the interface for Animation Montages, you can add the `Throw` animation to the montage by following the next exercise.

## Exercise 12.02 – Adding the Throw animation to the montage

Now that you have a better understanding of what Animation Montages are and how these assets work, it is time to add the `Throw` animation to the montage you created in *Exercise 12.01*, *Setting up the Animation Montage*. Although you will only be adding one animation to this montage, it is important to emphasize that you can add multiple unique animations to a montage that you can then play back. Now, let’s start by adding the `Throw` animation you imported into the project in [*Chapter 10*](B18531_10.xhtml#_idTextAnchor199), *Creating the SuperSideScroller Game*:

1.  In `Throw` animation asset. Then, left-click and drag it onto the timeline in the **Montage** section:

![Figure 12.4 – The Asset Browser window with animation-based assets ](img/Figure_12.04_B18531.jpg)

Figure 12.4 – The Asset Browser window with animation-based assets

Once an animation is added to the Animation Montage, the character skeleton in the **Preview** window will update to reflect this change and begin playing the animation:

![Figure 12.5 – The player character begins to animate ](img/Figure_12.05_B18531.jpg)

Figure 12.5 – The player character begins to animate

Now that the `Throw` animation has been added to the Animation Montage, you can move on to create the Anim Slot.

The **Anim Slot Manager** tab should be docked next to the **Asset Browser** tab on the right-hand side. If you don’t see the **Anim Slot Manager** tab, you can access it by navigating to the **Window** tab in the toolbar at the top of the **Animation Montage** editor window. There, left-click to select the option for **Anim Slot Manager**, and the window will appear.

By completing this exercise, you have added the `Throw` animation to your new Animation Montage and you were able to play back the animation to preview how it looks in the editor through **Persona**.

Now, you can move on to learn more about Anim Slots and Anim Slot Manager before adding your own unique Anim Slot to use for animation blending later in this chapter.

# Anim Slot Manager

`Face` to articulate to others that the slots within this group affect the face of the character. By default, Unreal Engine provides you with a group called `DefaultGroup` and an Anim Slot called `DefaultSlot`, which is in that group.

In the following exercise, we will create a new Anim Slot specifically for the upper body of the player character.

## Exercise 12.03 – Adding a new Anim Slot

Now that you have a better understanding of Anim Slots and `Upper Body`. Once you have this new slot created, it can then be used and referenced in your Animation Blueprint to handle animation blending, which you will do in a later exercise.

Let’s create the Anim Slot by doing the following:

1.  In **Anim Slot Manager**, left-click on the **Add Slot** option.
2.  When adding a new slot, Unreal will ask you to give this Anim Slot a name. Name this slot `Upper Body`. Anim Slot naming is important, much like naming any other assets and parameters, as you will be referencing this slot in the Animation Blueprint later.

With the Anim Slot created, you can now update the slot used for the Throw montage.

1.  In the `DefaultGroup.DefaultSlot`. Left-click, and from the drop-down menu, select `DefaultGroup.Upper Body`:

![Figure 12.6 – The new Anim Slot will appear in the drop-down list ](img/Figure_12.06_B18531.jpg)

Figure 12.6 – The new Anim Slot will appear in the drop-down list

Note

After changing the Anim Slot, you may notice that the player character stops animating and returns to the T pose. Don’t worry – if this happens, just close the Animation Montage and reopen it. Once reopened, the character will play the `Throw` animation again.

With your Anim Slot created and in place in the `Throw` montage, it is now time for you to update the Animation Blueprint so that the player character is aware of this slot and animates correctly based on it.

With this exercise complete, you have created your first Anim Slot using Anim Slot Manager, available in the Animation Montage. With this slot in place, it can now be used and referenced in the player character Animation Blueprint to handle the animation blending required to blend the `Throw` animation and the movement animations you implemented in the previous chapter. Before you do this, you need to learn more about the `Save Cached Pose` node in Animation Blueprints.

# Save Cached Pose

There are cases when working with complex animations and characters requires you to reference a pose that is outputted by a State Machine in more than one place. If you hadn’t noticed already, the output pose from your `Movement` State Machine cannot be connected to more than one other node. This is where the `Save Cached Pose` node comes in handy; it allows you to cache (or store) a pose that can then be referenced in multiple places at once. You will need to use this to set up the new Anim Slot for the upper body animation.

In the next exercise, you will implement the `Save Cached Pose` node to cache the `Movement` State Machine.

## Exercise 12.04 – Save Cached Pose of the Movement State Machine

To effectively blend the `Throw` animation, which uses the `Upper Body` Anim Slot you created in the previous exercise with the movement animations already in place for the player character, you need to be able to reference the `Movement` State Machine in the Animation Blueprint. To do this, do the following to implement the `Save Cached Pose` node in the Animation Blueprint:

1.  In the Anim Graph of `AnimBP_SuperSideScroller_MainCharacter`, right-click and search for `New Save Cached Pose`. Name this `Movement Cache`:

![Figure 12.7 – The pose will be evaluated once per frame and then cached ](img/Figure_12.07_B18531.jpg)

Figure 12.7 – The pose will be evaluated once per frame and then cached

1.  Now, instead of connecting your `Movement` state machine directly to the output pose, connect it to the cache node:

![Figure 12.8 – The Movement State Machine is being cached ](img/Figure_12.08_B18531.jpg)

Figure 12.8 – The Movement State Machine is being cached

1.  With the `Movement` State Machine pose being cached, all you have to do now is reference it. This can be done by searching for the `Use Cached Pose` node.

Note

All cached poses will show in the context-sensitive menu. Just make sure you select the cached pose with the name you gave it in *step 1*.

1.  With the cached pose node available, connect it to `Output Pose` of the Anim Graph:

![Figure 12.9 – The cached pose is now feeding directly to Output Pose ](img/Figure_12.09_B18531.jpg)

Figure 12.9 – The cached pose is now feeding directly to Output Pose

You will notice now, after *step 4*, that the main character will animate correctly and move as you expect after the last chapter. This proves that the caching of the `Movement` State Machine is working. The following figure shows the player character back in his `Idle` animation in the **Preview** window of the Animation Blueprint:

![Figure 12.10 – The main character is animating as expected ](img/Figure_12.10_B18531.jpg)

Figure 12.10 – The main character is animating as expected

Now that you have the caching of the `Movement` State Machine working, you will use this cache to blend animations through the skeleton based on the Anim Slot you created:

With this exercise complete, you now have the ability to reference the cached `Movement` State Machine pose anywhere you would like within the Animation Blueprint. With this accessibility in place, you can now use the cached pose to begin the blending between the cached movement pose and the `Upper Body` Anim Slot using a function called `Layered blend per bone`.

# Layered blend per bone

The node that you will use to blend animations here is called `Layered blend per bone`. This node masks out a set of bones on the character’s skeleton for an animation to ignore those bones.

In the case of our player character and the `Throw` animation, you will mask out the lower body so that only the upper body animates. The goal is to be able to perform the `Throw` and movement animations at the same time and have these animations blend together; otherwise, when you perform the throw, the movement animations would completely break.

In the following exercise, you will use `Layered blend per bone` to mask out the lower half of the player character so that the `Throw` animation only affects the upper body of the character.

## Exercise 12.05 – Blending animation with the Upper Body Anim Slot

The `Layered blend per bone` function allows us to blend the `Throw` animation with the movement animations you implemented in the previous chapter, and give you control over how much influence the `Throw` animation will have on the player character’s skeleton.

In this exercise, you will use the `Layered blend per bone` function to completely mask out the lower body of the character when playing the `Throw` animation so that it does not influence the character movement animation of the lower body.

Let’s begin by adding the `Layered blend per bone` node and discussing its input parameters and its settings:

1.  Inside the Animation Blueprint, right-click and search for `Layered blend per bone` in the `Layered blend per bone` node and its parameters:
    *   The first parameter, `Base Pose`, is for the base pose of the character; in this case, the cached pose of the `Movement` State Machine will be the base pose.
    *   The second parameter is the `Blend Poses 0` node that you want to layer on top of `Base Pose`; keep in mind that selecting `Blend Poses` and `Blend Weights` parameters. For now, you will only be working with one `Blend Poses` node.
    *   The last parameter is `Blend Weights`, which is how much `Blend Poses` will affect `Base Pose` on a scale from `0.0` to `1.0` as an alpha:

![Figure 12.11 – The Layered blend per bone node ](img/Figure_12.11_B18531.jpg)

Figure 12.11 – The Layered blend per bone node

Before you connect anything to this node, you will need to add a layer to its properties.

1.  Left-click to select the node and navigate to `0`, of this setup. Left-click on **+** next to **Branch Filters** to create a new filter.

There are again two parameters here, namely the following:

*   `Bone Name`: The bone to specify where the blending will take place and determine the child hierarchy of bones masked out. In the case of the main character skeleton for this project, set `Bone Name` to `Spine`. *Figure 12.12* shows how the `Spine` bone and its children are unassociated with the lower body of the main character. This can be seen in the `Skeleton` asset, `MainCharacter_Skeleton`:

![Figure 12.12 – The Spine bone and its children are associated with the upper body of the main character ](img/Figure_12.12_B18531.jpg)

Figure 12.12 – The Spine bone and its children are associated with the upper body of the main character

*   `Blend Depth`: The depth in which bones and their children will be affected by the animation. A value of `0` will not affect the rooted children of the selected bone.
*   `Mesh Space Rotation Blend`: Determines whether or not to blend bone rotations in mesh space or local space. Mesh space rotation refers to the skeletal mesh’s bounding box as its base rotation, while local space rotation refers to the local rotation of the bone name in question. In this case, we want the rotation blend to occur in mesh space, so we will set this parameter to `true`.

Blending is propagated to all the children of a bone to stop blending on particular bones, add them to the array, and make their blend depth value `0`. The final result is as follows:

![Figure 12.13 – You can set up multiple layers with one blend node ](img/Figure_12.13_B18531.jpg)

Figure 12.13 – You can set up multiple layers with one blend node

1.  With the settings in place on the `Layered blend per bone` node, you can connect the `Movement Cache` cached pose into the `Base Pose` node of the layered blend. Make sure you connect the output of the `Layered blend per bone` node to `Output Pose` of the Animation Blueprint:

![Figure 12.14 – Add the cached pose for the Movement State Machine to the Layered blend per bone node ](img/Figure_12.14_B18531.jpg)

Figure 12.14 – Add the cached pose for the Movement State Machine to the Layered blend per bone node

Now, it’s time to use the Anim Slot you created earlier to filter only the animations using this slot through the `Layered blend per bone` node.

1.  Right-click in the Anim Graph and search for `DefaultSlot`. Left-click to select the `Slot` node and navigate to `Slot Name` property. Left-click on this dropdown to find and select the `DefaultGroup.Upper Body` slot.

When changing the `Slot Name` property, the `Slot` node will update to represent this new name. The `Slot` node requires a source pose, which will again be a reference to the `Movement` State Machine. This means that you need to create another `Use Cached Pose` node for the `Movement Cache` pose.

1.  Connect the cached pose to the source of the `Slot` node:

![Figure 12.15 – Filtering the cached movement pose through the Anim Slot ](img/Figure_12.15_B18531.jpg)

Figure 12.15 – Filtering the cached movement pose through the Anim Slot

1.  All that is left to do now is to connect the `Upper Body` slot node to the `Blend Poses 0` input. Then, connect the final pose of the `Layered blend per bone` to the result of the `Output Pose` Animation Blueprint:

![Figure 12.16 – The final setup of the main character’s Animation Blueprint ](img/Figure_12.16_B18531.jpg)

Figure 12.16 – The final setup of the main character’s Animation Blueprint

With the Anim Slot and the `Layered blend per bone` node in place within the main character’s Animation Blueprint, you are finally done with the animation side of the main character.

With the Animation Blueprint updated, we can now move on to the next exercise, where we can finally preview the `Throw` animation in action.

## Exercise 12.06 – Previewing the Throw animation

In the previous exercise, you did a lot of work to allow animation blending between the player character’s `Movement` animations and the `Throw` animation by using the `Save Cached Pose` and `Layered blend per bone` nodes. Perform the following steps to preview the `Throw` animation in-game and see the fruits of your labor:

1.  Navigate to the `/MainCharacter/Blueprints/` directory and open the character’s `BP_SuperSideScroller_MainCharacter` Blueprint.
2.  If you recall, in the last chapter, you created `Enhanced Input Action` for throwing with `IA_Throw` .
3.  Inside `Event Graph` of the character’s Blueprint, right-click and search for `EnhancedInputAction IA_Throw` in the **Context Sensitive** drop-down search. Select it with a left-click to create the event node in the graph.

With this event in place, you need a function that allows you to play an Animation Montage when the player uses the left mouse button to throw.

1.  Right-click in `Event Graph` and search for `Play Montage`. Make sure not to confuse this with a similar function, `Play Anim Montage`.

The `Play Montage` function requires two important inputs:

*   `Montage to Play`
*   `In Skeletal Mesh Component`

Let’s first handle the `Skeletal Mesh` component.

1.  The player character has a `Skeletal Mesh` component that can be found in the `Mesh`. Left-click and drag out a `Get` reference to this variable and connect it to the `In Skeletal Mesh Component` input of this function:

![Figure 12.17 – The mesh of the player character connected to the In Skeletal Mesh Component input ](img/Figure_12.17_B18531.jpg)

Figure 12.17 – The mesh of the player character connected to the In Skeletal Mesh Component input

The last thing to do now is to tell this function which montage to play. Luckily for you, there is only one montage that exists in this project: `AM_Throw`.

1.  Left-click on the drop-down menu under the `Montage to Play` input and left-click to select `AM_Throw`.
2.  Finally, connect the `Triggered` execution output of the `EnhancedInputAction IA_Throw` event to the execution input pin of the `Play Montage` function:

![Figure 12.18 – Now the AM_Throw montage plays when the ThrowProjectile input is pressed ](img/Figure_12.18_B18531.jpg)

Figure 12.18 – Now the AM_Throw montage plays when the ThrowProjectile input is pressed

1.  Now, when you click your left mouse button, the player character will play the throwing Animation Montage.

Notice now how you can walk and run at the same time as throwing, and each animation blends together so as not to interfere with one another:

![Figure 12.19 – The player character can now move and throw ](img/Figure_12.19_B18531.jpg)

Figure 12.19 – The player character can now move and throw

Don’t worry about any bugs you might see when using the left mouse button action repeatedly to play the `Throw` montage; these issues will be addressed when you implement the projectile that will be thrown in a later chapter for this project. For now, you just want to know that the work done on the Anim Slot and the Animation Blueprint give the desired result for animation blending.

Let’s continue with the **SuperSideScroller** project by now creating the C++ class, the Blueprints, and the materials necessary to set up the enemy for use in the next chapter.

# The SuperSideScroller game enemy

With the player character animating correctly when moving and performing the `Throw` animation, it is time to talk about the enemy type that the **SuperSideScroller** game will feature.

This enemy will have a basic back-and-forth movement pattern and will not support any attacks; only by colliding with the player character will it be able to inflict damage.

In the next exercise, you will set up the base enemy class in C++ for the first enemy type and configure the enemy’s Blueprint and Animation Blueprint in preparation for [*Chapter 13*](B18531_13.xhtml#_idTextAnchor268), *Creating and Adding the Enemy Artificial Intelligence*, where you will implement the AI of this enemy. For the sake of efficiency and time, you will use the assets already provided by Unreal Engine 5 in the **SideScroller** template for the enemy. This means you will be using the skeleton, skeletal mesh, animations, and the Animation Blueprint of the default mannequin asset. Let’s begin by creating the first enemy class.

## Exercise 12.07 – Creating the enemy base C++ class

The goal of this exercise is to create a new enemy class from scratch and to have the enemy ready to use in [*Chapter 13*](B18531_13.xhtml#_idTextAnchor268), *Creating and Adding the Enemy Artificial Intelligence*, when you will develop the AI. To start, create a new enemy class in C++ by following these steps:

1.  In the editor, navigate to `SuperSideScrollreCharacter` parent class.
2.  Give this class a name and select a directory. Name this class `EnemyBase` and do not change the directory path. When ready, left-click on the **Create Class** button to have Unreal Engine create the new class for you.

Let’s create the folder structure in the content drawer for the enemy assets next.

1.  Head back to the Unreal Engine 5 editor, navigate to the content drawer, and create a new folder called `Enemy`:

![Figure 12.20 – The new Enemy folder ](img/Figure_12.20_B18531.jpg)

Figure 12.20 – The new Enemy folder

1.  In the `Enemy` folder, create another folder called `Blueprints`, where you will create and save the Blueprint assets for the enemy. Right-click and select `EnemyBase`, as shown here:

![Figure 12.21 – Now, the new EnemyBase class is available for you to create a Blueprint from ](img/Figure_12.21_B18531.jpg)

Figure 12.21 – Now, the new EnemyBase class is available for you to create a Blueprint from

1.  Name this `BP_Enemy`.

Now that you have the Blueprint for the first enemy using the `EnemyBase` class as the parent class, it is time to handle the Animation Blueprint. You will use the default Animation Blueprint that is provided to you by Unreal Engine in the `/Enemy/Blueprints` directory.

## Exercise 12.08 – Creating and applying the enemy Animation Blueprint

In the previous exercise, you created a Blueprint for the first enemy using the `EnemyBase` class as the parent class. In this exercise, you will be working with the Animation Blueprint.

The following steps will help you complete this exercise:

1.  Navigate to the `/Mannequin/Animations` directory and find the `ThirdPerson_AnimBP` asset.
2.  Now, duplicate the `ThirdPerson_AnimBP` asset. There are two ways to duplicate an asset:
    1.  Select the desired asset in the content drawer and press *Ctrl* + *W*.
    2.  Right-click on the desired asset in the content drawer and select `Duplicate` from the drop-down menu.
3.  Now, left-click and drag this duplicate asset into the `/Enemy/Blueprints` directory and select the option to move when you release the left-click mouse button.
4.  Name this duplicate asset `AnimBP_Enemy`. It is best to create a duplicate of an asset that you can later modify if you so desire without risking the functionality of the original.

With the enemy Blueprint and Animation Blueprint created, it’s time to update the enemy Blueprint to use the default `Skeletal Mesh` mannequin and the new Animation Blueprint duplicate.

1.  Navigate to `/Enemy/Blueprints` and open `BP_Enemy`.
2.  Next, navigate to the `Mesh` component and select it to access its **Details** panel. First, assign **SK_Mannequin** to the **Skeletal Mesh** parameter, as shown here:

![Figure 12.22 – You will use the default SK_Mannequin skeletal mesh for the new enemy ](img/Figure_12.22_B18531.jpg)

Figure 12.22 – You will use the default SK_Mannequin skeletal mesh for the new enemy

1.  Now, you need to apply the `AnimBP_Enemy` Animation Blueprint to the `Mesh` component. Navigate to the `Animation` category of the `Mesh` component’s `AnimBP_Enemy`:

![Figure 12.23 – Assign the new AnimBP_Enemy as the Anim class  ](img/Figure_12.23_B18531.jpg)

Figure 12.23 – Assign the new AnimBP_Enemy as the Anim class

1.  Lastly, you will notice that the character mesh is positioned and rotated incorrectly when previewing the character in the `X` = `0.000000`, `Y` = `0.000000`, `Z` = `-90.000000`)
2.  `0.000000`, Pitch= `0`, Yaw= `-90.000000`)
3.  `X` = `1.000000`, `Y` = `1.000000`, `Z` = `1.000000`)

The `Transform` settings will appear as follows:

![Figure 12.24 – The final Transform settings for the enemy character ](img/Figure_12.24_B18531.jpg)

Figure 12.24 – The final Transform settings for the enemy character

The following figure shows the settings of the **Mesh** component so far. Please make sure your settings match what is displayed in *Figure 12.25*:

![Figure 12.25 – The settings for the Mesh component of your enemy character ](img/Figure_12.25_B18531.jpg)

Figure 12.25 – The settings for the Mesh component of your enemy character

The last thing to do here is to create a Material Instance of the mannequin’s primary material so that this enemy can have a unique color that helps differentiate it from the other enemy type.

Let’s begin by first learning more about Materials and Material Instances.

# Materials and Material Instances

Before moving on to the next exercise, we need to first briefly discuss what Material Instances are before you can work with these assets and apply them to the new enemy character. Although this book is more focused on the technical aspects of game development using Unreal Engine 5, it is still important that you know, on a surface level, what Material Instances are and how they are used in video games. A Material Instance is an extension of a Material, where you do not have access or control over the base Material from which the Material Instance derives, but you do have control over the parameters that the creator of the Material exposes to you. Many parameters can be exposed to you to work with from inside Material Instances.

Note

For more information about Materials and Material Instances, please refer to the following Epic Games documentation pages: [https://docs.unrealengine.com/en-US/Engine/Rendering/Materials/index.xhtml](https://docs.unrealengine.com/en-US/Engine/Rendering/Materials/index.xhtml) and [https://docs.unrealengine.com/4.27/en-US/API/Runtime/Engine/Materials/UMaterialInstanceDynamic/](https://docs.unrealengine.com/4.27/en-US/API/Runtime/Engine/Materials/UMaterialInstanceDynamic/).

Unreal Engine provides us with an example of a Material Instance in the Side Scroller template project called `M_UE4Man_ChestLogo`, found in the `/Mannequin/Character/Materials/` directory. The following figure shows the set of exposed parameters given to the Material Instance based on the parent material, `M_Male_Body`. The most important parameter to focus on is the `Vector` parameter, called `BodyColor`. You will use this parameter in the Material Instance you create in the next exercise to give the enemy a unique color:

![Figure 12.26 – The list of parameters for the M_UE4Man_ChestLogo Material Instance asset ](img/Figure_12.26_B18531.jpg)

Figure 12.26 – The list of parameters for the M_UE4Man_ChestLogo Material Instance asset

In the following exercise, you will take this knowledge of Material Instances and apply them to create a unique Material Instance to be used for the enemy character you created earlier.

## Exercise 12.09 – Creating and applying the enemy Material Instance

Now that you have a basic understanding of what Material Instances are, it is time to create your own Material Instance from the `M_MannequinUE4_Body` asset. With this Material Instance, you will adjust the `BodyColor` parameter to give the enemy character a unique visual representation.

The following steps will help you complete this exercise:

1.  Navigate to the `Characters/Mannequin_UE4/Materials` directory to find the Material used by the default mannequin character, `M_MannequinUE4_Body`.
2.  A Material Instance can be created by right-clicking on the `Material` asset, `M_MannequinUE4_Body`, and left-clicking on the `MI_Enemy01`.

![Figure 12.27 – Any material can be used to create a Material Instance ](img/Figure_12.27_B18531.jpg)

Figure 12.27 – Any material can be used to create a Material Instance

Create a new folder called `Materials` in the `Enemy` folder. Left-click and drag the Material Instance into the `/Enemy/Materials` directory to move the asset to this new folder:

![Figure 12.28 – Rename the Material Instance MI_Enemy ](img/Figure_12.28_B18531.jpg)

Figure 12.28 – Rename the Material Instance MI_Enemy

1.  Double-left-click the Material Instance and find the **Details** panel on the left-hand side. There, you will find a **Vector Parameter** property called **BodyColor**. Make sure the checkbox is checked to enable this parameter, and then change its value to a red color. Now, the Material Instance should be colored red, as shown here:

![Figure 12.29 – Now, the enemy material is red ](img/Figure_12.29_B18531.jpg)

Figure 12.29 – Now, the enemy material is red

1.  Save the`BP_Enemy01` Blueprint. Select the `MI_Enemy`:

![Figure 12.30 – Assign the MI_Enemy material instance to the enemy character Mesh ](img/Figure_12.30_B18531.jpg)

Figure 12.30 – Assign the MI_Enemy material instance to the enemy character Mesh

1.  Now, the first enemy type is visually ready and has the appropriate Blueprint and Animation Blueprint assets prepared for the next chapter, where you will develop its AI:

![Figure 12.31 – The final enemy character set up ](img/Figure_12.31_B18531.jpg)

Figure 12.31 – The final enemy character set up

With this exercise complete, you have now created a Material Instance and applied it to the enemy character so that it has a unique visual representation.

Let’s conclude this chapter by moving on to a short activity that will help you better understand the blending of animations using the `Layered blend per bone` node that was used in the earlier exercises.

## Activity 12.01 – Updating Blend Weights

At the end of *Exercise 12.06*, *Previewing the Throw animation*, you were able to blend the movement animations and the `Throw` animation so that they could be played in tandem without negatively influencing each other. The result is the player character animating correctly when walking or running, while also performing the `Throw` animation on the upper body.

In this activity, you will experiment with the blend bias values and parameters of the `Layered blend per bone` node to have a better understanding of how animation blending works.

The following steps will help you complete the activity:

1.  Update the `Blend Weights` input parameter of the `Layered blend per bone` node so that there is absolutely no blending of the `Throw` animation additive pose with the base movement pose. Try using values here such as `0.0f` and `0.5f` to compare the differences in the animation.

Note

Make sure to return this value to `1.0f` after you are done so as not to affect the blending you set up in the previous exercise.

1.  Update the settings of the `Layered blend per bone` node to change which bone is affected by the blend so that the whole character’s body is affected by the blend. It’s a good idea to start with the root bone in the skeleton hierarchy of the `MainCharacter_Skeleton` asset.
2.  Keeping the settings from the previous step in place, add a new array element to the branch filters and, in this new array element, add the bone name and a blend depth value of `–1.0f`, which allows only the character’s left leg to continue to animate the movement correctly when blending the `Throw` animation.

Note

After this activity, make sure to return the settings of the `Layered blend per bone` node to the values you set at the end of the first exercise to ensure no progress is lost in the character’s animation.

The expected output for the first part of the activity is shown here:

![Figure 12.32 – Output showing the entire character’s body affected ](img/Figure_12.32_B18531.jpg)

Figure 12.32 – Output showing the entire character’s body affected

The expected output for the last part of the activity is shown here:

![Figure 12.33 – The left leg continues to animate the movement correctly when blending the Throw animation ](img/Figure_12.33_B18531.jpg)

Figure 12.33 – The left leg continues to animate the movement correctly when blending the Throw animation

Note

The solution to this activity can be found on GitHub here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions).

Before concluding this activity, please return the `Layered blend per bone` settings to the values you set at the end of *Exercise 12.05*, *Blending animation with the Upper Body Anim Slot*. If you do not return these values back to their original settings, the animation results in upcoming exercises and activities in the next chapters will not be the same. You can either set the original values manually or refer to the file with these settings at the following link: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Chapter12/Exercise12.05](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Chapter12/Exercise12.05).

With this activity complete, you now have a stronger understanding of how animation blending works and how blending weighting can affect the influence of additive poses on base poses using the `Layered blend per bone` node.

Note

There are a lot of techniques for animation blending that you haven’t used in this project, and it’s strongly recommended that you research these techniques, starting with the documentation at [https://docs.unrealengine.com/en-US/Engine/Animation/AnimationBlending/index.xhtml](https://docs.unrealengine.com/en-US/Engine/Animation/AnimationBlending/index.xhtml).

# Summary

With the enemy set up with the C++ class, Blueprint, and Material, you are ready to move on to the next chapter, where you will create the AI for this enemy by taking advantage of systems such as behavior trees in Unreal Engine 5.

From the exercises and activities of this chapter, you learned how to create an Animation Montage that allows the playing of animations. You also learned how to set up an Anim Slot within this montage to categorize it for the player character’s upper body.

Next, you learned how to cache the output pose of a State Machine by using the `Use Cached Pose` node so that this pose can be referenced in multiple instances for more complex Animation Blueprints. Then, by learning about the `Layered blend per bone` function, you were able to blend the base movement pose with the additive layer of the `Throw` animation by using the Anim Slot.

Lastly, you put together the base of the enemy by creating the C++ class, Blueprint, and other assets so that they will be ready for the next chapter. With the enemy ready, let’s move on to creating the AI of the enemy so that it can interact with the player.