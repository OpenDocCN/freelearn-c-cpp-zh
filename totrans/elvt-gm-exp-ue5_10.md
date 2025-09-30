# 10

# Creating the SuperSideScroller Game

So far, we have learned a lot about Unreal Engine, C++ programming, and general game development techniques. In the previous chapters, we covered collisions, tracing, how to use C++ with UE5, and even the Blueprint Visual Scripting system. On top of that, we gained crucial knowledge of skeletons, animations, and Animation Blueprints, all of which we will utilize in the upcoming project.

In this chapter, we will set up the project for a new SuperSideScroller game. You will be introduced to the different aspects of a side-scroller game, including power-ups, collectibles, and enemy AI, all of which we will be using in our project. You will also learn about the character animation pipeline in game development and learn how to manipulate the movement of our game’s character.

For our newest project, SuperSideScroller, we will use many of the same concepts and tools that we have used in previous chapters to develop our game features and systems. Concepts such as collision, input, and the HUD will be at the forefront of our project; however, we will also be diving into new concepts involving animation to recreate the mechanics of popular side-scrolling games. The final project will be a culmination of everything we have learned thus far in this book.

In this chapter, we’re going to cover the following main topics:

*   Project breakdown
*   The player character
*   Exploring the features of our side-scroller game
*   Understanding animations in Unreal Engine 5

By the end of this chapter, we’ll have a better idea of what we want to accomplish with our `SuperSideScroller` game, and we will have the project foundation to begin development.

# Technical requirements

For this chapter, you will need to have Unreal Engine 5 installed.

This chapter does not feature any C++ code, and all the exercises are performed within the UE5 editor. Let’s begin this chapter with a brief breakdown of the `SuperSideScroller` project.

The project for this chapter can be found in the Chapter10 folder of the code bundle for this book, which can be downloaded here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition).

# Project breakdown

Let’s consider the example of the classic game *Super Mario Bros*, released on the **Nintendo Entertainment System** (**NES**) console in 1985\. For those unfamiliar with the franchise, the general idea is this: the player takes control of Mario, who must traverse the many hazardous obstacles and creatures of the Mushroom Kingdom in the hope of rescuing Princess Peach from the sinister King Koopa, Bowser.

Note

To have an even better understanding of how the game works, check out this video of its gameplay: [https://www.youtube.com/watch?v=rLl9XBg7wSs](https://www.youtube.com/watch?v=rLl9XBg7wSs).

The following are the core features and mechanics of games in this genre:

*   `SuperSideScroller` game will be in 3D and not pure 2D, the movement of our character will work identically to that of Mario, only supporting vertical and horizontal movement:

![Figure 10.1 – A comparison of 2D and 3D coordinate vectors ](img/Figure_10.01_B18531.jpg)

Figure 10.1 – A comparison of 2D and 3D coordinate vectors

*   `SuperSideScroller` game will be no different. There are many different games, such as *Celeste*, *Hollow* *Knight*, and *Super Meat Boy*, as mentioned previously, that use the jumping feature – all of which are in 2D.
*   **Character Power-Ups**: Without character power-ups, many side-scrolling games lose their sense of chaos and replayability. For instance, in the game *Ori and the Blind Forest*, the developers introduce different character abilities that change how the game is played. Abilities such as the triple-jump or the air dash open a variety of possibilities to navigate the level and allow level designers to create interesting layouts based on the movement abilities of the player.
*   **Enemy AI**: Enemies with various abilities and behaviors are introduced to add a layer of challenge for the player, on top of the challenge of navigating the level solely through the use of the available movement mechanics.

Note

What are some ways that AI in games can interact with the player? For example, in *The Elder Scrolls V: Skyrim*, there are AI characters in various towns and villages that can have conversations with the player to exposit world-building elements such as history, sell items to the player, and even give quests to the player.

*   `SuperSideScroller` game will allow players to collect coins.

Now that we have evaluated the game mechanics that we want to support, we can break down the functionality of each mechanic as it relates to our `SuperSideScroller` game and what we need to do to implement these features.

# The player character

At the core of any game is the player character; that is, the entity in which our player will interact and play our game. For our `SuperSideScroller` project, we will be creating a simple character with custom meshes, animations, and logic behind it to give it the proper feel for a side-scroller game.

Almost all of the functionality that we want for our character is given to us by default when using the `Side Scroller` game project template in UE5\.

Note

At the time of writing, we are using Unreal Engine version 5.0.0; using another version of the engine could result in some differences in the editor, the tools, and how your logic will work later, so please keep this in mind.

In the next exercise, we will create our game project and set up our player character, while also exploring how we can manipulate the parameters of the character to improve upon its movement.

# Converting the Third Person template into a side-scroller

Back in Unreal Engine 4, the engine came with a `Side-Scroller` template that could be used as the base template for the `SuperSideScroller` project; however, in UE5, no such template exists. As a result, we will be using the `Third Person` template project provided by UE5 and updating some parameters to make it look and feel like a side-scroller game.

Let’s begin by creating our project.

## Exercise 10.01 – Creating the side-scroller project and using the Character Movement component

In this exercise, you will be setting up UE5 with the `Third Person` template. This exercise will help you get started with our game.

Follow these steps to complete the exercise:

1.  First, open the Epic Games Launcher, navigate to the **Unreal Engine** tab at the bottom of the options on the left-hand side, and select the **Library** option at the top.
2.  Next, you will be prompted with a window asking you to either open an existing project or create a new project of a certain category. Among these options is the **Games** category; select this option for our project. With your project category selected, you will be prompted to select the template for your project.
3.  Next, click on the **Third Person** option; because the **Side Scroller** template no longer exists, the **Third Person** template is the closest option we have.

Lastly, we need to set up the default project settings before Unreal Engine will create our project for us.

1.  Choose to base the project on `C++`, not `Blueprints`, include `Starter Content`, and use `Desktop/Console` as our platform. The remaining project settings can be left as their defaults. Select the desired location, name the project `SuperSideScroller`, and save the project in an appropriate directory of your choice.
2.  After these settings are applied, select **Create Project**. When it’s done compiling the engine, both Unreal Editor and Visual Studio will open, and we can move on to the next steps of this exercise:

![Figure 10.2 – The Unreal Engine editor should now be open ](img/Figure_10.02_B18531.jpg)

Figure 10.2 – The Unreal Engine editor should now be open

Now that our project has been created, we need to perform a handful of steps to change the `Third Person` template to a `Side Scroller`, starting with updating the input **Axis Mappings**. Follow these steps:

1.  We can access **Axis Mappings** via **Project Settings** by selecting the **Edit** drop-down menu at the top-left of the editor and selecting the **Project Settings** option.
2.  In **Project Settings**, we can find the **Input** option under the **Engine** category. Select the **Input** option to find the **Bindings** section, which contains both **Action** and **Axis Mappings** for the project:

![Figure 10.3 – The default Axis and Action Mappings ](img/Figure_10.03_B18531.jpg)

Figure 10.3 – The default Axis and Action Mappings

1.  For the needs of the `SuperSideScroller` project, we simply have to remove `MoveForward`, `TurnRate`, `Turn`, `LookUpRate`, and `LookUp`. You can remove a mapping by left-clicking the garbage can icon next to it.

These mappings are unnecessary for our project due to the behavior of the character controls for a side-scroller game. Now that the mappings have been updated, we can update the parameters within the **ThirdPersonCharacter** Blueprint. Follow these steps:

1.  Find the `Content/ThirdPersonCPP/Blueprints` directory. Then, open the asset.
2.  With the `-90.0f`. The final rotation should be `(Pitch=0.0,Yaw=-90.0,Roll=0.0)`. This will ensure that the character mesh will be facing the axis in which our side-scroller will move:

![Figure 10.4 – The updated Rotation values of the Mesh component ](img/Figure_10.04_B18531.jpg)

Figure 10.4 – The updated Rotation values of the Mesh component

1.  Next, we need to update the parameters within the `180.0f`, with the final rotation as `(Pitch=0.0,Yaw=180.0,Roll=0.0)`:

![Figure 10.5 0 – The updated Rotation values of the Camera Boom component ](img/Figure_10.05_B18531.jpg)

Figure 10.5 0 – The updated Rotation values of the Camera Boom component

1.  Now, we need to update the `500.0f` and set the `Z` value of `75.0f`. This will give us a good relative positioning of the **Follow Camera** component to the character mesh:

![Figure 10.6 – The updated Target Arm Length and Target Offset parameters ](img/Figure_10.06_B18531.jpg)

Figure 10.6 – The updated Target Arm Length and Target Offset parameters

1.  The final parameter we need to update in the `False`.

The next set of parameters can be within **Character Movement Component**, which we will talk about more later in this chapter. For now, all you need to know is that this component controls all aspects of the character's movement and allows us to customize it in a way to give us the game feel we desire. Follow these steps:

1.  Select `2.0f`. This will increase the gravity for our character:

![Figure 10.7 – The updated Gravity Scale parameter ](img/Figure_10.07_B18531.jpg)

Figure 10.7 – The updated Gravity Scale parameter

1.  Next, we need to decrease the value of the `3.0f`. The higher the value of **Ground Friction**, the more difficult it will be for the character to turn and move:

![Figure 10.8 – The updated Ground Friction parameter ](img/Figure_10.08_B18531.jpg)

Figure 10.8 – The updated Ground Friction parameter

Let’s adjust the parameters that control the jump velocity and the air control the player has while the character is in the air. We can find both parameters under `1000.0f`, and `0.8f`. Updating these values gives our character an interesting jump height and movement while in the air:

![Figure 10.9 – The updated Jump Z Velocity and Air Control parameters ](img/Figure_10.09_B18531.jpg)

Figure 10.9 – The updated Jump Z Velocity and Air Control parameters

The next set of parameters need to be set to help us later on in [*Chapter 13*](B18531_13.xhtml#_idTextAnchor268)*, Creating and Adding the Enemy Artificial Intelligence*, when we work with **Nav Meshes**. Under the **Nav Movement** section of **Character Movement Component**, we need to update both **Nav Agent Radius** and **Nav Agent Height** to fit the bounds of **Capsule Component** on our player character. Follow these steps:

1.  Set `42.0f` and `192.0f`:

![Figure 10.10 – The updated values of the Nav Agent Radius and Nav Agent Height parameters ](img/Figure_10.10_B18531.jpg)

Figure 10.10 – The updated values of the Nav Agent Radius and Nav Agent Height parameters

1.  Lastly, we need to adjust the `1.0f`; the final value will be `(X=1.0f,Y=0.0,Z=0.0)`:

![Figure 10.11 – The updated values of the Constrain to Plane and Plane Constraint Normal parameters ](img/Figure_10.11_B18531.jpg)

Figure 10.11 – The updated values of the Constrain to Plane and Plane Constraint Normal parameters

The final step is to add some simple Blueprint logic to the `Event Graph` area of `ThirdPersonCharacter` to allow our character to move from left to right. Follow these steps:

1.  In the **Event Graph** area, right-click in an empty space of the graph to open the context-sensitive menu, where we will look for the **InputAxis MoveRight** event. Select the **InputAxis MoveRight** event to add it to the graph:

![Figure 10.12 – This is the Axis Mapping we kept at the beginning of this exercise ](img/Figure_10.12_B18531.jpg)

Figure 10.12 – This is the Axis Mapping we kept at the beginning of this exercise

1.  The output parameter of the **InputAxis MoveRight** event is a float value called **Axis Value**. This returns a float value between 0 and 1, indicating the strength of the input in that direction. We will need to feed this value into a function called **Add Movement Input**. Right-click in another empty space and find this function to add it to the graph:

![Figure 10.13 – The Add Movement Input function ](img/Figure_10.13_B18531.jpg)

Figure 10.13 – The Add Movement Input function

1.  Connect the **Axis Value** output parameter of the **InputAxis MoveRight** event to the **Scale Value** input parameter of the **Add Movement Input** function, then connect the white execution pins, as shown in the following screenshot. This allows us to add character movement in a specified direction, as well as strength:

![Figure 10.14 – The final Blueprint logic of our character ](img/Figure_10.14_B18531.jpg)

Figure 10.14 – The final Blueprint logic of our character

1.  Lastly, we need to ensure that we pass in the right `1.0f` and leave the other axes at their default values of `0.0f`.

Now that you have completed the exercise, you have experienced first-hand the control you have over how the character moves and how small tweaks to **Character Movement Component** can drastically change how the character feels! Try changing the values such as **Max Walk Speed** and observe in-game how such changes affect the character.

## Activity 10.01 – Making our character jump higher

In this activity, we will be manipulating a new parameter (`jump`) that exists within the `CharacterMovement` component of the default `Side Scroller` Character Blueprint to observe how these properties affect how our character moves.

We will be implementing what we learned in the previous exercise and applying that to how to create our character power-ups and the general movement feel of our character.

Follow these steps to complete this activity:

1.  Head to the `SideScrollerCharacter` Blueprint and find the `CharacterMovement` component.
2.  Change this parameter from the default value of `1000.0f` to `2000.0f`.
3.  Compile and save the `SideScrollerCharacter` Blueprint and play it in the editor. Observe how high our character can jump using the *space bar* on your keyboard.
4.  Stop playing in the editor, return to the `SideScrollerCharacter` Blueprint, and update `2000.0f` to `200.0f`.
5.  Compile and save the Blueprint again, play it in the editor, and watch the character jump.

**Expected output**:

![Figure 10.15 – The expected output with the jumping character ](img/Figure_10.15_B18531.jpg)

Figure 10.15 – The expected output with the jumping character

Note

The solution to this activity can be found on GitHub here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions).

Now that we have completed this activity, we have a better understanding of how making a few changes to the `CharacterMovement` component parameters can affect our player character. We can use this later on when we need to give our character basic movement behaviors such as `1000.0f`.

We will also keep these parameters in mind when we develop our player character power-ups later in our project.

Now that we have established our game project and player character, let’s explore the other features of our `SuperSideScroller` game.

# Exploring the features of our side-scroller game

Now, we’ll take some time to lay out the specifics of the game we’ll be designing. Many of these features will be implemented in later chapters, but now is a good time to lay out the vision for the project. In the following sections, we will be discussing how we want to handle the different aspects of our game, such as the enemies the player will face, the power-ups available to the player, the collectibles for the player to collect, and how the **user interface** (**UI**) will work. Let’s begin by discussing the enemy character.

## Enemy character

One thing you should have noticed while playing the `SuperSideScroller` project is that there is no enemy AI by default. Let’s discuss the type of enemy we will want to support and how they will work.

The enemy will have a basic back-and-forth movement pattern and will not support any attacks; only by colliding with the player character will they be able to inflict any damage. However, we need to set the two locations to move between for the enemy AI, as well as decide whether the AI should change locations. Should they constantly move between locations, or should there be a pause before selecting a new location to move to? In [*Chapter 13*](B18531_13.xhtml#_idTextAnchor268), *Creating and Adding the Enemy Artificial Intelligence*, we will use the tools available in UE5 to develop this AI logic.

## Power-ups

The `SuperSideScroller` game project will support one type of power-up, in the form of a potion that the player can pick up from the environment. This potion power-up will increase the movement speed of the player and the maximum height to which the player can jump. These effects will only last a short duration before they are removed.

Keeping in mind what you implemented in *Exercise 10.01 – Creating the side-scroller project and using the Character Movement component*, and *Activity 10.01 – Making our character jump higher*, for the `CharacterMovement` component, you could develop a power-up that changes the effect of gravity on the character, which would provide interesting new ways to navigate the level and combat enemies.

## Collectibles

Collectibles in video games serve different purposes. In some cases, collectibles are used as a form of currency to purchase upgrades, items, and other goods. In others, collectibles serve to improve your score or reward you when enough collectibles have been collected. For the `SuperSideScroller` game project, the coins will serve a single purpose: to give the player the goal of collecting as many coins as they can without being destroyed by the enemy.

Let’s break down the main aspects of our collectible:

*   The collectible needs to interact with our player; this means that we need to use collision detection for the player to collect it and for us to add information to our UI.
*   The collectible needs a visual static mesh representation so that the player can identify it in the level.

The final element of our `SuperSideScroller` project is the brick block. The brick block will serve the following purposes for the `SuperSideScroller` game:

*   Bricks are used as an element of the level’s design. Bricks can be used to access otherwise unreachable areas; enemies can be placed on different elevated sections of bricks to provide variation in gameplay.
*   Bricks can contain collectible coins. This gives the player an incentive to try and see which blocks contain collectibles and which do not.

## Heads-Up Display (HUD)

The HUD UI can be used to display important and relevant information to the player, based on the type of game and the mechanics that you support. For the `SuperSideScroller` project, there will be one HUD element, which will show the player how many coins they have collected. This UI will be updated each time the player collects a coin, and it will reset back to `0` when the player is destroyed.

Now that we have laid out some of the specifics that we will be working toward as part of this project, let’s learn more about the default skeletal mesh provided by the project template in UE5.

## Exercise 10.02 – Exploring the Persona Editor and manipulating the default mannequin skeleton weights

Now that we have a better understanding of the different aspects of the `SuperSideScroller` project, let’s go ahead and take a deeper look into the default mannequin skeletal mesh that is given to us in the **Side Scroller** template project.

Our goal here is to learn more about the default skeletal mesh and the tools that are given to us in the Persona Editor so that we have a better understanding of how bones, bone weighting, and skeletons work inside UE5.

Follow these steps to complete the exercise:

1.  Open Unreal Editor and navigate to **Content Drawer**.
2.  Navigate to the `/Characters/Mannequins/Meshes/` folder and open the `SK_Mannequin` asset:

![Figure 10.16 – The SK_Mannequin asset is highlighted and visible here ](img/Figure_10.16_B18531.jpg)

Figure 10.16 – The SK_Mannequin asset is highlighted and visible here

Upon opening the **Skeleton** asset, the **Persona Editor** area will appear:

![Figure 10.17 – The Persona Editor ](img/Figure_10.17_B18531.jpg)

Figure 10.17 – The Persona Editor

Let’s briefly break down the Skeleton Editor of the Persona Editor:

*   On the left-hand side (*marked with a 1*), we can see the hierarchy of bones that exist in the skeleton. This is the skeleton that was made during the rigging process of this character. The `root` bone, as its name suggests, is the root of the skeletal hierarchy. This means that transformative changes to this bone will affect all of the bones in the hierarchy. From here, we can select a bone or a section of bones and see where they are on the character mesh.
*   Next, we see the **Skeletal Mesh** preview window (*marked with a 2*). It shows us our character mesh, and there are several additional options that we can toggle on/off that will give us a preview of our skeleton and weight painting.
*   On the right-hand side (*marked with a 3*), we have basic transformation options where we can modify individual bones or groups of bones. If the **Details** panel is not available, navigate to the **Window** tab at the top of the Persona Editor; you will find it in the list of options there. There are additional settings available that we will take advantage of in the next exercise. Now that we know more about what it is and what we are looking at, let’s see what the actual skeleton looks like on our mannequin.

1.  Navigate to **Character**, as shown in *Figure 10.10*:

![Figure 10.18 – The Character options menu  ](img/Figure_10.18_B18531.jpg)

Figure 10.18 – The Character options menu

This menu allows you to display the skeleton of the mannequin over the mesh itself.

1.  From the drop-down menu, select the **Bones** option. Then, make sure the option for **All Hierarchy** is selected. With this option selected, you will see the outlined skeleton rendering above the mannequin mesh:

![Figure 10.19 – The skeleton overlayed on top of the mannequin Skeletal Mesh ](img/Figure_10.19_B18531.jpg)

Figure 10.19 – The skeleton overlayed on top of the mannequin Skeletal Mesh

1.  Now, hide the mesh and simply preview the skeletal hierarchy, for which we can disable the **Mesh** property:
    *   Navigate to **Character** and, from the drop-down menu, select the **Mesh** option.
    *   Deselect the option for **Mesh**. The result should be as follows:

![Figure 10.20 – The skeletal hierarchy of the default character ](img/Figure_10.20_B18531.jpg)

Figure 10.20 – The skeletal hierarchy of the default character

For this exercise, let’s toggle the **Mesh** visibility back on so that we can see both the mesh and the skeleton hierarchy.

Finally, we’ll look at the weight scaling for our default character.

1.  To preview this, navigate to **Character** and, from the drop-down menu, select the **Mesh** option. Then, select the **Selected Bone Weight** option toward the bottom in the **Mesh Overlay Drawing** section:

![Figure 10.21 – The Selected Bone Weight option ](img/Figure_10.21_B18531.jpg)

Figure 10.21 – The Selected Bone Weight option

1.  Now, if we select a bone or a group of bones from our hierarchy, we can see how each bone affects a certain area of our mesh:

![Figure 10.22 – This is the weight scaling for the spine_03 bone ](img/Figure_10.22_B18531.jpg)

Figure 10.22 – This is the weight scaling for the spine_03 bone

You will notice that when we are previewing the weight scaling for a particular bone, there is a spectrum of colors across different sections of the Skeletal Mesh. This is the weight scaling shown visually instead of numerically. Colors such as `red`, `orange`, and `yellow` indicate larger weighting for a bone, meaning that the highlighted area of the mesh in these colors will be more affected. In areas that are `blue`, `green`, and `cyan`, they will still be affected, but not as significantly. Lastly, areas that have no overlay highlight will not be affected at all by the manipulation of the selected bone. Keep the hierarchy of the skeleton in mind –even though the left arm does not have an overlay color, it will still be affected when you are rotating, scaling, and moving the `spine_03` bone, since the arms are children of the `spine_03` bone. Please refer to the following screenshot to see how the arms are connected to the spine:

![Figure 10.23 – The clavicle_l and clavicle_r bones are children of the spine_03 bone ](img/Figure_10.23_B18531.jpg)

Figure 10.23 – The clavicle_l and clavicle_r bones are children of the spine_03 bone

Let’s continue by manipulating one of the bones on the mannequin Skeletal Mesh and see how these changes affect its animation. Follow these steps:

1.  In the `thigh_l` bone in the skeletal hierarchy:

![Figure 10.24 – Here, the thigh_l bone is selected ](img/Figure_10.24_B18531.jpg)

Figure 10.24 – Here, the thigh_l bone is selected

With the `thigh_l` bone selected, we have a clear indication of how the weight scaling will affect other parts of the mesh. Also, because of how the skeleton is structured, any modifications to this bone will not impact the upper body of the mesh:

![Figure 10.25 – On the skeletal bone hierarchy, the thigh_l bone is a child of the pelvis bone ](img/Figure_10.25_B18531.jpg)

Figure 10.25 – On the skeletal bone hierarchy, the thigh_l bone is a child of the pelvis bone

1.  Using our knowledge from earlier chapters, change the `thigh_l` bone. The following screenshot shows an example of the values you can use:

![Figure 10.26 – The thigh_l values updated ](img/Figure_10.26_B18531.jpg)

Figure 10.26 – The thigh_l values updated

After making these changes to the bone transform, you will see that the mannequin’s left leg has completely changed and looks ridiculous:

![Figure 10.27 – The left leg of the mannequin has completely changed ](img/Figure_10.27_B18531.jpg)

Figure 10.27 – The left leg of the mannequin has completely changed

1.  Next, in the **Details** panel, go to the **Preview Scene Settings** tab. Upon left-clicking this tab, you will see new options, displaying some default parameters and an **Animation** section. If **Preview Scene Settings** is not available, navigate to the **Window** tab at the top of the **Persona Editor** area; you will find it in the list of options there.
2.  Use the **Animation** section to preview animations and how they are affected by the changes that are made to the skeleton. For the **Preview Controller** parameter, change that to the **Use Specific Animation** option. By doing this, a new option labeled **Animation** will appear. The **Animation** parameter allows us to choose an animation associated with the character skeleton to preview.
3.  Next, left-click on the drop-down menu and select the `MF_Walk_Fwd` animation.
4.  Finally, you will see the mannequin character playing the walking animation, but their left leg is completely misplaced and mis-scaled:

![Figure 10.28 – Preview of the updated animation for the mannequin character ](img/Figure_10.28_B18531.jpg)

Figure 10.28 – Preview of the updated animation for the mannequin character

Before moving on, make sure to return the `thigh_l` bone to its original **Local Location**, **Local Rotation**, and **Scale**; otherwise, the animations moving forward will not look correct.

Now that you have completed the final part of our second exercise, you have experienced first-hand how skeletal bones affect characters and animations.

Now, let’s move on to our second activity. Here, we will manipulate a different bone on the mannequin character and observe the results of applying different animations.

## Activity 10.02 – Skeletal bone manipulation and animations

For this activity, we will put the knowledge we have gained about manipulating bones on the default mannequin into practice and affect how the animations are played out on the skeleton.

Follow these steps to complete this activity:

1.  Select the bone that will affect the entire skeleton.
2.  Change the scale of this bone so that the character is half its original size. Use these values to change `(X=0.500000, Y=0.500000, Z=0.500000)`.
3.  Apply the running animation to this Skeletal Mesh from the **Preview Scene Settings** tab and observe the animation for the half-sized character.

Here is the expected output:

![Figure 10.29 – The character has been halved in size and is performing the running animation ](img/Figure_10.29_B18531.jpg)

Figure 10.29 – The character has been halved in size and is performing the running animation

Note

The solution to this activity can be found on GitHub here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions).

With this activity complete, you now have practical knowledge of how bone manipulation of skeletons and Skeletal Meshes affects how animations are applied. You have also seen first-hand the effects weight scaling have on the bones of a skeleton.

Now that we have some experience with Skeletal Meshes, skeletons, and animations within UE5, let’s have a deeper discussion about these elements and how they work.

# Understanding animations in Unreal Engine 5

Let’s break down the main aspects of animations as they function inside Unreal Engine. More in-depth information about the topics in this section can be found in the documentation that is available directly from Epic Games: [https://docs.unrealengine.com/en-US/Engine/Animation](https://docs.unrealengine.com/en-US/Engine/Animation).

## Skeletons

Skeletons are Unreal Engine’s representation of the character rig that was made in external 3D software; we saw this in *Activity 10.02 – Skeletal bone manipulation and animations*. There isn’t much more to skeletons that we haven’t discussed already, but the main takeaway is that once the skeleton is in the engine, we can view the skeleton hierarchy, manipulate each bone, and add objects known as sockets. What sockets allow us to do is attach objects to the bones of our character. We can use these sockets to attach objects such as meshes and manipulate the transformation of the sockets without disrupting the bones’ transformation. In first-person shooters, typically, a weapon socket is made and attached to the appropriate hand.

## Skeletal Meshes

A Skeletal Mesh is a specific kind of mesh that combines the 3D character model and the hierarchy of bones that make up its skeleton. The main difference between a Static Mesh and a Skeletal Mesh is that Skeletal Meshes are required for objects that use animations, while Static Meshes cannot use animations due to their lack of a skeleton. We will look more into our main character Skeletal Mesh in the next chapter, but we will be importing our main character Skeletal Mesh in *Activity 10.03 – Importing more custom animations to preview the character running*, later in this chapter.

## Animation sequences

Finally, an animation sequence is an individual animation that can be played on a specific Skeletal Mesh; the mesh it applies to is determined by the skeleton selected while importing the animation into the engine. We will look at importing a character Skeletal Mesh and a single animation asset together in *Activity 10.03 – Importing more custom animations to preview the character running*.

Included in our animation sequence is a timeline that allows us to preview the animation frame by frame, with additional controls to pause, loop, rewind, and so on.

In the next exercise, you will import a custom character and an animation. The custom character will include a Skeletal Mesh and a skeleton, and the animation will be imported as an animation sequence.

## Exercise 10.03 – Importing and setting up the character and animation

For our final exercise, we will import our custom character and a single animation that we will use for the `SuperSideScroller` game’s main character, as well as create the necessary Character Blueprint and Animation Blueprint.

Note

Included with this chapter is a set of files in a folder labeled `Assets`, and it is these files that we will import into the engine. These assets come from Mixamo: [https://www.mixamo.com/](https://www.mixamo.com/). Feel free to create an account and view the free 3D character and animation content available there.

The `Assets` content is available in this book’s GitHub repository: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition).

Follow these steps to complete this exercise:

1.  Open Unreal Editor.
2.  In the `MainCharacter`. Within this folder, create two new folders called `Animation` and `Mesh`. The **Content Browser** area should now look as follows:

![Figure 10.30 – Folders added to the MainCharacter directory in the Content Browser area ](img/Figure_10.30_B18531.jpg)

Figure 10.30 – Folders added to the MainCharacter directory in the Content Browser area

1.  Next, let’s import our character mesh. Inside the `Mesh` folder, right-click and select the `Assets` folder that accompanies this chapter and find the `MainCharacter.fbx` asset inside the `Character Mesh` folder – for example, `\Assets\Character Mesh\MainCharacter.fbx` – and open that file.
2.  When selecting this asset, the `Skeletal Mesh` and `Import Mesh` are set to `check` in their respective checkboxes and leave every other option set to its default setting.
3.  Lastly, we can select the `Physics Asset`, which will automatically be created for us and assigned to `Skeletal Mesh`; and a `Skeleton Asset`.

Note

Ignore any warnings that may appear when importing the `FBX` file; they are unimportant and will not affect our project moving forward.

Now that we have our character, let’s import an animation. Follow these steps:

1.  Inside our `Animation` folder in the `MainCharacter` folder directory, right-click and select **Import**.
2.  Navigate to the directory where you saved the `Assets` folder that accompanies this chapter and locate the `Idle.fbx` asset inside the `Animations/Idle` folder – for example, `\Assets\Animations\Idle\Idle.fbx` – and open that file.

When selecting this asset, an almost identical window will appear as when we imported our character Skeletal Mesh. Since this asset is only an animation and not a Skeletal Mesh/skeleton, we don’t have the same options as before, but there is one crucial parameter that we need to set correctly: `Skeleton`.

The `Skeleton` parameter under the **Mesh** category of our **FBX** import options tells the animation to which skeleton the animation applies. Without this parameter set, we cannot import our animation, and applying the animation to the wrong skeleton can have disastrous results or cause the animation to not import altogether. Luckily for us, our project is simple, and we have already imported our character’s Skeletal Mesh and skeleton.

1.  Select `MainCharacter_Skeleton` and choose **Import All** at the bottom; leave all the other parameters set to their defaults:

![Figure 10.31 – The settings when importing the Idle.fbx animation ](img/Figure_10.31_B18531.jpg)

Figure 10.31 – The settings when importing the Idle.fbx animation

Understanding the importing process for both skeletal meshes and animations is crucial, and in the next activity, you will import the remaining animations. Let’s continue this exercise by creating both the Character Blueprint and the Animation Blueprint for the `SuperSideScroller` game’s main character.

Now, although the **Side Scroller** template project does include a Blueprint for our character and other assets such as an Animation Blueprint, we will want to create our own versions of these assets for the sake of organization and good practice as game developers.

1.  Create a new folder under our `MainCharacter` directory in the `Blueprints`. In this directory, create a new Blueprint based on the `SideScrollerCharacter` class under `All Classes`. Name this new Blueprint `BP_SuperSideScroller_MainCharacter`:

![Figure 10.32 – The SideScrollerCharacter class to be used as the parent class for our character Blueprint ](img/Figure_10.32_B18531.jpg)

Figure 10.32 – The SideScrollerCharacter class to be used as the parent class for our character Blueprint

1.  In our `Blueprints` directory, right-click in an empty area of the **Content Browser** area, hover over the **Animation** option, and select **Animation Blueprint**:

![Figure 10.33 – The Animation Blueprint option under the Animation category ](img/Figure_10.33_B18531.jpg)

Figure 10.33 – The Animation Blueprint option under the Animation category

1.  After we select this option, a new window will appear. This new window requires us to apply a parent class and a skeleton to our Animation Blueprint. In our case, use `MainCharacter_Skeleton`, select `AnimBP_SuperSideScroller_MainCharacter`:

![Figure 10.34 – The settings we need when creating our Animation Blueprint ](img/Figure_10.34_B18531.jpg)

Figure 10.34 – The settings we need when creating our Animation Blueprint

1.  When we open our Character Blueprint, `BP_SuperSideScroller_MainCharacter`, and select the **Mesh** component, we will find a handful of parameters that we can change:

![Figure 10.35 – The SuperSideScroller Character Blueprint using the mannequin Skeletal Mesh ](img/Figure_10.35_B18531.jpg)

Figure 10.35 – The SuperSideScroller Character Blueprint using the mannequin Skeletal Mesh

1.  Under the `MainCharacter` Skeletal Mesh and assign it to this parameter:

![Figure 10.36 – The settings we need for our Mesh component ](img/Figure_10.36_B18531.jpg)

Figure 10.36 – The settings we need for our Mesh component

While still in our Character Blueprint and with the **Mesh** component selected, we can find the **Animation** category just above the **Mesh** category. Luckily, by default, the **Animation Mode** parameter is already set to **Use Animation Blueprint**, which is the setting we need.

1.  Now, assign the `Anim` class parameter to our new Animation Blueprint, `AnimBP_SuperSideScroller_MainCharacter`. Finally, head back to our default `SideScrollerExampleMap` level and replace the default character with our new Character Blueprint.
2.  Next, make sure that we have `BP_SuperSideScroller_MainCharacter` selected in the **Content Browser** area, right-click on the default character in our map, and choose to replace it with our new character:

With our new character in the level, we can play in the editor and move around the level. The result should look something like what’s shown in the following screenshot; our character is in the default T-pose and moving around the level environment:

![Figure 10.37 – You now have the custom character running around the level ](img/Figure_10.37_B18531.jpg)

Figure 10.37 – You now have the custom character running around the level

With our final exercise complete, you have a full understanding of how to import custom Skeletal Meshes and animations. Additionally, you learned how to create a Character Blueprint and an Animation Blueprint from scratch and how to use those assets to create the base for the `SuperSideScroller` character.

Let’s move on to the final activity of this chapter, where you will be challenged to import the remaining animations for the character and preview the running animation inside Persona Editor.

## Activity 10.03 – Importing more custom animations to preview the character running

This activity aims to import the remaining animations, such as running for the player character, and preview the running animation on the character skeleton to ensure that it looks correct.

By the end of the activity, all of the player character animations will be imported into the project and you will be ready to use these animations to bring the player character to life in the next chapter.

Follow these steps to complete this activity:

1.  As a reminder, all of the animation assets we need to import exist in the `\Assets\Animations` directory, wherever you may have saved the original .`zip` folder. Import all of the remaining animations in the `MainCharacter/Animation` folder. Importing the remaining animation assets will work the same way as in *Exercise 10.03 – Importing and setting up the character and animation*, when you imported the `Idle` animation.
2.  Navigate to the `MainCharacter` skeleton and apply the `Running` animation you imported in the previous step.
3.  Finally, with the `Running` animation applied, preview the character animation in the **Persona Editor** area.

Here is the expected output:

![Figure 10.38 – The expected output of the character with additional custom imported assets ](img/Figure_10.38_B18531.jpg)

Figure 10.38 – The expected output of the character with additional custom imported assets

Note

The solution to this activity can be found on GitHub here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions).

With this final activity completed, you have now experienced first-hand the process of importing custom skeletal and animation assets into UE5\. This import process, regardless of the type of asset you are importing, is commonplace in the games industry, and you must be comfortable with it.

# Summary

With the player character skeleton, Skeletal Mesh, and animations imported into the engine, we can move on to the next chapter, where we will prepare the character movement and Update Animation Blueprint so that the character can animate while moving around the level.

From the exercises and activities of this chapter, you learned about how the skeleton and bones are used to animate and manipulate the character. With first-hand experience in importing and applying animations in UE5, you now have a strong understanding of the animation pipeline, from the character concept to the final assets being imported for your project.

We also took the necessary steps to outline what we want to accomplish with our `SuperSideScroller` game; that is, establishing how we want enemies to work, which power-ups to develop, how collectibles will work, and how the player HUD will look. Lastly, we explored how the character movement component works and how to manipulate its parameters to establish the character movement we desire for our game.

Additionally, you learned about what we will use in the next chapter, such as blend spaces for character movement animation blending. With the `SuperSideScroller` project template created and the player character ready, in the next chapter, we’ll animate the character with an Animation Blueprint.