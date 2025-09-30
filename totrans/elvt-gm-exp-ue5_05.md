# 5

# Query with Line Traces

In previous chapters, we learned about how we can reproduce the Third Person template project offered to us by the Unreal Engine team in order to understand some of the basic concepts of UE5’s workflow and framework.

In this chapter, you will start creating another game from scratch. In this game, the player will control a character from a top-down point of view (similar to games such as Metal Gear Solid 1, 2, and 3). A top-down perspective implies that the player controls a character that is seen as if it was being looked down upon, usually with the camera rotation being fixed (the camera doesn’t rotate). In our game, the player character must go from point A to point B without being hit by dodgeballs, which are being thrown at the player by the enemies that are spread throughout the level. The levels in this game will be maze-like in nature, and the player will have multiple paths to choose from, all of which will have enemies trying to throw dodgeballs at them.

In this chapter, we’ll cover the following topics:

*   Introduction to collision
*   Understanding and visualizing Line Traces (Single and Multi)
*   Sweep Traces
*   Trace Channels
*   Trace Responses

In the first section, we begin by getting to know what collision is in the world of video games.

# Technical requirements

The project for this chapter can be found in the Chapter05 folder of the code bundle for this book, which can be downloaded here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition).

# Introduction to collision

A collision is basically a point at which two objects come into contact with each other (for example, two objects colliding, an object hitting a character, a character walking into a wall, and so on). Most game development tools have their own set of features that allow for collision and physics to exist inside the game. This set of features is called a physics engine, which is responsible for everything related to collisions. It is responsible for executing Line Traces, checking whether two objects are overlapping each other, blocking each other’s movement, or bouncing off of a wall, and much more. When we ask the game to execute or notify us of these collision events, the game is essentially asking the physics engine to execute it and then show us the results of these collision events.

In the **Dodgeball** game you will be building, examples of where collision needs to be taken into account include checking whether enemies are able to see the player (which will be achieved using a Line Trace, which is covered in this chapter), simulating physics on an object that will behave just like a dodgeball, checking whether anything is blocking the player character’s movement, and much more.

Collision is one of the most important aspects of most games, so understanding it is crucial in order to get started with game development.

Before we start building our collision-based features, we will first need to set up our new **Dodgeball** project in order to support the game mechanics we will be implementing. This process starts with the steps described in the following section.

# Setting up your project

Let’s begin by creating our Unreal Engine project:

1.  Launch UE5\. Select the **Games** project category, then click on **Next**.
2.  Select the **Third Person template**, then click on **Next**.
3.  Make sure the first option is set to **C++** and not **Blueprint**.
4.  Select the location of the project according to your preference and name your project `Dodgeball`, then click on **Create Project**.

When the project is done being generated, you should see the following on your screen:

![Figure 5.1 – Dodgeball project loaded up](img/Figure_5.01_B18531.jpg)

Figure 5.1 – Dodgeball project loaded up

1.  After the code has been generated and the project opens up, enable the Enhanced Input plugin, just like we did in steps 1-3 of the *Understanding input actions and contexts* section in [*Chapter 4*](B18531_04.xhtml#_idTextAnchor099)*, Getting Started with Player Input*.
2.  After that, close the UE5 editor and open the files of the generated third-person `Character` class, `DodgeballCharacter`, in Visual Studio, as shown in the following figure:

![Figure 5.2 – Files generated in Visual Studio ](img/Figure_5.02_B18531.jpg)

Figure 5.2 – Files generated in Visual Studio

As mentioned earlier, your project is going to have a top-down perspective. Given that we’re starting this project from the Third Person template, we’ll have to change a few things before we turn this into a top-down game. This will mainly involve changing some lines of code in the existing `Character` class.

## Exercise 5.01: Converting DodgeballCharacter to a top-down perspective

In this exercise, you’ll be performing the necessary changes to your generated `DodgeballCharacter` class. Remember, it currently features a third-person perspective where the rotation of the character is dictated by the player’s input (*namely the mouse or right analog stick*).

In this exercise, you will change this to a top-down perspective, which remains the same regardless of the player’s input and the camera always follows the character from above.

The following steps will help you complete this exercise:

1.  Head to the `DodgeballCharacter` class’s constructor and update the `CameraBoom` properties, as mentioned in the following steps.
2.  Change `TargetArmLength`, which is a property of `CameraBoom`, to `900.0f` in order to add some distance between the camera and the player as follows:

    ```cpp
    // The camera follows at this distance behind the 
    // character
    CameraBoom->TargetArmLength = 900.0f;
    ```

3.  Next, add a line that sets the relative pitch to `-70`º using the `SetRelativeRotation` function so that the camera looks down at the player. The `FRotator` constructor’s parameters are `pitch`, `yaw`, and `roll`, respectively, as follows:

    ```cpp
    //The camera looks down at the player
    CameraBoom->SetRelativeRotation(FRotator(-70.f, 0.f, 0.f));
    ```

4.  Change `bUsePawnControlRotation` to `false` so that the camera’s rotation isn’t changed by the player’s movement input as follows:

    ```cpp
    // Don't rotate the arm based on the controller
    CameraBoom->bUsePawnControlRotation = false;
    ```

5.  Add a line that sets `bInheritPitch`, `bInheritYaw`, and `bInheritRoll` to `false` so that the camera’s rotation isn’t changed by the character’s orientation as follows:

    ```cpp
    // Ignore pawn's pitch, yaw and roll
    CameraBoom->bInheritPitch = false;
    CameraBoom->bInheritYaw = false;
    CameraBoom->bInheritRoll = false;
    ```

After we’ve made these modifications, we’re going to remove the character’s ability to jump (we don’t want the player to escape from the dodgeballs that easily) and to rotate the camera from the player’s rotation input.

1.  Go to the `SetupPlayerInputComponent` function in the `DodgeballCharacter`’s source file and remove the following lines of code in order to remove the ability to jump:

    ```cpp
    // REMOVE THESE LINES
    PlayerInputComponent->BindAction("Jump", IE_Pressed, this, 
      &ACharacter::Jump);
    PlayerInputComponent->BindAction("Jump", IE_Released, this, 
      Acharacter::StopJumping);
    ```

2.  Next, add the following lines in order to remove the player’s rotation input:

    ```cpp
    // REMOVE THESE LINES
    PlayerInputComponent->BindAxis("Turn", this, 
      &APawn::AddControllerYawInput);
    PlayerInputComponent->BindAxis("TurnRate", this, 
      &ADodgeballCharacter::TurnAtRate);
    PlayerInputComponent->BindAxis("LookUp", this, 
      &APawn::AddControllerPitchInput);
    PlayerInputComponent->BindAxis("LookUpRate", this, 
      &ADodgeballCharacter::LookUpAtRate);
    ```

This step is optional, but in order to keep your code clean, you should remove the declarations and implementations of the `TurnAtRate` and `LookUpAtRate` functions.

1.  After that, we’ll have to adapt this project to use the Enhanced Input system instead of the Legacy Input system. Go to this class’s header file and add a property for the Character’s Input Context as well as the Move Input Action, just like we did in steps 2 and 3 of *Exercise 4.02*. Add a declaration for the `Move` function, just like we did in step 14 of *Exercise 4.02*.
2.  Add the logic for adding the Character’s Input Context as well as binding the Move Input Action, just like we did in steps 4-10 of *Exercise 4.02*.
3.  Add the implementation to the `Move` function, just like we did in steps 14-18 of *Exercise 4.02*.
4.  Add the Enhanced Input dependency, just like we did in steps 19 and 20 of *Exercise 4.02*.
5.  Finally, after you’ve made these changes, run your project from Visual Studio.
6.  When the editor has loaded, go to `EnhancedPlayerInput` and the `EnhancedInputComponent`.
7.  After this, create the `IA_Move` Input Action asset and set it up just like we did in steps 1-3 of *Exercise 4.01*.
8.  Then, create the `IC_Character` Input Context asset and add a mapping for the `IA_Move` Input Action, just like we did in steps 4-14 of *Exercise 4.01*.
9.  The last thing you need to do to complete the Enhanced Input setup is to open the `IC_Character` and `IA_Move` properties, just like we did in step 22 of *Exercise 4.01*.
10.  After that, play the level. The camera’s perspective should look like the following and should not rotate based on the player’s input or the character’s rotation:

![Figure 5.3 – Locked camera rotation to a top-down perspective ](img/Figure_5.03_B18531.jpg)

Figure 5.3 – Locked camera rotation to a top-down perspective

That concludes the first exercise of this chapter and the first step of your new project, **Dodgeball**.

Next, you will be creating the `EnemyCharacter` class. This character will be the enemy that throws dodgeballs at the player while the player is in view. But the question that arises here is this: how will the enemy know whether it can see the player character or not?

That will be achieved with the power of **Line Traces** (also known as **Raycasts** or **Raytraces**), which you will be looking at in the following section.

# Understanding Line Traces

One of the most important features of any game development tool is its ability to execute Line Traces. These are available through the physics engine that the tool is using.

Line Traces are a way of asking the game to tell you whether anything stands between two points in the game world. The game will *shoot a ray* between those two points, specified by you, and return the objects that were hit (if any), where they were hit, at what angle, and much more.

In *Figure 5.4*, you can see a representation of a Line Trace where we assume object **1** is ignored and object **2** is detected,due to their Trace Channel properties (further explained in the following paragraphs):

![Figure 5.4 – A Line Trace being executed from point A to point B ](img/Figure_5.04_B18531.jpg)

Figure 5.4 – A Line Trace being executed from point A to point B

*Figure 5.4* is explained as follows:

*   The dashed line represents the Line Trace before it hits an object.
*   The arrows represent the direction of the Line Trace.
*   The dotted line represents the Line Trace after it hits an object.
*   The striped circle represents the Line Trace’s impact point.
*   The big squares represent two objects that are in the path of the Line Trace (objects **1** and **2**).

We notice that only object **2** was hit by the Line Trace and not object **1**, even though it is also in the path of the Line Trace. This is due to assumptions made about object **1**’s Trace Channel properties, which are discussed later in this chapter.

Line Traces are used for many game features, such as the following:

*   Checking whether a weapon hits an object when it fires
*   Highlighting an item that the player can interact with when the character looks at it
*   Rotating the camera around the player character automatically as it goes around corners

A common and important feature of Line Traces is **Trace Channels**. When you execute a Line Trace, you may want to check only specific types of objects, which is what Trace Channels are for. They allow you to specify filters to be used when executing a Line Trace so that it doesn’t get blocked by unwanted objects. Check the following examples:

*   You may want to execute a Line Trace only to check for objects that are visible. These objects would block the `Visibility` Trace Channel. For instance, invisible walls, which are invisible pieces of geometry used in games to block the player’s movement, would not be visible and therefore would not block the `Visibility` Trace Channel.
*   You may want to execute a Line Trace only to check for objects that can be interacted with. These objects would block the `Interaction` Trace Channel.
*   You may want to execute a Line Trace only to check for pawns that can move around the game world. These objects would block the `Pawn` Trace Channel.

You can specify how different objects react to different Trace Channels so that only some objects block specific Trace Channels and others ignore them. In our case, we want to know whether anything stands between the enemy and the player character so that we know whether the enemy can see the player. We will be using Line Traces for this purpose by checking for anything that blocks the enemy’s line of sight of the player character, using a `Tick` event.

In the next section, we will be creating the `EnemyCharacter` class using C++.

# Creating the EnemyCharacter C++ class

In our `EnemyCharacter` class will constantly be looking at the player character if they’re within view. This is the same class that will later throw dodgeballs at the player; however, we’ll leave that for the following chapter. In this chapter, we will be focusing on the logic that allows our enemy character to look at the player.

So, let’s get started as follows:

1.  Right-click on the **Content Browser** inside the editor and select **New C++ Class**.
2.  Choose the `Character` class as the parent class.
3.  Name the new class `EnemyCharacter`.

After you’ve created the class and opened its files in Visual Studio, let’s add the `LookAtActor` function declaration in its `header` file. This function should be `public` and not return anything, only receiving the `AActor* TargetActor` parameter, which will be the Actor it should be facing. Have a look at the following code snippet, which shows this function:

```cpp
// Change the rotation of the character to face the given 
// actor
void LookAtActor(AActor* TargetActor);
```

Note

Even though we only want the enemy to look at the player character, in order to execute good software development practices, we’re going to abstract this function a bit more and allow `EnemyCharacter` to look at any Actor. This is because the logic that allows an Actor to look at another Actor or the player character will be exactly the same.

Remember, you should not create unnecessary restrictions when writing code. If you can write similar code and simultaneously allow more possibilities, you should do so if that doesn’t overcomplicate the logic of your program.

Moving on ahead, if `EnemyCharacter` can’t see the `Target` `Actor`, it shouldn’t be looking at it. In order to check whether the enemy can see the Actor, it should be looking at the `LookAtActor` function that will call another function, which is the `CanSeeActor` function. This is what you’ll be doing in the following exercise.

## Exercise 5.02: Creating the CanSeeActor function that executes Line Traces

In this exercise, we will create the `CanSeeActor` function, which will return whether the enemy character can see the given Actor.

The following steps will help you complete this exercise:

1.  Create the declaration for a `public` `CanSeeActor` function in the header file of the `EnemyCharacter` class; it will return `bool` and receive a `const Actor* TargetActor` parameter, which is the Actor we want to look at. This function will be a `const` function, because it doesn’t change any of the class’s attributes, and the parameter will also be `const` because we won’t need to modify any of its properties, we’ll only need to access them, as follows:

    ```cpp
    // Can we see the given actor
    bool CanSeeActor(const AActor* TargetActor) const;
    ```

Now, let’s get to the fun part, which is executing the Line Trace.

In order to call functions related to line tracing, we’ll have to fetch the enemy’s current world with the `GetWorld` function. However, we haven’t included the `World` class in this file, so let’s do so in the following step.

Note

The `GetWorld` function is accessible to any Actor and will return the `World` object that the Actor belongs to. Remember, the world is necessary in order to execute the Line Trace.

1.  Open the `EnemyCharacter` source file and find the following code line:

    ```cpp
    #include "EnemyCharacter.h"
    ```

Add the following line right after the preceding line of code:

```cpp
#include "Engine/World.h"
```

1.  Next, create the implementation of the `CanSeeActor` function in the `EnemyCharacter` source file where you’ll start by checking whether our `TargetActor` is `nullptr`. If it is, then we return `false`, given that we have no valid Actor to check our sight o,f as follows:

    ```cpp
    bool AEnemyCharacter::CanSeeActor(const AActor * TargetActor) 
      const
    {
      if (TargetActor == nullptr)
      {
        return false;
      }
    }
    ```

Next, before we add our `Line Trace` function call, we need to set up some necessary parameters; we will be implementing these in the following steps.

1.  After the preceding `if` statement, create a variable to store all of the necessary data relative to the results of the Line Trace. Unreal Engine already has a built-in type for this called the `FHitResult` type, as follows:

    ```cpp
    // Store the results of the Line Trace
    FHitResult Hit;
    ```

This is the variable we will send to our Line Trace function, which will populate it with the relevant info of the executed Line Trace.

1.  Create two `FVector` variables for the `Start` and `End` locations of our Line Trace and set them to our enemy’s current location and our target’s current location, respectively, as follows:

    ```cpp
    // Where the Line Trace starts and ends
    FVector Start = GetActorLocation();
    FVector End = TargetActor->GetActorLocation();
    ```

2.  Next, set the Trace Channel we wish to compare against. In our case, we want to have a `Visibility` Trace Channel specifically designated to indicate whether an object blocks another object’s view. Luckily for us, such a Trace Channel already exists in UE5, as shown in the following code snippet:

    ```cpp
    // The trace channel we want to compare against
    ECollisionChannel Channel = ECollisionChannel::ECC_Visibility;
    ```

The `ECollisionChannel` `enum` represents all of the possible Trace Channels available to compare against. We will be using the `ECC_Visibility` value, which represents the `Visibility` Trace Channel.

1.  Now that we’ve set up all our necessary parameters, we can finally call the `LineTrace` function, `LineTraceSingleByChannel`, as follows:

    ```cpp
    // Execute the Line Trace
    GetWorld()->LineTraceSingleByChannel(Hit, Start, End, 
      Channel);
    ```

This function will consider the parameters we send it, execute the Line Trace, and return its results by modifying our `Hit` variable.

Before we continue, there are still a couple more things we need to consider.

If the Line Trace starts from within our enemy character, which is what will happen in our case, that means it’s very likely that the Line Trace will simply hit our enemy character immediately and just stop there because our character might block the `Visibility` Trace Channel. In order to fix that, we need to tell the Line Trace to ignore it.

1.  Use the built-in `FCollisionQueryParams` type, which allows us to give even more options to our Line Trace, as follows:

    ```cpp
    FCollisionQueryParams QueryParams;
    ```

2.  Now, update the `Line Trace` to ignore our enemy by adding itself to the list of Actors to ignore, as follows:

    ```cpp
    // Ignore the actor that's executing this Line Trace
    QueryParams.AddIgnoredActor(this);
    ```

We should also add our target to our list of Actors to ignore because we don’t want to know whether it blocks the `EnemySight` channel; we simply want to know whether something between the enemy and the player character blocks that channel.

1.  Add the `Target Actor` to the list of Actors to be ignored, as shown in the following code snippet:

    ```cpp
    // Ignore the target we're checking for
    QueryParams.AddIgnoredActor(TargetActor);
    ```

2.  Next, send our `FCollisionQueryParams` to the Line Trace by adding it as the last parameter of the `LineTraceSingleByChannel` function as follows:

    ```cpp
    // Execute the Line Trace
    GetWorld()->LineTraceSingleByChannel(Hit, Start, End, Channel, 
      QueryParams);
    ```

3.  Finalize our `CanSeeActor` function by returning whether the Line Trace hits anything or not. We can do that by accessing our `Hit` variable and checking whether there was a blocking hit, using the `bBlockingHit` property. If there was a blocking hit, that means we can’t see our `TargetActor`. This can be achieved with the following code snippet:

    ```cpp
    return !Hit.bBlockingHit;
    ```

Note

Although we won’t need any more information from the `Hit` result other than whether there was a blocking hit, the `Hit` variable can give us much more information on the Line Trace, such as the following:

Information on the Actor that was hit by the Line Trace (`nullptr` if no Actor was hit) by accessing the `Hit.GetActor()` function.

Information on the Actor component that was hit by the Line Trace (`nullptr` if no Actor component was hit) by accessing the `Hit.GetComponent()` function.

Information on the location of the hit by accessing the `Hit.Location` variable.

The distance of the hit can be found by accessing the `Hit.Distance` variable.

The angle at which the Line Trace hit the object, which can be found by accessing the `Hit.ImpactNormal` variable.

Finally, our `CanSeeActor` function is complete. We now know how to execute a Line Trace and we can use it for our enemy’s logic.

By completing this exercise, we have finished the `CanSeeActor` function; we can now get back to the `LookAtActor` function. However, there is something we should look at first, which is visualizing our Line Trace.

# Visualizing the Line Trace

When creating new logic that makes use of Line Traces, it is very useful to actually visualize the Line Trace while it’s being executed, which is something that the Line Trace function doesn’t allow you to do. In order to do that, we must use a set of helper debug functions that can draw objects dynamically at runtime, such as lines, cubes, spheres, and so on.

Let’s then add a visualization of our Line Trace. The first thing we must do in order to use the debug functions is to add the following `include` line of code under our last `include` line:

```cpp
#include "DrawDebugHelpers.h"
```

We will want to call the `DrawDebugLine` function in order to visualize the Line Trace, which needs the following inputs (that are very similar to the ones received by the Line Trace function):

*   The current `World`, which we will supply with the `GetWorld` function
*   The `Start` and `End` points of the line, which will be the same as the `LineTraceSingleByChannel` function
*   The desired color of the line in the game, which can be set to `Red`

Then, we can add the `DrawDebugLine` function call under our Line Trace function call, as shown in the following code snippet:

```cpp
// Execute the Line Trace
GetWorld()->LineTraceSingleByChannel(Hit, Start, End, Channel, 
  QueryParams);
// Show the Line Trace inside the game
DrawDebugLine(GetWorld(), Start, End, FColor::Red);
```

This will allow you to visualize the Line Trace as it is being executed, which is very useful.

Note

If you feel the need for it, you can also specify more of the visual Line Trace’s properties, such as its lifetime and thickness.

There are many `DrawDebug` functions available that will draw cubes, spheres, cones, donuts, and even custom meshes.

Now that we can both execute and visualize our Line Trace, let’s use the `CanSeeActor` function, which we created in the last exercise, inside the `LookAtActor` function.

## Exercise 5.03: Creating the LookAtActor function

In this exercise, we will be creating the definition of our `LookAtActor` function, which will change the enemy’s rotation so that it faces the given Actor.

The following steps will help you complete the exercise:

1.  Create the `LookAtActor` function definition in the `EnemyCharacter` source file.
2.  Start by checking whether our `TargetActor` is `nullptr` and returns nothing immediately if it is (because it’s not valid), as shown in the following code snippet:

    ```cpp
    void AEnemyCharacter::LookAtActor(AActor * TargetActor)
    {
      if (TargetActor == nullptr)
      {
        return;
      }
    }
    ```

3.  Next, we want to check whether we can see our `Target Actor`, using our `CanSeeActor` function:

    ```cpp
    if (CanSeeActor(TargetActor))
    {
    }
    ```

If this `if` statement is `true`, that means we can see the Actor, and we will set our rotation in such a way that we are facing that Actor. Luckily for us, there’s already a function within UE5 that allows us to do that: the `FindLookAtRotation` function. This function will receive as input two points in the level, point A (the `Start` point) and point B (the `End` point), and return the rotation that the object at the start point must have in order to face the object at the end point. Perform the following steps in order to use this function.

1.  Include `KismetMathLibrary` as shown in the following code snippet:

    ```cpp
    #include "Kismet/KismetMathLibrary.h"
    ```

2.  The `FindLookAtRotation` function must receive a `Start` and `End` points, which will be our enemy’s location and our Target Actor’s location, respectively, as follows:

    ```cpp
    FVector Start = GetActorLocation();
    FVector End = TargetActor->GetActorLocation();
    // Calculate the necessary rotation for the Start 
    // point to face the End point
    FRotator LookAtRotation = 
      UKismetMathLibrary::FindLookAtRotation(Start, End);
    ```

3.  Finally, set your enemy character’s rotation to the same value as our `LookAtRotation`, as follows:

    ```cpp
    //Set the enemy's rotation to that rotation
    SetActorRotation(LookAtRotation);
    ```

And that’s it for the `LookAtActor` function.

Now, the last step is to call the `LookAtActor` function inside the `Tick` event and send the player character as the `TargetActor`, as in the Actor that we want to look at.

1.  For us to fetch the character that is currently being controlled by the player, we use the `GameplayStatics` object. As with other UE5 objects, we must first include them as follows:

    ```cpp
    #include "Kismet/GameplayStatics.h"
    ```

2.  Next, head to your `Tick` function’s body and call the `GetPlayerCharacter` function from `GameplayStatics`, as follows:

    ```cpp
    // Fetch the character currently being controlled by 
    // the player
    ACharacter* PlayerCharacter = 
      UGameplayStatics::GetPlayerCharacter(this, 0);
    ```

This function receives the following as input:

*   A `World` context object, which is, essentially, an object that belongs to our current world and is used to let the function know which `World` object to access. This `World` context object can simply be the `this` pointer.
*   A player index, which, given that our game is supposed to be a single-player game, we can safely assume to be `0` (the first player).

1.  Next, call the `LookAtActor` function, sending the player character that we just fetched, as follows:

    ```cpp
    // Look at the player character every frame
    LookAtActor(PlayerCharacter);
    ```

2.  The last step of this exercise is to compile your changes in Visual Studio.

Now that you’ve completed this exercise, your `EnemyCharacter` class has all of the necessary logic to face the player character, if it’s within view, and we can start creating the `EnemyCharacter` Blueprint Class.

# Creating the EnemyCharacter Blueprint Class

Now that we have finished the logic for our `EnemyCharacter` C++ class, we must create our Blueprint Class that derives from it, as follows:

1.  Open our project in the editor.
2.  Go to the `Blueprints` folder inside the `ThirdPersonCPP` folder in the `Content Browser`.
3.  Right-click and select the option to create a new Blueprint Class.
4.  Expand the `EnemyCharacter` C++ class, and select it as the Parent Class.
5.  Name the Blueprint Class `BP_EnemyCharacter`.
6.  Open the Blueprint Class, select the `SKM_Quinn_Simple` and its `ABP_Quinn`.
7.  Change the `-90º` (on the *z-axis*) and its position on the *z-axis* to `-83` units.
8.  After you’ve set up the Blueprint Class, its mesh setup should look very similar to that of our **DodgeballCharacter** Blueprint Class.
9.  Drag an instance of the `BP_EnemyCharacter` class to your level in a location near an object that can block its line of sight, such as following location (the selected character is `EnemyCharacter`):

![Figure 5.5 – Dragging the BP_EnemyCharacter class into the level ](img/Figure_5.05_B18531.jpg)

Figure 5.5 – Dragging the BP_EnemyCharacter class into the level

Now we can finally play the game and verify that our enemy does look at our player character whenever it’s within view, as follows:

![Figure 5.6 – The enemy character with a clear view of the player using a Line Trace ](img/Figure_5.06_B18531.jpg)

Figure 5.6 – The enemy character with a clear view of the player using a Line Trace

1.  We can also see that the enemy stops seeing the player whenever it’s not within view, as shown in *Figure 5.7*:

![Figure 5.7 – The enemy losing sight of the player ](img/Figure_5.07_B18531.jpg)

Figure 5.7 – The enemy losing sight of the player

And that concludes our `EnemyCharacter`’s logic. In the following section, we will be looking at Sweep Traces.

# Sweep Traces

Before we continue with our project, it is important to know about a variant of the Line Trace, which is the **Sweep Trace**. Although we won’t be using these in our project, it is important to know about them and how to use them.

While the Line Trace basically *shoots a ray* between two points, the Sweep Trace will simulate *throwing an object* between two points in a straight line. The object that is being *thrown* is simulated (it doesn’t actually exist in the game) and can have various shapes. In the Sweep Trace, the `Hit` location will be the first point at which the virtual object (which we will call **shape**) hits another object if it were thrown from the start point to the end point. The shapes of the Sweep Trace can be either a box, a sphere, or a capsule.

The following is a representation of a Sweep Trace from point A to point B, where we assume that object `1` is ignored due to its Trace Channel properties, using a box shape:

![Figure 5.8 – Representation of a Sweep Trace ](img/Figure_5.08_B18531.jpg)

Figure 5.8 – Representation of a Sweep Trace

In *Figure 5.8*, we notice the following:

*   A Sweep Trace, using a box shape, being executed from point A to point B.
*   The dashed boxes represent the Sweep Trace before it hits an object.
*   The dotted boxes represent the Sweep Trace after it hits an object.
*   The striped circle represents the Sweep Trace’s impact point with object **2**, which is the point at which the Sweep Trace box shape’s surface and object **2**’s surface collide with each other.
*   The big squares represent two objects that are in the path of the Line Sweep Trace (objects **1** and **2**).
*   Object **1** is ignored in the Sweep Trace due to assumptions based on its Trace Channel properties.

Sweep Traces are more useful than regular Line Traces in a few situations. Let’s take the example of our enemy character, which can throw dodgeballs. If we wanted to add a way for the player to constantly visualize where the next dodgeball that the enemy throws will land, that could be better achieved with a Sweep Trace. We would do a Sweep Trace with the shape of our dodgeball (a sphere) toward our player, check the impact point, and show a sphere on that impact point, which would be visible to the player. If the Sweep Trace hits a wall or a corner somewhere, the player would know that if the enemy were to throw a dodgeball at that moment, that’s where it would hit first. You could use a simple Line Trace for the same purpose, but the setup would have to be rather complex in order to achieve the same quality of results, which is why Sweep Traces are a better solution in this case.

Let’s now take a quick look at how we can do a Sweep Trace in code.

## Exercise 5.04: Executing a Sweep Trace

In this exercise, we will implement a Sweep Trace in code. Although we won’t be using it for our project, by performing this exercise you will become familiar with such an operation.

Go to the end of the `CanSeeActor` function created in the preceding sections and perform the following steps:

1.  The function responsible for the Sweep Trace is `SweepSingleByChannel`, which is available within UE5 and requires the following parameters as inputs:

An `FHitResult` type to store the results of the sweep (we already have one of these, so there’s no need to create another variable of this type) as follows:

```cpp
// Store the results of the Line Trace
FHitResult Hit;
```

`Start` and `End` points of the sweep (we already have both of these, so there’s no need to create another variable of this type) as follows:

```cpp
// Where the Sweep Trace starts and ends
FVector Start = GetActorLocation();
FVector End = TargetActor->GetActorLocation();
```

1.  Use the intended rotation of the shape, which is in the form of an `FQuat` type (representing a quaternion). In this case, it’s set to a rotation of `0` on all axes, by accessing the `FQuat`’s `Identity` property as follows:

    ```cpp
    // Rotation of the shape used in the Sweep Trace
    FQuat Rotation = FQuat::Identity; 
    ```

2.  Now, use the intended Trace Channel to compare it against (we already have one of these, so there’s no need to create another variable of this type) as follows:

    ```cpp
    // The trace channel we want to compare against
    ECollisionChannel Channel = ECollisionChannel::ECC_Visibility;
    ```

3.  Finally, use the shape of a box for the Sweep Trace by calling the `FcollisionShape` `MakeBox` function and supplying it with the radius (on all three axes) of the box shape we want. This is shown in the following code snippet:

    ```cpp
    // Shape of the object used in the Sweep Trace
    FCollisionShape Shape = FCollisionShape::MakeBox(FVector(20.f, 
      20.f, 20.f));
    ```

4.  Next, call the `SweepSingleByChannel` function as follows:

    ```cpp
    GetWorld()->SweepSingleByChannel(Hit,
                                     Start,
                                     End,
                                     Rotation,
                                     Channel,
                                     Shape);
    ```

With these steps completed, we finish our exercise on Sweep Traces. Given that we won’t be using Sweep Traces in our project, you should comment out the `SweepSingleByChannel` function so that our `Hit` variable doesn’t get modified and lose the results from our Line Trace.

Now that we’ve concluded the section on Sweep Traces, let’s get back to our **Dodgeball** project and learn how to change an object’s response to a Trace Channel.

## Changing the Visibility Trace Response

In our current setup, every object that is visible blocks the `Visibility` Trace Channel; however, what if we wanted to change whether an object blocks that channel completely? In order to do this, we must change a component’s response to that channel. Have a look at the following example:

1.  We select the cube that we’ve been using to block the enemy’s sight in our level as shown in *Figure 5.9*:

![Figure 5.9 – Default spawn of the character ](img/Figure_5.09_B18531.jpg)

Figure 5.9 – Default spawn of the character

1.  Then, you go to the **Collision** section of this object’s **Details Panel** (its default place in the **Editor**’s interface) as follows:

![Figure 5.10 – Collision tab in the Details Panel in Unreal Engine ](img/Figure_5.10_B18531.jpg)

Figure 5.10 – Collision tab in the Details Panel in Unreal Engine

1.  Here, you’ll find several collision-related options. The one we want to pay attention to right now is the **Collision Presets** option. Its current value is **Default**; however, we want to change it according to our own preferences, so we will click on the drop-down box and change its value to **Custom**.
2.  Once you do this, you’ll notice a whole group of new options pop up as follows:

![Figure 5.11 – Collision Presets set to Custom ](img/Figure_5.11_B18531.jpg)

Figure 5.11 – Collision Presets set to Custom

This group of options allows you to specify how this object responds to Line Traces and object collision, and the type of collision object it is.

The option you should be paying attention to is **Visibility**. You’ll notice it’s set to **Block**, but you can also set it to **Overlap** and **Ignore**.

Right now, the cube is blocking the **Visibility** channel, which is why our enemy can’t see the character when it’s behind this cube. However, if we change the object’s response to the **Visibility** channel to either **Overlap** or **Ignore**, the object will no longer block Line Traces that check for **Visibility** (which is the case for the Line Trace you’ve just written in C++).

1.  Let’s change the cube’s response to the **Visibility** channel to **Ignore**, and then play the game. You’ll notice that the enemy is still looking toward the player character, even when it’s behind the cube:

![Figure 5.12 – The enemy character looking through an object at the player ](img/Figure_5.12_B18531.jpg)

Figure 5.12 – The enemy character looking through an object at the player

This is because the cube no longer blocks the **Visibility** channel, and so the Line Trace the enemy is executing no longer hits anything when trying to reach the player character.

Now that we’ve seen how we can change an object’s response to a specific Trace Channel, let’s change the cube’s response to the **Visibility** channel back to **Block**.

However, there’s one thing that’s worth mentioning. If we were to set the cube’s response to the **Visibility** channel to **Overlap** instead of **Ignore**, the result would be the same. But why is that, and what is the purpose of having these two responses? In order to explain that, we’ll look at Multi Line Traces.

## Multi Line Traces

When using the `CanSeeActor` function in *Exercise 5.02*, *Creating the CanSeeActor function that executes Line Traces*, you might have wondered to yourself about the name of the Line Trace function we used, `LineTraceSingleByChannel`, specifically about why it used the word *Single*. The reason for that is that you can also execute `LineTraceMultiByChannel`.

But how do these two Line Traces differ?

While the Single Line Trace will stop checking for objects that block it after it hits an object and tell us that was the object that it hit, the Multi Line Trace can check for any objects that are hit by the same Line Trace.

The Single Line Trace will do the following:

*   Ignore the objects that have their response set to either `Ignore` or `Overlap` on the Trace Channel being used by the Line Trace
*   Stop when it finds an object that has its response set to `Block`

However, instead of ignoring objects that have their response set to `Overlap`, the Multi Line Trace will add them as objects that were found during the Line Trace and only stop when it finds an object that blocks the desired Trace Channel (*or when it reaches the end point*). In the next figure, you’ll find an illustration of a Multi Line Trace being executed:

![Figure 5.13 – A Multi Line Trace being executed from point A to point B ](img/Figure_5.13_B18531.jpg)

Figure 5.13 – A Multi Line Trace being executed from point A to point B

In *Figure 5.13*, we notice the following:

*   The dashed line represents the Line Trace before it hits an object that blocks it.
*   The dotted line represents the Line Trace after it hits an object that blocks it.
*   The striped circles represent the Line Trace’s impact points, and only the last one of which is a blocking hit in this case.

The only difference between the `LineTraceSingleByChannel` and the `LineTraceMultiByChannel` functions, when it comes to their inputs, is that the latter must receive a `TArray<FHitResult>` input instead of a single `FHitResult`. All other inputs are the same.

Multi Line Traces are very useful when simulating the behavior of bullets with strong penetration that can go through several objects before stopping completely. Keep in mind that you can also do Multi Sweep Traces by calling the `SweepMultiByChannel` function.

Note

Another thing about the `LineTraceSingleByChannel` function that you might be wondering about is the `ByChannel` portion. This distinction has to do with using a Trace Channel, as opposed to the alternative, which is an Object Type. You can do a Line Trace that uses Object Types instead of Trace Channels by calling the `LineTraceSingleByObjectType` function, also available from the `World` object. Object Types are related to topics we will be covering in the following chapter, so we won’t be going into detail on this function just yet.

## The Camera Trace Channel

When changing our cube’s response to the `Visibility` Trace Channel, you may have noticed the other out-of-the-box Trace Channel: **Camera**.

This channel is used to specify whether an object blocks the line of sight between the camera’s spring arm and the character it’s associated with. In order to see this in action, we can drag an object to our level and place it in such a way that it will stay between the camera and our player character.

Have a look at the following example:

1.  We duplicate the `floor` object.

Note

You can easily duplicate an object in the level by holding the *Alt* key and dragging one of the *Move Tool*’s arrows in any direction.

![Figure 5.14 – Floor object being selected ](img/Figure_5.14_B18531.jpg)

Figure 5.14 – Floor object being selected

1.  Next, we change its **Transform** values as shown in the following figure:

![Figure 5.15 – Updating the Transform values ](img/Figure_5.15_B18531_B18531.jpg)

Figure 5.15 – Updating the Transform values

1.  Now when you play your game, you’ll notice that when the character goes under our duplicated `floor` object, you won’t lose sight of the player character, but; the spring arm will cause the camera to move down until you can see the character, as follows:

![Figure 5.16 – Changes in the camera angle ](img/Figure_5.16_B18531.jpg)

Figure 5.16 – Changes in the camera angle

1.  In order to see how the spring arm’s behavior differs when an object isn’t blocking the `Camera` Trace Channel, change our duplicated floor’s response to the `Camera` channel to `Ignore` and play the level again. What will happen is that when our character goes under the duplicated floor, we will lose sight of the character.

After you’ve done these steps, you can see that the `Camera` channel is used to specify whether an object will cause the spring arm to move the camera closer to the player when it intersects that object.

Now that we know how to use the existing Trace Channels, what if we wanted to create our own Trace Channels?

## Exercise 5.05: Creating a custom EnemySight Trace Channel

As we’ve discussed before, UE5 comes with two out-of-the-box Trace Channels: `Visibility` and `Camera`. The first one is a general-use channel that we can use to specify which objects block the line of sight of an object, whereas the second one allows us to specify whether an object blocks the line of sight between the camera’s spring arm and the character it’s associated with.

But how can we create our own Trace Channels? That’s what we’ll be looking into in this exercise. We will create a new `EnemySight` Trace Channel and use it to check whether the enemy can see the player character, instead of the built-in `Visibility` channel, as follows:

1.  Open **Project Settings** by pressing the **Edit** button at the top-left corner of the editor and go to the **Collision** section. There you’ll find the **Trace Channels** section. It’s currently empty because we haven’t yet created any of our own Trace Channels.
2.  Select the `EnemySight` and set its default response to `Block`, because we want most objects to do exactly that.
3.  After you’ve created the new Trace Channel, we must go back to our `EnemyCharacter` C++ class and change the trace we’re comparing against in our Line Trace, as follows:

    ```cpp
    // The trace channel we want to compare against
    ECollisionChannel Channel = ECollisionChannel::ECC_Visibility;
    ```

Given that we are no longer using the `Visibility` channel, we must reference our new channel. But how do we do that?

In your project’s directory, you’ll find the `Config` folder. This folder contains several `ini` files related to your project, such as `DefaultGame.ini`, `DefaultEditor.ini`, `DefaultEngine.ini`, and so on. Each of these contains several properties that will be initialized when the project is loaded. The properties are set by name-value pairs (`property=value`), and you can change their values as desired.

1.  When we created our `EnemySight` channel, the project’s `DefaultEngine.ini` file was updated with our new Trace Channel. Somewhere in that file, you’ll find the following line:

    ```cpp
    +DefaultChannelResponses=(Channel=ECC_GameTraceChannel1,
      DefaultResponse=ECR_Block,bTraceType=True,
      bStaticObject=False,
      Name="EnemySight")
    ```

Note

The preceding code line can be found highlighted at the following link: [https://packt.live/3eFpz5r](https://packt.live/3eFpz5r).

The preceding line says that there is a custom Trace Channel called `EnemySight` that has a default response of `Block` and, most importantly, is available in C++ using the `ECC_GameTraceChannel1` value of the collision `enum` we mentioned earlier, `ECollisionChannel`. This is the channel we’ll be referencing in the following code:

```cpp
// The trace channel we want to compare against
ECollisionChannel Channel = 
  ECollisionChannel::ECC_GameTraceChannel1;
```

1.  Verify that our enemy’s behavior remains the same after all of the changes we’ve made. This means that the enemy must still face the player character, as long as it’s within view of said enemy.

By completing this exercise, we now know how to make our own Trace Channels for any desired purpose.

Going back to our enemy character, there are still ways that we can improve its logic. Right now, when we fetch our enemy’s location as the start point of the Line Trace, that point is somewhere around the enemy’s hip, because that’s where the origin of the Actor is. However, that’s not usually where people’s eyes are, and it wouldn’t make much sense to have a humanoid character looking from its hip instead of its head.

So, let’s change that and have our enemy character check whether it sees the player character starting from its eyes, instead of its hip.

## Activity 5.01: Creating the SightSource property

In this activity, we will be improving our enemy’s logic to determine whether it should look at the player. Currently, the Line Trace that’s being done to determine that is being *shot* from around our character’s hips, (`0,0,0`) in our `BP_EnemyCharacter` blueprint. We want this to make a bit more sense, so we’ll make it so that the Line Trace starts somewhere close to our enemy’s eyes.

The following steps will help you complete the activity:

1.  Declare a new `SceneComponent` in our `EnemyCharacter` C++ class called `SightSource`. Make sure to declare this as a `UPROPERTY` with the `VisibleAnywhere`, `BlueprintReadOnly`, `Category = LookAt`, and `meta = (AllowPrivateAccess = “true”)` tags.
2.  Create this component in the `EnemyCharacter` constructor by using the `CreateDefaultSubobject` function, and attach it to `RootComponent`.
3.  Change the start location of the Line Trace in the `CanSeeActor` function to the `SightSource` component’s location, instead of the Actor’s location.
4.  Open the `BP_EnemyCharacter` Blueprint Class and change the `SightSource` component’s location to the location of the enemy’s head, `10, 0, 80`, as was done in the *Creating the EnemyCharacter Blueprint Class* section to the `SkeletalMeshComponent` property of `BP_EnemyCharacter`.

**Hint**: This can be achieved from the **Transform** tab in the **Editor panel** as shown in *Figure 5.17*.

![Figure 5.17 – Updating the SightSource component’s values ](img/Figure_5.17_B18531.jpg)

Figure 5.17 – Updating the SightSource component’s values

The following is the expected output:

![Figure 5.18 – The expected output showing the updated Line Trace from the hip to the eye ](img/Figure_5.18_B18531.jpg)

Figure 5.18 – The expected output showing the updated Line Trace from the hip to the eye

Note

The solution for this activity can be found on GitHub here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions).

By completing this activity, we have updated our `SightSource` property for our `EnemyCharacter`.

# Summary

By completing this chapter, you have added a new tool to your belt: Line Traces. You now know how to execute Line Traces and Sweep Traces, both Single and Multi, how to change an object’s response to a specific Trace Channel, and how to create your own Trace Channels.

You will quickly realize in the following chapters that these are essential skills when it comes to game development, and you will make good use of them on your future projects.

Now that we know how to use Line Traces, we’re ready for the next step, which is Object Collision. In the following chapter, you will learn how to set up collisions between objects and how to use collision events to create your own game logic. You will create the Dodgeball Actor, which will be affected by real-time physics simulation, the Wall Actors, which will block both the characters’ movements and the dodgeball, and the Actor responsible for ending the game when the player comes into contact with it.