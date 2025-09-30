# 14

# Spawning the Player Projectile

In the previous chapter, you made great progress with the enemy character’s AI by creating a behavior tree that would allow the enemy to randomly select points from the `BP_AIPoints` actor you created. This gave the `SuperSideScroller` game more life as you can now have multiple enemies moving around your game world. Additionally, you learned about the different tools available in **Unreal Engine 5** (**UE5**) that are used together to make AI of various degrees of complexity. These tools included the Navigation Mesh, behavior trees, and Blackboards.

Now that you have enemies running around your level, you need to allow the player to defeat these enemies with the player projectile you started to create at the end of the previous chapter. Our goal for this chapter is to use a custom `UAnimNotify` class that we will implement within our `Throw` Animation Montage to spawn the `Player Projectile`. Additionally, we will add polish elements to the projectile such as Particle Systems and Sound Cues.

In this chapter, we will cover the following topics:

*   How to use the `UAnimNotify` class to spawn the player projectile during the `Throw` Animation Montage.
*   Creating a new `Socket` for the main character skeleton from which the projectile will spawn.
*   Learn how to use Particle Systems and Soundcues to add a layer of visual and audio polish to the game.

By the end of this chapter, you will be able to play Animation Montages in both Blueprints and C++ and know how to spawn objects into the game world using C++ and the `UWorld` class. These elements of the game will be given audio and visual components as an added layer of polish, and your `SuperSideScroller` player character will be able to throw projectiles that destroy enemies.

Technical requirements

For this chapter, you will need the following technical requirements:

*   Unreal Engine 5 installed
*   Visual Studio 2019 installed
*   Unreal Engine 4.27 installed

The project for this chapter can be found in the `Chapter14` folder of the code bundle for this book, which can be downloaded here:

[https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition)

Let’s begin this chapter by learning about `UAnimNotify` class so that you can spawn the player projectile during the `Throw` Animation Montage.

# Anim Notifies and Anim Notify States

When it comes to creating polished and complex animations, there needs to be a way for animators and programmers to add custom events within the animation that will allow for additional effects, layers, and functionality to occur. The solution in UE5 is to use **Anim Notifies** and **Anim Notify States**.

The main difference between **Anim Notify** and **Anim Notify State** is that **Anim Notify State** possesses three distinct events that **Anim Notify** does not. These events are **Notify Begin**, **Notify End**, and **Notify Tick**, all of which can be used in Blueprints or C++. When it comes to these events, UE5 secures the following behaviors:

*   **Notify State** will always start with **Notify Begin Event**
*   **Notify State** will always finish with **Notify End Event**
*   **Notify Tick Event** will always take place between the **Notify Begin** and **Notify End** events

`Notify()`, to allow programmers to add functionality to the notify itself. It works with the mindset of *fire and forget*, meaning you don’t need to worry about what happens at the start, end, or anywhere in-between the `Notify()` event. It is due to this simplicity of `SuperSideScroller` game.

Before moving on to the following exercise, where you will create a custom **Anim Notify** in C++, let’s briefly discuss some examples of existing Anim Notifies that UE5 provides by default. A full list of default **Anim Notify** states can be seen in the following screenshot:

![Figure 14.1 – The full list of default Anim Notifies provided in UE5 ](img/Figure_14.01_B18531.jpg)

Figure 14.1 – The full list of default Anim Notifies provided in UE5

There are two **Anim Notifies** that you will be using later in this chapter: **Play Particle Effect** and **Play Sound**. Let’s discuss these two in more detail so that you are familiar with them by the time you use them:

*   **Play Particle Effect**: The **Play Particle Effect** notify, as its name suggests, allows you to spawn and play a Particle System at a certain frame of your animation. As shown in the following screenshot, you have options to change the **visual effects** (**VFX**) that are being used, such as updating the **Location Offset**, **Rotation Offset**, and **Scale** settings of the particle. You can even attach the particle to a specified **Socket Name** if you so choose:

![Figure 14.2 – The Details panel of the Play Particle Effect notify ](img/Figure_14.02_B18531.jpg)

Figure 14.2 – The Details panel of the Play Particle Effect notify

Note

Visual effects, or VFX for short, are crucial elements for any game. VFX, in UE5, are created using a tool called **Niagara** inside the editor. Niagara has been around since Unreal Engine 4 version 4.20, as a free plugin to improve the quality and pipeline for how VFX are made. **Cascade**, the previous VFX tool, will become deprecated in a later version of UE5\. You can learn more about Niagara here: [https://docs.unrealengine.com/en-US/Engine/Niagara/Overview/index.xhtml](https://docs.unrealengine.com/en-US/Engine/Niagara/Overview/index.xhtml).

A very common example that’s used in games is to use this type of notify to spawn dirt or other effects underneath the player’s feet while they walk or run. Having the ability to specify at which frame of the animation these effects spawn is very powerful and allows you to create convincing effects for your character.

*   **Play Sound**: The **Play Sound** notify allows you to play a **Soundcue** or **Soundwave** at a certain frame of your animation. As shown in the following screenshot, you have options for changing the sound being used, updating its **Volume Multiplier** and **Pitch Multiplier** values, and even having the sound follow the owner of the sound by attaching it to a specified **Attach Name**:

![Figure 14.3 – The Details panel of the Play Sound notify ](img/Figure_14.03_B18531.jpg)

Figure 14.3 – The Details panel of the Play Sound notify

Much like the example given for the **Play Particle Effect** notify, the **Play Sound** notify can also be commonly used to play the sounds of footsteps while the character is moving. By having control of exactly where on the animation timeline you can play a sound, it is possible to create believable sound effects.

Although you will not be using an **Anim Notify State**, it is still important to at least know the options that are available to you by default, as shown in the following screenshot:

![Figure 14.4 – The full list of default Anim Notify States provided to you in UE5 ](img/Figure_14.04_B18531.jpg)

Figure 14.4 – The full list of default Anim Notify States provided to you in UE5

Note

The two Notify states that are not available in Animation Sequences are the **Montage Notify Window** and **Disable Root Motion** states, as shown in the preceding screenshot. For more information regarding notifies, please refer to the following documentation: [docs.unrealengine.com/en-US/Engine/Animation/Sequences/Notifies/index.xhtml](http://docs.unrealengine.com/en-US/Engine/Animation/Sequences/Notifies/index.xhtml).

Now that you are more familiar with **Anim Notify** and **Anim Notify State**, let’s move on to the first exercise, where you will create a custom **Anim Notify** in C++ that you will use to spawn the player projectile.

## Exercise 14.01 – creating a UAnimNotify class

The main offensive ability that the player character will have in the `SuperSideScroller` game is the projectile that the player can throw at enemies. In the previous chapter, you set up the framework and base functionality of the projectile, but right now, there is no way for the player to use it. To make spawning, or throwing, the projectile convincing to the eye, you need to create a custom **Anim Notify** that you will then add to the **Throw** Animation Montage. This **Anim Notify** will let the player know it’s time to spawn the projectile.

Follow these steps to create the new `UAnimNotify` class:

1.  Inside UE5, navigate to the **Tools** option and *left-click* on the **New C++ Class** option.
2.  From the `Anim Notify` and *left-click* the **AnimNotify** option. Then, *left-click* the **Next** option to name the new class.
3.  Name this new class `Anim_ProjectileNotify`. Once it's been named, *left-click* the `Anim_ProjectileNotify.h`, and the source file, `Anim_ProjectileNotify.cpp`, available to you.
4.  The `UAnimNotify` base class has one function that needs to be implemented inside your class:

    ```cpp
    virtual void Notify(USkeletalMeshComponent* MeshComp, UAnimSequenceBase* Animation, const FAnimNotifyEventReference& EventReference); 
    ```

This function is called automatically when the notify is hit on the timeline it is being used in. By overriding this function, you will be able to add logic to the notify. This function also gives you access to both the `Skeletal Mesh` component of the owning notify and the Animation Sequence currently being played.

1.  Next, let’s add the override declaration of this function to the header file. In the `Anim_ProjectileNotify.h` header file, add the following code underneath `GENERATED_BODY()`:

    ```cpp
    public:   virtual void Notify(USkeletalMeshComponent* MeshComp, UAnimSequenceBase* Animation, const FAnimNotifyEventReference& EventReference) override;
    ```

Now that you’ve added the function to the header file, it is time to define the function inside the `Anim_ProjectileNotify` source file.

1.  Inside the `Anim_ProjectileNotify.cpp` source file, define the function and add a `UE_LOG()` call that prints the text `"Throw Notify"`, as shown in the following code:

    ```cpp
     void UAnim_ProjectileNotify::Notify(USkeletalMeshComponent* MeshComp, UAnimSequenceBase* Animation, const FAnimNotifyEventReference& EventReference)
    {
       Super::Notify(MeshComp, Animation, EventReference);
       UE_LOG(LogTemp, Warning, TEXT("Throw Notify"));
    }
    ```

For now, you will just use this `UE_LOG()` debugging tool to know that this function is being called correctly when you add this notify to the **Throw** Animation Montage in the next exercise.

In this exercise, you created the groundwork necessary to implement your own `Anim Notify` class by adding the following function:

```cpp
 Notify(USkeletalMeshComponent* MeshComp, UAnimSequenceBase* Animation, const FAnimNotifyEventReference& EventReference)
```

Inside this function, you are using `UE_LOG()` to print the custom text `"Throw Notify"` in the output log so that you know that this notify is working correctly.

Later in this chapter, you will update this function so that it calls logic that will spawn the player projectile, but first, let’s add the new notify to the **Throw** Animation montage.

## Exercise 14.02 – adding the new notify to the Throw Animation Montage

Now that you have your `Anim_ProjectileNotify` notify, it is time to add it to the **Throw** Animation Montage so that it can be of use to you.

In this exercise, you will add `Anim_ProjectileNotify` to the timeline of the **Throw** Animation Montage at the exact frame of the animation that you’d expect the projectile to spawn.

Follow these steps to achieve this:

1.  Back inside UE5, navigate to the `/MainCharacter/Animation/` directory. Inside this directory, *double-click* the `AM_Throw` asset to open the **Animation Montage** editor.

At the very bottom of the **Animation Montage** editor, you will find the timeline for the animation. By default, you will observe that the *red-colored bar* will be moving along the timeline as the animation plays.

1.  *Left-click* this red bar and manually move it to the 22nd frame, as close as you can, as shown in the following screenshot:

![Figure 14.5 – The red-colored bar allows you to manually position notifies anywhere on the timeline ](img/Figure_14.05_B18531.jpg)

Figure 14.5 – The red-colored bar allows you to manually position notifies anywhere on the timeline

The 22nd frame of the **Throw** animation is the exact moment in the throw that you would expect a projectile to spawn and be thrown by the player. The following screenshot shows the frame of the **Throw** animation, as seen inside the editor within **Persona**:

![Figure 14.6 – The exact moment the player projectile should spawn ](img/Figure_14.06_B18531.jpg)

Figure 14.6 – The exact moment the player projectile should spawn

1.  Now that you know the position on the timeline that the notify should be played, you can *right-click* on the thin red line within the **Notifies** timeline.

A popup will appear where you can add a **Notify** or a **Notify State**. In some cases, the **Notifies** timeline may be collapsed and hard to find; simply left-click on the word **Notifies** to toggle between collapsed and expanded.

1.  Select **Add Notify** and, from the options provided, find and select **Anim Projectile Notify**.
2.  After adding **Anim Projectile Notify** to the **Notifies** timeline, you will see the following:

![Figure 14.7 – Anim_ProjectileNotify successfully added to the Throw Animation Montage ](img/Figure_14.07_B18531.jpg)

Figure 14.7 – Anim_ProjectileNotify successfully added to the Throw Animation Montage

1.  With the `Anim_ProjectileNotify` notify in place on the **Throw** Animation Montage timeline, save the montage.
2.  If the **Output Log** window is not visible, please re-enable the window by navigating to the **Window** option and hovering over it to find the option for **Output Log**. Then, *left-click* to enable it.
3.  Now, use `PIE` and, once in-game, use the *left mouse button* to start playing the **Throw** montage.

At the point in the animation where you added the notify, you will now see the `Throw Notify` debugging log text appear in the output log.

As you may recall from [*Chapter 12*](B18531_12.xhtml#_idTextAnchor247), *Animation Blending and Montages*, you added the `Play Montage` function to the player character Blueprint – that is, `BP_SuperSideScroller_MainCharacter`. For the sake of learning C++ in the context of UE5, you will be moving this logic from Blueprint to C++ in the upcoming exercises. This is so that we don’t rely too heavily on Blueprint scripts for the base behavior of the player character.

With this exercise complete, you have successfully added your custom `Anim Notify` class, `Anim_ProjectileNotify`, to the `EnhancedInputAction` event, `ThrowProjectile`, is called when using the *left mouse button*. Before making the transition from playing the **Throw** Animation Montage in Blueprints to playing it in C++, let’s discuss playing Animation Montages some more.

# Playing Animation Montages

As you learned in [*Chapter 12*](B18531_12.xhtml#_idTextAnchor247), *Animation Blending and Montages*, these assets are useful for allowing animators to combine individual Animation Sequences into one complete montage. By splitting the montage into unique sections and adding notifies for particles and sound, animators and animation programmers can make complex sets of montages that handle all the different aspects of the animation.

But once the Animation Montage is ready, how do we play it on a character? You are already familiar with the first method, which is via Blueprints.

## Playing Animation Montages in Blueprints

In Blueprints, the **Play Montage** function can be used, as shown in the following screenshot:

![Figure 14.8 – The Play Montage function in Blueprints ](img/Figure_14.08_B18531.jpg)

Figure 14.8 – The Play Montage function in Blueprints

You have already used the `Play Montage` function to play the `AM_Throw` Animation Montage. This function requires the **Skeletal Mesh** component that the montage must be played on, and it requires the Animation Montage to play.

The remaining parameters are optional, depending on how your montage will work. Let’s have a quick look at these parameters:

*   **Play Rate**: The **Play Rate** parameter allows you to increase or decrease the playback speed of the Animation Montage. For faster playback, you would increase this value; otherwise, you would decrease it.
*   `1.0f` position instead of at `0.0f`.
*   **Starting Section**: The **Starting Section** parameter allows you to tell the Animation Montage to start at a specific section. Depending on how your montage is set up, you could have multiple sections created for different parts of the montage. For example, a shotgun weapon-reloading Animation Montage would include a section for the initial movement for reloading, a looped section for the actual bullet reload, and a final section for re-equipping the weapon so that it is ready to fire again.

When it comes to the outputs of the **Play Montage** function, you have a few different options:

*   **On Completed**: The **On Completed** output is called when the Animation Montage has finished playing and has been fully blended out.
*   **On Blend Out**: The **On Blend Out** output is called when the Animation Montage begins to blend out. This can occur during **Blend Out Trigger Time**, or if the montage ends prematurely.
*   **On Interrupted**: The **On Interrupted** output is called when the montage begins to blend out due to it being interrupted by another montage that is trying to play on the same skeleton.
*   **On Notify Begin** and **On Notify End**: Both the **On Notify Begin** and **On Notify End** outputs are called if you are using the **Montage Notify** option under the **Notifies** category in the Animation Montage. The name that’s given to **Montage Notify** is returned via the **Notify Name** parameter.

Now that we have a better understanding of the Blueprint implementation of the **Play Montage** function, let’s take a look at how to play animations in C++.

## Playing Animation Montages in C++

On the C++ side, there is only one thing you need to know about, and that is the `UAnimInstance::Montage_Play()` function. This function requires the Animation Montage to play, the play rate in which to play back the montage, a value of the `EMontagePlayReturnType` type, a `float` value for determining the start position to play the montage, and a `Boolean` value for determining whether playing this montage should stop or interrupt all montages.

Although you will not be changing the default parameter of `EMontagePlayReturnType`, which is `EMontagePlayReturnType::MontageLength`, it is still important to know the two values that exist for this enumerator:

*   `Montage` `Length`: The `Montage` `Length` value returns the length of the montage itself, in seconds.
*   `Duration`: The `Duration` value returns the play duration of the montage, which is equal to the length of the montage, divided by the play rate.

Note

For more details regarding the `UAnimMontage` class, please refer to the following documentation: [https://docs.unrealengine.com/en-US/API/Runtime/Engine/Animation/UAnimMontage/index.xhtml](https://docs.unrealengine.com/en-US/API/Runtime/Engine/Animation/UAnimMontage/index.xhtml).

You will learn more about the C++ implementation of playing an Animation Montage in the next exercise.

## Exercise 14.03 – playing the Throw animation in C++

Now that you have a better understanding of how to play Animation Montages in UE5, both via Blueprints and C++, it is time to migrate the logic for playing the **Throw** Animation Montage from Blueprints to C++. The reason behind this change is that the Blueprint logic was put into place as a placeholder method so that you could preview the **Throw** montage. This book is a more heavily focused C++ guide to game development, and as such, it is important to learn how to implement this logic in code.

Let’s begin by removing the logic from Blueprints, and then move on to recreating the logic in C++ inside the player character class.

Follow these steps to complete this exercise:

1.  Navigate to the player character Blueprint, `BP_SuperSideScroller_MainCharacter`, which can be found in the `/MainCharacter/Blueprints/` directory. *Double-click* this asset to open it.
2.  Inside this Blueprint, you will find the **EnhancedInputAction IA_Throw** event and the **Play Montage** function that you created to preview the **Throw** Animation Montage, as shown in the following screenshot. Delete this logic and then recompile and save the player character Blueprint:

![Figure 14.9 – You no longer need this placeholder logic inside the player character Blueprint ](img/Figure_14.09_B18531.jpg)

Figure 14.9 – You no longer need this placeholder logic inside the player character Blueprint

1.  Now, use `PIE` and attempt to throw with the player character by using the *left mouse button*. You will observe that the player character no longer plays the **Throw** Animation Montage. Let’s fix this by adding the required logic in C++.
2.  Open the header file for the player character in Visual Studio – that is, `SuperSideScroller_Player.h`.
3.  The first thing you need to do is create a new variable for the player character that will be used for the `Private` access modifier:

    ```cpp
    UPROPERTY(EditAnywhere)
    class UAnimMontage* ThrowMontage;
    ```

Now that you have a variable that will represent the `SuperSideScroller_Player.cpp` file.

1.  Before you can make the call to `UAnimInstance::Montage_Play()`, you need to add the following `include` directory to the existing list at the top of the source file to have access to this function:

    ```cpp
    #include "Animation/AnimInstance.h"
    ```

As we know from [*Chapter 9*](B18531_09.xhtml#_idTextAnchor183), *Adding Audio-Visual Elements*, the player character already has a function called `ThrowProjectile` that is called whenever the *left mouse button* is pressed. As a reminder, this is where the binding occurs in C++:

```cpp

//Bind the pressed action Throw to your ThrowProjectile function
EnhancedPlayerInput->BindAction(IA_Throw, ETriggerEvent::Triggered, this, &ASuperSideScroller_Player::ThrowProjectile);
```

1.  Update `ThrowProjectile` so that it plays `ThrowMontage`, which you set up earlier in this exercise. Add the following code to the `ThrowProjectile()` function. Then, we can discuss what is happening here:

    ```cpp
    void ASuperSideScroller_Player::ThrowProjectile()
    {
      if (ThrowMontage)
      {
        const bool bIsMontagePlaying = GetMesh()
        ->GetAnimInstance()->
          Montage_IsPlaying(ThrowMontage);
        if (!bIsMontagePlaying)
        {
          GetMesh()->GetAnimInstance()
          ->Montage_Play(ThrowMontage, 
            1.0f);
        }
        }    }
    ```

The first line is checking if `ThrowMontage` is valid; if we don’t have a valid Animation Montage assigned, there is no point in continuing the logic. It can also be dangerous to use a `NULL` object in further function calls as it could result in a crash. Next, we are declaring a new Boolean variable, called `bIsMontagePlaying`, that determines whether `ThrowMontage` is already playing on the player character’s skeletal mesh. This check is made because the **Throw** Animation Montage should not be played while it is already playing; this will cause the animation to break if the player repeatedly presses the *left mouse button*.

So long as the preceding conditions are met, it is safe to move on and play the Animation Montage.

1.  Inside the `If` statement, you are telling the player’s skeletal mesh to play `ThrowMontage` with a play rate of `1.0f`. This value is used so that the Animation Montage plays back at the speed it is intended to. Values larger than `1.0f` will make the montage play back faster, while values lower than `1.0f` will make the montage play back slower. The other parameters that you learned about, such as the start position or the `EMontagePlayReturnType` parameter, can be left at their defaults. Back inside the UE5 editor, perform a recompile of the code, as you have done in the past.
2.  After the code recompiles successfully, navigate back to the player character Blueprint, `BP_SuperSideScroller_MainCharacter`, which can be found in the `/MainCharacter/Blueprints/` directory. *Double-click* this asset to open it.
3.  In the `Throw Montage` parameter that you added.
4.  *Left-click* on the drop-down menu for the `Throw Montage` parameter to find the `AM_Throw` montage. *Left-click* again on the `AM_Throw` montage to select it for this parameter. Please refer to the following screenshot to see how the variable should be set up:

![Figure 14.10 – The Throw montage has been assigned the AM_Throw montage ](img/Figure_14.10_B18531.jpg)

Figure 14.10 – The Throw montage has been assigned the AM_Throw montage

1.  Recompile and save the player character Blueprint. Then, use `PIE` to spawn the player character and use the *left mouse button* to play `Throw Montage`. The following screenshot shows this in action:

![Figure 14.11 – The player character is now able to perform the Throw animation again ](img/Figure_14.11_B18531.jpg)

Figure 14.11 – The player character is now able to perform the Throw animation again

By completing this exercise, you have learned how to add an `Animation Montage` parameter to the player character, as well as how to play the montage in C++. In addition to playing the `Throw` input and causing the animation to break or not play entirely.

Note

Try setting the play rate of `Animation Montage` from `1.0f` to `2.0f` and recompile the code. Observe how increasing the play rate of the animation affects how the animation looks and feels for the player.

Before moving on to spawning the player projectile, let’s set up the `Socket` location in the player character’s **Skeleton** so that the projectile can spawn from the *player’s hand* during the **Throw** animation.

## Exercise 14.04 – creating the projectile spawn socket

To spawn the player projectile, you need to determine the **Transform** properties in which the projectile will spawn while primarily focusing on **Location** and **Rotation**, rather than **Scale**.

In this exercise, you will create a new **Socket** on the player character’s **Skeleton** that you can then reference in code to obtain the transform from which to spawn the projectile.

Let’s get started:

1.  Inside UE5, navigate to the `/MainCharacter/Mesh/` directory.
2.  In this directory, find the `MainCharacter_Skeleton.uasset`. *Double-click* to open this **Skeleton**.

To determine the best position for where the projectile should spawn, we need to add the **Throw** Animation Montage as the preview animation for the skeleton.

1.  In the `Preview Controller` parameter and select the **Use Specific Animation** option.
2.  Next, *left-click* on the drop-down menu to find and select the **AM_Throw** Animation Montage from the list of available animations.

Now, the player character’s **Skeleton** will start previewing the **Throw** Animation Montage, as shown in the following screenshot:

![Figure 14.12 – The player character previewing the Throw Animation Montage ](img/Figure_14.12_B18531.jpg)

Figure 14.12 – The player character previewing the Throw Animation Montage

As you may recall from *Exercise 14.02 – adding the notify to the Throw montage*, you added `Anim_ProjectileNotify` at the 22nd frame of the **Throw** animation.

1.  Using the timeline at the bottom of the **Skeleton** editor, move the red bar to as close to the 22nd frame as you can. Please refer to the following screenshot:

![Figure 14.13 – The same 22nd frame in which you added Anim_ProjectileNotify earlier ](img/Figure_14.13_B18531.jpg)

Figure 14.13 – The same 22nd frame in which you added Anim_ProjectileNotify earlier

At the 22nd frame of the **Throw** animation, the player character should look as follows:

![Figure 14.14 – The character’s hand in position to release a projectile ](img/Figure_14.14_B18531.jpg)

Figure 14.14 – The character’s hand in position to release a projectile

As shown in the preceding screenshot, at the 22nd frame of the **Throw** Animation Montage, the character’s hand is in position to release a projectile.

As you can see, the player character will be throwing the projectile from their *right hand*, so the new `Socket` should be attached to it. Let’s take a look at the skeletal hierarchy of the player character, as shown in the following screenshot:

![Figure 14.15 – The RightHand bone within the hierarchy of the player character’s skeleton ](img/Figure_14.15_B18531.jpg)

Figure 14.15 – The RightHand bone within the hierarchy of the player character’s skeleton

1.  From the skeletal hierarchy, find the **RightHand** bone. This can be found underneath the **RightShoulder** bone hierarchy structure.
2.  *Right-click* on the **RightHand** bone and *left-click* the **Add Socket** option from the list of options that appears. Name this socket **ProjectileSocket**.

Also, when adding a new `Socket`, the hierarchy of the entire **RightHand** will expand and the new socket will appear at the bottom.

1.  With `Socket` at the following location:

    ```cpp
    Location = (X=30.145807,Y=36.805481,Z=-10.23186)
    ```

The final result should look as follows:

![Figure 14.16 – The final position of ProjectileSocket at the 22nd frame of the Throw animation in world space ](img/Figure_14.16_B18531.jpg)

Figure 14.16 – The final position of ProjectileSocket at the 22nd frame of the Throw animation in world space

If your gizmo looks a bit different, that is because the preceding screenshot shows the socket location in world space, not local space.

1.  Now that `MainCharacter_Skeleton` asset.

With this exercise complete, you now know the location that the player projectile will spawn from. Since you used the `Anim_ProjectileNotify` will fire.

Now, let’s spawn the player projectile in C++.

## Exercise 14.05 – preparing the SpawnProjectile() Function

Now that you have **ProjectileSocket** in place and there is a location from which to spawn the player projectile, let’s add the code necessary to spawn the player projectile.

By the end of this exercise, you will have the function ready to spawn the projectile and it will be ready to call from the `Anim_ProjectileNotify` class.

Follow these steps:

1.  From Visual Studio, navigate to the `SuperSideScroller_Player.h` header file.
2.  You need a class reference variable for the `PlayerProjectile` class. You can do this using the `TsubclassOf` variable template class type. Add the following code to the header file, under the `Private` access modifier:

    ```cpp
    UPROPERTY(EditAnywhere)
    TSubclassOf<class APlayerProjectile> PlayerProjectile;
    ```

Now that you have the variable ready, it is time to declare the function you will use to spawn the projectile.

1.  Add the following function declaration under the declaration of the void `ThrowProjectile()` function and the `Public` access modifier:

    ```cpp
    void SpawnProjectile();
    ```

2.  Before preparing the definition of the `SpawnProjectile()` function, add the following `include` directories to the list of includes in the `SuperSideScroller_Player.cpp` source file:

    ```cpp
    #include "PlayerProjectile.h"
    #include "Engine/World.h"
    #include "Components/SphereComponent.h"
    ```

You need to include `PlayerProjectile.h` because it is required to reference the collision component of the projectile class. Next, you must use the `Engine/World.h` include to use the `SpawnActor()` function and access the `FActorSpawnParameters` struct. Lastly, you need to use the `Components/SphereComponent.h` include to update the collision component of the player projectile so that it will ignore the player.

1.  Next, create the definition of the `SpawnProjectile()` function at the bottom of the `SuperSideScroller_Player.cpp` source file, as shown here:

    ```cpp
    void ASuperSideScroller_Player::SpawnProjectile()
    {
    }
    ```

The first thing this function needs to do is check whether the `PlayerProjectile` class variable is valid. If this object is not valid, there is no point in continuing to try and spawn it.

1.  Update the `SpawnProjectile()` function so that it looks as follows:

    ```cpp
    void ASuperSideScroller_Player::SpawnProjectile()
    {
      if(PlayerProjectile)
        {
        }
    }
    ```

Now, if the `PlayerProjectile` object is valid, you’ll want to obtain the `UWorld` object that the player currently exists in and ensure that this world is valid before continuing.

1.  Update the `SpawnProjectile()` function to the following:

    ```cpp
    void ASuperSideScroller_Player::SpawnProjectile()
    {
      if(PlayerProjectile)
        {
          UWorld* World = GetWorld();
          if (World)
            {
            }
        }
    }
    ```

At this point, you have made safety checks to ensure that both `PlayerProjectile` and `UWorld` are valid, so now, it is safe to attempt to spawn the projectile. The first thing you must do is declare a new variable of the `FactorSpawnParameters` type and assign the player as the owner.

1.  Add the following code within the most recent `if` statement so that the `SpawnProjectile()` function looks like this:

    ```cpp
    void ASuperSideScroller_Player::SpawnProjectile()
    {
      if(PlayerProjectile)
        {
          UWorld* World = GetWorld();
          if (World)
            {
              FActorSpawnParameters SpawnParams;
              SpawnParams.Owner = this;
            }
        }
    }
    ```

As you learned previously, the `SpawnActor()` function call from the `UWorld` object will require the `FActorSpawnParameters` struct as part of the spawned object’s initialization. In the case of the player projectile, you can use the `this` keyword as a reference to the player character class for the owner of the projectile.

1.  Next, you need to handle the `Location` and `Rotation` parameters of the `SpawnActor()` function. Add the following lines under the latest line – that is, `SpawnParams.Owner = this`:

    ```cpp
    const FVector SpawnLocation = this->GetMesh()-
      >GetSocketLocation(FName("ProjectileSocket"));
    const FRotator Rotation = GetActorForwardVector().Rotation();
    ```

In the first line, you declare a new `FVector` variable called `SpawnLocation`. This vector uses the `Socket` location of the `ProjectileSocket` socket that you created in the previous exercise. The `Skeletal Mesh` component returned from the `GetMesh()` function contains a function called `GetSocketLocation()` that will return the location of the socket with the `FName` property that is passed in – in this case, `ProjectileSocket`.

In the second line, you are declaring a new `FRotator` variable called `Rotation`. This value is set to the player’s forward vector and converted into a `Rotator` container. This will ensure that the rotation – or in other words, the direction in which the player projectile will spawn – will be in front of the player, and it will move away from the player.

Now, all of the parameters required to spawn the projectile are ready.

1.  Add the following line underneath the code from the previous step:

    ```cpp
    APlayerProjectile* Projectile = World-
      >SpawnActor<APlayerProjectile>(PlayerProjectile, 
      SpawnLocation, 
      Rotation, SpawnParams);
    ```

The `World->SpawnActor()` function will return an object of the class you are attempting to spawn in – in this case, `APlayerProjectile`. This is why you are adding `APlayerProjectile* Projectile` before the actual spawning occurs. Then, you are passing in the `SpawnLocation`, `Rotation`, and `SpawnParams` parameters to ensure that the projectile is spawning where and how you want.

1.  Return to the editor to recompile the newly added code. After the code compiles successfully, this exercise is complete.

With this exercise complete, you now have a function that will spawn the player projectile class that is assigned inside the player character. By adding safety checks for the validity of both the projectile and the world, you can ensure that if an object is spawned, it is a valid object inside a valid world.

You set up the appropriate `location`, `rotation`, and `FActorSpawnParameters` parameters for the `UWorld SpawnActor()` function to ensure that the player projectile spawns at the right location, based on the socket location from the previous exercise, with the appropriate direction so that it moves away from the player, and with the player character as its owner.

Now, it is time to update the `Anim_ProjectileNotify` source file so that it spawns the projectile.

## Exercise 14.06 – updating the Anim_ProjectileNotify class

The function that allows the player projectile to spawn is ready, but you aren’t calling this function anywhere yet. Back in *Exercise 14.01 – creating a UAnim Notify class*, you created the `Anim_ProjectileNotify` class, while in *Exercise 14.02 – adding the notify to the Throw montage*, you added this notify to the **Throw** Animation Montage.

Now, it is time to update the `UanimNotify` class so that it calls the `SpawnProjectile()` function.

Follow these steps:

1.  In Visual Studio, open the `Anim_ProjectileNotify.cpp` source file.

In the source file, you have the following code:

```cpp
 #include "Anim_ProjectileNotify.h"
void UAnim_ProjectileNotify::Notify(USkeletalMeshComponent* MeshComp, UAnimSequenceBase* Animation, const FAnimNotifyEventReference& EventReference)
{
   Super::Notify(MeshComp, Animation, EventReference);
   UE_LOG(LogTemp, Warning, TEXT("Throw Notify"));
}
```

1.  Remove the `UE_LOG()` line from the `Notify()` function.
2.  Next, add the following `include` lines underneath `Anim_ProjectileNotify.h`:

    ```cpp
    #include "Components/SkeletalMeshComponent.h"
    #include "SuperSideScroller/SuperSideScroller_Player.h"
    ```

You need to include the `SuperSideScroller_Player.h` header file because it is required to call the `SpawnProjectile()` function you created in the previous exercise. We also included `SkeletalMeshComponent.h` because we will reference this component inside the `Notify()` function, so it’s best to include it here too.

The `Notify()` function passes in a reference to the owning `Skeletal Mesh`, labeled `MeshComp`. You can use this skeletal mesh to get a reference to the player character by using the `GetOwner()` function and casting the returned actor to your `SuperSideScroller_Player` class. We’ll do this next.

1.  Inside the `Notify()` function, add the following line of code:

    ```cpp
    ASuperSideScroller_Player* Player = 
      Cast<ASuperSideScroller_Player>(
      MeshComp->GetOwner());
    ```

2.  Now that you have a reference to the player, you need to add a validity check for the `Player` variable before making a call to the `SpawnProjectile()` function. Add the following lines of code after the line from the previous step:

    ```cpp
    if (Player)
    {
      Player->SpawnProjectile();
    }
    ```

3.  Now that the `SpawnProjectile()` function is being called from the `Notify()` function, return to the editor to recompile and hot-reload the code changes you have made.

Before you can use `PIE` to run around and throw the player projectile, you need to assign the `Player Projectile` variable from the previous exercise.

1.  Inside the `/MainCharacter/Blueprints` directory to find the `BP_SuperSideScroller_MainCharacter` Blueprint. *Double-click* to open the Blueprint.
2.  In the `Throw Montage` parameter, you will find the `Player Projectile` parameter. *Left-click* the drop-down option for this parameter and find `BP_PlayerProjectile`. *Left-click* on this option to assign it to the `Player Projectile` variable.
3.  Recompile and save the `BP_SuperSideScroller_MainCharacter` Blueprint.
4.  Now, use `PIE` and use the *left mouse button*. The player character will play the **Throw** animation and the player projectile will spawn.

Notice that the projectile is spawned from the `ProjectileSocket` function you created and that it moves away from the player. The following screenshot shows this in action:

![Figure 14.17 – The player can now throw the player projectile ](img/Figure_14.17_B18531.jpg)

Figure 14.17 – The player can now throw the player projectile

With this exercise complete, the player can now throw the player projectile. The player projectile, in its current state, is ineffective against enemies and just flies through the air. It took a lot of moving parts between the `Anim_ProjectileNotify` class, and the player character to get the player to throw the projectile.

In the upcoming section and exercises, you will update the player projectile so that it destroys enemies and play additional effects such as particles and sound.

# Destroying actors

So far in this chapter, we have put a lot of focus on spawning, or creating, actors inside the game world; the player character uses the `UWorld` class to spawn the projectile. UE5 and its base `Actor` class come with a default function that you can use to destroy, or remove, an actor from the game world:

```cpp
bool AActor::Destroy( bool bNetForce, bool bShouldModifyLevel )
```

You can find the full implementation of this function in Visual Studio by finding the `Actor.cpp` source file in the `/Source/Runtime/Engine/Actor.cpp` directory. This function exists in all the classes that extend from the `Actor` class, and in the case of UE5, it exists in all classes that can be spawned, or placed, inside the game world. To be more explicit, both the `EnemyBase` and `PlayerProjectile` classes are *children* of the `Actor` class, so they can be destroyed.

Looking further into the `AActor::Destroy()` function, you will find the following line:

```cpp
World->DestroyActor( this, bNetForce, bShouldModifyLevel );
```

We won’t be going into further detail about what exactly the `UWorld` class does to destroy an actor, but it is important to emphasize the fact that the `UWorld` class is responsible for both creating and destroying actors inside the world. Feel free to dig deeper into the source engine code to find more information about how the `UWorld` class handles destroying and spawning actors.

Now that you have more context regarding how UE5 handles destroying and removing actors from the game world, we’ll implement this ourselves for the enemy character.

## Exercise 14.07 – creating the DestroyEnemy() function

The main part of the gameplay for the `Super` `SideScroller` game is for the player to move around the level and use the projectile to destroy enemies. At this point in the project, you have handled the player movement and spawning the player projectile. However, the projectile does not destroy enemies yet.

To get this functionality in place, we’ll start by adding some logic to the `EnemyBase` class so that it knows how to handle its destruction and remove it from the game once it collides with the player projectile.

Follow these steps to achieve this:

1.  First, navigate to Visual Studio and open the `EnemyBase.h` header file.
2.  In the header file, create the declaration of a new function called `DestroyEnemy()` under the `Public` access modifier, as shown here:

    ```cpp
    public:
      void DestroyEnemy();
    ```

Make sure this function definition is written underneath `GENERATED_BODY()`, within the class definition.

1.  Save these changes to the header file and open the `EnemyBase.cpp` source file to add the implementation of this function.
2.  Below the `#include` lines, add the following function definition:

    ```cpp
    void AEnemyBase::DestroyEnemy()
    {
    }
    ```

For now, this function will be very simple. All you need to do is call the inherited `Destroy()` function from the base `Actor` class.

1.  Update the `DestroyEnemy()` function so that it looks like this:

    ```cpp
    void AEnemyBase::DestroyEnemy()
    {
      Destroy();
    }
    ```

2.  With this function complete, save the source file and return to the editor so that you can recompile and hot-reload the code.

With this exercise complete, the enemy character now has a function that can easily handle the destruction of the actor whenever you choose. The `DestroyEnemy()` function is publicly accessible so that it can be called by other classes, which will come in handy later when you handle the destruction of the player projectile.

The reason you’re creating a unique function to destroy the enemy actor is that you will use this function later in this chapter to add VFX and SFX to the enemy when they are destroyed by the player projectile.

Before polishing the elements of the enemy’s destruction, let’s implement a similar function inside the player projectile class so that it can also be destroyed.

## Exercise 14.08 – destroying projectiles

Now that the enemy characters can handle being destroyed through the new `DestroyEnemy()` function you implemented in the previous exercise, it is time to do the same for the player projectile.

By the end of this exercise, the player projectile will have a unique function to handle it being destroyed and removed from the game world.

Let’s get started:

1.  In Visual Studio, open the header file for the player projectile – that is, `PlayerProjectile.h`.
2.  Under the `Public` access modifier, add the following function declaration:

    ```cpp
    void ExplodeProjectile();
    ```

3.  Next, open the source file for the player projectile – that is, `PlayerProjectile.cpp`.
4.  Underneath the void `APlayerProjectile::OnHit` function, add the definition of the `ExplodeProjectile()` function:

    ```cpp
    void APlayerProjectile::ExplodeProjectile()
    {
    }
    ```

For now, this function will work identically to the `DestroyEnemy()` function from the previous exercise.

1.  Add the inherited `Destroy()` function to the new `ExplodeProjectile()` function, like so:

    ```cpp
    void APlayerProjectile::ExplodeProjectile()
    {
      Destroy();
    }
    ```

2.  With this function complete, save the source file and return to the editor so that you can recompile and hot-reload the code.

With this exercise complete, the player projectile now has a function that can easily handle the destruction of the actor whenever you choose. The reason you need to create a unique function to handle destroying the player projectile actor is the same reason you created the `DestroyEnemy()` function – you will use this function later in this chapter to add VFX and SFX to the player projectile when it collides with another actor.

Now that you have experience with implementing the `Destroy()` function inside both the player projectile and the enemy character, it is time to put these two elements together.

In the next activity, you will enable the player projectile to destroy the enemy character when they collide.

## Activity 14.01 – Allow the projectile to destroy enemies

Now that both the player projectile and the enemy character can handle being destroyed, it is time to go the extra mile and allow the player projectile to destroy the enemy character when they collide.

Follow these steps to achieve this:

1.  Add the `#include` statement for the `EnemyBase.h` header file toward the top of the `PlayerProjectile.cpp` source file.
2.  Within the void `APlayerProjectile::OnHit()` function, create a new variable of the `AEnemyBase*` type and call this variable `Enemy`.
3.  Cast the `OtherActor` parameter of the `APlayerProjectile::OnHit()` function to the `AEnemyBase*` class and set the `Enemy` variable to the result of this cast.
4.  Use an `if()` statement to check the validity of the `Enemy` variable.
5.  If the `Enemy` variable is valid, call the `DestroyEnemy()` function from this `Enemy`.
6.  After the `if()` block, make a call to the `ExplodeProjectile()` function.
7.  Save the changes to the source file and return to the UE5 editor.
8.  Use `PIE` and then use the player projectile against an enemy to observe the results.

The expected output is as follows:

![Figure 14.18 – The player throwing the projectile ](img/Figure_14.18_B18531.jpg)

Figure 14.18 – The player throwing the projectile

When the projectile hits the enemy, the enemy character is destroyed, as shown here:

![Figure 14.19 – The projectile and enemy have been destroyed ](img/Figure_14.19_B18531.jpg)

Figure 14.19 – The projectile and enemy have been destroyed

With this activity complete, the player projectile and the enemy character can be destroyed when they collide with each other. Additionally, the player projectile will be destroyed whenever another actor triggers its `APlayerProjectile::OnHit()` function.

With that, a major element of the `Super` `SideScroller` game has been completed: the player projectile spawning and the enemies being destroyed when they collide with the projectile. You can observe that destroying these actors is very simple and not very interesting to the player.

This is why, in the upcoming exercises in this chapter, you will learn more about **visual effects** and **audio effects**, or **VFX** and **SFX**, respectively. You will also implement these elements for the enemy character and player projectile.

Now that both the enemy character and the player projectile can be destroyed, let’s briefly discuss what VFX and SFX are, and how they will impact the project.

Note

The solution to this activity can be found on GitHub here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions).

# Understanding and implementing visual and audio effects

VFX such as Particle Systems and sound effects such as sound cues play an important role in video games. They add a level of polish on top of systems, game mechanics, and even basic actions that make these elements more interesting or more pleasing to perform.

Let’s start by understanding VFX, followed by SFX.

## VFX

VFX, in the context of UE5, are made up of what’s called **particle systems**. Particle Systems are made up of emitters, and emitters consist of modules. In these modules, you can control the appearance and behavior of the emitter using materials, meshes, and mathematical modules. The result can be anything from a fire torch or snow falling to rain, dust, and so on.

Note

You can learn more here: [https://docs.unrealengine.com/en-US/Resources/Showcases/Effects/index.xhtml](https://docs.unrealengine.com/en-US/Resources/Showcases/Effects/index.xhtml).

## Audio effects (SFX)

SFX, in the context of UE5, are made up of a combination of sound waves and sound cues:

*   Sound waves are `.wav` audio format files that can be imported into UE5.
*   Sound cues combine sound wave audio files with other nodes such as **Oscillator**, **Modulator**, and **Concatenator** to create unique and complex sounds for your game.

Note

You can learn more here: [https://docs.unrealengine.com/en-US/Engine/Audio/SoundCues/NodeReference/index.xhtml](https://docs.unrealengine.com/en-US/Engine/Audio/SoundCues/NodeReference/index.xhtml).

In the context of UE5, VFX were created using a tool called **Cascade**, where artists could combine the use of **materials**, **static meshes**, and **math** to create interesting and convincing effects for the game world. This book will not dive into how this tool works, but you can find information about Cascade here: [https://docs.unrealengine.com/4.27/en-US/RenderingAndGraphics/ParticleSystems/](https://docs.unrealengine.com/4.27/en-US/RenderingAndGraphics/ParticleSystems/).

In more recent versions of the engine, starting in the 4.20 update, there is a plugin called **Niagara** that can be enabled to create VFX. Niagara, unlike Cascade, uses a system similar to Blueprints, where you can visually script the behaviors of the effect rather than use preset modules with predefined behavior. You can find more information about Niagara here: [https://docs.unrealengine.com/en-US/Engine/Niagara/Overview/index.xhtml](https://docs.unrealengine.com/en-US/Engine/Niagara/Overview/index.xhtml). Furthermore, Cascade will be deprecated in new versions of UE5 and Niagara will be used. For the sake of this book, we will still use Cascade particle effects.

In [*Chapter 9*](B18531_09.xhtml#_idTextAnchor183), *Adding* *Audio-Visual Elements*, you learned more about audio and how audio is handled inside UE5\. All you need to know right now is that UE5 uses the `.wav` file format to import audio into the engine. From there, you can use the `.wav` file directly, referred to as sound waves in the editor, or you can convert these assets into sound cues, which allow you to add audio effects on top of the sound wave.

Lastly, there is one important class to know about that you will be referencing in the upcoming exercises, and this class is called `UGameplayStatics`. This is a static class in UE5 that can be used from both C++ and Blueprints, and it offers a variety of useful gameplay-related functions. The two functions you will be working with in the upcoming exercise are as follows:

```cpp
UGameplayStatics::SpawnEmitterAtLocation
UGameplayStatics:SpawnSoundAtLocation
```

These two functions work in very similar ways; they both require a `World` context object in which to spawn the effect, the Particle System or audio to spawn, and the location in which to spawn the effect. You will be using these functions to spawn the destroy effects for the enemy in the next exercise.

## Exercise 14.09 – adding effects when the enemy is destroyed

In this exercise, you will add new content to the project that comes included with this chapter and exercise. This includes theVFX andSFX, and all of their required assets. Then, you will update the `EnemyBase` class so that it can use audio and Particle System parameters to add the layer of polish needed when the enemy is destroyed by the player projectile.

By the end of this exercise, you will have an enemy that is visually and audibly destroyed when it collides with the player projectile.

Let’s get started:

1.  To begin, we need to migrate specific assets from the **Action RPG** project, which can be found in the **Learn** tab of **Unreal Engine Launcher**.
2.  From **Epic Games Launcher**, navigate to the **Samples** tab and, in the **UE Legacy Samples** category, you will find **Action RPG**:

![Figure 14.20 – The Action RPG sample project ](img/Figure_14.20_B18531.jpg)

Figure 14.20 – The Action RPG sample project

Note

You will be taking additional assets from the **Action RPG** project in later exercises of this chapter, so you should keep this project open to avoid redundantly opening the project. The assets for this exercise can be downloaded from [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Chapter14/Exercise14.09](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Chapter14/Exercise14.09).

1.  Left-click the **Action RPG** game project and then left-click the **Create Project** option.
2.  From here, select engine version 4.27 and choose which directory to download the project to. Then, *left-click* the **Create** button to start installing the project.
3.  Once the **Action RPG** project has finished downloading, navigate to the **Library** tab of **Epic Games Launcher** to find **ActionRPG** under the **My Projects** section.
4.  *Double-click* the **ActionRPG** project to open it in the UE5 editor.
5.  In the editor, find the **A_Guardian_Death_Cue** audio asset in the **Content Browser** interface. *Right-click* this asset and select **Asset Actions** and then **Migrate**.
6.  After selecting **Migrate**, you will be presented with all the assets that are referenced in **A_Guardian_Death_Cue**. This includes all audio classes and sound wave files. Choose **OK** from the **Asset Report** dialog window.
7.  Next, you will need to navigate to the `Content` folder for your `Super SideScroller` project and *left-click* **Select Folder**.
8.  Once the migration process is complete, you will see a notification in the editor stating that the migration was completed successfully.
9.  Do the same migration steps for the `P_Goblin_Death` VFX asset. The two primary assets you will be adding to the project are as follows:

    ```cpp
    A_Guardian_Death_Cue
    P_Goblin_Death
    ```

The `P_Goblin_Death` Particle System asset references additional assets such as materials and textures that are included in the `Effects` directory, while `A_Guardian_Death_Cue` references additional sound wave assets included in the `Assets` directory.

1.  After migrating these folders into your `Content` directory, open the UE5 editor for your `SuperSideScroller` project to find the new folders included in your project’s **Content Drawer**.

The particle you will be using for the enemy character’s destruction is called `P_Goblin_Death` and can be found in the `/Effects/FX_Particle/` directory. The sound you will be using for the enemy character’s destruction is called `A_Guardian_Death_Cue` and can be found in the `/Assets/Sounds/Creatures/Guardian/` directory. Now that the assets you need have been imported into the editor, let’s move on to the code.

1.  Open Visual Studio and navigate to the header file for the enemy base class – that is, `EnemyBase.h`.
2.  Add the following `UPROPERTY()` variable. This will represent the Particle System for when the enemy is destroyed. Make sure this is declared under the `Public` access modifier:

    ```cpp
    UPROPERTY(EditAnywhere, BlueprintReadOnly)
    class UParticleSystem* DeathEffect;
    ```

3.  Add the following `UPROPERTY()` variable. This will represent the sound for when the enemy is destroyed. Make sure this is declared under the `Public` access modifier:

    ```cpp
    UPROPERTY(EditAnywhere, BlueprintReadOnly)
    class USoundBase* DeathSound;
    ```

With these two properties defined, let’s move on and add the logic required to spawn and use these effects for when the enemy is destroyed.

1.  Inside the source file for the enemy base class, `EnemyBase.cpp`, add the following includes for the `UGameplayStatics` and `UWorld` classes:

    ```cpp
    #include "Kismet/GameplayStatics.h"
    #include "Engine/World.h"
    ```

You will be using the `UGameplayStatics` and `UWorld` classes to spawn the sound and Particle System into the world when the enemy is destroyed.

1.  Within the `AEnemyBase::DestroyEnemy()` function, you have one line of code:

    ```cpp
    Destroy();
    ```

2.  Add the following line of code above the `Destroy()` function call:

    ```cpp
    UWorld* World = GetWorld();
    ```

It is necessary to define the `UWorld` object before attempting to spawn a Particle System or sound because a `World` context object is required.

1.  Next, use an `if()` statement to check the validity of the `World` object you just defined:

    ```cpp
    if(World)
    {
    }
    ```

2.  Within the `if()` block, add the following code to check the validity of the `DeathEffect` property, and then spawn this effect using the `SpawnEmitterAtLocation` function from `UGameplayStatics`:

    ```cpp
    if(DeathEffect)
    {
        UGameplayStatics::SpawnEmitterAtLocation(World, 
          DeathEffect, GetActorTransform());
    }
    ```

It cannot be emphasized enough that you should ensure an object is valid before attempting to spawn or manipulate the object. By doing so, you can avoid engine crashes.

1.  After the `if(DeathEffect)` block, perform the same validity check of the `DeathSound` property and then spawn the sound using the `UGameplayStatics::SpawnSoundAtLocation` function:

    ```cpp
    if(DeathSound)
    {
        UGameplayStatics::SpawnSoundAtLocation(World, 
          DeathSound, GetActorLocation());
    }
    ```

Before calling the `Destroy()` function, you need to make checks regarding whether both the `DeathEffect` and `DeathSound` properties are valid, and if so, spawn those effects using the proper `UGameplayStatics` function. This ensures that regardless of whether either property is valid, the enemy character will still be destroyed.

1.  Now that the `AEnemyBase::DestroyEnemy()` function has been updated to spawn these effects, return to the UE5 editor to compile and hot-reload these code changes.
2.  Within the `/Enemy/Blueprints/` directory. *Double-click* the `BP_Enemy` asset to open it.
3.  In the `Death Effect` and `Death Sound` properties. *Left-click* on the drop-down list for the `Death Effect` property and find the `P_Goblin_Death` Particle System.
4.  Next, underneath the `Death Effect` parameter, *left-click* on the drop-down list for the `Death Sound` property and find the **A_Guardian_Death_Cue** sound cue.
5.  Now that these parameters have been updated and assigned the correct effect, compile and save the enemy Blueprint.
6.  Using `PIE`, spawn the player character and throw a player projectile at an enemy. If an enemy is not present in your level, please add one. When the player projectile collides with the enemy, the VFX and SFX you added will play, as shown in the following screenshot:

![Figure 14.21 – Now, the enemy explodes and gets destroyed in a blaze of glory ](img/Figure_14.21_B18531.jpg)

Figure 14.21 – Now, the enemy explodes and gets destroyed in a blaze of glory

With this exercise complete, the enemy character now plays a Particle System and a sound cue when it is destroyed by the player projectile. This adds a nice layer of polish to the game, and it makes it more satisfying to destroy the enemies.

In the next exercise, you will add a new Particle System and audio components to the player projectile so that it looks and sounds more interesting while it flies through the air.

## Exercise 14.10 – adding effects to the player projectile

In its current state, the player projectile functions the way it is intended to; it flies through the air, collides with objects in the game world, and is destroyed. However, visually, the player projectile is just a ball with a plain white texture.

In this exercise, you will add a layer of polish to the player projectile by adding both a Particle System and an audio component so that the projectile is more enjoyable to use.

Follow these steps to achieve this:

1.  Much like the previous exercises, we will need to migrate assets from the **Action RPG** project to our **SuperSideScroller** project. Please refer to *Exercise 14.09 – adding effects when the enemy is destroyed*, on how to install and migrate assets from the **Action RPG** project.

The two primary assets you will be adding to the project are as follows:

```cpp
P_Env_Fire_Grate_01
A_Ambient_Fire01_Cue
```

The `P_Env_Fire_Grate_01` Particle System asset references additional assets, such as materials and textures, that are included in the `Effects` directory, while `A_Ambient_Fire01_Cue` references additional sound wave and sound attenuation assets included in the `Assets` directory.

The particle you will be using for the player projectile is called `P_Env_Fire_Grate_01` and can be found in the `/Effects/FX_Particle/` directory. This is the same directory that’s used by the `P_Goblin_Death` VFX from the previous exercise. The sound you will be using for the player projectile is called `A_Ambient_Fire01_Cue` and can be found in the `/Assets/Sounds/Ambient/` directory.

1.  *Right-click* on each of these assets in the **Content Browser** interface of the **Action RPG** project and select **Asset Actions** and then **Migrate**.
2.  Make sure that you choose the directory of the **Content** folder for your **SuperSideScroller** project before confirming the migration.

Now that the required assets have been migrated to our project, let’s continue creating the player projectile class.

1.  Open Visual Studio and navigate to the header file for the player projectile class – that is, `PlayerProjectile.h`.
2.  Under the `Private` access modifier, underneath the declaration of the `UStaticMeshComponent* MeshComp` class component, add the following code to declare a new audio component for the player projectile:

    ```cpp
    UPROPERTY(VisibleDefaultsOnly, Category = Sound)
    class UAudioComponent* ProjectileMovementSound;
    ```

3.  Next, add the following code underneath the declaration of the audio component to declare a new Particle System component:

    ```cpp
    UPROPERTY(VisibleDefaultsOnly, Category = Projectile)
    class UParticleSystemComponent* ProjectileEffect;
    ```

Instead of using properties that can be defined within the Blueprint, such as in the enemy character class, these effects will be components of the player projectile. This is because these effects should be attached to the collision component of the projectile so that they move with the projectile as it travels across the level when thrown.

1.  With these two components declared in the header file, open the source file for the player projectile and add the following includes to the list of `include` lines at the top of the file:

    ```cpp
    #include "Components/AudioComponent.h"
    #include "Engine/Classes/Particles/ParticleSystemComponent.h"
    ```

You need a reference to both the audio component and the Particle System classes to create these subobjects using the `CreateDefaultSubobject` function, as well as to attach these components to **RootComponent**.

1.  Add the following lines to create the default subobject of the `ProjectileMovementSound` component, and to attach this component to **RootComponent**:

    ```cpp
    ProjectileMovementSound = CreateDefaultSubobject<UAudioComponent>
      (TEXT("ProjectileMovementSound"));
      ProjectileMovementSound
      ->AttachToComponent(RootComponent, 
      FAttachmentTransformRules::KeepWorldTransform);
    ```

2.  Next, add the following lines to create the default subobject for the `ProjectileEffect` component, and to attach this component to **RootComponent**:

    ```cpp
    ProjectileEffect = CreateDefaultSubobject<UParticle SystemComponent>(TEXT("Projectile
      Effect"));
    ProjectileEffect->AttachToComponent(RootComponent, 
      FAttachmentTransformRules::KeepWorldTransform);
    ```

3.  Now that you have created, initialized, and attached these two components to **RootComponent**, return to the UE5 editor to recompile and hot-reload these code changes.
4.  From the `Content Drawer` interface, navigate to the `/MainCharacter/Projectile/` directory. Find the `BP_PlayerProjectile` asset and *double-click* it to open the Blueprint.

In the **Components** tab, you will find the two new components you added using the preceding code. Observe that these components are attached to the **CollisionComp** component, also known as **RootComponent**.

1.  *Left-click* to select the `P_Env_Fire_Grate_01` VFX asset to this parameter, as shown in the following screenshot:

![Figure 14.22 – Assigning the VFX to the particle system component ](img/Figure_14.22_B18531.jpg)

Figure 14.22 – Assigning the VFX to the particle system component

1.  Before assigning the audio component, let’s adjust the `ProjectileEffect` VFX asset. Update the **Rotation** and **Scale** values of the **Transform** property for the VFX so that they match what is shown in the following screenshot:

![Figure 14.23 – The updated Transform of the Particle System component so that it fits better with the projectile ](img/Figure_14.23_B18531.jpg)

Figure 14.23 – The updated Transform of the Particle System component so that it fits better with the projectile

1.  Navigate to the `Viewport` tab within the Blueprint to view these changes to the `ProjectileEffect` should look as follows:

![Figure 14.24 – Now, the fire VFX has been scaled and rotated appropriately ](img/Figure_14.24_B18531.jpg)

Figure 14.24 – Now, the fire VFX has been scaled and rotated appropriately

1.  Now that the VFX has been set up, *left-click* the `ProjectileMovementSound` component and assign `A_Ambient_Fire01_Cue` to it.
2.  Save and recompile the `BP_PlayerProjectile` Blueprint. Use `PIE` and observe that when you throw the projectile, it now shows the VFX asset and plays the assigned sound:

![Figure 14.25 – The player projectile now has a VFX and an SFX as it flies through the air ](img/Figure_14.25_B18531.jpg)

Figure 14.25 – The player projectile now has a VFX and an SFX as it flies through the air

With this exercise complete, the player projectile now has a VFX and an SFX that play together while it flies through the air. These elements bring the projectile to life and make the projectile much more interesting to use.

Since the VFX and SFX have been created as components of the projectile, they are also destroyed when the projectile is destroyed.

In the next exercise, you will add a particle notify and a sound notify to the **Throw** Animation Montage to provide more of an impact when the player throws the player projectile.

## Exercise 14.11 – adding VFX and SFX notifies

So far, you have been implementing polish elements to the game via C++, which is a valid means of implementation. To give variety, and expand your knowledge of the UE5 toolset, this exercise will walk you through how to use notifies in Animation Montages to add Particle Systems and audio within the animation. Let’s get started!

Much like the previous exercises, we will need to migrate assets from the `Action RPG` project to our **SuperSideScroller** project. Please refer to *Exercise 14.09 – adding effects when the enemy is destroyed*, to learn how to install and migrate assets from the **Action RPG** project.

Follow these steps:

1.  Open the **ActionRPG** project and navigate to the **Content Browser** interface.

The two primary assets you will be adding to the project are as follows:

```cpp
P_Skill_001
A_Ability_FireballCast_Cue
```

The `P_Skill_001` Particle System asset references additional assets such as *materials* and *textures* that are included in the `Effects` directory, while `A_Ability_FireballCast_Cue` references additional *sound wave* assets included in the `Assets` directory.

The particle you will be using for the player when the projectile is thrown is called `P_Skill_001` and can be found in the `/Effects/FX_Particle/` directory. This is the same directory that was used by the `P_Goblin_Death` and `P_Env_Fire_Grate_01` VFX assets in the previous exercises. The sound you will be using for the enemy character destruction is called `A_Ambient_Fire01_Cue` and can be found in the `/Assets/Sounds/Ambient/` directory.

1.  *Right-click* on each of these assets in the **Content Browser** interface of the **Action RPG** project and select **Asset Actions** and then **Migrate**.
2.  Make sure that you choose the directory of the `Content` folder for your **SuperSideScroller** project before confirming the migration.

Now that the assets you need have been migrated into your project, let’s move on to adding the required notifies to the `AM_Throw` asset. Make sure that you return to your **SuperSideScroller** project before continuing with this exercise.

1.  From the `/MainCharacter/Animation/` directory. Find the `AM_Throw` asset and *double-click* it to open it.
2.  Underneath the preview window in the center of the `Anim_ProjectileNotify` earlier in this chapter.
3.  To the left of the **Notifies** track, you will find a **▼** sign that allows you to use additional notify tracks. *Left-click* to add a new notify track, as shown in the following screenshot:

![Figure 14.26 – Adding a new notify track ](img/Figure_14.26_B18531.jpg)

Figure 14.26 – Adding a new notify track

It is useful to add multiple tracks to the timeline to keep things organized when adding multiple notifies.

1.  In the same frame as `Anim_ProjectileNotify`, *right-click* within the new track you created in the previous step. From the `Play Particle Effect`.
2.  Once created, *left-click* to select the new notify and access its `P_Skill_001` VFX asset to the `Particle System` parameter.

Once you’ve added this new VFX, you will notice that the VFX is placed almost toward the bottom, where the player character’s feet are, but not exactly where you want it. This VFX should be placed directly on the floor, or at the base of the character. The following screenshot demonstrates this location:

![Figure 14.27 – The location of the particle notify is not on the ground ](img/Figure_14.27_B18531.jpg)

Figure 14.27 – The location of the particle notify is not on the ground

To fix this, you need to add a new `Socket` to the player character’s skeleton.

1.  Navigate to the `/MainCharacter/Mesh/` directory. *Double-click* the `MainCharacter_Skeleton` asset to open it.
2.  From the `EffectSocket`.
3.  *Left-click* this socket from the hierarchy of bones to view its current location. By default, its location is set to the same position as the **Hips** bone. The following screenshot shows this location:

![Figure 14.28 – The default location of this socket is in the center of the player’s skeleton ](img/Figure_14.28_B18531.jpg)

Figure 14.28 – The default location of this socket is in the center of the player’s skeleton

Using the `EffectSocket` so that its position is set to the following:

```cpp
(X=0.000000,Y=100.000000,Z=0.000000)
```

This position will be closer to the ground and the player character’s feet. The final location can be seen in the following screenshot:

![Figure 14.29 – Moving the socket’s location to the base of the player skeleton ](img/Figure_14.29_B18531.jpg)

Figure 14.29 – Moving the socket’s location to the base of the player skeleton

1.  Now that you have a location for the particle notify, return to the `AM_Throw` Animation Montage.
2.  Within the `Socket Name` parameter. Name it `EffectSocket`.

Note

If `EffectSocket` does not appear via the autocomplete, close and reopen the Animation Montage. Once it's reopened, the `EffectSocket` option should appear for you.

1.  Lastly, the scale of the particle effect is a little too big, so adjust the scale of the projectile so that its value is as follows:

    ```cpp
    (X=0.500000,Y=0.500000,Z=0.500000)
    ```

Now, when the particle effect is played via this notify, its position and scale will be correct, as shown here:

![Figure 14.30 – The particle now plays at the base of the player character’s skeleton ](img/Figure_14.30_B18531.jpg)

Figure 14.30 – The particle now plays at the base of the player character’s skeleton

1.  To add the `Play Sound` notify, add a new track to the **Notifies** timeline section; you should have three in total.
2.  On this new track, and at the same frame position as both the `Play Particle Effect` and `Anim_ProjectileNotify` notifies, *right-click* and select the **Play Sound** notify from the **Add Notify** selection. The following screenshot shows where this notify can be found:

![Figure 14.31 – The Play Sound notify that you learned about earlier in this chapter ](img/Figure_14.31_B18531.jpg)

Figure 14.31 – The Play Sound notify that you learned about earlier in this chapter

1.  Next, *left-click* to select the **Play Sound** notify and access its **Details** panel.
2.  From the `A_Ability_FireballCast_Cue`.

With the sound assigned, when the **Throw** animation is played back, you will see the VFX play and you will hear the sound. The **Notifies** tracks should look as follows:

![Figure 14.32 – The final notify set up on the Throw Animation Montage timeline ](img/Figure_14.32_B18531.jpg)

Figure 14.32 – The final notify set up on the Throw Animation Montage timeline

1.  Save the `AM_Throw` asset and use `PIE` to throw the player projectile.
2.  Now, when you throw the projectile, you will see the particle notify play the `P_Skill_001` VFX and you will hear the `A_Ability_FireballCast_Cue` SFX. The result will look as follows:

![Figure 14.33 – Now, when the player throws the projectile, powerful VFX and SFX are played ](img/Figure_14.33_B18531.jpg)

Figure 14.33 – Now, when the player throws the projectile, powerful VFX and SFX are played

With this final exercise complete, the player now plays powerful VFX and SFX when the player projectile is thrown. This gives the throw animation more power and it feels like the player character is using a lot of energy to throw the projectile.

In the final activity, you will use the knowledge you’ve gained from the last few exercises to add VFX and SFX to the player projectile when it is destroyed.

## Activity 14.02 – adding effects for when the projectile is destroyed

In this final activity, you will use the knowledge that you’ve gained from adding VFX and SFX elements to the player projectile and the enemy character to create an explosion effect for when the projectile collides with an object instead. The reason we’re adding this additional explosion effect is to add a level of polish on top of destroying the projectile when it collides with environmental objects. It would look awkward and out of place if the player projectile were to hit an object and disappear without any audio or visual feedback from the player.

You will add both a Particle System and sound cue parameters to the player projectile and spawn these elements when the projectile collides with an object.

Follow these steps to achieve the expected output:

1.  Inside the `PlayerProjectile.h` header file, add a new Particle System variable and a new sound base variable.
2.  Name the Particle System variable `DestroyEffect` and name the sound base variable `DestroySound`.
3.  In the `PlayerProjectile.cpp` source file, add the include for `UGameplayStatics` to the list of includes.
4.  Update the `APlayerProjectile::ExplodeProjectile()` function so that it now spawns both the `DestroyEffect` and `DestroySound` objects. Return to the UE5 editor and recompile the new C++ code. Inside the `BP_PlayerProjectile` Blueprint, assign the `P_Explosion` VFX, which is already included in your project by default, to the `Destroy Effect` parameter of the projectile.
5.  Assign the `Explosion_Cue` SFX, which is already included in your project by default, to the `Destroy Sound` parameter of the projectile.
6.  Save and compile the player projectile Blueprint.
7.  Use `PIE` to observe the new player projectile’s destruction VFX and SFX.

The expected output is as follows:

![Figure 14.34 – Projectile VFX and SFX ](img/Figure_14.34_B18531.jpg)

Figure 14.34 – Projectile VFX and SFX

With this activity complete, you now have experience with adding polished elements to the game. Not only have you added these elements through C++ code, but you’ve added elements through other tools from UE5\. At this point, you have enough experience to add Particle Systems and audio to your game without having to worry about how to implement these features.

Note

The solution to this activity can be found on GitHub here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions).

# Summary

In this chapter, you learned about the importance of VFX and SFX in the world of game development. Using a combination of C++ code and notifies, you were able to bring gameplay functionality to the player projectile and the enemy character colliding, as well as a layer of polish to this functionality by adding VFX and SFX. On top of this, you learned about how objects are spawned and destroyed in UE5.

Moreover, you learned about how Animation Montages are played, both from Blueprints and through C++. By migrating the logic of playing the **Throw** Animation Montage from Blueprint to C++, you learned how both methods work and how to use both implementations for your game.

By adding a new Animation Notify using C++, you were able to add this notify to the `UWorld->SpawnActor()` function and adding a new socket to the player skeleton, you were able to spawn the player projectile at the exact frame of the **Throw** animation, and at the exact position that you wanted to.

Lastly, you learned how to use the **Play Particle Effect** and **Play Sound** notifies within the **Throw** Animation Montage to add VFX and SFX to the throw of the player projectile. This chapter taught you about the different methods that exist inside UE5 when it comes to using VFX and SFX for your game.

Now that the player projectile can be thrown and destroy enemy characters, it is time to implement the final set of mechanics for the game. In the next chapter, you will create the collectibles that the player can collect, and you will also create a powerup for the player that will improve the player’s movement mechanics for a short period.