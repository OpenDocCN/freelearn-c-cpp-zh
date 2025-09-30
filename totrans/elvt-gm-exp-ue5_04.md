# 4

# Getting Started with Player Input

In the previous chapter, we created our C++ class, which inherits from the `Character` class, and added all the necessary `Actor` components to be able to see the game from the character’s perspective, as well as being able to see the character itself. We then created a `Blueprint` class that inherits from that C++ class in order to visually set up all of its necessary components.

In this chapter, we will be looking at these topics in more depth, as well as covering their C++ usage. We will learn about how player input works in UE5, how the engine handles input events (*key presses and releases*), and how we can use them to control logic in our game.

In this chapter, we will cover the following topics:

*   Understanding Input Actions and Contexts
*   Processing Player Input
*   Pivoting the camera around the character

By the end of this chapter, you will know about **Input Actions** and **Input Contexts**, how to create and modify them, how to listen to each of those mappings, and how to execute in-game actions when they’re pressed and released.

Note

In this chapter, we will be using an alternative version of the `Character` blueprint we created called `BP_MyTPC` in the previous chapter. This chapter’s version will have the default UE5 Mannequin mesh, not the one from Mixamo.

Let’s start this chapter by getting to know how UE5 abstracts the keys pressed by a player to make it easier for you to be notified of those events.

# Technical requirements

The project for this chapter can be found in the Chapter04 folder of the code bundle for this book, which can be downloaded here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition).

# Understanding Input Actions and Contexts

Player input is the thing that distinguishes video games from other forms of entertainment media – the fact that they’re interactive. For a video game to be interactive, it must take into account a player’s input. Many games do this by allowing the player to control a virtual character that acts upon the virtual world it’s in, depending on the keys and buttons that the player presses, which is exactly what we’ll be doing in this chapter.

Note

It’s important to note that UE5 has two input systems – the Legacy Input System, used since the start of UE4, and the new Enhanced Input System, introduced only in the last version of UE5 as an experimental system and now as a complete plugin in UE5\. We will be using the new Enhanced Input System in this book. If you wish to know more about UE5’s Legacy Input System, you can do so by accessing this link: [https://docs.unrealengine.com/4.27/en-US/InteractiveExperiences/Input/](https://docs.unrealengine.com/4.27/en-US/InteractiveExperiences/Input/)

Most game development tools nowadays allow you to abstract keypresses into **actions**, which allow you to associate a name (for example, *Jump*) with several different player inputs (pressing a button, flicking a thumbstick, and so on). In UE5, the way in which you can specify this is through the use of **Input Actions** combined with **Input Contexts** (also referred to as **Input Mapping Contexts**).

**Input Contexts** contain **Input Actions** that are associated with them, along with which keys will execute them, and **Input Actions** contain specifications as to how they will be executed. The combination of both of these assets allows you to do something when an **Input Action** is triggered but also easily change how that **Input Action** is triggered and by which keys.

In order to better understand how **Input Contexts** and **Input Actions** work together, let’s think of a game, such as *GTA*, where you have different gameplay contexts in which you control different people/objects with different keys.

For instance, when you’re controlling your player character running around the city, you use the movement keys to move the character around, and you use a different key to make your character jump. However, when you enter a car, the controls will change. The movement keys will now steer the car instead, and the same key that was used for jumping will now be used, for instance, for braking.

In this example, you have two different Input Contexts (controlling the character and controlling the vehicle), each with its own set of Input Actions. Some of those Input Actions are triggered by the same key, but that’s fine, because they’re done in different Input Contexts (for example, using the same key to cause your character to jump and to stop the vehicle).

Before we start looking into some of the Enhanced Input-related assets, because it’s a plugin, we’ll have to enable it. To enable it, follow these steps:

1.  Go to **Edit** | **Plugins** | **Built-In** | **Input** and tick the **Enabled** box for the **Enhanced Input** plugin. After you have done so, you’ll be prompted to restart the editor.
2.  Click the **Restart Now** button when this happens. After the editor restarts, and now that the **Enhanced Input** plugin has been enabled, you’ll need to tell the engine to use its classes to handle the player’s input.
3.  To do this, go to `EnhancedPlayerInput` and the `EnhancedInputComponent`. Now that the **Enhanced Input** plugin has been enabled and its classes are being used, we can proceed with this chapter’s content.

In order to know more about Input Contexts and Input Actions, let’s inspect them. Follow these steps:

1.  Right-click on the `IA_Movement`, and then open it. You should see the Input Action window, which has the following properties:

![Figure 4.1 – The Action window](img/Figure_4.01_B18531.jpg)

Figure 4.1 – The Action window

Now, let’s take a look at its options in detail:

*   `true`, another Input Action with a lower priority that will be triggered by the same key won’t be triggered.
*   **Trigger when Paused**: This specifies whether this Input Action can be triggered if the game is paused.
*   **Reserve All Mappings**: This specifies whether a higher priority Input Action will be triggered if it’s triggered by the same key.
*   **Value Type**: This specifies the type of value for this Input Action. Its values can be the following:
    *   **Digital (bool)**: Used for Input Actions that have a binary state – for instance, a jumping Input Action, in which the player is either pressing it or not, would use this value.
    *   **Axis 1D (float)**: Used for Input Actions that have a scalar state in one dimension – for instance, accelerating in a racing game, where you can use the gamepad’s triggers to control the throttle.
    *   **Axis 2D (Vector2D)**: Used for Input Actions that have a scalar state in two dimensions – for instance, actions for moving your character, which are done using two axes (the forward axis and the sideways axis), would be good candidates for using this value.
*   **Axis 3D (Vector)**: Used for Input Actions that have a scalar state in three dimensions. This value isn’t as likely to be used as the others, but you may find a use for it.
*   **Triggers**: This specifies the key events that will execute this Input Action. The values for this can be a combination of the following:
    *   **Chorded Action**: The Input Action is triggered as long as a different specified Input Action is also triggered.
    *   **Down**: The Input Action is triggered for every frame that the key exceeds the actuation threshold.

Note

The actuation threshold is the value at which a key’s input will be considered for executing an action. Binary keys (like the ones on a keyboard) have an input value of either `0` (not pressed) or `1` (pressed), while scalar keys, like the triggers on a gamepad, have an input value that goes continuously from `0` to `1` or, like the individual axes of the thumbsticks, that go continuously from `–1` to `1`.

*   **Hold**: The Input Action is triggered when the key has exceeded the actuation threshold for a specified amount of time. You can optionally specify whether it’s triggered once or for every frame.
*   **Hold and Release**: The Input Action is triggered when the key has exceeded the actuation threshold for a specified amount of time and then stops exceeding that actuation threshold.
*   **Pressed**: The Input Action is triggered once the key exceeds the actuation threshold and won’t be triggered again until it’s released.
*   **Pulse**: The Input Action is triggered at a specified interval as long as the key exceeds the actuation threshold. You can specify whether the first pulse triggers the Input Action and whether there’s a limit to how many times it can be called.
*   **Released**: The Input Action is triggered once the key stops exceeding the actuation threshold.
*   **Tap**: The Input Action is triggered when the key starts and then stops exceeding the Actuation Threshold, as long as it’s done within the specified amount of time.
*   **Modifiers**: This specifies the ways in which this Input Action’s input will be modified:
*   `0` if it’s lower than the lower threshold and as `1` if it’s higher than the upper threshold.
*   **FOV Scaling**: The key’s input will be scaled alongside the FOV (if the FOV increases, the key’s input will increase, and vice versa).
*   **Modifier Collection**: The key’s input will be modified according to the specified list of modifiers.
*   **Negate**: The key’s input will be inverted.
*   **Response Curve – Exponential**: An exponential curve will be applied on the key’s input.
*   **Response Curve – User Defined**: A user-defined curve will be applied on the key’s input.
*   **Scalar**: The key’s input will be scaled at each axis according to the scalar specified.
*   **Smooth**: The key’s input will be smoothed out across multiple frames.
*   **Swizzle Input Axis Values**: The key’s axis order will be switched.
*   **To World Space**: The key’s axes will be converted toworld space.

1.  After doing this, right-click on `IC_Character` and open it.

You should see the Input Action window pop up. Note that it has an empty **MAPPINGS** property.

![Figure 4.2 – The MAPPINGS property ](img/Figure_4.02_B18531.jpg)

Figure 4.2 – The MAPPINGS property

1.  Let’s now add a new mapping. Press the **+** button next to the **Mappings** property. You’ll notice a new property show up where you can specify the Input Action this mapping will be associated with.

This action can be triggered by several different keys, each of which can have its own triggers and modifiers, which work the same as the corresponding properties in the Input Action asset.

Note

When it comes to modifying the **Triggers** and **Modifiers** properties, the usual practice is to change the modifiers in the Input Context asset and the triggers in the Input Action asset.

Note

We will not be using these properties in this book, but for each Input Mapping Context, you can specify whether it can be modified by a player by ticking the **Is Player Mappable** property and specifying **Player Mappable Options**.

When we generated the `Third Person` template project back in [*Chapter 1*](B18531_01.xhtml#_idTextAnchor016), *Introduction to Unreal Engine*

, it came with some inputs already configured, which were the *W*, *A*, *S*, and *D* keys, as well as the `left thumbstick` for movement, the *spacebar* key, and the `gamepad bottom face` button for jumping.

For context, let’s consider an Xbox One controller, which can be broken down into the following:

*   The **left analog stick**, usually used for controlling movement in games
*   The **D-pad**, which can control movement and also has a variety of other uses
*   The **right analog stick**, usually used for controlling the camera and view perspective
*   The **face buttons** (**X**, **Y**, **A**, and **B**), which can have various uses depending on the game but usually allow the player to perform actions in the game world
*   The **bumpers and triggers** (**LB**, **RB**, **LT**, and **RT**), which can be used for actions such as aiming and shooting or accelerating and braking

Now that we’ve learned how to set up `Input Actions`, let’s add some of them in the next exercise.

## Exercise 4.01 – creating the movement and jump input actions

In this exercise, we’ll be adding the mappings for the *Movement* and *Jump* Input Actions.

To achieve this, follow these steps:

1.  Open the **IA_Movement** Input Action.
2.  Set its value type as **Axis2D**. We’ll make this an Input Action of type **Axis2D** because the character’s movement is done on two axes – the forward axis (the *Y* axis for this Input Action) and the sideways or right axis (the *X* axis for this Input Action):

![Figure 4.3 – The Value Type options ](img/Figure_4.03_B18531.jpg)

Figure 4.3 – The Value Type options

1.  Add a new trigger of type **Down** with an actuation threshold of **0,1**. This will ensure that this Input Action is called when one of its keys has an actuation threshold of at least **0,1**:

![Figure 4.4 – The Down trigger ](img/Figure_4.04_B18531.jpg)

Figure 4.4 – The Down trigger

1.  Open the **IC_Character** Input Context.
2.  Click the **+** icon to the right of the **Mappings** property to create a new mapping:

![Figure 4.5 – Adding a new action mapping ](img/Figure_4.05_B18531.jpg)

Figure 4.5 – Adding a new action mapping

1.  When you’ve done so, you should see a new empty mapping with its properties either empty or set to **None**:

![Figure 4.6 – The default settings of a new action mapping ](img/Figure_4.06_B18531.jpg)

Figure 4.6 – The default settings of a new action mapping

1.  Set the Input Action of this mapping (the first property that’s set to **None**) to **IA_Movement**:

![Figure 4.7 – The new IA_Movement mapping ](img/Figure_4.07_B18531.jpg)

Figure 4.7 – The new IA_Movement mapping

1.  Set the first key in this mapping to **Gamepad Left Thumbstick Y-Axis**.

![Figure 4.8 – The Gamepad Left Thumbstick Y-Axis key ](img/Figure_4.08_B18531.jpg)

Figure 4.8 – The Gamepad Left Thumbstick Y-Axis key

Note

If the key you want to set is from one of the input devices you have connected (for example, mouse, keyboard, or gamepad), you can click the button to the left of the key dropdown and then press the actual key you want to set, instead of searching for it in the list. For instance, if you want to set a mapping to use the *F* key on the keyboard, you can click that button, then press the *F* key, and then that key will be set for that mapping.

Because we want this key to control the Input Action’s *Y* axis instead of its *X* axis, we need to add the **Swizzle Input Axis Values** modifier with the **YXZ** value.

![Figure 4.9 – The Swizzle Input Axis modifier ](img/Figure_4.09_B18531.jpg)

Figure 4.9 – The Swizzle Input Axis modifier

1.  Click the **+** button to the right of the Input Action set for this mapping in order to add a new key and execute that Input Action:

![Figure 4.10 – The + button to the right of IA_Movement ](img/Figure_4.10_B18531.jpg)

Figure 4.10 – The + button to the right of IA_Movement

1.  Set the new key to **Gamepad Left Thumbstick X-Axis**. Because this will already control the movement Input Action’s *X* axis, we won’t need to add any modifiers.
2.  Add another key to the Input Action, this time the *W* key. Because this key will be used for moving forward, and therefore use the *Y* axis, it will need the same modifier that we added before – the **Swizzle Input Axis** modifier with the **YXZ** value.
3.  Add another key to the Input Action, this time the *S* key. Because this key will be used for moving backward, and therefore use the *Y* axis, it will need the same modifier we added before – the `–1` on the *Y* axis when this key is pressed (that is, when its input is `1`):

![Figure 4.11 – The Swizzle Input Axis Values and Negate modifiers ](img/Figure_4.11_B18531.jpg)

Figure 4.11 – The Swizzle Input Axis Values and Negate modifiers

1.  Add another key to the Input Action, this time the *D* key. Because this key will be used for moving right, and therefore use the positive end of the *X* axis, it won’t need any modifiers.
2.  Add another key to the Input Action, this time the *A* key. Because this key will be used for moving left, and therefore use the negative end of the *X* axis, it will need the `Negate` modifier, just like the *S* key.
3.  Create a new Input Action asset called `IA_Jump`, and then open it.
4.  Add a **Down** trigger and leave its actuation threshold as **0,5**:

![Figure 4.12 – The Down trigger ](img/Figure_4.12_B18531.jpg)

Figure 4.12 – The Down trigger

1.  Go back to the **IC_Character** Input Context asset and add a new Input Action to the **Mapping**s property – this time, the **IA_Jump** Input Action we just created:

![Figure 4.13 – The IA_Jump mapping ](img/Figure_4.13_B18531.jpg)

Figure 4.13 – The IA_Jump mapping

1.  Add two keys to this mapping – **Space Bar** and **Gamepad Face Button Bottom**. If you’re using an Xbox controller, this will be the *A* button, and if you’re using a PlayStation controller, this will be the *X* button:

![Figure 4.14 – The IA_Jump mapping keys ](img/Figure_4.14_B18531.jpg)

Figure 4.14 – The IA_Jump mapping keys

And with those steps completed, we’ve completed this chapter’s first exercise, where you’ve learned how you can specify Input Action Mappings in UE5, allowing you to abstract which keys are responsible for which in-game actions.

Let’s now take a look at how UE5 handles player input and processes it within the game.

# Processing Player Input

Let’s think about a situation where the player presses the **Jump** Input Action, which is associated with the *spacebar* key, to get the player character to jump. Between the moment the player presses the *Spacebar* key and the moment the game makes the player character jump, quite a few things have to happen to connect those two events.

Let’s take a look at all of the necessary steps that lead from one event to the other:

1.  `Hardware Input`: The player presses the *spacebar* key. UE5 will be listening to this keypress event.
2.  The `PlayerInput` class: After the key is pressed or released, this class will translate that key into an Input Action. If there is a corresponding Input Action, it will notify all classes that are listening to the action that it was just pressed, released, or updated. In this case, it will know that the *Spacebar* key is associated with the *Jump* Input Action.
3.  The `Player Controller` class: This is the first class to receive these keypress events, given that it’s used to represent a player in the game.
4.  The `Pawn` class: This class (and consequently the `Character` class, which inherits from it) can also listen to those keypress events, as long as they are possessed by a Player Controller. If so, it will receive these events after that class. In this chapter, we will be using our `Character` C++ class to listen to action and axis events.

Now that we know how UE5 handles player inputs, let’s see how we can listen to Input Actions in C++ in the next exercise.

## Exercise 4.02 – listening to movement and jump input actions

In this exercise, we will register the Input Actions we created in the previous section with our character class by binding them to specific functions in our character class using C++.

The main way for a `SetupPlayerInputComponent` function. The `MyThirdPersonChar` class should already have a declaration and an implementation for this function. Let’s have our character class listen to those events by following these steps:

1.  Open the `MyThirdPersonChar` class header file in Visual Studio, and make sure there’s a declaration for a `protected` function called `SetupPlayerInputComponent` that returns nothing and receives a `class UInputComponent* PlayerInputComponent` property as a parameter. This function should be marked as both `virtual` and `override`:

    ```cpp
    virtual void SetupPlayerInputComponent(class UInputComponent*
    PlayerInputComponent) override;
    ```

2.  Add a declaration for a `public` `class UInputMappingContext*` property called `IC_Character`. This property must be a `UPROPERTY` and have the `EditAnywhere` and `Category = Input` tags. This will be the Input Context we’ll be adding for the character’s input:

    ```cpp
    UPROPERTY(EditAnywhere, Category = Input)
    class UInputMappingContext* IC_Character;
    ```

3.  After that, we’ll need to add the Input Actions to listen for the character’s input. Add three `public` `class UInputAction*` properties, all of which must be `UPROPERTY` and have the `EditAnywhere` and `Category = Input` tags. Those two properties will be called the following:
    *   `IA_Move`

        ```cpp
        IA_JumpUPROPERTY(EditAnywhere, Category = Input)
        class UInputAction* IA_Move;
        UPROPERTY(EditAnywhere, Category = Input)
        class UInputAction* IA_Jump
        ```

4.  Open this class’s `source` file and make sure that this function has an implementation:

    ```cpp
    void AMyThirdPersonChar::SetupPlayerInputComponent(class 
    UInputComponent* PlayerInputComponent)
    {
    }
    ```

5.  Because in UE5 you can use either the legacy Input Component or the Enhanced Input Component, we need to account for this. Inside the previous function’s implementation, start by casting the `PlayerInputComponent` parameter to the `UEnhancedInputComponent` class and saving it inside a new `EnhancedPlayerInputComponent` property of type `UEnhancedInputComponent*`:

    ```cpp
    UEnhancedInputComponent* EnhancedPlayerInputComponent =
    Cast<UEnhancedInputComponent>(PlayerInputComponent);
    ```

Because we’ll be using `UEnhancedInputComponent`, we need to include it:

```cpp
#include "EnhancedInputComponent.h"
```

1.  If it’s not `nullptr`, cast the `Controller` property to `APlayerController` and save it in a local `PlayerController` property:

    ```cpp
    if (EnhancedPlayerInputComponent != nullptr)
    {
     APlayerController* PlayerController =
     Cast<APlayerController>(GetController());
    }
    ```

If the newly created `PlayerController` property isn’t `nullptr`, then we’ll need to fetch `UEnhancedLocalPlayerSubsystem` so that we can tell it to add the `IC_Character` Input Context and activate its Input Actions.

1.  To do this, create a new `UEnhancedLocalPlayerSubsystem*` property called `EnhancedSubsystem` and set it to return the value of the `ULocalPlayer::GetSubsystem` function. This function receives a template parameter representing the subsystem we want to fetch, which is `UEnhancedLocalPlayerSubsystem`, and a normal parameter of type `ULocalPlayer*`. This last parameter’s type is a representation of a player who’s controlling a pawn in the current instance of the game, and we’ll pass it by calling `PlayerController->GetLocalPlayer()`:

    ```cpp
    UEnhancedInputLocalPlayerSubsystem* EnhancedSubsystem =
    ULocalPlayer::GetSubsystem<UEnhancedInputLocalPlayerSubsystem>(PlayerController->GetLocalPlayer());
    ```

Because we’ll be using the `UEnhancedLocalPlayerSubsystem`, we need to include it:

```cpp
#include "EnhancedInputSubsystems.h"
```

1.  If the `EnhancedSubsystem` property isn’t `nullptr`, call its `AddMappingContext` function, which receives the following parameters:
    *   `UInputMappingContext* Mapping Context`: The Input Context we want to activate – in this case, the `IC_Character` property
    *   `int32 Priority`: The priority we want this Input Context to have, which we’ll pass as `1`

        ```cpp
        EnhancedSubsystem->AddMappingContext(IC_Character, 1);
        ```

2.  Because we’ll be using `UInputMappingContext`, we need to include it:

    ```cpp
    #include "InputMappingContext.h"
    ```

3.  Now that we’ve added the logic to activate the Input Context, let’s add the logic for listening to the Input Actions. Add the code for the following steps after we check whether `PlayerController` is `nullptr`, but still inside the brackets where we check whether `EnhancedPlayerInputComponent` is `nullptr`:

    ```cpp
    if (EnhancedPlayerInputComponent != nullptr)
    {
     APlayerController* PlayerController = 
     Cast<APlayerController>(GetController());
     if (PlayerController != nullptr)
     {
      ...
     }
     // Continue here
    }
    ```

In order to listen to the `IA_Movement` Input Action, we’ll call the `EnhancedPlayer InputComponent` `BindAction` function, which receives as parameters the following:

*   `UInputAction* Action`: The Input Action to listen to, which we’ll pass as the `IA_Movement` property.
*   `ETriggerEvent TriggerEvent`: The input event that will cause the function to be called. Because this Input Action is triggered for every frame in which it’s being used, and it’s triggered using the `Down` trigger, we’ll pass this as the `Triggered` event.
*   `UserClass* Object`: The object that the `callback` function will be called on – in our case, that’s the `this` pointer.
*   `HANDLER_SIG::TUObjectMethodDelegate <UserClass> ::FMethodPtr Func`: This property is a bit wordy, but it’s essentially a pointer to the function that will be called when this event happens, which we can specify by typing `&`, followed by the class’s name, then `::`, and finally, the function’s name. In our case, we want this to be the `Move` function, which we’ll be creating in a following step, so we’ll specify it with `& AMyThirdPersonChar::Move`:

    ```cpp
    EnhancedPlayerInputComponent->BindAction(IA_Move,
    ETriggerEvent::Triggered, this, &AMyThirdPersonChar
    ::Move);
    ```

Because we’ll be using `UInputAction`, we need to include it:

```cpp
#include "InputAction.h"
```

1.  Let’s now bind the function that will make the player character start jumping. In order to do this, duplicate the `BindAction` function call we added for the `IA_Move` Input Action, but make the following changes:
    *   Instead of passing the `IA_Move` Input Action, pass the `IA_Jump` Input Action.
    *   Instead of passing the `&AMyThirdPersonChar::Move` function, pass `&ACharacter::Jump`. This is the function that will make the character jump.
    *   Instead of passing `ETriggerEvent::Trigger`, pass `ETriggerEvent::Started`. This is so that we can be notified when the key starts and stops being pressed:

        ```cpp
        EnhancedPlayerInputComponent->BindAction(IA_Jump,
        ETriggerEvent::Started, this, &ACharacter::Jump);
        ```

2.  In order to bind the function that will make the player character stop jumping, let’s now duplicate the last `BindAction` function call that we did, but make the following changes to it:
    *   Instead of passing the `ETriggerEvent::Started`, we’ll pass `ETriggerEvent::Completed`, so that the function gets called when this Input Action stops being triggered.
    *   Instead of passing the `&ACharacter::Jump` function, pass `&ACharacter::StopJumping`. This is the function that will make the character stop jumping:

        ```cpp
        EnhancedPlayerInputComponent->BindAction(IA_Jump,
        ETriggerEvent::Completed, this, &ACharacter::StopJumping);
        ```

Note

All functions used to listen to Input Actions must receive either no parameters or a parameter of type `FInputActionValue&`. You can use this to check its value type and fetch the right one. For instance, if the Input Action that triggers this function has a `Digital` value type, its value will be of type `bool`, but if it has an `Axis2D` value type, its value will be of type `FVector2D`. The latter is the type we’ll be using for the `Move` function because that’s its corresponding value type.

Another option for listening to Input Actions is to use `Delegates`, which is outside the scope of this book.

1.  Let’s now create the `Move` function that we referenced in a previous step. Go to the class’s header file and add a declaration for a `protected` function called `Move`, which returns nothing and receives a `const FInputActionValue& Value` parameter:

    ```cpp
    void Move(const FInputActionValue& Value);
    ```

2.  Because we’re using `FInputActionValue`, we have to include it:

    ```cpp
    #include "InputActionValue.h"
    ```

3.  In the class’s source file, add this function’s implementation, where we’ll start by fetching the `Value` parameter’s input as `FVector2D`. We’ll do this by calling its `Get` function, passing as a template parameter the `FVector2D` type. We’ll also save its return value in a local variable called `InputValue`:

    ```cpp
    void AMyThirdPersonChar::Move(const FInputActionValue&
    Value)
    {
     FVector2D InputValue = Value.Get<FVector2D>();
    }
    ```

4.  Next, check whether the `Controller` property is valid (not `nullptr`) and whether the `InputValue` property’s `X` or `Y` value is different to `0`:

    ```cpp
    if (Controller != nullptr && (InputValue.X != 0.0f ||
    InputValue.Y != 0.0f))
    ```

If all of these conditions are `true`, we’ll then get the camera’s rotation on the *z* axis (yaw), so that we can move the character relative to where the camera is facing. To achieve this, we can create a new `FRotator` property called `YawRotation` with a value of `0` for pitch (rotation along the *y* axis) and roll (rotation along the *x* axis) and the value of the camera’s current yaw for the property’s yaw. To get the camera’s yaw value, we can call the Player Controller’s `GetControlRotation` function and then access its `Yaw` property:

```cpp
const FRotator YawRotation(0, Controller->
  GetControlRotation().Yaw, 0);
```

Note

The `FRotator` property’s constructor receives the `Pitch` value, the `Yaw` value, and then the `Roll` value.

*   After that, we’ll check whether the `InputValue`’s `X` property is different to `0`:

    ```cpp
    if (InputValue.X != 0.0f)
    {
    }
    ```

*   If it is, get the right vector of `YawRotation` and store it in an `Fvector RightDirection` property. You can get a rotator’s right vector by calling the `KistemMathLibrary` object’s `GetRightVector` function. A rotator or vector’s right vector is simply its perpendicular vector that points to its right. The result of this will be a vector that points to the right of where the camera is currently facing:

    ```cpp
    const Fvector RightDirection = 
      UkismetMathLibrary::GetRightVector(YawRotation);
    ```

*   We can now call the `AddMovementInput` function, which will make our character move in the direction we specify, passing as parameters the `RightDirection` and `InputValue` `X` properties:

    ```cpp
    AddMovementInput(RightDirection, InputValue.X);
    ```

*   Because we’ll be using both the `KismetMathLibrary` and `Controller` objects, we’ll need to include them at the top of this source file:

    ```cpp
    #include "Kismet/KismetMathLibrary.h"
    #include "GameFramework/Controller.h"
    ```

1.  After checking whether the `X` property of `InputValue` is different to `0`, check whether its `Y` property is different to `0`:

    ```cpp
    if (InputValue.X != 0.0f)
    {
     ...
    }
    if (InputValue.Y != 0.0f)
    {
    }
    ```

2.  If it is, call the `YawRotation` property’s `Vector` function and store its return value in an `FVector ForwardDirection` property. This function will convert `FRotator` to `FVector`, which is equivalent to getting a rotator’s `ForwardVector`. The result of this will be a vector that points forward of where the camera is currently facing:

    ```cpp
    const FVector ForwardDirection = YawRotation.Vector();
    ```

We can now call the `AddMovementInput` function, passing as parameters the `ForwardDirection` and `InputValue` `Y` properties:

```cpp
AddMovementInput(ForwardDirection, InputValue.Y);
```

1.  Before we compile our code, add the `EnhancedInput` plugin to our project’s `Build.cs` file in order to notify UE5 that we’ll be using this plugin in our project. If we don’t do this, parts of our project won’t compile.
2.  Open the `.Build.cs` file inside your project’s `Source/<ProjectName>` folder, which is a C# file and not a C++ file, located inside your project’s source folder.
3.  Open the file, and you’ll find the `AddRange` function from the `PublicDependencyModuleNames` property being called. This is the function that tells the engine which modules this project intends to use. As a parameter, an array of strings is sent with the names of all the intended modules for the project. Given that we intend on using UMG, we’ll need to add the `EnhancedInput` module after the `InputCore` module:

    ```cpp
    PublicDependencyModuleNames.AddRange(new string[] { "Core",
    "CoreUObject", "Engine", "InputCore", "EnhancedInput",
    "HeadMountedDisplay" });
    ```

4.  Now that you’ve notified the engine that we’ll be using the `EnhancedInput` module, compile your code, open the editor, and open your `BP_MyTPS` blueprint asset. Delete the `InputAction Jump` event, as well as the nodes connected to it. Do the same for the **InputAxis MoveForward** and **InputAxis MoveRight** events. We will be replicating this logic in C++ and need to remove its Blueprint functionality so that there are no conflicts when handling input.
5.  Next, set the **IC Character** property to the **IC_Character** Input Context, the **IA Move** property to the **IA_Movement** Input Action, and the **IA Jump** property to the **IA_Jump** Input Action:

![Figure 4.15 – The IC Character, IA Move, and IA Jump properties ](img/Figure_4.15_B18531.jpg)

Figure 4.15 – The IC Character, IA Move, and IA Jump properties

1.  Now, play the level. You should be able to move the character using the keyboard’s *W*, *A*, *S*, and *D* keys or the controller’s *left thumbstick*, as well as jumping with the *Spacebar* key or *gamepad face button bottom*:

![Figure 4.16 – The player character moving  ](img/Figure_4.16_B18531.jpg)

Figure 4.16 – The player character moving

After following all these steps, you will have concluded this exercise. You now know how to create and listen to your own **Input Action** events using C++ in UE5\. Doing this is one of the most important aspects of game development, so you’ve just completed an important step in your game development journey.

Now that we’ve set up all of the logic necessary to have our character move and jump, let’s add the logic responsible for rotating the camera around our character.

# Turning the camera around the character

Cameras are an extremely important part of games, as they dictate what and how the player will see in your game throughout the play session. When it comes to third-person games, which is what this project is about, the camera allows a player not only to see the world around them but also the character they’re controlling. Whether the character is taking damage, falling, or something else, it’s important for the player to always know the state of the character they are controlling and to be able to have the camera face the direction they choose.

Like every modern third-person game, we will always have the camera rotate around our player character. To have our camera rotate around our character, after setting up the **Camera** and **Spring Arm** components in [*Chapter 2*](B18531_02.xhtml#_idTextAnchor043), *Working with Unreal Engine*, let’s continue by adding a new **Look** Input Action. Follow these steps:

1.  Do this by duplicating the `IA_Look`. Because this new Input Action’s setup is similar to that of the **IA_Move** Input Action, we’ll leave this duplicated asset as is.
2.  Then, open the **IA_Character** Input Context and add a new mapping for the **IA_Look** Input Action.
3.  Add the following keys to this new mapping – **Mouse X**, **Mouse Y**, **Gamepad Right Thumbstick X-Axis**, and **Gamepad Right Thumbstick Y-Axis**. Because the *Y* keys will be controlling the Input Action’s *Y* axis, we’ll have to add the **Swizzle Input Axis Values** modifier to them (the **Mouse Y** and **Gamepad Right Thumbstick Y-Axis** keys). Additionally, because the **Mouse Y** key will make the camera go down when you mose the mouse up, we’ll have to also add a **Negate** modifier to it:

![Figure 4.17 – The mappings for the IA_Look Input Action](img/Figure_4.17_B18531.jpg)

Figure 4.17 – The mappings for the IA_Look Input Action

Let’s now add the C++ logic responsible for turning the camera with the player’s input:

1.  Go to the `MyThirdPersonChar` class’s header file and add a `public` `class UInputAction* IA_Look` property, which must be `UPROPERTY` and have the `EditAnywhere` and `Category = Input` tags:

    ```cpp
    UPROPERTY(EditAnywhere, Category = Input)
    class UInputAction* IA_Look;
    ```

2.  Next, add a declaration for a `protected` function called `Look`, which returns nothing and receives a `const FInputActionValue& Value` parameter:

    ```cpp
    void Look(const FInputActionValue& Value);
    ```

3.  Next, go to the `SetupPlayerInputComponent` function implementation, in the class’s source file, and duplicate the line responsible for listening to the `IA_Move` Input Action. In this duplicated line, change the first parameter to `IA_Look` and the last parameter to `&AMyThirdPersonChar::Look`:

    ```cpp
    EnhancedPlayerInputComponent->BindAction(IA_Look,
    ETriggerEvent::Triggered, this, &AMyThirdPersonChar::Look);
    ```

4.  Then, add the `Look` function’s implementation, where we’ll start by fetching the `Value` parameter’s input as `FVector2D`. We’ll do this by calling its `Get` function, passing as a `template` parameter the `FVector2D` type. We’ll also save its `return` value in a local variable called `InputValue`:

    ```cpp
    void AMyThirdPersonChar::Look(const FInputActionValue& Value)
    {
     FVector2D InputValue = Value.Get<FVector2D>();
    }
    ```

5.  If the `InputValue` `X` property is different to `0`, we’ll call the `AddControllerYawInput` function, passing this property as a parameter. After that, check whether the `InputValue` `Y` property is different to `0`, and then we’ll call the `AddControllerPitchInput` function, passing this property as a parameter:

    ```cpp
    if (InputValue.X != 0.0f)
    {
      AddControllerYawInput(InputValue.X);
    }
    if (InputValue.Y != 0.0f)
    {
      AddControllerPitchInput(InputValue.Y);
    }
    ```

Note

The `AddControllerYawInput` and `AddControllerPitchInput` functions are responsible for adding rotation input around the *z* (turning left and right) and *y* (looking up and down) axes respectively.

1.  After you’ve done this, compile your code, open the editor, and open your **BP_MyTPS** Blueprint asset. Set its **IA_Look** property to the **IA_Look** Input Action:

![Figure 4.18 – The camera is rotated around the player ](img/Figure_4.18_B18531.jpg)

Figure 4.18 – The camera is rotated around the player

When you play the level, you should now be able to move the camera by rotating the mouse or by tilting the controller’s *right thumbstick*:

![Figure 4.19 – The camera is rotated around the player ](img/Figure_4.19_B18531.jpg)

Figure 4.19 – The camera is rotated around the player

And that concludes the logic for rotating the camera around the player character with the player’s input. Now that we’ve learned how to add inputs to our game and associate them with in-game actions, such as jumping and moving the player character, let’s consolidate what we’ve learned in this chapter by going through how to add a new `Walk` action to our game from start to finish in the next activity.

# Activity 4.01 – adding walking logic to our character

In the current game, our character runs by default when we use the movement keys, but we need to reduce the character’s speed and make it walk.

So, in this activity, we’ll be adding logic that will make our character walk when we move it while holding the *Shift* key on the keyboard or the **Gamepad Face Button Right** key (*B* for the Xbox controller and *O* for the PlayStation controller).

To do this, follow these steps:

1.  Duplicate the `IA_Walk`. Because this new Input Action’s setup is similar to that of the **IA_Jump** Input Action, we’ll leave this duplicated asset as is.
2.  Then, open the **IA_Character** Input Context and add a new mapping for the **IA_Walk** Input Action. Add the following keys to this new mapping – **Left Shift**, and **Gamepad Face Button Right**.
3.  Open the `MyThirdPersonChar` class’s header file and add a `class UInputAction* IA_Walk` property, which must be `UPROPERTY` and have the `EditAnywhere` and `Category = Input` tags.
4.  Then, add declarations for two `protected` functions that return nothing and receive no parameters, called `BeginWalking` and `StopWalking`.
5.  Add the implementations for both these functions in the class’s source file. In the implementation of the `BeginWalking` function, change the character’s speed to 40% of its value by modifying the `CharacterMovementComponent` property’s `MaxWalkSpeed` property accordingly. To access the `CharacterMovementComponent` property, use the `GetCharacterMovement` function.

The implementation of the `StopWalking` function will be the inverse of that of the `BeginWalking` function, which will increase the character’s walk speed by 250%.

1.  Listen to the `Walk` action by going to the `SetupPlayerInputComponent` function’s implementation and adding two calls to the `BindAction` function, the first one of which passes as parameters the `IA_Walk` property, the `ETriggerEvent::Started` event, the `this` pointer, and this class’s `BeginWalking` function, while the second passes the `IA_Walk` property, the `ETriggerEvent::Completed` event, the `this` pointer, and this class’s `StopWalking` function.
2.  Compile your code, open the editor, open your `BP_MyTPS` Blueprint asset, and set the **IA_Walk** property to the **IA_Walk** Input Action.

After following these steps, you should be able to have your character walk, which decreases its speed and slightly changes its animation, by pressing either the keyboard’s *Left Shift* key or the controller’s **Face Button Right** key:

![Figure 4.20 – The character running (left) and walking (right)](img/Figure_4.20_B18531.jpg)

Figure 4.20 – The character running (left) and walking (right)

And that concludes our activity. Our character should now be able to walk slowly as long as the player is holding the **Walk** Input Action.

Note

The solution to this activity can be found on GitHub here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions).

# Summary

In this chapter, you’ve learned how to create and modify **Input Actions**, as well as add their mappings to an **Input Context**, which gives you some flexibility when determining which keys trigger a specific action or axis, how to listen to them, and how to execute in-game logic when they’re pressed and released.

Now that you know how to handle the player’s input, you can allow the player to interact with your game and offer the agency that video games are so well known for.

In the next chapter, we’ll start making our own game from scratch. It’ll be called **Dodgeball** and will consist of the player controlling a character trying to run away from enemies that are throwing dodgeballs at it. In that chapter, we will have the opportunity to start learning about many important topics, with a heavy focus on collisions.