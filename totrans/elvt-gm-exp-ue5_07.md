# 7

# Working with UE5 Utilities

In the previous chapter, we learned about the remaining collision-related concepts in UE5, such as collision events, object types, physics simulation, and collision components. We learned how to have objects collide against one another, changing their responses to different collision channels, as well as how to create collision presets, spawn actors, and use timers.

In this chapter, we will look at several UE5 utilities that will allow you to easily move logic from one project to another and keep your project well-structured and organized, which will make life much easier for you in the long run and also make it easier for other people in your team to understand your work and modify it in the future. Game development is a tremendously hard task and is rarely done individually, but rather in teams, so it’s important to take these things into account when building your projects.

We’ll cover the following topics in this chapter:

*   Good practices – loose coupling
*   Blueprint Function Libraries
*   Actor components
*   Exploring interfaces
*   Blueprint native events

# Technical requirements

The project for this chapter can be found in the Chapter07 folder of the code bundle for this book, which can be downloaded here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition).

# Good practices – loose coupling

We can use Blueprint Function Libraries to move some generic functions in our project from a specific actor to a Blueprint Function Library so that they can be used in other parts of our project’s logic.

We will use Actor components to move part of some actor classes’ source code into an Actor component so that we can easily use that logic in other projects. This will keep our project loosely coupled. Loose coupling is a software engineering concept that refers to having your project structured in such a way that you can easily remove and add things as you need. The reason you should strive for loose coupling is if you want to reuse parts of one of your projects for another project. As a game developer, loose coupling will allow you to do that much more easily.

A practical example of how you could apply loose coupling is if you had a player character class that was able to fly and also had an inventory that contained several usable items. Instead of implementing the code responsible for both of those things in that player character class, you would implement the logic for each of them in separate Actor components, which you would then add to the class. This will not only make it easier to add and remove things that this class will do, by simply adding and removing the Actor components responsible for those things, but also allow you to reuse those Actor components in other projects where you have a character that has an inventory or can fly. This is one of the main purposes of Actor components.

Interfaces, much like Actor components, make our project better structured and organized.

Let’s start by talking about Blueprint Function Libraries.

# Blueprint Function Libraries

In UE5, there’s a class called `BlueprintFunctionLibary` that is meant to contain a collection of static functions that don’t belong to any specific actor and can be used in multiple parts of your project.

For instance, some of the objects that we used previously, such as the `GameplayStatics` object and `Kismet` libraries such as `KismetMathLibrary` and `KismetSystemLibrary`, are Blueprint Function Libraries. These contain functions that can be used in any part of your project.

There is at least one function in our project that’s been created by us that can be moved to a Blueprint Function Library: the `CanSeeActor` function defined in the `EnemyCharacter` class.

In the first exercise of this chapter, we will create a Blueprint Function Library so that we can move the `CanSeeActor` function from the `EnemyCharacter` class to the `BlueprintFunctionLibrary` class.

## Exercise 7.01 – moving the CanSeeActor function to the Blueprint Function Library

In this exercise, we will be moving the `CanSeeActor` function that we created for the `EnemyCharacter` class to a Blueprint Function Library.

The following steps will help you complete this exercise:

1.  Open Unreal Editor.
2.  *Right-click* inside the **Content Browser** area and select **New C++ Class**.
3.  Choose **BlueprintFunctionLibrary** as the parent class of this C++ class (you’ll find it by scrolling to the end of the panel).
4.  Name the new C++ class `DodgeballFunctionLibrary`.
5.  After the class’s files have been generated in Visual Studio, open them and close the editor.
6.  In the header file of `DodgeballFunctionLibrary`, add a declaration for a `public` function called `CanSeeActor`. This function will be similar to the one we created in the `EnemyCharacter` class; however, there will be some differences.

The new `CanSeeActor` function will be `static`, will return a `bool`, and will receive the following parameters:

*   A `const UWorld* World` property, which we will use to access the `Line Trace` functions.
*   An `FVector Location` property, which we will use as the location of the actor that is checking whether it can see the target actor.
*   A `const AActor* TargetActor` property, which will be the actor we’re checking visibility for.
*   A `TArray<const AActor*> IgnoreActors` property, which will specify the actors that should be ignored during the `Line Trace` functions. This property can have an empty array as a default argument:

    ```cpp
    public:
    // Can we see the given actor
    static bool CanSeeActor(
    const UWorld* World,
    FVector Location,
    const AActor* TargetActor,
    TArray<const AActor*> IgnoreActors = TArray<const AActor*>());
    ```

1.  Create the implementation of this function in the class’s source file and copy the implementation of the `EnemyCharacter` class’s version into this new class. Once you’ve done that, make the following modifications to the implementation:
    *   Change the value of the `Start` location of the `Line Trace` to the `Location` parameter:

        ```cpp
        // Where the Line Trace starts and ends
        FVector Start = Location;
        ```

    *   Instead of ignoring this actor (using the `this` pointer) and `TargetActor`, ignore the entire `IgnoreActors` array using the `AddIgnoredActors` function of `FCollisionQueryParams` and send that array as a parameter:

        ```cpp
        FCollisionQueryParams QueryParams;
        // Ignore the actors specified
        QueryParams.AddIgnoredActors(IgnoreActors);
        ```

    *   Replace both calls to the `GetWorld` function with the received `World` parameter:

        ```cpp
        // Execute the Line Trace
        World->LineTraceSingleByChannel(Hit, Start, End, Channel, 
          QueryParams);
        // Show the Line Trace inside the game
        DrawDebugLine(World, Start, End, FColor::Red);
        ```

    *   Add the necessary includes to the top of the `DodgeballFunctionLibrary` class, as shown in the following code snippet:

        ```cpp
        #include "Engine/World.h"
        #include "DrawDebugHelpers.h"
        #include "CollisionQueryParams.h"
        ```

2.  Once you’ve created the new version of the `CanSeeActor` function inside `DodgeballFunctionLibrary`, head to our `EnemyCharacter` class and make the following changes:
    *   Remove the declaration and implementation of the `CanSeeActor` function, inside its header and source file, respectively.
    *   Remove the `DrawDebugHelpers` include, given that we will no longer need that file:

        ```cpp
        // Remove this line
        #include "DrawDebugHelpers.h"
        ```

    *   Add an include for `DodgeballFunctionLibrary`:

        ```cpp
        #include "DodgeballFunctionLibrary.h"
        ```

    *   Inside the class’s `LookAtActor` function, just before the `if` statement that calls the `CanSeeActor` function, declare a `const TArray<const AActor*> IgnoreActors` variable and set it to both the `this` pointer and the `TargetActor` parameter:

        ```cpp
        const TArray<const AActor*> IgnoreActors = {this, 
          TargetActor};
        ```

Note

Introducing the preceding code snippet may give you an IntelliSense error in Visual Studio. You can safely ignore it, as your code should compile with no issues regardless.

1.  Replace the existing call to the `CanSeeActor` function with the one we just created by sending the following as parameters:
    *   The current world, through the `GetWorld` function
    *   The `SightSource` component’s location, using its `GetComponentLocation` function
    *   The `TargetActor` parameter
    *   The `IgnoreActors` array we just created:

        ```cpp
        if (UDodgeballFunctionLibrary::CanSeeActor(
          GetWorld(),
          SightSource->GetComponentLocation(),
          TargetActor,
          IgnoreActors))
        ```

Now that you’ve made all those changes, compile your code, open your project, and verify that the `EnemyCharacter` class still looks at the player as it walks around, so long as it’s in the enemy character’s sight, as shown in the following screenshot:

![Figure 7.1 – The enemy character still looking at the player character ](img/Figure_7.01_B18531.jpg)

Figure 7.1 – The enemy character still looking at the player character

And that concludes our exercise. We’ve put our `CanSeeActor` function inside a Blueprint Function Library and can now reuse it for other actors that require the same type of functionality.

The next step in our project is going to be learning more about Actor components and how we can use them to our advantage. Let’s take a look.

# Actor components

As we saw in the first few chapters of this book, Actors are the main way to create logic in UE5\. However, we’ve also seen that Actors can contain several Actor components.

Actor components are objects that can be added to an Actor and can have multiple types of functionality, such as being responsible for a character’s inventory or making a character fly. Actor components must always belong to and live inside an Actor, which is referred to as their **Owner**.

There are several different types of existing Actor components. Some of these are as follows:

*   Code-only Actor components, which act as their own class inside an actor. They have their own properties and functions and can both interact with the Actor they belong to and be interacted with by it.
*   Mesh components, which are used to draw several types of Mesh objects (Static Meshes, Skeletal Meshes, and so on).
*   Collision components, which are used to receive and generate collision events.
*   Camera components.

This leaves us with two main ways to add logic to our Actors: directly in the `Actor` class or through `Actor` components. To follow good software development practices, namely loose coupling (mentioned previously), you should strive to use Actor components instead of placing logic directly inside an Actor whenever possible. Let’s take a look at a practical example to understand the usefulness of Actor components.

Let’s say you’re making a game where you have the player character and enemy characters, both of which have health, and where the player character must fight enemies, who can also fight back. If you had to implement the health logic, which includes gaining health, losing health, and tracking the character’s health, you’d have two options:

*   You can implement the health logic in a base character class, from which both the player character class and the enemy character class would inherit.
*   You can implement the health logic in an Actor component and add that component to both the player character and enemy character classes separately.

There are a few reasons why the first option is not a good option, but the main one is this: if you wanted to add another piece of logic to both character classes (for example, stamina, which would limit the strength and frequency of the characters’ attacks), doing so using the same approach of a base class wouldn’t be a viable option. Given that, in UE5, C++ classes can only inherit from one class and there’s no such thing as multiple inheritance, that would be very hard to manage. It would also only get more complicated and unmanageable the more logic you decided to add to your project.

With that said, when adding logic to your project that can be encapsulated in a separate component, allowing you to achieve loose coupling, you should always do so.

Now, let’s create a new Actor component that will be responsible for keeping track of an actor’s health, as well as gaining and losing that health.

## Exercise 7.02 – creating the HealthComponent Actor component

In this exercise, we will be creating a new actor component responsible for gaining, losing, and keeping track of an actor’s health (its Owner).

For the player to lose, we’ll have to make the player character lose health and then end the game when it runs out of health. We’ll want to put this logic inside an actor component so that we can easily add all this health-related logic to other actors if we need to.

The following steps will help you complete the exercise:

1.  Open the editor and create a new C++ class, whose parent class will be the `ActorComponent` class. Its name will be `HealthComponent`.
2.  Once this class has been created and its files have been opened in Visual Studio, go to its header file and add a protected `float` property called `Health`, which will keep track of the Owner’s current health points. Its default value can be set to the number of health points its Owner will start the game with. In this case, we’ll initialize it with a value of `100` health points:

    ```cpp
    // The Owner's initial and current amount health 
    // points
    UPROPERTY(EditDefaultsOnly, Category = Health)
    float Health = 100.f;
    ```

3.  Create a declaration for the function that’s responsible for taking health away from its Owner. This function should be `public`; return nothing; receive a `float Amount` property as input, which indicates how many health points its Owner should lose; and be called `LoseHealth`:

    ```cpp
    // Take health points from its Owner
    void LoseHealth(float Amount);
    ```

Now, in the class’s source file, let’s start by notifying it that it should never use the `Tick` event so that its performance can be slightly improved.

1.  Change the `bCanEverTick` property’s value to `false` inside the class’s constructor:

    ```cpp
    PrimaryComponentTick.bCanEverTick = false;
    ```

2.  Create the implementation for our `LoseHealth` function, where we’ll start by removing the `Amount` parameter’s value from our `Health` property:

    ```cpp
    void UHealthComponent::LoseHealth(float Amount)
    {
      Health -= Amount;
    }
    ```

3.  Now, in that same function, we’ll check whether the current amount of health is less than or equal to `0`, which means that it has run out of health points (*has died or been destroyed*):

    ```cpp
    if (Health <= 0.f)
    {
    }
    ```

4.  If the `if` statement is true, we’ll do the following things:
    *   Set the `Health` property to `0` to make sure that our Owner doesn’t have negative health points:

        ```cpp
        Health = 0.f;
        ```

    *   Quit the game, the same way we did in [*Chapter 6*](B18531_06.xhtml#_idTextAnchor134), *Setting Up* *Collision Objects*, when creating the `VictoryBox` class:

        ```cpp
        UKismetSystemLibrary::QuitGame(this,
                                      nullptr,
                                      EQuitPreference::Quit,
                                      true);
        ```

    *   Don’t forget to include the `KismetSystemLibrary` object:

        ```cpp
        #include "Kismet/KismetSystemLibrary.h"
        ```

With this logic done, whenever any actor that has `HealthComponent` runs out of health, the game will end. This isn’t exactly the behavior we want in our **Dodgeball** game. However, we’ll change it when we talk about interfaces later in this chapter.

In the next exercise, we’ll be making the necessary modifications to some classes in our project to accommodate our newly created `HealthComponent`.

## Exercise 7.03 – integrating the HealthComponent Actor component

In this exercise, we will be modifying our `DodgeballProjectile` class so that it damages the player’s character when it comes into contact with it, as well as the `DodgeballCharacter` class so that it has a `HealthComponent`.

Open the `DodgeballProjectile` class’s files in Visual Studio and make the following modifications:

1.  In the class’s header file, add a protected `float` property called `Damage` and set its default value to `34` so that our player character will lose all of its health points after being hit three times. This property should be a `UPROPERTY` and have the `EditAnywhere` tag so that you can easily change its value in its Blueprint class:

    ```cpp
    // The damage the dodgeball will deal to the player's 
      character
    UPROPERTY(EditAnywhere, Category = Damage)
    float Damage = 34.f;
    ```

In the class’s source file, we’ll have to make some modifications to the `OnHit` function.

1.  Since we’ll be using the `HealthComponent` class, we’ll have to add the `include` statement for it:

    ```cpp
    #include "HealthComponent.h"
    ```

2.  The existing cast that is being done for `DodgeballCharacter` from the `OtherActor` property, which we did in *s**tep 17* of *Exercise 6.01 – creating the Dodgeball class*, and is inside the `if` statement, should be done before that `if` statement and be saved inside a variable. Then, you should check whether that variable is `nullptr`. We are doing this to access the player character’s `HealthComponent` inside the `if` statement:

    ```cpp
    ADodgeballCharacter* Player = 
      Cast<ADodgeballCharacter>(OtherActor);
    if (Player != nullptr)
    {
    }
    ```

3.  If the `if` statement is true (that is, if the actor we hit is the player’s character), we want to access that character’s `HealthComponent` and reduce the character’s health. To access `HealthComponent`, we must call the character’s `FindComponentByClass` function and send the `UHealthComponent` class as a template parameter (to indicate the class of the component we want to access):

    ```cpp
    UHealthComponent* HealthComponent = Player->
    FindComponentByClass<UHealthComponent>();
    ```

Note

The `FindComponentByClass` function, included in the `Actor` class, will return a reference(s) to the actor component(s) of a specific class that the actor contains. If the function returns `nullptr`, that means the actor doesn’t have an Actor component of that class.

You may also find the `GetComponents` function inside the `Actor` class useful, which will return a list of all the Actor components inside that actor.

1.  After that, check whether `HealthComponent` is `nullptr`. If it isn’t, we’ll call its `LoseHealth` function and send the `Damage` property as a parameter:

    ```cpp
    if (HealthComponent != nullptr)
    {
      HealthComponent->LoseHealth(Damage);
    }
    Destroy();
    ```

2.  Make sure the existing `Destroy` function is called after doing the null check for `HealthComponent`, as shown in the previous code snippet.

Before we finish this exercise, we’ll need to make some modifications to our `DodgeballCharacter` class. Open the class’s files in Visual Studio.

1.  In the class’s header file, add a `private` property of the `class UhealthComponent*` type called `HealthComponent`:

    ```cpp
    class UHealthComponent* HealthComponent;
    ```

2.  In the class’s source file, add an `include` statement to the `HealthComponent` class:

    ```cpp
    #include "HealthComponent.h"
    ```

3.  At the end of the class’s constructor, create `HealthComponent` by using the `CreateDefaultSubobject` function and name it `HealthComponent`:

    ```cpp
    HealthComponent = 
      CreateDefaultSubobject<UHealthComponent>(
      TEXT("Health 
      Component"));
    ```

Once you’ve made all these changes, compile your code and open the editor. When you play the game, if you let your player character get hit by a dodgeball three times, you’ll notice that the game abruptly stops, as intended:

![Figure 7.2 – The enemy character throwing dodgeballs at the player character ](img/Figure_7.02_B18531.jpg)

Figure 7.2 – The enemy character throwing dodgeballs at the player character

Once the game is stopped, it will look as follows:

![Figure 7.3 – The editor after the player character runs out of health points and the game stops ](img/Figure_7.03_B18531.jpg)

Figure 7.3 – The editor after the player character runs out of health points and the game stops

And that completes this exercise. You now know how to create Actor components and how to access an actor’s Actor components. This is a very important step toward making your game projects more understandable and better structured, so good job.

Now that we’ve learned about Actor components, let’s learn about another way to make our projects better structured and organized: by using interfaces.

# Exploring interfaces

There’s a chance that you may already know about interfaces, given that other programming languages, such as Java, already have them. If you do, they work pretty similarly in UE5, but if you don’t, let’s see how they work, taking the example of the `HealthComponent` class we created.

As you saw in the previous exercise, when the `Health` property of the `HealthComponent` class reaches `0`, that component will simply end the game. However, we don’t want that to happen every time an actor’s health points run out: some actors may simply be destroyed, some may notify another actor that they have run out of health points, and so on. We want each actor to be able to determine what happens to them when they run out of health points. But how can we handle this?

Ideally, we would simply call a specific function that belongs to Owner of the `HealthComponent` class, which would then choose how to handle the fact that Owner has run out of health points. But in which class should you implement that function, given that our Owner can be of any class, so long as it inherits from the Actor class? As we discussed at the beginning of this chapter, having a class that’s responsible just for this would quickly become unmanageable. Luckily for us, interfaces solve this problem.

Interfaces are classes that contain a collection of functions that an object must have if it implements that interface. It essentially works as a contract that the object signs, saying that it will implement all the functions present on that interface. Then, you can simply check whether an object implements a specific interface and call the object’s implementation of the function defined in the interface.

In our specific case, we’ll want to have an interface that has a function that will be called when an object runs out of health points so that our `HealthComponent` class can check whether its Owner implements that interface and then call that function from the interface. This will make it easy for us to specify how each actor behaves when running out of health points: some actors may simply be destroyed, others may trigger an in-game event, and others may simply end the game (which is the case with our player character).

However, before we create our first interface, we should talk a bit about Blueprint native events.

# Blueprint native events

When using the `UFUNCTION` macro in C++, you can turn a function into a Blueprint native event by simply adding the `BlueprintNativeEvent` tag to that macro.

So, what is a Blueprint native event? It’s an event that is declared in C++ that can have a default behavior, which is also defined in C++, but that can be overridden in Blueprint. Let’s declare a Blueprint native event called `MyEvent` by declaring a `MyEvent` function using the `UFUNCTION` macro with the `BlueprintNativeEvent` tag, followed by the virtual `MyEvent_Implementation` function:

```cpp
UFUNCTION(BlueprintNativeEvent)
void MyEvent();
virtual void MyEvent_Implementation();
```

The reason why you have to declare these two functions is that the first one is the Blueprint signature, which allows you to override the event in Blueprint, while the second one is the C++ signature, which allows you to override the event in C++.

The C++ signature is simply the name of the event followed by `_Implementation`, and it should always be a `virtual` function. Given that you declared this event in C++, to implement its default behavior, you must implement the `MyEvent_Implementation` function, not the `MyEvent` function (that one should remain untouched). To call a Blueprint native event, you can simply call the normal function without the `_Implementation` suffix; in this case, `MyEvent()`.

In the next exercise, we’ll learn how to use Blueprint native events in practice, where we’ll create a new interface.

## Exercise 7.04 – creating the HealthInterface class

In this exercise, we will be creating an interface that’s responsible for handling how an object behaves when it runs out of health points.

To do this, follow these steps:

1.  Open the editor and create a new C++ class that inherits from `Interface` (called `Unreal Interface` in the scrollable menu) and call it `HealthInterface`.
2.  Once the class’s files have been generated and opened in Visual Studio, go to the newly created class’s header file. You’ll notice that the generated file has two classes – `UHealthInterface` and `IHealthInterface`.
3.  These will be used in combination when checking whether an object implements the interface and calls its functions. However, you should only add function declarations in the class prefixed with `I` – in this case, `IHealthInterface`. Add a `public` Blueprint native event called `OnDeath` that returns nothing and receives no parameters. This function will be called when an object runs out of health points:

    ```cpp
    UFUNCTION(BlueprintNativeEvent, Category = Health)
    void OnDeath();
    virtual void OnDeath_Implementation() = 0;
    ```

Note that the `OnDeath_Implementation` function declaration needs its own implementation. However, there is no need for the interface to implement that function because it would simply be empty. To notify the compiler that this function has no implementation in this class, we added `= 0` to the end of its declaration.

1.  Go to the `DodgeballCharacter` class’s header file. We’ll want this class to implement our newly created `HealthInterface`, but how do we do that? The first thing we have to do is include the `HealthInterface` class. Make sure you include it before the `.generated.h` `include` statement:

    ```cpp
    // Add this include
    #include "HealthInterface.h"
    #include "DodgeballCharacter.generated.h"
    ```

2.  Then, replace the line in the header file that makes the `DodgeballCharacter` class inherit from the `Character` class with the following line, which will make this class implement `HealthInterface`:

    ```cpp
    class ADodgeballCharacter : public ACharacter, public 
      IHealthInterface
    ```

3.  The next thing we have to do is implement the `OnDeath` function in the `DodgeballCharacter` class. To do this, add a declaration for the `OnDeath_Implementation` function that overrides the interface’s C++ signature. This function should be `public`. To override a `virtual` function, you must add the `override` keyword to the end of its declaration:

    ```cpp
    virtual void OnDeath_Implementation() override;
    ```

4.  In this function’s implementation, within the class’s source file, simply quit the game, the same way that is being done in the `HealthComponent` class:

    ```cpp
    void ADodgeballCharacter::OnDeath_Implementation()
    {
      UKismetSystemLibrary::QuitGame(this,
                                    nullptr,
                                    EQuitPreference::Quit,
                                    true);
    }
    ```

5.  Because we’re now using `KismetSystemLibrary`, we’ll have to include it:

    ```cpp
    #include "Kismet/KismetSystemLibrary.h"
    ```

6.  Now, we must go to our `HealthComponent` class’s source file. Because we’ll no longer be using `KistemSystemLibrary` and will be using the `HealthInterface` instead, replace the `include` statement for the first class with an `include` statement for the second one:

    ```cpp
    // Replace this line
    #include "Kismet/KismetSystemLibrary.h"
    // With this line
    #include "HealthInterface.h"
    ```

7.  Then, change the logic that is responsible for quitting the game when Owner runs out of health points. Instead of doing this, we’ll want to check whether Owner implements `HealthInterface` and, if it does, call its implementation of the `OnDeath` function. Remove the existing call to the `QuitGame` function:

    ```cpp
    // Remove this
    UKismetSystemLibrary::QuitGame(this,
                                  nullptr,
                                  EQuitPreference::Quit,
                                  true);
    ```

8.  To check whether an object implements a specific interface, we can call that object’s `Implements` function, using the interface’s class as a template parameter. The class of the interface that you should use in this function is the one that is prefixed with `U`:

    ```cpp
    if (GetOwner()->Implements<UHealthInterface>())
    {
    }
    ```

9.  Because we’ll be using methods that belong to the `Actor` class, we’ll also need to include it:

    ```cpp
    #include "GameFramework/Actor.h"
    ```

If this `if` statement is true, that means that our Owner implements `HealthInterface`. In this case, we’ll want to call its implementation of the `OnDeath` function.

1.  To do this, call it through the interface’s class (this time, the one that is prefixed with `I`). The function inside the interface that you’ll want to call is `Execute_OnDeath` (note that the function you should call inside the interface will always be its normal name prefixed with `Execute_`). This function must receive at least one parameter, which is the object that the function will be called on and that implements that interface; in this case, Owner:

    ```cpp
    if (GetOwner()->Implements<UHealthInterface>())
    {
      IHealthInterface::Execute_OnDeath(GetOwner());
    }
    ```

Note

If your interface’s function receives parameters, you can send them in the function call after the first parameter mentioned in the preceding step. For instance, if our `OnDeath` function received an `int` property as a parameter, you would call it with `IHealthInterface::Execute_OnDeath(GetOwner(), 5)`.

The first time you try to compile your code after adding a new function to an interface and then calling `Execute_ version`, you may get an `Intellisense` error. You can safely ignore this error.

Once you’ve made all these changes, compile your code and open the editor. When you play the game, try letting the character get hit by three dodgeballs:

![Figure 7.4 – The enemy character throwing dodgeballs at the player character ](img/Figure_7.04_B18531.jpg)

Figure 7.4 – The enemy character throwing dodgeballs at the player character

If the game ends after that, then that means that all our changes worked and the game’s logic remains the same:

![Figure 7.5 – The editor after the player character runs out of health points and the game stops ](img/Figure_7.05_B18531.jpg)

Figure 7.5 – The editor after the player character runs out of health points and the game stops

And with that, we conclude this exercise. You now know how to use interfaces. The benefit of the change that we just made is that we can now have other actors that lose health, as well as specify what happens when they run out of health points by using the `Health` interface.

Now, we will complete an activity where we’ll move all of the logic related to the `LookAtActor` function to its own Actor component and use it to replace the `SightSource` component we created.

## Activity 7.01 – moving the LookAtActor logic to an Actor component

In this activity, we’ll be moving all of the logic related to the `LookAtActor` function, inside the `EnemyCharacter` class, to its own Actor component (similarly to how we moved the `CanSeeActor` function to a Blueprint Function Library). This way, if we want an actor (that isn’t an `EnemyCharacter`) to look at another actor, we will simply be able to add this component to it.

Follow these steps to complete this activity:

1.  Open the editor and create a new C++ class that inherits from `SceneComponent`, called `LookAtActorComponent`.

Head to the class’s files, which are open in Visual Studio.

1.  Go to its header file and add a declaration for the `LookAtActor` function, which should be `protected`, return a `bool`, and receive no parameters.

Note

While the `LookAtActor` function of `EnemyCharacter` received the `AActor* TargetActor` parameter, this Actor component will have its `TargetActor` as a class property, which is why we won’t need to receive it as a parameter.

1.  Add a protected `AActor*` property called `TargetActor`. This property will represent the actor we want to look at.
2.  Add a protected `bool` property called `bCanSeeTarget`, with a default value of `false`, which will indicate whether `TargetActor` can be seen.
3.  Add a declaration for a public `FORCEINLINE` function, as covered in [*Chapter 6*](B18531_06.xhtml#_idTextAnchor134), *Setting Up Collision Objects*, called `SetTarget`, which will return nothing and receive `AActor* NewTarget` as a parameter. The implementation of this function will simply set the `TargetActor` property to the value of the `NewTarget` property.
4.  Add a declaration for a public `FORCEINLINE` function called `CanSeeTarget`, which will be `const`, return a `bool`, and receive no parameters. The implementation of this function will simply return the value of the `bCanSeeTarget` property.

Now, go to the class’s source file.

1.  In the class’s `TickComponent` function, set the value of the `bCanSeeTarget` property to the return value of the `LookAtActor` function call.
2.  Add an empty implementation of the `LookAtActor` function and copy the `EnemyCharacter` class’s implementation of the `LookAtActor` function into the implementation of `LookAtActorComponent`.
3.  Make the following modifications to the `LookAtActorComponent` class’s implementation of the `LookAtActor` function:
    1.  Change the first element of the `IgnoreActors` array to the Actor’s component’s Owner.
    2.  Change the second parameter of the `CanSeeActor` function call to this component’s location.
    3.  Change the value of the `Start` property to the location of Owner.

Finally, replace the call to the `SetActorRotation` function with a call to the `SetActorRotation` function of Owner.

1.  Because of the modifications we’ve made to the implementation of the `LookAtActor` function, we’ll need to add some includes to our `LookAtActorComponent` class and remove some includes from our `EnemyCharacter` class. Remove the includes to `KismetMathLibrary` and `DodgeballFunctionLibrary` from the `EnemyCharacter` class and add them to the `LookAtActorComponent` class.

We’ll also need to add an include to the `Actor` class since we’ll be accessing several functions belonging to that class.

Now, let’s make some further modifications to our `EnemyCharacter` class:

1.  In its header file, remove the declaration of the `LookAtActor` function.
2.  Replace the `SightSource` property with a property of the `UlookAtActorComponent*` type called `LookAtActorComponent`.
3.  In the class’s source file, add an include to the `LookAtActorComponent` class.
4.  Inside the class’s constructor, replace the references to the `SightSource` property with a reference to the `LookAtActorComponent` property. Additionally, the `CreateDefaultSubobject` function’s template parameter should be the `ULookAtActorComponent` class and its parameter should be `“Look At Actor Component”`.
5.  Remove the class’s implementation of the `LookAtActor` function.
6.  In the class’s `Tick` function, remove the line of code where you create the `PlayerCharacter` property, and add that exact line of code to the end of the class’s `BeginPlay` function.
7.  After this line, call the `SetTarget` function of `LookAtActorComponent` and send the `PlayerCharacter` property as a parameter.
8.  Inside the class’s `Tick` function, set the `bCanSeePlayer` property’s value to the return value of the `CanSeeTarget` function call of `LookAtActorComponent`, instead of the return value of the `LookAtActor` function call.

Now, there’s only one last step we have to do before this activity is completed.

1.  Close the editor (if you have it opened), compile your changes in Visual Studio, open the editor, and open the `BP_EnemyCharacter` Blueprint. Find `LookAtActorComponent` and change its location to `(10, 0, 80)`.

**Expected output**:

![Figure 7.6 – The enemy character looking at the player character remains functional ](img/Figure_7.06_B18531.jpg)

Figure 7.6 – The enemy character looking at the player character remains functional

And with that, we conclude our activity. You have applied your knowledge of refactoring part of an actor’s logic into an Actor component so that you can reuse it in other parts of your project, or even in other projects of your own.

Note

The solution for this activity can be found on [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions).

# Summary

You now know about several utilities that will help you keep your projects more organized and allow you to reuse the things that you make.

You learned how to create a Blueprint Function Library, create Actor components and use them to refactor the existing logic in your project, and create interfaces and call functions from an object that implements a specific interface. Altogether, these new topics will allow you to refactor and reuse all the code that you write in a project in that same project or another project.

In the next chapter, we’ll look at UMG, UE5’s system for creating user interfaces, and learn how to create user interfaces.