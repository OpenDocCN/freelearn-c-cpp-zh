# 17

# Using Remote Procedure Calls

In the previous chapter, we covered some critical multiplayer concepts, including the server-client architecture, connections and ownership, roles, and variable replication. We also learned how to make 2D Blend Spaces and use the `Transform (Modify) Bone` node to modify bones at runtime. We used that knowledge to create a basic first-person shooter character that walks, jumps, and looks around.

In this chapter, we’re going to cover **remote procedure calls** (**RPCs**), which is another important multiplayer concept that allows the server to execute functions on the clients and vice versa. So far, we’ve learned about variable replication as a form of communication between the server and the clients. However, to have proper communication, this isn’t enough. This is because the server may need to execute specific logic on the clients that doesn’t involve updating the value of a variable. The client also needs a way to tell its intentions to the server so that the server can validate the action and let the other clients know about it. This will ensure that the multiplayer world is synchronized between all of the connected clients. We’ll also cover how to use enumerations and expose them to the editor, as well as array index wrapping, which allows you to iterate an array in both directions and loop around when you go beyond its limits.

In this chapter, we’re going to cover the following main topics:

*   Understanding remote procedure calls
*   Exposing enumerations to the editor
*   Using array index wrapping

By the end of this chapter, you’ll understand how RPCs work to make the server and the clients execute logic on one another. You’ll also be able to expose enumerations to the editor and use array index wrapping to cycle through arrays in both directions.

# Technical requirements

For this chapter, you will need the following technical requirements:

*   Unreal Engine 5 installed
*   Visual Studio 2019 installed

The project for this chapter can be found in the `Chapter17` folder of the code bundle for this book, which can be downloaded here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition).

In the next section, we will look at RPCs.

# Understanding remote procedure calls

We covered variable replication in [*Chapter 16*](B18531_16.xhtml#_idTextAnchor345), *Getting Started with Multiplayer Basics*, and, while a very useful feature, it is a bit limited in terms of allowing custom code to be executed in remote game instances (client-to-server or server-to-client) for two main reasons:

*   The first reason is that variable replication is strictly a form of server-to-client communication, so there isn’t a way for a client to use variable replication to tell the server to execute some custom logic by changing the value of a variable.
*   The second reason is that variable replication, as the name suggests, is driven by the values of variables, so even if variable replication allowed client-to-server communication, it would require you to change the value of a variable on the client to trigger a `RepNotify` function on the server to run the custom logic, which is not very practical.

To solve this problem, Unreal Engine supports RPCs, which work just like normal functions that can be defined and called. However, instead of executing them locally, they will be executed on a remote game instance, without being tied to a variable. To be able to use RPCs, make sure you are defining them in an actor that has a valid connection and replication turned on.

There are three types of RPCs, and each one serves a different purpose:

*   Server RPC
*   Multicast RPC
*   Client RPC

Let’s look at these three types in detail and explain when to use them.

## Server RPC

You use a Server RPC every time you want the server to run a function on the actor that has defined the RPC. There are two main reasons why you would want to do this:

*   The first reason is security. When making multiplayer games, especially competitive ones, you always have to assume that the client will try to cheat. The way to make sure there is no cheating is by forcing the client to go through the server to execute the functions that are critical to gameplay.
*   The second reason is synchronicity. Since the critical gameplay logic is only executed on the server, the important variables are only going to be changed there, which will automatically trigger the variable replication logic to update the clients whenever they are changed.

An example of this would be when a client’s character tries to fire a weapon. Since there’s always the possibility that the client may try to cheat, you can’t just execute the fire weapon logic locally. The correct way of doing this is by having the client call a Server RPC that tells the server to validate the `Fire` action by making sure the character has enough ammo, has the weapon equipped, and so on. If everything checks out, then it will deduct the ammo variable, and finally, it will execute a Multicast RPC (covered shortly) that will tell all of the clients to play the fire animation on that character.

### Declaration

To declare a Server RPC, you can use the `Server` specifier on the `UFUNCTION` macro. Have a look at the following example:

```cpp
UFUNCTION(Server, Reliable, WithValidation)
void ServerRPCFunction(int32 IntegerParameter, float FloatParameter, AActor* ActorParameter); 
```

In the preceding code snippet, the `Server` specifier is used on the `UFUNCTION` macro to state that the function is a Server RPC. You can have parameters on a Server RPC just like a normal function, but with some caveats that will be explained later in this topic, as well as the purpose of the `Reliable` and `WithValidation` specifiers.

### Execution

To execute a Server RPC, you call it from a client on the actor instance that defined it. Take a look at the following example:

```cpp
void ARPCTest::CallMyOwnServerRPC(int32 IntegerParameter)
{
  ServerMyOwnRPC(IntegerParameter);
}
```

The preceding code snippet implements the `CallMyOwnServerRPC` function, which calls the `ServerMyOwnRPC` RPC function, defined in its own `ARPCTest` class, with an integer parameter. This will execute the implementation of the `ServerMyOwnRPC` function on the server version of that actor’s instance. We can also call a Server RPC from another actor’s instance, like so:

```cpp
void ARPCTest::CallServerRPCOfAnotherActor(AAnotherActor* OtherActor)
{
  if(OtherActor != nullptr)
  {
    OtherActor->ServerAnotherActorRPC();
  }
}
```

The preceding code snippet implements the `CallServerRPCOfAnotherActor` function, which calls the `ServerAnotherActorRPC` RPC function, defined in `AAnotherActor`, on the `OtherActor` instance, so long as it’s valid. This will execute the implementation of the `ServerAnotherActorRPC` function on the server version of the `OtherActor` instance.

## Multicast RPC

You use a Multicast RPC when you want the server to instruct all of the clients to run a function on the actor that has defined the RPC.

An example of this is when a client’s character tries to fire a weapon. After the client calls the Server RPC to ask permission to fire the weapon and the server has validated the request (the ammo has been deducted and the line trace/projectile was processed), we need to do a Multicast RPC so that all of the instances of that specific character play the fire animation.

### Declaration

To declare a Multicast RPC, you need to use the `NetMulticast` specifier on the `UFUNCTION` macro. Have a look at the following example:

```cpp
UFUNCTION(NetMulticast, Unreliable)
void MulticastRPCFunction(int32 IntegerParameter, float 
FloatParameter, AActor* ActorParameter); 
```

In the preceding code snippet, the `NetMulticast` specifier is used on the `UFUNCTION` macro to say that the function is a Multicast RPC. You can have parameters on a Multicast RPC just like a normal function, but with the same caveats as the Server RPC. The `Unreliable` specifier will be explained later in this topic.

### Execution

To execute a Multicast RPC, you must call it from the server on the actor instance that defined it. Take a look at the following example:

```cpp
void ARPCTest::CallMyOwnMulticastRPC(int32 IntegerParameter)
{
  MulticastMyOwnRPC(IntegerParameter);
}
```

The preceding code snippet implements the `CallMyOwnMulticastRPC` function, which calls the `MulticastMyOwnRPC` RPC function, defined in its own `ARPCTest` class, with an integer parameter. This will execute the implementation of the `MulticastMyOwnRPC` function on all of the clients’ versions of that actor’s instance. We can also call a Multicast RPC from another actor’s instance, like so:

```cpp
void ARPCTest::CallMulticastRPCOfAnotherActor(AAnotherActor* 
OtherActor)
{
  if(OtherActor != nullptr)
  {
    OtherActor->MulticastAnotherActorRPC();
  }
}
```

The preceding code snippet implements the `CallMulticastRPCOfAnotherActor` function, which calls the `MulticastAnotherActorRPC` RPC function, defined in `AAnotherActor`, on the `OtherActor` instance, so long as it’s valid. This will execute the implementation of the `MulticastAnotherActorRPC` function on all of the clients’ versions of the `OtherActor` instance.

## Client RPC

You use a Client RPC when you want the server to instruct only the owning client to run a function on the actor that has defined the RPC. To set the owning client, you need to call `SetOwner` on the server and set it with the client’s player controller.

An example of this would be when a character is hit by a projectile and plays a pain sound that only that client will hear. By calling a Client RPC from the server, the sound will only be played on the owning client and not on the other clients.

### Declaration

To declare a Client RPC, you need to use the `Client` specifier on the `UFUNCTION` macro. Have a look at the following example:

```cpp
UFUNCTION(Client, Unreliable)
void ClientRPCFunction(int32 IntegerParameter, float FloatParameter, Aactor* ActorParameter); 
```

In the preceding code snippet, the `Client` specifier is being used on the `UFUNCTION` macro to say that the function is a Client RPC. You can have parameters on a Client RPC just like a normal function, but with the same caveats as the Server RPC and the Multicast RPC. The `Unreliable` specifier will be explained later in this topic.

### Execution

To execute a Client RPC, you must call it from the server on the actor instance that defined it. Take a look at the following example:

```cpp
void ARPCTest::CallMyOwnClientRPC(int32 IntegerParameter)
{
  ClientMyOwnRPC(IntegerParameter);
}
```

The preceding code snippet implements the `CallMyOwnClientRPC` function, which calls the `ClientMyOwnRPC` RPC function, defined in its own `ARPCTest` class, with an integer parameter. This will execute the implementation of the `ClientMyOwnRPC` function on the owning client’s version of that actor’s instance. We can also call a Client RPC from another actor’s instance, like so:

```cpp
void ARPCTest::CallClientRPCOfAnotherActor(AAnotherActor* OtherActor)
{
  if(OtherActor != nullptr)
  {
    OtherActor->ClientAnotherActorRPC();
  }
}
```

The preceding code snippet implements the `CallClientRPCOfAnotherActor` function, which calls the `ClientAnotherActorRPC` RPC function, defined in `AAnotherActor`, on the `OtherActor` instance, so long as it’s valid. This will execute the implementation of the `ClientAnotherActorRPC` function on the owning client’s version of the `OtherActor` instance.

## Important considerations when using RPCs

RPCs are very useful, but there are a couple of things that you need to take into consideration when using them.

### Implementation

The implementation of an RPC differs slightly from that of a typical function. Instead of implementing the function as you normally do, you should only implement the `_Implementation` version of it, even though you didn’t declare it in the header file. Have a look at the following examples.

**Server RPC**:

```cpp
void ARPCTest::ServerRPCTest_Implementation(int32 IntegerParameter, float FloatParameter, AActor* ActorParameter)
{
}
```

In the preceding code snippet, we implemented the `_Implementation` version of the `ServerRPCTest` function, which uses three parameters.

**Multicast RPC**:

```cpp
void ARPCTest::MulticastRPCTest_Implementation(int32 IntegerParameter, float FloatParameter, AActor* ActorParameter)
{
}
```

In the preceding code snippet, we implemented the `_Implementation` version of the `MulticastRPCTest` function, which uses three parameters.

**Client RPC**:

```cpp
void ARPCTest::ClientRPCTest_Implementation(int32 IntegerParameter, float FloatParameter, AActor* ActorParameter)
{
}
```

In the preceding code snippet, we implemented the `_Implementation` version of the `ClientRPCTest` function, which uses three parameters.

As you can see from the previous examples, independent of the type of the RPC you are implementing, you should only implement the `_Implementation` version of the function and not the normal one, as demonstrated in the following code snippet:

```cpp
void ARPCTest::ServerRPCFunction(int32 IntegerParameter, float FloatParameter, AActor* ActorParameter)
{
}
```

In the preceding code, we’re defining the normal implementation of `ServerRPCFunction`. If you implement the RPC like this, you’ll get an error saying that it was already implemented. The reason for this is that when you declare the RPC function in the header file, Unreal Engine will automatically create the normal implementation internally, which once called, will execute the logic to send the RPC request through the network and when it reaches the remote computer it will call the `_Implementation` version there. Since you cannot have two implementations of the same function, it will throw a compilation error. To fix this, just make sure that you only implement the `_Implementation` version of the RPC.

Next, we will look at name prefixes.

### Name prefixes

In Unreal Engine, it’s good practice to prefix RPCs with their corresponding types. Have a look at the following examples:

*   A `RPCFunction` should be named `ServerRPCFunction`
*   A `RPCFunction` should be named `MulticastRPCFunction`
*   A `RPCFunction` should be named `ClientRPCFunction`

### Return value

Since the execution of RPCs is typically executed on different machines asynchronously, you can’t have a return value, so it always needs to be void.

### Overriding

You can override the implementation of an RPC to expand or bypass the parent’s functionality by declaring and implementing the `_Implementation` function in the child class without the `UFUNCTION` macro. Let’s look at an example.

The following is the declaration of the parent class:

```cpp
UFUNCTION(Server, Reliable)
void ServerRPCTest(int32 IntegerParameter); 
```

In the preceding code snippet, we have the declaration of the `ServerRPCTest` function in the parent class, which uses one integer parameter.

If we want to override the function on the child class, we would need to use the following declaration:

```cpp
virtual void ServerRPCTest_Implementation(int32 IntegerParameter) override;
```

In the preceding code snippet, we have overridden the declaration of the `ServerRPCTest_Implementation` function in the child class header file. The implementation of the function is just like any other override, with the possibility of calling `Super::ServerRPCTest_Implementation` if you still want to execute the parent functionality.

### Valid connection

For an actor to be able to execute its RPCs, they need to have a valid connection. If you try to call an RPC on an actor that doesn’t have a valid connection, then nothing will happen on the remote instance. You must make sure that the actor is either a player controller, is being possessed by one (if applicable), or that its owning actor has a valid connection.

### Supported parameter types

When using RPCs, you can add parameters just like any other function. At the time of writing, most common types are supported (such as `bool`, `int32`, `float`, `FText`, `FString`, `FName`, `TArray`, and so on), but not all of them, such as `TSet` and `TMap`. Among the types that are supported, the ones that you have to pay more attention to are the pointers to any `UObject` class or subclass, especially actors.

If you create an RPC with an actor parameter, then that actor also needs to exist on the remote game instance; otherwise, it will have a value of `nullptr`. Another important thing to take into account is that the instance name of each version of the actor can be different. This means that if you call an RPC with an actor parameter, then the instance name of the actor when calling the RPC might be different than the one when executing the RPC on the remote instance. Here is an example to help you understand this:

![Figure 17.1 – Displaying the name of the character instances in three clients ](img/Figure_17.01_B18531.jpg)

Figure 17.1 – Displaying the name of the character instances in three clients

In the preceding example, you can see three clients running (one of them is a listen server) and each window is displaying the name of all of the character instances. If you look at the `BP_ThirdPersonCharacter_C_0`, but on the `BP_ThirdPersonCharacter_C_1`. This means that if `BP_ThirdPersonCharacter_C_0` as an argument, then when the RPC is executed on the server, the parameter will be `BP_ThirdPersonCharacter_C_1`, which is the instance name of the equivalent character in that game instance.

### Executing RPCs on the target machine

You can call RPCs directly on their target machine and they will still execute. In other words, you can call a Server RPC on the server and it will execute, as well as a Multicast/Client RPC on the client, but in the latter case, it will only execute the logic on the client that called the RPC. Either way, in these cases, you can call the `_Implementation` version directly instead, to execute the logic faster.

The reason for this is that the `_Implementation` version just holds the logic to execute and doesn’t have the overhead of creating and sending the RPC request through the network that the regular call has.

Have a look at the following example of an actor that has authority on the server:

```cpp
void ARPCTest::CallServerRPC(int32 IntegerParameter)
{
  if(HasAuthority())
  {
    ServerRPCFunction_Implementation(IntegerParameter);
  }
  else ServerRPCFunction(IntegerParameter);
}
```

In the preceding example, you have the `CallServerRPC` function, which calls `ServerRPCFunction` in two different ways. If the actor is already on the server, then it calls `ServerRPCFunction_Implementation`, which will skip the overhead, as mentioned previously.

If the actor is not on the server, then it executes the regular call by using `ServerRPCFunction`, which adds the required overhead for creating and sending the RPC request through the network.

### Validation

When you define an RPC, you have the option of using an additional function to check whether there are any invalid inputs before the RPC is called. This is used to avoid processing the RPC if the inputs are invalid due to cheating or for some other reason.

To use validation, you need to add the `WithValidation` specifier to the `UFUNCTION` macro. When you use that specifier, you will be forced to implement the `_Validate` version of the function, which will return a Boolean stating whether the RPC can be executed.

Have a look at the following example:

```cpp
UFUNCTION(Server, Reliable, WithValidation)
void ServerSetHealth(float NewHealth);
```

In the preceding code snippet, we’ve declared a validated Server RPC called `ServerSetHealth`, which takes a float parameter for the new value of `Health`. Take a look at its implementation:

```cpp
bool ARPCTest::ServerSetHealth_Validate(float NewHealth)
{
  return NewHealth >= 0.0f && NewHealth <= MaxHealth;
}
void ARPCTest::ServerSetHealth_Implementation(float NewHealth)
{
  Health = NewHealth;
}
```

In the preceding code snippet, we implemented the `_Validate` function, which will check whether the new health is within 0 and the maximum value of the health. If a client tries to hack and call `ServerSetHealth` with `200` and `MaxHealth` is `100`, then the RPC won’t be called, which prevents the client from changing the health with values outside a certain range. If the `_Validate` function returns `true`, the `_Implementation` function is called as usual, which sets `Health` with the value of `NewHealth`.

### Reliability

When you declare an RPC, you are required to either use the `Reliable` or `Unreliable` specifier in the `UFUNCTION` macro. Here’s a quick overview of what they do:

*   `Reliable`: This is used when you want to make sure the RPC is executed, by repeating the request until the remote machine confirms its reception. This should only be used for RPCs that are very important, such as executing critical gameplay logic. Here is an example of how to use it:

    ```cpp
    UFUNCTION(Server, Reliable)
    void ServerReliableRPCFunction(int32 IntegerParameter); 
    ```

*   `Unreliable`: This is used when you don’t care whether the RPC is executed due to bad network conditions, such as playing a sound or spawning a particle effect. This should only be used for RPCs that aren’t very important or are called very frequently to update values since it wouldn’t matter if a couple didn’t get through. Here is an example of how to use it:

    ```cpp
    UFUNCTION(Server, Unreliable)
    void ServerUnreliableRPCFunction(int32 IntegerParameter);
    ```

Note

For more information on RPCs, please visit [https://docs.unrealengine.com/en-US/Gameplay/Networking/Actors/RPCs/index.xhtml](https://docs.unrealengine.com/en-US/Gameplay/Networking/Actors/RPCs/index.xhtml).

In the following exercise, you will learn how to implement the different types of RPCs.

## Exercise 17.01 – Using remote procedure calls

In this exercise, we’re going to create a **C++** project that uses the **Third Person** template and we’re going to expand it in the following way:

*   Add a new `Ammo` integer variable that defaults to `5` and replicates to all of the clients.
*   Add a fire animation that plays a fire sound and also create a **Fire Anim Montage** that is played when the server tells the client that the request to fire was valid.
*   Add a **No Ammo Sound** that will play when the server tells the client that they didn’t have sufficient ammo.
*   Every time the player presses the left mouse button, the client will perform a reliable and validated Server RPC that will check whether the character has sufficient ammo. If it does, it will subtract `1` from the `Ammo` variable and call an unreliable Multicast RPC that plays the fire animation in every client. If it doesn’t have ammo, then it will execute an unreliable Client RPC that will play `No Ammo Sound` that will only be heard by the owning client.
*   Schedule a timer that will prevent the client from spamming the fire button for `1.5s` after playing the fire animation.

Follow these steps to complete this exercise:

1.  Create a new `RPC` and save it to a location of your liking.
2.  Once the project has been created, it should open the editor as well as the Visual Studio solution.
3.  Close the editor and go back to Visual Studio.
4.  Open `RPCCharacter.h` and declare the protected `FireTimer` variable, which will be used to prevent the client from spamming the `Fire` action:

    ```cpp
    FTimerHandle FireTimer;
    ```

5.  Declare the protected replicated `Ammo` variable, which starts with `5` shots:

    ```cpp
    UPROPERTY(Replicated)
    int32 Ammo = 5;
    ```

6.  Next, declare the protected animation montage variable that will be played when the character fires:

    ```cpp
    UPROPERTY(EditDefaultsOnly, Category = "RPC Character")
    UAnimMontage* FireAnimMontage;
    ```

7.  Declare the protected sound variable that will be played when the character has no ammo:

    ```cpp
    UPROPERTY(EditDefaultsOnly, Category = "RPC Character")
    USoundBase* NoAmmoSound;
    ```

8.  Override the `Tick` function:

    ```cpp
    virtual void Tick(float DeltaSeconds) override;
    ```

9.  Declare the reliable and validated Server RPC for firing:

    ```cpp
    UFUNCTION(Server, Reliable, WithValidation, Category = "RPC Character")
    void ServerFire();
    ```

10.  Declare the unreliable Multicast RPC that will play the fire animation on all of the clients:

    ```cpp
    UFUNCTION(NetMulticast, Unreliable, Category = "RPC Character")
    void MulticastFire();
    ```

11.  Declare the unreliable Client RPC that will play a sound only in the owning client:

    ```cpp
    UFUNCTION(Client, Unreliable, Category = "RPC Character")
    void ClientPlaySound2D(USoundBase* Sound);
    ```

12.  Now, open the `RPCCharacter.cpp` file and include `GameplayStatics.h` for the PlaySound2D function and the UnrealNetwork.h so we can use the `DOREPLIFETIME_CONDITION` macro:

    ```cpp
    #include "Kismet/GameplayStatics.h""
    #include "Net/UnrealNetwork.h"
    ```

13.  At the end of the constructor, enable the `Tick` function:

    ```cpp
    PrimaryActorTick.bCanEverTick = true;
    ```

14.  Implement the `GetLifetimeReplicatedProps` function so that the `Ammo` variable will replicate to all of the clients:

    ```cpp
    void ARPCCharacter::GetLifetimeReplicatedProps(TArray< 
      FLifetimeProperty >& OutLifetimeProps) const
    {
      Super::GetLifetimeReplicatedProps(OutLifetimeProps);
      DOREPLIFETIME(ARPCCharacter, Ammo);
    }
    ```

15.  Next, implement the `Tick` function, which displays the value of the `Ammo` variable:

    ```cpp
    void ARPCCharacter::Tick(float DeltaSeconds)
    {
      Super::Tick(DeltaSeconds);
      const FString AmmoString = 
      FString::Printf(TEXT("Ammo = %d"), Ammo);
      DrawDebugString(GetWorld(), GetActorLocation(), 
      AmmoString, nullptr, FColor::White, 0.0f, true);
    }
    ```

16.  At the end of the `SetupPlayerInputController` function, bind the `Fire` action to the `ServerFire` function:

    ```cpp
    PlayerInputComponent->BindAction("Fire", IE_Pressed, this, &ARPCCharacter::ServerFire);
    ```

17.  Implement the fire Server RPC validation function:

    ```cpp
    bool ARPCCharacter::ServerFire_Validate()
    {
      return true;
    }
    ```

18.  Implement the fire Server RPC implementation function:

    ```cpp
    void ARPCCharacter::ServerFire_Implementation()
    {

    }
    ```

19.  Now, add the logic to abort the function if the fire timer is still active since we fired the last shot:

    ```cpp
    if (GetWorldTimerManager().IsTimerActive(FireTimer))
    {
      return;
    }
    ```

20.  Check whether the character has ammo. If it doesn’t, then play `NoAmmoSound` only in the client that controls the character and abort the function:

    ```cpp
    if (Ammo == 0)
    {
      ClientPlaySound2D(NoAmmoSound);
      return;
    }
    ```

21.  Deduct the ammo and schedule the `FireTimer` variable to prevent this function from being spammed while playing the fire animation:

    ```cpp
    Ammo--;
    GetWorldTimerManager().SetTimer(FireTimer, 1.5f, false);
    ```

22.  Call the fire Multicast RPC to make all the clients play the fire animation:

    ```cpp
    MulticastFire();
    ```

23.  Implement the fire Multicast RPC, which will play the fire animation montage:

    ```cpp
    void ARPCCharacter::MulticastFire_Implementation()
    {
      if (FireAnimMontage != nullptr)
      {
        PlayAnimMontage(FireAnimMontage);
      }
    }
    ```

24.  Implement the Client RPC that plays a 2D sound:

    ```cpp
    void ARPCCharacter::ClientPlaySound2D_Implementation(USoundBase* Sound)
    {
      UGameplayStatics::PlaySound2D(GetWorld(), Sound);
    }
    ```

Finally, you can launch the project in the editor.

1.  Compile the code and wait for the editor to fully load.
2.  Go to **Project Settings**, go to **Engine**, then **Input**, and add the **Fire** action binding:

![Figure 17.2 – Adding the new Fire action binding ](img/Figure_17.02_B18531.jpg)

Figure 17.2 – Adding the new Fire action binding

1.  Close **Project Settings**.
2.  In the `Content` folder, create a new folder called `Audio`, and open it.
3.  Click the `Exercise17.01\Assets` folder, and import `NoA``mmo.wav` and `Fire.wav`.
4.  Save both files.
5.  Go to the `Content\Characters\Mannequins\Animations` folder.
6.  Click the `Exercise17.01\Assets` folder, and import the `ThirdPersonFire.fbx` file. Make sure it’s using the `SK_Mannequin` skeleton and click **Import**.
7.  Open the new animation and put a `Play Sound` anim notify at `0.3` seconds using the `Fire` sound.
8.  On the **Details** panel, find the **Enable Root Motion** option and set it to **true**. This will prevent the character from moving when playing the animation.
9.  Save and close `ThirdPersonFire`.
10.  *Right-click* on `ThirdPersonFire` and pick **Create** | **Create AnimMontage**.
11.  The `Animations` folder should look like this:

![Figure 17.3 – The Animations folder for the Mannequin ](img/Figure_17.03_B18531.jpg)

Figure 17.3 – The Animations folder for the Mannequin

1.  Open `ABP_Manny` and go to `AnimGraph`.
2.  Find the `Control Rig` node and set `Alpha` to `0.0` to disable the automatic feet adjustment. You should get the following output:

![Figure 17.4 – Disabling the feet adjustment ](img/Figure_17.04_B18531.jpg)

Figure 17.4 – Disabling the feet adjustment

1.  Save and close `ABP_Manny`.
2.  Open `SK_Mannequin` in the `Content\Characters\Mannequins\Meshes` folder and retarget (as shown in *Exercise 16.04*) the `root` and `pelvis` bones so that they use `Animation`. The remaining bones should use `Skeleton`.
3.  Save and close `SK_Mannequin`.
4.  Go to `Content\ThirdPerson\Blueprints` and open the `BP_T``hirdPersonCharacter` blueprint.
5.  In `Class Defaults`, set `No Ammo Sound` to use `NoAmmo`, and set `Fire Anim Montage` to use `ThirdPersonFire_Montage`.
6.  Save and close `BP_ThirdPersonCharacter`.
7.  Go to `2`.
8.  Set the window size to `800x600` and play using **New Editor Window (PIE)**.

You should get the following output:

![Figure 17.5 – The result of this exercise ](img/Figure_17.05_B18531.jpg)

Figure 17.5 – The result of this exercise

By completing this exercise, you can play on each client. Every time you press the left mouse button, the character of the client will play the `1`. If you try to fire when the ammo is `0`, that client will hear `No Ammo Sound` and won’t do the fire animation, because the server didn’t call the Multicast RPC. If you try to spam the fire button, you’ll notice that it will only trigger a new fire once the animation has finished.

In this section, you learned how to use all of the different types of RPCs and their caveats. In the next section, we will look at enumerations and how to expose them to the editor.

# Exposing enumerations to the editor

An enumeration is a user-defined data type that holds a list of integer constants, where each item has a human-friendly name assigned by you, which makes the code easier to read. As an example, if we wanted to represent the different states that a character can be in, we could use an integer variable where `0` means it’s idle, `1` means it’s walking, and so on. The problem with this approach is that when you see code such as `if(State == 0)`, it’s hard to remember what `0` means unless you are using some type of documentation or comments to help you. To fix this problem, you should use enumerations, where you can write code such as `if(State == EState::Idle)`, which is much more explicit and easier to understand.

In C++, you have two types of enums – the older raw enums and the new enum classes, which were introduced in C++11\. If you want to use the new enum classes in the editor, your first instinct might be to do it in the typical way, which is by declaring a variable or a function that uses the enumeration with `UPROPERTY` or `UFUNCTION`, respectively.

The problem is, if you try to do that, you’ll get a compilation error. Take a look at the following example:

```cpp
enum class ETestEnum : uint8
{
  EnumValue1,
  EnumValue2,
  EnumValue3
};
```

In the preceding code snippet, we’ve declared an enum class called `ETestEnum` that has three possible values – `EnumValue1`, `EnumValue2`, and `EnumValue3`.

After that, try either of the following examples inside a class:

```cpp
UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "Test")
ETestEnum TestEnum;
UFUNCTION(BlueprintCallable, Category = "Test")
void SetTestEnum(ETestEnum NewTestEnum) { TestEnum = NewTestEnum; }
```

In the preceding code snippet, we declared a `UPROPERTY` variable and a `UFUNCTION` function that uses the `ETestEnum` enumeration. If you try to compile, you’ll get the following compilation error:

```cpp
error : Unrecognized type 'ETestEnum' - type must be a UCLASS, USTRUCT or UENUM
```

Note

In Unreal Engine, it’s good practice to prefix the name of an enumeration with the letter `E`. For example, you could have `EWeaponType` and `EAmmoType`.

This error happens because when you try to expose a class, struct, or enumeration to the editor with the `UPROPERTY` or `UFUNCTION` macro, you need to add it to the Unreal Engine Reflection System by using the `UCLASS`, `USTRUCT`, and `UENUM` macros, respectively.

Note

You can learn more about the Unreal Engine Reflection System at [https://www.unrealengine.com/en-US/blog/unreal-property-system-reflection](https://www.unrealengine.com/en-US/blog/unreal-property-system-reflection).

With that knowledge in mind, it is simple to fix the previous error. Just do the following:

```cpp
UENUM()
enum class ETestEnum : uint8
{
  EnumValue1,
  EnumValue2,
  EnumValue3
};
```

In the next section, we will look at the `TEnumAsByte` type.

## TEnumAsByte

If you want to expose a variable to the engine that uses a raw enum, then you need to use the `TEnumAsByte` type. If you declare a `UPROPERTY` variable using a raw enum (not enum classes), you’ll get a compilation error.

Have a look at the following example:

```cpp
UENUM()
enum ETestRawEnum
{
  EnumValue1,
  EnumValue2,
  EnumValue3
};
```

Let’s say you declare a `UPROPERTY` variable using `ETestRawEnum`, like so:

```cpp
UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "Test")
ETestRawEnum TestRawEnum;
```

You’ll get the following compilation error:

```cpp
error : You cannot use the raw enum name as a type for member variables, instead use TEnumAsByte or a C++11 enum class with an explicit underlying type.
```

To fix this error, you need to surround the enum type of the variable, which in this case is `ETestRawEnum`, with `TEnumAsByte<>`, like so:

```cpp
UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "Test")
TEnumAsByte<ETestRawEnum> TestRawEnum;
```

In the next section, we will look at the `UMETA` macro.

## UMETA

When you use the `UENUM` macro to add an enumeration to the Unreal Engine Reflection System, you can use the `UMETA` macro on each value of the enum. The `UMETA` macro, just like with other macros, such as `UPROPERTY` or `UFUNCTION`, can use specifiers that will inform Unreal Engine of how to handle that value. Let’s look at the most commonly used `UMETA` specifiers.

### DisplayName

This specifier allows you to define a new name that is easier to read for the enum value when it’s displayed in the editor.

Take a look at the following example:

```cpp
UENUM()
enum class ETestEnum : uint8
{
  EnumValue1 UMETA(DisplayName = "My First Option"),
  EnumValue2 UMETA(DisplayName = "My Second Option"),
  EnumValue3 UMETA(DisplayName = "My Third Option")
};
```

Let’s declare the following variable:

```cpp
UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "Test")
ETestEnum TestEnum;
```

When you open the editor and look at the `TestEnum` variable, you will see a dropdown where `EnumValue1`, `EnumValue2`, and `EnumValue3` have been replaced with `My First Option`, `My Second Option`, and `My Third Option`, respectively.

### Hidden

This specifier allows you to hide a specific enum value from the dropdown. This is typically used when there is an enum value that you only want to be able to use in C++ and not in the editor.

Take a look at the following example:

```cpp
UENUM()
enum class ETestEnum : uint8
{
  EnumValue1 UMETA(DisplayName = "My First Option"),
  EnumValue2 UMETA(Hidden),
  EnumValue3 UMETA(DisplayName = "My Third Option")
};
```

Let’s declare the following variable:

```cpp
UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "Test")
ETestEnum TestEnum;
```

When you open the editor and look at the `TestEnum` variable, you will see a dropdown. You should notice that `My Second Option` doesn’t appear in the dropdown and therefore can’t be selected.

Note

For more information on all of the UMETA specifiers, visit [https://docs.unrealengine.com/en-US/Programming/UnrealArchitecture/Reference/Metadata/#enummetadataspecifiers](https://docs.unrealengine.com/en-US/Programming/UnrealArchitecture/Reference/Metadata/#enummetadataspecifiers).

In the next section, we will look at the `BlueprintType` specifier for the `UENUM` macro.

## BlueprintType

This `UENUM` specifier will expose the enumeration to blueprints. This means that there will be an entry for that enumeration in the dropdown that is used when making new variables or inputs/outputs for a function, as shown in the following example:

![Figure 17.6 – Setting a variable to use the ETestEnum variable type ](img/Figure_17.06_B18531.jpg)

Figure 17.6 – Setting a variable to use the ETestEnum variable type

It will also create additional functions that you can call on the enumeration in the editor, as shown in the following example:

![Figure 17.7 – List of additional functions available when using BlueprintType ](img/Figure_17.07_B18531.jpg)

Figure 17.7 – List of additional functions available when using BlueprintType

### MAX

When using enumerations, it’s common to want to know how many values it has. In Unreal Engine, the standard way of doing this is by adding `MAX` as the last value, which will be automatically hidden in the editor.

Take a look at the following example:

```cpp
UENUM()
enum class ETestEnum : uint8
{
  EnumValue1,
  EnumValue2,
  EnumValue3,
  MAX
};
```

If you want to know how many values `ETestEnum` has in C++, you just need to do the following:

```cpp
const int32 MaxCount = static_cast<int32>(ETestEnum::MAX);
```

This works because enumerations in C++ are internally stored as numbers, where the first value is `0`, the second is `1`, and so on. This means that so long as `MAX` is the last value, it will always have the total number of values in the enumeration. An important thing to take into consideration is that for `MAX` to give you the correct value, you cannot change the internal numbering order of the enumeration, like so:

```cpp
UENUM()
enum class ETestEnum : uint8
{
  EnumValue1 = 4,
  EnumValue2 = 78,
  EnumValue3 = 100,
  MAX
};
```

In this case, `MAX` will be `101` because it will use the number immediately next to the previous value, which is `EnumValue3 = 100`.

Using `MAX` is only meant to be used in C++ and not in the editor because the `MAX` value is hidden in blueprints, as mentioned previously. To get the number of entries of an enumeration in blueprints, you should use the `BlueprintType` specifier in the `UENUM` macro to expose some useful functions on the context menu. After that, you just need to type the name of your enumeration in the context menu. If you select the **Get number of entries in ETestEnum** option, you will have a function that returns the number of entries of that enumeration.

In the next exercise, you will be using C++ enumerations in the editor.

## Exercise 17.02 – Using C++ enumerations in the editor

In this exercise, we’re going to create a new **C++** project that uses the **Third Person** template. We’re going to add the following:

*   An enumeration called `EWeaponType` that contains **three** weapons – a pistol, a shotgun, and a rocket launcher.
*   An enumeration called `EAmmoType` that contains **3** ammo types – bullets, shells, and rockets.
*   A variable called `Weapon` that uses `EWeaponType` to tell the type of the current weapon.
*   An integer array variable called `Ammo` that holds the amount of ammo for each type, which is initialized with a value of `10`.
*   When the player presses the `1`, `2`, or `3` key, the `Weapon` variable will be set to `Pistol`, `Shotgun`, or `Rocket Launcher`, respectively.
*   When the player presses the left mouse button, the ammo for the current weapon will be consumed.
*   With every `Tick` function call, the character will display the current weapon type and the equivalent ammo type and amount.

Follow these steps to complete this exercise:

1.  Create a new `Enumerations` and save it to a location of your liking.

Once the project has been created, it should open the editor as well as the Visual Studio solution.

1.  Close the editor and go back to Visual Studio.
2.  Open the `Enumerations.h` file.
3.  Create a macro called `ENUM_TO_INT32` that will convert an enumeration into an `int32` data type:

    ```cpp
    #define ENUM_TO_INT32(Value) static_cast<int32>(Value)
    ```

4.  Create a macro called `ENUM_TO_FSTRING` that will get the display name for a value of an `enum` data type and convert it into an `FString` datatype:

    ```cpp
    #define ENUM_TO_FSTRING(Enum, Value) FindObject<UEnum>(ANY_PACKAGE, TEXT(Enum), true)->GetDisplayNameTextByIndex(ENUM_TO_INT32(Value)).ToString()
    ```

5.  Declare the `EWeaponType` and `EammoType` enumerations:

    ```cpp
    UENUM(BlueprintType)
    enum class EWeaponType : uint8
    {
      Pistol UMETA(Display Name = "Glock 19"),
      Shotgun UMETA(Display Name = "Winchester M1897"),
      RocketLauncher UMETA(Display Name = "RPG"),    
      MAX
    };
    UENUM(BlueprintType)
    enum class EAmmoType : uint8
    {
      Bullets UMETA(DisplayName = "9mm Bullets"),
      Shells UMETA(Display Name = "12 Gauge Shotgun 
      Shells"),
      Rockets UMETA(Display Name = "RPG Rockets"),
      MAX
    };
    ```

6.  Open the `EnumerationsCharacter.h` file and add the `Enumerations.h` header before `EnumerationsCharacter.generated.h`:

    ```cpp
    #include "Enumerations.h"
    ```

7.  Declare the protected `Weapon` variable that holds the weapon type of the selected weapon:

    ```cpp
    UPROPERTY(BlueprintReadOnly, Category = "Enumerations Character")
    EWeaponType Weapon;
    ```

8.  Declare the protected `Ammo` array that holds the amount of ammo for each type:

    ```cpp
    UPROPERTY(EditDefaultsOnly, BlueprintReadOnly, Category = "Enumerations Character")
    TArray<int32> Ammo;
    ```

9.  Declare the protected overrides for the `Begin Play` and `Tick` functions:

    ```cpp
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaSeconds) override;
    ```

10.  Declare the protected input functions:

    ```cpp
    void Pistol();
    void Shotgun();
    void RocketLauncher();
    void Fire();
    ```

11.  Open the `EnumerationsCharacter.cpp` file and bind the new action bindings at the end of the `SetupPlayerInputController` function, as shown in the following code snippet:

    ```cpp
    PlayerInputComponent->BindAction("Pistol", IE_Pressed, this, &AEnumerationsCharacter::Pistol);
    PlayerInputComponent->BindAction("Shotgun", IE_Pressed, this, &AEnumerationsCharacter::Shotgun);
    PlayerInputComponent->BindAction("Rocket Launcher", IE_Pressed, this, &AEnumerationsCharacter::RocketLauncher);
    PlayerInputComponent->BindAction("Fire", IE_Pressed, this, &AEnumerationsCharacter::Fire);
    ```

12.  Next, implement the override for `BeginPlay` that executes the parent logic, but also initializes the size of the `Ammo` array with the number of entries in the `EAmmoType` enumeration. Each position in the array will also be initialized with a value of `10`:

    ```cpp
    void AEnumerationsCharacter::BeginPlay()
    {
      Super::BeginPlay();
      constexpr int32 AmmoTypeCount = 
      ENUM_TO_INT32(EAmmoType::MAX);
      Ammo.Init(10, AmmoTypeCount);
    }
    ```

13.  Implement the override for `Tick`:

    ```cpp
    void AEnumerationsCharacter::Tick(float DeltaSeconds)
    {
      Super::Tick(DeltaSeconds);
    }
    ```

14.  Convert the `Weapon` variable into `int32` and the `Weapon` variable into an `FString`:

    ```cpp
    const int32 WeaponIndex = ENUM_TO_INT32(Weapon);
    const FString WeaponString = ENUM_TO_FSTRING("EWeaponType", Weapon);
    ```

15.  Convert the ammo type into an `FString` and get the ammo count for the current weapon:

    ```cpp
    const FString AmmoTypeString = ENUM_TO_FSTRING("EAmmoType", Weapon);
    const int32 AmmoCount = Ammo[WeaponIndex];
    ```

We are using `Weapon` to get the ammo type string because the entries in `EAmmoType` match the type of ammo of the equivalent `EWeaponType`. In other words, `Pistol = 0` uses `Bullets = 0`, `Shotgun = 1` uses `Shells = 1`, and `RocketLauncher = 2` uses `Rockets = 2`, so it’s a 1-to-1 mapping that we can use in our favor.

1.  Display the name of the current weapon in the character’s location and its corresponding ammo type and ammo count, as shown in the following code snippet:

    ```cpp
    const FString String = FString::Printf(TEXT("Weapon = %s\nAmmo Type = %s\nAmmo Count = %d"), *WeaponString, *AmmoTypeString, AmmoCount);
    DrawDebugString(GetWorld(), GetActorLocation(), String, nullptr, FColor::White, 0.0f, true);
    ```

2.  Implement the equip input functions that set the `Weapon` variable with the corresponding value:

    ```cpp
    void AEnumerationsCharacter::Pistol()
    {
      Weapon = EWeaponType::Pistol;
    }
    void AEnumerationsCharacter::Shotgun()
    {
      Weapon = EWeaponType::Shotgun;
    }
    void AEnumerationsCharacter::RocketLauncher()
    {
      Weapon = EWeaponType::RocketLauncher;
    }
    ```

3.  Implement the fire input function that will use the weapon index to get the corresponding ammo type count and subtract `1`, so long as the resulting value is greater than or equal to 0:

    ```cpp
    void AEnumerationsCharacter::Fire()
    {
      const int32 WeaponIndex = ENUM_TO_INT32(Weapon);
      const int32 NewRawAmmoCount = Ammo[WeaponIndex] - 1;
      const int32 NewAmmoCount = 
      FMath::Max(NewRawAmmoCount, 0);
      Ammo[WeaponIndex] = NewAmmoCount;
    }
    ```

4.  Compile the code and run the editor.
5.  Go to **Project Settings**, go to **Engine**, then **Input**, and add the new action bindings:

![Figure 17.8 – Adding the Pistol, Shotgun, Rocket Launcher, and Fire bindings ](img/Figure_17.08_B18531.jpg)

Figure 17.8 – Adding the Pistol, Shotgun, Rocket Launcher, and Fire bindings

1.  Close **Project Settings**.
2.  Make sure the `1`. Click on **New Editor Window (PIE)**; you should get the following result:

![Figure 17.9 – The result of this exercise ](img/Figure_17.09_B18531.jpg)

Figure 17.9 – The result of this exercise

By completing this exercise, you can use the `0`.

In this section, you learned how to expose enumerations to the editor so that you can use them in blueprints. In the next section, we will look at array index wrapping, which allows you to iterate an array beyond its limits and wrap it back around from the other side.

# Using array index wrapping

Sometimes, when you use arrays to store information, you may want to iterate it in both directions and be able to wrap the index so that it doesn’t go beyond the index limit and crash the game. An example of this is the previous/next weapon logic in shooter games, where you have an array of weapons and you want to be able to cycle through them in a particular direction, and when you reach the first or the last index, you want to loop back around to the last and first index, respectively. The typical way of doing this would be as follows:

```cpp
AWeapon * APlayer::GetPreviousWeapon()
{
  if(WeaponIndex - 1 < 0)
  {
    WeaponIndex = Weapons.Num() - 1;
  }
  else
  {
    WeaponIndex--;
  }
  return Weapons[WeaponIndex];
}
AWeapon * APlayer::GetNextWeapon()
{
  if(WeaponIndex + 1 > Weapons.Num() - 1)
  {
    WeaponIndex = 0;
  }
  else
  {
    WeaponIndex++;
  }
  return Weapons[WeaponIndex];
}
```

In the preceding code, we set the `WeaponIndex` variable (declared as a member of the class) to loop back if the new weapon index is outside the limits of the weapons array, which can happen in two cases. The first case is when the player has the last weapon of the inventory equipped and we want the next weapon. In this case, it should go back to the first weapon.

The second case is when the player has the first weapon of the inventory equipped and we want the previous weapon. In this case, it should go to the last weapon.

While the example code works, it’s still quite a lot of code to solve such a trivial problem. To improve this code, there is a mathematical operation that will help you handle these two cases automatically in just one function. It’s called the modulo (represented in C++ by the `%` operator), which gives you the remainder of a division between two numbers.

So, how do we use the modulo to wrap the index of an array? Let’s rewrite the previous example using the modulo operator:

```cpp
AWeapon * APlayer::GetNewWeapon(int32 Direction)
{
  const int32 WeaponCount = Weapons.Num();
  const int32 NewRawIndex = WeaponIndex + Direction;
  const in32 NewWrappedIndex = NewIndex % WeaponCount;
  WeaponIndex = (NewClampedIndex + WeaponCount) % 
  WeaponCount;
  return Weapons[WeaponIndex];
}
```

This is the new version, and you can tell right away that it’s a bit harder to understand, but it’s more functional and compact. If you don’t use the variables to store the intermediate values of each operation, you can probably make the entire function in one or two lines of code.

Let’s break down the preceding code snippet:

*   `const int WeaponCount = Weapons.Num()`: We need to know the size of the array to determine the index where it should go back to `0`. In other words, if `WeaponCount = 4`, then the array has the `0`, `1`, `2`, and `3` indexes, which tells us that index `4` is the cutoff index where it should go back to `0`.
*   `const int32 NewRawIndex = WeaponIndex + Direction`: This is the new raw index that doesn’t care about the limits of the array. The `Direction` variable is used to indicate the offset we want to add to the current index of the array. This is either `-1` if we want the previous index or `1` if we want the next index.
*   `const int32 NewWrappedIndex = NewRawIndex % WeaponCount`: This will make sure that `NewWrappedIndex` is within the `0` to `WeaponCount - 1` interval and wrap around if needed, due to the modulo properties. So, if `NewRawIndex` is `4`, then `NewWrappedIndex` will become `0`, because there is no remainder from the division of `4 / 4`.

If `Direction` is always `1`, meaning we only want the next index, then the value of `NewWrappedIndex` is enough for what we need. If we also want to use `Direction` with `-1`, then we’ll have a problem, because the modulo operation won’t wrap the index correctly for negative indexes. So, if `WeaponIndex` is `0` and `Direction` is `-1`, then `NewWrappedIndex` will be `-1`, which is not correct. To fix this limitation, we need to do some additional calculations:

*   `WeaponIndex = (NewWrappedIndex + WeaponCount) % WeaponCount`: This will add `WeaponCount` to `NewWrappedIndex` to make it positive and apply the modulo again to get the correct wrapped index, which fixes the problem.
*   `return Weapons[WeaponIndex]`: This returns the weapon in the calculated `WeaponIndex` index position.

Let’s take a look at a practical example to help you visualize how all this works.

Weapons:

*   `[0] Knife`
*   `[1] Pistol`
*   `[2] Shotgun`
*   `[3] Rocket Launcher`

`WeaponCount = Weapons.Num()`, so it has a value of `4`.

Let’s assume that `WeaponIndex = 3` and `Direction = 1`.

Here, we would have the following:

*   `NewRawIndex = WeaponIndex + Direction`, so `3 + 1 = 4`
*   `NewWrappedIndex = NewRawIndex % WeaponCount`, so `4 % 4 = 0`
*   `WeaponIndex = (NewWrappedIndex + WeaponCount) % WeaponCount, so (0 + 4) % 4 = 0`

In this example, the starting value for `WeaponIndex` is `3`, which is `Rocket Launcher`, and we want the next weapon because `Direction` is set to `1`. Performing the calculations, `WeaponIndex` will now be `0`, which is `Knife`. This is the desired behavior because we have four weapons, so we circled back to the first index. In this case, since `NewRawIndex` is positive, we could’ve just used `NewWrappedIndex` without doing the extra calculations.

Let’s debug it again using different values.

Let’s assume that `WeaponIndex = 0` and `Direction = -1`:

*   `NewRawIndex = WeaponIndex + Direction`, so `0 + -1 = -1`
*   `NewWrappedIndex = NewIndex % WeaponCount`, so `-1 % 4 = -1`
*   `WeaponIndex = (NewWrappedIndex + WeaponCount) % WeaponCount`, so `(-1 + 4) % 4 = 3`

In this example, the starting value for `WeaponIndex` is `0`, which is `Knife`, and we want the previous weapon because `Direction` is set to -`1`. Doing the calculations, `WeaponIndex` will now be `3`, which is `Rocket Launcher`. This is the desired behavior because we have four weapons, so we circled back to the last index. In this case, since `NewRawIndex` is negative, we can’t just use `NewWrappedIndex`; we need to do the extra calculation to get the correct value.

In the next exercise, you’re going to use the knowledge you’ve acquired to cycle between an enumeration of weapons in both directions.

## Exercise 17.03 – Using array index wrapping to cycle between an enumeration

In this exercise, we’re going to use the project from *Exercise 17.02 – Using C++ enumerations in the editor*, and add two new action mappings for cycling the weapons. `Mouse Wheel Up` will go to the previous weapon type, while `Mouse Wheel Down` will go to the next weapon type.

Follow these steps to complete this exercise:

1.  First, open the Visual Studio project from *Exercise 17.02 – Using C++ enumerations in the editor*.

Next, you will be updating `Enumerations.h` and adding a macro that will handle the array index wrapping in a very convenient way.

1.  Open `Enumerations.h` and add the `GET_WRAPPED_ARRAY_INDEX` macro. This will apply the modulo formula that we covered previously:

    ```cpp
    #define GET_WRAPPED_ARRAY_INDEX(Index, Count) (Index % Count + Count) % Count
    ```

2.  Open `EnumerationsCharacter.h` and declare the new input functions for the weapon cycling:

    ```cpp
    void PreviousWeapon();
    void NextWeapon();
    ```

3.  Declare the `CycleWeapons` function, as shown in the following code snippet:

    ```cpp
    void CycleWeapons(int32 Direction);
    ```

4.  Open `EnumerationsCharacter.cpp` and bind the new action bindings in the `SetupPlayerInputController` function:

    ```cpp
    PlayerInputComponent->BindAction("Previous Weapon", IE_Pressed, this, &AEnumerationsCharacter::PreviousWeapon);
    PlayerInputComponent->BindAction("Next Weapon", IE_Pressed, this, &AEnumerationsCharacter::NextWeapon);
    ```

5.  Now, implement the new input functions, as shown in the following code snippet:

    ```cpp
    void AEnumerationsCharacter::PreviousWeapon()
    {
      CycleWeapons(-1);
    }
    void AEnumerationsCharacter::NextWeapon()
    {
      CycleWeapons(1);
    }
    ```

In the preceding code snippet, we defined the functions that handle the action mappings for `Previous Weapon` and `Next Weapon`. Each function uses the `CycleWeapons` function, with a direction of `-1` for the previous weapon and `1` for the next weapon.

1.  Implement the `CycleWeapons` function, which does the array index wrapping using the `Direction` parameter based on the current weapon index:

    ```cpp
    void AEnumerationsCharacter::CycleWeapons(int32 Direction)
    {
      const int32 WeaponIndex = ENUM_TO_INT32(Weapon);
      const int32 AmmoCount = Ammo.Num();
      const int32 NextRawWeaponIndex = WeaponIndex + 
      Direction;
      const int32 NextWeaponIndex = 
      GET_WRAPPED_ARRAY_INDEX(NextRawWeaponIndex , 
      AmmoCount);
      Weapon = static_cast<EWeaponType>(NextWeaponIndex);
    }
    ```

2.  Compile the code and run the editor.
3.  Go to **Project Settings**, go to **Engine**, then **Input**, and add the new action bindings:

![Figure 17.10 – Adding the Previous Weapon and Next Weapon bindings ](img/Figure_17.10_B18531.jpg)

Figure 17.10 – Adding the Previous Weapon and Next Weapon bindings

1.  Close **Project Settings**.
2.  Make sure that the `1`. Click on **New Editor Window (PIE)**; you should get the following result:

![Figure 17.11 – The result of this exercise ](img/Figure_17.11_B18531.jpg)

Figure 17.11 – The result of this exercise

By completing this exercise, you can use the mouse wheel to cycle between the weapons. If you select the rocket launcher and use the mouse wheel down to go to the next weapon, it will go back to the pistol. If you use the mouse wheel down to go to the previous weapon with the pistol selected, it will go back to the rocket launcher.

In the next activity, you will be adding the concept of weapons and ammo to the multiplayer FPS project we started in [*Chapter 16*](B18531_16.xhtml#_idTextAnchor345), *Getting Started with Multiplayer Basics*.

# Activity 17.01 – Adding weapons and ammo to the multiplayer FPS game

In this activity, you’ll add the concept of weapons and ammo to the multiplayer FPS project that we started in the previous chapter. You will need to use the different types of RPCs covered in this chapter to complete this activity.

Follow these steps to complete this activity:

1.  Open the `MultiplayerFPS` project from *Activity 16.01 – Creating a character for the multiplayer FPS project*.
2.  Create an `AnimMontage` slot called `Upper Body`.
3.  Import the animations (`Pistol_Fire.fbx`, `MachineGun_Fire.fbx`, and `Railgun_Fire.fbx`) from the `Activity17.01\Assets` folder into `Content\Player\Animations`.
4.  Create an `AnimMontage` for `Pistol_Fire`, `MachineGun_Fire`, and `Railgun_Fire`, and make sure they have the following configurations:
    *   `Blend In` time of `0.01` and a `Blend Out` time of `0.1`. Make sure it uses the `Upper Body` slot.
    *   `Blend In` time of `0.01` and a `Blend Out` time of `0.1`. Make sure it uses the `Upper Body` slot.
    *   `Upper Body` slot.
5.  Import `SK_Weapon.fbx` (with Material Import Method set to Create New Materials), `NoAmmo.wav`, `WeaponChange.wav`, and `Hit.wav` from the `Activity17.01\Assets` folder into `Content\Weapons`.
6.  Import `Pistol_Fire_Sound.wav` from `Activity17.01\Assets` into `Content\Weapons\Pistol` and use it on `Play Sound` in the `Pistol_Fire` animation.
7.  Create a simple green-colored material instance from `M_FPGun` called `MI_Pistol` and place it on `Content\Weapons\Pistol`.
8.  Import `MachineGun_Fire_Sound.wav` from `Activity17.01\Assets` into `Content\Weapons\MachineGun` and use it on `Play Sound` in the `MachineGun_Fire` animation.
9.  Create a simple red-colored material instance from `M_FPGun` called `MI_MachineGun` and place it on `Content\Weapons\MachineGun`.
10.  Import `Railgun_Fire_Sound.wav` from `Activity17.01\Assets` into `Content\Weapons\Railgun` and use it on `Play Sound` in the `Railgun_Fire` animation.
11.  Create a simple white-colored material instance from `M_FPGun` called `MI_Railgun` and place it on `Content\Weapons\Railgun`.
12.  Edit the `SK_Mannequin_Skeleton` and create a socket called `GripPoint` from `hand_r` with `Relative Location` set to `(X=-10.403845,Y=6.0,Z=-3.124871)` and `Relative Rotation` set to `(X=0.0,Y=0.0,Z=90.0)`.
13.  Add the following input actions to `Content\Player\Inputs`, using the knowledge you acquired in [*Chapter 4*](B18531_04.xhtml#_idTextAnchor099), *Getting Started with Player Input*:
    *   **IA_Fire (Digital)**: *Left Mouse Button*
    *   **IA_Pistol (Digital)**: *1*
    *   **IA_MachineGun (Digital)**: *2*
    *   **IA_Railgun (Digital)**: *3*
    *   **IA_PreviousWeapon (Digital)**: *Mouse Wheel Up*
    *   **IA_NextWeapon (Digital)**: *Mouse Wheel Down*
14.  Add the new input actions to `IMC_Player`.
15.  In `MultiplayerFPS.h`, create the `ENUM_TO_INT32(Enum)` macro, which casts an enumeration to `int32`, and the `GET_WRAPPED_ARRAY_INDEX(Index, Count)` macro, which uses array indexing wrapping to make sure the index is within the limits of the array.
16.  Create a header file called `EnumTypes.h` that holds the following enumerations:

`Pistol`, `MachineGun`, `Railgun`, `MAX`

`Single`, `Automatic`

`PistolBullets`, `MachineGunBullets`, `Slugs`, `MAX`

1.  Create a C++ class called `Weapon` that derives from the `Actor` class and has a skeletal mesh component called `Mesh` as the root component.

In terms of variables, it stores the name, the weapon type, the ammo type, the fire mode, how far the hitscan goes, how much damage the hitscan does when it hits, the fire rate, the animation montage to use when firing, and the sound to play when it has no ammo. In terms of functionality, it needs to be able to start the fire (and also stop the fire, because of the automatic fire mode), which checks whether the player can fire. If it can, then it plays the fire animation in all of the clients and shoots a line trace in the camera position and direction with the supplied length to damage the actor it hits. If it doesn’t have ammo, it will play a sound only on the owning client.

1.  Edit `FPSCharacter` so that it supports the new input actions for `Fire`, `Pistol`, `Machine Gun`, `Railgun`, `Previous Weapon`, and `Next Weapon`. In terms of variables, it needs to store the amount of ammo for each type, the currently equipped weapon, all of the weapons classes and spawned instances, the sound to play when it hits another player, and the sound when it changes weapons. In terms of functions, it needs to be able to equip/cycle/add weapons, manage ammo (add, remove, and get), handle when the character is damaged, play an anim montage on all of the clients, and play a sound on the owning client.
2.  Create `BP_Pistol` from `AWeapon`, place it on `Content\Weapons\Pistol`, and configure it with the following values:
    *   `Content\Weapons\SK_Weapon`
    *   `Content\Weapons\Pistol\MI_Pistol`
    *   `Pistol Mk I`
    *   `Pistol`
    *   `Pistol Bullets`
    *   `Automatic`
    *   `9999.9`, `5.0`, `0.5`
    *   `Content\Player\Animations\Pistol_Fire_Montage`
    *   `Content\Weapons\NoAmmo`
3.  Create `BP_MachineGun` from `Aweapon`, place it on `Content\Weapons\MachineGun`, and configure it with the following values:
    *   `Content\Weapons\SK_Weapon`
    *   `Content\Weapons\MachineGun\MI_MachineGun`
    *   `Machine Gun Mk I`
    *   `Machine Gun`
    *   `Machine Gun Bullets`
    *   `Automatic`
    *   `9999.9`, `5.0`, `0.1`
    *   `Content\Player\Animations\MachineGun_Fire_Montage`
    *   `Content\Weapons\NoAmmo`
4.  Create `BP_Railgun` from `Aweapon`, place it on `Content\Weapons\Railgun`, and configure it with the following values:
    *   `Content\Weapons\SK_Weapon`
    *   `Content\Weapons\Railgun\MI_Railgun`
    *   `Railgun Mk I`
    *   `Railgun`
    *   `Slugs`
    *   `Single`
    *   `9999.9`, `100.0`, `1.5`
    *   `Content\Player\Animations\Railgun_Fire_Montage`
    *   `Content\Weapons\NoAmmo`
5.  Configure `BP_Player` with the following values:
    *   `BP_Pistol`, `BP_MachineGun`, `BP_Railgun`)
    *   `Content\Weapons\Hit`
    *   `Content\Weapons\WeaponChange`
    *   `Content\Player\Inputs\IA_Fire`
    *   `Content\Player\Inputs\IA_Pistol`
    *   `Content\Player\Inputs\IA_MachineGun`
    *   `Content\Player\Inputs\IA_Railgun`
    *   `Content\Player\Inputs\IA_Previous`
    *   `Content\Player\Inputs\IA_NextWeapon`
6.  Make the mesh component block the visibility channel so that it can be hit by the hitscans of the weapons.
7.  Edit `ABP_Player` so that it uses a *Layered blend Per bone* node, with `Mesh Space Rotation Blend` enabled, on the `spine_01` bone so that the upper body animations use the `Upper Body` slot.
8.  Edit `WBP_HUD` so that it displays a white dot crosshair in the middle of the screen, the current weapon, and the ammo count under the `Health` and `Armor` indicators.

**Expected output**:

The result should be a project where each client will have weapons with ammo and will be able to use them to fire at and damage other players. You will also be able to select weapons by using the *1*, *2*, and *3* keys and by using the mouse wheel up and down to select the previous and next weapon, respectively:

![Figure 17.12 – The expected result of this activity ](img/Figure_17.12_B18531.jpg)

Figure 17.12 – The expected result of this activity

Note

The solution to this activity can be found on GitHub here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions).

By completing this activity, you should have a good idea of how RPCs, enumerations, and array index wrapping work.

# Summary

In this chapter, you learned how to use RPCs to allow the server and the clients to execute logic on one another. You also learned how enumerations work in Unreal Engine by using the `UENUM` macro and how to use array index wrapping, which helps you iterate an array in both directions and loops around when you go beyond its index limits.

By completing this chapter’s activity, you learned how to develop a basic playable game where players can shoot each other and switch between their weapons.

In the next chapter, we’ll learn where the instances of the most common gameplay framework classes exist in multiplayer, as well as learn about the `Player State` and `Game State` classes. We’ll also cover some new concepts in the game mode that are used in multiplayer matches, as well as some useful general-purpose, built-in functionality.