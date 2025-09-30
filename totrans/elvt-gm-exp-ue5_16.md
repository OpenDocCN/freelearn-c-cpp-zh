# 16

# Getting Started with Multiplayer Basics

In the previous chapter, we completed the SuperSideScroller game and used 1D Blend Spaces, animation blueprints, and animation montages. In this chapter, we’re going to build on that knowledge and learn how to add multiplayer functionality to a game using Unreal Engine.

Multiplayer games have grown quite a lot in the last decade. Games such as Fortnite, League of Legends, Rocket League, Overwatch, and Counter-Strike: Global Offensive have gained a lot of popularity in the gaming community and have had great success. Nowadays, almost all games need to have some kind of multiplayer experience to be more relevant and successful.

The reason for this is it adds a new layer of possibilities on top of the existing gameplay, such as being able to remotely play with friends in cooperative mode (also known as online co-op) or against people from all around the world, which greatly increases the longevity and value of a game.

In this chapter, we’re going to cover the following main topics:

*   Introduction to multiplayer basics
*   Understanding the server
*   Understanding the client
*   Packaging the project
*   Exploring connections and ownership
*   Getting to know roles
*   Understanding variable replication
*   Exploring 2D Blend Spaces
*   Transforming (modifying) bones

By the end of this chapter, you’ll know basic multiplayer concepts such as the server-client architecture, connections, actor ownership, roles and variable replication so that you can create a multiplayer game of your own. You’ll also be able to make a 2D Blend Space, which allows you to blend between animations laid out in a 2D grid. Finally, you’ll learn how to use Transform (Modify) Bone nodes to control Skeletal Mesh bones at runtime.

# Technical requirements

For this chapter, you will need the following technical requirements:

*   Unreal Engine 5 installed
*   Visual Studio 2019 installed

The project for this chapter can be found in the `Chapter16` folder of the code bundle for this book, which can be downloaded here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition).

In the next section, we will discuss the basics of multiplayer.

# Introduction to multiplayer basics

You may have heard the term multiplayer a lot while gaming, but what does it mean for game developers? Multiplayer, in reality, is just a set of instructions sent through the network (internet or local area network) between the server and its connected clients to give players the illusion of a shared world.

For this to work, the server needs to be able to talk to clients, but also the other way around (client to server). This is because clients are typically the ones that affect the game world, so they need a way to be able to inform the server of their intentions while playing the game.

An example of this back and forth communication between the server and a client is when a player tries to fire a weapon during a game. Have a look at the following diagram, which shows a client-server interaction:

![Figure 16.1 – Client-server interaction when firing a weapon ](img/Figure_16..01_B18531.jpg)

Figure 16.1 – Client-server interaction when firing a weapon

Let’s understand the preceding diagram:

1.  The player holds the left mouse button down and the client of that player tells the server that it wants to fire a weapon.
2.  The server validates whether the player can fire the weapon by checking the following:
    *   If the player is alive
    *   If the player has the weapon equipped
    *   If the player has enough ammo
3.  If all of the conditions are valid, then the server will do the following:
    *   Run the logic to deduct ammo.
    *   Spawn the projectile actor on the server, which is automatically sent to all of the clients.
    *   Play the fire animation on that character instance in all of the clients to ensure synchronicity between all of them, which helps to sell the idea that it’s the same world, even though it’s not.
4.  If any of the conditions fail, then the server tells the specific client what to do:
    *   The player is dead: Don’t do anything.
    *   The player doesn’t have the weapon equipped: Don’t do anything.
    *   The player doesn’t have enough ammo: Play an empty click sound.

Remember, if you want your game to support multiplayer, then it’s highly recommended that you do that as soon as possible in your development cycle. If you try to run a single-player project with multiplayer enabled, you’ll notice that some functionalities might just work, but probably most of them won’t be working properly or as expected.

The reason for that is when you execute the game in single-player, the code runs locally and instantly, but when you add multiplayer into the equation, you are adding external factors such as an authoritative server that talks to clients on a network with latency, as you saw in *Figure 16.1*.

To get everything working properly, you need to break the existing code into the following components:

*   Code that only runs on the server
*   Code that only runs on the client
*   Code that runs on both the server and the client

To add multiplayer support to games, UE5 comes with a very powerful and bandwidth-efficient network framework already built in that uses an authoritative server-client architecture.

Here is a diagram of how it works:

![Figure 16.2 – Server-client architecture in UE5 ](img/Figure_16..02_B18531.jpg)

Figure 16.2 – Server-client architecture in UE5

Here, you can see how the server-client architecture works in UE5\. Each player controls a client that communicates with the server using a two-way connection. The server runs a specific level with a game mode (that only exists in the server) and controls the flow of information so that the clients can see and interact with each other in the game world.

Note

Multiplayer can be a very advanced topic, so these next few chapters will serve as an introduction to help you understand the essentials, but it will not be an in-depth look. For that reason, some concepts might be omitted for the sake of simplicity.

At this point, you have an idea of how multiplayer basics work. Now, let’s dive in and see how servers work and what their responsibilities are.

# Understanding the server

The server is the most critical part of the architecture since it’s responsible for handling most of the work and making important decisions.

Here is an overview of the main responsibilities of a server:

*   **Creating and managing the shared world instance**: The server runs its instance of the game in a specific level and game mode (this will be covered in [*Chapter 18*](B18531_18.xhtml#_idTextAnchor404), *Using Gameplay Framework Classes in Multiplayer*), which will serve as the shared world between all of the connected clients. The level being used can be changed at any point in time and, if applicable, the server can bring along all of the connected clients with it automatically.
*   **Handling client join and leave requests**: If a client wants to connect to a server, it needs to ask for permission. To do this, the client sends a join request to the server, through a direct IP connection (explained in the next section) or an online subsystem such as Steam. Once the join request reaches the server, it will perform some validations to determine whether the request is accepted or rejected.

Some of the most common reasons why the server rejects a request are if the server is already at full capacity and can’t take any more clients or if the client is using an out-of-date version of the game. If the server accepts the request, then a player controller with a connection is assigned to the client and the `PostLogin` function in the game mode is called. From that point on, the client will enter the game and is now part of the shared world, where the player will be able to see and interact with other clients. If a client disconnects at any point in time, then all of the other clients will be notified and the `Logout` function in the game mode will be called.

*   **Spawning the actors that all of the clients need to know about**: If you want to spawn an actor that exists in all of the clients, then you need to do that on the server. The reason for this is that the server has the authority and is the only one that can tell each client to create an instance of that actor.

This is the most common way of spawning actors in multiplayer since most actors need to exist in all of the clients. An example of this would be with a power-up, which is something that all clients can see and interact with.

*   **Running critical gameplay logic**: To make sure that the game is fair to all of the clients, the critical gameplay logic needs to be executed on the server only. If the clients were responsible for handling the deduction of health, it would be very exploitable, because a player could use a tool to change the current value of health to 100% all the time in memory, so the player would never die in the game.
*   **Handling variable replication**: If you have a replicated variable (covered in the *Understanding variable replication* section), then its value should only be changed on the server. This will ensure that all of the clients will have the value updated automatically. You can still change the value on the client, but it will always be replaced with the latest value from the server to prevent cheating and to make sure all of the clients are in sync.
*   **Handling RPCs from the client**: The server needs to process the **remote procedure calls** (**RPCs**) (covered in [*Chapter 17*](B18531_17.xhtml#_idTextAnchor386), *Using Remote Procedure Calls*) that are sent from the clients.

Now that you know what a server does, we can talk about the two different ways of creating a server in UE5.

## Dedicated server

The dedicated server only runs the server logic, so you won’t see the typical window with the game running where you control a character as a normal player. This means that all the clients will connect to this server and its only job is to coordinate them and execute the critical gameplay logic. Additionally, if you run the dedicated server with the `-log` command prompt, you’ll have a console window that logs relevant information about what is happening on the server, such as if a client has connected or disconnected, and so on. You, as a developer, can also log your information by using the `UE_LOG` macro.

Using dedicated servers is a very common way of creating servers for multiplayer games, and since it’s more lightweight than a listen server (covered in the next section), you could just host it on a server stack and leave it running. Another advantage of dedicated servers is that it will make the game fairer for all players because the network conditions will be the same for everyone, and also because none of the clients has authority, so the possibility of a hack is reduced.

To start a dedicated server in UE5, you can use the following command arguments:

*   Run the following command to start a dedicated server inside an editor through a shortcut or Command Prompt:

    ```cpp
    "<UE5 Install Folder>\Engine\Binaries\Win64\UnrealEditor.exe" 
    "<UProject Location>" <Map Name> -server -game -log
    ```

Here’s an example:

```cpp
"C:\Program Files\Epic 
Games\UE_5.0\Engine\Binaries\Win64\UnrealEditor.exe" 
"D:\TestProject\TestProject.uproject" TestMap -server -game -log
```

*   Creating a packaged dedicated server requires a build of the project that’s been built specifically to serve as a dedicated server.

Note

You can find out more about setting up a packaged dedicated server at [https://docs.unrealengine.com/5.0/en-US/InteractiveExperiences/Networking/HowTo/DedicatedServers/](https://docs.unrealengine.com/5.0/en-US/InteractiveExperiences/Networking/HowTo/DedicatedServers/).

## The listen server

The listen server acts as a server and client at the same time, so you’ll also have a window where you can play the game as a client with this server type. It also has the advantage of being the quickest way of getting a server running in a packaged build, but it’s not as lightweight as a dedicated server, so the number of clients that can be connected at the same time will be limited.

To start a listen server, you can use the following command arguments:

*   Run the following command to start a listen server inside an editor through a shortcut or Command Prompt:

    ```cpp
    "<UE5 Install Folder>\Engine\Binaries\Win64\UnrealEditor.exe" 
    "<UProject Location>" <Map Name>?Listen -game
    ```

Here’s an example:

```cpp
"C:\Program Files\Epic 
Games\UE_5.0\Engine\Binaries\Win64\UnrealEditor.exe" 
"D:\TestProject\TestProject.uproject" TestMap?Listen -game
```

*   Using a packaged development build through a shortcut or Command Prompt:

    ```cpp
    "<Project Name>.exe" <Map Name>?Listen -game
    ```

Here’s an example:

```cpp
"D:\Packaged\TestProject\TestProject.exe" TestMap?Listen –game
```

Now that you know about the two different types of servers you have in Unreal Engine, we can now move on to its counterpart – the client and its responsibilities.

# Understanding the client

The client is the simplest part of the architecture because most of the actors will have the authority on the server, so in those cases, the work will be done on the server and the client will just obey its orders.

Here is an overview of the main responsibilities of a client:

*   **Enforcing variable replication from the server**: The server typically has authority over all of the actors that the client knows, so when the value of a replicated variable is changed on the server, the client needs to enforce that value as well.
*   **Handling RPCs from the server**: The client needs to process the RPCs (covered in [*Chapter 17*](B18531_17.xhtml#_idTextAnchor386), *Using Remote Procedure Calls*) that are sent from the server.
*   **Predicting movement when simulating**: When a client is simulating an actor (covered in the *Getting to know roles* section), it needs to locally predict where it’s going to be based on the actor’s velocity.
*   **Spawning the actors that only a client needs to know about**: If you want to spawn an actor that only exists on a client, then you need to do it on that specific client.

This is the least common way of spawning actors since there are fewer cases where you want an actor to only exist on a client. An example of this is the placement preview actor you see in multiplayer survival games, where the player controls a semi-transparent version of a wall that other players can’t see until it’s placed.

A client can join a server in a couple of different ways. Here is a list of the most common methods:

*   By opening the UE5 console (by default, this can be done with the `` ` `` key) in a development build and typing the following:

    ```cpp
    open <Server IP Address>
    ```

Here’s an example:

```cpp
open 194.56.23.4
```

*   Using the `Execute Console Command` Blueprint node. An example is as follows:

![Figure 16.3 – Joining a server with an example IP with the Execute Console Command node ](img/Figure_16..03_B18531.jpg)

Figure 16.3 – Joining a server with an example IP with the Execute Console Command node

*   Using the `ConsoleCommand` function in `APlayerController`, as follows:

    ```cpp
    PlayerController->ConsoleCommand("open <Server IP Address>");
    ```

Here’s an example:

```cpp
PlayerController->ConsoleCommand("open 194.56.23.4");
```

*   Using the editor executable through a shortcut or Command Prompt:

    ```cpp
    "<UE5 Install Folder>\Engine\Binaries\Win64\UnrealEditor.exe" 
    "<UProject Location>" <Server IP Address> -game
    ```

Here’s an example:

```cpp
"C:\Program Files\Epic Games\UE_5.0\Engine\Binaries\Win64\UnrealEditor.exe" "D:\TestProject\TestProject.uproject" 194.56.23.4 -game
```

*   Using a packaged development build through a shortcut or Command Prompt:

    ```cpp
    "<Project Name>.exe" <Server IP Address>
    ```

Here’s an example:

```cpp
"D:\Packaged\TestProject\TestProject.exe" 194.56.23.4
```

In the following exercise, we will test the **Third Person** template that comes with UE5 in multiplayer.

## Exercise 16.01 – Testing the Third Person template in multiplayer

In this exercise, we’re going to create a **Third Person** template project and play it in multiplayer.

Follow these steps to complete this exercise:

1.  Create a new `TestMultiplayer` and save it to a location of your choosing.

Once the project has been created, it should open the editor. Now, let’s test the project in multiplayer to see how it behaves.

1.  In the editor, to the right of the **Play** button, you have a button with three vertical dots. Click on it and you should see a list of options. Under the **Multiplayer Options** section, you can configure how many clients you want and specify the net mode, which has the following options:
    *   **Play Standalone**: Runs the game in single player
    *   **Play As Listen Server**: Runs the game with a listen server
    *   **Play As Client**: Runs the game with a dedicated server
2.  Make sure the `3`, and click on **New Editor Window (PIE)**.

You should see three windows on top of each other representing the three clients:

![Figure 16.4 – Launching three client windows with a listen server ](img/Figure_16..04_B18531.jpg)

Figure 16.4 – Launching three client windows with a listen server

As you can see, the server window is bigger than the client windows, so let’s change its size. Press *Esc* on your keyboard to stop playing.

1.  Once again, click on the button with the three vertical dots next to the **Play** button and pick the last option, **Advanced Settings**.
2.  Search for the `640x480` and close the **Editor Preferences** tab.

Note

This option will only change the size of the server window. If you want to change the size of the client window, you can modify the value of the **Multiplayer Viewport Size** option, which you can find by scrolling down a bit more in the same menu.

1.  Play the game again; you should see the following:

![Figure 16.5 – Launching three client windows using a 640x480 resolution with a listen server  ](img/Figure_16..05_B18531.jpg)

Figure 16.5 – Launching three client windows using a 640x480 resolution with a listen server

Once you start playing, you’ll notice that the title bars of the windows say **Server**, **Client 1**, and **Client 2**. Since you can control a character in the **Server** window, that means we’re running a listen server, where you have the server and a client running in the same window. When that happens, you should interpret the window title as **Server + Client 0** instead of just **Server** to avoid confusion.

By completing this exercise, you have a setup where you have a server and three clients running (**Client 0**, **Client 1**, and **Client 2**).

Note

When you have multiple windows running at the same time, you’ll notice that you can only have input focus on one window at a time. To shift the focus to another window, just press *Shift* + *F1* to lose the current input focus, and then click on the new window you want to focus on.

If you play the game in one of the windows, you’ll notice that you can move around and jump. When you do those actions the other clients will also be able to see that.

The reason why everything works is because the character movement component, which comes with the character class, replicates the location, rotation, and falling state (used to determine whether you are in the air or not) for you automatically. If you want to add a custom behavior such as an attack animation, you can’t just tell the client to play an animation locally when a key is pressed, because that will not work on the other clients. That’s why you need the server – to serve as an intermediary and tell all the clients to play the animation when one client presses the key.

In this exercise, we’ve learned how to test multiplayer in the editor. Now, let’s learn how to do the same on a packaged build.

# Packaging the project

Once you’ve finished the project, it’s good practice to package it so that you have a pure standalone version that doesn’t use the Unreal Engine editor. This will run faster and be more lightweight.

Follow these steps to create the packaged version of the file in *Exercise 16.01 – Testing the Third Person template in multiplayer*:

1.  Go to **Platforms** (to the right of the **Play** button) | **Windows** | **Package Project**.
2.  Pick a folder to place the packaged build and wait for it to finish.
3.  Once it has finished, go to the selected folder and open the `Windows` folder inside it.
4.  Right-click on **TestMultiplayer.exe** and pick **Create Shortcut**.
5.  Rename the new shortcut **Run Server**.
6.  Right-click on it and pick **Properties**.
7.  On the target, append `ThirdPersonMap?Listen -server`, which will create a listen server using `ThirdPersonMap`. You should end up with this:

    ```cpp
    "<Packaged Path>\Windows\TestMultiplayer.exe" 
      ThirdPersonMap?Listen -server
    ```

8.  Click **OK** and run the shortcut.
9.  You should get a Windows Firewall prompt; allow it.
10.  Leave the server running, go back to the folder (using ALT+TAB or pressing the Windows Key and selecting another window from the taskbar), and create another shortcut from **TestMultiplayer.exe**.
11.  Rename it `Run Client`.
12.  Right-click on it and pick **Properties**.
13.  On the target, append `127.0.0.1`, which is the IP of your local server. You should end up with `"<Packaged Path>\Windows\TestMultiplayer.exe" 127.0.0.1`.
14.  Click **OK** and run the shortcut.

You are now connected to the listen server, which means you can see each other’s characters. Every time you click on the **Run Client** shortcut, you’ll add a new client to the server so that you can have a few clients running on the same machine.

Once you are done testing the packaged build, you can hit ALT+F4 to close each window.

Now that we know how to test our packaged project in multiplayer, let’s take a look at connections and ownership, which allow us to have a two-way communication line between the server and the client.

# Exploring connections and ownership

When using multiplayer in Unreal Engine, an important concept to understand is that of a connection. When a client joins a server, it will get a new Player Controller with a connection associated with it.

If an actor doesn’t have a valid connection with the server, then it won’t be able to do replication operations such as variable replication (covered in the *Understanding variable replication* section) or call RPCs (covered in [*Chapter 17*](B18531_17.xhtml#_idTextAnchor386), *Using Remote Procedure Calls*).

If the Player Controller is the only actor that holds a connection, then does that mean that it’s the only place you can do replication operations? No, and that’s where the `GetNetConnection` function, defined in `AActor`, comes into play.

When doing replication operations (such as variable replication or calling RPCs) on an actor, the network framework will get the actor’s connection by calling the `GetNetConnection()` function on it. If the connection is valid, then the replication operation will be processed; if it’s not, nothing will happen. The most common implementations of `GetNetConnection()` are from `APawn` and `AActor`.

Let’s take a look at how the `APawn` class implements the `GetNetConnection()` function, which is typically used for characters:

```cpp
class UNetConnection* APawn::GetNetConnection() const
{
  // If we have a controller, then use its net connection
  if ( Controller )
  {
    return Controller->GetNetConnection();
  }
  return Super::GetNetConnection();
}
```

The preceding implementation, which is part of the UE5 source code, will first check whether the pawn has a valid controller. If the controller is valid, then it will use its connection. If the controller is not valid, then it will use the parent implementation of the `GetNetConnection()` function, which is on `AActor`:

```cpp
UNetConnection* AActor::GetNetConnection() const
{
  return Owner ? Owner->GetNetConnection() : nullptr;
}
```

The preceding implementation, which is also part of the UE5 source code, will check if the actor has a valid owner. If it does, it will use the owner’s connection; if it doesn’t, it will return an invalid connection. So, what is this `Owner` variable? Every actor has a variable called `Owner` (where you can set its value by calling the `SetOwner` function) that stores which actor owns it, so you can think of it as its parent actor.

Note

In a listen server, the connection for the character that’s controlled by its client will always be invalid. This is because that client is already a part of the server and therefore doesn’t need a connection.

Using the owner’s connection in this implementation of `GetNetConnection()` will work like a hierarchy. If, while going up the hierarchy of owners, it finds an owner that is a Player Controller or is being controlled by one, then it will have a valid connection and will be able to process replication operations. Have a look at the following example:

Imagine that a weapon actor was placed in the world and it’s just sitting there. In that situation, the weapon won’t have an owner, so if the weapon tries to do any replication operations, such as variable replication or calling RPCs, nothing will happen.

However, if a client picks up the weapon and calls `SetOwner` on the server with the value of the character, then the weapon will now have a valid connection. The reason for this is because the weapon is an actor, so to get its connection, it will use the `AActor` implementation of `GetNetConnection()`, which returns the connection of its owner. Since the owner is the client’s character, it will use the implementation of `GetNetConnection()` of `APawn`. The character has a valid Player Controller, so that is the connection returned by the function.

Here is a diagram to help you understand this logic:

![Figure 16.6 – Connections and ownership example of a weapon actor ](img/Figure_16..06_B18531.jpg)

Figure 16.6 – Connections and ownership example of a weapon actor

If the weapon has an invalid owner, then this is what will happen:

*   `AWeapon` doesn’t override the `GetNetConnection` function, so it will call the first implementation found in the class hierarchy, which is `AActor::GetNetConnection`.
*   The implementation of `AActor::GetNetConnection` calls `GetNetConnection` on its owner. Since there is no owner, the connection is invalid.

If the weapon has an valid owner, then this is what will happen:

*   `AWeapon` doesn’t override the `GetNetConnection` function, so it will call the first implementation found in the class hierarchy, which is `AActor::GetNetConnection`.
*   The implementation of `AActor::GetNetConnection` calls `GetNetConnection` on its owner. Since the owner of the weapon is the character that picked it up, then it will call `GetNetConnection` on it.
*   `ACharacter` doesn’t override the `GetNetConnection` function, so it will call the first implementation found in the class hierarchy, which is `APawn::GetNetConnection`.
*   The implementation of `APawn::GetNetConnection` uses the connection from the owning player controller. Since the owning player controller is valid, then it will use that connection for the weapon.

Note

For `SetOwner` to work as intended, it needs to be executed on the authority, which, in most cases, means the server. If you execute `SetOwner` on a game instance that is not the authority, it won’t be able to execute replication operations.

In this section, we learned how connections and ownership allow the server and client to communicate in both directions. Next, we’re going to learn about the concept of the roles of an actor, which tells us the version of the actor that is executing the code.

# Getting to know roles

When an actor is spawned on the server, it will create a version on the server, as well as one on each client. Since there are different versions of the same actor on different instances of the game (`Server`, `Client 1`, `Client 2`, and so on), it is important to know which version of the actor is which. This will allow us to know what logic can be executed in each of these instances.

To help with this situation, every actor has the following two variables:

*   `GetLocalRole()` function.
*   `GetRemoteRole()` function.

The return type of the `GetLocalRole()` and `GetRemoteRole()` functions is `ENetRole`, which is an enumeration that can have the following possible values:

*   `ROLE_None`: The actor doesn’t have a role because it’s not being replicated.
*   `ROLE_SimulatedProxy`: The current game instance doesn’t have authority over the actor and it’s not being controlled by a Player Controller. This means that its movement will be simulated/predicted by using the last value of the actor’s velocity.
*   `ROLE_AutonomousProxy`: The current game instance doesn’t have authority over the actor, but it’s being controlled by a Player Controller. This means that we can send more accurate movement information to the server, based on the player’s inputs, instead of just using the last value of the actor’s velocity.
*   `ROLE_Authority`: The current game instance has complete authority over the actor. This means that if the actor is on the server, the changes that are made to its replicated variables will be treated as the value that every client needs to enforce through variable replication.

Let’s have a look at the following example code snippet:

```cpp
ENetRole MyLocalRole = GetLocalRole();
ENetRole MyRemoteRole = GetRemoteRole();
FString String;
if(MyLocalRole == ROLE_Authority)
{
  if(MyRemoteRole == ROLE_AutonomousProxy)
  {
    String = "This version of the actor is the authority 
    and it's being controlled by a player on its client";
  }
  else if(MyRemoteRole == ROLE_SimulatedProxy)
  {
    String = "This version of the actor is the authority 
    but it's not being controlled by a player on its 
    client";
  }
}
else String = "This version of the actor isn't the authority";
GEngine->AddOnScreenDebugMessage(-1, 0.0f, FColor::Red, String);
```

The preceding code snippet will store the values of the local role and remote role in `MyLocalRole` and `MyRemoteRole`, respectively. After that, it will print different messages on the screen, depending on whether that version of the actor is the authority or whether it’s being controlled by a player on its client.

Note

It is important to understand that if an actor has a local role of `ROLE_Authority`, it doesn’t mean that it’s on the server; it means that it’s on the game instance that originally spawned it and therefore has authority over it.

If a client spawns an actor, even though the server and the other clients won’t know about it, its local role will still be `ROLE_Authority`. Most of the actors in a multiplayer game will be spawned by the server; that’s why it’s easy to misunderstand that the authority is always referring to the server.

Here is a table to help you understand the roles that an actor will have in different scenarios:

![Figure 16.7 – Roles that an actor can have in different scenarios ](img/Figure_16..07_B18531.jpg)

Figure 16.7 – Roles that an actor can have in different scenarios

In the preceding table, you can see the roles that an actor will have in different scenarios.

We’ll analyze each scenario and explain why the actor has that role in the following sections.

## Actor spawned on the server

The actor spawns on the server, so the server’s version of that actor will have the local role of `ROLE_Authority` and the remote role of `ROLE_SimulatedProxy`. For the client’s version of that actor, its local role will be `ROLE_SimulatedProxy` and the remote role will be `ROLE_Authority`.

## Actor spawned on the client

The actor was spawned on the client, so the client’s version of that actor will have the local role of `ROLE_Authority` and the remote role of `ROLE_SimulatedProxy`. Since the actor wasn’t spawned on the server, then it will only exist on the client that spawned it.

## Player-owned pawn spawned on the server

The pawn was spawned on the server, so the server’s version of that pawn will have the local role of `ROLE_Authority` and the remote role of `ROLE_AutonomousProxy`. For the client’s version of that pawn, its local role will be `ROLE_AutonomousProxy`, because it’s being controlled by a Player Controller, and the remote role of `ROLE_Authority`.

## Player-owned pawn spawned on the client

The pawn was spawned on the client, so the client’s version of that pawn will have the local role of `ROLE_Authority` and the remote role of `ROLE_SimulatedProxy`. Since the pawn wasn’t spawned on the server, then it will only exist on the client that spawned it.

## Exercise 16.02 – Implementing ownership and roles

In this exercise, we’re going to create a **C++** project that uses the **Third Person** template as a base and make it do the following:

*   Create a new actor called `EditAnywhere` variable called `OwnershipRadius`) and will set that character as its owner. When no character is within the radius, then the owner will be `nullptr`.
*   Display its local role, remote role, owner, and connection.

*   Edit **OwnershipRolesCharacter** and override the **Tick** function so that it displays its local role, remote role, owner, and connection.*   Add a macro called `ENetRole` into an `FString` value that can be printed on the screen.

Follow these steps to complete this exercise:

1.  Create a new **Third Person** template project using **C++** called **OwnershipRoles** and save it to a location of your liking.
2.  Once the project has been created, it should open the editor as well as the Visual Studio solution.
3.  Using the editor, create a new C++ class called `Actor`.
4.  Once it finishes compiling, Visual Studio should pop up with the newly created `.h` and `.cpp` files.
5.  Close the editor and go back to Visual Studio.
6.  In Visual Studio, open the `OwnershipRoles.h` file and add the following macro:

    ```cpp
    #define ROLE_TO_STRING(Value) FindObject<UEnum>(ANY_PACKAGE, TEXT("ENetRole"), true)->GetNameStringByIndex(static_cast<int32>(Value))
    ```

This macro will be used to convert the `ENetRole` enumeration that we get from the `GetLocalRole()` function and `GetRemoteRole()` into an `FString`. The way it works is by finding the `ENetRole` enumeration type through Unreal Engine’s reflection system. From there, it converts the `Value` parameter into an `FString` variable so that it can be printed on the screen.

1.  Now, open the `OwnershipTestActor.h` file and declare the protected variables for the static mesh component and the ownership radius, as shown in the following code snippet:

    ```cpp
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Ownership Test Actor")
    UStaticMeshComponent* Mesh;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Ownership Test Actor")
    float OwnershipRadius = 400.0f;
    ```

In the preceding code snippet, we declared the static mesh component and the `OwnershipRadius` variable, which allows you to configure the radius of the ownership.

1.  Next, delete the declaration of `BeginPlay` and move the constructor and the `Tick` function declarations to the protected area.
2.  Now, open the `OwnershipTestActor.cpp` file and add the required header files, as shown in the following code snippet:

    ```cpp
    #include "OwnershipRoles.h"
    #include "OwnershipRolesCharacter.h"
    #include "Kismet/GameplayStatics.h"
    ```

In the preceding code snippet, we included `OwnershipRoles.h`, `OwnershipRolesCharacter.h`, and `GameplayStatics.h` because we’ll be calling the `GetAllActorsOfClass` function.

1.  In the constructor definition, create the static mesh component and set it as the root component:

    ```cpp
    Mesh = CreateDefaultSubobject<UStaticMeshComponent>("Mesh");
    RootComponent = Mesh;
    ```

2.  Still in the constructor, set `bReplicates` to `true` to tell Unreal Engine that this actor replicates and should also exist in all of the clients:

    ```cpp
    bReplicates = true;
    ```

3.  Delete the `BeginPlay` function definition.
4.  In the `Tick` function, draw a debug sphere to help visualize the ownership radius, as shown in the following code snippet:

    ```cpp
    DrawDebugSphere(GetWorld(), GetActorLocation(), OwnershipRadius, 32, FColor::Yellow);
    ```

5.  Still in the `Tick` function, create the authority-specific logic that will get the closest `AOwnershipRolesCharacter` within the ownership radius. If it’s different from the current one, set it as the owner:

    ```cpp
    if (HasAuthority())
    {
      AActor* NextOwner = nullptr;
      float MinDistance = OwnershipRadius;
      TArray<AActor*> Actors;
      UGameplayStatics::GetAllActorsOfClass(this,
        AOwnershipRolesCharacter::StaticClass(), Actors);
      for (AActor* Actor : Actors)
      {
        const float Distance = GetDistanceTo(Actor);
        if (Distance <= MinDistance)
        {
          MinDistance = Distance;
          NextOwner = Actor;
        }
      }
      if (GetOwner() != NextOwner)
      {
        SetOwner(NextOwner);
      }
    }
    ```

Note

The preceding code is for demonstration purposes only, because running `GetAllActorsOfClass` on the `Tick` function every frame will take a big toll on performance. Ideally, you should execute this code only once (on `BeginPlay`, for example) and store the values so that we can query them in `Tick`.

1.  Still in the `Tick` function, convert the values for the local/remote roles (using the `ROLE_TO_STRING` macro we created earlier), the current owner, and the connection into strings:

    ```cpp
    const FString LocalRoleString = ROLE_TO_STRING(GetLocalRole());
    const FString RemoteRoleString = ROLE_TO_STRING(GetRemoteRole());
    const FString OwnerString = GetOwner() != nullptr ? GetOwner()->GetName() : TEXT("No Owner");
    const FString ConnectionString = GetNetConnection() != nullptr ? TEXT("Valid Connection") : TEXT("Invalid Connection");
    ```

2.  To finalize the `Tick` function, use `DrawDebugString` to print the strings we converted in the previous step on the screen:

    ```cpp
    const Fstring Values = Fstring::Printf(TEXT("LocalRole = %s\nRemoteRole = %s\nOwner = %s\nConnection = %s"), 
      *LocalRoleString, *RemoteRoleString, *OwnerString, 
      *ConnectionString);
    DrawDebugString(GetWorld(), GetActorLocation(), Values, nullptr, Fcolor::White, 0.0f, true);
    ```

Note

Instead of constantly using `GetLocalRole() == ROLE_Authority` to check whether the actor has authority, you can use the `HasAuthority()` helper function, defined in `AActor`.

1.  Next, open `OwnershipRolesCharacter.h` and declare the `Tick` function as protected:

    ```cpp
    virtual void Tick(float DeltaTime) override;
    ```

2.  Now, open `OwnershipRolesCharacter.cpp` and include `OwnershipRoles.h`, as shown in the following code snippet:

    ```cpp
    #include "OwnershipRoles.h"
    ```

3.  Implement the `Tick` function:

    ```cpp
    void AOwnershipRolesCharacter::Tick(float DeltaTime)
    {
      Super::Tick(DeltaTime);
    }
    ```

4.  Inside of the Tick function, convert the values for the local/remote roles (using the `ROLE_TO_STRING` macro we created earlier), the current owner, and the connection into strings:

    ```cpp
    const FString LocalRoleString = ROLE_TO_STRING(GetLocalRole());
    const FString RemoteRoleString = ROLE_TO_STRING(GetRemoteRole());
    const FString OwnerString = GetOwner() != nullptr ? GetOwner()- >GetName() : TEXT("No Owner");
    const FString ConnectionString = GetNetConnection() != nullptr ? 
      TEXT("Valid Connection") : TEXT("Invalid 
      Connection");
    ```

5.  Use `DrawDebugString` to print the strings we converted in the previous step on the screen:

    ```cpp
    const FString Values = FString::Printf(TEXT("LocalRole = 
      %s\nRemoteRole = %s\nOwner = %s\nConnection = %s"), 
      *LocalRoleString, *RemoteRoleString, *OwnerString, 
      *ConnectionString);
    DrawDebugString(GetWorld(), GetActorLocation(), Values, nullptr, FColor::White, 0.0f, true);
    ```

Finally, we can test the project.

1.  Run the code and wait for the editor to fully load.
2.  Create a new Blueprint called `OwnershipTestActor_BP` in the `Content` folder that derives from `OwnershipTestActor`. Set `Mesh` to use a cube mesh, and drop an instance of it in the world.
3.  Go to `Multiplayer Options`, set `2`.
4.  Set the window size to `800x600`.
5.  Play using **New Editor Window (PIE)**.

You should get the following output:

![Figure 16.8 – Expected result on the Server and Client 1 windows ](img/Figure_16..08_B18531.jpg)

Figure 16.8 – Expected result on the Server and Client 1 windows

By completing this exercise, you’ll have a better understanding of how connections and ownership work. These are important concepts to know as everything related to replication is dependent on them.

The next time you see that an actor is not doing replication operations, you’ll know that you need to check whether it has a valid connection and an owner.

Now, let’s analyze the values that are displayed in the server and client windows.

Note

The two figures for the server and client window will have three text blocks that say `Server Character`, `Client 1 Character`, and `Ownership Test Actor`, but that was added to the original screenshot to help you understand which character and actor are which.

## Output for the Server window

Have a look at the following output screenshot of the **Server** window from the previous exercise:

![Figure 16.9 – The Server window ](img/Figure_16..09_B18531.jpg)

Figure 16.9 – The Server window

In the preceding screenshot, you can see **Server Character**, **Client 1 Character**, and the **Ownership Test** cube actor.

First, let’s analyze the values for **Server Character**.

## Server Character

This is the character that the listen server is controlling. The values associated with this character are as follows:

*   `LocalRole = ROLE_Authority`: This character was spawned on the server, which is the current game instance.
*   `RemoteRole = ROLE_SimulatedProxy`: Because this character was spawned on the server, the other clients should only simulate it.
*   `Owner = PlayerController_0`: This character is being controlled by the client of the listen server, which uses the first `PlayerController` instance called `PlayerController_0`.
*   `Connection = Invalid Connection`: Because we’re the client of the listen server, there is no need for a connection.

Next, we are going to look at **Client 1 Character** in the same window.

## Client 1 Character

This is the character that **Client 1** is controlling. The values associated with this character are as follows:

*   `LocalRole = ROLE_Authority`: This character was spawned on the server, which is the current game instance.
*   `RemoteRole = ROLE_AutonomousProxy`: Because this character was spawned on the server, but it’s being controlled by another client.
*   `Owner = PlayerController_1`: This character is being controlled by another client, which uses the second `PlayerController` instance called `PlayerController_1`.
*   `Connection = Valid Connection`: Because this character is being controlled by another client, so a connection to the server is required.

Next, we are going to look at the **OwnershipTest** actor in the same window.

## The OwnershipTest actor

This is the cube actor that will set its owner to the closest character within a certain ownership radius. The values associated with this actor are as follows:

*   `LocalRole = ROLE_Authority`: This actor was placed in the level and spawned on the server, which is the current game instance.
*   `RemoteRole = ROLE_SimulatedProxy`: This actor was spawned in the server, but it’s not being controlled by any client.
*   `Owner` and `Connection`: They will have their values based on the closest character. If there isn’t a character inside the ownership radius, then they will have the values of `No Owner` and `Invalid Connection`, respectively.

Now, let’s analyze the values that are displayed in the `Client 1` window.

## Output for the Client 1 window

Have a look at the following output screenshot of the `Client 1` window from the previous exercise:

![Figure 16.10 – The Client 1 window ](img/Figure_16..10_B18531.jpg)

Figure 16.10 – The Client 1 window

The values for the `Client 1` window will be the same as those for the `Server` window, except the values of `LocalRole` and `RemoteRole` will be reversed because they are always relative to the game instance that you are in.

Another exception is that the server character has no owner and the other connected clients won’t have a valid connection. The reason for that is that clients don’t store player controllers and connections of other clients, only the server does, but this will be covered in more depth in [*Chapter 18*](B18531_18.xhtml#_idTextAnchor404), *Using Gameplay Framework Classes in Multiplayer*.

In this section, we’ve covered how roles are used to know which version of the actor the code is executing, which we can leverage to run specific code. In the next section, we will look at variable replication, which is one of the techniques that’s used by the server to keep the clients synchronized.

# Understanding variable replication

One of the ways the server can keep the clients synchronized is by using variable replication. The way it works is that every specific number of times per second (defined per actor in the `AActor::NetUpdateFrequency` variable, which is also exposed to blueprints) the variable replication system in the server will check whether there are any replicated variables (explained in the next section) in the client that need to be updated with the latest value.

If the variable meets all of the replication conditions, then the server will send an update to the client and enforce the new value.

For example, if you have a replicated `Health` variable and the client on its end uses a hacking tool to set the value of the variable from `10` to `100`, then the replication system will enforce the real value from the server and change it back to `10`, which nullifies the hack.

Variables are only sent to the client to be updated in the following situations:

*   The variable is set to replicate.
*   The value was changed on the server.
*   The value on the client is different on the server.
*   The actor has replication enabled.
*   The actor is relevant and meets all of the replication conditions.

One important thing to take into consideration is that the logic that determines whether a variable should be replicated or not is only executed `Actor::NetUpdateFrequency` times a second. In other words, the server doesn’t send an update request to a client immediately after you change the value of a variable on the server.

An example of this would be if you had an integer replicated variable called `Test` that has a default value of `5`. If you call a function on the server that sets `Test` to `3` and in the next line changes it to `8`, then only the latter change would send an update request to the clients. The reason for this is these two changes were made in-between the `NetUpdateFrequency` interval, so when the variable replication system executes, the current value is `8`, and because that is different from the value stored on the clients (which is still `5`), it will update them. If instead of setting it to `8`, you set it back to `5`, then no changes would be sent to the clients because the values haven’t changed.

In the following sections, we are going to cover how to replicate variables by using the `Replicated` and `ReplicatedUsing` specifiers, as well as the `DOREPLIFETIME` and `DOREPLIFETIME_CONDITION` macros.

## Replicated variables

In Unreal Engine, almost any variable type that can use the `UPROPERTY` macro can be set to replicate, and you can use two specifiers to do that. We will look at them in the following sections.

### Replicated

If you just want to say that a variable is replicated, then you can use the `Replicated` specifier.

Have a look at the following example:

```cpp
UPROPERTY(Replicated) 
float Health = 100.0f; 
```

In the preceding code snippet, we declared a float variable called `Health`, as we normally do. The difference is that we added `UPROPERTY(Replicated)` to let Unreal Engine know that the `Health` variable will be replicated.

### ReplicatedUsing

If you want to say that a variable is replicated and should call a function every time it’s updated, then you can use the `ReplicatedUsing=<Function Name>` specifier. Have a look at the following example:

```cpp
UPROPERTY(ReplicatedUsing=OnRep_Health) 
float Health = 100.0f;
UFUNCTION() 
void OnRep_Health()
{
  UpdateHUD(); 
}
```

In the preceding code snippet, we declared a float variable called `Health`. The difference is that we added `UPROPERTY(ReplicatedUsing=OnRep_Health)` to let Unreal Engine know that this variable will be replicated and should call the `OnRep_Health` function every time it’s updated, which, in this specific case, calls a function to update the HUD.

Typically, the naming scheme for the callback function is `OnRep_<Variable Name>`.

Note

The function that’s used in the `ReplicatedUsing` specifier needs to be marked as `UFUNCTION()`.

### GetLifetimeReplicatedProps

Besides marking the variable as replicated, you’ll also need to implement the `GetLifetimeReplicatedProps` function in the actor’s `cpp` file. One thing to take into consideration is that this function is automatically declared internally once you have at least one replicated variable, so you shouldn’t declare it in the actor’s header file. The purpose of this function is for you to tell how each replicated variable should replicate. You can do this by using the `DOREPLIFETIME` macro and its variants on every variable that you want to replicate.

### DOREPLIFETIME

This macro specifies that the replicated variable in a class (entered as arguments) will replicate to all the clients, without an extra condition.

Here’s its syntax:

```cpp
DOREPLIFETIME(<Class Name>, <Replicated Variable Name>); 
```

Have a look at the following example:

```cpp
void AVariableReplicationActor::GetLifetimeReplicatedProps(TArray< 
  FLifetimeProperty >& OutLifetimeProps) const
{
  Super::GetLifetimeReplicatedProps(OutLifetimeProps);
  DOREPLIFETIME(AVariableReplicationActor, Health);
}
```

In the preceding code snippet, we used the `DOREPLIFETIME` macro to tell the replication system that the `Health` variable in the `AVariableReplicationActor` class will replicate without an extra condition.

### DOREPLIFETIME_CONDITION

This macro specifies that the replicated variable in a class (entered as arguments) will replicate only to the clients that meet the condition (entered as an argument).

Here’s the syntax:

```cpp
DOREPLIFETIME_CONDITION(<Class Name>, <Replicated Variable Name>, <Condition>); 
```

The condition parameter can be one of the following values:

*   `COND_InitialOnly`: The variable will only replicate once, with the initial replication.
*   `COND_OwnerOnly`: The variable will only replicate to the owner of the actor.
*   `COND_SkipOwner`: The variable won’t replicate to the owner of the actor.
*   `COND_SimulatedOnly`: The variable will only replicate to actors that are simulating.
*   `COND_AutonomousOnly`: The variable will only replicate to autonomous actors.
*   `COND_SimulatedOrPhysics`: The variable will only replicate to actors that are simulating or to actors with `bRepPhysics` set to true.
*   `COND_InitialOrOwner`: The variable will only replicate once, with the initial replication or to the owner of the actor.
*   `COND_Custom`: The variable will only replicate if its `SetCustomIsActiveOverride` Boolean condition (used in the `AActor::PreReplication` function) is true.

Have a look at the following example:

```cpp
void AVariableReplicationActor::GetLifetimeReplicatedProps(TArray< 
  FLifetimeProperty >& OutLifetimeProps) const
{
  Super::GetLifetimeReplicatedProps(OutLifetimeProps);
  DOREPLIFETIME_CONDITION(AVariableReplicationActor, 
  Health, COND_OwnerOnly);
}
```

In the preceding code snippet, we used the `DOREPLIFETIME_CONDITION` macro to tell the replication system that the `Health` variable in the `AVariableReplicationActor` class will replicate only for the owner of this actor.

Note

There are more `DOREPLIFETIME` macros available, but they won’t be covered in this book. To see all of the variants, please check the `UnrealNetwork.h` file from the UE5 source code at [https://github.com/EpicGames/UnrealEngine/blob/release/Engine/Source/Runtime/Engine/Public/Net/UnrealNetwork.h](https://github.com/EpicGames/UnrealEngine/blob/release/Engine/Source/Runtime/Engine/Public/Net/UnrealNetwork.h).

Now that you have an idea of how variable replication works, let’s complete an exercise that uses the `Replicated` and `ReplicatedUsing` specifiers, as well as the `DOREPLIFETIME` and `DOREPLIFETIME_CONDITION` macros.

## Exercise 16.03 – Replicating variables using Replicated, ReplicatedUsing, DOREPLIFETIME, and DOREPLIFETIME_CONDITION

In this exercise, we’re going to create a **C++** project that uses the **Third Person** template as a base and add two variables to the character that replicate in the following way:

*   Variable `A` is a float that will use the `Replicated` specifier and the `DOREPLIFETIME` macro.
*   Variable `B` is an integer that will use the `ReplicatedUsing` specifier and the `DOREPLIFETIME_CONDITION` macro.
*   The Tick function of the character should increment A and B by 1 every frame, if it has authority, and call `DrawDebugString` to display theirs values on the location of the character.

Follow these steps to complete this exercise:

1.  Create a new `VariableReplication` and save it to a location of your choosing.
2.  Once the project has been created, it should open the editor as well as the Visual Studio solution.
3.  Close the editor and go back to Visual Studio.
4.  Open the `VariableReplicationCharacter.h` file and declare the protected `A` and `B` variables as `UPROPERTY` using their respective replication specifiers:

    ```cpp
    UPROPERTY(Replicated) 
    float A = 100.0f; 
    UPROPERTY(ReplicatedUsing = OnRepNotify_B) 
    int32 B; 
    ```

5.  Declare the `Tick` function as protected:

    ```cpp
    virtual void Tick(float DeltaTime) override;
    ```

6.  Since we’ve declared the `B` variable as `ReplicatedUsing = OnRepNotify_B`, we also need to declare the protected `OnRepNotify_B` callback function as `UFUNCTION`:

    ```cpp
    UFUNCTION() 
    void OnRepNotify_B(); 
    ```

7.  Now, open the `VariableReplicationCharacter.cpp` file and include the `UnrealNetwork.h` header file, which contains the definition of the `DOREPLIFETIME` macros that we’re going to use:

    ```cpp
    #include "Net/UnrealNetwork.h"
    ```

8.  Implement the `GetLifetimeReplicatedProps` function:

    ```cpp
    void AVariableReplicationCharacter::GetLifetimeReplicatedProps (TArray<FLifetimeProperty >& OutLifetimeProps) const
    {
      Super::GetLifetimeReplicatedProps(OutLifetimeProps);
    }
    ```

9.  Let the replication system know that the `A` variable won’t have any extra replication conditions:

    ```cpp
    DOREPLIFETIME(AVariableReplicationCharacter, A);
    ```

10.  Let the replication system know that the `B` variable will only replicate to the owner of this actor:

    ```cpp
    DOREPLIFETIME_CONDITION(AVariableReplicationCharacter, B, COND_OwnerOnly);
    ```

11.  Implement the `Tick` function:

    ```cpp
    void AVariableReplicationCharacter::Tick(float DeltaTime) 
    {
      Super::Tick(DeltaTime);
    }
    ```

12.  Next, run the authority-specific logic that adds `1` to `A` and `B`:

    ```cpp
    if (HasAuthority()) 
    { 
      A++; 
      B++; 
    } 
    ```

Since this character will be spawned on the server, only the server will execute this logic.

1.  Display the values of `A` and `B` on the location of the character:

    ```cpp
    const FString Values = FString::Printf(TEXT("A = %.2f    B = %d"), A, B); 
    DrawDebugString(GetWorld(), GetActorLocation(), Values, nullptr, FColor::White, 0.0f, true);
    ```

2.  Implement the `RepNotify` function for the `B` variable, which displays a message on the screen saying that the `B` variable was changed to a new value:

    ```cpp
    void AVariableReplicationCharacter::OnRepNotify_B() 
    {
      const FString String = FString::Printf(TEXT("B was 
      changed by the server and is now %d!"), B); 
      GEngine->AddOnScreenDebugMessage(-1, 0.0f, 
      FColor::Red,String); 
    }
    ```

Finally, you can test the project.

1.  Run the code and wait for the editor to fully load.
2.  Go to `Multiplayer Options`, set `2`.
3.  Set the window size to `800x600`.
4.  Play using `New Editor Window (PIE)`.

By completing this exercise, you will be able to play on each client and you’ll notice that the characters are displaying their respective values for `A` and `B`.

Now, let’s analyze the values that are displayed in the `Server` and `Client 1` windows.

Note

The two figures for the server and client window will have two text blocks that say `Server Character` and `Client 1 Character`, but that was added to the original screenshot to help you understand which character is which.

## Output for the Server window

In the **Server** window, you have the values for **Server Character**, which is the character controlled by the server, while in the background, you have the values for **Client 1 Character**:

![Figure 16.11 – The Server window ](img/Figure_16..11_B18531.jpg)

Figure 16.11 – The Server window

The outputs that can be observed are as follows:

*   `A = 651.00 B = 551`
*   `A = 592.00 B = 492`

At this specific point in time, `651` for `A` and `551` for `B`. The reason why `A` and `B` have different values is that `A` starts at `100` and `B` starts at `0`, which is the correct value after `551` ticks of `A++` and `B++`.

`59` ticks of `A++` and `B++`.

Next, we will look at the **Client 1** window.

## Output for the Client 1 window

In the **Client 1** window, you have the values for **Client 1 Character**, which is the character controlled by **Client 1**, while in the background, you have the values for **Server Character**:

![Figure 16.12 – The Client 1 window ](img/Figure_16..12_B18531.jpg)

Figure 16.12 – The Client 1 window

The outputs that can be observed are as follows:

*   `A = 651.00 B = 0`
*   `A = 592.00 B = 492`

`A` is `651`, which is correct, but `B` is `0`. The reason for this is that `A` is using `DOREPLIFETIME`, which doesn’t add any additional replication conditions, so it will replicate the variable and keep the client up to date every time the variable is changed on the server.

The `B` variable, on the other hand, uses `DOREPLIFETIME_CONDITION` with `COND_OwnerOnly`, and since `0`.

If you go back to the code and change the replication condition of `B` to use `COND_SimulatedOnly` instead of `COND_OwnerOnly`, you’ll notice that the results will be reversed in the `B` will be replicated for **Server Character**, but it won’t replicate for its own character.

Note

The reason why the `RepNotify` message is showing in the **Server** window instead of the **Client** window is that, when playing in the editor, both windows share the same process, so printing text on the screen won’t be accurate. To get the correct behavior, you’ll need to run the packaged version of the game.

# Exploring 2D Blend Spaces

In [*Chapter 2*](B18531_02.xhtml#_idTextAnchor043), *Working with Unreal Engine*, we created a 1D Blend Space to blend between the movement states (idle, walk, and run) of a character based on the value of the `Speed` axis. For that specific example, it worked pretty well because you only needed one axis, but if we wanted the character to also be able to strafe, then we couldn’t do that.

To contemplate that case, Unreal Engine allows you to create 2D Blend Spaces. The concept is almost the same; the only difference is that you have an extra axis for animations, so you can blend between them not only horizontally, but also vertically.

Let’s apply our knowledge of 1D Blend Spaces to the next exercise, where we will create a 2D Blend Space for the movement of a character that can also strafe.

## Exercise 16.04 – Creating a movement 2D Blend Space

In this exercise, we’re going to create a Blend Space that uses two axes instead of one. The vertical axis will be `Speed`, which will be between `0` and `200`. The horizontal axis will be `Direction`, which represents the relative angle (`-180 to 180`) between the velocity and the rotation/forward vector of the pawn.

The following diagram will help you calculate the direction in this exercise:

![Figure 16.13 – Direction values based on the angle between the forward vector and the velocity ](img/Figure_16..13_B18531.jpg)

Figure 16.13 – Direction values based on the angle between the forward vector and the velocity

The preceding diagram shows how the direction will be calculated. The forward vector represents the direction that the character is currently facing, while the numbers represent the angle that the forward vector would make with the velocity vector if it was pointing in that direction. If the character was looking in a certain direction and you pressed a key to move the character to the right, then the velocity vector would be perpendicular to the forward vector. This would mean that the angle would be 90º, so that would be our direction.

If we set up our 2D Blend Space with that logic in mind, we can use the correct animation based on the character’s movement angle.

Follow these steps to complete this exercise:

1.  Create a new `Blendspace2D` and save it to a location of your choosing.
2.  Once the project has been created, it should open the editor.
3.  Next, you will be importing the movement animations. In the editor, go to the **Content\Characters\Mannequins\Animations** folder.
4.  Click on the **Import** button.
5.  Go to the `fbx` files, and hit the **Open** button.
6.  In the import dialog, make sure you pick the `SK_Mannequin` skeleton and hit the **Import All** button.
7.  Save all of the new files in the `Assets` folder.

If you open any of the new animations, you will notice that the mesh is quite stretched on the *Z*-axis. So, let’s fix that by adjusting the skeleton retargeting settings.

1.  Go to **Content/Characters/Mannequins/Meshes/SK_Mannequin**. On the left, you should see the list of bones.
2.  Click on the cog icon to the right of the search box on the top and enable **Show Retargeting Options**.
3.  *Right-click* on the `root` bone and pick **Recursively Set Translation Retargeting Skeleton**.
4.  Finally, pick **Animation** from the drop-down for the **root** and **pelvis** bones.
5.  Save and close **SK_Mannequin**.
6.  Once that’s done, open the **Content Browser** area, click on the **Add** button, and pick **Animation | Blend Space**.
7.  Next, select the **SK_Mannequin** skeleton.
8.  Rename the Blend Space **BS_Movement** and open it.
9.  Set the horizontal `(-180 to 180)` and the vertical `(0 to 200)`, and make sure that you turn on `Snap to Grid` on both. You should end up with the following settings:

![Figure 16.14 – 2D Blend Space – Axis Settings ](img/Figure_16..14_B18531.jpg)

Figure 16.14 – 2D Blend Space – Axis Settings

1.  Drag the `0` and `-180`, `0`, and `180`.
2.  Drag the `200` and `0`.
3.  Drag the `200` and `-90`.
4.  Drag the `200` and `90`.
5.  Drag the `200` and `180` and `180`.

You should end up with a Blend Space that can be previewed by holding *Ctrl* and moving the mouse.

1.  Now, on the `5` to make the interpolation faster.
2.  Save and close the Blend Space.
3.  Now, let’s update the animation Blueprint so that it uses the new Blend Space.
4.  Go to **Content\Characters\Mannequins\Animations** and open **ABP_Manny**.
5.  Next, go to the event graph and create a new float variable called `Direction`.
6.  Add a new pin to the sequence and set the value of `-180` to 1`80`) between the character’s **velocity** and **rotation**:

![Figure 16.15 – Calculating the direction to use on the 2D Blend Space ](img/Figure_16..15_B18531.jpg)

Figure 16.15 – Calculating the direction to use on the 2D Blend Space

1.  In `0.0` to disable the automatic feet adjustment.
2.  Go to the **Walk / Run** state inside the **Locomotion** state machine where the old 1D Blend Space is being used, as shown in the following screenshot:

![Figure 16.16 – The Walk / Run state in the AnimGraph ](img/Figure_16..16_B18531.jpg)

Figure 16.16 – The Walk / Run state in the AnimGraph

1.  Replace that Blend Space with **BS_Movement** and use the **Direction** variable, like so:

![Figure 16.17 – The 1D Blend Space has been replaced with the new 2D Blend Space ](img/Figure_16..17_B18531.jpg)

Figure 16.17 – The 1D Blend Space has been replaced with the new 2D Blend Space

1.  Go to the `Idle` state inside the **Locomotion** state machine and change the animation to use **Idle_Rifle_Ironsights** instead.
2.  Save and close the animation Blueprint. Now, you need to update the character.
3.  Go to the **Content\ThirdPerson\Blueprints** folder and open **BP_ThirdPersonCharacter**.
4.  On the `Yaw` rotation always face the control rotation’s `Yaw`.
5.  Go to the character movement component and set `200`.
6.  Set `false`, which will prevent the character from rotating toward the direction of the movement.
7.  Select the `Mesh` component and on the **Details** panel, pick the **ABP_Manny** animation blueprint and the **SKM_Manny_Simple** skeletal mesh.
8.  Save and close the character Blueprint.

If you play the game now with two clients and move the character, it will walk forward and backward, but it will also strafe, as shown in the following screenshot:

![Figure 16.18 – Expected output on the Server and Client 1 windows ](img/Figure_16..18_B18531.jpg)

Figure 16.18 – Expected output on the Server and Client 1 windows

By completing this exercise, you have improved your understanding of how to create 2D Blend Spaces, how they work, and the advantages they provide compared to just using a regular 1D Blend Space.

In the next section, we will learn how to transform a character’s bone so that we can rotate the torso of the player up and down based on the camera’s pitch.

# Transforming (modifying) bones

There is a very useful node that you can use in **AnimGraph** called the **Transform (Modify) Bone** node, which allows you to translate, rotate, and scale a bone of a skeleton at runtime.

You can add it to `transform modify`, and picking the node from the list. If you click on the **Transform (Modify) Bone** node, you’ll have quite a few options on the **Details** panel.

Here’s an explanation of what the most relevant options do:

*   **Bone to Modify**: This option will tell the node what bone is going to be transformed.

Slightly below that option, you have three sections representing each transform operation (**Translation**, **Rotation**, and **Scale**). In each section, you can do the following:

*   **Translation**, **Rotation**, **Scale**: This option will tell the node how much of that specific transform operation you want to apply. The final result will depend on the mode you have selected (covered in the next section).

There are four ways you can set this value:

*   Setting a constant value such as (`X=0.0,Y=0.0,Z=0.0`).
*   Binding it to a function or a variable, by clicking on the drop-down on the right-hand side and picking one of the functions or variables available from the list.
*   Using a dynamic value that can be set from a function, even if it’s not exposed as a pin.
*   Using a variable so that it can be changed at runtime. To enable this, you need to perform the following steps (this example is for `Expose As Pin`. Once you do that, the text boxes for the constant value will disappear:

![Figure 16.19 – Selecting Expose As Pin ](img/Figure_16..19_B18531.jpg)

Figure 16.19 – Selecting Expose As Pin

1.  The **Transform (Modify) Bone** node will add an input so that you can plug in your variable:

![Figure 16.20 – Variable used as an input on the Transform (Modify) Bone node ](img/Figure_16..20_B18531.jpg)

Figure 16.20 – Variable used as an input on the Transform (Modify) Bone node

1.  **Setting the mode**

This will tell the node what to do with the value. You can pick from one of these three options:

*   **Ignore**: Don’t do anything with the supplied value.
*   **Add to Existing**: Grab the current value of the bone and add the supplied value to it.
*   **Replace Existing**: Replace the current value of the bone with the supplied value.

1.  **Setting the space**

This will define the space where the node should apply the transform. You can pick from one of these four options:

*   **World Space**: The transform will happen in the world space.
*   **Component Space**: The transform will happen in the skeletal mesh component space.
*   **Parent Bone Space**: The transform will happen in the parent bone’s space of the selected bone.
*   **Bone Space**: The transform will happen in the space of the selected bone.

1.  **Alpha**

This option allows you to control the amount of transform that you want to apply. As an example, if you have the `Alpha` value as a float, then you’ll have the following behavior with different values:

*   If `Alpha` is `0.0`, then no transform will be applied.
*   If `Alpha` is `0.5`, then it will only apply half of the transform.
*   If `Alpha` is `1.0`, then it will apply the entire transform.

In the next exercise, we will use the **Transform (Modify) Bone** node to enable the character from *Exercise 16.04 – creating a movement 2D Blend Space*, to look up and down based on the camera’s rotation.

## Exercise 16.05 – Creating a character that looks up and down

In this exercise, we’re going to use the project from *Exercise 16.04 – Creating a movement 2D Blend Space*, and enable the character to look up and down based on the camera’s rotation. To achieve this, we’re going to use the **Transform (Modify) Bone** node to rotate the **spine_03** bone in the component space based on the pitch of the camera.

Follow these steps to complete this exercise:

1.  First, you need to open the project from *Exercise 16.04 – Creating a movement 2D Blend Space*.
2.  Go to **Content\Characters\Mannequins\Animations** and open **ABP_Manny**.
3.  Go to **Event Graph** and create a float variable called **Pitch**.
4.  Add a new pin to the sequence and set the value of **Pitch** with the subtraction (or delta) between the character’s **rotation** and **base aim rotation**, as shown here:

![Figure 16.21 – Calculating the Pitch ](img/Figure_16..21_B18531.jpg)

Figure 16.21 – Calculating the Pitch

This will allow you to get the value of **Pitch** from the rotator, which is the only part of the delta rotation that we are interested in.

Note

The **Break Rotator** node allows you to separate a **Rotator** variable into three float variables that represent **Pitch**, **Yaw**, and **Roll**. This is useful when you want to access the value of each component or if you only want to work with one or two components, and not with the whole rotation.

As an alternative to using the **Break Rotator** node, you can right-click on **Return Value** and pick **Split Struct Pin**. Take into consideration that the **Split Struct Pin** option will only appear if **Return Value** is not connected to anything. Once you do the split, it will create three separate wires for **Roll, Pitch**, and **Yaw,** just like a break but without the extra node.

You should end up with the following:

![Figure 16.22 – Calculating the Pitch to look up using the Split Struct Pin option ](img/Figure_16..22_B18531.jpg)

Figure 16.22 – Calculating the Pitch to look up using the Split Struct Pin option

This logic uses the rotation of the pawn and subtracts it from the camera’s rotation to get the difference in **Pitch**, as shown in the following diagram:

![Figure 16.23 – How to calculate the delta Pitch ](img/Figure_16..23_B18531.jpg)

Figure 16.23 – How to calculate the delta Pitch

Note

You can double-click on a wire to create a reroute node, which allows you to bend the wire so that it doesn’t overlap with other nodes, which makes the code easier to read.

1.  Next, go to `Bone` node with the following settings:

![Figure 16.24 – Settings for the Transform (Modify) Bone node ](img/Figure_16..24_B18531.jpg)

Figure 16.24 – Settings for the Transform (Modify) Bone node

In the preceding screenshot, we’ve set **Bone to Modify** to **spine_03** because that is the bone that we want to rotate. We’ve also set **Rotation Mode** to **Add to Existing** because we want to keep the original rotation from the animation and add an offset to it. We can set the rest of the options to **Ignore** and remove **Expose As Pin** from the dropdown.

1.  Connect the **Transform (Modify) Bone** node to **Control Rig** and the **Output Pose**, as shown in the following screenshot:

![Figure 16.25 – The Transform (Modify) Bone node connected to Output Pose ](img/Figure_16..25_B18531.jpg)

Figure 16.25 – The Transform (Modify) Bone node connected to Output Pose

In the preceding screenshot, you can see the **AnimGraph**, which will allow the character to look up and down by rotating the **spine_03** bone based on the camera’s pitch. To connect the **Control Rig** node to the **Transform** (**Modify**) **Bone** node, we need to convert from local to component space. After the **Transform** (**Modify**) **Bone** node is executed we need to convert back to local space to be able to connect to the **Output Pose** node.

Note

We connect the **Pitch** variable to **Roll** because that bone in the skeleton is internally rotated that way. You can use **Split Struct Pin** on input parameters as well, so you don’t have to add a **Make Rotator** node.

If you test the project with two clients and move the mouse up and down on one of the characters, you’ll notice that it will pitch up and down, as shown in the following screenshot:

![Figure 16.26 – Characters looking up and down, based on the camera rotation ](img/Figure_16..26_B18531.jpg)

Figure 16.26 – Characters looking up and down, based on the camera rotation

By completing this final exercise, you should understand how to modify bones at runtime using the **Transform (Modify) Bone** node in an animation blueprint. This node can be used in various scenarios, so it may prove useful for you.

In the next activity, you’re going to put everything you’ve learned to the test by creating the character we’re going to use for our multiplayer FPS project.

# Activity 16.01 – Creating the character for the multiplayer FPS project

In this activity, you’ll create the character for the multiplayer FPS project that we’re going to build in the next few chapters. The character will have a few different mechanics, but for this activity, you just need to create a character that walks, jumps, looks up/down, and has two replicated stats: health and armor.

Follow these steps to complete this activity:

1.  Create a `MultiplayerFPS` without the starter content.
2.  Import the skeletal mesh and the animations from the `Activity16.01\Assets` folder and place them in the **Content\Player\Mesh** and **Content\Player\Animations** folders, respectively.
3.  Import the following sounds from the `Activity16.01\Assets` folder into `Content\Player\Sounds`:
    *   `Jump.wav`: Play this sound on the `Jump_From_Stand_Ironsights` animation with a `Play Sound` anim notify.
    *   `Footstep.wav`: Play this sound every time a foot is on the floor in every walk animation by using the `Play Sound` anim notify.
    *   `Spawn.wav`: Use this on the `SpawnSound` variable in the character.
4.  Set up the skeletal mesh by retargeting its bones and creating a socket called `Camera` that is a child of the head bone and has a `Relative Location` of (`X=7.88, Y=4.73, Z=-10.00`).
5.  Create a 2D Blend Space in `5`.
6.  Create the input actions using the knowledge you acquired in [*Chapter 4*](B18531_04.xhtml#_idTextAnchor099), *Getting Started with Player Input*:
    *   `W`, `S`, `A`, `D`
    *   `Mouse X`, `Mouse Y`
    *   `Spacebar`
7.  Add the new input actions to a new input mapping context called **IMC_Player**.
8.  Create a C++ class called `Armor Absorption`, which is the percentage of how much damage the armor absorbs.
9.  Has a constructor that initializes the camera, disables ticking, and sets `800` and `600`.
10.  On `Armor Absorption` variable and changes the damage value based on the following formula:

`Damage = (Damage * (1 - ArmorAbsorption)) - FMath::Min(RemainingArmor, 0);`

1.  Create an animation Blueprint in `Content\Player\Animations` called `ABP_Player` that has a `State Machine` with the following states:
    *   `Idle/Run`: Uses `BS_Movement` with the `Speed` and `Direction` variables.
    *   `Jump`: Plays the jump animation and transitions from the `Idle/Run` states when the `Is Jumping` variable is `true`.
    *   It also uses `Transform (Modify) Bone` to make the character look up and down based on the camera’s pitch.
2.  Create a `UMG` widget in `Content\UI` called `WBP_HUD` that displays the `Health` and `Armor` properties of the character in the `Health: 100` and `Armor: 100` formats using the knowledge you acquired in [*Chapter 15*](B18531_15.xhtml#_idTextAnchor322), *Exploring Collectibles, Power-Ups, and Pickups*.
3.  Create a Blueprint in `Content\Player` called `BP_Player` that derives from `FPSCharacter`:
    *   Set up the mesh component so that it has the following values:
        *   `Content\Player\Mesh\SK_Mannequin`
        *   `Content\Player\Animations\ABP_Player`
        *   `(X=0.0, Y=0.0, Z=-88.0)`
        *   `(X=0.0, Y=0.0, Z=-90.0)`
        *   `Content\Player\Inputs\IA_Move`
        *   `Content\Player\Inputs\IA_Look`
        *   `Content\Player\Inputs\IA_Jump`
        *   On the `Begin Play` event, it needs to create a widget instance of `WBP_HUD` and add it to the viewport.
4.  Create a Blueprint in `BP_GameMode` that derives from `BP_Player` as the **DefaultPawn** class.
5.  Create a test map in **Content\Maps** called **DM-Test** and set it as the default map in **Project Settings**.

**Expected output**:

The result should be a project where each client will have a first-person character that can move, jump, and look around. These actions will also be replicated so that each client will be able to see what the other client’s character is doing.

Each client will also have a HUD that displays the health and the armor values:

![Figure 16.27 – Expected output ](img/Figure_16..27_B18531.jpg)

Figure 16.27 – Expected output

Note

The solution for this activity can be found on GitHub here: [https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions](https://github.com/PacktPublishing/Elevating-Game-Experiences-with-Unreal-Engine-5-Second-Edition/tree/main/Activity%20solutions).

By completing this activity, you should have a good idea of how the server-client architecture, variable replication, roles, 2D Blend Spaces, and the **Transform (Modify) Bone** node work.

# Summary

In this chapter, we learned about some critical multiplayer concepts, such as how the server-client architecture works, the responsibilities of the server and the client, how the listen server is quicker to set up than a dedicated server but not as lightweight, ownership and connections, roles, and variable replication.

We also learned about some useful techniques for animation, such as how to use 2D Blend Spaces, which allow you to have a two-axis grid to blend between animations, and the `Transform (Modify) Bone` node, which can modify the bones of a skeletal mesh at runtime. To finish off this chapter, we created a first-person multiplayer project where you have characters that can walk, look, and jump around. This will be the foundation of the multiplayer first-person shooter project that we will be working on for the next few chapters.

In the next chapter, we’ll learn how to use RPCs, which allows clients and servers to execute functions on each other. We’ll also cover how to use enumerations in the editor and how to use array index wrapping to iterate an array in both directions and loop around when you go beyond its limits.