# Chapter 3. Exploration and Combat

We have a design for our game and an Unreal project set up for our game. It's now time to dive into the actual game code.

In this chapter, we'll be making a game character that moves around the world, defining our game data, and prototyping a basic combat system for the game. We will cover the following topics in this chapter:

*   Creating the player pawn
*   Defining characters, classes, and enemies
*   Keeping track of active party members
*   Creating a basic turn-based combat engine
*   Triggering a game over screen

This particular chapter is the most C++ heavy portion of the book, and provides the basic framework that the rest of the book is going to use. Because this chapter provides much of the backend of our game, the code in this chapter must work to completion before moving on to the rest of the content in the book. If you bought this book because you are a programmer looking for more background in creating a framework for an RPG, this chapter is for you! If you bought this book because you are a designer, and care more about building upon the framework rather than programming it from scratch, you will probably be more into the upcoming chapters because those chapters contain less C++ and more UMG and Blueprints. No matter who you are, it is a good idea to download the source code provided through the directions located in the preface of the book in case you get stuck or would like to skip chapters based on your interests.

# Creating the player pawn

The very first thing we are going to do is create a new Pawn class. In Unreal, the *Pawn* is the representation of a character. It handles the movement, physics, and rendering of a character.

Here's how our character pawn is going to work. The player is divided into two parts: there's the Pawn that, as mentioned earlier, is responsible for handling the movement, physics, and rendering. Then there's the Player Controller, responsible for translating the player's input into making the Pawn perform what the player wants.

## The Pawn

Now, let's create the actual pawn.

Create a new C++ class and select `Character` as the parent class for it. We will be deriving this class from the `Character` class because `Character` has a lot of built-in move functions that we can use for our field player. Name the class `RPGCharacter`. Open `RPGCharacter.h` and change the class definition using the following code:

[PRE0]

Next, open `RPGCharacter.cpp` and replace it with the following code:

[PRE1]

If you have ever created and worked with the C++ **ThirdPerson** game template, you will notice that we are not reinventing the wheel here. The `RPGCharacter` class should look familiar because it is a modified version of the ThirdPerson `Character` class code given to us when we create a C++ ThirdPerson template, provided for us to use by Epic Games.

Since we are not creating a fast-paced action game and are simply using the Pawn as an RPG character to maneuver out in the field, we eliminated mechanics that are often associated with action games, such as jumping. But we kept in the code that is important to us, which is the ability to move forward, backward, left, and right; rotational behaviors of the pawn; a camera that will follow the character around in an isometric view; a collision capsule for the character to be able to collide with collidable objects; configuration for character movement; and a camera boom, which will allow the camera to move closer to the character in case it runs into collisions such as a wall or other meshes, important for not blocking a player's view. If you want to edit the character mechanics, feel free to do so by following the comments in the code to change the values of some specific mechanics such as `TargetArmLength` to change the distance of the camera from the player, or adding jumping, which can be seen in the ThirdPerson character template that came with the Engine.

Because we derived the `RPGCharacter` class from the `Character` class, its default camera is not rotated for an isometric view; instead, the camera rotations and locations are zeroed out by default to the pawn's location. So what we did was add a `CameraBoom` relative location in `RPGCharacter.cpp` (`CameraBoom->RelativeLocation = FVector(0.f, 0.f, 500.f);`); this offsets the camera 500 units up on the *z* axis. Along with rotating the camera that follows the player -45 units on the pitch (`FollowCamera->RelativeRotation = FRotator(-45.f, 0.f, 0.f);`), we get a traditional isometric view. If you would like to edit these values to customize your camera even more, it is suggested; for instance, if you still think your camera is too close to the player, you can simply change the relative location of `CameraBoom` on the *z* axis to a value higher than 500 units, and/or adjust `TargetArmLength` to something larger than 300.

Lastly, if you take a look at the `MoveForward` and `MoveRight` movement functions, you will notice that no movement is being added to the pawn unless the value that is passed to `MoveForward` or `MoveRight` is not equal to 0\. Later on in this chapter, we will bind keys *W*, *A*, *S*, and *D* to these functions and set each one of these inputs to pass a scalar of 1 or -1 to the corresponding movement function as values. This 1 or -1 value is then used as a multiplier to the direction of the pawn, which will then allow the player to move in a specific direction based on its walk speed. For instance, if we set *W* as a keybind to `MoveForward` with a scalar of 1, and *S* as a keybind to `MoveFoward` with a scalar of -1, when the player presses *W*, the value in the `MoveFoward` function will be equal to 1 and cause the pawn to move in the positive forward direction as a result. Alternatively, if the player presses the *S* key, -1 is then passed into the value used by the `MoveForward` function, which will cause the pawn to move in the negative forward direction (in other words, backwards). Similar logic can be said about the `MoveRight` function, which is why we don't have a `MoveLeft` function—simply because pressing the *A* key would cause the player to move in the negative right direction, which is in fact left.

## The GameMode class

Now, in order to use this pawn as a player character, we need to set up a new game mode class. This game mode will then specify the default pawn and player controller classes to use. We'll also be able to make a Blueprint of the game mode and override these defaults.

Create a new class and choose `GameMode` as the parent class. Name this new class `RPGGameMode` (if `RPGGameMode` already exists in your project, simply navigate to your C++ source directory and proceed to open up `RPGGameMode.h`, as listed in the next step).

Open `RPGGameMode.h` and change the class definition using the following code:

[PRE2]

Just as we've done before, we're just defining a constructor for the CPP file to implement.

We're going to implement that constructor now in `RPGGameMode.cpp`:

[PRE3]

Here, we include the `RPGCharacter.h` file so that we can reference these classes. Then, in the constructor, we set the class as the default class to use for the Pawn.

Now, if you compile this code, you should be able to assign your new game mode class as the default game mode. To do this, go to **Edit** | **Project Settings**, find the **Default Modes** box, expand the **Default GameMode** drop-down menu, and select **RPGGameMode**.

However, we don't necessarily want to use this class directly. Instead, if we make a Blueprint, we can expose the properties of the game mode that can be modified in the Blueprint.

So, let's make a new Blueprint Class in **Content** | **Blueprints**, pick `RPGGameMode` as its parent class, and call it `DefaultRPGGameMode`:

![The GameMode class](img/B04548_03_01.jpg)

If you open the Blueprint and navigate to the **Defaults** tab, you can modify the settings for the game mode for **Default Pawn Class**, **HUD Class**, **Player Controller Class**, and more settings:

![The GameMode class](img/B04548_03_02.jpg)

However, we still have one extra step before we can test our new Pawn. If you run the game, you will not see the Pawn at all. In fact, it will appear as if nothing is happening. We need to give our Pawn a skinned mesh and also make the camera follow the pawn.

## Adding the skinned mesh

For now, we're just going to import the prototype character that comes with the ThirdPerson sample. To do this, make a new project based on the ThirdPerson sample. Locate the `ThirdPersonCharacter` Blueprint class in **Content** | **ThirdPersonCPP** | **Blueprints** and migrate it to the `Content` folder of your RPG project by right-clicking on the `ThirdPersonCharacter` Blueprint class and navigating to **Asset Actions** | **Migrate…**. This action should copy `ThirdPersonCharacter` with all its assets into your RPG project:

![Adding the skinned mesh](img/B04548_03_03.jpg)

Now, let's create a new Blueprint for our Pawn. Create a new Blueprint class and select **RPGCharacter** as the parent class. Name it **FieldPlayer**.

Expand **Mesh** located in the **Details** tab when selecting the **Mesh** component from the **Components** tab and choose **SK_Mannequin** as the skeletal mesh for the pawn. Next, expand **Animation** and choose **ThirdPerson_AnimBP** as the animation Blueprint to use. You will most likely need to move your character's mesh down the *z* axis so that the bottom of the character's feet meet the bottom of the collision capsule. Also be sure that the character mesh is facing the same direction that the blue arrow in the component is facing. You may need to rotate the character on the *z* axis as well to ensure that the character is facing the right direction:

![Adding the skinned mesh](img/B04548_03_04.jpg)

Finally, open your game mode Blueprint and change the pawn to your new **FieldPlayer** Blueprint.

Now, our character will be visible, but we may not be able to move it yet because we have not bound keys to any of our movement variables. To do so, go into **Project Settings** and locate **Input**. Expand **Bindings** and then expand **Axis Mappings**. Add an axis mapping by pressing the **+** button. Call the first axis mapping **MoveRight**, which should match the `MoveRight` movement variable you created earlier in this chapter. Add two key bindings for **MoveRight** by pressing the **+** button. Let one of those keys be *A* with a scale of -1 and other be *D* with a scale of 1\. Add another axis mapping for **MoveForward**; only this time, have a *W* key binding with a scale of 1 and an *S* key binding with a scale of -1:

![Adding the skinned mesh](img/B04548_03_05.jpg)

Once you play test, you should see your character moving and animating using the *W*, *A*, *S*, and *D* keys you bound to the player.

When you run the game, the camera should track the player in an overhead view. Now that we have a character that can explore the game world, let's take a look at defining characters and party members.

# Defining characters and enemies

In the last chapter, we covered how to use Data Tables to import custom data. Before that, we decided on what stats would play into combat and how. Now we're going to combine those to define our game's characters, classes, and enemy encounters.

## Classes

Remember that in [Chapter 1](ch01.html "Chapter 1. Getting Started with RPG Design in Unreal"), *Getting Started with RPG Design in Unreal*, we established that our characters have the following stats:

*   Health
*   Max health
*   Magic
*   Max magic
*   Attack power
*   Defense
*   Luck

Of these, we can discard health and magic because they vary during the game, while the other values are predefined based on the character class. The remaining stats are what we will define in the Data Table. As mentioned in [Chapter 1](ch01.html "Chapter 1. Getting Started with RPG Design in Unreal"), *Getting Started with RPG Design in Unreal*, we also need to store what the value should be at level 50 (the maximum level). Characters will also have some abilities they start out with, and some they learn as they level up.

We'll define these in the character class spreadsheet, along with the name of the class. So our character class schema will look something like the following:

*   Class name (string)
*   Starting max HP (integer)
*   Max HP at level 50 (integer)
*   Starting max MP (integer)
*   Max MP at level 50 (integer)
*   Starting attack (integer)
*   Attack at level 50 (integer)
*   Starting defense (integer)
*   Defense at level 50 (integer)
*   Starting luck (integer)
*   Luck at level 50 (integer)
*   Starting abilities (string array)
*   Learned abilities (string array)
*   Learned ability levels (integer array)

The ability string arrays will contain the ID of the ability (the value of the reserved `name` field in UE4). Additionally, there are two separate cells for learned abilities—one that contains the ability IDs, another that contains the levels at which those abilities are learned.

In a production game, one thing you might consider is writing a custom tool to help manage this data and reduce human error. However, writing such a tool is outside the scope of this book.

Now, instead of creating a spreadsheet for this, we're actually going to first create the class and then the Data Table inside Unreal. The reason for this is that at the time of writing, the proper syntax to specify arrays in a cell of a Data Table is not well documented. However, arrays can still be edited from inside the Unreal editor, so we'll simply create the table there and use Unreal's array editor.

Firstly, as usual, create a new class. The class will be used as an object that you can call from, so choose `Object` as the parent class. Name this class `FCharacterClassInfo` and, for organization purposes, path your new class to your `Source/RPG/Data` folder.

Open `FCharacterClassInfo.h` and replace the class definition with the following code:

[PRE4]

Most of this code should be familiar to you already; however, you may not recognize the last three fields. These are all of the `TArray` type, which is a dynamic array type provided by Unreal. Essentially, a `TArray` can have elements dynamically added to it and removed from it, unlike a standard C++ array.

Upon compiling this code, create a new folder called `Data` within your `Content` folder so that you can stay organized by keeping Data Tables that you create within the `Data` folder. Navigate to **Content** | **Data** in the Content Browser and create a new Data Table by right-clicking on **Content Browser** and choosing **Miscellaneous** | **Data Table**. Then, select **Character Class Info** from the drop-down list. Name your Data Table **CharacterClasses** and then double-click to open it.

To add a new entry, hit the **+** button. Then, give a name to the new entry by entering something in the **Row Name** field and pressing *Enter*.

After an entry has been added, you can select the entry in the **Data Table** pane and edit its properties in the **Row Editor** pane.

Let's add a Soldier class to the list. We will it give the name `S1` (which we'll use to refer to the character class from other Data Tables) and it will have the following properties:

*   **Class Name**: Soldier
*   **Start MHP**: 100
*   **Start MMP**: 100
*   **Start ATK**: 5
*   **Start DEF**: 0
*   **Start Luck**: 0
*   **End MHP**: 800
*   **End MMP**: 500
*   **End ATK**: 20
*   **End DEF**: 20
*   **End Luck**: 10
*   **Starting Abilities**: (leave empty for now)
*   **Learned Abilities**: (leave empty for now)
*   **Learned Ability Levels**: (leave empty for now)

When you are finished, your Data Table should look like this:

![Classes](img/B04548_03_06.jpg)

If you have more character classes that you would like to define, continue to add them to your Data Table.

## Characters

With classes defined, let's take a look at characters. Since most of the important combat-related data is already defined as part of a character's class, the character itself is going to be quite a bit simpler. In fact, for now, our characters will be defined by just two things: the name of the character and the character's class.

Firstly, create a new C++ class called `FCharacterInfo` whose parent is `Object`, and path it to the `Source/RPG/Data` folder. Now, replace the class definition in `FCharacterInfo.h` with this:

[PRE5]

As we did earlier, we're just defining the two fields for the character (character name and class ID).

After compiling, create a new Data Table in your `Data` folder that you created earlier from within the Content Browser and select **CharacterInfo** as the class; call it `Characters`. Add a new entry with the name `S1`. You can name this character whatever you like (we named our character soldier **Kumo**), but for class ID, enter `S1` (as this is the name of the Soldier class we defined earlier).

## Enemies

As for enemies, rather than defining a separate character and class information, we'll create a simplified combined table for these two pieces of information. An enemy generally does not have to deal with experience and leveling up, so we can omit any data related to this. Additionally, enemies do not consume MP as players do, so we can omit this data as well.

Therefore, our enemy data will have the following properties:

*   Enemy name (string array)
*   MHP (integer)
*   ATK (integer)
*   DEF (integer)
*   Luck (integer)
*   Abilities (string array)

Much like the previous Data Class creations, we create a new C++ class that derives from `Object`, but this time we will call it `FEnemyInfo` and place it with the rest of our data in the `Source/RPG/Data` directory.

At this point, you should have an understanding of how to construct the class for this data, but let's take a look at the struct header anyway. In `FEnemyInfo.h`, replace your class definition with the following:

[PRE6]

After compiling, create a new Data Table, select `EnemyInfo` as the class, and call the Data Table `Enemies`. Add a new entry with the name `S1` and the following properties:

*   **Enemy Name**: Goblin
*   **MHP**: 20
*   **ATK**: 5
*   **DEF**: 0
*   **Luck**: 0
*   **Abilities**: (leave empty for now)

At this point, we've got the data for a character, the character's class, and a single enemy for the character to fight. Next, let's start keeping track of which characters are in the active party and what their current stats are.

# Party members

Before we can keep track of party members, we'll need a way to track a character's current state, such as how much HP the character has or what it has equipped.

To do this, we'll create a new class named `GameCharacter`. As usual, create a new class and pick `Object` as the parent class.

The header for this class looks like the following code snippet:

[PRE7]

For now, we're keeping track of the character's name, character's source class information, and character's current stats. Later, we will use the `UCLASS` and `UPROPERTY` macros to expose information to the Blueprint. We'll add other information later as we work on the combat system.

As for the `.cpp` file, it will look like this:

[PRE8]

The `CreateGameCharacter` factory for our `UGameCharacter` class takes a pointer to an `FCharacterInfo` struct, which is returned from a Data Table, and also an `Outer` object, which is passed to the `NewObject` function. It then attempts to find the character class Data Table from a path, and if the result is not null, it locates the proper row in the Data Table, stores the result, and also initializes the stats and the `CharacterName` field. In the preceding code, you can see the path where the character class Data Table is located. You can get this path by right-clicking on your Data Table from the Content Browser, selecting **Copy Reference**, and then pasting the result into your code.

While this is currently a very basic bare-bones representation of a character, it will work for now. Next, we're going to store a list of these characters as the current party.

## The GameInstance class

We have already created a `GameMode` class, and this might seem like the perfect place to keep track of information such as party members and inventory, right?

However, `GameMode` does not persist between level loads! This means that unless you save some information to disk, you lose all of that data whenever you load a new area.

The `GameInstance` class was introduced to deal with just this sort of problem. A `GameInstance` class persists through the whole game, regardless of level loads, unlike `GameMode`. We're going to create a new `GameInstance` class to keep track of our persistent data, such as party members and inventory.

Create a new class, and this time, select `GameInstance` as the parent class (you'll have to search for it). Name it `RPGGameInstance`.

In the header file, we're going to add a `TArray` of the `UGameCharacter` pointers, a flag to know whether the game has been initialized, and an `Init` function. Your `RPGGameInstance.h` file should look like this:

[PRE9]

In the `Init` function for the game instance, we'll add a single default party member and then set the `isInitialized` flag to `true`. Your `RPGGameInstance.cpp` should look like this:

[PRE10]

You may run into a linker error at this point if you try to compile. It is recommended that before you move on, save and close everything. Then restart your project. After you do that, compile the project.

To set this class as your `GameInstance` class, in Unreal, open **Edit** | **Project Settings**, go to **Maps & Modes**, scroll down to the **Game Instance** box, and pick **RPGGameInstance** from the drop-down list. Finally, from the game mode, we override `BeginPlay` to call this `Init` function.

Open `RPGGameMode.h` and add `virtual void BeginPlay() override;` at the end of your class so that your header will now look like this:

[PRE11]

And in `RPGGameMode.cpp`, cast `RPGGameInstance` at `BeginPlay` so that `RPGGameMode.cpp` now looks like this:

[PRE12]

Once you compile the code, you now have a list of active party members. It's time to start prototyping the combat engine.

# Turn-based combat

So, as mentioned in [Chapter 1](ch01.html "Chapter 1. Getting Started with RPG Design in Unreal"), *Getting Started with RPG Design in Unreal*, combat is turn-based. All characters first choose an action to perform; then, the actions are executed in order.

Combat will be split into two main phases: **Decision**, in which all characters decide on their course of action; and **Action**, in which all characters execute their chosen course of action.

Let's create a class with an empty parent to handle combat for us, which we'll call `CombatEngine`, and path it to a new directory located in `Source/RPG/Combat`, where we can organize all of our combat-related classes. Formulate the header file to look like this:

[PRE13]

There's a lot going on here, so I'll explain.

Firstly, our combat engine is designed to be allocated when an encounter starts and deleted when combat is over.

An instance of `CombatEngine` keeps three `TArray`: one for combat order (a list of all participants in combat, in the order they will take turns in), another for a list of players, and the third one for a list of enemies. It also keeps track of `CombatPhase`. There are two main phases of combat: `Decision` and `Action`. Each round starts in `Decision`; in this phase, all characters are allowed to choose their course of action. Then, combat transitions to the `Action` phase; in this phase, all characters perform their previously chosen course of action.

The `GameOver` and `Victory` phases will be transitioned to when all enemies are dead or all players are dead, respectively (which is why the player and enemy lists are kept separate).

The `CombatEngine` class defines a `Tick` function. This will be called by the game mode of every frame as long as combat is not over, and it returns `true` when combat has finished (or `false` otherwise). It takes the duration of the last frame in seconds as a parameter.

There's also the `currentTickTarget` and `tickTargetIndex`. During the `Decision` and `Action` phases, we'll keep a pointer to a single character. For instance, during the `Decision` phase, this pointer starts with the first character in the combat order. At every frame, the character will be asked to make a decision—which will be a function that returns `true` if the character has made a decision, or `false` otherwise. If the function returns `true`, the pointer will advance to the next character and so on until all characters have decided at which point the combat transitions to the `Action` phase.

The CPP for this file is fairly big, so let's take it in small chunks. Firstly, the constructor and destructor are as follows:

[PRE14]

The constructor first assigns the player party and enemy party fields and then adds all players followed by all enemies to the combat order list. Finally, it sets the `tick` target index to 0 (the first character in the combat order) and the combat phase to `Decision`.

Next, the `Tick` function is as follows:

[PRE15]

Firstly, we switch on the current combat phase. In the case of `Decision`, it currently just selects the next character or, if there is no next character, switches to the `Action` phase. It is the same for `Action`—except that if there is no next character, it loops back to the `Decision` phase.

Later, this will be modified to call functions on the character in order to make and execute decisions (and additionally, the "select next character" code will only be called once the character has finished deciding or executing).

In the case of `GameOver` or `Victory`, `Tick` returning `true` means combat is over. Otherwise, it first checks whether all players are dead (in this case, it is game over) or whether all enemies are dead (in this case, players win combat). In both cases, the function will return `true` as combat is finished.

The very end of the function returns `false`, which means combat has not yet finished.

Next, we have the `SetPhase` function:

[PRE16]

This function sets the combat phase, and in the case of `Action` or `Decision`, it sets the `tick` target to the first character in the combat order. Both `Victory` and `GameOver` have stubs to handle the respective states.

Finally, we have the `SelectNextCharacter` function:

[PRE17]

This function starts at the current `tickTargetIndex` and, from there, finds the first non-dead character in the combat order. If one is found, it sets the `tick` target index to the next index and the `tick` target to the found character. Otherwise, it sets the `tick` target index to -1 and the `tick` target to a null pointer (which is interpreted to mean no remaining characters in combat order).

There's a very important thing missing at this point: characters cannot yet make or execute decisions.

Let's add this to the `GameCharacter` class. For now, they will just be stubs.

Firstly, we'll add the `testDelayTimer` field to `GameCharacter.h`. This will just be for testing purposes:

[PRE18]

Next, we add several public functions to the class:

[PRE19]

We split `Decision` and `Action` into two functions each—the first function tells the character to begin making a decision or executing an action, the second function essentially queries the character until the decision is made or action is finished.

The implementation for these two functions in `GameCharacter.cpp` will, for now, just log a message and a delay for 1 second:

[PRE20]

We're also going to add a pointer to the combat instance. Since the combat engine references characters, having characters reference the combat engine would produce a circular dependency. To solve this, we're going to add a forward declaration at the top of `GameCharacter.h` directly after our includes:

[PRE21]

Then, the `include` statement for the combat engine will actually be placed in `GameCharacter.cpp` rather than in the header file:

[PRE22]

Next, we'll make the combat engine call the `Decision` and `Action` functions. Firstly, we'll add a protected variable to `CombatEngine.h`:

[PRE23]

This will be used to switch between, for example, `BeginMakeDecision` and `MakeDecision`.

Next, we'll modify the `Decision` and `Action` phases in the `Tick` function. Firstly, we'll modify the `Decision` switch case:

[PRE24]

If `waitingForCharacter` is `false`, it calls `BeginMakeDecision` and sets `waitingForCharacter` to `true`.

Keep note of the brackets enclosing the whole case statement—if you do not add these brackets, you will get compile errors about the `decisionMade` initialization being skipped by the case statement.

Next, it calls `MakeDecision` and passes the frame time. If this function returns `true`, it selects the next character, or failing that, switches to the `Action` phase.

The `Action` phase looks identical to the following:

[PRE25]

Next, we'll modify `SelectNextCharacter` so that it sets `waitingForCharacter` to `false`:

[PRE26]

Finally, a few remaining details: our combat engine should set the `CombatInstance` pointer of all characters to point to itself, which we'll do in the constructor; then we'll clear the pointer in the destructor and also release enemy pointers. So first, create a pointer to `combatInstance` in `GameCharacter.h` right after your `UProperty` declarations and before your protected variables:

[PRE27]

Then, in `CombatEngine.cpp`, replace your constructor and deconstructor with this:

[PRE28]

The combat engine itself is almost fully functional at this point. We still need to hook it up to the rest of the game, but with a way to trigger combat and update it from the game mode.

So, firstly in our `RPGGameMode` class, we will add a pointer to the current combat instance and also override the `Tick` function; additionally, keep track of a list of enemy characters (decorated with `UPROPERTY` so that enemies can be garbage-collected):

[PRE29]

Next, in the `.cpp` file, we implement the `Tick` function:

[PRE30]

For now, this simply checks whether there is currently an instance of combat; if so, it calls that instance's `Tick` function. If it returns `true`, the game mode checks for either `Victory` or `GameOver` (for now, it just logs a message to the console). Then, it deletes the combat instance, sets the pointer to null, and clears the enemy party list (which will make the enemies eligible for garbage collection since the list was decorated with the `UPROPERTY` macro). It also enables the tick of the player actor (we're going to disable the tick when combat begins so that the player actor freezes in place for the duration of combat).

However, we aren't ready to start combat encounters just yet! There are no enemies for players to fight.

We have a table of enemies defined, but our `GameCharacter` class does not support being initialized from `EnemyInfo`.

To support this, we will add a new factory to the `GameCharacter` class (be sure you add the `include` statement for the `EnemyInfo` class as well):

[PRE31]

Also, the implementation of this constructor overload in `GameCharacter.cpp` would be as follows:

[PRE32]

It's very simple by comparison; simply assign the name and null for `ClassInfo` (as enemies do not have classes associated with them) and other stats (both MMP and MP are set to zero, as enemy abilities will not consume MP).

To test our combat system, we will create a function in `RPGGameMode.h` that can be called from the Unreal console:

[PRE33]

The `UFUNCTION(exec)` macro is what allows this function to be called as a console command.

The implementation of this function is placed in `RPGGameMode.cpp`, as follows:

[PRE34]

It locates the enemy Data Table, picks the enemy with ID `S1`, constructs a new `GameCharacter`, constructs a list of enemies, adds the new enemy character, and then creates a new instance of `CombatEngine`, passing the player party and the enemy list. It also disables the tick of the player actor so that the player stops updating when combat begins.

Finally, you should be able to test the combat engine. Start the game and press the tilde (*~*) key to bring up the console command textbox. Enter `TestCombat` and press *Enter*.

Take a look at the output window and you should see something like the following:

[PRE35]

This shows that the combat engine is working as intended—firstly, all characters make a decision, execute their decisions, then they make a decision again, and so on. Since nobody is actually doing anything (much less dealing any damage), combat just goes on forever at the moment.

There are two issues with this: firstly, the aforementioned problem that nobody actually does anything yet. Additionally, player characters need to have a different way of making decisions than enemies (player characters will need a UI to pick actions, whereas enemies should pick actions automatically).

We'll solve the first issue before tackling decision making.

## Performing actions

In order to allow characters to perform actions, we will boil all combat actions down to a single common interface. A good place to start is for this interface to map to what we already have—that is, the character's `BeginExecuteAction` and `ExecuteAction` functions.

Let's create a new `ICombatAction` interface for this, which can start off as a class that is not parented to anything and in a new path called `Source/RPG/Combat/Actions`; the `ICombatAction.h` file should look like this:

[PRE36]

`BeginExecuteAction` takes a pointer to the character that this action is executing for. `ExecuteAction`, as before, takes the duration of the previous frame in seconds.

In `ICombatAction.cpp`, remove the default constructor and deconstructor so that the file looks like this:

[PRE37]

Next, we can create a new empty C++ class to implement this interface. Just as a test, we'll replicate the functionality that the characters are already doing (that is, absolutely nothing) in a new class called `TestCombatAction` to be pathed to the `Source/RPG/Combat/Actions` folder.

Firstly, the header will be as follows:

[PRE38]

The `.cpp` file will be as follows:

[PRE39]

Next, we'll change the character so that it can store and execute actions.

Firstly, let's replace the test delay timer field with a combat action pointer. Later, we'll make it public for when we create a decision making system in `GameCharacter.h`:

[PRE40]

Also remember to include `ICombatAction` at the top of `GameCharacter.h`, followed by a class declaration for `ICombatAction`:

[PRE41]

Next, we need to change our decision functions to assign a combat action, and the action functions to execute this action in `GameCharacter.cpp`:

[PRE42]

Also remember to use `include TestCombatAction` at the top of `GameCharacter.cpp`:

[PRE43]

`BeginMakeDecision` now assigns a new instance of `TestCombatAction`. `MakeDecision` just returns `true`. `BeginExecuteAction` calls the function of the same name on the stored combat action, passing the character as the pointer. Finally, `ExecuteAction` calls the function of the same name, and if the result is `true`, it deletes the pointer and returns `true`; otherwise it returns `false`.

By running this and testing combat, you should get nearly identical output, but now it says `does nothing` instead of `executing action`.

Now that we have a way for characters to store and execute actions, we can work on a decision making system for characters.

## Making decisions

As we did with actions, we're going to create an interface for decision making that follows a similar pattern to the `BeginMakeDecision` and `MakeDecision` functions. Similar to the `ICombatAction` class, we will create an empty `IDecisionMaker` class and we will path it to a new directory, `Source/RPG/Combat/DecisionMakers`. The following will be `IDecisionMaker.h`:

[PRE44]

Also, remove the constructor and deconstructor to `IDecisionMaker.cpp`, so that it looks like this:

[PRE45]

Now, we can create the `TestDecisionMaker` C++ class and path it to `Source/RPG/Combat/DecisionMakers` as well. Then, program `TestDecisionMaker.h` as follows:

[PRE46]

Then, program `TestDecisionMaker.cpp` as follows:

[PRE47]

Next, we'll add a pointer to `IDecisionMaker` to the game character class and modify the `BeginMakeDecision` and `MakeDecision` functions to use the decision maker in `GameCharacter.h`:

[PRE48]

Also remember to include `ICombatAction` at the top of `GameCharacter.h` followed by a class declaration for `ICombatAction`:

[PRE49]

Next, replace the `BeginDestroy`, `BeginMakeDecision`, and `MakeDecision` functions in `GameCharacter.cpp` with this:

[PRE50]

Note that we delete the decision maker in the destructor. The decision maker will be assigned when the character is created, and should therefore be deleted when the character is released.

We will then include `TestDecisionMaker` implementations to allow each party to make combat decisions, so include `TestDecisionMaker` at the top of the class:

[PRE51]

The final step here is to assign a decision maker in the constructors for the character. To both constructor overloads, add the following line of code: `character->decisionMaker = new TestDecisionMaker();`. When you are finished, the player and enemy character constructors should look like this:

[PRE52]

Run the game and test combat again, and you should get very similar output to what was already there. However, the big difference is that it's now possible to assign different implementations of a decision maker to different characters, and those decision makers have an easy way to assign combat actions to be executed. For instance, it will now be easy to make our test combat action deal with the damage of a target. However, before we do this, let's make a small change to the `GameCharacter` class.

## Target selection

We're going to add a field to `GameCharacter` that identifies a character as either a player or an enemy. Additionally, we'll add a `SelectTarget` function that selects the first live character from either the current combat instance's `enemyParty` or `playerParty`, depending on whether this character is a player or an enemy.

Firstly, in `GameCharacter.h`, we'll add a public `isPlayer` field:

[PRE53]

Then, we'll add a `SelectTarget` function, as follows:

[PRE54]

In `GameCharacter.cpp`, we'll assign the `isPlayer` field in the constructors (this is easy enough, as we have separate constructors for players and enemies):

[PRE55]

Finally, the `SelectTarget` function is as follows:

[PRE56]

This first figures out which list (enemies or players) to use as potential targets and then goes through that list to find the first non-dead target. If there is no target, this function returns a null pointer.

## Dealing damage

Now that there's an easy way to select targets, let's make our `TestCombatAction` class finally deal some damage!

We'll add a couple of fields to maintain references to the character and the target, and also a constructor that takes the target as a parameter:

[PRE57]

Also, the implementation is by creating and updating the `BeginExecuteAction` function in `TestCombatAction.cpp`, as follows:

[PRE58]

And then have the constructor of the class set the target:

[PRE59]

Firstly, the constructor assigns the target pointer. Then, the `BeginExecuteAction` function assigns the character reference and checks to see whether the target is alive. If the target is dead, it picks a new target via the `SelectTarget` function we just created. If the target pointer is now null, there is no target and this function just returns null. Otherwise, it logs a message of the form *[character] attacks [target]*, subtracts some HP from the target, and sets the delay timer as before.

The next step is to change our `TestDecisionMaker` to pick a target and pass this target to the `TestCombatAction` constructor. This is a relatively simple change in `TestDecisionMaker.cpp`:

[PRE60]

At this point, you should be able to run the game, start a test encounter, and also see an output similar to the following:

[PRE61]

Finally, we have a combat system in which our two parties can attack each other and one or the other can win.

Next, we'll begin hooking this up to a user interface.

## Combat UI with UMG

To get started, we'll need to set up our project to properly import UMG and Slate-related classes.

First, open `RPG.Build.cs` (or `[ProjectName].Build.cs`) and change the first line of the constructor to the following code:

[PRE62]

This adds the `UMG`, `Slate`, and `SlateCore` strings to the existing string array.

Next, open `RPG.h` and make sure the following lines of code are there:

[PRE63]

Now compile the project. This may take a while.

Next, we're going to create a base class for the combat UI. Basically, we'll use this base class to allow our C++ game code to communicate with Blueprint UMG code by defining Blueprint-implementable functions in the header, which are functions that can be implemented by Blueprint and called from C++.

Create a new class named `CombatUIWidget` and select `UserWidget` as the parent class; then path it to `Source/RPG/UI`. Replace the contents of `CombatUIWidget.h` with the following code:

[PRE64]

For the most part, we're just defining a couple of functions. The `AddPlayerCharacterPanel` and `AddEnemyCharacterPanel` functions will be responsible for taking a character pointer and spawning a widget for that character (to display the character's current status).

Next, after compiling the code, back in the editor, create a new folder in the `Contents/Blueprints` directory called `UI`. In the `Content/Blueprints/UI` directory, create a new Widget Blueprint named `CombatUI`. After you've created and opened the Blueprint, go to **File** | **Reparent Blueprint** and select **CombatUIWidget** as the parent class.

In the **Designer** interface, create two Horizontal Box widgets and name them `enemyPartyStatus` and `playerPartyStatus`. These will hold child widgets for enemies and players respectively, to display the status of each character. For both of these, be sure to enable the **Is Variable** checkbox so that they will be available as variables to Blueprint. Save and compile the Blueprint.

We will position the `enemyPartyStatus` Horizontal Box at the top of the Canvas Panel. It will help to first set a top horizontal anchor.

Then set the values for the Horizontal Box as follows, **Offset Left**: 10, **Position Y**: 10, **Offset Right**: 10, **Size Y**: 200.

Proceed to position the `playerPartyStatus` Horizontal Box in a similar way; the only major difference is that we will anchor the box to the bottom of the Canvas Panel and position it so it spans the bottom of the screen:

![Combat UI with UMG](img/B04548_03_07.jpg)

Next, we'll create widgets to display player and enemy character statuses. Firstly, we'll make a base widget that each will inherit from.

Create a new Widget Blueprint and name it `BaseCharacterCombatPanel`. In this Blueprint, navigate to the graph, then add a new variable from the **MyBlueprint** tab, **CharacterTarget**, and select the **Game Character** variable type from the **Object Reference** category.

Next, we'll make separate widgets for the enemies and players.

Create a new Widget Blueprint and name it `PlayerCharacterCombatPanel`. Set the new Blueprint's parent to `BaseCharacterCombatPanel`.

In the **Designer** interface, add three text widgets. One label will be for the character's name, another for the character's HP, and third one for the character's MP. Position each Text Block so that they are anchored to the bottom left of the screen, and well within the 200 high pixels of the `playerPartyStatus` box size that we created in the `CombatUI` widget:

![Combat UI with UMG](img/B04548_03_08.jpg)

Also be sure to check **Size to Content** located in the **Details** panel of each Text Block so that the Text Block can resize if the content does not fit within the Text Block parameters.

Create a new binding for each of these by selecting the widget and clicking on **Bind** next to the **Text** input in the **Details** panel:

![Combat UI with UMG](img/B04548_03_09.jpg)

This will create a new Blueprint function that will be responsible for generating the Text Block.

To bind the HP Text Block, for example, you can execute the following steps:

1.  Right-click in an open area in the grid, search for **Get Character Target**, and then select it.
2.  Drag the output pin of this node and select **Get HP** under **Variables** | **Character Info**.
3.  Create a new **Format Text** node. Set the text to **HP: {HP}** and then connect the output of **Get HP** to the **HP** input of the **Format Text** node.
4.  Finally, connect the output of the **Format Text** node to the **Return** value of the **Return** node.

You can repeat similar steps for the character name and MP Text Blocks.

After you've created `PlayerCharacterCombatPanel`, you can repeat the same steps to create `EnemyCharacterCombatPanel`, except without the MP Text Block (as mentioned before, enemies do not consume MP). The only major difference is that the Text Blocks in `EnemyCharacterCombatPanel` need to be placed at the top of the screen to match the positioning of the `enemyPartyStatus` Horizontal Box from the `CombatUI` widget.

The resulting graph for displaying the MP will look something like the following screenshot:

![Combat UI with UMG](img/B04548_03_10.jpg)

Now that we have widgets for players and enemies, let's implement the `AddPlayerCharacterPanel` and `AddEnemyCharacterPanel` functions in the `CombatUI` Blueprint.

Firstly, we'll create a helper Blueprint function to spawn character status widgets. Name this new function `SpawnCharacterWidget` and add the following parameters to the input:

*   **Target Character** (of type Game Character Reference)
*   **Target Panel** (of type Panel Widget Reference)
*   **Class** (of type Base Character Combat Panel Class)

This function will perform the following steps:

1.  Create a new widget of the given class using **Create Widget**.
2.  Cast the new widget to the `BaseCharacterCombatPanel` type.
3.  Set the **Character Target** of the result to the **TargetCharacter** input.
4.  Add the new widget as a child of the **TargetPanel** input.

And that looks like this in Blueprint:

![Combat UI with UMG](img/B04548_03_11.jpg)

Next, in the event graph for the `CombatUI` Blueprint, right-click and add the `EventAddPlayerCharacterPanel` and `EventAddEnemyCharacterPanel` events. Hook each of these up to a `SpawnCharacterWidget` node, connecting the **Target** output to the **Target Character** input and the appropriate panel variable to the **Target Panel** input, as follows:

![Combat UI with UMG](img/B04548_03_12.jpg)

Finally, we can spawn this UI from our game mode at the beginning of combat and destroy it at the end of combat. In the header of `RPGGameMode`, add a pointer to `UCombatUIWidget` and also a class to spawn for the combat UI (so we can select a Widget Blueprint that inherits from our `CombatUIWidget` class); these should be public:

[PRE65]

Also make sure that `RPGGameMode.h` includes the `CombatWidget`; at this point, your list of includes at the top of `RPGGameMode.h` should look like this:

[PRE66]

At the end of the `TestCombat` function in `RPGGameMode.cpp`, we'll spawn a new instance of this widget, as follows:

[PRE67]

This creates the widget, adds the viewport to it, adds a mouse cursor, and then calls its `AddPlayerCharacterPanel` and `AddEnemyCharacterPanel` functions for all players and enemies respectively.

After combat is over, we'll remove the widget from the viewport and set the reference to null so it can be garbage-collected; your `Tick` function should now look like this:

[PRE68]

At this point, you can compile, but the game will crash if you test the combat. That is because you need to set `DefaultRPGGameMode` class defaults to use `CombatUI` as the `CombatUIClass` that you created in `RPGGameMode.h`. Otherwise, the system will not know that the `CombatUIClass` variable is to be pointing to `CombatUI`, which is a widget, and therefore won't be able to create the widget. Note that the editor may crash the first time you do this step.

![Combat UI with UMG](img/B04548_03_13.jpg)

Now, if you run the game and start combat, you should see the status of the goblin and the status of the player. Both should have their HP reducing until the goblin's health reaches zero; at this point, the UI disappears (as combat is over).

Next, we're going to change things so that instead of the player characters automatically making decisions, the player gets to choose their actions via the UI.

## UI-driven decision making

One idea is to change how the decision maker is assigned to the player—rather than assigning one when the player is first created, we could make our `CombatUIWidget` class implement the decision maker and just assign it when combat starts (and clear the pointer when combat ends).

We're going to have to make a couple of changes to `GameCharacter.cpp`. First, in the player overload of `CreateGameCharacter`, remove the following line of code:

[PRE69]

Then, in the `BeginDestroy` function, we'll wrap the `delete` line in an `if` statement:

[PRE70]

The reason for this is that the decision maker for players will be the UI—and we do not want to delete the UI manually (doing so would crash Unreal). Instead, the UI will be garbage-collected automatically as long as there are no `UPROPERY` decorated pointers to it.

Next, in `CombatUIWidget.h`, we'll make the class implement the `IDecisionMaker` interface and add `BeginMakeDecision` and `MakeDecision` as public functions:

[PRE71]

We're also going to add a couple of helper functions that can be called by our UI Blueprint graph:

[PRE72]

The first function retrieves a list of potential targets for the current character. The second function will give the character a new `TestCombatAction` with the given target.

Additionally, we'll add a function to be implemented in the Blueprint that will show a set of actions for the current character:

[PRE73]

We're also going to add a flag and a definition for `currentTarget`, as follows:

[PRE74]

This will be used to signal that a decision has been made (and that `MakeDecision` should return `true`).

The implementations of these four functions are fairly straightforward in `CombatUIWidget.cpp`:

[PRE75]

`BeginMakeDecision` sets the current target, sets the `finishedDecision` flag to `false`, and then calls `ShowActionsPanel` (which will be handled in our UI Blueprint graph).

`MakeDecision` simply returns the value of the `finishedDecision` flag.

`AttackTarget` assigns a new `TestCombatAction` to the character and then sets `finishedDecision` to `true` to signal that a decision has been made.

Finally, `GetCharacterTargets` returns an array of this character's possible opponents.

Since the UI now implements the `IDecisionMaker` interface, we can assign it as the decision maker for the player characters. Firstly, in the `TestCombat` function of `RPGGameMode.cpp`, we'll change the loop that iterates over the characters so that it assigns the UI as the decision maker:

[PRE76]

Then, we'll set the players' decision makers to null when combat is over:

[PRE77]

Now, player characters will use the UI to make decisions. However, the UI currently does nothing. We'll need to work in Blueprint to add this functionality.

Firstly, we'll create a widget for the attack target options. Name it `AttackTargetOption`, add a button, and put a Text Block in the button. Check **Size to Content** so that the button will dynamically resize to any Text Block that is in the button. Then position it at the top-left corner of the Canvas Panel.

In the Graph, add two new variables. One is the `targetUI` of the Combat UI Reference type. The other is the `target` of the Game Character Reference type. From the **Designer** view, click on your button, then scroll down the **Details** panel and click on **OnClicked** to create an event for the button. The button will use the `targetUI` reference to call the **Attack Target** function and the `target` reference (which is the target this button represents) to pass to the **Attack Target** function.

The graph for the button-click event is fairly simple; just route the execution to the **Attack Target** function of the assigned `targetUI` and pass the `target` reference as a parameter:

![UI-driven decision making](img/B04548_03_14.jpg)

Next, we'll add a panel for character actions to the main combat UI. This is a Canvas Panel with a single button child for **Attack** and a Vertical Box for the target list:

![UI-driven decision making](img/B04548_03_15.jpg)

Name the **Attack** button `attackButton`. Name the Vertical Box `targets`. And name the Canvas Panel encapsulating these items as `characterActions`. These should have **Is Variable** enabled so that they are visible to Blueprint.

Then, in the Blueprint graph, we'll implement the **Show Actions Panel** event. This will first route execution to a **Set Visibility** node, which will enable the **Actions** panel and then route execution to another **Set Visibility** node that hides the target list:

![UI-driven decision making](img/B04548_03_16.jpg)

The Blueprint graph for when the **Attack** button is clicked is fairly large, so we'll take a look at it in small chunks.

Firstly, create an `OnClicked` event for your `attackButton` by selecting the button in the **Designer** view and clicking on **OnClicked** in the **Events** portion of the **Details** panel. In the graph, we then use a **Clear Children** node when the button is clicked to clear out any target options that may have been previously added:

![UI-driven decision making](img/B04548_03_17.jpg)

Then, we use a **ForEachLoop** coupled with a **CompareInt** node to iterate over all characters returned by **Get Character Targets** that have HP > 0 (not dead):

![UI-driven decision making](img/B04548_03_18.jpg)

From the **>** (greater than) pin of the **CompareInt** node, we create a new instance of the **AttackTargetOption** widget and add it to the attack target list Vertical Box:

![UI-driven decision making](img/B04548_03_19.jpg)

Then, for the widget we just added, we connect a **Self** node to set its `targetUI` variable and pass the **Array Element** pin of the **ForEachLoop** to set its `target` variable:

![UI-driven decision making](img/B04548_03_20.jpg)

Finally, from the **Completed** pin of the **ForEachLoop**, we set the visibility of the target option list to **Visible**:

![UI-driven decision making](img/B04548_03_21.jpg)

After all this is done, we still need to hide the **Actions** panel when an action is chosen. We'll add a new function to the `CombatUI` called **Hide Action Panel**. This function is very simple; it just sets the visibility of the action panel to **Hidden**:

![UI-driven decision making](img/B04548_03_22.jpg)

Also, in the click handler in the **AttackTargetOption** graph, we connect the execution pin of the **Attack Target** node to this **Hide Action Panel** function:

![UI-driven decision making](img/B04548_03_23.jpg)

Lastly, you will need to bind the Text Block that was in the button located in the **AttackTargetOption** widget. So go into the **Designer** view and create a bind for the text just like you have done with previous Text Blocks in this chapter. Now in the graph, link **target** to the **Character Name**, and adjust the format of the text to show the `CharacterName` variable, and link it to the **return** node of your text. This Blueprint should show the current target's character name on the button:

![UI-driven decision making](img/B04548_03_24.jpg)

After all this, you should be able to run the game and start a test encounter, and on the player's turn, you'll see an **Attack** button that allows you to pick the goblin to attack.

Our combat engine is now fully functional. The final step of this chapter will be to create a game over screen so that when all party members have died, the player will see a **Game Over** message.

## Creating the game over screen

The first step is to create the screen itself. Create a new Widget Blueprint called **GameOverScreen**. We'll just add an image to which we can do a full-screen anchor, and zero out the offsets in the **Details** panel. You can also set the color to black. Also add a Text Block with the text **Game Over**, and a button with a child Text Block **Restart**:

![Creating the game over screen](img/B04548_03_25.jpg)

Create an `OnClicked` event for the **Restart** button. In the Blueprint graph, link the event for the button to Restart Game whose target is **Get Game Mode** (you may have to uncheck **Context Sensitive** to find this node):

![Creating the game over screen](img/B04548_03_26.jpg)

You will also need to show the mouse cursor here. The best way to do this is from **Event Construct**; link **Set Show Mouse Cursor**, whose target is **Get Player Controller**. Be sure to check the **Show Mouse Cursor** box. Between **Event Construct** and **Set Show Mouse Cursor**, put a 0.2-second delay so that you are assured that the mouse re-appears after you removed it when combat ended:

![Creating the game over screen](img/B04548_03_27.jpg)

Next, in `RPGGameMode.h`, we add a public property for the widget type to be used for game over:

[PRE78]

In the case of game over, we create the widget and add it to the viewport, which we can add as a condition nested in the `if( combatOver )` condition within `void ARPGGameMode::Tick( float DeltaTime )` in `RPGGameMode.cpp`:

[PRE79]

As you can see, we're also calling a `PrepareReset` function on the game instance. This function isn't defined yet, so we'll create it now in `RPGGameInstance.h` as a public function:

[PRE80]

Then implement it in `RPGGameInstance.cpp`:

[PRE81]

In this case, the purpose of `PrepareReset` is to set `isInitialized` to `false` so that the next time `Init` is called, the party members are reloaded. We are also emptying the `partyMembers` array so that when party members are added back into the array, we don't append them to instances of party members from our last playthrough (we don't want to reset the game with dead party members).

At this point, you can compile. But before we can test this, we need to set the **Game Over UIClass** that we created and set it to **GameOverScreen** as a class default in **DefaultRPGGameMode**:

![Creating the game over screen](img/B04548_03_28.jpg)

Much like the last time you did this, the editor may crash, but when you come back to **DefaultRPGGameMode**, you should see that **GameOverScreen** is set correctly.

In order to test this, we'll need to give the goblin more health than the player. Open the enemies table and give the goblin anything over 100 HP (for instance, 200 would do). Then, start an encounter and play until the main party member runs out of health. You should then see a **Game Over** screen pop up, and by clicking on **Restart**, you will restart the level and the main party member will be back up to 100 HP.

# Summary

In this chapter, we created a foundation for the core gameplay of an RPG. We have a character that can explore the overworld, a system for keeping track of party members, a turn-based combat engine, and a game over condition.

In the next chapters, we'll expand this by adding an inventory system, allowing the player to consume items, and give their party members equipment to boost their stats.