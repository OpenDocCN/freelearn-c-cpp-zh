# Chapter 10. Inventory System and Pickup Items

We want our player to be able to pick up items from the game world. In this chapter, we will code and design a backpack for our player to store items. We will display what the player is carrying in the pack when the user presses the *I* key.

As a data representation, we can use the `TMap<FString, int>` items covered in the previous chapter to store our items. When the player picks up an item, we add it to the map. If the item is already in the map, we just increase its value by the quantity of the new items picked up.

# Declaring the backpack

We can represent the player's backpack as a simple `TMap<FString, int>` item. To allow your player to gather items from the world, open the `Avatar.h` file and add the following `TMap` declaration:

[PRE0]

## Forward declaration

Before `AAvatar` class, notice that we have a `class APickupItem` forward declaration. Forward declarations are needed in a code file when a class is mentioned (such as the `APickupItem::Pickup( APickupItem *item );` function prototype), but there is no code in the file actually using an object of that type inside the file. Since the `Avatar.h` header file does not contain executable code that uses an object of the type `APickupItem`, a forward declaration is what we need.

The absence of a forward declaration will give a compiler error, since the compiler won't have heard of `class APickupItem` before compiling the code in `class AAvatar`. The compiler error will come at the declaration of the `APickupItem::Pickup( APickupItem *item );` function prototype declaration.

We declared two `TMap` objects inside the `AAvatar` class. This is how the objects will look, as shown in the following table:

| `FString` (name) | `int` (quantity) | `UTexture2D*` (im) |
| --- | --- | --- |
| GoldenEgg | 2 | ![Forward declaration](img/00142.jpeg) |
| MetalDonut | 1 | ![Forward declaration](img/00143.jpeg) |
| Cow | 2 | ![Forward declaration](img/00144.jpeg) |

In the `TMap` backpack, we store the `FString` variable of the item that the player is holding. In the `Icons` map, we store a single reference to the image of the item the player is holding.

At render time, we can use the two maps working together to look up both the quantity of an item that the player has (in his `Backpack` mapping) and the texture asset reference of that item (in the `Icons` map). The following screenshot shows how the rendering of the HUD will look:

![Forward declaration](img/00145.jpeg)

### Note

Note that we can also use an array of `struct` with an `FString` variable and `UTexture2D*` in it instead of using two maps.

For example, we can keep `TArray<Item> Backpack;` with a `struct` variable, as shown in the following code:

[PRE1]

Then, as we pick up items, they will be added to the linear array. However, counting the number of each item we have in the backpack will require constant reevaluation by iterating through the array of items each time we want to see the count. For example, to see how many hairbrushes you have, you will need to make a pass through the whole array. This is not as efficient as using a map.

## Importing assets

You might have noticed the **Cow** asset in the preceding screenshot, which is not a part of the standard set of assets that UE4 provides in a new project. In order to use the **Cow** asset, you need to import the cow from the **Content Examples** project. There is a standard importing procedure that UE4 uses.

In the following screenshot, I have outlined the procedure for importing the **Cow** asset. Other assets will be imported from other projects in UE4 using the same method. Perform the following steps to import the **Cow** asset:

1.  Download and open UE4's **Content Examples** project:![Importing assets](img/00146.jpeg)
2.  After you have downloaded **Content Examples**, open it and click on **Create Project**:![Importing assets](img/00147.jpeg)
3.  Next, name the folder in which you will put your `ContentExamples` and click on **Create**.
4.  Open your `ContentExamples` project from the library. Browse the assets available in the project until you find one that you like. Searching for `SM_` will help since all static meshes usually begin with `SM_` by convention.![Importing assets](img/00148.jpeg)

    Lists of static meshes, all beginning with SM_

5.  When you find an asset that you like, import it into your project by right-clicking on the asset and then clicking on **Migrate...**:![Importing assets](img/00149.jpeg)
6.  Click on **OK** in the **Asset Report** dialog:![Importing assets](img/00150.jpeg)
7.  Select the **Content** folder from your project that you want to add the **SM_Door** file to. For me, I want to add it to `Y:/Unreal Projects/GoldenEgg/Content`, as shown in the following screenshot:![Importing assets](img/00151.jpeg)
8.  If the import was completed successfully, you will see a message as follows:![Importing assets](img/00152.jpeg)
9.  Once you import your asset, you will see it show up in your asset browser inside your project:![Importing assets](img/00153.jpeg)

You can then use the asset inside your project normally.

## Attaching an action mapping to a key

We need to attach a key to activate the display of the player's inventory. Inside the UE4 editor, add an **Action Mappings +** called `Inventory` and assign it to the keyboard key *I*:

![Attaching an action mapping to a key](img/00154.jpeg)

In the `Avatar.h` file, add a member function to be run when the player's inventory needs to be displayed:

[PRE2]

In the `Avatar.cpp` file, implement the `ToggleInventory()` function, as shown in the following code:

[PRE3]

Then, connect the `"Inventory"` action to `AAvatar::ToggleInventory()` in `SetupPlayerInputComponent()`:

[PRE4]

# Base class PickupItem

We need to define how a pickup item looks in code. Each pickup item will derive from a common base class. Let's construct the base class for a `PickupItem` class now.

The `PickupItem` base class should inherit from the `AActor` class. Similar to how we created multiple NPC blueprints from the base NPC class, we can create multiple `PickupItem` blueprints from a single `PickupItem` base class, as shown in the following screenshot:

![Base class PickupItem](img/00155.jpeg)

Once you have created the `PickupItem` class, open its code in Visual Studio.

The `APickupItem` class will need quite a few members, as follows:

*   An `FString` variable for the name of the item being picked up
*   An `int32` variable for the quantity of the item being picked up
*   A `USphereComponent` variable for the sphere that you will collide with for the item to be picked up
*   A `UStaticMeshComponent` variable to hold the actual `Mesh`
*   A `UTexture2D` variable for the icon that represents the item
*   A pointer for the HUD (which we will initialize later)

This is how the code in `PickupItem.h` looks:

[PRE5]

The point of all these `UPROPERTY()` declarations is to make `APickupItem` completely configurable by blueprints. For example, the items in the **Pickup** category will be displayed as follows in the blueprints editor:

![Base class PickupItem](img/00156.jpeg)

In the `PickupItem.cpp` file, we complete the constructor for the `APickupItem` class, as shown in the following code:

[PRE6]

In the first two lines, we perform an initialization of `Name` and `Quantity` to values that should stand out to the game designer as being uninitialized. I used block capitals so that the designer can clearly see that the variable has never been initialized before.

We then initialize the `ProxSphere` and `Mesh` components using `PCIP.CreateDefaultSubobject`. The freshly initialized objects might have some of their default values initialized, but `Mesh` will start out empty. You will have to load the actual mesh later, inside blueprints.

For the mesh, we set it to simulate realistic physics so that pickup items will bounce and roll around if they are dropped or moved. Pay special attention to the line `ProxSphere->AttachTo( Mesh )`. This line tells you to make sure the pickup item's `ProxSphere` component is attached to the `Mesh` root component. This means that when the mesh moves in the level, `ProxSphere` follows. If you forget this step (or if you did it the other way around), then `ProxSphere` will not follow the mesh when it bounces.

## The root component

In the preceding code, we assigned `RootComponent` of `APickupItem` to the `Mesh` object. The `RootComponent` member is a part of the `AActor` base class, so every `AActor` and its derivatives has a root component. The root component is basically meant to be the core of the object, and also defines how you collide with the object. The `RootComponent` object is defined in the `Actor.h` file, as shown in the following code:

[PRE7]

So the UE4 creators intended `RootComponent` to always be a reference to the collision primitive. Sometimes the collision primitive can be capsule shaped, other times it can be spherical or even box shaped, or it can be arbitrarily shaped, as in our case, with the mesh. It's rare that a character should have a box-shaped root component, however, because the corners of the box can get caught on walls. Round shapes are usually preferred. The `RootComponent` property shows up in the blueprints, where you can see and manipulate it.

![The root component](img/00157.jpeg)

You can edit the ProxSphere root component from its blueprints once you create a blueprint based on the PickupItem class

Finally, the `Prox_Implementation` function gets implemented, as follows:

[PRE8]

A couple of tips here that are pretty important: first, we have to access a couple of *globals* to get the objects we need. There are three main objects we'll be accessing through these functions that manipulate the HUD: the controller (`APlayerController`), the HUD (`AMyHUD`), and the player himself (`AAvatar`). There is only one of each of these three types of objects in the game instance. UE4 has made finding them easy.

### Getting the avatar

The `player` class object can be found at any time from any place in the code by simply calling the following code:

[PRE9]

We then pass him the item by calling the `AAvatar::Pickup()` function defined earlier.

Because the `PlayerPawn` object is really an `AAvatar` instance, we cast the result to the `AAvatar` class, using the `Cast<AAvatar>` command. The `UGameplayStatics` family of functions are accessible anywhere in your code—they are global functions.

### Getting the player controller

Retrieving the player controller is from a *superglobal* as well:

[PRE10]

The `GetWorld()` function is actually defined in the `UObject` base class. Since all UE4 objects derive from `UObject`, any object in the game actually has access to the `world` object.

### Getting the HUD

Although this organization might seem strange at first, the HUD is actually attached to the player's controller. You can retrieve the HUD as follows:

[PRE11]

We cast the HUD object since we previously set the HUD to being an `AMyHUD` instance in blueprints. Since we will be using the HUD often, we can actually store a permanent pointer to the HUD inside our `APickupItem` class. We will discuss this point later.

Next, we implement `AAvatar::Pickup`, which adds an object of the type `APickupItem` to Avatar's backpack:

[PRE12]

In the preceding code, we check whether the pickup item that the player just got is already in his pack. If it is, we increase its quantity. If it is not in his pack, we add it to both his pack and the `Icons` mapping.

To add the pickup items to the pack, use the following line of code:

[PRE13]

The `APickupItem::Prox_Implementation` is the way this member function will get called.

Now, we need to display the contents of our backpack in the HUD when the player presses *I*.

# Drawing the player inventory

An inventory screen in a game such as *Diablo* features a pop-up window, with the icons of the items you've picked up in the past arranged in a grid. We can achieve this type of behavior in UE4.

There are a number of approaches to drawing a UI in UE4\. The most basic way is to simply use the `HUD::DrawTexture()` calls. Another way is to use Slate. Another way still is to use the newest UE4 UI functionality: **Unreal Motion Graphics** (**UMG**) Designer.

Slate uses a declarative syntax to lay out UI elements in C++. Slate is best suited for menus and the like. UMG is new in UE 4.5 and uses a heavily blueprint-based workflow. Since our focus here is on exercises that use C++ code, we will stick to a `HUD::DrawTexture()` implementation. This means that we will have to manage all the data that deals with the inventory in our code.

## Using HUD::DrawTexture()

We will achieve this in two steps. The first step is to push the contents of our inventory to the HUD when the user presses the *I* key. The second step is to actually render the icons into the HUD in a grid-like fashion.

To keep all the information about how a widget can be rendered, we declare a simple structure to keep the information concerning what icon it uses, its current position, and current size.

This is how the `Icon` and `Widget` structures look:

[PRE14]

You can add these structure declarations to the top of `MyHUD.h`, or you can add them to a separate file and include that file everywhere those structures are used.

Notice the four member functions on the `Widget` structure to get to the `left()`, `right()`, `top()`, and `bottom()` functions of the widget. We will use these later to determine whether a click point is inside the box.

Next, we declare the function that will render the widgets out on the screen in the `AMyHUD` class:

[PRE15]

A call to the `DrawWidgets()` function should be added to the `DrawHUD()` function:

[PRE16]

Next, we will fill the `ToggleInventory()` function. This is the function that runs when the user presses *I*:

[PRE17]

For the preceding code to compile, we need to add a function to `AMyHUD`:

[PRE18]

We keep using the `Boolean` variable in `inventoryShowing` to tell us whether the inventory is currently displayed or not. When the inventory is shown, we also show the mouse so that the user knows what he's clicking on. Also, when the inventory is displayed, free motion of the player is disabled. The easiest way to disable a player's free motion is by simply returning from the movement functions before actually moving. The following code is an example:

[PRE19]

### Exercise

Check out each of the movement functions with the `if( inventoryShowing ) { return; }` short circuit return.

## Detecting inventory item clicks

We can detect whether someone is clicking on one of our inventory items by doing a simple hit point-in-box hit. A point-in-box test is done by checking the point of the click against the contents of the box.

Add the following member function to `struct Widget`:

[PRE20]

The point-in-box test is as follows:

![Detecting inventory item clicks](img/00158.jpeg)

So, it is a hit if `p.X` is all of:

*   Right of `left() (p.X > left())`
*   Left of `right() (p.X < right())`
*   Below the `top() (p.Y > top())`
*   Above the `bottom() (p.Y < bottom())`

Remember that in UE4 (and UI rendering in general) the *y* axis is inverted. In other words, y goes down in UE4\. This means that `top()` is less than `bottom()` since the origin (the `(0, 0)` point) is at the top-left corner of the screen.

### Dragging elements

We can drag elements easily. The first step to enable dragging is to respond to the left mouse button click. First, we'll write the function to execute when the left mouse button is clicked. In the `Avatar.h` file, add the following prototype to the class declaration:

[PRE21]

In the `Avatar.cpp` file, we can attach a function to execute on a mouse click and pass the click request to the HUD, as follows:

[PRE22]

Then in `AAvatar::SetupPlayerInputComponent`, we have to attach our responder:

[PRE23]

The following screenshot shows how you can attach a render:

![Dragging elements](img/00159.jpeg)

Add a member to the `AMyHUD` class:

[PRE24]

Next, in `AMyHUD::MouseClicked()`, we start searching for the `Widget` hit:

[PRE25]

In the `AMyHUD::MouseClicked` function, we loop through all the widgets that are on the screen and check for a hit with the current mouse position. You can get the current mouse position from the controller at any time by simply looking up `PController->GetMousePosition()`.

Each widget is checked against the current mouse position, and the widget that got hit by the mouse click will be moved once a mouse is dragged. Once we have determined which widget got hit, we can stop checking, so we have a `return` value from the `MouseClicked()` function.

Hitting widget is not enough, though. We need to drag the widget that got hit when the mouse moves. For this, we need to implement a `MouseMoved()` function in `AMyHUD`:

[PRE26]

Don't forget to include a declaration in the `MyHUD.h` file.

The drag function looks at the difference in the mouse position between the last frame and this frame and moves the selected widget by that amount. A `static` variable (global with local scope) is used to remember the `lastMouse` position between the calls for the `MouseMoved()` function.

How can we link the mouse's motion to running the `MouseMoved()` function in `AMyHUD`? If you remember, we have already connected the mouse motion in the `Avatar` class. The two functions that we used were `AAvatar::Pitch()` (the y axis) and `AAvatar::Yaw()` (the x axis). Extending these functions will enable you to pass mouse inputs to the HUD. I will show you the `Yaw` function now, and you can extrapolate how `Pitch` will work from there:

[PRE27]

The `AAvatar::Yaw()` function first checks whether the inventory is showing or not. If it is showing, inputs are routed straight to the HUD, without affecting `Avatar`. If the HUD is not showing, inputs just go to `Avatar`.

### Exercises

1.  Complete the `AAvatar::Pitch()` function (y axis) to route inputs to the HUD instead of to `Avatar`.
2.  Make the NPC characters from [Chapter 8](part0056_split_000.html#1LCVG1-dd4a3f777fc247568443d5ffb917736d "Chapter 8. Actors and Pawns"), *Actors and Pawns*, give the player an item (such as `GoldenEgg`) when he goes near them.

# Summary

In this chapter, we covered how to set up multiple pickup items for the player to see displayed in the level and also pick up. In the next chapter, we will introduce *Monsters* and the player will be able to defend himself against the monsters using magic spells.