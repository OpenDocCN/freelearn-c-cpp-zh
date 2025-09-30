# Chapter 6. Blueprints

In this chapter, we will learn what Blueprints are and how they can be used to prototype your game. We will learn about:

*   Getting familiar with Blueprint editor
*   Various Blueprint graph types (for example, function graphs, event graphs, and so on)
*   Blueprint nodes
*   And, finally, we will create a simple Blueprint that can be placed in world or dynamically spawned while running the game

Blueprint Visual Scripting in Unreal Engine 4 is an extremely powerful and flexible node-based interface to create gameplay elements and provides artists and designers with the ability to program their game and to quickly iterate gameplay within the editor without writing a single line of code! Using Blueprints you can create and tweak gameplay, characters, inputs, environments, and virtually anything in your game.

Blueprints work by using graphs that contain various nodes connected to each other, which defines what the Blueprint does. For example, it can be gameplay events, spawning new Actors, or anything really.

# Different Blueprint types

Let's take a quick look at various Blueprint types that are available in Unreal Engine 4:

*   **Level Blueprint**: Level Blueprint is a special Blueprint that acts as a level-wide global event graph, which the user can neither remove nor create. Each level will have its own level Blueprint that the user can use to create events that pertain to the whole level. The user can use this graph to call events on a specific actor present in the level or play a Matinee sequence. Users who are familiar with Unreal Engine 3 (or UDK) should be familiar with this concept as this is similar to how Kismet worked in those Engines.
*   **Class Blueprint**: Commonly referred to as just Blueprint, is an asset that you create inside **Content Browser**. Once the asset is created, you define its behavior visually instead of typing any code. This Blueprint is saved as an asset in **Content Browser** so you can drag and drop this into your world as an instance or spawn dynamically in another Blueprint graph.
*   **Animation Blueprint**: These are specialized graphs that control the animation of a skeletal mesh by blending animations, controlling the bones directly, and outputting a final pose in each frame. Animation Blueprints will always have two graphs, namely **EventGraph** and **AnimGraph**.
*   **EventGraph**: This uses a collection of animation-related events to initiate a sequence of nodes, which updates the values used to drive animations within **Animgraph**.
*   **AnimGraph**: This is used to evaluate the final pose for your **Skeletal Mesh**. In this graph, you can perform animation blends or control bone transforms using **SkeletalControls**.
*   **Macro Library**: These are containers that can hold various macros or graphs that you can use multiple times in any other Blueprint class. Macro libraries cannot contain variables or inherit from other Blueprints or be placed in the level. They are just a collection of graphs that you use commonly and can be a time-saver. If you are referencing a macro in your Blueprint then changes to that macro will not be applied to your Blueprint until you recompile your Blueprint. Compiling a Blueprint means converting all the properties and graphs into a class that Unreal can use.
*   **Blueprint Interface**: These are graphs that contain one or more functions without implementation. Other classes that add this interface must include the functions in a unique manner. This has the same concept of interface in programming where you can access various objects with a common interface and share or send data to one another. Interface graphs have some limitations in that you cannot create variables, edit graphs, or add any components.

# Getting familiar with the Blueprint user interface

The Blueprint **User Interface** (**UI**) contains various tabs by default. In the following screenshot you can see the unified layout of the Blueprint UI:

![Getting familiar with the Blueprint user interface](img/B03950_06_01.jpg)

Let's take a look at these tabs:

*   **Components**
*   **My Blueprint**
*   **Toolbar**
*   **Graph editor**
*   **Details panel**

## Components tab

Most Blueprint classes can have different types of components. These can be light components, mesh components, UI components, and so on. In this section, we will see what they are and how we can use them in our Blueprint classes.

### What are components?

Components are the bits and pieces that make up the whole Actor. Components cannot exist on their own but when added to an Actor, the Actor will then have access to all the functionalities provided by the component. For example, think about a car. The wheels, body, lights, and so on can be considered as components and the car itself as the Actor. Then in the graph, you can access the component and do the logic for your car Actor. Components are always instanced and each Actor instance will have its own unique instance of components. If this were not the case, then, if we place multiple car Actors in world and if one starts moving, all the others will also move.

### Adding a component

To add a component to your Actor, click the **Add Component** button on the **Components** tab. After clicking the button it will show a list of various **Components** that you can add.

![Adding a component](img/B03950_06_02.jpg)

After adding a component, you will be prompted to give it a name. Components can also be directly added simply by dragging-and-dropping from **Content Browser** to the **Components** window.

To rename a component, you can select it in the **Components** tab and press *F2*.

### Note

The drag-and-drop method only applies to **StaticMeshes**, **SkeletalMeshes**, **SoundCues**, and **ParticleSystems**.

With the component selected, you can delete it by pressing the *Delete* key. You can also right-click on the component and select **Delete** to remove it as well.

### Transforming the component

Once the component is added and selected, you can use the transform tools (*W*, *E*, and *R*) to change the location, rotation, and scale of the component either by entering values in the **Details** panel or in the **Viewport** tab. When using moving, rotating, or scaling, you can press *Shift* to enable snapping, provided you have enabled grid snapping in the **Viewport** toolbar.

### Note

If the Component has any child components attached to it then moving, rotating or scaling that component will propagate the transformation to all child components too.

### Adding events for components

Adding events based on a component is very easy and can be done by different methods. Events created in this manner are specific to that component and need not be tested as to which component is involved:

*   **Adding events from the details panel**: When you select the component you will see all the events available for that component in the **Details** panel as buttons. When you click on any of them, the editor will create the event node specific for that component in the event graph.![Adding events for components](img/B03950_06_03.jpg)
*   **Adding events by right-clicking**: When you right-click on a component, you will see **Add Event** in the context menu. From there you can select any event you want and editor will create the event node specific to that component in the event graph.![Adding events for components](img/B03950_06_04.jpg)
*   **Adding events in the graph**: Once you select your component in the **My Blueprints** tab, you can right-click on the graph and get all the **Events** for that component.![Adding events for components](img/B03950_06_05.jpg)

## My Blueprints tab

The **My Blueprints** tab displays a list of **Graphs**, **Functions**, **Macros**, **Variables**, and so on that are contained within your Blueprint. This tab is dependent on the type of Blueprint. For example, a class Blueprint will have **EventGraph**, **ConstructionScript Graph**, **Variables**, **Functions**, **Macros**, and so on. An interface will only show the list of functions within it. A **Macro Library** will show only the macros created within it.

### Creation buttons

You can create new variables, functions, macros, event graphs, and event dispatchers inside the **My Blueprints** tab by clicking the shortcut button (**+**).

![Creation buttons](img/B03950_06_06.jpg)

You can also add them by clicking the **+Add New** drop-down button.

### Searching in my Blueprint

The **My Blueprint** tab also provides a search area to search for your variables, functions, macros, event graphs, and event dispatchers. You can search based on name, comment, or any other data.

### Categorizing in My Blueprint

It is always a good practice to organize your variables, functions, macros, event dispatchers, and so on into various categories. In the **My Blueprints** tab, you can have as many categories with sub-categories. Check the following screenshot:

![Categorizing in My Blueprint](img/B03950_06_07.jpg)

Here you can see how I have organized everything into various categories and sub-categories. To set a category for your variables, functions, macros, and event dispatchers, simply select them and in the **Details** panel you can type your new category name or select from an existing category. If you need sub-categories then you need to separate your sub-category name with a vertical bar key (**|**). For example, if you want **Health** as a sub-category in **Attributes**, you can set it like this: **Attributes** | **Health**.

## Toolbar

The toolbar provides access to common commands required while editing Blueprints. Toolbar buttons will be different depending on which mode (editing mode, play in editor mode, and so on) is active and which Blueprint type you are currently editing.

## Graph editor

Graph editor is the main area of your Blueprint. This is where you add new nodes and connect them to create the network that defines the scripted behavior. More information on how to create new nodes and various nodes will be explained later on in this book.

## Details panel

The **Details** panel provides access to properties of the selected **Components** or **Variables**. It contains a search field so you can search for a specific property.

# Blueprint graph types

As we mentioned before, Blueprints are assets that are saved in **Content Browser** that are used to create new types of Actors or script gameplay logic, events, and so on, giving both designers and programmers the ability to quickly iterate gameplay without writing a single line of code. In order for a Blueprint to have scripted behavior, we need to define how it behaves using various nodes in graph editor. Let's take a quick look at various graphs:

*   **Construction Script Graph**: Construction graph is executed the moment the Blueprint is initialized and whenever a change happens to any variables within the Blueprint. This means that whenever you place an instance of the Blueprint in the level and change its transformation or any variable, the construction graph is executed. This graph is executed once every time it is constructed and again when any of the properties or Blueprint is updated. This can be used to construct procedural elements or to set up values before the game begins.
*   **Event Graph**: This is where all the gameplay logic is contained, including interactivity and dynamic responses. Using various event nodes as entry points to functions, flow controls, and variables, you can script the behavior of the Blueprint. Event graphs are only executed when you start the game.
*   **Function Graph**: By default, this graph contains one single entry point with the name of the function. This node can never be deleted but you can move it around freely. Nodes in this graph are only executed when you call this function in the construction or event graph or from another Blueprint that is referencing the Blueprint that this function belongs to.
*   **Macro Graph**: This is like a collapsed graph that contains your nodes. Unlike function graphs, macros can have multiple inputs or outputs.
*   **Interface Graph**: Interface graphs are disabled and you cannot move, create graphs, variables, or components.

### Note

Only class Blueprints have **Construction Script** and it stops executing when gameplay begins and is considered completed before gameplay.

## Function graph

Function graphs are node graphs created inside a Blueprint and can be executed from another graph (such as **Event Graph** or **Construction Script**) or from another Blueprint. By default, function graphs contain a single execution pin that is activated when the function is called, causing the connected nodes to execute.

### Creating functions

Function graphs are created through **My Blueprints** tab and you can create as many functions as you want.

Inside **My Blueprints** tab you can hover your mouse over the functions header and click on **+Function** to add a new function

![Creating functions](img/B03950_06_08.jpg)

Clicking that button (the yellow highlighted button) will create a new function and prompts you to enter a new name for it.

### Graph settings

When you create a new function and select it, you will get some properties of that function, which you can change in the **Details** panel. Let's take a quick look at them.

![Graph settings](img/B03950_06_09.jpg)

*   **Description**: Appears as a tooltip when you hover your mouse over this function in another graph.
*   **Category**: Keeps this function in its given category (for organizational purpose only).
*   **Access Specifier**: Sometimes when you create functions, you don't want to access some of them in another Blueprint. Access specifiers let you specify what other objects can access this function.
*   **Public**: This means any object can access this function from anywhere. This is the default setting.
*   **Protected**: This means current Blueprint and any Blueprints derived from the current Blueprint can access this function.
*   **Private**: This setting means only the current Blueprint can access this function.
*   **Pure**: When enabled, this function is marked as a **Pure Function** and when disabled it is an **Impure Function**.

    *   **Pure Function** will not modify state or members of a class in any way and is considered a **Constant Function** that only outputs a data value and does not have an execution pin. These are connected to other **Data Pins** and are automatically executed when the data on them is required.
    *   **Impure Function** is free to modify any value in a class and contains an execution pin.

The following is a screenshot showing the difference between **Pure Function** and **Impure Function**:

![Graph settings](img/B03950_06_10.jpg)

### Editing functions

To define the functionality of the function you need to edit it. You can have as many inputs or outputs as you want, and can then create a node network between those inputs and outputs to define the functionality. To add input or output, you first need to select the function either in the **My Blueprint** tab or select the main pink node when you open the **Function Graph**. Then, in the **Details** panel, you will see a button labelled **New** that creates new inputs or outputs.

![Editing functions](img/B03950_06_11.jpg)

In this screenshot you can see how I added new inputs and outputs to **Function Example**.

### Note

**ReturnNode** is optional and will only appear if you have at least one output data pin. If you remove all output pins then **ReturnNode** is automatically removed and you can still use your function.

For example, in the following screenshot I created a Blueprint function that appends a prefix to my character name so I can use this one single function to change the prefix anytime I want.

![Editing functions](img/B03950_06_12.jpg)

Now, back in **Event Graph**, I call this function on the **Begin Play** event so I can set the character name when the game starts.

![Editing functions](img/B03950_06_13.jpg)

## Macro graph

Macro graphs are essentially collapsed graphs of nodes, which contain an entry point and exit point designated by tunnel nodes but cannot contain variables. Macro graphs can have any number of execution or data pins.

Macros can be created inside a **Class Blueprint** or **Level Blueprint** like functions or you can organize your **Macros** in a **Blueprint Macro Library**, which can be created in **Content Browser**.

**Blueprint Macro Library** can contain all your **Macros** in one place so you can use them in any other Blueprint. These can be real time-savers as they can contain most commonly used nodes and can transfer data. But changes to a macro graph are only reflected when the Blueprint containing that macro is recompiled.

To create a macro library you need to right-click in **Content Browser** and select **Blueprint Macro Library** from the Blueprints sub-category.

![Macro graph](img/B03950_06_14.jpg)

Once you select that option you have to select a parent class for your Macro. Most of the time we select Actor as the parent class. After the selection, you will be prompted to type a name for your Macro library and save it.

If you just created your Macro library, the editor will create a blank Macro named **NewMacro_0** and will be highlighted for you to rename.

As you did with functions, you can type a description and define a **Category** for your Macro. You also get an option to define a color for your Macro using **Instance Color**.

In the following screenshot you can see I created a Macro with multiple outputs and defined a **Description**, **Category**, and an **Instance Color** for the Macro:

![Macro graph](img/B03950_06_15.jpg)

In any other Blueprint I can now get this Macro and use it. If you hover you mouse over the Macro, you can see the description you set as a **Tooltip**.

![Macro graph](img/B03950_06_16.jpg)

## Interface graph

Interface graphs are a collection of functions without any implementation, which can be added to other Blueprints. Any Blueprint class implementing an interface will definitely contain all the functions from the interface. It is then up to the user to give functionality to the functions in that interface. Interface editor is similar to other Blueprints but you cannot add new variables, edit the graph, or add any components.

Interfaces are used to communicate between various Blueprints that share specific functionality. For example, if the player is having a **Flame Thrower** gun and in the game you have **Ice** and **Cloth**, both can take damage but one should melt and the other should burn. You can create a **Blueprint Interface** that contains a **TakeWeaponFire** function and have **Ice** and **Cloth** implement this interface. Then, in **Ice Blueprint**, you can implement the **TakeWeaponFire** function and make the ice melt and, in **Cloth Blueprint**, you can implement that same function and make the cloth burn. Now when you are firing your **Flame Thrower** you can simply call the **TakeWeaponFire** function and it calls them in those Blueprints.

To create a new interface, you need to right-click on the **Content Browser** and select **Blueprint Interface** from the Blueprints sub-category and then name it.

In the following example I named it **BP_TestInterface**:

![Interface graph](img/B03950_06_17.jpg)

If you just created your interface the editor will create a blank function named **NewFunction_0**, which will be highlighted for you to rename. If you implement this interface on any Blueprint then it will have this function.

In this example, I created a function called **MyInterfaceFunction**. We will use this to simply print out the Actor name that implements this interface.

![Interface graph](img/B03950_06_18.jpg)

To create functionality for this function, we first need to implement this interface in a Blueprint. So open your Blueprint where you want this to be implemented and select **Class Settings** in the **Toolbar**.

![Interface graph](img/B03950_06_19.jpg)

Now the **Details** panel will show the settings for this Blueprint and, under the **Interfaces** section, you can add your interface.

![Interface graph](img/B03950_06_20.jpg)

Once you add that interface, the **My Blueprints** tab will update to show you the interface functions. Now all you have to do is double-click on the function to open the graph and add functionality.

![Interface graph](img/B03950_06_21.jpg)

The reason why **MyInterfaceFunction** appears in the **My Blueprints** tab is because that function contains an output value. If you have an interface function without an output then it won't appear in the **My Blueprints** tab. Instead it appears under **Events** when right-clicking in your Blueprint. For example, in that same interface I created another function without output data.

![Interface graph](img/B03950_06_22.jpg)

This **AnotherInterfaceFunction** will not appear in the **My Blueprints** tab because it has no output. So, to implement this function in your Blueprint, you have to add this as an event.

![Interface graph](img/B03950_06_23.jpg)

# Blueprint node references

The behavior of a Blueprint object is defined using various nodes. Nodes can be **Events**, **Function Calls**, **Flow Control**, **Variables**, and so on that are used in the graph. Even though each type of node has a unique function, the way they are created and used is common.

Nodes are added to the graph by right-clicking inside the graph panel and selecting the node from the **Context Menu**. If a component inside Blueprint is selected, events and functions supported by that component are also listed.

![Blueprint node references](img/B03950_06_24.jpg)

After a node is added you can select it and move it around using the left mouse button. You can use *Ctrl* to add or remove from the current selection of nodes. Clicking and dragging inside the graph creates a **Marquee Selection** that adds to the current selection.

Nodes can have multiple inputs and outputs and are of two types: **Execution Pins** and **Data Pins**.

Execution pins start the flow of execution and when the execution is completed it activates an output execution pin to continue the flow. Execution pins are drawn as outlines when not wired and solid white when connected.

![Blueprint node references](img/B03950_06_25.jpg)

Data pins are nodes that transfer (such as taking and outputting) data from one node to the other. These nodes are type specific. That means they can be connected to variables of the same type. Some data pins are automatically converted if you connect them to another data pin that is not of the same type. For example, if you connect a `float` variable to `string`, the Blueprint editor will automatically insert a `float` to a `string` conversion node. Like execution pins, they are drawn as an outline when not connected, and a solid color when connected.

![Blueprint node references](img/B03950_06_26.jpg)

## Node colors

Nodes in Blueprint have different colors that show what kind of node it is.

A red-colored node means it's an event node and this is where execution starts.

![Node colors](img/B03950_06_27.jpg)

A blue-colored node means it can either be a function or an event being called. These nodes can have multiple inputs or outputs. The icon on top of the function will be changed based on whether it's a function or event.

![Node colors](img/B03950_06_28.jpg)

A purple-colored node can neither be created nor destroyed. You can see this node in **Construction Script** and **Functions**.

![Node colors](img/B03950_06_29.jpg)

A grey node can be a **Macro**, **Flow Control**, or **Collapsed** node.

![Node colors](img/B03950_06_30.jpg)

A green-colored node usually means a Pure function used to get a value.

![Node colors](img/B03950_06_31.jpg)

A cyan-colored node means it's a cast node. This node converts the given object to another.

![Node colors](img/B03950_06_32.jpg)

## Variables

Variables are properties that hold a value or an object reference. They can be accessed inside the Blueprint editor or from another Blueprint. They can be created to include data types (`float`, `integer`, `Boolean`, and so on) or reference types or classes. Each variable can also be an array. All types are color coded for easy identification.

## Math expression

Math expression nodes are essentially collapsed nodes that you can double-click to open the sub graph to see the functionality. Whenever you rename the node, the new expression is parsed and a new graph is generated. To rename the node, simply select it and press *F2*.

To create a **Math Expression** node, right-click on the graph editor and select **Add Math Expression** node. You will then be prompted to type your **Math Expression**.

For example, let's type this expression: *(vector(x, y, z)) + ((a + 1) * (b + 1))* and press *Enter*.

![Math expression](img/B03950_06_33.jpg)

You will now see that the **Math Expression** node has automatically parsed your expression and generated proper variables and a graph from your expression.

![Math expression](img/B03950_06_34.jpg)

The following operators are supported and can be combined with logical and comparison operators to create complex expressions:

*   **Multiplicative**: *, /, % (modulo)
*   **Additive**: +, -
*   **Relational**: <, >, <=, >=
*   **Equality**: == (equal), != (not equal)
*   **Logical**: || (or), && (and), ^ (power)

# Creating our first Blueprint class

Now that we have an idea of what Blueprint is and what it does, let's create a simple Blueprint actor that spins on its own and destroys itself after a few seconds with a particle effect and sound. After creating our Blueprint, we will drag and drop this into the world and we will also use the **Level Blueprint** to dynamically spawn this Blueprint while running the game.

## Creating a new Blueprint

To create this Blueprint, first right-click inside **Content Browser** and select **Blueprint Class**. Once you click that you will be prompted to select a parent class for the Blueprint. You need to specify a parent class for your Blueprint as it will inherit all properties from that parent class.

Even though you can choose all existing classes (even other Blueprint classes), let's take a look at the most common parent classes:

*   **Actor**: An Actor-based Blueprint can be placed or spawned in the level
*   **Pawn**: **Pawn** is what you can call an agent which you can possess and receives inputs from the controller
*   **Character**: This is an extended version of **Pawn** with the ability to walk, run, jump, crouch, and more
*   **Player Controller**: This is used to control the **Character** or **Pawn**
*   **Game Mode**: This defines the game being played
*   **Actor Component**: This is a reusable component that can be added to any actor
*   **Scene Component**: This is a component with scene transform and can be attached to other scene components

In this example, we will use the **Actor** class as our parent because we want to place it in the level and spawn at runtime. So choose **Actor** class and Unreal will create and place your new Blueprint in **Content Browser**. Double-click on your newly created Blueprint and this will open the Blueprint editor. By default, it should open the **Viewport** tab but if it doesn't then simply select the **Viewport** tab. This is where you can see and manipulate all of your components.

Now we need a component that will spin when this Blueprint is spawned. On the **Components** tab, click **Add Component** and select **Static Mesh** component. After you add the component, rename it to **Mesh Component** (you can choose whatever name you want but, for this example, let's choose that name) and note how the **Details** panel has been populated with **Static Mesh** properties.

In the **Details** panel, you can find the section that corresponds to your component type where you can assign the asset to use.

But, in this example, instead of directly assigning a mesh in the **Components** tab, we create a **Static Mesh** variable and use that to assign the mesh in the graph. This way, we can change the mesh without opening the Blueprint editor.

In the **My Blueprints** tab, create a new variable and set the type to **Static Mesh** (make sure to select **reference**).

### Tip

In versions before Unreal Engine 4.9, you can search for **Static Mesh** and simply select the reference. There was no additional options to select before 4.9.

After that, rename that variable to **My Mesh**. Since this variable is used to assign the asset to use with our **Static Mesh** component, let's expose this variable so that we can change it in the **Details** panel after placing it in world. To expose this variable, select it and enable **Editable** in the **Details** panel inside the Blueprint editor. After making it editable, compile the Blueprint (shortcut key: *F7*) and you will be able to assign a default mesh for the **My Mesh** variable. For this example, let's add a simple cube **Static Mesh**.

Now that our variable is set, we can assign it to our **Static Mesh** component. Since we know that **Construction Graph** is executed every time this Blueprint is initialized and whenever a variable or property is changed, that is where we are going to assign the mesh for our **Static Mesh** component. So, open the **Construction Graph** and:

*   Right-click on the graph editor and search for the **Get Mesh** component.
*   Select **Get Mesh** component from the context menu.
*   Click and drag from the output pin and release it. You will now see a new context menu and, in that resulting menu, search for **Set Static Mesh** and select it.
*   Right-click again on graph editor and search for **Get My Mesh**.
*   Select **Get My Mesh** and connect the output pin to the input (**New Mesh**) of the **Set Static Mesh** Blueprint node.
*   And, finally, connect the execution pin of **Construction Script** to **Set Static Mesh Blueprint** node and press **Compile** (shortcut key: *F7*).

If you check the **Viewport** tab after compiling, you will see your new mesh there. From this point, feel free to drag this Blueprint to the world and in the **Details** panel you can change **My Mesh** to any other **Static Mesh**.

### Tip

Press *Ctrl*+*E* to open the associated editor of the object you have selected in world.

## Spinning static mesh

In Blueprint editor, there are a couple of ways to rotate a mesh and in this section we will look into the simplest way, which is using a **Rotate Movement** component.

Open the Blueprint if you have closed it and add a new component called **Rotating Movement**. This component will make this Actor continuously rotate at a given rotation rate optionally around a specified point. This component has three main parameters that can be changed in the Blueprint graph. They are:

*   **Rotation Rate**: The speed at which this will update the **Roll**/**Pitch**/**Yaw** axis.
*   **Pivot Translation**: The pivot point at which we rotate. If set to zero then we rotate around the object's origin.
*   **Rotation in Local Space**: Whether rotation is applied in local space or world space.

You can create two new variables (**Rotator** and **Vector** variables) and make them editable so you can change it in the **Details** panel in world. The final graph should look like this:

![Spinning static mesh](img/B03950_06_35.jpg)

# Destroying our Blueprint Actor after some seconds

Once we place or spawn this Actor in world we will destroy this actor with a particle effect and sound. To do that:

*   Create a new variable (`float`) and name it **DestroyAfter**. Let's give it a default value of five seconds.
*   Go to **Event Graph** and add a new event called **Event BeginPlay**. This node is immediately executed when the game starts or when the actor is spawned in the game.
*   Right-click on the graph editor and search for **Delay** and add it. Connect **Event BeginPlay** to the **Delay** node. This node is used to call an action after a number of specified seconds.
*   The **Delay** node takes a `float` value, which is used for the duration. After the duration runs out, execution is continued to the next action. We will connect our **DestroyAfter** variable to the duration of **Delay**.
*   Right-click on the graph and search for **Spawn Emitter At Location**. This node will spawn the given particle effect at the specified location and rotation. Connect **Delay** to this node and set a particle effect by assigning it in the **Emitter Template**. To set the location, right-click on the graph and search for **GetActorLocation** and connect it to **Location pin**.
*   Right-click on the graph and search for **Spawn Sound At Location**. This node will spawn and play a sound at the given location. Connect **Spawn Emitter** node to this one.
*   And, finally, to destroy this actor, right-click on the graph editor and search for **DestroyActor** and connect it to **Spawn Sound** node.

The final graph should look like this:

![Destroying our Blueprint Actor after some seconds](img/B03950_06_36.jpg)

Now, when you place this actor in world and start the game you will see it spin and, after five seconds (or the value you used in **Destroy After**), this actor will be destroyed after spawning the particle effect and sound.

# Spawning our Blueprint class in Level Blueprint

We will now see how we can spawn this Blueprint Actor in world while the game is running, instead of directly placing when editing.

Before we continue we will make a change to the **DestroyAfter** variable in our spinning Blueprint Actor. Open our spinning actor's Blueprint editor and, in **Variables**, select the **DestroyAfter** variable and, in the **Details** panel, enable the **Expose On Spawn** setting.

![Spawning our Blueprint class in Level Blueprint](img/B03950_06_37.jpg)

This setting means this variable will be exposed in the **Spawn Actor** node.

Open your level and, on the toolbar, click the Blueprints button and select **Open Level** Blueprint. In **Level Blueprint** perform the following steps:

*   Right-click on the graph and search for **Event BeginPlay** and add it.
*   Right-click on the graph and search for **Spawn Actor** from **Class** and add it. This node will spawn the given actor class at the specified location, rotation and scale.
*   In the class pin set the class to our **Rotating Blueprint** Actor. Note how the **Destroy After** variable is now exposed to **Spawn** node. You can now adjust that value from that **Spawn** node.
*   Drag from the **Spawn Transform** node and release the left mouse button. From the resulting context menu, select **Make Transform**. The transform node contains 3D transformation including translation, rotation, and scale. For this example, let's set the **Location** to **0,0,300** so that is this Actor will be spawned 300 units above the ground.

The resulting graph should look like this:

![Spawning our Blueprint class in Level Blueprint](img/B03950_06_38.jpg)

If you play (*Alt*+*P*) or simulate (*Alt*+*S*) you will see this rotating Actor spawn **300** units above the ground and spinning.

# Summary

In this chapter, we have learned what components are and how we can use them to define a Blueprint Actor. We also learned about Blueprint nodes and how you can create them. From what you have learned in this chapter, you can take it even further by:

*   Spawning this actor when overlapping a trigger volume placed in the level
*   Playing a particle and sound effect when spawning this Blueprint
*   Applying damage to a player if the player is in a certain radius

In the next chapter, we will use Matinee to create a cut scene.