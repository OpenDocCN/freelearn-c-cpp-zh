# Chapter 2. Scripting and Data in Unreal

Now that we've got a design to work from, we can begin to develop the game.

Before we can do that, however, we'll be exploring the variety of ways in which we can work with game code and game data in the Unreal game engine.

This chapter will walk you through the steps necessary to get Unreal and Visual Studio installed, and to create a new Unreal Engine project. Additionally, you will learn how to create new C++ game code, work with Blueprints and Blueprint graphs, and work with custom data for your game. In this chapter, we will cover the following topics:

*   Downloading Unreal
*   Setting up Visual Studio for use with Unreal
*   Setting up a new Unreal project
*   Creating new C++ classes
*   Creating Blueprints and Blueprint graphs
*   Using Data Tables to import spreadsheet data

# Downloading Unreal

Before getting started, make sure that you have at least 18 GB of free disk space on your computer. You will need this disk space to hold the development environments for Unreal and also your project files.

We will now need to download Unreal. To do this, go to [https://www.unrealengine.com](https://www.unrealengine.com) and click on the **GET UNREAL** button.

Before you can download Unreal, you'll need to make an Epic Games account. The **GET UNREAL** button will redirect you to an account creation form, so fill it out and submit it.

After you've signed in, you'll see the **Download** button. This will download the installer for the Epic Games Launcher (from this launcher, you can download Unreal version 4.12).

# Downloading Visual Studio

We will need to start programming soon, so if you haven't already, now is the time to download Visual Studio, which is the integrated development environment that we will need to program the framework for our engine and game logic in C++. Luckily, Microsoft provides Visual Studio Community for free.

To download Visual Studio Community, go to [https://www.visualstudio.com/](https://www.visualstudio.com/) and download Community 2015\. This will download the installer for Visual Studio. After it downloads, simply run the installer. Note that Visual Studio Community 2015 does not install C++ by default, so be sure that under **Features**, you are installing Visual C++, Common Tools for Visual C++ 2015, and Microsoft Foundation Classes for C++. If you do not have C++ installed, you will not be able to write or compile code written for UE4 in Visual Studio since UE4 is built on C++.

# Setting up Visual Studio for Unreal

After you've installed Visual Studio, there are some steps you can take to make it easier to work with C++ code in Unreal. These steps are not at all necessary, and can be safely skipped.

## Adding the Solution Platforms drop-down list

On the right-hand side of the toolbar is a drop-down arrow, as shown in the following screenshot:

![Adding the Solution Platforms drop-down list](img/B04548_02_01.jpg)

Click on this button, hover over the **Add** or **Remove** buttons, and click on **Solution Platforms** to add the menu to the toolbar.

The **Solution Platforms** drop-down list allows you to switch the project between target platforms (for instance, Windows, Mac, and so on).

## Disabling the Error List tab

The error list in Visual Studio shows you the errors that it detects in your code before you even compile the project. While normally this is incredibly useful, in Unreal, it can frequently detect false positives and become more annoying than helpful.

To disable the error list, first close the **Error List** tab (you can find this in the bottom pane, as shown in the following screenshot):

![Disabling the Error List tab](img/B04548_02_10.jpg)

Then, navigate to **Tools** | **Options**, expand the **Projects and Solutions** group, and uncheck the **Always show Error List if build finishes with errors** option:

![Disabling the Error List tab](img/B04548_02_11.jpg)

# Setting up a new Unreal project

Now that you have both Unreal and Visual Studio downloaded and installed, we're going to create a project for our game.

Unreal comes with a variety of starter kits that you can use, but for our game, we'll be scripting everything from scratch.

After signing into Epic Games Launcher, you'll first want to download the Unreal Engine. This book uses version 4.12\. You may use a later version, but depending on the version, some code and the navigation of the engine may slightly differ. The steps for creating a new project are as follows:

1.  Firstly, in the **Unreal Engine** tab, select **Library**. Then, under **Engine Versions**, click on **Add Versions** and select the version you'd like to download.
2.  After the engine has downloaded, click on the **Launch** button.
3.  Once the Unreal Engine has launched, click on the **New Project** tab. Then, click on the **C++** tab and select **Basic Code**.
4.  Finally, choose a location for your project and give it a name (in my case, I named the project `RPG`).

In my case, after the project was created, it automatically closed the engine and opened Visual Studio. At this point, I've found it best to close Visual Studio, go back to Epic Games Launcher, and relaunch the engine. Then, open your new project from here. Finally, after the editor has launched, go to **File** | **Open Visual Studio**.

The reason for this is because, while you can launch the editor by compiling your Visual Studio project, in some rare cases you may have to close the editor any time you want to compile a new change. If, on the other hand, you launch Visual Studio from the editor (rather than the other way around), you can make a change in Visual Studio and then compile the code from within the editor.

At this point, you have an empty Unreal project and Visual Studio that are ready to go.

# Creating a new C++ class

We're now going to create a new C++ class with the following steps:

1.  To do this, from the Unreal editor, click on **File** | **New C++ Class**. We'll be creating an Actor class, so select **Actor** as the base class. Actors are the objects that are placed in the scene (anything from meshes, to lights, to sounds, and more).
2.  Next, enter a name for your new class, such as `MyNewActor`. Hit **Create Class**. After it adds the files to the project, open `MyNewActor.h` in Visual Studio. When you create a new class using this interface, it will generate both a header file and a source file for your class.
3.  Let's just make our actor print a message to the output log when we start our game. To do this, we'll use the `BeginPlay` event. `BeginPlay` is called once the game has started (in a multiplayer game, this might be called after an initial countdown, but in our case, it will be called immediately).
4.  The `MyNewActor.h` file (which should already be open at this point) should contain the following code after the `GENERATED_BODY()` line:

    [PRE0]

5.  Then, in `MyNewActor.cpp`, add a log that prints **Hello, world!** in the `void AnyNewActor::BeginPlay()` function, which runs as soon as the game starts:

    [PRE1]

6.  Then, switch back to the editor and click on the **Compile** button in the main toolbar.
7.  Now that your actor class has compiled, we need to add it to the scene. To do this, navigate to the **Content Browser** tab located at the bottom of the screen. Search for `MyNewActor` (there's a search bar to help you find it) and drag it into the scene view, which is the level viewport. It's invisible, so you won't see it or be able to click on it. However, if you scroll the **Scene/World Outliner** pane (on the right-hand side) to the bottom, you should see the **MyNewActor1** actor has been added to the scene:![Creating a new C++ class](img/B04548_02_02.jpg)
8.  To test your new actor class, click on the **Play** button. You should see a yellow **Hello, world!** message printed to the console, as shown in the following screenshot. This can be seen in the **Output Log** tab on the right-hand side of the **Content Browser** tab at the bottom of the screen:![Creating a new C++ class](img/B04548_02_03.jpg)

Congratulations, you have created your first actor class in Unreal.

# Blueprints

Blueprints in Unreal is a C++ based visual scripting language built proprietary to Unreal. Blueprints will allow us to create code without the need to touch a line of text in an IDE such as Visual Studio. Instead, Blueprints allows us to create code through the use of drag and drop visual nodes, and connect them together to create nearly any kind of functionality you desire. Those of you who have come from UDK may find some similarity between Kismet and Blueprints, but unlike Kismet, Blueprints allows you to have full control over the creation and modification of functions and variables. It also compiles, which is something Kismet did not do.

Blueprints can inherit from C++ classes, or from other Blueprints. So, for instance, you might have an `Enemy` class. An enemy might have a **Health** field, a **Speed** field, an **Attack** field, and a **Mesh** field. You can then create multiple enemy templates by creating Blueprints that inherit from your `Enemy` class and changing each type of enemy's Health, Speed, Attack, and Mesh.

You can also expose parts of your C++ code to Blueprint graphs so that your Blueprint graphs and your core game code can communicate and work with each other. As an example, your inventory code may be implemented in C++, and it might expose functions to Blueprints so that a Blueprint graph can give items to the player.

## Creating a new Blueprint

The steps to create a new Blueprint are as follows:

1.  In the **Content Browser** pane, create a new Blueprint folder by clicking on the **Add New** drop-down list and selecting **New Folder**, then renaming the folder `Blueprint`. Inside this folder, right-click and select **Blueprints** | **Blueprint Class**. Select **Actor** as the parent class for the Blueprint.
2.  Next, give a name to your new Blueprint, such as `MyNewBlueprint`. To edit this Blueprint, double-click on its icon in the **Content Browser** tab.
3.  Next, switch to the **Event Graph** tab.
4.  If **Event Begin Play** is not already there, right-click on the graph and expand **Add Event**; then, click on **Event Begin Play**. If you ever need to move around nodes such as **Event Begin Play**, you can simply left-click on the node and drag it anywhere on the graph you want. You can also navigate through the graph by holding down the right-click mouse button and dragging across the screen:![Creating a new Blueprint](img/B04548_02_05.jpg)

    This will add a new event node to the graph.

5.  Next, right-click and begin typing `print` into the search bar. You should see the **Print String** option in the list. Click on it to add a new **Print String** node to your graph.
6.  Next, we want to have this node triggered when the **Event Begin Play** node is triggered. To do this, drag from the output arrow of the **Event Begin Play** node to the input arrow of the **Print String** node:![Creating a new Blueprint](img/B04548_02_06.jpg)
7.  Now, the **Print String** node will be triggered when the game begins. However, let's take this one step further and add a variable to our Blueprint.
8.  On the left-hand side in the **My Blueprint** pane, click on the **Variable** button. Give a name to your variable (such as `MyPrintString`) and change the **Variable Type** drop-down list to **String**.
9.  To feed the value of this variable into our **Print String** node, right-click and search for `MyPrintString`. You should see a **Get My Print String** node available in the list. Click on this to add the node to your graph:![Creating a new Blueprint](img/B04548_02_07.jpg)
10.  Next, just as you did to connect **Event Begin Play** and **Print String** together, drag from the output arrow of the **Get My Print String** node to the input pin of the **Print String** node that is right next to the **In String** label.
11.  Finally, switch over to the **Defaults** tab. At the very top, under the **Defaults** section, there should be a text field for editing the value of the `MyPrintString` variable. Enter whatever text you'd like into this field. Then, to save your Blueprint, first press the **Compile** button in the **Blueprint** window and then click on the **Save** button next to it.

## Adding a Blueprint to the scene

Now that you've created the Blueprint, simply drag it from the **Content Browser** tab into the scene. Just as with our custom actor class, it will be invisible, but if you scroll **Scene Outliner** to the bottom, you'll see the **MyNewBlueprint** item in the list.

To test our new Blueprint, press the **Play** button. You should see that the text you entered is briefly printed to the screen (it will also show up in the **Output Log**, but it may be difficult to spot amidst the other output messages).

## Blueprints for Actor classes

You can choose other classes for a Blueprint to inherit from. For instance, let's create a new Blueprint to inherit from the custom `MyNewActor` class we created earlier:

1.  To do this, start creating a new Blueprint as before. Then, when choosing a parent class, search for `MyNewActor`. Click on the **MyNewActor** entry in the list:![Blueprints for Actor classes](img/B04548_02_12.jpg)
2.  You can name this Actor whatever you want. Next, open the Blueprint and click on **Save**. Now, add the Blueprint to your scene and run the game. You should now have two **Hello, world!** messages logged to the console (one from our placed actor and the other from our new Blueprint).

# Using Data Tables to import spreadsheet data

In Unreal, **Data Tables** are a method of importing and using custom game data exported from a spreadsheet application. To do this, you first ensure that your spreadsheet follows some guidelines for format; additionally, you write a C++ struct that contains the data for one row of the spreadsheet. Then, you export a CSV file and select your C++ struct as the data type for that file.

## The spreadsheet format

Your spreadsheet must follow some simple rules in order to correctly export to Unreal.

The very first cell must remain blank. After this, the first row will contain the names of the fields. These will be the same as the variable names in your C++ struct later, so do not use spaces or other special characters.

The first column will contain the **lookup key** for each entry. That is, if the first cell of the first item in this spreadsheet is 1, then in Unreal, you would use 1 to find that entry. This must be unique for every row.

Then, the following columns contain the values for each variable.

## A sample spreadsheet

Let's create a simple spreadsheet to import into Unreal. It should look like this:

![A sample spreadsheet](img/B04548_02_08.jpg)

As mentioned in the previous section:

*   Column **A** contains the lookup keys for each row. The first cell is empty and the following cells have the lookup keys for each row.
*   Column **B** contains the values for the `SomeNumber` field. The first cell contains the field name (`SomeNumber`) and the following cells contain the values for that field.
*   Column **C** contains the values for the `SomeString` field. Just as with column **B**, the first cell contains the name of the field (`SomeString`) and the following cells contain the values for that field.

I'm using Google Spreadsheets—with this, you would click on **File** | **Download as** | **Comma-separated values (.csv, current sheet)** to export this to CSV. Most spreadsheet applications have the ability to export to the CSV format.

At this point, you have a CSV file that can be imported into Unreal. However, do not import it yet. Before we do that, we'll need to create the C++ struct for it.

## The Data Table struct

Just as you created the actor class earlier, let's create a new class. Choose `Actor` as the parent class and give it a name such as `TestCustomData`. Our class won't actually inherit from `Actor` (and, for that matter, it won't be a class), but doing this allows Unreal to generate some code in the background for us.

Next, open the `TestCustomData.h` file and replace the entire file with the following code:

[PRE2]

Notice how the variable names are exactly the same as the header cells in the spreadsheet—this is important, as it shows how Unreal matches columns in the spreadsheet to the fields of this struct.

Next, remove everything from the `TestCustomData.cpp` file, with the exception of the `#include` statements.

Now, switch back to the editor and click on **Compile**. It should compile without any issues.

Now that you've created the struct, it's time to import your custom spreadsheet.

## Importing the spreadsheet

Next, simply drag your CSV file into the **Content Browser** tab. This will show a pop-up window, asking you to pick how you want to import the data and also what type of data it is. Leave the first drop-down list to **Data Table** and expand the second drop-down list to pick **TestCustomData** (the struct you just created).

Click on **OK** and it will import the CSV file. If you double-click the asset in the **Content Browser** tab, you'll see a list of the items that were in the spreadsheet:

![Importing the spreadsheet](img/B04548_02_09.jpg)

## Querying the spreadsheet

You can query the spreadsheet in order to find particular rows by name.

We'll add this to our custom actor class, `MyNewActor`. The first thing we need to do is expose a field to a Blueprint, allowing us to assign a Data Table for our actor to use.

Firstly, add the following code just after the `GENERATED_BODY` line:

[PRE3]

The preceding code will expose the Data Table to Blueprint and allow it to be edited within Blueprint. Next, we'll fetch the first row and log its `SomeString` field. In the `MyNewActor.cpp` file, add this code to the end of the `BeginPlay` function:

[PRE4]

You will also need to add `#include TestCustomData.h` at the top of your `MyNewActor.cpp` file so that you can see the Data Table properties in it.

Compile the code in the editor. Next, open up the Blueprint you created from this actor class. Switch to the **Class Defaults** tab and find the **My New Actor** group (this should be at the very top). This should show a **Data Table** field that you can expand to pick the CSV file you imported.

Compile and save the Blueprint and then press **Play**. You should see the value of `SomeString` for the entry `2` logged to the console.

# Summary

In this chapter, we set up Unreal and Visual Studio and created a new project. Additionally, we learned how to create new actor classes in C++, what a Blueprint is, and how to create and use Blueprint graphs for visual scripting. Finally, we learned how to import custom data from spreadsheet applications and query them from the game code.

In the next chapter, we'll start diving into some actual gameplay code and start prototyping our game.