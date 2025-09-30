# Chapter 11. Packaging Project

Throughout this book, you learned the basics of Unreal Engine 4\. In this final chapter, we will recap all that, as well as see how to package your project into a standalone game. You will also learn how to package the game for quick distribution and package a game as a release version.

# Recap

In the first chapter, you learned the difference between Unreal Engine versions. As I have mentioned, the launcher version is a binary version compiled by Epic and is ready for you to use. But, if you want to get the latest build that is not yet available through launcher, then your only choice is getting the source code from GitHub. If you are going for the source code version of Unreal Engine then I recommend getting the source from the promoted branch. Epic works hard on the promoted build for their artists and designers, so most of the time it is updated daily and you get the latest stuff too! if you really want to get your hands dirty or you have that urge to grab the latest and the most cutting-edge build, then you should go for the master branch. Keep in mind that this branch tracks live changes directly from Epic, it might be buggy and it might even fail to compile.

Once you get the engine up and running, you can start importing your assets into **Content Browser**. This is where you save and edit the assets that are used in your game. **Content Browser** offers a lot of functionality such as searching based on keyword, tags, asset type, filters, etc. and you can use the **Collections** feature in **Content Browser** to add references to your most commonly used assets. When searching, you can exclude specific keywords by adding the hyphen (-) before the name. For example, if you want to exclude all assets that contain the name `floor`, then you can search in **Content Browser** as `-floor`. This will show you all assets that do not contain the word floor.

Another great feature of **Content Browser** is the **Developers** folder. This is especially useful when you are working in a team where you want to try out different techniques or assets in your game without affecting other parts. One thing to remember is that you should only use this strictly for personal or experimental work and you should never include references to external assets outside this folder. For example, if you made an asset that you want to try out before adding it to the game, then you can create a test level inside your **Developers** folder and test out everything there. Think of the **Developers** folder as your own private playground where you can do whatever you want without affecting others work. The **Developers** folder is not enabled by default. To enable it, click on **View Options** at the bottom right corner of your **Content Browser** and select **Show Developers Folder**:

![Recap](img/B03950_11_01.jpg)

Once you enable that, you will see a new folder called **Developers** under your **Content** folder in **Content Browser**:

![Recap](img/B03950_11_02.jpg)

The name of the folder inside the **Developers** folder is automatically set to your Windows username. If you are using **Source Control** (for example, Perforce or Subversion), then you can see the `Other` **Developers** folder by enabling the **Other Developers** checkbox available under **Filters** | **Other Filters**:

![Recap](img/B03950_11_03.jpg)

Knowing this will help you a lot when you are working with a team or when you have lots of assets.

Just like how you use **Content Browser** to find assets that are imported, you use **World Outliner** to find assets that are placed in your level. You can also use **Layers** to organize assets that are placed in the level. Both of these windows can be summoned from **Window** in the menu bar:

![Recap](img/B03950_11_04.jpg)

In [Chapter 3](ch03.html "Chapter 3. Materials"), *Materials*, you learned about the awesome **Material Editor** and the common nodes that we will use. A good material artist can totally change the realism of your game. Mainly materials and post processing gives you the power to make the game look realistic or cartoony. The common material expressions that we learned are not just used for coloring your assets. For example, create the following material network and apply to a simple mesh (for example, a sphere) and see what happens:

![Recap](img/B03950_11_05.jpg)

If you find yourself using a specific network multiple times, then it's better for you to create a material function which can tidy up your graph and make it more organized.

As you continue developing your game, you will eventually start tweaking with **Post Process Volume**. This lets you modify the overall look and feel of your game. By combining **Post Process** in blueprints or C++ you can even use it to affect your game play too. A perfect example for this is the detective vision from the Batman Arkham series games. You can use materials in post process to highlight a specific object in world or even use it to render outlines for meshes that are behind other objects.

Another crucial part of the game that determines the final look is lighting. In this book, you learned about different light mobilities, the differences between them including common light settings and how it affects the game world. You also learned about Lightmass Global Illumination which is the static global illumination solver developed by Epic Games.

As you know by now, Lightmass is used to bake lighting and because of that, dynamic lights are not supported by Lightmass. When using Lightmass for your game, you need to make sure that you have a second UV channel for all your static meshes (that are not set to movable) to have proper shadows. If you want to use dynamic lights (that means lights that can change any of their properties at runtime-think of the day and night cycle as an example), Epic has included support for **Light Propagation Volume** (**LPV**). At the time of writing this book, LPV is in experimental stage and is not yet ready for production. One extra thing that is worth mentioning here is the ability to change bounced lighting color. Take a look at the following material network:

![Recap](img/B03950_11_06.jpg)

Using the **GIReplace** material node, you can change the color of the bounced light. If you apply the preceding material to a mesh and use Lightmass to build lighting, the result of the bounced light will be red color instead of white. Even though we don't need to have a different color for bounced lights, we can still use this node to darken or brighten the bounced lighting without the need to adjust Lightmass settings.

Once we have all the base setups, we then jump to Blueprints. **Blueprint Visual Scripting** is a powerful and flexible node-based editor that lets artists and designers quickly prototype their game. Mainly, we work with two common Blueprint types and they are **Level Blueprint** and **Class Blueprint**. Inside these Blueprints, we have Event Graph, Function Graph, and Macro Graph. In **Class Blueprints**, we add components to define what that Blueprint is and how they behave. Nodes in Blueprint have various colors applied to them to indicate what kind of node they are. Once you start using Blueprints, you will get familiar with all the node colors and what they mean. We saw how to create a **Class Blueprint** from an `Actor` class and how to spawn it dynamically in the game. We also saw how we can interact with objects in world through **Level Blueprint**. We placed triggers in the level and in **Level Blueprint** we created overlap events for these triggers and learned how to play a Matinee sequence.

Matinee is one of the powerful tools in Unreal Engine 4 that is mainly used to create cinematics. You learned about Matinee UI and how to create a basic cut scene. Since Matinee is similar to other nonlinear video editors, it is easy for video editing professionals to get familiar with Matinee. Even though Matinee is used for cinematics, you can also use it for gameplay-related elements such as opening doors, elevator movement etc.. You can even use it to export your existing cinematics as image sequences or in the AVI format.

After learning about Matinee, we continued to the next chapter to learn about **Unreal Motion Graphics** (**UMG**). UMG is a UI authoring tool developed by Epic. Using UMG, we created a simple HUD for the player and learned how to communicate with the player Blueprint to show a health bar for the player. We also made a 3D widget for the player that floats on top of the character's head.

Continuing from there, you learned more about the Cascade Particle System. You learned about Particle Editor and various other windows available inside Cascade Editor. After learning the basics, you created a basic particle system using GPU Sprites including collision. Lastly, we took the particle system to Blueprints and learned how to randomly burst the particles using Custom events and delay node.

Finally, we dived into the magic world of C++. There you learned about various versions of Visual Studio 2015 and how to download Visual Studio 2015 Community Edition. Once we have the IDE installed, we created a new C++ project based on the Third Person template. From there we extended it to include health and health regeneration for our character class. You also learned how to expose variables and functions to Blueprints and how to access them in Blueprints.

# Packaging the project

Now that you have learned most of the basics of Unreal Engine 4, let's see how to package your game. Before we package the game, we need to make sure that we set a default map for our game which will be loaded when your packaged game starts. You can set the **Game Default Map** option from the **Project Settings** window. For example, you can set the **Game Default Map** option to your main menu map:

![Packaging the project](img/B03950_11_07.jpg)

To set a default map for the game, please follow these steps:

1.  Click on the **Edit** menu.
2.  Click on **Project Settings**.
3.  Select **Maps & Modes**.
4.  Choose your new map in **Game Default Map**.

## Quick packaging

Once you set the **Game Default Map** option, you need to select the **Build Configuration**:

![Quick packaging](img/B03950_11_08.jpg)

There are three types of build configurations available Packaging the project:

*   `DebugGame`: This configuration will include all the debug information. For testing purposes, you can use this configuration.
*   `Development`: This configuration offers better performance compared to the `DebugGame` configuration build because of minimal debugging support.
*   `Shipping`: This should be the setting you should choose when you want to distribute the game.

Once you have selected your build configuration, you can package your game from **File** | **Package Project** and then select your platform. For example, here is the option to package your game for **Windows 64-bit**:

![Quick packaging](img/B03950_11_09.jpg)

Once you select that option, editor might prompt you to select a target directory to save the packaged game. Once you set the path, editor will start building and cooking the content for the selected platform. If the packaging is successful, you will see the packaged game under the target directory you set earlier.

## Packaging the release version

The previously mentioned method is for quickly packaging and distributing the game to end users. However the preceding method cannot build DLCs or patches for your game so in this section, you will learn how to create a release version for your game.

To start let's first open the **Project Launcher** window. **Project Launcher** provides advanced workflows to packaging your game:

![Packaging the release version](img/B03950_11_10.jpg)

To create a custom launch profile, click on the plus (**+**) button as shown in the preceding screenshot. Once you click on that you will see a new window with new settings as follows:

![Packaging the release version](img/B03950_11_11.jpg)

In the preceding window, do the following:

1.  Enable the **Build** checkbox.
2.  Set the **Build Configuration** option to **Shipping**.
3.  Set the dropdown to **By the book**.
4.  In this example we selected **WindowsNoEditor** to test on Windows.
5.  Select the culture. This is used for localization. By default, **en-US** is selected.

Once all those settings are done, expand the **Release/DLC/Patching Settings** and **Advanced Settings** sections:

![Packaging the release version](img/B03950_11_12.jpg)

Inside those sections do the following:

1.  Enable **Create a release version of the game for distribution**.
2.  Set the name of the new release to **1.0**.
3.  Enable **Store all content in a single file (UnrealPak)**.
4.  Set the **Cooker build** **configuration** section to **Shipping**.
5.  Add the `–stage` command line as **Additional Cooker Option**. Note that you do not press enter after typing it. Simply click anywhere else to apply that command.

After setting this, set the last two options of **Package** and **Deploy** to **Do not package** and **Do not deploy** respectively:

![Packaging the release version](img/B03950_11_13.jpg)

Once all those are done, click on the **Back** button on the top right corner of the **Project Launcher** window and you will see your new profile ready to build:

![Packaging the release version](img/B03950_11_14.jpg)

Simply click on the **Launch** button, **ProjectLauncher** will build, cook, and package your game. This might take time depending on the complexity of your game:

![Packaging the release version](img/B03950_11_15.jpg)

If the packaging was successful, then you can see that in the **ProjectLauncher** window:

![Packaging the release version](img/B03950_11_16.jpg)

You can find your packaged game in your project folder under **Saved** | **StagedBuilds** | **WindowsNoEditor** folder. Now you can distribute this packaged game to other users.

# Summary

Throughout this book, you learned the basics of Unreal Engine 4\. We started this journey with you learning how to download the engine and saw how to import your own assets. From there you learned about Material Editor and its common aspects. Then you learned about Post Process, how to use lights and the importance of lights in video games. You also learned about Blueprints which is the visual scripting language of Unreal Engine 4\. We continued our journey from Blueprints to UMG which you can use to create any kind of menu in the game. Since a game is nothing without visual effects and cut scenes, you learned about Cascade Particle Editor and Matinee. From there we dived into the world of C++ to learn the basics of this awesome language. Finally you learned how to package the game and distribute it to others.

# References

Your journey of learning Unreal Engine 4 does not stop here. You can extend your knowledge even further by visiting these links:

*   *Unreal Engine* *Community*

    [https://forums.unrealengine.com/](https://forums.unrealengine.com/)

*   *Unreal Engine* *Official Twitch Streams*

    [http://www.twitch.tv/unrealengine](http://www.twitch.tv/unrealengine)

*   *Unreal Engine* *YouTube channel*

    [https://www.youtube.com/user/UnrealDevelopmentKit/videos](https://www.youtube.com/user/UnrealDevelopmentKit/videos)

*   *Unreal Engine* *AnswerHub*

    [https://answers.unrealengine.com/index.html](https://answers.unrealengine.com/index.html)

*   *Unreal Engine* *Documentation*

    [https://docs.unrealengine.com/latest/INT/GettingStarted/index.html](https://docs.unrealengine.com/latest/INT/GettingStarted/index.html)