# Chapter 4. Pause Menu Framework

At this point, we have created a basic combat engine for our game. We can now dive into out-of-combat operations such as the creation of a pause menu screen, where we will be able to view player stats and edit inventory.

In this chapter, we will create the first part of our menu system, which is to design and create a framework for a pause menu. We will cover the following topics in this chapter:

*   UMG pause screen initial setup
*   UMG background color
*   UMG text
*   UMG buttons
*   UMG inventory submenu
*   UMG equipment submenu
*   Key binding
*   Button programming

# UMG pause screen initial setup

For our pause screen, we will need to think about quite a few things regarding the design. As listed in the earlier chapter, the pause screen will give the ability to the player to view party members, equip and unequip equipment, use items, and so on. So we must design our pause screen with that sort of functionality in mind.

To design the pause screen, we will be using **Unreal Motion Graphics** (**UMG**), which is a separate portion of UE4 that allows us to design virtual user interfaces without the need for programs such as Adobe Flash. UMG is very intuitive and does not require programming knowledge in order to use it.

To start with, we will navigate to our already created **Blueprints** | **UI** folder and create a Widget Blueprint for our pause menu. To do this, right-click on your **UI** folder and then navigate to **User Interface** | **Widget Blueprint**:

![UMG pause screen initial setup](img/B04548_04_01.jpg)

Name the Widget Blueprint as `Pause`:

![UMG pause screen initial setup](img/B04548_04_02.jpg)

The Widget Blueprint will allow you to use UMG to design any user interface; we will be using this widget to design our own UI for the pause menu.

To start designing the pause menu, open the **Pause** Widget Blueprint by double-clicking on it from within **Content Browser**. You should see the **Designer** screen that looks like the following screenshot:

![UMG pause screen initial setup](img/B04548_04_03.jpg)

We are first going to create an area for our first screen where we want the pause menu to be. We will first be adding a Canvas Panel that acts as a container to allow several widgets to be laid out within it. This is a great place to start because we will need to feature several navigation points, which we will design in the form of buttons within our pause screen. To add a Canvas Panel, navigate to **Palette** | **Panel** | **Canvas Panel**. Then, drag the Canvas Panel into your **Designer** viewport (note that if you already have a Canvas Panel in your **Hierarchy** tab by default, you can skip this step):

![UMG pause screen initial setup](img/B04548_04_05.jpg)

You should see a few new things in your pause menu. Firstly, you will see that under the **Hierarchy** tab, there is now **CanvasPanel** in **Root**. This means that in the root of the pause screen is the Canvas Panel that we just added. You will also notice that your Canvas Panel, while selected, contains details that can be seen in the **Details** tab. The **Details** tab will allow you to edit properties of any selected item. We will be using these areas of Widget Blueprint frequently throughout our development process.

We now need to think about what sorts of navigation points and information we need on our screen when the player presses the pause button. Based on the functionality, the following are the items we will need to lay out on our first screen:

*   Characters along with their stats (level, HP, MP, and experience/next level)
*   The **Inventory** button
*   The **Equipment** button
*   The **Exit** button
*   Gold

# UMG background color

Before we begin creating texts and buttons for our menu, we should first make a background color that will be laid behind the texts and buttons of the pause screen. To do this, navigate to **Palette** | **Common** | **Image**. Then, drag and drop the image onto the Canvas Panel so that the image is within the Canvas Panel. From here, locate the **Anchors** drop-down menu under **Details** | **Slots**. Select the **Anchors** option that creates anchors on all the four corners of the canvas.

This is an icon that looks like a large square covering the entire canvas located on the bottom-right of the **Anchors** drop-down menu:

![UMG background color](img/B04548_04_07.jpg)

Once this is done, set the **Offset Right** and **Offset Bottom** values to 0\. This will ensure that, just like the left and the top of the image, the right and the bottom of the image will start at 0, thus, allowing the image to stretch to all our anchor points that are positioned at all four corners of our canvas:

![UMG background color](img/B04548_04_08.jpg)

To make the background image a little easier on the eyes, we should make it a bit more of a dull color. To adjust the color, navigate to **Details** | **Appearance** | **Color and Opacity** and then click on the rectangular box next to it. This will open a **Color Picker** box where we can pick any color we want. In our example, we will use a dull blue:

![UMG background color](img/B04548_04_10.jpg)

Press **OK** once you are finished. You will notice that your image name is something like **Image_###**; we should adjust this so that it is more descriptive. To rename the image, simply navigate to **Details** and change the name. We will change the name to **BG_Color**. Lastly, change the **ZOrder** value in **Details** | **Slot** to **-1**. This will ensure that the background is drawn behind other widgets:

![UMG background color](img/B04548_04_12.jpg)

# UMG text

Now that we have finished creating a background for our menu, it is time to lay out our text and navigation. We will add text by navigating to **Common** | **Text**. Then, we will drag and drop the text into the Canvas Panel located in the **Hierarchy** tab:

![UMG text](img/B04548_04_14.jpg)

Note down a few important details. Firstly, you will see that in the **Hierarchy** tab, the Text Block is located within the Canvas Panel. This means that the Canvas Panel is acting as a container for the text; thus, it can only be seen if the player is navigating through the Canvas Panel. You will also notice that the **Details** tab has changed in order to include specific properties for the text. Some really important details are listed here, such as position, size, text, and anchors. Lastly, you should see that the selected text is in the form of a movable and resizable text box, which means we can place and edit this however we choose to. For now, we won't worry about making the pause screen look pretty, we just need to focus on the layout and functionality. A common layout will be one that navigates the eyes left to right and top to bottom. Since we are starting with the characters, we will make our first text be the character names. Also, we will have them start at the top-left corner of the pause menu.

Firstly, we will add the text necessary for the character names or classes. While selecting the text, navigate to **Details** | **Content** | **Text** and type the name of the first class—`Soldier`. You will notice that the text that you wrote in the **Content** tab is now written in your Text Block in the **Designer** view. However, it is small, which makes the text hard to see. Change its size by navigating to **Details** | **Appearance** | **Font**. Here, you can change the size to something larger such as **48**:

![UMG text](img/B04548_04_15.jpg)

Position the text by navigating to **Details** | **Slot** and moving the text so that it is located on the top-left corner, but give it some padding. In our example, we will set **Position X** to **100** and **Position Y** to **100** so that there is a 100-pixel padding for **Soldier**. Lastly, we will rename the text as **Header_Soldier**:

![UMG text](img/B04548_04_17.jpg)

You will notice that the font size does not change and this is because you must press **Compile** at the top-left corner of the window. You will need to press **Compile** whenever you make technical changes such as these in your **Design** view. Once compiled, you should see that your font is resized. However, the text is too large for the Text Block. You can fix this by simply checking **Size to Content**, which is located in **Details** | **Slot**:

![UMG text](img/B04548_04_19.jpg)

Now that we have created the header for our first character, we can continue to create more texts for its stats. We will start by creating a font for HP. To do so, you will need another Text Block on your canvas:

![UMG text](img/B04548_04_20.jpg)

From here, you can position your Text Block so that it is somewhere below the **Header_Soldier** text. In this example, we will place it at a **Position X** value of **200** and a **Position Y** value of **200**:

![UMG text](img/B04548_04_21.jpg)

We will then write content for the text; in this case, the content will be **HP**. Here, we will give a font size of **32** to the text and compile it; then, we will check **Size to Content**:

![UMG text](img/B04548_04_23.jpg)

Lastly, we will name this widget **Menu_HP**:

![UMG text](img/B04548_04_24.jpg)

As you can see, all we did was added text that says **HP** in the menu; however, you will also need actual numbers for the HP displayed on the screen. For now, we are simply going to make a blank Text Block on the right-hand side of the HP text. Later on in this chapter, we will tie this in with the code we created for the character HP in the previous chapter. So for now, drag and drop a Text Block as a child of your Canvas Panel:

![UMG text](img/B04548_04_25.jpg)

Rename it as **Editable_Soldier_HP**. Then, position it so that it is to the right of **Menu_HP**. In this case, we can set the **Position X** value as **300** and the **Position Y** value as **200**:

![UMG text](img/B04548_04_27.jpg)

Lastly, we can change the font style to **Regular**, the font size to **32**, check **Size to Content**, and compile:

![UMG text](img/B04548_04_28.jpg)

Now that you know what the layout is like and how we can create text blocks for characters and their stats, you can proceed to create the other necessary stats for your character such as level, MP, and experience/next level. After you have finished laying out characters and their stats, your final result should look something like the following:

![UMG text](img/B04548_04_29.jpg)

At this point, you may also move on to creating more characters. For instance, if you wanted to create a Healer, you could have easily copied most of the content we created for the Soldier and its layout in our pause screen. Your pause screen with placeholders for both Soldier and Healer stats may look something like the following:

![UMG text](img/B04548_04_30.jpg)

The last placeholder we need to make on this screen is for gold that is collected by the player. Just like we did for the party stats, create a Text Block and make sure the content of the text is **Gold**. Place this somewhere away from the character stats, for example, in the bottom-left corner of the pause screen. Then, rename the Text Block as **Menu_Gold**. Finally, create a second Text Block, place it to the right of **Menu_Gold**, and call it **Editable_Gold**:

![UMG text](img/B04548_04_32.jpg)

Like the empty text boxes in the character stats, we will link **Editable_Gold** with the gold accumulated in the game later on.

We can now move on to creating buttons for our menu, which will eventually navigate to submenus.

# UMG buttons

So far, we have created the first screen of our pause menu that includes all of our characters and placeholders for their stats and gold. The next thing we need to design is buttons, which will be the last portion of our first pause screen. Much like buttons in other software packages, they are typically used to trigger events built around mouse clicks. A programmer can simply have their button listen to a press from a mouse button and cause an action or series of actions to occur based around that button click. The buttons we are creating will be used as navigation to submenus since we need a way of navigating through the inventory and equipment screens. Therefore, on our main screen, we will need a button for both inventory and equipment. We will also need a button to go to the pause menu and resume playing the game as well.

Let us start by creating our first button. Navigate to **Palette** | **Common** | **Button** and place it in your Canvas Panel under the **Hierarchy** tab:

![UMG buttons](img/B04548_04_34.jpg)

For organization purposes, we will lay our first button on the top-right of the menu. So the best thing to do is navigate to **Details** | **Slot** | **Anchors** and anchor the button at the top-right. This will ensure that as the screen or objects resize, the button aligns to the right-hand side:

![UMG buttons](img/B04548_04_35.jpg)

You should notice that the anchor icon on your screen moves to the top-right of the screen. You will also notice that the **Position X** value changes to a negative number that reflects the size of your screen, since the origin of the button position is placed at the opposite end of the screen; the values of **Position X** of this particular button are now flipped. This concept may be confusing at first, but it will make the math for the placement of each button much easier in the long run:

![UMG buttons](img/B04548_04_36.jpg)

Change the **Position X** value to **-200** (since **Position X** of the button is now **-1920** to be positioned from the left and **-100** to be positioned to the right, to add **100** pixels of padding would be **-200**) and **Position Y** value to **100**. Name this button **Button_Inventory**:

![UMG buttons](img/B04548_04_38.jpg)

We will now add text to the button. So select **Text** under **Palette** | **Common** | **Text** and drag it into the button. You will notice that the text is within the button in both the **Hierarchy** tab and the **Designer** view:

![UMG buttons](img/B04548_04_39.jpg)

You may also notice that the text is not sized to our liking and it does not fit into the button. Rather than resizing the button right away, let us resize the text to our liking first and then resize the button around the text so that the text is readable. Navigate to **Details** | **Appearance** | **Font** and change the font size to **48**:

![UMG buttons](img/B04548_04_40.jpg)

Then, under **Details** | **Content** | **Text**, change the text to **Inventory** and the name of the Text widget to **Menu_Inventory**:

![UMG buttons](img/B04548_04_41.jpg)

Click on **Button_Inventory**. You may be thinking that checking **Size to Content** would be the right idea here, but it isn't in our circumstances because you will be creating multiple buttons, each with a different text in it. Therefore, if they were all sized to the content (content being the text within the button), all your buttons would be sized differently, which is very unattractive. Instead, you should pick a button size that will easily fit all the text, even for your longest worded text. For this button, we will change the **Size X** value to **350** and **Size Y** value to **100**:

![UMG buttons](img/B04548_04_42.jpg)

You will notice that the button is drawn off the screen and this is because the button, like every other object, is still drawn from the top-left of the object, so we will need to adjust our **Position X** value again; however, the math is easy since we are anchored at the top-right. All we need to do is take the horizontal sizes of our button, 350, and then subtract it from where the button thinks the right edge of our screen is due to the anchor, which is 0\. So this gives us *0 - 350 = -350*. Then, we take -350 and subtract the 100 pixels of padding that we want, which gives us *-350 - 100 = -450*, which is the value we should change our **Position X** to:

![UMG buttons](img/B04548_04_43.jpg)

Now that we have a button perfectly placed, we can place more buttons. We will use the same steps to create an **Equipment** button below the **Inventory** button. Once you have completed creating the **Equipment** button, it can be placed just below the **Inventory** button:

![UMG buttons](img/B04548_04_44.jpg)

We can also create an **Exit** button, which we will place at the bottom-right of the screen:

![UMG buttons](img/B04548_04_45.jpg)

There you have it—we have finished designing the first screen of our pause menu. You will notice that we have not yet programmed the buttons to do anything, and this is because we do not have screens for the buttons to navigate to, so it won't make sense to program the buttons just yet. The next steps will be to design our submenus.

# The UMG inventory submenu

As mentioned earlier, we need to create submenus for our buttons to navigate to. Using UMG, there are several ways to create submenus, but the most straightforward way is to create a new Widget Blueprint for each submenu and then bind the Widget Blueprints together.

Since we will need many of the same items in our submenus such as the character names and most of their stats, we can save a lot of time by carefully making a copy of our main pause menu, renaming it, and then editing it to fit whatever submenus we need. Since we have initially saved the main pause menu as **Pause**, we may want to first rename it so that it is more descriptive. So head back into your **Content Browser**, locate where you saved your pause menu, and rename it by right-clicking on the pause menu widget and selecting **Rename**. Rename this file as **Pause_Main**:

![The UMG inventory submenu](img/B04548_04_46.jpg)

Next, make a duplicate of **Pause_Main** by right clicking on the file and selecting **Duplicate**:

![The UMG inventory submenu](img/B04548_04_47.jpg)

Rename this as **Pause_Inventory**:

![The UMG inventory submenu](img/B04548_04_48.jpg)

We will now be able to design an **Inventory** screen. Open up your newly created **Pause_Inventory** Widget Blueprint. You will notice that it is an exact duplicate of **Pause_Main**. From here, we can edit out what is not needed. First of all, we are not planning to have any items that affect XP, so we can remove XP Text Blocks from our characters:

![The UMG inventory submenu](img/B04548_04_49.jpg)

Also, we will not need to keep a track of gold in our **Inventory** screen either. So, we can remove gold.

For ease of creation, we will also make the navigation around the main pause screen and its submenus "old school", by using **Pause_Main** as a central hub to all submenus, such as **Pause_Inventory** and **Pause_Equipment**, and only allowing the player to enter the **Equipment** screen if they are backed out to **Pause_Main** and they press the **Equipment** button from there. Based on the idea behind this design, we may also remove the **Inventory** and **Equipment** buttons from this screen:

![The UMG inventory submenu](img/B04548_04_51.jpg)

We can, however, keep the **Exit** button, but based on our ideas behind the screen navigation, we should rename this button and its Text Block to reflect backing out of the screen and going to **Pause_Main** when pressed. So we can select **Button_Exit** and rename it as **Button_Back**:

![The UMG inventory submenu](img/B04548_04_52.jpg)

Then, select the Text Block within the **Exit** button, rename it as **Menu_Back**, and change the text to **Back**:

![The UMG inventory submenu](img/B04548_04_53.jpg)

In the previous chapter, we defined more stats than just HP and MP; we also defined attack power, defense, and luck. While health and magic potions don't typically affect any stats other than HP or MP, you may later on want to create items that are usable and effect things such as luck, defense, or attack power. In preparation of this, we will create placeholders for these three other stats for each character in the same way you created the HP and MP Text Blocks. We will be positioning these stats below the HP and MP stats. Note that, if you run out of room for these stats, you may need to play around with spacing. Also, remember to name all Text Blocks you make with something very descriptive so that you can identify them when the time comes to reference them.

When you are finished adding stats, your Text Blocks should look something like the following screenshot:

![The UMG inventory submenu](img/B04548_04_54.jpg)

We now need a place that will populate our inventory. Since we are unsure about how many items the player in the game will carry, it will be safest to create a Scroll Box that will be populated with our inventory. We will also want to create a Scroll Box that is wide enough in case we have items that have very long names. If you designed your screen like I have, you should have plenty of room for a Scroll Box. To create a Scroll Box, navigate to **Palette** | **Panel** | **Scroll Box**:

![The UMG inventory submenu](img/B04548_04_55.jpg)

Then, drag it into your Canvas Panel under the **Hierarchy** tab:

![The UMG inventory submenu](img/B04548_04_56.jpg)

For now, rename the Scroll Box as **ScrollBox_Inventory**. Then, change the position so that it is placed in the middle of the screen while neatly taking up a wide amount space on the screen. I will change my **Position X** value to **700**, **Position Y** value to **200**, **Size X** value to 600, and **Size Y** value to **600**. When you are finished, your Scroll Box should look something like the following screenshot:

![The UMG inventory submenu](img/B04548_04_57.jpg)

In a later chapter, we will be dynamically inserting items into the Scroll Box and creating logic to apply the effects of the items to each character.

To finish off this screen, you should notify the player about which screen they are currently viewing. So create another Text Block and size the font to something large, such as 48 pixels. Pick an anchor for your Text Block that is at the center-top. This will make it such that your Text Block recognizes the 0 *X* position as the middle of the screen and the *0* Y position as the top of the screen. So you can now put **0** as the **Position X** value and pad the **Position Y** value:

![The UMG inventory submenu](img/B04548_04_59.jpg)

You will notice that the inventory is not quite written at the middle of the screen, so adjust the **Position X** value until it is. I changed my **Position X** value to half of the size of the Text Block, which came out to be -135:

![The UMG inventory submenu](img/B04548_04_60.jpg)

At this point, you can save and compile your **Inventory** screen. We are done for now.

# The UMG equipment submenu

The last submenu we need to design is the equipment submenu. Since our submenu for equipment will be very similar to the submenu for inventory, the easiest way to start would be to navigate back to **Content Browser**, duplicate **Pause_Inventory**, and rename it as **Pause_Equipment** so that **Pause_Equipment** is a direct copy of **Pause_Inventory**. Next, open up **Pause_Equipment**.

We will be designing this screen in a similar way to the **Inventory** screen. We will still be using the Scroll Box to populate items (in this case, equipment). We will be mostly keeping the same stats for each character; we will continue utilizing the Back button that will eventually navigate back to the pause screen. Let us edit the differences. First, change the title of the screen from **Inventory** to **Equipment** and reposition it so that it is horizontally aligned to the middle of the screen:

![The UMG equipment submenu](img/B04548_04_61.jpg)

Next, we will need to edit the character stats. We may have equipment in this game that when equipped, will change the AP, DP, and Lk stats. However, we will most likely not have equipment that will have an effect on HP and MP. We also know that we will need weapons and armor for each character. Therefore, we can easily edit the text of HP and MP out with weapon and armor (which I will call **Weap** and **Arm** to save space). In terms of details, I will change the name of the **Menu_HP** text block to **Menu_Weapon** and the text of the Text Block to **Weap**. We will do something similar to **Menu_MP** by changing it to an armor slot:

![The UMG equipment submenu](img/B04548_04_62.jpg)

Follow similar naming conventions when switching out any other character's HP and MP with weapon and armor placeholders. When you are finished, your screen should look like the following screenshot:

![The UMG equipment submenu](img/B04548_04_63.jpg)

Since our characters will be equipping weapons and armor, we will need placeholders for these slots. Eventually, we will allow the player to select equipment that they want to equip, and the equipment will appear in the appropriate weapon or armor slot. The type of widget that would be most appropriate is **Border**. This will contain a Text Block that will change when a weapon or armor is equipped. To do this, select **Border** from **Palette** | **Common** | **Border**. Drag the Border into the Canvas Panel:

![The UMG equipment submenu](img/B04548_04_65.jpg)

Then, position the Canvas Panel so that it is in the same position as the Text Block that is placed to the right of the soldier's **Menu_Weapon**. At this point, you may delete the Text Block that is to the right of **Menu_Weapon**. It was originally used as the text for HP, and we will no longer need it:

![The UMG equipment submenu](img/B04548_04_67.jpg)

We will still need text in the border, so drop text from **Palette** | **Common** | **Text** into your Border:

![The UMG equipment submenu](img/B04548_04_68.jpg)

You can keep the defaults of the Text Block for now, except you will notice that the border did not resize and everything is still white. Navigate back to your Border and check **Size to Content**. Under **Appearance** | **Brush Color**, change the **A** value to **0**. The **A** value is alpha. When the alpha is 0, the color is completely transparent, and when the alpha is 1, the color is completely opaque; anywhere in between is only slightly transparent. We don't really care about seeing the color of the block, we want it to be invisible to the player:

![The UMG equipment submenu](img/B04548_04_69.jpg)

Lastly, change the Border name to something descriptive such as **Border_Weapon**:

![The UMG equipment submenu](img/B04548_04_70.jpg)

Navigate back to the Text Block within the border. Name the Text Block **Text_Weapon**, and change the font to a regular style at 32 pixels to match the rest of the Text Blocks:

![The UMG equipment submenu](img/B04548_04_71.jpg)

Now that you know how to design borders and Text Blocks for the soldier's weapon, you can design Borders and Text Blocks for the soldier's armor and any other character's weapon and armor. When you are finished, you should have something that looks like the following screenshot:

![The UMG equipment submenu](img/B04548_04_72.jpg)

At this point, we are finished designing all our screens that appear when the player presses the pause button. The next steps will be to program the functionality of these screens.

# Key binding

We are now going to bind a key to open up the pause menu, and only allow this to happen when the player is not in battle (in other words, the player is out in the field). Since we already have a **FieldPlayer** set up from the previous chapter, we can easily create actions within our **FieldPlayer** Blueprint class that will control our pause menu. To start, navigate to **Blueprints** | **Open Blueprint Class…** | **FieldPlayer**:

![Key binding](img/B04548_04_73.jpg)

At this point, we are going to want to have the pause screen pop up when the player presses a key; in this case, we will use *P* for pause. To do this, we will first need to create a key event that will fire off a set of actions of our choice after a specific key is pressed. To start this key event, right-click on your Event Graph, which will open up all actions that can be associated with this Blueprint, and then navigate to **Input** | **KeyEvents** | **P**:

![Key binding](img/B04548_04_74.jpg)

Once done, this will create a key event for the press and release of *P*. You will notice that this event has pressed and released executables that work as they are described, an action can occur when the player presses *P* or when the player releases *P*. For our actions regarding the pausing of the game and the pop up of the pause menu, we will use the released executable, because in order for the released executable to be called, it would mean that the player has gone through the act of pressing and releasing the key. It is often the best practice for a player to commit to a button press just like a player commits to a move in Chess by letting go of a piece. Before we pop up the pause menu, let us pause the game by creating a call to the **Set Game Paused** function that can be found by right-clicking in the Event Graph and navigating to **Game** | **Set Game Paused**:

![Key binding](img/B04548_04_75.jpg)

Within the **Set Game Paused** node, check **Paused** so that it is set to true, and link the **Released** executable of the key event **P** to the in pin of the **Set Game Paused**. Your game should now pause whenever the player presses and releases *P*:

![Key binding](img/B04548_04_76.jpg)

From here, we will pop up the pause menu. To do so, upon the game being paused, we will create the main pause screen by right-clicking on the Event Graph and navigating to **User Interface** | **Create Widget**. This allows us to create instances of any widget that we made. We can create our **Pause_Main** here by pressing the **Select Class** drop-down menu within **Create Widget** and selecting **Pause_Main**:

![Key binding](img/B04548_04_78.jpg)

Next, we can link the out pin of **Set Game Paused** to the in pin of the **Create Pause_MainWidget**:

![Key binding](img/B04548_04_79.jpg)

This will make it such that after the game is paused, **Pause_Main** will be created. Although we are creating **Pause_Main**, this will still not pop up on screen until we tell it to draw on screen. To do so, we will need to create a call to the **Add to Viewport** function that can add any graphic to a viewport. To do this, left-click and drag out the **Return Value** pin from the **Create Pause_Main** widget node, and select **User Interface** | **Viewport** | **Add to Viewport**. This will create a new **Add to Viewport** node:

![Key binding](img/B04548_04_80.jpg)

If you test this, you will notice that the game pauses and the pause menu pops up, but it is missing a mouse cursor. To add the mouse cursor, we will first need to get the player controller by right-clicking on the Event Graph and navigating to **Game** | **Get Player Controller** | **Get Player Controller**:

![Key binding](img/B04548_04_81.jpg)

From here, simply left-click and drag out the **Return Value** pin from the **Get Player Controller** node, then select **Variables** | **Mouse Interface** | **Set Show Mouse Cursor**:

![Key binding](img/B04548_04_82.jpg)

Once done, link the out pin of the **Add to Viewport** node to the in pin of **Set Show Mouse Cursor**. What this will do is set the **Show Mouse Cursor** variable (which is a Boolean) to either true or false after the pause menu has been displayed in the viewport. The **Set Show Mouse Cursor** variable needs to also get the player controller because the player controller holds the mouse input information.

If you playtest this now, you will notice that the mouse cursor still does not show up; this is because **Show Mouse Cursor** within **Set Show Mouse Cursor** is unchecked, meaning that **Show Mouse Cursor** is set to false, so check the box whenever you want to show the mouse cursor.

At this point, your menu should pop up perfectly after pressing *P* and the mouse should be completely visible and controllable. Your level blueprint should now look like the following screenshot:

![Key binding](img/B04548_04_84.jpg)

You will notice that none of the buttons work in the actual menu so we cannot exit the pause menu or view any of the submenus. This is because we have not programmed any of the menu buttons yet. We will now focus on programming those buttons.

# Button programming

Now that we have completed player access to the pause menu, we will now focus on navigation within the main pause menu and its submenus. At this point, head back into your **Pause_Main** widget. Let us first create the navigation to **Pause_Inventory**. To do this, click on the **Inventory** button:

![Button programming](img/B04548_04_85.jpg)

Navigate to **Details** | **Events** | **OnClicked** and then press the **+** button:

![Button programming](img/B04548_04_86.jpg)

Clicking on the **+** button will automatically open the Event Graph of **Pause_Main** and also create an **OnClicked** event:

![Button programming](img/B04548_04_87.jpg)

The **OnClicked** event will work similar to our key bind in the previous section where we created something that allows us to press a key, which triggers an event that can then trigger a series of actions. Only this time, the **OnClicked** event is bound to our **Inventory** button and will only trigger when the user has left-clicked on the **Inventory** button. What we will want to do is, when we click on the **Inventory** button, have it create a **Pause_Inventory** widget that gets displayed on the screen. This should sound very familiar because we just did something like this with **Pause_Main**. So firstly, create a widget and attach it to the **OnClicked** event. Next, you will notice that the **Class** pin in the widget is empty so we need to select a class. You will be selecting **Pause_Inventory** since we want to create the inventory widget when the button is pressed:

![Button programming](img/B04548_04_89.jpg)

Lastly, just add this widget to the viewport so the user can see the inventory being displayed. In the end, your Blueprint should look like the following screenshot:

![Button programming](img/B04548_04_90.jpg)

If you test your pause screen now, you should notice that you are able to navigate to your inventory screen, but you are not able to navigate back to your main pause screen. This is easily fixable. Simply open your **Pause_Inventory** widget, press the **Back** button, navigate to **Details** | **Events** | **OnClicked**, and then press the **+** button:

![Button programming](img/B04548_04_91.jpg)

Just like our last button, the Event Graph will automatically open up and an **OnClicked** event for our button will be created; only this time, the event is bound to our **Back** button:

![Button programming](img/B04548_04_92.jpg)

From here, you will set the screen to remove itself by linking the **OnClicked** event to **Remove from Parent**.

When you are finished creating the Blueprint for your **Back** button, it should look like the following screenshot:

![Button programming](img/B04548_04_93.jpg)

You can now create the same setup for navigation from **Pause_Equipment** by making sure that when we click on the **Equipment** button, we create a **Pause_Equipment** widget and display it; when we click on the **Back** button in the **Pause_Equipment**, we navigate back to the **Pause_Main** removing the **Pause_Inventory** screen.

The next step is to allow the player to exit when they click on the **Exit** button. To do this, you must first create an **OnClicked** event on the **Exit** button within **Pause_Main**. Again, when you press the **+** button of **OnClicked** within the **Design** view, an **OnClicked** button for the **Exit** button will be created in the Event Graph:

![Button programming](img/B04548_04_94.jpg)

From here, you will set the screen to remove itself by linking the **OnClicked** event to **Remove from Parent**:

![Button programming](img/B04548_04_95.jpg)

Your screens should now navigate perfectly, but we are not done yet. The last thing you will notice is that the game is still paused when the pause menu is exited. We will need to unpause the game. This fix is very simple. Within the **Pause_Main** Blueprint, simply link **Set Game Paused** to **Remove from Parent** so that when the widget is removed, the game unpauses:

![Button programming](img/B04548_04_96.jpg)

You may notice that when leaving the pause menu, the mouse cursor is still present. You can remove the mouse cursor by simply creating a **Set Show Mouse Cursor** node and having it connected to your **OnClicked Button_Exit** event after you unpause the game, which would be similar to how you added a mouse cursor in the first place, this time making sure the checkbox within the **Set Show Mouse Cursor** node is unchecked meaning that S**et Show Mouse Cursor** is set to false, and attaching a **Get Player Controller** to it.

There you have it. We are now finished with the navigation of our pause menu and its submenus.

# Summary

In this chapter, we completed pause menu placeholders for other important aspects of our game such as the **Inventory** and **Equipment** Scroll Boxes that will hold the inventory/equipment that we acquire in the game. We will continue to add to this pause menu in the next few chapters by covering the tracking of stats, gold, items, and equipment.