# Chapter 7. Matinee

Matinee provides the ability to keyframe various properties of actors over time, either dynamically in gameplay or in cinematic game sequences. The system is based on specialized tracks in which you can place keyframes on certain properties of an actor. The **user interface** (**UI**) of Matinee is similar to other nonlinear video editing software, which makes it easier and familiar for video editors.

In this chapter, we will create a Matinee sequence and learn how we can play it through **Level Blueprint**. So to get started, let's start Unreal Engine 4 and create a new project based on **Third Person Template**.

# Creating a new Matinee

To open the Matinee UI, we first need to create the Matinee asset. You can create a Matinee asset by clicking on the **Matinee** button and selecting **Add Matinee** in the level editor toolbar. When you click on it, you might get a warning saying that Undo/Redo data will be reset. This is because, while you are in the Matinee mode, some changes are translated into keyframes and editor needs to clear the undo stack. Click on **Continue** and a new Matinee Actor will be placed in the level and the Matinee editor will open. Let's take a closer look at the Matinee window:

![Creating a new Matinee](img/B03950_07_01.jpg)

Creating new Matinee Actor

This is the Matinee Actor icon:

![Creating a new Matinee](img/B03950_07_02.jpg)

Matinee Actor placed in world

After creating a new Matinee Actor, it will automatically open the **Matinee** Window. If it doesn't, then select the **Matinee Actor** in world and click on **Open Matinee** in the **Details** panel.

![Creating a new Matinee](img/B03950_07_03.jpg)

## Matinee window

Let's take a quick look at the Matinee window:

![Matinee window](img/B03950_07_04.jpg)

The Matinee window consists of:

*   **Toolbar**: This contains all the common buttons for Matinee editor, such as playing the Matinee, stopping it, and so on. Let's take a closer look at the toolbar buttons:

    *   **Add key**: This adds a new keyframe at the current selected track.
    *   **Interpolation**: This sets the default interpolation mode when adding new keys.
    *   **Play**: This plays a preview from the current position in the track view at normal speed to the end of the sequence.
    *   **Loop**: This loops the preview in the loop section.
    *   **Stop**: This stops the preview playback. Clicking twice will rewind the sequence and place the time bar at the beginning of Matinee.
    *   **Reverse**: This reverses the preview playback.
    *   **Camera**: This creates a new camera Actor in world.
    *   **Playback Speed**: This adjusts the playback speed.
    *   **Snap Setting**: This sets the timeline scale for snapping.
    *   **Curves**: This toggles curve editor.
    *   **Snap**: This toggles snapping of time cursor and keys.
    *   **Time to frames**: This snaps the timeline cursor to the setting selected in the **Snap Setting** dropdown. It is only enabled if **Snap Setting** is using frames per second.
    *   **Fixed Time**: This locks playback of Matinee to the frame rate specified in **Snap Setting**. It is only enabled if **Snap Setting** is using frames per second.
    *   **Sequence**: This fits the timeline view to the entire sequence.
    *   **Selected**: This fits the timeline view to the selected keys.
    *   **Loop**: This fits the timeline view to the loop section.
    *   **Loop sequence**: This automatically sets the start and end of the loop section to the entire sequence.
    *   **End**: This moves to the end of the track.
    *   **Record**: Opens the **Matinee Recorder** window.
    *   **Movie**: This allows you to export the Matinee as a movie or image sequences.

    Since Matinee is similar to other nonlinear video editors, you can use the following common shortcut keys:

    *   *J* to play the sequence backward
    *   *K* to stop/pause
    *   *L* to play the sequence forward
    *   Plus (*+*) to zoom in to the time line
    *   Minus (*-*) to zoom out of the time line

*   **Curve editor**: This allows you to visualize and edit the animation curves used by tracks in the Matinee sequence. This allows for fine control over properties that change over time. Certain tracks with animation curves can be edited in curve Editor by toggling the **Curve** button. Clicking on it will send the curve information to curve editor where the curve will be visible to the user.
*   **Tracks** This is the heart of the Matinee window. This is where you set all your keyframes for your tracks and organize them into tabs, groups, and folders. By default, when you create a Matinee, the length is set to 5 seconds.![Matinee window](img/B03950_07_05.jpg)

    *   **Tabs**: These are used for organization purposes. You can put your tracks into various tabs. For example, you can put all your lights in your Matinee to the **Lights** tab, camera to the **Camera** tab, and so on. The **All** tab will show all tracks in your sequence.
    *   **Track List**: This is where you create tracks that can create keyframes in the timeline and organize them into different groups. You can also create new folders and organize all groups into separate folders.
    *   **Timeline Info**: This shows information about the timeline including the current time, where the cursor is, and the total length of the sequence.
    *   **Timeline**: This shows all the tracks within the sequence and this is where we manipulate objects, animate cameras, and so on using keyframes. The green area shows the loop section (in between the green markers). At the bottom of track view, you can see a small black bar, which is called the **Time Bar**. If you click on and hold it, you can scrub the timeline forward or backward, which allows you to quickly preview the animation. To adjust the length of the sequence, you move the far right red marker to the length you want this Matinee to be.

## Manipulating an object

Matinee can be used to create cut scenes where you move the camera and manipulate objects or it can be used for simple gameplay elements such as opening doors and moving lifts. In this example, we will see how we can move a simple cube from one location to another location.

From **Engine Content**, we will drag and drop the **Cube** mesh into our world. This is located in the `Engine Content\BasicShapes` folder.

![Manipulating an object](img/B03950_07_06.jpg)

To get **Engine Content**, you need to enable it in **Content Browser**.

1.  At the bottom right corner of **Content Browser**, you can see **View Options**.
2.  Click on it and then enable **Show Engine Content**.

![Manipulating an object](img/B03950_07_07.jpg)

After placing our **Cube** in world, let's open the **Matinee** editor window. Make sure the **Cube** is selected in world and right-click in the track list area and select **Add New Empty Group**. You will now be prompted to type a name for your group. Let's call it **Cube_Movement**.

### Note

Note that if you see a notification at the bottom-right corner of your screen saying **Cube Mobility** has been changed to **Movable**, don't panic. Actors that are being manipulated in Matinee must set the **Mobility** to **Movable**.

If you click on this group in Matinee now, you can see the **Cube** in world will be automatically selected for you. This is because, when we created the group, we had the **Cube** selected in world and whatever object you have selected in world will automatically be hooked to the group you create.

To move the cube in world, we need to add a **Movement Track** to our **Cube_Movement** group. To create this track:

1.  Right-click on our **Empty Group** (**Cube_Movement**).
2.  Select **Add New** **Movement Track**.

![Manipulating an object](img/B03950_07_08.jpg)

This will add a new movement track to our **Empty Group** and will set the current position of our cube as the first keyframe.

![Manipulating an object](img/B03950_07_09.jpg)

The small triangle in the beginning of the timeline is the keyframe

Now, we want the cube to move to the right by some distance and, by the end of this sequence, it should come back to its default position. So let's scrub the time bar to the middle of the sequence (since the default length is 5 seconds long, we will move the time bar to **2.5**) and go back to **Viewport** editor. There, we select and move the cube by some distance to the right side (*Y* axis) and press *Enter*. Note that now Matinee has created a new keyframe for you at the time slot **2.5** and you will see a dotted yellow line that represents the movement path of the cube.

![Manipulating an object](img/B03950_07_10.jpg)

To set the keyframe at the exact time (for example, precisely at **2.5**) you can left-click on the key frame to select it and then right-click and select **Set Time**. You will now be prompted to enter the new time to set the keyframe. Here, you can type and set **2.5**.

If you scrub the time bar now, you will see the cube move from its original position to the new position that we keyframed at time **2.5**. Now, to get the cube back to its original position at the end of the sequence, we can simply copy paste the first keyframe to the end of the sequence. To do so, click on the first keyframe and press *Ctrl*+*C* to copy it. Then, scrub the time bar to the end of the sequence and press *Ctrl*+*V* to paste it. The finished Matinee should look like this:

![Manipulating an object](img/B03950_07_11.jpg)

If you hit **Play** in the toolbar now, you will see the cube move from its original location to the new location and then, by the end of sequence, it will go back to its original location.

Now that our Matinee is ready, we will see how to play the Matinee in game. What we are going to do is place a trigger box in level and, when our player overlaps it, Matinee will play. When our player steps out of the trigger box, Matinee will stop.

To place a trigger box in world, you need to drag it and drop it into the viewport from the **Modes** tab (which is under **Place** in the **Volume** category). If you don't have the **Modes** tab, then:

1.  Press *Shift*+*1* to open it (make sure your viewport is in focus).
2.  In the **Modes** tab, go to the **Place** mode (*Shift*+*1*).
3.  Select the **Volumes** tab.
4.  Drag and drop the **Trigger Volume** box.

![Manipulating an object](img/B03950_07_12.jpg)

Once the trigger box is placed in world (feel free to adjust the size of the trigger box), right-click on it and navigate to **Add Event** | **OnActorBeginOverlap**.

![Manipulating an object](img/B03950_07_13.jpg)

This will add a new **Overlap Event** for our **Trigger Volume** in **Level Blueprint**. Since we need to stop the Matinee after exiting the trigger, we will right-click again on the **Trigger Volume** and navigate to **Add Event** | **OnActorEndOverlap**. We now have two events (**Begin Overlap** and **End Overlap**) in our **Level Blueprint**.

![Manipulating an object](img/B03950_07_14.jpg)

As you can see, both overlap events give us the actor that is currently overlapping this **Trigger Volume**. We will use this information to play the Matinee only when a character is overlapping. To do so, we will have to follow this process:

1.  Click and drag from the other Actor pin in the **OnActorBeginOverlap** event. From the resulting context window, type **Cast to Character** and select it.
2.  Connect the execution pin of **OnActorBeginOverlap** to the **Cast** node we just created.
3.  To play the Matinee, we first need to create a reference of it in **Level Blueprint**. To do so, select the Matinee icon in world and right-click inside the **Level Blueprint**. From the resulting context window, select **Create a reference to Matinee Actor**. This will add a new node, which is referred to the Matinee Actor in world. From this node, drag a new wire and type **Play** and select it.
4.  Connect the output (unnamed) execution pin of the **Character** node to the **Play** node of Matinee.
5.  To stop the Matinee when exiting the trigger, you can do the same setup as previously, but instead of the play node, use the **Stop** node.

The final graph should look like this:

![Manipulating an object](img/B03950_07_15.jpg)

Now, when you play the game and overlap the trigger, our Matinee will play.

![Manipulating an object](img/B03950_07_16.jpg)

## Cutscene camera

Since you have learned how to create a Matinee and move an object, it is time to learn how to create a simple cut-scene. In this section, we will create a camera that focuses on the cube when Matinee is triggered.

To create a camera, let's first position the viewport camera at the right location. In your editor **Viewport**, navigate to the place where you want the Matinee camera to be. In the following screenshot, you can take a look at where I placed the camera:

![Cutscene camera](img/B03950_07_17.jpg)

After navigating to your desired location, open the **Matinee** window. On the toolbar, click on the **Camera** button (this will prompt you to enter a new group name) to create a camera at your current **Viewport** camera location.

![Cutscene camera](img/B03950_07_18.jpg)

This will also create a new **Camera** group with two tracks. They are **Field of View** (**FOV**) and **Movement**. Since we don't use the FOV track, you can right-click on it and select **Delete Track**, or simply press *Delete* to remove it from the track list.

With the **Movement Track** of the camera selected, scrub the time bar to the end of the sequence. Then in the editor **Viewport**, select the camera created by Matinee and move it to a new location. In this example, I moved the camera to the right side and rotated it by 30 degrees. In the following screenshots, you can see the initial location of the camera and the new location at the end of the sequence.

![Cutscene camera](img/B03950_07_19.jpg)

This is the new location of the camera:

![Cutscene camera](img/B03950_07_20.jpg)

If you play now and trigger the Matinee from the **Trigger Volume** we placed earlier, you will see the cube moving as usual but you will not see it from the camera perspective. To see it through the camera we placed now, you need to add a **Director Track** to your Matinee. Let's take a look at what a **Director Group** is.

### Director group

Director group serves the main function of controlling the visual and audio of your Matinee. The important function of this group is to control which camera group is chosen to be seen in the sequence. We use this group to cut between one camera and the next when we have multiple cameras in Matinee.

To create a new **Director Group**, right-click on the track list and select **Add New Director Group**. A new separate group will be opened on top of all other groups.

![Director group](img/B03950_07_21.jpg)

Since we only have one camera in this group, we will add that one to our director track. Select the director track and press *Enter*. A new pop up will ask you which track to choose, so select **MyCamera** group (this is the group we created using the **Camera** button in Matinee toolbar). The name **MyCamera** was something I chose. A new keyframe will be added to the director track that says **MyCamera [Shot0010]**. This means that whenever this Matinee is played, you will see through the **MyCamera** group. Later, if you add more cameras, you can switch between cameras in **Director Group**.

The end result should look like this:

![Director group](img/B03950_07_22.jpg)

Now, if you play the Matinee in the game, you will see it through the new **Camera** view.

![Director group](img/B03950_07_23.jpg)

Sometimes, when cutscenes are played, it's better to disable **Player** movement (so that when the cutscene is active all player inputs, such as moving around, will be disabled) and HUD and all that. To do these, select the Matinee Actor in world and then in the **Details** panel, you can set the necessary options.

![Director group](img/B03950_07_24.jpg)

# Summary

Matinee is a very powerful tool to create in game cinematics. With multiple cameras and other visual/audio effects, you can create good-looking and professional cinematics. Since you learned how to manipulate objects and cameras in this chapter, you should now try to create an elevator movement with a camera that acts as a CCTV.