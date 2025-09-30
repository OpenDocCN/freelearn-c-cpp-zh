# Chapter 7. Terrain and Cinematics

In this chapter, we will cover a few level-enhancing features. We will create some outdoor terrain for our level as well as add a short cinematic sequence at the start of the game level.

In this chapter, we will look at the following topics:

*   Creating an outdoor terrain
*   Adding a shortcut for a cinematic sequence at the beginning of the same level

# Introducing terrain manipulation

Terrain manipulation is needed when you want to create large natural landscape areas, such as mountainous or valley areas that are covered with foliage. This can be in the form of trees/grass, lakes, and rivers that are covered with rocks or snow, and so on. The Landscape tool in Unreal Engine 4 allows you to creatively design a variety of terrains for your game maps easily, while allowing the game to run at a reasonable frame rate.

When playing in a map that has large outdoor terrains, for example, maps with a large number of trees or many elevations, such as mountains, the effective frame rate is expected to be reduced due to an increase in the number of polygons that need to be rendered on the screen. Hence, being well-versed in landscaping so that polygon counts are kept under control is important to ensure that the map is actually playable. It is also good to bear in mind to make use of optimization techniques, such as LOD and fog to mask the distant places, which can give you a sense of unending open land. If you are planning to create an open world, you can also use the Procedural Foliage tool (available in Unreal 4.8 and higher versions) to spawn these features for you.

Let's get ourselves familiarized with the Unreal Landscaping tool and start creating some outdoor environments for our game level. We will learn how to perform simple contouring of the outdoor space with low hills, grass, and trees. Then, we will create a small pond in the area. For more accurate landscaping, we can import a height map to help us with the creation of the landscape.

## Exercise – creating hills using the Landscape tool

Let's perform the following steps to create hills using the Landscape tool:

1.  Open `Chapter6.umap` and save it under `Chapter7_Terrain.umap`.
2.  Go to **Modes**, click on the Landscape tool (the icon looks like a mountain) and then click on **Manage**.
3.  Select **Create New** (the other option here is to make use of a height map, which we will cover later in the chapter).
4.  To select a Material, you can click on the search icon and type `M_Ground_Grass`, or go to **Content Browser** | **Content** | **Materials**, select **M_Ground_Grass**, and click on the arrow next to **Landscape Material** to assign the material.
5.  For this example, we are going to leave all of the landscape settings at their default values that are listed, as follows. The next section will explain the options for the rest of the values in further detail:

    *   **Scale**: X = 100 Y = 100 Z = 100
    *   **Section Size**: 63 x 63 quads
    *   **Section Per Component**: 1 x 1 section
    *   **Number of Components**: 8 x 8
    *   **Overall Resolution**: 505 x 505

The following screenshot shows the top view of the grass landscape that we have created. Notice the 64 green squares. You will need to switch to the **Top** view to view it.

![Exercise – creating hills using the Landscape tool](img/B03679_07_01.jpg)

Now, we'll switch over to the **Perspective** view. The grass landscape seems like it's covering half the house. Take a look at the following screenshot:

![Exercise – creating hills using the Landscape tool](img/B03679_07_02.jpg)

Note that if we had created the landscape on an empty map, we would not have this issue, as we would have built the house on the landscape grass instead. So, we have to perform an additional step here to move the landscape grass under the house so that we do not have a house that's submerged under the grass. You need to select **Landscape** and **LandscapeGizmoActiveActor** from **World Outliner**, as shown on the right-hand side of the following screenshot. Remember to switch **Mode** back to **Place**, instead of the **Landscape** we were in to create the grass. The **Place** mode allows the translation/rotation of the selected object. Move the grass to just below the house, as shown in the following screenshot:

![Exercise – creating hills using the Landscape tool](img/B03679_07_03.jpg)

### Note

Note that this step is performed because we add the landscape grass after we've built the house.

Now, we are ready to sculpt this flat land into some terrain. Go to **Modes** | **Landscape** | **Sculpt** again. Use the Sculpt tool, **Circle Brush**, and the **Smooth Falloff** combination, as shown in the upcoming screenshot. The default settings should be as follows:

*   **Brush Size**: **2048**
*   **Brush Falloff**: **0.5**
*   **Tool Strength**: **0.3**

To illustrate the size of the 2048 brush, I have switched to the **Top** view:

![Exercise – creating hills using the Landscape tool](img/B03679_07_04.jpg)

When **Brush Size** is set to **1000**, the brush radius is reduced, as shown in the following screenshot:

![Exercise – creating hills using the Landscape tool](img/B03679_07_05.jpg)

Now that we have an idea about the difference in radii, we will switch back to the **Perspective** view. Position your working screen to a slightly angled top perspective view, as shown in the following screenshot. Set **Brush size** to **1000** and **Tool Strength** to **0.4**:

![Exercise – creating hills using the Landscape tool](img/B03679_07_06.jpg)

Start by creating low hills around the house by clicking on the area around the house. I used a mix between a brush size of 1000 and 2048.

The following screenshot shows how the area looked after I worked on it for a bit. Note that the area in front of the wide windows where I created a depression. This is achieved by holding *Ctrl* and then clicking on the area. This depression will take the form of a lake in front of the dining area.

![Exercise – creating hills using the Landscape tool](img/B03679_07_07.jpg)

Create two box BSPs to fill up the depressed area. Apply the Lake Water material to the box BSPs. The following screenshot shows the same area with the box BSPs put in place. Use the Translation tool to keep both BSP areas on the same ground level at the location of the depression.

![Exercise – creating hills using the Landscape tool](img/B03679_07_08.jpg)

Next, I touched up the external area of the house. Use the **Unlit** mode to help you see the house better. This screenshot shows you how the house and area around it look after touching them up with the **MyGreyWall** material:

![Exercise – creating hills using the Landscape tool](img/B03679_07_09.jpg)

Go back to the **Lit** mode, build the level, and then take a look at it. Adjust any lighting in the map so that it's lit up appropriately. Rebuild until you are satisfied with what you get.

Add trees and plants to make the area a little more realistic. I have downloaded a package from Marketplace that has some foliage to help me with this.

Go to Marketplace on the Unreal Start Page. Under **Environments**, look for free downloadable content called **Open World Demo Collection**. The following screenshot shows free **Open World Demo Collection** in Marketplace. After downloading the package, add it to the project that you are working on.

![Exercise – creating hills using the Landscape tool](img/B03679_07_10.jpg)

We now have a basic outdoor terrain for our map.

## Landscape creation options

After going through the preceding exercise, you now have a good idea about how landscaping in Unreal Engine 4 fundamentally functions. In this section, we will add to the skills we have acquired so far and learn how to adjust or utilize features/functions of the Landscaping tool that is available to us.

### Multiple landscapes

It is possible to have multiple landscapes in the same map. This allows you to split the creation process into different layers. If you have more than one landscape in the map, you will need to select a layer before modifying it.

### Using custom material

You can import any material you want to use for the landscape; you can make your own grass, crops, sand texture, and so on. Since the custom material is mostly used for large areas of the map, it is good to bear in mind that you need to keep the material repeatable and optimized.

### Importing height maps and layers

Why do we use height maps in landscaping? These allow a quicker and more precise way to create elevations/troughs in the Unreal Editor. For example, we can use a height map to store elevation information for a mountain that is 3000m in height and of a certain diameter. When we import the height map, the terrain is automatically shaped according to it. It is definitely a time-saving method that helps us create more precise landscape features without having to click, click, click to sculpt.

Height maps and layers can first be created externally using common tools, such as Photoshop, World Machine, ZBrush, and Mudbox by artists. Detailed instructions need to be followed to ensure the successful importation of the height map. This can be found in the Unreal Engine 4 documentation at [https://docs.unrealengine.com/latest/INT/Engine/Landscape/Custom/index.html](https://docs.unrealengine.com/latest/INT/Engine/Landscape/Custom/index.html).

### Scale

The **Scale** settings determine the scaling of the landscape. We have used X: 100 and Y: 100 to give us the area of the land that this landscape will cover. The Z value is kept as 100 to provide some height to create elevation.

### The number of components

A component is the basic unit for rendering and culling. There is a fixed cost that's associated with the overall number of components; hence, it is capped at 32 x 32\. Going beyond this value would affect the performance of your game level.

### Section Size

**Section Size** determines how large each section is. It determines how the landscape is divided up. Large sections mean fewer overall components because the pie is divided into larger chunks. Fewer chunks to manage indicate a lower overall CPU cost.

However, a large section is not as effective when managing the LOD as compared to a smaller section. When there are smaller sections, we also get smaller component sizes (when the pie is of the same size, cutting it into smaller chunks indicates that you have less on your plate if you take one chunk). Since components are the basic unit used for culling and rendering, this means quicker responses to LOD changes due to the reduced area. LOD determines the number of vertices that need to be calculated. If LOD is more effective, we have fewer calculations to do, and the CPU cost is more optimized with smaller sections.

The catch here is balancing the size of the sections to avoid having too many components to go through and too few components might result in poor LOD management.

### Note

**Sections Per Component**

You have options ranging from 1 x 1 or 2 x 2 sections per component. What this means is that you have the option of having either one or four sections in each component. Since a component is the most basic unit in rendering and culling, for the 1 x 1 section, you can have one section rendered at the same time. For 2 x 2 sections per component, you can have four sections rendered at the same time. To limit the number of calculations needed to render a component, the size of each section should not be too large.

# Introducing cinematics

Cinematics were developed largely for motion pictures, films, and movies. Today, we apply cinematic techniques to non-interactive game sequences, known as **cut scenes**, to enhance the gaming experience. The overall gaming experience has to be designed with cut scenes in mind as they usually fulfil certain game design objectives. These objectives are often slotted in between gameplay to enrich the storytelling experience in games.

Very much like shooting a movie, we would need to decide what kind of shots need to be taken, which angles to shoot from, how much zooming is needed, how many cameras to use, and what path the camera needs to take in order to develop a motion picture sequence of our object/objects of focus. The techniques employed to create this clip are known as **cinematic techniques**.

So, in this chapter, we will first go through a few key objectives that explain why cinematics are needed in games, and you learn a couple of simple cinematic techniques that we could use. You will also learn about the tools that Unreal Engine 4 offers to apply the techniques we have learned in order to create appropriate cinematic sequences for your game.

Cinematic techniques are created by cinematic experts who focus on creating cut scenes for your games. Alternatively, you could also engage a cinematic creation contracting company to get this done for you professionally.

# Why do we need cut scenes?

When a game is designed, a fair amount of the game designing time is put into designing how players interact with the objects in the game and how this interaction can be made fun. The interactive portion of the game needs to be supplemented and cut scenes can help fill the gaps.

Cut scenes can be employed in games to help designers tell a story when you are playing the game. This technique can be employed before the game begins to draw the players into the mission itself and justifies why a mission has to be accomplished for the player. This helps the player to understand the storyline, create meaning for their actions, and draw the player into the game.

Another objective of cut scenes can be to highlight key areas in the game in order to give the players a glimpse of what to expect and provide subtle hints to successfully win the game. This information would be useful, especially in difficult to beat game levels or when the player is meeting the chief monster in the game.

Game designers also sometimes use cut scenes to reward players after a difficult battle. They amplify the effect of their success and play out the happy ending of their win in order to create positive emotions in the players. I am sure that there are endless creative ways to utilize cut scenes in games and how we could positively include them to enhance the gaming experience.

However, it is necessary to ensure that the use of cut scenes is justified well because cut scenes actually take the control of the game away from the player. Games are expected to be interactive, and we do not want to convert this into a passive multimedia experience when there are too many cut scenes.

Keeping these basic game design objectives in mind, let's now explore some technical cinematic fundamentals that will provide you the groundwork to design your own cinematics in games.

# Cinematic techniques

The camera is the main tool that's used to create effects for cinematics. You can achieve various cinematic effects by adjusting the camera functions and finding/moving the camera to a good spot to capture a significant key object(s) of interest. This section will provide some technical terms that you can use to describe to your coworker/contractor how a particular cinematic sequence should be recorded.

## Adjusted camera functions

Here are some commonly used functions that you can adjust on a camera to capture a scene.

### Zoom

Zooming in on an object gives you a closer view on the object; providing more details about it. Zooming out takes your view further away from the object; it provides a perspective for the object with regard to its surroundings.

Zooming is achieved by adjusting the focal length of the camera lens; the camera itself stays in the same position.

### Field of view

**Field of view** (**FOV**) is the area that is visible from a particular position and orientation in space. FOV for a camera is dependent on the lens and can be expressed as *FOV = 2 arctan(SensorSize/2f),* where *f* is the focal length.

For humans, FOV is the area that we can see without moving our head. The horizontal FOV kind of ends at the outer corner of the eye, as shown in the following image, which is about 62 degrees to the left-hand side and right-hand side (source: [http://buildmedia.com/what-are-survey-accurate-visual-simulations/](http://buildmedia.com/what-are-survey-accurate-visual-simulations/)):

![Field of view](img/B03679_07_11.jpg)

What this means is that whatever is outside this FOV is not visible to the entity.

### Depth of field

**Depth of field** (**DOF**) is best expressed as a photo, such as the following one, where only the object of interest is very sharp and anything behind is it blurred. In the following image (source: [http://vegnews.com/articles/page.do?catId=2&pageId=2125](http://vegnews.com/articles/page.do?catId=2&pageId=2125)), the gyoza/dumplings appear sharp and beyond these, the bowl/bottle is blurred. The small DOF in the photo allows the foreground (gyoza) to be emphasized and the background to be de-emphasized. This is a very good technique to bring visual attention to objects of interest in photography as well as in cinematics.

![Depth of field](img/B03679_07_12.jpg)

DOF is also known to provide an effective focus range. The method to determine this range is to measure the distance between the closest object and farthest object in a scene that appears to be sharp in an image. Although a lens is made to focus on one distance at a time, the gradual decrease in sharpness is difficult to perceive under normal viewing conditions.

## Camera movement

In filming, the camera is positioned at different angles and locations, and the camera moves with the actor/vehicle and so on. This camera movement can be described using some of the terms here.

### Tilt

The camera is moved in a similar way to how you nod your head. The camera is pivoted at a fixed spot, and turning it up/down is known as tilting. The following figure shows the side view of the camera with arrows illustrating the tilting:

![Tilt](img/B03679_07_13.jpg)

### Pan

The camera is moved in a similar way to how you turn your head to look to the left-hand side and the right-hand side. The camera is pivoted at a fixed spot and turning it to the left-hand side/right-hand side is known as **panning**. This figure shows the top view of the camera with arrows demonstrating how panning works:

![Pan](img/B03679_07_14.jpg)

### Dolly/track/truck

A dolly moves the entire camera toward or away from the object. It is quite similar to zooming in/out since you also going closer/further to the object, except that dollying moves the camera along a path toward/away from the object.

Trucking moves the camera sideways, that is, to the left-hand side or right-hand, along a track. Trucking is often confused with panning. The entire camera moves in trucking, but in panning, the camera stays in a fixed location and only the lens is swept to the left-hand side/right-hand side. Tracking is a specific form of trucking as it follows an object of interest in parallel. The following figure shows the back view of a camera dollying along a path:

![Dolly/track/truck](img/B03679_07_15.jpg)

### Pedestal

Pedestal is the moving of the camera up and down a vertical track. The following figure illustrates the camera moving up and down a vertical track:

![Pedestal](img/B03679_07_16.jpg)

## Capturing a scene

When capturing a scene, the overall scene is what matters most. You need to keep certain things in mind, such as what comprises the scene and its lighting; what you select determines how impactful the cut scene is. Here are a few factors that need to be addressed when composing a good cut scene.

### Lighting

Light affects how a scene shows up in photo/cut scene. We need to have the right lighting in place to capture the mood of the scene.

### Framing

Framing decides how the shot needs to be taken. Everything in the frame is important and you should pay attention to everything that is within the frame. How each shot transitions to the next also needs to be considered when creating a cut scene.

#### Some framing rules

The framing rules are as follows:

*   Make sure the horizontals are level in the frame and the verticals are straight up along the frame.
*   The Rule of Thirds. This rule divides the frame into nine sections. The points of interest should occur at one-third or two-thirds of the way up (or across) the frame rather than in the center. For example, the sky takes up approx. Two-thirds of this frame.
*   Strategic empty spaces are provided in front, above, or behind the subject to allow space for the subject to move into/look into.
*   Avoid having half an object captured in the frame.

#### Shot types

Here are some terms used to describe shots that can be taken for the frame:

*   **Extreme Wide Shot (EWS) /** **Extreme Long Shot (ELS)**: This shot puts the subject into the environment. The shot is taken from a distance so that the environment around the subject can be seen. This type of a shot is very often used to establish a scene.
*   **Wide Shot (WS) / Long Shot (LS)**: In a wide or long shot, the subject takes up the full frame. The subject is in the frame entirely with little space around it.
*   **Medium Shot (MS)**: The medium shot has more of the subject in the frame and less of the environment.
*   **Close Up Shot (CU)**: The subject covers approximately half the frame. This increases the focus on the subject.
*   **Extreme Close Up Shot (ECU)**: The camera focuses on an important part of the subject.

### Shot plan

This is a plan that describes how the scene will be captured. It also describes how many cameras to use, the sequence in which the cameras come on, and the kind of shots that need to be taken in order to play out the required effect for the scene.

# Getting familiar with the Unreal Matinee Editor

The Unreal Matinee Editor is similar to nonlinear video editors, so it is quite easy to pick up if you already have experience using software such as Adobe Flash. Creating keyframes for cameras and moving them along paths combined with modifying camera properties creates the matinee/cut scene for games. Additionally, you can also make or convert static objects to become dynamic and then animate them using this Matinee Editor.

# Exercise – creating a simple matinee sequence

Now, let's get hands-on and create a simple matinee sequence for your game. The plan is to showcase the area that we created at the beginning of the game. We will start with an extreme wide shot taken from the front of the house. We will use the dolly to take the camera toward the large windows in the dining area, into the kitchen area, and then the fireplace. Then, using the second camera, from the corner of the room, we will move toward a running guy and focus on his face.

Create a new matinee sequence from the ribbon, as shown in the following screenshot. Click on **Matinee** and select **Add Matinee**:

![Exercise – creating a simple matinee sequence](img/B03679_07_17.jpg)

This opens up the Matinee Editor, as shown in the following screenshot:

![Exercise – creating a simple matinee sequence](img/B03679_07_18.jpg)

To create the first camera, we will right-click on the **Tracks** area and select **Add New Camera Group**:

![Exercise – creating a simple matinee sequence](img/B03679_07_19.jpg)

Going back to the map, you can see a small window in the corner of the map that shows what the camera is looking at. This screenshot shows where our first shot starts:

![Exercise – creating a simple matinee sequence](img/B03679_07_20.jpg)

To create the next key where the next shot has to be taken, go back to the **Camera1** track, click on the small red arrow at 0.0 in the **Movement** row, and hit *Enter*. This duplicates the key. Press *Ctrl* and click and drag the red arrow to 2.00\. This screenshot shows how to do it correctly:

![Exercise – creating a simple matinee sequence](img/B03679_07_21.jpg)

Now, click on the red arrow at 2.00 and go back to **Camera1** in the map. Right-click on it and select **Pilot 'Camera Actor1'**, as shown in this screenshot:

![Exercise – creating a simple matinee sequence](img/B03679_07_22.jpg)

Move the viewport to the position you want to have the second keyframe in. This screenshot shows the position of the second keyframe camera:

![Exercise – creating a simple matinee sequence](img/B03679_07_23.jpg)

When the viewport is positioned, as shown in the preceding screenshot, click on the icon in the top-left corner of the viewport to stop the pilot mode in order to fix the keyframe here. The location of the icon is shown here:

![Exercise – creating a simple matinee sequence](img/B03679_07_24.jpg)

Following the shot plan we decided on, I have moved **Camera1** along the path up to the fireplace. To add the second camera, repeat the steps to create a new camera group and name the new camera as `Camera2`.

Now, move the first keyframe to the end of Camera1's final keyframe timeline. For me, this is set at **8.50s**; I moved the camera to the corner of the room, as shown in the following screenshot:

![Exercise – creating a simple matinee sequence](img/B03679_07_25.jpg)

Repeat the steps to create keyframes for **Camera2**, move it along the path toward the running man, and then focus on the running man's face.

Now, we have two cameras that need to be told which one is playing at which part along the timeline. To do so, we need to create a new director group. The director group will dictate which camera is on air and what will be showing on screen. Go back to **Tracks** in the Matinee Editor. Right-click and select **Add New Director Group**, as shown in this screenshot:

![Exercise – creating a simple matinee sequence](img/B03679_07_26.jpg)

This creates a **Director** track above the camera tracks. Select the newly added **Director** track at 0.00, go to the ribbon at the top, and select **Add Key**, as shown in this screenshot:

![Exercise – creating a simple matinee sequence](img/B03679_07_27.jpg)

The contextual menu will request that you select **Camera1** or **Camera2**. In this case, select **Camera1**. This fills up the entire duration of the cinematics. To create a key at 8.50s where **Camera1** and **Camera2** overlap, click on the **Director** track again and select **Add Key**. This time round, select **Camera2**. Move this key to **8.50**. This screenshot shows where the cameras are set up so that they can play correctly:

![Exercise – creating a simple matinee sequence](img/B03679_07_28.jpg)

Finally, we are ready to play the cut scene. To tell the game to play the cut scene when starting the game, we need to use Blueprint. I hope you still remember how to use the Blueprint Editor. Click and open the Level Blueprint. Add the **Event BeginPlay** node and right-click and search for **Play**. Select the **Play Matineee Actor** option and link the nodes, as shown in the following screenshot. Now, save and play the level. You will see the entire matinee play before you control the player in the level.

![Exercise – creating a simple matinee sequence](img/B03679_07_29.jpg)

# Summary

We covered terrain creation and matinee creation in this chapter. I hope you were able to enhance the game level with the new skills we explored.

Terrain manipulation covers large areas of a map; hence, we also went through the factors that affect the playability of the map. We also went through a simple exercise to create the outdoor terrain of our map with some hills and a lake.

Matinee creation involves a lot more technical planning before we start playing around with the editor itself. The use of the editor is pretty simple as it is similar to current video editors in the market. The techniques to create good cinematics were covered to help you understand their backgrounds a little better.

This is the last chapter of the book and the final summary. I sincerely hope that you enjoyed reading this book and had fun playing around with Unreal Engine 4\. Lastly, I would like to wish you all the best in creating your own games. Do keep at it; there is always more to learn and other new tools out there to help you create what you want. I am sure that you love creating games; if not, you would not have survived this boring book right to the end. This book only serves to introduce you to the world of game development and shows you the basic tools to create a game using Unreal Engine. The rest of this journey is now left to you to create a game that is fun. Good luck! Don't forget to drop me a note to let me know about the games you create in the future. I am waiting to hear from you.