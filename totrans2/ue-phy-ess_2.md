# Chapter 2. Physics Asset Tool

PhAT stands for Physics Asset Tool. Imagine you drop a dice. Based on the reality of the physical rules in the world we live, the dice drops differently on wood, stone, glass, and carpet. The same can be said for glass, plastic ball, and enemy body (more complex). Using PhAT, you will learn how to simulate reality based on the physical rules that exist in the game world. This is kind of animated, but with more logics and tools.

In this chapter, you will first learn about the most important items in PhAT and then use them in a practical example.

# Navigating to PhAT

Before we start working in **Unreal Editor**, we will need to have a project to work with. Perform the following steps:

1.  First, open **Unreal Editor** by clicking on the **Launch** button from Unreal Engine Launcher.
2.  Start a new project from the **Project** browser by selecting the **New Project** tab. Then, select **Third Person** and make sure that **With Starter Content** is selected. Give the project a name (`phat_test`). Once you are finished, click on **Create Project**.

After everything comes up on your screen, in **Content Browser**, locate the **ThirdPersonBP** folder and click on the **Character** folder. Find **ThirdPersonSkelMesh** and then right-click on the list and select **Create** | **Physics Asset**, as shown in the following screenshot. Then, click on **Ok** in the **New Asset** window.

### Note

**ThirdPersonSkelMesh** can be found by the name **SK_Mannequin** in the newer version of Unreal Engine 4.

![Navigating to PhAT](img/image00201.jpeg)![Navigating to PhAT](img/image00202.jpeg)

Now, if everything works fine, you will find your selected mesh in **Physics Asset Tool** or **PhAT**. This kind of editor allows you to put some custom controllers on parts of your character's mesh based on your bones. These controllers are sensitive to physical aspects, such as world gravity, movements of other parts, rotations on each part, and collisions between parts.

![Navigating to PhAT](img/image00203.jpeg)

We will select a human like mesh. Imagine that this is an enemy and a player should shoot this object. After shooting, on hit, the enemy character falls on the surface. For simulating this scenario, set your camera as shown in the following screenshot, and click on **Simulate** from the top menu:

![Navigating to PhAT](img/image00204.jpeg)

Then, set your camera to this view and click on **Simulate**.

In the preceding screenshot, you can see that the character falls, but the way it falls is not natural in many ways. The following image is a screenshot of this action. Here, you can see that the hands and backbones are in an unreal position. PhAT can solve all these problems by customizing each bone movement based on physical rules (such as gravity).

![Navigating to PhAT](img/image00205.jpeg)

The problem persists on the back and hands as the mesh falls.

Click on **Simulate** again to go back to the normal mode.

## The PhAT environment

PhAT has a couple of gadgets and sections. Here, we will cover most of the commonly used sections:

*   **Hierarchy**: On the right-hand side of the stage, there is a list of bones entangled with meshes on the character. This is used for animation purposes. Some bones have a bold font, whereas some don't. When you assign a physical asset to a bone, Unreal Engine automatically makes it bold. This is useful for addressing the bones involved with a glance over them. You can turn it on/off using the **Windows** button from the top menu.
*   **Save**: This saves the current asset on the model.
*   **Find in CB**: This locates the current asset in **Content Browser**.
*   **Simulate**: This runs the simulation.
*   Little arrow in front of **Simulate**: This changes the simulation mode between the **Real** and **No Gravity** types of simulation.
*   **Selected Simulation**: When you select one bone and then hit simulate, the only selected bone and its branch (or children) will react to simulation. Let's take a look at an example:

    1.  Navigate to **Hierarchy** on the right-hand side and select **Bones with Bodies**. This just shows the bones with assets.![The PhAT environment](img/image00206.jpeg)
    2.  Now, right-click on **lowerarm_r** in the bones list and then on **Selected Simulation** to make it active:![The PhAT environment](img/image00207.jpeg)
    3.  Then, click on **Simulate**. As you can see, only a part of the left hand shows some movement, but the rest of the body does not move:![The PhAT environment](img/image00208.jpeg)
    4.  Click again on **Simulate** to go back to the normal scene.
    5.  Now, click on **upperarm_r** in **Hierarechy**. Then, click on **Selected Simulation** again. As you can see, now the entire hand is part of the simulation, but the body is not moving:![The PhAT environment](img/image00209.jpeg)
    6.  Later when we start adding assets, this simulation type will be useful and handy. For now, just try experimenting with more bones and simulate your selection.

*   **Body Mode**: This selects the different types of visual presentation on the body.
*   **Translation**, **Rotate**, and **Scale**: These are tools for editing each asset.
*   **Details**: In this section, you will find some related properties for each asset. When you select different assets in **Hierarchy**, the properties related to the selected asset are displayed here. You can turn this on/off using the **Window** button from the top menu:![The PhAT environment](img/image00210.jpeg)

## The PhAT example and experience

In order to create a new asset on each bone, it's better to clean our object from the previous ones. There is a quick method to perform this, that is, click on **Edit** from the top menu bar and then on **Select All Objects**. Now, click on *Delete* on your keyboard. Try this and then click on **Undo** from the same menu. You can also try another useful method, which is explained in the next section.

### Deleting current assets

Perform the following steps to delete current assets:

1.  Set your camera as shown in the following screensahot:![Deleting current assets](img/image00211.jpeg)
2.  Now, click on **Simulate** and wait until the body fully lies down on the ground, as shown in the following screenshot. It is possible that your result would be a bit different, but no problem. You can try and click on **Simulate** to reach the closest form of the image:![Deleting current assets](img/image00212.jpeg)
3.  Now, press *Ctrl* and right-click to hold the head and move your mouse up not fast, but not slow either. It looks like the mesh is hanging to your mouse pointer, as shown in the following screenshot:![Deleting current assets](img/image00213.jpeg)

    Without releasing the right mouse button, move your mouse in the left and right direction over the stage and watch the elements of the mesh connecting and moving with each other. You will soon discover a red line that indicates the path, in which your mouse just moved, as shown in the following screenshot:

    ![Deleting current assets](img/image00214.jpeg)

    If you release the mouse button, you can use **Simulate** to switch back to the normal mode and create it back again.

4.  Now leave the mouse, and the character will fall on the surface. Grab all the other parts of the character and experience the same movement with different mouse speed. This is kind of fun with PhAT and also shows how the body is connected with bones. You can track this in **Hierarchy**.
5.  Click on **Simulate** to go back to the normal mode and right click on **spine_03** in **Hierarchy**. As you can see in the following screenshot, this part is selected in a different color. Now, click on **Simulate**. Then, grab this part (*Ctrl*, right-click and hold) and move it. Select **Simulate** again to return to normal. This time, select **foot_r** and try to simulate and grab it and move the character. As you can see, when you grab the body and move, other parts create a kind of rhythm, which depends on your mouse movement on the screen. This is caused by the default physic assets of each part.![Deleting current assets](img/image00215.jpeg)
6.  Now, right-click on **clavicle_r** and select **Delete All Bodies Below**, as shown in the following screenshot. It looks like nothing seriously happened here, but when you click on simulate, boom! The left arm remains still, and a totally different behavior is displayed. Stop the simulation, right-click on **clavicle_l**, select **Delete All Bodies Below**, and click on **Simulate** again. Now, both hands appear still, and it doesn't behave like before. Select **Undo** to track the difference between the previous and current behavior.![Deleting current assets](img/image00216.jpeg)

    This practice is essential to understand what PhAT can provide us in the game. When you select **Delete All Bodies Below**, you can delete all the previous physic assets on the bones that will connect to your selected bone. You can track this in **Hierarchy**. For example, **clavicle_r** is connected to **upperarm_r**, same logic (image it as chain of bones), **upperarm_r** is connected to **lowerarm_r**. And, **lowerarm_r** is connected to hand_r. Here, hand_r is last member of these series of the bones. (this is important for the last bone). If you apply any physical rule to **clavicle_r**, the energy flows through the other bones until the end: **hand_r**. When you select **Delete All Bodies Below**, you can actually delete this path to navigate energy. This is why the hands look solid in simulation.

    This is a basic understanding of what PhAT is used for. You will learn how to design certain assets that guide and reflect the physical input from the environment (such as gravity or collision) through the bones of the character.

    To see how these assets will function and navigate energy, select **top bone to down bone (or bones)** in **Hierarchy** and check whether **clavicle_r** is connected to **spine_03**. The bigger bone sends/navigates the physic assets/energy to **clavicle_r**.

    ![Deleting current assets](img/image00217.jpeg)
7.  Try the same with **thigh_l** and **thigh_r**. Here, **pelvis** is the higher bone in **Hierarchy**.

### Adding and customizing current assets

Now, as we experience the effect of PhAT assets on the bones and also have an understanding of how bones are connected and guide the energy, it's time to create our own physics assets over the body. Refer to the last part, try to delete the different bones asset, and simulate the scene. Then, select **Undo** and **Redo**. This is a creative practice for game designers.

Now, click on **Edit** from the top menu bar, select **All Objects**, and press *Delete* on your keyboard. Now, we can delete all the physic assets on the body. Then, select **All Bones** in **Hierarchy**:

![Adding and customizing current assets](img/image00218.jpeg)

You will find a detailed list of all the bones in **Hierarchy**. It seems that it has more bones with more detailed names. We can select some of them and establish our physic assets over them in this chapter; you can experiment with different bones and get more realistic results.

1.  Change your camera as follows. Then, click on **Simulation**.![Adding and customizing current assets](img/image00219.jpeg)

    1.  Nothing will happen. Now, click on **Body Mode** at the top and select **Constraint Mode**. The screen will go blank.
    2.  Navigate back to **Body Mode** by clicking once again on **Constraint Mode** and select **Body Mode**. You will see the body again. Now, click on **pelvis** in **Hierarchy**, right-click on it, select **New Body**, and click on **Ok** in the **New Asset** box:![Adding and customizing current assets](img/image00220.jpeg)
    3.  Now, click on **Body Mode** at the top and select **Constraint Mode**. Something will appear on the screen this time.
    4.  This is your first asset. Now, click on **Simulate** to try it in the real world. Also, drag and try to move it around (holding *Ctrl* and the right mouse button). This is mostly similar to how you check your asset functionality, create and edit, choose **Simulate** and then test with mouse.

    ![Adding and customizing current assets](img/image00221.jpeg)
2.  Switch to **Body Mode** at the top to view the entire body. Then, select **thigh_l** in **Hierarchy** and create a new body for it. Now, view the changes. Click on **Simulate** and try to move it with your mouse. As you can see, the movement is between two parts: **pelvis** and **thigh_l**, which is represented in bold fonts in **Hierarchy**.![Adding and customizing current assets](img/image00222.jpeg)

    1.  Switch to **Bones With Bodies** in **Hierarchy** to have just these bones in the list. Now, switch to **Constraint Mode** at the top. Now, you can see the new physic assets that you put over your bones. At this point, we have two.![Adding and customizing current assets](img/image00223.jpeg)
    2.  Now, click on **thigh_l** and then on **Selected Simulation** at the top to make it active. Then, click on **Simulate**. As you can see, only the selected part will move. Test this with the mouse. It does not move naturally, and, compared to the movements of a real human body, it is absolutely unnatural.![Adding and customizing current assets](img/image00224.jpeg)
    3.  Now, we want to fix this so that it can move and work based on real physics. So, stop the simulation and click on **thigh_l** in **Hierarchy**. Under **Details**, locate **Angular Limits** and change the options, as shown in the following screenshot:![Adding and customizing current assets](img/image00225.jpeg)
    4.  Now, change your camera view, as shown in the following screenshot, and click on **Rotation** at the top.
    5.  Once you select **Rotation**, you will see a colorful bold arc shape around your selection. Use this to click and rotate the physic asset. Also, a blue triangle like shape will appear. This indicates the maximum range in which your bone can play and move.
    6.  Click on the green bold arc around your selected point, press *Alt* (*very important*), and rotate the point so that it points to the ground, as shown in the following screenshot:![Adding and customizing current assets](img/image00226.jpeg)
    7.  Now, click on **Simulate** and change your rendering options so that it looks similar to following screenshot:![Adding and customizing current assets](img/image00227.jpeg)
    8.  Basically, changing **MeshRender Mode** to **None** allows you to see the main physic assets behavior during the test. Also, setting **ConstraintRender Mode** to **All Limits** allows you to observe the movement of small tiny colorful lines (which are indicated by the arrows), as shown in the following screenshot. Understanding and controlling these lines are the key to the soft physic assets on your mesh. The way they move, the area they cover, and the other details are located in the **Detail** section.![Adding and customizing current assets](img/image00228.jpeg)
    9.  Stay in the simulation mode and move the object a couple of times. It moves from left to right and always stays with the blue tiny line in the range of the blue triangular shape. Now, stop the simulation. In **Details**, inside **Angular Limits**, change **Swing2Limit Angle** from `45` (the current one) to `12.0` and run the simulation again:![Adding and customizing current assets](img/image00229.jpeg)
    10.  Now, the movements become tighter. Also, the blue triangle appears smaller. Change your view and then **MeshRender Mode** to **Solid**, as shown in the following screenshot. Then, check the movements of the left leg or the movement of **thigh_l** under your new physic asset (in a better way).![Adding and customizing current assets](img/image00230.jpeg)
    11.  Now, let's change **Swing2Limit Angle** from `12` (the current one) to `45` and run the simulation again.

3.  In **Hierarchy**, switch to **All Bones** and then to **Body Mode** from the top menu. Finally, select **thigh_r** from the bone list, right-click on it, and select **New Body**. This is the second leg.

    1.  Switch to **Constraint Mode** at the top. Similar to the last bone, make the rotation and details look similar to the following screenshot. Also, turn off **Selected Simulation** from the top menu:![Adding and customizing current assets](img/image00231.jpeg)
    2.  Change **MeshRender Mode** to **Solid** and check the behavior of the body when you move it on the stage. The legs move much more naturally than before, but it needs to move in different angles as well, as shown in the following screenshot:![Adding and customizing current assets](img/image00232.jpeg)
    3.  To apply more details, play with rotation and angles to reach the following screenshot:![Adding and customizing current assets](img/image00233.jpeg)
    4.  Try the same process with the **spine_01**, **spine_02**, and **spine_03** bones. Then, fix the limitations, angles, and work on **clavicle_l** and **clavicle_r**. Finally, work on **neck_01** and **head**. This way, you can grow your physic assets on the body in a properly organized way. You can test the overall movement and, when everything looks fine, increase the assets on smaller bones (such as fingers).

4.  As you go further, you should be aware of a correction that needs to be performed at the end of each series of bones. Similar to fingers, head, and toes, this is simply related to physical energy. When the body moves, based on the physic assets and the way you edit these, the body reflects and guides the movements from one bone to another. If there is no bone after the current one, such as **head** or the last part of the finger, it sometimes ends in an unusual vibrate-like movements over the bone during the simulation and the real game. To guide this energy in a controlled way, click on the bone in **Body Mode**, expand the **Physics** section in **Details**, set **Angular Damping** to `5` and **Linear damping** to `1`, and simulate. Now, check your movements and change these numbers until the problem is fixed.![Adding and customizing current assets](img/image00234.jpeg)

# Summary

Now, you can use PhAT to make your character move and reflect, which is similar to the real world. Keep in mind that working on PhAT is presenting the game story over the character behavior. It is an art because all the details and time that you spend here are directly monitored and tried by the player (perhaps hundreds of times while playing the game). It is good practice to imagine a character with some special behaviors, like movement, impact on other objects like walls, guns or fire, in real world; and then, try to recreate that behavior using Unreal Engine. It is a physical simulation with many details which we have gone through in this chapter. As a game designer, more practice will guarantee high quality results.

You can import the rigged mesh from the 3D software and apply the same physic assets to them. The mesh should be rigged in a proper way, which is similar to what we used as default in this chapter. Try the online sources by searching Unreal Engine 4 PhAT on YouTube. Also, if you practise the example, you can use the resources of Unreal Engine 3 and tuts in this area.