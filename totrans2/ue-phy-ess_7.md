# Chapter 7. Creating a Vehicle Blueprint

In this chapter, we will create a working **Vehicle Blueprint** from scratch using the default assets provided by Unreal Engine 4 with the **VehicleGame** project example as well as using assets which we will be creating by ourselves. First, we will start with an overview of what **Vehicle Blueprint** will be composed of and then move on to the specifics on how to create the different blueprints for all of their aspects. Following the overview, we will cover the following topics:

*   Creating Vehicle Blueprints
*   Editing Vehicle Blueprints
*   Setting up user controls
*   Testing the vehicle

There is a lot of content to cover in this chapter, so let's get started.

# Vehicle Blueprint – a content overview

A vehicle in Unreal Engine 4 contains a number of different types of assets:

*   A **Skeletal Mesh**
*   A **Physics Asset**
*   An **Animation Blueprint**
*   A **Vehicle Blueprint**
*   One or more **Wheel Blueprints**
*   A **Tire Type Data Asset**

Let's start by creating the necessary game project so that we have access to a **Vehicle Skeletal Mesh** by default, and we don't have to create our own in a third-party 3D modeling program. To do so, let's open the Epic GamesUnreal launcher and navigate to the **Learn** tab. Here, scroll down to the **Example Game Projects** section and find the **Vehicle Game** project template. Select this project and then the **Download** option:

![Vehicle Blueprint – a content overview](img/image00322.jpeg)

Once successfully downloaded, we can create the project by navigating to the **Library** tab of the Unreal Engine launcher, scroll to the very bottom of the page to the **Vault** section, and select the **Create Project** option for **Vehicle Game**:

![Vehicle Blueprint – a content overview](img/image00323.jpeg)

When we select the **Create Project** option, it will ask us for a name of the project. Let's call this project `Vehicle_PhyProject`. After the Unreal Engine launcher creates the vehicle project for us, we should see it available in our list of projects in the **My Projects** section of the **Library** tab. Now, double-click on the project image to open the Unreal Engine 4 editor for this project. By default, this project contains all the assets necessary to create a working **Vehicle Blueprint**, such as a **Skeletal Mesh**, **Wheel Blueprints**, a **Vehicle Animation Blueprint**, and so on, but we will only use the **Skeletal Mesh** that the project provides, and we will create every other aspect of the **Vehicle Blueprint** from scratch step by step.

Now that we have successfully created the **Vehicle Game Project**, feel free to explore some of the content that it contains and play the game in the **Desert Rally Race** level to see how our final result will be for this chapter. Once we are satisfied exploring the game project, let's begin by creating a new folder in **Content Browser** that will house all of our content for this chapter. First, navigate to **Content Browser** and highlight the **Content** folder at the very top of the content hierarchy in the top-left corner of the browser. Once highlighted, we can either right-click on the **Content** folder, select the **New Folder** option, or left-click on the **Add New** drop-down menu and select **New Folder**. Name this folder `VehicleContent`. With the folder in place, we will now navigate to the **Vehicles** folder, the **VH_Buggy** folder and then to the **Mesh** folder; this folder contains the **SK_Buggy_Vehicle Skeletal Mesh** that we need to begin this lesson. Let's left-click and drag the **SK_Buggy_Vehicle** asset to our **VehicleContent** folder and select it to create a copy. Name this copy `SK_Buggy_NewVehicle`. If we want to create our own **Skeletal Mesh**, here are some of the things we should keep in mind.

The basic, bare minimum, art setup required to create a proper vehicle is just a **Skeletal Mesh**. The type of vehicle will dictate how complicated an art setup we will need, and special considerations may need to be given to the suspension. For example, a tank does not require a special suspension setup, whereas a dune buggy (such as the one in the **Vehicle Game** project example) will require additional joints to make the exposed components move in a believable way.

Some of the more important basic information we need to know about setting up our vehicle in a third-party art program (such as 3ds Max or Maya) is that we want the vehicle mesh to point to the positive *X* direction. Next, we will need to measure the radius of our wheels in centimeters for use in Unreal Engine 4 because we had discussed earlier in this book that Unreal Engine 4 uses centimeters as its unit of measurement, where 1 **Unreal** **Unit** (**uu**) is equal to 1 **centimeter** (**cm**). The minimum number of **Joints** required for a four-wheeled vehicle is 5:1 and 4 wheels; this will change depending on the number of wheels the vehicle has (remember what we discussed in [Chapter 5](part0055.xhtml#aid-1KEEU2 "Chapter 5. Physics Damping, Friction, and Physics Bodies"), *Physics Damping, Friction, and Physics Bodies*). The wheel and root joints should be aligned with the *X* direction looking forward and the *Z* direction looking upwards. By doing so, this will ensure that the wheel will roll on the *Y* axis and steer on the *Z* axis. All the other joints can be arranged as required, but it should be noted that things such as **Look At** nodes for the **Animation Blueprint** assume that the *X* direction is forward. To prevent visual oddities, the joints for our wheels should be accurately centered, as shown in the following screenshot:

![Vehicle Blueprint – a content overview](img/image00324.jpeg)

The visual mesh will not be used for collision detection; however, if the wheel mesh is off-center, it will look as if the wheel is broken and will be really noticeable due to motion blur.

For binding purposes, we can use either the standard smooth bind for Maya or the skin modifier for 3ds Max. Wheels should only have weights on one joint so that they can spin free with no odd deformation. For shocks and struts, we can get away with some fancy skinning, but it will require more thought on the Unreal Engine Editor side.

Lastly, vehicles are simply exported as **Skeletal Meshes** with no special considerations when you import the asset to Unreal Engine 4.

Now that we have our own copy of the **Skeletal Mesh** vehicle, we can create our **Physics Asset** for this vehicle by right-clicking on **SK_Buggy_NewVehicle** from our **VehicleContent** folder, selecting the **Create** option from the drop-down menu and then **Physics Asset**. Then, name this asset `PA_Buggy_NewVehicle` and leave all the setup options to their default values, as shown in the following screenshot:

![Vehicle Blueprint – a content overview](img/image00325.jpeg)

Now, double-click on the new **Physics Asset**, and we should see something similar to this screenshot in **PhAT**:

![Vehicle Blueprint – a content overview](img/image00326.jpeg)

We obtain this result because the **Physics Asset Tool (PhAT)** in Unreal Engine 4 attempts to wrap the vertices that are skinned to a joint as best as it can. **PhAT** does not currently have a way to effectively handle the recreation of the constraints that hold all the **Physics Bodies** together, so what we need to do is delete all the existing **Physics Bodies** in **Hierarchy** so that we can start building them from the root joint. By doing so, all of our constraints will be created correctly.

To do this, navigate to **Hierarchy**, press *Shift* and left-click on all the options, and press the *Delete* key; this will remove all the **Physics Bodies** from the asset:

![Vehicle Blueprint – a content overview](img/image00327.jpeg)

Starting with the root joint: **rootJNT**, let's create the **Physics Bodies** on the joints of our vehicles. Keep in mind that we only need a **Physics Body** on a joint that either needs to be physically simulated or affects the bounds of our vehicle. For our vehicle, a box shape for the root/main body and spheres for each of the wheels will serve us just fine, but we will add additional **Physics Bodies** to get the desired behavior that we want for other parts of the vehicle (such as the antenna).

For our **Buggy Vehicle**, we will have a total of 10 **Physics Bodies**. The end result should look similar to the following image:

![Vehicle Blueprint – a content overview](img/image00328.jpeg)

To accomplish this, we will first create the large bounding box that surrounds the main body of **Buggy Vehicle**. Select the **rootJNT** option from the **Hierarchy** panel on the right-hand side and then right-click and select the **Add Box** option. Make sure to use the **Translation** and **Scale** tools to shape the **Physics Body** box to match the shape of the body of the buggy as best as you can. Next, let's navigate to the **Collision** section of the **Details** panel and make sure that **Simulation Generates Hit Events** is set to `True`. Lastly, in the **Details** panel of the newly created box, **Physics Body**, make sure that **Physics Type** is set to `Default`.

Now, let's create the **Physics Bodies** for the four wheels of our vehicle. Each **Physics Body** will be of the same size and shape, and have the same properties associated to them. Let's select **F_L_wheelJNT** from the **Hierarchy** panel, right-click on this option, and select the **Add Sphere** option. Use the **Scale** and **Translation** tools to position the spherical **Physics Body** around the front-left wheel of the vehicle and change its **Physics Type** to **Kinematic**. Follow this process for the remaining three wheels: **F_R_wheelJNT**, **B_R_wheelJNT**, and **B_K_wheelJNT**. Lastly, let's navigate to the **Collision** section of the **Details** panel for each of the four wheels and make sure that **Simulation Generates Hit Events** is set to `True`.

![Vehicle Blueprint – a content overview](img/image00329.jpeg)

Moving on, let's now create the **Physics Bodies** for the front and back bumpers of the vehicle by right-clicking on **rootJNT** and selecting the **Add Sphyl** option, which will create a capsule-shaped **Physics Body**. Use the **Scale** and **Translation** tools to position this **Physics Body** around the front bumper of the vehicle and set its **Physics Type** to `Default`. Repeat this process for the back bumper of the vehicle as well. Lastly, let's navigate to the **Collision** section of the **Details** option for both. Make sure that **Simulation Generates Hit Events** is set to `True`.

Before we move on to creating the **Physics Body** for the antenna, let's first create the two box shapes for the left-hand side and the right-hand side suspensions for the buggy. With the **rootJNT** bone joint selected in **Hierarchy** on the right-hand side, right-click and select the **Add Box** option. Lastly, let's navigate to the **Collision** section of the **Details** panel and make sure that **Simulation Generates Hit Events** is set to `True`. Then, the scale and position of the box should look similar to the following screenshot:

![Vehicle Blueprint – a content overview](img/image00330.jpeg)

Now, create an additional **Box Physics Body** and shape and transform it so that it covers the other side of the buggy's suspensions. Then, set both their **Physics Type** parameters to `Default`. Next, let's navigate to the **Collision** section of the **Details** panel and make sure that **Simulation Generates Hit Events** is set to `True`.

Lastly, let's set up the **Physics Body** for the antenna of our vehicle so that we can simulate a responsive antenna when we move in our vehicle. To do this, select the **Antenna_01** option from the **Hierarchy** panel, right-click on it, and select the **Add Sphyl** option. Next, set **Physics Type** to `Default`, and set the following parameters in the **Details** panel for our antenna's **Physics Body**:

*   **Mass Scale**: Set this parameter to `0.01`
*   **Angular Damping**: Set this parameter to `10.0`
*   **Linear Damping**: Set this parameter to `3.0`

To finish off the **Physics Body** for our antenna, make sure that the **Antenna_01** joint is selected in the **Hierarchy** panel. Then, select **Constraint Mode** from the **Body Mode** drop-down menu:

![Vehicle Blueprint – a content overview](img/image00331.jpeg)

While in **Constraint Mode** and with the **Antenna_01** joint selected, set the following parameters in the **Details Panel**:

*   **Angular Swing 1 Motion**: Set this parameter to `Limited`
*   **Angular Twist Motion**: Set this parameter to `Locked`
*   **Angular Swing 2 Motion**: Set this parameter to `Limited`
*   **Swing 1 Limit** **Angle**: Set this parameter to `1.0`
*   **Swing 2 Limit Angle**: Set this parameter to `1.0`
*   **Swing Limit Stiffness**: Set this parameter to `500.0`
*   **Swing Limit Damping**: Set this parameter to `50.0`

With these changes in place, let's return to **Body Mode**, select **rootJNT** from the **Hierarchy** panel and then the **Selected Simulation** option, and select **Simulate** to see how our **Physics Bodies** are affected by the gravity of simulation:

![Vehicle Blueprint – a content overview](img/image00332.jpeg)

We can see that the wheels rotate in a strange manner that don't make too much sense in the context of a working vehicle for a racing game, but we will change this behavior later when we create the blueprints for the vehicle.

# Vehicle Blueprints – a section overview

In this section, we discussed the necessary components that make up a working **Vehicle Blueprint**, and we looked at the necessary details of how to create the **Physics Bodies** for our vehicle. Lastly, we used **PhAT** in Unreal Engine 4 to recreate the necessary **Physics Body** components of the **Buggy Vehicle** to establish a working **Physical Body** for the vehicle. Now that we have created the **Physics Asset** from the premade **Skeletal Mesh** that is created by default when working with the **VehicleGame** project, we can now move on and work on **Vehicle Blueprints**.

# Creating the Vehicle Blueprints

To create a new **Vehicle Blueprint**, let's first navigate to our **VehicleContent** folder in **Content Browser** and then right-click on an area that is empty. From the context drop-menu, we will select the **Blueprint Class** option, click on the drop-down **All Classes** menu, search for the **WheeledVehicle Pawn** class, and name this blueprint `BP_NewVehicle`.

![Creating the Vehicle Blueprints](img/image00333.jpeg)

The **WheeledVehicle Pawn** blueprint contains an inherited component called **VehicleMovement**. This component allows you to have more control over the wheels and the overall behavior for the vehicle.

Next, we will need to create two different types of **Wheel Blueprints** for our vehicle (one for the front wheels and one for the back wheels). To get this going, let's navigate to the **VehicleContent** folder in **Content Browser**, right-click on an area of **Content Browser** that is empty, and select the **Blueprint Class** option. Now, in the context sensitive drop-down menu, enter `Vehicle Wheel` to locate the **VehicleWheel Blueprint Object** class. We will create two different **VehicleWheel Blueprint** classes (one named `BP_FrontWheel` and another named `BP_BackWheel`).

In most cases, we will want to have at least two wheel types: that is, one wheel type that is affected by steering and another that is affected by the vehicle handbrake. Additionally, we can set differing radii, mass, width, handbrake effect, suspension, and many other properties to give our vehicle the handling we desire.

Now, we can move on and create the **TireType** data asset that we will need for our **VehicleWheel** blueprint. To create a new **TireType** data asset in **Content Browser**, we need to right-click on an area of the **VehicleContent** folder that is empty, select the **Miscellaneous** option and then the **TireType** option from the context sensitive drop-down menu that appears:

![Creating the Vehicle Blueprints](img/image00334.jpeg)

Let's name this asset `DA_TireType` and then right-click onthe asset to open **Generic Asset Editor**. The **TireType** data asset has only one single value: **Friction Scale**. This value not only affects the raw friction of the wheel, but also scales the value for how difficult or easy it is for a wheel to slide during a hard turn. There is a property slot in the **VehicleWheel** blueprint for the **TireType** data asset that we will use once the time comes.

Lastly, we have to create the **Animation** blueprint. We will use this to animate our **Buggy Vehicle**. To do this, navigate to **Content Browser** and then go to the **VehicleContent** folder. Now, in an empty area, right-click and select the **Animation** option from the drop-down menu and then select **Animation Blueprint**:

![Creating the Vehicle Blueprints](img/image00335.jpeg)

When we first create an **Animation Blueprint**, it will ask us for a **Target Skeleton** to use; we will select the **SK_Buggy_Vehicle_Skeleton** option from the **Target Skeleton** drop-down list. We also want to make sure that we select the **VehicleAnimInstance** option from the **Parent** **Class** context sensitive drop-down list and name this animation blueprint `BP_VehicleAnimation`. Before we move on and discuss how to edit the different blueprints and data assets we created to complete our vehicle, let's briefly discuss what animation blueprints are.

An **Animation Blueprint** is a specialized blueprint that contains graphs used to control the animation of a skeletal mesh. It can perform blending of animations, has direct control of the bones in a skeleton, and outputs a final pose for our skeletal mesh in each frame. The **Controller** directs the pawn or character to move based on the player input or decisions made based on what the game play dictates. Each pawn has a **Skeletal Mesh** component that references the **Skeletal Mesh** to animate and has an instance of an **Animation Blueprint**.

Through the use of its two graphs, the **Animation Blueprint** can access properties of the owning pawn, compute the values used for blending, state transitions, or driving **Anim Montages**, and can calculate the current pose of the skeletal mesh based on the blending of animation sequences and direct transformations of the skeleton from **Skeletal Controls**. When we work with animation blueprints, we have to keep in mind that there are two main components that work in correlation with one another to create the final animation for each frame. One is **Event Graph** that we can recognize from other blueprints. This is in charge of performing updates to values that can be used in **Anim Graph** to drive **State Machines**, **Blend Spaces**, or other nodes that allows you to blend between multiple animation sequences or poses that fire off notifications to other systems, thereby enabling dynamically-driven animation effects to take place.

There is one **Event Graph** in every **Animation Blueprint** that uses a collection of special animation-based events to initiate sequences of actions. The most common use of **Event Graph** is to update the values used by **Blend Spaces** and other blend nodes to drive animations in the **Anim Graph**.

The **Anim Graph** is used to evaluate the final pose of a skeletal mesh for the current frame. By default, each **Animation Blueprint** has an **Anim Graph**. This graph can contain animation nodes that are placed in it to sample animation sequences, perform animation blends, or control bone transformations using **Skeletal Controls**. The resulting pose is then applied to our **Skeletal Mesh** for each frame in the game.

![Creating the Vehicle Blueprints](img/image00336.jpeg)

# Creating the Vehicle Blueprints – a section review

In this section, we looked at the necessary blueprints and data assets. We will move on and edit their properties to obtain the behaviors we need for our working **Buggy Vehicle**. First, we created the **WheeledVehicle** blueprint. This is the main blueprint for our vehicle. Then, we created two types of **Wheel Blueprints** (one for our front wheels and another for our back wheels). Further, we created the **TireType** data asset. This is necessary to control the **Friction** property for our wheels. Lastly, we created our animation blueprint for the **Buggy** skeletal mesh, and we discussed about **Animation Blueprints** and its functionalities in detail. Now that we have created the necessary blueprint and data assets for our vehicle, we can move on and edit the properties of these assets.

# Editing the Vehicle Blueprints

With the vehicle blueprints created, let's now move on and edit the properties of these blueprints in order to obtain the behaviors we want for our vehicle. We will begin by working with the **BP_VehicleAnimation** blueprint by double-clicking on our **Content Browser** and opening its **Anim Graph**; which opens by default. The first node we will create is the **Mesh Space Ref Pose** node, and this is used to return the mesh space reference pose for our skeletal mesh in the **Animation Blueprint**. To create this node, right-click on an area of the **Anim Graph** that is empty. Now, from the context menu, we will search for the **Mesh Space Ref Pose** node:

![Editing the Vehicle Blueprints](img/image00337.jpeg)

Next, we will need a **Wheel Handler** node. This is used to alter the wheel transformation based on the setup in the **Wheeled Vehicle** blueprint; keep in mind that this will only work when the owner is of the **Wheeled Vehicle** class. The **Wheel Handler** node also handles the animation needs of our wheels, such as the spinning, the steering, the handbrake, and the suspension. There is no additional setup required; this node obtains all the necessary information from the wheels and transforms it into animation on the bone that the wheel is associated with. To create the **Wheel Handler** node in the **Anim Graph** of our **Vehicle Animation** blueprint, we need to right-click on an area of the graph that is empty. Then, from the context-sensitive menu, we can search for **Wheel Handler**. Finally, we can connect the **Component Pose** output of the **Mesh Space Ref Pose** node to the **Component Pose** input of **Wheel Handler**, as shown in the following screenshot:

![Editing the Vehicle Blueprints](img/image00338.jpeg)

Unless we have additional struts or other suspension needs, we would connect the **Component Pose** output of the **Wheeler Handler** node to the **Result** output node of the **Final Animation Pose**; if we do this, a **Component to Local** node will automatically be generated between the **Wheel Handler** and the **Final Animation Pose** nodes so that it can convert **Component Space Pose** to **Local Space Pose**. As our **Vehicle Physics Asset** and **Vehicle Skeletal Mesh** contain bones for the vehicle suspension, we will want to create additional nodes to handle the joints that affect the suspension polygons. To do this, pull the output of the **Wheel Handler** node and use the context-sensitive drop-down menu; we will search for the **Look At** node. In the **Details** panel under the **Skeletal Control** section of the **Look At** node, we will want to edit the **Bone to Modify** and **Look at Bone** properties so that we can modify the four bones we have on our vehicle's skeletal mesh, and we have the **Look at Bone** look at our wheel joints. Let's create four different **Look At** nodes and set each individual property for the **Bone to Modify** and **Look at Bone** settings:

1.  First node:

    *   **Bone to Modify**: Select this property as `F_L_Suspension`
    *   **Look at Bone**: Select this property as `F_L_wheelJNT`

2.  Second node:

    *   **Bone to Modify**: Select this property as `F_R_Suspension`
    *   **Look at Bone**: Select this property as `F_R_wheelJNT`

3.  Third node:

    *   **Bone to Modify**: Select this property as `B_R_Suspension`
    *   **Look at Bone**: Select this property as `B_R_wheelJNT`

4.  Fourth node:

    *   **Bone to Modify**: Select this property as `B_L_Suspension`
    *   **Look at Bone**: Select this property as `B_L_wheelJNT`

With the four **Look At** nodes in place, we can now connect the output of the last **Look At** node to the **Result** input node of the **Final Animation Pose** node. Our final **Vehicle Animation** blueprint should look similar to the following screenshot:

![Editing the Vehicle Blueprints](img/image00339.jpeg)

With these nodes in place, we are done with the **Vehicle Animation** blueprint. We can now move on and edit our **Tire** data asset.

Similar to what we have discussed earlier, the **Tire** data asset only has one property value to edit: **Friction Scale**. Let's navigate back to **Content Browser** and to our **VehicleContent** folder. Now, we double-click on the **DA_Tire Tire Type** asset. In the **Generic Asset Editor**, let's change the **Friction Scale** property from its default value of `1.0` to a new value of `2.0`:

![Editing the Vehicle Blueprints](img/image00340.jpeg)

With this change to our **Tire** data asset in place, we can now move on and edit our **Wheel** blueprints. Navigate back to **Content Browser** and to our **VehicleContent** folder so that we can double-click our **BP_BackWheel** blueprint and edit its properties. As discussed previously, the properties of the front and back wheels will vary slightly because the front wheels will be responsible for steering, whereas the back wheels will be responsible for responding to the handbrake.

The properties that we need to initially set are as follows:

*   **Shape Radius**: This property determines the radius of the shape used for the vehicle wheel.
*   **Shape Width**: This property determines the width of the shape used for the vehicle wheel.
*   **Affected by Handbrake**: This property determines whether or not the wheel is affected by the handbrake that the player uses to stop the vehicle. This parameter is typically used for back wheels only, not for the front wheels.
*   **Steer Angle**: This specifies the maximum angle that the wheel can rotate in both the positive and negative directions, that is, turning left and right.
*   **Tire Type**: This property determines the **TireType** data asset that the wheel will use for its **Friction Scale** property.

The **Shape Radius** and **Shape Width** properties are determined by the size of the wheels, and in this specific case, these are the back wheels on our vehicle, so for these settings, let's set the following parameters:

*   **Shape Radius**: Set this parameter as `57.0`
*   **Shape Width**: Set this parameter as `30.0`

Again, keep in mind that these values will change depending on the vehicle being used and the size of the wheels. Next, we will need to change the value of the **Steer Angle** property. As we will work with the **BP_BackWheel Wheel** blueprint, and the back wheel will not control the steering; we will set the **Steer Angle** property from its default value of `70.0` to a value of `0.0`.

Moving on, we need to make sure that the **BP_BackWheel Wheel** blueprint has the **Affected by Handbrake** property set to `True` so that these wheels are affected when the player uses the brakes of the vehicle to slow it down and allow it to stop. Lastly, we need to set **Tire Type** from its default value of `DefaultTireType` to `DA_Tire` from the drop-down menu so that this **Tire Type** is used by our **BP_BackWheel**.

![Editing the Vehicle Blueprints](img/image00341.jpeg)

Now that we have completed the **BP_BackWheel** blueprint, let's navigate back to **Content Browser** and to our **VehicleContent** folder and then double-click and open the **BP_FrontWheel** blueprint. If we take a look at the skeletal mesh for our vehicle, we will see that the back wheels are slightly larger than the front wheels; this will be important when you set the values for the **Shape Radius** and **Shape Width** parameters of the **BP_FrontWheel** blueprint. Set the following values for the **Shape Radius** and **Shape Width** properties:

*   **Shape Radius**: Set this parameter to `52.0`
*   **Shape Width**: Set this parameter to `23.0`

We can see that the shape radius is only 5 units less than the **BP_BackWheel** value and the shape width is only 7.0 units less; this is the difference in **Unreal Units** (**uu**) between the two types of wheels.

As we will work with the **BP_FrontWheel** blueprint, we will want to uncheck the **Affected by Handbrake** property so that it is `False` because the front wheels of our vehicle should not react at all to the handbrake. Before we set the **Steer Angle** parameter, we have to understand that this angle is the max angle that the wheel can rotate in both the positive and negative directions, that is, turning left and right. For our **Buggy Vehicle**, any value between `50` and `60` works best, but for the sake of providing a value for testing purposes, let's set the **Steering Angle** value to `55`.

Last but not least, let's make sure that the **TireType** parameter is using our **DA_Tire** data asset.

![Editing the Vehicle Blueprints](img/image00342.jpeg)

Before we move on to editing the **BP_NewVehicle Vehicle** blueprint, let's take some time here to briefly define some of the parameters of our **Wheel** blueprints that we didn't edit so that we have a better understanding of the overall functionality of the **Wheel** blueprint. Here are the additional properties we can manipulate for our **Wheel** blueprint for further customization:

*   **Lat Stiff Max Load**: This is the maximum normalized tire load, in which the tire can deliver no more lateral (sideways) stiffness, irrespective of how much extra load is applied to the tire.
*   **Lat Stiff Value**: This determines how much lateral stiffness can be given to the lateral slip.
*   **Long Stiff Value**: This determines how much longitudinal stiffness can be given to the longitudinal slip.
*   **Suspension Force Offset**: This is the vertical offset from the vehicle center of mass where suspension forces are applied.
*   **Suspension Max Raise**: This value determines how far the wheel can go above its resting position.
*   **Suspension Max Drop**: This value determines how far the wheel can go below its resting position.
*   **Suspension Natural Frequency**: This determines the oscillation frequency of suspension; most cars have values between `5` and `10`.
*   **Suspension Damping Ratio**: This value is the rate at which energy is dissipated from the spring of the vehicle. Most cars have values between `0.8` and `1.2`; values less than `1` are more sluggish, whereas values greater than `1` is twitchier.
*   **Max Brake Torque**: This sets the maximum brake torque of our vehicle in **Newton** **Meters** (**Nm**).
*   **Max Hand** **Brake Torque**: This property sets the maximum handbrake torque for this wheel in Nm. A handbrake should have a stronger brake torque than the brake. This will be ignored for wheels that are not affected by the handbrake.

Now that we have a better understanding of **Wheel Blueprints**, let's move on to our **Vehicle Blueprint**. Navigate back to **Content Browser** and to our **VehicleContent** folder. Let's double-click on **BP_NewVehicle** and focus on the **Details** panel when we select the **Mesh (Inherited)** component from the **Components** tab in the top-left corner of our blueprint. Keep in mind that we need to click on the **Open Full** blueprint editor before viewing the **Viewport**, **Construction Script**, and **Event Graph** tabs in the blueprint.

The first thing we will do is select the **Mesh (Inherited)** component in the **Components** tab. Then, in its **Details** panel, we will change the **Anim Blueprint Generated Class** property and the **Skeletal Mesh** property. For the **Anim Blueprint Generated Class** property, we want to ensure that our `BP_VehicleAnimation` blueprint is selected from the context-sensitive drop-down menu; we do this because we want our vehicle to use the animation blueprint that we set up earlier. For the **Skeletal Mesh** property, we will use the `SK_Buggy_NewVehicle` skeletal mesh that we made a copy of in the **VehicleContent** folder. With these properties in place, we can now edit the parameters of the **VehicleMovement (Inherited)** component, where we will implement the **Wheel Blueprints** for our vehicle.

Select the **VehicleMovement (Inherited)** component from the **Components** tab. Then, in its **Details** panel, let's find the **Wheel Setups** parameters, where we can set **Wheel Class** and **Bone Names** that we want to use for each individual wheel for our vehicle. In the **Wheel Setups** section, set the following parameters for the four wheels:

1.  **Wheel 0**:

    *   **Wheel Class**: Set this parameter to `BP_FrontWheel`
    *   **Bone Name**: Set this parameter to `F_L_wheelJNT`

2.  **Wheel 1**:

    *   **Wheel Class**: Set this parameter to `BP_FrontWheel`
    *   **Bone Name**: Set this parameter to `F_R_wheelJNT`

3.  **Wheel 2**:

    *   **Wheel Class**: Set this parameter to `BP_BackWheel`
    *   **Bone Name**: Set this parameter to `B_L_wheelJNT`

4.  **Wheel 3**:

    *   **Wheel Class**: Set this parameter to `BP_BackWheel`
    *   **Bone Name**: Set this parameter to `B_R_wheelJNT`

    ![Editing the Vehicle Blueprints](img/image00343.jpeg)

Keep in mind that when we use a unique skeletal mesh, the **Bone Name** properties will be different depending on how they are named. We can also add more wheels to the vehicle setup by clicking on the **+** sign next to the **Wheel Setups** option.

The last thing we need to do for this **Vehicle Blueprint** is implement a third-person view camera position behind and slightly above our vehicle. To create a **Camera** component, we need to navigate to the **Components** tab. From the **Add Component** option, we can search the **Camera** component. Name this component `VehicleCamera` and set its position and rotation values as follows:

*   **Location**:

    *   **X**: Set value to `-490.0`
    *   **Y**: Set value to `0.0`
    *   **Z**: Set value to `310.0`

*   **Rotation**:

    *   **Roll (X)**: Set value to `0.0`
    *   **Pitch (Y)**: Set value to `-10.0`
    *   **Yaw (Z)**: Set value to `0.0`

Finally, we want to make sure that the **Use Pawn Control Rotation** option from the **Details** panel of the **VehicleCamera** component is unchecked so that it is set to `False`. With these parameters in place, we are now ready to start setting up our **User Controls** so that we can begin to test our **Vehicle Blueprint**.

# Editing the Vehicle Blueprints – a section review

In this section, we set up the basic functionality for all of our vehicle blueprints. First, we added the functionality to our **Vehicle** animation blueprint by creating a **Mesh Space Ref Pose** node, connected it to a **Wheel Handler** node, and implemented four different **Look At** nodes for each of our **Suspension Bones** that are attached to our vehicle. Next, we set up the **Friction Scale** value for our **Tire Type** data asset. Then, we set up the parameters required for our two different **Wheel Blueprints** so that we get the appropriate behaviors for our front and back wheels. Lastly, we set up the parameters for our **Vehicle Blueprint** by applying the necessary skeletal mesh and animal blueprint for the vehicle. We also took the time to implement the two **Wheel Blueprints** and associated them with the four wheel bones of the vehicle's skeletal mesh. With these blueprints in place, we can now implement the user controls for our vehicle so that the player can actually drive the vehicle in the game environment.

# Setting up user controls

When we use the **Vehicle Game** project example, there are default **User Inputs** already in place that allows you to control the vehicle in the game, but we will take a look at the input settings so that we have a better understanding of what they are. To view the current input controls, let's navigate to **Project Settings** by first left-clicking on the **Edit** drop-down and selecting **Project Settings**; be sure to either be in a blueprint or a level to gain access to **Edit** options:

![Setting up user controls](img/image00344.jpeg)

From **Project Settings**, navigate to the **Input** option in the **Engine** section so that we gain access to **Action Mappings** and **Axis Mappings** for our controls. By default, we already have the **MoveForward** mapping and the **MoveRight** mapping in place that utilize a combination of keyboard keys and gamepad buttons; for our purposes, we will only need to use a few of these buttons. Let's expand the **Axis Mappings** drop-down list and first view the **MoveForward** option; we will see multiple buttons that are used to move our vehicle forward, such as the *W*, *S*, up, and down keys for example. We will remove all the options, except the *W* and *S* keys to ensure that we don't have any unnecessary key bindings for our controls; to do this, just click on the **X** button located next to each option to remove it. We will also see a **Scale** value next to each key binding: `1` and `-1`. This refers to the direction that the **MoveForward** control will actually move the player or vehicle forward or backward; the same idea applies to the **MoveRight** option as well. Let's expand the **MoveRight Axis** mapping and remove all the key bindings, except the *A* and *D* keys.

The last thing we want to do here is evaluate **Handbrake Action Mapping**. By default, we have multiple key bindings, but we want to remove all of them, except the **Space Bar** option:

![Setting up user controls](img/image00345.jpeg)

Before we move on, let's briefly discuss the differences between **Action Mappings** and **Axis Mappings**:

*   **Action Mapping**: These mappings are for key presses and releases, such as the pressing and releasing of *spacebar*
*   **Axis Mapping**: These mappings allow inputs that have a continuous range and direction

Overall, **Action** and **Axis Mappings** provide a mechanism to easily map keys and axes to input behaviors by inserting a layer of indirection between an input behavior and the keys or the game pad buttons that initiate it.

The final step is for us to create a **Game Mode** blueprint so that when we play in-editor, we are able to drive around in our vehicle. Let's navigate back to **Content Browser** and to our **VehicleContent** folder so that we can create the **Game Mode** blueprint. Once in the **VehicleContent** folder, let's right-click `o`n an area of the **Content** folder that is empty, select the **Blueprint Class** option. Then, from **Common Classes**, select the **Game Mode** option and name this new blueprint `BP_VehicleGameMode`. Now, double-click on this new blueprint, and under the **Details** panel, we will find the section labeled as **Classes**. We will change the **Default Pawn** class from `DefaultPawn` to `BP_NewVehicle`. This ensures that when we play the game, by default, it will use our **BP_NewVehicle Pawn** class.

The last thing we need to do is apply this new **BP_VehicleGameMode** to our **Project Settings** by navigating back to **Project Settings**, and under the **Project** section, we will find the **Maps & Modes** option. It's here that we can apply **BP_VehicleGameMode** by expanding the **Default Modes** section. Now, from the **Default Game Mode** drop-down list, we can select the `BP_VehicleGameMode` option. For future reference, when we create levels, we can navigate to the **Settings** option while in the main **Level Editor** and select **World Settings**. This allows you to view your **World Settings** located next to the **Details** panel on the right-hand side of the screen. In **World Settings**, we will find the **Game Mode** section. Here, we can see the **Game Mode Override** parameter and select **BP_VehicleGameMode**. With this in place, we can play the game and see our vehicle in action, but we will see that we are unable to move our vehicle when we press the *W*, *A*, *S*, and *D* keys.

![Setting up user controls](img/image00346.jpeg)

We can now move on and add the input action events in our **BP_NewVehicle** so that we are able to move around and control our vehicle.

# Setting up user controls – a section review

In this section, we created the **Axis Mappings** so that the vehicle could gain the ability to move forward/backward and right/left. We also implemented the ability to use the handbrake with the **Action Mapping** in **Project Settings**. Lastly, we created a new **Game Mode** blueprint and implemented the **Game Mode** in the **Project** and **World Settings** of our level. With these in place, we can move on and add behaviors to our **BP_NewVehicle Event Graph**. This allows you to control your vehicle.

# Scripting movement behaviors

Before we can have our vehicle move through various player controls, we need to script the blueprint behaviors in **BP_NewVehicle Event Graph** by taking advantage of the **VehicleMovement (Inherited) Component** variable. To start with, let's navigate to **Content Browser** and to our **VehicleContent** folder so that we can double-click and open **BP_NewVehicle**.

In an empty area of **Event Graph**, let's right-click and use the context-sensitive drop-down menu to search for our **Input Axis MoveForward** event node so that we can control the forward and backward throttle of our vehicle. Next, we need to grab a **Get** variable of the **VehicleMovement (Inherited)** component. To do this, we have to hold down the *CTRL* key and then click and drag the **VehicleMovement** component from the **Components** tab onto our **Event Graph**. Then, we can pull the **VehicleMovement** variable and search for the **Set Throttle Input** action node from the context-sensitive drop-down menu that appears. Finally, we can connect the main executable pin of **Input Axis MoveForward Event** to the input executable pin of the **Set Throttle Input** node, and we need to connect the **Axis Value** float output of our event to the **Throttle** float value input, as shown in the following screenshot:

![Scripting movement behaviors](img/image00347.jpeg)

What this logic does is it uses **Axis Value** for our **Input Axis MoveForward** option, which will either be `1` or `-1` depending on the keys that are pressed, and applies this value to the **Throttle** of our vehicle, which results in our vehicle moving forward or backward.

Next, let's set up the logic to steer our vehicle by right-clicking on an empty area of our **Event Graph** and search for the **Input Axis MoveRight** event. We will also need a copy of the **Vehicle Movement Component** variable so that we can pull this copy and search for the **Set Steering** **Input** action node from the context sensitive drop-down menu. Connect the output executable pin of the **Input Axis MoveRight** event node to the input executable pin of the **Set Steering Input** node. Also, connect the **Axis Value Float** output to the **Steering Float** input, as shown in the following screenshot:

![Scripting movement behaviors](img/image00348.jpeg)

Lastly, we need to set up the logic for the handbrake so that when the player presses and releases the *spacebar*, the handbrake will react appropriately to the input. To begin with, let's right-click `o`n an empty area of **Event Graph** and search for the **Input Action Handbrake** event node. Next, we will need to create a copy of the **Vehicle Movement Component** variable, and from this variable, we need to pull from it and search for the **Set Handbrake Input** action node from the context-sensitive drop-down menu. We then need to check the **New Handbrake Boolean** input variable of the **Set Handbrake Input** node so that it uses the handbrake of our vehicle to come to a halt. Next, we need to create a copy of the **Vehicle Movement Component** variable and the **Set Handbrake Input** node, but for this copy, we want to make sure that the **New Handbrake Boolean** input variable is unchecked. Lastly, we need to connect the **Pressed** output executable pin of the **Input Action Handbrake** node to the input executable pin of the **Set Handbrake Input** node that has its **New Handbrake Boolean** set to `True`. Then, connect the **Released** output executable pin of the **Input Action Handbrake** node to the input executable pin of the **Set Handbrake Input** node that has its **New Handbrake** set to `False`.

![Scripting movement behaviors](img/image00349.jpeg)

With the logic in place in our **BP_NewVehicle Vehicle** blueprint, we can compile and save the content and then navigate to the **DesertRallyRace** level so that we can play in-editor and test our vehicle. Again, make sure that **World Settings** has the **GameMode Override** parameter set to our **BP_VehicleGameMode** blueprint before the testing phase.

Now, when we play the game, we will be able to move our vehicle forward and backward with the *W* and *S* keys, steer the vehicle with the *A* and *D* keys, and use the handbrake with *spacebar* to have our vehicle come to a halt. We will also see that the wheels spin when it moves either forward or backward, the front wheels turn in the direction we press, and the physics of our vehicle work as expected.

# Scripting movement behaviors – a section review

In this section, we worked on scripting the necessary behaviors of our vehicle so that when we play the game, we were able to move and steer our vehicle. First, we implemented **Set Throttle Input** in conjunction with our **Input Axis MoveForward** event node. Then, we used the **Set Steering Input** node with our **Input Axis MoveRight** event. Lastly, we associated the **Set Handbrake** node functionality with the **Input Action Handbrake** event node. Now that we are able to drive our vehicle in the game, we can evaluate its behavior and test how it feels.

# Testing the vehicle

When we test our vehicle, we have to keep in mind the controls and the feel we are trying to create when we drive the vehicle. The behaviors of a vehicle will drastically differ depending on the type of gameplay we are going for, such as the drastic difference between the vehicles in Mario Kart as compared to the ones from the Forza series.

If tweaks or changes are necessary to obtain the desired behavior, the main aspect to view is the **VehicleMovement (Inherited)** component in the **BP_NewVehicle** blueprint, where it has various parameters in its **Details** panel that we can change in order to change the behavior of the vehicle, such as **Differential Setup** or **Transmission Setup**. We can also use **VH_Buggy** and the other default vehicle content that is provided by Epic Games when we use the **Vehicle Game Project** example as a reference point to change the way our vehicle behaves.

Use the vehicle we have created in this chapter as a stepping stone to create a unique vehicle that behaves in different ways. Also, feel free to play around with the settings in the **Animation**, **Wheel**, and **Vehicle** **Blueprints** to see what we can create.

# Summary

In this chapter, we created our own working vehicle with the **Vehicle Game Project Example** template step by step from scratch. In the process of doing so, we accomplished certain tasks.

First, we downloaded and created a project using the **Vehicle Game Project Example** template so that we could have access to several resources and content available to create a basic vehicle and a template racing game. Then, we created our own **Physics Asset** using **PhAT** with the default buggy skeletal mesh as a base and implemented our own **Physics Bodies** to the vehicle.

Next, we created all the necessary blueprints and data assets required in constructing a working vehicle for our game. To begin with, we created a **Wheel Vehicle Blueprint** component that contained the **VehicleMovement (Inherited) Component** class. Then, we created two different types of **Wheel Blueprints** (one for the front wheels and another for the back wheels). Each has its own set of unique parameters. Lastly, we created the **Vehicle Animation Blueprint** component required to obtain the proper motion of our wheels when we drive.

Additionally, we then edited each of these blueprints so that we could obtain the proper behavior for our vehicle. We also set up the user controls for our vehicle by editing **Input Action** and **Axis Mappings** so that the appropriate key bindings were set for our vehicle to move forward/backward in order to use the handbrake and steer left/right.

Then, we implemented the **Blueprint** logic within our **BP_NewVehicle Wheeled Vehicle Blueprint** by implementing the **Input Action** and **Axis Mapping** event nodes to the appropriate **Vehicle Movement** actions such as setting the throttle value, and the steering input values.

Lastly, we set up our own **Game Mode Blueprint** that utilizes our **BP_NewVehicle Pawn Blueprint** class and implemented that **Game Mode** into the **Project Settings**, as well as to the **World Settings** of our level. From there, we were able to play in-game and drive our vehicle around the level, and we posed the challenge of changing the parameters of our **BP_NewVehicle** in order to obtain unique behaviors for our vehicle.

In the next chapter, we will be covering advanced physics topics and troubleshooting concepts like pragmatic physics.