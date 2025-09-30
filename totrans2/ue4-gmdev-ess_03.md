# Chapter 3. Materials

Material is an asset that defines the look of a mesh with various graph nodes that include images (textures) and math expressions. Since Unreal Engine 4 utilizes **Physically Based Rendering** (**PBR**), creating realistic materials such as metal, concrete, bricks, and so on, can be quite easy. Materials in Unreal Engine define everything about the surface of the mesh, such as its color, shininess, bumpiness, and tessellation, and can even animate objects by manipulating the vertices! At this point you might think *Ok, Materials are only used for meshes* but, no, they are not actually limited to meshes. You use Materials for decals, post process, and light functions too.

Creating a Material is a pretty straightforward process. All you have to do is right-click in **Content Browser**, select **Material**, and give it a name. Done!

![Materials](img/B03950_03_01.jpg)

# Material user interface

Now that we know what a Material is and what it does, let's take a look at the user interface of Material graph.

![Material user interface](img/B03950_03_02.jpg)

## Toolbar

The **Toolbar** panel contains various buttons that help to preview graph nodes, remove isolated nodes, Material stats, and so on. Let's take a look at what these buttons do:

*   **Save**: Applies the changes you made to the Material and saves the asset![Toolbar](img/B03950_03_03.jpg)
*   **Find in CB**: Navigates and selects this Material in **Content Browser**![Toolbar](img/B03950_03_04.jpg)
*   **Apply**: Applies the changes to the Material. Note that this will not save the Material![Toolbar](img/B03950_03_05.jpg)
*   **Search**: Searches for Material expressions or comments![Toolbar](img/B03950_03_06.jpg)
*   **Home**: Navigates to and selects the main canvas node![Toolbar](img/B03950_03_07.jpg)
*   **Clean Up**: Removes unconnected nodes![Toolbar](img/B03950_03_08.jpg)
*   **Connectors**: Shows or hides unconnected pins![Toolbar](img/B03950_03_09.jpg)
*   **Live Preview**: Toggles a real-time update of preview material![Toolbar](img/B03950_03_10.jpg)
*   **Live Nodes**: Toggles a real-time update of graph nodes![Toolbar](img/B03950_03_11.jpg)
*   **Live Update**: Recompiles a shader for every node in the graph![Toolbar](img/B03950_03_12.jpg)
*   **Stats**: Toggles Material stats and compilation errors
*   **Mobile Stats**: Same as stats but for mobile![Toolbar](img/B03950_03_13.jpg)

Live nodes might be confusing for new users so I'll explain about them further.

### Live preview

Sometimes we need to preview the result of a specific node before connecting it to the main node or for debugging purposes.

To preview a node you need to right-click on the node and select **Start Previewing Node**.

![Live preview](img/B03950_03_14.jpg)

Unless you enable **Live Preview**, you will not see any changes in the preview material.

### Tip

You can press the spacebar to force a preview.

### Live nodes

This will show a real-time update of nodes due to changes made by expressions to that node. See the following example:

![Live nodes](img/B03950_03_15.jpg)

In the preceding screenshot, the **Sine** node is getting a constant update from **Time**, multiplied by one. If you enable **Live Nodes**, you will see the **Sine** node pulsing between black and white. If you change the **Multiply** value from **1** to anything else (for example, **5**) you will not see the changes unless you enable **Live Update** too.

### Live update

When enabled, all expressions are compiled whenever you make a change, such as adding a new node, deleting a node, changing a property, and so on. If you have a complex graph, it is recommended to disable this option as it has to compile all nodes every time you make a change.

## Preview panel

The **Preview** panel shows the result of the Material that you are currently editing. You can navigate in preview Material using these options:

*   **Rotate the mesh**: Drag with the left mouse button
*   **Pan**: Drag with the middle mouse button
*   **Zoom**: Drag with the right mouse button
*   **Update light**: Hold *L* and drag with the left mouse button

In the top-right corner of the preview viewport you can change some settings. This changes the preview mesh to the selected primitive shape:

![Preview panel](img/B03950_03_16.jpg)

This changes the preview mesh to a custom mesh. You need to select a **Static Mesh** in **Content Browser**:

![Preview panel](img/B03950_03_17.jpg)

This toggles the rendering of the grid in the preview viewport:

![Preview panel](img/B03950_03_18.jpg)

This toggles real-time rendering in the preview viewport:

![Preview panel](img/B03950_03_19.jpg)

## Details panel

The **Details** panel shows all the properties you can edit when you select a node in the graph. If no nodes are selected, it will show the properties of the Material itself.

For more information on these settings, please visit the Material properties documentation at [https://docs.unrealengine.com/latest/INT/Engine/Rendering/Materials/MaterialProperties/index.html](https://docs.unrealengine.com/latest/INT/Engine/Rendering/Materials/MaterialProperties/index.html).

## Graph panel

This is the main area where you create all the nodes that decide how the Material should look and behave. By default, a Material graph contains one master node that has a series of inputs, and this master node cannot be deleted. Some of the inputs are grayed out and can be enabled by changing the **Blend** mode in the **Details** panel.

![Graph panel](img/B03950_03_20.jpg)

## Palette panel

The **Palette** panel lists all the graph nodes and Material functions that can be placed in the graph using drag and drop.

### Tip

Using the **Category** option, you can filter **Palette** contents between expressions or Material functions.

# Common material expressions

There are some common Material nodes that we use most of the time when we create a material. To create a node you need to right-click on the graph canvas and search for it, or you can use the **Palette** window to drag and drop. Some nodes also have shortcut keys assigned to them.

Let's take a look at these common nodes.

## Constant

Constant expression outputs a single float value and can be connected to almost any input. You can convert a constant expression to a parameter and make real-time changes to the Material instance. You can also access a parameter through Blueprint or C++ and see the changes in the game.

*   **Shortcut key**: Hold *1* and click on the graph area
*   **Parameter shortcut key**: Hold *S* and click on the graph area
*   **Example usage**: Brighten or darken a texture

### Tip

Constant parameter is called a scalar parameter

![Constant](img/B03950_03_21.jpg)

You can see a constant expression (0.2) being used to darken a texture.

## Constant2Vector

The Constant2Vector expression outputs two float values, which is a two-channel vector value (for example, red channel and green channel). You can convert Constant2Vector to a parameter and make real-time changes to the Material instance or access it in Blueprint or C++ to make dynamic changes to the material while playing the game.

*   **Shortcut key**: Hold *2* and click on the graph area
*   **Parameter shortcut key**: Hold *V* and click on the graph area
*   **Example usage**: Adjust the UVs of a texture separately

![Constant2Vector](img/B03950_03_22.jpg)

You can see a Constant2Vector being used to tile a texture in the preceding screenshot.

## Constant3Vector

The Constant3Vector expression outputs three float values, which is a three-channel vector value (for example, red channel, green channel, and blue channel). You can convert Constant3Vector to a parameter and make real-time changes to a Material instance or access it in Blueprint or C++ to make dynamic changes to a material while playing the game.

*   **Shortcut key**: Hold *3* and click on the graph area
*   **Parameter shortcut key**: Hold *V* and click on the graph area
*   **Example usage**: Change the color of a given texture

![Constant3Vector](img/B03950_03_23.jpg)

You can see Constant3Vector being used to color a grayscale texture in the preceding screenshot.

## Texture coordinate (TexCoord)

The texture coordinate expression outputs texture UV coordinates as a two-channel vector (for example, U and V), which helps with tiling and also allows you to use different UV coordinates.

*   **Shortcut key**: Hold *U* and click on the graph area

![Texture coordinate (TexCoord)](img/B03950_03_24.jpg)

The preceding screenshot shows a texture coordinate being used to tile a texture. You can see the values used by looking at the **Details** panel in the bottom left corner.

## Multiply

This expression multiplies the given inputs and outputs the result:

*   Multiplication happens per channel. For example, if you multiply two vectors (0.2, 0.3, 0.4) and (0.5, 0.6, 0.7), the actual process is the following:

    [PRE0]

    So the output is as follows:

    [PRE1]

*   The **Multiply** node expects inputs to be the same type unless one of them is constant. In short, you cannot multiply Constant2Vector and Constant3Vector, but you can multiply Constant2Vector or Constant3Vector by a constant expression.

    *   **Shortcut key**: Hold *M* and click on the graph area

    ![Multiply](img/B03950_03_25.jpg)

The preceding screenshot shows a multiply node being used to boost an emissive effect.

## Add

This expression adds the given inputs and outputs the result:

Addition happens per channel. For example, if you add two vectors (1, 0, 0) and (0, 1, 0), the actual process is the following:

[PRE2]

So the output is as follows:

[PRE3]

The **Add** node expects inputs to be the same type unless one of them is constant. In short, you cannot add Constant2Vector and Constant3Vector, but you can add Constant2Vector or Constant3Vector to a constant expression. Let's take a look at why it is like this. See the following screenshot:

![Add](img/B03950_03_26.jpg)

Here we are trying to add Constant3Vector and Constant2Vector but it will not work. This is because, when the Material editor tries to compile the **Add** node, it fails since the last element of Constant3Vector has nothing to add to. It will be like the following calculation:

[PRE4]

But you can add Constant3Vector to a constant expression, as in the following figure:

![Add](img/B03950_03_27.jpg)

The result will be as follows:

[PRE5]

And that will compile fine.

*   **Shortcut key**: Hold *A* and click on the graph area![Add](img/B03950_03_28.jpg)

## Divide

The divide expression divides the given inputs and outputs the result:

Division happens by channel. For example, if you divide two vectors (0.2, 0.3, 0.4) and (0.5, 0.6, 0.7), the actual process is like this:

[PRE6]

So the output is as follows:

[PRE7]

The **Divide** node expects inputs to be the same type unless one of them is constant. In short, you cannot divide Constant2Vector by Constant3Vector, but you can divide Constant2Vector or Constant3Vector by a constant expression.

*   **Shortcut key**: Hold *D* and click in the graph area![Divide](img/B03950_03_29.jpg)

## Subtract

This expression subtracts the given inputs and outputs the result:

Subtraction happens by channel. For example, if you subtract two vectors (0.2, 0.3, 0.4) and (0.5, 0.6, 0.7), the actual process is the following:

[PRE8]

So the output is as follows:

[PRE9]

The **Subtract** node expects inputs to be the same type unless one of them is constant. In short, you cannot subtract Constant2Vector from Constant3Vector, but you can subtract Constant2Vector or Constant3Vector from a constant expression.

*   **Shortcut key**: No shortcut key![Subtract](img/B03950_03_30.jpg)

## Texture sample (Texture2D)

Texture sample outputs the given texture. It also outputs all four channels (namely, red, green, blue, and alpha) from the texture separately so you can use them for various things. This is especially useful if you work on multiple grayscale textures (such as mask textures, roughness textures, and so on). Instead of importing multiple textures, you can just create one texture in Photoshop and assign other textures to different channels and, in Material editor, you can get each channel and do all the fancy things. Oh, and did I mention Texture2D can use movie textures too?

You can convert **Texture Sample** to **TextureSampleParameter2D** and change textures in real-time via Material instance. You can also change textures dynamically in the game through Blueprints or C++.

*   **Shortcut key**: Hold *T* and click in the graph area
*   **Parameter shortcut key**: No shortcut key

![Texture sample (Texture2D)](img/B03950_03_31.jpg)

## Component mask

The component mask expression can extract different channels from the input, which should be a vector channel such as **Constant2Vector**, **Constant3Vector**, **Constant4Vector**, **TextureSample**, and so on. For example, you know Constant4Vector has only one output, which is RGBA. So, if you want the green channel from RGBA, you use a component mask. You can right-click on a component **Mask** and convert it into a **Parameter** and make real-time changes in Material instance.

*   **Shortcut key**: No shortcut key
*   **Parameter shortcut key**: No shortcut key![Component mask](img/B03950_03_32.jpg)

In this screenshot, we extract the alpha channel and plug it into **Opacity** and plug the RGB channel into **Base Color**.

## Linear interpolate (lerp)

This blends two textures or values based on alpha. When the alpha value is **0** (black color), **A** input is used. If the alpha value is **1** (white color), **B** input is used. Most of the time, this is used to blend two textures based on a mask texture.

*   **Shortcut key**: Hold *L* and click in the graph area
*   **Example usage**: Blend two textures based on the alpha value, which can be a constant or a mask texture![Linear interpolate (lerp)](img/B03950_03_33.jpg)

Here, the lerp node is outputting 100% of input **A** because the alpha value is **0**. If we set the alpha value to **1** then we'll see 100% of **B**. If alpha is **0.5** then we'll see a blend of both **A** and **B**.

## Power

The **Power** node multiplies the base input by itself with Exp times. For example, if you have **4** in **Base** and **6** in **Exp** then the actual process is like this:

[PRE10]

So the result of **Power** is `4096`.

If you apply a **Texture** to **Base** input and have a constant value (for example, **4**) then the **Texture** is multiplied four times.

*   **Shortcut key**: Hold *E* and click in the graph area
*   **Example usage**: Adjust the contrast of the height map or ambient occlusion map![Power](img/B03950_03_34.jpg)

The preceding image shows a Power node being used to boost the contrast of a **Texture Sample**.

## PixelDepth

**PixelDepth** outputs the distance to the camera of the pixel currently being rendered. This can be useful to alter the appearance of the material based on the distance from the player.

*   **Shortcut key**: No shortcut key
*   **Example usage**: Change the color of an object based on the distance from the player![PixelDepth](img/B03950_03_35.jpg)

If you apply the previous material to a mesh, then the color of the mesh will be changed based on the distance to the player camera.

![PixelDepth](img/B03950_03_36.jpg)

The preceding screenshot shows how the mesh will look closer to the player camera.

![PixelDepth](img/B03950_03_37.jpg)

The preceding screenshot shows how the mesh will look when it's farther away from the player camera.

## Desaturation

As the title says, the **Desaturation** expression desaturates its input. Simply put, it can convert a color image to grayscale based on a certain percentage.

*   **Shortcut key**: No shortcut key

![Desaturation](img/B03950_03_38.jpg)

## Time

This expression outputs the **Time** passage of the game (in seconds). This is used if you want your Material to change over time.

*   **Shortcut key**: No shortcut key
*   **Example usage**: Create a pulsing Material![Time](img/B03950_03_39.jpg)

In the previous material, we multiply **Time** by a constant expression. The result of the **Multiply** node is plugged into the **Sine** node, which outputs a continuous oscillating waveform that outputs the value in a range of **-1** to **1**. We then use a **ConstantBiasScale** node to prevent the value from going below **0**. A **ConstantBiasScale** node is basically a node that adds a bias value to the input and multiplies it by a scale value. By default, bias is set to **0.5** and scale to **1**. So if the **Sine** value is **-1**, then the result is `(-1 + 1) * 0.5`, which equals **0**.

## Fresnel

**Fresnel** creates rim lighting, which means it will highlight the edges of the mesh.

*   **Shortcut key**: No shortcut key![Fresnel](img/B03950_03_40.jpg)

The result of the previous network is as follows:

![Fresnel](img/B03950_03_41.jpg)

## Material types

Now that you know some of the basic expressions, let's take a look at different Material types. First of all, obviously, is the main Material editor, but then you also have Material instances, Material functions, and layered Materials.

## Material instances

Material instance is used to change the appearance of a Material without recompiling it. When you change any value in Material editor and apply it, it will recompile the whole shader and create a set of shaders. When you create a Material instance from that Material, it will use the same set of shaders so you can change the values in real time without recompiling anything. But when you use **Static Switch Parameter** or **Component Mask Parameter** in your **Parent Material**, then it's different because each of those parameters has unique combinations. For example, let's say you have **Material_1** with no **Static Switch Parameter**, and **Material_2** with **Static Switch Parameter** called **bEnableSwitch**. **Material_1** will create only one set of shaders, while **Material_2** will create two sets of shaders with **bEnableSwitch = False** and **bEnableSwitch = True**.

An example workflow is to create a master Material that contains all the necessary parameters and let the designers make different versions.

There are two types of Material instances. They are:

*   Material Instance Constant
*   Material Instance Dynamic

Only Material Instance Constant has a user interface. Material Instance Dynamic has no user interface and cannot be created in content browser.

### Material Instance Constant

As the title says, **Material Instant Constant** (**MIC**) is only editable in the editor. That means you cannot change the values at runtime. MIC exposes all parameters you created in the parent Material. You can create your own groups and organize all your parameters nicely.

![Material Instance Constant](img/B03950_03_42.jpg)

Material Instance User Interface

*   **Toolbar (1)**: The following are toolbar options:

    *   **Save**: Saves the asset
    *   **Find in CB**: Navigates to this asset in Content Browser and selects it
    *   **Params**: Exposes all parameters from Parent Material
    *   **Mobile Stats**: Toggles Material stats for Mobile

*   **Details (2)**: Displays all the parameters from parent Material and other properties of Material instance. Here you can also assign a physics Material and override the base properties of the parent Material, such as blend mode, two-sided, and so on.
*   **Instance parents (3)**: Here you will see a chain of parents all the way up to the main master Material. The instance currently being edited is shown in bold.
*   **Viewport (4)**: The viewport displays the material on a mesh so you can see your changes in real time. You can change the preview shape in the top-right corner. This is the same as it was in Material editor.

### Material Instance Constant example

In order for Material instance to work, we need a master Material with parameters. Let's create a simple Material that will change its color based on the distance to the player, that is, when the player is near the mesh it will have a red color, and as they move further away it will change its color. Note that there are 21 parameter expressions in UE4\.

Right now we will stick with two common parameters, and they are as follows:

*   Scalar parameter
*   Vector parameter![Material Instance Constant example](img/B03950_03_43.jpg)

As you can see in the previous screenshot, we created two vector parameters (**Color1**, **Color2**) and two scalar parameters (**TransitionDistance**, **Speed**). We will use these parameters to modify in real time. To create an instance of this Material you need to right-click on this Material in **Content Browser** and select **Create Material Instance**. This will create a new instance Material right next to this Material.

If you open that instance you will see all these parameters there, and you can edit them in real time without having to wait for the Material to recompile:

![Material Instance Constant example](img/B03950_03_44.jpg)

To change values in Material instance, you need to override them first. You need to click the checkbox near the parameter to override the values. As shown in the following screenshot:

![Material Instance Constant example](img/B03950_03_44A.jpg)

## Material functions

Material functions are graphs that contain a set of nodes that can be used across any number of Materials. If you often find yourself creating complex networks then it's better to make a Material function so you can contain all these complex networks in one single node. One thing to keep in mind is that Material function cannot contain any parameter nodes (for example, **Scalar Parameter**, **Vector Parameter**, **Texture Parameter**, and so on). To pass data into a Material function, you need to use a special **FunctionInput** node. Similarly, if you want data out of a Material function, you need to use the **FunctionOutput** node. By default, Material function creates one output for you but you can create more outputs if you want.

The UI of Material function is almost the same as of Material editor. If you check the **Details** panel you will see some options to help you get the most out of your Material function. Let's take a look at these options:

*   **Description**: This appears as a tooltip when you hover the mouse on this function node in Material graph.
*   **Expose to Library**: Enable this to show your Material function when you right-click inside your Material graph.
*   **Library Categories**: This list the categories this function belongs to. By default, it belongs to the **Misc** category but you can change it and add as many categories as you want.

### Tip

Material functions cannot be applied surface, so if you want to use a Material function you must use it in a Material.

### Material function example

To create a Material function, first right-click in **Content Browser** and go to **Materials & Textures** and select **Material Function**. In this example, we will create a Material function called **Normal Map Adjuster** that can boost the intensity of a normal map. Let's see what we need to create such a function:

*   **Texture [INPUT]**: Obviously we need to pass a texture that needs to be modified.
*   **Intensity [INPUT]**: We also need to pass how intense the normal should be. A value of **0** means no changes to the normal map and a value of **1** means a boosted normal effect.
*   **Result [OUTPUT]**: Finally we will output the result, which we can connect to the normal channel in our Material.

### Tip

The final output node (result) can be renamed with any custom name you want. Select the node and, in the **Details** panel, change **Output Name**.

Open your Material function and right-click on the graph and search for **Input**.

![Material function example](img/B03950_03_45.jpg)

Select the **FunctionInput** node. You will see some properties in the **Details** panel for the **Input** node you just selected.

![Material function example](img/B03950_03_46.jpg)

Let's take a look at these settings:

*   **Input Name**: A custom name for the input. You can name it whatever you want. Here, I called it **Normal Texture**.
*   **Description**: Will be used as a tooltip when you hover over this input in Material graph.
*   **Input Type**: Defines the type of input for this node.
*   **Preview Value**: Value to use if this input is not connected in Material graph. Only used if **Use Preview Value as Default** is checked.
*   **Use Preview Value as Default**: If checked, it will use the **Preview Value** and will mark this input as optional. So when you use this function, you can leave this input unconnected. But if you disable this option, then you must connect the required node to this when in Material graph.
*   **Sort Priority**: Arranges this input in relation to other input nodes.

Let's create a simple network to boost the normal effect. Take a look at the following screenshot:

![Material function example](img/B03950_03_47.jpg)

Here we are extracting the red, green, and blue channels separately. The reason behind this is simply that we need to multiply **Intensity** (scalar input value) by only the blue channel to increase the normal effect. The **Intensity** needs to be clamped between **0** and **1** and then inverted using the **1-x** (OneMinus) node because, when we use this Material function in a Material, we need **0** to have the default normal intensity and **1** to really boost the effect. Without the OneMinus node it will be the opposite, that is, **0** will boost the normal map effect and **1** will have a regular effect.

Now that the function is done, click the **Save** button on the toolbar.

### Tip

Saving automatically compiles the Material.

Now to get this into Material, right-click inside the Material graph and search for **NormalMapAdjuster**. Then all you have to do is plug a **Normal** map and a **Scalar Parameter** to **NormalMapAdjuster** and connect it to the **Normal** channel.

### Tip

If it doesn't show up in the context menu, make sure you enabled **Expose to Library** in Material Function.

![Material function example](img/B03950_03_48.jpg)

In your Material instance you can adjust **NormalIntensity** in real time.

#### Layered Material

Layered Materials are basically *Materials within Materials* and exist as an extension of Material function. The basic workflow is as follows: you create a **Make Material Attribute** (which features all the material attributes, such as **Base Color**, **Metallic**, **Specular**, **Roughness**, and so on) and you connect your nodes to it. Then you connect the output of **Make Material Attributes** to the input of the **Output Result** node.

Layered Materials are most beneficial when your assets have different layers of materials. For example, think about a character with different elements such as metallic armor, leather gloves, skin, and so on. Defining each of these materials and blending them in a conventional way will make the material complexity increase significantly. If you use layered Material in such cases, you can define each of those materials as a single node and blend them very easily.

##### Creating layered Material using make material attributes

For this example we will create two simple layered Materials and blend them together in our final material. First, create a Material function and open it. In Material function, follow these steps:

1.  Right-click on the graph editor and search for **Make Material Attributes** and select the node from that menu.
2.  Create a **Constant3Vector** node and connect it to **BaseColor** of **Make Material Attributes**.
3.  Create a constant value and connect it to **Metallic** of **Make Material Attributes**.
4.  Create one more constant value and connect that to **Roughness** of **Make Material Attributes**.
5.  Finally, connect **Make Material Attributes** to the output of Material function.

The final Material function should look like this. Note the values I'm using for constant nodes.

![Creating layered Material using make material attributes](img/B03950_03_49.jpg)

Since we want this to be **Metallic**, we set **Metallic** to **1**.

We will create a duplicate of this same Material function and make it a non-metallic Material with a different color. See the following image:

![Creating layered Material using make material attributes](img/B03950_03_50.jpg)

This is a non-metallic Material and we are going to blend these two Materials in our Material editor using a default **Material Layer Blend** function.

Make sure you expose both of these Material functions so we can use them in Material editor.

Open an existing Material or create a new one in **Content Browser** and open it:

1.  Right-click on the graph and search for your Material functions (select both of them).
2.  Right-click again on the graph and search and select **MatLayerBlend_Simple**.
3.  Connect your Material functions to **MatLayerBlend_Simple**. Connect one function to **Base Material** and the other one to **Top Material**.
4.  Now, to blend these two materials we need an **Alpha (Scalar)** value. A value of **1** (white) will output **Base Material** and a value of **0** will output **Top Material**. A value of **0.5** will output a mix of both **Base** and **Top** materials.

Since we are using layered Material we cannot directly connect this to the Material editor like other nodes. To make this work, there are two ways we can connect.

###### Method 1:

We can make the material use Material attributes instead of regular nodes. To use this feature, click anywhere on the graph and in the **Details** panel select **Use Material Attributes**:

![Method 1:](img/B03950_03_51.jpg)

When you enable this, the main material node will show only one node called Material attributes so you can connect the output of **MatLayerBlend_Simple** to this node.

The following is a screenshot of the final material using this method:

![Method 1:](img/B03950_03_52.jpg)

###### Method 2:

In this method, instead of using Material attributes for the main node we use **BreakMaterialAttributes** and connect them as regular nodes:

1.  Right-click on the graph area and search and select **BreakMaterialAttributes**.
2.  Connect the output of **MatLayerBlend_Simple** to **BreakMaterialAttributes**.
3.  And finally, connect all the output nodes of **BreakMaterialAttributes** to the main node of Material editor.

The following is a screenshot of the final material using this method:

![Method 2:](img/B03950_03_53.jpg)

# Summary

In the next chapter we will use post processing techniques to enhance the look of your scene. We will also create a simple Material and use it in post process Material.