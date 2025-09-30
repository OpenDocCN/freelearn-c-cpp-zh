# Chapter 4. Post Process

Post Process in Unreal Engine 4 allows you to create a variety of artistic effects and change the overall look and feel of the whole game. Post Process effects are activated using Post Process Volumes and can be used individually to affect only a specific area or the entire scene. You can have multiple Post Process Volumes overlapping and render their effects based on their priority. Post Process Volumes can be used to add or modify simple effects such as Bloom, Lens Flares, Eye Adaptation, Depth of Field, and so on and they can also be used to get advanced effects using Materials. Another great feature of Post Process Volume is **Look up Table** (**LUT**), which is used to store color transformations from image editing software, such as Adobe Photoshop or GIMP. They are very easy to set up and can yield very good results. We will get into LUT later in this chapter.

When you first start a project without starter content, there will be no Post Process Volumes present in the scene, so Engine will use the default settings. You can change these settings per project under **Project Settings**:

1.  Click on **Edit** in the menu bar.
2.  Click on **Project Settings**.
3.  Go to the **Rendering** section.
4.  Expand **Default Postprocessing Settings**:![Post Process](img/B03950_04_01.jpg)

Here, you will see the default settings for Unreal Engine when there is no Post Process Volume in your scene. You can modify these settings or add a Post Process Volume to override them independently.

# Adding Post Process

To use Post Process, you need a Post Process Volume in your scene:

1.  Go to the **Modes** tab (if you closed it, press *Shift* + *1*).
2.  Select the **Volumes** tab.
3.  Drag and drop Post Process Volume into the scene:![Adding Post Process](img/B03950_04_02.jpg)

You now have a Post Process Volume in your scene. However, it only shows the effects when the player is inside that volume. To make it affect the whole scene perform the following steps:

1.  Select **Post Process Volume**
2.  In the **Details** panel, scroll down and expand the **Post Process Volume** section
3.  Enable **Unbound**

Enabling **Unbound** will ignore the bounds of this volume and affect the whole scene. Now, let's take a quick look at these Post Process settings:

![Adding Post Process](img/B03950_04_03.jpg)

*   **Priority**: If multiple volumes are overlapping each other, then the volume with higher priority overrides the lower one.
*   **Blend Radius**: This is the radius of the volume used for blending. Generally, a value of `100` works best. This setting is ignored if you have **Unbound** enabled.
*   **Blend Weight**: This defines the influence of properties. `0` means no effect and `1` means full effect.
*   **Enabled**: This enables or disables this volume.
*   **Unbound**: If enabled, then Post Process effects will ignore the bounds of this volume and will affect the whole scene.

## LUT

LUTs are color neutral textures unwrapped to a 256 x 16 size texture. They are used to create unique artistic effects and are modified using image editing software such as Adobe Photoshop. If you are not familiar with Photoshop, you can use free and open source software such as GIMP. The following is an image of the default LUT texture:

![LUT](img/B03950_04_04.jpg)

The procedure of LUT is as follows:

1.  First you take a screenshot of your world and bring it into Photoshop.
2.  On top of that screenshot, you insert the LUT texture.
3.  Then on top of both, apply color manipulations (for example: adjustment layer).
4.  Now select the LUT texture and save it with your color manipulation as PNG or TGA.
5.  Finally, import your LUT into Unreal Engine.

    ### Note

    Note that after you import your LUT into **Content Browser**, open it and set the **Texture Group** to **ColorLookupTable**. This is an important step and should not be skipped.

To apply the LUT, select the Post Process volume, and under the **Scene Color** section, you can enable **Color Grading** and set your LUT texture:

![LUT](img/B03950_04_05.jpg)

With the **Color Grading Intensity** option, you can change the intensity of the effect.

## Post Process Materials

Post Process Materials help you create custom post processing with the help of Material Editor. You need to create a Material with your desired effect and assign it to **Blendables** in Post Process Volume. Click on the plus sign to add more slots:

![Post Process Materials](img/B03950_04_06.jpg)

Before I explain about Post Process Materials, let's take a quick look at one of the most important Post Process nodes in Material Editor:

*   **Scene Texture**: This node has multiple selections that output different textures:![Post Process Materials](img/B03950_04_07.jpg)
*   **UVs** (optional): This input tiles the texture. For UV operations on the **SceneTexture** node, it is good to use the **ScreenPosition** node instead of the regular **Texture Coordinate** node.
*   **Color**: This outputs the final texture as RGBA. If you want to multiply this with a color, you first need to use a component mask to extract R, G, and B and then multiply it by your color.
*   **Size**: This outputs the size (width and height) of the texture.
*   **InvSize**: This is the inverse of the **Size** output. (1/width and 1/height).

    ### Note

    It is important to keep in mind that you should only use Post Process Materials when you really need them. For **Color Correction** and various other effects, you should stick with the settings from Post Process Volume since they are more efficient and optimized.

## Creating a Post Process Material

With Post Process Material, you can create your own custom Post Processing effects. Some examples are:

*   Highlighting a specific object in your game
*   Rendering occluded objects
*   Edge detection, and so on

In this quick example, we will see how to highlight an object in our world. To render a specific object separately, we need to put them to a custom depth buffer. The good thing is, it's as easy as clicking on a checkbox.

Select your Static Mesh and under the **Rendering** section, expand the options and enable **Render Custom Depth**:

![Creating a Post Process Material](img/B03950_04_08.jpg)

Now that the mesh is rendered in the `CustomDepth` buffer, we can use this information in Material Editor to mask out and render it separately. To do that:

1.  Create a new Material and open it.
2.  First thing to do now is to set **Material Domain** to **Post Process**. This will disable all inputs except **Emissive Color**.
3.  Now, right-click on the graph and search for **SceneTexture** and select it. Set **Scene Texture Id** to **CustomDepth**.
4.  **CustomDepth** outputs a raw value so let's divide it by the distance we want.
5.  Add a new **Divide** node and connect **CustomDepth** to input *A*. Select the divide node and for *Const B* give a high value (for example: `100000000`). Remember, 1 Unreal Unit is equal to `1` cm so if you give a small value like `100` or `1000`, you need to go really close to the object to see the effect. This is why we use a very large value.
6.  Add a **Clamp** node and connect **Divide** to the first input of the **Clamp** node.
7.  Create a **Lerp** node and connect the output of **Clamp** to the *Alpha* input of **Lerp**. The **Lerp** node will blend input `A` and `B` based on the alpha value. If the alpha value is `1`, then input *A* is used. If it is `0` then input *B* is used.
8.  Create another **SceneTexture** node and set its *Scene Texture Id* to **PostProcessInput0**. **PostProcessInput0** outputs the final HDR color so make sure you use this. There's another output called **SceneColor**, which does the same but it outputs lower quality of the current scene.
9.  Right-click on the graph again and search for the **Desaturation** node. Connect **PostProcessInput0** *Color* output to **Desaturation** input. We will use this to desaturate the whole scene except our mesh with **CustomDepth**.
10.  Connect the **Desaturation** output to *Lerp B* and **PostProcessInput0** to *Lerp A*, and finally, connect the **Lerp** to **Emissive Color**.

Here is the final screenshot of the whole graph:

![Creating a Post Process Material](img/B03950_04_09.jpg)

And in this example scene, I've applied this Material to Post Process Blendables and you can see the effect:

![Creating a Post Process Material](img/B03950_04_10.jpg)

Everything that is in color has Render Custom Depth enabled so the Post Process Material is masking them out and applying the desaturation to the entire scene.

# Summary

In next chapter, we will add lights and discuss Light Mobility, Lightmass, and Dynamic Lights.