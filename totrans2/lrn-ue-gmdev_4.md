# Chapter 4. Material and Light

In this chapter, we will learn in detail about the materials and the lights in Unreal Engine 4\. We have grouped both Material and Light together in this chapter because how an object looks is largely determined by both—material and lighting.

Material is what we apply to the surface of an object and it affects how the object looks in the game. Material/Shader programming is a hot ongoing research topic as we always strive to improve the texture performance—seeking higher graphic details/realism/quality with limited CPU/GPU rendering power. Researchers in this area need to find ways to make the models we have in a game look as real as possible, with as little calculations/data size as possible.

Lighting is also a very powerful tool in world creation. There are many uses of light. Lights can create a mood for the level. When effectively used, it can be used to focus attention on objects in the level and guide players through your level. Light also creates shadow. In a game level, shadow needs to be created artificially. Hence, we will also learn how we get shadows rendered appropriately for our game.

# Materials

In the previous chapter, we briefly touched on what a material is and what a texture is. A texture is like a simple image file in the format of `.png`/`.tga`. A material is a combination of different elements, including textures to create a surface property that we apply to our objects in the game. We have also briefly covered what UV coordinates are and how we use them to apply a 2D texture to the surface of a 3D object.

So far, we have only learned how to apply materials that are available in the default Unreal Engine. In this chapter, we will dive deeper into how we can actually create our own custom material in Unreal Engine 4\. Fundamentally, the material creation for the objects falls into the scope of an artist. For special customized textures, they are sometimes hand painted by 2D artists using tools such as Photoshop or taken from photographs of textures from the exact objects we want, or similar objects. Textures can also be tweaked from existing texture collection to create the customized material that is needed for the 3D models. Due to the vast number of realistic textures needed, textures are sometimes also generated algorithmically by the programmers to allow more control over its final look. This is also an important research area for the advancing materials for computer graphics.

Material manipulation here falls under the scope of a specialized group of programmers known as **graphic programmers**. They are sometimes also researchers that look into ways to better compress texture, improve rendering performance, and create special dynamic material manipulation.

# The Material Editor

In Unreal Engine 4, material manipulation can be achieved using the Material Editor. What this editor offers is the ability to create material expressions. Material expressions work together to create an overall surface property for the material. You can think of them as mathematical formulas that add/multiply together to affect the properties of a material. The Material Editor makes it easy to edit/formulate material expressions to create customized material and provides the capability to quickly preview the changes in the game. Through Unreal's Blueprint capabilities and programming, we can also achieve dynamic manipulation of materials as needed by the game.

## The rendering system

The rendering system in Unreal Engine 4 uses the DirectX 11 pipeline, which includes deferred shading, global illumination, lit translucency, and post processing. Unreal Engine 4 has also started branching to work with the newest DirectX 12 pipeline for Windows 10, and DirectX 12 capabilities will be available to all.

## Physical Based Shading Model

Unreal Engine 4 uses the **Physical Based Shading Model** (**PBSP**). This is a concept used in many modern day game engines. It uses an approximation of what light does in order to give an object its properties. Using this concept, we give values (0 to 1) to these four properties: **Base Color**, **Roughness**, **Metallic**, and **Specular** to approximate the visual properties.

For example, the bark of a tree trunk is normally brown, rough, and not very reflective. Based on what we know about how the bark should look like, we would probably set the metallic value to low value, roughness to a high value, and the base color to display brown with a low specular value.

This improves the process of creating materials as it is more intuitive as visual properties are governed by how light reacts, instead of the old method where we approximate the visual properties based on how light should behave.

For those who are familiar with the old terms used to describe material properties, you can think of it as having **Diffuse Color** and **Specular Power** replaced by **Base Color**, **Metallic**, and **Roughness**.

The advantage of using PBSP is that we can better approximate material properties with more accuracy.

## High Level Shading Language

The Material Editor enables visual scripting of the **High Level Shading Language** (**HLSL**), using a network of nodes and connection. Those who are completely new to the concept of shaders or HLSL should go on to read the next section about shaders, DirectX and HLSL first, so that you have the basic foundation on how the computer renders material information on the screen. HLSL is aproprietary shading language developed by Microsoft. OpenGL has its own version, known as GLSL. HLSL is the programming language used to program the stages in the graphics pipeline. It uses variables that are similar to C programming and has many intrinsic functions that are already written and available for use by simply calling the function.HLSL shaders can be compiled at author-time or at runtime, and set at runtime into the appropriate pipeline stage.

## Getting started

To open the Material Editor in Unreal Engine 4, go to **Content Browser** | **Material** and double-click on any material asset. Alternatively, you can select a material asset, right-click to open the context menu and select **Edit** to view that asset in the Material Editor.

If you want to learn how to create a new material, you can try out the example, which is covered in the upcoming section.

## Creating a simple custom material

We will continue to use the levels we have created. Open `Chapter3Level.umap` and rename it `Chapter4Level.umap` to prevent overwriting what we have completed at the end of the previous chapter.

To create a new Material asset in our game package, go to **Content Browser** | **Material**. With **Material** selected, right-click to open the contextual menu, navigate to **New Asset** | **Material**. This creates the new material in the `Material` folder (we want to place assets in logical folders so that we can find game assets easily). Alternatively, you can go to **Content Browser** | **New** | **Material**.

![Creating a simple custom material](img/B03679_04_01.jpg)

Rename the new material to `MyMaterial`. The following screenshot shows the new **MyMaterial** correctly created:

![Creating a simple custom material](img/B03679_04_02.jpg)

Note that the thumbnail display for the new **MyMaterial** shows a grayed-out checkered material. This is the default material when no material has been applied.

To open the Material Editor to start designing our material, double-click on **MyMaterial**. The following screenshot shows the Material Editor with a blank new material. The spherical preview of the material shows up as black since no properties have been defined yet.

![Creating a simple custom material](img/B03679_04_03.jpg)

Let's start to define some properties for the **MyMaterial** node to create our very own unique material. **Base Color**, **Metallic**, and **Roughness** are the three values we will learn to configure first.

**Base Color** is defined by the red, green, and blue values in the form of a vector. To do so, we will drag and drop **Constant3Vector** from **MyPalette** on the right-hand side into the main window where the **MyMaterial** node is in. Alternatively, you can right-click to open the context menu and type `vector` into the search box to filter the list. Click and select **Constant3Vector** to create the node. Double-click on the **Constant3Vector** to display the **Color Picker** window. The following screenshot shows the setting of **Constant3Vector** we want to use to create a material for a red wall. (**R** = **0.4**, **G** = **0.0**, **B** = **0.0**, **H** = **0.0**, **S** = **1.0**, **V** = **0.4**):

![Creating a simple custom material](img/B03679_04_04.jpg)

Connect the **Constant3Vector** to the **MyMaterial** node as shown in the following screenshot by clicking and dragging from the small circle from the **Constant3Vector** node to the small circle next to the **Base Color** property in the **MyMaterial** node. This **Constant3Vector** node now provides the base color to the material. Notice how the spherical preview on the left updates to show the new color. If the color is not updated automatically, make sure that the **Live Preview** setting on the top ribbon is selected.

![Creating a simple custom material](img/B03679_04_05.jpg)

Now, let us set the **Metallic** value for the material. This property takes a numerical value from 0 to 1, where 1 is for a 100% metal. To create an input for a value, click and drag **Constant** from **MyPalette** or right-click in the Material Editor to open the menu; type in `Constant` into the search box to filter and select **Constant** from the filtered list. To edit the value in the constant, click on the **Constant** node to display the **Details** window and fill in the value. The following screenshot shows how the material would look if **Metallic** is set to 1:

![Creating a simple custom material](img/B03679_04_06.jpg)

After seeing how the **Metallic** value affects the material, let us see what **Roughness** does. **Roughness** also takes a **Constant** value from 0 to 1, where 0 is completely smooth and makes the surface very reflective. The left-hand screenshot shows how the material looks when **Roughness** is set to 0, whereas the right-hand screenshot shows how the material will look when **Roughness** is set to 1:

![Creating a simple custom material](img/B03679_04_07.jpg)

We want to use this new material to texture the walls. So, we have set **Metallic** as **0.3** and **Roughness** as **0.7**. The following screenshot shows the final settings we have for our first custom material:

![Creating a simple custom material](img/B03679_04_08.jpg)

Go to **MyMaterial** in **Content Browser** and duplicate **MyMaterial**. Rename it `MyWall_Grey`. Change the base color to gray using the following values as shown in the picker node for the **Constant3Vector** value for **Base Color**. (**R** = **0.185**, **G** = **0.185**, **B** = **0.185**, **H** = **0.0**, **S** = **0.0**, **V** = **0.185**):

![Creating a simple custom material](img/B03679_04_09.jpg)

The following screenshot shows the links for the **MyWall_Grey** node. (**Metallic** = **0.3**, **Roughness** = **0.7**):

![Creating a simple custom material](img/B03679_04_10.jpg)

## Creating custom material using simple textures

To create a material using textures, we must first select a texture that is suitable. Textures can be created by artists or taken from photos of materials. For learning purposes, you can find suitable free source images from the Web, such as [www.textures.com](http://www.textures.com), and use them. Remember to check for conditions of usage and other license-related clauses, if you plan to publish it in a game.

There are two types of textures we need for a custom material using a simple texture. First, the actual texture that we want to use. For now, let us keep this selection simple and straightforward. Select this texture based on the color and it should have the overall properties of what you want the material to look like. Next, we need a normal texture. If you still remember what a normal map is, it controls the bumps on a surface. The normal map gives the grooves in a material. Both of these textures will work together to give you a realistic-looking material that you can use in your game.

In this example, we will create another wood texture that we will use to replace the wood texture from the default package that we have already applied in the room.

Here, we will start first by importing the textures that we need in Unreal Engine. Go to **Content Browser** | **Textures**. Then click on the **Import** button at the top. This opens up a window to browse to the location of your texture. Navigate to the folder location where your texture is saved, select the texture and click on **Open**. Note that if you are importing textures that are not in the power of two (256 x 256, 1024 x 1024, and so on), you would have a warning message. Textures that are not in the power of two should be avoided due to poor memory usage. If you are importing the example images that I am using, they are already converted to the power of two so you would not get this warning message on screen.

Import both **T_Wood_Light** and **T_Wood_Light_N**. **T_Wood_Light** will be used as the main texture, we want to have, and **T_Wood_Light_N** is the normal map texture, which we will use for this wood.

Next, we follow the same steps to create a new material, as in the previous example. Go to **Content Browser** | **Material**. With the **Material** folder selected, to open the contextual menu, navigate to **New Asset** | **Material**. Rename the new material `MyWood`.

Now, instead of selecting **Constant3Vector** to provide values to the base color, we will use **TextureSample**. Go to **MyPalette** and type in `Texture` to filter the list. Select **TextureSample**, drag and drop it into the Material Editor. Click on the **TextureSample** node to display the **Details** panel, as shown in the following screenshot. On the **Details** panel, go to **Material Expression Texture Base** and click on the small arrow next to it. This opens up a popup with all the suitable assets that you can use. Scroll down to select **T_Wood_Light**.

![Creating custom material using simple textures](img/B03679_04_11.jpg)

Now, we have configured **TextureSample** with the wood texture that we have imported into the editor earlier. Connect **TextureSample** by clicking on the white hollow circle connector, dragging it and dropping it on the **Base Color** connector on the **MyWood** node.

Repeat the same steps to create a **TextureSample** node for the **T_Wood_Light_N** normal map texture and connect it to the **Normal** input for **MyWood**.

The following screenshot shows the settings that we want to have for **MyWood**. To have a little glossy feel for our wood texture, set **Roughness** to **0.2** by using a **Constant** node. (Recap: drag and drop a **Constant** node from **MyPalette** and set the value to **0.2**, connect it to the **Roughness** input of **MyWood**.)

![Creating custom material using simple textures](img/B03679_04_12.jpg)

## Using custom materials to transform the level

Using the custom materials that we have created in the previous two examples, we will replace the current materials that we have used.

The following screenshot shows the before and after look of the first room. Notice how the new custom materials have transformed the room into a modern looking room.

![Using custom materials to transform the level](img/B03679_04_13.jpg)

From the preceding screenshot, we also have added a Point Light and placed it onto the lamp prop, making it seem to be emitting light. The following screenshot shows the Point Light setting we have used (**Light Intensity** = **1000.0**, **Attenuation Radius** = **1000.0**):

![Using custom materials to transform the level](img/B03679_04_14.jpg)

Next, we added a ceiling to cover up the room. The ceiling of the wall uses the same box geometry as the rest of the walls. We have applied the **M_Basic_Wall** material onto it.

Then, we use the red wall material (**MyMaterial**) to replace the material on wall with the door frame. The gray wall material (**MyWall_Grey**) is used to replace the brick material for the walls at the side. The glossy wood material (**MyWood**) is used to replace the wooden floor material.

# Rendering pipeline

For an image to appear on the screen, the computer must draw the images on the screen to display it. The sequence of steps to create a 2D representation of a scene by using both 2D and 3D data information is known as the graphics or rendering pipeline. Computer hardware such as **central processing unit** (**CPU**) and **graphics processing unit** (**GPU**) are used to calculate and manipulate the input data needed for drawing the 3D scene.

As games are interactive and rely heavily on real-time rendering, the amount of data necessary for rendering moving scenes is huge. Coordinate position, color, and all display information needs to be calculated for each vertex of the triangle polygon and at the same time, taking into account the effect of overlapping polygons before they can be displayed on screen correctly. Hence, it is very crucial to optimize both the CPU and GPU capabilities to process this data and deliver them timely on the screen. Continuous improvement in this area has been made over the years to allow better quality images to be rendered at higher frame rates for a better visual effect. At this point, games should run at a minimum frame rate of 30fps in order for players to have a reasonable gaming experience.

The rendering pipeline today uses a series of programmable shaders to manipulate information about an image before displaying the image on the screen. We'll cover shaders and Direct3D 11 graphics pipeline in more detail in the upcoming section.

# Shaders

Shaders can be thought of as a sequence of programming codes that tells a computer how an image should be drawn. Different shaders govern different properties of an image. For example, Vertex Shaders give properties such as position, color, and UV coordinates for individual vertices. Another important purpose of vertex shaders is to transform vertices with 3D coordinates into the 2D screen space for display. Pixel shaders processes pixels to provide color, z-depth, and alpha value information. Geometry shader is responsible for processing data at the level of a primitive (triangle, line, and vertex).

Data information from an image is passed from one shader to the next for processing before they are finally output through a frame buffer.

Shaders are also used to incorporate post-processing effects such as Volumetric Lighting, HDR, and Bloom effects to accentuate images in a game.

The language which shaders are programmed in depends on the target environment. For Direct3D, the official language is HLSL. For OpenGL, the official shading language is **OpenGL Shading Language** (**GLSL**).

Since most shaders are coded for a GPU, major GPU makers Nvidia and AMD have also tried developing their own languages that can output for both OpenGL and Direct3D shaders. Nvidia developed Cg (deprecated now after version 3.1 in 2012) and AMD developed Mantle (used in some games, such as *Battlefield 4*, that were released in 2014 and seems to be gaining popularity among developers). Apple has also recently released its own shading language known as **Metal Shading Language** for iOS 8 in September 2014 to increase the performance benefits for iOS. Kronos has also announced a next generation graphics API based on OpenGL known as **Vulkan** in early 2015, which appears to be strongly supported by member companies such as Valve Corporation.

The following image is taken from a Direct3D 11 graphics pipeline on MSDN ([http://msdn.microsoft.com/en-us/library/windows/desktop/ff476882(v=vs.85).aspx](http://msdn.microsoft.com/en-us/library/windows/desktop/ff476882(v=vs.85).aspx)). It shows the programmable stages, which data can flow through to generate real-time graphics for our game, known as the rendering pipeline state representation.

![Shaders](img/B03679_04_15.jpg)

The information here is taken from Microsoft MSDN page. You can use the Direct3D 11API to configure all of the stages. Stages such as vertex, hull, domain, geometry, and pixel-shader (those with the rounded rectangular blocks), are programmable using HLSL. The ability to configure this pipeline programmatically makes it flexible for the game graphics rendering.

What each stage does is explained as follows:

| Stage | Function |
| --- | --- |
| Input-assembler | This stage supplies data (in the form of triangles, lines, and points) to the pipeline. |
| Vertex-shader | This stage processes vertices such as undergoing transformations, skinning, and lighting. The number of vertices does not change after undergoing this stage. |
| Geometry-shader | This stage processes entire geometry primitives such as triangles, lines, and a single vertex for a point. |
| Stream-output | This stage serves to stream primitive data from the pipeline to memory while on its way to the rasterizer. |
| Rasterizer | This clips primitives and prepare the primitives for the pixel-shader. |
| Pixel-shader | Pixel manipulation is done here. Each pixel in the primitive is processed here, for example, pixel color. |
| Output-merger | This stage combines the various output data (pixel-shader values, depth, and stencil information) with the contents of the render target and depth/stencil buffers to generate the final pipeline result. |
| Hull-shader, tessellator, and domain-shader | These tessellation stages convert higher-order surfaces to triangles to prepare for rendering. |

To help you better visualize what happens in each of the stages, the following image shows a very good illustration of a simplified rendering pipeline for vertices only. The image is taken from an old Cg tutorial. Note that different APIs have different pipelines but rely on similar basic concepts in rendering (source: [http://goanna.cs.rmit.edu.au/~gl/teaching/rtr&3dgp/notes/pipeline.html](http://goanna.cs.rmit.edu.au/~gl/teaching/rtr&3dgp/notes/pipeline.html)).

![Shaders](img/B03679_04_16.jpg)

Example flow of how graphics is displayed:

*   The CPU sends instructions (compiled shading language programs) and geometry data to the graphics processing unit, located on the graphics card.
*   The data is passed through into the vertex shader where vertices are transformed.
*   If the geometry shader is active in the GPU, the geometry changes are performed in the scene.
*   If a tessellation shader is active in the GPU, the geometries in the scene can be subdivided. The calculated geometry is triangulated (subdivided into triangles).
*   Triangles are broken down into fragments. Fragment quads are modified according to the fragment shader.
*   To create the feel of depth, the z buffer value is set for the fragments and then sent to the frame buffer for displaying.

# APIs – DirectX and OpenGL

Both DirectX and OpenGL are collections of **application programming interfaces** (**APIs**) used for handling multimedia information in a computer. They are the two most common APIs used today for video cards.

DirectX is created by Microsoft to allow multimedia related hardware, such as GPU, to communicate with the Windows system. OpenGL is the open source version that can be used on many operating system including Mac OS.

The decision to use DirectX or OpenGL APIs to program is dependent on operating system of the target machine.

## DirectX

Unreal Engine 4 was first launched using DirectX11\. Following the announcement that DirectX 12 ships with Windows 10, Unreal has created a DirectX 12 branch from the 4.4 version to allow developers to start creating games using this new DirectX 12.

An easy way to identify APIs that are a part of DirectX is that the names all begin with Direct. For computer games, the APIs that we are most concerned about are Direct3D, which is the graphical API for drawing high performance 3D graphics in games, and DirectSound3D, which is for the sound playback.

DirectX APIs are integral in creating high-performance 2D and 3D graphics for the Windows operating system. For example, DirectX11 is supported in Windows Vista, Windows 7 and Windows 8.1\. The latest version of DirectX can be updated through service pack updates. DirectX 12 is known to be shipped with Windows 10.

### DirectX12

Direct3D 12 was announced in 2014 and has been vastly revamped from Direct3D 11 to provide significant performance improvement. This is a very good link to a video posted on the MSDN blog that shows the tech demo for DirectX 12: [http://channel9.msdn.com/Blogs/DirectX-Developer-Blog/DirectX-Techdemo](http://channel9.msdn.com/Blogs/DirectX-Developer-Blog/DirectX-Techdemo).

(If you are unfamiliar with Direct3D 11 and have not read the *Shaders* section earlier, read that section before proceeding with the rest of the DirectX section.)

#### Pipeline state representation

If you can recall from the *Shaders* section, we have looked at the programmable pipeline for Direct3D 11\. The following image is the same from the *Shaders* section (taken from MSDN) and it shows a series of programmable shaders:

![Pipeline state representation](img/B03679_04_17.jpg)

In Direct3D 11, each of the stages is configurable independently and each stage is setting states on the hardware independently. Since many stages have the capability to set the same hardware state due to interdependency, this results in hardware mismatch overhead. The following image is an excellent illustration of how hardware mismatch overhead happens:

![Pipeline state representation](img/B03679_04_18.jpg)

The driver will normally record these states from the application (game) first and wait until the draw time, when it is ready to send it to the display monitor. At draw time, these states are then queried in a control loop before they are is translated into a GPU code for the hardware in order to render the correct scene for the game. This creates an additional overhead to record and query for all the states at draw time.

In Direct3D 12, some programmable stages are grouped to form a single object known as **pipeline state object** (**PSO**) so that the each hardware state is set only once by the entire group, preventing hardware mismatch overhead. These states can now be used directly, instead of having to spend resources computing the resulting hardware states before the draw call. This reduces the draw call overhead, allowing more draw calls per frame. The PSO that is in use can still be changed dynamically based on whatever hardware native instructions and states that are required.

![Pipeline state representation](img/B03679_04_19.jpg)

#### Work submission

In Direct3D 11, work submission to the GPU is immediate. What is new in Direct3D 12 is that it uses command lists and bundles that contain the entire information needed to execute a particular workload.

Immediate work submission in Direct3D 11 means that information is passed as a single stream of command to the GPU and due to the lack of the entire information, these commands are often deferred until the actual work can be done.

When work submission is grouped in the self-contained command list, the drivers can precompute all the necessary GPU commands and then send that list to the GPU, making Direct3D 12 work submission a more efficient process. Additionally, the use of bundles can be thought of as a small list of commands that are grouped to create a particular object. When this object needs to be duplicated on screen, this bundle of commands can be "played back" to create the duplicated object. This further reduces computational time needed in Direct3D 12.

#### Resource access

In Direct3D 11, the game creates resource views that bind these views to slots at the shaders. These shaders then read the data from these explicit bound slots during a draw call. If the game wants to draw using different resources, it will be done in the next draw call with a different view.

In Direct3D 12, you can create various resource views by using descriptor heaps. Each descriptor heap can be customized to be linked to a specific shader using specific resources. This flexibility to design the descriptor heap allows you to have full control over the resource usage pattern, fully utilizing modern hardware capabilities. You are also able to describe more than one descriptor heap that is indexed to allow easy flexibility to swap heaps, to complete a single draw call.

# Lights

We have briefly gone through the types of light in [Chapter 1](ch01.html "Chapter 1. An Overview of Unreal Engine"), *An Overview of Unreal Engine*. Let us do a quick recap first. Directional Light emits beams of parallel lights. Point Light emits light like a light bulb (from a single point radially outward in all directions). Spot Light emits light in a conical shape outwards and Sky Light mimics light from the sky downwards on the objects in the level.

In this chapter, we will learn how to use these basic lights to illuminate an interior area. We have already placed a Point Light in [Chapter 2](ch02.html "Chapter 2. Creating Your First Level"), *Creating Your First Level*, and learned how to adjust its intensity to 1700\. Here in this chapter, we will learn more about the parameters that we can adjust with each type of light to create the lighting that we want.

Let us first view a level that has been illuminated using these Unreal lights. Load `Chapter4Level_Prebuilt.umap`, build and play the level to look around. Click on the lights that are placed in the level and you will notice that most of lights used are Point or Spot Light. These two forms of lights are quite commonly found in interior lighting.

The next section will guide you to extend the level on your own. Alternatively, you can use the `Chapter4Level_Prebuilt` level to help you along in the creation of your own level since it does take a fair amount of time to create the entire level. If you wish to skip to the next section, feel free to simply use the prebuilt version of the map provided, and go through the other examples in this chapter using the prebuilt map as a reference. However, it will be a great opportunity to revise what you have learned in the previous chapters and extend the level on your own.

Before we embark on the optional exercise to extend the level, let us go through a few tutorial examples on how we can place and configure the different types of light.

## Configuring a Point Light with more settings

Open `Chapter4Level.umap` and rename it `Chapter4Level_PointLight.umap`.

Go to **Modes** | **Lights**, drag and drop a Point Light into the level. As Point Light emits light equally in all directions from a single point, **Attenuation Radius**, **Intensity**, and **Color** are the three most common values that are configured for a Point Light.

### Attenuation Radius

The following screenshot shows when the Point Light has its default **Attenuation Radius** of **1000**. The radius of the three blue circles is based on the attenuation radius of the Point Light and is used to show its area of effect on the environment.

![Attenuation Radius](img/B03679_04_20.jpg)

The following screenshot shows when the attenuation radius is reduced to 500\. In this situation, you probably cannot see any difference in the lighting since the radius is still larger than the room itself:

![Attenuation Radius](img/B03679_04_21.jpg)

Now, let us take a look at what happens when we adjust the radius much smaller. The following screenshot shows the difference in light brightness when the radius changes. The image on the left is when the attenuation radius is set as 500 and the right when attenuation radius is set as 10.

![Attenuation Radius](img/B03679_04_22.jpg)

### Intensity

Another setting for Point Light is **Intensity**. Intensity affects the brightness of the light. You can play around the Intensity value to adjust the brightness of the light. Before we determine what value to use for this field and how bright we want our light to be, you should be aware of another setting, **Use Inverse Squared Falloff**.

#### Use Inverse Squared Falloff

Point Lights and Spot Lights have physically based inverse squared falloff set on, as default. This setting is configurable as a checkbox found in the **Light** details under **Advanced**. The following screenshot shows where this property is found in the **Details** panel:

![Use Inverse Squared Falloff](img/B03679_04_23.jpg)

Inverse squared falloff is a physics law that describes how light intensity naturally fades over distance. When we have this setting, the units for intensity use the same units as the lights we have in the real world, in lumens. When inverse squared distance falloff is not used, intensity becomes just a value.

In the previous chapter where we have added our first Point Light, we have set intensity as 1700\. This is equivalent to the brightness of a light bulb that has 1700 lumens because inverse squared distance falloff is used.

### Color

To adjust the color of Point Light, go to **Light** | **Color**. The following screenshot shows how the color of the light can be adjusted by specifying the RGB values or using the color picker to select the desired color:

![Color](img/B03679_04_24.jpg)

## Adding and configuring a Spot Light

Open `Chapter4Level.umap` and rename it `Chapter4Level_SpotLight.umap`. Go to **Modes** | **Lights**, drag and drop a Spot Light into the level.

The brightness, visible influence radius, and color of a Spot Light can be configured in the same way as the Point Light through the value of **Intensity**, **Attenuation Radius**, and **Color**.

Since Point Light has light emitting in all directions and a Spot Light emits light from a single point outwards in a conical shape with a direction, the Spot Light has additional properties such as inner cone and outer cone angle, which are configurable.

### Inner cone and outer cone angle

The unit for the outer cone angle and inner cone angle is in degrees. The following screenshot shows the light radius that the spotlight has when the outer cone angle = 20 (on the left) and outer cone angle = 15 (on the right). The inner cone angle value did not produce much visible results in the screenshot, so very often the value is 0\. However, the inner cone angle can be used to provide light in the center of the cone. This would be more visible for lights with a wider spread and certain IES Profiles.

![Inner cone and outer cone angle](img/B03679_04_25.jpg)

## Using the IES Profile

Open `Chapter4Level_PointLight.umap` and rename it `Chapter4Level_IESProfile.umap`.

IES Light Profile is a file that contains information that describes how a light will look. This is created by light manufacturers and can be downloaded from the manufacturers' websites. These profiles could be used in architectural models to render scenes with realistic lighting. In the same way, the IES Profile information can be used in Unreal Engine 4 to render more realistic lights. IES Light Profiles can be applied to a Point Light or a Spot Light.

### Downloading IES Light Profiles

IES Light Profiles can be downloaded from light manufacturers' websites. Here's a few that you can use:

*   **Cooper** **Industries**: [http://www.cooperindustries.com/content/public/en/lighting/resources/design_center_tools/photometric_tool_box.html](http://www.cooperindustries.com/content/public/en/lighting/resources/design_center_tools/photometric_tool_box.html)
*   **Philips**: [http://www.usa.lighting.philips.com/connect/tools_literature/photometric_data_1.wpd](http://www.usa.lighting.philips.com/connect/tools_literature/photometric_data_1.wpd)
*   **Lithonia**: [http://www.lithonia.com/photometrics.aspx](http://www.lithonia.com/photometrics.aspx)

### Importing IES Profiles into the Unreal Engine Editor

From **Content Browser**, click on **Import**, as shown in the following screenshot:

![Importing IES Profiles into the Unreal Engine Editor](img/B03679_04_26.jpg)

I prefer to have my files in a certain order, hence I have created a new folder called `IESProfile` and created subfolders with the names of the manufacturers to better categorize all the light profiles that were imported.

### Using IES Profiles

Continuing from the previous example, select the right Spot Light which we have in the scene and make sure it is selected. Go to the **Details** panel and scroll down to show the Light Profile of the light.

Then go to **Content Browser** and go to the `IESProfile` folder where we have imported the light profiles into. Click on one of the profiles that you want, drag and drop it on the IES Texture of the Spot Light. Alternatively, you can select the profile and go back to the **Details** panel of the **Light** and click on the arrow next to **IES Texture** to apply the profile on the Spot Light. In the following screenshot, I applied one of the profiles downloaded from the Panasonic website labeled **144907**.

![Using IES Profiles](img/B03679_04_27.jpg)

I reconfigured the Spot Light with **Intensity** = **1000**, **Attenuation Radius** = **1000**, **Outer Cone Angle** = **40**, and **Inner Cone Angle** = **0**.

Next, I deleted the other Spot Light and replaced it with a Point Light where I set **Intensity** = **1000** and **Attenuation Radius** = **1000**. I also set the **Rotation-Y** = **-90** and then applied the same IES Profile to it. The following screenshot shows the difference when the same light profile is applied to a Spot Light and a Point Light. Note that the spread of the light in the Spot Light is reduced. This reinforces the concept that a Spot Light provides a conical shaped light with a direction spreading from the point source outwards. The outer cone angle determines this spread. The point light emits light in all directions and equally out, so it did not attenuate the light profile settings allowing the full design of this light profile to be displayed on the screen. This is one thing to keep in mind while using the IES Light Profile and which types of light to use them on.

![Using IES Profiles](img/B03679_04_28.jpg)

## Adding and configuring a Directional Light

Open `Chapter4Level.umap` and rename it `Chapter4Level_DirectionalLight.umap`.

We have already added a Directional Light into our level in [Chapter 2](ch02.html "Chapter 2. Creating Your First Level"), *Creating Your First Level*, and it provides parallel beams of light into the level.

Directional Light can also be used to light the level by controlling the direction of the sun. The screenshot on the left shows the Directional Light when the **Atmosphere Sun Light** checkbox is unchecked. The screenshot on the right shows the Directional Light when the **Atmosphere Sun Light** checkbox is checked. When the **Atmosphere Sun Light** checkbox is checked, you can control the direction of the sunlight by adjusting the rotation of Directional Light.

![Adding and configuring a Directional Light](img/B03679_04_29.jpg)

The following screenshot shows how this looks when **Rotation-Y** = **0**. This looks like an early sunset scene:

![Adding and configuring a Directional Light](img/B03679_04_30.jpg)

### Example – adding and configuring a Sky light

Open `Chapter4Level_DirectionalLight.umap` and rename it `Chapter4Level_Skylight.umap`.

In the previous example, we have added sunlight control in the Directional Light. Build and compile to see how the level now looks.

Now, let us add a Sky Light into the level by going to **Modes** | **Lights** and then clicking and dragging Sky Light into the level. When adding a Sky Light to the level, always remember to build and compile first in order to see the effect of the Sky Light.

What does a Sky Light do? Sky Light models the color/light from the sky and is used to light up the external areas of the level. So the external areas of the level look more realistic as the color/light is reflecting off the surfaces (instead of using simple white/colored light).

The following screenshot shows the effect of a Sky Light. The left image shows the Sky Light not in the level. The right one shows the Sky Light. Note that the walls now have a tinge of the color of the sky.

![Example – adding and configuring a Sky light](img/B03679_04_31.jpg)

## Static, stationary, or movable lights

After learning how to place and configure the different lights, we need to consider what kind of lights we need in the level. If you are new to the concept of light, you might want to briefly go through the useful light terms section to help in your understanding.

The following screenshot shows the **Details** panel where you can change a light to be static, stationary, or movable.

![Static, stationary, or movable lights](img/B03679_04_32.jpg)

**Static** and **Stationary** light sounds pretty much similar. What is the difference? When do you want to use a **Static** light and when do you want to use a **Stationary** light?

### Common light/shadow definitions

The common light/shadow definitions are as follows:

*   **Direct Light**: This is the light that is present in the scene directly due to a light source.
*   **Indirect Light**: This is the light in the scene that is not directly from a light source. It is reflected light bouncing around and it comes from all sides.
*   **Light Map**: This is a data structure that stores the light/brightness information about an object. This makes the rendering of the object much quicker because we already know its color/brightness information in advance and it is not necessary to compute this during runtime.
*   **Shadow Map**: This is a process created to make dynamic shadows. It is fundamentally made up of two passes to create shadows. More passes can be added to render nicer shadows.

### Static Light

In a game, we always want to have the best performance, and Static Light will be an excellent option because a Static Light needs only to be precomputed once into a Light Map. So for a Static Light, we have the lowest performance cost but in exchange, we are unable to change how the light looks, move the light, and integrate the effect of this light with moving objects (which means it is unable to create a shadow for the moving object as it moves within the influence of the light) into the environment during gameplay. However, a Static Light can cast shadow on the existing stationary objects that are in the level within its influence of radius. The radius of influence is based on the source radius of the light. In return for low performance cost, a Static Light has quite a bit of limitation. Hence, Static Lights are commonly used in the creation of scenes targeted for devices with low computational power.

### Stationary Light

Stationary Light can be used in situations when we do not need to move, rotate, or change the influence radius of the light during gameplay, but allow the light the capacity to change color and brightness. Indirect Light and shadows are prebaked in Light Map in the same way as Static Light. Direct Light shadows are stored within Shadow Maps.

Stationary Light is medium in performance cost as it is able to create static shadow on static objects through the use of distance field shadow maps. Completely dynamic light and shadows is often more than 20 times more intensive.

### Movable Light

Movable Light is used to cast dynamic light and shadows for the scene. This should be used sparingly in the level, unless absolutely necessary.

## Exercise – extending your game level (optional)

Here are the steps that I have taken to extend the current **Level4** to the prebuilt version of what we have right now. They are by no means the only way to do it. I have simply used a Geometry Brush to extend the level here for simplicity. The following screenshot shows one part of the extended level:

![Exercise – extending your game level (optional)](img/B03679_04_33.jpg)

### Useful tips

Group items in the same area together when possible and rename the entity to help you identify parts of the level more quickly. These simple extra steps can save time when using the editor to create a mock-up of a game level.

### Guidelines

If you plan to extend the game level on your own, open and load `Level4.umap`. Then save map as `Level4_MyPreBuilt.umap`. You can also open a copy of the extended level to copy assets or use it as a quick reference.

#### Area expansion

We will start by extending the floor area of the level.

##### Part 1 – lengthening the current walkway

The short walkway was extended to form an L-shaped walkway. The dimensions of the extended portion are X1200 x Y340 x Z40.

| BSPs needed | X | Y | Z |
| --- | --- | --- | --- |
| Ceiling | 1200 | 400 | 40 |
| Floor | 1200 | 400 | 40 |
| Left wall | 1570 | 30 | 280 |
| Right wall | 1260 | 30 | 280 |

##### Part 2 – creating a big room (living and kitchen area)

The walkway leads to a big room at the end, which is the main living and kitchen area.

| BSPs needed | X | Y | Z |
| --- | --- | --- | --- |
| Ceiling | 2000 | 1600 | 40 |
| Floor | 2000 | 1600 | 40 |
| The left wall dividing the big room and walkway (the wall closest to you as you enter the big room from the walkway) | 30 | 600 | 340 |
| The light wall dividing the big room and walkway (the wall closest to you as you enter the big room from the walkway) | 30 | 600 | 340 |
| The left wall of the big room (where the kitchen area is) | 1200 | 30 | 340 |
| The right wall of the big room (where the dining area is) | 2000 | 30 | 340 |
| The left wall to the door (the wall across the room as you enter from the walkway, where the window seats are) | 30 | 350 | 340 |
| The right wall to the door (the wall across the room as you enter from the walkway, where the long benches are) | 30 | 590 | 340 |
| Door area (consists of brick walls, door frames, and door) |
| Wall filler left | 30 | 130 | 340 |
| Wall filler right | 30 | 126 | 340 |
| Door x 2 | 20 | 116 | 250 |
| Side door frame x 2 | 25 | 4 | 250 |
| Horizontal door frame | 25 | 242 | 5 |
| Side brick wall x 2 | 30 | 52 | 340 |
| Horizontal brick wall | 30 | 242 | 74 |

##### Part 3 – creating a small room along the walkway

To create the walkway to the small room, duplicate the same doorframe that we have created in the first room.

| BSPs needed | X | Y | Z |
| --- | --- | --- | --- |
| Ceiling | 800 | 600 | 40 |
| Floor | 800 | 600 | 40 |
| Side wall x 2 | 30 | 570 | 340 |
| Opposite wall (wall with the windows) | 740 | 30 | 340 |

##### Part 4 – Creating a den area in the big room

| BSPs needed | X | Y | Z |
| --- | --- | --- | --- |
| Sidewall x 2 | 30 | 620 | 340 |
| Wall with shelves | 740 | 30 | 340 |

#### Creating windows and doors

Now that we are done with rooms, we can work on the doors and windows.

##### Part 1 – creating large glass windows for the dining area

To create the windows, we use a subtractive Geometry Brush to create holes in the wall. First, create one of size X144 x Y30 x Z300 and place it right in the middle between the ceiling and ground. Duplicate this and convert it to an additive brush; adjust the size to X142 x Y4 x Z298.

Apply **M_Metal_Copper** for the frame and **M_Glass** to the addition brush, which was just created. Now, group them and duplicate both the brushes four times to create five windows. The screenshot of the dining area windows is shown as follows:

![Part 1 – creating large glass windows for the dining area](img/B03679_04_34.jpg)

##### Part 2 – creating an open window for the window seat

To create the window for the window seat area, create a subtractive geometry brush of size X50 x Y280 x Z220\. For this window, we have a protruding ledge of X50 x Y280 x Z5 at the bottom of the window. Then for the glass, we duplicate the subtractive brush of size X4 x Y278 x Z216, convert it to additive brush and adjust it to fit.

Apply **M_Metal_Brushed** for the frame and **M_Glass** to the addition brush that was just created.

![Part 2 – creating an open window for the window seat](img/B03679_04_35.jpg)

##### Part 3 – creating windows for the room

For the room windows, create a subtractive brush of size X144 x Y40 x Z94\. This is to create a hollow in the wall for the prop frame: **SM_WindowFrame**. Duplicate the subtractive brush and prop to create two windows for the room.

##### Part 4 – creating the main door area

For the main door area, we start by creating the doors and its frame, then the brick walls around the door and lastly, the remaining concrete plain wall.

We have two doors with frames then some brick wall to augment before going back to the usual smooth walls. Here are the dimensions for creating this door area:

| BSPs needed | X | Y | Z |
| --- | --- | --- | --- |
| Actual door x 2 | 20 | 116 | 250 |
| Side frame x 2 | 25 | 4 | 250 |
| Top frame | 25 | 242 | 5 |

Here are the dimensions for creating the area around the door:

| BSPs needed | X | Y | Z |
| --- | --- | --- | --- |
| Brick wall side x 2 | 30 | 52 | 340 |
| Brick wall top | 30 | 242 | 74 |
| Smooth wall left | 30 | 126 | 340 |
| Smooth wall right | 30 | 130 | 360 |

#### Creating basic furniture

Let us begin it part by part as follows.

##### Part 1 – creating a dining table and placing chairs

For the dining table, we will be customizing a wooden table with a table top of size X480 x Y160 x Z12 and two legs each of size X20 x Y120 x Z70 placed 40 from the edge of the table. Material used to texture is **M_Wood_Walnut**.

Then arrange eight chairs around the table using **SM_Chair** from the `Props` folder.

##### Part 2 – decorating the sitting area

There are two low tables in the middle and one low long table at the wall. Place three **SM_Couch** from the `Props` folder around the low tables. Here are the dimensions for the larger table:

| BSPs needed | X | Y | Z |
| --- | --- | --- | --- |
| Square top | 140 | 140 | 8 |
| Leg x 2 | 120 | 12 | 36 |

Here are the dimensions for the smaller table:

| BSPs needed | X | Y | Z |
| --- | --- | --- | --- |
| Leg x 2 | 120 | 12 | 36 |

Here are the dimensions for a low long table at the wall:

| BSPs needed | X | Y | Z |
| --- | --- | --- | --- |
| Block | 100 | 550 | 100 |

##### Part 3 – creating the window seat area

Next to the open window, place a geometry box of size X120 x Y310 x Z100\. This is to create a simplified seat by the window.

##### Part 4 – creating the Japanese seating area

The Japanese square table with surface size X200 x Y200 x Z8 and 4 short legs, each of size X20 x Y20 x Z36) is placed close to the corner of the table.

To create a leg space under the table, I used a subtractive brush (X140 x Y140 x Z40) and placed it on the ground under the table. I used the corner of this subtractive brush as a guide as to where to place the short legs for the table.

##### Part 5 – creating the kitchen cabinet area

This is a simplified block prototype for the kitchen cabinet area. The following are the dimensions for L-shaped area:

| BSPs needed | Material | X | Y | Z |
| --- | --- | --- | --- | --- |
| Shorter L: cabinet under tabletop | **M_Wood_Walnut** | 140 | 450 | 100 |
| Longer L: cabinet under tabletop | **M_Wood_Walnut** | 890 | 140 | 100 |
| Shorter L: tabletop | **M_Metal_Brushed_Nickel** | 150 | 450 | 10 |
| Longer L: tabletop | **M_Metal_Brushed_Nickel** | 900 | 150 | 10 |
| Shorter L: hanging cabinet | **M_Wood_Walnut** | 100 | 500 | 100 |
| Longer L: hanging cabinet | **M_Wood_Walnut** | 900 | 100 | 100 |

The following are the dimensions for the island area (hood):

| BSPs needed | Material | X | Y | Z |
| --- | --- | --- | --- | --- |
| Hood (wooden area) | **M_Wood_Walnut** | 400 | 75 | 60 |
| Hood (metallic area) | **M_Metal_Chrome** | 500 | 150 | 30 |

The following are the dimensions for the island area (table):

| BSPs needed | Material | X | Y | Z |
| --- | --- | --- | --- | --- |
| Cabinet under the table | **M_Wood_Walnut** | 500 | 150 | 100 |
| Tabletop | **M_Metal_Chrome** | 550 | 180 | 10 |
| Sink (use a subtractive brush) | **M_Metal_Chrome** | 100 | 80 | 40 |
| Stovetop | **M_Metal_Burnished_Steel** | 140 | 100 | 5 |

# Summary

In this chapter, we covered in-depth information about materials and lights. We learned how the rendering system works and the underlying graphics pipeline/technology such as Directx 11, DirectX 12, and OpenGL/Vulkan. We also learned how to use the Unreal 4 Material Editor to create custom materials and apply it into your level.

We also explored the different types of lights and adjusting **Intensity**, **Attenuation Radius**, and other settings to customize lights for the level. We also learned how to import IES light profiles from light manufacturer's website to create realistic lights for the level. We learned about the differences between **Static**, **Stationary**, and **Movable** lights and how the different lights cast shadows for the level.

In the next chapter, we will learn about animation and artificial intelligence in games. Stay tuned for more!