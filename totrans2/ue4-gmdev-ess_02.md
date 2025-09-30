# Chapter 2. Importing Assets

In the previous chapter, you learned the basics of Unreal Engine. In this chapter, you will learn about importing assets from Autodesk 3ds Max.

# Creating asset in a DCC application

In the previous chapter, you learned how to use BSP to block out a level. However, we need to replace them with static meshes for better performance and more control of materials, collisions, and so on. We will create models in the **Digital Content Creation** (**DCC**) application (such as Autodesk 3ds Max, Autodesk Maya, Blender, and so on) that are imported into Unreal Engine through Content Browser. Unreal Engine supports the import of both FBX and OBJ but its recommended to use the FBX format.

The following screenshot is an example asset that I will use in this chapter:

![Creating asset in a DCC application](img/B03950_02_01.jpg)

### Note

Note that at the time of writing this book, Unreal Engine import pipeline uses FBX 2014\. Trying to import using a different version might result in incompatibilities.

A few things that you need to keep in mind when modeling are as follows:

*   **Units**: **Unreal Units** (**UU**) are critical when modeling assets for games. Incorrect units will result in assets looking larger or smaller than they are supposed to look. 1 Unreal Unit is equal to 1 cm. The sample character that comes with Unreal Engine 4 is 196 cm high. So when you are modeling assets for Unreal Engine 4, it's best to use a box that is 196 cm high as reference.

    ### Note

    To learn how to change units for Autodesk 3ds Max, you can refer to [https://knowledge.autodesk.com/support/3ds-max/learn-explore/caas/CloudHelp/cloudhelp/2015/ENU/3DSMax/files/GUID-69E92759-6CD9-4663-B993-635D081853D2-htm.html](https://knowledge.autodesk.com/support/3ds-max/learn-explore/caas/CloudHelp/cloudhelp/2015/ENU/3DSMax/files/GUID-69E92759-6CD9-4663-B993-635D081853D2-htm.html).

    To learn how to change units for Blender, you can refer to [http://www.katsbits.com/tutorials/blender/metric-imperial-units.php](http://www.katsbits.com/tutorials/blender/metric-imperial-units.php).

    *   **Pivot Point**: This represents the local center and local coordinate system of an object. When you import a mesh into Unreal Engine, the pivot point of that mesh (as it was in your DCC application) determines the point where any transformation (such as move, rotate, and scale) will be performed. Generally, it is best to keep your meshes at origin (0, 0, 0) and set your pivot point to one corner of the mesh for proper alignment in Unreal Engine.
    *   **Triangulation**: Remember that, the Unreal Engine importer will automatically convert the quads to triangles so there is no skipping from triangles.
    *   **UV**: When you do UVs for assets, you can go beyond the 0-1 space, especially when you are dealing with big objects. UV channel 1 (which is channel 0 in Unreal) is used for texturing and UV channel 2 (which is channel 1 in Unreal) is used for lightmaps.

# Creating collision meshes

You can create collision meshes and export them with your asset. Unreal Engine 4 provides a collision generator for static meshes but there are times when we have to create our own custom collision shapes especially if the mesh has an opening (such as doors or walls with window cutouts). In this section, we will see both options.

### Tip

Collision shapes should always stay simple because it is much faster to calculate simple shapes.

## Custom collision shapes

Collision meshes are identified by Unreal importer based on their names. There are three types of collision shapes that you can define. They are as follows:

*   **UBX_MeshName**: UBX stands for Unreal Box and as the name says, it should be in a box shape. You cannot move the vertices in any way or else it will not work.
*   **USP_MeshName**: USP stands for Unreal Sphere and as the name says, it should be in the sphere shape. The number of segments of this sphere does not matter (although somewhere around 6-10 seems to be good) but you cannot move any vertices around.
*   **UCX_MeshName**: UCX stands for Unreal Convex and as the name says, it should be a convex shape and should not be hollow or dented. This is the most commonly used collision shape because basic shapes such as boxes and spheres can be generated right inside Unreal.

In the following screenshot, you can see the red wireframe object, which is what I created for the collision shape:

![Custom collision shapes](img/B03950_02_02.jpg)

## Unreal Engine 4 collision generator

Collision shapes for static meshes can be generated inside the static mesh editor. To open this editor, double-click on a static mesh asset in **Content Browser** and click on the **Collision** menu, which will then list all the options for **Collision**.

![Unreal Engine 4 collision generator](img/B03950_02_03.jpg)

# Simple shapes

The first three options in this menu are simple shapes and they are as follows:

*   **Sphere Collision**: This creates a simple sphere collision shape
*   **Capsule Collision**: This creates a simple capsule collision shape
*   **Box Collision**: This creates a simple box collision shape![Simple shapes](img/B03950_02_04.jpg)

## K-DOP shapes

**K Discrete Oriented Polytope** (**K-DOP**) shapes are basically bounding volumes. The numbers (10, 18, and 26) represents the K-axis aligned planes.

# Auto convex collision

This option is used to create much more accurate collision shapes for your models. Once you click on this option, a new dock window appears at the bottom-right corner of static mesh editor; using **Max Hulls** (the number of hulls to be created to best match the shape of the object) and **Max Hull Verts** (which determines the complexity of the collision hulls) you can create more complex collision shapes for your **Static Mesh**.

As you can see in the following screenshot, the auto convex result is reasonably accurate:

![Auto convex collision](img/B03950_02_05.jpg)

### Tip

Collision shapes support transformation (move, rotate, and scale) and you can duplicate them to have multiple collisions. Click on the collision shape inside static mesh editor and you can switch between move, rotate, and scale using *W*, *E*, and *R*. You can use *Alt* + left click drag (or *Ctrl* + *W*) to duplicate an existing collision.

# Materials

Unreal Engine can import materials and textures to apply to the mesh while exporting from 3D application. From Autodesk 3ds Max, only the basic materials are supported. They are **Standard** and **Multi/Sub-Object**. In those basic materials, only specific features are supported. This means FBX will not export all settings but it supports certain maps or textures used in that material type.

In the following example, you can see a model with multiple materials assigned.

### Note

Note that it is very important to have unique names for each sub material in the **Multi/Sub-Object** material. Each sub material has a unique name as shown in the following screenshot:

![Materials](img/B03950_02_06.jpg)

# LOD

**Level of Detail** (**LOD**) is a way to limit the impact of meshes as they get farther away from the camera. Each LOD will have reduced triangles and vertices compared to the previous one and can have simpler materials. That means base LOD (**LOD 0**) will be the high quality mesh that appears when the player is near. As the player goes farther from that object, it will change to **LOD 1** with reduced triangles and vertices than **LOD 0** and as the player goes even farther away it will switch to **LOD 2**, which has much fewer triangles and vertices than **LOD 1**.

The following figure should give you an idea about what LOD does. The mesh on the left is base LOD (**LOD 0**), the middle is **LOD 1**, and the right is **LOD 2**.

### Note

More information about LODs can be found at [https://docs.unrealengine.com/latest/INT/Engine/Content/Types/StaticMeshes/HowTo/LODs/index.html](https://docs.unrealengine.com/latest/INT/Engine/Content/Types/StaticMeshes/HowTo/LODs/index.html).

![LOD](img/B03950_02_07.jpg)

# Exporting and importing

We will now cover how to export and import a mesh into Unreal.

## Exporting

Exporting a mesh is a pretty straightforward process. You can export multiple meshes in a single FBX file or export each mesh individually. Unreal importer can import multiple meshes as separate assets or combine them as a single asset by enabling the **Combine Meshes** option at import time.

In the following screenshot, you can see that I have selected both the collision mesh and the model for exporting:

![Exporting](img/B03950_02_08.jpg)

### Note

**Smoothing Groups** should be turned on when exporting, otherwise Unreal Engine will show a warning when importing.

## Importing

Importing a mesh into Unreal is simple. There are three ways you can import. They are explained here.

### Context menu

You can right-click inside **Content Browser** and select **Import to <Your folder name>**.

![Context menu](img/B03950_02_09.jpg)

### Drag and drop

As the name states, you can easily drag a FBX or OBJ model from **Windows Explorer** to **Content Browser** to import.

### Content Browser import

**Content Browser** has an **Import** button that you can use to import meshes.

![Content Browser import](img/B03950_02_10.jpg)

### Automatic import

If you place FBX files in your project's **Content** folder (including any subfolders), Unreal will automatically detect this and trigger the import process (if you have the editor open. Otherwise, the next time you run it).

### Configuring automatic import

You can choose whether you want this option enabled or disabled. To configure, go to **Edit** | **Editor Preferences** | **Loading & Saving** | **Auto Reimport**.

![Configuring automatic import](img/B03950_02_11.jpg)

*   **Monitor Content Directories**: This enables or disables automatic importing of assets.
*   **Directories to Monitor**: This adds or removes a path (it can be a virtual package path such as `\Game\MyContent\` or an absolute path such as `C:\My Contents`) for the engine to monitor new content.
*   **Auto Create Assets**: If enabled, any new FBX files will not be automatically imported.
*   **Auto Delete Assets**: If enabled, and you delete the FBX file from Explorer, Unreal Engine will prompt whether you want to delete the asset file as well.

### Result

When you import your asset, you will see the **Import Options** dialog. You can read all about the import settings here:

![Result](img/B03950_02_12.jpg)

Once you click on **Import** (or **Import All** if you're importing multiple FBX files) you will see the asset in **Content Browser**. In the following screenshot, see how Unreal automatically imported the material from Autodesk 3ds Max:

![Result](img/B03950_02_13.jpg)

If you double-click on the static mesh (**Tower_Example**), you will see the static mesh editor. In the following screenshot, you can see that Unreal successfully imported my custom collision shape too.

![Result](img/B03950_02_14.jpg)

# Summary

In the next chapter, you will learn more about **Materials** and **Textures**.