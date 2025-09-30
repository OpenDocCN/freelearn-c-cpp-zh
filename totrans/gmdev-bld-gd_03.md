# 3

# Adding and Creating Textures

In a typical 3D workflow, one of the most common properties you would add to a material is texture. A **texture** is an image file that is responsible for the textured look of a model so surfaces don’t show just flat colors. Although objects you come across in real life have a perceived color, they also have a characteristic look that is defined by this property in 3D applications. For example, both a flower and a sandy surface may have a yellow color, but you know a flower’s petal would look smoother, whereas grains of sand would look gritty.

Most day-to-day objects have wear and tear. Look around and you’ll see that most surfaces will either have chipped paint, a slight deformation, or some scratches. Imagine the barrel you designed in the first two chapters has been in use for some time. It’d naturally have a few scratches on the metal rings. You can only go so far by applying colors to your materials and altering the roughness values. If you want to achieve a more realistic look, you’ve got to apply textures to your models.

Some 3D professionals only focus and gain expertise on certain domains. Texturing is one of these domains besides modeling, lighting, and animation. Typically, a texturing specialist will employ the help of classic image editing applications such as *Adobe Photoshop*, *GIMP*, and so on to create textures. Then, the artist will bring these textures into Blender so that they can be applied to surfaces. If you are not skilled in creating textures from scratch, you will learn in this chapter how you can still rely on existing textures out there created by other artists.

Preparing and using textures with the aforementioned workflow often sounds static because you need access to the source file of these textures. Luckily, there is a dynamic way to create your own textures within Blender, so you don’t have to go back and forth between Blender and other software.

This is not a “one is better than the other” situation because each method has its own place and merits. You’ll get to know new parts of Blender to facilitate both methods so you can make an informed decision about which texturing method to use. To that end, we are going to cover the following list of topics:

*   Understanding UVs and texture coordinates
*   Using the UV Editor
*   Importing and applying a texture
*   Creating textures procedurally
*   Exporting your textures

By the end of this chapter, you’ll have learned how to prepare your models for texturing, apply available textures, and create your own textures dynamically. The practice you’ll gain in this chapter will give you insight into choosing the right method of texturing for your projects.

# Technical requirements

This book’s GitHub repo ([https://github.com/PacktPublishing/Game-Development-with-Blender-and-Godot](https://github.com/PacktPublishing/Game-Development-with-Blender-and-Godot)) will have a `Chapter 3` folder with `Start` and `Finish` folders in it for you to compare your work with as you go. These folders also contain other dependencies such as the texture files necessary to follow and complete the exercises.

Although you worked on a barrel in the previous chapters, we’ll only use the standard Blender objects, such as a cube and a plane, to keep things simple so you can focus on the texturing workflow.

# Understanding UVs and texture coordinates

While you are modeling, you are altering the coordinates of the vertices of a model. Thus, you are working with spatial coordinates. To apply a texture over your model, you need to work in a different kind of coordinate system that is called **texture coordinates** or **UVs**. Let’s see how these two terms relate to each other.

The spatial coordinate system is often described with the **XYZ** acronym since we often use X, Y, and Z axes to define the position of 3D objects. Similarly, **UV** is another acronym but it is used in the texturing workflow; the letters U and V were picked to describe the texture coordinate system. So, UV doesn’t really stand for ultraviolet.

The process that maps UV coordinates to XYZ coordinates is called **UV unwrapping**. Via this method, you tell Blender how a graphic file is mapped to XYZ coordinates. If unwrapping sounds counterintuitive, you could try to reverse the process in your mind. What kind of texture would you need so that if you wrapped it around your 3D model, it would fit perfectly? Let’s analyze the following figure where a graphic file that is painted with a checkerboard texture is applied to a standard cube:

![Figure 3.1 – A 2D checkerboard texture wrapping a 3D object ](img/Figure_3.01_B17473.jpg)

Figure 3.1 – A 2D checkerboard texture wrapping a 3D object

In *Figure 3.1*, you see a cube with a checkerboard texture on the left. In the middle part, you see the cube as if gift wrap is being peeled off. Finally, the cube is fully unwrapped on the right side; its texture is laid flat. The texture file is actually all of the checkerboard parts, and it exists as a 2D graphic file.

The reason we are using words such as unwrapping and 2D graphic files is because we are simulating a real-life 3D object on a flat screen. In reality, that cube would occupy a space, have a volume, and it would be full of the material it was made of. For example, a cube that might be a child’s toy made of wood. Or, it might be a six-sided die, most likely made of acrylic. If you cut into it, you’d see the material.

To change the nature of the problem from a 3D volume problem to a 2D graphics problem, you need a new tool. You’ve been working with Blender’s default interface, which is conveniently set up to edit XYZ coordinates. For editing UVs, you need the **UV Editor**, which you will discover in the following section.

# Using the UV Editor

Blender comes with preset workspaces so you can focus on a particular workflow. So far, you’ve been in the **Layout** workspace. You can see it as the active tab just under the header of the application, next to the **Help** menu. You should create a new file and switch to the **UV Editing** workspace by clicking the appropriate tab. *Figure 3.2* is what you’ll see when you are in the **UV Editing** workspace.

![Figure 3.2 – UV Editing is one of many default workspaces in Blender ](img/Figure_3.02_B17473.jpg)

Figure 3.2 – UV Editing is one of many default workspaces in Blender

In the **UV Editing** workspace, the application will mainly be divided into two sections: the left side, which is called **UV Editor**, shows a bunch of squares laid out on a flat surface, and the right side shows the default cube. The black dots you see in **UV Editor** are actually the vertices of the cube in **3D Viewport**. You might notice that if you counted the dots in **UV Editor**, they don’t add up to the number of vertices the cube has. There are more points in **UV Editor** because some of those points will eventually merge once those squares in **UV Editor** are folded around the edges and wrapped around your 3D object.

At this point, all of the vertices of the cube should be selected for you by Blender. However, if you happen to select a vertex of the cube, you’ll see that the squares in **UV Editor** will disappear. That’s because we haven’t turned on the **sync** mode yet. At the top-left corner of **UV Editor**, you’ll see a button with an icon that looks like two diagonal arrows going in opposite directions. If you have that button pressed, you’ll notice that selecting the vertices in either view will synchronize.

When you add a new cube, Blender unwraps that cube by default. The general layout of the vertices in **UV Editor** resembles a T shape, like what you saw in *Figure 3.1*. Similar to **3D Viewport**, the vertices in **UV Editor** will form edges and faces, but it’s all 2D in **UV Editor**. As mentioned earlier, we have converted the 3D-ness of the model to a 2D representation so we can work with graphics files.

**UV Editor** is where you can see how the points in the editor map or correlate to a texture file. To do that, we need to bring a texture file as follows:

1.  Open the `Chapter 3` folder.
2.  Open the `Start` folder.
3.  Drag and drop `pips.png` into the **UV Editor** area.

If you open that PNG file in your computer’s default image viewing application, you’ll notice that it has transparent parts. Its dimensions of 1024x1024 are not fully painted. It just happens that the file’s non-transparent areas come right under the faces in **UV Editor**, therefore the faces in **3D Viewport**.

Powers of two

Sooner or later, you’ll notice that most texture files come in certain standard dimensions such as 512, 1024, 2048, and so on. Although these files don’t have to be square, which means you could actually have 256x512 as dimensions, it’d still pay off to keep either dimension in powers of two. This is due to algorithms that are employed by GPUs so that they run more efficiently.

So far, we have taken advantage of Blender’s default UV layout for a cube and have seen how UV faces can overlap with the texture file we have been previewing in **UV Editor**. However, if you enable **Material Preview** in **3D Viewport**, you won’t see the die texture applied to the cube. That’s because we haven’t yet told Blender to use the die texture in the material assigned to the cube. Let’s do that in the following section.

# Importing and applying a texture

When you've dragged the texture file into **UV Editor**, you have effectively imported it, but, in reality, the material for the cube doesn’t know how to use that texture yet. That being said, the material has all of the information it needs to map 3D vertices to 2D texture coordinates thanks to **UV Editor**. It just needs to be told which texture to apply to the cube.

To accomplish this, we’ll switch to a new workspace so we can connect textures with materials. Also, we’ll import another texture using a different method and assign it to the cube’s material to showcase how you can use the same UV information with different texture files.

Just like when you switched to the **UV Editing** workspace, it’s now time to switch to a different workspace for convenience. The sixth workspace, labeled as **Shading**, is the one you are looking for. We’ll do our work in the lower half of the new workspace, which looks like a grid; it’s called the **Shader Editor**. The upper part is still the same old **3D Viewport**, but **Material Preview** is automatically turned on so you can see your changes reflect immediately. So, the **Shading** workspace should look similar to what you see in *Figure 3.3*.

![Figure 3.3 – Shading is one of many convenient workspaces set up for you ](img/Figure_3.03_B17473.jpg)

Figure 3.3 – Shading is one of many convenient workspaces set up for you

As you discovered in [*Chapter 2*](B17473_02.xhtml#_idTextAnchor032), *Building Materials and Shaders*, Blender files come with a default material. We’ll modify that default material to understand the texture workflow. The **Shader Editor** area is already populated with two entities that make up the material as follows:

*   **Principled BSDF** (**Principled** in short form)
*   **Material Output**

These are called nodes. The node on the left, **Principled**, holds the properties you already saw in the previous chapter. A lot of these properties have little circles on the left side. These circles, which are called sockets, can connect to other nodes’ sockets. We don’t have enough nodes to create meaningful connections yet but we will soon.

Speaking of connectivity, **Principled** has an output that is connected to the **Material Output** node. If you hold your mouse down on the **Surface** input of **Material Output** and drag the connection away, you’ll eventually break the connection between those two nodes. Then, the cube will look black since there is no surface information. Try to reconnect those nodes by dragging the **BSDF** output to the **Surface** input. The default gray color will be reestablished.

Nodes vs code

In the previous chapter, you were told that shaders are lines of code that instruct the GPU what to display. When you use nodes in **Shader Editor**, you are actually writing code, but you are coding visually. As the order of lines is important in conventional programming, the nodes and the connections coming in and out of the nodes are also important. However, visual programming is easier to conceptualize.

When we were modeling the barrel in [*Chapter 1*](B17473_01.xhtml#_idTextAnchor013), *Creating Low-Poly Models*, we needed to add 3D objects to the scene. We did that by pressing *Shift + A*. We’ll do something similar. In this case, we’ll add new nodes to **Shader Editor**. Blender is context-sensitive, which means the same shortcuts will yield similar results if your mouse is over different workspaces, areas, and interfaces. If you press *Shift + A* over **Shader Editor**, you’ll see a list come up and show entities that are relevant to **Shader Editor**.

When this pop-up menu opens, it’s positioned exactly so that the mouse cursor is right over the **Search** button. To add a texture node, perform the following steps:

1.  Click **Search** in the **Add** menu.
2.  Type `Image` with your keyboard.
3.  Select **Image Texture** in the filtered results.
4.  Click anywhere near the other nodes.

This will introduce an **Image Texture** node to **Shader Editor**, just as you see in the following figure:

![Figure 3.4 – An Image Texture node in Shader Editor ](img/Figure_3.04_B17473.jpg)

Figure 3.4 – An Image Texture node in Shader Editor

You have already imported the `pips.png` file when you were working with **UV Editor**, so there is no need to import that file again. We’ll just recall it. As usual, the button to the left of the **New** button in the **Image Texture** node will bring up a list; select **pips.png** from that list. Then, attach the **Color** output of **Image Texture** to the **Base Color** input of **Principled**. This will apply the texture to the cube’s faces. Voilà, the default cube now looks like a six-sided die as seen in *Figure 3.5*:

![Figure 3.5 – The texture file is applied to the model via its material ](img/Figure_3.05_B17473.jpg)

Figure 3.5 – The texture file is applied to the model via its material

A six-sided die has pips, usually marked with a variable number of circles on each side. What if you wanted to have a different looking six-sided die, with the numbers represented by Roman numerals? To import and apply a new texture, perform the following steps:

1.  Create a new **Image Texture** node with the help of *Shift+A*.
2.  Click the **Open** button.
3.  Select `roman.png` in this chapter’s `Start` folder.
4.  Connect this **Image Texture** node's **Color** to the **Principled** node’s **Base Color**.

Since the texture coordinates are already mapped in **UV Editor**, you can easily swap textures that have similar shapes with different designs.

When you work with more complex models, you’ve got more work to do in adjusting the UVs; as long as the UV coordinates are aligned with the right parts of the texture, you’re good. However, imagine a different scenario. How would you go about modeling surfaces that look like they are showing a repeating pattern with slight deviations? In the following section, we’ll look into a different texture workflow.

# Creating textures procedurally

The word “**procedural**” has been used a lot in recent years, especially in the video game industry, to describe different things. Although one might say everything we have done so far is following a certain procedure, the word means something else in our context. When we imported the texture file in the preceding section, it was already designed for us. In other terms, it was a static file. The word “procedural,” on the other hand, is a fancy word that means dynamic.

In a dynamic or procedural texturing workflow, the goal is to expose certain parameters of the texture so that the texture can be changed on the fly, instead of going back to a graphic editing application. Since it’s all dynamic, you won’t have to import graphic files, and you’ll be able to change aspects of the final texture. For example, if the six-sided die was using a procedural texture, it’d be like changing the color and/or the size of the pips.

Procedural textures have another benefit besides their dynamism. Static texture files would need you to do the prior UV work so that the vertices would be aligned with the parts of the texture. In a procedural workflow, the pattern in the texture might be seamless, so you don’t need to worry about the UVs. Seamless, in our context, means that the pattern repeats in a perfect way to wrap around the model.

We’ll create a procedural lava texture as you see in *Figure 3.6* in Blender so you can change its parameters to have a different looking texture.

![Figure 3.6 – Hot lava flowing through solidified crust ](img/Figure_3.06_B17473.jpg)

Figure 3.6 – Hot lava flowing through solidified crust

In a new Blender scene, after deleting the default cube, perform the following steps:

1.  Add a **Plane**.
2.  Switch to the **Shading** workspace.
3.  Bring up the default **Material** or create a new one.
4.  Rename the material if you desire.

Nothing new or exciting so far, but we’ll utilize the following five new nodes very soon:

*   **Noise Texture**: Perlin noise is a mix of black and white values that are mixed together in a gradual way, so the result looks like a soup of grayscale values. Blender’s noise texture is similar to Perlin, but the values are not grayscale; they come with random colors.
*   **Bump**: It is used to simulate height fluctuations so surfaces could look bumpy.
*   **Color Ramp**: Another name for this node would have been color mapper, but since it’s using a gradient, the word “ramp” implies that the transition is smooth.
*   **Emission**: Under normal light, hot objects have a glowing effect. This shader would help you simulate a hot piece of steel coming out of an oven or a bright lightbulb.
*   **Mix Shader**: It’s a shader that mixes two shaders to create a combined shader.

Before we move on to how to mix and match the preceding list of nodes, which kind of look like a recipe’s ingredients, here is a little bit of explanation as to why they were chosen. When you want to create your own procedural textures, a similar process might help you pick the nodes that are helpful instead of making wild guesses about which nodes to select. Also, after the explanation, try to imagine which one will connect to which. So, here we go.

**Noise Texture** is quite literally a texture that comes with some noise; the color variation in this noise texture is used in the **Bump** node to simulate different heights. So, **Noise Texture** is like the data and the **Bump** node is its visual representation in a sense. Then comes **Color Ramp**, shown as **ColorRamp**, which assigns color information to different height values. If you've ever seen a miniature landscape, it’s like painting hilltops white because of snow and the lower areas with different shades of green depending on the elevation.

Hence, the first three nodes are taking care of most of the work for simulating elevation. Let’s assume this lava texture is portraying a recent formation, so we are not after just displaying cooled-down lava. We would like to see steaming hot, glowing lava in between the blackened and dried-out lava. So, we’ll need an **Emission** shader for that. Finally, since the elevation is its own thing and we are adding the emission part, we’ll need **Mix Shader** to combine both.

While working with nodes, you can drag and drop the nodes to arrange a cleaner layout for yourself to make sense of what’s going on. Without further ado, let’s continue.

1.  Add the aforementioned five nodes.
2.  Connect as follows:
    *   **Noise Texture**’s **Color** to **Bump**’s **Height**
    *   **Noise Texture**’s **Fac** to **ColorRamp**’s **Fac**
    *   **Bump**’s **Normal** to **Principled BSDF**’s **Normal**
    *   **ColorRamp**’s **Color** to **Mix Shader**’s **Fac**
    *   **Principled BSDF**’s **BSDF** to **Mix Shader**’s first input **Shader**
    *   **Emission Shader**’s **Emission** to **Mix Shader**’s second input **Shader**
    *   **Mix Shader**’s **Shader** output to **Material Output**’s **Surface**

There is no left or right direction when it comes to connecting nodes. Some people consider a group of nodes as a unit and arrange them close to each other. So, sometimes, the last output node from that group connects almost vertically to another group of nodes. That being said, having a general flow of left to right would fit the preceding instructions. Whichever way you arrange your nodes, the layout might resemble what you see in *Figure 3.7*.

![Figure 3.7 – Lava texture’s node arrangement ](img/Figure_3.07_B17473.jpg)

Figure 3.7 – Lava texture’s node arrangement

Let’s look at the values these nodes will have by following the original order of the node list as much as possible.

## Noise Texture

For **Noise Texture**, the following values were used:

*   **Type** defines the dimensions that are used in the creation of the noise, which involves complex operations. It’s used in more advanced cases, so we’ll leave the default **3D** value.
*   The **Scale** property works more like a zoom factor. Too low, and you are closer to the noisy surface. Too high, and you are seeing a larger portion of the noisy landscape as if you are climbing up in an airplane. In this case, we set **Scale** to **3.0**.
*   The **Detail** property is self-explanatory. Although having a lower value will certainly result in a muddy look, having a higher number beyond a certain value won’t add much to the quality. It will simply increase the calculation time. A value of **8.0** is chosen in our case.
*   **Roughness** is not the same concept you saw in [*Chapter 2*](B17473_02.xhtml#_idTextAnchor032), *Building Materials and Shaders*. That one affected the reflective properties of a surface. This one is about how rough the edges are, in a sense. In other words, how roughly the noise values are blending into each other, and a value of **0.5** is enough.
*   The **Distortion** property creates swirls and wavy patterns. Perhaps a little might be necessary for a flowing lava look. You could experiment with it, but beyond a certain value when there is too much distortion, things will look too fragmented. So, **0.2** is good enough.

## Bump

This node will use the data provided by **Noise Texture** so it can represent different color values as different height values. This is why the **Height** input was connected to the **Color** output since there can’t be just one height value for the whole surface, so we had to feed it a set of colors.

Leaving the **Invert** setting unchecked, the following are the other values used:

*   The **Strength** value determines the effect of the mapping between color values and the final bumps. It works like a percentage since the values can be anywhere between *0.0* and *1.0*. We’ll leave it at **1.0**.
*   The **Distance** property is a multiplier of some sort. It works in conjunction with the **Strength** property. Setting either one of them to *0* will result in a totally flat surface. Perhaps the best way to describe this property is that it keeps the details set in **Noise Texture**. Any value closer to *1.0* will show a washed-out surface, whereas higher values will fill in more details. Thus, a value of **3.0** will yield a detailed enough result.

## Emission

This is a very simple node and it’s responsible for making surfaces look glowing. We’ll discover lights in [*Chapter 4*](B17473_04.xhtml#_idTextAnchor060), *Adjusting Cameras and Lights*, but if you want your objects to act like they are emitting or radiating light, then you can use this node. Examples might be a piece of hot iron or fluorescent lightbulbs; in our case, lava.

Since this is such a simple node, we have only the following two properties:

*   The self-explanatory **Color** property is for picking which color the surface will emit. For hot lava, you can switch to the **Hex** values on the interface and choose **FF8400**.
*   The **Strength** value, which is **100.0** in our case, defines the intensity of the emission. This is a unit measured in Watts so you can be scientific about it, but picking arbitrary values for visual fidelity works most of the time too.

## ColorRamp

The **ColorRamp** node is used for mapping input values to colors with the help of a gradient that works like a threshold. The description is deceptively simple, but there is a lot going on under the hood. So, let’s unpack it.

Most of the time, you’ll be connecting both the input and output sockets of a node to other nodes. However, there are times when it is totally acceptable to use only one type of socket. For example, in the **Emission** shader, you didn’t have to use the input sockets to define the **Color** and **Strength** values. Instead, you handpicked their values. So, the node acts like a source of information.

Then, there are some nodes where it makes much more sense to connect the input socket to another node’s output socket. **ColorRamp** is one of those nodes, and it works like a modifier by factoring in incoming values. **Noise Texture**’s data will be a factor (Fac for short) in creating a lava surface, so we connect the two **Fac** sockets.

Once the data is factored in, we need a system to process it. This is done via the gradient in the **ColorRamp** node. The concept of a gradient might sound weird at first. If you were to connect the **Color** of **Noise Texture** directly to **Material Output**, you’d see that there are smaller and larger zones of colors. If you do that, remember to undo it so that the nodes are connected correctly once again. We need a way to turn these flat but colored zones to elevation.

The gradient is going to help us define which zones are higher or lower so we can assign the appropriate color to different elevations later. In essence, the gradient is a tool to define and blend in those zones with the help of color stops. By default, there are two color stops, but you can use the plus and minus buttons above the gradient to add and remove more color stops. These stops have a square shape with a little triangle right above them. It is possible to drag these stops, which will change the zone transitions we mentioned earlier.

When you have a lot of stops, it’s sometimes difficult to click and drag them, so use **active color stop** to step between them. When you add a fresh **ColorRamp** node, the active stop is marked as **0** and it is to the left of the label that says **Pos**, which indicates the position of the active stop. Both the active stop and the position fields show necessary UI elements for you to change the values once you hover; also, you can click and enter a value. So, by using the active color stop and **Pos**, you can mark exactly where the color stops are going to be if you don’t want to drag them around.

Last but not least, there is a color picker right above the **Fac** socket. You can use that to set the color for the active stop.

Since this is not a straightforward node, we could benefit from some visual aid. *Figure 3.8* is a zoomed-in look at the **ColorRamp** node.

![Figure 3.8 – A close-up look at the ColorRamp node ](img/Figure_3.8_B17473.jpg)

Figure 3.8 – A close-up look at the ColorRamp node

The preceding figure should help you see what we have talked about so far. Also, just like you are able to zoom in and out with your mouse’s scroll functionality in the 3D view, you can do so in **Shader Editor**. It will help you see some of the properties’ names and values more clearly.

Now, it’s time to use all of this information and mark our transitions; you’ll be interacting with all of the elements just presented. To that end, perform the following steps:

1.  Use the plus/minus buttons to have four color stops.
2.  Set `0`, then do as follows:
    1.  Set `0.45`.
    2.  Set color in the `000000`.
3.  Set `1`, then do as follows:
    1.  Set `0.53`.
    2.  Set color in the `FFFFFF`.
4.  Set `2`, then do as follows:
    1.  Set `0.94`.
    2.  Set color in the `FFFFFF`.
5.  Set `3`, then do as follows:
    1.  Set `1.00`.
    2.  Set color in the `636363`.

Notice that we are only picking grayscale values. In a real landscape, higher areas will be cooler lava, and the lower areas will be hot pools of lava. So, to represent that idea, we are picking dark and white colors. Usually, the whiter something is, the hotter it is. The proximity of the stops to each other determines how smooth or sharp the transitions are.

Although we have been working with the **ColorRamp** node, the colors for our lava texture will be defined in the **Principled BSDF** and **Emission** shaders and will be combined in **Mix Shader**. For the time being, we have utilized the data from **Noise Texture** and transformed that data with the help of a gradient and its carefully chosen values. We’ll revisit the factor concept again in the *Mix Shader* section, but before that, let’s visit our trusty friend **Principled BSDF**.

## Principled BSDF

We actually saw this node in [*Chapter 2*](B17473_02.xhtml#_idTextAnchor032), *Building Materials and Shaders*, but it was displayed as part of the **Material Properties** interface. When you create a new material, it uses this shader by default. It combines a great deal of other shaders in its body. For example, it has an emission socket, but since we can’t do both the hot and cool part of the lava formation in one go, we are using a separate **Emission** shader.

We’ll leave most options unchanged, but the following are the non-default values chosen for this exercise:

*   `4A4A4A` as the value in the **Hex** section of the color interface.
*   The `0.2`.
*   `0.2` in this exercise.

You can refer to *Figure 2.5* in [*Chapter 2*](B17473_02.xhtml#_idTextAnchor032), *Building Materials and Shaders*, and read the explanation in the *Discovering Shaders* section for a refresher in understanding how multiple properties work together and affect the final look.

## Mix Shader

It blends one shader into another determined by the value in **Factor**. For the **Factor** socket’s value, if you pick **0.0**, the first shader will be used entirely. If you choose **1.0**, it means that the second shader will be utilized.

The range of decimal values is between 0 and 1 but it’s hard to know what to choose since we can’t just arbitrarily determine how much of which shader to use. This is why we connected the **Color** output from **ColorRamp** as a factor so that the fluctuation in **Noise Texture** would trickle down and affect this node. The effect is cascading. In other words, every single pixel that’s going to be painted either dark (for dried lava) or orange (for hot lava) should be decided based on where **ColorRamp** thinks it belongs in **Noise Texture**. Thus, the color stops act like thresholds and this is all factored in, in **Mix Shader**.

Once all of the nodes have been set and attached, feel free to play with the values in all of them, especially **ColorRamp**. You’ll notice that the hot lava parts are sort of cooler at the shore, and denser and brighter in the middle. Try to approach the color stops close to each other and see how these hot zones in the lava pools change.

Creating this kind of texture using conventional image editing applications such as *Adobe Photoshop* might have been possible, but those applications are layer-based and it’s not always easy to keep things non-destructive. The power you have with a node-based approach is quick iterations. One thing for sure is you don’t have to reimport your texture to see the changes. It’s all happening live in front of your eyes.

However, at the end of the day, since you are developing a game, you’ll have to export your texture so the game engine of your choice can use it. In the following and final section, we’ll see how we can export our lava texture to the file system.

# Exporting your textures

In later chapters, when we get close to working with Godot Engine, we’ll look into asset and project management in more detail. However, after all the hard work we have done with the lava material, it’s now time to learn how to export the texture.

We’ll do a few interesting but necessary things in this section to export our texture. First, we’ll change Blender’s rendering engine. Then, we’ll add an **Image Texture** node in the middle of our material without connecting it to anything. Weird, right? Blender works mysteriously sometimes.

## Changing the rendering engine

We have been using the default **Eevee** rendering engine so far. **Eevee** is a real-time rendering engine that gives you really fast results. Most game engines have their own internal real-time rendering engines that are responsible for calculating lights and shadows. So, **Eevee** is a good way to simulate in Blender what you’ll most likely experience when you export your assets to a game engine. However, the speed and convenience come with a few penalties.

Blender has another engine that is called **Cycles**. **Cycles** is a very accurate but slow rendering engine compared to **Eevee**. **Cycles**’ accuracy is due to the fact that it tackles advanced lighting calculations, which leads to quality results such as showing reflective and transparent surfaces much better, displaying more accurate shadows, and even creating volumetric effects such as haze and fog. The following is a link to an article that demonstrates both engines’ capabilities and differences with use cases: [https://cgcookie.com/articles/blender-cycles-vs-eevee-15-limitations-of-real-time-rendering](https://cgcookie.com/articles/blender-cycles-vs-eevee-15-limitations-of-real-time-rendering).

In this book, we are not covering advanced enough topics that would require us to make a hard decision between **Eevee** and **Cycles**. So, **Eevee** has been fine for our purposes. However, when you work with procedural textures, there is no way, at least with the version of Blender we’re using, for **Eevee** to export the lava texture. We’ll have to switch to the **Cycles** engine. Luckily, it’s done just with the click of a button.

In the **Properties** panel on the right, the second tab from the top, which looks like the preview display of a DSLR camera, is going to open **Render Properties**. The drop-down list at the top will show **Eevee**; let’s change that to **Cycles**. Also, if you have a decent graphics card, you might want to change the third dropdown, **Device**, value to **GPU compute** so that your graphic card can do the heavy lifting instead of your good old CPU.

Looking down in that long list of properties, you’ll see a panel with the header **Bake**. If you expand the header, you’ll see a **Bake** button. We’ll click that button soon, but we need to prepare what we’ll bake first.

## Baking a texture File

When we worked with the cube and die textures, we used an **Image Texture** node to bind an existing image from the file system. Our situation is different when the texture is procedural since this has been happening live in the memory. We need to figure out a way to bake this information into a file. Since there is no such file, we need to pretend that we have one, as follows:

1.  Add an **Image Texture** node.
2.  Click the **New** button.
3.  Type `lava` in the name section.
4.  Click the **OK** button.

We won’t be connecting `lava` will be packaged with the material. Blender will make an educated guess and will bake the procedural texture parts into this image.

Now is the time to hit that **Bake** button in **Render Properties**. A progress bar at the bottom will indicate that Blender is doing its thing. Once the process is finished, the bottom-left corner of the **Shading** workspace will fill with the lava texture. That little section that displays the baked texture is called **Image Editor**.

If you look at the baked image, you’ll notice that some details are lost. The pool of hot lava has warmer and cooler spots in `1.0`.

In the `lava.png` in your file system. This file can now be imported into a new Blender file and used in an `pips.png` to a cube.

Mission accomplished. If you chose the same values as those written in this chapter, you should have the procedural lava texture you see in *Figure 3.6*. Additionally, you have created a static version of it. Let’s summarize what else has been accomplished in this chapter.

# Summary

This chapter started off with a brief discussion about what textures are and why they might be needed. To recap, if you are fine with models that have just the color info on their surface, you are done as soon as the modeling and material application process is finished. If you think you need to show distinctive qualities on your models’ surfaces, you need to utilize textures.

To that end, you discovered how a new coordinate system—one that involves mapping spatial coordinates to texture coordinates via a method called UV unwrapping—might be necessary. Once the UV unwrapping is done, you can apply and swap different textures to your 3D models since the mapping from 2D to 3D is established.

Although creating textures with image editing applications is quite possible, you also know how to create textures procedurally in Blender. This is a powerful method, especially when it comes to surfaces that are hard to UV unwrap, such as landscapes.

Last but not least, you learned how to change the rendering engine to be able to export your procedural texture to your file system. Although this file is static and can no longer be updated automatically (unless you overwrite it with a new export, of course), you have the benefit of sharing the file easily.

You’ve been using Blender’s interface and your mouse to move around the scene and rotate the view to have a better look at your models, materials, and so on. In the following chapter, you’ll learn how to work with Camera and Light objects to create a composition where you can arrange objects in your scene under the best light conditions possible.

# Further reading

To read more about what each shader node does, you can refer to the official documentation at the following link: [https://docs.blender.org/manual/en/2.93/render/shader_nodes/](https://docs.blender.org/manual/en/2.93/render/shader_nodes/).

For further practice, imagine where else the method for the lava texture could be used. Perhaps, with carefully planned values and more color variations, the hot lava might be rust, and the cool lava might be paint?

If you are curious and would like to investigate different software out there capable of producing procedural textures, you can give *Adobe Substance Designer* a try. It’s a powerful program dedicated solely to creating textures. Not all of the nodes are labeled the same, but there are a lot of similar nodes to Blender’s. In fact, if you practice your skills there and look at other people’s creations, you might gain insight into creating such textures in Blender.