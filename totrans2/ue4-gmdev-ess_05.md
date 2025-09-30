# Chapter 5. Lights

Lighting is an important factor in your game, which can be easily overlooked, and wrong usage can severely impact on performance. But with proper settings, combined with post process, you can create very beautiful and realistic scenes.

In this chapter, we will look into different light mobilities and learn more about Lightmass Global Illumination, which is the static Global Illumination solver created by Epic games. We will also learn how to prepare assets to be used with it.

# Lighting basics

In this section, we will see how to place lights and how to adjust some important values.

## Placing lights

In Unreal Engine 4, lights can be placed in two different ways. Through the modes tab or by right-clicking in the level:

*   **Modes tab**: In the **Modes** tab, go to the place tab (*Shift* + *1*) and go to the **Lights** section. From there you can drag and drop various lights.![Placing lights](img/B03950_05_01.jpg)
*   **Right-clicking**: Right-click in viewport and in **Place Actor** you can select your light.![Placing lights](img/B03950_05_02.jpg)

Once a light is added to the level, you can use the transform tool (*W* to move, *E* to rotate) to change the position and rotation of your selected light.

### Tip

Since **Directional Light** casts light from an infinite source, updating their location has no effect.

## Various lights

Unreal Engine 4 features four different types of light Actors. They are:

*   **Directional Light**: Simulates light from a source that is infinitely far away. Since all shadows cast by this light will be parallel, this is the ideal choice for simulating sunlight.
*   **Spot Light**: Emits light from a single point in a cone shape. There are two cones (inner cone and outer cone). Within the inner cone, light achieves full brightness and between the inner and outer cone a falloff takes place, which softens the illumination. That means after the inner cone, light slowly loses its illumination as it goes to the outer cone.
*   **Point Light**: Emits light from a single point to all directions, much like a real-world light bulb.
*   **Sky Light**: Does not really emit light, but instead captures the distant parts of your scene (for example, Actors that are placed beyond **Sky Distance Threshold**) and applies them as light. That means you can have lights coming from your atmosphere, distant mountains, and so on. Note that **Sky Light** will only update when you rebuild your lighting or by pressing **Recapture Scene** (in the **Details** panel with **Sky Light** selected).

## Common light settings

Now that we know how to place lights into a scene, let's take a look at some of the common settings of a light. Select your light in a scene and in the **Details** panel you will see these settings:

*   **Intensity**: Determines the intensity (energy) of the light. This is in lumens so, for example, 1700 lm corresponds to a 100 W bulb.
*   **Light Color**: Determines the color of the light.
*   **Attenuation Radius**: Sets the limit of the light. It also calculates the falloff of the light. This setting is only available in **Point Lights** and **Spot Lights**.![Common light settings](img/B03950_05_03.jpg)

    Attenuation Radius from left to right: 100, 200, 500.

*   **Source Radius**: Defines the size of specular highlights on surfaces. This effect can be subdued by adjusting the **Min Roughness** setting. This also affects building light using **Lightmass**. A larger **Source Radius** will cast softer shadows. Since this is processed by **Lightmass**, it will only work on **Lights** with mobility set to **Static**.![Common light settings](img/B03950_05_04.jpg)

    Source Radius 0\. Notice the sharp edges of the shadow.

    ![Common light settings](img/B03950_05_05.jpg)

    Source Radius 5\. Notice the soft edges of the shadow.

*   **Source Length**: Same as **Source Radius**.

## Light mobility

Light mobility is an important setting to keep in mind when placing lights in your level because this changes the way light works and impacts on performance. There are three settings that you can choose. They are as follows:

*   **Static**: A completely static light that has no impact on performance. This type of light will not cast shadows or specular on dynamic objects (for example, characters, movable objects, and so on). Example usage: Use this light where the player will never reach, such as distant cityscapes, ceilings, and so on. You can literally have millions of lights with static mobility.
*   **Stationary**: This is a mix of static and dynamic lights and can change its color and brightness while running the game, but cannot move or rotate. Stationary lights can interact with dynamic objects and are used where the player can go.
*   **Movable**: This is a completely dynamic light and all properties can be changed at runtime. Movable lights are heavier on performance so use them sparingly.

Only four or fewer stationary lights are allowed to overlap each other. If you have more than four stationary lights overlapping each other, the light icon will change to red X, which indicates that the light is using dynamic shadows at a severe performance cost!

![Light mobility](img/B03950_05_06.jpg)

In the following screenshot, you can easily see the overlapping light.

![Light mobility](img/B03950_05_07_revised.jpg)

In **View** Mode, you can change to **Stationary Light Overlap** to see which light is causing an issue.

# Lightmass Global Illumination

Lightmass is the high-quality static Global Illumination solver created by Epic games. **Global Illumination** (**GI**) means the process that simulates indirect lighting (for example, light bouncing and color bleeding from surfaces). In Unreal Engine, light bounces by default with Lightmass and is based on the base color of your material, which controls how much light should bounce from the surface of the object. Even though a more highly saturated color will bounce more light, and a less saturated color will bounce less, it all depends on the scene. In a simple room-like scene, this can be noticeable, whereas if it's an outdoor day scene this might not be that noticeable.

Let's take a quick look at the following scene:

![Lightmass Global Illumination](img/B03950_05_08.jpg)

This is a simple scene in unlit mode.

![Lightmass Global Illumination](img/B03950_05_09.jpg)

Now I added one **Directional Light** and this is how it looks with no GI. That means there is only direct lighting and no indirect lighting (meaning there is no light bouncing).

![Lightmass Global Illumination](img/B03950_05_10.jpg)

The previous screenshot is with static GI and you can see how the whole scene came to life with GI. Notice how the pillars are casting shadows. These are called **Indirect Shadows** since they are from **Indirect Light**.

The intensity and color of indirect light depends on the light and base color of the material that the light is bouncing off. To illustrate this effect, let's take a look at the following two screenshots:

![Lightmass Global Illumination](img/B03950_05_11.jpg)

Here I applied a pure red material (the red value is 1.0) to the sphere and you can see bounced lighting picked up the base color of the red sphere changing the environment. This is called color bleeding and is most noticeable with highly saturated colors.

![Lightmass Global Illumination](img/B03950_05_12.jpg)

In this screenshot, I changed the value of red to 0.1 and rebuilt the lighting. And since red is more dark now, less light is bouncing. This is because darker colors will absorb the light instead of reflecting it.

Now that we know what Lightmass is, let's take a look at how we can prepare our assets to use Lightmass and learn more about Lightmass settings.

## Preparing your assets for precomputed lighting

In order for your asset to have clean light and shadow details, it is necessary to have uniquely unwrapped UV to represent its own space to receive dark and light information. One rule of thumb when creating lightmap UVs is that the UV face should never overlap with any other face within the UV space. This is because if they are overlapping, then after light building, the lightmap corresponding to that space will be applied to both faces, which will result in inaccurate lighting and shadow errors. Overlapping faces are good for normal texture UVs since the texture resolution will be higher for each face, but the same rule does not apply for lightmap UVs. In a 3D program, we unwrap lightmap UVs to a new channel and use that channel in Unreal Engine 4.

![Preparing your assets for precomputed lighting](img/B03950_05_13.jpg)

Here, you can see I'm using the second channel in my mesh for lightmap.

### Note

Unreal starts counting from 0 while most 3d programs count from 1\. That means UV channel 1 in the 3d program is UV channel 0 in Unreal, and UV channel 2 means UV channel 1 in Unreal. Here, in the previous screenshot, you can see the **Light Map Coordinate Index** is **1**, which means it is using the 2nd UV channel in mesh.

Even though you can generate lightmap UVs in Unreal Engine 4, it is highly recommended to create these UVs in a 3d program (for example, Autodesk Maya, Autodesk 3dsmax, Modo, and so on) in order to have clean lightmaps. Before creating a lightmap UV you have to set up the grid setting in your 3d app's UV editor. For example, if you have an asset that requires a lightmap resolution of 128, then your grid setting should be *1/126*, which is *0.00793650*. 128 will be the lightmap texture resolution. Higher values, such as 256, 512, 1024, and so on, will result in high-quality lightmaps but will also increase memory usage. Once we decide what lightmap resolution we need for our asset, we subtract 2 (you can also use 4) from that resolution. The reason behind this is that in order for Lightmass to calculate correctly without any filter bleeding errors, it is recommended to have a minimum of 2 pixel gaps between UVs. So if your asset is going to use a lightmap resolution of 128, it will be *128 – 2 = 126*. The reason why we divide it by 1 is that by default, Lightmass uses a 1 pixel border for filtering purposes.

Once you import your mesh into Unreal Engine 4, you set the Light Map Resolution for your Static Mesh. This value controls how good the shadow will look when another object casts a shadow onto this object.

Lightmaps are textures generated by Unreal Engine and overlayed on top of your scene. Since this is a texture, it should be in power of two (for example, 16, 32, 64, 128, 256, 512, 1024, etc.).

![Preparing your assets for precomputed lighting](img/B03950_05_14.jpg)

The floor in the preceding screenshot has a lightmap resolution of **32**. Notice inaccurate shadows on the floor.

![Preparing your assets for precomputed lighting](img/B03950_05_15.jpg)

The floor in the preceding screenshot has a lightmap resolution of **256**. Notice better shadows on the floor.

### Note

Even though increasing the lightmap resolution gives accurate shadows, it is not a good idea to increase it for every mesh in your level as it will severely increase build times and may even crash the whole editor. For smaller objects, it is always a good idea to keep it low.

In Unreal Engine 4, you can generate lightmap UVs when importing your mesh by enabling **Generate Lightmap UVs**.

![Preparing your assets for precomputed lighting](img/B03950_05_16.jpg)

In case you miss this option, you can still generate lightmap UVs after importing. To do that perform the following steps:

1.  Double-click on the **Static Mesh** in **Content Browser**.
2.  Then, under the **LOD** tab, enable **Generate Lightmap UVs**.
3.  Select **Source Lightmap Index**. Most of the time this will be **0** since that is normal texture UVs, and Unreal generates your lightmap UVs from texture UVs.
4.  Set **Destination Lightmap Index**. This is where Unreal will save your newly created lightmap UVs. Set this to 1.
5.  Click **Apply Changes** to generate lightmap UVs.![Preparing your assets for precomputed lighting](img/B03950_05_17.jpg)

    ### Note

    If you already have a lightmap UV in the **Destination Lightmap Index**, it will be replaced when generating a new one.

You can preview UVs by clicking on the **UV** button in the toolbar and selecting your **UV Channel**.

![Preparing your assets for precomputed lighting](img/B03950_05_18.jpg)

## Building a scene with Lightmass

Building a scene with Lightmass is a pretty straightforward process. In order to have high-quality static Global Illumination (aka **Precomputed Lighting**), you need to have a **Lightmass Importance Volume** in your scene. This is because in many maps, we have areas large enough and the playable area is actually smaller. So instead of calculating lighting for the whole scene, which can increase light building heavily, we limit the area by using **Lightmass Importance Volume**.

Once we have a **Lightmass Importance Volume** in the scene and start light building, Lightmass will only calculate lighting within the volume. All objects outside the volume will only get one bounce of light with low quality.

To enclose the playable area in **Lightmass Importance Volume** you just have to drag and drop it from the **Modes** tab. Just like other objects, you can use transform tools (*W* to move, *E* to rotate, and *R* to scale) to adjust **Lightmass Importance Volume** in your scene. Once that is done, all you have to do is build the lighting from the **Build** button.

![Building a scene with Lightmass](img/B03950_05_19.jpg)

Alternatively, you can simply press the **Build** button, which will build the lighting. Lightmass has four different quality levels that you can choose from. They are **Preview**, **Medium**, **High**, and **Production**.

![Building a scene with Lightmass](img/B03950_05_20.jpg)

*   **Preview**: Can be used while developing and this results in building the light faster.
*   **Production**: When your project is near-complete or ready to ship you should use the production setting since it makes the scene more realistic and corrects various light bleed errors.

### Tip

Lighting quality are just presets. There are lots of settings that should be tweaked to get the desired effect you want in your game.

## Tweaking Lightmass settings

Lightmass offers a lot of options in **World Settings**, which you can tweak to get the best visual quality. You can access them by clicking on **Settings** and selecting **World Settings**.

![Tweaking Lightmass settings](img/B03950_05_21.jpg)

In **World Settings**, expand **Lightmass Settings** and you will see various settings you can tweak to get the most out of **Lightmass**.

![Tweaking Lightmass settings](img/B03950_05_22.jpg)

Controlling these settings helps you get the best visual quality when using Lightmass. Let's take a look at these settings:

*   **Static Lighting Level**: This setting calculates the detail when building the light. Smaller values will have more detail but greatly increase build time! Larger values can be used for huge levels to lower build times.
*   **Num Indirect Lighting Bounces**: This determines how many times the light should bounce off surfaces. 0 is direct lighting only, meaning there will be no Global Illumination, and 1 is one bounce of indirect lighting, and so on. Bounce 1 contributes most to the visual quality, and successive bounces are nearly free but do not add very much light since bounced light gets weaker after each bounce.![Tweaking Lightmass settings](img/B03950_05_23.jpg)

    Num Indirect Lighting Bounces set to 1

*   **Indirect Lighting Quality**: Higher settings result in fewer artifacts such as noise, splotchiness, and so on, but will also increase build time. Using this setting with **Indirect Lighting Smoothness** helps to get detailed indirect shadows and ambient occlusion.
*   **Indirect Lighting Smoothness**: Higher values will cause Lightmass to smooth out indirect lighting but will lose detailed indirect shadows.![Tweaking Lightmass settings](img/B03950_05_24.jpg)

    Indirect Lighting Quality and Smoothness set to 1.0

    ![Tweaking Lightmass settings](img/B03950_05_25.jpg)

    Indirect Lighting Quality: 4.0 and Indirect Lighting Smoothness: 0.5\. Notice the difference in the shadow cast by the pillar

*   **Environment Color**: Think of this as a big sphere surrounding the level, emitting color in all direction. That is, this acts as the HDR environment.
*   **Environment Intensity**: Scales the intensity of **Environment Color**.
*   **Diffuse Boost**: This is an effective way of increasing the intensity of indirect lighting in your scene. Since indirect lighting bounces off surfaces, this value will boost the influence of the color.
*   **Use Ambient Occlusion**: Enables static ambient occlusion. Since ambient occlusion requires dense lighting samples, it will not look good in **Preview** builds. It's better to tweak ambient occlusion settings while you are building using production preset.
*   **Direct Illumination Occlusion Fraction**: How much ambient occlusion to be applied to direct lighting.
*   **Indirect Illumination Occlusion Fraction**: How much ambient occlusion to be applied to indirect lighting.
*   **Occlusion Exponent**: Higher values increase the ambient occlusion contrast.
*   **Fully Occluded Samples Fraction**: This value determines how much Ambient Occlusion an object should generate on other objects.
*   **Max Occlusion Distance**: Maximum distance for an object to cause occlusion on another object.
*   **Visualize Material Diffuse**: Overrides normal direct lighting and indirect lighting with the material diffuse term exported to Lightmass.![Tweaking Lightmass settings](img/B03950_05_26.jpg)

    Visualize Material Diffuse enabled

*   **Visualize Ambient Occlusion**: Overrides normal direct lighting and indirect lighting with ambient occlusion. This is useful when you are tweaking **Ambient Occlusion** settings.![Tweaking Lightmass settings](img/B03950_05_27.jpg)

    Visualize Ambient Occlusion enabled

*   **Volume Light Sample Placement Scale**: Scales the distance at which volume lighting samples are placed.

### Note

All these Lightmass settings require lighting rebuild. So if you change any of these settings, make sure you rebuild the lighting for the changes to take effect.

**Volume Light Samples** are placed by Lightmass in the level after light building, and are used for dynamic objects such as characters, since Lightmass only generates lightmaps for static objects. This is also called **Indirect Lighting Cache**.

In the following screenshots, you can see how the movable object (red sphere) is lit using **Indirect Lighting Cache**:

![Tweaking Lightmass settings](img/B03950_05_28.jpg)

With Indirect Lighting Cache

![Tweaking Lightmass settings](img/B03950_05_29.jpg)

Without Indirect Lighting Cache

### Note

**Volume Light Samples** are only placed within **Lightmass Importance Volume** and on static surfaces.

**Indirect Lighting Cache** also helps with previewing objects with unbuilt lighting. After light building, if you move a static object, it will automatically use **Indirect Lighting Cache** until the next light build.

To visualize volume lighting samples, click on **Show** | **Visualize** | **Volume Lighting Samples**.

![Tweaking Lightmass settings](img/B03950_05_30.jpg)

Volume Lighting Samples previewed in the level.

### Note

You can adjust **Global Illumination Intensity** and **Color** in **Post Process Volume**. In **Post Process Volume**, expand **Post Process Settings** | **Global Illumination** and there you see settings for **Color** and **Intensity**.

![Tweaking Lightmass settings](img/B03950_05_31.jpg)

To toggle specific lighting components for debugging, you can use the various lighting component flags under the **Show** | **Lighting Components** section. For example, if you want to preview your scene without any direct lighting, you can turn off **Direct Lighting** and preview your scene in **Indirect Lighting** only. Keep in mind that these are only editor features and do not affect your game. These are only for debugging purposes.

![Tweaking Lightmass settings](img/B03950_05_32.jpg)

# Summary

In this chapter, we learned about lights and how they can improve the realism of your scene by using Lightmass Global Illumination, and how to prepare our assets to use with Lightmass. We also learned about various lights and common settings. In the next chapter we will dive into the best and most unique feature of Unreal Engine 4: Blueprints.