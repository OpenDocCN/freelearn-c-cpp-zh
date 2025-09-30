# Chapter 6. A Particle System and Sound

In this chapter, we will touch on the components of a game that are extremely important but often go unnoticed unless they are badly designed and out of place. Yes, we will cover particle system and sound in this chapter. In most games, they blend in so naturally that they are easily forgotten. They can also be used to create the most memorable moments in a game.

Just to recap, particle systems are used very often to create sparks, explosions, smoke, rain, snow, and other similar effects in a game that are dynamic, kind of fuzzy, and random in nature. Sound can be in the form of ambient sounds, such as the sound of rustling leaves and wind, one-off sounds, such as a pan dropping in the kitchen, or repetitive sounds, such as the running steps of a character or even music playing on the radio. Sound can be used to set the mood of a game, alert the player to something that needs attention, and provide realism to a level to make a place come alive. Let's get started.

# What is a particle system?

A particle system is a way to model fuzzy objects, such as rain, fire, clouds, smoke, and water, which do not have smooth, well-defined surfaces and are nonrigid. The system is an optimized method to achieve such fluid-looking and dynamic visual representations by controlling the movement, behavior, interaction, and look of many tiny geometry objects or sprites.

Using a combination of different particles made of different shapes, sizes, materials, and textures, with different movement speeds, rotation direction/speeds, spawn rates, concentration, visibility duration, and many more factors, we are able to create a huge variety of dynamic complex systems.

In this chapter, we will learn about the components of the particle system using Unreal's Particle System editor and Cascade editor and use these editors to create a few additions for your level.

# Exploring an existing particle system

We will start by first seeing what kind of particle systems we get in the default package of Unreal Engine 4\. Go to **Content Browser** | **Game** | **Particles**. There are a couple of particle systems that we can already drag and place in the level and check out how they look.

To open a particle system, simply double-click on any of the systems. Let's take a look at **P_Fire** together. Feel free to check out the rest of the systems as well. However, I will use this as an example to understand what we need in order to create a new particle system for our level. This screenshot shows **P_Fire** in the editor:

![Exploring an existing particle system](img/B03679_06_01.jpg)

On the left-hand side is **Viewport** where we can preview the particle system. On the right-hand side, in the **Emitters** tab, you can see several columns of boxes with **Flames** (twice), **Smoke**, **Embers**, and **Sparks** mentioned on top of each of the columns.

Emitters can be thought of as separate components that make up the particle system, and you can give each emitter different properties depending on what you want to create. When you put a bunch of emitters together, you will see them combining to give you a whole visual effect. In this **P_Fire** particle system, you can see flames moving in an unpredictable manner with some sparks and embers floating around and smoke simulating a fire bursting into flames. In the next section, let's go through more concrete terminology that describes the particle system in Unreal Engine 4.

# The main components of a particle system

Very briefly, the following paragraph (taken from the official Unreal 4 documentation that's available online) very aptly describes the relationship between the different components that are used in particle systems:

> *"Modules, which define particle behavior and are placed within...Emitters, which are used to emit a specific type of particle for an effect, and any number of which can be placed within a...Particle System, which is an asset available in the Content Browser, and which can then in turn be referenced by an...Emitter Actor, which is a placeable object that exists within your level, controlling where and how the particles are used in your scene.",*

Read this several times to make sure that you are clear on the relationship between the different components.

So, as described in the earlier section where we looked at **P_Fire**, we know that the emitters are labelled as **Flames**, **Embers**, **Sparks**, **Smoke**, and so on. The different properties of each of the emitters are defined by adding modules, such as **Lifetime**, **Initial Velocity**, and so on, into them. Together, all the emitters make up a particle system. Lastly, when you place the emitters in your game level, you are, in fact, dragging the emitter actor, which references a particular particle system.

## Modules

The **Default Required** and **Spawn** modules are the modules that every emitter needs to have. There is also a long list of other optional modules that the Cascade Particle editor offers to customize your particle system. In the current version of the editor that I am using, I have the **Acceleration**, **Attractor**, **Beam**, **Camera**, **Collision**, **Color**, **Event**, **Kill**, **Lifetime**, **Location**, **Orbit**, **Orientation**, **Parameter**, **Rotation**, **Rotation Rate**, **Size**, **Spawn**, **SubUV**, **Vector Field**, and **Velocity** modules.

We will cover a few of the frequently used modules from this long list of modules through a simple exercise that's based on **P_Fire**. I understand that it would be very boring and not very useful when grasping the basics here if I simply gave you all those definitions that you can find easily online. Instead, we will go through this section by customizing **P_Fire** to create a fireplace for our level. At the same time, we will go through the key values within the different modules that you can adjust. Thus, you can take a look at how these values impact the look of the particle system.

For more detailed documentation on the definition of each module and parameter, you can refer to the Unreal 4 online documentation ([https://docs.unrealengine.com/latest/INT/Engine/Rendering/ParticleSystems/Reference/index.html](https://docs.unrealengine.com/latest/INT/Engine/Rendering/ParticleSystems/Reference/index.html)).

The commonly used modules are listed as follows:

| Module | Key parameters it can control |
| --- | --- |
| **Required** | Material used for the particles |
| **Spawn** | Rate and distribution of the spawn |
| **Initial Size** | Size of the initial particle |
| **Lifetime** | Time duration for which the particle stays visible |
| **Color Over Life** | Color of the particles over their lifetimes |

# The design principles of a particle system

The design principles of a particle system can be configured through a research and iterative creative process. Let's take a look at each one of them in the following section.

## Research

Details are probably key to designing a realistic particle system. Very often, creating a particle system lies in the realm of an artist as we need an artistic touch to create a visually appealing and somewhat realistic replica of the effect that we want to create.

For starters, it is good to research a little on what the actual effect looks like. Here are some steps to help you get started:

*   Identify the different components that are needed (break the particle effects down into the different components).
*   Determine the relationship among the different components (the size of the particles that are relative to one another, spawn rate, lifetimes, and so on).
*   Next, look at other similar effects that are created in the **Computer Graphics** (**CG**) space. The reason for doing this is that sometimes, actual effects can be a little too monotonous, and there are many amazing visual effect people out there who you can learn from to spice things up a little. So, it is a great idea to spend a little time checking out what others have done already, rather than spending a whole lot of time experimenting and not getting what you want to achieve.

## The iterative creative process

Creating the perfect looking particle system that you want usually involves quite a bit of tweaking and playing around with the parameters that you have. The key to doing this is knowing what parameters there are and what they affect. During the initial phase of design, you should also try adding or removing certain modules to see how they actually impact the overall look of the system. This does not mean that more is always better. Additionally, it is also wise to save backup copies of your iterations so that you can always go back to the previous versions easily.

Being extremely proficient in creating the particle system, I think, involves a combination of good design planning, having the patience to iterate, and making small adjustments to get the look that you eventually want.

# Example – creating a fireplace particle system

In this example, we will duplicate **P_Fire** and edit it to create a fire for a fireplace in the level. We will also change a part of the current level in which we have to place this new fireplace particle system.

Go to **Content Browser** | **Particles**, select **P_Fire**, and duplicate it. Rename it `P_Fireplace`. This screenshot shows how **P_Fireplace** is created in the `Particles` folder:

![Example – creating a fireplace particle system](img/B03679_06_02.jpg)

Let's open **Chapter5Level** and rename it `Chapter6Level` first. We will first add a fireplace structure to the level to set the context for this fireplace effect. This will help you follow the creation process better. This screenshot shows the original living room space:

![Example – creating a fireplace particle system](img/B03679_06_03.jpg)

The following screenshot shows the modified living room space with a fireplace:

![Example – creating a fireplace particle system](img/B03679_06_04.jpg)

This screenshot shows a zoomed in version of the fireplace structure if you intend to construct it:

![Example – creating a fireplace particle system](img/B03679_06_05.jpg)

Zooming in on the metal vents will look like this:

![Example – creating a fireplace particle system](img/B03679_06_06.jpg)

What we did here was delete the lights and low cabinet structure and replaced it with this:

*   **TopWoodPanel** (material: **M_Wood_Walnut**): X = 120, Y = 550, Z = 60
*   Concrete pillars around the glass (material: **M_Brick_Cut_Stone**)
*   **ConcretePillar_L** and **ConcretePillar_R**: X = 100, Y = 150, Z = 220
*   **ConcretePillar_Top**: X = 100, Y = 250, Z = 100
*   Fireplace glass and inside (material: **M_Glass**)
*   **Fireglass**: X = 5, Y = 250, Z = 120
*   **MetalPanel** and **MetalPanel_Subtractive**: X = 40, Y = 160, Z = 10
*   **FireVent1** to **FireVent5**

Use the BSP subtractive cylinder with the following setting, as shown in the following screenshot. Here, **Z** is **10**, **Outer Radius** is **3**, and **Sides** is **8**:

![Example – creating a fireplace particle system](img/B03679_06_07.jpg)

The lower extended structure (made up of two BSPs) consists of the following:

*   **Thinner extension platform**: X = 140, Y = 550, Z = 10
*   **Thicker base**: X = 120, Y = 550, Z = 30

## Crafting P_Fireplace

Now, double-click on **P_Fireplace** to open up the Cascade Particle System editor. Since we duplicated it from **P_Fire**, it has the same emitters as **P_Fire**: the two **Flame**, one **Smoke**, one **Sparks**, one **Embers**, and one **Distortion** module.

Observe the current viewport. What do you see? The original **P_Fire** effect is more like a sequence of random bursts of flames that disappear pretty quickly after the initial burst. What kind of fire do we need for the fireplace that we have created? We need more or less continuous and slower moving flames that hover in a fixed position.

With this difference and objective in mind, we will next determine which of the components of **P_Fire** we want to keep as our fire effect for the fireplace.

### Observing the solo emitters of the system

Using the solo button and checkbox in each of the modules, toggle **S** on or off, and alternatively mark/unmark the checkbox to observe the individual components. This screenshot shows you the location of the solo button and checkbox:

![Observing the solo emitters of the system](img/B03679_06_08.jpg)

### Deleting non-essential emitters

The first step was to delete the second **Flame** emitter (the first being the left-most) and the **Smoke** emitter. The reason for this was, I think, so that I could work with a single flame to create a fire for the fireplace. The **Smoke** emitter was removed mainly because of it is a gas/electric fire; thus, I would expect less smoke. You could alternatively unmark the checkbox at the top of the window to hide the entire emitter first before deleting it permanently.

### Focusing on editing the Flame emitter

Keeping the only **Flame** emitter, the flame was still appearing at random spots within a certain radius and then disappearing quickly after that. We will address each of the issues here one by one:

*   **Configure Lifetime**: So, since we need to have the fire burning continuously instead of in short bursts, I will first adjust the **Lifetime** property so that the fire burns for a longer period of time before disappearing. Change **Distribution Float Uniform**, with **Min** kept as **0.7**, **Max** as **1.0**, and **Distribution Constant** as **1.2**.
*   **Remove Const Accleration+**: Now, the flame lingers longer on screen before disappearing. However, the flames seem to be drifting away from the spawn location after they are spawned. For a fireplace, flames more or less remain in the same location. So, I turn off **Const Acceleration+** in the **Flames** module by unmarking the checkbox. The flames now seem to be moving away from the spawn location a lot less.
*   **Remove Initial Velocity**: After removing the acceleration module, it still seems like the flames are moving away; my guess for this is that the particles had some initial velocity, and so I turned off this module to confirm my suspicion and it seemed to work.
*   **Configure Spawn**: The flames looked quite sparse as they are small, and this creates some blank space within the spawn area during short intervals. I could adjust the size of the flame to make it bigger, but when I did this, the flame looked too distorted. So, I decided to increase the spawn rate instead so that more flames could occur per minute. Change the spawn rate for **Rate Scale Distribution** from **5.0** to **20.0**. Increase **Distribution Float Constant** from **1.0** to **3.0**.

### Looking at the complete particle system

Now, I've turned the other emitters back on again to look at the whole particle system effect and also see if it requires more editing. It looks pretty okay for a fireplace fire now so I've stopped here. Feel free to go ahead and adjust the other properties to improve the design. These are the very basics of modifying an existing particle system, and I hope you have familiarized yourself with the particle system editor through this exercise.

# Sound and music

Sound and music are an essential part of the game experience. Ever watched television with the volume switched off? Just watching subtitles and lip movements is not enough. You want to hear what the character on the screen is saying and how they are saying it. For games, it is pretty much similar, and on top of this, pretty often, you get cues through the sound and music. If you have played *Alien: Isolation*, you need to listen to the sounds in the game to know whether you have an alien coming in your direction. This can be a matter of life and death in the game. It pretty much determines whether you end up as a winner or simply a delicious meal for the alien. So, are we ready now to learn how sound and music are created for games, and how we use the Unreal Editor to incorporate them into our game level?

# How do we produce sound and music for games?

Many game productions have original music written for in-game scenarios; some also use actual songs sung by professional singers as theme songs. Music in games is a big thing and it's dearly remembered by fans of the game. Sometimes, the music itself is enough to trigger memories of the gaming experience. Thus, game studios need to spend time creating suitable music to complement their games.

If you are a huge fan of video game music, there are also concerts that you can go to where the orchestra plays music from popular games (check out Video Games Live at [http://www.videogameslive.com/index.php?s=home](http://www.videogameslive.com/index.php?s=home)).

Creating music for a game is very similar to composing music for a piece; it should trigger appropriate emotions when it's played. The choice of music needs to match the pace and situations of the game. Using a JRPG game as an example, you should be able to differentiate between in-battle music versus the music that's played when you are in a menu, loading the game, or when you've just won a battle. Very often, music is created on the basis of the needs of the game, and the music composer has to probably come up with a few different versions and let the team and/or management review it before the best piece is selected.

If you do not intend to create original music or sound for your game, you can find many free downloadable sounds and music online these days. When using free online music and sounds, do ensure that you do not violate any digital rights or copyrights when incorporating them in your game.

# Audio quality

The reason why we are discussing audio quality is because sound quality, like image quality, is of huge importance these days. We already use the 4K resolution image quality today, and there will be more devices and games that would support this in the future. How about sounds? The listening experience needs to match the quality of the image and provide more than just mono or stereo sounds. Sound experience has also progressed to multichannel surround sound, starting at 5.1, 7.1, and beyond these days, to obtain a life-like immersive audio experience. This is definitely something to think about when creating, storing, and playing audio files.

# How are sounds recorded?

Sounds are generated in the form of analog waves, which are continuous waves, which you'll see shortly in the upcoming figure. We can record surround sound through a recording device. For multichannel sound recording, you need to have certain methods to record music that can use a simple recording setup known as **Deca Tree**. Here, microphones are placed in a particular fashion to capture sounds from the left, right, front, and back of the source. There are also many processing techniques that can filter and convert sounds that are recorded to mimic the various components needed for each of the channels.

We take samples of the analog sound waves that are produced by a piano at close intervals (the rate at which the samples are taken between intervals is known as sampling frequency). The process of taking samples from analog waves to store them digitally is known as **Pulse Code Modulation** (**PCM**). These samples can be stored in uncompressed PCM-like formats or be compressed into a smaller and more manageable file size using audio compression techniques. Wav, MP3, Ogg Vorbis, Dolby TrueHD, and DTS-HD are some of the formats that audio is commonly saved as. Ideally, we want to save audio into a lossless compressed format so that we get a small manageable file that contains amazing sounds.

When the digital format of the sound is played back, the analog sound wave is reconstructed using the stored information. Close resemblance to the original analog sound waves is one way to ensure sounds of good quality. By increasing the number of channels to create a 3D sound effect using the basic 5.1 surround, which requires five speakers, one for front left, one front right, one center, one back left (as surround), one back right (as surround) and a subwoofer, also greatly improves the listening experience.

# The Unreal audio system

We now have a general understanding of why we need audio in games and how it's created and recorded. Let's learn about the Unreal audio system and the editor that can be used to import these audio files into the game, and we'll also learn about the tools that can be used to edit and control playbacks.

# Getting audio into Unreal

How do we get the audio files into Unreal? What do you need to take note of?

## The audio format

Unreal supports the importing of sounds only in the `.wav` format. The `.wav` format is a widely used format that can store raw uncompressed sound data.

## The sampling rate

The sampling rate is recommended at 44100 Hz or 22050 Hz. As mentioned earlier, the sampling rate determines how often the analog wave is recorded. The higher the frequency (measured in Hertz or Hz), the more data points of the analog wave that are collected, which aids in a better reconstruction of the wave.

## Bit depth

The bit depth is set as 16\. It determines the granularity at which the amplitude of the audio wave can be recorded, which is also known as the resolution of the sound. For a bit depth of 16, you can get up to 65,536 integer values (216). The reason why we are concerned with the bit depth is because during the sampling process of the analog waves, the actual value of the amplitude of the wave is approximated to one of the integer values that can be stored based on the bit depth. The following figure shows two different bit depths. The figure on the left-hand side illustrates when the bit depth is low, and the signal is more inaccurately sampled because it is sampled in larger increments. The figure on the right-hand side illustrates when the bit depth is higher, and it can be sampled at smaller increments, resulting in a more accurate representation of the wave:

![Bit depth](img/B03679_06_09.jpg)

The loss in accuracy of the representation of the wave can be termed as a quantization error. When the bit depth is too low, the quantization error is high.

The **Signal to Quantization Noise Ratio** (**SQNR**) is the measurement used to determine the quality of this conversion. It is calculated using the ratio between the maximum nominal signal strength and the quantization error. The better the ratio, the better the conversion.

## Supported sound channels

Unreal currently supports channels such as mono, stereo, 2.1, 4.1, 5.1 6.1, and 7.1.

When importing files into Unreal, take note of the file naming convention that is in place so that the right sound is played from the right channel.

The following table shows the 7.1 surround sound configuration with all the file naming conventions that are necessary for the correct playback:

| **Speakers** | Front-left |   | Front-center |   | Front-right |
| **Extension** | `_fl` |   | `_fc` |   | `_fr` |
|   |   |   |   |   |   |
| **Speakers** | Side-left |   | Low frequency (commonly known as subwoofer) |   | Side-right |
| **Extension** | `_sl` |   | `_lf` |   | `_sr` |
|   |   |   |   |   |   |
| **Speakers** | Back-left |   |   |   | Back-right |
| **Extension** | `_bl` |   |   |   | `_br` |

This table shows you the files that are used for the 5.1 surround system:

| **Speakers** | Front-left |   | Front-center |   | Front-right |
| **Extension** | `_fl` |   | `_fc` |   | `_fr` |
|   |   |   |   |   |   |
| **Speakers** | Side-left |   | Low frequency (commonly known as subwoofer) |   | Side-right |
| **Extension** | `_sl` |   | `_lf` |   | `_sr` |

This table shows you the files that are used for the 4.0 system:

| **Speakers** | Front-left |   |   |   | Front-right |
| **Extension** | `_fl` |   |   |   | `_fr` |
|   |   |   |   |   |   |
| **Speakers** | Side-left |   |   |   | Side-right |
| **Extension** | `_sl` |   |   |   | `_sr` |

# Unreal sound formats and terminologies

There are a couple of terms in the Unreal Sound system that we need to get acquainted with:

*   **Sound waves**: These are the actual audio files that are in the `.wav` format.
*   **Sound cues**: This is the control system for a sound wave file. Sound cues are what we use to manipulate the volume, start, and end of the sound waves. So, in order to control how an audio file is played in the game, you can edit the properties on the Sound Cue, which, in turn, affects the wave file or files that it is associated with.
*   **Ambient Sound Actor**: This is the class actor that you add to the game level. This actor is associated with the Sound Cue to play the audio files that you need for the game.

Now, we are ready to use the Sound Editor in Unreal.

# The Sound Cue Editor

Since we are not editing the actual audio file per se, the sound editor in Unreal is known as the Sound Cue Editor. We are, in fact, editing the way the sound can be played through a control device known as a Sound Cue.

Let's learn more about the functionalities of the Sound Cue Editor.

## How to open the Sound Cue Editor

Go to **Content Browser** | **Audio**. Go to any Sound Cue file, and double-click to open the Sound Cue Editor. This screenshot shows where I could find a Sound Cue in **Content Browser**:

![How to open the Sound Cue Editor](img/B03679_06_10.jpg)

When you double-click on a Sound Cue, the Sound Cue Editor opens up, and it looks quite a lot like the Blueprint Editor with modules and lines. This screenshot shows you what the Sound Cue Editor for **Collapse_Cue** looks like:

![How to open the Sound Cue Editor](img/B03679_06_11.jpg)

Notice that in the preceding screenshot **Collapse_Cue** it has two inputs called **Wave Player: Collapse 01** and **Wave Player: Collapse 02**. These are joined to a **Random** node, and the output goes to the final node known as **Output**. What this does is that when this Sound Cue is played, one of the two collapse sounds gets randomly selected and is played. This creates a variety when sounds are played in the same circumstance; they are both collapse sound effects but slightly different.

We will learn more about the components that we could use to design the Sound Cues later. We'll also go through an exercise later to create our own Sound Cue in the editor.

# Exercise – importing a sound into the Unreal Editor

You may come across a situation where you have created your own audio effect file and want to use it in the game. We will first start by importing this file.

For this exercise, I have used an audio clip downloaded from a Wikipedia site ([https://en.wikipedia.org/wiki/The_Four_Seasons_(Vivaldi)](https://en.wikipedia.org/wiki/The_Four_Seasons_(Vivaldi))) with a Vivaldi piece from The Four Seasons. This is shared by John Harrison.

This file is in the Oggs format, and yes, Unreal only supports `.wav` files. First, I converted the file type from `.ogg` to `.wav` using software that's listed on the Vorbis website at [http://vorbis.com/software/](http://vorbis.com/software/). Be careful about the WAV file settings that Unreal is expecting it to be in.

After getting the right wav file, we are ready to import it into the Sound Editor. Go to **Content Browser** | **Content** | **Audio**, right-click on it to display the contextual menu, navigate to **New Asset** | **Import to /Game/Audio**, and browse to the folder where you saved the `.wav` file and select it. This screenshot shows where you can find the function in the editor to import the `.wav` file:

![Exercise – importing a sound into the Unreal Editor](img/B03679_06_12.jpg)

This screenshot shows you how the Vivaldi WAV file is successfully imported as a sound wave in the `Audio` folder with the WAV file settings:

![Exercise – importing a sound into the Unreal Editor](img/B03679_06_13.jpg)

Next, create a Sound Cue for the Vivaldi sound wave that we have just imported. To recap, a Sound Cue is used to control the playback of the sound wave file. A sound wave file merely has the contents of the audio file. Right-click on the sound wave asset, as shown in this screenshot, and select **Create Cue** in the contextual menu:

![Exercise – importing a sound into the Unreal Editor](img/B03679_06_14.jpg)

Double-click on the newly created Sound Cue (which has a default name with the same name as the sound wave file with a `Cue` suffix). In the example here, it will be `Vivaldi_Spring_Allegro_ByJohnHarrison_Cue`. Double-click on this Cue to view the contents. The following screenshot shows the contents of `Vivaldi_Spring_Allegro_ByJohnHarrison_Cue`. The wave player output is connected directly to **Output**. This is the simplest connection for a Sound Cue where we input the wave to the **Output**.

![Exercise – importing a sound into the Unreal Editor](img/B03679_06_15.jpg)

Now, let's hear the sound we have imported. Within the Sound Cue Editor, look for the **Play Cue** button in the top-left corner of the editor. Take a look at the following screenshot for location of the button. After clicking the button, you would hear the music we have just imported. You have just successfully imported a custom wave file into Unreal. Now, let's transfer it to the game level.

![Exercise – importing a sound into the Unreal Editor](img/B03679_06_16.jpg)

# Exercise – adding custom sounds to a level

In order to place sound in the level, you need to use the **Ambient Sound** node to associate it with a sound cue, which would, in turn, play the audio files.

To create an **Ambient Sound** node, go to **Modes** | **All Classes**, drag and drop **Ambient Sound** into the game level:

![Exercise – adding custom sounds to a level](img/B03679_06_17.jpg)

Click on the Ambient Sound Actor that you have just placed into the level, and rename it `AmbientSound_Vivaldi`. In the **Details** panel, scroll to the **Sound** section, click on the arrow next to **Sound** to display the sound assets that you have in the game level packages, as shown in the following screenshot. Select **Vivaldi_Spring_Allegro_ByJohnHarrison_Cue**.

![Exercise – adding custom sounds to a level](img/B03679_06_18.jpg)

Check whether you can still hear the music by clicking on the **Play** button in the **Details** panel of **AmbientSound_Vivaldi**. Now, let's build the level and run it. Notice that the music plays when you start the level.

# Configuring the Sound Cue Editor

Double-click on **Vivaldi_Spring_Allegro_ByJohnHarrison_Cue** to open the Sound Cue Editor. Notice that on the right-hand side, there is **Palette** with a list of nodes, as shown in the following screenshot. These nodes can be used to control how the sounds are played or heard.

![Configuring the Sound Cue Editor](img/B03679_06_19.jpg)

If you find that your sound design cannot be achieved using the nodes in the list, you can alternatively request for new nodes to be created via the UE4 source code.

# Summary

Both particles and sound are very interesting components of a game and require very specialized skills that are very apt for their design and creation. Particle system creators often have strong artistic and technical backgrounds; an artistic touch is needed to create suitable textures, and a technical ability helps to adjust distributions/values that create an appropriate overall effect. Audio engineers often have a strong music background. They are probably composers and musicians themselves with a passion for games.

In the first half of the chapter, we learned about what a particle system is. We learned how particle systems are used to create in-game effects, such as falling snow, rainfall, flames, fireworks, explosion effects, and much more. A particle system can efficiently render small moving fuzzy particles using textures through a combination of emitters. Each emitter has many configurable modules that can control properties, such as a spawn rate, lifetime, velocity, and the acceleration needed to create the required effect. In this chapter, we covered how to edit an existing fire explosion particle system, turn it into a fireplace effect, and place it in a living room. Through this example, we also went through some basic principles that could be applied to the particle system design process, and how to make minor adjustments to a few popular basic modules to create the effect we wanted.

The second half of the chapter covered how to include sounds in a level. We learned how sounds/music are conceptualized, created, recorded, and eventually, imported into the Unreal Editor. We also covered the audio format that the Unreal Editor currently supports, and a little explanation of each of the components is given to give you a better insight into sounds. Next, we went through a simple exercise to import an online audio file and get the music we have downloaded playing in the game level.

I hope you have gained a little more understanding about the creation process of the particle system and the audio effects that are needed for the games in this chapter. We will continue to improve our game level with a little terrain editing and also create cinematic effects in the next chapter.