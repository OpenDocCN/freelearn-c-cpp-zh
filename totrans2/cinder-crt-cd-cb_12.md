# Chapter 12. Using Audio Input and Output

In this chapter, we will learn how to generate sounds using examples of ways to generate sounds driven by physics simulation. We will also present examples of visualizing sound with audio reactive animations.

The following recipes will cover:

*   Generating a sine oscillator
*   Generating sound with frequency modulation
*   Adding a delay effect
*   Generating sound upon the collision of objects
*   Visualizing FFT
*   Making sound-reactive particles

# Generating a sine oscillator

In this recipe, we will learn how to generatively create a sine wave oscillator by manipulating the sound card's **PCM** (**Pulse-code Modulation** ) audio buffer. The frequency of the sine wave will be defined by the mouse's y coordinate.

We will also draw the sine wave for a visual representation.

## Getting ready

Include the following files:

[PRE0]

And add the following useful `using` statements:

[PRE1]

## How to do it…

We will create a sine wave oscillator using the following steps:

1.  Declare the following member variables and the callback method:

    [PRE2]

2.  In the `setup` module we will initialize the variables and create the audio callback using the following code:

    [PRE3]

3.  In the `update` module we will update `mFrequency` based on the mouse's `y` position. The mouse's position will be mapped and clamped to a frequency value between `0` and `5000`:

    [PRE4]

    Let's implement the audio callback. We'll begin by resizing `mOutput` if necessary. Then we will calculate and interpolate `mPhaseAdd`, and then loop through all the values in the audio buffer and calculate their values based on the sine of `mPhase` and add `mPhaseAdd` to `mPhase`:

    [PRE5]

4.  Finally, we need to draw the sine wave. In the `draw` method, we will clear the background with black and draw a scaled up sine wave with a line strip using the values stored in `mOutput`:

    [PRE6]

5.  Build and run the application. Move the mouse vertically to change the frequency. A line representing the generated sine wave is shown in the following screenshot:![How to do it…](img/8703OS_12_01.jpg)

## How it works…

We are manipulating the PCM buffer. PCM is a method to represent audio through values' samples at regular intervals. By accessing the PCM buffer, we can directly manipulate the audio signal that will be output by the sound card.

Every time the `audioCallback` method is called, we receive a sample of the PCM buffer, where we calculate the values to generate a continuous sine wave.

In the `update` module, we calculate the frequency by mapping the mouse's `y` position.

In the following line in the `audioCallback` implementation, we calculate how much `mPhase` has to increase based on a sample rate of `44100` to generate a wave with a frequency of `mFrequency`:

[PRE7]

# Generating sound with frequency modulation

In this recipe, we will learn how to modulate a sine wave oscillator using another low frequency sine wave.

We will be basing this recipe on the previous recipe, where the `y` position of the mouse controlled the frequency of the sine wave; in this recipe, we will use the `x` position of the mouse to control the modulation frequency.

## Getting ready

We will be using the code from the previous recipe, *Generating a sine oscillator*.

## How to do it…

We will multiply the sine wave created in the previous recipe with another low frequency sine wave.

1.  Add the following member variables:

    [PRE8]

2.  Add the following in the `setup` module to initialize the variables created previously:

    [PRE9]

3.  In the `update` module, add the following code to calculate the modulation frequency based on the `x` position of the mouse cursor:

    [PRE10]

4.  We will need to calculate another sine wave using `mModFrequency`, `mModPhase`, and `mModPhaseAdd`, and use it to modulate our first sine wave.

    The following is the implementation of `audioCallback`:

    [PRE11]

5.  Build and run the application. Move the mouse cursor over the y axis to determine the frequency, and over the x axis to determine the modulation frequency.

We can see how the sine wave created changes in the previous recipe, in the amplitude as it is multiplied by another low frequency sine wave.

![How to do it…](img/8703OS_12_02.jpg)

## How it works…

We calculate a second sine wave with a **low frequency oscillation** (**LFO**) and use it to modulate the first sine wave. To modulate the waves, we multiply them by each other.

# Adding a delay effect

In this recipe, we will learn how to add a delay effect to the frequency modulation audio generated in the previous recipe.

## Getting ready

We will use the source code from the previous recipe, *Generating sound with frequency modulation*.

## How to do it…

We will store our audio values and play them after an interval to achieve a delay effect using the following steps:

1.  Add the following member variables:

    [PRE12]

    Let's initialize the variables created above and initialize our delay line with zeros.

    Then add the following in the `setup` method:

    [PRE13]

2.  In the implementation of our `audioCallback` method, we will read back from the buffer the values that were generated in the frequency modulation and calculate the delay.

    The final value is again passed into the buffer for output.

    Add the following code in the `audioCallback` method:

    [PRE14]

3.  Build and run the application. By moving the mouse in the x axis, you control the oscillator frequency, and by moving the mouse in the y axis, you control the modulation frequency. The output will contain a delay effect as shown in the following screenshot:![How to do it…](img/8703OS_12_03.jpg)

## How it works...

A delay is an audio effect where an input is stored and then played back after a determined amount of time. We achieve this by creating a buffer the size of `mDelay` multiplied by the frequency rate. Each time `audioCallback` gets called, we read from the delay line and update the delay line with the current output value. We then add the delay value to the output and advance `mDelayIndex`.

# Generating sound upon the collision of objects

In this recipe, we will learn how to apply simple physics to object particles and generate sound upon the collision of two objects.

## Getting ready

In this example, we are using code described in the recipe *Generating a sine oscillator* in this chapter, so please refer to that recipe.

## How to do it…

We will create a Cinder application to illustrate the mechanism:

1.  Include the following necessary header files:

    [PRE15]

2.  Add members to the application's `main` class for particle simulation:

    [PRE16]

3.  Add members to the application's `main` class to make the particles interactive:

    [PRE17]

4.  Add members for the generation of sound:

    [PRE18]

5.  Initialize the particle system inside the `setup` method:

    [PRE19]

6.  Initialize the members to generate sound and register an audio callback inside the `setup` method:

    [PRE20]

7.  Implement the `resize` method to update the attractor position whenever an application window will be resized:

    [PRE21]

8.  Implement the mouse events handlers for mouse interaction with particles:

    [PRE22]

    [PRE23]

9.  Inside the `update` method, add the following code for sound frequency calculation:

    [PRE24]

10.  Inside the `update` method, add the following code for particle movement calculation. At this point, we are detecting collisions and calculating the sound frequency:

    [PRE25]

11.  Update position of dragging particle, if any, and update particle system:

    [PRE26]

12.  Draw particles by implementing the `draw` method as follows:

    [PRE27]

13.  Implement audio callback handler as covered in the recipe *Generating a sine oscillator*.

## How it works…

We are generating random particles with applied physics and collision detection. While collision is detected, a frequency of a sine wave is calculated based on the particles' radii.

![How it works…](img/8703OS_12_04.jpg)

Inside the `update` method, we are iterating through the particles and checking the distance between each of them to detect collision, if it occurs. A generated frequency is calculated from the radii of the colliding particles—the bigger the radius, the lower the frequency of the sound.

# Visualizing FFT

In this recipe, we will show an example of **FFT** (**Fast Fourier Transform**) data visualization on a circular layout with some smooth animation.

## Getting ready

Save you favorite music piece in assets folder with the name `music.mp3`.

## How to do it…

We will create visualization based on an example FFT analysis using the following steps:

1.  Include the following necessary header files:

    [PRE28]

2.  Add the following members to your main application class:

    [PRE29]

3.  Inside the `setup` method, initialize the members and load the sound file from the assets folder. We are decomposing the signal into 32 frequencies using FFT:

    [PRE30]

4.  Implement the `update` method as follows:

    [PRE31]

5.  Implement the `draw` method as follows:

    [PRE32]

6.  Implement the `drawFft` method as follows:

    [PRE33]

## How it works…

We can divide visualization into bands, and the grey circle with alpha in the center. Bands are straight representations of data calculated by the `audio::calculateFft` function, and animated with some smoothing by going back towards the center. The grey circle shown in the following screenshot represents the average level of all the bands.

FFT is an algorithm to compute **DFT** (**Discrete Fourier Transform**), which decomposes the signal into list of different frequencies.

![How it works…](img/8703OS_12_05.jpg)

# Making sound-reactive particles

In this recipe, we will show an example of audio visualization based on audio-reactive particles.

## Getting ready

Save your favorite music piece in assets folder with the name `music.mp3`.

Please refer to [Chapter 6](ch06.html "Chapter 6. Rendering and Texturing Particle Systems"), *Adding a Tail to Our Particles*, for instructions on how to draw particles with a tile.

## How to do it…

We will create a sample audio-reactive visualization using the following steps:

1.  Add the following necessary header files:

    [PRE34]

2.  Add the following members for audio playback and analysis:

    [PRE35]

3.  Add the following members for particle simulation:

    [PRE36]

4.  Inside the `setup` method, initialize the simulation of the members and particles:

    [PRE37]

5.  Inside the `setup` method, initialize camera and audio playback:

    [PRE38]

6.  Implement the `resize` method for updating camera properties in regards to resizing windows:

    [PRE39]

7.  Inside the `update` method, implement a simple beat detection. We are decomposing the signal into 32 frequencies using FFT:

    [PRE40]

8.  Also, inside the `update` method, calculate the particle simulation:

    [PRE41]

9.  Implement the `draw` method as follows:

    [PRE42]

## How it works…

A particle is drawn as a black dot, or more precisely a sphere and a line as a tail. Due to specific frequency difference, forces repelling particles from the center of the attractor are applied, with a random vector added to these forces.

![How it works…](img/8703OS_12_06.jpg)

## There's more…

You might want to customize the visualization for a specific music piece.

### Adding GUI to tweak parameters

We will add GUI that affects particles' behavior using the following steps:

1.  Add the following necessary header file:

    [PRE43]

2.  Add the following member to your application's `main` class:

    [PRE44]

3.  At the end of the `setup` method, initialize GUI using the following code:

    [PRE45]

4.  At the and of the `draw` method, add the following code:

    [PRE46]