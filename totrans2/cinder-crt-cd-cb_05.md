# Chapter 5. Building Particle Systems

In this chapter we will cover:

*   Creating a particle system in 2D
*   Applying repulsion and attraction forces
*   Simulating particles flying in the wind
*   Simulating flocking behavior
*   Making our particles sound reactive
*   Aligning particles to processed images
*   Aligning particles to mesh surfaces
*   Creating springs

# Introduction

Particle systems are a computational technique of using a large number of small graphic objects to perform different types of simulations such as explosions, wind, fire, water, and flocking.

In this chapter, we are going to learn how to create and animate particles using popular and versatile physics algorithms.

# Creating a particle system in 2D

In this recipe, we are going to learn how we can build a basic particle system in two dimensions using the Verlet algorithm.

## Getting ready

We will need to create two classes, a `Particle` class representing a single particle, and a `ParticleSystem` class to manage our particles.

Using your IDE of choice, create the following files:

*   `Particle.h`
*   `Particle.cpp`
*   `ParticleSystem.h`
*   `ParticleSystem.cpp`

## How to do it…

We will learn how we can create a basic particle system. Perform the following steps to do so:

1.  First, let's declare our `Particle` class in the `Particle.h` file and include the necessary Cinder files:

    [PRE0]

2.  Let's add, to the class declaration, the necessary member variables – `ci::Vec2f` to store the position, previous position, and applied forces; and `float` to store particle radius, mass, and drag.

    [PRE1]

3.  The last thing needed to finalize the `Particle` declaration is to add a constructor that takes the particle's initial position, radius, mass, and drag, and methods to update and draw the particle.

    The following is the final `Particle` class declaration:

    [PRE2]

4.  Let's move on to the `Particle.cpp` file and implement the `Particle` class.

    The first necessary step is to include the `Particle.h` file, as follows:

    [PRE3]

5.  We initialize the member variables to the values passed in the constructor. We also initialize `forces` to `zero` and `prevPosition` to the initial position.

    [PRE4]

6.  In the `update` method, we need to create a temporary `ci::Vec2f` variable to store the particle's position before it is updated.

    [PRE5]

7.  We calculate the velocity of the particle by finding the difference between current and previous positions and multiplying it by `drag`. We store this value in `ci::Vec2f` temporarily for clarity.

    [PRE6]

8.  To update the particle's position, we add the previously calculated velocity and add `forces` divided by `mass`.

    [PRE7]

9.  The final steps in the `update` method are to copy the previously stored position to `prevPosition` and reset `forces` to a `zero` vector.

    The following is the complete `update` method implementation:

    [PRE8]

10.  In the `draw` implementation, we simply draw a circle at the particle's position using its radius.

    [PRE9]

11.  Now with the `Particle` class complete, we need to begin working on the `ParticleSystem` class. Move to the `ParticleSystem.h` file, include the necessary files, and create the `ParticleSystem` class declaration.

    [PRE10]

12.  Let's add a destructor and methods to update and draw our particles. We'll also need to create methods to add and destroy particles and finally a `std::vector` variable to store the particles in this system. The following is the final class declaration:

    [PRE11]

13.  Moving to the `ParticleSystem.cpp` file, let's begin working on the implementation. The first thing we need to do is include the file with the class declaration.

    [PRE12]

14.  Now let's implement the methods one by one. In the destructor, we iterate through all the particles and delete them.

    [PRE13]

15.  The `update` method will be used to iterate all the particles and call `update` on each of them.

    [PRE14]

16.  The `draw` method will iterate all the particles and call `draw` on each of them.

    [PRE15]

17.  The `addParticle` method will insert the particle on the `particles` container.

    [PRE16]

18.  Finally, `destroyParticle` will delete the particle and remove it from the particles' list.

    We'll find the particles' iterator and use it to delete and later remove the object from the container.

    [PRE17]

19.  With our classes ready, let's go to our application's class and create some particles.

    In our application's class, we need to include the `ParticleSystem` header file and the necessary header to use random numbers at the top of the source file:

    [PRE18]

20.  Declare a `ParticleSystem` object on our class declaration.

    [PRE19]

21.  In the `setup` method we can create 100 particles with random positions on our window and random radius. We'll define the mass to be the same as the radius as a way to have a relation between size and mass. `drag` will be set to 9.5.

    Add the following code snippet inside the setup method:

    [PRE20]

22.  In the `update` method, we need to update the particles by calling the `update` method on `mParticleSystem`.

    [PRE21]

23.  In the `draw` method we need to clear the screen, set up the window's matrices, and call the `draw` method on `mParticleSystem`.

    [PRE22]

24.  Build and run the application and you will see 100 random circles on screen, as shown in the following screenshot:![How to do it…](img/8703OS_5_1.jpg)

In the next recipes we will learn how to animate the particles in organic and appealing ways.

## How it works...

The method described previously uses a popular and versatile Verlet integrator. One of its main characteristics is an implicit approximation of velocity. This is accomplished by calculating, on each update of the simulation, the distance traveled since the last update of the simulation. This allows for greater stability as velocity is implicit to position and there is less chance these will ever get out of sync.

The `drag` member variable represents resistance to movement and should be a number between 0.0 and 1.0\. A value of 0.0 represents such a great resistance that the particle will not be able to move. A value of 1.0 represents absence of resistance and will make the particle move indefinitely. We applied `drag` in step 7, where we multiplied `drag` by the velocity:

[PRE23]

## There's more...

To create a particle system in 3D it is necessary to use a 3D vector instead of a 2D one.

Since Cinder's vector 2D and 3D vector classes have a very similar class structure, we simply need to change `position`, `prevPosition`, and `forces` to be `ci::Vec3f` objects.

The constructor will also need to take a `ci::Vec3f` object as an argument instead.

The following is the class declaration with these changes:

[PRE24]

The `draw` method should also be changed to allow for 3D drawing; we could, for example, draw a sphere instead of a circle:

[PRE25]

## See also

*   For more information on the implementation of the Verlet algorithm, please refer to the paper by Thomas Jakobsen, located at [http://www.pagines.ma1.upc.edu/~susin/contingut/AdvancedCharacterPhysics.pdf](http://www.pagines.ma1.upc.edu/~susin/contingut/AdvancedCharacterPhysics.pdf)
*   For more information on the Verlet integration, please read the wiki at [http://en.wikipedia.org/wiki/Verlet_integration](http://en.wikipedia.org/wiki/Verlet_integration)

# Applying repulsion and attraction forces

In this recipe, we will show how you can apply repulsion and attraction forces to the particle system that we have implemented in the previous recipe.

## Getting ready

In this recipe, we are going to use the code from the *Creating particle system in 2D* recipe.

## How to do it…

We will illustrate how you can apply forces to the particle system. Perform the following steps:

1.  Add properties to your application's main class.

    [PRE26]

2.  Set the default value inside the `setup` method.

    [PRE27]

3.  Implement the `mouseMove` and `mouseDown` methods, as follows:

    [PRE28]

4.  At the beginning of the `update` method, add the following code snippet:

    [PRE29]

## How it works…

In this example we added interaction to the particles engine introduced in the first recipe. The attraction force is pointing to your mouse cursor position but the repulsion vector points in the opposite direction. These forces were calculated and applied to each particle in steps 3 and 4, and then we made the particles follow your mouse cursor, but when you click on the left mouse button, they are suddenly moves away from the mouse cursor. This effect can be achieved with basic vector operations. Cinder lets you perform vector calculations pretty much the same way you usually do on scalars.

The repulsion force is calculated in step 3\. We are using the normalized vector beginning at the mouse cursor position and the end of the particle position, multiplied by the repulsion factor, calculated on the basis of the distance between the particle and the mouse cursor position. Using the `repulsionRadius` value, we can limit the range of the repulsion.

We are calculating the attraction force in step 4 taking the vector beginning at the particle position and the end at the mouse cursor position. We are multiplying this vector by the `attrFactor` value, which controls the strength of the attraction.

![How it works…](img/8703OS_5_2.jpg)

# Simulating particles flying in the wind

In this recipe, we will explain how you can apply Brownian motion to your particles. Particles are going to behave like snowflakes or leaves flying in the wind.

## Getting ready

In this recipe we are going to use the code base from the *Creating a particle system in 2D* recipe.

## How to do it…

We will add movement to particles calculated from the Perlin noise and sine function. Perform the following steps to do so:

1.  Add the necessary headers.

    [PRE30]

2.  Add properties to your application's main class.

    [PRE31]

3.  Set the default value inside the `setup` method.

    [PRE32]

4.  Change the number of the particles, their radius, and mass.

    [PRE33]

5.  At the beginning of the `update` method, add the following code snippet:

    [PRE34]

## How it works…

The main movement calculations and forces are applied in step 5\. As you can see we are using the Perlin noise algorithm implemented as a part of Cinder. It provides a method to retrieve Brownian motion vectors for each particle. We also add `oscilationVec` that makes particles swing from left-to-right and backwards, adding more realistic behavior.

![How it works…](img/8703OS_5_3.jpg)

## See also

*   **Perlin noise original source**: [http://mrl.nyu.edu/~perlin/doc/oscar.html#noise](http://mrl.nyu.edu/~perlin/doc/oscar.html#noise)
*   **Brownian motion**: [http://en.wikipedia.org/wiki/Brownian_motion](http://en.wikipedia.org/wiki/Brownian_motion)

# Simulating flocking behavior

Flocking is a term applied to the behavior of birds and other flying animals that are organized into a swarm or flock.

From our point of view, it is especially interesting that flocking behavior can be simulated by applying only three rules to each particle (Boid). These rules are as follows:

*   **Separation**: Avoid neighbors that are too near
*   **Alignment**: Steer towards the average velocity of neighbors
*   **Cohesion**: Steer towards the average position of neighbors

## Getting ready

In this recipe, we are going to use the code from the *Creating a particle system in 2D* recipe.

## How to do it…

We will implement the rules for flocking behavior. Perform the following steps to do so:

1.  Change the number of the particles, their radius, and mass.

    [PRE35]

2.  Add a definition for new methods and properties to the `Particle` class inside the `Particle.h` header file.

    [PRE36]

3.  Set the default values for `maxspeed` and `maxforce` at the end of the `Particle` constructor inside the `Particle.cpp` source file.

    [PRE37]

4.  Implement the new methods of the `Particle` class inside the `Particle.cpp` source file.

    [PRE38]

5.  Add a method for the separation rule.

    [PRE39]

6.  Add a method for the alignment rule.

    [PRE40]

7.  Add a method for the cohesion rule.

    [PRE41]

8.  Change the `update` method to read as follows

    [PRE42]

9.  Change the `drawing` method of `Particle`, as follows:

    [PRE43]

10.  Change the `update` method of `ParticleSystem` inside the `ParticleSystem.cpp` source file, as follows:

    [PRE44]

## How it works…

Three rules for flocking—separation, alignment, and cohesion—were implemented starting from step 4 and they were applied to each particle in step 10\. In this step, we also prevented Boids from going out of the window boundaries by resetting their positions.

![How it works…](img/8703OS_5_12.jpg)

## See also

*   **Flocking**: [http://en.wikipedia.org/wiki/Flocking_(behavior)](http://en.wikipedia.org/wiki/Flocking_(behavior))

# Making our particles sound reactive

In this recipe we will pick on the previous particle system and add animations based on **fast Fourier transform** (**FFT**) analysis from an audio file.

The FFT analysis will return a list of values representing the amplitudes of several frequency windows. We will match each particle to a frequency window and use its value to animate the repulsion that each particle applies to all other particles.

This example uses Cinder's FFT processor, which is only available on Mac OS X.

## Getting ready

We will be using the same particle system developed in the previous recipe, *Creating a particle system in 2D*. Create the `Particle` and `ParticleSystem` classes described in that recipe, and include the `ParticleSystem.h` file at the top of the application's source file.

## How to do it…

Using values from the FFT analysis we will animate our particles. Perform the following steps to do so:

1.  Declare a `ParticleSystem` object on your application's class and a variable to store the number of particles we will create.

    [PRE45]

2.  In the `setup` method we'll create 256 random particles. The number of particles will match the number of values we will receive from the audio analysis.

    The particles will begin at a random position on the window and have a random size and mass. `drag` will be `0.9`.

    [PRE46]

3.  In the `update` method, we have to call the `update` method on the particle system.

    [PRE47]

4.  In the `draw` method, we have to clear the background, calculate the window's matrices, and call the `draw` method on the particle system.

    [PRE48]

5.  Now let's load and play an audio file. We start by including the necessary files to load, play, and perform the FFT analysis. Add the following code snippet at the top of the source file:

    [PRE49]

6.  Now declare `ci::audio::TrackRef`, which is a reference to an audio track.

    [PRE50]

7.  In the `setup` method we will open a file dialog to allow the user to select which audio file to play.

    If the retrieved path is not empty, we will use it to load and add a new audio track.

    [PRE51]

8.  We'll check if `mAudio` was successfully loaded and played. We will also enable the PCM buffer and looping.

    [PRE52]

9.  Now that we have an audio file playing, we need to start animating the particles. First we need to apply an elastic force towards the center of the window. We do so by iterating the over all particles and adding a force, which is one-tenth of the difference between the particle's position and the window's center position.

    Add the following code snippet to the `update` method:

    [PRE53]

10.  Now we have to calculate the FFT analysis. This will be done once after every frame in the update.

    Declare a local variable `std::shared_ptr<float>`, where the result of the FFT will be stored.

    We will get a reference to the PCM buffer of `mAudio` and perform an FFT analysis on its left channel. It is a good practice to perform a test to check the validity of `mAudio` and its buffer.

    [PRE54]

11.  We will use the values from the FFT analysis to scale the repulsion each particle is applying.

    Add the following code snippet to the `update` method:

    [PRE55]

12.  Build and run the application; you will be prompted to select an audio file. Select it and it will begin playing. The particles will move and push each other around according to the audio's frequencies.![How to do it…](img/8703OS_5_6.jpg)

## How it works…

We created a particle for each one of the values the FFT analysis returns and made each particle repulse every other particle according to its correspondent frequency window amplitude. As the music evolves, the animation will react accordingly.

## See also

*   To learn more about fast Fourier transform please visit [http://en.wikipedia.org/wiki/Fast_Fourier_transform](http://en.wikipedia.org/wiki/Fast_Fourier_transform)

# Aligning particles to a processed image

In this recipe, we will show how you can use techniques you were introduced to in the previous recipes to make particles align to the edge detected in the image.

## Getting ready

In this recipe, we are going to use the particles' implementation from the *Creating a particle system in 2D* recipe; the image processing example from the *Detecting edges* recipe in [Chapter 3](ch03.html "Chapter 3. Using Image Processing Techniques"), *Using Image Processing Techniques*; as well as simulating repulsion covered in the *Applying repulsion and attraction forces* recipe.

## How to do it…

We will create particles aligning to the detected edges in the image. Perform the following steps to do so:

1.  Add an `anchor` property to the `Particle` class in the `Particle.h` file.

    [PRE56]

2.  Set the `anchor` value at the end of the `Particle` class constructor in the `Particle.cpp` source file.

    [PRE57]

3.  Add a new property to your application's main class.

    [PRE58]

4.  At the end of the `setup` method, after image processing, add new particles, as follows:

    [PRE59]

5.  Implement the `update` method for your main class, as follows:

    [PRE60]

6.  Change the `draw` method for `Particle` inside the `Particle.cpp` source file to read as follows

    [PRE61]

## How it works…

The first major step was to allocate particles at some characteristic points of the image. To do so, we detected the edges, which was covered in the *Detecting edges* recipe in [Chapter 3](ch03.html "Chapter 3. Using Image Processing Techniques"), *Using Image Processing Techniques*. In step 4 you can see that we iterated through each pixel of each processed image and placed particles only at detected features.

You can find another important calculation in step 5, where we tried to move back the particles to their original positions stored in the `anchor` property. To disorder particles, we used the same repulsion code that we used in the *Applying repulsion and attraction forces* recipe.

![How it works…](img/8703OS_5_8.jpg)

## See also

*   To learn more about fast Fourier transform, please visit [http://en.wikipedia.org/wiki/Fast_Fourier_transform](http://en.wikipedia.org/wiki/Fast_Fourier_transform)

# Aligning particles to the mesh surface

In this recipe, we are going to use a 3D version of the particles' code base from the *Creating a particle system in 2D* recipe. To navigate in 3D space, we will use `MayaCamUI` covered in the *Using MayaCamUI* recipe in [Chapter 2](ch02.html "Chapter 2. Preparing for Development"), *Preparing for Development*.

## Getting ready

To simulate repulsion, we are using the code from the *Applying repulsion and attraction forces* recipe with slight modifications for three-dimensional space. For this example, we are using the `ducky.mesh` mesh file that you can find in the `resources` directory of the Picking3D sample inside the Cinder package. Please copy this file to the `assets` folder in your project.

## How to do it…

We will create particles aligned to the mesh. Perform the following steps to do so:

1.  Add an `anchor` property to the `Particle` class in the `Particle.h` file.

    [PRE62]

2.  Set the `anchor` value at the end of the `Particle` class constructor in the `Particle.cpp` source file.

    [PRE63]

3.  Add the necessary headers in your main class.

    [PRE64]

4.  Add the new properties to your application's main class.

    [PRE65]

5.  Set the default values inside the `setup` method.

    [PRE66]

6.  At the end of the `setup` method, add the following code snippet:

    [PRE67]

7.  Add methods for camera navigation.

    [PRE68]

8.  Implement the `update` and `draw` methods for your main application class.

    [PRE69]

9.  Replace the `draw` method for `Particle` inside the `Particle.cpp` source file to read as follows

    [PRE70]

## How it works…

Firstly, we created particles in place of vertices of the mesh that you can see in step 6.

![How it works…](img/8703OS_5_9.jpg)

You can find another important calculation in step 8 where we tried to move particles back to their original positions stored in the `anchor` property. To displace the particles, we used the same repulsion code that we used in the *Applying repulsion and attraction forces* recipe but with slight modifications for three-dimensional space. Basically, it is about using `Vec3f` types instead of `Vec2f`.

![How it works…](img/8703OS_5_10.jpg)

# Creating springs

In this recipe, we will learn how we can create springs.

**Springs** are objects that connect two particles and force them to be at a defined rest distance.

In this example, we will create random particles, and whenever the user presses a mouse button, two random particles will be connected by a new spring with a random rest distance.

## Getting ready

We will be using the same particle system developed in the previous recipe, *Creating a particle system in 2D*. Create the `Particle` and `ParticleSystem` classes described in that recipe and include the `ParticleSystem.h` file at the top of the application source file.

We will be creating a `Spring` class, so it is necessary to create the following files:

*   `Spring.h`
*   `Spring.cpp`

## How to do it…

We will create springs that constrain the movement of particles. Perform the following steps to do so:

1.  In the `Spring.h` file, we will declare a `Spring` class. The first thing we need to do is to add the `#pragma once` macro and include the necessary files.

    [PRE71]

2.  Next, declare the `Spring` class.

    [PRE72]

3.  We will add member variables, two `Particle` pointers to reference the particles that will be connected by this spring, and the `rest` and `strengthfloat` variables.

    [PRE73]

4.  Now we will declare the constructor that will take pointers to two `Particle` objects, and the `rest` and `strength` values.

    We will also declare the `update` and `draw` methods.

    The following is the final `Spring` class declaration:

    [PRE74]

5.  Let's implement the `Spring` class in the `Spring.cpp` file.

    In the constructor, we will set the values of the member values to the ones passed in the arguments.

    [PRE75]

6.  In the `update` method of the `Spring` class, we will calculate the difference between the particles' distance and the spring's rest distance, and adjust them accordingly.

    [PRE76]

7.  In the `draw` method of the `Spring` class, we will simply draw a line connecting both particles.

    [PRE77]

8.  Now we will have to make some changes in the `ParticleSystem` class to allow the addition of springs.

    In the `ParticleSystem` file, include the `Spring.h` file.

    [PRE78]

9.  Declare the `std::vector<Spring*>` member in the class declaration.

    [PRE79]

10.  Declare the `addSpring` and `destroySpring` methods to add and destroy springs to the system.

    The following is the final `ParticleSystem` class declaration:

    [PRE80]

11.  Let's implement the `addSpring` method. In the `ParticleSystem.cpp` file, add the following code snippet:

    [PRE81]

12.  In the implementation of `destroySpring`, we will find the correspondent iterator for the argument `Spring` and remove it from springs. We will also delete the object.

    Add the following code snippet in the `ParticleSystem.cpp` file:

    [PRE82]

13.  It is necessary to alter the `update` method to update all springs.

    The following code snippet shows what the final update should look like:

    [PRE83]

14.  In the `draw` method, we will also need to iterate over all springs and call the `draw` method on them.

    The final implementation of the `ParticleSystem::draw` method should be as follows:

    [PRE84]

15.  We have finished creating the `Spring` class and making all necessary changes to the `ParticleSystem` class.

    Let's go to our application's class and include the `ParticleSystem.h` file:

    [PRE85]

16.  Declare a `ParticleSystem` object.

    [PRE86]

17.  Create some random particles by adding the following code snippet to the `setup` method:

    [PRE87]

18.  In the `update` method, we will need to call the `update` method on `ParticleSystem`.

    [PRE88]

19.  In the `draw` method, clear the background, define the window's matrices, and call the `draw` method on `mParticleSystem`.

    [PRE89]

20.  Since we want to create springs whenever the user presses the mouse, we will need to declare the `mouseDown` method.

    Add the following code snippet to your application's class declaration:

    [PRE90]

21.  In the `mouseDown` implementation we will connect two random particles.

    Start by declaring a `Particle` pointer and defining it as a random particle in the particle system.

    [PRE91]

22.  Now declare a second `Particle` pointer and make it equal to the first one. In the `while` loop, we will set its value to a random particle in `mParticleSystem` until both particles are different. This will avoid the case where both pointers point to the same particle.

    [PRE92]

23.  Now we'll create a `Spring` object that will connect both particles, define a random rest distance, and set `strength` to `1.0`. Add the created spring to `mParticleSystem`.

    The following is the final `mouseDown` implementation:

    [PRE93]

24.  Build and run the application. Every time a mouse button is pressed, two particles will become connected with a white line and their distance will remain unchangeable.![How to do it…](img/87030s_5_11.jpg)

## How it works…

A `Spring` object will calculate the difference between two particles and correct their positions, so that the distance between the two particles will be equal to the springs' rest value.

By using their masses, we will also take into account each particle's mass, so that the correction will take into account the particles' weight.

## There's more…

The same principle can also be applied to particle systems in 3D.

If you are using a 3D particle, as explained in the *There's more…* section of the *Creating a particle system in 2D* recipe, the `Spring` class simply needs to change its calculations to use `ci::Vec3f` instead of `ci::Vec2f`.

The `update` method of the `Spring` class would need to look like the following code snippet:

[PRE94]