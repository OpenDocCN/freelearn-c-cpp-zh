# Sharing Objects with the Flyweight Pattern

We previously learned about object pools in [Chapter 7](part0112.html#3APV00-04600e4b10ea45a2839ef4fc3675aeb7), *Improving Performance with Object Pools*, and that they are great for avoiding slowdowns in our game due to dynamic memory allocation. But, there are still other steps that we can take to reduce the amount of memory that we use to begin with.

When creating projects, you'll often run into times where you want to have many objects on the screen at once. While computers have become much more powerful over the past few years, they still can't handle thousands of complex game objects on the screen by themselves.

In order to accomplish this feat, programmers need to think of ways to lighten the memory load on their program. Using the Flyweight pattern, we abstract the common parts of our object and share them with only the data that's unique to each instance (such as position and current health) being created.

# Chapter overview

In this chapter, we will construct a particle system consisting of two parts: the particle itself, which will be a simple struct, as well as a particle system class that contains the system's data.

We will construct two different types of particle system: an explosion that moves on its own, and a static one that spawns at the position of our player's ship. We will also explore two ways to deal with the system data. The first will be for each particle system to contain its own copy of the system data. Then, after learning about the Flyweight pattern, we will use it to construct separate system data classes that we can assign using files or code. Then, each particle system will simply reference an instance of the system data that it needs.

# Your objectives

This chapter will be split into a number of topics. It will contain a simple step-by-step process from beginning to end. Here is the outline of our tasks:

*   Introduction to particles
*   Implementing particles in Mach5
*   Why memory is still an issue
*   Introduction to the Flyweight pattern
*   Transitioning to ParticleSystems

# Introductions to particles

In game development, you may have heard of particles. They are typically small 2D sprites or simple 3D models that are created in order to simulate *fuzzy* things such as fires, explosions, and smoke trails to add visual flair to your projects. This visual flair is sometimes referred to as *juiciness*. Made popular by indie developers *Martin Jonasson* and *Petri Purho*, making a game *juicy* makes it more enjoyable to play and increases the feedback the player receives by playing the game.

This is usually something worked on more toward the end of development of titles in order to polish the project and add more feedback, but it's a good example of how we can want to have many things on the screen at one time.

For more information on juiciness and to watch their Martin and Petri's GDC talk on the subject, check out [http://www.gamasutra.com/view/news/178938/Video_Is_your_game_juicy_enough.php](http://www.gamasutra.com/view/news/178938/Video_Is_your_game_juicy_enough.php).

The reason that these objects are so simple is because they are spawned hundreds and sometimes thousands of times, and this is done over and over again.

# Implementing particles in Mach5

Now that we know what particles are, let's put them into Mach5 so we can get an example of how they work. We will be creating particles to follow our ship while it moves in a similar fashion to a smoke trail. This will be a great way to show an example of particles on the screen but, to have something to show, we will first need to bring a new archetype into the game.

To do that, open up the `Example Code` folder for this chapter and bring the `particle.tga` file into the `EngineTest/Textures` folder of your Visual Studio project.

After that, open up the `EngineTest/ArcheTypes` folder, create a new text file called `Particle.ini`, and fill it with the following info:

[PRE0]

After that, we need the Mach5 engine to support our new object, so go to the `EngineTest` folder and then double-click on the `PreBuild.bat` file. The `M5ArcheTypes.h` file will be updated to include our particle:

[PRE1]

Nice! Now that we have the object in the game, there's still the issue of putting in the Particle component. Since this component is not exclusive to our game, let's move over to the Core/Components filter and create a new filter called `ParticleComp`. From there, create two new files, `ParticleComponent.h` and `ParticleComponent.cpp`, making sure their locations are set to the `Mach5-master\EngineTest\EngineTest\Source\` folder.

In the `.h` file, use the following code:

[PRE2]

This class looks similar to other components that we've added in the past, but this time we've added a `startScale` property to keep track of what scale our object had at the start of its life, and an `endScale` property to be a modifier on how to change the scale. We also have `lifeTime`, which will be how long this object should live before we remove it, and `lifeLeft`, which will be how much longer this object has to live. Finally, since we are going to change our scale, we added another function, `Lerp`, to linearly interpolate between a starting and ending value.

In the `.cpp` file, use the following code:

[PRE3]

This code will modify the object's scale by using the `Lerp` function to interpolate between the starting and ending scale. We also will modify how much life the particle has left, and if it has none, mark the particle for deletion:

[PRE4]

**Linear interpolation** (**Lerp**) allows us to obtain a value between `start` and `end` using the `fraction` property for how far along the transition it should be. If `fraction` is `0`, we would get the value of `start`. If we give `1`, we will get the value of `end`. If it's `.5`, then we would get the half-way point between `start` and `end`.

For more information on interpolation including linear interpolation, check out *Keith Maggio*'s notes on the topic at [https://keithmaggio.wordpress.com/2011/02/15/math-magician-lerp-slerp-and-nlerp/](https://keithmaggio.wordpress.com/2011/02/15/math-magician-lerp-slerp-and-nlerp/).

[PRE5]

The `Clone` function allows us to create a copy of this object. It will create a new version of this component, and we will initialize the values of the new component with the values we currently have. This is used by the Mach5 engine in the creation of new game objects:

[PRE6]

Just like before, the `FromFile` function will read in our `ini` file we created previously and will use the values from it to set the properties of this component. In our case, here we set `lifeTime`, `lifeLeft`, and `endScale`.

Finally, let's start putting these objects into our game. Open up the `PlayerInputComponent.cpp` file and add the following to the top of the `Update` function:

[PRE7]

This will cause a particle to get spawned in every single frame and have the same position as our ship. Now, if we run the game, we should see some cool stuff! We can see this in the following screenshot:

![](img/00057.jpeg)

As you can see, our ship now has a trail following behind it. Each part is a particle!

# Why memory is still an issue

The particle system that we are currently showing is probably running well enough on some computers, but note that a large number of the variables that we have created hold data that will never change once we've initialized them. Now, generally in programming we would mark a variable that wouldn't change as `const`, but we don't set the variable until we read from a file. We could potentially make the variables static, but there's also the chance that we may want to have more particle systems in the future and I don't want to create an archetype for each one.

If we continue to spawn many particles, the memory that it takes up will increase and we will be wasting valuable space in memory that we could be using for other purposes. To solve this issue, we will employ the Flyweight pattern.

# Introduction to the Flyweight pattern

The Gang of Four states that a Flyweight is a shared object that can be used in multiple contexts simultaneously. Similarly to flyweight in boxing, which is the lightweight boxing category, we can have a lighter object that can be used in different places in our system simultaneously.

While not used terribly often nowadays, the Flyweight pattern can be very helpful in scenarios when memory is constrained.

A Flyweight will consist of two parts: the intrinsic state and the extrinsic state. The intrinsic state is the part that can be shared. The extrinsic state is modified based on the context it's being used in and, as such, cannot be shared.

Let's take a look at a UML diagram to see a closer look:

![](img/00058.jpeg)

We have the **FlyweightFactory** class, which is used to manage the Flyweights. Whenever we request one, we will either give one that's been created or create a new one ourselves.

The **Flyweight** object itself has data that is of whatever type is needed, as long as it won't change depending on the object that we're working with.

Finally, we have the **ConcreteFlyweight**, which acts as our extrinsic information that can access and use our **Flyweight** via the **FlyweightFactory**.

# Transitioning to ParticleSystems

So with that in mind, what we will do is separate the information that will be shared by each particle, which we will call a `ParticleSystem`:

[PRE8]

The class acts as our intrinsic state, which is shared. Since the starting scale, end scale, and lifetime of our object never change, it makes sense for these variables to be shared instead of each object having one. In our previous example, we only had one particle system, but we may want the ability to have more as well, and it's when we start using it that some of the benefits of the Flyweight pattern become even more apparent. That's why we gave this class two virtual functions: `Init` and `Update`. We can have our extrinsic state call these functions, giving the function information about the particular object we're dealing with, and then we can modify it using these properties.

# Creating different system types

Let's add a new type of particle system in addition to our current one that doesn't move. Let's call it `Moving` and our previous one, `Static`. To differentiate between the two, let's add an `enum`:

[PRE9]

We can now modify the original `ParticleComponent` class, by removing the previously created variables and instead including a reference to the kind of `ParticleSystem` we wish to use:

[PRE10]

The `ParticleComponent` class acts as our extrinsic state, holding information about how much time it has left and the properties from the `M5Component` class, such as a reference to the object we want to create.

At this point, we need to create two classes to refer to each of these:

[PRE11]

# Developing the ParticleFactory

We need some way for our `ParticleComponent` to access this information. With that in mind, we will make use of the Factory design pattern that we learned about in [Chapter 5](part0096.html#2RHM00-04600e4b10ea45a2839ef4fc3675aeb7), *Decoupling Code via the Factory Method Pattern*, and create a `ParticleFactory` class:

[PRE12]

This `ParticleFactory` class is what we use to manage the creation of these Flyweights and to ensure that, if the object is already located in our map, we will return it. Otherwise, we will create a new object to be able to access it. I also added an `objectCount` variable to help us know how many objects currently exist and to verify that no memory leaks are occurring.

The `ParticleSystems` variable is of type map, which is actually one of my favorite containers in the `stl` and can be considered an *associative array*. By that, I mean instead of memorizing numbers in order to access certain indexes of an array, you can use a different type, such as a `string,` or in this case, an `enum`.

For more information on the map container, check out [http://www.cprogramming.com/tutorial/stl/stlmap.html](http://www.cprogramming.com/tutorial/stl/stlmap.html).

After this, we will need to define the two static variables:

[PRE13]

# Using the ParticleFactory

Next, we will need to adjust our previously created Particle archetype and component to reflect these changes.

First, we want to change our `.ini` file. Since the `Particle` object is meant for all particle types, instead of having the properties being set there, we will instead set a base type for us to use:

[PRE14]

This simplifies the particle object itself, but it's for a good cause. We will now update the code of the `ParticleComponent` class as follows:

[PRE15]

In this instance, you'll notice that instead of modifying the scale and/or movement being done here, we use the `ParticleFactory` to update our code based on the `particleType` property:

[PRE16]

Here, we call the `Init` function for our particle system based on its type from the factory:

[PRE17]

We are now going to set our particle type based on what is marked on the `ini` file.

But, of course, now that we are using the `GetParticleSystem` function, we need to implement it for our code to compile:

[PRE18]

In this script, we make use of the `particleSystems` map that we talked about earlier. The first thing that we do is check if there is an object in the map that has our `ParticleType` in it. If not, then we need to create one. In this case, I added a `switch` statement that will assign different values depending on the value mentioned in the `case` statement, but you could easily read these values from a text file in a similar manner to how files are read for archetypes. You'll notice that we are calling new in order to create these, so we will need to call `delete` on them as well in order to avoid any memory leaks. To accomplish this, I've added in a destructor for the `ParticleFactory` class:

[PRE19]

Finally, we need to write the implementations for our different `ParticleSystems`:

[PRE20]

The `Lerp` function does the same for either particle type, so it's fine the way it was:

[PRE21]

The static version of the `Init` and `Update` functions will just set our velocity to `0` so we don't move:

[PRE22]

For our moving particle system, we will set our velocity to a random number in the *x* and *y* axis, causing a nice explosion effect!

Now, instead of creating a copy of this data each time, we will have one copy that we will access, as shown in the following screenshot:

![](img/00059.jpeg)

As we play, you'll notice that we now have a new particle system working and it's doing its job quite well.

# Summary

Over the course of this chapter, we learned about particles and how they can be used in order to improve the polish of our game project. We learned how we can implement a particle system inside of the Mach5 engine, and then learned about the Flyweight pattern and how it can be used effectively in order to reduce the memory usage on your projects. We saw how to do this by making use of the Factory pattern too, while making it a lot easier for us to create new particle system types as well. Keeping this in mind, it will be a lot easier in the future to break apart pieces of your programs that stay consistent and only create additional variables when you need to!

Moving forward, in the next chapter we will dive into graphics and the concepts needed to understand how our code will affect moving and animating game objects.