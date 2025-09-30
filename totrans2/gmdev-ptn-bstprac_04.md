# Artificial Intelligence Using the State Pattern

In the last chapter, we discussed the Component Object Model. Giving an entity a behavior is now as simple as just creating a new component and having that control the game object.

Whenever someone starts to make a game, they begin by writing gameplay code. That is the fun stuff. Everyone wants to see graphics and physics take effect on screen. Things such as a pause screen, options menu, or even a second level are an afterthought. The same happens for organizing the behaviors of a player. Programmers are excited to make a player jump and make a player dash, but with each new ability a player has, there are combinations that you may want to disallow. For example, the player might not be allowed to dash while jumping, or may only be able to dash every 3 seconds. The State pattern solves these problems.

By coding the Game State Manager first, the problem of switching to a menu or pausing is solved. By coding finite State Machines as a component for the game object, the problem of complex behavior or a player or enemy is solved. By adding multiple State Machines to the same game object, complex behavior can be created with very simple code, as is seen in many games, and is a widely used feature built into the Unreal Engine and given visual editors in Unity using Hutong Games LLC's popular Playmaker extension.

# Chapter overview

In this chapter, we will create a simple State Machine to control the player via input, as well as create an enemy State Machine which will detect when the player gets close to it and will follow them when in range. We will also look at the base `StateMachineComponent` class in the Mach5 Engine and show that, by writing code for a few states, we can create a more complex behavior quite easily. We will also show that adding more than one State Machine to an object can create multiple behaviors to run at the same time, avoiding duplicated state code.

# Your objectives

This chapter will be split into a number of topics. It will contain a simple step-by-step process from beginning to end. Here is the outline of our tasks:

*   The State pattern explained
*   Introduction to State Machines
*   An overview of enumerations
*   Doing things based on our states
*   Why if statements could get you fired
*   Expanding on the State Machine
*   The State pattern in action--M5StateMachine
*   The State pattern in action--StageManager
*   Issues with FSMs

# The State pattern explained

The State pattern is a way to allow a game object to change its behavior and functionality in response to different stimuli within the game, especially when a variable or a condition within that object changes, as those changes may trigger a change in state. The state of an object is managed by some context (commonly referred to in the game industry as a machine), but the states tell the machine when to change states and thereby functionality. The State pattern contains two parts: the state and the context. The context object holds the current state and can be used by the state class to change what state should be run, whereas the state object holds the actual functionality:

![](img/00033.jpeg)

In the Mach5 Engine, there is a class that already uses the State pattern (`M5StateMachine`) but, before we dive into a completed version, let's actually build one from scratch.

There are multiple ways to implement the State pattern or to get a State-like behavior. We'll go over some of the commonly seen versions and the pros and cons of using them before moving to our final version.

# Introduction to State Machines

We often write code to react to things happening within the game environment based on the expectations of us, as well as our players. For instance, if we are creating a 2D side-scrolling platformer game, when the player presses one of the arrow keys, we're expecting the player's character to move and, whenever we press the spacebar, we expect the sprite to jump into the air. Or perhaps in a 3D game, when our player sees a panel with a large button, they expect to be able to press it.

Tons of things in our ordinary lives act this way as well, reactive to certain stimuli. For instance, when you use your television remote, you expect certain things to happen, or even when swiping or tapping on your mobile phone. Based on the stimuli provided, the *state* of our object may change. We call something that can be in one of multiple states at a time a State Machine.

Almost every program you write can be considered a State Machine of some sort. The second that you add in an `if` statement to your project, you have developed code that can be in at least one of those states. That being said, you don't want to have a bunch of `switch` and/or `if` statements inside of your code as it can quickly get out of hand and make it difficult to understand exactly what it is that your code is doing.

As programmers, we often want to take our problems and break them down over and over again until they're in their simplest form, so let's see a possible way to do that. In game development, you'll hear references to an **FSM** which stands for **Finite State Machine**. Finite means that there are only a certain number of states and that they are all clearly defined for what they can do and how they will change between states.

# An overview of enumerations

Let's say we're going to create a simple enemy. This enemy will not do anything by default, but if the player is nearby, it will move toward them. However, if the player gets too far away from them, then they will stop their pursuit. Finally, if the player shoots the enemy, it will die. So, keeping that in mind, we can extract the states that we'll need. They are as follows:

*   Idle
*   Follow
*   Death

While we are creating our State Machine, we need some way to keep track of what state our objects are going to be in. One may think a way to do this would be to have a `bool` (Boolean value of true or false) for each possible state there is and then set them all to `false`, except for the state that we're in. This is a very bad idea.

Another thought could be to just have an integer and then set a value for each one that there is. This is also a bad idea, as using numbers in this way is basically the same thing as using magic numbers in our code, since the numbers have no logic to them for people to read. Alternatively, you could have `#defines` for each possible value, but that will allow people to put in whatever number they want without any protections at all. Instead, whenever we see a series of things where only one of them is true at a time, we can make use of the programming feature of enumerations, called enums for short.

The basic concept of using enumerations is that you get to create your own custom data types which are restricted to only have a certain list of values. Unlike integers or `#defines`, these numbers are expressed using constants and allow us to have all of the advantages of having a value, such as being able to compare values. In our case, an `enum` for our states would look something like the following:

[PRE0]

# Acting on states

Now that we have our states defined, let's now make it so that we can actually do something in our code based on what state our object is in. For this first example, I'm going to update the `ChasePlayerComponent` class that already exists in the `EngineTest` project.

From the Solution Explorer tab on the right-hand side, open up the `SpaceShooter/Components/ChasePlayerComp` folder and access the `ChasePlayerComponent.h` file. From there, replace the class with the following changes in bold:

[PRE1]

The `FollowPlayer` and `GetDistanceFromPlayer` functions are going to be helper functions for our functionality. We've added our state `enum` to store each of the possible states we can be in, and we added the `m_currentState` variable to hold the current state we are in. To determine when we should switch states, we have two other values, `m_followDistance` and `m_loseDistance`, which are the distance in pixels that our player needs to be from the enemy to follow them, and then how far the player needs to get away to escape, respectively.

Now that we have that finished, let's first go ahead and add in the helper functions at the bottom of the `ChasePlayerComponent.cpp` file so that we can have the proper functionality, once we update our other functions:

[PRE2]

These functions use some basic linear algebra in order to move our object toward the player and to get the distance between two positions.

Diving into the mathematics behind it is out of the scope of this book, but if you're interested in learning more, I highly suggest you check out the following link. The code is written for Cocos2D so it will not be exactly the same as what Mach5 would use, but the concepts are explained very well: [https://www.raywenderlich.com/35866/trigonometry-for-game-programming-part-1](https://www.raywenderlich.com/35866/trigonometry-for-game-programming-part-1).

Now that we have that functionality in, we need to update a couple of things. First of all, we will use the constructor to set the initial value of our `currentState` variable:

[PRE3]

Next, we need to tell our object to read in the values of our object through its INI file:

[PRE4]

`FromFile` is only called once on the first object that gets created in initialization. In order to make it easy to tweak values without having to recompile the project, Mach 5 reads in information from a file to set variables. We haven't modified the `.ini` file yet, but we will once we finish all of these modifications:

[PRE5]

We then need to go to Windows Explorer and move to the project's `EngineTest/EngineTest/ArcheTypes` folder, and then access the `Raider.ini` file and add the new properties to the object:

[PRE6]

If a text editor doesn't open for you, feel free to use Notepad. In this case, we are adding in two new properties which represent the values we created earlier.

Then, we need to update our stage so it's a little easier for us to do some testing. Back in Windows Explorer, open up the `EngineTest/EngineTest/Stages` folder and then open up the `Level01.ini` file and set it to the following:

[PRE7]

With this, our level will just have our player in the center of the world and an enemy Raider positioned at (`100`, `10`). With all of that accomplished, save the files and dive back into our `ChasePlayerComponent.cpp` file and replace the `Update` function with the following:

[PRE8]

Save everything and go ahead and run the project. If all goes well, you should see a scene like this:

![](img/00034.jpeg)

Notice that our enemy is not moving at the beginning due to it being in the Idle state. However, if we move closer to it, it would look something like this:

![](img/00035.jpeg)

You'll see that it now follows us without stopping. If we manage to move far enough away from the enemy though, they'll stop:

![](img/00036.jpeg)

This clearly shows the basic principles of the State pattern in use, though there are a number of things we can do to improve this, which we will talk about soon.

# Issues with conditionals

The next thing we need to consider is how we should do something based on what state we are in. When writing programs, conditional clauses such as the `if` and `switch` statements that we learned about earlier may make your code more difficult to manage. Sometimes, when writing code for specific functionality, writing if statements is completely understandable, especially if it makes sense when you are writing it. For example, the following code makes perfect sense:

[PRE9]

However, if you are writing something where you are checking what the type of an object is, or whether a variable is of a certain type, that is a bit of an issue. For instance, look at the following function:

[PRE10]

As you can see, if we start going down this path, we will need to add many different checks throughout our project, which will make our code hard to change if we ever decide to add more things to support here. First of all, instead of a bunch of `if`/`else` statements, when we see something that's comparing the same value and doing something based off of that value, we should be using a `switch` statement, like we did earlier, with a few modifications:

[PRE11]

But in this particular case, we are just calling a different function based on the value, with each of the functions being some kind of attack. Instead, we should make use of polymorphism and have the code automatically do the correct thing:

[PRE12]

Now whenever we call `AttackPlayer`, it will do the correct thing automatically.

Just remember that creating complex behavior leads to ugly code being written and increases the likelihood of bugs. If you forget about a condition that needs to be there, your game hopefully would break, letting you know there is a problem, but it could not do anything. Then, when you find your game crashes down the road, your life becomes a lot more complex and your game could become unplayable or just plainly not fun.

Robert Elder has a link of the subject which I think explains the kind of crazy things that you can do with conditional statements, which would almost certainly get you fired: [http://blog.robertelder.org/switch-statements-statement-expressions/](http://blog.robertelder.org/switch-statements-statement-expressions/).

Don't lose sleep over having conditionals in your code, but make sure that you only include them when you actually need them there. As you continue coding, you'll have a better idea as to when it's a good idea or not, but it is something to keep in mind.

# Expanding on the State Machine

So currently, you'll notice that in the Idle state we are setting our velocity to `0,0` every single frame. In this simple example, it's not a terribly big deal, but this overdoing of calculations is something that we'd like to avoid in the future. We only really need to do it once, right when we enter the state. We may also want to do certain actions when we leave the state, but we won't be able to do that in the current form of our State Machine, so we are going to need to redo some stuff.

First, let's go back to the `ChasePlayerComponent.h` file and add the following bold function definitions:

[PRE13]

So instead of having our `Update` function handle everything, we've now created three functions for each of the different times that our state can be in: entering a new state, updating based on that state, and then what to do when we leave the state. Aside from that, we also have a `SetNewState` function which will take care of changing the state to something else. All of the functions take in a `State` enum to choose how to execute, with the `Update` state also having the time that passed this frame, and the `SetNewState` having an option for saying it's the first time you've set a state so you don't need to leave the previous one. After that, we need to actually add in the functionality for these new functions:

[PRE14]

And then, we need to update our `Update` function to just call our correct function:

[PRE15]

We also need to change our constructor so that instead of setting the current state, we set it ourselves:

[PRE16]

First of all, note that I am calling the `M5DEBUG_PRINT` function. This is to make it easy to tell that we are changing between different states. For the purposes of this demonstration, I commented out the `Update` function's version, but it could be useful for you to check it out. Note in this version, we have a `switch` statement for each of the functions and do something differently based on the state that is set in there.

In my version of the editor, by default the text will not be displayed on the screen. To fix this issue, go to the `SplashStage.cpp` file and comment out the following bold code:

[PRE17]

Now let's run our project!

![](img/00037.jpeg)

You can tell from the editor when we are switching our states and that the code is being called correctly!

This version works pretty well, but there are some issues with it; namely that it involves a lot of rewriting, and we will need to copy/paste this functionality and make changes anytime we want to make a new version. Next, we will take a look at the State Machine included in the Mach5 Engine and the advantages that it has over what we've been talking about so far.

# The State pattern in action - the M5StateMachine class

The Mach5 Engine itself also has its own implementation of a State Machine, using inheritance to allow users to not have to rewrite the base functionality over and over again and using function pointers instead of having one function for each state. A function pointer is what it sounds like--a pointer to the address in memory where the function is--and we can call it from that information.

To learn more about function pointers and how they are used, check out [http://www.cprogramming.com/tutorial/function-pointers.html](http://www.cprogramming.com/tutorial/function-pointers.html).

You can take a look at the base version of one here, starting with the `Header` file:

[PRE18]

In the preceding code, note that we finally broke apart the `StateMachine` and the `State` object into their own classes, with the state function having its own `Enter`, `Update`, and `Exit` functions. The State Machine keeps track of the current state that we are in and updates appropriately using the `Update` and `SetNextState` functions, and a `SetStateState` function is used to dictate what state we should start from. The implementation for the class looks a little something like this:

[PRE19]

This system provides a template that we can expand upon, in order to create more interesting behavior that does something a bit more complex. Take, for example, the `RandomGoComponent` class, whose header looks like this:

[PRE20]

This class contains three states, `Find`, `Rotate`, and `Go`, which have been added as objects in the `RandomGoComponent`. Each of the states has their own `Enter`, `Update`, and `Exit` functionality, in addition to the constructor and a reference to their parent. The implementation for the classes looks something like this:

[PRE21]

This class will just tell our main State Machine where its intended location is. This only needs to be done once, so it is done in the `Enter` state. The `Update` state just states that after this is done, we want to move to the `Rotate` state, and `Exit` does nothing. Technically, we could not create it, and that would be fine as well since the base class doesn't do anything as well, but it is here if you wish to expand upon it:

[PRE22]

The `Rotate` state will just rotate the character till it is facing the location that it wants to go to. If it is within the range of the rotation, it will then switch to the `Go` state. Before leaving though, it will set the velocity of our parent to the appropriate direction in the `Exit` function:

[PRE23]

The `Go` state merely checks whether the enemy intersects with the target that we are set to go to. If it does, we then set our state to move back to the `Find` state and start everything over again, and also stop the player from moving in the `Exit` function:

[PRE24]

As you can see, this works in a very similar way to what we have done before--setting our first state, getting the initial values from the INI file, and then setting things properly when cloned. Finally, we also have a `GetState` function which will return the current state that the player has using a switch like we talked about previously.

To see this in action, go ahead and go to the `Raider.ini` file and modify the code to fit the following:

[PRE25]

If all went well, save the file and then run the project!

![](img/00038.jpeg)

Now we will see the enemy continually move into new areas, rotating before going there!

# The State pattern in action - StageManager

Another aspect of the Mach5 Engine that uses the State pattern is the `M5StageManager` class:

[PRE26]

Since there will only be one of these in the game, all of the functionality has been made static similarly to a Singleton but, depending on the state that the project is in, it will do different things. Take, for example, changing what stage we are in. I'm sure you'll find that it looks very similar to how we changed states earlier:

[PRE27]

I highly advise taking a closer look at the file and going through each function to see how they interact with each other.

# Issues with FSMs

We've seen some of the ways in which FSMs can be valuable things to add to your projects and how they can make simple AI behaviors much easier, but there are some issues with them.

Traditional FSMs such as the ones we've displayed here can, over time, become unmanageable as you continue to add many different states to them. The difficult part is keeping the number of states to a minimum while also adding complexity by adding new contexts in which your characters can respond.

You'll also have a lot of similar code being written as you'll be rebuilding different behaviors that have pieces of others, which can also be time-consuming. Another thing that's been going on recently in the game industry is AI programmers moving on to more complex ways of handing AI, such as behavior trees.

If you're interested in why some people believe that the age of Finite State Machines is over, check out [http://aigamedev.com/open/article/fsm-age-is-over/](http://aigamedev.com/open/article/fsm-age-is-over/). A look at the issues with FSMs, as well as some potential solutions to fix those issues, can be found here: [http://aigamedev.com/open/article/hfsm-gist/](http://aigamedev.com/open/article/fsm-age-is-over/).

# Summary

In this chapter, we learned about the State pattern, which is a way to allow a game object to change its behavior and functionality in response to different stimuli within the game. We learned about the State and the Context (Machine) and how they are used together. We then learned how we can use the State pattern to gain some exposure toward AI programming, as well as how our project's Game State Manager works and why it's important. Of course, FSMs are most popular in being used for AI, but can also be used in UI as well as dealing with user input, making them another useful tool to have in your arsenal.