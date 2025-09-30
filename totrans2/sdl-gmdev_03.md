# Chapter 3. Working with Game Objects

All games have objects, for example, players, enemies, **non-player character** (**NPC**), traps, bullets, and doors. Keeping track of all these objects and how they interact with each other is a big task and one that we would like to make as simple as possible. Our game could become unwieldy and difficult to update if we do not have a solid implementation. So what can we do to make our task easier? We can start by really trying to leverage the power of **object-oriented programming** (**OOP**). We will cover the following in this chapter:

*   Using inheritance
*   Implementing polymorphism
*   Using abstract base classes
*   Effective inheritance design

# Using inheritance

The first powerful feature of OOP we will look at is inheritance. This feature can help us enormously when developing our reusable framework. Through the use of inheritance, we can share common functionality between similar classes and also create subtypes from existing types. We will not go into too much detail about inheritance itself but instead we will start to think about how we will apply it to our framework.

As mentioned earlier, all games have objects of various types. In most cases, these objects will have a lot of the same data and require a lot of the same basic functions. Let's look at some examples of this common functionality:

*   Almost all of our objects will be drawn to the screen, thus requiring a `draw` function
*   If our objects are to be drawn, they will need a location to draw to, that is, x and y position variables
*   We don't want static objects all the time, so we will need an `update` function
*   Objects will be responsible for cleaning up after themselves; a function that deals with this will be important

This is a good starting point for our first game object class, so let's go ahead and create it. Add a new class to the project called `GameObject` and we can begin:

[PRE0]

### Note

The public, protected, and private keywords are very important. Public functions and data are accessible from anywhere. Protected status restricts access to only those classes derived from it. Private members are only available to that class, not even its derived classes.

So, there we have our first game object class. Now let's inherit from it and create a class called `Player`:

[PRE1]

What we have achieved is the ability to reuse the code and data that we originally had in `GameObject` and apply it to our new `Player` class. As you can see, a derived class can override the functionality of a parent class:

[PRE2]

Or it can even use the functionality of the parent class, while also having its own additional functionality on top:

[PRE3]

Here we call the `draw` function from `GameObject` and then define some player-specific functionality.

### Note

The `::` operator is called the scope resolution operator and it is used to identify the specific place that some data or function resides.

Okay, so far our classes do not do much, so let's add some of our SDL functionality. We will add some drawing code to the `GameObject` class and then reuse it within our `Player` class. First we will update our `GameObject` header file with some new values and functions to allow us to use our existing SDL code:

[PRE4]

We now have some new member variables that will be set in the new `load` function. We are also passing in the `SDL_Renderer` object we want to use in our `draw` function. Let's define these functions in an implementation file and create `GameObject.cpp`:

First define our new `load` function:

[PRE5]

Here we are setting all of the values we declared in the header file. We will just use a start value of `1` for our `m_currentRow` and `m_currentFrame` values. Now we can create our `draw` function that will make use of these values:

[PRE6]

We grab the texture we want from `TextureManager` using `m_textureID` and draw it according to our set values. Finally we can just put something in our `update` function that we can override in the `Player` class:

[PRE7]

Our `GameObject` class is complete for now. We can now alter the `Player` header file to reflect our changes:

[PRE8]

We can now move on to defining these functions in an implementation file. Create `Player.cpp` and we'll walk through the functions. First we will start with the `load` function:

[PRE9]

Here we can use our `GameObject::load` function. And the same applies to our `draw` function:

[PRE10]

And let's override the `update` function with something different; let's animate this one and move it in the opposite direction:

[PRE11]

We are all set; we can create these objects in the `Game` header file:

[PRE12]

Then load them in the `init` function:

[PRE13]

They will then need to be added to the `render` and `update` functions:

[PRE14]

We have one more thing to add to make this run correctly. We need to cap our frame rate slightly; if we do not, then our objects will move far too fast. We will go into more detail about this in a later chapter, but for now we can just put a delay in our main loop. So, back in `main.cpp`, we can add this line:

[PRE15]

Now build and run to see our two separate objects:

![Using inheritance](img/6821OT_03_01.jpg)

Our `Player` class was extremely easy to write, as we had already written some of the code in our `GameObject` class, along with the needed variables. You may have noticed, however, that we were copying code into a lot of places in the `Game` class. It requires a lot of steps to create and add a new object to the game. This is not ideal, as it would be easy to miss a step and also it will get extremely hard to manage and maintain when a game goes beyond having two or three different objects.

What we really want is for our `Game` class not to need to care about different types; then we could loop through all of our game objects in one go, with separate loops for each of their functions.

# Implementing polymorphism

This leads us to our next OOP feature, polymorphism. What polymorphism allows us to do is to refer to an object through a pointer to its parent or base class. This may not seem powerful at first, but what this will allow us to do is essentially have our `Game` class need only to store a list of pointers to one type and any derived types can also be added to this list.

Let us take our `GameObject` and `Player` classes as examples, with an added derived class, `Enemy`. In our `Game` class we have an array of `GameObject*`:

[PRE16]

We then declare four new objects, all of which are `GameObject*`:

[PRE17]

In our `Game::init` function we can then create instances of the objects using their individual types:

[PRE18]

Now they can be pushed into the array of `GameObject*`:

[PRE19]

The `Game::draw` function can now look something like this:

[PRE20]

Notice that we are looping through all of our objects and calling the `draw` function. The loop does not care that some of our objects are actually `Player` or `Enemy`; it handles them in the same manner. We are accessing them through a pointer to their base class. So, to add a new type, it simply needs to be derived from `GameObject`, and the `Game` class can handle it.

*   So let's implement this for real in our framework. First we need a base class; we will stick with `GameObject`. We will have to make some changes to the class so that we can use it as a base class:

    [PRE21]

Notice that we have now prefixed our functions with the virtual keyword. The virtual keyword means that when calling this function through a pointer, it uses the definition from the type of the object itself, not the type of its pointer:

[PRE22]

In other words, this function would always call the `draw` function contained in `GameObject`, neither `Player` nor `Enemy`. We would never have the overridden behavior that we want. The virtual keyword would ensure that the `Player` and `Enemy` draw functions are called.

Now we have a base class, so let's go ahead and try it out in our `Game` class. We will start by declaring the objects in the `Game` header file:

[PRE23]

Now declare along with our `GameObject*` array:

[PRE24]

Now create and load the objects in the `init` function, then push them into the array:

[PRE25]

So far, so good; we can now create a loop that will draw our objects and another that will update them. Now let's look at the `render` and `update` functions:

[PRE26]

As you can see, this is a lot tidier and also much easier to manage. Let us derive one more class from `GameObject` just so that we nail this concept down. Create a new class called `Enemy`:

[PRE27]

We will define the functions of this class the same as `Player` with only the `update` function as an exception:

[PRE28]

Now let's add it to the game. First, we declare it as follows:

[PRE29]

Then create, load, and add to the array:

[PRE30]

We have just added a new type and it was extremely quick and simple. Run the game to see our three objects, each with their own different behavior.

![Implementing polymorphism](img/6821OT_03_02.jpg)

We have covered a lot here and have a really nice system for handling our game objects, yet we still have an issue. There is nothing stopping us from deriving a class without the `update` or `draw` functions that we are using here, or even declaring a different function and putting the `update` code in there. It is unlikely that we, as the developers, would make this mistake, but others using the framework may. What we would like is the ability to force our derived classes to have their own implementation of a function we decide upon, creating something of a blueprint that we want all of our game objects to follow. We can achieve this through the use of an abstract base class.

# Using abstract base classes

If we are to implement our design correctly, then we have to be certain that all of our derived classes have a declaration and definition for each of the functions we want to access through the base class pointer. We can ensure this by making `GameObject` an abstract base class. An abstract base class cannot be initialized itself; its purpose is to dictate the design of derived classes. This gives us reusability as we know that any object we derive from `GameObject` will immediately work in the overall scheme of the game.

An abstract base class is a class that contains at least one pure virtual function. A pure virtual function is a function that has no definition and must be implemented in any derived classes. We can make a function pure virtual by suffixing it with `=0`.

# Should we always use inheritance?

Inheritance and polymorphism are both very useful and really show off the power of object-oriented programming. However, in some circumstances, inheritance can cause more problems than it solves, and therefore, we should bear in mind a few rules of thumb when deciding whether or not to use it.

## Could the same thing be achieved with a simpler solution?

Let's say we want to make a more powerful `Enemy` object; it will have the same behavior a regular `Enemy` object will have but with more health. One possible solution would be to derive a new class `PowerEnemy` from `Enemy` and give it double health. In this solution the new class will seem extremely sparse; it will use the functionality from `Enemy` but with one different value. An easier solution would be to have a way to set the health of an `Enemy` class, whether through an accessor or in the constructor. Inheritance isn't needed at all.

## Derived classes should model the "is a" relationship

When deriving a class, it is a good idea for it to model the "is a" relationship. This means that the derived class should also be of the same type as the parent class. For example, deriving a `Player2` class from `Player` would fit the model, as `Player2` "is a" `Player`. But let's say, for example, we have a `Jetpack` class and we derive `Player` from this class to give it access to all the functionality that a `Jetpack` class has. This would not model the "is a" relationship, as a `Player` class is not a `Jetpack` class. It makes a lot more sense to say a `Player` class has a `Jetpack` class, and therefore, a `Player` class should have a member variable of type `Jetpack` with no inheritance; this is known as containment.

## Possible performance penalties

On platforms such as PC and Mac, the performance penalties of using inheritance and virtual functions are negligible. However, if you are developing for less powerful devices such as handheld consoles, phones, or embedded systems, this is something that you should take into account. If your core loop involves calling a virtual function many times per second, the performance penalties can add up.

# Putting it all together

We can now put all of this knowledge together and implement as much as we can into our framework, with reusability in mind. We have quite a bit of work to do, so let's start with our abstract base class, `GameObject`. We are going to strip out anything SDL-specific so that we can reuse this class in other SDL projects if needed. Here is our stripped down `GameObject` abstract base class:

[PRE31]

The pure virtual functions have been created, forcing any derived classes to also declare and implement them. There is also now no `load` function; the reason for this is that we don't want to have to create a new `load` function for each new project. We can be pretty sure that we will need different values when loading our objects for different games. The approach we will take here is to create a new class called `LoaderParams` and pass that into the constructor of our objects.

`LoaderParams` is simply a class that takes values into its constructor and sets them as member variables that can then be accessed to set the initial values of an object. While it may just seem that we are moving the parameters from the `load` function to somewhere else, it is a lot easier to just create a new `LoaderParams` class than to track down and alter the `load` function of all of our objects.

So here is our `LoaderParams` class:

[PRE32]

This class holds any values we need when creating our object exactly the same way as our `load` function used to do.

We have also removed the `SDL_Renderer` parameter from the `draw` function. We will instead make our `Game` class a singleton, such as `TextureManager`. So, we can add the following to our `Game` class:

[PRE33]

In the `Game.cpp`, we have to define our static instance:

[PRE34]

Let's also create a function in the header file that will return our `SDL_Renderer` object:

[PRE35]

Now that `Game` is a singleton, we are going to use it differently in our `main.cpp` file:

[PRE36]

Now when we want to access the `m_pRenderer` value from `Game`, we can use the `getRenderer` function. Now that `GameObject` is essentially empty, how do we achieve the code-sharing we originally had? We are going to derive a new generic class from `GameObject` and call it `SDLGameObject`:

[PRE37]

With this class we can create our reusable SDL code. First, we can use our new `LoaderParams` class to set our member variables:

[PRE38]

We can also use the same `draw` function as before, making use of our singleton `Game` class to get the renderer we want:

[PRE39]

`Player` and `Enemy` can now inherit from `SDLGameObject`:

[PRE40]

The `Player` class can be defined like so (the `Enemy` class is very similar):

[PRE41]

Now that everything is in place, we can go ahead and create the objects in our `Game` class and see everything in action. We won't add the objects to the header file this time; we will use a shortcut and build our objects in one line in the `init` function:

[PRE42]

Build the project. We now have everything in place to allow us to easily reuse our `Game` and `GameObject` classes.

# Summary

We have covered a lot of complex subjects in this chapter, and the concepts and ideas will take some time to sink in. We have covered the ability to easily create classes without having to rewrite a lot of similar functionality and the use of inheritance and how it allows us to share code between similar classes. We looked at polymorphism and how it can make object management a lot cleaner and reusable while abstract base classes took our inheritance knowledge up a notch by creating the blueprint we want all of our objects to follow. Finally, we put all our new knowledge into the context of our framework.