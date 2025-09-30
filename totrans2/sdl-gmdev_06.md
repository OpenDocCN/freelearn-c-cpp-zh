# Chapter 6. Data-driven Design

With the previous chapter adding the ability to create and handle game states, our framework has really begun to take shape. In this chapter, we will explore a new way to create our states and objects by removing the need to hardcode the creation of our objects at compile time. To do this we will parse through an external file, in our case an XML file, which lists all of the objects needed for our state. This will make our states generic as they can be completely different simply by loading up an alternate XML file. Taking `PlayState` as an example, when creating a new level we would need to create a new state with different objects and set up objects we want for that level. If we could instead load the objects from an external file, we could reuse the same `PlayState` and simply load the correct file depending on the current level we want. Keeping classes generic like this and loading external data to determine their state is called **Data-driven Design**.

In this chapter we will cover:

*   Loading XML files using the **TinyXML** library
*   Creating a **Distributed Factory**
*   Loading objects dynamically using the factory and an XML file
*   Parsing a state from an XML file
*   Fitting everything together into the framework

# Loading XML files

I have chosen to use XML files because they are so easy to parse. We are not going to write our own XML parser, rather we will use an open source library called TinyXML. TinyXML was written by *Lee Thomason* and is available under the zlib license from [http://sourceforge.net/projects/tinyxml/](http://sourceforge.net/projects/tinyxml/).

Once downloaded the only setup we need to do is to include a few of the files in our project:

*   `tinyxmlerror.cpp`
*   `tinyxmlparser.cpp`
*   `tinystr.cpp`
*   `tinystr.h`
*   `tinyxml.cpp`
*   `tinyxml.h`

Also, at the top of `tinyxml.h`, add this line of code:

[PRE0]

By doing this we ensure that we are using the STL versions of the TinyXML functions. We can now go through a little of how an XML file is structured. It's actually fairly simple and we will only give a brief overview to help you get up to speed with how we will use it.

## Basic XML structure

Here is a basic XML file:

[PRE1]

The first line of the file defines the format of the XML file. The second line is our `Root` element; everything else is a child of this element. The third line is the first child of the root element. Now let's look at a slightly more complicated XML file:

[PRE2]

As you can see we have now added children to the first child element. You can nest as many children as you like. But without a good structure, your XML file may become very hard to read. If we were to parse the above file, here are the steps we would take:

1.  Load the XML file.
2.  Get the root element, `<ROOT>`.
3.  Get the first child of the root element, `<ELEMENTS>`.
4.  For each child, `<ELEMENT>` of `<ELEMENTS>`, get the content.
5.  Close the file.

Another useful XML feature is the use of attributes. Here is an example:

[PRE3]

We have now stored the text we want in an attribute named `text`. When this file is parsed, we would now grab the `text` attribute for each element and store that instead of the content between the `<ELEMENT></ELEMENT>` tags. This is especially useful for us as we can use attributes to store lots of different values for our objects. So let's look at something closer to what we will use in our game:

[PRE4]

This is slightly more complex. We define each state in its own element and within this element we have objects and textures with various attributes. These attributes can be loaded in to create the state.

With this knowledge of XML you can easily create your own file structures if what we cover within this book is not to your needs.

# Implementing Object Factories

We are now armed with a little XML knowledge but before we move forward, we are going to take a look at Object Factories. An object factory is a class that is tasked with the creation of our objects. Essentially, we tell the factory the object we would like it to create and it goes ahead and creates a new instance of that object and then returns it. We can start by looking at a rudimentary implementation:

[PRE5]

This function is very simple. We pass in an ID for the object and the factory uses a big switch statement to look it up and return the correct object. Not a terrible solution but also not a particularly good one, as the factory will need to know about each type it needs to create and maintaining the switch statement for many different objects would be extremely tedious. Just as when we covered looping through game objects in [Chapter 3](ch03.html "Chapter 3. Working with Game Objects"), *Working with Game Objects*, we want this factory not to care about which type we ask for. It shouldn't need to know all of the specific types we want it to create. Luckily this is something that we can definitely achieve.

## Using Distributed Factories

Through the use of Distributed Factories we can make a generic object factory that will create any of our types. Distributed factories allow us to dynamically maintain the types of objects we want our factory to create, rather than hard code them into a function (like in the preceding simple example). The approach we will take is to have the factory contain `std::map` that maps a string (the type of our object) to a small class called `Creator` whose only purpose is the creation of a specific object. We will register a new type with the factory using a function that takes a string (the ID) and a `Creator` class and adds them to the factory's map. We are going to start with the base class for all the `Creator` types. Create `GameObjectFactory.h` and declare this class at the top of the file.

[PRE6]

We can now go ahead and create the rest of our factory and then go through it piece by piece.

[PRE7]

This is quite a small class but it is actually very powerful. We will cover each part separately starting with `std::map m_creators`.

[PRE8]

This map holds the important elements of our factory, the functions of the class essentially either add or remove from this map. This becomes apparent when we look at the `registerType` function:

[PRE9]

This function takes the ID we want to associate the object type with (as a string), and the creator object for that class. The function then attempts to find the type using the `std::mapfind` function:

[PRE10]

If the type is found then it is already registered. The function then deletes the passed in pointer and returns `false`:

[PRE11]

If the type is not already registered then it can be assigned to the map and then `true` is returned:

[PRE12]

As you can see, the `registerType` function is actually very simple; it is just a way to add types to the map. The `create` function is very similar:

[PRE13]

The function looks for the type in the same way as `registerType` does, but this time it checks whether the type was not found (as opposed to found). If the type is not found we return `0`, and if the type is found then we use the `Creator` object for that type to return a new instance of it as a pointer to `GameObject`.

It is worth noting that the `GameObjectFactory` class should probably be a singleton. We won't cover how to make it a singleton as this has been covered in the previous chapters. Try implementing it yourself or see how it is implemented in the source code download.

# Fitting the factory into the framework

With our factory now in place, we can start altering our `GameObject` classes to use it. Our first step is to ensure that we have a `Creator` class for each of our objects. Here is one for `Player`:

[PRE14]

This can be added to the bottom of the `Player.h` file. Any object we want the factory to create must have its own `Creator` implementation. Another addition we must make is to move `LoaderParams` from the constructor to their own function called `load`. This stops the need for us to pass the `LoaderParams` object to the factory itself. We will put the `load` function into the `GameObject` base class, as we want every object to have one.

[PRE15]

Each of our derived classes will now need to implement this `load` function. The `SDLGameObject` class will now look like this:

[PRE16]

Our objects that derive from `SDLGameObject` can use this `load` function as well; for example, here is the `Player::load` function:

[PRE17]

This may seem a bit pointless but it actually saves us having to pass through `LoaderParams` everywhere. Without it, we would need to pass `LoaderParams` through the factory's `create` function which would then in turn pass it through to the `Creator` object. We have eliminated the need for this by having a specific function that handles parsing our loading values. This will make more sense once we start parsing our states from a file.

We have another issue which needs rectifying; we have two classes with extra parameters in their constructors (`MenuButton` and `AnimatedGraphic`). Both classes take an extra parameter as well as `LoaderParams`. To combat this we will add these values to `LoaderParams` and give them default values.

[PRE18]

In other words, if the parameter is not passed in, then the default values will be used (0 in both cases). Rather than passing in a function pointer as `MenuButton` did, we are using `callbackID` to decide which callback function to use within a state. We can now start using our factory and parsing our states from an XML file.

# Parsing states from an XML file

The file we will be parsing is the following (`test.xml` in source code downloads):

[PRE19]

We are going to create a new class that parses our states for us called `StateParser`. The `StateParser` class has no data members, it is to be used once in the `onEnter` function of a state and then discarded when it goes out of scope. Create a `StateParser.h` file and add the following code:

[PRE20]

We have three functions here, one public and two private. The `parseState` function takes the filename of an XML file as a parameter, along with the current `stateID` value and a pointer to `std::vector` of `GameObject*` for that state. The `StateParser.cpp` file will define this function:

[PRE21]

There is a lot of code in this function so it is worth covering in some depth. We will note the corresponding part of the XML file, along with the code we use, to obtain it. The first part of the function attempts to load the XML file that is passed into the function:

[PRE22]

It displays an error to let you know what happened if the XML loading fails. Next we must grab the root node of the XML file:

[PRE23]

The rest of the nodes in the file are all children of this root node. We must now get the root node of the state we are currently parsing; let's say we are looking for `MENU`:

[PRE24]

This piece of code goes through each direct child of the root node and checks if its name is the same as `stateID`. Once it finds the correct node it assigns it to `pStateRoot`. We now have the root node of the state we want to parse.

[PRE25]

Now that we have a pointer to the root node of our state we can start to grab values from it. First we want to load the textures from the file so we look for the `<TEXTURE>` node using the children of the `pStateRoot` object we found before:

[PRE26]

Once the `<TEXTURE>` node is found, we can pass it into the private `parseTextures` function (which we will cover a little later).

[PRE27]

The function then moves onto searching for the `<OBJECT>` node and, once found, it passes it into the private `parseObjects` function. We also pass in the `pObjects` parameter:

[PRE28]

At this point our state has been parsed. We can now cover the two private functions, starting with `parseTextures`.

[PRE29]

This function gets the `filename` and `ID` attributes from each of the texture values in this part of the XML:

[PRE30]

It then adds them to `TextureManager`.

[PRE31]

The `parseObjects` function is quite a bit more complicated. It creates objects using our `GameObjectFactory` function and reads from this part of the XML file:

[PRE32]

The `parseObjects` function is defined like so:

[PRE33]

First we get any values we need from the current node. Since XML files are pure text, we cannot simply grab ints or floats from the file. TinyXML has functions with which you can pass in the value you want to be set and the attribute name. For example:

[PRE34]

This sets the variable `x` to the value contained within attribute `"x"`. Next comes the creation of a `GameObject` ***** class using the factory.

[PRE35]

We pass in the value from the `type` attribute and use that to create the correct object from the factory. After this we must use the `load` function of `GameObject` to set our desired values using the values loaded from the XML file.

[PRE36]

And finally we push `pGameObject` into the `pObjects` array, which is actually a pointer to the current state's object vector.

[PRE37]

# Loading the menu state from an XML file

We now have most of our state loading code in place and can make use of this in the `MenuState` class. First we must do a little legwork and set up a new way of assigning the callbacks to our `MenuButton` objects, since this is not something we could pass in from an XML file. The approach we will take is to give any object that wants to make use of a callback an attribute named `callbackID` in the XML file. Other objects do not need this value and `LoaderParams` will use the default value of `0`. The `MenuButton` class will make use of this value and pull it from its `LoaderParams`, like so:

[PRE38]

The `MenuButton` class will also need two other functions, one to set the callback function and another to return its callback ID:

[PRE39]

Next we must create a function to set callbacks. Any state that uses objects with callbacks will need an implementation of this function. The most likely states to have callbacks are menu states, so we will rename our `MenuState` class to `MainMenuState` and make `MenuState` an abstract class that extends from `GameState`. The class will declare a function that sets the callbacks for any items that need it and it will also have a vector of the `Callback` objects as a member; this will be used within the `setCallbacks` function for each state.

[PRE40]

The `MainMenuState` class (previously `MenuState`) will now derive from this `MenuState` class.

[PRE41]

Because `MainMenuState` now derives from `MenuState`, it must of course declare and define the `setCallbacks` function. We are now ready to use our state parsing to load the `MainMenuState` class. Our `onEnter` function will now look like this:

[PRE42]

We create a state parser and then use it to parse the current state. We push any callbacks into the `m_callbacks` array inherited from `MenuState`. Now we need to define the `setCallbacks` function:

[PRE43]

We use `dynamic_cast` to check whether the object is a `MenuButton` type; if it is then we do the actual cast and then use the objects `callbackID` as the index into the `callbacks` vector and assign the correct function. While this method of assigning callbacks could be seen as not very extendable and could possibly be better implemented, it does have a redeeming feature; it allows us to keep our callbacks inside the state they will need to be called from. This means that we won't need a huge header file with all of the callbacks in.

One last alteration we need is to add a list of texture IDs to each state so that we can clear all of the textures that were loaded for that state. Open up `GameState.h` and we will add a `protected` variable.

[PRE44]

We will pass this into the state parser in `onEnter` and then we can clear any used textures in the `onExit` function of each state, like so:

[PRE45]

Before we start running the game we need to register our `MenuButton` type with the `GameObjectFactory`. Open up `Game.cpp` and in the `Game::init` function we can register the type.

[PRE46]

We can now run the game and see our fully data-driven `MainMenuState`.

# Loading other states from an XML file

Our `MainMenuState` class now loads from an XML file. We need to make our other states do the same. We will only cover the code that has changed, so assume that everything else has remained the same when following through this section.

## Loading the play state

We will start with `PlayState.cpp` and its `onEnter` function.

[PRE47]

We must also add the new texture clearing code that we had in `MainMenuState` to the `onExit` function.

[PRE48]

These are the only alterations that we will need to do here but we must also update our XML file to have something to load in `PlayState`.

[PRE49]

Our `Enemy` object will now need to set its initial velocity in its load function rather than the constructor, otherwise the `load` function would override it.

[PRE50]

Finally we must register these objects with the factory. We can do this in the `Game::init` function just like the `MenuButton` object.

[PRE51]

## Loading the pause state

Our `PauseState` class must now inherit from `MenuState` as we want it to contain callbacks. We must update the `PauseState.h` file to first inherit from `MenuState`.

[PRE52]

We must also declare the `setCallbacks` function.

[PRE53]

Now we must update the `PauseState.cpp` file, starting with the `onEnter` function.

[PRE54]

The `setCallbacks` function is exactly like `MainMenuState`.

[PRE55]

Finally we must add the texture clearing code to `onExit`.

[PRE56]

And then update our XML file to include this state.

[PRE57]

## Loading the game over state

Our final state is `GameOverState`. Again this will be very similar to other states and we will only cover what has changed. Since we want `GameOverState` to handle callbacks it will now inherit from `MenuState`.

[PRE58]

We will then declare the `setCallbacks` function.

[PRE59]

The `onEnter` function should be looking very familiar now.

[PRE60]

The texture clearing method is the same as in the previous states, so we will leave you to implement that yourself. In fact `onExit` is looking so similar between states that it would be a good idea to make a generic implementation for it in `GameState` and just use that; again we will leave that to you.

You may have noticed the similarity between the `onEnter` functions. It would be great to have a default `onEnter` implementation but, unfortunately, due to the need to specify different callback functions, our callback implementation will not allow this and this is one of its main flaws.

Our `AnimatedGraphic` class will now need to grab the `animSpeed` value from `LoaderParams` in its `load` function.

[PRE61]

We will also have to register this type with `GameObjectFactory`.

[PRE62]

And finally we can update the XML file to include this state:

[PRE63]

We now have all of our states loading from the XML file and one of the biggest benefits of this is that you do not have to recompile the game when you change a value. Go ahead and change the XML file to move positions or even use different textures for objects; if the XML is saved then you can just run the game again and it will use the new values. This is a huge time saver for us and gives us complete control over a state without the need to recompile our game.

# Summary

Loading data from external files is an extremely useful tool in programming games. This chapter enabled our game to do this and applied it to all of our existing states. We also covered how the use of factories enabled us to create objects dynamically at runtime. The next chapter will cover even more data-driven design as well as tile maps so that we can really decouple our game and allow it to use external sources rather than hardcoded values.