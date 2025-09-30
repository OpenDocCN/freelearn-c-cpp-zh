# Chapter 1. Under the Hood - Setting up the Backend

# Introduction

What is the heart of any given piece of software? The answer to this question becomes apparent gradually while building a full-scale project, which can be a daunting task to undertake, especially when starting from scratch. It’s the design and capability of the back-end that either drives a game forward with full force by utilizing its power, or crashes it into obscurity through unrealized potential. Here, we’re going to be talking about that very foundation that keeps any given project up and standing.

In this chapter, we're going to be covering the following topics:

*   Utility functions and filesystem specifics for Windows and Linux operating systems
*   The basics of the entity component system pattern
*   Window, event, and resource management techniques
*   Creating and maintaining application states
*   Graphical user interface basics
*   Essentials for the 2D RPG game project

There's a lot to cover, so let's not waste any time!

# Pacing and source code examples

All of the systems we're going to be talking about here could have entire volumes dedicated to them. Since time, as well as paper, is limited, we're only going to be briefly reviewing their very basics, which is just enough to feel comfortable with the rest of the information presented here.

### Note

Keep in mind that, although we won't be going into too much detail in this particular chapter, the code that accompanies this book is a great resource to look through and experiment with for more detail and familiarity. It's greatly recommended to review it while reading this chapter in order to get a full grasp of it.

# Common utility functions

Let's start by taking a look at a common function, which is going to be used to determine the full absolute path to the directory our executable is in. Unfortunately, there is no unified way of doing this across all platforms, so we're going to have to implement a version of this utility function for each one, starting with Windows:

[PRE0]

First, we check if the `RUNNING_WINDOWS` macro is defined. This is the basic technique that can be used to actually let the rest of the code base know which OS it's running on. Next, another definition is made, specifically for the Windows header files we're including. It greatly reduces the number of other headers that get included in the process.

With all of the necessary headers for the Windows OS included, let us take a look at how the actual function can be implemented:

[PRE1]

First, we obtain the handle to the process that was created by our executable file. After the temporary path buffer is constructed and filled with the path string, the name, and extension of our executable is removed. We top it off by adding a trailing slash to the end of the path and returning it as a `std::string`.

It will also come in handy to have a way of obtaining a list of files inside a specified directory:

[PRE2]

Just like the directory function, this is specific to the Windows OS. It returns a vector of strings that represent file names and extensions. Once one is constructed, a path string is cobbled together. The `l_search` argument is provided with a default value, in case one is not specified. All files are listed by default.

After creating a structure that will hold our search data, we pass it to another Windows specific function that will find the very first file inside a directory. The rest of the work is done inside a `do-while` loop, which checks if the located item isn't in fact a directory. The appropriate items are then pushed into a vector, which gets returned later on.

## The Linux version

As mentioned previously, both of the preceding functions are only functional on Windows. In order to add support for systems running Linux-based OSes, we're going to need to implement them differently. Let's start by including proper header files:

[PRE3]

As luck would have it, Linux does offer a single-call solution to finding exactly where our executable is located:

[PRE4]

Note that we're still adding a trailing slash to the end.

Obtaining a file list of a specific directory is slightly more complicated this time around:

[PRE5]

We start off in the same fashion as before, by creating a vector of strings. A pointer to the directory stream is then obtained through the `opendir()` function. Provided it isn't `NULL`, we begin modifying the search string. Unlike the fancier Windows alternative, we can't just pass a search string into a function and let the OS do all of the matching. In this case, it falls more under the category of matching a specific search string inside a filename that gets returned, so star symbols that mean anything need to be trimmed out.

Next, we utilize the `readdir()` function inside a `while` loop that's going to return a pointer to directory entry structures one by one. We also want to exclude any directories from the file list, so the entry's type is checked for not being equal to `DT_DIR`.

Finally, the string matching begins. Presuming we're not just looking for any file with any extension (represented by `"*.*"`), the entry's name will be compared to the search string by length first. If the length of the string we're searching is longer than the filename itself, it's safe to assume we don't have a match. Otherwise, the search string is analyzed again to determine whether the filename is important for a positive match. Its first character being a period would denote that it isn't, so the file name's ending segment of the same length as the search string is compared to the search string itself. If, however, the name is important, we simply search the filename for the search string.

Once the procedure is complete, the directory is closed and the vector of strings representing files is returned.

## Other miscellaneous helper functions

Sometimes, as text files are being read, it's nice to grab a string that includes spaces while still maintaining a whitespace delimiter. In cases like that, we can use quotes along with this special function that helps us read the entire quoted segment from a whitespace delimited file:

[PRE6]

The first segment of the stream is fed into the argument string. If it does indeed start with a double quote, a `while` loop is initiated to append to said string until it ends with another double quote, or until the stream reaches the end. Lastly, all double quotes from the string are erased, giving us the final result.

Interpolation is another useful tool in a programmer's belt. Imagine having two different values of something at two different points in time, and then wanting to predict what the value would be somewhere in between those two time frames. This simple calculation makes that possible:

[PRE7]

Next, let's take a look at a few functions that can help us center instances of `sf::Text` better:

[PRE8]

Working with SFML text can be tricky sometimes, especially when centering it is of paramount importance. Some characters, depending on the font and other different attributes, can actually exceed the height of the bounding box that surrounds the `sf::Text` instance. To combat that, the first function iterates through every single character of a specific text instance and fetches the font glyph used to represent it. Its height is then checked and kept track of, so that the maximum height of the entire text can be determined and returned.

The second function can be used for setting the absolute center of a `sf::Text` instance as its origin, in order to achieve perfect results. After its local bounding box is obtained and the maximum height is calculated, this information is used to move the original point of our text to its center.

## Generating random numbers

Most games out there rely on some level of randomness. While it may be tempting to simply use the classical approach of `rand()`, it can only take you so far. Generating random negative or floating point numbers isn't straightforward, to say the least, plus it has a very lousy range. Luckily, newer versions of C++ provide the answer in the form of uniform distributions and random number engines:

[PRE9]

First, note the `include` statements. The `random` library provides us with everything we need as far as number generation goes. On top of that, we're also going to be using SFML's mutexes and locks, in order to prevent a huge mess in case our code is being accessed by several separate threads.

The `std::random_device` class is a random number generator that is used to seed the engine, which will be used for further generations. The engine itself is based on the *Marsenne Twister* algorithm, and produces high-quality random *unsigned integers* that can later be filtered through a **uniform distribution** object in order to obtain a number that falls within a specific range. Ideally, since it is quite expensive to keep constructing and destroying these objects, we're going to want to keep a single copy of this class around. For this very reason, we have integer and float distributions together in the same class.

For convenience, the parenthesis operators are overloaded to take in ranges of numbers of both *integer* and *floating point* types. They invoke the `Generate` method, which is also overloaded to handle both data types:

[PRE10]

Before generation can begin, we must establish a lock in order to be thread-safe. Because the order of `l_min` and `l_max` values matters, we must check if the provided values aren't in reverse, and swap them if they are. Also, the uniform distribution object has to be reconstructed if a different range needs to be used, so a check for that is in place as well. Finally, after all of that trouble, we're ready to return the random number by utilizing the parenthesis operator of a distribution, to which the engine instance is fed in.

# Service locator pattern

Often, one or more of our classes will need access to another part of our code base. Usually, it's not a major issue. All you would have to do is pass a pointer or two around, or maybe store them once as data members of the class in need. However, as the amount of code grows, relationships between classes get more and more complex. Dependencies can increase to a point, where a specific class will have more arguments/setters than actual methods. For convenience's sake, sometimes it's better to pass around a single pointer/reference instead of ten. This is where the **service locator** pattern comes in:

[PRE11]

As you can see, it's just a `struct` with multiple pointers to the core classes of our project. All of those classes are forward-declared in order to avoid unnecessary `include` statements, and thus a bloated compilation process.

# Entity component system core

Let's get to the essence of how our game entities are going to be represented. In order to achieve highest maintainability and code compartmentalization, it's best to use composition. The entity component system allows just that. For the sake of keeping this short and sweet, we're not going to be delving too deep into the implementation. This is simply a quick overview for the sake of being familiar with the code that will be used down the line.

The ECS pattern consists of three cornerstones that make it possible: entities, components, and systems. An entity, ideally, is simply an identifier, as basic as an integer. Components are containers of data that have next to no logic inside them. There would be multiple types of components, such as position, movable, drawable, and so on, that don't really mean much by themselves, but when composed, will form complex entities. Such composition would make it incredibly easy to save the state of any entity at any given time.

There are many ways to implement components. One of them is simply having a base component class, and inheriting from it:

[PRE12]

The `Component` type is simply an *enum class* that lists different types of components we can have in a project. In addition to that, this base class also offers a means of filling in component data from a string stream, in order to load them more easily when files are being read.

In order to properly manage sets of components that belong to entities, we would need some sort of manager class:

[PRE13]

As you can see, this is a fairly basic approach at managing these sets of data we call entities. The `EntityId` data type is simply a type definition for an **unsigned integer**. Creation of components happens by utilizing a factory pattern, lambdas and templates. This class is also responsible for loading entities from files that may look a little like this:

[PRE14]

The `Attributes` field is a bit mask, the value of which is used to figure out which component types an entity has. The actual component data is stored in this file as well, and later loaded through the `ReadIn` method of our component base class.

The last piece of the puzzle in ECS design is systems. This is where all of the logic happens. Just like components, there can be many types of systems responsible for collisions, rendering, movement, and so on. Each system must inherit from the system's base class and implement all of the pure virtual methods:

[PRE15]

Systems have signatures of components they use, as well as a list of entities that meet the requirements of said signatures. When an entity is being modified by the addition or removal of a component, every system runs a check on it in order to add it to or remove it from itself. Note the inheritance from the `Observer` class. This is another pattern that aids in communication between entities and systems.

An `Observer` class by itself is simply an interface with one purely virtual method that must be implemented by all derivatives:

[PRE16]

It utilizes messages that get sent to all observers of a specific target. How the derivative of this class reacts to the message is completely dependent on what it is.

Systems, which come in all shapes and sizes, need to be managed just as entities do. For that, we have another manager class:

[PRE17]

This too utilizes the factory pattern, in that types of different classes are *registered* by using templates and lambdas, so that they can be constructed later, simply by using a `System` data type, which is an `enum class`. Starting to see the pattern?

The system manager owns a data member of type `MessageHandler`. This is another part of the observer pattern. Let us take a look at what it does:

[PRE18]

Message handlers are simply collections of `Communicator` objects, as shown here:

[PRE19]

Each possible type of `EntityMessage`, which is just another *enum class*, is tied to a communicator that is responsible for sending out a message to all of its observers. Observers can subscribe to or unsubscribe from a specific message type. If they are subscribed to said type, they will receive the message when the `Dispatch` method is invoked.

The `Communicator` class itself is fairly simple:

[PRE20]

As you can gather, it supports the addition and removal of observers, and offers a way to broadcast a message to all of them. The actual container of observers is simply a vector of pointers:

[PRE21]

# Resource management

Another vital part of larger projects is an efficient way of managing resources. Since we're going to have several types of resources, such as textures, fonts, and sounds, it would make sense to have separate managers for all of them. It's time for a base class:

[PRE22]

The idea behind this particular resource management system is certain segments of code *requiring* and later *releasing* a certain resource identifier. The first time a resource is required it will be loaded into memory and kept there. Every time it's required after that will simply increment an integer that gets stored with it. The integer represents how many instances of code rely on this resource being loaded. Once they are done using the resource, it begins being released, which brings the counter down each time. When it reaches zero, the resource is removed from memory.

It's fair to point out that our resource manager base class utilizes the **Curiously Recurring Template Pattern** for setting up the resource instances after they're created. As manager classes don't really need to be stored together in the same container anywhere, static polymorphism makes a lot more sense than using virtual methods. Since textures, fonts, and sounds may be loaded in different ways, each subsequent manager must implement their own version of the `Load` method, like so:

[PRE23]

Each single manager also has its own file, listing the relationships between names of resources and their paths. For textures, it can look something like this:

[PRE24]

It simply avoids the need to pass around paths and filenames, by instead relating a name to each resource.

# Windows system

There's a lot that goes on behind the scenes when it comes to dealing with open windows. Everything from window dimensions and titles to keeping track of and dealing with special events is centralized within a designated window class:

[PRE25]

Note the two highlighted methods. They will be used as call-backs in the event manager we'll discuss in the near future. Also note the return method for an object type `Renderer`. It’s a utility class that simply invokes the `.draw` call on a `RenderWindow`, thus localizing it and making it much easier to use shaders. More information on that will be revealed in [Chapter 6](ch06.html "Chapter 6. Adding Some Finishing Touches - Using Shaders"), *Adding Some Finishing Touches – Using Shaders*.

# Application states

Another important aspect of a more complex application is keeping track of and managing its states. Whether the player is in the thick of the game, or simply browsing through the main menu, we want it to be handled seamlessly, and more importantly, be self-contained. We can start this by first defining different types of states we'll be dealing with:

[PRE26]

For seamless integration, we want each state to behave in a predictable manner. This means that a state has to adhere to an interface we provide:

[PRE27]

Every state in the game will have its own view that it can alter. In addition to that, it is given the hooks to implement logic for various different scenarios, such as the state's creation, destruction, activation, deactivation, updating, and rendering. Lastly, it enables the possibility of being blended with other states during updating and rendering, by providing the `m_transparent` and `m_transcendent` flags.

Managing these states is pretty straightforward:

[PRE28]

The `StateManager` class is one of the few classes in the project that utilizes the shared context, since the states themselves may need access to any part of the code base. It also uses the factory pattern to dynamically create any state that is bound to a state type during runtime.

In order to keep things simple, we're going to be treating the loading state as a special case, and only allow one instance of it to be alive at all times. Loading might happen during the transition of any state, so it only makes sense.

One final thing that's worth noting about the state manager is it's keeping a list of state dependants. It's simply an STL container of classes that inherit from this interface:

[PRE29]

Because classes that deal with things such as sounds, GUI elements, or entity management need to support different states, they must also define what happens inside them as a state is created, changed, or removed, in order to properly allocate/de-allocate resources, stop updating data that is not in the same state, and so on.

## Loading state

So, how exactly are we going to implement this loading state? Well, for flexibility and easy progress tracking by means of rendering fancy loading bars, threads are going to prove invaluable. Data that needs to be loaded into memory can be loaded in a separate thread, while the loading state itself continues to get updated and rendered in order to show us that things are indeed happening. Just knowing that the application did not hang on us should create a warm and fuzzy feeling.

First, let us implement the very basics of this system by providing an interface any threaded worker can use:

[PRE30]

It has its own thread, which is bound to the pure virtual method called `Work`. The thread is launched whenever the `Begin()` method is invoked. In order to protect the data from being accessed from multiple threads at once, a `sf::Mutex` class is used by creating a lock during sensitive calls. Everything else within this very basic class is simply there to provide information to the outside world about the worker’s state.

## File loader

With threads out of the way, we can focus on actually loading some files now. This method is going to focus on working with text files. However, using binary formats should work in pretty much the exact same way, minus all the text processing.

Let's take a look at the base class for any file loading class we can think of:

[PRE31]

It's a distinct possibility that two or more files may need to be loaded at some point. The `FileLoader` class keeps track of all of the paths that get added to it, along with a number that represents the number of lines within that file. This is useful for determining the amount of progress that has been made while loading. In addition to the line count for each individual file, a total line count is also kept track of.

This class provides a single purely virtual method, called `ProcessLine`. It will be the way derivatives can define exactly how the file is loaded and processed.

First, let us get the basic stuff out of the way:

[PRE32]

The `ResetForNextFile()` virtual method is optional to implement, but can be used in order to clear the state of some internal data that needs to exist while a file is being loaded. Since file loaders that implement this class will only have the ability to process one line at a time inside a single method, any temporary data that would normally be stored as a local variable within that method would instead need to go somewhere else. This is why we must make sure that there is actually a way to know when we're done with one file and start loading another, as well as to perform some sort of action, if necessary.

### Note

Note the mutex locks in the two getter methods above. They’re there to make sure those variables aren’t written to and read from at the same time.

Now, let's get into the code that is going to be executed in a different thread:

[PRE33]

A private method for counting all the lines in whatever files are about to be loaded is called first. If, for any reason, the total line count is zero, there is no purpose in proceeding, so the `Worker::Done()` method is invoked just before a return. This little bit of code is really easy to forget, but is extremely important in order for this to work. All it does is set the `m_done` flag of the `Worker` base class to `true`, which lets outside code know that the process is finished. Since there is currently no way to check if an SFML thread is actually finished, this is pretty much the only option.

We begin looping through different files that need to get loaded and invoke the reset method before work begins. Note the lack of checking as we're attempting to open a file. This will be explained when we cover the next method.

As each line of the file is being read, it's important to make sure that all the line count information is updated. A temporary lock for the current thread is established, in order to prevent two threads from accessing the line count as its modified. In addition to that, lines that start with a pipe symbol are excluded, since this is our standard comment pragma.

Finally, a `stringstream` object is constructed for the current line, and passed into the `ProcessLine()` method. For extra points, it returns a *boolean* value that can signal an error and stop the current file from being processed any further. If that happens, the remaining lines within that specific file are added to the total count, and the loop is broken.

The final piece of the puzzle is this chunk of code, responsible for verifying file validity and determining the amount of work ahead of us:

[PRE34]

After initial zero values for line counts are set up, all added paths are iterated over and checked. We first trim out any paths that are empty. Each path is then attempted to be opened, and erased if that operation fails. Finally, in order to achieve accurate results, the file input stream is ordered to ignore empty lines. After a lock is established, `std::count` is used to count the amount of lines in a file. That number is then added to the amount of total lines we have, the path iterator is advanced, and the file is properly closed.

Since this method eliminates files that were either non-existent or unable to be opened, there is no reason to check for that again anywhere else.

## Implementing the loading state

Everything is now in place in order for us to successfully implement the loading state:

[PRE35]

The state itself will keep a vector of pointers to different file loader classes, which will have lists of their own files respectively. It also provides a way for these objects to be added. Also, note the `Proceed()` method. This is another call-back that will be used in the event manager we're about to cover soon.

For the visual portion, we will be using the bare essentials of graphics: a bit of text for the progress percentage, and a rectangle shape that represents a loading bar.

Let's take a look at all of the setup this class will do once it's constructed:

[PRE36]

First, a font manager is obtained through the shared context. The font with a name `"Main"` is required and used to set up the text instance. After all of the visual bits are set up, the event manager is used to register a call-back for the loading state. This will be covered soon, but it's quite easy to deduce what's happening by simply looking at the arguments. Whenever the spacebar is pressed, the `Proceed` method of the `State_Loading` class is going to be invoked. The actual instance of the class is passed in as the last argument.

Remember that, by design, the resources we require must also be released. A perfect place to do that for the loading state is exactly as it is destroyed:

[PRE37]

In addition to the font being released, the call-back for the spacebar is also removed.

Next, let us actually write some code that's going to bring the pieces together into a cohesive, functional whole:

[PRE38]

The first check is used to determine if all of the file loaders have been removed from the vector due to finishing. The `m_manualContinue` flag is used to let the loading state know if it should wait for the spacebar to be pressed, or if it should just dispel itself automatically. If, however, we still have some loaders in the vector, the top one is checked for having concluded its work. Given that's the case, the loader is popped and the vector is checked again for being empty, which would require us to update the loading text to represent completion.

To keep this process fully automated, we need to make sure that after the top file loader is removed, the next one is started, which is where the following check comes in. Finally, the progress percentage is calculated, and the loading text is updated to represent that value, just before the loading bar's size is adjusted to visually aid us.

Drawing is going to be extremely straightforward for this state:

[PRE39]

The render window is first obtained through the shared context, and then used to draw the text and rectangle shape that represent the loading bar together.

The `Proceed` call-back method is equally straightforward:

[PRE40]

It has to make a check first, to make sure that we don't switch states before all the work is through. If that's not the case, the state manager is used to switch to a state that was created **before** the loading commenced.

All of the other loading state logic pretty much consists of single lines of code for each method:

[PRE41]

Although this looks fairly simple, the `Activate()` method holds a fairly important role. Since the loading state is treated as a special case here, one thing has to be kept in mind: it is *never* going to be removed before the application is closed. This means that every time we want to use it again, some things have to be reset. In this case, it's the `m_originalWork` data member, that's simply the count of all the loader classes. This number is used to calculate the progress percentage accurately, and the best place to reset it is inside the method, which gets called every time the state is activated again.

# Managing application events

Event management is one of the cornerstones that provide us with fluid control experience. Any key presses, window changes, or even custom events created by the GUI system we'll be covering later are going to be processed and handled by this system. In order to effectively unify event information coming from different sources, we first must unify their types by enumerating them correctly:

[PRE42]

SFML events come first, since they are the only ones following a strict enumeration scheme. They are then followed by the live SFML input types and four GUI events. We also enumerate event information types, which are going to be used inside this structure:

[PRE43]

Because we care about more than just the event type that took place, there needs to be a good way of storing additional data that comes with it. C++11's unrestricted union is a perfect candidate for that. The only downside is that now we have to worry about manually managing the data inside the union, which comes complete with data allocations and direct invocation of destructors.

As event call-backs are being invoked, it's a good idea to provide them with the actual event information. Because it's possible to construct more complex requirements for specific call-backs, we can't get away with unions this time. Any possible information that may be relevant needs to be stored, and that's precisely what is done here:

[PRE44]

This structure is filled with every single bit of information that is available as the events are processed, and then passed as an argument to the call-back that gets invoked. It also provides a `Clear()` method, because instead of being created only for the time during the call-back, it lives inside the binding structure:

[PRE45]

A binding is what actually allows events to be grouped together in order to form more complex requirements. Think of it in terms of multiple keys needing to be pressed at once in order to perform an action, such as *Ctrl* + *C* for copying a piece of text. A binding for that type of situation would have two events it's waiting for: the *Ctrl* key and the *C* key.

## Event manager interface

With all of the key pieces being covered, all that's left is actually managing everything properly. Let's start with some type definitions:

[PRE46]

All bindings are attached to specific names that get loaded from a `keys.cfg` file when the application is started. It follows a basic format like this:

[PRE47]

Of course these are very basic examples. More complex bindings would have multiple events separated by white spaces.

Call-backs are also stored in an *unordered map*, as well as tied to the name of a binding that they're watching. The actual call-back containers are then grouped by state, in order to avoid multiple functions/methods getting called when similar keys are pressed. As you can imagine, the event manager is going to be inheriting from a `StateDependent` class for this very reason:

[PRE48]

Once again, this is quite simple. Since this is a state-dependent class, it needs to implement the `ChangeState()` and `RemoveState()` methods. It also keeps track of when the window focus is obtained/lost, in order to avoid polling events of minimized/unfocused windows. Two versions of `AddCallback` are provided: one for a specified state, and one for the current state. Separate `HandleEvent()` methods are also available for every event type supported. So far, we only have two: SFML events, and GUI events. The latter is going to be used in the upcoming section.

# Use of graphical user interfaces

A friendly way of interfacing with applications in a day and age where computers are basically a necessity inside every household is a must. The entire subject of GUIs could fill multiple books by itself, so for the sake of keeping this simple, we are only going to scratch the surface of what we have to work with:

[PRE49]

Interface management, quite predictably, is also dependent on application states. The interfaces themselves are also assigned names, which is how they are loaded and stored. Mouse input, as well as text enter events, are both utilized in making the GUI system work, which is why this class actually uses the event manager and registers three call-backs with it. Not unlike other classes we have discussed, it also uses the factory method, in order to be able to dynamically create different types of elements that populate our interfaces.

Interfaces are described as groups of elements, like so:

[PRE50]

Each element also supports styles for the three different states it can be in: neutral, hovered, and clicked. A single style file describes what an element would look like under all of these conditions:

[PRE51]

The `Neutral` style serves as a base for the other two, which is why they only define attributes that are different from it. Using this model, interfaces of great complexity can be constructed and customized to do almost anything.

# Representing a 2D map

Maps are another crucial part of having a decently complex game. For our purposes, we're going to be representing 2D maps that support different layers in order to fake 3D depth:

[PRE52]

As you can see, this class is actually inheriting from the `FileLoader`, which we covered earlier. It also supports something that's referred to as `MapLoadee*`, which are simply classes that will store certain data inside map files, and need to be notified when such data is encountered during the loading process. It's simply an interface that they have to implement:

[PRE53]

The map files themselves are fairly straightforward:

[PRE54]

A good candidate for a `MapLoadee` here would be a class that handles entities being spawned. The two entity lines would be directly handled by it, which creates a nice level of separation between codes that shouldn't really overlap.

# Sprite system

Since we're working on a 2D game, the most likely candidate for the way graphics are going to be done is a sprite sheet. Unifying the way sprite sheet cropping and animations are handled is key to not only minimizing code, but also creating a simple, neat interface that's easy to interact with. Let us take a look at how that can be done:

[PRE55]

The `SpriteSheet` class itself isn't really that complex. It offers helper methods for cropping the sheet down to a specific rectangle, altering the stored direction, defining different attributes, such as spacing, padding, and so on, and manipulating the animation data.

Animations are stored in this class by name:

[PRE56]

The interface of an animation class looks like this:

[PRE57]

First, the `Frame` data type is simply a type definition of an integer. This class keeps track of all necessary animation data, and even provides a way to set up specific frame ranges (also referred to as actions), which can be used for something such as an entity only *attacking* something if the attack animation is within that specific action range.

The obvious thing about this class is that it does not represent any single type of animation, but rather all the common elements of every type. This is why three different purely virtual methods are provided, so that different types of animation can define how the frame step is handled, define the specific method, the location of cropping, and the exact process of the animation being loaded from a file. This helps us separate directional animations, where every row represents a character facing a different way, from simple, sequential animations of frames following each other in a linear order.

# Sound system

Last, but definitely not least, the sound system deserves a brief overview. It probably would be a surprise to nobody at this point to learn that sounds are also reliant upon application states, which is why we're inheriting from `StateDependent` again:

[PRE58]

The `AudioManager` class is responsible for managing auditory resources, in the same way textures and fonts are managed elsewhere. One of the bigger differences here is that we can actually play sounds in 3D space, hence the use of a `sf::Vector3f` structure wherever a position needs to be represented. Sounds are also grouped by specific names, but there is a slight twist to this system. SFML can only handle about 255 different sounds playing all at once, which includes `sf::Music` instances as well. It's because of this that we have to implement a recycling system that utilizes discarded instances of sounds, as well as a static limit of the maximum number of sounds allowed all at once.

Every different sound that is loaded and played has specific set up properties that can be tweaked. They are represented by this data structure:

[PRE59]

`audioName` is simply the identifier of the audio resource that is loaded in memory. The volume of a sound can obviously be tweaked, as well as its pitch. The last two properties are slightly more intricate. A sound at a point in space would begin to grow quieter and quieter, as we begin to move away from it. The minimum distance property describes the unit distance from the sound source, after which the sound begins to lose its volume. The rate at which this volume is lost after that point is reached is described by the attenuation factor.

# Summary

That was quite a lot of information to take in. In the span of around forty pages we have managed to summarize the better part of the entire code base that would make any basic to intermediate complexity game tick. Keep in mind that although many topics got covered here, all of the information was rather condensed. Feel free to look through the code files we provide until you feel comfortable to proceed to actually building a game, which is precisely what's coming in the next chapter. See you there!