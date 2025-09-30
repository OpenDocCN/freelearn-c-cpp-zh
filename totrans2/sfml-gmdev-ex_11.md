# Chapter 11. Don't Touch the Red Button! – Implementing the GUI

We covered the fundamentals and created the building blocks necessary for graphical user interface assembly in the course of the last chapter. Although that might seem like lots of code, a lot more goes into making it tick. Proper management of interfaces, good support from the rest of the code base, and user-friendly semantics of the GUI system itself are all paramount. Let's finish our goal set in [Chapter 10](ch10.html "Chapter 10. Can I Click This? – GUI Fundamentals"), *Can I Click This? – GUI Fundamentals*, and finally provide our users with a means of interfacing.

In this chapter, we will cover the following topics:

*   Management of interfaces and their events
*   Expansion of the event manager class for additional GUI support
*   Creation of our first element type
*   Integration and use of our GUI system

With all the pieces in place, let's bring our interfaces to life!

# The GUI manager

The puppet master in the background, in charge of the entire show in this case, has to be the `GUI_Manager` class. It is responsible for storing all the interfaces in the application as well as maintaining their states. All mouse input processing originates from this class and is passed down the ownership tree. Let's begin by getting some type definitions out of the way:

[PRE0]

We will use the `std::unordered_map` data structure that indexes them by name to store the interface data. The interface data containers also need to be grouped by game states, which is what the next type definition is for. Similarly, GUI events need to be indexed by their relevant game state. The events themselves are stored in a `std::vector`.

Additionally, since we will be creating elements in a factory-like fashion, much like we did before, a factory type definition is created. The main difference here is that the `lambda` functions we'll be storing need to take in a pointer to the owner interface in order to be constructed correctly.

Lastly, we're going to be mapping element type strings to actual enumeration values for the same. Once again, the `std::unordered_map` type comes to the rescue.

Now, here is the class definition itself:

[PRE1]

Right off the bat, we can tell that the factory method for elements is going to be used due to the presence of a `RegisterElement` method. It stores a `lambda` function with an owner interface pointer as its sole argument, which returns a `GUI_Element` type with a blank name, constructed from a given type denoted by the `l_id` argument. Its private method friend, `CreateElement`, will use the stored `lambda` functions and return pointers to newly created memory.

One last thing to note before diving into the implementation of this class is the existence of a `LoadStyle` method that takes in a `GUI_Element` type. The manager class is responsible for de-serializing style files and properly setting up elements based on them to avoid cluttering up the element and interface classes.

## Implementing the GUI manager

With the class header out of the way, we can dive right into implementing our GUI manager. The constructor of the `GUI_Manager` class is defined like this:

[PRE2]

It requires a pointer to the event manager and shared context structures as arguments and sets them up through the initializer list, along with a default value for the current state. Inside the body, we can see that this class first registers three element types that we're going to be working with. It also populates the element type map, which will be used for checks further down the line. Finally, it registers three callbacks: two for the left mouse button being pressed and released and one for text being entered. Note that these callbacks are registered to be called regardless of the state the application is in.

[PRE3]

The destructor removes all of the callbacks registered in the constructor and iterates over every single interface for proper de-allocation of dynamically allocated memory. The interface and event containers are then cleared.

Let's take a look at how an interface is added to the GUI manager:

[PRE4]

Dynamic memory for an interface is allocated and an attempt to insert it is made when a valid application state and an unused interface name is provided. Any issues when inserting are caught by the return value of the `emplace` method, which gets stored in the `i` variable. If it fails, the memory is de-allocated and `false` is returned to signify failure. Otherwise, `true` is returned.

Obtaining an interface is as simple as it gets:

[PRE5]

If a state provided as an argument is found, and an interface with the name provided is also located, it gets returned. Failure to find either a valid state or the correct interface is represented by a return value of `nullptr`.

Removing an interface is achieved by manipulating the container structure:

[PRE6]

### Note

Note that the `delete` keyword appears if both the state and the interface are found. Sometimes, it's very easy to forget the de-allocation of no longer used memory on the heap, which results in memory leaks.

Since the GUI manager needs to keep track of the current application state, the following method is necessary:

[PRE7]

In addition to changing the current state data member, it also invokes the `HandleRelease` method to prevent sticky interface and element states. If an element is clicked and the state suddenly changes, that same element will remain in the `CLICKED` state until it is hovered over unless `HandleRelease` is called.

Now, let's handle the mouse input to provide interaction with our interfaces:

[PRE8]

This method, just like its `HandleRelease` brother, takes in a single argument of the type `EventDetails`. For now, simply ignore that as it does not affect `GUI_Manager` at all and will be dealt later in this chapter.

Firstly, it obtains the current mouse position relative to the window. Next, an iterator to the interface container is obtained and checked for validity. Every interface that belongs to the current state is then iterated over in reverse order, which gives newly added interfaces priority. If it is active and the mouse position falls within its boundaries, its `OnClick` method is invoked, with the mouse position passed in as the argument. The interface's `m_beingMoved` flag is then checked because the click might've been within the boundaries of its title bar. If so, the `BeginMoving` method is called to complete the drag operation. At this point, we simply return from the method in order to prevent a left click from affecting more than one interface at a time.

Handling the left mouse button release follows the same convention:

[PRE9]

The only difference here is that every interface which is in a `Clicked` state has its `OnRelease` method called, as well as the `StopMoving` method if it is in a state of being dragged.

Lastly, let's not forget about our text field elements as they need to be notified whenever some text is entered:

[PRE10]

This is a quite simple snippet of code. Whenever text is entered, we attempt to find an active and focused element. Once we find one, its `OnTextEntered` method is invoked with the text information passed in as the argument.

Adding GUI events is as simple as pushing them back onto a `std::vector` data structure:

[PRE11]

In order to properly handle these events, we must have a way to obtain them:

[PRE12]

This is similar to the way SFML handles events, in that it takes in a reference to a `GUI_Event` data type and overwrites it with the last event in the event vector, right before popping it. It also returns a Boolean value to provide an easy way for it to be used in a `while` loop.

Next, let's work on updating the interfaces:

[PRE13]

After the current mouse position is obtained, every interface that belongs to the current application state is iterated over. If the interface is currently active, it gets updated. The `Hover` and `Leave` events are only considered if the interface in question is not currently being dragged, as we did with the smaller GUI elements inside interfaces.

Now it's time to draw all of these interfaces onto the screen:

[PRE14]

Once again, this method iterates over all interfaces that belong to the current application state. If they're active, each re-draw flag is checked and the appropriate re-draw methods are invoked. Finally, a pointer to the `sf::RenderWindow` is passed into the `Draw` method of an interface so it can draw itself.

It would be good to have a method for creating these types automatically because we're working with factory-produced element types:

[PRE15]

If the provided element type is a `Window`, a new interface is created, to which a pointer of `GUI_Manager` is passed as its second argument. In the case of any other element type being passed in, the factory container is searched and the stored `lambda` function is invoked with the `l_owner` argument passed in to it.

Lastly, let's discuss the de-serialization of interfaces. A method is needed to load files formatted in this way:

[PRE16]

Next, let's work on loading our interfaces from a file. We're not going to cover how the file itself is read as it's pretty much identical to how we usually do it:

[PRE17]

Let's start with creating an interface:

[PRE18]

As suggested by the file format, it first needs to read in its name and the name of the style file. If adding an interface with the loaded name fails, an error is printed out and the file reading is stopped. Otherwise, a pointer to this freshly added window is obtained and its overloaded **>>** operator is used to read in additional information from the stream, which we covered back in the interface section of this chapter.

Next, an attempt is made to load the style file that was read in earlier by calling the `LoadStyle` method, which we will be covering shortly. If it fails, an error message is printed out. Lastly, its content size is adjusted based on its current style.

Handling element de-serialization, in its most basic form, is quite similar:

[PRE19]

The element type, the name, position, and the style values are read in from the file. An element type is obtained after running the text that was read into the `type` variable through our helper method `StringToType`. An interface that the element needs to be added to is obtained by using the name passed in as an argument to the `LoadInterface` method. The `AddElement` method of the obtained interface is called in order to create the appropriate element type on the heap. If it's successful, the element is obtained by name and its additional information is read in by utilizing its overloaded `>>` operator. The `LoadStyle` method is invoked once again in order to read the style of an element from a file. Let's take a look at what this looks like:

[PRE20]

With this serving as an example, it's time to try and read it in. Once again, we're going to skip the code that reads the file as it is redundant. With that in mind, let's take a look:

[PRE21]

Note the two `GUI_Style` structures that are set up here: they keep track of the main style that serves as a parent and the temporary style that's currently being read in. Let's keep moving further down this method, inside the actual `while` loop:

[PRE22]

If a `State` keyword is encountered and `currentState` is not set up, the name of the state is read in. Otherwise, we print out an error message:

[PRE23]

When encountering a `/State` keyword, we can safely assume that the style currently being processed has ceased. The state is then determined based on the string that was read in denoting it.

If the state is `Neutral`, we need to set it to be the parent style, which means that every unset property of the other styles will also be inherited from this one. The `UpdateStyle` method is then invoked for each of the three supported states in order to overwrite the default values. If it is anything other than `Neutral`, the `UpdateStyle` method is only invoked once for that state. The `TemporaryStyle` variable is then overwritten with `ParentStyle` to simulate inheritance.

Finally, let's see how every different style feature is supported:

[PRE24]

Every color value is first read in as four separate integers and then stored in a `sf::Color` structure which gets assigned to the appropriate data member of the style structure. Padding and text values are simply streamed in. One exception to this is the `TextOriginCenter` tag. It does not contain any additional information and its mere existence simply means that the origin of the text element should always be centered.

# The label element

A label element is the simplest GUI type yet. It supports all of the default stylistic features but it doesn't do much else other than contain a certain string value that can be loaded in or set at runtime.

Let's take a look at its constructor and destructor:

[PRE25]

This is nothing short of child's play in comparison to the code we've written before. Its name, type, and owner are set up in the initializer list and there's nothing else to it.

The de-serialization of this type of element is also fairly simple. Recall the following line from an interface file:

[PRE26]

Since the `GUI_Manager` class takes care of all of this information except the last part, the `ReadIn` method of this element might look like this:

[PRE27]

Now, we have to implement the event methods of this element. In this case, it's nothing more than simply adjusting the state of the label:

[PRE28]

The final bit of code is responsible for how this element is drawn:

[PRE29]

After the background rectangle is drawn, the glyph is checked to see whether it needs to be drawn as well. Lastly, the text is rendered right on top of the last two visual attributes.

# The text field element

In order to implement a text field element successfully, we need to define how it responds to input correctly. Firstly, let's set up a new element type by creating the text field element class and implementing the constructor, as shown here:

[PRE30]

This element can also have a default text value when loaded, so let's express that by providing a custom version of the `ReadIn` method:

[PRE31]

As you probably know, text fields do not change state if a mouse button is released. This allows them to be focused until a mouse click is registered elsewhere. We have already implemented that functionality in the `GUI_Interface` class as the `DefocusTextfields` method. All that's left to do now is ignore release events:

[PRE32]

Lastly, let's take a look at drawing this element:

[PRE33]

It is quite simple in nature. So far, we have only worried about drawing the background solid behind the text that this element holds. The glyph is also supported here but we're not going to be using it.

# The scrollbar element

All of that support for interface scrolling and control elements implies the existence of the scrollbar element. Its purpose is to move around the visible area of the content texture in order to reveal elements that are positioned further out than its size allows, which could be along any axis. With that knowledge, let's take a stab at working out the basic class definition of the scrollbar element:

[PRE34]

Firstly, we enumerate both possible types of sliders: horizontal and vertical. The actual `GUI_Scrollbar` class overwrites three of the original methods the parent class provides, in addition to implementing all of the purely virtual ones.

Among its private data members, the scrollbar keeps track of its own type, which contains another drawable object to represent the slider and maintains information about the last known mouse coordinates, as well as the percentage value of scroll it's currently at.

Let's start with the easy part – the constructor:

[PRE35]

It's pretty straightforward so far. The element type is set to `Scrollbar` and the `m_isControl` flag is set to `true` to tell the owner interface which layer to draw it on.

Next up, the `SetPosition` method needs to be overwritten to make sure that the scrollbar is positioned correctly:

[PRE36]

Due to the nature of this particular element, one axis has to be always set to `0` in order to keep it positioned on the right edge.

For now, the type of a scrollbar will be read in from the interface file. To make that happen, we may want to handle de-serialization like this:

[PRE37]

Let's handle the events next, starting with `OnClick`:

[PRE38]

Since we only want scrolling to happen when the slider part is being dragged, the state of this element is only set to `Clicked` if the mouse coordinates are inside the slider. They then get stored in the `m_moveMouseLast` data member to prevent the slider from jumping.

The remaining three events are not needed for anything other than adjusting the state:

[PRE39]

The style updating also has to be altered to maintain the desired functionality of the scrollbar:

[PRE40]

The size of the scrollbar is set to match the size of the owner interface on the relevant axis after the parent `UpdateStyle` is called.

Next, we have to define a custom way of applying style attributes to scrollbar elements, due to their unique nature:

[PRE41]

After the parent `ApplyStyle` is invoked and the slider color is set, the position of the element is overwritten to keep it at 0 on the axis of action and right near the edge on the perpendicular axis. The size of the background solid is determined by the size of the interface on the scroll axis. Its style attributes determine the other size value.

The position of the slider is modified on the non-operational axis to always match the position of the element itself. Calculating its size along the scrolling axis is as simple as dividing the size of the owner window by the result of dividing its content size by the same window size.

With the style part of this element complete, let's work on moving it and affecting its owner interface:

[PRE42]

All of the code above only needs to be executed if the state of this element is `Clicked`. It's then obvious that the slider of the scrollbar is being dragged up and down. If the current mouse position is not the same as the last position from a previous iteration, the difference between them is calculated and the current position of the mouse is stored for later reference.

Firstly, the slider is moved by the difference of the mouse positions between the last two iterations. It is then checked to see if it is outside the boundaries of the interface, in which case, its position gets reset to the closest edge.

Lastly, the scroll percentage value is calculated by dividing the slider's position on the relevant axis by the difference of the window size and the slider size. The relevant update method for scrolling is then invoked and this element is marked to be re-drawn to reflect its changes.

The last thing we need to do is define how the scrollbar element is drawn:

[PRE43]

For now, it only uses two rectangle shapes, however, this can easily be expanded to support textures as well.

# Integrating the GUI system

In order to use the GUI system, it needs to first exist. Just like in previous chapters, we need to instantiate and update the GUI classes we built. Let's start by adding the GUI manager and the font manager to the `SharedContext.h` file:

[PRE44]

We need to keep a pointer to the GUI manager and the font manager in the `Game` class, as with all of the other classes that are shared through the `SharedContext` structure, starting with the header:

[PRE45]

These pointers are, of course meaningless, unless they actually point to valid objects in memory. Let's take care of the allocation and de-allocation of resources in the `Game.cpp` file:

[PRE46]

Next, we can look at updating all of the interfaces in the application and handling GUI events:

[PRE47]

Note that the `GUI_Event` instance is forwarded to the `EventManager` class. We're going to be expanding it soon.

Finally, let's handle drawing our interfaces:

[PRE48]

In order for the GUI to be always drawn above the rest of the scene, the window view has to be set to the default before the interfaces are drawn. It then needs to be set back in order to maintain a consistent camera position, which might look something like this:

![Integrating the GUI system](img/B04284_11_01.jpg)

# Expanding the event manager

GUI events need to be handled for every possible state of the application in order to keep them from piling up, much like SFML events. In order to avoid writing all of that extra code, we're going to use something that was built solely for the purpose of handling them: the event manager.

Let's start by expanding the `EventType` enumeration to support GUI events:

[PRE49]

It's important to keep these custom event types at the very bottom of the structure because of the way the code we've written in the past works.

Our previous raw implementation of the `EventManager` class relied on the fact that any given event can be represented simply by a numeric value. Most SFML events, such as key bindings, fit into that category but a lot of other event types, especially custom events, require additional information in order to be processed correctly.

Instead of using numbers, we need to switch to a lightweight data structure like this:

[PRE50]

The union ensures that no memory is wasted and that we can still use numeric representations of event types, as well as custom data-types, such as the `GUI_Event` structure. `GUI_Event` belongs to a union, which is why it couldn't use `std::string` typed data members.

### Tip

If the boost library is used, all of this code can be reduced to `boost::variant<int, GUI_Event>`.

One additional change is that we want to be able to pass the GUI event information to the callback methods that are registered. This information will also be held by our `EventDetails` structure:

[PRE51]

Now, let's adjust the `Binding` structure:

[PRE52]

We had to use `const char*` data types to hold element and interface names because of union restrictions. While that only applies to GUI-related events, this memory still needs to be de-allocated. When a binding is being destroyed, all of the event information is iterated over and checked to see if it is any of the four GUI event types, in which case the memory is safely de-allocated.

Next, we need a separate method that handles just the GUI events. Overloading the `HandleEvent` method with a different argument type seems like a good choice here:

[PRE53]

We need to make sure that no GUI events are processed in the original `HandleEvent` method:

[PRE54]

If the event is of one of the four GUI types, the iteration is skipped. Handling the GUI events themselves is quite simple and can be done in this manner:

[PRE55]

While iterating over the events inside bindings, their types are checked. Anything that is not a GUI event is skipped over. If the type of a processed event matches the type inside the binding, additional information is checked in the `EventInfo` structure, namely the interface and element names. If those match too, they are recorded as event details and the event count is incremented.

The last chunk of code that needs attention is the `LoadBindings` method. We need to adjust it to support interface and element name-loading from the `keys.cfg` file, which should look something like this:

[PRE56]

The first line represents a normal type of event, while the second line is a GUI event, which requires two identifiers to be loaded instead of just one. Let's adjust it:

[PRE57]

After the event type is loaded in as usual, it is checked to see if it matches any of the four GUI events. The window and element strings are then read in and copied to the newly allocated memory of `char*` via the `std::strcpy` method.

### Note

Keep in mind that when memory for `char*` types is allocated to match a given string, it also needs an additional space for the null-terminating character at the end.

# Re-implementing the main menu

In order to demonstrate how much easier it is building interactivity in this way, let's re-construct the main menu, starting by creating its `.interface` file:

[PRE58]

The interface is set to have zero padding on both axes, be immovable, and have no title bar. All three buttons in this interface, as well as its title, can be represented by labels with different styles. Speaking of which, let's take a look at the style of our main menu interface:

[PRE59]

As you can see, it only defines the most basic attributes and does not aim to be visually responsive by itself. The button label style, however, is a little different:

[PRE60]

When its state changes, the label's background color is adjusted as well, unlike the label that represents the title of the main menu:

[PRE61]

With all of the visual elements out of the way, let's adjust the main menu state to load and maintain this interface:

[PRE62]

In addition to all of the required methods that a state has to implement, we only need two callbacks to handle GUI clicks. This is all set up in the `OnCreate` method of the main menu state:

[PRE63]

Firstly, the main menu interface is loaded from a file and placed on screen. The event manager is then used to set up callbacks for the **Play** and **Quit** button actions. This is already much cleaner than the previous approach.

Once the state is destroyed, the interface and two callbacks must be removed, as shown here:

[PRE64]

The text of the **Play** button must be changed if a `GAME` state exists:

[PRE65]

That leaves us with our two callbacks, which look like this:

[PRE66]

This illustrates perfectly how easy it is to use our new GUI with an improved event manager for fast and responsive results. The main menu was created with roughly 20 lines of code, or fewer, and looks like this:

![Re-implementing the main menu](img/B04284_11_02.jpg)

# Summary

At the beginning of [Chapter 10](ch10.html "Chapter 10. Can I Click This? – GUI Fundamentals"), *Can I Click This? – GUI Fundamentals*, our main goal was to achieve a simple yet powerful means of interfacing with our own application. Throughout this chapter, additional topics such as interface and event management, creation and integration of new element types, and expansion of existing code were covered in depth. The effectiveness of all the work that was put into the GUI cannot be measured in any other way but success. We are now left with a system that is capable of producing efficient, responsive, and fast results with the minimum amount of effort and code. Furthermore, you should now have the skills necessary to build even more element types that will enable this system to do amazing things.

In the next chapter, we're going to be covering the management and usage of sound and music elements in SFML. See you there!