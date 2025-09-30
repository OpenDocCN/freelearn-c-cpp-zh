# Chapter 10. Can I Click This? – GUI Fundamentals

What do humans and machines really have in common, in the non-Turing sense of the word? It seems like the everyday life of an average human nowadays is almost synonymous with operating the large number of contraptions our species has created, yet most of us don't even speak the same language as the devices we use, which creates a need for some kind of translation. Now it's not as if we can't learn how to speak to machines directly, but it's simply too tedious and time-consuming as our brains work in a completely different way to a common processor. A gray area exists in which relatively intuitive actions performed by humans can also be understood and interpreted by machines without the need for ever getting involved with the underlying complexities - the means of interfacing.

In this chapter, we will cover the following topics:

*   Implementation of core data types for all GUI elements
*   Utilizing SFML's render textures to achieve GUI layering
*   Laying down the fundamentals of smooth and responsive GUI interactions by using stylistic attributes

There is quite a bit of ground to cover so let's get started!

# Use of copyrighted resources

Before we begin, it's only fair to credit the true creators of the fonts and images used in the next two chapters:

*Fantasy UI Elements* by *Ravenmore* at [http://dycha.net/](http://dycha.net/) under the CC-BY 3.0 license:

[http://opengameart.org/content/fantasy-ui-elements-by-ravenmore](http://opengameart.org/content/fantasy-ui-elements-by-ravenmore)

*Vegur font* by *Arro* under the CC0 license (public domain):

[http://www.fontspace.com/arro/vegur](http://www.fontspace.com/arro/vegur)

More information about all of the licenses that apply to these resources can be found here:

[http://creativecommons.org/publicdomain/zero/1.0/](http://creativecommons.org/publicdomain/zero/1.0/)

[http://creativecommons.org/licenses/by/3.0/](http://creativecommons.org/licenses/by/3.0/)

# What is a GUI?

A GUI, short for **Graphical User Interface**, is a visual intermediary between the user and a piece of software which serves as a control mechanism for digital devices or computer programs. Using this type of interface is faster and easier than relying on text-based controls, such as typing commands.

Before any code is written, we need to outline the desired features of our GUI system, which is going to consist of three major components:

*   **Element**: Every GUI surface that is drawn onto the screen
*   **Interface**: A special kind of element that serves as a container for other elements and can be moved around as well as scrolled
*   **Manager**: The class that is in charge of keeping GUI interfaces in line and behaving

All of the elements in this system need to be able to adapt to a different state when they are hovered over by a mouse or clicked on. Style-sets also need to be applied to different states, resulting in interfaces becoming responsive. Lastly, you must be able to load the interfaces from files at runtime and tie them to code based on an event or a set of events taking place within them.

## GUI style

Unifying the way styles are applied and used on GUI surfaces is crucial if you need customization and flexibility. To put it simply, modifying and applying every single stylistic attribute of every possible type of element manually would be a nightmare and any kind of code re-use would be impossible. This calls for a custom data type that can be used all across the board: the `GUI_Style` structure.

First things first, any and all GUI elements should be able to support the three following states:

[PRE0]

Although these states are not only meant for graphical purposes, each one of them is also defined as a set of visual properties in order to simulate interaction and fluidity, represented by a set of style attributes:

[PRE1]

An element or interface can alter every single one of these properties and adjust itself to look completely different based on the state it's in. If they are not defined, the default values set in the constructor take precedence, as shown here:

[PRE2]

All of this is useless if we don't have drawable objects to modify, so let's fix that:

[PRE3]

This basic structure will be a part of every single element and interface allowing them to be represented by any combination of these four drawable elements.

# Expansion of utility functions

In order to keep things simple and easy to read, it's always a good idea to create utility-type functions out of any code that is going to be used frequently. When dealing with interface de-serialization, many elements have to read in parameters that have spaces in them. Our solution to this problem is to put the string in double quotes and define an inline function to read the data. The perfect spot for that is in the `Utilities.h` file:

[PRE4]

A word is loaded from the string stream object into the string provided as an argument. Its first character is checked to see if it is a double quote. If it is, a `while` loop keeps reading in words and appends them to the argument string until either its last character is a double quote or the end of the stream is reached.

Following that, all of the double quotes in the string are erased.

# Font management

Before we start to build the structure of our graphical user interface, we need a way to manage and handle the loading and unloading of fonts automatically, just like we did with textures. The effort we put into the resource manager written back in [Chapter 6](ch06.html "Chapter 6. Set It in Motion! – Animating and Moving around Your World"), *Set It in Motion! – Animating and Moving around Your World*, is about to pay off. In order to manage fonts, all we need to do is create a `FontManager.h` file and write the following code:

[PRE5]

This defines the font resource configuration file in the constructor, as well as the specific way of loading font files using the `Load` method. The resource manager class that we implemented earlier makes this process very simple, so let's keep going!

# The core of all elements

The `GUI_Element` class is the core of every single element and interface. It provides key functionality that higher level objects rely on, as well as enforcing the implementation of necessary methods, which leads to several distinctive element types.

A definition of the different element types is a good place to start:

[PRE6]

Each element has to hold different styles it can switch to, based on its state. The `unordered_map` data structure suits our purposes pretty well:

[PRE7]

A forward declaration of the owner class is also necessary to prevent cross-inclusion:

[PRE8]

Next, we can begin shaping the `GUI_Element` class:

[PRE9]

The most essential part of any GUI element is how it responds to events. This is where the magic of pure-virtual methods comes in. Style application methods, however, are not purely virtual. An element doesn't handle its style any differently to a default element.

Every element also needs to have a name, a position, a set of styles for every possible state, a visual component that can be drawn, a type and state identifiers, and a pointer to an owner class. It also needs to keep track of whether it needs to be re-drawn, its active status, and a flag that denotes whether it is a control or not. These properties are represented by a set of private data members of the `GUI_Element` class.

With a rough idea of this structure hammered out, let's shape the finer details of the element class.

## Implementing the GUI element class

The class we're about to begin implementing is a cornerstone of every single interface and element. It will define how our GUI system behaves. With that in mind, let's start by taking a look at the constructor, as we have quite a bit to initialize:

[PRE10]

The element name, type, and a pointer to the owner class arguments are taken in and passed to the appropriate data members. Other additional flags are also initialized to the default values. There's nothing out of the ordinary so far. Let's take a look at how this class is destroyed:

[PRE11]

Since there is no dynamic memory allocation going on anywhere in this class, releasing resources is also fairly simple. The method for that specific purpose is simply invoked here. It looks a little like this:

[PRE12]

We only have to concern ourselves with those textures and fonts that are required by the element itself, so each style is iterated over and its resources are released by the respective methods, which all look similar to the one shown:

[PRE13]

If a font is released, the only difference is the manager that is being used.

Speaking of styles, we need to have a regulated way of modifying them. The `UpdateStyle` method takes care of that job:

[PRE14]

Two arguments are expected by this method: the state being modified and a style structure that will be used to replace the existing structure. While overwriting the relevant style is as simple as using the assignment operator, some resource management has to take place before that happens. We need to know if the style being replaced requires different resources to the other one. If it does, the older textures and fonts are released, while the new ones are reserved by using two more helper methods which both look something like this:

[PRE15]

The font equivalent for this method uses a different manager but is otherwise identical.

Once the style is overwritten, we check if the state being modified is the same state as the element. If so, this particular element is marked to be re-drawn via the `SetRedraw` method, and its style is applied via the `ApplyStyle` method, which is what we'll take a look at next:

[PRE16]

This chunk of code is responsible for connecting the style of an element to its visual representation. It first invokes a few helper methods which help us break down the code into smaller, more manageable chunks. The owner interface needs to be alerted afterwards because any modification of the element style may result in size changes. If the element is not an interface control and isn't its own owner, the `AdjustContentSize` method of the `GUI_Interface` class is called, with the `this` keyword passed in as an argument. We will get to implement it soon.

Let's take a look at the first helper method, which deals with text style:

[PRE17]

A different font, color, and character size can be applied to the text for each distinct style that an element can have. The text's origin also needs to be re-calculated every time this happens because these attributes can be manipulated at any point. The position of the text is then updated with the padding value of the current style being factored in.

Background style application follows the same basic idea:

[PRE18]

This shows how we add support for the background image and solid elements. Both of these elements are adjusted by applying the visual attributes of a current style and having their positions re-set.

Finally, the glyph of an element is altered in the same fashion:

[PRE19]

Next, let's take a look at the changing element states:

[PRE20]

The element must be marked for re-drawing if its state is changed as different states may have style elements that are also different. That, however, is only done if the state provided as an argument does not match the current state, which is done to conserve resources.

Setting the element position also deserves some attention:

[PRE21]

Since all elements are owned by a container structure, their positions must also honor the padding of those containers. Once the element position is set, the padding of the container interface is obtained. If the element position on either axis is less than that padding, the position is set to be at least as far away from the edge as the interface allows it to be.

Here's an important bit of code that can make or break interactions with any GUI surface:

[PRE22]

The `IsInside` method is used to determine whether a certain point in space is inside an element. Calculating intersections using its normal position yields incorrect results because of its relativity to its owner. Instead, it uses a `GetGlobalPosition` method to fetch the element's position in screen space, as opposed to local space, in the render texture of the owner interface. With a bit of basic bounding box collision magic, it then determines if a point provided as an argument is within the element, based on the size of its current style.

Obtaining global positions of elements can be done like so:

[PRE23]

Firstly, the element's local position is grabbed. The method then determines if this element has an owner and if does not own itself. If it does, the fetched position is simply the final result and is returned. Otherwise, the owner's global position is obtained through the use of this very method and added to the local position. Furthermore, if the element is not a control type, the horizontal and vertical scroll values are subtracted from its position in order to honor interface scrolling.

To cap things off, here are a few setters and getters that are not straightforward:

[PRE24]

### Note

Note the methods `SetActive` and `SetText`. Whenever an element is modified, we must set its re-draw flag to `true`, otherwise it won't be updated until another event requires it.

# Defining GUI events

Providing fluid interactivity with the interface and a painless way of associating changes with actions inside your application may be the most important criterion to separate good GUI systems from bad ones. As we are already learning SFML, we can use the SFML method and omit events.

Firstly, we have to define all the possible events that could take place in an interface. Create a `GUI_Event.h` file and construct an enumeration, as shown here:

[PRE25]

We must also define a custom structure in the same file that is used to hold event information:

[PRE26]

The first thing to talk about here is the structure. It should be possible to merely use `sf::Vector2f` here. That would work fine under most circumstances but, a few lines below that, you see the importance of `ClickCoordinates`. Based on the type of event we're going to be working with, it's going to need to store different data in the `GUI_Event` structure. By using a *union* inside this structure, we're going to avoid allocating additional memory, but that comes at a price. Unions cannot have members that have member functions, virtual functions, or are derivatives of other classes. It is because of this restriction that we are forced to define our own `struct` that holds two *floats* and represents a point.

### Tip

The boost library could potentially be useful in a situation like this as it provides `boost::variant`, which is a type-safe union container that doesn't have these limitations. It also has little or no overhead.

The actual event structure holds an event type that is used to determine which member of the union is active, as well as names of the element and interface the event originated from. If you have a good eye for detail, you may have asked yourself by now why we're using `const char*` data types instead of `std::string`. Simplifying data types of data members is another sign that this structure will be incorporated into a union. Unfortunately, `std::string` falls into the same trap as `sf::Vector2f` and cannot be used in a union without extra work.

# The interface class

An interface, in its simplest meaning, is a container of elements. It's a window that can be moved around and scrolled and has all of the same features and event hooks as a regular element. Efficiency is also a great concern, as dealing with lots of elements in a single window is a definite possibility. Those problems can be dealt with by carefully designing a way of drawing elements at the appropriate time.

The way we want our interfaces to draw content is by using three separate textures for different purposes, as shown below:

![The interface class](img/B04284_10_01.jpg)

*   The **background** layer is used for drawing backdrop elements
*   The **content** layer is where all of the elements of the interface are drawn
*   The **controls** layer hosts elements such as scrollbars that manipulate the content layer and don't need to be scrolled

With the design details out of the way, element storage deserves some attention. As it happens, the `std::unordered_map` structure serves this purpose well:

[PRE27]

Next, a forward declaration of the owner class is needed to prevent cross-inclusion:

[PRE28]

All of this brings us to the `GUI_Interface` class:

[PRE29]

### Note

Note the declarations of `friend` classes. Both the `GUI_Element` and `GUI_Manager` need to have access to private and protected members of this class.

For now, let's only focus on the private members and leave the public ones for the implementation section of this chapter.

In addition to having an element container, an interface also defines the amount of padding it has that elements most honor, a pointer to its parent class if it has one, as well as the manager class and a set of textures that represent its different layers. The rest of the data members, as well as the omitted methods, can't be fully understood unless we talk about implementation details, so let's get right to it!

## Implementing the interface class

A nice place to start, as always, is with the class constructor:

[PRE30]

Quite a lot of data members are initialized through the initializer list here. Firstly, the parent class `GUI_Element` needs to know the name, type, and owner of the interface. One of the `GUI_Interface` arguments is its name, which gets passed to the `GUI_Element` constructor. The type is, of course, set to `Window`, and the `this` keyword is passed in as the owner of the interface. Additionally, the parent of the interface is initialized to its default value `nullptr` and a pointer to the `GUI_Manager` class is stored inside the `m_guiManager` data member.

After the data member initialization, we enter the constructor's body, in which three `sf::RenderTexture` objects are allocated dynamically. These are the textures that are used to render the background, content, and control layers of an interface.

Next, let's take a look at freeing up all of these resources in the destructor:

[PRE31]

The three texture instances, of course, have to be deleted, as well as every single element that still resides in the element container at the time of destruction. Afterwards, the element container is cleared.

Setting the position of an interface is slightly more complex, so let's take a look:

[PRE32]

Firstly, the `SetPosition` method of the parent class is invoked in order to adjust the actual position. There is no need to fix what isn't broken. Next, the three sprites that represent the background, content, and control layers have their positions adjusted as well. Lastly, you set up the title bar. The solid background shape's position is set to be right above the interface, while the text of the visual component is used as a title and adjusted to take the same position as the title bar background, except with text padding included.

Empty windows aren't very useful or entertaining, so let's provide a way in which elements can be added to them:

[PRE33]

It is important to avoid name clashes so check the name provided as the second argument against the element container in order to prevent duplicates. If none are found, a `CreateElement` method of the `GUI_Manager` class is used to create an element of the relevant type on the heap and return its memory address. After verifying that it has indeed been created, the element's name and owner properties are set before it gets inserted into the element container. The interface then sets two flags to re-draw the content and control layers.

Any interface needs to have a way to provide access to its elements. That's where the `GetElement` method comes in:

[PRE34]

It simply locates the element in the `std::unordered_map` using its find method and returns it. If the element isn't found, `nullptr` is returned instead. Easy.

Next, we need to have a way to remove elements from interfaces:

[PRE35]

Following the same example as the `GetElement` method, the element is first located inside the container. The dynamic memory is then de-allocated by using the `delete` operator and the element itself is removed from the container. The interface is marked to re-draw its content and control layers and the `AdjustContentSize` method is invoked to re-size the content texture, if needed.

We need to override the original `IsInside` method because interfaces occupy additional space due to their title bars, as shown here:

[PRE36]

The parent class method is invoked first to determine if `l_point` is inside the space an interface is occupying. If not, the result of the title bar bounding box's `contains` method is returned to determine if `l_point` is inside that.

Next is shown the de-serialization portion of the code:

[PRE37]

All interfaces first read in the element padding *x* and *y* values, as well as the state and title parameters. It then uses the `ReadQuotedText` utility function which we defined earlier to read in the actual title of the interface. Based on the strings read in, it then sets the `m_movable` and `m_showTitleBar` flags to reflect those values.

Now comes the fun part. Let's define what happens when an interface is clicked:

[PRE38]

Firstly, we invoke one of the private helper methods responsible for removing focus from all of the `Textfield` GUI elements. This will be covered in more depth later. Another problem is dragging, when a click is detected in an interface. If the mouse position is in the title bar area and the interface itself is movable, we set the `m_beingMoved` flag to `true` to indicate interface dragging.

In a case of it just being a regular click anywhere else within the interface boundaries, we first set up an event that is going to be dispatched, indicating that a click has happened. The type is set to `Click`, the interface name is copied as a *c string*, and the mouse coordinates are also set up. The `AddEvent` method of the `GUI_Manager` class is invoked with our newly created event as an argument. This first event indicates that a click happened within the interface itself and not in any particular element.

That is quickly followed by a loop that iterates over every single element in the interface. Their `IsInside` method is called to determine whether the click that took place was also within any of the elements. If so, the `OnClick` method of that particular element is invoked with the mouse position passed in as an argument. The same event that was set up before the loop is then slightly modified to contain the name of the element and is fired again, indicating that the click also affects it. The interface's state is then changed to `CLICKED`. The result of this is quite appealing to the eye:

![Implementing the interface class](img/B04284_10_02.jpg)

Next, let's take a look at the opposite side of clicking—the `OnRelease` method:

[PRE39]

Just like before, an event is set up and fired, indicating that a release happened within this specific interface. Every element is then iterated over and their states are checked. If the element is in a `Clicked` state, its `OnRelease` method is called and another event is fired, indicating the release of the left mouse button within that element. The state of the interface is then set to `Neutral`.

An interface also needs to deal with text being entered:

[PRE40]

This method is going to be invoked whenever an SFML event `sf::Event::TextEntered` is received by our window. Each element is iterated over until we find one that is of the type `Textfield` and is currently in a `Clicked` state. The backspace key being pressed is handled by having the last character of our element's text attribute trimmed. Note that we're returning from the method in multiple places in order to avoid several `Textfield` elements receiving the same text that is being entered.

Lastly, we need to check the boundaries of the character value that has been received. Any characters below ID `32` or above `126` are reserved for other purposes and we're not interested in those. If a regular letter or number is typed in, we want to update our text attribute by adding that character to it.

### Tip

The full table of ASCII characters can be found here: [http://www.asciitable.com/](http://www.asciitable.com/)

Since we're on the subject of handling text field elements, let's take a look at a method that we used before when handling a `Click` event:

[PRE41]

When handling text fields, it's important to bear in mind that they lose focus every time a mouse is left-clicked. If that wasn't so, we would end up with text entered across multiple textboxes, and that is no good. Making a text field lose focus is as simple as constructing a `Release` event and sending it to every `Textfield` element that an interface possesses.

The next two methods are grouped together due to their similarities:

[PRE42]

An event is constructed including the mouse coordinates when the mouse hovers over an interface, as opposed to when the mouse leaves the interface area as in the `OnLeave` method. `OnHover` and `OnLeave` are only called once per event as they do not deal with elements. That job is left to the `Update` method:

[PRE43]

After the mouse position is obtained, the `m_beingMoved` flag is checked to determine whether or not an interface is currently being dragged. If it is, and the saved position of the mouse is different to where the mouse is currently located, that difference is calculated and the interface's location is adjusted based on it. With that out of the way, let's take a look at the omitted chunk of code:

[PRE44]

We begin by checking if the current element needs to be re-drawn. The relevant flag for re-drawing the entire interface is set to `true` if one is encountered, while taking into account whether it's a control element or not.

When iterating over the list of all elements, their active status is checked. If an element is active, it gets updated. If the interface currently isn't being moved and the mouse is inside both the interface and the element, but not the title bar, the element's current state is checked. A `Hover` event needs to be dispatched and the `OnHover` method needs to be called if the element's current state is `Neutral`. However, if the mouse is not over the element, or the current interface's state is `Focused`, a `Leave` event is created and submitted, along with the `OnLeave` method being invoked.

Now, let's bring all of this hard work to the screen and render the interface:

[PRE45]

This is quite simple, thanks to our design involving three different render textures. In order to draw an interface successfully, the sprites for the background, content, and control layers have to be drawn in that specific order. If the `m_showTitleBar` flag is set to `true`, the title background must also be drawn along with the text.

Whereas the `Update` method does most of the work, moving interfaces requires a bit more preparation. Let's begin by defining two helper methods for movement, starting with the one used to initiate the process:

[PRE46]

If the conditions to move an interface are met, this method is invoked in order to save the mouse position at the point where dragging began.

We also have a simple line of code to stop interface movement:

[PRE47]

Since interfaces are quite different from normal GUI elements, they have to define their own way of obtaining their global position, as shown here:

[PRE48]

When it obtains its actual position, it needs to follow through the chain of parent interfaces and sum all of their positions. A `while` loop serves as a nice way of doing this; the final position is returned when it concludes.

The style application of an interface also differs from the usual element types. Let's take a look:

[PRE49]

The `ApplyStyle` method is invoked first because the parent class does a great job of setting up most of the visual components correctly. The background elements then need to be changed to have positions with absolute zero values because interfaces render these drawables to a texture and not to a screen. Regardless of the position of the interface, the positions of these elements will not change.

Next, the title bar background is set up to match the size of the interface on the *x* axis and should have a height of 16 pixels on the *y* axis. This hardcoded value can be tweaked at any time. Its position is then set to be right above the interface. The fill color of the title background is defined by the element color property of its style.

The last four lines set up the position of the title bar text and glyph. The position of the title bar background is summed together with the relevant padding to obtain the final position of these two attributes.

Rendering time! Let's draw all of these visuals onto their respective textures, starting with the background layer:

[PRE50]

Firstly, a check is made in order to be sure that the background texture is the same size as the current style dictates. If it isn't, the texture is recreated with the correct size.

The next line is extremely important for good looking results. At first glance, it simply clears the texture to the color black. If you look closely, however, you will notice that it has four arguments instead of three. The last argument is the **alpha channel**, or the transparency value for the color. The texture cleared to black appears as a large black square, and that's not what we want. Instead, we want it to be completely empty before drawing elements to it, which is what the alpha value of *0* will do.

Next, the `ApplyStyle` method is invoked in order to adjust the visual parts of the interface to match the current style. The background solid and the background image are then drawn onto the background texture. The texture's `display` method *must* be called in order to show all of the changes made to it, just like the render window.

Lastly, the background sprite is bound to the background texture and its visible area is cropped to the interface size in order to prevent overflow. The redraw flag is set to `false` to indicate that this process is complete.

A very similar process also needs to occur for the content layer:

[PRE51]

The content texture is checked for dimensions. The only difference here is that we're keeping a manual track of its size in the `m_contentSize` float vector, which will be covered later.

After the texture is cleared, we iterate over all of the elements inside the interface and check whether they are active or a control element. If all of these conditions are satisfied, the element's style is applied and it is rendered onto the content texture, which gets passed in as the argument of the `Draw` method. Its re-draw flag is then set to `false`.

After displaying the texture and binding it to a relevant sprite, it too gets cropped, except that, this time, we use the `m_scrollHorizontal` and `m_scrollVertical` data members as the first two arguments in order to account for scrolling. Consider the following illustration:

![Implementing the interface class](img/B04284_10_03.jpg)

Scrolling an interface means moving the cropped rectangle across the content texture. The `m_contentRedraw` flag then gets set to `false` to signify that the re-draw process has concluded. It leaves us with a result that looks like this:

![Implementing the interface class](img/B04284_10_04.jpg)

The final layer of the interface follows an almost identical path:

[PRE52]

The main difference here is that the texture is aiming to match the size of the current style, just like the background layer. Only the control elements are drawn this time.

The subject of interface scrolling keeps popping up, so let's take a look at how it is done:

[PRE53]

Both the horizontal and vertical adjustment methods take in a percentage value that tells the interface how much it should be scrolled. The actual amount of pixels an interface should be offset by is calculated by first dividing the difference of its content size on the relevant axis and the size of the interface itself by a hundred, and multiplying the result by the percentage argument. The texture rectangle is then obtained to maintain the proper width and height of the content area, which is then re-set with the scroll values as the first two arguments. This effectively simulates the scroll sensation of an interface.

Adding, removing, or manipulating different elements inside an interface may alter its size. Here's a method to solve those problems:

[PRE54]

Before examining it in depth, I can show you that, inside the class definition, this method looks like this:

[PRE55]

Its only argument has a default value of `nullptr`, which enables the method to detect size changes with or without a reference element.

If an element is provided as an argument, which usually happens when one is added to an interface, its bottom-right corner coordinates are calculated using its position and size. If these coordinates are somewhere outside of the content size boundaries, the content size is adjusted to be larger and the control redraw flag is set to `true` because the physical dimensions of the sliders will be changing. The method then returns in order to prevent the rest of the logic from being executed.

Without a reference element, a float vector is set up to keep track of the farthest point within the interface texture, the original value of which is the interface size. Every active non-control element is then iterated over and checked to see if it exceeds the furthest point in the texture, which simply gets overwritten on a relevant axis. If an element is found that pokes outside of these boundaries, its bottom-right corner position is stored and the control layer is marked for re-drawing. The content size itself is set to the farthest corner of the interface after all of the elements have been checked.

This final code snippet concludes the interface class.

# Summary

Just as a book without binding is simply a stack of papers, the code we've written doesn't become what it needs to be unless it is properly incorporated and managed. The groundwork we've laid down in this chapter will aid us greatly in implementing a fully functional GUI system but it only represents all the pieces being laid out.

So far, we have covered the basic design of GUI elements and windows, as well as implementing quite a few useful features that different types of elements can use. While that is a lot of code, we're not quite done yet. In the next chapter, we will be bringing all of the pieces we worked on together, as well as creating actual GUI elements. See you there!