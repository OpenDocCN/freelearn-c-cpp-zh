# Chapter 4. Have Thy Gear Ready - Building Game Tools

Making games is a fine art. It is entirely possible, of course, to make art with the most basic of tools, but most commonly, developers need a strong toolkit supporting them in order to efficiently and professionally create quick, painless edits to their game. Building the said toolkit is arguably on a par with the difficulty of building the actual game, but the work spent on proper tools offsets the difficulty and frustrations that come with direct file edits.

In this chapter, we're going to be covering these topics:

*   Building a graphical means of file management
*   File loading in a separate thread
*   Establishing a state and means of controls for map editing

There's lots of ground to cover, so let's get started!

# Use of copyrighted resources

As usual, let us give proper thanks to the artists and their assets that made this possible:

*   *Folder Orange* by *sixsixfive* under the **CC0** license (public domain): [https://openclipart.org/detail/212337/folder-orange](https://openclipart.org/detail/212337/folder-orange)
*   *Generic Document* by *isendrak* under the **CC0** license (public domain): [https://openclipart.org/detail/212798/generic-document](https://openclipart.org/detail/212798/generic-document)
*   *Tango Media Floppy* by *warszawianka* under the **CC0** license (public domain): [https://openclipart.org/detail/34579/tango-media-floppy](https://openclipart.org/detail/34579/tango-media-floppy)
*   *Close* by *danilo* under the **CC0** license (public domain): [https://openclipart.org/detail/215431/close](https://openclipart.org/detail/215431/close)
*   *Hand Prints* by *kattekrab* under the **CC0** license (public domain): [https://openclipart.org/detail/16340/hand-prints](https://openclipart.org/detail/16340/hand-prints)
*   *Paint Brush with Dark Red Dye* by *Astro* under the **CC0** license (public domain): [https://openclipart.org/detail/245360/Paint-Brush-with-Dye-11](https://openclipart.org/detail/245360/Paint-Brush-with-Dye-11)
*   *Primary Eraser* by *dannya* under the **CC0** license (public domain): [https://openclipart.org/detail/199463/primary-eraser](https://openclipart.org/detail/199463/primary-eraser)
*   *Mono Tool Rect Selection* by *dannya* under the **CC0** license (public domain): [https://openclipart.org/detail/198758/mono-tool-rect-selection](https://openclipart.org/detail/198758/mono-tool-rect-selection)
*   *Color Bucket Red* by *frankes* under the **CC0** license (public domain): [https://openclipart.org/detail/167327/color-bucket-red](https://openclipart.org/detail/167327/color-bucket-red)

# File management

The success and usability of the map editor tool is going to rely heavily on one specific interfacing element here, which is file access and management. In order to provide efficient means of file access, loading and saving, we are going to work on developing the means of visually guiding our user through the file system. The entire system consists of a few moving parts. For now, let us solely focus on the interface aspect of this idea.

## File manager interface

Before we can successfully work with any kind of map data, it is important to have a comfortable means of loading and saving. This can be offloaded to a file manager interface, which is going to be responsible for displaying directory information. Let us take a look at what ours is going to look like:

![File manager interface](img/image_04_001.jpg)

With this goal in mind, let us begin planning a class for it, starting with the header:

[PRE0]

Evidently, this class is a slightly more complex manifestation of a wrapper for a `GUI_Interface` instance. It is responsible for keeping track of the current directory we are in, as well as invoking a callback function/method when a file is selected to be loaded or saved. The callback function only takes a string argument, which carries the full path to the file that was selected to be loaded or saved, and can be registered like this:

[PRE1]

Nothing too complicated yet. Let us move on to actually implementing the class!

### Implementing the file manager

With the class definition out of the way, it is time to take a look at the actual code that makes the file manager tick. Let's start by implementing the constructor of this class:

[PRE2]

First, we load the interface and store its pointer in the designated data member. We also want to store the current state of the application, and obtain the style names of the elements, called `FolderEntry` and `FileEntry`, which are then removed. This makes the interface file a sort of template that gets filled in with all the right information later.

Once the appropriate content size and offset are set, the interface is positioned in the centre of the screen. We then subscribe to relevant GUI interface events and set our file manager directory as the current directory the application is in.

The callbacks and interfaces created in this class obviously need to be removed once they are no longer in use. This is where the destructor comes in:

[PRE3]

Next, it is important for the file manager class to have a way to easily change its current directory:

[PRE4]

A couple of interesting things happened. Right after the argument is stored; all of the backward slashes in the directory string are replaced with forward slashes, in order to maintain compatibility with multiple other operating systems that do not play well with the former. The interface is then instructed to destroy all elements it has that begin with the `""Entry_""` string. This is done in order to clear out all file and directory entries that may already exist. Finally, `ListFiles()` method is invoked, which populates the file manager with all of the files and folders inside the new directory. Let' us take a look at how that can be done:

[PRE5]

First, the `Directory` element is obtained in order to change its text. It represents the full path of the current working directory. The complete file list inside that directory is then obtained, including other folders. After it gets sorted alphabetically and by type, the parent directory element is obtained to calculate the starting coordinates of the first element on the list, which is, in turn, iterated over. Non-physical directories, such as `"."` or `".."`, are dismissed. A new element is then added to the interface, with an appropriate name that varies depending on whether we are working with a file or a folder. That same element is then updated to have the entry name, be in the right position and have a correct style attached to it. Finally, the *y* coordinate is incremented for the next element on the list.

With the directory structure being visually represented, let us take a look at what needs to happen when one of its entries is actually clicked:

[PRE6]

The first check here lets us know whether the item clicked on was a directory or a file. In case of a folder click, we want to be able to traverse the filesystem by taking its name and adding it onto our existing directory path. The vertical scroll of the interface is then set back to zero, in order to move the content back up to the top if any scrolling has been done.

A file click is a simpler matter. All we need to do in that case is obtain the text-field element that holds the filename, and change its contents to the name of the file that was just clicked on.

All of this works perfectly for forward-traversal, but what if we want to go backwards? The parent directory element helps us out here:

[PRE7]

Here, it simply comes down to basic string manipulation. The very last instance of the forward slash character is first attempted to be located inside the directory string. If one is found, the string is simply *clipped* at that point, in order to drop everything that comes after it. The shortened path is then set as the current directory, where the rest of the magic that we've already covered happens.

The last piece of the puzzle in making this work is handling the button press action:

[PRE8]

First, we need to make sure the action callback is actually set. If it is, it gets invoked with the path to the currently selected file as its argument. The only other action button we have to worry about after this point is the close button:

[PRE9]

It simply invokes the `Hide()` method, which is covered here, along with its counterpart:

[PRE10]

When an interface is hidden, it's simply set to inactive. Showing it requires setting it back to being active, except we also want to position it in the absolute centre of the screen in this instance. In addition to that, it is a good idea to refresh its content, as the file structure may have changed while it was hidden. Lastly, the interface is focused, in order to bring it to the front of the drawing queue.

The final bits of helpful code for this class consist of these methods:

[PRE11]

They help our other classes interface with this one more easily, by allowing them to determine if the file manager is in `Save` or `Load` mode, and to switch between the two.

## Loading files in a separate thread

We have covered the threaded worker base class previously in [Chapter 1](ch01.html "Chapter 1. Under the Hood - Setting up the Backend"), *Under the Hood - Setting up the Backend*. This is exactly where it will come in handy. In order to make the application seem more user-friendly, we want to render a nice loading bar that shows progress while the files are being loaded. Let us start by first defining a data type, used to store file paths that need to be loaded:

[PRE12]

The `size_t` here represents the number of lines that are in the file, which makes it easy for us to determine the current loading progress. With that out of the way, let us work on the header file:

[PRE13]

Any `FileLoader` class in our code base needs to implement the `ProcessLine` method, which simply defines what needs to happen as each individual line of the file is being parsed. If necessary, it can also take advantage of `SaveToFile`, which, as the name states, defines the process of writing the class data out, and `ResetForNextFile`. The latter method is invoked after every file that has finished loading, in order to give derivative classes a chance to clean up their internal state.

As far as data members go, we have a list of loader paths that are to be loaded, the number of total lines of all files that are supposed to be parsed, and the number of the current line being read.

### Implementing the file loader

Let us start simply, and cover the one-liner methods first:

[PRE14]

The constructor simply initializes a few of the class data members to their default values. The `AddFile()` method inserts the argument to the file container with the line count *zero*. The next two methods are simple getters, while the last two are not even implemented, as they are optional.

Next, let us work on the method that will actually be running in a thread and parse the file information:

[PRE15]

First, a private method for counting all file lines is invoked. This is necessary, as we want to be able to calculate our progress, and knowing how much work there is in total is needed for that. If after this method is called, the total number of lines is zero, we simply return as there is nothing to process.

We then enter a loop that runs once for each file on the list. The class is reset for new file iteration, and a line from the input stream is created. The number of lines remaining to be processed is created, and another loop is entered that will execute once for each line in the file. Our `sf::Mutex` object is then locked in order to safely manipulate the two line data members that are used for progress tracking.

If the first character of our line is a pipe, `|`, it means we ran into a commented line and should just skip the current iteration. Otherwise, an `std::stringstream` of the current line is created and passed into the pure virtual `ProcessLine()` method, which is encapsulated in an `if` statement to catch a possible failure, in which case the remainder of lines inside the current file are simply added to the current line counter and the loop is broken out of.

Once the processing of all files is complete, the `Done()` method is invoked in order to terminate the thread and let the outside code know we've finished.

Another equally as important process is counting the lines of all file entries inside this class:

[PRE16]

This one is fairly straightforward. After the two counters are zeroed out, we begin iterating over each path inside the file list. If the name of it is empty, the element is removed. Otherwise, we attempt to open the files. If that fails, the path is also erased. Otherwise, the file stream is requested to not skip whitespaces, and we enter a `sf::Mutex` lock, where the number of lines in the file stream is calculated using `std::count`, and added to the total line counter. The path iterator is then moved forward, and the file is closed.

## The loading state

The last piece of the threaded file loading puzzle is the loading state. In order to avoid other logic going on and simply focus on the graphical progress representation, it's a good idea to just switch to a dedicated state that will handle all loading logic inside it. Let us begin by defining a data type for holding pointers to `FileLoader*` instances:

[PRE17]

The actual loading state header will end up looking something like this:

[PRE18]

As you can see, we have one event callback method, a couple of helper methods, the container for loader pointers, an instance of `sf::Text` and `sf::RectangleShape` to represent the loading bar, a number to represent the progress percentage, and the number of lines inside all files we originally started with.

### Implementing the loading state

All of this data needs to be initialized before it being used, so let us take a look at the `OnCreate()` method:

[PRE19]

Because we are going to be using text, we need to have a font to work with. After one is acquired and all of the stylistic text settings are handled, we set up the rectangle to be exactly in the centre of the screen and register an event callback for proceeding out of the loading state, if the manual continue flag is set to `true`.

Destroying this state also means the event callback and the font need to be released:

[PRE20]

Next, let us take a look at the updated logic:

[PRE21]

First, a check is made to determine if we are ready to exit the state, given all of the work that has been done. If the manual continue flag is set to `false`, we simply invoke the Proceed callback directly by passing `nullptr` as the `EventDetails` pointer, since it is not used there anyway. The update method is then returned from.

If we still have some work to do, the first element on the loader list is checked to see if it's done. If it is, the loader is removed, and if it was the last one, the size of the rectangle is set to match the full size of the window on the x axis, which shows full completion. The text in the middle is also updated to let the user know they need to press the spacebar key to continue. Finally, the update method is returned from once again, to prevent further logic from executing.

If none of those conditions were met, the first element on the loader list is checked to see if it has started its work. If it has not yet started, its `Begin` method is invoked. This is quickly followed by the percentage calculation, which is then used to update the text in the middle of the screen and adjust the size of the progress bar rectangle to match said percentage.

Drawing in this state simply comes down to two calls:

[PRE22]

All we need to do here is render the rectangle and the text instances.

Next, let us take a look at the helper method that updates our text instance:

[PRE23]

After the text string is updated, its position is updated to be directly in the middle of the screen. Since updating its contents may change the bounding box, and thus how it is centered, a helper function inside our `Utils` namespace is used to center it properly.

Next, let us calculate the actual progress of the loading procedure:

[PRE24]

After an absolute value of `100.f` is created, the current progress is calculated by first determining the progress of how many files have been already loaded out of the number we began with, followed by the progress of the current file being calculated and used to determine absolute progress, which is then returned.

Once all of the work is done, the `Proceed()` method is invoked to return to the previous state:

[PRE25]

Obviously it needs to check if the list of file loaders is actually empty first. If it is, the state manager is instructed to switch to the state that comes just before this one, which means it is the one that initiated the loading procedure.

Finally, what would a class be without some helper methods? Let us take a look at them now:

[PRE26]

# Creating the map editor state

Now we're finally ready to actually tackle the state, in which all of the map editing is going to take place. Let us take a gander at its header file:

[PRE27]

This `State_MapEditor` class is going to be the frontier that deals with the most general editor events. Note the highlighted data member here. We have not yet covered this class, but it is responsible for handling the finer aspects of control for this application. It will be covered in the next chapter.

Aside from the `MapControls` class, we also have the file manager, a string for the path to a file that is currently being worked on, and a *boolean* flag that keeps track of whether the game map should be redrawn or not.

## Implementing the state

As always, let us begin by tackling the construction of all the important data inside this state:

[PRE28]

After all of the event callbacks are set up, the file manager class is provided with its own callback for either loading or saving a file, as well as the starting directory it needs to be in. In this case, appropriately enough, it is the maps folder. The manager is then hidden, and another interface is loaded and positioned on screen. `MapEditorTop` is the control strip on the very top of the screen that has buttons for creating a new map, loading, saving, and exiting the application:

![Implementing the state](img/image_04_002.jpg)

Once the state is finished and is about to be destroyed, it needs to remove all call-backs that it set up. This can be done in the `OnDestroy()` method:

[PRE29]

In addition to callbacks, the map is also purged just before its size is set back to absolute zero. Since we are on the subject of callbacks, let us just cover most of them in a single go:

[PRE30]

When the **New** map button is clicked, we want to invoke a special method of the `MapControls` class that will handle it. If the **Load** button is clicked, we simply switch the mode of the file manager to load, and show it on screen.

Clicking the **Save** button can have two behaviors. First, if we are dealing with a fresh, new map that has not been saved yet, it is the same as clicking the **Save As...** button, which switches the file manager to save mode and shows it on screen. However, if we have loaded a map or have previously saved a new one, the state remembers where it was saved, as well as its name. Prompting the user to enter a filename again would be pointless here, so the map is simply written to the exact same location, with the exact same name.

Finally, if the **Exit** button is clicked, we simply switch back to the main menu state and remove this one.

With the UI code out of the way, let us take a look at what needs to happen when a map is being loaded:

[PRE31]

Since we want a nice loading bar to appear as a map is being read in, we are going to be using the loading state. After it is obtained, both the particle system and the map are purged. The map, which inherits from the `FileLoader` class, is then reset. The file path that was provided as an argument is then added to it to be loaded, and the loading state itself is set up to automatically dismiss itself once the loading is done. At the same time, we make sure that the map is going to be re-drawn as the map editor state resumes, and that it remembers the path of the map if it is to be saved later. Finally, we can switch to the loading state.

Next, let us work on the code that is responsible for saving the map:

[PRE32]

This is much simpler than the previous method. The path is simply passed to the `SaveToFile` method of the game map class, and stored for later use.

The actual callback of the file manager that mediates between the load and save methods can be implemented like :

[PRE33]

Depending on the mode the file manager is in, the appropriate method is called with the path being passed in as the argument. The actual interface is then hidden.

Because we want to re-draw the map after it was loaded, the perfect place for that logic is inside the `Activate()` method, as it gets called right when a state is switched to:

[PRE34]

If the `m_mapRedraw` flag is not on, there is no need to do anything at this point. Otherwise, we want to redraw the map and provide the `mapControls` class with the tile-sheet texture name, so that it can perform its own logic, such as, for example, tile selection.

Next, let us take a look at what needs to be updated while the application is in this state:

[PRE35]

Alongside the `mapControls` class, the game map, ECS system manager, and the particle system also need to be updated, because we are going to be using all of these classes while building maps. Predictably enough, these are the same objects that also need to be drawn:

[PRE36]

Note the `from` and `to` variables. The `mapControl` class is going to provide us with a way to switch between layers/elevations, so we need to obtain that information before anything is rendered, in order to make sure only the appropriate layers are drawn on screen. `DrawSelectedLayers` simply returns a *boolean* value that determines whether or not all layers should be drawn, or just the selected ones. Once the loop has iterated over the appropriate elevations, we make sure to draw the remaining particles that are above the maximum elevation, provided, of course, everything needs to be rendered. This is topped off by the map controls being drawn over everything else.

For other outside communications with this class, we provide two basic setter methods:

[PRE37]

These are going to be used inside the control classes to communicate events, such as a new map being created, or it needing to be re-drawn.

# Building the control mechanism

While building maps, the user tends to run into situations where they need more than just a tile being placed where the mouse clicks. It would definitely be useful to have tools that would enable them to freely pan around, select chunks of the map for deletion or copying, erase them and so on. Our control class is going to serve this exact purpose. It will provide a set of tools that can be used for multiple different situations:

[PRE38]

The preceding control mode enumeration represents a couple of the most common tools that come in a variety of different pieces of software. We're going to implement some of them here, and leave the rest up to you! In the end, we should have a control interface that looks a little like this:

![Building the control mechanism](img/image_04_003.jpg)

Let us get to actually writing out the header for the control class. For clarity, we are going to be discussing its methods and data members separately, starting with the member functions:

[PRE39]

In addition to all of the helper methods for setting and getting the class parameters, we have a whole bunch of event callbacks, as well as individual update methods for every kind of map tool we're going to be working with. Next, let us take a look at the data members we are going to be working with:

[PRE40]

Alongside the `ControlMode` that this class is currently in, we are also going to be storing a couple of flags. The `m_action` flag will be used with tools, as well as `m_secondaryAction`. The former simply denotes whether the left mouse button is pressed or not, while the latter is used with an action that can only happen once the mouse position has changed. This will prove useful when we are trying to optimize certain things to not happen, unless they absolutely have to. The last two flags signify whether we are currently right-click panning, and whether only the selected layers should be drawn on screen.

Below that, there are a couple of 2D vectors, used to store mouse information, such as its current position, where a left-click first happened, the difference between the current and last frame in the mouse position, its current position in tile coordinates, and its starting position in tile coordinates. Additionally, we also have a floating point value for the current zoom factor.

For the brush that will be used to paint with, we simply use a `TileMap` structure, just like the game map class does. Since the brush is going to have to be drawn on screen, we need to store a texture for it, as well as another drawable object that will be used to show it. Finally, a `sf::RectangleShape` type is going to more than suffice for showing the boundaries of the map on screen.

Additional code separation, especially when code is becoming quite lengthy, is always a good idea. For this purpose, other non-general-control logic is going to be spread out into two additional interface classes: a tile selector, and the selection options. A tile selector is a simple window that shows the entire tile-sheet and allows the user to select tiles they want to paint with, while selection options is a separate interface that provides us with a myriad of settings that can be tweaked when specific things on screen are selected. Both of these classes are going to be covered in the next chapter.

Lastly, we have another interface, named `m_mapSettings`, the logic of which is going to be handled within the `MapControls` class. When creating new maps, we need a neat little window that is going to allow us to configure the size of the map, its default friction value, and the name of the tile-sheet it is going to be using. This is exactly the purpose the map settings interface is going to serve.

## Implementing controls

There are quite a few data members to initialize, so let us take a look at how the constructor manages it:

[PRE41]

As you can see, there is quite a lot going on here. Let us zip through it section by section. Right after the arguments of the constructor are processed, we set up the data members of this class to hold their initial values. Shortly after that, the custom interface classes get set up, with all the necessary arguments being passed to their constructors. For now, we are not going to be worrying about them, as they will be covered in the next chapter.

Let us take a look at the actual constructor body next:

[PRE42]

Right after all event callbacks get set up, we begin working on the interfaces. The actual tools interface is loaded and positioned on screen, as well as the new map settings window, which we are going to keep track of by storing its pointer as one of our data members. It gets positioned in the centre of the screen and set as inactive for the time being.

The next segment simply deals with the stylistic aspects of the brush drawable, as well as the map boundaries rectangle. These values can obviously be customized to look completely different.

Lastly, we need to make sure to populate the drop-down element for sheet selection inside the new map settings interface. After the element is obtained and cleared of all other entries, the list of all filenames that are of type `.tilesheet` inside the appropriate location is obtained and iterated over, stripping away the file format from each one and adding it to the drop-down list, which is then re-drawn to reflect all changes.

Keep in mind that all interfaces and callbacks that were created here need to be removed, which is all that happens in the destructor. For that specific reason, we are not going to be covering that here, as it is redundant.

Let us take a look at what needs to happen when this class is being updated:

[PRE43]

First, we handle any possible changes in size of the map class. The map boundary rectangle is updated here to reflect them. Next, we must make sure the mouse is updated properly. All of that logic is contained within the `UpdateMouse` method, which is invoked here. Finally, depending on the current `ControlMode`, we need to invoke the appropriate update method for the specific tool that is selected. The pan tool is special in a way, because it will be updated when it is selected as a tool, and when the right mouse button is being pressed as well.

Drawing all of these objects may be simpler than you think:

[PRE44]

In this specific instance, all we need to render is the rectangle of the `mapBoundaries`, the brush, if the `ControlMode` is set to `Brush`, and the `SelectionOptions` class, which has its own `Draw` method. More on that will be covered in the next chapter.

Next, let us implement everything necessary to keep track of all the relevant mouse information:

[PRE45]

After the current mouse position is obtained, it is used to compute the coordinate difference between the current frame and the previous one.

### Note

Since the mouse difference is expressed in **global coordinates**, we must remember to multiply them by the *zoom factor*.

The mouse position is then stored for the next frame, so this process can take place all over again. The current `sf::View` is then obtained for calculating the current **global** position of the camera. From this, we can calculate the global mouse position (adjusted for zoom, of course), and the mouse tile position, which is simply the tile that's being pointed at.

The current mouse tile position is then checked against the calculated result for being different. If it is, and the left mouse button is currently being pressed (as shown by the `m_action` data member), the secondary action flag is turned on. The mouse tile position is then stored for the next frame.

The next method in the mouse variety deals with left and right clicks, and can be implemented like so:

[PRE46]

Because something else may have already processed a mouse event, we need to check for the event details that get submitted as an argument. We do not want to accidentally paint some tiles on the map if we are simply interacting with an interface, for example. Next, the key code of the event is checked to see whether it is the left mouse button. If it is not, all we need to worry about is setting the right-click pan flag to `true` and returning.

If we indeed have a left-click, on the other hand, the current mouse position is stored as both the starting and the current positions. A very similar process to updating the mouse takes place here, leading to the calculation of the global mouse coordinates. They are then passed into the `MouseClick()` method of the selection options class, which returns a *boolean* flag, signifying whether any entities or particle emitters have been selected. We will be dealing with that in the next chapter. If that is not the case, however, both the action and secondary action flags are set to `true` in order to use the currently selected tool.

In the same way that for every action there is an equal and opposite reaction, for each click we need to have a release:

[PRE47]

All we need to worry about here is resetting all of the action flags that are used while the mouse is active. This includes the right-click panning, and both action flags. The selection options interface also needs to be notified of a release.

A neat little feature that is going to help out a lot is being able to zoom in and out. It is handled here as an event:

[PRE48]

If this event has not already been processed by something else, we proceed to calculate the amount of zoom that needs to happen. A `float factor` value is defined here, and is multiplied by the change in the mouse wheel position. In order for it to be treated as a scale factor, it is subtracted from `1.f`, and then used to zoom in the view. Finally, in order to keep track of the current zoom value, we must multiply it by the said scale factor.

The next event we need to worry about is one of the tools being selected:

[PRE49]

This is quite simple, as we basically map the names of elements to their `ControlMode` counter-parts. The appropriate mode is then selected on the bottom.

Speaking of tools, each one of them has their own individual update method. Let us begin by taking a look at how the pan tool is updated:

[PRE50]

We obviously do not want the screen to move if the mouse is not being clicked, or if the mouse position delta between frames is absolute zero. Given that both those conditions are satisfied, however, all we need to do is move the centre of the view to a different location. This location is calculated by adding its current position with the mouse position difference, which has to have its sign flipped. We do this, because as the mouse is clicked and moved left, for example, the view needs to shift right in order to feel natural. The same is true for the *x* axis.

In the case of a brush tool, the logic goes like this:

[PRE51]

First, the global position of the tile the mouse is over currently is calculated, which the brush drawable is set to match. Doing it like this creates a feel of the brush being locked to a grid. Another method is then invoked for placing the tiles:

[PRE52]

The first and most obvious check here is to make sure that both the primary and secondary actions are on. We do not want to be placing tiles if the mouse is not being clicked, or if it already has been clicked, but is still at the same location. Otherwise, we are good to go on painting, which begins by the brush tile map being placed on the game maps tile map at the current mouse tile position, starting at the lowest layer currently selected by the selection options. Even though we may be able to shift through elevations at ease, we still need to tell this method about the lowest current elevation selected, because the brush tile map itself still begins at elevation *0*.

After the map has been updated, the tile coordinate range to be redrawn is calculated and passed to the `MapControls` class to be rendered on screen. We do not want to re-draw the whole map, as that would take more time and introduce latency. Lastly, the secondary action flag is set to `false` in order to indicate that a placement has been made at these coordinates already.

The next tool we need to update is the selection box:

[PRE53]

As you can see, all of that logic is handled by the `SelectionOptions` class. For now, we simply need to worry about invoking this method

The same `SelectionOptions` interface may be responsible for manipulating our brush, which means we need to have a method for redrawing it to reflect changes:

[PRE54]

First, the real pixel brush size is calculated from the size of its tile map. If it does not match the current dimensions of the texture that represents it, the texture needs to be re-created. Once that is taken care of, the texture is cleared to all transparent pixels, and we begin iterating over each tile and layer inside said brush. Given it is a valid tile that has proper ties to an information structure that holds its sprite for rendering, the latter is set to the correct position on the texture and drawn to it.

Once this is done, the texture's display method is invoked to show all the changes, and the drawable object of the brush is bound to the texture again. The drawables size and texture rectangle is also reset, because the dimensions of the texture could have changed.

In this type of an application, it's important to have a quick and easy way of deleting something that's currently selected. For this, we're going to be processing the event that's bound to the *Delete* key on your keyboard:

[PRE55]

This is a very simple callback. It simply checks if the current `ControlMode` is selected, and passes its details to another callback that belongs to the `selectionOptions` class. It will be dealing with all removals.

When a new tool is being selected, we must reset all data members we work with to their initial values in order to avoid weird bugs. This is where the `ResetTools()` method comes in:

[PRE56]

It simply resets certain mouse data to a default uninitialized state. The `m_selectionOptions Reset()` method is also invoked, so that it can deal with its own resetting. Lastly, the `tileSelector` interface is hidden here as well.

Another useful little method is for resetting the zoom of the current view to a normal level:

[PRE57]

By dividing `1.f` by the current zoom factor, we obtain a scale value, which, when scaled by, the view returns to its normal state.

Next, let us see what needs to happen in order for this class to change its `ControlMode`:

[PRE58]

After the tools are reset, the mode passed in as the argument is stored. If the mode being applied is a brush, it needs to be re-drawn. Lastly, the `selectionOptions` class is notified of the mode change, so that it can perform its own logic.

Finally, one of the last key pieces of code is the creation of a new map:

[PRE59]

First, we obtain the size values from the text-fields of the map settings interface. In addition to that, we also grab the friction value, as well as the current selection of the tile-sheet drop–down menu. If the latter is empty, we simply return, as no tile-sheet has been selected.

If we do proceed, the particle system and the map both need to be purged. The `MapEditor` state is then notified to reset its save path, which forces the user to re-enter a filename when saving.

The map's size is then set up, alongside the default friction value. The selected tile-sheet file is added for further loading in a separate thread, and its name is registered inside the game map's internal `TileSet` data member.

Finally, the loading state is obtained, the tile-set is added to it, and the manual continue flag is set to `false`, in order to make the loading screen simply go back to the current state after it is done. The new map settings interface is then hidden, and we can finally switch to the loading state.

In case a mistake happens, the user must have a way to close the new `m_mapSettings` interface:

[PRE60]

This callback gets invoked when the **close** button of the interface is pressed. All it does is simply hiding it.

Finally, we have a bunch of setters and getters that do not add up to much on their own, but are useful in the long run:

[PRE61]

You may have noticed that we have not yet covered the bucket and eraser tools. This is what is usually referred to as homework, which should serve as good practice:

[PRE62]

Keep in mind that as we have not yet implemented everything that makes the map editor tick, this should probably wait until the next chapter is wrapped up.

# Summary

In this chapter, we have introduced and implemented the concept of graphical file management, as well as laid the foundations for one of the most important tools a small RPG-style game uses. There is still a lot left to do before we can start reaping the benefits of having proper tools. In the next chapter, we will be covering the finishing touches of the map editor, as well as implementing a different tool for managing entities. See you there!