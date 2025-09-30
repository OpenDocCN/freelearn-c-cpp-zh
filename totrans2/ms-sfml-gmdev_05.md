# Chapter 5. Filling the Tool Belt - a few More Gadgets

The last chapter established a firm ground for us to build on. It is time to take full advantage of it and finish what we started, by building a robust set of tools, ready to take on a wide variety of design problems.

In this chapter, we are going to be covering these topics:

*   Implementation of selection options
*   Design and programming of a tile selection window
*   Management of entities

There is quite a lot of code to cover, so let us just jump into it!

# Planning the selection options

Versatile selection options are important when creating a responsive and useful application. Without them, any sort of software can feel unintuitive, clunky, or unresponsive at best. In this particular case, we are going to be dealing with selecting, copying, and placing tiles, entities, and particle emitters.

Let us see what such an interface might look like:

![Planning the selection options](img/image_05_001.jpg)

In order to get there, we need to create a flexible class, designed to be able to handle any possible combination of options and controls. Let us start by going over the most basic data types that are going to come in handy when developing this system:

[PRE0]

First, the selection mode needs to be enumerated. As shown in the preceding snippet, there are three modes we are going to be working with at the moment, although this list can easily be expanded in the future. The `NameList` data type is going to be used to store the contents of entity and particle directories. This is simply the return format for the utility function we are going to be relying on.

With the data types out of the way, let us take a stab at creating the blueprint of our `SelectionOptions` class:

[PRE1]

In order to keep things simple, let us focus on talking about the methods we need first, before covering data members. As far as public methods go, we have pretty much the assortment anyone would expect. Alongside the `Show()` and `Hide()` methods, which are going to be used to manipulate the interface this class encapsulates, we pretty much only have a few setters and getters, used to manipulate the `ControlMode` and `SelectMode`, select specific entities or particle emitters, and obtain tile selection ranges, as well as the range of layer visibility/selection. Additionally, this class also needs to provide plenty of callback methods for numerous controls of the interface we are working with.

The private methods mainly consist of code used to update the interface and the visual representation of its selection onscreen, as well as methods for updating each possible mode the selection interface can be in. It's topped off by a private method `DeleteSelection()`, which is going to be useful when removing tiles, entities, or particle emitters.

Finally, let us take a gander at all of the data members that are going to be used to preserve the state of this class:

[PRE2]

We start by storing the current selection mode, alongside the `RectangleShape` object, used to visually represent the selection being made. In order to make our tools feel more responsive and lively, we are going to be providing a number of different colors, used to represent different states of selection. For example, the `m_selectStartColor` and `m_selectEndColor` data members are used to differentiate the tile selection that's still being made, and its final state, when the mouse button is released. In addition to colors, we also have two vector types that store the tile selection range for both axes, and a *boolean* flag, used to determine when the rectangle shape should be updated.

For the other two states, we need to store the entity identifier and its position component, given we are in an entity selection mode, and a pointer to the particle emitter, provided we are currently working with particles. This is also where the contents of particle and entity directories are going to be stored, in order to populate the drop-down list with the appropriate values.

Additionally, we need to keep track of the layer selection range, as well as pointers to the `selectionOptions` interface, the `MapControl` class that was covered in the previous chapter, and a map tile selector class, which will be covered shortly. Keep in mind that only the `m_selectionOptions` interface is technically owned by this class. The other two classes encapsulate their own interfaces, thus managing their destruction.

Lastly, we need to have access to the `eventManager`, `guimanager`, the game `map` instance, the tile `brush`, entityManager, and the `particleSystem`.

## Implementing selection options

With all of this data to properly initialize, we have quite a lot of work to do in the constructor:

[PRE3]

After all of the arguments are properly stored away, the default values of all data members are set up. This ensures that the initial state of the selection is defined. The body of the constructor is used to appropriately deal with other tasks:

[PRE4]

Here, all of the proper callbacks are set up, the interface the class owns is loaded, positioned and hidden, and the color values are initialized. Finally, the contents of the entity and particle emitter directories are obtained and stored.

We're not going to be covering the destructor here, because it simply deals with removing all callbacks and the interface that is set up.

Speaking of interfaces, the outside code needs to be able to easily show and hide the `selectionOptions` window:

[PRE5]

The desired effect is achieved by either setting the interface as active or inactive. In the former case, the `guiManager` is also used in order to position the `selectionOptions` interface above everything else, by bringing it to the front.

Because this interface/class is a sort of helper, it depends on the control mode of our editor. This relationship requires the `selectionOptions` class to be notified of `controlMode` changes:

[PRE6]

It's only necessary to worry about the `Brush` and `Select` modes, as this interface is not even needed for anything else. In case a `Brush` is selected, the interface is enabled and focused, while its `TileSelect` element is also enabled. This ensures we can select tiles we want to paint with. If the selection tool is picked, we want the buttons for solidity toggling and selection copying to be enabled instead.

The actual selection mode switching needs to be handled too, and can be done like so:

[PRE7]

First, the `Reset()` method is invoked. It is used to disable all unnecessary interface elements and zero out the selection data members to their default values. After the actual selection mode is stored and the interface is set to active, we begin dealing with the actual mode-specific logic.

If we are in the tile selection mode, it simply involves enabling a number of interface elements, as well as setting their text to match the context. For the sake of simplicity, all of the element manipulation in this method is omitted.

Dealing with entity and emitter modes is similar, yet includes an additional step, which is populating the drop-down menu with appropriate values. In both cases, the drop-down element is obtained and purged of its current entries. The appropriate directory list is then iterated over; adding each entry to the drop-down, making sure the file type is removed. Once this is done, the drop-down menu is instructed to be re-drawn.

Let us take a look at what needs to happen when our selection options class is instructed to select a particular entity:

[PRE8]

First, the argument could be used to de-select an entity, as well as select it. If the appropriate de-select value is passed, or an entity position component with the provided identifier has not been found, the related interface elements are adjusted to match the situation.

If an entity with the provided ID does exist, the proper elements are enabled and adjusted. The entity position component as well as its identifier is stored for later use, and the information text element of the selection options interface is changed to reflect the ID of the entity selected. It is also marked to be updated, by manipulating the *boolean* flag `m_selectUpdate`.

A very similar process takes place when selecting an emitter:

[PRE9]

It is simpler in a sense that we are only working with a pointer to a particle emitter. If `nullptr` is passed in, proper elements are disabled and adjusted. Otherwise, the interface is updated to reflect the information of the emitter that is selected, while also marking the `selectionOptions` interface is properly updated afterwards.

We obviously also need a way to switch between the different selection modes, hence this callback:

[PRE10]

It simply cycles through all of the options for selection. One thing worthy of pointing out here is that if the interface before cycling is in tile mode, we want to make sure that the `ControlMode` is switched to `Select`.

Another feature we want to work on is opening up and dealing with tiles being selected from the tile-sheet:

[PRE11]

First, we deal with just opening the `tileSelector` interface, provided it is not set to active yet. On the other hand, if the interface is open, the select button being pressed indicates the user attempting to copy their selection to the brush. The `mapControls` class is instructed to switch its mode to `Brush`, which is then passed into the `tileSelector` class's `CopySelection()` method, responsible for copying actual tile data. Since it returns a *boolean* value that indicates its success, the method is invoked inside an `if` statement, which allows us to update the solidity element of the interface and request a brush re-draw, provided the copying procedure was successful. At any rate, the information text element of the `selectionOptions` interface is then updated to hold the total count of tiles that have been selected and copied to the brush.

Toggling the solidity of the current portion of the map being selected or the brush itself is also possible in our tile editor:

[PRE12]

First, we obviously can not toggle the solidity of a selection, if the control mode is not set to either the `Brush` or `Select` mode. With that being covered, the solidity state label is obtained, as well as its text. After flipping its value to its opposite and updating the element's text, we establish a range of tiles that will be modified. In the case of a brush having its solidity toggled, the range encapsulates the entire structure. On the other hand, the map selection range is used when dealing with the select mode.

### Note

The `m_selectRangeX` and `m_selectRangeY` data members represent the selection range of the map tiles. Each range is responsible for its own axis. For example, `m_selectRangeX.x` is the **starting ** *X* coordinate, and `m_selectRangeX.y` is the **ending** *X* coordinate.

After the range is properly established, we simply need to iterate over it and obtain tiles from the appropriate `TileMap`, setting their solidity to the appropriate value.

Copying a certain portion of the map to the brush could also prove to be a useful feature:

[PRE13]

We begin by checking if a selection actually was made, which can be done by checking any of the select range data members. Afterwards, the size of the selection is calculated by subtracting the start-points of the selection from the end-points, and increasing the size by one unit on both axes. This is done in order to compensate for inclusive ranges that start and end on the same exact tile number.

Once the brush tile map is purged and resized, some local variables are set up in order to aid the rest of the code. The three *unsigned integers* are going to be used as index coordinates for the brush tile map, in order to map the copied tiles correctly. The two *boolean* flags and the *unsigned short *changes are going to keep track of solidity changes, in order to update the GUI element that denotes what solidity state the selection is in.

Next, the tile loops are entered. After the map tile at the specific coordinates is obtained and passes the validity check, the brush tile at the current coordinates denoted by `b_x`, `b_y`, and `b_l` is set to hold the same tile ID. The solidity changes of the tile are then detected and noted, in order to determine if we have a mixed selection of solidities. Finally, all other tile properties are transferred to the brush, by utilizing the overloaded `=` operator.

In order to keep the interface up–to–date with our actions, the current layer selection range is checked for exceeding the actual range of total layers supported by the application. If, for example, we support four total layers and the current selected layer is two while the brush has all of its layers filled, we want to adjust the current layer selection to honour that by calculating the layer difference, adjusting the highest layer selected to match the maximum layer supported by the application, and subtract the difference from the lowest layer, hence preserving the proper range of the brush.

Lastly, a method for updating the selection options elevation selection text is invoked, the map controls class is instructed to switch to the `Brush` mode, and the selection options interface is updated with the information of the brush tile count and solidity.

Let us drift away from the topic of placing, editing, or copying tiles for a second, and talk about actually placing entities or emitters when the **Place** button is pressed:

[PRE14]

We are not going to be using this functionality to do anything with tiles, because that is what the mouse is designated for. If the `selectionOptions` interface is in the proper `Select` mode, the value of the drop-down menu is obtained and checked for not being empty. The **Place** button can also act as the **Edit** button under appropriate circumstances, such as when an entity or particle emitter is selected, so in both cases, the appropriate values are checked for representing a selection, or lack thereof. If nothing is selected, the drop-down value is used to add a new entity or emitter of the selected type. The `SaveOptions()` method is then invoked, so in either case, the information currently stored in the `selectionOptions` interface is saved to either the newly created object, or one that was already selected.

Pressing the **Remove** button can be handled like so:

[PRE15]

As you can see, a different method is invoked here, with a *boolean* flag being passed to it, denoting whether the *Shift* key is being held down, controlling how much of the current selection is removed. Let us take a look at the actual delete method:

[PRE16]

Once again, we deal with all three different selection types: tile, entity, and particle emitters. If we are working with tiles, the selection range is checked. Provided something actually is selected, the layer range is defined, based on whether the argument says everything should be deleted. The map is then instructed to remove the tiles and cleat its render texture within the calculated ranges.

In the cases of entities and particle emitters, it's much less complicated. The selected entity/emitter is simply removed, and the appropriate `SelectX` method is invoked shortly after, passing in a value for nothing being selected.

Next, let us handle the ***+*** and ***-*** buttons that control the elevation selection being pressed:

[PRE17]

Here, we want to handle the button clicks in a specific way. Keep in mind that support for selecting ranges of layers is also something of great importance. Consider the following illustration:

![Implementing selection options](img/image_05_002.jpg)

Simply clicking either a plus or a minus would affect the low number, which represents the lowest elevation selected. Holding a *Shift* key would increase the high number, controlling the highest elevation. For this, two integers, `low` and `high`, are set up, alongside a *boolean* flag that determines if a *Shift* key is being held or not. Based on that and the event name, the numbers are adjusted to represent the changes in elevation.

Next, we branch out the logic once again. If a `Brush` mode is selected, we do not want to deal with any changes of the high elevation at all. Instead, only the low layer selection is used here. After a new value for it is established by adding the layer delta to the already selected low elevation, the range is checked for exceeding the boundaries of [0;`Sheet::NumLayers`). Provided that passes, the low elevation selection is updated with the new value, as is the high value, which simply takes the low elevation and adds the thickness of the brush to it, represented by the brush's highest elevation.

The `Select` mode follows the same basic principle, with one exception: it also handles the high elevation. With the deltas properly added to the current values, the range is checked for exceeding the allowed limits. The next check deals with how we control shift-clicks depending on whether both the low and high values are the same. If they are, the deltas are simply added to the low value, which is copied over to the high elevation, preserving the equality. Otherwise, both low and high values are simply overwritten with the preceding newly calculated range.

In both cases, it is also important to invoke the `SelectionElevationUpdate()` method, which makes sure the interface elements are kept up-to-date, like so:

[PRE18]

After making sure the selection options interface is actually active, the elevation label is updated with the proper layer range. The `SaveOptions()` callback is then invoked with `nullptr` for its argument. It is responsible for actually saving the interface's information to whatever object happens to be selected. Let us take a look at this method now:

[PRE19]

The most obvious first check is to make sure we are not in tile mode, because there is nothing to save there. Afterwards, the values from the text-fields representing *X*, *Y*, and *Z* coordinates are obtained and converted to numbers. This is where our logic branches out once again.

In the case of dealing with an entity, we must first make sure one is selected. If it is, its position is changed to that of the values just obtained from the interface. We do not need to use the *Z* coordinate here, because that is replaced by the elevation.

The *Z* coordinate is, however, used when dealing with particle emitters. After obtaining the additional value of the emit rate from the interface and converting it to a proper number, all of these values are applied to the current particle emitter selected.

Now, the piece of code that makes everything else tick:

[PRE20]

At this point, we want to make sure the selection drawable is updated, provided the `m_selectUpdate` flag is enabled. The rest of the code can be skipped if the `mapControls` class is not letting us know that the left mouse button is pressed. However, if it is, an appropriate update method is invoked, depending on what `selectMode` the interface is in.

A good way to keep an application looking neat and responsive is having neat indicators of certain selections being made, like so:

![Implementing selection options](img/image_05_003.jpg)

Let us take a look at how the selection rectangle can be updated for entities and emitters:

[PRE21]

As always, our logic branches out, depending on the selection mode we are in. Provided we are working with entities, a few checks are necessary in order to make sure one is selected. If it is, the next problem at hand is giving the rectangle a proper size, origin, and position. The easiest way to do that is by obtaining the colloidal component of an entity and manipulating it based on the collision primitive. If the entity doesn't have that type of component, we attempt to use the next best thing - its sprite sheet. Finally, if there's only a position component to work with, the rectangle is centered at the entity's position and given a fixed size of *32x32*.

Dealing with emitters is quite similar, minus the entire component headache. Provided one is selected, its 2D position is obtained and used to centre the rectangle, while giving it a static size of *32x32*.

Let us move on to updating the tile selection next:

[PRE22]

This is the actual method that handles tile selection logic. First, the coordinates of the starting tile that got clicked are obtained along with the current mouse position in tile coordinates. This information is used to calculate absolute global coordinates for the rectangle that will be used to represent the selection. The actual rectangle is then updated with this information, as well as set to have the `m_selectStartColor` color. Finally, all that is left to do is save this information as the current selection range, making sure it is in ascending order.

Next, updating entity selection deserves a peek:

[PRE23]

A check is needed to make sure the `mapControls` are in action in the same fashion as tile updating. Also, we obviously cannot update an entity that is not even selected, so a check is needed for that as well. The final bit of logic simply deals with moving the entity by the mouse position difference and updating our `selectionOptions` interface to hold its current position and elevation. The layer selection range is also updated to hold the elevation information. Finally, the select update flag is set to `true`, which requests the selection rectangle to be updated.

It's time to wrap up the updating logic. The only remaining mode left to update is the particle emitter selection:

[PRE24]

Just as before, the map control primary action flag is checked before proceeding, as well as the actual selection being made. The `X` and `Y` attributes of the particle emitter position are pushed by the mouse delta, while the `Z` coordinate is preserved as is. Afterwards, it is only a matter of updating the interface with the most recent position of the particle emitter, and marking the selection drawable for updating.

The last few pieces of the puzzle involve us dealing with mouse input correctly:

[PRE25]

As you recall from the previous chapter, this method is invoked by the `mapControls` class. It is required to return a *boolean* value that denotes whether a selection has been made or not, so that the `mapControls` class can deal with its own logic if the set of tools can give the artist a boost they have been looking for latter is true. When dealing with tiles, this method always needs to return `true`, allowing the control class to know that an action is taking place regardless.

While in entity mode, the `FindEntityAtPoint` method of the `entityManager` class is invoked, with the global position, as well as the layer selection range being passed in as arguments. The latter is only true if the user of the tile editor has decided to only make selected layers visible. It will return an entity ID if an entity has been found at a specific point in space, which is then used to call `SelectEntity`. To determine whether an entity has been selected, the ID is checked for not being equal to a known value for *not found*.

A very similar procedure is used to select a particle emitter. Because most emitters are single points in space, a `sf::Vector2f` needs to be used here simply to define the area around the position that can be clicked in order to select it.

Lastly, if a selection hasn't been made, the position text-fields of the selection options interface are filled in with the global coordinates of the click. This allows easier positioning of objects in the world before placement.

Surprisingly, quite a lot needs to happen when a mouse button is released. Let's take a look:

[PRE26]

Most of this logic is concerned with dealing with tile selection. The first thing we need to worry about is setting the selection rectangle to its final color, indicating the selection is made. After that, the interface buttons for copying and removing the selection are made visible, and a loop is used to check the selection in order to determine the solidity situation of the entire chunk, which is then saved to the appropriate interface element.

The entity and emitter modes do not need quite as much maintenance for such a simple task. All we need to worry about here is setting the selection rectangle colors appropriately.

As modes are being switched, all of the important data needs to be reset in order to avoid strange bugs:

[PRE27]

In addition to ranges and IDs being reset, the actual selection of the `DropDownMenu` of entities/emitters needs zeroing-out. Finally, all of the mode-specific GUI elements we have been working with need to be disabled and/or set to their neutral values.

Finally, we are left with only one essential chunk of code left to cover - the `Draw()` method:

[PRE28]

The only thing that we really need to draw is the selection rectangle. As it is quite evident here, it does not need to be drawn if no selection of any kind has been made. This includes checking all three select modes.

For the sake of completion, we only have a couple of getter methods left to look over:

[PRE29]

This concludes the `selectionOptions` class.

# Building the tile selector

When working with tile maps, it is important to have a fast and intuitive way of accessing the tile-sheet, selecting its contents and painting them directly onto the game map. A good set of tools can give the artist the boost they have been looking for, while an unmanageable application is only a hindrance. Let us take a peek at what we are going to be building:

![Building the tile selector](img/image_05_004.jpg)

This interface, just like most others we have been working with, is going to be much easier to manage when wrapped in a class of its own:

[PRE30]

Just like before, we have `Show()` and `Hide()` methods to manage its visibility, as well as a couple of callbacks. Note the highlighted method. It is going to be used for setting the texture of the tile-sheet the map is using.

The data members are quite predictable for a class like this. Alongside the classes that this object relies on, we keep track of a pointer to the actual interface it is going to be manipulating, an instance of a `sf::RenderTexture` that we are going to be drawing to, the sprite that will be used to display the render texture, a rectangle shape, start and end coordinates, and a *boolean* flag for the actual selection drawable. Lastly, `m_sheetTexture` is going to simply keep track of the texture identifier until it is time to release it.

## Implementing the tile selector

Let us begin by setting all of this data up inside the constructor:

[PRE31]

After the arguments are taken care of, the three callback methods we need are set up. The interface is then loaded and stored as one of the data members, just before its content rectangle size and offset are changed in order to allow space for control elements, such as the close button to be positioned comfortably. The interface is then centered on–screen and set to inactive. Finally, the rectangle shape used to represent tile selection is initialized to its default state as well.

Let us take a look at the destructor of this class next, in order to make sure we are not forgetting to release certain resources:

[PRE32]

After all three callbacks are released, it is imperative to make sure the tile-sheet texture is removed as well, provided its identifier is not empty.

Speaking of the tile-sheet texture, let us see how one can be assigned to this class:

[PRE33]

After the current tile-sheet texture is properly released, the new one is assigned and retrieved. Because of this, the actual selector texture that will be passed to the main GUI element of our interface needs to be re-drawn and passed into said element.

A similar procedure takes place when the interface needs to be updated:

[PRE34]

It simply consists of the tile-sheet, as well as the selector rectangle being drawn to the render texture. The interface is then instructed to re-draw its content, as it was changed.

Next, let us provide a way for outside classes to copy the current tile-sheet selection to a `TileMap` structure:

[PRE35]

Obviously, we cannot copy anything if nothing has been selected. The first check takes care of that. The `TileMap` passed in as the argument is then purged in preparation for being overwritten. The tile coordinate range is then calculated, and the `TileMap` argument is re-sized to match the size of the selection. After a couple of local variables are established to help us calculate the *1D* coordinate index, we begin iterating over the calculated range of tiles one by one, adding them to the tile map. Because we're not working with any sort of depth when dealing with a tile-sheet, the layer is always going to be set to the value `0`.

The following code deals with the mouse-click and mouse-release events, which are vital when making a selection:

[PRE36]

If we are dealing with a mouse-left click, we simply need to make note of the mouse coordinates at this point in time, as well as reset the `m_selected` flag to `false`. On the other hand, if the left mouse button has been released, the final mouse position is first checked for not going into negative values on both axes. The end coordinates are then stored, and the `m_selected` flag is set to `true`.

The remaining chunk of code simply deals with making sure the start and end coordinates are stored in an ascending order, and calculating the proper position and size of the selector rectangle. The `UpdateInterface()` method is then invoked, which makes sure everything is re-drawn.

Let us wrap this up by quickly looking over some of the helper methods of this class:

[PRE37]

The `Show()` and `Hide()` methods simply manipulate the interfaces activity, while the `Close` callback just invokes `Hide`. Just like that, all of the pieces fit together and we are left with a fully functional map editor!

# Summary

Building tools for a game may not be the easiest or the most pleasant task in the world, but in the end, it always pays off. Dealing with text files, endless copy-pasting, or other botch-like solutions may work fine in the short term, but nothing beats a fully equipped set of tools, ready to take on any project with the click of a button! Although the editor we have built is geared towards a very specific task, the idea behind it can, with enough time and energy, be applied to any set of production problems.

In the next chapter, we are going to be covering the basics and general uses of shaders in SFML. The OpenGL shading language, along with SFML's built in support for shaders, is going to allow us to create a very basic day and night cycle. See you there!