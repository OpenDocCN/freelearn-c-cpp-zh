# Chapter 9. On the Level – Eskimo

*In our next game, we'll go over some important features most games require, but which are not directly related to gameplay. So we'll step over the architecture side of things and talk about reading and writing data, using scene transitions, and creating custom events that your whole application can listen to.*

*But, of course, I'll add a few gameplay ideas as well!*

This time, you'll learn how to:

*   Create scene transitions
*   Load external data
*   Save data using `UserDefault`
*   Create your own game events with the dispatcher
*   Use the accelerometer
*   Reuse Box2D bodies

# The game – Eskimo

Little Eskimo boy is late for supper. It is your mission, should you choose to accept it, to guide the little fella back to his igloo.

This is a Box2D game, and the controls are very simple. Tilt the device and the Eskimo will move. If you tap the screen, the Eskimo switches shape between a snow ball and a block of ice, each shape with its own physical characteristics and degrees of maneuverability. The ball has a higher friction, for instance, and the block of ice has none.

And the only way the Eskimo may reach his destination is by hitting the gravity switches spread out all over the screen.

Eskimo combines elements from an arcade game with elements of a puzzle game, as each level was planned with one perfect solution in mind as to how to take the little Eskimo home. Note, however, that multiple solutions are possible.

![The game – Eskimo](img/00034.jpeg)

Download the `4198_09_FINAL_PROJECT.zip` file and run the game when you have a chance. Once again, there is no need for extraneous typing as the logic used in the game is pretty much old news to you, and we'll go over the new bits in depth.

## The game settings

This is a portrait-only game and accelerometer-based, so it should not autorotate. It was designed for the regular iPhone and its screen resolution size is set to `kResolutionShowAll`, so the screen settings are similar to the ones in our previous game.

Designing a game for the iPhone screen and using the `kResolutionShowAll` parameter will result in the so-called **letterbox** view when playing the game in screens that do not match the iPhone's 1.5 ratio. This means you see borders around the game screen. Alternatively, you could use the `kResolutionNoBorders` parameter, which results in a **zoom-in** effect, causing the game to play at full screen but the areas around the borders will be cropped.

The following screenshot illustrates these two cases:

![The game settings](img/00035.jpeg)

The one on the left is the game screen on the iPad, using `kResolutionShowAll`. The one on the right uses `kResolutionNoBorders`. Note how the screen is zoomed in and cropped on the second one. When using `kResolutionNoBorders`, it's important to design your game so that no vital gameplay element appears too close to the borders as it may not be displayed.

## Organizing the game

Once again, there is a `b2Sprite` class, and the `Eskimo` and `Platform` classes extend `b2Sprite`. Then there are the regular `Sprite` classes, `GSwitch` (which stands for gravity switch) and `Igloo`. The logic runs collision detection between these last two and `Eskimo`, but I chose not to have them as sensor bodies because I wanted to show you that 2D collision logic for the Cocos2d-x elements can coexist with collision logic for the Box2D elements just fine.

But most importantly, this game now has three scenes. So far in this book, we've only used one scene per game. This game's scene objects will wrap `MenuLayer`, `LevelSelectLayer`, and `GameLayer`. Here's a brief note on all three:

*   In `MenuLayer`, you have the option to play the game, which will take you to `LevelSelectLayer` or to play a tutorial for the game, which will take you to `GameLayer`.
*   In `LevelSelectLayer`, you may choose which available level you want to play, and that will take you to `GameLayer`. Or you may go back to `MenuLayer`.
*   In `GameLayer`, you play the game, and may go back to `MenuLayer` upon game over.

The following image illustrates all three scenes in the game:

![Organizing the game](img/00036.jpeg)

# Using scenes in Cocos2d-x

Scenes are mini applications themselves. If you have experience as an Android developer, you may think of scenes as activities. Of all the classes based on node, the `Scene` application is the most architecturally relevant, because the `Director` class runs a scene, in effect running your application.

Part of the benefit of working with scenes is also part of the drawback: they are wholly independent and ignorant of each other. The need to share information between scenes will be a major factor when planning your game class structure.

Also, memory management may become an issue. A currently running scene will not give up its ghost until a new scene is up and running. So, when you use transition animations, keep in mind that for a few seconds, both scenes will exist in memory.

In Eskimo, I initialize scenes in two different ways. With `MenuLayer` and `LevelSelectLayer`, each time the user navigates to either one of these scenes, a new layer object is created (either a new `MenuLayer` or a new `LevelSelectLayer`).

`GameLayer`, however, is different. It is a singleton `Layer` class that never stays out of memory after its first instantiation, therefore speeding up the time from level selection to the actual playing. This may not work for every game, however. As I mentioned earlier, when transitioning between scenes, both scenes stay in memory for a few seconds. But here we are adding to that problem by keeping one layer in memory the whole time. Eskimo, however, is not very big memory-wise. Note that we could still have the option of creating special conditions for when `GameLayer` should be destroyed, and conditions when it should not.

So let me show you how to create scene transitions. First, with a `Scene` class that creates a fresh copy of its `Layer` each time it's created.

# Time for action – creating a scene transition

You have, of course, been using scenes all along.

1.  Hidden in `AppDelegate.cpp`, you've had lines like:

    [PRE0]

2.  So, in order to change scenes, all you need to do is tell the `Director` class which scene you wish it to run. Cocos2d-x will then get rid of all the content in the current scene, if any (all their destructors will be called), and a new layer will be instantiated and wrapped inside the new `Scene`.
3.  Breaking the steps down a little further, this is how you usually create a new scene for `Director`:

    [PRE1]

4.  The static `MenuLayer::scene` method will create a blank scene, and then create a new instance of `MenuLayer` and add it as a child to the new scene.
5.  Now you can tell `Director` to run it as follows:

    [PRE2]

6.  The logic changes a little if you wish to use a transition effect. So, inside our `MenuLayer.cpp` class, this is how we transition to `LevelSelectLayer`:

    [PRE3]

    The code just described creates a new transition object that will slide in the new scene from the right-hand side of the screen to lie on top of the current one. The transition will take `0.2` seconds.

## *What just happened?*

You created a scene transition animation with Cocos2d-x.

As I mentioned earlier, this form of scene change will cause a new instance of the new layer to be created each time, and destroyed each time it's replaced by a new scene. So, in our game, `MenuLayer` and `LevelSelectLayer` are instantiated and destroyed as many times as the user switches between them.

There is also the option to use `pushScene` instead of `replaceScene`. This creates a stack of `scene` objects and keeps them all in memory. This stack can be navigated with `popScene` and `popToRootScene`.

Now let me show you how to do the same thing but with a singleton layer.

It should be no surprise to you by now that you will find many examples of these transition classes in the `Tests`, project at `tests/cpp-tests/Classes/TransitionsTest`.

# Time for action – creating transitions with a singleton Layer class

We first need to make sure the layer in question can only be instantiated once.

1.  The `scene` static method in `GameLayer` looks like this:

    [PRE4]

    This layer receives two parameters when created: the game level it should load and the number of levels completed by the player. We create a new `Scene` object and add `GameLayer` as its child.

2.  But take a look at the static `create` method in `GameLayer`:

    [PRE5]

3.  An `_instance` static property is declared at the top of `GameLayer.cpp` as follows:

    [PRE6]

    We can check, then, if the one instance of `GameLayer` is currently in memory and instantiate it if necessary.

4.  The scene transition to `GameLayer` will look, on the surface, to be exactly like the regular kind of transition. So, in `LevelSelectLayer`, we have the following:

    [PRE7]

## *What just happened?*

We have created a `Scene` transition with a `Layer` class that never gets destroyed, so we don't have to instantiate new platform and gravity switch sprites with each new level.

There are, of course, problems and limitations with this process. We cannot transition between the two `GameLayer` objects, for instance, as we only ever have one of these objects.

There are also some special considerations when leaving `GameLayer` and when getting back to it. For instance, we must make sure we have our main loop running when we get back to `GameLayer`.

The only way to do that is by unscheduling it whenever leaving `GameLayer` and scheduling it again when returning, as follows:

[PRE8]

### Tip

Again, architecturally speaking, there are even better options. Possibly the best one is creating your own game elements cache, or game manager, with object pools and everything that needs instantiating stored inside it. And then have this cache be a singleton that every scene can access. This is also the best way to share game-relevant data between scenes.

# Loading external data from a .plist file

Eskimo has only five game levels, plus a tutorial level (feel free to add more). The data for these levels exist inside a l`evels.plist` file, stored inside the `Resources` folder. A `.plist` file is an XML-formatted data file, and as such can be created in any text editor. Xcode, however, offers a nice GUI to edit the files.

Let me show you how to create them inside Xcode.

# Time for action – creating a .plist file

You could, of course, create this in any text editor, but Xcode makes it extra easy to create and edit `.plist` files.

1.  Inside Xcode, go to **New** | **File...** and then select **Resource** and **Property List**. When asked where to save the file, choose any location you want.![Time for action – creating a .plist file](img/00037.jpeg)
2.  You need to decide what the **Root** element of your `.plist` file will be—either an **Array** or a **Dictionary** (the default) type. For Eskimo, the **Root** element is **Array** containing a series of dictionaries, each holding the data for a level in the game.
3.  By selecting the **Root** element, you get a plus sign indicator right next to the **Type** declaration. Clicking on this plus sign will add an element to **Root**. You can then pick the data type for this new item. The options are **Boolean**, **Data**, **Date**, **Number**, **String**, and again **Array** and **Dictionary**. The last two can contain subitems in the tree, just like the **Root** element.
4.  Keep adding elements to the tree, trying to match the items in the following screenshot:![Time for action – creating a .plist file](img/00038.jpeg)

## *What just happened?*

You just created a property list file in Xcode. This is XML-structured data that Cocos2d-x can load and parse. You've used them already when loading particles and sprite sheet information.

# Loading the level data

In Eskimo, since I only have five levels, I chose to have one `.plist` file that contains all levels. This may not be the best option in a larger game.

Although Apple devices will load and parse the `.plist` files quickly, the same may not be true for other targets. So limit the size of your `.plist` files by organizing the data into multiple files. You've probably seen games that divide their levels into multiple groups or packs. This is a simple way to create an extra preloading screen your game can use to parse level data. This can also be used as a means to keep file sizes to a minimum.

In Eskimo, we could have the `.plist` files containing 10 levels each, for instance, and then 10 groups of these, totaling 100 levels.

So it's time to load our `.plist` file and parse the data for our levels.

# Time for action – retrieving data from the .plist file

The level data is loaded in `GameLayer`.

1.  Inside the `GameLayer` constructor, we load the data like this:

    [PRE9]

    Cocos2d-x will take care of mapping `FileUtils` to the correct target. There is `FileUtils` for each platform that is supported by the framework and they all can be made to work with the `.plist` format. Sweet! If the data in the `.plist` file is an **Array**, you must convert it to `ValueVector`; if it's **Dictionary**, you must convert it to a `ValueMap`. We'll do that next when we load the data for a specific level.

    ### Note

    If we divide the levels into multiple `.plist` files, then we would need logic to refresh the `_levels` array each time a new `.plist` file is loaded.

2.  Inside the `loadLevel` method, we load the data for the level like this:

    [PRE10]

    Here, the data in the `.plist` file is **Dictionary**, so we must convert the data into a `ValueMap`.

    And that's it for the loading and parsing. Now we can proceed to retrieving data for our level.

    Each level dictionary starts with the data regarding the level's gravity (a level may start with a different gravity value), the start point where the player should be placed, and the end point where the igloo should be placed.

3.  These values are retrieved like this in our code:

    [PRE11]

4.  Inside this same dictionary, we have an array for platforms and an array for gravity switches. These are retrieved like this:

    [PRE12]

5.  These arrays contain even more dictionaries containing data for the creation and placement of platforms and gravity switches in each level. This data is passed to the corresponding `Platform` and `GSwitch` classes, and boom—you've got yourself a level.

    [PRE13]

## *What just happened?*

Parsing and retrieving data from a property list file is a breeze with Cocos2d-x. You will always work with either an array of values or a dictionary of values and map these to a `ValueVector` or `ValueMap` respectively.

# Saving game data

When planning your games, you may soon decide you wish to store data related to your application, such as highest score or user preferences. In Cocos2d-x, you can do this by simply accessing the `UserDefault` singleton.

With `UserDefault`, you can store integers, floats, doubles, strings, and Boolean with just one simple call per each data type, as follows:

[PRE14]

The other methods are `setFloatForKey`, `setDoubleForKey`, `setStringForKey`, and `setBoolForKey`. To retrieve data, you use their respective getters.

I'll show you next how to use that in our game.

# Time for action – storing the completed levels

Open the `LevelSelectLayer` class.

1.  This is how the number of levels completed is retrieved from inside the layer's constructor:

    [PRE15]

2.  Initially, `_levelsCompleted` will equal `0` if no data is present. So we store level 1 as "unlocked". This is how that's done:

    [PRE16]

3.  Then, whenever we start a new level, we update the number of levels completed if the new level number is larger than the value stored.

    [PRE17]

    ### Note

    You don't have to flush the data (using `flush`) each time you update every single bit in it. You can group multiple updates under one flush, or find a spot in your logic where you can safely flush updates before exiting the app. Nodes come with extremely helpful methods for this: `onEnter`, `onExit`, `onEnterTransitionDidFinish`, and `onExitTransitionDidStart`.

## *What just happened?*

For small bits of data related to your game, settings, and preferences, `UserDefault` is an excellent way to store information. Cocos2d-x once again will map this to whatever local storage is available in each target system.

# Using events in your game

Earlier versions of the framework used an Objective-C-inspired feature of notifications. But this particular API is already on its way to being deprecated. Instead, you should use the all-knowing `Director` and its `Dispatcher` (the same object we've been talking to when listening to touch events).

If you have ever worked with an MVC framework or developed a game AI system, you are probably familiar with a design pattern called the **Observer Pattern**. This consists of a central message dispatcher object other objects can subscribe to (observe) in order to listen to special messages, or order it to dispatch their own messages to other subscribers. In other words, it's an event model.

With Cocos2d-x, this is done very quickly and easily. Let me give you an example used in Eskimo.

# Time for action – using the event dispatcher

If we want the `Platform` sprite to listen to the special notification `NOTIFICATION_GRAVITY_SWITCH`, all we need to do is add `Platform` as an observer.

1.  Inside the `Platform` class, in its constructor, you will find these lines:

    [PRE18]

    And yes, it is one line of code! It is best to create a macro for both the dispatcher and the add listener code; so, something like this:

    [PRE19]

    This way the same line of code we used before would look like this:

    [PRE20]

2.  The message (or notification), `NOTIFICATION_GRAVITY_SWITCH`, is created as a static string in `GameLayer`:

    [PRE21]

    The one-line call to the `Director` class's dispatcher tells it that the `Platform` objects will listen to this defined message, and when such a message is dispatched, every `Platform` object will call the `onGravityChanged` method. This method does not need to be a block as I showed here, but it is more readable to have the handler appear as close to the `Add Listener` call as possible. So, simple blocks are a good way to organize listeners and their handlers.

3.  In the game, each gravity switch is color coded, and when the Eskimo hits a switch, the platform's texture changes to reflect the new gravity by switching to the color of the activated gravity switch. This is all done through a simple notification we dispatch inside `GameLayer` when a collision with a `GSwitch` object is detected inside the main loop. This is how we do that:

    [PRE22]

    Or, if you are using the macro, use this:

    [PRE23]

4.  You can also add a `UserData` object in the custom event as a second parameter in the dispatch. This can be retrieved from the `EventCustom *` event in the event handler, like this:

    [PRE24]

5.  When `Platform` objects are destroyed, the `Node` destructor will take care of removing the node as a listener.

## *What just happened?*

You have just learned how to make your life as a developer much, much easier. Adding an application-wide event model to your game is such a powerful way to improve flow and interactivity between objects and it's so simple to use that I'm sure you'll soon implement this feature in all your games.

# Using the accelerometer

Now let's move to the few new topics related to gameplay, the first of which is the use of accelerometer data. Again, nothing could be simpler.

# Time for action – reading accelerometer data

Just as you do with `touch` events, you need to tell the framework you want to read accelerometer data.

1.  You tell the framework you wish to use the accelerometer with this one call inside any `Layer` class:

    [PRE25]

2.  Then, just as you've done with `touch` events, you subscribe to the `accelerometer` events from the event dispatcher as follows:

    [PRE26]

3.  In Eskimo, the accelerometer data changes the value of a `Point` vector called `_acceleration`.

    [PRE27]

    This value is then read inside the main loop and used to move the Eskimo. In the game, only one axis is updated at a time, depending on the current gravity. So you can only ever move the Eskimo on the `X` axis or the `Y` axis with the accelerometer data, but never both at the same time.

    ### Note

    Keep in mind that there is also a `Z` axis value in the `Acceleration` data. It might come in handy someday!

## *What just happened?*

Yep. With a couple of lines, you added accelerometer controls to your game.

It is common practice to add extra filters to these accelerometer values, as results may vary between devices. These filters are ratios you apply to acceleration to keep values within a certain range. You can also find a variety of formulas for these ratios online. But these will depend on how sensitive you need the controls to be or how responsive.

And, in the game, we only update the Eskimo with the accelerometer data if the sprite is touching a platform. We can quickly ascertain that by checking whether or not the `_player` body has a contact list, as follows:

[PRE28]

# Reusing b2Bodies

In Eskimo, we have a pool of `b2Bodies` that are used inside the `Platform` objects and we also change the shape of the little Eskimo whenever the player taps the screen. This is possible because Box2D makes it very easy to change the fixture data of a `b2Body` fixture without having to destroy the actual body.

Let me show you how.

# Time for action – changing a b2Body fixture

All you have to do is make a call to `body->DestroyFixture`. Not surprisingly, this should be done outside the simulation step.

1.  Inside the methods `makeCircleShape` and `makeBoxShape` in the `Eskimo` class, you will find these lines:

    [PRE29]

    Here we just state that if there is a fixture for this body, destroy it. We can then switch from a box to a circle fixture when the player taps the screen, but use the same body throughout.

2.  We use this feature with platforms too. Platforms inside the pool that are not being used in the current level are set to inactive as follows:

    [PRE30]

    This removes them from the simulation.

3.  And when they are reinitialized to be used in a level, we destroy their existing fixture, update it to match the data from the `.plist` file, and set the body to active once again. This is how we do that:

    [PRE31]

## *What just happened?*

So, just as we've been doing with pools of sprites, we can apply the same logic to `b2Bodies` and never instantiate anything inside the main loop.

Now, let's see how Android handles all this level-loading business.

# Time for action – running the game in Android

Time to deploy the game to Android.

1.  Navigate to the `proj.android` folder and open the file `AndroidManifest.xml` in a text editor. Then go to the folder `jni` and open the file `Android.mk` in a text editor.
2.  In the `AndroidManifest.xml` file, edit the following line in the `activity` tag as follows:

    [PRE32]

3.  Next, let's edit the make file, so open the `Android.mk` file and edit the lines in `LOCAL_SRC_FILES` to read:

    [PRE33]

4.  Now import the project into Eclipse and build it.
5.  You can now save it and run the game in your Android device.

## *What just happened?*

By now, you should be an expert at running your code in Android and hopefully your experience with Eclipse has been a good one.

And that's all folks!

Play the game. Check out the source code (which is chock-full of comments). Add some new levels and make the little Eskimo's life a living hell!

## Have a go hero

The gameplay for Eskimo could be further improved with a few new ideas that would force the player to make more errors.

It is a common feature in these types of games to evaluate the degree of "completeness" in which a level was played. There could be a time limit for each level and pick-up items for the Eskimo, and the player could be evaluated at the end of each level and awarded a bronze, silver, or golden star based on his or her performance. And new groups of levels may only be unlocked if a certain number of golden stars were acquired.

# Summary

Yes, you have a cool idea for a game, great! But a lot of effort will go into structuring and optimizing it. Cocos2d-x can help with both sides of the job.

Yes, scenes can be a bit cumbersome depending on your needs, but they are undisputed memory managers. When `Director` kills a scene, it kills it dead.

Loading external data can not only help with memory size, but also bring in more developers into your project, focusing specifically on level design and the external data files that create them.

And events can quickly become a must in the way you structure your games. Pretty soon, you will find yourself thinking in terms of events to handle game states and menu interactivity, among other things.

Now, let's move to a whole new language!