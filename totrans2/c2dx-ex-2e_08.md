# Chapter 8. Getting Physical – Box2D

*It's time to tackle physics! Cocos2d-x comes bundled with Box2D and Chipmunk. These are so-called 2D physics engines – the first written in C++ and the second in C. Chipmunk has a more recent Objective-C port but Cocos2d-x must use the original one written in C for portability.*

*We'll be using Box2D for the examples in this book. The next two games I'll show you will be developed with that engine, starting with a simple pool game to illustrate all the main points about using Box2D in your projects.*

In this chapter, you will learn:

*   How to set up and run a Box2D simulation
*   How to create bodies
*   How to use the debug draw feature to quickly test your concepts
*   How to use collision filters and listeners

# Building a Box2D project with Cocos2d-x

With version 3.x of the framework, we no longer need to specify that we want to use a physics engine. The projects add these APIs by default. So, all you need in order to create a Box2D project is to create a regular Cocos2d-x project as we've been doing with the examples so far.

There is, however, one extra step you need to perform if you wish to use something called a debug draw in your project. So let's set that up now.

# Time for action – using debug draw in your Box2D project

Let's start by creating the project. In my machine, I created a game called MiniPool in my desktop. Here are the steps:

1.  Open Terminal and enter the following command:

    [PRE0]

2.  Open the new project in Xcode.
3.  Now navigate to the `Tests` folder inside the Cocos2d-x framework folder. This can be found in `tests/cpp-tests/Classes`. Then open the `Box2DTestBed` folder.
4.  Drag the files `GLES-Render.h` and `GLES-Render.cpp` to your project in Xcode.
5.  You can also open the `Box2dTest.cpp` class in the test folder `Box2DTest`, as we're going to copy and paste a few of the methods from there.
6.  In the `HelloWorldScene.h` header file, leave the includes in place, but change the class declarations to match these:

    [PRE1]

7.  Then add this `include` statement at the top:

    [PRE2]

8.  Then, in the `HelloWorldScene.cpp` implementation file, replace the lines between the `using namespace CocosDenshion` and `HelloWorld::scene` methods with these:

    [PRE3]

9.  Now comes the implementation of the `draw` methods. You can copy and paste most of this code from the `Box2DTest` folder:

    [PRE4]

## *What just happened?*

The `GLES-Render` class is necessary to use the debug draw feature in Box2D. This will draw all the elements from the simulation on the screen. The debug draw object is created inside the `initPhysics` method alongside the Box2D simulation (`b2World`). We'll go over that logic in a moment.

As the comment inside the `draw` method states, the debug draw feature should be switched off once you're done developing your game. So all the lines pertaining to that object as well as the `draw` method should be commented out when you're ready for a release version.

# So what is a physics engine?

The famous Isaac Newton said, *every action has a reaction*. Right after he said, *who the hell threw that apple?*

So far in our games, we have covered very simple collision systems, basically only ever checking to see if simple shapes (circles and rectangles) overlapped each other. The reactions from these collisions were also very simple in our games so far: with vector inversions or simply by making things disappear once they touch. With Box2D, you get way more!

Box2D is a very robust collision detection engine and can certainly be used just for that purpose. But the simulation will also process and return a bunch of information derived from the collisions and the interactions between bodies, meaning how the objects should behave, based on their shapes, mass, and all the forces at play in the simulation.

## Meeting Box2D

At the core of the engine, you have the `b2World` object. This is the simulation. You fill the world with `b2Body` objects, and then you step through the simulation with `b2World->Step()`. And you take the results of the simulation and display them to the user through your sprites, by grabbing a `b2Body` object's position and rotation and applying them to a sprite.

The debug draw object allows you to see the simulation without using any sprites. Sort of like a version of our test project from [Chapter 6](part0087_split_000.html#page "Chapter 6. Quick and Easy Sprite – Victorian Rush Hour"), *Quick and Easy Sprite – Victorian Rush Hour*.

## Meeting the world

Most of the time, the physics simulation will mean the creation of a `b2World` object. Note, however, that you *can* get interesting results managing more than one `world` object in the same game, for multiple views for instance. But that's for another book.

In our simplified basic project, the world is created like this:

[PRE5]

Box2D has its own vector structure, `b2Vec2`, and we use it here to create the world's gravity. The `b2World` object receives that as its parameter. A simulation does not always require gravity, of course; in that case, the argument will be a `(0, 0)` vector.

`SetAllowSleeping` means if objects are not moving and therefore not generating derived data, skip checking for derived data from those objects.

`SetContinuousPhysics` means we have some fast objects in our hands, which we'll later point out to the simulation, so it can pay extra attention for collisions.

Then we create the debug draw object. This is optional, as I said before. The flags indicate what you wish to see in the drawing. In the code we saw before, we only want to see the shapes of the objects.

Then comes `PTM_RATIO`, the defined constant we passed as a parameter to the debug draw. Box2D uses meters instead of pixels for a variety of reasons that are really entirely unnecessary for anyone to know. Except for one reason, **pixel to meter** (**PTM**), so every pixel position value used in the game will be divided by this ratio constant. If the result from this division ever gets above 10 or below 0.1, increase or decrease the value for `PTM_RATIO` accordingly.

You have some leeway, of course. By all means, play with this value once your game is completed, and pay special attention to the subtle differences in speed (another common value for this ratio is 100).

## Running the simulation

As I said before, you use the `Step` method to run the simulation, usually inside your main loop, though not necessarily:

[PRE6]

You need to pass it the time step, here represented by the delta time in the main loop. Then pass the number of velocity iterations and position iterations in the step. This basically means how many times velocity and position will be processed inside a step.

In the previous example, I'm using the default values from the Box2D template in Cocos2d-x. Usually, a fixed time step is better than the delta, and a higher value for position iteration may be necessary if things move really fast in your game. But always remember to play with these values, aiming at finding the lowest possible ones.

## No Ref objects in Box2D

Box2D does not use `Ref` objects. So, no memory management! Remember to get rid of all the Box2D objects through `delete` and not `release`. If you knew it already... well, you remember:

[PRE7]

### Note

As I mentioned before, C++11 introduces smart pointers, which are memory managed, meaning you *don't* have to delete these objects yourself. However, the topic of shared pointers is beyond the scope of this book, and using unique pointers in this chapter would add way too many lines that had nothing to do with Box2D. And although smart pointers are amazing, their syntax and usage is, well, let's say very "C++ish".

## Meeting the bodies

The `b2Body` object is the thing you'll spend most of your time dealing with inside a Box2D simulation. You have three main types of `b2Bodies`: dynamic, static, and kinematic. The first two are of greater importance and are the ones we'll use in our game.

Bodies are created by combining a body definition with a body fixture. The body definition is a structure that holds information about type, position, velocity, and angle, among other things. The fixture holds information about the shape, including its density, elasticity, and friction.

So, to create a circle that is 40 pixels wide, you would use the following:

[PRE8]

To create a box that is 40 pixels wide, you would use this:

[PRE9]

Note that you use the `world` object to create the bodies. And also note that boxes are created with half their desired width and height.

Density, friction, and restitution all have default values, so you don't always need to set these.

# Our game – MiniPool

Our game consists of sixteen balls (circles), one cue (box), and a pool table made out of six lines (edges) and six pockets (circles). This is all there is to it as far as the Box2D simulation is concerned.

Download the final project from this book's **Support** page if you wish to follow along with the final code. Box2D is a complex API and it will be best to review and expose the logic rather than work on it by doing a lot of typing. So there will be no start project to work from this time. You may choose any manner to add files from the finished project to the one we started when I showed you how to set up the debug draw object. The final game will look like this:

![Our game – MiniPool](img/00030.jpeg)

## Game settings

This is a portrait-orientation-only game, with no screen rotation allowed, and universal application. The game is designed for the regular iPhone (320 x 480) and its resolution size is set to `kResolutionShowAll`. This will show borders around the main screen in devices that do not match the 1.5 screen ratio of the iPhone.

[PRE10]

Note that I use the iPhone's dimensions to identify larger screens. So the iPad and the iPhone retina are considered to be two times 320 x 480 and the retina iPad is considered to be four times 320 x 480.

## Sprite plus b2Body equal to b2Sprite

The most common way to work with `b2Body` objects in Cocos2d-x is to combine them with sprites. In the games I'll show you, I created a class called `b2Sprite` that extends sprite with the addition of a `_body` member property that points to its very own `b2Body`. I also add a few helper methods to deal with our pesky `PTM_RATIO`. Feel free to add as many of these as you think necessary.

`b2Body` objects have an incredibly helpful property called `userData`. You can store anything you wish inside it and the bodies will carry it with them throughout the simulation. So, what most developers do is that they store inside the body's `userData` property a reference to the instance of sprite wrapping it. So `b2Sprite` knows about its body, and the body knows about its `b2Sprite`.

### Tip

As a matter of fact, composition is key when working with Box2D. So, when designing your games, make sure every object knows of every other object or can get to them quickly. This will help immensely.

## Creating the pool table

In the debug draw view, this is what the table looks like:

![Creating the pool table](img/00031.jpeg)

All the elements seen here are created inside the `initPhysics` method in `GameLayer.cpp`. The table has no visual representation other than the background image we use in the game. So there are no sprites attached to the individual pockets, for example.

The `pocket` bodies are created inside a `for` loop, with the best algorithm I could come up with to distribute them correctly on screen. This logic is found in the `initPhysics` method, so let's take a look at that and see how our first `b2Body` objects are created:

[PRE11]

The `pocket` bodies are static bodies and we determine in their fixture definition that they should behave like sensors:

[PRE12]

This switches off all the physics from an object and turns it into a collision hot spot. A sensor serves only to determine if something is touching it or not.

### Tip

It's almost always best to ignore Box2D sensors and use your own sprites or points in your collision logic. One neat feature in sensors is that they make it very easy to determine when something has just ceased touching them, as you'll see once we cover contact listeners.

## Creating edges

If a shape can only be hit on one side, an edge is probably what you need. Here is how we create edges in our game:

[PRE13]

So the same `b2Body` object can have as many edges as you need. You set an edge with its start and end points (in this case, the `b2Vec2` structures) and add it as a fixture to the body, with a density of `0`.

## Creating the ball objects

In the game, there is a class called `Ball` that extends `b2Sprite`, used for both the target balls and the cue ball. These objects are also created inside the `initPhysics` method. Here is the basic configuration of that object:

[PRE14]

The `friction` fixture property involves the reaction of two touching surfaces (two bodies). In this case, we want to create "friction" with the table surface, which is not a body at all. So what we need to use instead is **damping**. This will apply a similar effect to friction, but without the need for an extra surface. Damping can be applied to the linear velocity vector of a body as follows:

[PRE15]

And to the angular velocity as follows:

[PRE16]

Also, the white ball is set to be a bullet:

[PRE17]

This will make the simulation pay extra attention to this object in terms of collision. We could make all balls in the game behave as bullets, but this is not only unnecessary (something revealed through testing) but also not very processing-friendly.

## Creating collision filters

In the `ball` object, there is a `filter` property inside the fixture definition that we use to mask collisions. Meaning we determine what bodies can collide with each other. The cue ball receives a different value for `categoryBits` than the other balls.

[PRE18]

When we create the cue body, we set a `maskBits` property in its fixture definition as follows:

[PRE19]

We set this to the same value as the white ball's `categoryBits`.

The result of all this? Now the cue can only hit bodies with the same `categoryBits`, which here means the cue can only collide with the white ball.

It is possible to add more than one category to a mask with a bitwise `|` option, as seen here:

[PRE20]

Or to collide with everything except the cue ball, for instance, as seen in the following line:

[PRE21]

## Creating the cue

The cue ball also extends `b2Sprite`, and its body is set as a box.

[PRE22]

It has very high damping values because, in the rare occasions when the player misses the cue ball, the cue will not fly off the screen but halt a few pixels from the white ball.

If we wanted to create the cue ball as a trapezium or a triangle, we would need to give the `b2PolygonShape` option the vertices we want. Here's an example of this:

[PRE23]

And the vertices must be added counterclockwise to the array. Meaning, if we add the top vertex of the triangle first, the next vertex must be the one to the left.

Once all the elements are in place, the debug draw looks like this:

![Creating the cue](img/00032.jpeg)

## Creating a contact listener

Besides collision filters, one other feature in Box2D that helps with collision management is the creation of a contact listener.

Inside the `initPhysics` method, we create the `world` object like this:

[PRE24]

Our `CollisionListener` class extends the Box2D `b2ContactListener` class, and it must implement at least one of the following methods:

*   `void BeginContact(b2Contact* contact);`
*   `void EndContact(b2Contact* contact);`
*   `void PreSolve(b2Contact* contact, const b2Manifold* oldManifold);`
*   `void PostSolve(b2Contact* contact, const b2ContactImpulse* impulse);`

These events are all related to a contact (collision) and are fired at different stages of a contact.

### Note

Sensor objects can only ever fire the `BeginContact` and `EndContact` events.

In our game, we implement two of these methods. The first is:

[PRE25]

You can see now how important the `userData` property is. We can quickly access sprites attached to the bodies listed in the `b2Contact` object through the `userData` property.

Besides that, all our sprites have a `_type` property that behaves like identifying tags in our logic. Note that you could certainly use the Cocos2d-x tags for that, but I find that at times, if you can combine the `Sprite` tags with their `_type` value, you may produce interesting sorting logic.

So, in `BeginContact`, we track the collisions between balls and pockets. But we also track collision between balls. In the first case, the balls are turned invisible when they touch the pockets. And, in the second case, we play a sound effect whenever two balls touch each other, but only if they are at a certain speed (we determine that through a `b2Sprite` helper method that retrieves the squared magnitude of a sprite's velocity vector).

The other method in our listener is:

[PRE26]

Here, we listen to a collision before its reactions are calculated. If there is a collision between the cue and white ball, we play a sound effect and we hide the cue.

### Note

If you want to force your own logic to the collision reaction and override Box2D on this, you should do so in the `PreSolve` method. In this game, however, we could have added all this collision logic to the `BeginContact` method and it would work just as well.

## The game controls

In the game, the player must click on the white ball and then drag his or her finger to activate the cue ball. The farther the finger gets from the white ball, the more powerful the shot will be.

So let's add the events to handle user input.

# Time for action – adding the touch events

We'll deal with `onTouchBegan` first.

1.  In the `onTouchBegan` method, we start by updating the game state:

    [PRE27]

2.  Next, we check on the value of `_canShoot`. This returns `true` if the white ball is not moving.

    [PRE28]

3.  Next, we determine whether the touch is landing on the white ball. If it is, we start the game if it is not currently running yet and we make our timer visible. Here's the code to do this:

    [PRE29]

    Note that we use a larger radius for the white ball in our logic (four times larger). This is because we don't want the target area to be too small, since this game will run on both iPhones and iPads. We want the player to comfortably hit the white ball with his or her finger.

4.  We store where in the ball the point lies. This way, the player can hit the ball at different spots, causing it to move at different angles:

    [PRE30]

    Since we made the white ball a much larger target for our `touch` event, now we must make sure the actual point picked by the player lies within the ball. So we may have to make some adjustments here.

5.  We pass the point to our `LineContainer` object and we prepare the cue body to be used, as follows:

    [PRE31]

    We once again have a `LineContainer` node so we can draw a dashed line between the cue and the spot on the ball where the cue will hit. This serves as a visual aid for the player to prepare his or her shot. The visual aid effect is demonstrated here:

    ![Time for action – adding the touch events](img/00033.jpeg)
6.  In `onTouchMoved`, we only need to move the cue body based on the position of the player's finger. So we calculate the distance between the moving touch and the white ball. If the cue body is still too close to the ball, we set its `body` object to `sleep` and its `texture` object to `invisible`.

    [PRE32]

7.  Otherwise, we awaken the body and call the `placeCue` method as follows:

    [PRE33]

    This method then calculates the angle and position of the cue body and transforms the cue's `b2Body` method accordingly. The `SetTransform` option of a `b2Body` method takes care of both its position and angle.

8.  Finally, in `onTouchEnded`, we let go of the cue body as follows:

    [PRE34]

    We use `ApplyLinearImpulse`. This method receives a vector for the impulse to be applied and the position on the body where this impulse should be applied.

    The `_pullback` variable stores the information of how far the cue body was from the ball when the player released the cue body. The farther it was, the strongest the shot will be.

## *What just happened?*

We added the `touch` events that allow the player to hit the white ball with the cue body. The process is a very simple one. We first need to make sure the player is touching the white ball; then we move the cue body as the player drags his or her finger. Finally, when the touch is released, we make the cue spring towards the white ball with `ApplyLinearImpulse`.

We may also move a body in Box2D by using `SetLinearVelocity` or `ApplyForce`, each with subtle and not-so-subtle differences. I recommend that you play around with these.

## The main loop

As I showed you before, the simulation only requires that you call its `Step()` method inside the main loop. Box2D takes care of all of its side of the bargain.

What remains usually is the rest of the game logic: scoring, game states, and updating your sprites to match the `b2Bodies` method.

It's important to call the `update` method of each ball and cue. This is what our `b2Sprite update` method looks like:

[PRE35]

All you need to do is make sure the `Sprite` method matches the information in the `b2Body` object. And make sure that you convert meters back to pixels when you do so.

So let's add our main loop.

# Time for action – adding the main loop

It's inside our main loop that we update our `b2World` object.

1.  Start by updating the simulation as follows:

    [PRE36]

2.  Next, we need to determine if the game has finished by checking on the number of balls currently in play. We use the following for that:

    [PRE37]

3.  Next, we continue to update the sprites as follows:

    [PRE38]

4.  And we also determine when it's time to allow the player a new shot. I decided to only let that happen if the white ball has stopped. And the quickest way to determine that is to check on its vector. Here's how:

    [PRE39]

## *What just happened?*

We added our main loop. This will update the Box2D simulation and then it's up to us to take care of positioning our sprites based on the resulting information.

### Note

One very important aspect of Box2D is understanding what can be changed inside a `b2World::Step` call and what can't.

For instance, a body cannot be made inactive (`b2Body::SetActive`) or be destroyed (`b2World::DestroyBody`) inside a step. You will need to check on conditions outside the step to make these changes. For instance, in our game, we check to see if the ball sprites are visible or not, and if not then we set their bodies as inactive. And all this is done *after* the call to `b2World::Step`.

## Adding a timer to our game

In MiniPool, we count the number of seconds it takes the player to clear the table. Let me show you how to do that.

# Time for action – creating a timer

We create timers in pretty much the same way we create our main loop.

1.  First, we add a second scheduled event by adding this line to our `GameLayer` constructor:

    [PRE40]

2.  With this, we create a separate timer that will run the `ticktock` method every `1.5` seconds (I decided in the end that `1.5` seconds looked better).
3.  The method keeps updating the value of the `_time` property and displaying it in the `_timer` label.

    [PRE41]

## *What just happened?*

We added a timer to our game by scheduling a second update—specifying the time interval we wanted—using the `schedule` method.

If you wish to remove a timer, all you need to do is call the `unschedule(SEL_SCHEDULE selector)` method of nodes anywhere in your class.

Now, let's take our Box2D game to Android.

# Time for action – running the game in Android

Follow these steps to deploy a Box2D game to Android:

1.  Open the `Android.mk` file in a text editor (you'll find it in the folder `proj.android/jni`).
2.  Edit the lines in `LOCAL_SRC_FILES` to read:

    [PRE42]

3.  Open the manifest file and set the app orientation to `portrait`.
4.  Import the game into Eclipse and wait till all classes are compiled.
5.  Build and run your application.

## *What just happened?*

That was it. There is no difference between building a game that uses Box2D and one that does not. The Box2D API is already included in the `make` file, in the line where the classes in the external folder are imported.

And, of course, you don't need to add the `GLES-Render` class in your final project.

## Have a go hero

A few changes to make gameplay more interesting could be: add a limit to the number of times the white ball can hit a pocket; and another option is to have the timer work as a countdown one, so the player has a limited time to clear the table before time runs out.

Also, this game could do with a few animations. An `Action` method to scale down and fade out a ball when it hits a pocket would look very nice.

## Pop quiz

Q1\. What is the main object in a Box2D simulation?

1.  `b2Universe`.
2.  `b2d`.
3.  `b2World`.
4.  `b2Simulation`.

Q2\. A `b2Body` object can be of which type?

1.  `b2_dynamicBody`, `b2_sensorBody`, `b2_liquidBody`.
2.  `b2_dynamicBody`, `b2_staticBody`, `b2_kinematicBody`.
3.  `b2_staticBody`, `b2_kinematicBody`, `b2_debugBody`.
4.  `b2_kinematicBody`, `b2_transparentBody`, `b2_floatingBody`.

Q3\. Which of the following list of properties can be set in a fixture definition?

1.  Density, friction, restitution, shape.
2.  Position, density, bullet state.
3.  Angular damping, active state, friction.
4.  Linear damping, restitution, fixed rotation.

Q4\. If two bodies have the same unique value for their `maskBits` property in their fixture definition, this means:

1.  The two bodies can never collide.
2.  The two bodies will only trigger begin contact events.
3.  The two bodies can only collide with each other.
4.  The two bodies will only trigger end contact events.

# Summary

Nowadays, it seems like everybody in the world has played or will play a physics-based game at some point in their lives. Box2D is by far the most popular engine in the casual games arena. The commands you learned here can be found in pretty much every port of the engine, including a JavaScript one that is growing in popularity as we speak.

Setting up the engine and getting it up and running is remarkably simple—perhaps too much so. A lot of testing and value tweaking goes into developing a Box2D game and pretty soon you learn that keeping the engine performing as you wish is the most important skill to master when developing physics-based games. Picking the right values for friction, density, restitution, damping, time step, PTM ratio, and so on can make or break your game.

In the next chapter, we'll continue to use Box2D, but we'll focus on what else Cocos2d-x can do to help us organize our games.