# Chapter 9. A Breath of Fresh Air – Entity Component System Continued

In the previous chapter, we discussed the benefits of using aggregation versus simple inheritance. While not necessarily intuitive at first glance, entities composed of multiple components and operated on by systems inarguably enable higher flexibility and re-usability of code, not to mention a more convenient environment for future growth. Well, "The future is now!" as the popular expression states. A house is useless without a good foundation, just as much as a good foundation is useless without a house built on top of it. Since we already have a solid foundation, laying bricks until a proper structure emerges is what's next.

In this chapter, we will be:

*   Implementing basic movement
*   Developing a system for updating sprite sheets
*   Revisiting and implementing entity states
*   Studying the collision within the entity component system paradigm

# Adding entity movement

Within the entity component system paradigm, movement of a particular body is quantified by all the forces imposed on it. The collection of these forces can be represented by a movable component:

[PRE0]

This component takes away physics elements from the second project of this book, namely the velocity, speed and acceleration attributes. In order to simplify the code, the velocity cap is represented by a single float this time, as it is unlikely we will ever need to limit the velocity differently based on its axis.

Let's take a look at the rest of the movable component class:

[PRE1]

The constructor here initializes the data members to some default values, which are later replaced by ones from de-serialization:

[PRE2]

For purposes of easily manipulating velocity within a certain range, we provide the `AddVelocity` method:

[PRE3]

After adding the provided argument velocity, the end result is checked for being higher than the maximum allowed value on each axis. If it is, the velocity is capped at the maximum allowed value with the appropriate sign.

[PRE4]

Applying friction to the current velocity is also regulated. In order to avoid friction forcing the velocity to change its sign, it's checked to not be equal to zero, as well as if the difference between absolute values of current velocity and provided friction isn't going to be negative. If it is, the velocity is set to zero. Otherwise, the friction value is added to current velocity with an appropriate sign.

In order for an entity to move, it has to be accelerated. Let's supply a method for that:

[PRE5]

For the sake of convenience, we provide the same method, overloaded to take in two types of arguments: a float vector and two separate float values. All it does is simply add the argument values to current acceleration.

Lastly, entities can also be moved based on a provided direction, instead of calling the `Accelerate` method manually:

[PRE6]

Based on the direction provided as an argument, the entity's speed is added to the acceleration vector.

## The movement system

With the movement component designed, let's take a stab at implementing the actual system that will move our entities around:

[PRE7]

First, an `Axis` enumeration is created, in order to simply the code in one of the private helper methods of this class. We then forward-declare a `Map` class, in order to be able to use it in the header. With that, comes a `Map` data member, as well as a public method for providing the movement system with an instance of `Map`. A few private helper methods are also needed in order to make the code more readable. Let's begin by setting up our constructor:

[PRE8]

The requirements for this system consist of two components: position and movable. In addition to that, this system also subscribes to the `Is_Moving` message type, in order to respond to it.

Next, let's update our entity information:

[PRE9]

As the requirements of this system suggest, it will be operating on the position component, as well as the movable component. For every entity that belongs to this system, we want to update its physics and adjust its position in accordance to its velocity and the time that has passed in between frames, producing movement based on forces.

Let's take a look at the movement step method:

[PRE10]

The friction value of the tile an entity is standing on is obtained first. It gets applied to the entity's movable component right after its velocity is updated based on the acceleration value.

Next, we must make sure that diagonal movement is handled correctly. Consider the following illustration:

![The movement system](img/B04284_09_01.jpg)

According to the Pythagorean theorem, the squared hypotenuse of a right triangle, which represents diagonal movement, is equal to the sum of its squared sides. In other words, its hypotenuse is shorter than the sum of both of its sides. Characters that move down-right, for example, would appear to move faster than they do in a single direction, unless we cap their velocity based on the magnitude of the velocity vector, also known as the hypotenuse of the triangle in our illustration. Once the magnitude is calculated, it is checked for exceeding the maximum possible velocity of an entity. If it does, it gets normalized and multiplied by the value of maximum velocity, in order to impose slower diagonal movement.

Obtaining the tile friction can be done like so:

[PRE11]

A tile pointer is set up before a `while` loop is initiated. It will keep trying to fetch a tile at a provided location while decreasing the elevation each time. This means that tile friction is effectively yielded from the top-most tile that a player is over. If a tile hasn't been found, the default friction value is returned instead.

As you might be able to guess by now, the movement system needs to respond to quite a few events, due to its importance:

[PRE12]

First, it handles two colliding events, to which it responds by calling the private `StopEntity` method in order to halt an entity on a specified axis. Next, we have four movement events. In cases of `Moving_Left` and `Moving_Right`, the private `SetDirection` method is invoked in order to update the direction of an entity. Moving up and down, however, is a little bit different. We want the entity's direction to only change if it has no velocity on the *x* axis. Otherwise, it ends up moving rather cheesily.

Next up, message handling:

[PRE13]

Here, we're only concerned with a single message type: `Is_Moving`. It's a message, designed to trigger another one being sent when the entity becomes idle. First, the system is checked for having the entity in question. Its movable component is then acquired, the velocity of which is checked for being at absolute zero. Given that that's the case, an event is created to signify the entity becoming idle.

All we have left now are the private helper methods. It's all of the redundant logic, the existence of which within methods saves us from code duplication. The first one we'll examine is responsible for halting an entity:

[PRE14]

After obtaining its movable component, the entity then has its velocity set to zero on an axis, provided as the argument to this method.

[PRE15]

The `SetDirection` method updates the direction of a movable component. A message is then dispatched to notify all the other systems of this change.

Finally, we're down to a single setter method for the `Map` class:

[PRE16]

In order for entities to have dynamic friction, the movement system has to have access to the `Map` class, so it gets set up in the game state:

[PRE17]

This last code snippet concludes the implementation of the movement system. Our entities are now able to move, based on the forces inflicted on them. Having support for movement, however, does not actually generate movement. This is where the entity state system comes in.

## Implementing states

Movement, much like many other actions and events that are relevant to entities are contingent upon their current state being satisfactory. A dying player should not be able to move around. Relevant animations should be played, based on its current state. Enforcing those laws requires the entity to have a state component:

[PRE18]

As you can tell already, this is a very simple chunk of code. It defines its own enumeration of possible entity states. The component class itself simply provides a setter and a getter, as well as the required method for de-serialization. The rest is, as always, left up to the system to hash out.

### The state system

Since most of the system headers from here on out are going to look pretty much the same, they will be omitted. With that said, let's begin by implementing the constructor and destructor of our state system:

[PRE19]

All this system requires is the state component. It also subscribes to two message types: `Move` and `Switch_State`. While the latter is self-explanatory, the `Move` message is what gets sent by the methods in the game state in order to move the player. Because movement is entirely dependent on the entity state, this is the only system that handles this type of message and determines whether the state is appropriate for motion.

Next, let's take a look at the `Update` method:

[PRE20]

All that happens here is a simple check of the entity's current state. If it's in motion, a message `Is_Moving` is dispatched. If you recall, this type of message is handled by the movement system, which fires an event when the entity becomes idle. That event is handled by our state system:

[PRE21]

All it does is invoke a private method `ChangeState`, which alters the current state of an entity to `Idle`. The third argument here is simply a flag for whether the state change should be forced or not.

The last public method we'll be dealing with here is `Notify`:

[PRE22]

The `Move` message is handled by obtaining the state of an entity it targets. If the entity isn't dying, a `Moving_X` event is constructed based on which direction the message holds. Once the event is dispatched, the entity's state is changed to `Walking`.

The `Switch_State` message simply alters the current state of an entity without forcing it, by invoking this private method:

[PRE23]

After the state is obtained, the `l_force` flag is checked. If it's set to `false`, the state is only altered if the entity isn't currently `DYING`. We don't want anything to snap entities out of death randomly. The state is changed regardless of that, if the `l_force` flag is set to `true`.

Now we have control over what can happen to an entity, based on its current state. With that in place, the entities are now ready to be controlled.

## The entity controller

The idea behind having a separate system be responsible for moving an entity around is not only that we get to decide which entities are capable of being moved, but also further separation of logic, and hooks for future A.I. implementations. Let's take a look at the controller component:

[PRE24]

Yes, it's just an empty component, that is simply used as a way to tell the control system that the entity it belongs to can be controlled. There might be some additional information it needs to store in the future, but for now, it's simply a "flag."

The actual control system is extremely simple to implement. Let's begin with the constructor:

[PRE25]

It imposes requirements for position, movable and controller components, in order to be able to move the entity, which is the only purpose of this system. The actual movement is handled by processing entity events like so:

[PRE26]

All four events invoke the same private method, which simply calls the `Move` method of a movable component and passes in the appropriate direction:

[PRE27]

After this humble addition to our code-base, we can finally move the player around with the keyboard:

![The entity controller](img/B04284_09_02.jpg)

The only problem now is entities looking like they're sliding on ice, due to complete lack of animations. To resolve this issue, the animation system must be introduced.

# Animating the entities

If you recall from previous chapters, the `SpriteSheet` class we built already has great support for animations. There is no reason to annex that at this point, especially since we're only dealing with sprite-sheet based graphics. This saves us a lot of time and allows sprite-sheet animations to be handled by a single system, with no additional component overhead.

Let's start implementing the sprite sheet animation system, as always, by getting the constructor out of the way:

[PRE28]

Since entity animations are, so far, entirely state-based, this system requires a state component, in addition to the sprite sheet component. It also subscribes to the `State_Changed` message type in order to respond to state changes by playing the appropriate animation. Updating all of the entities is the area where this system has most of its logic, so let's take a look at the `Update` method:

[PRE29]

First, both the sprite sheet and state components are obtained. The sprite sheet is then updated and the current name of the animation is retrieved. If an attack animation is no longer playing, a message of `Switch_State` type is sent out in order to put the entity back to an `Idle` state. Otherwise, the animation is checked for currently being within the "action" frame range, which is specified in the sprite sheet file. If it is, an `Attack_Action` message is sent out to the current entity, which can later be used by different systems to implement combat. On the other hand, if the death animation has concluded, a `Dead` message is dispatched.

Next, let's work on handling messages:

[PRE30]

All possible messages this system would be interested in deal with specific entities, so that check is made first. For now, we'll only be dealing with a single message type: `State_Changed`. Every time a state is changed, we'll be altering the animation of the entity. The only possible exception here is the `Hurt` state, which will be dealt with later.

The last bit of code we need is the private `ChangeAnimation` method:

[PRE31]

After obtaining the entity's sprite sheet component, it simply invokes its `SetAnimation` method to change the current animation that's playing. This code is redundant enough to warrant a separate method.

Upon successful compilation, we can see that our entities are now animated:

![Animating the entities](img/B04284_09_03.jpg)

# Handling collisions

Making entities bump into each other, as well as into all the lush environments we'll be building is a mechanic, without which most games out there would not be able to function. In order for that to be possible, these animated images zooming around the screen must have a component, which represents their solidity. Bounding boxes worked really well for us in the past, so let's stick to them and begin constructing the collidable body component:

[PRE32]

Every collidable entity must have a bounding box that represents the solid portion of it. That's exactly where the `m_AABB` rectangle comes in. In addition to that, the bounding box itself can be offset by a number of pixels, based on what kind of entity it is, as well as have a different origin. Lastly, we want to keep track of whether an entity is currently colliding on any given axis, which warrants the use of `m_collidingOnX` and `m_collidingOnY` flags.

The constructor of this component might look a little something like this:

[PRE33]

After initializing the default values to some of its data members, this component, like many others, needs to have a way to be de-serialized:

[PRE34]

Here are a few unique setter and getter methods that we'll be using:

[PRE35]

Finally, we arrive at the key method of this component, `SetPosition`:

[PRE36]

In order to support different types of origins, the position of the bounding box rectangle must be set differently. Consider the following illustration:

![Handling collisions](img/B04284_09_04.jpg)

The origin of the actual bounding box rectangle is always going to be the top-left corner. To position it correctly, we use its width and height to compensate for differences between several possible origin types.

## The collision system

The actual collision magic doesn't start happening until we have a system responsible for accounting for every collidable body in the game. Let's begin by taking a look at the data types that are going to be used in this system:

[PRE37]

For proper collision detection and response, we're also going to need a data structure that is capable of holding collision information, which can later be sorted and processed. For that, we're going to be using a vector of `CollisionElement` data types. It's a structure, consisting of a float, representing area of collision, a pointer to a `TileInfo` instance, which carries all of the information about a tile, and a simple float rectangle, which holds the bounding box information of a map tile.

In order to detect collisions between entities and tiles, the collision system needs to have access to a `Map` instance. Knowing all of that, let's get started on implementing the class!

### Implementing the collision system

As always, we're going to be setting up the component requirements right inside the constructor of this class:

[PRE38]

As you can see, this system imposes requirements of position and collidable components on entities. Its `m_gameMap` data member is also initialized to `nullptr`, until it gets set up via the use of this method:

[PRE39]

Next up is the oh-so-common update method that makes everything behave as it should:

[PRE40]

For clarity, the update method uses two other helper methods: `CheckOutOfBounds` and `MapCollisions`. While iterating over all collidable entities, this system obtains their position and collidable component. The latter is updated, using the entity's latest position. It also has its Boolean collision flags reset. After all entities have been updated, the private `EntityCollisions` method is invoked to process entity-on-entity intersection tests. Note the very beginning of this method. It immediately returns in case the map instance hasn't been properly set up.

First, the entity is checked for being outside the boundaries of our map:

[PRE41]

If the entity has somehow ended up outside the map, its position gets reset.

At this point, we begin running the tile-on-entity collision test:

[PRE42]

A collision information vector named `c` is set up. It will contain all the important information about what the entity is colliding with, the size of the collision area and properties of the tile it's colliding with. The entity's bounding box is then obtained from the collidable component. A range of coordinates to be checked is calculated, based on that bounding box, as shown here:

![Implementing the collision system](img/B04284_09_05.jpg)

Those coordinates are immediately put to use, as we begin iterating over the calculated range of tiles, checking for collisions:

[PRE43]

Once a solid tile is encountered, its bounding box, tile information and area of intersection details are gathered and inserted into the vector `c`. It's important to stop the layer loop if a solid tile is detected, otherwise collision detection may not function properly.

After all the solids the entity collides with in the calculated range have been found, they all must be sorted:

[PRE44]

After sorting, we can finally begin resolving collisions:

[PRE45]

Since resolution of one collision could potentially resolve another as well, the bounding box of an entity must be checked for intersections here as well, before we commit to resolving it. The actual resolution is pretty much the same as it was in [Chapter 7](ch07.html "Chapter 7. Rediscovering Fire – Common Game Design Elements"), *Rediscovering Fire – Common Game Design Elements*.

Once the resolution details are calculated, the position component is moved based on it. The collidable component has to be updated here as well, because it would end up getting resolved multiple times and moved incorrectly otherwise. The last bit we need to worry about is adding a collision event to the entity's event queue and calling the `CollideOnX` or `CollideOnY` method in the collidable component to update its flags.

Now for entity-on-entity collisions:

[PRE46]

This method checks all entities against all other entities for collisions between their bounding boxes, by using the `intersects` method, kindly provided by SFML's rectangle class. For now, we don't have to worry about responding to these types of collisions, however, we will be using this functionality in future chapters.

Lastly, just like its movement counterpart, the collision system requires a pointer to the `Map` class, so let's give it one in the game state's `OnCreate` method:

[PRE47]

This final code snippet gives the collision system all of the power it needs, in order to keep the entities from walking through solid tiles, as so:

![Implementing the collision system](img/B04284_09_06.jpg)

# Summary

Upon completing this chapter, we've successfully moved away from inheritance-based entity design and reinforced our code-base with a much more modular approach, thus avoiding many pitfalls that composition leaves behind. A chain is only as strong as its weakest link, and now we can rest assured that the entity segment will hold.

In the next two chapters, we will be discussing how to make the game more interactive and user friendly by adding a GUI system, as well as adding a few different types of elements, managing their events and providing room for them to be graphically customizable. See you there!