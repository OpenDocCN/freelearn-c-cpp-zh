# Chapter 3. Get Your Hands Dirty – What You Need to Know

Game development can often be a tedious process to bear. In many instances, the amount of time spent on writing a specific chunk of code, implementing a certain set of features, or revising an old code that you or someone else had written shows very few results that can be immediately appreciated; which is why you may at some point see a game developer's face light up in instant joy when a flashier segment of their project sees the light of day. Seeing your game actually come to life and begin changing before your very eyes is the reason most of our fellow game developers do what they do. Those moments make writing tons of code that show little stimulating results possible.

So, now that we have our game structure ready, it's time to focus on the fun, flashy parts!

In this chapter, we will cover:

*   The game of choice for our first project and its history
*   Building the game we've chosen
*   Common game programming elements
*   Additional SFML elements needed to complete our project
*   Building helper elements for all our game projects
*   Effective debugging and common problem solving in games

# Introducing snake

If right now you're imagining building a game with *Solid Snake* wearing his trademark bandana, we're not quite there yet, although the eagerness to do so is understandable. However, if you pictured something like the following, you're right on point:

![Introducing snake](img/4284_03_01.jpg)

First published by *Gremlin* in 1976 under the name "Blockade", the snake concept is one of the most famous game types of all time. Countless ports have been written for this type of mechanic, such as *Surround* by *Atari* in 1978 and *Worm* by *Peter Trefonas*. Pretty much any platform that crosses one's mind has a port of snake on it, even including the early monochrome *Nokia* phones, such as the *3310* and *6110*. The graphics changed from port to port and improved with time. However, the main idea and the rules remained the same ever since its humble beginnings:

*   The snake can move in four total directions: up, down, left, and right
*   Eating an apple makes the snake grow in length
*   You cannot touch the walls or your own body, otherwise the game is over

Other things may vary depending on which version of the game you play, such as the score you receive for eating an apple, the amount of lives you have, the speed at which the snake moves, the size of the playing field, obstacles, and so on.

# Game design decisions

Certain versions of snake run differently; however, for the sake of paying homage to the classical approach, we will be implementing a snake that moves based on a **grid**, as illustrated next:

![Game design decisions](img/4284_03_02.jpg)

Taking this approach makes it easier to later check for collision between the snake segments and the apple. Grid movement basically means updating at a static rate. This can be achieved by utilizing a fixed time-step, which we covered back in [Chapter 2](ch02.html "Chapter 2. Give It Some Structure – Building the Game Framework"), *Give It Some Structure – Building the Game Framework*.

The outside area symbolizes the boundaries of the game, which in the case of a grid-based movement would be in the range of *[1;Width-1]* and *[1;Height-1]*. If the snake head isn't within that range, it's safe to say that the player has crashed into a wall. All the grid segments here are 16px by 16px big; however, that can be adjusted at any time.

Unless the player runs out of lives, we want to cut the snake at the point of intersection if its head collides with its body and decrease the amount of lives left. This adds a little variety to the game without being too unbalanced.

Lastly, you've probably already picked up on the fact that we're using very simplistic graphical representations of what a snake is in this game. This is done mainly to keep things simple for now, as well as to add the charm of a classic to the mix. It wouldn't be terribly complicated to use sprites for this, however, let's not worry about that just yet.

# Implementing the snake structure

Let's now create the two files we'll be working with: `Snake.h` and `Snake.cpp`. Prior to actually developing the snake class, a definition of some data types and structures is in order. We can begin by actually defining the structure that our apple eating serpent will be made out of, right in the snake header file:

[PRE0]

As you can tell, it's a very simple structure that contains a single member, which is an *integer vector* representing the position of the segment on the grid. The constructor here is utilized to set the position of the segment through an *initializer list*.

### Tip

Before moving past this point, make sure you're competent with the **Standard Template Library** and the data containers it provides. We will specifically be using `std::vector` for our needs.

We now have the segment type defined, so let's get started on actually storing the snake somewhere. For beginner purposes, `std::vector` will do nicely! Before going too far with that, here's a neat little trick for curing our code of "long-line-itus":

[PRE1]

As you should already know from your *C/C++* background, `using` is a neat little keyword that allows the user to define aliases for the known data types. By using our clean new definitions together with the `auto` keyword, we're preventing a scenario like the following from ever happening:

[PRE2]

It's a simple matter of convenience and is completely optional to use, however, we will be equipping this useful tool all the way through this book.

One last type we need to define before beginning to really work on the snake class, is the direction enumeration:

[PRE3]

Once again, it's nothing too fancy. The snake has four directions it can move in. We also have a possibility of it standing still, in which case we can just set the direction to `NONE`.

# The snake class

Before designing any object, one must ask oneself what it needs. In our case, the snake needs to have a direction to move towards. It also needs to have lives, keep track of the score, its speed, whether it lost or not, and whether it lost or not. Lastly, we're going to store a rectangle shape that will represent every segment of the snake. When all these are addressed, the header of the snake class would look something like the following:

[PRE4]

Note that we're using our new type alias for the snake segment vector. This doesn't look that helpful just yet, but it's about to be, really soon.

As you can see, our class has a few methods defined that are designed to split up the functionality, such as `Lose()`, `Extend()`, `Reset()`, and `CheckCollision()`. This will increase code re-usability as well as readability. Let's begin actually implementing these methods:

[PRE5]

The constructor is pretty straightforward. It takes one argument, which is the size of our graphics. This value gets stored for later use and the member of type `sf::RectangleShape` gets its size adjusted based on it. The subtraction of one pixel from the size is a very simple way of maintaining that the snake segments appear visually slightly separated, as illustrated here:

![The snake class](img/4284_03_03.jpg)

The constructor also calls the `Reset()` method on the last line. A comment in the header file states that this method is responsible for moving the snake into its starting position. Let's make that happen:

[PRE6]

This chunk of code will be called every time a new game begins. First, it will clear the snake segment vector from the previous game. After that, some snake segments will get added. Because of our implementation, the first element in the vector is always going to be the head. The coordinates for the snake pieces are hardcoded for now, just to keep it simple.

Now we have a three-piece snake. The first thing we do now is set its direction to `None`. We want no movement to happen until a player presses a key to move the snake. Next, we set up some arbitrary values for the speed, the lives, and the starting score. These can be adjusted to your liking later. We also set the `m_lost` flag to `false` in order to signify a new round taking place.

Before moving on to more difficult to implement methods, let's quickly cover all the helper ones:

[PRE7]

These methods are fairly simple. Having descriptive names helps a lot. Let's take a look at the `Extend` method now:

[PRE8]

This preceding method is the one responsible for actually growing out our snake when it touches an apple. The first thing we did was create a reference to the *last* element in the segment vector, called `tail_head`. We have a fairly large *if-else statement* chunk of code next, and both cases of it require access to the last element, so it's a good idea to create the reference now in order to prevent duplicated code.

### Tip

The `std::vector` container overloads the **bracket operator** in order to support random access via a numeric index. It being similar to an array enables us to reference the last element by simply using an index of `size() - 1`. The random access speed is also constant, regardless of the number of elements in this container, which is what makes the `std::vector` a good choice for this project.

Essentially, it comes down to two cases: either the snake is longer than one segment or it's not. If it does have more than one piece, we create another reference, called `tail_bone`, which points to the *next to last* element. This is needed in order to determine where a new piece of the snake should be placed upon extending it, and the way we check for that is by comparing the `position.x` and `position.y` values of the `tail_head` and `tail_bone` segments. If the x values are the same, it's safe to say that the difference between the two pieces is on the y axis and vice versa. Consider the following illustration, where the orange rectangle is `tail_bone` and the red rectangle is `tail_head`:

![The snake class](img/4284_03_04.jpg)

Let's take the example that's facing left and analyze it: `tail_bone` and `tail_head` have the same *y* coordinate, and the *x* coordinate of `tail_head` is greater than that of `tail_bone`, so the next segment will be added at the same coordinates as `tail_head`, except the x value will be increased by one. Because the `SnakeSegment` constructor is conveniently overloaded to accept coordinates, it's easy to perform this simple math at the same time as pushing the segment onto the back of our vector.

In the case of there only being one segment in the vector, we simply check the direction of our snake and perform the same math as we did before, except that this time it's based on which way the head is facing. The preceding illustration applies to this as well, where the orange rectangle is the head and the red rectangle is the piece that's about to be added. If it's facing left, we increase the *x* coordinate by one while leaving *y* the same. Subtracting from x happens if it's facing right, and so on. Take your time to analyze this picture and associate it with the previous code.

Of course, none of this would matter if our snake didn't move. That's exactly what is being handled in the update method, which in our case of a *fixed time-step* is referred to as a "tick":

[PRE9]

The first two lines in the method are used to check if the snake should be moved or not, based on its size and direction. As mentioned earlier, the `Direction::None` value is used specifically for the purpose of keeping it still. The snake movement is contained entirely within the `Move` method:

[PRE10]

We start by iterating over the vector *backwards*. This is done in order to achieve an *inchworm* effect of sorts. It is possible to do it without iterating over the vector in reverse as well, however, this serves the purpose of simplicity and makes it easier to understand how the game works. We're also utilizing the *random access operator* again to use numeric indices instead of the vector *iterators* for the same reasons. Consider the following illustration:

![The snake class](img/4284_03_05.jpg)

We have a set of segments in their positions before we call the `tick` method, which can be referred to as the "beginning state". As we begin iterating over our vector backwards, we start with the segment #3\. In our `for` loop, we check if the index is equal to `0` or not in order to determine if the current segment is the front of the snake. In this case, it's not, so we set the position of segment #3 to be the *same* as the segment #2\. The preceding illustration shows the piece to be, sort of, in between the two positions, which is only done for the purpose of being able to see both of them. In reality, segment #3 is sitting right on top of segment #2.

After the same process is applied again to the second part of the snake, we move on to its head. At this point, we simply move it across one space in the axis that corresponds to its facing direction. The same idea applies here as it did in the illustration before this one, but the sign is reversed. Since in our example, the snake is facing right, it gets moved to the coordinates *(x+1;y)*. Once that is done, we have successfully moved our snake by one space.

One last thing our tick does is call the `CheckCollision()` method. Let's take a look at its implementation:

[PRE11]

First, there's no need to check for a collision unless we have over four segments. Understanding certain scenarios of your game and putting in checks to not waste resources is an important part of game development. If we have over four segments of our snake, we create a reference to the head again, because in any case of collision, that's the first part that would hit another segment. There is no need to check for a collision between all of its parts twice. We also skip an iteration for the head of the snake, since there's obviously no need to check if it's colliding with itself.

The basic way we check for a collision in this grid-based game is essentially by comparing the position of the head to the position of the current segment represented by our iterator. If both positions are the same, the head is intersecting with the body. The way we resolve this was briefly covered in the *Game design decisions* section of this chapter. The snake has to be cut at the point of collision until the player runs out of lives. We do this by first obtaining an integer value of the segment count between the end and the segment being hit. STL is fairly flexible with its iterators, and since the memory in the case of using a vector is all laid out contiguously, we can simply subtract our current iterator from the last element in the vector to obtain this value. This is done in order to know how many elements to remove from the back of the snake up until the point of intersection. We then invoke the method that is responsible for cutting the snake. Also, since there can only be one collision at a time, we break out of the `for` loop to not waste any more clock cycles.

Let's take a look at the `Cut` method:

[PRE12]

At this point, it's as simple as looping a certain amount of times based on the `l_segments` value and popping the elements from the back of the vector. This effectively slices through the snake.

The rest of the code simply decreases the amount of lives left, checks if it's at zero, and calls the `Lose()` method if there are no more lives.

Phew! That's quite a bit of code. One thing still remains, however, and that is rendering our square serpent to the screen:

[PRE13]

Quite similarly to a lot of the methods we've implemented here, there's a need to iterate over each segment. The head itself is drawn outside of the loop in order to avoid unnecessary checks. We set the position of our `sf::RectangleShape` that graphically represents a snake segment to its grid position multiplied by the `m_size` value in order to obtain the pixel coordinates on the screen. Drawing the rectangle is the last step of implementing the snake class in its entirety!

# The World class

Our snake can now move and collide with itself. While functional, this doesn't make a really exciting game. Let's give it some boundaries and something to munch on to increase the score by introducing the `World` class.

While it's possible to just make separate objects for everything we talk about in here, this project is simple enough to allow certain aspects of itself to be nicely contained within a single class that can manage them without too much trouble. This class takes care of everything to do with keeping the game boundaries, as well as maintaining the apple the player will be trying to grab.

Let's take a look at the class header:

[PRE14]

As you can see from the preceding code, this class also keeps track of how big the objects in the game are. Aside from that, it simply retains four rectangles for the boundary graphics, a circle for drawing the apple, and an integer vector to keep track of the apple's coordinates, which is named `m_item`. Let's start implementing the constructor:

[PRE15]

Up until the complex looking `for` loops, we simply initialize some member values from the local constructor variables, set the color and radius of the apple circle, and call the `RespawnApple()` method in order to place it somewhere on the grid.

The first `for` loop just iterates four times for each of the four sides of the game screen in order to set up a red rectangle wall on each side. It sets a dark red color for the rectangle fill and proceeds with checking the index value. First, we determine if the index is an even or an odd value by checking it with the following expression: `if(!((i + 1) % 2)){...`. This is done in order to know how big each wall has to be on a specific axis. Because it has to be as large as one of the screen dimensions, we simply make the other one as large as all the other graphics on the screen, which is represented by the `m_blockSize` value.

The last `if` statement checks if the index is below two. If it is, we're working with the top-left corner, so we simply set the position of the rectangle to (0,0). Since the origin of all the rectangle-based drawables in SFML is always the top-left corner, we don't need to worry about that in this case. However, if the index is 2 or higher, we set the origin to the size of the rectangle, which effectively makes it the bottom right corner. Afterwards, we set the position of the rectangle to be the same as the size of the screen, which puts the shape all the way down to the bottom right corner. You can simply set all the coordinates and origins by hand, but this approach makes the initialization of the basic features more automated. It may be hard to see the use for it now, but in more complicated projects this kind of thinking will come in handy, so why not start now?

Since we have our walls, let's take a look at how one might go about re-spawning the apple:

[PRE16]

The first thing we must do is determine the boundaries within which the apple can be spawned. We do so by defining two values: `maxX` and `maxY`. These are set to the window size divided by the block size, which gives us the number of spaces in the grid, from which we must then subtract 2\. This is due to the fact that the grid indices begin with 0, not 1, and because we don't want to spawn the apple within the right or bottom walls.

The next step is to actually generate the random values for the apple coordinates. We use our pre-calculated values here and set the *lowest* possible random value to `1`, because we don't want anything spawning in the top wall or the left wall. Since the coordinates of the apple are now available, we can set the `m_appleShape` graphic's position in pixel coordinates by multiplying the grid coordinates by the size of all our graphics.

Let's actually make all these features come to life by implementing the update method:

[PRE17]

First, we check if the player's position is the same as that of the apple. If it is, we have a collision and the snake gets extended, the score increases, and the apple gets re-spawned. Next, we determine our grid size and check if the player coordinates are anywhere outside of the designated boundaries. If that's the case, we call the `Lose()` method to illustrate the collision with the wall and give the player a "game over".

In order to not keep the player blind, we must display the boundaries of the game, as well as the main point of interest - the apple. Let's draw everything on screen:

[PRE18]

All we have to do is iterate four times and draw each of the four respective boundaries. Then we draw the apple, which concludes our interest in this method.

One more thing to point out is that the other classes might need to know how big the graphics need to be, and for this reason, let's implement a simple method for obtaining that value:

[PRE19]

This concludes the `World` class.

# Time to integrate

Much like how a hammer is useless without someone using it, so are our two classes without being properly adopted by the `Game` class. Since we didn't write all that code just to practise typing, let's work on putting all the pieces together. First, we need to actually add two new members to the `Game` class, and you might already have guessed what they are:

[PRE20]

Next, let's initialize these members. Since both of them have constructors that take arguments, it's the time for *initializer list*:

[PRE21]

Next, we need to process some input. As you may recall from the previous chapters, utilizing events for live input is really delayed and should never be used for anything else but checking for key presses that aren't time sensitive. Luckily, SFML provides means of obtaining the real-time state of the keyboard through the `sf::Keyboard` class. It only contains the static functions and is never meant to be initialized. One of those functions is exactly what we need here: `isKeyPressed(sf::Keyboard::Key)`. The sole argument that it takes is the actual key you want to check the state of, which can be obtained through the use of the `sf::Keyboard::Key` enumeration, as follows:

[PRE22]

Something we don't want the snake to do is to go in the direction that is opposite to its current one. At any given time, there should only be three directions it can go in, and the use of the `GetDirection()` method ensures that we don't send the snake in reverse, essentially eating itself. If we have the proper combination of input and its current direction, it's safe to adjust its direction through the use of the `SetDirection()` method.

Let's get things moving by updating both our classes:

[PRE23]

As mentioned previously, we're using *fixed time-step* here, which incorporates the snake speed in order to update the appropriate amount of times per second. This is also where we check if the player has lost the game and reset the snake if he has.

We're really close now. Time to draw everything on screen:

[PRE24]

Much like before, we simply invoke the `Render` methods of both our classes and pass in a reference to `sf::RenderWindow`. With that, our game is actually playable! Upon successful compilation and execution of our project, we should end up with something looking like this following image:

![Time to integrate](img/4284_03_06.jpg)

The snake will be still at first, until one of the four arrow keys is pressed. Once it does start moving, it will be able to eat the apple and grow by one segment, collide with its own tail and lose it twice before it dies, and end the game if the player crashes into a wall. The core version of our game is complete! Pat yourself on the back, as you just created your first game.

# Hunting bugs

As proud and satisfied as you may be with your first project, nothing is ever perfect. If you've spent some time actually playing the game, you may have noticed an odd event when quickly mashing the buttons, looking something like this:

![Hunting bugs](img/4284_03_07.jpg)

The image represents the difference between two sequential updates. It seems that earlier it was facing the right direction and then it's facing left and missing its tail. What happened? Try to figure it out on your own before continuing, as it perfectly illustrates the experience of fixing game flaws.

Playing around with it some more reveals certain details that narrow down our problem. Let's break down what happens when a player starts mashing keys quickly:

*   The snake is facing right.
*   Any arrow key other than the left or right is pressed.
*   The direction of the snake gets set to something else, let's say up.
*   The right key is pressed before the game has a chance to update.
*   Since the snake's direction is no longer set to right or left, `if` statement in the input handler is satisfied and sets the direction to left.
*   The game updates the snake and moves it left by one space. The head collides with its tail and it gets cut off.

Yes, it seems that our direction checking is flawed and causes this bug. Once again, spend some time trying to think of a way to fix this before moving on.

# Fixing bugs

Let's discuss the several approaches that might be used in a situation like this. First, the programmer might think about putting a flag somewhere that remembers if the direction has already been set for the current iteration and gets reset afterwards. This would prevent the bug we're experiencing, but would also lock down the number of times a player can interact with the snake. Let's say it moves once a second. That would mean that if you press a key at the beginning of that second, you wouldn't be able to change your mind and hit another key quickly to rectify your wrong decision before the snake moves. That's no good. Let's move on to a new idea.

Another approach may be to keep track of the original direction before any changes were made to that *iteration*. Then, once the update method gets called, we could check if the original direction, before any changes were made, is the opposite of the newest direction that we've received. If it is, we could simply ignore it and move the snake in the direction before any changes were made. This would fix the bug and not present us with a new one, but it comes with keeping track of one more variable and might get confusing. Imagine that in the future you're presented with a similar bug or a request for a feature that needs you to keep track of another variable on top of this one. Imagine that happens one more time, then another. Very soon, your checking statement might look a little something like this:

[PRE25]

Now that is what we call a mess. On top of that, imagine you have to check the same variables four times for four different conditions. It quickly becomes apparent that this is a bad design and it shouldn't be used by anyone with intentions of ever showing their code to another person.

You may ask how we can rectify our problem then. Well, we could simply not rely on the use of a variable in the snake class to determine its direction, and instead implement a method that looks at its structure and spits out the direction it's facing, as shown next:

[PRE26]

First, we check if the snake is *1* segment long or less; in this case, it doesn't matter which direction it's facing as it wouldn't eat itself if it only had a head, and it wouldn't even have a direction if there are no segments in the vector at all. Assuming it's longer than one segment, we obtain two references: the head and the neck, which is the second piece of the snake right after the head. Then, we simply check the positions of both of them and determine the direction the snake is facing using the same logic as before, while implementing the snake class, as illustrated in the following image:

![Fixing bugs](img/4284_03_08.jpg)

This will return a proper direction that won't be altered unless the snake moves, so let's adjust our input handling code to cater to these changes:

[PRE27]

Voila! No more of our snake turning inside out.

There's one more fault with the game that didn't get addressed here on purpose. Try to find it and fix it in order to practise resolving problems like this in the future.

### Tip

Hint: It has to do with how many segments the snake has when the game starts.

If you want to do this one fairly, do your best not to reference the code of the finished project that came with this book, as that has it fixed already.

# Going the extra mile

A functional game is far from a fully finished product. Sure, we have everything we wanted in the beginning, but it still leaves things to be desired, such as keeping track of the score and showing how many lives we have. At first, your main instinct might be to just add a bit of text somewhere on the screen that simply prints the number of lives you have left. You may even be tempted to do as little as simply printing it out in the console window. If that's the case, the purpose of this part is to change your way of thinking by introducing something that we will be using and improving over the course of this book: the textbox.

If that name doesn't really mean anything to you, simply imagine a chat window on any given communication application, such as *MSN Messenger* or *Skype*. Whenever a new message is added, it's added to the bottom as the older messages are moved up. The window holds a certain number of messages that are visible at one time. That's not only useful for the purpose of the game printing a casual message, but can also be used for debugging. Let's start by writing our header, as usual:

[PRE28]

We begin by defining the data type for the container of all the messages. In this case, we went with `std::vector` again, simply because that's the more familiar choice at this point. Just to make it look better and more readable, we've added a rectangle shape as one of the members of the class that will be used as a backdrop. On top of that, we have introduced a new data type: `sf::Text`. This is a drawable type that represents any typed characters or strings of characters, and can be adjusted in size, font, and color, as well as transformed, much like any other drawable in SFML.

Let's start implementing our fancy new feature:

[PRE29]

As you can see, it has two constructors, one of which can be used to initialize some default values and the other that allows customization by passing in some values as arguments. The first argument is the number of lines that are visible in the textbox. It is followed by the character size in pixels, the width of the entire textbox in pixels, and float vector that represents the position on the screen where it should be drawn at. All that these constructors do is invoke the `Setup` method and pass all these arguments to it, so let's take a look at it:

[PRE30]

Aside from initializing its member values, this method defines an offset float vector that will be used to space the text appropriately and provide some padding from the top-left corner. It also sets up our `sf::Text` member by first creating a font to which it's bound, setting the initial string to nothing, setting up the character size and color, and setting its position on the screen to the provided position argument with the proper offset factored in. Additionally, it sets up the size of the backdrop by using the width that was provided and multiplying the number of visible lines by the result of the multiplication of the character size and a constant floating point value of 1.2, in order to account for spacing between the lines.

### Tip

From time to time, it does simply come down to playing with code to seeing what really works. Finding certain numeric constants that work in all cases is one of the situations where it's just a matter of testing in order to determine the correct value. Don't be afraid to try out new things and see what works.

Since we're utilizing a vector to store our messages, adding a new one or removing them all is as simple as using the `push_back` and `clear` methods:

[PRE31]

In the case of adding a new message, checking whether we have more of them than we can see would be a good idea. Having something around that we're not going to see or need ever again is wasteful, so the very first message that is definitely out of sight at that time is removed from the message container.

We're very close to actually finishing this neat feature. The only thing left now is drawing it, which, as always, is taken care of by the `Render` method:

[PRE32]

The code begins with `std::string` being set up to hold all the visible messages on the screen. Afterwards, it's as simple as looping over the message vector and appending the text of each message to our local `std::string` variable with a new line symbol at the end. Lastly, after checking the local variable and making sure it isn't empty, we must set our `m_content` member of type `sf::Text` to hold the string we've been pushing our messages to and draw both the background and the text on the screen. That's all there is to the `Textbox` class.

After adding an instance of `Textbox` as a member to our game class, we can start setting it up:

[PRE33]

After passing some constant values to the `Setup` method of our `m_textbox` member, we immediately start using it right there in the constructor by actually outputting our first message. Let's finish integrating it fully by making one last adjustment to the `Game::Render()` method:

[PRE34]

It's the same as both the classes we've implemented before this, except that the text box is now the last thing we draw, which means it will be displayed over everything else. After adding more messages to the game to be printed and compiling our project, we should end up with something like this:

![Going the extra mile](img/4284_03_09.jpg)

This text box, in its most basic form, is the last addition to our snake game that we will be covering in this book. Feel free to play around with it and see what else you can come up with to spice up the game!

# Common mistakes

A fairly common thing people often forget is the following line:

[PRE35]

If you notice, the numbers being generated are exactly the same each time you launch the game-chances are that you haven't seeded the random number generator or you haven't provided a proper seed. It's recommended to always use a unix timestamp, as shown.

### Tip

The use of this particular random function should be restricted to something that isn't related to security and cryptography. Using it in combination with the modulus operator can produce incredibly non-uniform results due to the introduced bias.

Another fairly common problem is the programmers' choice of the data container to hold their structures. Let's take the following for example:

[PRE36]

This defines the type of our `SnakeContainer`. If you've compiled the code we've written, you will notice that it runs fairly smoothly. Now consider this next line of code:

[PRE37]

Because of the way these two containers are implemented in STL, nothing else changes in our code, so feel free to try to change the data type of your `SnakeContainer` from `std::vector` to `std::deque`. After compiling and running the project, you will definitely pick up on the hit on performance. Why is that happening? Well, even though `std::vector` and `std::deque` can be used basically in the same way, they're fundamentally different under the hood. The vector offers the certainty of its elements being contiguous in memory, while the double ended queue does not. There are also differences in performances, depending on where the most inserts and removals are done. If you're unsure about which container to use, make sure to either look it up or benchmark it yourself. Never just blindly assume, unless performance isn't the main concern to you.

Lastly, on a more open-ended note, don't be afraid to play with, modify, change, hack, or otherwise alter any piece of code that you see. The biggest mistake you can make is the mistake of not learning by breaking and fixing things. Consider the code we've written as only a push in the right direction and not a specific recipe. If understanding something better means you have to break it first, so be it.

# Summary

Game development is a great journey to embark on. You had taken your first and second steps earlier, but now you have boarded the plane with your first, fully functional game in the bag. You are now officially a game developer! Where will this plane of opportunity take you and how long will it be there for? All of that is entirely up to you. While you're still not in the air, however, we will do our best to inspire you and show you all the different places to go to and the wonderful experiences to be had there. One thing is definitely for sure, however, and that is that this is not the end. If your enthusiasm has led you this far, there's only one direction to head to, and that's forward.

A lot was covered in this chapter, and now it's impossible to say that you haven't gotten your hands dirty while paying homage to one of the all time arcade classics. In the next chapter, we will take on input handling and event management in order to provide flexibility and fluent means of interaction between you and your application, all while introducing our brand new project for the next few chapters. There's still a lot to learn and many lines of code to write, so don't spend too much time hesitating to proceed onto the next chapter. A brand new adventure is waiting to unfold. See you there!