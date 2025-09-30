# Chapter 2. You Plus C++ Plus Cocos2d-x

*This chapter will be aimed at two types of developers: the original Cocos2d developer who is scared of C++ but won't admit it to his friends and the C++ coder who never even heard of Cocos2d and finds Objective-C funny looking.*

*I'll go over the main syntax differences Objective-C developers should pay attention to and the few code style changes involved in developing with Cocos2d-x that C++ developers should be aware of. But first, a quick introduction to Cocos2d-x and what it is all about.*

You will learn the following topics:

*   What Cocos2d-x is and what it can do for you
*   How to create classes in C++
*   How to memory manage your objects in Cocos2d-x and C++
*   What you get out of Ref

# Cocos2d-x – an introduction

So what is a 2D framework? If I had to define it in as few words as possible, I'd say rectangles in a loop.

At the heart of Cocos2d-x, you find the `Sprite` class and what that class does, in simple terms, is keep a reference to two very important rectangles. One is the image (or texture) rectangle, also called the source rectangle, and the other is the destination rectangle. If you want an image to appear in the center of the screen, you will use `Sprite`. You will pass it the information of what and where that image source is and where on the screen you want it to appear.

There is not much that needs to be done to the first rectangle, the source one; but there is a lot that can be changed in the destination rectangle, including its position on the screen, its size, opacity, rotation, and so on.

Cocos2d-x will then take care of all the OpenGL drawing necessary to display your image where you want it and how you want it, and it will do so inside a render loop. Your code will most likely tap into that same loop to update its own logic.

Pretty much any 2D game you can think of can be built with Cocos2d-x with a few sprites and a loop.

### Note

In Version 3.x of the framework, there was a mild separation between Cocos2d-x and its counterpart Cocos2d. It dropped the prefix CC in favor of namespaces, embraced C++11 features, and became that much nicer to work with because of it.

## Containers

Also important in Cocos2d-x is the notion of containers (or nodes). These are all the objects that can have sprites inside them (or other nodes.) This is extremely useful at times because by changing aspects of the container, you automatically change aspects of its children. Move the container and all its children will move with it. Rotate the container and well, you get the picture!

The containers are: `Scene`, `Layer`, and `Sprite`. They all inherit from a base container class called **node**. Each container will have its peculiarities, but basically you will arrange them as follows:

*   `Scene`: This will contain one or more `Node`, usually `Layer` types. It is common to break applications into multiple scenes; for instance, one for the main menu, one for settings, and one for the actual game. Technically, each scene will behave as a separate entity in your application, almost as subapplications themselves, and you can run a series of transition effects when changing between scenes.
*   `Layer`: This will most likely contain `Sprite`. There are a number of specialized `Layer` objects aimed at saving you, the developer, some time in creating things such as menus for instance (`Menu`), or a colored background (`LayerColor`). You can have more than one `Layer` per scene, but good planning makes this usually unnecessary.
*   `Sprite`: This will contain your images and be added as children to `Layer` derived containers. To my mind, this is the most important class in all of Cocos2d-x, so much so, that after your application initializes, when both a `Scene` and a `Layer` object are created, you could build your entire game only with sprites and never use another container class in Cocos2d-x.
*   `Node`: This super class to all containers blurs the line between itself and `Layer`, and even `Sprite` at times. It has its own set of specialized subclasses (besides the ones mentioned earlier), such as `MotionStreak`, `ParallaxNode`, and `SpriteBatchNode`, to name a few. It can, with a few adjustments, behave just as `Layer`. But most of the time you will use it to create your own specialized nodes or as a general reference in polymorphism.

## The Director and cache classes

After containers comes the all-knowing `Director` and all-encompassing cache objects. The `Director` object manages scenes and knows all about your application. You will make calls to it to get to that information and to change some of the things such as screen size, frame rate, scale factor, and so forth.

The caches are collector objects. The most important ones are `TextureCache`, `SpriteFrameCache`, and `AnimationCache`. These are responsible for storing key information regarding those two important rectangles I mentioned about earlier. But every type of data that is used repeatedly in Cocos2d-x will be kept in some sort of cache list.

Both `Director` and all cache objects are singletons. These are special sort of classes that are instantiated only once; and this one instance can be accessed by any other object.

## The other stuff

After the basic containers, the caches and the `Director` object, comes the remaining 90 percent of the framework. Among all this, you will find:

*   **Actions**: Animations will be handled through these and what a treat they are!
*   **Particles**: Particles systems for your delight.
*   **Specialized nodes**: For things such as menus, progress bars, special effects, parallax effect, tile maps, and much, much more.
*   **The macros, structures, and helper methods**: Hundreds of time-saving, magical bits of logic. You don't need to know them all, but chances are that you will be coding something that can be easily replaced by a macro or a helper method and feel incredibly silly when you find out about it later.

## Do you know C++?

Don't worry, the C part is easy. The first plus goes by really fast, but that second plus, oh boy!

Remember, it is C. And if you have coded in Objective-C with the original Cocos2d, you know good old C already even if you saw it in between brackets most of the time.

But C++ also has classes, just like Objective-C, and these classes are declared in the interface files just like in Objective-C. So let's go over the creation of a C++ class.

# The class interface

This will be done in a `.h` file. We'll use a text editor to create this file since I don't want any code hinting and autocompletion features getting in the way of you learning the basics of C++ syntax. So for now at least, open up your favorite text editor. Let's create a class interface!

# Time for action – creating the interface

The interface, or header file, is just a text file with the `.h` extension.

1.  Create a new text file and save it as `HelloWorld.h`. Then, enter the following lines at the top:

    [PRE0]

2.  Next, add the namespace declaration:

    [PRE1]

3.  Then, declare your class name and the name of any inherited classes:

    [PRE2]

4.  Next, we add the properties and methods:

    [PRE3]

5.  We finish by closing the `#ifndef` statement:

    [PRE4]

## *What just happened?*

You created a header file in C++. Let's go over the important bits of information:

*   In C++ you include, you do not import. The `import` statement in Objective-C checks whether something needs to be included; `include` does not. But we accomplish the same thing through that clever use of definitions at the top. There are other ways to run the same check (with `#pragma once`, for instance) but this one is added to any new C++ files you create in Xcode.
*   You can make your life easier by declaring the namespaces you'll use in the class. These are similar to packages in some languages. You may have noticed that all the uses of `cocos2d::` in the code are not necessary because of the namespace declaration. But I wanted to show you the bit you can get rid of by adding a namespace declaration.
*   So next you give your class a name and you may choose to inherit from some other class. In C++ you can have as many super classes as you want. And you must declare whether your super class is public or not.
*   You declare your `public`, `protected` and `private` methods and members between the curly braces. `HelloWorld` is the constructor and `~HelloWorld` is the destructor (it will do what `dealloc` does in Objective-C).
*   The `virtual` keyword is related to overrides. When you mark a method as `virtual,` you are telling the compiler not to set in stone the owner of the method, but to keep it in memory as execution will reveal the obvious owner. Otherwise, the compiler may erroneously decide that a method belongs to the super and not its inheriting class.

    Also, it's good practice to make all your destructors `virtual`. You only need use the keyword once in the super class to mark potential overrides, but it is common practice to repeat the `virtual` keyword in all subclasses so developers know which methods are overrides (C++11 adds a tag `override,` which makes this distinction even clearer, and you will see examples of it in this book's code). In this case, `init` comes from `Layer` and `HelloWorld` wants to override it.

    [PRE5]

*   Oh yes, in C++ you must declare overrides in your interfaces. No exceptions!

The `inline` method is something new to you, probably. These methods are added to the code by the compiler wherever they are called for. So every time I make a call to `addTwoIntegers`, the compiler will replace it with the lines for the method declared in the interface. So the `inline` method works just as statements inside a method; they do not require their own bit of memory in the stack. But if you have a two-line `inline` method called 50 times in your program, it means that the compiler will add a hundred lines to your code.

# The class implementation

This will be done in a `.cpp` file. So let's go back to our text editor and create the implementation for our `HelloWorld` class.

# Time for action – creating the implementation

The implementation is a text file with the `.cpp` extension:

1.  Create a new text file and save it as `HelloWorld.cpp`. At the top, let's start by including our header file:

    [PRE6]

2.  Next, we implement our constructor and destructor:

    [PRE7]

3.  Then comes our static method:

    [PRE8]

4.  And then come our two remaining public methods:

    [PRE9]

## *What just happened?*

We created the implementation for our `HelloWorld` class. Here are the most important bits to take notice of:

*   The `HelloWorld::` scope resolution is not optional here. Every single method declared in your interface belongs to the new class that needs the correct scope resolution in the implementation file.
*   You also need the scope resolution when calling the super class like `Layer::init()`. There is no built-in `super` keyword in the standard C++ library.
*   You use `this` instead of `self`. The `->` notation is used when you're trying to access an object's properties or methods through a pointer to the object (a pointer is the information of where you find the actual object in memory). The `.` (dot) notation is used to access an object's methods and properties through its actual instance (the blob of memory that comprises the actual object).
*   We create an `update` loop, which takes a float for its delta time value simply by calling `scheduleUpdate`. You will see more options related to this later in this book.
*   You can use the `auto` keyword as the type of an object if it's obvious enough to the compiler which type an object is.
*   The `inline` methods, of course, are not implemented in the class since they exist only in the interface.

And that's enough of syntax for now. C++ is one of the most extensive languages out there and I do not wish to leave you with the impression that I have covered all of it. But it is a language made by developers for developers. Trust me, you will feel right at home working with it.

The information listed previously will become clearer once we move on to building the games. But now, onwards to the big scary monster: memory management.

# Instantiating objects and managing memory

There is no **Automatic Reference Counting** (**ARC**) in Cocos2d-x, so Objective-C developers who have forgotten memory management might have a problem here. However, the rule regarding memory management is very simple with C++: if you use `new`, you must delete. C++11 makes this even easier by introducing special pointers that are memory-managed (these are `std::unique_ptr` and `std::shared_ptr`).

Cocos2d-x, however, will add a few other options and commands to help with memory management, similar to the ones we have in Objective-C (without ARC). This is because Cocos2d-x, unlike C++ and very much like Objective-C, has a root class. The framework is more than just a C++ port of Cocos2d. It also ports certain notions of Objective-C to C++ in order to recreate its memory-management system.

Cocos2d-x has a `Ref` class that is the root of every major object in the framework. It allows the framework to have `autorelease` pools and `retain` counts, as well other Objective-C equivalents.

When instantiating Cocos2d-x objects, you have basically two options:

*   Using static methods
*   The C++ and Cocos2d-x style

## Using static methods

Using static methods is the recommended way. The three-stage instantiation process of Objective-C, with `alloc`, `init`, and `autorelease`/`retain`, is recreated here. So, for instance, a `Player` class, which extends `Sprite`, might have the following methods:

[PRE10]

For instantiation, you call the static `create` method. It will create a new `Player` object as an empty husk version of `Player`. No major initialization should happen inside the constructor, just in case you may have to delete the object due to some failure in the instantiation process. Cocos2d-x has a series of macros for object deletion and release, like the `CC_SAFE_DELETE` macro used previously.

You then initialize the super through one of its available methods. In Cocos2d-x, these `init` methods return a `boolean` value for success. You may now begin filling the `Player` object with some data.

If successful, then initialize your object with its proper data if not done in the previous step, and return it as an `autorelease` object.

So in your code the object would be instantiated as follows:

[PRE11]

Even if the `player` variable were a member of the class (say, `m_player`), you wouldn't have to retain it to keep it in scope. By adding the object to some Cocos2d-x list or cache, the object is automatically retained. So you may continue to address that memory through its pointer:

[PRE12]

## The C++ and Cocos2d-x style

In this option, you would instantiate the previous `Player` object as follows:

[PRE13]

`Player` could do without a static method in this case and the `player` pointer will not access the same memory in future as it's set to be autoreleased (so it would not stick around for long). In this case, however, the memory would not leak. It would still be retained by a Cocos2d-x list (the `addChild` command takes care of that). You can still access that memory by going over the children list added to `this`.

If you needed the pointer to be a member property you could use `retain()` instead of `autorelease()`:

[PRE14]

Then sometime later, you would have to release it; otherwise, it will leak:

[PRE15]

Hardcore C++ developers may choose to forget all about the `autorelease` pool and simply use `new` and `delete`:

[PRE16]

This will not work. You have to use `autorelease`, `retain`, or leave the previous code without the `delete` command and hope there won't be any leak.

C++ developers must keep in mind that `Ref` is managed by the framework. This means that objects are being internally added to caches and the `autorelease` pool even though you may not want this to happen. When you create that `Player` sprite, for instance, the `player.png` file you used will be added to the texture cache, or the sprite frame cache. When you add the sprite to a layer, the sprite will be added to a list of all children of that layer, and this list will be managed by the framework. My advice is, relax and let the framework work for you.

Non-C++ developers should keep in mind that any class not derived from `Ref` should be managed the usual way, that is, if *you* are creating a new object you must delete it at some point:

[PRE17]

# What you get with Ref

With `Ref` you get managed objects. This means that `Ref` derived objects will have a reference count property, which will be used to determine whether an object should be deleted from memory or not. The reference count is updated every time an object is added or removed from a Cocos2d-x collection object.

For instance, Cocos2d-x comes with a `Vector` collection object that extends the functionality of the C++ standard library vector (`std::vector`) by increasing and decreasing the reference count when objects are added and removed from it. For that reason, it can only store `Ref` derived objects.

Once again, every `Ref` derived class can be managed the way things used to be managed in Objective-C before ARC- with `retain` counts and `autorelease` pools.

C++, however, comes packed with its own wonderful dynamic list classes, similar to the ones you would find in Java and C#. But for `Ref` derived objects, you would probably be best served by Cocos2d-x managed lists, or else remember to retain and release each object when applicable. If you create a class which does not extend `Ref` and you need to store instances of this class in a list container, then choose the standard library ones.

In the examples that follow in this book I will code primarily from within the framework, so you will get to see plenty of examples of `cocos2d::Vector` being used, for instance, but I will also use a `std::vector` instance or two in some of the games.

# Summary

Hopefully, non-C++ developers have now learned that there is nothing to be feared from the language, and hardcore C++ developers have not scoffed too much at the notion of a root class and its retains and autoreleases.

All the stuff that root classes have brought to languages such as Java and Objective-C will forever be a moot point. The creepy, underlying operations that go on behind your back with root objects cannot be shut down or controlled. They are not optional, and this forceful nature of root objects has bothered C++ developers ever since notions such as garbage collectors first surfaced.

Having said that, memory management of `Ref` objects is extremely helpful and I hope even the most distrustful developers will soon learn to be thankful for it.

Furthermore, Cocos2d-x is awesome. So let's create a game already!