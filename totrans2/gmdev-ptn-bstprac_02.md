# One Instance to Rule Them All - Singletons

Now that we've learned what a design pattern is, as well as why we'd want to use them, let's first talk about a design pattern that most people learn, the Singleton pattern.

The Singleton pattern is probably the most well-known pattern and it is also the one out there that is most often misused. It definitely has the most controversy surrounding it, so when discussing this pattern it is as important (or even more important) to know when not to apply it.

# Chapter overview

In this chapter, we will explain about the pattern and many arguments for and against it. We will describe how and why core systems within the Mach5 engine such as the Graphics Engine and Object Manager are utilized as Singletons. Finally, we will explain a number of different ways to implement this in C++, along with the pros and cons of each choice.

# Your objective

This chapter will be split into a number of topics. It will contain a simple step-by-step process from beginning to end. Here is the outline of our tasks:

*   An overview of class access specifiers
*   Pros and cons of global access
*   Understanding the `static` keyword
*   What is a Singleton?
*   Learning about templates
*   Templatizing Singletons
*   The advantages and disadvantages of only one instance
*   The Singleton in action: the `Application` class
*   Design decisions

# An overview on class access specifiers

When using an object-oriented programming language, one of the most important features included is the ability to hide data, preventing classes from accessing properties and functions of another class type by default. By using access specifiers such as `public`, `private`, and `protected`, we can dictate specifically how the data and/or functions can be accessed from other classes:

[PRE0]

A class can have unlimited variables or functions that are `public`, `private`, or `protected` and can even control access to entire sections of the class:

[PRE1]

When you place a labelled section with an access modifier's name and a `:` next to it, until there is another section label, all of the parts of the class that are listed will use that specific one.

When we use the `public` access modifier, we are saying that this variable or function can be used or accessed from anywhere within our program, even outside of the class we've created. Declaring a variable outside of a function or class, or marking a variable as `public` and `static`, is often referred to as being a global variable. We will be talking about global variables in the next section, but for right now, let's go over the other access specifiers as well.

When `private` is used, we are restricting the usage of our variable or function to being allowed only inside of the class, or from `friend` functions. By default, all of the variables and functions in a class are `private`.

For more information on friend functions, check out [http://en.cppreference.com/w/cpp/language/friend](http://en.cppreference.com/w/cpp/language/friend).

The third type, `protected`, is the same as a `private` type except that it can still be accessed by child (or derived) classes. This can be quite useful when using inheritance so you can still access those variables and/or functions.

# The static keyword

Another thing that is important to know before diving into the Singleton pattern is what the `static` keyword means, as it's something that we will be using the functionality of when building this pattern. When we use the `static` keyword, there are three main contexts that it'll be used in:

*   Inside a function
*   Inside a class definition
*   In front of a global variable in a program with multiple files

# Static keyword inside a function

The first one, being used inside of a function, basically means that once the variable has been initialized, it will stay in the computer's memory until the end of the program, keeping the value that it has through multiple runs of the function. A simple example would be something like this:

[PRE2]

Now if we were to call this, it would look something like the following:

[PRE3]

And when we call it, the following would be displayed:

![](img/00019.jpeg)

As you can see, the value continues to exist, and we can access and/or modify its contents as we see fit in the function. This could be used for a number of things, such as maybe needing to know what happened the last time that you called this function, or to store any kind of data between any calls. It's also worth noting that static variables are shared by all instances of the class, and due to that, if we had two variables of type `StaticExamples`, they would both display the same `enemyCount`. We will utilize the fact that, if an object is created this way, it will always be available later on in this chapter.

# Static keyword in class definitions

The second way is by having a variable or function in a class being defined as `static`. Normally, when you create an instance of a class, the compiler has to set aside additional memory for each variable that is contained inside of the class in consecutive blocks of memory. When we declare something as `static`, instead of creating a new variable to hold data, a single variable is shared by all of the instances of the class. In addition, since it's shared by all of the copies, you don't need to have an instance of the class to call it. Take a look at the following bolded code to create our variable:

[PRE4]

Now, in the preceding code we define a variable and a function, but this isn't all the prep work we need to do. When creating a static variable, you cannot initialize it from within the class, and instead need to do it in a `.cpp` file instead of the `.h` file we could use for the class definition. You'll get errors if you do not initialize it, so it's a good idea to do that. In our case, it'd look like the following:

[PRE5]

Note that, when we initialize, we also need to include the type, but we use the `ClassName::variableName` template similar to how you define functions in `.cpp` files. Now that everything's set up, let's see how we can access them inside our normal code:

[PRE6]

Note that instead of accessing it via creating a variable, we can instead just use the class name followed by the scope operator (`::`) and then select which static variable or function we'd like to use. When we run it, it'll look like this:

![](img/00020.jpeg)

As you can see, it works perfectly!

# Static as a file global variable

As you may be aware, C++ is a programming language closely related to the C programming language. C++ was designed to have most of the same functionality that C had and then added more things to it. C was not object-oriented, and so, when it created the `static` keyword, it was used to indicate that source code in other files that are part of your project cannot access the variable, and that only code inside of your file can use it. This was designed to create class-like behavior in C. Since we have classes in C++ we don't typically use it, but I felt I should mention it for completeness.

# Pros and cons of global variables

To reiterate, a global variable is a variable that is declared outside of a function or class. Doing this makes our variable accessible in every function, hence us calling it global. When being taught programming in school, we were often told that global variables are a bad thing or at least, that modifying global variables in a function is considered to be poor programming practice.

There are numerous reasons why using global variables is a bad idea:

*   Source code is the easiest to understand when the scope of the elements used is limited. Adding in global variables that can be read or modified anywhere in the program makes it much harder to keep track of where things are being done, as well as making it harder to comprehend when bringing on new developers.
*   Since a global variable can be modified anywhere, we lose any control over being able to confirm that the data contained in the variable is valid. For instance, you may only want to support up to a certain number, but as a global variable this is impossible to stop. Generally, we advise using `getter`/`setter` functions instead for this reason.
*   Using global variables tightens how coupled our programs are, making it difficult to reuse aspects of our projects as we need to grab from a lot of different places to make things work. Grouping things that are connected to each other tends to improve projects.
*   When working with the linker, if your global variable names are common, you'll often have issues when compiling your project. Thankfully, you'll get an error and have to fix the issue in this case. Unfortunately, you may also have an issue where you are trying to use a locally scoped variable in a project but  end up selecting the global version due to mistyping the name or relying too heavily on intelligence and selecting the first thing you see, which I see students doing on multiple occasions.
*   As the size of projects grow, it becomes much harder to do maintenance and/or make changes to/on global variables, as you may need to modify many parts of your code to have it adjust correctly.

This isn't to say that global access is entirely bad. There are some reasons why one would consider using it in their projects:

*   Not knowing what a local variable is
*   Not understanding how to create classes
*   Wanting to save keystrokes
*   Not wanting to pass around variables all the time to functions
*   Not knowing where to declare a variable, so making it global means anyone can get it
*   To simplify our project for components that need to be accessible anywhere within the project

Aside from the last point, those issues are really bad reasons for wanting to use global variables, as they may save you some time up front, but as your projects get larger and larger it'll be a lot more difficult to read your code. In addition, once you make something global it's going to be a lot more difficult to convert it to not be global down the road. Think that, instead of using global variables, you could instead pass parameters to different functions as needed, making it easier to understand what each function does and what it needs to work with to facilitate its functionality.

That's not to say that there isn't any time when using a global variable is a reasonable or even a good idea. When global variables represent components that truly need to be available throughout your project, the use of global variables simplifies the code of your project, which is similar to what we are aiming to accomplish.

*Norm Matloff* also has an article explaining times that he feels like global variables are necessary when writing code. If you want to hear an alternative take, check out [http://heather.cs.ucdavis.edu/~matloff/globals.html](http://heather.cs.ucdavis.edu/~matloff/globals.html).

Basically, always limit your variables to the minimal scope needed for the project and not any more. This especially comes to mind when you only ever need one of something, but plan to use that one object with many different things. That's the general idea of the Singleton design pattern and is the reason why it's important that we understand the general usage before moving onwards.

# What is a Singleton?

The Singleton pattern in a nutshell is where you have a class that you can access anywhere within your project, due to the fact that only one object (instance) of that class is created (instantiated). The pattern provides a way for programmers to give access to a class's information globally by creating a single instance of an object in your game.

Whereas there are quite a few issues with using global variables, you can think of a Singleton as an *improved* global variable due to the fact that you cannot create more than one. With this in mind, the Singleton pattern is an attractive choice for classes that only have a unique instance in your game project, such as your graphics pipeline and input libraries, as having more than one of these in your projects doesn't make sense.

This single object uses a static variable and static functions to be able to access the object without having to pass it through all of our code.

In the Mach5 engine, Singletons are used for the application's, input, graphics, and physics engines. They are also used for the resource manager, object manager, and the game state manager. We will be taking a much closer look at one of the more foundational ones in the engine, the `Application` class, later on in this chapter. But before we get to it, let's dive into how we can actually create one of our very own.

There are multiple ways to implement the Singleton pattern or to get Singleton-like behavior. We'll go over some of the commonly seen versions and their pros and cons before moving to our final version, which is how the Mach5 engine uses it.

One very common way of implementing the functionality of the Singleton pattern would look something like the following:

![](img/00021.gif)

Through code, it will look a little something like this:

[PRE7]

In this class, we have a function called `GetInstance` and a single property called `instance`. Note that we are using pointers in this instance, and only allocating memory to create our Singleton if we are actually using it. The instance property represents the one and only version of our class, hence it being made `static.` As it is private though, there is no way for others to access its data unless we give them access to it. In order to give this access, we created the `GetInstance` function. This function will first check whether instance exists and if it doesn't yet, it will dynamically allocate the memory to create one, set instance to it, and then return the object.

This will only work if instance is properly set to `0` or `nullptr` when initialized, which thankfully is the default behavior of static pointers in C++.

# Keeping the single in Singleton

As we've mentioned previously, one of the most important parts of the Singleton pattern is the fact that there is only one of those objects. That causes some issues with the original code that we've written, namely that with some simple usage of C++ it is quite easy to have more than one of these classes created by other programmers on your team. First and most obviously, they can just create a `Singleton` variable (a variable of type `Singleton`) like the following:

[PRE8]

In addition, as a higher-level programming language, C++ will try to do some things automatically for you when creating classes to eliminate some of the busy work that would be involved otherwise. One of these things is automatically creating some functionality between classes to enable you to create or copy objects of a custom class that we refer to as a constructor and copy constructor. In our case, you can also create a copy of your current object in the following way:

[PRE9]

The compiler will also create a default destructor and an assignment operator, moving the data from one object to the other.

Thankfully, that's a simple enough thing to fix. If we create these functions ourselves (declaring an explicit version), C++ notes that we want to do something special, so it will not create the defaults. So to fix our problem, we will just need to add an assignment operator and some constructors that are private, which you can see in the bold code that we've changed:

[PRE10]

If you are using C++ 11 or above, it is also possible for us to instead mark the functions we don't want to use as deleted, which would look like this:
`Singleton() = delete;`
`~Singleton() = delete;`
`Singleton(const Singleton &) = delete;`
`Singleton& operator=(const Singleton&) = delete;`
For more information on the delete keyword, check out [http://www.stroustrup.com/C++11FAQ.html#default](http://www.stroustrup.com/C++11FAQ.html#default).

Another thing that may possibly be an issue is that instance is a pointer. This is because, as a pointer, our users have the ability to call delete on it and we want to make sure that the object will always be available for our users to access. To minimize this issue, we could change our pointer to be a reference, instead, by changing the function to the following (note the return type and that we use `*instance` now on the last line):

[PRE11]

Programmers are used to working with references as aliases for objects that exist somewhere else in our project. People would be surprised if they ever saw something like:

[PRE12]

While technically doable, programmers won't expect to ever use delete on the address of a reference. The nice thing about using references is that, when you need them in code, you know that they exist because they're managed somewhere else in the code--and you don't need to worry about how they are used.

# Deleting our object correctly

People also are used to looking for memory leaks with pointers and not references, so that perhaps leaves us with an issue as, in our current code, we allocate memory but don't actually delete it.

Now, technically, we haven't created a memory leak. Memory leaks appear when you allocate data and lose all of your references to it. Also, modern operating systems take care of deallocating a process's memory when our project is quit.

That's not to say that it's a good thing though. Depending on what information the Singleton class uses, we could have references to things that no longer exist at some point.

To have our object delete itself correctly, we need to destroy the Singleton when our game shuts down. The only issue is we need to make sure that we do it only when we are sure no one will be using the Singleton afterwards.

However, as we want to talk about best practices, it's much better for us to actually solve this issue by removing resource leaks whenever we see them. A solution to this very problem was created by *Scott Meyers* in his book *More Effective C++*, which uses some of the features of the compiler, namely that a static variable located in a function will exist throughout our program's running time. For instance, let's take the following function:

[PRE13]

The `numberOfEnemies` variable is created and has been initialized before any code in the project has been executed, most likely when the game was being loaded. Then, once `SpawnEnemy` is called for the first time, it will have already been set to `0` (or `nullptr`). Conveniently, as the object is not allocated dynamically, the compiler will also create code so that, when the game exists, it will call the deconstructor for our object automatically.

With that in mind, we can modify our Singleton class to the following:

[PRE14]

Specifically note the changes we've made to the `GetInstance` function and the removal of our class instance variable. This method provides the simplest way to destroy the `Singleton` class automatically and it works fine for most purposes.

# Learning about templates

Another technique to add to your toolbox of programming concepts that we will use in the next section is the idea of templates. **Templates** are a way for you to be able to create generic classes that can be extended to have the same functionality for different datatypes. It's another form of abstraction, letting you define a base set of behavior for a class without knowing what type of data will be used on it. If you've used the STL before, you've already been using templates, perhaps without knowing it. That's why the list class can contain any kind of object.

Here's an example of a simple templated class:

[PRE15]

In this case, we created our `TemplateExample` class and it has three functions. The constructor and deconstructor look normal, but then I have this `TemplateFunction` function which takes in an object of type `T`, and returns an object of type `T`. This `T` comes from the first line of our example code with the template `<class T>` section of our code. Anywhere that there is a `T` it will be replaced with whatever class we want to use this template with.

Now, unlike regular functions, we have to define templated functions within our `.h` file, so that, when we need to create an object using this template, it will know what the functions will do. In addition to this, the syntax is also a bit different:

[PRE16]

In this example, I'm just printing out text to display when a certain functionality is called, but I also want to point out the usage of `std::cout` and that using it will require you to add `#include <iostream>` to the top of your file.

We are using the standard library's `cout` function in this instance, instead of the `printf` that we have been using, because `cout` allows us to feed in `obj`--no matter what its type is--to display something, which isn't possible with `printf` by default.

Once that's finished, we can go ahead and use this inside of our project:

[PRE17]

As you can see, this will create three different kinds of `TemplateExample` class objects using different types. When we call the `TemplatedFunction` function, it will print out exactly the way we were hoping:

![](img/00022.jpeg)

Later on, when we learn about abstract types, we can use templates with them to handle any kind of data. In our case right now, we are going to use this functionality to allow us to make as many Singletons as we'd like!

# Templatizing Singletons

Now, assuming we get our Singleton working just the way that we want it to, you may wish to create more Singletons in the future. You could create them all from scratch, but a better thing to do is instead create a consistent approach, creating templates and inheritance to create a single implementation that you can use for any class. At the same time, we can also learn about an alternative way of creating a `Singleton` class, which will look something like the following:

[PRE18]

You'll notice that most of the differences have to do with the class itself. The very first line in our code above uses the `template` keyword which tells the compiler that we are creating a template, and `typename T` tells the compiler that, when we create a new object using this, the type `T` will be replaced with whatever the class we want it to be based on is.

I also want to point out the use of a static cast to convert our Singleton pointer to a `T`. `static_cast` is used in code generally when you want to reverse an implicit conversion. It's important to note that `static_cast` performs no runtime checks for if it's correct or not. This should be used if you know that you refer to an object of a specific type, and thus a check would be unnecessary. In our case, it is safe because we will be casting from a Singleton object to the type that we've derived from it (`T`).

Of course, it may be useful to see an example of this being used, so let's create an example of a class that we could use as a Singleton, perhaps something to manage the high scores for our game:

[PRE19]

Notice here that, when we declare our `HighScoreManager` class, we say that it's derived from the `Singleton` class and, in turn, we pass the `HighScoreManager` class to the `Singleton` template. This pattern is known as the curiously recurring template pattern.

For more information on the curiously recurring template pattern, check out [https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern).

After defining the class, let's go ahead and add in an example implementation for the function we've created for this class:

[PRE20]

By using the templatized version of our class, we don't need to create the same materials as in the preceding class. We can just focus on the stuff that is particular to what this class needs to do. In this case, it's checking our current high score, and setting it to whatever we pass in if we happen to beat it.

Of course, it's great to see our code in action, and in this case I used the `SplashStage` class, which is located in the Mach5 `EngineTest` project, under `SpaceShooter/Stages/SplashStage.cpp`. To do so, I added the following bolded lines to the `Init` function:

[PRE21]

In this case, our instance has been created by us creating a new `HighScoreManager`. If that is not done, then our project could potentially crash when calling `GetInstance`, so it's very important to call it. Then call our `CheckHighScore` functions a number of times to verify that the functionality works correctly. Then, in the `Shutdown` function, add the following bolded line to make sure the Singleton is removed correctly:

[PRE22]

With all of that gone, go ahead, save the file, and run the game. The output will be as follows:

![](img/00023.jpeg)

As you can see, our code works correctly!

Note that this has the same disadvantages we discussed with our initial version of the script, with the fact that we have to manually create the object and remove it; but it takes away a lot of the busy work when creating a number of Singletons in your project. If you're going to be creating a number of them in your project, this could be a good method to look into.

# Advantages/disadvantages of using only one instance

There is the possibility that as you continue your project, something that looks at the time to be a thing that you'll only need one of will suddenly turn into something you need more of down the road. In games, one of the easiest examples would be that of a player. When starting the game, you may think you're only going to have one player, but maybe later you decide to add co-op. Depending on what you did before, that can be a small or huge change to the project.

Finally, one of the more common mistakes we see once programmers learn about Singletons, is to create managers for everything, and then make the managers all Singletons.

# The Singleton in action - the Application class

The Singleton pattern achieves its ability to be accessible anywhere easily by having a special function that we use to get the `Singleton` object. When this function is called, we will check whether that object has been created yet. If it has, then we will simple return a reference to the object. If not, we will create it, and then return a reference to the newly created object.

Now, in addition to having this way to access it, we also want to block off our user from being able to create them, so we will need to define our class constructors to be private.

Now that we have an understanding of some implementations of the Singleton, we have one other version, which is what we actually used within the Mach5 engine.

In Mach5, the only Singletons that are included are aspects of the engine code. The engine code is designed to work with any game, meaning there is nothing gameplay-specific about it, which means that it doesn't need to have instances since they're just instructions. Building the engine in this way makes it much easier in the future to bring this to other games, since it's been removed from anything that's game-specific.

In this case, let's open up the `M5App.h` file which is in the `EngineTest` project under `Core/Singletons/App` and take a look at the class itself:

[PRE23]

Now, the Mach5 engine follows the Singleton pattern. However, it is done in a different way from the others that we've looked at so far. You may notice in the class definition that every single function and variable that was created was made static.

This provides us with some unique benefits, namely that we don't need to worry about the user creating multiple versions of the class, because they'll only be restricted to using static properties and variables that are shared by everything. This means we don't need to worry about all of those fringe cases we mentioned in the previous examples that we've seen. This is possibly due to the fact that the Mach5 engine classes have no need to have child classes; there's no need for us to create a pointer or even call a `GetInstance` function.

You'll also notice the `Init`, `Update`, and `Shutdown` functions mentioned previously. We mentioned before that it was a disadvantage to manually have to create and destroy our `singleton` classes, but there are some distinct benefits to having this control. In the previous examples we had, the order in which classes were created was up to the compiler as we couldn't control the order. However, with our game engine it makes sense to create our Application (`M5App`) before we start up the graphics library (`M5Gfx`) and the only way we can make sure that happens is by telling our engine to do so, which you can look at if you open up the `Main.cpp` file and look at the `WinMain` function, which is what opens first when we create our project. I've gone ahead and bolded the uses of `M5App`:

[PRE24]

Afterwards, we can look at the `Init` function of `M5App` and see that it will initialize the other Singletons in our project:

[PRE25]

By having this control, our users have a much better idea as to the flow and order that things will be created. But, of course, with that great power comes great responsibility.

The Singleton pattern is used only for single-threaded applications. Should you be developing a multithreaded game, you'll want to use the Double-Checked Locking pattern instead, which was created by *Doug Schmidt* and *Tim Harrison*. If you're interested in learning more about it, check out [https://en.wikipedia.org/wiki/Double-checked_locking](https://en.wikipedia.org/wiki/Double-checked_locking).

# Summary

In this chapter, we have demystified a lot of programming concepts in a quick refresher. We also started learning about our first design pattern, the Singleton, which is intended to allow us to always have access to a class's functions and variables due to the fact that there will only ever be one of these objects.

We discussed some of the typical downfalls of using the Singleton pattern, such as the possibility that objects could have multiple copies of them in the future, even if this is unlikely.

We learned about three different kinds of method for creating Singletons, starting off with the *Singleton*, then extending it and templating parts of it to create the curiously reoccurring template pattern, and then we saw a final all-static version of getting the same effect with minimal hassle.

Each of these methods has their own pros and cons, and we hope that you use them effectively, where they are relevant. Now that we've touched on the design pattern everyone is familiar with, we can move towards our next challenge: learning about how to deal with logic that is specific to each of our individual games.