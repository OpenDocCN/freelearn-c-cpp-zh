# Controlling the UI via the Command Pattern

In the last chapter, we dived deeply into the bits and bytes of computer memory so we could make our components more efficient and easier to debug. Understanding these details can be the difference between a game running at 60 frames per second or 30 frames per second. Knowing how to control memory usage is an important aspect of becoming a great programmer. It is also one of the hardest things about programming. In this chapter, we will take a break from low-level programming and look at something high level.

The user interface, or UI, is just as important as memory management or stage switching. You could even argue that it is more important because the player doesn't care about the low-level details. They just want a game that is fun. However, it doesn't matter how fun the gameplay is; if the UI is difficult to navigate or control, the fun level drops fast.

Can you remember a time when you played a game with terrible controls? Did you keep playing the game? It is interesting because, for something so important, it often has the chance of being left until the end of a project. Even in this book, we had to wait until the eighth chapter to cover it. However, great games make the design of the UI and the user experience a top priority.

There are a lot of great books on how to design user interfaces and craft the user experience. This isn't one of those books. Instead, we will look at how the code behind the UI can be implemented in a flexible way that works with the rest of our engine. The first step to making a great UI is designing the code that will make buttons and other input easy to create and change.

We will start by looking at a very simple but powerful pattern that allows us to decouple our function calls from the objects that want to call them. While we are discussing the pattern, we will look at some of the syntactically ugly and confusing ways C++ allows us to treat functions as if they were objects. We will also see how the Mach5 Engine uses this pattern to create clickable UI buttons.

# Chapter overview

This chapter is all about separating the user interface and input from the actions they perform. We will learn about the Command pattern and how it can help us decouple our code. We will do this by first understanding the problem then looking at how this could be solved in a C style fashion. Then after looking at the Command pattern in depth, we will see how it is implemented in the Mach5 Engine.

# Your objectives

The following lists the things to be accomplished in this chapter:

*   Learn the naive approach to handling input and why it should be avoided
*   Implement the Command pattern using function pointers and the class method pointer
*   Learn how the Mach5 Engine uses the Command pattern
*   Implement UI buttons within the Mach5 Engine

# How can we control actions through buttons?

In [Chapter 3](part0064.html#1T1400-04600e4b10ea45a2839ef4fc3675aeb7), *Improving on the Decorator Pattern with the Component Object Model*, we implemented game objects. Now that we have them, it seems trivial to create buttons on the screen. In fact, in genres such as real-time strategy, there is no difference between clickable buttons and game objects. The player can click on any unit or building and give them orders.

At first thought, our buttons could just be game objects. They both have a position, scale, and texture, and that texture will be drawn to the screen. Depending on the game, you might draw your buttons using orthographic projection while the objects will be drawn using perspective projection. However, the differences go deeper than that.

At its core, a button has an action that needs to be performed when it is clicked or selected. This behavior is usually simple; it doesn't require creating an entire state machine class. It does however, require a little thought so we don't end up hardcoding button functionality all over our high-level modules or repeating similar code in many different places.

In [Chapter 5](part0096.html#2RHM00-04600e4b10ea45a2839ef4fc3675aeb7), *Decoupling Code via the Factory Method Pattern*, we saw an extremely naive way to handle a button click on a menu screen. Recall that this code was written by one of the authors early in their programming career:

[PRE0]

There are a lot of problems with this code:

*   First, the rectangular click region is hardcoded to the aspect ratio in full screen mode. If we were to switch from widescreen 16:9 aspect ratio to standard 4:3 aspect ratio or even if we changed from full screen to windowed mode, this code wouldn't work correctly.
*   Second, the click region is based on the screen and not the button itself. If the button position or size were to change, this code wouldn't work correctly.
*   Third, this menu screen is coupled to the Windows `GetSystemMetrics` function instead of an encapsulated platform code class like the `M5App` class. This means if we want to run on a different operating system or platform, this menu and possibly all menus need to be modified.
*   Finally, the state (stage in Mach5) switching action is hardcoded to the menu. If we decide to perform a different action, we need to modify the menu. If this action can be performed by both a button click and keyboard input, we need to update and maintain both sections of code.

As you can see, this isn't an ideal way to handle buttons in a game. This is basically the worst way you can implement buttons. This code is very likely to break if anything changes. It would be nice if the author could say this code was only written as a demonstration of what not to do. Unfortunately, a book like the one you are reading didn't exist at the time, so he had to learn the hard way.

# Callback functions

A better way to deal with these button actions is with callback functions. Callback functions in C/C++ are implemented using pointers to functions. They allow you to pass functions around as if they were variables. This means functions can be passed to other functions, returned from functions, or even stored in a variable and called later. This allows us to decouple a specific function from the module that will call it. It is a C style way to change which function will be called at runtime.

Just as pointers to `int` can only point at `int`, and pointers to `float` can only point at `float`, a pointer to a function can only point at a function with the same signature. An example would be the function:

[PRE1]

This function takes a single `int` as a parameter and returns an `int`. This return value and parameter list are the function's signature. So, a pointer to this function would be:

[PRE2]

We haven't given the function pointer a name, so it should look like this:

[PRE3]

Note that the parentheses around the variable name `pFunc` are required, otherwise the compiler will think this is a prototype of a function that returns a pointer to an `int`.

We can now create a pointer to a specific function and call that function through the variable:

[PRE4]

The output for the preceding code is as follows:

![](img/00052.jpeg)

Figure 8 1 - Function pointer output

Notice that we didn't need to take the address of the `Square` function (although that syntax is allowed); this is because in C and C++ the name of the function is already a pointer to that function. That is why we can call `pFunc` without needing to dereference it. Unfortunately, everything about function pointers is weird until you get used to them. You must work at remembering the syntax since it doesn't work the same as pointers to variables.

By looking at a larger example, we can get familiar with this syntax. Let's write a program with three different ways to fill an array with values and a way to print the array:

[PRE5]

Our goal with this program is to write a function that can fill an array with any fill function, including one that hasn't been written yet. Since we have a common function signature, we can create a function called `FillAndPrint` that will take a pointer to any function with a matching signature as a parameter. This will allow `FillAndPrint` to be decoupled from a specific fill function and allow it to be used with functions that do not exist yet. The prototype for `FillAndPrint` will look like this:

[PRE6]

This is incredibly ugly and difficult to read. So, let's use a `typedef` to clean up the code a little. Remember that a `typedef` allows us to give a different, hopefully more readable, name to our type:

[PRE7]

In main, the user of this code can pick which fill function they want to use or even write a completely new one (if the signature is the same), without changing `FillAndPrint`:

[PRE8]

Here is what this code would output to the command line:

![](img/00053.jpeg)

Figure 8 2 - Using FillAndPrint in different ways

We could even allow the user to pick the fill at runtime if we included a `helper` function to select and return the correct fill function:

[PRE9]

This is a very simple example, but you can already see how using function pointers allows us to write flexible code. `FillAndPrint` is completely decoupled from any specific function call. Unfortunately, you can also see two flaws with this system. The functions must have the exact same signature, and the parameters of the function must be passed to the user of the function pointer.

These two problems make function pointers interesting and powerful, but not the best solution for in-game buttons that need to support a wide variety of actions with varying parameter lists. Additionally, we might want to support actions that use C++ member functions. So far, all the examples that we have seen were C style global functions. We will solve these problems in a moment, but first we should look at how we will trigger our button click.

# Repeated code in the component

We have the problem of wanting to decouple a specific function call from the place that calls it. It would be nice to be able to create a button component that could save a function pointer or something like it, and call it when the component is clicked.

One solution could be to create a new component for every action we want to execute. For example, we might want to create a component that will change the stage to the main menu. We could create a component class that knows how to perform that exact action:

[PRE10]

The preceding case is a very simple example because it is only calling a static function with the parameter hardcoded, but the function pointer as well as the function parameters could easily be passed in to the constructor of this component. In fact, we could pass any object to the constructor and hardcode a specific method call in the update function. For example, we could pass an `M5Object` to a component such as the one above. The button click might change the texture of the object. For example:

[PRE11]

Unfortunately, there is a big problem with code like this; the action is completely coupled to the button click. This is bad for two reasons. First, we can't use this action for a keyboard or controller press unless we add additional keys to our UI button click component. Second, what happens when we have a list of actions that we want to perform? For example, synchronizing the movement of multiple UI objects, or scripting an in-game cut scene. Since the actions require the mouse to be pressed on the object, our action is very limited.

The other reason this approach is bad is because we must repeat the exact same mouse click test code in every button component that we create. What we would like to do is decouple the action from the button click component. We would need to create a separate UI button component and an action class. By doing that, we would factor out the part of the code that repeats, and we would gain the ability to use the actions on their own.

# The Command pattern explained

The Command pattern is exactly the pattern that solves our problem. The purpose of the Command pattern is to decouple the requester of an action from the object that performs the action. That is exactly the problem we have. Our requester is the button, and it needs to be decoupled from whatever specific function call will be made. The Command pattern takes our concept of a function pointer and wraps it into a class with a simple interface for performing the function call. However, this pattern allows us more flexibility. We will easily be able to encapsulate function pointers with multiple parameters, as well as with C++ object and member functions. Let's start off easy with just two simple functions that have the same parameter count and return type:

[PRE12]

The Command pattern encapsulates a request into an object, and it gives a common interface to perform that request. In our example, we will call our interface method `Execute()`, but it could be called anything. Let's look at the `Command` abstract class:

[PRE13]

As you can see, the Command pattern interface is very simple--it is just a single method. As usual, we mark the method as pure virtual so the base class can't be instantiated. Additionally, we create an empty virtual destructor so the correct derived class destructor will be called when needed. As I said, the name of the method isn't important. I have seen examples such as `Do`, `DoAction`, `Perform`, and so on. Here we call it `Execute` because that was the name in the original book written by the Gang of Four.

Right from the start, we gain a benefit over function pointers by using this pattern. For every derived class we are writing the `Execute` method, which means we can directly hardcode any function and any parameters in that `Execute` function. Recall that when using function pointers, we needed to pass in parameters at the time of the call:

[PRE14]

In this example, we are just hardcoding the function call and the function parameter in place. This may not seem very useful now for such a simple function, but it could be used in-game. As we will see later, the Mach5 Engine has a command to quit the game. The command directly calls `StageManager::Quit()`.

In most cases, we probably don't want to hardcode the function and parameters. This is where the power of this pattern shows. In this next example, we can use the fact that both functions have the same signature. That means we can create a function pointer, and pass the function, and parameters to the command. The benefit here is that because the command is an object, it has a constructor. So, we can construct an object with an action and the parameters that will be used by that action:

[PRE15]

There are a few interesting things going on here. The first is that this command can call any function that returns an `int` and takes one `int` as a parameter. That means it can work for Square and Cube, but also any other functions that we come up with later. The next interesting thing is that we can set the action and parameter in the constructor; this allows us to save parameters within the class and use them later. We could not do this by using function pointers alone. Finally, you may have noticed that we are passing in a pointer to an `int`, instead of just an `int`. This demonstrates how we can save the return value of a function call, and also allows us to think about these commands in a more flexible way.

Commands are not just for quitting the game or changing the stage. We could have a command that changes the position of a game object when executed, or perhaps swaps the position of the player and an enemy based on some user input or a button click. By using commands, we can control everything about the game via the UI. That sounds a lot like a level editor.

Now that we have seen two types of commands, let's look at how the client would use them. We will start out with a simple main function. We will be constructing the command in the same function that calls it, but these could be set via a function call instead. The important thing is that at the point where the client calls Execute, they don't need to know which function is being called, or what parameters (if any) are needed:

[PRE16]

The output for the preceding code is as follows:

[PRE17]

As we can see, the client could call different functions using the same interface, and does not need to care about function parameters. For such a simple pattern, the Command pattern is amazing. And it gets even better.

# Two parameters and beyond

We saw that one limitation of using function pointers was that the signatures must be the same. They must have the same return type, as well as the same parameter types and count. We can already see that this isn't true with the Command pattern. The client doesn't need to know or care about the specific signature at call time since every command shares the common Execute interface. As an example, let's look at a function with more than one parameter and create a command for that type. Here is the function:

[PRE18]

As we mentioned before, the complexity of the function isn't important. For now, let's focus on functions that take more than one parameter, as in the case of this Add function. To make our code easier to read, let's create a `typedef` for this signature too:

[PRE19]

Finally, let's create a `Command` for all functions that match this signature:

[PRE20]

The `main` function is now updated to the following. Here we are only showing the parts of the code that changed:

[PRE21]

The output for the preceding code is as follows:

[PRE22]

As you can see, we can easily create a new command for every function pointer signature we need. When the client calls the method, they don't need to know how many parameters are used. Unfortunately, even though our commands can take multiple arguments, those arguments are stuck using only the `int`. If we wanted them to use the float, we would need to make new commands or use the create a template command.

In a real-world scenario, you could get away with creating the commands as you need them, and only creating them for the types you need. Another option, and one that is more common, is to have commands call C++ class methods, since the method has the option to use class variables instead of passed in parameters.

# Pointers to member functions

So far, we have seen how we can use the Command pattern with function pointers and allow the client to call our functions without caring about the parameter types or counts. This is incredibly useful. But what about using commands with C++ objects? While we can get commands to work with objects, we need to think about the problem a little first.

The most basic way to call member functions is to simply hardcode them in the `Execute` method. For example, we could pass in an object to a command constructor and always call a very specific function. In the example, `m_gameObject` is a pointer to an object that was passed to the constructor. However, `Draw` is the hardcoded method that we always call. This is the same as hardcoding the function in `Square5Command`:

[PRE23]

Since `m_gameObject` is a variable, the object that will call `Draw` can change, but we are still always calling `Draw`. In this case, we don't have the option to call something else. This is still useful, but we would like the ability to call any method on a class type. So, how do we get this ability? We need to learn about pointers to member functions.

Using pointers to member functions isn't that different from pointers to non-member functions. However, the syntax is a little stranger than you might expect. Recall that when calling a non-static class method, the first parameter is always implicit to the pointer:

[PRE24]

The `this` pointer is what allows the class method to know which instance of the class it needs to modify. The compiler automatically passes it in as the first parameter to all non-static member functions, and the address of the `this` pointer is used as an offset for all member variables:

[PRE25]

Even though it is implicitly passed in and is not part of the parameter list, we still have access to the `this` pointer in our code:

[PRE26]

It is important to understand this because normal functions and member functions are not the same. Class members are part of the class scope and they have an implicit parameter. So, we can't save pointers to them like normal functions. The signature of a class method includes the class type, meaning we must use the scope resolution operator:

[PRE27]

Just having the correct pointer type is not enough. The class member access operators, known as the dot operator (`.`) and arrow operator ( `->` ), are not designed to work with arbitrary function pointers. They are designed to work with known data types or known function names as declared in the class. Since our function pointer isn't known until runtime, these operators won't work. We need different operators that will know how to work with member function pointers. These operators are the pointer to member operators, ( `.*` ) and ( `->*`).

Unfortunately, these operators have lower precedence than the function call operator. So, we need to add an extra set of parentheses around our object and our member function pointer:

[PRE28]

There is a lot more to pointers to members. This section here was just a short introduction. If you want more information, please go to [https://isocpp.org/wiki/faq/pointers-to-members](https://isocpp.org/wiki/faq/pointers-to-members).

# Pointer to member command

Now that we know how to use pointers to member functions, we can create commands that can take an object and a specific member function to call. Just like before, we will use a simple example. The example class isn't designed to do anything interesting, it is just used to demonstrate the concepts:

[PRE29]

Here is a simple class called `SomeObject`. It has a constructor that takes an `int` parameter and uses it to set the private member variable `m_x`. It also has two functions: one that will print the value to the screen and one that changes the value. For now, we are keeping things simple by giving both member functions the same signature and not taking any arguments. This allows us to create a `typedef` for this type of method. Remember that the class type is part of the function signature:

[PRE30]

This creates a type called `SomeObjectMember` that can easily be used as a function parameter, function return type, or even saved as a member to another class (of course, that is exactly what we will do next). Even if you feel very comfortable with the syntax of function pointers and pointer to member functions, it is still good practice to make these `typedefs`. They make the code more readable for everyone, as you will see in the next code example:

[PRE31]

Since the syntax of calling a member function pointer can be tricky to get right, it can be useful to use a `#define` macro. While most of the time, macros should be avoided, this is one of the few times they can help by making your code more readable:

[PRE32]

This changes our `Execute` function to this:

[PRE33]

All we have done is hide the ugliness away in a macro, but at least people will have a better understanding of what it is doing. It is important to note that this macro only works with object pointers because it uses the arrow star operator ( `->*` ).

Now, in main we can create commands to object members:

[PRE34]

The following is the class diagram of command hierarchy:

![](img/00054.jpeg)

Figure 8.3 - The command hierarchy

Even though this is just a simple demo, we can see the client code is the same whether they are calling a function pointer or a pointer to a member function, and regardless of parameter count. Unfortunately, we still need to create a `typedef` for every function and class type we need. However, C++ templates can help us here too. We can create a template command class that can call class methods of a specific signature (in our case, `void (Class::*)(void)`) that will work for all classes:

[PRE35]

As you can see in the `Execute` method, this is limited to only calling methods without arguments, but it could easily be modified to suit your game's needs.

# The benefits of the command pattern

If looking at all that crazy code makes your eyes glaze over, you are not alone. The complex syntax of function pointers and pointer to member functions calls are some of the most difficult parts of C++. For that reason, many people avoid them. However, they also miss out on the power offered by such features.

On the other hand, just because something is powerful, it doesn't mean it is always the right tool for the job. Simple is often better and, because of the many levels of indirection, code like we just saw has the chance to cause a lot of bugs. It will be up to you to decide if using these tools is right for your project. That being said, let's discuss some of the benefits of using the Command pattern so you can better decide when and where to use it.

# Treating a function call like an object

The biggest benefit of using the Command pattern is that we are encapsulating the function or method call and the parameters. This means that everything needed for the call can be passed to another function, returned from a function, or stored as a variable for later use. This is an extra level of indirection over only using function or method pointers, but it means the client doesn't need to worry about the details. They only need to decide when to execute the command.

This might not seem very useful since we need to know all the function arguments before we pass it to the client. However, this situation can happen more often than you might think. The fact that the client doesn't need to know the details of the function call means that systems such as the UI can be incredibly flexible, and possibly even read from a file.

In the above example, it is obvious that at the time of the call, the client doesn't know which command exists at a given array index. This is by design. What might not be so obvious, is that the array could have been populated using the return value from a function instead of hardcoded calls to a new operator (which we learned in [Chapter 5](part0096.html#2RHM00-04600e4b10ea45a2839ef4fc3675aeb7), *Decoupling Code via the Factory Method Pattern*, leads to inflexible code). This flexibility means that the function to be executed can be changed at runtime.

A perfect example of this is a context sensitive *action button* in a game. Since there is a limited number of buttons on a gamepad, it is often useful to have the action of button change depending on what the player is doing. This could mean one button is responsible for talking to an NPC, picking up an item, opening a door, or triggering a *quick time event* depending on the player's location and what they are doing.

Without the Command pattern, the logic involved in organizing, maintaining, and executing all the possible actions in a game would be incredibly complex. With the Command pattern, it is giving every actionable item a command, and making it available when the player is near.

# Physically decoupling the client and the function call

One aspect of good design is low coupling. We have talked about this a lot before, and it applies here as well. First, since the client is only dependent on the base `Command` class, it is easier to test. This is because both the client and the specific function calls or actions can be tested independently to ensure that they work. Furthermore, since these unit tests are testing smaller amounts of code, we can be more confident that all possible cases are tested. This also means that the client or the commands have a better chance to be reused because of the low coupling within this project.

Second, the client is less likely to break when changes to the code base occur. Since the client doesn't know which functions or methods are called, any changes to parameter counts or method names are local only to the commands that implement the changed methods. If more commands need to be added, those commands will automatically work with the existing client because they will use the `Command` class interface.

Finally, compile times can be reduced because the client needs to include fewer header files. Including fewer header files can lower the compile time since every time the header changes, every source file that includes it must be recompiled. Even the smallest change to a comment in a header file means that all the function calls from that header need to be rechecked for correct syntax at compile time and relinked at link time. Since our client doesn't know the details of the functions calls, there are no header files to include.

# Temporal decoupling

This type of decoupling isn't talked about much because it only applies to a few situations and, most of the time, this isn't what we want. Usually, when we call a function we want it to execute immediately. We have a specific algorithm in our code and the timing and order of that code is very important. This isn't true of all code. One situation is multithreaded code, in which multiple paths of code are executing simultaneously. Other situations are UI or context sensitive buttons, where the action to be executed is set up in advance instead of hardcoded in place. Let's look at some code as an example:

[PRE36]

In all four of the above situations, the functions and parameters are given. However, the command versions can be passed to other methods, called and/or recalled based on the need of the client.

# Undo and redo

Another major benefit of having the call details packaged together in a class is the ability to undo an operation. Every modern desktop application, as well as the best web applications being made these days, features the ability to undo the last action or actions. This should be a standard that you strive to follow when implementing a level editor for your game.

Implementing a single level of undo in an application can seem like a large task. The naive approach might be to save the entire state of the application, possibly to a file, and reload that state when we need to undo. Depending on the application, there might be a lot of data to save. This method doesn't scale well in applications that can have dozens or hundreds of levels of undo. As the user does more actions, you would need to make sure to delete the oldest state before saving the current one.

This simple approach is even more difficult when you also need to implement redo. Obviously, the text editors and tools that we use every day don't store hundreds of undo and redo files on the hard drive. There must be a better way.

Instead of saving the entire state of the program, you only need to save information about the action that happened, and what data was changed. Saving a function and the parameters to the function sounds a lot like the Command pattern. Let's look at a simple example of moving a game object from one place to another in a level editor. We could create a command like this:

[PRE37]

By adding the `Undo` method to the command interface and making sure to save the old data that will be modified in the `Execute` method, performing undo and redo becomes incredibly simple. First, we need to implement a command for every action that can be performed in our editor. Then, when the user interacts with the editor, instead of directly calling a function, they always call a command and add it to the end of our array of commands. Undoing and redoing is just a matter of calling the `Execute` or `Undo` method of the current array index.

It might seem like a lot of work to create all those commands, and it is. However, that work is replacing the work of hardcoding function calls when a user presses keys or clicks the mouse. In the end, you will build a better system that people will want to use.

# Easy UI with commands in Mach5

Now that we have seen what the Command pattern is, let's look at how it is used in the Mach5 Engine. You will be surprised that there isn't much code here. That is because using the Command pattern is easy once you understand the code behind it. In this section, we will look at both the component responsible for the mouse click and the commands that are used within the engine.

Let's have a look at the `M5Command` class:

[PRE38]

Here is the `M5Command` class used in the Mach5 Engine. As you can see, it looks almost identical to the `Command` class we used in the example. The only difference is that since we plan on using this within a component, it needs to have a virtual constructor. That way we can make a copy of it without knowing the true type.

The code for the `UIButtonComponent` class is as follows:

[PRE39]

As you can see, our UI button is a component. This means that any game object has the potential to be clicked. However, this class is specifically designed to work with objects that are in screen space, which is how the operating system gives us the mouse coordinates. The rest of the code here looks like you might expect. As part of the `UIButtonComponent` class, we have a private `M5Command`. Although this class is simple, it will be worth it for us to go through and see what each method does:

[PRE40]

The constructor is simple (as are most component constructors) since they are designed to be created via a factory. We set the component type and make sure to set the command pointer to null so we set ourselves up for safer code later:

[PRE41]

The destructor is where that null pointer comes in handy. It is perfectly legal to delete a null pointer, so we know that this code will work, even if this component never receives a command:

[PRE42]

The `Update` function is where we perform the test to see if the mouse click intersects the rectangle created by the object. As we mentioned before, this class could work with all objects, but to simplify the code we decided we would only use this class for screen space items. The code that is important in this decision is the `GetMouse` function. This function always returns coordinates in screen space. It would be possible to check if the object was in screen space or world space and convert the coordinates using the `M5Gfx` method `ConvertScreenToWorld`.

That null pointer comes in handy here as well. Since we know that the command pointer is valid or null, we can do a debug assert to test our code before we execute it:

[PRE43]

The `Clone` method looks like you might expect after reading [Chapter 6](part0112.html#3APV00-04600e4b10ea45a2839ef4fc3675aeb7), *Creating Objects with the Prototype Pattern*. This is one situation where we always need to test for null before using the command. We can't clone a null command and it is completely valid to clone this component, whether the command has been set or not:

[PRE44]

The `SetOnClick` method allows us to set and reset the command that is associated with this component. Again, we don't need to test our command before deleting. We also don't need to test if the method parameter is non-null, because a null value is perfectly acceptable.

Even though we haven't done it for this class, this class could easily be expanded to include an `OnMouseOver` event that gets triggered when the mouse is inside the object rectangle but the mouse isn't clicked. A feature like this could have lots of uses for both UI and world objects. Implementing it would be as easy as swapping the two conditional statements in the `Update` function:

[PRE45]

# Using commands

Now that we have seen the base `M5Command` class and `UIButtonComponent` class, let's look at one of the derived commands to see how it is used in the game. The command that we will look at is a common one needed in games. This is the action that will allow us to change stages from one to the next:

[PRE46]

When used with a `UIButtonComponent`, this will allow the user to click a button and change to a new stage. As you can see, there are two ways to change the stage in the constructor and in the `SetNextStage` method. This allows the user the ability to create a command and decide later what stage it will switch to. The `Execute` method is as simple as can be since the `StageManager` is a Singleton:

[PRE47]

The following is the output:

![](img/00055.jpeg)

Figure 8 4 - An example of UIButtons in the Mach5 Engine

To be truly flexible, we would want all `UIButtons` loaded from a file. As with game objects, it would be best if menus and levels were not coupled to specific commands. At the very least, we would prefer to avoid hardcoding positions and sizes for each button. This proved to be easy with game objects. The Player or Raider game objects are so specific that when reading a level file, we only need to overwrite the position of each object. The size, texture name, and other attributes can be read from the more specific archetype file.

Buttons are more difficult since each one may use a different texture name, have a different size, and use a different command. We can't set this data in a button archetype file because all buttons will be different. Furthermore, game commands that need to control a specific game object are difficult to load from file since we have no information about the object except the type. This means that while we could create and load a command that controls the player, which we only have one of, we can't create and load a command that controls an arbitrary Raider, since we could have many per stage.

Having a high-quality level editor would solve both issues because the tool can manage data better. This could even include assigning object IDs that could be used by commands in the game. For this book, defining archetypes for every button worked well. While this may seem like a lot of work, the data in each archetype file would have otherwise been hardcoded into a `.cpp` file.

# Summary

In this chapter, we focused on creating flexible, reusable buttons. Even though the UI may not be as fun to code or talk about as gameplay mechanics, to the player, it is just as important. That is why creating a good system to add and manage the UI in an intelligent way is so vital to making a great game.

We took an in-depth look at C++ function pointers and pointers to members. This is well known for being confusing and difficult. However, by mastering the techniques, we could create flexible commands that can call any C style function or C++ object method.

While this technique isn't always needed, in the case of UI, it allowed us to create an incredibly flexible system. Our UI objects and most commands can be set up and read from a file. If you were to create a level editor, you could easily use this system to create and read all UI buttons and commands from a file.

Now that we have a flexible system for creating the UI, let's move on to another problem everyone has when making games. In the next chapter, we will talk about a pattern that will allow us to better separate our engine code from our gameplay code.