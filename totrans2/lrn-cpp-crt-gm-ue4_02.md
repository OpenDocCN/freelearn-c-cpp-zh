# Chapter 2. Variables and Memory

To write your C++ game program, you will need your computer to remember a lot of things. Things such as where in the world is the player, how many hit points he has, how much ammunition he has left, where the items are in the world, what power-ups they provide, and the letters that make up the player's screen name.

The computer that you have actually has a sort of electronic sketchpad inside it called *memory,* or RAM. Physically, computer memory is made out of silicon and it looks something similar to what is shown in the following screenshot:

![Variables and Memory](img/00022.jpeg)

Does this RAM look like a parking garage? Because that's the metaphor we're going to use

RAM is short for Random Access Memory. It is called random access because you can access any part of it at any time. If you still have some CDs lying around, they are an example of non-random access. CDs are meant to be read and played back in order. I still remember jumping tracks on Michael Jackson's *Dangerous* album way back when switching tracks on a CD took a lot of time! Hopping around and accessing different cells of RAM, however, doesn't take much time at all. RAM is a type of fast memory access known as flash memory.

RAM is called volatile flash memory because when the computer is shut down, RAM's contents are cleared and the old contents of RAM are lost unless they were saved to the hard disk first.

For permanent storage, you have to save your data into a hard disk. There are two main types of hard disks, platter-based **Hard Disk Drives** (**HDDs**) and **Solid-state Drives** (**SSDs**). SSDs are more modern than platter-based HDDs, since they use RAM's fast-access (Flash) memory principle. Unlike RAM, however, the data on an SSD persists after the computer is shut down. If you can get an SSD, I'd highly recommend that you use it! Platter-based drives are outdated. We need a way to reserve a space on the RAM and read and write from it. Fortunately, C++ makes this easy.

# Variables

A saved location in computer memory that we can read or write to is called a *variable*.

A variable is a component whose value can vary. In a computer program, you can think of a variable as a container, into which you can store some data. In C++, these data containers (variables) have types. You have to use the right type of data container to save your data in your program.

If you want to save an integer, such as 1, 0, or 20, you will use an `int` type container. You can use float-type containers to carry around floating-point (decimal) values, such as 38.87, and you can use string variables to carry around strings of letters (think of it as a "string of pearls", where each letter is a pearl).

You can think of your reserved spot in RAM like reserving parking space in a parking garage: once we declare our variable and get a spot for it, no one else (not even other programs running on the same machine) will be given that piece of RAM by the operating system. The RAM beside your variable might be unused or it might be used by other programs.

### Tip

The operating system exists to keep programs from stepping on each other's toes and accessing the same bits of computer hardware at the same time. In general, civil computer programs should not read or write to each other's memory. However, some types of cheat programs (for example, maphacks) secretly access your program's memory. Programs such as PunkBuster were introduced to prevent cheating in online games.

## Declaring variables – touching the silicon

Reserving a spot on computer memory using C++ is easy. We'll want to name our chunk of memory that we will store our data in with a good, descriptive name.

For example, say, we know that player **hit points** (**hp**) will be an integer (whole) number, such as 1, 2, 3, or 100\. To get a piece of silicon to store the player's hp in memory, we will declare the following line of code:

[PRE0]

This line of code reserves a small chunk of RAM to store an integer (`int` is short for integer), called hp. The following is an example of our chunk of RAM used to store the player's hp. This reserves a parking space for us in memory (among all the other parking spaces), and we can refer to this space in memory by its label (hp).

![Declaring variables – touching the silicon](img/00023.jpeg)

Among all the other spaces in memory, we get one spot to store our hp data

Notice how the variable space is type-marked in this diagram as **int**: if it is a space for a double or a different type of variable. C++ remembers the spaces that you reserve for your program in memory not only by name but by the type of variable it is as well.

Notice that we haven't put anything in hp's box yet! We'll do that later—right now, the value of the hp variable is not set, so it will have the value that was left in that parking space by the previous occupant (the value left behind by another program, perhaps). Telling C++ the type of the variable is important! Later, we will declare a variable to store decimal values, such as 3.75.

### Reading and writing to your reserved spot in memory

Writing a value into memory is easy! Once you have an `hp` variable, you just write to it using the `=` sign:

[PRE1]

Voila! The player has 500 hp.

Reading the variable is equally simple. To print out the value of the variable, simply put this:

[PRE2]

This will print the value stored inside the hp variable. If you change the value of hp, and then use `cout` again, the most up-to-date value will be printed, as shown here:

[PRE3]

## Numbers are everything

Something that you need to get used to when you start computer programming is that a surprising number of things can be stored in computer memory as just numbers. A player's hp? As we just saw in the previous section, hp can just be an integer number. If the player gets hurt, we reduce this number. If the player gains health, we increase the number.

Colors can be stored as numbers too! If you've used standard image editing programs, there are usually sliders that indicate color as how much red, green, and blue are being used, such as Pixelmator's color sliders. A color is then represented by three numbers. The purple color shown in the following screenshot is (R=127, G=34, B=203):

![Numbers are everything](img/00024.jpeg)

What about world geometry? These are also just numbers: all we have to do is store a list of 3D space points (x, y, and z coordinates) and then store another list of points that explain how those points can be connected to form triangles. In the following screenshot, we can see how 3D space points are used to represent world geometry:

![Numbers are everything](img/00025.jpeg)

The combination of numbers for colors and numbers for 3D space points will let you draw large and colored landscapes in your game world.

The trick with the preceding examples is how we interpret the stored numbers so that we can make them mean what we want them to mean.

## More on variables

You can think of variables as animal-carrying cases. A cat carrier can be used to carry a cat, but not a dog. Similarly, you should use a float-type variable to carry decimal-valued numbers. If you store a decimal value inside an `int` variable, it will not fit:

[PRE4]

What's really happening here is that C++ does an automatic type conversion on 38.87, *transmogrifying* it to an integer to fit in the `int` carrying case. It drops the decimal to convert 38.87 into the integer value 38.

So, for example, we can modify the code to include the use of three types of variables, as shown in the following code:

[PRE5]

In the first three lines, we declare three boxes to store our data parts into, as shown here:

[PRE6]

These three lines reserve three spots in memory (like parking spaces). The next three lines fill the variables with the values we desire, as follows:

[PRE7]

In computer memory, this will look as shown in the following figure:

![More on variables](img/00026.jpeg)

You can change the contents of a variable at any time. You can write a variable using the `=` assignment operator, as follows:

[PRE8]

You can also read the contents of a variable at any time. That's what the next three lines of code do, as shown here:

[PRE9]

Take a look at this line:

[PRE10]

There are two uses of the word `hp` in this line. One is between double quotes, while the other is not. Words between double quotes are always output exactly as you typed them. When double quotes are not used (for example, `<< hp <`), a variable lookup is performed. If the variable does not exist, then you will get a compiler error (undeclared identifier).

There is a space in memory that is allocated for the name, a space for how many `goldPieces` the player has, and a space for the hp of the player.

### Tip

In general, you should always try to store the right type of data inside the right type of variable. If you happen to store the wrong type of data, your code may misbehave.

## Math in C++

Math in C++ is easy to do; + (plus), - (minus), * (times), / (divide by) are all common C++ operations, and proper BEDMAS order will be followed (Brackets, Exponents, Division, Multiplication, Addition, and Subtraction). For example, we can do as shown in the following code:

[PRE11]

Another operator that you might not be familiar with yet is % (modulus). Modulus (for example, 10 % 3) finds the remainder of when `x` is divided by `y`. See the following table for examples:

| Operator (name) | Example | Answer |
| --- | --- | --- |
| + (plus) | 7 + 3 | 10 |
| - (minus) | 8 - 5 | 3 |
| * (times) | 5*6 | 30 |
| / (division) | 12/6 | 2 |
| % (modulus) | 10 % 3 | 1 (because 10/3 is 3 the remainder = 1). |

However, we often don't want to do math in this manner. Instead, we usually want to change the value of a variable by a certain computed amount. This is a concept that is harder to understand. Say the player encounters an imp and is dealt 15 damage.

The following line of code will be used to reduce the player's hp by 15 (believe it or not):

[PRE12]

You might ask why. Because on the right-hand side, we are computing a new value for hp (`hp-15`). After the new value for hp is found (15 less than what it was before), the new value is written into the hp variable.

### Tip

**Pitfall**

An uninitialized variable has the bit pattern that was held in memory for it before. Declaring a variable does not clear the memory. So, say we used the following line of code:

[PRE13]

The second line of code reduces the hp by 15 from its previous value. What was its previous value if we never set hp = 100 or so? It could be 0, but not always.

One of the most common errors is to proceed with using a variable without initializing it first.

The following is a shorthand syntax for doing this:

[PRE14]

Besides `-=`, you can use += to add some amount to a variable, *= to multiply a variable by an amount, and /= to divide a variable by some amount.

### Exercises

Write down the value of `x` after performing the following operations; then, check with your compiler:

| Exercises | Solutions |
| --- | --- |
| `int x = 4; x += 4;` | 8 |
| `int x = 9; x-=2;` | 7 |
| `int x = 900; x/=2;` | 450 |
| `int x = 50; x*=2;` | 100 |
| `int x = 1; x += 1;` | 2 |
| `int x = 2; x -= 200;` | -198 |
| `int x = 5; x*=5;` | 25 |

## Generalized variable syntax

In the previous section, you learned that every piece of data that you save in C++ has a type. All variables are created in the same way; in C++, variable declarations are of the form:

[PRE15]

The `variableType` tells you what type of data we are going to store in our variable. The `variableName` is the symbol we'll use to read or write to that piece of memory.

## Primitive types

We previously talked about how all the data inside a computer will at some point be a number. Your computer code is responsible for interpreting that number correctly.

It is said that C++ only defines a few basic data types, as shown in the following table:

| `Char` | A single letter, such as 'a', 'b', or '+' |
| `Short` | An integer from -32,767 to +32,768 |
| `Int` | An integer from -2,147,483,647 to +2,147,483,648 |
| `Float` | Any decimal value from approx. -1x1038 to 1x1038 |
| `Double` | Any decimal value from approx. -1x10308 to 1x10308 |
| `Bool` | true or false |

There are unsigned versions of each of the variable types mentioned in the preceding table. An unsigned variable can contain natural numbers, including 0 (x >= 0). An unsigned `short`, for example, might have a value between 0 and 65535.

### Note

If you're further interested in the difference between float and double, please feel free to look it up on the Internet. I will keep my explanations only to the most important C++ concepts used for games. If you are curious about something that's covered by this text, feel free to look it up.

It turns out that these simple data types alone can be used to construct arbitrarily complex programs. "How?" you ask. Isn't it hard to build a 3D game using just floats and integers?

It is not really difficult to build a game from float and int, but more complex data types help. It will be tedious and messy to program if we used loose floats for the player's position.

## Object types

C++ gives you structures to group variables together, which will make your life a lot easier. Take an example of the following block of code:

[PRE16]

The way this looks in memory is pretty intuitive; a Vector is just a chunk of memory with three floats, as shown in the following figure:

![Object types](img/00027.jpeg)

### Tip

Don't confuse the struct Vector in the preceding screenshot with the `std::vector` of the STL. The Vector object above is meant to represent a three-space vector, while STL's `std::vector` type represents a sized collection of values.

Here are a couple of review notes about the preceding code listing:

First, even before we use our Vector object type, we have to define it. C++ does not come with built-in types for math vectors (it only supports scalar numbers, and they thought that was enough!). So, C++ lets you build your own object constructions to make your life easier. We first had the following definition:

[PRE17]

This it tells the computer what a Vector is (it's 3 floats, all of which are declared to be sitting next to each other in the memory). The way a Vector will look in the memory is shown in preceding figure.

Next, we use our Vector object definition to create a Vector instance called `v`:

[PRE18]

The `struct` Vector definition doesn't actually create a Vector object. You can't do `Vector.x = 1`. "Which object instance are you talking about?" the C++ compiler will ask. You need to create a Vector instance first, such as Vector v1; then, you can do assignments on the v1 instance, such as v1.x = 0.

We then use this instance to write values into `v`:

[PRE19]

### Tip

We used commas in the preceding code to initialize a bunch of variables on the same line. This is okay in C++. Although you can do each variable on its own line, the approach shown here is okay too.

This makes `v` look as in the preceding screenshot. Then, we print them out:

[PRE20]

In both the lines of code here, we access the individual data members inside the object by simply using a dot (`.`). `v.x` refers to the `x` member inside the object `v`. Each Vector object will have exactly three floats inside it: one called `x`, one called `y`, and one called `z`.

### Exercise – Player

Define a C++ data struct for a Player object. Then, create an instance of your Player class and fill each of the data members with values.

#### Solution

Let's declare our Player object. We want to group together everything to do with the player into the Player object. We do this so that the code is neat and tidy. The code you read in Unreal Engine will use objects such as these everywhere; so, pay attention:

[PRE21]

The struct Player definition is what tells the computer how a Player object is laid out in memory.

### Tip

I hope you noticed the mandatory semicolon at the end of the struct declaration. struct object declarations need to have a semicolon at the end, but functions do not. This is just a C++ rule that you must remember.

Inside a Player object, we declared a string for the player's name, a float for his hp, and a Vector object for his complete xyz position.

When I say object, I mean a C++ struct (or later, we will introduce the term class).

Wait! We put a Vector object inside a Player object! Yes, you can do that.

After the definition of what a Player object has inside it, we actually create a Player object instance called me and assign it some values.

After the assignment, the me object looks as shown in the following figure:

![Solution](img/00028.jpeg)

## Pointers

A particularly tricky concept to grasp is the concept of pointers. Pointers aren't that hard to understand but can take a while to get a firm handle on.

Say we have, as before, declared a variable of the type Player in memory:

[PRE22]

We now declare a pointer to the Player:

[PRE23]

The `*` characters usually make things special. In this case, the `*` makes `ptrMe` special. The `*` is what makes `ptrMe` a pointer type.

We now want to link `ptrMe` to me:

[PRE24]

### Tip

This linkage step is very important. If you don't link the pointer to an object before you use the pointer, you will get a memory access violation.

`ptrMe` now refers to the same object as me. Changing `ptrMe` will change me, as shown in the following figure:

![Pointers](img/00029.jpeg)

## What can pointers do?

When we set up the linkage between the pointer variable and what it is pointing to, we can manipulate the variable that is pointed to through the pointer.

One use of pointers is to refer to the same object from several different locations of the code. The Player object is a good candidate for being pointed to. You can create as many pointers as you wish to the same object. Objects that are pointed to do not necessarily know that they are being pointed at, but changes can be made to the object through the pointers.

For instance, say the player got attacked. A reduction in his hp will result, and this reduction will be done using the pointer, as shown in the following code:

[PRE25]

Here's how the Player object looks now:

![What can pointers do?](img/00030.jpeg)

So, we changed `me.name` by changing `ptrMe->name`. Because `ptrMe` points to me, changes through `ptrMe` affect me directly.

Besides the funky arrow syntax (use `->` when the variable is a pointer), this concept isn't all that hard to understand.

## Address of operator &

Notice the use of the `&` symbol in the preceding code example. The `&` operator gets the memory address of a variable. A variable's memory address is where it lives in the computer memory space. C++ is able to get the memory address of any object in your program's memory. The address of a variable is unique and also, kind of, random.

Say, we print the address of an integer variable `x`, as follows:

[PRE26]

On the first run of the program, my computer prints the following:

[PRE27]

This number (the value of `&x`) is just the memory cell where the variable `x` is stored. What this means is that in this particular launch of the program, the variable `x` is located at memory cell number 0023F744, as shown in the following figure:

![Address of operator &](img/00031.jpeg)

Now, create and assign a pointer variable to the address of `x`:

[PRE28]

What we're doing here is storing the memory address of `x` inside the variable `px`. So, we are metaphorically pointing to the variable `x` using another different variable called `px`. This might look something similar to what is shown in the following figure:

![Address of operator &](img/00032.jpeg)

Here, the variable `px` has the address of the variable `x` inside it. In other words, the variable `px` is a reference to another variable. Differencing `px` means to access the variable that `px` is referencing. Differencing is done using the `*` symbol:

[PRE29]

### The Null pointers

A null pointer is a pointer variable with the value `0`. In general, most programmers like to initialize pointers to Null (`0`) on the creation of new pointer variables. Computer programs, in general, can't access the memory address `0` (it is reserved), so if you try to reference a Null pointer, your program will crash, as shown in the following screenshot:

![The Null pointers](img/00033.jpeg)

### Tip

Pointer Fun with Binky is a fun video about pointers. Take a look at [http://www.youtube.com/watch?v=i49_SNt4yfk](http://www.youtube.com/watch?v=i49_SNt4yfk).

## cin

`cin` is the way C++ traditionally takes input from the user into the program. `cin` is easy to use, because it looks at the type of variable it will put the value into as it puts it in. For example, say we want to ask the user his age and store it in an `int` variable. We can do that as follows:

[PRE30]

## printf()

Although we have used `cout` to print out variables so far, you need to know about another common function that is used to print to the console. This function is called the `printf` function. The `printf` function is included in the `<iostream>` library, so you don't have to `#include` anything extra to use it. Some people in the gaming industry prefer `printf` to `cout` (I know I do), so let's introduce it.

Let's proceed to how `printf()` works, as shown in the following code:

[PRE31]

### Tip

**Downloading the example code**

You can download the example code files from your account at [http://www.packtpub.com](http://www.packtpub.com) for all the Packt Publishing books you have purchased. If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files e-mailed directly to you.

We start with a format string. The format string is like a picture frame, and the variables will get plugged in at the locations of the `%` in the format string. Then, the entire thing gets dumped out to the console. In the preceding example, the integer variable will be plugged into the location of the first `%` (`%d`), and the character will be plugged into the location of the second `%` (`%c`), as shown in the following screenshot:

![printf()](img/00034.jpeg)

You have to use the right format code to get the output to format correctly; take a look at the following table:

| Data type | Format code |
| --- | --- |
| Int | %d |
| Char | %c |
| String | %s |

To print a C++ string, you must use the `string.c_str()` function:

[PRE32]

The `s.c_str()` function accesses the C pointer to the string, which `printf` needs.

If you use the wrong format code, the output won't appear correctly or the program might crash.

### Exercise

Ask the user his name and age and take them in using `cin`. Then, issue a greeting for him at the console using `printf()` (not `cout`).

### Solution

This is how the program will look:

[PRE33]

### Tip

A string is actually an object type. Inside it is just a bunch of chars!

# Summary

In this chapter, we spoke about variables and memory. We talked about mathematical operations on variables and how simple they were in C++.

We also discussed how arbitrarily complex data types can be built using a combination of these simpler data types, such as floats, integers, and characters. Constructions such as this are called objects.