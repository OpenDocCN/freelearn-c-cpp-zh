# Chapter 4. Improve Programming with Functions, Math, and Timing

As a digital artist, I need special conditions to be able to work. We all need our own environment and ambience to be productive. Even if each one of us has his/her own way, there are many things in common.

In this chapter, I want to give you elements that will make you more comfortable to write source code that is easily readable, reusable and, as much as possible, beautiful. Like Yin and Yang, for me there has always been a Zen-like quality to the artistic and coding sides of me. Here is where I can deliver some programming pearls of wisdom to bring peace of mind to your creative side.

We are going to learn something we have already used a bit before: functions. They contribute to improve both readability and efficiency at the same time. As we do that, we'll touch on some mathematics and trigonometry often used in many projects. We'll also talk about some approaches to calculation optimization, and we'll finish this chapter with timing-related events or actions within the Arduino's firmware.

It is going to be a very interesting chapter before the real dive into pure Arduino's projects!

# Introducing functions

A function is a piece of code defined by a name and that can be reused/executed from many different points in a C program. The name of a function has to be *unique* in a C program. It is also `global`, which means, as you already read for variables, it can be used everywhere in the C program containing the function declaration/definition in its scope (see the The scope concept section in [Chapter 3](ch03.html "Chapter 3. C Basics – Making You Stronger"), *C Basics – Making You Stronger*).

A function can require special elements to be passed to it; these are called **arguments**. A function can also produce and return **results**.

## Structure of a function

A function is a block of code that has a header and a body. In standard C, a function's declaration and definition are made separately. The declaration of the function is specifically called the declaration of the prototype of the function and has to be done in the **header file** (see [Chapter 2](ch02.html "Chapter 2. First Contact with C"), *First Contact with C*).

### Creating function prototypes using the Arduino IDE

The Arduino IDE makes our life easier; it creates function prototypes for us. But in special cases, if you need to declare a function prototype, you can do that in the same code file at the beginning of the code. This provides a nice way of source code centralization.

Let's take an easy example, we want to create a function that sums two integers and produces/returns the result. There are two arguments that are integer type variables. In this case, the result of the addition of these two `int` (integer) values is also an `int` value. It doesn't have to be, but for this example it is. The prototype in that case would be:

[PRE0]

### Header and name of functions

Knowing what prototype looks like is interesting because it is similar to what we call the header. The header of a function is its first statement definition. Let's move further by writing the global structure of our function `mySum`:

[PRE1]

The header has the global form:

[PRE2]

`returnType` is a variable type. By now, I guess you understand the `void` type better. In the case where our function doesn't return anything, we have to specify it by choosing `returnType` equals to `void`.

`functionName` has to be chosen to be easy to remember and should be as self-descriptive as possible. Imagine supporting code written by someone else. Finding `myNiceAndCoolMathFunction` requires research. On the other hand, `mySum` is self-explanatory. Which code example would you rather support?

The Arduino core (and even C) follows a naming convention called camel case. The difference between two words, because we *cannot* use the blank/space character in a function name, is made by putting the first letter of words as uppercase characters. It isn't necessary, but it is recommended especially if you want to save time later. It's easier to read and makes the function self-explanatory.

`mysum` is less readable than `mySum`, isn't it? Arguments are a series of variable declarations. In our `mySum` example, we created two function arguments. But we could also have a function without arguments. Imagine a function that you need to call to produce an action that would always be the same, not depending on variables. You'd make it like this:

[PRE3]

### Note

Variables declared within a function are known only to the function containing them. This is what's known as "scope". Such variables declared within a function cannot be accessed anywhere else, but they can be "passed". Variables which can be passed are known as **arguments**.

### Body and statements of functions

As you probably intuitively understood, the body is the place where everything happens; it's where all of a function's instruction steps are constructed.

Imagine the body as a real, pure, and new block of source code. You can declare and define variables, add conditions, and play with loops. Imagine the body (of instructions) as where the sculptor's clay is shaped and molded and comes out in the end with the desired effect; perhaps in one piece or many, perhaps in identical copies, and so on. It's the manipulation of what there is, but remember: Garbage in, garbage out!

You can also, as we just introduced, return a variable's value. Let's create the body of our `mySum` example:

[PRE4]

`int result;` declares the variable, and names it `result`. Its scope is the same as the scope of arguments. `result = m + n;` contains two operators, and you already know that `+` has a higher precedence than `=` which is pretty good, as the mathematical operation is made first and then the result is stored in the `result` variable. This is where the magic happens; take two operators, and make one out of them. Remember that, in a combination of multiple mathematical operations, do not forget the order of precedence; it's critical so that we don't get unexpected results.

At last, `return result;` is the statement that makes the function call resulting into a value. Let's check an actual example of the Arduino code to understand this better:

[PRE5]

As you have just seen, the `mySum` function has been defined and called in the example. The most important statement is `currentResult = mySum(i,i+1);`. Of course, the `i` and `i+1` trick is interesting, but the most important thing to recognize here is the usage of the variable `currentResult` that was declared at the beginning of the`loop()` function.

In programming, it's important to recognize that everything at the right (contents) goes into the left (the new container). According to the precedence rules, a function call has a precedence of 2 against 16 for the `=` assignment operator. It means the call is made first and the function returns the result of the `+` operation, as we designed it. From this point of view, you just learned something very important: *The call statement of a function returning a result is a value*.

You can check *Appendix B, Operator Precedence in C and C++* for the entire precedencies list. As with all values within a variable, we can store it into another, here inside the integer variable `result`.

## Benefits of using functions

Programming is about writing pieces of code for general and specific purposes. Using functions is one of the best ways of segmenting your code.

### Easier coding and debugging

Functions can really help us to be better organized. While designing the program, we often use pseudo-codes and this is also the step when we notice that there are a lot of common statements. These common statements may often be put inside functions.

The function/call pattern is also easier to debug. We have only one part of code where the function sits. If there is a problem, we can debug the function itself just once, and all the calls will then be fixed instead of modifying the whole part of the code.

![Easier coding and debugging](img/7584_04_001.jpg)

Functions make your code easier to debug

### Better modularity helps reusability

Some part of your code will be high level and general. For instance, at some point, you may need a series of statements that can cut an array into pieces, then regroup all values following a basic rule. This series could be the body of a function. In another way, coding a function that converts Fahrenheit units into Celsius could interest you. These two examples are general-purpose functions.

In contrast, you can also have a specific function whose sole purpose is to convert U.S. Dollars to French Francs. You may not call it very often, but if occasionally necessary, it is always ready to handle that task.

In both cases, the function can be used and of course, re-used. The idea behind this is to save time. It also means that you can grab some already existing functions and re-use them. Of course, it has to be done following some principles, such as:

*   Code licensing
*   Respect the API of the function that can be a part of a library
*   Good match for your purpose

**Code licensing** issue is an important point. We are used to grabbing, testing, and copy/pasting things, but the code you find isn't always in the public domain. You have to take care of the license file that is often included in a code release archive, and in the first line of the code too, where comments can help you understand the conditions to respect the re-use of it.

**Application Programming Interface** (**API**) means you have to conform yourself to some documentation before using the material related to that API. I understand that purists would consider this a small abuse, but it is a pretty pragmatic definition.

Basically, an API defines specifications for routines, data structures, and other code entities that can be re-used inside other programs. An API specification can be documentation of a library. In that case, it would precisely define what you can and cannot do.

The good-match principle can seem obvious, but sometimes out of convenience we find an existing library and choose to use it rather than coding our own solution. Unfortunately, sometimes in the end we only add more complication than originally intended. Doing it ourselves may fulfil the simple need, and will certainly avoid the complexities and idiosyncrasies of a more comprehensive solution. There's also the avoidance of a potential performance hit; you don't buy a limo when all you really need is to walk to the supermarket just down the street.

### Better readability

It is a consequence of the other benefits, but I want to make you understand that this is more vital than commenting your code. Better readability means saving time to focus on something else. It also means easier code upgrade and improvement steps.

# C standard mathematical functions and Arduino

As we have already seen, almost all standard C and C++ entities supported by the compiler **avr-g++** should work with Arduino. This is also true for C mathematical functions.

This group of functions is a part of the (famous) C standard library. A lot of functions of this group are inherited in C++. There are some differences between C and C++ in the use of complex numbers. C++ doesn't provide complex numbers handling from that library but from its own C++ standard library by using the class template `std::complex`.

Almost all these functions are designed to work with and manipulate floating-point numbers. In standard C, this library is known as `math.h` (a filename), which we mention in the header of a C program, so that we can use its functions.

## Trigonometric C functions in the Arduino core

We often need to make some trigonometric calculations, from determining distances an object has moved, to angular speed, and many other real-world properties. Sometimes, you'll need to do that inside Arduino itself because you'll use it as an autonomous smart unit without any computers in the neighborhood.

The Arduino core provides the classic trigonometric functions that can be summarized by writing their prototypes. A major part of these return results in radians. Let's begin by reviewing our trigonometry just a bit!

### Some prerequisites

I promise, I'll be quick and light. But the following lines of text will save you time looking for your old and torn school book. When I learn knowledge from specific fields, I personally like to have all I need close at hand.

#### Difference between radians and degrees

**Radian** is the unit used by many trigonometric functions. Then, we have to be clear about radians and degrees, and especially how to convert one into the other. Here is the official radian definition: **Alpha** is a ratio between two distances and is in radian units.

![Difference between radians and degrees](img/7584_04_002.jpg)

Radian definition

**Degree** is the 1/360 of a full rotation (complete circle). Considering these two definitions and the fact that a complete rotation equals 2π, we can convert one into the other:

### Note

angleradian = angledegree x π/180

angledegree = angleradian x 180/π

#### Cosine, sine, and tangent

Let's see the trigonometric triangle example:

![Cosine, sine, and tangent](img/7584_04_003.jpg)

Considering the angle A in radians, we can define cosine, sine, and tangent as follows:

*   cos(A) = b/h
*   sin(A) = a/h
*   tan(A) = sin(A)/cos(A) = a/b

Cosine and sine evolve from -1 to 1 for value of angles in radians, while tangent has some special points where it isn't defined and then evolves cyclically from -∞ to +∞. We can represent them on the same graph as follows:

![Cosine, sine, and tangent](img/7584_04_004.jpg)

Graphical cosine, sine, and tangent representation

Yes, of course, those functions oscillate, infinitely reproducing the same evolutions. It is good to keep in mind that they can be used for pure calculations but also to avoid overly linear value evolution in the time by replacing linearity by smoother oscillations. We'll see that a bit later.

We know how to calculate a cosine/sine/tangent when we have an angle, but how to calculate an angle when we already have the cosine/sine/tangent?

#### Arccosine, arcsine, and arctangent

Arccosine, arcsine, and arctangent are called inverse trigonometric functions. These functions are used to calculate an angle when you already have the distance ratios that I mentioned before.

They are called inverse because this is the inverse/reciprocal process of the previously seen trigonometric function. Basically, these functions provide you an angle, but considering the periodicity, they provide a lot of angles. If k is an integer, we can write:

*   sin (A) = x ó A = arcsin(x) + 2kπ or y = π – arcsin(x) + 2kπ
*   cos (A) = x ó A = arccos(x) + 2kπ or y = 2π – arccos (x) + 2kπ
*   tan (A) = x ó A = arctan(x) + kπ

These are the right mathematical relationships. Practically, in usual cases, we can drop the full rotation cases and forget about the 2kπ of the cosine and sine cases and kπ of the tangent case.

### Trigonometry functions

`Math.h` contains the trigonometry function's prototype, so does the Arduino core:

*   `double cos (double x);` returns the cosine of `x` radians
*   `double sin (double x);` returns the sine of `x` radians
*   `double tan (double x);` returns the tangent of `x` radians
*   `double acos (double x);` returns A, the angle corresponding to cos (A) = `x`
*   `double asin (double x);` returns A, the angle corresponding to sin (A) = `x`
*   `double atan (double x);` returns A, the angle corresponding to tan (A) = `x`
*   `double atan2 (double y, double x);` returns arctan (`y`/`x`)

## Exponential functions and some others

Making calculations, even basic ones, involves other types of mathematical functions, namely power, absolute value, and so on. The Arduino core then implements those. Some mathematical functions are given as follows:

*   `double pow (double x, double y);` returns `x` to power `y`
*   `double exp (double x);` returns the exponential value of `x`
*   `double log (double x);` returns the natural logarithm of `x` with `x` > 0
*   `double log10 (double x);` returns the logarithm of `x` to base 10 with `x` > 0
*   `double square (double x);` returns the square of `x`
*   `double sqrt (double x);` returns the square root of `x` with `x` >= 0
*   `double fabs (double x);` returns the absolute value of `x`

Of course, mathematical rules, especially considering range of values, have to be respected. This is why I added some conditions of `x` to the list.

All these functions are very useful, even for solving small problems. One day, I was teaching someone at a workshop and had to explain about measuring temperature with a sensor. This student was quite motivated but didn't know about these functions because she only played with inputs and outputs without converting anything (because she basically didn't need that). We then learned these functions, and she ended by even optimizing her firmware, which made me so proud of her!

Now, let's approach some methods of optimization.

# Approaching calculation optimization

This section is an approach. It means it doesn't contain all the advanced tips and tricks for programming optimizations, but contains the optimizations on pure calculation.

Generally, we design an idea, code a program, and then optimize it. It works fine for huge programs. For smaller ones, we can optimize while coding.

### Note

Normally, our firmware is small and so I'd suggest that you consider this as a new rule: Write each statement keeping optimization in mind.

I could add something else right now: Don't kill the readability of your code with too many cryptic optimization solutions; I thought of *pointers* while writing that. I'll add a few lines about them in order to make you familiar with, at least, the concept.

## The power of the bit shift operation

If I consider an array to store things, I almost always choose the size as a power of two. Why? Because the compiler, instead of performing the array indexing by using a CPU-intensive multiply operation, can use the more efficient bit shift operation.

### What are bit operations?

Some of you must have already understood the way I work; I'm using a lot of pretexts to teach you new things. Bitwise operators are specific operators for bits. Some cases require this kind of calculation. I can quote two cases that we'll learn about in the next part of this book:

*   Using shift registers for multiplexing
*   Performing arithmetic operations, for powers of 2, involving the multiply and divide operator

There are four operators and two bit shift operators. Before we dive into it, let's learn a bit more about the binary numeral system.

### Binary numeral system

We are used to counting using the decimal system, also called decimal numeral system or base-10 number system. In this system, we can count as:

0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12...

Binary numeral system is the system used under the hood in computers and digital electronic devices. It is also named base-2 system. In this system we count as follows:

0, 1, 10, 11, 100, 101, 110, 111...

#### Easily converting a binary number to a decimal number

A nice trick to convert from binary to decimal, start by counting the position of 0 and 1, starting from the index 0.

Let's take 110101\. It can be represented as follows:

| Positions | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
|   | 1 | 0 | 1 | 0 | 1 | 1 |

Then, I can write this sum of multiplications and it equals the decimal version of my 110101 number:

1 x 20 + 0 x 21 + 1 x 22 + 0 x 23 + 1 x 24 + 1 x 25 = 1 + 4 + 16 + 32 = 53

Each bit *decides* if we have to consider the power of 2, considering its position.

### AND, OR, XOR, and NOT operators

Let's have a look at these four operators.

#### AND

The bitwise AND operator is written with a single ampersand: `&`. This operator operates on each bit position independently according to the following rules:

*   0 `&` 0 == 0
*   0 `&` 1 == 0
*   1 `&` 0 == 0
*   1 `&` 1 == 1

Let's take a real example with integers, which are a 16-bit value:

[PRE6]

To find the result easily, we have to compare each bit one by one for each position while following the preceding rules.

#### OR

The bitwise OR operator is written with a single vertical bar: `|`. It can be done by pressing *Alt* + *Shift* + *l* (letter L) on OSX and *Shift* + *\* on other PC keyboards. This operator operates on each bit position independently according to the following rules:

*   0 `|` 0 == 0
*   0 `|` 1 == 1
*   1 `|` 0 == 1
*   1 `|` 1 == 1

#### XOR

The bitwise XOR operator is written with a single caret symbol: `^`. This operator operates on each bit position independently according to the following rules:

*   0 `^` 0 == 0
*   0 `^` 1 == 1
*   1 `^` 0 == 1
*   1 `^` 1 == 0

It is the exclusive version of OR, thus the name XOR.

#### NOT

The bitwise XOR operator is written with a tilde symbol: `~`. It is a unary operator, which means, if you remember this term correctly, it can apply to one number only. I call it the *bit changer* in my workshops. It changes each bit to its opposite:

*   `~`0 == 1
*   `~`1 == 0

Let's take a real example with integers, which are 16-bit values as you know:

[PRE7]

As you already know, the `int` type in C is a signed type ([Chapter 3](ch03.html "Chapter 3. C Basics – Making You Stronger"), *C Basics – Making You Stronger*) that is able to encode numbers from -32,768 to 32,767—negative numbers too.

### Bit shift operations

Coming from C++, the left shift and the right shift operators are respectively symbolized by `<<` and `>>`. It is easy to remember, the double << goes to the left, and the other one >> to the right. Basically, it works like this:

[PRE8]

It is quite easy to see how it works. You shift all bits from a particular number of positions to the left or to the right. Some of you would have noticed that this is the same as multiplying or dividing by 2\. Doing `<< 1` means multiply by 2, `>> 1` means divide by 2\. `<< 3` means multiply by 8 (23), `>> 5` means divide by 32 (25), and so on.

### It is all about performance

Bitwise operations are primitive actions directly supported by the processor. Especially with embedded systems, which are still not as powerful as normal computers, using bitwise operations can dramatically improve performance. I can write two new rules:

*   Using power of 2 as the array size drives the use of bit shift operators internally/implicitly while the CPU performs index calculations. As we just learned, multiplying/dividing by 2 can be done very efficiently and quickly with bit shift.
*   All your multiplications and divisions by a power of 2 should be replaced by bit shifting.

This is the nicest compromise between cryptic code and an efficient code. I used to do that quite often. Of course, we'll learn real cases using it. We are still in the most theoretical part of this book, but everything here will become clear quite soon.

## The switch case labels optimization techniques

The `switch`…`case` conditional structure can also be optimized while you are writing it.

### Optimizing the range of cases

The first rule is to place all cases of the considered switch in the narrowest range possible.

In such a case, the compiler produces what we call a *jump table of case labels*, instead of generating a huge `if`-`else`-`if` cascade. The jump table based `switch`…`case` statement 's performance is independent of the number of case entries in the `switch` statement.

### Note

So, place all cases of the switch in the narrowest range possible.

### Optimizing cases according to their frequency

The second rule is to place all cases sorted from the most frequently occurring to the least frequently occurring when you know the frequency.

As mentioned before, in cases where your `switch` statement contains cases placed far apart, because you cannot handle that in another way, the compiler replaces the `switch` statement and generates `if`-`else`-`if` cascades. It means it will always be better to reduce the potential number of comparisons; this also means that if the cases that are most probable are placed at the beginning, you maximize your chances to do that.

### Note

So, place all cases sorted from the most frequently occurring to the least frequently occurring.

## Smaller the scope, the better the board

As I already mentioned when we talked about a variables' scope, always use the smallest scope possible for any variables. Let's check this example with a function named `myFunction`:

[PRE9]

`temporaryVariable` is only required in one case, when `valueToTest` equals `1`. If I declare `temporaryVariable` outside of the `if` statement, whatever the value of `valueToTest`, `temporaryVariable` will be created.

In the example I cite, we save memory and processing; in all cases where `valueToTest` is not equal to `1`, the variable `temporaryVariable` is not even created.

### Note

Use the smallest scope possible for all your variables.

## The Tao of returns

Functions are usually designed with a particular idea in mind, they are modules of code able to perform specific operations through the statements that they include and are also able to return a result. This concept provides a nice way to forget about all those specific operations performed inside the function when we are outside of the function. We know the function has been designed to provide you a result when you give arguments to it.

Again, this is a nice way to focus on the core of your program.

### Direct returns concept

As you may have already understood, declaring a variable creates a place in memory. That place cannot be used by something else, of course. The process that creates the variable consumes processor time. Let's take the same previous example detailed a bit more:

[PRE10]

What could I improve to try to avoid the use of `temporaryVariable`? I could make a *direct return* as follows:

[PRE11]

In the longer version:

*   We were inside the `valueToTest == 1` case thus `valueToTest` equals `1`
*   I directly put the calculation in the `return` statement

In that case, there is no more temporary variable creation. There are some cases where it can be more readable to write a lot of temporary variables. But now, you are aware that it is worth finding compromises between readability and efficiency.

### Note

Use a direct return instead of a lot of temporary variables.

### Use void if you don't need return

I often read code including functions with a return type that didn't return anything. The compiler may warn you about that. But in case it didn't, you have to take care of it. A call to a function that provides a return type will always pass the return value even if nothing inside the function's body is really returned. This has a CPU cost.

### Note

Use `void` as a return type for your functions if they don't return anything.

## Secrets of lookup tables

**Lookup tables** are one of the most powerful tricks in the programming universe. They are arrays containing precalculated values and thus replace heavy runtime calculations by a simpler array index operation. For instance, imagine you want to track positions of something by reading distances coming from a bunch of distance sensors. You'll have *trigonometric* and probably *power* calculations to perform. Because they can be time consuming for your processor, it would be smarter and cheaper to use array content reading instead of those calculations. This is the usual illustration for the use of lookup tables.

These lookup tables can be precalculated and stored in a static program's storage memory, or calculated at the program's initialization phase (in that case, we call them *prefetched lookup tables*).

Some functions are particularly expensive, considering the CPU work. Trigonometric functions are one such function that can have bad consequences as the storage space and memory are limited in embedded systems. They are typically prefetched in code. Let's check how we can do that.

### Table initialization

We have to precalculate the cosine **Look Up Table** (**LUT**). We need to create a small precision system. While calling cos(x) we can have all values of x that we want. But if we want to prefetch values inside an array, which has by design a finite size, we have to calculate a finite number of values. Then, we cannot have our cos(x) result for all float values but only for those calculated.

I consider precision as an angle of 0.5 degrees. It means, for instance, that the result of cosine of 45 degrees will be equal to the cosine of 45 degrees 4 minutes in our system. Fair enough.

Let's consider the Arduino code. You can find this code in the `Chapter04`/`CosLUT`/ folder:

[PRE12]

`cosLUT` is declared as an array of the type `float` with a special size. 360 * 1/(precision in degrees) is just the number of elements we need in our array considering the precision. Here, precision is 0.5 degrees and of course, the declaration could be simplified as follows:

[PRE13]

We also declared and defined a `DEG2RAD` constant that is useful to convert degrees to radians. We declared `cosinePrecision` and `cosinePeriod` in order to perform those calculations once.

Then, we defined an `initCosineLUT()`function that performs the precalculation inside the `setup()` function. Inside its body, we can see a loop over index `i`, from `i=0` to the size of the array minus one. This loop precalculates values of cosine(x) for all values of x from 0 to 2π. I explicitly wrote the x as `i * DEG2RAD * precision` in order to keep the precision visible.

At the board initialization, it calculates all the lookup table values once and provides these for further calculation by a simple array index operation.

### Replacing pure calculation with array index operations

Now, let's retrieve our cosine values. We can easily retrieve our values by accessing our LUT through another function, shown as follows:

[PRE14]

`angle * 1 / cosinePrecision` gives us the angle considering the given precision of our LUT. We apply a modulo operation considering the `cosinePeriod` value to wrap values of higher angles to the limit of our LUT, and we have our index. We directly return the array value corresponding to our index.

We could also use this technique for root square prefetching. This is the way I used it in another language when I coded my first iOS application named **digital collisions** **(** [http://julienbayle.net/blog/2012/04/07/digital-collisions-1-1-new-features](http://julienbayle.net/blog/2012/04/07/digital-collisions-1-1-new-features)). If you didn't test it, this is an application about generative music and visuals based on physical collision algorithms. I needed a lot of distance and rotation calculations. Trust me, this technique turned the first sluggish prototype into a fast application.

## Taylor series expansion trick

There is another nice way to save CPU work that requires some math. I mean, a bit more advanced math. Following words are very simplified. But yes, we need to focus on the C side of things, and not totally on mathematics.

Taylor series expansions are a way to approximate almost every mathematical expression at a particular point (and around it) by using polynomial expressions.

### Note

A polynomial expression is similar to the following expression:

P(x) = a + bx + cx2 + dx3

P(x) is a polynomial function with a degree 3\. a, b, c, and d are float numbers.

The idea behind a Taylor series is that we can approximate an expression by using the first term of the theoretical infinite sums that represent this expression. Let's take some examples.

For instance, considering x evolving from -π and π; we can write the function sine as follows:

sin(x) ≈ x - x3/6 + x5/120 - x7/5040

The sign ≈ means "approximately equals". Inside a reasonable range, we can replace sin(x) by x - x3/6 + x5/120 - x7/5040\. There is no magic, just mathematical theorems. We can also write x evolving from -2 to 3 as follows:

ex ≈ 1 + x + x2/2 + x3/6 + x4/24

I could add some other examples here, but you'll be able to find this in the *Appendix D, Some Useful Taylor Series for Calculation Optimization*. These techniques are some powerful tricks to save CPU time.

## The Arduino core even provides pointers

Pointers are more complicated techniques for beginners in C programming but I want you to understand the concept. They are not data, but they point to the starting point of a piece of data. There are at least two ways to pass data to a function or something else:

*   Copy it and pass it
*   Pass a pointer to it

In the first case, if the amount of data is too large our memory stack would explode because the whole data would be copied in the stack. We wouldn't have a choice other than a pointer pass.

In this case, we have the reference of the place where the data is stored in memory. We can operate exactly as we want but only by using pointers. Pointers are a smart way to deal with any type of data, especially arrays.

# Time measure

Time is always something interesting to measure and to deal with, especially in embedded software that is, obviously, our main purpose here. The Arduino core includes several time functions that I'm going to talk about right now.

There is also a nice library that is smartly named **SimpleTimer Library** and designed as a GNU LGPL 2.1 + library by *Marcello Romani*. This is a good library based on the `millis()` core function which means the maximum resolution is 1 ms. This will be more than enough for 99 percent of your future projects. *Marcello* even made a special version of the library for this book, based on `micros()`.

The Arduino core library now also includes a native function that is able to have a resolution of 8 microseconds, which means you can measure time delta of 1/8,000,000 of a second; quite precise, isn't it?

I'll also describe a higher resolution library **FlexiTimer2** in the last chapter of the book. It will provide a high-resolution, customizable timer.

## Does the Arduino board own a watch?

The Arduino board chip provides its *uptime*. The *uptime* is the time since the board has started. It means you cannot natively store absolute time and date without keeping the board up and powered. Moreover, it will require you to set up the absolute time once and then keep the Arduino board powered. It is possible to keep the board autonomously powered. I also talk about that later in this book.

### The millis() function

The core function `millis()` returns the number of milliseconds since the board has been started the last time. For your information, 1 millisecond equals 1/1000 of a second.

The Arduino core documentation also provides that this number will go back to zero after approximately 50 days (this is called the timer overflow). You can smile now, but imagine your latest installation artistically illustrating the concept of time in the MoMA in NYC which, after 50 days, would get totally messed up. You would be interested to know this information, wouldn't you? The return format of `millis()` is *unsigned long*.

Here is an example you'll have to upload to your board in the next few minutes. You can also find this code in the `Chapter04`/ `measuringUptime`/ folder:

[PRE15]

Can you optimize this (only for pedagogical reasons as this is a very small program)? Yes, indeed, we can avoid the use of the `measuredTime` variable. It would look more like this:

[PRE16]

It is also beautiful in its simplicity, isn't it? I'm sure you'll agree. So upload this code on your board, start the Serial Monitor, and look at it.

### The micros() function

If you needed more precision, you could use the `micros()` function. It provides uptime with a precision of 8 microseconds as written before but with an overflow of approximately 70 minutes (significantly less than 50 days, right?). We gain precision but loose overflow time range. You can also find the following code in the `Chapter04`/`measuringUptimeMicros`/ folder:

[PRE17]

Upload it and check the Serial Monitor.

## Delay concept and the program flow

Like Le Bourgeois Gentilhomme who spoke prose without even realizing it, you've already used the `delay()` core function and haven't realized it. Delaying an Arduino program can be done using the `delay()` and `delayMicroseconds()` functions directly in the `loop()` function.

Both functions drive the program to make a pause. The only difference is that you have to provide a time in millisecond to `delay()` and a time in microseconds to `delayMicroseconds()`.

### What does the program do during the delay?

Nothing. It waits. This sub-subsection isn't a joke. I want you to focus on this particular point because later it will be quite important.

### Note

When you call `delay` or `delayMicroseconds` in a program, it stops its execution for a certain amount of time.

Here is a small diagram illustrating what happens when we power on our Arduino:

![What does the program do during the delay?](img/7584_04_012.jpg)

One lifecycle of a Arduino's firmware

Now here is a diagram of the firmware execution itself, which is the part that we will work with, in the next rows:

![What does the program do during the delay?](img/7584_04_005.jpg)

The firmware life cycle with the main part looping

Accepting the fact that when `setup()` stops, the `loop()` function begins to loop, everything in `loop()` is continuous. Now look at the same things when delays happen:

![What does the program do during the delay?](img/7584_04_006.jpg)

The firmware life cycle with the main part looping and breaking when delay() is called

The whole program breaks when `delay()` is called. The length of the break depends on the parameter passed to `delay()`.

We can notice that everything is done sequentially and in time. If a statement execution takes a lot of time, Arduino's chip executes it, and then continues with the next task.

In that very usual and common case, if one particular task (statements, function calls, or whatever) takes a lot of time, the whole program could be hung and produce a hiccup; consider the user experience.

Imagine that concrete case in which you have to read sensors, flip-flop some switches, and write information to a display *at the same time*. If you do that sequentially and you have a lot of sensors, which is quite usual, you can have some lag and slowdown in the display of information because that task is executed after the other one in `loop()`.

![What does the program do during the delay?](img/7584_04_007.jpg)

An Arduino board busy with many inputs and outputs

I usually teach my students at least two concepts in dealing with that only-one-task property that can feel like a limitation:

*   Thread
*   Interrupt handler (and subsequent interrupt service routine concept)

I obviously teach another one: *The polling*. **The polling is a special interrupt case from where we will begin.**

### The polling concept – a special interrupt case

You know the poll term. I can summarize it as "ask, wait for an answer, and keep it somewhere".

If I wanted to create a code that reads inputs, and performs something when a particular condition would be verified with the value of these inputs, I would write this pseudo-code:

[PRE18]

What could be annoying here? I cyclically poll new information and have to wait for it.

During this step, nothing more is done, but imagine that the input value remains the same for a long time. I'd request this value cyclically in the loop, constraining the other tasks to wait for nothing.

It sounds like a waste of time. Normally, polling is completely sufficient. It has to be written here instead of what other raw programmers could say to you.

We are creators, we need to make things communicate and work, and we can and like to test, don't we? Then, you just learned something important here.

### Note

Don't design complex program workarounds before having tested basic ones.

One day, I asked some people to design basic code. Of course, as usual, they were connected to the Internet and I just agreed because we are almost all working like that today, right? Some people finished before others.

Why? A lot of the people who finished later tried to build a nice multithreaded workaround using a messaging system and an external library. The intention was good, but in the time we had, they didn't finish and only had a nice Arduino board, some wired components, and a code that wasn't working on the table.

Do you want to know what the others had on their desktop? A polling-based routine that was driving their circuits perfectly! Time wasted by this polling-based firmware was just totally unimportant considering the circuit.

### Note

Think about hardcore optimizations, but first test your basic code.

### The interrupt handler concept

Polling is nice but a bit time consuming, as we just figured out. The best way would be to be able to control when the processor would have to deal with inputs or outputs in a smarter way.

Imagine our previously drawn example with many inputs and outputs. Maybe, this is a system that has to react according to a user action. Usually, we can consider the user inputs as much slower than the system's ability to answer.

This means we could create a system that would interrupt the display as soon as a particular event would occur, such as a user input. This concept is called an *event-based interrupt system*.

The *interrupt* is a signal. When a particular event occurs, an interrupt message is sent to the processor. Sometimes it is sent externally to the processor (hardware interrupt) and sometimes internally (software interrupt).

This is how the disk controller or any external peripheral informs the processor of the main unit that it has to provide this or that at the right moment.

The interrupt handler is a routine that handles the interrupt by doing something. For instance, on the move of the mouse, the computer operating system (commonly called the OS) has to redraw the cursor in another place. It would be crazy to let the processor itself test each millisecond whether the mouse has moved, because the CPU would be running at 100 percent utilization. It seems smarter to have a part of the hardware for that purpose. When the mouse movement occurs, it sends an interrupt to the processor, and this later redraws the mouse.

In the case of our installation with a huge number of inputs and outputs, we can consider handling the user inputs with an interrupt. We would have to implement what is called an **Interrupt Service Routine** (**ISR**), which is a routine called only when a physical world event occurs, that is, when a sensor value changes or something like that.

Arduino now provides a nice way to attach an interrupt to a function and it is now easy to design an ISR (even if we'll learn to do that a bit later). For instance, we can now react to the change of the value of an analog thermal sensor using ISR. In this case, we won't permanently poll the analog input, but we'll let our low-level Arduino part do that. Only when a value changes (rises or falls) depending on how we have attached the interrupt, would this act as a trigger and a special function would execute (for instance, the LCD display gets updated with the new value).

Polling, ISR, and now, we'll evoke threads. Hang on!

### What is a thread?

A thread is a running program flow in which the processor executes a series of tasks, generally looping, but not necessarily.

With only one processor, it is usually done by *time-division multiplexing*, which means the processor switches between the different threads according to time, that is, context switching.

![What is a thread?](img/7584_04_008.jpg)

Time-division multiplexing provides multitasking

More advanced processors provide the *multithread* feature. These behave as if they would be more than just one, each part dealing with a task at the same time.

![What is a thread?](img/7584_04_009.jpg)

Real multithreading provides tasks happening at the same time

Without going deeper into computer processors, as we aren't dealing with them right now, I can say threads are nice techniques to use in programming to make tasks run simultaneously.

Unfortunately, the Arduino core doesn't provide multithreading, nor does any other microcontroller. Because Arduino is an open source hardware project, some hackers have designed a variant of the Arduino board and created some Freeduino variant providing *concurrency*, an open source programming language, and an environment designed especially with multithreading in mind. This is out of topic here, but at least, you now have some leads if you are interested.

Let's move to the second solution to go beyond the one-task-at-a-time constraint, if we need it.

### A real-life polling library example

As introduced in the first line of this section, Marcello's library is a very nice one. It provides a polling-based way to launch timed actions.

Those actions are generally function calls. Functions that behave like that are sometimes known as callback functions. These functions are generally called as an argument to another piece of code.

Imagine that I want to make our precious LED on the Arduino board blink every 120 milliseconds. I could use a delay but it would totally stop the program. Not smart enough.

I could hack a hardware timer on the board, but that would be overkill. A more practical solution that I would use is a callback function with Marcello's `SimpleTimer` library. Polling provides a simple and inexpensive way (computationally speaking) to deal with applications that are not timer dependent while avoiding the use of interrupts that raise more complex problems like hardware timer overconsumption (hijacking), which leads to other complicated factors.

However, if you want to call a function every 5 milliseconds and that function needs 9 milliseconds to complete, it will be called every 9 milliseconds. In our case here, with 120 milliseconds required to produce a nice and eye-friendly, visible blink, we are very safe.

For your information, you don't need to wire anything more than the USB cable between the board and your computer. The board-soldered LED on Arduino is wired to digital pin 13\. Let's use it.

But first, let's download the `SimpleTimer` library for your first use of an external library.

#### Installing an external library

Download it from [http://playground.arduino.cc/Code/SimpleTimer](http://playground.arduino.cc/Code/SimpleTimer), and extract it somewhere on your computer. You will typically see a folder with at least two files inside:

*   A header file (`.h` extension)
*   A source code file (`.cpp` extension)

Now, you can see for yourself what they are. Within these files, you have the source code. Open your sketchbook folder (see [Chapter 1](ch01.html "Chapter 1. Let's Plug Things"), *Let's Plug Things*), and move the library folder into the `libraries` folder if it exists, else create this special folder:

![Installing an external library](img/7584_04_010.jpg)

The header and the source code of SimpleTimer by Marcello Romani

The next time you'll start your Arduino IDE, if you go to **Sketch** | **Import Library**, you'll see a new library at the bottom.

![Installing an external library](img/7584_04_011.jpg)

In order to include a library, you can click on it in this menu and it will write `#include <libname.h>` in your code. You can also type this by yourself.

#### Let's test the code

Upload this next code and reboot Arduino; I'm going to explain how it works. You can also find this code in the `Chapter04`/`simpleTimerBlinker`/ folder:

[PRE19]

This library is easy to use in our case. You have to include it first, of course. Then you have to declare an instance of `SimpleTimer`, which is an object construct, by declaring it.

Then I'm using a `currentLEDState` Boolean value to store the current state of the LED explicitly. At last, I declare/define `ledPin` with the number of the pin I need (in this case, 13) to make the LED blink. `setup()` is basically done with some initialization. The most important one here is the `timer.setInterval()` function.

Maybe, this is your first method call. The object timer has and embeds some methods that we can use. One of them is `setInterval`, which takes two variables:

*   A time interval
*   A callback function

We are passing a function name here (a piece of code) to another piece of code. This is the structure of a typical callback system.

`loop()` is then designed by calling the `run()` method of the timer object at each run. This is required to use it. At least, the callback function `blink()` is defined with a small trick at the end.

The comparison is obvious. I test the current state of the LED, if it is already switched on, I switch it off, else I switch it on. Then, I invert the state, which is the trick. I'm using the `!` (not) unary operator on this Boolean variable in order to flip its value, and I assign the inverted value to the Boolean itself. I could have made this too:

[PRE20]

There's really no performance gain, one way or the other. It's simply a personal decision; use whichever you prefer.

I'm personally considering the flip as a general action that has to be done every time, independent of the state. This is the reason why I proposed that you put it outside of the test structure.

# Summary

This completes the first part of this book. I hope you have been able to absorb and enjoy these first (huge) steps. If not, you may want to take the time to review something you may not have clarity on; it's always worth it to better understand what you're doing.

We know a bit more about C and C++ programming, at least enough to lead us safely through the next two parts. We can now understand the basic tasks of Arduino, we can upload our firmware, and we can test them with the basic wiring.

Now, we'll move a step further into a territory where things are more practical, and less theoretical. Prepare yourself to explore new physical worlds, where you can make things talk, and communicate with each other, where your computer will be able to respond to how you feel and react, and without wires sometimes! Again, you may want to take a little time to review something you might still be a little hazy on; knowledge is power.

The future is now!