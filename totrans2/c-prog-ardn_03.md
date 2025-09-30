# Chapter 3. C Basics – Making You Stronger

C programming isn't that hard. But it requires enough work at the beginning. Fortunately, I'm with you and we have a very good friend since three chapters – our Arduino board. We will now go deep into the C language, and I'll do my best to be more concrete and not abstract.

This chapter and the next one are truly C language-oriented because the Arduino program design requires knowledge in programming logic statements. After these two chapters, you'll be able to read any code in this book; these strong basics will also help you in further projects, even those not related to Arduino.

I will also progressively introduce new concepts that we will use later, such as functions. Don't be afraid if you don't understand it that well, I like my students to hear some words progressively sometimes even without a proper definition at first, because it helps further explanation.

So if I don't define it but talk about it, just relax, explanations are going to come further. Let's dive in.

# Approaching variables and types of data

We already used variables in the previous chapters' examples. Now, let's understand this concept better.

## What is a variable?

A **variable** is a memory storage location bounded to a symbolic name. This reserved memory area can be filled or left empty. Basically, it is used to store different types of values. We used the variable `ledPin` in our previous examples, with the keyword `int`.

Something very useful with variables is the fact that we can change their content (the value) at runtime; this is also why they are called variables, compared to constants that also store values, but that cannot be changed while the program is running.

## What is a type?

Variables (and constants) are associated with a type. A type, also called **data type**, defines the possible nature of data. It also offers a nice way to directly reserve a space with a defined size in memory. C has around 10 main types of data that can be extended as we are going to see here.

I'm deliberately only explaining the types we'll use a lot in Arduino programming. This fits with approximately 80 percent of other usual C data types and will be more than enough here.

Basically, we are using a type when we declare a variable as shown here:

[PRE0]

A space of a particular size (the size related to the `int` type) is reserved in memory, and, as you can see, if you only write that line, there is still no data stored in that variable. But keep in mind that a memory space is reserved, ready to be used to store values.

| Type | Definition | Size in memory |
| --- | --- | --- |
| `void` | This particular type is used only in *function* declarations and while defining pointers with unknown types. We'll see that in the next chapter. |   |
| `boolean` | It stores `false` or `true`. | 1 byte (8 bit) |
| `char` | It stores single-quoted characters such as `'a'` as *numbers*, following the ASCII chart ([http://en.wikipedia.org/wiki/ASCII_chart](http://en.wikipedia.org/wiki/ASCII_chart)).It is a *signed* type and stores numbers from -128 to 127; it can be unsigned and then stores numbers from 0 to 255. | 1 byte |
| `byte` | It stores numbers as *8-bit unsigned* data that means from 0 to 255. | 8 bits |
| `int` | It stores numbers as *2-bytes signed* data which means from -32,768 to 32,767 it can also be unsigned and then store numbers from 0 to 65,535. | 2 bytes (16 bit) |
| `word` | It stores numbers as *2-bytes unsigned* data exactly as *unsigned* `int` does. | 2 bytes (16 bit) |
| `long` | It stores numbers as *4-bytes signed* data, which means from -2,147,483,648 to 2,147,483,647 and can be unsigned and then stores numbers from 0 to 4,294,967,295. | 4 bytes (32 bit) |
| `float` | It basically stores numbers with a decimal point from -3.4028235E + 38 to 3.4028235E + 38 as *4-bytes signed* data.Be careful of the required precision; they only have six to seven decimal digits and can give strange rounding results sometimes. | 4 bytes (32 bit) |
| `double` | It generally stores `float` values with a precision two times greater than the `float` value.Be careful, in the Arduino IDE and the board, the `double` implementation is exactly the same as `float`; that means with only six to seven decimal digits of precision. | 4 bytes (32 bit) |
| Array | Array is an ordered structure of consecutive elements of the same type that can each be accessed with an index number. | number of elements x size of elements' type |
| `string` | It stores text strings in an array of `char` where the last element is `null` that is a particular character (ASCII code 0). Be careful of the "s" in lower case at the beginning of `string`. | number of elements * 1 byte |
| `String` | It is a particular structure of data, namely a class, that provides a nice way to use and work with strings of text.It comes with methods/functions to easily concatenate strings, split strings, and much more. Be careful of the capital "S" at the beginning of `String`. | available every time with the `length()` method |

### The roll over/wrap concept

If you go beyond the possible bounds of a type, the variable rolls over to the other side of the boundary.

The following is an example:

[PRE1]

It happens in both directions, subtracting `1` from an `int` variable storing `-32768` results in `32767`. Keep that in mind.

## Declaring and defining variables

We are going to describe how to declare then define variables and learn how to do both at the same time.

### Declaring variables

Declaration of a variable is a statement in which you specify an *identifier*, a *type*, and eventually the variable's dimensions.

An identifier is what we call the *name of the variable*. You know what the type is too. The dimensions are useful for arrays, for instance, but also for `String` (which are processed as arrays internally).

In C and all other strongly-typed languages such as Java and Python, we *must* declare variables before using them. Anyway, the compiler will complain in case you forget the declaration.

### Defining variables

The following table contains some examples of variable definition:

| Type | Example |
| --- | --- |
| `boolean` |  
[PRE2]

 |
| `char` |  
[PRE3]

 |
| `byte` |  
[PRE4]

 |
| `int` |  
[PRE5]

 |
| `word` |  
[PRE6]

 |
| `long` |  
[PRE7]

 |
| `float` |  
[PRE8]

 |
| `double` |  
[PRE9]

 |
| Array |  
[PRE10]

 |
| `string` |  
[PRE11]

 |

Defining a variable is the act of assigning a value to the memory area previously reserved for that variable.

Let's declare and define some variables of each type. I have put some explanations in the code's comments.

Here are some examples you can use, but you'll see in each piece of code given in this book that different types of declaration and definition are used. You'll be okay with that as soon as we'll wire the board.

Let's dig a bit more into the `String` type.

# String

The `String` type deserves a entire subchapter because it is a bit more than a type. Indeed, it is an object (in the sense of object-oriented programming).

Objects come with special properties and functions. Properties and functions are available natively because `String` is now a part of the Arduino core and can be seen as a pre-existing entity even if your IDE contains no line.

Again, the framework takes care of things for you, providing you a type/object with powerful and already coded functions that are directly usable.

Check out [http://arduino.cc/en/Reference/StringObject](http://arduino.cc/en/Reference/StringObject) in the Arduino website.

## String definition is a construction

We talked about definition for variables, but objects have a similar concept called **construction**.

For `String` objects, I'm talking about *construction* instead of *definition* here but you can consider both terms equal. Declaring a `String` type in Arduino core involves an object constructor, which is an object-oriented programming concept; we don't have to handle it, fortunately, at this point.

[PRE12]

## Using indexes and search inside String

`Strings` are arrays of `char` elements. This means we can access any element of a `String` type through their indexes.

Keep in mind that indexes start at `0` and not at `1`. The `String` objects implement some functions for this particular purpose.

### charAt()

Considering a `String` type is declared and defined as follows:

[PRE13]

The statement `myString.charAt(3)` returns the fourth element of the string that is: `l`. Notice the specific notation used here: we have the name of the `String` variable, a dot, then the name of the function (which is a method of the `String` object), and the parameter `3` which is passed to the function.

### Note

The `charAt()` function returns a character at a particular position inside a string.

**Syntax**: `string.charAt(int);`

`int` is an integer representing an index of the `String` value.

**Returns type**: `char`

Let's learn about other similar functions. You'll use them very often because, as we have already seen, communicating at a very low-level point of view includes parsing and processing data, which can very often be strings.

### indexOf() and lastIndexOf()

Let's consider the same declaration/definition:

[PRE14]

`myString.indexOf('r')` equals `8`. Indeed, `r` is at the ninth place of the value of the string`myString`. `indexOf(val)` and looks for the first occurrence of the value `val`.

If you want to begin your search from a particular point, you can specify a start point like that: `indexOf(val,start`), where `start` is the index from where the function begins to search for the character `val` in the string. As you have probably understood, the second argument of this function (`start`) can be omitted, the search starts from the first element of the string by default, which is `0`.

### Note

The `indexOf()` function returns the first occurrence of a string or character inside a string.

**Syntax**: `string.indexOf(val, from);`

`val` is the value to search for which can be a string or a character. `from` is the index to start the search from, which is an `int` type. This argument can be omitted. The search goes forward.

**Returns type**: `int`

Similarly, `lastIndexOf(val,start)` looks for the last occurrence of `val`, searching **backwards** from `start`, or from the last element if you omit `start`.

The `lastIndexOf()` function returns the last occurrence of a string or character inside a string.

### Note

**Syntax**: `string.lastIndexOf(val, from);`

`val` is the value to search for which is a string or a character. `from` is the index to start the search from which is an `int` type. This argument can be omitted. The search goes backwards.

**Returns type**: `int`

### startsWith() and endsWith()

The `startsWith()` and `endsWith()` functions check whether a string starts or ends with, respectively, another string passed as an argument to the function.

[PRE15]

### Note

The `startsWith()` function returns `true` if a string starts with the same characters as another string.

**Syntax**: `string.startsWith(string2);`

`string2` is the string pattern with which you want to test the string.

**Returns type**: `boolean`

I guess, you have begun to understand right now. `endsWith()` works like that too, but compares the string pattern with the end of the string tested.

### Note

The `endsWith()` function returns `true` if a string ends with the same characters as another string.

**Syntax**: `string.endsWith(string2);`

`string2` is the string pattern with which you want to test the string.

**Returns type**: `boolean`

## Concatenation, extraction, and replacement

The preceding operations also introduce new C operators. I'm using them here with strings but you'll learn a bit more about them in a more global context further.

### Concatenation

Concatenation of strings is an operation in which you take two strings and you glue them together. It results in a new string composed of the previous two strings. The order is important; you have to manage which string you want appended to the end of the other.

#### Concat()

Arduino core comes with the `string.concat()` function, which is especially designed for this purpose.

[PRE16]

### Note

The `concat()` function appends one string to another (that is concatenate in a defined order).

**Syntax**: `string.concat(string2);`

`string2` is a string and is appended to the end of string. Remember that, the previous content of the string is overwritten as a result of the concatenation.

**Returns type**: `int` (the function returns `1` if the concatenation happens correctly).

#### Using the + operator on strings

There is another way to concatenate two strings. That one doesn't use a function but an operator: `+`.

[PRE17]

This code is the same as the previous one. `+` is an operator that I'll describe better a bit later. I'm giving you something more here: a condensed notation for the `+` operator:

[PRE18]

This can also be written as:

[PRE19]

Try it. You'll understand.

### Extract and replace

String manipulation and alteration can be done using some very useful functions extracting and replacing elements in the string.

#### substring() is the extractor

You want to extract a part of a string. Imagine if the Arduino board sends messages with a specific and defined communication protocol:

`<output number>.<value>`

The output number is coded with two characters every time, and the value with three (45 has to be written as 045). I often work like that and pop out these kind of messages from the serial port of my computer via the USB when I need to; for instance, send a command to light up a particular LED with a particular intensity. If I want to light the LED on the fourth output at 100/127, I send:

[PRE20]

Arduino *needs* to understand this message. Without going further with the communication protocol design, as that will be covered in [Chapter 7](ch07.html "Chapter 7. Talking over Serial"), *Talking Over Serial*, I want to introduce you to a new feature—splitting strings.

[PRE21]

This piece of code splits the message received by Arduino into two parts.

### Note

The `substring()` function extracts a part of a string from a start index (included) to another (not included).

**Syntax**: `string.substring(from, to);`

`from` is the start index. The result includes the content of the `from` string element. `to` is the end index. The result doesn't include the content of the `end` string element, it can be omitted.

**Returns type**: `String`

Let's push the concept of string extract and split it a bit further.

#### Splitting a string using a separator

Let's challenge ourselves a bit. Imagine I don't know or I'm not sure about the message format (two characters, a dot, and three characters, that we have just seen). This is a real life case; while learning to make things, we often meet strange cases where those *things* don't behave as expected.

Imagine I want to use the dot as a separator, because I'm very sure about it. How can I do that using the things that we have already learned? I'd need to extract characters. OK, I know `substring()` now!

But I also need an index to extract the content at a particular place. I also know how to find the index of an occurrence of a character in a string, using `indexOf()`.

Here is how we do that:

[PRE22]

Firstly, I find the split point index (the place in the string where the dot sits). Secondly, I use this result as the last element of my extracted substring. Don't worry, the last element isn't included, which means `currentOutputNumber` doesn't contain the dot.

At last, I'm using `splitPointIndex` one more time as the start of the second part of the string that I need to extract. And what? I add the integer `1` to it because, as you master `substring()` now and know, the element corresponding to the start index is always included by the `substring()` operation. We don't want that dot because it is only a separator. Right?

Don't worry if you are a bit lost. Things will become clearer in the next subchapters and especially when we'll make Arduino process things, which will come a bit later in the book.

#### Replacement

Replacements are often used when we want to convert a communication protocol to another. For instance, we need to replace a part of a string by another to prepare a further process.

Let's take our previous example. We now want to replace the dot by another character because we want to send the result to another process that only understands the space character as a separator.

[PRE23]

Firstly, I put the content of the `receivedMessage` variable into another variable named `originalMessage` because I know the `replace()` function will definitely modify the processed string. Then I process `receivedMessage` with the `replace()` function.

### Note

The `replace()` function replaces a part of a string with another string.

**Syntax**: `string.replace(substringToReplace, replacingSubstring);`

`from` is the start index. The result includes the content of a `from` string element. `to` is the end index. The result doesn't include the content of an `end` string element, it can be omitted. Remember that, the previous content of the string is overwritten as a result of the replacement (copy it to another string variable if you want to keep it).

**Returns type**: `int` (the function returns `1` if the concatenation happens correctly).

This function can, obviously, replace a character by another character of course. A string is an array of characters. It is not strange that one character can be processed as a string with only one element. Let's think about it a bit.

## Other string functions

There are some other string processing functions I'd like to quickly quote here.

### toCharArray()

This function copies all the string's characters into a "real" character array, also named, for internal reasons, a buffer. You can check [http://arduino.cc/en/Reference/StringToCharArray](http://arduino.cc/en/Reference/StringToCharArray).

### toLowerCase() and toUpperCase()

These functions replace the strings processed by them by the same string but with all characters in lowercase and uppercase respectively. You can check [http://arduino.cc/en/Reference/StringToLower](http://arduino.cc/en/Reference/StringToLower) and [http://arduino.cc/en/Reference/StringToUpperCase](http://arduino.cc/en/Reference/StringToUpperCase). Be careful, as it overwrites the string processed with the result of this process.

### trim()

This function removes all whitespace in your string. You can check [http://arduino.cc/en/Reference/StringTrim](http://arduino.cc/en/Reference/StringTrim). Again, be careful, as it overwrites the strings processed with the result of this process.

### length()

I wanted to end with this functioin. This is the one you'll use a lot. It provides the length of a string as an integer. You can check [http://arduino.cc/en/Reference/StringLength](http://arduino.cc/en/Reference/StringLength).

## Testing variables on the board

The following is a piece of code that you can also find in the folder `Chapter03/VariablesVariations/`:

[PRE24]

Upload this code to your board, then switch on the serial monitor. At last, reset the board by pushing the reset button and observe. The board writes directly to your serial monitor as shown in the following screenshot:

![Testing variables on the board](img/7584_03_002.jpg)

The serial monitor showing you what your board is saying

### Some explanations

All explanations will come progressively, but here is a small summary of what is happening right now.

I first declare my variables and then define some in `setup()`. I could have declared and defined them at the same time.

Refreshing your memory, `setup()` is executed only one time at the board startup. Then, the `loop()` function is executed infinitely, sequentially running each row of statement.

In `loop()`, I'm first testing `myBoolean`, introducing the `if()` conditional statement. We'll learn this in this chapter too.

Then, I'll play a bit with the `char`, `int`, and `String` types, printing some variables, then modifying them and reprinting them.

The main point to note here is the `if()` and `else` structure. Look at it, then relax, answers will come very soon.

# The scope concept

The scope can be defined as a particular property of a variable (and functions, as we'll see further). Considering the source code, the scope of a variable is that part of the code where this variable is visible and usable.

A variable can be *global* and then is visible and usable everywhere in the source code. But a variable can also be *local*, declared inside a function, for instance, and that is visible only inside this particular function.

The scope property is *implicitly* set by the place of the variable's declaration in the code. You probably just understood that every variable could be declared globally. Usually, I follow my own *digital haiku*.

### Note

Let each part of your code know only variables that it has to know, no more.

Trying to minimize the scope of the variables is definitely a winning way. Check out the following example:

[PRE25]

We could represent the code's scope as a box more or less imbricated.

![The scope concept](img/7584_03_03.jpg)

Code's scope seen as boxes

The external box represents the source code's highest level of scope. Everything declared at this level is visible and usable by all functions; it is the global level.

Every other box represents a particular scope in itself. Every variable declared in one scope cannot be seen and used in higher scopes neither in the same level ones.

This representation is very useful to my students who always need more visuals. We'll also use this metaphor while we talk about *libraries*, especially. What is declared in libraries can be used in our code if we include some specific headers at the beginning of the code, of course.

# static, volatile, and const qualifiers

**Qualifiers** are the keywords that are used to change the processor's behavior considering the *qualified* variable. In reality, the compiler will use these qualifiers to change characteristics of the considered variables in the binary firmware produced. We are going to learn about three qualifiers: `static`, `volatile`, and `const`.

## static

When you use the `static` qualifier for a variable inside a function, this makes the variable persistent between two calls of the function. Declaring a variable inside a function makes the variable, implicitly, local to the function as we just learned. It means only the function can know and use the variable. For instance:

[PRE26]

This variable is seen in the `myFunction` function only. But what happens after the first loop? The previous value is lost and as soon as `int aLocalVariable;` is executed, a new variable is set up, with a value of zero. Check out this new piece of code:

[PRE27]

This variable is seen in the `myFunction` function only and, after adding an argument has modified it, we can play with its new value.

In this case, the variable is qualified as `static`. It means the variable is declared *only* the first time. This provides a useful way to keep trace of something and, at the same time, make the variable, containing this trace, local.

## volatile

When you use the `volatile` qualifier in a variable declaration statement, this variable is loaded from the RAM instead of the storage register memory space of the board. The difference is subtle and this qualifier is used in specific cases where your code itself doesn't have the control of something else executed on the processor. One example, among others, is the use of interrupts. We'll see that a bit later.

Basically, your code runs normally, and some instructions are triggered not by this code, but by another process such as an external event. Indeed, our code doesn't know when and what **Interrupt Service Routine** (**ISR**) does, but it stops when something like that occurs, letting the CPU run ISR, then it continues. Loading the variable from the RAM prevents some possible inconsistencies of variable value.

## const

The `const` qualifier means constant. Qualifying a variable with `const` makes it unvariable, which can sound weird.

If you try to write a value to a `const` variable after its declaration/definition statement, the compiler gives an error. The scope's concept applies here too; we can qualify a variable declared inside a function, or globally. This statement defines and declares the `masterMidiChannel` variable as a constant:

[PRE28]

This is equivalent to:

[PRE29]

### Note

There is *no* semicolon after a `#define` statement.

`#define` seems a bit less used as `const`, probably because it cannot be used for constant arrays. Whatever the case, `const` can always be used. Now, let's move on and learn some new operators.

# Operators, operator structures, and precedence

We have already met a lot of operators. Let's first check the arithmetic operators.

## Arithmetic operators and types

Arithmetic operators are:

*   `+` (plus sign)
*   `-` (minus)
*   `*` (asterisk)
*   `/` (slash)
*   `%` (percent)
*   `=` (equal)

I'm beginning with the last one: `=` **.** It is the **assignment** operator. We have already used it a lot to define a variable, which just means to assign a value to it. For instance:

[PRE30]

For the other operators, I'm going to distinguish two different cases in the following: character types, which include `char` and `String`, and numerical types. Operators can change their effect a bit according to the types of variables.

### Character types

`char` and `String` can only be processed by `+`. As you may have guessed, `+` is the concatenation operator:

[PRE31]

In this code, concatenation of `myResultString` and `myString` results in the `Hello World` string.

### Numerical types

With all numerical types (`int`, `word`, `long`, `float`, `double`), you can use the following operators:

*   `+` (addition)
*   `-` (subtraction)
*   `*` (multiplication)
*   `/` (division)
*   `%` (modulo)

A basic example of multiplication is shown as follows:

[PRE32]

### Note

As soon as you use a `float` or `double` type as one of the operand, the floating point calculation process will be used.

In the previous code, the result of `OutputOscillatorAmplitude * multiplier` is a `float` value. Of course, division by zero is *prohibited*; the reason is math instead of C or Arduino.

**Modulo** is simply the remainder of the division of one integer by another one. We'll use it a lot to keep variables into a controlled and chosen range. If you make a variable grow to infinite but manipulate its modulo by 7 for instance, the result will always be between 0 (when the growing variable will be a multiple of 7) and 6, constraining the growing variable.

## Condensed notations and precedence

As you may have noticed, there is a condensed way of writing an operation with these previously explained operators. Let's see two equivalent notations and explain this.

Example 1:

[PRE33]

Example 2:

[PRE34]

These two pieces of code are equivalent. The first one teaches you about the precedence of operators. There is a table given in *Appendix B, Operator Precedence in C and C++* with all precedencies. Let's learn some right now.

`+`, `-`, `*`, `/`, and `%` have a greater precedence over `=`. That means `myInt1 + myInt2` is calculated before the assignment operator, then, the result is assigned to `myInt1`.

The second one is the condensed version. It is equivalent to the first version and thus, precedence applies here too. A little tricky example is shown as follows:

[PRE35]

You need to know that `+` has a higher precedence over `+=`. It means the order of operations is: first, `myInt2 + myInt2` then `myInt1 +` the result of the freshly made calculation `myInt2 + myInt2`. Then, the result of the second is assigned to `myInt1`. This means it is equivalent to:

[PRE36]

## Increment and decrement operators

I want to point you to another condensed notation you'll meet often: the double operator.

[PRE37]

`++` is equivalent to `+=1`, `--` is equivalent to `-=1`. These are called *suffix increment* (`++`) and *suffix decrement* (`--`). They can also be used as *prefix*. `++` and `--` as prefixes have lower precedencies than their equivalent used as suffix but in both cases, the precedence is very much higher than `+`, `-`, `/`, `*`, and even `=` and `+=`.

The following is a condensed table I can give you with the most used cases. In each group, the operators have the same precedence. It drives the expression `myInt++ + 3` to be ambiguous. Here, the use of parenthesis helps to define which calculation will be made first.

| Precedencies groups | Operators | Names |
| --- | --- | --- |
| 2 | `++``--``()``[]` | Suffix incrementSuffix decrementFunction callArray element access |
| 3 | `++``--` | Prefix incrementPrefix decrement |
| 5 | `*``/``%` | MultiplicationDivisionModulo |
| 6 | `+``-` | AdditionSubtraction |
| 16 | `=``+=``-=``*=``/=``%=` | AssignmentAssignment by sumAssignment by differenceAssignment by productAssignment by quotientAssignment by remainder |

I guess you begin to feel a bit better with operators, right? Let's continue with a very important step: types conversion.

# Types manipulations

When you design a program, there is an important step consisting of choosing the right type for each variable.

## Choosing the right type

Sometimes, the choice is constrained by external factors. This happens when, for instance, you use the Arduino with an external sensor able to send data coded as integers in 10 bits (210 = 1024 steps of resolution). Would you choose `byte` type knowing it only provides a way to store number from 0 to 255? Probably not! You'll choose `int`.

Sometimes you have to choose it yourself. Imagine you have data coming to the board from a Max 6 framework patch on the computer via your serial connection (using USB). Because it is the most convenient, since you designed it like that, the patch pops out `float` numbers encapsulated into string messages to the board. After having parsed, cut those messages into pieces to extract the information you need (which is the `float` part), would you choose to store it into `int`?

That one is a bit more difficult to answer. It involves a *conversion* process.

## Implicit and explicit types conversions

Type conversion is the process that changes an entity data type into another. Please notice I didn't talk about variable, but entity.

It is a consequence of C design that we can convert only the values stored in variables, others keep their type until their lives end, which is when the program's execution ends.

Type conversion can be *implicitly* done or *explicitly* made. To be sure everyone is with me here, I'll state that *implicitly means not visibly and consciously written*, compared to *explicitly that means specifically written in code*, here.

### Implicit type conversion

Sometimes, it is also called *coercion*. This happens when you don't specify anything for the compiler that has to make an automatic conversion following its own basic (but often smart enough) rules. The classic example is the conversion of a `float` value into an `int` value.

[PRE38]

I'm using the assignment operator (`=`) to put the content of `myFloat` into `myInt`. It causes **truncation** of the `float` value, that is, the *removal of the decimal part*. You have definitely lost something if you continue to work only with the `myInt` variable instead of `myFloat`. It can be okay, but you have to keep it in mind.

Another less classic example is the implicit conversion of `int` type to `float`. `int` doesn't have a decimal part. The implicit conversion to `float` won't produce something other than a decimal part that equals zero. This is the easy part.

But be careful, you could be surprised by the implicit conversion of `int` to `float`. Integers are encoded over 32 bits, but `float`, even if they are 32 bits, have a *significand* (also called mantissa) encoded over 23 bits. If you don't remember this precisely, it is okay. But I want you to remember this example:

[PRE39]

The output of the code is shown as follows:

![Implicit type conversion](img/7584_03_004.jpg)

Strange results from int to float implicit conversion

I stored `123456789` into a `long int` type, which is totally legal (`long int` are 32-bits signed integers that are able to store integersfrom `-2147483648` to `2147483647`). After the assignment, I'm displaying the result that is: **123456792.00**. We expected `123456789.00` of course.

### Note

Implicit types conversions rules:

*   `long int` to `float` can cause wrong results
*   `float` to `int` removes the decimal part
*   `double` to `float` rounds digit of double
*   `long int` to `int` drops the encoded higher bits

### Explicit type conversion

If you want to have predictable results, every time you can convert types explicitly. There are six conversion functions included in the Arduino core:

*   char()
*   int()
*   float()
*   word()
*   byte()
*   long()

We can use them by passing the variable you want to convert as an argument of the function. For instance, `myFloat = float(myInt)`; where `myFloat` is a `float` type and `myInt` is an `int` type. Don't worry about the use, we'll use them a bit later in our firmware.

### Note

My rule about conversion: Take care of each type conversion you make. None should be obvious for you and it can cause an error in your logic, even if the syntax is totally correct.

# Comparing values and Boolean operators

We now know how to store entities into variables, convert values, and choose the right conversion method. We are now going to learn how to compare variable values.

## Comparison expressions

There are six comparison operators:

*   `==` (equal)
*   `!=` (not equal)
*   `<` (less than)
*   `>` (greater than)
*   `<=` (less than or equal to)
*   `>=` (greater than or equal to)

The following is a comparison expression in code:

[PRE40]

An expression like that does nothing, but it is legal. Comparing two elements produces a result and in this small example, it isn't used to trigger or make anything. `myInt1 > myFloat` is a comparison expression. The result is, obviously, `true` or `false`, I mean it is a `boolean` value. Here it is `false` because `4` is not greater than `5.76`. We can also combine comparison expressions together to create more complex expressions.

## Combining comparisons with Boolean operators

There are three Boolean operators:

*   `&&` (and)
*   `||` (or)
*   `!` (not)

It is time to remember some logic operations using three small tables. You can read those tables like column element + comparison operator + row element; the result of the operation is at the intersection of the column and the row.

The binary operator AND, also written as `&&`:

| `&&` | true | false |
| true | true | false |
| false | false | false |

Then the binary operator OR, also written as ||:

| `&#124;&#124;` | true | false |
| true | true | true |
| false | true | false |

Lastly, the unary operator NOT, also written as `!`:

|   | true | false |
| `!` | false | true |

For instance, true `&&` false = false, false `||` true = true. `&&` and `||` are *binary operators*, they can *compare* two expressions.

`!` is a *unary operator* and can only work with one expression, negating it logically. `&&` is the logical AND. It is true when both expressions compared are true, false in all other cases. `||` is the logic OR. It is true when one expression at least is true, false when they are both false. It is the inclusive OR. `!` is the negation operator, the NOT. It basically inverts false and true into true and false.

These different operations are really useful and necessary when you want to carry out some tests in your code. For instance, if you want to compare a variable to a specific value.

### Combining negation and comparisons

Considering two expressions A and B:

*   NOT(A `&&` B) = (NOT A `||` NOT B)
*   NOT (A `||` B) = (NOT A `&&` NOT B)

This can be more than useful when you'll create conditions in your code. For instance, let's consider two expressions with four variables a, b, c, and d:

*   a `<` b
*   c `>=` d

What is the meaning of `!`(a `<` b)? It is the negation of the expression, where:

`!`(a `<` b) equals (a `>=` b)

The opposite of *a strictly smaller than b* is *a greater than or equal to b*. In the same way:

`!`(c `>=` d) equals (c `<` d)

Now, let's combine a bit. Let's negate the global expression:

(a `<` b) `&&` (c `>=` d) and `!`((a `<` b) `&&` (c `>=` d)) equals (`!`(a `<` b) `||` `!`(c `>=` d)) equals (a `>=` b) `||` (c `<` d)

Here is another example of combination introducing the *operators precedence* concept:

[PRE41]

Both of my statements are equivalent. Precedence occurs here and we can now add these operators to the previous precedencies table (check *Appendix B, Operator Precedence in C and C++*). I'm adding the comparison operator:

| Precedencies groups | Operators | Names |
| --- | --- | --- |
| 2 | `++``--``()``[]` | Suffix incrementSuffix decrementFunction callArray element access |
| 3 | `++``--` | Prefix incrementPrefix decrement |
| 5 | `*``/``%` | MultiplicationDivisionModulo |
| 6 | `+``-` | AdditionSubtraction |
| 8 | `<``<=``>``>=` | Less thanLess than or equal toGreater thanGreater than or equal to |
| 9 | `==``!=` | Equal toNot equal to |
| 13 | `&&` | Logical AND |
| 14 | `&#124;&#124;` | Logical OR |
| 15 | `?:` | Ternary conditional |
| 16 | `=``+=``-=``*=``/=``%=` | AssignmentAssignment by sumAssignment by differenceAssignment by productAssignment by quotientAssignment by remainder |

As usual, I cheated a bit and added the precedence group 15 that contains a unique operator, the ternary conditional operator that we will see a bit later. Let's move to conditional structures.

# Adding conditions in the code

Because I studied Biology and have a Master's diploma, I'm familiar with organic and living behaviors. I like to tell my students that the code, especially in interaction design fields of work, has to be alive. With Arduino, we often build machines that are able to "feel" the real world and interact with it by *acting* on it. This couldn't be done without *condition* statements. This type of statement is called a control structure. We used one conditional structure while we tested our big code including variables display and more.

## if and else conditional structure

This is the one we used without explaining. You just learned patience and zen. Things begin to come, right? Now, let's explain it. This structure is very intuitive because it is very similar to any conditional pseudo code. Here is one:

If the value of the variable `a` is smaller than the value of variable `b`, switch on the LED. Else switch it off.

Now the real C code, where I simplify the part about the LED by giving a state of 1 or 0 depending on what I want to further do in the code:

[PRE42]

I guess it is clear enough. Here is the general syntax of this structure:

[PRE43]

Expression evaluation generally results in a Boolean value. But numerical values as a result of an expression in this structure can be okay too, even if a bit less explicit, which I, personally, don't like. An expression resulting in the numerical value `0` equals `false` in C in Arduino core, and equals `true` for any other values.

### Note

Being implicit often means making your code shorter and cuter. In my humble opinion, it also means to be very unhappy when you have to support and maintain a code several months later when it includes a bunch of implicit things without any comments.

I push my students to be explicit and verbose. We are not here to code things to a too small amount of memory, believe me. We are not talking about reducing a 3 megabytes code to 500 kilobytes but more about reducing a 200 kilobytes code to 198 kilobytes.

### Chaining an if…else structure to another if…else structure

The following is the modified example:

[PRE44]

The first `if` test is: if `a` is smaller than `b`. If it is `true`, we put the value `1` inside the variable `ledState`. If it is `false`, we go to the next statement `else`.

This `else` contains another test on `b`: is `b` greater than `0`? If it is, we put the value `0` inside the variable `ledState`. If it is `false`, we can go to the last case, the last `else`, and we put the value `1` inside the variable `ledState`.

### Tip

**One frequent error – missing some cases**

Sometimes, the `if`… `else` chain is so complicated and long that we may miss some case and no case is verified. Be clear and try to check the whole universe of cases and to code the conditions according to it.

A nice tip is to try to put all cases on paper and try to find the *holes*. I mean, where the part of the variable values are not matched by tests.

### if…else structure with combined comparisons expressions

The following is the previous example where I commented a bit more:

[PRE45]

We can also write it in the following way considering the comment I wrote previously in the code:

[PRE46]

It could be considered as a more condensed version where you have all statements for the switch on the LED in one place, same for switching it off.

### Finding all cases for a conditional structure

Suppose you want to test a temperature value. You have two specific limits/points at which you want the Arduino to react and, for instance, alert you by lighting an LED or whatever event to interact with the real world. For instance, the two limits are: 15-degree Celsius and 30-degree Celsius. How to be sure I have all my cases? The best way is to use a pen, a paper, and to draw a bit.

![Finding all cases for a conditional structure](img/7584_03_05.jpg)

Checking all possible T values

We have three parts:

*   T < 15
*   T > 15 but T <30
*   T > 30

So we have three cases:

*   T< 15
*   T >15 and T < 30
*   T > 30

What happens when T = 30 or T = 15? These are holes in our logic. Depending on how we designed our code, it could happen. Matching all cases would mean: include T = 15 and T = 30 cases too. We can do that as follows:

[PRE47]

I included these two cases into my comparisons. 15-degree Celsius is included in the second temperature interval and 30-degree Celsius in the last one. This is an example of how we can do it.

I'd like you to remember to use a pen and a paper in this kind of cases. This will help you to design and especially make some breaks from the IDE that is, in designing steps, really good. Let's now explore a new conditional structure.

## switch…case…break conditional structure

Here, we are going to see a new conditional structure. The standard syntax is shown as follows:

[PRE48]

`var` is compared for equality to each case label. If `var` equals a particular `label` value, the statements in this case are executed until the next `break`. If there is no match and you have used the optional `default:` case, the statements of this case are executed. Without the `default:` case, nothing is done. `label` must be a value, not a character, not string. Let's take a more concrete example:

[PRE49]

This code is equivalent to:

[PRE50]

Are you okay?

### Note

What I want to say is, when you want to compare a variable to many unique values, use `switch`…`case`…`break`, else use `if`…`else`.

When you have comparison intervals, `if`…`else` is more convenient because you can use `<` and `>` whereas in `switch`…`case`…`break` you cannot. Of course, we could combine both. But remember to keep your code as simple as you can.

## Ternary operator

This strange notation is often totally unknown to my students. I used to say, "Hey! This is more C than Arduino" when they answer "That is why we have forgotten about it". Naughty students! This ternary operator takes three elements as input. The syntax is `(expression) ? val1 : val2`.

The expression is tested. If it is `true`, this whole statement returns (or equals) `val1`, if it is `false`, it equals `val2`.

Again imagine our Arduino, the temperature sensor, and only one limit which is 20 degree Celsius. I want to turn the LED blue if `T` is smaller than the limit, and red if `T` is greater or equal to 20 degree Celsius. Here is how we would use the two ternary operators:

[PRE51]

It can be a nice notation, especially if you don't need statement execution in each case but only variable assignments.

# Making smart loops for repetitive tasks

A **loop** is a series of events repeating themselves in time. Basically, computers have been designed, at first, to make a lot of calculations repeatedly to save human's time. Designing a loop to repeat tasks that have to be repeated seems a natural idea. C natively implements some ways to design loops. Arduino core naturally includes three loop structures:

*   `for`
*   `while`
*   `do`…`while`

## for loop structure

The `for` loop statement is quite easy to use. It is based on, at least, one counter starting from a particular value you define, and increments or decrements it until another defined value. The syntax is:

[PRE52]

The counter is also named `index`. I'm showing you a real example here:

[PRE53]

This basic example defines a loop that prints all integers from `0` to `99`. The declaration/definition of the integer type variable `i` is the first element of the `for` structure. Then, the condition describes in which case the statements included in this loop have to be executed. At last, the `i++` increment occurs.

Pay attention to the increment element. It is defined with the increment as a suffix. It means here that the increment occurs after the end of the execution of the statements for a considered `i` value.

Let's break the loop for the first two and last two `i` values and see what happens. Declaration of the integer variable `i` for the first and second iteration is shown as follows:

*   `i = 0`, is `i` smaller than `100`? yes, `println(0)`, increment `i`
*   `i = 1`, is `i` smaller than `100`? yes, `println(1)`, increment `i`

For the last two iterations the value of `i` is shown as follows:

*   `i = 99`, is `i` smaller than `100`? yes, `println(99)`, increment `i`
*   `i = 100`, is `i` smaller than `100`? no, stop the loop

Of course, the index could be declared before the `for` structure, and only defined inside the `for` structure. We could also have declared and defined the variable before and we would have:

[PRE54]

This seems a bit strange, but totally legal in C and for the Arduino core too.

### Tip

**The scope of index**

If the index has been declared inside the `for` loop parenthesis, its scope is only the `for` loop. This means that this variable is *not* known or *not* usable outside of the loop.

It normally works like that for any variable declared inside the statements of a `for` loop. This isn't something to do, even if it is totally legal in C. Why not? Because it would mean you'd declare a variable each time the loop runs, which isn't really smart. It is better to declare it outside of the loop, one time, then to use it inside of it, whatever the purpose (index, or variable to work with inside statements).

### Playing with increment

Increment can be something more complex than only using the increment operator.

#### More complex increments

First, instead of writing `i++`, we could have written `i = i + 1`. We can also use other kind of operations like subtraction, multiplication, division, modulo, or combinations. Imagine that you want to print only odd numbers. Odd numbers are all of the form 2n + 1 where *n* is an integer. Here is the code to print odd numbers from 1 to 99:

[PRE55]

First values of `i` are: `1`, `3`, `5`, and so on.

#### Decrements are negative increments

I just want to remix the previous code into something else in order to shake your mind a bit around increments and decrements. Here is another code making the same thing but printing odd numbers from `99` to `1`:

[PRE56]

All right? Let's complicate things a bit.

### Using imbricated for loops or two indexes

It is also possible to use more than one index in a `for` structure. Imagine we want to calculate a multiplication table until 10 x 10\. We have to define two integer variables from 1 to 10 (0 being trivial). These two indexes have to vary from 1 to 10\. We can begin by one loop with the index `x`:

[PRE57]

This is for the first index. The second one is totally similar:

[PRE58]

How can I mix those? The answer is the same as the answer to the question: what is a multiplication table? I have to keep one index constant, and multiply it by the other one going from 1 to 10\. Then, I have to increment the first one and continue doing the same with the other and so on. Here is how to we do it:

[PRE59]

This code prints all results of `x*y` where `x` and `y` are integers from 1 to 10, one result on each line. Here are the first few steps:

*   `x = 1`, `y = 1`… print the result
*   `x = 1`, `y = 2`… print the result
*   `x = 1`, `y = 3`… print the result

`x` is incremented to `2` each time the inside `for` loop (the one with `y`) ends, then `x` is fixed to `2` and `y` grows until `x = 10` and `y = 10` where the `for` loop ends.

Let's improve it a bit, only for aesthetic purposes. It is also a pretext to tweak and play with the code to make you more comfortable with it. Often, multiplication tables are drawn as follows:

![Using imbricated for loops or two indexes](img/7584_03_006.jpg)

Classic view of a multiplication table

We need to go to the next line each time one of the index (and only one) reaches the limit which is the value `10`.

[PRE60]

Check the code, each time `y` reaches `10`, a new line is created. The `for` loop is a powerful structure to repeat tasks. Let's check another structure.

## while loop structure

The `while` loop structure is a bit simpler. Here is the syntax:

[PRE61]

The expression is evaluated as a Boolean, `true` or `false`. While the expression is `true`, statements are executed, then as soon as it will be `false`, the loop will end. It obviously, often, requires declaration and definition outside of the `while` structure. Here is an example doing the same results than our first `for` loop printing all integers from 0 to 99:

[PRE62]

Indeed, you *have* to take care of the increment or decrement explicitly inside your statements; I'll say a few words on infinite loops a bit later. We could have condensed the code a bit more by doing that:

[PRE63]

The `while` loop structure tests the expression before doing even executing the first statement. Let's check a similar structure doing that differently.

## do…while loop structure

The `do`…`while` loop structure is very similar to the `while` structure, but makes its expression evaluation at the end of the loop, which means after the statements execution. Here is the syntax:

[PRE64]

Here is an example on the same model:

[PRE65]

It means that even if the first result of the expression evaluation is `false`, the statements will be executed on time. This is not the case with the `while` structure.

## Breaking the loops

We learned how to create loops driven by indexes that define precisely how these loops will live. But how can we stop a loop when an *external* event occurs? External is taken in the sense of external to the loop itself including its indexes. In that case, the loop's condition itself wouldn't include the external element.

Imagine that we have a process running 100 times in *normal* conditions. But we want to interrupt it, or modify it according to another variable that has a greater scope (declared outside of the loop, at least).

Thanks to the `break` statement for making that possible for us. `break;` is the basic syntax. When `break` is executed, it exits the current loop, whatever it is, based on: `do`, `for`, and `while`. You already saw `break` when we talked about the `switch` conditional structure. Let's illustrate that.

Imagine a LED. We want its intensity to grow from 0 to 100 percent then to go back to 0, every time. But we also want to use a nice distance sensor that resets this loop each time the distance between a user and the sensor is greater than a value.

### Note

It is based on a real installation I made for a museum where a system has to make a LED blink smoothly when the user was far and to switch off the LED when the user was near, like a living system calling for users to meet it.

I designed it very simply as follows:

[PRE66]

This whole loop was included inside the global `loop()` function in the Arduino board and the complete test about the distance was executed each time the `loop()` function occurs, waiting for users.

## Infinite loops are not your friends

Be careful of infinite loops. The problem isn't really the infinite state of loops, but the fact that a system, whatever it is including Arduino, which is running an infinite loop does only that! Nothing that is after the loop can be executed because the program won't go outside the loop.

If you understand me correctly, `loop()`—the basic Arduino core function—is an infinite loop. But it is a controlled loop well designed and Arduino core based. It can (and is) interrupted when functions are called or other special events occur, letting us, users, design what we need inside of this loop. I used to call "the event's driver and listener" because it is the place where our main program runs.

There are many ways to create infinitely looped processes. You can define a variable in `setup()`, making it grow in `loop()` and test it each time `loop()` runs in order to reset it to the initial value, for instance. It takes benefits of the already existing `loop()` loop. Here is this example in C for Arduino:

[PRE67]

This `i` grows from `0` to `threshold – 1` then goes back to `0`, grows again, infinitely, taking benefits of `loop()`.

There are also other ways to run loops infinitely in a controlled manner that we'll see a bit later in the more advanced part of the book, but you have been warned: take care of those infinite loops.

# Summary

We learned a lot of abstract things in this important chapter. From type to operator's precedencies, to conditional structure, now we are going to learn new structures and syntaxes that will help us make more efficient blocks of code and, especially, more reusable ones. We can now learn about functions. Let's dive into the next C/C++ knowledge chapters and we will be able to test our Arduino after that.