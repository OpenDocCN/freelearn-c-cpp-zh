# Chapter 3. If, Else, and Switch

In the previous chapter, we discussed the importance of memory and how it can be used to store data inside a computer. We spoke about how memory is reserved for your program using variables, and how we can include different types of information in our variables.

In this chapter, we will talk about how to control the flow of our program and how we can change what code gets executed by branching the code using control flow statements. Here, we'll discuss the different types of control flow, as follows:

*   If statements
*   How to check whether things are equal using the `==` operator
*   Else statements
*   How to test for inequalities (that is, how to check whether one number is greater or smaller than another using the operators >, >=, <, <=, and !=)
*   Using logical operators (such as not (!), and (&&), or (||))
*   Our first example project with Unreal Engine
*   Branching in more than two ways:

    *   The else if statement
    *   The switch statement

# Branching

The computer code we wrote in [Chapter 2](part0022_split_000.html#KVCC2-dd4a3f777fc247568443d5ffb917736d "Chapter 2. Variables and Memory"), *Variables and Memory* went in one direction: straight down. Sometimes, we might want to be able to skip parts of the code. We might want the code to be able to branch in more than one direction. Schematically, we can represent this in the following manner:

![Branching](img/00035.jpeg)

A flowchart

In other words, we want the option to not run certain lines of code under certain conditions. The preceding figure is called a flowchart. According to this flowchart, if and only if we are hungry, then we will go prepare a sandwich, eat it, and then go and rest on the couch. If we are not hungry, then there is no need to make a sandwich, so we will simply rest on the couch.

We'll use flowcharts in this book only sometimes, but in UE4, you can even use flowcharts to program your game (using something called blueprints).

### Note

This book is about C++ code, so we will always transform our flowcharts into actual C++ code in this book.

# Controlling the flow of your program

Ultimately, what we want is the code to branch in one way under certain conditions. Code commands that change which line of code gets executed next are called control flow statements. The most basic control flow statement is the `if` statement. To be able to code `if` statements, we first need a way to check the value of a variable.

So, to start, let's introduce the `==` symbol, which is used to check the value of a variable.

## The == operator

In order to check whether two things are equal in C++, we need to use not one but two equal signs (`==`) one after the other, as shown here:

[PRE0]

If you run the preceding code, you will notice that the output is this:

[PRE1]

In C++, 1 means true, and 0 means false. If you want the words true or false to appear instead of 1 and 0, you can use the `boolalpha` stream manipulator in the `cout` line of code, as shown here:

[PRE2]

The `==` operator is a type of comparison operator. The reason why C++ uses `==` to check for equality and not just `=` is that we already used up the `=` symbol for the assignment operator! (see the *More on variables* section in [Chapter 2](part0022_split_000.html#KVCC2-dd4a3f777fc247568443d5ffb917736d "Chapter 2. Variables and Memory"), *Variables and Memory*). If we use a single `=` sign, C++ will assume that we want to overwrite `x` with `y`, not compare them.

## Coding if statements

Now that we have the double equals sign under our belt, let's code the flowchart. The code for the preceding flowchart figure is as follows:

[PRE3]

### Tip

This is the first time we are using a `bool` variable! A `bool` variable either holds the value `true` or the value `false`.

First, we start with a `bool` variable called `isHungry` and just set it to `true`.

Then, we use an `if` statement, as follows:

[PRE4]

The if statement acts like a guard on the block of code below it. (Remember that a block of code is a group of code encased within `{` and `}`.)

![Coding if statements](img/00036.jpeg)

You can only read the code between { and } if `isHungry==true`

You can only get at the code inside the curly braces when `isHungry == true`. Otherwise, you will be denied access and forced to skip over that entire block of code.

### Tip

We an achieve the same effect by simply writing the following line of code:

[PRE5]

This can be used as an alternative for the following:

[PRE6]

The reason people might use the `if( isHungry )` form is to avoid the possibility of making mistakes. Writing `if( isHungry = true )` by accident will set `isHungry` to true every time the `if` statement is hit! To avoid this possibility, we can just write `if( isHungry )` instead. Alternatively, some (wise) people use what are called Yoda conditions to check an if statement: `if( true == isHungry )`. The reason we write the `if` statement in this way is that, if we accidentally write `if( true = isHungry )`, this will generate a compiler error, catching the mistake.

Try to run this code segment to see what I mean:

[PRE7]

The following lines show the output of the preceding lines of code:

[PRE8]

The line of code that has `(x = y)` overwrites the previous value of `x` (which was 4) with the value of `y` (which is 5). Although we were trying to check whether `x` equals `y`, what happened in the previous statement was that `x` was assigned the value of `y`.

## Coding else statements

The `else` statement is used to have our code do something in the case that the `if` portion of the code does not run.

For example, say we have something else that we'd like to do in case we are not hungry, as shown in the following code snippet:

[PRE9]

[PRE10]

There are a few important things that you need to remember about the `else` keyword, as follows:

*   An `else` statement must always immediately follow after an `if` statement. You can't have any extra lines of code between the end of the if block and the corresponding else block.
*   You can never go into both the if and the corresponding else blocks. It's always one or the other.![Coding else statements](img/00037.jpeg)

    The else statement is the way you will go if `isHungry` is not equal to true

You can think of the `if`/`else` statements as a guard diverting people to either the left or the right. Each person will either go towards the food (when `isHungry==true`), or they will go away from the food (when `isHungry==false`).

## Testing for inequalities using other comparison operators (>, >=, <, <=, and !=)

Other logical comparisons can be easily done in C++. The > and < symbols mean just what they do in math. They are the greater than (>) and less than (<) symbols, respectively. >= has the same meaning as the ≥ symbol in math. <= is the C++ code for ≤. Since there isn't a ≤ symbol on the keyboard, we have to write it using two characters in C++. `!=` is how we say "not equal to" in C++. So, for example, say we have the following lines of code:

[PRE11]

We can ask the computer whether `x > y` or `x < y` as shown here:

[PRE12]

### Tip

We need the brackets around the comparisons of x and y because of something known as operator precedence. If we don't have the brackets, C++ will get confused between the << and < operators. It's weird and you will better understand this later, but you need C++ to evaluate the (x < y) comparison before you output the result (<<). There is an excellent table available for reference at [http://en.cppreference.com/w/cpp/language/operator_precedence](http://en.cppreference.com/w/cpp/language/operator_precedence).

# Using logical operators

Logical operators allow you to do more complex checks, rather than checking for a simple equality or inequality. Say, for example, the condition to gain entry into a special room requires the player to have both the red and green keycards. We want to check whether two conditions hold true at the same time. To do this type of complex logic statement checks, there are three additional constructs that we need to learn: the *not* (`!`), *and* (`&&`), and *or* (`||`) operators.

## The Not (!) operator

The `!` operator is handy to reverse the value of a `boolean` variable. Take an example of the following code:

[PRE13]

The `if` statement here checks whether or not you are wearing socks. Then, you are issued a command to get some socks on. The `!` operator reverses the value of whatever is in the `boolean` variable to be the opposite value.

We use something called a truth table to show all the possible results of using the `!` operator on a `boolean` variable, as follows:

| wearingSocks | !wearingSocks |
| --- | --- |
| true | false |
| false | true |

So, when `wearingSocks` has the value true, `!wearingSocks` has the value `false` and vice versa.

### Exercises

1.  What do you think will be the value of `!!wearingSocks` when the value of `wearingSocks` is true?
2.  What is the value of `isVisible` after the following code is run?

[PRE14]

### Solution

1.  If `wearingSocks` is true, then `!wearingSocks` is false. Therefore, `!!wearingSocks` becomes true again. It's like saying *I am not not hungry*. Not not is a double negative, so this sentence means that I am actually hungry.
2.  The answer to the second question is false. `hidden` was true, so `!hidden` is false. false then gets saved into the `isVisible` variable.

### Tip

The `!` operator is sometimes colloquially known as bang. The preceding bang bang operation (`!!`) is a double negative and a double logical inversion. If you bang-bang a `bool` variable, there is no net change to the variable. If you bang-bang an `int` variable, it becomes a simple `bool` variable(`true` or `false`). If the `int` value is greater than zero, it is reduced to a simple `true`. If the `int` value is 0 already, it is reduced to a simple `false`.

## The And (&&) operator

Say, we only want to run a section of the code if two conditions are true. For example, we are only dressed if we are wearing both socks and clothes. You can use the following code to checks this:

[PRE15]

## The Or (||) operator

We sometimes want to run a section of the code if either one of the variables is `true`.

So, for example, say the player wins a certain bonus if he finds either a special star in the level or the time that he takes to complete the level is less than 60 seconds, in which case you can use the following code:

[PRE16]

# Our first example with Unreal Engine

We need to get started with Unreal Engine.

### Tip

A word of warning: when you open your first Unreal project, you will find that the code looks very complicated. Don't get discouraged. Simply focus on the highlighted parts. Throughout your career as a programmer, you will often have to deal with very large code bases containing sections that you do not understand. However, focusing on the parts that you do understand will make this section productive.

Open the **Unreal Engine Launcher** app (which has the blue-colored UE4 icon ![Our first example with Unreal Engine](img/00038.jpeg)). Select **Launch Unreal Engine 4.4.3**, as shown in the following screenshot:

![Our first example with Unreal Engine](img/00039.jpeg)

### Tip

If the **Launch** button is grayed out, you need to go to the **Library** tab and download an engine (~3 GB).

Once the engine is launched (which might take a few seconds), you will be in the **Unreal Project Browser** screen (black-colored UE4 icon ![Our first example with Unreal Engine](img/00040.jpeg)), as shown in the following screenshot.

Now, select the **New Project** tab in the UE4 project browser. Scroll down until you reach **Code Puzzle**. This is one of the simpler projects that doesn't have too much code, so it's good to start with. We'll go to the 3D projects later.

![Our first example with Unreal Engine](img/00041.jpeg)

Here are a few things to make a note of in this screen:

*   Be sure you're in the **New Project** tab
*   When you click on **Code Puzzle**, make sure that it is the one with the **C++** icon at the right, not **Blueprint Puzzle**
*   Enter a name for your project, `Puzzle`, in the **Name** box (this is important for the example code I will give you to work on later)
*   If you want to change the storage folder (to a different drive), click the down arrow so that the folder appears. Then, name the directory where you want to store your project.

After you've done all this, select **Create Project**.

Visual Studio 2013 will open with the code of your project.

Press *Ctrl*+*F5* to build and launch the project.

Once the project compiles and runs, you should see the Unreal Editor, as shown in the following screenshot:

![Our first example with Unreal Engine](img/00042.jpeg)

Looks complicated? Oh boy, it sure is! We'll explore some of the functionality in the toolbars at the side later. For now, just select **Play** (marked in yellow), as shown in the preceding screenshot.

This launches the game. This is how it should look:

![Our first example with Unreal Engine](img/00043.jpeg)

Now, try clicking on the blocks. As soon as you click on a block, it turns orange, and this increases your score.

What we're going to do is find the section that does this and change the behavior a little.

Find and open the `PuzzleBlock.cpp` file.

### Tip

In Visual Studio, the list of files in the project is located inside the **Solution Explorer**. If your **Solution Explorer** is hidden, simply click on **View**/**Solution Explorer** from the menu at the top.

Inside this file, scroll down to the bottom, where you'll find a section that begins with the following words:

[PRE17]

![Our first example with Unreal Engine](img/00044.jpeg)

`APuzzleBlock` is the class name, and `BlockClicked` is the function name. Whenever a puzzle block gets clicked on, the section of code from the starting { to the ending } is run. Hopefully, exactly how this happens will make more sense later.

It's kind of like an `if` statement in a way. If a puzzle piece is clicked on, then this group of the code is run for that puzzle piece.

We're going to walk through the steps to make the blocks flip colors when they are clicked on (so, a second click will change the color of the block from orange back to blue).

Perform the following steps with the utmost care:

1.  Open `PuzzleBlock.h` file. After line 25 (which has this code):

    [PRE18]

    Insert the following code after the preceding lines of code:

    [PRE19]

2.  Now, open `PuzzleBlock.cpp` file. After line 40 (which has this code):

    [PRE20]

    Insert the following code after the preceding lines:

    [PRE21]

3.  Finally, in `PuzzleBlock.cpp`, replace the contents of the `void APuzzleBlock::BlockClicked` section of code (line 44) with the following code:

    [PRE22]

### Tip

Only replace inside the `void APuzzleBlock::BlockClicked (UPrimitiveComponent* ClickedComp)`statement.

Do not replace the line that starts with `void APuzzleBlock::BlockClicked`. You might get an error (if you haven't named your project Puzzle). You've been warned.

So, let's analyze this. This is the first line of code:

[PRE23]

This line of code simply flips the value of `bIsActive`. `bIsActive` is a `bool` variable (it is created in `APuzzleBlock.h`). If `bIsActive` is true, `!bIsActive` will be false. So, whenever this line of code is hit (which happens with a click on any block), the `bIsActive` value is reversed (from `true` to `false` or from `false` to `true`).

Let's consider the next block of code:

[PRE24]

We are simply changing the block color. If `bIsActive` is true, then the block becomes orange. Otherwise, the block turns blue.

## Exercise

By now, you should notice that the best way to get better at programming is by doing it. You have to practice programming a lot to get significantly better at it.

Create two integer variables, called x and y, and read them in from the user. Write an `if`/`else` statement pair that prints the name of the bigger-valued variable.

## Solution

The solution of the preceding exercise is shown in the following block of code:

[PRE25]

### Tip

Don't type a letter when `cin` expects a number. `cin` can fail and give a bad value to your variable if that happens.

## Branching code in more than two ways

In the previous sections, we were only able to make the code branch in one of the two ways. In pseudocode, we had the following code:

[PRE26]

### Tip

Pseudocode is *fake code*. Writing pseudocode is a great way to brainstorm and plan out your code, especially if you are not quite used to C++.

This code is a little bit like a metaphorical fork in the road, with only one of two directions to choose from.

Sometimes, we might want to branch the code in more than just two directions. We might want the code to branch in three ways, or even more. For example, say the direction in which the code goes depends on what item the player is currently holding. The player can be holding one of three different items: a coin, key, or sand dollar. And C++ allows that! In fact, in C++, you can branch in any number of directions as you wish.

## The else if statement

The `else if` statement is a way to code in more than just two possible branch directions. In the following code example, the code will go in one of the three different ways, depending on whether the player is holding the `Coin`, `Key`, or `Sanddollar` objects:

[PRE27]

### Note

Note that the preceding code only goes in one of the three separate ways! In an `if`, `else if`, and `else if` series of checks, we will only ever go into one of the blocks of code.

![The else if statement](img/00045.jpeg)

### Exercise

Use C++ program to answer the questions that follow. Be sure to try these exercises in order to gain fluency with these equality operators.

[PRE28]

Write some new lines of code at the spot that says (`// *** Write new...`):

1.  Check whether `x` and `y` are equal. If they are equal, print `x and y are equal`. Otherwise, print `x and y are not equal`.
2.  An exercise on inequalities: check whether `x` is greater than `y`. If it is, print `x is greater than y`. Otherwise, print `y is greater than x`.

### Solution

To evaluate equality, insert the following code:

[PRE29]

To check which value is greater insert the following code:

[PRE30]

## The switch statement

The `switch` statement allows your code to branch in multiple ways. What the `switch` statement will do is look at the value of a variable, and depending on its value, the code will go in a different direction.

We'll also introduce the `enum` construct here:

[PRE31]

Switches are like coin sorters. When you drop 25 cent into the coin sorter, it finds its way into the 25 cent pile. Similarly, a `switch` statement will simply allow the code to jump down to the appropriate section. The example of sorting the coins is shown in the following figure:

![The switch statement](img/00046.jpeg)

The code inside the `switch` statement will continue to run (line by line) until the `break;` statement is hit. The `break` statement jumps you out of the `switch` statement. Take a look at the following diagram to understand how the switch works:

![The switch statement](img/00047.jpeg)

1.  First, the `Food` variable is inspected. What value does it have? In this case, it has `Fish` inside it.
2.  The `switch` command jumps down to the correct case label. (If there is no matching case label, the switch will just be skipped).
3.  The `cout` statement is run, and `Here fishy fishy fishy` appears on the console.
4.  After inspecting the variable and printing the user response, the break statement is hit. This makes us stop running lines of code in the switch and exit the switch. The next line of code that is run is just what would otherwise have been the next line of code in the program if the switch had not been there at all (after the closing curly brace of the switch statement). It is the print statement at the bottom, which says "End of switch".

### Switch versus if

Switches are like the `if` / `else if` / `else` chains from earlier. However, switches can generate code faster than `if` / `else if` / `else if` / `else` chains. Intuitively, switches only jump to the appropriate section of the code to execute. If / else if / else chains might involve more complicated comparisons (including logical comparisons), which might take more CPU time. The main reason you will use the `if` statements is to do more with your own custom comparisons inside the brackets.

### Tip

An enum is really an int. To verify this, print the following code:

[PRE32]

You will see the integer values of the enum—just so you know.

Sometimes, programmers want to group multiple values under the same switch `case` label. Say, we have an `enum`, object as follows:

[PRE33]

A programmer wants to group all the greens together, so he writes a `switch` statement as follows:

[PRE34]

In this case, `Zucchini` falls through and executes the same code as `Broccoli`. The non-green vegetables are in the `default` case label. To prevent a fall through, you have to remember to insert an explicit `break` statement after each `case` label.

We can write another version of the same switch that does not let Zucchini fall through, by the explicit use of the keyword `break` in the switch:

[PRE35]

Note that it is good programming practice to `break` the `default` case as well, even though it is the last `case` listed.

### Exercise

Complete the following program, which has an `enum` object with a series of mounts to choose from. Write a `switch` statement that prints the following messages for the mount selected:

| Horse | The steed is valiant and mighty |
| Mare | This mare is white and beautiful |
| Mule | You are given a mule to ride. You resent that. |
| Sheep | Baa! The sheep can barely support your weight. |
| Chocobo | Chocobo! |

Remember, an `enum` object is really an `int` statement. The first entry in an `enum` object is by default 0, but you can give the `enum` object any starting value you wish using the = operator. Subsequent values in the `enum` object are `ints` arranged in order.

### Tip

**Bit-shifted enum**

A common thing to do in an `enum` object is to assign a bit-shifted value to each entry:

[PRE36]

The bit-shifted values should be able to combine the window properties. This is how the assignment will look:

[PRE37]

Checking which `WindowProperties` have been set involves a check using `bitwise AND`:

[PRE38]

Bit shifting is a technique that is slightly beyond the scope of this text, but I've included this tip just so you know about it.

### Solution

The solution of the preceding exercise is shown in the following code:

[PRE39]

# Summary

In this chapter, you learned how to branch the code. Branching makes it possible for the code to go in a different direction instead of going straight down.

In the next chapter, we will move on to a different kind of control flow statement that will allow you to go back and repeat a line of code a certain number of times. The sections of code that repeat will be called loops.