# *Chapter 2*: Software Setup and C Programming for Microcontroller Boards

In this chapter, you will review the basic configuration of the IDEs used for programming the Blue Pill and Curiosity Nano microcontroller boards, as well as learn the basics of the C programming language necessary for coding applications for the Blue Pill and the Curiosity Nano. This is by no means a comprehensive C tutorial. It contains important information to understand and complete the exercises explained in all the chapters of this book. In this chapter, we're going to cover the following main topics:

*   Introducing the C programming language
*   Introducing Curiosity Nano microcontroller board programming
*   Introducing Blue Pill microcontroller board programming
*   Example – Programming and using the microcontroller board's internal LED

By the end of this chapter, you will have received a solid introduction to the C programming language, including a set of programming instructions useful for developing many small and mid-sized microcontroller projects with the Blue Pill and Curiosity Nano microcontroller boards. This chapter also covers the use of the internal LED, which both the Blue Pill and the Curiosity Nano have. This can be very useful for quickly showing digital results (for example, confirming actions in your project).

# Technical requirements

The software that we will use in this chapter is the Arduino and MPLAB X IDEs for programming the Blue Pill and the Curiosity Nano, respectively. Their installation process was described in [*Chapter 1*](B16413_01_Final_NM_ePub.xhtml#_idTextAnchor014)*,* *Introduction to Microcontrollers and Microcontroller Boards*. We will also use the same code examples that were used in the aforementioned chapter.

In this chapter, we will also use the following hardware:

*   A solderless breadboard.
*   The Blue Pill and Curiosity Nano microcontroller boards.
*   A micro USB cable for connecting your microcontroller boards to a computer.
*   The ST-LINK/V2 electronic interface needed to upload the compiled code to the Blue Pill. Remember that the ST-Link/V2 requires four female-to-female DuPont wires.

These are fundamental hardware components that will suffice for the examples described in this chapter, and will also prove useful in other more complex projects explained in other chapters.

The code used in this chapter can be found at the book's GitHub repository here:

[https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter02](https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter02)

The Code in Action video for this chapter can be found here: [https://bit.ly/3xwFvPA](https://bit.ly/3xwFvPA)

The next section explains a concise introduction to the C programming language.

# Introducing the C programming language

The **C programming language** was initially created in the early seventies for developing the UNIX operating system, but it has been ported to practically all operating systems ever since. It is a mid-level programming language because it shares properties from high-level languages such as Python and low-level languages, for example, the assembly language. The C language is generally easier to program than low-level languages because it is very *human-readable* and there are many libraries available that facilitate the development of software applications, among other reasons. It is also very efficient for programming embedded systems. C is one of the most popular coding languages, and virtually all microcontrollers can be programmed with C compilers – Blue Pill and Curiosity Nano are no exceptions.

The C language is not completely portable among different families and manufacturers of microcontrollers. For example, the I/O ports and the interrupts are not programmed the same in both Blue Pill and Curiosity Nano. That is why two types of C compilers and different libraries are needed for programming both microcontroller boards. In fact, the Arduino IDE used for programming the Blue Pill uses a variant of C called **C++**. C++ is a powerful extension of the C programming language that incorporates features such as object-oriented and low-memory level programming.

The following section explains the basics of the C language structure. This section includes an explanation of the `#include` directive, writing comments, understanding variables, using constants, a keywords list, declaring functions, evaluating expressions, and writing loops in C.

## The basic structure of the C language

As with other programming languages, C makes it possible to declare program elements such as constants, types, functions, and variables in separate files called `.h`. This can help to organize C instructions and reduce clutter in your main C code. A library is a header file containing program elements (such as functions) that can be shared with other C programmers or constantly used in different C programs. C language compilers contain important libraries that we will use in this book. The header files can be included (that is, linked and compiled) along your main program using the `#include` directive; hence, the programming elements declared in the header file will be called and used in your C program.

There are many useful standard and non-standard libraries. We will review and use both. The `#include` directive is a special instruction for the C compiler and not a regular C instruction. It should be written at the beginning of the program and without a semicolon at the end. Only the C statements have a semicolon at the end. There are three ways to write and apply the `#include` directive. These are as follows:

*   `#include <file_name.h>`: This type of directive uses the less than and greater than symbols, meaning that the header file (`.h`) is placed in the compiler path. You don't need to write the complete path to the header file.
*   `#include "file_name.h"`: This type of directive uses double quotes. The header file is stored in the project's directory.
*   `#include "sub_directory_name/file_name.h"`: This directive type tells the compiler that the header file is placed in a sub-directory. Please note that the slash symbol is applied depending on the operating system that you are using. For example, Windows computers use a backslash (*\*) symbol as a directory separator. Linux and Mac computers use the forward-slash (*/*) symbol.

The next sub-section shows how to define and use header files.

### Example of the #include directive

The following program example shows how to include a header file that is placed in the project's directory:

```cpp
#include "main_file.h"
int main(void)
{
    x = 1;
    y = 2;
    z = x+y;
}
```

In the preceding example, the `x`, `y`, and `z` variables were declared in the `main_file.h` header file, so they are not declared in the main program. The header file (`file.h`) contains the following code declaring the three variables used in the main code:

```cpp
int x;
int y;
int z;
```

We could declare the variables in the main program and not declare the variables in a header file (`.h`). It is up to you whether you want to write program elements in header files. We will learn more about variables later in this chapter.

Note

The C language is case sensitive, so be careful when writing C code. Most C language instructions are written in non-capitalized letters. Be careful when you declare a variable, too. For example, the variables *x* and *X* are different in C.

There are standard libraries that come with the C language and many programmers make good use of them. The `stdio.h` library (stored as a header file) is widely used in C programming. It defines several macros, variable types, and also specialized functions for performing data input and output; for example, taking input letters from a keyboard or writing text to the console. The console is a text-based area provided by the IDE where reading data from a keyboard or writing text or special characters happens.

This is a short C program example using the `<stdio.h>` directive:

```cpp
// program file name: helloworld.c
#include <stdio.h>  
int main()  
{  // start main block of instructions
     printf("Hello world!"); 
     return 0;
} 
```

C program files are stored with the `.c` extension (such as `mainprogram.c`). The C++ program files are generally stored with the `.cpp` extension (for example, `mainprogram.cpp`).

The `printf()`, which displays characters (for example, a text message) on the IDE's console. As you can see from the preceding program example, we wrote some comments explaining each line of code. The next section shows the different ways of writing comments in the C language.

### Using comments in C

**Comments** are either blocks or lines of text that don't affect the functioning of C programs. Writing comments in C programming is useful because they can be used to explain and clarify the meaning or the functioning of instructions, functions, variables, and so on. All the comments that we write in the program are ignored by the compiler. There are a couple of ways of writing comments in C:

*   Using double slashes (`//`): This makes a single-line comment.
*   Using slashes and asterisks (`/*  */`): This makes a comment with a block of text.

This code example demonstrates how to use both types of comments:

```cpp
/**********************************************************
Program: Helloworld.c 
Purpose: It shows the text "Hello, world!!" on the IDE's console. 
Author: M. Garcia.
Program creation date: September 9, 2020.
Program version: 1.0 
**********************************************************/
#include <stdio.h>  //standard I/O library
int main(void)
{
    int x; // we declare an integer variable

    printf("Hello, world!!"); 
    x=1; // we assign the value of 1 to variable x.
}
```

Tip

It is a good programming practice to write the code's purpose, version number and date, and the author's name(s) as comments at the beginning of your C program.

The next section describes how to declare and use variables in C programming. Variables are very useful, and you will use them in most of the chapters of this book.

### Understanding variables in C

A **variable** is a name (also called an identifier) assigned via programming to a microcontroller memory storage area that holds data temporarily. There are specific types of variables in C that hold different types of data. The variable types determine the layout and size of the variable's assigned microcontroller memory (generally, its internal random-access memory or RAM).

We must declare a variable in the C language first to use it in your code. The variable declaration has two parts – a data type and an identifier, using this syntax: `<data_type> <identifier>`. The following explains both:

*   A **data type** (or just type) defines the type of data to be stored in the variable (for example, an integer number). There are many data types and their modifiers. The following table describes the four main types:

![Table 2.1 – The main four data types used in the C language](img/Table_2.1_B16413.jpg)

Table 2.1 – The main four data types used in the C language

*   Each type from *Table 2.1* has the modifiers `unsigned`, `signed`, `short`, and `long`, among others. For example, we can declare a variable that holds unsigned integers as `unsigned int x;`.
*   There is another type named `void`. This type has no value and is generally used to define a function type that returns nothing.
*   An identifier is a unique name identifying the variable. Identifiers can be written with the letters a..z or A..Z, the numbers 0..9, and the underscore character: _. The identifier must not have spaces, and the first character must not be a number. Remember that identifiers are case-sensitive. In addition, an identifier should have fewer than 32 characters according to the ANSI C standard.

For example, let's declare a variable named x that can hold a floating-point number:

`float x;`

In the preceding line of code example, the C compiler will assign variable *x* a particular memory allocation holding only floating-point numbers.

Now, let's use that variable in the following line of code:

`x=1.10;`

As you can see, we store the floating-point value of 1.10 in the variable named *x*. The following example demonstrates how to use a variable in a C program:

```cpp
/* program that converts from Fahrenheit degrees to Celsius degrees. Written by Miguel Garcia-Ruiz. Version 1.0\. Date: Sept. 9, 2020
*/
#include <stdio.h> // standard I/O library to write text
int main(void) // It won't return any value
{
    float celsius_degrees;
    float fahrenheit_degrees=75.0;
    // Calculate the conversion:
    celsius_degrees=(fahrenheit_degrees-32)*5/9;
    // printf displays the result on the console:
    printf("%f",celsius_degrees); 
}
```

You can initialize a variable with a value when it is declared, as shown in the preceding example for the `fahrenheit_degrees` variable.

We can also store strings in a variable using double quotes at the beginning and end of the string. Here's an example:

`char  name = "Michael";`

The preceding example shows how a string is stored in a char variable type, which is an array of characters.

### Declaring local and global variables

There are two types of variables in C depending on where they are declared. They can have different values and purposes:

*   **Global variables**: These are declared outside all the functions from your code. These variables can be used in any function and through the whole program.
*   **Local variables**: Local variables are declared inside a function. They only work inside the function that were declared, so their value cannot be used outside that function. Have a look at this example containing both global and local variables:

```cpp
#include<stdio.h>
// These are global variables:
int y;
int m;
int x;
int b;
int straight_line_equation() {
    y=m*x+b;
    return y;
}
int main(){
    int answer;  // this is a local variable
    m=2;
    x=3;
    b=5;
    answer = straight_line_equation();
    printf(" %d\n  ",answer);
    return 0;  // this terminates  program
}
```

In the preceding example, the global variables *y*, *m*, *x*, and *b* work in all programs, including inside the `straight_line_equation()` function.

### Using constants

**Constants** (also called constant variables) can be used to define a variable that has a value that does not change throughout the entire program. Constants in C are useful for defining mathematical constants. This is the syntax for declaring a constant:

`const <data_type> <identifier>=<value>;`

Here, the data type can be either `int`, `float`, `char`, or `double`, or their modifiers, for example:

```cpp
const float euler_constant=2.7183;
const char A_character='a';
```

You can also declare variables using the `#define` directive. It is written at the beginning of a program, right after the `#include` directive, without a semicolon at the end of the line, using this syntax: `#define` `<identifier> <value>` .

We don't need to declare the constant's data type. The compiler will determine that dynamically. The following examples show how to declare constants:

```cpp
#define PI 3.1416
#define value1 11
#define char_Val 'z'
```

The next section deals with keywords from the C language that are widely used in C programs.

### Applying keywords

The ANSI C standard defines a number of **keywords** that have a specific purpose in  C programming. These keywords cannot be used to name variables or constants. These are the keywords (statements) that you can use in your C code:

`auto, break, case, char, const, continue, default, do, double, else,  enum, extern, float, for, goto, if, int, long, register, return, short, signed sizeof, static, struct, switch, typedef, union, unsigned, void, volatile, while.`

The compilers used to compile programs for the Blue Pill and Curiosity Nano boards have additional keywords. We will list them in this chapter. The following section explains what functions in C are.

### Declaring functions in C

A `main()`. This function is written in C programs, and other functions are called from it. You can logically divide your code up into functions to make it more readable and to group instructions that are related to the same task, giving the instructions some structure. Functions in C are defined more or less like algebraic functions where you have a function name, a function definition, and a function parameter(s).

The general form for defining a function in C is the following:

```cpp
<return_data_type> <function_name> (parameter list) {    <list of instructions>
    return <expression>; //optional
}
```

The `return` statement allows a value from a function to be returned, and this returned value is used in other parts of the program. The return statement is optional since you can code a function that does not return a value.

Tip

It is a good programming practice to indent the instructions contained in a function block. This gives the function more visual structure and readability.

The following function example shows how to use parameters and how data is returned from a function, where `number1` and `number2` are the function parameters:

```cpp
int maxnumber(int number1, int number2) {
    /* Declaring a local variable to store the result: */
    int result1;
    if (number1 > number2)
        result1 = number1;
    else
        result1 = number2;
    return result1; 
}
```

In the preceding example, the function returns the results of the comparison between the two numbers.

Tip

Make sure that the function's data type has the same type as the variable used in the `return` statement.

If, for some reason, you don't need to return a value from a function, you can use the `void` statement instead of defining the function's data type, for example:

```cpp
void error_message ()
{
    printf("Error.");
}
```

In the preceding example, we are not using the `return 0` statement in the function because it's not returning any value. We can then `error_message();`.

### Calling a function

Once we declare a function, we need to **call** it, that is, run it in another part of your code. This transfers the program control to the called function and it will run the instruction(s) contained in it. After executing all the instructions from the function, the program control resumes, running instructions from the main program.

To call a function, you will need to write the function name and the required values for the parameters. If your function returns a value, you can store it in a variable. For example, let's call the `max()` function that we explained previously:

```cpp
int result2;
result2=maxnumber(4,3);
```

In this example, the result of the number comparison made by the `maxnumber()` function will be stored in the `result2` variable.

### Evaluating expressions (decision statements)

The C language provides a way to declare one or more **logic conditions** that can be evaluated (tested) by the program, as well as some statements that need to be executed according to the result of that evaluation, that is, if the condition is either true or false.

The C programming language assumes that the true value is any non-null or non-zero value. It is false if the value is zero or null. C has the following decision-making statements:

*   `if` (expression_to_evaluate) {statements}: This has a Boolean expression in the decision that is followed by one or more statements to be run if the decision is true, for example:

    ```cpp
    #include <stdio.h>
    void main(){
    	int x;
    	x=11;
    	if (x>10) {
    		printf("yes, x is greater than 10");
    	}
    }
    ```

*   `if` (decision) {statements} `else` {statements}: The `else` component can be used after an `if` statement and can be useful when running one or more statements if the decision is false, for example:

    ```cpp
    #include <stdio.h>
    void main(){
      int x;
      x=5;
      if (x>10) {
         printf("yes, x is greater than 10");
      }
      else {
        printf("no, x is not greater than 10");
      }
    }
    ```

    In the preceding example, the x variable is analyzed, and if x is greater than 10, it will print out this message on the IDE's console: `yes, x is greater than 10`, otherwise it will print out `no, x is not greater than 10`.

    Tip

    Be careful when you evaluate two variables with the `if` statement. Use double equal signs for that (==). If you use only one equal sign, the compiler will raise an error. Do it like this: `if` (x==y) {statements}

*   The `switch` statement compares the value of a variable against a number of possible values, which are called cases. Each case from the `switch` statement has a unique name (identifier). If a match is not found in the list of cases, then the default statement will be executed and the program control goes out of the `switch` with the list of cases. The optional `break` statement is used to terminate the program control outside of the `switch` block. This is useful if, for some reason, you don't want the `switch` statement to keep evaluating the rest of the cases. The following is the syntax for the `switch` statement:

```cpp
switch( expression_to_evaluate)
{
    case value1:
        <statement(s)>;
        break;
    case value_n:
        <statement(s)>;
        break;
}
```

The preceding code shows the syntax for the `switch` statement, including its break sentence. The following code is an example of using `switch`, which will compare the variable age against three cases. In case the variable has a value of `10`, it will print out the following text: `the person is a child`:

```cpp
#include <stdio.h>
void main(){
	int age;
	age=10;
	switch (age)
	{
		case 10:
			printf ("the person is a child");
			break;
		case 30:
			printf ("the person is an adult");
			break;
		case 80:
			printf ("the person is a senior citizen");
			break;
	}	
}
```

So far, we have reviewed how to logically evaluate an expression. The next section explains how to run one or more statements repeatedly. This can be useful for some repetitive tasks for the microcontroller board, such as reading data from an input microcontroller port continuously.

### Understanding loops

A `for`, `while`, and `do..while`:

#### for loop

The `for` loop repeats one or more statements contained in its block until a test expression becomes false. This is the syntax of the `for` loop:

```cpp
for (<initialization_variable>;         <test_expression_with_variable>; <update_variable>)
{
    <statement(s)_to_run>;
} 
```

In the preceding syntax, the `counter_variable` initialization is executed once. Then, the expression is evaluated with `counter_variable`. If the tested expression is false, the loop is terminated. If the evaluated expression is true, the block statement(s) are executed, and `counter_variable` is updated. `counter_variable` is a local variable that only works in the `for` loop. This example prints out a list of numbers from 1 to 10 on the IDE's console:

```cpp
for (int x=1; x<=10; x++)
{
    printf("%d ", x);
}
```

Please note that the x++ statement is the same as writing x=x+1.

#### while loop

The `while` loop repeats one or more statements from its block while a given condition is true, testing its condition prior to executing the statements. When its condition tests false, the loop terminates. Here is the syntax for the `while` loop statement:

```cpp
while (<test_expression>) 
{
    statement(s); 
}
```

The preceding code is the syntax for the `while` loop. The following is example code that uses the `while` loop, counting from 0 to 10:

```cpp
int x = 0;
while (x <= 10)
{
    // \n it will display the next number in a new 
    // line of text:
    printf("%d \n", x); 
    x=x+1;
}
```

#### do..while loop:

This type of loop is very similar to the `while` loop. The `do..while` loop executes its block statement(s) at least once. The expression is evaluated at the end of the block. The process continues until the evaluated expression is false.

The following is the syntax for the `do..while` loop:

```cpp
do
{
    statement(s);
}
while (<test_expression>);
```

The following example uses the `do..while` loop, counting numbers from 5 to 50, while the sum is < 50:

```cpp
int number=5;
do
{
    number=number+5;
    printf("%d ", number);
}
while (number < 50);
```

In the preceding code, the variable called `number` has the value 5 added to it and the variable is printed out on the IDE's console at least once, and then the variable is evaluated.

### The infinite loops

You can also program an **infinite loop**, which, of course, will run endlessly (the loop does not terminate) until we abort the program (or disconnect the power from the microcontroller board!). Infinite loops can be useful for showing a result from a microcontroller continuously, reading data from a microcontroller board continuously without stopping it, and so on.

You can do this using any of the three types of loops. The following are some examples of infinite loops:

```cpp
for(; ;)
{
    printf("this text will be displayed endlessly!");
}
while(1) 
{
    printf("this text will be displayed endlessly!");
}
do
{
    printf("this text will be displayed endlessly!");
}
while (1);
```

As you can see from the preceding code, programming endless loops is easy and simple.

### The break and continue keywords in loops

You can `break` keyword. The `break` statement of the following example will stop the `for` loop, but the statement will run only once:

```cpp
for (int x=1; x<=10; x++)
{
    printf("%d ", x);
    break;
}
```

You can use the `break` statement in any of the three types of loops.

The `continue` keyword. This example will not print out the second line of text:

```cpp
for (int x=1; x<=10; x++)
{
    printf("%d ", x);
    continue;
    printf("this line won't be displayed.");
}
```

The preceding code displays the value of x without displaying the next line of text because of the `continue` statement, moving the program control to the beginning of the `for` loop.

The next section deals with a number of C statements and functions that were created specifically for the Curiosity Nano microcontroller board and that are slightly different from those for the Blue Pill board.

# Introducing Curiosity Nano microcontroller board programming

As you learned from [*Chapter 1*](B16413_01_Final_NM_ePub.xhtml#_idTextAnchor014), *Introduction to Microcontrollers and Microcontroller Boards*, the Curiosity Nano can be programmed using ANSI C language, explained in this chapter, using the MPLAB X IDE.

The basic structure of a C program for the Curiosity Nano is similar to the one explained above using the `main()` function, but its declaration changes. You have to include the keyword void in it, as follows:

```cpp
//necessary IDE's library defining input-output ports:
#include "mcc_generated_files/mcc.h"
void main(void) //main program function
{
    // statements
}
```

The file `16F15376_Curiosity_Nano_IOPorts.zip` from the book's GitHub page contains the necessary `IO_RD1_GetValue()` function will read an analog value from the Curiosity Nano's RD1 port.

The following are useful functions that you can use for programming the Curiosity Nano, which is already defined by the MPLAB X compiler. Note that `xxx` means the Curiosity Nano's port name. Please read [*Chapter 1*](B16413_01_Final_NM_ePub.xhtml#_idTextAnchor014), *Introduction to Microcontrollers and Microcontroller Boards*, to familiarize yourself with the Curiosity Nano's I/O port names and their respective chip pins:

*   `IO_xxx_SetHigh();`: This function writes the logic HIGH (3.3 V) value on the specified pin (port).
*   `IO_xxx_SetLow();`: This function writes the logic LOW (0 V) value on the specified pin (port).
*   `IO_xxx_GetValue();`: This function returns the logic (digital) value (either HIGH or LOW) that is read from the specified port. HIGH is returned as 1\. LOW is returned as 0.
*   `ADC_GetConversion(xxx);`: This function reads an analog value from the specified port and returns a value from 0 to 1023 corresponding to the analog-to-digital conversion done on the read value.
*   `SYSTEM_Initialize();`:  This function initializes the microcontroller ports.
*   `__delay_ms(number_milliseconds);`: This function pauses the program for a number of milliseconds (there are 1,000 milliseconds in one second).
*   `IO_xxx_Toggle();`: This function toggles the port's value to its opposite state of the specified port. If the port has a logic of HIGH (1), this function will toggle it to 0, and vice versa.

We will use some of the preceding functions in an example explained later in this chapter.

*Figure 2.1* shows the Curiosity Nano's pins. Bear in mind that many of them are I/O ports:

![Figure 2.1 – Curiosity Nano's pins configuration](img/Figure_2.1_B16413.jpg)

Figure 2.1 – Curiosity Nano's pins configuration

We have configured the following ports from the Curiosity Nano microcontroller board as I/O ports. We did this in all the Curiosity Nano's software project files from this book. The ports' pins can be seen in *Figure 2.1*. Some of them are used throughout this book:

RA0, RA1, RA2, RA3, RA4, RA5, RB0, RB3, RB4, RB5, RC0, RC1, RC7, RD0, RD1, RD2, RD3, RD5, RD6, RD7, RE0, RE1, and SW0.

The following section explains the basic programming structure and important functions for the Blue Pill board microcontroller board coding, which are somewhat different from the Curiosity Nano board.

# Introducing Blue Pill microcontroller board programming

As you learned from [*Chapter 1*](B16413_01_Final_NM_ePub.xhtml#_idTextAnchor014), *Introduction to Microcontrollers and Microcontroller Boards*, you can program the Blue Pill board using the Arduino IDE, along with a special library installed in the IDE. Remember that this IDE uses C++ language, which is an extension of C. Programs are also called sketches in Arduino IDE programming. All the sketches must have two functions, called `setup()` and `loop()`.

The `setup()` function is used to define variables, define input or output ports (board pins), define and open a serial port, and so on, and this function will run only once. It must be declared before the `loop()` function.

The `loop()` function is the main block of your code and will run the main statements of your program. This `loop()` function will run repeatedly and endlessly. Sketches do not require the `main()` function.

This is the main structure for your sketches (programs):

```cpp
void setup() 
{
    statement(s);
}
void loop() 
{
    statement(s);
}
```

Here is how to define pins (a microcontroller board's ports) either as inputs or outputs:

```cpp
void setup ( ) 
{
 // it sets the pin as output.
    pinMode (pin_number1, OUTPUT);
 // it sets the pin as input 
    pinMode (pin_number2, INPUT); 
}
```

An input port will serve to read data from a sensor or switch, and an output port will be used to send data to another device or component, turn on an LED, and suchlike.

Tip

Programming in the Arduino IDE is case-sensitive. Be careful when you write function names, define variables, and so on.

As you can see from the preceding code, each block of statements is enclosed in curly brackets, and each statement ends with a semicolon, similar to ANSI C. These are useful functions that can be used for programming the Blue Pill:

*   `digitalWrite(pin_number, value);`: This function writes a HIGH (3.3 V) or LOW (0 V) value on the specified pin (port); for example, `digitalWrite(13,HIGH);` will send a HIGH value to pin (port) number 13.

    Note

    You must previously declare `pin_number` as `OUTPUT` in the `setup()` function.

*   `digitalRead(pin_number);`: This function returns either a logic HIGH (3.3 V) or logic LOW (0 V) value that is read from a specified pin (port), for example, `val = digitalRead(pin_number);`.

    Note

    You must previously declare `pin_number` as `INPUT` in the `setup()` function.

*   `analogWrite(pin_number, value);`: This function writes (sends) an analog value (0..65535) to a specified PIN (output port) of the Blue Pill.
*   `analogRead(pin_number);`: This function returns an analog value read from the specified PIN. The Blue Pill has 10 channels (ports or pins that can be used as analog inputs) with a 12-bit `analogRead()` function will map input voltages between 0 and 3.3 volts into integer numbers between 0 and 4095, for example:
*   `int val = analogRead(A7);`
*   `delay(number_of_milliseconds);`: This function pauses the program for the specified amount of time defined in milliseconds (remember that there are one thousand milliseconds in a second).

    Tip

    You can also use the C language structure explained in this section for programming the Arduino microcontroller boards, with the only difference being that the range of values for `analogWrite()` will be 0...255 instead of 0...65535, and `analogRead()` will have a range of 0 to 1023 instead of 0 to 4095.

*Figure 2.2* shows the I/O ports and other pins from the Blue Pill:

![Figure 2.2 – The Blue Pill's pins configuration](img/Figure_2.2_B16413.jpg)

Figure 2.2 – The Blue Pill's pins configuration

The ports' pins can be seen in *Figure 2.2*. Some of them are used in this book's chapters. The Blue Pill has the following analog ports: A0, A1, A2, A3, A4, A5, A6, A7, B0, and B1\. The following are digital I/O ports: C13, C14, C15, B10, B11, B12, B13, B14, B15, A8, A9, A10, A11, A12, A15, B3, B4, B5, B6, B7, B8, and B9.

Just remember that in the code, the ports are referenced as `PA0`, `PA1`, and so on, adding a letter `P`.

We will use some of the preceding functions in an example in the next section.

# Example – Programming and using the microcontroller board's internal LED

In this section, we will use common statements from C/C++ languages for controlling an internal LED from the Blue Pill and the Curiosity Nano boards. The internal LED can be very useful for quickly verifying the state of I/O ports, showing data from sensors, and so on, without the need to connect an LED with its respective resistor to a port. The next section will show how to compile and send a piece of code to the microcontroller boards using their internal LED.

## Programming the Blue Pill's internal LED

This section covers the steps for programming the internal LED. You don't need to connect any external electronic component, such as external LEDs. Using the internal LED from the Blue Pill is useful for quickly testing out and showing the result or variable value from a program. You will only need to use the microcontroller boards. The following steps demonstrate how to upload and run the program to the Blue Pill:

1.  Connect the ST-LINK/V2 interface to the Blue Pill, as explained in [*Chapter 1*](B16413_01_Final_NM_ePub.xhtml#_idTextAnchor014), *Introduction to Microcontrollers and Microcontroller Boards*.
2.  Connect the USB cable to the Blue Pill and your computer. Insert the Blue Pill into the solderless breadboard. *Figure 2.3* shows the internal LED from the Curiosity Nano and the Blue Pill boards:![Figure 2.3 – The Blue Pill (top) and the Curiosity Nano's internal LEDs](img/Figure_2.3_B16413.jpg)

    Figure 2.3 – The Blue Pill (top) and the Curiosity Nano's internal LEDs

3.  Open Arduino IDE. Write the following program in its editor:

    ```cpp
    /*
      Blink
      This program turns on the Blue Pill's internal LED   on for one second, then off for two seconds,   repeatedly.
      Version number: 1.
      Date: Sept. 18, 2020.
      Note: the internal LED is internally connected to   port PC13.
      Written by Miguel Garcia-Ruiz.
     */
    void setup() 
    {
      pinMode(PC13, OUTPUT);
    }
    void loop() 
    {
      digitalWrite(PC13, HIGH);
      delay(1000);
      digitalWrite(PC13, LOW);
      delay(2000);             // it waits for two seconds
    }
    ```

4.  Click on the `PC13` for `LED_BUILTIN`.

You could leave the Blue Pill without inserting it in a solderless breadboard because we are not connecting any component or wire to the Blue Pill's ports in the preceding example.

### Programming the Curiosity Nano's internal LED

Similar to the Blue Pill, you can use the Curiosity Nano's internal LED to quickly show data from sensors, and so on, without connecting an LED to a port. The whole project containing this example and other supporting files necessary for compiling it on the MPLAB X IDE is stored on the GitHub page. It is a zip file called `16F15376_Curiosity_Nano_LED_Blink_Delay.zip`.

Follow these steps to run the program on the MPLAB X IDE:

1.  Connect the USB cable to the Curiosity Nano and insert the board in the solderless breadboard. Unzip the `16F15376_Curiosity_Nano_LED_Blink_Delay.zip` file.
2.  On the MPLAB X IDE, click on **File/Open Project** and then open the project.
3.  Double-click on the project folder and click on the `Source Files` folder.
4.  Click on `main.c` and you will see the following source code:

    ```cpp
    /*
    This program makes the on-board LED to blink once a second (1000 milliseconds).
    Ver. 1\. July, 2020\. Written by Miguel Garcia-Ruiz
    */
    //necessary library generated by MCC:
    #include "mcc_generated_files/mcc.h" 
    void main(void) //main program function
    {
        // initializing the microcontroller board:
        SYSTEM_Initialize(); 
        //it sets up LED0 as output: 
        LED0_SetDigitalOutput();
        while (1) //infinite loop
        {
            LED0_SetLow(); //it turns off the on-board LED
            __delay_ms(1000); //it pauses the program for                           //1 second 
            LED0_SetHigh(); //it turns on on-board LED and                         //RE0 pin
            __delay_ms(1000); //it pauses the program for                           //1 second 
        }
    }
    ```

5.  Compile and run the code by clicking on the run icon (colored green), which is on the top menu. If everything went well, you will see Curiosity Nano's internal LED blinking.

As you can see from the preceding example, it has useful C functions specifically created for the Curiosity Nano board, such as the following:

`SetLow(), SetHigh() and __delay_ms().`

Those functions are essential for making projects with microcontroller boards, and they are used in other chapters of this book.

# Summary

In this chapter, we learned how to properly configure and set up the MPLAB X and the Arduino IDEs for the C microcontroller board programming. We were introduced to the C programming language, and in particular, a set of C language instructions necessary for programming the Blue Pill and microcontroller boards. To practice what you have learned about the C language, we looked at a number of practical circuits using the boards' internal and external LEDs. The instructions and structure learned in this chapter can be applied to the rest of this book.

[*Chapter 3*](B16413_03_Final_NM_ePub.xhtml#_idTextAnchor041), *Turning an LED On and Off Using a Push Button*, will focus on how to connect a push button with a pull-up resistor to a microcontroller board, as well as how to minimize electrical noise when using the push button. It will also explain how to set up a microcontroller board's input port via software, along with possible applications of push buttons.

# Further reading

*   Gay, W. (2018). *Beginning STM32: Developing with FreeRTOS, libopencm3, and GCC*. St. Catharines, ON: Apress.
*   Microchip Technology (2019). *MPLAB X IDE User's Guide.* Retrieved from [https://ww1.microchip.com/downloads/en/DeviceDoc/50002027E.pdf](https://ww1.microchip.com/downloads/en/DeviceDoc/50002027E.pdf).