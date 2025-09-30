# C++ Concepts

In this chapter, we will explore the basics of writing a C++ program. Here, we will cover just enough to wrap our heads around the capabilities of the C++ programming language. This will be required to understand the code used in this book.

To run the examples, use Visual Studio 2017\. You can download the community version for free at [https://visualstudio.microsoft.com/vs/](https://visualstudio.microsoft.com/vs/):

![](img/db0f902d-4364-49e3-9ae1-534447c62654.png)

The topics covered in this chapter are as follows:

*   Program basics
*   Variables
*   Operators
*   Statements
*   Iteration
*   Functions
*   Arrays and pointers
*   `Struct` and `Enum`
*   Classes and inheritance

# Program basics

C++ is a programming language, but what exactly is a program? A program is a set of instructions executed in sequence to give a desired output.

Let's look at our first program:

```cpp
#include <iostream> 
// Program prints out "Hello, World" to screen 
int main() 
{ 
    std::cout<< "Hello, World."<<std::endl; 
    return 0; 
} 
```

We can look at this code line by line.

The hash (`#`) `include` is used when we want to include anything that is using valid C++ syntax. In this case, we are including a standard C++ library in our program. The file we want to include is then specified inside the `<>` angle brackets. Here, we are including a file called `iostream.h`. This file handles the input and output of data to the console/screen.

On the second line, the `//` double slash marks the initiation of a code comment. Comments in code are not executed by the program. They are mainly to tell the person looking at the code what the code is currently doing. It is good practice to comment your code so that when you look at code you wrote a year ago, you will know what the code does.

Basically, `main()` is a function. We will cover functions shortly, but a `main` function is the first function that is executed in a program, also called the entry point. A function is used to perform a certain task. Here, the printing of `Hello, World` is tasked to the `main` function. The contents that need to be executed must be enclosed in the curly brackets of the function. The `int` preceding the `main()` keyword suggests that the function will return an integer. This is why we have returned 0 at the end of the main function, suggesting that the program can be executed and can terminate without errors.

When we want to print out something to the console/screen, we use the `std::cout` (console out) C++ command to send something to the screen. Whatever we want to send out should start and end with the output operator, `<<`. Furthermore, `<<std::endl` is another C++ command, which specifies the end of a era line and that nothing else should be printed on the line afterward. We have to use the prefix before `std::` to tell C++ that we are using the standard namespace with the `std` namespace. But why are namespaces necessary? We need namespaces because anyone can declare a variable name with `std`. How would the compiler differentiate between the two types of `std`? For this, we have namespaces to differentiate between the two.

Note that the two lines of code we have written in the main function have a semicolon (`;`) at the end of each line. The semicolon tells the compiler that this is the end of the instructions for that line of code so that the program can stop reading when it gets to the semicolon and go to the next line of instruction. Consequently, it is important to add a semicolon at the end of each line of instruction as it is mandatory.

The two lines of code we wrote before can be written in one line as follows:

```cpp
std::cout<< "Hello, World."<<std::endl;return 0; 
```

Even though it is written in a single line, for the compiler, there are two instructions with both instructions ending with a semicolon.

The first instruction is to print out `Hello, World` to the console, and the second instruction is to terminate the program without any errors.

It is a very common mistake to forget semicolons, and it happens to beginners as well as experienced programmers every now and then. So it's good to keep this in mind, for when you encounter your first set of compiler errors.

Let's run this code in Visual Studio using the following steps:

1.  Open up Visual Studio and create a new project by going to File | New | Project.
2.  On the left-hand side, select Visual C++ and then Other. For the Project Type, select Empty Project. Give this project a Name. Visual Studio automatically names the first project `MyFirstProject`. You can name it whatever you like.
3.  Select the Location that you want the project to be saved in:

![](img/e0d2ffa8-f51c-466e-9909-8c7b997c796e.png)

4.  Once the project is created, in Solution Explorer, right-click and select Add | New Item:

![](img/552989f8-944f-4b65-bc77-e97a21607b67.png)

5.  Create a new `.cpp` file, called the `Source` file:

![](img/0d158b87-6031-4a65-b7ba-5e5c18ccdeb6.png)

6.  Copy the code at the start of the section into the `Source.cpp` file.
7.  Now press the *F5* key on the keyboard or press the Local Window Debugger button at the top of the window to run the application.
8.  A popup of the console should appear upon running the program. To make the console stay so that we can see what is happening, add the following highlighted lines to the code:

```cpp
#include <iostream> 
#include <conio.h>
// Program prints out "Hello, World" to screen 
int main() 
{ 
   std::cout << "Hello, World." << std::endl;       
    _getch();
   return 0; 
} 

```

What `_getch()` does is it stalls the program and waits for a character input to the console without printing the character to the console. So, the program will wait for some input and then close the console.

To see what is printed to the console, we just add it for convenience. To use this function, we need to include the `conio.h` header.

9.  When you run the project again, you will see the following output:

![](img/fcf0fe01-072d-4c2b-b94c-10af283b00e3.png)

Now that we know how to run a basic program, let's look at the different data types that are included in C++.

# Variables

A variable is used to store a value. Whatever value you store in a variable is stored in the memory location associated with that memory location. You assign a value to a variable with the following syntax.

We can first declare a variable type by specifying a type and then the variable name:

```cpp
Type variable;
```

Here, `type` is the variable type and `variable` is the name of the variable.

Next, we can assign a value to a variable:

```cpp
Variable = value;
```

Now that value is assigned to the variable.

Or, you can both declare the variable and assign a value to it in a single line, as follows:

```cpp
type variable = value;
```

Before you set a variable, you have to specify the variable type. You can then use the equals sign (`=`)  to assign a value to a variable.

Let's look at some example code:

```cpp
#include <iostream> 
#include <conio.h>
// Program prints out value of n to screen 
int main() 
{ 
   int n = 42;  
std::cout <<"Value of n is: "<< n << std::endl;     
    _getch();
   return 0; 
} 
```

Replace the previous code with this code in `Source.cpp` and run the application. This is the output you should get:

![](img/210ab313-7c05-4037-9b0e-552296501051.png)

In this program, we specify the data type as `int`. An `int` is a C++ data type that can store integers. So, it cannot store decimal values. We declare a variable called `n`, and then we assign a value of `42` to it. Do not forget to add the semicolon at the end of the line.

In the next line, we print the value to the console. Note that to print the value of `n`, we just pass in `n` in `cout` and don't have to add quotation marks.

On a 32-bit system, an int variable uses 4 bytes (which is equal to 32 bits) of memory. This basically means the int data type can hold values between 0 and 2^(32)-1 (4,294,967,295). However, one bit is needed to describe the sign for the value (positive or negative), which leaves 31 bits remaining to express the actual value. Therefore, a signed int can hold values between -2^(31) (-2,147,483,648) and 2^(31)-1 (2,147,483,647).

Let's look at some other data types:

*   `bool`: A bool can have only two values. It can either store `true` or `false`.
*   `char`: These stores integers ranging between *-128* and *127*. Note that `char` or character variables are used to store ASCII characters such as single characters—letters, for example.
*   `short` and `long`: These are also integer types, but they are able to store more information than just int. The size of int is system-dependent and `long` and `short` have fixed sizes irrespective of the system used.
*   `float`: This is a floating point type. This means that it can store values with decimal spaces such as 3.14, 0.000875, and -9.875\. It can store data with up to seven decimal places.
*   `double`: This is a `float` with more precision. It can store decimal values up to 15 decimal places.

| **Data type** | **Minimum** | **Maximum** | **Size (bytes)** |
| `bool` | `false` | `true` | 1 |
| `char` | -128 | 127 | 1 |
| `short` | -32768 | 327677 | 2 |
| `int` | -2,147,483,648 | 2,147,483,647 | 4 |
| `long` | -2,147,483,648 | 2,147,483,647 | 4 |
| `float` | 3.4 x 10-38 | 3.4 x 1038 | 4 |
| `double` | 1.7 x 10-308 | 1.7 x 10308 | 8 |

You also have unsigned data types of the same data type used to maximize the range of values they can store. Unsigned data types are used to store positive values. Consequently, all unsigned values start at 0.

So, `char` and unsigned `char` can store positive values from *0* to *255*. Similar to unsigned `char`, we have unsigned `short`, `int`, and `long`.

You can assign values to `bool`, `char`, and `float`, as follows:

```cpp
#include <iostream> 
#include <conio.h> 
// Program prints out value of bool, char and float to screen 
int main() 
{ 
   bool a = false; 
   char b = 'b'; 
   float c = 3.1416f; 
   unsigned int d = -82; 

   std::cout << "Value of a is : " << a << std::endl; 
   std::cout << "Value of b is : " << b << std::endl; 
   std::cout << "Value of c is : " << c << std::endl; 
   std::cout << "Value of d is : " << d << std::endl; 

   _getch(); 
   return 0; 
} 
```

This is the output when you run the application:

![](img/9a39d2bf-6f9a-4811-ada9-21637386af5c.png)

Everything is printing fine except `d`, which was assigned `-82`. What happened here? Well that's because `d` can store only unsigned values, so if we assign it `-82`, it gives a garbage value. Change it to just `82` without the negative sign and it will print the correct value:

![](img/37ecd2c1-7dd6-4c2b-afd4-efad93a7f10e.png)

Unlike int, `bool` stores a binary value where `false` is `0` and `true` is `1`. So, when you print out the values of `true` and `false,` the output will be `1` and `0`, respectively. 

Basically, `char` stores characters specified with single quotation marks, and values with decimals are printed just how you stored the values in the floats. An `f` is added at the end of the value when assigning a `float`, to tell the system that it is a float and not a double.

# Strings

Variables that are non-numerical are either a single character or a series of characters called strings. In C++, a series of characters can be stored in a special variable called a string. A string is provided through a standard `string` class.

To declare and use `string` objects, we have to include the string header file. After `#include <conio.h>`, also add `#include <string>` at the top of the file.

A string variable is declared in the same way as other variable types, except before the string type you have to use the `std` namespace.

If you don't like adding the `std::` namespace prefix, you can also add the line using the `std` namespace after `#include`. This way, you won't have to add the `std::` prefix, as the program will understand well enough without it. However, it can be printed out just like other variables:

```cpp
#include <iostream> 
#include <conio.h> 
#include <string> 

// Program prints out values to screen 

int main() 
{ 

   std::string name = "The Dude"; 

   std::cout << "My name is: " << name << std::endl; 

   _getch(); 
   return 0; 
} 
```

Here is the output:

![](img/5e15a6ea-e0d9-43d3-84ab-2a98a765b715.png)

# Operators

An operator is a symbol that performs a certain operation on a variable or expression. So far, we have used the `=` sign, which calls an assignment operator that assigns a value or expression from the right-hand side of the equals sign to a variable on the left-hand side.

The simplest form of other kinds of operators are arithmetic operators such as `+`, `-`, `*`, `/`, and `%`. These operators operate on a variable such as `int` and `float`. Let's look at some of the use cases of these operators:

```cpp
#include <iostream> 
#include <conio.h> 
// Program prints out value of a + b and x + y to screen 
int main() 
{ 
   int a = 8; 
   int b = 12; 
   std::cout << "Value of a + b is : " << a + b << std::endl; 

   float x = 7.345f; 
   float y = 12.8354; 
   std::cout << "Value of x + y is : " << x + y << std::endl; 

   _getch(); 
   return 0; 
} 
```

The output of this is as follows:

![](img/65128404-ecb2-46e3-b7b7-59628d79f69c.png)

Let's look at examples for other operations as well:

```cpp
#include <iostream> 
#include <conio.h> 

// Program prints out values to screen 

int main() 
{ 
   int a = 36; 
   int b = 5; 

   std::cout << "Value of a + b is : " << a + b << std::endl; 
   std::cout << "Value of a - b is : " << a - b << std::endl; 
   std::cout << "Value of a * b is : " << a * b << std::endl; 
   std::cout << "Value of a / b is : " << a / b << std::endl; 
   std::cout << "Value of a % b is : " << a % b << std::endl; 

   _getch(); 
   return 0; 
} 
```

The output is as follows:

![](img/7c4b2181-08a3-41f4-9d1d-204d800b3126.png)

The `+`, `-`, `*`, and `/` signs are self-explanatory. However, there is one more arithmetic operator: `%`, which is called the modulus operator. It returns the remainder of a division.

How many times is 5 contained in 36? The answer is 7 times with a remainder of 1\. That's why the result is 1.

Apart from the arithmetic operators, we also have an increment/decrement operator.

In programming, we increment variables often. You can do `a=a+1;` to increment and `a=a-1;` to decrement a variable value. Alternatively, you can even do `a+=1;` and `a-=1;` to increment and decrement, but in C++ programming there is an even shorter way of doing that, which is by using the `++` and `--` signs to increment and decrement the value of a variable by `1`.

Let's look at an example of how to use it to increment and decrement a value by `1`:

```cpp
#include <iostream> 
#include <conio.h> 

// Program prints out values to screen 

int main() 
{ 

   int a = 36; 
   int b = 5; 

   std::cout << "Value of ++a is : " << ++a << std::endl; 
   std::cout << "Value of --b is : " << --b << std::endl; 

   std::cout << "Value of a is : " << a << std::endl; 
   std::cout << "Value of b is : " << b << std::endl; 

   _getch(); 
   return 0; 
} 
```

The output of this is as follows:

![](img/c5c178b9-4af7-4629-aa5b-ba4199bcf6b9.png)

Consequently, the `++` or `--` operator increments the value permanently. If the `++` operator is to the left of the variable, it is called a pre-increment operator. If it is put afterward, it is called a post-increment operator. There is a slight difference between the two. If we put `++` on the other side, we get the following output:

![](img/c82be4c4-d840-4ff4-bb17-fc06c49869c1.png)

In this case, `a` and `b` are incremented and decremented in the next line. So, when you print the values, it prints out the correct result.

It doesn't make a difference here, as it is a simple example, but overall it does make a difference, and it is good to understand this difference. In this book, we will mostly be using post-increment operators.

In fact, this is how C++ got its name; it is an increment of C.

Apart from arithmetic, increment, and decrement operators, you also have logical and comparison operators.

The logical operators are shown in the following table:

| **Operator** | **Operation** |
| `!` | NOT |
| `&&` | AND |
| `&#124;&#124;` | OR |

Here are the comparison operators:

| **Operator** | **Comparison** |
| `==` | Equal to |
| `!=` | Not equal to |
| `<` | Less than |
| `>` | Greater than |
| `<=` | Less than equal to |
| `>=` | Greater than equal to |

We will cover these operators in the next section.

# Statements

A program may not always be linear. Depending on your requirements, you might have to branch out or bifurcate, repeat a set of code, or take a decision. For this, there are conditional statements and loops.

In a conditional statement, you check whether a condition is true. If it is, you will go ahead and execute the statement.

The first of the conditional statements is the `if` statement. The syntax for this looks as follows:

```cpp
If (condition) statement; 

```

Let's look at how to use this in the following code. Let's use one of the comparison operators here:

```cpp
#include <iostream> 
#include <conio.h> 

// Program prints out values to screen 

int main() 
{ 
   int a = 36; 
   int b = 5; 

   if (a > b) 
   std::cout << a << " is greater than " << b << std::endl; 

   _getch(); 
   return 0; 
} 

```

The output is as follows:

![](img/e402a9c9-70c9-43f8-9eb0-f17d2f4dfe28.png)

We check the whether `a` is greater than `b`, and if the condition is true, then we print out the statement.

But what if the opposite is true? For this, we have the `if...else` statement, which is a statement that basically executes the alternate statement. The syntax looks like this:

```cpp
if (condition) statement1; 
else statement2; 
```

Let's look at it in code:

```cpp
#include <iostream> 
#include <conio.h> 

// Program prints out values to screen 

int main() 
{ 

   int a = 2; 
   int b = 28; 

   if (a > b) 
   std::cout << a << " is greater than " << b << std::endl; 
   else 
   std::cout << b << " is greater than " << a << std::endl; 

   _getch(); 
   return 0; 
}
```

Here, the values of `a` and `b` are changed so that `b` is greater than `a`:

![](img/d33a93af-67d9-45e0-b2da-ceaf56148e7d.png)

One thing to note is that after the `if` and `else` conditions, C++ will execute a single line of statement. If there are multiple statements after `if` or `else`, then the statements need to be in curly brackets, as shown:

```cpp

   if (a > b) 
   {      
         std::cout << a << " is greater than " << b << std::endl; 
   } 
   else 
   { 
         std::cout << b << " is greater than " << a << std::endl; 
   }    
```

You can also have the `if` statements after using `else if`:

```cpp
#include <iostream> 
#include <conio.h> 

// Program prints out values to screen 

int main() 
{ 

   int a = 28; 
   int b = 28; 

   if (a > b) 
   {      
         std::cout << a << " is greater than " << b << std::endl; 
   } 
   else if (a == b)  
{ 

         std::cout << a << " is equal to " << b << std::endl; 
   } 
   else 
   { 
         std::cout << b << " is greater than " << a << std::endl; 
   } 

   _getch(); 
   return 0; 
} 
```

The output is as follows:

![](img/92d616ff-1edb-4d1f-a7fb-d23a7cab258d.png)

# Iteration

Iteration is the process of calling the same statement repeatedly. C++ has three iteration statements: the `while`, `do...while`, and `for` statements. Iteration is also commonly referred to as loops.

The `while` loop syntax looks like the following:

```cpp
while (condition) statement;
```

Let's look at it in action:

```cpp
#include <iostream> 
#include <conio.h>  
// Program prints out values to screen  
int main() 
{  
   int a = 10;  
   int n = 0;  
   while (n < a) { 

         std::cout << "value of n is: " << n << std::endl;  
         n++;    
   } 
   _getch(); 
   return 0;  
} 
```

Here is the output of this code:

![](img/2725dc8f-d8ce-4816-8b9c-9e54c0f3558a.png)

Here, the value of `n` is printed to the console until the condition is met.

The `do while` statement is almost the same as a `while` statement except, in this case, the statement is executed first and then the condition is tested. The syntax is as follows:

```cpp
do statement  
while (condition); 
```

You can give it a go yourself and see the result.

The loop that is most commonly used in programming is the `for` loop. The syntax for this looks as follows:

```cpp
for (initialization; continuing condition; update) statement; 
```

The `for` loop is very self-contained. In `while` loops, we have to initialize `n` outside the `while` loop, but in the `for` loop, the initialization is done in the declaration of the `for` loop itself.

Here is the same example as the `while` loop but with the `for` loop:

```cpp
#include <iostream> 
#include <conio.h>  
// Program prints out values to screen  
int main() 
{  
   for (int n = 0; n < 10; n++)       
         std::cout << "value of n is: " << n << std::endl;  
   _getch(); 
   return 0; 
} 
```

The output is the same as the `while` loop but at look how compact the code is compared to the `while` loop. Also, `n` is scoped locally to the `for` loop body.

We can also increment `n` by `2` instead of `1`, as shown:

```cpp
#include <iostream> 
#include <conio.h> 

// Program prints out values to screen  
int main() 
{  
   for (int n = 0; n < 10; n+=2)      
         std::cout << "value of n is: " << n << std::endl; 
   _getch(); 
   return 0; 
} 
```

Here is the output of this code:

![](img/c036ead1-efc4-45f9-acf8-b5cf3d0dd940.png)

# Jump statements

As well as condition and iteration statements, you also have the `break` and `continue` statements.

The `break` statement is used to break out of an iteration. We can leave a loop and force it to quit if a certain condition is met.

Let's look at the `break` statement in use:

```cpp
#include <iostream> 
#include <conio.h>  
// Program prints out values to screen  
int main() 
{  
   for (int n = 0; n < 10; n++) 
   {         
         if (n == 5) {               
               std::cout << "break" << std::endl; 
               break; 
         } 
         std::cout << "value of n is: " << n << std::endl; 
   }  
   _getch(); 
   return 0; 
} 
```

The output of this is as follows:

![](img/88644fcd-80df-47cb-bcbc-d165d4353fc9.png)

The `continue` statement will skip the current iteration and continue the execution of the statement until the end of the loop. In the `break` code, replace `break` with `continue` to see the difference:

```cpp
#include <iostream> 
#include <conio.h> 

// Program prints out values to screen 

int main() 
{ 

   for (int n = 0; n < 10; n++) 
   { 
         if (n == 5) { 

               std::cout << "continue" << std::endl; 

               continue; 
         } 
         std::cout << "value of n is: " << n << std::endl; 
   } 
   _getch(); 
   return 0; 
} 
```

Here is the output when `break` is replaced with `continue`:

![](img/18313505-c6f3-493b-b6b5-3faaf9960e19.png)

# Switch statement

The last of the statements is the `switch` statement. A `switch` statement checks for several cases of values, and if a value matches the expression, then it executes the corresponding statement and breaks out of the `switch` statement. If it doesn't find any of the values, then it will output a default statement.

The syntax for it looks as follows:

```cpp
switch( expression){ 

case constant1:  statement1; break; 
case constant2:  statement2; break; 
. 
. 
. 
default: default statement; break;  

}  
```

This looks very familiar to the `else if` statements, but this is more sophisticated. Here is an example:

```cpp
#include <iostream> 
#include <conio.h> 

// Program prints out values to screen 

int main() 
{ 
   int a = 28; 

   switch (a) 
   { 
   case 1: std::cout << " value of a is " << a << std::endl; break; 
   case 2: std::cout << " value of a is " << a << std::endl; break; 
   case 3: std::cout << " value of a is " << a << std::endl; break; 
   case 4: std::cout << " value of a is " << a << std::endl; break; 
   case 5: std::cout << " value of a is " << a << std::endl; break; 
   default: std::cout << " value a is out of range " << std::endl; break; 
   } 

   _getch(); 
   return 0; 
} 
```

The output is as follows:

![](img/0e139fba-5539-47cb-887f-01c76a864c8b.png)

Change the value of `a` to equal `2` and you will see that it prints out the statement when the `2` case is correct.

Also note that it is important to add the `break` statement. If you forget to add it, then the program will not break out of the statement.

# Functions

So far, we have written all of our code in the main function. This is fine if you are doing a single task, but once you start doing more with a program, the code will become bigger and over a period of time everything will be in the main function, which will look very confusing.

With functions, you can break your code up into smaller, manageable chunks. This will enable you to structure your program better.

A function has the following syntax:

```cpp
type function name (parameter1, parameter2) {statements;}
```

Going from left to right, `type` here is the return type. After performing a statement, a function is capable of returning a value. This value could be of any type, so we specify a type here. A function has only one variable at a time.

The function name is the name of the function itself.

Then, inside brackets, you will pass in parameters. These parameters are variables of a certain type that are passed into the function to perform a certain function.

Here is an example: two parameters are passed in but you can pass as many parameters you want. You can pass in more than one parameter per function, and each parameter is separated by a comma.

Let's look at this example:

```cpp
#include <iostream> 
#include <conio.h> 

// Program prints out values to screen 

void add(int a, int b)  
{ 
   int c = a + b; 

   std::cout << "Sum of " << a << " and " << b << " is " << c <<   
   std::endl;  
}  
int main() 
{ 
   int x = 28; 
   int y = 12; 

   add(x, y); 

   _getch(); 
   return 0; 
}   
```

Here, we create a new function called `add`. For now, make sure the functions are added before the `main` function; otherwise, `main` will not know that the function exists.

The `add` function doesn't return anything so we use the `void` keyword at the start of the function. Not all functions have to return a value. Next, we name the function `add` and then pass in two parameters, which are `a` and `b` of the `int` type.

In the function, we create a new variable called `c` of the `int` type, add the values of the arguments passed in, and assign it to `c`. The new `add` function finally prints out the value of `c`.

Furthermore, in the `main` function, we create two variables called `x` and `y` of the `int` type, call the `add` function, and pass in `x` and `y` as arguments.

When we call the function, we pass the value of `x` to `a` and the value of `y` to `b`, which is added and stored in `c` to get the following output:

![](img/354f93b0-a0d7-48bd-be85-984f3576a4f5.png)

When you create new functions, make sure they are written above the main function; otherwise, it will not be able to see the functions and the compiler will throw errors.

Now let's write one more function. This time, we will make sure the function returns a value. Create a new function called `multiply`, as follows:

```cpp
int multiply(int a, int b) { 

   return a * b; 

}
```

In the `main` function, after we've called the `add` function, add the following lines:

```cpp
   add(x, y); 

   int c = multiply(12, 32); 

   std::cout << "Value returned by multiply function is: " << c <<  
   std::endl; 
```

In the `multiply` function, we have a return type of `int`, so the function will expect a return value at the end of the function, which we return using the `return` keyword. The returned value is the `a` variable multiplied by the `b` variable.

In the `main` function, we create a new variable called `c`; call the `multiply` function and pass in `12` and `32`. After being multiplied, the return value will be assigned to the value of `c`. After this, we print out the value of `c` in the `main` function.

The output of this is as follows:

![](img/967e8f0e-e1b1-482a-8263-24577e2445f2.png)

We can have a function with the same name, but we can pass in different variables or different numbers of them. This is called **function overloading**.

Create a new function called `multiply`, but this time pass in floats and set the return value to a float as well:

```cpp
float multiply(float a, float b) { 

   return a * b; 

} 
```

This is called function overloading, where the function name is the same, but it takes different types of arguments.

In the `main` function, after we've printed the value of `c`, add the following code:

```cpp
float d = multiply(8.352f, -12.365f); 
std::cout << "Value returned by multiply function is: " << d << std::endl;
```

So, what is this `f` after the float value? Well, `f` just converts the doubles to floats. If we don't add the `f`, then the value will be treated as a double by the compiler.

When you run the program, you'll get the value of `d` printed out:

![](img/b65d3cd7-08d3-45c4-8696-e3e0bbef664d.png)

# Scope of variables

You may have noticed that we have two variables called `c` in the program right now. There is a `c` in the `main` function as well as a `c` in the `add` function. How is it that they are both named `c` but have different values?

In C++, there is the concept of a local variable. This means that the definition of a variable is confined to the local block of code it is defined in. Consequently, the `c` variable in the `add` function is treated differently to the `c` variable in the `main` function.

There are also global variables, which need to be declared outside of the function or block of code. Any piece of code written between curly brackets is considered to be a block of code. Consequently, for a variable to be considered a global variable, it needs to be in the body of the program or it needs to be declared outside a block of code of a function.

# Arrays

So far, we have only looked at single variables, but what if we want a bunch of variables grouped together? Like the ages of all the students in a class, for example. You can keep creating separate variables, `a`, `b`, `c`, `d`, and so on, and to access each you would have to call each of them, which is cumbersome, as you won't know the kind of data they hold.

To organize data better, we can use arrays. Arrays use continuous memory space to store values in a series, and you can access each element with an index number.

The syntax for arrays is as follows:

```cpp
type name [size] = { value0, value1, ....., valuesize-1};
```

So, we can store the ages of five students as follows:

```cpp
int age[5] = {12, 6, 18 , 7, 9 }; 
```

When creating an array with a set number of values, you don't have to specify a size but it is a good idea to do so. To access each value, we use the index from `0` - `4` as the first element with a value of `12` at the *0*^(th) index and the last element, `9`, in the fourth index.

Let's see how to use this in code:

```cpp
#include <iostream> 
#include <conio.h> 

// Program prints out values to screen 
int main() 
{ 
   int age[5] = { 12, 6, 18 , 7, 9 }; 

   std::cout << "Element at the 0th index " << age[0]<< std::endl; 
   std::cout << "Element at the 4th index " << age[4] << std::endl; 

   _getch(); 
   return 0; 
} 
```

The output is as follows:

![](img/fd294672-6bf8-4079-a37a-e8fc68a426ad.png)

To access each element in the array, you can use a `or` loop:

```cpp
#include <iostream> 
#include <conio.h> 

// Program prints out values to screen 
int main() 
{ 
   int age[5] = { 12, 6, 18 , 7, 9 }; 

   for (int i = 0; i < 5; i++) { 

         std::cout << "Element at the "<< i << "th index is: " << 
         age[i] << std::endl;  
   }  
   _getch(); 
   return 0; 
} 
```

The output of this is as follows:

![](img/f2fb6e1c-1141-4cd6-9ece-33babd21ca9a.png)

Instead of calling `age[0]` and so on, we use the `i` index from the `for` loop itself and pass it into the `age` array to print out the index and the value stored at the index.

The `age` array is a single-dimension array. In graphics programming, we have seen that we use a two-dimensional array, which is mostly a 4x4 matrix. Let's look at an example of a two-dimensional 4x4 array. A two-dimensional array is defined as follows:

```cpp
int matrix[4][4] = {
{2, 8, 10, -5},
{15, 21, 22, 32},
{3, 0, 19, 5},
{5, 7, -23, 18}
};
```

To access each element, you use a nested `for` loop.

Let's look at this in the following code:

```cpp
#include <iostream> 
#include <conio.h> 

// Program prints out values to screen 
int main() 
{ 

   int matrix[4][4] = { 
                              {2, 8, 10, -5}, 
                              {15, 21, 22, 32}, 
                              {3, 0, 19, 5}, 
                              {5, 7, -23, 18} 
   }; 

   for (int x = 0; x < 4; x++) { 
         for (int y = 0; y < 4; y++) { 
               std::cout<< matrix[x][y] <<" "; 
         } 
         std::cout<<""<<std::endl; 
   } 

   _getch(); 
   return 0; 
} 
```

The output is as follows:

![](img/6ce38321-4a90-44a5-ae31-f4e3c4b05723.png)

As a test, create two matrices and attempt to carry out matrix multiplication.

You can even pass arrays as parameters to functions, shown in the following example.

Here, the `matrixPrinter` function doesn't return anything but prints out the values stored in each element of the 4x4 matrix:

```cpp
#include <iostream> 
#include <conio.h> 

void matrixPrinter(int a[4][4]) { 

   for (int x = 0; x < 4; x++) { 
         for (int y = 0; y < 4; y++) { 
               std::cout << a[x][y] << " "; 
         } 
         std::cout << "" << std::endl; 
   } 
} 

// Program prints out values to screen 
int main() 
{ 

   int matrix[4][4] = { 
                            {2, 8, 10, -5}, 
                            {15, 21, 22, 32}, 
                            {3, 0, 19, 5}, 
                            {5, 7, -23, 18} 
   }; 

   matrixPrinter(matrix); 

   _getch(); 
   return 0; 
} 

```

We can even use an array of `char` to create a string of words. Unlike `int` and `float` arrays, the characters in an array don't have to be in curly brackets and they don't need to be separated by a comma.

To create a character array, you define it as follows:

```cpp
   char name[] = "Hello, World !"; 
```

You can print out the values just by calling out the name of the array, as follows:

```cpp
#include <iostream> 
#include <conio.h> 

// Program prints out values to screen 
int main() 
{ 

   char name[] = "Hello, World !"; 

   std::cout << name << std::endl; 

   _getch(); 
   return 0; 
} 
```

The output of this is as follows:

![](img/ac0cc047-284e-4f87-9c3e-82df5035dfa5.png)

# Pointers

Whenever we declare new variables so that we can store values in them, we actually send a memory allocation request to the operating system. The operating system will try to reserve a block of continuous memory for our application if there is enough free memory left.

When we want to access the value stored in that memory space, we call the variable name.

We don't have to worry about the memory location where we have stored the value. However, what if we want to get the address of the location where the variable is stored?

The address that locates the variable within the memory is called a reference to the variable. To access this, we use an address of the `&` operator. To get the address location, we place the operator before the variable.

Pointers are variables, and like any other variables they are used to store a value; however, this specific variable type allows the storage of the address—the reference—of another variable.

In C/C++, every variable can also be declared as a pointer that holds a reference to a value of a certain data type by preceding its variable name with an asterisk (`*`). This means, for example, that an `int` pointer holds a reference to a memory address where a value of an `int` may be stored.

A pointer can be used with any built-in or custom data type. If we access the value of a `pointer` variable, we will simply get the memory address it references. So, in order to access the actual value a `pointer` variable references, we have to use the so-called dereferencing operator (`*`).

If we have a variable called `age` and assign a value to it, to get the reference address location we use `&age` to store this address in a variable. To store the reference address, we can't just use a regular variable; we have to use a `pointer` variable and use the dereference operator before it to access the address, as follows:

```cpp
   int age = 10;  
   int *location = &age; 
```

Here, the pointer location will store the address of where the `age` variable value is stored.

If we print the value of `location`, we will get the reference address where `age` is stored:

```cpp
#include <iostream> 
#include <conio.h>  
// Program prints out values to screen 
int main() 
{  
   int age = 10;  
   int *location = &age; 
   std::cout << location << std::endl;  
   _getch(); 
   return 0; 
} 
```

This is the output:

![](img/b6df33e6-80d7-4ce0-b645-f33759b7c0b0.png)

This value might be different for you, as the location will be different from machine to machine.

To get the location of where the `location` variable itself is stored, we can print out `&location` as well.

This is the memory location of the variable on my system memory:

![](img/b6df33e6-80d7-4ce0-b645-f33759b7c0b0.png)

Let's look at another example:

```cpp
#include <iostream> 
#include <conio.h> 

// Program prints out values to screen 
int main() 
{ 
   int age = 18; 
   int *pointer; 

   pointer = &age; 

   *pointer = 12; 

   std::cout << age << std::endl; 

   _getch(); 
   return 0; 
}  
```

Here, we create two `int` variables; one is a regular `int` and the other is a pointer type.

We first set the `age` variable equal to `18`, then we set the address of `age`, and assign it to the `pointer` variable called `pointer`.

The `int` pointer is now pointing to the same address where the `age` variable stores its `int` value.

Next, use the dereference operator on the `pointer` variable to give us access to the `int` values stored at the referenced address and change the current value to `12`.

Now, when we print out the value of the `age` variable, we will see that the previous statement has indeed changed the value of the `age` variable. A null pointer is a pointer that is not pointing to anything, and is set as follows:

```cpp
   int *p = nullptr; 
```

Pointers are very much associated with arrays. As arrays are nothing but continuous sequences of memory, we can use pointers with them.

Consider our arrays example from the arrays section:

```cpp
int age[5] = { 12, 6, 18 , 7, 9 }; 
```

Instead of using the index, we can use pointers to point to the values in the array.

Consider the following code:

```cpp
#include <iostream> 
#include <conio.h> 

// Program prints out values to screen 
int main() 
{ 
   int *p = nullptr; 
   int age[5] = { 12, 6, 18 , 7, 9 }; 

   p = age; 

   std::cout << *p << std::endl; 

   p++; 

   std::cout << *p << std::endl; 

   std::cout << *(p + 3) << std::endl; 

std::cout << *p << std::endl; 

   _getch(); 
   return 0; 
} 
```

In the `main` function, we create a pointer called `pointer`, as well as an array with five elements. We assign the array to the pointer. This causes the pointer to get the location of the address of the first element of the array. So, when we print the value pointed to by the pointer, we get the value of the first element of the array.

With `pointer`, we can also increment and decrement as a regular `int`. However, unlike a regular `int` increment, which increments the value of the variable when you increment a pointer, it will point to the next memory location. So, when we increment `p` it is now pointing to the next memory location of the array. Incrementing and decrementing a pointer means moving the referenced address by a certain number of bytes. The number of bytes depends on the data type that is used for the `pointer` variable.

Here, the pointer is the `int` type, so when we move the pointer by one, it moves 4 bytes and points to the next integer. When we print the value that `p` is pointing to now, it prints the second element's value.

We can also get the value of other elements in the array by getting the pointer's current location and by adding to it the *n*^(th) number you want to get from the current location using `*(p + n)`, where `n` is the *n*^(th) element from `p`. So, when we do `*(p + 3)`, we will get the third element from where `p` is pointing to currently. Since `p` was incremented to the second element, the third element from the second element is the fifth element, and so the value of the fifth element is printed out.

However, this doesn't change the location to which `p` is pointing, which is still the second position.

Here is the output:

![](img/c84858ca-fb47-4851-864f-9a3f0ffc6b5f.png)

# Structs

Structures or structs are used to group data together. A `struct` can have different data elements in it, called members, integers, floats, chars, and so on. You can create many objects of a similar `struct` and store values in the `struct` for data management.

The syntax of `struct` is as follows:

```cpp
struct name{ 

type1 name1; 
type2 name2; 
. 
. 
} ; 

```

An object of `struct` can be created as follows:

```cpp
struct_name     object_name; 
```

An object is an instance of `struct` where we can assign properties to the data types we created while creating the `struct`. An example of this is as follows.

In a situation in which you want to maintain a database of student ages and the height of a section, your `struct` definition will look like this:

```cpp
struct student { 

   int age; 
   float height; 

}; 
```

Now you can create an array of objects and store the values for each student:

```cpp
int main() 
{ 

   student section[3]; 

   section[0].age = 17; 
   section[0].height = 39.45f; 

   section[1].age = 12; 
   section[1].height = 29.45f; 

   section[2].age = 8; 
   section[2].height = 13.45f; 

   for (int i = 0; i < 3; i++) { 

         std::cout << "student " << i << " age: " << section[i].age << 
           " height: " << section[i].height << std::endl; 
   } 

   _getch(); 
   return 0; 
} 
```

Here is the output of this:

![](img/985f22ee-a493-4659-898e-c272f9b9520f.png)

# Enums

Enums are used for enumerating items in a list. When comparing items, it is easier to compare names rather than just numbers. For example, the days in a week are Monday to Sunday. In a program, we will assign Monday to 0, Tuesday to 1, and Sunday to 7, for example. To check whether today is Friday, you will have to count to and arrive at 5\. However, wouldn't it be easier to just check if `Today == Friday`?

For this, we have enumerations, declared as follows:

```cpp
enum name{ 
value1, 
value2, 
. 
. 
. 
};
```

So, in our example, it would be something like this:

```cpp
#include <iostream> 
#include <conio.h> 

// Program prints out values to screen 

enum Weekdays { 
   Monday = 0, 
   Tuesday, 
   Wednesday, 
   Thursday, 
   Friday, 
   Saturday, 
   Sunday, 
}; 

int main() 
{ 

   Weekdays today; 

   today = Friday; 

   if (today == Friday) { 
         std::cout << "The weekend is here !!!!" << std::endl; 
   } 

   _getch(); 
   return 0; 
} 
```

The output of this is as follows:

![](img/e3ebe1ed-2404-4b2e-95e8-632edc7e607a.png)

Also note that, `Monday = 0`. If we don't use initializers, the first item's value is set to `0`. Each following item that does not use an initializer will use the value of the preceding item plus `1` for its value.

# Classes

In C++, structs and classes are identical. You can do exactly the same thing with both of them. The only difference is the default access specifier: `public` for structs and `private` for classes.

The declaration of a class looks like the following code:

```cpp
class name{ 

access specifier: 

name(); 
~name(); 

member1; 
member2; 

} ; 
```

A class starts with the `class` keyword, followed by the name of the class.

In a class, we first specify the access specifiers. There are three access specifiers: `public`, `private`, and `protected`:

*   `public`: All members are accessible from anywhere.
*   `private`: Members are accessible from within the class itself only.
*   `protected`: Members are accessed by other classes that inherit from the class.

By default, all members are private.

Furthermore, `name();` and `~name();` are called the constructor and destructor of a class. They have the same name as the name of the class itself.

The constructor is a special function that gets called when you create a new object of the class. The destructor is called when the object is destroyed.

We can customize a constructor to set values before using the member variables. This is called constructor overloading.

Notice that although the constructor and destructor are functions no return is provided. This is because they are not there for returning values.

Let's look at an example of a class where we create a class called `shape`. This has two member variables for the `a` and `b` sides and a member function, which calculates and prints the area:

```cpp
class shape {  
   int a, b;  
public: 

   shape(int _length, int _width) { 
         a = _length;  
         b = _width; 

         std::cout << "length is: " << a << " width is: " << b << 
                      std::endl; 
   } 

   void area() { 

         std::cout << "Area is: " << a * b << std::endl; 
   } 
}; 
```

We use the class by creating objects of the class.

Here, we create two objects, called `square` and `rectangle`. We set the values by calling the custom constructor, which sets the value of `a` and `b`. Then, we call the `area` function of the object by using the dot operator by pressing the `.` button on the keyboard after typing the name of the object:

```cpp
int main() 
{  
   shape square(8, 8); 
   square.area(); 

   shape rectangle(12, 20); 
   rectangle.area(); 

   _getch(); 
   return 0; 
} 
```

The output is as follows:

![](img/717a6526-c61b-429d-840f-8fd28cd8a2a1.png)

# Inheritance

One of the key features of C++ is inheritance, with which we can create classes that are derived from other classes so that derived or the child class automatically includes some of its parent's member variables and functions.

For example, we looked at the `shape` class. From this, we can have a separate class called `circle` and another class called `triangle` that has the same properties as other shapes, such as area.

The syntax for an inherited class is as follows:

```cpp
class inheritedClassName: accessSpecifier parentClassName{ 

};   
```

Note that `accessSpecifier` could be `public`, `private`, or `protected` depending on the minimum access level you want to provide to the parent member variables and functions.

Let's look at an example of inheritance. Consider the same `shape` class, which will be the parent class:

```cpp
class shape { 

protected:  
   float a, b;  
public: 
    void setValues(float _length, float _width) 
   { 
         a = _length; 
         b = _width; 

         std::cout << "length is: " << a << " width is: " << b <<  
         std::endl; 
   }  
   void area() {  
         std::cout << "Area is: " << a * b << std::endl; 
   } 

}; 
```

Since we want the `triangle` class to access `a` and `b` of the parent class, we have to set the access specifier to protected, as shown previously; otherwise, it will be set to private by default. In addition to this, we also change the data type to floats for more precision. After doing this, we create a `setValues` function instead of the constructor to set the values for `a` and `b`. We then create a child class of `shape` and call it `triangle`:

```cpp
class triangle : public shape { 

public: 
   void area() { 

         std::cout << "Area of a Triangle is: " << 0.5f * a * b << 
                      std::endl; 
   } 

}; 
```

Due to inheritance from the `shape` class, we don't have to add the `a` and `b` member variables, and we don't need to add the `setValues` member function either, as this is inherited from the `shape` class. We just add a new function called `area`, which calculates the area of a triangle.

In the main function, we create an object of the `triangle` class, set the values, and print the area, as follows:

```cpp
int main() 
{ 

   shape rectangle; 
   rectangle.setValues(8.0f, 12.0f); 
   rectangle.area(); 

   triangle tri; 
   tri.setValues(3.0f, 23.0f); 
   tri.area(); 

   _getch(); 
   return 0; 
}
```

Here is the output of this:

![](img/724e481f-d5f1-4dac-a6cd-1a66fa15d3ac.png)

To calculate the area of `circle`, we modify the `shape` class and add a new overloaded `setValues` function, as follows:

```cpp
#include <iostream> 
#include <conio.h> 

class shape { 

protected: 

   float a, b; 

public: 

   void setValues(float _length, float _width) 
   { 
         a = _length; 
         b = _width; 

         std::cout << "length is: " << a << " height is: " << b << 
                      std::endl; 
   } 

 void setValues(float _a)
{
a = _a;
} 
   void area() { 

         std::cout << "Area is: " << a * b << std::endl; 
   } 

};
```

We will then add a new inherited class, called `circle`:

```cpp
class circle : public shape { 

public: 
   void area() { 

         std::cout << "Area of a Circle is: " << 3.14f * a * a << 
                      std::endl; 
   } 

}; 
```

In the main function, we create a new `circle` object, set the radius, and print the area:

```cpp
int main() 
{ 

   shape rectangle; 
   rectangle.setValues(8.0f, 12.0f); 
   rectangle.area(); 

   triangle tri; 
   tri.setValues(3.0f, 23.0f); 
   tri.area(); 

   circle c; 
   c.setValues(5.0f); 
   c.area(); 

   _getch(); 
   return 0; 
}
```

Here is the output of this:

![](img/12973d84-a6e2-44ed-b42c-4c574172df4f.png)

# Summary

In this chapter, we covered the basics of programming—from what variables are and how to store values in them, to looking at operators and statements, to how to decide when each is required. After that, we looked at iterators and functions, which can be used to make our job simpler and automate the code as much as possible. Arrays and pointers help us to group and store data of a similar type, and with `struct` and `enum` we can create custom data types. Finally, we looked at classes and inheritance, which is the crux of using C++ and makes it convenient to define our data types with custom properties.

In the next chapter, we will look at the foundation of graphics programming and explore how three-dimensional and two-dimensional objects are displayed on the screen.