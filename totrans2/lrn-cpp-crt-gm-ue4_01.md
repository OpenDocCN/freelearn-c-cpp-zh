# Chapter 1. Coding with C++

You're a first-time programmer. You have a lot to learn!

Academics often describe programming concepts in theory but like to leave implementation to someone else, preferably someone from the industry. We don't do that in this book—in this book, we will describe the theory behind C++ concepts and implement our own game as well.

The first thing I will recommend is that you do the exercises. You cannot learn to program simply by reading. You must work with the theory with the exercises.

We are going to get started by programming very simple programs in C++. I know that you want to start playing your finished game right now. However, you have to start at the beginning to get to that end (if you really want to, skip over to [Chapter 12](part0080_split_000.html#2C9D02-dd4a3f777fc247568443d5ffb917736d "Chapter 12. Spell Book"), *Spell Book,* or open some of the samples to get a feel for where we are going).

In this chapter, we will cover the following topics:

*   Setting up a new project (in Visual Studio and Xcode)
*   Your first C++ project
*   How to handle errors
*   What are building and compiling?

# Setting up our project

Our first C++ program will be written outside of UE4\. To start with, I will provide steps for both Xcode and Visual Studio 2013, but after this chapter, I will try to talk about just the C++ code without reference to whether you're using Microsoft Windows or Mac OS.

## Using Microsoft Visual C++ on Windows

In this section, we will install a code editor for Windows, Microsoft's Visual Studio. Please skip to the next section if you are using a Mac.

### Note

The Express edition of Visual Studio is the free version of Visual Studio that Microsoft provides on their website. Go to [http://www.visualstudio.com/en-us/products/visual-studio-express-vs.aspx](http://www.visualstudio.com/en-us/products/visual-studio-express-vs.aspx) to start the installation process.

To start, you have to download and install **Microsoft Visual Studio Express 2013 for Windows Desktop**. This is how the icon for the software looks:

![Using Microsoft Visual C++ on Windows](img/00003.jpeg)

### Tip

Do not install **Express 2013 for Windows**. This is a different package and it is used for different things than what we are doing here.

Once you have Visual Studio 2013 Express installed, open it. Work through the following steps to get to a point where you can actually type in the code:

1.  From the **File** menu, select **New Project...**, as shown in the following screenshot:![Using Microsoft Visual C++ on Windows](img/00004.jpeg)
2.  You will get the following dialog:![Using Microsoft Visual C++ on Windows](img/00005.jpeg)

    ### Tip

    Note that there is a small box at the bottom with the text **Solution name**. In general, **Visual Studio Solutions** might contain many projects. However, this book only works with a single project, but at times, you might find it useful to integrate many projects into the same solution.

3.  There are five things to take care of now, as follows:

    1.  Select **Visual C++** from the left-hand side panel.
    2.  Select **Win32 Console Application** from the right-hand side panel.
    3.  Name your app (I used `MyFirstApp`).
    4.  Select a folder to save your code.
    5.  Click on the **OK** button.

4.  After this an **Application Wizard** dialog box opens up, as shown in the following screenshot:![Using Microsoft Visual C++ on Windows](img/00006.jpeg)
5.  We have four things to take care of in this dialog box, as follows:

    1.  Click on **Application Settings** in the left-hand side panel.
    2.  Ensure that **Console application** is selected.
    3.  Select **Empty project**.
    4.  Click on **Finish**.

Now you are in the Visual Studio 2013 environment. This is the place where you will do all your work and code.

However, we need a file to write our code into. So, we will add a C++ code file to our project, as shown in the following screenshot:

![Using Microsoft Visual C++ on Windows](img/00007.jpeg)

Add your new source code file as shown in the following screenshot:

![Using Microsoft Visual C++ on Windows](img/00008.jpeg)

You will now edit `Source.cpp`. Skip to the Your First C++ Program section and type in your code.

## Using XCode on a Mac

In this section, we will talk about how to install Xcode on a Mac. Please skip to the next section if you are using Windows.

Xcode is available on all Mac machines. You can get Xcode using the Apple App Store (it's free), as shown here:

![Using XCode on a Mac](img/00009.jpeg)

1.  Once you have Xcode installed, open it. Then, navigate to **File** | **New** | **Project...** from the system's menu bar at the top of your screen, as shown in the following screenshot:![Using XCode on a Mac](img/00010.jpeg)
2.  In the New Project dialog, select **Application** under **OS X** on the left-hand side of the screen, and select **Command Line Tool** from the right-hand side pane. Then, click on **Next**:![Using XCode on a Mac](img/00011.jpeg)

    ### Note

    You might be tempted to click on the **SpriteKit Game** icon, but don't click on it.

3.  In the next dialog, name your project. Be sure to fill in all the fields or Xcode won't let you proceed. Make sure that the project's **Type** is set to **C++** and then click on the **Next** button, as shown here:![Using XCode on a Mac](img/00012.jpeg)
4.  The next popup will ask you to choose a location in order to save your project. Pick a spot on your hard drive and save it there. Xcode, by default, creates a Git repository for every project you create. You can uncheck **Create git repository** —we won't cover Git in this chapter—as shown in the following screenshot:![Using XCode on a Mac](img/00013.jpeg)

### Tip

Git is a **Version control system**. This basically means that Git keeps the snapshots of all the code in your project every so often (every time you *commit* to the repository). Other popular **source control management** tools (**scm**) are Mercurial, Perforce, and Subversion. When multiple people are collaborating on the same project, the scm tool has the ability to automatically merge and copy other people's changes from the repository to your local code base.

Okay! You are all set up. Click on the **main.cpp** file in the left-hand side panel of Xcode. If the file doesn't appear, ensure that the folder icon at the top of the left-hand side panel is selected first, as shown in the following screenshot:

![Using XCode on a Mac](img/00014.jpeg)

# Creating your first C++ program

We are now going to write some C++ source code. There is a very good reason why we are calling it the source code: it is the source from which we will build our binary executable code. The same C++ source code can be built on different platforms such as Mac, Windows, and iOS, and in theory, an executable code doing the exact same things on each respective platform should result.

In the not-so-distant past, before the introduction of C and C++, programmers wrote code for each specific machine they were targeting individually. They wrote code in a language called assembly language. But now, with C and C++ available, a programmer only has to write code once, and it can be deployed to a number of different machines simply by sending the same code through different compilers.

### Tip

In practice, there are some differences between Visual Studio's flavor of C++ and Xcode's flavor of C++, but these differences mostly come up when working with advanced C++ concepts, such as templates.

One of the main reasons why using UE4 is so helpful is that UE4 will erase a lot of the differences between Windows and Mac. The UE4 team did a lot of magic in order to get the same code to work on both Windows and Mac.

### Note

**A real-world tip**

It is important for the code to run in the same way on all machines, especially for networked games or games that allow things such as shareable replays. This can be achieved using standards. For example, the IEEE floating-point standard is used to implement decimal math on all C++ compilers. This means that the result of computations such as `200 * 3.14159` should be the same on all the machines.

Write the following code in Microsoft Visual Studio or in Xcode:

[PRE0]

Press *Ctrl* + *F5* to run the preceding code in Visual Studio, or press ![Creating your first C++ program](img/00015.jpeg) + *R* to run in Xcode.

The first time you press *Ctrl* + *F5* in Visual Studio, you will see this dialog:

![Creating your first C++ program](img/00016.jpeg)

Select **Yes** and **Do not show this dialog again**—trust me, this will avoid future problems.

The first thing that might come to your mind is, "My! A whole lot of gibberish!"

Indeed, you rarely see the use of the hash (#) symbol (unless you use Twitter) and curly brace pairs `{` `}` in normal English texts. However, in C++ code, these strange symbols abound. You just have to get used to them.

So, let's interpret this program, starting from the first line.

This is the first line of the program:

[PRE1]

This line has two important points to be noted:

1.  The first thing we see is an `#include` statement. We are asking C++ to copy and paste the contents of another C++ source file, called `<iostream>`, directly into our code file. The `<iostream>` is a standard C++ library that handles all the sticky code that lets us print text to the screen.
2.  The second thing we notice is a `//` comment. C++ ignores any text after a double slash (`//`) until the end of that line. Comments are very useful to add in plain text explanations of what some code does. You might also see `/* */` C-style comments in the source. Surrounding any text in C or C++ with slash-star `/*` and star-slash `*/` gives an instruction to have that code removed by the compiler.

This is the next line of code:

[PRE2]

The comments beside this line explain what the `using` statement does: it just lets you use a shorthand (for example, `cout`) instead of the fully qualified name (which, in this case, would be `std::cout`) for a lot of our C++ code commands. Some people don't like a `using namespace std;` statement; they prefer to write the `std::cout` longhand every time they want to use `cout`. You can get into long arguments over things like this. In this section of the text, we prefer the brevity that we get with the `using namespace` `std;` statement.

This is the next line:

[PRE3]

This is the application's starting point. You can think of `main` as the start line in a race. The `int main()` statement is how your C++ program knows where to start; take a look at the following figure:

![Creating your first C++ program](img/00017.jpeg)

If you don't have an `int main()` program marker or if `main` is spelled incorrectly, then your program just won't work because the program won't know where to start.

The next line is a character you don't see often:

[PRE4]

This `{` character is not a sideways mustache. It is called a curly brace, and it denotes the starting point of your program.

The next two lines print text to the screen:

[PRE5]

The `cout` statement stands for console output. Text between double quotes will get an output to the console exactly as it appears between the quotes. You can write anything you want between double quotes except a double quote and it will still be valid code.

### Tip

To enter a double quote between double quotes, you need to stick a backslash (`\`) in front of the double quote character that you want inside the string, as shown here:

[PRE6]

The `\` symbol is an example of an escape sequence. There are other escape sequences that you can use; the most common escape sequence you will find is `\n`, which is used to jump the text output to the next line.

The last line of the program is the `return` statement:

[PRE7]

This line of code indicates that the C++ program is quitting. You can think of the `return` statement as returning to the operating system.

Finally, the end of your program is denoted by the closing curly brace, which is an opposite-facing sideways mustache:

[PRE8]

## Semicolons

Semicolons (;) are important in C++ programming. Notice in the preceding code example that most lines of code end in a semicolon. If you don't end each line with a semicolon, your code will not compile, and if that happens, you can be fired from your job.

## Handling errors

If you make a mistake while entering code, then you will have a syntax error. In the face of syntax errors, C++ will scream murder and your program will not even compile; also, it will not run.

Let's try to insert a couple of errors into our C++ code from earlier:

![Handling errors](img/00018.jpeg)

Warning! This code listing contains errors. It is a good exercise to find all the errors and fix them!

As an exercise, try to find and fix all the errors in this program.

### Note

Note that if you are extremely new to C++, this might be a hard exercise. However, this will show you how careful you need to be when writing C++ code.

Fixing compilation errors can be a nasty business. However, if you input the text of this program into your code editor and try to compile it, it will cause the compiler to report all the errors to you. Fix the errors, one at a time, and then try to recompile. A new error will pop up or the program will just work, as shown in the following screenshot:

![Handling errors](img/00019.jpeg)

Xcode shows you the errors in your code when you try to compile it

The reason I am showing you this sample program is to encourage the following workflow as long as you are new to C++:

1.  Always start with a working C++ code example. You can fork off a bunch of new C++ programs from the *Your First C++ Program* section.
2.  Make your code modifications in small steps. When you are new, compile after writing each new line of code. Do not code for one to two hours and then compile all that new code at once.
3.  You can expect it to be a couple of months before you can write code that performs as expected the first time you write it. Don't get discouraged. Learning to code is fun.

## Warnings

The compiler will flag things that it thinks might be mistakes. These are another class of compiler notices known as warnings. Warnings are problems in your code that you do not have to fix for your code to run but are simply recommended to be fixed by the compiler. Warnings are often indications of code that is not quite perfect, and fixing warnings in code is generally considered good practice.

However, not all warnings are going to cause problems in your code. Some programmers prefer to disable the warnings that they do not consider to be an issue (for example, warning 4018 warns against signed/unsigned mismatch, which you will most likely see later).

# What is building and compiling?

You might have heard of a computer process term called compiling. Compiling is the process of converting your C++ program into code that can run on a CPU. Building your source code means the same thing as compiling it.

See, your source `code.cpp` file will not actually run on a computer. It has to be compiled first for it to run.

This is the whole point of using Microsoft Visual Studio Express or Xcode. Visual Studio and Xcode are both compilers. You can write C++ source code in any text-editing program—even in Notepad. But you need a compiler to run it on your machine.

Every operating system typically has one or more C++ compilers that can compile C++ code to run on that platform. On Windows, you have Visual Studio and Intel C++ Studio compiler. On Mac, there is Xcode, and on all of Windows, Mac, and Linux, there is the **GNU Compiler Collection** (**GCC**).

The same C++ code that we write (Source) can be compiled using different compilers for different operating systems, and in theory, they should produce the same result. The ability to compile the same code on different platforms is called portability. In general, portability is a good thing.

## Scripting

There is another class of programming languages called scripting languages. These include languages such as PHP, Python, and ActionScript. Scripted languages are not compiled—for JavaScript, PHP, and ActionScript, there is no compilation step. Rather, they are interpreted from the source as the program is run. The good thing about scripting languages is that they are usually platform-independent from the first go, because interpreters are very carefully designed to be platform-independent.

### Exercise – ASCII art

Game programmers love ASCII art. You can draw a picture using only characters. Here's an example of an ASCII art maze:

[PRE9]

Construct your own maze in C++ code or draw a picture using characters.

# Summary

To sum it up, we learned how to write our first program in the C++ programming language in our integrated development environment (IDE, Visual Studio, or Xcode). This was a simple program, but you should count getting your first program to compile and run as your first victory. In the upcoming chapters, we'll put together more complex programs and start using Unreal Engine for our games.

![Summary](img/00020.jpeg)

The preceding screenshot is of your first C++ program and the following screenshot is of its output, your first victory:

![Summary](img/00021.jpeg)