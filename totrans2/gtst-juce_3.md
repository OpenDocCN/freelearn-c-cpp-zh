# Chapter 3. Essential Data Structures

JUCE includes a range of important data structures, many of which could be seen as replacements for some of the standard library classes. This chapter introduces the essential classes for JUCE development. In this chapter we will cover the following topics:

*   Understanding the numerical types
*   Specifying and manipulating strings of text using the `String` class
*   Measuring and displaying time
*   Specifying file paths in a cross-platform manner using the `File` class (including access to the user's home space, the `Desktop` and `Documents` locations)
*   Using dynamically allocated arrays: the `Array` class
*   Employing smart pointer classes

By the end of this chapter, you will be able to create and manipulate data in a range of JUCE's essential classes.

# Understanding the numerical types

The word size of some the basic data types (`char`, `int`, `long`, and so on) varies across platforms, compilers, and CPU architectures. A good example is the type `long`. In Xcode on Mac OS X, `long` is 32 bits wide when compiling 32-bit code and 64 bits wide when compiling 64-bit code. In Microsoft Visual Studio on Windows, `long` is always 32 bits wide. (The same applies to the unsigned versions too.) JUCE defines a handful of primitive types to assist the writing of platform-independent code. Many of these have familiar names and may be the same names used in other libraries and frameworks in use by your code. These types are defined in the `juce` namespace; therefore, can be disambiguated using the `juce::` prefix if necessary. These primitive types are: `int8` (8-bit signed integer), `uint8` (8-bit unsigned integer), `int16` (16-bit signed integer), `uint16` (16-bit unsigned integer), `int32` (32-bit signed integer), `uint32` (32-bit unsigned integer), `int64` (64-bit signed integer), `uint64` (64-bit unsigned integer), `pointer_sized_int` (a signed integer that is the same word size as a pointer on the platform), `pointer_sized_uint` (an unsigned integer that is the same word size as a pointer on the platform), and `juce_wchar` (a 32-bit Unicode character type).

In many cases the built-in types are sufficient. For example, JUCE internally makes use of the `int` data type for a number of purposes, but the preceding types are available where the word size is critical. In addition to this, JUCE does not define special data types for `char`, `float`, or `double`. Both floating-point types are assumed to be compliant with IEEE 754, and the `float` data type is assumed to be 32 bits wide and the `double` data type 64 bits wide.

One final utility in this regard addresses the issue that writing 64-bit literals in code differs across compilers. The `literal64bit()` macro can be used to write such literals if needed:

[PRE0]

JUCE also declares some basic template types for defining certain geometry; the `Component` class uses these in particular. Some useful examples are `Point<ValueType>`, `Line<ValueType>`, and `Rectangle<ValueType>`.

# Specifying and manipulating text strings

In JUCE, text is generally manipulated using the `String` class. In many ways, this class may be seen as an alternative to the C++ Standard Library `std::string` class. We have already used the `String` class for the basic operations in earlier chapters. For example, in [Chapter 2](ch02.html "Chapter 2. Building User Interfaces"), *Building User Interfaces*, strings were used to set the text appearing on a `TextButton` object and used to store a dynamically changing string to display in response to mouse activity. Even though these examples were quite simple, they harnessed the power of the `String` class to make setting and manipulating the strings straightforward for the user.

The first way this is achieved is through storing strings using **reference counted** objects. That is to say, when a string is created, behind the scenes JUCE allocates some memory for the string, stores the string, and returns a `String` object that refers to this allocated memory in the background. Straight copies of this string (that is, without any modifications) are simply new `String` objects that refer to this same shared memory. This helps keep code efficient by allowing `String` objects to be passed by value between functions, without the potential overhead of copying large chunks of memory in the process.

To illustrate some of these features, we will use a console, rather than a Graphical User Interface (GUI), application in the first instance. Create a new Introjucer project named `Chapter03_01`; changing the **Project Type** to **Console Application,** and only selecting **Create a Main.cpp file** in the **Files to Auto-Generate** menu. Save the project and open it into your Integrated Development Environment (IDE).

## Posting log messages to the console

To post messages to the console window, it is best to use JUCE's `Logger` class. Logging can be set to log a text file, but the default behavior is to send the logging messages to the console. A simple "Hello world!" project using a JUCE `String` object and the `Logger` class is shown as follows:

[PRE1]

The first line of code in the `main()` function stores a pointer to the current logger such that we can reuse it a number of times in later examples. The second line creates a JUCE `String` object from the literal C string `"Hello world!"`, and the third line sends this string to the logger using its `writeToLog()` function. Build and run this application, and the console window should look something like the following:

[PRE2]

JUCE reports the first line automatically; this may be different if you have a later version of JUCE from the GIT repository. This is followed by any logging messages from your application.

## String manipulation

While this example is more complex than an equivalent using standard C strings, the power of JUCE's `String` class is delivered through the storage and manipulation of strings. For example, to concatenate strings, the `+` operator is overloaded for this purpose:

[PRE3]

Here, separate strings are constructed from literals for `"Hello"`, the space in between, and `"world!"`, then the final `message` string is constructed by concatenating all three. The stream operator `<<` may also be used for this purpose for a similar result:

[PRE4]

The stream operator concatenates the right-hand side of the expression onto the left-hand side of the expression, in-place. In fact, using this simple case, the `<<` operator is equivalent to the `+=` operator when applied to strings. To illustrate this, replace all the instances of `<<` with `+=` in the code.

The main difference is that the `<<` operator may be more conveniently chained into longer expressions without additional parentheses (due to the difference between the precedence in C++ of the `<<` and `+=` operators). Therefore, the concatenation can be done all on one line, as with the `+` operator, if needed:

[PRE5]

To achieve the same results with `+=` would require cumbersome parentheses for each part of the expression: `(((message += "Hello") += " ") += "world!")`.

The way the internal reference counting of strings works in JUCE means that you rarely need to be concerned about unintended side effects. For example, the following listing works as you might expect from reading the code:

[PRE6]

This produces the following output:

[PRE7]

Breaking this down into steps, we can see what happens:

*   `String string1 ("Hello");`: The `string1` variable is initialized with a literal string.
*   `String string2 = string1;`: The `string2` variable is initialized with `string1`; they now refer to exactly the same data behind the scenes.
*   `string1 << " world!";`: The `string1` variable has another literal string appended. At this point `string1` refers to a completely new block of memory containing the concatenated string.
*   `log->writeToLog ("string1: " + string1);`: This logs `string1`, showing the concatenated string `Hello world!`.
*   `log->writeToLog ("string2: " + string2);`: This logs `string2`; this shows that `string1` still refers to the initial string `Hello`.

One really useful feature of the `String` class is its numerical conversion capabilities. Generally, you can pass a numerical type to a `String` constructor, and the resulting `String` object will represent that numerical value. For example:

[PRE8]

Other useful features are conversions to uppercase and lowercase. Strings may also be compared using the `==` operator.

# Measuring and displaying time

The JUCE `Time` class provides a cross-platform way to specify, measure, and format date and time information in a human-readable fashion. Internally, the `Time` class stores a value in milliseconds relative to midnight on 1st January 1970\. To create a `Time` object that represents the current time, use `Time::getCurrentTime()` like the following:

[PRE9]

To bypass the creation of the `Time` object, you can access the millisecond counter as a 64-bit value directly:

[PRE10]

The `Time` class also provides access to a 32-bit millisecond counter that measures time since system startup:

[PRE11]

The important point to note about `Time::getMillisecondCounter()` is that it is independent of the system time, and would be unaffected by changes to the system time either by the user changing the time, changes due to national daylight saving, and so on.

## Displaying and formatting time information

Displaying time information is straightforward; the following example gets the current time from the operating system, formats it as a string, and sends it to the console output:

[PRE12]

This illustrates the four option flags available to the `Time::toString()` function. The output on the console will be something like:

[PRE13]

For more comprehensive options, the `Time::formatted()` function allows the user to specify a format using a special format string (using a system equivalent to the standard C `strftime()` function). Alternatively, you can obtain the various parts of the date and time information (day, month, hour, minute, time zone, and so on), and combine them into a string yourself. For example, the same preceding format can be achieved as follows:

[PRE14]

## Manipulating time data

`Time` objects may also be manipulated (with the help from the `RelativeTime` class) and compared with other `Time` objects. The following example shows the creation of three time values, based on the current time, using a one-hour offset:

[PRE15]

The output of this should be something like this:

[PRE16]

To compare two `Time` objects, the standard comparison operators may be used. For example, you could wait for a specific time, like the following:

[PRE17]

Two things to note here are that:

*   The value passed to the `RelativeTime` constructor is in seconds (all the other time values need to use one of the static functions as shown earlier for hours, minutes, and so on).
*   The call to `Thread::sleep()` uses values in milliseconds and this sleeps the calling thread. The `Thread` class will be examined further in [Chapter 5](ch05.html "Chapter 5. Helpful Utilities"), *Helpful Utilities*.

## Measuring time

The time values returned from the `Time::getCurrentTime()` function should be accurate for most purposes, but as pointed out earlier, the *current time* could be changed by the user modifying the system time. An equivalent to the preceding example, using `Time::getMillisecondCounter()` that is not susceptible to such changes, is shown as follows:

[PRE18]

Both the `Time::getCurrentTime()` and `Time::getMillisecondCounter()` functions have a similar accuracy, which is within a few milliseconds on most platforms. However, the `Time` class also provides access to a higher resolution counter that returns values as a double precision (64-bit) floating-point value. This function is `Time::getMillisecondCounterHiRes()`, and is also relative to the system start-up as is the value returned from the `Time::getMillisecondCounter()` function. One application of this is to measure the time that certain pieces of code have taken to execute, as shown in the following example:

[PRE19]

This records the current time by polling the higher resolution counter, performing a large number of floating point additions, and polling the higher resolution counter again to determine the duration between these two points in time. The output should be something like this:

[PRE20]

Of course, the results here are dependent on the optimization settings in the compiler and the runtime system.

## Specifying file paths

JUCE provides a relatively cross-platform way of specifying and manipulating file paths using the `File` class. In particular, this provides a means of accessing various special directories on the user's system, such as the `Desktop` directory, their user `Documents` directory, application preferences directories, and so on. The `File` class also provides functions for accessing information about a file (for example, creation date, modification date, file size) and basic mechanisms for reading and writing file contents (although other techniques may be more appropriate for large or complex files). In the following example, a string is written to a text file on disk (using the `File::replaceWithText()` function), then read back into a second string (using the `File::loadFileAsString()` function), and displayed in the console:

[PRE21]

The `File` object in this case is initialized with the path `./chapter03_01_test.txt`. It should be noted that this file may not exist at this point, and on first run it will not exist until the call to the `File::replaceWithText()` function (and on subsequent runs this file will exist, but will be overwritten at that point). The `./` character sequence at the front of this path is a common idiom specifying that the remainder of the path should be relative to the current directory (or current working directory). In this simple case, the current working directory is likely to be the directory where the executable file is located. The following screenshot shows this location relative to the Introjucer project on the Mac platform:

![Specifying file paths](img/3316_03_01.jpg)

This is not a reliable method; however, it will work if the working directory is specifically where you want to save a file.

## Accessing various special directory locations

It is more precise to use one of the `File` class's special locations, as shown as follows:

[PRE22]

The steps for accessing the file location in this directory are split across several lines for clarity in this code. Here, you can see the code to obtain the location of the current executable file, then its parent directory, and then create a file reference for our text file that is relative to this directory. Much of this code may be compacted on a single logical line using a chain of function calls:

[PRE23]

Due to the length of some of the identifiers in this code and the page width in this book, this code still occupies four physical lines of code. Nevertheless, this illustrates how you can employ this function calls to suit your needs and preferences for code layout.

## Obtaining various information about files

The `File` class can provide useful information about files. One important test is whether a file exists; this can be determined using `File::exists()`. If a file does exist, then more information may be obtained, such as its creation date, modification date, and size. These are illustrated in the following example:

[PRE24]

Assuming you ran all of the preceding examples, the file should exist on your system and the information will be reported in the console something like as follows:

[PRE25]

## Other special locations

In addition to `File::currentExecutableFile`, other special locations known to JUCE are:

*   `File::userHomeDirectory`
*   `File::userDocumentsDirectory`
*   `File::userDesktopDirectory`
*   `File::userApplicationDataDirectory`
*   `File::commonApplicationDataDirectory`
*   `File::tempDirectory`
*   `File::currentExecutableFile`
*   `File::currentApplicationFile`
*   `File::invokedExecutableFile`
*   `File::hostApplicationPath`
*   `File::globalApplicationsDirectory`
*   `File::userMusicDirectory`
*   `File::userMoviesDirectory`
*   `File::userPicturesDirectory`

Each of these names is fairly self-explanatory. In some cases, these special locations are not applicable on some platforms. For example, there is no such thing as the `Desktop` on the iOS platform.

## Navigating directory structures

Ultimately, a `File` object resolves to an absolute path on the user's system. This can be obtained using the `File::getFullPathName()` function if needed:

[PRE26]

In addition to this, the relative path passed to `File::getChildFile()` can contain one or more references to parent directories using the double period notation (that is, the "`..`" character sequence). In this next example, we create a simple directory structure as shown in the screenshot following this code listing:

[PRE27]

![Navigating directory structures](img/3316_03_02.jpg)

This creates five directories in total, using only two calls to the `File::createDirectory()`function. Since this is dependent on the user's permissions to create files in this directory, the function returns a `Result` object. This contains a state to indicate if the function succeeded or not (which we check with the `Result::wasOk()` function), and more information can be gained about any errors if needed. Each call to the `File::createDirectory()` function ensures that it creates any intermediate directories if required. Therefore, on the first call, it creates the root directory, directory `1`, and directory `1/a`. On the second call, the root already exists, so it needs only to create directories `2` and `2/a`.

The console output for this should be something like this:

[PRE28]

Of course, the first line will be different, depending on your system, but the remaining five lines should be the same. These paths are displayed relative to the root of the directory structure we have created using the `File::getRelativePathFrom()` function. Notice that the final line shows that the `rel` object refers to the same directory as the `dir2b` object, but we created this `rel` object relative to the `dir1a` object by using the function call `dir1a.getChildFile("../../2/b")`. That is, we navigate two levels up the directory structure then access the directories below.

The `File` class also includes features to check for a file's existence, to move and copy files within the filesystem (including moving the file to the **Trash** or **Recycle Bin**), and to create legal filenames on particular platforms (for example, avoiding colon and slash characters).

# Using dynamically allocated arrays

While most instances of JUCE objects can be stored in regular C++ arrays, JUCE offers a handful of arrays that are more powerful, somewhat comparable to the C++ Standard Library classes, such as `std::vector`. The JUCE `Array` class offers many features; these arrays can be:

*   Dynamically sized; items can be added, removed, and inserted at any index
*   Sorted using custom comparators
*   Searched for particular content

The `Array` class is a template class; its main template argument, `ElementType`, must meet certain criteria. The `Array` class moves its contents around by copying memory during resizing and inserting elements, this could cause problems with certain kinds of objects. The class passed as the `ElementType` template argument must also have both a copy constructor and an assignment operator. The `Array` class, in particular, works well with primitive types and some commonly used JUCE classes, for example, the `File` and `Time` classes. In the following example, we create an array of integers, add five items to it, and iterate over the array, sending the contents to the console:

[PRE29]

This should produce the output:

[PRE30]

Notice that the JUCE `Array` class supports the C++ indexing subscript operator `[]`. This will always return a valid value even if the array index is out of bounds (unlike a built-in array). There is a small overhead involved in making this check; therefore, you can avoid the bounds checking by using the `Array::getUnchecked()` function, but you must be certain that the index is within bounds, otherwise your application may crash. The second `for()` loop can be rewritten as follows to use this alternative function, because we have already checked that out indices will be in-range:

[PRE31]

## Finding the files in a directory

The JUCE library uses the `Array` objects for many purposes. For example, the `File` class can fill an array of `File` objects with a list of child files and directories it contains using the `File::findChildFiles()` function. The following example should post a list of files and directories in your user `Documents` directory to the console:

[PRE32]

Here, the `File::findChildFiles()` function is passed the array of `File` objects, to which it should add the result of the search. It is also told to find both files and directories using the value `File::findFilesAndDirectories` (other options are the `File::findDirectories` and `File::findFiles` values). Finally, it is told not to search recursively.

## Tokenizing strings

Although it is possible to use `Array<String>` to hold an array of JUCE `String` objects, there is a dedicated `StringArray` class to offers additional functionality when applying array operations to string data. For example, a string can be **tokenized** (that is, broken up into smaller strings based on whitespace in the original string) using the `String::addTokens()` function, or divided into strings representing lines of text (based on newline character sequences found within the original string) using the `String::addLines()` function. The following example tokenizes a string, then iterates over the resulting `StringArray` object, posting its contents to the console:

[PRE33]

## Arrays of components

User interfaces comprising banks of similar controls, such as buttons and sliders, can be managed effectively using arrays. However, the JUCE `Component` class and its subclasses do not meet the criteria for storage as an object (that is, by value) in a JUCE `Array` object. These must be stored as arrays of pointers to these objects instead. To illustrate this, we need a new Introjucer project with a basic window as used throughout [Chapter 2](ch02.html "Chapter 2. Building User Interfaces"), *Building User Interfaces*. Create a new Introjucer project, such as this, name it `Chapter03_02`, and open it into your IDE. To the end of the `MainWindow` constructor in `Main.cpp`, add the following line:

[PRE34]

In the `MainComponent.h` file change the code to:

[PRE35]

Notice that the `Array` object here is an array of pointers to `TextButton` objects (that is, `TextButton*)`. In the `MainComponent.cpp` file change the code to:

[PRE36]

Here, we create 10 buttons and using a `for()` loop, adding these buttons to an array, and basing the name of the button on the loop counter. The buttons are allocated using the `new` operator (rather than the static allocation used in [Chapter 2](ch02.html "Chapter 2. Building User Interfaces"), *Building User Interfaces*), and it is these pointers that are stored in the array. (Notice also, that there is no need for the `&` operator in the function call to `Component::addAndMakeVisible()` because the value is already a pointer.) In the `resized()` function, we use a `Rectangle<int>` object to create a rectangle that is inset from the `MainContentComponent` object's bounds rectangle by 10 pixels all the way around. The buttons are positioned within this smaller rectangle. The height for each button is calculated by dividing the height of our rectangle by the number of buttons in the button array. The `for()` loop then positions each button, based on its index within the array. Build and run the application; its window should present 10 buttons arranged in a single column.

There is one major flaw with the preceding code. The buttons allocated with the `new` operator are never deleted. The code should run fine, although you will get an assertion failure when the application is exited. The message into the console will be something like:

[PRE37]

To solve this, we could delete the buttons in the `MainComponent` destructor like so:

[PRE38]

However, it is very easy to forget to do this kind of operation when writing complex code.

## Using the OwnedArray class

JUCE provides a useful alternative to the `Array` class that is dedicated to pointer types: the `OwnedArray` class. The `OwnedArray` class always stores pointers, therefore should not include the `*` character in the template parameter. Once a pointer is added to an `OwnedArray` object, it takes ownership of the pointer and will take care of deleting it when necessary (for example, when the `OwnedArray` object itself is destroyed). Change the declaration in the `MainComponent.h` file, as highlighted in the following:

[PRE39]

You should also remove the code from the destructor in the `MainComponent.cpp` file, because deleting objects more than once is equally problematic:

[PRE40]

Build and run the application, noticing that the application will now exit without problems.

This technique can be extended to using broadcasters and listeners. Create a new GUI-based Introjucer project as before, and name it `Chapter03_03`. Change the `MainComponent.h` file to:

[PRE41]

This time we use an `OwnedArray<Button>` object rather than an `OwnedArray<TextButton>` object. This simply avoids the need to typecast our button pointers to different types when searching for the pointers in the array, as we do in the following code. Also, notice here that we added a `Label` object to our component, made our component a button listener, and that we do not need a destructor. Change the `MainComponent.cpp` file to:

[PRE42]

Here, we add the label in the constructor, reduce the width of the bank of buttons to occupy only the left half of the component, and position the label at the top in the right-half. In the button listener callback, we can obtain the index of the button using the `OwnedArray::indexOf()` function to search for the pointer (incidentally, the `Array` class also has an `indexOf()` function for searching the items). Build and run the application and notice that our label reports which button was clicked. Of course, the elegant thing about this code is that we need only change the value in the `for()` loop when the buttons are created in our constructor to change the number of buttons that are created; everything else works automatically.

## Other banks of controls

This approach may be applied to other banks of controls. The following example creates a bank of sliders and labels, keeping each corresponding component updated with the appropriate value. Create a new GUI-based Introjucer project, and name it `Chapter03_04`. Change the `MainComponent.h` file to:

[PRE43]

Here, we have arrays of sliders and labels and our component is both a label listener and a slider listener. Now, update the `MainComponent.cpp` file to contain the include directive, the constructor, and the `resized()` function:

[PRE44]

Here, we use a `for()` loop to create the components and add them to the corresponding arrays. In the `resized()` function, we create two helper rectangles, one for the bank of sliders and one for the bank of labels. These are positioned to occupy the left half and right half of the main component respectively.

In the listener callback functions, the index of the broadcasting component is looked up in its array, and this index is used to set the value of the other corresponding component. Add these listener callback functions to the `MainComponent.cpp` file:

[PRE45]

Here, we use the `String` class to perform the numerical conversions. After moving some of the sliders, the application window should look similar to the following screenshot:

![Other banks of controls](img/3316_03_03.jpg)

Hopefully, these examples illustrate the power of combining JUCE array classes with other JUCE classes to write elegant, readable, and powerful code.

# Employing smart pointer classes

The `OwnedArray` class may be considered a manager of smart pointers, in the sense that it manages the lifetime of the object to which it points. JUCE includes a range of other smart pointer types to help solve a number of common issues when writing code using pointers. In particular, these help avoid mismanagement of memory and other resources.

Perhaps the simplest smart pointer is implemented by the `ScopedPointer` class. This manages a single pointer and deletes the object to which it points when no longer needed. This may happen in two ways:

*   When the `ScopedPointer` object itself is destroyed
*   When a new pointer is assigned to the `ScopedPointer` object

One use of the `ScopedPointer` class is as an alternative means of storing a `Component` objects (or one of its subclasses). In fact, adding subcomponents in the Introjucer applications graphical editor adds the components to the code as `ScopedPointer` objects in a similar way to the example that follows. Create a new Introjucer project named `Chapter03_05`. The following example achieves an identical result to the `Chapter02_02` project, but uses `ScopedPointer` objects to manage the components rather than statically allocating them. Change the `MainComponent.h` file to:

[PRE46]

Notice that we use a `ScopedPointer<Button>` object rather than a `ScopedPointer<TextButton>` object for the same reasons we used an `OwnedArray<Button>` object in preference to an `OwnedArray<TextButton>` object previously. Change the `MainComponent.cpp` file as follows:

[PRE47]

The main changes here are to use the `->` operator (which the `ScopedPointer` class overloads to return the pointer it contains) rather than the `.` operator. The components are all explicitly allocated use the `new` operator, but other than that, the code is almost identical to the `Chapter02_02` project.

Other useful memory management classes in JUCE are:

*   `ReferenceCountedObjectPtr<ReferenceCountedObjectClass>`: This allows you to write classes such that instances can be passed around in a similar way to the `String` objects. The lifetime is managed by the object maintaining its own counter that counts the number of references that exists to the object in the code. This is particularly useful in multi-threaded applications and for producing graph or tree structures. The `ReferenceCountedObjectClass` template argument needs to inherit from the `ReferenceCountedObject` class.
*   `MemoryBlock`: This manages a block of resizable memory and is the recommended method of managing raw memory (rather than using the standard `malloc()` and `free()` functions, for example).
*   `HeapBlock<ElementType>`: Similar to the `MemoryBlock` class (in fact a `MemoryBlock` object contains a `HeapBlock<char>` object), but this is a smart pointer type and supports the `->` operator. As it is a template class, it also points to an object or objects of a particular type.

# Summary

This chapter has outlined some of the core classes in JUCE that provide a foundation for building JUCE applications and provide a framework for building applications that are idiomatic to the JUCE style. These classes provide further foundations for the remainder of this book. Each of these classes contains far more functionality than outlined here. Again, it is essential that you review the JUCE class documentation for each of the classes introduced in this chapter. Many of these classes are used heavily in the JUCE Demo application and the code for the Introjucer application. These should also serve as useful for further reading. The next chapter introduces classes for handling files, especially media files, such as image and sound files.