# Chapter 13. Improving your C Programming and Creating Libraries

This is the last chapter of this book and is the most advanced, but not the most complex. You will learn about C code optimization through several typical examples that will bring you a bit further and make you more capable for your future projects using Arduino. I am going to talk about libraries and how they can improve the reusability of your code to save time in the future. I will describe some tips to improve the performance of your code by using bit-shifting instead of the usual operators, and by using some memory management techniques. Then, I will talk about reprogramming the Arduino chip itself and debugging our code using an external hardware programmer.

Let's go.

# Programming libraries

I have already spoken about libraries in [Chapter 2](ch02.html "Chapter 2. First Contact with C"), *First Contact with C*. We can define it as a set of implementations of behavior already written using a particular language that provides some interfaces by which all the available behaviors can be called.

Basically, a library is something already written and reusable in our own code by following some specifications. For example, we can quote some libraries included in the Arduino core. Historically, some of those libraries had been written independently, and over time, the Arduino team as well as the whole Arduino community incorporated them into the growing core as natively available libraries.

Let's take the EEPROM library. In order to check files related to it, we have to find the right folder on our computer. On OS X, for instance, we can browse the contents of the `Arduino.app` file itself. We can go to the `EEPROM` folder in `Contents`/`Resources`/`Java`/`libraries`/. In this folder, we have three files and a folder named `examples` containing all the examples related to the EEPROM library:

![Programming libraries](img/7584_13_001.jpg)

The EEPROM library on our computer (an OS X system)

We have the following files:

*   `EEPROM.h`, containing the headers of the library
*   `EEPROM.cpp`, containing the code itself
*   `keywords.txt`, containing some parameters to color the keywords of the library

Because of the location of these files in the folder hierarchy, they are available as parts of the core EEPROM library. This means we can include this library as soon as we have the Arduino environment installed on our computer without downloading anything else.

The simple statement `include <EEPROM.h>` includes the library in our code and makes all the features of this library available for further use.

Let's enter code in these files.

## The header file

Let's open `EEPROM.h`:

![The header file](img/7584_13_002.jpg)

EEPROM.h displayed in Xcode IDE

In this file, we can see some preprocessor directives starting with the `#` character. This is the same one that we use to include libraries in our Arduino code. Here, this is a nice way to not include the same header twice. Sometimes, while coding, we include a lot of libraries and at compilation time, we would have to check that we didn't include the same code twice. These directives and especially the `ifndef` directive mean: "If the `EEPROM_h` constant has not been defined, then do the following statements".

This is a trick commonly known as **include guards**. The first thing we are doing after this test is defining the `EEPROM_h` constant. If in our code we or some other libraries include the EEPROM library, the preprocessor wouldn't reprocess the following statements the second time it sees this directive.

We have to finish the `#ifndef` directive with the `#endif` directive. This is a common block in the header files and you'll see it many times if you open other library header files files. What is contained inside this block? We have another inclusion related to C integer types: `#include <inttypes.h>`.

The Arduino IDE contains all the required C headers in the library. As we have already mentioned, we could use pure C and C++ code in our firmware. We didn't until now because the functions and types we've been using have already been coded into the Arduino core. But please keep in mind that you have the choice to include other pure C code in your firmware and in this last chapter, we will also talk about the fact you can also follow pure AVR processor-type code too.

Now we have a class definition. This is a C++ feature. Inside this class, we declare two function prototypes:

*   `uint8_t read(int)`
*   `void write(int, uint8_t)`

There is a function to read something, taking an integer as an argument and returning an unsigned integer that is 8 bits long (which is a byte). Then, there is another function to write something that takes an integer and a byte and returns nothing. These prototypes refer to the definition of these functions in the other `EEPROM.cpp` file.

## The source file

Let's open `EEPROM.cpp`:

![The source file](img/7584_13_003.jpg)

The source file of the EEPROM library is displayed in the Xcode IDE

The file begins by including some headers. `avr/eeprom.h` refers to the AVR type processor's EEPROM library itself. In this library example, we just have a library referring to and making a better interface for our Arduino programming style than the original pure AVR code. This is why I chose this library example. This is the shortest but the most explicit example, and it teaches us a lot.

Then we include the `Arduino.h` header in order to have access to standard types and constants of the Arduino language itself. At last, of course, we include the header of the EEPROM library itself.

In the following statements, we define both functions. They call other functions inside their block definition:

*   `eeprom_read_byte()`
*   `eeprom_write_byte()`

Those functions come from the AVR EEPROM library itself. The EEPROM Arduino library is only an interface to the AVR EEPROM library itself. Why wouldn't we try to create a library ourselves?

# Creating your own LED-array library

We are going to create a very small library and test it with a basic circuit including six LEDs that are not multiplexed.

## Wiring six LEDs to the board

Here is the circuit. It basically contains six LEDs wired to Arduino:

![Wiring six LEDs to the board](img/7584_13_005.jpg)

Six LEDs wired to the board

The circuit diagram is shown as follows:

![Wiring six LEDs to the board](img/7584_13_006.jpg)

Another diagram of the six LEDs wired directly to Arduino

I won't discuss the circuit itself, except to mention that I put in a 1 kΩ resistor. I took the worst case where all LEDs would be switched on at the same time. This would drive a lot of current, and so this acts as security for our Arduino. Some authors wouldn't use it. I'd prefer to have some LEDs dimming a bit in order to protect my Arduino.

## Creating some nice light patterns

Here is code for lighting up the LEDs according to some patterns, all hardcoded. A pause is made between each pattern display:

[PRE0]

This code works correctly. But how could we make it more elegant and, especially, more reusable? We could embed the `for()` blocks into functions. But these would only be available in this code. We'd have to copy and paste them by remembering the project in which we designed them in order to reuse them in another project.

By creating a small library that we can use over and over again, we can save time in the future in coding as well as processing. With some periodic modifications, we can arrive at the perfect module for its intended task, which will get better and better until there's no need to even touch it anymore because it performs more perfectly than anything else out there. At least that's what we hope for.

## Designing a small LED-pattern library

At first, we can design our function's prototype in a header. Let's call the library `LEDpatterns`.

### Writing the LEDpatterns.h header

Here is how a possible header could be:

[PRE1]

We first write our include guards. Then we include the Arduino library. Then, we define a class named `LEDpatterns` with the `public` functions including a constructor that has the same name as the class itself.

We also have two internal (`private`) variables related to the first pin on which LEDs are wired and related to the total number of LEDs wired. LEDs would have to be contiguously wired in that example.

### Writing the LEDpatterns.cpp source

Here is the source code of the C++ library:

[PRE2]

At the beginning, we retrieve all the `include` libraries. Then we have the constructor, which is a special method with the same name as the library. This is the important point here. It takes two arguments. Inside its body, we put all the pins from the first one to the last one considering the LED number as a digital output. Then, we store the arguments of the constructor inside the `private` variables previously defined in the header `LEDpatterns.h`.

We can then declare all our functions related to those created in the first example without the library. Notice the `LEDpatterns::` prefix for each function. I won't discuss this pure class-related syntax here, but keep in mind the structure.

### Writing the keyword.txt file

When we look at our source code, it's very helpful to have things jump out at you, and not blend into the background. In order to correctly color the different keywords related to our new created library, we have to use the `keyword.txt` file. Let's check this file out:

[PRE3]

In the preceding code we can see the following:

*   Everything followed by `KEYWORD1` will be colored in orange and is usually for classes
*   Everything followed by `KEYWORD2` will be colored in brown and is for functions
*   Everything followed by `LITERAL1` will be colored in blue and is for constants

It is very useful to use these in order to color your code and make it more readable.

## Using the LEDpatterns library

The library is in the `LEDpatterns` folder in `Chapter13` and you have to put it in the correct folder with the other libraries, which we have done. We have to restart the Arduino IDE in order to make the library available. After having done that, you should be able to check if it is in the menu **Sketch** | **Import Library**. `LEDpatterns` is now present in the list:

![Using the LEDpatterns library](img/7584_13_007.jpg)

The library is a contributed one because it is not part of the Arduino core

Let's now check the new code using this library. You can find it in the `Chapter13`/`LEDLib` folder:

[PRE4]

In the first step, we include the `LEDpatterns` library. Then, we create the instance of `LEDpatterns` named `ledpattern`. We call the constructor that we designed previously with two arguments:

*   The first pin of the first LED
*   The total number of LEDs

`ledpattern` is an instance of the `LEDpatterns` class. It is referenced throughout our code, and without `#include`, it would not work. We have also invoked each method of this instance.

If the code seems to be cleaner, the real benefit of such a design is the fact that we can reuse this library inside any of our projects. If we want to modify and improve the library, we only have to modify things in the header and the source file of our library.

# Memory management

This section is a very short one but not a less important one at all. We have to remember we have the following three pools of memory on Arduino:

*   Flash memory (program space), where the firmware is stored
*   **Static Random Access Memory** (**SRAM**), where the sketch creates and manipulates variables at runtime
*   EEPROM is a memory space to store long-term information

Flash and EEPROM, compared to SRAM, are non-volatile, which means the data persists even after the power is turned off. Each different Arduino board has a different amount of memory:

*   ATMega328 (UNO) has:

    *   Flash 32k bytes (0.5k bytes used by the bootloader)
    *   SRAM 2k bytes
    *   EEPROM 1k bytes

*   ATMega2560 (MEGA) has:

    *   Flash 256k bytes (8k bytes used by the bootloader)
    *   SRAM 8k bytes
    *   EEPROM 4k bytes

A classic example is to quote a basic declaration of a string:

[PRE5]

That takes 32 bytes into SRAM. It doesn't seem a lot but with the UNO, you *only* have 2048 bytes available. Imagine you use a big lookup table or a large amount of text. Here are some tips to save memory:

*   If your project uses both Arduino and a computer, you can try to move some calculation steps from Arduino to the computer itself, making Arduino only trigger calculations on the computer and request results, for instance.
*   Always use the smallest data type possible to store values you need. If you need to store something between 0 and 255, for instance, don't use an `int` type that takes 2 bytes, but use a `byte` type instead
*   If you use some lookup tables or data that won't be changed, you can store them in the Flash memory instead of the SRAM. You have to use the `PROGMEM` keyword to do that.
*   You can use the native EEPROM of your Arduino board, which would require making two small programs: the first to store that information in the EEPROM, and the second to use it. We did that using the PCM library in the [Chapter 9](ch09.html "Chapter 9. Making Things Move and Creating Sounds"), *Making Things Move and Creating Sounds*.

# Mastering bit shifting

There are two bit shift operators in C++:

*   `<<` is the left shift operator
*   `>>` is the right shift operator

These can be very useful especially in SRAM memory, and can often optimize your code. `<<` can be understood as a multiplication of the left operand by 2 raised to the right operand power.

`>>` is the same but is similar to a division. The ability to manipulate bits is often very useful and can make your code faster in many situations.

## Multiplying/dividing by multiples of 2

Let's multiply a variable using bit shifting.

[PRE6]

The second row multiplies the variable `a` by `2` to the third power, so `b` now contains `32`. On the same lines, division can be carried out as follows:

[PRE7]

`b` contains `3` because `>> 2` equals division by 4\. The code can be faster using these operators because they are a direct access to binary operations without using any function of the Arduino core like `pow()` or even the other operators.

## Packing multiple data items into bytes

Instead of using a big, two-dimensional table to store, for instance, a bitmap shown as follows:

[PRE8]

We can use use the following code:

[PRE9]

In the first case, it takes 7 x 5 = 35 bytes per bitmap. In the second one, it takes only 5 bytes. I guess you've just figured out something huge, haven't you?

## Turning on/off individual bits in a control and port register

The following is a direct consequence of the previous tip. If we want to set up pins 8 to 13 as output, we could do it like this:

[PRE10]

But this would be better:

[PRE11]

In one pass, we've configured the whole package into one variable directly in memory, and no `pinMode` function, structure, or variable name needs to be compiled.

# Reprogramming the Arduino board

Arduino natively uses the famous bootloader. This provides a nice way to upload our firmware using the virtual serial port on the USB. But we might be interested to go ahead without any bootloader. How and why? Firstly, that would save some Flash memory. It also provides a way to avoid the small delay when we power on or reset our board before it becomes active and starts running. It requires an external programmer.

I can quote the AVR-ISP, the STK500, or even a parallel programmer (a parallel programmer is described at [http://arduino.cc/en/Hacking/ParallelProgrammer](http://arduino.cc/en/Hacking/ParallelProgrammer)). You can find an AVR-ISP at Sparkfun Electronics.

I used this one a couple of times to program an Arduino FIO-type board for specific wireless applications in a project connecting cities named The Village in 2013.

![Reprogramming the Arduino board](img/7584_13_008.jpg)

The Pocket AVR programmer by Sparkfun Electronics

This programmer can be wired using 2 x 5 connectors to the ICSP port on the Arduino board.

![Reprogramming the Arduino board](img/7584_13_009.jpg)

The ICSP connector of Arduino

In order to reprogram the processor of Arduino, we have to first close the Arduino IDE, and then check the preferences file (`preferences.txt` on a Mac, located in `Contents`/`Resources`/`Java`/`lib` inside the `Arduino.app` package itself). On a Windows 7 PC and higher, this file is located at: `c:\Users\<USERNAME>\AppData\Local\Arduino\preferences.txt`. In Linux it is located at: `~/arduino/preferences.ard`.

We have to change the `upload.using` value that is initially set to bootloader to the correct identifier that fits your programmer. This can be found in the content of the Arduino application package on OS X or inside the Arduino folders on Windows. For instance, if you display the `Arduino.app` content, you can find this file: `Arduino.app/Contents/Resources/Java/hardware/arduino/programmers.txt`.

Then we can start the Arduino IDE to upload the sketch using our programmer. To revert back to the normal bootloader behavior, we have to first reupload the bootloader that fits with our hardware. Then, we have to change back the `preferences.txt` file, and it will work as the initial board.

# Summary

In this chapter, we learned more about designing libraries, and we are now able to design our projects a bit differently, keeping in mind reusability of the code or part of the code in future projects. This can save time and also improves readability.

We can also explore existing libraries and enjoy the world of open source by taking them, hacking them, and making them fit our needs. This is a really open world into which we have just made our first steps.

# Conclusion

We are at the end of this book. You have probably read everything and also tested some pieces of code with your own hardware, and I'm sure you are now able to imagine your future and advanced projects with Arduino.

I wanted to thank you for being so focused and interested. I know you are now almost in the same boat as myself, you want to learn more, test more, and check and use new technologies in order to achieve your craziest project. I'd like to say one last thing: do it, and do it now!

In most cases, people are afraid of the huge amount of work that they can imagine in the first steps just before they start. But you have to trust me, don't think too much about details or about optimization. Try to make something simple, something that works. Then you'll have ways to optimize and improve it.

One last piece of advice for you: don't think too much, and make a lot. I have seen too many unfinished projects by people having wanted to think, think, think instead of just starting and making.

Take care and continue exploring!