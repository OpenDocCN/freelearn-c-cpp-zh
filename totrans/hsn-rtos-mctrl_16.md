# Tips for Creating a Well-Abstracted Architecture

 Throughout this book, simple examples of minimal complexity have been provided. Our focus has been to keep the code clear and readable to illustrate the particular **real-time operating system** (**RTOS**) concepts being addressed and keep the interactions with hardware as easily understood as possible. However, in the real world, the best code bases for long-term development are those that allow developers to move quickly with great flexibility and determination to meet targets. This chapter provides suggestions on how to go about architecting, creating, growing, and maintaining a code base that will be flexible enough for long-term use. We'll be exploring these concepts with real code by cleaning up some of the code developed in earlier chapters through the addition of flexibility and better portability to different hardware.

This chapter is valuable to anyone interested in reusing code across multiple projects. While the concepts presented here are by no means original, they are focused solely on firmware in embedded systems. The concepts covered are applicable to bare-metal systems, as well as highly reusable RTOS task-based systems. By following the guidelines here, you'll be able to create a flexible code base that adapts to many different projects, regardless of what hardware it happens to be running on. Another side effect (or direct intention) of architecting a code base in this manner is extremely testable code.

In this chapter, we'll cover the following topics:

*   Understanding abstraction
*   Writing reusable code
*   Organizing source code

# Technical requirements

To run the code introduced in this chapter, you will need the following:

*   A Nucleo F767 development board
*   A micro-USB cable
*   STM32CubeIDE and source code (instructions in [Chapter 5](84a945dc-ff6c-4ec8-8b9c-84842db68a85.xhtml), *Selecting an IDE*, under the *Setting up our IDE* section)
*   SEGGER J-Link, Ozone, and SystemView (instructions in [Chapter 6](699daa80-06ae-4acc-8b93-a81af2eb774b.xhtml), *Debugging Tools for Real-Time Systems*)

All source code for this chapter is available at [https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_12](https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_12).

# Understanding abstraction

If our goal is to create a code base that will be usable for a long time, we need flexibility. Source code (just like product feature sets and business tactics) isn't chiseled out of rock—it tends to morph into different forms over time. If our source code is to be flexible, it needs to be able to morph and adapt. Only then will it be able to provide a solid foundation for implementing different feature sets of a product (or entire product lines) as the business landscape driving its development changes. Abstraction is a core tenet of flexibility.

In our context, abstraction means representing a single instance of a complex implementation with a representation that can be applied to many different instances. For example, let's take another look at an earlier example from [Chapter 1](39404421-bf7a-4283-bf78-c396818be4b4.xhtml), *Introducing Real-Time Systems*:

![](img/b20cbc53-cee9-43a5-90d9-baab6fd2a939.png)

The diagram itself is an abstract representation of the hardware required for a closed-loop control system. The exact part numbers of the ADC, driver circuitry, and **microcontroller unit** (**MCU**) aren't shown in the diagram; they could be almost anything. 

There are at least two primary reasons for using abstractions when creating a flexible code base:

*   Grasping an abstraction is fast.
*   Abstractions provide flexibility.

# Grasping an abstraction is fast

Understanding a well-written abstraction in code is similar to understanding a simple flow chart. Just as you don't need to understand every interconnect and resistor value when observing a flowchart (versus a schematic), reading through a well-commented header file of an abstraction provides nearly all of the information required to use any of the underlying implementations. There is no need to get buried under the details and idiosyncrasies of each implementation.

This limited bird's-eye view means that future developers are more likely to *consume* the code since it is presented in a well-defined, well-documented, and consistent manner. The overall knowledge and time required to grasp an abstraction is much less than that required if implementing the same functionality from scratch.

# An example with abstraction

If you saw a call to the following function call, you would probably have a fair chance at guessing what the function did, even without any proper comments:

```cpp
bufferX[i] = adcX->ReadAdcValue();
bufferY[i] = adcY->ReadAdcValue();
bufferZ[i] = adcZ->ReadAdcValue();
```

The preceding code is fairly self-explanatory—we're reading ADC values and storing them in 3 different buffers. If all of our calls to get ADC readings use the same `ReadAdcValue()` calling convention and descriptively name the ADC channel, understanding the code is fast and easy.

# An example without abstraction

On the flip side, imagine that instead you were given the following lines of code (they are functionally equivalent to the preceding code):

```cpp
bufferX[i] = adc_avg(0, 1);
bufferY[i] = adc_avg(1, 1);
bufferZ[i] = HAL_ADC_GetValue(adc2_ch0_h);
```

This immediately raises several questions, such as what the arguments being passed into `adc_avg()` and `HAL_ADC_GetValue()` are. At a minimum, we'd likely need to track down the relevant function prototypes and read through them:

```cpp
/**
  * return an average of numSamp samples collected
  * by the ADC who's channel is defined by chNum
  * @param chNum channel number of the ADC 
  * @param numSamp number of samples to average
  * @retval avera 
**/
uint32_t adc_avg(uint8_t chNum, uint16_t numSamp);
```

OK, so `adc_avg()` takes an ADC channel as the first parameter and the number of samples to average as the second parameter—passing `1` to the second parameter provides a single reading. Now, what about this other call to `HAL_ADC_GetValue(adc2_ch0_h)`? We'd better go find the prototype for it:

```cpp
/**
  * @brief Gets the converted value from 
  * data register of regular channel.
  * @param hadc pointer to a ADC_HandleTypeDef 
  * structure that contains
  * the configuration information for the 
  * specified ADC.
  * @retval Converted value
  */
uint32_t HAL_ADC_GetValue(ADC_HandleTypeDef* hadc)
```

It turns out `adc2_ch0_h` is a handle—probably to channel `0` on the ADC2 STM32 peripheral... now, where's that schematic... is everything wired properly? Should channel `0` really be stored in `bufferZ`? That seems a little odd... 

OK, so this might be a *bit* contrived, but if you've been coding long enough, you've likely seen far worse. The takeaway here is that the consistency provided by a good abstraction makes reading code faster and easier than attempting to track down and understand the details of each specific implementation.

# Abstractions provide flexibility

Since a proper abstraction isn't directly tied to an implementation, creating an abstraction for functionality provides flexibility in the way the functionality is implemented, even though the interface to that functionality is consistent. In the following figure, there are five different physical implementations of an ADC value—all represented by the same, simple abstraction, which is `int32_t ReadAdcValue( void );`:

![](img/caae5a5d-4f82-468f-a0c5-a006adb7209e.png)

Although the function call remains consistent, there can be drastically different implementations of the ADC. In this diagram alone, there are five different ways for an ADC to provide data through the `ReadAdcValue` function. The ADC could be sitting on a local communication bus, such as I2C, SPI, or UART. It could be an internal ADC that is present on the MCU itself. Alternatively, the ADC reading may be coming from a remote node of an external network. Since there is a consistent, abstract interface, the underlying implementation isn't all that significant. A consumer of the interface doesn't need to be concerned with all of the details required to configure the ADC, collect the reading, and so on; the ADC only needs to make a call to `ReadAdcValue` to access the most up-to-date reading.

There are, of course, many considerations to be made here, such as how recent the reading is, how quickly it must be collected, the resolution and scaling of the underlying reading, and so on. These types of details need to be provided by each provider and consumer that is implementing an abstraction. Naturally, there are cases where this level of abstraction is not appropriate for various reasons, which need to be evaluated on a case-by-case basis. If, for example, an algorithm needs to be run each time a new reading is taken, having it blindly poll `ReadAdValue` asynchronously won't work reliably.

There are many examples of abstraction in the real world. Let's say you're a developer in an organization that makes many different products that all incorporate similar core components. For example, if you're designing a family of process controllers, you'll likely be interfacing with ADCs, DACs, and communication stacks. Each controller may have a slightly different user-facing feature set, but the underlying core components could be shared. Drivers for the ADCs, DACs, and algorithms can all share common code. By sharing common code across multiple products, developers only need to invest their time into writing common code once. As the customer-facing feature sets change over time, individual components may be replaced as needed, as long as they are loosely coupled to one another. Even the underlying MCU doesn't need to be the same, provided its hardware is sufficiently abstracted.

Let's take a closer look at the ADC in a controller as a specific example. The simplest way for a control algorithm to use ADC readings is to take raw readings from the device and use them directly. To reduce the number of source files, the drivers for the ADC, communication peripheral, and algorithm *could* all be combined into a single source file.

Note that for precision applications, there are many issues with using raw readings directly, even without worrying about code elegance and abstraction. Ensuring consistent scaling and offsets and providing a flexible amount of resolution are all easier when code does not interface to the raw units (ADC counts) directly.

When code space and RAM are at a premium, or a quick and dirty  one-off, or a proof of concept are all that is desired, this approach might be acceptable. The resulting architecture might look something like the following:

![](img/d45447e5-da94-4c21-8ca4-0246d88c8294.png)

A few things should jump out when looking at this architecture:

*   The `algorithm.c` file is coupled with both the MCU and a specific ADC on a specific bus.
*   If either the MCU or the ADC changes, a new version of `algorithm.c` will need to be created.
*   The visual similarity between the links to the MCU and ADC ICs look very much like chains. This is not an accident. Code like this tightly binds whatever algorithm is inside `algorithm.c` to the underlying hardware in a way that is very inflexible.

There are also a few side effects that might not be as obvious:

*   `algorithm.c` will be very difficult (maybe even impossible) to run independently of the hardware. This makes it very hard to test the algorithm in isolation. It also makes it very difficult to test all of the corner cases and error conditions that only occur when something goes wrong in the hardware.
*   The immediate, useful life of `algorithm.c` will be limited to this single MCU and specific ADC. To add support for additional MCUs or ADC ICs, `#define` functions will need to be used; otherwise, the entire file will need to be copied and modified.

On the other hand, `algorithm.c` could be written so it doesn't rely directly on the underlying hardware. Instead, it could rely on an abstract interface to the ADC. In this case, our architecture looks more like this:

![](img/600d8f23-0a45-4f92-9130-1852ad7f10e0.png)

The core points to observe in this variation are as follows:

*   `algorithm.c` has no direct reliance on any specific hardware configuration. Different ADCs and MCUs can be used interchangeably, assuming they correctly implement the interface required. This means it could move to an entirely different platform and be used as is, without modification.
*   The chains have been replaced by ropes, which *tie *together abstractions with their underlying implementations, rather than tightly bind `algorithm.c` to the underlying hardware.
*   Only the implementations are tightly bound to the hardware.

Less obvious points that are also worth mentioning are as follows:

*   `ADC Driver` isn't *completely *coupled to the hardware. While this particular driver will probably only support a single ADC, the ADC hardware itself won't be necessary for getting the code to work. The hardware can be imitated by simulating SPI traffic. This allows testing the ADC driver independently of the underlying hardware.
*   Both `SPI Driver` and `ADC Driver` can be used in other applications without rewriting them. This is a really big advantage to writing reusable code; it is flexible enough to repurpose with no additional work (or side effects).

Now that we have a few examples of abstraction covered, let's consider why using abstractions may be important for projects.

# Why abstraction is important

It is important to ensure your architecture is using abstractions if the following points apply:

*   Common components will be reused in other projects.
*   Portability to different hardware is desirable.
*   Code will be unit tested.
*   Teams will be working in parallel

For projects that are part of a larger code base, all four of the preceding points are generally desirable, since they all contribute to decreased time to market in the medium term. They also lead to decreased long-term maintenance costs for the code base:

*   It is easier to create quality documentation once for the abstraction, rather than to thoroughly document every intricate piece of spaghetti code that reimplements the same functionality in a slightly different way.
*   Abstractions provide ways of cleanly decoupling hardware from many of the other interfaces used in a project.
*   Abstracting hardware interfaces makes unit testing code much easier (allowing programmers to run unit tests on their development machine, instead of on the target hardware).
*   Unit tests are similar to a type of documentation that's always up to date (if they are run regularly). They provide a source of truth for what the code is intended to do. They also provide a safety net when making changes or providing new implementations, ensuring nothing has been forgotten or inadvertently changed.
*   Consistent abstractions lead to code bases that are more quickly understood by new team members. Each project in a code base is slightly more familiar than the last one since there's a large amount of commonality and consistency between them.
*   Loosely coupled code is easier to change. The mental burden for understanding a well-encapsulated module is much lower than trying to understand a sprawling implementation spanning multiple portions of a project. Changes to the well-encapsulated module are more likely to be made correctly and without side effects (especially when unit testing is employed). 

When abstractions are not used, the following symptoms commonly occur:

*   New developers have a hard time making changes since each change has a ripple effect.
*   It takes new developers a long time to understand a piece of code well enough to be comfortable changing it.
*   Parallel development is very difficult.
*   Code is tightly coupled to a specific hardware platform.

For a real-world example where abstraction is required, we don't need to look any further than FreeRTOS itself. FreeRTOS wraps all of the device-specific functionality in two files, `port.c` and `portmacros.h`. To add support to a new hardware platform, only these files need to be created/modified. All of the other files that make up FreeRTOS have only a single copy, shared across dozens of ports for different hardware platforms. Libraries such as FatFs, lwIP, and many others also make use of hardware abstractions; it is the only way they can reasonably provide support for a large range of hardware.  

# Recognizing opportunities to reuse code

There is no absolute rule to follow when determining whether formalized abstractions should be used (if abstractions are not already present). However, there are some hints, which are as follows:

*   **If you're writing code that can be used by more than one project**: Interfacing with the underlying hardware should be done through an abstraction (the ADC driver and algorithm in the preceding section are examples of this).  Otherwise, the code will be tied to the specific piece of hardware it was written for.
*   **If your code interacts with a vendor-specific API**: Creating a light abstraction layer above it will reduce vendor lock-in. After interfaces are commonly used and set up, you'll start to gravitate toward making vendor-specific APIs conform to your code base, which makes trying out different implementations quick and easy. It also insulates the bulk of your code from changes the vendor might make to the API over time.
*   **If the module is in the center of the stack and interacts with other sub-modules**: Using formalized interfaces will reduce the coupling to the other modules, making it easier to replace them in the future.

One common misconception around code reuse is that creating a copy of code is the same as reusing the code. If a copy of the code has been created, it is effectively not being reused—let's look at why.

# Avoiding the copy-paste-modify trap

So, we've got a piece of code that has been proven to work well and we have a new project coming up. How should we go about creating the new project—just copy the working project and start making changes? After all, if the code is being copied, it is being reused, right? Creating copies of a code base like this can inadvertently create a mountain of technical debt over time. The problem isn't the act of copying and modifying the code, it is trying to maintain all of the copies over time.

Here's a look at what a monolithic architecture for `algorithm.c` might look like over the course of six projects. Let's assume that that actual algorithm is intended to be identical across all six projects:

![](img/81931152-a848-4cbb-a4df-c611fce7ed0a.png)

Here are some of the main points in the diagram:

*   It is impossible to tell whether the actual algorithm being used is the same since there are six copies of the file.
*   In some cases, `algorithm.c` is implemented with different hardware. Since these changes were made in `algorithm.c`, it is isn't easily tell whether or not the algorithm implemented is actually the same without examining each file in detail.

Now, let's take a look at the *drawbacks* to copy-paste-modify in our example:

*   If `Algo` has a bug, it will need to be fixed in six different places.
*   Testing a potential fix for `Algo` will need to be validated separately for each project. The only way to tell if a fix corrected the bug is probably by testing on actual hardware "in-system"; this is probably a very time-intensive task and it is potentially technically difficult to hit all of the edge cases.
*   The forked `Algo` function will likely morph over time (possibly inadvertently); this will further complicate maintenance because examining the differences between implementations will be even more difficult.
*   Bugs are harder to find, understand, and fix because of all of the slight differences between the six projects.
*   Creating project 7 may come with a high degree of uncertainty (it is hard to tell exactly which features of `Algo` will be brought in, which intricacies/bugs from the `SPI` or `ADC` drivers will follow, and so on).
*   If `MCU1` goes obsolete, porting `algorithm.c` will need to happen four separate times.

All of these duplicates can be avoided by creating consistent reusable abstractions for the common components:

*   Each common component needs to have a consistent *interface*.
*   Any code that is meant to be reused uses the *interface* rather than the *implementation (*`Algo` would use an ADC interface).
*   Common drivers, interfaces, and middleware should only have one copy.
*   Implementations are provided by the use of **board support packages** (**BSPs**), which provide an implementation for required interfaces.

If the same algorithm were designed using the preceding guidelines, we might have something that looks more like this:

![](img/23f33de7-3134-4344-902e-d9579e703b55.png)

Here are some of the main points in the diagram:

*   There is only one copy of `algorithm.c`—it is immediately obvious that the algorithm used is identical across all six projects.
*   Even though there are six projects, there are only four BSP folders—`BSP1` has been reused across three projects.
*   An `ADC` interface is specified in a common location (Interfaces).
*   BSPs define an implementation of ADC, which is tied to specific hardware. These implementations are used by `main.c` and passed to `algorithm.c`.
*   The `ADC` interface, which is referenced by `Algo`, rather than a specific implementation. 
*   There is only one copy of the `I2C` and `SPI` drivers for `MCU1` and `MCU2`.
*   There is only one copy of the driver for the SPI-based ADC.
*   There is only one copy of the driver for the I2C-based ADC.

Reused code has the following advantages:

*   If `Algo` has a bug, it will only need to be fixed in one place.
*   Although final integration testing for `Algo` will still need to be performed in-system with real hardware (but probably only needs to be performed on the four BSP's, rather than all six projects), the bulk of the testing and development can be done by mocking the ADC interface, which is fast and simple.
*   It is impossible for `Algo` to morph over time since there is only one copy. It will always be trivial to see whether or not the algorithm used is different between projects.
*   Bugs are easier to find, understand, and fix due to the decreased interdependencies of the dependencies. A bug in `Algo` is guaranteed to show up in all six projects (since there is only one copy). However, it is less likely to occur, since testing `Algo` during development was easier, thanks to the interface.
*   Creating project 7 is likely to be fast and efficient with a high degree of certainty due to all of the consistency across the other six projects.
*   If `MCU1` goes obsolete, porting `algorithm.c` isn't even necessary since it has no direct dependency on an MCU—only the `ADC` interface. Instead, a different BSP will need to be selected/developed.

One exception to copy-paste-modify is *extremely* low-level code that needs to be written to support similar but different hardware. This is typically the driver-level code that directly interfaces with MCU peripheral hardware registers. When two MCU families share the same peripherals with only minor differences, it can be tempting to try and develop common code to implement them both, but this is often more confusing for everyone (both the original author and the maintenance developers). 

In these cases, it can be quite time-intensive and error-prone to force an existing piece of code to support a different piece of hardware, especially as the code ages and more hardware platforms are added. Eventually, if the code base becomes old enough, a new hardware target will vary significantly enough that it will no longer be remotely viable to incorporate those changes into the existing low-level code. As long as the low-level drivers are conforming to the same interface, they will still hold quite a bit of value in the long term. Keeping this low-level code easily understood and bug-free is the highest priority, followed by conforming to a consistent interface.

Now that we have a good idea of what abstraction is, let's take a closer look at some real-world examples of how to write code that can be easily reused.

# Writing reusable code

When you are first getting started with creating abstractions, it can be difficult to know exactly what should be abstracted versus what should be used directly. To make code fully reusable, a module should only perform one function and reference interfaces for the other pieces of functionality. Any hardware-specific calls must go through interfaces, rather than deal with the hardware directly. This is true for accessing actual hardware (such as specific pins) and also MCU-specific APIs (such as STM32 HAL).

# Writing reusable drivers 

There are a few different levels of drivers that are fairly common in embedded development. MCU peripheral drivers are the drivers used to provide a convenient API to hardware included on the MCU. These types of drivers were developed in [Chapter 10](dd741273-db9a-4e9a-a699-b4602e160b84.xhtml), *Drivers and ISRs*. Another commonly used driver is a driver for a specific IC, which is what was alluded to in the preceding ADC example:

![](img/7835c57e-ba6e-49f5-8136-64be3b84944e.png)

Peripheral drivers sit immediately above the hardware. IC drivers sit above (and often use) peripheral drivers in the stack. If an IC driver is meant to work across multiple MCUs, it must use interfaces that are completely agnostic to the underlying MCU hardware. For example, STM32 HAL can be thought of as a type of peripheral driver, but it does not provide MCU-independent abstractions for the peripherals. In order to create IC drivers that are portable across MCUs, they must only access MCU-independent interfaces.

# Developing an LED interface

To illustrate the initial concept in detail, let's take a look at a simple driver that we've been using since the first examples introduced in this book—an LED driver. A simplified version of an interface to drive the LEDs on our Nucleo board has been used since the very first examples in earlier chapters. This interface is located at `BSP\Nucleo_F767ZI_GPIO.c/h`***.*** This code fully abstracted the LEDs from the underlying hardware with a struct named `LED`. The `LED` struct has two function pointers: `On` and `Off`. As expected, the intention of these two functions is to turn an LED on and off. The beauty of this is that the calling code doesn't need to be concerned with the implementation of the LED at all. Each LED could have a completely different hardware interface. It might require positive or negative logic to drive an external transistor or be on a serial bus of some sort. The LED could even be on a remote panel requiring **remote procedure calls** (**RPCs**) to another board entirely. However, regardless of how the LED is turned on and off, the interface remains the same. 

To try and keep things simple, `Nucleo_F767ZI_GPIO.c/h` defined the LED struct in the header file. As we move through this current example, we'll extract the interface definition from the header file, making it completely standalone, requiring no external dependencies. The lack of dependencies will guarantee that we can move the new interface definition to entirely different platforms, without requiring any code specific to a particular MCU at all.

Our new, independent LED interface will be called `iLED`. 

The lowercase "i" is a convention used by some C++ programmers to indicate a class that only contains virtual functions, which is effectively an interface definition. Since we're only dealing with C in this book (not C++), we'll stick to structs and function pointers to provide the necessary decoupling. The methods outlined here are conceptually similar to pure virtual classes in C++.

The interface is defined in the new `Interfaces***/***iLed.h` file; the core of the contents is as follows:

```cpp
typedef void (*iLedFunc)(void);

typedef struct
{
    //On turns on the LED - regardless of the driver logic
 const iLedFunc On;

    //Off turns off the LED, regardless of the driver logic
 const iLedFunc Off;
}iLed;
```

Let's break down exactly what is going on in the preceding definition:

1.  We create a new type:`iLedFunc`. Now, `typedef void (*iLedFunc)(void);` defines the `iLedFunc` type as a function pointer to a function that takes no arguments and returns nothing.
2.  The `iLed` struct is defined as any other struct—we can now create instances of this struct. We're defining a struct so it is convenient to bundle together all of the function pointers and pass a reference to the structs around it.
3.  Each `iLedFunc` member is defined as `const` so it can only be set once at the time of definition. This protects us (or other developers) from accidentally overwriting the value of the function pointer (which can be potentially disastrous). The compiler will catch any attempts to write to the `On` or `Off` function pointers and throw an error.

It is extremely important that the header file defining the interface includes as few dependencies as possible to keep it as loosely coupled as possible. The more dependencies this file has, the less future flexibility there will be.

That does it for the interface definition. There is no functionality provided by the preceding code; it only defined an interface. In order to create an implementation of the `iLed` interface, we'll need two more files.

The following is an excerpt from `ledImplementation.h`:

```cpp
#include <iLed.h>
extern iLed BlueLed;
extern iLed GreenLed;
extern iLed RedLed;
```

This header file brings in the `iLed.h` interface definition and declares three instances of `iLed`, which are `BlueLed`, `GreenLed`, and `RedLed`. These implementations of the `iLed` interface can be used by any piece of code that includes `ledImplementation.h`***.  ***The `extern` keyword ensures only one copy is ever created, regardless of how many different code modules use `ledImplementation.h`.

Next, we need to provide definitions for the `iLed` instances; this is done in `ledImplementation.c`.

Only the code for `GreenLed` is shown here. The `BlueLed` and `RedLed` implementations only differ in the GPIO pin they set:

```cpp
void GreenOn ( void ) {HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_SET);}
void GreenOff ( void ) {HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0,
                                          GPIO_PIN_RESET);}
iLed GreenLed = { GreenOn, GreenOff };
```

Breaking it down, we can observe the following:

1.  `GreenOn` defines an inline function that turns on the green LED on our Nucleo development board. It takes no parameters and returns nothing, so it can be used as `iLedFunc`, as defined in the previous code.
2.  `GreenOff` defines an inline function that turns off the green LED on our Nucleo development board. It is can also be used as `iLedFunc`.
3.  An instance of `iLed` is created and named `GreenLed`. The `iLedFunc` function pointers `GreenOn` and `GreenOff` are passed in during initialization. The order of the functions defined in `iLed` is critical. Since `On` is defined in the `iLed` struct first, the first function pointer passed in (`GreenOn`) will be assigned to `On`.

The only code that relies on specific hardware so far is `ledImplementation.c`. 

A pointer to `GreenLed` can now be passed to different pieces of code that only bring in `iLed.h`—they won't be tied to `HAL_GPIO_WritePin` in any way. An example of this is `hardwareAgnosticLedDriver.c/h`.

The following is an excerpt from `hardwareAgnosticLedDriver.h`:

```cpp
#include <iLed.h>
void doLedStuff( iLed* LedPtr );
```

The only `include` function required by this hardware-agnostic driver is `iLed.h`.

For `hardwareAgnosticLedDriver.h` to be truly hardware agnostic, it must not include any hardware-specific files. It must only access hardware through hardware-independent interfaces, such as `iLed`.  

The following is a trivial example that simply turns a single LED on or off. The excerpt is from `hardwareAgnosticLedDriver.c`:

```cpp
void doLedStuff( iLed* LedPtr )
{
    if( LedPtr != NULL )
    {
        if(LedPtr->On != NULL)
        {
              LedPtr->On();
        }

        if( LedPtr->Off != NULL )
        {
              LedPtr->Off();
        }
    }
}
```

Breaking it down, we can observe the following:

1.  `doLedStuff` takes in a pointer to a variable of the `iLed` type as a parameter. This allows any implementation of the `iLed` interface to be passed in `doLedStuff`, which provides complete flexibility in how the `On` and `Off` functions are implemented without tying `hardwareAgnosticLedDriver` to any specific hardware.
2.  If your interface definition supports leaving out functionality by setting pointers to `NULL`, they will need to be checked to ensure they are not set to `NULL`. Depending on the design, these checks might not be necessary since the values for `On` and `Off` are only able to be set during initialization.
3.  The actual implementations of `On` and `Off` are called by using the `LedPtr` pointer and calling them like any other function.

A full example using `doLedStuff` is found in `mainLedAbstraction.c`:

```cpp
#include <ledImplementation.h>
#include <hardwareAgnosticLedDriver.h>

HWInit();

while(1)
{
    doLedStuff(&GreenLed);
    doLedStuff(&RedLed);
    doLedStuff(&BlueLed);
}
```

Breaking it down, we can observe the following:

1.  The implementations for `GreenLed`, `RedLed`, and `BlueLed` are brought in by including `ledImplementation.h`.
2.  `doLedStuff` is brought in by including `hardwareAgnosticLedDriver.h`.
3.  We provide the implementation for `doLedStuff` by passing in a pointer to the desired instance of `iLed`. In this example, we're toggling each of the green, red, and blue LEDs on the development board by passing the `GreenLed`, `RedLed`, and `BlueLed` implementations to `doLedStuff`.

This example simply toggled the single LEDs, but the complexity is arbitrary. By having well-defined interfaces, tasks can be created that take in pointers to instances of the interface. The tasks can be reused across multiple projects without touching them all—only a new implementation of the interface needs to be created when support for new hardware is required. When there is a considerable amount of code implemented by the hardware-agnostic task, this can dramatically decrease the total amount of time spent on the project.

Let's take a look at a simple example of passing instances of interfaces into tasks.

# Reusing code containing tasks

RTOS tasks are among the best suited for reuse because (when well-written) they offer single-purpose functionality that can be easily prioritized against the other functions the system must perform. In order for them to be easily reused in the long term, they need to have as few direct ties to the underlying platform as possible. Using interfaces as described previously works extremely well for this purpose since the interface fully encapsulates the desired functionality while decoupling it from the underlying implementation. To further ease the setup of the FreeRTOS task, the creation of the task can be wrapped inside an initialization function. 

`mainLedTask.c` uses `ledTask.c/h` to show an example of this. The following excerpt is from `ledTask.h`:

```cpp
#include <iLed.h>
#include <FreeRTOS.h>
#include <task.h>

TaskHandle_t LedTaskInit( iLed* LedPtr, uint8_t Priority, uint16_t
                                                       StackSize);
```

A few significant notes on this simple header file are as follows:

*   The only files included are those required for FreeRTOS and `iLed.h`, none of which are directly dependent on any specific hardware implementation.  
*   The priority of the task is brought in as an argument of the initialization function. This is important for flexibility because over time, LED tasks are likely to require different priorities relative to the rest of the system.
*   `StackSize` is also parameterized—this is required because, depending on the underlying implementation of `LedPtr`, the resulting task may need to use different amounts of stack space.
*   `LedTaskInit` returns `TaskHandle_t`, which can be used by the calling code to control or delete the resulting task.

`ledTask.c` contains the definition of `LedTaskInit`:

```cpp
TaskHandle_t LedTaskInit(iLed* LedPtr, uint8_t Priority, uint16_t StackSize)
{
  TaskHandle_t ledTaskHandle = NULL;
  if(LedPtr == NULL){while(1);}
  if(xTaskCreate(ledTask, "ledTask", StackSize, LedPtr, Priority, 
                 &ledTaskHandle) != pdPASS){while(1);}

  return ledTaskHandle;
}
```

This initialization function performs the same functions as what we've typically seen in `main`, but now, it is neatly encapsulated into a single file, which can be used across multiple projects. Functions taken care of by `LedTaskInit` include the following:

*   Checking that `LedPtr` is not `NULL`.
*   Creating a task that runs the `ledTask` function and passing `LedPtr` to it, which provides a specific implementation of the `iLed` interface for that `ledTask` instance. `ledTask` is created with the specified `Priority` task and `StackSize`.
*   Verifying whether it has been created successfully before `LedTaskInit` returns the handle to the task that was created.

`ledTask.c` also contains the actual code for `ledTask`:

```cpp
void ledTask( void* LedPtr)
{
 iLed* led = (iLed*) LedPtr;
  while(1)
  {
 led->On();
    vTaskDelay(100);
 led->Off();
    vTaskDelay(100);
  }
}

```

First, `LedPtr` needs to be cast from `void*` into `iLed*`. After this cast, we are able to call the functions of our `iLed` interface. The underlying hardware calls will depend on the implementation of `LedPtr`. This is also the reason for allowing a `StackSize` variable during initialization—`LedPtr` may have a more complex implementation in some cases, which could require a larger stack.

Thanks to `LedTaskInit`, creating tasks that map the `LedPtr` implementations into the task is extremely easy.

The following is an excerpt from `mainLedTask.c`:

```cpp
int main(void)
{
  HWInit();
  SEGGER_SYSVIEW_Conf();
  //ensure proper priority grouping for freeRTOS
  HAL_NVIC_SetPriorityGrouping(NVIC_PRIORITYGROUP_4);
  LedTaskInit(&GreenLed, tskIDLE_PRIORITY+1, 128);
 LedTaskInit(&BlueLed, tskIDLE_PRIORITY+2, 128);
 LedTaskInit(&RedLed, tskIDLE_PRIORITY+3, 128);
  vTaskStartScheduler();
```

`GreenLed`, `BlueLed`, and `RedLed` are passed into `LedTaskInit` to create three independent tasks with varying priorities and potentially different stack sizes. All of the hardware-specific code has been kept out of `ledTask.c/h`. When this technique is used for complex tasks, significant time savings and increased confidence can be realized. Along the lines of increasing confidence in the code we write, let's take a quick look at exactly how providing an abstracted interface makes testing tasks easier.

# Testing flexible code

Since the `iLed` interface isn't directly reliant on any hardware, it is extremely easy to push alternative implementations to `ledTask`. Rather than passing in one of the actual hardware implementations for `iLed`, we could pass in anything we like to either `LedTaskInit` (for integration-level tests) or `ledTask` (for unit tests). The implementations, in these cases, would likely set variables in the testing environment when called. For example, `On` could set a Boolean to `TRUE` when called and `Off` could set the same Boolean to `FALSE`. These types of tests can be used to verify the logic of the task, without requiring any hardware at all, provided a compiler and an alternative environment is set up on the development machine. FreeRTOS ports exist for desktop OSes that allow testing relative priorities of tasks (without any real-time guarantees). Specific timing dependencies aren't able to be tested this way, but it does allow developers to gain a considerable amount of confidence in the middle layers of code.

See the *Further reading* section for articles that cover unit testing in more detail.

Now that we have an idea of how to write reusable code, we need to make sure it is being stored in a way that allows it to be used across multiple projects without creating unnecessary copies or creating strange inter-project dependencies.

# Organizing source code

A well-organized source tree is extremely important if a code base is intended to evolve and grow over time. If projects are meant to live in isolation as atomic entities that never interact with one another, there is little reason to have a strategy when it comes to source control; but if code reuse is a goal, then having a clear idea of how specific projects should fit together with common code is a must.

# Choosing locations for source files

Any piece of code that is likely to be used in more than the original project where it is first created should live in a common location (not tied to a specific project). Even if the code started out as being specific to one particular project, it should be moved as soon as it is used by more than one project. Pieces of common code will be different for each team, but will likely include the following:

*   **BSPs**: There are often multiple pieces of firmware created for each board. The BSP folder in the code base for this book doesn't have any subfolders (mainly because there, the code only supports a single platform). If this book supported multiple platforms, the BSP folder would likely contain a `Nucleo_F767` subfolder.
*   **In-house common code**:This can include custom domain-specific algorithms or drivers for ICs that are commonly used across multiple products or projects. Any code here should be able to be well-abstracted and used across multiple MCUs.
*   **Third-party common code**:If multiple projects include source code from a third party, it belongs in a central location. Items such as FreeRTOS and any other middleware can be kept in this central location.
*   **MCU-specific code**:Each MCU family should ideally have its own folder. This will likely include things such as STM32 HAL and any custom peripheral drivers developed for that MCU. Ideally, most of the code referenced in these MCU-specific directories will be done so through common interfaces (shown in the ADC example at the beginning of the chapter).
*   **Interface definitions**:If interfaces are used extensively, having all of them in one place is extremely convenient.
*   **Project folders**:Each project will likely have its own folder (sometimes containing sub-projects). Ideally, projects won't reference code in other projects—only code that is housed in the common areas. If projects start to have inter-dependencies, take a step back and evaluate why, and whether or not it makes sense to move those dependencies to a common location.

The specific folder structure will likely be dependent on your team's version control system and branching strategy.

# Dealing with changes

One of the biggest drawbacks to having code that is common to many projects is the implications of a change. Directory structure changes can be some of the most challenging to deal with, especially if there are a large number of projects. Although painful, this type of refactoring is often necessary over time as teams' needs and strategies change. Performing regular check-ins and tagging your repository should be all that is necessary to provide confidence that directory restructuring changes, while painstaking, aren't particularly dangerous.

If you're coming from a high-level language and hear the word *interface,* you might immediately think of something that is set in stone from the first time it is used. Although it is generally good to keep interfaces consistent, there is a bit of latitude to change them (especially when first starting out). Internal interfaces in this specific use case are considerably more forgiving than a public API for a couple of reasons:

*   Nearly all of the low-level MCU-based applications are going to have compile-time checks against a given *interface.* There are no dynamically loaded libraries that will mysteriously cease to work properly at runtime if an interface changes over time—(most) errors will be caught at compile time.
*   These interfaces are generally internal with full visibility of where they are used, which makes it possible to evaluate the impact of potential changes.

Changes to individual files (such as a shared algorithm) are also a common source of concern. The best advice here is to evaluate whether or not what you are changing is still providing the same functionality or whether it should be an extension or entirely new. Sometimes, working with projects in a vacuum doesn't force us to make these decisions explicitly, but as soon as that piece of code is shared across many projects, the stakes are higher.

# Summary

After reading this chapter, you should have a good understanding of why code reuse is important and also how to achieve it. We've looked at the details of using abstraction in an embedded environment and created fully hardware-agnostic interfaces that increase the flexibility of code. We also learned how to use these interfaces in conjunction with tasks to increase code reuse across projects. Finally, we touched on some aspects of storing shared source code.

At this point, you should have enough knowledge to start thinking about how to apply these principles to your own code base and projects. As your code base starts to have more common code that is reused between projects, you'll begin to reap the benefits of a shared code base, such as fewer bugs, more maintainable code, and decreased development time. Remember, it takes practice to become good at creating reusable code with abstract interfaces. Not all implementations need to move to fully reused components at the same time, either—but it is important to start the journey.

Now that we have some background of abstraction, in the next chapter, we'll continue to build flexible architectures by looking more deeply at how queues can be used to provide loosely coupled architectures.

# Questions

As we conclude, here is a list of questions for you to use to test your knowledge on this chapter's material. You will find the answers in the *Assessments* section of the appendix:

1.  Creating abstractions is only something that can be done using a full desktop OS:
    *   True
    *   False
2.  Only object-oriented code such as C++ can benefit from well-defined interfaces:
    *   True
    *   False
3.  Four examples of why abstraction is important were given. Name one.
4.  Copying code into a new project is the best way to reuse it:
    *   True
    *   False
5.  Tasks are extremely specific; they cannot be reused between projects:
    *   True
    *   False

# Further reading

*   For an in-depth discussion about multi-layer drivers, see TinyOS TEP101, which uses a layered approach to drivers. The interface approach described in this chapter fits quite well with the TinyOS HPL, HAL, and HIL approach:[ 
    https://github.com/tinyos/tinyos-main/blob/master/doc/txt/tep101.txt](https://github.com/tinyos/tinyos-main/blob/master/doc/txt/tep101.txt)
*   Here are some more additional resources that should help you out:
    *   [https://embeddedartistry.com/blog/2019/08/05/practical-decoupling-techniques-applied-to-a-c-based-radio-driver/](https://embeddedartistry.com/blog/2019/08/05/practical-decoupling-techniques-applied-to-a-c-based-radio-driver/)
    *   [https://embeddedartistry.com/blog/2020/01/27/leveraging-our-build-systems-to-support-portability/](https://embeddedartistry.com/blog/2020/01/27/leveraging-our-build-systems-to-support-portability/)
    *   [https://embeddedartistry.com/blog/2020/01/20/prototyping-for-portability-lightweight-architectural-strategies/](https://embeddedartistry.com/blog/2020/01/20/prototyping-for-portability-lightweight-architectural-strategies/)
    *   [https://blog.wingman-sw.com/archives/282 - Unit Testing RTOS dependent code- RTOS Test Double](https://blog.wingman-sw.com/archives/282)