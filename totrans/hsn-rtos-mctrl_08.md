# Debugging Tools for Real-Time Systems

Serious debugging tools are incredibly important in serious embedded systems development. Complex RTOS-based systems can have many tasks and dozens of ISRs that need to be completed in a timely manner. Figuring out whether everything is working properly (or *why* it isn't) is way easier with the right tools. If you've been troubleshooting with the occasional print statement or blinking LEDs, you're in for a treat!

We'll be making heavy use of Ozone and SystemView throughout the remainder of this book but first, we'll need to get them set up and look at a quick introduction. Toward the end of this chapter, we'll take a look at other debugging tools, as well as techniques for reducing the number of bugs that get written in the first place.

In a nutshell, we will be covering the following in this chapter:

*   The importance of excellent debugging tools
*   Using SEGGER J-Link
*   Using SEGGER Ozone
*   Using SEGGER SystemView
*   Other great tools

# Technical requirements

Several pieces of software will be installed and configured in this chapter. Here's what you should already have on hand:

*   A Nucleo F767 development board
*   A micro-USB cable

*   A Windows PC (the Windows OS is only required by the ST-Link Reflash utility)
*   STM32CubeIDE (ST-Link drivers are required for the ST-Link Reflash utility)

All source code for this chapter can be downloaded from [https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapters5_6](https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapters5_6).

# The importance of excellent debugging tools

When developing any piece of software, it's all too easy to start writing code without thinking about all of the details. Thanks to code generation tools and third-party libraries, we can very quickly develop a feature-filled application and have it running on actual hardware in fairly short order. However, when it comes to getting *every* part of a system working 100% of the time, things are a bit more difficult. If a system is stood up too quickly and the components weren't properly tested before integrating them, there would be pieces that work *most* of the time, but not always.

Often with embedded systems, only a few parts of the underlying application are visible. It can be challenging to evaluate the overall system health from a user's viewpoint. Historically, good debug tooling was less common for embedded work than non-embedded. Putting print statements everywhere only gets you so far, causes timing problems, and so on. Blinking LEDs is cumbersome and doesn't provide much insight. Analyzing signals via hardware can help to verify symptoms but doesn't always isolate the root cause of an issue. Trying to figure out what code is actually running (and when) in an event-driven system is really challenging without the tools to help visualize execution.

This is why having a variety of familiar tools at your disposal is extremely helpful. It allows you to focus your efforts on developing small portions of the application confidently. Confidence comes from rigorously verifying each piece of functionality as it is developed and integrated with the rest of the system. However, in order to perform verification, we need to have transparency in different portions of the code (not just the parts that are observable from outside the system). Many times during verification, situations arise when there is a need to observe inter-task execution.

There are two important areas that help us achieve the objectives of system transparency and observable task relationships: RTOS-aware debugging and RTOS visualization.

# RTOS-aware debugging

With traditional debugging setups used for bare-metal (for example, no OS) coding, there was only one stack to observe. Since the programming model was a single super loop with some interrupts, this wasn't much of a problem. At any point in time, the state of the system could be discerned by the following:

*   Knowing which function the **program counter** (**PC**) was in
*   Knowing which interrupts were active 
*   Looking at the value of key global variables
*   Observing/unwinding the stack

With an RTOS-based system, the basic approach is very similar but the programming model is extended to include multiple tasks running in *parallel*. Remember, each task is effectively an isolated infinite loop. Since each task has its own stack and can be in different operating states, some additional information is required to discern the overall system state:

*   Knowing the current operational state of each task
*   Knowing which task and function the PC was in
*   Knowing which interrupts are active 
*   Looking at the value of key global variables
*   Observing/unwinding the stack of each task

Due to the constrained nature of embedded systems, stack usage is often a concern because of the limited RAM of MCUs. In a bare-metal application, there is only one stack. In an RTOS application, each task has its own stack, which means we have more to monitor. Using a debugging system that provides RTOS-aware stack information helps to quickly evaluate the stack usage of each task in the system.

Monitoring the worst-case performance of the event response is also a critical aspect of real-time systems development. We must ensure that the system will respond to critical events in a timely manner. 

There are many different ways to approach this problem. Assuming the event originates with a hardware signal outside the MCU (which is true most of the time), a logic analyzer or oscilloscope can be used to monitor the signal. Code can be inserted in the application to toggle a pin on the MCU after that event has been serviced and the difference in time can be monitored. Depending on the system, access to test equipment, and the events in question, this hardware-centric method may be convenient. 

Another method is to use software in combination with **instrumentation** in the RTOS. With this method, small hooks are added into the RTOS that notify the monitoring system when events happen. Those events are then transmitted out of the MCU and onto a development PC running a viewing program. This method is what we'll be focusing on in this book—using SEGGER SystemView. This allows a tremendous amount of information and statistics to be collected with very little development effort. The slight downside to this method is that there is a very small amount of uncertainty added since it is a purely software/firmware approach. It relies on the MCU to record when the events happen, which means if an interrupt is significantly delayed in being serviced, it will not be recorded accurately. There is also a strong dependency on the availability of RAM or CPU cycles. This approach can become inconclusive on heavily loaded systems without adequate RAM. However, these downsides have workarounds and aren't encountered on most systems.

# RTOS visualization

Having the ability to see which tasks are running and how they are interacting is also important. In a preemptive scheduling environment, complex relationships can develop between tasks. For example, in order for an event to be serviced, there might be a few tasks that need to interact with one another. On top of that, there may be several more tasks all vying for processor time. In this scenario, a poorly designed system that is consistently missing deadlines may only be perceived as being *sluggish* from the user's perspective. With task visualization, a programmer can literally *see* the relationships between all tasks in the system, which helps considerably with analysis.

We will work through a real-world example of visualizing scenarios such as this one in [Chapter 8](c6d7a0c6-6f18-4e06-a372-cd1605942ecd.xhtml), *Protecting Data and Synchronizing Tasks*, with the `mainSemPriorityInversion.c` demo.

The ability to easily discern what state tasks are in over a period of time is extremely helpful when unraveling complex inter-task relationships. SEGGER SystemView will also be used to visualize inter-task relationships.

In order to perform an in-depth analysis on a running system, we'll need a way to attach to the MCU and get information out of it. On Cortex-M MCUs, this is most efficiently done with an external debug probe.

# Using SEGGER J-Link

A debug probe is a device that allows a computer to communicate and program the non-volatile flash of an MCU. It communicates with special hardware on the MCU (called Coresight on ARM Cortex-M processors). The SEGGER J-Link and J-Trace family of debug probes are among the most popular in the industry. SEGGER also offers useful software that integrates with their tools free of charge. The accessibility of these tools and the quality of the accompanying software makes this an excellent fit for use in this book.

If you plan on using a paid IDE, the IDE vendor likely has their own proprietary debug probes available. Many excellent software features will likely be tied to their hardware. For example, ARM Keil uVision MDK integrates with ARM Ulink probes and IAR offers their I-Jet debug probes. IDEs such as these also integrate with third-party probes but be aware of what trade-offs there may be before making a purchasing decision.

There are many options when selecting debug probes from SEGGER —we'll briefly go through some of the options currently available and look at the hardware requirements for each.

# Hardware options

SEGGER has many different hardware options that cover a wide range of pricing and capabilities. For a complete and current list, check out their website at [https://www.segger.com/products/debug-probes/j-link/models/model-overview/](https://www.segger.com/products/debug-probes/j-link/models/model-overview/).

The models generally fit into two main categories: debuggers with full Cortex-M Trace support and those without. 

# Segger J-Trace

The debuggers with full trace support are referred to as J-Trace. The **Cortex Embedded Trace Macrocell** (**Cortex ETM**) is an extra piece of hardware inside the MCU that allows every instruction that has been executed to be recorded. Transmitting all of this information off the MCU requires a few extra pins for clocking the data out (a clock line and 1-4 data lines). Having the ability to trace every instruction the MCU has executed enables functionality such as code coverage, which provides insight into how much code has been executed (line by line). Knowing exactly which lines of code have been executed and when gives us the opportunity to see where a program is spending most of its time. When we know which individual lines of code are executed most often, it is possible to optimize that small portion of code when improved performance is required.

In order to take full advantage of the advanced trace features, all of the following is required:

*   The MCU must have ETM hardware.
*   The specific MCU package must bring the ETM signals out to pins.
*   The peripheral configuration must not share ETM signals with other functions.
*   The system circuit must be designed to incorporate ETM signals and a connector.

The most common connector used for Debug and ETM signals is a 0.05" pitch header with the following pinout (the trace signals are highlighted):

![](img/606fde5a-281e-425d-98da-39ae354e764b.png)

All of this functionality comes at a price, of course. The J-Trace models are at the high end of SEGGER's spectrum, both in terms of functionality and price (typically over USD$1000). Unless you're developing fully custom hardware, also expect to pay for a full evaluation board (over USD$200) rather than the low-cost development hardware used in this book. While these costs are typically completely reasonable for a full-blown engineering budget during new product development, they are too expensive to be widely accessible for individuals.

# SEGGER J-Link

SEGGER J-Link has been around in many different forms and has grown to encompass several models. Typically, the higher-end models provide faster clock speeds and a richer experience (faster downloads, responsive debugging, and so on). A few **EDU** models are sold at an extremely large discount for educational purposes (thus the **EDU** designation). These models are fully featured but may not be used for commercial purposes.

The most common connector used for the Cortex-M is a 0.05" pitch header with the following pinout. Notice, this connector's pinout is the same as the first 10 pins from the Debug+Trace connector (refer to the following diagram).

![](img/01e54c91-a4cc-400f-9059-3deda9ecb214.png)

SEGGER has done an excellent job designing software interfaces that aren't tied to the underlying hardware. Because of this, their software tools work across different hardware debugger models without modification. This has also resulted in the hardware option we'll be using in this book—the SEGGER J-Link on-board.

# SEGGER J-Link on-board

The specific hardware variant of ST-Link we'll be using in our exercises isn't actually made by SEGGER. It is the ST-Link circuitry already included on the Nucleo development board. Nucleo boards have two separate sub-circuits: programming hardware and target hardware. 

The programming hardware sub-circuit is generally referred to as an ST-Link. This **programming hardware** is actually another STM MCU that is responsible for communicating with the PC and programming the **target hardware**—the STM32F767\. Since Nucleo hardware is primarily aimed at the ARM Mbed ecosystem, the ST-Link MCU is programmed with firmware that implements both the ST-Link and Mbed functionalities:

![](img/f7807694-5d5b-492d-9d3b-4070bf3f9174.png)

In order to use the programming hardware on the Nucleo board as a SEGGER JLink, we will be replacing its firmware with SEGGER J-Link on-board firmware. 

# Installing J-Link

Detailed installation instructions are available from SEGGER at [https://www.segger.com/products/debug-probes/j-link/models/other-j-links/st-link-on-board/](https://www.segger.com/products/debug-probes/j-link/models/other-j-links/st-link-on-board/). A few notes are also included here for convenience. In order to convert the on-board ST-Link to a J-Link, we'll be downloading and installing two pieces of software: the J-Link tools and the ST-Link re-flashing utility. You should already have the necessary ST-Link drivers installed from the STM32CubeIDE installation carried out in the previous chapter:

A Windows PC is only required for the J-Link Reflash utility (it is distributed as `*.exe`). If you're not using a Windows PC for development and STM32CubeIDE isn't installed, make sure you install USB drivers for the ST-Link (the optional *step 1* in the following list).

1.  If you don't have STM32CubeIDE installed already, download and install the ST-Link drivers from [http://www.st.com/en/development-tools/stsw-link009.html](http://www.st.com/en/development-tools/stsw-link009.html) (this step is optional). 
2.  Download the appropriate J-Link utilities for your OS from [https://www.segger.com/downloads/jlink](https://www.segger.com/downloads/jlink).
3.  Install the J-Link utilities— the default options are fine.
4.  Download the SEGGER J-Link Reflash utility (for Windows OS only) from [https://www.segger.com/downloads/jlink#STLink_Reflash](https://www.segger.com/downloads/jlink#STLink_Reflash).
5.  Unzip the contents of `STLinkReflash_<version>.zip`— it will contain two files:
    *   `JLinkARM.dll`
    *   `STLinkReflash.exe`

Now, we will convert ST-Link to J-Link.

# Converting ST-Link to J-Link

Follow these steps to upload J-Link firmware to the ST-Link on the Nucleo development board:

1.  Plug in a micro USB cable to `CN1` on the Nucleo board and attach it to your Windows PC.
2.  Open `STLinkReflash_<version>.exe`.
3.  Read through and accept the two license agreements.
4.  Select the first option: Upgrade to J-Link.

The debugging hardware on the Nucleo board is now effectively a SEGGER J-Link!

Now that a J-Link is present, we will be able to use other SEGGER software tools, such as Ozone and SystemView, to debug and visualize our applications.

# Using SEGGER Ozone

SEGGER Ozone is a piece of software that is meant to debug an already-written application. Ozone is independent of the underlying programming environment used to create the application. It can be used in many different modes, but we'll focus on importing an `*.elf` file and crossreferencing it with source code to provide FreeRTOS-aware debugging capability to a project created with any toolchain. Let's take a quick look at the various file types we will be working with in Ozone.

# File types used in the examples

There are several file types used when programming and debugging embedded systems. These files are common across many different processors and software products and not exclusive to Cortex-M MCUs or the software used in this book.

**Executable and Linkable Format** (**ELF**) files are an executable format that has the ability to store more than the straight `*.bin` or `*.hex` files commonly flashed directly into an MCU's ROM. The `*.elf` files are similar to the `*.hex` files in that they contain all of the binary machine code necessary to load a fully functional project onto a target MCU. The `*.elf` files also contain links to the original source code filenames and line numbers. Software such as Ozone uses these links to display source code while debugging the application:

*   `*.bin`: A straight binary file (just 1s and 0s). This file format can be directly "burned" into an MCU's internal flash memory, starting at a given address.
*   `*.hex`:Usually a variant of Motorolla S-record format. This ASCII-based file format contains both absolute memory addresses and their contents.
*   `*.elf`: Contains both the executable code as well as a header that is used to cross-reference each memory segment to a source file. This means a single ELF file contains enough information to program the target MCU and also cross-reference all of the source code used to create the binary memory segments.

The ELF file does not *contain* the actual source code used, it only contains absolute file paths that crossreference memory segments to the original source code. This is what allows us to open a `*.elf` file in Ozone and step through the C source code while debugging.

*   `*.svd`: Contains information that maps registers and descriptions to the memory map of the target device. By providing an accurate `*.svd` file, Ozone will be able to display peripheral views that are very helpful when troubleshooting MCU peripheral code.

A `*.svd` file is a file that is usually included with an IDE that supports your MCU. For example, STM32Cube IDE's location for the `*.svd` files is
*`C:\ST\STM32CubeIDE_1.2.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.productdb.debug_1.2.0.201912201802\resources\cmsis\STMicroelectronics_CMSIS_SVD`.* 

There are other file types used in embedded systems' development as well. This is by no means an exhaustive list—just the ones we'll be using most in the context of the example projects.

# Installing SEGGER Ozone

To install SEGGER Ozone, follow these two simple steps:

1.  Download SEGGER Ozone: [https://www.segger.com/downloads/jlink/](https://www.segger.com/downloads/jlink/)
2.  Install it using the default options.

Now, let's cover the necessary steps to create a FreeRTOS-aware Ozone project and take a quick look at some of the interesting features.

# Creating Ozone projects

Since Ozone is completely independent of the programming environment, in order to debug with it, some configuration is required, which is covered in the following steps:

All projects included in the source tree for this book already have Ozone projects created for them. The following steps are for your reference—you'll only need to go through these steps for your own future projects. Ozone project files, the `*.jdebug` files, are already included for all of the projects in this book.

1.  When first opened, select the Create a New Project prompt.
2.  For the Device field, select STM32F767ZI.
3.  For Peripherals, input the directory and location of the `STM32F7x7.svd` file:

![](img/fd3c1524-135b-4700-b079-ff896763055e.png)

4.  On the Connection Settings dialog screen, default values are acceptable.
5.  On the Program File dialog screen, navigate to the `*.elf` file that is generated by TrueStudio. It should be in the Debug folder of your project:

![](img/cfbb146e-e29c-481c-b0d1-f2a8e1ed6ee6.png)

6.  Save the `*.jdebug` project file and close Ozone.
7.  Open the `*.jdebug` file with a text editor.
8.  Add a line to the `*.jdebug` project file to enable the FreeRTOS plugin (only add the last line in bold):

```cpp
void OnProjectLoad (void) {
  //
  // Dialog-generated settings
  //
  Project.SetDevice ("STM32F767ZI");
  Project.SetHostIF ("USB", "");
  Project.SetTargetIF ("JTAG");
  Project.SetTIFSpeed ("50 MHz");
  Project.AddSvdFile ("C:\Program Files (x86)\Atollic\TrueSTUDIO for STM32 9.3.0\ide\plugins\com.atollic.truestudio.tsp.stm32_1.0.0.20190212-0734\tsp\sfr\STM32F7x7.svd");
  Project.SetOSPlugin("FreeRTOSPlugin_CM7");
```

It is a good idea to copy the `*.svd` file to a location used for storing source code or build tools since the installation directory for the IDEs is likely to change over time and between machines.

These steps can be adapted to set up Ozone for any other MCU supported by SEGGER family debuggers.

# Attaching Ozone to the MCU

Time to roll up our sleeves and get our hands dirty! The next sections will make more sense if you've got some hardware up and running, so you can follow along and do some exploring. Let's get everything set up:

1.  Open the STM32Cube IDE and open the `Chapter5_6` project.
2.  Right-click on `Chapter5_6` and select Build. This will compile the project into an `*.elf` file (that is, `C:\projects\packtBookRTOS\Chapters5_6\Debug\Chapter5_6.elf`).
3.  Open Ozone.
4.  Select Open Existing Project from the wizard.
5.  Select `C:\projects\packtBookRTOS\Chapters5_6\Chapters5_6.jdebug`.
6.  Use Ozone to download the code to the MCU (click the power button):

![](img/2a4459a6-9c8c-4e75-b87f-b2a1f15e4bb7.png)

7.  Push the play button to start the application (you should see the red, blue, and green LEDs flashing).

If your paths are different from what was used when creating the `*.jdebug` files, you'll need to reopen the `.elf` file (go to File | Open and select the file built in *step 2*).

Those same six steps can be repeated for any of the projects included in this book. You can also create a copy of the `.jdebug` file for other projects by simply opening a different `*.elf` file.

You may want to bookmark this page. You'll be following these same steps for the 50+ example programs throughout the rest of the book!

# Viewing tasks

A quick view overview of tasks can be seen by enabling the FreeRTOS task view. Using these tasks can prove to be very beneficial while developing RTOS applications: 

1.  After the MCU program has been started, pause execution by clicking the *Pause* button.
2.  Now, navigate to View | FreeRTOS Task View:

![](img/0a46ef4b-1ae5-47aa-9ddf-7a2fe247262e.png)

This view shows many useful pieces of information at a glance:

*   Task names and priorities.
*   Timeout: How many *ticks* a blocked task has until it is forced out of the blocked state.
*   Each task's stack usage (only the current stack usage is shown by default). Maximum stack usage is disabled in the preceding screenshot (seen by N/A)—(details on configuring FreeRTOS to monitor maximum stack usage will be covered in [Chapter 17](50d2b6c3-9a4e-45c3-9bfc-1c7f58de0b98.xhtml), *Troubleshooting Tips and Next Steps*).
*   Mutex Count: How many mutexes a task currently holds.
*   Notifications: Details on each task's notifications.

Having a bird's eye view of all of the tasks in the system can be a huge help when developing an RTOS-based application—especially during the initial phases of development.

# Task-based stack analysis

One of the challenges with debugging an RTOS with non-RTOS aware tools is analyzing the call stack of each task. When the system halts, each task has its own call stack. It is quite common to need to analyze the call stack for multiple tasks at a given point in time. Ozone provides this capability by using FreeRTOS Task View in conjunction with Call Stack View.

After opening both views, each task in FreeRTOS Task View can be double-clicked to reveal the current call stack of that task in Call Stack View. To reveal local variables for each task on a function-by-function basis, open Local Data view. In this view, local variables for the current function highlighted in the call stack will be visible.

 An example combining the task-based call stack analysis with a local variable view is shown here:

![](img/93051213-82ab-47a3-9536-18189311a414.png)

Notice the following from this screenshot: 

1.  When the MCU was stopped, it was in the "IDLE" task (shown by executing in the Status column).
2.  Double-clicking on "task 3" shows the call stack for "task 3". Currently, `vTaskDelay` is at the top of the stack.
3.  Double-clicking on `StartTask3` updates the Local Data window to show values for all local variables in `StartTask3`.
4.  Local Variables for `StartTask3` shows the current values for all local variables in `StartTask3`. 

SEGGER Ozone provides both a task-aware call stack view and a heads-up view for all running tasks in the system. This combination gives us a powerful tool to dive into the most minute details of each task running on the system. But what happens when we need a bigger picture view of the system? Instead of looking at each task individually, what if we'd prefer to look at the interaction *between* tasks in the system? This is an area where SEGGER SystemView can help.

# Using SEGGER SystemView

SEGGER SystemView is another software tool that can be used with SEGGER debug probes. It provides a means to visualize the flow of tasks and interrupts in a system. SystemView works by adding a small amount of code into the project. FreeRTOS already has `Trace Hook Macros`, which was specifically designed for adding in this type of third-party functionality.

Unlike Ozone, SystemView doesn't have any programming or debugging capabilities, it is only a **viewer***.*

# Installing SystemView

There are two main steps required for making your system visible with SystemView. The software needs to be installed and source code must be instrumented so it will communicate its status over the debug interface.

# SystemView installation

To install SystemView, follow these steps:

1.  Download SystemView for your OS. This is the main binary installer ([https://www.segger.com/downloads/free-utilities](https://www.segger.com/downloads/free-utilities)).
2.  Install using the default options.

# Source code configuration

In order for SystemView to show a visualization of tasks running on a system, it must be provided with information such as task names, priorities, and the current state of tasks. There are hooks present in FreeRTOS for nearly everything SystemView needs. A few configuration files are used to set up a mapping between the trace hooks already present in FreeRTOS and used by SystemView. Information needs to be collected, which is where the specific RTOS configuration and SystemView target sources come into play:

The source code included with this book already has all of these modifications performed, so these steps are *only necessary* if you'd like to add SystemView functionality to *your own FreeRTOS-based projects.*

1.  Download SystemView FreeRTOS V10 Configuration (v 2.52d was used) from [https://www.segger.com/downloads/free-utilities ](https://www.segger.com/downloads/free-utilities)and apply `FreeRTOSV10_Core.patch` to the FreeRTOS source tree using your preferred diff tool.

2.  Download and incorporate SystemView Target Sources (v 2.52h was used) from [https://www.segger.com/downloads/free-utilities](https://www.segger.com/downloads/free-utilities):
    1.  Copy all the source files into the `.\SEGGER` folder in your source tree and include them for compilation and linking. In our source tree, the `SEGGER` folder is located in `.\Middlewares\Third_Party\SEGGER`.
    2.  Copy `SystemView Target Sources\Sample\FreeRTOSV10\SEGGER_SYSVIEW_FreeRTOS.c/h` into the `SEGGER` folder and include it for compilation and linking.
    3.  Copy `.\Sample\FreeRTOSV10\Config\SEGGER_SYSVIEW_Config_FreeRTOS.c` into the`SEGGER` folder and include it for compilation and linking.

3.  Make the following changes to `FreeRTOSConfig.h`:
    1.  At the end of the file, add an include for `SEGGER_SYSVIEW_FreeRTOS.h`: `#include "SEGGER_SYSVIEW_FREERTOS.h"`.
    2.  Add `#define INCLUDE_xTaskGetIdleTaskHandle 1`.
    3.  Add `#define ``INCLUDE_pxTaskGetStackStart 1`.
4.  In `main.c` make the following changes:
    1.  Include `#include <SEGGER_SYSVIEW.h>`.
    2.  Add a call to `SEGGER_SYSVIEW_Conf()` after initialization and before the scheduler is started.

Since SystemView is called inside the context of each task, you'll likely find that the minimum task stack size will need to be increased to avoid stack overflows. This is because the SystemView library requires a small amount of code that runs on the target MCU (which increases the call depth and the number of functions that are placed on the stack). For all the gory details on how to troubleshoot stack overflows (and how to avoid them), see [Chapter 17](50d2b6c3-9a4e-45c3-9bfc-1c7f58de0b98.xhtml), *Troubleshooting Tips and Next Steps.*

# Using SystemView

After the source-code side of SystemView is straightened out, using the application is very straightforward. To start a capture, make sure you have a running target and your debugger and MCU are connected to the computer:

1.  Push the *Play* button.
2.  Select the appropriate target device settings (shown here):

![](img/4fc401c7-2698-4f87-99c8-24e6419c5516.png)

SystemView requires a *running* target. It will not show any information for a halted MCU (there are no events to display since it is not running). Make sure the LEDs on the board are blinking; follow the steps from the *Attaching Ozone to the MCU* section if they're not.

3.  After a second or so, events will be streaming into the log in the top-left Events view and you will see a live graphical view of the tasks as they're currently executing in Timeline:

![](img/713ce462-f002-4fb6-a19f-448c610f01f1.png)

*   Timeline shows a visual representation of task execution, including different states.
*   The Events view shows a list of events. Selected events are linked to the timeline.
*   The Context viewshows statistics for all events.
*   Terminal can be used to show printf-like messages from your code.

There are many more useful features, which will be covered while exploring code.

We're finally done installing the development software! If you've followed along so far, you now have a fully operational IDE, an RTOS visualization solution, and an extremely powerful RTOS-aware debugging system at your disposal. Let's see what other tools can be useful during the course of embedded systems development.

# Other great tools

The tools covered in this chapter certainly aren't the only ones available for debugging and troubleshooting embedded systems. There are many other tools and techniques that we simply don't have scope to cover (or that weren't a good fit due to the specific constraints placed on the tools used in this book). These topics are mentioned in the following section, with additional links in the *Further reading *section at the end of the chapter.

# Test-driven development 

Given the fact that the title of this chapter starts with the word *debugging,* it only seems appropriate to mention the ideal way to *debug* code is to not write buggy code in the first place. Unit testing isn't a single piece of software, but a component of **Test-Driven Development** (**TDD**)—a development methodology that inverts the way embedded engineers traditionally go about developing their systems. 

Instead of writing a bunch of code that doesn't work and then debugging it, test-driven development starts out by writing tests. After tests are written, production code is written until the tests pass. This approach tends to lead to code that is both testable and easily refactored. Since individual functions are tested using this approach, the resulting production code is much less tied to the underlying hardware (since it isn't easy to test code tied to real hardware). Forcing tests to be written at this level tends to lead to loosely coupled architecture, which is discussed in [Chapter 13](e728e173-c9b2-4bb8-91c8-ed348ccf9518.xhtml), *Creating Loose Coupling with Queues*. Using the techniques in [Chapter 13](e728e173-c9b2-4bb8-91c8-ed348ccf9518.xhtml), *Creating Loose Coupling with Queues, *works very well in conjunction with unit testing and TDD.

Generally, TDD isn't as popular in embedded systems. But if it is something you're interested in learning more about, check out a book written specifically on the topic—*Test Driven Development for Embedded C* by J*ames Grenning*.

# Static analysis

Static analysis is another way of reducing the number of bugs that creep into a code base. *Static* refers to the fact that the code doesn't need to be executing for this analysis to take place. A static analyzer looks for common programming errors that are syntactically correct (for example, they compile) but are *likely* to create buggy code (that is, out-of-bounds array access, and so on) and provides relevant warnings.

There are many commercially available packages for static analysis, as well as some that are freely available. Cppcheck is included in STM32CubeIDE (simply right-click on a project and select Run C/C++ Code Analysis). A link to a **Free Open Source Software** (**FOSS**) static analyzer from the Clang project is included at the end of this chapter. PVS-Studio Analyzer is an example of a commercial package that can be used freely for non-commercial projects.

# Percepio Tracealyzer

Percepio Tracealyzer is a tool similar to SEGGER SystemView in that it helps the developer to visualize system execution. Tracealyzer takes less effort to set up out of the box than SystemView and provides a more aesthetically focused user experience than SystemView. However, since it is supplied by a different company, the cost of the software is not included with the purchase of a SEGGER debug probe. You can find out more about Tracealyzer at [https://percepio.com/tracealyzer/](https://percepio.com/tracealyzer/).

# Traditional testing equipment

Before all of the attractive pieces of software for visualizing RTOS behavior on a computer screen existed, this task would fall to more traditional test equipment.

Logic analyzers have been around since MCUs first came into existence and are still among the most versatile tools an embedded system engineer can have in their kit. With a logic analyzer, the timing can be directly measured between when an input enters the system and when an output is provided by the system, as well as the timing between each of the tasks. Looking at the raw low-level signals going in and out of an MCU provides a level of visibility and gut feel for when something isn't right in a way that hexadecimal digits on a screen simply can't. Another advantage of habitually instrumenting at the hardware level - glitches in timing and other erratic behaviors are often noticed without directly looking for them.

Other tools you'll want to acquire if you're just starting out with embedded systems include a handheld **digital multi-meter** (**DMM**) and oscilloscope for measuring analog signals.

# Summary

In this chapter, we've covered why having access to excellent debugging tools is important. The exact tools we'll be using to analyze system behavior (SEGGER Ozone and SystemView) have been introduced. You've also been guided through how to get these tools set up for use with future projects. Toward the end, we touched on a few other tools that won't be covered in this book just to raise awareness of them.

Now that we've covered MCU and IDE selection, and we have all of our tooling squared away, we have enough background to get into the real meat of RTOS application development. 

Using this toolset will help you gain an in-depth understanding of RTOS behavior and programming as we dive into working examples in the upcoming chapters. You'll also be able to use this same tooling to create high-performing, custom real-time applications in the future.

In the next chapter, we'll get started with writing some code and go into more detail regarding the FreeRTOS scheduler.

# Questions

1.  J-Link hardware must be purchased to use the tools in this book.
    *   True
    *   False
2.  The only way to evaluate the effectiveness of a real-time system is to wait and see whether something breaks because a deadline was missed. 

1.  *   True
    *   False

3.  Since RTOSes have one stack per task, they are impossible to debug using a debugger since only the main system stack is visible. 

1.  *   True
    *   False

4.  The only way to ensure a system is completely functional is to write all of the code and then debug it all at once at the end of the project. 

1.  *   True
    *   False

5.  What is the style of testing called where each individual module is tested?
    *   Unit testing
    *   Integration testing
    *   System testing
    *   Black box testing
6.  What is the term given for writing tests before developing production code?

# Further reading

*   *Test-Driven Development for Embedded C* by *James Grenning*
*   SEGGER Ozone manual (UM08025): [https://www.segger.com/downloads/jlink/UM08025](https://www.segger.com/downloads/jlink/UM08025) 
*   SEGGER SystemView manual (UM08027): [https://www.segger.com/downloads/jlink/UM08027](https://www.segger.com/downloads/jlink/UM08027)
*   Clang Static Analyzer: [https://clang-analyzer.llvm.org](https://clang-analyzer.llvm.org)
*   PVS-Studio Analyzer: [https://www.viva64.com/en/pvs-studio/](https://www.viva64.com/en/pvs-studio/)