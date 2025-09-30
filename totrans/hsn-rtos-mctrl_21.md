# Troubleshooting Tips and Next Steps

This chapter explores some of the most useful tips and tools for analyzing and troubleshooting an RTOS-based system. Periodically checking your system during development, as well as having a few standard steps to take when troubleshooting, can be a huge timesaver when evaluating a problematic system – things don't always go as planned! After we've covered some tips, we'll take a look at some of the next steps we can take to continue learning and sharpening our embedded programming skills.

In this chapter, we will cover the following topics:

*   Useful tips
*   Using assertions
*   Next steps

# Technical requirements

No hardware or software is required for this chapter.

# Useful tips

Beginning development with an RTOS can be quite a shift if you've only used a *bare-metal* programming approach, especially if you're also shifting from 8-bit MCUs to a 32-bit MCU such as the STM32F7 we've been using in the examples throughout this book. Here are some tips that should help keep your project on track and help you work through issues when they come up.

# Using tools to analyze threads

Being able to get a clear understanding of what all the threads in a system are doing is a huge help – for novices and experts alike. Tooling is especially helpful for this. Using a visualization tool such as SEGGER SystemView or Percepio Tracealyzer can be invaluable in understanding interactions between various tasks and interrupts in a system (see [Chapter 6](699daa80-06ae-4acc-8b93-a81af2eb774b.xhtml), *Debugging Tools for Real-Time Systems*, for details). 

Having an RTOS-aware debugger is also a huge help since it allows us to stack the analysis of multiple tasks. This debugger can be part of your IDE or a standalone debugger such as SEGGER Ozone (see [Chapters 5](84a945dc-ff6c-4ec8-8b9c-84842db68a85.xhtml), *Selecting an IDE*, and [Chapter 6](699daa80-06ae-4acc-8b93-a81af2eb774b.xhtml), *Debugging Tools for Real-Time Systems*). 

# Keeping an eye on memory usage

Memory usage is a very important aspect to consider when using an RTOS. Unlike a super-loop with a single stack – which, along with the heap, would consume whatever RAM was *left over* – each FreeRTOS task's stack needs to be explicitly sized. In [Chapter 15](0f98e454-9804-4589-9854-5c38c9d8d416.xhtml), *FreeRTOS Memory Management*, in the *Keeping an eye on stack space* section, we showed you how to observe the available stack space, as well as how to implement hooks if an overflow was detected.  

If your application is using dynamic memory allocation, you should strongly consider enabling and implementing the failed MALLOC hooks provided by FreeRTOS. This was covered in [Chapter 15](0f98e454-9804-4589-9854-5c38c9d8d416.xhtml), *FreeRTOS Memory Management*, in the *Keeping an eye on heap space* section as well.

# Stack overflow checking

If you have a memory protection unit available, it is an excellent idea to make use of it since it will detect access violations such as stack overflows with better reliability than any of the software-based solutions (see [Chapter 15](0f98e454-9804-4589-9854-5c38c9d8d416.xhtml), *FreeRTOS Memory Management*, the *Using a memory protection unit* section).

Another way of keeping an eye on the stack is to set up stack monitoring, which was also covered in [Chapter 15](0f98e454-9804-4589-9854-5c38c9d8d416.xhtml), *FreeRTOS Memory Management*, in the *Keeping an eye on stack space* section.

A real-world example of debugging a system that has a stack overflow and checking memory is covered in the next section. 

# Fixing SystemView dropped data

In the examples we've looked at throughout this book, SystemView shows that we can stream data visualization by running code on the MCU to store events in a local buffer. The contents of the buffer are then transferred data via debug hardware to the PC for viewing. Sometimes, during high utilization, you'll see large red blocks in the trace, as shown in the following screenshot:

![](img/da563df0-de19-4425-9970-77792736f005.png)

These blocks indicate that SystemView has detected dropped packets. The frequency of dropped packets can be decreased by doing any of the following:

*   Increasing the size of the SystemView buffer on the MCU. `SEGGER_SYSVIEW_Conf.h` defines the buffer on line 132\. It is important to note that since this buffer resides on the MCU, increasing the size of the buffer will decrease the memory available to other pieces of code.:

```cpp
#define SEGGER_SYSVIEW_RTT_BUFFER_SIZE
```

*   Increasing the clock speed of the debugger under Target Interface and Speed. In some cases, a debugger that supports a faster clock will help (for example, a dedicated SEGGER J-Link or J-Trace).  
*   Decreasing the traffic to the debugger hardware while SystemView is running. To do this, you can, for example, close any live trace windows in open debug sessions (such as Ozone or STM32CubeIDE).

In the next section, we'll learn how to debug our system by using assertions.

# Using assertions

Assertions are excellent tools for catching conditions that simply *shouldn't happen*. They provide us with a simple means to check assumptions. See the *Creating a task – checking the return value* section of [Chapter 7](2fa909fe-91a6-48c1-8802-8aa767100b8f.xhtml) , *The FreeRTOS Scheduler,* for an example of how to add simple assertions to prevent code from running when the system is in an unacceptable state.

A special FreeRTOS flavor of the assert construct is `configAssert`.

# configAssert

`configAssert` is used throughout FreeRTOS as a way of guarding against an improperly configured system. Sometimes, it is triggered when a non-interrupt version of the API is called from inside an ISR. Often times, code inside an interrupt will attempt to call a FreeRTOS API, but its logical priority is higher than what the RTOS will allow. 

Rather than allowing an application to run with undefined behavior, FreeRTOS will regularly test a set of assertions to ensure all prerequisites are met. On their own, these checks are helpful at preventing a system from careening completely out of control with no hope of figuring out what the problem is. Instead, the system is immediately halted when the invalid condition occurs. FreeRTOS also contains thorough documentation on the underlying reasons the assertion has failed (sometimes with links to web-based documentation).

Don't ever *cover up* a `configAssert` by disabling it in any way. They are often the first notification that a serious configuration problem exists. Disabling the assertion will only compound the underlying issue, making it harder to find later.

Let's go through an example that shows what the normal symptoms of a system halted with `configAssert` might look like, as well as the steps that can be taken to diagnose and solve the underlying issue.

# Debugging a hung system with configAssert()

When you first bring up the codebase and create some example code to introduce SystemView, several problems need to be worked through. 

Here's our example: 

After it was ensured that all of the code was syntactically correct and the LEDs were blinking, it's time to connect SystemView to the running application and get some timing diagrams. The first couple of times SystemView is connected, a few events are shown, but then the system goes unresponsive:

*   The LEDs stopped blinking
*   No additional events were showing up in SystemView, as shown in the following screenshot:

![](img/77b5a1cb-28e6-42b7-918f-3d9e7d3c2b50.png)

Let's diagnose and solve the underlying issue in a couple of steps.

# Collecting the data

Sometimes, it is tempting to take guesses as to what might be happening or make assumptions about the system. Rather than doing either of these things, we'll simply connect our debugger to the system to see what the problem is.

Since SEGGER Ozone is exceptionally good at connecting to a running system without modifying its state, we're able to connect to the hung application without disrupting anything. This allows us to start debugging an application after it's crashed, even if it was previously *not* running through the debugger. This can come in very handy during product development since it allows us to run the system normally, without constantly starting it from the debugger. Let's learn how to do this:

1.  Set up Ozone with the same code that is running on the target. Note that the development board must be connected via USB (see [Chapter 6](699daa80-06ae-4acc-8b93-a81af2eb774b.xhtml),* Debugging Tools for Real-Time Systems*, for details). 

2.  After that, select Attach to Running Program:

![](img/41be5a57-4cec-4dca-ac53-f2cbadfdfbf7.png)

3.  Upon attaching and pausing execution, we're greeted with the following screen and are immediately able to make some observations:

![](img/ad2fb10e-af88-4011-b4f1-80085668d43b.png)

Notice the following:

*   The LEDs have stopped blinking because we're spending all of our time in an infinite loop because of a failed assertion.
*   By looking at the Call Stack, we can see that the offending function is `SEGGER_SYSVIEW_RecordSystime`, which is apparently making a call to a function called `_cbGetTime`, which in turn calls `xTaskGetTickCountFromISR`.
*   Reading through the detailed comment above line 760, it sounds like there may be some misconfigured NVIC priority bits.
*   The maximum acceptable value of `ulMaxPROGROUPValue` (which can be seen by hovering over the selected variable) is `1`.

Now that we know *which* assertion failed, it's time to figure out the root cause of *why* exactly it failed.

# Digging deeper – SystemView data breakpoints

So far, we've determined where our processor is stuck, but we haven't uncovered anything to help us determine what needs to be changed to get the system operational again. Here are the steps we need to take to uncover the root cause of the issue:

1.  Let's take a look at the assertion again. Here, our goal is to troubleshoot exactly why it is failing. Run the following command:

```cpp
configASSERT( ( portAIRCR_REG & portPRIORITY_GROUP_MASK ) <= ulMaxPRIGROUPValue );
```

2.  Using SystemView's memory viewer, analyze the value of `portAIRCR_REG` in `port.c`:

![](img/f63839d2-8f89-41be-9a5c-9ad3882f9c87.png)

3.  Since this is a hardcoded memory location, we can Set Data Breakpoint, which will pause execution each time the memory location is written. This can be a quick way to track down all of the ways a variable is accessed, without attempting to search through the code:

![](img/d1dc5491-6065-4e40-968d-9084327b8ebf.png)

4.  Upon restarting the MCU, the write breakpoint is immediately hit. Although the program counter is pointing to `HAL_InitTick`, the actual data write to the `0xE000ED0C` address was done in the previous function, that is, `HAL_NVIC_SetPriorityGrouping`.  This is exactly what we expect since the assert is related to interrupt priority groups:

![](img/13d927f0-2b5d-4d58-91a0-bb42989983bb.png)

5.  Some quick searching through the code for `NVIC_PRIORITYGROUP_4` reveals the following comment in `stm32f7xx_hal_cortex.c`:

```cpp
* @arg NVIC_PRIORITYGROUP_4: 4 bits for preemption priority
*                            0 bits for subpriority
```

**Priority grouping**: The interrupt controller (NVIC) allows the bits that define each interrupt's priority to be split between bits that define the interrupt's preemption priority bits, as well as the bits that define the interrupt's sub-priority. For simplicity, all bits must be defined to be preemption priority bits. The following assertion will fail if this is not the case (if some bits represent a sub-priority).

Based on this information, there should be `0` bits for the subpriority. So, why was the value of the priority bits in `portAIRCR_REG` non-zero? 

From the *ARM® Cortex® -M7 Devices Generic User Guide, *we can see that to achieve 0 bits of subpriority, the value of the **AIRCR** register masked with **0x00000700** must read as 0 (it had a value of **3** when we looked at the value in memory):

![](img/46013ad9-c773-49cd-aa75-e6d70d88a8f8.png)

Here is the explanation for `PRIGROUP` in the same manual. Notice that `PRIGROUP` must be set to 0b000 for 0 subpriority bits:

![](img/6d06da99-492e-4154-9329-f06a638790a1.png)

This certainly warrants further investigation... why was the value of `PRIOGROUP` 3 instead of 0? Let's take another look at that `configAssert()` line:

```cpp
configASSERT( ( portAIRCR_REG & portPRIORITY_GROUP_MASK ) <= ulMaxPRIGROUPValue );

```

Note the following definition of `ulMaxPRIOGROUPValue` in `port.c`. It is defined as *static*, which means it has a permanent home in memory:

```cpp
#if( configASSERT_DEFINED == 1 )
 static uint8_t ucMaxSysCallPriority = 0;
  static uint32_t ulMaxPRIGROUPValue = 0;
```

Let's set up another data breakpoint for `ulMaxPRIGROUPValue` and restart the MCU again, but this time, we'll watch each time it is accessed:

*   As expected, something was accessed by the `BaseType_t xPortStartScheduler( void )` function in `port.c`
*   The curious part about the data access breakpoint is that it is hit when the program counter is inside `SEGGER_RTT.c`, which doesn't look right since `ulMaxPRIGROUPValue` is privately scoped to `xPortStartScheduler` in `port.c`
*   Looking at the debugger – the problem is staring right at us:
    *   The `ulMaxPRIGROUPValue` static variable is being stored in `0x2000 0750`.
    *   The data write breakpoint was hit with the stack pointer at `0x200 0740`.
    *   The stack has been overrun:

![](img/1c02901d-20e3-48fc-bd63-57e41bd18de6.png)

We've just uncovered a stack overflow**. **It manifested itself as a write into a static variable (which happened to trigger a `configAssert` in an unrelated part of the system). This type of wildly unexpected behavior is a common side effect of stack overflows.

Currently, the minimum values of each stack in `main.c` has been set to 128 words (1 word = 4 bytes), so increasing this to 256 words (1 KB) gives us plenty of headroom.

*This example is fairly representative of what happens when functionality is added to a preexisting task that was working properly previously.* If the new functionality requires more functions to be called (with each having local variables), those variables will consume stack space. In this example, this problem only showed up after adding the SEGGER print functionality to an existing task. Because there wasn't additional stack space available, the task overflowed its stack and corrupted the memory that was being used by another task.

The problem in this example would have likely been caught if we had the stack overflow hooks set up – it would have certainly been caught if the MPU port was being used.

# Next steps

Now that you've been through this book and tinkered with each ready-to-run example – wait... you haven't run the examples yet?!  Time to get started on that! They have been included because having hands-on experience will help drive these concepts home, providing you with both valuable practice and a base development environment you can use for your own projects.

So, assuming you've already run through the examples included, an excellent next step to gain an even more in-depth understanding of FreeRTOS is to read *Richard Barry's* book, *Mastering the FreeRTOS™ Real-Time Kernel*. This book focuses on how to apply the general knowledge that is required to get started with embedded systems and build a solid foundation for future development. Mastering FreeRTOS, however, is laser-focused on the specific details of FreeRTOS, with examples for each of the APIs. Having a hardware environment set up, a basic understanding of the fundamentals, and debug/visualization tooling at hand will help you get the most out of his book. After you have a system up and running, the code provided in *Mastering the FreeRTOS™ Real-Time Kernel* can be easily tested and experimented with using real hardware and a visual debugging system.

While we're on the subject of building solid foundations, you'll want to consider getting acquainted with test-driven development. As you start to create loosely coupled code, as we did in [Chapter 12](8e78a49a-1bcd-4cfe-a88f-fb86a821c9c7.xhtml),* Tips on Creating Well-Abstracted Architecture*, and [Chapter 13](e728e173-c9b2-4bb8-91c8-ed348ccf9518.xhtml),* Creating Loose Coupling with Queues*, testing these subsystems is a natural next step. *James Grenning* has many resources available on his website ([https://blog.wingman-sw.com](https://blog.wingman-sw.com)), specifically for embedded C/C++. Other TDD resources specific to embedded C include *Matt Chernosky's* site ([http://www.electronvector.com/](http://www.electronvector.com/)) and the unique *Throw the Switch *([http://www.throwtheswitch.org/](http://www.throwtheswitch.org/)). A great all-around embedded resource that's been created from decades of hard-earned experience is *Jack Gannsle's* site, which you can access at [http://www.ganssle.com/](http://www.ganssle.com/).

# Summary

In this final chapter, we covered a few tips that will help smooth out some of the bumps in the road of your RTOS journey, as well as a few suggested next steps.

That's it, folks! I hope you've enjoyed this hands-on introduction to developing firmware for real-time embedded systems using FreeRTOS, STM32, and SEGGER tools. Now, it's time to get out there and start understanding systems, solving problems, and analyzing your solution! I'd love to hear about how you've applied what you've learned in this book – give me a shout on LinkedIn, Twitter, or GitHub! If you've really enjoyed this book and think others would also like it, consider leaving a review – they help spread the word!

# Questions

As we conclude this book, here is a list of questions for you to test your knowledge regarding this chapter's material. You will find the answers in the *Assessments* section of the *Appendix*:

1.  When your system crashes after you've added an interrupt or used a new RTOS primitive, what steps should you take?
2.  Name one common cause of unexpected behavior (caused by firmware) when developing with an RTOS.
3.  Since your system has no way of outputting data (no exposed serial port or communication interface), it will be impossible to debug.
    *   True
    *   False