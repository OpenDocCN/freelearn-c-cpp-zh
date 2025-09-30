# Assessments

# Chapter 1

1.  No. A system with real-time requirements simply means that actions need to be deterministic. The timing requirements are determined by the needs of each system.
2.  No. There are several different ways to achieve real-time performance.
3.  No.
4.  Any system that has a deterministic response to a given event can be considered as *real-time*.
5.  Most industrial controls, closed-loop control systems, UAV flight controllers, **Anti-Lock Braking Systems** (**ABS**), **Engine Control Units** (**ECUs**), inkjet printers, test equipment (such as oscilloscopes and network analyzers), and so on.
6.  An MCU-based RTOS's strong point is moderately complex systems.

# Chapter 2

1.  Both of the above options.
2.  False.
3.  Complex super loops tend to have a large amount of variability in how long it takes them to execute the loop. This can lead to poor determinism in the system, since there is no easy way to provide a means for higher-priority work to take precedence over everything else happening in the loop.
4.  Interrupts and DMA can both be used to improve the response of super loops to external events. They allow hardware peripherals to be serviced without waiting to be polled during a super-loop cycle.
5.  There is only one super loop being run in the system. It shares the system stack. Tasks, however, each receive their own dedicated stack. Each task receives a priority, unlike a superloop, which has no inherent concept of prioritization.
6.  Prioritization.
7.  A preemptive scheduler attempts to ensure that the task with the highest priority is always the one executing.

# Chapter 3

1.  Queues.
2.  Yes.
3.  Semaphore.
4.  Networking stacks or anything where a maximum number of simultaneous users must be enforced.
5.  Priority inheritance.
6.  Mutex.
7.  Priority inversion allows lower-priority tasks to take precedence over a higher-priority task. This is dangerous because it increases the chances of a high-priority task missing a deadline.

# Chapter 4

1.  Firmware programming, especially for MCUs, is extremely low-level, meaning it is very close to the hardware. There are often hardware-specific features that firmware engineers must be familiar with to get the best performance out of an MCU. 
2.  False.
3.  Hardware peripherals.
4.  Rapid prototyping, pre-existing hardware, community, consistent high-level APIs across different MCUs.
5.  Evaluation boards often showcase a product's main differentiating qualities. They are also designed to be as complete as possible, providing easy access to all aspects of a device.
6.  Sleep current, wake-up time, power efficiency (uA/MHz), the functionality of low-power modes, and power supply voltage.
7.  To make it accessible for the widest number of readers – so make sure to get one and work through the exercises on some real hardware!

# Chapter 5

1.  False. The ideal IDE will reflect personal/organizational preferences. A particular IDE that fits well into one team or workflow may not be suitable somewhere else.

2.  False. Many of the freely available IDEs are well suited for professional embedded system development.
3.  False. Vendor-supplied IDEs will often vary widely in their quality. Be careful of getting too tightly bound to a vendor's IDE, especially if you prefer to use MCUs from other vendors.
4.  False. At a minimum, we would expect software-generated code to be syntactically correct the first time. Beyond this, the code generation is only as good as the frontend supplying it, which tends to evolve more slowly than the underlying code bases (so you'll still need to write in customizations later on).
5.  False. The IDE for this book was selected based on cost and only considered compatibility with STM32 devices.
6.  Device selection, hardware bring-up, and middleware integration. *Why* it is useful in each of these areas is covered in the *Considering STMCube* section.

# Chapter 6

1.  False. In this chapter, the ST-Link on the Nucleo development board was re-flashed to provide the same functionality as a J-Link.
2.  False. There are many ways to verify the timing requirements of a real-time system. Segger SystemView provides a means to measure response time, as does looking at system inputs and outputs via a traditional logic analyzer.
3.  False. An RTOS-aware debugger provides the ability to view all of the stacks in the system. This is also an option with any Open GDB-based debugging using Eclipse, as mentioned in the previous chapter.
4.  False. Each module that you write should be tested as thoroughly as possible to minimize any surprises and complex interactions when it is time to integrate the modules and perform a system-level test.
5.  Unit testing. In unit testing, each individual module is tested as it is developed. Integration testing is testing to ensure multiple modules work as expected after they have been "integrated" with one another. System testing tests the complete system (typically after everything has been integrated). Black-box testing is simply a style of testing that assumes nothing about the system inside the "black box," and only compares outputs against the expected behavior given a set of inputs.
6.  **Test-Driven Development** (**TDD**).

# Chapter 7

1.  There are two options – `xTaskCreate()` and `xTaskCreateStatic()`.
2.  True. `xTaskCreate()` may fail if the required memory is not available.
3.  True. `vTaskStartScheduler()` may fail if the required memory for the IDLE task is not available.
4.  False. The required RAM for each task is 64 bytes plus the task stack size. The exact stack size requirements are completely dependent on your code, not FreeRTOS.
5.  False. Tasks can be removed by calling `vTaskDelete()`, provided a compatible heap is used (see [Chapter 15](0f98e454-9804-4589-9854-5c38c9d8d416.xhtml), *FreeRTOS Memory Management*, for details).

# Chapter 8

1.  Synchronization; shared resource protection.
2.  Priority inversion (visit the *Priority inversion (how not to use semaphores) section *for details)
3.  **MUT**utual **EX**clusion, which refers to how access to a shared resource is controlled.
4.  They limit priority inversion by ensuring high-priority tasks block as little as possible by automatically raising the priority of a low-priority task holding a mutex that the high priority task is waiting on.
5.  False. Although easy to use, software timers have limitations including jitter and frequency.

# Chapter 9

1.  A queue can hold any data type, thanks to the underlying `void*` input parameter.
2.  A task waiting to send data to a queue is placed into the blocked state (suspended if `portMAX_DELAY` was specified).

3.  There were three considerations mentioned: data ownership of the underlying value, ensuring the correct data type is passed into the queue, and making sure the data stays intact (by not placing it onto a volatile stack).
4.  False. Task notifications only store a single `uint32_t` and allow a single task with a known task handle to be unblocked. Queues are capable of storing any datatype and can be used across multiple arbitrary tasks.
5.  False. Task notifications only store a single `uint32_t`.  
6.  Speed and RAM efficiency.

# Chapter 10

1.  Interrupt-driven drivers are more complex because there are at least three pieces of code involved (setup code, ISR code, and callback code). With a polled driver, all of this happens serially.  
2.  False. Only functions ending in `FromISR` may be called within an ISR.
3.  False. Interrupts take precedence over the scheduler since the scheduler should be configured to run from the lowest priority interrupt. 
4.  DMA – it uses hardware to transfer data between peripherals and memory, without any CPU intervention.
5.  Direct Memory Access.
6.  Attempting to receive data at any point in time is very difficult to do well with a raw buffer. Raw buffers can also become a bit complex when receiving data of unknown length.

# Chapter 11

1.  False.
2.  False. Timing trade-offs such as increased latency and lower determinism, along with less communication bandwidth, must also be taken into account before deciding a shared hardware peripheral is acceptable.
3.  All of the above.
4.  False. Stream buffers can be used by a single writer and a single reader. These writers and readers don't need to be the same task. If there are multiple writers or multiple readers, then a synchronization mechanism (such as a mutex) is required.
5.  Mutexes.

# Chapter 12

1.  False. Abstractions are useful even in the smallest MCUs.
2.  False. A method for implementing consistent interfaces was presented in this chapter.
3.  Possible answers include the following:
    *   Common components will be reused in other projects.
    *   Portability to different hardware is desirable.
    *   Code will be unit tested.
    *   Teams will be working in parallel.
4.  False. Review the *Avoiding the copy-paste-modify trap* section for more details.
5.  False. When properly written, tasks can be excellent candidates for reuse across projects (see the *Reusing code containing tasks* section for more details).

# Chapter 13

1.  False. Queues create a definitive interface, which decouples components from one another.
2.  False. Any datatype can be placed into a queue.
3.  No, omitting the underlying formatting allows more flexibility for the producers of items to be queued. If the data isn't tied to a specific format, the format can be modified without affecting the queue or the consumer of data coming out of the queue.
4.  Possible answers include the following:
    *   A queued item's lifetime doesn't need to be taken into consideration since a copy of it is made. 
    *   The queued item's scope doesn't need to be taken into account if it is passed by value into the queue. 
    *   If an item is passed by reference, a clear understanding of who *owns* the item is necessary, as well as who is responsible for freeing the resources associated with it.
5.  Possible answers include the following:
    *   Latency introduced by deep queues
    *   Non-deterministic behavior caused by requests sitting in a queue instead of being immediately executed (or rejected)
    *   Memory constraints

# Chapter 14

1.  **CMSIS-RTOS** stands for **Cortex Microcontroller Software Interface Standard - Real-Time Operating System**. The CMSIS-RTOS specification was written by ARM, but there are many vendors that can elect to supply CMSIS-RTOS-compliant interfaces in their RTOSes.
2.  Linux and Android.
3.  False.
4.  False.

# Chapter 15

1.  False.
2.  False.
3.  There are five implementations: `heap_1.c` through `heap_5.c`.
4.  `vApplicationStackOverflowHook` and `vApplicationMallocFailedHook`.
5.  **MPU** stands for **Memory Protection Unit**. It is used to guard against illegal memory access, especially as a way to partition tasks so they are only allowed access to their own memory space.

# Chapter 16

1.  *Multi-core* means multiple cores on the same IC while *multi-processor* means multiple processors (ICs) in the same design.
2.  True. Asymmetric architectures don't require the various processing cores to be treated in the same way, so any combination of operating systems and bare-metal programming languages can be used (within the restrictions of the hardware).
3.  False. There are many aspects to consider when selecting the *best* bus for a given application since each application will have its own set of unique circumstances and requirements.
4.  The additional complexity needs to be weighed against the possibility of not performing the same work twice. When reusable subsystems are developed, they can create considerable cost savings under the right circumstances. They have little to no **nonrecurring engineering** (**NRE**) costs associated with them when re-used.

# Chapter 17

1.  You should do the following:
    1.  Connect a debugger.
    2.  Figure out where the program stopped.
    3.  If it is a `configASSERT`, read the comments surrounding the assertion. If it fails before the scheduler starts, you've likely overflowed your FreeRTOS heap.
2.  Any one of the following:
    *   Task stack overflows
    *   Misprioritized ISRs
    *   Inadequate heap size
3.  False. Debugging tools such as Segger SystemView exist, which provide both printf-style output as well as instrumentation for observing the code's behavior.