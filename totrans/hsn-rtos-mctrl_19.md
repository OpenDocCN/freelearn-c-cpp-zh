# FreeRTOS Memory Management

So far, we've worked through many examples of creating FreeRTOS primitives; however, when these primitives were initially created, there wasn't much of an explanation as to where the memory was coming from. In this chapter, we'll learn exactly where the memory comes from, along with when and how it is allocated. Choosing when and how memory is allocated allows us to make trade-offs between coding convenience, timing determinism, potential regulatory requirements, and code standards. We'll conclude by looking at different measures that can be taken to ensure application robustness.

In a nutshell, this chapter covers the following:

*   Understanding memory allocation
*   Static and dynamic allocation of FreeRTOS primitives
*   Comparing FreeRTOS heap implementations
*   Replacing `malloc` and `free`
*   Implementing FreeRTOS memory hooks
*   Using a **memory protection unit** (**MPU**)

# Technical requirements

To complete the hands-on exercises in this chapter, you will require the following:

*   A Nucleo F767 development board
*   A Micro-USB cable
*   STM32CubeIDE and source code (see the instructions in [Chapter 5](84a945dc-ff6c-4ec8-8b9c-84842db68a85.xhtml), *Selecting an IDE,* under the section *Setting up our IDE*)

*   SEGGER JLink, Ozone, and SystemView (see the instructions in [Chapter 6](699daa80-06ae-4acc-8b93-a81af2eb774b.xhtml), *Debugging Tools for Real-Time Systems)*

All source code for this chapter is available from [https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_15](https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_8).

# Understanding memory allocation

Memory allocation isn't necessarily at the top of a developer's list of favorite topics to consider when developing an application—it just isn't all that glamorous. Dynamic allocation of memory—that is, allocating memory as it is needed rather than at the beginning of the program—is the norm. With desktop-oriented development, memory is generally available whenever it is needed, so it isn't given a second thought; it is simply a `malloc` call away. When it is finished, it will be unallocated with `free`.

Unlike the carefree dynamic memory allocation schemes in a desktop environment, programmers of deeply embedded systems that use MCUs will often need to be more careful about how (and if) memory is dynamically allocated. In an embedded system, regulatory, RAM, and timing constraints can all play a role in whether/how memory can be dynamically allocated.

Many high-reliability and safety-critical coding standards, such as MISRA-C, will not allow the use of dynamic allocation. In this case, it is still perfectly acceptable to use static allocation. Some coding standards disallow dynamic allocation after all tasks are created (*Ten Rules for Safety Critical Coding* by *JPL*, for example). In this case, static allocation or FreeRTOS's `heap_1.c` implementation would be reasonable.

RAM may be severely limited on a given platform. On the surface, this seems like the perfect use case for dynamic memory allocation; after all, if there is limited memory, it can be given back when it's not in use! In practice, however, things don't always go this smoothly when there is limited heap space available. When small heaps are used to allocate space for arbitrarily sized objects with different lifetimes, fragmentation will often occur eventually (this will be covered in more depth with an example later).

Finally, a need for highly deterministic timing can also limit the options for dynamic allocation. If a portion of code has tight timing constraints, it is sometimes easier to avoid using dynamic allocation, rather than contriving tests that attempt to mimic worst-case timing for a call to `malloc`. It is also worth noting (again) that `malloc` isn't guaranteed to succeed, *especially* on an embedded system with limited memory. Having a large amount of dynamic allocation in a memory-constrained, multithreaded system can create some very complex use cases that have the potential to fail at runtime. Thoroughly testing such a system is a very serious challenge.

With this background information on why memory allocation is so important in constrained embedded systems, let's take a closer look at *where* memory comes from in a few different use cases.

# Static memory

Static memory's lifespan is the entire duration of a program. Global variables, as well as any variables declared inside functions using the `static` specifier, will be placed into static memory and they will have a lifetime equal to that of the program.

For example, both `globalVar` and `staticVar` are located in static memory and will persist for the entire lifetime of the program. The initialization of `staticVar` only occurs once during the initial program load:

```cpp
uint8_t globalVar = 12;

void myFunc( void )
{
  static uint8_t staticVar = 0;
  ...
}
```

When variables are declared as static, memory is guaranteed to be available. All of the global and static variables defined by the program are placed into their locations during the linking phase. As long as the amount of memory has been properly configured, the linker guarantees that space is available for these variables.

The downside is that because static variables have such a long lifespan, static variables will *always* be consuming space, even when they are not being used.

# Stack memory

A stack is used for function-scoped storage. Each time a function is called, information for that function (like its parameters and local variables) are placed onto a stack. When the function exits, all of the information that was placed onto the stack is removed (this is why passing pointers to local variables is a problem). In FreeRTOS, each task has its own private stack whose size is determined when the task is created.

Since stacks have such an orderly way of being accessed, it isn't possible for them to become fragmented, like a heap. It is possible, however, to overflow the stack by placing more information onto it than its size allows.

On the Cortex-M, there is also one additional stack—the main stack. The main stack is used by ISRs and the FreeRTOS kernel. The kernel and ISRs execute in a privileged mode that modifies the **main stack pointer** (**MSP**). Tasks execute on the process stack and use the **process stack pointer** (**PSP**). All of the stack pointer operations are taken care of by hardware and the kernel depending on whether the kernel, interrupt, or task (process) is currently being executed. It is not something that users of the RTOS API will normally need to worry about.

Initialization of the stack and heap takes place in `Chapter_*\startup\startup_stm32f767xx.s`. The exact size of the main stack is defined in the linker script `STM32F767ZI_FLASH.ld`. If necessary, the size of the stack and heap available to the system before the FreeRTOS scheduler is started can be adjusted by modifying `_Min_Heap_Size` or `_Min_Stack_Size`:

```cpp
_Min_Heap_Size = 0x200; /* required amount of heap */
_Min_Stack_Size = 0x400; /* required amount of stack */
```

It is best to try and keep these both to minimal sizes since any RAM used here will be unavailable to the tasks. These stacks/heaps are only for code that is run before the scheduler is started, as well as the ISRs. This is *not* the same stack that is used by any of the tasks.

Occasionally, you may run into a problem where you'll need to run some memory-intensive initialization code (the USB stack is a good example of this). If the initialization functions are called outside of a task (before the scheduler starts), then they will use the main stack. In order to keep this stack as small as possible and allow more memory to be used for tasks, move memory-intensive initialization inside a task This will allow the RTOS heap to have the additional RAM that would have gone unused after initialization had the main stack size been increased.

The FreeRTOS kernel manipulates the **process stack pointer** (**PSP**) to point to the task stack that has context (is in the running state).

For the most part, you won't need to be immediately concerned with the various stack pointers—they are taken care of by the kernel and C runtime. If you happen to be developing code that will transition between an RTOS and *bare metal* (that is, a bootloader), then you will need to understand how/when to properly switch the current stack pointer.

The most important thing to keep in mind with stacks is that they must be adequately sized to hold all of the local variables that a task will execute for the deepest call level. We'll discuss ways of getting a handle on this in the *Keeping an eye on stack space* section.

# Heap memory

The heap is the portion of memory that is used when a dynamic allocation using `malloc` is called. It is also where a FreeRTOS task stack and **task control block** (**TCB**) are stored when they are created by calling `xTaskCreate()`.

In an MCU FreeRTOS system, there will typically be two heaps created:

*   **System heap**: Defined in the startup and linker scripts described previously. This will *not *be available for use by the final application code when allocating space for RTOS primitives.
*   **FreeRTOS heap**: Used when creating tasks and other primitives and defined in ***Inc\****FreeRTOSConfig.h***. It can be resized by adjusting the following line:

```cpp
#define configTOTAL_HEAP_SIZE ((size_t)15360)
```

Currently, this line is defining a 15 KB heap. This heap must be adequately sized to accommodate the following:

*   Stacks (and **TCBs**) for all tasks that are created using `xTaskCreate`
*   Queues, semaphores, mutexes, event groups, and software timers created using `x*Create`

Here's a visual representation of where all of the different variables will come from:

![](img/5ac3322f-699f-4f2e-b212-06791ecef725.png)

There are two possible locations for FreeRTOS primitives and stacks:

*   Statically allocated space for a stack and a **TCB**, passed to a task when calling `xTaskCreateStatic()`
*   Dynamically allocated space for a stack/TCB, created when calling `xTaskCreate()`

The C heap is only used for any items that are created without the use of the FreeRTOS heap implementation, while the C stack is only used before the scheduler is started, as well as by ISRs. When using an RTOS, it is best to minimize the size of the C heap as much as possible, or entirely. This will leave more available RAM to allocate to the RTOS heap or static variables.

# Heap fragmentation

In an embedded system with limited RAM, heap fragmentation can be a very serious issue. A heap becomes fragmented when items are loaded into the heap and then removed at different points in time. The problem is that if many items that are being removed aren't adjacent to one another, a larger contiguous region of space won't necessarily be available:

![](img/94bc05c1-17f7-40b2-97a6-1f9d3d1de749.png)

In the preceding example, space won't be successfully allocated for item 8\. Even though there is sufficient free space, there isn't enough *contiguous* free space to accommodate the size of item 8\. This is especially problematic because it will only occur at runtime, and under certain circumstances that are dependent on the size and timing of when items in the heap are allocated and freed.

Now that we've covered the basics of memory allocation, let's look at some different ways that FreeRTOS primitives can be created to be placed in static or heap memory.

# Static and dynamic allocation of FreeRTOS primitives

Details on the mechanics of creating tasks were covered in [Chapter 7](2fa909fe-91a6-48c1-8802-8aa767100b8f.xhtml), *The FreeRTOS Scheduler.* Here, we will only focus on the differences in *where* the memory is coming from and what its *lifetime* is. This will help illuminate the implications of choosing different allocation schemes.

Memory for tasks can either be allocated dynamically or statically. Dynamic allocation allows the memory used by the task to be returned by calling `vTaskDelete()` if the task no longer needs to run (see [Chapter 7](2fa909fe-91a6-48c1-8802-8aa767100b8f.xhtml), *The FreeRTOS Scheduler*, for details). Dynamic allocation can occur at any point in the program, whereas static allocation occurs before the program starts. The static variants of FreeRTOS API calls follow the same initialization scheme—the standard calls use dynamic allocation (pulling memory from the FreeRTOS heap). All FreeRTOS API functions with `CreateStatic` in their names (such as `xTaskCreateStatic`) take additional arguments for referencing preallocated memory. As opposed to the dynamic allocation approach, the memory passed to `*CreateStatic` variants will typically be statically allocated buffers, which are present for the entire program's lifetime.

While the naming of the `*CreateStatic` API variants suggests that the memory is static, this isn't actually a requirement. For example, you could allocate a buffer memory on the stack and pass the pointer to a `*CreateStatic` API function call; however, you'll need to be sure that the lifetime of the primitive created is limited to that function! You may also find it useful to allocate memory using an allocation scheme outside of the FreeRTOS heap, in which case you could also use the `*CreateStatic` API variants. If you choose to utilize either of these methods, then to avoid memory corruption, you'll need to have detailed knowledge of the lifetime of both the FreeRTOS primitive being created and the allocated memory!

# Dynamic allocation examples

Nearly all of the code presented has used dynamic allocation to create FreeRTOS primitives (tasks, queues, mutexes, and so on). Here are two examples to serve as a quick refresher before we look at the differences between creating primitives using static allocation.

# Creating a task

When a task is created using dynamically allocated memory, the call will look something like this (see [Chapter 7](2fa909fe-91a6-48c1-8802-8aa767100b8f.xhtml), *The FreeRTOS Scheduler*, for more details on the parameters that are not related to memory allocation):

```cpp
BaseType_t retVal = xTaskCreate( Task1, "task1", StackSizeWords, NULL,
        tskIDLE_PRIORITY + 2, tskHandlePtr);
assert_param(retVal != pdPASS);
```

There are a few relevant pieces of information, relevant to memory allocation, to note about this call:

*   The call to `xTaskCreate` may fail. This is because there is no guarantee that enough space will be available for storing the task's stack and TCB on the FreeRTOS heap. The only way to ensure that it was created successfully is to check the return value, `retVal`.
*   The only parameter to do with a stack is the requested size of the stack.

When created in this manner, if it is appropriate for a task to terminate itself, it may call `xTaskDelete(NULL)` and the memory associated with the task's stack and TCB will be available to be reused.

The following are a few points to note regarding dynamic allocation:

*   Primitive creation may fail at runtime if no heap space is available.
*   All memory that FreeRTOS allocates for the primitive will be automatically freed when the task is deleted (as long as `Heap_1` is not used and `INCLUDE_vTaskDelete` is set to `1` in `FreeRTOSConfig.h`). This doesn't include memory that was dynamically allocated by *user*-supplied code in the actual task; the RTOS is unaware of any dynamic allocation initiated by user-supplied code. It is up to you to free this code when appropriate.

*   `configSUPPORT_DYNAMIC_ALLOCATION` must be set to 1 in `FreeRTOSConfig.h` for dynamic allocation to be available:

![](img/f9fad1c7-0e46-4cf6-92ee-b12a7bbdae96.png)

When creating a task using dynamic allocation, all of the memory used for the task, the task's stack, and **TCB** is allocated from the FreeRTOS heap, as shown in the preceding diagram.

Next, let's take a look at the different ways of creating queues.

# Creating a queue

For a detailed explanation and working examples regarding how to create queues using dynamically allocated memory, see [Chapter 9](495bdcc0-2a86-4b22-9628-4c347e67e49e.xhtml), *Intertask *Communication, in the section on *Passing data through queues by value*. As a quick review, to create a queue of length `LED_CMD_QUEUE_LEN` that holds elements of the `uint8_t` type, we'd go through the following steps:

1.  Create the queue:

```cpp
ledCmdQueue = xQueueCreate(LED_CMD_QUEUE_LEN,
                            sizeof(uint8_t));
```

2.  Verify that the queue was created successfully by checking the handle, `ledCmdQueue`, is not `NULL`:

```cpp
assert_param(ledCmdQueue != NULL);
```

Now that we've reviewed a few examples of dynamic allocation (which will pull memory during runtime from the FreeRTOS heap), let's move on to static allocation (which reserves memory during compilation/linking, before the application is ever run).

# Static allocation examples

FreeRTOS also has a method for creating primitives that don't require us to dynamically allocate memory. This is an example of creating primitives with statically allocated memory.

# Creating a task

To create a task using a preallocated stack and TCB (requiring no dynamic allocation), use a call similar to the following:

```cpp
StackType_t GreenTaskStack[STACK_SIZE];
StaticTask_t GreenTaskTCB;
TaskHandle_t greenHandle = NULL;
greenHandle = xTaskCreateStatic(    GreenTask, "GreenTask", STACK_SIZE,
                                    NULL, tskIDLE_PRIORITY + 2,
 GreenTaskStack, &GreenTaskTCB);
assert_param( greenHandle != NULL );
```

There are several notable differences between this static allocation and the previous method of dynamic allocation:

*   Instead of a return value of `pdPASS`, the `xTaskCreateStatic` function returns a task handle.
*   Task creation using `xTaskCreateStatic` will always succeed, provided that the stack pointer and TCB pointers are non-null.
*   As an alternative to checking the `TaskHandle_t`, `StackType_t` and `StaticTask_t` could be checked instead; as long as they are not `NULL`, the task will always be successfully created.
*   Tasks can also be *deleted*, even if they were created with `xTaskCreateStatic`. FreeRTOS will only take the steps necessary to remove the task from the scheduler; freeing associated memory is the responsibility of the caller.

Here's where the task's stack and TCB are located when we use the previous call:

![](img/615b2e6c-0f53-4b53-a793-463d43079c13.png)

Static creation allows more flexibility in memory allocation than the name implies. Strictly speaking, a call to `vTaskDelete` will only remove a statically created task from the schedule. Since FreeRTOS will no longer access memory from that task's stack or TCB, it is safe to repurpose this memory for other purposes. It is conceivably possible to allocate the stack and TCB from the stack memory rather than static memory. An example of deleting a task created using `xTaskCreateStatic` can be found in `main_staticTask_Delete.c`.

# Creating a queue

Now let's take a look at the steps for creating a queue using static memory for both the buffer and queue structure. This code is an excerpt from `mainStaticQueueCreation.c`:

1.  Define a variable for holding the queue structure used by FreeRTOS:

```cpp
static StaticQueue_t queueStructure;
```

2.  Create a raw array, appropriately sized to hold the queue contents:

    *   A simple C array of the target datatype can be used; in this case, our queue will hold a datatype of `uint8_t`.
    *   Use `#define` to define the array length:

```cpp
#define LED_CMD_QUEUE_LEN 2
static uint8_t queueStorage[LED_CMD_QUEUE_LEN];
```

3.  Create the queue within the same length as the array that was previously defined:

```cpp
ledCmdQueue = xQueueCreateStatic(LED_CMD_QUEUE_LEN,                                   sizeof(uint8_t), queueStorage, &queueStructure );
```

Here's a breakdown of the parameters:

*   *   `LED_CMD_QUEUE_LEN`: Number of elements in the queue
    *   `sizeof(uint8_t)`: Size of each element (in bytes)
    *   `queueStorage`: Raw array used for storing elements in the queue (used only by FreeRTOS)
    *   `queueStructure`: Pointer to `StatisQueue_t`, the queue structure used internally by FreeRTOS

4.  Check the queue handle `ledCmdQueue`, to ensure that the queue was correctly created by verifying that it is not `NULL`. Unlike dynamically allocated queues, it is unlikely that this call will fail, but leaving the check ensures that if the queue is ever changed to be dynamically allocated, errors will still be caught:

```cpp
assert_param(ledCmdQueue != NULL);
```

5.  Put it all together:

```cpp
static QueueHandle_t ledCmdQueue = NULL;
static StaticQueue_t queueStructure;
#define LED_CMD_QUEUE_LEN 2
static uint8_t queueStorage[LED_CMD_QUEUE_LEN];
ledCmdQueue = xQueueCreateStatic(LED_CMD_QUEUE_LEN,
            sizeof(uint8_t),
            queueStorage, &queueStructure );
assert_param(ledCmdQueue != NULL);
```

The only difference between creating queues with static allocation and creating them with dynamic allocation is how the memory is supplied—both calls return queue handles. Now that we've seen examples of creating queues and tasks without using dynamically allocated memory, what happens if we have a requirement for *no* dynamic allocation to take place?

# Eliminating all dynamic allocation

In most of the examples that we've seen, we've focused on working with the dynamic allocation scheme variants when creating FreeRTOS primitives. This has been primarily for ease of use and brevity, enabling us to focus on the core RTOS concepts rather than worrying about exactly where memory was coming from and how we were accessing it.

All FreeRTOS primitives can be created with either dynamically allocated memory or preallocated memory. To avoid all dynamic allocation, simply use the `CreateStatic` version of a `create` function, as we've done in the preceding example when we created a task. Some `CreateStatic` versions exist for queues, mutexes, semaphores, stream buffers, message buffers, event groups, and timers. They share the same arguments as their dynamic counterparts, but also require a pointer to preallocated memory to be passed to them. The `CreateStatic` equivalents don't require any memory allocation to take place during runtime.

You would consider using the static equivalents for the following reasons:

*   They are guaranteed to never fail because of a lack of memory.
*   All of the checks that are needed to ensure memory is available happen during linking (before the application binary is created). If memory is not available, it will fail at link time, rather than runtime.
*   Many standards targeting safety-critical applications prohibit the use of dynamically allocated memory.
*   Internal embedded C coding standards will occasionally prohibit the use of dynamic allocation.

Memory fragmentation could be added to this list as well, but this isn't an issue unless memory is freed (for example, `heap_1` could be used to eliminate heap fragmentation concerns).

Now that we have an understanding of the differences between dynamic and static allocation, let's dive into the differences of FreeRTOS's dynamic allocation schemes—the five heap implementations. Moving ahead, we will see what these different definitions look like in a file (globals, static allocation, and so on). We will also understand the difference between the main stack and task-based stacks where they live in a FreeRTOS heap.

# Comparing FreeRTOS heap implementations

Because FreeRTOS targets such a wide range of MCUs and applications, it ships with five different dynamic allocation schemes, all of which are implemented with a heap. The different heap implementations allow different levels of heap functionality. They are included in the `portable/MemMang` directory as `heap_1.c`, `heap_2.c`, `heap_3.c`, `heap_4.c`, and `heap_5.c`.

**A note on memory pools:** 
Many other RTOSes include memory pools as an implementation for dynamic memory allocation. A memory pool achieves dynamic allocation by only allocating and freeing fixed-size blocks. By fixing the block size, the problem of fragmentation is avoided in memory-constrained environments.

The downside to memory pools is that the blocks need to be sized for each specific application. If they are too large, they will waste precious RAM; too small, and they'll be unable to hold large items. In order to make things easier on users and avoid wasting RAM, Richard Barry elected to exclusively use heaps for dynamic allocation in FreeRTOS.

In order for projects to properly link after compilation, it is important to only have one of the heap implementations visible to the linker. This can either be accomplished by removing the unused files or not including the unused heap files in the list of files available to the linker. For this book, the extra files in `Middleware\Third_Party\FreeRTOS\Source\portable\MemMang` have been removed. For this chapter, however, all of the original implementations are included in `Chapter_15\Src\MemMang`: it is the only place where examples use a heap other than `heap_4.c`.

All of the various heap options exist to enable a project to get exactly the functionality it needs without requiring anything more (in terms of program space or configuration). They also allow for trade-offs between flexibility and deterministic timing. The following is a list of the various heap options:

*   `heap_1`: Allocation only—no freeing is allowed. This is best suited for simple applications that don't free anything after initial creation. This implementation, along with `heap_2`, provides the most deterministic timing since neither heap ever performs a search for adjacent free blocks to combine.
*   `heap_2`: Allocation and freeing are both allowed, but adjacent free blocks are not combined. This limits appropriate use cases to those applications that can know/guarantee they are reusing a number of items that are the same size each time. This heap implementation is not a great fit for applications that make use of `vPortMalloc` and `vPortFree` explicitly (for example, applications that allocate memory dynamically themselves), unless there is a very large degree of discipline in ensuring that only a small subset of possible sizes are used.
*   `heap_3`: Wraps standard `malloc`/`free` implementations to provide thread safety.
*   `heap_4`: The same as `heap_2` but combines adjacent free space. Allows locating the entire heap by giving an absolute address. Well suited for applications to use dynamic allocation.
*   `heap_5`: The same as `heap_4` but allows for creating a heap that is distributed across different noncontiguous memory regions—for example, a heap could be scattered across internal and external RAM.

Here's a quick comparison between all of the heap implementations:

| **Heap name** | **Thread safe** | **Allocation** | **Free** | **Combine adjacent free space** | **Multiple memory regions** | **Determinism** |
| `heap_1.c` | ✓ | ✓ |  |  |  | ↑ |
| `heap_2.c` | ✓ | ✓ | ✓ |  |  | ↑ |
| `heap_3.c` | ✓ | ✓ | ✓ | ✓* |  | ? |
| `heap_4.c` | ✓ | ✓ | ✓ | ✓ |  | → |
| `heap_5.c` | ✓ | ✓ | ✓ | ✓ | ✓ | → |
| `std C lib` | ? | ✓ | ✓ | ✓* |  | ? |

(*) Most, if not all, included heap implementations will combine free space.

Since determinism is dependent on the C library implementation that we happen to be using, it isn't possible to provide general guidance here. Typically, general-purpose heap implementations are created to minimize fragmentation, which requires additional CPU resources (time) and decreases the determinism of the timing, depending on how much memory is moved around.

Each C implementation may approach dynamic allocation differently. Some will make adding thread safety as easy as defining implementations for `__mallock_lock` and `__malloc_unlock`, in which case a single mutex is all that is required. In other cases, they will require a few implementations for implementing mutex functionality.

# Choosing your RTOS heap implementation

So, how do you go about choosing which heap implementation to use? First, you need to ensure that you're able to use dynamic allocation (many standards for safety-critical applications disallow it). If you don't need to free allocated memory, then `heap_1.c` is a potential option (as is avoiding a heap entirely).

From a coding perspective, the main difference between using `heap_1` and static allocation is when the checks for memory availability are performed. When using the `*CreateStatic` variants, you'll be notified at link time that you don't have enough memory to support the newly created primitive. This requires a few extra lines of code each time a primitive is created (to allocate buffers used by the primitive). When using `heap_1`, as long as checks are performed (see [Chapter 7](2fa909fe-91a6-48c1-8802-8aa767100b8f.xhtml), *The FreeRTOS Scheduler*) to determine task creation success, then the checking will be performed at runtime. Many applications that are appropriate for the `heap_1` implementation will also create all required tasks before starting the scheduler. Using dynamic memory allocation in this way isn't much different from static allocation; it simply moves the checking from link time to runtime, while reducing the amount of code required to create each RTOS primitive.

If you're working on an application that only requires *one* datatype to be freed, `heap_2` might be an option. If you choose to go down this route, you'll need to be very careful to document this limitation for future maintainers of the code. Failure to understand the limited use case of `heap_2` can easily result in memory fragmentation. In the worst-case scenario, fragmentation might potentially occur after the application has been running for an extended period of time and might not occur until the final code has been released and the hardware is fielded.

When dynamic memory is used, then `heap_3`, `heap_4`, or `heap_5` can be used. As mentioned earlier, `heap_3` simply wraps whatever C runtime implementation of `malloc` and `free` is available, to make it thread safe so it can be used by multiple tasks. This means that its behavior is going to be dependent on the underlying runtime implementation. If your system has RAM in several different, noncontiguous memory locations (for example, internal and external RAM) then `heap_5` can be used to combine all of these locations into one heap; otherwise, `heap_4` provides the same allocation, freeing, and adjacent block collation capabilities as `heap_5`. These are the two general-purpose heap implementations. Since they include code that will collate free blocks, it is possible that they will run for different periods of time when freeing memory. In general, it is best to avoid calls to `vPortMalloc` and `vPortFree` in code that requires a high degree of determinism. In `heap_4` and `heap_5`, calls to `vPortFree` will have the most amount of timing variability, since this is when adjacent block collation occurs.

In general, avoiding dynamic allocation will help to provide more robust code with less effort—memory leaks and fragmentation are impossible if memory is never freed. On the other end of the spectrum, if your application makes use of standard library functions, such as `printf` and string manipulation, you'll likely need to replace the versions of `malloc` and `free` that were included with thread-safe implementations. Let's take a quick look at what's involved in making sure other parts of the application don't end up using a heap implementation that isn't thread safe.

# Replacing malloc and free

Many C runtimes will ship with an implementation of `malloc`, but the embedded, oriented versions won't necessarily be thread safe by default. Because each C runtime is different, the steps needed to make `malloc` thread safe will vary. The included STM toolchain used in this book includes `newlib-nano` as the C runtime library. The following are a few notes regarding `newlib-nano`:

*   `newlib-nano` uses `malloc` and `realloc` for `stdio.h` functionality (that is, `printf`).
*   `realloc` is not directly supported by FreeRTOS heap implementations.
*   `FreeRTOSConfig.h` includes the `configUSE_NEWLIB_REENTRANT` setting to make `newlib` thread safe, but it needs to be used in conjunction with the appropriate implementations of all stubs. This will allow you to use newlib-based `printf`, `strtok`, and so on in a thread-safe manner. This option also makes general use case calls to `malloc` and `free` safe to use from anywhere, without you needing to explicitly use `pvPortMalloc` and `vPortFree`.

See the Dave Nadler link in the *Further reading* section for more information and detailed instructions on how to use `newlib` safely in a FreeRTOS project with the GNU toolchain.

Luckily, there aren't any calls to raw `malloc` in the example code included in this book. Normally, the STM HAL USB CDC implementation would include a call to `malloc`, but this was converted to a statically defined variable instead, which enables us to simply use the heap implementations included with FreeRTOS.

The `malloc` call in the STM-supplied USB stack was especially sinister because it occurred inside the USB interrupt, which makes it especially difficult to guarantee thread safety during `malloc`. This is because, for every call to `malloc`, interrupts would need to be disabled from within the tasks and also within interrupts that made calls to `malloc` (USB in this case). Rather than go through this trouble, the dynamic allocation was removed altogether.

Now that we've come to terms with different safety options using dynamic allocation, let's take a look at some additional tools that FreeRTOS has for reporting the health of our stacks and heaps.

# Implementing FreeRTOS memory hooks

When many people first start programming in an RTOS, one of the immediate challenges is figuring out how to properly size the stack for each task. This can lead to some frustration during development because when a stack is overrun, the symptoms can range from odd behavior to a full system crash.

# Keeping an eye on stack space

`vApplicationStackOverflowHook` provides a very simple way of eliminating most of the oddball behavior and halting the application. When enabling `configCHECK_FOR_STACK_OVERFLOW #define` in `FreeRTOSConfig.h`, any time a stack overflow is detected by FreeRTOS, `vApplicationStackOverflowHook` will be called.

There are two potential values for `configCHECK_FOR_STACK_OVERFLOW`:

*   `#define configCHECK_FOR_STACK_OVERFLOW 1`: Checks the stack pointer location upon task exit.
*   `#define configCHECK_FOR_STACK_OVERFLOW 2`: Fills the stack with a known pattern and checks for the pattern upon exit.

The first method checks the task stack pointer as the task exits the running state. If the stack pointer is pointing to an invalid location (where the stack shouldn't be), then an overflow has occurred:

![](img/1207b1ce-d347-4cc1-96bb-7285577c892a.png)

This method is very fast, but it has the potential to miss some stack overflows—for example, if the stack has grown beyond its originally allocated space, but the stack pointer happens to be pointing to a valid spot when checked, then the overflow will be missed. To combat this, a second method is also available.

When setting `configCHECK_FOR_STACK_OVERFLOW` to 2, method 1 will be used, but a second method will also be employed. Instead of simply checking where the stack pointer is located after the task has exited the running state, the top 16 bytes of the stack can be watermarked and analyzed upon exit. This way, if at any point during the task run the stack has overflowed and the data in the top 16 bytes has been modified, an overflow will be detected:

![](img/ec24bea1-bff4-4004-9cc1-ab96e11ad5f5.png)

This method helps to ensure that, even if a stack overflow has occurred (or nearly occurred) at any point during the task execution, it will be detected, as long as the overflow passed through the upper 16 words of the stack.

While these methods are good for catching stack overflows, they are not perfect—for example, if an array is declared on a task stack and extends past the end of the stack with only the end of the array being modified, then a stack overflow won't be detected.

So, to implement a very simple hook that will stop execution when a stack overflow occurs, we'll take the following simple steps:

1.  In `FreeRTOSConfig.h`, define the configuration flag:

```cpp
#define configCHECK_FOR_STACK_OVERFLOW 2
```

2.  In a `*.c` file, add the stack overflow hook:

```cpp
void vApplicationStackOverflowHook( void )
{
 __disable_irq();
 while(1);
}
```

This very simple method disables all interrupts and executes an infinite loop, leaving no question that something has gone wrong. At this point, a debugger can be used to analyze which stack has overflowed.

# Keeping an eye on heap space

If your application makes regular use of the FreeRTOS heap, then you should strongly consider using the `configUSE_MALLOC_FAILED_HOOK` configuration and associated hook, `vApplicationMallocFailedHook`. This hook is called anytime a call to `pvMalloc()` fails.

Of course, while you're doing this, you're being a responsible programmer and checking the return value of `malloc` and handling these error cases anyway... so this hook may be redundant.

The steps for setting this up are the same as in the previous hook:

1.  Add the following in `FreeRTOSConfig.h`:

```cpp
#define configUSE_MALLOC_FAILED_HOOK 1
```

2.  In a `*.c` file, add the failed `malloc` hook:

```cpp
void vApplicationMallocFailedHook( void )
{
 __disable_irq();
 while(1);
}
```

There are also two helpful API functions that can be called on a regular basis to help get a general sense of the available space:

*   `xPortGetFreeHeapSize()`
*   `xPortGetMinimumEverFreeHeapSize()`

These functions return the available heap space and the least amount of free heap space ever recorded. They don't, however, give any clue as to whether or not the free space is fragmented into small blocks.

So, what happens if none of these safeguards provide enough peace of mind that each of your tasks is playing nicely with the rest of the system? Read on!

# Using a memory protection unit (MPU)

A **memory protection unit** (**MPU**) continuously monitors memory access at a hardware level to make absolutely certain that only legal memory accesses are occurring; otherwise, an interrupt is raised and immediate action can be taken. This allows many common errors (which might otherwise go unnoticed for a period of time) to be immediately detected.

Problems like stack overflows that make a stack flow into the memory space reserved for another task are immediately caught when using an MPU, even if they can't be detected by `vApplicationStackOverflowHook`. Buffer overflows and pointer errors are also stopped dead in their tracks when an MPU is utilized, which makes for a more robust application.

The STM32F767 MCU includes an MPU. In order to make use of it, the MPU-enabled port must be used: `GCC\ARM_CM4_MPU`. This way, restricted tasks can be created by using `xTaskCreateRestricted`, which contains the following additional parameters:

```cpp
typedef struct xTASK_PARAMTERS
{
 pdTASK_CODE pvTaskCode;
 const signed char * const pcName;
 unsigned short usStackDepth;
 void *pvParameters;
 unsigned portBASE_TYPE uxPriority;
 portSTACK_TYPE *puxStackBuffer;
 xMemoryRegion xRegions[ portNUM_CONFIGURABLE_REGIONS ];
} xTaskParameters; 
```

Restricted tasks have limited execution and memory access rights.

`xTaskCreate` can be used to create tasks that either operate as standard user mode tasks or privileged mode tasks. In privileged mode, a task has access to the entire memory map, whereas in user mode, it only has access to its own flash and RAM that isn't configured for privileged access only.

In order for all of this to come together, the MPU ports of FreeRTOS also require variables to be defined in the linker file:

| **Variable name** | **Description** |
| `__FLASH_segment_start__` | The start address of the flash memory |
| `__FLASH_segment_end__` | The end address of the flash memory |
| `__privileged_functions_end__` | The end address of the `privileged_functions` named section |
| `__SRAM_segment_start__` | The start address of the SRAM memory |
| `__SRAM_segment_end__` | The end address of the SRAM memory |
| `__privileged_data_start__` | The start address of the `privileged_data` section |
| `__privileged_data_end__` | The end address of the `privileged_data` section |

These variables will be placed into the `*.LD` file.

Congratulations! You're now ready to develop your application using the MPU to protect against invalid data access.

# Summary

In this chapter, we've covered static and dynamic memory allocation, all of the available heap implementations in FreeRTOS, and how to implement memory hooks that let us keep an eye on our stacks and heaps. By understanding the trade-offs to be made when using the different allocation schemes, you'll be in a good position to choose the most appropriate method for each of your future projects.

In the next chapter, we'll discuss some of the details of using FreeRTOS in a multicore environment.

# Questions

As we conclude, here is a list of questions for you to test your knowledge regarding this chapter's material. You will find the answers in the *Assessments* section of the Appendix:

1.  With FreeRTOS, using dynamically allocated memory is extremely safe because it guards against heap fragmentation:
    *   True
    *   False
2.  FreeRTOS requires dynamically allocated memory to function:
    *   True
    *   False
3.  How many different heap implementations ship with FreeRTOS?
4.  Name two hook functions that can be used to notify you about problems with the heap or stack.
5.  What is an MPU used for?

# Further reading

*   *The Power of 10: Rules for Developing Safety-Critical Code* by *Gerard J. Holzmann*: [http://web.eecs.umich.edu/~imarkov/10rules.pdf](http://web.eecs.umich.edu/~imarkov/10rules.pdf)
*   Dave Nadler – newlib and FreeRTOS re-entry: [http://www.nadler.com/embedded/newlibAndFreeRTOS.html](http://www.nadler.com/embedded/newlibAndFreeRTOS.html)
*   FreeRTOS stack overflow checking: [https://www.freertos.org/Stacks-and-stack-overflow-checking.html](https://www.freertos.org/Stacks-and-stack-overflow-checking.html)