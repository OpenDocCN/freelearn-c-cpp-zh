# Choosing an RTOS API

So far, we've only used the native FreeRTOS API in all of our examples. However, this isn't the only API available for using FreeRTOS. Sometimes, there are secondary goals when developing code – it might need to be reused across other projects with other MCU-based embedded operating systems. Other times, code needs to be interoperable with fully featured operating systems. You may also want to utilize code that has been previously developed for a full operating system. In order to support these goals, there are two other APIs for FreeRTOS that are worth considering alongside the native API – CMSIS-RTOS and POSIX. 

In this chapter, we'll investigate the features, trade-offs, and limitations of these three APIs when creating applications based on FreeRTOS. 

This chapter covers the following topics:

*   Understanding generic RTOS APIs
*   Comparing FreeRTOS and CMSIS-RTOS
*   Comparing FreeRTOS and POSIX
*   Deciding which API to use

# Technical requirements

To complete the hands-on exercises in this chapter, you will require the following:

*   A Nucleo F767 dev board
*   A micro-USB cable
*   STM32CubeIDE and source code (for instructions, visit [Chapter 5](84a945dc-ff6c-4ec8-8b9c-84842db68a85.xhtml), *Selecting an IDE, *and read the section *Setting up our IDE*
*   SEGGER JLink, Ozone, and SystemView (for instructions, read [Chapter 6](699daa80-06ae-4acc-8b93-a81af2eb774b.xhtml), *Debugging Tools for Real-Time Systems*)

All the source code for this chapter is available from [https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_14](https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_13).

# Understanding generic RTOS APIs

An RTOS API defines the programming interface that the user interacts with when using the RTOS. Native APIs expose all of the RTOS's functionality. So far in this book, we've been using the native FreeRTOS API only. This was done to make it easier to search for help for a given function and to rule out any possibility of a poorly behaving wrapper layer between FreeRTOS and a generic API. However, this is not the only API option for FreeRTOS. There are also generic APIs available that can be used to interface with the RTOS functionality – but instead of being tied to a specific RTOS, they can be used across multiple operating systems. 

These generic APIs are usually implemented as a wrapper layer above the native RTOS API (the exception to this is RTX, which has only the CMSIS-RTOS API). Here we can see where a typical API would live in a generic **Advanced RISC Machines** (**ARM**) firmware stack:

![](img/b39ecd2b-e4eb-40a4-977a-3915ca5de469.png)

As we can see from the arrows in the preceding diagram, there is no single abstraction that blocks the user code from accessing the lowest level of functionality. Each layer adds another potential API to be used, while the lower-level functionality is still available.

There are two generic APIs that can be used to access a subset of FreeRTOS's functionality:

*   **CMSIS-RTOS**: ARM has defined a vendor-agnostic API targeting MCUs called the **Cortex Microcontroller Software Interface-RTOS** (CMSIS-RTOS).  
*   **POSIX**: The **Portable Operating System Interface** (**POSIX**) is another example of a generic interface that is common across multiple vendors and hardware. This API is more commonly used in full general-purpose operating systems, such as Linux.

We will be discussing these generic APIs in depth throughout this chapter. But first, let's take a look at their advantages and disadvantages.

# Advantages of generic APIs

Using a generic RTOS API such as CMSIS-RTOS or POSIX provides several benefits to programmers and middleware vendors. A programmer can write code once and run it on multiple MCUs, changing out the RTOS as desired with few or no changes to their application code. Middleware vendors are also able to write their code to interact with a single API and then support multiple RTOSes and hardware.

As you may have noticed from the previous diagram, CMSIS-RTOS and POSIX APIs do not require exclusive access to FreeRTOS. Since these APIs are implemented as layers on top of the native FreeRTOS API, code can make use of either the more generic API or the native RTOS API at the same time. So, it is perfectly acceptable for some portions of an application to use the CMSIS-RTOS interface while others use the native FreeRTOS API.  

For example, if a GUI provider ships their code and it interfaces to CMSIS-RTOS, there is nothing to prevent additional development with the native FreeRTOS API. The GUI vendor's code can be brought in using CMSIS-RTOS, while other code in the system uses the native FreeRTOS API, without the CMSIS-RTOS wrapper.

With these benefits, it seems that a generic API would be the obvious answer to everything. But that's not true. 

# Disadvantages of generic APIs

What a general-purpose API gains in uniformity, it loses in specificity. A general-purpose, one-size-fits-all implementation needs to be generic enough to be applicable for the majority of RTOSes. This leads to the unique portions being left out of the standardized interface, which can sometimes include some very interesting features.

Since the RTOS vendors themselves aren't always the ones providing support for CMSIS-RTOS, there is the potential that the version of CMSIS-RTOS being shipped is lagging behind the RTOS release cycle. This means that RTOS updates to CMSIS-RTOS *might* not be included as often as for the native API.

There is also the problem of obtaining support if problems are encountered – an RTOS vendor will generally be more willing (and capable) to help with code they actually provided. Often, it will be very difficult to get support for an abstraction that the RTOS vendor hasn't written – both because they are likely to be unfamiliar with it and the abstraction itself can contain bugs/functionality that isn't present in the base RTOS code.

Now that we have a general idea of what a general-purpose RTOS API is, let's take a closer look and compare the FreeRTOS and CMSIS-RTOS APIs.

# Comparing FreeRTOS and CMSIS-RTOS

There is a common misconception that there is an RTOS named CMSIS-RTOS. CMSIS-RTOS is actually just an API definition. Its *implementation* is largely a glue layer to the underlying RTOS, but where functional differences exist between the two, some *glue code* will be present to map functionality.

ARM developed CMSIS-RTOS with the same goal in mind as when CMSIS was developed: to add a consistent layer of abstraction that reduces vendor lock-in. The original CMSIS was meant to reduce Silicon vendor lock-in by providing uniform methods for middleware to access common Cortex-M functionality. It accomplished this goal – there are only a few variants of FreeRTOS *ports* for the thousands of Cortex-M-based MCUs it supports. Likewise, ARM is now attempting to reduce RTOS vendor lock-in by making the RTOS itself easier to change out – by providing a consistent API (CMSIS-RTOS) that is vendor-agnostic.

This chapter refers to CMSIS-RTOS, but this information is specific to the current version of CMSIS-RTOS, which is CMSIS-RTOS v2 (which has a different API from CMSIS-RTOS v1). CMSIS-RTOS v2 is also commonly referred to as CMSIS-RTOS2\. The exact version that this chapter references is CMSIS-RTOS 2.1.3.

There are some primary FreeRTOS features that are also exposed by CMSIS-RTOS. Here's a quick overview (more details are included in the *Cross-referencing CMSIS-RTOS and FreeRTOS functions* section):

*   **Tasks**: This is the functionality for creating and deleting tasks with both static and dynamically allocated stacks.
*   **Semaphores/mutexes**: Binary and counting semaphores as well as mutexes are present in CMSIS-RTOS.
*   **Queues**: The Queue APIs are very similar between FreeRTOS's native API and the CMSIS-RTOS API.
*   **Software timers**: The Software Timer APIs are very similar between FreeRTOS's native API and the CMSIS-RTOS API.
*   **Event groups**: This is used to synchronize multiple tasks.
*   **Kernel/scheduler control**: Both APIs have the ability to start/stop tasks and monitor the system.

The feature sets of FreeRTOS and CMSIS-RTOS do not overlap completely. There are some features of FreeRTOS that are not available through CMSIS-RTOS:

*   **Stream and message buffers**: The flexible and efficient queue alternative
*   **Queue sets**: Used for blocking on multiple queues or semaphores
*   **Co-routines**: An explicit time-shared solution for running multiple functions when RAM is too limited to run multiple tasks

Likewise, there are also some features of CMSIS-RTOS that are not available from an off-the-shelf version of FreeRTOS, primarily MemoryPools. For a current list of CMSIS-RTOS2 functions, see [https://arm-software.github.io/CMSIS-FreeRTOS/General/html/functionOverview.html#rtos_api2](https://arm-software.github.io/CMSIS-FreeRTOS/General/html/functionOverview.html#rtos_api2).

**A special Note on ST Cube CMSIS-RTOS**

It is important to note that when applications are developed using ST Cube, the CMSIS-RTOS version adaptation layer, `cmsis_os2.c`, is a fork from the original API written by ARM. Many of the changes relate to how the CMSIS-RTOS layer interacts with the system clock. For documentation for the original ARM-supplied CMSIS-FreeRTOS implementation, visit [https://arm-software.github.io/CMSIS-FreeRTOS](https://arm-software.github.io/CMSIS-FreeRTOS).

# Considerations during migration

There are a few noteworthy differences between programming with the CMSIS-RTOS API compared with doing so using the FreeRTOS API.

CMSIS-RTOS task creation functions take the stack size in *bytes*, as opposed to in *words* in FreeRTOS. So, making calls to `xTaskCreate` in FreeRTOS with a stack size of 128 words equates to calling CMSIS-RTOS `osThreadNew` with an argument of 512 bytes.

CMSIS-RTOS has fewer functions than FreeRTOS but relies on attribute structs as input to those functions. For example, in FreeRTOS, there are many families of functions that have `FromISR` equivalents. The `FromISR` variants typically won't block at all – they *need* to be called if an RTOS API call is made from inside an ISR, but they can also be used selectively in other places. In the CMSIS-RTOS layer, the ISR context is automatically detected. The `FromISR` API is *automatically* used, depending on whether the caller is being executed within the ISR context or the application context.  `portYIELD_FROM_ISR` is also called automatically. The trade-off for simplicity here is that any blocking delays specified inside an ISR call will be ignored, since the `FromISR` variants are all non-blocking (since it is never a good idea to block for multiple milliseconds inside an ISR). This is in contrast to the FreeRTOS method of protecting against misuse of the RTOS API from within an ISR context – a `configASSERT` instance will fail, resulting in an infinite loop that halts the entire application.

With respect to protecting against misuse of RTOS API functionality from an ISR context, CMSIS-RTOS will return error codes when its functions are misused from inside an ISR context. In FreeRTOS, the same misuse will generally result in a failed `configASSERT` instance with a detailed comment, which halts the entire program. As long as the programmer is being responsible and rigorously checking return values, these errors will be detected. FreeRTOS is a bit more vocal about the errors, by not allowing program execution to continue (verbose comments explaining the reason for the misconfiguration and suggested solutions are almost always present in the FreeRTOS source code when this happens).

# Cross-referencing CMIS-RTOS and FreeRTOS functions

Here is a complete comparison of CMSIS-RTOS functions and their associated FreeRTOS functions. Feel free to skim the tables now if you're interested in finding out how various FreeRTOS functions are called from the CMSIS-RTOS API. Otherwise, use the tables as a reference when porting code between the CMSIS-RTOS and FreeRTOS APIs.

# Delay functions

Delay functions map cleanly between the two APIs:

| **CMSIS-RTOS name** | **FreeRTOS functions called** | **Notes** |
| `osDelay ` | `vTaskDelay` | `osDelay` is in ms or ticks, depending on which documentation and comments you believe. Be sure to check your CMSIS-RTOS implementation of `osDelay()` if a `Systick` frequency of something other than 1 kHz is used! |
| `osDelayUntil` | `vTaskDelayUntil`, `xTaskGetTickCount` |  |

These basic delay functions work in very similar ways – the biggest difference to keep in mind is that CMSIS-RTOS specifies `osDelay` in milliseconds instead of *ticks*, as FreeRTOS does.

# EventFlags

`oseventFlags` in CMSIS-RTOS maps to `EventGroups` in FreeRTOS. The `FromISR` variant of the FreeRTOS API is automatically used when CMSIS-RTOS functions are called from inside an ISR:

| **CMSIS-RTOS name** | **FreeRTOS functions called** | **Notes** |
| `oseventFlagsClear` | `xEventGroupsClearBits`, `xEventGroupGetBitsFromISR` |  |
| `osEventFlagsDelete` | `vEventGroupDelete` |  |
| `osEventFlagsGet` | `xEventGroupGetBits, xEventGroupGetBitsFromISR` |  |
| `osEventFlagsNew` | `xEventGroupCreateStatic`, `xEventGroupCreate` |  |
| `osEventFlagsSet` | `xEventGroupSetBits, xEventGroupSetBitsFromISR` |  |
| `osEventFlagsWait` | `xEventGroupWaitBits` |  |

`EventFlags` in CMSIS-RTOS work similarly to `EventGroups` in FreeRTOS, with nearly 1:1 mapping.

# Kernel control and information

The kernel interfaces are similar, although some timer implementations that STM has provided aren't all that intuitive, specifically `osKernelGetSysTimerCount` and `osKernelGetSysTimerCount`. Also, some functions will return errors if there are issues within the context of an ISR:

*   `osKernelInitialize`
*   `osKernelRestoreLock`
*   `osKernelStart3`
*   `osKernelUnlock`

Pay special attention to the notes in this table:

| **CMSIS-RTOS name** | **FreeRTOS functions called** | **Notes** |
| `osKernelGetInfo` | `static strings representing FreeRTOS version` |  |
| `osKernelGetState` | `xTaskGetSchedulerState` |  |
| `osKernelGetSysTimerCount` | `xTaskGetTickCount` | This returns `xTaskGetTickCount() *` (`SysClockFreq` / `configTICK_RATE_HZ`). |
| `osKernelGetSysTimerFreq` | ST HAL SystemCoreClock global variable |  |
| `osKernelGetTickCount` | `xTaskGetTickCount` |  |
| `osKernelGetTickFreq` | `configTICK_RATE_HZ` | This is *not* the `SysTick` frequency (that is, `1` `kHz)(SysClockFreq` is being returned (160 MHz)). |
| `osKernelInitialize` | `vPortDefineHeapRegions` (only if `Heap5` is used) |  |
| `osKernelLock` | `xTaskGetSchedulerState`, `vTaskSuspendAll` |  |
| `osKernelRestoreLock` | `xTaskGetSchedulerState`, `vTaskSuspendAll` |  |
| `osKernelStart` | `vTaskStartScheduler` |  |
| `osKernelUnlock` | `xTaskGetSchedulerState`, `xTaskResumeAll` |  |

Be aware of the slight differences in time units when moving between kernel-oriented functions using the STM-supplied CMSIS-RTOS port and the native FreeRTOS API.

# Message queues

Message queues are quite similar. In CMSIS-RTOS, all queues are registered by name, which can make for a richer debugging experience. Also, CMSIS-RTOS supports static allocation via attributes passed in as function parameters.

Any functions called from inside an ISR will automatically be forced to use the `FromISR` equivalent functions and finish the ISR with a call to `portYIELD_FROM_ISR`. This results in any blocking times being effectively set to `0`. So, for example, if a queue doesn't have space available, a call to `osMessageQueuePut` will return immediately from inside an ISR, even if a blocking timeout is specified:

| **CMSIS-RTOS name** | **FreeRTOS functions called** | **Notes** |
| `osMessageQueueDelete` | `vQueueUnregisterQueue`, `vQueueDelete` |  |
| `osMessageQueueGet` | `xQueueReceive` | The `FromISR` variant is automatically called and `portYIELD_FROM_ISR` is automatically called if inside an ISR. |
| `osMessageQueueGetCapacity` | `pxQueue->uxLength` |  |
| `osMessageQueueGetCount` | `uxQueueMessagesWaiting`, `uxQueueMessagesWaitingFromISR` |  |
| `osMessageQueueGetMsgSize` | `pxQueue->uxItemSize` |  |
| `osMessageQueueGetSpace` | `uxQueueSpacesAvailable` | `taskENTER_CRITICAL_FROM_ISR` is automatically called if this function is executed from within an ISR. |
| `osMessageQueueNew` | `xQueueCreateStatic`, `xQueueCreate` |  |
| `osMessageQueuePut` | `xQueueSendToBack`, `xQueueSendToBackFromISR` | The `msg_prior` parameter is ignored in the STM port. |
| `osMessageQueueReset` | `xQueueReset` |  |

Queues are very similar between CMSIS-RTOS and FreeRTOS, but it is worth noting that CMSIS-RTOS doesn't have an equivalent of `xQueueSendToFront`, so it will not be possible to place items at the front of a queue using CMSIS-RTOS.

# Mutexes and semaphores

Mutexes are also similar between the two APIs, with some considerations to keep in mind:

*   In CMSIS-RTOS, the recursive mutex API functions are automatically called, depending on the type of mutex created.
*   In CMSIS-RTOS, static allocation is supported via attributes passed in as function parameters.
*   `osMutexAcquire`, `osMutexRelease`, `osMutexDelete`, and `osMutexRelease` will always fail by returning `osErrorISR` if called within an ISR context.
*   `osMutexGetOwner` and `osMutexNew` will always return `NULL` when called from within an ISR.

With those points in mind, here are the relationships between mutexes in CMSIS-RTOS and FreeRTOS APIs:

| **CMSIS-RTOS name** | **FreeRTOS functions called** | **Notes** |
| `osMutexAcquire` | `xSemaphoreTake`, `xSemaphoreTakeRecursive` | The `takeRecursive` variant is automatically called when the mutex is recursive. |
| `osMutexRelease` | `xSemaphoreGive`, `xSemaphoreGiveRecursive` | The `takeRecursive` variant is automatically called when the mutex is recursive. |
| `osMutexDelete` | `vSemaphoreDelete`, `vQueueUnregisterQueue` |  |
| `osMutexGetOwner` | `xSemaphoreGetMutexHolder` | This always returns `NULL` if called from inside an ISR, which is identical to the expected behavior when the mutex is available. |
| `osMutexNew` | `xSemaphoreCreateRecursiveMutexStatic`, `xSemaphoreCreateMutexStatic`,`xSemaphoreCreateRecursiveMutex`, `xSemaphoreCreateMutex`, `vQueueAddToRegistry` | Different mutex types are created depending on the value of the `osMutexAttr_t` pointer passed into the function. |
| `osMutexRelease` | `xSemaphoreGiveRecursive`, `xSemaphoreGive` |  |

While the mutex functionality is very similar between the APIs, the way in which it is achieved is quite different. FreeRTOS uses many different functions to create mutexes, while CMSIS-RTOS achieves the same functionality by adding parameters to fewer functions. It also records the mutex type and automatically calls the appropriate FreeRTOS function for recursive mutexes.

# Semaphores 

The `FromISR` equivalents of semaphore functions are automatically used when necessary. Static and dynamically allocated semaphores, along with binary and counting semaphores, are all created using `osSemaphoreNew`. 

The fact that semaphores are implemented using queues under the hood in FreeRTOS is evident here, as evidenced by the use of the Queue API to extract information for the semaphores:

| **CMSIS-RTOS name** | **FreeRTOS functions called** | **Notes** |
| `osSemaphoreAcquire` | `xSemaphoreTakeFromISR`, `xSemaphoreTake`, `portYIELD_FROM_ISR` | The automatic ISR context is accounted for. |
| `osSemaphoreDelete` | `vSemaphoreDelete`, `vQueueUnregisterQueue` |  |
| `osSemaphoreGetCount` | `osSemaphoreGetCount`, `uxQueueMessagesWaitingFromISR` |  |
| `osSemaphoreNew` | `xSemaphoreCreateBinaryStatic`, `xSemaphoreCreateBinary`, `xSemaphoreCreateCountingStatic`, `xSemaphoreCreateCounting`, `xSemaphoreGive`, `vQueueAddToRegistry` | All semaphore types are created using this function. Semaphores are automatically given unless the initial count is specified as `0`. |
| `osSemaphoreRelease` | `xSemaphoreGive`, `xSemaphoreGiveFromISR` |  |

In general, semaphore functionality maps very cleanly between CMSIS-RTOS and FreeRTOS, although the function names differ.

# Thread flags

CMSIS-RTOS thread flag usage should be reviewed independently (a link to the detailed documentation is provided). As you can see from the FreeRTOS fuctions called, they are built on top of `TaskNotifications`. Again, ISR-safe equivalents are automatically substituted when the calls are made within an ISR context:

| **CMSIS-RTOS name** | **FreeRTOS functions called** | **Notes** |
| `osThreadFlagsClear` | `xTaskGetCurrentTaskHandle`, `xTaskNotifyAndQuery`, `xTaskNotify` | [https://www.keil.com/pack/doc/CMSIS/RTOS2/html/group__CMSIS__RTOS__ThreadFlagsMgmt.html](https://www.keil.com/pack/doc/CMSIS/RTOS2/html/group__CMSIS__RTOS__ThreadFlagsMgmt.html) |
| `osThreadFlagsGet` | `xTaskGetCurrentTaskHandle`, `xTaskNotifyAndQuery` |  |
| `osThreadFlagsSet` | `xTaskNotifyFromISR`, `xTaskNotifyAndQueryFromISR`, `portYIELD_FROM_ISR`, `xTaskNotify`, `xTaskNotifyAndQuery` |  |
| `osThreadFlagsWait` | `xTaskNotifyWait` |  |

`ThreadFlags` and `TaskNotifications` have the largest potential for different behavior between the two APIs. Most of this will depend on how they are used in a specific application, so it is best to review the `ThreadFlags` documentation in detail before attempting to port `TaskNofications` to `ThreadFlags`.

# Thread control/information

The basic threading API is very similar between CMSIS-RTOS and FreeRTOS, with the exception of CMSIS-RTOS's `osThreadGetStackSize`, which has no equivalent in FreeRTOS. Other minor differences include the addition of `osThreadEnumerate`, which uses several FreeRTOS functions while it lists the tasks in the system, as well as different names for states (CMSIS-RTOS lacks a `suspend` state). In CMSIS-RTOS, both static and dynamic thread/task stack allocation is supported through the same function, `osThreadNew`.

If `osThreadTerminate` is called while using the FreeRTOS Heap1 implementation (discussed in the next chapter), an infinite loop with no delay will be entered.  

Be aware that CMSIS-RTOS v2  `osThreadAttr_t.osThreadPriority` requires 56 different task priorities! Therefore, `configMAX_PRIORITIES` in `FreeRTOSConfig.h` must have a value of 56, or the implementation of `osThreadNew()` will need to be scaled to fit into the available number of priorities:

| **CMSIS-RTOS name** | **FreeRTOS functions called** | **Notes** |
| `osThreadEnumerate` | `vTaskSuspendAll,``uxTaskGetNumberOfTasks,``uxTaskGetSystemState,``xTaskResumeAll` | This suspends the system and populates an array of task handles. |
| `osThreadExit` | `vTaskDelete` | This ends the current thread if `HEAP1` is being used. This function will cause the caller to go into a tight infinite loop, consuming as many CPU cycles as available given the caller's priority. |
| `osThreadGetCount` | `uxTaskGetNumberOfTasks` |  |
| `osThreadGetId` | `xTaskGetCurrentTaskHandle` |  |
| `osThreadGetName` | `pcTaskGetName` |  |
| `osThreadGetPriority` | `uxTaskPriorityGet` |  |
| `osThreadGetStackSize` | always returns `0` | `https://github.com/ARM-software/CMSISFreeRTOS/issues/14` |
| `osThreadGetStackSpace` | `uxTaskGetStackHighWaterMark` |  |
| `osThreadGetState` | `eTaskGetState` |  

&#124; **FreeRTOS Task State** &#124; **CMSIS-RTOS** &#124;
&#124; `eRunning` &#124; `osThreadRunning` &#124;
&#124; `eReady` &#124; `osThreadReady` &#124;
&#124; `eBlocked` &#124; `osThreadBlocked` &#124;
&#124; `eSuspended` &#124;  &#124;
&#124; `eDeleted` &#124; `osThreadTerminated` &#124;
&#124; `eInvalid` &#124; `osThreadError` &#124;

 |
| `osThreadNew` | `xTaskCreateStatic`,`xTaskCreate` |  |
| `osThreadResume` | `vTaskResume` |  |
| `osThreadSetPriority` | `vTaskPrioritySet` |  |
| `osThreadSuspend` | `vTaskSuspend` |  |
| `osThreadTerminate` | `vTaskDelete` | If `Heap1` is used, this function returns `osError`. |
| `osThreadYield` | `taskYIELD` |  |

Most of the thread controls are a simple 1:1 mapping, so they are straightforward to substitute between the two APIs.

# Timers

Timers are equivalent, with static and dynamic allocation both being defined by the same `osTimerNew` function: 

| **CMSIS-RTOS name** | **FreeRTOS functions called** | **Notes** |
| `osTimerDelete` | `xTimerDelete` | If `Heap1` is used, this function returns `osError`. It also frees up `TimerCallback_t*` used by the timer to be deleted. |
| `osTimerGetName` | `pcTimerGetName` |  |
| `osTimerIsRunning` | `xTimerIsTimerActive` |  |
| `osTimerNew` | `xTimerCreateStatic`, `xTimerCreate` | Automatic allocation for `TimerCallback_t`. |
| `osTimerStart` | `xTimerChangePeriod` |  |
| `osTimerStop` | `xTimerStop` |  |

Timers are very similar between the two APIs, but beware of attempting to use `osTimerDelete` with `Heap1`.

# Memory pools

Memory pools are a popular dynamic allocation technique commonly found in embedded RTOSes. FreeRTOS does not currently supply a memory pool implementation out of the box. A design decision was made in early development to eliminate it because it added extra user-facing complexity and wasted too much RAM.

ARM and ST have elected to not supply any memory pool implementations on top of FreeRTOS.

That concludes our complete cross-reference of the CMSIS-RTOS and FreeRTOS APIs. It should have been helpful in quickly determining what differences you need to be aware of. While CMSIS-RTOS can be used with RTOSes from different vendors, it does not contain all of the features that FreeRTOS has to offer (such as stream buffers). 

Now that we've seen a comparison between the native FreeRTOS API and the CMSIS-RTOS v2 API, let's take a look at an example of an application using CMSSI-RTOS v2.

# Creating a simple CMSIS-RTOS v2 application

Armed with an understanding of the differences between the native FreeRTOS API and the CMSIS-RTOS v2 API, we can develop a bare-bones application with two tasks that blink some LEDs. The goal of this application is to develop code that is only dependent on the CMCSIS-RTOS API rather than the FreeRTOS API. All the code found here resides in `main_taskCreation_CMSIS_RTOSV2.c`*.*

This example is similar to those found in [Chapter 7](2fa909fe-91a6-48c1-8802-8aa767100b8f.xhtml), *The FreeRTOS Scheduler*; this one only sets up tasks and blinks LEDs. Follow these steps:

1.  Initialize the RTOS using `osStatus_t osKernelInitialize (void)`, checking the return value before continuing:

```cpp
osStatus_t status;
status = osKernelInitialize();
assert(status == osOK);
```

2.  Since CMSIS-RTOS uses structs to pass in thread attributes, populate an `osThreadAttr_t` structure from `cmsis_os2.h`:

```cpp
/// Attributes structure for thread.
typedef struct {
  const char *name;    ///< name of the thread
  uint32_t attr_bits;  ///< attribute bits
  void *cb_mem;        ///< memory for control block
  ///< size of provided memory for control block
  uint32_t cb_size; 
  void *stack_mem;     ///< memory for stack
  uint32_t stack_size; ///< size of stack
  ///< initial thread priority (default: osPriorityNormal)
  osPriority_t priority;
  TZ_ModuleId_t tz_module; ///< TrustZone module identifier
  uint32_t reserved;       ///< reserved (must be 0)
} osThreadAttr_t;
```

**Note**: Unlike FreeRTOS stack sizes, which are defined in the number of *words* the stack will consume (4 bytes for Cortex-M7), CMSIS-RTOS sizes are always defined in *bytes.* Previously, when using the FreeRTOS API, we were using 128 words for the stack size. Here, to achieve the same stack size, we'll use 128 * 4 = 512 bytes.

```cpp
  #define STACK_SIZE 512
  osThreadAttr_t greenThreadAtrribs = {   .name = "GreenTask",
                                          .attr_bits = osThreadDetached,
                                          .cb_mem = NULL,
                                          .cb_size = 0,
                                          .stack_mem = NULL,
                                          .stack_size = STACK_SIZE,
                                          .priority = osPriorityNormal,
                                          .tz_module = 0,
                                          .reserved = 0};
```

In the preceding code, we can see the following:

*   *   Only `osThreadDetachted` is supported for `attr_bits`.
    *   The first task to be created will use dynamic allocation, so the control block and stack-related variables (`cb_mem, cb_size, stack_mem, stack_size`) will be set to `0` and `NULL`.
    *   Normal priority will be used here.
    *   Cortex-M7 MCUs (STM32F759) do not have a trust zone.

3.  Create the thread by calling `osThreadNew()` and passing in a pointer to the function that implements the desired thread, any task arguments, and a pointer to the `osThreadAttr_t` structure. The prototype for `osThreadNew(` is as follows:

```cpp
osThreadId_t osThreadNew (    osThreadFunc_t func, 
                              void *argument, 
                              const osThreadAttr_t *attr);
```

Here is the actual call to `osThreadNew()`, which creates the `GreenTask` thread. Again, be sure to check that the thread has been successfully created before moving on:

```cpp
greenTaskThreadID = osThreadNew( GreenTask, NULL,
                            &greenThreadAtrribs);
assert(greenTaskThreadID != NULL);
```

4.  The `GreenTask` function will blink the green LED (on for 200 ms and off for 200 ms):

```cpp
void GreenTask(void *argument)
{
  while(1)
  {
    GreenLed.On();
    osDelay(200);
    GreenLed.Off();
    osDelay(200);
  }
}
```

It is worth noting that, unlike the case in FreeRTOS's `vTaskDelay()` where the delay is dependent on the underlying tick frequency, CMSIS-RTOS's `osDelay()` is suggested by Keil/ARM documentation to be specified in milliseconds.  However, the documentation also refers to the argument as *ticks.* Since a tick isn't necessarily 1 ms long, be sure to check your implementation of `osDelay()` in `cmsis_os2.c`. For example, in the copy of `cmsis_os2.c`obtained from STM, no conversion is performed between ticks and ms.

5.  Start the scheduler:

```cpp
status = osKernelStart();
assert(status == osOK);
```

This call should not return when successful.

`main_taskCreation_CMSIS_RTOSV2.c` also contains an example of starting a task with statically allocated memory for the task control block and task stack.

Static allocation requires computing the sizes of RTOS control blocks (such as `StaticTask_t`) that are specific to the underlying RTOS. To reduce the coupling of code to the underlying RTOS, an additional header file should be used to encapsulate all RTOS-specific sizes. In this example, this file is named `RTOS_Dependencies.h`.

Tasks created from statically allocated memory use the same `osThreadCreate()` function call as before. This time, the `cb_mem, cb_size, stack_mem, stack_size` variables will be populated with pointers and sizes.

6.  Define an array, which will be used as the task stack:

```cpp
#define STACK_SIZE 512
static uint8_t RedTask_Stack[STACK_SIZE];
```

7.  Populate `RTOS_Dependencies.h` with the size of the FreeRTOS task control block used for static tasks:

```cpp
#define TCB_SIZE (sizeof(StaticTask_t))
```

8.  Define an array that's large enough to hold the task control block:

```cpp
uint8_t RedTask_TCB[TCB_SIZE];
```

9.  Create an `osThreadAttr_t` struct containing all of the name, pointer, and task priorities:

```cpp
osThreadAttr_t redThreadAtrribs = { .name = "RedTask",
        .attr_bits = osThreadDetached,
        .cb_mem = RedTask_TCB,
        .cb_size = TCB_SIZE,
        .stack_mem = RedTask_Stack,
        .stack_size = STACK_SIZE,
        .priority = osPriorityNormal,
        .tz_module = 0,
        .reserved = 0};
```

10.  Create the `RedTask` thread, making sure that it has been successfully created before moving on:

```cpp
redTaskThreadID = osThreadNew( RedTask, NULL, &redThreadAtrribs);
assert(redTaskThreadID != NULL);
```

`main_taskCreate_CMSIS_RTOSV2.c` can be compiled and flashed onto the Nucleo board and used as a starting point to experiment with the remainder of the CMSIS-RTOSv2 API. You can use this basic program to jump-start additional CMSIS-RTOSv2 API experimentation.

Now that we have an understanding of a commonly used MCU-centric API for FreeRTOS, let's move on to a standard that has been around since the 1980s and is still going strong.

# FreeRTOS and POSIX

The **Portable Operating System Interface **(**POSIX**) was developed to provide a unified interface for interacting with operating systems, making code more portable between systems.

At the time of writing, FreeRTOS has a beta implementation for a subset of the POSIX API. The POSIX headers that have been (partly) ported are listed here:

*   `errno.h`
*   `fcntl.h`
*   `mqueue.h`
*   `mqueue.h`
*   `sched.h`
*   `semaphore.h`
*   `signal.h`
*   `sys/types.h`
*   `time.h`
*   `unistd.h`

Generally speaking, threading, queues, mutexes, semaphores, timers, sleep, and some clock functions are implemented by the port. This feature set sometimes covers enough of a real-world use case to enable porting applications that have been written to be POSIX-compliant to an MCU supporting FreeRTOS. Keep in mind that FreeRTOS does not supply a filesystem on its own without additional middleware, so any application requiring filesystem access will need some additional components before it will be functional.

Let's take a look at what a minimal application using the POSIX API looks like.

# Creating a simple FreeRTOS POSIX application

Similarly to the CMSIS API example, the POSIX API example will just blink two LEDs at different intervals.

Note that after FreeRTOS POSIX moves out of FreeRTOS Labs, the download location (and the corresponding instructions) will likely change.

First, the POSIX wrapper needs to be downloaded and brought into the source tree. Perform the following steps:

1.  Download the FreeRTOS Labs distribution ( [https://www.freertos.org/a00104.html](https://www.freertos.org/a00104.html) ). Go to [https://www.freertos.org/FreeRTOS-Plus/FreeRTOS_Plus_POSIX/index.html](https://www.freertos.org/FreeRTOS-Plus/FreeRTOS_Plus_POSIX/index.html) for up-to-date download instructions.
2.  Import the selected `FreeRTOS_POSIX` files into your source tree. In the example, they reside in `Middleware\Third_Party\FreeRTOS\FreeRTOS_POSIX`.
3.  Add the necessary `include` paths to the compiler and linker by modifying the project properties within STM32CubeIDE:

![](img/8e80008c-b3c6-4cb5-8e1a-3f47a5134084.png)

4.  Make sure to add the following `#define` lines to `Inc/FreeRTOSConfig.h`:

```cpp
#define configUSE_POSIX_ERRNO 1
#define configUSE_APPLICATION_TASK_TAG 1
```

Now that POSIX APIs are available, we'll use `pthreads` and `sleep` in `main_task_Creation_POSIX.c`:

1.  Bring in the necessary header files:

```cpp
// FreeRTOS POSIX includes
#include <FreeRTOS_POSIX.h>
#include <FreeRTOS_POSIX/pthread.h>
#include <FreeRTOS_POSIX/unistd.h>
```

2.  Define the necessary function prototypes:

```cpp
void GreenTask(void *argument);
void RedTask(void *argument);
void lookBusy( void );
```

3.  Define global variables to store the thread IDs:

```cpp
pthread_t greenThreadId, redThreadId;
```

4.  Use `pthread_create()` to create a thread/task:

```cpp
int pthread_create( pthread_t *thread, const pthread_attr_t *attr,
                    void *(*start_routine) (void *), void *arg);
```

Here's some information about the preceding code:

*   *   `thread`: A pointer to a `pthread_t` struct, which will be filled out by `pthread_create()`
    *   `attr`: A pointer to a struct containing the attributes of the thread
    *   `start_routine`: A pointer to the function implementing the thread
    *   `arg`: The arguments to pass to the thread's function
    *   Returns `0` in the case of success and `errrno` in the case of failure (the contents of `pthread_t *thread` will be undefined in the case of failure)

Here, two threads are started using the functions declared earlier – `GreenTask()` and `RedTask()`:

```cpp
retVal = pthread_create( &greenThreadId, NULL, GreenTask, NULL);
assert(retVal == 0);

retVal = pthread_create( &redThreadId, NULL, RedTask, NULL);
assert(retVal == 0);
```

5.  Start the scheduler:

```cpp
vTaskStartScheduler();
```

When the scheduler is started, both `GreenTask()` and `ReadTask()` will be switched into context as required. Let's have a quick look at each of these functions.

`GreenTask()` is using `sleep()`, brought in from `unistd.h`. Now, `sleep()` will force the task to block for the desired number of seconds (in this case, 1 second after turning the LED on and 1 second after turning the LED off):

```cpp
void GreenTask(void *argument)
{
  while(1)
  {
    GreenLed.On();
    sleep(1);
    GreenLed.Off();
    sleep(1);
  }
}
```

`RedTask()` is similar, sleeping for 2 seconds after the red LED is turned off:

```cpp
void RedTask( void* argument )
{
  while(1)
  {
    lookBusy();
    RedLed.On();
    sleep(1);
    RedLed.Off();
    sleep(2);
  }
}
```

Now `TaskCreation_POSIX` can be compiled and loaded onto the Nucleo board.  You're free to use this as a starting point for experimenting with more portions of the POSIX API. Next, let's look at some of the reasons why you might want to use the POSIX API.

# Pros and cons to using the POSIX API

There are two primary reasons to consider using the POSIX API for FreeRTOS:

*   **Portability to general-purpose operating systems: **By definition, the goal of POSIX is *portability. *There are many general-purpose operating systems meant to be run on CPUs with MMUs that are POSIX-compliant. Increasingly, there are also several lightweight operating systems aimed at MCUs that are also POSIX-compliant. If your goal is to run your code base on these types of systems, the POSIX API is the interface to use. It is the only API for FreeRTOS that will allow code to be portable to a fully fledged operating system (rather than a real-time kernel).
*   **Third-party POSIX libraries**: Many open source libraries are written to interface via POSIX. Having the ability to bring in *some* POSIX-compatible third-party code (as long as it only accesses the portions that have been ported by FreeRTOS) has the potential to quickly boost a project's functionality.

Of course, there are some drawbacks to using the POSIX API as well:

*   **Still in beta**: At the time of writing (early 2020), the POSIX API is still in FreeRTOS Labs. Here's an explanation from `freertos.org`:

<q>The POSIX library and documentation are in the FreeRTOS Labs.  The libraries in the FreeRTOS Labs download directory are fully functional, but undergoing optimizations or refactoring to improve memory usage, modularity, documentation, demo usability, or test coverage.  They are available as part of the FreeRTOS-Labs download: [https://www.freertos.org/a00104.html](https://www.freertos.org/a00104.html).</q>

*   **Being limited to the POSIX API may reduce efficiency**: Having code that is portable between many different operating systems running on both MCUs and CPUs will come with a cost. Any code that you'd like to make portable to any platform that supports POSIX will need to contain only POSIX functionality (that is implemented by FreeRTOS). Since only a small subset of the FreeRTOS API is exposed through POSIX, you'll be giving up some of the more efficient implementations. Some of the most time- and CPU-efficient functionality (such as stream buffers and direct task notifications) won't be available if you're aiming to have ultra-portable code that uses only the POSIX API.

Having the POSIX API available to ease the addition of third-party code is an exciting development for embedded developers. It has the potential to bring a large amount of functionality into the embedded space very quickly. But keep this in mind: although today's MCUs are extremely powerful, they're not general-purpose processors. You'll need to be mindful of all of the code's interaction and resource requirements, especially with systems that have real-time requirements.

So, we have three primary options regarding which API to utilize when interacting with FreeRTOS. What kinds of considerations should be made when choosing between them?

# Deciding which API to use

Deciding which API to use is largely based on *where* you'd like your code to be portable to and *what* experience various team members have. For example, if you're interested in being able to try out different Cortex-M RTOS vendors, CMSIS-RTOS is a natural choice. It will allow different operating systems to be brought in without changing the application-level code. 

Similarly, if your application code needs to be capable of running both in a Linux environment on a fully featured CPU as well as on an MCU, the FreeRTOS POSIX implementation would make a lot of sense.

Since both of these APIs are layered *on top of* the native FreeRTOS API, you'll still be able to use any FreeRTOS-specific functionality that is required. The following sections should provide some points for consideration and help you decide when each API should be chosen. As usual, there is often no right or wrong choice – just a set of trade-offs to be made.

# When to use the native FreeRTOS API

There are some cases when using only the native FreeRTOS API is advantageous:

*   **Code consistency**:If an existing code base is already using the native FreeRTOS API, there is little benefit to writing new code that adds an additional layer of complexity (and a different API) on top of it. Although the functionality is similar, the actual function signatures and data structures are different. Because of these differences, having inconsistency between which API is used by old and new code might be very confusing for programmers unfamiliar with the code base.

*   **Support**:If the API you'd like to use is not written by the same writer as the RTOS, there is a very good chance that the RTOS vendor won't be able/willing to provide support for problems that arise (since the issue could be relevant only to the generic API wrapper layer and not the underlying RTOS). When you're first starting out with an RTOS, you'll likely find it is easier to get support (both by the vendor and forums) if you're referencing their code rather than a third-party wrapper.
*   **Simplicity**: When asking an RTOS vendor which API to use, the response will generally be "*the native API we wrote*." On the surface, this may seem a bit self-serving. After all, if you're using their native API, porting your code to another vendor's operating system won't be as easy. However, there's a bit more to this recommendation than first meets the eye. Each RTOS vendor generally has a strong preference for the style they've chosen when writing their code (and API). Gluing this native API to a different one may be a bit of a paradigm shift. Sometimes this extra layer of glue is so thin as to barely be noticed. Other times, it can turn into a sticky mess, requiring considerable extra code to be written on top of a native API and making it more confusing for developers well versed with the native API.
*   **Code space**:Since each of the generic APIs is a wrapper around the native FreeRTOS API, they will require a small amount of additional code space. On larger 32-bit MCUs, this will rarely be a consideration.

# When to use the CMSIS-RTOS API

Use CMSIS-RTOS when you'd like your code to be portable to other ARM-based MCUs. Some of the other RTOSes that are aimed at MCUs and support the CMSIS-RTOS API include the following:

*   Micrium uCOS
*   Express Logic ThreadX
*   Keil RTX
*   Zephyr Project

By using only functions provided by CMSIS-RTOS API, your code will run on top of any compatible operating system without modification.  

# When to use the POSIX API

Use the POSIX port when you'd like your code to be portable to these operating systems, or if there is a library that relies on the POSIX API that you'd like to include in your MCU project: 

*   Linux
*   Android
*   Zephyr
*   Nuttx (POSIX)
*   Blackberry QNX

While each of the POSIX-compatible operating systems just listed implements portions of POSIX, not all of the feature sets will necessarily intersect. When writing code that is intended to be run across multiple targets, a *least common denominator* approach will need to be taken – be sure to only use the smallest number of features commonly available across all target platforms.

It is also worth noting that since POSIX-compliant open source applications are designed for fully fledged PCs, they may utilize libraries that are not suitable for an MCU (for example, a filesystem that is not present using the core FreeRTOS kernel).

# Summary

In this chapter, we've covered three different APIs that can be used with FreeRTOS – the native FreeRTOS API, CMSIS-RTOS, and POSIX. You should now be familiar with all of the different APIs available for interacting with FreeRTOS and have an understanding of why they exist, as well as an understanding of when it is appropriate to use each one. Moving forward, you will be well positioned to make informed decisions about which API to use, depending on your particular project's requirements.

In the next chapter, we'll switch gears from discussing how to interact with FreeRTOS at a high level and discuss some of the low-level details of memory allocation. 

# Questions

As we conclude, here is a list of questions for you to test your knowledge regarding this chapter's material. You will find the answers in the *Assessments* section of the *Appendix*:

1.  What is CMSIS-RTOS, and which vendor supplies its implementation?
2.  Name a common operating system that makes heavy use of POSIX.
3.  It is important to choose wisely between the CMSIS-RTOS and FreeRTOS APIs because only one is available at a time:
    *   True
    *   False
4.  By using the POSIX API, any program written for Linux can be easily ported to run on FreeRTOS:
    *   True
    *   False

# Further reading

*   The CMSIS-RTOS v2 API documentation: [https://www.keil.com/pack/doc/CMSIS/RTOS2/html/](https://www.keil.com/pack/doc/CMSIS/RTOS2/html/group__CMSIS__RTOS.html)
*   FreeRTOS POSIX API - [https://www.freertos.org/FreeRTOS-Plus/FreeRTOS_Plus_POSIX/index.html](https://www.freertos.org/FreeRTOS-Plus/FreeRTOS_Plus_POSIX/index.html)
*   A detailed list of FreeRTOS POSIX ported functions
*   Zephyr POSIX implementation for the STM32 F7 Nucleo-144 dev board: [https://docs.zephyrproject.org/latest/boards/arm/nucleo_f767zi/doc/index.html](https://docs.zephyrproject.org/latest/boards/arm/nucleo_f767zi/doc/index.html)