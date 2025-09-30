# The FreeRTOS Scheduler

The FreeRTOS scheduler takes care of all task switching decisions. The most basic things you can do with an RTOS include creating a few tasks and then starting the scheduler – which is exactly what we'll be doing in this chapter. Creating tasks and getting the scheduler up and running will become something you'll be well accustomed to after some practice. Even though this is straightforward, it doesn't always go smoothly (especially on your first couple of tries), so we'll also be covering some common problems and how to fix them. By the end, you'll be able to set up your own RTOS application from scratch and know how to troubleshoot common problems.

We'll start by covering two different ways of creating FreeRTOS tasks and the advantages each offer. From there, we'll cover how to start the scheduler and what to look for to make sure it is running. Next, we'll briefly touch on memory management options. After that, we'll take a closer look at task states and cover some tips on optimizing your application so that it uses task states effectively. Finally, some troubleshooting tips will be offered.

Here's what we'll cover in this chapter:

*   Creating tasks and starting the scheduler
*   Deleting tasks
*   Trying out the code
*   Task memory allocation
*   Understanding FreeRTOS task states
*   Troubleshooting startup problems

# Technical requirements

To carry out the exercises in this chapter, you will require the following:

*   Nucleo F767 development board
*   Micro USB cable

*   STM32CubeIDE and its source code
*   SEGGER JLink, Ozone, and SystemView installed

For the installation instructions for STM32CubeIDE and its source code, please refer to [Chapter 5](84a945dc-ff6c-4ec8-8b9c-84842db68a85.xhtml), *Selecting an IDE*. For SEGGER JLink, Ozone, and SystemView, please refer to [Chapter 6](699daa80-06ae-4acc-8b93-a81af2eb774b.xhtml),* Debugging Tools for Real-Time Systems.*

You can find the code files for this chapter here: [https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_7](https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_7). For individual files, whose code snippets can be found throughout the text, please go to the `src` folder.

You can build live projects that can be run with the STM32F767 Nucleo by downloading the entire tree and importing `Chapter_7` as an Eclipse project. To do this, go to [https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers](https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers).

# Creating tasks and starting the scheduler

In order to get an RTOS application up and running, a few things need to happen:

1.  The MCU hardware needs to be initialized.
2.  Task functions need to be defined.
3.  RTOS tasks need to be created and mapped to the functions that were defined in *step 2*.
4.  The RTOS scheduler must be started.

It is possible to create additional tasks after starting the scheduler. If you are unsure of what a task is, or why you would want to use one, please review [Chapter 2](84f04852-827d-4e79-99d7-6c954ba3e93c.xhtml), *Understanding RTOS Tasks.*

Let's break down each of these steps.

# Hardware initialization

Before we can do anything with the RTOS, we need to make sure that our hardware is configured properly. This will typically include carrying out activities such as ensuring GPIO lines are in their proper states, configuring external RAM, configuring critical peripherals and external circuitry, performing built-in tests, and so on. In all of our examples, MCU hardware initialization can be performed by calling `HWInit()`, which performs all of the basic hardware initialization required:

```cpp
int main(void)
{
    HWInit();
```

In this chapter, we'll be developing an application that blinks a few LED lights. Let's define the behavior we'll be programming and take a look at what our individual task functions look like.

# Defining task functions

Each of the tasks, that is, `RedTask`, `BlueTask`, and `GreenTask`, has a function associated with it. Remember – a task is really just an infinite `while` loop with its own stack and a priority. Let's cover them one by one.

`GreenTask` sleeps for a little while (1.5 seconds) with the Green LED on and then deletes itself. There are a few noteworthy items here, some of which are as follows:

*   Normally, a task will contain an infinite `while` loop so that it doesn't return. `GreenTask` still doesn't return since it deletes itself.
*   You can easily confirm `vTaskDelete` doesn't allow execution past the function call by looking at the Nucleo board. The green light will only be on for 1.5 seconds before shutting off permanently. Take a look at the following example, which is an excerpt from `main_taskCreation.c`:

```cpp
void GreenTask(void *argument)
{
  SEGGER_SYSVIEW_PrintfHost("Task1 running \
                 while Green LED is on\n");
  GreenLed.On();
  vTaskDelay(1500/ portTICK_PERIOD_MS);
  GreenLed.Off();

  //a task can delete itself by passing NULL to vTaskDelete
  vTaskDelete(NULL);

  //task never get's here
  GreenLed.On();
}
```

The full source file for `main_taskCreation.c` is available at [https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/blob/master/Chapter_7/Src/main_taskCreation.c](https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/blob/master/Chapter_7/Src/main_taskCreation.c).

`BlueTask` blinks the blue LED rapidly for an indefinite period of time, thanks to the infinite `while` loop. However, the blue LED blinks are cut short because `RedTask` will delete `BlueTask` after 1 second. This can be seen in the following example, which is an excerpt from `Chapter_7/Src/main_taskCreation.c`:

```cpp
void BlueTask( void* argument )
{
  while(1)
  {
    SEGGER_SYSVIEW_PrintfHost("BlueTaskRunning\n");
    BlueLed.On();
    vTaskDelay(200 / portTICK_PERIOD_MS);
    BlueLed.Off();
    vTaskDelay(200 / portTICK_PERIOD_MS);
  }
}
```

`RedTask` deletes `BlueTask` on its first run and then continues to blink the red LED indefinitely. This can be seen in the following excerpt from `Chapter_7/Src/main_taskCreation.c`:

```cpp
void RedTask( void* argument )
{
  uint8_t firstRun = 1;

  while(1)
  {
    lookBusy();

    SEGGER_SYSVIEW_PrintfHost("RedTaskRunning\n");
    RedLed.On();
    vTaskDelay(500/ portTICK_PERIOD_MS);
    RedLed.Off();
    vTaskDelay(500/ portTICK_PERIOD_MS);

    if(firstRun == 1)
    {
      vTaskDelete(blueTaskHandle);
      firstRun = 0;
    }
  }
}
```

So, the preceding functions don't look like anything special – and they're not. They are simply standard C functions, two of which have infinite `while` loops in them. How do we go about creating FreeRTOS tasks out of these plain old functions? 

# Creating tasks

Here's what the prototype for FreeRTOS task creation looks like:

```cpp
BaseType_t xTaskCreate( TaskFunction_t pvTaskCode,
                        const char * const pcName,    
                        configSTACK_DEPTH_TYPE  usStackDepth,
                        void *pvParameters,
                        UBaseType_t uxPriority,
                        TaskHandle_t *pxCreatedTask);

```

In our example, the call to the preceding prototype looks like this:

```cpp

retVal = xTaskCreate(Task1, "task1", StackSizeWords, NULL, tskIDLE_PRIORITY + 2, tskHandlePtr);
```

This function call might be a little longer than expected – let's break it down:

*   `Task1`: The name of the function that implements the infinite `while` loop that makes up the task.
*   `"task1"`: This is a human-friendly name used to reference the task during debugging (this is the string that shows up in tools such as Ozone and SystemView).
*   `StackSizeWords`: The number of *words* reserved for the task's stack.
*   `NULL`: A pointer that can be passed to the underlying function. Make sure the pointer is still valid when the task finally runs after starting the scheduler.
*   `tskIDLE_PRIORITY + 2 `: This is the priority of the task being created. This particular call is setting the priority to two levels higher than the priority of the IDLE task (which runs when no other tasks are running).

*   `TaskHandlePtr`: This is a pointer to a `TaskHandle_t` data type (this is a *handle* that can be passed to other tasks to programmatically reference the task).
*   **Return value**: The `x` prefix of `**x**TaskCreation` signifies that it returns something. In this case, either `pdPASS` or `errCOULD_NOT_ALLOCATE_REQUIRED_MEMORY ` is returned, depending on whether or not heap space was successfully allocated. **You must check this return value!**

At least one task needs to be created before starting the scheduler. Because the call to start the scheduler doesn't return, it won't be possible to start a task from `main` after making a call to start the scheduler. Once the scheduler is started, tasks can create new tasks as necessary.

Now that we've got a good idea of what the input parameters for creating a task are, let's take a look at why it's so important to check the return value.

# Checking the return value

When creating a few tasks in `main` before starting the scheduler, it's necessary to check the return values as each task is created. Of course, there are many ways to accomplish this. Let's take a look at two of them:

1.  The first is by wrapping the call in an `if` statement with an inlined infinite `while` loop:

```cpp
if( xTaskCreate(GreenTask, "GreenTask", STACK_SIZE, NULL, 
                tskIDLE_PRIORITY + 2, NULL) != pdPASS){ while(1) }
```

2.  The second is by using ASSERT rather than the infinite `while` loop. If your project has ASSERT support, then it would be better to use ASSERT, rather than the infinite `while` loop. Since our project already has HAL included, we can make use of the `assert_param` macro:

```cpp
retVal = xTaskCreate(BlueTask, "BlueTask", STACK_SIZE, NULL, tskIDLE_PRIORITY + 1, &blueTaskHandle);
assert_param(retVal == pdPASS);
```

`assert_param` is an STM supplied macro that checks whether a condition is true. If the condition evaluates as false, then `assert_failed` is called. In our implementation, `assert_failed` prints out the failing function name and line and enters an infinite `while` loop:

```cpp
void assert_failed(uint8_t *file, uint32_t line)
{ 
  SEGGER_SYSVIEW_PrintfHost("Assertion Failed:file %s \
                            on line %d\r\n", file, line);
  while(1);
}
```

You will learn more about using assertions and how to configure them in [Chapter 17](50d2b6c3-9a4e-45c3-9bfc-1c7f58de0b98.xhtml), *Troubleshooting Tips and Next Steps.*

Now that we have created some tasks, let's get the scheduler started and the code running on our hardware, and watch some lights blink!

# Starting the scheduler

With all of the options we have for creating tasks, you might be thinking that starting the scheduler would be a complex affair. You'll be pleasantly surprised at how easy it is:

```cpp
//starts the FreeRTOS scheduler - doesn't
//return if successful
vTaskStartScheduler();
```

Yep, just one line of code and no input parameters!

The `v` in front of the function name indicates it returns void. In reality, this function never returns – unless there is a problem. It is the point where `vTaskStartScheduler()` is called that the program transitions from a traditional single super loop to a multi-tasking RTOS.

After the scheduler is started, we'll need to think about and understand the different states the tasks are in so we can debug and tune our system properly.

For reference, here's the entirety of `main()` we've just built up through the various examples. This excerpt has been taken from `main_taskCreation.c`:

```cpp
int main(void)
{
 HWInit();

 if (xTaskCreate(GreenTask, "GreenTask", 
 STACK_SIZE, NULL,
 tskIDLE_PRIORITY + 2, NULL) != pdPASS)
 { while(1); }

 assert_param(xTaskCreate(BlueTask, "BlueTask", STACK_SIZE,NULL,
 tskIDLE_PRIORITY + 1, &blueTaskHandle) == pdPASS);

  xTaskCreateStatic( RedTask, "RedTask", STACK_SIZE, NULL,
                     tskIDLE_PRIORITY + 1,
                     RedTaskStack, &RedTaskTCB);

  //start the scheduler - shouldn't return unless there's a problem
  vTaskStartScheduler();

  while(1){}
}
```

Now that we've learned how to create tasks and get the scheduler up and running, the last detail to cover in this example is how to go about deleting a task.

# Deleting tasks

In some cases, it may be advantageous to have a task run and, eventually, after it has accomplished everything it needs to, remove it from the system. For example, in some systems with fairly involved startup routines, it might be advantageous to run some of the late initialization code inside a task. In this case, the initialization code would run, but there is no need for an infinite loop. If the task is kept around, it will still have its stack and TCB wasting FreeRTOS heap space. Deleting the task will free the task's stack and TCB, making the RAM available for reuse.

All of the critical initialization code should be run long before the scheduler starts.

# The task deletes itself

The simplest way to delete a task after it has finished doing useful work is to call `vTaskDelete()` with a `NULL` argument from within the task, as shown here:  

```cpp
void GreenTask(void *argument)
{
  SEGGER_SYSVIEW_PrintfHost("Task1 running \
                             while Green LED is on\n");
  GreenLed.On();
  vTaskDelay(1500/ portTICK_PERIOD_MS);
  GreenLed.Off();

  //a task can delete itself by passing NULL to vTaskDelete
  vTaskDelete(NULL);

  //task never get's here
  GreenLed.On();
}
```

This will immediately terminate the task code. The memory on the FreeRTOS heap associated with the TCB and task stack will be freed when the IDLE task runs.

In this example, the green LED will turn on for 1.5 seconds and then shut off. As noted in the code, the instructions after `vTaskDelete()` will never be reached.

# Deleting a task from another task

In order to delete a task from another task, `blueTaskHandle` needs to be passed to `xTaskCreate` and its value populated. `blueTaskHandle` can then be used by other tasks to delete `BlueTask,` as shown here:

```cpp
TaskHandle_t blueTaskHandle;
int main(void)
{
    HWInit();
    assert_param( xTaskCreate(BlueTask, "BlueTask", STACK_SIZE,
                  NULL, tskIDLE_PRIORITY + 1, &blueTaskHandle) == 
                  pdPASS);
    xTaskCreateStatic( RedTask, "RedTask", STACK_SIZE, NULL,
                       tskIDLE_PRIORITY + 1, RedTaskStack,
                       &RedTaskTCB);
    vTaskStartScheduler();
    while(1);
}

void RedTask( void* argument )
{
    vTaskDelete(blueTaskHandle);
}
```

The actual code in `main.c` results in the blue LED blinking for ~ 1 second before being deleted by `RedTask`. At this point, the blue LED stops blinking (since the task turning the LED on/off isn't running anymore).

There are a few things to keep in mind before deciding that deleting tasks is desirable:

*   The heap implementation used must support freeing memory (refer to [Chapter 15,](0f98e454-9804-4589-9854-5c38c9d8d416.xhtml)* FreeRTOS Memory Management*, for details).
*   Like any embedded heap implementation, it is possible for a heavily used heap to become fragmented if different sized elements are constantly being added and removed.
*   `#define configTaskDelete` must be set to `true` in `FreeRTOSConfig.h`.

That's it! We now have a FreeRTOS application. Let's get everything compiled and program the image onto the Nucleo board.

# Trying out the code

Now that you've learned how to set up a few tasks, let's go through how to get it running on our hardware. Running the examples, experimenting with breakpoints to observe execution, and sifting through traces in SystemView will greatly enhance your intuition of how an RTOS behaves. 

Let's experiment with the preceding code:

1.  Open the `Chapter_7 STM32CubeIDE` project and set `TaskCreationBuild` as the active build:

![](img/6c62a307-c9bf-4c88-bcd8-aac7862704c5.png)

2.  Right-click on the project and select Build Configurations.
3.  Select the desired build configuration (`TaskCreationBuild` contains `main_taskCreation.c`).
4.  Select Build Project to build the active configuration.

After that, experiment with using Ozone to load and single-step through the program (details on how to do this were covered in [Chapter 6](699daa80-06ae-4acc-8b93-a81af2eb774b.xhtml), *Debugging Tools for Real-Time Systems*). SystemView can also be used to watch the tasks run in real time. Here's a quick example of a bird's-eye view of what's going on:

![](img/f8490c6b-8777-48f7-b594-0cd1f063089d.png)

Let's go over this step by step:

1.  `GreenTask` sleeps for 1.5 seconds, then deletes itself and never runs again (notice the absence of additional tick lines in the `GreenTask` row).
2.  `BlueTask` executes for 1 second before being deleted by `RedTask`.
3.  `RedTask` continues to blink the red LED indefinitely.
4.  `RedTask` deletes `BlueTask`. Deletions aren't trivial – we can see from the callout it takes 7.4 ms to delete `BlueTask`.

Congratulations, you've just made it through writing, compiling, loading, and analyzing an RTOS application! What?! You haven't gone through and actually *run* the application on hardware yet?! Really? If you're serious about learning, you should *seriously* consider getting a Nucleo board so that you can run the examples on actual hardware. All of the examples in this book are full-blown projects, ready to go!

One of the things we glossed over here was whyacall to `xTaskCreate()` can fail. That's an excellent question – let's find out!

# Task memory allocation

One of the parameters for `xTaskCreate()` defines the task's stack size. But where does the RAM being used for this stack come from? There are two options – *dynamically allocated* memory and *statically allocated* memory. 

Dynamic memory allocation is implemented with a heap. FreeRTOS ports contain several different options regarding how heaps are implemented. [Chapter 15](0f98e454-9804-4589-9854-5c38c9d8d416.xhtml), *FreeRTOS Memory Management*, provides details on how to select an appropriate heap implementation for a given project. For now, it is sufficient to assume a heap is available.

Static allocation permanently reserves RAM for a variable for the life of the program. Let's see what each approach looks like.

# Heap allocated tasks

The call from the beginning of this section uses the heap to store the stack:

```cpp
xTaskCreate(Task1, "task1", StackSizeWords, TaskHandlePtr, tskIDLE_PRIORITY + 2, NULL);
```

`xTaskCreate()` is the simpler of the two methods to call. It will use memory from the FreeRTOS heap for Task1's stack and the **Task Control Block** (**TCB**).

# Statically allocated tasks

Tasks that are created without using the FreeRTOS heap require the programmer to perform allocation for the task's stack and TCB before creating the task. The static version of task creation is `xTaskCreateStatic()`.

The FreeRTOS prototype for `xTaskCreateStatic()` is as follows:

```cpp
TaskHandle_t xTaskCreateStatic( TaskFunction_t pxTaskCode,
                                 const char * const pcName,
                                 const uint32_t ulStackDepth,
                                 void * const pvParameters,
                                 UBaseType_t uxPriority,
                                 StackType_t * const puxStackBuffer,
                                 StaticTask_t * const pxTaskBuffer );

```

Let's take a look at how this is used in our example, which creates a task with a statically allocated stack:

```cpp
static StackType_t RedTaskStack[STACK_SIZE];
static StaticTask_t RedTaskTCB;
xTaskCreateStatic( RedTask, "RedTask", STACK_SIZE, NULL,
                   tskIDLE_PRIORITY + 1,
                   RedTaskStack, &RedTaskTCB);
```

Unlike `xTaskCreate()`, `xTaskCreateStatic()` is guaranteed to always create the task, provided `RedTaskStack` or `RedTaskTCB` isn't `NULL`. As long as your toolchain's linker can find space in RAM to store the variables, the task will be created successfully.

`configSUPPORT_STATIC_ALLOCATION` must be set to `1` in `FreeRTOSConfig.h` if you wish to make use of the preceding code.

# Memory protected task creation

Tasks can also be created in a memory protected environment, which guarantees a task only accesses memory specifically assigned to it. Implementations of FreeRTOS are available that take advantage of on-board MPU hardware.

Please refer to[Chapter 4](c52d7cdb-b6cb-41e8-8d75-72494bc9d4d3.xhtml), *Selecting the Right MCU*, for details on MPUs. You can also find a detailed example on how to use the MPU in [Chapter 15](0f98e454-9804-4589-9854-5c38c9d8d416.xhtml), *FreeRTOS Memory Management.*

# Task creation roundup

Since there are a few different ways of creating tasks, you might be wondering which one should be used. All implementations have their strengths and weaknesses, and it really does depend on several factors. The following table shows a summary of the three ways of creating tasks, with their relative strengths represented by arrows – ⇑ for better, ⇓ for worse, and  ⇔ for neutral:

| **Characteristic** | **Heap** | **MPU Heap** | **Static ** |
| Ease of use | ⇑ | ⇓ | ⇔ |
| Flexibility | ⇑ | ⇓ | ⇔ |
| Safety | ⇓ | ⇑ | ⇔ |
| Regulatory Compliance | ⇓ | ⇑ | ⇔ |

As we can see, there is no clear-cut answer as to which system to use. However, if your MCU doesn't have an MPU on-board, there won't be an option to use the MPU variant.

The FreeRTOS heap-based approach is the easiest to code, as well as the most flexible of the three choices. This flexibility comes from the fact that tasks can be deleted, rather than simply forgotten about. Statically created tasks are the next easiest, with only an additional two lines required to specify the TCB and the task stack. They aren't as flexible since there is no way to free memory defined by a static variable. Static creation can also be more desirable in some regulatory environments that forbid the use of the heap altogether, although in most cases, the most FreeRTOS heap-based approach is acceptable – especially heap implementations 1, 2, and 3.  

*What is a heap implementation?* Don't worry about it yet. We'll learn about heap options for FreeRTOS in detail in [Chapter 15](0f98e454-9804-4589-9854-5c38c9d8d416.xhtml), *FreeRTOS Memory Management*.

The MPU variant is the most involved of the three, but it is also the safest since the MPU guarantees the task isn't writing outside of its allowed memory.  

Using statically defined stacks and TCBs has the advantage that the total program footprint can be analyzed by the linker. This ensures that if a program compiles and fits into the hardware constraints of the MCU, it won't fail to run due to a lack of heap space. With heap-based task creation, it is possible for a program to compile but have a runtime error that causes the entire application not to run. In other cases, the application may run for some time but then fail due to a lack of heap memory.

# Understanding FreeRTOS task states

As explained in [Chapter 2](84f04852-827d-4e79-99d7-6c954ba3e93c.xhtml), *Understanding RTOS Task*s, all of the context switching between tasks happens *in the background*, which is very convenient for the programmer responsible for implementing tasks. This is because it frees them from adding code into each task that attempts to load balance the system. While the task code isn't *explicitly* performing the task state transitions, it *is* interacting with the kernel. Calls to the FreeRTOS API cause the kernel's scheduler to run, which is responsible for transitioning the tasks between the necessary states.

# Understanding different task states

Each transition shown in the following state diagram is caused by either an API call being made by your code or an action being taken by the scheduler. This is a simplified graphical overview of the possible states and transitions, along with a description of each:

![](img/05c6737c-baa0-4650-8e6a-6c3587bd589b.png)

Let's look at them one by one.

# Running

A task in the running state is performing work; it is the only task that is in context. It will run until it either makes a call to an API that causes it to move to the `Blocked` state or it gets switched out of context by the scheduler due to a higher priority (or time-sliced task of equal priority). Examples of API calls that would cause a task to move from `Running` to `Blocked` include attempting to read from an empty queue or attempting to take a mutex that isn't available.

# Ready

Tasks that are sitting in the ready state are simply waiting for the scheduler to give them processor context so they can run. For example, if *Task A* has gone into the `Blocked` state, waiting on an item to be added to a queue it was waiting on, then *Task A* will move into the `Ready` state. The scheduler will evaluate whether or not *Task A* is the highest priority task ready to run in the system. If *Task A* is the highest priority task that is ready, it will be given processor context and change to the `Running` state. Note that tasks can share the same priority. In this case, the scheduler will switch them between `Ready` and `Running` by using a round-robin scheduling scheme (see [Chapter 2](84f04852-827d-4e79-99d7-6c954ba3e93c.xhtml),* Understanding RTOS Tasks*, for an example of this).

# Blocked

A `Blocked` task is a task that is waiting for something. There are two ways for a task to move out of the `Blocked` state: either an event will trigger a transition from `Blocked` to `Ready` for the task, or a timeout will occur, placing the task in the `Ready` state. 

This is a very important feature of an RTOS:* each blocking call is time-bound*. That is, a task will only block while waiting for an event for as long as the programmer specifies it can be blocked. This is an important distinction between RTOS firmware programming and general-purpose application programming. For example, *an attempt to take a mutex that will fail if the mutex isn't available within the specified amount of time*. The same is true for API calls that accept and push data onto queues, as well as all the other non-interrupt API calls in FreeRTOS.

While a task is in the `Blocked` state, it doesn't consume any processor time. When a task is transitioned out of the `Blocked` state by the scheduler, it will be moved to the `Ready` state, allowing the calling task to run when it becomes the highest priority task in the system.

# Suspended

The `Suspended` state is a bit of a special case since it requires explicit FreeRTOS API calls to enter and exit. Once a task enters the `Suspended` state (via the `vTaskSuspend()` API call), it is effectively ignored by the scheduler until the `vTaskRusme()` API call is made. This state causes the scheduler to effectively ignore the task until it is moved into the `Ready` state by an explicit API call. Just like the `Blocked` state, the `Suspended` state will not consume any processor time.

Now that we understand the various task states and how they interact with different parts of the RTOS, we can learn how to optimize an application so that it makes efficient use of tasks.

# Optimizing task states

Thoughtful optimizations can be made to minimize the time tasks stay in the `Running` state. Since a task only consumes significant CPU time when in the `Running` state, it is usually a good idea to minimize time spent there on legitimate work.

As you'll see, polling for events does work but is usually an unnecessary waste of CPU cycles. If properly balanced with task priorities, the system can be designed to be both responsive to important events while also minimizing CPU time. There can be a few different reasons for optimizing an application in this way.

# Optimizing to reduce CPU time

Often, an RTOS is used because many different activities need to happen almost simultaneously. When a task needs to take action because an event occurs, there are a few ways of monitoring for the event.

Polling is when a value is continuously read in order to capture a transition. An example of this would be waiting for a new ADC reading. A *polled* read might look something like this:

```cpp
uint_fast8_t freshAdcReading = 0;
while(!freshAdcReading)
{
    freshAdcReading = checkAdc();
}
```

While this code *will* detect when a new ADC reading has occurred, it will also cause the task to continually be in the `Running` state. If this happens to be the highest priority task in the system, this will *starve* the other tasks of CPU time. This is because there is nothing to force the task to move out of the `Running` state – it is continually checking for a new value.

To minimize the time a task spends in the `Running` state (continually polling for a change), we can use the hardware included in the MCU to perform the same check without CPU intervention. For example, **interrupt service routines** (**ISRs**) and **direct memory access** (**DMA**) can both be used to offload some of the work from the CPU onto different hardware peripherals included in the MCU. An ISR can be interfaced with RTOS primitives to notify a task when there is valuable work to be done, thereby eliminating the need for CPU-intensive polling. [Chapter 8](c6d7a0c6-6f18-4e06-a372-cd1605942ecd.xhtml), *Protecting Data and Synchronizing Tasks*, will cover polling in more detail, as well as multiple efficient alternatives.  

# Optimizing to increase performance

Sometimes, there are tight timing requirements that need a low amount of jitter. Other times, a peripheral requiring a large amount of throughput may need to be used. While it may be possible to meet these timing requirements by polling inside a high priority task, it is often more reliable (and more efficient) to implement the necessary functionality inside an ISR. It may also be possible to not involve the processor at all by using DMA. Both of these options prevent tasks from expending worthless CPU cycles on polling loops and allow them to spend more time on useful work.

Take a look at the *Introducing DMA* section in [Chapter 2](84f04852-827d-4e79-99d7-6c954ba3e93c.xhtml), *Understanding RTOS Tasks*, for a refresher on DMA. Interrupts are also covered.

Because interrupts and DMA can operate completely below the RTOS (not requiring any kernel intervention), they can have a dramatically positive effect on creating a deterministic system. We'll look at how to write these types of drivers in detail in [Chapter 10](dd741273-db9a-4e9a-a699-b4602e160b84.xhtml), *Drivers and ISRs*.

# Optimizing to minimize power consumption

With the prevalence of battery-powered and energy harvesting applications, programmers have another reason to make sure the system is using as few CPU cycles as possible. Similar ideas are present in creating power-conscious solutions, but instead of maximizing determinism, the focus is often on saving CPU cycles and operating with slower clock rates.

There is an additional feature in FreeRTOS that is available for experimentation in this space – the tickless IDLE task. This trades timing accuracy for a reduction in how often the kernel runs. Normally, if the kernel was set up for a 1 ms tick rate (waiting up to every millisecond to check for the next activity), it would wake up and run the code at 1 kHz. In the case of a *tickless* IDLE task, the kernel only wakes up when necessary.

Now that we've covered some starting points on how to improve an already running system, let's turn our attention to something more dire: a system that doesn't start at all!

# Troubleshooting startup problems

So, let's say you're working on a project and things haven't gone as planned. Instead of being rewarded with blinky lights, you're left staring at a very non-blinky piece of hardware. At this stage, it's usually best to get the debugger up and running, rather than making random guesses about what might be wrong and sporadically changing sections of code.

# None of my tasks are running!

Most often, startup problems in the early stages of development will be caused by not allocating enough space in the FreeRTOS heap. There are typically two symptoms that result from this.

# Task creation failed

In the following case, the code will get *stuck* before running the scheduler (no lights will be blinking). Perform the following steps to determine why:

1.  Using a debugger, step through task creation until you find the offending task. This is easy to do because all of our attempts to create tasks will only progress if the tasks were successfully created. 
2.  In this case, you'll see that `xTaskCreate` doesn't return `pdPASS ` when creating `BlueTask`. The following code is requesting a 50 KB stack for `BlueTask`:

```cpp
int main(void)
{
  HWInit();

  if (xTaskCreate(GreenTask, "GreenTask", 
                  STACK_SIZE, NULL,
                  tskIDLE_PRIORITY + 2, NULL) != pdPASS)
      { while(1); }

  //code won't progress past assert_failed (called by
  //assert_param on failed assertions)
  retval = (xTaskCreate(BlueTask, "BlueTask",
               STACK_SIZE*100,NULL,
               tskIDLE_PRIORITY + 1, &blueTaskHandle);
  assert_param(retVal == pdPASS);
```

You can find the full source for this example here: [https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/blob/master/Chapter_7/Src/main_FailedStartup.c](https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/blob/master/Chapter_7/Src/main_FailedStartup.c).

Here's the code for `assert_failed`. The infinite `while` loop makes it very easy to track down the offending line using a debug probe and looks at the call stack:

```cpp
void assert_failed(uint8_t *file, uint32_t line)
{ 
  SEGGER_SYSVIEW_PrintfHost("Assertion Failed:file %s \
                            on line %d\r\n", file, line);
  while(1);    
}
```

3.  Using the Ozone call stack, the failed assertion can be tracked back to creating `BlueTask` on line 37 of `main_FailedStartup.c`:

![](img/a97373bd-d3f0-4926-926f-e459b995ec4d.png)

4.  After determining the cause of failure to be a task that failed to be created, it is time to consider increasing the FreeRTOS heap by modifying `FreeRTOSConfig.h`. This is done by modifying `configTOTAL_HEAP_SIZE` (it's currently set to 15 KB). This excerpt has been taken from `Chapter_7/Inc/FreeRTOSConfig.h`***:***

```cpp
#define configTOTAL_HEAP_SIZE ((size_t)15360)
```

Unlike stack size specifications, which are specified in *words* (for example, `configMINIMAL_STACK_SIZE` ) and passed as arguments to `xTaskCreate`, `configTOTAL_HEAP_SIZE ` is specified in bytes.

Care needs to be taken when increasing `configTOTAL_HEAP_SIZE`. See the *Important notes* section on considerations to be made.

# Scheduler returns unexpectedly

It is also possible to run into issues with `vStartScheduler` returning this:

```cpp
//start the scheduler - shouldn't return unless there's a problem
vTaskStartScheduler();

//if you've wound up here, there is likely
//an issue with overrunning the freeRTOS heap
while(1)
{
}
```

This is simply another symptom of the same underlying issue – inadequate heap space. The scheduler defines an IDLE task that requires `configMINIMAL_STACK_SIZE` words of heap space (plus room for the TCB).

If you're reading this section because you *actually* *have* a program that isn't starting and you're *not* experiencing either of these symptoms, do not worry! There's an entire chapter in the back of this book, just for you. Check out [Chapter 17](50d2b6c3-9a4e-45c3-9bfc-1c7f58de0b98.xhtml), *Troubleshooting Tips and Next Steps*. It was actually created from real-world problems that were encountered during the creation of the example code in this book.

There are a few more considerations to be made if you have an application that is refusing to start.

# Important notes

RAM on MCU-based embedded systems is usually a scarce resource. When increasing the heap space available to FreeRTOS (`configTOTAL_HEAP_SIZE`), you'll be reducing the amount of RAM available to non-RTOS code.

There are several factors to be aware of when considering increasing the heap available to FreeRTOS via `configTOTAL_HEAP_SIZE`:

*   If a significantly sized non-RTOS stack has been defined – that is, the stack that is used by any code that isn't running inside a task (typically configured inside the startup file). Initialization code will use this stack, so if there are any deep function calls, this stack won't be able to be made especially small. USB stacks that have been initialized before the scheduler is started can be a culprit here. One possible solution to this on RAM-constrained systems is to move the bloated initialization code into a task with a large enough stack. This may allow for the non-RTOS stack to be minimized further.  
*   ISRs will be making use of the non-RTOS stack as well, but they'll need it for the entire duration of the program. 
*   Consider using statically allocated tasks instead – it's guaranteed there will be enough RAM when the program runs.

A more in-depth discussion on memory allocation can be found in [Chapter  15](0f98e454-9804-4589-9854-5c38c9d8d416.xhtml)*, FreeRTOS Memory Management.*

# Summary

In this chapter, we've covered the different ways of defining tasks and how to start the FreeRTOS scheduler. Along the way, we covered some more examples of using Ozone, SystemView, and STM32CubeIDE (or any Eclipse CDT-based IDE). All of this information was used to create a live demo that tied all of the RTOS concepts regarding task creation with the mechanics of actually loading and analyzing code running on embedded hardware. There were also some suggestions on how *not*to monitor for events (polling).  

In the next chapter, we'll introduce what you *should* be using for event monitoring. Multiple ways of implementing inter-task signaling and synchronization will be covered – all through examples. There's going to be LOTS of code and a bunch of hands-on analysis using the Nucleo board.

# Questions

As we conclude this chapter, here is a list of questions so that you can test your knowledge regarding this chapter's material. You will find the answers in the *Assessments* section of the *Appendix*:

1.  How many options are available when starting FreeRTOS tasks?
2.  The return value needs to be checked when calling `xTaskCreate()`.
    *   True
    *   False
3.  The return value needs to be checked when calling `vTaskStartScheduler()`.
    *   True
    *   False
4.  Because RTOSes are bloated middleware, FreeRTOS requires a huge heap for storing all of the task stacks, regardless of what functions the task is performing.
    *   True
    *   False
5.  Once a task has been started, it can never be removed.
    *   True
    *   False

# Further reading

*   Free RTOS customization (`FreeRTOSConfig.h`): [https://www.freertos.org/a00110.htm](https://www.freertos.org/a00110.html)