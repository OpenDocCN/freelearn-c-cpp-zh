# Protecting Data and Synchronizing Tasks

What do race conditions, corrupt data, and missed real-time deadlines all have in common? Well, for one, they are all mistakes that can be easily made when operations are performed in parallel. These are also mistakes that are avoidable (in part) through using the right tools.

This chapter covers many of the mechanisms that are used to synchronize tasks and protect shared data. All the explanations in this chapter will contain example code and analysis that will have been performed using Ozone and SystemView.

First, we will explore the differences between semaphores and mutexes. Then, you will understand how, when, and why to use a semaphore. You'll also learn about race conditions and see how a mutex can avoid such situations. Example code will be provided throughout. The concept of race conditions will be introduced and fixed using a mutex in live code that can be run and analyzed on the Nucleo development board. Finally, FreeRTOS software timers will be introduced and a discussion of common real-world use cases for RTOS-based software timers and MCU hardware peripheral timers will be provided.

We will cover the following topics in this chapter:

*   Using semaphores
*   Using mutexes
*   Avoiding race conditions
*   Using software timers

# Technical requirements

To complete the hands-on exercises in this chapter, you will require the following:

*   Nucleo F767 development board
*   Micro USB cable
*   ST/Atollic STM32CubeIDE and its source code (the instructions for this can be found in [Chapter 5](84a945dc-ff6c-4ec8-8b9c-84842db68a85.xhtml)*, Selecting an IDE – Setting Up Our IDE*)
*   SEGGER JLink, Ozone, and SystemView ([Chapter 6](699daa80-06ae-4acc-8b93-a81af2eb774b.xhtml), *Debugging Tools for Real-Time Systems*)

The easiest way to build the examples in this chapter is to build all Eclipse *configurations* at once, and then load and view them using Ozone. To do this, follow these steps:

1.  In STM32CubeIDE, right-click on the project.
2.  Select Build.
3.  Select Build All. All the examples will be built into their own named subdirectory (this may take a while). 
4.  In Ozone, you can now quickly load each `<exampleName>.elf` file. See [Chapter 6](699daa80-06ae-4acc-8b93-a81af2eb774b.xhtml),* Debugging Tools for Real-Time Systems*, for instructions on how to do this. The correct source files that are linked in the executable will be automatically displayed.

All the source code for this chapter can be found at [https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_8](https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_8).

# Using semaphores

We've mentioned several times now that tasks are meant to be programmed so that they're *running in parallel*. This means that, by default, they have no relation to one another in time. No assumptions can be made as to where tasks are in their execution with respect to one another – unless they are explicitly synchronized. Semaphores are one mechanism that's used to provide synchronization between tasks.  

# Synchronization via semaphores

The following is a diagram of the abstract example we covered back in [Chapter 2](a410ddd6-10eb-4e97-965e-e390f4dc2890.xhtml), *Task Signaling and Communication Mechanisms*:

![](img/d82cc6e2-4a19-4439-ba39-632cfccaedb0.png)

The preceding diagram shows `TaskB` waiting on a semaphore from `TaskA`. Each time `TaskB` acquires the desired semaphore, it will continue its loop. `TaskA` repeatedly *gives* a semaphore, which effectively synchronizes when `TaskB` runs. Now that we have a full development environment set up, let's take a look at what this looks like with some actual code. Then, we'll run it on hardware and blink a few LEDs to see exactly what this behavior looks like in the real world.

# Setting up the code

First, the semaphore needs to be created, and its handle (or pointer) has to be stored so that it can be used between tasks. The following excerpt has been taken from `mainSemExample.c`:

```cpp
//create storage for a pointer to a semaphore
SemaphoreHandle_t semPtr = NULL;

int main(void)
{
    //.... init code removed.... //

    //create a semaphore using the FreeRTOS Heap
 semPtr = xSemaphoreCreateBinary(); //ensure pointer is valid (semaphore created successfully)
 assert_param(semPtr != NULL);
```

The semaphore pointer, that is, `semPtr`, needs to be placed in a location that is accessible to other functions that need access to the semaphore. For example, don't declare `semPtr` as a local variable inside a function – it won't be available to other functions and it will go out of scope as soon as the function returns.

To see what's going on with the source code *and* see how the system is reacting, we'll associate a few different LEDs with task A and task B.

`Task A` will toggle the green LED and *give* a semaphore every five times it's run through the blinking loop, as shown in the following excerpt from `mainSemExample.c`:

```cpp
void GreenTaskA( void* argument )
{
  uint_fast8_t count = 0;
  while(1)
  {
    //every 5 times through the loop, give the semaphore
    if(++count >= 5)
    {
      count = 0;
      SEGGER_SYSVIEW_PrintfHost("Task A (green LED) gives semPtr");
 xSemaphoreGive(semPtr);
    }
    GreenLed.On();
    vTaskDelay(100/portTICK_PERIOD_MS);
    GreenLed.Off();
    vTaskDelay(100/portTICK_PERIOD_MS);
  }
}
```

`Task B`, on the other hand, will rapidly blink the blue LED three times after successfully *taking* the semaphore, as shown in the following excerpt from `mainSemExample.c`:

```cpp
/**
 * wait to receive semPtr and triple blink the Blue LED
 */
void BlueTaskB( void* argument )
{
  while(1)
  {
 if(xSemaphoreTake(semPtr, portMAX_DELAY) == pdPASS)
    {   
        //triple blink the Blue LED
        for(uint_fast8_t i = 0; i < 3; i++)
        {
            BlueLed.On();
            vTaskDelay(50/portTICK_PERIOD_MS);
            BlueLed.Off();
            vTaskDelay(50/portTICK_PERIOD_MS);
        }
    }
    else
    {
        // This is the code that will be executed if we time out
        // waiting for the semaphore to be given
    }
  }
}
```

Great! Now that our code is ready, let's see what this behavior looks like.

FreeRTOS allows for indefinite delays in certain circumstances through the use of `portMAX_DELAY`. As long as `#define INCLUDE_vTaskSuspend 1` is present in `FreeRTOSConfig.h`, the calling task will be suspended indefinitely and the return value of `xSemaphoreTake()` can be safely ignored. When `vTaskSuspend()` is not defined as 1, `portMAX_DELAY` will result in a very long delay (0xFFFFFFF RTOS ticks (~ 49.7 days) on our system), but not an infinite one.

# Understanding the behavior

Here's what this example looks like when viewed using SystemView:

![](img/61787d72-0a4f-4239-8f1c-e77633e1f56a.png)

Notice the following:

*   Blocking with semaphores is efficient as each task is only using 0.01% of the CPU time.
*   A task that is blocked because it is waiting on a semaphore won't run until it is available. This is true even if it is the highest-priority task in the system and no other tasks are currently `READY`.

Now that you've seen an efficient way of synchronizing tasks with a semaphore, let's have a look at another way of achieving the same behavior using polling.  

# Wasting cycles – synchronization by polling

The following example has the exact same behavior as when we're looking at LEDs from the outside of the board – the observable pattern of the LEDs is exactly the same as the previous example. The difference is how much CPU time is being used by continuously reading the same variable.

# Setting up the code

Here's the updated `GreenTaskA()` – only a single line has changed. This excerpt has been taken from `mainPolledExample.c`:

```cpp
void GreenTaskA( void* argument )
{
  uint_fast8_t count = 0;
  while(1)
  {
    //every 5 times through the loop, set the flag
    if(++count >= 5)
    {
      count = 0;
      SEGGER_SYSVIEW_PrintfHost("Task A (green LED) sets flag");
 flag = 1; //set 'flag' to 1 to "signal" BlueTaskB to run
```

Instead of calling `xSmeaphoreGive()`, we're simply setting the `flag` variable to `1`.

A similar small change has been made to `BlueTaskB()`, trading out a `while` loop that polls on `flag`, instead of using `xSemaphoreTake()`. This can be seen in the following excerpt from `mainPolledExample.c`:

```cpp
void BlueTaskB( void* argument )
{
  while(1)
  {
      SEGGER_SYSVIEW_PrintfHost("Task B (Blue LED) starts "\
                                "polling on flag");

    //repeateadly poll on flag. As soon as it is non-zero,
    //blink the blue LED 3 times
 while(!flag);    SEGGER_SYSVIEW_PrintfHost("Task B (Blue LED) received flag");
```

These are the only changes that are required. `BlueTaskB()` will wait to move on (indefinitely) until `flag` is set to something other than `0`. 

To run this example, use the `Chapter_8/polledExample` file's build configuration.

# Understanding the behavior

Since only a few changes were made, we might not expect there to be *that* much of a difference in terms of how the MCU is behaving, given the new code. However, the output that can be observed with SystemView tells a different story:

![](img/06f10d4a-30cf-4cfb-9886-9e31036a2b96.png)

Note the following:

*   `BlueTaskB` is now using 100% of the CPU time while polling the value of `flag` (the 70% CPU load is lower because the task is sleeping while actually blinking the LED).
*   Even though `BlueTaskB` is hogging the CPU, `GreenTaskA` still runs consistently since it has a higher priority. `GreenTaskA` would be starved of CPU if it was a lower priority than `BlueTaskB`.

So, synchronizing tasks by polling on a variable *does* work as expected, but there are some side effects: increased CPU utilization and a strong dependency on task priorities. Of course, there are ways of reducing the CPU load of `BlueTaskB`. We could have added a delay between polling, like so:

```cpp
while(!flag)
{
    vTaskDelay(1);
}
```

This will reduce the CPU load of `BlueTaskB` to around 5%. Beware, though, that this delay also guarantees that `BlueTaskB` has a worst-case delay of at *least* 1 RTOS tick period (1 ms, in our setup).

# Time-bound semaphores

Earlier, we mentioned that one of the critical aspects of RTOSes was their ability to provide a way to time-bound operations; that is, they can guarantee a call doesn't stop a task from executing any longer than is desirable. An RTOS *does not guarantee the successful timeliness of an operation*. It only promises that the call will be returned in an amount of time. Let's have another look at the call for taking a semaphore:

```cpp
BaseType_t xSemaphoreTake( SemaphoreHandle_t xSemaphore,
                            TickType_t xTicksToWait );

```

From the preceding code, we can see the following:

*   `semPtr` is just a pointer to the semaphore.   
*   `maxDelay` is the interesting part of this call – it specifies the maximum amount of time to wait for the semaphore (in RTOS *tick* units). 

*   The return value is `pdPASS` (the semaphore was taken in time) or `pdFALSE` (the semaphore was not taken in time). *It is extremely important to check this return value.*

If a semaphore were to be taken successfully, the return value would be `pdPASS`. This is the only case where the task will continue because a semaphore was given. If the return value is not `pdPASS`, the call to `xSemaphoreTake()` has failed, either because of a timeout or a programming error (such as passing in an invalid `SemaphoreHandle_t`). Let's take a more in-depth look at this with an example.

# Setting up the code 

In this example, we'll be using all three LEDs on the dev board to indicate different states:

*   **Green LED**: `GreenTaskA()` blinks at a steady 5 Hz with a 50% duty cycle.
*   **Blue LED**: Rapid blinks three times when `TaskB()` receives the semaphore within 500 ms.
*   **Red LED**: Turned on after a timeout from `xSemaphoreTake()`. This is left on until it's reset by `TaskB()`, as long as it receives the semaphore within 500 ms of starting to wait for it.

In many systems, missing a deadline can be a cause for (major) concern. It all depends on what it is you're implementing. This example is just a simple loop with a red light for when a deadline is missed. However, other systems may require (emergency) procedures to be taken to prevent significant failure/damage if a deadline is missed.

`GreenTaskA()` has two responsibilities:

*   Blink the green LED
*   *Give* the semaphore at pseudo-random intervals

These responsibilities can be seen in the following code:

```cpp
void GreenTaskA( void* argument )
{
    uint_fast8_t count = 0;
    while(1)
    {
        uint8_t numLoops = StmRand(3,7);
        if(++count >= numLoops)
        {
            count = 0;
 xSemaphoreGive(semPtr);
        }
 greenBlink();
    }
}
```

`TaskB()` also has two responsibilities:

*   Blink the blue LED (as long as the semaphore shows up within 500 ms).
*   Turn on the red LED (if the semaphore doesn't show up within 500 ms). The red LED will stay on until the semaphore has successfully been taken within 500 ms of starting to wait for it:

```cpp
void TaskB( void* argument )
{
    while(1)
    {
 //'take' the semaphore with a 500mS timeout                    
        if(xSemaphoreTake(semPtr, 500/portTICK_PERIOD_MS) == pdPASS)
        {
 //received semPtr in time
            RedLed.Off();
            blueTripleBlink();
        }
        else
 {
 //this code is called when the 
 //semaphore wasn't taken in time 
            RedLed.On();
        }
    }
}
```

This setup guarantees that `TaskB() ` will be taking some action *at least* every 500 ms. 

# Understanding the behavior

When building and loading the firmware included in the `semaphoreTimeBound` build configuration, you'll see something similar to the following when using SystemView:

![](img/c0f7a646-6c8c-4ca0-ac34-905facf9676c.png)

Note the following:

1.  **Marker 1** indicates `TaskB` didn't receive the semaphore within 500 ms. Notice there is no followup execution from `TaskB` – it immediately went back to taking the semaphore again.
2.  **Marker 2** indicates `TaskB` received the semaphore within 500 ms. Looking at the graph, we can see it was actually around 200 ms. The periodic lines (circled in the preceding image) in the `TaskB` lane are the blue LED turning on and off.
3.  After blinking the blue LED, `TaskB` goes back to waiting for the semaphore.

Log messages are indicated by blue *i* icons within the timeline, which helps to associate descriptive comments in code while visualizing behavior. Double-clicking the blue boxes automatically jumps the terminal to the associated log message.

You'll notice that the blue LED doesn't always blink – occasionally, the red LED blinks instead. Each time the red LED blinks, this indicates that `semPtr` was not taken within 500 ms. This shows that the code is attempting to take a semaphore as an upper bound on the amount of time acceptable before *giving up* on the semaphore, possibly triggering an error condition.

As an exercise, see if you can capture a red blink and track where the timeout occurred using the terminal output (on the right) and the timeline output (on the bottom) – how much time elapsed from when `TaskB` attempted to *take the semaphore* and when the red LED blinked? Now, modify the 500 ms timeout in the source code, compile and upload it with Ozone, and watch for the change in SystemView.  

# Counting semaphores

While binary semaphores can only have values between 0 and 1, counting semaphores can have a wider range of values. Some use cases for counting semaphores include simultaneous connections in a communication stack or static buffers from a memory pool.

For example, let's say we have a TCP/IP stack that supports multiple simultaneous TCP sessions, but the MCU only has enough RAM to support three simultaneous TCP sessions. This would be a perfect use case for a counting semaphore.

The counting semaphore for this application needs to be defined so that it has a maximum count of `3` and an initial value of `3` (three TCP sessions are available):

```cpp
SemaphoreHandle_t semPtr = NULL;
semPtr = xSemaphoreCreateCounting( /*max count*/3, /*init count*/ 3);
if(semPtr != NULL)
```

The code that requests to open a TCP session would *take* `semPtr`, reducing its count by 1:

```cpp
if(xSemaphoreTake( semPtr, /*timeoutTicks*/100) == pdPASS)
{
    //resources for TCP session are available
}
else
{
    //timed out waiting for session to become available
}
```

Whenever a TCP session is closed, the code closing the session *gives* `semPtr`, increasing its count by 1:

```cpp
xSemaphoreGive( semPtr );
```

By using a counting semaphore, you can control access to a limited number of available TCP sessions. By doing this, we're accomplishing two things:

*   Limiting the number of simultaneous TCP sessions, thus keeping resource usage in check.
*   Providing time-bound access for creating a TCP session. This means the code is able to specify how long it will wait for a session to become available.

Counting semaphores are useful for controlling access to a shared resource when more than one instance is available.

# Priority inversion (how not to use semaphores)

Since semaphores are used to synchronize multiple tasks and guard shared resources, does this mean we can use them to protect a piece of data that's being shared between two tasks? Since each task needs to know when it is safe to access the data, the tasks need to be synchronized, right? The danger with this approach is that semaphores have no concept of task priority. A higher-priority task waiting on a semaphore being held by a lower-priority task will wait, regardless of what else might be going on in the system. An example of *why* this can become a problem will be shown here.

Here's the conceptual example we covered in [Chapter 3](a410ddd6-10eb-4e97-965e-e390f4dc2890.xhtml)*, Task Signaling and Communication Mechanisms*:

![](img/470ca651-eb41-4c68-98c5-0f8b92f88990.png)

The main problems with this sequence are *steps 3* and *4*. `TaskB` shouldn't be able to preempt `TaskC` if a higher-priority (`TaskA`) task is waiting on the semaphore. Let's look at an example of this *in the wild* with some real code and observe the behavior first-hand!

# Setting up the code

For the actual example, we'll maintain the exact same function names as the theoretical example we covered previously. The *shared resource* will be the function that's used to blink the LEDs. 

The *shared LEDs* are only an example. In practice, you'll often find that data that's been shared between tasks needs to be protected. There is also the chance that the multiple tasks may attempt to use the same hardware peripheral, in which case access to that resource may need to be protected.

To provide some visual feedback, we'll also assign some LEDs to the various tasks. Let's have a look at the code.

# Task A (highest priority)

Task A is responsible for blinking the green LED, but only *after* `semPtr` has been taken (within 200 ms of requesting it). The following excerpt has been taken from `mainSemPriorityInversion.c`*:*

```cpp
while(1)
{
    //'take' the semaphore with a 200mS timeout
    SEGGER_SYSVIEW_PrintfHost("attempt to take semPtr");
 if(xSemaphoreTake(semPtr, 200/portTICK_PERIOD_MS) == pdPASS)
    {
        RedLed.Off();
        SEGGER_SYSVIEW_PrintfHost("received semPtr");
 blinkTwice(&GreenLed);
 xSemaphoreGive(semPtr);
    }
    else
    {
        //this code is called when the 
 //semaphore wasn't taken in time
        SEGGER_SYSVIEW_PrintfHost("FAILED to receive "
                                    "semphr in time");
        RedLed.On();
    }
    //sleep for a bit to let other tasks run
    vTaskDelay(StmRand(10,30));
}
```

This task is the primary focal point of this example, so make sure that you have a solid understanding of the conditional statements around the semaphore being taken within the specified period of time. The semaphore won't always be taken in time.

# Task B (medium priority)

Task B periodically utilizes the CPU. The following excerpt has been taken from `mainSemPriorityInversion.c`:

```cpp
uint32_t counter = 0;
while(1)
{
    SEGGER_SYSVIEW_PrintfHost("starting iteration %ui", counter);
    vTaskDelay(StmRand(75,150));
    lookBusy(StmRand(250000, 750000));
}
```

This task sleeps between 75 and 150 ticks (which doesn't consume CPU cycles) and then performs a busy loop for a variable number of cycles using the `lookBusy()` function. Note that `TaskB` is the medium priority task.

# Task C (low priority)

Task C is responsible for blinking the blue LED, but only *after* the `semPtr` has been taken (within 200 ms of requesting it). The following excerpt has been taken from `mainSemPriorityInversion.c`:

```cpp
while(1)
  {
    //'take' the semaphore with a 200mS timeout
    SEGGER_SYSVIEW_PrintfHost("attempt to take semPtr");
 if(xSemaphoreTake(semPtr, 200/portTICK_PERIOD_MS) == pdPASS)
    {
      RedLed.Off();
      SEGGER_SYSVIEW_PrintfHost("received semPtr");
      blinkTwice(&BlueLed);
 xSemaphoreGive(semPtr);
    }
    else
    {
 //this code is called when the semaphore wasn't taken in time
      SEGGER_SYSVIEW_PrintfHost("FAILED to receive "
                                    "semphr in time");
      RedLed.On();
    }
  }
```

`TaskC()` is relying on the same semaphore as `TaskA()`. The only difference is that `TaskC()` is blinking the blue LED to indicate the semaphore was taken successfully.

# Understanding the behavior

Using Ozone, load `Chapter8_semaphorePriorityInversion.elf` and start the processor. Then, open SystemView and observe the runtime behavior, which will be analyzed here.

There are a few key aspects to keep in mind when looking at this trace:

*   `TaskA` is the highest-priority task in the system. Ideally, if `TaskA` is ready to run, it should be running. Because `TaskA` shares a resource with a lower-priority task (`TaskC`), it will be delayed while `TaskC` is running (if `TaskC` is holding the resource).
*   `TaskB` should not run when `TaskA`* could* run since `TaskA` has a higher priority.
*   We've used the terminal output of SystemView (as well as turned on the red LED) to provide a notification when either `TaskA` or `TaskC` has failed to acquire `semPtr` in time:

```cpp
SEGGER_SYSVIEW_PrintfHost("FAILED to receive "
"semphr in time");
```

Here's how this will look in SystemView:

![](img/adf42c90-b716-439b-b1f0-cdc628b436ed.png)

The numbers in this graph line up with the theoretical example, so if you've been following along closely, you may already know what to expect:

1.  `TaskC` (the lowest-priority task in the system) acquires a binary semaphore and starts to do some work (blinking the blue LED).
2.  Before `TaskC` completes its work, `TaskB` does some work.
3.  The highest-priority task (`TaskA`) interrupts and attempts to acquire the same semaphore, but is forced to wait because `TaskC` has already acquired the semaphore.
4.  `TaskA` times out after 200 ms because `TaskC` didn't have a chance to run (the higher-priority task, `TaskB`, was running instead). It lights up the red LED because of the failure.

The fact that the lower-priority task (`TaskB`) was running while a higher-priority task was ready to run (`TaskA`) but waiting on a shared resource is called *priority inversion*. This is a reason to avoid using semaphores to protect shared resources.

If you look closely at the example code, you'll realize that a semaphore was acquired and then the task holding the semaphore was put to sleep... DON'T EVER DO THIS in a real system. Keep in mind that this is a contrived example *designed to visibly fail.* See the *Using mutexes* section for more information on critical sections.

Luckily, there is an RTOS primitive that has been *specifically designed* for protecting shared resources, all while minimizing the effect of priority inversion – the mutex.

# Using mutexes

**Mutex** stands for **mutual exclusion** – they are explicitly designed to be used in situations where access to a shared resource should be mutually exclusive – meaning the shared resource can only be used by one piece of code at a time.  At their heart, mutexes are simply binary semaphores with one (very important) difference: priority inheritance. In the previous example, we saw the highest-priority task waiting on two lower-priority tasks to complete, which caused a priority inversion. Mutexes address this issue with something called *priority inheritance*. 

When a higher-priority task attempts to take a mutex and is blocked, the scheduler will elevate the priority of the task that holds the mutex to the same level as the blocked task. This guarantees that the high-priority task will acquire the mutex and run as soon as possible. 

# Fixing priority inversion

Let's have another try at protecting the shared resource, but this time, we'll use a mutex instead of a semaphore. Using a mutex should help *minimize *priority inversion since it will effectively prevent the mid-priority task from running.

# Setting up the code

There are only two significant differences in this example:

*   We'll use `xSemaphoreCreateMutex()` instead of `xSemaphoreCreateBinarySemaphore()`.
*   No initial `xSemaphoreGive()` call is required since the mutex will be initialized with a value of 1\. Mutexes are designed to be taken only when needed and then given back.

Here's our updated example with the only significant change. This excerpt can be found in `mainMutexExample.c`:

```cpp
mutexPtr = xSemaphoreCreateMutex();
assert_param(mutexPtr != NULL);
```

There are some additional name changes related to the `semPtr` to `mutexPtr` variable name change, but there is nothing functionally different.

# Understanding the behavior

Using Ozone, load `Chapter8_mutexExample.elf` and run the MCU. Here's what to expect when looking at the board:

*   You'll see double blinking green and blue LEDs. The LED blinks of each color will not overlap one another, thanks to the mutex.
*   There will only be a few red LED blips every once in a while. This reduction is caused by `TaskB` not being allowed to take priority over `TaskC` (and blocking `TaskA`). This is a  lot better than before, but why are we still seeing red occasionally?

By opening SystemView, we'll see something like the following:

![](img/8fb2b978-2760-4b89-a8b4-a28c3d4b0b09.png)

Looking through the terminal messages, you'll notice that `TaskA` – the highest-priority task in the system – has never missed a mutex. This is what we expect since it has priority over everything else in the system. Why does `TaskC` occasionally miss a mutex (causing a red LED)?

1.  `TaskC` attempts to take the mutex, but it is being held by `TaskA`.
2.  `TaskA` returns the mutex, but it is immediately taken again. This is caused by a variable amount of delay in `TaskA` between calls to the mutex. When there is no delay, `TaskC` isn't allowed to run between when `TaskA` returns the mutex and attempts to take it again. This is reasonable since `TaskA` has a higher priority (though this might not be desirable in your system).
3.  `TaskC` times out, waiting for the mutex.

So, we've improved our condition. `TaskA`, which is the highest-priority task, isn't missing any mutexes any more. But what are some best practices to follow when using mutexes? Read on to find out.

# Avoiding mutex acquisition failure

While mutexes *help* to provide protection against some priority inversion, we can take an additional step to make sure the mutex doesn't become an unnecessary crutch. The section of code that's protected by the mutex is referred to as a *critical section:*

```cpp
if(xSemaphoreTake(mutexPtr, 200/portTICK_PERIOD_MS) == pdPASS)
{
    //critical section is here
 //KEEP THIS AS SHORT AS POSSIBLE
    xSemaphoreGive(mutexPtr);
}
```

Taking steps to ensure this critical section is as short as possible will help in a few areas:

*   Less time in the critical section makes the shared data more available. The less time a mutex is being held, the more likely it is that another task will gain access in time.
*   Minimizing the amount of time low priority tasks hold mutexes also minimizes the amount of time they spend in an elevated priority (if they have a high priority). 
*   If a low priority task is blocking a higher-priority task from running, the high priority task will have more variability (also known as jitter) in how quickly it is able to react to events.

Avoid the temptation to acquire a mutex at the beginning of a long function. Instead, access data throughout the function and return the mutex before exiting:

```cpp
if(xSemaphoreTake(mutexPtr, 200/portTICK_PERIOD_MS) == pdPASS)
{
    //critical section starts here
    uint32_t aVariable, returnValue;
    aVariable = PerformSomeOperation(someOtherVarNotProtectedbyMutexPtr);
    returnValue = callAnotherFunction(aVariable);

    protectedData = returnValue; //critical section ends here
    xSemaphoreGive(mutexPtr);
}
```

The preceding code can be rewritten to minimize the critical section. This still accomplishes the same goals as providing mutual exclusion for `protectedData`, but the amount of time the mutex is held for is reduced:

```cpp
uint32_t aVariable, returnValue;
aVariable = PerformSomeOperation(someOtherVarNotProtectedbyMutexPtr);
returnValue = callAnotherFunction(aVariable);

if(xSemaphoreTake(mutexPtr, 200/portTICK_PERIOD_MS) == pdPASS)
{
    //critical section starts here
    protectedData = returnValue; //critical section ends here
    xSemaphoreGive(mutexPtr);
}
```

In the preceding examples, there were no `else` statements listed in case the action didn't complete in time. Remember, it is extremely important that the consequences of a missed deadline are understood and that the appropriate action is taken. If you *don't* have a good understanding of the required timing (and the consequences of missing it), then it is time to get the team together for a discussion.

Now that we have a basic understanding of mutexes, we'll take a look at how they can be used to protect data that's being shared across multiple tasks.

# Avoiding race conditions

So, when do we need to use mutexes and semaphores? Any time there is a shared resource between multiple tasks, either a mutex or a semaphore should be used. Standard binary semaphores *can* be used for resource protection, so in some special cases (such as semaphores being accessed from ISRs), semaphores can be desirable. However, you must understand how waiting on the semaphore will affect the system.

We'll see an example of a semaphore being used to protect a shared resource in [Chapter 10](dd741273-db9a-4e9a-a699-b4602e160b84.xhtml), *Drivers and ISRs.*

We saw a mutex in action in the previous example, but what would it look like if there was no mutex and we only wanted one of the blue or green LEDs to be on at a time?

# Failed shared resource example

In our previous mutex example, the LEDs were the shared resource being protected by the mutex. Only one LED was able to blink at a time – either green or blue. It would perform the entire double blink before the next double blink. 

Let's take a look at why this is important with a more realistic example. In the real world, you'll often find shared data structures and hardware peripherals among the most common resources that need to be protected.

Accessing a data structure in an atomic fashion is very important when the structure contains multiple pieces of data that must be correlated with one another. An example would be a multi-axis accelerometer providing three readings for the X, Y, and Z axes. In a high-speed environment, it is important for all three readings to be correlated with one another to accurately determine the device's movement over time:

```cpp
struct AccelReadings
{
    uint16_t X;
    uint16_t Y;
    uint16_t Z;
};
struct AccelReadings sharedData;
```

`Task1()` is responsible for updating the data in the structure:

```cpp
void Task1( void* args)
{
    while(1)
    {
        updateValues();
        sharedData.X = newXValue;
        sharedData.Y = newYValue;
        sharedData.Z = newZValue;
    }
}

```

On the other hand, `Task2()` is responsible for reading the data from the structure:

```cpp
void Task2( void* args)
{
    uint16_t myX, myY, myZ;
    while(1)
    {
        myX = sharedData.X;
        myY = sharedData.Y;
        myZ = sharedData.Z;
        calculation(myX, myY, myZ);
    }
}
```

If one of the readings isn't properly correlated with the others, we'll wind up with an incorrect estimation of the device's movement. `Task1` may be attempting to update all three readings, but in the middle of gaining access, `Task2` comes along and attempts to read the values. As a result, `Task2` receives an incorrect representation of the data because it was in the middle of being updated:

![](img/206b11b9-a073-439f-98d6-f2e84c1c2774.png)

Access to this data structure can be protected by putting all access to the shared data inside a critical section. We can do this by wrapping access in a mutex:

```cpp
void Task1( void* args)
{
    while(1)
    {
        updateValues();
        if(xSemaphoreTake(mutexPtr, timeout) == pdPASS)
        {
            sharedData.X = newXValue;    //critical section start
            sharedData.Y = newYValue;
            sharedData.Z = newZValue;    //critical section end
            xSemaphoreGive(mutexPtr);
        }
        else { /* report failure */}
    }
}
```

It is important to wrap the read accesses as well:

```cpp

void Task2( void* args)
{
    uint16_t myX, myY, myZ;
    while(1)
    {
        if(xSemaphoreTake(mutexPtr, timeout) == pdPASS)
        {
            myX = sharedData.X; //critical section start
            myY = sharedData.Y;
            myZ = sharedData.Z; //critical section end
 xSemaphoreGive(mutexPtr);

            //keep the critical section short
            calculation(myX, myY, myZ);
        }
        else{ /* report failure */ }
    }
}
```

Now that data protection has been covered, we'll take another look at inter-task synchronization. Semaphores were used for this previously, but what if your application calls for actions to occur at a consistent rate? FreeRTOS software timers are one possible solution.

# Using software timers

Just like the name states, software timers are timers that are implemented with software. In MCUs, it is extremely common to have many different hardware peripheral timers available. These are often high resolution and have many different modes and features that are used to offload work from the CPU. However, there are two downsides to hardware timers:

*   Since they are part of the MCU, you'll need to create an abstraction above them to prevent your code from becoming tightly coupled to the underlying MCU hardware. Different MCUs will have slightly different implementations for timers. Because of this, it is easy for code to become dependent on the underlying hardware.
*   They will generally take more development time to set up than using the software-based timer that has already been provided by the RTOS.

Software timers alleviate this coupling by implementing multiple timer channels via software, rather than hardware. So, instead of an application being dependent on specific hardware, it can be used (without modification) on any platform the RTOS supports, which is extremely convenient.

There are techniques we can use to reduce the firmware's tight coupling to the underlying hardware. *[Chapter 12](8e78a49a-1bcd-4cfe-a88f-fb86a821c9c7.xhtml), Tips on Creating Well Abstracted Architecture*, will outline some of the techniques that can be used to eliminate the tight coupling between hardware and firmware.

You may have noticed a task called `TmrSvc` in the SystemView screenshots. This is the software timer service task. Software timers are implemented as a FreeRTOS task, using many of the same underlying primitives that are available. They have a few configuration options, all of which can be set in `FreeRTOSConfig.h`:

```cpp
/* Software timer definitions. */
#define configUSE_TIMERS 1
#define configTIMER_TASK_PRIORITY ( 2 )
#define configTIMER_QUEUE_LENGTH 10
#define configTIMER_TASK_STACK_DEPTH 256
```

In order to have access to software timers, `configUSE_TIMERS` must be defined as `1`. As shown in the preceding snippet, the priority of the timer task, as well as the queue length (number of available timers) and stack depth, can all be configured through `FreeRTOSConfig.h`

*But software timers are a FreeRTOS feature – why do **I** need to worry about stack depth?!*

There's one thing to keep in mind with software timers: *the code that's executed when the timer fires is executed inside the context of the Software Timer Task.* This means two things:

*   Each callback function executes on the `TmrSvc` task's stack. Any RAM (that is, local variables) that's used in the callback will come from the `TmrSvc` task.  
*   Any long actions that are performed will block other software timers from running, so treat the callback function you pass to the software timer similar to the way you would an ISR – don't deliberately delay the task, and keep everything as short as possible.   

The best way to get familiar with software timers is to actually use them in a real system.

# Setting up the code

Let's have a look at a few simple examples to see software timers in action. There are two main ways of using software timers: oneshot and repeat.  We'll cover each with an example.

# Oneshot timers

A *oneshot* is a timer that fires only *one* time. These types of timers are common in both hardware and software and come in very handy when a fixed delay is desired. A oneshot timer can be used when you wish to execute a *short* piece of code after a fixed delay, without blocking the calling code by using `vTaskDelay()`. To set up a oneshot timer, a timer callback must be specified and a timer created.

The following is an excerpt from `mainSoftwareTimers.c`:

1.  Declare a `Timer` callback function that can be passed to `xTimerCreate()`. This callback is executed when the timer fires. Keep in mind that the callback is executed within the timer task, so it needs to be non-blocking!

```cpp
void oneShotCallBack( TimerHandle_t xTimer );
```

2.  Create a timer. Arguments define whether or not the timer is a oneshot or repeating timer (repeating timers *auto-reload* in FreeRTOS).
3.  Perform some due diligence checks to make sure the timer was created successfully by checking that the handle is not `NULL`.
4.  Issue a call to `xTimerStart()` and ensure the `uxAutoReload` flag is set to `false` (again, the prototype for `xTimerCreate()` is as follows):

```cpp
TimerHandle_t xTimerCreate (    const char * const pcTimerName, 
                                const TickType_t xTimerPeriod, 
                                const UBaseType_t uxAutoReload,
                                void * const pvTimerID, 
                                TimerCallbackFunction_t pxCallbackFunction );
```

5.  So, to create a *one-shot* timer, we need to set `uxAutoReload` to `false`:

```cpp
TimerHandle_t oneShotHandle = 
xTimerCreate(   "myOneShotTimer",        //name for timer
                2200/portTICK_PERIOD_MS, //period of timer in ticks
                pdFALSE,                 //auto-reload flag
                NULL,                    //unique ID for timer
                oneShotCallBack);        //callback function
assert_param(oneShotHandle != NULL);     //ensure creation
xTimerStart(oneShotHandle, 0);           //start with scheduler
```

6.  `oneShotCallBack()` will simply turn off the blue LED after 1 second has elapsed:

```cpp

void oneShotCallBack( TimerHandle_t xTimer )
{
    BlueLed.Off();
}
```

Remember that the code that is executing inside the software timer must be kept short. All software timer callbacks are serialized (if one callback performs long operations, it could potentially delay others from executing).

# Repeat timers

Repeat timers are similar to oneshot timers, but instead of getting called only *once*, they get called *repeatedly*. After a repeat timer has been started, its callback will be executed repeatedly every `xTimerPeriod` ticks after being started. Since repeat timers are executed within the `TmrSvc` task, they can provide a lightweight alternative to tasks for short, non-blocking functions that need to be run periodically. The same considerations regarding stack usage and execution time apply to oneshot timers.  

The steps are essentially the same for repeat timers: just set the value of the auto-reload flag to `pdTRUE`.

Let's take a look at the code in `mainSoftwareTimers.c`:

```cpp
TimerHandle_t repeatHandle = 
xTimerCreate(   "myRepeatTimer",         //name for timer
                500 /portTICK_PERIOD_MS, //period of timer in ticks
                pdTRUE,                  //auto-reload flag
                NULL,                    //unique ID for timer
                repeatCallBack);          //callback function
assert_param(repeatHandle != NULL);
xTimerStart(repeatHandle , 0);
```

The repeating timer will toggle the green LED:

```cpp
void repeatCallBack( TimerHandle_t xTimer )
{
    static uint32_t counter = 0;
    if(counter++ % 2)
    {
        GreenLed.On();
    }
    else
    {
        GreenLed.Off();
    }
}
```

In the preceding code, a static variable is used for the `counter` variable so that its value persists across function calls, while still hiding the variable from all the code outside of the `repeatCallBack()` function.

# Understanding the behavior

Upon performing a reset, you'll see the blue LED turn on. To start the FreeRTOS scheduler and the timers, push the blue *USER* button, *B1*, in the lower left of the board. The blue LED will turn off after 2.2 seconds. This only happens once since the blue LED has been set up as a oneshot timer.  The green LED toggles every 500 ms since it was set up with a repeat timer.

Let's take a look at the output of the SystemView terminal. In the terminal, all the times are relative to the start of the RTOS scheduler. The blue LED oneshot is only executed once, 2.2 seconds in, while the green LED is toggled every 500 ms:

![](img/f8ef5719-e49e-42e7-b672-54d0d91c69c3.png)

This same information is also available on the timeline. Note that the times are relative to the cursor on the timeline; they are not absolute like they are in the terminal:

![](img/28b0dad1-ce2d-4327-a51a-1b8c833ba74f.png)

Now that we know how to set up software timers and understand their behavior, let's discuss when they can be used.

# Software timer guidelines

Software times can be really useful, especially since they're so easy to set up. They are also fairly lightweight because of the way they have been coded in FreeRTOS – they don't require significant code or CPU resources when used. 

# Example use cases

Here are some use cases to help you out:

*   To periodically perform an action (auto-reload mode). For example, a timer callback function could give a semaphore to a reporting task to provide periodic updates about the system.
*   To perform an event only once at some point in the future, without blocking the calling task in the meantime (which would be required if `vTaskDelay()` was used instead).

# Considerations

Keep these considerations in mind:

*   The priority of the timer service task can be configured in `FreeRTOSConfig.h` by setting `configTIMER_TASK_PRIORITY`.

*   Timers can be modified after being created, restarted, and deleted.
*   Timers can be created statically (similar to static task creation) to avoid dynamic allocation from the FreeRTOS heap.
*   All callbacks are executed in the Software Timer Service Task  – they must be kept short and not block!

# Limitations

So, what's not to love about software timers? Not too much, as long as the following are kept in mind:

*   **Jitter**: Since the callbacks are executed within the context of a task, their exact execution time will depend on all the interrupts in the system, as well as any higher-priority tasks. FreeRTOS allows this to be tuned by adjusting the priority of the timer task being used (which must be balanced with the responsiveness of other tasks in the system).
*   **Single Priority**: All software timer callbacks execute inside the same task.
*   **Resolution**: A software timer's resolution is only as precise as the FreeRTOS tick rate (defined as 1 ms for most ports).

If lower jitter or higher resolution is required, it probably makes sense to use a hardware timer with ISRs instead of software timers.

# Summary

In this chapter, we covered many different aspects of synchronizing tasks and protecting shared data between tasks. We also covered semaphores, mutexes, and software timers. Then, we got our hands dirty by writing some code for each of these types and took a deep dive into analyzing the code's behavior using our Nucleo development board and SystemView. 

Now, you have some tools at your disposal for solving synchronization problems, such as one task notifying another that an event has occurred (semaphores). This means you're able to safely share data between tasks by properly wrapping access in a mutex. You also know how to save a bit of RAM when performing simple operations, that is, by using software timers for small periodic operations, instead of dedicated tasks.

In the next chapter, we'll cover more crucial RTOS primitives that are used for inter-task communication and provide the foundations for many RTOS-based applications.

# Questions

As we conclude this chapter, here is a list of questions for you to test your knowledge regarding this chapter's material. You will find the answers in the *Assessments* section of the Appendix:

1.  What are semaphores most useful for?
2.  Why is it dangerous to use semaphores for data protection? 
3.  What does mutex stand for? 
4.  Why are mutexes better for protecting shared data? 
5.  With an RTOS, there is no need for any other type of timer since many instances of software timers are available. 
    *   True
    *   False

# Further reading

*   A Microsoft paper that provides more detail on problems with semaphores: [https://www.microsoft.com/en-us/research/publication/implementing-condition-variables-with-semaphores/](https://www.microsoft.com/en-us/research/publication/implementing-condition-variables-with-semaphores/)
*   Phillip Koopman on race conditions: [http://course.ece.cmu.edu/~ece642/lectures/26_raceconditions.pdf](http://course.ece.cmu.edu/~ece642/lectures/26_raceconditions.pdf)