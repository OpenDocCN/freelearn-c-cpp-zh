# Intertask Communication

Now that we're able to create tasks, it's time to start passing data between them. After all, you don't often run into systems that have a bunch of parallel tasks operating completely independently of one another; normally, you will need to pass some data between different tasks in the system. This is where intertask communication comes into play.

In FreeRTOS, intertask communication can be achieved using queues and direct task notifications. In this chapter, we'll cover a few different use cases for queues using examples and discuss the pros and cons of each. We will look at all of the details regarding tasks that block while waiting for an item to appear in the queue, as well as timeouts. Once we have looked at queues, we'll move on to task notifications and learn why we should use them and when.

In a nutshell, we will be covering the following topics:

*   Passing data through queues by value
*   Passing data through queues by reference
*   Direct task notifications

# Technical requirements

To complete the exercises in this chapter, you will require the following:

*   Nucleo F767 development board
*   Micro-USB cable
*   STM32CubeIDE and source code (see the instructions in [Chapter 5](84a945dc-ff6c-4ec8-8b9c-84842db68a85.xhtml), *Selecting an IDE – Setting Up Our IDE*)
*   SEGGER JLink, Ozone, and SystemView (see [Chapter 6](699daa80-06ae-4acc-8b93-a81af2eb774b.xhtml), *Debugging Tools for Real-Time Systems*)

The easiest way to build the examples is to build all Eclipse configurations at once, then load and view them using Ozone:

1.  In *STM32CubeIDE*, right-click on the project.
2.  Select Build.
3.  Select Build All. All examples will be built into their own named subdirectory (this may take a while). 
4.  In Ozone*,* you can now quickly load each `<exampleName>.elf` file—see `Chapter6` for instructions on how to do this.  The correct source files that are linked into the executable will automatically be displayed.

All of the example code in this chapter can be downloaded from [https://https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_9](https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_9). Each `main*.c` has its own Eclipse-based configuration inside the `Chapter_9` project, ready to compile and load onto the Nucleo board.

# Passing data through queues by value

Like semaphores and mutexes, queues are among the most widely used (and implemented) structures when operating across multiple asynchronously executing tasks. They can be found in nearly every operating system, so it is very beneficial to understand how to use them. We'll take a look at several different ways of using queues and interacting with them to affect a task's state. 

In the following examples, we'll learn how to use queues as a means of sending *commands* to an LED state machine. First, we'll examine a very simple use case, passing a single one-byte value to a queue and operating on it.

# Passing one byte by value

In this example, a single `uint8_t` is set up to pass individual enumerations, `(LED_CMDS)`, defining the state of one LED at a time or all of the LEDs (on/off).  Here's a summary of what is covered in this example:

*   `ledCmdQueue`: A queue of one-byte values (`uint8_t`) representing an enumeration defining LED states.
*   `recvTask`: This task receives a byte from the queue, executes the desired action, and immediately attempts to receive the next byte from the queue.

*   `sendingTask`: This task sends enumerated values to the queue using a simple loop, with a 200 ms delay between each send (so the LEDs turning on/off are visible).

So, let's begin:

1.  Set up an `enum` to help us describe the values that are being passed into the queue:

The following is an excerpt from `mainQueueSimplePassByValue.c`:

```cpp
typedef enum
{
  ALL_OFF = 0,
  RED_ON = 1,
  RED_OFF = 2,
  BLUE_ON = 3,
  BLUE_OFF= 4,
  GREEN_ON = 5,
  GREEN_OFF = 6,
  ALL_ON = 7

}LED_CMDS;
```

2.  Similar to the initialization paradigm of semaphores, queues must first be created and their handles stored so they can be used to access the queue later. Define a handle to be used to point at a queue that is to be used for passing around instances of `uint8_t`:

```cpp
static QueueHandle_t ledCmdQueue = NULL;
```

3.  Create the queue (verifying its successful creation before continuing) using the `xQueueCreate()` function:

```cpp
QueueHandle_t xQueueCreate( UBaseType_t uxQueueLength,
                             UBaseType_t uxItemSize );
```

Let's quickly outline what we see here:

*   `uxQueueLength`: The maximum number of elements the queue can hold
*   `uxItemSize`: The size (in bytes) of each element in the queue
*   Return value: A handle to the queue that is created (or `NULL` upon error)

Our call to `xQueueCreate` will look like this:

```cpp
ledCmdQueue = xQueueCreate(2, sizeof(uint8_t));
assert_param(ledCmdQueue != NULL);
```

Let's outline what we see here:

*   The queue holds up to `2` elements.
*   Each element is sized to hold `uint8_t` (a single byte is large enough to store the value of any enumeration we have explicitly defined).
*   `xQueueCreate` returns a handle to the queue created, which is stored in `ledCmdQueue`.  This "handle" is a global that will be used by various tasks when accessing the queue.

The beginning of our `recvTask()` looks like this:

```cpp
void recvTask( void* NotUsed )
{
  uint8_t nextCmd = 0;

  while(1)
  {
    if(xQueueReceive(ledCmdQueue, &nextCmd, portMAX_DELAY) == pdTRUE)
    {
      switch(nextCmd)
      {
        case ALL_OFF:
          RedLed.Off();
          GreenLed.Off();
          BlueLed.Off();
        break;
        case GREEN_ON:
          GreenLed.On();
        break;
```

Let's have a close look at the actual queue receive line highlighted in the preceding code:

`if(xQueueReceive(ledCmdQueue, &nextCmd, portMAX_DELAY) == pdTRUE)`

*   The handle `ledCmdQueue` is used to access the queue.
*   A local `uint8_t`, `nextCmd`, is defined on the stack. The address of this variable (a pointer) is passed. `xQueueReceive` will copy the next `LED_CMD` enumeration (stored as a byte in the queue) into `nextCmd`.

*   An infinite timeout is used for this access—that is, this function will never return if nothing is added to the queue (the same as timeouts for mutex and semaphore API calls).

The `if( <...> == pdTRUE)` is redundant since the delay time is infinite; however, it is a good idea to set up error handling ahead of time so that if a noninfinite timeout is later defined, the error state won't be forgotten about down the road. It is also possible for `xQueueReceive()` to fail for other reasons (such as an invalid queue handle).

The `sendingTask` is a simple `while` loop that uses prior knowledge of the enum values to pass different values of `LED_CMDS` into `ledCmdQueue`:

```cpp
void sendingTask( void* NotUsed )
{
  while(1)
  {
    for(int i = 0; i < 8; i++)
    {
      uint8_t ledCmd = (LED_CMDS) i;
 xQueueSend(ledCmdQueue, &ledCmd, portMAX_DELAY);
      vTaskDelay(200/portTICK_PERIOD_MS);
    }
  }
}
```

The arguments for the sending side's `xQueueSend()` are nearly identical to the receiving side's `xQueueReceive()`, the only difference being that we're sending data *to* the queue this time:

`xQueueSend(ledCmdQueue, &ledCmd, portMAX_DELAY);`

*   `ledCmdQueue`: The handle for the queue to send the data to
*   `&ledCmd`: The address of the data to pass to the queue
*   `portMax_DELAY`: The number of RTOS ticks to wait for the queue space to become available (if the queue is full)

Similar to timeouts from `xQueueReceive` when nothing is in the queue before the timeout value is reached, calls to `xQueueSend` can time out if the queue remains full beyond the specified timeout and the item isn't added to the queue. If your application has a noninfinite timeout (which in nearly all cases it should), you'll need to consider what should happen in this case. Courses of action could range from simply dropping the data item (it will be lost forever) to throwing an assert and going into some type of emergency/panic state with an emergency shutdown. A reboot is also popular in some contexts. The exact behavior will generally be dictated by the type of project/product you're working on.

Feel free to build and download `queueSimplePassByValue` to the Nucleo dev board. You'll notice that the LEDs follow the pattern defined by the definition of the `LED_CMDS` enum: `ALL_OFF`, `RED_ON`, `RED_OFF`, `BLUE_ON`, `BLUE_OFF`, `GREEN_ON`, `GREEN_OFF`, `ALL_ON`, with 200 ms between each transition.

But what if we decide we'd like to operate on more than one LED at a time? We *could* add more values to the existing `LED_CMDS` enum, such as `RED_ON_BLUE_ON_GREEN_OFF`, but that would be a lot of very error-prone typing, especially if we had more than 3 LEDs (8 LEDs results in 256 enum values to cover all combinations of each LED being on/off). Instead, let's look at how we can use a struct to describe the LED command and pass that through our queue.

# Passing a composite data type by value

FreeRTOS queues (and most other FreeRTOS API functions) take in `void*` as arguments for the individual data types that are being operated on. This is done to provide flexibility for the application writer as efficiently as possible. Since `void*` is simply a pointer to *anything* and the sizes of the elements in the queue is defined when it is created, queues can be used to pass anything between tasks.

The use of `void*` for interacting with queues acts as a double-edged sword. It provides the ultimate amount of flexibility, but also provides the very real possibility for you to pass the *wrong* data type into the queue, potentially without a warning from the compiler. You must keep track of the data type that is being stored in each queue!

We'll use this flexibility to pass in a composite data type created from a struct of instances of `uint8_t` (each of which is only one bit wide) to describe the state of all three LEDs:

Excerpt from `mainQueueCompositePassByValue.c`:

```cpp
typedef struct
{
  uint8_t redLEDState : 1; //specify this variable as 1 bit wide 
  uint8_t blueLEDState : 1; //specify this variable as 1 bit wide 
  uint8_t greenLEDState : 1; //specify this variable as 1 bit wide 
  uint32_t msDelayTime; //min number of mS to remain in this state
}LedStates_t;
```

We'll also create a queue that is able to hold eight copies of the entire `LedStates_t` struct:

`ledCmdQueue = **xQueueCreate(8, sizeof(LedStates_t)**);`

Like the last example, `recvTask` waits until an item is available from the `ledCmdQueue` queue and then operates on it (turning LEDs on/off as required):

`mainQueueCompositePassByValue.c recvTask`:

```cpp
if(xQueueReceive(ledCmdQueue, &nextCmd, portMAX_DELAY) == pdTRUE)
{
    if(nextCmd.redLEDState == 1)
        RedLed.On();
    else
        RedLed.Off();
    if(nextCmd.blueLEDState == 1)
        BlueLed.On();
    else
        BlueLed.Off();
    if(nextCmd.greenLEDState == 1)
        GreenLed.On();
    else
        GreenLed.Off();
}
vTaskDelay(nextCmd.msDelayTime/portTICK_PERIOD_MS);
```

Here are the responsibilities of the primary loop of `recvTask`:

*   Each time an element is available from the queue, each field of the struct is evaluated and the appropriate action is taken.  All three LEDs are updated with a single command, sent to the queue. 
*   The newly created `msDelayTime` field is also evaluated (it is used to add a delay before the task attempts to receive from the queue again). This is what slows down the system enough so that the LED states are visible.

`mainQueueCompositePassByValue.c sendingTask`:

```cpp
while(1)
  {
    nextStates.redLEDState = 1;
    nextStates.greenLEDState = 1;
    nextStates.blueLEDState = 1;
    nextStates.msDelayTime = 100;

    xQueueSend(ledCmdQueue, &nextStates, portMAX_DELAY);

    nextStates.blueLEDState = 0; //turn off just the blue LED
    nextStates.msDelayTime = 1500;
    xQueueSend(ledCmdQueue, &nextStates, portMAX_DELAY);

    nextStates.greenLEDState = 0;//turn off just the green LED
    nextStates.msDelayTime = 200;
    xQueueSend(ledCmdQueue, &nextStates, portMAX_DELAY);

    nextStates.redLEDState = 0;
    xQueueSend(ledCmdQueue, &nextStates, portMAX_DELAY);
  }
```

The loop of `sendingTask` sends a few commands to `ledCmdQueue` – here are the details:

*   `sendingTask` looks a bit different from before. Now, since a struct is being passed, we can access each field, setting multiple fields before sending `nextStates` to the queue.
*   Each time `xQueueSend` is called, the contents of `nextStates` is copied into the queue before moving on. As soon as `xQueueSend()` returns successfully, the value of `nextStates` is copied into the queue storage; `nextStates` does not need to be preserved.

To drive home the point that the value of `nextStates` is copied into the queue, this example changes the priorities of tasks so that the queue is filled completely by `sendingTask` before being emptied by `recvTask`. This is accomplished by giving `sendingTask` a higher priority than `revcTask`. Here's what our task definitions look like (`asserts` are present in the code but are not shown here to reduce clutter):

```cpp
xTaskCreate(recvTask, "recvTask", STACK_SIZE, NULL, tskIDLE_PRIORITY + 1, 
            NULL);
xTaskCreate(sendingTask, "sendingTask", STACK_SIZE, NULL, 
            configMAX_PRIORITIES – 1, NULL);

```

`sendingTask` is configured to have the highest priority in the system. `configMAX_PRIORITIES` is defined in `Chapter9/Inc/FreeRTOSConfig.h` and is the number of priorities available. FreeRTOS task priorities are set up so that `0` is the lowest priority task in the system and the highest priority available in the system is `configMAX_PRIORITIES - 1`.  

This prioritization setup allows `sendingTask` to repeatedly send data to the queue until it is full (because `sendingTask `has a higher priority). After the queue has filled, `sendingTask` will block and allow `recvTask` to remove an item from the queue.  Let's take a look at how this plays out in more detail.

# Understanding how queues affect execution

Task priorities work in conjunction with primitives such as queues to define the system's behavior. This is especially critical in a preemptive RTOS application because context is always given based on priority. Programmed queue interactions need to take into account task priorities in order to achieve the desired operation. Priorities need to be carefully chosen to work in conjunction with the design of individual tasks.

In this example, an infinite wait time was chosen for `sendingTask` so that it could fill the queue.

Here's a diagram depicting the preceding setup in action:

![](img/c0b0fd48-7b98-4823-a855-827f6d308d16.png)

Take a look at this example using Ozone to step through the code and understand its behavior. We can go through a few iterations of the `sendingTask while` loop step by step, watching the `ledCmdQueue` data structure and breakpoint setup in each of these tasks:

1.  Make sure you have built the `queueCompositePassByValue` configuration.
2.  Open Ozone by double-clicking Chapter_9\Chapter_9.jdebug.
3.  Go to File | Open | Chapter_9\queueCompositePassByValue\Chapter9_QueuePassCompositeByValue.elf.
4.  Open the global variables view and observe `ledCmdQueue` as you step through the code.
5.  Put a breakpoint in `recvTask` to stop the debugger whenever an item is removed from the queue.
6.  When `recvTask` runs the first time, you'll notice that `uxMessagesWaiting` will have a value of `8` (the queue is filled):

![](img/ce2e3d08-4535-4a07-a44a-fa292f887793.png)

Getting comfortable with whatever debugger you are using *before* you run into serious problems is always a good idea. A second-nature level of familiarity frees your mind to focus on the problem at hand rather than the tools being used.

# Important notes on the examples 

The previous example's main purpose was to illustrate the following points:

*   Queues can be used to hold arbitrary data.
*   Queues interact with task priorities in interesting ways.

There were several trade-offs made to simplify behavior and make the example easier to understand:

*   **The task receiving from the queue was a low priority**: In practice, you'll need to balance the priority of tasks that are receiving from queues (to keep latency low and prevent queues from filling up) against the priority of other events in the system.
*   **A long queue was used for commands**: Deep queues combined with a low-priority task receiving from them will create latencyin a system. Because of the combination of low task priority and long queue length, this example contains several seconds worth of queued commands. Elements were added onto the queue that wouldn't be executed until several seconds after they were added because of the depth/priority combination.
*   **An infinite timeout was used when sending items to the queue**: This will cause `sendTask()` to wait indefinitely for a slot to become available. In this case, this was the behavior we wanted (for simplicity), but in an actual time-critical system, you'll need to keep in mind exactly how long a task is able to wait before an error occurs.

We're not quite done exploring the flexibility of queues. Next, we'll take a look at a special case of passing data by reference to a queue.

# Passing data through queues by reference

Since the data type of a queue is arbitrary, we also have the ability to pass data by reference instead of by value. This works in a similar way to passing arguments to a function by reference.

# When to pass by reference

Since a queue will make a copy of whatever it is holding, if the data structure being queued is large, it will be inefficient to pass it around by value:

*   Sending and receiving from queues forces a copy of the queue element each time.
*   The resulting queue gets very large for large data items if large structures are queued.

So, when there are large items that need to be queued, passing the items by reference is a good idea. Here's an example of a larger structure. After the compiler pads out this struct, it ends up being 264 bytes in size:

`mainQueueCompositePassByReference.c`:

```cpp
#define MAX_MSG_LEN 256
typedef struct
{
  uint32_t redLEDState : 1;
  uint32_t blueLEDState : 1;
  uint32_t greenLEDState : 1;
  uint32_t msDelayTime; //min number of mS to remain in this state
  //an array for storing strings of up to 256 char
  char message[MAX_MSG_LEN];
}LedStates_t;
```

Rather than copy 264 bytes every time an item is added or removed from `ledCmdQueue`, we can define `ledCmdQueue` to hold a pointer (4 bytes on Cortex-M) to `LedStates_t` :

`ledCmdQueue = xQueueCreate(8, **sizeof(LedStates_t*****)**);`

Let's look at the difference between passing by value and passing by reference:

**Passing by value:**

*   `ledCmdQueue` **size:** ~ 2 KB  (264 bytes * 8 elements).
*   264 bytes copied each time `xQueueSend()` or `xQueueReceive()` is called.
*   The original copy of `LedStates_t` that is added to queue can be discarded immediately (a full copy is present inside the queue).

**Passing by reference:**

*   `ledCmdQueue` **size:** 32 bytes (4 bytes * 8 elements).
*   4 bytes copied (the size of a pointer) each time `xQueueSend()` or `xQueueReceive()` is called.
*   The original copy of `LedStates_t` that is added to the queue *must be kept* until it is no longer needed (this is the only copy in the system; only a pointer to the original structure was queued).

When passing by reference, we're making a trade-off between increased efficiency, (potentially) reduced RAM consumption, and more complex code. The extra complexity comes from ensuring that the original value remains valid the entire time it is needed. This approach is very similar to passing references to structures as parameters to functions.

A few instances of `LedStates_t` can be created as well:

```cpp
static LedStates_t ledState1 = {1, 0, 0, 1000, 
  "The quick brown fox jumped over the lazy dog.
  The Red LED is on."};
static LedStates_t ledState2 = {0, 1, 0, 1000,
  "Another string. The Blue LED is on"};
```

Using Ozone, we can easily look at what we've created:

1.  `uxItemSize` of `ledCmdQeue` is 4 bytes, exactly as we would expect, because the queue is holding pointers to `LedStates_t`.

2.  The actual sizes of `ledState1` and `ledState2` are both 264 bytes, as expected:

![](img/cdf8b1af-61d8-460b-8a37-a1b9fe848914.png)

To send an item to the queue, go through the following steps:

1.  Create a pointer to the variable and pass in the address to the pointer:

```cpp
void sendingTask( void* NotUsed )
{
  LedStates_t* state1Ptr = &ledState1;
  LedStates_t* state2Ptr = &ledState2;

  while(1)
  {
    xQueueSend(ledCmdQueue, &state1Ptr, portMAX_DELAY);
    xQueueSend(ledCmdQueue, &state2Ptr, portMAX_DELAY);
  }
}
```

2.  To receive items from the works, simply define a pointer of the correct data type and pass the address to the pointer:

```cpp
void recvTask( void* NotUsed )
{
    LedStates_t* nextCmd;

    while(1)
    {
        if(xQueueReceive(ledCmdQueue, &nextCmd, portMAX_DELAY) == 
                                                           pdTRUE)
        {
            if(nextCmd->redLEDState == 1)
            RedLed.On();
```

When operating on an item taken out of the queue, remember that you've got a pointer that needs to be dereferenced (that is, `nextCmd->redLEDState`).

Now for the catch(es)...

# Important notes

Passing by reference can be more efficient than passing by value for moving large data structures around, but several things need  to be kept in mind:

*   **Keep the datatypes straight**:Because the argument to a queue is of the `void*` data type, the compiler won't be able to warn you that you're supplying an address to a struct instead of to a pointer.

*   **Keep the queued data around**: Unlike passing data by value, when a queue holds pointers to the data, the underlying data passed to the queue needs to stay until it is used. This has the following implications:
    *   The data must not live on the stack—no local function variables! Although this *can* be made to work, it is generally a bad idea to define variables on a stack in the *middle* of a call chain and then push a pointer onto a queue. By the time the receiving task pulls the pointer off of the queue, the stack of the sending task is likely to have changed. Even if you do get this to work under some circumstances (such as when the receiving task has a higher priority than the sending task), you'll have created a very brittle system that is likely to break in a very subtle way in the future.
    *   A stable storage location for the underlying variable is a must. Global and statically allocated variables are both acceptable.  If you'd like to limit access to a variable, use static allocation inside a function.  This will keep the variable in memory, just as if it was a global, but limit access to it:

```cpp
void func( void ) 
{
    static struct MyBigStruct myVar;
```

*   *   You should dynamically allocate space for the variable (if dynamic allocation is acceptable in your application).  See [Chapter 15](0f98e454-9804-4589-9854-5c38c9d8d416.xhtml)*, FreeRTOS Memory Management*, for details on memory management, including dynamic allocation.
*   **Who owns the data? **When a queue has a copy of a struct, the queue owns that copy. As soon as the item is removed from the queue, it disappears. Contrast this with a queue holding a *pointer* to data. When the pointer is removed from the queue, the data is still present in its previous location. Data ownership needs to be made very clear. Will the task receiving the pointer from the queue become the new owner (and be responsible for freeing dynamically allocated memory if it was used)? Will the original task that sent the pointer still maintain ownership? These are all important questions to consider up front.

Now that we've discussed passing around huge amounts of data (avoid it whenever possible!), let's talk about an efficient way of passing around small amounts of data.

# Direct task notifications

Queues are an excellent workhorse of an RTOS because of their flexibility. Sometimes, all of this flexibility isn't needed and we'd prefer a more lightweight alternative. Direct task notifications are similar to the other communication mechanisms discussed, except that they do not require the communication object to first be instantiated in RAM. They are also faster than semaphores or queues (between 35% and 45% faster).

They do have some limitations, the largest two being that only one task can be notified at a time and notifications can be sent by ISRs but not received.

Direct task notifications have two main components: the notification itself (which behaves very much like how a semaphore or queue behaves when unblocking a task) and a 32-bit notification value. The notification value is optional and has a few different uses. A notifier has the option of overwriting the entire value or using the notification value as if it were a bitfield and setting a single bit. Setting individual bits can come in handy for signaling different behaviors that you'd like the task to be made aware of without resorting to a more complicated command-driven implementation based on queues.

Take our LEDs, for example. If we wanted to create a simple LED handler that quickly responded to a change request, a multi-element queue wouldn't be necessary; we can make use of the built-in 32-bit wide notification value instead.

If you're thinking *task notifications sound a bit like semaphores*, you'd be right! Task notifications can also be used as a faster alternative to semaphores.

Let's see how task notifications can be utilized to issue commands and pass information to a task by working through an example. 

# Passing simple data using task notifications

In this example, our goal is to have `recvTask` set LED states, which it has been doing this entire chapter. This time, instead of allowing multiple copies of future LED states to pile up and execute some time in the future, `recvTask` will execute just one state change at a time.

Since the notification value is built into the task, no additional queue needs to be created—we just need to make sure that we store the task handle of `recvTask`, which will be used when we send it notifications.

Let's look at how we do this by looking at some `mainTaskNotifications.c` excerpts***:***

1.  Outside of `main`, we'll define some bitmasks and a task handle:

```cpp
#define RED_LED_MASK 0x0001
#define BLUE_LED_MASK 0x0002
#define GREEN_LED_MASK 0x0004
static xTaskHandle recvTaskHandle = NULL; 
```

2.  Inside `main`, we'll create the `recvTask` and pass it the handle to populate:

```cpp
retVal = xTaskCreate(recvTask, "recvTask", STACK_SIZE, NULL, 
                     tskIDLE_PRIORITY + 2, &recvTaskHandle);
assert_param( retVal == pdPASS);
assert_param(recvTaskHandle != NULL);
```

3.  The task receiving the notification is set up to wait on the next incoming notification and then evaluate each LED's mask, turning LEDs on/off accordingly:

```cpp
void recvTask( void* NotUsed )
{
    while(1)
    {
        uint32_t notificationvalue = ulTaskNotifyTake( pdTRUE, 
                                              portMAX_DELAY );
        if((notificationvalue & RED_LED_MASK) != 0)
            RedLed.On();
        else
            RedLed.Off();
```

4.  The sending task is set up to send a notification, overwriting any existing notifications that may be present. This results in `xTaskNotify` always returning `pdTRUE`:

```cpp
void sendingTask( void* NotUsed )
{
    while(1)
    {
 xTaskNotify( recvTaskHandle, RED_LED_MASK,
 eSetValueWithOverwrite);
        vTaskDelay(200);
```

This example can be built using the `directTaskNotification` configuration and uploaded to the Nucleo. It will sequentially blink each LED as the notifications are sent to `recvTask`.

# Other options for task notifications

From the `eNotifyAction` enumeration in `task.h`, we can see that the other options for notifying the task include the following:

```cpp
eNoAction = 0, /* Notify the task without updating its notify value. */
 eSetBits, /* Set bits in the task's notification value. */
 eIncrement, /* Increment the task's notification value. */
 eSetValueWithOverwrite, /* Set the task's notification value to a specific value even if the previous value has not yet been read by the task. */
 eSetValueWithoutOverwrite /* Set the task's notification value if the previous value has been read by the task. */
```

Using these options creates some additional flexibility, such as using notifications as binary and counting semaphores. Note that some of these options change how `xTaskNotify` returns, so it will be necessary to check the return value in some cases.

# Comparing direct task notifications  to queues

Compared to queues, task notifications have the following features:

*   They always have storage capacity of exactly one 32-bit integer.
*   They do not offer a means of waiting to push a notification to a busy task; it will either overwrite an existing notification or return immediately without writing.
*   They can only be used with only one receiver (since the notification value is stored inside the receiving task).
*   They are faster.

Let's take a look at a real-world example using SystemView to compare the direct notification code we just wrote against the first queue implementation.

The queue implementation from `mainQueueSimplePassByValue.c` looks like this when performing `xQueueSend`:

![](img/48be5eee-d3e5-4efe-a3a7-55bc0053821e.png)

The direct task notification looks like this when calling `xTaskNotify`:

![](img/0ceaaee1-65e4-454e-9b56-25f5b965ec78.png)

As we can see from the preceding screenshots, the direct task notification is in the range of 25–35% faster than using a queue in this particular use case. There's also no RAM overhead for storing a queue when using direct task notifications.

# Summary

You've now learned the basics of how to use queues in a variety of scenarios, such as passing simple and composite elements by value and reference. You're aware of the pros and cons of using queues to store references to objects and when it is appropriate to use this method. We also covered some of the detailed interactions between queues, tasks, and task priorities. We finished with a simple real-world example of how to use task notifications to efficiently drive a small state machine.

As you become more accustomed to using RTOSes to solve a wide variety of problems, you'll find new and creative ways of using queues and task notifications. Tasks, queues, semaphores, and mutexes are truly the building blocks of RTOS-based applications and will help you go a long way.

We're not completely done with any of these elements yet, though—there's still a lot of more advanced material to cover related to using all of these primitives in the context of ISRs, which is up next!

# Questions

As we conclude, here is a list of questions for you to test your knowledge regarding this chapter's material. You will find the answers in the *Assessments* section of the *Appendix*:

1.  What data types can be passed to queues?
2.  What happens to the task attempting to operate on the queue while it is waiting?
3.  Name one consideration that needs to be made when passing by reference to a queue?
4.  Task notifications can completely replace queues:
    *   True
    *   False
5.  Task notifications can send data of any type:
    *   True
    *   False
6.  What are the advantages of task notifications over queues?

# Further reading

*   **Explanation of all constants of **`FreeRTOSConfig.h`**:** [https://www.freertos.org/a00110.html](https://www.freertos.org/a00110.html)
*   **FreeRTOS direct task notifications:** [https://www.freertos.org/RTOS-task-notifications.html](https://www.freertos.org/RTOS-task-notifications.html)