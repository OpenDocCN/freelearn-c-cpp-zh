# Creating Loose Coupling with Queues

Now that we've covered ways of architecting source code for flexibility, we'll take that a step further and explore how queues can be used to provide natural interface definitions for data exchange. 

In this chapter, we'll develop a simple command queue that can be accessed from multiple physical interfaces. By the end of this chapter, you'll have an excellent understanding of why using common queue definitions is desirable, as well as how to implement both sides of an extremely flexible command queue. This will help you create flexible architectures with implementations that aren't tied to underlying hardware or physical interfaces.

We will cover the following topics in this chapter:

*   Understanding queues as interfaces
*   Creating a command queue
*   Reusing a queue definition for a new target

# Technical requirements

To complete the hands-on experiments included in this chapter, you will require the following:

*   A Nucleo F767 development board
*   A Micro-USB cable
*   STM32CubeIDE and source code (for instructions, visit the *Setting up our IDE* section in [Chapter 5](84a945dc-ff6c-4ec8-8b9c-84842db68a85.xhtml)*, Selecting an IDE*)
*   SEGGER J-Link, Ozone, and SystemView (for instructions, visit [Chapter 6](699daa80-06ae-4acc-8b93-a81af2eb774b.xhtml), *Debugging Tools for Real-Time Systems*)
*   Python >= 3.8

All source code for this chapter is available at [https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_13](https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_13).

# Understanding queues as interfaces

If you've just finished reading the previous chapter, you've likely picked up on the fact that there are many techniques that can be used to create quality code at one time and then reuse that same code across multiple projects. Just like using layers of abstractions is a technique that increases the likelihood of reusing code across multiple hardware platforms, using queues as interfaces also increases the likelihood that code will be used for more than just one project. 

The concepts presented in this chapter aren't limited to queues—they also apply to stream and message buffers. However, since queues have been around since the beginning of FreeRTOS (and are the most commonly available primitive), we'll use them in our examples. Let's take a look at why using queues is a good idea.

# Queues make excellent interface definitions

Queues provide a very clear line of abstraction. In order to pass data through a queue and get a desired behavior on the other side, all of the data must be present and both the sender and receiver must have a common understanding of what the data format is. This clean line forces a level of conscientious thought as to what exactly needs to be communicated. Sometimes, this level of active thought isn't present when implementing individual functions. The delineation provided by the queue forces additional thought about what the exact information required is, as well as what its format should be. Responsible developers will be more likely to ensure these types of definitive interfaces are thoroughly documented.

When a queue is viewed as an interface to a subsystem, it pays dividends to document the functionality that will be provided, as well as the exact formatting required to use the subsystem. Often, the fact that an interface is well defined will increase the likelihood of reuse since it will be easily understood.

# Queues increase flexibility 

Queues are beautiful in their simplicity—a sender places something in the queue and whatever task is monitoring the queue will receive the data and act on it. The only things the sender and the received task need to share are the code required for interacting with the queue and the definition of the data flowing through the queue. Since the list of shared resources is so short, there is a natural decoupling effect when queues are used.

Because of the clean break provided by the queue, the exact implementation of the functionality could change over time. The same functionality can be implemented in many different ways, which won't immediately affect the sender, as long as the queue interface doesn't change.

It also means that data can be sent to the queue from anywhere; there are no explicit requirements for the physical interface—only the data format. This means that a queue can be designed to receive the same data stream from many different interfaces, which can provide system-level flexibility. The functionality doesn't need to be tied to a specific physical interface (such as Ethernet, USB, UART, SPI, CAN, and so on).

# Queues make testing easier

Similar to the way hardware abstractions provide easy insertion points for test data, queues also provide excellent places to enter test data. This provides a very convenient entry point for entering test data for code under development. The flexibility of implementation mentioned in the previous section also applies here. If a piece of code is sending data to a queue and expects a response from another queue, the actual implementation doesn't necessarily need to be used—it can be simulated by responding to the command. This approach makes it possible to develop the other side of code in the absence of fully implemented functionality (in case the hardware or subsystem is still under development). This approach is also extremely useful when running unit-level tests; the code-under-test can be easily isolated from the rest of the system.

Now that we've covered some of the reasons to use queues as interfaces, let's take a look at how this plays out through an example.

# Creating a command queue

To see how a queue can be used to keep an architecture loosely coupled, we'll take a look at an application that accepts commands over USB and lights LEDs. While the example application itself is very simple, the concepts presented here scale extremely well. So, regardless of whether there are only a few commands or hundreds, the same approach can be used to keep the architecture flexible.

This application also shows another example of how to keep higher-level code loosely coupled to the underlying hardware. It ensures the LED command code only uses a defined interface to access a **Pulse Width Modulation** (**PWM**) implementation, rather than directly interacting with the MCU registers/HAL. The architecture consists of the following major components:

*   **A USB driver**: This is the same USB stack that has been used in previous examples. `VirtualCommDriverMultiTask.c/h` has been extended to provide an additional stream buffer to efficiently receive data from a PC (`Drivers/HandsOnRTOS/VirtualCommDriverMultiTask.c/h`).
*   **iPWM**: An additional interface definition (`iPWM`) has been created to describe very simple PWM functionality (defined in `Chapter_13/Inc/iPWM.h`).
*   **PWM implementation**: The implementation of three `iPWM` interfaces for the Nucleo hardware is found in `Chapter13/Src/pwmImplementation.c` and `Chapter13/Src/pwmImplementation.h`.  
*   **An LED command executor**: The state machine that drives LED states using pointers to implementations of `iPWM` (`Chapter_13/Src/ledCmdExecutor.c`).
*   `main`: The `main` function, which ties all of the queues, drivers, and interfaces together and kicks off the FreeRTOS scheduler (`Chapter_13/Src/mainColorSelector.c`). 

We'll get into the details of exactly how all of these parts fit together and the details of their responsibilities; but first, let's discuss what is going to be placed in the command queue.

# Deciding on queue contents

When using queues as a way of passing commands to different parts of the system, it is important to think about what the queue should actually hold, instead of what might be coming *across the wire* in a physical sense. Even though a queue might be used to hold payloads from a datastream with header and footer information, the actual contents of the queue will usually only contain the parsed payload, rather than the entire message.

Using this approach allows more flexibility in the future to retarget the queue to work over other physical layers.

Since `LedCmdExecution()` will be operating primarily on the `iPWM` pointers to interface with the LEDs, it is convenient for the queue to hold a data type that can be used directly by `iPWM`.   

The `iPWM` definition from `Chapter13/Inc/iPWM.h` is as follows:

```cpp
 typedef void (*iPwmDutyCycleFunc)( float DutyCycle );

 typedef struct
 {
   const iPwmDutyCycleFunc SetDutyCycle;
 }iPWM; 
```

This struct only (currently) consists of a single function pointer: `iPwmDutyCycleFunc`.  `iPwmDutyCycleFunc` is defined as a constant pointer—after the initialization of the `iPWM` struct, the pointer can never be changed. This helps guarantee the pointer won't be overwritten, so constantly checking to ensure it isn't `NULL` won't be necessary. 

Wrapping the function pointer in a struct such as `iPWM` provides the flexibility of adding additional functions while keeping refactoring to a minimum. We'll be able to pass a single pointer to the `iPWM` struct to functions, rather than individual function pointers.

If you are creating an *interface* definition that will be shared with other developers, it is important to be very careful to coordinate and communicate changes among your team!

The `DutyCycle` argument is defined as `float`, which makes it easy to keep the interface consistent when interfacing with hardware that has different underlying resolutions. In our implementation, the MCU's timer (`TIM`) peripherals will be configured to have a 16-bit resolution, but the actual code interfacing to `iPWM` doesn't need to be concerned with the available resolution; it can simply map the desired output from `0.00` (off)  to `100.00` (on).

For most applications, `int32_t` would have been preferred over `float` since it has a consistent representation and is easier to serialize. `float` is used here to make it easier to see the differences in the data model versus communication. Also, most people tend to think of PWM as a percentage, which maps naturally to `float`. 

There are two main considerations when deciding on what data `LedCmd` contains: 

*   `ledCmdExecutor` will be dealing with `iPWM` directly, so it makes sense to store floats in `LedCmd`.   
*   We'd also like our LED controller to have different modes of operation, so it will also need a way of passing that information. We'll only have a handful of commands here, so a `uint8_t` 8-bit unsigned integer is a good fit. Each `cmdNum` case will be represented by `enum` (shown later).

This results in the following structure for `LedCmd`:

```cpp
typedef struct
{
  uint8_t cmdNum;
  float red;
  float green;
  float blue;
}LedCmd;
```

The **LED Cmd Executor**'s primary interface will be a queue of `LedCmd`s. State changes will be performed by writing new values in the queue.

Since this structure is only 13 bytes, we'll simply pass it by value. Passing by reference (a pointer to the structure) would be faster, but it also complicates the ownership of the data. These trade-offs are discussed in [Chapter 9](495bdcc0-2a86-4b22-9628-4c347e67e49e.xhtml), *Intertask Communication*.

Now that we have a data model defined, we can look at the remaining components of this application.

# Defining the architecture

The command executor architecture is composed of three primary blocks; each block executes asynchronously to the others and communicates via a queue and stream buffer:

*   **LED Cmd Executor**: `LedCmdExecution` in `ledCmdExecutor.c`receives data from `ledCmdQueue` and actuates the LEDs via pointers to `iPWM` (one for each color). `LedCmdExecution` is a FreeRTOS task that takes `CmdExecArgs` as an argument upon creation.
*   **Frame protocol decoding**: `mainColorSelector.c` receives raw data from the stream buffer populated by the USB virtual comm driver, ensures valid framing, and populates the `LedCmd` queue.
*   **The USB virtual comm driver**:The USB stack is spread across many files; the primary user-entry point is `VirtualCommDriverMultiTask.c`. 

Here's a visual representation of how all of these major components stack up and flow together. Major blocks are listed on the left, while representations of the data they operate on are to the right:

![](img/eb1d337c-df75-4ffe-a67f-2b97b6f97078.png)

Let's take a closer look at each one of these components.

# ledCmdExecutor

`ledCmdExecutor.c`implements a simple state machine, whose state is modified when a command is received from the queue.

The available commands are explicitly enumerated by `LED_CMD_NUM`. Each *command* has been given a human-friendly enumeration, along with an explicit definition. The enums are explicitly defined so they can be properly enumerated on the PC side. We also need to make sure the numbers assigned are <= 255, since we'll only be allocating 1 byte in the frame to the command number:

```cpp
typedef enum
{
  CMD_ALL_OFF = 0,
  CMD_ALL_ON = 1,
  CMD_SET_INTENSITY = 2,
  CMD_BLINK = 3
}LED_CMD_NUM;
```

The only public function is `LedCmdExecution`, which will be used as a FreeRTOS task: `void LedCmdExecution(void* Args)`.

`void* Args` actually has a type of `CmdExecArgs`. However, the function signature for a FreeRTOS task requires a single parameter of `void*`. The actual data type being passed into `LedCmdExecution` is a pointer to this struct:

```cpp
typedef struct
{
  QueueHandle_t ledCmdQueue; 
  iPWM * redPWM;
  iPWM * bluePWM;
  iPWM * greenPWM;
}CmdExecArgs;
```

Passing in references to everything allows multiple instances of the task to be created and run simultaneously. It also provides extremely loose coupling to the underlying `iPWM` implementations.

`LedCmdExecution` has a few local variables to track state:

```cpp
LED_CMD_NUM currCmdNum = CMD_ALL_OFF;
bool ledsOn = false;
LedCmd nextLedCmd;
param_assert(Args == NULL);
CmdExecArgs args = *(CmdExecArgs*)Args;
```

Let's take a closer look at these variables:

*   `currCmdNum`: Local storage for the current command being executed.
*   `ledsOn`: Local storage used by the `blink` command to track state.
*   `nextLedCmd`: Storage for the next command coming from the queue.
*   `args`: A local variable containing the arguments passed in through the `void* Args` parameter of our task (notice the explicit cast and check to ensure `NULL` hasn't been passed in instead).

To ensure none of the pointers change, we're making a local copy. This also could have been accomplished by defining the `CmdExecArgs` struct to contain the `const` variables that could only be set at initialization to save a bit of space.

This main loop has two responsibilities. The first responsibility, seen in the following code, is copying a value from `ledCmdQueue` into the `nextLedCmd` reading, setting the appropriate local variables and the duty cycle of the LEDs.

`ledCmdExecutor.c` is part of the main loop:

```cpp
if(xQueueReceive(args.ledCmdQueue, &nextLedCmd, 250) == pdTRUE)
{
    switch(nextLedCmd.cmdNum)
    {
        case CMD_SET_INTENSITY:
            currCmdNum = CMD_SET_INTENSITY;
            setDutyCycles( &args, nextLedCmd.red, 
                           nextLedCmd.green, nextLedCmd.blue);
            break;
        case CMD_BLINK:
            currCmdNum = CMD_BLINK;
            blinkingLedsOn = true;
            setDutyCycles(&args, nextLedCmd.red, 
                            nextLedCmd.green, nextLedCmd.blue);
            break;
        //additional cases not shown
    }
}
```

The second part of the main loop, seen in the following code, executes if no command has been received from `ledCmdQueue` within 250 ticks (250 ms, since our configuration uses a 1 kHz tick). This code toggles the LEDs between their last commanded duty cycle and `OFF`:

`ledCmdExecutor.c` is the second half of the main loop:

```cpp
else if (currCmdNum == CMD_BLINK)
{
    //if there is no new command and we should be blinking
    if(blinkingLedsOn)
    {
        blinkingLedsOn = false;
        setDutyCycles(&args, 0, 0, 0);
    }
    else
    {
        blinkingLedsOn = true;
        setDutyCycles(  &args, nextLedCmd.red, 
                        nextLedCmd.green, nextLedCmd.blue);
    }
}
```

Finally, the `setDutyCycles` helper function uses the `iPWM` pointers to actuate the PWM duty cycles for the LEDs. The `iPWM` pointers were verified as not being `NULL` before the main loop, so the check doesn't need to be repeated here:

```cpp
void setDutyCycles( const CmdExecArgs* Args, float RedDuty,                            float GreenDuty, float BlueDuty)
{
  Args->redPWM->SetDutyCycle(RedDuty);
  Args->greenPWM->SetDutyCycle(GreenDuty);
  Args->bluePWM->SetDutyCycle(BlueDuty);
}
```

That wraps up the high-level functionality of our LED command executor. The main purpose of creating a task like this was to illustrate a way to create an extremely loosely coupled and scalable system. While it is silly to toggle a few LEDs in this way, this design paradigm is perfectly scalable to complex systems and capable of being used on different hardware without modification.

Now that we have an idea of what the code does at a high level, let's take a look at how the `LedCmd` struct is populated.

# Frame decoding

As data comes in from the USB, it is placed in a stream buffer by the USB stack. The `StreamBuffer` function for incoming data can be accessed from `GetUsbRxStreamBuff()` in `Drivers/HandsOnRTOS/VirtualCommDriverMultiTask.c`:

```cpp
StreamBufferHandle_t const * GetUsbRxStreamBuff( void )
{
  return &vcom_rxStream;
}
```

This function returns a constant pointer to `StreamBufferHandle_t`. This is done so the calling code can access the stream buffer directly, but isn't able to change the pointer's value.

The protocol itself is a strictly binary stream that starts with 0x02 and ends with a CRC-32 checksum, transmitted in little endian byte order:

![](img/20ce3e11-1209-4cf8-9e54-c4b59575bbf4.png)

There are many different ways of serializing data. A simple binary stream was chosen here for simplicity. A few points should be considered:

*   The `0x02` header is a convenient delimiter that can be used to find the (possible) start of a frame. It is not sufficiently unique since any of the other bytes in the message can also be `0x02` (it is a binary stream, not ASCII). The CRC-32 at the end provides assurance that the frame was correctly received.
*   Since the frame has exactly 1 byte per LED value, we can represent the 0-100% duty cycle with 0-255 and we are guaranteed to have valid, in-range parameters, without any additional checking.
*   This simple method of framing is extremely rigid and provides no flexibility whatsoever. The moment we need to send something else over the wire, we're back to square one. A more flexible (and complex) serialization method is required if flexibility is desired.

The `frameDecoder` function is defined in `mainColorSelector.c`:

```cpp
void frameDecoder( void* NotUsed)
{
  LedCmd incomingCmd;
  #define FRAME_LEN 9
  uint8_t frame[FRAME_LEN];
  while(1)
  {
    memset(frame, 0, FRAME_LEN);
    while(frame[0] != 0x02)
    {
      xStreamBufferReceive( *GetUsbRxStreamBuff(), frame, 1,
                                             portMAX_DELAY);
    }
    xStreamBufferReceive( *GetUsbRxStreamBuff(),
      &frame[1],
      FRAME_LEN-1,
      portMAX_DELAY);
    if(CheckCRC(frame, FRAME_LEN))
    {
      incomingCmd.cmdNum = frame[1];
      incomingCmd.red = frame[2]/255.0 * 100;
      incomingCmd.green = frame[3]/255.0 * 100;
      incomingCmd.blue = frame[4]/255.0 * 100;
      xQueueSend(ledCmdQueue, &incomingCmd, 100);
    }
  }
}
```

Let's break it down line by line:

*   Two local variables, `incomingCmd` and `frame`, are created. `incomingCmd` is used to store the fully parsed command. `frame` is a buffer of bytes that is used to store exactly one frame's worth of data while this function parses/verifies it:

```cpp
LedCmd incomingCmd;
#define FRAME_LEN 9
uint8_t frame[FRAME_LEN];
```

*   At the beginning of the loop, the contents of `frame` are cleared. Only clearing the first byte is strictly necessary so we can accurately detect `0x02` and since the frame is binary and has a well-defined length (only variable-length strings *need* to be null-terminated). However, it is very convenient to see `0` for unpopulated bytes, if you happen to be watching the variable during debugging:

```cpp
memset(frame, 0, FRAME_LEN);
```

*   A single byte is copied from the `StreamBuffer` function into the frame until `0x02` is detected. This should indicate the start of a frame (unless we were unlucky enough to start acquiring data in the middle of a frame with a binary value of `0x02` in the payload or CRC):

```cpp
 while(frame[0] != 0x02)
 {
     xStreamBufferReceive( *GetUsbRxStreamBuff(), frame, 1, 
                                            portMAX_DELAY);
 }
```

*   The remaining bytes of the frame are received from `StreamBuffer`. They are placed in the correct index of the `frame` array:

```cpp
xStreamBufferReceive( *GetUsbRxStreamBuff(), &frame[1], 
                           FRAME_LEN-1, portMAX_DELAY);
```

*   The entire frame's CRC is evaluated. If the CRC is invalid, this data is discarded and we'll begin looking for the start of another frame:

```cpp
if(CheckCRC(frame, FRAME_LEN))
```

*   If the frame was intact, `incomingCmd` is filled out with the values in the frame:

```cpp
 incomingCmd.cmdNum = frame[1];
 incomingCmd.red = frame[2]/255.0 * 100;
 incomingCmd.green = frame[3]/255.0 * 100;
 incomingCmd.blue = frame[4]/255.0 * 100;
```

*   The populated command is sent to the queue, which is being watched by `LedCmdExecutor()`. Up to `100` ticks may elapse, waiting for space in the queue to become available before the command is discarded:

```cpp
 xQueueSend(ledCmdQueue, &incomingCmd, 100);
```

It is important to note that none of the framing protocol is being placed in `LedCmd`, which will be sent through the queue – only the payload is. This allows more flexibility in how data is acquired before being queued, as we will see in the *Reusing a queue definition for a new targe**t *section.

Choosing the number of slots available in the queue can have important effects on the response of the application to incoming commands. The more slots that are available, the higher the likelihood that a command will incur a significant delay before being executed. For a system that requires more determinism on when (and if) a command will be executed, it is a good idea to limit the queue length to only a single slot and perform a protocol-level acknowledgment, based on whether the command was successfully queued.

Now that we've seen how the frame is decoded, the only remaining piece of the puzzle is how data is placed into the USB's receiving stream buffer.

# The USB virtual comm driver

The USB's receiving `StreamBuffer` is populated by `CDC_Receive_FS()` in `Drivers/HandsOnRTOS/usbd_cdc_if.c`. This will look similar to the code from [Chapter 11](c76b2fa5-28ac-4467-bb7e-68593a27f9ce.xhtml), *S**haring Hardware Peripherals across Tasks, *where the transmit side of the driver was developed:

```cpp
static int8_t CDC_Receive_FS(uint8_t* Buf, uint32_t *Len)
{
  /* USER CODE BEGIN 6 */
  portBASE_TYPE xHigherPriorityTaskWoken = pdFALSE;

  USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
  xStreamBufferSendFromISR( *GetUsbRxStreamBuff(),
        Buf,
        *Len,
        &xHigherPriorityTaskWoken);

  USBD_CDC_ReceivePacket(&hUsbDeviceFS);
  portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
  return (USBD_OK);
  /* USER CODE END 6 */
}
```

Using a stream buffer instead of a queue allows larger blocks of memory to be copied from the USB stack's internal buffers while providing a queue-like interface that has flexibility in the number of bytes copied out of it. This flexibility is one of the reasons coding the protocol layer was so straightforward.

Remember that since a stream buffer is being used, only one task can be a designated reader. Otherwise, access to the stream buffer must be synchronized (that is, by a mutex).

That wraps up all of the MCU-side code in this example. Since this example relies on a binary protocol over USB, let's have a look at how the code can be used.

# Using the code

One of the goals for choosing this example was to have an approachable, real-world use case that was relevant. Most of the time, the use cases for the embedded systems we develop do not include a person typing away on a terminal emulator. To that end, a very simple GUI was created using Python to make it simple to send commands to the Nucleo board. The script is `Chapter_13/PythonColorSelectorUI/colorSelector.py`.

A Windows binary is also included (`Chapter_13/PythonColorSelectorUI/colorSelector.exe`). `*.exe` doesn't require Python to be installed. For other operating systems, you'll need to install the requisite packages listed in `Chapter_13/PythonColorSelectorUI/requirements.txt`  and run a Python 3 interpreter to use the script:

1.  You'll first need to select the STM virtual comm port:

![](img/be28c33c-1f8b-49dd-9873-da348ac9bbcf.png)

2.  After the port has been successfully opened, use the sliders and various buttons to actuate the LEDs on the Nucleo development board. A command frame is constructed on each UI update event and immediately sent over USB to the MCU. The ASCII-encoded hex dump of the last frame sent is displayed:

![](img/3b2c45b4-1df7-46c6-9979-2ef793ab26d5.png)

Alternatively, a terminal application capable of sending binary data could also be used (RealTerm is an example for Windows).

So, we have some blinky lights and a (not-so) flashy UI. Let's address the real takeaway of this exercise—the flexibility we've built into our application by using a queue in the way that we did.

# Reusing a queue definition for a new target

On the surface, it might be hard to appreciate just how flexible a setup such as this one is. On the command-entry side, we have the ability to acquire commands from anything, not just the binary framed protocol over USB. Since the data being placed into the queue was abstracted to not include any protocol-specific information, the underlying protocol could change without requiring any changes downstream.

Let's take a look at a few examples:

*   We could write a different routine for parsing incoming data that uses a comma-separated ASCII string, where duty cycles are represented by percentages between 0 and 100 and a string-based enumeration, terminated by a new line: `BLINK, 20, 30, 100\n`. This would result in the following value being placed in `ledCmdQueue`:

```cpp
LedCmd cmd = {.cmdNum=3, .red=20, .blue=30, .green=100};
xQueueSend(ledCmdQueue, &cmd, 100);
```

*   The underlying interface could change completely (from USB to UART, SPI, I2C, Ethernet, an IoT framework, and so on). 
*   The commands could come in the form of a non-serialized data source (discrete duty cycles or physical pins of the MCU, for example):

![](img/9c4850a9-ddc6-40e9-8acd-66723eba2b2f.png)

There is also no reason for a queue to be limited to being populated by a single task. If it makes sense from a system design perspective, a command executor's queue could be populated by any number of simultaneous sources. The possibilities of how to get data into the system are truly endless, which is great news—especially if you're developing something considerably more complex that will be used in multiple systems. You're free to invest time in writing quality code once, knowing that the exact code will be used in multiple applications because it is flexible enough to adapt while still maintaining a consistent interface.

There are two components of the ledCmd executor that provide flexibility—the queue interface and the `iPWM` interface.

# The queue interface

After the  `LedCmdExecution()` task is started, the only interaction it has with the higher-level code in the system is through its command queue. Because of this, we are free to change the underlying implementation without directly affecting the higher-level code (as long as the data passed through the queue has the same meaning). 

For example, blinking could be implemented differently and none of the code feeding the queue would need to change. Since the only requirements for passing data through the queue are `uint8_t` and three floating-point numbers, we're also free to completely rewrite the implementation of `LedCmdExecution` (without the `iPWM` interface, for example). This change will only affect a single file that starts the task—`mainColorSelector.c`, in this example. Any other files that deal with the queue directly will be unaffected. If there were multiple sources feeding the queue (such as a USB, I2C, IoT framework, and so on), none of them would need to be modified, even after completely changing the underlying `LedCmdExecution` implementation.

# The iPWM interface

In this implementation, we've taken flexibility one step further by using a flexible interface to actuate the LEDs (`iPWM`). Since all of the calls (such as `setDutyCycles`) are defined by a flexible interface (instead of interacting directly with hardware), we're free to substitute any other implementation of `iPWM` in place of the MCU's `TIM` peripheral implementation included here. 

For example, the LED on the other end of the `iPWM` interface could be an addressable LED driven by a serial interface, which requires a serial data stream rather than PWM. It may even be remotely located and require another protocol to actuate it. As long as it can be represented by a percentage between 0 and 100, it can be controlled by this code.

Now—realistically—in the real world, you're not likely to go to all of this trouble just to blink a few LEDs! Keep in mind that this is an example with fairly trivial functionality, so we're able to keep our focus on the actual architectural elements. In practice, flexible architectures provide a foundation to build long-living, adaptable code bases on.

All of this flexibility does come with a warning—always be careful of the systems you are designing and ensure they meet their primary requirements. There are always tradeoffs to be made, whether they are between performance, initial development time, BOM cost, code elegance, flash space, or maintainability. After all, a beautifully extensible design doesn't do anyone any good if it doesn't fit in the available ROM and RAM!

# Summary

In this chapter, you have gained first-hand experience in creating a simple end-to-end command executor architecture. At this point, you should be quite comfortable with creating queues and will have started to gain a deeper understanding of how they can be used to achieve specific design goals, such as flexibility. You can apply variations of these techniques to many real-world projects. If you're feeling particularly adventurous, feel free to implement one of the suggested protocols or add an entry point for another interface (such as a UART). 

In the next chapter, we'll change gears a bit and discuss the available APIs for FreeRTOS, investigating when and why you might prefer one over the others.

# Questions

As we conclude, here is a list of questions for you to test your knowledge of this chapter's material. You will find the answers in the *Assessments* section of the appendix:

1.  Queues decrease design flexibility since they create a rigid definition of data transfer that must be adhered to:
    *   True
    *   False 
2.  Queues don't work well with other abstraction techniques; they must only contain simple data types:
    *   True
    *   False 
3.  When using queues for commands acquired from a serial port, should the queue contain exactly the same information and formatting as the underlying serialized data stream? Why?  
4.  Name a reason why passing data by value into queues is *easier* than passing by reference.  
5.  Name one reason why it is necessary to carefully consider the depth of queues in a real-time embedded system?