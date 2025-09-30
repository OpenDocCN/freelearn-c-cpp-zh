# Drivers and ISRs

Interacting with the peripherals of a **microcontroller unit** (**MCU**) is extremely important in many applications. In this chapter, we'll discuss several different ways of implementing peripheral drivers. Up to this point, we've been using blinking LEDs as a means of interacting with our development board. This is about to change. As we seek to gain a deeper understanding of peripheral drivers, we'll start to focus on different ways of implementing a driver for a common communication peripheral—the **u****niversal asynchronous receiver/transmitter** (**UART**). As we transfer data from one UART to another, we'll uncover the important role that peripheral and **direct memory access** (**DMA**) hardware plays when creating efficient driver implementations. 

We'll start by exploring a UART peripheral by implementing an extremely simple polled receive-only driver inside a task. After taking a look at the performance of that driver, we'll take a close look at **interrupt service routines** (**ISRs**) and the different ways they can interact with the RTOS kernel. The driver will be re-implemented using interrupts. After that, we'll add in support for a DMA-based driver. Finally, we'll explore a few different approaches to how the drivers can interact with the rest of the system and take a look at a newer FreeRTOS feature—stream buffers. Throughout this chapter, we'll keep a close eye on overall system performance using SystemView. By the end, you should have a solid understanding of the trade-offs to be made when writing drivers that can take advantage of RTOS features to aid usability.

The following topics will be covered in this chapter:

*   Introducing the UART 
*   Creating a polled UART driver
*   Differentiating between tasks and ISRs
*   Creating ISR-based drivers
*   Creating DMA-based drivers
*   FreeRTOS stream buffers
*   Choosing a driver model
*   Using third-party libraries (STM HAL)

# Technical requirements

To complete the exercises in this chapter, you will require the following:

*   A Nucleo F767 dev board
*   Jumper wires—20 to 22 AWG (~0.65 mm) solid core wire
*   A Micro-USB cable
*   STM32CubeIDE and source code (instructions in the *Setting up Our IDE* section in [Chapter 5](84a945dc-ff6c-4ec8-8b9c-84842db68a85.xhtml), *Selecting an IDE*)
*   SEGGER J-Link, Ozone, and SystemView (instructions in [Chapter 6](699daa80-06ae-4acc-8b93-a81af2eb774b.xhtml), *Debugging Tools for Real-Time Systems*)

All source code used in this chapter is available at [https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_10](https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_10).

# Introducing the UART

As we briefly covered in [Chapter 4](c52d7cdb-b6cb-41e8-8d75-72494bc9d4d3.xhtml), *Selecting the Right MCU*, the acronym **UART** stands for **Universal Asynchronous Receiver/Transmitter**. UART hardware takes bytes of data and transmits them over a wire by modulating the voltage of a signal line at a predetermined rate:

![](img/fc5fb57b-0534-4d40-b752-ae02f2483362.png)

The **asynchronous** nature of a UART means no additional clock line is needed to monitor individual bit transitions. Instead, the hardware is set up to transition each bit at a specific frequency (baud rate). The UART hardware also adds some extra framing to the beginning and end of each packet it transmits. Start and stop bits signal the beginning and end of a packet. These bits (along with an optional parity bit) are used by the hardware to help guarantee the validity of packets (which are typically 8 bits long).

A more general form of UART hardware is the **USART** **universal synchronous/asynchronous receiver transmitter** (**USART**). USARTs are capable of transferring data either synchronously (with the addition of a clock signal) or asynchronously (without a clock signal).

UARTs are often used to communicate between different chips and systems. They form the foundation of many different communication solutions, such as RS232, RS422, RS485, Modbus, and so on. UARTs can also be used for multi-processor communication and to communicate with different chips in the same system—for example, WiFi and Bluetooth transceivers.  

In this chapter, we'll be developing a few iterations of a UART driver. In order to be able to observe system behavior, we'll be tying two UARTs on the Nucleo development board together, as in the following diagram. The two connections in the diagram will tie the transmit signal from UART4 to the receive signal of USART2\. Likewise, they'll tie USART2 Tx to UART4 Rx. This will allow bidirectional communication between the UARTs. The connections should be made with pre-terminated **jumper wires** or 20-22 AWG (~0.65 mm) solid core wires:

![](img/1b44a7da-7e56-4f6a-8bd2-ed54043bb1fb.png)

Now that the connections are made, let's take a closer look at what else needs to happen before we can consider transferring data between peripherals on this chip.

# Setting up the UART

As we can see from the following simplified block diagram, there are a few components involved when setting up a UART for communication. The UART needs to be properly configured to transmit at the correct baud rate, parity settings, flow control, and stop bits. Other hardware that interacts with the UART will also need to be configured properly:

![](img/316246b5-dc2a-4957-abd5-c2795961bf83.png)

Here's a list of steps that will need to be taken to get UART4 set up. Although we're using UART4 as an example, the same steps will apply to most other peripherals that attach to pins of the MCU:

1.  Configure the GPIO lines. Since each GPIO pin on the MCU can be shared with many different peripherals, they must be configured to connect to the desired peripheral (in this case, the UART). In this example, we'll cover the steps to connect PC10 and PC11 to UART4 signals:

You can read more about the pinout of the STM32F7xx series MCUs in *Section 3, Pinouts and Pin Description*, of STM's STM32F767xx datasheet *DoCID 029041*. Datasheets will typically contain information specific to exact models of MCUs, while reference manuals will contain general information about an entire family of MCUs. The following excerpt is of a table is from the datasheet and shows alternate function pin mappings:

![](img/cbd883b1-430b-4c99-84bd-5e5203a066a5.png)

2.  Reference the desired port and bit. (In this case, we'll be setting up port `C` bit `11` to map to the `UART4_Rx` function).
3.  Find the desired alternate function for the pin (`UART4_Rx`).
4.  Find the alternate function number (`AF8`) to use when configuring the GPIO registers.
5.  Set up the appropriate GPIO registers to correctly configure the hardware and map the desired UART peripheral to the physical pins. 

An STM-supplied `HAL` function is used here for simplicity. The appropriate GPIO registers will by written when `HAL_GPIO_Init` is called.  All we need to do is fill in a `GPIO_InitTypeDef` struct and pass a reference to `HAL_GPIO_Init`; in the following code, the `10` GPIO pin and the `11` GPIO pin on port `C` are both initialized to alternative push/pull functions. They are also mapped to `UART4` by setting the alternate function member to `AF8`—as determined in step 4:

```cpp
GPIO_InitTypeDef GPIO_InitStruct = {0};
//PC10 is UART4_TX PC11 is UART4_RX
GPIO_InitStruct.Pin = GPIO_PIN_10 | GPIO_PIN_11;
GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
GPIO_InitStruct.Pull = GPIO_NOPULL;
GPIO_InitStruct.Alternate = GPIO_AF8_UART4;
HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
```

6.  Enable the necessary peripheral clocks. Since each peripheral clock is turned off by default (for power saving), the UART's peripheral clock must be turned on by writing to the **reset and clock control** (**RCC**) register. The following line is also from `HAL`:

```cpp
__UART4_CLK_ENABLE();
```

7.  Configure the interrupts (if using them) by configuring settings in the **nested vector interrupt controller** (**NVIC**)—details will be included in the examples where appropriate.
8.  Configure the DMA (if using it)—details will be included in the examples where appropriate.
9.  Configure the peripheral with the necessary settings, such as baud rate, parity, flow control, and so on. 

The following code is an excerpt from the `STM_UartInit` function in `BSP/UartQuickDirtyInit.c.`, where **`Baudrate`**and **`STM_UART_PERIPH`** are input parameters of `STM_UartInit`, which makes it very easy to configure multiple UART peripherals with similar settings, without repeating all of the following code every time:

```cpp
HAL_StatusTypeDef retVal;
UART_HandleTypeDef uartInitStruct;
uartInitStruct.Instance = STM_UART_PERIPH;
uartInitStruct.Init.BaudRate = Baudrate;
uartInitStruct.Init.WordLength = UART_WORDLENGTH_8B;
uartInitStruct.Init.StopBits = UART_STOPBITS_1;
uartInitStruct.Init.Parity = UART_PARITY_NONE;
uartInitStruct.Init.Mode = UART_MODE_TX_RX;
uartInitStruct.Init.HwFlowCtl = UART_HWCONTROL_NONE;
uartInitStruct.Init.OverSampling = UART_OVERSAMPLING_16;
uartInitStruct.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
uartInitStruct.hdmatx = DmaTx;
uartInitStruct.hdmarx = DmaRx;
uartInitStruct.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
retVal = HAL_UART_Init(&uartInitStruct);
assert_param(retVal == HAL_OK);
```

10.  Depending on the desired transmit method (such as polled, interrupt-driven, or DMA), some additional setting up will be required; this setting up is typically performed immediately before beginning a transfer.

Let's see how all of this plays out by creating a simple driver to read data coming into USART2. 

# Creating a polled UART driver

When writing low-level drivers, it's a must to read through the datasheet in order to understand how the peripheral works. Even if you're not writing a low-level driver from scratch, it is always a good idea to gain some familiarity with the hardware you'll be working with. The more familiarity you have, the easier it will be to diagnose unexpected behavior, as well as to create efficient solutions.

You can read more about the UART peripheral we're working with in *Chapter 34* of the*STM **RM0410 STM32F76xxx* reference manual (*USART*).

Our first **driver** will take an extremely simple approach to getting data from the UART and into a queue that can be easily monitored and consumed by any task in the system. By monitoring the **receive not empty** ( `RXNE`) bit of the UART peripheral's **interrupt status register** (`ISR`), the driver can determine when a new byte is ready to be transferred from the **receive data register** (`RDR`) of the UART into the queue. To make this as easy as possible, the `while` loop is placed in a task (`polledUartReceive`), which will let other higher-priority tasks run.  

The following is an excerpt from `Chapter_10/Src/mainUartPolled.c`:

```cpp
void polledUartReceive( void* NotUsed )
{
    uint8_t nextByte;
    //setup UART
    STM_UartInit(USART2, 9600, NULL, NULL);
    while(1)
    {
        while(!(USART2->ISR & USART_ISR_RXNE_Msk)); nextByte = USART2->RDR; xQueueSend(uart2_BytesReceived, &nextByte, 0); }
}
```

There is another simple task in this example as well; it monitors the queue and prints out whatever has been received:

```cpp
void uartPrintOutTask( void* NotUsed)
{
    char nextByte;    
    while(1)
    {
        xQueueReceive(uart2_BytesReceived, &nextByte, portMAX_DELAY);
        SEGGER_SYSVIEW_PrintfHost("%c", nextByte);    
    }
}
```

Now that our driver is ready, let's see how it performs. 

# Analyzing the performance

The preceding code (`uartPolled`)can be programmed onto the MCU and we can take a look at the performance using SEGGER SystemView:

![](img/8f262810-3376-400f-9f34-745d331f11ca.png)

After looking at the execution using SystemView, we quickly realize that—although easy to program—this driver is *horrifically inefficient*:

1.  SystemView reports that this driver is utilizing *over 96%* of the CPU's resources.
2.  The queue is being called at 960 Hz (which makes perfect sense given the initial baud rate of 9,600 baud).

We can see that, while easy to implement, this solution comes with significant performance penalties—all while servicing a fairly slow peripheral. Drivers that service peripherals by polling have trade-offs.

# Pros and cons of a polled driver

Here are some of the advantages of using a polled driver:

*   It is easy to program.
*   Any task has immediate access to data in the queue.

At the same time, there are many issues with this approach:

*   It must be one of the highest priority tasks in the system.
*   There is a high chance of data loss when not executing at high priority.
*   It is extremely wasteful of CPU cycles.

In this example, we're only transferring data at 9,600 baud. Granted, most of the time was spent spinning on the `RXNE` bit, but transferring every byte as it is received in a queue is also fairly expensive (when compared to pushing bytes into a simple array-based buffer). To put this into perspective, USART2 on STM32F767 running at 216 MHz has a maximum baud rate of 27 Mbaud, which would mean we would need to add each character to the shared queue nearly 3 million times a second (it is currently adding < 1,000 characters per second). Transferring this much data through a queue quickly isn't feasible on this hardware since queue additions take 7 µS each (even if the CPU was doing nothing else, we'd be capable of transferring less than 143,000 characters per second into the queue).

More importantly, there are few opportunities to speed up this polled approach, since we may receive a new character once every millisecond. If any other task was executing for more than 2 ms, the peripheral could potentially be overrun (a new byte is received and overwrites the buffer before the previous byte is read). Because of these limitations, there are very specific circumstances where polled drivers are most useful.

# Usage of polled drivers

There are a few circumstances where polled drivers are especially helpful:

*   **System verification**: This is perfectly acceptable when performing initial system verification, but at that stage of development, it is debatable whether an RTOS should be used at all. If the application happens to be truly single purpose, there is nothing else to be done while waiting for data to be transferred, and there are no power considerations, this would also be an acceptable approach.
*   **Special cases**: Occasionally, there may be times when a very special-purpose piece of code is needed for a limited scope. For example, a peripheral may need to be serviced with an extremely low amount of latency. In other cases, the event being polled for could happen extremely quickly. When events are happening in the order of nanoseconds or microseconds ns or µs (instead of ms, as in the previous example), it often makes more sense to simply poll for the event, rather than create a more elaborate synchronization scheme. In event-driven systems, adding in blocking calls must be carefully considered and clearly documented.

Conversely, if a given event is happening very infrequently and there are no specific timing constraints, a polled approach may also be perfectly acceptable.

While the driver in this example focused on the receiving side, where poll-based drivers are rarely acceptable, it is more common to find them used to transmit data. This is because space between the characters is generally acceptable since it doesn't result in loss the of data. This allows the driver to be run at a lower priority so that other tasks in the system have a chance to run. There are a few cases where there is a reasonable argument for using a polled transmit driver that blocks while the transmission is taking place:

*   The code using the driver must block until the transmission is complete.
*   The transfer is a short amount of data.
*   The data rate is reasonably high (so the transfer takes a relatively small amount of time).

If all of these conditions are met, it *may* make sense to simply use a polled approach, rather than a more elaborate interrupt- or DMA-driven approach, which will require the use of callbacks and, potentially, task synchronization mechanisms. However, depending on how you choose to structure your drivers, it is also possible to have the convenience of blocking calls but without the inefficiency of a polled transfer wasting CPU cycles. To take advantage of any of the non-polled approaches, we'll need to develop another skill—programming ISRs.

# Differentiating between tasks and ISRs

Before we jump into coding a peripheral driver that utilizes interrupts, let's take a quick look at how interrupts compare to FreeRTOS tasks.

There are many similarities between tasks and ISRs:

*   Both provide a way of achieving **parallel** code execution.
*   Both only run when required.
*   Both can be written with C/C++ (ISRs generally no longer need to be written in assembly code).

But there are also many differences between tasks and ISRs:

*   **ISRs are brought into context by hardware; tasks gain context by the RTOS kernel**: Tasks are always brought into context by the FreeRTOS kernel. Interrupts, on the other hand, are generated by hardware in the MCU. There are usually a few different ways of configuring the generation (and masking) of interrupts.
*   **ISRs must exit as quickly as possible; tasks are more forgiving**: FreeRTOS tasks are often set up to run in a similar way to an infinite `while` loop—they will be synchronized with the system using primitives (such as queues and semaphores) and switched into context according to their priority. At the complete opposite end of the spectrum are ISRs, which should generally be coded so that they exit quickly. This *quick exit* ensures that the system can respond to other ISRs, which keeps everything responsive and ensures no interrupts are missed because a single routine was hogging the CPU.
*   **ISR functions do not take input parameters; tasks can**: Unlike tasks, ISRs can never have input parameters. Since an interrupt is triggered because of a hardware state, the most important job of the ISR is to read the hardware state (through memory-mapped registers) and take the appropriate action(s). For example, an interrupt can be generated when a UART receives a byte of data. In this case, the ISR would read a status register, read (and store) the byte received in a static variable, and clear the interrupt.

Most (but not all) peripherals on STM32 hardware will automatically clear interrupt flags when certain registers are read. Regardless of how the interrupt is cleared, it is important to ensure the interrupt is no longer pending—otherwise, the interrupt will fire continuously and you will always be executing the associated ISR!

*   **ISRs may only access a limited ISR-specific subset of the FreeRTOS API**: FreeRTOS is written in a way that provides flexibility while balancing convenience, safety, and performance. Accessing data structures such as queues from a task is extremely flexible (for example, tasks making API calls to a queue can easily block for any period of time). There is an additional set of functions that are available to ISRs for operating on queues, but these functions have a limited subset of functionality (such as not being able to block—the call always immediately returns). This provides a level of safety since the programmer can't shoot themself in the foot by calling a function that blocks from inside an ISR. Calling a non-ISR API function from inside an ISR will cause FreeRTOS to trigger `configASSERT`.
*   **ISRs may operate completely independently of all RTOS code**: There are many cases where an ISR operates on such a low level that it doesn't *need* access to any of the FreeRTOS API at all. In this case, the ISR simply executes as it normally would without an RTOS present. The kernel never gets involved (and no tasks will interrupt execution).  This makes it very convenient for creating flexible solutions that blend high-performing ISRs (operating completely *underneath* the RTOS) with extremely convenient tasks.
*   **All ISRs share the same system stack; each task has a dedicated stack**: Each task receives a private stack, but all of the ISRs share the same system stack. This is noteworthy only because, when writing ISRs, you'll need to ensure you reserve enough stack space to allow them to run (possibly simultaneously) if they are nested.

Now that we've covered the differences between tasks and ISRs, let's take a look at how they can be used together to create very powerful event-driven code.

# Using the FreeRTOS API from interrupts

Most of the FreeRTOS primitives covered so far have ISR-safe versions of their APIs. For example, `xQueueSend()` has an equivalent ISR-safe version, `xQueueSendFromISR()`. There are a few differences between the ISR-safe version and the standard call:

*   The `FromISR` variants won't block. For example, if `xQueueSendFromISR` encounters a full queue, it will immediately return.
*   The `FromISR` variants require an extra parameter, `BaseType_t *pxHigherPriorityTaskWoken`, which will indicate whether or not a higher-priority task needs to be switched into context immediately following the interrupt.
*   Only interrupts that have a *logically* lower priority than what is defined by `configMAX_API_CALL_INTERRUPT_PRIORITY` in `FreeRTOSConfig.h` are permitted to call FreeRTOS API functions (see the following diagram for an example).

The following is an overview of how the `FreeRTOSConfig.h` and `main_XXX.c` files configure interrupts for the examples in this book. Some noteworthy items are as follows:

*   `main_XXX.c` makes a call to `NVIC_SetPriorityGrouping(0)` after all STM HAL initialization is performed (`HAL` sets priority grouping upon initialization). This allows all 4 bits of the **nested interrupt vector controller** (**NVIC**) to be used for priorities and results in 16 priority levels. 
*   `FreeRTOSConfig.h` is used to set up the relationship between FreeRTOS API calls and NVIC priorities. The Cortex-M7 defines `255` as being the lowest priority level and `0` as being the highest. Since the STM32F7 only implements 4 bits, these 4 bits will be shifted into the 4 MSB bits; the lower 4 bits won't affect operation (see the following diagram): 
    *   `configKERNEL_INTERRUPT_PRIORITY` defines the lowest priority interrupt in our system (and the ISR priority of the FreeRTOS tasks, since the scheduler is called within a SysTick interrupt). Because 4 bits yields a possible range of `0` (highest priority) to `15` (lowest priority), the lowest NVIC priority used will be `15`. When setting `configKERNEL_INTERRUPT_PRIORITY`, `15` needs to be shifted left into the 8 bit representation (used directly in the CortexM registers) as `(15 << 4) | 0x0F = 0xFF` or `255`.  Since the lowest 4 bits are don't cares, `0xF0` (decimal 240) is also acceptable.  
    *   `configMAX_SYSCALL_INTERRUPT_PRIORITY` defines the (logically) highest priority interrupt that is allowed to make calls to the FreeRTOS API. This is set to `5` in our examples. Shifting left to fill out the 8 bits gives us a value of `0x50` or `0x5F` (decimal 80 or 95, respectively):

![](img/c94abcbe-e57f-476c-b15f-15272efe8f64.png)

As we can see in the preceding diagram, there are some cases where ISRs can be set up to execute at a priority above anything the RTOS might be doing. When configured as `0` to `4` NVIC priorities, ISRs are identical to traditional "bare-metal" ISRs.

It is *very* important to ensure that the interrupt priority is properly configured *before* enabling the interrupt by calling `NVIC_SetPriority` with a priority of <= 5\. If an interrupt with a priority that is logically higher than `configMAX_SYSCALL_INTERRUPT_PRIORITY` calls a FreeRTOS API function, you'll be greeted with a `configASSERT` failure (see [Chapter 17](50d2b6c3-9a4e-45c3-9bfc-1c7f58de0b98.xhtml), *Troubleshooting Tips and Next Steps*, for more details on `configASSERT`).

Now that we have an understanding of the differences between tasks and ISRs, as well as some of the ground rules for using FreeRTOS API functions from within ISRs, let's take another look at the polled driver to see how it can be implemented more efficiently.

# Creating ISR-based drivers

In the first iteration of the UART driver, a task polled the UART peripheral registers to determine when a new byte had been received. The constant polling is what caused the task to consume > 95% of CPU cycles. The most meaningful work done by this task-based driver was transferring bytes of data out of the UART peripheral and into the queue. 

In this iteration of the driver, instead of using a task to continuously poll the UART registers, we'll set up the `UART2` peripheral and NVIC to provide an interrupt when a new byte is received.

# Queue-based driver

First, let's look at how to more efficiently implement the polled driver (previously implemented by polling the UART registers within a task). In this implementation, instead of using a task to repeatedly poll the UART registers, a function will be used to set up the peripheral to use interrupts and initiate the transfer. A complete set of ISR function prototypes can be found in the startup file (for the STM32F767 used in our examples, this file is `Chapter_10/startup_stm32f767xx.s`).

Each `*_IRQHandler` instance in `startup_stm32f767xx.s` is used to map the function name to an address in the interrupt vector table. On ARM Cortex-M0+, -M3, -M4, and -M7 devices, this vector table can be relocated by an offset at runtime. See *Further reading* for some links to more information on these concepts.

This example has four primary components:

*   `uartPrintOutTask`: This function initializes USART2 and associated hardware, starts a reception, and then prints anything placed in the `uart2_BytesReceived` queue.
*   `startReceiveInt`: Sets up an interrupt-based reception for USART2.
*   `USART2_IRQHandler`: An ISR is issued when an interrupt occurs for the USART2 peripheral.
*   `startUart4Traffic`: Starts a continuous stream of data transmitted from UART4 to be received by USART2 (provided the jumpers are correctly set).

Let's take a look at each component in detail. All excerpts in this section are from `Chapter_10/Src/mainUartInterruptQueue.c`.

# uartPrintOutTask 

The only task in this example is `uartPrintOutTask`: 

```cpp
void uartPrintOutTask( void* NotUsed)
{
  char nextByte;
  STM_UartInit(USART2, 9600, NULL, NULL);
  startReceiveInt();

  while(1)
  {
    xQueueReceive(uart2_BytesReceived, &nextByte, portMAX_DELAY);
    SEGGER_SYSVIEW_PrintfHost("%c", nextByte);
  }
}
```

`uartPrintOutTask` does the following:

*   Performs all peripheral hardware initialization by calling `STM_UartInit`
*   Starts an interrupt-based reception by calling `startReceiveInt`
*   *Consumes* and prints each character as it is added to the `uart2_BytesReceived` queue by calling `xQueueReceive`

# startReceiveInt

The `startReceiveInt` function starts an interrupt-driven reception:

```cpp
static bool rxInProgress = false;

void startReceiveInt( void )
{
    rxInProgress = true;
    USART2->CR3 |= USART_CR3_EIE; //enable error interrupts
    //enable peripheral and Rx not empty interrupts
    USART2->CR1 |= (USART_CR1_UE | USART_CR1_RXNEIE);  

    NVIC_SetPriority(USART2_IRQn, 6);
    NVIC_EnableIRQ(USART2_IRQn);
}
```

`startReceiveInt` sets up everything required to receive data on USART2:

*   `rxInProgress` is a flag used by the ISR to indicate a reception is in progress. The ISR (`USART2_IRQHandler()`) will not attempt to write to the queue until `rxInProgress` is true.
*   USART2 is configured to generate `receive` and `error` interrupts and is then enabled.
*   The `NVIC_SetPriority` function (defined by CMSIS in `Drivers/CMSIS/Include/corex_cm7.h`) is used to set the interrupt priority. Since this interrupt will call a FreeRTOS API function, this priority must be set at or *below* thelogical priority defined by  `configLIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY` in `FreeRTOSConfig.h`. On ARM CortexM processors, smaller numbers signify a higher logical priority—in this example, `#define configLIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY 5`, so assigning a priority of `6` to `USART2_IRQn` will be adequate for allowing the ISR to make calls to the ISR-safe function (`xQueueSendFromISR`) provided by FreeRTOS.
*   Finally, the interrupt requests generated by USART2 will be enabled by making a call to `NVIC_EnableIRQ`.   If `NVIC_EnableIRQ` is not called, USART2 will still generate requests, but the interrupt controller (the "IC" in NVIC) will not *vector* the program counter to the ISR (`USART2_IRQHandler` will never be called).

In this example, as with nearly all the code in this chapter, we're writing directly to the hardware peripheral registers and not using considerable amounts of abstraction. This is done to keep the focus on how the RTOS interacts with the MCU. If code reuse is one of your goals, you'll need to provide some level of abstraction above raw registers (or STM HAL code, if you're using it). Some guidelines on this can be found in [Chapter 12](8e78a49a-1bcd-4cfe-a88f-fb86a821c9c7.xhtml), *Tips on Creating Well-Abstracted Architecture*.

# USART2_IRQHandler

Here is the code for `USART2_IRQHandler`:

```cpp
void USART2_IRQHandler( void )
{
    portBASE_TYPE xHigherPriorityTaskWoken = pdFALSE;
    SEGGER_SYSVIEW_RecordEnterISR();

    //error flag clearing omitted for brevity

    if( USART2->ISR & USART_ISR_RXNE_Msk)
    {
        uint8_t tempVal = (uint8_t) USART2->RDR;

        if(rxInProgress)
        {
            xQueueSendFromISR(uart2_BytesReceived, &tempVal, 
                              &xHigherPriorityTaskWoken);
      }
      SEGGER_SYSVIEW_RecordExitISR();
      portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}
```

Let's take a closer look at each component of the ISR:

*   The USART registers are directly read to determine whether or not the receive not empty (`RXNE`) is set. If it is, the contents of the receive data register (`RDR`) are stored to a temporary variable (`tempVal`)—this read clears the interrupt flag. If a receive is in progress, `tempVal` is sent to the queue.
*   Calls to `SEGGER_SYSVIEW_RecordEnterISR` and `SEGGER_SYSVIEW_RecordExitISR` are made upon entry and exit, which gives SEGGER SystemView the visibility to display the interrupt with all of the other tasks in the system.
*   The `xHigherPriorityTaskWoken` variable is initialized to false. This variable is passed to the `xQueueSendFromISR` function and is used to determine whether a high-priority task (higher than the one currently in the non-ISR context) is blocking because it is waiting on an empty queue. In this case, `xHigherPriorityTaskWoken` will be set to true, indicating a higher-priority task should be woken immediately after the ISR exits. When the call to `portYIELD_FROM_ISR` is made, if `xHigherPriorityTaskWoken` is true, the scheduler will immediately switch to the higher-priority task.

Now that the ISR has been written, we'll need to make sure it will actually be called by the hardware at the appropriate time.

# Tips for linking ISRs

When writing ISRs from scratch (as we've done in the previous example), one area that can prove to be a source of unexpected trouble is ensuring your ISR is properly linked in (and executed). That is, even if you've properly set up the peripheral to generate interrupts, your new ISR might never be called because it isn't named properly (instead, the default implementation, defined in a startup file, will likely be called). Here are some tips to make sure that shiny new ISR can be found and properly linked in with the rest of the application:

*   STM32 `*_IRQHandler` function names *usually* contain the *exact* peripheral name from the datasheet as a sub-string. For example, USART2 maps to `USART2_IRQHandler` (notice the "S") and UART4 maps to `UART4_IRQHandler` (no "S" in the peripheral or function name).
*   When writing a new implementation for an ISR, it is a good idea to copy and paste the exact `_IQRHandler` name from the startup file. This reduces the chance of typos, which can cause debug headaches!
*   STM start-up files implement default handlers for every interrupt as an infinite loop. If you notice your application becoming unresponsive, it is possible you've enabled an interrupt and your `*_IRQHandler` definition isn't being linked in properly.
*   If you happen to be implementing `*_IRQHandler` inside a C++ file, be sure to use `extern "C"` to prevent *name mangling*. For example, the USART2 definition would be written as `extern "C" void USART2_IRQHandler( void)`. This also means the ISR definition must *not* be inside a class.

When implementing ISRs, take your time and be sure to get the details (such as the *exact* name) right. Don't rush into attempting to debug the rest of your application without first ensuring the ISR is called when expected. Using breakpoints inside the ISR is an excellent way of doing this.

# startUart4Traffic

The final component that needs to be explored in this example is how data will be sent to UART2\. These examples are meant to simulate external data being received by USART2.  To achieve this without additional hardware, we wired together UART4 Tx and USART2 RX pins earlier in the chapter. The call to `startUart4Traffic()` is a `TimerHandler` prototype. A oneshot timer is started and set to fire 5 seconds after the application starts.

The function that does all of the heavy lifting is `SetupUart4ExternalSim()`. It sets up a continuous circular DMA transfer (which executes without CPU intervention) that transmits the string `data from uart4` repeatedly. A full example using DMA will be covered later in this chapter – for now, it is sufficient to realize that data is being sent to USART2 without involvement from the CPU.

`startUart4Traffic()` creates a *continuous* stream of bytes that will be transmitted out of UART4 Tx and into UART2 Rx (with no flow control). Depending on the selected baud rate and the amount of time it takes for the receiving code to execute, we can expect that, eventually, a byte will be missed on the receiving side during some examples. Keep this in mind when running examples on your own. See the *Choosing a driver model* section for more details on selecting the appropriate driver type for your application.

# Performance analysis

Now, let's take a look at the performance of this implementation by compiling `mainUartInterruptQueue`, loading it onto the MCU, and using SystemView to analyze the actual execution:

![](img/1db4a7e2-4ed1-4022-864f-efba7ed13ab6.png)

This time around, things look considerably better.  Here are some noteworthy items from the preceding screenshot:

1.  The ISR responsible for dealing with the incoming data on USART2 Rx is only consuming around 1.6% of the CPU (much better than the 96% we saw when we were using a polled approach).
2.  We are still receiving 960 bytes per second—the same as before.
3.  The small tick mark shown here is the exact point in time when `tempVal` is added to `uart2_BytesReceived` by the call to the `xQueueSendFromISR` FreeRTOS API function.
4.  We can see the effect of `portYIELD_FROM_ISR` here. The light-blue portion of the `uartPrint` task indicates that the task is ready to run. This happens because the `uartPrint` task is ready to run since there is an item in the queue. The call to `portYIELD_FROM_ISR` forces the scheduler to immediately evaluate which task should be brought into context. The green portion (starting at ~21 uS) is SystemView's way of signifying that the task is in a **running** state.
5.  After the `uartPrint` task begins running, it removes the next character from the queue and prints it using `SEGGER_SYSVIEW_PrintfHost`.  

By switching from a poll-based driver to an interrupt-based driver, we've significantly reduced the CPU load. Additionally, systems that use an interrupt-based driver can run other tasks while still receiving data through USART2\. This driver also uses a queue-based approach, which provides a very convenient ring buffer, allowing characters to be continuously received and added to the queue, then read whenever it is convenient for higher-level tasks. 

Next, we'll work through an example of a similar driver that doesn't use a queue at all.

# A buffer-based driver

Sometimes, the exact size of a transfer is known in advance. In this case, a pre-existing buffer can be passed to the driver and used in place of a queue. Let's take a look at an example of a buffer-based driver, where the exact number of bytes is known in advance. The hardware setup for this example is identical to the previous examples—we'll concentrate on receiving data through USART2.

Instead of using a queue, `uartPrintOutTask` will supply its own buffer to the `startReceiveInt` function. Data received by USART2 will be placed directly in the local buffer until the desired number of bytes have been added, then a semaphore will be given by the ISR to provide notification of the completion. The entire message will be printed as a single string, rather than 1 byte at a time, as it is received (which was done in the last example).

Just like the previous example, there are four main components. However, their responsibilities vary slightly:

*   `startReceiveInt`: Sets up an interrupt-based reception for USART2 and configures the necessary variables used by the ISR for the transfer.
*   `uartPrintOutTask`: This function initializes USART2 and associated hardware, starts a reception, and waits for completion (with a deadline of 100 ms). The complete message is either printed or a timeout occurs and an error is printed.
*   `USART2_IRQHandler`: An ISR is issued when an interrupt occurs for the USART2 peripheral.
*   `startUart4Traffic`: Starts a continuous stream of data transmitted from UART4 to be received by USART2 (provided the jumpers are correctly set).

Let's take a look at each component in detail. All excerpts in this section are from `Chapter_10/Src/mainUartInterruptBuffer.c`.

# startReceiveInt

The `startReceiveInt` function is very similar to the one used for the queue-based driver:

```cpp
static bool rxInProgress = false;
static uint_fast16_t rxLen = 0;
static uint8_t* rxBuff = NULL;
static uint_fast16_t rxItr = 0;

int32_t startReceiveInt( uint8_t* Buffer, uint_fast16_t Len )
{
    if(!rxInProgress && (Buffer != NULL))
    {
        rxInProgress = true;
        rxLen = Len;
        rxBuff = Buffer;
        rxItr = 0;
        USART2->CR3 |= USART_CR3_EIE; //enable error interrupts
        USART2->CR1 |= (USART_CR1_UE | USART_CR1_RXNEIE);
        NVIC_SetPriority(USART2_IRQn, 6);
        NVIC_EnableIRQ(USART2_IRQn);
        return 0;
    }
    return -1;
}
```

Here are the notable differences in this setup:

*   This variant takes in a pointer to a buffer (`Buffer`), as well as the desired length of the transfer (`Len`). A couple of global variables, `rxBuff` and `rxLen` (which will be used by the ISR), are initialized using these parameters.
*   `rxInProgress` is used to determine whether a reception is already in progress (returning `-1`  if it is).
*   An iterator (`rxItr`) that is used to index into the buffer is initialized to `0`.

All of the remaining functionality of `startReceiveInt` is identical to the example covered in the *Queue-based driver* section earlier in the chapter.

# uartPrintOutTask

The `uartPrintOutTask` function that is responsible for printing out data received by USART2 is a bit more complex in this example. This example is also capable of comparing the received data against an expected length, as well as some rudimentary error detection:

1.  The buffer and length variables are initialized and the UART peripheral is set up:

```cpp
void uartPrintOutTask( void* NotUsed)
{
  uint8_t rxData[20];
  uint8_t expectedLen = 16;
  memset((void*)rxData, 0, 20);

  STM_UartInit(USART2, 9600, NULL, NULL);
```

2.  Then, the body of the `while` loop starts a reception by calling `startReceiveInt` and then waits for the `rxDone` semaphore for up to 100 RTOS ticks for the transfer to complete.
3.  If the transfer completes in time, the total number of bytes received is compared against `expectedLen`. 

4.  If the correct number of bytes are present, the content of `rxData` is printed. Otherwise, a message providing an explanation of the discrepancy is printed:

```cpp
while(1)
{
    startReceiveInt(rxData, expectedLen);
    if(xSemaphoreTake(rxDone, 100) == pdPASS)
    {
        if(expectedLen == rxItr)
        {
            SEGGER_SYSVIEW_PrintfHost("received: ");
            SEGGER_SYSVIEW_Print((char*)rxData);
        }
        else
        {
            SEGGER_SYSVIEW_PrintfHost("expected %i bytes received" 
                                      "%i", expectedLen, rxItr);
```

The remainder of the `while` loop and function simply prints `timeout` if the semaphore is not taken within 100 ticks.

# USART2_IRQHandler

This ISR is also slightly more involved since it is required to keep track of the position in a queue:

1.  Private globals are used by `USART2_IRQHandler` because they need to be accessible by both the ISR and used by both `USART2_IRQHandler`  and `startReceiveInt`:

```cpp
static bool rxInProgress = false;
static uint_fast16_t rxLen = 0;
static uint8_t* rxBuff = NULL;
static uint_fast16_t rxItr = 0;
```

2.  The same paradigm for storing `xHigherPriorityTaskWoken` and SEGGER SystemView tracing is used in this ISR, just like in the last example:

```cpp
void USART2_IRQHandler( void )
{
     portBASE_TYPE xHigherPriorityTaskWoken = pdFALSE;
     SEGGER_SYSVIEW_RecordEnterISR();
```

3.  Next, errors are checked by reading the overrun (`ORE`), noise error (`NE`), framing error (`FE`), and parity error (`PE`) bits in the interrupt state register (`USART2->ISR`). 

If an error is present, it is cleared by a write to the interrupt clear register (`USART2->ICR`) and the `rxDone` semaphore is given. It is the responsibility of the caller code to check the number of bits in the buffer by looking at the `rxItr` variable (shown in the next code block) to ensure the correct number of bits were successfully received:

```cpp

 if( USART2->ISR & ( USART_ISR_ORE_Msk |
                     USART_ISR_NE_Msk |
                     USART_ISR_FE_Msk |
                     USART_ISR_PE_Msk ))
{
    USART2->ICR |= (USART_ICR_FECF |
                    USART_ICR_PECF |
                    USART_ICR_NCF |
                    USART_ICR_ORECF);
    if(rxInProgress)
    {
        rxInProgress = false;
        xSemaphoreGiveFromISR(rxDone,
        &xHigherPriorityTaskWoken);
    }
}
```

4.  Next, the ISR checks whether a new byte has been received (by reading the `RXNE` bit of `USART2->ISR`). If a new byte is available, it is pushed into the `rxBuff` buffer and the `rxItr` iterator is incremented. 

After the desired number of bytes have been added to the buffer, the `rxDone` semaphore is given to notify `uartPrintOutTask`:

```cpp
if( USART2->ISR & USART_ISR_RXNE_Msk)
{
    uint8_t tempVal = (uint8_t) USART2->RDR;
    if(rxInProgress)
    {
        rxBuff[rxItr++] = tempVal;
        if(rxItr >= rxLen)
        {
            rxInProgress = false;
            xSemaphoreGiveFromISR(rxDone, &xHigherPriorityTaskWoken);
        }
    }
}
SEGGER_SYSVIEW_RecordExitISR();
portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
```

Don't forget to put a breakpoint in the ISR to make sure it is being called.

# startUart4Traffic

Identical to the previous example, this function sets up a DMA transfer to push data out of the UART4 Tx pin into the USART2 Rx pin.

# Performance analysis

Now, let's take a look at the performance of this driver implementation. There are several aspects to consider. Unless a transfer is complete, the ISR will normally only transfer a byte into `rxBuff`. In this case, the interrupt is fairly short, taking less than 3 us to complete:

![](img/5ba829c9-7e44-4a49-94c4-9c0caabc9a6f.png)

After all 16 bytes have been received, the ISR execution gets a bit more interesting and looks a bit more similar to the previous example:

![](img/0ff7273f-d44e-4b1b-b4a1-f67eb8937997.png)

Here are some noteworthy points from the preceding screenshot:

1.  After all the bytes have been placed into `rxBuff`, the `rxDone` semaphore is given from the ISR using `xSemaphoreGiveFromISR`.
2.  The task is unblocked after the interrupt is executed by taking the available semaphore (`xSemaphoreTake(rxDone,100)`).
3.  The exact contents of `rxBuff` are printed. Note that each line contains the entire string, rather than individual characters. This is because this implementation collects an entire buffer's worth of data before using a semaphore to indicate completion.

Finally, let's have a look at the complete tally of CPU usage:

![](img/13db11e2-8cf2-4deb-849d-d83e05492654.png)

Here are some noteworthy items from the preceding screenshot:

1.  The ISR for this implementation is using 0.34% of the CPU (instead of 1.56% when each character was pushed to a queue from inside the ISR).
2.  The FreeRTOS scheduler is using only using 0.06% of the CPU instead of 0.94% (each time items are added to queues, the scheduler runs to determine whether or not tasks should be unblocked because of the addition).
3.  The frequency of the USART2 ISR remains at 960 Hz, exactly the same as the previous examples, but now the frequency of the `print` task has been reduced to only 60 Hz, since the `uartPrint` task that only runs after 16 bytes has been transferred into `rxBuff`.

As we can see, this ISR implementation of the driver uses even fewer CPU cycles than the queue-based approach. Depending on the use case, it can be an attractive alternative. These types of drivers are commonly found in non-RTOS-based systems, where callback functions will be used instead of semaphores. This approach is flexible enough to be used with or without an RTOS by placing a semaphore in the callback. While slightly more complex, this is one of the most flexible approaches for code bases that see a large amount of reuse in different applications.

To summarize, the two variants of drivers implemented with an ISR so far have been the following:

*   **A queue-based driver**: Delivers incoming data to tasks by pushing received data into a queue one character at a time.
*   **A buffer-based driver**: Delivers incoming data to a single buffer that is pre-allocated by the calling function.

On the surface, it may seem silly to have two different implementations that both take incoming data from a peripheral and present it to the higher layers of code. It is important to realize these two different variants of a driver for the same hardware vary both in their implementation, efficiency, and ultimately, the interface provided to higher-level code. They may both be moving bytes from the UART peripheral, but they provide higher-level code with drastically different programming models.  These different programming models are each suited to solving different types of problems. 

Next, we'll look at how another piece of hardware inside the MCU can be used to lighten the burden on the CPU when moving large amounts of data.

# Creating DMA-based drivers 

We saw that, compared to a polled approach, the interrupt-based driver is considerably better in terms of CPU utilization. But what about applications with a high data rate that require millions of transfers per second? The next step in improved efficiency can be obtained by having the CPU involved as little as possible by pushing most of the work for transferring data around onto specialized peripheral hardware within the MCU. 

A short introduction to DMAwas covered in [Chapter 2](84f04852-827d-4e79-99d7-6c954ba3e93c.xhtml), *Understanding RTOS Tasks*, in case you need a refresher before diving into this example.

In this example, we'll work through creating a driver using the same buffer-based interface as the interrupt-based driver. The only difference will be the use of DMA hardware to transfer bytes out of the peripheral's read data register (`RDR`) and into our buffer. Since we already have a good handle on configuring the USART2 peripheral from our other drivers, the first order of business for this variant is to figure out how to get data from `USART2->RDR` to the DMA controller and then into memory.

# Configuring DMA peripherals

STM32F767 has two DMA controllers. Each controller has 10 channels and 8 streams to map DMA requests from one location in the MCU to another. On the STM32F767 hardware, streams can do the following:

*   Can be thought of as a way to *flow* data from one address to another
*   Can transfer data from peripherals to RAM or RAM to peripherals
*   Can transfer data from RAM to RAM
*   Can only transfer data between two points at any given moment in time

Each stream has up to 10 channels for mapping a peripheral register into a given stream. In order to configure the DMA controller to handle requests from the USART2 `receive`, we'll reference table 27 from the *STM32F7xx **RM0410* reference manual:

![](img/a36bd6b0-a7a7-423d-aa0b-30625a0871e0.png)

In this table, we can see that DMA1 Channel 4, Stream 5 is the appropriate setup to use to handle requests from `USART2_RX`. If we were also interested in handling requests for the transmit side, Channel 4, Stream 6 would also need to be set up.

Now that we know the channel and stream numbers, we can add some initialization code to set up the DMA1 and USART2 peripherals:

*   `DMA1_Stream5` will be used to transfer data from the `receive` data register of USART2 directly into a buffer in RAM.
*   `USART2` will not have interrupts enabled (they are not needed since DMA will perform all transfers from the peripheral register to RAM).
*   `DMA1_Stream5` will be set up to trigger an interrupt after the entire buffer has been filled.

The next few snippets are from the `setupUSART2DMA` function in `Chapter_10/src/mainUartDMABuff.c`:

1.  First, the clock to the DMA peripheral is enabled, interrupt priorities are set up, and the interrupts are enabled in the NVIC:

```cpp
void setupUSART2DMA( void )
{
  __HAL_RCC_DMA1_CLK_ENABLE();

  NVIC_SetPriority(DMA1_Stream5_IRQn, 6);
  NVIC_EnableIRQ(DMA1_Stream5_IRQn);
```

2.  Next, the DMA stream is configured by filling out a `DMA_HandleTypeDef` struct (`usart2DmaRx`) and using `HAL_DMA_Init()`:

```cpp
  HAL_StatusTypeDef retVal;
  memset(&usart2DmaRx, 0, sizeof(usart2DmaRx));
  usart2DmaRx.Instance = DMA1_Stream5; //stream 5 is for USART2 Rx

  //channel 4 is for USART2 Rx/Tx
  usart2DmaRx.Init.Channel = DMA_CHANNEL_4;

  //transfering out of memory and into the peripheral register
  usart2DmaRx.Init.Direction = DMA_PERIPH_TO_MEMORY;
  usart2DmaRx.Init.FIFOMode = DMA_FIFOMODE_DISABLE; //no FIFO

  //transfer 1 at a time
  usart2DmaRx.Init.MemBurst = DMA_MBURST_SINGLE; 
  usart2DmaRx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;

  //increment 1 byte at a time
  usart2DmaRx.Init.MemInc = DMA_MINC_ENABLE;

  //flow control mode set to normal
  usart2DmaRx.Init.Mode = DMA_NORMAL; 

  //write 1 at a time to the peripheral
  usart2DmaRx.Init.PeriphBurst = DMA_PBURST_SINGLE; 

  //always keep the peripheral address the same (the RX data
  //register is always in the same location)
  usart2DmaRx.Init.PeriphInc = DMA_PINC_DISABLE;

  usart2DmaRx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;

  usart2DmaRx.Init.Priority = DMA_PRIORITY_HIGH;
  retVal = HAL_DMA_Init(&usart2DmaRx);
  assert_param( retVal == HAL_OK );

  //enable transfer complete interrupts
  DMA1_Stream5->CR |= DMA_SxCR_TCIE; 

  //set the DMA receive mode flag in the USART
  USART2->CR3 |= USART_CR3_DMAR_Msk;
```

`HAL` initialization provides some sanity checking on the values passed to it. Here's a highlight of the most immediately relevant portions:

*   `DMA1_Stream5` is set as the instance. All calls that use the `usart2DmaRx` struct will reference stream `5`.
*   Channel `4` is attached to stream `5`.
*   Memory incrementing is enabled. The DMA hardware will automatically increment the memory address after a transfer, filling the buffer.
*   The peripheral address is not incremented after each transfer—the address of the USART2 receive data register (`RDR`) doesn't ever change.
*   The `transfer complete` interrupt is enabled for `DMA1_Stream5`.
*   USART2 is set up for `DMA receive mode`. It is necessary to set this bit in the USART peripheral configuration to signal that the peripheral's receive register will be mapped to the DMA controller.

Additional details about how this struct is used can be found by looking at the `DMA_HandleTypeDef` struct definition in `stm32f7xx_hal_dma.h` (line 168) and `HAL_DMA_Init()` in `stm32f7xx_hal_dma.c` (line 172). Cross-reference the registers used by the HAL code with section 8 (page 245) in the *STM32F76xxx RM0410* reference manual. This same technique is often most productive for understanding *exactly* what the `HAL` code is doing with individual function parameters and struct members.

Now that the initial DMA configuration is done, we can explore a few different interrupt implementations using DMA instead of interrupts.

# A buffer-based driver with DMA 

Here's an implementation of a driver with identical functionality to the one in the *A buffer-based driver* section. The difference is the DMA version of the driver doesn't interrupt the application every time a byte is received. The only `interrupt` generated is when the entire transfer is complete. To realize this driver, we only need to add the following ISR:

```cpp
void DMA1_Stream5_IRQHandler(void)
{
  portBASE_TYPE xHigherPriorityTaskWoken = pdFALSE;
  SEGGER_SYSVIEW_RecordEnterISR();

  if(rxInProgress && (DMA1->HISR & DMA_HISR_TCIF5))
 {
 rxInProgress = false;
 DMA1->HIFCR |= DMA_HIFCR_CTCIF5;
 xSemaphoreGiveFromISR(rxDone, &xHigherPriorityTaskWoken);
 }
  SEGGER_SYSVIEW_RecordExitISR();
  portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}
```

The significant portions of the driver are in bold. If a reception is in progress (based on the value of `rxInProgress` and the transmit complete flag, `DMA_HISR_TCIF5`, the following takes place:

*   The DMA interrupt flag is cleared.
*   The `rxDone` semaphore is given.

This is all that is required when using DMA-based transfers since the DMA controller does all of the bookkeeping associated with the buffer. At this point, the rest of the code functions in an identical way to the `interrupt` version (the only difference is that less CPU time is spent servicing interrupts).

# Performance analysis

Let's take a look at the performance of the DMA-based implementation compared to the interrupt-driven approach:

![](img/ff5bfae7-0dec-4e40-9edb-58d00bcffe90.png)

This time around, we can make the following observations about the overall system behavior:

1.  The (DMA) ISR is now consuming < 0.1% of CPU cycles at 9,600 baud.
2.  The Scheduler CPU's consumption is still very low.
3.  The frequency of the ISR has been reduced to only 60 Hz (from 960 Hz). This is because, rather than creating an interrupt for every byte, there is only an interrupt generated at the end of the transfer of 16 bytes. The Idle task is being context-switched significantly less often as well. Although it seems trivial with these simple examples, excessive context-switching can become a very real problem in large applications with many tasks and interrupts.  

The overall flow is similar to that of the interrupt buffer-based approach, with the only difference being that there is only a single ISR executed when the entire transfer is complete (instead of one interrupt for each byte transferred):

![](img/99336104-c2df-4bba-9049-a32f2313ccf2.png)

From the preceding screenshot, we can observe the following:

1.  The DMA ISR is executed once (after all 16 bytes are transferred into the buffer). A semaphore is shown by the tick mark that arrow 1 is pointing to in the screenshot.

2.  The ISR wakes up the blocked `uartPrint` function. Arrow 2 is pointing to where the semaphore is taken. 
3.  The two i infoboxes show where the console print messages are generated (~35 and 40 us after the final byte has been received). The remainder of the time this task spends is on re-initializing the buffer and setting up the next transfer.

Here is a wider view of all of the processor activity. Notice that the only activity occurs approximately once every 16 ms (after all the bytes have been transferred into memory):

![](img/2c06339b-87a6-4308-90c0-cca4b32f7e05.png)

The real capability of a fully DMA-based approach is most valuable when transferring large amounts of data very quickly. The following example shows the same dataset (only 16 bytes) transferred at 256,400 baud (the fastest that could be reliably achieved without error due to poor signal integrity).

The baud rate can be easily changed in the examples by modifying `#define BAUDRATE`  in `main<exampleame>.c`. They are configured so that a single change will modify both the USART2 and UART4 baud rates.

The following is an example of transfers being made at 256,000 baud. A new set of 16 bytes is available in the buffer, approximately every 624 µS:

![](img/9903f26b-f4aa-4047-a4d1-90c6f33ebf80.png)

By increasing the baud rate from 9,600 to 256,000, our CPU usage has increased from around 0.5% to around 11%. This is in line with the 26x increase in baud rate—all of the function's calls are proportionate to the baud rate:

![](img/09e72b29-d689-42e0-b687-40eacf3fe51e.png)

Notice the following:

*   The DMA interrupt consumes 2.29%.
*   Our `uartPrint` task is the highest consumer of CPU cycles (a little over 6%).

Even though we've proved to ourselves that it is possible to efficiently transfer data quickly by using DMA, this current setup doesn't have the same convenience that the interrupt-driven queue solution did. Tasks rely on entire blocks to be transferred, rather than using a queue. This might be fine or might be an inconvenience, depending on what the goals of the higher-level code are. 

Character-based protocols will tend to be easier to implement when written on top of a queue-based driver API, rather than a buffer-based driver API (such as the one we've implemented here). However, we saw in the *Queue-based driver* section that queues become computationally expensive very quickly. Each byte took around 30 us to be added to the queue. Transferring data at 256,000 baud would consume most of the available CPU in the UART ISR alone (a new byte is received every 40 us and it takes 30 us to process).

In the past, if you really needed to implement a character-oriented driver, you could roll your own highly efficient ring buffer implementation and feed it directly from low-level ISRs (bypassing most of the FreeRTOS primitives to save time).  However, as of FreeROTS 10, there is another alternative—stream buffers.

# Stream buffers (FreeRTOS 10+)

Stream buffers combine the convenience of a queue-based system with the speed closer to that of the raw buffer implementations we created previously. They have some flexibility limitations that are similar to the limitations of task notification systems compared to semaphores. *Stream buffers can only be used by one sender and one receiver at a time.* Otherwise, they'll need external protection (such as a mutex), if they are to be used by multiple tasks.

The programming model for stream buffers is very similar to queues, except that instead of functions being limited to queueing one item at a time, they can queue multiple items at a time (which saves considerable CPU time when queuing blocks of data). In this example, we'll explore stream buffers through an efficient DMA-based circular buffer implementation for UART reception.

The goals of this driver example are the following:

*   Provide an easy-to-use character-based queue for users of the driver.
*   Maintain efficiency at high data rates.
*   Always be ready to receive data.

So, let's begin!

# Using the stream buffer API

First, let's take a look at an example of how the stream buffer API will be used by `uartPrintOutTask` in this example. The following excerpts are from `mainUartDMAStreamBufferCont.c`.

Here's a look at the definition of `xSttreamBufferCreate()`:

```cpp
StreamBufferHandle_t xStreamBufferCreate( size_t xBufferSizeBytes,
                                          size_t xTriggerLevelBytes);
```

Note the following in the preceding code:

*   `xBufferSizeBytes` is the number of bytes the buffer is capable of holding.
*   `xTriggerLevelBytes` is the number of bytes that need to be available in the stream before a call to `xStreamBufferReceive()` will return (otherwise, a timeout will occur).

The following example code sets up a stream buffer:

```cpp
#define NUM_BYTES 100
#define MIN_NUM_BYTES 2
StreamBufferHandle_t rxStream = NULL;
rxStream = xStreamBufferCreate( NUM_BYTES , MIN_NUM_BYTES);
assert_param(rxStream != NULL);
```

In the preceding snippet, we can observe the following:

*   `rxStream` is capable of holding `NUM_BYTES` (100 bytes).
*   Each time a task blocks data from being added to the stream, it won't be unblocked until at least `MIN_NUM_BYTES` (2 bytes) are available in the stream. In this example, calls to `xStreamBufferReceive` will block until a minimum of 2 bytes are available in the stream (or a timeout occurs).
*   If using the FreeRTOS heap, be sure to check that there is enough space for the allocation of the stream buffer by checking the returned handle isn't `NULL`.

The function for receiving data from a stream buffer is `xStreamBufferReceive()`:

```cpp
size_t xStreamBufferReceive( StreamBufferHandle_t xStreamBuffer,
                             void *pvRxData,
                             size_t xBufferLengthBytes,
                             TickType_t xTicksToWait );
```

Here is a straightforward example of receiving data from a stream buffer:

```cpp
void uartPrintOutTask( void* NotUsed)
{
    static const uint8_t maxBytesReceived = 16;
    uint8_t rxBufferedData[maxBytesReceived];

    //initialization code omitted for brevity
    while(1)
    {
        uint8_t numBytes = xStreamBufferReceive(  rxStream,
 rxBufferedData,
 maxBytesReceived,
 100 );
        if(numBytes > 0)
        {
          SEGGER_SYSVIEW_PrintfHost("received: ");
          SEGGER_SYSVIEW_Print((char*)rxBufferedData);
        }
        else
        {
          SEGGER_SYSVIEW_PrintfHost("timeout");
 ...
```

In the preceding snippet, note the following:

*   `rxStream`: The pointer/handle to `StreamBuffer`.
*   `rxBufferedData`: The local buffer that bytes will be copied into.
*   `maxBytesReceived`: The maximum number of bytes that will be copied into `rxBufferedData`.
*   The timeout is `100` ticks (`xStreamBufferReceive()` will return after at least `xTriggerLevelBytes` (`2` in this example) are available or 100 ticks have elapsed).

Calls to `xStreamBufferReceive()` behave in a similar way to a call to `xQueueReceive()` in that they both block until data is available. However, a call to `xStreamBufferReceive()` will block until the minimum number of bytes (defined when calling `xStreamBufferCreate()`) or the specified number of ticks has elapsed.

In this example, the call to `xStreamBufferReceive()` blocks until one of the following conditions is met:

*   The number of bytes in the buffer exceeds `MIN_NUM_BYTES` (`2` in this example). If more bytes are available, they will be moved into `rxBufferedData`—but only up to the `maxBytesReceived` bytes (`16` in this example).
*   A timeout occurs. All available bytes in the stream are moved into `rxBufferedData` .  The exact number of bytes placed into `rxBufferedData` is returned by `xStreamBuffereReceive() -` ( `0` or `1` in this example).

Now that we have a good idea of what the receiving side looks like, let's look at some of the details of the driver itself.

# Setting up double-buffered DMA

As we saw earlier, using DMA can be very beneficial for reducing CPU usage (versus interrupts). However, one of the features that wasn't covered in the last example was continuously populating a queue (the driver required block-based calls to be made before data could be received). The driver in this example will transfer data into the stream buffer constantly, without requiring any intervention from the code calling it. That is, the driver will always be receiving bytes and pushing them into the stream buffer. 

Always receiving data presents two interesting problems for a DMA-based system:

*   How to deal with **roll-over**—when a buffer has been completely filled and high-speed data could still be coming in.
*   How to terminate transfers before a buffer is completely filled. DMA transfers typically require the number of bytes to be specified before the transfer starts. However, we need a way to stop the transfer when data has stopped being received and copy that data into the stream buffer.

DMA double buffering will be used to ensure our driver will always be able to accept data (even when a single buffer has been filled). In the previous example, a single buffer was filled and an interrupt was generated, then the data was operated on directly before restarting the transfer. With double buffering, a second buffer is added. After the DMA controller fills the first buffer, it automatically starts filling the second buffer:

![](img/02139e92-a978-4408-b468-da094f33ff01.png)

After the first buffer is filled and the interrupt is generated, the ISR can safely operate on data in the first buffer, `rxData1`, while the second buffer, `rxData2`, is filled. In our example, we're transferring that data into the FreeRTOS stream buffer from inside the ISR. 

It is important to note that `xStreamBufferSendFromISR()` adds a *copy* of the data to the stream buffer, not a reference. So, in this example, as long as the DMA ISR's call to `xStreamBufferSendFromISR()` executes before `rxData2` has been filled, data will be available with no loss. This is unlike traditional **bare-metal** double-buffer implementations since higher-level code making calls to `xStreamBufferReceive() ` isn't required to extract data from `rxData1` before  `rxData2` is filled. It only needs to call `xStreamBufferReceive()` before the stream buffer has been filled:

![](img/17705c53-8c3d-4ee4-8bd1-494fa6b123eb.png)

Even if you're programming for an MCU without an explicit **double-buffer mode**, most DMA controllers will have a **circular** mode with **half-transfer** and **full-transfer** interrupts. In this case, the same functionality can be achieved by generating an interrupt after each half of the buffer is filled.

The secondary buffer, `rxData2`, is set up by writing its address to the `DMA_SxM1AR` register (some casting is required to keep the compiler from complaining too loudly that we're writing a pointer to a 32-bit memory address):

```cpp
//setup second address for double buffered mode
DMA1_Stream5->M1AR = (uint32_t) rxData2;
```

Interestingly enough, STM HAL doesn't support double-buffer mode directly. In fact, calls to `HAL_DMA_Start` explicitly disable the mode. So, some manual setup with registers is required (after letting `HAL` take care of most of the leg work):

```cpp
//NOTE: HAL_DMA_Start explicitly disables double buffer mode
// so we'll explicitly enable double buffer mode later when
// the actual transfer is started
if(HAL_DMA_Start(&usart2DmaRx, (uint32_t)&(USART2->RDR), (uint32_t)rxData1,
                 RX_BUFF_LEN) != HAL_OK)
{
    return -1;
}

//disable the stream and controller so we can setup dual buffers
__HAL_DMA_DISABLE(&usart2DmaRx);
//set the double buffer mode
DMA1_Stream5->CR |= DMA_SxCR_DBM;
//re-enable the stream and controller
__HAL_DMA_ENABLE(&usart2DmaRx);
DMA1_Stream5->CR |= DMA_SxCR_EN;
```

After the DMA stream is enabled, the UART is enabled, which will start transfers (this is identical to the previous examples).

# Populating the stream buffer

The stream buffer will be populated from inside the DMA ISR:

```cpp
void DMA1_Stream5_IRQHandler(void)
{
    uint16_t numWritten = 0;
    uint8_t* currBuffPtr = NULL;
    portBASE_TYPE xHigherPriorityTaskWoken = pdFALSE;
    SEGGER_SYSVIEW_RecordEnterISR();

    if(rxInProgress && (DMA1->HISR & DMA_HISR_TCIF5))
    {
        if(DMA1_Stream5->CR & DMA_SxCR_CT)
            currBuffPtr = rxData1;
        else
            currBuffPtr = rxData2;

        numWritten = xStreamBufferSendFromISR(  rxStream,
                                                currBuffPtr,
                                                RX_BUFF_LEN,
                                                &xHigherPriorityTaskWoken);
        while(numWritten != RX_BUFF_LEN);

        DMA1->HIFCR |= DMA_HIFCR_CTCIF5;
    }
    SEGGER_SYSVIEW_RecordExitISR();
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}
```

Here are some of the more noteworthy items in this ISR:

*   `if(rxInProgress && (DMA1->HISR & DMA_HISR_TCIF5))`: This line guards against the stream buffer being written to before the scheduler is started. Even if the ISR was to execute before the scheduler was started, `rxInProgress` wouldn't be true until after everything was initialized. Checking the transmit complete flag, `DMA_HISR_TCIF5`, guarantees that a transfer has, indeed, completed (rather than entering the ISR because of an error).
*   `DMA1_Stream5->CR & DMA_SxCR_CT`: Checks the current target bit. Since this bit indicates which target buffer (`DMA_SxM0AR`  or `DMA_SxM1AR`) is currently being **filled** by the DMA controller, we'll take the other and push that data into the stream buffer.
*   The call to `xStreamBufferSendFromISR` pushes the entirety of `rxBuff1` or `rxBuff2` (each of an `RX_BUFF_LEN` length) into `rxStream` in one go.

A few things to remember are as follows:

*   Data is being transferred to the stream by value (not reference). That is, FreeRTOS is using `memcpy` to make a copy of all of the data moving into the stream buffer (and again when removing data). The larger the buffer, the more time it will take to copy—additional RAM will also be used.
*   Instead of performing the copy inside the interrupt, under certain circumstances, it may be preferable to signal a semaphore or task notification and perform the copy in a high-priority task instead—for example, if a large buffer is being filled. However, you'll need to guarantee that other interrupts don't starve the task performing the `xStreamBufferSend` or data will be lost.
*   There are trade-offs when using DMA. Larger buffers mean fewer interruptions to transfer data, but they also mean an increase in latency. The larger the buffer, the longer the data will sit in the buffer before being processed.
*   This implementation is only well suited to continuous data streams—if the data stream stops, the last DMA transfer will never complete.

This approach to pushing received data from a peripheral into memory works very well when data is continuously streaming. It can also work extremely well for the reception of messages with a known number of bytes. However, there are some ways to improve it.

# Improving the stream buffer

In order to deal with an intermittent data stream, there are two possible approaches (for this specific setup):

*   The USART peripheral on this MCU is capable of detecting an "idle line" and generating an interrupt by setting the `USART_CR1:IDLEE` bit when an idle line is detected.
*   The USART peripheral also has a `receive` timeout that can also generate an interrupt after no start bits have been detected for a specified number of bit times (0-16,777,215). 
    *   This timeout is specified in the `USART_RTOR:RTO[23:0]` register.
    *   The feature can be enabled with `USART_CR2:RTOEN` and the interrupts can be enabled with  `USART_CR1:RTOIE`.

Either of these features could be used to generate a USART interrupt, cut the DMA transfer short, and transfer the data to the stream buffer.

For extremely high baud rates, care needs to be taken when using the idle line approach because the number of interrupts generated is only capped by the baud rate. If there is inter-character spacing (idle time between each character being sent), you'll wind up with an interrupt-driven approach (with even more overhead than normal).

On the other hand, using the `receive` timeout feature means additional latency before processing the incoming data. As usual, there is no *one-size-fits-all* solution here.

# Analyzing the performance

So, how does this DMA stream buffer implementation compare to the ISR-based queue implementation? Well, on one hand, there is no comparison... *the ISR based implementation doesn't work at 256,400 baud*. At this baud rate, a new character is received every 39 uS. With the ISR taking around 18 us to execute, we simply don't have enough time to also run `printUartTask()` reliably without dropping data:

![](img/a06c4874-d4bf-439c-ad97-ae7f348fc6ed.png)

Notice that there is absolutely no time spent on the Idle task—the CPU is completely consumed by attempting to keep up with the incoming data from UART2.

As you can see in the following screenshot, data is occasionally dropped when the processor is set up to receive data at 256,400 baud using an ISR that executes once per character:

![](img/b455799d-dae7-45c5-8b44-5b0a964e9d05.png)

Now, for comparison, here's the (nearly) equivalent implementation using stream buffers and DMA:

![](img/c7c73613-e70d-4ef6-bcd9-679451cc9d15.png)

The combination of stream buffers and DMA has freed up quite a bit of the CPU time; the queue-based ISR implementation consumed > 100% of the CPU. As we can see in the following processing breakdown, the total CPU usage for a stream buffer using DMA is around 10%:

![](img/32c61441-bf2d-4c51-8527-25d5e6805684.png)

Note the following:

*   The DMA-/stream buffer-based solution leaves nearly 90% of the CPU cycles available for other tasks.
*   More time is being spent printing debug statements (and pulling bytes off the queue) than servicing the DMA ISR.
*   The multi-byte stream buffer transactions also eliminate a large amount of context switching (notice the scheduler is only utilizing around 1% CPU), which will leave more contiguous time for other processing tasks.

So, now that we've worked through a very simple example of each driver type, which one should you implement?

# Choosing a driver model

Selecting the *best* driver for a given system depends on several different factors:

*   How is the calling code designed?
*   How much delay is acceptable?
*   How fast is data moving?
*   What type of device is it?

Let's answer these questions one by one.

# How is the calling code designed?

What is the intended design of higher-level code using the driver? Will it operate on individual characters or bytes as they come in? Or does it make more sense for the higher-level code to batch transfers into blocks/frames of bytes? 

Queue-based drivers are very useful when dealing with unknown amounts (or streams) of data that can come in at any point in time. They are also a very natural fit for code that processes individual bytes—`uartPrintOutTask` was a good example of this:

```cpp
 while(1)
 {
     xQueueReceive(uart2_BytesReceived, &nextByte, portMAX_DELAY);
    //do something with the byte received
     SEGGER_SYSVIEW_PrintfHost("%c", nextByte);
 }
```

While ring-buffer implementations (such as the one in the preceding code) are perfect for streamed data, other code naturally gravitates toward operating on blocks of data. Say, for example, our high-level code is meant to read in one of the structures defined in [Chapter 9](495bdcc0-2a86-4b22-9628-4c347e67e49e.xhtml), *Intertask Communication*, over a serial port.

The following excerpt is from `Chapter_9/MainQueueCompositePassByValue.c`:

```cpp
typedef struct
{
    uint8_t redLEDState : 1;     
    uint8_t blueLEDState : 1;
    uint8_t greenLEDState : 1;
    uint32_t msDelayTime;
}LedStates_t;
```

Rather than operate on individual bytes, it is very convenient for the receiving side to pull in an instance of the entire struct at once. The following code is designed to receive an entire copy of `LedStates_t` from a queue. After the struct is received, it can be operated on by simply referencing members of the struct, such as checking `redLEDState`, in this example:

```cpp
LedStates_t nextCmd;
while(1)
{
    if(xQueueReceive(ledCmdQueue, &nextCmd, portMAX_DELAY) == pdTRUE)
    {
        if(nextCmd.redLEDState == 1)
            RedLed.On();
        else
             . . .
```

This can be accomplished by **serializing** the data structure and passing it over the communication medium. Our `LedStates_t` struct can be serialized as a block of 5 bytes. All three red, green, and blue state values can be packed into 3 bits of a byte and the delay time will take 4 bytes:

![](img/9a624631-0150-4a45-986d-39bf6189d06c.png)

Serialization is a broad topic in itself. There are trade-offs to be made for portability, ease of use, code fragility, and speed. A discussion on all of these points is outside the scope of this chapter. Details of endianness and the *best* way of serializing/deserializing this particular data structure have been purposely ignored in the diagram. The main takeaway is that the struct can be represented by a block of 5 bytes.

In this case, it makes sense for the underlying peripheral driver to operate on a buffer of 5 bytes, so a buffer-based approach that groups a transfer into a block of 5 bytes is more natural than a stream of bytes. The following pseudo-code outlines an approach based on the buffer-based driver we wrote in the previous section:

```cpp
uint8_t ledCmdBuff[5];
startReceiveInt(ledCmdBuff, 5);
//wait for reception to complete
xSemaphoreTake(cmdReceived, portMAX_DELAY);
//populate an led command with data received from the serial port
LedStates_t ledCmd = parseMsg(ledCmdBuff);
//send the command to the queue
xQueueSend(ledCmdQueue, &ledCmd, portMAX_DELAY);
```

In a situation like the previous one, we have covered two different approaches that can provide efficient implementations:

*   A buffer-based driver (receiving 5 bytes at a time)
*   A stream buffer (the receiving side can be configured to acquire 5 bytes at a time)

FreeRTOS message buffers could also be used instead of a stream buffer to provide a more flexible solution. Message buffers are built on top of stream buffers, but have a more flexible blocking configuration. They allow different message sizes to be configured per `receive` call, so the same buffer can be used to group receptions into a size of 5 bytes (or any other desired size) each time `xMessageBufferReceive` is called. With stream buffers, the message size is rigidly defined when creating the stream buffer by setting the `xTriggerLevelBytes` parameter in `xStreamBufferCreate`.  Unlike stream buffers, message buffers will only return full messages, not individual bytes.

# How much delay is acceptable?

Depending on the exact function being implemented, minimal delay may be desired. In this case, buffer-based implementations can sometimes have a slight advantage. They allow the calling code to be set up as an extremely high priority, without causing significant context switching in the rest of the application. 

With a buffer-based setup, after the last byte of a message is transferred, the task will be notified and immediately run. This is better than having the high-priority task perform byte-wise parsing of the message since it will be interrupting other tasks continually each time a byte is received. In a byte-wise queue-based approach, the task waiting on the queue would need to be set to a very high priority if the incoming message was extremely important. This combination causes quite a bit of task context switching versus a buffer approach, which only has a single semaphore (or direct task notification) when the transfer is finished.

Sometimes, timing constraints are so tight neither queues nor an entire block transfer may be acceptable (bytes might need to be processed as they come in). This approach will sometimes eliminate the need for intermediate buffers as well. A fully custom ISR can be written in these cases, but it won't be easily reused. Try to avoid lumping **business logic** (application-level logic not immediately required for servicing the peripheral) into ISRs whenever possible. It complicates testing and reduces code reuse. After a few months (or years) of writing code like this, you'll likely notice that you've got dozens of ISRs that look *almost* the same but behave in subtlety different ways, which can make for buggy systems when modifications to higher-level code are required.

# How fast is data moving?

While extremely convenient, queues are a fairly expensive way to pass individual bytes around a system. Even an interrupt-based driver has limitations on how long it has to deal with incoming data. Our example used a meager 9,600 baud transfer. Individual characters were transferred into the queue within 40 us of being received, but what happens if the baud rate is 115,200 baud? Now, instead of having around 1 character per millisecond, each character would need to be added to the queue in less than 9 us. A driver that takes 40 us per interrupt isn't going to be acceptable here, so using a simple queue approach isn't a viable option.

We saw that the stream buffer implementation with DMA was a viable solution in place of a queue. Using some type of double-buffering technique for high-speed, continuous streams of data is critical. This becomes an especially convenient technique when coupled with a highly efficient RTOS primitive, such as stream buffers or message buffers.

Interrupts and DMA-based drivers that moved data directly into a **raw** memory buffer are also quite viable when speeds are high, but they don't have the convenience of a queue-like interface.

# What type of device are you interfacing?

Some peripherals and external devices will naturally lean toward one implementation or another. When receiving asynchronous data, queues are a fairly natural choice because they provide an easy mechanism for constantly capturing incoming data. UARTs, USB virtual comms, network streams, and timer captures are all very naturally implemented with a byte-wise queue implementation (at least at the lowest level).

Synchronous-based devices, such as a **serial peripheral interface** (**SPI**) and **Inter-Integrated Circuit** (**I2C**), are easily implemented with block-based transfers on the master side since the number of bytes is known ahead of time (the master needs to supply the clock signal for both bytes sent and bytes received).

# When to use queue-based drivers

Here are some cases where it is an advantage to use a queue as the interface of a driver:

*   When the peripheral/application needs to receive data of an unknown length
*   When data must be received asynchronously to requests
*   When a driver should receive data from multiple sources without blocking the caller
*   When data rates are sufficiently slow to allow a minimum of 10's of µS per interrupt (when being implemented on the hardware, in this example)

# When to use buffer-based drivers

Some cases where raw buffer-based drivers are extremely useful are as follows:

*   When large buffers are required because large amounts of data will be received at once
*   During transaction-based communication protocols, especially when the length of the received data is known in advance

# When to use stream buffers

Stream buffers provide speed closer to that of raw buffers, but with the added benefit of providing an efficient queue API. They can generally be used anywhere a standard queue would be used (as long as there is only one consumer task). Stream buffers are also efficient enough to be used in place of raw buffers, in many cases. As we saw in the `mainUartDMAStreamBufferCont.c` example, they can be combined with circular DMA transfers to provide true continuous data capture, without using a significant number of CPU cycles.

These are just some of the considerations you'll likely face when creating drivers; they are mainly aimed at communication peripherals (since that's what our examples covered). There are also some considerations to be made when choosing to use third-party libraries and drivers.

# Using third-party libraries (STM HAL)

If you've been following along closely, you may have noticed a few things:

*   STM HAL (the vendor-supplied hardware abstraction layer) is used for initial peripheral configuration. This is because HAL does a very good job of making peripheral configuration easy. It is also extremely convenient to use tools such as STM Cube to generate some boilerplate code as a point of reference when first interacting with a new chip.
*   When it is time to implement details of interrupt-driven transactions, we've been making a lot of calls directly to MCU peripheral registers, rather than letting HAL manage transactions for us. There were a couple of reasons for this:
    *   We wanted to be closer to the hardware to get a better understanding of how things were really working in the system.
    *   Some of the setups weren't directly supported by HAL, such as DMA double buffering.

In general, you should use as much vendor-supplied code as you (or your project/company) are comfortable with. If the code is well written and works reliably, then there *usually* aren't too many arguments for *not* using it.  

That being said, here are some potential issues when using vendor-supplied drivers:

*   They may use polling instead of interrupts or DMA.
*   Tying into interrupts may be cumbersome or inflexible.
*   There is potentially *lots* of extra overhead since many chips/use cases are likely covered by drivers (they need to solve *everyone's* problems, not just yours).
*   It might take longer to fully grasp and understand a complex API than working directly with the peripheral hardware (for simple peripherals).

The following are examples of when to write **bare-metal** drivers:

*   When a vendors driver is broken/buggy
*   When speed matters
*   When an exotic configuration is required
*   As a learning exercise

Ideally, transitioning between third-party drivers and your own drivers would be perfectly seamless. If it isn't, it means that the higher-level code is tightly coupled to the hardware. This tight coupling is perfectly acceptable for sufficiently small *one-off* and *throw-away* projects, but if you're attempting to develop a code base for the long term, investing in creating a loosely coupled architecture will pay dividends. Having loose coupling (eliminating dependencies between the exact driver implementation and higher-level code) also provides flexibility in the implementation of the individual components. Loose coupling ensures transitioning between custom drivers and third-party drivers doesn't necessitate a major rewrite of high-level code. Loose coupling also makes testing small portions of the code base in isolation possible—see [Chapter 12](8e78a49a-1bcd-4cfe-a88f-fb86a821c9c7.xhtml)*, Tips on Creating Well-Abstracted Architecture*, for details.

# Summary

In this chapter, we introduced three different ways of implementing low-level drivers that interface with hardware peripherals in the MCU. Interrupts and polled- and DMA-based drivers were all covered through examples and their performance was analyzed and compared using SEGGER SystemView. We also covered three different ways that FreeRTOS can interact with ISRs: semaphores, queues, and stream buffers. Considerations for choosing between the implementation options were also discussed, as well as when it is appropriate to use third-party peripheral drivers (STM HAL) and when "rolling your own" is best.

To get the most out of this chapter, you're encouraged to run through it on actual hardware. The development board was chosen (in part) with the hope that you might have access to Arduino shields. After running through the examples, an excellent next step would be to develop a driver for a shield or another piece of real-world hardware. 

This chapter was really just the tip of the iceberg when it comes to driver implementation. There are many additional approaches and techniques that can be used when creating efficient implementations, from using different RTOS primitives beyond what is presented in this chapter to configuring MCU-specific functionality. Your designs don't need to be limited by what happens to be provided by a vendor.

You should now have a solid understanding of the many different ways low-level drivers can be implemented. In the next chapter, we'll take a look at how these drivers can be safely presented to higher-level code across multiple tasks. Providing easy access to drivers makes developing the final application fast and flexible.

# Questions

As we conclude, here is a list of questions for you to test your knowledge of this chapter's material. You will find the answers in the *Assessments* section of the appendix:

1.  What type of driver is more complicated to write and use?
    *   Polled
    *   Interrupt-driven
2.  True or false: In FreeRTOS, it is possible to call any RTOS function from any ISR?
    *   True
    *   False
3.  True or false: When using an RTOS, interrupts are constantly fighting the scheduler for CPU time?
    *   True
    *   False
4.  Which technique for a peripheral driver requires the fewest CPU resources when transferring large amounts of high-speed data?
    *   Polling
    *   Interrupt
    *   DMA
5.  What does DMA stand for?
6.  Name one case when using a raw buffer-based driver is *not* a good idea.

# Further reading

*   *Chapter 4* in the *RM0410 STM32F76xxx* reference manual (*USART*)
*   B1.5.4, **Exception priorities and preemption** section in the *Arm®v7-M Architecture* reference manual
*   FreeRTOS.org's explanation of CortexM priorities, at [https://www.freertos.org/RTOS-Cortex-M3-M4.html](https://www.freertos.org/RTOS-Cortex-M3-M4.html)