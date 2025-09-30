# Sharing Hardware Peripherals across Tasks

In the previous chapter, we went through several examples of creating drivers, but they were only used by a single task. Since we're creating a multi-tasking asynchronous system, a few additional considerations need to be made to ensure that the peripherals exposed by our drivers can safely be used by multiple tasks. Preparing a driver for use by multiple tasks requires a number of additional considerations.

Accordingly, this chapter first illustrates the pitfalls of a shared peripheral in a multi-tasking, real-time environment. After understanding the problem we're trying to solve, we'll investigate potential solutions for wrapping a driver in a way that provides an easy-to-use abstraction layer that is safe to use across multiple tasks. We'll be using the STM32 USB stack to implement a **Communication Device Class** (**CDC**) to provide an interactive **Virtual COM Port** (**VPC**). Unlike the previous chapter, which took an extremely low-level approach to driver development, this chapter focuses on writing threadsafe code on top of an existing driver stack.

In a nutshell, we will cover the following topics:

*   Understanding shared peripherals
*   Introducing the STM USB driver stack 
*   Developing a StreamBuffer USB virtual COM port
*   Using mutexes for access control

# Technical requirements

To complete the hands-on experiments in this chapter, you'll require the following:

*   Nucleo F767 Dev Board
*   Micro-USB cable (x2)
*   STM32CubeIDE and source code (instructions in [Chapter 5](84a945dc-ff6c-4ec8-8b9c-84842db68a85.xhtml), *Selecting an IDE*, under the section entitled *Setting up our IDE*)
*   SEGGER JLink, Ozone, and SystemView (instructions in [Chapter 6](699daa80-06ae-4acc-8b93-a81af2eb774b.xhtml), *Debugging Tools for Real-Time Systems*)
*   STM USB virtual COM port drivers:
    *   Windows: The driver should install automatically from Windows Update ([https://www.st.com/en/development-tools/stsw-stm32102.html](https://www.st.com/en/development-tools/stsw-stm32102.html)).
    *   Linux/ macOS: These use built-in virtual COM port drivers.

*   Serial Terminal Client:
    *   Tera Term (or similar) (Windows)
    *   minicom (or similar) (Linux /macOS)
    *   miniterm.py (cross-platform serial client also included with Python modules used in *[Chapter 13](e728e173-c9b2-4bb8-91c8-ed348ccf9518.xhtml), Creating Loose Coupling with Queues*)

All source code for this chapter is available from [https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_11](https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_11).

# Understanding shared peripherals

A hardware peripheral is similar to any other shared resource. When there is a single resource with multiple tasks that need access to the resource, some sort of arbitration needs to be created to guarantee orderly access to the resource across tasks. In the previous chapter, we focused on different ways of developing low-level peripheral drivers. Some guidance as to driver selection was provided and it was suggested that the appropriate interface the driver provides should be based on how the driver was to be used in the system ([Chapter 10](dd741273-db9a-4e9a-a699-b4602e160b84.xhtml), *Drivers and ISR's,* under the section entitled *Choosing a driver model*).

Shared resources were covered conceptually in [Chapter 3](a410ddd6-10eb-4e97-965e-e390f4dc2890.xhtml),* Task Signaling and Communication Mechanisms.*

There are many different examples of sharing peripherals in real-world applications. Communication peripherals such as SPI, I2C, USARTs, and ethernet peripherals can all be used by multiple tasks simultaneously, provided the timing constraints of the application allow for it and the drivers are written in a way that provides safe concurrent access. Since all of the blocking RTOS calls can be time-bound, it is easy to detect when accessing a shared peripheral is causing timing issues.

It is important to remember that sharing a single peripheral across multiple tasks creates delays and uncertainty in timing.

In some cases where timing is critical, it is best to avoid sharing a peripheral and instead use dedicated hardware. This is part of the reason why there are multiple bus-based peripherals available, including SPI, USART's, and I2C. Even though the hardware for each of these communication buses is perfectly capable of addressing multiple devices, sometimes it is best to use a dedicated peripheral.

In other cases, a driver for a piece of hardware may be so specific that it is best to dedicate an entire peripheral to it for performance reasons. High bandwidth peripherals will typically fall into this category. An example of this would be a medium bandwidth ADC sampling thousands or tens of thousands of data points per second. The most efficient way of interacting with devices such as these is to use DMA as much as possible, transferring data from the communication bus (like SPI) directly into RAM.

# Defining the peripheral driver

This chapter provides fully fleshed-out examples of interacting with a driver in a real-world situation. A USB virtual COM port was chosen because it won't require any additional hardware, other than a second micro-USB cable.

Our goal is to make it easy to interact with the Nucleo board using USB CDC in a reasonably efficient way. Desirable features for interaction include the following:

*   The ability to easily write to a USB virtual COM port from multiple tasks.
*   Efficient event-driven execution (avoiding wasteful polling as much as possible).
*   Data should be sent over USB immediately (avoid delayed sending whenever possible).
*   Calls should be non-blocking (tasks may add data to be sent without waiting for the actual transaction).
*   Tasks may choose how long to wait for space to be available before data is dropped.

These design decisions will have several implications:

*   **Transmit timing uncertainty**: While data is queued in a non-blocking manner, the exact timing of the transfer is not guaranteed. This is not an issue for this specific example, but if this were being used for time-sensitive interactions, it could be. USB CDC isn't a great choice for something with extremely sensitive timing requirements to begin with.
*   **Trade-offs between buffer size and latency**: In order to provide sufficient space for transmitting large messages, the queue can be made larger. However, it takes longer for data to exit a large queue than a small one. If latency or timing is a consideration, this time needs to be taken into account.
*   **RAM usage**: The queue requires additional RAM, on top of what the USB buffers already require.
*   **Efficiency**: This driver represents a trade-off between ease of use and efficiency. There are effectively two buffers – the buffer used by USB and the queue. To provide ease of use, data will be copied by value *twice*, once into the queue and once into the USB transmit buffer. Depending on the required bandwidth, this could present a significant performance constraint.

First, let's take a high-level look at the STM USB device driver stack to better understand the options we have when interfacing with the STM-supplied CDC driver.

# Introducing the STM USB driver stack

STM32CubeMX was used as a starting point to generate a USB device driver stack with CDC support. Here's an overview of the significant USB source files and where they reside, relative to the root of the repository: [https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/](https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/)

1.  Low-level HAL USB files:

```cpp
DRIVERS\STM32F7XX_HAL_DRIVER\
|--Inc
|    |stm32f7xx_ll_usb.h
|--Src
     |stm32f7xx_ll_usb.c
```

The `stm32f7xx_ll_usb.c/h` files are the lowest level files, which provide access to the USB hardware peripherals. These files are used by the STM-supplied USB driver stack middleware.

2.  STM USB device stack:

```cpp
MIDDLEWARE\ST\
|--STM32_USB_Device_Library
    |----Class
    | |----CDC
    |     |----Inc
    |         | usbd_cdc.h
    | ----Src
    | usbd_cdc.c
    |----Core
        |----Inc
        |    usbd_core.h
        |     usbd_ctlreq.h
        |     usbd_def.h
        |     usbd_ioreq.h
        |----Src
               usbd_core.c
               usbd_ctlreq.c
               usbd_ioreq.c
```

The preceding files implement the core USB device and CDC class functionality. These are also supplied by STM. These provide most of the functionality required for dealing with USB transactions and enumeration.

3.  Most interaction with the USB library will take place at the CDC interface level, in the BSP folder:

```cpp
BSP\
    Nucleo_F767ZI_Init.c
    Nucleo_F767ZI_Init.h
    usbd_cdc_if.c
    usbd_cdc_if.h
    usbd_conf.c
    usbd_conf.h
    usbd_desc.c
    usbd_desc.h
    usb_device.c
    usb_device.h
```

Here's a brief description of each source file pair and its purpose. These files are the most likely files to be modified during USB development:

*   `Nucleo_F767ZI_Init.c/h`: Initialization code for the MCU, which is specific to this hardware. Functions such as clock and individual pin configuration happen here.
*   `usbd_cdc_if.c/h`: (STM Cube generated). Contains the USB device CDC interface functions. `CDC_Transmit_FS()` is used to transmit data from the MCU to the USB host (a PC in this case). `CDC_Receive_FS()` is used to receive data from the USB host.
*   `usbd_conf.c/h`: (STM Cube generated). Used to map functions and required callbacks of `stm32f7xx_hal_pcd.c` (the USB peripheral control driver) to `stm32f7xx_ll_usb.c` (the low-level USB peripheral interface driver).
*   `usbd_desc.c/h`: (STM Cube generated). USB device descriptors that are used during USB enumeration are defined here. This is where product and vendor identification numbers are defined (PID, VID). 
*   `usb_device.c/h`: (STM Cube generated). Contains the top-level function for initializing the USB stack. This file contains `MX_USB_DEVICE_Init()`, which is used to initialize the entire USB device driver stack. `MX_USB_DEVICE_Init()` should be called *after* all lower-level clock and pin initialization has been performed (`HWInit()` in `Nucleo_F767ZI_Init.c` performs this initialization).

Now that we have a general idea of how the code is structured, let's create a simple example to better understand how to interact with it.

# Using the stock CDC drivers

`mainRawCDC.c` contains a minimal amount of code to configure the MCU hardware and USB device stack. It will allow the MCU to enumerate over USB as a virtual COM port when a micro-USB cable is plugged into CN1 (and goes to a USB host such as a PC) and power is applied through CN13\. It will attempt to send two messages over USB: *test* and *message:*

1.  The USB stack is initialized by using the `MX_USB_Device_Init()` function after the hardware is fully initialized:

```cpp
int main(void)
{
  HWInit();=
 MX_USB_DEVICE_Init();
```

2.  There is a single task that outputs two strings over USB, with a forced 100 tick delay after the second transmission using a naive call to `usbd_cdc_if.c`: `CDC_Transmit_FS`:

```cpp
void usbPrintOutTask( void* NotUsed)
{
  while(1)
  {
    SEGGER_SYSVIEW_PrintfHost("print test over USB");
 CDC_Transmit_FS((uint8_t*)"test\n", 5);
    SEGGER_SYSVIEW_PrintfHost("print message over USB");
 CDC_Transmit_FS((uint8_t*)"message\n", 8);
    vTaskDelay(100);
  }
}
```

3.  After compiling and loading this application to our target board, we can observe the output of the USB port by opening a terminal emulator (Tera Term in this case). You'll likely see something similar to the following screenshot:

![](img/5731c4e7-e36c-41b1-9ec5-d9f7a7c0fa60.png)

Since we were outputting a single line containing test and then a single line containing message, we would hope that the virtual serial port would contain that same sequence, but there are multiple *test* lines that aren't always followed by a *message* line.

Watching this same application run from SystemView shows that the code is executing in the order that we would expect:

![](img/0fb6c15f-018a-42f6-ac32-75986b1a9ff9.png)

Upon closer inspection of `CDC_Transmit_FS`, we can see that there is a return value that should have been inspected. `CDC_Transmit_FS` first checks to ensure that there isn't already a transfer being performed before overwriting the transmit buffer with new data. Here are the contents of `CDC_Transmit_FS`(automatically generated by STM Cube):

```cpp
  uint8_t result = USBD_OK;
  /* USER CODE BEGIN 7 */
 USBD_CDC_HandleTypeDef *hcdc =
 (USBD_CDC_HandleTypeDef*)hUsbDeviceFS.pClassData

 if (hcdc->TxState != 0){
 return USBD_BUSY;
 }
  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, Buf, Len);
  result = USBD_CDC_TransmitPacket(&hUsbDeviceFS);
  /* USER CODE END 7 */
  return result;
```

Data will only be transmitted if there isn't already a transfer in progress (indicated by `hcdc->TxState`). So, to ensure that all of the messages are transmitted, we have a number of options here.

1.  We could simply wrap each and every call to `CDC_Transmit_FS` in a conditional statement to check whether the transfer was successful:

```cpp
int count = 10;
while(count > 0){
    count--;
      if(CDC_Transmit_FS((uint8_t*)"test\n", 5) == USBD_OK)
        break;
      else
        vTaskDelay(2);
}
```

There are several downsides to this approach:

*   *   It is slow when attempting to transmit multiple messages back to back (because of the delay between each attempt).
    *   If the delay is removed, it will be extremely wasteful of CPU, since the code will essentially poll on transmission completion.
    *   It is undesirably complex. By forcing the calling code to evaluate whether a low-level USB transaction was valid, we're adding a loop and nested conditional statements to something that could potentially be very simple. This will increase the likelihood that it is coded incorrectly and reduce readability.

2.  We could write a new wrapper based on `usbd_cdc_if.c` that uses FreeRTOS stream buffers to efficiently move data to the USB stack. This approach has a few caveats:
    *   To keep the calling code simple, we'll be tolerant of dropped data (if space in the stream buffer is unavailable).
    *   To support calls from multiple tasks, we'll need to protect access to the stream buffer with a mutex.
    *   The stream buffer will effectively create a duplicate buffer, thereby consuming additional RAM.
3.  We could use a FreeRTOS queue instead of a stream buffer. As seen in *[Chapter 10](dd741273-db9a-4e9a-a699-b4602e160b84.xhtml), Drivers and ISRs,* we would receive a performance hit when using a queue (relative to a stream buffer) since it would be moving only a single byte at a time. However, a queue wouldn't require being wrapped in a mutex when used across tasks.

The *best* solution depends on many factors (there's a list of considerations at the end of [Chapter 10](dd741273-db9a-4e9a-a699-b4602e160b84.xhtml), *Drivers and ISRs*). For this example, we'll be using a stream buffer implementation. There is plenty of room for the extra space required by the buffer. The code here is only intended to support occasional short messages, rather than a fully reliable data channel. This limitation is mainly being placed to minimize complexity to make the examples easier to read.

Let's now have a look at how options 2 and 3 look, relative to the STM HAL drivers already present:

![](img/ad3be02b-ea4a-49fc-aeb0-608c6b975f20.png)

For this driver, we'll be modifying the stubbed out HAL-generated code supplied by ST (`usbd_cdc_if.c`) as a starting point. Its functionality will be replaced by our newly created `VirtualCommDriver.c`. This will be detailed in the next section.

We'll also make a very small modification to the CDC middleware supplied by STM (`usbd_cdc.c/h`) to enable a non-polled method for determining when transfers are finished. The `USBD_CDC_HandleTypeDef` struct in `usbd_cdc.h` already has a variable named `TxState` that can be polled to determine when a transmission has completed. But, to increase efficiency, we'd like to avoid polling. To make this possible, we'll add another member to the struct – a function pointer that will be called when a transfer is complete: `usbd_cdc.h`(additions in **bold**):

```cpp
typedef struct
{
  uint32_t data[CDC_DATA_HS_MAX_PACKET_SIZE / 4U]; /* Force 32bits
                                                      alignment */
  uint8_t CmdOpCode;
  uint8_t CmdLength;
  uint8_t *RxBuffer;
  uint8_t *TxBuffer;
  uint32_t RxLength;
  uint32_t TxLength;
  //adding a function pointer for an optional call back function
 //when transmission is complete
 void (*TxCallBack)( void );
  __IO uint32_t TxState;
  __IO uint32_t RxState;
}
USBD_CDC_HandleTypeDef;
```

We'll then add the following code to `usbd_cdc.c.` (additions in bold):

```cpp
    }
    else
    {
      hcdc->TxState = 0U;
      if(hcdc->TxCallBack != NULL)
 {
 hcdc->TxCallBack();
 }
    }
    return USBD_OK;
  }
```

This addition executes the function pointed to by `TxCallBack` if it has been provided (indicated by a non-NULL value). This happens when `TxState` in the CDC struct is set to 0. `TxCallBack` was also initialized to NULL in `USBD_CDC_Init()`. 

Modifying drivers supplied by STM will make it harder to migrate between different versions of HAL. These considerations must be weighed against any advantages they provide.
NOTE: More recent versions of HAL and STMCubeIDE include support for `TxCallBack`, so this modification won't be necessary if you're starting from scratch with the latest released code from ST.

# Developing a StreamBuffer USB virtual COM port

`VirtualComDriver.c` is located in the top-level `Drivers` folder (since we're likely to use it in a future chapter). It is available here: [https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Drivers/HandsOnRTOS/](https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Drivers/HandsOnRTOS/)

First, we'll walk through each of the functions that have been created, and their purpose.

# Public functions

`VirtualComDriver.c`  currently has three publicly available functions: 

*   `TransmitUsbDataLossy`
*   `TransmitUsbData` 
*   `VirtualCommInit`

`TransmitUsbDataLossy` is simply a wrapper around a stream buffer function call. It uses an ISR-safe variant, which is guaranteed not to block (but may also not copy all data into the buffer). The number of bytes copied into the buffer is returned. In this case, it is up to the calling code to determine whether or not to finish copying data into the buffer:

```cpp
int32_t TransmitUsbDataLossy(uint8_t const* Buff, uint16_t Len)
{
  int32_t numBytesCopied = xStreamBufferSendFromISR( txStream, Buff, Len,
                                                                   NULL);
  return numBytesCopied;
}
```

`TransmitUsbData` provides a bit more convenience. It will block up to two ticks waiting for space to become available in the buffer. This is broken into two calls in case the buffer fills part way through the initial transfer. It is likely that enough space will be available 1 tick later when the second call to `xStreamBufferSend` is made. In most cases, there will be very little dropped data using this method:

```cpp
int32_t TransmitUsbData(uint8_t const* Buff, uint16_t Len)
{
  int32_t numBytesCopied = xStreamBufferSend( txStream, Buff, Len, 1);
  if(numBytesCopied != Len)
  {
    numBytesCopied += xStreamBufferSend( txStream, Buff+numBytesCopied,
                                                Len-numBytesCopied, 1);
  }
  return numBytesCopied;
}
```

`VirtualCommInit` performs all of the setup required for both the USB stack and the necessary FreeRTOS task. The stream buffer is being initialized with a trigger level of 1 to minimize the latency between when `TransmitUsbData` is called and when the data is moved into the USB stack. This value can be adjusted in conjunction with the maximum blocking time used in `xStreamBufferReceive` to achieve better efficiency by ensuring that larger blocks of data are transferred simultaneously:

```cpp
void VirtualCommInit( void )
{
  BaseType_t retVal;
  MX_USB_DEVICE_Init();
  txStream = xStreamBufferCreate( txBuffLen, 1);
  assert_param( txStream != NULL);
  retVal = xTaskCreate(usbTask, "usbTask", 1024, NULL,
             configMAX_PRIORITIES, &usbTaskHandle);
  assert_param(retVal == pdPASS);
}
```

These are all of the publicly available functions. By modifying slightly the interaction with the stream buffer, this driver can be optimized for many different use cases. The remainder of the functionality is provided by functions that aren't publicly accessible.

# Private functions

`usbTask` is a private function that takes care of the initial setup of our CDC overrides. It also monitors the stream buffer and task notifications, making the required calls to the CDC implementation provided by STM.

Before starting its main loop, there are a few items that need to be initialized:

1.  The task must wait until all of the underlying peripherals and USB stack initialization are performed. This is because the task will be accessing data structures created by the USB CDC stack:

```cpp
  USBD_CDC_HandleTypeDef *hcdc = NULL;

  while(hcdc == NULL)
  {
    hcdc = (USBD_CDC_HandleTypeDef*)hUsbDeviceFS.pClassData;
    vTaskDelay(10);
  }
```

2.  A task notification is given, provided a transmission is not already in progress. The notification is also taken, which allows for an efficient way to block in case a transfer is already in progress: 

```cpp
  if (hcdc->TxState == 0)
  {
    xTaskNotify( usbTaskHandle, 1, eSetValueWithOverwrite);
  }
  ulTaskNotifyTake( pdTRUE, portMAX_DELAY );
```

3.  `usbTxComplete` is the callback function that will be executed when a transmission is finished. The USB CDC stack is ready to accept more data to be transmitted. Setting the `TxCallBack` variable to `usbTxComplete` configures the structure used by `usbd_cdc.c`, allowing our function to be called at the right time:

```cpp
hcdc->TxCallBack = usbTxComplete;
```

4.  `usbTxComplete` is short, only consisting of a few lines that will provide a task notification and force a context switch to be evaluated (so `usbTask` will be unblocked as quickly as possible):

```cpp
void usbTxComplete( void )
{
  portBASE_TYPE xHigherPriorityTaskWoken = pdFALSE;
  xTaskNotifyFromISR( usbTaskHandle, 1, eSetValueWithOverwrite,
                                    &xHigherPriorityTaskWoken);
  portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}
```

A function pointed to by `TxCallBack` is executed within the USB ISR, so any code executed by the callback must be kept extremely brief, call only ISR-safe versions of FreeRTOS functions, and have its priority properly configured.

5.  The infinite `while` loop portion of `usbTask` follows:

```cpp
while(1)
  {
    SEGGER_SYSVIEW_PrintfHost("waiting for txStream");
    uint8_t numBytes = xStreamBufferReceive( txStream, usbTxBuff,
                                       txBuffLen, portMAX_DELAY);
    if(numBytes > 0)
    {
      SEGGER_SYSVIEW_PrintfHost("pulled %d bytes from txStream",
                                                      numBytes);
      USBD_CDC_SetTxBuffer(&hUsbDeviceFS, usbTxBuff, numBytes);
      USBD_CDC_TransmitPacket(&hUsbDeviceFS);
      ulTaskNotifyTake( pdTRUE, portMAX_DELAY );
      SEGGER_SYSVIEW_PrintfHost("tx complete");
    }
  }

```

The task notification provides an efficient way to gate transmissions without polling: 

*   Whenever a transmission has finished, the callback (`usbTxComplete`) will be executed from the USB stack. `usbTxComplete` will provide a notification that will unblock the `usbTask`, at which point it will go out to the stream buffer and collect as much data as it can in one call, copying all available data into `usbTxBuff` (up to `numBytes` bytes). 
*   If a transmission is complete, `usbTask` will block indefinitely until data shows up in the stream buffer (`txStream`). `usbTask` won't be consuming any CPU time while blocking, but it will also automatically unblock whenever data is available.

This method provides a very efficient way of queueing data, while also providing good throughput and low latency. Any tasks adding data to the queue don't need to block or wait until their data is transmitted.

# Putting it all together

There's a fair amount going on here, with multiple sources of asynchronous events. Here's a sequence diagram of how all of these functions fit together:

![](img/91dc6143-3515-4e56-88d8-d0d6fda668ae.png)

Here are a few noteworthy items from the preceding diagram:

*   Calls to `TransmitUsbData` and `TransmitUsbDataLossy` are non-blocking. If space is available, data is transferred into the stream buffer, `txStream`, and the number of bytes copied is returned. Partial messages may be copied into the buffer (which happens under extremely high load when the buffer gets filled).
*   Two things need to happen before a packet of data is sent via `USBD_CDC_TransmitPacket`:
    *   `usbTask` must receive a task notification, indicating that it is clear to send data.
    *   Data must be available in `txStream`. 
*   Once transmission has started, the USB stack will be called by `OTG_FS_IRQHandler` in `stm32f7xx_it.c` until the transfer is complete, at which point the function pointed to by `TxCallBack` (`usbTxComplete`) will be called. This callback is executed from within the USB ISR, so the ISR-safe version of `vTaskNotify` (`vTaskNotifyFromISR`) must be used.

In `mainStreamBuffer.c`, (available from [https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Chapter_11/Src/mainUsbStreamBuffer.c](https://github.com/PacktPublishing/Hands-On-RTOS-with-Microcontrollers/tree/master/Drivers/STM32F7xx_HAL_Driver)), the virtual COM port is initialized with a single line, once the hardware initialization has been performed:

```cpp
int main(void)
{
  HWInit();
 VirtualCommInit();
```

A single task has been created in `mainStreamBuffer.c` to push data over to the USB:

```cpp
void usbPrintOutTask( void* NotUsed)
{
  const uint8_t testString[] = "test\n";
  const uint8_t messageString[] = "message\n";

  while(1)
  {
    SEGGER_SYSVIEW_PrintfHost("add \"test\" to txStream");
    TransmitUsbDataLossy(testString, sizeof(testString));
    SEGGER_SYSVIEW_PrintfHost("add \"message\" to txStream");
    TransmitUsbDataLossy(messageString, sizeof(messageString));
    vTaskDelay(2);
  }
}
```

This results in output that alternates as we would expect, thanks to the buffering provided by the stream buffer:

![](img/fb8c955c-520e-4f57-95f7-71397b969164.png)

Let's now take a look at a single transfer using SystemView:

![](img/3c6a6009-497b-4811-af2d-a338ecf2916e.png)

All of the tasks and ISRs are arranged in ascending priority. Numbers in the SystemView terminal on the right have corresponding numbers on the timeline:

1.  The first item, *test\n*, was added to the buffer. `usbTask` is now ready to run (indicated by the blue box).
2.  The second item, *message\n*, was added to the buffer. After the `usbPrint` task blocks, `usbTask` is brought into context by the scheduler.

3.  All 15 bytes are copied from the stream buffer, `txStream`, and placed into the local `usbTxBuff`. This buffer is fed into the USB stack using `USBD_CDC_SetTxBuffer` and a transfer is started with `USBD_CDC_TransmitPacket`. The USB stack takes care of the transfer and issues a callback when it is finished (`usbTxComplete`). This callback sends a task notification to `usbTask`, signaling that the transfer is complete.
4.  `usbTask` receives the task notification and continues with the loop.
5.  `usbTask` begins waiting on data to become available in `txStream`.

This general sequence repeats every 2 ms, which translates into about 1,000 lines being transmitted each second. Keep in mind that the delay is present to make analysis easier. The non-lossy `TransmitUsbData()` could be utilized instead with no delay, but seeing *exactly *what is occurring is a bit more of a challenge:

![](img/174b876a-76ed-4aec-af89-56c36dca0b79.png)

The total CPU time consumed is around 10%, with most of the time spent in `usbTask` and `usbPrint.`

If we wanted to minimize CPU usage, at the expense of introducing a bit more latency between when a message was first printed and when it was transmitted over the line, the following changes could be made:

The following is an excerpt from `VirtualCommDriver.c`:

1.  Increase the trigger value used to initialize `txStream` from 1 to 500\. This will cause the buffer to attempt to gather 500 bytes before returning data:

```cpp
void VirtualCommInit( void )
{
  MX_USB_DEVICE_Init();
 txStream = xStreamBufferCreate( txBuffLen, 500);
```

2.  Decrease the maximum amount of time to wait on data to become available in the stream from an infinite timeout to 100 ticks. This will guarantee that the stream is emptied at least once every 100 ticks (which happens to be 100 ms with the current configuration). This minimizes context switching and how often `usbTask` will need to run. It also allows for more data to be transferred to the USB stack at a time:

```cpp
uint8_t numBytes = xStreamBufferReceive( txStream, usbTxBuff,
                                             txBuffLen, 100);
```

Increasing the trigger value of the stream buffer from 1 to 500 bytes and increasing the available block time from 1 to 100 ticks *reduces the CPU usage of*** `usbTask`** *by a whopping 94%*:

![](img/a193835d-8bf2-4f13-8a88-dbd4a4b945b9.png)

Now, this means we also have an increase in latency – the amount of time it takes between when a call to `TransmitUsbDataLossy` is made and when that message is transmitted across the USB cable. So, there is a trade-off to be made. In this simple example, where the use case is just a simple printout with a human looking at the text, 10 Hz is likely more than fast enough.

Now that we have most of our USB driver written, we can add in some additional safety measures to guarantee that `VirtualCommDriver` is safe to use across multiple tasks.

# Using mutexes for access control

Since we implemented our driver with a stream buffer, if we are interested in having more than one task write to it, access to the stream buffer must be protected by a mutex. Most of the other FreeRTOS primitives, such as queues, don't have this limitation; they are safe to use across multiple tasks without any additional effort. Let's take a look at what would be required to extend VirtualCommDriver to make it usable by more than one task.

# Extending VirtualCommDriver

To make usage for the users of VirtuCommPortDriver as easy as possible, we can incorporate all of the mutex handling within the function call itself, rather than requiring users of the function to manage the mutex.

An additional file, `VirtualCommDriverMultiTask.c`has been created to illustrate this:

1.  A mutex is defined and created, along with all of the other variables required across multiple functions in this source file:

```cpp
#define txBuffLen 2048
uint8_t vcom_usbTxBuff[txBuffLen];
StreamBufferHandle_t vcom_txStream = NULL;
TaskHandle_t vcom_usbTaskHandle = NULL;
SemaphoreHandle_t vcom_mutexPtr = NULL;
```

To prevent multiple copies of this mutex from being created for each compilation unit `VirtualComDriverMultitTask` is included in, we won't define our *private global* variables as having *static* scope this time. Since we don't have namespaces in C, we'll prepend the names with `vcom_` in an attempt to avoid naming collisions with other globals.

2.  The mutex is initialized in `VirtualCommInit()`:

```cpp
vcom_mutexPtr = xSemaphoreCreateMutex();
assert_param(vcom_mutexPtr != NULL);
```

3.  A new `TransmitUsbData()`function has been defined. It now includes a maximum delay (specified in milliseconds):

```cpp
int32_t TransmitUsbData(uint8_t const* Buff, uint16_t Len, int32_t DelayMs)
```

4.  Define a few variables to help keep track of elapsed time:

```cpp
const uint32_t delayTicks = DelayMs / portTICK_PERIOD_MS;
const uint32_t startingTime = xTaskGetTickCount();
uint32_t endingTime = startingTime + delayTicks;
```

The previous calls to `xStreamBufferSend` are wrapped inside the mutex, `vcom_mutexPtr`. `remainingTime` is updated after each blocking FreeRTOS API call to accurately limit the maximum amount of time spent in this function:

```cpp
  if(xSemaphoreTake(vcom_mutexPtr, delayTicks ) == pdPASS)
  {
    uint32_t remainingTime = endingTime - xTaskGetTickCount();
    numBytesCopied = xStreamBufferSend( vcom_txStream, Buff, Len,
 remainingTime);

    if(numBytesCopied != Len)
    {
      remainingTime = endingTime - xTaskGetTickCount();
      numBytesCopied += xStreamBufferSend(  vcom_txStream, 
                                            Buff+numBytesCopied, 
                                            Len-numBytesCopied,
 remainingTime);
    }

    xSemaphoreGive(vcom_mutexPtr);
  }
```

A new main file, `mainUsbStreamBufferMultiTask`, was created to illustrate usage:

1.  `usbPrintOutTask` was created. This takes a number as an argument as a means to easily differentiate which task is writing:

```cpp
void usbPrintOutTask( void* Number)
{
#define TESTSIZE 10
  char testString[TESTSIZE];
  memset(testString, 0, TESTSIZE);
  snprintf(testString, TESTSIZE, "task %i\n", (int) Number);
  while(1)
 {
 TransmitUsbData((uint8_t*)testString, sizeof(testString), 100);
 vTaskDelay(2);
 }
}

```

2.  Two instances of `usbPrintOutTask` are created, passing in the numbers *1* and *2*. A cast to `(void*)` prevents complaints from the compiler:

```cpp
retVal = xTaskCreate(   usbPrintOutTask, "usbprint1", 
                        STACK_SIZE, (void*)1, tskIDLE_PRIORITY + 2, 
                        NULL);
assert_param( retVal == pdPASS);
retVal = xTaskCreate(    usbPrintOutTask, "usbprint2", 
                         STACK_SIZE, (void*)2, tskIDLE_PRIORITY + 
                         2, 
                         NULL);
assert_param( retVal == pdPASS);
```

Now, multiple tasks are able to send data over the USB. The amount of time that each call to `TransmitUsbData` may block is specified with each function call.

# Guaranteeing atomic transactions

Sometimes, it is desirable to transmit a message and then be confident that the response is for that message. In these cases, a mutex can be used at a higher level. This allows for groups of messages to be clustered together. An example of when this technique can be especially useful is a single peripheral servicing multiple physical ICs across multiple tasks:

![](img/7d19521f-e780-41d2-b8c5-fba0dc26bff4.png)

In the preceding diagram, the same peripheral (SPI1) is used to service two different ICs. Although the SPI peripheral is shared, there are separate chip select lines (CS1 and CS2) for each IC. There are also two completely independent drivers for these devices (one is an ADC and one is a DAC). In this situation, a mutex can be used to group multiple messages going to the same device together so they all occur when the correct chip select line is activated; things wouldn't go well if the ADC was meant to receive data when CS2 was asserted (the DAC would receive the data instead). 

This approach can work well when all of the the following conditions exist:

*   Individual transfers are fast.
*   Peripherals have low latency.
*   Flexibility (at least several ms, if not 10's of ms) as to exactly when transfers can take place.

Shared hardware isn't much different from any other shared resource. There are many other real-world examples that haven't been discussed here.

# Summary

In this chapter, we took a deep dive into creating an efficient interface to a complex driver stack that was very convenient to use. Using stream buffers, we analyzed trade-offs between decreasing latency and minimizing CPU usage. After a basic interface was in place, it was extended to be used across multiple tasks. We also saw an example of how a mutex could be used for ensuring that a multi-stage transaction remained atomic, even while the peripheral was shared between tasks.

Throughout the examples, we focused on performance versus ease of use and coding effort. Now that you have a good understanding of why design decisions are being made, you should be in a good position to make informed decisions regarding your own code base and implementations. When the time comes to implement your design, you'll also have a solid understanding of the steps that need to be taken to guarantee race condition-free access to your shared peripheral.

So far, we've been discussing trade-offs when creating drivers, so that we write something that is as close to perfect for our use case as possible. Wouldn't it be nice if (at the beginning of a new project) we didn't need to re-invent the wheel by copying, pasting, and modifying all of these drivers every time? Instead of continually introducing low-level, hard-to-find bugs, we could simply bring in everything we know that works well and get to work adding new features required for the new project? With a well-architected system, this type of workflow is entirely possible! In the next chapter, we'll cover several tips on creating a firmware architecture that is flexible and doesn't suffer from the copy-paste-modify trap many firmware engineers find themselves stuck in.

# Questions

As we conclude, here is a list of questions for you to test your knowledge regarding this chapter's material. You will find the answers in the *Assessments* section of the Appendix:

1.  It is *always* best to minimize the number of hardware peripherals being used:
    *   True
    *   False
2.  When sharing a hardware peripheral across multiple tasks, the only concern is creating threadsafe code that ensures that only one task has access to the peripheral at a time: 
    *   True
    *   False
3.  What trade-offs do stream buffers allow us to make when creating them?
    *   Latency
    *   CPU efficiency
    *   Required RAM size
    *   All of the above
4.  Stream buffers can be used directly by multiple tasks:
    *   True
    *   False
5.  What is one of the mechanisms that can be used to create threadsafe atomic access to a peripheral for the entire duration of a multi-stage message?