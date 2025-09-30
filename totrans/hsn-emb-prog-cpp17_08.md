# Testing OS-Based Applications

Often, an embedded system uses a more or less regular **Operating System** (**OS**), which means that, often much, is the same as on our desktop OS in terms of runtime environment and tools, especially when targeting embedded Linux. Yet, differences in terms of performance and access offered by the embedded hardware versus our PC makes it essential to consider where to perform which parts of developing and testing, as well as how to integrate this into our development workflow.

In this chapter, we'll cover the following topics:

*   Developing cross-platform code
*   Debugging and testing cross-platform code under Linux
*   Effectively using cross-compilers
*   Creating a build system that supports multiple targets

# Avoiding real hardware

One of the biggest advantages of OS-based development on platforms such as embedded Linux is that it's so similar to a regular desktop Linux installation. Especially when running an OS such as a Debian-based Linux distribution (Armbian, Raspbian, and others) on SoCs, we have practically the same tools available, with the entire package manager, compiler collections, and libraries available with a few keystrokes.

This is, however, also its biggest pitfall.

We can write code, copy it over to the SBC, compile it there, run the test, and make changes to the code before repeating the process. Or, we can even write the code on the SBC itself, essentially using it as our sole development platform.

The main reasons why we should never do this are as follows:

*   A modern PC is much faster.
*   Testing on real hardware should never be done until the final stages of development.
*   Automated integration testing is made much harder.

Here, the first point seems fairly obvious. What takes a single or dual-core ARM SoC a good minute to compile will quickly go from start of compilation to linking the objects in ten seconds or less with a relatively modern multi-core, multithreaded processor at 3+ GHz, and a toolchain that supports multi-core compilation.

This means that, instead of waiting half a minute or longer before we can run a new test or start a debugging session, we can do so almost instantly.

The next two points are related. While it may seem advantageous to test on the real hardware, it comes with its own complications. One thing is that this hardware relies on a number of external factors to work properly, including its power supply, any wiring between power sources, peripherals, and signal interfaces. Things such as electromagnetic interference may also cause issues, in terms of signal degradation, as well as interrupts being triggered due to electromagnetic coupling.

An example of electromagnetic coupling became apparent while developing the club status service project of [Chapter 3](47e0b6fb-cb68-43c3-9453-2dc7575b1a46.xhtml), *Developing for Embedded Linux and Similar Systems*. Here, one of the signal wires for the switches ran alongside 230V AC wiring. Changes in the current on this mains wiring induced pulses in the signal wire, causing false interrupt trigger events.

All of these potential hardware-related issues show that such tests aren't nearly as deterministic as we would wish them to be. The potential result of this is that project development takes much longer than planned, with debugging being complicated due to conflicting and non-deterministic test results.

Another effect of a focus on developing on and for real hardware is that it makes automated testing much harder. The reason for this is that we cannot use any generic build cluster and, for example, Linux VM-based testing environment, as is common with mainstream **Continuous** **Integration** (**CI**) services.

Instead of this, we would have to somehow integrate something such as an SBC into the CI system, having it either cross-compile and copy the binary to the SBC for running the test, or compile it on the SBC itself, which gets us back to the first point.

In the next few sections, we'll look at a of approaches to make embedded Linux-based development as painless as possible, starting with cross-compilation.

# Cross-compiling for SBCs

The compile process takes the source files, turning them into an intermediate format, after which this format can be used to target a specific CPU architecture. For us, this means that we aren't limited to compiling applications for an SBC on that SBC itself, but we can do so on our development PC.

To do so for an SBC such as the Raspberry Pi (Broadcom Cortex-A-based ARM SoCs), we need to install the `arm-linux-gnueabihf` toolchain, which targets the ARM architecture with hard float (hardware floating point) support, outputting Linux-compatible binaries.

On a Debian-based Linux system, we can install the entire toolchain with the following commands:

```cpp
sudo apt install build-essential
sudo apt install g++-arm-linux-gnueabihf
sudo apt install gdb-multiarch  
```

The first command installs the native GCC-based toolchain for the system (if it wasn't already installed), along with any common related tools and utilities, including `make`, `libtool`, `flex`, and others. The second command installs the actual cross-compiler. Finally, the third package is the version of the GDB debugger that supports multiple architectures, which we'll need later on for doing remote debugging on the real hardware, as well as for analyzing core dumps produced when our application crashes.

We can now use the g++ compiler for the target SBC using its full name on the command line:

```cpp
arm-linux-gnueabihf-g++  
```

To test whether the toolchain was properly installed, we can execute the following command, which should tell us the compiler details including the version:

```cpp
arm-linux-gnueabihf-g++ -v  
```

In addition to this, we may need to link with some shared libraries that exist on the target system. For this, we can copy the entire contents of the `/lib` and `/usr` folders and include them as part of the system root for the compiler:

```cpp
mkdir ~/raspberry/sysroot
scp -r pi@Pi-system:/lib ~/raspberry/sysroot
scp -r pi@Pi-system:/usr ~/raspberry/sysroot  
```

Here, `Pi-system` is the IP address or network name of the Raspberry Pi or similar system. After this, we can tell GCC to use these folders instead of the standard paths using the `sysroot` flag:

```cpp
--sysroot=dir  
```

Here `dir` would be the folder where we copied these folders to, in this example that would be `~/raspberry/sysroot`.

Alternatively, we can just copy the header and library files we require and add them as part of the source tree. Whichever approach is the easiest mostly depends on the dependencies of the project in question.

For the club status service project, we require at the very least the headers and libraries for WiringPi, as well as those for the POCO project and its dependencies. We could determine the dependencies we need and copy the required includes and library files that are missing from the toolchain we installed earlier. Unless there's a pressing need to do so, it's far easier to just copy the entire folders from the SBC's OS.

As an alternative to using the `sysroot` method, we can also explicitly define the paths to the shared libraries that we wish to use while linking our code. This of course comes with its own set of advantages and disadvantages.

# Integration test for club status service

In order to test the club status service on a regular desktop Linux (or macOS or Windows) system before we embark on cross-compiling and testing on real hardware, a simple integration test was written, which uses mocks for the GPIO and I2C peripherals.

In the source code for the project covered in [Chapter 3](47e0b6fb-cb68-43c3-9453-2dc7575b1a46.xhtml), *Developing for Embedded Linux and Similar Systems*, the files for these peripherals are found in the `wiring` folder of that project.

We start with the `wiringPi.h` header:

```cpp
#include <Poco/Timer.h>

#define  INPUT              0
#define  OUTPUT                   1
#define  PWM_OUTPUT         2
#define  GPIO_CLOCK         3
#define  SOFT_PWM_OUTPUT          4
#define  SOFT_TONE_OUTPUT   5
#define  PWM_TONE_OUTPUT          6
```

We include a header from the POCO framework to allow us to easily create a timer instance later on. Then, we define all possible pin modes, just as the actual WiringPi header defines:

```cpp
#define  LOW                0
#define  HIGH               1

#define  PUD_OFF                  0
#define  PUD_DOWN           1
#define  PUD_UP                   2

#define  INT_EDGE_SETUP          0
#define  INT_EDGE_FALLING  1
#define  INT_EDGE_RISING         2
#define  INT_EDGE_BOTH           3
```

These defines define further pin modes, including the digital input levels, the possible states of the pull-ups and pull-downs on the pins, and finally the possible types of interrupts, defining the trigger or triggers for an interrupt:

```cpp
typedef void (*ISRCB)(void); 
```

This `typedef` defines the format for an interrupt callback function pointer.

Let's now look at the `WiringTimer` class:

```cpp
class WiringTimer {
    Poco::Timer* wiringTimer;
    Poco::TimerCallback<WiringTimer>* cb;
    uint8_t triggerCnt;

 public:
    ISRCB isrcb_0;
    ISRCB isrcb_7;
    bool isr_0_set;
    bool isr_7_set;

    WiringTimer();
    ~WiringTimer();
    void start();
    void trigger(Poco::Timer &t);
 };
```

This class is the integral part of the GPIO-side of our mock implementation. Its main purpose is to keep track of which of the two interrupts we're interested in have been registered, and to trigger them at regular intervals using the timer, as we'll see in a moment:

```cpp
int wiringPiSetup(); 
void pinMode(int pin, int mode); 
void pullUpDnControl(int pin, int pud); 
int digitalRead(int pin);
int wiringPiISR(int pin, int mode, void (*function)(void));
```

Finally, we define the standard WiringPi functions before moving on the implementation:

```cpp
#include "wiringPi.h"

#include <fstream>
#include <memory>

WiringTimer::WiringTimer() {
   triggerCnt = 0;
   isrcb_0 = 0;
   isrcb_7 = 0;
   isr_0_set = false;
   isr_7_set = false;

   wiringTimer = new Poco::Timer(10 * 1000, 10 * 1000);
   cb = new Poco::TimerCallback<WiringTimer>(*this, 
   &WiringTimer::trigger);
}
```

In the class constructor, we set the default values before creating the timer instance, configuring it to call our callback function every ten seconds, after an initial 10-second delay:

```cpp
WiringTimer::~WiringTimer() {
   delete wiringTimer;
   delete cb;
}
```

In the destructor, we delete the timer callback instance:

```cpp
void WiringTimer::start() {
   wiringTimer->start(*cb);
}
```

In this function, we actually start the timer:

```cpp
void WiringTimer::trigger(Poco::Timer &t) {
    if (triggerCnt == 0) {
          char val = 0x00;
          std::ofstream PIN0VAL;
          PIN0VAL.open("pin0val", std::ios_base::binary | std::ios_base::trunc);
          PIN0VAL.put(val);
          PIN0VAL.close();

          isrcb_0();

          ++triggerCnt;
    }
    else if (triggerCnt == 1) {
          char val = 0x01;
          std::ofstream PIN7VAL;
          PIN7VAL.open("pin7val", std::ios_base::binary | std::ios_base::trunc);
          PIN7VAL.put(val);
          PIN7VAL.close();

          isrcb_7();

          ++triggerCnt;
    }
    else if (triggerCnt == 2) {
          char val = 0x00;
          std::ofstream PIN7VAL;
          PIN7VAL.open("pin7val", std::ios_base::binary | std::ios_base::trunc);
          PIN7VAL.put(val);
          PIN7VAL.close();

          isrcb_7();

          ++triggerCnt;
    }
    else if (triggerCnt == 3) {
          char val = 0x01;
          std::ofstream PIN0VAL;
          PIN0VAL.open("pin0val", std::ios_base::binary | std::ios_base::trunc);
          PIN0VAL.put(val);
          PIN0VAL.close();

          isrcb_0();

          triggerCnt = 0;
    }
 }

```

This last function in the class is the callback for the timer. The way it functions is that it keeps track of how many times it has been triggered, with it setting the appropriate pin level in the form of a value in a file that we write to disk.

After the initial delay, the first trigger will set the lock switch to `false`, the second the status switch to `true`, the third the status switch back to `false`, and finally the fourth trigger sets the lock switch back to `true`, before resetting the counter and starting over again:

```cpp
namespace Wiring {
   std::unique_ptr<WiringTimer> wt;
   bool initialized = false;
}
```

We add a global namespace in which we have a `unique_ptr` instance for a `WiringTimer` class instance, along with an initialization status indicator.

```cpp
int wiringPiSetup() {
    char val = 0x01;
    std::ofstream PIN0VAL;
    std::ofstream PIN7VAL;
    PIN0VAL.open("pin0val", std::ios_base::binary | std::ios_base::trunc);
    PIN7VAL.open("pin7val", std::ios_base::binary | std::ios_base::trunc);
    PIN0VAL.put(val);
    val = 0x00;
    PIN7VAL.put(val);
    PIN0VAL.close();
    PIN7VAL.close();

    Wiring::wt = std::make_unique<WiringTimer>();
    Wiring::initialized = true;

    return 0;
 }
```

The setup function is used to write the default values for the mocked GPIO pin inputs value to disk. We also create the pointer to a `WiringTimer` instance here:

```cpp
 void pinMode(int pin, int mode) {
    // 

    return;
 }

 void pullUpDnControl(int pin, int pud) {
    // 

    return;
 }
```

Because our mocked implementation determines the behavior of the pins, we can ignore any input on these functions. For testing purposes, we could add an assert to validate that these functions have been called at the right times with the appropriate settings:

```cpp
 int digitalRead(int pin) {
    if (pin == 0) {
          std::ifstream PIN0VAL;
          PIN0VAL.open("pin0val", std::ios_base::binary);
          int val = PIN0VAL.get();
          PIN0VAL.close();

          return val;
    }
    else if (pin == 7) {
          std::ifstream PIN7VAL;
          PIN7VAL.open("pin7val", std::ios_base::binary);
          int val = PIN7VAL.get();
          PIN7VAL.close();

          return val;
    }

    return 0;
 }
```

When reading the value for one of the two mocked pins, we open its respective file and read out its content, which is either the 1 or 0 set by the setup function or by the callback:

```cpp
//This value is then returned to the calling function.

 int wiringPiISR(int pin, int mode, void (*function)(void)) {
    if (!Wiring::initialized) { 
          return 1;
    }

    if (pin == 0) { 
          Wiring::wt->isrcb_0 = function;
          Wiring::wt->isr_0_set = true;
    }
    else if (pin == 7) {
          Wiring::wt->isrcb_7 = function;
          Wiring::wt->isr_7_set = true;
    }

    if (Wiring::wt->isr_0_set && Wiring::wt->isr_7_set) {
          Wiring::wt->start();
    }

    return 0;
 }
```

This function is used to register an interrupt and its associated callback function. After an initial check that the mock has been initialized by the setup function, we then continue to register the interrupt for one of the two specified pins.

Once both pins have had an interrupt set for them, we start the timer, which will in turn start generating events for the interrupt callbacks.

Next is the I2C bus mock:

```cpp
int wiringPiI2CSetup(const int devId);
int wiringPiI2CWriteReg8(int fd, int reg, int data);
```

We just need two functions here: the setup function and the simple one-byte register write function.

The implementation is as follows:

```cpp
#include "wiringPiI2C.h"

#include "../club.h"

#include <Poco/NumberFormatter.h>

using namespace Poco;

int wiringPiI2CSetup(const int devId) {
   Club::log(LOG_INFO, "wiringPiI2CSetup: setting up device ID: 0x" 
                                        + NumberFormatter::formatHex(devId));

   return 0;
}
```

In the setup function, we log the requested device ID (I2C bus address) and return a standard device handle. Here, we use the `log()` function from the `Club` class to make the mock integrate into the rest of the code:

```cpp
int wiringPiI2CWriteReg8(int fd, int reg, int data) {
    Club::log(LOG_INFO, "wiringPiI2CWriteReg8: Device handle 0x" + NumberFormatter::formatHex(fd) 
                                        + ", Register 0x" + NumberFormatter::formatHex(reg)
                                        + " set to: 0x" + NumberFormatter::formatHex(data));

    return 0;
}
```

Since the code that would call this function wouldn't be expecting a response, beyond a simple acknowledgment that the data has been received, we can just log the received data and further details here. The `NumberFormatter` class from POCO is used here as well for formatting the integer data as hexadecimal values like in the application, for consistency.

We now compile the project and use the following command-line command:

```cpp
make TEST=1  
```

Running the application (under GDB, to see when new threads are created/destroyed) now gets us the following output:

```cpp
 Starting ClubStatus server...
 Initialised C++ Mosquitto library.
 Created listener, entering loop...
 [New Thread 0x7ffff49c9700 (LWP 35462)]
 [New Thread 0x7ffff41c8700 (LWP 35463)]
 [New Thread 0x7ffff39c7700 (LWP 35464)]
 Initialised the HTTP server.
 INFO:       Club: starting up...
 INFO:       Club: Finished wiringPi setup.
 INFO:       Club: Finished configuring pins.
 INFO:       Club: Configured interrupts.
 [New Thread 0x7ffff31c6700 (LWP 35465)]
 INFO:       Club: Started update thread.
 Connected. Subscribing to topics...
 INFO:       ClubUpdater: Starting i2c relay device.
 INFO:       wiringPiI2CSetup: setting up device ID: 0x20
 INFO:       wiringPiI2CWriteReg8: Device handle 0x0, Register 0x6 set to: 0x0
 INFO:       wiringPiI2CWriteReg8: Device handle 0x0, Register 0x2 set to: 0x0
 INFO:       ClubUpdater: Finished configuring the i2c relay device's registers.  
```

At this point, the system has been configured with all interrupts set and the I2C device configured by the application. The timer has started its initial countdown:

```cpp
 INFO:       ClubUpdater: starting initial update run.
 INFO:       ClubUpdater: New lights, clubstatus off.
 DEBUG:      ClubUpdater: Power timer not active, using current power state: off
 INFO:       ClubUpdater: Red on.
 DEBUG:      ClubUpdater: Changing output register to: 0x8
 INFO:       wiringPiI2CWriteReg8: Device handle 0x0, Register 0x2 set to: 0x8
 DEBUG:      ClubUpdater: Finished writing relay outputs with: 0x8
 INFO:       ClubUpdater: Initial status update complete.  
```

The initial status of the GPIO pins has been read out and both switches are found to be in the `off` position, so we activate the red light on the traffic light indicator by writing its position in the register:

```cpp
 INFO:       ClubUpdater: Entering waiting condition. INFO:       ClubUpdater: lock status changed to unlocked
 INFO:       ClubUpdater: New lights, clubstatus off.
 DEBUG:      ClubUpdater: Power timer not active, using current power state: off
 INFO:       ClubUpdater: Yellow on.
 DEBUG:      ClubUpdater: Changing output register to: 0x4
 INFO:       wiringPiI2CWriteReg8: Device handle 0x0, Register 0x2 set to: 0x4
 DEBUG:      ClubUpdater: Finished writing relay outputs with: 0x4
 INFO:       ClubUpdater: status switch status changed to on
 INFO:       ClubUpdater: Opening club.
 INFO:       ClubUpdater: Started power timer...
 DEBUG:      ClubUpdater: Sent MQTT message.
 INFO:       ClubUpdater: New lights, clubstatus on.
 DEBUG:      ClubUpdater: Power timer active, inverting power state from: on
 INFO:       ClubUpdater: Green on.
 DEBUG:      ClubUpdater: Changing output register to: 0x2
 INFO:       wiringPiI2CWriteReg8: Device handle 0x0, Register 0x2 set to: 0x2
 DEBUG:      ClubUpdater: Finished writing relay outputs with: 0x2
 INFO:       ClubUpdater: status switch status changed to off
 INFO:       ClubUpdater: Closing club.
 INFO:       ClubUpdater: Started timer.
 INFO:       ClubUpdater: Started power timer...
 DEBUG:      ClubUpdater: Sent MQTT message.
 INFO:       ClubUpdater: New lights, clubstatus off.
 DEBUG:      ClubUpdater: Power timer active, inverting power state from: off
 INFO:       ClubUpdater: Yellow on.
 DEBUG:      ClubUpdater: Changing output register to: 0x5
 INFO:       wiringPiI2CWriteReg8: Device handle 0x0, Register 0x2 set to: 0x5
 DEBUG:      ClubUpdater: Finished writing relay outputs with: 0x5
 INFO:       ClubUpdater: setPowerState called.
 DEBUG:      ClubUpdater: Writing relay with: 0x4
 INFO:       wiringPiI2CWriteReg8: Device handle 0x0, Register 0x2 set to: 0x4
 DEBUG:      ClubUpdater: Finished writing relay outputs with: 0x4
 DEBUG:      ClubUpdater: Written relay outputs.
 DEBUG:      ClubUpdater: Finished setPowerState.
 INFO:       ClubUpdater: lock status changed to locked
 INFO:       ClubUpdater: New lights, clubstatus off.
 DEBUG:      ClubUpdater: Power timer not active, using current power state: off
 INFO:       ClubUpdater: Red on.
 DEBUG:      ClubUpdater: Changing output register to: 0x8
 INFO:       wiringPiI2CWriteReg8: Device handle 0x0, Register 0x2 set to: 0x8
 DEBUG:      ClubUpdater: Finished writing relay outputs with: 0x8  
```

Next, the timer starts triggering the callback function repeatedly, causing it to go through its different stages. This allows us to ascertain that the basic functioning of the code is correct.

At this point, we can start implementing more complex test cases, conceivably even implementing scriptable test cases using an embedded Lua, Python runtime or similar.

# Mock versus hardware

An obvious question to ask when mocking away large sections of code and hardware peripherals is how realistic the resulting mock is. We obviously want to be able to cover as many real-life scenarios as possible with our integration test before we move to testing on the target system.

If we want to know which test cases we wish to cover in our mock, we have to look both at our project requirements (what it should be able to handle), and which situations and inputs can occur in a real-life scenario.

For this, we would analyze the underlying code to see what conditions can occur, and decide on which ones are relevant for us.

In the case of the WiringPi mocks we looked at earlier, a quick glance at the source code for the library's implementation makes it clear just how much we simplified our code compared to the version we would be using on our target system.

Looking at the basic WiringPi setup function, we see that it does the following:

*   Determines the exact board model and SoC to get the GPIO layout
*   Opens the Linux device for the memory-mapped GPIO pins
*   Sets the memory offsets into the GPIO device and uses `mmap()` to map specific peripherals such as PWM, timer, and GPIO into memory

Instead of ignoring calls to `pinMode()`, the implementation does the following:

*   Appropriately sets the hardware GPIO direction register in the SoC (for input/output mode)
*   Starts PWM, soft PWM, or Tone mode on a pin (as requested); sub-functions set the appropriate registers

This continues with the I2C side, where the setup function implementation looks like this:

```cpp
int wiringPiI2CSetup (const int devId) { 
   int rev; 
   const char *device; 

   rev = piGpioLayout(); 

   if (rev == 1) { 
         device = "/dev/i2c-0"; 
   } 
   else { 
         device = "/dev/i2c-1"; 
   } 

   return wiringPiI2CSetupInterface (device, devId); 
} 
```

Compared to our mock implementation, the main difference is in that an I2C peripheral is expected to be present on the in-memory filesystem of the OS, and the board revision determines which one we pick.

The last function that gets called tries to open the device, as in Linux and similar OSes every device is simply a file that we can open and get a file handle to, if successful. This file handle is the ID that gets returned when the function returns:

```cpp
int wiringPiI2CSetupInterface (const char *device, int devId) { 
   int fd; 
   if ((fd = open (device, O_RDWR)) < 0) { 
         return wiringPiFailure (WPI_ALMOST, "Unable to open I2C device: %s\n", 
                                                                                                strerror (errno)); 
   } 

   if (ioctl (fd, I2C_SLAVE, devId) < 0) { 
         return wiringPiFailure (WPI_ALMOST, "Unable to select I2C device: %s\n",                                                                                                strerror (errno)); 
   } 

   return fd; 
} 
```

After opening the I2C device, the Linux system function, `ioctl()`, is used to send data to the I2C peripheral, in this case, the address of the I2C slave device that we wish to use. If successful, we get a non-negative response and return the integer that's our file handle.

Writing and reading the I2C bus is also handled using `ioctl()`, as we can see in the same source file:

```cpp
static inline int i2c_smbus_access (int fd, char rw, uint8_t command, int size, union i2c_smbus_data *data) { 
   struct i2c_smbus_ioctl_data args; 

   args.read_write = rw; 
   args.command    = command; 
   args.size       = size; 
   args.data       = data; 
   return ioctl(fd, I2C_SMBUS, &args); 
} 
```

This same inline function is called for every single I2C bus access. With the I2C device that we wish to use already selected, we can simply target the I2C peripheral and have it transmit the payload to the device.

Here, the `i2c_smbus_data` type is a simple union to support various sizes for the return value (when performing a read operation):

```cpp
union i2c_smbus_data { 
   uint8_t byte; 
   uint16_t word; 
   uint8_t block[I2C_SMBUS_BLOCK_MAX + 2]; 
}; 
```

Here, we mostly see the benefit of using an abstract API. Without it, we would have peppered our code with low-level calls that would have been much harder to mock away. What we also see is that there are a number of conditions that we should likely be testing as well, such as a missing I2C slave device, read and write errors on the I2C bus that may result in unexpected behavior, as well as unexpected input on GPIO pins, including for interrupt pins as was noted at the beginning of this chapter already.

Although obviously not all scenarios can be planned for, efforts should be made to document all realistic scenarios and incorporate them into the mocked-up implementation, so that they can be enabled at will during integration and regression testing and while debugging.

# Testing with Valgrind

Valgrind is the most commonly used collection of open source tools for analyzing and profiling everything from the cache and heap behavior of an application to memory leaks and potential multithreading issues. It works in tandem with the underlying operating system as, depending on the tool used, it has to intercept everything from memory allocations to instructions related to multithreading and related. This is the reason why it is only fully supported under Linux on 64-bit x86_64 architectures.

Using Valgrind on other supported platforms (Linux on x86, PowerPC, ARM, S390, MIPS, and ARM, also Solaris and macOS) is definitely also an option, but the primary development target of the Valgrind project is x86_64/Linux, making it the best platform to do profiling and debugging on, even if other platforms will be targeted later on.

On the Valgrind website at [http://valgrind.org/info/platforms.html](http://valgrind.org/info/platforms.html), we can see a full overview of the currently supported platforms.

One very attractive property of Valgrind is that none of its tools require us to alter the source code or resulting binary in any fashion. This makes it very easy to integrate into an existing workflow, including automated testing and integration systems.

On Windows-based system, tools such as Dr. Memory ([http://drmemory.org/](http://drmemory.org/)) are available as well, which can handle at least the profiling of memory-related behavior. This particular tool also comes with Dr. Fuzz, a tool that can repeatedly call functions with varying inputs, potentially useful for integration testing.

By using an integration test such as what we looked at in the previous section, we're free to fully analyze the behavior of our code from the comfort of our PC. Since all of Valgrind's tools significantly slow down the execution of our code (10-100 times), being able to do most of the debugging and profiling on a fast system means that we can save a significant amount of time before embarking on testing on the target hardware.

Of the tools we'll likely use the most often, **Memcheck**, **Helgrind**, and **DRD** are useful for detecting memory allocation and multithreading issues. Once our code passes through these three tools, while using an extensive integration test that provides wide coverage of the code, we can move on to profiling and optimizing.

To profile our code, we then use **Callgrind** to see where our code spends the most of the time executing, followed by **Massif** to do profiling of heap allocations. With the information we can glean from this data, we can make changes to the code to streamline common allocation and de-allocation cases. It might also show us where it might make sense to use a cache to reuse resources instead of discarding them from memory.

Finally, we would run another cycle of MemCheck, Helgrind, and DRD to ensure that our changes didn't cause any regressions. Once we're satisfied, we move on to deploying the code on the target system and see how it performs there.

If the target system also runs Linux or other supported OSes, we can use Valgrind on there as well, to check that we didn't miss anything. Depending on the exact platform (OS and CPU architecture), we may run into limitations of the Valgrind port for that platform. These can include errors such as *unhandled instruction*, where the tool hasn't had a CPU instruction implemented and hence Valgrind cannot continue.

By extending the integration test to use the SBC instead of a local process, we can set up a continuous integration system whereby, in addition to the tests on a local process, we also run them on real hardware, taking into account the limitations of the real hardware platform relative to the x86_64-based Linux system used for most of the testing.

# Multi-target build system

Cross-compilation and multi-target build systems are among the words that tend to frighten a lot of people, mostly because they evoke images of hugely complicated build scripts that require arcane incantations to perform the desired operation. In this chapter, we'll be looking at a simple Makefile-based build system, based on a build system that has seen use in commercial projects across a range of hardware targets.

The one thing that makes a build system pleasant to use is to be able to get everything set up for compilation with minimal fuss and have a central location from which we can control all relevant aspects of building the project, or parts of it, along with building and running tests.

For this reason, we have a single Makefile at the top of the project, which handles all of the basics, including the determining of which platform we run on. The only simplification we're making here is that we assume a Unix-like environment, with MSYS2 or Cygwin on Windows, and Linux, BSD, and OS X/macOS and others using their native shell environments. We could, however, also adapt it to allow for Microsoft Visual Studio, **Intel Compiler Collection** (**ICC**), and other compilers, so long as they provide the basic tools.

Key to the build system are simple Makefiles, in which we define the specific details of the target platform, for example, for a standard Linux system running on x86_x64 hardware:

```cpp
 TARGET_OS = linux
 TARGET_ARCH = x86_64

 export CC = gcc
 export CXX = g++
 export CPP = cpp
 export AR = ar
 export LD = g++
 export STRIP = strip
 export OBJCOPY = objcopy

 PLATFORM_FLAGS = -D__PLATFORM_LINUX__ -D_LARGEFILE64_SOURCE -D __LINUX__
 STD_FLAGS = $(PLATFORM_FLAGS) -Og -g3 -Wall -c -fmessage-length=0 -ffunction-sections -fdata-sections -DPOCO_HAVE_GCC_ATOMICS -DPOCO_UTIL_NO_XMLCONFIGURATION -DPOCO_HAVE_FD_EPOLL
 STD_CFLAGS = $(STD_FLAGS)
 STD_CXXFLAGS = -std=c++11 $(STD_FLAGS)
 STD_LDFLAGS = -L $(TOP)/build/$(TARGET)/libboost/lib \
                         -L $(TOP)/build/$(TARGET)/poco/lib \
                         -Wl,--gc-sections
 STD_INCLUDE = -I. -I $(TOP)/build/$(TARGET)/libboost/include \
                         -I $(TOP)/build/$(TARGET)/poco/include \
                         -I $(TOP)/extern/boost-1.58.0
 STD_LIBDIRS = $(STD_LDFLAGS)
 STD_LIBS = -ldl -lrt -lboost_system -lssl -lcrypto -lpthread

```

Here, we can set the names of the command-line tools that we'll be using for compiling, creating archives, stripping debug symbols from binaries, and so on. The build system will use the target OS and architecture to keep the created binaries separate so that we can use the same source tree to create binaries for all target platforms in one run.

We can see how we separate the flags that we'll be passing to the compiler and linker into different categories: platform-specific ones, common (standard) flags, and finally flags specific for the C and C++ compiler. The former is useful when integrating external dependencies that have been integrated into the source tree, yet are written in C. These dependencies we'll find in the `extern` folder, as we'll see in more detail in a moment.

This kind of file will be heavily customized to fit a specific project, adding the required includes, libraries, and compile flags. For this example file, we can see a project that uses the POCO and Boost libraries, along with OpenSSL, tweaking the POCO library for the target platform.

First, let's look at the top of the configuration file for macOS:

```cpp
TARGET_OS = osx
 TARGET_ARCH = x86_64

 export CC = clang
 export CXX = clang++
 export CPP = cpp
 export AR = ar
 export LD = clang++
 export STRIP = strip
 export OBJCOPY = objcopy
```

Although the rest of the file is almost the same, here we can see a good example of generalizing what a tool is called. Although Clang supports the same flags as GCC, its tools are called differently. With this approach, we just write the different names once in this file and everything will just work.

This continues with the Linux on ARM target, which is set up as a cross-compilation target:

```cpp
TARGET_OS = linux
 TARGET_ARCH = armv7
 TOOLCHAIN_NAME = arm-linux-gnueabihf

 export CC = $(TOOLCHAIN_NAME)-gcc
 export CXX = $(TOOLCHAIN_NAME)-g++
 export AR = $(TOOLCHAIN_NAME)-ar
 export LD = $(TOOLCHAIN_NAME)-g++
 export STRIP = $(TOOLCHAIN_NAME)-strip
 export OBJCOPY = $(TOOLCHAIN_NAME)-objcopy
```

Here, we see the reappearance of the cross-compilation toolchain for ARM Linux platforms, which we looked at earlier in this chapter. To save ourselves typing, we define the basic name once so that it is easy to redefine. This also shows how flexible Makefiles are. With some more creativity, we could create a set of templates that would generalize entire toolchains into a simple Makefile to be included by the main Makefile depending on hints in the platform's Makefile (or other configuration file), making this highly flexible.

Moving on, we'll look at the main Makefile as found in the root of the project:

```cpp
ifndef TARGET
 $(error TARGET parameter not provided.)
 endif
```

Since we cannot guess what platform the user wants us to target, we require that the target is specified, with the platform name as the value, for example, `linux-x86_x64`:

```cpp
export TOP := $(CURDIR)
 export TARGET
```

Later on in the system, we'll need to know which folder we're in on the local filesystem so that we can specify absolute paths. We use the standard Make variable for this and export it as our own environment variable, along with the build target name:

```cpp
UNAME := $(shell uname)
 ifeq ($(UNAME), Linux)
 export HOST = linux
 else
 export HOST = win32
 export FILE_EXT = .exe
 endif
```

Using the (command-line) `uname` command, we can check which OS we're running on, with each OS that supports the command in its shell returning its name, such as `Linux` for Linux and `Darwin` for macOS. On pure Windows (no MSYS2 or Cygwin), the command doesn't exist, which would get us the second part of this `if/else` statement.

This statement could be expanded to support more OSes, depending on what the build system requires. In this case, it is only used to determine whether executables we create should have a file extension:

```cpp
ifeq ($(HOST), linux)
 export MKDIR   = mkdir -p
 export RM            = rm -rf
 export CP            = cp -RL
 else
 export MKDIR   = mkdir -p
 export RM            = rm -rf
 export CP            = cp -RL
 endif
```

In this `if/else` statement, we can set the appropriate command-line commands for common file operations. Since we're taking the easy way out, we're assuming the use of MSYS2 or similar Bash shell on Windows.

We could take the concept of generalizing further at this point as well, splitting off the OS file CLI tools as its own set of Makefiles, which we can then include as part of OS-specific settings:

```cpp
include Makefile.$(TARGET)

 export TARGET_OS
 export TARGET_ARCH
 export TOOLCHAIN_NAME
```

At this point, we use the target parameter provided to the Makefile to include the appropriate configuration file. After exporting some details from it, we now have a configured build system:

```cpp
all: extern-$(TARGET) core

 extern:
    $(MAKE) -C ./extern $(LIBRARY)

 extern-$(TARGET):
    $(MAKE) -C ./extern all-$(TARGET)

 core:
    $(MAKE) -C ./Core

 clean: clean-core clean-extern

 clean-extern:
    $(MAKE) -C ./extern clean-$(TARGET)

 clean-core:
    $(MAKE) -C ./Core clean

 .PHONY: all clean core extern clean-extern clean-core extern-$(TARGET)
```

From this single Makefile, we can choose to compile the entire project or just the dependencies or the core project. We can also compile a specific external dependency and nothing else.

Finally, we can clean the core project, the dependencies, or both.

This top Makefile is primarily for controlling the underlying Makefiles. The next two Makefiles are found in the `Core` and `extern` folders. Of these, the `Core` Makefile simply directly compiles the project's core:

```cpp
include ../Makefile.$(TARGET) 

OUTPUT := CoreProject 

INCLUDE = $(STD_INCLUDE) 
LIBDIRS = $(STD_LIBDIRS) 

include ../version 
VERSIONINFO = -D__VERSION="\"$(VERSION)\"" 
```

As the first step, we include the Makefile configuration for the target platform so that we have access to all of its definitions. These could also have been exported in the main Makefile, but this way we're free to customize the build system even more.

We specify the name of the output binary that we're building, before some small tasks, including opening the `version` file (with Makefile syntax) in the root of the project, which contains the version number of the source we're building from. This is prepared to be passed as a preprocessor definition into the compiler:

```cpp
ifdef RELEASE 
TIMESTAMP = $(shell date --date=@`git show -s --format=%ct $(RELEASE)^{commit}` -u +%Y-%m-%dT%H:%M:%SZ) 
else ifdef GITTIME 
TIMESTAMP = $(shell date --date=@`git show -s --format=%ct` -u +%Y-%m-%dT%H:%M:%SZ) 
TS_SAFE = _$(shell date --date=@`git show -s --format=%ct` -u +%Y-%m-%dT%H%M%SZ) 
else 
TIMESTAMP = $(shell date -u +%Y-%m-%dT%H:%M:%SZ) 
TS_SAFE = _$(shell date -u +%Y-%m-%dT%H%M%SZ) 
endif 
```

This is another section where we rely on having a Bash shell or something compatible around, as we use the date command in order to create a timestamp for the build. The format depends on what parameter was passed to the main Makefile. If we're building a release, we take the timestamp from the Git repository, with the Git commit tag name used to retrieve the commit timestamp for that tag before formatting it.

If `GITTIME` is passed as parameter, the timestamp of the most recent Git commit is used. Otherwise, the current time and date is used (UTC).

This bit of code is intended to solve one of the issues that comes with having lots of test and integration builds: keeping track of which ones were built when and with which revision of the source code. It could be adapted to other file revision systems, as long as it supports similar functionality with the retrieving of specific timestamps.

Of note is the second timestamp we're creating. This is a slightly different formatted version of the timestamp that is affixed to the produced binary, except when we're building in release mode:

```cpp
CFLAGS = $(STD_CFLAGS) $(INCLUDE) $(VERSIONINFO) -D__TIMESTAMP="\"$(TIMESTAMP)\"" 
CXXFLAGS = $(STD_CXXFLAGS) $(INCLUDE) $(VERSIONINFO) -D__TIMESTAMP="\"$(TIMESTAMP)\"" 

OBJROOT := $(TOP)/build/$(TARGET)/obj 
CPP_SOURCES := $(wildcard *.cpp) 
CPP_OBJECTS := $(addprefix $(OBJROOT)/,$(CPP_SOURCES:.cpp=.o)) 
OBJECTS := $(CPP_OBJECTS) 
```

Here, we set the flags we wish to pass to the compiler, including the version and timestamp, both being passed as preprocessor definitions.

Finally, the sources in the current project folder are collected and the output folder for the object files is set. As we can see here, we'll be writing the object files to a folder underneath the project root, with further separation by the compile target:

```cpp
.PHONY: all clean 

all: makedirs $(CPP_OBJECTS) $(C_OBJECTS) $(TOP)/build/bin/$(TARGET)/$(OUTPUT)_$(VERSION)_$(TARGET)$(TS_SAFE) 

makedirs: 
   $(MKDIR) $(TOP)/build/bin/$(TARGET) 
   $(MKDIR) $(OBJROOT) 

$(OBJROOT)/%.o: %.cpp 
   $(CXX) -o $@ $< $(CXXFLAGS) 
```

This part is fairly generic for a Makefile. We have the `all` target, along with one to make the folders on the filesystem, if they don't exist yet. Finally, we take in the array of source files in the next target, compiling them as configured and outputting the object file in the appropriate folder:

```cpp
$(TOP)/build/bin/$(TARGET)/$(OUTPUT)_$(VERSION)_$(TARGET)$(TS_SAFE): $(OBJECTS) 
   $(LD) -o $@ $(OBJECTS) $(LIBDIRS) $(LIBS) 
   $(CP) $@ $@.debug 
ifeq ($(TARGET_OS), osx) 
   $(STRIP) -S $@ 
else 
   $(STRIP) -S --strip-unneeded $@      
endif 
```

After we have created all of the object files from our source files, we want to link them together, which happens in this step. We can also see where the binary will end up: in a `bin` sub-folder of the project's build folder.

The linker is called, and we create a copy of the resulting binary, which we post-fix with `.debug` to indicate that it is the version with all of the debug information. The original binary is then stripped of its debug symbols and other unneeded information, leaving us with a small binary to copy to the remote test system and a larger version with all of the debug information for when we need to analyze core dumps or do remote debugging.

What we also see here is a small hack that got added due to an unsupported command-line flag by Clang's linker, requiring the implementation of a special case. While working on cross-platform compiling and similar tasks, one is likely to run into many of such small details, all of which complicate the writing of a universal build system that simply works:

```cpp
clean: 
   $(RM) $(CPP_OBJECTS) 
   $(RM) $(C_OBJECTS) 
```

As a final step, we allow for the generated object files to be deleted.

The second sub-Makefile in `extern` is also of note, as it controls all of the underlying dependencies:

```cpp
ifndef TARGET 
$(error TARGET parameter not provided.) 
endif 

all: libboost poco 

all-linux-%: 
   $(MAKE) libboost poco 

all-qnx-%: 
   $(MAKE) libboost poco 

all-osx-%: 
   $(MAKE) libboost poco 

all-windows: 
   $(MAKE) libboost poco 
```

An interesting feature here is the dependency selector based on the target platform. If we have dependencies that shouldn't be compiled for a specific platform, we can skip them here. This feature also allows us to directly instruct this Makefile to compile all dependencies for a specific platform. Here, we allow for the targeting of QNX, Linux, OS X/macOS, and Windows, while ignoring the architecture:

```cpp
libboost: 
   cd boost-1.58.0 && $(MAKE) 

poco: 
   cd poco-1.7.4 && $(MAKE) 
```

The actual targets merely call another Makefile at the top of the dependency project, which in turn compiles that dependency and adds it to the build folder, where it can be used by the `Core`'s Makefile.

Of course, we can also directly compile the project from this Makefile using an existing build system, such as here for OpenSSL:

```cpp
openssl: 
   $(MKDIR) $(TOP)/build/$(TARGET)/openssl 
   $(MKDIR) $(TOP)/build/$(TARGET)/openssl/include 
   $(MKDIR) $(TOP)/build/$(TARGET)/openssl/lib 
   cd openssl-1.0.2 && ./Configure --openssldir="$(TOP)/build/$(TARGET)/openssl" shared os/compiler:$(TOOLCHAIN_NAME):$(OPENSSL_PARAMS) && \ 
     $(MAKE) build_libs 
   $(CP) openssl-1.0.2/include $(TOP)/build/$(TARGET)/openssl 
   $(CP) openssl-1.0.2/libcrypto.a $(TOP)/build/$(TARGET)/openssl/lib/. 
   $(CP) openssl-1.0.2/libssl.a $(TOP)/build/$(TARGET)/openssl/lib/. 
```

This code works through all of the usual steps of building OpenSSL by hand, before copying the resulting binaries to their target folders.

One issue with cross-platform build systems one may notice is that a common GNU tool such as Autoconf is extremely slow on OSes such as Windows, due to it launching many processes as it runs hundreds of tests. Even on Linux, this process can take a long time, which is very annoying and time consuming when running through the same build process multiple times a day.

The ideal case is having a simple Makefile in which everything is predefined and in a known state so that no library discovery and such are needed. This was one of the motivations behind adding the POCO library source code to one project and having a simple Makefile compile it:

```cpp
include ../../Makefile.$(TARGET) 

all: poco-foundation poco-json poco-net poco-util 

poco-foundation: 
   cd Foundation && $(MAKE) 

poco-json: 
   cd JSON && $(MAKE) 

poco-net: 
   cd Net && $(MAKE) 

poco-util: 
   cd Util && $(MAKE) 

clean: 
   cd Foundation && $(MAKE) clean 
   cd JSON && $(MAKE) clean 
   cd Net && $(MAKE) clean 
   cd Util && $(MAKE) clean 
```

This Makefile then calls the individual Makefile for each module, as in this example:

```cpp
include ../../../Makefile.$(TARGET) 

OUTPUT = libPocoNet.a 
INCLUDE = $(STD_INCLUDE) -Iinclude 
CFLAGS = $(STD_CFLAGS) $(INCLUDE) 
OBJROOT = $(TOP)/extern/poco-1.7.4/Net/$(TARGET) 
INCLOUT = $(TOP)/build/$(TARGET)/poco 
SOURCES := $(wildcard src/*.cpp) 
HEADERS := $(addprefix $(INCLOUT)/,$(wildcard include/Poco/Net/*.h)) 

OBJECTS := $(addprefix $(OBJROOT)/,$(notdir $(SOURCES:.cpp=.o))) 

all: makedir $(OBJECTS) $(TOP)/build/$(TARGET)/poco/lib/$(OUTPUT) $(HEADERS) 

$(OBJROOT)/%.o: src/%.cpp 
   $(CC) -c -o $@ $< $(CFLAGS) 

makedir: 
   $(MKDIR) $(TARGET) 
   $(MKDIR) $(TOP)/build/$(TARGET)/poco 
   $(MKDIR) $(TOP)/build/$(TARGET)/poco/lib 
   $(MKDIR) $(TOP)/build/$(TARGET)/poco/include 
   $(MKDIR) $(TOP)/build/$(TARGET)/poco/include/Poco 
   $(MKDIR) $(TOP)/build/$(TARGET)/poco/include/Poco/Net 

$(INCLOUT)/%.h: %.h 
   $(CP) $< $(INCLOUT)/$< 

$(TOP)/build/$(TARGET)/poco/lib/$(OUTPUT): $(OBJECTS) 
   -rm -f $@ 
   $(AR) rcs $@ $^ 

clean: 
   $(RM) $(OBJECTS) 
```

This Makefile compiles the entire `Net` module of the library. It's similar in structure to the one for compiling the project core source files. In addition to compiling the object files, it puts them into an archive so that we can link against it later, and copies this archive as well as the header files to their place in the build folder.

The main reason for compiling the library for the project was to allow for specific optimizations and tweaks that wouldn't be available with a precompiled library. By having everything but the basics stripped out of the library's original build system, trying out different settings was made very easy and even worked on Windows.

# Remote testing on real hardware

After we have done all of the local testing of our code and are reasonably certain that it should work on the real hardware, we can use the cross-compile build system to create a binary that we can then run on the target system.

At this point, we can simply copy the resulting binary and associated files to the target system and see whether it works. The more scientific way to do this is to use GDB. With the GDB server service installed on the target Linux system, we can connect to it with GDB from our PC, either via the network or a serial connection.

For SBCs running a Debian-based Linux installation, the GDB server can be easily installed:

```cpp
sudo apt install gdbserver  
```

Although it is called `gdbserver`, its essential function is that of a remote stub implementation for the debugger, which runs on the host system. This makes `gdbserver` very lightweight and simple to implement for new targets.

After this, we want to make sure that `gdbserver` is running by logging in to the system and starting it in one of a variety of ways. We can do so for TPC connections over the network like this:

```cpp
gdbserver host:2345 <program> <parameters>  
```

Or we can attach it to a running process:

```cpp
gdbserver host:2345 --attach <PID>  
```

The `host` part of the first argument refers to the name (or IP address) of the host system that will be connecting. This parameter is currently ignored, meaning that it can also be left empty. The port section has to be a port that is not currently in use on the target system.

Or we can use some kind of serial connection:

```cpp
gdbserver /dev/tty0 <program> <parameters>
gdbserver --attach /dev/tty0 <PID>  
```

The moment we launch `gdbserver`, it pauses the execution of the target application if it was already running, allowing us to connect with the debugger from the host system. While on the target system, we can run a binary that has been stripped of its debug symbols; these are required to be present in the binary that we use on the host side:

```cpp
$ gdb-multiarch <program>
(gdb) target remote <IP>:<port>
Remote debugging using <IP>:<port>  
```

At this point, debug symbols would be loaded from the binary, along with those from any dependencies (if available). Connecting over a serial connection would look similar, just with the address and port replaced with the serial interface path or name. The `baud` rate of the serial connection (if not the default 9,600 baud) is specified as a parameter to GDB when we're starting:

```cpp
$ gdb-multiarch -baud <baud rate> <program>  
```

Once we have told GDB the details of the remote target, we should see the usual GDB command-line interface appear, allowing us to step through, analyze, and debug the program as if it was running locally on our system.

As mentioned earlier in this chapter, we're using `gdb-multiarch` as this version of the GDB debugger supports different architectures, which is useful since we'll likely be running the debugger on an x86_64 system, whereas the SBC is very likely ARM-based, but could also be MIPS or x86 (i686).

In addition to running the application directly with `gdbserver`, we can also start `gdbserver` to just wait for a debugger to connect:

```cpp
gdbserver --multi <host>:<port>  
```

Or we can do this:

```cpp
gdbserver --multi <serial port>  
```

We would then connect to this remote target like this:

```cpp
$ gdb-multiarch <program>
(gdb) target extended-remote <remote IP>:<port>
(gdb) set remote exec-file <remote file path>
(gdb) run  
```

At this point, we should find ourselves at the GDB command-line interface again, with the program binary loaded on both target and host.

A big advantage of this method is that `gdbserver` does not exit when the application that's being debugged exits. In addition, this mode allows us to debug different applications simultaneously on the same target, assuming that the target supports this.

# Summary

In this chapter, we looked at how to develop and test embedded, OS-based applications. We learned how to install and use a cross-compilation toolchain, how to do remote debugging using GDB, and how to write a build system that allows us to compile for a wide variety of target systems with minimal effort required to add a new target.

At this point, you are expected to be able to develop and debug an embedded application for a Linux-based SBC or similar, while being able to work in an efficient way.

In the next chapter, we'll be looking at how to develop for and test applications for more constrained, MCU-based platforms.