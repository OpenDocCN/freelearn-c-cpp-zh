# Developing for Embedded Linux and Similar Systems

Small, SoC-based systems are everywhere these days, from smartphones, video game consoles, and smart television sets, to infotainment systems in cars and airplanes. Consumer devices relying on such systems are extremely common.

In addition to consumer devices, they're also found as part of industrial and building-level controller systems, where they monitor equipment, respond to input, and execute scheduled tasks for whole networks of sensors and actuators. Compared to MCUs, SoCs are not as resource-limited, usually running a full **operating system** (**OS**) such as a Linux-derived OS, VxWorks, or QNX.

In this chapter, we will cover the following topics:

*   How to develop drivers for OS-based embedded systems
*   Ways to integrate peripherals
*   How to handle and implement real-time performance requirements
*   Recognizing and dealing with resource limitations

# Embedded operating systems

An OS is typically used with an embedded system when you're writing your application directly for the system's hardware, which is an unrealistic proposal. What an OS provides to the application is a number of APIs that abstract away the hardware and functionality implemented using this hardware, such as network communications or video output.

The trade-off here is between convenience and both code size and complexity.

Whereas a bare metal implementation ideally implements only those features it needs, an operating system comes with a task scheduler, along with functionality that the application being run may not ever need. For this reason, it's important to know when to use an OS instead of developing directly for the hardware, understanding the complications that come with either.

Good reasons to use an OS are if you have to be able to run different tasks simultaneously (multitasking, or multithreading). Implementing your own scheduler from scratch is generally not worth the effort. Having the need to run a non-fixed number of applications, and being able to remove and add them at will, is also made significantly easier by using an OS.

Finally, features such as advanced graphics output, graphics acceleration (such as OpenGL), touch screens, and advanced network functionality (for example, SSH and encryption) can be made much easier to implement when you have access to an OS and readily accessible drivers, and the APIs related to them.

Commonly used embedded operating systems include the following:

| **Name** | **Vendor** | **License** | **Platforms** | **Details** |
| Raspbian | Community-based | Mainly GPL, similar | ARM (Raspberry Pi) | Debian Linux-based OS |
| Armbian | Community-based | GPLv2 | ARM (various boards) | Debian Linux-based OS |
| Android | Google | GPLv2, Apache | ARM, x86, x86_64 | Linux-based |
| VxWorks | Wind River (Intel) | Proprietary | ARM, x86, MIPS, PowerPC, SH-4 | RTOS, monolithic kernel |
| QNX | BlackBerry | Proprietary | ARMv7, ARMv8, x86 | RTOS, microkernel |
| Windows IoT | Microsoft | Proprietary | ARM, x86 | Formerly known as Windows Embedded |
| NetBSD | NetBSD Foundation | 2-clause BSD | ARM, 68k, MIPS, PowerPC, SPARC, RISC-V, x86, and so on | Most portable BSD-based OS |

What all of these OSes have in common is that they handle basic functionality such as memory and task management, while offering access to hardware and OS functionality using programming interfaces (APIs).

In this chapter, we will specifically focus on SoC and SBC-based systems, which reflects in the preceding list of operating systems. Each of these OSes is meant to be used on a system with at least a few megabytes of RAM and in the order of megabytes to gigabytes of storage.

If the target SoC or SBC is not yet targeted by an existing Linux distribution, or one wishes to heavily customize the system, one can use the tools from the Yocto Project ([http://www.yoctoproject.org/](http://www.yoctoproject.org/)).

Linux-based embedded OSes are quite prevalent, with Android being a well-known example. It is mostly used on smartphones, tablets, and similar devices, which heavily rely on graphical user interaction, while relying on the Android application infrastructure and related APIs. Due to this level of specialization, it is not well-suited to other use cases.

Raspbian is based on the very common Debian Linux distribution, aimed at basically just the Raspberry Pi series of SBCs. Armbian is similar, but covers a far wider range of SBCs. Both of these are community efforts. This is similar to the Debian project, which can also be used directly for embedded systems. The main advantage of the Raspbian, Armbian, and other similar projects is that they provide ready-made images to be used with the target SBC.

Like Linux-based OSes, NetBSD has the advantage of being open source, meaning that you have full access to the source code and can heavily customize any aspect of the operating system, including support for custom hardware. One big advantage NetBSD and similar BSD-based OSes have is that the OS is built from a single codebase, and managed by a single group of developers. This often simplifies the development and maintenance of an embedded project.

The BSD license (three- or two-clause) offers a major benefit for commercial projects, as this license only requires one to provide attribution instead of requiring the manufacturer to provide the full source code of the OS on request. This can be very relevant if one makes certain modifications to the source code, adding code modules that one wants to keep closed source.

Recent PlayStation gaming consoles, for example, use a modified version of FreeBSD, allowing Sony to heavily optimize the OS for the hardware and its use as a gaming console without having to release this code together with the rest of the OS's source code.

Proprietary options also exist, such as the offerings from BlackBerry (QNX) and Microsoft (Windows IoT, formerly Windows Embedded, formerly Windows CE). These tend to require a license fee per device and require the assistance of the manufacturer for any customization.

# Real-time OSes

The basic requirement for a real-time OS (RTOS) is that it can guarantee that tasks will be executed and finished within a certain time span. This allows one to use them for real-time applications where variability (jitter) between the execution times of a batch of the same task is not acceptable.

From this, we can draw the basic distinction between hard and soft real-time OSes: with low jitter, the OS is hard real-time, as it can guarantee that a given task will always be executed with practically the same delay. With higher jitter, the OS can usually but not always execute a task with the same delay.

Within these two categories, we can again distinguish between event-driven and time-sharing schedulers. The former switches tasks based on priority (priority scheduling), whereas the latter uses a timer to regularly switch tasks. Which design is better depends on what one uses the system for.

The main thing that time sharing has over event-driven schedulers is that since it gives far more CPU time to lower-priority tasks as well, it can make a multitasking system seem to run much smoother.

Generally, one would only use an RTOS if your project requirements are such that one must be able to guarantee that inputs can be handled within a strictly defined time window. For applications such as robotics and industrial applications, it can be crucial that an action is performed in exactly the same time span every time, with failure to do so resulting in the disruption of a production line or an inferior product.

With the example project that we will be looking at later in this chapter, we do not use an RTOS, but a regular Linux-based OS, as no hard timing requirements exist. Using an RTOS would impose an unneeded burden and likely increase complexity and costs.

One way to regard an RTOS is to get as close to the real-time nature of programming directly for the hardware (bare metal) without having to give up all of the conveniences of using a full-blown OS.

# Custom peripherals and drivers

A peripheral is defined as an ancillary device that adds I/O or other functionality to a computer system. This can be anything from an I2C, SPI, or SD card controller to an audio or graphics device. Most of those are part of the physical SoC, with others added via interfaces that the SoC exposes to the outside world. Examples of external peripherals would be RAM (via the RAM controller) and a **real-time clock** (**RTC**).

One issue that one will likely encounter when using cheaper SBCs such as the Raspberry Pi, Orange Pi, and countless similar systems is that they usually lack an RTC, meaning that when they are powered off, they no longer keep track of the time. The thought behind this is usually that those boards will be connected to the internet anyway, so the OS can use an online time service (**Network Time Protocol**, or **NTP**) to synchronize the system time, thus saving board space.

One might end up using an SBC in a situation where no internet connection is available, or where the delay before online time synchronization is unacceptable, or any of a myriad of other reasons. In this case, one may want to add an RTC peripheral to the board and configure the OS to make use of it.

# Adding an RTC

One can cheaply get RTCs as a ready-to-use module, often based around the DS1307 chip. This is a 5V module, which connects to the SBC (or MCU) via the I2C bus:

![](img/5881b3bf-15ed-4189-ab38-619ca08aa8c8.png)

This image is of a small DS1307-based RTC module. As one can see, it has the RTC chip, a crystal, and an MCU. The last of these is used to communicate with the host system, regardless of whether it is an SoC or MCU-based board. All that one requires is the ability to provide the desired voltage (and current) the RTC module operates on, along with an I2C bus.

After connecting the RTC module to the SBC board, the next goal is to have the OS also use it. For this, we must make sure that the I2C kernel module is loaded so that we can use I2C devices.

Linux distributions for SBCs, such as Raspbian and Armbian, usually come with drivers for a number of RTC modules. This allows us to relatively quickly set up the RTC module and integrate it with the OS. With the module we looked at earlier, we require the I2C and DS1307 kernel modules. For a Raspbian OS on a first-generation Raspberry Pi SBC, these modules would be called `i2c-dev`, `2cbcm2708`, and `rtc-ds1307`.

First, you have to enable these modules so that they are loaded when the system starts. For Raspbian Linux, one can edit the `/etc/modules` file to do so, as well as other configuration tools made available for this platform. After a reboot, we should be able to detect the RTC device on the I2C bus using an I2C scanner tool.

With the RTC device working, we can remove the fake-hwclock package on Raspbian. This is a simple module that fakes an RTC, but merely stores the current time in a file before the system is shut down so that on the next boot the filesystem dates and similar will be consistent due to resuming from that stored date and time, without any new files one creates suddenly being *older* than the existing files.

Instead, we'll be using the hwclock utility, which will use any real RTC to synchronize the system time with. This requires one to modify the way the OS starts, with the location of the RTC module passed as boot parameters in the following form:

```cpp
rtc.i2c=ds1307,1,0x68
```

This will initialize an RTC (`/dev/rtc0`) device on the I2C bus, with address 0x68.

# Custom drivers

The exact format and integration of drivers (kernel modules) with the OS kernel differs for each OS and thus would be impossible to fully cover here. We will, however, look at how the driver for the RTC module we used earlier is implemented for Linux.

In addition, we will look at how to use an I2C peripheral from user space later in this chapter, in the club room monitoring example. Using a user space-based driver (library) is often a good alternative to implementing it as a kernel module.

The RTC functionality is integrated into the Linux kernel, with the code for it found in the `/drivers/rtc` folder (on GitHub, at [https://github.com/torvalds/linux/tree/master/drivers/rtc](https://github.com/torvalds/linux/tree/master/drivers/rtc)).

The `rtc-ds1307.c` file contains two functions we need to read and set the RTC, respectively: `ds1307_get_time()` and `ds1307_set_time()`. The basic functionality of these functions is very similar to what we'll be using in the club room monitoring example later in this chapter, where we simply integrate I2C device support into our application.

A major advantage of communicating with I2C, SPI, and other such peripherals from user space is that we are not limited by the compile environment supported by the OS kernel. Taking the Linux kernel as an example, it is written mostly in C with some assembly. Its APIs are C-style APIs and thus we would have to use a distinctly C-style coding approach to writing our kernel modules.

Obviously, this would negate most of the advantages, not to mention the point, of attempting to write these modules in C++ to begin with. When moving our module code to user space and using it either as part of an application or as a shared library, we have no such limitations and can freely use any and all C++ concepts and functionality.

For completeness' sake, the basic template for a Linux kernel module looks as follows:

```cpp
#include <linux/module.h>       // Needed by all modules 
#include <linux/kernel.h>       // Needed for KERN_INFO 

int init_module() { 
        printk(KERN_INFO "Hello world.n"); 

        return 0; 
} 

void cleanup_module() { 
        printk(KERN_INFO "Goodbye world.n"); 
} 
```

This is the requisite Hello World example, written in C++-style.

One final consideration when considering kernel- and user space-based driver modules is that of context switches. From an efficiency point of view, kernel modules are faster and have lower latency because the CPU does not have to switch from a user to kernel space context and back repeatedly to communicate with a device and pass messages from it back to the code communicating with it.

For high bandwidth devices (such as storage and capturing), this could make the difference between a smoothly functioning system and one that severely lags and struggles to perform its tasks.

However, when considering the club room monitoring example in this chapter and its occasional use of an I2C device, it should be obvious that a kernel module would be severe overkill without any tangible benefits.

# Resource limitations

Even though SBCs and SoCs tend to be fairly powerful, they are still no direct comparison to a modern desktop system or server. They have distinct limits in terms of RAM, storage size, and lack of expansion options.

With wildly varying amounts of (permanently installed) RAM, you have to consider the memory needs of the applications one wishes to run on the system before even considering the relatively sluggish CPU performance.

As SBCs tend to not have any, or significant amounts of, storage with a high endurance rate (meaning it can be written to often without limited write cycles to take into account), they generally do not have swap space and keep everything in the available RAM. Without the fallback of swap, any memory leaks and excessive memory usage will rapidly lead to a non-functioning or constantly restarting system.

Even though CPU performance on SBCs has increased significantly over the years for commonly available models, it is generally still advisable to use a cross-compiler to produce code for the SBC on a fast desktop system or server.

More on development issues and solutions will be covered in [Chapter 6](7d5d654f-a027-4825-ab9e-92c369b576a8.xhtml), *Testing OS-Based Applications*, and [Appendix](ddead19d-4726-49ec-b780-34689efdd0b7.xhtml), *Best Practices*.

# Example – club room monitoring

In this section, we will be looking at a practical implementation of an SBC-based solution that performs the following functionality for a club room:

*   Monitors the status of the club door's lock
*   Monitors the club status switch
*   Sends status change notifications over MQTT
*   Provides a REST API for the current club status
*   Controls status lights
*   Controls the power in the club room

The basic use case here is that we have a club room for which we want to be able to monitor the status of its lock, and have a switch inside the club to regulate whether the non-permanent power outlets in the club are powered on or not. Turning the club status switch to *on* would provide power to those outlets. We also want to send out a notification over MQTT so that other devices in the club room or elsewhere can update their status.

MQTT is a simple, binary publish/subscribe protocol on top of TCP/IP. It offers a lightweight communication protocol, suitable for resource-restricted applications such as sensor networks. Each MQTT client communicates with a central server: the MQTT broker.

# Hardware

The block diagram of the `clubstatus` system looks as follows:

![](img/bfb888d8-bf9f-4dab-9366-d473d1c7dd7f.png)

For the SBC platform, we use a Raspberry Pi, either the Raspberry Pi B+ model or a newer member of the B-series, such as the Raspberry Pi 3 Model B:

![](img/97073e5a-4311-4381-bf99-025dc76193c3.png)

The main features we are looking for in the SBC system are an Ethernet connection and, of course, the Raspberry Pi-compatible **general-purpose input/output** (**GPIO**) header.

With this board, we'll use a standard Raspbian OS installation on the μSD card. No special configuration is needed beyond this. The primary reason for choosing the B+ model or similar is that these have a standard mounting hole pattern.

# Relays

To control the status lights and the non-permanent power outlets in the room, we use a number of relays, in this case four relays:

| **Relay** | **Function** |
| 0 | Power status of non-permanent outlets |
| 1 | Green status light |
| 2 | Yellow status light |
| 3 | Red status light |

The idea here is that the power status relay is connected to a switch that controls the mains power to outlets that are not powered when the club status is off. The status lights indicate the current club status. The next section provides the details on the implementation of this concept.

To simplify the design, we will use a ready-made relay board containing four relays, which are driven by an NXP PCAL9535A I/O port chip (GPIO expander) connected to the I2C bus of the Raspberry Pi SBC:

![](img/5853c932-146d-46f6-abc9-8537243bc361.png)

This particular board is the Seeed Studio Raspberry Pi Relay Board v1.0: [http://wiki.seeedstudio.com/Raspberry_Pi_Relay_Board_v1.0/](http://wiki.seeedstudio.com/Raspberry_Pi_Relay_Board_v1.0/). It offers the four relays we require, allowing us to switch lights and switches up to 30 VDC (direct current) or 250 VAC (alternating current). This allows one to connect practically any type of lighting and further relays and kin.

The connection with the SBC is made by stacking the relay board on top of the SBC using its GPIO header, which allows us to add further boards on top of the relay board. This allows us to add the debounce functionality to the system, as indicated in the wiring plan diagram.

# Debounce

The debounce board has the debouncing of switch signals as a requirement, as well as providing the Raspberry Pi board with power. The theory and reason behind the debouncing of mechanical switches is that the signal provided by those switches is not clean, meaning that they don't immediately switch from open to closed. They will briefly close (make contact) before the springiness of the metal contacts causes them to open again and rapidly move between these two states, before finally settling into its final position, as we can see in the following diagram from an oscilloscope connected to a simple switch:

![](img/279f998d-afcf-427d-aeae-3b6f2fcc04e6.png)

The result of this property is that the signal that arrives at the SBC's GPIO pins will rapidly change for a number of milliseconds (or worse). Performing any kind of action based upon these switch input changes would therefore lead to immense problems, as one cannot easily distinguish between a desired switch change and the rapid bouncing of the switch contacts during this change.

It is possible to debounce a switch either in hardware or software. The latter solution involves the starting of a timer when the state of the switch first changes. The assumption behind this is that after a certain time (in milliseconds) has expired, the switch is in a stable state and can be safely read out. This approach has disadvantages in that it puts an extra burden on the system by taking up one or more timers, or pausing the program's execution. Also, using interrupts on the input for the switch requires one to disable interrupts while the timer is running, adding further complexity to the code.

Debouncing in hardware can be done using discrete components, or using an SR latch (consisting of two NAND gates). For this application, we will use the following circuit, which works well with the most commonly used SPST (single-pole, single-throw) type of switch:

![](img/78c9ed38-dcb3-4eb3-85a1-733bf19c5bd6.png)

The concept behind this circuit is that when the switch is open, the capacitor is charged via R1 (and D1), causing the input on the inverting Schmitt trigger circuit (U1) to go high, resulting in the GPIO pin of the SBC connected to the output of U1 to read low. When the switch closes, the capacitor is discharged to the ground over R2.

Both the charging and discharging will take a certain amount of time, which adds latency before a change is registered on the input of U1\. The charging and discharging rates are determined by the values of R1 and R2, the formulas for which are as follows:

*   Charging: ![](img/0ec96986-b5cd-4e73-9ad3-38681f477c37.png)
*   Discharging: ![](img/6472a5a6-f642-45ee-89c1-8826d4387364.png)

Here, *V(t)* is the voltage at time *t* (in seconds). *V[S]* is the source voltage and *t* is the time in seconds after the source voltage has been applied. R is the circuit resistance in Ohm and C the capacitance in farads. Finally, *e* is a mathematical constant with the value of 2.71828 (approximately), also known as Euler's number.

For the charging and discharging of capacitors, the RC time constant, tau (τ), is used, which is defined as follows:

![](img/4c70f6d7-152d-44d3-a793-6c3786e82007.png)

This defines the time it takes for the capacitor to be charged up to 63.2% (1τ), then 86% (2τ). The discharging of a capacitor for 1τ from fully charged will reduce its charge to 37%, and 13.5% after 2τ. One of the things one notices here is that a capacitor is never fully charged or discharged; the process of charging or discharging just slows down to the point where it becomes almost imperceptible.

With the values that we used for our debounce circuit, we get the following charge time constant for charging:

![](img/32b180c1-b53b-4bb7-82ed-e324fb5f3094.png) 

The discharge time is as follows:

![](img/5f7819ea-e9e2-461d-a593-c4fdd7ba7a39.png)

This corresponds to 51 and 22 microseconds, respectively.

Like any Schmitt trigger, it has so-called hysteresis, meaning that it has dual thresholds. This effectively adds a dead zone in the output response above and below, which the output will not change:

![](img/bc437d21-ffee-4425-9d2f-3184fc28df6d.png)

The hysteresis from a Schmitt trigger is usually used to remove noise from an incoming signal by setting explicit trigger levels. Even though the RC circuit we are already using should filter out practically all noises, adding a Schmitt trigger adds that little bit more insurance without any negative repercussions.

When available, it is also possible to use the hysteresis functionality of an SBC's GPIO pins. For this project and the chosen debounce circuit, we also want the inverting property of the chip so that we get the expected high/low response for the connected switch instead of having to invert the meaning in software.

# Debounce HAT

Using the information and debounce circuit from the previous section, a prototype board is assembled:

![](img/3c655ae5-9b38-479e-8178-6676cf9e0cd9.png)

This prototype implements two debounce channels for the two switches that are required by the project. It also adds a screw terminal to connect the SBC power connection to. This allows one to power the SBC via the 5V header pins instead of having to use the micro-USB connector of the Raspberry Pi. For integration purposes, it's usually easier to just run the wires directly from the power supply into a screw terminal or similar than to bodge on a micro-USB plug.

This prototype is, of course, not a proper HAT, as defined by the Raspberry Pi Foundation's rules. These require the following features:

*   It has a valid EEPROM containing vendor information, GPIO map, and device information connected to the `ID_SC` and `ID_SD` I2C bus pins on the Raspberry Pi SBC
*   It has the modern 40-pin (female) GPIO connector, also spacing the HAT from the SBC by at least 8 millimeters
*   It follows the mechanical specification
*   If providing power to the SBC via the 5V pins, the HAT has to be able to provide at least 1.3 amperes continuously

With the required I2C EEPROM (CAT24C32) and other features added, we can see what a full version using the six channels offered by the inverting hex Schmitt trigger IC (40106) looks like:

![](img/f94f1673-7d99-4eb5-b684-35e094620374.png)

The files for this KiCad project can be found at the author's GitHub account at [https://github.com/MayaPosch/DebounceHat](https://github.com/MayaPosch/DebounceHat). With the extended number of channels, it would be relatively easy to integrate further switches, relays, and other elements into the system, possibly monitoring things like windows and such with various sensors that output a high/low signal.

# Power

For our project, the required voltages we need are 5V for the Raspberry Pi board and a second voltage for the lights that we switch on and off via the relays. The power supply we pick has to be able to provide sufficient power to the SBC and the lights. For the former, 1-2 A should suffice, with the latter depending on the lights being used and their power requirements.

# Implementation

The monitoring service will be implemented as a basic `systemd` service, meaning that it will be started by the operating system when the system starts, and the service can be monitored and restarted using all the regular systemd tools.

We will have the following dependencies:

*   POCO
*   WiringPi
*   libmosquittopp (and libmosquitto)

The libmosquitto dependency ([https://mosquitto.org/man/libmosquitto-3.html](https://mosquitto.org/man/libmosquitto-3.html)) is used to add MQTT support. The libmosquittopp dependency is a wrapper around the C-based API to provide a class-based interface, which makes integration into C++ projects easier.

The POCO framework ([https://pocoproject.org/](https://pocoproject.org/)) is a highly portable set of C++ APIs, which provides everything from network-related functions (including HTTP) to all common low-level functions. In this project, its HTTP server will be used, along with its support for handling configuration files.

Finally, WiringPi ([http://wiringpi.com/](http://wiringpi.com/)) is the de facto standard header for accessing and using the GPIO header features on the Raspberry Pi and compatible systems. It implements APIs to communicate with I2C devices and UARTs, and uses PWM and digital pins. In this project, it allows us to communicate with the relay board and the debounce board.

The current version of this code can also be found at the author's GitHub account: [https://github.com/MayaPosch/ClubStatusService](https://github.com/MayaPosch/ClubStatusService).

We will start with the main file:

```cpp
#include "listener.h"

 #include <iostream>
 #include <string>

 using namespace std;

 #include <Poco/Util/IniFileConfiguration.h>
 #include <Poco/AutoPtr.h>
 #include <Poco/Net/HTTPServer.h>

 using namespace Poco::Util;
 using namespace Poco;
 using namespace Poco::Net;

 #include "httprequestfactory.h"
 #include "club.h"
```

Here, we include some basic STL functionality, along with the HTTP server and `ini` file support from POCO. The listener header is for our MQTT class, with the `httprequestfactory` and club headers being for the HTTP server and the main monitoring logic, respectively:

```cpp
int main(int argc, char* argv[]) {
          Club::log(LOG_INFO, "Starting ClubStatus server...");
          int rc;
          mosqpp::lib_init();

          Club::log(LOG_INFO, "Initialised C++ Mosquitto library.");

          string configFile;
          if (argc > 1) { configFile = argv[1]; }
          else { configFile = "config.ini"; }

          AutoPtr<IniFileConfiguration> config;
          try {
                config = new IniFileConfiguration(configFile);
          }
          catch (Poco::IOException &e) {
                Club::log(LOG_FATAL, "Main: I/O exception when opening configuration file: " + configFile + ". Aborting...");
                return 1;
          }

          string mqtt_host = config->getString("MQTT.host", "localhost");
          int mqtt_port = config->getInt("MQTT.port", 1883);
          string mqtt_user = config->getString("MQTT.user", "");
          string mqtt_pass = config->getString("MQTT.pass", "");
          string mqtt_topic = config->getString("MQTT.clubStatusTopic",    "/public/clubstatus");
          bool relayactive = config->getBool("Relay.active", true);
          uint8_t relayaddress = config->getInt("Relay.address", 0x20);
```

In this section, we initialize the MQTT library (libmosquittopp) and try to open the configuration file, using the default path and name if nothing is specified in the command-line parameters.

POCO's `IniFileConfiguration` class is used to open and read in the configuration file, throwing an exception if it cannot be found or opened. POCO's `AutoPtr` is equivalent to C++11's `unique_ptr`, allowing us to create a new heap-based instance without having to worry about disposing of it later.

Next, we read out the values that we are interested in for the MQTT and relay board functionality, specifying defaults where it makes sense to do so:

```cpp
Listener listener("ClubStatus", mqtt_host, mqtt_port, mqtt_user, mqtt_pass);

    Club::log(LOG_INFO, "Created listener, entering loop...");

    UInt16 port = config->getInt("HTTP.port", 80);
    HTTPServerParams* params = new HTTPServerParams;
    params->setMaxQueued(100);
    params->setMaxThreads(10);
    HTTPServer httpd(new RequestHandlerFactory, port, params);
    try {
          httpd.start();
    }
    catch (Poco::IOException &e) {
          Club::log(LOG_FATAL, "I/O Exception on HTTP server: port already in use?");
          return 1;
    }
    catch (...) {
          Club::log(LOG_FATAL, "Exception thrown for HTTP server start. Aborting.");
          return 1;
    }
```

In this section, we start the MQTT class, providing it with the parameters it needs to connect to the MQTT broker. Next, the HTTP server's configuration details are read out and a new `HTTPServer` instance is created.

The server instance is configured with the provided port and some limits for the maximum number of threads the HTTP server is allowed to use, as well as for the maximum queued connections it can keep. These parameters are useful to optimize system performance and fit code like this into systems with fewer resources to spare.

New client connections are handled by the custom `RequestHandlerFactory` class, which we will look at later:

```cpp

             Club::mqtt = &listener;
             Club::start(relayactive, relayaddress, mqtt_topic);

             while(1) {
                   rc = listener.loop();
                   if (rc){
                         Club::log(LOG_ERROR, "Disconnected. Trying to 
                         reconnect...");
                         listener.reconnect();
                   }
             }

             mosqpp::lib_cleanup();
             httpd.stop();
             Club::stop();

             return 0;
 }
```

Finally, we assign a reference to the Listener instance we created to the static `Club` class's `mqtt` member. This will allow the `Listener` object to be used more easily later on, as we will see.

With calling `start()` on `Club`, the monitoring and configuring of the connected hardware will be handled and we are done with that aspect in the main function.

Finally, we enter a loop for the MQTT class, ensuring that it remains connected to the MQTT broker. Upon leaving the loop, we will clean up resources and stop the HTTP server and others. However, since we are in an infinite loop here, this code will not be reached with this implementation.

Since this implementation would be run as a service that runs 24/7, a way to terminate the service cleanly is not an absolute requirement. A relatively easy way to do this would be to add a signal handler that would interrupt the loop once triggered. For simplicity's sake, this has been left out of this project.

# Listener

The class declaration for the `Listener` class looks like this:

```cpp
class Listener : public mosqpp::mosquittopp {
          //

 public:
          Listener(string clientId, string host, int port, string user, string pass);
          ~Listener();

          void on_connect(int rc);
          void on_message(const struct mosquitto_message* message);
          void on_subscribe(int mid, int qos_count, const int* granted_qos);

          void sendMessage(string topic, string& message);
          void sendMessage(string& topic, char* message, int msgLength);
 };
```

This class provides a simple API to connect to an MQTT broker and send messages to said broker. We inherit from the `mosquittopp` class, re-implementing a number of callback methods to handle the events of connecting newly received messages and completed subscriptions to MQTT topics.

Next, let's have a look at the implementation:

```cpp
#include "listener.h"

 #include <iostream>

 using namespace std;
 Listener::Listener(string clientId, string host, int port, string user, string pass) : mosquittopp(clientId.c_str()) {
          int keepalive = 60;
          username_pw_set(user.c_str(), pass.c_str());
          connect(host.c_str(), port, keepalive);
 }

 Listener::~Listener() {
          //
 }
```

In the constructor, we assign the unique MQTT client identification string using the mosquittopp class's constructor. We use a default value for the keep alive setting of 60 seconds, meaning the time for which we will keep a connection open to the MQTT broker without any side sending a control or other message.

After setting a username and password, we connect to the MQTT broker:

```cpp
void Listener::on_connect(int rc) {
    cout << "Connected. Subscribing to topics...n";

          if (rc == 0) {
                // Subscribe to desired topics.
                string topic = "/club/status";
                subscribe(0, topic.c_str(), 1);
          }
          else {
                cerr << "Connection failed. Aborting subscribing.n";
          }
 }
```

This callback function is called whenever a connection attempt has been made with the MQTT broker. We check the value of `rc` and if the value is zero—indicating success—we start subscribing to any desired topics. Here, we subscribe to just one topic: /club/status. If any other MQTT clients send a message to this topic, we will receive it in the following callback function:

```cpp

 void Listener::on_message(const struct mosquitto_message* message) {
          string topic = message->topic;
          string payload = string((const char*) message->payload, message->payloadlen);

          if (topic == "/club/status") {
                string topic = "/club/status/response";
                char payload[] = { 0x01 }; 
                publish(0, topic.c_str(), 1, payload, 1); // QoS 1\.   
          }     
 }
```

In this callback function, we receive a struct with the MQTT topic and payload. We then compare the topic to the topic strings we subscribed to, which in this case is just the /club/status topic. Upon receiving a message for this topic, we publish a new MQTT message with a topic and payload. The last parameter is the **quality of service** (**QoS**) value, with in this case setting is the *deliver at least once* flag. This guarantees that at least one other MQTT client will receive our message.

The MQTT payload is always a binary, that is, `1` in this example. To make it reflect the status of the club room (opened or closed), we would have to integrate the response from the static `Club` class, which we will be looking at in the next section.

First, we look at the remaining functions for the `Listener` class:

```cpp
 void Listener::on_subscribe(int mid, int qos_count, const int* granted_qos) {
          // 
 }

 void Listener::sendMessage(string topic, string &message) {
          publish(0, topic.c_str(), message.length(), message.c_str(), true);
 }

 void Listener::sendMessage(string &topic, char* message, int msgLength) {
          publish(0, topic.c_str(), msgLength, message, true);
 }
```

The callback function for a new subscription is left empty here, but could be used to add logging or such functionality. Furthermore, we have an overloaded `sendMessage()` function, which allows other parts of the application to also publish MQTT messages.

The main reason to have these two different functions is that sometimes it's easier to use a `char*` array to send, for example, an array of 8-bit integers as part of a binary protocol, whereas other times an STL string is more convenient. This way, we get the best of both worlds, without having to convert one or the other whenever we wish to send an MQTT message anywhere in our code.

The first parameter to `publish()` is the message ID, which is a custom integer we can assign ourselves. Here, we leave it at zero. We also make use of the *retain* flag (last parameter), setting it to true. This implies that whenever a new MQTT client subscribes to the topic we published a retained message on, this client will always receive the last message that was published on that particular topic.

Since we will be publishing the status of the club rooms on an MQTT topic, it is desirable that the last status message is retained by the MQTT broker so that any client that uses this information will immediately receive the current status the moment it connects to the broker, instead of having to wait for the next status update.

# Club

The club header declares the classes that form the core of the project, and is responsible for dealing with the inputs from the switches, controlling the relays, and updating the status of the club room:

```cpp
#include <wiringPi.h>
 #include <wiringPiI2C.h>
```

The first thing of note in this header file are the includes. They add the basic WiringPi GPIO functionality to our code, as well as those for I2C usage. Further WiringPi one could include for other projects requiring such functionality would be SPI, UART (serial), software PWM, Raspberry Pi (Broadcom SoC) specific functionality, and others:

```cpp
enum Log_level {
    LOG_FATAL = 1,
    LOG_ERROR = 2,
    LOG_WARNING = 3,
    LOG_INFO = 4,
    LOG_DEBUG = 5
 };
```

We define the different log levels we will be using as an `enum`:

```cpp
 class Listener;
```

We forward declare the `Listener` class, as we will be using it in the implementation for these classes, but don't want to include the entire header for it yet:

```cpp
class ClubUpdater : public Runnable {
          TimerCallback<ClubUpdater>* cb;
          uint8_t regDir0;
          uint8_t regOut0;
          int i2cHandle;
          Timer* timer;
          Mutex mutex;
          Mutex timerMutex;
          Condition timerCnd;
          bool powerTimerActive;
          bool powerTimerStarted;

 public:
          void run();
          void updateStatus();
          void writeRelayOutputs();
          void setPowerState(Timer &t);
 };
```

The `ClubUpdater` class is responsible for configuring the I2C-based GPIO expander, which controls the relays, as well as handling any updates to the club status. A `Timer` instance from the POCO framework is used to add a delay to the power status relay, as we will see when we look at the implementation.

This class inherits from the POCO `Runnable` class, which is the base class that's expected by the POCO `Thread` class, which is a wrapper around native threads.

The two `uint8_t` member variables mirror two registers on the I2C GPIO expander device, allowing us to set the direction and value of the output pins on the device, which effectively controls the attached relays:

```cpp
class Club {
          static Thread updateThread;
          static ClubUpdater updater;

          static void lockISRCallback();
          static void statusISRCallback();

 public:
          static bool clubOff;
          static bool clubLocked;
          static bool powerOn;
          static Listener* mqtt;
          static bool relayActive;
          static uint8_t relayAddress;
          static string mqttTopic;      // Topic we publish status updates on.

          static Condition clubCnd;
          static Mutex clubCndMutex;
          static Mutex logMutex;
          static bool clubChanged ;
          static bool running;
          static bool clubIsClosed;
          static bool firstRun;
          static bool lockChanged;
          static bool statusChanged;
          static bool previousLockValue;
          static bool previousStatusValue;

          static bool start(bool relayactive, uint8_t relayaddress, string topic);
          static void stop();
          static void setRelay();
          static void log(Log_level level, string msg);
 };
```

The `Club` class can be regarded as the input side of the system, setting up and handling the ISRs (interrupt handlers), as well as acting as the central (static) class with all of the variables pertaining to the club status, such as the status of the lock switch, status switch, and status of the power system (club open or closed).

This class is made fully static so that it can be used freely by different parts of the program to inquire about the room status.

Moving on, here is the implementation:

```cpp
#include "club.h"

 #include <iostream>

 using namespace std;

 #include <Poco/NumberFormatter.h>

 using namespace Poco;

 #include "listener.h"
```

Here, we include the `Listener` header so that we can use it. We also include the POCO `NumberFormatter` class to allow us to format integer values for logging purposes:

```cpp
 #define REG_INPUT_PORT0              0x00
 #define REG_INPUT_PORT1              0x01
 #define REG_OUTPUT_PORT0             0x02
 #define REG_OUTPUT_PORT1             0x03
 #define REG_POL_INV_PORT0            0x04
 #define REG_POL_INV_PORT1            0x05
 #define REG_CONF_PORT0               0x06
 #define REG_CONG_PORT1               0x07
 #define REG_OUT_DRV_STRENGTH_PORT0_L 0x40
 #define REG_OUT_DRV_STRENGTH_PORT0_H 0x41
 #define REG_OUT_DRV_STRENGTH_PORT1_L 0x42
 #define REG_OUT_DRV_STRENGTH_PORT1_H 0x43
 #define REG_INPUT_LATCH_PORT0        0x44
 #define REG_INPUT_LATCH_PORT1        0x45
 #define REG_PUD_EN_PORT0             0x46
 #define REG_PUD_EN_PORT1             0x47
 #define REG_PUD_SEL_PORT0            0x48
 #define REG_PUD_SEL_PORT1            0x49
 #define REG_INT_MASK_PORT0           0x4A
 #define REG_INT_MASK_PORT1           0x4B
 #define REG_INT_STATUS_PORT0         0x4C
 #define REG_INT_STATUS_PORT1         0x4D
 #define REG_OUTPUT_PORT_CONF         0x4F
```

Next, we define all of the registers of the target GPIO expander device, the NXP PCAL9535A. Even though we only use two of these registers, it's generally a good practice to add the full list to simplify later expansion of the code. A separate header can be used as well to allow one to easily use different GPIO expanders without significant changes to your code, or any at all:

```cpp
 #define RELAY_POWER 0
 #define RELAY_GREEN 1
 #define RELAY_YELLOW 2
 #define RELAY_RED 3
```

Here, we define which functionality is connected to which relay, corresponding to a specific output pin of the GPIO expander chip. Since we have four relays, four pins are used. These are connected to the first bank (of two in total) of eight pins on the chip.

Naturally, it is important that these definitions match up with what is physically hooked up to those relays. Depending on the use case, one could make this configurable as well:

```cpp
bool Club::clubOff;
 bool Club::clubLocked;
 bool Club::powerOn;
 Thread Club::updateThread;
 ClubUpdater Club::updater;
 bool Club::relayActive;
 uint8_t Club::relayAddress;
 string Club::mqttTopic;
 Listener* Club::mqtt = 0;

 Condition Club::clubCnd;
 Mutex Club::clubCndMutex;
 Mutex Club::logMutex;
 bool Club::clubChanged = false;
 bool Club::running = false;
 bool Club::clubIsClosed = true;
 bool Club::firstRun = true;
 bool Club::lockChanged = false;
 bool Club::statusChanged = false;
 bool Club::previousLockValue = false;
 bool Club::previousStatusValue = false;
```

As `Club` is a fully static class, we initialize all of its member variables before we move into the `ClubUpdater` class's implementation:

```cpp
void ClubUpdater::run() {
    regDir0 = 0x00;
    regOut0 = 0x00;
    Club::powerOn = false;
    powerTimerActive = false;
    powerTimerStarted = false;
    cb = new TimerCallback<ClubUpdater>(*this, &ClubUpdater::setPowerState);
    timer = new Timer(10 * 1000, 0);
```

When we start an instance of this class, its `run()` function gets called. Here, we set a number of defaults. The direction and output register variables are initially set to zero. The club room power status is set to false, and the power timer-related Booleans are set to false, as the power timer is not active yet. This timer is used to set a delay before the power is turned on or off, as we will see in more detail in a moment.

By default, the delay on this timer is ten seconds. This can, of course, also be made configurable:

```cpp
if (Club::relayActive) {
    Club::log(LOG_INFO, "ClubUpdater: Starting i2c relay device.");
    i2cHandle = wiringPiI2CSetup(Club::relayAddress);
    if (i2cHandle == -1) {
        Club::log(LOG_FATAL, string("ClubUpdater: error starting          
        i2c relay device."));
        return;
    }

    wiringPiI2CWriteReg8(i2cHandle, REG_CONF_PORT0, 0x00);
    wiringPiI2CWriteReg8(i2cHandle, REG_OUTPUT_PORT0, 0x00);

    Club::log(LOG_INFO, "ClubUpdater: Finished configuring the i2c 
    relay device's registers.");
}
```

Next, we set up the I2C GPIO expander. This requires the I2C device address, which we passed to the `Club` class earlier on. What this setup function does is ensure that there is an active I2C device at this address on the I2C bus. After this, it should be ready to communicate with. It is also possible to skip this step via setting the relayActive variable to false. This is done by setting the appropriate value in the configuration file, which is useful when running integration tests on a system without an I2C bus or connected device.

With the setup complete, we write the initial values of the direction and output registers for the first bank. Both are written with null bytes so that all eight pins they control are set to both output mode and to a binary zero (low) state. This way, all relays connected to the first four pins are initially off:

```cpp
          updateStatus();

          Club::log(LOG_INFO, "ClubUpdater: Initial status update complete.");
          Club::log(LOG_INFO, "ClubUpdater: Entering waiting condition.");

          while (Club::running) {
                Club::clubCndMutex.lock();
                if (!Club::clubCnd.tryWait(Club::clubCndMutex, 60 * 1000)) {.
                      Club::clubCndMutex.unlock();
                      if (!Club::clubChanged) { continue; }
                }
                else {
                      Club::clubCndMutex.unlock();
                }

                updateStatus();
          }
 }
```

After completing these configuration steps, we run the first update of the club room status, using the same function that will also be called later on when the inputs change. This results in all of the inputs being checked and the outputs being set to a corresponding status.

Finally, we enter a waiting loop. This loop is controlled by the `Club::running` Boolean variable, allowing us to break out of it via a signal handler or similar. The actual waiting is performed using a condition variable, which we wait for here until either a time-out occurs on the one-minute wait (after which, we return to waiting after a quick check), or we get signaled by one of the interrupts that we will set later on for the inputs.

Moving on, we look at the function that's used to update the status of the outputs:

```cpp
void ClubUpdater::updateStatus() {
    Club::clubChanged = false;

    if (Club::lockChanged) {
          string state = (Club::clubLocked) ? "locked" : "unlocked";
          Club::log(LOG_INFO, string("ClubUpdater: lock status changed to ") + state);
          Club::lockChanged = false;

          if (Club::clubLocked == Club::previousLockValue) {
                Club::log(LOG_WARNING, string("ClubUpdater: lock interrupt triggered, but value hasn't changed. Aborting."));
                return;
          }

          Club::previousLockValue = Club::clubLocked;
    }
    else if (Club::statusChanged) {           
          string state = (Club::clubOff) ? "off" : "on";
          Club::log(LOG_INFO, string("ClubUpdater: status switch status changed to ") + state);
          Club::statusChanged = false;

          if (Club::clubOff == Club::previousStatusValue) {
                Club::log(LOG_WARNING, string("ClubUpdater: status interrupt triggered, but value hasn't changed. Aborting."));
                return;
          }

          Club::previousStatusValue = Club::clubOff;
    }
    else if (Club::firstRun) {
          Club::log(LOG_INFO, string("ClubUpdater: starting initial update run."));
          Club::firstRun = false;
    }
    else {
          Club::log(LOG_ERROR, string("ClubUpdater: update triggered, but no change detected. Aborting."));
          return;
    }
```

The first thing we do when we enter this update function is to ensure that the `Club::clubChanged` Boolean is set to false so that it can be set again by one of the interrupt handlers.

After this, we check what has changed exactly on the inputs. If the lock switch got triggered, its Boolean variable will have been set to true, or the variable for the status switch will likely have been triggered. If this is the case, we reset the variable and compare the newly read value with the last known value for that input.

As a sanity check, we ignore the triggering if the value hasn't changed. This could happen if the interrupt got triggered due to noise, such as when the signal wire for a switch runs near power lines. Any fluctuation in the latter would induce a surge in the former, which can trigger the GPIO pin's interrupt. This is one obvious example of both the reality of dealing with a non-ideal physical world and a showcase for the importance of both the hardware and software in how they affect the reliability of a system.

In addition to this check, we log the event using our central logger, and update the buffered input value for use in the next run.

The last two cases in the if/else statement deal with the initial run, as well as a default handler. When we initially run this function the way we saw earlier, no interrupt will have been triggered, so obviously we have to add a third situation to the first two for the status and lock switches:

```cpp
    if (Club::clubIsClosed && !Club::clubOff) {
          Club::clubIsClosed = false;

          Club::log(LOG_INFO, string("ClubUpdater: Opening club."));

          Club::powerOn = true;
          try {
                if (!powerTimerStarted) {
                      timer->start(*cb);
                      powerTimerStarted = true;
                }
                else { 
                      timer->stop();
                      timer->start(*cb);
                }
          }
          catch (Poco::IllegalStateException &e) {
                Club::log(LOG_ERROR, "ClubUpdater: IllegalStateException on timer start: " + e.message());
                return;
          }
          catch (...) {
                Club::log(LOG_ERROR, "ClubUpdater: Unknown exception on timer start.");
                return;
          }

          powerTimerActive = true;

          Club::log(LOG_INFO, "ClubUpdater: Started power timer...");

          char msg = { '1' };
          Club::mqtt->sendMessage(Club::mqttTopic, &msg, 1);

          Club::log(LOG_DEBUG, "ClubUpdater: Sent MQTT message.");
    }
    else if (!Club::clubIsClosed && Club::clubOff) {
          Club::clubIsClosed = true;

          Club::log(LOG_INFO, string("ClubUpdater: Closing club."));

          Club::powerOn = false;

          try {
                if (!powerTimerStarted) {
                      timer->start(*cb);
                      powerTimerStarted = true;
                }
                else { 
                      timer->stop();
                      timer->start(*cb);
                }
          }
          catch (Poco::IllegalStateException &e) {
                Club::log(LOG_ERROR, "ClubUpdater: IllegalStateException on timer start: " + e.message());
                return;
          }
          catch (...) {
                Club::log(LOG_ERROR, "ClubUpdater: Unknown exception on timer start.");
                return;
          }

          powerTimerActive = true;

          Club::log(LOG_INFO, "ClubUpdater: Started power timer...");

          char msg = { '0' };
          Club::mqtt->sendMessage(Club::mqttTopic, &msg, 1);

          Club::log(LOG_DEBUG, "ClubUpdater: Sent MQTT message.");
    }
```

Next, we check whether we have to change the status of the club room from closed to open, or the other way around. This is determined by checking whether the club status (`Club::clubOff`) Boolean has changed relative to the `Club::clubIsClosed` Boolean, which stores the last known status.

Essentially, if the status switch is changed from on to off or the other way around, this will be detected and a change to the new status will be started. This means that a power timer will be started, which will turn the non-permanent power in the club room on or off after the preset delay.

The POCO `Timer` class requires that we first stop the timer before starting it if it has been started previously. This requires us to add one additional check.

In addition, we also use our reference to the MQTT client class to send a message to the MQTT broker with the updated club room status, here as either an ASCII 1 or 0\. This message can be used to trigger other systems, which could update an online status for the club room, or be put to even more creative uses.

Naturally, the exact payload of the message could be made configurable.

In the next section, we will update the colors on the status light, taking into account the state of power in the room. For this, we use the following table:

| **Color** | **Status switch** | **Lock switch** | **Power status** |
| Green | On | Unlocked | On |
| Yellow | Off | Unlocked | Off |
| Red | Off | Locked | Off |
| Yellow and red | On | Locked | On |

Here is the implementation:

```cpp

    if (Club::clubOff) {
          Club::log(LOG_INFO, string("ClubUpdater: New lights, clubstatus off."));

          mutex.lock();
          string state = (Club::powerOn) ? "on" : "off";
          if (powerTimerActive) {
                Club::log(LOG_DEBUG, string("ClubUpdater: Power timer active, inverting power state from: ") + state);
                regOut0 = !Club::powerOn;
          }
          else {
                Club::log(LOG_DEBUG, string("ClubUpdater: Power timer not active, using current power state: ") + state);
                regOut0 = Club::powerOn; 
          }

          if (Club::clubLocked) {
                Club::log(LOG_INFO, string("ClubUpdater: Red on."));
                regOut0 |= (1UL << RELAY_RED); 
          } 
          else {
                Club::log(LOG_INFO, string("ClubUpdater: Yellow on."));
                regOut0 |= (1UL << RELAY_YELLOW);
          } 

          Club::log(LOG_DEBUG, "ClubUpdater: Changing output register to: 0x" + NumberFormatter::formatHex(regOut0));

          writeRelayOutputs();
          mutex.unlock();
    }
```

We first check the state of the club room power, which tells us what value to use for the first bit of the output register. If the power timer is active, we have to invert the power state, as we want to write the current power state, not the future state that is stored in the power state Boolean.

If the club room's status switch is in the off position, then the state of the lock switch determines the final color. With the club room locked, we trigger the red relay, otherwise we trigger the yellow one. The latter would indicate the intermediate state, where the club room is off but not yet locked.

The use of a mutex here is to ensure that the writing of the I2C device's output register—as well as updating the local register variable—is done in a synchronized manner:

```cpp
    else { 
                Club::log(LOG_INFO, string("ClubUpdater: New lights, clubstatus on."));

                mutex.lock();
                string state = (Club::powerOn) ? "on" : "off";
                if (powerTimerActive) {
                      Club::log(LOG_DEBUG, string("ClubUpdater: Power timer active,    inverting power state from: ") + state);
                      regOut0 = !Club::powerOn; // Take the inverse of what the timer    callback will set.
                }
                else {
                      Club::log(LOG_DEBUG, string("ClubUpdater: Power timer not active,    using current power state: ") + state);
                      regOut0 = Club::powerOn; // Use the current power state value.
                }

                if (Club::clubLocked) {
                      Club::log(LOG_INFO, string("ClubUpdater: Yellow & Red on."));
                      regOut0 |= (1UL << RELAY_YELLOW);
                      regOut0 |= (1UL << RELAY_RED);
                }
                else {
                      Club::log(LOG_INFO, string("ClubUpdater: Green on."));
                      regOut0 |= (1UL << RELAY_GREEN);
                }

                Club::log(LOG_DEBUG, "ClubUpdater: Changing output register to: 0x" +    NumberFormatter::formatHex(regOut0));

                writeRelayOutputs();
                mutex.unlock();
          }
 }
```

If the club room's status switch is set to on, we get two other color options, with green being the usual one, which sees both the club room unlocked and the status switch enabled. If, however, the latter is on but the room is locked, we would get yellow and red.

After finishing the new contents of the output register, we always use the `writeRelayOutputs()` function to write our local version to the remote device, thus triggering the new relay state:

```cpp
void ClubUpdater::writeRelayOutputs() {
    wiringPiI2CWriteReg8(i2cHandle, REG_OUTPUT_PORT0, regOut0);

    Club::log(LOG_DEBUG, "ClubUpdater: Finished writing relay outputs with: 0x" 
                + NumberFormatter::formatHex(regOut0));
 }
```

This function is very simple, and uses WiringPi's I2C API to write a single 8-bit value to the connected device's output register. We also log the written value here:

```cpp
   void ClubUpdater::setPowerState(Timer &t) {
          Club::log(LOG_INFO, string("ClubUpdater: setPowerState called."));

          mutex.lock();
          if (Club::powerOn) { regOut0 |= (1UL << RELAY_POWER); }
          else { regOut0 &= ~(1UL << RELAY_POWER); }

          Club::log(LOG_DEBUG, "ClubUpdater: Writing relay with: 0x" +    NumberFormatter::formatHex(regOut0));

          writeRelayOutputs();

          powerTimerActive = false;
          mutex.unlock();
 }
```

In this function, we set the club room power state to whatever value its Boolean variable contains. We use the same mutex as we used when updating the club room status colors. However, we do not create the contents of the output register from scratch here, instead opting to toggle the first bit in its variable.

After toggling this bit, we write to the remote device as usual, which will cause the power in the club room to toggle state.

Next, we look at the static `Club` class, starting with the first function we call to initialize it:

```cpp
bool Club::start(bool relayactive, uint8_t relayaddress, string topic) {
          Club::log(LOG_INFO, "Club: starting up...");

          relayActive = relayactive;
          relayAddress = relayaddress;
          mqttTopic = topic;

          wiringPiSetup();

          Club::log(LOG_INFO,  "Club: Finished wiringPi setup.");

          pinMode(0, INPUT);
          pinMode(7, INPUT);
          pullUpDnControl(0, PUD_DOWN);
          pullUpDnControl(7, PUD_DOWN);
          clubLocked = digitalRead(0);
          clubOff = !digitalRead(7);

          previousLockValue = clubLocked;
          previousStatusValue = clubOff;

          Club::log(LOG_INFO, "Club: Finished configuring pins.");

          wiringPiISR(0, INT_EDGE_BOTH, &lockISRCallback);
          wiringPiISR(7, INT_EDGE_BOTH, &statusISRCallback);

          Club::log(LOG_INFO, "Club: Configured interrupts.");

          running = true;
          updateThread.start(updater);

          Club::log(LOG_INFO, "Club: Started update thread.");

          return true;
 }
```

With this function, we start the entire club monitoring system, as we saw earlier in the application entry point. It accepts a few parameters, allowing us to turn the relay functionality on or off, the relay's I2C address (if using a relay), and the MQTT topic on which to publish changes to the club room status.

After setting the values for member variables using those parameters, we initialize the WiringPi framework. There are a number of different initialization functions offered by WiringPi, which basically differ in how one can access the GPIO pins.

The `wiringPiSetup()` function we use here is generally the most convenient one to use, as it will use virtual pin numbers that map to the underlying Broadcom SoC pins. The main advantage of the WiringPi numbering is that it remains constant between different revisions of the Raspberry Pi SBCs.

With the use of either Broadcom (BCM) numbers or the physical position of the pins in the header on the SBC's circuit board, we risk that this changes between board revisions, but the WiringPi numbering scheme can compensate for this.

For our purposes, we use the following pins on the SBC:

|  | **Lock switch** | **Status switch** |
| BCM | 17 | 4 |
| Physical position | 11 | 7 |
| WiringPi | 0 | 7 |

After initializing the WiringPi library, we set the desired pin mode, making both of our pins into inputs. We then enable a pull-down on each of these pins. This enables a built-in pull-down resistor in the SoC, which will always try to pull the input signal low (referenced to ground). Whether or not one needs a pull-down or pull-up resistor enabled for an input (or output) pin depends on the circumstances, especially the connected circuit.

It's important to look at the behavior of the connected circuit; if the connected circuit has a tendency to "float" the value on the line, this would cause undesirable behavior on the input pin, with the value randomly changing. By pulling the line either low or high, we can be certain that what we read on the pin is not just noise.

With the mode set on each of our pins, we read out the values on them for the first time, which allows us to run the update function from the `ClubUpdater` class with the current values in a moment. Before we do that, however, we first register our interrupt methods for both pins.

An interrupt handler is little more than a callback that gets called whenever the specified event occurs on the specified pin. The WiringPi ISR function accepts the pin number, the type of event, and a reference to the handler function we wish to use. For the event type we picked here, we will have our interrupt handler triggered every time the value on the input pin goes from high to low, or the other way around. This means that it will be triggered when the connected switch goes from on to off, or off to on.

Finally, we started the update thread by using the `ClubUpdater` class instance and pushing it into its own thread:

```cpp
void Club::stop() {
          running = false;
 }
```

Calling this function will allow the loop in the `run()` function of `ClubUpdater` to end, which will terminate the thread it runs in, allowing the rest of the application to safely shut down as well:

```cpp
void Club::lockISRCallback() {
          clubLocked = digitalRead(0);
          lockChanged = true;

          clubChanged = true;
          clubCnd.signal();
 }

 void Club::statusISRCallback() {
          clubOff = !digitalRead(7);
          statusChanged = true;

          clubChanged = true;
          clubCnd.signal();
 }
```

Both of our interrupt handlers are pretty simple. When the OS receives the interrupt, it triggers the respective handler, which results in them reading the current value of the input pin, inverting the value as needed. The `statusChanged` or `lockChanged` variable is set to true to indicate to the update function which of the interrupts got triggered.

We do the same for the `clubChanged` Boolean variable before signaling the condition variable on which the `run` loop of `ClubUpdate` is waiting.

The last part of this class is the logging function:

```cpp
void Club::log(Log_level level, string msg) {
    logMutex.lock();
    switch (level) {
          case LOG_FATAL: {
                cerr << "FATAL:t" << msg << endl;
                string message = string("ClubStatus FATAL: ") + msg;
                if (mqtt) {
                      mqtt->sendMessage("/log/fatal", message);
                }

                break;
          }
          case LOG_ERROR: {
                cerr << "ERROR:t" << msg << endl;
                string message = string("ClubStatus ERROR: ") + msg;
                if (mqtt) {
                      mqtt->sendMessage("/log/error", message);
                }

                break;
          }
          case LOG_WARNING: {
                cerr << "WARNING:t" << msg << endl;
                string message = string("ClubStatus WARNING: ") + msg;
                if (mqtt) {
                      mqtt->sendMessage("/log/warning", message);
                }

                break;
          }
          case LOG_INFO: {
                cout << "INFO: t" << msg << endl;
                string message = string("ClubStatus INFO: ") + msg;
                if (mqtt) {
                      mqtt->sendMessage("/log/info", message);
                }

                break;
          }
          case LOG_DEBUG: {
                cout << "DEBUG:t" << msg << endl;
                string message = string("ClubStatus DEBUG: ") + msg;
                if (mqtt) {
                      mqtt->sendMessage("/log/debug", message);
                }

                break;
          }
          default:
                break;
    }

    logMutex.unlock();
 }
```

We use another mutex here to synchronize the log outputs in the system log (or console) and to prevent concurrent access to the MQTT class when different parts of the application call this function simultaneously. As we will see in a moment, this logging function is used in other classes as well.

With this logging function, we can log both locally (system log) and remotely using MQTT.

# HTTP request handler

Whenever POCO's HTTP server receives a new client connection, it uses a new instance of our `RequestHandlerFactory` class to get a handler for the specific request. Because it's such a simple class, it's fully implemented in the header:

```cpp
#include <Poco/Net/HTTPRequestHandlerFactory.h>
 #include <Poco/Net/HTTPServerRequest.h>

 using namespace Poco::Net;

 #include "statushandler.h"
 #include "datahandler.h"

 class RequestHandlerFactory: public HTTPRequestHandlerFactory { 
 public:
          RequestHandlerFactory() {}
          HTTPRequestHandler* createRequestHandler(const HTTPServerRequest& request) {
                if (request.getURI().compare(0, 12, "/clubstatus/") == 0) { 
                     return new StatusHandler(); 
               }
                else { return new DataHandler(); }
          }
 };
```

Our class doesn't do a whole lot more than compare the URL that the HTTP server was provided to determine which type of handler to instantiate and return. Here, we can see that if the URL string starts with `/clubstatus`, we return the status handler, which implements the REST API.

The default handler is a simple file server, which attempts to interpret the request as a filename, as we will see in a moment.

# Status handler

This handler implements a simple REST API, returning a JSON structure containing the current club status. This can be used by an external application to show real-time information on the system, which is useful for a dashboard or website.

Due to its simplicity, this class is also fully implemented in its header:

```cpp
#include <Poco/Net/HTTPRequestHandler.h>
 #include <Poco/Net/HTTPServerResponse.h>
 #include <Poco/Net/HTTPServerRequest.h>
 #include <Poco/URI.h>

 using namespace Poco;
 using namespace Poco::Net;

 #include "club.h"

 class StatusHandler: public HTTPRequestHandler { 
 public: 
          void handleRequest(HTTPServerRequest& request, HTTPServerResponse& response)  {         
                Club::log(LOG_INFO, "StatusHandler: Request from " +                                                     request.clientAddress().toString());

                URI uri(request.getURI());
                vector<string> parts;
                uri.getPathSegments(parts);

                response.setContentType("application/json");
                response.setChunkedTransferEncoding(true); 

                if (parts.size() == 1) {
                      ostream& ostr = response.send();
                      ostr << "{ "clubstatus": " << !Club::clubOff << ",";
                      ostr << ""lock": " << Club::clubLocked << ",";
                      ostr << ""power": " << Club::powerOn << "";
                      ostr << "}";
                }
                else {
                      response.setStatus(HTTPResponse::HTTP_BAD_REQUEST);
                      ostream& ostr = response.send();
                      ostr << "{ "error": "Invalid request." }";
                }
          }
 };
```

We use the central logger function from the `Club` class here to register details on incoming requests. Here, we just log the IP address of the client, but one could use the POCO `HTTPServerRequest` class's API to request even more detailed information.

Next, the URI is obtained from the request and we split the path section of the URL into a vector instance. After setting the content type and a transfer encoding setting on the response object, we check that we did indeed get the expected REST API call, at which point we compose the JSON string, obtain the club room status information from the `Club` class, and return this.

In the JSON object, we include information about the club room's status in general, inverting its Boolean variable, as well as the status of the lock and the power status, with a 1, indicating that the lock is closed or the power is on, respectively.

If the URL path had further segments, it would be an unrecognized API call, which would lead us to return an HTTP 400 (Bad Request) error instead.

# Data handler

The data handler is called whenever no REST API call is recognized by the request handler factory. It tries to find the specified file, read it from disk, and return it, along with the proper HTTP headers. This class is also implemented in its header:

```cpp
#include <Poco/Net/HTTPRequestHandler.h>
 #include <Poco/Net/HTTPServerResponse.h>
 #include <Poco/Net/HTTPServerRequest.h>
 #include <Poco/URI.h>
 #include <Poco/File.h>

 using namespace Poco::Net;
 using namespace Poco;

 class DataHandler: public HTTPRequestHandler { 
 public: 
    void handleRequest(HTTPServerRequest& request, HTTPServerResponse& response) {
          Club::log(LOG_INFO, "DataHandler: Request from " + request.clientAddress().toString());

          // Get the path and check for any endpoints to filter on.
          URI uri(request.getURI());
          string path = uri.getPath();

          string fileroot = "htdocs";
          if (path.empty() || path == "/") { path = "/index.html"; }

          File file(fileroot + path);

          Club::log(LOG_INFO, "DataHandler: Request for " + file.path());
```

We make the assumption here that any files to be served can be found in a subfolder of the folder in which this service is running. The filename (and path) is obtained from the request URL. If the path was empty, we assign it a default index file to be served instead:

```cpp
          if (!file.exists() || file.isDirectory()) {
                response.setStatus(HTTPResponse::HTTP_NOT_FOUND);
                ostream& ostr = response.send();
                ostr << "File Not Found.";
                return;
          }

          string::size_type idx = path.rfind('.');
          string ext = "";
          if (idx != std::string::npos) {
                ext = path.substr(idx + 1);
          }

          string mime = "text/plain";
          if (ext == "html") { mime = "text/html"; }
          if (ext == "css") { mime = "text/css"; }
          else if (ext == "js") { mime = "application/javascript"; }
          else if (ext == "zip") { mime = "application/zip"; }
          else if (ext == "json") { mime = "application/json"; }
          else if (ext == "png") { mime = "image/png"; }
          else if (ext == "jpeg" || ext == "jpg") { mime = "image/jpeg"; }
          else if (ext == "gif") { mime = "image/gif"; }
          else if (ext == "svg") { mime = "image/svg"; }
```

We first check that the resulting file path is valid and that it is a regular file, not a directory. If this check fails, we return an HTTP 404 File Not Found error.

After passing this check, we try to obtain the file extension from the file path to try and determine a specific MIME type for the file. If this fails, we use a default MIME type for plain text:

```cpp
                try {
                      response.sendFile(file.path(), mime);
                }
                catch (FileNotFoundException &e) {
                      Club::log(LOG_ERROR, "DataHandler: File not found exception    triggered...");
                      cerr << e.displayText() << endl;

                      response.setStatus(HTTPResponse::HTTP_NOT_FOUND);
                      ostream& ostr = response.send();
                      ostr << "File Not Found.";
                      return;
                }
                catch (OpenFileException &e) {
                      Club::log(LOG_ERROR, "DataHandler: Open file exception triggered: " +    e.displayText());

                      response.setStatus(HTTPResponse::HTTP_INTERNAL_SERVER_ERROR);
                      ostream& ostr = response.send();
                      ostr << "Internal Server Error. Couldn't open file.";
                      return;
                }
          }
 };
```

As the final step, we use the response object's `sendFile()` method to send the file to the client, along with the MIME type we determined earlier.

We also handle the two exceptions this method can throw. The first one occurs when the file cannot be found for some reason. This results in us returning another HTTP 404 error.

If the file cannot be opened for some reason, we return an HTTP 500 Internal Server Error instead, along with the text from the exception.

# Service configuration

With the Raspbian Linux distribution for Raspberry Pi SBCs, system services are usually managed with `systemd`. This uses a simple configuration file, with our club monitoring service using something like the following:

```cpp
[Unit] 
Description=ClubStatus monitoring & control 

[Service] 
ExecStart=/home/user/clubstatus/clubstatus /home/user/clubstatus/config.ini 
User=user 
WorkingDirectory=/home/user/clubstatus 
Restart=always 
RestartSec=5 

[Install] 
WantedBy=multi-user.target 
```

This service configuration specifies the name of the service, with the service being started from the "`user`" user account's folder, and the configuration file for the service being found in the same folder. We set the working directory for the service, also enabling the automatic restarting of the service after five seconds if it were to fail for whatever reason.

Finally, the service will be started after the system has started to the point where a user can log in to the system. This way, we are sure that networking and other functionality has been started already. If one starts a system service too soon, it could fail due to missing functionality on account of things not having been initialized yet.

Next, here is the INI file configuration file:

```cpp
[MQTT]
 ; URL and port of the MQTT server.
 host = localhost
 port = 1883

 ; Authentication
 user = user
 pass = password

 ; The topic status on which changes will be published.
 clubStatusTopic = /my/topic

 [HTTP]
 port = 8080

 [Relay]
 ; Whether an i2c relay board is connected. 0 (false) or 1 (true).
 active = 0
 ; i2c address, in decimal or hexadecimal.
 address = 0x20
```

The configuration file is divided into three sections, MQTT, HTTP, and Relay, with each section containing the relevant variables.

For MQTT, we have the expected options for connecting to the MQTT broker, including password-based authentication. We also specify the topic regarding which club status updates will be published here.

The HTTP section just contains the port we will be listening on, with the server listening on all interfaces by default. If necessary, one could make the network interface a used configurable as well by making this property configurable before starting the HTTP server.

Finally, the Relay section allows us to turn the relay board feature on or off, as well as configure the I2C device address if we are making use of this feature.

# Permissions

Since both the GPIO and I2C are treated as common Linux devices, they come with their own set of permissions. Assuming one wishes to avoid running the service as root, we need to add an account that runs the service to both the `gpio` and `i2c` user groups:

```cpp
    sudo usermod -a -G gpio user
    sudo usermod -a -G i2c user
```

After this, we need to restart the system (or log out and in again) for the changes to take effect. We should now be able to run the service without any issues.

# Final results

With the application and `systemd` service configured and installed on the target SBC, it will automatically start and configure itself. To complete the system, you could install it along with a suitable power supply into an enclosure, into which you would run the signal wires from the switches, the network cable, and so on.

One implementation of this system was installed at the Entropia hackerspace in Karlsruhe, Germany. This setup uses a real traffic light (legally obtained) outside the club door with 12 volt LED lights for status indication. The SBC, relay board, debounce board, and power supply (5V and 12V MeanWell industrial PSU) are all integrated into a single, laser-cut wooden enclosure:

![](img/fdcc5ed3-6f1c-4c43-a51e-d14a36225368.png)

However, you are free to integrate the components any way you wish. The main thing to consider here is that the electronics are all safely protected from harm and accidental contact as the relay board could be switching mains voltage, along with possibly the mains voltage line for the power supply.

# Example – basic media player

Another basic example of an SBC-based embedded system is a media player. This can involve both audio and audio-visual (AV) media formats. The difference between an SBC-based system being used to play back media with regular keyboard and mouse input, and an embedded SBC-based media player, is that in the latter's case the system can only ever be used for that purpose, with the software and user interface (physical- and software-wise) both optimized for media player use.

To this end, a software-based frontend has to be developed, along with a physical interface peripheral, using which the media player can be controlled. This could be something as simple as a series of switches connected to the GPIO pins, with a regular HDMI display for output. Alternatively, one could use a touch screen, although this would require a more complex driver setup.

Since our media player system stores media files locally, we want to use an SBC that supports external storage beyond the SD card. Some SBCs come with a SATA connection, allowing us to connect a hard disk drive (HDD) of capacities far exceeding those of SD cards. Even if we stick to compact 2.5" HDDs, which are roughly the same size as many popular SBCs, we can easily and fairly cheaply get multiple terabytes worth of storage.

Beyond the storage requirement, we also need to have a digital video output, and we want to either use the GPIO or the USB side for the user interface buttons.

A very suitable board for this purpose is the LeMaker Banana Pro, which comes with the H3 ARM SoC, hardware SATA, and Gigabit Ethernet support, as well as a full-sized HDMI output with 4k video decoding support:

![](img/74a1aea9-04a3-4e25-9ac3-f4bc3020306d.png)

After going through the basics of installing Armbian or similar OSes on the SBC, we can set up a media player application on the system, having it start together with the OS and configuring it to both load a playlist and to listen to events on a number of GPIO pins. These GPIO pins would be connected to a number of control switches, allowing us to scroll through the playlist and start, pause, and stop playlist items.

Other interaction methods are possible, such as an infrared or radio-based remote control, each of which come with their own advantages and disadvantages.

We will be working through the creation of this media player system and turning it into an infotainment system in the following chapters:

*   [Chapter 6](7d5d654f-a027-4825-ab9e-92c369b576a8.xhtml), *Testing OS-Based Applications*
*   [Chapter 8](886aecf2-8926-4aec-8045-a07ae2cdde84.xhtml), *Example - Linux-Based Infotainment System*
*   [Chapter 11](47e0b6fb-cb68-43c3-9453-2dc7575b1a46.xhtml), *Developing Embedded Systems with Qt*

# Summary

In this chapter, we looked at OS-based embedded systems, exploring the many operating systems available to us, with the most significant differences, especially those of real-time operating systems. We also saw how one would integrate an RTC peripheral into an SBC-based Linux system and explored user space- and kernel space-based driver modules, along with their advantages and disadvantages.

Along with the example project in this chapter, the reader should now have a good idea of how to translate a set of requirements into a functioning OS-based embedded system. The reader will know how to add external peripherals and use them from the OS.

In the next chapter, we will be looking at developing for resource-restricted embedded systems, including 8-bit MCUs and their larger brethren.