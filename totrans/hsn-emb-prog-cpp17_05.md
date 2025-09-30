# Resource-Restricted Embedded Systems

Using a smaller embedded system such as a microcontroller (MCU) means having small amounts of RAM, CPU power, and storage. This chapter deals with planning for and making efficient use of limited resources taking into account the wide range of currently available MCUs and **System-on-Chip** (**SoC**) solutions. We will be considering the following aspects

*   Selecting the right MCU for a project
*   Concurrency and memory management
*   Adding sensors, actuators, and network access
*   Bare-metal development versus real-time OSes

# The big picture for small systems

When first confronted with a new project that requires the use of at least one type of MCU, it can seem like an overwhelming task to. As we saw in [Chapter 1](0ff8cac9-3155-45e1-af05-7005fc419dd6.xhtml), *What are Embedded Systems?*, there is a large number of MCUs to choose from, even if we limit ourselves to just those that have been released recently.

It may seem obvious to start by asking how many bits one needs, as in selecting between 8-bit, 16-bit, and 32-bit MCUs, or something as easy to quantify as clock speed, but these metrics are sometimes misleading and often don't lend themselves well to narrowing down the product selection. As it turns out, the parent categories are availability of sufficient I/O and the integrated peripherals to make the hardware happen in a lean and reliable way, as well as processing power tailored to the requirements faced at design-time and predicted to emerge throughout the product life-time.

So in more detail we need to answer questions like these:

*   **Peripherals**: Which peripherals are needed to interact with the rest of the system?
*   **CPU**: What level of CPU power is needed to run the application code?
*   **Floating point**: Do we need hardware floating point support?
*   **ROM**: How much ROM do we need to store the code?
*   **RAM**: How much RAM is required to run the code?
*   **Power and thermals**: What are the electrical power and thermal limitations?

Each MCU family has its own strengths and weaknesses, though one of the most important factors to pick one MCU family over another the quality of its development tools. For hobby and other noncommercial projects, one would primarily consider the strength of the community and the available free development tools, while in the context of commercial projects one would also look at the support one could expect from the MCU manufacturer and possible third parties.

A key aspect of embedded development is in-system programming and debugging. Since programming and debugging are intertwined, we'll be looking at the corresponding interface options later to be able to identify what satisfies our requirements and constraints.

A popular and powerful debugging interface has become synonymous to the underlying Joint Test Action Group (JTAG) IEEE standard 1149.1 and easily recognized by signals frequently labeled TDI, TDO, TCK, TMS and TRST, defining the aptly-named Test Action Port (TAP). The larger standard has since been expanded up to 1149.8 and not all versions apply to digital logic, so we'll limit our scope to 1149.1 and a reduced pin count version described under 1149.7\. For now we just require that at least one of the full-featured JTAG, SWD and UPDI interfaces be supported.

Debugging MCU-based systems along with on-chip debugging, using both command-line tools and IDEs, is something that we will take an in-depth look at in [Chapter 7](d8237285-fcb7-4bbc-84f3-e45568598865.xhtml), *Testing Resource-Limited Platforms*.

Finally, if we are going to be making products containing the chosen MCU for an active production phase of a few years, it's vital that we ensure the MCU availability (or that of compatible replacements) for at least that period. Reputable manufacturers provide product life cycle information as part of their supply chain management, with discontinuation notices being sent 1 to 2 years in advance, and recommendations for lifetime buys.

For many applications, it is hard to ignore the wide availability of cheap, powerful, and easy-to-use Arduino compatible boards, especially the popular ones designed around the AVR family of MCUs. Among these, the ATmega MCUs—the mega168/328, and in particular the mega1280/2560 variants—provide significant amounts of processing power, ROM, and RAM for both high-level functionality and the handling of data for input, control, and telemetry, as well as a differentiated but rich sets of peripherals and GPIO.

All of these aspects make prototyping exceedingly simple before even committing to a more definitive variant with lower specifications and (hopefully) better BOM cost. As an example, the ATmega2560 "MEGA" board is shown as follows, and we will look at other boards in more detail later in this chapter as we work through a number of examples on how to develop for the AVR platform.

![](img/bf763d40-f2e9-4f8c-88ca-e25e963aa6c8.png)

Generally, one would pick a number of MCUs that might work for the project, get the development boards, hook them up to the rest of the projected system components (often on their own development or breakout boards), and start developing the software for the MCU that will make everything work together.

As more and more parts of the system become finalized, the number of development boards and bread-boarded components will dwindle until one reaches the point where one starts working on the final **printed circuit board** (**PCB**) layout. This will go through a number of iterations as well, as issues get ironed out, last-minute features are added, and the system as a whole is tested and optimized.

MCUs in such systems work on a physical level with the hardware, thus it is often a requirement to specify both hardware and software in tandem, if only because the software is so reliant on the hardware functionality. A common theme encountered in the industry is hardware modularity, either as small add-on PCBs with minimum added complexity, adding sensor or communication interfaces to devices such as temperature controllers and variable-frequency drives, or as full-fledged DIN rail modules connected to a common serial bus.

# Example – Machine controller for a laser cutter

One of the fastest and most accurate ways to cut a wide range of materials is using a high-power laser. With the price of carbon dioxide (CO[2]) having dropped sharply over the years, this has led to widespread use of affordable (cheap) laser cutters as shown in the following image:

![](img/1db0afad-701b-4921-aa11-cebb4a9fb4e3.png)

While it's perfectly possible to operate a laser cutter with nothing more than just a basic enclosure and the stepper motion control board that move the head across the machin bed, from a usability and safety point of view, this is not desirable. Still, many of the cheap laser cutters one can purchase online, however, do not come with any safety or usability features whatsoever.

# Functional specification

To complete the product, we need to add a control system that uses sensors and actuators to monitor and control the state of the machine, ensuring that it is always in a safe state and shutting down the laser beam if necessary. This means protecting access to each of the following three sections:

![](img/897f1ff6-32a8-4358-b81d-d2bdd82b8d69.png)

The cutting beam is usually generated by a CO[2] laser, a type of gas laser that was invented in 1964\. The application of a high voltage causes current flow and thereby excitement of the gas molecules in the bore that make up the gain medium, ultimately resulting in the formation of a coherent beam of **long-wavelength infrared** (**LWIR**) or IR-C, light at a wavelength of 9.4 or 10.6 µm.

One characteristic of LWIR is that it is strongly absorbed by a large number of materials, so that it can be used for engraving, cutting, and even surgery on tissues as the water in biological tissues efficiently absorbs the laser beam. This also makes it obvious why even brief exposure of one's skin to a CO[2] laser's beam is extremely dangerous.

To achieve safe operation, exposure to laser light must be inhibited by locking the enclosure during normal operation, deactivating the laser power supply, and closing a beam shutter or preferably a combination of these measures when any of the interlocks is opened or any other safety condition is no longer satisfied.

For example, temperature limits have to be upheld: most CO[2] lasers comprise of water-cooled gas discharge tube, which can quickly crack or bend in case of a cooling fault. What's more, the cutting process creates irritating or toxic fumes that need to be continuously removed from the enclosure so as not to contaminate the optics and exit into the environment when the lid is opened.

These requirements necessitate that we monitor cooling water flow and temperature, air flow for the exhaust, and the air flow resistance (pressure drop over mass flow) of over the exhaust filter.

Finally, we also want to make it convenient to use the laser cutter and avoid having to "bring your own device" to process the design in a machine-specific way, then convert it and upload it to the stepper motion controller board via USB. Instead, we want to load the design project from an SD card or USB stick and use a simple LCD and buttons to set options.

# The design requirements

With the earlier requirements in mind, we can formulate a list of features needed for the control system:

*   Operator safety:
    *   Interlock switches on access panels (closed with the panel closed)
    *   Locking mechanism (mechanically locking access panel; redundant)
    *   Emergency stop

*   Laser cooling:
    *   Pump relay
    *   Temperature sensor in water tank (cooling capacity, inlet temperature)
    *   Temperature sensor on valve cooling exhaust (mantle temperature)
    *   Flow sensor (water flow speed; redundant)
*   Air exhaust:
    *   Fan relay
    *   Air filter status (differential pressure sensor)
    *   Fan speed (RPM)
*   Laser module:
    *   Laser power relay
    *   Beam shutter (redundant)
*   User interface
    *   Alert indicators for:
        *   Panel interlock
        *   Air filter condition
        *   Fan status
        *   Pump status
        *   Water temperature
    *   Indicator LEDs for:
        *   Standby
        *   Starting
        *   Operation
        *   Emergency stop
        *   Cool down
*   Communication:
    *   USB communication with stepper board (UART)
    *   Motion control: generate stepper motor instructions
    *   Read files from SD card/USB stick
    *   Accept files over Ethernet/Wi-Fi
    *   NFC reader to identify users

# Implementation-related choices

As pointed out at the beginning of this chapter, mid-range MCUs are currently capable of providing the resources to satisfy most, if not all of our design requirements. So one of the tough questions is what we'll be spending our money on: hardware components or software development? Imponderabilities aside, we'll now take a closer look at three candidate solutions:

*   A single mid-range AVR MCU board (ATmega2560)
*   A higher-end Cortex-M3 MCU board (SAM3X8E)
*   A tandem of mid-range MCU board and an SBC with OS

We're pretty close to meeting hte design requirements with just an Arduino Mega (ATmega2560), as the first five sections require little in terms of CPU speed, just a number of digital input and output pins and a few analog ones depending on the exact sensors we'll be using or at most a peripheral interface to make use of (for example, for MEMS pressure sensors).

The challenge starts with motion control feature under communications in the previous feature list, where we suddenly have to convert a **vector graphics file** (**.svg**) to a series of stepper commands. This is a compound problem of data transfer, file parsing, path generation, and what is known in the robotic world as inverse kinematics. USB communications can also be problematic for our 8-bit MCU, mostly because of peak processor loads coinciding with timeouts for USB endpoint communication or UART RX buffer register handling.

The key is knowing when to change gears. Motion control is time critical as it's tied to the inertia of the physical world. Additionally, we're constrained by the processing and bandwidth resources of our controller to make control and data transfers, buffering, and ultimately the processing and output generation itself happen. As a general pattern, more capable internal or external peripherals can relax timing requirements by handling events and memory transactions themselves, reducing context switching and processing overhead. Here's an incomplete list of such considerations:

*   Simple UART requires collecting every byte upon RX Complete (RXC). Failure to do so results in data loss, as indicated by the DOR flag. A few controllers such as ATmega8u2 through ATmega32u4 provide native hardware flow control via RTS/CTS lines, which can prevent USB-UART converters such as PL2303 and FT232 from sending, forcing them to do the buffering instead until UDR is conveniently emptied again.

*   Dedicated USB host peripherals such as the MAX3421 are connected via SPI and effectively remove USB timing requirements for mass storage integration.
*   UART aside, network communication peripherals are inherently buffered in software due to the complexity of the layer stack. For Ethernet, the W5500 is an attractive solution.
*   It sometimes makes sense to add another smaller MCU that independently handles I/O and pattern generation while implementing an interface of our choice – e.g. serial or parallel. This is already the case with some Arduino boards featuring an ATmega16u2 for USB serial conversion.

The NFC reader feature requirement calls for **Near-Field Communication** (**NFC**, a subset of RFID) to prevent unauthorized use of the laser cutter, which would add the biggest burden of all. Not due to the communicating with the NFC reader itself, but due to the increase in code size and CPU requirements to handle cryptography with certificates depending on the security level chosen. We would also need a secure place to store the certificates which usually bumps up MCU specs.

Now we are at the point where we consider the more advanced options. The simpler ATmega2560 remains a great fit with its large amount of GPIO and can read SD cards over SPI along with communicating with an external integrated ethernet chip. However, the computationally or memory intensive tasks in motion control and NFC reader feature list would likely overburden the MCU or lead to convoluted "optimized" solutions with inferior maintainability if one were to try.

Upgrading the MCU to an ARM Cortex-M3 such as found on the Arduino Due development board, would likely resolve all those bottlenecks. It would preserve the large number of GPIO we got accustomed to on the ATmega2560, while increasing CPU performance significantly. The stepper drive patterns can be generated on the MCU, which also presents with native USB support, along with other advanced peripherals (USART, SPI and I2C and HSMCI, which also have DMA).

A basic NFC tag reader could be connected via a UART, SPI, or I2C, and this design choice would lead to a system as shown:

![](img/a97fac06-4f02-4255-b48c-c2ba8fb95b60.png)

The third embodiment involving an SBC would again make use of the ATmega2560 and add a low-powered SBC running an OS. This SBC would handle any CPU-intensive tasks, Ethernet and Wi-Fi connectivity, USB (host) tasks, and so on. It would communicate with the ATmega side via a UART, possibly adding a digital isolator or level shifter in between the boards to accommodate the 3.3V (SBC) and 5V TTL (Atmega) logic levels.

Choosing the SBC + MCU solution would substantially change the software challenges but only slightly reorganize our system on the hardware side. This would look as follows:

![](img/07063017-8a6b-4a0e-98ce-25e7f373a6fa.png)

As with most development processes, there are only a few absolute answers, and many solutions pass functional requirements as *good enough* after trade-offs between power usage, complexity, and maintenance requirements affecting the final design choice.

In this particular example, one could choose either the higher-end single or dual-board solution, and it would most likely entail the same amount of effort to satisfy the requirements. One of the main differences would be that the OS-based solution adds the need to perform frequent OS updates, on account of it being a network-connected system running a full-blown OS whereas embedded ethernet controllers with offloaded hardwired TCP/IP stack and memories tend to be more robust and proven.

The Cortex-M3-based option (or the even faster Cortex-M4) would feature just our own code, and thus would be unlikely to have any common security issues that could be easily targeted. We wouldn't be off the hook for maintenance, but our code would be small enough to validate and read through in its entirety, with the only letdown that the Arduino Due design fails to break out the pins for RMII to hook up an external Ethernet PHY, discouraging the use of its internal Ethernet MAC.

Running down the checklist we put together at the beginning of this chapter, but this time with the ATmega2560 + SBC and application in mind, gives us the following distribution of duties:

*   **Peripherals**: The MCU side will mostly need GPIO, some analog (ADC) inputs, Ethernet, USB, along with SPI and/or I2C.
*   **CPU**: The required MCU performance is time-critical but minor, except for when we need to do the processing of the vector path elements into stepper instructions. The SBC side can be sophisticated as long as enough commands can be queued for MCU-side execution and time-critical interaction is avoided.
*   **Floating point**: The stepper instruction conversion algorithm on an MCU executes substantially faster if we have hardware floating point support. The length and time scales involved may make fixed point arithmetic feasible, relaxing this requirement.
*   **ROM**: The entire MCU code will likely fit into a few kilobytes since it's not very complex. The SBC code will be larger by orders of magnitude just by invoking high-level libraries to provide the desired functionality but this will be more than offset by the similarly scaled mass storage and processing capabilities.
*   **RAM**: A few KB of SRAM on the MCU should suffice. The stepper instruction conversion algorithm may require modifications to fit into the SRAM limitations with its buffering and processing data requirements. In a worst-case scenario, buffers can be downsized.
*   **Power and thermals**: In the light of the laser cutter system's power needs and cooling system, we have got no significant power or thermal limitations. The section containing the control system also houses the main power supply and is already equipped with an appropriately sized cooling fan.

It's important to note at this point that although we realized the complexity and requirements of the task at hand sufficiently to draw conclusions leading us to a selection of hardware components, the aspects of how to achieve them in detail are still left to the software developer.

For example, we could define our own data structures and formats and implement the machine-specific path generation and motion control ourselves, or adopt a (RS-274) G-code intermediate format which has been well-established in numerical control applications for several decades, and that lends itself well to generating motion control commands. G-code and has also found widespread acceptance in the diy hardware community, expecially for FDM 3D printing.

One noteworthy mature open source implementation of G-code based motion control is GRBL, introduced as:

Grbl is a free, open source, high performance software for controlling the motion of machines that move, that make things, or that make things move, and will run on a straight Arduino. If the maker movement was an industry, Grbl would be the industry standard.
--https://github.com/gnea/grbl

Most likely we'll have to add halt and emergency stop features for different violations of our safety checks. While temperature excursions or a clogged filter would preferably just halt the laser cutter and permit resuming the job with the issues resolved, an interlock tripped by opening the enclosure must result in immediate shutdown of the laser, even without finishing the last command for a path segment and motion.

The choice to modularize the motion control task and produce G-code for it has benefits beyond the availability of proven implementations, allowing us to easily add usability features like manual control for setup and calibration as well as testability using previously generated, human-readable codes on the machine side just as inspection on the output of our file interpretation and path generation algorithms.

With the list of requirements, the initial design completed, and a deepened understanding of how we are going to achieve our goals, the next step would be to obtain a development board (or boards) with the chosen MCU and/or SoC, along with any peripherals so that one can get started on developing the firmware and integrating the system.

While the full implementation of the machine control system as described in this example is beyond the scope of this book, an in-depth understanding of the development for both microcontroller and SBC target varieties will be strived for in the remainder of this chapter and [Chapter 6](7d5d654f-a027-4825-ab9e-92c369b576a8.xhtml), *Testing OS-Based Applications*, [Chapter 8](4416b2de-d86a-4001-863d-b167635a0e10.xhtml), *Example - Linux-Based Infotainment* System, and [Chapter 11](c90e29ad-2e13-4838-a9c2-885209717513.xhtml), *Developing for Hybrid SoC/FPGA Systems*, respectively.

# Embedded IDEs and frameworks

While the application development for SoCs tends to be quite similar to desktop and server environments, as we saw in the previous chapter, MCU development requires a far more intimate knowledge of the hardware that one is developing for, sometimes down to the exact bits to set in a particular register.

There exist some frameworks that seek to abstract away such details for particular MCU series, so that one can develop for a common API without having to worry about how it is implemented on a specific MCU. Of these, the Arduino framework is the most well-known outside of industrial applications, though there are also a number of commercial frameworks that are certified for production use.

Frameworks such as the **Advanced Software Framework** (**ASF**) for AVR and SAM MCUs can be used with a variety of IDEs, including Atmel Studio, Keil µVision, and IAR Embedded Workbench.

A non-exhaustive list of popular embedded IDEs follows:

| **Name** | **Company** | **License** | **Platforms** | **Notes** |
| Atmel Studio | Microchip | Proprietary | AVR, SAM (ARM Cortex-M). | Originally developed by Atmel before being bought by Microchip. |
| µVision | Keil (ARM) | Proprietary | ARM Cortex-M, 166, 8051, 251. | Part of the **Microcontroller Development Kit** (**MDK**) toolchain. |
| Embedded Workbench | IAR | Proprietary | ARM Cortex-M, 8051, MSP430, AVR, Coldfire, STM8, H8, SuperH, etc. | Separate IDE for each MCU architecture. |
| MPLAB X | Microchip | Proprietary | PIC, AVR. | Uses the Java-based NetBeans IDE as foundation. |
| Arduino | Arduino | GPLv2 | Some AVR and SAM MCUs (extendable). | Java-based IDE. Only supports its own C dialect language. |

The main goal of an IDE is to integrate the entire workflow into a single application, from writing the initial code to programming the MCU memory with the compiled code and debugging the application while it runs on the platform.

Whether to use a full IDE is a matter of preference, however. All of the essential features are still there when using a basic editor and the tools from the command line, although frameworks such as the ASF are written to deeply integrate with IDEs.

One of the main advantages of the popular Arduino framework is that it has more or less standardized an API for various MCU peripherals and other functionality that is supported across an ever-growing number of MCU architectures. Coupled with the open source nature of the framework, it makes for an attractive target for a new project. This is particularly attractive when it comes to prototyping, courtesy of a large number of libraries and drivers written for this API.

Unfortunately, the Arduino IDE is unfortunately focused purely on a stripped-down dialect of the C programming language, despite its core libraries making widespread use of C++. Still this enables us to integrate just the libraries into our own embedded C++ projects, as we will see later in this chapter.

# Programming MCUs

After we have compiled our code for the target MCU, the binary image needs to be written to a controller memory prior to execution and debugging. In this section we will look at the varied ways in which this can be accomplished. These days only factory-side programming is done with test sockets, or better yet at the wafer level before a known good die is bonded to a leadframe and encapsulated. Surface-mount parts already rule out easy removal of an MCU for (repeated) programming.

A number of (frequently vendor-specific) options for in-circuit programming exist, distinguished by the peripherals they use and the memories they affect.

So a pristine MCU often needs to be programmed using an external programming adapter. These generally work by setting the pins of the MCU so that it enters programming mode, after which the MCU accepts the data stream containing the new ROM image.

Another option that is commonly used is to add a boot loader to the first section of the ROM, which allows the MCU to program itself. This works by having the boot loader check on startup whether it should switch to programming mode or continue loading the actual program, placed right after the boot loader section.

# Memory programming and device debugging

External programming adapters often utilize dedicated interfaces and associated protocols which permit programming and debugging of the target device. Protocols with which one can program an MCU include the following:

| **Name** | **Pins** | **Features** | **Description** |
| **SPI (ISP)** | **4** | program | **Serial Peripheral Interface** (**SPI**), used with older AVR MCUs to access its Serial Programmer mode (**In-circuit Serial Programming** (**ISP**)). |
| **JTAG** | **5** | program debug
boundary | Dedicated, industry-standard on-chip interface for programming and debugging support. Supported on AVR ATxmega devices. |
| **UPDI** | **1** | program debug | The **Unified Programming and Debug Interface** (**UDPI**) used with newer AVR MCUs, including ATtiny devices. It's a single-wire interface that's the successor to the two-wire PDI found on ATxmega devices. |
| **HVPP/****HVSP** | **17/****5** | program | High Voltage Parallel Programming / High Voltage Serial Programming. AVR programming mode using 12V on the reset pin and direct access to 8+ pins. Ignores any internal fuse setting or other configuration option. Mostly used for in-factory programming and for recovery. |
| **TPI** | **3** | program | Tiny Programming Interface, used with some ATtiny AVR devices. These devices also lack the number of pins for HVPP or HVSP. |
| **SWD** | **3** | program debug
boundary | Serial Wire Debug. Similar to reduced pin count JTAG with two lines, but uses ARM Debug Interface features, allowing a connected debugger to become a bus master with access to the MCU's memory and peripherals. |

ARM MCUs generally provide JTAG as their primary means of programming and debugging. On 8-bit MCUs, JTAG is far less common, which is mostly due to the complexity of its requirements.

AVR MCUs tend to offer In-System Programming (ISP) via SPI in addition to high voltage programming modes. Entering programming mode requires that the reset pin be held low during programming and verification and released and strobed at the end of the programming cycle.

One requirement for ISP is that the relevant (SPIEN fuse bit) in the MCU is set to enable the in-system programming interface. Without this bit set, the device won’t respond on the SPI lines. Without JTAG available and enabled via the JTAGEN fuse bit, only HVPP or HVSP are available to recover and reprogram the chip. In the latter case, the unusual set of pins and the 12V supply voltage do not necessarily integrate well into the board circuitry.

The physical connections required for most serial programming interfaces are fairly simple, even when the MCU has already been integrated into a circuit as shown in the following diagram:

![](img/b03d2974-c53c-4473-aec3-c599459b49e2.png)

Here, the external oscillator is optional if an internal one exists. The **PDI**, **PDO**, and **SCK** lines correspond to their respective SPI lines. The Reset line is held active (low) during programming. After connecting to the MCU in this manner, we are free to write to its flash memory, EEPROM, and configuration fuses.

On newer AVR devices, we find the **Unified Programming and Debug Interface** (**UPDI**), which uses just a single wire (in addition to the power and ground lines) to connect to the target MCU to provide both programming and debug support.

This interface simplifies the previous connection diagram to the following:

![](img/b0bfa79c-4d7e-4d8e-bec0-5e232d4c017f.png)

This favorably compares to JTAG (IEEE 1149.1) on the ATxmega (when enabled) as follows:

![](img/bd797ef1-480e-4ff2-91b1-9b42f5bd02d6.png)

Thereduced pin count JTAG standard (IEEE 1149) implemented on the ATxmega requires only one clock TCKC, one data wire TMSC and is aptly called Compact JTAG. Of these interfaces, UPDI still requires the fewest connections with the target device. Apart from that, both support similar features for AVR MCUs.

For other systems using JTAG for programming and debugging, no standard connection exists. Each manufacturer uses their own preferred connector, ranging from 2 x 5 pins (Altera, AVR) to 2 x 10 pins (ARM), or a single 8-pin connector (Lattice).

With JTAG being more a protocol standard rather than a physical specification, one should consult the documentation for one's target platform for the specific details.

# Boot loader

The boot loader has been introduced as a small extra application that uses an existing interface (for example, UART or Ethernet) to provide self-programming capabilities. On the AVR, a boot loader section of 256 bytes to 4 KB can be reserved in its flash. This code can perform any number of user-defined tasks, from setting up a serial link with a remote system, to booting from a remote image over Ethernet using PXE.

At its core, an AVR boot loader is no different from any other AVR application, except that when compiling it one extra linker flag is added to set the starting byte address for the boot loader:

```cpp
--section-start=.text=0x1800 
```

Replace this address with a similar one for the specific MCU that you're using (for AVR depending on the BOOTSZ flags set and controller used, see datasheet table about Boot Size Configuration: Boot Reset Address, where, for example, the boot reset address is 0xC00 is in words and the section start is defined in bytes). This ensures that the boot loader code will be written to the proper location in the MCU's ROM. Writing the boot loader code to the ROM is almost always done via ISP.

AVR MCUs divide the flash ROM into two sections: the **no-read-while-write **(**NRWW**) (for most, if not all application memory space) and **read-while-write** (**RWW**) sections. In brief, this means that the RWW section can be safely erased and rewritten without affecting the CPU's operation. This is why the boot loader resides in the NRWW section and also why it's not easy to have the boot loader update itself.

Another important detail is that the boot loader can also not update the fuses that set various flags in the MCU. To change these, one has to externally program the device.

After programming the MCU with a boot loader, one would generally set the flags in the MCU that let the processor know that a boot loader has been installed. In the case of AVR, these flags are BOOTSZ and BOOTRST.

# Memory management

The storage and memory system of microcontrollers consists out of multiple components. There is a section of **read-only-memory** (**ROM**) that is only written to once when the chip is programmed, but which cannot normally be altered by the MCU itself, as we saw in the previous section.

The MCU may also have a bit of persistent storage, in the form of EEPROM or equivalent. Finally, there are CPU registers and the **random-access memory** (**RAM**). This results in the following exemplary memory layout:

![](img/eefb6e49-e5ff-4360-b08c-a862fb6e0530.png)

The use of a modified Harvard architecture (split program and data memory at some architectural level, generally with the data buses) is common with MCUs. With the AVR architecture, for example, the program memory is found in the ROM, which for the ATmega2560 is connected using its own bus with the CPU core, as one can seen on the block diagram for this MCU, which we looked at previously in [Chapter 1](0ff8cac9-3155-45e1-af05-7005fc419dd6.xhtml), *What Are Embedded Systems?*

A major advantage of having separate buses for these memory spaces is that one can address each of them separately, which makes better use of the limited addressing space available to an 8-bit processor (1 and 2 byte wide address). It also allows for concurrent accesses while the CPU is busy with the other memory space, further optimizing the available resources.

For the data memory in the SRAM, we are then free to use it as we want. Here, we do need at least a stack to be able to run a program. Depending on how much SRAM is left in the MCU, we can then also add a heap. Applications of moderate complexity can be realized with only stack and statically allocated memory though, not involving higher-level language features that produce code with heap allocations.

# Stack and heap

Whether one needs to initialize the stack on the MCU that one is programming for depends on how low-level one wishes to go. When using the C-runtime (on AVR: `avr-libc`), the runtime will handle initializing the stack and other details by letting the linker place naked code into init sections, for example specified by:

```cpp
__attribute__ ((naked, used, section (".init3")))
```

Preceding the execution of any of our own application code.

The standard RAM layout on AVR is to start with the `.data` variables at the beginning of the RAM, followed by `.bss`. The stack is started from the opposite site of the RAM, growing towards the beginning. There will be room left between the end of the `.bss` section and the end of the stack illustrated as follows:

![](img/c84909aa-7715-46f6-9538-5760568e9748.png)

Since the stack grows depending on the depth of the function calls in the application being run, it is hard to say how much space is available. Some MCUs allow one to use external RAM as well, which would be a possible location for the heap as follows:

![](img/e568f5de-0419-4838-bbe9-3b238d5f9280.png)

The AVR Libc library implements a `malloc()` memory allocator routine, optimized for the AVR architecture. Using it, one can implement one's own `new` and `delete` functionality as well—if one so desires—since the AVR toolchain does not implement either.

In order to use external memory with an AVR MCU for heap storage, one would have to make sure that the external memory has been initialized, after which the address space becomes available to `malloc()`. The start and end of the heap space is hereby defined by these global variables:

```cpp
char * __malloc_heap_start 
char * __malloc_heap_end 
```

The AVR documentation has the following advice regarding adjusting the heap:

If the heap is going to be moved to external RAM, `__malloc_heap_end` must be adjusted accordingly. This can either be done at runtime, by writing directly to this variable, or it can be done automatically at link-time, by adjusting the value of the symbol `__heap_end`.

# Interrupts, ESP8266 IRAM_ATTR

On a desktop PC or server the entire application binary would be loaded into RAM. On MCUs though it is common to leave as many of the program instructions in the ROM as possible until they are needed. This means that most of our application's instructions cannot be executed immediately, but first have to be fetched from ROM before the CPU of our MCU can fetch them via the instruction bus to be executed.

On the AVR, each possible interrupt is defined in a vector table, which is stored in ROM. This offers either default handlers for each interrupt type, or the user-defined version. To mark an interrupt routine, one either uses the `__attribute__((signal))` attribute, or uses the `ISR()` macro:

```cpp
#include <avr/interrupt.h> 

ISR(ADC_vect) { 
         // user code 
} 
```

This macro handles the details of registering an interrupt. One just has to specify the name and define a function for the interrupt handler. This will then get called via the interrupt vector table.

With the ESP8266 (and its successor, the ESP32) we can mark the interrupt handler function with a special attribute, `IRAM_ATTR`. Unlike the AVR, the ESP8266 MCU does not have built-in ROM, but has to use its SPI peripheral to load any instructions into RAM, which is obviously quite slow.

An example of using this attribute with an interrupt handler looks as follows:

```cpp
void IRAM_ATTR MotionModule::interruptHandler() {
          int val = digitalRead(pin);
          if (val == HIGH) { motion = true; }
          else { motion = false; }
 }
```

Here, we have an interrupt handler that is connected to the signal from a motion detector, connected to an input pin. As with any well-written interrupt handler, it is quite simple and meant to be quickly executed before returning to the normal flow of the application.

Having this handler in ROM would mean that the routine would not respond near-instantly to the motion sensor's output changing. Worse, it would cause the handler to take much longer to finish, which would consequently delay the execution of the rest of the application code.

By marking it with `IRAM_ATTR`, we can avoid this problem, since the entire handler will already be in RAM when it's needed, instead of the whole system stalling as it waits for the SPI bus to return the requested data before it can continue.

Note that, tempting as it may seem, this kind of attribute should be used sparingly, as most MCUs have much more ROM than RAM. In the case of ESP8266, there are 64kB RAM for code execution complemented by possibly megabytes of external Flash ROM.

When compiling our code, the compiler will put instructions marked with this attribute into a special section, so that the MCU knows to load it into RAM.

# Concurrency

With a few exceptions, MCUs are single-core systems. Multitasking is not something that is generally done; instead, there's a single thread of execution with timers and interrupts adding asynchronous methods of operation.

Atomic operations are generally supported by compilers and AVR is no exception. The need for atomic blocks of instructions can be seen in the following cases. Keep in mind that while a few exceptions exist (MOVW to copy a register pair and indirect addressing via X, Y, Z pointers), instructions on an 8 bit architecture generally only affect 8 bit values.

*   A 16 bit variable is byte-wise read in the main function and updated in an ISR.
*   A 32 bit variable is read, modified and subsequently stored back in either main function or ISR while the other routine could try to access it.
*   The execution of a block of code is time-critical (bitbanging I/O, disabling JTAG).

A basic example for the first case is given in the AVR libc documentation:

```cpp
#include <cinttypes> 
#include <avr/interrupt.h> 
#include <avr/io.h> 
#include <util/atomic.h> 

volatile uint16_t ctr; 

ISR(TIMER1_OVF_vect) { 
   ctr--; 
} 

int main() { 
         ctr = 0x200; 
         start_timer(); 
         sei(); 
         uint16_t ctr_copy; 
         do { 
               ATOMIC_BLOCK(ATOMIC_FORCEON) 
               { 
                     ctr_copy = ctr; 
               } 
         } 
         while (ctr_copy != 0); 

         return 0; 
} 
```

In this code, a 16-bit integer is being changed in the interrupt handler, while the main routine is copying its value into a local variable. We call `sei()` (SEt global Interrupt flag) to ensure that the interrupt register is in a known state. The `volatile` keyword hints to the compiler that this variable and how it's accessed should not be optimized in any way.

Because we included the AVR atomic header, we can use the `ATOMIC_BLOCK` macro, along with the `ATOMIC_FORCEON` macro. What this does is create a code section that is guaranteed to be executed atomically, without any interference from interrupt handlers and the like. The parameter we pass to `ATOMIC_BLOCK` forces the global interrupt status flag into an enabled state.

Since we set this flag to the same state before we started the atomic block, we do not need to save the previous value of this flag, which saves resources.

As noted earlier, MCUs tend to be single-core systems, with limited multitasking and multithreading capabilities. For proper multithreading and multitasking, one would need to do context switches, whereby not only the stack pointer of the running task is saved, but also the state of all registers and related.

This means that while it would be possible to run multiple threads and tasks on a single MCU, in the case of 8-bit MCUs such as the AVR and PIC (8-bit range), the effort would most likely not be worth it, and would require a significant amount of labor.

On more powerful MCUs (like the ESP8255 and ARM Cortex-M), one could run **real-time OSes** (**RTOSes**), which implement exactly such context switching, without having to do all of the heavy lifting. We will look at RTOSes later in this chapter.

# AVR development with Nodate

Microchip provides a binary version of the GCC toolchain for AVR development. At the time of writing, the most recent release of AVR-GCC is 3.6.1, containing GCC version 5.4.0\. This implies full support for C++14 and limited support for C++17.

Using this toolchain is pretty easy. One can simply download it from the Microchip website, extract it to a suitable folder, and add the folder containing the GCC executable files to the system path. After this, it can be used to compile AVR applications. Some platforms will have the AVR toolchain available via a package manager as well, which makes the process even easier.

One thing that one may notice after installing this GCC toolchain is that there is no C++ STL available. As a result, one is limited to just the C++ language features supported by GCC. As the Microchip AVR FAQ notes:

*   Obviously, none of the C++ related standard functions, classes, and template classes are available.
*   The operators new and delete are not implemented; attempting to use them will cause the linker to complain about undefined external references. (This could perhaps be fixed.)
*   Some of the supplied include files are not C++ safe, that is, they need to be wrapped into `extern"C" { . . . }`. (This could certainly be fixed, too.)
*   Exceptions are not supported. Since exceptions are enabled by default in the C++ frontend, they explicitly need to be turned off using `-fno-exceptions` in the compiler options. Failing this, the linker will complain about an undefined external reference to `__gxx_personality_sj0`.

With the lack of a Libstdc++ implementation that would contain the STL features, we can only add such functionality by using a third-party implementation. These include versions that provide essentially the full STL, as well as lightweight re-implementations that do not follow the standard STL API. An example of the latter is the Arduino AVR core, which provides classes such as String and Vector, which are similar to their STL equivalents albeit with some limitations and differences.

An upcoming alternative to the Microchip AVR GCC toolchain is LLVM, a compiler framework to which experimental support for AVR as been recently added, and which at some point in the future should allow producing binaries for AVR MCUs, all the while providing full STL functionality via its Clang frontend (C/C++ support).

![](img/5b4b8498-6d84-46e3-9887-ab7249b81b3d.png)

Consider this an abstract snapshot of LLVM development—all the while illustrating the general concept of LLVM and its emphasis on Intermediate Representation.

Unfortunately the PIC range of MCUs, despite also being owned by Microchip and resembling AVR in many ways, does at this point not have a C++ compiler available for it from Microchip until one moves up to the PIC32 (MIPS-based) range of MCUs.

# Enter Nodate

You could at this point opt to use one of the IDEs we discussed previously in this chapter, but that wouldn't be nearly as educational for AVR development itself. For this reason, we will look at a simple application developed for an ATmega2560 board that uses a modified version of the Arduino AVR core, called Nodate ([https://github.com/MayaPosch/Nodate](https://github.com/MayaPosch/Nodate)). This framework restructures the original core to allow it to be used as a regular C++ library instead of only with the Arduino C-dialect parser and frontend.

Installing Nodate is pretty easy: simply download to a suitable location on one's system and have the `NODATE_HOME` system variable point to the root folder of the Nodate installation. After this, we can take one of the example applications as a basis for a new project.

# Example – CMOS IC Tester

Here, we will look at a more full-featured example project, implementing an **integrated circuit** (**IC**) tester for 5V logic chips. In addition to probing chips with its GPIO pins, this project also reads a chip description and test program (in the form of a logic table) from an SD card over SPI. User control is added in the form of a serial-based command-line interface.

First, we look at the `Makefile` for this Nodate project, as found in the root of the project:

```cpp
ARCH ?= avr

 # Board preset.
 BOARD ?= arduino_mega_2560

 # Set the name of the output (ELF & Hex) file.
 OUTPUT := sdinfo

 # Add files to include for compilation to these variables.
 APP_CPP_FILES = $(wildcard src/*.cpp)
 APP_C_FILES = $(wildcard src/*.c)

 #
 # --- End of user-editable variables --- #
 #

 # Nodate includes. Requires that the NODATE_HOME environment variable has been set.
 APPFOLDER=$(CURDIR)
 export

 all:
    $(MAKE) -C $(NODATE_HOME)

 flash:
    $(MAKE) -C $(NODATE_HOME) flash

 clean:
    $(MAKE) -C $(NODATE_HOME) clean
```

The first item we specify is the architecture we are targeting, since Nodate can be used to target other MCU types as well. Here, we specify AVR as the architecture.

Next, we use the preset for the Arduino Mega 2560 development board. Inside Nodate, we have a number of presets like these, which define a number of details about the board. For the Arduino Mega 2560, we get the following presets:

```cpp
MCU := atmega2560 
PROGRAMMER := wiring 
VARIANT := mega # "Arduino Mega" board type
```

If no board preset is defined, one has to define those variables in the project's Makefile and pick an existing value for each variable, each of which is defined as its own Makefile within the Nodate AVR subfolders. Alternatively, one can add one's own MCU, programmer, and (pin) variant file to Nodate, along with a new board preset, and use that.

With the makefile complete it is time to implement the main function:

```cpp
#include <wiring.h>
 #include <SPI.h>
 #include <SD.h>

 #include "serialcomm.h"
```

The wiring header provides access to all GPIO-related functionality. Furthermore, we include headers for the SPI bus, the SD card reader device, and a custom class that wraps the serial interface, as we will see in more detail in a moment:

```cpp
int main () {
    init();
    initVariant();

    Serial.begin(9600);

    SPI.begin();
```

Upon entering the main function, we initialize the GPIO functionality with a call to `init()`. The next call loads the pin configuration for the particular board we are targeting (the `VARIANT` variable on the top or in the board preset Makefile).

After this, we start the first serial port with a speed of 9,600 baud, followed by the SPI bus, and finally the output of a welcome message, as follows:

```cpp
   Serial.println("Initializing SD card...");

    if (!SD.begin(53)) {
          Serial.println("Initialization failed!");
          while (1);
    }

    Serial.println("initialization done.");

    Serial.println("Commands: index, chip");
    Serial.print("> ");
```

An SD card is expected to be attached to the Mega board at this point, containing a list of available chips we can test. Here, pin 53 is the hardware SPI chip-select pin that is conveniently located next to the rest of the SPI pins on the board.

Assuming the board is hooked up properly and the card can be read without issues, we are presented with a command-line prompt on the console screen:

```cpp
          while (1) {
                String cmd;
                while (!SerialComm::readLine(cmd)) { }

                if (cmd == "index") { readIndex(); }
                else if (cmd == "chip") { readChipConfig(); }
                else { Serial.println("Unknown command.");      }

                Serial.print("> ");
          }

          return 0;
 }
```

This loop simply waits for input to arrive on the serial input, after which it will attempt to execute the received command. The function we call for reading from the serial input is blocking, returning only if it has either received a newline (user pressed *Enter*), or its internal buffer size was exceeded without receiving a newline. In the latter case, we simply dismiss the input and try to read from the serial input once more. This concludes the `main()` implementation.

Let's now look at the header of the `SerialComm` class:

```cpp
#include <HardwareSerial.h>      // UART.

 static const int CHARBUFFERSIZE 64

 class SerialComm {
          static char charbuff[CHARBUFFERSIZE];

 public:
          static bool readLine(String &str);
 };
```

We include the header for the hardware serial connection support. This gives us access to the underlying UART peripheral. The class itself is purely static, defining the maximum size of the character buffer, and the function to read a line from the serial input.

Next is its implementation:

```cpp
#include "serialcomm.h"

 char SerialComm::charbuff[CHARBUFFERSIZE];

 bool SerialComm::readLine(String &str) {
          int index = 0;

          while (1) {
                while (Serial.available() == 0) { }

                char rc = Serial.read();
                Serial.print(rc);

                if (rc == '\n') {
                      charbuff[index] = 0;
                      str = charbuff;
                      return true;
                }

                if (rc >= 0x20 || rc == ' ') {
                      charbuff[index++] = rc;
                      if (index > CHARBUFFERSIZE) {
                            return false;
                      }
                }
          }

          return false;
 }
```

In the `while` loop, we first enter a loop that runs while there are no characters to be read in the serial input buffer. This makes it a blocking read.

Since we want to be able to see what we're typing, in the next section we echo back any character we have read. After this, we check whether we have received a newline character. If we did, we add a terminating null byte to the local buffer and read it into the String instance we were provided a reference to, after which we return true.

A possible improvement one could implement here is that of a backspace feature, where the user could delete characters in the read buffer by using the backspace key. For this, one would have to add a case for the backspace control character (ASCII 0x8), which would delete the last character from the buffer, and optionally also have the remote terminal delete its last visible character.

With no newline found yet, we continue to the next section. Here, we check whether we have received a valid character considered as ASCII 0x20, or a space. If we did, we continue to add the new character to the buffer and finally check whether we have reached the end of the read buffer. If we did not, we return false to indicate that the buffer is full yet no newline has been found.

Next are the handler functions `readIndex()` and `readChipConfig()` for the `index` and `chip` commands, respectively:

```cpp
void readIndex() {
          File sdFile = SD.open("chips.idx");
          if (!sdFile) {
                Serial.println("Failed to open IC index file.");
                Serial.println("Please check SD card and try again.");
                while(1);
          }

          Serial.println("Available chips:");
          while (sdFile.available()) {
                Serial.write(sdFile.read());
          }

          sdFile.close();
 }
```

This function makes heavy use of the `SD` and associated `File` classes from the Arduino SD card library. Essentially, we open the chips index file on the SD card, ensure we got a valid file handle, then proceed to read out and print each line in the file. This file is a simple line-based text-file, with one chip name per line.

At the end of the handler code, we're done reading from SD and the file handle can be closed with `sdFile.close()`. The same applies to the slightly more lengthy upcoming `readChipHandler()` implementation.

# Usage

As an example, when we run the test with a simple HEF4001 IC (4000 CMOS series Quad 2-Input OR Gate) hooked up, we have to add a file to the SD card which contains the test description and control data for this IC. The `4001.ic` test file is shown here as it lends itself to following along the code that parses it and performs the corresponding tests.

```cpp
HEF4001B
Quad 2-input NOR gate.
A1-A2: 22-27, Vss: GND, 3A-4B: 28-33, Vdd: 5V
22:0,23:0=24:1
22:0,23:1=24:0
22:1,23:0=24:0
22:1,23:1=24:0
26:0,27:0=25:1
26:0,27:1=25:0
26:1,27:0=25:0
26:1,27:1=25:0
28:0,29:0=30:1
28:0,29:1=30:0
28:1,29:0=30:0
28:1,29:1=30:0
33:0,32:0=31:1
33:0,32:1=31:0
33:1,32:0=31:0
33:1,32:1=31:0
```

The first three lines are printed verbatim as we saw earlier, with the remaining lines specifying individual test scenarios. These tests are lines and use the following format:

```cpp
<pin>:<value>,[..,]<pin>:<value>=<pin>:<value>
```

We write this file as `4001.ic` along with an updated `index.idx` file (containing the '4001' entry on a new line) to the SD card. to support more ICs we would simply repeat this pattern with their respective test sequences and list them in the index file.Finally there is the handler for the chip configuration, which also starts the testing procedure:

```cpp
 void readChipConfig() {
          Serial.println("Chip name?");
          Serial.print("> ");
          String chip;
          while (!SerialComm::readLine(chip)) { }
```

We start by asking the user for the name of the IC, as printed out earlier by the `index` command:

```cpp
          File sdFile = SD.open(chip + ".ic");      
          if (!sdFile) {
                Serial.println("Failed to open IC file.");
                Serial.println("Please check SD card and try again.");
                return;
          }

          String name = sdFile.readStringUntil('\n');
          String desc = sdFile.readStringUntil('\n');
```

We attempt to open the file with the IC details, continuing with reading out the file contents, starting with the name and description of the IC that we are testing:

```cpp
          Serial.println("Found IC:");
          Serial.println("Name: " + name);
          Serial.println("Description: " + desc);   

          String pins = sdFile.readStringUntil('\n');
          Serial.println(pins);
```

After displaying the name and description of this IC, we read out the line that contains the instructions on how to connect the IC to the headers of our Mega board:

```cpp

          Serial.println("Type 'start' and press <enter> to start test.");
          Serial.print("> ");
          String conf;
          while (!SerialComm::readLine(conf)) { }
          if (conf != "start") {
                Serial.println("Aborting test.");
                return;
          }
```

Here, we ask the user for confirmation on whether to start testing the IC. Any command beyond `start` will abort the test and return to the central command loop.

Upon receiving `start` as a command, the testing begins:

```cpp
          int result_pin, result_val;
          while (sdFile.available()) {
                // Read line, format:
                // <pin>:<value>, [..,]<pin>:<value>=<pin>:<value>
                pins = sdFile.readStringUntil('=');
                result_pin = sdFile.readStringUntil(':').toInt();
                result_val = sdFile.readStringUntil('\n').toInt();
                Serial.print("Result pin: ");
                Serial.print(result_pin);
                Serial.print(", expecting: ");
                Serial.println(result_val);
                Serial.print("\n");

                pinMode(result_pin, INPUT);
```

As the first step, we read out the next line in the IC file, which should contain the first test. The first section contains the input pin settings, with the section after the equal sign containing the IC's output pin and its expected value for this test.

We print out the board header number that the result pin is connected to and the expected value. Next, we set the result pin to be an input pin so that we can read it out after the test has finished:

```cpp
                int pin;
                bool val;
                int idx = 0;
                unsigned int pos = 0;
                while ((idx = pins.indexOf(':', pos)) > 0) {
                      int pin = pins.substring(pos, idx).toInt();
                      pos = idx + 1; // Move to character beyond the double colon.

                      bool val = false
                      if ((idx = pins.indexOf(",", pos)) > 0) {
                            val = pins.substring(pos, idx).toInt();
                            pos = idx + 1;
                      }
                      else {
                            val = pins.substring(pos).toInt();
                      }

                      Serial.print("Setting pin ");
                      Serial.print(pin);
                      Serial.print(" to ");
                      Serial.println(val);
                      Serial.print("\n");
                      pinMode(pin, OUTPUT);
                      digitalWrite(pin, val);
                }
```

For the actual test, we use the first String we read out from the file for this test, parsing it to get the values for the input pins. For each pin, we first get its number, then get the value (`0` or `1`).

We echo these pin numbers and values to the serial output, before setting the pin mode for these pins to output mode and then writing the test value to each of them, as follows:

```cpp

                delay(10);

                int res_val = digitalRead(result_pin);
                if (res_val != result_val) {
                      Serial.print("Error: got value ");
                      Serial.print(res_val);
                      Serial.println(" on the output.");
                      Serial.print("\n");
                }
                else {
                      Serial.println("Pass.");
                }
          }     

          sdFile.close();
 }
```

After leaving the inner loop, all of the input values will have been set. We just have to wait briefly to ensure that the IC has had time to settle on its new output values before we attempt to read out the result value on its output pin.

IC validation is a simple read on the result pin, after which we compare the value we received with the expected value. The result of this comparison is then printed to the serial output.

With the test complete, we close the IC file and return to the central command loop to await the next instructions.

After flashing the program to the Mega board and connecting with it on its serial port, we get the following result:

```cpp
    Initializing SD card...
    initialization done.
    Commands: index, chip
    > index  
```

After starting up, we get the message that the SD card was found and successfully initialized. We can now read from the SD card. We also see the available commands.

Next, we specify the `index` command to get an overview of the available ICs we can test:

```cpp
    Available chips:
    4001
    > chip
    Chip name?
    > 4001
    Found IC:
    Name: HEF4001B
    Description: Quad 2-input NOR gate.
    A1-A2: 22-27, Vss: GND, 3A-4B: 28-33, Vdd: 5V
    Type 'start' and press <enter> to start test.
    > start  
```

With just one IC available to test, we specify the `chip` command to enter the IC entry menu, after which we enter the IC's specifier.

This loads the file we put on the SD card and prints the first three lines. It then waits to give us time to hook up the chip, following the header numbers on the Mega board and the pin designations for the IC as provided by its datasheet.

After checking that we didn't get any of our wires crossed, we type `start` and confirm. This starts the test:

```cpp
    Result pin: 24, expecting: 1
    Setting pin 22 to 0
    Setting pin 23 to 0
    Pass.
    Result pin: 24, expecting: 0
    Setting pin 22 to 0
    Setting pin 23 to 1
    Pass.
    Result pin: 24, expecting: 0
    Setting pin 22 to 1
    Setting pin 23 to 0
    [...]
    Result pin: 31, expecting: 0
    Setting pin 33 to 1
    Setting pin 32 to 0
    Pass.
    Result pin: 31, expecting: 0
    Setting pin 33 to 1
    Setting pin 32 to 1
    Pass.
    >  
```

For each of the four identical OR gates in the chip, we run through the same truth table, testing each input combination. This specific IC passed with flying colors and can be safely used in a project.

This kind of testing device would be useful for testing any kind of 5V-level IC, including 74 and 4000 logic chips. It would also be possible to adapt the design to use the PWM, ADC, and other pins to test ICs that aren't strictly digital in their inputs and outputs.

# ESP8266 development with Sming

For ESP8266-based development, no official development tools exist from its creator (Espressif) beyond a bare-metal and RTOS-based SDK. Open source projects including Arduino then provide a more developer-friendly framework to develop applications with. The C++ alternative to Arduino on ESP8266 is Sming ([https://github.com/SmingHub/Sming](https://github.com/SmingHub/Sming)), which is an Arduino-compatible framework, similar to Nodate for AVR, which we looked at in the previous section.

In the next chapter ([Chapter 5](886aecf2-8926-4aec-8045-a07ae2cdde84.xhtml), *Example - Soil Humidity Monitor with Wi-Fi*) we will take an in-depth look at developing with this framework on the ESP8266.

# ARM MCU development

Developing for ARM MCU platforms isn't significantly different from developing for AVR MCUs, except that C++ is far better supported, and there exists a wide range of toolchains to choose from, as we saw at the beginning of this chapter with just the list of popular IDEs. The list of available RTOSes for Cortex-M is much larger than for AVR or ESP8266 as well.

Using a free and open source compiler including GCC and LLVM to target a wide range of ARM MCU architectures (Cortex-M-based and similar) is where developing for ARM MCUs offers a lot of freedom, along with easy access to the full C++ STL (though one might want to hold off on exceptions).

When doing bare-metal development for Cortex-M MCUs, one may have to add this linker flag to provide basic stubs for some functionality that is normally provided by the OS:

```cpp
-specs=nosys.specs 
```

One thing that makes ARM MCUs less attractive is that there are far fewer *standard* boards and MCUs, such as with what one sees with AVR in the form of the Arduino boards. Although the Arduino foundation at one point made the Arduino Due board based around a SAM3X8E Cortex-M3 MCU, this board uses the same form factor and roughly same pin layout (just being 3.3V I/O-based instead of 5V) as the ATmega2560-based Arduino Mega board.

Because of this design choice a lot of the functionality of the MCU has not been broken out and is inaccessible unless one is very handy with a soldering iron and thin wires. This functionality includes the Ethernet connection, tens of GPIO (digital) pins, and so on. This same lack of breaking out all pins also happens with the Arduino Mega (ATmega2560) board, but on this Cortex-M MCU it becomes even more noticeable.

The result of this is that as a development and prototyping board, there aren't any obvious generic picks. One might be tempted to just use the relatively cheap and plentiful prototyping boards like those provided by STMicroelectronics for their range of Cortex-M-based MCUs.

# RTOS usage

With the limited resources available on the average MCU, and the generally fairly straightforward process loop in the applications that run on them, it is hard to make a case for using an RTOS on these MCUs. It's not until one has to do complicated resource and task management that it becomes attractive to use an RTOS in order to save development time.

The benefit of using an RTOS thus lies mostly in preventing one from having to reinvent the wheel. This is however something that has to be decided on a case-by-case basis. For most projects, having to integrate an RTOS into the development toolchain is more likely than an unrealistic idea that would add more to the workload than it would lighten it.

For projects where one is, for example, trying to balance CPU time and system resources between different communication and storage interfaces, as well as a user interface, the use of an RTOS might make a lot of sense, however.

As we saw in this chapter, a lot of embedded development uses a simple loop (super-loop) along with a number of interrupts to handle real-time tasks. When sharing data between an interrupt function and the super-loop, it is the responsibility of the developer to ensure that it is done safely.

Here, an RTOS would offer a scheduler and even the ability to run tasks (processes) that are isolated from each other (especially on MCUs that have a **Memory Management Unit** (**MMU**)). On a multi-core MCU, an RTOS easily allows one to make effective use of all cores without having to do one's own scheduling.

As with all things, the use of an RTOS isn't just a collection of advantages. Even ignoring the increase in ROM and RAM space requirements that will likely result from adding an RTOS to one's project, it will also fundamentally change some system interactions and may (paradoxically) result in interrupt latency increasing.

This is why, although the name has *real-time* in it, it is very hard to get more real-time than to use a simple execution loop and a handful of interrupts. The benefit of an RTOS, thus, is absolutely something about which no blanket statements can be made, especially when a support library or framework for bare-metal programming (such as the Arduino-compatible ones addressed in this chapter) is already available to make prototyping and developing for production as simple as tying a number of existing libraries together.

# Summary

In this chapter, we took a look at how to select the right MCU for a new project, as well as how to add peripherals and deal with Ethernet and serial interface requirements in a project. We considered how memory is laid out in a variety of MCUs and how to deal with the stack and heap. Finally, we looked at an example AVR project, how to develop for other MCU architectures, and whether to use an RTOS.

At this point, the reader is expected to be able to argue why they would pick one MCU over another, based on a set of project requirements. They should be capable of implementing simple projects using the UART and other peripherals, and understand proper memory management as well as the use of interrupts.

In the next chapter, we will take a good look at how to develop for the ESP8266, in the form of an embedded project that will keep track of soil moisture levels and control a watering pump when needed.