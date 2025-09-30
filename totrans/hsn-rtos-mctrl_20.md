# Multi-Processor and Multi-Core Systems

So far, we've discussed many different ways of programming a single **microcontroller unit** (**MCU**). But what if the task at hand requires more processing than a single-core MCU can supply? What if the mechanical constraints of the system dictate the use of multiple MCUs physically distributed in the system while working together to complete a task? What about cases where reliability is paramount and a single failed processor results in a catastrophic system failure? All of these cases require the use of more than one processing core and, in some cases, more than one MCU. 

This chapter explores multi-core and multi-processor solutions and their different applications. First, we'll take a look at the different design requirements that might drive a multi-core/processor solution. We'll then dive a bit deeper into the different ways FreeRTOS can be used in multi-core/processor systems. Finally, some recommendations on choosing an inter-processor communication scheme will be presented.

In a nutshell, we will cover the following topics:

*   Introducing multi-core and multi-processor systems
*   Exploring multi-core systems 
*   Exploring multi-processor systems 
*   Exploring inter-processor communication
*   Choosing between multi-core and multi-processor systems 

# Technical requirements

There are no technical requirements for this chapter.

# Introducing multi-core and multi-processor systems 

First, let's get our terminology straight. A **multi-core** design is a single chip with multiple CPUs inside it, with at least some memory shared between the cores:

![](img/90863091-a1a4-4e4b-b106-686ff0e5ac41.png)

Multi-core parts span a very broad range, from the larger, 64-bit parts that have multiple identical CPU cores to the ARM big.LITTLE architecture, which incorporates both high-bandwidth CPUs and power-conscious MCUs in the same package. Recently, multi-core MCUs have also become more commonly available. **G****raphics processing units** (**GPUs**) can also be grouped into the multi-core category.

A **multi-processor** **system** is one where there are multiple processor chips in the design. For the purposes of our discussions here, these chips can reside on the same **printed circuit board assembly** (**PCBA**) or different PCBAs distributed throughout a larger system:

![](img/6471e083-789e-4918-9c19-9bc468a40ed1.png)

Both multi-core and multi-processor topologies can be found in many different places, such as smartphones, small networked sensing devices, industrial automation equipment, test equipment, medical equipment, appliances, and of course, a range of computing devices, such as desktops, laptops, and so on. 

There are many different driving forces for using these two different topologies, beyond a simple need for more or faster processing. Sometimes, a system needs to come online *instantly,* without waiting for a full **general-purpose operating system** (**GPOS**) to boot. Occasionally, there are regulatory requirements that are easier to meet by segregating system functions into multiple cores (and code bases) so that only a portion of the total code (or system) is required to go through a stringent review. There could be electro-mechanical considerations in a system (such as long wire runs to motors/actuators or sensitive analog signals) that are best addressed by having a processor in close physical proximity. In high-reliability systems, redundancy is very common.

Now we have a general idea of the terminology, let's get into some additional details and use cases for these systems, starting with multi-core designs.

# Exploring multi-core systems

First, let's cover a few different types of multi-core systems. They have two primary types of configurations/architectures: heterogeneous and homogeneous. A heterogeneous system is one that has multiple cores, but they are different in some way. Contrast this with a homogeneous system, where all CPUs can be treated identically and interchangeably.

# Heterogeneous multi-core systems

A heterogeneous multi-core system has at least two processing cores in the same device and includes differences in either the processor architecture of the core or the way the cores access shared resources, such as system memory, peripherals, or I/O. For example, at the lower end of the spectrum, we can have multiple MCU cores on the same chip. The LPC54100 series from NXP incorporates a Cortex-M0+ and a Cortex-M4, both running at 150 Mhz, in the same package.

In this device, the MCU cores are different, but their connection to system peripherals is identical—except for instruction and data buses, which are only available on the Cortex-M4:

![](img/af09d5a0-4ec4-48dd-8dc3-63829ce3ad4b.png)

We can use systems like these in different ways:

*   **Segmenting hard real-time operations from more general-purpose computing**: The M0+ could handle low-level peripheral or hardware access/control, while the M4 handles the higher-level functionality required, such as GUIs and connectivity.
*   **Power conscious design**: Low-level control and interfacing is performed on the lower-power M0+, only activating the M4 when computationally expensive operations are required.

Since the LPC54100 has two MCU cores, we'll focus on bare-metal programming (no operating system) and operating systems that don't require a full-blown **memory management unit** (**MMU**), such as FreeRTOS. Running different (or multiple copies of the same) operating systems on the two cores is called **asymmetric multi-processing**. 

The name *asymmetric* comes from the fact that the two cores are treated differently from one another—there is *asymmetry* between them. This is quite a bit different from the *symmetric* multi-core approached used on desktop-based operating systems, where the various cores are all treated equally. Symmetric multi-core systems will be covered in the *Homogeneous multi-core systems*section.

For example, we could run multiple copies of FreeRTOS on each of the two cores:

![](img/ba090720-8cbe-4cc1-a479-c5880bdaf153.png)

In a configuration like this, the two cores run completely independently from one another. Even though FreeRTOS is being run on both cores, there is no flash program space shared between the cores—each core has a firmware image that is independent from the other. RAM behaves in the same way—the full RAM memory space is available to both cores, but by default, each core will receive its own area for stack, heap, global variables, and so on.

So, each core is running its own program—how do the two programs coordinate activities between each other? We need some way of passing information back and forth—but how?

# Inter-core communication

Information sharing between the cores is possible, but is subject to the same concurrent-access considerations that any other multi-threaded environment has, which is why mailbox hardware is typically included onchip. This hardware is dedicated to facilitating communication between the two cores. Mailboxes will generally have the following features:

*   **Hardware mutex functionality**: Used to protect RAM shared between the two cores. The idea is identical to mutexes in a pure software environment—they are used to provide mutually exclusive access to a shared resource.
*   **Interrupts to/from each core**: These interrupts can be raised by a core after writing data to a shared area of memory, alerting the other core that a message/data is available.

# Legacy application extension

We're not limited to running FreeRTOS on both cores—any mixture of RTOSes or bare metal can be mixed or matched between the cores. Let's say a bare-metal legacy application already existed but some new additional functionality was required to take advantage of a new opportunity. For example, to stay competitive, the device might need a *facelift* and have a GUI, web frontend, or IoT stack build added to it. The new functionality could potentially be developed separately from the underlying legacy code, leaving the legacy code largely intact and undisturbed.

For example, the legacy code could be run on the Cortex-M0+, while the new functionality is added to the Cortex-M4:

![](img/17c744a6-20cc-4737-8882-a1555c2455fe.png)

In a setup like this, whether shared RAM is used as a data exchange between the cores will depend greatly on how comfortable a team is in modifying the legacy code base and how the application is structured. For example, rather than modifying an existing code base to use proper mailbox-implemented mutexes before accessing a shared data structure, it might be preferable to use a pre-existing hardware interface as the data transfer mechanism, treating the secondary CPU more like an external client. Since many legacy systems use UARTs as the primary interface to the system, it is possible to use these data streams as an interface between the processors, keeping modifications to the legacy code to a minimum:

![](img/5a5c1620-8b8f-4a82-abea-8644bd06f944.png)

This approach avoids significant modifications to the legacy code base at the expense of using a slower interface (physical peripherals are slower and more CPU-intensive than simple memory transfers) and routing signals outside the processor. Although far from ideal, this approach can be used to test the viability of a new opportunity before investing significant engineering effort in a more elegant solution:

![](img/504b6b0b-95b7-4272-98a4-b8241746eb62.png)

This type of approach allows the team to focus on developing new interfaces for an existing system—whose core functionality doesn't need to change—with minimal impact on the original system.

Depending on the circumstances, it may also make more sense to leave the legacy code on the original MCU, rather than porting it to a core inside a new MCU. Each project will likely have its own the constraints required to guide this decision. Although all of this this might look like a simple task from a very high level, each project usually has some hidden complexities that need to be considered.

# High-demand hard real-time systems

At the other end of the heterogenous multi-core spectrum from an NXP LPC54100 would be a device such as the NXP i.Mx8, which contains two Cortex-A72s, four Cortex-A53s, two Cortex-M4Fs, one DSP, and two GPUs. A system such as this one will generally be used where extremely computationally intensive operations are required, in addition to low-latency or hard real-time interactions with hardware. Computer vision, AI, on-target adaptive machine learning, and advanced closed-loop control systems are all reasonable applications for the i.Mx8\. So, instead of incorporating an i.Mx8 (or similar CPU) into a product, why not use a more general purpose computing solution for a system that requires this much computing power? After all, general-purpose computers have had GPUs and multi-core CPUs for a decade or more, right?

In some systems, it might be perfectly acceptable to run a more general-purpose computing hardware and operating system. However, when there are *hard real-time requirements*(the system is considered to have failed if a real-time deadline was missed), a GPOS won't be sufficient. A compelling reason for using a device such as the i.Mx8, rather than simply a GPOS on top of a CPU/GPU combination, is that hard real-time capable low-latency cores such as the Cortex-M4 are used to handle hard real-time tasks, where extremely reliable low latency is paramount. The higher-throughput hardware is used for doing the computationally *heavy lifting* operations, where throughput is important, but higher latency and less determinism can be tolerated:

![](img/62e8105e-1277-4da8-8deb-dbc0674427c6.png)

The smaller MCU-based cores are extremely good at performing low-level exchanges with hardware such as sensors and actuators. Timing-sensitive operations requiring the use of specialized timing peripherals are best left to the MCU hardware. For example, a motor control system might require directly controlling an H bridge and reading data from an encoder that uses an obscure/proprietary timing format. This is fairly straightforward to implement using an MCU that has dedicated timing hardware. Differential PWM signals with dead-time insertion used for motor control and high-resolution timing capture are both fairly common features. All of this tightly controlled, low-latency control structure can be implemented using the MCU and its specialized peripherals (either on bare metal or on an RTOS), then higher-level commands can be exposed to a GPOS. Specifically on the i.Mx8, we can now perform very low-level, timing-sensitive operations using MCUs, while simultaneously performing the high-level, massively parallel operations required for computer vision, machine learning, and AI using the higher-performance Cortex-A processors, DSP, and GPUs.

Heterogeneous systems aren't limited to embedded systems! Heterogeneous topologies have existed for very large computing clusters for decades, but we're keeping our focus on examples most relevant to the embedded space.

So, now that we've covered some examples of heterogenous multi-core systems, what about homogeneous multi-core systems?

# Homogeneous multi-core systems

As you might expect from the name, a homogeneous multi-core system is one where all of the cores are the same. These types of multi-core systems have been traditionally found in desktop computing. Rather than having individual cores tailored to perform a few types of tasks very well (as with heterogenous systems), there are multiple cores that are all identical. Rather than programming individual cores with specific tasks that are tied to the cores, all of the cores are treated identically. This type of approach is referred to as symmetric multi-processing (there is symmetry between all of the cores in the system); they are all treated identically. In a symmetric system, cores will be exposed to a single kernel, rather than divided up into multiple kernels/schedulers. 

Even in asymmetric multi-processing setups, there can be components that are symmetric. For example, the i.Mx8 mentioned earlier will usually have the Cortex-A53 cores set up in a symmetric multi-processing arrangement, where all four cores are available for scheduling by a single kernel (and all treated identically).

But what about when there is a need for processors in different physical locations? Or what if a single processor is limited in its functionality by the number of pins it has available?

# Exploring multi-processor systems

Similar to the way multi-core systems are excellent for segmenting firmware functionality and providing parallel execution, multi-processor systems are useful in many situations for a variety of reasons. Let's take a look at a few examples.

# Distributed systems

Embedded systems often have a very large amount of interaction with the physical world. Unlike the digital realm, where 1s and 0s can literally be sent around the world without a second thought, the physical world is a harsh place for sensitive analog signals—minimizing the distanced traversed can be critical. It is a good idea to keep analog processing as close to its source as possible. For a mixed signal system with analog components, this means keeping the signal paths as short as possible and getting the sensitive analog signals processed and converted into their digital representations as close to the source as possible:

![](img/66eb9800-a090-4657-91b4-a75808a6832b.png)

In medium-to-high power systems, reducing the distance traversed by wires carrying current to control motors, solenoids, and other actuators will reduce the radiated electromagnetic emissions of the system (always a good idea). If the I/O in question is physically removed from the rest of the system, including an MCU in close proximity is an excellent way of localizing the digitization of the sensitive signals, which makes the system more immune to **electromagnetic interference** (**EMI**) while simultaneously minimizing the amount of wiring. In high vibration and motion environments, fewer wires means fewer potential points of mechanical failure, which results in higher reliability, less downtime, and fewer service calls. 

# Parallel development

Using multiple processors also makes it very easy to provide a level of parallelism in the actual development of the system. Since teams will often find it easiest to focus on a well-defined subsystem, creating multiple subsystems makes running true parallel development (and reducing the overall schedule) a possibility. Each subsystem can be demarcated by its own processor and communication interface, along with a clear list of the responsibilities of the subsystem:

![](img/9a64a938-2190-42ed-8139-e5234fd9d286.png)

This approach also has the advantage of encouraging each team to fully test their system in isolation, documenting the interfaces and functionality as they move through development. Finally, it tends to keep any surprises during integration to a minimum, since the team is forced to put more thought into the entire architecture before starting development.

# Design reuse

As processors begin to have a plethora of I/O connected to them, they may still have plenty of processing resources available but run out of available pins. At this point, there is a decision to make. ICs meant to provide port expansion are available, but should they be used? If you're designing a system with reuse in mind, it is important to see whether a subsystem approach can be employed, instead of creating a huge monolithic design, where all of the hardware and firmware is intertwined and tightly coupled. Sometimes, when the pin capacities of a single MCU are reached, it is an indication that the MCU is performing the functionality of several different subsystems. Often, if these subsystems are broken down and individually developed, they can be *dropped* into future products without modification, which can greatly decrease future projects' risks and schedule.

# High-reliability systems

High-reliability systems will often include multiple cores or processors for their critical functionality. However, rather than using this extra processing power to run individual parallel operations, they are set up for some level of redundancy. There are different ways of achieving redundancy. One path to creating a redundant system is for the cores to run in lockstep with one another. The results of each processor are meticulously checked against one another to detect any discrepancies. If a problem is found, the core (or processor) is taken offline and reset, with a set of tests run to ensure it comes back up correctly—then, it is put back into service. 

In systems like these, there can be environmental considerations, such as EMI from running motors, solenoid valves, or other actuators. Sometimes the source of the environmental noise is more extraordinary, such as solar radiation, which is often a concern for high-altitude and space-bound systems.

Now that we've explored the reasons why having multiple processors in a system can be useful, let's take a look at how to get all of these processors talking to one another.

# Exploring inter-processor communication

Inter-processor communication was mentioned briefly in the context of distributed systems. Let's take a look at some of the considerations that go into choosing a suitable inter-processor bus.

# Choosing the right communication medium

There are many considerations when choosing the communication medium used between processors, which we can break into a few different major categories.

The first is **timing**. In a real-time system, timing considerations are often some of the most important. If a message sent between nodes doesn't make it to its destination on time and intact, it can have serious consequences:

*   **Latency**: How long will it take for a message to be sent and a response to be received? Having the ability to react quickly to communication between subsystems is often quite important.
*   **Maximum jitter**: How much variability is there in the latency? Each system has its own requirements for how much variability is acceptable.
*   **Error detection/reliability**: Does the communication medium provide a way of determining whether a message was received correctly and on time?
*   **Throughput**: How much data can be pushed over the communication medium? For communication mediums that contain control data, throughput will often be measured in messages, rather than raw data (such as KB/sec or MB/sec). Often, maximum reliability and minimal latency will come at the cost of raw data transfer throughput—each message will contain additional overhead and handshaking.

The next category of considerations is **physical requirements**. Sometimes, physical requirements are quite important, other times they may hardly be a constraint. Here are some simple points to consider:

*   **Noise immunity**: Does the communication channel need to run through an electrically noisy environment? What types of cabling are required for proper EMI shielding?
*   **Number of nodes in the system**: How many nodes are required in the complete system? Most standards will have an upper bound on the number of connections due to electrical constraints.
*   **Distance**: How long will the run need to be? Will it be a short, chip-to-chip run within the PCB or a long run between buildings? Distributed systems can have widely different meanings to different developers and industries.
*   **Required peripherals**: How much extra circuitry is acceptable? What kinds/sizes of connectors can be tolerated?

Then, we have **development team/project constraints**. Each team and project is fairly unique, but there are some common topics that should be covered:

*   **Complexity**: How much code is required to get the protocol up and running? Has the required external circuitry been proven to be functional? Does our team feel like the features provided by the solution are worth the development time required to implement it?
*   **Existing familiarity**: Has anyone on the team used this communication scheme before and is that experience directly relevant to the current project/product? Do we need to learn something new that is a better fit, rather than using something we're already comfortable with but isn't actually the best solution?
*   **Budget**: Does the communication scheme require any expensive components, such as exotic ICs, connectors, or proprietary stacks? Is it worth buying in aspects of the solution or contracting out some of the implementation?

As you can imagine from the long list of considerations, there is no *one-size-fits-all* communication mechanism that is an excellent fit for all applications. That's why we have so many to choose from. 

For example, while an industrial Ethernet communication solution may provide excellent latency and noise performance, the fact that it requires specialized hardware will make it unsuitable for many applications where it is not an explicit requirement. On the flip side, a low-performance serial protocol such as RS-232 may be extremely easy to implement but have an unacceptably high amount of EMI and be susceptible to noise when used at high speeds. On the other hand, the complexity of a full TCP/IP stack might put off many would-be adopters, unless someone on the team already has familiarity with it and a driver stack is readily available for the target platform.

# Communication standards

From the previous list of considerations, we can see that choosing a method for inter-processor communication isn't one size fits all. To help provide an idea of what's available, here are some examples of commonly used buses for MCU-based systems and some brief commentary on how they might be useful in a multi-processor system. This list is by no means exhaustive. Also, each standard has its own merits under different circumstances.

# Controller area network

A **controller area network** (**CAN**) is the communication backbone for many subsystems in the automotive industry. The advantages of CAN are its robust physical layer, a prioritized messaging scheme, and multi-master bus arbitration. Many MCUs include dedicated CAN peripherals, which helps to ease the implementation. CAN is most naturally suited for shorter messages, since the data field of extended frames may only contain up to 8 bytes.

# Ethernet

Nearly all medium- to high-performance MCUs have provisions for Ethernet, requiring an external PHY, magnetics, and a connector for the hardware implementation. The sticking point here is ensuring suitable networking protocol stacks are available. The advantage of this approach is a wide range of options for popular protocols that run on top of TCP and UDP, as well as readily available, inexpensive hardware that can be used to build out a full network if required. 

Similar to Modbus, Ethernet will often be chosen as the externally facing interface, rather than an inter-processor bus. Depending on the system architecture and hardware availability, there might not be a reason that it couldn't be used for both.

# Inter-integrated communication bus

**Inter-integrated communication bus** (**I2C**) is most often used for communicating with low-bandwidth peripherals, such as sensors and EEPROMs. Most often, an MCU will be configured as the I2C bus master with one or more slave I2Cs. However, many MCUs contain I2C controllers that can be used to implement either the master or slave side of I2C. There are many aspects of the I2C protocol that make it non-deterministic, such as the ability for slaves to hold the clock line until they are ready to receive more data (clock stretching) and multi-master arbitration.

# Local interconnect network

**Local interconnect network** (**LIN**) is a commonly used automotive network subsystem for a maximum of 16 nodes when a full CAN is too complex or expensive to implement. The LIN physical layer is less fault-tolerant than CAN, but it is also more deterministic, since there can only be one bus master. STM32 USARTS will often have some helpful LIN-mode functionality built into the peripheral, but an external PHY IC is still required.

# Modbus

**Modbus** is a protocol that historically ran on top of an RS-485 physical layer and is very popular in the industrial space as an externally facing protocol (although these days, the protocol is commonly found on top of TCP). Modbus is a fairly simple register-oriented protocol. 

# Serial peripheral interface

A **serial peripheral interface** (**SPI**) can also be very useful as an easy-to-implement, highly deterministic inter-processor communication medium, especially when the accuracy of a slave isn't high enough to achieve the tight tolerances required for high baud rates on an asynchronous serial port. All the same drawbacks for custom asynchronous protocols also exist for SPI-based custom protocols, with the additional constraint that slave devices will have hard real-time constraints imposed based on how quickly the master needs responses back from the slave(s). 

Since the SPI clock is driven by the master, it is the only device that can initiate a transfer. For example, if a slave is required to have a response ready within 30 µS of receiving a command from the master and it takes the slave 31 S, the transfer is likely to be worthless. This can make SPI very attractive when tight determinism is required, but unnecessarily difficult to implement otherwise. Depending on the environment, the MCU's onboard SPI peripheral might need to be used with external differential transceivers to increase signal integrity.

# USB as an inter-processor communication bus

Now that more medium- to high-performance MCUs include a USB host, it is becoming more viable as an inter-processor communication bus. Whether or not USB is viable in a given application hinges on the number of nodes and the availability of a full USB stack and developers that can harness it. While the USB virtual comm class used in this book wasn't deterministic since it used bulk endpoints, interrupt transfers can be used to achieve deterministic scheduling of transfers over USB, since they are polled by the host at a rate defined during enumeration. For example, on a high-speed USB link (which will often require an external PHY), this equates to messages up to 1 KB polled every 125 µS.

We've only scratched the surface of the possibilities for inter-processor communication in this section—there are many other options available, each with their own features, advantages, and disadvantages, depending on your project's requirements. 

Now that we have a good understanding of what multi-core and multi-processor systems are, some common topologies, and some ways of communicating between the processors, let's take a step back and evaluate whether a multi-core or multi-processor design is necessary.

# Choosing between multi-core and multi-processor systems

With more powerful MCUs and CPUs being announced every month, there is a virtually endless number of options to pick from. Multi-core MCUs are becoming more common. But the real question is—do you really need multiple cores or multiple processors in your design? Yes, they are readily available, but will it ultimately help or hurt the design in the long run? 

# When to use multi-core MCUs

There are several cases where multi-core MCUs are an excellent fit:

*   When true parallel-processing is required and space is constrained
*   When tightly coupled parallel threads of execution are required

If your design is space-constrained, requires true parallel processing, or communication speed between two parallel processes is extremely critical, a multi-core MCU may be the best option. If the application requires parallel processing from multiple cores and can't be implemented using other hardware already present on the MCU—for example, running multiple CPU-intensive algorithms in parallel—a multi-core MCU might be the best fit for the application.

However, it is important to be aware of some downsides and alternatives. A multi-core MCU will likely be more challenging to replace (both in finding a replacement and porting the code) than discrete MCUs. Does the application truly need parallel execution at the CPU level or is there simply a need to perform some operations (such as communication) in parallel? If there is parallel functionality required that can be implemented using dedicated peripheral hardware (for example, filling communication buffers using DMA connected to a hardware peripheral), implementing the *parallel* functionality could be achieved without a second core.

Some potential alternatives to multi-core MCUs are as follows:

*   Offloading some processing to hardware peripherals
*   Ensuring DMA is utilized as much as possible
*   Multiple MCUs

# When to use multi-processor systems

Multi-processor systems are useful in a wide variety of circumstances, such as the following:

*   When subsystem reuse is possible
*   When multiple teams are available to work on a large project in parallel
*   When the device is large and physically dispersed
*   When EMI considerations are paramount

However, while multi-processor systems are useful, they do have some potential drawbacks:

*   Additional latency compared to having a single MCU.
*   Real-time multi-processor communication can become complex and time-consuming to implement. 
*   Additional up-front planning is required to ensure proper subsystems are developed.

# Summary

In this chapter, you were introduced to both multi-core and multi-processor systems and we covered some examples of each. You should now have an understanding of what the differences between them are and when designing a system using either approach is appropriate. Several examples of inter-processor communication schemes were also introduced, along with some highlights and advantages of each, as they relate to embedded real-time systems.

The great thing about multi-core and multi-processor topologies is that once you have a solid understanding of the building blocks for the concurrent system design (which we've covered), creating systems with more cores is just a matter of judiciously placing hardware where concurrent processing and abstraction will have the most impact.

In the next chapter, we'll be covering some of the problems you'll likely encounter during development and some potential solutions.

# Questions

As we conclude, here is a list of questions for you to test your knowledge on this chapter's material. You will find the answers in the *Assessments* section of the appendix:

1.  What is the difference between a multi-core architecture and a multi-processor architecture?
2.  A mixture of operating systems and bare-metal programming can be used in an asymmetric multi-processing architecture.
    *   True
    *   False
3.  When selecting an inter-processor communication bus, the bus with the highest available transfer rate should always be used.
    *   True
    *   False
4.  Should multi-processor solutions be avoided because they add complexity to the architecture?

# Further reading

*   NXP AN11609—LPC5410x dual core usage: [https://www.nxp.com/docs/en/data-sheet/LPC5410X.pdf](https://www.nxp.com/docs/en/data-sheet/LPC5410X.pdf)
*   Keil—USB concepts: [https://www.keil.com/pack/doc/mw/USB/html/_u_s_b__concepts.html](https://www.keil.com/pack/doc/mw/USB/html/_u_s_b__concepts.html)