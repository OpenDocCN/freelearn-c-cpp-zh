# Chapter 8. Designing Visual Output Feedback

Interaction is everything about control and feedback. You control a system by performing actions upon it. You can even modify it. The system gives you feedback by providing useful information about what it does when you modify it.

In the previous chapter, we learned more about us controlling Arduino than Arduino giving us feedback. For instance, we used buttons and knobs to send data to Arduino, making it working for us. Of course, there are a lot of point of view, and we can easily consider controlling an LED and giving feedback to Arduino. But usually, we talk about feedback when we want to qualify a return of information from the system to us.

Arkalgud Ramaprasad, Professor at the Department of Information and Decision Sciences at the College of Business Administration, University of Illinois, Chicago, defines feedback as follows:

> "Information about the gap between the actual level and the reference level of a system parameter which is used to alter the gap in some way."

We already talked about some visual output in [Chapter 5](ch05.html "Chapter 5. Sensing with Digital Inputs"), *Sensing Digital Inputs*, when we tried to visualize the result of our button push events. This visual rendering resulting from our push events was feedback.

We are now going to talk about the design of visual feedback systems based especially on LEDs driven by the Arduino board. LEDs are the easiest systems with which to provide visual feedback from Arduino.

We are going to learn about the following:

*   How to use basic monochromatic LEDs
*   How to make LED matrices and how to multiplex LEDs
*   How to use RGB LEDs

We will finish the chapter by introducing the LCD display device.

# Using LEDs

LEDs can be monochromatic or polychromatic. Indeed, there are many types of LEDs. Before going though some examples, let's discover some of these LED types.

## Different types of LEDs

Usually, LEDs are used both to block the current coming from a line to its cathode leg and to give light feedback when the current goes into its anode:

![Different types of LEDs](img/7584_08_001.jpg)

The different models that we can find are as follows:

*   Basic LEDs
*   **OLED** (**Organic LED** made by layering the organic semi-conductor part)
*   **AMOLED** (**Active Matrix OLED** provides a high density of pixels for big size screens)
*   **FOLED** (**Flexible** **OLED**)

We will only talk about basic LEDs here. By the term "basic", I mean an LED with discrete components like the one in the preceding image.

The package can vary from two-legged components with a molded epoxy-like lens at the top, to surface components that provide many connectors, as shown in the following screenshot:

![Different types of LEDs](img/7584OS_08_002.jpg)

We can also sort them, using their light's color characteristics, into:

*   Monochromatic LEDs
*   Polychromatic LEDs

In each case, the visible color of an LED is given by the color of the molded epoxy cap; the LED itself emits the same wavelength.

### Monochromatic LEDS

Monochromatic LEDs emit one color only.

The most usual monochromatic LEDs emit constant colors at each voltage need.

### Polychromatic LEDs

Polychromatic LEDs can emit more than one color, depending on several parameters such as voltage but also depending on the leg fed with current in case of an LED with more than one leg.

The most important characteristic here is controllability. Polychromatic LEDs have to be easily controllable. This means that we should be able to control each color by switching it on or off.

Here is a classic RGB LED with common cathode and three different anodes:

![Polychromatic LEDs](img/7584OS_08_003.jpg)

This type of LED is the way to go with our Arduino stuff. They aren't expensive (around 1.2 Euros per 100 LEDs ), considering the fact that we can control them easily and produce a very huge range of colors with them.

We are going to understand how we can deal with multiple LEDs and also polychromatic RGB LEDs in the following pages.

## Remembering the Hello LED example

In Hello LED, we made an LED blink for 250 ms of every 1000 ms that pass. Let's see its schematic view once again to maintain the flow of your reading:

![Remembering the Hello LED example](img/7584OS_08_032.jpg)

The code for Hello LED is as follows:

[PRE0]

Intuitively, in the next examples, we are going to try using more than one LED, playing with both monochromatic and polychromatic LEDs.

## Multiple monochromatic LEDs

Since we are talking about feedback here, and not just pure output, we will build a small example showing you how to deal with multiple buttons and multiple LEDs. Don't worry if you are totally unable to understand this right now; just continue reading.

### Two buttons and two LEDs

We already spoke about playing with multiple buttons in [Chapter 5](ch05.html "Chapter 5. Sensing with Digital Inputs"), *Sensing Digital Inputs*. Let's build a new circuit now.

Here are the schematics:

![Two buttons and two LEDs](img/7584_08_004.jpg)

It's preferable to continue drawing the electric diagram related for each schematic.

Basically, the multiple buttons example from [Chapter 5](ch05.html "Chapter 5. Sensing with Digital Inputs"), *Sensing Digital Inputs*; however, we have removed one button and added two LEDs instead.

![Two buttons and two LEDs](img/7584_08_005.jpg)

As you know, the digital pins of Arduino can be used as inputs or outputs. We can see that two switches are connected on one side to a 5 V Arduino pin and on the other side to the digital pins 2 and 3, with one pull-down resistor related to each of those latter pins, sinking the current to Arduino's ground pin.

We can also see that an LED is connected each to digital pin 8 and 9 on one side; both are connected to Arduino's ground pin.

Nothing's really incredible about that.

Before you design a dedicated firmware, you need to briefly cover something very important: coupling. It is a must to know for any interface design; more widely for interaction design.

### Control and feedback coupling in interaction design

This section is considered a subchapter for two main reasons:

*   Firstly, it sounds great and is key to keeping the motivation groove on
*   Secondly, this part is the key for all your future human-machine interface design

As you already know, Arduino (thanks to its firmware) links the control and feedback sides. It is really important to keep this in mind.

Whatever the type of the external system may be, it is often considered as human from the Arduino point of view. As soon as you want to design an interaction system, you will have to deal with that.

We can summarize this concept with a very simple schematic in order to fix things in the mind.

Indeed, you have to understand that the firmware we are about to design will create a control-feedback coupling.

A **control/feedback coupling** is a set of rules that define how a system behaves when it receives orders from us and how it reacts by giving us (or not) feedback.

This hard-coded set of rules is very important to understand.

![Control and feedback coupling in interaction design](img/7584_08_006.jpg)

But, imagine that you want to control another system with Arduino. In that case, you may want to make the coupling outside Arduino itself.

See the second figure **EXTERNAL SYSTEM 2**, where I put the coupling outside Arduino. Usually, **EXTERNAL SYSTEM 1** is us and **EXTERNAL SYSTEM 2** is a computer:

![Control and feedback coupling in interaction design](img/7584_08_007.jpg)

We can now quote a real-life example. As with many users of interfaces and remote controllers, I like and I need to control complex software on my computer with minimalistic hardware gears.

I like the minimalistic and open source **Monome interface** ([http://monome.org](http://monome.org)) designed by Brian Crabtree. I used it a lot, and still use it sometimes. It is basically a matrix of LEDs and buttons. The amazing trick under the hood is that there is NO coupling inside the gear.

![Control and feedback coupling in interaction design](img/7584OS_08_008.jpg)

The preceding image is of Monome 256 by Brian Crabtree and its very well-made wooden case.

If it isn't directly written like that in all the docs, I would like to define it to my friends and students like this: "The Monome concept is the most minimalistic interface you'll ever need because it only provides a way of controlling LEDs; beside of that, you have many buttons, but there are no logical or physical links between buttons and LEDs."

If Monome doesn't provide a real, already made coupling between buttons and LEDs, it's because it would be very restrictive and would even remove all the creativity!

Since there is a very raw and efficient protocol designed ([http://monome.org/data/monome256_protocol.txt](http://monome.org/data/monome256_protocol.txt)) to especially control LEDs and read buttons pushes, we are ourselves able to create and design our own coupling. Monome is also provided with the **Monome Serial Router**, which is a very small application that basically translates the raw protocol into **OSC** ([http://archive.cnmat.berkeley.edu/OpenSoundControl/](http://archive.cnmat.berkeley.edu/OpenSoundControl/)) or **MIDI** ([http://www.midi.org/](http://www.midi.org/)). We will discuss them in later sections of this chapter. These are very common in multimedia interaction design; OSC can be transported over networks, while MIDI is very suited for links between music-related equipment such as sequencers and synthesizers.

This short digression wouldn't be complete without another schematic about the Monome.

Check it and let's learn more about it after that:

![Control and feedback coupling in interaction design](img/7584_08_009.jpg)

The smart minimalistic Monome interface in its usual computer-based setup

Here is a schematic of the Monome 64 interface, in that usual computer-based setup inside of which the coupling occurs. This is the real setup that I used on stage for a music performance many times ([https://vimeo.com/20110773](https://vimeo.com/20110773)).

I designed a specific coupling inside Max 6, translating specific messages from/to the Monome itself, but from/to the software too, especially Ableton Live ([https://www.ableton.com](https://www.ableton.com)).

This is a very powerful system that controls things and provides feedback with which you can basically build your coupling from the ground up and transform your raw and minimalistic interface into whatever you need.

This was a small part of a more global monologue about interaction design.

Let's build this coupling firmware right now, and see how we can couple controls and feedback into a basic sample code.

### The coupling firmware

Here, we only use the Arduino switches and LEDs and no computer actually.

Let's design a basic firmware, including coupling, based on this pseudocode:

*   If I push switch 1, LED 1 is switched on, and if I release it, LED 1 is switched off
*   If I push switch 2, LED 2 is switched on, and if I release it, LED 2 is switched off

In order to manipulate new elements and ideas, we are going to use a library named `Bounce`. It provides an easy way to debounce digital pin inputs. We already spoke about debouncing in the *Understanding the debounce concept* section of [Chapter 5](ch05.html "Chapter 5. Sensing with Digital Inputs"), *Sensing Digital Inputs*. Reminding you of that a bit: if no button absorbs the bounce totally when you push it, we can smoothen things and filter the non-desired harsh value jumps by using software.

You can find instructions about the `Bounce` library at [http://arduino.cc/playground/Code/Bounce](http://arduino.cc/playground/Code/Bounce).

Let's check that piece of code:

[PRE1]

You can find it in the `Chapter08/feedbacks_2x2/` folder.

This code includes the Bounce header file, that is, the Bounce library, at the beginning.

Then I defined four constants according to the digital input and output pins, where we put switches and LEDs in the circuit.

The Bounce library requires to instantiate each debouncer, as follows:

[PRE2]

I chose a debounce time of 7 ms. This means, if you remember correctly, that two value changes occurring (voluntarily or non-voluntarily) very fast in a time interval of less than 7ms wouldn't be considered by the system, avoiding strange and uncanny bouncing results.

The `setup()` block isn't really difficult, it only defines digital pins as inputs for buttons and outputs for LEDs (please remember that digital pins can be both and that you have to choose at some point).

`loop()` begins by the update of both debouncers, after which we read each debounced button state value.

At last, we handle the LED controls, depending on the button states. Where does the coupling occur? Of course, at this very last step. We couple our control (buttons pushed) to our feedback (LED lights) in that firmware. Let's upload and test it.

### More LEDs?

We basically just saw how to attach more than one LED to our Arduino. Of course, we could do the very same way with more than two LEDs. You can find code handling six LEDs and six switches in the `Chapter05/feedbacks_6x6/` folder.

But hey, I have a question for you: how would you handle more LEDs with an Arduino Uno? Please don't answer that by saying "I'll buy an Arduino MEGA" because then I would ask you how you would handle more than 50 LEDs.

The right answer is **multiplexing**. Let's check how we can handle a lot of LEDs.

# Multiplexing LEDs

The concept of multiplexing is an interesting and efficient one. It is the key to having a bunch of peripherals connected to our Arduino boards.

Multiplexing provides a way to use few I/O pins on the board while using a lot of external components. The link between Arduino and these external components is made by using a multiplexer/demultiplexer (also shortened to mux/demux).

We spoke about input multiplexing in [Chapter 6](ch06.html "Chapter 6. Sensing the World – Feeling with Analog Inputs"), *Playing with Analog Inputs*.

We are going to use the 74HC595 component here. Its datasheet can be found at [http://www.nxp.com/documents/data_sheet/74HC_HCT595.pdf](http://www.nxp.com/documents/data_sheet/74HC_HCT595.pdf).

This component is an 8-bit serial-in / serial-or-parallel-out. This means it is controlled through a serial interface, basically using three pins with Arduino and can drive with eight of its pins.

I'm going to show you how you can control eight LEDs with only three pins of your Arduino. Since Arduino Uno contains 12 digital usable pins (I'm not taking 0 and 1, as usual), we can easily imagine using 4 x 75HC595 to control 4 x 8 = 32 monochromatic LEDs with this system. I'll provide the code to do that as well.

## Connecting 75HC595 to Arduino and LEDs

As we learnt together with the CD4051 and the multiplexing of analog inputs, we are going to wire the chip to a 75HC595 shift register in order to mux/demux eight digital output pins. Let's check the wiring:

![Connecting 75HC595 to Arduino and LEDs](img/7584_08_010.jpg)

We have the Arduino supplying power to the breadboard. Each resistor provides 220 ohms resistance.

The 75HC595 grabs the GND and 5 V potential for its own supply and configuration.

Basically, 74HC595 needs to be connected through pins 11, 12, and 14 in order to be controlled by a serial protocol handled here by Arduino.

Let's check 74HC595 itself:

![Connecting 75HC595 to Arduino and LEDs](img/7584_08_011.jpg)

*   Pins 8 and 16 are used for internal power supply.
*   Pin 10 is named **Master Reset**, and in order to activate it, you have to connect this pin to the ground. That is the reason why, in normal operational states of work, we drive it to 5 V.
*   Pin 13 is the output enable input pin and has to be kept active in order to make the whole device output currents. Connecting it to the ground does this.
*   Pin 11 is the shift register clock input.
*   Pin 12 is the storage register clock input, also named **Latch**.
*   Pin 14 is the serial data input.
*   Pin 15 and pins 1 to 7 are the output pins.

Our small and inexpensive serial link to the Arduino, handled by pins 11, 12 and 14, provides an easy way to control and basically load eight bits into the device. We can cycle over the eight bits and send them serially to the device that stores them in its registers.

These types of devices are usually referred to as **Shift Registers** we shift bits from 0 to 7 while loading them.

Then, each state is outputted to the correct output from Q0 to Q7, transposing the previously transmitted states over serial.

This is a direct illustration of the serial-to-parallel conversion that we talked about in the previous chapter. We had a data flow coming sequentially, retained until the register is globally loaded, then pushing this to many output pins.

Now, let's visualize the wiring diagram:

![Connecting 75HC595 to Arduino and LEDs](img/7584_08_012.jpg)

An eight-LED array with resistors wired to the 74HC595 shift register

## Firmware for shift register handling

We are going to learn how to design a firmware specifically for these kinds of shift registers. This firmware is basically made for the 595 but wouldn't require a lot of modifications to be used with other integrated circuits. You'd specially have to take care about three serial pins, Latch, Clock, and Data.

Because I want to teach you each time a bit more than the exact content evoked by each chapter title, I created a very inexpensive and small random groove machine for you. Its purpose is to generate random bytes. These bytes will then be sent to the shift register in order to feed or not each LED. You'll then have then a neat random pattern of LEDs.

You can find the code for this in the `Chapter08/ Multiplexing_8Leds/` folder.

Let's check it:

[PRE3]

### Global shift register programming pattern

First, let's check the global structure.

I first define the 3 three pins of the 595 shift register. Then, I set up each of them as output in the `setup()` block.

Then, I have a pattern that looks similar to the following:

[PRE4]

This is the usual pattern for shift- registering operations.The `latch-pin`, as evoked explained before, is the one providing us a way to inform the integrated circuit about the fact that we want to load it with data, and then we want it to apply these this data to its outputs.

This is a bit like saying:

*   Latch-pin LOW = "Hi there, let's store what I'm about to send to you."
*   Latch-pin HIGH = "Ok, now use the data I just sent to commute to your outputs or not."

Then, we have this `shiftOut()`. This function provides an easy way to send data per entire bytes packets to a specific pin (the data pin) using a specific clock/ rate speed over a particular pin (the clock pin), and given an order of transmission (`MSBFIRST` or `LSBFIRST`).

Even though we aren't going to describe the things under- the- hood here, you have to understand the MSB and LSB concept.

Let's consider a byte: `1 0 1 0 0 1 1 0`.

The **MSB** is the abbreviation of **Most Significant Bit**. This bit is at the left-most position (the one of the bit having of the greatest value). Here, its value is `1`.

The **LSB** is stands for the **Least Significant Bit**. This bit is at the right-most position (the bit of the smallest value) It is the bit the most at the right (the one of the bit having the smallest value). Here, its value is `0`.

By fixing this argument in the `shiftOut()` function, we are providing special information about the sense of the transmission. Indeed, we can send the previous byte by sending these bits: `1` then, `0`, then `1 0 0 1 1 0` (MSBFIRST), or by sending these bits: `0 1 1 0 0 1 0 1` (LSBFIRST).

### Playing with chance and random seeds

I would like to provide an example from my personal ways of programming. Here, I'm going to describe an inexpensive and small system generating random bytes. These bytes will then be sent to the the 595, and our 8 eight-LEDs array will have a very random state.

Random, in computers, isn't really random. Indeed, the `random()` function is a pseudo-random number generator. It can also be named a **deterministic random bit generator** (**DRBG**). Indeed, the sequence is (totally) determined by a small set of initial values, including the seed.

For a particular seed, a pseudo-random number generator generates the same number sequences each time the same number sequences.

But, we you can use a trick here to disturb determinism a little bit more.

Imagine that you make the seed vary sometimes. You can also introduce an external factor of randomness into your system. As we already explained before in this book, there is always some electronic noises coming going to from to the ADC even if nothing is wired to the analog inputs. You can use that external/physical noise by reading the analog input 0, for instance.

As we now well know, analog `analogRead()` provides a number from 0 to 1023\. This is a huge resolution for our purpose here.

This is what I have put in the firmware.

I defined a counter variable and a byte. I'm first reading the value coming from the ADC for the analog pin 0 in the `setup()` first. Then, I'm generating generated a random byte with a `for()` loop and the `bitWrite()` function.

I'm writing each bit of the byte `LED_states` using numbers generated by the `random(2)` number function, which gives 0 or 1, randomly. Then, I'm using use the pseudo-random-generated byte into the structure previously described.

I'm redefining each 5000 `loop()` execution of the seed by reading the ADC for the analog pin 0.

### Note

If you want to use `random()` with computers, including Arduino and embedded systems, grab some physical and external noise.

Now, let's move further.

We can use many 74HC595 shift registers for LED handling, but imagine that you need to save some more digital pins. Okay, we saw we can save a lot using shift registers. One shift registers requires three digital pins and drives eight LEDs. It means we save five pins with each shift register, considering we wire eight LEDs.

What if you need A LOT more? What if you need to save all the other pins for switches handling, for instance?

Let's daisy chain now!

## Daisy chaining multiple 74HC595 shift registers

A **daisy chain** is a wiring scheme used to link multiple devices in a sequence or even a ring.

Indeed, since we already understood a bit more about how shift registers work, we could have the idea to extend this to multiple shift registers wired together, couldn't we?

I'm going to show you how to do this by using the **ShiftOutX** library by Juan Hernandez. I had very nice results with Version 1.0, and I advise you to use this one.

You can download it here: [http://arduino.cc/playground/Main/ShiftOutX](http://arduino.cc/playground/Main/ShiftOutX). You can install it by following the procedure explained in the appendice.

### Linking multiple shift registers

What would each shift register need to know about?

The serial clock, the latch, and the data are the necessary points of information that have to be transmitted all along the device chain. Let's check a schematic:

![Linking multiple shift registers](img/7584_08_013.jpg)

Two shift registers daisy chained driving 16 monochromatic LEDs with only three digital pins on the Arduino

I used the same colors as with the previous circuit for the clock (blue), latch (green), and serial data (orange).

The serial clock and latch are shared across the shift registers. The command/order coming from Arduino to synchronize serial communication with the clock and to tell shift registers to store or apply data received to their output has to be coherent.

The serial data coming from Arduino first goes into the first shift register, which sends the serial data to the second one. This is the core of the chaining concept.

Let's check the circuit diagram to put this in mind:

![Linking multiple shift registers](img/7584_08_014.jpg)

Circuit diagram of two daisy-chained shift registers driving 16 monochromatic LEDs

### Firmware handling two shift registers and 16 LEDs

The firmware includes the `ShiftOutX` library ShiftOutX as wrote before. It provides very easy and smooth handling for daisy chaining of shift registers.

Here is the the code for the firmware.

You can find it in the `Chapter08/Multiplexing_WithDaisyChain/` folder:

[PRE5]

The ShiftOutX library can be used in many ways. We are using it here following in the same way that we did with `ShiftOut`, the library part of the core and suited for the use of only one shift register.

First, we have to include the library by using **Sketch | Import Library | ShiftOutX**.

It includes two header files at the beginning, namely, `ShiftOutX.h` and `ShiftPinNo.h`.

Then, we define a new variable storing the number of shift registers in the chain.

At last, we instantiate the ShiftOutX library by using the following code:

[PRE6]

The code in `setup()` changed a bit. Indeed, there are no more setup statements for digital pins. This part is handled by the library, which can look weird but is very usual. Indeed, when you instantiated the library before, you passed three pins of Arduino as arguments, and in fact, this statement also sets up the pins as outputs.

The `loop()` block is almost the same as before. Indeed, I included again the small random groove machine with the analog read trick. But I'm creating two random bytes, this time. Indeed, this is because I need 16 values and I want to use the `shiftOut_16` function to send all my data in the same statement. It is quite easy and usual to generate bytes, then aggregate them into an `unsigned short int` datatype by using bitwise operators.

Let's detail this operation a bit.

When we generate random bytes, we have two series of 8 eight bits. Let's take the following example:

[PRE7]

If we want to store them in one place, what could we do? We can shift one and then add the shifted one to the other one, couldn't we?

[PRE8]

Then, if we add a byte using the bitwise operator (`|`), we get:

[PRE9]

The result seems to be a concatenation of all the bits.

This is what we are doing in this part of the code. Then we use `shiftOut_16()` to send all the data to the two shift registers. Hey, what should we do with the four shift registers? The same thing in the same way!

Probably we would have to shift more using `<< 32`, `<< 16`, and again `<<8`, in order to store all our the bytes into a variable that we could send using `shiftOut_32()` functions.

By using this library, you can have two groups, each one containing eight shift registers.

What does that mean?

It means that you can drive 2 x 8 x 8 = 128 outputs using only four pins (two latches but common serial clock and data). It sounds crazy, doesn't it?

In real life, it is totally possible to use only one Arduino to make this kind of architecture, but we would have to take care of something very important, the current amount. In this particular case of 128 LEDs, we should imagine the worst case when all the LEDs would be switched on. The amount of current driven could even burn the Arduino board, which would protect itself by resetting, sometimes. But personally, I wouldn't even try.

### Current short considerations

The Arduino board, using USB power supply, cannot drive more than 500 mA. All combined pins cannot drive more than 200 mA, and no pin can drive more than 40 mA. It can vary a bit from one board type to another, but these are real, absolute maximum ratings.

We didn't make these considerations and the following calculations because, in our examples, we only used a few devices and components, but you could sometimes be tempted to build a huge device such as I made sometimes, for example, with the Protodeck controller.

Let's take an example in order to look closer at some current calculations.

Imagine that you have an LED that needs around 10 mA to bright light up correctly (without burning at the second blink!!)

This would mean you'd have 8 x 10 mA for one eight -LEDs array, driven by one 595 shift register, if all LEDs were to be switched on at the same time.

80 mA would be the global current driven by one 595 shift register from the Arduino Vcc source.

If you had more 595 shift registers, the magnitude of the current would increase. You have to know that all integrated circuits also consume current. Their consumption isn't generally taken into consideration, because it is very small. For instance, the 595 shift register circuit only consumes around 80 micro Amperes itself, which means 0.008 mA. Compared to our LEDs, it is negligible. Resistors consume current too, even if they are often used to protect LEDs, they are very useful.

Anyway, we are about to learn another very neat and smart trick that can be used for monochromatic or RGB LEDs.

Let's move to a world full of colors.

# Using RGB LEDs

RGB stands for Red, Green, and Blue, as you were probably guessing.

I don't talk about LEDs that can change their color according to the voltage you apply to them. LEDs of this kind exists, but as far as I experimented, these aren't the way to go, especially while still learning steps.

I'm talking about common cathode and common anode RGB LEDs.

## Some control concepts

What do you need to control an LED?

You need to be able to apply a current to its legs. More precisely, you need to be able to create a difference of potential between its legs.

The direct application of this principle is what we have already tested in the first part of this chapter, which remind us how we can switch on an LED: we you need to control the current using digital output pins of our Arduino, knowing the LED we want to control has its node wired to the output pin and its cathode wired to the ground, with a resistor on the line too.

We can discuss the different ways of controls, and you are going to understand that very quickly with the next image.

In order to make a digital output sourcing current, we need to write with `digitalWrite` to it a value of `HIGH`. In that this case, the considered digital output will be internally connected to a 5 V battery and will produce a voltage of 5 V. That means that the wired LED between it and the ground will be fed by a current.

In the other case, if we apply 5 V to an LED and if we want to switch it on, we need to write a value of `LOW` to the digital pin to which it is linked. In this case, the digital pin will be internally connected to the ground and will sink the current.

These are the two ways of controlling the current.

Check the following diagram:

![Some control concepts](img/7584_08_015.jpg)

## Different types of RGB LEDs

Let's check the two common RGB LEDs:

![Different types of RGB LEDs](img/7584_08_016.jpg)

There are basically three LEDs in one package, with different types of wiring inside. The way of making this package isn't really about wiring inside, but I won't debate that here.

If you followed me correctly, you may have guessed that we need more digital outputs to connect RGB LEDs. Indeed, the previous section talked about saving digital pins. I guess you understand why it could be important to save pins and to plan our circuit architectures carefully.

## Lighting an RGB LED

Check this basic circuit:

![Lighting an RGB LED](img/7584_08_017.jpg)

An RGB LED wired to Arduino

Check the code now. You can find it in the `Chapter08/One_RGB_LED/` folder.

[PRE10]

Again, some tips are present inside this code.

### Red, Green, and Blue light components and colors

First, what is the point here? I want to make the RGB LED cycle through all the possible states. Some math can help to list all the states.

We have an ordered list of three elements, each one of which can be on or off. Thus, there are 23 states, that which means eight states in total:

| R | G | B | Resulting color |
| --- | --- | --- | --- |
| Off | Off | Off | OFF |
| Off | Off | On | Blue |
| Off | On | Off | Green |
| Off | On | On | Cyan |
| On | Off | Off | Red |
| On | Off | On | Purple |
| On | On | Off | Orange |
| On | On | On | White |

Only by switching each color component on or off, can we change the global RGB LED state.

Don't forget that the system works exactly as if we were controlling three monochromatic LEDS through three digital outputs from Arduino.

First, we define three variables storing the different colors LED connectors.

Then, in the `setup()`, we set those 3 three pins as output.

### Multiple imbricated for() loops

At last, the `loop()` block contains triple-imbricated `for()` loops. What's that? It is nice efficient way to be sure to match all the cases possible. It is also an easy way to cycle each number possible. Let's check the first steps, in order to understand this imbricated loops concept better.:

*   1st step: **r = 0, g = 0, and b = 0** implies everything is OFF, then pauses for 150ms in that state
*   2nd step: **r = 0, g = 0, and b = 1** implies only BLUE is switched on, then pauses for 150ms in that state
*   3rd step: **r = 0, g = 1, and b = 0** implies only GREEN is switched on, then pauses for 150ms in that state

The innermost loop is always the one executed the most number of times.

Is that okay? You bet, it is!

You also may have noticed that I didn't write HIGH or LOW as arguments for the `digitalWrite()` function. Indeed, HIGH and LOW are constants defined in the Arduino core library and are only replace the values 1 and 0, respectively.

In order to prove this, and especially to show you for the first time where the Arduino core files sit, the important file to check here is `Arduino.h`.

On a Windows systems, it can be found in the `Arduino` folder inside some subdirectories, depending upon the version of the IDE.

On OS X, it is in `Arduino.app/Contents/Resources/Java/hardware/arduino/cores/arduino/Arduino.h`. We can see the content of an application package by right-clicking on the package itself.

In this file, we can read a big list of constants, among many other definitions.

And finally, we can retrieve the following:

[PRE11]

Yes, the HIGH and LOW keywords are just constants for 1 and 0.

This is the reason why I'm directly feeding `digitalWrite()` with `0` and `1` through the imbricated loops, cycling over all the states possible for each LED, and as a consequence, over all states for the RGB LED.

Using this concept, we are going to dig further by making an LED array.

# Building LED arrays

LED arrays are basically LEDs wired as a matrix.

We are going to build a 3 x 3 LEDs matrix together. This is not that hard, and we'll approach this task with a really nice, neat and smart concept that can really optimize your hardware designs.

Let's check the simplest schematic of this book:

![Building LED arrays](img/7584_08_018.jpg)

An LED can blink when a current feeds it, when a voltage is applied to its legs

In order to switch off the LED shown in the preceding screenshot, we can stop to create the 5 V current at its node. No voltage means no current feeding. We can also cut the circuit itself to switch off the LED. And at last, we can change the ground by putting adding a 5 V source current.

This means that as soon as the difference of potential is cancelled, the LED is switched off.

An LED array is based on these double controls possible.

We are going to introduce a new component right here, the transistor.

## A new friend named transistor

A **transistor** is a special component that we introduced a bit in the first part of this book.

![A new friend named transistor](img/7584_08_019.jpg)

The usual NPN transistor with its three legs

This component is usually used in three main cases:

*   As a digital switch in a logical circuit
*   As a signal amplifier
*   As a voltage stabilizer combined with other components

Transistors are the most widespread components in the world. They are not only used as discrete components (independent ones) but are also combined with many others into a high-density system, for instance, in processors.

## The Darlington transistors array, ULN2003

We are going to use a transistor here, as included inside an integrated circuit named ULN2003\. What a pretty name! A more explicit one is **High-current** **Darlington Transistors Array**. Ok, I know that doesn't help!

![The Darlington transistors array, ULN2003](img/7584_08_020.jpg)

Its datasheet can be found at

[http://www.ti.com/lit/ds/symlink/uln2003a.pdf](http://www.ti.com/lit/ds/symlink/uln2003a.pdf).

It contains seven pins named inputs and seven named outputs. We can see also a 0 V pin (the number 8) and the COM pin 9 too.

The principle is simple and amazing:

*   0 V has to be connected to the ground
*   If you apply 5 V to the input *n*, the output *n* is commuted to ground

If you apply 0 V to the input *n*, the output *n* will get disconnected.

This can easily be used as a current sink array of switches.

Combined with 74HC595, we'll drive our 3 x 3 LED matrix right now:

![The Darlington transistors array, ULN2003](img/7584_08_021.jpg)

A case where inputs 1 and 2 are fed, resulting in the commutation of outputs 1 and 2 (pin 16 and 14)

## The LED matrix

Let's check how we can wire our matrix, keeping in mind that we have to be able to control each LED independently, of course.

This kind of design is very usual. You can easily find ready made matrices of LEDs wired like this, sold in packages with connectors available related to rows and columns.

An LED matrix is basically an array where:

*   Each row pops out a connector related to all the anodes of that row
*   Each column pops out a connector related to all the cathodes of that column

This is not law, and I found some matrices wired totally in the opposite way and sometimes quite strangely. So, be careful and check the datasheet. Here, we are going to study a very basic LED matrix in order to dig into that concept:

![The LED matrix](img/7584_08_022.jpg)

A basic 3 x 3 LED matrix

Let's look at the LED matrix architecture concept.

How can we control it? By controlling, I mean addressing the good LED to a good behavior, from being switched on or off.

Let's imagine that, if we want to light up the **LED 2**, we have to:

*   Connect **ROW 1** to 5 V
*   Connect **COLUMN 2** to the ground

Good! We can light up that **LED 2**.

Let's move further. Let's imagine that, if we want to light up the **LED 2** and **LED 4**, we have to:

*   Connect **ROW 1** to 5 V
*   Connect **COLUMN 2** to the ground
*   Connect **ROW 2** to 5 V
*   Connect **COLUMN 1** to the ground

Did you notice something?

If you follow the steps carefully, you should have something strange on your matrix:

**LED 1**, **LED 2**, **LED 4**, and **LED5** would be switched ON

Problem appeared: if we put 5 V to the **ROW 1**, how can you distinguish **COLUMN 1** and **COLUMN 2**?

We are going to see that it isn't hard at all and that it just uses a small trick related to our persistence of vision.

## Cycling and POV

We can take care of the problem encountered in the previous section by cycling our matrix quickly.

The trick is switching ON only one column at a time. This could also work by switching ON only one row at a time, of course.

Let's take our previous problem: If we want to light up the **LED 2** and **LED 4**, we have to:

*   Connect **ROW 1** to 5 V and **COLUMN 1** to 5 V only
*   Then, put connect **ROW 2** to 5 V and **COLUMN 2** to 5 V only

If we we are doing that this very quickly, our eyes won't see that there is only one LED switched on at a time.

The pseudo code would be:

[PRE12]

## The circuit

First, the circuit has to be designed. Here is how it looks:

![The circuit](img/7584_08_023.jpg)

Arduino wired to a 595 shift register driving each row and column through an ULN2003

Let's now check the circuit diagram:

![The circuit](img/7584_08_024.jpg)

Circuit diagram showing the handling of matrix rows and columns

We have the now well-known shift register 74HC595.

This one is wired to a ULN2003 shift register and to the matrix' rows, the ULN2003 being wired to the columns of the matrix.

What is that design pattern?

The shift register grabs data from the serial protocol-based messages sent by the Arduino from its digital pin 2\. As we tested before, the shift register is clocked to Arduino, and as soon as its latch pin is connected to HIGH (=(equal to 5 V), it drives an output to 5V or not, depending upon the data sent to it by Arduino. As a consequence, we can control each row of the matrix, feeding them rows with 5V or not through the data sent to the shift register.

In order to switch on LEDs, we have to close the circuit on which they are plugged, the electrical line, I mean. We can feed the **ROW 1** with a 5V current, but if we don't put this or that column to the ground, the circuit won't be closed and no LED will be switched on. Right?

The ULN2003 was made precisely for the purpose of ground commutation, as we already saw. And if we feed 5V to one of its input, it commutes the corresponding out *n* pin to the ground. So, with our 595 shift registers, we can control the 5V commutation for rows, and the ground commutation for columns. We now have total control over our matrix.

Especially, we are going to check the code, including the power cycle of columns previously explained.

## The 3 x 3 LED matrix code

You can find the following 3 x 3 LED matrix code in the `Chapter08/LedMatrix3x3/` folder:

[PRE13]

This code is quite self-explanatory with comments, but let's check it out a bit more.

The global structure reminds the one in Multiplexing_8Leds.

We have an integers array named LED_states. We are storing data for each LED state inside of it. The setup() block is quite easy, defining each digital pin used in the communication with the 595 shift- register and then grabbing a random seed from the ADC. The loop() is a bit more tricky. At first, we generating nine random values and store them in the LED_states array. Then, we initialize/define some values:

*   `data` is the byte sent to the shift register
*   `dataRow` is the part of the byte handling row state (commuted to 5V or not)
*   `dataColumn` is the part of the byte handling column state (commuted to the ground or not)
*   `currentLed` keeps the trace of the current handled handled by the LED

Then, those imbricated loops occur.

For each column (first for() loop), we activate it the loop by using a small/cheap and fast bitwise operator:

[PRE14]

`(4 – c)` goes from `4` to `2`, all along this first `loop()` ; function; then, `dataColumn` goes from: `0 0 0 1 0 0 0 0` to `0 0 0 0 1 0 0 0`, and at last `0 0 0 0 0 1 0 0`.

What's going on right here? It is all about coding.

The first three bits (beginning at the left, the MSB bit) handle the rows of our matrix. Indeed the three rows are connected to the `Q0`, `Q1`, and `Q2` pins of the 595 shift register.

The second three-bit group handles the ULN2003, which itself handles the columns.

By feeding 5 V from `Q0`, `Q1`, and `Q2` of the 595, we handle rows. By feeding 5 V from `Q3`, `Q4`, and `Q5` of the 595, we handle columns through the ULN2003.

Good!

We still have two bits not unused bits right here, the last two.

Let's take look at our our code again.

At each column turn of the for() loop, we move the bit corresponding to the column to the right, commuting each column to the ground cyclically.

Then, for each column, we cycle the row on the same mode, testing the state of the corresponding LED that we have to push to the 595\. If the LED has to be switched on, we store the corresponding bit in the dataRow variable with the same bitwise operation trick.

Then, we sum those two parts, resulting in the data variable.

For instance, if we are on the second row and the second column and the LED has to be switched on, then the data stored will be:

`0 1 0 0 1 0 0 0`.

If we are at (1,3), then the data stored will be data will store:

`1 0 0 0 0 1 0 0`.

Then, we have the pattern that adds Latch to LOW, shifting out bits stored in data to the shift- register, and then putting adds Latch to HIGH to commit data to the Q0 to Q7 outputs, feeding the right elements in the circuits.

At the end of each row handled, we reset the three bits corresponding to the first three rows and increment the `currentLed` variable.

At the end of each column handled, we reset the three bits corresponding to the next three columns.

This global imbricated structure makes us ensures that we'll have only one LED switched on at a time.

What is the consequence of the current consumption?

We'll only have one LED fed, which means we'll have our maximum consumption potentially divided by nine. Yes, that sounds great!

Then, we have the pattern grabbing that grabs a new seed, each 5000 loop() turn.

We just learned how to handle LED matrices quite easily and to reduce our power consumption at the same time.

But, I'm not satisfied. Usually, creators and artists are generally never completely satisfied, but here, trust me it's different; we could do better things than just switching on and off LEDs. We could dim them too and switch them from a very low intensity to a very high one, making some different shades of light.

# Simulating analog outputs with PWM

As we know very well by now, it's okay to switch on/off LEDs, and as we are going to see in the next chapter, to switch on/off many things too by using digital pins as output on the Arduino.

We also know how to read states from digital pins set up as inputs, and even values from 0 to 1023 from the analog inputs from in the ADC.

As far as we know, there isn't analog output on the Arduino.

What would an analog output add? It would provide a way to write values other than only 0 and 1, I mean 0 V and 5 V. This would be nice but would require an expensive DAC.

Indeed, there isn't a DAC on Arduino boards.

## The pulse-width modulation concept

The **pulse-width modulation** is a very common technique used to mimic analog output behavior.

Let's put that another way.

Our digital outputs can only be at 0 V or 5 V. But at a particular time-interval, if we switch them on/off quickly, then we can calculate a mean value depending on the time passed at 0 V or 5 V. This mean can easily be used as a value.

Check the following schematic to know know more about the concept of duty cycle:

![The pulse-width modulation concept](img/7584_08_025.jpg)

The concept of duty cycle and PWM

The mean of the time spent at 5V defines the duty cycle. This value is the mean time when the pin is at 5V and is given as a percentage.

`analogWrite()` is a special function that can generate a steady square wave at a specific duty cycle until the next call.

According to the Arduino core documentation, the PWM signal pulses at a frequency of 490 Hz. I didn't (yet) verify this, but it would really only be possible with an oscilloscope, for instance.

### Note

Be careful: PWM isn't available on every pin of your board!

For instance, Arduino Uno and Leonardo provide PWM on digital pins numbers 3, 5, 6, 9, 10, and 11.

You have to know this before trying anything.

## Dimming an LED

Let's check a basic circuit in order to test PWM:

![Dimming an LED](img/7584_08_026.jpg)

Let's look at the circuit diagram, even if it's obvious:

![Dimming an LED](img/7584_08_027.jpg)

We'll use the Fading example by David A. Mellis and modified by Tom Igoe. Check it in **File** | **examples** | **03.Analog** | **Fading**. We are going to change the `ledPin` value from `9` to `11` to fit our circuit.

Here it is, modified:

[PRE15]

Upload it, test it, and love it!

### A higher resolution PWM driver component

Of course, there are components providing higher resolutions of PWM. Here, with native Arduino boards, we have an 8-bit resolution (256 values). I wanted to point out to you the Texas Instrument, TLC5940\. You can find its datasheet here: [http://www.ti.com/lit/ds/symlink/tlc5940.pdf](http://www.ti.com/lit/ds/symlink/tlc5940.pdf).

![A higher resolution PWM driver component](img/7584_08_028.jpg)

TLC5950, the 16-channel LED driver that provides PWM control

Be careful, it is a constant-current sink driver. This means that it sinks the current and does not feed the current. For instance, you'd have to connect cathodes of your LEDs to the `OUT0` and `OUT15` pins, not anodes. If you want to use a specific driver like that, you won't use `analogWrite()`, of course. Why? Because this driver works as a shift register, wired through a serial connection with our Arduino.

I'd suggest using a nice library named tlc5940arduino, and available on Google code at

[http://code.google.com/p/tlc5940arduino/](http://code.google.com/p/tlc5940arduino/)

We'll see, in the third part of this book, how to write messages on LED matrices. But, there is also a nice way to use highest resolution displays: LCD.

# Quick introduction to LCD

**LCD** means **Liquid Crystal Display**. We use LCD technology everyday in watches, digicode display, and so on. Look around you, and check these small or great LCDs.

There exist two big families of LCD displays:

*   Character LCD is based on a matrix of characters (columns x rows)
*   Graphical LCD , is based on a pixel matrix

We can find a lot of printed circuit boards that include an LCD and the connectors to interface them with Arduino and other systems for cheap, nowadays.

There is now a library included in the Arduino Core that is really easy to use. Its name is **LiquidCrystal**, and it works with all LCD displays that are compatible with the Hitachi HD44780 driver. This driver is really common.

Hitachi developed it as a very dedicated driver, that includes a micro-controller itself, specifically to drive alphanumeric characters LCDs and to connect to the external world easily too, which can be done by a specific link using, usually, 16 connectors, including power supply for the external circuit itself and the backlight supply too:

![Quick introduction to LCD](img/7584_08_029.jpg)

A 16 x 2 character LCD

We are going to wire it and display some messages on it.

## HD44780-compatible LCD display circuit

Here is the basic circuit of the HD44780-compatible LCD display:

![HD44780-compatible LCD display circuit](img/7584_08_030.jpg)

A 16 x 2 character LCD wired to Arduino and a potentiometer controlling its contrast

The corresponding circuit diagram is as follows:

![HD44780-compatible LCD display circuit](img/7584_08_031.jpg)

Circuit diagram of the character LCD, the potentiometer, and the Arduino board

LED+ and LED- aren't necessary as far as you have sufficient light. Using the potentiometer, you can also set the contrast of the LCD in order to have enough readability.

By the way, LED+ and LED- are, respectively, backlight anode and backlight cathode of the internal LED used for the backlight. You can drive these from Arduino, but it can lead to more consumption. Please read the LCD instructions and datasheet carefully.

## Displaying some random messages

Here is some neat firmware. You can find it in the `Chapter08/basicLCD/` folder:

[PRE16]

First, we have to include the `LiquidCrystal` library. Then, we define two variables:

*   `manyMessages` is an array of String for message storage
*   `counter` is a variable used for time tracing

Then, we initialize the `LiquidCrystal` library by passing some variables to its constructor, corresponding to each pin used to wired the LCD to the Arduino. Of course, the order of pins matters. It is: `rs`, `enable`, `d4`, `d5`, `d6`, and `d7.`

In the `setup()`, we define the size of the LCD according to the hardware, here that would be 16 columns and two rows.

Then, we statically store some messages in each element of the String array.

In the `loop()` block, we first place the cursor to the first place of the LCD.

We test the expression `(millis() – counter > 5000)` , and if it is true, we clear the whole LCD. Then, I'm printing a message defined by chance. Indeed, `random(4)` produces a pseudo-random number between 0 and 3 , and that index being random, we print a random message to the LCD from among the four defined in `setup()` to the LCD, on the first row.

Then, we store the current time in order to be able to measure the time passed since the last random message was displayed.

Then, we put the cursor at the first column of the second row, then, we print a String composed by constant and variable parts displaying the time in milliseconds since the last reset of the Arduino board.

# Summary

In this long chapter, we learned to deal with many things, including monochromatic LEDs to RGB LEDs, using shift registers and transistor arrays, and even introduce the LCD display. We dug a bit deeper into displaying visual feedbacks from the Arduino without necessarily using a computer.

In many cases of real life design, we can find projects using Arduino boards totally standalone and, without a computer. Using special libraries and specific components, we now know that we can make our Arduino feeling, expressing, and reacting.

In the following chapter, we are going to explain and dig into some other concepts, such as making Arduino move and eventually generating sounds too.