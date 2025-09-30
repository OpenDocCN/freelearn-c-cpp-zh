# *Chapter 6*: Morse Code SOS Visual Alarm with a Bright LED

This chapter describes how to build a very noticeable visual alarm with a **super-bright LED** connected to a microcontroller board. The LED will show an SOS message (this is used when someone or a group of people is in danger or in distress). In this chapter, you will learn how to control a super-bright LED with a microcontroller board. The reason we are using a super-bright LED in this chapter is to increase the visibility of the SOS message with it. This chapter will be beneficial for future electronic projects because you will learn how to control an LED with a transistor working as a switch. The application made in this chapter can be used by people in distress while they are hiking, at sea, and in similar scenarios.

In this chapter, we are going to cover the following main topics:

*   Understanding Morse code and the SOS message
*   Introducing super-bright LEDs and calculating their necessary resistors
*   Connecting the resistor and the super-bright LED to the microcontroller board
*   Coding the SOS Morse code signal
*   Testing the visual alarm

By the end of this chapter, you will be able to properly connect a super-bright LED to the Curiosity Nano and Blue Pill microcontroller boards and to generate a Morse code message with a microcontroller board.

# Technical requirements

The software tools that you will be using in this chapter are the MPLAB-X and Arduino IDEs for editing and uploading your programs to the Curiosity Nano and the Blue Pill microcontroller boards, respectively.

The code used in this chapter can be found at the book's GitHub repository:

[https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter06](https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter06)

The Code in Action video for this chapter can be found here: [https://bit.ly/3iXDlEP](https://bit.ly/3iXDlEP)

In this chapter, we will use the following pieces of hardware:

*   A solderless breadboard.
*   The Blue Pill and Curiosity Nano microcontroller boards.
*   A micro USB cable for connecting your microcontroller boards to a computer.
*   The ST-Link/V2 electronic interface needed to upload the compiled code to the Blue Pill. Bear in mind that the ST-Link/V2 requires four female-to-female DuPont wires.
*   One 5 mm trough-hole 625 nm orange-red, super-bright LED, manufacturer number BL-BJU334V-1, made by American Bright Optoelectronics Corporation, or something similar.
*   Two 1 k ohm resistors rated at one-quarter watt. These resistors are for the LED and the transistor.
*   One 220-ohm resistor rated at one-quarter watt. This is for the LED.
*   One **2N2222** transistor, TO-92 package.
*   A dozen male-to-male DuPont wires.

The following section describes what Morse code is and why we are using it in this project.

# Understanding Morse code and the SOS message

**Morse code** is a telecommunication technique used for encoding, receiving, and sending **alphanumeric** and **special characters** by applying signal sequences with different duration. Morse code is named after Samuel Morse, a telegraph inventor. This code is important because it was commonly used for radio and wired communication over long distances, in particular, for sending and receiving telegrams. Nowadays, Morse code is still used in amateur (*ham*) radio communications because it can be reliably decoded by people when electromagnetic atmospheric conditions are unfavorable. More importantly, Morse code can be used in an emergency by sending SOS messages in the form of light, audio, or electromagnetic signals. Morse code is still sometimes used in aviation as a radio navigation aid.

Each character in Morse code is made with `-` symbol) and `.` symbol). A dot is one signal unit, and a dash is three signal units. Each alphabet letter, numerals, and special characters are encoded using a combination of dots and/or dashes. A space between letters is one signal unit (a dot), and a space between words is seven signal units. This code is generally transmitted on an information-carrying medium, for example, visible light and electromagnetic radio waves. The following is a list with the letters and numbers encoded using international Morse code:

*   A (`.-`)
*   B (`-...`)
*   C (`-.-.`)
*   D (`-..`)
*   E (`.`)
*   F (`..-.`)
*   G (`- -.`)
*   H (`....`)
*   I (`..`)
*   J (`.---`)
*   K (`-.-`)
*   L (`.-..`)
*   M (`- -`)
*   N (`-.`)
*   O (`- - -`)
*   P (`.- -.`)
*   Q (`- -.-`)
*   R (`.-.`)
*   S (`. . .`)
*   T (`-`)
*   U (`..-`)
*   V (`...-`)
*   W (`.- -`)
*   X (`-..-`)
*   Y (`-.--`)
*   Z (`- -..`)
*   1 (`.- - - -`)
*   2 (`..- - -`)
*   3 (`...- -`)
*   4 (`. . . . -`)
*   5 (`. . . . .`)
*   6 (`-. . . .`)
*   7 (`- - . . .`)
*   8 (`- - - . .`)
*   9 (`- - - -.`)
*   0 (`- - - - -`)

Special characters, such as `$` and `#` and other characters from languages other than English, have also been encoded with Morse code, but showing them is beyond the scope of this chapter. For example, this is how we encode the word PARIS in Morse code: .- -.   .-   .-.   ..   ...

This is another example. The word HELP can be codified as ….   .   .-..   .--.

Another longer example is the word ALIVE: .- .-.. .. ...- .

A commonly used distress message encoded with Morse code is made up of the letters SOS. It is formed by three dots, three dashes, and three dots (`. . .  ---  . . .`). This message has been internationally applied and recognized by treaties, originally used for maritime emergency situations. Its origin is uncertain, but popular usage associates SOS with phrases such as *Save Our Ship* or *Save Our Souls*. Nonetheless, SOS is easy to remember and used in an emergency and is shorter than coding other words, such as *HELP*, in Morse code.

In this chapter, we will use the SOS message to make a visual alarm, showing that message making the dots and dashes by turning on and off a super-bright LED with the Blue Pill and the Curiosity Nano boards.

The next section describes a brief introduction to super-bright LEDs, and what type of resistor can be connected to them for our Morse code purposes.

# Introducing super-bright LEDs and calculating their necessary resistors

A **super-bright LED** is a **light-emitting diode** (**LED**) that glows with high intensity, higher than regular LEDs. LED brightness (light intensity) is calculated in **millicandelas** (**mcd**). Bear in mind that 1,000 mcd equals 1 candela. Candelas typically measure how much light is generated at the light source, in this case, an LED, but candelas can be used to measure other light sources, such as light bulbs. The super-bright LED that we use in this chapter is rated as 6,000 mcd, emitting a nice and powerful orange glow, which is quite bright when connected to a proper current-limiting resistor. In comparison, typical LEDs are rated at a range of about 50 to 200 mcd.

Super-bright LEDs have a special design to increase light diffusion by using a transparent glass coating and reflective material. However, some super-bright LEDs have a reduced **viewing angle** (the observation angle with the LED light looks more intense) of about 35 degrees, such as the one we are using in this chapter, whereas the viewing angle of other regular and super-bright LEDs is 120%. This viewing angle depends on their cost, efficiency, and applications.

As with regular LEDs, super-bright LEDs require a certain voltage to power them, typically between 2 and 3 volts. That's why we need to connect a current-limiting resistor to an LED to reduce its voltage. We can use the formula R=(Vs-Vf)/If for calculating a current-limiting resistor for an LED in the following cases:

*   **Vs** = supplied voltage. The Blue Pill and Curiosity Nano output ports provide 3.3 V.
*   **Vf** = forward voltage, which is the voltage that drops through a resistor.
*   **If** is the forward **amperage** (**amps**).
*   **R** is the resistor value that we want to calculate.

A commonly used resistor value for connecting to LEDs is 1 k ohm. The drop voltage with that resistor is 1.8 V when applying a supply voltage of 3.3 volts and a current drawing 1.5 milliamps (or 0.0015 amps). Let's apply the preceding formula to confirm this resistor value:

![](img/B16413_06_001.png)

The resistor connected to a super-bright LED will determine the number of amps it will draw. Typical resistor values that are used for connecting to LEDs are 220, 330, and 1 k ohms when using either 3.3 V or 5 V as the supply voltage that many microcontroller boards supply. The resistor value is not critical in the majority of LED applications (unless you are connecting an LED to a Blue Pill microcontroller board, as you will see in the next section), so we can use other resistors with similar values. First of all, you can determine whether the super-bright LED works. Go to the *Testing the visual alarm* section at the end of this chapter.

Important note

Do not stare at a super-bright LED directly on top of it when it is turned on (glowing), as this may hurt your eyes. You can momentarily look at it from the side.

*Figure 6.1* shows the super-bright LED used in this chapter:

![Figure 6.1 – The BL-BJU334V-1 super-bright LED](img/Figure_6.1_B16413.jpg)

Figure 6.1 – The BL-BJU334V-1 super-bright LED

*Figure 6.1* depicts a BL-BJU334V-1 5 mm through-hole LED that emits bright orange light in the 625 nm wavelength. Its left pin is the anode (positive lead), and the right pin is the cathode (negative lead) which is the short one. There are other types of super-bright LEDs with higher brightness and sizes. We decided to use this one in particular for this chapter because is low cost and suitable for inserting it in a solderless breadboard and connecting it to a microcontroller board. The next section deals with the connection of a super-bright LED to the Blue Pill and Curiosity Nano boards.

# Connecting the resistor and the super-bright LED to the microcontroller board

This section shows how to use a super-bright LED connected to a microcontroller board to display a Morse code message. We begin by explaining how to connect a super-bright LED to one of the input ports of the Blue Pill and how to use a transistor as a switch to control the super-bright LED. Then, we describe how to connect the super-bright LED to the Curiosity Nano board.

*Figure 6.2* shows a Fritzing diagram containing a super-bright LED:

![Figure 6.2 – A super-bright LED connected to a Blue Pill's I/O port](img/Figure_6.2_B16413.jpg)

Figure 6.2 – A super-bright LED connected to a Blue Pill's I/O port

As you can see from *Figure 6.2*, the super-bright LED's anode is connected to a 1 k ohm current-limiting resistor. The resistor is connected to output port B12, providing 3.3 V to it every time a dot or dash from a Morse code character is sent to it. The following are the steps for connecting all the components shown in *Figure 6.2*:

1.  Connect the 1 k ohm resistor to the Blue Pill's B12 port.
2.  Connect the resistor to the super-bright LED's anode pin (its longest leg, which is the left one shown in *Figure 6.1*).
3.  Connect the super-bright LED's cathode pin (its shortest leg) to the lower solderless breadboard's row ground.
4.  Connect the Blue Pill's GND pin to the solderless breadboard's lower rail.

When doing those steps, be sure to connect the right LED polarity.

Important note

Do *not* connect a super-bright LED directly to the power supply, as you will damage it. You should connect a current-limiting resistor to its anode (the long LED leg).

Each Blue Pill I/O port can handle up to 6 mA (milliamps). Be careful, because in some cases and configurations, a super bright LED could consume more than 6 mA of current. That's why we connected a 1 k ohm resistor to its anode to limit the current (and its voltage) arriving at the LED. If anything connected to a Blue Pill port is drawing more than 6 mA, you will damage the Blue Pill board.

Before we connect everything to a Blue Pill microcontroller board port, we need to know how many amps in the I/O **B12** port are consumed by the LED to see whether it is below the maximum amps that the port can handle. The amps (current) consumed by the LED from *Figure 6.2* is calculated using the following formula:

![](img/B16413_06_002.png)

The following are the descriptions for the symbols:

*   Vs = 3.3 V, which is the supplied voltage given by the output port generated when making the Morse code's dots and dashes.
*   Vf = 1.8 V, which is the forward voltage that drops across the 1 k resistor (we measured it with a multimeter).
*   R is the resistor value (1 k ohm).

Thus, ![](img/B16413_06_003.png), or 1.5 milliamps, well below the maximum 6 mA that each Blue Pill port can handle, so we can safely use a 1 k ohm resistor for our super-bright LED if we connect it to a port that produces 3.3 V.

It is important to note that the resistor value and the voltage arriving at the resistor from the port will determine the LED brightness. We could use a resistor with a lower value for making the LED glow brighter. For example, we could use a 220-ohm resistor. Its dropping voltage is 1.9 V (we measured it with a multimeter). Let's calculate its amperage (amps) using Vs = 3.3 V: I=(Vs-Vf)/R=(3.3-1.9)/220=6 mA, which is the limit amps that a Blue Pill port can handle. If we use a 220-ohm (or a lower value) resistor for connecting it to a super-bright LED, we should use a transistor working as a switch for protecting the Blue Pill port by not drawing current directly from it. *Figure 6.3* shows a Fritzing diagram with a 2N2222 transistor. It will close its *switch*, that is, connect its collector and emitter internally, when it receives a voltage to its base (transistor pin number 2), and thus connect the LED's cathode to the ground:

![Figure 6.3 – The super-bright LED switched on/off by the 2N2222 transistor](img/Figure_6.3_B16413.jpg)

Figure 6.3 – The super-bright LED switched on/off by the 2N2222 transistor

As you can see from *Figure 6.3*, the LED is connected to a 220-ohm resistor that is connected to the Blue Pill's 3.3 V pin. This resistor draws the current from the Blue Pill's 3.3 pin, which handles up to 300 mA, and not from the B12 port handling up to 6 mA. Thus, the transistor controls the power flow to another part of the circuit switching the LED's ground. Every time the transistor receives a certain voltage at its base, it saturates and creates the binary on/off switch effect between its collector and emitter pins. We need to connect a resistor to its base to reduce its voltage and thus properly saturate it. A typical value for saturating the transistor used in this chapter is 1 k ohm when applying 3.3 V to it. The low-cost and popular transistor used in *Figure 6.3* has the part number 2N2222 (in a TO-92 package). This transistor handles up to 600 mA, enough to drive our super-bright LED. The following are the steps for connecting all the components shown in *Figure 6.3*:

1.  Connect the Blue Pill's GND pin to the breadboard's upper row.
2.  Connect the 1 k ohm resistor to the Blue Pill's B12 port.
3.  Connect the 1 k ohm resistor to the 2N2222 transistor's base (pin number **2**).
4.  Connect the 2N2222 transistor's emitter (pin number **1**) to the ground (the breadboard's upper row).
5.  Connect the LED's cathode pin (its shortest leg) to the 2N2222's collector (pin number **3**).
6.  Connect the 220-ohm resistor to the LED's anode pin (its longest leg).
7.  Connect the 220-ohm resistor to Blue Pill's 3V3 (3.3) pin.

*Figure 6.4* shows the 2N2222 pinout:

![Figure 6.4 – The 2N2222 transistor pin numbers](img/Figure_6.4_B16413.jpg)

Figure 6.4 – The 2N2222 transistor pin numbers

*Figure 6.4* shows a 2N2222 transistor in its TO-92 package, showing its three pins. Pin 1 is its emitter, pin 2 is its base, and pin 3 is its collector. The N shown on the transistor means that the 2N2222 is an NPN-type transistor, an internal configuration composed of negative-positive-negative layers. That big N letter is not actually shown on commercial transistors; it appears only on the Fritzing diagram shown in *Figure 6.4*. Instead, commercial transistors show their part number on them.

*Figure 6.5* shows the electronic symbol for the transistor:

![Figure 6.5 – The transistor's electronic diagram (NPN type)](img/B16413_Figure_6.5.jpg)

Figure 6.5 – The transistor's electronic diagram (NPN type)

As you can see from *Figure 6.5*, the symbol shows the 2N2222's pin numbers. This pin order will change in other types of transistors.

*Figure 6.6* shows how everything is connected:

![Figure 6.6 – Connecting the super-bright LED to a 2N2222 transistor](img/Figure_6.6_B16413.jpg)

Figure 6.6 – Connecting the super-bright LED to a 2N2222 transistor

*Figure 6.6* shows the 2N2222 transistor connected to the BluePill's ground and 3.3 (3.3 V) pins. The transistor is connected at the left of the super-bright LED. The next section deals with connecting the super-bright LED to the Curiosity Nano board using a resistor and a transistor.

## Connecting the super-bright LED to the Curiosity Nano

This section explains how to control the super-bright LED from the Curiosity Nano board to display the SOS Morse code message.

*Figure 6.7* is a Fritzing diagram showing how to connect a super-bright LED to the Curiosity Nano:

![Figure 6.7 – A super-bright LED connected to a Curiosity Nano's I/O port](img/Figure_6.7_B16413.jpg)

Figure 6.7 – A super-bright LED connected to a Curiosity Nano's I/O port

As you can see in *Figure 6.7*, we use a 1 k ohm resistor to connect the super-bright LED, which is a similar circuit for the Blue Pill microcontroller board explained in a previous section. Here are the steps for connecting the components shown in *Figure 6.7*:

1.  Connect the Curiosity Nano's RD3 pin to a 1 k ohm resistor.
2.  Connect the resistor to the super-bright LED's anode pin (its longest leg).
3.  Connect the super-bright LED's cathode pin (its shortest leg) to the Curiosity Nano's GND pin.

Be aware that in theory, each of the Curiosity Nano I/O ports can handle up to 12.8 milliamps (500 milliamps/39 ports), and the Curiosity Nano's voltage regulator (VBUS) can handle up to 500 milliamps. This may change a bit with the ambient temperature, according to its manufacturer. We could make the LED glow brighter by connecting a lower-value resistor, or we could connect more than one LED to a port, but this will draw more current, potentially damaging the microcontroller board. Since each port cannot support a lot of current, we need to connect a transistor to switch the LED's ground. *Figure 6.8* shows how all the components are connected:

![Figure 6.8 – The super-bright LED switched on/off by the transistor, connected to the Curiosity Nano board](img/Figure_6.8_B16413.jpg)

Figure 6.8 – The super-bright LED switched on/off by the transistor, connected to the Curiosity Nano board

As you can see in *Figure 6.8*, the resistor connected to the LED's anode pin is connected to the Curiosity Nano's VTG port that provides 3.3 V. These are the steps for connecting all the components:

1.  Insert the Curiosity Nano, the 2N2222 transistor, and the super-bright LED to the solderless breadboard.
2.  Connect the transistor's pin 1 to the Curiosity Nano's ground (GND) pin.
3.  Connect the 1 k ohm resistor to Curiosity Nano's RD3 port.
4.  Connect the 1 k ohm resistor to the transistor's pin 2\.
5.  Connect the super-bright LED's cathode (its shortest pin) to the transistor's pin 3\.
6.  Connect the 220-ohm resistor to the super-bright LED's anode (its longest pin).
7.  Connect the 220-ohm resistor to Curiosity Nano's VTG pin.

*Figure 6.9* shows how everything is connected:

![Figure 6.9 – The transistor and the LED connected to the Curiosity Nano](img/Figure_6.9_B16413.jpg)

Figure 6.9 – The transistor and the LED connected to the Curiosity Nano

As you can see from *Figure 6.9*, the 2N2222 transistor is switching the super-bright LED's ground.

The next section describes how the SOS signal can be programmed on the Blue Pill and the Curiosity Nano boards.

# Coding the SOS Morse code signal

This section describes the code necessary for turning the LED for showing the SOS Morse signal, which runs on the Blue Pill board, on and off. The following code shows the main functions used for defining the SOS Morse code message and for sending it to the board's output port. The next code segment defines the necessary dot, dash, and space values, as well as the port label:

```cpp
int led=PB12;
int dot_duration=150; 
int dash_duration=dot_duration*3; 
int shortspace_duration=dot_duration; 
int space_duration=dot_duration*7;
```

The next function sets up the output port (`B12`) for turning the LED on and off:

```cpp
void setup() {
       pinMode (led,OUTPUT);
} 
```

These functions define the letter S and O to be used in the SOS message:

```cpp
void S() {
          dot();
          dot();
          dot();
          shortspace();
}

void O() {
          dash();
          dash();
          dash();
          shortspace();
}
```

The following functions define the time spaces in between the letters and the space in between each SOS message sent to the output port:

```cpp
void shortspace() {
          delay(shortspace_duration);
} 
void space() {
          delay (space_duration);
} 
void dot() {
          digitalWrite(led,HIGH); 
          delay (dot_duration); 
          digitalWrite(led,LOW); 
          delay(dot_duration); 
}

void dash() {
          digitalWrite(led,HIGH); 
          delay(dash_duration); 
          digitalWrite(led,LOW); 
          delay(dash_duration); 
}
```

The following is the main function, which defines the SOS message and its leading time space:

```cpp
void loop() {
          S(); O(); S();  
          space();  
}
```

The preceding code shows that the Blue Pill's `B12` port is used as an output for turning the super-bright LED on and off. In this example code, each dot has a duration of 150 milliseconds (stored in the `dot_duration` variable), and each dash has a duration of three times that of each dot. There is a short space of time in between each letter, made by the `shortspace()` function. Also, there is a time space in between each SOS word, made by the `space()` function. The letters S and O from the SOS Morse message are encoded by the functions `S()` and `O()`, respectively.

Bear in mind that the code uploaded to the GitHub repository has many comments explaining most of the instructions from the preceding code.

Note

You can run the preceding code on an Arduino microcontroller board. Just change the output port number `PB12` to any Arduino digital port. For example, the Arduino Uno microcontroller board has digital ports `2` to `13`.

The following section shows how to code the SOS message example on the Curiosity Nano board.

## The SOS message code for the Curiosity Nano

The SOS message code for the Curiosity Nano microcontroller board is similar to the one that runs on the Blue Pill board. The next code segment defines the necessary dot, dash, and space values:

```cpp
#include "mcc_generated_files/mcc.h"
const int dot_duration=150; 
const int dash_duration=dot_duration*3; 
const int shortspace_duration=150; 
const int space_duration=dot_duration*7;         
```

The following functions define the time spaces in between the letters and the space in between each SOS message sent to the output port:

```cpp
void shortspace() {
          __delay_ms(shortspace_duration);
} 
void space() {
         __delay_ms(space_duration);
} 
void dot() { 
          IO_RD3_SetHigh(); 
          __delay_ms(dot_duration); 
          IO_RD3_SetLow(); 
          __delay_ms(dot_duration); 
}        
void dash() { 
          IO_RD3_SetHigh();  
          __delay_ms(dash_duration); 
          IO_RD3_SetLow();
          __delay_ms(dash_duration); 
}
```

These functions define the letters S and O to be used in the SOS message:

```cpp
void S() { 
          dot();
          dot();
          dot();
          shortspace();
}
void O() { 
          dash();
          dash();
          dash();
          shortspace();
}
```

The following is the `main` function, which defines the SOS message and its leading time space:

```cpp
void main(void)
{
    SYSTEM_Initialize();
    IO_RD3_SetLow(); 
    while (1) 
    {
      S(); O(); S(); 
      space();
    }
}
```

As you can see from the preceding code, `RD3` is used as an output port for driving the 2N2222 transistor and therefore switching the LED on and off.

Bear in mind that the code uploaded to the GitHub repository contains many comments explaining its main parts.

The next section describes how can we test out a super-bright LED to see whether it works correctly, as well as testing the speed of the SOS message shown by the LED.

# Testing the visual alarm

In this section, we will focus on how to test the super-bright LED, as well as how to test the speed of the SOS message shown by the LED.

You can test the super-bright LED to establish whether it works OK with a power supply, as shown in *Figure 6.10*:

![Figure 6.10 – Connecting the super-bright LED to a battery set](img/Figure_6.10_B16413.jpg)

Figure 6.10 – Connecting the super-bright LED to a battery set

The following are the steps for connecting everything according to *Figure 6.10*:

1.  Connect the super-bright LED's anode (its longest leg) to a 1 k ohm resistor.
2.  Connect the resistor to the battery set's positive terminal. The battery set should provide around 3 V (supplied by two AA batteries), which are enough for testing our super-bright LED.
3.  Connect the battery set's negative terminal to the super-bright LED's cathode (its shortest leg).

After connecting everything as per the preceding steps, the LED should glow. If not, check the LED polarity, and whether the batteries have enough energy. Also check the connections. If the LED doesn't glow, it doesn't work and needs to be discarded.

If you compare the brightness of a regular LED against the super-bright LED that we used in this chapter, you will notice that the regular LED glows more uniformly and more **omnidirectional**, whereas the super-bright LED has a viewing angle of 35 degrees, meaning that it will be brighter from its top rather than from its sides. Try it! Observe very briefly both LEDs from the side. Remember that you shouldn't stare at the super-bright LED perpendicularly (top view), as it may hurt your eyes.

You can also test the speed of the SOS message. Change the value originally declared in the `dot_duration` variable in the preceding code for both the Blue Pill and Curiosity Nano. A smaller value will make the SOS message glow faster on the super-bright LED.

# Summary

In this chapter, we learned what a super-bright LED is and how we can connect it to a microcontroller board. We also reviewed how to use the super-bright LED as a powerful visual alarm, since it glows much more intensely than conventional LEDs. We also summarized what Morse code is, how it is used worldwide, and how to show the SOS Morse code message, which is used in visual alarms as a distress message, by turning on and off the super-bright LED connected to the Blue Pill and Curiosity Nano microcontroller boards. Connecting a super-bright LED is not straightforward, since we will need to know how much current it will draw, because the microcontroller boards' output ports can handle a very limited amount of current, in the order of a few milliamps. This chapter will be beneficial for readers who would like to control an LED in other electronic projects. It also points out the importance of carefully calculating the amps that are drawn by a super-bright LED and using the right resistor or transistor to connect to it, so as to avoid damaging the microcontroller board.

The next chapter will focus on using a small microphone connected to a microcontroller board to detect two clapping sounds in a row to activate a process on a microcontroller board.

# Further reading

*   Choudhuri, K. B. R. (2017). *Learn Arduino Prototyping in 10 days*. Birmingham, UK: Packt Publishing Ltd.
*   Gay, W. (2018). *Beginning STM32: Developing with FreeRTOS, libopencm3, and GCC*. New York, NY: Apress.
*   Horowitz, P., Hill, W. (2015). *The Art of Electronics*. [3rd ed.] Cambridge University Press: New York, NY.
*   Microchip (2019). *PIC16F15376 Curiosity Nano Hardware User Guide*. Microchip Technology, Inc. Available from: [http://ww1.microchip.com/downloads/en/DeviceDoc/50002900B.pdf](http://ww1.microchip.com/downloads/en/DeviceDoc/50002900B.pdf).
*   2N2222 (2013). P2N2222A transistor datasheet. Semiconductor Component Industries, LLC. Available from:

    [https://www.onsemi.com/pub/Collateral/P2N2222A-D.PDF](https://www.onsemi.com/pub/Collateral/P2N2222A-D.PDF).

*   LED (n.d.) BL-BJ33V4V-1 super-bright LED datasheet. Bright LED Electronics Corp. Available from: http://www.maxim4u.com/download.php?id=1304920&pdfid=446ED6935162B290D3BC0AF8E0E068B8&file=0168\bl-bj33v4v-1_4138472.pdf.