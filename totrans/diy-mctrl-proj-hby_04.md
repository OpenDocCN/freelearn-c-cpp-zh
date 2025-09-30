# *Chapter 4*: Measuring the Amount of Light with a Photoresistor

This chapter focuses on how to connect a **photoresistor**, an electronic component that measures the amount of light from the environment, to an input port of both the Blue Pill and Curiosity Nano microcontroller boards. In this chapter's exercise, we will analyze with a photoresistor whether a plant receives enough light.

In this chapter, we are going to cover the following main topics:

*   Understanding sensors
*   Introducing photoresistors
*   Connecting a photoresistor to a microcontroller board port
*   Coding the photoresistor values and setting up ports
*   Testing the photoresistor

By the end of this chapter, you will have learned how to connect an analog sensor to a microcontroller board, and how to analyze analog data obtained from a photoresistor. The knowledge and experience learned in this chapter will be useful in other chapters that require the use of a sensor.

# Technical requirements

The software tools that you will be using in this chapter are the Arduino IDE and the MPLAB-X for editing and uploading your programs to the Blue Pill and the Curiosity Nano boards, respectively.

The code used in this chapter can be found in the book's GitHub repository here:

[https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter04](https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter04)

The Code in Action video for this chapter can be found here: [https://bit.ly/3gNY4bt](https://bit.ly/3gNY4bt)

In this chapter, we will also use the following pieces of hardware:

*   A solderless breadboard.
*   The Blue Pill and Curiosity Nano microcontroller boards.
*   A micro-USB cable for connecting your microcontroller boards to a computer.
*   The ST-Link/V2 electronic interface needed for uploading the compiled code to the Blue Pill. Bear in mind that the ST-Link/V2 requires four female-to-female DuPont wires.
*   A green, yellow, and red LED.
*   Three 220-ohm resistors rated at one-quarter watt. These resistors are for the 3 LEDs.
*   One 220-ohm resistor rated at one-quarter watt for the photoresistor connected to the Curiosity Nano.
*   One 10 K ohm resistor, rated at one-quarter watt.
*   A **photoresistor sensor module**.
*   Three male-to-female DuPont wires for connecting the sensor module to the solderless breadboard.
*   A dozen male-to-male DuPont wires for connecting the resistor and the push button to the breadboard and the microcontroller boards.

The next section describes a brief introduction to photoresistors, their electrical characteristics, and how they are used in electronic projects.

# Understanding sensors

In this section, we explain what sensors are and the importance of sensor modules. It is important to understand first what sensors are and what their purpose is before you use them in practical microcontroller board applications, for example, using photoresistors, which are a useful type of sensor. This information about sensors is essential for further sections.

## Defining sensors

A **sensor** is an electronic component, device, or module that measures physical input from an environment or a particular situation (for example, a sensor measuring water temperature in a fish tank). Sensors are useful for detecting changes in physical variables, such as humidity, temperature, light, vibrations, and movement, among others. Those physical variations are manifested in the sensors by changing their electric/electronic properties, such as changes in resistance and conductivity in the sensors.

There are different types of sensors with different applications. For example, motion sensors can detect human or pet movements when they pass across the sensor's field of view. If this sensor detects motion, it sends a signal to a computer or microcontroller board and then the board should do something about it, such as open an automatic door, set off an alarm, turn on a lightbulb, and so on. Other types of sensors include infrared, ultrasonic, temperature, pressure, and touch sensors.

## What are sensor modules?

A sensor can be part of a small electronic circuit called a **sensor module**. It contains other electronic components besides a sensor, such as resistors, transistors, LEDs, and integrated circuits, among others. The purpose of those extra components in a sensor module is to support and facilitate the reading, analysis, and transmission of signals coming from a sensor. Some sensor modules convert analog to digital data. Analog data are voltage variations, that is, analog to physical variables from an environment. For example, 0.5 volts obtained from a temperature sensor could be equivalent to 25 degrees Celsius. Digital data coming from a sensor module can contain either logical-level *HIGH* or logical-level *LOW* values.

Note

Voltages representing digital logic levels change depending on the microcontroller board that you are using. The I/O ports from both the Blue Pill and Curiosity Nano microcontroller boards use 3.3 volts, which is equivalent to logical-level HIGH. Logical-level LOW is equivalent to 0 volts.

An analog signal from a sensor is sent electronically via a wire or a wireless communication medium (for example, Bluetooth) to a microcontroller board, which will process it, and then do something about it, as shown in *Figure 4.1*:

![Figure 4.1 – A temperature sensor connected to the microcontroller board's port](img/Figure_4.1_B16413.jpg)

Figure 4.1 – A temperature sensor connected to the microcontroller board's port

The data coming from the sensor (displayed in the diagram from *Figure 4.1*) is read by the microcontroller board's input port. This data can be adapted and shown to be human-readable (for example, showing the temperature numerically in a display as degrees Celsius or Fahrenheit). The same happens with analog and/or digital signals coming from a sensor module. Its data is sent to a microcontroller board port or ports via a wired or wireless medium.

Note

Microcontroller boards' ports can be set up as either input or output ports via coding. Please refer to [*Chapter 2*](B16413_02_Final_NM_ePub.xhtml#_idTextAnchor029), *Software Setup and C Programming for Microcontroller Boards*, on how to program them.

The next section focuses on photoresistors, a commonly used type of sensor, describing their function, representation, and applications.

# Introducing photoresistors

This section introduces you to photoresistors, which are very useful for many applications, for example, for measuring the amount of light. In this chapter, we will define what photoresistors are, their classification, and how they are connected to an electronic circuit.

A **photoresistor** is an electronic component made with light-sensitive material, changing its resistance according to the amount of visible light that it detects. There are different types of photoresistors. Some of them detect **ultraviolet** (**UV**) light and others detect infrared light. The latter is used in TV sets, where its infrared light sensor receives data from a remote control. *Figure 4.2* shows a common photoresistor used in the examples of this chapter. The photoresistor used in this chapter detects regular light that humans can see. It does not detect infrared nor UV light.

![Figure 4.2 – A typical photoresistor](img/Figure_4.2_B16413.jpg)

Figure 4.2 – A typical photoresistor

From *Figure 4.2*, you can see that the photoresistor has two pins (also called legs). They are connected to an electronic circuit similar to regular resistors, so photoresistors do not have polarity. They work as a regular resistor that changes its resistance according to the amount of light it receives through its transparent cover. Here are some technical specifications of the photoresistor:

*   Size: 5mm diameter (width)
*   Range of resistance: 200 K ohms (dark) to typically 5 K to 10 K ohms and nearly 0 ohms when it receives full brightness.
*   Power supply: Up to 100V, using less than 1 milliamp of current on average, depending on the power supply voltage.

*Figure 4.3* shows an electrical diagram containing a photoresistor and how it can be connected to a microcontroller board.

![Figure 4.3 – Electrical diagram with a photoresistor](img/Figure_4.3_B16413.jpg)

Figure 4.3 – Electrical diagram with a photoresistor

As you can see in *Figure 4.3*, a photoresistor's electrical symbol is represented with a circle and some incoming arrows indicating that light rays are hitting the photoresistor's surface. The photoresistor is connected to the power supply to a **Ground** pin from a microcontroller board and the other pin is connected to a voltage pin (**Vcc**) and an **Input port** from a microcontroller board. That is one way to connect a photoresistor. The **10K ohm** resistor works as a pull-down resistor. This is because the input port where the photoresistor is connected to at some point will receive zero volts when the photoresistor is not receiving any light. Remember that we should not leave an input pin without connecting to anything, otherwise its state will be floating (inputting random voltages).

Note

The photoresistor should be connected to an analog input port from the microcontroller board since the port will receive a changing analog voltage from the photoresistor because the resistance presented at the photoresistor will change according to the amount of light received by the photoresistor.

The next section explains how to connect and use photoresistors in both the Blue Pill and Curiosity Nano boards to measure the amount of light in an environment.

# Connecting a photoresistor to a microcontroller board port

This section shows how to connect a photoresistor to the Blue Pill and the Curiosity Nano boards to *read* the amount of light from an environment (for example, a living room).

In this section, we also explain how to use three LEDs to indicate that a room is well illuminated by turning a green LED on, that the light is dim by turning a yellow LED on, or that it is dark by turning a red LED on. The electronic circuit with the photoresistor sensor can be placed close to a plant and the circuit can be useful to know whether a plant needs more light or not. The next section shows how to build the circuit with the Blue Pill board.

## Connecting a photoresistor to a Blue Pill board

The connection of a photoresistor sensor to a Blue Pill microcontroller board is simple. It can be directly connected to an input analog port, provided that you use a pull-down resistor. The circuit shown in *Figure 4.4* describes how to do it, and it is based on the electrical diagram from *Figure 4.3*.

![Figure 4.4 – The LEDs and photoresistor connected to a Blue Pill board](img/Figure_4.4_B16413.jpg)

Figure 4.4 – The LEDs and photoresistor connected to a Blue Pill board

As you can see in *Figure 4.4*, the three LEDs will show the amount of light detected by the photoresistor. The following are the steps for connecting the components to the Blue Pill board:

1.  Connect the Curiosity Nano's GND pins to the upper and lower rails, of the solderless breadboard, as shown in *Figure 4.4*.
2.  Connect one pin of the photoresistor to the microcontroller board's ground pin labeled **3V3**.
3.  Connect the other pin of the photoresistor to the Blue Pill's pin labeled **B1**. **B1** will be used as an analog input port.
4.  Connect a **10K ohm** resistor to the ground and to the photoresistor's leg that is connected to input port **B1** of the Blue Pill.
5.  Now, connect the resistors to the output ports **B12**, **B14**, and **B15** of the Blue Pill.
6.  As the last step, connect the green, yellow, and red LEDs' anodes to the three resistors and then connect the LEDs' cathodes to the ground.

The 3V3 pin from the Blue Pill provides 3.3 volts, which are enough for applying voltage (feeding) to the photoresistor.

Important note

Do not apply **5 volts** (**5V**) to the Blue Pill's input ports, because you may damage the Blue Pill. That's why you will connect the 3V3 voltage pin to the photoresistor, so it will provide up to 3.3 volts as an analog voltage output and the Blue Pill will use up that voltage to measure the amount of light. Remember that the voltage coming from the photoresistor will change according to its resistance.

We are connecting a **10K ohm** pull-down resistor to the Blue Pill's input port **B1**, as shown in *Figure 4.4*, forcing the port to have a 0-volt input when there's no voltage coming from the photoresistor. Remember that the input ports should receive some voltage or even 0 volts to avoid random voltages.

Note

You may want to try a different value for the pull-down 10K ohm resistor connected to the Blue Pill, shown in *Figure 4.4*, depending on the light level range that you would like to detect.

*Figure 4.5* shows how all the components are connected to the microcontroller board. This circuit is based on the Fritzing diagram shown in *Figure 4.4*.

![Figure 4.5 – Connecting the photoresistor and the LEDs to the Blue Pill](img/Figure_4.5_B16413.jpg)

Figure 4.5 – Connecting the photoresistor and the LEDs to the Blue Pill

As you can see in *Figure 4.5*, the photoresistor is connected to the Blue Pill's port B1\. The green LED is turning on, meaning that the amount of light that the plant is receiving should be OK. Remember that you will need to connect the ST-Link/V2 electronic interface to the Blue Pill in order to upload the program to it from the Arduino IDE, as explained in [*Chapter 1*](B16413_01_Final_NM_ePub.xhtml#_idTextAnchor014), *Introduction to Microcontrollers and Microcontroller Boards*.

In this section, you learned how to connect an easy-to-use photoresistor to the Blue Pill board and how to show its values through LEDs.

The next section describes how to code the photoresistor example shown in *Figure 4.4* and *Figure 4.5*.

# Coding the photoresistor values and setting up ports

This section shows how to code a Blue Pill application for reading data from a photoresistor.

The following code should run on the microcontroller board circuit shown in *Figure 4.4* and *Figure 4.5*:

```cpp
int photoresistorPin = PB1;
int photoresistorReading;
void setup(void) {
    pinMode(PB12, OUTPUT); 
    pinMode(PB14, OUTPUT); 
    pinMode(PB15, OUTPUT); 
}
void loop(void) {
    photoresistorReading = analogRead(photoresistorPin); 
    digitalWrite(PB12, LOW);
    digitalWrite(PB14, LOW);
    digitalWrite(PB15, LOW);
    if (photoresistorReading < 600) { 
        digitalWrite(PB15, HIGH);
    } else if (photoresistorReading < 1000) {  
        digitalWrite(PB14, HIGH);
    } else {
        digitalWrite(PB12, HIGH);  
  }
  delay(500);
}
```

As you can see from the code, one of its most important functions is this one: `analogRead(photoresistorPin);`. Internally, that function will perform an `pinMode()` function to declare the PB12, PB14, and PB15 ports as outputs.

The code can be downloaded from the book's GitHub repository. Its filename is `photoresistor_bluepill.ino`. The code from the repository contains useful comments explaining the functions and variables used in it.

Note

You can run the same code for the Blue Pill on Arduino microcontroller boards such as the Arduino Uno. Just change the Blue Pill's input port PB1 for the Arduino's analog port 0 (it is labeled as A0 on the Arduino Uno), and change the Blue Pill's output ports PB12, PB14, and PB15 for the Arduino digital ports 8, 9, and 10\. You may also need to change the values (thresholds) for the decisions from the code.

The following section describes how to connect the LEDs and a photoresistor to the Curiosity Nano microcontroller board, following the example from the Blue Pill board.

## Connecting a photoresistor to a Curiosity Nano board

Following the Blue Pill example from the previous section, we can show the amount of light in an environment by turning on LED lights using the Curiosity Nano microcontroller board, as depicted in *Figure 4.6*.

![Figure 4.6 – The LEDs and photoresistor connected to a Blue Pill board](img/Figure_4.6_B16413.jpg)

Figure 4.6 – The LEDs and photoresistor connected to a Blue Pill board

As it is shown in *Figure 4.6*, the three LEDs will be used to show the amount of light detected by the photoresistor. The following are the steps for connecting the components to the Curiosity Nano board:

1.  Connect the Curiosity Nano's GND pins to the upper and lower rails of the solderless breadboard, as shown in *Figure 4.6*.
2.  Connect one pin of the photoresistor to the Curiosity Nano's pin labeled VTG. This pin provides 3.3 volts.
3.  Connect the other pin of the photoresistor to the Curiosity Nano's pin labeled RA0\. RA0 will be used as an analog input port.
4.  Connect a 220-ohm resistor to the ground and to the photoresistor's leg that is connected to port RA0\. This will be a pull-down resistor.
5.  Now, connect the 220-ohm resistors to ports RD1, RD2, and RD3 and to the anodes of the 3 LEDs, respectively.
6.  As the last step, connect the LEDs' cathodes to the ground.

We are connecting a 220-ohm pull-down resistor to the Curiosity Nano's input port, as shown in *Figure 4.6*, forcing the port to have a 0-volt input when there's no voltage coming from the photoresistor. Remember that the input ports should receive some voltage or even 0 volts to avoid random voltages.

Note

You may want to try a different value for the pull-down 220-ohm resistor shown in *Figure 4.6*, depending on the light level range that you would like to detect.

The photoresistor from *Figure 4.6* will change its resistance according to the amount of light that it receives from the environment, thus the voltage that passes through it will change. These voltage differences will be read by a microcontroller board's input port. Our code running on the microcontroller board will then compare the analog voltage values and determine whether the light is very low, normal, or too bright by turning on the yellow, green, or red LEDs respectively.

The next code shows how to read the photoresistor values through the analog port RA0:

```cpp
#include <xc.h>
#include <stdio.h>
#include "mcc_generated_files/mcc.h"
static uint16_t reading_photoresistor=0;
void main(void)
{
    SYSTEM_Initialize(); 
    ADC_Initialize();
    while (1)
    {
        IO_RD1_SetLow();
        IO_RD2_SetLow();
        IO_RD3_SetLow();
        reading_photoresistor =            ADC_GetConversion(channel_ANA0); 
        if (reading_photoresistor>=0 &&                 reading_photoresistor <=128)
        {
            IO_RD1_SetHigh();
        } else if (reading_photoresistor>= 129 &&                    reading_photoresistor<=512)
        {
            IO_RD2_SetHigh();
        } else
        {
            IO_RD3_SetHigh();
        }
        __delay_ms(200); 
    } 
}
```

Please bear in mind that the code uploaded to the book's GitHub online repository has many comments explaining almost all its instructions. In the preceding code, one of the most important functions is `ADC_GetConversion(channel_ANA0);`, which reads the voltage changes from the photoresistor and performs the analog to digital conversion of those values. `channel_ana0` is a label given to port RA0.

The code can be downloaded from the book's GitHub repository. Its filename is `Chapter4_Curiosity_Nano_code_project.zip`. The code from the repository contains useful comments explaining the functions and variables used in it.

*Figure 4.7* shows how the photoresistor and the LEDs are connected to the Curiosity Nano's ports. The circuit shown in *Figure 4.7* is based on the Fritzing diagram shown in *Figure 4.6*.

![Figure 4.7 – Connecting the photoresistor and the LEDs to the Curiosity Nano](img/Figure_4.7_B16413.jpg)

Figure 4.7 – Connecting the photoresistor and the LEDs to the Curiosity Nano

As you can see in *Figure 4.7*, the photoresistor is connected to the Curiosity Nano's port RA0\. The green LED is turning on in the figure, meaning that the amount of light that the plant is receiving should be OK.

The next section shows how to obtain analog data about the amount of light from an environment using a photoresistor sensor module.

## Connecting a photoresistor sensor module to the microcontroller boards

This section explains how to use a photoresistor sensor module for measuring the amount of light from an environment. This module can be bought separately or as part of a sensors kit from many online sources. The photoresistor sensor module contains tiny electronic components that facilitate the connection to a microcontroller board and its photoresistor use. For example, the module that we are using in this section contains extra components such as resistors and a variable resistor that adjusts the threshold level of a digital *high* value, which is sent by the sensor module if it receives a certain amount of light. Of course, this digital value must be sent to a digital input port of the microcontroller board.

The next section shows how to connect the photoresistor sensor module to the Blue Pill.

### Connecting the photoresistor sensor module to the Blue Pill board

This section explains how to connect a photoresistor module to the microcontroller board using just three wires. The photoresistor sensor module shown in *Figure 4.8* has four connectors: **GND** (short for **ground**), **A0** (short for **analog output**), **D0** (short for **digital output**), and +5V. The order of these connectors may change if your sensor module is from a different brand, but their purpose is the same.

![ Figure 4.8 – A photoresistor sensor module connected to a Blue Pill board](img/Figure_4.8_B16413.jpg)

Figure 4.8 – A photoresistor sensor module connected to a Blue Pill board

Here are the steps on how to connect the sensor module to the Blue Pill:

1.  Connect the sensor module's GND pin (in some sensor modules, it is labeled with the – symbol) to the microcontroller board's ground connector labeled GND.
2.  Connect the sensor module's +5V (in some modules, it is labeled with the + symbol) pin to the 3V3 voltage pin from the microcontroller board. Even though the 3V3 pin provides 3.3 volts, it is enough for applying voltage (feeding) to the sensor module.

    Important note

    Do not apply **5 volts** (**5V**) to the Blue Pill's input ports, because you may damage the Blue Pill. That's why you will connect the 3V3 voltage pin to the sensor module's +5V pin, so the module will provide up to 3.3 Volts as an analog voltage output, and the Blue Pill will use up that voltage to measure the amount of light.

3.  Connect the module's A0 pin to the Blue Pill's B1 pin (which is an input port).
4.  Now, connect the resistors to Blue Pill ports B12, B14, and B15\. They are output ports.
5.  As the last step, connect the green, yellow, and red LEDs' anodes to those resistors and then connect the LEDs' cathodes to the ground.

As you can see in *Figure 4.8*, we are not connecting a pull-down resistor to the Blue Pill's input port because the sensor module already contains a pull-down resistor. This allows us to save some space in the circuit and time to connect extra components to it.

The following section describes how to use the sensor module with the Curiosity Nano.

### Connecting the photoresistor sensor module to the Curiosity Nano board

In this section, we will analyze how the sensor module's analog pin is connected to the Curiosity Nano, as is shown in *Figure 4.9*. Please remember that this circuit does not need the pull-down resistor. We are using the same input port that was used for connecting the single photoresistor, described in an earlier section of this chapter. *Figure 4.9* shows a Fritzing diagram that includes the photoresistor module connected to the Curiosity Nano board.

![Figure 4.9 – The photoresistor module connected to Curiosity Nano's RA5 input port](img/Figure_4.9_B16413.jpg)

Figure 4.9 – The photoresistor module connected to Curiosity Nano's RA5 input port

Following the Fritzing diagram from *Figure 4.9*, these are the steps for connecting the photoresistor sensor module to the Curiosity Nano:

1.  Connect the module's GND pin (in some modules, it is labeled with the - symbol) to the microcontroller board's ground connector, labeled GND.
2.  Connect the module's +5V pin (in some modules, it is labeled with the + symbol) to the VTG pin from the microcontroller board. Even though the VTG pin provides 3.3 volts, it may be enough to apply voltage (feeding) to the sensor module.

    Important note

    Do not apply 5V to the Curiosity Nano's input ports because you may damage the microcontroller board. That's why you will connect the Curiosity Nano's VTG voltage pin to the sensor module's +5V pin, so the sensor module will provide up to 3.3 Volts as an output and the Curiosity Nano will use up that voltage to measure the amount of light.

3.  Connect the sensor module's A0 pin to the Curiosity Nano's RA5 pin (which is an input port).
4.  Now, connect the resistors that will protect the 3 LEDs to ports RD1, RD2, and RD3\. They are output ports.
5.  As the last step, connect the green, yellow, and red LEDs' anodes to those resistors and then connect the LEDs' cathodes to the ground.

As you can see from *Figure 4.9*, similar to the example with the Blue Pill, we are not connecting a pull-down resistor to the Curiosity Nano's input port because the sensor module already contains a pull-down resistor.

You can use the same code used for the photoresistor used in *Figure 4.6* and *Figure 4.7* to get analog values from the photoresistor sensor module shown in *Figure 4.9*. You may need to adjust the values in the code if necessary. You can experiment with a number of environments (for example, using the sensor in a living room or a bedroom) to see the analog values obtained by the microcontroller board from the sensor change and thus adjust the code accordingly.

The next section describes how to test the photoresistor if something is wrong with it, for example, the LEDs are not turning on, as a way to troubleshoot it.

In this section, you learned how to connect the photoresistor module and the LEDs to the Curiosity Nano's ports. This section is important because it shows how practical and easy to use the photoresistor modules are, which facilitate the connections in microcontroller board projects.

# Testing out the photoresistor

This section focuses on how to test out a photoresistor to see if it is working OK. First of all, remember that the photoresistor used in this chapter does not have polarity, so you can safely connect any of its pins (legs) to a microcontroller board's input port.

You also need to make sure that the pull-down resistor connected to the photoresistor has the right value. For example, the pull-down resistor used in the Blue Pill example from *Figure 4.4* is 10K ohm, and we used a 220-ohm resistor for the Curiosity Nano example from *Figure 4.6*. We found those resistor values experimentally. You can try out different resistors connected to the photoresistor to see if the voltage passing through the photoresistor changes widely. Ideally, that voltage should be changing between 0 and 3.3 volts, or close to those values, because in our circuit examples from this chapter, we connected one pin of the photoresistor to 3.3 volts.

In order to see if the photoresistor is working OK, you can use a multimeter. Follow the next steps for testing the photoresistor with a multimeter:

1.  Connect the multimeter's red probe (test lead) to one pin of the photoresistor.
2.  Connect the multimeter's black probe to the other pin of the photoresistor.
3.  Turn on the multimeter and set it up for measuring resistance (ohms).
4.  Cover the photoresistor with your hand and uncover it. Its resistance should change in the multimeter because the light the photoresistor is receiving changes.

If you don't have a multimeter, you can test out a photoresistor with just a voltage source, such as batteries, and an LED (any color will do). *Figure 4.10* shows how to connect the photoresistor to the LED:

![Figure 4.10 – Testing out the photoresistor](img/Figure_4.10_B16413.jpg)

Figure 4.10 – Testing out the photoresistor

As you can see from *Figure 4.10*, the light from the LED should change (for example, changing from dim to bright) when the photoresistor receives different amounts of light, so try to cover it with your hand and see what happens. If the LED light does not change, it is probable that the photoresistor is damaged, so you will need to replace it. Here are the steps for connecting the components:

1.  Connect the positive (`+`) battery terminal to one pin of the photoresistor.
2.  Connect the other photoresistor pin to the LED's anode pin.
3.  Connect the LED's cathode pin to the negative (`-`) battery terminal.

Be careful when connecting the LED's pins. If it is connected in reverse, the LED will not turn on.

# Summary

In this chapter, we learned what sensors are and their applications in electronic projects. This is important because we will continue applying sensors in other chapters from this book. We also defined photoresistors, and how they are classified. We also learned how to connect a photoresistor sensor module to the Blue Pill and Curiosity Nano boards' input ports, and how to analyze and use analog data from a photoresistor.

In this chapter, you learned important pieces of information. We have defined what a sensor and a photosensor is. You can now read data from them using microcontroller boards. The chapter also described how to connect a photosensor module to a microcontroller board.

[*Chapter 5*](B16413_05_Final_NM_ePub.xhtml#_idTextAnchor069), *Humidity and Temperature Measurement*, will explain what a humidity and temperature sensor is and how can we acquire and use its analog data with both the Blue Pill and Curiosity Nano microcontroller boards. This can have a number of applications, such as measuring the temperature and humidity of a greenhouse.

# Further reading

*   Horowitz, P., Hill, W. (2015), *The art of electronics* [3rd ed.], Cambridge University Press: New York, NY.
*   Microchip (2019), *PIC16F15376 Curiosity Nano Hardware User Guide*, Microchip Technology, Inc. Available from: [http://ww1.microchip.com/downloads/en/DeviceDoc/50002900B.pdf](http://ww1.microchip.com/downloads/en/DeviceDoc/50002900B.pdf)
*   Mims, F.M. (2000), *Getting started in electronics*, Lincolnwood, IL: Master Publishing, Inc.