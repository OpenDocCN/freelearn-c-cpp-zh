# *Chapter 14*: COVID-19 20-Second Hand Washing Timer

This chapter describes a useful project where you will make a touchless timer by waving at an ultrasonic sensor. This timer will count the minimum time of 20 seconds recommended by health authorities for properly washing our hands for preventing contamination from viruses such as SARS-CoV-2 that produces COVID-19 symptoms. The project involves an inexpensive ultrasonic sensor that detects when a user waves at the sensor by measuring the distance between the user and the circuit, triggering the counting. This application must be enclosed in a waterproof container to avoid soaking the circuit while the user washes their hands and damaging it. We explain at the end of the chapter how to do this.

In this chapter, we will cover the following main topics:

*   Programming the counter (timer)
*   Showing the timer on an LCD
*   Connecting an ultrasonic sensor to the microcontroller board
*   Putting everything together – think of a protective case for the project!
*   Testing the timer

By the end of this chapter, you will have learned how to properly connect an ultrasonic sensor and an LCD to a microcontroller board. In addition, you will learn how to read input values from a sensor to activate the 20-second counting. You will also learn how to code an efficient and effective timer that runs on a microcontroller board.

# Technical requirements

The software tool that you will be using in this chapter is the Arduino IDE for editing and uploading your programs to the Blue Pill microcontroller board.

The code used in this chapter can be found in the book's GitHub repository:

[https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter14](https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter14)

The Code in Action video for this chapter can be found here: [https://bit.ly/3gQZdPf](https://bit.ly/3gQZdPf)

In this chapter, we will use the following pieces of hardware:

*   One solderless breadboard.
*   One Blue Pill microcontroller board.
*   One micro-USB cable for connecting your microcontroller board to a computer and a power bank.
*   One USB power bank.
*   One ST-INK/V2 electronic interface, needed for uploading the compiled code to the Blue Pill. Bear in mind that the ST-LINK/V2 requires four female-to-female DuPont wires.
*   One HC-SR04 ultrasonic sensor.
*   One 1602 16x2 LCD.
*   One 2.2k ohm resistor, 1/4 watt. This is for the LCD.
*   A dozen male-to-male and a dozen male-to-female DuPont wires.

The next section explains how to code the 20-second timer that runs on the Blue Pill microcontroller board.

# Programming the counter (timer)

This section shows you how to code the 20-second timer. Programming a timer like this one is not trivial, since a user could activate the timer many times by waving at the ultrasonic sensor more than once while the counter is on. The program should not take into account those multiple activations if the 20-second counting is going on; otherwise, the counter will re-start multiple times and the counting will not be accurate. We also need to consider saving energy by turning off the LCD when not in use.

We should code our 20-second timer application by following these steps:

1.  Read values from the ultrasonic sensor.
2.  Check whether the user is waving at the sensor within 15 centimeters of the sensor. If this happens, do this:

    a) Turn on the LCD light.

    b) Show the **Lather hands** message and show the 20-second counting on the LCD.

    c) When the counter finishes, show this message on the LCD: **Rinse your hands**.

    d) Wait for 4 seconds and then turn off the LCD to conserve energy.

    e) Return to *step 1*.

The following is the Arduino IDE code that runs on the Blue Pill, programmed following the preceding steps. The following code snippet shows how the variables and constants are defined. The HC-SR04's echo digital value is obtained from port `A9` (labeled `PA9`) and stored in the `echo` variable. Similarly, the trigger value is sent to the ultrasonic sensor through port `A8` (`PA8`) and stored in the `trigger` variable. Please note that the `dist_read` variable stores the distance between an object (for example, a waving hand) and the ultrasonic sensor, measured in centimeters:

```cpp
#include "elapsedMillis.h" 
elapsedMillis timeElapsed;
#include <LiquidCrystal.h> 
const int rs = PB11, en = PB10, d4 = PB0, d5 = PA7, d6 =    PA6, d7 = PA5;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7); 
#define backLight PB12    
#define trigger PA8 
#define echo PA9   
const int max_dist=200; 
float duration,dist_read=0.0;  
```

As you can see from the preceding code, it uses a library called `elapsedMillis.h` to calculate the time in milliseconds that has elapsed on the counting. This library can also be obtained from [https://www.arduino.cc/reference/en/libraries/elapsedmillis/](https://www.arduino.cc/reference/en/libraries/elapsedmillis/). You can find this library in the code folder uploaded to the book's GitHub repository. It is useful because by using this library we avoid using the `delay()` function in the 20-second counting. If we use the `delay()` function, the Blue Pill's counting and reading sensor values from the ports could mess up. Note that the library is written between double quotes because in C++ this means that the library is in the same folder as the source code, which is a library that does not belong to the original Arduino IDE installation. The code also uses the `LiquidCrystal.h` library, used for controlling the 1602 LCD. This library already comes with the standard Arduino IDE installation so there is no need for installing it separately.

This code snippet sets up the LCD and the Blue Pill ports:

```cpp
void setup() {
  lcd.begin(16, 2); 
  pinMode(trigger, OUTPUT);
  pinMode(echo,INPUT);
  pinMode(backLight, OUTPUT);
}
```

The following code segment shows the code's main loop, which reads the ultrasonic sensor values and calculates the distance between the user's waving hand and the sensor:

```cpp
void loop() {
  digitalWrite(trigger, LOW); 
  delayMicroseconds(2); 
  digitalWrite(trigger, HIGH); 
  delayMicroseconds(10); 
  digitalWrite(trigger, LOW); 
  duration = pulseIn(echo, HIGH);
  dist_read = (duration*.0343)/2;
```

The following code segment from the main loop function calculates whether the distance between the user and the sensor is equal to or less than 15 centimeters, then activates the 20-second counter and shows it on the LCD:

```cpp
  if ((dist_read<=15) & (dist_read>0)) 
  {
    lcd.display();
    digitalWrite(backLight, HIGH); 
    timeElapsed=0;
    lcd.setCursor(0, 0);
    lcd.print("lather hands :) ");
    lcd.setCursor(0, 1);
    lcd.print("  ");
    while (timeElapsed < 21000)
    { 
        lcd.setCursor(0, 1);
        lcd.print(timeElapsed / 1000);
     }
     lcd.setCursor(0, 0);
     lcd.print("rinse hands :)  ");   
     delay(4000);
  }
   lcd.noDisplay();
   digitalWrite(backLight, LOW);
}
```

Tip

You can also run the previous code on Arduino microcontroller boards. You just need to change the port numbers used for the LCD and sensor connections. For example, if you are using an Arduino Uno board, change this line to `const int rs=12,en=11,d4=5,d5=4,d6=3,d7=2;` using Arduino board digital ports `12`, `11`, `5`, `4`, `3`, and `2`, respectively. You will also need to change these lines to the following:

`#define backLight 6`

`#define trigger 7`

`#define echo 8`

So, you will use Arduino digital ports 6, 7 and 8 for the ultrasonic sensor.

Bear in mind that the code uploaded to the GitHub repository contains many comments explaining its most important parts.

The next section explains how to connect the 1602 LCD to the Blue Pill to show the 20-second count on it.

# Showing the timer on an LCD

In this section, we explain how to connect and use the 1602 LCD to show the timer on it. *Figure 14.1* shows the Fritzing diagram similar to the one explained in [*Chapter 5*](B16413_05_Final_NM_ePub.xhtml#_idTextAnchor069), *Humidity and Temperature Measurement*:

![Figure 14.1 – The LCD connected to the Blue Pill microcontroller board](img/Figure_14.1_B16413.jpg)

Figure 14.1 – The LCD connected to the Blue Pill microcontroller board

The following are the steps for connecting the LCD to the Blue Pill, following the diagram from *Figure 14.1*:

1.  Connect the Blue Pill's **GND** (also labeled as **G**) pins to the solderless breadboard rails.
2.  Connect the Blue Pill's **5V** pin (providing 5 volts) to the breadboard rails.
3.  Connect the USB cable to the Blue Pill and then to your computer or a USB power bank.
4.  Insert the LCD's 16 pins into the solderless breadboard.
5.  Connect the LCD's **VSS** pin to ground (the lower breadboard rail).
6.  Connect the LCD's **VDD** pin to 5 volts (the lower breadboard rail).
7.  Connect the 2.2k ohm resistor to the LCD's **V0** pin and to ground (the lower breadboard rail).
8.  Connect the LCD's **RS** pin to the Blue Pill's **B11** pin.
9.  Connect the LCD's **RW** pin to ground (lower breadboard rail).
10.  Connect the LCD's **E** pin to the Blue Pill's **B10** pin.
11.  Connect the LCD's **D4** pin to the Blue Pill's **B0** pin.
12.  Connect the LCD's **D5** pin to the Blue Pill's **A7** pin.
13.  Connect the LCD's **D6** pin to the Blue Pill's **A6** pin.
14.  Connect the LCD's **D7** pin to the Blue Pill's **A5** pin.
15.  Connect the LCD's **A** pin to the Blue Pill's port **B12**.
16.  Connect the LCD's **K** pin to ground (lower breadboard rail).
17.  The LCD's **D0**, **D1**, **D2**, and **D3** pins are not connected.

After doing the preceding steps, you have accomplished connecting the LCD to the Blue Pill board. The LCD will be useful for showing the 20-second count. Well done!

*Figure 14.2* shows how everything is connected:

![Figure 14.2 – The Blue Pill microcontroller board connected to the LCD ](img/Figure_14.2_B16413.jpg)

Figure 14.2 – The Blue Pill microcontroller board connected to the LCD

As you can see from *Figure 14.2*, the 1602A LCD is easy to connect to the Blue Pill. The 2.2k ohm connected to the LCD's pin **V0** sets up the LCD's contrast.

Tip

You can use a 50k ohm variable resistor (also known as a **potentiometer**) instead of the 2.2k ohm resistor connected to the LCD's pin **V0** to adjust the display contrast. Just connect the potentiometer's middle pin to **V0**, one pin to ground, and the other pin to **5V**.

Please note that the LCD pin **A** (pin no. 15) is connected to the Blue Pill's **B12** port, which controls the LCD by turning its back light on or off via coding.

Tip

Allow enough space on the solderless breadboard between the Blue Pill and the rest of the electronic components (LCD, ultrasonic sensor, and so on) to facilitate the connection of the ST-Link/V2 interface to the Blue Pill.

The next section explains how to use the ultrasonic sensor to see whether a user is waving at the sensor to trigger the 20-second timer.

# Connecting an ultrasonic sensor to the microcontroller board

This section explains how an ultrasonic sensor works, and it describes how to connect the HC-SR04 sensor to the Blue Pill microcontroller board, describing how to use its four-pin functions. The ultrasonic sensor will be used to check whether the user waves at it to initiate the 20-second counting.

## What is an ultrasonic sensor?

Ultrasonic waves are sound waves that have a frequency that is higher than the frequencies that most human beings can hear, which is above 20,000 Hz. Ultrasonic sounds, or ultrasound, can have different applications, including something called **echolocation**, used by animals such as bats for identifying how far their prey is using reflected sounds. The same principle is applied in ultrasonic sensors.

An **ultrasonic sensor** is a dedicated electronic component that generally contains a number of electronic parts such as resistors, transistors, diodes, a crystal clock, a special microphone, and a speaker. Many ultrasonic sensors are technically modules, because they integrate a number of electronic parts, and this integration as a module facilitates the connection with other devices such as microcontroller boards. An ultrasonic sensor measures the distance between an object (for example, a waving hand) and the sensor by using ultrasonic sound waves. The sensor emits ultrasonic sound waves through a speaker and receives through a microphone the reflected ultrasonic waves that hit the object. The sensor measures the time it takes between the sound wave emission and reception.

## How does an ultrasonic sensor work?

An ultrasonic sensor (such as the **HC-SR04**) emits and receives ultrasonic sound waves (working like **sonar**) to determine the distance to an object. Sonar is an echolocation device used for detecting objects underwater, emitting sound pulses (generally using ultrasound frequencies), measuring the time it takes for the reflection of those pulses, and calculating the distance between the object and the sonar device. The ultrasonic sensor used in this chapter is not meant to be used underwater, as we can see later.

Some ultrasonic sensors (such as the HC-SR04 used in this chapter) use sound waves with a frequency of 40 kHz, well above the range of sound frequencies that the human ear can perceive on average, which is 20 Hz to 20 kHz.

This is how the HC-SR04 ultrasonic sensor works:

1.  A microcontroller board sends a digital signal to the sensor's `Trig` pin, triggering (initiating) the ultrasonic wave emission through the sensor's speaker.
2.  When a high-frequency sound wave hits an object, it is reflected back to the sensor and this reflection is picked up by the sensor's microphone.
3.  The sensor sends out a digital signal to the microcontroller board through its `Echo` pin.
4.  A microcontroller board receives that digital signal from the `Echo` pin, encoding the duration between the sound wave emission and reception.

The distance between the sensor and the object is calculated as follows:

D=(T*C) ⁄ 2

The symbols denote the following:

*   **D**: Distance
*   **T**: Time it takes between ultrasonic wave emission and reception (duration)
*   **C**: General speed of sound (343 m/s in dry air)

The distance is divided by 2 because we need just the sound wave's return distance.

*Figure 14.3* shows a Fritzing diagram of the ultrasonic sensor:

![Figure 14.3 – The HC-SR04 ultrasonic sensor pinout](img/Figure_14.3_B16413.jpg)

Figure 14.3 – The HC-SR04 ultrasonic sensor pinout

From *Figure 14.3*, you can see the sensor pinout. The **VCC** pin is connected to a 5-volt power supply. The **Trig** and **Echo** pins are connected to the microcontroller board's digital ports. The **GND** pin is connected to ground.

Here are the technical characteristics of the HC-SR04 sensor:

*   **Operating voltage**: DC 5 volts
*   **Operating frequency**: 40 kHz
*   **Operating current**: 15 mA
*   **Maximum operational range**: 4 meters
*   **Minimum operational range**: 2 centimeters
*   **Resolution**: 0.3 centimeters
*   **Measuring angle**: 30 degrees (sensor's field of view)

*Figure 14.4* shows the HC-SR04 ultrasonic sensor used in this chapter:

![Figure 14.4 – The HC-SR04 ultrasonic sensor](img/Figure_14.4_B16413.jpg)

Figure 14.4 – The HC-SR04 ultrasonic sensor

From *Figure 14.4*, you can see that the sensor has two speaker-like components. One of them is actually a small speaker emitting ultrasonic sound signals and the other one is a microphone that captures those signals back after they bounce on an object. *Figure 14.5* shows the back of the HC-SR04 ultrasonic sensor:

![Figure 14.5 – The back side of the HC-SR04 sensor](img/Figure_14.5_B16413.jpg)

Figure 14.5 – The back side of the HC-SR04 sensor

As you can see from *Figure 14.5*, the back of the sensor contains electronic components such as resistors, transistors, and integrated circuits that support the generation and reception of ultrasonic signals.

There are other types of ultrasonic sensors, such as the Maxbotix MaxSonar ultrasonic sensor. Its Fritzing diagram is shown in *Figure 14.6*:

![Figure 14.6 – The Maxbotix MaxSonar ultrasonic sensor](img/Figure_14.6_B16413.jpg)

Figure 14.6 – The Maxbotix MaxSonar ultrasonic sensor

The sensor shown in *Figure 14.6* can be used with microcontroller boards. The Maxbotix MaxSonar is an accurate and long-range ultrasonic sensor (it can measure distances up to 6.45 meters), but it is expensive and requires connecting seven wires to its seven pins. The HC-SR04 sensor will suffice for our 20-second timer application. It is low cost and easy to connect to, requiring only four wires.

The timer starts when the ultrasonic sensor detects a user waving at it, so the sensor will trigger the timer.

*Figure 14.7* shows how to connect the HC-SR04 ultrasonic sensor to the Blue Pill:

![Figure 14.7 – The ultrasonic sensor connected to the Blue Pill microcontroller board](img/Figure_14.7_B16413.jpg)

Figure 14.7 – The ultrasonic sensor connected to the Blue Pill microcontroller board

From *Figure 14.7*, you can see that one of the breadboard's lower rails is connected to the Blue Pill's **5V** pin. The other rail is connected to Blue Pill's ground. The following are the steps for connecting the HC-SR04 sensor to the Blue Pill in addition to the steps followed in *Figure 14.1*, following the diagram from *Figure 14.7*:

1.  Connect the sensor's **VCC** pin to the breadboard's low rail that is connected to **5V**.
2.  Connect the sensor's **Trig** pin to the Blue Pill's **A8** pin.
3.  Connect the sensor's **Echo** pin to the Blue Pill's **A9** pin.
4.  Connect the sensor's **GND** pin to the breadboard's lower rail that is connected to ground.

*Figure 14.8* shows how everything is connected:

![Figure 14.8 – The Blue Pill connected to the LCD and the ultrasonic sensor](img/Figure_14.8_B16413.jpg)

Figure 14.8 – The Blue Pill connected to the LCD and the ultrasonic sensor

Please note from *Figure 14.8* that all the connections from the ultrasonic sensor are done on its back side. This is to avoid any cable obstructing the ultrasonic signals sent and received on the front of the sensor. Also note that the 1602 LCD's power (pin **VDD**) is connected to Blue Pill's pin **5V**. If you feed the LCD with 3.3 volts, it may not work.

Tip

Make sure that there are no wires obstructing the field of view of HC-SR04 ultrasonic sensor; otherwise, they will produce erratic or false measurements and results with the 20-second counting.

Also note from *Figure 14.8* that the breadboard's upper rail is connected to the Blue Pill's **G** pin (in some Blue Pills it is labeled as **GND**), which serves to connect the LCD's ground and its 2.2k ohm resistor that is used to preset the LCD's contrast.

The next section explains how to encase the whole project to protect it from dust, water, and so on, and to facilitate its use in a place for washing hands.

# Putting everything together – think of a protective case for the project!

This section shows how you can place the electronic circuit with the ultrasonic sensor inside a protective case. The section also shows some suggestions on how to fit everything in a plastic or glass container, because if you use the 20-second counter in a bathroom or in a place close to a hand washing sink, you will need to protect the circuit against water spilling and soap stains that can damage the electronic components used in this 20-second counter project. We do not recommend you connect the Blue Pill board to a wall USB adapter for security reasons. It is best to connect the Blue Pill to a USB power bank.

If you can't fit the whole 20-second counter circuit (including its solderless breadboard) in a plastic or glass container, try connecting the Blue Pill on a smaller solderless breadboard such as a half breadboard. Detach the ultrasonic sensor and the LCD from the breadboard and position and attach them inside the container with strong adhesive tape. A Fritzing diagram about this smaller circuit is shown in *Figure 14.9*:

![Figure 14.9 – Connecting everything on a small solderless breadboard](img/Figure_14.9_B16413.jpg)

Figure 14.9 – Connecting everything on a small solderless breadboard

As you can see from *Figure 14.9*, by using a half-breadboard you can make all the connections more compact so you can use a small container such as an empty instant coffee jar or any plastic container that has a lid. You could mount the ultrasonic sensor on the lid and place the rest inside of the container or jar. You will need to use female-to-male DuPont wires to connect the LCD and the ultrasonic sensor to the half breadboard.

*Figure 14.10* shows a prototype design with a custom-made case giving you an idea of how you could encase all the components and protect the electronics from water spilling:

![Figure 14.10 – A 3D prototype design containing the whole project](img/Figure_14.10_B16413.jpg)

Figure 14.10 – A 3D prototype design containing the whole project

As you can see from *Figure 14.10*, that case could be a plastic box where the 1602A LCD is placed on top of it. The HC-SR04 sensor is placed at the front of the case. The interior of the case could contain the small breadboard, wires, the Blue Pill, the resistor, the power bank, and the USB cable. Don't place the whole project very close to a hand wash sink, just in case.

# Testing the timer

This section shows how to test out the 20-second timer.

Once you insert the electronic circuit with the sensor, the Blue Pill, and the LCD in a protective case, try it in a bathroom. Carefully place it close to a hand washing sink if you can, to facilitate activating it and seeing the counting while you wash your hands. See whether you can fix it to a wall or a surface so it won't move and that no one accidentally knocks it over while waving at it. Safety first!

You should connect the Blue Pill to a portable power bank that has a USB socket. This is to avoid connecting the Blue Pill to a wall USB adapter to make it safer to use in an environment such as a bathroom, as shown in *Figure 14.11*:

![Figure 14.11 – A power bank connected to the Blue Pill microcontroller board](img/Figure_14.11_B16413.jpg)

Figure 14.11 – A power bank connected to the Blue Pill microcontroller board

You can test out everything with a small power bank, such as the one shown in *Figure 14.11*.

Try activating the timer by waving at the sensor numerous times. You will see that sometimes the circuit counts up to 21\. This is because most microcontroller boards (including the Blue Pill) do not calculate the time very accurately. Try adding a variable and a decision to the code to show the count up to number 20\. Hint: Stop showing the counting when it reaches 20\. It is not critical if it counts up to 21 when you wash your hands for trying to *destroy* the virus that causes COVID-19\. The longer the counting the better.

You can adjust the detected distance if you feel that the user needs to wave at the sensor at a different distance. Try changing the value from this line of code:

`if ((dist_read<=15) & (dist_read>0))`

The `15` value means that the LCD will activate and show the counting if you are waving at the sensor at a distance of 15 centimeters or less from the sensor. Try to change the value to a greater number, perhaps 20 centimeters.

If you think that the text and numbers shown on the LCD need more contrast, try changing the 2.2k ohm resistor to a smaller one, such as 1 k ohm. This may happen if your bathroom or the place where you will use the counter is too bright.

This is an interesting test: try the 20-second counter with people of different ages, to see whether the ultrasonic sensor can detect different hand sizes. For example, see whether the sensor detects the waving hands of small children and adults.

# Summary

In this chapter, we learned the basics of coding an easy-to-read 20-second counter. This count is recommended by many health authorities for properly washing our hands during that time in an attempt to destroy some viruses such as the one that causes COVID-19\. The chapter also explained how the HC-SR04 ultrasonic sensor works for activating the counter. One major skill that you gained on completing the project from this chapter is that you learned how to connect a practical LCD to a microcontroller board, and how we could show the counting on an LCD. You can use the LCD in other projects that require showing numeric or text data from a microcontroller board.

We have covered in this chapter a practical way to obtain data from a sensor, process it on the microcontroller board, and do something about it such as showing results on an LCD. Obtaining data from sensors and processing it is one of the main applications of microcontrollers, leveraging their simplicity for connecting sensors to their input/output ports.

# Further reading

*   Choudhuri, K. B. R. (2017), *Learn Arduino Prototyping in 10 Days*, Birmingham, UK: Packt Publishing Ltd
*   Gay, W. (2018), *Beginning STM32: Developing with FreeRTOS, libopencm3 and GCC*, New York, NY: Apress
*   HC-SR04 (2013), *HC-SR04 user's manual V1.0\. Cytron Technologies*, available from [https://docs.google.com/document/d/1Y-yZnNhMYy7rwhAgyL_pfa39RsB-x2qR4vP8saG73rE/edit?usp=sharing](https://docs.google.com/document/d/1Y-yZnNhMYy7rwhAgyL_pfa39RsB-x2qR4vP8saG73rE/edit?usp=sharing)
*   Horowitz, P. and Hill, W. (2015), *The Art of Electronics*, [3rd ed.] Cambridge University Press: New York, NY
*   LCD1602 (2009), *LCM module data sheet TC1602A-01T*, Tinsharp Industrial Co., Ltd. available from [https://cdn-shop.adafruit.com/datasheets/TC1602A-01T.pdf](https://cdn-shop.adafruit.com/datasheets/TC1602A-01T.pdf)
*   Microchip (2019), *PIC16F15376 Curiosity Nano hardware user guide*, Microchip Technology, Inc. available from [http://ww1.microchip.com/downloads/en/DeviceDoc/50002900B.pdf](http://ww1.microchip.com/downloads/en/DeviceDoc/50002900B.pdf)