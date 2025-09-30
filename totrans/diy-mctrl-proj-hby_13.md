# *Chapter 13*: COVID-19 Social-Distancing Alert

When the world celebrated the arrival of the year 2020, a pandemic was arising due to a new disease: COVID-19\. With the emergence of this pandemic, all human activities were affected to a lesser or greater degree.

The education sector has been one of the most affected in this sense. All schools worldwide temporarily suspended their activities, since the risk of contagion in these environments can be very high. After a few months of lockdowns, schools around the world gradually began to resume face-to-face activities, following rigorous standards of disinfection and protocols to ensure physical distancing between students and school staff (Global Education Cluster, 2020).

The recommendation of the **World Health Organization** (**WHO**) for physical distancing is to remain at least 1 **meter** (**m**) (3 **feet** (**ft**)) apart between people, with 2 m (6 ft) being the most general recommendation to minimize the risk of contagion in children (KidsHealth, 2021). This measure is known as **social distancing**. Furthermore, technology sometimes acts as a great way to enforce these measures.

In this chapter, you will learn how to create a device that uses microcontroller technology to enforce social distancing to help children get used to maintaining a safe physical distance. When they are not at a safe physical distance, they will receive a sound alert from the device. The device you will create can be used by children as a wearable device for daily use by putting it in a case and using it as a necklace, as shown in the following screenshot:

![Figure 13.1 – Wearable social-distancing device for children](img/B16413_Figure_13.1.jpg)

Figure 13.1 – Wearable social-distancing device for children

In this chapter, we will cover the following main topics:

*   Programming a piezoelectric buzzer
*   Connecting an ultrasonic sensor to the microcontroller board
*   Writing a program for getting data from the ultrasonic sensor
*   Testing the distance meter

By completing this chapter, you will know how to program an electronic measurement of distance ranges using an **STM32 Blue Pill board**. You will also learn how to play an alarm when the distance is measured as less than 2 m.

Important note

This project is only for demonstration and learning purposes. Please do not use it as a primary social-distancing alarm for preventing the risk of COVID-19 contagion.

# Technical requirements

The hardware components that will be needed to develop the social-distancing alarm are listed here:

*   One solderless breadboard.
*   One Blue Pill microcontroller board.
*   One ST-LINK/V2 electronic interface is needed for uploading the compiled code to the Blue Pill board. Bear in mind that the ST-LINK/V2 interface requires four female-to-female jumper wires.
*   One HC-SR04 ultrasonic sensor.
*   One buzzer.
*   Male-to-male jumper wires.
*   Female-to-male jumper wires.
*   A power source.
*   Cardboard for the case.

As usual, you will require the Arduino **integrated development environment** (**IDE**) and the GitHub repository for this chapter, which can be found at [https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter13](https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter13)

The Code in Action video for this chapter can be found here: [https://bit.ly/3gS2FKJ](https://bit.ly/3gS2FKJ)

Let's begin!

# Programming a piezoelectric buzzer

In this section, you will learn what a buzzer is, how to interface it with the **STM32 Blue Pill**, and how to write a program to build an alert sound.

A **piezoelectric buzzer** is a device that generates tones and beeps. It uses a piezoelectric effect, which consists of piezoelectric materials converting mechanical stress into electricity and electricity into mechanical vibrations. Piezoelectric buzzers contain a crystal with these characteristics, which changes shape when voltage is applied to it.

As has been common in these chapters, you can find a generic breakout module that is pretty straightforward to use, as shown in the following screenshot:

![Figure 13.2 – Piezoelectric buzzer breakout board](img/B16413_Figure_13.2.jpg)

Figure 13.2 – Piezoelectric buzzer breakout board

This breakout board connects to the STM32 Blue Pill microcontroller board with three pins, outlined as follows:

*   **Input/Output** (**I/O**): This pin must be connected to a digital output of the microcontroller.
*   **Voltage Common Collector** (**VCC**): Pin to supply power to the sensor (**5 volts**, or **5V**).
*   **Ground** (**GND**): Ground connection.

Next, you will learn how to interface these pins with the Blue Pill microcontroller board.

## Connecting the components

You will need a solderless breadboard to connect the buzzer to the STM32 Blue Pill microcontroller and a wire to connect the components. Follow these steps:

1.  You need to place the STM32 Blue Pill and the buzzer into the solderless breadboard and leave space in the solderless breadboard to connect the jumper wires.
2.  Connect the GND pin of the sensor to a GND terminal of the SMT32 Blue Pill.
3.  Next, you need to connect the VCC pin to the 5V bus of the STM32 Blue Pill.
4.  Finally, connect the I/O pin of the buzzer to pin B12 of the Blue Pill. The following screenshot shows all the components connected to the solderless breadboard:

![Figure 13.3 – Piezoelectric buzzer interface to the Blue Pill](img/B16413_Figure_13.3.jpg)

Figure 13.3 – Piezoelectric buzzer interface to the Blue Pill

The following screenshot represents all the wiring between the STM32 Blue Pill and the piezoelectric buzzer and compiles the steps we just went through:

![Figure 13.4 – Circuit for piezoelectric buzzer connection](img/B16413_Figure_13.4.jpg)

Figure 13.4 – Circuit for piezoelectric buzzer connection

Up to now, we have explored piezoelectric buzzers and their components and functionality. You have also learned how to connect them to an STM32 Blue Pill microcontroller board using a solderless breadboard.

Now, you are ready to write a program in the C language to reproduce an audible alert in the buzzer. Don't forget to use the `STLink` to upload the script to the STM32 Blue Pill microcontroller board.

Let's start developing a program to play an audible alert with the STM32 Blue Pill, as follows:

1.  Let's get started defining which pin of the STM32 Blue Pill card pins will be used to play a sound in the buzzer. Run the following code:

    ```cpp
    const int PB12 pin (labeled B12 on the Blue Pill).
    ```

2.  Next, we will leave the `setup()` part empty. You will not need to initialize code for this script.
3.  The complete code is in the `loop()` part, as illustrated in the following code snippet:

    ```cpp
    void loop() 
    {
      tone(pinBuzzer, 1200);
      delay(250);
      noTone(pinBuzzer);
      delay(500);
      tone(pinBuzzer, 800);
      delay(250);
      noTone(pinBuzzer);
      delay(500);
    }
    ```

    We are using two new functions: `tone()` and `noTone()`. Let's see what their functionality is.

    `tone()` generates a square wave with a specific frequency from a pin. Its syntax is `tone(pin, frequency, duration)`, where the `pin` parameter is the pin of the Blue Pill to which the buzzer is connected. `frequency` is the frequency of the tone in `unsigned int`. The `duration` parameter is the tone's duration in `unsigned long` type.

    `noTone()` stops the generation of the square wave that was started with `tone()`. An error will not be generated if a tone has not been previously generated. Its syntax is `noTone(pin)`, where `pin` is the pin that is generating the tone.

    So, the preceding code starts a 1,200 Hz tone and holds it for 250 ms with the `delay()` function. Later, it stops it and waits 500 ms to generate a new tone during 250 ms, now 800 Hz, and stops it again with the same 500-ms pause. These steps are repeated as long as the program is running to simulate an alert sound.

The code for this functionality is now complete. You can find the complete sketch in the `Chapter13/buzzer` folder in the GitHub repository.

Let's view how we have advanced our learning. We discovered a component to play tones, learned how to connect it to the STM32 Blue Pill microcontroller, and wrote the code to play an audible alert.

The skills you have acquired so far in this section will allow you to create other electronic systems that require play and audible alerts. Coming up next, we will learn about ultrasonic sensors.

# Connecting an ultrasonic sensor to the microcontroller board

Before moving ahead, we need to learn about the functionality of the HC-SR04 ultrasonic sensor, how to interface it with the **STM32 Blue Pill**, and how to write a program to measure the distance between the sensor and another object.

This sensor emits an ultrasonic wave. When this wave collides with an object, the wave is reflected and received by the sensor. When the reflected signal is received, the sensor can calculate the time it took to be reflected, and thus the distance of the collision object can be measured.

The sensor can be seen in the following screenshot:

![Figure 13.5 – Ultrasonic sensor](img/B16413_Figure_13.5.jpg)

Figure 13.5 – Ultrasonic sensor

This sensor board connects to the STM32 Blue Pill microcontroller board with four pins, outlined as follows:

*   **Trigger**: This pin enables the ultrasonic wave.
*   **Echo**: This pin receives the reflected wave.
*   **VCC**: The pin to supply power to the sensor (5V).
*   **GND**: Ground connection.

Next, it's time to interface these pins with the Blue Pill microcontroller.

## Connecting the components

A solderless breadboard will be required to connect the buzzer to the STM32 Blue Pill microcontroller and wire to connect the components. Proceed as follows:

1.  You need to place the STM32 Blue Pill and the sensor into the solderless breadboard and leave space to connect the jumper wires.
2.  Connect the GND pin of the sensor to a GND terminal of the SMT32 Blue Pill.
3.  Next, you need to connect the VCC pin to the 5V bus of the STM32 Blue Pill.
4.  Finally, connect the trigger pin of the buzzer to pin C14 and the echo pin to the C13 pin of the Blue Pill. The following screenshot shows all the components connected to the solderless breadboard:

![Figure 13.6 – Piezoelectric buzzer interface to the Blue Pill](img/B16413_Figure_13.6.jpg)

Figure 13.6 – Piezoelectric buzzer interface to the Blue Pill

The following screenshot represents all the wiring between the STM32 Blue Pill and the ultrasonic sensor:

![Figure 13.7 – Circuit for the ultrasonic sensor connection](img/B16413_Figure_13.7.jpg)

Figure 13.7 – Circuit for the ultrasonic sensor connection

Up to now, you have learned how to connect a sensor to the STM32 Blue Pill microcontroller board using a solderless breadboard.

Now, you will learn how to write a program in the C language to reproduce an audible alert in the buzzer. Don't forget to use the `STLink` to upload the script to the STM32 Blue Pill microcontroller board.

# Writing a program for getting data from the ultrasonic sensor

In this section, you will learn how to write a program to gather data from the ultrasonic sensor. Let's start, as follows:

1.  First, we will define which pins of the STM32 Blue Pill card will be used to read the sensor data. Also, we will declare two variables to save the duration of the sound-wave travel and another for calculating the distance traveled, as illustrated in the following code snippet:

    ```cpp
    const int pinTrigger = PC14;
    const int pinEcho = PC13;
    long soundWaveTime;
    long distanceMeasurement;
    ```

    The selected pins were the PC13 and PC14 pins (labeled C13 and C14 on the Blue Pill).

2.  Next, in the `setup()` function, begin the serial communication. You will set the trigger pin as an output pin and the echo pin as an input pin. We need to initialize the trigger in the `LOW` value. The code is illustrated in the following snippet:

    ```cpp
    void setup() {
      Serial.begin(9600);
      pinMode(pinTrigger, OUTPUT);
      pinMode(pinEcho, INPUT);
      digitalWrite(pinTrigger, LOW);
    }
    ```

3.  Now, we will code the `loop()` function. We need to start the ultrasonic wave, wait 10 **seconds** (**sec**), and turn off the wave. The code is illustrated in the following snippet:

    ```cpp
    void loop()
    {
      digitalWrite(pinTrigger, HIGH);
      delayMicroseconds(10); 
      digitalWrite(pinTrigger, LOW);
      ...
    }
    ```

4.  The next step is to read the echo pin of the sensor to know the total travel time of the wave. We do this with the `pulseIn()` function and store it in the variable we declared at the beginning, for this purpose. To calculate the distance, we take the value of the return pulse and divide it by 59 to obtain the distance in **centimeters** (**cm**), as illustrated in the following code snippet:

    ```cpp
    void loop()
    {
      digitalWrite(pinTrigger, HIGH);
      delayMicroseconds(10); 
      digitalWrite(pinTrigger, LOW);
      soundWaveTime = pulseIn(pinEcho, HIGH);
      distanceMeasurement = soundWaveTime/59;
      ...
    }
    ```

5.  Finally, you will show the distance value between the sensor and any object in front of our device in the serial console, as follows:

    ```cpp
    void loop()
    {
      digitalWrite(pinTrigger, HIGH);
      delayMicroseconds(10); 
      digitalWrite(pinTrigger, LOW);
      soundWaveTime = pulseIn(pinEcho, HIGH);
      distanceMeasurement = soundWaveTime/59;
      Serial.print("Distance: ");
      Serial.print(distanceMeasurement);
      Serial.println("cm");
      delay(500);
    }
    ```

The code for this functionality is now complete. You can find the complete sketch in the `Chapter13/ultrasonic` folder in the GitHub repository.

At the end of this section, you have learned how to write a program in the C language to measure the distance between an object and an ultrasonic sensor connected to the STM32.

With these skills, you will be able to develop electronic projects that require distance measurement, such as car-reverse-impact prevention.

# Testing the distance meter

Before testing the distance meter, we will need to wire together the buzzer and the ultrasonic sensor to the SMT32 Blue Pill in the solderless breadboard. The following screenshot illustrates a complete circuit diagram including the STM32, ultrasonic sensor, and buzzer together in the solderless breadboard:

![Figure 13.8 – Full circuit diagram of our social-distancing device](img/B16413_Figure_13.8.jpg)

Figure 13.8 – Full circuit diagram of our social-distancing device

The following screenshot shows how everything should be connected in the actual system:

![Figure 13.9 – The buzzer and ultrasonic sensor connections](img/B16413_Figure_13.9.jpg)

Figure 13.9 – The buzzer and ultrasonic sensor connections

Now, to complete the connection of the complete social-distancing device, we will need to write a new script combining the `Chapter13/buzzer` and `Chapter13/ultrasonic` scripts. The new script will be named `Chapter13/distance_meter`. Follow these steps:

1.  We need to declare the constants and variables of both scripts and add a new script to define the safety distance between the sensor device and another object. The code to do this is illustrated in the following snippet:

    ```cpp
    const int pinTrigger = PC14;
    const int pinEcho = PC13;
    const int pinBuzzer = PB12;
    const int distanceSafety = 200;
    long soundWaveTime;
    long distanceMeasurement;
    ```

    For COVID-19 social distancing, we will use 200 cm (2 m).

2.  The `setup()` function remains the same as the ultrasonic script, as illustrated in the following code snippet:

    ```cpp
    void setup() {
      Serial.begin(9600);
      pinMode(pinTrigger, OUTPUT);
      pinMode(pinEcho, INPUT);
      digitalWrite(pinTrigger, LOW);
    }
    ```

3.  Finally, in the `loop()` function, we will include a conditional to verify if our social-distancing device is physically separated less than 2 m from another person (object). If this is the case, play the audible alert. Here is the code to do this:

    ```cpp
    void loop()
    {
      digitalWrite(pinTrigger, HIGH);
      delayMicroseconds(10); 
      digitalWrite(pinTrigger, LOW);
      soundWaveTime = pulseIn(pinEcho, HIGH);
      distanceMeasurement = soundWaveTime/59;
      Serial.print("Distance: ");
      Serial.print(distanceMeasurement);
      Serial.println("cm");
      delay(500);
      if (distanceMeasurement < distanceSafety) {
        Serial.println("Sound alert");
        tone(pinBuzzer, 1200);
        delay(250);
        noTone(pinBuzzer);
        delay(500);
        tone(pinBuzzer, 800);
        delay(250);
        noTone(pinBuzzer);
        delay(500);
      }
    }
    ```

Now, you can measure social distancing, and it can be possible to use our device as a necklace in schools to maintain a safe physical distance, *only as a complement to the official safety instructions*.

To achieve this, we can create a cardboard case and insert our device in it. Print the template shown in the following screenshot—you can download this from the `Chapter13/cardboard` GitHub folder:

![Figure 13.10 – A cardboard-case template for our device](img/B16413_Figure_13.10.jpg)

Figure 13.10 – A cardboard-case template for our device

To better fit our electronic device in the case, it is recommended to change the jumper wires used to build the prototype (male-to-male) to male-to-female jumper wires and power it with a 5V battery, as shown in the following screenshot:

![Figure 13.11 – Adapted connections to fit into a cardboard-case template](img/B16413_Figure_13.11.jpg)

Figure 13.11 – Adapted connections to fit into a cardboard-case template

Finally, cut and glue the case and put the device we just created into the case to create a wearable device, as shown in the following screenshot:

![Figure 13.12 – A cardboard-case template for our device](img/B16413_Figure_13.12.jpg)

Figure 13.12 – A cardboard-case template for our device

Using this device, you will know whether you are at a safe distance to avoid possible COVID-19 infections by droplets.

We have reached the end of [*Chapter 13*](B16413_13_Final_NM_ePub.xhtml#_idTextAnchor173). Congratulations!

# Summary

What did we learn in this project? Firstly, we learned how to connect a piezoelectric buzzer to our Blue Pill microcontroller board and code a program to play an audible alarm. Then, we wrote a program to measure the distance between our electronic device and another object.

We also learned how to combine the two projects to create a social-distancing device that can be used to maintain a safe physical distance in this COVID-19 pandemic—for example, in schools, because children are more distracted and are more friendly and sociable.

It is important to remind you that this project is intended for learning purposes and should not be used as a primary alarm for preventing the risk of COVID-19 contagion in any circumstances. This is mainly because, at this time, we know that the main risk is airborne.

In the next chapter, we will learn to build a 20-second hand-washing timer.

# Further reading

*   Global Education Cluster. *Safe back to school: a practitioner's guide*. UNESCO. 2020:

    [https://healtheducationresources.unesco.org/library/documents/safe-back-school-practitioners-guide](https://healtheducationresources.unesco.org/library/documents/safe-back-school-practitioners-guide)

*   KidsHealth. *Coronavirus (COVID-19): Social Distancing With Children*. 2021:

    [https://kidshealth.org/en/parents/coronavirus-social-distancing.html](https://kidshealth.org/en/parents/coronavirus-social-distancing.html)