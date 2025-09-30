# *Chapter 8*: Gas Sensor

An indoor environment with good air quality is essential to guarantee a healthy environment (Marques and Pitarma, 2017). The **MQ-2 gas sensor** can be an excellent way to measure the quality parameters of indoor air or as an early fire detection system. In this chapter, you will learn how to build a practical system for detecting gases in the environment (which we will call a **gas sensor**) and connect the MQ-2 gas sensor to a **Blue Pill microcontroller card**.

The following main topics will be covered in this chapter:

*   Introducing the MQ-2 gas sensor
*   Connecting a gas sensor to the STM32 microcontroller board
*   Writing a program to read the gas concentration over the sensor board
*   Testing the system

At the end of this chapter, you will know about the operation of an MQ-2 gas sensor, and you will be able to connect it correctly to the STM32 microcontroller card and view the data obtained from the sensor. You will be able to apply what you have learned in projects that require the use of sensors to detect substances such as flammable gases or alcohol or measure air quality.

# Technical requirements

The hardware components that will be needed to develop the gas sensor are as follows:

*   One solderless breadboard
*   One Blue Pill board
*   ST-Link/V2
*   One MQ-2 breakout module
*   Seven male-to-male jumper wires
*   One LED 8x8 matrix
*   One 7219 breakout board
*   A 5 V power source

These components are widespread, and there will be no problems in getting them easily. On the software side, you will require the Arduino IDE and the GitHub repository for this chapter: [https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter08](https://github.com/PacktPublishing/DIY-Microcontroller-Projects-for-Hobbyists/tree/master/Chapter08)

The Code in Action video for this chapter can be found here: [https://bit.ly/2UpGDGs](https://bit.ly/2UpGDGs)

Let's first start by describing the characteristics of the MQ-2 gas sensor.

# Introducing the MQ-2 gas sensor

In this section, we will get to know the details of the main hardware component to build our gas sensor: the **MQ-2 sensor**. This sensor is recommended to detect LPG, propane, alcohol, and smoke, mainly with concentrations between 300 and 10,000 **parts per million** (**ppm**). So, we can say that it is a sensor to detect smoke and flammable gases.

Concentration refers to the amount of gas in the air and is measured in ppm. That is, if you have 2,000 ppm of LPG, it means that in a million gas molecules, only 2,000 ppm would be LPG and 998,000 ppm other gases.

The MQ-2 gas sensor is an electrochemical sensor that varies its resistance when exposed to certain gasses. It includes a small heater to raise the circuit's internal temperature, which provides the necessary conditions for the detection of substances. With the 5 V connection on the pins, the sensor is kept warm enough to function correctly.

Important note

The sensor can get very hot, so it should not be touched during operation.

The MQ series gas sensors are analog, making them easy to implement with any microcontroller card, such as the STM32 Blue Pill. It is very common to find the MQ-2 sensor in breakout modules, which facilitates connection and use since it will only be necessary to power it up and start reading its data. These breakout modules have a **digital output** (**DO**) that we can interpret as the presence (*LOW*) or absence (*HIGH*) of any gas detected by the sensor. *Figure 8.1* shows the MQ-2 gas sensor with a breakout board:

![Figure 8.1 – MQ-2 gas sensor with a breakout board](img/B16413_Figure_8.1.jpg)

Figure 8.1 – MQ-2 gas sensor with a breakout board

In the next section, we will learn how to connect the MQ-2 sensor to our solderless breadboard to obtain its reading data through digital and analog means.

# Connecting a gas sensor to the STM32 microcontroller board

In this section, we will build a gas sensor device utilizing the **STM32 Blue Pill** microcontroller board and a gas sensor module using the hardware components listed in the *Technical requirements* section. The gas sensor breakout board connects to the STM32 Blue Pill with four pins:

*   **Analog output** (**AO**): This pin generates an analog signal and must be connected to an analog input of the microcontroller.
*   **DO**: This pin generates a digital signal and must be connected to a digital input of the microcontroller.
*   **VCC**: Pin to supply power to the sensor (5 V).
*   **GND**: Ground connection.

For this project, you will learn how to interface the MQ-2 module with the STM32 board to acquire data in a digital and analog way. Let's start with the digital option.  

## Interfacing for digital reading

Now we are going to connect the electronic components to the breadboard, do the wiring, and finally connect everything to the STM32 Blue Pill:

1.  In connecting the components, place the sensor module and the STM32 Blue Pill on a solderless breadboard with enough space to add the wiring layer, as shown in *Figure 8.2*. The hardware connections for this project are exceptionally effortless:![Figure 8.2 – Components on the breadboard](img/B16413_Figure_8.2.jpg)

    Figure 8.2 – Components on the breadboard

2.  Next, power up the Blue Pill with an external power source. Connect the **5 V pin** to the red rail on the breadboard and a **G pin** to the blue track, as shown in *Figure 8.3*:![Figure 8.3 – Connections to the power supply](img/B16413_Figure_8.3.jpg)

    Figure 8.3 – Connections to the power supply

3.  Connect the **GND pin** of the MQ-2 sensor to a GND terminal of the SMT32 Blue Pill. Next, you need to connect the **VCC pin** to the 5 V bus of the Blue Pill, as shown in the following figure. In this section, we will read the DO, so it must be connected to a digital input on the Blue Pill card. Connect the DO of the MQ-2 sensor to pin B12 of the Blue Pill, as shown in *Figure 8.4*:

![Figure 8.4 – MQ-2 sensor connection for digital reading](img/B16413_Figure_8.4.jpg)

Figure 8.4 – MQ-2 sensor connection for digital reading

Finally, you need to use a power source such as batteries to power up the board. *Figure 8.5* summarizes all the hardware connections:

![Figure 8.5 – Circuit for the MQ-2 sensor connection for digital reading](img/B16413_Figure_8.5.jpg)

Figure 8.5 – Circuit for the MQ-2 sensor connection for digital reading

The preceding figure shows all the connections between the STM32 Blue Pill and the electronic parts. *Figure 8.6* presents the schematics for this project:

![Figure 8.6 – Schematics for the MQ-2 sensor connection for digital reading](img/B16413_Figure_8.6.jpg)

Figure 8.6 – Schematics for the MQ-2 sensor connection for digital reading

*Figure 8.7* shows how everything must be connected in our DIY gas sensor device:

![Figure 8.7 – Gas sensor device for digital reading](img/B16413_Figure_8.7.jpg)

Figure 8.7 – Gas sensor device for digital reading

In this subsection, we learned how to connect the electronics to create our gas sensor device with digital reading. Next, we will see how to connect it so that the reading is analog.

## Interfacing for analog reading

Only one step will be necessary to change how our hardware device reads data from the sensor to be an analog reading instead of a digital one:

1.  Disconnect the jumper wire from the DO pin and connect it to the AO pin of the MQ-2 sensor. Also, instead of connecting to pin B12, connect to pin AO of the Blue Pill, as shown in *Figure 8.8*:

![Figure 8.8 – MQ-2 sensor connection for analog reading](img/B16413_Figure_8.8.jpg)

Figure 8.8 – MQ-2 sensor connection for analog reading

*Figure 8.9* summarizes all the hardware connections:

![Figure 8.9 – Circuit for the MQ-2 sensor connection for analog reading](img/B16413_Figure_8.9.jpg)

Figure 8.9 – Circuit for the MQ-2 sensor connection for analog reading

*Figure 8.10* presents the schematics for the analog reading device:

![Figure 8.10 – Schematics for the MQ-2 sensor connection for analog reading](img/B16413_Figure_8.10.jpg)

Figure 8.10 – Schematics for the MQ-2 sensor connection for analog reading

Let's recap. In this section, we learned how to connect the hardware components to create our gas sensor device. You learned how to connect the MQ-2 sensor to the STM32 Blue Pill microcontroller board to obtain its data in two ways: digitally and in analog form.

In the next section, we will create the C code that obtains the MQ-2 sensor data from the STM32 Blue Pill microcontroller.

# Writing a program to read the gas concentration over the sensor board

In this section, we will learn how to code a program to read data from our gas sensor and show it on the serial monitor if gas is present in the environment.

As in the previous section, we'll first learn how to read data digitally and also in analog form.

## Coding for digital reading

Let's start writing the code:

1.  Define which pin of the STM32 Blue Pill microcontroller will be used as input for reading the data from the sensor. Here's the code that shows how to do that:

    ```cpp
    const int sensorPin = PB12;
    boolean sensorValue = true;
    ```

    The selected pin was `PB12` (labeled B12 on the Blue Pill board). A Boolean variable was declared and initialized to `true`. This variable will be used for storing the sensor data.

2.  Next, in the `setup()` part, we need to start the serial data transmission and assign the speed of the transfer (`9600` bps as a standard value):

    ```cpp
    void setup() {
      Serial.begin(9600);
    }
    ```

3.  Indicate to the microcontroller the type of pin assigned to `PB12`:

    ```cpp
    void setup() {
      Serial.begin(9600);
      pinMode(sensorPin, INPUT);
    }
    ```

4.  Now comes `loop()` with the rest of the sketch. The first lines read the input pin's data sensor and display its value in the serial console:

    ```cpp
    void loop() {
      sensorValue = digitalRead(sensorPin);
      Serial.print("Sensor value: ");
      Serial.println(sensorValue);
      if (sensorValue) {
        Serial.println("No gas present");
        delay(1000);
      } else  {
        Serial.println("Gas presence detected");
        delay(1000);
      }
    }
    ```

    The value read from the sensor could be `TRUE` or `FALSE`; remember, we are reading a digital value. If the value is `TRUE`, then gas is not present in the environment; otherwise, gas was detected. This behavior occurs because the MQ-2 sensor has a negated output; the module's LED must also light up in this state since it is internally with a 5 V resistance. When there is no presence of gas, the LED turns off, and the output is logic 1 (5 V).

    The code for digital reading is now complete. You can find the complete sketch available in the `Chapter8/gas_digital` folder in the GitHub repository.

Now we have the complete code for reading the DO of the MQ-2 sensor. You can upload it to the STM32 microcontroller. You can now see, in the **serial monitor**, the sensor readings as shown in *Figure 8.11*. The most normal thing is for the reading to indicate no presence of any gas:

![Figure 8.11 – Serial monitor reading the DO of the sensor with no gas presence](img/B16413_Figure_8.11.jpg)

Figure 8.11 – Serial monitor reading the DO of the sensor with no gas presence

Now, **being very careful about fire safety**, bring a lit match to the sensor and put it out when close to the sensor to generate smoke. The serial monitor will change as soon as the smoke impregnates the sensor (as shown in *Figure 8.12*):

![Figure 8.12 – Serial monitor reading the DO of the sensor with the presence of gas](img/B16413_Figure_8.12.jpg)

Figure 8.12 – Serial monitor reading the DO of the sensor with the presence of gas

As we can see, it is like reading any digital input. The sensitivity of the sensor is configured through the variable resistance included in the breakout module. Turning to the right becomes more sensitive, and we need less gas present to activate the output. In the same way, if we turn it to the left, a more significant presence of gas will be needed to activate the output.

So far, we have learned how to read the gas sensor in digital form. In the following subsection, we are going to obtain its reading from the AO.

## Coding for analog reading

When using the AO, different levels of gas presence are obtained. The module has a heating chamber in which the gas enters. This gas will continue to be detected until the chamber is empty. The sensor's voltage output will be proportional to the gas concentration in the chamber.

In Short, the higher the gas concentration, the higher the voltage output, and the lower the gas concentration, the lower the voltage output.

Let's get started with the code:

1.  Create a copy of the `Chapter8/gas_digital` project and change the name to `Chapter8``/gas_analog`. Remember to rename the folder and the INO file.
2.  Change the sensor pin to `0` (labeled A0 on the Blue Pill), remove the Boolean variable, and assign a threshold level for the sensor readings. We will use a value of `800`, to be sure the sensor has gas in its chamber:

    ```cpp
    const int sensorPin = 0;
    const int gasThreshold = 800;
    ```

3.  Keep `setup()` without modifications:

    ```cpp
    void setup() {
      Serial.begin(9600);
      pinMode(sensorPin, INPUT);
    }
    ```

4.  The code in `loop()` will be using the same logic but with a few changes:

    ```cpp
    void loop() {
      int sensorValue = analogRead(sensorPin);
      Serial.print("Sensor value: ");
      Serial.println(sensorValue);
      if (sensorValue > gasThreshold) {
        Serial.println("Gas presence detected");
      } else {
        Serial.println("No gas present");
      }
      delay(1000);
    }
    ```

5.  To read the sensor value, we use the `analogRead()` function. The value read is stored in the `sensorValue` variable; the next step will be to compare its value with the threshold. If the sensor value is higher than the threshold, this means gas was detected.

Now that our sketch is complete, upload it to the Blue Pill board. To test that our project works, just like the digital reading version, bring a lit match to the sensor and put it out when close to the sensor to generate smoke. Please, do not forget to be very careful about fire safety. *Figure 8.13* shows the serial monitor when smoke starts to be detected:

![Figure 8.13 – Serial monitor reading the AO of the sensor with gas present](img/B16413_Figure_8.13.jpg)

Figure 8.13 – Serial monitor reading the AO of the sensor with gas present

This section helped us to learn how to create code in C to read the data from the MQ-2 sensor to know if there is a concentration of gas or smoke in the environment. In addition, the skills to read the sensor value in both analog and digital form were acquired. In the next section, we will create a simple way of knowing directly in the hardware device if there is gas or smoke concentration without seeing the serial monitor on a computer.

# Testing the system

In this last section of the chapter, we will connect an **8x8 LED matrix** to display an alert if the sensor detects the presence of gas in the environment.

An LED matrix is a set of LEDs grouped into rows and columns. By turning on these LEDs, you can create graphics or text, which are widely used for billboards and traffic signs.

There is an electronic component for small-scale projects called an 8x8 LED matrix. It is composed of 64 LEDs arranged in eight rows and eight columns (see *Figure 8.14*):

![Figure 8.14 – LED matrix 8x8](img/B16413_Figure_8.14.jpg)

Figure 8.14 – LED matrix 8x8

As you can see in the previous figure, the 8x8 LED matrix has pins to control the rows and columns, so it is impossible to control each LED independently.

This limitation implies having to use 16 digital signals and refreshes the image or text continuously. Therefore, the integrated MAX7219 and MAX7221 circuits have been created to facilitate this task; the circuits are almost identical and interchangeable using the same code.

In addition to these integrated circuits, breakout modules have been created integrating the 8x8 LED matrix and the MAX7219 circuit, in addition to having output connectors to put several modules in a cascade. *Figure 8.15* shows the 8x8 LED matrix breakout module:

![Figure 8.15 – LED matrix 8x8 breakout module](img/B16413_Figure_8.15.jpg)

Figure 8.15 – LED matrix 8x8 breakout module

The input pins of the module are as follows:

*   **VCC**: Module power supply
*   **GND**: Ground connection
*   **DIN**: Serial data input
*   **CS**: Chip select input
*   **CLK**: Serial clock input

The output pins are almost identical, only instead of **DIN** there is **DOUT**, which will allow cascading with other modules, but we will not learn about this functionality in this chapter.

*Figure 8.16* shows how to connect the MAX7219 8x8 LED matrix module to our STM32 Blue Pill board:

![Figure 8.16 – LED matrix 8x8 breakout module interfacing to the STM32 Blue Pill](img/B16413_Figure_8.16.jpg)

Figure 8.16 – LED matrix 8x8 breakout module interfacing to the STM32 Blue Pill

Now it is time to create the code to display an alert on our LED matrix. We will update the `Chapter8/gas_digital` sketch. Let's start coding!

1.  To make the process easier, we will use a library called `LedControlMS` that facilitates the use of the 8x8 LED matrix module. To start with the installation, download the library from our GitHub: `Chapter8/library`.
2.  To install, go to the **Sketch** menu | **Include Library** | **Add .ZIP Library…** (see *Figure 8.17*) and select the downloaded file, and it is ready to be used:![Figure 8.17 – Adding the LedControlMS library](img/B16413_Figure_8.17.jpg)

    Figure 8.17 – Adding the LedControlMS library

3.  In our script, we are going to add the library:

    ```cpp
    #include "LedControlMS.h";
    ```

4.  We must indicate the number of display modules that we are using; in our case, it is one. We will initialize the library, pointing out the pins of the STM32 Blue Pill board to which the module will be connected, as well as the variable with the number of modules:

    ```cpp
    const int numDisplays = 1;
    const int sensorPin = PB12;
    boolean sensorValue = true;
    LedControl lc = LedControl(7, 8, 5, numDisplays);
    ```

5.  By default, the matrix is in power-saving mode, so it is necessary to wake it up. If it were more than one module, a loop would be required, but in this case, it is only one, so we will do it directly:

    ```cpp
    void setup() {
      Serial.begin(9600);
      pinMode(sensorPin, INPUT);
      0 in the code refers to the first array of a possible set of interconnected arrays.
    ```

6.  Finally, we write the character we want to show. We will use the `writeString()` function in the `else` statement to indicate in the LED matrix that there is gas; we will show a letter `A` representing an alert:

    ```cpp
    void loop() {
      sensorValue = digitalRead(sensorPin);
      Serial.print("Sensor value: ");
      Serial.println(sensorValue);
      if (sensorValue) {
        Serial.println("No gas present");
        delay(1000);
      } else  {
        Serial.println("Gas presence detected");
        lc.writeString(0, "A");
        delay(1000);
      }
    }
    ```

    We are ready to upload our script to the microcontroller and test that the system works. As in the previous section, to test it, bring a lit match to the sensor and put it out when close to the sensor to generate smoke. Again, do not forget to be very careful about fire safety. *Figure 8.18* shows the complete gas sensing device, including the sensor and LED matrix module connected to the STM32 microcontroller:

![Figure 8.18 – Gas sensor device](img/B16413_Figure_8.18.jpg)

Figure 8.18 – Gas sensor device

Until now, in this section, we have learned how to handle an 8x8 LED matrix and use it to have a visual alert on our gas sensor device.

In this chapter, we learned how to read to code programs to read a gas sensor in a digital and analog way. This has allowed us to reinforce our knowledge of data acquisition from sensors in different forms of outputs. This knowledge will empower us to create more complex embedded systems, such as automating homes using sensors in the environment.

# Summary

We had so much to learn in this chapter! First, we learned how to connect the MQ-2 gas sensor to the STM32 Blue Pill microcontroller board, both digitally and with an AO reading. We then wrote two pieces of code to read digital and analog sensor values. Last, we tested the device to understand its operation, displaying the sensor data in the serial console.

This project gave us the skills to read different kinds of sensor data to use this knowledge according to our needs. For instance, you can display some sensors in a room to monitor the environment in real-time.

In the next chapter, we will enter the fascinating world of the so-called Internet of Things. With the knowledge that we will acquire, we will create projects that connect to the internet and access our information remotely.

# Further reading

Marques G. & Pitarma R. (2017). *Monitoring Health Factors in Indoor Living Environments Using Internet of Things.* In: Rocha Á., Correia A., Adeli H., Reis L., & Costanzo S. (eds) Recent Advances in Information Systems and Technologies. WorldCIST 2017\. Advances in Intelligent Systems and Computing, vol. 570\. Springer, Cham. [https://doi.org/10.1007/978-3-319-56538-5_79](https://doi.org/10.1007/978-3-319-56538-5_79)