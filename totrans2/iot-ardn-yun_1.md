# Chapter 1. Building a Weather Station Connected to the Cloud

This chapter will introduce you to the powerful features of the Arduino Yún microcontroller board. In this chapter, you will learn how to create a simple weather station that will send data to the cloud using the features of the web-based service **Temboo**. Temboo is not 100 percent free, but you will be able to make 1000 calls to Temboo per month using their free plan. You will learn how to connect sensors that measure temperature, humidity, and light level to your Arduino Yún. These sensors will first be separately tested to make sure that the hardware connections you made are correct.

Then, we are going to use the Temboo Arduino libraries to send these measurements to the cloud and to different web services so that they can be accessed remotely regardless of where you are in the world. Temboo is a web-based service that allows you to connect different web services together and proposes ready-to-use libraries for the Arduino Yún.

For example, the first thing we are going to do with Temboo is to send the data from your measurements to a Google Docs spreadsheet, where they will be logged along with the measurement data. Within this spreadsheet, you will be able to plot this data right in your web browser and see the data that arrives getting stored in your Google Docs account.

Then, we will use Temboo again to send an automated e-mail based on the recorded data. For example, you would like to send an alert when the temperature drops below a certain level in your home, indicating that a heater has to be turned on.

Finally, we will finish the chapter by using Temboo to post the data at regular intervals on a Twitter account, for example, every minute. By doing this, we can have a dedicated Twitter account for your home that different members of your family can follow to have live information about your home.

After completing this chapter, you'll be able to apply what you learned to other projects than just weather-related measurements. You can apply what you see in this chapter to any project that can measure data, in order to log this data on the Web and publish it on Twitter.

The Arduino Yún board is shown in the following image:

![Building a Weather Station Connected to the Cloud](img/8007OS_01_01.jpg)

# The required hardware and software components

Of course, you need to have your Arduino Yún board ready on your desk along with a micro USB cable to do the initial programming and testing. Also, we recommend that you have a power socket to the micro USB adapter so that you can power on your Arduino Yún directly from the wall without having your computer lying around. This will be useful at the end of the project, as you will want your Arduino Yún board to perform measurements autonomously.

You will then need the different sensors which will be used to sense data about the environment. For this project, we are going to use a DHT11 sensor to measure temperature and humidity and a simple photocell to measure light levels. DHT11 is a very cheap digital temperature and humidity sensor that is widely used with the Arduino platform. You can also use a DHT22 sensor, which is more precise, as the Arduino library is the same for both sensors. There are several manufacturers for these sensors, but you can find them easily, for example, on SparkFun or Adafruit. For the photocell, you can use any brand that you wish; it just needs to be a component that changes its resistance according to the intensity of the ambient light.

To make the DHT11 sensor and photocell work, we will need a 4.7k Ohm resistor and a 10k Ohm resistor as well. You will also need a small breadboard with at least two power rails on the side and some male-male jumper wires to make the electrical connections between the different components.

On the software side, you will need the latest beta version of the Arduino IDE, which is the only IDE that supports the Arduino Yún board (we used Version 1.5.5 when doing this project). You will also need the DHT library for the DHT11 sensor, which can be downloaded from [https://github.com/adafruit/DHT-sensor-library](https://github.com/adafruit/DHT-sensor-library).

To install the library, simply unzip the files and extract the `DHT` folder to your `libraries` folder in your main Arduino folder.

# Connecting the sensors to the Arduino Yún board

Before doing anything related to the Web, we will first make sure that our hardware is working correctly. We are going to make the correct hardware connections between the different components and write a simple Arduino sketch to test all these sensors individually. By doing this, we will ensure that you make all the hardware connections correctly, and this will help a lot if you encounter problems in the next sections of this chapter that use more complex Arduino sketches.

The hardware connections required for our project are actually quite simple. We have to connect the DHT11 sensor and then the part responsible for the light level measurement with the photocell by performing the following steps:

1.  First, we connect the Arduino Yún board's +5V pin to the red rail on the breadboard and the ground pin to the blue rail.
2.  Then, we connect pin number 1 of the DHT11 sensor to the red rail on the breadboard and pin number 4 to the blue rail. Also, connect pin number 2 of the sensor to pin number 8 of the Arduino Yún board.
3.  To complete the DHT11 sensor connections, clamp the 4.7k Ohm resistor between pin numbers 1 and 2 of the sensor.

For the photocell, first place the cell in series with the 10k Ohm resistor on the breadboard. This pull-down resistor will ensure that during the operation, if there is no light at all, the voltage seen by the Arduino board will be 0V. Then, connect the other end of the photocell to the red rail on the breadboard and the end of the resistor to the ground. Finally, connect the common pin to the Arduino Yún board analog pin A0.

The following image made using the Fritzing software summarizes the hardware connections:

![Connecting the sensors to the Arduino Yún board](img/8007OS_01_02.jpg)

Now that the hardware connections are done, we will work on testing the sensors without uploading anything to the Web. Let's go through the important parts of the code.

First, we have to import the library for the DHT11 sensor, as follows:

[PRE0]

Then, we need to declare a couple of variables that will store the measurements, as shown in the following code. These variables are declared as floats because the DHT sensor library returns float numbers.

[PRE1]

Also, we can define the sensor pin and sensor type as follows:

[PRE2]

Create the DHT instance as follows:

[PRE3]

Now, in the `setup()` part of the sketch, we need to start the serial connection, as follows:

[PRE4]

Next, in order to initialize the DHT sensor, we have the following:

[PRE5]

In the `loop()` part, we are going to perform the different measurements. First, we will calculate the temperature and humidity, as follows:

[PRE6]

Then, measure the light level, as follows:

[PRE7]

Finally, we print all the data on the serial monitor, as shown in the following code:

[PRE8]

Repeat this every 2 seconds, as shown:

[PRE9]

The complete sketch for this part can be found at [https://github.com/openhomeautomation/geeky-projects-yun/tree/master/chapter1/sensors_test](https://github.com/openhomeautomation/geeky-projects-yun/tree/master/chapter1/sensors_test).

Now it's time to test the sketch and upload it to the Arduino board. Then, open the serial monitor and you should have the data that comes from the sensors being displayed, as shown in the following screenshot:

![Connecting the sensors to the Arduino Yún board](img/8007OS_01_03.jpg)

If you can see the different measurements being displayed as in the previous screenshot, it means that you have made the correct hardware connections on your breadboard and that you can proceed to the next sections of this chapter.

If it is not the case, please check all the connections again individually by following the instructions in this section. Please make sure that you haven't forgotten the 4.7k Ohm resistor with the DHT sensor, as the measurements from this sensor won't work without it.

### Tip

**Downloading the example code**

You can download the example code files for all Packt books you have purchased from your account at [http://www.packtpub.com](http://www.packtpub.com). If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files e-mailed directly to you.

All the up-to-date code for the four projects of this book can also be found at [https://github.com/openhomeautomation/geeky-projects-yun](https://github.com/openhomeautomation/geeky-projects-yun).

# Creating a Temboo account

The next step in this project is to create and set up an account on the web service Temboo, so you can use the wide range of services provided by Temboo to upload data to Google Docs and to use their Gmail and Twitter libraries. This account will actually be used in the whole book for the other projects as well.

To do so, the first step is to simply go to the Temboo website at [http://temboo.com/](http://temboo.com/).

On the main page, simply enter your e-mail address to register and click on **Sign up**, as shown in the following screenshot:

![Creating a Temboo account](img/8007OS_01_04.jpg)

You will then be asked to enter some basic information about your account, such as your account name, as shown in the following screenshot:

![Creating a Temboo account](img/8007OS_01_05.jpg)

Then, you will be prompted to create your first app. Ensure that you save the details of your account, such as the name of your first app and the key that will be given to you; we are going to need it for the rest of this book.

In case you need to get these values again or create a new application, you can always access this data in the **My Account** section of the Temboo website by clicking on the **MANAGE** button below **APPLICATIONS**, just as it is displayed in the following screenshot:

![Creating a Temboo account](img/8007OS_01_06.jpg)

We are now all set to start using the Temboo libraries that are made specifically for the Arduino Yún board and to upload some data to the cloud.

## Sending data to Google Docs and displaying it

In this section, we are going to use our first Temboo library (called a **Choreo**) to upload the measurements of the Arduino Yún to the Web and log the data into a Google Docs spreadsheet.

First, let's have a look at what a Choreo is and how you can generate the code for your Arduino Yún board. If you go to the main Temboo page, you will see that you can choose different platforms and languages, such as Arduino, JavaScript, or Python. Each of these links will allow you to select a Choreo, which is a dedicated library written for the platform you chose and can interface with a given web service such as Google Docs.

For the Arduino platform, Temboo even offers to generate the entire code for you. You can click on the Arduino icon on the Temboo website and then click on Arduino Yún; you will get access to a step-by-step interface to generate the code. However, as we want to get complete control of our device and write our own code, we won't use this feature for this project.

Google Docs is really convenient as it's an online (and free) version of the popular Office software from Microsoft. The main difference is that because it's all in the cloud, you don't have to store files locally or save them—it's all done online. For our project, the advantage is that you can access these files remotely from any web browser, even if you are not on your usual computer. You just need your Google account name and password and can access all your files.

If you don't have a Google account yet, you can create one in less than five minutes at [https://drive.google.com/](https://drive.google.com/).

This will also create an account for the Gmail service, which we will also use later. Please make sure that you have your Google Docs username and password as you are going to need them soon.

Before we start writing any Arduino code, we need to prepare a Google Docs spreadsheet that will host the data. Simply create a new one at the root of your Google Docs account; you can name it whatever you wish (for example, `Yun`). This is done from the main page of Google Docs just by clicking on **Create**.

In the spreadsheet, you need to set the name of the columns for the data that will be logged; that is, `Time`, `Temperature`, `Humidity`, and `Light level`. This is shown in the following screenshot:

![Sending data to Google Docs and displaying it](img/8007OS_01_07.jpg)

Now, let's start building the Arduino sketch inside the Arduino IDE. We first need to import all the necessary libraries, as follows:

[PRE10]

The `Bridge` library is something that was introduced for the Arduino Yún board and is responsible for making the interface between the Linux machine of the Yún and the Atmel processor, where our Arduino sketch will run. With this library, it's possible to use the power of the Linux machine right inside the Arduino sketch.

The `Process` library will be used to run some programs on the Linux side, and the Temboo file will contain all the information that concerns your Temboo account. Please go inside this file to enter the information corresponding to your own account. This is as shown in the following code:

[PRE11]

### Note

Note that we also included a debug mode in the sketch that you can set to `true` if you want some debug output to be printed on the serial monitor. However, for an autonomous operation of the board, we suggest that you disable this debugging mode to save some memory inside Yún.

In the sketch, we then have to enter the Google Docs information. You need to put your Google username and password here along with the name of the spreadsheet where you want the data to be logged, as shown in the following code:

[PRE12]

In the `setup()` part of the sketch, we are now starting the bridge between the Linux machine and the Atmel microcontroller by executing the following line of code:

[PRE13]

We are also starting a date process so that we can also log the data of when each measurement was recorded, as shown in the following code:

[PRE14]

The date will be in the format: date of the day followed by the time. The date process we are using here is actually a very common utility for Linux, and you can look for the documentation of this function on the Web to learn more about the different date and time formats that you can use.

Now, in the `loop()` part of the sketch, we send the measurements continuously using the following function:

[PRE15]

Let's get into the details of this function. It starts by declaring the Choreo (the Temboo service) that we are going to use:

[PRE16]

The preceding function is specific to Google Docs spreadsheets and works by sending a set of data separated by commas on a given row. There are Choreos for every service that Temboo connects to, such as Dropbox and Twitter. Please refer to the Temboo documentation pages to get the details about this specific Choreo. After declaring the Choreo, we have to add the different parameters of the Choreo as inputs. For example, the Google username, as shown in the following line of code:

[PRE17]

The same is done with the other required parameters, as shown in the following code:

[PRE18]

The important part of the function is when we actually format the data so that it can be appended to the spreadsheet. Remember, the data needs to be delimited using commas so that it is appended to the correct columns in the spreadsheet, as shown in the following code:

[PRE19]

The Choreo is then executed with the following line of code:

[PRE20]

The function is then repeated every 10 minutes. Indeed, these values usually change slowly over the course of a day, so this is useless to the data that is logging continuously. Also, remember that the number of calls to Temboo is limited depending on the plan you chose (1000 calls per month on a free plan, which is approximately 1 call per hour). This is done using the delay function, as follows:

[PRE21]

For demonstration purposes, the data is logged every 10 minutes. However, you can change this just by changing the argument of the `delay()` function. The complete code for this part can be found at [https://github.com/openhomeautomation/geeky-projects-yun/tree/master/chapter1/temboo_log](https://github.com/openhomeautomation/geeky-projects-yun/tree/master/chapter1/temboo_log).

You can now upload the sketch to the Arduino Yún board and open the Google Docs spreadsheet to see what's happening. It's all synchronized live with the Google Docs servers, so you do not need to refresh anything. After a while, you should see the first set of measurements being logged, as shown in the following screenshot:

![Sending data to Google Docs and displaying it](img/8007OS_01_08.jpg)

In order to show you what can be done using this project, we used the integrated chart capabilities of Google Docs to plot this data using the measurements that we obtained for over 24 hours. The following screenshot is an extract from the raw data:

![Sending data to Google Docs and displaying it](img/8007OS_01_09.jpg)

Now, to actually plot some data, you can simply use the **Insert charts** function of Google Docs. We chose the simple **Line** graph for our data. The following screenshot shows the results for temperature and humidity:

![Sending data to Google Docs and displaying it](img/8007OS_01_10.jpg)

We did the same for light level measurements, as shown in the following screenshot:

![Sending data to Google Docs and displaying it](img/8007OS_01_11.jpg)

These charts can be placed automatically in their respective sheets inside your spreadsheet and will, of course, be updated automatically as new data comes in. You can also use the sharing capabilities of Google Docs to share these sheets with anyone, so they can also follow the measurements of your home.

## Creating automated e-mail alerts

In this part, we are not only going to build on what we did in the previous section with Google Docs but also create some automated e-mail alerts on top with a Google account. This time, we will use the Temboo library that interfaces directly with Gmail, in this case, to automatically send an e-mail using your account.

What we will do is program the Arduino Yún board to send an e-mail to the chosen address if the temperature goes below a given level, for example, indicating that you should turn on the heating in your home.

Compared to the previous Arduino sketch, we need to add the destination e-mail address. I used my own address for testing purposes, but of course, this destination address can be completely different from the one of your Gmail account. For example, if you want to automatically e-mail somebody who is responsible for your home if something happens, execute the following line of code:

[PRE22]

Please note that sending an e-mail to yourself might be seen as spam by your Gmail account. So, it's advisable to send these alerts to another e-mail of your choice, for example, on a dedicated account for these alerts. We also need to set a temperature limit in the sketch. In my version of the project, it is the temperature under which the Arduino Yún will send an e-mail alert, but you can of course modify the meaning of this temperature limit, as shown in the following line of code:

[PRE23]

In the `loop()` part of the sketch, what changes compared to the sketch of the previous section is that we can compare the recorded temperature to the limit. This is done with a simple `if` statement:

[PRE24]

Then, the alert mechanism occurs in the new function called `sendTempAlert` that is called if the temperature is below the limit. The function also takes a string as an argument, which is the content of the message that will be sent when the alert is triggered. Inside the function, we start again by declaring the type of Choreo that we will use. This time, the Choreo that we will use is specific to Gmail and is used to send an e-mail with the subject and body of the message, as shown in the following line of code:

[PRE25]

Just as the Choreo we used to log data into Google Docs, this new Choreo requires a given set of parameters that are defined in the official Temboo documentation. We need to specify all the required inputs for the Choreo, for example, the e-mail's subject line that you can personalize as well, as shown in the following line of code:

[PRE26]

The body of the message is defined in the following line of code:

[PRE27]

Note that the `message` variable is the one passed in the `loop()` part of the sketch and can be personalized as well, for example, by adding the value of the measured temperature. Finally, the Choreo is executed with the following line of code:

[PRE28]

The complete code for this part can be found at [https://github.com/openhomeautomation/geeky-projects-yun/tree/master/chapter1/temboo_alerts](https://github.com/openhomeautomation/geeky-projects-yun/tree/master/chapter1/temboo_alerts).

Now, you can compile and update the sketch to your Yún. You can also go to the Gmail interface to check for new e-mails. If the temperature indeed drops below the value that you set as a limit, the following is what you should receive in your inbox:

![Creating automated e-mail alerts](img/8007OS_01_12.jpg)

Again, you can play with this sketch and create more complex alerts based on the data you measured. For example, you can add the humidity and light level in the mix and create dedicated limits and alerts for these values. You can also program Arduino Yún so that it e-mails you the data itself at regular intervals, even if no temperature limit is reached.

# Making your Arduino Yún board tweet sensor data

Finally, in the last part of this project, we will make your Arduino Yún board send its own messages on Twitter. You can even create a new Twitter account just for your Yún board, and you can tell people you know to follow it on Twitter so that they can be informed at all times about what's going on in your home!

The project starts on the Twitter website because you have to declare a new app on Twitter. Log in using your Twitter credentials and then go to [https://apps.twitter.com/](https://apps.twitter.com/).

Now, click on **Create New App** to start the process, as shown in the following screenshot:

![Making your Arduino Yún board tweet sensor data](img/8007OS_01_13.jpg)

You will need to give some name to your app. For example, we named ours `MyYunTemboo`. You will need to get a lot of information from the Twitter website. The first things you need to get are the API key and the API secret. These are available in the **API Keys** tab, as shown in the following screenshot:

![Making your Arduino Yún board tweet sensor data](img/8007OS_01_14.jpg)

Make sure that the **Access level** of your app is set to **Read**, **Write**, and **Direct** messages. This might not be active by default, and in the first tests, my Arduino board did not respond anymore because I didn't set these parameters correctly. So, make sure that your app has the correct access level.

Then, you are also going to need a token for your app. You can do this by going to the **Your access token** section. From this section, you need to get the **Access token** and the **Access token secret**. Again, make sure that the access level of your token is correctly set.

We can now proceed to write the Arduino sketch, so the Arduino Yún board can automatically send tweets. The Twitter Choreo is well known for using a lot of memory on the Yún, so this sketch will only tweet data without logging data into your Google Docs account. I also recommend that you disable any debugging messages on the serial port to preserve the memory of your Yún. In the sketch, you first need to define your Twitter app information, as shown in the following code:

[PRE29]

Then, the sketch will regularly tweet the data about your home with the following function:

[PRE30]

This function is repeated every minute using a `delay()` function, as follows:

[PRE31]

Of course, this delay can be changed according to your needs. Let's see the details of this function. It starts by declaring the correct Choreo to send updates on Twitter:

[PRE32]

Then, we build the text that we want to tweet as a string. In this case, we just formatted the sensor data in one string, as shown in the following code:

[PRE33]

The access token and API key that we defined earlier are declared as inputs:

[PRE34]

The text that we want to tweet is also simply declared as an input of the Twitter Choreo with the string variable we built earlier:

[PRE35]

The complete code for this part can be found at [https://github.com/openhomeautomation/geeky-projects-yun/tree/master/chapter1/temboo_twitter](https://github.com/openhomeautomation/geeky-projects-yun/tree/master/chapter1/temboo_twitter).

Now that the Arduino sketch is ready, we can test it. You can simply upload the code to your Arduino Yún, and wait for a moment. Your board should automatically connect to the Twitter feed that you chose and print the data as a new message, as shown in the following screenshot:

![Making your Arduino Yún board tweet sensor data](img/8007OS_01_15.jpg)

If nothing shows up on your Twitter account, there are several things that you can check. I already mentioned memory usage; try to disable the debug output on the serial port to free some memory. Also, make sure that you have entered the correct information about your Twitter app; it is quite easy to make a mistake between different API keys and access tokens.

For this project, I used the Twitter account of my website dedicated to home automation, but of course, you can create a dedicated Twitter account for the project so that many people can follow the latest updates about your home!

You can also combine the code from this part with the idea of the previous section, for example, to create automated alerts based on the measured data and post messages on Twitter accordingly.

# Summary

Let's summarize what we learned in this chapter. We built a simple weather measurement station based on the Arduino Yún board that sends data automatically into the cloud.

First, you learned how to connect simple sensors to your Arduino Yún board and to write a test sketch for the Yún board in order to make sure that all these sensors are working correctly.

Then, we interfaced the Arduino Yún board to the Temboo services by using the dedicated Temboo libraries for the Yún. Using these libraries, we logged data in a Google Docs spreadsheet, created automated e-mail alerts based on our measurements, and published these measurements on Twitter.

To take it further, you can combine the different parts of this project together, and also add many Arduino Yún boards to the project, for example, in two different areas of your home. In the next chapter, we are going to use the power of the Temboo libraries again to send power measurement data to the Web, so the energy consumption of your home can be monitored remotely.