# Intel Edison and IoT (Home Automation)

In [Chapter 2](bada9944-ec60-4e8f-8d88-0085dd1c8210.xhtml), *Weather Station (IoT)*, we dealt with transferring data from Edison to the cloud platform. Here, in this chapter, we'll be doing just the opposite. We'll be controlling devices using the Internet. When we talk about IoT, the first thing that usually comes to mind is home automation. Home automation is basically controlling and monitoring home electrical appliances using an interface, which may be a mobile application, a web interface, a wall touch unit, or more simply, your own voice. So, here in this chapter, we'll be dealing with the various concepts of home automation using the MQTT protocol; then, we'll be controlling an electrical load with an Android application and a **Windows Presentation Foundation** (**WPF**) application using the MQTT protocol. Some of the topics that we will discuss are:

*   The various concepts of controlling devices using the Internet MQTT protocol

*   Using Edison to push data and get data using the MQTT protocol

*   LED control using the MQTT protocol

*   Home automation use cases using the MQTT protocol

*   The controller application in Android (MyMqtt) and in WPF (to be developed)

This chapter will use a companion application named MyMqtt, which can be downloaded from the Play Store. Credit goes to the developer (Instant Solutions) for developing the application and uploading it to the Play Store for free. MyMqtt can be found here: [h](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[t](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[t](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[p](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[s](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[://p](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[l](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[a](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[y](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[.](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[g](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[o](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[o](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[g](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[l](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[e](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[.](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[c](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[o](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[m](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[/s](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[t](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[o](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[r](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[e](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[/a](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[p](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[p](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[s](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[/d](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[e](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[t](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[a](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[i](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[l](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[s](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[?i](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[d](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[=a](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[t](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[.](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[t](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[r](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[i](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[p](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[w](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[i](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[r](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[e](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[.](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[m](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[q](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[t](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[t](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[.](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[c](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[l](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[i](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[e](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[n](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[t](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[&h](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[l](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[=e](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)[n](https://play.google.com/store/apps/details?id=at.tripwire.mqtt.client&hl=en)

We are going to develop our own controller for PC as a WPF application that will implement the protocol and control your Edison.

To develop the WPF application, we are going to use Microsoft Visual Studio. You can download it at [h](https://msdn.microsoft.com/)[t](https://msdn.microsoft.com/)[t](https://msdn.microsoft.com/)[p](https://msdn.microsoft.com/)[s](https://msdn.microsoft.com/)[://m](https://msdn.microsoft.com/)[s](https://msdn.microsoft.com/)[d](https://msdn.microsoft.com/)[n](https://msdn.microsoft.com/)[.](https://msdn.microsoft.com/)[m](https://msdn.microsoft.com/)[i](https://msdn.microsoft.com/)[c](https://msdn.microsoft.com/)[r](https://msdn.microsoft.com/)[o](https://msdn.microsoft.com/)[s](https://msdn.microsoft.com/)[o](https://msdn.microsoft.com/)[f](https://msdn.microsoft.com/)[t](https://msdn.microsoft.com/)[.](https://msdn.microsoft.com/)[c](https://msdn.microsoft.com/)[o](https://msdn.microsoft.com/)[m](https://msdn.microsoft.com/).

# Controlling devices using the Internet - concepts

When it comes to control devices using the Internet, some key factors come to play. Firstly, is the technique to be used. There are lot of techniques in this field. A quick workaround is the use of REST services, such as HTTP `GET` requests, where we get data from an existing database.

Some of the workarounds are discussed here.

# REST services

One of the most commonly-used techniques for obtaining the desired data is by an HTTP `GET` call. Most of the IoT platforms that exist in the market have REST APIs exposed. There, we can send values from the device to the platform using an HTTP `POST` request, and at the same time get data by an HTTP `GET` request. Infact, in [Chapter 2](bada9944-ec60-4e8f-8d88-0085dd1c8210.xhtml), *Weather Station (IoT)*, where we used `dweet.io` to send data from a device, we used an SDK. Internally, the SDK also performs a similar HTTP `POST` call to send in data.

# Instructions or alerts (present on most IoT platforms)

In certain IoT platforms, we have certain ready-made solutions where we just need to call a certain web service and the connection is established. Internally, it may use REST APIs, but for the benefit of the user, they have come up with their own SDK where we implement.

Internally, a platform may follow either a REST call, MQTT, or Web Sockets. However, we just use an SDK where we don't implement it directly, and by using the platform's SDK, we are able to establish a connection. It is entirely platform-specific. Here, we are discussing one of the workarounds,where we use the MQTT protocol to control our devices directly without the use of any IoT platforms.

# Architecture

In a typical system, the IoT platform acts as a bridge between the user and the protocols to the controller, as shown in the following figure:

![](img/image001.jpg)

Architecture of the IoT system for controlling devices

The preceding image depicts a typical workflow or architecture of controlling devices using the Internet. It is to be noted that the user may directly control the controller without the use of an IoT platform, as we do here. However, normally a user will use the IoT platform, which also provides more enhanced security. The user may use any web interface, mobile application, or a wall control unit to control the device using any standard protocol. Here in the image, only REST, MQTT, and Web Sockets are included. However, there are more protocols that can be used, such as the AMQP protocol, the MODBUS protocol, and so on. The choice of the protocol depends mainly on how sensitive the system is and how stable the system needs to be.

# MQTT protocol overview

The MQTT protocol is based on the publish-subscribe architecture. It's a very lightweight protocol, where message exchange happens asynchronously. The main usage of the MQTT protocol is in places of low bandwidth and low processing power. A small code footprint is required for establishing an MQTT connection. Every communication in the MQTT protocol happens through a medium called a broker. The broker is either subscribed or published. If you want the data to flow from Edison to a server, then you publish the data via the broker. A dashboard or an application subscribes to the broker with the channel credentials and provides the data. Similarly, when we control the device from any application, Edison will act as a subscriber and our application will act as a publisher. That's how the entire system works out. The following screenshot explains the concept:

![](img/image002.jpg)

Overflow where Edison acts as a publisher

In the preceding screenshot, we see Edison acting as a publisher. This is one type of use case, where we need to send data from Edison, as with a similar example shown in [Chapter 2](bada9944-ec60-4e8f-8d88-0085dd1c8210.xhtml), *Weather Station (IoT)*. The application will get the data and act as a publisher. The following screenshot depicts the use case that will be used in this chapter: the use of Edison as a subscriber:

![](img/image003.jpg)

Overflow where Edison acts as a subscriber

In the preceding case, we have some controls on the application. These controls send signals to Edison via the MQTT broker. Now, in this case, the application will act as a publisher and Edison acts as a subscriber.

It is to be noted that in a single system, you can make the endpoint (device or application) act both as a publisher as well as a subscriber. This occurs when we want to get some data from the IoT device, such as the Intel Edison, and also control the device in emergency cases. The same may also occur when we need to control the home's electrical appliances, as well as monitor them remotely. Although most systems are deployed based on a closed loop feedback control, there is always room to monitor them remotely, and at the same time have control based on feedback received from the sensors.

To implement the MQTT protocol, we are not going to set our own server but use an existing one. [https://iot.eclipse.org/](https://iot.eclipse.org/) has provided a sandbox server which will be used for the upcoming projects. We're just going to set up our broker and then publish and subscribe to the broker. For the Intel Edison side, we are going for Node.js and its related libraries. For the application end, we are going to use an already available application named MyMqtt for Android. If anyone wants to develop his or her own application, then you need to import the `paho` library to set up MQTT. We are also developing a PC application, where we will again use MQTT to communicate.

For details on the eclipse IoT project on MQTT and other standards, please refer to the following link:

[https://iot.eclipse.org/standards/](https://iot.eclipse.org/standards/)

In the following section, we'll set up and configure Edison for our project and also set up the development environment for the WPF application.

The paho project can be accessed through this link:

[https://eclipse.org/paho/](https://eclipse.org/paho/)

# Using Intel Edison to push data by using the MQTT protocol

As previously mentioned, this short section will show users how to push data from Edison to an Android device using the MQTT protocol. The following screenshot depicts the workflow:

![](img/image004.jpg)

Workflow of pushing data from the Edison to the Android application

From the preceding illustration, it is clear that we first obtain readings from the temperature sensor and then use the MQTT broker to push the readings to the Android application.

Firstly, we are going to connect the temperature sensor to Edison. Make a reference of the circuit from [Chapter 2](bada9944-ec60-4e8f-8d88-0085dd1c8210.xhtml), *Weather Station (IoT)*. After it is connected, fire up your editor to write the following Node.js code:

[PRE0]

The code written here is similar to what we used in the [Chapter 2](bada9944-ec60-4e8f-8d88-0085dd1c8210.xhtml), *Weather Station (IoT)*. Here, the difference is that we are not sending it to `dweet.io` but to the MQTT broker. We're publishing the data obtained to a particular channel in the MQTT broker.

However, to execute this code, you must have the MQTT dependency installed via `npm`. Type in the following command in the PuTTY console:

[PRE1]

This will install the MQTT dependency.

In the preceding code, we initially imported the required libraries or dependency. For this case, we need the `mraa` and the `mqtt` libraries:

[PRE2]

Then, we need to initialize the analog pin to read the temperature. After that, we convert the raw readings to the standard value.

We declare the client variable, which will handle the MQTT publish operation:

[PRE3]

Here, [https://iot.eclipse.org/](https://iot.eclipse.org/) is the free broker that we are using.

Next, in the `sendData` function, the initial temperature processing is computed before the data is published to the channel:

[PRE4]

The name of the channel is `avirup/temperature`. Please note the type of `sendTemp`. The initial processed value is obtained in the variable temperature. Here in `client.publish`, the second parameter has to be a string. Thus, we store the temperature value as a string type in `sendTemp`. Finally, we print the temperature into the console.

We have also provided a delay of 1 second. Now run this Node.js file using the `node` command.

The screenshot is as follows:

![](img/6639_03_01.png)

Output console log

As seen in the preceding screenshot, the log is displayed. Now we need to see this data in the Android MyMqtt application.

While carrying out this mini-project, as well as the later one to be discussed under MQTT, please change the channel name. One of my projects may be live and it could create an issue. One can go for the `NAME_OF_THE_USER/VARIABLE_NAME` convention.

Open up the MyMqtt application in Android and browse to Settings. There, in the field of Broker URL, insert `iot.eclipse.org`. You will have used this on your Node.js snippet as well:

![](img/6639_03_02.jpg)

Screenshot of MyMqtt—1

Next, go to the Subscribe option and enter your channel name based on your Node.js code. In our case, it was `avirup/temperature`:

![](img/6639_03_03.jpg)

Screenshot of MyMqtt—2

Click on Add to add the channel and then finally go to the dashboard to visualize your data:

![](img/6639_03_04.jpg)

Screenshot of MyMqtt—3

If your code on the device is running in parallel to this, then you should get live data feed in this dashboard.

So, now you can visualize the data that you are sending from Edison.

# Getting data to Edison by using MQTT

We have been talking about home automation controlling electrical loads, but everything has a starting point. The most basic kick-starter is controlling Edison over the Internet—that's what it's all about.

When you have a device that is controllable over the Internet, we recommend controlling the electrical loads. In this other mini-project, we are going to control a simple LED that is already attached to pin `13` of Intel Edison. There is no need for any external hardware for this, as we are using an in-built functionality. Now, open your editor and type in the following code:

[PRE5]

The preceding code will subscribe to the channel in the broker and wait for incoming signals.

Initially, we've declared the GPIO pin `13` as the output mode because the onboard LED is connected to this pin:

![](img/image009.jpg)

Onboard LED location

The location of the onboard LED is shown in the preceding image.

On having a close look at the code, we see that it initially imports the library and then sets the GPIO pin configuration. Then, we use a variable client to initiate the MQTT connection to the broker.

After that, we move on to subscribe our device to the channel, which in this case is named as `avirup/control/#`.

We have an event handler, `handleMessage()`. This event handler will deal with incoming messages. The incoming message will be stored in the packet variable. We've also implemented a callback method, `callback()`, which needs to be called from `handleMessage()`.

This enables us to receive multiple messages. Also note that, unlike other Node.js snippets, we haven't implemented any loop. The functionality is actually handled by the `callback()` method.

Finally, inside the function we obtain the payload, which is the message. It is then converted to a string and then condition checking is performed. We also print the value received to the console.

Now push this code to your Edison using FileZilla and run the code.

Once you run the code, you won't see anything in the console. The reason behind that is there is no message. Now, go to the Android application, MyMqtt, and browse to the Publish section of the application.

We need to insert the channel name here. In this case, it is `avirup/control`:

![](img/6639_03_05.png)

Publish MyMqtt

In the Topic section, enter the channel name, and in the Message section enter the message to be sent to Edison.

Now, in parallel, run your Node.js code.

Once your code is up and running, we will send a message. Type `ON` in the Message field and click Publish:

![](img/6639_03_06.png)

Send control signals

Once you have published from the application, it should be reflected on the PuTTY console:

![](img/6639_03_07.png)

Message send and receive—MQTT

Now you should see that the LED is turned on.

Similarly, send a message, `OFF`, to turn off the onboard LED:

![](img/6639_03_08.png)

Message send and receive. The LED should turn off

It's also worth noting that this will work even if Edison and the device aren't connected to the same network.

Now you can control your Intel Edison with your Android application. Virtually speaking, you can now control your home. In the following section, we'll deep dive into the home automation scenario and also develop a WPF application to control.

# Home automation using Intel Edison, MQTT, Android, and WPF

Until now we have learned about the MQTT protocol and how to subscribe and publish data, both using the application and Edison. Now we will be dealing with a real use case where we'll control an electrical load using Intel Edison, which again will be controlled by the Internet. Here is a quick introduction about what we will be dealing with:

*   Hardware components and circuits
*   Developing a WPF application to control Intel Edison
*   Using MQTT to stitch everything together

Since we've already seen how to control Edison using an Android application, this section won't concentrate on that; instead, it will mainly deal with the WPF application. This is just to give you a brief idea about how a PC can control IoT devices, not only in home automation, but also in various other use cases, both in simple proof of concept scenarios to industry standard solutions.

# Hardware components and circuit

When we are dealing with electrical load, we simply cannot directly connect it to Edison or any other boards, as it will end up frying. For dealing with these loads, an interfacing circuit is used called a relay. A relay in its crude form is a series of electromechanical switches. They operate on a DC voltage and control AC sources. Components that will be used are listed as follows:

*   Intel Edison
*   5V relay module
*   Electric bulb wires

Before going into the circuitry, we'll discuss relays first:

![](img/6639_03_09.jpg)

Relay schematics. Picture credits: [h](http://www.phidgets.com/docs/3051_User_Guide)[t](http://www.phidgets.com/docs/3051_User_Guide)[t](http://www.phidgets.com/docs/3051_User_Guide)[p](http://www.phidgets.com/docs/3051_User_Guide)[://w](http://www.phidgets.com/docs/3051_User_Guide)[w](http://www.phidgets.com/docs/3051_User_Guide)[w](http://www.phidgets.com/docs/3051_User_Guide)[.](http://www.phidgets.com/docs/3051_User_Guide)[p](http://www.phidgets.com/docs/3051_User_Guide)[h](http://www.phidgets.com/docs/3051_User_Guide)[i](http://www.phidgets.com/docs/3051_User_Guide)[d](http://www.phidgets.com/docs/3051_User_Guide)[g](http://www.phidgets.com/docs/3051_User_Guide)[e](http://www.phidgets.com/docs/3051_User_Guide)[t](http://www.phidgets.com/docs/3051_User_Guide)[s](http://www.phidgets.com/docs/3051_User_Guide)[.](http://www.phidgets.com/docs/3051_User_Guide)[c](http://www.phidgets.com/docs/3051_User_Guide)[o](http://www.phidgets.com/docs/3051_User_Guide)[m](http://www.phidgets.com/docs/3051_User_Guide)[/d](http://www.phidgets.com/docs/3051_User_Guide)[o](http://www.phidgets.com/docs/3051_User_Guide)[c](http://www.phidgets.com/docs/3051_User_Guide)[s](http://www.phidgets.com/docs/3051_User_Guide)[/3051_](http://www.phidgets.com/docs/3051_User_Guide)[U](http://www.phidgets.com/docs/3051_User_Guide)[s](http://www.phidgets.com/docs/3051_User_Guide)[e](http://www.phidgets.com/docs/3051_User_Guide)[r](http://www.phidgets.com/docs/3051_User_Guide)[_](http://www.phidgets.com/docs/3051_User_Guide)[G](http://www.phidgets.com/docs/3051_User_Guide)[u](http://www.phidgets.com/docs/3051_User_Guide)[i](http://www.phidgets.com/docs/3051_User_Guide)[d](http://www.phidgets.com/docs/3051_User_Guide)[e](http://www.phidgets.com/docs/3051_User_Guide)

The red rectangular area represents the electromagnet. We excite the electromagnet with a DC voltage, and that triggers the mechanical switch. Having a closer look at the preceding image, we can see three ports where the AC load is connected: common, normally closed, and normally open. In default conditions, that is when the electromagnet is not excited, and the common and normally closed ports are connected. What we are interested in for now is the normally open port.

The image of the relay used is shown as follows:

![](img/image015.jpg)

Relay unit. Picture credits: Seed Studio

The electrical load will have a live and neutral wire. Connect either one according to the following circuit:

![](img/image016.jpg)

Basic relay connection

With reference to the preceding figure, **Vcc** and **Gnd** are connected to the controller. The AC source connects one end of the electrical load directly, while the other is via the relay. A part of it connects the common port, while the other may be in **normally closed** (**NC**) or **normally open** (**NO**). When you have the other end of the electrical load connected to the NC port, then by default without excitation of the electromagnet, the circuit is complete. Since we don't want the bulb to be operating when the electromagnet isn't excited, connect it to the **NO** port, rather than **NC**. Thus, when the electromagnet is operating by applying voltage on **Vcc** and **Gnd** as ground, the mechanical switch flips to the **NO** position, thus connecting it with the common port.

The whole idea behind the operation of a relay is the use of electromechanical switches to complete a circuit. However, it is worth noting that not all relays operate on the same principle; some relays use solid state devices to operate.

**Solid State Relays** (**SSRs**) don't have any movable parts unlike that of electromechanical relays. SSRs uses photo-couplers to isolate the input and the output. They change electrical signals to optical signals, which propagates through space and thus isolates the entire circuit. The coupler on the receiving end is connected to any switching device, such as a MOSFET, to perform the switching action.

There are some advantages of using SSRs over electromechanical relays. They are as follows:

*   They provide high speed, high frequency switching operations
*   There is failure of contact points
*   They generate minimal noise
*   They don't generate operation noise

Although we will use electromechanical relays for now, if the use case deals with high frequency switching, then it's better to go with SSRs. It is also to be noted that when exposed to long usage, SSRs are known to heat up.

# Final circuit

The entire connection is shown in the following figure:

![](img/image017.jpg)

Circuit diagram for home automation project

The circuit adds Intel Edison because the relay circuit will be controlled by the controller. The relay here just acts as an interfacing unit to the AC load.

While the relay is being operated, please do not touch the underside of it or you may get an AC electric shock, which can be dangerous.

To test whether the circuit is working or not, try out a simple program using the Arduino IDE:

[PRE6]

The code should switch the position of the switch from the NC position to the NO position, thus completing the circuit, leading your bulb to glow. Don't forget to switch on the AC power supply.

Once you have the final circuit ready, we'll move forward with the development of the WPF application, which will control Edison.

# Android application for controlling Intel Edison using MQTT

In the previous section, we saw how an Android application can be used to subscribe and publish to a channel using a broker. Here, in this section, we'll develop our own Android application for controlling the device using MQTT. The section won't concentrate on the set up of the Android, but will concentrate on the development side of it. We're going to use the Android Studio IDE for the development of the application. Make sure it's configured with all the latest SDKs.

Open your Android Studio:

![](img/6639_03_10.jpg)

Android Studio—1

Now, select Start a new Android Studio project:

![](img/6639_03_11.png)

Android Studio—set up application name

Enter a name for your application; here, we've entered `MQTT`. Click on Next to continue:

![](img/6639_03_12.png)

Android Studio: set API level

Now select the Minimum SDK version. Select API 23: Android 6.0 (Marshmallow). Now let's select the type of activity:

![](img/6639_03_13.png)

Set activity

Select Empty Activity and click on Next:

![](img/6639_03_14.png)

Set start-up activity name

Give a name to your activity and click on Finish. It may take a few minutes to set up your project. After it's done, you may see a screen like this:

![](img/6639_03_15.jpg)

Design page. activity_name.xml

If you have a closer look over the project folder, you will notice that we have folders such as `java`, `res`, `values`, and so on. Let's have a closer look at what these folders actually contain:

*   `java`: This contains all the `.java` source files for your project. The main activity, named as `MainActivity.java`, is also contained in this project.
*   `res/drawable`: This is a directory for drawable components for this project. It won't be used for the moment.
*   `res/layout`: This contains all the files responsible for the applications UI.
*   `res/values`: This is a kind of directory for various other `xml` files that contain definitions of resources, such as string and color.
*   `AndroidManifest.xaml`: This is a manifest file that defines the application as well as the permissions required by the application.
*   `build.gradle`: This is an auto-generated file that contains information such as `compileSdkVersion`, `buildToolsVersion`, `applicationID`, and so on.

In this application, we will be using a third-party resource or library known as the eclipse `paho` library for MQTT. These dependencies need to be added to `build.gradle`.

There should be two `build.gradle` files. We need to add the dependencies in the `build.gradle(Module:app)` file:

[PRE7]

A dependency block should already exist, so you need not write the entire thing again. In that case, just write `compile('org.eclipse.paho:org.eclipse.paho.android.service:1.0.3-SNAPSHOT') { exclude module: 'support-v4'` in the already present dependency block. Immediately after you paste the code, Android Studio will ask you to sync gradle. It is necessary that you sync gradle before proceeding:

![](img/6639_03_16.jpg)

Add dependencies

Now we need to add permissions and services to our project. Browse to `AndroidManifest.xml` and add the following permission and services:

[PRE8]

After this is done, we will move forward with the UI. The UI needs to be designed under the layout, in the `activity_main.xml` file.

We'll have the following UI components:

*   `EditText`: This is for the broker
*   `URL EditText`: This is for the `EditText` port for the channel

Button to connect:

*   On button to send the on signal
*   Off button to send the off signal

Drag and drop the previously mentioned components in the designer window. Alternatively, you can directly write it in the text view.

For your reference, the XML code of the final design is shown as follows. Write your code inside the relative layout tab:

[PRE9]

Now click on the Design view; you will see that a UI has been created, which should be somewhat similar to that of the following screenshot:

![](img/6639_03_17.png)

Application design

Now have a closer look at the preceding code to try to find out the properties that were used. Basic properties such as `height`, `width`, and `position` are set, which is understandable from the code. The main properties are `text`, `id `and `hint` of the `EditText`. Each component in the Android UI should have a unique ID. Beside that, we set a hint such that the user knows exactly what to enter in the text areas. For ease, we have defined the text such that while deploying, we don't need to do that again. In the final application, remove the text properties. There is another option to get your values from `strings.xml`, which can be found under values for the texts or the hints:

[PRE10]

Now that we have the UI ready, we need to implement our code that will use these UI components to interact with the device using the MQTT protocol. We also have the dependencies in place. The main Java code is written in `MainActivity.java`.

Before proceeding further with the `MainActivity.java` activity, let's create a class that will handle the MQTT connection. This will make the code a lot easier to understand and more efficient. Have a look at the following screenshot to see the location of the `MainActivity.java` file:

![](img/6639_03_18.png)

Right click on the highlighted folder and click on new | java class. This class will handle all the required data exchanges happening between the application and the MQTT broker:

[PRE11]

The code that is pasted earlier may seem complicated at first glance, but it's actually very simple once you understand it. It is assumed that the reader has a basic understanding of object-oriented programming concepts.

The statements that import the packages are all done automatically. After creating the class, implement the `MqttCallback` interface. This will add the abstract methods that are required to be overridden.

Initially, we write a parameterized constructor for this class. We also create a global reference variable for the `MqttClient` and the `MqttCallback` classes. Three global variables are also created for `serverURI`, `port`, and `clientID`:

[PRE12]

The parameters are the broker `URI`, `port` number, and the `clientID`.

Next, we have created three global variables that are set to the parameters. In the `MqttConnect` method, we initially form a string as we take input as just the server URI. Here, we append it with `tcp://` and the port number and also create an object for the `MemoryPersistence` class:

[PRE13]

Next, we create the objects for the global reference variables using the `new` keyword:

[PRE14]

Please note the parameters as well.

The preceding code is surrounded by a try catch block to handle exceptions. The catch block is shown as follows:

[PRE15]

The connection part is achieved. The next phase is to create the `publish` method that will publish the data to the broker.

The parameter is just the `message` of type string:

[PRE16]

`client.publish` is used to publish data. The parameter is a string which is the `clientID` or `channelID` and an object of type `MqttMessage`. `MqttMessage` contains our message. However, it doesn't accept strings. It uses a byte array. In the try block, we first convert the string to a byte array and then publish the final message by using the `MqttMessage` class to the specific channel.

For this specific application, the overridden methods aren't required, so we leave it as is.

Now head back to the `MainActivity.java` class. We will use the `MqttClass` that we just created to do the publish action. The main task here is to get data from the UI and use it to connect to the broker using the class that we just wrote.

The `MainActivity.java` will contain the following code by default:

[PRE17]

Whenever the application is opened, the `onCreate` method is triggered. On having a closer look at the activity life cycle, the concept will be clear.

The life cycle callbacks are:

1.  `onCreate()`
2.  `onStart()`
3.  `onResume()`
4.  `onPause()`
5.  `onStop()`
6.  `onDestroy()`

More details on the life cycle can be obtained from:

[https://developer.android.com/guide/components/activities/activity-lifecycle.html](https://developer.android.com/guide/components/activities/activity-lifecycle.html)

Now we need to assign some reference variables to the UI components. We'll do that on a global level.

Before the start of `onCreate` method, that is before the keyword override, add the following lines:

[PRE18]

Now, in the `onCreate` method, we need to assign the reference variables we just declared and explicitly typecast them to the class type:

[PRE19]

In the preceding lines, we have explicitly type-casted them to `EditText` and `Button`, and bound them to the UI components.

Now we will create a new event handler for the connect button:

[PRE20]

The preceding block is activated when we press the connect button. The block contains a method whose parameter is view. The code that needs to be executed when the button is pressed needs to be written inside the `onCLick(View v)` method.

Before that, create a global reference variable for the class that you created before:

[PRE21]

Next, inside the method, get the text from the edit boxes. Declare the global variables for those of the type string beforehand:

[PRE22]

Now, write the following code inside the `onClick` method:

[PRE23]

Once we get the data, we will create an object for the `MqttClass` class and pass the strings as parameters, and we will also invoke the `MqttConnect` method:

[PRE24]

Now we'll create similar cases for the `ON` and `OFF` methods:

[PRE25]

We have used the `MqttPublish` method of `MqttClass`. The parameter is just a string and is based on the `onClick` method that when it is activated, it publishes the data.

Now the application is ready and can be deployed on your device. You must turn on the developer mode on your Android device and to deploy it, connect your device to a PC and press the Run button. You should now have the application running on your device. To test your application, you can directly use Edison or just use the MyMqtt application.

# Windows Presentation Foundation application for controlling using MQTT

WPF is a powerful UI framework for building Windows desktop client applications. It supports a broad spectrum of application features including models, controls, graphics layout, data binding, documents, and security. The programming is based on C# for the core logic and XAML for the UI.

# Sample "Hello World" application in WPF

Before moving on to the development of an application for controlling Intel Edison, let's have a brief look at how we can integrate certain basic features such as a button click event, handling displaying data, and so on. Open up your Visual Studio and select New Project.

In PCs with low RAM, the installation of Visual Studio may take a while, as will opening Visual for the first time:

The reason we are working with WPF is that it will be used in multiple topics, such as those in this chapter and in the upcoming chapters on robotics. In robotics, we'll be developing software to control robots. It is also assumed that the reader has an understanding of Visual Studio. For detailed information about how to work with Visual Studio and WPF, refer to the following link: 
[https://msdn.microsoft.com/en-us/library/aa970268(v%3Dvs.110).aspx](https://msdn.microsoft.com/en-us/library/aa970268(v%3Dvs.110).aspx)

![](img/6639_03_19.jpg)

Create new project in WPF

Click on New Project, then under the Visual C# section, click on WPF Application. Enter a name such as `Mqtt Controller` in the field of Name and click on OK.

Once you click OK, the project will be created:

![](img/6639_03_20.jpg)

WPF project created

Once the project is created, you should get a display similar to this. If some display components are missing from your window, then go to View and select those. Now have a close look on the Solution Explorer, which is visible on the right-hand side of the image.

There, have a look at the project structure:

![](img/6639_03_21.png)

Solution Explorer

An application has two main components. The first is the UI, which will be designed in `MainWindow.xaml`, and the second is the logic, which will be implemented in `MainWindow.xaml.cs`.

The UI is designed using XAML, while the logic is implemented in C#.

To start with, we'll just have one button control: a field where the user will enter some text and an area where the entered text will be displayed. After we have a fair idea about handling events, we can move forward to the implementation of MQTT.

Initially, we'll design the UI for the double click on `MainPage.xaml.cs`. It's in this file that we'll add the UI's XAML components. The code is written in XAML and much of the work can be accomplished by the use of drag and drop feature. From the toolbox situated on the right-hand side of the application, look up the following items:

*   `Button`
*   `TextBlock`
*   `TextBox`

There are two ways of adding the components. The first is to manually add the code in the XAML view of the page, while the second is to drag and drop from the components' toolbox. A few things to note are as follows.

The Designer window can be edited according to your wishes. A quick workaround for this is to select the component you want to edit, which can be done in the Properties window.

Properties can also be edited using XAML:

![](img/6639_03_22.jpg)

Visual Studio layout

In the preceding screenshot, we've changed the background color and added the components. Note the properties window where the background color is highlighted.

The `TextBox` is the area where the user will enter the text and the `TextBlock` is the area where it will be displayed. Once you have the components placed on the design view and have edited their properties, mainly the names of the components, we'll add the event handlers. For a shortcut of the design shown in the preceding screenshot, write the following XAML code within the `grid` tag:

[PRE26]

Now in the Designer window, double click on the button to create an event handler for a click event. The events that are available can be viewed in the Properties window, as shown in the following screenshot:

![](img/6639_03_23.png)

Event properties of button

Once you have double clicked, you will automatically be redirected to `MainWindow.xaml.cs` along with a method self-generated for the event.

You will get a method something similar to the following code:

[PRE27]

Here, we are going to implement the logic. Initially, we will read the data as written in the `TextBox`. If it's empty, we'll display a message saying that it cannot be empty. Then, we'll just pass the message to the `TextBlock`. The following code does the same thing:

[PRE28]

The preceding code initially reads the data and then checks if it's null or empty and then outputs the data into the `TextBlock`:

![](img/6639_03_24.jpg)

Application run—1

Press *F5* to run your application and then the preceding screens should appear. Next, delete the text in the TextBox and click on the Click me button:

![](img/6639_03_25.png)

Empty text

Now, enter any text in the TextBox and press the Click me button. Your entered text should be displayed following in the TextBlock:

![](img/6639_03_26.png)

WPF HelloWorld

Now that we know how to make a simple WPF application, we are going to edit the application itself to implement the MQTT protocol. To implement the MQTT protocol, we have to use a library, which will be added using the nugget package manager.

Now browse to References and click on Manage Nugget Packages and add the `M2Mqtt` external library:

![](img/6639_03_27.jpg)

NuGet package manager

Once we have the packages, we can use them in our project. For this project, we'll be using the following UI components in `MainWindow.xaml`:

*   A TextBox for entering the channel ID
*   A TextBlock to display the latest control command
*   A button to set the status as on
*   A button to set the status as off
*   A button to Connect

Feel free to design the UI on your own:

![](img/6639_03_28.jpg)

UI for controller application

In the preceding screenshot, you will see that the design is updated and a button has also been added. The code for the preceding design is pasted as follows. The TextBox is the area where we'll enter the channel ID, and then we will use the buttons to turn an LED on and off, and the Connect button to connect to the service. Now, as done previously, we will create event handlers for click events for the two buttons mentioned previously. To add click events, simply double click on each button:

[PRE29]

The preceding code is mentioned in the grid tag.

Now, once you have the design, move on to `MainWindow.xaml.cs` and write the main code. You will notice that a constructor and two event handler methods already exist.

Add the following namespace to use the library using:

[PRE30]

Now create an instance of the `MqttClient` class and declare a global string variable:

[PRE31]

Next, in the event handler for the Connect button, connect it to the broker using the channel ID.

The entire code for the Connect button's event handler is mentioned as follows:

[PRE32]

In the preceding snippet, we read the data from the `textbox` that contains the channel ID. If it's null, we ask the user to enter it again. Then, finally, we connect it to the channel ID. Note that it is inside the `try catch` block.

There are two more event handlers. We need to publish some value to the channel they are connected to.

In the `on` button's event handler, insert the following code:

[PRE33]

As seen in the preceding code, the parameter for the `Publish` method is the topic, which is the `channelID` and a `byte[] array` which contains the message.

Similarly, for the `off` method, we have:

[PRE34]

That's it. That's the entire code for your MQTT controller for home automation. The entire code is pasted as follows for your reference:

[PRE35]

Press the *F5* or the Start button to execute this code:

![](img/6639_03_29.jpg)

Application running

Next, in the TextBox, enter the `channelID`. Here, we'll be entering it as `avirup/control` and then we will press the Connect button:

![](img/6639_03_30.jpg)

Application running—2

Now open your PuTTY console and log in to Intel Edison. Verify that the device is connected to the Internet using the `ifconfig` command. Next, just run the Node.js script. Next, press the `ON` button:

![](img/6639_03_31.jpg)

MQTT controlled by WPF application

Similarly, on pressing the `OFF` button, you will see a screen similar to the following:

![](img/6639_03_32.jpg)

MQTT controlled by WPF

Keep pressing `ON` and `OFF` and you will see the effect on Intel Edison. Now that we remember that we have connected the relay and the electric bulb, the effect should be visible by now. If the main switch of the AC power supply is turned off, then you won't see the bulb getting turned on, but you will hear a `tick` sound. That suggest that the relay is now in the `ON` position. The image of the hardware setup is shown as follows:

![](img/image039.jpg)

Hardware setup for home automation

Thus, you have a home automation setup ready and you can control it by the PC application or the Android application.

If you are in office network, then sometimes port `1883` is blocked. In those cases, it is recommended to use your own personal network.

# Open-ended task for the reader

Now, you may have got a brief idea about how things must work in home automation. We have covered multiple areas in this niche. The task that is left for the reader is not only to integrate a single control command, but multiple control commands. This will allow you to control multiple devices. Add more functionality in the Android and the WPF application and go with more string control commands. Connect more relay units to the device for interfacing.

# Summary

In this chapter, we've learned about the idea of home automation in its crude form. We also learned about how we can control an electrical load using relays. Not only that, but also we learned how to develop a WPF application and implement the MQTT protocol. On the device end, we used a Node.js code to connect our device to the Internet and subscribe to certain channels using the broker and ultimately receive signals to control itself. In the Android side of the system, we have used an already available MyMqtt application and used it to both to get and publish data. However, we also covered the development of the Android application in detail and showcased the use of it in implementing the MQTT protocol to control devices.

In [Chapter 4](3fa86b30-3d51-4628-a827-db7b3e31f3e7.xhtml), *Intel Edison and Security System*, we are going to learn how to deal with image processing and speech processing applications using Intel Edison. [Chapter 4](3fa86b30-3d51-4628-a827-db7b3e31f3e7.xhtml), *Intel Edison and Security System*, will mainly deal with Python and the usage of some open source libraries.