# Working with Signals and Slots

Thus far, we have learned how to create applications and display various kinds of widgets. If that were all that GUI applications were made of, that would be the end of the matter. But there is more that we need to do in order to make our applications usable. In this chapter, we will set about the following:

*   Understanding the concept behind signals and slots
*   Learning the different ways to connect signals and slots

GUI toolkits usually provide a means to react to things that occur within an application. Nothing is left to chance. Every tick that happens within the application is registered and taken note of. For example, when you move a window or resize it, the action gets registered, and provided ample code has been written, it will be executed as a reaction to the moving or resizing of the window. For every action that occurs, a number of outcomes may happen. Essentially, the questions we want to answer are as follows: what do we do when a particular action or event has occurred? How do we handle it?

One way to implement the ability to react to an action that has occurred is by using the design pattern called the **Observer Pattern**.

In the Observer Pattern design, an observable object communicates its state change to other objects that are observing it. For instance, any time an object (A) wants to be notified of a state change of some other object (B), it first has to identify that object (B) and register itself as one of the objects that should receive such notification of the state change. Sometime in the future, when the state of an object (B) occurs, object (B) will go through a list of objects it keeps that want to be informed regarding the state change. This will, at this point, include object (A):

![](img/8e26039c-0d1e-4b15-ae9c-f33929c73fc2.png)

From the preceding diagram, the **Subject** circle is termed the observable object, while the circles in the bounded box are the observers. They are being notified of the state change of the **Subject** as its count variable is increased from 1 to 5.

Some events or actions that may occur within our application that we will be interested in and would want to react to include the following:

*   A window being resized
*   A button clicked
*   Pressing the return key
*   A widget being dragged
*   A mouse hovering over the widget

In the case of a button, a typical response to a click of a mouse would be to start a download process or send an email.

# Signals and slots

In Qt, this action-response scheme is handled by signals and slots. This section will include a few definitions, and then we shall jump into an example for further explanation.

A signal is a message that is passed to communicate that the state of an object has changed. This signal may carry information about the change that has occurred. For instance, when a window has been resized, the signal will usually carry the coordinates of the new state (or size) of the window. Sometimes, a signal may carry no extra information, such as that of a button click.

A slot is a specific function of an object that is called whenever a certain signal has been emitted. Since slots are functions, they will embody lines of code that perform an action, such as closing a window, disabling a button, and sending an email, to mention but a few.

Signals and slots have to be connected (in code). Without writing code to connect a signal and a slot, they will exist as independent entities.

Most of the widgets in Qt come with a number of signals and slots. However, it is possible to write your own signals and slots too.

So what do a signal and a slot look like?

Consider the following code listing:

```cpp
#include <QApplication>
#include <QPushButton>
int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   QPushButton *quitButton = new QPushButton("Quit");
   QObject::connect(quitButton, SIGNAL(clicked()),
           &app, SLOT(quit()));
   quitButton->show();
   return app.exec();
}
```

As usual, we shall use the following steps to compile the project:

1.  Create a new folder with an appropriate name of your choosing
2.  Create a `.cpp` file named `main.cpp`

3.  Issue the following commands in the Terminal:

```cpp
% qmake -project
% qmake 
% make 
% ./executable_file
```

Be sure to edit the `.pro` file to include the `widget` module during compilation.

Compile and run the application.

An instance of `QPushButton` is created, `quitButton`. The `quitButton` instance here is the observable object. Anytime this button is clicked, the `clicked()` signal will be emitted. The `clicked()` signal here is a method belonging to the `QPushButton` class that has only been earmarked as a signal.

The `quit()` method of the `app` object is called, which terminates the `event` loop.

To specify what should happen when `quitButton` has been clicked, we pass `app` and say that the `quit()` method on the `app` object should be called. These four parameters are connected by the static function, `connect()`, of the `QObject` class.

The general format is (`objectA`, *signals* (`methodOnObjectA()`), `objectB`, *slots* (`methodOnObjectB()`)).

The second and final parameters are the signatures of the methods representing the signals and the slots. The first and third parameters are pointers and should contain the address to objects. Since `quitButton` is already a pointer, we simply pass it as it is. On the other hand, `&app` would return the address of `app`.

Now, click on the button and the application will close:

![](img/585eba7e-8d2f-4e6e-a267-190e61b0a482.png)

When this application is run, you should see the following.

The example we have just illustrated is quite primitive. Let's write an application where a change in the state of one widget is passed to another widget. Not only will the signal be connected to a slot, but data will be carried along:

```cpp
#include <QApplication>
#include <QVBoxLayout>
#include <QLabel>
#include <QDial>
int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   QWidget *window = new QWidget;
   QVBoxLayout *layout = new QVBoxLayout;
   QLabel *volumeLabel = new QLabel("0");
   QDial *volumeDial= new QDial;
   layout->addWidget(volumeDial);
   layout->addWidget(volumeLabel);
   QObject::connect(volumeDial, SIGNAL(valueChanged(int)), volumeLabel,
   SLOT(setNum(int)));
   window->setLayout(layout);
   window->show();
   return app.exec();
}
```

This is yet another simple program that illustrates how data is passed between the signal and slot. An instance of `QVBoxLayout` is created, `layout`. A `QLabel` instance, `volumeLabel`, is created and will be used to display changes that occur. It is initialized with the string `0`. Next, an instance of `QDial` is created with `QDial *volumeDial = new QDial`. The `QDial` widget is a knob-like looking widget that is graduated with a minimum and maximum range of numbers. With the aid of a mouse, the knob can be turned, just like you would turn up the volume on a speaker or radio.

These two widgets, `volumeLabel` and `volumeDial`, are then added to the layout using the `addWidget()` method.

Whenever we change to move the knob of `QDial`, a signal called `valueChanged(int)` is emitted. The slot named `setNum(int)` of the `volumeLabel` object is a method that accepts an `int` value.

Note how the connection between the signals and slots is established in the following code:

```cpp
QObject::connect(volumeDial, SIGNAL(valueChanged(int)), volumeLabel, SLOT(setNum(int)));
```

This literally establishes a connection that reads "*Anytime the* `QDial` *changes its value, call the* `setNum()` *method of the* `volumeLabel` *object and pass it an* `int` *value*." There can be a number of state changes that may occur in `QDial`. The connection further makes it explicit that we are only interested in the value that has changed when the knob (`QDial`) was moved, which, in turn, emitted its current value through the `valueChanged(int)` signal.

To dry run the program, let's assume that the range of `QDial` is representing a radio volume range between `0` and `100`. If the knob of `QDial` is changed to half of the range, the `valueChanged(50)` signal will be emitted. Now, the value 50 will be passed to the `setNum(50)` function. This will be used to set the text of the label, `volumeLabel` in our example, to display 50.

Compile the application and run it. The following output will be displayed on the first run:

![](img/8d01a7f7-6300-47d4-9ecd-11d73df6ed4d.png)

As you can see, the initial state of `QDial` is zero. The following label shows that too. Move the dial, and you will see that the label will have its value change accordingly. The following screenshot shows the state of the application after the knob has been moved to half of the range:

![](img/8df0bd5f-3c2b-4242-9dc3-7c0ff5fcc75a.png)

Move the knob around and observe how the label changes accordingly. This is all made possible by means of the signals and slots mechanism.

# Signals and slots configuration

It is not only possible to connect one signal to one slot, but to connect one signal to more than one slot. This involves repeating the `QObject::connect()` call and, in each instance, specifying the slot that should be called when a particular signal has been emitted.

# Single signal, multiple slots

In this section, we shall concern ourselves with how to connect a single signal to multiple slots.

Examine the following program:

```cpp
#include <QApplication>
#include <QVBoxLayout>
#include <QLabel>
#include <QDial>
#include <QLCDNumber>
int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   QWidget *window = new QWidget;
   QVBoxLayout *layout = new QVBoxLayout;
   QLabel *volumeLabel = new QLabel("0");
   QDial *volumeDial= new QDial;
   QLCDNumber *volumeLCD = new QLCDNumber;
   volumeLCD->setPalette(Qt::red);
   volumeLabel->setAlignment(Qt::AlignHCenter);
   volumeDial->setNotchesVisible(true);
   volumeDial->setMinimum(0);
   volumeDial->setMaximum(100);
   layout->addWidget(volumeDial);
   layout->addWidget(volumeLabel);
   layout->addWidget(volumeLCD);
   QObject::connect(volumeDial, SIGNAL(valueChanged(int)), volumeLabel,
   SLOT(setNum(int)));
   QObject::connect(volumeDial, SIGNAL(valueChanged(int)), volumeLCD ,   
   SLOT(display(int)));
   window->setLayout(layout);
   window->show();
   return app.exec();
}
```

We want to illustrate how one signal can be connected to two different slots, or more than one slot for that matter. The widget that will be emitting the signal is an instance of `QDial`, `volumeDial`. An instance of `QLCDNumber`, `volumeLCD` is created. This widget displays information in an LCD-like digit form. Note `volumeLabel` is an instance of a `QLabel`. These two widgets shall provide the two slots.

To make the text of `volumeLCD` stand out, we set the color of the display to red with `volumeLCD->setPalette(Qt::red);`.

The fact that `layout` is an instance of `QVBoxLayout` means that widgets added to this layout will flow from top to bottom. Each widget added to the layout will be centered around the middle as we set `setAlignment(Qt::AlignHCenter);` on `volumeLabel`:

```cpp
volumeDial->setNotchesVisible(true);
volumeDial->setMinimum(0);
volumeDial->setMaximum(100);
```

The graduations on `volumeDial` are visible when the `setNotchesVisible(true)` method is called. The default argument to `setNotchesVisible()` is `false`, which makes the small ticks (graduations) on the dial invisible. The range for our `QDial` instance is set by calling `setMinimum(0)` and `setMaximum(100)`.

The three widgets are added accordingly with each call to the `addWidget()` method:

```cpp
layout->addWidget(volumeDial);
layout->addWidget(volumeLabel);
layout->addWidget(volumeLCD);
```

Now, `volumeDial` emits the signal, `valueChanged(int)`, which we connect to the `setNum(int)` slot of `volumeLabel`. When the knob of `volumeDial` changes, the current value will be sent for display in `volumeLabel`:

```cpp
QObject::connect(volumeDial, SIGNAL(valueChanged(int)), volumeLabel, SLOT(setNum(int)));
QObject::connect(volumeDial, SIGNAL(valueChanged(int)), volumeLCD , SLOT(display(int)));
```

This same signal, `valueChanged(int)` of `volumeDial`, is also connected to the `display(int)` slot of `volumeLCD`.

The total effect of these two connections is that when there is a change in `volumeDial`, both `volumeLabel` and `volumeLCD` will be updated with the current value of `volumeDial`. All this happens at the same time without the application clogging up, all thanks to the efficient design of signals and slots.

Compile and run the project. A typical output of the program is as follows:

![](img/5b15b2eb-974e-4750-9c6a-60cc246ca323.png)

In the preceding screenshot, when the `QDial` widget (that is the round-looking object) was moved to 32, both `volumeLabel` and `volumeLCD` were updated. As you move the dial, `volumeLabel` and `volumeLCD` will receive the updates by way of signals and will update themselves accordingly.

# Single slot, multiple signals

In the next example, we shall connect two signals from different widgets to a single slot. Let's modify our earlier program as follows:

```cpp
#include <QApplication>
#include <QVBoxLayout>
#include <QLabel>
#include <QDial>
#include <QSlider>
#include <QLCDNumber>
int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   QWidget *window = new QWidget;
   QVBoxLayout *layout = new QVBoxLayout;
   QDial *volumeDial= new QDial;
   QSlider *lengthSlider = new QSlider(Qt::Horizontal);
   QLCDNumber *volumeLCD = new QLCDNumber;
   volumeLCD->setPalette(Qt::red);
   lengthSlider->setTickPosition(QSlider::TicksAbove);
   lengthSlider->setTickInterval(10);
   lengthSlider->setSingleStep(1);
   lengthSlider->setMinimum(0);
   lengthSlider->setMaximum(100);
   volumeDial->setNotchesVisible(true);
   volumeDial->setMinimum(0);
   volumeDial->setMaximum(100);
   layout->addWidget(volumeDial);
   layout->addWidget(lengthSlider);
   layout->addWidget(volumeLCD);
   QObject::connect(volumeDial, SIGNAL(valueChanged(int)), volumeLCD ,
   SLOT(display(int)));
   QObject::connect(lengthSlider, SIGNAL(valueChanged(int)), volumeLCD 
   , SLOT(display(int)));
   window->setLayout(layout);
   window->show();
   return app.exec();
}
```

In the `include` statements, we add the line, `#include <QSlider>`, to add the `QSlider` class, which is a widget that can be set to a value within a given range:

```cpp
QApplication app(argc, argv);
QWidget *window = new QWidget;
QVBoxLayout *layout = new QVBoxLayout;
QDial *volumeDial= new QDial;
QSlider *lengthSlider = new QSlider(Qt::Horizontal);
QLCDNumber *volumeLCD = new QLCDNumber;
volumeLCD->setPalette(Qt::red);
```

The `QSlider` widget is instantiated and passed `Qt::Horizontal`, which is a constant that changes the orientation of the widgets such that it is presented horizontally. Everything else is the same as we saw in previous examples. The window and layout are instantiated, together with the `QDial` and `QSlider` objects:

```cpp
lengthSlider->setTickPosition(QSlider::TicksAbove);
lengthSlider->setTickInterval(10);
lengthSlider->setSingleStep(1);
lengthSlider->setMinimum(0);
```

The first widget that shall emit a signal in this example is the `volumeDial` object. But now, the `QSlider` instance also emits a signal that allows us to get the state of the `QSlider` whenever it has changed.

To show the graduations on `QSlider`, we invoke the `setTickPosition()` method and pass the constant, `QSlider::TicksAbove`. This will show the graduations on top of the slider, very similar to how the graduations on a straight edge appear.

The `setMinimum()` and `setMaximum()` variables are used to set the range of values for our `QSlider` instance. The range here is between `0` and `100`.

The `setTickInterval(10)` method on the `lengthSlider` object is used to set the interval between the ticks.

The `QVBoxLayout` object, `layout`, adds the `lengthSlider` widget object to the list of widgets it will house with the line, `layout->addWidget(lengthSlider);`:

```cpp
QObject::connect(volumeDial, SIGNAL(valueChanged(int)), volumeLCD , SLOT(display(int)));
QObject::connect(lengthSlider, SIGNAL(valueChanged(int)), volumeLCD , SLOT(display(int)));
```

There are two calls to the static method, `connect()`. The first call will establish a connection between the `valueChanged(int)` signal of `volumeDial` with the  `display(int)` slot of `volumeLCD`. As a result, whenever the `QDial` object changes, the value will be passed to the `display(int)` slot for display.

From a different object, we shall connect the `valueChanged(int)` signal of `lengthSlider` to the same slot, `display()`, of the `volumeLCD` object.

The remainder of the program is the same as usual.

Compile and run the program from the command line as we have done for the previous examples.

The first time the application is run, the output should be similar to the following:

![](img/82b87945-6637-4767-997d-70074c28d257.png)

Both `QDial` and `QSlider` are at zero. Now, we will move the `QDial` to 48\. See how the `QLCDNumber` is updated accordingly:

![](img/54a31e9b-fe3d-4b95-94a3-f192f739af09.png)

With the way we have set up our signals and slots, it will also be possible for `QSlider` to also update the same widget, `volumeLCD`. When we move `QSlider`, we will see that `volumeLCD` is updated immediately by its value:

![](img/5cfdb730-c33f-41bd-9780-3e3e3f68b6ca.png)

As can be seen, `QSlider` has been moved to the tail end of its range and the value has been passed onto `volumeLCD`.

# Summary

In this chapter, we took a look at the core concept of signals and slots in Qt. After creating our first application, we looked at the various ways in which signals and slots can be connected.

We saw how to connect one signal from a widget to multiple slots. This is a typical way to set up signals and slots, especially when a change in the state of a widget has to be communicated to many other widgets.

To show how flexible signals and slots could be configured, we also looked at an example where multiple signals were connected to one slot of a widget. This type of arrangement is useful when different widgets can be used to achieve the same effect on a widget.

In [Chapter 4](d0636a57-1cac-4853-836b-850c773e82db.xhtml), *Implementing Windows and Dialog*, we shall change our style of writing applications and study how to make full-blown window applications.