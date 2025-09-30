# Creating Widgets and Layouts

In this chapter, we shall take a look at what widgets are and the various kinds that are available for creating GUIs. For most GUI applications that you will write, Qt is laden with sufficient widgets to implement it. Coupled with widgets are layout classes, which help us to arrange and position the widgets for better appeal.

By the end of this chapter, you should be aware of the following:

*   Understand and know how to use widgets
*   Know the classes needed to lay out widgets

# Widgets

**Widgets** are the graphical components with which we construct user interfaces. A familiar example of such a component is a textbox. This is the component that is used to capture our email address or last and first names on forms in a GUI application.

There are a few critical points to note regarding widgets in Qt:

*   Information is passed to widgets by way of events. For a textbox, an example of an event could be when a user clicks within the textbox or when the `return` key has been pressed while a textbox cursor is blinking.
*   Every widget can have a parent widget or children widgets.
*   Widgets that do not have a parent widget become a window when the `show()` function is called on them. Such a widget will be enclosed in a window with buttons to close, maximize, and minimize it.
*   A child widget is displayed within its parent widget.

Qt organizes its classes with heavy use of inheritance, and it is very important to have a good grasp of this. Consider the following diagram:

![](img/992ba93e-5c34-4cfa-b166-bb98ef7c8ca0.png)

At the very top of the hierarchy is the **QObject**. A lot of classes inherit from the **QObject** class. The **QObject** class also contains the mechanisms of signals and slots and event management, among other things.

Furthermore, widgets that share common behavior are grouped together. **QCheckBox**, **QPushButton**, and **QRadioButton** are all buttons of the same kind and thus inherit from **QAbstractButton**, which holds properties and functions that are shared by all buttons. This principle also applies to **QAbstractScrollArea** and its children, **QGraphicsView** and **QTextEdit**.

To put into practice some of what we have just learned, let's create a simple Qt program with only one widget.

This Qt application displays only one button. Open a text file and name it how you want with the suffix `.cpp`. 

Most of the examples will require that you create a new folder where the source code will be stored. This will allow for easy compilation of the program as a project.

Insert the following lines of codes. Create a new folder and move the `.cpp` file into it:

```cpp
#include <QApplication>
#include <QPushButton>
int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   QPushButton myButton(QIcon("filesaveas.png"),"Push Me");
   myButton.setToolTip("Click this to turn back the hands of time");
   myButton.show();
   return app.exec();
}
```

The purpose of this application is to show how a widget without a parent object becomes the main window when executed. The button that will be created will include an icon and a tooltip.

For starters, this application looks similar to the one we wrote at the tail end of [Chapter 1](a157893d-287f-42c4-8166-a7d414e09035.xhtml), *Introducing Qt 5*. In this application, a button named `myButton` is declared. An instance of `QIcon` is passed as the first argument to the default constructor of `QPushButton`. This reads the file named `filesaveas.png` (which, for now, should be in the same folder as the source code file on GitHub). The text `"Push Me"` is passed as the second argument. This text will be displayed on the button.

The next line, `myButton.setToolTip("Click this to turn back the hands of time");`, is used to set a tooltip on the button. A tooltip is a piece of text or a message that is displayed when you rest the mouse cursor over a widget. It usually holds extra or explanatory information over and above what the widget might be displaying.

Lastly, we call the `show()` function on the `myButton` object to unhide it and draw it to the screen. In this application, we only have one widget, `QPushButton`. What could be the parent of this widget? Well, if unspecified, the parent defaults to `NULL`, which tells Qt that the widget is without a parent. When displaying such a widget, it will be enclosed in a window on account of this reasoning.

Save the file and run the following commands to compile your application. Change directory to the new folder you created that houses the `.cpp` file created:

The commands that should be run in a Terminal or on the command line begin with a `%` sign, which represents the prompt on the Terminal. Depending on the setup of your Terminal, this might be slightly different, but the command is all the characters after the `%` sign.

```cpp
% qmake -project
```

From the name of the `.pro` file, it tells us that the name of the folder where the `.cpp` file is located is called `qbutton`. This name should, therefore, change to whichever folder name the `.cpp` file is located in when you issue the preceding command.

Now, remember to add the following line to the `qbutton.pro` beneath `INCLUDEPATH += .`:

```cpp
QT += widgets
```

Continue with the following commands:

```cpp
% qmake
% make
```

Run the application from the command line according to an issue:

```cpp
% ./qbutton
```

You should obtain the following screenshot:

![](img/9f1f9558-204d-4049-9512-730bfa11429f.png)

The preceding screenshot shows what you will see when the program is run for the first time:

![](img/da9815e3-7d1d-44d2-9d42-0a610aa3d853.png)

The tooltip that was specified within the code is displayed when we rest our cursor on the button, as seen in the preceding screenshot.

The button also shows the image for those cases when you want to add an image to a button in order to improve the intuitiveness of a UI.

A few observations worthy of note are the following:

*   The `setToolTip()` function is not found in the `QPushButton` class. Instead, it is one of the functions that belongs to the `QWidget` class.
*   This highlights the usefulness that classes get by means of inheritance.
*   The property or member of the `QWiget` class that stores the value of the tooltip is `toolTip`.

To cap off this section on widgets, let's customize a `QLabel` and display it. This time, an instance of `QLabel` will have its font changed and shall display a longer text than usual.

Create a file named `qlabel_long_text.cpp` in a newly created folder and insert the following code:

```cpp
#include <QApplication>
#include <QString>
#include <QLabel>
int main(int argc, char *argv[])
{
            QApplication app(argc, argv);
            QString message = "'What do you know about this business?' the King said to Alice.\n'Nothing,' said Alice.\n'Nothing whatever?' persisted the King.\n'Nothing whatever,' said Alice.";
            QLabel label(message);
            label.setFont(QFont("Comic Sans MS", 18));
            label.setAlignment(Qt::AlignCenter);
            label.show();
            return app.exec();
}
```

The structure of our Qt programs has not changed that much. The first three (3) lines have the `include` directives adding the headers for the classes we will be using.

As usual, the arguments to the `main()` function are passed to `app()`. The `message` variable is a `QString` object that holds a long string. `QString` is the class used when working with strings. It has a host of functionalities not available in C++ string.

An instance of `QLabel` is created, `label`, and `message` is passed to this. To change the style by which the label string is displayed, we pass an instance of `QFont` to the `setFont` function. We select the font style *Comic Sans MS*, with a point size of *18*, to the constructor of `QFont`.

To align all the text in the middle, we call the `setAlignment` function on the `label` object and pass the `Qt::AlignCenter` constant.

Lastly, we display the widget by calling the `show` function on the `label` object.

As usual, we shall issue the following codes on the command line to compile and run this program:

```cpp
% qmake -project
% qmake
% ./qlabel_long_text
```

Remember to add `QT += widgets` to the `.pro` file.

The output of the program appears as follows. All the text on the lines are centered in the middle:

![](img/59ee3589-ee41-4b91-bf76-22e89706f327.png)

Once again, the only widget within the `label` application becomes the main window because it has no parent object associated with it. Secondly, the widget becomes a window because the `show()` method was called on `label`.

# Layouts

Up to this point, we have been creating applications that only have one widget serving as the main component and, by extension, a window too. However, GUI applications are usually made up of several widgets that come together to communicate a process to the user. One way in which we can make use of multiple widgets is to use layouts to serve as the canvas into which we insert our widgets.

Consider the following class inheritance diagram:

![](img/f3ddb47b-3a07-4249-a70e-7db974f170be.png)

It is important to consider the classes used in laying out widgets. As usual, the top class from which the `QLayout` abstract class inherits is `QObject`. Also, `QLayout` makes use of multiple inheritances by inheriting from `QLayoutItem`. The concrete classes here are `QBoxLayout`, `QFormLayout`, `QGridLayout`, and `QStackedLayout`. `QHBoxLayout` and `QVBoxLayout` further refine what the `QBoxLayout` class is by adding orientation to how the widgets within a layout might be arranged.

The following table provides a brief description of what the major layouts do:

| **Layout class** | **Description** |
| `QFormLayout` | The `QFormLayout` class ([https://doc.qt.io/qt-5/qformlayout.html](https://doc.qt.io/qt-5/qformlayout.html)) manages forms of input widgets and their associated labels. |
| `QGridLayout` | The `QGridLayout` class ([https://doc.qt.io/qt-5/qgridlayout.html](https://doc.qt.io/qt-5/qgridlayout.html)) lays out widgets in a grid. |
| `QStackedLayout` | The `QStackedLayout` class ([https://doc.qt.io/qt-5/qstackedlayout.html](https://doc.qt.io/qt-5/qstackedlayout.html)) provides a stack of widgets where only one widget is visible at a time. |
| `QVBoxLayout` | The `QVBoxLayout` class ([https://doc.qt.io/qt-5/qvboxlayout.html](https://doc.qt.io/qt-5/qvboxlayout.html)) lines up widgets vertically. |
| `QHBoxLayout` | The `QHBoxLayout` class ([https://doc.qt.io/qt-5/qhboxlayout.html](https://doc.qt.io/qt-5/qhboxlayout.html)) lines up widgets horizontally. |

We need to lay out the widgets for two main reasons:

*   To allow us to display more than one widget.
*   To present the many widgets in our interface nicely and intuitively to allow the UI to be useful. Not all GUIs allows users to do their work well. Bad layout can confuse the users of a system and make them struggle to use it properly.

Let's create a simple program to illustrate how to use some of the layout classes.

# QGridLayout

The `QGridLayout` is used to arrange widgets by specifying the number of rows and columns that will be filled up by multiple widgets. A grid-like structure mimics a table in that it has rows and columns and widgets are inserted as cells where a row and column meet.

Create a new folder and, using of any editor, create a file named `main.cpp`:

```cpp
#include <QApplication>
#include <QPushButton>
#include <QGridLayout>
#include <QLineEdit>
#include <QDateTimeEdit>
#include <QSpinBox>
#include <QComboBox>
#include <QLabel>
#include <QStringList>
int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   QWidget *window = new QWidget;
   QLabel *nameLabel = new QLabel("Open Happiness");
   QLineEdit *firstNameLineEdit= new QLineEdit;
   QLineEdit *lastNameLineEdit= new QLineEdit;
   QSpinBox *ageSpinBox = new QSpinBox;
   ageSpinBox->setRange(1, 100);
   QComboBox *employmentStatusComboBox= new QComboBox;
   QStringList employmentStatus = {"Unemployed", "Employed", "NA"};
   employmentStatusComboBox->addItems(employmentStatus);
   QGridLayout *layout = new QGridLayout;
   layout->addWidget(nameLabel, 0, 0);
   layout->addWidget(firstNameLineEdit, 0, 1);
   layout->addWidget(lastNameLineEdit, 0, 2);
   layout->addWidget(ageSpinBox, 1, 0);
   layout->addWidget(employmentStatusComboBox, 1, 1,1,2);
   window->setLayout(layout);
   window->show();
   return app.exec();
}
```

The aim of the program is to illustrate how to use a layout object. To fill up the layout, other widgets will be discussed too.

In the preceding code, `*window` is an instance of `QWidget`. For now, keep this object in to see and how we shall turn it into a window.

The widgets that we are going to insert into our layout are created thereafter, namely `name`, `firstnameLineEdit`, and `lastNameLineEdit`.

Some prefer to name their variables by appending the name of the class that they are instantiating to it. The CamelCase naming scheme is being used here too.

`QLineEdit` is essentially the class for creating textboxes. `QSpinbox` is a widget that allows for the selection of a value between a given range. In this case, `ageSpinBox->setRange(1, 100)` sets the range of possible values between `1` and `100`.

Next, we instantiate the `QComboBox` class to create a widget with drop-down values specified by a list of strings stored in `QStringList`. The list of strings, `employmentStatus`, is then passed to `employmentStatusComboBox` by calling its `addItems()` method. These will become the options that will be displayed when the widget is clicked.

Since we want to layout our widgets in a grid layout, we create an object from the `QGridLayout`, `*layout`. To add the widgets to the layout, the `addWIdget()` method is called and each time, the widget, along with two (2) numbers that specify the row and column where the widget is to be inserted is specified:

```cpp
layout->addWidget(nameLabel, 0, 0);
layout->addWidget(firstNameLineEdit, 0, 1);
layout->addWidget(lastNameLineEdit, 0, 2);
layout->addWidget(ageSpinBox, 1, 0);
layout->addWidget(employmentStatusComboBox, 1, 1,1,2);
```

The first widget to be inserted into the layout object is the label, `nameLabel`. This occupies the first row and first column of the grid. The first row is represented by the second parameter `0` while the first column is represented by `0`. This resolves to the selection of the first cell of the grid to keep `nameLabel`.

The second widget that is added to the layout is `firstNameLineEdit`. This widget will be inserted on the first row, marked by `0`, and on the second column marked by `1`. Next to this widget is added the `lastNameLineEdit` widget, also sitting on the same row, `0`.

The `ageSpinBox` widget will be fixed on the second row marked by `1` and in the first column, marked by `0`.

The `employmentStatusComboBox` widget is added to the `layout` object and further stretches out by specifying the `rowspan` with the last `(1, 2)` arguments that are passed along:

```cpp
window->setLayout(layout);
window->show();
```

The `window` object is without a layout. To set the layout of the widget, call `setLayout` and pass in the layout object, which holds the other widgets.

Because `window`, which is basically a widget, has no parent object, it will become a window when we call the `show()` method on it. Also, all the widgets that were added to the layout object via the `addWidget()` method are children of the `layout` object.

Run the project by issuing the commands to create the project and compiling on the command line.

You should see this on successful compilation:

![](img/723bdba6-e78d-41f7-a4e7-52cc22fe88a6.png)

Notice how the drop-down widget stretches to fill the third column. The placement of the widgets conforms to how we laid out the widgets as we called `addWidget()`. Experiment by clicking on the `ageSpinBox` to observe how it behaves.

In the next section, we shall take a look at a useful layout class called `QFormLayout`.

# QFormLayout

For those instances when you simply need to place a number of widgets together in a two-column layout, the `QFormLayout` is useful. You may choose to construct a form using `QGridLayout`, but for form presentation, `QFormLayout` is most suited.

Take, for instance, the following code. It illustrates a form that has labels in the first column and the actual control for taking user input in the second column:

```cpp
#include <QApplication>
#include <QFormLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QSpinBox>
#include <QComboBox>
#include <QStringList>
int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   QWidget *window = new QWidget;
   QLineEdit *firstNameLineEdit= new QLineEdit;
   QLineEdit *lastNameLineEdit= new QLineEdit;
   QSpinBox *ageSpingBox = new QSpinBox;
   QComboBox *employmentStatusComboBox= new QComboBox;
   QStringList employmentStatus = {"Unemployed", "Employed", "NA"};
   ageSpingBox->setRange(1, 100);
   employmentStatusComboBox->addItems(employmentStatus);
   QFormLayout *personalInfoformLayout = new QFormLayout;
   personalInfoformLayout->addRow("First Name:", firstNameLineEdit);
   personalInfoformLayout->addRow("Last Name:", lastNameLineEdit );
   personalInfoformLayout->addRow("Age", ageSpingBox);
   personalInfoformLayout->addRow("Employment Status",
   employmentStatusComboBox);
   window->setLayout(personalInfoformLayout);
   window->show();
   return app.exec();
}
```

The code should look familiar by now. We instantiate objects of the various widgets we want to show in the form. Thereafter, the layout is created:

```cpp
QFormLayout *personalInfoformLayout = new QFormLayout;
```

An instance of `QFormLayout` is created. Anytime we want to add a widget to the layout, `*personalInformformLayout`, we shall call the `addRow()` method, pass a string representing the label and finally the widget we want to align with the label:

```cpp
personalInfoformLayout->addRow("First Name:", firstNameLineEdit);
```

`"First Name: "` is the label and the widget here is `firstNameLineEdit`.

The other widgets are added to the layout like this: 

```cpp
window->setLayout(personalInfoformLayout);
```

`personalInfoformLayout` is then passed to the `setLayout()` method of the `QWidget` instance. This means that the layout for the application window, `window`, is `personalInfoformLayout`.

Remember that the `QWidget` instance, `window`, will become the main window of the application since its `show()` method is called.

`QForm` eliminates the need to specify columns and rows by giving us an easy way to add a row to our layout, and each time we do so, we can specify the label and the widget we want displayed.

You should see this output when you compile and run the project:

![](img/a9a274fc-eae8-472f-ab2d-ef8d3f985723.png)

The preceding screenshot shows how widgets are aligned in those layouts. A form is presented in a question-and-answer manner. The labels are usually on the left-hand side while the widgets that take the user input are on the right-hand side.

# Layouts with direction

There are layouts that provide direction of growth when widgets are added to them. There are instances where we want to align all widgets within a layout horizontally or vertically.

The `QHBoxLayout` and `QVBoxLayout` classes provide this functionality.

# QVBoxLayout

In a `QVBoxLayout` layout, widgets are aligned vertically and they are packed in the layout from top to bottom.

Consider the following diagram:

![](img/46430e28-a761-40e8-a132-f87e584a5ac4.png)

For `QVBoxLayout`, the arrow gives the direction of growth in which the widgets are added to the layout. The first widget, **widget 1**, will occupy the top of the layout, while the last call to `addWidget()` will make **widget 5** occupy the bottom of the layout.

To illustrate how to use the `QVBoxLayout`, consider the following program:

```cpp
#include <QApplication>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   QWidget *window = new QWidget;
   QLabel *label1 = new QLabel("Username");
   QLabel *label2 = new QLabel("Password");
   QLineEdit *usernameLineEdit = new QLineEdit;
   usernameLineEdit->setPlaceholderText("Enter your username");
   QLineEdit *passwordLineEdit = new QLineEdit;
   passwordLineEdit->setEchoMode(QLineEdit::Password);
   passwordLineEdit->setPlaceholderText("Enter your password");
   QPushButton *button1 = new QPushButton("&Login");
   QPushButton *button2 = new QPushButton("&Register");
   QVBoxLayout *layout = new QVBoxLayout;
   layout->addWidget(label1);
   layout->addWidget(usernameLineEdit);
   layout->addWidget(label2);
   layout->addWidget(passwordLineEdit);
   layout->addWidget(button1);
   layout->addWidget(button2);
   window->setLayout(layout);
   window->show();
   return app.exec();
}
```

In previous examples, we indicated the reason why we create an instance of `QWidget`. Two labels are created with the strings `"Username"` and `"Password"`. A textbox, `QLineEdit` instance is also created to receive both username and password input. On the `passwordLineEdit` object, the `setEchoMode()` method is passed the constant `QLineEdit::Password` that masks the input of that textbox and replaces it with dots to prevent the characters that are typed from being readable.

A placeholder text within `passwordLineEdit` is set via the `setPlaceholderText()` method. The placeholder text gives further information about the purpose of the textbox.

Two push buttons are also created, `button1` and `button2`. An instance of `QVBoxLayout` is created. To add widgets to the layout, the `addWidget()` method is called and passed the specific widget. The very first widget passed to `addWidget` will appear on top when displayed. Likewise, the last widget added will show on the bottom, which in this case is `button2`.

The layout for the `window` widget instance is set by passing `layout` to `setLayout()`.

Finally, the `show()` method is called on the window. Compile the project and run it to see the output:

![](img/e006132f-51fa-43ea-b774-7dc6e1a2eca3.png)

In the preceding screenshot, we can see that the first widget that was added to the layout was the label, `label1`, while `button2` (with the text Register) was the last widget occupying the bottom.

# QHBoxLayout

The `QHBoxLayout` layout class is very similar in use to `QVBoxLayout`. Widgets are added to the layout by calling the `addWidget()` method.

Consider the following diagram:

![](img/6e64441d-3ad5-4118-afd3-1e923f9d64af.png)

The arrow in the diagram shows the direction in which widgets grow in number as they are added to a `QHBoxLayout`. The first widget added to this layout is **widget 1**, while **widget 3** is the last widget to be added to the layout.

A small application to allow users to enter a URL makes use of this layout type:

```cpp
#include <QApplication>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLineEdit>
int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   QWidget *window = new QWidget;
   QLineEdit *urlLineEdit= new QLineEdit;
   QPushButton *exportButton = new QPushButton("Export");
   urlLineEdit->setPlaceholderText("Enter Url to export. Eg, http://yourdomain.com/items");
   urlLineEdit->setFixedWidth(400);
   QHBoxLayout *layout = new QHBoxLayout;
   layout->addWidget(urlLineEdit);
   layout->addWidget(exportButton);
   window->setLayout(layout);
   window->show();
   return app.exec();
}
```

A textbox or `QLineEdit` and button are created. A placeholder is set on the `QLineEdit` instance, `urlLineEdit`. To enable the placeholder to be seen, we stretch `urlLineEdit` by setting `setFixedWidth` to `400`.

An instance of `QHBoxLayout` is created and passed to the `layout` pointer. The two widgets, `urlLineEdit` and `exportButton`, are added to the `layout` via the `addWidget()` method.

The layout is set against `window` and the `show()` method of the window is called.

Compile the application and run it. You should see the following output:

![](img/6b6a8c51-00cc-4aaf-9957-c096be8df11d.png)

Refer to [Chapter 1](a157893d-287f-42c4-8166-a7d414e09035.xhtml), *Introducing Qt 5*, to compile the application. For easy compilation process, remember to create a new folder and add the `.cpp` file to it. As usual, the `.pro` file will need to be changed to include the widgets module.

Because the button was added to the layout after textbox, it appears accordingly, standing next to the textbox. If another widget had been added to the layout, it would also appear after the button, `exportButton`.

# Summary

In this chapter, we have looked at a number of widgets that are useful in creating GUI applications. The process of compilation remains the same. We also learned how to use layouts to present and arrange multiple widgets.

Up to this point, our application does not do anything. The `QPushButton` instances, when clicked, do nothing along with the other widgets that are action driven.

In the next chapter, we shall learn how to animate our application so that it responds to actions, thus making them useful.