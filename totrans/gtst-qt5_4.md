# Implementing Windows and Dialog

In the previous chapter, we learned how to animate our application by using signals and slots to trigger and respond to actions that occur within our application. So far, we have been concentrating on examples that are contained in only one file and do not expressly describe a full working application. To do so, we will need to change the style in which our applications are written, and also adopt a number of new conventions.

In this chapter, we shall work with Windows in Qt, so that by the end of the chapter, you should be able to do the following:

*   Understand how to subclass and create a custom window application
*   Add a menu bar to a window
*   Add a toolbar to a window
*   Use the various dialog (boxes) to communicate information to the user

# Creating a custom window

To create a window(ed) application, we usually call the `show()` method on an instance of `QWidget` and that makes that widget, to be contained in a window of its own, along with its child widgets displayed in it.

A recap of such a simple application is as follows:

```cpp
#include <QApplication>
#include <QMainWindow>
#include <QLabel>
int main(int argc, char *argv[])
{
   QApplication a(argc, argv);
   QMainWindow mainWindow;
   mainWindow.show();
   return a.exec();
}
```

`mainWindow` here is an instance of `QMainWindow`, which is derived from `QWidget`. As such, by calling the `show()` method, a window will appear. If you were to replace `QMainWindow` with `QLabel`, this will still work.

But this style of writing applications is not the best. Instead, from this point onward, we shall define our own custom widget, in which we shall define child widgets and make connections between signals and sockets.

Now, let's rewrite the preceding application by sub-classing `QMainWindow`. We have chosen to subclass `QMainWindow` because we need to illustrate the menu and toolbars.

We start off by creating a new folder and defining a header file. The name of our header file here is `mainwindow.h`, but feel free to name it how you want and remember to add the `.h` suffix. This file should basically contain the following:

```cpp
#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <QLabel>
class MainWindow : public QMainWindow
{
   Q_OBJECT
   public:
       MainWindow();
};
#endif
```

We include the Qt classes `QMainWindow`, and `QLabel` in our header file. Then, we subclass `QMainWindow` and call it `MainWindow`. The constructor of this new class is declared with the following:

```cpp
public:
   MainWindow();
```

The entire class definition is wrapped within an `#ifndef ... #endif` directive, which tells the preprocessor to ignore its content if it is accidentally included multiple times in a file.

It is possible to use the non-standard, but widely used, preprocessor directive, `#pragma once`.

Take notice of the `Q_OBJECT` macro. This is what makes the signals and slots mechanism possible. Remember that the C++ language does not know about the keywords used to set up signals and slots. By including this macro, it becomes part of the C++ syntax.

What we have defined so far is just the header file. The body of the main program has to live in some other `.cpp` file. For easy identification, we call it `mainwindow.cpp`. Create this file within the same folder and add the following lines of code:

```cpp
#include "mainwindow.h"
MainWindow::MainWindow()
{
   setWindowTitle("Main Window");
   resize(400, 700);
   QLabel *mainLabel = new QLabel("Main Widget");
   setCentralWidget(mainLabel);
   mainLabel->setAlignment(Qt::AlignCenter);
}
```

We include the header file that we defined earlier with the first line of code. The default constructor of our sub-classed widget, `MainWindow`, is defined.

Notice how we call the method that sets the title of the window. `setWindowTitle()` is invoked and can be accessed from within the constructor since it is an inherited method from `QWindow`. There is no need to use the `this` keyword. The size of the window is specified by calling the `resize()` method and passing two integer values to be used as the dimensions of the window.

An instance of a `QLabel` is created, `mainLabel`. The text within the label is aligned to the center by calling `mainLabel->setAlignment(Qt::AlignCenter)`.

A call to `setCentralWidget()` is important as it situates any class that inherits from `QWidget` to occupy the interior of the window. Here, `mainLabel` is being passed to `setCentralWidget`, and that will make it the only widget to be displayed within the window.

Consider the structure of `QMainWindow` in the following diagram:

![](img/60b5e682-55d6-4878-afe7-5aaded0dbb90.png)

At the very top of every window is the **Menu Bar**. Elements such as the file, edit, and help menus go there. Below that, are the **Toolbars**. Contained within the **Toolbars** are the **Dock Widgets**, which are collapsible panels. Now, the main controls within the window must be put in the **Central Widget** location. Since a UI is made up of several widgets, it will be good to compose a widget that will contain child widgets. This parent widget is what you will stick into the **Central Widget** area. To do this, we call `setCentralWidget()` and pass in the parent widget. At the bottom of the window, is the **Status Bar**.

To run the application, we need to create an instance of our custom window class. Create a file called `main.cpp` within the same folder where the header and `.cpp` files are located. Add the following lines of code to `main.cpp`:

```cpp
#include <QApplication>
#include "mainwindow.h"
int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   MainWindow mainwindow;
   mainwindow.show();
   return app.exec();
}
```

We include the header file `mainwindow.h`, which contains the declaration of our custom class, `MainWindow`. Without this, the compiler wouldn't know where to find the definition of the `MainWindow` class.

An instance of `MainWindow` is created and the `show()` method is called on it. We still have to call the `show()` method on `mainwindow`. `MainWindow`, which is a subclass of `QMainWindow`, and behaves just like any widget out there. Furthermore, as we already know, to cause a widget to appear, you have to call the `show()` method on it.

To run the program, move into the folder via the command line and issue the following commands:

```cpp
% qmake -project
```

Add `QT += widgets` to the `.pro` file that is generated. Now continue with the next set of commands:

```cpp
% qmake
% make
```

Examine the `.pro` file for a second. At the very bottom of the file, we have the following lines:

```cpp
HEADERS += mainwindow.h
SOURCES += main.cpp mainwindow.cpp
```

The headers are automatically collected and added to `HEADERS`. Similarly, the `.cpp` files are collected and added to `SOURCES`. Always remember to check this file when there are compilation errors to ensure that all required files have been added.

To run the program, issue the following command:

```cpp
% ./classSimpleWindow
```

For those who work on the macOS, the correct command you will need to issue in order to run the executable is as follows:

```cpp
% ./classSimpleWindow.app/Contents/MacOS/classSimpleWindow
```

The running application should appear, as follows:

![](img/938e02c9-499f-4930-b00b-da7e73f3b26f.png)

# Menu bar

Most applications hold a set of clickable(s) that reveal a list of another set of actions that expose more functionality to the user. The most popular among these are the File, Edit, and Help menus.

In Qt, menu bars occupy the very top of the window. We shall create a short program to make use of the menu bar.

Three files must be created in a newly created folder. These are as follows:

*   `main.cpp`
*   `mainwindow.h`
*   `mainwindow.cpp`

The `main.cpp` file will remain as before in terms of content. Therefore, copy the `main.cpp` file from the previous section. Let's examine the `mainwindow.h` file:

```cpp
#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <QApplication>
#include <QAction>
#include <QtGui>
#include <QAction>
#include <QMenuBar>
#include <QMenu>
#include <Qt>
class MainWindow : public QMainWindow
{
   Q_OBJECT
   public:
       MainWindow();
   private slots:
   private:
       // Menus
       QMenu *fileMenu;
       QMenu *helpMenu;
       // Actions
       QAction *quitAction;
       QAction *aboutAction;
       QAction *saveAction;
       QAction *cancelAction;
       QAction *openAction;
       QAction *newAction;
       QAction *aboutQtAction;

};
#endif
```

Once more, the header file is enclosed in an `ifndef` directive to prevent errors that may occur as a result of multiple inclusions of this file.

To create a menu within the window, you need instances of `QMenu`. Each menu, such as the File menu, will have sub-menus or items that make up the menu. The File menu usually has the Open, New, and Close sub-menus.

A typical image of a Menu bar is as follows, with the File, Edit, and Help menus. The File menu items under the File menu are New..., Open..., Save, Save As..., and Quit:

![](img/30d1b054-175c-40e9-af76-2448906c0f2a.png)

Our application will have only two menus, namely, `fileMenu` and `helpMenu`. The other instances of `QAction` are the individual menu items: `quitAction`, `saveAction`, `cancelAction`, and `newAction`.

Both the menu and sub-menu items are defined as members of the class in the header file. Furthermore, this kind of declaration will allow users to modify their behavior and also to easily access them when connecting them to sockets.

Now, let's switch to the `mainwindow.cpp`. Copy the following code into `mainwindow.cpp`:

```cpp
#include "mainwindow.h"
MainWindow::MainWindow()
{
   setWindowTitle("SRM System");
   setFixedSize(500, 500);
   QPixmap newIcon("new.png");
   QPixmap openIcon("open.png");
   QPixmap closeIcon("close.png");
   // Setup File Menu
   fileMenu = menuBar()->addMenu("&File");
   quitAction = new QAction(closeIcon, "Quit", this);
   quitAction->setShortcuts(QKeySequence::Quit);
   newAction = new QAction(newIcon, "&New", this);
   newAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_C));
   openAction = new QAction(openIcon, "&New", this);
   openAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_O));
   fileMenu->addAction(newAction);
   fileMenu->addAction(openAction);
   fileMenu->addSeparator();
   fileMenu->addAction(quitAction);
   helpMenu = menuBar()->addMenu("Help");
   aboutAction = new QAction("About", this);
   aboutAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_H));
   helpMenu->addAction(aboutAction);
   // Setup Signals and Slots
   connect(quitAction, &QAction::triggered, this, &QApplication::quit);
}
```

The header file, `mainwindow.h`, is included at the beginning of the file to make available the class declaration and Qt classes that will be used in the program.

In the default constructor of our custom class, `MainWindow`, we start by setting the name of our window by calling `setWindowTitle()` and giving it an appropriate name. The size of our window is then established by calling `setFixedSize()`. This is demonstrated in the following code block:

```cpp
QPixmap newIcon("new.png");
QPixmap openIcon("open.png");
QPixmap closeIcon("close.png");
```

Menu items can be displayed with images beside them. To associate an image or icon with a menu item, `QAction`, you need to first capture that image within an instance of `QPixmap`. Three such images are captured in the `newIcon`, `openIcon`, and `closeIcon` variables. These will be used further down the code.

Let's set up the `fileMenu` as follows:

```cpp
fileMenu = menuBar()->addMenu("&File");
quitAction = new QAction(closeIcon, "Quit", this);
quitAction->setShortcuts(QKeySequence::Quit);
```

To add a menu to the window, a call to `menuBar()` is made. This returns an instance of `QMenu`, and we call `addMenu` on that object specifying the name of the menu we want to add. Here, we call our first menu, File. The `"&"` sign in front of the F in File will make it possible to press *Alt* + *F* on the keyboard.

`quitAction` is passed an instance of `QAction()`. `closeIcon` is the image we want to associate with this sub-menu. `"Quit"` is the display name and the `this` keyword makes the `quitAction` a child widget of `MainWindow`.

A shortcut to a sub-menu is associated with `quitAction` by calling `setShortcuts()`. By using `QKeySequence::Quit`, we mask the need to cater for platform-specific key sequences that are used.

`newAction` and `openAction` follow the same logic in their creation.

Now that we have our menu in `fileMenu` and the menu items in `quitAction`, `newAction`, and `openActions`, we need to link them together:

```cpp
fileMenu->addAction(newAction);
fileMenu->addAction(openAction);
fileMenu->addSeparator();
fileMenu->addAction(quitAction);
```

To add a sub-menu item, we call the `addAction()` method on the `QMenu` instance, `fileMenu`, and pass the required `QAction` instance. The `addSeparator()` is used to insert a visual marker in our list of menu items. It also returns an instance of `QAction`, but we are not interested in that object at this moment.

A second menu is added to the application along with its only sub-menu item:

```cpp
helpMenu = menuBar()->addMenu("Help");
aboutAction = new QAction("About", this);
aboutAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_H));
helpMenu->addAction(aboutAction);
```

`QAction` encapsulates a general idea of an action that can be inserted into widgets. Here, we used `QAction` to insert actions into our menus.

These `QAction` instances emit the `triggered` signal, which can be connected to a socket to cause the application to change, as follows:

```cpp
connect(quitAction, &QAction::triggered, this, &QApplication::quit);
```

When connecting a signal to a slot within a class definition, simply call the `connect()` method and pass in the parameters as you would do normally. The first parameter is the object that is going to emit the signal we are interested in. `&QAction::triggered` is one way of specifying the triggered signal. This is the same as writing `SIGNAL(triggered())`. The `this` keyword refers to the `MainWindow` object that will be created in the future. The quit slot is specified by `&QApplication::quit`.

The signal and slot connected will create a situation where, when the File menu is opened and the Close button is clicked, the application will close.

The last file needed to run this example is the `main.cpp` file. The previous `main.cpp` file created should be copied over to this project.

Compile and run the project. A typical output should be as follows:

![](img/50cfe2b4-db39-4bac-a735-a8e38c568181.png)

On a Mac, press the key combination *Command* + *Q* and that will close the application. On Linux and Windows, *Alt* + *F4* should do the same. This is made possible by the following line of code:

```cpp
quitAction->setShortcuts(QKeySequence::Quit);
```

This line of code blurs out the difference by relying on Qt's `QKeySequence::Quit`, depending on the OS in use.

Click on the File menu and select New:

![](img/8001058b-a450-40fb-bfd5-c53630e89a43.png)

Nothing happens. That is because we did not define what should happen when the user clicks on that action. The last menu item, Quit, on the other hand, closes the application as defined by the socket and slot we declared.

Also, take note of how each menu item has an appropriate icon or image in front of it.

Visit the Packt website to obtain the images for this book.

# Toolbar

Beneath the menu bar is a panel that is usually referred to as toolbar. It contains a set of controls that could be widgets or instances of `QAction`, just as we saw in their use in creating the menu bar. This also means that you may choose to replace the `QAction` with a widget, such as a regular `QPushButton` or `QComboBox`.

Toolbars may be fixed to the top of the window (beneath the menu bar) and can be pinned there or made to float around the dock widget.

Once again, we will need to create a new project or modify the one from the previous section of this chapter. The files that we will be creating are `main.cpp`, `mainwindow.h`, and `mainwindow.cpp`.

The `main.cpp` file remains the same, as follows. We only instantiate our custom class and call `show()` on it:

```cpp
#include <QApplication>
#include "mainwindow.h"
int main(int argc, char *argv[])
{
   QApplication app(argc, argv);
   QCoreApplication::setAttribute(Qt::AA_DontUseNativeMenuBar); //
   MainWindow mainwindow;
   mainwindow.show();
   return app.exec();
}
```

The `mainwindow.h` file will essentially contain the `QAction` members that will hold the actions in our toolbar:

```cpp
#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <QApplication>
#include <QAction>
#include <QPushButton>
#include <QAction>
#include <QMenuBar>
#include <QMenu>
#include <QtGui>
#include <Qt>
#include <QToolBar>
#include <QTableView>
class MainWindow : public QMainWindow
{
   Q_OBJECT
   public:
       MainWindow();
   private slots:
   private:
       // Menus
       QMenu *fileMenu;
       QMenu *helpMenu;
       // Actions
       QAction *quitAction;
       QAction *aboutAction;
       QAction *saveAction;
       QAction *cancelAction;
       QAction *openAction;
       QAction *newAction;
       QAction *aboutQtAction;
       QToolBar *toolbar;
       QAction *newToolBarAction;
       QAction *openToolBarAction;
       QAction *closeToolBarAction;
};
#endif
```

This header file appears the same as before. The only difference is the `QToolbar` instance, `*toolbar`, and the `QAction` objects that will be shown within the toolbar. These are `newToolBarAction`, `openToolBarAction`, and `closeToolBarAction`. The `QAction` instances that are used in a menu are the same as the ones used for toolbars.

Note that there are no slots being declared.

The `mainwindow.cpp` file will contain the following:

```cpp
#include "mainwindow.h"
MainWindow::MainWindow()
{
   setWindowTitle("Form in Window");
   setFixedSize(500, 500);
   QPixmap newIcon("new.png");
   QPixmap openIcon("open.png");
   QPixmap closeIcon("close.png");
   // Setup File Menu
   fileMenu = menuBar()->addMenu("&File");
   quitAction = new QAction(closeIcon, "Quit", this);
   quitAction->setShortcuts(QKeySequence::Quit);
   newAction = new QAction(newIcon, "&New", this);
   newAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_C));
   openAction = new QAction(openIcon, "&New", this);
   openAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_O));
   fileMenu->addAction(newAction);
   fileMenu->addAction(openAction);
   fileMenu->addSeparator();
   fileMenu->addAction(quitAction);
   helpMenu = menuBar()->addMenu("Help");
   aboutAction = new QAction("About", this);
   aboutAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_H));
   helpMenu->addAction(aboutAction);
   // Setup Tool bar menu
   toolbar = addToolBar("main toolbar");
   // toolbar->setMovable( false );
   newToolBarAction = toolbar->addAction(QIcon(newIcon), "New File");
   openToolBarAction = toolbar->addAction(QIcon(openIcon), "Open File");
   toolbar->addSeparator();
   closeToolBarAction = toolbar->addAction(QIcon(closeIcon), "Quit Application");
   // Setup Signals and Slots
   connect(quitAction, &QAction::triggered, this, &QApplication::quit);
   connect(closeToolBarAction, &QAction::triggered, this, &QApplication::quit);
}
```

The same set of icons used for the menu bar will be used for the toolbars too.

To obtain an instance of the Windows toolbar for further manipulation, call the `addTooBar()` method, which will return an instance of a `QToolBar`. The method accepts any text that is used as the title of the window. It also adds the toolbar to the window.

The toolbar at this point can be moved around within the window. To fix it to the top of the window, call the `toolbar->setMovable(false);` function on the instance of the `QToolBar`, `toolbar`:

```cpp
newToolBarAction = toolbar->addAction(QIcon(newIcon), "New File");
openToolBarAction = toolbar->addAction(QIcon(openIcon), "Open File");
toolbar->addSeparator();
closeToolBarAction = toolbar->addAction(QIcon(closeIcon), "Quit Application");
```

Two `QAction` objects are created and passed to the `newToolBarAction` and `openToolBarAction` objects. We pass the `QIcon` object that becomes the image on the `QAction` and a name or text to be displayed as a tooltip. A separator is added to the toolbar by calling the `addSeparator()` method. The last control, `closeToolBarAction`, contains an image to be displayed on the toolbar.

To link the trigger signal of `closeToolBarAction` to the quit slot of the window, we do the following:

```cpp
connect(closeToolBarAction, &QAction::triggered, this, &QApplication::quit);
```

To compile this project as a recap, run the following commands:

```cpp
% qmake -project
```

Add `QT += widgets` to the `.pro` file that is generated and make sure all three files are listed in the bottom of the file:

Proceed to issue the following commands in order to build the project:

```cpp
% qmake
% make
% ./name_of_executable
```

If everything went well, you will see the following:

![](img/e49d334c-3989-4115-99c9-479ade2fadc8.png)

The preceding screenshot shows the toolbar beneath the File and Help menus. Three icons show three `QAction` objects that represent the New, Open, and Close actions. Only the last button (to close the application) action works. That is because we only defined a single signal-slot connection for the `closeToolBarAction` and `QAction` objects.

By hovering the mouse over the toolbar menu items, some text appears. This message is called a tooltip. As can be seen in the preceding diagram, the Open File message is derived from the last parameter of the following line:

```cpp
openToolBarAction = toolbar->addAction(QIcon(openIcon), "Open File");
```

As noted earlier, a toolbar can be moved around within a window as follows:

![](img/8872f67b-6a3c-4f2b-95cc-430acafc0f19.png)

As you can see, by clicking on the three vertical dots on the left-hand side of the toolbar and moving it, you can detach the toolbar from the top to either the left, right, or bottom. To display this kind of functionality, issue the following command:

```cpp
toolbar->setMovable(false);
```

This will fix the toolbar to the top so that it can't be moved around.

# Adding other widgets

So far, we have only added a menu bar and a toolbar to our window. To add other widgets that might make our application useful, we have to add more members to our header file. In this section, we shall create a simple application that appends personal details to a displayable list.

There will be a form where the details of a number of contacts will be received. This detail will then be added to a list on the window. As more contacts are added, the list will grow. We shall base these on the previous section's code and continue to build on it.

As usual, you create a new folder with the three files, namely, `main.cpp`, `mainwindow.cpp`, and `mainwindow.h`. The `main.cpp` file will remain as before from the previous sections.

The `mainwindow.h` file should contain the following lines of code:

```cpp
#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <QApplication>
#include <QLabel>
#include <QLineEdit>
#include <QDate>
#include <QDateEdit>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QPushButton>
#include <QMessageBox>
#include <QAction>
#include <QMenuBar>
#include <QMenu>
#include <QtGui>
#include <Qt>
#include <QToolBar>
#include <QTableView>
#include <QHeaderView>
```

The file imports the classes that will be used in declaring the members within our custom class. The whole file is wrapped with the `#ifndef` directive so that the header file can be included multiple times without yielding errors.

Add the following lines of code to the same header file, `mainwindow.h`:

```cpp
class MainWindow : public QMainWindow
{
   Q_OBJECT
   public:
       MainWindow();
   private slots:
       void saveButtonClicked();
```

We then declare our default constructor for our class.

There is only one slot in our application that will be used to move the content of a number of widgets into a list.

Continue the code listing by adding the following lines of code that will add the members of the class and define the prototype of some `helper` methods:

```cpp
   private:
       // Widgets
       QWidget *mainWidget;
       QVBoxLayout *centralWidgetLayout;
       QGridLayout *formLayout;
       QHBoxLayout *buttonsLayout;
       QLabel *nameLabel;
       QLabel *dateOfBirthLabel;
       QLabel *phoneNumberLabel;
       QPushButton *savePushButton;
       QPushButton *newPushButton;
       QLineEdit *nameLineEdit;
       QDateEdit *dateOfBirthEdit;
       QLineEdit *phoneNumberLineEdit;
       QTableView *appTable;
       QStandardItemModel *model;
       // Menus
       QMenu *fileMenu;
       QMenu *helpMenu;
       // Actions
       QAction *quitAction;
       QAction *aboutAction;
       QAction *saveAction;
       QAction *cancelAction;
       QAction *openAction;
       QAction *newAction;
       QAction *aboutQtAction;
       QAction *newToolBarAction;
       QAction *openToolBarAction;
       QAction *closeToolBarAction;
       QAction *clearToolBarAction;
       // Toolbar
       QToolBar *toolbar;
       // Icons
       QPixmap newIcon;
       QPixmap openIcon;
       QPixmap closeIcon;
       QPixmap clearIcon;
       // init methods
       void clearFields();
       void createIcons();
       void createMenuBar();
       void createToolBar();
       void setupSignalsAndSlot();
       void setupCoreWidgets();
};
#endif
```

The members include layout and other widget classes, classes for our menu, toolbars, and their associated `QAction` objects.

As you can see, the code is borrowed from the previous section with the exception of the widgets being added.

The private methods, `createIcons()`, `createMenuBar()`, `createToolBar()`, `setupSignalsAndSlot()`, and `setupCoreWidgets()`, will be used to refactor the code that should live in our default constructor. The `clearFields()` method will be used to clear the data from a number of widgets.

In the `mainwindow.cpp` file, we shall define our class with the following lines of code:

```cpp
#include "mainwindow.h"
#include "mainwindow.h"
MainWindow::MainWindow()
{
   setWindowTitle("Form in Window");
   setFixedSize(500, 500);
   createIcons();
   setupCoreWidgets();
   createMenuBar();
   createToolBar();
   centralWidgetLayout->addLayout(formLayout);
   centralWidgetLayout->addWidget(appTable);
   centralWidgetLayout->addLayout(buttonsLayout);
   mainWidget->setLayout(centralWidgetLayout);
   setCentralWidget(mainWidget);
   setupSignalsAndSlots();
}
```

The default constructor has been refactored a great deal here. The building blocks of code have been moved away into functions to help make the code readable.

Now, we only set the window title and size of the application window. Next, we call the method that will create the icons that will be used by the various widgets. Another function call is made to set up the core widgets by calling the `setupCoreWidgets()` method. The menu and toolbars are created by calling the `createMenuBar()` and `createToolBar()` methods.

The layout object, `centralWidgetLayout`, is the main layout of our application. We add the `formLayout` object first, followed by the `appTable` object. As you can see, it is possible to insert a layout into another layout. Lastly, we insert the `buttonsLayout` object, which contains our buttons.

The `mainWidget` object's layout is set to `centralWidgetLayout`. This `mainWidget` object is then set as the main widget that should occupy the center of the window, as was demonstrated in the first diagram of this chapter.

All signals and slots will be set up in the `setupSignalsAndSlot()` method.

Add the following lines of code to the `mainwindow.cpp` file that defines the `createIcons()` method:

```cpp
void MainWindow::createIcons() {
    newIcon = QPixmap("new.png");
    openIcon = QPixmap("open.png");
    closeIcon = QPixmap("close.png");
    clearIcon = QPixmap("clear.png");
}
```

The `createIcons()` method will pass instances of `QPixmap` to the members that were declared in `mainwindow.h`.

The definition of `setupCoreWidgets()` is as follows, in `mainwindow.cpp`:

```cpp
void MainWindow::setupCoreWidgets() {
   mainWidget = new QWidget();
   centralWidgetLayout = new QVBoxLayout();
   formLayout = new QGridLayout();
   buttonsLayout = new QHBoxLayout();
   nameLabel = new QLabel("Name:");
   dateOfBirthLabel= new QLabel("Date Of Birth:");
   phoneNumberLabel = new QLabel("Phone Number");
   savePushButton = new QPushButton("Save");
   newPushButton = new QPushButton("Clear All");
   nameLineEdit = new QLineEdit();
   dateOfBirthEdit = new QDateEdit(QDate::currentDate());
   phoneNumberLineEdit = new QLineEdit();
   // TableView
   appTable = new QTableView();
   model = new QStandardItemModel(1, 3, this);
   appTable->setContextMenuPolicy(Qt::CustomContextMenu);
   appTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch); /** Note **/
   model->setHorizontalHeaderItem(0, new QStandardItem(QString("Name")));
   model->setHorizontalHeaderItem(1, new QStandardItem(QString("Date of Birth")));
   model->setHorizontalHeaderItem(2, new QStandardItem(QString("Phone Number")));   appTable->setModel(model)

   QStandardItem *firstItem = new QStandardItem(QString("G. Shone"));
   QDate dateOfBirth(1980, 1, 1);
   QStandardItem *secondItem = new QStandardItem(dateOfBirth.toString());
   QStandardItem *thirdItem = new QStandardItem(QString("05443394858"));
   model->setItem(0,0,firstItem);
   model->setItem(0,1,secondItem);
   model->setItem(0,2,thirdItem);
   formLayout->addWidget(nameLabel, 0, 0);
   formLayout->addWidget(nameLineEdit, 0, 1);
   formLayout->addWidget(dateOfBirthLabel, 1, 0);
   formLayout->addWidget(dateOfBirthEdit, 1, 1);
   formLayout->addWidget(phoneNumberLabel, 2, 0);
   formLayout->addWidget(phoneNumberLineEdit, 2, 1);
   buttonsLayout->addStretch();
   buttonsLayout->addWidget(savePushButton);
   buttonsLayout->addWidget(newPushButton);
}
```

Here, we are just instantiating objects to be used within the application. There is nothing out of the ordinary here. `nameLineEdit` and `phoneNumberLineEdit` will be used to collect the name and phone number of contacts about to be saved. `dateOfBirthEdit` is a special kind of textbox that allows you to specify a date. `savePushButton` and `newPushButton` are buttons that will be used to trigger the saving of the contact and the clearing of the list.

The labels and line edit controls will be used in the `formLayout` object, which is a `QGridLayout` instance. `QGridLayout` allows widgets to be specified using columns and rows.

To save a contact, this means we will save it to a widget that can display a list of items. Qt has a number of such widgets. These include `QListView`, `QTableView`, and `QTreeView`.

When the `QListView` is used in displaying information, it will typically appear as in the following screenshot:

![](img/888d6e5f-7929-48f2-b591-542d7dc51a2e.png)

`QTableView` will use columns and rows to display data or information in cells as follows:

![](img/6785168c-b18c-4ac9-aa72-83a5ffedf13f.png)

To show hierarchical information, `QTreeView` is also used, as in the following screenshot:

![](img/a6707e92-a5ed-463e-aaf9-3f73dc596906.png)

An instance of `QTableView` is passed to `appTable`. We need a model for our `QTableView` instance. The model will hold the data that will be displayed in our table. When data is added or removed from the model, its corresponding view will be updated to show the change that has occurred, automatically. The model here is an instance of `QStandardItemModel`. The line `QStandardItemModel(1, 3, this)` will create an instance with one row and three columns. The `this` keyword is used to make the model a child of the `MainWindow` object:

```cpp
appTable->setContextMenuPolicy(Qt::CustomContextMenu);
```

This line is used to help us define a custom action that should happen when we raise a context menu on the table:

```cpp
appTable->horizontalHeader()->setSectionResizeMode(
QHeaderView::Stretch); /** Note **/
```

The preceding line is important and enables the headers of our table to stretch out fully. This is the result when we omit that line (as shown in an area bounded by the red box):

![](img/66e16468-6820-4379-b2d2-69249b06729d.png)

Ideally, we want our table to have the following header, so that it looks like this:

![](img/502811d8-85bb-465b-9568-e8c9bccae68f.png)

To set the header for the table, we can do so with the following lines of code:

```cpp
model->setHorizontalHeaderItem(0, new QStandardItem(QString("Name")));
```

The table for displaying the contacts needs headers. The `setHorizontalHeaderItem()` method on the model object uses the first parameter to indicate the position where the new `QStandardItem(QString())` should be inserted. Because our table uses three columns, the line is repeated three times for the headers, Name, Date of Birth, and Phone Number:

```cpp
appTable->setModel(model);
QStandardItem *firstItem = new QStandardItem(QString("G. Shone"));
QDate dateOfBirth(1980, 1, 1);
QStandardItem *secondItem = new QStandardItem(dateOfBirth.toString());
QStandardItem *thirdItem = new QStandardItem(QString("05443394858"));
model->setItem(0,0,firstItem);
model->setItem(0,1,secondItem);
model->setItem(0,2,thirdItem);
```

We make `model` the model of our `QTableView` by calling `setModel()` on `appTable` and passing `model` as a parameter.

To populate our model, which updates its view, `QTableView`, we shall create instances of `QStandardItem`. Each cell in our table has to be encapsulated in this class. `dateOfBirth` is of the `QDate` type, so we call `toString()` on it and pass it to `new QStandardItem()`. `firstItem` is inserted into our model by specifying the row and column as in the line `model->setItem(0, 0, firstItem);`.

This is done for the second and third `QStandardItem` objects.

Now, let's populate our `formLayout` object. This is of the `QGridLayout` type. To insert widgets into our layout, use the following lines of code:

```cpp
formLayout->addWidget(nameLabel, 0, 0);
formLayout->addWidget(nameLineEdit, 0, 1);
formLayout->addWidget(dateOfBirthLabel, 1, 0);
formLayout->addWidget(dateOfBirthEdit, 1, 1);
formLayout->addWidget(phoneNumberLabel, 2, 0);
formLayout->addWidget(phoneNumberLineEdit, 2, 1);
```

We add widgets to the layout by calling `addWidget()`, supplying the widget, and the row and column it is supposed to fill. `0, 0` will fill the first cell, `0, 1` will fill the second cell on the first row, and `1, 0` will fill the first cell on the second row.

The following code adds buttons to the `QHBoxLayout` instance of `buttonsLayout`:

```cpp
buttonsLayout->addStretch();
buttonsLayout->addWidget(savePushButton);
buttonsLayout->addWidget(newPushButton);
```

To push `savePushButton` and `newPushButton` to the right, we first add a stretch that will expand and fill the empty space by calling `addStretch()` before a call to add the widgets is made by `addWidget()`.

Before we come to the menus in the application, add the following code. To include menus and a toolbar to our application, add the definition of `createMenuBar()` and `createToolBar()` to the `mainwindow.cpp` file:

```cpp
void MainWindow::createMenuBar() {
   // Setup File Menu
   fileMenu = menuBar()->addMenu("&File");
   quitAction = new QAction(closeIcon, "Quit", this);
   quitAction->setShortcuts(QKeySequence::Quit);
   newAction = new QAction(newIcon, "&New", this);
   newAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_C));
   openAction = new QAction(openIcon, "&New", this);
   openAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_O));
   fileMenu->addAction(newAction);
   fileMenu->addAction(openAction);
   fileMenu->addSeparator();
   fileMenu->addAction(quitAction);
   helpMenu = menuBar()->addMenu("Help");
   aboutAction = new QAction("About", this);
   aboutAction->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_H));
   helpMenu->addAction(aboutAction);
}
void MainWindow::createToolBar() {
   // Setup Tool bar menu
   toolbar = addToolBar("main toolbar");
   // toolbar->setMovable( false );
   newToolBarAction = toolbar->addAction(QIcon(newIcon), "New File");
   openToolBarAction = toolbar->addAction(QIcon(openIcon), "Open File");
   toolbar->addSeparator();
   clearToolBarAction = toolbar->addAction(QIcon(clearIcon), "Clear All");
   closeToolBarAction = toolbar->addAction(QIcon(closeIcon), "Quit Application");
}
```

The preceding code is familiar code that adds a toolbar and menus to our window. The final lines of code define the `setupSignalsAndSlots()` method:

```cpp
void MainWindow::setupSignalsAndSlots() {
   // Setup Signals and Slots
   connect(quitAction, &QAction::triggered, this, &QApplication::quit);
   connect(closeToolBarAction, &QAction::triggered, this, &QApplication::quit);
   connect(savePushButton, SIGNAL(clicked()), this, SLOT(saveButtonClicked()));
}
```

In the preceding code, we connect the triggered signal of `quitAction` to the quit slot of `QApplication`. The triggered signal of `closeToolBarAction` is connected to the same, to achieve the effect of closing the application.

The `clicked()` signal of `savePushButton` is connected to the slot, `saveButtonClicked()`. Because it is defined within our class, the `this` keyword is used in the third parameter.

The exact operation that ensures that the information input into the form is saved, is defined by the `saveButtonClicked()` function that serves a slot.

To define our slot, add the following code to `mainwindow.cpp`:

```cpp
void MainWindow::saveButtonClicked()
{
  QStandardItem *name = new QStandardItem(nameLineEdit->text());
  QStandardItem *dob = new QStandardItem(dateOfBirthEdit->date().toString());
   QStandardItem *phoneNumber = new QStandardItem(phoneNumberLineEdit->text());
   model->appendRow({ name, dob, phoneNumber});
   clearFields();
}
```

When `saveButtonClicked()` is invoked, we shall extract the values within the controls, `nameLinedEdit`, `dateOfBirthEdit`, and `phoneNumberLineEdit`. We append them to the model by calling `appendRow()` on the model object. We can access the model object because it is a member point variable in our class definition.

After appending the new contact information into the list, all the fields are cleared and reset with a call to `clearFields()`.

To clear the fields, we call `clearFields()`, which is defined in `mainwindow.cpp` as follows:

```cpp
void MainWindow::clearFields()
{
   nameLineEdit->clear();
   phoneNumberLineEdit->setText("");
   QDate dateOfBirth(1980, 1, 1);
   dateOfBirthEdit->setDate(dateOfBirth);
}
```

The `nameLineEdit` object is reset to an empty string by calling the `clear()` method. This method also doubles as a slot. Another way to set a `QLineEdit` object to an empty string is by setting the text to `""` by calling the `setText("")`:

Because `QDateEdit` accepts dates, we have to create an instance of `date` and pass it to `setDate()` of `dateOfBirthEdit`.

Compile and run the project. You should see the following output:

![](img/4d7a1cf3-4101-4cb6-a5dc-1c13ec5f94d5.png)

To add a new contact, complete the form and click on the Save button:

![](img/ef6535e7-49b9-44d5-a228-e984c353d355.png)

After clicking on the Save button, you should see the following:

![](img/b28d12a8-35dd-48b8-8837-1a2f8a987d87.png)

# Adding dialog boxes

There are times when an application needs to inform the user of an action or to receive input for further processing. Usually, another window, typically small in size, will appear with such information or instructions. In Qt, the `QMessageBox` provides us with the functionality to raise alerts and receive input using `QInputDialog`.

There are different messages, as explained in the following table:

![](img/6262d2fe-0591-4cc7-b1c7-c4626529abe6.png)

To raise an instance of `QMessage` to communicate a recently accomplished task to the user, the following code listing can serve as an example:

```cpp
QMessageBox::information(this, tr("RMS System"), tr("Record saved successfully!"),QMessageBox::Ok|QMessageBox::Default,
QMessageBox::NoButton, QMessageBox::NoButton);
```

The preceding code listing will yield an output such as the following:

![](img/b70d4daf-c9c9-4298-9519-ec4aab559d4a.png)

This `QMessageBox` instance is being used to communicate to the user that an operation was successful.

The icon and number of buttons on a `QMessageBox` instance is configurable.

Let's complete the contact application being written to show how `QMessageBox` and `QInputDialog` are used.

Choose to build upon the example in the previous section or create a new folder with the three main files we have been working with so far, that is, `main.cpp`, `mainwindow.cpp`, and `mainwindow.h`.

The `mainwindow.h` file should contain the following:

```cpp
#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <QApplication>
#include <QLabel>
#include <QLineEdit>
#include <QDate>
#include <QDateEdit>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QPushButton>
#include <QMessageBox>
#include <QAction>
#include <QMenuBar>
#include <QMenu>
#include <QtGui>
#include <Qt>
#include <QToolBar>
#include <QTableView>
#include <QHeaderView>
#include <QInputDialog>
class MainWindow : public QMainWindow
{
   Q_OBJECT
   public:
       MainWindow();
   private slots:
       void saveButtonClicked();
       void aboutDialog();
       void clearAllRecords();
       void deleteSavedRecord();
   private:
       // Widgets
       QWidget *mainWidget;
       QVBoxLayout *centralWidgetLayout;
       QGridLayout *formLayout;
       QHBoxLayout *buttonsLayout;
       QLabel *nameLabel;
       QLabel *dateOfBirthLabel;
       QLabel *phoneNumberLabel;
       QPushButton *savePushButton;
       QPushButton *clearPushButton;
       QLineEdit *nameLineEdit;
       QDateEdit *dateOfBirthEdit;
       QLineEdit *phoneNumberLineEdit;
       QTableView *appTable;
       QStandardItemModel *model;
       // Menus
       QMenu *fileMenu;
       QMenu *helpMenu;
       // Actions
       QAction *quitAction;
       QAction *aboutAction;
       QAction *saveAction;
       QAction *cancelAction;
       QAction *openAction;
       QAction *newAction;
       QAction *aboutQtAction;
       QAction *newToolBarAction;
       QAction *openToolBarAction;
       QAction *closeToolBarAction;
       QAction *clearToolBarAction;
       QAction *deleteOneEntryToolBarAction;
       // Icons
       QPixmap newIcon;
       QPixmap openIcon;
       QPixmap closeIcon;
       QPixmap clearIcon;
       QPixmap deleteIcon;
       // Toolbar
       QToolBar *toolbar;
       void clearFields();
       void createIcons();
       void createMenuBar();
       void createToolBar();
       void setupSignalsAndSlots();
       void setupCoreWidgets();
};
#endif
```

The only notable change is the increase in the number of slots. The `saveButtonClicked()` slot will be reimplemented to pop up a message telling the user of a successful save action. The `aboutDialog()` slot will be used to show an about message. This is usually a window that conveys information about the program and usually contains copyright, help, and contact information.

The `clearAllRecords()` slot will invoke a question message box that will prompt the user of the destructive action about to be taken. `deleteSavedRecord()` will use `QInputDialog` to accept input from the user as to which row to remove from our list of saved contacts.

`QAction *aboutQtAction` will be used to invoke the slot to display the about page or message. We shall also add a toolbar action, `QAction *deleteOneEntryToolBarAction`, that will be used to invoke a dialog box that will receive input from the user. Observe these three inputs, `QPixmap deleteIcon`, `QPixmap clearIcon`, and `QPixmap deleteIcon`, as we add more actions to the window and, likewise, the `QPushButton*clearPushButton`, which is replacing `newPushButton` in the previous example.

Everything else about the header file remains the same. The two extra classes imported are the `QMessageBox` and `QInputDialog` classes.

In the `mainwindow.cpp` file, we define the default constructor of the `MainWindow` class as follows:

```cpp
#include "mainwindow.h"
MainWindow::MainWindow()
{
   setWindowTitle("RMS System");
   setFixedSize(500, 500);
   setWindowIcon(QIcon("window_logo.png"));
   createIcons();
   setupCoreWidgets();
   createMenuBar();
   createToolBar();
   centralWidgetLayout->addLayout(formLayout);
   centralWidgetLayout->addWidget(appTable);
   //centralWidgetLayout->addStretch();
   centralWidgetLayout->addLayout(buttonsLayout);
   mainWidget->setLayout(centralWidgetLayout);
   setCentralWidget(mainWidget);
   setupSignalsAndSlots();
}
```

This time, we want to give the whole application an icon that will show up in a taskbar or dock when it is running. To do this, we call the `setWindowIcon()` method and pass in an instance of `QIcon("window_logo.png")`.

The `window_logo.png` file is included in the project, along with the other image files being used as an attachment on the Packt site for this book.

Everything remains the same as before in the previous example. The methods that are setting up the various parts of the application have been modified slightly.

The `setupSignalsAndSlots()` method is implemented with the following lines of code:

```cpp
void MainWindow::setupSignalsAndSlots() {
   // Setup Signals and Slots
   connect(quitAction, &QAction::triggered, this, &QApplication::quit);
   connect(aboutAction, SIGNAL(triggered()), this, SLOT(aboutDialog()));
   connect(clearToolBarAction, SIGNAL(triggered()), this, SLOT(clearAllRecords()));
   connect(closeToolBarAction, &QAction::triggered, this, &QApplication::quit);
   connect(deleteOneEntryToolBarAction, SIGNAL(triggered()), this, SLOT(deleteSavedRecord()));
   connect(savePushButton, SIGNAL(clicked()), this, SLOT(saveButtonClicked()));
   connect(clearPushButton, SIGNAL(clicked()), this, SLOT(clearAllRecords()));
}
```

The `triggered()` signal of `aboutAction` is connected to the `aboutDialog()`. slot. This method raises a dialog box that is used to display a window with some information about the application and a logo of the app (which we have defined by calling `setWindowIcon()`):

```cpp
void MainWindow::aboutDialog()
{
   QMessageBox::about(this, "About RMS System","RMS System 2.0" "<p>Copyright &copy; 2005 Inc." "This is a simple application to demonstrate the use of windows," "tool bars, menus and dialog boxes");
}
```

The static method, `QMessageBox::about()`, is called with `this` as its first argument. The title of the window is the second argument, and a string that describes the application is given as the third parameter.

At runtime, click on the Help menu and then click on About. You should see the following output:

![](img/c4a86188-c4c5-434a-9052-e43a41ea2e2d.png)

The third signal-slot connection that is established in the `setupSignalsAndSlots()` method is as follows:

```cpp
connect(clearToolBarAction, SIGNAL(triggered()), this, SLOT(clearAllRecords()));
```

In the `clearAllRecords()` slot, we will first ask the user with the aid of a prompt if they are sure they want all the items in a model to be removed. This can be achieved by the following code:

```cpp
int status = QMessageBox::question( this, tr("Delete Records ?"), tr("You are about to delete all saved records "
"<p>Are you sure you want to delete all records "),                                   QMessageBox::No|QMessageBox::Default, QMessageBox::No|QMessageBox::Escape, QMessageBox::NoButton);
if (status == QMessageBox::Yes)
   return model->clear();
```

`QMessageBox::question` is used to raise a dialog to ask the user a question. It has two main buttons, Yes and No. `QMessageBox::No|QMessageBox::Default` sets the No option as the default selection. `QMessageBox::No|QMessageBox::Escape` makes the escape key have the same effect as clicking on the No option.

Whatever option the user chooses will be stored as `int` in the status variable. It will then be compared to the `QMessageBox::Yes` constant. This way of asking the user a Yes or No question is not informative enough, especially when a destructive operation will ensue when the user clicks Yes. We shall use the alternative form as defined in `clearAllRecords()`:

```cpp
void MainWindow::clearAllRecords()
{
   */
   int status = QMessageBox::question( this, tr("Delete all Records ?"), tr("This operation will delete all saved records. " "<p>Do you want to remove all saved records ? "
 ), tr("Yes, Delete all records"), tr("No !"),  QString(), 1, 1);
   if (status == 0) {
       int rowCount = model->rowCount();
       model->removeRows(0, rowCount);
   }
}
```

As usual, the parent object is pointed to by `this`. The second parameter is the title of the dialog box and the string of the question follows. We shall make the first option verbose by passing Yes, Delete all records. The user, upon reading, will know what effect the clicking of the button will have. The No ! parameter will be displayed on the button that represents the other answer to the question. `QString()` is being passed so that we don't display the third button. When the first button is clicked, `0` will be returned to `status`. When the second button or option is clicked, `1` will be returned. By specifying `1`, we make the `"No !"` button the default button of the dialog box. We select `1` again, as the last parameter specifies that `"No !"` should be the button selected when the escape button is pressed.

If the user clicks on the Yes, Delete all records button, then status will store `0`. In the body of the `if` statement, we obtain the number of rows in our model object. A call to `removeRows` is made and we specify that all the entries from the first, represented by `0`, to the `rowCount`, should be removed. However, if the user clicks on the No ! button, the application will do nothing, as we don't specify that in the `if` statement.

The dialog window should appear as follows when the Clear All button is clicked:

![](img/aaf6dc67-d05e-484b-ba4c-4211f55bc803.png)

The `saveButtonClicked()` slot has also been modified to show a simple message to the user that the operation has been successful, as demonstrated in the following block of code:

```cpp
void MainWindow::saveButtonClicked()
{
   QStandardItem *name = new QStandardItem(nameLineEdit->text());
   QStandardItem *dob = new QStandardItem(dateOfBirthEdit->date().toString());
   QStandardItem *phoneNumber = new QStandardItem(phoneNumberLineEdit->text());
   model->appendRow({ name, dob, phoneNumber});
   clearFields();
   QMessageBox::information(this, tr("RMS System"), tr("Record saved successfully!"),
                            QMessageBox::Ok|QMessageBox::Default,
                            QMessageBox::NoButton, QMessageBox::NoButton);
}
```

The two last parameters are constants that prevent buttons from showing in the message box.

To allow the application to remove certain rows from the table, the `deleteSaveRecords()` method is used to raise an input-based dialog box that receives the `rowId` of the row we want to remove through the model:

```cpp
void MainWindow::deleteSavedRecord()
{
   bool ok;
   int rowId = QInputDialog::getInt(this, tr("Select Row to delete"), tr("Please enter Row ID of record (Eg. 1)"),
  1, 1, model->rowCount(), 1, &ok );
   if (ok)
   {
       model->removeRow(rowId-1);
   }
}
```

The `this` keyword refers to the parent object. The second parameter to the call of the static method `QInputDialog::getInt()` is used as the title of the dialog window. The request is captured in the second parameter. The third parameter here is used to specify the default number of the input field. `1`, and `model->rowCount()`, are the minimum and maximum values that should be accepted.

The last but one parameter, `1`, is the incremental step between the minimum and maximum value. `True` or `False` will be stored in `&ok`. When the user clicks OK, `True` will be stored in `&ok` and, based on that, the `if` statement will call the `removeRow` on the model object. Whatever value that the user inputs will be passed to `rowId`. We pass `rowId-1` to get the actual index of the row in the model.

The connection to this slot is made by executing the following command:

```cpp
connect(deleteOneEntryToolBarAction, SIGNAL(triggered()), this, 
SLOT(deleteSavedRecord()));
```

`deleteOneEntryToolBarAction` is the last but one action on the toolbar.

The following screenshot is what will appear when the user clicks on this action:

![](img/f66acf9a-878e-4a38-9202-a1832c602367.png)

The method that sets up the toolbar is given as follows:

```cpp
void MainWindow::createToolBar() {
   // Setup Tool bar menu
   toolbar = addToolBar("main toolbar");
   // toolbar->setMovable( false );
   newToolBarAction = toolbar->addAction(QIcon(newIcon), "New File");
   openToolBarAction = toolbar->addAction(QIcon(openIcon), "Open File");
   toolbar->addSeparator();
   clearToolBarAction = toolbar->addAction(QIcon(clearIcon), "Clear All");
   deleteOneEntryToolBarAction = toolbar->addAction(QIcon(deleteIcon), "Delete a record");
   closeToolBarAction = toolbar->addAction(QIcon(closeIcon), "Quit Application");
}
```

All the other methods are borrowed from the previous section and can be obtained from the source code attached to this book.

To recap, this is what you should see after compiling and running the project:

![](img/0a4b9f91-2d37-4eb4-91d4-818f48da3985.png)

Remember that the reason we already have an entry in the model object is because we created such an entry within the `setupCoreWidgets()` method.

Fill in the name, date of birth, and phone number fields and click on Save. This will add an extra line to the table in the window. A dialog message will tell you if the operation was successful.

To delete a row within the table, select the desired row and click on the recycle bin icon, and confirm whether you really want to delete the entry.

# Summary

In this chapter, we have seen how to create menus, toolbars, and how to use dialog boxes to receive further input and display information to the user. 

In [Chapter 5](48ced144-5e85-4e89-9ddf-80e876e83b0f.xhtml), *Managing Events, Custom Signals, and Slots*, we will explore the use of events and more on signals and slots.