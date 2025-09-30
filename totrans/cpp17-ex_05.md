# Qt Graphical Applications

In [Chapter 4](34ba2c98-98be-433d-b868-9aa7745dd6a6.xhtml), *Library Management System with Pointers*, we developed abstract datatypes and a library management system. However, those applications were text-based. In this chapter, we will look into three graphical applications that we will develop with the Qt graphical library:

*   **Clock**: We will develop an analog clock with hour, minute, and second hands, with lines to mark hours, minutes, and seconds
*   **The drawing program**: A program that draws lines, rectangles, and ellipses in different colors
*   **The editor**: A program where the user can input and edit text

We will also learn about the Qt library:

*   Windows and widgets
*   Menus and toolbars
*   Drawing figures and writing text in the window
*   How to catch mouse and keyboard events

# Creating the clock application

In this chapter and the next chapter, we will work with Qt, which is an object-oriented class library for graphical applications. We will also work with Qt Creator, instead of Visual Studio, which is an integrated development environment.

# Setting up the environment

When creating a new graphical project in Qt Creator, we select New File or Project in the File menu, which makes the New File or Project dialog window become visible. We select Qt Widgets Application, and click the Choose button.

Then the Introduction and Project Location dialog becomes visible. We name the project `Clock`, place it in an appropriate location, and click the Next button. In the KitSelection dialog, we select the latest version of the Qt library, and click Next. In the Class Information dialog, we name the base class of the application `clock`. Normally, the window of a graphical application inherits a `window` class. In this case, however, we are dealing with a relatively simple application. Therefore, we inherit the Qt class `QWidget`, even though a widget often refers to a smaller graphical object that is often embedded in the window. In Qt Creator, it is possible to add forms. However, we do not use that feature in this chapter. Therefore, we uncheck the Generate form option.

All class names in Qt start with the letter `Q`.

Finally, in the Project Management dialog, we simply accept the default values and click Finish to generate the project, with the files `Clock.h` and `Clock.cpp`.

# The Clock class

The project is made up by the files `Clock.h`, `Clock.cpp`, and `Main.cpp`. The class definition looks a little bit different compared to the classes of the previous chapters. We enclose the class definition with *include guards*. That is, we must enclose the class definition with the preprocessor directive `ifndef`, `define`, and `endif`. The preprocessor performs text substitutions.

The `ifndef` and `endif` directives work as the `if` statement in C++. If the condition is not true, the code between the directives is omitted. In this case, the code is included only if the `CLOCK_H` macro has not previously been defined. If the code is included, the macro becomes defined at the next line with the `define` directive. In this way, the class definition is included in the project only once. Moreover, we also include the system header files `QWidget` and `QTimer` in the `Clock.h` header file rather than the `Clock.cpp` definition file.

**Clock.h:**

```cpp
    #ifndef CLOCK_H 
    #define CLOCK_H 

    #include <QWidget> 
    #include <QTimer> 
```

Since `Clock` is a subclass of the Qt `QWidget` class, the `Q_OBJECT` macro must be included, which includes certain code from the Qt library. We need it to use the `SIGNAL` and `SLOT` macros shown here:

```cpp
    class Clock : public QWidget { 
      Q_OBJECT 
```

The constructor takes a pointer to its parent widget, for which the default is `nullptr`:

```cpp
    public: 
      Clock(QWidget* parentWidgetPtr = nullptr); 
```

The `paintEvent` method is called by the framework every time the window needs to be repainted. It takes a pointer to a `QPaintEvent` object as parameter, which can be used to determine in which way the repainting shall be performed:

```cpp
    void paintEvent(QPaintEvent *eventPtr); 
```

`QTimer` is a Qt system class that handles a timer. We will use that to move the hands of the clock:

```cpp
      private: 
        QTimer m_timer; 
    }; 

    #endif // CLOCK_H 
```

The definition file is mainly made up of the `paintEvent` method, which handles the painting of the clock.

**Clock.cpp:**  

```cpp
    #include <QtWidgets> 
    #include "Clock.h"
```

In the constructor, we call the base class `QWidget` with the `parentWidgetPtr` parameter (which may be `nullptr`):

```cpp
    Clock::Clock(QWidget* parentWidgetPtr /* = nullptr */) 
     :QWidget(parentWidgetPtr) { 
```

We set the title of the window to `Clock`. In Qt, we always use the `tr` function for literal text, which in turn calls the Qt method `translate` in the Qt `QCoreApplication` class that makes sure the text is translated into a form suitable to be displayed. We also resize the size of the window to 1000 x 500 pixels, which is appropriate for most screens:

```cpp
    setWindowTitle(tr("Clock")); 
    resize(1000, 500); 
```

We need a way to connect the timer with the clock widget: when the timer has finished its countdown, the clock shall be updated. For that purpose, Qt provides us with the Signal and Slot system. When the timer reaches its countdown, it calls its method `timeout`. We use the `connect` method together with the `SIGNAL` and `SLOT` macros to connect the call to `timeout` with the call to the `update` method in the Qt `QWidget` class, which updates the drawing of the clock. The `SIGNAL` macro registers that the call to timeout shall raise a signal, the `SLOT` macro registers that the update method shall be called when the signal is raised, and the `connect` method connects the signal with the slot. We have set up a connection between the timer's timeout and the update of the clock:

```cpp
      m_timer.setParent(this); 
      connect(&m_timer, SIGNAL(timeout()), this, SLOT(update())); 
      m_timer.start(100); 
    }
```

The `paintEvent` method is called every time the window needs to be repainted. It may be due to some external cause, such as the user resizes the window. It may also be due to a call to the `update` method of the `QMainWindow` class, which in turn eventually calls `paintEvent`.

In this case, we do not need any information about the event, so we enclose the `eventPtr` parameter in comments. The `width` and `height` methods give the width and height of the paintable part of the window, in pixels. We call the `qMin` method to decide the minimum side of the window, and the `currentTime` method of the `QTime` class to find the current time for the clock:

```cpp
    void Clock::paintEvent(QPaintEvent* /* eventPtr */) { 
      int side = qMin(width(), height()); 
      QTime time = QTime::currentTime();
```

The `QPainter` class can be viewed as a painting canvas. We start by initializing it to appropriate aliasing. We then call the `translate` and `scale` methods to transform the physical size in pixels to the logical size of `200` * `200` units:

```cpp
      QPainter painter(this); 
      painter.setRenderHint(QPainter::Antialiasing); 
      painter.setRenderHint(QPainter::TextAntialiasing); 
      painter.translate(width() / 2, height() / 2); 
      painter.scale(side / 200.0, side / 200.0); 
```

We paint 60 lines for the minutes. Every fifth line shall be a little bit longer to mark the current hours. For each minute, we draw a line, and then we call the Qt `rotate` method, which rotates the drawing by `6` degrees. In this way, we rotate the drawing by 6 degrees 60 times, which sums up to 360 degrees, a whole lap:

```cpp
      for (int second = 0; second <= 60; ++second) { 
        if ((second % 5) == 0) { 
          painter.drawLine(QPoint(0, 81), QPoint(0, 98)); 
        } 
        else { 
          painter.drawLine(QPoint(0, 90), QPoint(0, 98)); 
        } 
```

A complete leap is 360 degrees. For each line we rotate by `6` degrees, since 360 divided by 60 is `6` degrees. When we are finished with the rotations, the drawing is reset to its original settings:

```cpp
        painter.rotate(6); 
      }  
```

We obtain the current hour, minute, second, and millisecond from the `QTime` object:

```cpp
      double hours = time.hour(), minutes = time.minute(), 
             seconds = time.second(), milliseconds = time.msec(); 
```

We set the pen color to black and the background color to gray:

```cpp
      painter.setPen(Qt::black); 
      painter.setBrush(Qt::gray);
```

We define the endpoints of the hour hand. The hour hand is a little bit thicker and shorter than the minute and second hands. We define three points that constitute the endpoint of the hour hand. The base of the hour hand is 16 units long and located 8 units from the origin. Therefore, we set the x coordinate of the base points to `8` and `-8`, and the y coordinate to `8`. Finally, we define the length of the hour hand to `60` units. The value is negative in order to correspond with current rotation:

```cpp
      { static const QPoint hourHand[3] = 
          {QPoint(8, 8), QPoint(-8, 8), QPoint(0, -60)};
```

The `save` method saves the current settings of the `QPointer` object. The settings are later restored by the `restore` method:

```cpp
        painter.save(); 
```

We find out the exact angle of the current hour hand by calculating the hours, minutes, seconds, and milliseconds. We then rotate to set the hour hand. Each hour corresponds to 30 degrees, since we have 12 hours, and 360 degrees divided by 12 is 30 degrees:

```cpp
        double hour = hours + (minutes / 60.0) + (seconds / 3600.0) + 
                      (milliseconds / 3600000.0); 
        painter.rotate(30.0 * hour); 
```

We call the `drawConvexPloygon` method with the three points of the hour hand:

```cpp
        painter.drawConvexPolygon(hourHand, 3); 
        painter.restore(); 
      } 
```

We draw the minute hand in the same way. It is a little bit thinner and longer than the hour hand. Another difference is that while we had 12 hours, we now have 60 minutes. This gives that each minute corresponds to `6` degrees, since 360 degrees divided by 60 is 6 degrees:

```cpp
      { static const QPoint minuteHand[3] = 
          {QPoint(6, 8), QPoint(-6, 8), QPoint(0, -70)}; 
        painter.save(); 
```

When calculating the current minute angle, we use the minutes, seconds, and milliseconds:

```cpp
        double minute = minutes + (seconds / 60.0) + 
                        (milliseconds / 60000.0); 
        painter.rotate(6.0 * minute); 
        painter.drawConvexPolygon(minuteHand, 3); 
        painter.restore(); 
      }
```

The drawing of the second hand is almost identical to the drawing of the previous minute hand. The only difference is that we only use seconds and milliseconds to calculate the second angle:

```cpp
      { static const QPoint secondHand[3] = 
          {QPoint(4, 8), QPoint(-4, 8), QPoint(0, -80)}; 

        painter.save(); 
        double second = seconds + (milliseconds / 1000); 
        painter.rotate(6.0 * second); 
        painter.drawConvexPolygon(secondHand, 3); 
        painter.restore(); 
      } 
    } 
```

# The main function

In the `main` function, we initialize and start the Qt application. The `main` function can take the parameters `argc` and `argv`. It holds the command-line arguments of the applications; `argc` holds the number of arguments and the `argv` array holds the arguments themselves. The first entry of `argv` always holds the path to the execution file, and the last entry is always `nullptr`. The `QApplication` class takes `argc` and `argv` and initializes the Qt application. We create an object of our `Clock` class, and call `show` to make it visible. Finally, we call `exec` of the `QApplication` object.

**Main.cpp:**  

```cpp
    #include <QApplication> 
    #include "Clock.h" 

    int main(int argc, char *argv[]) { 
      QApplication application(argc, argv); 
      Clock Clock; 
      Clock.show(); 
      return application.exec(); 
    }
```

To execute the application, we select the Run option on the project:

![](img/4e8a7f8f-c0bc-4cfb-b4b7-8aa2fa5e81f1.png)

The execution will continue until the user closes the `Clock` window by pressing the close button in the top-right corner:

![](img/f489094c-a214-400d-bdb7-32e03e57e385.png)

# Setting up reusable classes for windows and widgets

In graphical applications, there are windows and widgets. A window is often a complete window with a frame holding title, menu bar, and buttons for closing and resizing the window. A widget is often a smaller graphical object, often embedded in a window. In the *Clock* project, we used only a `widget` class that inherits the `QWidget` class. However, in this section we will leave the *Clock* project and look into more advanced applications with both a window and a widget. The window holds the frame with the menu bar and toolbar, while the widget is located in the window and takes care of the graphical content.

In the following sections of this chapter, we will look into a drawing program and an editor. Those applications are typical document applications, where we open and save documents, as well as also cut, copy, paste, and delete elements of the document. In order to add menus and toolbars to the window, we need to inherit the two Qt classes, `QMainWindow` and `QWidget`. We need `QMainWindow` to add menus and toolbars to the window frame, and `QWidget` to draw images in the window's area.

In order to reuse the document code in the applications introduced in the remaining part of this chapter and in the next chapter, in this section, we define the classes `MainWindow` and `DocumentWidget`. Those classes will then be used by the drawing program and the editor later in the following sections of this chapter. `MainWindow` sets up a window with the `File` and `Edit` menus and toolbars, while `DocumentWidget` provides a framework that sets up skeleton code for the `New`, `Open`, `Save`, `SaveAs`, `Cut`, `Copy`, `Paste`, `Delete`, and `Exit` items. In this section, we will not create a new Qt project, we will just write the classes `MainWindow` and `DocumentWidget`, which are used as base classes in the drawing program and editor later in this chapter, and the `LISTENER` macro, which is used to set up menu and toolbar items.

# Adding a listener

A listener is a method that is called when the user selects a menu item or a toolbar item. The `Listener` macro adds a listener to the class.

**Listener.h:**

```cpp
    #ifndef LISTENER_H 
    #define LISTENER_H 

    #include <QObject>
```

Due to Qt rules regarding menus and toolbars, the listener called by the Qt Framework in response to a user action must be a function rather than a method.

A method belongs to a class, while a function is free-standing.

The `DefineListener` macro defines both a friendly function and a method. The Qt Framework calls the function, which in turns calls the method:

```cpp
    #define DEFINE_LISTENER(BaseClass, Listener)           
      friend bool Listener(QObject* baseObjectPtr) {       
         return ((BaseClass*) baseObjectPtr)->Listener();  
      }                                                    
      bool Listener()                                      
```

The `Listener` macro is defined as a pointer to the method:

```cpp
    #define LISTENER(Listener) (&::Listener) 
```

The listener method takes an `QObject` pointer as a parameter and returns a Boolean value:

```cpp
    typedef bool (*Listener)(QObject*); 
    #endif // LISTENER_H 
```

# The base window class

The `MainWindow` class sets up a document window with the `File` and `Edit` menus and toolbars. It also provides the `addAction` method, which is intended for subclasses to add application-specific menus and toolbars.

**MainWindow.h:**

```cpp
    #ifndef MAINWINDOW_H 
    #define MAINWINDOW_H 

    #include <QMainWindow> 
    #include <QActionGroup> 
    #include <QPair> 
    #include <QMap> 

    #include "Listener.h" 
    #include "DocumentWidget.h" 

    class MainWindow : public QMainWindow { 
      Q_OBJECT 

      public: 
        MainWindow(QWidget* parentWidgetPtr = nullptr); 
        ~MainWindow(); 

      protected: 
        void addFileMenu(); 
        void addEditMenu(); 
```

The `addAction` method adds a menu item, with a potential accelerator key, toolbar icon, and listeners to mark the item with a checkbox or a radio button:

```cpp
      protected: 
        void addAction(QMenu* menuPtr, QString text, 
                       const char* onSelectPtr, 
                       QKeySequence acceleratorKey = 0, 
                       QString iconName = QString(), 
                       QToolBar* toolBarPtr = nullptr, 
                       QString statusTip = QString(), 
                       Listener enableListener = nullptr, 
                       Listener checkListener = nullptr, 
                       QActionGroup* groupPtr = nullptr); 
```

We use the `DefineListener` macro to add a listener to decide whether a menu item shall be enabled. The listeners return `true` if the item shall be enabled. `DocumentWidget` is a sub class of the Qt class `QWidget`, which we will define in the next section. With the `DEFINE_LISTENER` macro, we add the `isSaveEnabled`, `isCutEnabled`, `isCopyEnabled`, `isPasteEnabled`, and `isDeleteEnabled` methods to the `MainWindow` class. They will be called when the user selects a menu item:

```cpp
        DEFINE_LISTENER(DocumentWidget, isSaveEnabled); 
        DEFINE_LISTENER(DocumentWidget, isCutEnabled); 
        DEFINE_LISTENER(DocumentWidget, isCopyEnabled); 
        DEFINE_LISTENER(DocumentWidget, isPasteEnabled); 
        DEFINE_LISTENER(DocumentWidget, isDeleteEnabled); 
```

The `onMenuShow` method is called before a menu becomes visible; it calls the listener of the items of the menu to decide whether they shall be disabled or annotated by a checkbox or a radio button. It is also called by the framework in order to disable toolbar icons:

```cpp
      public slots: 
        void onMenuShow();
```

The `m_enableMap` and `m_checkMap` fields hold maps of listeners for the menu items. The preceding `onMenuShow` method uses them to decide whether to disable the item, or annotate it with a checkbox or a radio button:

```cpp
      private: 
        QMap<QAction*,QPair<QObject*,Listener>> m_enableMap, 
                                                m_checkMap; 
    }; 

    #endif // MAINWINDOW_H 
```

**MainWindow.cpp:**  

```cpp
    #include "MainWindow.h" 
    #include <QtWidgets> 
```

The constructor calls the constructor of the Qt `QMainWindow` class, with the parent widget pointer as its parameter:

```cpp
    MainWindow::MainWindow(QWidget* parentWidgetPtr /*= nullptr*/) 
     :QMainWindow(parentWidgetPtr) { 
    } 
```

When a menu item is added, it is connected to an action. The destructor deallocates all actions of the menu bar:

```cpp
    MainWindow::~MainWindow() { 
      for (QAction* actionPtr : menuBar()->actions()) { 
        delete actionPtr; 
      } 
    } 
```

The `addFileMenu` method adds the standard `File` menu to the menu bar; `menubar` is a Qt method that returns a pointer to the menu bar of the window:

```cpp
    void MainWindow::addFileMenu() { 
      QMenu* fileMenuPtr = menuBar()->addMenu(tr("&File"));
```

Similar to the `connect` method which connects the menu item with the `onMenuShow` method in the following code snippet. The Qt macros `SIGNAL` and `SLOT` ensure that `onMenuShow` is called before the menu becomes visible. The `onMenuShow` method sets the enable, checkbox, and radio bottom status for each item of the menu before the menu becomes visible. It also sets the enable status of toolbars images. The `aboutToShow` method is called before each menu becomes visible in order to enable or disable the items, and to possibly mark them with check boxes or radio buttons:

```cpp
      connect(fileMenuPtr, SIGNAL(aboutToShow()), this, 
              SLOT(onMenuShow())); 
```

The Qt `addToolBar` method adds a toolbar to the window's frame. When we call `addAction` here, the menu item will be added to the menu and, if present, to the toolbar:

```cpp
      QToolBar *fileToolBarPtr = addToolBar(tr("File")); 
```

The `addAction` method adds the `New`, `Open`, `Save`, `SaveAs`, and `Exit` menu items. It takes the following parameters:

*   A pointer to the menu the item shall belong to.
*   The item text. The ampersand (`&`) before the text (`&New`) indicates that the next letter (`N`) will be underlined, and that the user can select that item by pressing *Alt*-*N*.
*   Accelerator information. `QKeySequence` is a Qt enumeration holding accelerator key combinations. `QKeySequence::New` indicates that the user can select the item by pressing *Ctrl*-*N*. The text `Ctrl+N` will also be added to the item text.
*   The name of an icon file (`new`). The icon of the file is displayed both to the left of the item text and on the toolbar. The icon file itself is added to the project in Qt Creator.
*   A pointer to the toolbar, `nullptr` if the item is not connected to a toolbar.
*   The text displayed when the user hovers with the mouse over the toolbar item. Ignored if the item is not connected to a toolbar.
*   Listeners (default `nullptr`) that are called before the menu and toolbar become visible, and deciding whether the item is enabled or marked with a checkbox or a radio button:

```cpp
  addAction(fileMenuPtr, tr("&New"), SLOT(onNew()), 
            QKeySequence::New, tr("new"), fileToolBarPtr, 
            tr("Create a new file")); 

  addAction(fileMenuPtr, tr("&Open"), SLOT(onOpen()), 
            QKeySequence::Open, tr("open"), fileToolBarPtr, 
            tr("Open an existing file")); 
```

When there are no changes in the document since it was last saved, the document does not need to be saved and the `Save` item shall be disabled. Therefore, we add an extra parameter, indicating that the `isSaveEnabled` method shall be called to enable or disable the menu and toolbar item:

```cpp
      addAction(fileMenuPtr, tr("&Save"), SLOT(onSave()), 
                QKeySequence::Save, tr("save"), fileToolBarPtr, 
                tr("Save the document to disk"), 
                LISTENER(isSaveEnabled)); 
```

The `SaveAs` menu item has no key sequence. Moreover, it does not have a toolbar entry. Therefore, the name of the icon file and the toolbar text are default `QString` objects and the toolbar pointer is `nullptr`:

```cpp
      addAction(fileMenuPtr, tr("Save &As"), SLOT(onSaveAs()), 
                0, QString(), nullptr, QString(), 
                LISTENER(isSaveEnabled)); 
```

The `addSeparator` method adds a horizontal line between two items:

```cpp
      fileMenuPtr->addSeparator(); 
      addAction(fileMenuPtr, tr("E&xit"), 
                SLOT(onExit()), QKeySequence::Quit); 
    } 
```

The `addEditMenu` method adds the `Edit` menu to the window's menu bar in the same way as the preceding `File` menu:

```cpp
    void MainWindow::addEditMenu() { 
      QMenu* editMenuPtr = menuBar()->addMenu(tr("&Edit")); 
      QToolBar* editToolBarPtr = addToolBar(tr("Edit")); 
      connect(editMenuPtr, SIGNAL(aboutToShow()), 
              this, SLOT(onMenuShow())); 

      addAction(editMenuPtr, tr("&Cut"), SLOT(onCut()), 
                QKeySequence::Cut, tr("cut"), editToolBarPtr, 
          tr("Cut the current selection's contents to the clipboard"), 
                LISTENER(isCutEnabled)); 

      addAction(editMenuPtr, tr("&Copy"), SLOT(onCopy()), 
                QKeySequence::Copy, tr("copy"), editToolBarPtr, 
         tr("Copy the current selection's contents to the clipboard"), 
                LISTENER(isCopyEnabled)); 

      addAction(editMenuPtr, tr("&Paste"), SLOT(onPaste()), 
                QKeySequence::Paste, tr("paste"), editToolBarPtr, 
        tr("Paste the current selection's contents to the clipboard"), 
                LISTENER(isPasteEnabled)); 

      editMenuPtr->addSeparator(); 
      addAction(editMenuPtr, tr("&Delete"), SLOT(onDelete()), 
                QKeySequence::Delete, tr("delete"), editToolBarPtr, 
                tr("Delete the current selection"), 
                LISTENER(isDeleteEnabled)); 
    } 
```

The `addAction` method adds a menu item to the menu bar and a toolbar icon to the toolbar. It also connects the item with the `onSelectPtr` method that is called when the user selects the item, and methods that enable the item and annotate it with a checkbox or radio button. An accelerator is added to the action, unless it is zero. The `groupPtr` parameter defines whether the item is part of a group. If `checkListener` is not `nullptr`, the item is annotated with a checkbox if `groupPtr` is `nullptr`, and with a radio button if it is not. In the case of radio buttons, only one radio button in the group will be marked at the same time:

```cpp
    void MainWindow::addAction(QMenu* menuPtr, QString itemText, 
                               const char* onSelectPtr, 
                               QKeySequence acceleratorKey /* = 0 */, 
                               QString iconName /*= QString()*/, 
                               QToolBar* toolBarPtr /*= nullptr*/, 
                               QString statusTip /*= QString()*/, 
                               Listener enableListener /*= nullptr*/, 
                               Listener checkListener /*= nullptr*/, 
                               QActionGroup* groupPtr /*= nullptr*/) { 
      QAction* actionPtr; 
```

If `iconName` is not empty, we load the icon from the file in the project resource and then create a new `QAction` object with the icon:

```cpp
      if (!iconName.isEmpty()) { 
        const QIcon icon = QIcon::fromTheme("document-" + iconName, 
                           QIcon(":/images/" + iconName + ".png")); 
        actionPtr = new QAction(icon, itemText, this); 
      } 
```

If `iconName` is empty, we create a new `QAction` object without the icon:

```cpp
      else { 
        actionPtr = new QAction(itemText, this); 
      }
```

We connect the menu item to the selection method. When the user selects the item, or clicks on the toolbar icon, `onSelectPtr` is called:

```cpp
      connect(actionPtr, SIGNAL(triggered()), 
              centralWidget(), onSelectPtr); 
```

If the accelerator key is not zero, we add it to the action pointer:

```cpp
      if (acceleratorKey != 0) { 
        actionPtr->setShortcut(acceleratorKey); 
      } 
```

Finally, we add the action pointer to the menu pointer in order for it to process the user's item selection:

```cpp
      menuPtr->addAction(actionPtr); 
```

If `toolBarPtr` is not `nullptr`, we add the action to the toolbar of the window:

```cpp
      if (toolBarPtr != nullptr) { 
        toolBarPtr->addAction(actionPtr); 
      } 
```

If the status tip is not empty, we add it to the tooltip and status tip of the toolbar:

```cpp
      if (!statusTip.isEmpty()) { 
          actionPtr->setToolTip(statusTip); 
          actionPtr->setStatusTip(statusTip); 
      } 
```

If the enable listener is not null, we add to `m_enableMap` a pair made up of a pointer to the central widget of the window and the listener. We also call the listener to initialize the enable status of the menu item and toolbar icon:

```cpp
      if (enableListener != nullptr) { 
        QWidget* widgetPtr = centralWidget(); 
        m_enableMap[actionPtr] = 
          QPair<QObject*,Listener>(widgetPtr, enableListener); 
        actionPtr->setEnabled(enableListener(widgetPtr)); 
      } 
```

In the same way, if the check listener is not null, we add a pointer to the central widget of the window and the listener to `m_checkMap`. Both `m_enableMap` and `m_checkMap` are used by `onMenuShow`, as follows. We also call the listener to initialize the check status of the menu item (toolbar icons are not checked):

```cpp
      if (checkListener != nullptr) { 
        actionPtr->setCheckable(true); 
        QWidget* widgetPtr = centralWidget(); 
        m_checkMap[actionPtr] = 
          QPair<QObject*,Listener>(widgetPtr, checkListener); 
        actionPtr->setChecked(checkListener(widgetPtr)); 
      } 
```

Finally, if the group pointer is not null, we add the action to it. In that way, the menu item will be annotated by a radio button rather than a checkbox. The framework does also keep track of the groups and makes sure only one of the radio buttons of each group is marked at the same time:

```cpp
      if (groupPtr != nullptr) { 
        groupPtr->addAction(actionPtr); 
      } 
    } 
```

The `onMenuShow` method is called before a menu or toolbar icon becomes visible. It makes sure each item is enabled or disabled, and that the items are annotated with checkboxes or radio buttons.

We start by iterating through the enable map. For each entry in the map, we look up the widget and the enable function. We call the function, which returns `true` or `false`, and use the result to enable or disable the item by calling `setEnabled` on the action object pointer:

```cpp
    void MainWindow::onMenuShow() { 
      for (QMap<QAction*,QPair<QObject*,Listener>>::iterator i = 
           m_enableMap.begin(); i != m_enableMap.end(); ++i) { 
        QAction* actionPtr = i.key(); 
        QPair<QObject*,Listener> pair = i.value(); 
        QObject* baseObjectPtr = pair.first; 
        Listener enableFunction = pair.second; 
        actionPtr->setEnabled(enableFunction(baseObjectPtr)); 
      } 
```

In the same way, we iterate through the check map. For each entry in the map, we look up the widget and the check function. We call the function and use the result to check the item by calling `setCheckable` and `setChecked` on the action object pointer. The Qt Framework makes sure the item is annotated by radio buttons if it belongs to a group, and a checkbox if it does not:

```cpp
      for (QMap<QAction*,QPair<QObject*,Listener>>::iterator i = 
           m_checkMap.begin(); i != m_checkMap.end(); ++i) { 
        QAction* actionPtr = i.key(); 
        QPair<QObject*,Listener> pair = i.value(); 
        QObject* baseObjectPtr = pair.first; 
        Listener checkFunction = pair.second; 
        actionPtr->setCheckable(true); 
        actionPtr->setChecked(checkFunction(baseObjectPtr)); 
      } 
    } 
```

# The base widget class

`DocumentWidget` is a skeleton framework for applications that handle documents. It handles the loading and saving of the document, and provides methods to be overridden by subclasses for the `Cut`, `Copy`, `Paste`, and `Delete` menu items.

While the preceding `MainWindow` class handles the window frame, with its menus and toolbars, the `DocumentWidget` class handles the drawing of the window's content. The idea is that the subclass of `MainWindow` creates an object of a subclass to `DocumentWidget` that it puts at the centrum of the window. See the constructors of `DrawingWindow` and `EditorWindow` in the following sections.

**DocumentWidget.h:**

```cpp
    #ifndef DOCUMENTWIDGET_H 
    #define DOCUMENTWIDGET_H 

    #include "Listener.h" 
    #include <QWidget> 
    #include <QtWidgets> 
    #include <FStream> 
    using namespace std; 

    class DocumentWidget : public QWidget { 
      Q_OBJECT 
```

The constructor takes the name of the application, to be displayed at the top banner of the window, the filename mask to be used when loading and storing documents with the standard file dialogs, and a pointer to a potential parent widget (normally the enclosing main window):

```cpp
      public: 
        DocumentWidget(const QString& name, const QString& fileMask, 
                       QWidget* parentWidgetPtr); 
        ~DocumentWidget();
```

The `setFilePath` method sets the path of the current document. The path is displayed at the top banner of the window and is given as a default path in the standard load and save dialogs:

```cpp
      protected: 
        void setFilePath(QString filePath); 
```

When a document has been changed, the modified flag (sometimes called the dirty flag) is set. This causes an asterisk (`*`) to appear next to the file path at the top banner of the window, and the `Save` and `SaveAs` menu items to be enabled:

```cpp
      public: 
        void setModifiedFlag(bool flag); 
```

The `setMainWindowTitle` method is an auxiliary method that puts together the title of the window. It is made up by the file path and a potential asterisk (`*`) to indicate whether the modified flag is set:

```cpp
      private: 
        void setMainWindowTitle(); 
```

The `closeEvent` method is overridden from `QWidget` and is called when the user closes the window. By setting fields of the `eventPtr` parameter, the closing can be prevented. For example, if the document has not been saved, the user can be asked if they want to save the document or cancel the closing of the window:

```cpp
      public: 
        virtual void closeEvent(QCloseEvent* eventPtr); 
```

The `isClearOk` method is an auxiliary method that displays a message box if the user tries to close the window or exit the application without saving the document:

```cpp
      private: 
        bool isClearOk(QString title); 
```

The following methods are called by the framework when the user selects a menu item or clicks a toolbar icon. In order for that to work, we mark the methods as slots, which is necessary for the `SLOT` macro in the `connect` call:

```cpp
      public slots: 
        virtual void onNew(); 
        virtual void onOpen(); 
        virtual bool onSave(); 
        virtual bool onSaveAs(); 
        virtual void onExit();
```

When a document has not been changed, it is not necessary to save it. In that case, the `Save` and `SaveAs` menu items and toolbars images shall be disabled. The `isSaveEnabled` method is called by `onMenuShow` before the `File` menu becomes visible. It returns true only when the document has been changed and needs to be saved:

```cpp
    virtual bool isSaveEnabled(); 
```

The `tryWriteFile` method is an auxiliary method that tries to write the file. If it fails, a message box displays an error message:

```cpp
    private: 
        bool tryWriteFile(QString filePath); 
```

The following methods are virtual methods intended to be overridden by subclasses. They are called when the user selects the `New`, `Save`, `SaveAs`, and `Open` menu items:

```cpp
      protected: 
        virtual void newDocument() = 0; 
        virtual bool writeFile(const QString& filePath) = 0; 
        virtual bool readFile(const QString& filePath) = 0; 
```

The following methods are called before the edit menu becomes visible, and they decide whether the `Cut`, `Copy`, `Paste`, and `Delete` items shall be enabled:

```cpp
      public: 
        virtual bool isCutEnabled(); 
        virtual bool isCopyEnabled(); 
        virtual bool isPasteEnabled(); 
        virtual bool isDeleteEnabled(); 
```

The following methods are called when the user selects the `Cut`, `Copy`, `Paste`, and `Delete` items or toolbar icons:

```cpp
      public slots: 
        virtual void onCut(); 
        virtual void onCopy(); 
        virtual void onPaste(); 
        virtual void onDelete(); 
```

The `m_applicationName` field holds the name of the application, not the document. In the next sections, the names will be *Drawing* and *Editor*. The `m_fileMask` field holds the mask that is used when loading and saving the document with the standard dialogs. For instance, let us say that we have documents with the ending `.abc`. Then the mask could be `Abc files (.abc)`. The `m_filePath` field holds the path of the current document. When the document is new and not yet saved, the field holds the empty string.

Finally, `m_modifiedFlag` is true when the document has been modified and needs to be saved before the application quits:

```cpp
      private: 
        QString m_applicationName, m_fileMask, m_filePath; 
        bool m_modifiedFlag = false; 
    }; 
```

Finally, there are some overloaded auxiliary operators. The addition and subtraction operators add and subtract a point with a size, and a rectangle with a size:

```cpp
    QPoint& operator+=(QPoint& point, const QSize& size); 
    QPoint& operator-=(QPoint& point, const QSize& size); 

    QRect& operator+=(QRect& rect, int size); 
    QRect& operator-=(QRect& rect, int size); 
```

The `writePoint` and `readPoint` methods write and read a point from an input stream:

```cpp
    void writePoint(ofstream& outStream, const QPoint& point); 
    void readPoint(ifstream& inStream, QPoint& point); 
```

The `writeColor` and `readColor` methods write and read a color from an input stream:

```cpp
    void writeColor(ofstream& outStream, const QColor& color); 
    void readColor(ifstream& inStream, QColor& color); 
```

The `makeRect` method creates a rectangle with `point` as its center and `size` as its size:

```cpp
    QRect makeRect(const QPoint& centerPoint, int halfSide); 
    #endif // DOCUMENTWIDGET_H 
```

**DocumentWidget.cpp:**

```cpp
    #include <QtWidgets> 
    #include <QMessageBox> 

    #include "MainWindow.h" 
    #include "DocumentWidget.h" 
```

The constructor sets the name of the application, the file mask for the save and load standard dialogs, and a pointer to the enclosing parent widget (usually the enclosing main window):

```cpp
    DocumentWidget::DocumentWidget(const QString& name, 
                    const QString& fileMask, QWidget* parentWidgetPtr) 
     :m_applicationName(name), 
      m_fileMask(fileMask), 
      QWidget(parentWidgetPtr) { 
        setMainWindowTitle(); 
      } 
```

The destructor does nothing, it is included for completeness only:

```cpp
    DocumentWidget::~DocumentWidget() { 
      // Empty. 
    } 
```

The `setFilePath` method calls `setMainWindowTitle` to update the text on the top banner of the window:

```cpp
    void DocumentWidget::setFilePath(QString filePath) { 
      m_filePath = filePath; 
      setMainWindowTitle(); 
    } 
```

The `setModifiedFlag` method also calls `setMainWindowTitle` to update the text on the top banner of the window. Moreover, it calls `onMenuShow` on the parent widget to update the icons of the toolbars:

```cpp
    void DocumentWidget::setModifiedFlag(bool modifiedFlag) { 
      m_modifiedFlag = modifiedFlag; 
      setMainWindowTitle(); 
      ((MainWindow*) parentWidget())->onMenuShow(); 
    } 
```

The title displayed at the top banner of the toolbar is the application name, the document file path (if not empty), and an asterisk if the document has been modified without being saved:

```cpp
    void DocumentWidget::setMainWindowTitle() { 
      QString title= m_applicationName + 
              (m_filePath.isEmpty() ? "" : (" [" + m_filePath + "]"))+ 
              (m_modifiedFlag ? " *" : ""); 
      this->parentWidget()->setWindowTitle(title); 
    } 
```

The `isClearOk` method displays a message box if the document has been modified without being saved. The user can select one of the following buttons:

*   Yes: The document is saved, and the application quits. However, if the saving fails, an error message is displayed and the application does not quit.
*   No: The application quits without saving the document.
*   Cancel: The closing of the application is cancelled. The document is not saved.

```cpp
    bool DocumentWidget::isClearOk(QString title) { 
      if (m_modifiedFlag) { 
        QMessageBox messageBox(QMessageBox::Warning, 
                               title, QString()); 
        messageBox.setText(tr("The document has been modified.")); 
        messageBox.setInformativeText( 
                   tr("Do you want to save your changes?")); 
        messageBox.setStandardButtons(QMessageBox::Yes | 
                              QMessageBox::No | QMessageBox::Cancel); 
        messageBox.setDefaultButton(QMessageBox::Yes); 

        switch (messageBox.exec()) { 
          case QMessageBox::Yes: 
            return onSave(); 

          case QMessageBox::No: 
            return true; 

          case QMessageBox::Cancel: 
            return false; 
        } 
      } 

      return true; 
    } 
```

If the document is cleared, `newDocument` is called, which is intended to be overridden by a subclass to perform application-specific initialization. Moreover, the modified flag and the file path are cleared. Finally, the Qt `update` method is called to force a repainting of the window's content:

```cpp
    void DocumentWidget::onNew() { 
      if (isClearOk(tr("New File"))) { 
        newDocument(); 
        setModifiedFlag(false); 
        setFilePath(QString()); 
        update(); 
      } 
    } 
```

If the document is cleared, `onOpen` uses the standard open dialog to obtain the file path of the document:

```cpp
    void DocumentWidget::onOpen() { 
      if (isClearOk(tr("Open File"))) { 
        QString file = 
          QFileDialog::getOpenFileName(this, tr("Open File"), 
                       tr("C:\Users\Stefan\Documents\" 
                          "A A_Cpp_By_Example\Draw"), 
                  m_fileMask + tr(";;Text files (*.txt)")); 
```

If the file was successfully read, the modified flag is cleared, the file path is set, and `update` is called to force a repainting of the window:

```cpp
        if (!file.isEmpty()) { 
          if (readFile(file)) { 
            setModifiedFlag(false); 
            setFilePath(file); 
            update(); 
          } 
```

However, if the reading was not successful, a message box with an error message is displayed:

```cpp
          else { 
            QMessageBox messageBox; 
            messageBox.setIcon(QMessageBox::Critical); 
            messageBox.setText(tr("Read File")); 
            messageBox.setInformativeText(tr("Could not read "") + 
                                          m_filePath  + tr(""")); 
            messageBox.setStandardButtons(QMessageBox::Ok); 
            messageBox.setDefaultButton(QMessageBox::Ok); 
            messageBox.exec(); 
          } 
        } 
      } 
    } 
```

The `ifSaveEnabled` method simply returns the value of `m_modifiedFlag`. However, we need the method for the listener to work:

```cpp
    bool DocumentWidget::isSaveEnabled() { 
      return m_modifiedFlag; 
    } 
```

The `onSave` method is called when the user selects the `Save` or `SaveAs` menu item or toolbar icon. If the document has already been given a name, we simply try to write the file. However, if it has not yet been given a name we call `OnSaveAs`, which displays the standard Save dialog for the user:

```cpp
    bool DocumentWidget::onSave() { 
      if (!m_filePath.isEmpty()) { 
        return tryWriteFile(m_filePath); 
      } 
      else { 
        return onSaveAs(); 
      } 
    } 
```

The `onSaveAs` method is called when the user selects the `SaveAs` menu item (there is no toolbar icon for this item). It opens the standard open dialog and tries to write the file. If the writing was not successful, `false` is returned. The reason for this is that `isClearOk` closes the window only if the writing was successful:

```cpp
    bool DocumentWidget::onSaveAs() { 
      QString filePath = 
              QFileDialog::getSaveFileName(this, tr("Save File"), 
                   tr("C:\Users\Stefan\Documents\" 
                      "A A_Cpp_By_Example\Draw"), 
                m_fileMask + tr(";;Text files (*.txt)")); 

      if (!filePath.isEmpty()) { 
        return tryWriteFile(filePath); 
      } 
      else { 
        return false; 
      } 
    } 
```

The `tryWriteFile` method tries to write the file by calling write, which is intended to be overridden by a subclass. If it succeeded, the modified flag and the file path are set. If the file was not successfully written, a message box with an error message is displayed:

```cpp
    bool DocumentWidget::tryWriteFile(QString filePath) { 
      if (writeFile(filePath)) { 
        setModifiedFlag(false); 
        setFilePath(filePath); 
        return true; 
      } 
      else { 
        QMessageBox messageBox; 
        messageBox.setIcon(QMessageBox::Critical); 
        messageBox.setText(tr("Write File")); 
        messageBox.setInformativeText(tr("Could not write "") + 
                                      filePath  + tr(""")); 
        messageBox.setStandardButtons(QMessageBox::Ok); 
        messageBox.setDefaultButton(QMessageBox::Ok); 
        messageBox.exec(); 
        return false; 
      } 
    } 
```

The `onExit` method is called when the user selects the `Exit` menu item. It checks whether it is clear to close the window, and exits the application if it is:

```cpp
    void DocumentWidget::onExit() { 
      if (isClearOk(tr("Exit"))) { 
        qApp->exit(0); 
      } 
    } 
```

The default behavior of `isCutEnabled` and `isDeleteEnabled` is to call `isCopyEnabled`, since they often are enabled on the same conditions:

```cpp
    bool DocumentWidget::isCutEnabled() { 
      return isCopyEnabled(); 
    } 

    bool DocumentWidget::isDeleteEnabled() { 
      return isCopyEnabled(); 
    } 
```

The default behavior of `onCut` is to simply call `onCopy` and `onDelete`:

```cpp
    void DocumentWidget::onCut() { 
      onCopy(); 
      onDelete(); 
    } 
```

The default behavior of the rest of the cut-and-copy methods is to return `false` and do nothing, which will leave the menu items disabled unless the subclass overrides the methods:

```cpp
    bool DocumentWidget::isCopyEnabled() { 
      return false; 
    } 

    void DocumentWidget::onCopy() { 
      // Empty. 
    } 

    bool DocumentWidget::isPasteEnabled() { 
      return false; 
    } 

    void DocumentWidget::onPaste() { 
      // Empty. 
    } 

    void DocumentWidget::onDelete() { 
      // Empty. 
} 
```

Finally, `closeEvent` is called when the user tries to close the window. If the window is ready to be cleared, `accept` is called on `eventPtr`, which causes the window to be closed, and `exit` is called on the global `qApp` object, which causes the application to quit:

```cpp
    void DocumentWidget::closeEvent(QCloseEvent* eventPtr) { 
      if (isClearOk(tr("Close Window"))) { 
        eventPtr->accept(); 
        qApp->exit(0); 
      } 
```

However, if the window is not ready to be cleared, `ignore` is called on `eventPtr`, which causes the window to remain open (and the application to continue):

```cpp
      else { 
        eventPtr->ignore(); 
      } 
    } 
```

Moreover, there are also the set of auxiliary functions for handling points, sizes, rectangles, and color. The following operators add and subtract a point with a size, and return the resulting point:

```cpp
    QPoint& operator+=(QPoint& point, const QSize& size) { 
      point.setX(point.x() + size.width()); 
      point.setY(point.y() + size.height()); 
      return point; 
    } 

    QPoint& operator-=(QPoint& point, const QSize& size) { 
      point.setX(point.x() - size.width()); 
      point.setY(point.y() - size.height()); 
      return point; 
    } 
```

The following operators add and subtract an integer from a rectangle, and return the resulting rectangle. The addition operator expands the size of the rectangle in every direction, while the subtraction operator shrinks the rectangle in every direction:

```cpp
    QRect& operator+=(QRect& rect, int size) { 
      rect.setLeft(rect.left() - size); 
      rect.setTop(rect.top() - size); 
      rect.setWidth(rect.width() + size); 
      rect.setHeight(rect.height() + size); 
      return rect; 
    } 

    QRect& operator-=(QRect& rect, int size) { 
      rect.setLeft(rect.left() + size); 
      rect.setTop(rect.top() + size); 
      rect.setWidth(rect.width() - size); 
      rect.setHeight(rect.height() - size); 
      return rect; 
    } 
```

The `writePoint` and `readPoint` functions write and read a point from a file. They write and read the *x* and *y* coordinates separately:

```cpp
    void writePoint(ofstream& outStream, const QPoint& point) { 
      int x = point.x(), y = point.y(); 
      outStream.write((char*) &x, sizeof x); 
      outStream.write((char*) &y, sizeof y); 
    } 

    void readPoint(ifstream& inStream, QPoint& point) { 
      int x, y; 
      inStream.read((char*) &x, sizeof x); 
      inStream.read((char*) &y, sizeof y); 
      point = QPoint(x, y); 
    } 
```

The `writeColor` and `readColor` functions write and read a color from a file. A color is made up of the `red`, `green`, and `blue` components. Each component is an integer value between `0` and `255` inclusive. The methods write and read the components from a file stream:

```cpp
    void writeColor(ofstream& outStream, const QColor& color) { 
      int red = color.red(), green = color.green(), 
      blue = color.blue(); 
      outStream.write((char*) &red, sizeof red); 
      outStream.write((char*) &green, sizeof green); 
      outStream.write((char*) &blue, sizeof blue); 
    } 

    void readColor(ifstream& inStream, QColor& color) { 
      int red, green, blue; 
      inStream.read((char*) &red, sizeof red); 
      inStream.read((char*) &green, sizeof green); 
      inStream.read((char*) &blue, sizeof blue);
```

When the components have been read, we create a `QColor` object that we assign the `color` parameter:

```cpp
      color = QColor(red, green, blue); 
    } 
```

The `makeRect` function creates a rectangle centered around the point:

```cpp
    QRect makeRect(const QPoint& centerPoint, int halfSide) { 
      return QRect(centerPoint.x() - halfSide, 
                   centerPoint.y() - halfSide, 
                   2 * halfSide, 2 * halfSide); 
    } 
```

# Building the drawing program

Let's now start a new project, where we take advantage of the main window and document widget classes of the previous section—*The drawing program*. We will start with a basic version in this chapter, and we will continue to build a more advanced version in the next chapter. With the drawing program of this chapter we can draw lines, rectangles, and ellipses in different colors. We can also save and load our drawings. Note that in this project the window and widget classes inherit from the `MainWindow` and `DocumentWidget` classes of the previous section.

# The Figure base class

The figures of the application constitute a class hierarchy where the `Figure` is the base class. Its subclasses are `Line`, `RectangleX`, and `EllipseX`, which are described later on. We cannot use the names *Rectangle* and *Ellipse* for our classes, since that would clash with Qt methods with the same names. I have chosen to simply add an '`X`' to the names.

The `Figure` class is abstract, which means that we cannot create an object of the class. We can only use it as a base class, which sub classes inherit.

**Figure.h:**

```cpp
    #ifndef FIGURE_H 
    #define FIGURE_H 

    enum FigureId {LineId, RectangleId, EllipseId}; 

    #include <QtWidgets> 
    #include <FStream> 
    using namespace std; 

    class Figure { 
      public: 
        Figure(); 
```

The following methods are pure virtual, which means that they do not need to be defined. A class with at least one pure virtual method becomes abstract. The sub classes must define all the pure virtual methods of all its base classes, or become abstract themselves. In this way, it is guaranteed that all methods of all non-abstract classes are defined.

Each sub class defines `getId` and returns the identity enumeration of its class:

```cpp
    virtual FigureId getId() const = 0; 
```

Each figure has a first and last point, and it is up to each sub class to define them:

```cpp
    virtual void initializePoints(QPoint point) = 0; 
    virtual void setLastPoint(QPoint point) = 0; 
```

The `isClick` method returns `true` if the figure is hit by the point:

```cpp
    virtual bool isClick(QPoint mousePoint) = 0; 
```

The `move` method moves the figures a certain distance:

```cpp
    virtual void move(QSize distance) = 0; 
```

The `draw` method draws the figure on the painter area:

```cpp
    virtual void draw(QPainter &painter) const = 0; 
```

The `write` and `read` methods write and read the figure from a file; `write` is constant since it does not change the figure:

```cpp
    virtual bool write(ofstream& outStream) const; 
    virtual bool read(ifstream& inStream); 
```

The `color` method returns the color of the figure. It comes in two versions, where the first version is constant and returns a reference to a constant `QColor` object, while the second version is non-constant and returns a reference to a non-constant object:

```cpp
    const QColor& color() const {return m_color;} 
    QColor& color() {return m_color;}
```

The `filled` methods apply to two-dimensional figures (rectangles and ellipses) only. They return `true` if the figure is filled. Note that the second version returns a reference to the `m_filled` field, which allows the caller of the method to modify the value of `m_filled`:

```cpp
    virtual bool filled() const {return m_filled;} 
    virtual bool& filled() {return m_filled;} 
```

When a figure is marked, it is drawn with small squares at its corners. The side of the squares are defined by the static field `Tolerance`:

```cpp
    static const int Tolerance; 
```

The `writeColor` and `readColor` methods are auxiliary methods that read and write a color. They are static since they are called by methods outside the `Figure` class hierarchy:

```cpp
        static void writeColor(ofstream& outStream, 

                               const QColor& color); 
        static void readColor(ifstream& inStream, QColor& color); 
```

Each figure has a color, and it could be marked or filled:

```cpp
      private: 
        QColor m_color; 
        bool m_marked = false, m_filled = false; 
    }; 

    #endif 
```

The `Figure.cpp` file holds the definitions of the `Figure` class. It defines the `Tolerance` field as well as the `write` and `read` methods.

**Figure.cpp:**  

```cpp
    #include "..\MainWindow\DocumentWidget.h" 
    #include "Figure.h" 
```

`Tolerance` must be defined and initialized in global space since it is static. We define the size of the mark squares to be `6` pixels:

```cpp
    const int Figure::Tolerance(6); 
```

The default constructor is called only when figures are read from a file:

```cpp
    Figure::Figure() { 
      // Empty. 
    }
```

The `write` and `read` methods write and read the color of the figure, and whether the figure is filled:

```cpp
    bool Figure::write(ofstream& outStream) const { 
      writeColor(outStream, m_color); 
      outStream.write((char*) &m_filled, sizeof m_filled); 
      return ((bool) outStream); 
    } 

    bool Figure::read(ifstream& inStream) { 
      readColor(inStream, m_color); 
      inStream.read((char*) &m_filled, sizeof m_filled); 
      return ((bool) inStream); 
    } 
```

# The Line sub class

The `Line` class is a sub class of `Figure`. It becomes non-abstract by defining each pure virtual method of `Figure`. A line is drawn between two end-points, represented by the `m_firstPoint` to `m_lastPoint` fields in `Line`:

![](img/4723c801-7788-48b8-8399-9bc2e7f4d2b2.png)

**Line.h:**

```cpp
    #ifndef LINE_H 
    #define LINE_H 

    #include <FStream> 
    using namespace std; 

    #include "Figure.h" 

    class Line : public Figure { 
      public:
```

The default constructor is called only when reading `Line` objects from a file; `getId` simply returns the identity enumeration of the line:

```cpp
    Line(); 
    FigureId getId() const {return LineId;} 
```

A line has two endpoints. Both points are set when the line is created, the second point is then modified when the user moves it:

```cpp
    void initializePoints(QPoint point); 
    void setLastPoint(QPoint point); 
```

The `isClick` method returns `true` if the mouse click is located on the line (with some tolerance):

```cpp
    bool isClick(QPoint mousePoint); 
```

The `move` method moves the line (both its end-points) the given distance:

```cpp
    void move(QSize distance); 
```

The `draw` method draws the line on the `QPainter` object:

```cpp
    void draw(QPainter& painter) const; 
```

The `write` and `read` methods write and read the end-points of the line from a file stream:

```cpp
    bool write(ofstream& outStream) const; 
    bool read(ifstream& inStream); 
```

The first and last points of the line are stored in the `Line` object:

```cpp
    private: 
      QPoint m_firstPoint, m_lastPoint; 
    }; 

    #endif 
```

The `Line.cpp` file defines the methods of the `Line` class.

**Line.cpp:** 

```cpp
    #include "..\MainWindow\DocumentWidget.h" 
    #include "Line.h" 

    Line::Line() { 
      // Empty. 
    }
```

The `initializePoints` method is called when the user adds a new line to the drawing. It sets both its end-points:

```cpp
    void Line::initializePoints(QPoint point) { 
      m_firstPoint = point; 
      m_lastPoint = point; 
    } 
```

The `setLastPoint` method is called when the user has added the line and modifies its shape. It sets the last point:

```cpp
    void Line::setLastPoint(QPoint point) { 
      m_lastPoint = point; 
    } 
```

The `isClick` method tests whether the user has clicked with the mouse on the line. We have two cases to consider. The first case is a special case that occurs when the line is completely vertical, when the *x*-coordinates of the end-points are equal. We use the Qt `QRect` class to create a rectangle surrounding the line, and test whether the point is enclosed in the rectangle:

![](img/d9377e8e-8a2a-4fdf-8c0f-3e6f23276884.png)

```cpp
    bool Line::isClick(QPoint mousePoint) { 
      if (m_firstPoint.x() == m_lastPoint.x()) { 
        QRect lineRect(m_firstPoint, m_lastPoint); 
        lineRect.normalized(); 
        lineRect += Tolerance; 
        return lineRect.contains(mousePoint); 
      }
```

In a general case, where the line is not vertical, we start by creating an enclosing rectangle and test if the mouse point is in it. If it is, we set `leftPoint` to the leftmost point of `firstPoint` and `lastPoint`, and `rightPoint` to the rightmost point. We then calculate the width (`lineWidth`) and height (`lineHeight`) of the enclosing rectangle, as well as the distance between `rightPoint` and `mousePoint` in the *x* and *y* directions (`diffWidth` and `diffHeight`).

![](img/f051abb6-3200-45a4-bd8e-1ce97c0443a2.png)

Due to uniformity, the following equation is true if the mouse pointer hits the line:

![](img/94c97888-f928-420d-9f23-9f7308d283d9.png)

However, in order for the left-hand expression to become exactly zero, the user has to click exactly on the line. Therefore, let us allow for a small tolerance. Let's use the `Tolerance` field:

![](img/fb006091-e45c-4af5-9ac4-13234e8b1bfe.png)

```cpp
        else { 
          QPoint leftPoint = (m_firstPoint.x() < m_lastPoint.x()) 
                             ? m_firstPoint : m_lastPoint, 
                 rightPoint = (m_firstPoint.x() < m_lastPoint.x()) 
                              ? m_lastPoint : m_firstPoint; 

          if ((leftPoint.x() <= mousePoint.x()) && 
              (mousePoint.x() <= rightPoint.x())) { 
            int lineWidth = rightPoint.x() - leftPoint.x(), 
                lineHeight = rightPoint.y() - leftPoint.y(); 

            int diffWidth = mousePoint.x() - leftPoint.x(), 
                diffHeight = mousePoint.y() - leftPoint.y(); 
```

We must convert `lineHeight` to a double in order to perform non-integer division:

```cpp
          return (fabs(diffHeight - (((double) lineHeight) / 
                       lineWidth) * diffWidth) <= Tolerance); 
        } 
```

If the mouse point is located outside the rectangle enclosing the line, we simply return `false`:

```cpp
        return false; 
      } 
    } 
```

The `move` method simply moves both the endpoints of the line:

```cpp
    void Line::move(QSize distance) { 
      m_firstPoint += distance; 
      m_lastPoint += distance; 
    } 
```

When drawing the line, we set the pen color and draw the line. The `color` method of the `Figure` class returns the color of the line:

```cpp
    void Line::draw(QPainter& painter) const { 
      painter.setPen(color()); 
      painter.drawLine(m_firstPoint, m_lastPoint); 
    } 
```

When writing the line, we first call `write` in `Figure` to write the color of the figure. We then write the endpoints of the line. Finally, we return the Boolean value of the output stream, which is `true` if the writing was successful:

```cpp
    bool Line::write(ofstream& outStream) const { 
      Figure::write(outStream); 
      writePoint(outStream, m_firstPoint); 
      writePoint(outStream, m_lastPoint); 
      return ((bool) outStream); 
    }
```

In the same way, when reading the line, we first call `read` in `Figure` to read the color of the line. We then read the endpoints of the line and return the Boolean value of the input stream:

```cpp
    bool Line::read(ifstream& inStream) { 
      Figure::read(inStream); 
      readPoint(inStream, m_firstPoint); 
      readPoint(inStream, m_lastPoint); 
      return ((bool) inStream); 
    } 
```

# The Rectangle sub class

`RectangleX` is a sub class of `Figure` that handles a rectangle. Similar to `Line`, it holds two points, which holds opposite corners of the rectangle:

**Rectangle.h**  

```cpp
    #ifndef RECTANGLE_H 
    #define RECTANGLE_H 

    #include <FStream> 
    using namespace std; 

    #include "Figure.h" 

    class RectangleX : public Figure { 
      public: 
```

Similar to the preceding `Line` class, `RectangleX` has a default constructor that is used when reading the object from a file:

```cpp
        RectangleX(); 
        virtual FigureId getId() const {return RectangleId;} 

        RectangleX(const RectangleX& rectangle); 

        virtual void initializePoints(QPoint point); 
        virtual void setLastPoint(QPoint point); 

        virtual bool isClick(QPoint mousePoint); 
        virtual void move(QSize distance); 
        virtual void draw(QPainter& painter) const; 

        virtual bool write(ofstream& outStream) const; 
        virtual bool read(ifstream& inStream); 

      protected: 
        QPoint m_topLeft, m_bottomRight; 
    }; 

    #endif 
```

**Rectangle.cpp**  

```cpp
    #include "..\MainWindow\DocumentWidget.h" 
    #include "Rectangle.h" 

    RectangleX::RectangleX() { 
      // Empty. 
    } 
```

The `initializePoints` and `setLastPoint` methods work in a way similar to their counterparts in `Line`: `initializePoints` sets both the corner points, while `setLastPoint` sets the last corner point:

```cpp
    void RectangleX::initializePoints(QPoint point) { 
      m_topLeft = point; 
      m_bottomRight = point; 
    } 

    void RectangleX::setLastPoint(QPoint point) { 
      m_bottomRight = point; 
    } 
```

The `isClick` method is simpler than its counterpart in `Line`:

```cpp
    bool RectangleX::isClick(QPoint mousePoint) { 
      QRect areaRect(m_topLeft, m_bottomRight); 
```

If the rectangle is filled, we simply check whether the mouse click hit the rectangle by calling `contains` in `QRect`:

```cpp
      if (filled()) { 
        return areaRect.contains(mousePoint); 
      } 
```

If the rectangle is not filled, we need to check whether the mouse clicked on the border of the rectangle. To do so, we create two slightly smaller and larger rectangles. If the mouse click hit the larger rectangle, but not the smaller one, we consider the rectangle border to be hit:

```cpp
      else { 
        QRect largeAreaRect(areaRect), smallAreaRect(areaRect); 

        largeAreaRect += Tolerance; 
        smallAreaRect -= Tolerance; 

        return largeAreaRect.contains(mousePoint) && 
               !smallAreaRect.contains(mousePoint); 
      } 

      return false; 
    } 
```

When moving the rectangle, we simply move the first and last corners:

```cpp
    void RectangleX::move(QSize distance) { 
      addSizeToPoint(m_topLeft, distance); 
      addSizeToPoint(m_bottomRight, distance); 
    } 
```

When drawing a rectangle, we first set the pen color by calling `color` in `Figure`:

```cpp
    void RectangleX::draw(QPainter& painter) const { 
      painter.setPen(color()); 
```

If the rectangle is filled, we simply call `fillRect` on the `QPainter` object:

```cpp
      if (filled()) { 
        painter.fillRect(QRect(m_topLeft, m_bottomRight), color()); 
      } 
```

If the rectangle is unfilled, we disable the brush to make the rectangle hollow, and we then call `drawRect` on the `QPainter` object to draw the border of the rectangle:

```cpp
      else { 
        painter.setBrush(Qt::NoBrush); 
        painter.drawRect(QRect(m_topLeft, m_bottomRight)); 
      } 
    } 
```

The `write` method first calls `write` in `Figure`, and it then writes the first and last corners of the rectangle:

```cpp
    bool RectangleX::write(ofstream& outStream) const { 
      Figure::write(outStream); 
      writePoint(outStream, m_topLeft); 
      writePoint(outStream, m_bottomRight); 
      return ((bool) outStream); 
    }
```

In the same way, `read` first calls `read` in `Figure`, and then reads the first and last corners of the rectangle:

```cpp
    bool RectangleX::read (ifstream& inStream) { 
      Figure::read(inStream); 
      readPoint(inStream, m_topLeft); 
      readPoint(inStream, m_bottomRight); 
      return ((bool) inStream); 
    } 
```

# The Ellipse sub class

`EllipseX` is a sub class of `RectangleX` that handles an ellipse. Part of the functionality of `RectangleX` is reused in `EllipseX`. More specifically, `initializePoints`, `setLastPoint`, `move`, `write`, and `read` are overridden from `RectangleX`.

**Ellipse.h:**  

```cpp
    #ifndef ELLIPSE_H 
    #define ELLIPSE_H 

    #include "Rectangle.h" 

    class EllipseX : public RectangleX { 
      public: 
        EllipseX(); 
        FigureId getId() const {return EllipseId;} 

        EllipseX(const EllipseX& ellipse); 

        bool isClick(QPoint mousePoint); 
        void draw(QPainter& painter) const; 
    }; 

    #endif 
```

**Ellipse.cpp:**  

```cpp
    #include "..\MainWindow\DocumentWidget.h" 
    #include "Ellipse.h" 

    EllipseX::EllipseX() { 
      // Empty. 
    }
```

The `isClick` method of `EllipseX` is similar to its counterpart in `RectangleX`. We use the Qt `QRegion` class to create elliptic objects that we compare to the mouse click:

```cpp
    bool EllipseX::isClick(QPoint mousePoint) { 
      QRect normalRect(m_topLeft, m_bottomRight); 
      normalRect.normalized(); 
```

If the ellipse is filled, we create an elliptic region and test whether the mouse click hit the region:

```cpp
      if (filled()) { 
        QRegion normalEllipse(normalRect, QRegion::Ellipse); 
        return normalEllipse.contains(mousePoint); 
      } 
```

If the ellipse in unfilled, we create slightly smaller and larger elliptic regions. If the mouse click hit the smaller region, but not the smaller one, we consider the border of the ellipse to be hit:

```cpp
      else { 
        QRect largeRect(normalRect), smallRect(normalRect); 
        largeRect += Tolerance; 
        smallRect -= Tolerance; 

        QRegion largeEllipse(largeRect, QRegion::Ellipse), 
                smallEllipse(smallRect, QRegion::Ellipse); 

        return (largeEllipse.contains(mousePoint) && 
                !smallEllipse.contains(mousePoint)); 
      } 
    } 
```

When drawing an ellipse, we first set the pen color by calling `color` in `Figure`:

```cpp
    void EllipseX::draw(QPainter& painter) const { 
      painter.setPen(color()); 
```

If the ellipse is filled, we set the brush and draw the ellipse:

```cpp
      if (filled()) { 
        painter.setBrush(color()); 
        painter.drawEllipse(QRect(m_topLeft, m_bottomRight)); 
      }
```

If the ellipse is unfilled, we set the brush to hollow and draw the ellipse border:

```cpp
      else { 
        painter.setBrush(Qt::NoBrush); 
        painter.drawEllipse(QRect(m_topLeft, m_bottomRight)); 
      } 
    }
```

# Drawing the window

The `DrawingWindow` class is a sub class to the `MainWindow` class of the previous section.

**DrawingWindow.h:**  

```cpp
    #ifndef DRAWINGWINDOW_H 
    #define DRAWINGWINDOW_H 

    #include <QMainWindow> 
    #include <QActionGroup> 

    #include "..\MainWindow\MainWindow.h" 
    #include "DrawingWidget.h" 

    class DrawingWindow : public MainWindow { 
      Q_OBJECT 

      public: 
        DrawingWindow(QWidget* parentWidgetPtr = nullptr); 
        ~DrawingWindow(); 

      public: 
        void closeEvent(QCloseEvent *eventPtr)
             { m_drawingWidgetPtr->closeEvent(eventPtr); } 

      private: 
        DrawingWidget* m_drawingWidgetPtr; 
        QActionGroup* m_figureGroupPtr; 
    }; 

    #endif // DRAWINGWINDOW_H 
```

**DrawingWindow.cpp:**  

```cpp
    #include "..\MainWindow\DocumentWidget.h" 
    #include "DrawingWindow.h"
```

The constructor sets the size of the window to `1000` * `500` pixels:

```cpp
    DrawingWindow::DrawingWindow(QWidget* parentWidgetPtr 
                                 /* = nullptr */) 
     :MainWindow(parentWidgetPtr) { 
      resize(1000, 500); 
```

The `m_drawingWidgetPtr` field is initialized to point at an object of the `DrawingWidget` class, which is then set to the center part of the window:

```cpp
      m_drawingWidgetPtr = new DrawingWidget(this); 
      setCentralWidget(m_drawingWidgetPtr); 
```

The standard file menu is added to the window menu bar:

```cpp
      addFileMenu(); 
```

We then add the application-specific format menu. It is connected to the `onMenuShow` method of the `DocumentWidget` class of the previous section:

```cpp
      { QMenu* formatMenuPtr = menuBar()->addMenu(tr("F&ormat")); 
        connect(formatMenuPtr, SIGNAL(aboutToShow()), 
                this, SLOT(onMenuShow())); 
```

The format menu holds the color and fill items:

```cpp
        addAction(formatMenuPtr, tr("&Color"), 
                  SLOT(onColor()), QKeySequence(Qt::ALT + Qt::Key_C), 
                  QString(), nullptr, tr("Figure Color")); 
```

The fill item will be enabled when the next figure of the drawing program is a two-dimensional figure (rectangle or ellipse):

```cpp
        addAction(formatMenuPtr, tr("&Fill"), 
                  SLOT(onFill()), QKeySequence(Qt::CTRL + Qt::Key_F), 
                  QString(), nullptr, tr("Figure Fill"), 
                  LISTENER(isFillEnabled)); 
      } 
```

For the figure menu, we create a new action group for the line, rectangle, and ellipse item. Only one of them shall be marked at the same time:

```cpp
      { m_figureGroupPtr = new QActionGroup(this); 

        QMenu* figureMenuPtr = menuBar()->addMenu(tr("F&igure")); 
        connect(figureMenuPtr, SIGNAL(aboutToShow()), 
                this, SLOT(onMenuShow()));
```

The currently selected item shall be marked with a radio button:

```cpp
        addAction(figureMenuPtr, tr("&Line"), 
                  SLOT(onLine()), QKeySequence(Qt::CTRL + Qt::Key_L), 
                  QString(), nullptr, tr("Line Figure"), nullptr, 
                  LISTENER(isLineChecked), m_figureGroupPtr); 
        addAction(figureMenuPtr, tr("&Rectangle"), 
                  SLOT(onRectangle()), 
                  QKeySequence(Qt::CTRL + Qt::Key_R), 
                  QString(), nullptr, tr("Rectangle Figure"), nullptr, 
                  LISTENER(isRectangleChecked), m_figureGroupPtr); 
        addAction(figureMenuPtr, tr("&Ellipse"), 
                  SLOT(onEllipse()), 
                  QKeySequence(Qt::CTRL + Qt::Key_E), 
                  QString(), nullptr, tr("Ellipse Figure"), nullptr, 
                  LISTENER(isEllipseChecked), m_figureGroupPtr); 
      } 
    } 
```

The destructor deallocates the figure group that was dynamically allocated in the constructor:

```cpp
    DrawingWindow::~DrawingWindow() { 
      delete m_figureGroupPtr; 
    } 
```

# Drawing the widget

`DrawingWidget` is a sub class of `DocumentWidget` in the previous section. It handles mouse input, painting of the figures, as well as saving and loading of the drawing. It also provides methods for deciding when the menu items shall be marked and enabled.

**DrawingWidget.h:**  

```cpp
    #ifndef DRAWINGWIDGET_H 
    #define DRAWINGWIDGET_H 

    #include "..\MainWindow\MainWindow.h" 
    #include "..\MainWindow\DocumentWidget.h" 
    #include "Figure.h" 

    class DrawingWidget : public DocumentWidget { 
      Q_OBJECT 

      public: 
        DrawingWidget(QWidget* parentWidgetPtr); 
        ~DrawingWidget(); 
```

The `mousePressEvent`, `mouseReleaseEvent`, and `mouseMoveEvent` are overridden methods that are called when the user presses or releases one of the mouse keys or moves the mouse:

```cpp
      public: 
        void mousePressEvent(QMouseEvent *eventPtr); 
        void mouseReleaseEvent(QMouseEvent *eventPtr); 
        void mouseMoveEvent(QMouseEvent *eventPtr); 
```

The `paintEvent` method is called when the window needs to be repainted. That can happen for several reasons. For instance, the user can modify the size of the window. The repainting can also be forced by a call to the `update` method, which causes `paintEvent` to be called eventually:

```cpp
        void paintEvent(QPaintEvent *eventPtr); 
```

The `newDocument` method is called when the user selects the new menu item, `writeFile` is called when the user selects the save or save as item, and `readFile` is called when the user selects the open item:

```cpp
      private: 
        void newDocument() override; 
        bool writeFile(const QString& filePath); 
        bool readFile(const QString& filePath); 
        Figure* createFigure(FigureId figureId); 
```

The `onColor` and `onFill` methods are called when the user selects the color and fill menu items:

```cpp
      public slots: 
        void onColor(); 
        void onFill(); 
```

The `isFillEnabled` method is called before the user selects the format menu. If it returns `true`, the fill item becomes enabled:

```cpp
        DEFINE_LISTENER(DrawingWidget, isFillEnabled);
```

The `isLineChecked`, `isRectangleChecked`, and `isEllipseChecked` methods are also called before the figure menu becomes visible. The items become marked with a radio button if the methods return `true`:

```cpp
        DEFINE_LISTENER(DrawingWidget, isLineChecked); 
        DEFINE_LISTENER(DrawingWidget, isRectangleChecked); 
        DEFINE_LISTENER(DrawingWidget, isEllipseChecked); 
```

The `onLine`, `onRectangle`, and `isEllipse` methods are called when the user selects the line, rectangle, and ellipse menu items:

```cpp
        void onLine(); 
        void onRectangle(); 
        void onEllipse(); 
```

When running, the application can hold the `Idle`, `Create`, or `Move` modes:

*   `Idle`: When the application is waiting for input from the user.
*   `Create`: When the user is adding a new figure to the drawing. Occurs when the user presses the left mouse button without hitting a figure. A new figure is added and its end-point is modified until the user releases the mouse button.
*   `Move`: When the user is moving a figure. Occurs when the user presses the left mouse button and hitting a figure. The figure is moved until the user releases the mouse button.

```cpp
      private: 
        enum ApplicationMode {Idle, Create, Move}; 
        ApplicationMode m_applicationMode = Idle; 
        void setApplicationMode(ApplicationMode mode); 
```

The `m_currColor` field holds the color of the next figure to be added by the user; `m_currFilled` decides whether the next figure (if it is a rectangle or an ellipse) shall be filled. The `m_addFigureId` method holds the identity integer of the next type of figure (line, rectangle, or ellipse) to be added by the user:

```cpp
        QColor m_currColor = Qt::black; 
        bool m_currFilled = false; 
        FigureId m_addFigureId = LineId; 
```

When the user presses a mouse button and moves a figure, we need to store the previous mouse point in order to calculate the distance the figure has been moved since the last mouse events:

```cpp
        QPoint m_mousePoint;
```

Finally, `m_figurePtrList` holds pointers to the figures of the drawing. The top-most figure in the drawing is placed at the end of the list:

```cpp
        QList<Figure*> m_figurePtrList; 
    }; 

    #endif // DRAWINGWIDGET_H 
```

**DrawingWidget.cpp:**

```cpp
    #include "..\MainWindow\DocumentWidget.h" 
    #include "DrawingWidget.h" 

    #include "Line.h" 
    #include "Rectangle.h" 
    #include "Ellipse.h" 
```

The constructor calls the constructor the base class `DocumentWidget` with the title `Drawing`. It also sets the save and load mask to `Drawing files (*.drw)`, which means that the default files selected by the standard save and load dialogs have the suffix `drw`:

```cpp
    DrawingWidget::DrawingWidget(QWidget* parentWidgetPtr) 
     :DocumentWidget(tr("Drawing"), tr("Drawing files (*.drw)"), 
                     parentWidgetPtr) { 
      // Empty. 
    } 
```

The destructor deallocates the figure pointers of the figure pointer list:

```cpp
    DrawingWidget::~DrawingWidget() { 
      for (Figure* figurePtr : m_figurePtrList) { 
        delete figurePtr; 
      } 
    } 
```

The `setApplicationMode` method sets the application mode and calls `onMenuShow` in the main window for the toolbar icons to be correctly enabled:

```cpp
    void DrawingWidget::setApplicationMode(ApplicationMode mode) { 
      m_applicationMode = mode; 
      ((MainWindow*) parent())->onMenuShow(); 
    }
```

When the user selects the new menu item, `newDocument` is called. The figures of the figure pointer list are deallocated, and the list itself is cleared:

```cpp
    void DrawingWidget::newDocument() { 
      for (Figure* figurePtr : m_figurePtrList) { 
        delete figurePtr; 
      } 
      m_figurePtrList.clear(); 
```

The next figure to be added by the user is a black line, and the filled status is `false`:

```cpp
      m_currColor = Qt::black; 
      m_addFigureId = LineId; 
      m_currFilled = false; 
    } 
```

The `writeFile` method is called when the user selects the save or save as menu items:

```cpp
    bool DrawingWidget::writeFile(const QString& filePath) { 
      ofstream outStream(filePath.toStdString()); 
```

We start by writing the current color and fill status. We then continue by writing the size of the figure pointer list, and the figures themselves:

```cpp
      if (outStream) { 
        writeColor(outStream, m_currColor); 
        outStream.write((char*) &m_currFilled, sizeof m_currFilled); 

        int size = m_figurePtrList.size(); 
        outStream.write((char*) &size, sizeof size); 
```

For each figure, we first write its identity number, and we then write the figure itself:

```cpp
        for (Figure* figurePtr : m_figurePtrList) { 
          FigureId figureId = figurePtr->getId(); 
          outStream.write((char*) &figureId, sizeof figureId); 
          figurePtr->write(outStream); 
        } 

        return ((bool) outStream); 
      } 
```

If the file was not possible to open, `false` is returned:

```cpp
      return false; 
    }
```

The `readFile` method is called when the user selects the open menu item. In the same way as in `writeFile` previously, we read the color and fill status, the size of the figure pointer list, and then the figures themselves:

```cpp
    bool DrawingWidget::readFile(const QString& filePath) { 
      ifstream inStream(filePath.toStdString()); 

      if (inStream) { 
        readColor(inStream, m_currColor); 
        inStream.read((char*) &m_currFilled, sizeof m_currFilled); 

        int size; 
        inStream.read((char*) &size, sizeof size); 
```

When reading the figure, we first read its identity number, and call `createFigure` to create an object of the class corresponding to the figure's identity number. We then read the fields of the figure by calling `read` on its pointer. Note that we do not really know (or care) what kind of figure it is. We simply call read to the figure pointer, which in fact points to an object of `Line`, `RectangleX`, or `EllipseX`:

```cpp
        for (int count = 0; count < size; ++count) { 
          FigureId figureId = (FigureId) 0; 
          inStream.read((char*) &figureId, sizeof figureId); 
          Figure* figurePtr = createFigure(figureId); 
          figurePtr->read(inStream); 
          m_figurePtrList.push_back(figurePtr); 
        } 

        return ((bool) inStream); 
      } 

      return false; 
    } 
```

The `createFigure` method dynamically creates an object of the `Line`, `RectangleX`, or `EllipseX` class, depending on the value of the `figureId` parameter:

```cpp
    Figure* DrawingWidget::createFigure(FigureId figureId) { 
      Figure* figurePtr = nullptr; 

      switch (figureId) { 
        case LineId: 
          figurePtr = new Line(); 
          break; 

        case RectangleId: 
          figurePtr = new RectangleX(); 
          break; 

        case EllipseId: 
          figurePtr = new EllipseX(); 
          break; 
      } 

      return figurePtr; 
    } 
```

The `onColor` method is called when the user selects the color menu item. It sets the color of the next figure to be added by the user:

```cpp
    void DrawingWidget::onColor() { 
      QColor newColor = QColorDialog::getColor(m_currColor, this); 

      if (newColor.isValid() && (m_currColor != newColor)) { 
        m_currColor = newColor; 
        setModifiedFlag(true); 
      } 
    } 
```

The `isFillEnabled` method is called before the format menu becomes visible, and returns `true` if the next figure to be added by the user is a rectangle or an ellipse:

```cpp
    bool DrawingWidget::isFillEnabled() { 
      return (m_addFigureId == RectangleId) || 
             (m_addFigureId == EllipseId); 
    } 
```

The `onFill` method is called when the user selects fill menu item. It inverts the `m_currFilled` field. It also sets the modified flag since the document has been affected:

```cpp
    void DrawingWidget::onFill() { 
      m_currFilled = !m_currFilled; 
      setModifiedFlag(true); 
    } 
```

The `isLineChecked`, `isRectangleChecked`, and `isEllipseChecked` methods are called before the figure menu becomes visible. If they return `true`, the items become checked with a radio button if the next figure to be added is the figure in question:

```cpp
    bool DrawingWidget::isLineChecked() { 
      return (m_addFigureId == LineId); 
    } 

    bool DrawingWidget::isRectangleChecked() { 
      return (m_addFigureId == RectangleId); 
    } 

    bool DrawingWidget::isEllipseChecked() { 
      return (m_addFigureId == EllipseId); 
    } 
```

The `onLine`, `onRectangle`, and `onEllipse` methods are called when the user selects the items in the figure menu. They set the next figure to be added by the user to the figure in question:

```cpp
    void DrawingWidget::onLine() { 
      m_addFigureId = LineId; 
    } 

    void DrawingWidget::onRectangle() { 
      m_addFigureId = RectangleId; 
    } 

    void DrawingWidget::onEllipse() { 
      m_addFigureId = EllipseId; 
    } 
```

The `mousePressEvent` method is called every time the user presses one of the mouse keys. First, we need to check if they have pressed the left mouse key:

```cpp
    void DrawingWidget::mousePressEvent(QMouseEvent* eventPtr) { 
      if (eventPtr->buttons() == Qt::LeftButton) { 
```

In the call to `mouseMoveEvent` in the following snippet, we need to keep track of the latest mouse point in order to calculate the distance between mouse movements. Therefore, we set `m_mousePoint` to the mouse point:

```cpp
        m_mousePoint = eventPtr->pos(); 
```

We iterate through the figure pointer list and, for each figure, we check if the figure has been hit by the mouse click by calling `isClick`. We need to iterate backwards in a rather awkward manner in order to find the top-most figure first. We use the `reverse_iterator` class and the `rbegin` and `rend` methods in order to iterate backwards:

```cpp
        for (QList<Figure*>::reverse_iterator iterator = 
             m_figurePtrList.rbegin(); 
             iterator != m_figurePtrList.rend(); ++iterator) {
```

We use the dereference operator (`*`) to obtain the figure pointer in the list:

```cpp
              Figure* figurePtr = *iterator; 
```

If the figure has been hit by the mouse click, we set the application mode to move. We also place the figure at the end of the list, so that it appears to be top-most in the drawing, by calling `removeOne` and `push_back` on the list. Finally, we break the loop since we have found the figure we are looking for:

```cpp
          if (figurePtr->isClick(m_mousePoint)) { 
            setApplicationMode(Move); 
            m_figurePtrList.removeOne(figurePtr); 
            m_figurePtrList.push_back(figurePtr); 
            break; 
          } 
        } 
```

If the application mode is still idle (has not moved), we have not found a figure hit by the mouse click. In that case, we set the application mode to create and call `createFigure` to find a figure to copy. We then set the color and filled status as well as the points of the figure. Finally, we add the figure pointer to the figure pointer list by calling `push_back` (which is added at the end of the list in order for it to appear at the top of the drawing) and set the modified flag to `true`, since the drawing has been modified:

```cpp
        if (m_applicationMode == Idle) { 
          setApplicationMode(Create); 
          Figure* newFigurePtr = createFigure(m_addFigureId); 
          newFigurePtr->color() = m_currColor; 
          newFigurePtr->filled() = m_currFilled; 
          newFigurePtr->initializePoints(m_mousePoint); 
          m_figurePtrList.push_back(newFigurePtr); 
          setModifiedFlag(true); 
        } 
      } 
    } 
```

The `mouseMoveEvent` is called every time the user moves the mouse. First, we need to check that the user presses the left mouse key when they move the mouse:

```cpp
    void DrawingWidget::mouseMoveEvent(QMouseEvent* eventPtr) { 
      if (eventPtr->buttons() == Qt::LeftButton) { 
        QPoint newMousePoint = eventPtr->pos();
```

We then check the application mode. If we are in the process of adding a new figure to the drawing, we modify its last point:

```cpp
        switch (m_applicationMode) { 
          case Create: 
            m_figurePtrList.back()->setLastPoint(m_mousePoint); 
            break; 
```

If we are in the process of moving a figure, we calculate the distance since the last mouse event and move the figure placed at the end of the figure pointer list. Remember that the figure hit by the mouse click was placed at the end of the figure pointer list in the preceding `mousePressEvent`:

```cpp
          case Move: { 
              QSize distance(newMousePoint.x() - m_mousePoint.x(), 
                             newMousePoint.y() - m_mousePoint.y()); 
              m_figurePtrList.back()->move(distance); 
              setModifiedFlag(true); 
            } 
            break; 
        } 
```

Finally, we update the current mouse point for the next call to `mouseMoveEvent`. We also call the update method to force a repainting of the window:

```cpp
        m_mousePoint = newMousePoint; 
        update(); 
      } 
    } 
```

The `mouseReleaseEvent` method is called when the user releases one of the mouse buttons. We set the application mode to idle:

```cpp
    void DrawingWidget::mouseReleaseEvent(QMouseEvent* eventPtr) { 
      if (eventPtr->buttons() == Qt::LeftButton) { 
        setApplicationMode(Idle); 
      } 
    } 
```

The `paintEvent` method is called every time the window needs to be repainted. It may happen for several reasons. For instance, the user may have changed the size of the window. It may also be a result of a call to `update` in the Qt `QWidget` class, which forces a repainting of the window and an eventual call to `paintEvent`.

We start by creating a `QPainter` object, which can be regarded as canvas to paint on, and set suitable rendering. We then iterate through the figure pointer list, and draw each figure. In this way, the last figure in the list is drawn at the top of the drawing:

```cpp
    void DrawingWidget::paintEvent(QPaintEvent* /* eventPtr */) { 
      QPainter painter(this); 
      painter.setRenderHint(QPainter::Antialiasing); 
      painter.setRenderHint(QPainter::TextAntialiasing); 

      for (Figure* figurePtr : m_figurePtrList) { 
        figurePtr->draw(painter); 
      } 
    } 
```

# The main function

Finally, we start the application in the `main` function by creating an application object, showing the main window and executing the application.

**Main.cpp:**

```cpp
    #include "DrawingWindow.h" 
    #include <QApplication> 

    int main(int argc, char *argv[]) { 
      QApplication application(argc, argv); 
      DrawingWindow drawingWindow; 
      drawingWindow.show(); 
      return application.exec(); 
    }
```

The following output is received:

![](img/25b6b2a9-d07d-4e9e-b8c8-28b2475353a8.png)

# Building an editor

The next application is an editor, where the user can input and edit text. The current input position is indicated by a caret. It is possible to move the caret with the arrow keys and by clicking with the mouse.

# The Caret class

The `Caret` class handles the caret; that is, the blinking vertical line marking the position of the next character to be input.

**Caret.h:**

```cpp
    #ifndef CARET_H 
    #define CARET_H 

    #include <QObject> 
    #include <QWidget> 
    #include <QTimer> 

    class Caret : public QObject { 
      Q_OBJECT 

      public: 
        Caret(QWidget* parentWidgetPtr = nullptr);

```

The `show` and `hide` methods show and hide the caret. In this application, the caret is never hidden. However, in the advanced version in the next chapter, the caret will be hidden on some occasions:

```cpp
        void show(); 
        void hide(); 
```

The `set` method sets the current size and position of the caret, and `paint` paints it on the `QPainter` object:

```cpp
        void set(QRect rect); 
        void paint(QPainter& painter); 
```

The `onTimer` method is called every time the caret blinks:

```cpp
      public slots: 
        void onTimer(void); 

      private: 
        QWidget* m_parentWidgetPtr; 
```

The `m_visible` field is true when the caret is visible:

```cpp
        bool m_visible, m_blink; 
```

The `m_rect` field handles the timer that makes the caret blink:

```cpp
        QRect m_rect; 
```

The `m_timer` field handles the timer that makes the caret blink:

```cpp
        QTimer m_timer; 
    }; 

    #endif // CARET_H 
```

The `Caret.cpp` file holds the definitions of the methods of the `Caret` class.

**Caret.cpp:**

```cpp
    #include "Caret.h" 
    #include <QPainter>
```

The constructor connects the timer signal to `onTimer`, with the result that `onTimer` is called for every timeout. The timer is then initialized to `500` milliseconds. That is, `onTimer` will be called every `500` milliseconds, and the caret becomes shown and hidden every 500 milliseconds:

```cpp
    Caret::Caret(QWidget* parentWidgetPtr) 
      :m_parentWidgetPtr(parentWidgetPtr) { 
      m_timer.setParent(this); 
      connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimer())); 
      m_timer.start(500); 
    } 
```

The `show` and `hide` methods set the `m_visible` field and force a repainting of the caret area by calling `update` on the parent window:

```cpp
    void Caret::show() { 
      m_visible = true; 
      m_parentWidgetPtr->update(m_rect); 
    } 

    void Caret::hide() { 
      m_visible = false; 
      m_parentWidgetPtr->update(m_rect); 
    } 
```

The `set` method sets the size and position of the caret. However, the width of the caret is always set to one, which makes it appear as a thin vertical line:

```cpp
    void Caret::set(QRect rect) { 
      m_rect = rect; 
      m_rect.setWidth(1); 
      m_parentWidgetPtr->update(m_rect); 
    } 
```

The `onTimer` method is called every 500 milliseconds. It inverts `m_blink` and forces a repaint of the caret. This gives the result that the caret blinks at an interval of one second:

```cpp
    void Caret::onTimer(void) { 
      m_blink = !m_blink; 
      m_parentWidgetPtr->update(m_rect); 
    }
```

The `paint` method is called every time the caret needs to be repainted. The caret is drawn if both `m_visible` and `m_blink` are true, which they are if the caret is set to be visible and the caret is blinking; that is, that the caret is visible in the blinking interval. The area of the caret is cleared before the call to paint, so that if no drawing occurs, the caret is cleared:

```cpp
    void Caret::paint(QPainter& painter) { 
      if (m_visible && m_blink) { 
        painter.save(); 
        painter.setPen(Qt::NoPen); 
        painter.setBrush(Qt::black); 
        painter.drawRect(m_rect); 
        painter.restore(); 
      } 
    } 
```

# Drawing the editor window

`EditorWindow` is a sub class of `MainWindow` in the previous section. It handles the closing of the window. Moreover, it also handles the key press event.

**EditorWindow.h:**  

```cpp
    #ifndef EDITORWINDOW_H 
    #define EDITORWINDOW_H 

    #include <QMainWindow> 
    #include <QActionGroup> 
    #include <QPair> 
    #include <QMap> 

    #include "..\MainWindow\MainWindow.h" 
    #include "EditorWidget.h" 

    class EditorWindow : public MainWindow { 
      Q_OBJECT 

      public: 
        EditorWindow(QWidget* parentWidgetPtr = nullptr); 
        ~EditorWindow(); 
```

The `keyPressEvent` method is called every time the user presses a key, and `closeEvent` is called when the user tries closing the window:

```cpp
      protected: 
        void keyPressEvent(QKeyEvent* eventPtr); 
        void closeEvent(QCloseEvent* eventPtr); 

      private: 
        EditorWidget* m_editorWidgetPtr; 
    }; 

    #endif // EDITORWINDOW_H 
```

The `EditorWindow` class is in fact rather small. It only defines the constructor and the destructor, as well as the `keyPressEvent` and `closePressEvent` methods.

**EditorWindow.cpp:**

```cpp
    #include "EditorWindow.h" 
    #include <QtWidgets> 
```

The constructor sets the size of the window to `1000` * `500` pixels and adds the standard file menu to the menu bar:

```cpp
    EditorWindow::EditorWindow(QWidget* parentWidgetPtr /*= nullptr*/) 
     :MainWindow(parentWidgetPtr) { 
      resize(1000, 500); 
      m_editorWidgetPtr = new EditorWidget(this); 
      setCentralWidget(m_editorWidgetPtr); 
      addFileMenu(); 
    } 

    EditorWindow::~EditorWindow() { 
      // Empty. 
    } 
```

The `keyPressEvent` and `closeEvent` methods just pass the message to their counterpart methods in the editor widget, which is located at the center of the window:

```cpp
    void EditorWindow::keyPressEvent(QKeyEvent* eventPtr) { 
      m_editorWidgetPtr->keyPressEvent(eventPtr); 
    } 

    void EditorWindow::closeEvent(QCloseEvent* eventPtr) { 
      m_editorWidgetPtr->closeEvent(eventPtr); 
    }
```

# Drawing the editor widget

The `EditorWidget` class is a sub class of `DocumentWidget` of the previous section. It catches the key, mouse, resizing, and closing events. It also overrides the methods for saving and loading documents.

**EditorWidget.h:**

```cpp
    #ifndef EDITORWIDGET_H 
    #define EDITORWIDGET_H 

    #include <QWidget> 
    #include <QMap> 
    #include <QMenu> 
    #include <QToolBar> 
    #include <QPair> 
    #include "Caret.h" 

    #include "..\MainWindow\DocumentWidget.h" 

    class EditorWidget : public DocumentWidget { 
      Q_OBJECT 

      public: 
        EditorWidget(QWidget* parentWidgetPtr); 
```

The `keyPressEvent` is called when the user presses a key, and `mousePressEvent` is called when the user clicks with the mouse:

```cpp
        void keyPressEvent(QKeyEvent* eventPtr); 
        void mousePressEvent(QMouseEvent* eventPtr); 
```

The `mouseToIndex` method is an auxiliary method that calculates the index of the character the user clicks at with the mouse:

```cpp
      private: 
        int mouseToIndex(QPoint point); 
```

The `paintEvent` method is called when the window needs to be repainted, and `resizeEvent` is called when the user resizes the window. We catch the resize event in this application because we want to recalculate the number of characters that fits on each line:

```cpp
      public: 
        void paintEvent(QPaintEvent* eventPtr); 
        void resizeEvent(QResizeEvent* eventPtr);
```

Similar to the drawing program in the previous section, `newDocument` is called when the user selects the New menu item, `writeFile` is called when the user selects the save or save as items, and `readFile` is called when the user selects the open item:

```cpp
      private: 
        void newDocument(void); 
        bool writeFile(const QString& filePath); 
        bool readFile(const QString& filePath); 
```

The `setCaret` method is called to set the caret as a response to user input or a mouse click:

```cpp
      private: 
        void setCaret(); 
```

When the user moves the caret up or down, we need to find the index of character over or under the caret. The easiest way to do that is to simulate a mouse click:

```cpp
        void simulateMouseClick(int x, int y); 
```

The `calculate` method is an auxiliary method that calculates the number of lines, and the position of each character on each line:

```cpp
      private: 
        void calculate(); 
```

The `m_editIndex` field holds the index of the position for the user to input text. That position is also where the caret is visible:

```cpp
        int m_editIndex = 0; 
```

The `m_caret` field holds the caret of the application:

```cpp
        Caret m_caret; 
```

The text of the editor is stored in `m_editorText`:

```cpp
        QString m_editorText; 
```

The text of the editor may be distributed over several lines; `m_lineList` keeps track of the first and last index of each line:

```cpp
        QList<QPair<int,int>> m_lineList; 
```

The preceding `calculate` method calculates the rectangle of each character in the editor text, and places them in `m_rectList`:

```cpp
        QList<QRect> m_rectList;
```

In the application of this chapter, all characters hold the same font, which is stored in `TextFont`:

```cpp
        static const QFont TextFont; 
```

`FontWidth` and `FontHeight` hold the width and height of a character in `TextFont`:

```cpp
         int FontWidth, FontHeight; 
    }; 

    #endif // EDITORWIDGET_H 
```

The `EditorWidget` class is rather large. It defines the functionality of the editor.

**EditorWidget.cpp:**

```cpp
    #include "EditorWidget.h" 
    #include <QtWidgets> 
    using namespace std; 
```

We initialize the text font to 12-point `Courier New`:

```cpp
    const QFont EditorWidget::TextFont("Courier New", 12); 
```

The constructor sets the title to `Editor` and the file suffix for the standard Load and Save dialogs to `edi`. The height and average width, in pixels, of a character in the text font are set with the Qt `QMetrics` class. The rectangle of each character is calculated, and the caret is set to the first character in the text:

```cpp
    EditorWidget::EditorWidget(QWidget* parentWidgetPtr) 
     :DocumentWidget(tr("Editor"), tr("Editor files (*.edi)"), 
                     parentWidgetPtr), 
      m_caret(this), 
      m_editorText(tr("Hello World")) { 
      QFontMetrics metrics(TextFont);
      FontHeight = metrics.height();
      FontWidth = metrics.averageCharWidth();
      calculate(); 
      setCaret(); 
      m_caret.show(); 
    } 
```

The `newDocument` method is called when the user selects the new menu item. It clears the text, sets the caret, and recalculates the character rectangles:

```cpp
    void EditorWidget::newDocument(void) { 
      m_editIndex = 0; 
      m_editorText.clear(); 
      calculate(); 
      setCaret(); 
    } 
```

The `writeFile` method is called when the user selects the save or save as menu items. It simply writes the current text of the editor:

```cpp
    bool EditorWidget::writeFile(const QString& filePath) { 
      QFile file(filePath); 
      if (file.open(QIODevice::WriteOnly | QIODevice::Text)) { 
        QTextStream outStream(&file); 
        outStream << m_editorText; 
```

We use the `Ok` field of the input stream to decide if the writing was successful:

```cpp
        return ((bool) outStream.Ok); 
      } 
```

If it was not possible to open the file for writing, `false` is returned:

```cpp
      return false; 
    } 
```

The `readFile` method is called when the user selects the load menu item. It reads all the text of the editor by calling `readAll` on the input stream:

```cpp
    bool EditorWidget::readFile(const QString& filePath) { 
      QFile file(filePath); 

      if (file.open(QIODevice::ReadOnly | QIODevice::Text)) { 
        QTextStream inStream(&file); 
        m_editorText = inStream.readAll(); 
```

When the text has been read, the character rectangles are calculated, and the caret is set:

```cpp
        calculate(); 
        setCaret(); 
```

We use the `Ok` field of the input stream to decide if the reading was successful:

```cpp
        return ((bool) inStream.Ok); 
      } 
```

If it was not possible to open the file for reading, `false` is returned:

```cpp
      return false; 
    }
```

The `mousePressEvent` is called when the user presses one of the mouse buttons. If the user presses the left button, we call `mouseToIndex` to calculate the index of the character clicked at, and set the caret to that index:

```cpp
    void EditorWidget::mousePressEvent(QMouseEvent* eventPtr) { 
      if (eventPtr->buttons() == Qt::LeftButton) { 
        m_editIndex = mouseToIndex(eventPtr->pos()); 
        setCaret(); 
      } 
    } 
```

The `keyPressEvent` is called when the user presses a key. First, we check if it is an arrow key, the delete, backspace, or return key. If it is not, we insert the character at the position indicated by the caret:

```cpp
    void EditorWidget::keyPressEvent(QKeyEvent* eventPtr) { 
      switch (eventPtr->key()) { 
```

If the key is the left-arrow key, and if the edit caret is not already located at the beginning of the text, we decrease the edit index:

```cpp
        case Qt::Key_Left: 
          if (m_editIndex > 0) { 
            --m_editIndex; 
          } 
          break; 
```

If the key is the right-arrow key, and if the edit caret is not already located at the end of the text, we increase the edit index:

```cpp
        case Qt::Key_Right: 
          if (m_editIndex < m_editorText.size()) { 
            ++m_editIndex; 
          } 
          break; 
```

If the key is the up-arrow key, and if the edit caret is not already located at the top of the editor, we call `similateMouseClick` to simulate that the user clicks with the mouse at a point slightly over the current index. In that way, the new edit index will at the line over the current line:

```cpp
        case Qt::Key_Up: { 
            QRect charRect = m_rectList[m_editIndex]; 

            if (charRect.top() > 0) { 
              int x = charRect.left() + (charRect.width() / 2), 
                  y = charRect.top() - 1; 
              simulateMouseClick(x, y); 
            } 
          } 
          break; 
```

If the key is the down-arrow key, we call `similateMouseClick` to simulate that the user clicks with the mouse at a point slightly under the current index. In that way, we the edit carat will be located at the character directly beneath the current character. Note that if the index is already at the bottom line, nothing happens:

```cpp
        case Qt::Key_Down: { 
            QRect charRect = m_rectList[m_editIndex]; 
            int x = charRect.left() + (charRect.width() / 2), 
                y = charRect.bottom() + 1; 
            simulateMouseClick(x, y); 
          } 
          break; 
```

If the user presses the delete key, and the edit index is not already beyond the end of the text, the current character is removed:

```cpp
        case Qt::Key_Delete: 
          if (m_editIndex < m_editorText.size()) { 
            m_editorText.remove(m_editIndex, 1); 
            setModifiedFlag(true); 
          } 
          break; 
```

If the user presses the backspace key, and the edit index is not already at the beginning of the text, the character before the current character is removed:

```cpp
        case Qt::Key_Backspace: 
          if (m_editIndex > 0) { 
            m_editorText.remove(--m_editIndex, 1); 
            setModifiedFlag(true); 
          } 
          break; 
```

If the user presses the return key, the newline character (`n`) is inserted:

```cpp
        case Qt::Key_Return: 
          m_editorText.insert(m_editIndex++, 'n'); 
          setModifiedFlag(true); 
          break;
```

If the user presses a readable character, it is given by the `text` method, and we insert its first character at the edit index:

```cpp
        default: { 
            QString text = eventPtr->text(); 

            if (!text.isEmpty()) { 
              m_editorText.insert(m_editIndex++, text[0]); 
              setModifiedFlag(true); 
            } 
          } 
          break; 
      }  
```

When the text has been modified, we need to calculate the character rectangles, set the caret, and force a repaint by calling `update`:

```cpp
      calculate(); 
      setCaret(); 
      update(); 
    } 
```

The `similateMouseClick` method simulates a mouse click by calling `mousePressEvent` and `mousePressRelease` with the given point:

```cpp
    void EditorWidget::simulateMouseClick(int x, int y) { 
      QMouseEvent pressEvent(QEvent::MouseButtonPress, QPointF(x, y), 
                       Qt::LeftButton, Qt::NoButton, Qt::NoModifier); 
      mousePressEvent(&pressEvent); 
      QMouseEvent releaseEvent(QEvent::MouseButtonRelease, 
                               QPointF(x, y), Qt::LeftButton, 
                               Qt::NoButton, Qt::NoModifier); 
      mousePressEvent(&releaseEvent); 
    } 
```

The `setCaret` method creates a rectangle holding the size and position of the caret, and then hides, sets, and shows the caret:

```cpp
    void EditorWidget::setCaret() { 
      QRect charRect = m_rectList[m_editIndex]; 
      QRect caretRect(charRect.left(), charRect.top(), 
                      1, charRect.height()); 
      m_caret.hide(); 
      m_caret.set(caretRect); 
      m_caret.show(); 
    }
```

The `mouseToIndex` method calculates the edit index of the given mouse point:

```cpp
    int EditorWidget::mouseToIndex(QPoint mousePoint) { 
      int x = mousePoint.x(), y = mousePoint.y(); 
```

First, we set the `y` coordinate to the text, in case it is below the text:

```cpp
      if (y > (FontHeight * m_lineList.size())) { 
        y = ((FontHeight * m_lineList.size()) - 1); 
      } 
```

We calculate the line of the mouse point:

```cpp
      int lineIndex = y / FontHeight; 
      QPair<int,int> lineInfo = m_lineList[lineIndex]; 
      int firstIndex = lineInfo.first, lastIndex = lineInfo.second; 
```

We find the index on that line:

```cpp
      if (x > ((lastIndex - firstIndex + 1) * FontWidth)) { 
        return (lineIndex == (m_lineList.size() - 1)) 
               ? (lineInfo.second + 1) : lineInfo.second; 
      } 
      else { 
        return firstIndex + (x / FontWidth); 
      } 

      return 0; 
    } 
```

The `resizeEvent` method is called when the user changes the size of the window. The character rectangles are recalculated since the lines may be shorter or longer:

```cpp
    void EditorWidget::resizeEvent(QResizeEvent* eventPtr) { 
      calculate(); 
      DocumentWidget::resizeEvent(eventPtr); 
    } 
```

The `calculate` method is called every time there has been a change in the text or when the window size has been changed. It iterates through the text and calculates the rectangle for each character:

```cpp
    void EditorWidget::calculate() { 
      m_lineList.clear(); 
      m_rectList.clear(); 
      int windowWidth = width();
```

First, we need to divide the text into lines. Each line continues until it does not fit in the window, until we reach a new line, or until the text ends:

```cpp
      { int firstIndex = 0, lineWidth = 0; 
        for (int charIndex = 0; charIndex < m_editorText.size(); 
             ++charIndex) { 
          QChar c = m_editorText[charIndex]; 

          if (c == 'n') { 
            m_lineList.push_back 
                       (QPair<int,int>(firstIndex, charIndex)); 
            firstIndex = charIndex + 1; 
            lineWidth = 0; 
          } 
          else { 
            if ((lineWidth + FontWidth) > windowWidth) { 
              if (firstIndex == charIndex) { 
                m_lineList.push_back 
                           (QPair<int,int>(firstIndex, charIndex)); 
                firstIndex = charIndex + 1; 
              } 
              else { 
                m_lineList.push_back(QPair<int,int>(firstIndex, 
                                                    charIndex - 1)); 
                firstIndex = charIndex; 
              } 

              lineWidth = 0; 
            } 
            else { 
              lineWidth += FontWidth; 
            } 
          } 
        } 

        m_lineList.push_back(QPair<int,int>(firstIndex, 
                                            m_editorText.size() - 1)); 
      } 
```

We then iterate through the lines and, for each line, calculate the rectangle of each character:

```cpp
      { int top = 0; 
        for (int lineIndex = 0; lineIndex < m_lineList.size(); 
             ++lineIndex) { 
          QPair<int,int> lineInfo = m_lineList[lineIndex]; 
          int firstIndex = lineInfo.first, 
              lastIndex = lineInfo.second, left = 0; 

          for (int charIndex = firstIndex; 
               charIndex <= lastIndex; ++charIndex){ 
            QRect charRect(left, top, FontWidth, FontHeight); 
            m_rectList.push_back(charRect); 
            left += FontWidth; 
          } 

          if (lastIndex == (m_editorText.size() - 1)) { 
            QRect lastRect(left, top, 1, FontHeight); 
            m_rectList.push_back(lastRect); 
          } 

          top += FontHeight; 
        } 
      } 
    } 
```

The `paintEvent` method is called when the window needs to be repainted:

```cpp
    void EditorWidget::paintEvent(QPaintEvent* /*eventPtr*/) { 
      QPainter painter(this); 
      painter.setRenderHint(QPainter::Antialiasing); 
      painter.setRenderHint(QPainter::TextAntialiasing); 
      painter.setFont(TextFont); 
      painter.setPen(Qt::black); 
      painter.setBrush(Qt::white); 
```

We iterate through the text of the editor and, for each character except the new line, we write in its appropriate position:

```cpp
      for (int index = 0; index < m_editorText.length(); ++index) { 
        QChar c = m_editorText[index]; 

        if (c != 'n') { 
          QRect rect = m_rectList[index]; 
          painter.drawText(rect, c); 
        } 
      } 

      m_caret.paint(painter); 
    }
```

# The main function

Finally, the `main` function works in a way similar to the previous applications of this chapter—we create an application, create an editor window, and execute the application.

**Main.cpp:**

```cpp
#include "EditorWindow.h" 
#include <QApplication> 

int main(int argc, char *argv[]) { 
  QApplication application(argc, argv); 
  EditorWindow editorWindow; 
  editorWindow.show(); 
  return application.exec(); 
} 
```

The following output is obtained:

![](img/a562a9ab-9bc0-4cbc-99f8-d3d7382a9605.png)

# Summary

In this chapter, we have developed three graphical applications with the Qt library—an analog clock, a drawing program, and an editor. The clock shows the current hour, minute, and second. In the drawing program we can draw lines, rectangles, and ellipses, and in the editor, we can input and edit text.

In the next chapter, we will continue to work with the applications, and develop more advanced versions.