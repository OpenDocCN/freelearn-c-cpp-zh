# Scripting

In this chapter, you will learn how to bring scripting facilities to your programs. You will gain knowledge of how to use JavaScript to implement the logic and details of your game, without having to rebuild the main game engine. These skills will also be useful in the last part of the book when we work with Qt Quick. Although the environment we will focus on blends best with Qt applications, there are other options if you don't like JavaScript. We will also show how you can use Python to make your games scriptable.

The main topics covered in this chapter are as listed:

*   Executing JavaScript code
*   Interaction between C++ and JavaScript
*   Implementing a scripting game
*   Integrating the Python interpreter

# Why script?

You might ask yourself, "why should I use any scripting language if I can implement everything I need in C++"? There are a number of benefits to providing a scripting environment to your games. Most modern games really consist of two parts. One is the main game engine that implements the core of the game (data structures, processing algorithms, and the rendering layer) and exposes an API to the other component, which provides details, behavior patterns, and action flows for the game. This other component is sometimes written in a scripting language. The main benefit of this is that story designers can work independently from the engine developers, and they don't have to rebuild the whole game just to modify some of its parameters or check whether the new quest fits well into the existing story. This makes the development much quicker compared to the monolithic approach.

Another benefit is that this development opens the game to modding—skilled end users can extend or modify the game to provide some added value to the game. It's also a way to implement extensions of the game on top of the existing scripting API without having to redeploy the complete game binary to every player. Finally, you can reuse the same game driver for other games and just replace the scripts to obtain a totally different product.

In this chapter, we will use the Qt QML module to implement scripting. This module implements QML language used in Qt Quick. Since QML is JavaScript-based, Qt QML includes a JavaScript engine and provides API for running JavaScript code. It also allows you to expose C++ objects to JavaScript and vice versa.

We will not discuss the details of the JavaScript language itself, as there are many good books and websites available where you can learn JavaScript. Besides, the JavaScript syntax is very similar to that of C, and you shouldn't have any problems understanding the scripts that we use in this chapter even if you haven't seen any JavaScript code before.

# Evaluating JavaScript expressions

To use Qt QML in your programs, you have to enable the script module for your projects by adding the `QT += qml` line to the project file.

C++ compilers do not understand JavaScript. Therefore, to execute any script, you need to have a running interpreter that will parse the script and evaluate it. In Qt, this is done with the `QJSEngine` class. This is a JavaScript runtime that handles the execution of script code and manages all the resources related to scripts. It provides the `evaluate()` method, which can be used to execute JavaScript expressions. Let's look at a "Hello World" program using `QJSEngine`:

```cpp
#include <QCoreApplication>
#include <QJSEngine>

int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);
    QJSEngine engine;
    engine.installExtensions(QJSEngine::ConsoleExtension);
    engine.evaluate("console.log('Hello World!');");
    return 0;
} 
```

This program is very simple. First, it creates an application object that is required for the script environment to function properly and instantiates a `QJSEngine` object. Next, we ask `QJSEngine` to install the console extension—the global `console` object that can be used to print messages to the console. It's not part of the ECMAScript standard, so it's not available by default, but we can easily enable it using the `installExtensions()` function. Finally, we call the `evaluate()` function to execute the script source given to it as a parameter. After building and running the program, you will see a well-known `Hello World!` printed to the console with the `js:` prefix.

By default, `QJSEngine` provides built-in objects defined by ECMA-262 standard, including `Math`, `Date`, and `String`. For example, a script can use `Math.abs(x)` to get the absolute value of a number.

If you don't get any output, it probably means that the script didn't get executed properly, possibly because of an error in the script's source code. To verify that, we can check the value returned from `evaluate()`:

```cpp
QJSValue result = engine.evaluate("console.log('Hello World!')");
if (result.isError()) {
    qDebug() << "JS error:" << result.toString();
}
```

This code checks whether there is an exception or a syntax error and if yes, it displays the corresponding error message. For example, if you omit the closing single quote in the script source text and run the program, the following message will be displayed:

```cpp
JS error: "SyntaxError: Expected token `)'"
```

You can see that `evaluate()` returns a `QJSValue`. This is a special type that is used to exchange data between the JavaScript engine and the C++ world. Like `QVariant`, it can hold a number of primitive types (`boolean`, `integer`, `string`, and so on). However, it is in fact much more powerful, because it can hold a reference to a JavaScript object or function that lives in the JavaScript engine. Copying a `QJSValue` will produce another object that references the same JavaScript object. You can use the member functions of `QJSValue` to interact with the objects from C++. For example, you can use `property()` and `setProperty()` to manipulate the object's properties and `call()` to call the function and get the returned value as another `QJSValue`.

In the previous example, `QJSEngine::evaluate()` returned an `Error` object. When the JavaScript code runs successfully, you can use the returned value later in your C++ code. For example, the script can calculate the amount of damage done to a creature when it is hit with a particular weapon. Modifying our code to use the result of the script is very simple. All that is required is to store the value returned by `evaluate()` and then it can be used elsewhere in the code:

```cpp
QJSValue result = engine.evaluate("(7 + 8) / 2");
if (result.isError()) {
    //...
} else {
    qDebug() << result.toNumber();
}
```

# Time for action – Creating a JavaScript editor

Let's do a simple exercise and create a graphical editor to write and execute scripts. Start by creating a new Qt Widgets project and implement a main window composed of two plain text edit widgets (`ui->codeEditor` and `ui->logWindow`) that are separated using a vertical splitter. One of the edit boxes will be used as an editor to input code and the other will be used as a console to display script results. Then, add a menu and toolbar to the window and create actions to open (`ui->actionOpenDocument`) and save (`ui->actionSaveDocument` and `ui->actionSaveDocumentAs`) the document, create a new document (`ui->actionNewDocument`), execute the script (`ui->actionExecuteScript`), and to quit the application (`ui->actionQuit`). Remember to add them to the menu and toolbar.

As a result, you should receive a window similar to the one shown in the following screenshot:

![](img/bbc56b53-86e7-40f9-b8f1-19528ce6d0a9.png)

Connect the quit action to the `QApplication::quit()` slot. Then, create an `openDocument()` slot and connect it to the `triggered` signal of the appropriate action. In the slot, use `QFileDialog::getOpenFileName()` to ask the user for a document path, as follows:

```cpp
void MainWindow::openDocument()
{
    QString filePath = QFileDialog::getOpenFileName(
        this, tr("Open Document"),
        QDir::homePath(), tr("JavaScript Documents (*.js)"));
    if(filePath.isEmpty()) {
        return;
    }
    open(filePath);
}
```

In a similar fashion, implement the New, Save, and Save As action handlers. Lastly, create the `open(const QString &filePath)` slot that should read the document and put its contents into the code editor:

```cpp
void MainWindow::open(const QString &filePath)
{
    QFile file(filePath);
    if(!file.open(QFile::ReadOnly | QFile::Text)) {
        QMessageBox::critical(this, tr("Error"), tr("Can't open file."));
        return;
    }
    setWindowFilePath(filePath);
    ui->codeEditor->setPlainText(QString::fromUtf8(file.readAll()));
    ui->logWindow->clear();
}
```

The `windowFilePath` property of `QWidget` can be used to associate a file with a window. When this property is set, Qt will automatically adjust the window title and even add a proxy icon on macOS, allowing convenient access to the file. You can then use this property in actions related to using the file—when saving a document, you can check whether this property is empty and ask the user to provide a filename. Then, you can reset this property when creating a new document or when the user provides a new path for the document.

At this point, you should be able to run the program and use it to create scripts and save and reload them in the editor.

Now, to execute the scripts, add a `QJSEngine m_engine` member variable to the window class. Create a new slot, call it `run`, and connect it to the execute action. Put the following code in the body of the slot:

```cpp
void MainWindow::run()
{
    ui->logWindow->clear();
    QTextCursor logCursor = ui->logWindow->textCursor();
    QString scriptSourceCode = ui->codeEditor->toPlainText();
    QJSValue result = m_engine.evaluate(scriptSourceCode, 
                                        windowFilePath());
    if(result.isError()) {
        QTextCharFormat errFormat;
        errFormat.setForeground(Qt::red);
        logCursor.insertText(tr("Exception at line %1:\n")
            .arg(result.property("lineNumber").toInt()), errFormat);
        logCursor.insertText(result.toString(), errFormat);
        logCursor.insertBlock();
        logCursor.insertText(result.property("stack").toString(), 
                             errFormat);
    } else {
        QTextCharFormat resultFormat;
        resultFormat.setForeground(Qt::blue);
        logCursor.insertText(result.toString(), resultFormat);
    }
}
```

Build and run the program. To do so, enter the following script in the editor:

```cpp
function factorial(n) {
    if(n < 0) {
        return;
    }
    if(n == 0) {
        return 1;
    }
    return n * factorial(n - 1);
}

factorial(7)
```

Save the script in a file called `factorial.js` and then run it. You should get an output as shown:

![](img/fa9c7836-4685-414d-94e9-9cfadb238755.png)

Next, replace the script with the following one:

```cpp
function factorial(n) {
    return N;
}

factorial(7) 
```

Running the script should yield the following result:

![](img/6785935d-8c68-4c9c-8aaa-77fa656db629.png)

# What just happened?

The `run()` method clears the log window and evaluates the script using the method that we learned earlier in this chapter. If the evaluation is successful, it prints the result in the log window, which is what we see in the first screenshot shown in the previous section.

In the second attempt, we made an error in the script using a nonexistent variable. Evaluating such code results in an exception. In addition to reporting the actual error, we also use the `lineNumber` property of the returned `Error` object to report the line that caused the problem. Next, we display the `stack` property of the error object, which returns the backtrace (a stack of function calls) of the problem, which we also print on the log.

# Global object state

Let's try another script. The following code defines the `fun` local variable, which is assigned an anonymous function that returns a number:

```cpp
var fun = function() {
    return 42;
}
```

You can then call `fun()` like a regular function, as follows:

![](img/c4ec89e0-cd77-4c85-8a77-49db05956220.png)

Now, let's look at what happens if we delete the definition of `fun` from the script, but still keep the invocation:

![](img/7d8c75cc-760c-4cfd-a217-006cdb4f3e46.png)

We still get the same result even though we didn't define what `fun` means! This is because any variables at the top scope become properties of the global object. The state of the global object is preserved during the existence of `QJSEngine`, so the `fun` variable will remain available until it's overwritten or the engine is destroyed.

To prevent users from accidentally changing the global object with local variables, we can wrap the provided code in an anonymous function:

```cpp
QString wrappedCode =
    QStringLiteral("(function() { %1\n})()").arg(scriptSourceCode);
QJSValue result = m_engine.evaluate(wrappedCode, windowFilePath());
```

In this case, the JavaScript code must use the `return` statement to actually return a value to the editor:

```cpp
var fun = function() {
    return 42;
}
return fun();
```

Removing the `fun` variable initialization will now result in an error:

```cpp
ReferenceError: fun is not defined
```

However, removing the `var` keyword will make the variable global and preserved. A malicious user can also break the existing global object's properties. For example, evaluating `Math.floor = null;` will make the built-in `Math.floor` function unavailable in all subsequent calls.

There isn't really a good way to guard or reset the global object. If you are concerned about malicious scripts, destroying and creating a new `QJSEngine` object is the best option. If you need to run multiple scripts that are not allowed to interfere with each other, you have to create a separate `QJSEngine` for each of them. However, in most applications, such sandboxing seems to be an overkill.

# Exposing C++ objects and functions to JavaScript code

So far, we were only evaluating some standalone scripts that can make use of built-in JavaScript features. Now, it is time to learn to use data from your programs in the scripts. This is done by exposing different kinds of entities to and from scripts.

# Accessing C++ object's properties and methods

The simplest way to expose a C++ object to JavaScript code is to take advantage of Qt's meta-object system. `QJSEngine` is able to inspect `QObject` instances and detect their properties and methods. To use them in scripts, the object has to be visible to the script. The easiest way to make this happen is to add it to the engine's global object. As you remember, all data between the script engine and C++ is exchanged using the `QJSValue` class, so first we have to obtain a JS value handle for the C++ object:

```cpp
QJSEngine engine;
QPushButton *button = new QPushButton("Button");
// ...
QJSValue scriptButton = engine.newQObject(button);
engine.globalObject().setProperty("pushButton", scriptButton); 
```

`QJSEngine::newQObject()` creates a JavaScript object wrapping an existing `QObject` instance. We then set the wrapper as a property of the global object called `pushButton`. This makes the button available in the global context of the engine as a JavaScript object. All the properties defined with `Q_PROPERTY` are available as properties of the object, and every slot is accessible as a method of that object. In JavaScript, you will be able to use the `pushButton` object like this:

```cpp
pushButton.text = 'My Scripted Button';
pushButton.checkable = true;
pushButton.setChecked(true);
pushButton.show();
```

Qt slots conventionally return `void`. They technically can have any return type, but Qt won't use the return value, so in most cases, there is no sense in returning any value. On the contrary, when you expose a C++ method to the JavaScript engine, you often want to return a value and receive it in JavaScript. In these cases, you should not create slots, as that will break the convention. You should make the method invokable instead. To do this, place the method declaration in a regular `public` scope and add `Q_INVOKABLE` before it:

```cpp
public:
    Q_INVOKABLE int myMethod();
```

This macro instructs **moc** to make this method invokable in the meta-object system so that Qt will be able to call it at runtime. All invokable methods are automatically exposed to scripts.

# Data type conversions between C++ and JavaScript

Qt will automatically convert arguments and return types of methods to its JavaScript counterparts. The supported conversions include the following:

*   Basic types (`bool`, `int`, `double`, and such) are exposed without changes
*   Qt data types (`QString`, `QUrl`, `QColor`, `QFont`, `QDate`, `QPoint`, `QSize`, `QRect`, `QMatrix4x4`, `QQuaternion`, `QVector2D`, and such) are converted to objects with the available properties
*   `QDateTime` and `QTime` values are automatically converted to JavaScript `Date` objects
*   Enums declared with `Q_ENUM` macro can be used in JavaScript
*   Flags declared with `Q_FLAG` macro can be used as flags in JavaScript
*   `QObject*` pointers will be automatically converted to JavaScript wrapper objects
*   `QVariant` objects containing any supported types are recognized
*   `QVariantList` is an equivalent of a JavaScript array with arbitrary items
*   `QVariantMap` is an equivalent of a JavaScript object with arbitrary properties
*   Some C++ list types (`QList<int>`, `QList<qreal>`, `QList<bool>`, `QList<QString>`, `QStringList`, `QList<QUrl>`, `QVector<int>`, `QVector<qreal>`, and `QVector<bool>`) are exposed to JavaScript without performing additional data conversions

If you want more fine-grained control over data type conversions, you can simply use `QJSValue` as an argument type or a return type. For example, this will allow you to return a reference to an existing JavaScript object instead of creating a new one each time. This approach is also useful for creating or accessing multidimensional arrays or other objects with complex structure. While you can use nested `QVariantList` or `QVariantMap` objects, creating `QJSValue` objects directly may be more efficient.

Qt will not be able to recognize and automatically convert a custom type. Attempting to access such method or property from JavaScript will result in an error. You can use the `Q_GADGET` macro to make a C++ data type available to JavaScript and use `Q_PROPERTY` to declare properties that should be exposed.

For more information on this topic, refer to the Data Type Conversion Between QML and C++ documentation page.

# Accessing signals and slots in scripts

`QJSEngine` also offers the capability to use signals and slots. The slot can be either a C++ method or a JavaScript function. The connection can be made either in C++ or in the script.

First, let's see how to establish a connection within a script. When a `QObject` instance is exposed to a script, the object's signals become the properties of the wrapping object. These properties have a `connect` method that accepts a function object that is to be called when the signal is emitted. The receiver can be a regular slot or a JavaScript function. The most common case is when you connect the signal to an anonymous function:

```cpp
pushButton.toggled.connect(function() {
    console.log('button toggled!');
});
```

If you need to undo the connection, you will need to store the function in a variable:

```cpp
function buttonToggled() {
    //...
}
pushButton.toggled.connect(buttonToggled);
//...
pushButton.toggled.disconnect(buttonToggled);
```

You can define the `this` object for the function by providing an extra argument to `connect()`:

```cpp
var obj = { 'name': 'FooBar' };
pushButton.clicked.connect(obj, function() {
    console.log(this.name);
});
```

You can also connect the signal to a signal or slot of another exposed object. To connect the `clicked()` signal of an object called `pushButton` to a `clear()` slot of another object called `lineEdit`, you can use the following statement:

```cpp
pushButton.clicked.connect(lineEdit.clear);
```

Emitting signals from within the script is also easy—just call the signal as a function and pass to it any necessary parameters:

```cpp
pushButton.clicked();
spinBox.valueChanged(7);
```

To create a signal-slot connection on the C++ side where the receiver is a JavaScript function, you can utilize C++ lambda functions and the `QJSValue::call()` function:

```cpp
QJSValue func = engine.evaluate(
    "function(checked) { console.log('func', checked); }");
QObject::connect(&button, &QPushButton::clicked, [func](bool checked) {
    QJSValue(func).call({ checked });
});
```

# Time for action – Using a button from JavaScript

Let's put all this together and build a complete example of a scriptable button:

```cpp
int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    QJSEngine engine;
    engine.installExtensions(QJSEngine::ConsoleExtension);
    QPushButton button;
    engine.globalObject().setProperty("pushButton", engine.newQObject(&button));
    QString script =
        "pushButton.text = 'My Scripted Button';\n"
        "pushButton.checkable = true;\n"
        "pushButton.setChecked(true);\n"
        "pushButton.toggled.connect(function(checked) {\n"
        "  console.log('button toggled!', checked);\n"
        "});\n"
        "pushButton.show();";
    engine.evaluate(script);

    QJSValue func = engine.evaluate(
          "function(checked) { console.log('button toggled 2!', checked); }");
    QObject::connect(&button, &QPushButton::clicked, [func](bool checked) {
        QJSValue(func).call({ checked });
    });
    return app.exec();
}
```

In this code, we expose the function to JavaScript and execute code that sets some properties of the button and accesses its `toggled` signal. Next, we create a JavaScript function, store a reference to it in the `func` variable, and connect the `toggled` signal of the button to this function from C++ side.

# Restricting access to C++ classes from JavaScript

There are cases when you want to provide a rich interface for a class to manipulate it from within C++ easily, but to have strict control over what can be done using scripting, you want to prevent scripters from using some of the properties or methods of the class.

The safest approach is to create a wrapper class that only exposes the allowed methods and signals. This will allow you to design your original classes freely. For example, if you want to hide some methods, it's quite easy—just don't make them slots and don't declare them with `Q_INVOKABLE`. However, you may want them to be slots in the internal implementation. By creating a wrapper class, you can easily hide slots of the internal class from the JavaScript code. We'll show how to apply this approach later in this chapter.

Another issue may arise if the data types used by your internal object cannot be directly exposed to JavaScript. For example, if one of your methods returns a `QVector<QVector<int>>`, you will not be able to call such a method directly from JavaScript. The wrapper class is a good place to put the required data conversion operations.

You should also be aware that JavaScript code can emit any signals of exposed C++ objects. In some cases, this can break the logic of your application. If you're using a wrapper, you can just connect the signal of the internal class to the signal of the exposed wrapper. The script will be able to connect to the wrapper's signal, but it won't be able to emit the original signal. However, the script will be able to emit the wrapper's signal, and this can affect all the other JavaScript code in the engine.

If all or almost all APIs of the class are safe to expose to JavaScript, it's much easier to make the objects themselves available, instead of creating wrappers. If you want to restrict access to certain methods, keep in mind that JavaScript code can only access public and protected methods declared with `Q_INVOKABLE` and slots. Remember that you can still connect signals to non-slot methods if you use the `connect()` variant that takes a function pointer as an argument. JavaScript code also cannot access any private methods.

For properties, you can mark them inaccessible from scripts using the `SCRIPTABLE` keyword in the `Q_PROPERTY` declaration. By default, all properties are scriptable, but you can forbid their exposure to scripts by setting `SCRIPTABLE` to `false`, as shown in the following example:

```cpp
Q_PROPERTY(QString internalName READ internalName SCRIPTABLE false) 
```

# Creating C++ objects from JavaScript

We've only exposed the existing C++ objects to JavaScript so far, but what if you want to create a new C++ object from JavaScript? You can do this using what you already know. A C++ method of an already exposed object can create a new object for you:

```cpp
public:
    Q_INVOKABLE QObject* createMyObject(int argument) {
        return new MyObject(argument);
    }
```

We use `QObject*` instead of `MyObject*` in the function signature. This allows us to import the object into the JS engine automatically. The engine will take ownership of the object and delete it when there are no more references to it in JavaScript.

Using this method from JavaScript is also pretty straightforward:

```cpp
var newObject = originalObject.createMyObject(42);
newObject.slot1();
```

This approach is fine if you have a good place for the `createMyObject` function. However, sometimes you want to create new objects independently of the existing ones, or you don't have any objects created yet. For these situations, there is a neat way to expose the constructor of the class to the JavaScript engine. First, you need to make your constructor invokable in the class declaration:

```cpp
public:
    Q_INVOKABLE explicit MyObject(int argument, QObject *parent = nullptr);
```

Then, you should use the `newQMetaObject()` function to import the *meta-object* of the class to the engine. You can immediately assign the imported meta-object to a property of the global object:

```cpp
engine.globalObject().setProperty("MyObject",
     engine.newQMetaObject(&MyObject::staticMetaObject));
```

You can now invoke the constructor by calling the exposed object with the `new` keyword:

```cpp
var newObject = new MyObject(42);
newObject.slot1();
```

# Exposing C++ functions to JavaScript

Sometimes you just want to provide a single function instead of an object. Unfortunately, `QJSEngine` only supports functions that belong to `QObject`-derived classes. However, we can hide this implementation detail from the JavaScript side. First, create a subclass of `QObject` and add an invokable member function that proxies the original standalone function:

```cpp
Q_INVOKABLE double factorial(int x) {
    return superFastFactorial(x);
}
```

Next, expose the wrapper object using the `newQObject()` function, as usual. However, instead of assigning this object to a property of the global object, extract the `factorial` property from the object:

```cpp
QJSValue myObjectJS = engine.newQObject(new MyObject());
engine.globalObject().setProperty("factorial",
                                  myObjectJS.property("factorial"));
```

Now, the JavaScript code can access the method as if it were a global function, like `factorial(4)`.

# Creating a JavaScript scripting game

Let's perfect our skills by implementing a game that allows players to use JavaScript. The rules are simple. Each player has a number of entities that move on the board. All entities move in turns; during each turn, the entity can stand still or move to an adjacent tile (cardinally or diagonally). If an entity moves to the tile occupied by another entity, that entity is killed and removed from the board.

At the beginning of the game, all entities are placed randomly on the board. An example of a starting position is displayed on the following image:

![](img/561a69d4-121f-4b71-ba94-27e22c8dcfee.png)

Each player must provide a JavaScript function that receives an entity object and returns its new position. This function will be called when one of the player's entities should move. Additionally, the player may provide an initialization function that will be called at the beginning of the game. The state of the board and entities on it will be exposed through a property of the global JavaScript object.

In our game, the players will compete to create the best survival strategy. Once the game is started, the players have no control over the entities, and the provided JavaScript functions must account for any possible game situation. When only entities of one player remain on the board, that player wins. The rules allow any number of players to participate, although we will only have two players in our example.

# Time for action – Implementing the game engine

We will use the Graphics View framework to implement the board visualization. We will not provide too many details about the implementation, since we focus on scripting in this chapter. The basic skills you learned in [Chapter 4](33efb525-a584-4f9a-afaa-fe389d4a0400.xhtml), *Custom 2D Graphics with Graphics View*, should be enough for you to implement this game. The full code of this example is provided with the book. However, we will highlight the architecture of the project and briefly describe how it works.

The game engine implementation consists of two classes:

*   The `Scene` class (derived from `QGraphicsScene`) manages the graphics scene, creates items, and implements the general game logic
*   The `Entity` class (derived from `QGraphicsEllipseItem`) represents a single game entity on the board

Each `Entity` object is a circle with 0.4 radius and (0, 0) center. It is initialized in the constructor, using the following code:

```cpp
setRect(-0.4, -0.4, 0.8, 0.8);
setPen(Qt::NoPen);
```

We will use the `pos` property (inherited from `QGraphicsItem`) to move the circle on the board. The tiles of the board will have a unit size, so we can just treat `pos` as integer `QPoint` instead of `QPointF` with `double` coordinates. We will zoom in to the graphics view to achieve the desired visible size of the entities.

The `Entity` class has two special properties with getters and setters. The `team` property is the number of the player this entity belongs to. This property also defines the color of the circle:

```cpp
void Entity::setTeam(int team) {
    m_team = team;
    QColor color;
    switch(team) {
    case 0:
        color = Qt::green;
        break;
    case 1:
        color = Qt::red;
        break;
    }
    setBrush(color);
}
```

The `alive` flag indicates whether the entity is still in play. For simplicity, we will not immediately delete the killed entity objects. We will just hide them instead:

```cpp
void Entity::setAlive(bool alive)
{
    m_alive = alive;
    setVisible(alive);
    //...
}
```

Let's turn our attention to the `Scene` class. First, it defines some game configuration options:

*   The `fieldSize` property determines the two-dimensional size of the board
*   The `teamSize` property determines how many entities each player has at the beginning of the game
*   The `stepDuration` property determines the number of milliseconds passed between executing the next round of turns

The setter of the `fieldSize` property adjusts the scene rect so that the graphics view is correctly resized at the beginning of the game:

```cpp
void Scene::setFieldSize(const QSize &fieldSize)
{
    m_fieldSize = fieldSize;
    setSceneRect(-1, -1,
                 m_fieldSize.width() + 2,
                 m_fieldSize.height() + 2);
}
```

The execution of each round of the game will be done in the `step()` function. In the constructor, we initialize a `QTimer` object responsible for calling this function:

```cpp
m_stepTimer = new QTimer(this);
connect(m_stepTimer, &QTimer::timeout,
        this, &Scene::step);
m_stepTimer->setInterval(1000);
```

In the `setStepDuration()` function, we simply change the interval of this timer.

The `QVector<Entity*> m_entities` private field of the `Scene` class will contain all the entities in play. The game is started by calling the `start()` function. Let's take a look at it:

```cpp
void Scene::start() {
    const int TEAM_COUNT = 2;
    for(int i = 0; i < m_teamSize; i++) {
        for(int team = 0; team < TEAM_COUNT; team++) {
            Entity* entity = new Entity(this);
            entity->setTeam(team);
            QPoint pos;
            do {
                pos.setX(qrand() % m_fieldSize.width());
                pos.setY(qrand() % m_fieldSize.height());
            } while(itemAt(pos, QTransform()));
            entity->setPos(pos);
            addItem(entity);
            m_entities << entity;
        }
    }
    //...
    m_stepTimer->start();
}
```

We create the requested number of entities for each team and place them at random locations on the board. If we happen to choose an already occupied place, we go on the next iteration of the `do`-`while` loop and choose another location. Next, we add the new item to the scene and to the `m_entities` vector. Finally, we start our timer so that the `step()` function will be called periodically.

In the `main()` function, we initialize the random number generator because we want to get new random numbers each time:

```cpp
qsrand(QDateTime::currentMSecsSinceEpoch());
```

Then, we create and initialize the `Scene` object, and we create a `QGraphicsView` to display our scene.

The game engine is almost ready. We only need to implement the scripting.

# Time for action – Exposing the game state to the JS engine

Before we move on to executing the players' scripts, we need to create a `QJSEngine` and insert some information into its global object. The scripts will use this information to decide the optimal move.

First, we add the `QJSEngine m_jsEngine` private field to the `Scene` class. Next, we create a new `SceneProxy` class and derive it from `QObject`. This class will expose the permitted API of `Scene` to the scripts. We pass a pointer to the `Scene` object to the constructor of the `SceneProxy` object and store it in a private variable:

```cpp
SceneProxy::SceneProxy(Scene *scene) :
    QObject(scene), m_scene(scene)
{
}
```

Add two invokable methods to the class declaration:

```cpp
Q_INVOKABLE QSize size() const;
Q_INVOKABLE QJSValue entities() const;
```

The implementation of the `size()` function is pretty straightforward:

```cpp
QSize SceneProxy::size() const {
    return m_scene->fieldSize();
}
```

However, the `entities()` function is a bit trickier. We cannot add `Entity` objects to the JS engine because they are not based on `QObject`. Even if we could, we prefer to create a proxy class for entities as well.

Let's do this right now. Create the `EntityProxy` class, derive it from `QObject`, and pass a pointer to the underlying `Entity` object to the constructor, like we did in `SceneProxy`. Declare two invokable functions and a signal in the new class:

```cpp
class EntityProxy : public QObject
{
    Q_OBJECT
public:
    explicit EntityProxy(Entity *entity, QObject *parent = nullptr);
    Q_INVOKABLE int team() const;
    Q_INVOKABLE QPoint pos() const;
    //...
signals:
    void killed();
private:
    Entity *m_entity;
};
```

Implementation of the methods just forward the calls to the underlying `Entity` object:

```cpp
int EntityProxy::team() const
{
    return m_entity->team();
}

QPoint EntityProxy::pos() const
{
    return m_entity->pos().toPoint();
}
```

The `Entity` class will be responsible for creating its own proxy object. Add the following private fields to the `Entity` class:

```cpp
EntityProxy *m_proxy;
QJSValue m_proxyValue;
```

The `m_proxy` field will hold the proxy object. The `m_proxyValue` field will contain the reference to the same object added to the JS engine. Initialize these fields in the constructor:

```cpp
m_proxy = new EntityProxy(this, scene);
m_proxyValue = scene->jsEngine()->newQObject(m_proxy);
```

We modify the `Entity::setAlive()` function to emit the `killed()` signal when the entity is killed:

```cpp
void Entity::setAlive(bool alive)
{
    m_alive = alive;
    setVisible(alive);
    if (!alive) {
        emit m_proxy->killed();
    }
}
```

It's generally considered bad practice to emit signals from outside the class that owns the signal. If the source of the signal is another `QObject`-based class, you should create a separate signal in that class and connect it to the destination signal. In our case, we cannot do that, since `Entity` is not a `QObject`, so we choose to emit the signal directly to avoid further complication.

Create the `proxy()` and `proxyValue()` getters for these fields. We can now return to the `SceneProxy` implementation and use the entity proxy:

```cpp
QJSValue SceneProxy::entities() const
{
    QJSValue list = m_scene->jsEngine()->newArray();
    int arrayIndex = 0;
    for(Entity *entity: m_scene->entities()) {
        if (entity->isAlive()) {
            list.setProperty(arrayIndex, entity->proxyValue());
            arrayIndex++;
        }
    }
    return list;
}
```

# What just happened?

First, we ask the JS engine to create a new JavaScript array object. Then, we iterate over all entities and skip entities that were already killed. We use `QJSValue::setProperty` to add the proxy object of each entity to the array. We need to specify the index of the new array item, so we create the `arrayIndex` counter and increment it after each insertion. Finally, we return the array.

This function completes the `SceneProxy` class implementation. We just need to create a proxy object and add it to the JS engine in the constructor of the `Scene` class:

```cpp
SceneProxy *sceneProxy = new SceneProxy(this);
m_sceneProxyValue = m_jsEngine.newQObject(sceneProxy);
```

# Time for action – Loading scripts provided by users

Each player will provide their own strategy script, so the `Scene` class should have a field for storing all provided scripts:

```cpp
QHash<int, QJSValue> m_teamScripts;
```

Let's provide the `setScript()` function that accepts the player's script and loads it into the JS engine:

```cpp
void Scene::setScript(int team, const QString &script) {
    QJSValue value = m_jsEngine.evaluate(script);
    if (value.isError()) {
        qDebug() << "js error: " << value.toString();
        return;
    }
    if(!value.isObject()) {
        qDebug() << "script must return an object";
        return;
    }
    m_teamScripts[team] = value;
}
```

In this function, we try to evaluate the provided code. If the code returned a JavaScript object, we put it in the `m_teamScripts` hash table. We expect that the provided object has the `step` property containing the function that decides the entity's move. The object may also contain the `init` property that will be executed at the beginning of the game.

In the `main()` function, we load the scripts from the project's resources:

```cpp
scene.setScript(0, loadFile(":/scripts/1.js"));
scene.setScript(1, loadFile(":/scripts/2.js"));
```

The `loadFile()` helper function simply loads the content of the file to a `QString`:

```cpp
QString loadFile(const QString& path) {
    QFile file(path);
    if (!file.open(QFile::ReadOnly)) {
        qDebug() << "failed to open " << path;
        return QString();
    }
    return QString::fromUtf8(file.readAll());
}
```

If you want to allow users to provide their scripts without needing to recompile the project, you can accept the script files from the command-line arguments instead:

```cpp
QStringList arguments = app.arguments();
if (arguments.count() < 3) {
    qDebug() << "usage: " << argv[0] << " path/to/script1.js path/to/script2.js";
    return 1;
}
scene.setScript(0, loadFile(arguments[1]));
scene.setScript(1, loadFile(arguments[2]));
```

To set the command-line arguments for your project, switch to the Projects pane, select Run in the left column and locate the Command line arguments input box. The provided project contains two sample scripts in the `scripts` subdirectory.

# Time for action – Executing the strategy scripts

First, we need to check whether the player provided an `init` function and execute it. We'll do it in the `Scene::start()` function:

```cpp
for(int team = 0; team < TEAM_COUNT; team++) {
    QJSValue script = m_teamScripts.value(team);
    if (script.isUndefined()) {
        continue;
    }
    if (!script.hasProperty("init")) {
        continue;
    }
    m_jsEngine.globalObject().setProperty("field", m_sceneProxyValue);
    QJSValue scriptOutput = script.property("init").call();
    if (scriptOutput.isError()) {
        qDebug() << "script error: " << scriptOutput.toString();
        continue;
    }
}
```

In this code, we use `isUndefined()` to check whether the code was provided and parsed successfully. Next, we use `hasProperty()` to check whether the returned object contains the optional `init` function. If we found it, we execute it using `QJSValue::call()`. We provide some information about the board by assigning our `SceneProxy` instance to the `field` property of the global object.

The most exciting part is the `step()` function that implements the actual game execution. Let's take a look at it:

```cpp
void Scene::step() {
    for(Entity* entity: m_entities) {
        if (!entity->isAlive()) {
            continue;
        }
        QJSValue script = m_teamScripts.value(entity->team());
        if (script.isUndefined()) {
            continue;
        }
        m_jsEngine.globalObject().setProperty("field", m_sceneProxyValue);

        QJSValue scriptOutput =
            script.property("step").call({ entity->proxyValue() });
        //...
    }
}
```

First, we iterate over all entities and skip the killed ones. Next, we use `Entity::team()` to determine which player this entity belongs to. We extract the corresponding strategy script from the `m_teamScripts` field and extract the `step` property from it. Then, we try to call it as a function and pass the current entity's proxy object as an argument. Let's see what we do with the script output:

```cpp
if (scriptOutput.isError()) {
    qDebug() << "script error: " << scriptOutput.toString();
    continue;
}
QJSValue scriptOutputX = scriptOutput.property("x");
QJSValue scriptOutputY = scriptOutput.property("y");
if (!scriptOutputX.isNumber() || !scriptOutputY.isNumber()) {
    qDebug() << "invalid script output: " << scriptOutput.toVariant();
    continue;
}
QPoint pos(scriptOutputX.toInt(), scriptOutputY.toInt());
if (!moveEntity(entity, pos)) {
    qDebug() << "invalid move";
}
```

We try to interpret the function's return value as an object with `x` and `y` properties. If both properties contain numbers, we construct a `QPoint` from them and call our `moveEntity()` function that tries to execute the move chosen by the strategy.

We will not blindly trust the value returned by the user's script. Instead, we carefully check whether the move is valid:

```cpp
bool Scene::moveEntity(Entity *entity, QPoint pos) {
    if (pos.x() < 0 || pos.y() < 0 ||
        pos.x() >= m_fieldSize.width() ||
        pos.y() >= m_fieldSize.height())
    {
        return false; // out of field bounds
    }
    QPoint posChange = entity->pos().toPoint() - pos;
    if (posChange.isNull()) {
        return true; // no change
    }
    if (qAbs(posChange.x()) > 1 || qAbs(posChange.y()) > 1) {
        return false; // invalid move
    }
    QGraphicsItem* item = itemAt(pos, QTransform());
    Entity* otherEntity = qgraphicsitem_cast<Entity*>(item);
    if (otherEntity) {
        otherEntity->setAlive(false);
    }
    entity->setPos(pos);
    return true;
}
```

We check that the new position is in bounds and is not too far from the entity's current position. If everything is correct, we execute the move. If another entity was on the destination tile, we mark it as killed. The function returns `true` if the move was successful.

That's it! Our game is ready to run. Let's create some strategy scripts to play with.

# Time for action – Writing a strategy script

Our first script will simply select a random move:

```cpp
{
    "step": function(current) {
        function getRandomInt(min, max) {
          return Math.floor(Math.random() * (max - min)) + min;
        }
        return {
            x: current.pos().x + getRandomInt(-1, 2),
            y: current.pos().y + getRandomInt(-1, 2),
        }
    }
}
```

Of course, a more intelligent strategy can beat this script. You can find a more advanced script in the code bundle. First, when it sees an enemy entity nearby, it always goes for the kill. If there is no such enemy, it tries to move away from the closest ally, attempting to fill the whole board. This script will easily wipe out the randomly moving enemy:

![](img/26146c49-3498-414e-b68c-7060c4d5d998.png)

Of course, there is always room for improvement. Try to think of a better strategy and write a script that can win the game.

# Have a go hero – Extending the game

There are a couple of ways for you to improve the game implementation. For example, you can detect when a player has won and display a pop-up message. You can also allow an arbitrary number of players. You just need to replace the `TEAM_COUNT` constant with a new property in the `Scene` class and define more team colors. You can even create a GUI for users to provide their scripts instead of passing them as command-line arguments.

The scripting environment can also be improved. You can provide more helper functions (for example, a function to calculate the distance between two tiles) to make creating scripts easier. On the other hand, you can modify the rules and reduce the amount of available information so that, for example, each entity can only see other entities at a certain distance.

As discussed earlier, each script has ways to break the global object or emit the signals of the exposed C++ objects, affecting the other players. To prevent that, you can create a separate `QJSEngine` and a separate set of proxy objects for each player, effectively sandboxing them.

# Python scripting

Qt QML is an environment that is designed to be part of the Qt world. Since not everyone knows or likes JavaScript, we will present another language that can easily be used to provide scripting environments for games that are created with Qt. Just be aware that this will not be an in-depth description of the environment—we will just show you the basics that can provide foundations for your own research.

A popular language used for scripting is Python. There are two variants of Qt bindings that are available for Python: PySide2 and PyQt. PySide2 is the official binding that is available under LGPL. PyQt is a third-party library that is available under GPL v3 and a commercial license.

PyQt is not available under LGPL, so for commercial closed-source products, you need to obtain a commercial license from Riverbank computing!

These bindings allow you to use the Qt API from within Python—you can write a complete Qt application using just Python. However, to call Python code from within C++, you will need a regular Python interpreter. Luckily, it is very easy to embed such an interpreter in a C++ application.

First, you will need Python installed, along with its development package. For example, for Debian-based systems, it is easiest to simply install the `libpythonX.Y-dev` package, where `X.Y` stands for the version of Python available in the repository:

```cpp
sudo apt-get install libpython3.5-dev
```

We will use Python 3.5 in our example, but later minor versions should also be compatible with our code.

Then, you need to tell qmake to link your program against the library. For Linux, you can use `pkgconfig` to do this automatically:

```cpp
CONFIG += link_pkgconfig no_keywords
# adjust the version number to suit your needs
PKGCONFIG += python-3.5m
```

The `no_keywords` configuration option tells the build system to disable Qt-specific keywords (`signals`, `slots`, and `emit`). We have to do this because Python headers use the `slots` identifier that would conflict with the same Qt keyword. You can still access the Qt keywords if you write them as `Q_SIGNALS`, `Q_SLOTS`, and `Q_EMIT`.

For Windows, you need to manually pass information to the compiler:

```cpp
CONFIG += no_keywords
INCLUDEPATH += C:\Python35\include
LIBS += -LC:\Python35\include -lpython35
```

To call Python code from within a Qt app, the simplest way is to use the following code:

```cpp
#include <Python.h>
#include <QtCore>

int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);
    Py_Initialize();
    const char *script = "print(\"Hello from Python\")";
    PyRun_SimpleString(script);
    Py_Finalize();
    return app.exec();
} 
```

This code initializes a Python interpreter, then invokes a script by passing it directly as a string, and finally, it shuts down the interpreter before invoking Qt's event loop. Such code makes sense only for simple scripting. In real life, you'd want to pass some data to the script or fetch the result. For that, we have to write some more code. As the library exposes the C API only, let's write a nice Qt wrapper for it.

# Time for action – Writing a Qt wrapper for embedding Python

As the first task, we will implement the last program using an object-oriented API. Create a new console project and add the following class to it:

```cpp
class QtPython : public QObject {
    Q_OBJECT
public:
    QtPython(QObject *parent = 0);
    ~QtPython();
    void run(const QString &program);

private:
    QVector<wchar_t> programNameBuffer;
};
```

The implementation file should look like this:

```cpp
#include <Python.h>
//...
QtPython::QtPython(QObject *parent) : QObject(parent) {
    QStringList args = qApp->arguments();
    if (args.count() > 0) {
        programNameBuffer.resize(args[0].count());
        args[0].toWCharArray(programNameBuffer.data());
        Py_SetProgramName(programNameBuffer.data());
    }
    Py_InitializeEx(0);
}

QtPython::~QtPython() {
    Py_Finalize();
}

void QtPython::run(const QString &program) {
    PyRun_SimpleString(qPrintable(program));
}
```

Then, add a `main()` function, as shown in the following snippet:

```cpp
int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    QtPython python;
    python.run("print('Hello from Python')");
    return 0;
}
```

Finally, open the `.pro` file and tell Qt to link with the Python library, as was shown earlier.

# What just happened?

We created a class called `QtPython` that wraps the Python C API for us.

Never use a `Q` prefix to call your custom classes, as this prefix is reserved for official Qt classes. This is to ensure that your code will never have a name clash with future code added to Qt. The Qt prefix, on the other hand, is meant to be used with classes that are extensions to Qt. You probably still shouldn't use it, but the probability of a name clash is much smaller and yields a lesser impact than clashes with an official class. It is best to come up with an application-specific prefix or use a namespace.

The class constructor creates a Python interpreter, and the class destructor destroys it. We use `Py_InitializeEx(0)`, which has the same functionality as `Py_Initialize()`, but it does not apply C signal handlers, as this is not something we would want when embedding Python. Prior to this, we use `Py_SetProgramName()` to inform the interpreter of our context. We also defined a `run()` method, taking `QString` and returning `void`. It uses `qPrintable()`, which is a convenience function that extracts a C string pointer from a `QString` object, which is then fed into `PyRun_SimpleString()`.

Never store the output of `qPrintable()`, as it returns an internal pointer to a temporary byte array (this is equivalent to calling `toLocal8Bit().constData()` on a string). It is safe to use directly, but the byte array is destroyed immediately afterward; thus, if you store the pointer in a variable, the data may not be valid later when you try using that pointer.

The most difficult work when using embedded interpreters is to convert values between C++ and the types that the interpreter expects. With Qt Script, the `QScriptValue` type was used for this. We can implement something similar for our Python scripting environment.

# Time for action – Converting data between C++ and Python

Create a new class and call it `QtPythonValue`:

```cpp
class QtPythonValue {
public:
    QtPythonValue();
    QtPythonValue(const QtPythonValue &other);
    QtPythonValue& operator=(const QtPythonValue &other);

    QtPythonValue(int val);
    QtPythonValue(const QString &str);
    ~QtPythonValue();

    int toInt() const;
    QString toString() const;
    bool isNone() const;

private:
    QtPythonValue(PyObject *ptr);
    void incRef();
    void incRef(PyObject *val);
    void decRef();

    PyObject *m_value;
    friend class QtPython;
};
```

Next, implement the constructors, the assignment operator, and the destructor, as follows:

```cpp
QtPythonValue::QtPythonValue() {
    incRef(Py_None);
}
QtPythonValue::QtPythonValue(const QtPythonValue &other) {
    incRef(other.m_value);
}
QtPythonValue::QtPythonValue(PyObject *ptr) {
    m_value = ptr;
}
QtPythonValue::QtPythonValue(const QString &str) {
    m_value = PyUnicode_FromString(qPrintable(str));
}
QtPythonValue::QtPythonValue(int val) {
    m_value = PyLong_FromLong(val);
}
QtPythonValue &QtPythonValue::operator=(const QtPythonValue &other) {
    if(m_value == other.m_value) {
        return *this;
    }
    decRef();
    incRef(other.m_value);
    return *this;
}
QtPythonValue::~QtPythonValue()
{
    decRef();
}
```

Then, implement the `incRef()` and `decRef()` functions:

```cpp
void QtPythonValue::incRef(PyObject *val) {
    m_value = val;
    incRef();
}
void QtPythonValue::incRef() {
    if(m_value) {
        Py_INCREF(m_value);
    }
}
void QtPythonValue::decRef() {
    if(m_value) {
        Py_DECREF(m_value);
    }
}
```

Next, implement conversions from `QtPythonValue` to C++ types:

```cpp
int QtPythonValue::toInt() const {
    return PyLong_Check(m_value) ? PyLong_AsLong(m_value) : 0;
}

QString QtPythonValue::toString() const {
    return PyUnicode_Check(m_value) ?
        QString::fromUtf8(PyUnicode_AsUTF8(m_value)) : QString();
}

bool QtPythonValue::isNone() const {
    return m_value == Py_None;
}
```

Finally, let's modify the `main()` function to test our new code:

```cpp
int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    QtPython python;
    QtPythonValue integer = 7, string = QStringLiteral("foobar"), none;
    qDebug() << integer.toInt() << string.toString() << none.isNone();
    return 0;
} 
```

When you run the program, you will see that the conversion between C++ and Python works correctly in both directions.

# What just happened?

The `QtPythonValue` class wraps a `PyObject` pointer (through the `m_value` member), providing a nice interface to convert between what the interpreter expects and our Qt types. Let's see how this is done. First, take a look at the three private methods: two versions of `incRef()` and one `decRef()`. `PyObject` contains an internal reference counter that counts the number of handles on the contained value. When that counter drops to 0, the object can be destroyed. Our three methods use adequate Python C API calls to increase or decrease the counter in order to prevent memory leaks and keep Python's garbage collector happy.

The second important aspect is that the class defines a private constructor that takes a `PyObject` pointer, effectively creating a wrapper over the given value. The constructor is private; however, the `QtPython` class is declared as a friend of `QtPythonValue`, which means that only `QtPython` and `QtPythonValue` can instantiate values by passing `PyObject` pointers to it. Now, let's take a look at public constructors.

The default constructor creates an object pointing to a `None` value, which represents the absence of a value. The copy constructor and assignment operator are pretty standard, taking care of bookkeeping of the reference counter. Then, we have two constructors—one taking `int` and the other taking a `QString` value. They use appropriate Python C API calls to obtain a `PyObject` representation of the value. Note that these calls already increase the reference count for us, so we don't have to do it ourselves.

The code ends with a destructor that decreases the reference counter and three methods that provide safe conversions from `QtPythonValue` to appropriate Qt/C++ types.

# Have a go hero – Implementing the remaining conversions

Now, you should be able to implement other constructors and conversions for `QtPythonValue` that operates on the `float`, `bool`, or even on `QDate` and `QTime` types. Try implementing them yourself. If needed, take a look at the Python documentation to find appropriate calls that you should use.

The documentation for Python 3.5 is available online at [https://docs.python.org/3.5/](https://docs.python.org/3.5/). If you've installed a different Python version, you can find the documentation for your version on the same website.

We'll give you a head start by providing a skeleton implementation of how to convert `QVariant` to `QtPythonValue`. This is especially important, because Python makes use of two types whose equivalents are not available in C++, namely, tuples and dictionaries. We will need them later, so having a proper implementation is crucial. Here's the code:

```cpp
QtPythonValue::QtPythonValue(const QVariant &variant)
{
    switch(variant.type()) {
    case QVariant::Invalid:
        incRef(Py_None);
        return;
    case QVariant::String:
        m_value = PyUnicode_FromString(qPrintable(variant.toString()));
        return;
    case QVariant::Int:
        m_value = PyLong_FromLong(variant.toInt());
        return;
    case QVariant::LongLong:
        m_value = PyLong_FromLongLong(variant.toLongLong());
        return;
    case QVariant::List: {
        QVariantList list = variant.toList();
        const int listSize = list.size();
        PyObject *tuple = PyTuple_New(listSize);
        for(int i = 0; i < listSize; ++i) {
            PyTuple_SetItem(tuple, i, QtPythonValue(list.at(i)).m_value);
        }
        m_value = tuple;
        return;
    }
    case QVariant::Map: {
        QVariantMap map = variant.toMap();
        PyObject *dict = PyDict_New();
        for(auto iter = map.begin(); iter != map.end(); ++iter) {
            PyDict_SetItemString(dict, qPrintable(iter.key()),
                                 QtPythonValue(iter.value()).m_value);
        }
        m_value = dict;
        return;
    }
    default:
        incRef(Py_None);
        return;
    }
}
```

The highlighted code shows how to create a tuple (which is a list of arbitrary elements) from `QVariantList` and how to create a dictionary (which is an associative array) from `QVariantMap`. You should also add a `QtPythonValue` constructor that takes `QStringList` and produces a tuple.

We have written quite a lot of code now, but there is no way of binding any data from our programs with Python scripting so far. Let's change that.

# Time for action – Calling functions and returning values

The next task is to provide ways to invoke Python functions and return values from scripts. Let's start by providing a richer `run()` API. Implement the following method in the `QtPython` class:

```cpp
QtPythonValue QtPython::run(const QString &program,
    const QtPythonValue &globals, const QtPythonValue &locals)
{
    PyObject *retVal = PyRun_String(qPrintable(program),
        Py_file_input, globals.m_value, locals.m_value);
    return QtPythonValue(retVal);
} 
```

We'll also need a functionality to import Python modules. Add the following methods to the class:

```cpp
QtPythonValue QtPython::import(const QString &name) const
{
    return QtPythonValue(PyImport_ImportModule(qPrintable(name)));
}

QtPythonValue QtPython::addModule(const QString &name) const
{
    PyObject *retVal = PyImport_AddModule(qPrintable(name));
    Py_INCREF(retVal);
    return QtPythonValue(retVal);
}

QtPythonValue QtPython::dictionary(const QtPythonValue &module) const
{
    PyObject *retVal = PyModule_GetDict(module.m_value);
    Py_INCREF(retVal);
    return QtPythonValue(retVal);
}
```

The last piece of the code is to extend `QtPythonValue` with this code:

```cpp
bool QtPythonValue::isCallable() const {
    return PyCallable_Check(m_value);
}

QtPythonValue QtPythonValue::attribute(const QString &name) const {
    return QtPythonValue(PyObject_GetAttrString(m_value, qPrintable(name)));
}

bool QtPythonValue::setAttribute(const QString &name, const QtPythonValue &value) {
    int retVal = PyObject_SetAttrString(m_value, qPrintable(name), value.m_value);
    return retVal != -1;
}

QtPythonValue QtPythonValue::call(const QVariantList &arguments) const {
    return QtPythonValue(
        PyObject_CallObject(m_value, QtPythonValue(arguments).m_value));
}

QtPythonValue QtPythonValue::call(const QStringList &arguments) const {
    return QtPythonValue(
        PyObject_CallObject(m_value, QtPythonValue(arguments).m_value));
}
```

Finally, you can modify `main()` to test the new functionality:

```cpp
int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    QtPython python;

    QtPythonValue mainModule = python.addModule("__main__");
    QtPythonValue dict = python.dictionary(mainModule);
    python.run("foo = (1, 2, 3)", dict, dict);
    python.run("print(foo)", dict, dict);

    QtPythonValue module = python.import("os");
    QtPythonValue chdir = module.attribute("chdir");
    chdir.call(QStringList() << "/home");
    QtPythonValue func = module.attribute("getcwd");
    qDebug() << func.call(QVariantList()).toString();

    return 0;
}
```

You can replace `/home` with a directory of your choice. Then, you can run the program.

# What just happened?

We did two tests in the last program. First, we used the new `run()` method, passing to it the code that is to be executed and two dictionaries that define the current execution context—the first dictionary contains global symbols and the second contains local symbols. The dictionaries come from Python's `__main__` module (which, among other things, defines the `print` function). The `run()` method may modify the contents of the two dictionaries—the first call defines the tuple called `foo`, and the second call prints it to the standard output.

The second test calls a function from an imported module; in this case, we call two functions from the `os` module—the first function, `chdir`, changes the current working directory, and the other, called `getcwd`, returns the current working directory. The convention is that we should pass a tuple to `call()`, where we pass the needed parameters. The first function takes a string as a parameter; therefore, we pass a `QStringList` object, assuming that there is a `QtPythonValue` constructor that converts `QStringList` to a tuple (you need to implement it if you haven't done it already). Since the second function does not take any parameters, we pass an empty tuple to the call. In the same way, you can provide your own modules and call functions from them, query the results, inspect dictionaries, and so on. This is a pretty good start for an embedded Python interpreter. Remember that a proper component should have some error checking code to avoid crashing the whole application.

You can extend the functionality of the interpreter in many ways. You can even use PyQt5 to use Qt bindings in scripts, combining Qt/C++ code with Qt/Python code.

# Have a go hero – Wrapping Qt objects into Python objects

At this point, you should be experienced enough to try and implement a wrapper for the `QObject` instances to expose signals and slots to Python scripting. If you decide to pursue the goal, [https://docs.python.org/3/](https://docs.python.org/3/) will be your best friend, especially the section about extending Python with C++. Remember that `QMetaObject` provides information about the properties and methods of Qt objects and `QMetaObject::invokeMethod()` allows you to execute a method by its name. This is not an easy task, so don't be hard on yourself if you are not able to complete it. You can always return to it once you gain more experience in using Qt and Python.

Before you head on to the next chapter, try testing your knowledge about scripting in Qt.

# Pop quiz

Q1\. Which is the method that you can use to execute JavaScript code?

1.  `QJSValue::call()`
2.  `QJSEngine::evaluate()`
3.  `QJSEngine::fromScriptValue()`

Q2\. What is the name of the class that serves as a bridge to exchange data between JS engine and C++?

1.  `QObject`
2.  `QJSValue`
3.  `QVariant`

Q3\. If you want to expose a C++ object to the script, which class must this object be derived from?

1.  `QObject`
2.  `QJSValue`
3.  `QGraphicsItem`

Q4\. Which of the following kinds of functions is not available to JavaScript code?

1.  Signals
2.  `Q_INVOKABLE` methods
3.  Slots
4.  Global functions

Q5\. When is a `PyObject` instance destroyed?

1.  When its value is set to `Py_None`
2.  When its internal reference counter drops to 0
3.  When the corresponding `QtPythonValue` is destroyed

# Summary

In this chapter, you learned that providing a scripting environment to your games opens up new possibilities. Implementing a functionality using scripting languages is usually faster than doing the full write-compile-test cycle with C++, and you can even use the skills and creativity of your users who have no understanding of the internals of your game engine to make your games better and more feature-rich. You were shown how to use `QJSEngine`, which blends the C++ and JavaScript worlds together by exposing Qt objects to JavaScript and making cross-language signal-slot connections. You also learned the basics of scripting with Python. There are other scripting languages available (for example, Lua), and many of them can be used along with Qt. Using the experience gained in this chapter, you should even be able to bring other scripting environments to your programs, as most embeddable interpreters offer similar approaches to that of Python.

In the next chapter, you will be introduced to Qt Quick—a library for creating fluid and dynamic user interfaces. It may not sound like it's related to this chapter, but Qt Quick is based on Qt QML. In fact, any Qt Quick application contains a `QJSEngine` object that executes JavaScript code of the application. Being familiar with this system will help you understand how such applications work. You will also be able to apply the skills you've learned here when you need to access C++ objects from Qt Quick and vice versa. Welcome to the world of Qt Quick.