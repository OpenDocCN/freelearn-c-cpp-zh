# Animations in Graphics View

The previous chapter gave you a lot of information about powers of Graphics View framework. With that knowledge, we can now proceed to implementing our first 2D game. Down the road, we will learn more about Qt's property system, explore multiple ways of performing animations, and add gamepad support to our application. By the end of the chapter, you will know all the most useful features of Graphics View.

Main topics covered in this chapter are as listed:

*   Using timers
*   Camera control
*   Parallax scrolling
*   Qt's property system
*   The Animation framework
*   Using Qt Gamepad module

# The jumping elephant or how to animate the scene

By now, you should have a good understanding of the items, the scene, and the view. With your knowledge of how to create items, standard and custom ones, of how to position them on the scene, and of how to set up the view to show the scene, you can make pretty awesome things. You can even zoom and move the scene with the mouse. That's surely good, but for a game, one crucial point is still missing—you have to animate the items.

Instead of going through all possibilities of how to animate a scene, let's develop a simple jump-and-run game where we recap parts of the previous topics and learn how to animate items on a screen. So let's meet Benjamin, the elephant:

![](img/cf9b50c4-ec05-414d-9adc-2556f4e0d602.jpg)

# The game play

The goal of the game is for Benjamin to collect the coins that are placed all over the game field. Besides walking right and left, Benjamin can, of course, also jump. In the following screenshot, you see what this minimalistic game should look like at the end:

![](img/27c2c13b-fbef-429e-8a1d-f5323ee7778b.png)

# Time for action - Creating an item for Benjamin

Let's create a new Qt Widgets project and start making our game. Since the project will become more complex than our previous projects, we will not be giving you precise instructions for editing the code. If at any time you are unsure about the changes you make, you can look at the reference implementation provided with the book. It also contains the image files you can use to implement the game.

Let's now look at how we can mobilize Benjamin. First, we need a custom item class for him. We call the `Player` class and choose `QGraphicsPixmapItem` as the base class, because Benjamin is a PNG image. In the item's `Player` class, we further create a private field of integer type and call it `m_direction`. Its value signifies in which direction Benjamin walks—left or right—or if he stands still. Next, we implement the constructor:

```cpp
Player::Player(QGraphicsItem *parent)
    : QGraphicsPixmapItem(parent)
    , m_direction(0)
{
    QPixmap pixmap(":/elephant");
    setPixmap(pixmap);
    setOffset(-pixmap.width() / 2, -pixmap.height() / 2);
}
```

In the constructor, we set `m_direction` to `0`, which means that Benjamin isn't moving at all. If `m_direction` is `1`, Benjamin moves right, and if the value is `-1`, he moves left. In the body of the constructor, we set the image for the item by calling `setPixmap()`. The image of Benjamin is stored in the Qt Resource system; thus, we access it through `QPixmap(":/elephant")`, with elephant as the given alias for the actual image of Benjamin. Finally, we use the `setOffset()` function to change how the pixmap is positioned in the item's coordinate system. By default, the origin point corresponds to the top-left corner of the pixmap, but we prefer to have it at the center of the pixmap so that applying transformations is easier.

When you are unsure of how to specify the path to your resource, you can ask Qt Creator about it. To do that, expand the Resources branch in the project tree, locate the resource, and select the Copy Path... entry in its context menu.

Next, we create a getter and setter function for the `m_direction` field:

```cpp
int Player::direction() const {
    return m_direction;
}

void Player::setDirection(int direction)
{
    m_direction = direction;
    if (m_direction != 0) {
        QTransform transform;
        if (m_direction < 0) {
            transform.scale(-1, 1);
        }
        setTransform(transform);
    }
}
```

The `direction()` function is a standard getter function for `m_direction` returning its value. The `setDirection()` setter function additionally checks in which direction Benjamin is moving. If he is moving left, we need to flip his image so that Benjamin looks to the left, the direction in which he is moving. If he is moving toward the right, we restore the normal state by assigning an empty `QTransform` object, which is an identity matrix.

We cannot use `QGraphicsItem::setScale` here, because it only supports the same scale factors for *x* and *y* axes. Fortunately, `setTransform()` enables us to set any affine or perspective transformation.

So, we now have our item of the `Player` class for the game's character, which shows the image of Benjamin. The item also stores the current moving direction, and based on that information, the image is flipped vertically if needed.

# The playing field

Since we will have to do some work on the scene, we subclass `QGraphicsScene` and name the new class `MyScene`. There, we implement one part of the game logic. This is convenient since `QGraphicsScene` inherits `QObject` and thus we can use Qt's signal and slot mechanism.

The scene creates the  environment in which our elephant will be walking and jumping. Overall, we have a view fixed in size holding a scene, which is exactly as big as the view. We do not take size changes of the view into account, since they will complicate the example too much.

All animations inside the playing field are done by moving the items, not the scene. So we have to distinguish between the view's, or rather the scene's, width and the width of the elephant's virtual "world" in which he can move. In order to handle the movement properly, we need to create a few private fields in the `MyScene` class.

The width of this virtual world is defined by the `int m_fieldWidth` field and has no (direct) correlation with the scene. Within the range of `m_fieldWidth`, which is 500 pixels in the example, Benjamin or the graphics item can be moved from the minimum *x* coordinate, defined by `qreal m_minX`, to the maximum x coordinate, defined by `qreal m_maxX`. We keep track of his actual *x* position with the `qreal m_currentX` variable. Next, the minimum *y* coordinate the item is allowed to have is defined by `qreal m_groundLevel`. We have to also take into account the item's size.

Lastly, what is left is the view, which has a fixed size defined by the scene's bounding rectangle size, which is not as wide as `m_fieldWidth`. So the scene (and the view) follows the elephant while he walks through his virtual world of the `m_fieldWidth` length. Take a look at the following diagram to see the variables in their graphical representation:

![](img/020be3fc-c2f7-4b63-9d9f-e010c72e9ea4.png)

# Time for action - Making Benjamin move

The next thing we want to do is make our elephant movable. In order to achieve that, we add a `QTimer m_timer` private member to `MyScene`. `QTimer` is a class that can emit the `timeout()` signal periodically with the given interval. In the `MyScene` constructor, we set up the timer with the following code:

```cpp
m_timer.setInterval(30);
connect(&m_timer, &QTimer::timeout, 
        this, &MyScene::movePlayer); 
```

First, we define that the timer emits the timeout signal every 30 milliseconds. Then, we connect that signal to the scene's slot called `movePlayer()`, but we do not start the timer yet. The timer will be started when the player presses a key to move.

Next, we need to handle the input events properly and update the player's direction. We introduce the `Player * m_player` field that will contain a pointer to the player object and the `int m_horizontalInput` field that will accumulate the movement commands, as we'll see in the next piece of code. Finally, we reimplement the `keyPressEvent` virtual function:

```cpp
void MyScene::keyPressEvent(QKeyEvent *event)
{
    if (event->isAutoRepeat()) {
        return;
    }
    switch (event->key()) {
    case Qt::Key_Right:
        addHorizontalInput(1);
        break;
    case Qt::Key_Left:
        addHorizontalInput(-1);
        break;
    //...
    }
}
void MyScene::addHorizontalInput(int input)
{
    m_horizontalInput += input;
    m_player->setDirection(qBound(-1, m_horizontalInput, 1));
    checkTimer();
}
```

As a small side note, whenever code snippets in the following code passages are irrelevant for the actual detail, we will skip the code but will indicate missing code with `//...` so that you know that it is not the entire code. We will cover the skipped parts later when it is more appropriate.

# What just happened?

In the key press event handler, we first check whether the key event was triggered because of an auto-repeat. If this is the case, we exit the function, because we only want to react on the first real key press event. Also, we do not call the base class implementation of that event handler since no item on the scene needs to get a key press event. If you do have items that could and should receive events, do not forget to forward them while reimplementing event handlers at the scene.

If you press and hold a key down, Qt will continuously deliver the key press event. To determine whether it was the first real key press or an autogenerated event, use `QKeyEvent::isAutoRepeat()`. It returns `true` if the event was automatically generated.

As soon as we know that the event was not delivered by an auto repeat, we react to the different key presses. Instead of calling the `setDirection()` method of the `Player *m_player` field directly, we use the `m_horizontalInput` class field to accumulate the input value. Whenever it's changed, we ensure the correctness of the value before passing it to `setDirection()`. For that, we use `qBound()`, which returns a value that is bound by the first and the last arguments. The argument in the middle is the actual value that we want to get bound, so the possible values in our case are restricted to -1, 0, and 1.

You might wonder, why not simply call `m_player->setDirection(1)` when the right key is pressed? Why accumulate the inputs in the  `m_horizontalInput` variable? Well, Benjamin is moved by the left and right arrow keys. If the right key is pressed, 1 is added; if it gets released, -1 is added. The same applies for the left key, but only the other way around. The addition of the value rather than setting it is now necessary because of a situation where a user presses and holds the right key, and the value of `m_direction` is therefore 1\. Now, without releasing the right key, they also press and hold the left key. Therefore, the value of `m_direction` is getting decreased by one; the value is now 0 and Benjamin stops. However, remember that both keys are still being pressed. What happens when the left key is released? How would you know in this situation in which direction Benjamin should move? To achieve that, you would have to find out an additional bit of information—whether the right key is still pressed down or not, which seems too much trouble and overhead. In our implementation, when the left key is released, 1 is added and the value of `m_direction` becomes 1, making Benjamin move right. Voilà! All without any concern about what the state of the other button might be.

After calling `setDirection()`, we call the `checkTimer()` function:

```cpp
void MyScene::checkTimer()
{
    if (m_player->direction() == 0) {
        m_timer.stop();
    } else if (!m_timer.isActive()) {
        m_timer.start();
    }
}
```

This function first checks whether the player moves. If not, the timer is stopped, because nothing has to be updated when our elephant stands still. Otherwise, the timer gets started, but only if it isn't already running. We check this by calling `isActive()` on the timer.

When the user presses the right key, for example, at the beginning of the game, `checkTimer()` will start `m_timer`. Since its `timeout` signal was connected to `movePlayer()`, the slot will be called every 30 milliseconds till the key is released.

Since the `movePlayer()` function is a bit longer, let's go through it step by step:

```cpp
const int direction = m_player->direction();
if (0 == direction) {
    return;
}
```

First, we cache the player's current direction in a local variable to avoid multiple calls of `direction()`. Then, we check whether the player is moving at all. If they aren't, we exit the function because there is nothing to animate:

```cpp
const int dx = direction * m_velocity;
qreal newX = qBound(m_minX, m_currentX + dx, m_maxX);
if (newX == m_currentX) {
    return;
}
m_currentX = newX;
```

Next, we calculate the shift the player item should get and store it in `dx`. The distance the player should move every 30 milliseconds is defined by the `int m_velocity` member variable, expressed in pixels. You can create setter and getter functions for that variable if you like. For us, the default value of 4 pixels will do the job. Multiplied by the direction (which could only be 1 or -1 at this point), we get a shift of the player by 4 pixels to the right or to the left. Based on this shift, we calculate the new *x* position of the player. Next, we check whether that new position is inside the range of `m_minX` and `m_maxX`, two member variables that are already calculated and set up properly at this point. Then, if the new position is not equal to the actual position, which is stored in `m_currentX`, we proceed by assigning the new position as the current one. Otherwise, we exit the function since there is nothing to move.

The next question to tackle is whether the view should always move when the elephant is moving, which means that the elephant would always stay, say, in the middle of the view. No, he shouldn't stay at a specific point inside the view. Rather, the view should be fixed when the elephant is moving. Only if he reaches the borders should the view follow. Let's say that when the distance between the elephant's center and the window's border is less than 150 pixels, we will try to shift the view:

```cpp
const int shiftBorder = 150;
const int rightShiftBorder = width() - shiftBorder;

const int visiblePlayerPos = m_currentX - m_worldShift;
const int newWorldShiftRight = visiblePlayerPos - rightShiftBorder;
if (newWorldShiftRight > 0) {
    m_worldShift += newWorldShiftRight;
}
const int newWorldShiftLeft = shiftBorder - visiblePlayerPos;
if (newWorldShiftLeft > 0) {
    m_worldShift -= newWorldShiftLeft;
}
const int maxWorldShift = m_fieldWidth - qRound(width());
m_worldShift = qBound(0, m_worldShift, maxWorldShift);
m_player->setX(m_currentX - m_worldShift);
```

The `int m_worldShift` class field shows how much we have already shifted our world to the right. First, we calculate the actual coordinate of our elephant in the view and save it to the `visiblePlayerPos` variable. Then, we calculate its position relative to the allowed area defined by the `shiftBorder` and `rightShiftBorder` variables. If `visiblePlayerPos` is beyond the right border of the allowed area, `newWorldShiftRight` will be positive, we need to shift the world by `newWorldShiftRight` to the right. Similarly, when we need to shift it to the left, `newWorldShiftLeft` will be positive, and it will contain the needed amount of shift. Finally, we update the position of `m_player` using a `setX()` helper method that is similar to `setPos()` but leaves the *y* coordinate unchanged.

Note that the value for `shiftBorder` is randomly chosen. You can alter it as you like. Of course, you can create a setter and getter for this parameter too.

The last important part to do here is to apply the new value of `m_worldShift` by setting positions of the other world items. While we're at it, we will implement parallax scrolling.

# Parallax scrolling

Parallax scrolling is a trick to add an illusion of depth to the background of the game. This illusion occurs when the background has different layers that move at different speeds. The nearest background must move faster than the ones farther away. In our case, we have these four backgrounds ordered from the most distant to the nearest:

The sky:

![](img/1acb98b6-41ce-451d-a796-4cb4ba3178cf.png)

The trees:

![](img/daaf9dd3-06e5-4035-8036-c6bd39caf251.jpg)

The grass:

![](img/3e86e498-e1a9-4832-afdd-035d7cf2cd6c.jpg)

The ground:

![](img/0862d948-cce9-4aac-949f-5283d933e3e9.png)

# Time for action - Moving the background

The scene will create a graphics item for each part of the background and store pointers to them in the `m_sky`, `m_grass`, and `m_trees` private fields. Now the question is how to move them at different speeds. The solution is quite simple—the slowest one, the sky, is the smallest image. The fastest background, the ground and the grass, are the largest images. Now when we take a look at the end of the `movePlayer()` function's slot, we see this:

```cpp
qreal ratio = qreal(m_worldShift) / maxWorldShift;
applyParallax(ratio, m_sky);
applyParallax(ratio, m_grass);
applyParallax(ratio, m_trees);
```

The `applyParallax()` helper method contains the following code:

```cpp
void MyScene::applyParallax(qreal ratio, QGraphicsItem* item) {
    item->setX(-ratio * (item->boundingRect().width() - width()));
}
```

# What just happened?

What are we doing here? At the beginning, the sky's left border is the same as the view's left border, both at point (0, 0). At the end, when Benjamin has walked to the maximum right, the sky's right border should be the same as the view's right border. So, at this position, the shift of the sky will be equal to the sky's width (`m_sky->boundingRect().width()`) minus the width of the view (`width()`). The shift of the sky depends on the position of the camera and, consequently, the value of the `m_worldShift` variable; if it is far to the left, the sky isn't shifted, and if the camera is far to the right, the sky is maximally shifted. Thus, we have to multiply the sky's maximum shift value with a factor based on the current position of the camera. The relation to the camera's position is the reason this is handled in the `movePlayer()` function. The factor we have to calculate has to be between 0 and 1\. So we get the minimum shift (`0 * shift`, which equals 0) and the maximum shift (`1 * shift`, which equals `shift`). We name this factor as `ratio`.

How far the world was shifted is saved in `m_worldShift`, so by dividing `m_worldShift` by `maxWorldShift`, we get the needed factor. It is 0 when the player is to the far left and 1 if they are to the far right. Then, we have to simply multiply `ratio` with the maximum shift of the sky.

The same calculation is used for the other background items, so it is moved to a separate function. The calculation also explains why a smaller image is moving slower. It's because the overlap of the smaller image is less than that of the larger one, and since the backgrounds are moved in the same time period, the larger one has to move faster.

# Have a go hero - Adding new background layers

Try to add additional background layers to the game, following the preceding example. As an idea, you can add a barn behind the trees or let an airplane fly through the sky.

# The Animation framework

For now, we have calculated and applied new positions for our graphics items manually. However, Qt provides a way to do it automatically, called the Animation framework.

The framework is an abstract implementation of animations, so it can be applied to any `QObject`, such as widgets, or even plain variables. Graphics, items can be animated too, and we will get to this topic soon. Animations are not restricted to the object's coordinates. You can animate color, opacity, or a completely invisible property.

To create an animation, you typically need to perform the following steps:

1.  Create an animation object (such as `QPropertyAnimation`)
2.  Set the object that should be animated
3.  Set the name of the property to be animated
4.  Define how exactly the value should change (for example, set starting and ending values)
5.  Start the animation

As you probably know, calling an arbitrary method by name is not possible in C++, and yet, the animation objects are able to change arbitrary properties at will. This is possible because "property" is not only a fancy name, but also another powerful feature of the `QObject` class and Qt meta-object compiler.

# Properties

In [Chapter 3](ebffc011-752f-4dbe-a383-0917a002841d.xhtml), *Qt GUI Programming*, we edited predefined properties of widgets in the form editor and used their getter and setter methods in the code. However, until now, there wasn't a real reason for us to declare a new property. It'll be useful with the Animation framework, so let's pay more attention to properties.

Only classes that inherit `QObject` can declare properties. To create a property, we first need to declare it in a private section of the class (usually right after the `Q_OBJECT` mandatory macro) using a special `Q_PROPERTY` macro. That macro allows you to specify the following information about the new property:

*   The property name—a string that identifies the property in the Qt meta system.
*   The property type—any valid C++ type can be used for a property, but animations will only work with a limited set of types.
*   Names of the getter and setter method for this property. If declared in `Q_PROPERTY`, you must add them to your class and implement them properly.
*   Name of the signal that is emitted when the property changes. If declared in `Q_PROPERTY`, you must add the signal and ensure that it's properly emitted.

There are more configuration options, but they are less frequently needed. You can learn more about them from the The Property System documentation page.

The Animation framework supports the following property types: `int`, `unsigned int`, `double`, `float`, `QLine`, `QLineF`, `QPoint`, `QPointF`, `QSize`, `QSizeF`, `QRect`, `QRectF`, and `QColor`. Other types are not supported, because Qt doesn't know how to interpolate them, that is, how to calculate intermediate values based on the start and end values. However, it's possible to add support for custom types if you really need to animate them.

Similar to signals and slots, properties are powered by **moc**, which reads the header file of your class and generates extra code that enables Qt (and you) to access the property at runtime. For example, you can use the `QObject::property()` and `QObject::setProperty()` methods to get and set properties by name.

# Time for action - Adding a jump animation

Go to the `myscene.h` file and add a private `qreal m_jumpFactor` field. Next, declare a getter, a setter, and a change signal for this field:

```cpp
public:
    //...
    qreal jumpFactor() const;
    void setJumpFactor(const qreal &jumpFactor);
signals:
    void jumpFactorChanged(qreal);
```

In the header file, we declare the `jumpFactor` property by adding the following code just after the `Q_OBJECT` macro:

```cpp
Q_PROPERTY(qreal jumpFactor
           READ jumpFactor
           WRITE setjumpFactor
           NOTIFY jumpFactorChanged)
```

Here, `qreal` is the type of the property, `jumpFactor` is the registered name, and the following three lines register the corresponding member functions of the `MyScene` class in the property system. We'll need this property to make Benjamin jump, as we will see later on.

The `jumpFactor()` getter function simply returns the `m_jumpFactor` private member, which is used to store the actual position. The implementation of the setter looks like this:

```cpp
void MyScene::setjumpFactor(const qreal &pos) {
    if (pos == m_jumpFactor) {
        return;
    }
    m_jumpFactor = pos;
    emit jumpFactorChanged(m_jumpFactor);
} 
```

It is important to check whether `pos` will change the current value of `m_jumpFactor`. If this is not the case, exit the function, because we don't want the change signal to be emitted even if nothing has changed. Otherwise, we set `m_jumpFactor` to `pos` and emit the signal that informs about the change.

# Property animations

We implemented the horizontal movement using a `QTimer`. Now, let's try a second way to animate items—the Animation framework.

# Time for action - Using animations to move items smoothly

Let's add a new private member called `m_jumpAnimation` of the `QPropertyAnimation *` type, and initialize it in the constructor of `MyScene`:

```cpp
m_jumpAnimation = new QPropertyAnimation(this);
m_jumpAnimation->setTargetObject(this);
m_jumpAnimation->setPropertyName("jumpFactor");
m_jumpAnimation->setStartValue(0);
m_jumpAnimation->setKeyValueAt(0.5, 1);
m_jumpAnimation->setEndValue(0);
m_jumpAnimation->setDuration(800);
m_jumpAnimation->setEasingCurve(QEasingCurve::OutInQuad); 
```

# What just happened?

For the instance of `QPropertyAnimation` created here, we define the item as a parent; thus, the animation will get deleted when the scene deletes the item, and we don't have to worry about freeing the used memory. Then, we define the target of the animation—our `MyScene` class—and the property that should be animated, `jumpFactor`, in this case. Then, we define the start and the end value of that property; in addition to that, we also define a value in between, by setting `setKeyValueAt()`. The first argument of the `qreal` type defines time inside the animation, where 0 is the beginning and 1 the end, and the second argument defines the value that the animation should have at that time. So your `jumpFactor` element will get animated from 0 to 1 and back to 0 in 800 milliseconds. This was defined by `setDuration()`. Finally, we define how the interpolation between the start and end value should be done and call `setEasingCurve()`, with `QEasingCurve::OutInQuad` as an argument.

Qt defines up to 41 different easing curves for linear, quadratic, cubic, quartic, quintic, sinusoidal, exponential, circular, elastic, back easing, and bounce functions. These are too many to describe here. Instead, take a look at the documentation; simply search for `QEasingCurve::Type`.

In our case, `QEasingCurve::OutInQuad` ensures that the jump speed of Benjamin looks like an actual jump: fast in the beginning, slow at the top, and fast at the end again. We start this animation with the `jump` function:

```cpp
void MyScene::jump()
{
    if (QAbstractAnimation::Stopped == m_jumpAnimation->state()) {
        m_jumpAnimation->start();
    }
}
```

We only start the animation by calling `start()` when the animation isn't running. Therefore, we check the animation's state to see whether it has been stopped. Other states could be `Paused` or `Running`. We want this jump action to be activated whenever the player presses the Space key on their keyboard. Therefore, we expand the `switch` statement inside the key press event handler using this code:

```cpp
case Qt::Key_Space:
    jump();
    break; 
```

Now the property gets animated, but Benjamin will still not jump. Therefore, we handle the changes of the `jumpFactor` value at the end of the `setJumpFactor` function:

```cpp
void MyScene::setJumpFactor(const qreal &jumpFactor)
{
    //...
    qreal groundY = (m_groundLevel - m_player->boundingRect().height()  
                                                                 / 2);
    qreal y = groundY - m_jumpAnimation->currentValue().toReal() * 
                                                      m_jumpHeight;
    m_player->setY(y);
    //...
}
```

When our `QPropertyAnimation` is running, it will call our `setJumpFactor()` function to update the property's value. Inside that function, we calculate the *y* coordinate of the player item to respect the ground level defined by `m_groundLevel`. This is done by subtracting half of the item's height from the ground level's value since the item's origin point is in its center. Then, we subtract the maximum jump height, defined by `m_jumpHeight`, which is multiplied by the actual jump factor. Since the factor is in the range of 0 and 1, the new *y* coordinate stays inside the allowed jump height. Then, we alter the player item's *y* position by calling `setY()`, leaving the *x* coordinate as the same. Et voilà, Benjamin is jumping!

# Have a go hero - Letting the item handle Benjamin's jump

Since the scene is already a `QObject`, adding a property to it was easy. However, imagine that you want to create a game for two players, each controlling a separate `Player` item. In this case, the jump factors of two elephants need to be animated independently, so you want to make an animated property in the `Player` class, instead of putting it to the scene.

The `QGraphicsItem` item and all standard items introduced so far don't inherit  `QObject` and thus can't have slots or emit signals; they don't benefit from the `QObject` property system either. However, we can make them use `QObject`! All you have to do is add `QObject` as a base class and add the `Q_OBJECT` macro:

```cpp
class Player : public QObject, public QGraphicsPixmapItem {
    Q_OBJECT
//...
};
```

Now you can use properties, signals, and slots with items too. Be aware that `QObject` must be the first base class of an item.

A word of warning
Only use `QObject` with items if you really need its capabilities. `QObject` 
adds a lot of overhead to the item, which will have a noticeable impact on performance when you have many items, so use it wisely and not only because you can.

If you make this change, you can move the `jumpFactor` property from `MyScene` to `Player`, along with a lot of related code. You can make the code even more consistent by handling the horizontal movement in `Player` as well. Let `MyScene` handle the input events and forward the movement commands to `Player`.

# Time for action - Keeping multiple animations in sync

Now we'll start implementing the coin class. We can use a simple `QGraphicsEllipseItem` object, but we'll need to animate its properties, so let's create a new `Coin` class and derive it from `QObject` and `QGraphicsEllipseItem`. Define two properties: `opacity` of the `qreal` type and `rect` of the `QRect` type. This is done only by the following code:

```cpp
class Coin : public QObject, public QGraphicsEllipseItem
{
    Q_OBJECT
    Q_PROPERTY(qreal opacity READ opacity WRITE setOpacity)
    Q_PROPERTY(QRectF rect READ rect WRITE setRect)
//...
};
```

No function or slot was added, because we simply used built-in functions of `QGraphicsItem` and associated them with the properties.

If you want an item that inherits from `QObject` and `QGraphicsItem`, you can directly inherit `QGraphicsObject`. Moreover, it already registers all general `QGraphicsItem` properties in the metasystem, including `pos`, `scale`, `rotation`, and `opacity`. All properties come with corresponding notification signals, such as `opacityChanged()`. However, when you inherit `QGraphicsObject`, you cannot, at the same time, inherit `QGraphicsEllipseItem` or any other item class. So in this case, we will need to either implement painting of the ellipse manually or add a child `QGraphicsEllipseItem` that can perform the painting for us.

Next, we'll create the `explode()` function that will start some animations when the player collects the coin. Create a Boolean private field in the class and use it to ensure that each coin can only explode once:

```cpp
void Coin::explode()
{
    if (m_explosion) {
        return;
    }
    m_explosion = true;
    //...
}
```

We want to animate our two properties by two `QPropertyAnimation` objects. One fades the coin out, while the other scales the coin in. To ensure that both animations get started at the same time, we use `QParallelAnimationGroup`, as follows:

```cpp
QPropertyAnimation *fadeAnimation = 
    new QPropertyAnimation(this, "opacity");
//...
QPropertyAnimation *scaleAnimation = new QPropertyAnimation(this, "rect");
//...
QParallelAnimationGroup *group = new QParallelAnimationGroup(this);
group->addAnimation(scaleAnimation);
group->addAnimation(fadeAnimation);
connect(group, &QParallelAnimationGroup::finished,
        this,  &Coin::deleteLater);
group->start(); 
```

# What just happened?

You already know how to set up a single property animation, so we omitted the code for it. After setting up both animations, we add them to the group animation by calling `addAnimation()` on the group, while passing a pointer to the animation we would like to add. Then, when we start the group; `QParallelAnimationGroup` ensures that all assigned animations start at the same time.

When both animations have finished, `group` will emit the `finished()` signal. We connected that signal to the `deleteLater()` slot of our class so that the coin object gets deleted when it's no longer visible. This handy slot is declared in the `QObject` class and is useful in many cases.

In some cases, you may want to stop an animation. You can do that by calling the `stop()` method. It's also possible to pause and resume an animation using `pause()` and `resume()`. Using these methods on a `QParallelAnimationGroup` will affect all transformations added to that group.

# Chaining multiple animations

What if we wanted to perform an animation at the end of another animation? We could connect the `finished()` signal of the first animation to the `start()` slot of the second one. However, a much more convenient solution is to use `QSequentialAnimationGroup`. For example, if we want coins to scale and *then* to fade, the following code will do the trick:

```cpp
QSequentialAnimationGroup *group = new QSequentialAnimationGroup(this);
group->addAnimation(scaleAnimation);
group->addAnimation(fadeAnimation);
group->start();
```

# Adding gamepad support

The player can use the keyboard to play our game, but it would be nice to also allow playing it using a gamepad. Fortunately, Qt provides the Qt Gamepad add-on that allows us to do this easily. As opposed to Qt Essentials (for example, Qt Widgets), add-ons may be supported on a limited number of platforms. As of Qt 5.9, Qt Gamepad supports Windows, Linux, Android, macOS, iOS, and tvOS (including the tvOS remote).

# Working with gamepads in Qt

The starting point of the gamepad API is the `QGamepadManager` class. The singleton object of this class can be obtained using the `QGamepadManager::instance()` function. It allows you to request the list of identifiers of the available gamepads using the `connectedGamepads()` function. The `gamepadConnected()` signal can be used to detect new gamepads on the fly. `QGamepadManager` also provides API for configuring buttons and axes on the gamepad and is able to save the configuration to the specified settings file.

After you detected that one or multiple gamepads are available in the system, you should create a new `QGamepad` object and pass the obtained device identifier as a constructor's argument. You can use the first available gamepad or allow the user to select which gamepad to use. In this case, you can utilize the gamepad's `name` property that returns a readable name of the device.

The `Gamepad` object contains a dedicated property for each axis and button. This gives you two ways to receive the information about the state of the controls. First, you can use the getter of the property to check the current state of a button or an axis. For example, the `buttonL1()` function will return `true` if the L1 button is currently pressed, and the `axisLeftX()` will return the current horizontal position of the left stick as a `double` value that is in the range of -1 to 1\. For trigger buttons (for example, `buttonL2()`), the property contains a `double` value that ranges from 0 (not pressed) to 1 (fully pressed).

The second way is to use the signals corresponding to each property. For example, you can connect to the gamepad's `buttonL1Changed(bool value)` and `axisLeftXChanged(double value)` signals to monitor the changes of the corresponding properties.

Finally, the `QGamepadKeyNavigation` class can be used to quickly add gamepad support to a keyboard-oriented application. When you create an object of this class, your application will begin receiving key events caused by gamepads. By default, `GamepadKeyNavigation` will emulate up, down, left, right, back, forward, and return keys when the corresponding gamepad buttons are pressed. However, you can override the default mapping or add your own mapping for other gamepad buttons.

# Time for action - Handling gamepad events

Let's start with adding the Qt Gamepad add-on to our project by editing the `jrgame.pro` file:

```cpp
QT += core gui widgets gamepad
```

This will make the headers of the library available to our project and tell `qmake` to link the project against this library. Now add the following code to the constructor of the `MyScene` class:

```cpp
QList<int> gamepadIds = QGamepadManager::instance()->connectedGamepads();
if (!gamepadIds.isEmpty()) {
    QGamepad *gamepad = new QGamepad(gamepadIds[0], this);
    connect(gamepad, &QGamepad::axisLeftXChanged,
            this, &MyScene::axisLeftXChanged);
    connect(gamepad, &QGamepad::axisLeftYChanged,
            this, &MyScene::axisLeftYChanged);
}
```

The code is pretty straightforward. First, we use 
`QGamepadManager::connectedGamepads`  to get the list of IDs of the available gamepads. If some gamepads were found, we create a `QGamepad` object for the first found gamepad. We pass `this` to its constructor, so it becomes a child of our `MyScene` object, and we don't need to worry about deleting it. Finally, we connect the gamepad's `axisLeftXChanged()` and `axisLeftYChanged()` signals to new slots in the `MyScene` class. Now, let's implement these slots:

```cpp
void MyScene::axisLeftXChanged(double value)
{
    int direction;
    if (value > 0) {
        direction = 1;
    } else if (value < 0) {
        direction = -1;
    } else {
        direction = 0;
    }
    m_player->setDirection(direction);
    checkTimer();
}

void MyScene::axisLeftYChanged(double value)
{
    if (value < -0.25) {
        jump();
    }
}
```

The `value` argument of the signals contains a number from -1 to 1\. It allows us not
only to detect whether a thumbstick was pressed, but also to get more precise information
about its position. However, in our simple game, we don't need this precision. In the `axisLeftXChanged()` slot, we calculate and set the elephant's `direction` based on the sign of the received value. In the `axisLeftYChanged()` slot, if we receive a large enough negative value, we interpret it as a `jump` command. This will help us avoid accidental jumps. That's all! Our game now supports both keyboards and gamepads.

If you need to react to other buttons and thumbsticks of the gamepad, use the other signals of the `QGamepad` class. It's also possible to read multiple gamepads at the same time by creating multiple `QGamepad` objects with different IDs.

# Item collision detection

Whether the player item collides with a coin is checked by the scene's `checkColliding()` function, which is called after the player item has moved horizontally or vertically.

# Time for action - Making the coins explode

The implementation of `checkColliding()` looks like this:

```cpp
void MyScene::checkColliding()
{
    for(QGraphicsItem* item: collidingItems(m_player)) {
        if (Coin *c = qgraphicsitem_cast<Coin*>(item)) {
            c->explode();
        }
    }
}
```

# What just happened?

First, we call the scene's `QGraphicsScene::collidingItems()` function, which takes the item for which colliding items should be detected as a first argument. With the second, optional argument, you can define how the collision should be detected. The type of that argument is `Qt::ItemSelectionMode`, which was explained earlier. By default, an item will be considered colliding with `m_player` if the shapes of the two items intersect.

Next, we loop through the list of found items and check whether the current item is a `Coin` object. This is done by trying to cast the pointer to `Coin`. If it is successful, we explode the coin by calling `explode()`. Calling the `explode()` function multiple times is no problem, since it will not allow more than one explosion. This is important since `checkColliding()` will be called after each movement of the player. So the first time the player hits a coin, the coin will explode, but this takes time. During this explosion, the player will most likely be moved again and thus collides with the coin once more. In such a case, `explode()` may be called multiple times.

The `qgraphicsitem_cast<>()` is a faster alternative to `dynamic_cast<>()`. However, it will properly work for custom types only if they implement `type()` properly. This virtual function must return a different value for each custom item class in the application.

The `collidingItems()` function will always return the background items as well, since the player item is above all of them most of the time. To avoid the continuous check if they actually are coins, we use a trick. Instead of using `QGraphicsPixmapItem` directly, we subclass it and reimplement its virtual `shape()` function, as follows:

```cpp
QPainterPath BackgroundItem::shape() const {
  return QPainterPath();
} 
```

We already used the `QPainterPath` class in the previous chapter. This function just returns an empty `QPainterPath`. Since the collision detection is done with the item's shape, the background items can't collide with any other item since their shape is permanently empty. Don't try this trick with `boundingRect()` though, because it must always be valid.

Had we done the jumping logic inside `Player`, we could have implemented the item collision detection from within the item itself. `QGraphicsItem` also offers a `collidingItems()` function that checks against colliding items with itself. So `scene->collidingItems(item)` is equivalent to `item->collidingItems()`.

If you are only interested in whether an item collides with another item, you can call `collidesWithItem()` on the item, passing the other item as an argument.

# Finishing the game

The last part we have to discuss is the scene's initialization. Set the initial values for all fields and the constructor, create the `initPlayField()` function that will set up all the items, and call that function in the constructor. First, we initialize the sky, trees, ground, and player item:

```cpp
void MyScene::initPlayField()
{
    setSceneRect(0, 0, 500, 340);
    m_sky = new BackgroundItem(QPixmap(":/sky"));
    addItem(m_sky);
    BackgroundItem *ground = new BackgroundItem(QPixmap(":/ground"));
    addItem(ground);
    ground->setPos(0, m_groundLevel);
    m_trees = new BackgroundItem(QPixmap(":/trees"));
    m_trees->setPos(0, m_groundLevel - m_trees->boundingRect().height());
    addItem(m_trees);
    m_grass = new BackgroundItem(QPixmap(":/grass"));
    m_grass->setPos(0,m_groundLevel - m_grass->boundingRect().height());
    addItem(m_grass);
    m_player = new Player();
    m_minX = m_player->boundingRect().width() * 0.5;
    m_maxX = m_fieldWidth - m_player->boundingRect().width() * 0.5;
    m_player->setPos(m_minX, m_groundLevel - m_player->boundingRect().height() / 2);
    m_currentX = m_minX;
    addItem(m_player);
    //...
}
```

Next, we create coin objects:

```cpp
m_coins = new QGraphicsRectItem(0, 0, m_fieldWidth, m_jumpHeight);
m_coins->setPen(Qt::NoPen);
m_coins->setPos(0, m_groundLevel - m_jumpHeight);
const int xRange = (m_maxX - m_minX) * 0.94;
for (int i = 0; i < 25; ++i) {
    Coin *c = new Coin(m_coins);
    c->setPos(m_minX + qrand() % xRange, qrand() % m_jumpHeight);
}
addItem(m_coins);
```

In total, we are adding 25 coins. First, we set up an invisible item with the size of the virtual world, called `m_coins`. This item should be the parent to all coins. Then, we calculate the width between `m_minX` and `m_maxX`. That is the space where Benjamin can move. To make it a little bit smaller, we only take 94 percent of that width. Then, in the `for` loop, we create a coin and randomly set its *x* and *y* position, ensuring that Benjamin can reach them by calculating the modulo of the available width and of the maximal jump height. After all 25 coins are added, we place the parent item holding all the coins on the scene. Since most coins are outside the actual view's rectangle, we also need to move the coins while Benjamin is moving. Therefore, `m_coins` must behave like any other background. For this, we simply add the following code to the `movePlayer()` function, where we also move the background by the same pattern:

```cpp
applyParallax(ratio, m_coins);
```

# Have a go hero - Extending the game

That's it. This is our little game. Of course, there is much room to improve and extend it. For example, you can add some barricades Benjamin has to jump over. Then, you would have to check whether the player item collides with such a barricade item when moving forward, and if so, refuse movement. You have learned all the necessary techniques you need for that task, so try to implement some additional features to deepen your knowledge.

# A third way of animation

Besides `QTimer` and `QPropertyAnimation`, there is a third way to animate the scene. The scene provides a slot called `advance()`. If you call that slot, the scene will forward that call to all items it holds by calling `advance()` on each one. The scene does that twice. First, all item `advance()` functions are called with `0` as an argument. This means that the items are about to advance. Then, in the second round, all items are called passing 1 to the item's `advance()` function. In that phase, each item should advance, whatever that means—maybe moving, maybe a color change, and so on. The scene's slot advance is typically called by a `QTimeLine` element; with this, you can define how many times during a specific period of time the timeline should be triggered.

```cpp
QTimeLine *timeLine = new QTimeLine(5000, this);
timeLine->setFrameRange(0, 10); 
```

This timeline will emit the `frameChanged()` signal every 5 seconds for 10 times. All you have to do is connect that signal to the scene's `advance()` slot, and the scene will advance 10 times in 50 seconds. However, since all items receive two calls for each advance, this may not be the best animation solution for scenes with a lot of items where only a few should advance.

# Pop quiz

Q1\. Which of the following is a requirement for animating a property?

1.  The name of the property must start with "`m_`".
2.  Getter and setter of the property must be slots.
3.  The property must be declared using the `Q_PROPERTY` macro.

Q2\. Which class sends a signal when a gamepad button is pressed or released?

1.  `QGamepad`
2.  `QWidget`
3.  `QGraphicsScene`

Q3\. What is the difference between the `shape()` and `boundingRect()` functions of `QGraphicsItem`?

1.  `shape()` returns the bounding rectangle as a `QPainterPath` instead of a `QRectF`
2.  `shape()` causes the item to be repainted.
3.  `share()` can return a more precise description of the item's boundaries than `boundingRect()`

# Summary

In this chapter, you deepened your knowledge about items, about the scene, and about the view. While developing the game, you became familiar with different approaches of how to animate items, and you were taught how to detect collisions. As an advanced topic, you were introduced to parallax scrolling.

After having completed the two chapters describing Graphics View, you should now know almost everything about it. You are able to create complete custom items, you can alter or extend standard items, and with the information about the level of detail, you even have the power to alter an item's appearance, depending on its zoom level. You can transform items and the scene, and you can animate items and thus the entire scene.

Furthermore, as you saw while developing the game, your skills are good enough to develop a jump-and-run game with parallax scrolling, as it is used in highly professional games. We also learned how to add gamepad support to our game. To keep it fluid and highly responsive, finally we saw some tricks on how to get the most out of Graphics View.

When we worked with widgets and the Graphics View framework, we had to use some general purpose Qt types, such as `QString` or `QVector`. In simple cases, their API is pretty obvious. However, these and many other classes provided by Qt Core module are very powerful, and you will greatly benefit from deeper knowledge of them. When you develop a serious project, it's very important to understand how these basic types work and what dangers they may pose when used incorrectly. In the next chapter, we will turn our attention to this topic. You will learn how you can work with text in Qt, which containers you should use in different cases, and how to manipulate various kind of data and implement a persistent storage. This is essential for any game that is more complicated than our simple examples.