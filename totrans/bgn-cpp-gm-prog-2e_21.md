# *第二十章*：游戏对象和组件

在本章中，我们将编写与上一章开头讨论的实体-组件模式相关的所有编码。这意味着我们将编写基础`Component`类，其他所有组件都将从这个类派生。我们还将充分利用我们对智能指针的新知识，以便我们不必担心跟踪为这些组件分配的内存。我们还将在本章中编写`GameObject`类。

本章我们将涵盖以下主题：

+   准备编写组件

+   编写组件基类

+   编写碰撞器组件

+   编写图形组件

+   编写更新组件

+   编写游戏对象类

在我们开始编码之前，让我们更详细地讨论一下组件。请注意，在本章中，我将尝试加强实体-组件系统如何结合在一起，以及所有组件如何组成一个游戏对象。我不会解释每一行或甚至每一个逻辑块或已经多次见过的 SFML 相关代码。这些细节需要你自己去研究。

# 准备编写组件

在你完成本章的过程中，会有很多错误，其中一些可能看起来没有逻辑。例如，你可能会得到错误信息，说某个类不存在，而实际上它正是你已经编写的类之一。原因在于，当一个类中存在错误时，其他类无法可靠地使用它，否则也会出现错误。正因所有类之间相互关联的特性，我们直到下一章的结尾才能消除所有错误，再次获得可执行的代码。本可以分小块向各个类和项目添加代码，这样项目出现错误的频率会更高。然而，逐步进行意味着需要不断在各个类之间切换。当你构建自己的项目时，这有时是一种好的做法，但我认为对于这个项目来说，最有教育意义的事情是帮助你尽可能快地构建它。

# 编写组件基类

在`Header Files/GameObjects`过滤器中创建一个新的头文件，命名为`Component.h`，并添加以下代码：

```cpp
#pragma once
#include "GameObjectSharer.h"
#include <string>
using namespace std;
class GameObject;
class Component {
public:
    virtual string getType() = 0;
    virtual string getSpecificType() = 0;
    virtual void disableComponent() = 0;
    virtual void enableComponent() = 0;
    virtual bool enabled() = 0;
    virtual void start(GameObjectSharer* gos, GameObject* self) = 0;
};
```

这是每个游戏对象中每个组件的基类。纯虚函数意味着组件不能被实例化，必须首先继承。函数允许访问组件的类型和特定类型。组件类型包括碰撞器、图形、变换和更新，但根据游戏需求还可以添加更多类型。具体类型包括标准图形、入侵者更新、玩家更新等。

有两个函数允许组件被启用和禁用。这很有用，因为组件可以在使用之前测试它是否当前已启用。例如，你可以调用 `enabled` 函数来测试在调用 `update` 函数之前组件的更新组件是否已启用，或者图形组件在调用 `draw` 函数之前是否已启用。

`start` 函数可能是最有趣的功能，因为它将其参数之一设为一个新的类类型。`GameObjectSharer` 类将在所有组件实例化后提供对所有游戏对象的访问。这将给每个游戏对象中的每个组件提供查询详细信息甚至获取指向另一个游戏对象中特定数据指针的机会。例如，所有侵略者的更新组件都需要知道玩家变换组件的位置，以便知道何时开火。在 `start` 函数中，可以访问任何对象的任何部分。关键是每个特定的组件将决定它们需要什么，并且在关键的游戏循环中不需要查询另一个游戏对象的详细信息。

包含该组件的 `GameObject` 也会传递给 `start` 函数，这样任何组件都可以了解更多关于自己的信息。例如，图形组件需要了解变换组件，以便知道在哪里绘制自己。作为第二个例子，侵略者和玩家飞船的更新组件需要指向它们自己的碰撞器组件的指针，这样它们就可以在移动时更新其位置。

随着我们继续前进，我们将看到更多 `start` 函数的使用案例。

在 `Source Files/GameObjects` 过滤器中创建一个新的源文件，命名为 `Component.cpp`，并添加以下代码：

```cpp
/*********************************
******THIS IS AN INTERFACE********
*********************************/
```

由于 `Component` 类永远不能实例化，我将前面的注释放在 `Component.cpp` 中作为提醒。

# 编写碰撞器组件

《太空侵略者++》游戏将只包含一种简单的碰撞器类型。它将是一个围绕对象的矩形框，就像我们在《僵尸末日》和《乒乓》游戏中使用的那样。然而，很容易想象你可能需要其他类型的碰撞器；可能是一个圆形碰撞器，或者是一个非包围的碰撞器，就像我们在《托马斯迟到了》游戏中用于托马斯和鲍勃的头、脚和侧面的那些。

因此，将有一个基类 `ColliderComponent`（继承自 `Component`），它将处理所有碰撞器的基本功能，以及 `RectColliderComponent`，它将添加包围矩形形状碰撞器的特定功能。然后可以根据正在开发的游戏需求添加新的碰撞器类型。

接下来是特定碰撞器的基类，`ColliderComponent`。

## 编写 `ColliderComponent` 类

在`Header Files/GameObjects`过滤器中创建一个新的头文件，命名为`ColliderComponent.h`，并添加以下代码：

```cpp
#pragma once
#include "Component.h"
#include <iostream>
class ColliderComponent : public Component
{
private:
    string m_Type = "collider";
    bool m_Enabled = false;
public:
    /****************************************************
    *****************************************************
    From Component interface
    *****************************************************
    *****************************************************/
    string Component::getType() {
        return m_Type;
    }
    void Component::disableComponent() {
        m_Enabled = false;
    }
    void Component::enableComponent() {
        m_Enabled = true;
    }
    bool Component::enabled() {
        return m_Enabled;
    }
   void Component::start(GameObjectSharer* gos, GameObject* self)
   {

    }
};
```

`ColliderComponent`类从`Component`类继承。在前面的代码中，你可以看到`m_Type`成员变量被初始化为`"collider"`，而`m_Enabled`被初始化为`false`。

在`public`部分，代码覆盖了`Component`类的纯虚函数。研究它们，以便熟悉它们，因为它们在所有组件类中都以非常相似的方式工作。`getType`函数返回`m_Type`。`disableComponent`函数将`m_Enabled`设置为`false`。`enableComponent`函数将`m_Enabled`设置为`true`。`enabled`函数返回`m_Enabled`的值。`start`函数没有代码，但将被许多更具体的基于组件的类覆盖。

在`Source Files/GameObjects`过滤器中创建一个新的源文件，命名为`ColliderComponent.cpp`，并添加以下代码：

```cpp
/*
All Functionality in ColliderComponent.h
*/
```

我在`ColliderComponent.cpp`中添加了前面的注释，以提醒自己所有功能都在头文件中。

## 编写 RectColliderComponent 类

在`Header Files/GameObjects`过滤器中创建一个新的头文件，命名为`RectColliderComponent.h`，并添加以下代码：

```cpp
#pragma once
#include "ColliderComponent.h"
#include <SFML/Graphics.hpp>
using namespace sf;
class RectColliderComponent : public ColliderComponent
{
private:
    string m_SpecificType = "rect";
    FloatRect m_Collider;
    string m_Tag = "";
public:
    RectColliderComponent(string name);
    string getColliderTag();
    void setOrMoveCollider(
        float x, float y, float width, float height);

    FloatRect& getColliderRectF();
    /****************************************************
    *****************************************************
    From Component interface base class
    *****************************************************
    *****************************************************/
    string getSpecificType() {
        return m_SpecificType;
    }

    void Component::start(
        GameObjectSharer* gos, GameObject* self) {}
};
```

`RectColliderComponent`类从`ColliderComponent`类继承。它有一个`m_SpecificType`变量，初始化为`"rect"`。现在可以查询向量中的任何`RectColliderComponent`实例，该向量包含通用的`Component`实例，并确定它具有类型`"collider"`和特定类型`"rect"`。所有基于组件的类都将具有此功能，因为这是`Component`类的纯虚函数所提供的。

还有一个名为`m_Collider`的`FloatRect`实例，它将存储此碰撞器的坐标。

在`public`部分，我们可以查看构造函数。注意，它接收一个`string`。传入的值将是标识此`RectColliderComponent`附加到的游戏对象类型的文本，例如入侵者、子弹或玩家的飞船。这样就可以确定哪些类型的对象相互碰撞了。

在重写函数之前还有三个函数；记下它们的名称和参数，然后我们将在编写它们的定义时稍后讨论它们。

注意，`getSpecificType`函数定义返回`m_SpecificType`。

在`Source Files/GameObjects`过滤器中创建一个新的源文件，命名为`RectColliderComponent.cpp`，并添加以下代码：

```cpp
#include "RectColliderComponent.h"
RectColliderComponent::RectColliderComponent(string name) {
    m_Tag = "" + name;
}
string RectColliderComponent::getColliderTag() {
    return m_Tag;
}
void RectColliderComponent::setOrMoveCollider(
    float x, float y, float width, float height) {

    m_Collider.left = x;
    m_Collider.top = y;
    m_Collider.width = width;
    m_Collider.height = height;
}
FloatRect& RectColliderComponent::getColliderRectF() {
    return m_Collider;
}
```

在构造函数中，传入的`string`值被分配给`m_Tag`变量，而`getColliderTag`函数则通过类的实例使该值可用。

`setOrMoveCollider`函数将`m_Collider`定位到作为参数传入的坐标。

`getColliderRectF`函数返回对`m_Collider`的引用。这非常适合使用`FloatRect`类的`intersects`函数与另一个碰撞器进行碰撞测试。

我们现在已经完成了碰撞器，可以继续进行图形处理。

# 编写图形组件

Space Invaders ++游戏将只有一种特定的图形组件。它被称为`StandardGraphicsComponent`。与碰撞器组件一样，我们将实现一个基本的`GraphicsComponent`类，以便于将来添加其他图形相关组件。例如，在经典的太空侵略者街机版本中，侵略者会通过两个动画帧上下摆动手臂。一旦你了解了`StandardGraphicsComponent`的工作原理，你将能够轻松地编写另一个类（可能是`AnimatedGraphicsComponent`），它每隔半秒左右使用不同的`Sprite`实例绘制自己。你也可以有一个具有着色器（可能是`ShaderGraphicsComponent`）的图形组件，以实现快速和酷炫的效果。除了这些之外，还有更多可能性。

## 编写`GraphicsComponent`类

在`Header Files/GameObjects`筛选器中创建一个新的头文件，命名为`GraphicsComponent.h`，并添加以下代码：

```cpp
#pragma once
#include "Component.h"
#include "TransformComponent.h"
#include <string>
#include <SFML/Graphics.hpp>
#include "GameObjectSharer.h"
#include <iostream>
using namespace sf;
using namespace std;
class GraphicsComponent : public Component {
private:
    string m_Type = "graphics";
    bool m_Enabled = false;
public:
    virtual void draw(
        RenderWindow& window,
        shared_ptr<TransformComponent> t) = 0;
    virtual void initializeGraphics(
        string bitmapName,
        Vector2f objectSize) = 0;
    /****************************************************
    *****************************************************
    From Component interface
    *****************************************************
    *****************************************************/
    string Component::getType() {
        return m_Type;
    }
    void Component::disableComponent() {
        m_Enabled = false;
    }
    void Component::enableComponent() {
        m_Enabled = true;
    }
    bool Component::enabled() {
        return m_Enabled;
    }
    void Component::start(
        GameObjectSharer* gos, GameObject* self) {}
};
```

之前的大部分代码实现了`Component`类的纯虚函数。对于`GraphicsComponent`类来说，新的是`draw`函数，它有两个参数。第一个参数是`RenderWindow`实例的引用，以便组件可以绘制自己，而第二个参数是`GameObject`的`TransformComponent`实例的共享智能指针，以便在游戏的每一帧可以访问诸如位置和缩放等关键数据。

`initializeGraphics`函数也是`GraphicsComponent`类中新增的，它也有两个参数。第一个是一个`string`值，表示要使用的图形文件的文件名，而第二个是一个`Vector2f`实例，它将代表游戏世界中对象的大小。

这两个函数都是纯虚函数，这使得`GraphicsComponent`类成为抽象类。任何从`GraphicsComponent`继承的类都需要实现这些函数。在下一节中，我们将看到`StandardGraphicsComponent`是如何做到这一点的。

在`Source Files/GameObjects`筛选器中创建一个新的源文件，命名为`GraphicsComponent.cpp`，并添加以下代码：

```cpp
/*
All Functionality in GraphicsComponent.h
*/
```

之前的注释是一个提醒，说明代码都在相关的头文件中。

## 编写`StandardGraphicsComponent`类

在`Header Files/GameObjects`筛选器中创建一个新的头文件，命名为`StandardGraphicsComponent.h`，并添加以下代码：

```cpp
#pragma once
#include "Component.h"
#include "GraphicsComponent.h"
#include <string>
class Component;
class StandardGraphicsComponent : public GraphicsComponent {
private:
    sf::Sprite m_Sprite;
    string m_SpecificType = "standard";
public:
    /****************************************************
    *****************************************************
    From Component interface base class
    *****************************************************
    *****************************************************/
    string Component::getSpecificType() {
        return m_SpecificType;
    }

    void Component::start(
        GameObjectSharer* gos, GameObject* self) {
    }
    /****************************************************
    *****************************************************
    From GraphicsComponent
    *****************************************************
    *****************************************************/
    void draw(
        RenderWindow& window,
        shared_ptr<TransformComponent> t) override;
    void initializeGraphics(
        string bitmapName,
        Vector2f objectSize) override;
};
```

`StandardGraphicsComponent`类有一个`Sprite`成员。它不需要一个`Texture`实例，因为每个帧都会从`BitmapStore`类中获取。这个类还重写了`Component`和`GraphicsComponent`类中所需的所有函数。

让我们编码两个纯虚函数`draw`和`initializeGraphics`的实现。

在`Source Files/GameObjects`筛选器中创建一个新的源文件，命名为`StandardGraphicsComponent.cpp`，并添加以下代码：

```cpp
#include "StandardGraphicsComponent.h"
#include "BitmapStore.h"
#include <iostream>
void StandardGraphicsComponent::initializeGraphics(
    string bitmapName,
    Vector2f objectSize)
{
    BitmapStore::addBitmap("graphics/" + bitmapName + ".png");
    m_Sprite.setTexture(BitmapStore::getBitmap(
        "graphics/" + bitmapName + ".png"));
    auto textureSize = m_Sprite.getTexture()->getSize();
    m_Sprite.setScale(float(objectSize.x) / textureSize.x, 
        float(objectSize.y) / textureSize.y);    
    m_Sprite.setColor(sf::Color(0, 255, 0)); 
}
void StandardGraphicsComponent::draw(
    RenderWindow& window,
    shared_ptr<TransformComponent> t)
{
    m_Sprite.setPosition(t->getLocation());
    window.draw(m_Sprite);
}
```

在`initializeGraphics`函数中，调用了`BitmapStore`类的`addBitmap`函数，并将图像的文件路径以及游戏世界中对象的尺寸传递进去。

接下来，检索刚刚添加到`BitmapStore`类的`Texture`实例，并将其设置为`Sprite`的图像。随后，将`getTexture`和`getSize`两个函数串联起来以获取纹理的尺寸。

下一条代码使用`setScale`函数使`Sprite`与纹理大小相同，而纹理的大小被设置为游戏世界中此对象的尺寸。

然后，`setColor`函数为`Sprite`应用绿色色调。这给它增添了一丝复古的感觉。

在`draw`函数中，使用`setPosition`和`TransformComponent`的`getLocation`函数将`Sprite`移动到指定位置。接下来，我们将编码`TransformComponent`类。

最后一行代码将`Sprite`绘制到`RenderWindow`。

# 编码 TransformComponent 类

在`Header Files/GameObjects`筛选器中创建一个新的头文件，命名为`TransformComponent.h`，并添加以下代码：

```cpp
#pragma once
#include "Component.h"
#include<SFML/Graphics.hpp>
using namespace sf;
class Component;
class TransformComponent : public Component {
private:
    const string m_Type = "transform";
    Vector2f m_Location;
    float m_Height;
    float m_Width;
public:
    TransformComponent(
        float width, float height, Vector2f location);
    Vector2f& getLocation();
    Vector2f getSize();
    /****************************************************
    *****************************************************
    From Component interface
    *****************************************************
    *****************************************************/
    string Component::getType()
    {
        return m_Type;
    }
    string Component::getSpecificType()
    {
        // Only one type of Transform so just return m_Type
        return m_Type;
    }
    void Component::disableComponent(){}
    void Component::enableComponent(){}
    bool Component::enabled()
    {
        return false;
    }
    void Component::start(GameObjectSharer* gos, GameObject* self)    {}
};
```

此类有一个`Vector2f`用于存储游戏世界中对象的定位，一个`float`用于存储高度，另一个`float`用于存储宽度。

在`public`部分，有一个构造函数，我们将使用它来设置此类实例，以及两个函数`getLocation`和`getSize`，我们将使用它们来共享对象的定位和尺寸。我们在编码`StandardGraphicsComponent`类时已经使用了这些函数。

`TransformComponent.h`文件中的剩余代码是`Component`类的实现。

在`Source Files/GameObjects`筛选器中创建一个新的源文件，命名为`TransformComponent.cpp`，并添加以下代码：

```cpp
#include "TransformComponent.h"
TransformComponent::TransformComponent(
    float width, float height, Vector2f location)
{
    m_Height = height;
    m_Width = width;
    m_Location = location;
}
Vector2f& TransformComponent::getLocation() 
{
    return m_Location;
}
Vector2f TransformComponent::getSize() 
{
    return Vector2f(m_Width, m_Height);
}
```

实现此类中的三个函数很简单。构造函数接收一个尺寸和一个位置，并初始化相应的成员变量。当请求时，`getLocation`和`getSize`函数返回这些数据。请注意，值是通过引用返回的，因此它们可以被调用代码修改。

接下来，我们将编码所有与更新相关的组件。

# 编码更新组件

如你所料，我们将编写一个继承自`Component`类的`UpdateComponent`类。它将包含每个`UpdateComponent`所需的所有功能，然后我们将编写从`UpdateComponent`派生的类。这些类将包含针对游戏中单个对象的功能。对于这个游戏，我们将有`BulletUpdateComponent`、`InvaderUpdateComponent`和`PlayerUpdateComponent`。当你在自己的项目中工作时，如果你想创建一个以特定独特方式行为的游戏对象，只需为它编写一个新的基于更新的组件，然后你就可以开始了。基于更新的组件定义行为。

## 编写 UpdateComponent 类

在`Header Files/GameObjects`筛选器中创建一个新的头文件，命名为`UpdateComponent.h`，并添加以下代码：

```cpp
#pragma once
#include "Component.h"
class UpdateComponent : public Component
{
private:
    string m_Type = "update";
    bool m_Enabled = false;
public:
    virtual void update(float fps) = 0;

    /****************************************************
    *****************************************************
    From Component interface
    *****************************************************
    *****************************************************/
    string Component::getType() {
        return m_Type;
    }
    void Component::disableComponent() {
        m_Enabled = false;
    }
    void Component::enableComponent() {
        m_Enabled = true;
    }
    bool Component::enabled() {
        return m_Enabled;
    }
    void Component::start(
        GameObjectSharer* gos, GameObject* self) {
    }
};
```

`UpdateComponent`只提供一项功能：`update`函数。这个函数是纯虚函数，因此任何希望成为`UpdateComponent`可用实例的类都必须实现它。

在`Source Files/GameObjects`筛选器中创建一个新的源文件，命名为`UpdateComponent.cpp`，并添加以下代码：

```cpp
/*
All Functionality in UpdateComponent.h
*/
```

这是一个有用的注释，提醒我们这个类的所有代码都在相关的头文件中。

## 编写 BulletUpdateComponent 类

在`Header Files/GameObjects`筛选器中创建一个新的头文件，命名为`BulletUpdateComponent.h`，并添加以下代码：

```cpp
#pragma once
#include "UpdateComponent.h"
#include "TransformComponent.h"
#include "GameObjectSharer.h"
#include "RectColliderComponent.h"
#include "GameObject.h"
class BulletUpdateComponent : public UpdateComponent
{
private:
    string m_SpecificType = "bullet";
    shared_ptr<TransformComponent> m_TC;
    shared_ptr<RectColliderComponent> m_RCC;
    float m_Speed = 75.0f;

    int m_AlienBulletSpeedModifier;
    int m_ModifierRandomComponent = 5;
    int m_MinimumAdditionalModifier = 5;
    bool m_MovingUp = true;
public:
    bool m_BelongsToPlayer = false;
    bool m_IsSpawned = false;
    void spawnForPlayer(Vector2f spawnPosition);
    void spawnForInvader(Vector2f spawnPosition);
    void deSpawn();
    bool isMovingUp();
    /****************************************************
    *****************************************************
    From Component interface base class
    *****************************************************
    *****************************************************/
    string Component::getSpecificType() {
        return m_SpecificType;
    }

    void Component::start(
        GameObjectSharer* gos, GameObject* self) {        
        // Where is this specific invader
        m_TC = static_pointer_cast<TransformComponent>(
            self->getComponentByTypeAndSpecificType(
                "transform", "transform"));
        m_RCC = static_pointer_cast<RectColliderComponent>(
            self->getComponentByTypeAndSpecificType(
                "collider", "rect"));
    }
    /****************************************************
    *****************************************************
    From UpdateComponent
    *****************************************************
    *****************************************************/
    void update(float fps) override;
};
```

如果你想了解子弹的行为/逻辑，你需要花一些时间学习成员变量名称和类型，因为我不会精确解释子弹是如何行为的；我们已经多次覆盖了这些主题。然而，我会指出，有一些变量用于处理基本操作，如移动，还有一些变量用于帮助在特定范围内随机化每颗子弹的速度，以及布尔值用于标识子弹属于玩家还是入侵者。

你现在还不知道但必须在这里学习的关键点是，每个`BulletUpdateComponent`实例将持有对拥有游戏对象的`TransformComponent`实例的共享指针和对拥有游戏对象的`RectColliderComponent`实例的共享指针。

现在，仔细看看重写的`start`函数。在`start`函数中，上述共享指针被初始化。代码通过使用拥有游戏对象的`getComponentByTypeAndSpecificType`函数（`self`是一个指向拥有游戏对象的指针）来实现这一点。我们将在稍后的部分中编码`GameObject`类，包括这个函数。

在`Source Files/GameObjects`筛选器中创建一个新的源文件，命名为`BulletUpdate.cpp`，并添加以下代码：

```cpp
#include "BulletUpdateComponent.h"
#include "WorldState.h"
void BulletUpdateComponent::spawnForPlayer(
    Vector2f spawnPosition)
{
    m_MovingUp = true;
    m_BelongsToPlayer = true;
    m_IsSpawned = true;

    m_TC->getLocation().x = spawnPosition.x;
    // Tweak the y location based on the height of the bullet 
    // The x location is already tweaked to the center of the player
    m_TC->getLocation().y = spawnPosition.y - m_TC->getSize().y;
    // Update the collider
    m_RCC->setOrMoveCollider(m_TC->getLocation().x,
        m_TC->getLocation().y, 
        m_TC->getSize().x, m_TC->getSize().y);
}
void BulletUpdateComponent::spawnForInvader(
    Vector2f spawnPosition)
{
    m_MovingUp = false;
    m_BelongsToPlayer = false;
    m_IsSpawned = true;
    srand((int)time(0));
    m_AlienBulletSpeedModifier = (
        ((rand() % m_ModifierRandomComponent)))  
        + m_MinimumAdditionalModifier;    
    m_TC->getLocation().x = spawnPosition.x;
    // Tweak the y location based on the height of the bullet 
    // The x location already tweaked to the center of the invader
    m_TC->getLocation().y = spawnPosition.y;
    // Update the collider
    m_RCC->setOrMoveCollider(
        m_TC->getLocation().x, m_TC->
        getLocation().y, m_TC->getSize().x, m_TC->getSize().y);
}
void BulletUpdateComponent::deSpawn()
{
    m_IsSpawned = false;
}
bool BulletUpdateComponent::isMovingUp()
{
    return m_MovingUp;
}
void BulletUpdateComponent::update(float fps)
{
    if (m_IsSpawned)
    {    
        if (m_MovingUp)
        {
            m_TC->getLocation().y -= m_Speed * fps;
        }
        else
        {
            m_TC->getLocation().y += m_Speed / 
                m_AlienBulletSpeedModifier * fps;
        }
        if (m_TC->getLocation().y > WorldState::WORLD_HEIGHT 
            || m_TC->getLocation().y < -2)
        {
            deSpawn();
        }
        // Update the collider
        m_RCC->setOrMoveCollider(m_TC->getLocation().x, 
            m_TC->getLocation().y, 
            m_TC->getSize().x, m_TC->getSize().y);
    }
}
```

前两个函数是 `BulletUpdateComponent` 类独有的；它们是 `spawnForPlayer` 和 `spawnForInvader`。这两个函数都为成员变量、变换组件和碰撞器组件准备行动。每个都略有不同。例如，对于玩家拥有的子弹，它被准备从玩家的船顶向上移动，而对于入侵者的子弹，它被准备从入侵者的底部向下移动屏幕。要注意的关键是，所有这些都可以通过变换组件和碰撞器组件的共享指针来实现。此外，请注意，`m_IsSpawned` 布尔值被设置为 true，这使得这个更新组件的 `update` 函数准备好在每一帧调用游戏。

在 `update` 函数中，子弹以适当的速度在屏幕上下移动。它被测试以查看是否已经消失在屏幕顶部或底部，并且碰撞器被更新以包裹当前位置，以便我们可以测试碰撞。

这是我们在这本书中看到的相同逻辑；新的地方是我们用来与其他组成游戏对象的组件通信的共享指针。

子弹只需要被生成并测试碰撞；我们将在下一章中看到如何做。现在，我们将编写入侵者的行为代码。

## 编写 InvaderUpdateComponent 类

在 `Header Files/GameObjects` 过滤器中创建一个新的头文件，命名为 `InvaderUpdateComponent.h`，并添加以下代码：

```cpp
#pragma once
#include "UpdateComponent.h"
#include "TransformComponent.h"
#include "GameObjectSharer.h"
#include "RectColliderComponent.h"
#include "GameObject.h"
class BulletSpawner;
class InvaderUpdateComponent : public UpdateComponent
{
private:
    string m_SpecificType = "invader";
    shared_ptr<TransformComponent> m_TC;
    shared_ptr < RectColliderComponent> m_RCC;
    shared_ptr < TransformComponent> m_PlayerTC;
    shared_ptr < RectColliderComponent> m_PlayerRCC;
    BulletSpawner* m_BulletSpawner;
    float m_Speed = 10.0f;
    bool m_MovingRight = true;
    float m_TimeSinceLastShot;
    float m_TimeBetweenShots = 5.0f;
    float m_AccuracyModifier;
    float m_SpeedModifier = 0.05;
    int m_RandSeed;
public:
    void dropDownAndReverse();
    bool isMovingRight();
    void initializeBulletSpawner(BulletSpawner* 
        bulletSpawner, int randSeed);
    /****************************************************
    *****************************************************
    From Component interface base class
    *****************************************************
    *****************************************************/
    string Component::getSpecificType() {
        return m_SpecificType;
    }
    void Component::start(GameObjectSharer* gos, 
        GameObject* self) {

        // Where is the player?
        m_PlayerTC = static_pointer_cast<TransformComponent>(
            gos->findFirstObjectWithTag("Player")
            .getComponentByTypeAndSpecificType(
                "transform", "transform"));
        m_PlayerRCC = static_pointer_cast<RectColliderComponent>(
            gos->findFirstObjectWithTag("Player")
            .getComponentByTypeAndSpecificType(
                "collider", "rect"));
        // Where is this specific invader
        m_TC = static_pointer_cast<TransformComponent>(
            self->getComponentByTypeAndSpecificType(
                "transform", "transform"));
        m_RCC = static_pointer_cast<RectColliderComponent>(
            self->getComponentByTypeAndSpecificType(
                "collider", "rect"));
    }
    /****************************************************
    *****************************************************
    From UpdateComponent
    *****************************************************
    *****************************************************/
    void update(float fps) override;    
};
```

在类声明中，我们可以看到编写入侵者行为所需的全部功能。有一个指向变换组件的指针，这样入侵者就可以移动，以及一个指向碰撞器组件的指针，这样它就可以更新其位置并被碰撞：

```cpp
shared_ptr<TransformComponent> m_TC;
shared_ptr < RectColliderComponent> m_RCC;
```

有指向玩家变换和碰撞器的指针，这样入侵者可以查询玩家的位置并决定何时射击子弹：

```cpp
shared_ptr < TransformComponent> m_PlayerTC;
shared_ptr < RectColliderComponent> m_PlayerRCC;
```

接下来，有一个 `BulletSpawner` 实例，我们将在下一章中编写。`BulletSpawner` 类将允许入侵者或玩家生成子弹。

接下来是一系列我们将用来控制速度、方向、射击速率、入侵者瞄准的精确度以及发射子弹速度的变量。熟悉它们，因为它们将在函数定义中的相当深入的逻辑中使用：

```cpp
float m_Speed = 10.0f;
bool m_MovingRight = true;
float m_TimeSinceLastShot;
float m_TimeBetweenShots = 5.0f;
float m_AccuracyModifier;
float m_SpeedModifier = 0.05;
int m_RandSeed;
```

接下来，我们可以看到三个新的公共函数，系统中的不同部分可以调用这些函数使入侵者稍微向下移动并改变方向，测试移动方向，并分别传递上述 `BulletSpawner` 类的指针：

```cpp
void dropDownAndReverse();
bool isMovingRight();
void initializeBulletSpawner(BulletSpawner* 
        bulletSpawner, int randSeed);
```

一定要研究 `start` 函数，其中初始化了指向入侵者和玩家的智能指针。现在，我们将编写函数定义。

在 `Source Files/GameObjects` 过滤器中创建一个新的源文件，名为 `InvaderUpdate.cpp`，并添加以下代码：

```cpp
#include "InvaderUpdateComponent.h"
#include "BulletSpawner.h"
#include "WorldState.h"
#include "SoundEngine.h"
void InvaderUpdateComponent::update(float fps)
{
    if (m_MovingRight)
    {
        m_TC->getLocation().x += m_Speed * fps;
    }
    else
    {
        m_TC->getLocation().x -= m_Speed * fps;
    }
    // Update the collider
    m_RCC->setOrMoveCollider(m_TC->getLocation().x, 
        m_TC->getLocation().y, m_TC->getSize().x, m_TC-
      >getSize().y);
    m_TimeSinceLastShot += fps;

    // Is the middle of the invader above the 
   // player +- 1 world units
    if ((m_TC->getLocation().x + (m_TC->getSize().x / 2)) > 
        (m_PlayerTC->getLocation().x - m_AccuracyModifier) &&
        (m_TC->getLocation().x + (m_TC->getSize().x / 2)) < 
        (m_PlayerTC->getLocation().x + 
        (m_PlayerTC->getSize().x + m_AccuracyModifier)))
    {
        // Has the invader waited long enough since the last shot
        if (m_TimeSinceLastShot > m_TimeBetweenShots)
        {
            SoundEngine::playShoot();
            Vector2f spawnLocation;
            spawnLocation.x = m_TC->getLocation().x + 
                m_TC->getSize().x / 2;
            spawnLocation.y = m_TC->getLocation().y + 
                m_TC->getSize().y;
            m_BulletSpawner->spawnBullet(spawnLocation, false);
            srand(m_RandSeed);
            int mTimeBetweenShots = (((rand() % 10))+1) / 
                WorldState::WAVE_NUMBER;
            m_TimeSinceLastShot = 0;            
        }
    }
}
void InvaderUpdateComponent::dropDownAndReverse()
{
    m_MovingRight = !m_MovingRight;
    m_TC->getLocation().y += m_TC->getSize().y;
    m_Speed += (WorldState::WAVE_NUMBER) + 
        (WorldState::NUM_INVADERS_AT_START 
       - WorldState::NUM_INVADERS) 
        * m_SpeedModifier;
}
bool InvaderUpdateComponent::isMovingRight()
{
    return m_MovingRight;
}
void InvaderUpdateComponent::initializeBulletSpawner(
    BulletSpawner* bulletSpawner, int randSeed)
{
    m_BulletSpawner = bulletSpawner;
    m_RandSeed = randSeed;
    srand(m_RandSeed);
    m_TimeBetweenShots = (rand() % 15 + m_RandSeed);
    m_AccuracyModifier = (rand() % 2);
    m_AccuracyModifier += 0 + static_cast <float> (
        rand()) / (static_cast <float> (RAND_MAX / (10)));
}
```

这段代码很多。实际上，其中没有我们之前没有见过的 C++ 代码。它只是控制入侵者行为的逻辑。让我们概述一下它所做的一切，并方便地重新打印部分代码。

### 解释 `update` 函数

第一个 `if` 和 `else` 块根据需要将入侵者向右或向左移动每一帧：

```cpp
void InvaderUpdateComponent::update(float fps)
{
    if (m_MovingRight)
    {
        m_TC->getLocation().x += m_Speed * fps;
    }
    else
    {
        m_TC->getLocation().x -= m_Speed * fps;
    }
```

接下来，将碰撞器更新到新位置：

```cpp
    // Update the collider
    m_RCC->setOrMoveCollider(m_TC->getLocation().x, 
        m_TC->getLocation().y, m_TC->getSize().x, m_TC 
      ->getSize().y);
```

这段代码追踪自上次入侵者开火以来已经过去的时间，然后测试玩家是否位于入侵者左侧或右侧一个世界单位的位置（+ 或 - 用于随机精度修正，使得每个入侵者都略有不同）：

```cpp
   m_TimeSinceLastShot += fps;

    // Is the middle of the invader above the 
   // player +- 1 world units
    if ((m_TC->getLocation().x + (m_TC->getSize().x / 2)) > 
        (m_PlayerTC->getLocation().x - m_AccuracyModifier) &&
        (m_TC->getLocation().x + (m_TC->getSize().x / 2)) < 
        (m_PlayerTC->getLocation().x + 
        (m_PlayerTC->getSize().x + m_AccuracyModifier)))
    {
```

在前面的 `if` 测试中，另一个测试确保入侵者自上次射击以来已经等待了足够长的时间。如果是这样，那么就会开火。播放声音，计算子弹的生成位置，调用 `BulletSpawner` 实例的 `spawnBullet` 函数，并计算下一次射击前的新随机等待时间：

```cpp
        // Has the invader waited long enough since the last shot
        if (m_TimeSinceLastShot > m_TimeBetweenShots)
        {
            SoundEngine::playShoot();
            Vector2f spawnLocation;
            spawnLocation.x = m_TC->getLocation().x + 
                m_TC->getSize().x / 2;
            spawnLocation.y = m_TC->getLocation().y + 
                m_TC->getSize().y;
            m_BulletSpawner->spawnBullet(spawnLocation, false);
            srand(m_RandSeed);
            int mTimeBetweenShots = (((rand() % 10))+1) / 
                WorldState::WAVE_NUMBER;
            m_TimeSinceLastShot = 0;            
        }
    }
}
```

`BulletSpawner` 类的详细信息将在下一章中揭晓，但作为一个对未来的预览，它将是一个具有一个名为 `spawnBullet` 的函数的抽象类，并将由 `GameScreen` 类继承。

### 解释 `dropDownAndReverse` 函数

在 `dropDownAndReverse` 函数中，方向被反转，垂直位置增加一个入侵者的高度。此外，入侵者的速度相对于玩家清除的波数和剩余要摧毁的入侵者数量而增加。清除的波数越多，剩余的入侵者越少，入侵者的移动速度就越快：

```cpp
void InvaderUpdateComponent::dropDownAndReverse()
{
    m_MovingRight = !m_MovingRight;
    m_TC->getLocation().y += m_TC->getSize().y;
    m_Speed += (WorldState::WAVE_NUMBER) + 
        (WorldState::NUM_INVADERS_AT_START 
      - WorldState::NUM_INVADERS) 
        * m_SpeedModifier;
}
```

下一个函数很简单，但为了完整性而包含在内。

### 解释 `isMovingRight` 函数

这段代码简单地提供了访问当前移动方向的方法：

```cpp
bool InvaderUpdateComponent::isMovingRight()
{
    return m_MovingRight;
}
```

它将用于测试是否需要检查屏幕左侧（当向左移动时）或右侧（当向右移动时）的碰撞，并允许碰撞触发对 `dropDownAndReverse` 函数的调用。

### 解释 `initializeBulletSpawner` 函数

我已经提到过，`BulletSpawner` 类是抽象的，将由 `GameScreen` 类实现。当调用 `GameScreen` 类的 `initialize` 函数时，这个 `initializeBulletSpawner` 函数将在每个入侵者上被调用。正如你所看到的，第一个参数是 `BulletSpawner` 实例的指针。这使每个 `InvaderUpdateComponent` 都能够调用 `spawnBullet` 函数：

```cpp
void InvaderUpdateComponent::initializeBulletSpawner(
    BulletSpawner* bulletSpawner, int randSeed)
{
    m_BulletSpawner = bulletSpawner;
    m_RandSeed = randSeed;
    srand(m_RandSeed);
    m_TimeBetweenShots = (rand() % 15 + m_RandSeed);
    m_AccuracyModifier = (rand() % 2);
    m_AccuracyModifier += 0 + static_cast <float> (
        rand()) / (static_cast <float> (RAND_MAX / (10)));
}
```

`initializeBulletSpawner` 函数中的其余代码设置了使每个入侵者与其他入侵者略有不同行为的随机值。

## 编写 `PlayerUpdateComponent` 类

在 `Header Files/GameObjects` 过滤器中创建一个新的头文件，命名为 `PlayerUpdateComponent.h`，并添加以下代码：

```cpp
#pragma once
#include "UpdateComponent.h"
#include "TransformComponent.h"
#include "GameObjectSharer.h"
#include "RectColliderComponent.h"
#include "GameObject.h"
class PlayerUpdateComponent : public UpdateComponent
{
private:
    string m_SpecificType = "player";
    shared_ptr<TransformComponent> m_TC;
    shared_ptr<RectColliderComponent> m_RCC;
    float m_Speed = 50.0f;
    float m_XExtent = 0;
    float m_YExtent = 0;
    bool m_IsHoldingLeft = false;
    bool m_IsHoldingRight = false;
    bool m_IsHoldingUp = false;
    bool m_IsHoldingDown = false;
public:
    void updateShipTravelWithController(float x, float y);
    void moveLeft();
    void moveRight();
    void moveUp();
    void moveDown();
    void stopLeft();
    void stopRight();
    void stopUp();
    void stopDown();
    /****************************************************
    *****************************************************
    From Component interface base class
    *****************************************************
    *****************************************************/
    string Component::getSpecificType() {
        return m_SpecificType;
    }
    void Component::start(GameObjectSharer* gos, GameObject* self) {        
        m_TC = static_pointer_cast<TransformComponent>(self->
            getComponentByTypeAndSpecificType(
                "transform", "transform"));
        m_RCC = static_pointer_cast<RectColliderComponent>(self->
            getComponentByTypeAndSpecificType(
                "collider", "rect"));        
    }
    /****************************************************
    *****************************************************
    From UpdateComponent
    *****************************************************
    *****************************************************/
    void update(float fps) override;
};
```

在 `PlayerUpdateComponent` 类中，我们拥有所有必要的布尔变量来跟踪玩家是否按下了键盘键，以及可以切换这些布尔值的函数。我们之前没有见过像 `m_XExtent` 和 `M_YExtent float` 类型变量这样的东西，我们将在查看它们在函数定义中的使用时解释它们。

注意，就像 `BulletUpdateComponent` 和 `InvaderUpdateComponent` 类一样，我们为这个游戏对象的变换和碰撞组件使用了共享指针。正如我们所期待的，这些共享指针在 `start` 函数中被初始化。

在 `Source Files/GameObjects` 过滤器中创建一个新的源文件，命名为 `PlayerUpdate.cpp`，并添加以下代码：

```cpp
#include "PlayerUpdateComponent.h"
#include "WorldState.h"
void PlayerUpdateComponent::update(float fps)
{
    if (sf::Joystick::isConnected(0))
    {
        m_TC->getLocation().x += ((m_Speed / 100) 
            * m_XExtent) * fps;
        m_TC->getLocation().y += ((m_Speed / 100) 
            * m_YExtent) * fps;        
    }
    // Left and right
    if (m_IsHoldingLeft)
    {
        m_TC->getLocation().x -= m_Speed * fps;
    }
    else if (m_IsHoldingRight)
    {
        m_TC->getLocation().x += m_Speed * fps;
    }
    // Up and down
    if (m_IsHoldingUp)
    {
        m_TC->getLocation().y -= m_Speed * fps;
    }
    else if (m_IsHoldingDown)
    {
        m_TC->getLocation().y += m_Speed * fps;
    }

    // Update the collider
    m_RCC->setOrMoveCollider(m_TC->getLocation().x, 
        m_TC->getLocation().y, m_TC->getSize().x, 
        m_TC->getSize().y);

    // Make sure the ship doesn't go outside the allowed area
    if (m_TC->getLocation().x >
        WorldState::WORLD_WIDTH - m_TC->getSize().x) 
    {
        m_TC->getLocation().x = 
            WorldState::WORLD_WIDTH - m_TC->getSize().x;
    }
    else if (m_TC->getLocation().x < 0)
    {
        m_TC->getLocation().x = 0;
    }
    if (m_TC->getLocation().y > 
        WorldState::WORLD_HEIGHT - m_TC->getSize().y)
    {
        m_TC->getLocation().y = 
            WorldState::WORLD_HEIGHT - m_TC->getSize().y;
    }
    else if (m_TC->getLocation().y < 
        WorldState::WORLD_HEIGHT / 2)
    {
        m_TC->getLocation().y = 
            WorldState::WORLD_HEIGHT / 2;
    }
}    
void PlayerUpdateComponent::
    updateShipTravelWithController(float x, float y)
{
    m_XExtent = x;
    m_YExtent = y;
}
void PlayerUpdateComponent::moveLeft()
{
    m_IsHoldingLeft = true;
    stopRight();
}
void PlayerUpdateComponent::moveRight()
{
    m_IsHoldingRight = true;
    stopLeft();
}
void PlayerUpdateComponent::moveUp()
{
    m_IsHoldingUp = true;
    stopDown();
}
void PlayerUpdateComponent::moveDown()
{
    m_IsHoldingDown = true;
    stopUp();
}
void PlayerUpdateComponent::stopLeft()
{
    m_IsHoldingLeft = false;
}
void PlayerUpdateComponent::stopRight()
{
    m_IsHoldingRight = false;
}
void PlayerUpdateComponent::stopUp()
{
    m_IsHoldingUp = false;
}
void PlayerUpdateComponent::stopDown()
{
    m_IsHoldingDown = false;
}
```

在更新函数的第一个 `if` 块中，条件是 `sf::Joystick::isConnected(0)`。当玩家将游戏手柄插入 USB 端口时，此条件返回 true。在 `if` 块内部，变换组件的水平和垂直位置都被改变了：

```cpp
…((m_Speed / 100) * m_YExtent) * fps;
```

上述代码在将目标速度乘以 `m_YExtent` 之前将其除以 100。`m_XExtent` 和 `m_YExtent` 变量将在每一帧更新，以保存表示玩家在水平和垂直方向上移动游戏手柄摇杆的程度。值的范围是从 -100 到 100，因此上述代码的效果是当摇杆位于任何全范围或该范围的分数之一时，以全速移动变换组件；当它部分位于中心（完全不移动）和全范围之间时，则以该速度的分数移动。这意味着如果玩家选择使用游戏手柄而不是键盘，他们将能够更精细地控制飞船的速度。

我们将在*第二十二章*中看到更多关于游戏手柄操作细节，*使用游戏对象和构建游戏*。

`update` 函数的其余部分响应代表玩家按下的或释放的键盘键的布尔变量。

在处理游戏手柄和键盘之后，碰撞组件被移动到新位置，一系列的 `if` 块确保玩家飞船不会移出屏幕或超过屏幕中间的上方点。

下一个函数是 `updateShipTravelWithController` 函数；当控制器被插入时，它将更新每一帧拇指摇杆移动或静止的程度。

剩余的函数更新表示是否使用键盘按键来移动飞船的布尔值。请注意，更新组件不处理发射子弹。我们本来可以在这里处理它，而且有些游戏可能出于某些原因这样做。在这个游戏中，从`GameInputHandler`类中处理射击子弹要直接一些。正如我们将在*第二十二章*中看到的那样，`GameInputHandler`类将调用所有让`PlayerUpdateComponent`类知道游戏手柄和键盘发生什么的函数。在前一章中，我们在`GameInputHandler`类中编写了键盘响应的基本代码。

现在，让我们来编写`GameObject`类，它将包含所有各种组件实例。

# 编码 GameObject 类

我将在这个课程中非常详细地讲解代码，因为它对于其他所有课程的工作原理至关重要。然而，我认为你们通过查看整个代码并首先研究它，也会有所收获。考虑到这一点，在`Header Files/GameObjects`过滤器中创建一个新的头文件，命名为`GameObject.h`，并添加以下代码：

```cpp
#pragma once
#include <SFML/Graphics.hpp>
#include <vector>
#include <string>
#include "Component.h"
#include "GraphicsComponent.h"
#include "GameObjectSharer.h"
#include "UpdateComponent.h"
class GameObject {
private:
    vector<shared_ptr<Component>> m_Components;
    string m_Tag;
    bool m_Active = false;
    int m_NumberUpdateComponents = 0;
    bool m_HasUpdateComponent = false;
    int m_FirstUpdateComponentLocation = -1;
    int m_GraphicsComponentLocation = -1;
    bool m_HasGraphicsComponent = false;
    int m_TransformComponentLocation = -1;
    int m_NumberRectColliderComponents = 0;
    int m_FirstRectColliderComponentLocation = -1;
    bool m_HasCollider = false;
public:
    void update(float fps);
    void draw(RenderWindow& window);
    void addComponent(shared_ptr<Component> component);
    void setActive();
    void setInactive();
    bool isActive();
    void setTag(String tag);
    string getTag();
    void start(GameObjectSharer* gos);
    // Slow only use in init and start
    shared_ptr<Component> getComponentByTypeAndSpecificType(
        string type, string specificType);
    FloatRect& getEncompassingRectCollider();
    bool hasCollider();
    bool hasUpdateComponent();
    string getEncompassingRectColliderTag();
    shared_ptr<GraphicsComponent> getGraphicsComponent();
    shared_ptr<TransformComponent> getTransformComponent();
    shared_ptr<UpdateComponent> getFirstUpdateComponent();
};
```

在前面的代码中，请务必仔细检查变量、类型、函数名及其参数。

在`Source Files/GameObjects`过滤器中创建一个新的源文件，命名为`GameObject.cpp`，然后研究并添加以下代码：

```cpp
#include "DevelopState.h"
#include "GameObject.h"
#include <iostream> 
#include "UpdateComponent.h"
#include "RectColliderComponent.h"
void GameObject::update(float fps)
{
    if (m_Active && m_HasUpdateComponent)
    {
        for (int i = m_FirstUpdateComponentLocation; i < 
            m_FirstUpdateComponentLocation + 
            m_NumberUpdateComponents; i++) 
        {
            shared_ptr<UpdateComponent> tempUpdate =
                static_pointer_cast<UpdateComponent>(
             m_Components[i]);
            if (tempUpdate->enabled()) 
            {
                tempUpdate->update(fps);
            }
        }
    }
}
void GameObject::draw(RenderWindow& window)
{
    if (m_Active && m_HasGraphicsComponent)
    {
        if (m_Components[m_GraphicsComponentLocation]->enabled())
        {
            getGraphicsComponent()->draw(window, 
                getTransformComponent());
        }
    }
}
shared_ptr<GraphicsComponent> GameObject::getGraphicsComponent() 
{
    return static_pointer_cast<GraphicsComponent>(
        m_Components[m_GraphicsComponentLocation]);
}
shared_ptr<TransformComponent> GameObject::getTransformComponent() 
{
    return static_pointer_cast<TransformComponent>(
        m_Components[m_TransformComponentLocation]);
}
void GameObject::addComponent(shared_ptr<Component> component)
{
    m_Components.push_back(component);
    component->enableComponent();

   if (component->getType() == "update") 
    {
        m_HasUpdateComponent = true;
        m_NumberUpdateComponents++;
        if (m_NumberUpdateComponents == 1) 
        {
            m_FirstUpdateComponentLocation = 
                m_Components.size() - 1;
        }
    }
    else if (component->getType() == "graphics") 
    {
        // No iteration in the draw method required
        m_HasGraphicsComponent = true;
        m_GraphicsComponentLocation = m_Components.size() - 1;
    }
    else if (component->getType() == "transform") 
    {
        // Remember where the Transform component is
        m_TransformComponentLocation = m_Components.size() - 1;
    }
    else if (component->getType() == "collider" && 
        component->getSpecificType() == "rect") 
    {
        // Remember where the collider component(s) is
        m_HasCollider = true;
        m_NumberRectColliderComponents++;
        if (m_NumberRectColliderComponents == 1) 
        {
            m_FirstRectColliderComponentLocation = 
                m_Components.size() - 1;
        }
    }    
}
void GameObject::setActive()
{
    m_Active = true;
}
void GameObject::setInactive()
{
    m_Active = false;
}
bool GameObject::isActive()
{
    return m_Active;
}
void GameObject::setTag(String tag)
{
    m_Tag = "" + tag;
}
std::string GameObject::getTag()
{
    return m_Tag;
}
void GameObject::start(GameObjectSharer* gos) 
{
    auto it = m_Components.begin();
    auto end = m_Components.end();
    for (it;
        it != end;
        ++it)
    {
        (*it)->start(gos, this);
    }
}
// Slow - only use in start function
shared_ptr<Component> GameObject::
   getComponentByTypeAndSpecificType(
    string type, string specificType) {
    auto it = m_Components.begin();
    auto end = m_Components.end();
    for (it;
        it != end;
        ++it)
    {
        if ((*it)->getType() == type)
        {
            if ((*it)->getSpecificType() == specificType)
            {
                return  (*it);
            }
        }
    }
    #ifdef debuggingErrors        
        cout << 
            "GameObject.cpp::getComponentByTypeAndSpecificType-" 
            << "COMPONENT NOT FOUND ERROR!" 
            << endl;
    #endif
        return m_Components[0];
}
FloatRect& GameObject::getEncompassingRectCollider() 
{
    if (m_HasCollider) 
    {
        return (static_pointer_cast<RectColliderComponent>(
            m_Components[m_FirstRectColliderComponentLocation]))
            ->getColliderRectF();
    }
}
string GameObject::getEncompassingRectColliderTag() 
{
    return static_pointer_cast<RectColliderComponent>(
        m_Components[m_FirstRectColliderComponentLocation])->
        getColliderTag();
}
shared_ptr<UpdateComponent> GameObject::getFirstUpdateComponent()
{
    return static_pointer_cast<UpdateComponent>(
        m_Components[m_FirstUpdateComponentLocation]);
}
bool GameObject::hasCollider() 
{
    return m_HasCollider;
}
bool GameObject::hasUpdateComponent()
{
    return m_HasUpdateComponent;
}
```

小贴士

在继续之前，请务必研究前面的代码。以下解释假设你们对变量名和类型、函数名、参数和返回类型有基本了解。

## 解释 GameObject 类

让我们逐个函数地查看`GameObject`类，并重新打印代码，以便于讨论。

### 解释 update 函数

`update`函数在游戏循环的每一帧为每个游戏对象调用一次。像我们的大多数其他项目一样，需要当前帧率。在`update`函数内部，会进行一个测试，以查看这个`GameObject`实例是否处于活动状态并且有一个更新组件。游戏对象不必有更新组件，尽管在这个项目中所有游戏对象确实都有。

接下来，`update` 函数遍历它拥有的所有组件，从 `m_FirstUpdateComponent` 开始，一直到 `m_FirstUpdateComponent + m_NumberUpdateComponents`。这段代码暗示一个游戏对象可以拥有多个更新组件。这样你可以设计具有行为层的游戏对象。这种行为分层在*第二十二章*，*使用游戏对象和构建游戏*中进一步讨论。在这个项目中，所有游戏对象只有一个更新组件，因此你可以简化（并加快）`update` 函数中的逻辑，但我建议在阅读*第二十二章*，*使用游戏对象和构建游戏*之前保持原样。

正因为一个组件可能是我们创建的许多类型之一，所以我们创建一个临时的更新相关组件（`tempUpdate`），将组件从组件向量转换为 `UpdateComponent`，并调用 `update` 函数。`UpdateComponent` 类的具体派生并不重要；它将实现 `update` 函数，因此 `UpdateComponent` 类型足够具体：

```cpp
void GameObject::update(float fps)
{
    if (m_Active && m_HasUpdateComponent)
    {
        for (int i = m_FirstUpdateComponentLocation; i < 
            m_FirstUpdateComponentLocation + 
            m_NumberUpdateComponents; i++) 
        {
            shared_ptr<UpdateComponent> tempUpdate =
                static_pointer_cast<UpdateComponent>(
             m_Components[i]);
            if (tempUpdate->enabled()) 
            {
                tempUpdate->update(fps);
            }
        }
    }
}
```

当我们在后面的部分到达 `addComponent` 函数时，我们将看到如何初始化各种控制变量，例如 `m_FirstUpdateComponentLocation` 和 `m_NumberOfUpdateComponents`。

### 解释绘制函数

`draw` 函数检查游戏对象是否处于活动状态并且它有一个图形组件。如果确实如此，则检查图形组件是否启用。如果所有这些测试都成功，则调用 `draw` 函数：

```cpp
void GameObject::draw(RenderWindow& window)
{
    if (m_Active && m_HasGraphicsComponent)
    {
        if (m_Components[m_GraphicsComponentLocation]->enabled())
        {
            getGraphicsComponent()->draw(window, 
                getTransformComponent());
        }
    }
}
```

`draw` 函数的结构暗示并非每个游戏对象都必须自己绘制。我在*第十九章*，*游戏编程设计模式 – 开始 Space Invaders ++ 游戏*中提到，你可能希望游戏对象作为不可见的触发区域（没有图形组件）来响应玩家经过它们，或者作为暂时不可见的游戏对象（暂时禁用但具有图形组件）。在这个项目中，所有游戏对象都有一个永久启用的图形组件。

### 解释获取图形组件函数

此函数返回一个指向图形组件的共享指针：

```cpp
shared_ptr<GraphicsComponent> GameObject::getGraphicsComponent() 
{
    return static_pointer_cast<GraphicsComponent>(
        m_Components[m_GraphicsComponentLocation]);
}
```

`getGraphicsComponent` 函数允许任何拥有包含的游戏对象实例的代码访问图形组件。

### 解释获取变换组件函数

此函数返回一个指向变换组件的共享指针：

```cpp
shared_ptr<TransformComponent> GameObject::getTransformComponent() 
{
    return static_pointer_cast<TransformComponent>(
        m_Components[m_TransformComponentLocation]);
}
```

`getTransformComponent` 函数允许任何拥有包含的游戏对象实例的代码访问变换组件。

### 解释添加组件函数

`addComponent` 函数将在下一章中编写的工厂模式类中使用。该函数接收一个指向 `Component` 实例的共享指针。函数内部首先发生的事情是将 `Component` 实例添加到 `m_Components` 向量中。接下来，使用 `enabled` 函数启用该组件。

接下来是一系列 `if` 和 `else if` 语句，用于处理每种可能的组件类型。当识别出组件的类型时，各种控制变量被初始化，以使类中其余部分的逻辑能够正确工作。

例如，如果检测到更新组件，则初始化 `m_HasUpdateComponent`、`m_NumberUpdateComponents` 和 `m_FirstUpdateComponentLocation` 变量。

作为另一个例子，如果检测到具有 `rect` 特定类型的碰撞器组件，则初始化 `m_HasCollider`、`m_NumberRectColliderComponents` 和 `m_FirstRectColliderComponent` 变量：

```cpp
void GameObject::addComponent(shared_ptr<Component> component)
{
    m_Components.push_back(component);
    component->enableComponent();

   if (component->getType() == "update") 
    {
        m_HasUpdateComponent = true;
        m_NumberUpdateComponents++;
        if (m_NumberUpdateComponents == 1) 
        {
            m_FirstUpdateComponentLocation = 
                m_Components.size() - 1;
        }
    }
    else if (component->getType() == "graphics") 
    {
        // No iteration in the draw method required
        m_HasGraphicsComponent = true;
        m_GraphicsComponentLocation = m_Components.size() - 1;
    }
    else if (component->getType() == "transform") 
    {
        // Remember where the Transform component is
        m_TransformComponentLocation = m_Components.size() - 1;
    }
    else if (component->getType() == "collider" && 
        component->getSpecificType() == "rect") 
    {
        // Remember where the collider component(s) is
        m_HasCollider = true;
        m_NumberRectColliderComponents++;
        if (m_NumberRectColliderComponents == 1) 
        {
            m_FirstRectColliderComponentLocation = 
                m_Components.size() - 1;
        }
    }    
}
```

注意，`GameObject` 类在配置或设置实际组件方面不起作用。所有这些都在下一章中我们将编写的工厂模式类中处理。

### 解释获取器和设置器函数

以下代码是一系列非常简单的获取器和设置器：

```cpp
void GameObject::setActive()
{
    m_Active = true;
}
void GameObject::setInactive()
{
    m_Active = false;
}
bool GameObject::isActive()
{
    return m_Active;
}
void GameObject::setTag(String tag)
{
    m_Tag = "" + tag;
}
std::string GameObject::getTag()
{
    return m_Tag;
}
```

前面的获取器和设置器函数提供了有关游戏对象的信息，例如它是否处于活动状态以及它的标签是什么。它们还允许您设置标签并告诉我们游戏对象是否处于活动状态。

### 解释 `start` 函数

`start` 函数非常重要。正如我们在编写所有组件时看到的那样，`start` 函数提供了访问任何游戏对象中任何组件的能力。当所有 `GameObject` 实例都由其组件组成后，将调用 `start` 函数。在下一章中，我们将看到这是如何发生的，以及 `start` 函数在每一个 `GameObject` 实例上被调用的时机。正如我们所见，在 `start` 函数中，它遍历每个组件并共享一个新的类实例，一个 `GameObjectSharer` 实例。这个 `GameObjectSharer` 类将在下一章中编写，并将提供从任何类访问任何组件的能力。我们看到了入侵者需要知道玩家的位置以及当编写各种组件时如何使用 `GameObjectSharer` 参数。当对每个组件调用 `start` 时，也会传入 `this` 指针，以便每个组件可以轻松访问其包含的 `GameObject` 实例：

```cpp
void GameObject::start(GameObjectSharer* gos) 
{
    auto it = m_Components.begin();
    auto end = m_Components.end();
    for (it;
        it != end;
        ++it)
    {
        (*it)->start(gos, this);
    }
}
```

让我们继续到 `getComponentByTypeAndSpecificType` 函数。

### 解释 `getComponentByTypeAndSpecificType` 函数

`getComponentByTypeAndSpecificType` 函数有一个嵌套的 `for` 循环，用于查找与第一个 `string` 参数匹配的组件类型，然后查找第二个 `string` 参数中特定组件类型的匹配项。它返回一个指向基类 `Component` 实例的共享指针。这意味着调用代码需要确切知道返回的是哪种派生 `Component` 类型，以便将其转换为所需类型。这不应该是一个问题，因为他们当然请求了类型和特定类型：

```cpp
// Slow only use in start
shared_ptr<Component> GameObject::getComponentByTypeAndSpecificType(
    string type, string specificType) {
    auto it = m_Components.begin();
    auto end = m_Components.end();
    for (it;
        it != end;
        ++it)
    {
        if ((*it)->getType() == type)
        {
            if ((*it)->getSpecificType() == specificType)
            {
                return  (*it);
            }
        }
    }
    #ifdef debuggingErrors        
        cout << 
            "GameObject.cpp::getComponentByTypeAndSpecificType-" 
            << "COMPONENT NOT FOUND ERROR!" 
            << endl;
    #endif
        return m_Components[0];
}
```

这个函数中的代码相当慢，因此它打算在主游戏循环之外使用。在函数的末尾，如果已经定义了 `debuggingErrors`，代码将向控制台写入错误信息。这是因为，如果执行到达这个点，意味着没有找到匹配的组件，游戏将会崩溃。控制台输出的信息应该使得错误易于查找。崩溃的原因可能是函数被调用时使用了无效的类型或特定类型。

### 解释 `getEncompassingRectCollider` 函数

`getEncompassingRectCollider` 函数检查游戏对象是否有碰撞体，如果有，则将其返回给调用代码：

```cpp
FloatRect& GameObject::getEncompassingRectCollider() 
{
    if (m_HasCollider) 
    {
        return (static_pointer_cast<RectColliderComponent>(
            m_Components[m_FirstRectColliderComponentLocation]))
            ->getColliderRectF();
    }
}
```

值得注意的是，如果你将此项目扩展以处理多种类型的碰撞体，那么这段代码也需要进行修改。

### 解释 `getEncompassingRectColliderTag` 函数

这个简单的函数返回碰撞体的标签。这将有助于确定正在测试碰撞的对象类型：

```cpp
string GameObject::getEncompassingRectColliderTag() 
{
    return static_pointer_cast<RectColliderComponent>(
        m_Components[m_FirstRectColliderComponentLocation])->
        getColliderTag();
}
```

我们还有几个函数需要讨论。

### 解释 `getFirstUpdateComponent` 函数

`getFirstUpdateComponent` 使用 `m_FirstUpdateComponent` 变量来定位更新组件，并将其返回给调用代码：

```cpp
shared_ptr<UpdateComponent> GameObject::getFirstUpdateComponent()
{
    return static_pointer_cast<UpdateComponent>(
        m_Components[m_FirstUpdateComponentLocation]);
}
```

现在我们将简要介绍几个获取器，然后我们就完成了。

### 解释最终的获取器函数

这两个剩余的函数返回一个布尔值（每个），以告知调用代码游戏对象是否有碰撞体和/或更新组件：

```cpp
bool GameObject::hasCollider() 
{
    return m_HasCollider;
}
bool GameObject::hasUpdateComponent()
{
    return m_HasUpdateComponent;
}
```

我们已经完整地编写了 `GameObject` 类。现在我们可以看看如何使用它（以及它将包含的所有组件）。

# 摘要

在本章中，我们已经完成了所有将游戏对象绘制到屏幕上、控制它们的行为以及通过碰撞让它们与其他类交互的代码。从本章中要吸取的最重要的一点不是任何特定基于组件的类是如何工作的，而是实体-组件系统是多么灵活。如果你想创建一个以某种方式行为的游戏对象，就创建一个新的更新组件。如果它需要了解游戏中的其他对象，可以在`start`函数中获取适当的组件指针。如果它需要以某种花哨的方式绘制，比如使用着色器或动画，就在`draw`函数中编写一个执行这些操作的图形组件。如果你需要多个碰撞器，就像我们在《托马斯迟到了》项目中为托马斯和鲍勃做的，这没有任何问题：编写一个新的基于碰撞器的组件。

在下一章中，我们将编写文件输入和输出系统，以及将构建所有游戏对象并将它们与组件组合的工厂类。
