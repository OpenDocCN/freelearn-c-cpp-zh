# 第二章. 你、C++ 和 Cocos2d-x

*本章将针对两种类型的开发者：害怕 C++ 但不会向朋友承认的原始 Cocos2d 开发者，以及从未听说过 Cocos2d 且认为 Objective-C 看起来很奇怪的 C++ 程序员。*

*我将概述 Objective-C 开发者应该注意的主要语法差异，以及在使用 Cocos2d-x 开发时涉及的一些代码风格更改，C++ 开发者应该了解。但首先，让我们快速介绍一下 Cocos2d-x 以及它是什么以及它所涉及的一切。*

您将学习以下主题：

+   Cocos2d-x 是什么以及它能为您做什么

+   如何在 C++ 中创建类

+   如何在 Cocos2d-x 和 C++ 中管理您的对象内存

+   从 Ref 获取的内容

# Cocos2d-x – 简介

那么，什么是 2D 框架呢？如果我要用尽可能少的词来定义它，我会说是在循环中的矩形。

在 Cocos2d-x 的核心，您会发现 `Sprite` 类以及这个类所做的工作，简单来说，就是保持对两个非常重要的矩形的引用。一个是图像（或纹理）矩形，也称为源矩形，另一个是目标矩形。如果您想让图像出现在屏幕中央，您将使用 `Sprite`。您将传递有关图像源是什么以及在哪里显示的信息，以及您想在屏幕上的哪个位置显示它。

对于第一个矩形，即源矩形，不需要做太多工作；但在目标矩形中可以改变很多东西，包括其在屏幕上的位置、大小、不透明度、旋转等。

Cocos2d-x 将负责所有必要的 OpenGL 绘制工作，以显示您想要的位置和方式显示您的图像，并且它将在渲染循环内完成这些工作。您的代码很可能会利用同一个循环来更新自己的逻辑。

几乎任何您能想到的 2D 游戏都可以使用 Cocos2d-x 和一些精灵以及循环来构建。

### 注意

在框架的 3.x 版本中，Cocos2d-x 和其对应版本 Cocos2d 之间存在轻微的分离。它放弃了 CC 前缀，转而使用命名空间，采用了 C++11 的特性，因此它变得更加易于使用。

## 容器

在 Cocos2d-x 中，也很重要的是容器（或节点）的概念。这些都是可以包含精灵（或其他节点）的对象。在某些时候，这非常有用，因为通过改变容器的一些方面，您会自动改变其子节点的某些方面。移动容器，所有子节点都会随之移动。旋转容器，嗯，您应该能想象出来！

容器包括：`Scene`、`Layer` 和 `Sprite`。它们都继承自一个名为 **node** 的基本容器类。每个容器都会有其独特之处，但基本上您将按照以下方式排列它们：

+   `Scene`: 这将包含一个或多个 `Node`，通常是 `Layer` 类型。将应用程序拆分为多个场景很常见；例如，一个用于主菜单，一个用于设置，一个用于实际游戏。技术上，每个场景都将作为你应用程序中的独立实体行为，几乎就像子应用程序一样，你可以在场景之间切换时运行一系列过渡效果。

+   `Layer`: 这很可能会包含 `Sprite`。有一些专门的 `Layer` 对象旨在为你，开发者，节省时间，例如创建菜单（`Menu`）或彩色背景（`LayerColor`）。每个场景可以有多个 `Layer`，但良好的规划通常使这变得不必要。

+   `Sprite`: 这将包含你的图像，并作为子元素添加到由 `Layer` 派生的容器中。在我看来，这是 Cocos2d-x 中最重要的类，如此重要，以至于在应用程序初始化后，当创建了一个 `Scene` 和一个 `Layer` 对象时，你只需使用精灵就能构建整个游戏，而无需在 Cocos2d-x 中使用另一个容器类。

+   `Node`: 这个超级类对所有容器模糊了其自身与 `Layer`，甚至 `Sprite` 之间的界限。它有一系列专门的子类（除了之前提到的那些），例如 `MotionStreak`、`ParallaxNode` 和 `SpriteBatchNode`，仅举几例。经过一些调整，它可以表现得就像 `Layer` 一样。但大多数时候，你会用它来创建自己的专用节点或作为多态中的通用参考。

## 导演和缓存类

在容器之后，是无所不知的 `Director` 和包罗万象的缓存对象。`Director` 对象管理场景，并了解你应用程序的所有信息。你将调用它来获取这些信息，并更改一些事情，如屏幕大小、帧率、缩放因子等等。

缓存是收集器对象。其中最重要的有 `TextureCache`、`SpriteFrameCache` 和 `AnimationCache`。这些负责存储关于我之前提到的两个重要矩形的关键信息。但 Cocos2d-x 中使用的任何重复数据类型都将保存在某种缓存列表中。

`Director` 和所有缓存对象都是单例。这些是特殊类型的类，它们只实例化一次；并且这个实例可以被任何其他对象访问。

## 其他东西

在基本容器、缓存和 `Director` 对象之后，是框架剩余的 90%。在这其中，你会发现：

+   **动作**：动画将通过这些来处理，它们是多么美妙啊！

+   **粒子**：粒子系统，让你的快乐倍增。

+   **专用节点**：用于菜单、进度条、特殊效果、视差效果、瓦片地图等等。

+   **宏、结构和辅助方法**：数百个节省时间的神奇逻辑片段。你不需要知道它们全部，但很可能你会编写一些可以用宏或辅助方法轻松替换的代码，并在后来发现时感到非常愚蠢。

## 你知道 C++吗？

别担心，C 部分很简单。第一个加号过得非常快，但那个第二个加号，哎呀！

记住，这是 C。如果你使用原始 Cocos2d 在 Objective-C 中编码过，即使你大部分时间看到的是在括号中，你也已经熟悉了古老的 C 语言。

但 C++也有类，就像 Objective-C 一样，这些类在接口文件中声明，就像在 Objective-C 中一样。所以，让我们回顾一下 C++类的创建。

# 类接口

这将在`.h`文件中完成。我们将使用文本编辑器来创建这个文件，因为我不想任何代码提示和自动完成功能干扰你学习 C++语法的基础知识。所以至少现在，打开你最喜欢的文本编辑器。让我们创建一个类接口！

# 是时候行动了——创建接口

接口，或头文件，只是一个带有`.h`扩展名的文本文件。

1.  创建一个新的文本文件，并将其保存为`HelloWorld.h`。然后，在顶部输入以下行：

    ```cpp
    #ifndef __HELLOWORLD_H__
    #define __HELLOWORLD_H__
    #include "cocos2d.h" 
    ```

1.  接下来，添加命名空间声明：

    ```cpp
    using namespace cocos2d;
    ```

1.  然后，声明你的类名和任何继承的类名：

    ```cpp
    class HelloWorld : public cocos2d::Layer {

    ```

1.  接下来，我们添加属性和方法：

    ```cpp
    protected:
    int _score;

    public:

        HelloWorld();
        virtual ~HelloWorld();

        virtual bool init();
        static cocos2d::Scene* scene();
        CREATE_FUNC(HelloWorld);
        void update(float dt);
        inline int addTwoIntegers (int one, int two) {
            return one + two;
        }
    };
    ```

1.  我们通过关闭`#ifndef`语句来完成：

    ```cpp
    #endif // __HELLOWORLD_H__
    ```

## *刚才发生了什么？*

你在 C++中创建了一个头文件。让我们回顾一下重要的信息：

+   在 C++中，你包含，而不是导入。Objective-C 中的`import`语句检查是否需要包含某些内容；`include`则不检查。但我们通过在顶部使用定义的巧妙方式完成相同的事情。还有其他方法可以运行相同的检查（例如使用`#pragma once`），但这个是在你创建的任何新的 Xcode C++文件中添加的。

+   你可以通过声明在类中使用的命名空间来使你的生活更简单。这些在有些语言中类似于包。你可能已经注意到，由于命名空间声明，代码中所有对`cocos2d::`的使用都是不必要的。但我想要展示的是，通过添加命名空间声明可以去掉的部分。

+   所以接下来，给你的类起一个名字，你可以选择从其他类继承。在 C++中，你可以有任意多的超类。你必须声明你的超类是否是公开的。

+   你在花括号之间声明你的`public`、`protected`和`private`方法和成员。`HelloWorld`是构造函数，`~HelloWorld`是析构函数（它将执行 Objective-C 中的`dealloc`所做的操作）。

+   `virtual` 关键字与重写有关。当你将一个方法标记为 `virtual` 时，你是在告诉编译器不要将方法的所有权固定下来，而是将其保留在内存中，因为执行将揭示明显的主人。否则，编译器可能会错误地决定一个方法属于超类而不是其继承类。

    此外，将所有析构函数设置为 `virtual` 是一种良好的实践。你只需要在超类中使用一次关键字来标记潜在的覆盖，但通常的做法是在所有子类中重复 `virtual` 关键字，这样开发者就知道哪些方法是覆盖（C++11 添加了一个 `override` 标签，这使得这种区分更加清晰，你将在本书的代码中看到它的例子）。在这种情况下，`init` 来自 `Layer`，而 `HelloWorld` 想要覆盖它。

    ```cpp
        virtual bool init();
    ```

+   噢，是的，在 C++ 中，你必须在你的接口中声明重写。没有例外！

`inline` 方法对你来说可能是新的。这些方法在它们被调用的地方由编译器添加到代码中。所以每次我调用 `addTwoIntegers`，编译器都会用接口中声明的方法的行来替换它。所以 `inline` 方法就像方法内部的表达式一样工作；它们不需要在栈中占用自己的内存。但是如果你在程序中调用了一个两行的 `inline` 方法 50 次，这意味着编译器将向你的代码中添加一百行。

# 类实现

这将在一个 `.cpp` 文件中完成。所以让我们回到我们的文本编辑器，为我们的 `HelloWorld` 类创建实现。

# 是时候采取行动——创建实现

实现是一个具有 `.cpp` 扩展名的文本文件：

1.  创建一个新的文本文件，并将其保存为 `HelloWorld.cpp`。在顶部，让我们先包含我们的头文件：

    ```cpp
    #include "HelloWorld.h"
    ```

1.  接下来，我们实现构造函数和析构函数：

    ```cpp
    HelloWorld::HelloWorld () {
        //constructor
    }

    HelloWorld::~HelloWorld () {
        //destructor
    }
    ```

1.  然后是我们的静态方法：

    ```cpp
    Scene* HelloWorld::scene() {
        auto scene = Scene::create();

        auto layer = HelloWorld::create();

        scene->addChild(layer);

        return scene;
    }
    ```

1.  然后是我们的两个剩余的公共方法：

    ```cpp
    bool HelloWorld::init() {
        // call to super
        if ( !Layer::init() )
        {
            return false;
        }

        //create main loop 
        this->scheduleUpdate();

        return true;
    }

    void HelloWorld::update (float dt) {
        //the main loop
    }
    ```

## *刚才发生了什么？*

我们为我们的 `HelloWorld` 类创建了实现。以下是需要注意的最重要部分：

+   在这里，`HelloWorld::` 作用域解析不是可选的。你接口中声明的每个方法都属于需要正确作用域解析的新类。

+   当调用超类如 `Layer::init()` 时，也需要作用域解析。在标准 C++ 库中没有内置的 `super` 关键字。

+   你使用 `this` 而不是 `self`。当你试图通过指向对象的指针（指针是你在内存中找到实际对象的信息）来访问对象的属性或方法时，使用 `->` 符号。使用 `.`（点）符号通过其实例（构成实际对象的内存块）来访问对象的方法和属性。

+   我们创建了一个 `update` 循环，它通过调用 `scheduleUpdate` 简单地接受一个浮点数作为其 delta 时间值。你将在本书后面的部分看到更多与此相关的选项。

+   如果编译器足够清楚对象的类型，你可以使用`auto`关键字作为对象的类型。

+   当然，`inline`方法不会在类中实现，因为它们只存在于接口中。

至此，语法讲解就到这里。C++是现有最广泛的语言之一，我不希望给你留下我已经涵盖了所有内容的印象。但这是一个由开发者为开发者制作的语言。相信我，你将感到与它一起工作非常自在。

之前列出的信息在我们开始构建游戏后将会更加清晰。但现在，让我们直面这个令人畏惧的大怪物：内存管理。

# 实例化对象并管理内存

Cocos2d-x 中没有**自动引用计数（ARC**），因此忘记内存管理的 Objective-C 开发者可能会在这里遇到问题。然而，关于内存管理的规则在 C++中非常简单：如果你使用`new`，你必须删除。C++11 通过引入特殊的内存管理的指针（这些是`std::unique_ptr`和`std::shared_ptr`）使这一点变得更加容易。

然而，Cocos2d-x 会添加一些其他选项和命令来帮助进行内存管理，类似于我们在 Objective-C（没有 ARC）中使用的那些。这是因为 Cocos2d-x，与 C++不同，与 Objective-C 非常相似，有一个根类。这个框架不仅仅是 Cocos2d 的 C++端口，它还将 Objective-C 的一些概念移植到 C++中，以重新创建其内存管理系统。

Cocos2d-x 有一个`Ref`类，它是框架中每个主要对象的根。它允许框架拥有`autorelease`池和`retain`计数，以及其他 Objective-C 等效功能。

当实例化 Cocos2d-x 对象时，你基本上有两个选项：

+   使用静态方法

+   C++和 Cocos2d-x 风格

## 使用静态方法

使用静态方法是推荐的方式。Objective-C 的三阶段实例化过程，包括`alloc`、`init`和`autorelease`/`retain`，在这里被重新创建。例如，一个扩展`Sprite`的`Player`类可能具有以下方法：

```cpp
Player::Player () {
    this->setPosition  ( Vec2(0,0) );
}

Player* Player::create () {

    auto player = new Player();
    if (player && player->initWithSpriteFrameName("player.png")) {
        player->autorelease();
        return player;
    }
    CC_SAFE_DELETE(player);
    return nullptr;
}
```

对于实例化，你调用静态的`create`方法。它将创建一个新的`Player`对象，作为`Player`的空壳版本。构造函数内部不应该进行任何主要初始化，以防你可能因为实例化过程中的某些失败而需要删除该对象。Cocos2d-x 有一系列用于对象删除和释放的宏，就像之前使用的`CC_SAFE_DELETE`宏一样。

然后，通过其可用方法之一初始化超类。在 Cocos2d-x 中，这些`init`方法返回一个`boolean`值表示成功。现在，你可以开始用一些数据填充`Player`对象。

如果成功，那么在之前的步骤中没有完成的情况下，使用正确的数据初始化你的对象，并以一个`autorelease`对象的形式返回它。

因此，在你的代码中，对象将按照以下方式实例化：

```cpp
auto player = Player::create();
this->addChild(player);//this will retain the object
```

即使 `player` 变量是类的一个成员（比如说，`m_player`），你也不必保留它以保持其作用域。通过将对象添加到某个 Cocos2d-x 列表或缓存中，对象会自动保留。因此，你可以继续通过其指针来引用该内存：

```cpp
m_player = Player::create();
this->addChild(m_player);//this will retain the object
//m_player still references the memory address 
//but does not need to be released or deleted by you
```

## C++ 和 Cocos2d-x 风格

在这个选项中，你会按照以下方式实例化之前的 `Player` 对象：

```cpp
auto player = new Player();
player->initWithSpriteFrameName("player.png");
this->addChild(player);
player->autorelease();
```

在这种情况下，`Player` 可以没有静态方法，并且 `player` 指针在将来不会访问相同的内存，因为它被设置为自动释放（所以它不会长时间存在）。然而，在这种情况下，内存不会泄漏。它仍然会被 Cocos2d-x 列表（`addChild` 命令负责这一点）保留。你仍然可以通过遍历添加到 `this` 的子项列表来访问该内存。

如果你需要指针作为成员属性，你可以使用 `retain()` 而不是 `autorelease()`：

```cpp
m_player = new Player();
m_player->initWithSpriteFrameName("player.png");
this->addChild(m_player);
m_player->retain();
```

然后，在某个时候，你必须释放它；否则，它将会泄漏：

```cpp
m_player->release();
```

硬核的 C++ 开发者可能会选择忘记所有关于 `autorelease` 池的事情，而仅仅使用 `new` 和 `delete`：

```cpp
Player * player = new Player();
player->initWithSpriteFrameName("player.png");
this->addChild(player);
delete player;//This will crash!
```

这将不会起作用。你必须使用 `autorelease`、`retain` 或者让之前的代码不带 `delete` 命令，并希望不会出现任何泄漏。

C++ 开发者必须记住，`Ref` 是由框架管理的。这意味着对象正在被内部添加到缓存和 `autorelease` 池中，即使你可能不希望这种情况发生。例如，当你创建那个 `Player` 精灵时，你使用的 `player.png` 文件将被添加到纹理缓存或精灵帧缓存中。当你将精灵添加到图层时，精灵将被添加到该图层的所有子项列表中，而这个列表将由框架管理。我的建议是，放松并让框架为你工作。

非 C++ 开发者应该记住，任何没有从 `Ref` 派生的类都应该以通常的方式管理，也就是说，如果你 *创建* 一个新对象，你必须在某个时候删除它：

```cpp
MyObject* object = new MyObject();
delete object;
```

# 使用 Ref 得到的东西

使用 `Ref` 你可以得到托管对象。这意味着 `Ref` 派生对象将有一个引用计数属性，该属性将用于确定对象是否应该从内存中删除。每当对象被添加到或从 Cocos2d-x 集合对象中移除时，引用计数都会更新。

例如，Cocos2d-x 附带了一个 `Vector` 集合对象，它通过在对象被添加到或从其中移除时增加和减少引用计数来扩展 C++ 标准库向量 (`std::vector`) 的功能。因此，它只能存储 `Ref` 派生对象。

再次强调，每个 `Ref` 派生类都可以像在 ARC 之前 Objective-C 中管理事物一样进行管理——使用 `retain` 计数和 `autorelease` 池。

然而，C++自带了许多自己非常棒的动态列表类，类似于你在 Java 和 C#中找到的类。但对于`Ref`派生对象，你可能最好使用 Cocos2d-x 管理的列表，或者记得在适用的情况下保留和释放每个对象。如果你创建了一个不扩展`Ref`的类，并且需要将这个类的实例存储在列表容器中，那么请选择标准库中的那些。

在本书接下来的示例中，我将主要在框架内部进行编码，因此你们将有机会看到许多`cocos2d::Vector`的使用示例，例如，但我也将在一些游戏中使用一个或两个`std::vector`实例。

# 总结

希望非 C++开发者现在已经了解到这个语言没有什么可怕的地方，而核心 C++开发者也没有对根类及其保留和自动释放的概念嘲笑得太多。

所有根类为 Java 和 Objective-C 等语言带来的东西将永远是一个无意义的问题。那些在你背后进行的令人毛骨悚然的底层操作无法关闭或控制。它们不是可选的，而根对象这种强制性质自从垃圾收集器等概念首次出现以来就一直困扰着 C++开发者。

话虽如此，`Ref`对象的内存管理非常有帮助，我希望即使是那些最不信任的开发者也会很快学会对此表示感谢。

此外，Cocos2d-x 非常出色。那么，让我们现在就创建一个游戏吧！
