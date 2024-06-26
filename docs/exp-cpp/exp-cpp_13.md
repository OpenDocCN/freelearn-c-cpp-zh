# 第十一章：使用设计模式设计策略游戏

游戏开发是软件工程中最有趣的话题之一。C++在游戏开发中被广泛使用，因为它的效率。然而，由于该语言没有 GUI 组件，因此它被用于后端。在本章中，我们将学习如何在后端设计策略游戏。我们将整合几乎所有我们在之前章节中学到的内容，包括设计模式和多线程。

我们将设计的游戏是一个名为**读者和扰乱者**的策略游戏。在这里，玩家创建单位，称为读者，他们能够建造图书馆和其他建筑物，以及士兵，他们保卫这些建筑物免受敌人的攻击。

在本章中，我们将涵盖以下主题：

+   游戏设计简介

+   深入游戏设计的过程

+   使用设计模式

+   设计游戏循环

# 技术要求

在整个本章中，将使用带有`-std=c++2a`选项的 g++编译器来编译示例。您可以在[`github.com/PacktPublishing/Expert-CPP`](https://github.com/PacktPublishing/Expert-CPP)找到本章中将使用的源文件。

# 游戏设计简介

在本章中，我们将设计一个策略游戏的后端，玩家可以创建单位（工人、士兵）、建造建筑物，并与敌人战斗。无论您设计的是策略游戏还是第一人称射击游戏，都有一些基本组件是相同的，例如游戏物理，用于使游戏对玩家更真实和沉浸。

在几乎所有游戏中都有一些重复的游戏设计组件，例如碰撞检测机制、音频系统、图形渲染等。在设计游戏时，我们可以区分引擎和游戏，或者开发一个紧密联系的应用程序，代表引擎和游戏作为一个单一的结果。将游戏引擎单独设计允许它在后续版本中进行扩展，甚至用于其他游戏。毕竟，游戏具有相同的机制和相同的流程。它们主要通过情节线有所不同。

在设计游戏引擎时，您应该仔细规划将使用引擎设计的游戏类型。虽然大多数基本功能是相同的，独立于游戏类型，但在 3D 射击游戏和策略游戏中有区别。在策略游戏中，玩家会在一个大的游戏场地上进行单位的战略部署。游戏世界是从俯视角度显示的。

# 读者和扰乱者游戏简介

游戏的基本理念很简单：玩家拥有有限的资源。这些资源可以用来为游戏角色创建建筑物。我们称这些角色单位，分为读者和士兵。读者是聪明的角色，他们建造图书馆和其他建筑物。每个建成的图书馆可以容纳多达 10 名读者。如果玩家将 10 名读者移入图书馆，经过一定时间后，图书馆会产生一名教授。教授是一个强大的单位，可以一次摧毁三名敌方士兵。教授可以为玩家的士兵制造更好的武器。

游戏从一个已建好的房子开始，有两名士兵和三名读者。房子每 5 分钟产生一个新的读者。读者可以建造新的房子，然后产生更多的读者。他们还可以建造兵营，生产士兵。

玩家的目标是建造五座图书馆，每座图书馆至少产生一名教授。玩家在游戏过程中必须保卫自己的建筑物和读者免受敌人的攻击。敌人被称为**扰乱者**，因为他们的目标是打扰读者的主要目标：在图书馆里学习。

# 策略游戏组件

正如我们之前提到的，我们的策略游戏将包括基本组件-读者和士兵（我们将称它们为单位），建筑物和地图。游戏地图包含游戏中每个对象的坐标。我们将讨论游戏地图的简化版本。现在，让我们利用我们的项目设计技能来分解游戏本身。

游戏包括以下角色单位：

+   一位读者

+   一名士兵

+   一位教授

它还包括以下建筑：

+   一座图书馆

+   一座房子

+   一座兵营

现在，让我们讨论游戏的每个组件的属性。游戏角色具有以下属性：

+   生命点数（一个整数，在每次来自敌方的攻击后减少）

+   力量（一个整数，定义单位对敌方单位造成的伤害量）

+   类型（读者，士兵或教授）

生命属性应该有一个基于单位类型的初始值。例如，读者的初始生命点数为 10，而士兵的生命点数为 12。在游戏中互动时，所有单位都可能受到敌方单位的攻击。每次攻击都被描述为生命点数的减少。我们减少生命点数的数量取决于攻击者的力量值。例如，士兵的力量值设定为 3，这意味着士兵发动的每次攻击都会使受害者的生命点数减少 3。当受害者的生命点数变为零时，角色单位将被摧毁。

建筑物也是如此。建筑物有一个完全建成的建造持续时间。完整的建筑物也有生命点数，敌方部队造成的任何损害都会减少这些生命点数。以下是建筑物属性的完整列表：

+   生命点数

+   类型

+   建造持续时间

+   单位生产持续时间

单位生产持续时间是生产新角色单位所需的时间。例如，一个兵营每 3 分钟生产一个士兵，一座房子每 5 分钟生产一个读者，一座图书馆在最后一个缺失的读者进入图书馆时立即产生一名教授。

现在我们已经定义了游戏组件，让我们讨论它们之间的互动。

# 组件之间的互动

读者和扰乱者游戏设计中的下一个重要事项是角色之间的互动。我们已经提到读者可以建造建筑物。在游戏中，这个过程应该得到照顾，因为每种类型的建筑都有其建造持续时间。因此，如果读者忙于建筑过程，我们应该测量时间，以确保建筑物在指定时间后准备好。然而，为了使游戏变得更好，我们应该考虑到不止一个读者可以参与建筑过程。这应该使建筑物的建造速度更快。例如，如果一名读者在 5 分钟内建造一座兵营，那么两名读者应该在 2 分半钟内建造一座兵营，依此类推。这是游戏中复杂互动的一个例子，并可以用以下图表来描述：

![](img/e52d3229-e4d3-4f13-b697-607876bc466e.png)

复杂互动

接下来是攻击处理。当一个单位受到敌人的攻击时，我们应该减少被告的生命点数。被告本身可以攻击攻击者（为了自卫）。每当有多个攻击者或被告时，我们应该相应地处理每个受攻击单位的生命点数减少。我们还应该定义每个单位的攻击持续时间。一个单位不应该很快地攻击另一个单位。为了使事情更加自然，我们可以在每次攻击之间引入 1 秒或 2 秒的暂停。以下图表描述了简单的攻击互动。这将在本章后面用类互动图表替换：

![](img/888a94bb-3bb8-4339-abab-f453c0a64778.png)

简单攻击互动

在游戏中发生了更大的互动。游戏中有两个组，其中一个由玩家控制，另一个由游戏自动控制。这意味着我们作为游戏设计者有责任定义敌方力量的生命周期。游戏将自动创建读者，他们将被分配创建图书馆、兵营和房屋的任务。每个士兵都应该负责保卫建筑和读者（人们）。而士兵们也应该不时地组成小组进行进攻任务。

我们将设计一个平台，让玩家创建一个帝国；然而，游戏也应该创建敌人以使游戏完整。玩家将面临来自敌人的定期攻击，而敌人将通过建造更多建筑和生产更多单位来发展。总的来说，我们可以用以下图表来描述这种互动：

![](img/e4dd9522-8e38-463f-855d-946467b87157.png)

玩家和自动玩家之间的互动

在设计游戏时，我们将经常参考上述类图。

# 设计游戏

虽然游戏不是典型的软件，但其设计与常规应用程序设计并无太大不同。我们将从主要实体开始，并进一步分解为类及其关系。

在前一节中，我们讨论了所有必要的游戏组件及其交互。我们进行了项目开发生命周期的需求分析和收集。现在，我们将开始设计游戏。

# 设计角色单位

以下类图表示了一个读者：

![](img/b494e5af-c72e-493b-bd01-f51ffc39c83b.png)

当我们浏览其他角色单位时，我们将为每个角色单位创建一个基类。每个特定单位将继承自该基类，并添加其特定的属性（如果有）。以下是角色单位的完整类图：

![](img/98c96e09-c293-4218-a3dc-f4f5e713e7a5.png)

注意基类-它是一个接口，而不是一个常规类。它定义了要在派生类中实现的纯虚函数。以下是代码中`CharacterUnit`接口的样子：

```cpp
class CharacterUnit
{
public:
  virtual void attack(const CharacterUnit&) = 0;
  virtual void destroy() = 0;
  virtual int get_power() const = 0;
  virtual int get_life_points() const = 0;
};
```

`attack()`方法减少角色的生命点数，而`destroy()`摧毁角色。摧毁意味着不仅从场景中移除角色，还停止了单位正在进行的所有交互（如建筑建造、自卫等）。

派生类为`CharacterUnit`接口类的纯虚函数提供了实现。让我们来看一下`Reader`角色单位的代码：

```cpp
class Reader : public CharacterUnit
{
public:
  Reader();
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;

public:
  void attack(const CharacterUnit& attacker) override {
    decrease_life_points_by_(attacker.get_power());
  }

  void destroy() override {
    // we will leave this empty for now
  }

  int get_life_points() const override {
    return life_points_;
  }

  int get_power() const override {
    return power_;
  }

private:
  void decrease_life_points_(int num) {
    life_points_ -= num;
    if (life_points_ <= 0) {
      destroy();
    }
  }

private:
  int life_points_;
  int power_;
};
```

现在，我们可以通过以下任何一种方式声明`Reader`单位：

```cpp
Reader reader;
Reader* pr = new Reader();
CharacterUnit* cu = new Reader();
```

我们将主要通过它们的基接口类来引用角色单位。

注意复制构造函数和赋值运算符。我们故意将它们标记为删除，因为我们不希望通过复制其他单位来创建单位。我们将使用`Prototype`模式来实现这一行为。这将在本章后面讨论。

在需要对不同类型的单位执行相同操作的情况下，具有`CharacterUnit`接口至关重要。例如，假设我们需要计算两名士兵、一名读者和一名教授对建筑物造成的完整伤害。我们可以自由地将它们都称为`CharacterUnits`，而不是保留三个不同的引用来引用三种不同类型的单位。以下是具体操作：

```cpp
int calculate_damage(const std::vector<CharacterUnit*>& units)
{
  return std::reduce(units.begin(), units.end(), 0, 
            [](CharacterUnit& u1, CharacterUnit& u2) {
                return u1.get_power() + u2.get_power();
            }
  );
}
```

`calculate_damage()`函数抽象出了单位类型；它不关心读者或士兵。它只调用`CharacterUnit`接口的`get_power()`方法，这个方法保证了特定对象的实现。

随着进展，我们将更新角色单位类。现在，让我们继续设计建筑物的类。

# 设计建筑物

建筑类与角色单位类似，具有共同的接口。例如，我们可以从以下定义房屋类开始：

```cpp
class House
{
public:
  House();
  // copying will be covered by a Prototype
  House(const House&) = delete;
  House& operator=(const House&) = delete;

public:
  void attack(const CharacterUnit&);
  void destroy();
  void build(const CharacterUnit&);
  // ...

private:
  int life_points_;
  int capacity_;
  std::chrono::duration<int> construction_duration_;
};
```

在这里，我们使用`std::chrono::duration`来保持`House`施工持续时间的时间间隔。它在`<chrono>`头文件中定义为一定数量的滴答和滴答周期，其中滴答周期是从一个滴答到下一个滴答的秒数。

`House`类需要更多细节，但我们很快会意识到我们需要一个所有建筑的基本接口（甚至是一个抽象类）。本章将描述的建筑共享某些行为。`Building`的接口如下：

```cpp
class IBuilding
{
public:
  virtual void attack(const CharacterUnit&) = 0;
  virtual void destroy() = 0;
  virtual void build(CharacterUnit*) = 0;
  virtual int get_life_points() const = 0;
};
```

注意`Building`前面的`I`前缀。许多开发人员建议为接口类使用前缀或后缀以提高可读性。例如，`Building`可能已被命名为`IBuilding`或`BuildingInterface`。我们将对先前描述的`CharacterUnit`使用相同的命名技术。

`House`、`Barrack`和`Library`类实现了`IBuilding`接口，并且必须为纯虚方法提供实现。例如，`Barrack`类将如下所示：

```cpp
class Barrack : public IBuilding
{
public:
  void attack(const ICharacterUnit& attacker) override {
    decrease_life_points_(attacker.get_power());
  }

  void destroy() override {
    // we will leave this empty for now
  }

  void build(ICharacterUnit* builder) override {
    // construction of the building
  }

  int get_life_points() const override {
    return life_points_;
  }

private:
  int life_points_;
  int capacity_;
  std::chrono::duration<int> construction_duration_;
};
```

让我们更详细地讨论施工持续时间的实现。在这一点上，`std::chrono::`持续时间点，作为一个提醒，告诉我们施工应该需要指定的时间。还要注意，类的最终设计可能会在本章的过程中发生变化。现在，让我们找出游戏组件如何相互交互。

# 设计游戏控制器

为角色单位和建筑设计类只是设计游戏本身的第一步。游戏中最重要的事情之一是设计这些组件之间的交互。我们应该仔细分析和设计诸如两个或更多角色建造一个建筑的情况。我们已经为建筑引入了施工时间，但我们没有考虑到一个建筑可能由多个读者（可以建造建筑的角色单位）来建造。

我们可以说，由两个读者建造的建筑应该比一个读者建造的建筑快两倍。如果另一个读者加入建设，我们应该重新计算持续时间。然而，我们应该限制可以在同一建筑上工作的读者数量。

如果任何读者受到敌人的攻击，那应该打扰读者建造，以便他们可以集中精力进行自卫。当一个读者停止在建筑上工作时，我们应该重新计算施工时间。攻击是另一种类似于建筑的情况。当一个角色受到攻击时，它应该通过反击来进行自卫。每次攻击都会减少角色的生命值。一个角色可能会同时受到多个敌方角色的攻击。这将更快地减少他们的生命值。

建筑有一个计时器，因为它会周期性地产生角色。设计最重要的是游戏动态-也就是循环。在每个指定的时间段，游戏中会发生一些事情。这可能是敌人士兵的接近，角色单位建造某物，或其他任何事情。一个动作的执行并不严格地与另一个无关的动作的完成相关。这意味着建筑的施工与角色的创建同时进行。与大多数应用程序不同，即使用户没有提供任何输入，游戏也应该保持运行。如果玩家未执行任何操作，游戏不会冻结。角色单位可能会等待命令，但建筑将不断地完成它们的工作-生产新的角色。此外，敌方玩家（自动化的）力求胜利，从不停顿。

# 并发动作

游戏中的许多动作是同时发生的。正如我们刚才讨论的，建筑的建造不应该因为一个没有参与建造的单位被敌人攻击而停止。如果敌人发动攻击，建筑也不应该停止生产新角色。这意味着我们应该为游戏中的许多对象设计并发行为。

在 C++中实现并发的最佳方法之一是使用线程。我们可以重新设计单位和建筑，使它们包括一个可以在其基类中重写的动作，该动作将在单独的线程中执行。让我们重新设计`IBuilding`，使其成为一个抽象类，其中包含一个额外的`run()`虚函数：

```cpp
class Building
{
public:
  virtual void attack(const ICharacterUnit&) = 0;
  virtual void destroy() = 0;
  virtual void build(ICharacterUnit*) = 0;
  virtual int get_life_points() const = 0;

public:  
 void run() {
 std::jthread{Building::background_action_, this};
 }

private:
  virtual void background_action_() {
 // no or default implementation in the base class 
 }
};
```

注意`background_action_()`函数；它是私有的，但是虚的。我们可以在派生类中重写它。`run()`函数不是虚的；它在一个线程中运行私有实现。在这里，派生类可以为`background_action_()`提供一个实现。当一个单位被分配来建造建筑时，将调用`build()`虚函数。`build()`函数将计算建造时间的工作委托给`run()`函数。

# 游戏事件循环

解决这个问题的最简单方法是定义一个事件循环。事件循环如下所示：

```cpp
while (true)
{
  processUserActions();
  updateGame();
}
```

即使用户（玩家）没有任何操作，游戏仍会通过调用`updateGame()`函数继续进行。请注意，上述代码只是对事件循环的一般介绍。正如你所看到的，它会无限循环，并在每次迭代中处理和更新游戏。

每次循环迭代都会推进游戏的状态。如果用户操作处理时间很长，可能会阻塞循环。游戏会短暂地冻结。我们通常用**每秒帧数**（**FPS**）来衡量游戏的速度。数值越高，游戏越流畅。

我们需要设计游戏循环，使其在游戏过程中持续运行。设计它的重要之处在于用户操作处理不会阻塞循环。

游戏循环负责游戏中发生的一切，包括 AI。这里的 AI 指的是我们之前讨论过的敌方玩家的自动化。除此之外，游戏循环处理角色和建筑的动作，并相应地更新游戏的状态。

在深入游戏循环设计之前，让我们先了解一些设计模式，这些模式将帮助我们完成这个复杂的任务。毕竟，游戏循环本身也是一个设计模式！

# 使用设计模式

使用**面向对象**（**OOP**）**编程**范式来设计游戏是很自然的。毕竟，游戏代表了一组对象，它们之间进行了密集的互动。在我们的策略游戏中，有单位建造的建筑。单位会抵御来自敌方单位的攻击等等。这种相互通信导致了复杂性的增长。随着项目的发展和功能的增加，支持它将变得更加困难。很明显，设计是构建项目中最重要的（如果不是最重要的）部分之一。整合设计模式将极大地改善设计过程和项目支持。

让我们来看一些在游戏开发中有用的设计模式。我们将从经典模式开始，然后讨论更多与游戏相关的模式。

# 命令模式

开发人员将设计模式分为创建型、结构型和行为型三类。命令模式是一种行为设计模式。行为设计模式主要关注对象之间通信的灵活性。在这种情况下，命令模式将一个动作封装在一个包含必要信息以及动作本身的对象中。这样，命令模式就像一个智能函数。在 C++中实现它的最简单方法是重载一个类的`operator()`，如下所示：

```cpp
class Command
{
public:
  void operator()() { std::cout << "I'm a smart function!"; }
};
```

具有重载`operator()`的类有时被称为**函数对象**。前述代码几乎与以下常规函数声明相同：

```cpp
void myFunction() { std::cout << "I'm not so smart!"; }
```

调用常规函数和`Command`类的对象看起来很相似，如下所示：

```cpp
myFunction();
Command myCommand;
myCommand();
```

这两者之间的区别在于，当我们需要为函数使用状态时，这一点就显而易见了。为了为常规函数存储状态，我们使用静态变量。为了在对象中存储状态，我们使用对象本身。以下是我们如何跟踪重载运算符的调用次数：

```cpp
class Command
{
public:
  Command() : called_(0) {}

  void operator()() {
    ++called_;
    std::cout << "I'm a smart function." << std::endl;
    std::cout << "I've been called" << called_ << " times." << std::endl;
  }

private:
  int called_;
};
```

每个`Command`类的实例的调用次数是唯一的。以下代码声明了两个`Command`的实例，并分别调用了两次和三次：

```cpp
Command c1;
Command c2;
c1();
c1();
c2();
c2();
c2();
// at this point, c1.called_ equals 2, c2.called_ equals 3
```

现在，让我们尝试将这种模式应用到我们的策略游戏中。游戏的最终版本具有图形界面，允许用户使用各种按钮和鼠标点击来控制游戏。例如，要让一个角色单位建造一座房子，而不是兵营，我们应该在游戏面板上选择相应的图标。让我们想象一个带有游戏地图和一堆按钮来控制游戏动态的游戏面板。

游戏为玩家提供以下命令：

+   将角色单位从 A 点移动到 B 点

+   攻击敌人

+   建造建筑

+   安置房屋

游戏命令的设计如下：

![](img/03f3119a-3a41-4fd1-b3ac-f5245be1e74b.png)

每个类封装了动作逻辑。客户端代码不关心处理动作。它操作命令指针，每个指针将指向具体的**Command**（如前图所示）。请注意，我们只描述了玩家将执行的命令。游戏本身使用命令在模块之间进行通信。自动命令的示例包括**Run**，**Defend**，**Die**和**Create**。以下是游戏中命令的更广泛的图表：

![](img/87e3d426-3977-49b5-ac4e-266dfcb3a93b.png)

前述命令执行游戏过程中出现的任何事件。要监听这些事件，我们应该考虑使用观察者模式。

# 观察者模式

观察者模式是一种允许我们订阅对象状态变化的架构机制。我们说我们观察对象的变化。观察者模式也是一种行为设计模式。

大多数策略游戏都包含资源的概念。这可能是岩石、黄金、木材等。例如，在建造建筑时，玩家必须花费 20 单位的木材、40 单位的岩石和 10 单位的黄金。最终，玩家将耗尽资源并必须收集资源。玩家创建更多角色单位并指派它们收集资源 - 几乎就像现实生活中发生的情况一样。

现在，假设我们的游戏中有类似的资源收集或消耗活动。当玩家指派单位收集资源时，他们应该在每次收集到一定数量的资源时通知我们。玩家是“资源收集”事件的订阅者。

建筑也是如此。建筑物生产角色 - 订阅者会收到通知。角色单位完成建筑施工 - 订阅者会收到通知。在大多数情况下，订阅者是玩家。我们更新玩家仪表板，以便在玩游戏时保持游戏状态最新；也就是说，玩家在玩游戏时可以了解自己拥有多少资源、多少单位和多少建筑物。

观察者涉及实现一个存储其订阅者并在事件上调用指定函数的类。它由两个实体组成：订阅者和发布者。如下图所示，订阅者的数量不限于一个：

![](img/5441e6d9-b94b-435b-ac15-bf16bd0be9a3.png)

例如，当角色单位被指定建造建筑时，它将不断努力建造，除非它被停止。可能会有各种原因导致这种情况发生：

+   玩家决定取消建筑施工过程。

+   角色单位必须保护自己免受敌人的攻击，并暂停施工过程。

+   建筑已经完成，所以角色单位停止在上面工作。

玩家也希望在建筑完成时收到通知，因为他们可能计划在建筑完成后让角色单位执行其他任务。我们可以设计建筑过程，使其在事件完成时通知其监听者（订阅者）。以下类图还涉及一个 Action 接口。将其视为命令模式的实现：

![](img/2ae90c0d-17e8-4028-8bfa-e944ffac3e4a.png)

根据观察者开发类，我们会发现游戏中几乎所有实体都是订阅者、发布者或两者兼而有之。如果遇到类似情况，可以考虑使用中介者-另一种行为模式。对象通过中介者对象相互通信。触发事件的对象会让中介者知道。然后中介者将消息传递给任何与对象状态“订阅”相关的对象。以下图表是中介者集成的简化版本：

![](img/db974456-125d-46b8-bb44-65bf4da3fb29.png)

每个对象都包含一个中介者，用于通知订阅者有关更改的信息。中介者对象通常包含彼此通信的所有对象。在事件发生时，每个对象通过中介者通知感兴趣的各方。例如，当建筑施工完成时，它会触发中介者，中介者会通知所有订阅的各方。为了接收这些通知，每个对象都应该事先订阅中介者。

# Flyweight 模式

Flyweight 是一种结构设计模式。结构模式负责将对象和类组装成更大、更灵活的结构。Flyweight 允许我们通过共享它们的共同部分来缓存对象。

在我们的策略游戏中，屏幕上渲染了许多对象。在游戏过程中，对象的数量会增加。玩家玩得越久，他们创建的角色单位和建筑就越多（自动敌人也是如此）。游戏中的每个单位都代表一个包含数据的单独对象。角色单位至少占用 16 字节的内存（用于其两个整数数据成员和虚拟表指针）。

当我们为了在屏幕上渲染单位而向单位添加额外字段时，情况变得更糟；例如，它们的高度、宽度和精灵（代表渲染单位的图像）。除了角色单位，游戏还应该有一些补充物品，以提高用户体验，例如树木、岩石等装饰物品。在某个时候，我们会得出结论，我们有大量对象需要在屏幕上渲染，每个对象几乎代表相同的对象，但在其状态上有一些小差异。Flyweight 模式在这里发挥了作用。对于角色单位，其高度、宽度和精灵在所有单位中存储的数据几乎相同。

Flyweight 模式建议将一个重对象分解为两个：

+   一个不可变的对象，包含相同类型对象的相同数据

+   一个可变对象，可以从其他对象中唯一标识自己

例如，移动的角色单位有自己的高度、长度和精灵，所有这些对于所有角色单位都是重复的。因此，我们可以将这些属性表示为具有相同值的单个不可变对象，对于所有对象的属性都是相同的。然而，角色单位在屏幕上的位置可能与其他位置不同，当玩家命令单位移动到其他位置或开始建造建筑时，单位的位置会不断变化直到达到终点。在每一步，单位都应该在屏幕上重新绘制。通过这样做，我们得到以下设计：

![](img/969db8dd-89fb-4fc8-91eb-c7c9a186c889.png)

左侧是修改前的`CharacterUnit`，右侧是使用享元模式进行了最近修改。游戏现在可以处理一堆`CharacterUnit`对象，而每个对象都将存储对几个`UnitData`对象的引用。这样，我们节省了大量内存。我们将每个单位独有的值存储在`CharacterUnit`对象中。这些值随时间变化。尺寸和精灵是恒定的，所以我们可以保留一个具有这些值的单个对象。这些不可变数据称为**内在状态**，而对象的可变部分（`CharacterUnit`）称为**外在状态**。

我们有意将数据成员移动到`CharacterUnit`，从而将其从接口重新设计为抽象类。正如我们在第三章中讨论的那样，抽象类几乎与可能包含实现的接口相同。`move()`方法是所有类型单位的默认实现的一个例子。这样，派生类只提供必要的行为，因为所有单位共享生命点和力量等共同属性。

在优化内存使用之后，我们应该处理复制对象的问题。游戏涉及大量创建新对象。每个建筑物都会产生一个特定的角色单位；角色单位建造建筑物，游戏世界本身渲染装饰元素（树木、岩石等）。现在，让我们尝试通过整合克隆功能来改进`CharacterUnit`。在本章的早些时候，我们有意删除了复制构造函数和赋值运算符。现在，是时候提供一个从现有对象创建新对象的机制了。

# 原型模式

这种模式让我们能够独立于它们的类型创建对象的副本。以下代码代表了`CharacterUnit`类的最终版本，关于我们最近的修改。我们还将添加新的`clone()`成员函数，以便整合原型模式：

```cpp
class CharacterUnit
{
public:
  CharacterUnit() {}
  CharacterUnit& operator=(const CharacterUnit&) = delete;
  virtual ~Character() {}

 virtual CharacterUnit* clone() = 0;

public:
  void move(const Point& to) {
    // the graphics-specific implementation
  }
  virtual void attack(const CharacterUnit&) = 0;
  virtual void destroy() = 0;

  int get_power() const { return power_; }
  int get_life_points() const { return life_points_; }

private:
  CharacterUnit(const CharacterUnit& other) {
    life_points_ = other.life_points_;
    power_ = other.power_;
  }

private:
  int life_points_;
  int power_;
};
```

我们删除了赋值运算符，并将复制构造函数移到了私有部分。派生类重写了`clone()`成员函数，如下所示：

```cpp
class Reader : public CharacterUnit
{
public:
 Reader* clone() override {
 return new Reader(*this);
 }

 // code omitted for brevity
};
```

原型模式将克隆委托给对象。通用接口允许我们将客户端代码与对象的类解耦。现在，我们可以克隆一个角色单位，而不知道它是`Reader`还是`Soldier`。看下面的例子：

```cpp
// The unit can have any of the CharacterUnit derived types
CharacterUnit* new_unit = unit->clone();
```

动态转换在我们需要将对象转换为特定类型时非常有效。

在本节中，我们讨论了许多有用的设计模式。如果您对这些模式还不熟悉，可能会感到有些不知所措；然而，正确使用它们可以让我们设计出灵活和易维护的项目。让我们最终回到之前介绍的游戏循环。

# 设计游戏循环

策略游戏拥有最频繁变化的游戏玩法之一。在任何时间点，许多动作会同时发生。读者完成他们的建筑；兵营生产士兵；士兵受到敌人的攻击；玩家命令单位移动、建造、攻击或逃跑；等等。游戏循环处理所有这些。通常，游戏引擎提供了一个设计良好的游戏循环。

当我们玩游戏时，游戏循环运行。正如我们已经提到的，循环处理玩家的动作，更新游戏状态，并渲染游戏（使状态变化对玩家可见）。它在每次迭代中都这样做。循环还应该控制游戏的速率，即其 FPS。游戏循环的一次迭代的常见术语是帧，这就是为什么我们强调 FPS 作为游戏速度的原因。例如，如果你设计一个以 60FPS 运行的游戏，这意味着每帧大约需要 16 毫秒。

在本章早些时候用于简单游戏循环的以下代码：

```cpp
while (true)
{
  processUserActions();
  updateGame();
}
```

如果没有长时间的用户操作需要处理，上述代码将运行得很快。在快速的机器上运行得更快。你的目标是坚持每帧 16 毫秒。这可能需要我们在处理操作和更新游戏状态后稍微等待一下，就像下图所示：

![](img/73b78d78-f5e3-4e4d-8398-eefd4d363046.png)

每次更新都会按固定的数量推进游戏时间，这需要固定的现实时间来处理。另一方面，如果处理时间超过了帧的指定毫秒数，游戏就会变慢。

游戏中发生的一切大部分都在游戏的更新部分中涵盖，就像前面的图表所示。大多数情况下，更新可能需要同时执行多个操作。此外，正如我们之前提到的，我们必须为游戏中发生的一些操作保持计时器。这主要取决于我们想要使游戏变得多么详细。例如，建造一个建筑物可能被表示为两种状态：初始和最终。

在图形设计方面，这两种状态应该代表两种不同的图像。第一张图像包含建筑的一些基本部分，可能包括周围的一些岩石，就像它刚准备开始施工一样。下一张图像代表最终建成的建筑。当一个角色单位刚开始建造建筑时，我们向玩家展示第一张图像（基础部分和周围的一些岩石）。当建筑完成时，我们用包含最终建筑的图像替换第一张图像。为了使过程更加自然（更接近现实世界），我们人为地延长了时间。这意味着我们在两个图像状态之间保持一个持续 30 秒或更长的计时器。

我们描述了最简单的情况，细节最少。如果我们需要使游戏更加详细，例如在建筑物施工过程中渲染每一个变化，我们应该在很多图像之间保持很多计时器，每个图像代表施工的每一步。再次看一下前面的图表。更新游戏后，我们等待*N*毫秒。等待更多毫秒会使游戏的流程更接近现实生活。如果更新花费的时间太长，导致玩家体验滞后怎么办？在这种情况下，我们需要优化游戏，使其适应最优用户体验的时间框架。现在，假设更新游戏需要执行数百个操作；玩家已经建立了一个繁荣的帝国；现在正在建造大量建筑，并用许多士兵攻击敌人。

每个角色单位的每个动作，比如从一个点移动到另一个点，攻击一个敌人单位，建造一个建筑等，都会及时显示在屏幕上。现在，如果我们一次在屏幕上渲染数百个单位的状态会怎样？这就是我们使用多线程方法的地方。每个动作都涉及独立修改对象的状态（对象可以是游戏中的任何一个单位，包括静态建筑）。

# 总结

设计游戏是一项复杂的任务。我们可以将游戏开发视为一个独立的编程领域。游戏有不同的类型，其中之一是策略游戏。策略游戏设计涉及设计单位和建筑等游戏组件。通常，策略游戏涉及收集资源、建立帝国和与敌人战斗。游戏过程涉及游戏组件之间的动态交流，比如角色单位建造建筑和收集资源，士兵保卫土地免受敌人侵袭等。

为了正确设计策略游戏，我们需要结合面向对象设计技能和设计模式。设计模式在设计整个游戏以及其组件之间的交互方面起着重要作用。在本章中，我们讨论了命令模式，它将动作封装在对象下；观察者模式，用于订阅对象事件；以及中介者模式，用于将观察者提升到组件之间复杂交互的水平。

游戏最重要的部分是其循环。游戏循环控制渲染、游戏状态的及时更新以及其他子系统。设计它涉及使用事件队列和定时器。现代游戏使用网络，允许多个玩家通过互联网一起玩游戏。

在下一章中，我们将介绍 C++中的网络编程，这样你就会拥有将网络编程融入游戏中所需的技能。

# 问题

1.  重写私有虚函数的目的是什么？

1.  描述命令设计模式。

1.  飞行权重模式如何节省内存使用？

1.  观察者模式和中介者模式有什么区别？

1.  为什么我们将游戏循环设计为无限循环？

# 进一步阅读

+   *《游戏开发模式与最佳实践：更好的游戏，更少的麻烦》John P. Doran, Matt Casanova 著*：[`www.amazon.com/Game-Development-Patterns-Best-Practices/dp/1787127834/`](https://www.amazon.com/Game-Development-Patterns-Best-Practices/dp/1787127834/)。
