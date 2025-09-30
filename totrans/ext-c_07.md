# 第七章

# 组成和聚合

在前一章中，我们讨论了封装和信息隐藏。在这一章中，我们继续讨论 C 语言中的面向对象，我们将讨论两个类之间可能存在的各种关系。最终，这将使我们能够扩展我们的对象模型，并将对象之间的关系作为即将到来的章节中的内容表达出来。

作为本章的一部分，我们讨论：

+   两个对象及其对应类之间可能存在的关系类型：我们将讨论**拥有**和**存在**关系，但本章的重点将是拥有关系。

+   **组合**作为我们的第一种拥有关系：我们将给出一个示例来演示两个类之间的真实组合关系。使用这个示例，我们探索了在组合情况下通常具有的内存结构。

+   **聚合**作为第二种拥有关系：它与组合类似，因为它们都处理拥有关系。但它们是不同的。我们将给出一个单独的完整示例来涵盖聚合案例。聚合和组合之间的区别将在与这些关系相关的内存布局中显现。

这是涵盖 C 语言中面向对象编程的四个章节中的第二个。下一章将介绍被称为**继承**的“存在”关系。

# 类之间的关系

对象模型是一组相关对象。关系数量可能很多，但两个对象之间可能存在的关系类型却有限。通常，在对象（或其对应的类）之间可以发现两种关系类别：拥有关系和存在关系。

我们将在本章深入探讨拥有关系，并在下一章涵盖存在关系。此外，我们还将看到各种对象之间的关系如何导致它们对应类之间的关系。在处理这些关系之前，我们需要能够区分类和对象。

# 对象与类

如果您还记得前一章，我们有两种构建对象的方法。一种方法是**原型基于**的，另一种是**类基于**的。

在基于原型的方法中，我们构建一个对象要么是空的（没有任何属性或行为），要么是从现有对象克隆而来。在这种情况下，“实例”和“对象”意味着同一件事。因此，基于原型的方法可以读作基于对象的方法；一种从空对象而不是类开始的方法。

在基于类的方法中，我们无法在没有蓝图的情况下构建对象，这个蓝图通常被称为**类**。因此，我们应该从类开始。然后，我们可以从这个类中实例化一个对象。在前一章中，我们解释了隐式封装技术，它将类定义为一组放入头文件中的声明。我们还给出了一些示例，展示了这在 C 语言中的实现方式。

现在，作为本节的一部分，我们想更多地讨论类和对象之间的差异。虽然这些差异似乎很微不足道，但我们想深入研究并仔细研究它们。我们首先通过一个例子开始。

假设我们定义一个类，`Person`。它具有以下属性：`name`、`surname`和`age`。我们不会讨论行为，因为差异通常来自属性，而不是行为。

在 C 中，我们可以这样编写具有公共属性的`Person`类：

```cpp
typedef struct {
  char name[32];
  char surname[32];
  unsigned int age;
} person_t;
```

代码框 7-1：C 中的 Person 属性结构

在 C++中：

```cpp
class Person {
public:
  std::string name;
  std::string family;
  uint32_t age;
};
```

代码框 7-2：C++中的 Person 类

上述代码框是相同的。事实上，当前的讨论可以应用于 C 和 C++，甚至其他面向对象的语言，如 Java。一个类（或对象模板）是一个蓝图，它只决定了每个对象所需的属性，而不是这些属性在一个特定对象中可能具有的值。实际上，每个对象都有其自己的特定值集，这些值与从同一类实例化的其他对象中存在的相同属性相对应。

当基于一个类创建一个对象时，首先分配其内存。这个分配的内存将作为属性值的占位符。之后，我们需要用一些值初始化属性值。这是一个重要的步骤，否则，对象在被创建后可能会处于无效状态。正如您已经看到的，这个步骤被称为*构造*。

通常有一个专门执行构造步骤的函数，这被称为*构造函数*。在上一章中找到的示例中的`list_init`和`car_construct`函数是构造函数。完全有可能在构建对象的过程中，我们需要为该对象所需的其他对象、缓冲区、数组、流等资源分配更多的内存。对象拥有的资源必须在释放拥有者对象之前被释放。

我们还有一个另一个与构造函数类似的功能，它负责释放任何分配的资源。它被称为*析构函数*。同样，在上一章中找到的示例中的`list_destroy`和`car_destruct`函数是析构函数。在析构一个对象后，其分配的内存被释放，但在那之前，所有拥有的资源和它们相应的内存必须被释放。

在继续之前，让我们总结一下到目前为止我们已经解释的内容：

+   类是一个蓝图，用作创建对象的映射。

+   可以从同一个类中创建许多对象。

+   一个类决定了每个基于该类创建的未来对象应该具有哪些属性。它并没有说明它们可能具有的值。

+   类本身不消耗任何内存（除了 C 和 C++以外的某些编程语言之外）并且只存在于源级别和编译时。但对象存在于运行时并且消耗内存。

+   在创建对象时，首先发生内存分配。此外，内存释放是对象的最后一个操作。

+   在创建对象时，应该在内存分配之后立即构造它。它也应该在分配之前立即销毁。

+   一个对象可能拥有一些资源，如流、缓冲区、数组等，在对象被销毁之前必须释放。

既然你已经知道了类和对象之间的区别，我们可以继续解释两个对象及其对应类之间可能存在的不同关系。我们将从组合开始。

# 组合

正如“组合”一词所暗示的，当一个对象包含或拥有另一个对象——换句话说，它由另一个对象组成——我们说它们之间存在组合关系。

例如，一辆汽车有一个引擎；汽车是一个包含引擎对象的物体。因此，汽车和引擎对象之间存在组合关系。组合关系必须满足的一个重要条件是：*包含对象的生存期绑定到容器对象的生存期*。

只要容器对象存在，包含对象就必须存在。但是当容器对象即将被销毁时，包含对象必须先被销毁。这个条件意味着包含对象通常是容器内部的私有对象。

包含对象的一些部分可能仍然可以通过容器类的公共接口（或行为函数）访问，但包含对象的生存期必须由容器对象内部管理。如果一段代码可以在不破坏容器对象的情况下破坏包含对象，那么它违反了组合关系，这种关系就不再是组合关系。

以下示例，*示例 7.1*，展示了汽车对象和引擎对象之间的组合关系。

它由五个文件组成：两个头文件，声明了 `Car` 和 `Engine` 类的公共接口；两个源文件，包含了 `Car` 和 `Engine` 类的实现；最后是一个源文件，包含了 `main` 函数并执行了一个使用汽车及其引擎对象的简单场景。

注意，在某些领域，我们可以在汽车对象之外拥有引擎对象；例如，在机械工程 CAD 软件中。因此，各种对象之间的关系类型由问题域决定。为了我们的示例，想象一个引擎对象不能存在于汽车对象之外的领域。

以下代码框显示了 `Car` 类的头文件：

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_7_1_CAR_H
#define EXTREME_C_EXAMPLES_CHAPTER_7_1_CAR_H
struct car_t;
// Memory allocator
struct car_t* car_new();
// Constructor
void car_ctor(struct car_t*);
// Destructor
void car_dtor(struct car_t*);
// Behavior functions
void car_start(struct car_t*);
void car_stop(struct car_t*);
double car_get_engine_temperature(struct car_t*);
#endif
```

代码框 7-3 [ExtremeC_examples_chapter7_1_car.h]：`Car` 类的公共接口

正如你所见，前面的声明是以与我们上一章最后一个例子中 `List` 类所做的方式进行的，*例子 6.3*。其中一个不同之处在于我们为构造函数选择了一个新的后缀；`car_new` 而不是 `car_construct`。另一个不同之处在于我们只声明了属性结构 `car_t`。我们没有定义其字段，这被称为 *前向声明*。`car_t` 结构的定义将在代码框 7-5 所示的源文件中。请注意，在前面的头文件中，类型 `car_t` 被视为一个不完整类型，尚未定义。

以下代码框包含了 `Engine` 类的头文件：

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_7_1_ENGINE_H
#define EXTREME_C_EXAMPLES_CHAPTER_7_1_ENGINE_H
struct engine_t;
// Memory allocator
struct engine_t* engine_new();
// Constructor
void engine_ctor(struct engine_t*);
// Destructor
void engine_dtor(struct engine_t*);
// Behavior functions
void engine_turn_on(struct engine_t*);
void engine_turn_off(struct engine_t*);
double engine_get_temperature(struct engine_t*);
#endif
```

代码框 7-4 [ExtremeC_examples_chapter7_1_engine.h]: Engine 类的公共接口

以下代码框包含了为 `Car` 和 `Engine` 类实现的代码。我们首先从 `Car` 类开始：

```cpp
#include <stdlib.h>
// Car is only able to work with the public interface of Engine
#include "ExtremeC_examples_chapter7_1_engine.h"
typedef struct {
  // Composition happens because of this attribute
  struct engine_t* engine;
} car_t;
car_t* car_new() {
  return (car_t*)malloc(sizeof(car_t));
}
void car_ctor(car_t* car) {
  // Allocate memory for the engine object
  car->engine = engine_new();
  // Construct the engine object
  engine_ctor(car->engine);
}
void car_dtor(car_t* car) {
  // Destruct the engine object
  engine_dtor(car->engine);
  // Free the memory allocated for the engine object
  free(car->engine);
}
void car_start(car_t* car) {
  engine_turn_on(car->engine);
}
void car_stop(car_t* car) {
  engine_turn_off(car->engine);
}
double car_get_engine_temperature(car_t* car) {
  return engine_get_temperature(car->engine);
}
```

代码框 7-5 [ExtremeC_examples_chapter7_1_car.c]: Car 类的定义

前面的代码框展示了汽车是如何包含发动机的。正如你所见，我们有一个新的属性作为 `car_t` 属性结构的一部分，它是 `struct engine_t*` 类型。组合正是因为这个属性而发生的。

尽管在这个源文件中，类型 `struct engine_t*` 仍然是不完整的，但在运行时它可以指向一个完整的 `engine_t` 类型的对象。这个属性将指向作为 `Car` 类构造函数一部分将要构建的对象，它将在析构函数中释放。在两个地方，汽车对象都存在，这意味着发动机的生命周期包含在汽车的生命周期中。

`engine` 指针是私有的，并且没有指针从实现中泄漏出来。这是一个重要的注意事项。当你实现组合关系时，不应该有指针泄漏出来，否则它会使外部代码能够改变包含对象的内部状态。就像封装一样，当它提供对对象私有部分的直接访问时，不应该有指针泄漏出来。私有部分应该始终通过行为函数间接访问。

代码框中的 `car_get_engine_temperature` 函数提供了对发动机的 `temperature` 属性的访问。然而，关于这个函数有一个重要的注意事项。它使用了发动机的公共接口。如果你注意观察，你会看到 *汽车的私有实现* 正在消耗 *发动机的公共接口*。

这意味着汽车本身对发动机的实现细节一无所知。这正是它应该的方式。

*两个不同类型的对象，在大多数情况下，不应该知道彼此的实现细节*。这是信息隐藏所规定的。记住，汽车的行为被认为是发动机的外部行为。

这样，我们可以用替代的实现替换引擎的实现，只要新的实现提供了引擎头文件中声明的相同公共函数的定义，它应该就能正常工作。

现在，让我们看看`Engine`类的实现：

```cpp
#include <stdlib.h>
typedef enum {
  ON,
  OFF
} state_t;
typedef struct {
  state_t state;
  double temperature;
} engine_t;
// Memory allocator
engine_t* engine_new() {
  return (engine_t*)malloc(sizeof(engine_t));
}
// Constructor
void engine_ctor(engine_t* engine) {
  engine->state = OFF;
  engine->temperature = 15;
}
// Destructor
void engine_dtor(engine_t* engine) {
  // Nothing to do
}
// Behavior functions
void engine_turn_on(engine_t* engine) {
  if (engine->state == ON) {
    return;
  }
  engine->state = ON;
  engine->temperature = 75;
}
void engine_turn_off(engine_t* engine) {
  if (engine->state == OFF) {
    return;
  }
  engine->state = OFF;
  engine->temperature = 15;
}
double engine_get_temperature(engine_t* engine) {
  return engine->temperature;
}
```

Code Box 7-6 [ExtremeC_examples_chapter7_1_engine.c]：引擎类的定义

前面的代码只是使用了隐式封装方法来处理其私有实现，这与之前的示例非常相似。但有一点需要注意。如您所见，`engine`对象并不知道一个外部对象将要将其包含在组合关系中。这就像现实世界一样。当一家公司制造引擎时，并不清楚哪个引擎将进入哪辆汽车。当然，我们本可以保留对容器`car`对象的指针，但在这个例子中，我们不需要这样做。

下面的代码框演示了创建`car`对象并调用其一些公开 API 以提取有关汽车引擎信息的场景：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include "ExtremeC_examples_chapter7_1_car.h"
int main(int argc, char** argv) {
  // Allocate memory for the car object
  struct car_t *car = car_new();
  // Construct the car object
  car_ctor(car);
  printf("Engine temperature before starting the car: %f\n",
          car_get_engine_temperature(car));
  car_start(car);
  printf("Engine temperature after starting the car: %f\n",
          car_get_engine_temperature(car));
  car_stop(car);
  printf("Engine temperature after stopping the car: %f\n",
          car_get_engine_temperature(car));
  // Destruct the car object
  car_dtor(car);
  // Free the memory allocated for the car object
  free(car);
  return 0;
}
```

Code Box 7-7 [ExtremeC_examples_chapter7_1_main.c]：示例 7.1 的主函数

要构建前面的示例，首先我们需要编译前三个源文件。然后，我们需要将它们链接在一起以生成最终的可执行目标文件。请注意，主源文件（包含`main`函数的源文件）只依赖于汽车公开的接口。因此，在链接时，它只需要`car`对象的私有实现。然而，`car`对象的私有实现依赖于引擎接口的公开接口；因此，在链接时，我们需要提供`engine`对象的私有实现。因此，我们需要链接所有三个目标文件才能得到最终的可执行文件。

以下命令显示了如何构建示例并运行最终的可执行文件：

```cpp
$ gcc -c ExtremeC_examples_chapter7_1_engine.c -o engine.o
$ gcc -c ExtremeC_examples_chapter7_1_car.c -o car.o
$ gcc -c ExtremeC_examples_chapter7_1_main.c -o main.o
$ gcc engine.o car.o main.o -o ex7_1.out
$ ./ex7_1.out
Engine temperature before starting the car: 15.000000
Engine temperature after starting the car: 75.000000
Engine temperature after stopping the car: 15.000000
$
```

Shell Box 7-1：示例 7.1 的编译、链接和执行

在本节中，我们解释了两个对象之间可能存在的一种关系类型。在下一节中，我们将讨论另一种关系。它与组合关系有相似的概念，但有一些显著的区别。

# 聚合

聚合也涉及一个包含另一个对象的容器对象。主要区别在于，在聚合中，包含对象的生存期独立于容器对象的生存期。

在聚合中，包含的对象甚至可以在容器对象构建之前就被构建。这与组合相反，在组合中，包含的对象的生存期应该短于或等于容器对象的生存期。

以下示例，*示例 7.2*，演示了聚合关系。它描述了一个非常简单的游戏场景，其中玩家拿起枪，射击多次，然后放下枪。

`player` 对象将暂时成为容器对象，而 `gun` 对象将作为被包含对象，只要玩家对象持有它。枪对象的生命周期独立于玩家对象的生命周期。

以下代码框展示了 `Gun` 类的头文件：

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_7_2_GUN_H
#define EXTREME_C_EXAMPLES_CHAPTER_7_2_GUN_H
typedef int bool_t;
// Type forward declarations
struct gun_t;
// Memory allocator
struct gun_t* gun_new();
// Constructor
void gun_ctor(struct gun_t*, int);
// Destructor
void gun_dtor(struct gun_t*);
// Behavior functions
bool_t gun_has_bullets(struct gun_t*);
void gun_trigger(struct gun_t*);
void gun_refill(struct gun_t*);
#endif
```

代码框 7-8 [ExtremeC_examples_chapter7_2_gun.h]: 枪类的公共接口

正如你所见，我们只声明了 `gun_t` 属性结构，因为我们还没有定义其字段。正如我们之前解释的，这被称为前置声明，它导致了一个不完整类型，不能被实例化。

以下代码框展示了 `Player` 类的头文件：

```cpp
#ifndef EXTREME_C_EXAMPLES_CHAPTER_7_2_PLAYER_H
#define EXTREME_C_EXAMPLES_CHAPTER_7_2_PLAYER_H
// Type forward declarations
struct player_t;
struct gun_t;
// Memory allocator
struct player_t* player_new();
// Constructor
void player_ctor(struct player_t*, const char*);
// Destructor
void player_dtor(struct player_t*);
// Behavior functions
void player_pickup_gun(struct player_t*, struct gun_t*);
void player_shoot(struct player_t*);
void player_drop_gun(struct player_t*);
#endif
```

代码框 7-9 [ExtremeC_examples_chapter7_2_player.h]: 玩家类的公共接口

上述代码框定义了所有玩家对象的公共接口。换句话说，它定义了 `Player` 类的公共接口。

再次，我们必须转发 `gun_t` 和 `player_t` 结构的声明。我们需要声明 `gun_t` 类型，因为 `Player` 类的一些行为函数的参数是这种类型。

`Player` 类的实现如下：

```cpp
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ExtremeC_examples_chapter7_2_gun.h"
// Attribute structure
typedef struct {
  char* name;
  struct gun_t* gun;
} player_t;
// Memory allocator
player_t* player_new() {
  return (player_t*)malloc(sizeof(player_t));
}
// Constructor
void player_ctor(player_t* player, const char* name) {
  player->name =
      (char*)malloc((strlen(name) + 1) * sizeof(char));
  strcpy(player->name, name);
  // This is important. We need to nullify aggregation pointers
  // if they are not meant to be set in constructor.
  player->gun = NULL;
}
// Destructor
void player_dtor(player_t* player) {
  free(player->name);
}
// Behavior functions
void player_pickup_gun(player_t* player, struct gun_t* gun) {
  // After the following line the aggregation relation begins.
  player->gun = gun;
}
void player_shoot(player_t* player) {
  // We need to check if the player has picked up the gun
  // otherwise, shooting is meaningless
  if (player->gun) {
    gun_trigger(player->gun);
  } else {
    printf("Player wants to shoot but he doesn't have a gun!");
    exit(1);
  }
}
void player_drop_gun(player_t* player) {
  // After the following line the aggregation relation
  // ends between two objects. Note that the object gun
  // should not be freed since this object is not its
  // owner like composition.
  player->gun = NULL;
}
```

代码框 7-10 [ExtremeC_examples_chapter7_2_player.c]: 玩家类的定义

在 `player_t` 结构内部，我们声明了一个即将指向 `gun` 对象的指针属性 `gun`。我们需要在构造函数中将它置为空，因为与组合不同，这个属性不是作为构造函数的一部分设置的。

如果需要在构造时设置聚合指针，则应将目标对象的地址作为参数传递给构造函数。然后，这种情况被称为 *强制聚合*。

如果聚合指针可以在构造函数中留为 `null`，那么它就是一个 *可选聚合*，如前面的代码所示。在构造函数中置空可选聚合指针是很重要的。

在函数 `player_pickup_gun` 中，聚合关系开始，当玩家丢弃枪时，在函数 `player_drop_gun` 中结束。

注意，在解除聚合关系后，我们需要将指针 `gun` 置为空。与组合不同，容器对象不是被包含对象的 *所有者*。因此，它对其生命周期没有控制权。因此，我们不应该在任何地方释放玩家实现代码中的枪对象。

在可选的聚合关系中，我们可能在程序中的某个点没有设置被包含对象。因此，在使用聚合指针时应该小心，因为对未设置或 `null` 的指针的任何访问都可能导致段错误。这就是为什么在函数 `player_shoot` 中，我们检查 `gun` 指针是否有效的原因。如果聚合指针为空，这意味着使用玩家对象的代码正在误用它。如果是这种情况，我们将通过返回进程的 *退出* 代码 1 来中止执行。

以下代码是 `Gun` 类的实现：

```cpp
#include <stdlib.h>
typedef int bool_t;
// Attribute structure
typedef struct {
  int bullets;
} gun_t;
// Memory allocator
gun_t* gun_new() {
  return (gun_t*)malloc(sizeof(gun_t));
}
// Constructor
void gun_ctor(gun_t* gun, int initial_bullets) {
  gun->bullets = 0;
  if (initial_bullets > 0) {
    gun->bullets = initial_bullets;
  }
}
// Destructor
void gun_dtor(gun_t* gun) {
  // Nothing to do
}
// Behavior functions
bool_t gun_has_bullets(gun_t* gun) {
  return (gun->bullets > 0);
}
void gun_trigger(gun_t* gun) {
  gun->bullets--;
}
void gun_refill(gun_t* gun) {
  gun->bullets = 7;
}
```

代码框 7-11 [ExtremeC_examples_chapter7_2_gun.c]：枪类定义

上述代码很简单，并且以这种方式编写，枪对象不知道它将被包含在任何对象中。

最后，以下代码框演示了一个简短的场景，该场景创建了一个 `player` 对象和一个 `gun` 对象。然后，玩家拿起枪并使用它直到没有弹药。之后，玩家重新装填枪并重复同样的操作。最后，他们丢弃了枪：

```cpp
#include <stdio.h>
#include <stdlib.h>
#include "ExtremeC_examples_chapter7_2_player.h"
#include "ExtremeC_examples_chapter7_2_gun.h"
int main(int argc, char** argv) {
  // Create and constructor the gun object
  struct gun_t* gun = gun_new();
  gun_ctor(gun, 3);
  // Create and construct the player object
  struct player_t* player = player_new();
  player_ctor(player, "Billy");
  // Begin the aggregation relation.
  player_pickup_gun(player, gun);
  // Shoot until no bullet is left.
  while (gun_has_bullets(gun)) {
    player_shoot(player);
  }
  // Refill the gun
  gun_refill(gun);
  // Shoot until no bullet is left.
  while (gun_has_bullets(gun)) {
    player_shoot(player);
  }
  // End the aggregation relation.
  player_drop_gun(player);
  // Destruct and free the player object
  player_dtor(player);
  free(player);
  // Destruct and free the gun object
  gun_dtor(gun);
  free(gun);
  return 0;
}
```

代码框 7-12 [ExtremeC_examples_chapter7_2_main.c]：示例 7.2 的主函数

正如你所见，`gun` 和 `player` 对象是相互独立的。创建和销毁这些对象的责任逻辑是 `main` 函数。在执行过程中某个时刻，它们形成一个聚合关系并执行其角色，然后在另一个时刻，它们再次分离。在聚合关系中重要的是，容器对象不应改变包含对象的生存期，只要遵循这个规则，就不会出现内存问题。

下面的 shell 框中展示了如何构建示例并运行生成的可执行文件。正如你所见，*代码框 7-12* 中的 `main` 函数没有产生任何输出：

```cpp
$ gcc -c ExtremeC_examples_chapter7_2_gun.c -o gun.o $ gcc -c ExtremeC_examples_chapter7_2_player.c -o player.o $ gcc -c ExtremeC_examples_chapter7_2_main.c -o main.o $ gcc gun.o player.o main.o -o ex7_2.out $ ./ex7_2.out $
```

Shell 框 7-2：示例 7.2 的编译、链接和执行

在为真实项目创建的对象模型中，聚合关系的数量通常大于组合关系的数量。此外，由于为了建立聚合关系，至少在容器对象的公共接口中需要一些专门的行为函数来设置和重置包含的对象，因此聚合关系在外部更明显。

如前例所示，`gun` 和 `player` 对象从一开始就是分离的。它们短暂地建立关系，然后再次分离。这意味着聚合关系是临时的，而组合关系是永久的。这表明组合是对象之间一种更强的*拥有*（to-have）关系形式，而聚合则表现出较弱的关系。

现在，一个问题浮现在脑海中。如果两个对象之间的聚合关系是临时的，那么它们对应的类之间的聚合关系也是临时的吗？答案是：不是。聚合关系在类型之间是永久的。如果将来有极小的可能性，两个不同类型的对象基于聚合关系建立关系，那么它们的类型应该永久处于聚合关系中。这也适用于组合关系。

即使存在聚合关系的可能性很低，我们也应该在容器对象的属性结构中声明一些指针，这意味着属性结构将永久改变。当然，这仅适用于基于类的编程语言。

组合和聚合都描述了某些对象的拥有。换句话说，这些关系描述了一种“拥有”或“有”的情况；一个玩家**拥有**一把枪，或者一辆车**有**一个引擎。每次你感觉到一个对象拥有另一个对象时，这意味着它们之间（以及它们对应的类）应该存在组合关系或聚合关系。

在下一章中，我们将通过查看*继承*或*扩展*关系来继续我们关于关系类型的讨论。

# 摘要

在本章中，我们讨论了以下主题：

+   类和对象之间可能的关系类型。

+   类、对象、实例和引用之间的区别和相似之处。

+   组合，意味着包含的对象完全依赖于其容器对象。

+   聚合，其中包含的对象可以自由地生活，而不依赖于其容器对象。

+   聚合可以在对象之间是临时的，但在它们的类型（或类）之间是永久定义的。

在下一章中，我们继续探索面向对象编程（OOP），主要解决它基于的两个进一步支柱：继承和多态。
