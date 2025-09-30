

# 第十三章：虚构造函数和工厂

在 C++中，任何类的任何成员函数，包括其析构函数，都可以被声明为虚函数——唯一的例外是构造函数。没有虚函数，在编译时就可以知道调用成员函数的对象的确切类型。因此，构造的对象类型在编译时总是已知的，在构造函数调用时已知。尽管如此，我们经常需要构造在运行时才知道类型的对象。本章描述了几个相关的模式和惯用法，以各种方式解决这个设计问题，包括工厂模式。

本章将涵盖以下主题：

+   为什么无法使构造函数成为虚函数

+   如何使用工厂模式来延迟构造对象类型的选择，直到编译时

+   使用 C++惯用法来多态地构造和复制对象

# 技术要求

本章的示例代码可以在以下 GitHub 链接中找到：[`github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP_Second_Edition/tree/master/Chapter13`](https://github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP_Second_Edition/tree/master/Chapter13).

# 为什么构造函数不能是虚函数

我们已经理解了多态是如何工作的——当通过基类指针或引用调用虚函数时，该指针或引用用于访问类中的 v 指针。v 指针用于识别对象的真正类型，即对象创建时所用的类型。这可能就是基类本身，或者是任何派生类之一。实际上调用的是该对象上的成员函数。那么，为什么构造函数不能这样做呢？让我们来调查一下。

## 对象何时获得其类型？

很容易理解为什么我们之前描述的过程不能用于创建*虚构造函数*。首先，从先前过程的描述中可以明显看出——作为其中的一部分，我们*识别了对象创建时所用的类型*。这只能在对象构造之后发生——在构造之前，我们还没有这种类型的对象，只有一些未初始化的内存。另一种看待方式是——在虚函数被调度到正确的类型之前，需要查找 v 指针。谁将正确的值放入 v 指针中？考虑到 v 指针唯一地标识了对象的类型，它只能在构造过程中初始化。这意味着它在此之前没有被初始化。但如果它没有被初始化，就不能用它来调度虚函数调用。因此，我们再次意识到构造函数不能是虚函数。

对于层次结构中的派生类，确定类型的流程更加复杂。我们可以尝试观察对象在构造过程中的类型。最简单的方法是使用 `typeid` 操作符，它返回有关对象类型的详细信息，包括类型的名称：

```cpp
// Example 01
#include <iostream>
#include <typeinfo>
using std::cout;
using std::endl;
template <typename T>
auto type(T&& t) { return typeid(t).name(); }
class A {
  public:
  A() { cout << "A::A(): " << type(*this) << endl; }
  virtual
  ~A() { cout << "A::~A(): " << type(*this) << endl; }
};
class B : public A {
  public:
  B() { cout << "B::B(): " << type(*this) << endl; }
  ~B() { cout << "B::~B(): " << type(*this) << endl; }
};
class C : public B {
  public:
  C() { cout << "C::C(): " << type(*this) << endl; }
  ~C() { cout << "C::~C(): " << type(*this) << endl; }
};
int main() {
  C c;
}
```

运行这个程序产生以下结果：

```cpp
A::A(): 1A
B::B(): 1B
C::C(): 1C
C::~C(): 1C
B::~B(): 1B
A::~A(): 1A
```

`std::typeinfo::name()` 调用返回的类型名称是所谓的名称混淆类型名称——这是编译器用来识别类型的内部名称，而不是像 `class A` 这样的可读名称。如果你想了解未混淆的类型，你可以使用像 GCC 中的 `c++filt` 程序这样的去混淆器：

```cpp
$ c++filt -t 1A
A
```

我们也可以编写一个小的 C++ 函数来去混淆类型名称，但实现方式因编译器而异（没有可移植版本）。例如，这是为 GCC 编写的代码：

```cpp
// Example 2
#include <cxxabi.h>
template <typename T> auto type(T&& p) {
  int r;
  std::string name;
  char* mangled_name =
    abi::__cxa_demangle(typeid(p).name(), 0, 0, &r);
  name += mangled_name;
  ::free(mangled_name);
  return name;
}
```

注意，去混淆函数返回一个 C 字符串（一个 `char*` 指针），必须由调用者显式释放。现在程序打印去混淆后的名称，如 `A`、`B` 和 `C`。这足以满足我们的需求，但在某些情况下，你可能会注意到类型并没有按预期打印出来：

```cpp
class A {
  public:
  void f() const { cout << type(*this) << endl; }
};
...
C c;
c.f();
```

如果我们调用函数 `f()`，其类型报告为 `C`，而不是我们可能预期的 `const C`（对象在 `const` 成员函数内部是 `const` 的）。这是因为 `typeid` 操作符移除了 `const` 和 `volatile` 限定符以及类型中的任何引用。要打印这些，你必须自己找出它们：

```cpp
// Example 03
template <typename T> auto type(T&& p) {
  std::string name;
  using TT = std::remove_reference_t<T>;
  if (std::is_const<TT>::value) name += "const ";
  if (std::is_volatile<TT>::value) name += "volatile ";
  int r;
  name += abi::__cxa_demangle(typeid(p).name(), 0, 0, &r);
  return name;
}
```

无论你选择如何打印类型，在这些示例中构造了多少个对象？源代码只说了一个，类型为 `C` 的 `c` 对象：

```cpp
int main() {
  C c;
}
```

运行时输出显示三个，即每种类型的一个。两个答案都是正确的——当类型为 `C` 的对象被构造时，必须首先构造基类 `A`，因此会调用其构造函数。然后，构造中间基类 `B`，之后才会构造 `C`。析构函数的执行顺序是相反的。由 `typeid` 操作符报告的对象构造函数或析构函数中的类型与正在运行构造函数或析构函数的对象的类型相同。

看起来，类型，如虚拟指针所示，在构造过程中正在改变！当然，这是假设 `typeid` 操作符返回的是动态类型，即虚拟指针指示的类型，而不是在编译时可以确定的静态类型。标准指出，这确实是情况。这意味着，如果我们从每个构造函数中调用相同的虚拟方法，我们实际上会调用这个方法的三种不同的重写吗？这很容易找到答案：

```cpp
// Example 04
class A {
  public:
  A() { whoami(); }
  virtual ~A() { whoami(); }
  virtual void whoami() const {
    std::cout << "A::whoami" << std::endl;
  }
};
class B : public A {
  public:
  B() { whoami(); }
  ~B() { whoami(); }
  void whoami() const override {
    std::cout << "B::whoami" << std::endl;
  }
};
class C : public B {
  public:
  C() { whoami(); }
  ~C() { whoami(); }
  void whoami() const override {
    std::cout << "C::whoami" << std::endl;
  }
};
int main() {
  C c;
  c.whoami();
}
```

现在，我们将创建一个类型为`C`的对象，并在创建之后调用`whoami()`来确认它——对象的动态类型是`C`。这是从构造过程的开始就是正确的；我们要求编译器构造一个类型为`C`的对象，但在构造过程中对象的动态类型发生了变化：

```cpp
A::whoami
B::whoami
C::whoami
C::whoami
C::whoami
B::whoami
A::whoami
```

很明显，随着对象构造的进行，虚拟指针值已经改变。一开始，它将对象类型识别为`A`，即使最终类型是`C`。这是否因为我们是在栈上创建了对象？如果对象是在堆上创建的，会有所不同吗？我们可以很容易地找到答案：

```cpp
C* c = new C;
c->whoami();
delete c;
```

运行修改后的程序会产生与原始程序完全相同的结果。

另一个原因是因为构造函数不能是虚拟的，或者更普遍地说，为什么正在构造的对象的类型必须在构造点在编译时已知，是因为编译器必须知道为对象分配多少内存。内存量由类型的大小决定，即由`sizeof`运算符。`sizeof(C)`的结果是一个编译时常量，因此为新对象分配的内存量始终在编译时已知。这无论是我们在栈上还是堆上创建对象都是正确的。

核心问题是这样的——如果程序创建了一个`T`类型的对象，那么在代码的某个地方会有一个对`T::T`构造函数的显式调用。之后，我们可以在程序的其余部分隐藏`T`类型，例如，通过通过基类指针访问对象，或者通过擦除类型（参见*第六章*，*理解类型擦除*）。但是，代码中必须至少有一个对`T`类型的显式提及，那就是在构造的时候。

一方面，我们现在有一个非常合理的解释，说明了为什么构造对象永远不能是多态的。另一方面，这并没有解决可能需要构造一个在编译时类型未知的设计挑战。考虑设计一个游戏——玩家可以为他们的团队招募或召唤任意数量的冒险者，并建立定居点和城市。为每种生物种类和每种建筑类型拥有一个单独的类似乎是合理的，但当我们有一个冒险者加入团队或一座建筑被建立时，我们必须构造这些类型之一的对象，直到玩家选择它，游戏才能知道要构造哪个对象。

如同软件中的常规做法，解决方案涉及添加另一个间接层。

# 工厂模式

我们面临的问题，即如何在运行时决定创建特定类型的对象，显然是一个非常常见的设计问题。设计模式正是针对这类问题的解决方案，而且对于这个问题也有一个模式——它被称为工厂模式。工厂模式是一种创建型模式，它为几个相关的问题提供了解决方案——如何将创建哪个对象的决策委托给派生类，如何使用单独的工厂方法创建对象，等等。我们将逐一回顾工厂模式的这些变体，从基本的工厂方法开始。

## 工厂方法的基本原理

在其最简单的形式中，工厂方法构建一个在运行时指定的类型的对象：

```cpp
class Base { ... };
class Derived : public Base { ... };
Base* p = ClassFactory(type_identifier, ... arguments );
```

我们如何在运行时识别要创建的对象？我们需要为工厂可以创建的每种类型提供一个运行时标识符。在最简单的情况下，这些类型的列表在编译时是已知的。

考虑一个游戏设计，玩家可以从菜单中选择要构建的建筑类型。程序有一个可以构建的建筑列表，每个建筑由一个对象表示，并为每个对象分配一个标识符：

```cpp
// Example 05
enum Buildings {
  FARM, FORGE, MILL, GUARDHOUSE, KEEP, CASTLE
};
class Building {
  public:
  virtual ~Building() {}
};
class Farm : public Building { ... };
class Forge : public Building { ... };
```

当玩家选择建筑类型时，游戏程序也会选择相应的标识符值。现在，程序可以使用工厂方法构建建筑：

```cpp
Building* new_farm = MakeBuilding(FARM);
```

注意，工厂函数接受类型标识符参数并返回基类的指针。返回的对象应该具有与类型标识符相对应的类型。工厂是如何实现的？记住上一节的结论——在程序的某个地方，每个对象都必须显式地使用其真实类型进行构造。工厂模式并不取消这一要求；它只是隐藏了构造发生的地方：

```cpp
// Example 05
Building* MakeBuilding(Buildings building_type) {
  switch (building_type) {
    case FARM: return new Farm;
    case FORGE: return new Forge;
    ...
  }
}
```

类型标识符与对象类型之间的对应关系编码在工厂内部的`switch`语句中。由于只有一个工厂方法，并且其类型在编译时声明，因此返回类型必须对所有由工厂构建的类型相同。在最简单的情况下，它是基类指针，尽管如果你遵循本书中描述的现代内存所有权习惯用法*第三章*，*内存和所有权*，那么工厂应该返回对基类的唯一指针，`std::unique_ptr<Building>`。

```cpp
// Example 06:
class Building {
  public:
  enum Type {FARM, FORGE, ...};
  virtual ~Building() {}
  auto MakeBuilding(Type building_type);
};
auto Building::MakeBuilding(Type building_type) {
  using result_t = std::unique_ptr<Building>;
  switch (building_type) {
    case FARM: return result_t{new Farm};
    case FORGE: return result_t{new Forge};
    ...
  }
}
```

在极少数需要共享所有权的场合，可以通过将对象从唯一指针移动到共享指针`std::shared_ptr<Building>`来创建共享所有权（但这是由调用者做出的决定，而不是工厂本身）。

我们在这里做出的另一个设计选择（独立于使用拥有指针）是将类型标识符和工厂函数移动到基类中。这对于封装和保持所有相关代码和类型更接近是有用的。

这就是工厂方法的基本形式。有许多变体使其更适合特定问题。我们将在下面回顾其中的一些变体。

## 工厂方法的澄清

注意，“工厂方法”这个术语的使用存在一些歧义。在本章中，我们用它来描述基于某些运行时信息创建不同类型对象的函数。还有一个与之不相关的、有时以相同名称引入的设计模式：这个模式不是构建不同的类，而是以不同的方式构建相同的类。以下是一个简短的例子：假设我们有一个类来表示平面上的一个点。这个点由其坐标 `x` 和 `y` 描述：

```cpp
class Point {
  double x_ {};
  double y_ {};
  public:
  Point(double x, double y) : x_(x), y_(y) {}
};
```

到目前为止，一切顺利。但同一个点可以用极坐标，例如，来描述。因为这些是描述同一个点的两种方式，我们不需要一个单独的类，但我们可能想要一个新的构造函数，它可以从指定的极坐标创建笛卡尔点：

```cpp
class Point() {
  ...
  Point(double r, double angle);
};
```

但这行不通：新的构造函数和从 `x` 和 `y` 来的原始构造函数都接受完全相同的参数，因此重载解析无法确定你指的是哪一个。一个解决方案是使用不同单位测量的量（在我们的例子中是长度和角度）使用不同的类型。但它们必须是真正不同的类型，而不仅仅是别名。有时，这样的单位模板库正是你所需要的，但如果你坚持使用双精度浮点数，你需要其他方法来根据调用者的意图调用不同的构造函数，而不仅仅是根据参数。

处理这个问题的方法之一是切换到工厂构建。我们不会使用构造函数，而是将所有 `Point` 对象都使用静态工厂方法来构建。请注意，在使用这种方法时，构造函数本身通常是私有的：

```cpp
// Example 07
class Point {
  double x_ {};
  double y_ {};
  Point(double x, double y) : x_(x), y_(y) {}
  public:
  static Point new_cartesian(double x, double y) {
    return Point(x, y);
  }
  static Point new_polar(double r, double phi) {
    return Point(r*std::cos(phi), r*std::sin(phi));
  }
};
Point p1(Point::new_cartesian(3, 4));
Point p2(Point::new_polar(5, 0.927295));
```

这种设计是可行的，但在现代 C++ 中，更受欢迎的替代方案是使用多个构造函数，并通过唯一定义的类型标签来区分它们：

```cpp
// Example 08
class Point {
  double x_ {};
  double y_ {};
  public:
  struct cartesian_t {} static constexpr cartesian {};
  Point(cartesian_t, double x, double y) : x_(x), y_(y) {}
  struct polar_t {} static constexpr polar {};
  Point(polar_t, double r, double phi) :
    Point(cartesian, r*std::cos(phi), r*std::sin(phi)) {}
};
Point p1(Point::cartesian, 3, 4);
Point p2(Point::polar, 5, 0.927295);
```

在这个例子中，我们创建了两个独特的类型，`Point::polar_t` 和 `Point::cartesian_t`，以及相应的变量，并使用它们作为标签来指定我们想要的构建类型。构造函数的重载不再模糊，因为每个都有一个独特的第一参数类型。委托构造函数使这种方法更具吸引力。

虽然使用静态函数以不同方式构建相同类型的对象有时被称为工厂方法，但它也可以被视为建造者模式的一个变体（特别是当我们使用具有类似方法的单独建造者类而不是静态方法时）。无论如何，更现代的模式——使用标签——可以替代这两种模式。在明确了术语之后，让我们回到基于运行时信息构建不同类型对象的原问题。

## 工厂方法的论据

在我们的简单示例中，构造函数没有接受任何参数。如果不同类型的构造函数有不同的参数，向构造函数传递参数会带来一些问题——毕竟，`MakeBuilding()` 函数必须用一些特定的参数声明。一个看起来很直接的选择是将工厂做成可变模板，并将参数简单地转发给每个构造函数。直接的实现可能看起来像这样：

```cpp
// Example 09
template <typename... Args>
auto Building::MakeBuilding(Type type, Args&&... args) {
  using result_t = std::unique_ptr<Building>;
  switch (type) {
    case FARM: return
      result_t{new Farm(std::forward<Args>(args)...)};
    case FORGE: return
      result_t{new Forge(std::forward<Args>(args)...)};
    ...
  }
}
```

这段代码可能甚至会在一段时间内编译，但迟早你会遇到以下错误。让我们给我们要构建的两个类提供一些构造函数参数：

```cpp
// Example 09
class Farm : public Building {
  public:
  explicit Farm(double size);
};
class Forge : public Building {
  public:
  static constexpr size_t weaponsmith = 0x1;
  static constexpr size_t welder = 0x2;
  static constexpr size_t farrier = 0x4;
  Forge(size_t staff, size_t services);
};
std::unique_ptr<Building> forge =
  Building::MakeBuilding(Building::FORGE, 2,
    Forge::weaponsmith | Forge::welder | Forge::farrier);
```

`Forge` 类使用位掩码作为标志来标记在锻造处提供哪些服务（处理少量非排他性选项的一个简单且有效的方法）。例如，如果 `(services & Forge::farrier)` 为 `true`，那么在锻造处工作的两位工匠中的一位可以为马钉蹄铁。简单、优雅，但……无法编译。

编译器错误将提到没有匹配的构造函数可用于从两个整数构造 `Farm` 类。但我们并不是试图构造一个 `Farm`！这个错误迟早会困扰到每个人。问题是，在编译时，我们无法确定我们不是试图构造一个 `Farm`：这是一个运行时决策。函数 `MakeBuilding()` 必须编译，这意味着其整个实现必须编译，包括以 `case FARM` 开头的行。你第一个想法可能是用 `if constexpr` 替换 `switch` 语句，但这不会起作用，因为我们用来选择要构建哪个类的条件不是 `constexpr`，而是一个运行时值——这正是工厂模式的意义所在。

尝试使用为 `Forge` 准备的参数来构造一个 `Farm` 是一个错误。然而，这是一个运行时错误，并且只能在运行时检测到。这仍然让我们面临如何使永远不会运行的代码有效的问题。问题是，农场没有我们可以用于所有错误参数的构造函数（但希望永远不会），最简单的解决方案是提供一个：

```cpp
// Example 09
class Farm : public Building {
  public:
  explicit Farm(...) { abort(); }
  ...
};
```

我们必须对我们可能用工厂构造的所有类型都做同样的事情。可变参数函数构造器是“最后的手段”重载——它仅在没有任何其他重载与参数匹配时才会被选中。因为它匹配任何参数，所以编译错误将消失，如果程序中出现问题，将被运行时错误所取代。为什么不简单地将这个构造函数添加到基类中呢？我们可以这样做，但基类构造函数在没有 `using` 语句的情况下在派生类中是不可见的，所以我们仍然必须为每个派生类添加一些内容。

只为了让每个类能够与工厂创建模式一起使用而必须修改每个类，这确实是一个缺点，尤其是新的构造函数可以在任何地方使用，而不仅仅是工厂函数中（不幸的是，这会产生不良后果）。像往常一样，通过引入一个重载模板，我们可以通过引入一个重载模板来解决这个问题，以构建我们的对象：

```cpp
// Example 10
template <typename T, typename... Args>
auto new_T(Args&&... args) ->
  decltype(T(std::forward<Args>(args)...))* {
  return new T(std::forward<Args>(args)...);
}
template <typename T>
T* new_T(...) { abort(); return nullptr; }
template <typename... Args>
auto Building::MakeBuilding(Type type, Args&&... args) {
  using result_t = std::unique_ptr<Building>;
  switch (type) {
    case FARM: return
      result_t{new_T<Farm>(std::forward<Args>(args)...)};
    case FORGE: return
      result_t{new_T<Forge>(std::forward<Args>(args)...)};
    ...
  }
}
```

好消息是，现在我们不需要修改任何类：任何带有正确参数的工厂调用都会编译并转发到正确的构造函数，而任何尝试使用错误参数创建对象的操作都会导致运行时错误。坏消息是，任何尝试使用错误参数创建对象的操作都会导致运行时错误。这包括我们从未计划运行的死代码（例如，使用`Forge`的参数创建`Farm`），也包括我们在调用工厂时可能犯的任何错误。

一旦你开始实现可变参数模板解决方案，它可能看起来就不那么有吸引力了，有一个更简单的选择：创建一个参数对象，其层次结构与我们要创建的对象的层次结构相匹配。让我们假设，在我们的游戏中，玩家可以为要构建的每个建筑选择升级。用户界面当然必须提供特定于建筑的选项，用户选择的结果存储在特定于建筑的对象中：

```cpp
// Example 11
struct BuildingSpec {
  virtual Building::Type type() const = 0;
};
struct FarmSpec : public BuildingSpec {
  Building::Type type() const override {
    return Building::FARM;
  }
  bool with_pasture;
  int number_of_stalls;
};
struct ForgeSpec : public BuildingSpec {
  Building::Type type() const override {
    return Building::FORGE;
  }
  bool magic_forge;
  int number_of_apprentices;
};
```

注意，我们在参数对象中包含了类型标识符，没有理由用两个必须始终正确匹配的参数调用工厂方法；这只会增加出错的可能性。这样，我们就可以保证在每个工厂调用中类型标识符和参数是一致的：

```cpp
// Example 11
auto Building::MakeBuilding(const BuildingSpec& spec) {
  using result_t = std::unique_ptr<Building>;
  switch (spec.type()) {
    case FARM: return result_t{
      new Farm(static_cast<const FarmSpec&>(spec))};
    case FORGE: return result_t{
      new Forge(static_cast<const ForgeSpec&>(spec))};
    ...
  }
}
```

注意，工厂模式通常与我们在*第九章*中看到的命名参数模式配合得很好，*命名参数、方法链和构建器模式*，以避免需要指定长的参数列表。规范对象本身就成了我们可以用来指定命名参数的选项对象：

```cpp
// Example 11
class FarmSpec {
  ...
  bool with_pasture {};
  int number_of_stalls {};
  FarmSpec() = default;
  FarmSpec& SetPasture(bool with_pasture) {
    this->with_pasture = with_pasture;
    return *this;
  }
  FarmSpec& SetStalls(int number_of_stalls) {
    this->number_of_stalls = number_of_stalls;
    return *this;
  }
};
struct ForgeSpec : public BuildingSpec {
  ...
  bool magic_forge {};
  int number_of_apprentices {};
  ForgeSpec() = default;
  ForgeSpec& SetMagic(bool magic_forge) {
    this->magic_forge = magic_forge;
    return *this;
  }
  ForgeSpec& SetApprentices(int number_of_apprentices) {
    this->number_of_apprentices = number_of_apprentices;
    return *this;
  }
};
...
std::unique_ptr<Building> farm =
  Building::MakeBuilding(FarmSpec()
                         .SetPasture(true)
                         .SetStalls(2));
std::unique_ptr<Building> forge =
  Building::MakeBuilding(ForgeSpec()
                         .SetMagic(false)
                         .SetApprentices(4));
```

这种技术可以与以下章节中展示的其他工厂变体结合使用，这样我们就可以在构造函数需要时传递参数。

## 动态类型注册

到目前为止，我们假设类型的完整列表在编译时已知，并且可以编码在类型标识符对应表中（在我们的例子中通过 switch 语句实现）。在程序的全局范围内，这个要求是不可避免的：因为每个构造函数调用都必须在某个地方显式编写，因此可以构造的类型列表在编译时是已知的。然而，我们的解决方案比这更受限制——我们有一个硬编码在工厂方法中的所有类型的列表。没有添加到工厂中，就不能创建额外的派生类。有时，这种限制并不像看起来那么糟糕——例如，游戏中的建筑列表可能不会经常改变，即使它改变了，也必须有一个完整的列表手动更新，以便正确生成菜单，图片和声音出现在正确的位置等等。

尽管如此，分层设计的优点之一是，可以在不修改任何操作该层次结构的代码的情况下，稍后添加派生类。新的虚拟函数只需插入到现有的控制流程中，并提供必要的定制行为。我们可以为工厂构造函数实现同样的想法。

首先，每个派生类都必须负责构建自身。这是必要的，因为我们已经了解到，显式调用构造函数必须在某个地方编写。如果它不在通用代码中，它就必须是创建新派生类时添加的代码的一部分。例如，我们可以有一个静态工厂函数：

```cpp
class Forge : public Building {
  public:
  static Building* MakeBuilding() { return new Forge; }
};
```

其次，类型的列表必须在运行时可扩展，而不是在编译时固定。我们仍然可以使用`enum`，但每次添加新的派生类时，它都必须更新。或者，我们可以在运行时为每个派生类分配一个整数标识符，确保标识符是唯一的。无论如何，我们需要一个将这些标识符映射到工厂函数的映射，而且它不能是一个在编译时固定的`switch`语句或其他任何东西。这个映射必须是可扩展的。我们可以使用`std::map`来实现这一点，但如果类型标识符是整数，我们也可以使用一个按类型标识符索引的函数指针的`std::vector`：

```cpp
class Building;
using BuildingFactory = Building*(*)();
std::vector<BuildingFactory> building_registry;
```

现在，为了注册一个新的类型，我们只需生成一个新的标识符，并将相应的工厂函数添加到向量中：

```cpp
size_t building_type_count = 0;
void RegisterBuilding(BuildingFactory factory) {
  building_registry.push_back(factory));
  ++building_type_count;
}
```

这种注册机制可以封装在基类本身中：

```cpp
// Example 12
class Building {
  static size_t building_type_count;
  using BuildingFactory = Building* (*)();
  static std::vector<BuildingFactory> registry;
  public:
  static size_t (BuildingFactory factory) {
    registry.push_back(factory);
    return building_type_count++;
  }
  static auto MakeBuilding(size_t building_type) {
    BuildingFactory RegisterBuilding factory =
        registry[building_type];
    return std::unique_ptr<Building>(factory());
  }
};
std::vector<Building::BuildingFactory> Building::registry;
size_t Building::building_type_count = 0;
```

这个基类具有工厂函数表和已注册派生类型的计数作为静态数据成员。它还具有两个静态函数：一个用于注册新类型，另一个用于构造类中注册的一个类型的对象。请注意，注册函数返回与工厂函数关联的类型标识符。我们很快就会用到这个。

现在，我们只需要将每个新的建筑类型添加到注册表中。这是分两步完成的——首先，我们需要为每个建筑类添加一个注册方法，如下所示：

```cpp
class Forge : public Building {
  public:
  static void Register() {
    RegisterBuilding(Forge::MakeBuilding);
  }
};
```

第二，我们需要确保在游戏开始之前调用所有`Register()`方法，并确保我们知道每个建筑类型的正确标识符。这就是`RegisterBuilding()`函数返回的值变得重要的地方，因为我们将它作为类型标识符存储在类内部：

```cpp
// Example 12
class Forge : public Building {
  public:
  static void Register() {
    RegisterBuilding(Forge::MakeBuilding);
  }
  static const size_t type_tag;
};
const size_t Forge::type_tag =
  RegisterBuilding(Forge::MakeBuilding);
```

注册发生在静态变量的初始化过程中，在`main()`开始之前。

工厂函数不必是静态成员函数：任何可以通过函数指针调用的东西都可以工作。例如，我们可以使用没有捕获的 lambda：

```cpp
// Example 12
class Farm : public Building {
  public:
  static const size_t type_tag;
};
const size_t Farm::type_tag =
  RegisterBuilding([]()->Building* { return new Farm; });
```

我们必须显式指定返回类型，因为函数指针类型被定义为没有参数且返回`Building*`的函数，而 lambda 被推断为返回`Farm*`的函数，除非我们转换返回值或指定返回类型。

现在，调用`Building::MakeBuilding(tag)`将构造一个与标识符`tag`注册的类型对象。标签的值——类型标识符——作为每个类的静态成员存储，因此我们不必知道它，也不会出错：

```cpp
std::unique_ptr<Building> farm =
  Building::MakeBuilding(Farm::type_tag);
std::unique_ptr<Building> forge =
  Building::MakeBuilding(Forge::type_tag);
```

在我们的解决方案中，标识符值与类型之间的对应关系直到运行时才知道——在我们运行程序之前，我们无法说出哪个建筑物的 ID 是 5。通常，我们不需要知道这一点，因为正确的值会自动存储在每个类中。

注意，这个实现与编译器为真正的虚函数生成的代码非常相似——虚函数调用是通过存储在表中并通过唯一标识符（虚指针）访问的函数指针完成的。主要区别是唯一标识符是与每个类型关联的静态数据成员。尽管如此，这几乎就是一个*虚拟构造函数*。

这种动态类型注册模式有许多变体。在某些情况下，显式指定类型标识符比在程序启动时生成它们更好。特别是像“农场”和“锻造厂”这样的可读名称可能很有用。在这种情况下，我们可以将工厂函数指针存储在以字符串为索引的`std::map`容器中（`std::map<std::string, BuildingFactory>`）。

另一个修改是允许更通用的可调用对象作为工厂函数。我们可以通过使用`std::function`而不是函数指针来泛化`BuildingFactory`类型：

```cpp
using BuildingFactory = std::function<Building*()>;
```

我们仍然可以将静态工厂方法注册并用作派生类的工厂，但我们也可以使用 lambda 和自定义仿函数：

```cpp
// Example 13
class Forge : public Building {
  public:
  static const size_t type_tag;
};
class ForgeFactory {
  public:
  Building* operator()() const { return new Forge; }
};
const size_t Forge::type_tag =
  RegisterBuilding(ForgeFactory{});
```

这些动态工厂的实现，无论是使用函数指针还是更通用的 std::function，都与我们在*第六章*，“理解类型擦除”中探讨的类型擦除模式非常相似。要构建的对象的具体类型嵌入在函数或函数对象的代码中，其声明没有提及这些类型。这允许我们将这些函数存储在单个函数表或映射中。同样，*第六章*，“理解类型擦除”中的其他类型擦除实现也可以使用。

为了简化，我们没有为我们的工厂方法使用任何参数。然而，我们在上一节中探讨了传递参数的选项。可变模板与函数指针（我们必须提前声明工厂函数的签名）配合得不好，因此传递参数的最可能模式将是参数规范对象：

```cpp
// Example 14
struct BuildingSpec {};
class Building {
  ...
  using BuildingFactory =
    Building* (*)(const BuildingSpec&);
  static auto MakeBuilding(size_t building_type,
                           const BuildingSpec& spec) {
    BuildingFactory factory = registry[building_type];
    return std::unique_ptr<Building>(factory(spec));
  }
};
struct FarmSpec : public BuildingSpec {
  bool with_pasture {};
  int number_of_stalls {};
  FarmSpec() = default;
  FarmSpec& SetPasture(bool with_pasture) {
    this->with_pasture = with_pasture;
    return *this;
  }
  FarmSpec& SetStalls(int number_of_stalls) {
    this->number_of_stalls = number_of_stalls;
    return *this;
  }
};
class Farm : public Building {
  public:
  explicit Farm(const FarmSpec& spec);
  ...
};
const size_t Farm::type_tag = RegisterBuilding(
  [](const BuildingSpec& spec)->Building* {
    return new Farm(static_cast<const FarmSpec&>(spec));
  });
struct ForgeSpec : public BuildingSpec { ... };
class Forge : public Building { ... };
std::unique_ptr<Building> farm =
  Building::MakeBuilding(FarmSpec()
                         .SetPasture(true)
                         .SetStalls(2));
std::unique_ptr<Building> forge =
  Building::MakeBuilding(ForgeSpec()
                         .SetMagic(false)
                         .SetApprentices(4));
```

在我们迄今为止的所有工厂构造函数中，关于构建哪个对象的决策是由程序的外部输入驱动的，并且构建是通过相同的工厂方法完成的（可能使用对派生类的委托）。现在我们将看到工厂的不同变体，它用于解决一个稍微不同的场景。

## 多态工厂

考虑一个稍微不同的问题——想象在我们的游戏中，每个建筑都生产某种单位，并且单位的类型与建筑的类型唯一相关联。城堡招募骑士，巫师塔训练法师，蜘蛛山产生巨型蜘蛛。现在，我们的通用代码不仅构建在运行时选择的建筑类型，还创建新的单位，其类型在编译时也不为人知。我们已经有建筑工厂。我们可以以类似的方式实现单位工厂，其中每个建筑都有一个与其关联的唯一单位标识符。但这个设计将单位与建筑之间的对应关系暴露给了程序的其他部分，这实际上并不是必要的——每个建筑都知道如何构建*正确的*单位；程序的其他部分没有必要也知道它。

这个设计挑战需要一种稍微不同的工厂——工厂方法决定创建一个单位，但确切是哪个单位则由建筑来决定。这是模板模式的应用，结合了工厂模式——整体设计是工厂，但单位类型由派生类定制：

```cpp
// Example 15
class Unit {};
class Knight : public Unit { ... };
class Mage : public Unit { ... };
class Spider : public Unit { ... };
class Building {
  public:
  virtual Unit* MakeUnit() const = 0;
};
class Castle : public Building {
  public:
  Knight* MakeUnit() const { return new Knight; }
};
class Tower : public Building {
  public:
  Mage* MakeUnit() const { return new Mage; }
};
class Mound : public Building {
  public:
  Spider* MakeUnit() const { return new Spider; }
};
```

每座建筑都有一个用于创建相应单位的工厂，我们可以通过基类 `Building` 访问这些工厂方法：

```cpp
std::vector<std::unique_ptr<Building>> buildings;
std::vector<std::unique_ptr<Unit>> units;
for (const auto& b : buildings) {
  units.emplace_back(b->MakeUnit());
}
```

使用多态并通过基类中的虚拟函数（通常是纯虚拟函数）访问的工厂被称为抽象工厂模式。

本例中没有展示建筑本身的工厂方法——单元工厂可以与我们所学的任何建筑工厂实现共存（伴随本章的源代码示例使用的是示例 12 中的建筑工厂）。从建筑构建单位的通用代码只写一次，当添加新的建筑和单位派生类时不需要更改。

注意，所有 `MakeUnit()` 函数的返回类型都不同。尽管如此，它们都是同一虚拟 `Building::MakeUnit()` 函数的重写。这些被称为**协变返回类型**——重写方法的返回类型可能是被重写方法返回类型的派生类。在我们的例子中，返回类型与类类型相匹配，但通常这并不是必需的。任何基类和派生类都可以用作协变类型，即使它们来自不同的层次结构。然而，只有这样的类型才能是协变的，除此之外的例外，重写的返回类型必须与基类虚拟函数相匹配。

当我们尝试使工厂返回除原始指针之外的内容时，协变返回类型的严格规则会带来一些问题。例如，假设我们想要返回 `std::unique_ptr` 而不是原始指针。但是，与 `Unit*` 和 `Knight*` 不同，`std::unique_ptr<Unit>` 和 `std::unique_ptr<Knight>` 不是协变类型，不能用作虚拟方法和其重写的返回类型。

我们将在下一节考虑这个解决方案以及与工厂方法相关的几个其他特定于 C++ 的问题。

# C++ 中的工厂类似模式

在 C++ 中，用于解决特定设计需求和约束的基本工厂模式的变体有很多。在本节中，我们将考虑其中的一些。这绝对不是 C++ 中工厂类似模式的独家列表，但理解这些变体应该为读者准备将他们从本书中学到的技术结合起来，以解决与对象工厂相关的各种设计挑战。

## 多态复制

到目前为止，我们考虑了对象构造函数的替代方案——要么是默认构造函数，要么是带有参数的构造函数之一。然而，可以将类似的模式应用于复制构造函数——我们有一个对象，我们想要复制它。

这在许多方面是一个类似的问题——我们有一个通过基类指针访问的对象，我们想要调用它的复制构造函数。由于我们之前讨论的原因，包括编译器需要知道分配多少内存在内的原因，实际的构造函数调用必须在静态确定的类型上完成。然而，将我们带到特定构造函数调用的控制流可以在运行时确定，这再次需要应用工厂模式。

我们将使用的工厂方法来实现多态复制与上一节中的 Unit 工厂示例有些相似——实际的构建必须由每个派生类来完成，派生类知道要构建哪种类型的对象。基类实现了控制流，决定了将构建某个人的副本，并且派生类定制了构建部分：

```cpp
// Example 16
class Base {
  public:
  virtual Base* clone() const = 0;
};
class Derived : public Base {
  public:
  Derived* clone() const override {
    return new Derived(*this);
  }
};
Base* b0 = ... get an object somewhere ...
Base* b1 = b->clone();
```

我们可以使用 typeid 操作符（可能结合本章前面使用过的解名函数）来验证指针 `b1` 确实指向一个 `Derived` 对象。

我们刚刚通过继承实现了多态复制。在*第六章* *理解类型擦除*中，我们看到了另一种在构建时已知类型但后来丢失（或擦除）的对象的复制方法。这两种方法在本质上并没有不同：在实现类型擦除复制时，我们自行构建了一个虚表。在本章中，我们让编译器为我们完成这项工作。在任何特定情况下，首选的实现方式主要取决于代码周围的其它内容。

注意，我们又使用了协变返回类型，因此我们被限制为只能返回原始指针。假设我们想返回唯一指针。由于只有基类和派生类的原始指针被认为是协变的，我们必须始终返回基类唯一指针：

```cpp
class Base {
  public:
  virtual std::unique_ptr<Base> clone() const = 0;
};
class Derived : public Base {
  public:
  std::unique_ptr<Base> clone() const override {
    return std::unique_ptr<Base>(new Derived(*this));
  }
};
std::unique_ptr<Base> b(... make an object ...);
std::unique_ptr<Base> b1 = b->clone();
```

在许多情况下，这并不是一个重大的限制。然而，有时它可能导致不必要的转换和强制类型转换。如果返回精确类型的智能指针很重要，我们将考虑这个模式的另一个版本。

## CRTP 工厂和返回类型

从派生类的工厂复制构造函数中返回 `std::unique_ptr<Derived>` 的唯一方法是将基类的虚拟 `clone()` 方法返回相同的类型。但这是不可能的，至少如果我们有多个派生类——对于每个派生类，我们需要 `Base::clone()` 的返回类型是该类。但只有一个 `Base::clone()`！或者有吗？幸运的是，在 C++ 中，我们有一种简单的方法可以将一个变成多个——那就是模板。如果我们模板化基类，我们就可以使每个派生类的基类返回正确的类型。但要做到这一点，我们需要基类以某种方式知道将从它派生出的类的类型。当然，也有一个模式，在 C++ 中被称为“奇特重复模板模式”，我们在*第八章* *奇特重复模板模式*中讨论过。现在，我们可以结合 CRTP 和工厂模式：

```cpp
// Example 18
template <typename Derived> class Base {
  public:
  virtual std::unique_ptr<Derived> clone() const = 0;
};
class Derived : public Base<Derived> {
  public:
  std::unique_ptr<Derived> clone() const override {
    return std::unique_ptr<Derived>(new Derived(*this));
  }
};
std::unique_ptr<Derived> b0(new Derived);
std::unique_ptr<Derived> b1 = b0->clone();
```

`auto` 返回类型使得编写这样的代码显著减少了冗余。在这本书中，我们通常不使用它们来明确指出哪个函数返回什么。

`Base`类的模板参数是从它派生的一个类，因此得名。如果你愿意，甚至可以使用静态断言来强制这种限制：

```cpp
template <typename Derived> class Base {
  public:
  virtual std::unique_ptr<Derived> clone() const = 0;
  Base() {
    static_assert(std::is_base_of_v<Base, Derived>;
  }
};
```

我们必须将静态断言隐藏在类构造函数中的原因是在类本身的声明中，`Derived`类型是不完整的。

注意，由于`Base`类现在知道派生类的类型，我们甚至不需要`clone()`方法为虚函数：

```cpp
// Example 19
template <typename Derived> class Base {
  public:
  std::unique_ptr<Derived> clone() const {
    return std::unique_ptr<Derived>(
      new Derived(*static_cast<const Derived*>(this)));
  }
};
class Derived : public Base<Derived> { ... };
```

这种方法存在一些显著的缺点，至少就我们迄今为止的实现方式而言。首先，我们必须将基类做成模板，这意味着我们不再有一个通用的指针类型可以在我们的通用代码中使用（或者我们必须更广泛地使用模板）。其次，这种方法只有在没有更多类从`Derived`类派生时才有效，因为基类的类型不跟踪第二次派生——只有实例化了`Base`模板的那个派生。总的来说，除了在必须返回确切类型而不是基类型的情况下非常重要的一些特定情况外，这种方法不推荐使用。

另一方面，这种实现有一些吸引人的特性，我们可能希望保留。具体来说，我们消除了`clone()`函数的多个副本，每个派生类一个，并得到了一个模板来为我们自动生成它们。在下一节中，我们将向您展示如何保留 CRTP 实现的有用特性，即使我们必须放弃通过模板技巧将协变返回类型的概念扩展到智能指针。

## CRTP 用于工厂实现

到目前为止，我们已经多次提到，虽然 CRTP 有时被用作设计工具，但它同样可能被用作实现技术。现在我们将专注于使用 CRTP 来避免在每个派生类中编写`clone()`函数。这不仅仅是为了减少打字——代码写得越多——特别是那些被复制和修改的非常相似的代码——你犯错误的可能性就越大。我们已经看到如何使用 CRTP 自动为每个派生类生成一个`clone()`版本。我们只是不想为了这样做而放弃通用的（非模板）基类。如果我们把克隆委托给只处理那个的特殊基类，我们实际上并不需要这样做：

```cpp
// Example 20
class Base {
  public:
  virtual Base* clone() const = 0;
};
template <typename Derived> class Cloner : public Base {
  public:
  Base* clone() const {
    return new Derived(*static_cast<const Derived*>(this));
  }
};
class Derived : public Cloner<Derived> {
  ...
};
Base* b0(new Derived);
Base* b1 = b0->clone();
```

在这里，为了简单起见，我们回到了返回原始指针，尽管我们也可以返回`std::unique_ptr<Base>`。我们无法返回`Derived*`，因为在解析`Cloner`模板时，并不知道`Derived`总是从`Base`派生。

这种设计使我们能够从`Base`派生出任意数量的类，通过`Cloner`间接实现，而且再也不需要编写另一个`clone()`函数。它仍然存在一个限制，即如果我们从`Derived`派生另一个类，它将无法正确复制。在许多设计中，这并不是一个问题——明智的自利应该引导你避免深层层次结构，并使所有类成为两种类型之一：永远不会实例化的抽象基类，以及从这些基类之一派生出来的具体类，但永远不会从另一个具体类派生。

## 工厂和 Builder

到目前为止，我们主要使用的是工厂函数，或者更一般地说，是像 lambda 这样的函数对象。在实践中，我们同样可能需要一个工厂类。这通常是因为构建对象所需的运行时信息比仅仅类型标识符和一些参数更复杂。这也是我们可能选择使用 Builder 模式来创建对象的原因，因此工厂类也可以被视为一个具有工厂方法的构建类，用于创建具体对象。我们在本章前面看到的 Unit 工厂就是一个这样的模式示例：Building 类及其所有派生类充当单元对象的构建器（而且建筑对象本身是由另一个工厂创建的，这又是一个证明，即即使是简单的代码也很难仅用一种模式来简化）。然而，在这种情况下，我们使用工厂类有一个特殊的原因：每个派生建筑类都构建自己的单元对象。

让我们现在考虑使用工厂类的一个更常见的场景：决定构建哪个类以及如何构建的运行时数据的整体复杂性，以及我们需要执行此操作的非平凡代码量。虽然我们可以使用工厂函数和一些全局对象来处理所有这些，但这将是一个糟糕的设计，缺乏凝聚力和封装。这将容易出错且难以维护。将所有相关代码和数据封装到一个类或少数相关类中会更好。

对于这个例子，我们将解决一个非常常见（但仍然具有挑战性）的序列化/反序列化问题。在我们的情况下，我们有一些从相同基类派生的对象。我们希望通过将它们写入文件来实现序列化，然后从该文件中恢复对象。在本章的最后一个例子中，我们将结合我们学到的几种方法来设计和实现工厂。

我们将从基类开始。基类将利用我们之前学到的动态类型注册表。此外，它将声明一个纯虚`Serialize()`函数，每个派生类都需要实现以将自身序列化到文件中：

```cpp
// Example 21
class SerializerBase {
  static size_t type_count;
  using Factory = SerializerBase* (*)(std::istream& s);
  static std::vector<Factory> registry;
  protected:
  virtual void Serialize(std::ostream& s) const = 0;
  public:
  virtual ~SerializerBase() {}
  static size_t RegisterType(Factory factory) {
    registry.push_back(factory);
    return type_count++;
  }
  static auto Deserialize(size_t type, std::istream& s) {
    Factory factory = registry[type];
    return std::unique_ptr<SerializerBase>(factory(s));
  }
};
std::vector<SerializerBase::Factory>
  SerializerBase::registry;
size_t SerializerBase::type_count = 0;
```

任何派生类都需要实现`Serialize()`函数以及注册反序列化函数：

```cpp
// Example 21
class Derived1 : public SerializerBase {
  int i_;
  public:
  Derived1(int i) : i_(i) {...}
  void Serialize(std::ostream& s) const override {
    s << type_tag << " " << i_ << std::endl;
  }
  static const size_t type_tag;
};
const size_t Derived1::type_tag =
  RegisterType([](std::istream& s)->SerializerBase* {
    int i; s >> i; return new Derived1(i); });
```

只有派生类本身才有关于其状态的信息，为了重新构成对象必须保存什么，以及如何保存。在我们的例子中，序列化总是在`Serialize()`函数中完成的，而反序列化是在我们向类型注册表中注册的 lambda 中完成的。不用说，这两者必须相互一致。有一些基于模板的技巧可以确保这种一致性，但它们与我们正在研究的工厂构建无关。

我们已经处理了序列化部分——我们只需要在任何一个对象上调用 Serialize 即可：

```cpp
std::ostream S ... – construct the stream as needed
Derived1 d(42);
d.Serialize(S);
```

反序列化本身并不特别困难（大部分工作由派生类完成），但其中足够的样板代码足以证明工厂类的必要性。工厂对象将读取整个文件并反序列化（重新创建）其中记录的所有对象。当然，这些对象有许多可能的用途。由于我们正在构建在编译时类型未知的对象，我们必须通过基类指针来访问它们。例如，我们可以将它们存储在唯一指针的容器中：

```cpp
// Example 21
class DeserializerFactory {
  std::istream& s_;
  public:
  explicit DeserializerFactory(std::istream& s) : s_(s) {}
  template <typename It>
  void Deserialize(It iter) {
    while (true) {
      size_t type;
      s_ >> type;
      if (s_.eof()) return;
      iter = SerializerBase::Deserialize(type, s_);
    }
  }
};
```

这个工厂逐行读取整个文件。首先，它只读取类型标识符（每个对象在序列化时都必须写入）。基于这个标识符，它将剩余的反序列化过程调度到为相应类型注册的正确函数。工厂使用插入迭代器（如后插入迭代器）将所有反序列化的对象存储在容器中：

```cpp
// Example 21
std::vector<std::unique_ptr<SerializerBase>> v;
DeserializerFactory F(S);
F.Deserialize(std::back_inserter(v));
```

使用这种方法，我们可以处理任何从 SerializerBase 派生的类，只要我们能想出一个将其写入文件并恢复的方法。我们可以处理更复杂的状态和具有多个参数的构造函数：

```cpp
// Example 21
class Derived2 : public SerializerBase {
  double x_, y_;
  public:
  Derived2(double x, double y) : x_(x), y_(y) {...}
  void Serialize(std::ostream& s) const override {
    s << type_tag << " " << x_ << " " << y_ << std::endl;
  }
  static const size_t type_tag;
};
const size_t Derived2::type_tag =
  RegisterType([](std::istream& s)->SerializerBase* {
    double x, y; s >> x >> y;
    return new Derived2(x, y);
});
```

只要我们知道如何再次构造一个特定的对象，我们同样可以轻松地处理具有多个构造函数的类：

```cpp
// Example 21
class Derived3 : public SerializerBase {
  bool integer_;
  int i_ {};
  double x_ {};
  public:
  Derived3(int i) : integer_(true), i_(i) {...}
  Derived3(double x) : integer_(false), x_(x) {...}
  void Serialize(std::ostream& s) const override {
    s << type_tag << " " << integer_ << " ";
    if (integer_) s << i_; else s << x_;
    s << std::endl;
  }
  static const size_t type_tag;
};
const size_t Derived3::type_tag =
  RegisterType([](std::istream& s)->SerializerBase* {
    bool integer; s >> integer;
    if (integer) {
      int i; s >> i; return new Derived3(i);
    } else {
      double x; s >> x; return new Derived3(x);
  }
});
```

C++中有许多工厂模式的变体。如果你理解了本章的解释并跟随了示例，这些替代方案对你来说应该不会构成特别的挑战。

# 摘要

在本章中，我们学习了为什么不能使构造函数成为虚拟的，以及当我们真的需要一个虚拟构造函数时应该做什么。我们学习了如何通过使用工厂模式和其变体之一来构建和复制在运行时类型已知的对象。我们还探讨了几个工厂构造函数的实现，它们在代码组织方式和将行为委派给系统不同组件的方式上有所不同，并比较了它们的优缺点。我们还看到了多个设计模式如何相互作用。

虽然，在 C++中，构造函数必须使用要构造的对象的真实类型来调用——总是这样——但这并不意味着应用程序代码必须指定完整的类型。工厂模式允许我们编写代码，通过使用与类型关联的标识符间接指定类型（*创建第三种类型的对象*），或关联的对象类型（*创建与这种建筑类型相匹配的单位*），甚至相同的类型（*给我一个这个的副本，无论它是什么*）。

在下一章中，我们将学习的设计模式是模板方法模式，这是经典面向对象模式之一，在 C++中，它对我们设计类层次结构有额外的含义。

# 问题

1.  为什么 C++不允许虚拟构造函数？

1.  工厂模式是什么？

1.  你如何使用 Factory 模式来实现虚拟构造函数的效果？

1.  你如何实现虚拟拷贝构造函数的效果？

1.  你如何一起使用模板和 Factory 模式？

1.  你如何一起使用 Builder 和 Factory 模式？
