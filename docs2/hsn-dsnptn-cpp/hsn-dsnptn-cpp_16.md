# 16

# 适配器和装饰者

本章探讨了面向对象编程（**OOP**）中的两个经典模式——适配器模式和装饰者模式。这些模式只是 Erich Gamma、Richard Helm、Ralph Johnson 和 John Vlissides 在《设计模式——可重用面向对象软件元素》一书中介绍的二十三个原始设计模式中的两个。作为一个面向对象的语言，C++可以利用这些模式，就像任何其他语言一样。但是，正如通常情况那样，泛型编程为经典模式带来了某些优势、变化，以及随之而来的新挑战。

本章节涵盖了以下主题：

+   适配器和装饰者模式是什么？

+   两者之间的区别是什么？

+   这些模式可以解决哪些设计问题？

+   这些模式如何在 C++中使用？

+   泛型编程如何帮助设计适配器和装饰者？

+   其他不同模式如何提供类似问题的替代解决方案？

# 技术要求

本章的示例代码可以在以下 GitHub 链接中找到：[`github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/master/Chapter16`](https://github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/master/Chapter16)。

# 装饰者模式

我们将从这个研究开始，先定义这两个经典模式。正如我们将看到的，在纸上，模式以及它们之间的区别非常清晰。然后，C++介入，通过允许介于两者之间的设计解决方案来模糊界限。尽管如此，这些简单案例的清晰性仍然是有帮助的，即使在我们增加复杂性时它可能会变得混乱。让我们从清晰的地方开始。

装饰者模式也是一种结构型模式；它允许向对象添加行为。经典的装饰者模式扩展了由一个类执行的操作的行为。它通过添加新的行为来**装饰**这个类，并创建了一个新装饰类型的对象。装饰者实现了原始类的接口，并将请求从其接口转发到那个类，但它还执行了在转发请求之前和之后的一些额外操作——这些就是**装饰**。这种装饰者有时被称为“类包装器”。

## 基本装饰者模式

我们将从一个尽可能接近经典定义的 C++装饰者模式示例开始。为此示例，我们将设想设计一个设定在中世纪时期的幻想游戏（真实生活，只是有龙和精灵等等）。当然，没有战斗的中世纪时代是什么样的？因此，在我们的游戏中，玩家可以选择适合他/她的单位，并在被召唤时进行战斗。以下是基本的`Unit`类——至少是战斗相关的部分：

```cpp
// Example 01
class Unit {
  public:
  Unit(double strength, double armor) :
    strength_(strength), armor_(armor) {}
  virtual bool hit(Unit& target) {
    return attack() > target.defense();
  }
  virtual double attack() = 0;
  virtual double defense() = 0;
  protected:
  double strength_;
  double armor_;
};
```

单位有`strength`属性，它决定了其攻击力，以及`armor`属性，它提供防御。攻击和防御的实际值由派生类——具体的单位——计算，但战斗机制本身就在这里——如果攻击力强于防御力，单位就成功地击中了目标（当然，这是一个非常简化的游戏方法，但我们想使示例尽可能简洁）。

现在，游戏中实际有哪些单位？人类军队的支柱是英勇的`Knight`。这个单位拥有坚固的盔甲和锋利的剑，使它在攻击和防御上都获得加成：

```cpp
// Example 01
class Knight : public Unit {
  public:
  using Unit::Unit;
  double attack() { return strength_ + sword_bonus_; }
  double defense() { return armor_ + plate_bonus_; }
  protected:
  static constexpr double sword_bonus_ = 2;
  static constexpr double plate_bonus_ = 3;
};
```

与骑士战斗的是粗鲁的巨魔。巨魔挥舞着简单的木棍，穿着破旧的皮革，这两者都不是很好的战争工具，给它们带来了一些战斗上的惩罚：

```cpp
// Example 01
class Ogre : public Unit {
  public:
  using Unit::Unit;
  double attack() { return strength_ + club_penalty_; }
  double defense() { return armor_ + leather_penalty_; }
  protected:
  static constexpr double club_penalty_ = -1;
  static constexpr double leather_penalty_ = -1;
};
```

另一方面，巨魔一开始就非常强壮：

```cpp
Knight k(10, 5);
Ogre o(12, 2);
k.hit(o); // Yes!
```

在这里，骑士凭借他的攻击加成和敌人的薄弱盔甲，成功地击中了巨魔。但游戏远未结束。随着单位的战斗，幸存者获得经验，最终成为老兵。老兵单位仍然是同一种单位，但它获得了攻击和防御加成，反映了它的战斗经验。在这里，我们不想改变任何类接口，但想修改`attack()`和`defense()`函数的行为。这就是装饰者模式的工作，以下是对`VeteranUnit`装饰者的经典实现：

```cpp
// Example 01
class VeteranUnit : public Unit {
  public:
  VeteranUnit(Unit& unit,
              double strength_bonus,
              double armor_bonus) :
    Unit(strength_bonus, armor_bonus), unit_(unit) {}
  double attack() { return unit_.attack() + strength_; }
  double defense() { return unit_.defense() + armor_; }
  private:
  Unit& unit_;
};
```

注意，这个类直接从`Unit`类继承，所以在类层次结构中，它位于具体单位类（如`Knight`或`Ogre`）的旁边。我们仍然有原始单位，它被装饰并成为老兵——`VeteranUnit`装饰者包含对其的引用。它的使用方式是装饰一个单位，然后继续使用装饰过的单位，但它不会删除原始单位：

```cpp
// Example 01
Knight k(10, 5);
Ogre o(12, 2);
VeteranUnit vk(k, 7, 2);
VeteranUnit vo(o, 1, 9);
vk.hit(vo); // Another hit!
```

在这里，我们两个老对手都达到了他们的第一个军衔等级，胜利再次属于骑士。但经验是最好的老师，我们的巨魔又提升了一个等级，并且，随着它，附有巨大防御加成的魔法符文盔甲：

```cpp
VeteranUnit vvo(vo, 1, 9);
vk.hit(vvo); // Miss!
```

注意，在这个设计中，我们可以装饰一个装饰过的对象！这是故意的，并且随着单位等级的提升，加成会叠加。这次，经验丰富的战士的防御力证明对骑士来说太过强大。

正如我们之前提到的，这是一个经典的装饰者模式，直接来自教科书。它在 C++中工作，但有一些限制。第一个限制相当明显，即使我们一旦拥有装饰过的单位，原始单位也必须保留，并且这些对象的生命周期必须仔细管理。对于这样的实际问题，有实际的解决方案，但本书的重点是结合设计模式和泛型编程，以及这种配对创造的新设计可能性。因此，我们的创新之路将带我们走向其他地方。

第二个问题在 C++中更为普遍。最好通过一个例子来说明。游戏的设计师为`Knight`单位添加了特殊能力——它可以向前冲锋攻击敌人，获得短期的攻击加成。这个加成只对下一次攻击有效，但在激烈的战斗中，这也许正是所需要的：

```cpp
// Example 02
class Knight : public Unit {
  public:
  Knight(double strength, double armor) :
  Unit(strength, armor), charge_bonus_(0) {}
  double attack() {
    double res = strength_ + sword_bonus_ + charge_bonus_;
    charge_bonus_ = 0;
    return res;
  }
  double defense() { return armor_ + plate_bonus_; }
  void charge() { charge_bonus_ = 1; }
  protected:
  double charge_bonus_;
  static constexpr double sword_bonus_ = 2;
  static constexpr double plate_bonus_ = 3;
};
```

充能加成是通过调用`charge()`成员函数激活的，持续一次攻击，然后重置。当玩家激活充能时，游戏执行以下代码：

```cpp
Knight k(10, 5);
Ogre o(12, 2);
k.charge();
k.hit(o);
```

当然，我们期望老兵骑士也能向前冲锋，但在这里我们遇到了问题——我们的代码无法编译：

```cpp
VeteranUnit vk(k, 7, 2);
vk.charge(); // Does not compile!
```

问题的根源在于`charge()`方法是`Knight`类接口的一部分，而`VeteranUnit`装饰器是从`Unit`类派生出来的。我们可以将`charge()`函数移动到基类`Unit`中，但这是一种糟糕的设计——`Ogre`也是从`Unit`派生出来的，而哥布林不能冲锋，因此它们不应该有这样的接口（这违反了公共继承的*is-a*原则）。

这是一个固存在于我们实现装饰器对象的方式中的问题——`Knight`和`VeteranUnit`都从同一个基类`Unit`派生，但它们对彼此一无所知。有一些丑陋的解决方案，但这是一种基本的 C++限制；它处理*交叉转换*（在相同层次结构的另一分支中转换类型）不佳。但语言一手拿走，另一手又给予——我们有许多更好的工具来处理这个问题，我们将在下一章学习这些工具。

## C++风格的装饰器

在实现 C++中的经典装饰器时，我们遇到了两个问题——首先，装饰对象没有接管原始对象的所有权，因此两者都必须保留（如果装饰需要稍后移除，这可能不是问题，而是特性之一，这也是装饰器模式以这种方式实现的原因之一）。另一个问题是，装饰后的`Knight`根本不是`Knight`，而是`Unit`。如果装饰器本身是从被装饰的类派生出来的，我们就可以解决第二个问题。这意味着`VeteranUnit`类没有固定的基类——基类应该是被装饰的任何类。这种描述与**Curiously Recurring Template Pattern**（**CRTP**）完全吻合（这种 C++习语在本书的*第八章*中已有描述，即《Curiously Recurring Template Pattern》）。要应用 CRTP，我们需要将装饰器做成模板，并从模板参数继承：

```cpp
// Example 03
template <typename U>
class VeteranUnit : public U {
  public:
  VeteranUnit(U&& unit,
              double strength_bonus,
              double armor_bonus) :
    U(unit), strength_bonus_(strength_bonus),
    armor_bonus_(armor_bonus) {}
  double attack() { return U::attack() + strength_bonus_; }
  double defense() { return U::defense() + armor_bonus_; }
  private:
  double strength_bonus_;
  double armor_bonus_;
};
```

现在，要将一个单位提升为老兵状态，我们必须将其转换为混凝土`unit`类的装饰版本：

```cpp
// Example 03
Knight k(10, 5);
Ogre o(12, 2);
k.hit(o); // Hit!
VeteranUnit<Knight> vk(std::move(k), 7, 2);
VeteranUnit<Ogre> vo(std::move(o), 1, 9);
vk.hit(vo); // Hit!
VeteranUnit<VeteranUnit<Ogre>> vvo(std::move(vo), 1, 9);
vk.hit(vvo); // Miss...
vk.charge(); // Compiles now, vk is a Knight too
vk.hit(vvo); // Hit with the charge bonus!
```

这是我们之前章节结尾处看到的相同场景，但现在它使用了模板装饰器。注意其中的差异。首先，`VeteranUnit`是一个从像`Knight`或`Ogre`这样的具体单位派生出来的类。因此，它能够访问基类的接口：例如，一个老兵骑士`VeteranUnit<Knight>`也是一个`Knight`，并且拥有从`Knight`继承来的成员函数`charge()`。其次，装饰过的单位明确地接管了原始单位的所有权——为了创建一个老兵单位，我们必须将原始单位移动到其中（老兵单位的基类是通过原始单位移动构造的）。原始对象被留在未指定的移动后状态，对这个对象唯一安全的操作是调用析构函数。请注意，至少对于单位类的简单实现，`move`操作只是一个复制，所以原始对象仍然是可用的，但你不应依赖于它——对移动后状态做出假设是一个即将发生的错误。

值得指出的是，我们对于`VeteranUnit`构造函数的声明强制并要求这种所有权的转移。如果我们试图在不从原始单位移动的情况下构建一个老兵单位，它将无法编译：

```cpp
VeteranUnit<Knight> vk(k, 7, 2); // Does not compile
```

通过只提供一个接受右值引用的构造函数，即`Unit&&`，我们要求调用者同意所有权的转移。

到目前为止，为了演示目的，我们已经在栈上作为局部变量创建了所有单位对象。在任何非平凡程序中，这都不会起作用——我们需要这些对象在创建它们的函数完成之后还能存在。我们可以集成装饰器对象和内存所有权机制，并确保在创建装饰版本之后删除移动后的原始单位。

假设在整个程序中通过唯一指针（在任何给定时间每个对象都有一个明确的拥有者）来管理所有权。以下是实现方法。首先，为我们需要使用的指针声明别名是方便的：

```cpp
using Unit_ptr = std::unique_ptr<Unit>;
using Knight_ptr = std::unique_ptr<Knight>;
```

虽然任何单位都可以被`Unit_ptr`指针拥有，但我们不能通过它调用特定的单位成员函数，例如`charge()`，因此我们可能还需要指向具体类的指针。正如我们接下来将要看到的，我们需要在这些指针之间移动对象。从派生类指针移动到基类指针是很容易的：

```cpp
Knight_ptr k(new Knight(10, 5));
Unit_ptr u(std::move(k)); // Now k is null
```

反方向的操作要困难一些；`std::move`不会隐式地工作，就像我们不能在没有显式转换的情况下从`Unit*`转换为`Knight*`一样。我们需要一个*移动转换*：

```cpp
// Example 04
template <typename To, typename From>
std::unique_ptr<To> move_cast(std::unique_ptr<From>& p) {
 return std::unique_ptr<To>(static_cast<To*>(p.release()));
}
```

在这里，我们使用`static_cast`将类型转换为派生类，如果假设的关系（即基对象确实是预期的派生对象）是正确的，那么这将是有效的，否则结果是不确定的。如果我们想，我们可以在运行时测试这个假设，使用`dynamic_cast`代替。这里是一个进行测试的版本，但仅当断言被启用时（我们可以抛出一个异常而不是断言）：

```cpp
// Example 04
template <typename To, typename From>
std::unique_ptr<To> move_cast(std::unique_ptr<From>& p) {
#ifndef NDEBUG
 auto p1 =
   std::unique_ptr<To>(dynamic_cast<To*>(p.release()));
 assert(p1);
 return p1;
#else
 return std::unique_ptr<To>(static_cast<To*>(p.release()));
#endif
}
```

如果所有对象都将由唯一指针的实例拥有，那么`VeteranUnit`装饰器必须在构造函数中接受一个指针并将对象从这个指针中移动出来：

```cpp
// Example 04
template <typename U> class VeteranUnit : public U {
  public:
  template <typename P>
  VeteranUnit(P&& p,
              double strength_bonus,
              double armor_bonus) :
    U(std::move(*move_cast<U>(p))),
    strength_bonus_(strength_bonus),
    armor_bonus_(armor_bonus) {}
  double attack() { return U::attack() + strength_bonus_; }
  double defense() { return U::defense() + armor_bonus_; }
  private:
  double strength_bonus_;
  double armor_bonus_;
};
```

这里棘手的部分在于`VeteranUnit<U>`的基类`U`的初始化——我们必须将单元从唯一指针移动到基类，然后将其移动到派生类的移动构造函数中（没有简单地将对象从一个唯一指针移动到另一个唯一指针的方法；我们需要将其包装到派生类中）。我们还要确保不泄漏任何内存。原始的唯一指针被释放，因此它的析构函数将不会做任何事情，但我们的`move_cast`返回一个新的唯一指针，现在它拥有相同的对象。这个唯一指针是一个临时变量，将在新对象初始化结束时被删除，但在我们使用它的对象来构造一个新的派生对象（即`VeteranUnit`）之前不会删除。单元对象的移动初始化本身在我们的情况下与复制相比没有节省任何时间，但这是一个良好的实践，以防更重的单元对象提供了一个优化的移动构造函数。

下面是如何在管理资源（在我们的例子中是单元）的程序中使用这个新装饰器的示例：

```cpp
// Example 04
Knight_ptr k(new Knight(10, 5));
  // Knight_ptr so we can call charge()
Unit_ptr o(new Ogre(12, 2));
  // Could be Orge_ptr if we needed one
Knight_ptr vk(new VeteranUnit<Knight>(k, 7, 2));
Unit_ptr vo(new VeteranUnit<Ogre>(o, 1, 9));
Unit_ptr vvo(new VeteranUnit<VeteranUnit<Ogre>>(vo, 1, 9));
vk->hit(*vvo); // Miss
vk->charge(); // Works because vk is Knight_ptr
vk->hit(*vvo); // Hit
```

注意，我们没有重新定义`hit()`函数——它仍然通过引用接受一个单元对象。这是正确的，因为这个函数不拥有对象——它只是对其操作。没有必要向其中传递拥有指针——这会暗示所有权的转移。

注意，严格来说，这个例子和上一个例子之间几乎没有区别——被移动的单元无论如何都不应该被访问。从实际的角度来看，存在显著的差异——被移动的指针不再拥有对象。它的值是空值，因此，在对象被提升之后尝试对其原始单元进行操作将很快变得明显（程序将解引用空指针并崩溃）。

正如我们所见，我们可以装饰已经装饰过的类，因为装饰器的作用是累积的。同样，我们可以将两个不同的装饰器应用于同一个类。每个装饰器都为该类添加了特定的新行为。在我们的游戏引擎中，我们可以打印每次攻击的结果，无论是否命中。但如果结果不符合预期，我们不知道原因。为了调试，打印攻击和防御值可能很有用。我们不想对所有单位都这样做，但对我们感兴趣的代码部分，我们可以使用一个调试装饰器，该装饰器为单位添加新行为以打印计算的中间结果。

`DebugDecorator`使用与之前装饰器相同的设计理念——它是一个类模板，生成一个从要装饰的对象派生的类。它的`attack()`和`defense()`虚拟函数将调用转发到基类并打印结果：

```cpp
// Example 05
template <typename U> class DebugDecorator : public U {
  public:
  using U::U;
  template <typename P> DebugDecorator(P&& p) :
    U(std::move(*move_cast<U>(p))) {}
  double attack() {
    double res = U::attack();
    cout << "Attack: " << res << endl;
    return res;
  }
  double defense() {
    double res = U::defense();
    cout << "Defense: " << res << endl;
    return res;
  }
};
```

在这个例子中，我们省略了动态内存分配，并依赖于移动对象本身来实现所有权的转移。我们没有理由不能同时拥有可堆叠的装饰器和唯一指针：

```cpp
// Example 06
template <typename U> class VeteranUnit : public U {
  ...
};
template <typename U> class DebugDecorator : public U {
  using U::U;
  public:
  template <typename P>
  DebugDecorator(std::unique_ptr<P>& p) :
    U(std::move(*move_cast<U>(p))) {}
  double attack() override {
    double res = U::attack();
    cout << "Attack: " << res << endl;
    return res;
  }
  double defense() override {
    double res = U::defense();
    cout << "Defense: " << res << endl;
    return res;
  }
  using ptr = std::unique_ptr<DebugDecorator>;
  template <typename... Args>
  static ptr construct(Args&&... args) { return
    ptr{new DebugDecorator(std::forward<Args>(args)...)};
  }
};
```

在实现装饰器时，你应该小心不要意外地以意想不到的方式改变基类的行为。例如，考虑`DebugDecorator`的这种可能实现：

```cpp
template <typename U> class DebugDecorator : public U {
  double attack() {
    cout << "Attack: " << U::attack() << endl;
    return U::attack();
  }
};
```

这里有一个微妙的错误——装饰后的对象，除了预期的打印输出之外，还隐藏了对原始行为的一个变化——它在基类上调用`attack()`两次。如果两次调用`attack()`返回不同的值，打印的值可能是不正确的，而且任何一次性攻击加成，如骑士冲锋，也将被取消。

`DebugDecorator`为它装饰的每个成员函数添加了非常相似的行为。C++有一套丰富的工具，旨在专门提高代码重用和减少重复。让我们看看我们是否能做得更好，并提出一个更可重用、更通用的装饰器。

## 多态装饰器和它们的局限性

一些装饰器非常特定于它们所修改的类，其行为是精确针对的。而另一些则非常通用，至少在原则上如此。例如，一个记录函数调用并打印返回值的调试装饰器，如果能够正确实现，可以与任何函数一起使用。

使用 C++14 或更高版本的`variadic`模板、参数包和完美前向，这样的实现相当直接：

```cpp
// Example 07
template <typename Callable> class DebugDecorator {
  public:
  template <typename F>
  DebugDecorator(F&& f, const char* s) :
    c_(std::forward<F>(f)), s_(s) {}
  template <typename ... Args>
  auto operator()(Args&& ... args) const {
    cout << "Invoking " << s_ << endl;
    auto res = c_(std::forward<Args>(args) ...);
    cout << "Result: " << res << endl;
    return res;
  }
  private:
  Callable c_;
  const std::string s_;
};
```

这个装饰器可以围绕任何可调用对象或函数（任何可以用`()`语法调用的东西）包装，无论有多少个参数。它打印自定义字符串和调用结果。然而，写出可调用类型通常是棘手的——最好让编译器为我们完成这项工作，使用模板参数推导：

```cpp
// Example 07
template <typename Callable>
  auto decorate_debug(Callable&& c, const char* s) {
  return DebugDecorator<Callable>(
    std::forward<Callable>(c), s);
}
```

这个模板函数推断出`Callable`的类型，并用调试包装器对其进行装饰。现在我们可以将其应用于任何函数或对象。下面是一个装饰后的函数示例：

```cpp
// Example 07
int g(int i, int j) { return i - j; } // Some function
auto g1 = decorate_debug(g, "g()"); // Decorated function
g1(5, 2); // Prints "Invoking g()" and "Result: 3"
```

我们还可以装饰一个可调用对象：

```cpp
// Example 07
struct S {
  double operator()() const {
    return double(rand() + 1)/double(rand() + 1);
  }
};
S s; // Callable
auto s1 =
  decorate_debug(s, "rand/rand"); // Decorated callable
s1(); s1(); // Prints the result, twice
```

注意，我们的装饰器不会接管可调用对象的所有权（如果我们想这样做，可以写成那样）。

我们甚至可以装饰一个 lambda 表达式，这实际上只是一个隐式类型的可调用对象。这个例子中的 lambda 定义了一个具有两个整数参数的可调用对象：

```cpp
// Example 07
auto f2 = decorate_debug(
  [](int i, int j) { return i + j; }, "i+j");
f2(5, 3); // Prints "Invoking i+j" and "Result: 8"
```

在我们的例子中，我们决定在装饰器类构造函数和辅助函数中都将可调用对象进行转发。通常，可调用对象是通过值传递的，并且假设它们易于复制。在所有情况下，装饰器存储其数据成员中的可调用对象的副本都很重要。如果你通过引用捕获它，那么一个微妙的错误正在等待发生：

```cpp
template <typename Callable> class DebugDecorator {
  public:
  DebugDecorator(const Callable& c, const char* s) :
    c_(c), s_(s) {}
  ...
  private:
  const Callable& c_;
  const std::string s_;
};
```

装饰一个函数很可能会正常工作，但装饰一个 lambda 表达式可能会失败（尽管不一定立即显现）。`const Callable& c_` 成员将被绑定到一个临时的 lambda 对象上：

```cpp
auto f2 = decorate_debug(
  [](int i, int j) { return i + j; }, "i+j");
```

这个对象的生存期在语句末尾的分号处结束，任何随后的`f2`使用都会访问一个悬垂引用（地址检查工具可以帮助检测此类错误）。

我们的装饰器有一些局限性。首先，当我们尝试装饰一个不返回任何内容的功能时，它就不够用了，比如下面的 lambda 表达式，它增加其参数但不返回任何内容：

```cpp
auto incr = decorate_debug([](int& x) { ++x; }, "++x");
int i;
incr(i); // Does not compile
```

问题出在`DebugDecorator`内部`auto res = ...`行的`void res`表达式。这是有道理的，因为我们不能声明`void`类型的变量。这个问题可以使用 C++17 中的`if constexpr`来解决：

```cpp
// Example 08
template <typename Callable> class DebugDecorator {
  public:
  ...
  template <typename... Args>
  auto operator()(Args&&... args) const {
    cout << "Invoking " << s_ << endl;
    using r_t = decltype(c_(std::forward<Args>(args)...));
    if constexpr (!std::is_same_v<res_t, void>) {
      auto res = c_(std::forward<Args>(args)...);
      cout << "Result: " << res << endl;
      return res;
    } else {
      c_(std::forward<Args>(args)...);
    }
  }
    private:
    Callable c_;
    const std::string s_;
};
```

在 C++17 之前，if constexpr 最常用的替代方法是函数重载（第一个参数是`std::true_type`或`std::false_type`，这取决于由相应函数提供的 if constexpr 的分支）：

```cpp
// Example 08a
template <typename Callable> class DebugDecorator {
  public:
  ...
  template <typename... Args>
  auto operator()(Args&&... args) const {
    cout << "Invoking " << s_ << endl;
    using r_t = decltype(c_(std::forward<Args>(args)...));
    return this->call_impl(std::is_same<res_t, void>{},
                           std::forward<Args>(args)...);
    }
    private:
    Callable c_;
    const std::string s_;
    template <typename... Args>
    auto call_impl(std::false_type, Args&&... args) const {
      auto res = c_(std::forward<Args>(args)...);
      cout << "Result: " << res << endl;
      return res;
    }
    template <typename... Args>
    void call_impl(std::true_type, Args&&... args) const {
      c_(std::forward<Args>(args)...);
  }
};
```

第二个局限性是，我们的装饰器的`auto`返回类型推断得并不完全准确——例如，如果一个函数返回`double&`，则装饰后的函数将只返回`double`。最后，包装成员函数调用是可能的，但需要稍微不同的语法。

现在，C++的模板机制非常强大，有方法可以使我们的泛型装饰器更加通用。这些方法也使其更加复杂。这样的代码应该放在库中，比如标准库，但在大多数实际应用中，调试装饰器不值得付出这样的努力。

另一个限制是，装饰器越通用，它能做的就越少。就目前而言，我们能够采取的、对任何函数或成员函数（甚至在我们的装饰器中产生一个良好的调试信息可能需要使用编译器扩展，见*示例 09*）有意义的操作非常有限。我们可以添加一些调试打印，只要它定义了流输出操作符，就可以打印结果。我们可以在多线程程序中锁定互斥锁来保护非线程安全的函数调用。可能还有一些更通用的操作。但总的来说，不要被追求最通用代码本身所迷惑。

无论我们是否有某种通用的还是非常具体的装饰器，我们通常都需要向对象添加多个行为。我们已经看到了一个这样的例子。现在，让我们更系统地回顾一下应用多个装饰器的问题。

## 可组合的装饰器

我们希望在这里拥有的装饰器属性有一个名字，叫做可组合性。如果行为可以被分别应用于同一个对象，则它们是可组合的：在我们的情况下，如果我们有两个装饰器，`A`和`B`。因此，`A(B(object))`应该应用两种行为。可组合性的替代方案是显式创建组合行为：如果没有可组合性，我们需要编写一个新的装饰器，`AB`。由于为几个装饰器的任何组合编写新代码都是不可能的，即使装饰器的数量相对较少，可组合性是一个非常重要的属性。

幸运的是，使用我们的装饰器方法，可组合性并不难实现。我们在早期的游戏设计中使用的 CRTP 装饰器自然是可以组合的：

```cpp
template <typename U> class VeteranUnit : public U { ... };
template <typename U> class DebugDecorator : public U { ...
};
Unit_ptr o(new DebugDecorator<Ogre>(12, 2));
Unit_ptr vo(new DebugDecorator<VeteranUnit<Ogre>>(o, 1, 9));
```

每个装饰器都继承自它装饰的对象，因此保留了其接口，除了添加的行为。请注意，装饰器的顺序很重要，因为新的行为是在装饰调用之前或之后添加的。`DebugDecorator`应用于它装饰的对象，并为其提供调试功能，因此`VeteranUnit<DebugDecorator<Ogre>>`对象会调试对象的基础部分（`Ogre`），这同样很有用。

我们的（某种程度上）通用装饰器也可以组合使用。我们已经有了一个可以与许多不同的可调用对象一起工作的调试装饰器，并且我们提到了可能需要使用互斥锁来保护这些调用。现在我们可以以类似的方式（以及类似的限制）实现这样的锁定装饰器，就像多态调试装饰器一样：

```cpp
// Example 10
template <typename Callable> class LockDecorator {
  public:
  template <typename F>
  LockDecorator(F&& f, std::mutex& m) :
    c_(std::forward<F>(f)), m_(m) {}
  template <typename ... Args>
  auto operator()(Args&& ... args) const {
    std::lock_guard<std::mutex> l(m_);
    return c_(std::forward<Args>(args) ...);
  }
  private:
  Callable c_;
  std::mutex& m_;
};
template <typename Callable>
auto decorate_lock(Callable&& c, std::mutex& m) {
  return
    LockDecorator<Callable>(std::forward<Callable>(c), m);
}
```

再次强调，我们将使用`decorate_lock()`辅助函数来委托给编译器解决可调用对象的正确类型这一繁琐的工作。现在我们可以使用互斥锁来保护一个非线程安全的函数调用：

```cpp
std::mutex m;
auto safe_f = decorate_lock([](int x) {
  return unsafe_f(x); }, m
);
```

如果我们想要通过互斥锁来保护一个函数，并在调用时进行调试打印，我们不需要编写一个新的*锁定调试装饰器*，而是可以按顺序应用这两个装饰器：

```cpp
auto safe_f = decorate_debug(
  decorate_lock(
    [](int x) { return unsafe_f(x); },
    m
  ),
  "f(x)");
```

这个例子展示了可组合性的好处——我们不需要为每种行为组合编写特殊的装饰器（想想如果它们不可组合，你将需要为五种不同主要装饰器的任何组合编写多少装饰器！）。

这种可组合性在我们的装饰器中很容易实现，因为它们保留了原始对象（至少是我们感兴趣的部分）的接口——行为改变了，但接口没有改变。当一个装饰器被用作另一个装饰器的原始对象时，保留的接口再次被保留，依此类推。

保留接口是装饰器模式的一个基本特征。它也是其最严重的限制之一。我们的锁定装饰器并不像乍看之下那么有用（所以当你需要使代码线程安全时，不要在代码中到处强行添加锁）。正如我们接下来将看到的，无论实现多么好，并不是每个接口都可以被做成线程安全的。这就是我们不仅要修改行为，还要改变接口的时候。这是适配器模式的工作。

# 适配器模式

我们在上一个部分结束时提出了装饰器模式具有来自保留装饰后接口的特定优势，并且这些优势有时会变成限制。适配器模式是一个更通用的模式，可以在这种情况下使用。

适配器模式被定义得非常广泛——它是一种结构型模式，允许一个类的接口被用作另一个不同的接口。它允许一个现有的类在期望不同接口的代码中使用，而不需要修改原始类。这样的适配器有时被称为**类包装器**，因为它们**包装**在类周围并呈现不同的接口。你可能还记得，装饰器有时也被称为**类包装器**，原因大致相同。

然而，适配器模式是一个非常通用、广泛的模式。它可以用来实现几个其他更具体定义的模式——特别是装饰器模式。装饰器模式更容易理解，所以我们首先处理了它。现在，我们将转向一般情况。

## 基本适配器模式

让我们继续上一个部分的最后一个例子——锁定装饰器。它在一个锁下调用任何函数，因此没有其他由相同互斥锁保护的函数可以在任何其他线程上同时被调用。在某些情况下，这足以使整个代码线程安全。通常，这并不够。

为了演示这一点，我们将实现一个线程安全的队列对象。队列是一个中等复杂的数据结构，即使没有线程安全，但幸运的是，我们不需要从头开始——C++标准库中有`std::queue`。我们可以按照先进先出的顺序将对象推入队列并从队列中取出对象，但只能在一个线程上——例如，同时从两个不同的线程向同一个队列推送两个对象是不安全的。但我们有一个解决方案——我们可以将锁定队列作为基本队列的装饰器来实现。由于我们这里不关心空基优化（`std::queue`不是一个空类）并且我们必须转发每个成员函数调用，所以我们不需要继承，而可以使用组合。我们的装饰器将包含队列和锁。包装`push()`方法很容易。`std::queue`中有两个版本的`push()`方法——一个移动对象，一个复制对象。我们应该用锁保护这两个版本：

```cpp
// Example 11
template <typename T> class locking_queue {
  using mutex = std::mutex;
  using lock_guard = std::lock_guard<mutex>;
  using value_type = typename std::queue<T>::value_type;
  void push(const value_type& value) {
    lock_guard l(m_);
    q_.push(value);
  }
  void push(value_type&& value) {
    lock_guard l(m_);
    q_.push(value);
  }
  private:
  std::queue<T> q_;
  mutex m_;
};
```

现在，让我们把注意力转向从队列中获取元素。标准队列有三个相关的成员函数——首先，是`front()`，它允许我们访问队列的前端元素，但不从队列中移除它。然后是`pop()`，它移除前端元素但不返回任何内容（它不提供对前端元素的访问——它只是移除它）。这两个函数在队列为空时不应该被调用——没有错误检查，但结果是不确定的。

最后，是第三个函数`empty()`；如果队列不为空，它返回`false`，然后我们可以调用`front()`和`pop()`。如果我们用锁定来装饰它们，我们就能写出如下代码：

```cpp
locking_queue<int> q;
q.push(5);
... sometime later in the program ...
if (!q.empty()) {
  int i = q.front();
  q.pop();
}
```

每个函数本身都是线程安全的。但它们的整体组合并不是。理解这一点非常重要。首先，我们调用`q.empty()`。假设它返回`false`，这意味着我们知道队列中至少有一个元素。接下来，我们在下一行通过调用`q.front()`来访问它，它返回`5`。但这只是程序中许多线程中的一个。另一个线程同时正在执行相同的代码（这正是练习的目的）。那个线程也调用了`q.empty()`，并且也得到了`false`——正如我们刚才说的，队列中有一个元素，而我们还没有做任何删除操作。第二个线程也调用了`q.front()`，并且也得到了`5`。这已经是一个问题——两个线程都试图从队列中取出一个元素，但取到的却是同一个。但问题更严重——我们的第一个线程现在调用了`q.pop()`并从队列中移除了`5`。现在队列是空的，但第二个线程并不知道这一点——它之前调用了`q.empty()`。因此，第二个线程现在也调用了`q.pop()`，这次是在一个空队列上。在这种情况下，最好的情况是程序会立即崩溃。

我们刚刚看到了一个一般问题的具体案例——一系列操作，每个操作都是线程安全的，但作为一个整体不是线程安全的。实际上，这个*锁定队列*完全无用，无法用它来编写线程安全的代码。我们需要的是一个单一的线程安全函数，它在一个锁下执行整个事务，作为一个不可中断的操作（这种事务被称为`std::queue`接口不提供这样的事务性 API）。

因此，现在我们需要一个新的模式——一个将类的现有接口转换为不同接口所需的新模式。这不能通过装饰器模式来完成，但这正是适配器模式解决的问题。既然我们已经同意需要一个不同的接口，我们只需决定它应该是什么。我们新的单个`pop()`成员函数应该完成所有这些——如果队列不为空，它应该从队列中移除第一个元素并返回它，通过复制或移动，给调用者。如果队列为空，它应该根本不改变队列的状态，但以某种方式通知调用者队列是空的。一种方法是通过返回两个值来实现——元素本身（如果有的话）和一个布尔值，告诉我们队列是否为空。以下是锁定队列的`pop()`部分，现在它是一个适配器，而不是装饰器：

```cpp
// Example 11
template <typename T> class locking_queue {
  ... the push() is unchanged ...
  bool pop(value_type& value) {
    lock_guard l(m_);
    if (q_.empty()) return false;
    value = std::move(q_.front());
    q_.pop();
    return true;
  }
  private:
  std::queue<T> q_;
  mutex m_;
};
```

注意，我们不需要改变`push()`——单个函数调用已经完成了我们需要的所有操作，因此这部分接口只是通过我们的适配器一对一地转发。这个版本的`pop()`在从队列中移除元素时返回`true`，否则返回`false`。如果返回`true`，则将元素保存到提供的参数中，但如果返回`false`，则参数保持不变。如果元素类型`T`是可移动赋值的，则使用移动而不是复制。

当然，这并不是这种原子`pop()`的唯一可能接口。另一种方式是返回一个包含元素和布尔值的对。一个显著的区别是，现在没有方法可以保留元素不变——它是返回值，它总是必须有一些东西。自然的方式是，如果队列中没有元素，就默认构造该元素（这暗示了对元素类型`T`的限制——它必须是可以默认构造的）。

在 C++17 中，更好的选择是返回一个`std::optional`：

```cpp
// Example 12
template <typename T> class locking_queue {
  ... the push() is unchanged ...
  std::optional<value_type> pop() {
    lock_guard l(m_);
    if (q_.empty()) return std::nullopt;
    value_type value = std::move(q_.front());
    q_.pop();
    return { value };
  }
};
```

根据需要此队列的应用代码，可能有一种接口更可取，因此也有其他设计它的方法。在所有情况下，我们最终都会有两个成员函数，`push()`和`pop()`，它们都受到相同的互斥锁的保护。现在，任何数量的线程可以同时执行这些操作的任何组合，并且行为是明确的。这意味着`locking_queue`对象是线程安全的。

将对象从其当前接口转换为特定应用所需的接口，而不需要重写对象本身，这就是适配器模式的目的和用途。可能需要转换各种接口，因此存在许多不同类型的适配器。我们将在下一节中了解其中的一些。 

## 函数适配器

我们刚刚看到了一个类适配器，它改变了类的接口。另一种接口是函数（成员函数或非成员函数）。一个函数有一定的参数，但我们可能想要用不同的参数集调用它。这需要一个适配器。这种适配器的一个常见应用被称为 currying（或 currying 多个）函数的参数。这意味着我们有一个多个参数的函数，我们固定其中一个参数的值，因此我们不必在每次调用时指定它。一个例子是，如果我们有 `f(int i, int j)`，但我们需要 `g(i)`，这相当于 `f(i, 5)`，只是不需要每次都输入 `5`。

这里有一个更有趣的例子，我们将实际逐步实现一个适配器。`std::sort` 函数接受一个迭代器范围（要排序的序列），但它也可以用三个参数调用——第三个是比较对象（默认情况下使用 `std::less`，它反过来会在排序的对象上调用 `operator<()`）。

现在，我们想要做的是其他事情——我们想要模糊地比较浮点数，带有容差——如果两个数 `x` 和 `y` 足够接近，那么我们不认为一个比另一个小。只有当 `x` 远远小于 `y` 时，我们才想要强制执行排序顺序，即 `x` 在 `y` 之前。

下面是我们的比较函数对象（一个可调用对象）：

```cpp
// Example 13
struct much_less {
  template <typename T>
  bool operator()(T x, T y) {
    return x < y && std::abs(x - y) > tolerance);
  }
  static constexpr double tolerance = 0.2;
};
```

这个比较对象可以与标准排序一起使用：

```cpp
std::vector<double> v;
std::sort(v.begin(), v.end(), much_less());
```

然而，如果我们经常需要这种排序，我们可能想要 curry 最后一个参数，并为自己创建一个只有两个参数的适配器，即迭代器和隐含的排序函数。下面是一个这样的适配器——它非常简单：

```cpp
// Example 13
template<typename RandomIt>
  void sort_much_less(RandomIt first, RandomIt last) {
  std::sort(first, last, much_less());
}
```

现在，我们可以用两个参数调用排序函数：

```cpp
// Example 13
std::vector<double> v;
sort_much_less(v.begin(), v.end());
```

现在，如果我们经常以这种方式调用排序来对整个容器进行排序，我们可能想要再次更改接口并创建另一个适配器：

```cpp
// Example 14
template<typename Container> void sort_much_less(Container&
   c) {
std::sort(c.begin(), c.end(), much_less());
}
```

在 C++20 中，`std::sort` 和其他 STL 函数有接受范围的变体；它们是容器适配器的一般化。现在，我们程序中的代码看起来更加简单：

```cpp
// Example 14
std::vector<double> v;
sort_much_less(v);
```

需要指出的是，C++14 提供了编写这种简单适配器的替代方案，通常应优先选择；我们可以使用 lambda 表达式，如下所示：

```cpp
// Example 15
auto sort_much_less = [](auto first, auto last) {
  return std::sort(first, last, much_less());
};
```

当然，比较函数 much_less() 本身就是一个可调用对象，因此它也可以是一个 lambda 表达式：

```cpp
// Example 15a
auto sort_much_less = [](auto first, auto last) {
  return std::sort(first, last,
    [](auto x, auto y) {
      static constexpr double tolerance = 0.2;
      return x < y && std::abs(x - y) > tolerance;
    }); };
```

容器适配器的编写同样简单：

```cpp
// Example 16
auto sort_much_less = [](auto& container) {
  return std::sort(container.begin(), container.end(),
                   much_less());
};
```

注意，你无法在同一个程序中以相同的名称同时拥有这两个（lambda 表达式不能以这种方式*重载*；实际上，它们根本不是函数，而是对象（你可以从 lambda 中创建一个重载集，如*第二章**，类和函数模板*所示）。

回到用一些固定或绑定的常量值调用函数的问题，我们应该说这是一个如此常见的需求，以至于 C++标准库提供了一个用于此目的的标准可定制适配器，即`std::bind`。以下是一个示例，展示了它的用法：

```cpp
// Example 17
using namespace std::placeholders; // For _1, _2 etc
int f3(int i, int j, int k) { return i + j + k; }
auto f2 = std::bind(f3, _1, _2, 42);
auto f1 = std::bind(f3, 5, _1, 7);
f2(2, 6);     // Returns 50
f1(3);     // Returns 15
```

这种标准适配器有其自己的*迷你语言* - `std::bind`的第一个参数是要绑定的函数，其余的是它的参数，按顺序排列。应该绑定的参数被指定的值替换。应该保持自由的参数被占位符`_1`、`_2`等替换（不一定按此顺序；也就是说，我们也可以改变参数的顺序）。返回值是不指定的类型，必须使用`auto`捕获。我们唯一知道的是返回值可以像函数一样调用，具有与占位符一样多的参数。它也可以在任何期望可调用的上下文中用作函数，例如，在另一个`std::bind`中：

```cpp
// Example 17
...
auto f1 = std::bind(f3, 5, _1, 7);
auto f0 = std::bind(f1, 3);
f1(3);    // Returns 15
f0();         // Also returns 15
```

然而，这些对象是可调用的，而不是函数，你会发现如果你尝试将它们中的一个分配给函数指针：

```cpp
// Example 17
int (*p3)(int, int, int) = f3;    // OK
int (*p1)(int) = f1;            // Does not compile
```

相反，如果 lambda 没有捕获，则可以将其转换为函数指针：

```cpp
auto l1 = [](int i) { return f3(5, i, 7); }
int (*p1)(int) = l1;            // OK
```

尽管`std::bind`很有用，但它并没有使我们摆脱学习如何编写自己的函数适配器的需求 - 它最大的局限性是`std::bind`不能绑定模板函数。我们无法编写以下内容：

```cpp
auto sort_much_less = std::bind(std::sort, _1, _2, much_less()); // No!
```

这将无法编译。在模板内部，我们可以绑定它的特定实例化，但至少在我们的排序示例中，这实际上并没有给我们带来任何好处：

```cpp
template<typename RandomIt>
void sort_much_less(RandomIt first, RandomIt last) {
  auto f = std::bind(std::sort<RandomIt, much_less>,
                     _1, _2, much_less());
  f(first, last, much_less());
}
```

如我们在本节开头提到的，装饰器可以被视为适配器模式的一个特例。有时，这种区别并不在于模式的具体应用，而在于我们选择如何看待它。

## 适配器或装饰器

到目前为止，我们描述装饰器为一种我们用来增强现有接口的模式，而适配器则是用来转换（适配）接口，以便与期望不同接口的代码集成。这种区别并不总是清晰的。

例如，让我们考虑一个简单的类，它将系统调用`std::time`的结果适配为可打印的日期格式（`std::chrono`提供了这种功能，但它是一个易于理解的例子）。函数`std::time`返回一个`std::time_t`类型的值，它是一个整数，包含自过去某个标准时刻以来的秒数，这个时刻被称为“纪元开始”。另一个系统函数`localtime`将这个值转换为包含日期元素的 struct：年、月和日（以及小时、分钟等）。通常的日历计算相当复杂（这也是为什么`std::chrono`不像我们希望的那样简单），但就目前而言，让我们假设系统库做了正确的事情，我们只需要以正确的格式打印日期。例如，以下是如何以美国格式打印当前日期的方法：

```cpp
const std::time_t now = std::time(nullptr);
const tm local_tm = *localtime(&now);
cout << local_tm.tm_mon + 1 << "/" <<
        local_tm.tm_mday << "/" <<
        local_tm.tm_year + 1900;
```

我们想要创建一个适配器，将秒数转换为特定格式的日期，并允许我们打印它；我们需要为美国格式（月份在前）、欧洲格式（日期在前）和 ISO 格式（年份在前）分别创建适配器。

适配器的实现相当直接：

```cpp
// Example 18
class USA_Date {
  public:
  explicit USA_Date(std::time_t t) : t_(t) {}
  friend std::ostream& operator<<(std::ostream& out,
                                  const USA_Date& d) {
    const tm local_tm = *localtime(&d.t_);
    out << local_tm.tm_mon + 1 << "/" <<
           local_tm.tm_mday << "/" <<
           local_tm.tm_year + 1900;
    return out;
  }
  private:
  const std::time_t t_;
};
```

其他两种日期格式在打印字段顺序上相似，除了我们打印字段的顺序。事实上，它们如此相似，我们可能想要重构代码以避免编写三个几乎相同的类。最简单的方法是使用模板，并将字段顺序编码在“格式代码”中，该代码指定我们打印日期（字段 0）、月份（字段 1）和年份（字段 2）的顺序。例如，“格式”210 表示年份，然后是月份，然后是日期——ISO 日期格式。格式代码可以是一个整数模板参数：

```cpp
// Example 19
template <size_t F> class Date {
  public:
  explicit Date(std::time_t t) : t_(t) {}
  friend std::ostream& operator<<(std::ostream& out,
                                  const Date& d) {
    const tm local_tm = *localtime(&d.t_);
    const int t[3] = { local_tm.tm_mday,
                       local_tm.tm_mon + 1,
                       local_tm.tm_year + 1900 };
    constexpr size_t i1 = F/100;
    constexpr size_t i2 = (F - i1*100)/10;
    constexpr size_t i3 = F - i1*100 - i2*10;
    static_assert(i1 >= 0 && i1 <= 2 && ..., "Bad format");
    out << t[i1] << "/" << t[i2] << "/" << t[i3];
    return out;
  }
  private:
  const std::time_t t_;
};
using USA_Date = Date<102>;
using European_Date = Date<12>;
using ISO_Date = Date<210>;
```

我们的小包装器将一个类型（一个整数）适配为在代码中使用，该代码期望以特定格式提供日期。或者它是否用`operator<<()`装饰了整数？最好的答案是……无论哪个对你思考特定问题更有帮助。重要的是要记住，最初用模式语言说话的目的：我们这样做是为了有一个紧凑且普遍理解的方式来描述我们的软件问题和我们选择的解决方案。当多个模式似乎产生类似的结果时，你选择描述让你能够关注对你最重要的方面。

到目前为止，我们只考虑了转换运行时接口的适配器，这些接口是我们程序执行时调用的接口。然而，C++也有编译时接口——我们在上一章考虑的一个主要例子是基于策略的设计。这些接口并不总是恰好符合我们的需求，因此我们必须学会编写编译时适配器。

## 编译时适配器

在*第十六章*，“适配器和装饰器”中，我们学习了政策，它们是类的构建块——它们让程序员可以为特定的行为定制实现。作为一个例子，我们可以实现这个基于政策的智能指针，它可以自动删除它拥有的对象。政策是删除的特定实现：

```cpp
// Chapter 15, Example 08
template <typename T,
          template <typename> class DeletionPolicy =
                                    DeleteByOperator>
class SmartPtr {
  public:
  explicit SmartPtr(T* p = nullptr,
    const DeletionPolicy<T>& del_policy =
                             DeletionPolicy<T>())
  : p_(p), deletion_policy_(deletion_policy)
  {}
  ~SmartPtr() {
    deletion_policy_(p_);
  }
  ... pointer interface ...
  private:
  T* p_;
  DeletionPolicy<T> deletion_policy_;
};
```

注意，删除策略本身也是一个模板——这是一个*模板*参数。默认的删除策略是使用`operator delete`：

```cpp
template <typename T> struct DeleteByOperator {
  void operator()(T* p) const {
    delete p;
  }
};
```

然而，对于在用户指定的堆上分配的对象，我们需要一个不同的删除策略，该策略将内存返回到那个堆：

```cpp
template <typename T> struct DeleteHeap {
  explicit DeleteHeap(MyHeap& heap) : heap_(heap) {}
  void operator()(T* p) const {
    p->~T();
    heap_.deallocate(p);
  }
  private:
  MyHeap& heap_;
};
```

然后，我们必须创建一个政策对象来与指针一起使用：

```cpp
MyHeap H;
SmartPtr<int, DeleteHeap<int>> p(new int, H);
```

这项政策并不非常灵活，然而——它只能处理一种类型的堆——`MyHeap`。如果我们把堆类型设为第二个模板参数，就可以使政策更通用。只要堆有`deallocate()`成员函数来返回内存给它，我们就可以使用任何与这个政策兼容的堆类：

```cpp
// Example 20
template <typename T, typename Heap> struct DeleteHeap {
  explicit DeleteHeap(Heap& heap) : heap_(heap) {}
  void operator()(T* p) const {
    p->~T();
    heap_.deallocate(p);
  }
  private:
  Heap& heap_;
};
```

当然，如果我们有一个使用不同名称的成员函数的堆类，我们可以使用类适配器使该类也能与我们的政策一起工作。但我们有一个更大的问题——我们的政策与我们的智能指针不兼容。以下代码无法编译：

```cpp
SmartPtr<int, DeletelHeap> p; // Does not compile
```

原因再次是接口不匹配，但现在它是一种不同类型的接口——`template <typename T, template <typename> class DeletionPolicy> class SmartPtr {};` 模板期望第二个参数是一个只有一个类型参数的模板。相反，我们有`DeleteHeap`模板，它有两个类型参数。这就像尝试调用一个只有一个参数但使用两个参数的函数——这是行不通的。我们需要一个适配器来将我们的双参数模板转换为单参数模板，并且我们必须将第二个参数固定为特定的堆类型（如果我们有多个堆类型，我们不需要重写策略，我们只需要编写几个适配器）。我们可以使用继承来创建这个适配器，`DeleteMyHeap`（并记得将基类的构造函数带入派生适配器类的范围）：

```cpp
// Example 20
template <typename T>
struct DeleteMyHeap : public DeleteHeap<T, MyHeap> {
  using DeleteHeap<T, MyHeap>::DeleteHeap;
};
```

我们也可以用模板别名来做同样的事情：

```cpp
// Example 21
template <typename T>
using DeleteMyHeap = DeleteHeap<T, MyHeap>;
```

这个第一个版本显然要长得多。然而，我们必须学习两种编写模板适配器的方法，因为模板别名有一个主要限制。为了说明这一点，让我们考虑另一个需要适配器的例子。我们将从实现任何 STL 兼容序列容器的流插入操作符开始，这些容器的元素定义了这样的操作符。它是一个简单的函数模板：

```cpp
// Example 22
template <template <typename> class Container, typename T>
std::ostream& operator<<(std::ostream& out,
                         const Container<T>& c) {
  bool first = true;
  for (auto x : c) {
  if (!first) out << ", ";
    first = false;
    out << x;
  }
  return out;
}
```

这个`template`函数有两个类型参数，容器类型和元素类型。容器本身是一个带有单个类型参数的模板。编译器从第二个函数参数（在任何`operator<<()`中的第一个参数总是流）推导出容器类型和元素类型。我们可以在一个简单的容器上测试我们的插入操作符：

```cpp
// Example 22
template <typename T> class Buffer {
  public:
  explicit Buffer(size_t N) : N_(N), buffer_(new T[N_]) {}
  ~Buffer() { delete [] buffer_; }
  T* begin() const { return buffer_; }
  T* end() const { return buffer_ + N_; }
  ...
  private:
  const size_t N_;
  T* const buffer_;
};
Buffer<int> buffer(10);
... fill the buffer ...
cout << buffer; // Prints all elements of the buffer
```

但这只是一个玩具容器，并不很有用。我们真正想要的是打印真实容器的元素，例如`std::vector`：

```cpp
std::vector<int> v;
... add some values to v ...
cout << v;
```

不幸的是，这段代码无法编译。原因是`std::vector`实际上不是一个只有一个类型参数的模板，尽管我们这样使用它。它有两个参数 - 第二个是分配器类型。这个分配器有一个默认值，这就是为什么我们可以写`std::vector<int>`并且它能编译。但是，即使有这个默认参数，这仍然是一个有两个参数的模板，而我们的流插入操作符被声明为只接受只有一个参数的容器模板。我们可以通过编写适配器来解决这个问题（大多数 STL 容器实际上都是与默认分配器一起使用的）。编写这个适配器最简单的方法是使用别名：

```cpp
template <typename T> using vector1 = std::vector<T>;
vector1<int> v;
...
cout << v; // Does not compile either
```

不幸的是，这同样无法编译，现在我们可以展示我们之前提到的模板别名限制 - 模板别名不用于模板参数类型推导。当编译器试图确定使用`cout`和`v`作为参数调用`operator<<()`的模板参数类型时，模板别名`vector1`是“不可见”的。在这种情况下，我们必须使用一个派生类适配器：

```cpp
// Example 22
template <typename T>
struct vector1 : public std::vector<T> {
  using std::vector<T>::vector;
};
vector1<int> v;
...
cout << v;
```

顺便说一下，如果你注意到了前面的章节，你可能已经意识到我们已经遇到了模板模板参数额外参数的问题，并且通过将这些参数声明为变长模板参数来解决它：

```cpp
// Example 23
template <typename T,
  template <typename, typename...> class Container,
  typename... Args>
std::ostream& operator<<(std::ostream& out,
                         const Container<T, Args...>& c) {
  ...
}
```

现在我们可以让`operator<<()`打印任何容器，所以我们不再需要担心适配器，对吧？并不完全是这样：我们仍然无法打印的容器之一是`std::array`，它是一个只有一个类型和一个非类型参数的类模板。我们可以声明一个重载来处理这种情况：

```cpp
// Example 23
template <typename T,
  template <typename, size_t> class Container, size_t N>
std::ostream& operator<<(std::ostream& out,
                         const Container<T, N>& c) {
  ...
}
```

但我们可能还有另一种类型的容器，它不适合这两种模板（无论是必须这样做还是因为它只是旧代码的一部分，该代码以不同的方式编写）。然后，我们再次必须使用适配器。

我们现在已经看到了如何实现装饰器来增强类和函数接口以实现所需的行为，以及当现有接口不适合特定应用时如何创建适配器。装饰器，甚至更不用说适配器，都是非常通用和灵活的模式，可以用来解决许多问题。毫不奇怪，通常一个问题可以用多种方式解决，因此有选择使用模式的余地。在下一节中，我们将看到这样一个案例。

# 适配器与策略

适配器和策略（或策略）模式是一些更通用的模式，C++ 为这些模式增加了泛型编程能力。这往往扩展了它们的可用性，有时也模糊了模式之间的界限。模式本身定义得非常明确 - 策略提供自定义实现，而适配器则改变接口并向现有接口添加功能（后者是装饰器方面，但正如我们所见，大多数装饰器都是作为适配器实现的）。我们还在上一章中看到，C++ 扩展了基于策略的设计能力；特别是，C++ 中的策略可以添加或删除接口的部分，以及控制实现。因此，虽然模式不同，但它们在可以用于的问题类型上存在显著的重叠。当问题在广义上可以适用于两种方法时，比较这两种方法是有教育意义的。对于这个练习，我们将考虑设计自定义值类型的问题。

简而言之，值类型是一种主要像 `int` 那样行为的类型。通常，这些类型是数字。虽然我们有一组内置类型用于这些，但我们可能想要操作有理数、复数、张量、矩阵或与它们关联有单位的数字（米、克等等）。这些值类型支持一系列操作，如算术运算、比较、赋值和复制。根据值所代表的内容，我们可能只需要这些操作的一小部分 - 例如，我们可能需要支持矩阵的加法和乘法，但不允许除法，并且在大多数情况下，比较矩阵以任何非等性可能都没有意义。同样，我们可能不希望允许米和克的相加。

更普遍地说，人们通常希望有一个具有有限接口的数值类型 - 如果我们不希望允许表示此类数值的操作编译，我们会希望这样。这样，一个包含无效操作的程序就根本无法编写。为了通用，我们的设计必须允许我们逐步构建接口。例如，我们可能希望一个可以用于等性比较、有序（定义了小于操作）和可加的值，但没有乘法或除法。这似乎是一个非常适合装饰器（或更普遍地，适配器）模式的问题：装饰器可以添加比较运算符或加法运算符等行为。另一方面，创建一个通过插入正确的策略来配置功能集的类型正是策略模式的目的。

## 适配器解决方案

让我们先来考察适配器解决方案。我们将从一个基本的价值类型开始，它在接口中几乎不支持任何功能，然后我们可以逐一添加所需的功能。

这里是我们的初始 `Value` 类模板：

```cpp
// Example 24
template <typename T> class Value {
  public:
  using basic_type = T;
  using value_type = Value;
  explicit Value() : val_(T()) {}
  explicit Value(T v) : val_(v) {}
  Value(const Value&) = default;
  Value& operator=(const Value&) = default;
  Value& operator=(basic_type rhs) {
    val_ = rhs;
    return *this;
  }
  protected:
  T val_ {};
};
```

`Value` 的值是可复制和可赋值的，无论是从底层类型如 `int` 还是另一个 `Value`。如果我们想拥有不可复制的值，我们也可以将这些功能移动到适配器中，但你会发现，在阅读完本章的其余部分后，这个改动很容易实现。

为了方便起见，我们还将使我们的 `Value` 可打印（在任何真实情况下，你可能会希望这是一个单独且可配置的功能，但这使得示例更简单，而没有去掉任何重要内容）。

```cpp
// Example 24
template <typename T> class Value {
  public:
  friend std::ostream& operator<<(std::ostream& out,
                                  Value x) {
    out << x.val_;
    return out;
  }
  friend std::istream& operator>>(std::istream& in,
                                  Value& x) {
    in >> x.val_;
    return in;
  }
  ...
};
```

我们使用 *友元工厂*，这在 *第十二章*，*友元工厂* 中有描述，来生成这些函数。到目前为止，我们能够对 `Value` 做的只是初始化它，也许将它赋值给另一个值，或者打印它：

```cpp
// Example 24
using V = Value<int>;
V i, j(5), k(3);
i = j;
std::cout << i;     // Prints 5
```

对于这个类，我们别无他法——没有用于相等或不等的比较，也没有算术运算。然而，我们可以创建一个适配器来添加比较接口：

```cpp
// Example 24
template <typename V> class Comparable : public V {
  public:
  using V::V;
  using V::operator=;
  using value_type = typename V::value_type;
  using basic_type = typename value_type::basic_type;
  Comparable(value_type v) : V(v) {}
  friend bool operator==(Comparable lhs, Comparable rhs) {
    return lhs.val_ == rhs.val_;
  }
  friend bool operator==(Comparable lhs, basic_type rhs) {
    return lhs.val_ == rhs;
  }
  friend bool operator==(basic_type lhs, Comparable rhs) {
    return lhs == rhs.val_;
  }
  ... same for the operator!= ...
};
```

这是一个类适配器——它从它增强的类中继承而来，因此继承了所有接口并添加了一些更多——完整的比较运算符集。请注意，在处理值类型时，通常使用值传递而不是引用传递（将引用传递给 `const` 也没有错，一些编译器可能会将两种版本都优化到相同的结果）。

我们熟悉这些适配器的使用方式：

```cpp
using V = Comparable<Value<int>>;
V i(3), j(5);
i == j; // False
i == 3; // True
5 == j; // Also true
```

那是其中一个功能。关于更多功能呢？没问题——`Ordered` 适配器可以非常相似地编写，只是它提供了 `<`、`<=`、`>` 和 `>=` 运算符（或者在 C++20 中，是 `<=>` 运算符）：

```cpp
// Example 24
template <typename V> class Ordered : public V {
  public:
  using V::V;
  using V::operator=;
  using value_type = typename V::value_type;
  using basic_type = typename value_type::basic_type;
  Ordered(value_type v) : V(v) {}
  friend bool operator<(Ordered lhs, Ordered rhs) {
    return lhs.val_ < rhs.val_;
  }
  friend bool operator<(basic_type lhs, Ordered rhs) {
    return lhs < rhs.val_;
  }
  friend bool operator<(Ordered lhs, basic_type rhs) {
    return lhs.val_ < rhs;
  }
  ... same for the other operators ...
};
```

我们可以将这两个适配器结合起来——正如我们所说的，它们是可组合的，并且可以在任何顺序下工作：

```cpp
using V = Ordered<Comparable<Value<int>>>;
// Or Comparable<Ordered<...>
V i(3), j(5);
i == j; // False
i <= 3; // True
```

一些操作或功能需要更多的工作。如果我们的值类型是数值类型，例如 `Value<int>`，我们可能希望有一些算术运算，比如加法和乘法。这里有一个允许加法和减法的装饰器：

```cpp
// Example 24
template <typename V> class Addable : public V {
  public:
  using V::V;
  using V::operator=;
  using value_type = typename V::value_type;
  using basic_type = typename value_type::basic_type;
  Addable(value_type v) : V(v) {}
  friend Addable operator+(Addable lhs, Addable rhs) {
    return Addable(lhs.val_ + rhs.val_);
  }
  friend Addable operator+(Addable lhs, basic_type rhs) {
    return Addable(lhs.val_ + rhs);
  }
  friend Addable operator+(basic_type lhs, Addable rhs) {
    return Addable(lhs + rhs.val_);
  }
  ... same for the operator- ...
};
```

装饰器使用起来非常简单：

```cpp
using V = Addable<Value<int>>;
V i(5), j(3), k(7);
k = i + j; // 8
```

我们还可以将 `Addable` 与其他装饰器结合使用：

```cpp
using V = Addable<Ordered<Value<int>>>;
V i(5), j(3), k(7);
if (k - 1 < i + j) { ... yes it is ... }
```

但我们有一个问题，到目前为止，这个问题只是因为好运才被隐藏起来。我们本来也可以这样写：

```cpp
using V = Ordered<Addable<Value<int>>>;
V i(5), j(3), k(7);
if (k - 1 < i + j) { ... }
```

这个例子与上一个例子之间不应该有任何区别。相反，我们得到了一个编译错误：最后一行没有有效的`operator<`可以使用。这里的问题是`i + j`表达式使用了来自`Addable`适配器的`operator+()`，而这个操作符返回的是类型为`Addable<Value<int>>`的对象。比较操作符期望的是类型`Ordered<Addable<Value<int>>>`，并且不会接受“部分”类型（从基类到派生类的隐式转换不存在）。令人不满意的解决方案是要求`Addable`始终是顶层装饰器。这不仅感觉不正确，而且也没有带我们走得很远：我们接下来想要的装饰器是`Multipliable`，它也会遇到同样的问题。当某物既是`Addable`又是`Multipliable`时，我们不能让两者都位于顶层。

注意，我们比较操作符返回`bool`时没有遇到任何问题，但一旦我们必须返回装饰后的类型本身，这正是`operator+()`所做的，组合性就会崩溃。为了解决这个问题，每个返回装饰后类型的操作符都必须返回原始（最外层）类型。例如，如果我们的值类型是`Ordered<Addable<Value<int>>>`，两个值相加的结果应该具有相同的类型。当然，问题是`operator+()`是由`Addable`装饰器提供的，它只知道`Addable`及其基类。我们需要在层次结构中添加一个中间类（`Addable<...>`），以返回其派生类型（`Ordered<Addable<...>>`）的对象。这是一个非常常见的设计问题，并且有一个模式：Curiously Recurring Template Pattern，或 CRTP（参见同名的*第八章**，Curiously Recurring Template Pattern*）。将此模式应用于我们的装饰器需要一些递归思考。我们将介绍两个主要思想，然后我们只需通过一个相当大的代码示例。

首先，每个装饰器都将有两个模板参数。第一个与之前相同：它是链中的下一个装饰器，或者在链的末尾是`Value<int>`（当然，这个模式不仅限于`int`，但我们通过在整个示例中保持相同的基类型来简化示例）。第二个参数将是最外层类型；我们将称之为“最终值类型”。因此，我们所有的装饰器都将这样声明：

```cpp
template <typename V, typename FV> class Ordered : ...
```

但在我们的代码中，我们仍然想要写

```cpp
using V = Ordered<Addable<Value<int>>>;
```

这意味着我们需要为第二个模板参数提供一个默认值。这个值可以是我们在装饰器中其他地方不会使用的任何类型；`void`将非常合适。我们还需要为这个默认类型提供一个部分模板特化，因为如果最终值类型没有明确指定，我们必须以某种方式确定它：

```cpp
template <typename V, typename FV = void> class Ordered;
template <typename V> class Ordered<V, void>;
```

现在，我们将逐步分析我们的“嵌套”类型 `Ordered<Addable<Value< int>>>`。在最外层，我们可以将其视为 `Ordered<T>`，其中 `T` 是 `Addable<Value<int>>`。由于我们没有指定 `Ordered` 模板的第二个类型参数 `FV`，我们将得到默认值 `void`，并且模板实例化 `Ordered<T>` 将使用 `Ordered` 模板的局部特化。即使我们没有指定“最终值类型” `FV`，我们也知道那是什么：它就是 `Ordered<T>` 本身。

现在我们需要确定要继承的基类。由于每个装饰器都从它装饰的类型继承，它应该是 `T`，即 `Addable<U>`（其中 `U` 是 `Value<int>`）。但这不会起作用：我们需要将正确的最终值类型传递给 `Addable`。因此，我们应该从 `Addable<U, FV>` 继承，其中 `FV` 是最终值类型 `Ordered<T>`。不幸的是，我们没有在代码中写出 `Addable<U, FV>`：我们有 `Addable<U>`。我们需要做的是以某种方式找出由相同的模板 `Addable` 但具有不同的第二个类型参数（`Ordered<T>` 而不是默认的 `void`）生成的类型。

这是在 C++ 模板中非常常见的问题，并且有一个同样常见的解决方案：模板重新绑定。我们所有的装饰器模板都需要定义以下模板别名：

```cpp
template <typename V, typename FV = void>
class Ordered : public ... some base class ... {
  public:
  template <typename FV1> using rebind = Ordered<V, FV1>;
};
```

现在，给定类型 `T`，它是装饰器模板之一的一个实例化，我们可以找出由相同的模板但具有不同的第二个模板参数 `FV` 产生的类型：它是 `T::template rebind<FV>`。这就是我们的 `Ordered<V>` 需要继承以传递正确的最终值类型给下一个装饰器的：

```cpp
// Example 25
template <typename V, typename FV = void>
class Ordered : public V::template rebind<FV> { ... };
```

这个类模板表明，给定类型 `Ordered<T, FV>`，我们将从重新绑定到相同最终值类型 `FV` 的类型 `T` 继承，并忽略 `T` 的第二个模板参数。这个例外是最外层的类型，其中模板参数 `FV` 是 `void`，但我们知道最终的值类型应该是什么，因此我们可以重新绑定到那个类型：

```cpp
// Example 25
template <typename V> class Ordered<V, void> :
  public V::template rebind<Ordered<V>> { ... };
```

注意语法，使用关键字 `template`：一些编译器将接受 `V:: rebind<Ordered<V>>`，但这是不正确的，标准要求这种确切的语法。

现在我们可以把所有东西放在一起。在装饰器链中间的通用情况下，我们必须将最终值类型传递给基类：

```cpp
// Example 25
template <typename V, typename FV = void>
class Ordered : public V::template rebind<FV> {
  using base_t = typename V::template rebind<FV>;
  public:
  using base_t::base_t;
  using base_t::operator=;
  template <typename FV1> using rebind = Ordered<V, FV1>;
  using value_type = typename base_t::value_type;
  using basic_type = typename value_type::basic_type;
  explicit Ordered(value_type v) : base_t(v) {}
  friend bool operator<(FV lhs, FV rhs) {
    return lhs.val_ < rhs.val_;
  }
  ... the rest of the operators ...
};
```

为了方便起见，引入了类型别名 `base_t`，这使得编写使用语句变得更容易。请注意，在依赖于模板参数的任何类型之前，我们需要使用 `typename` 关键字；我们不需要这个关键字来指定基类，因为基类始终是一个类型，所以编写 `typename` 将是多余的。

最外层类型的特殊情况，其中最终值类型未指定并默认为 `void`，非常相似：

```cpp
// Example 25
template <typename V> class Ordered<V, void>
  : public V::template rebind<Ordered<V>> {
  using base_t = typename V::template rebind<Ordered>;
  public:
  using base_t::base_t;
  using base_t::operator=;
  template <typename FV1> using rebind = Ordered<V, FV1>;
  using value_type = typename base_t::value_type;
  using basic_type = typename value_type::basic_type;
  explicit Ordered(value_type v) : base_t(v) {}
  friend bool operator<(Ordered lhs, Ordered rhs) {
    return lhs.val_ < rhs.val_;
  }
  ... the rest of the operators ...
};
```

特化与一般情况有两种不同之处。除了基类之外，操作符的参数不能是 `FV` 类型，因为它代表 `void`。相反，我们必须使用由模板生成的类的类型，在模板定义内部可以简单地称为 `Ordered`（当在类中使用时，模板的名称指的是特定的实例化 - 你不需要重复模板参数）。

对于那些操作符返回值的装饰器，我们需要确保始终使用正确的最终值类型来指定返回类型。在一般情况下，这是第二个模板参数 `FV`：

```cpp
// Example 25
template <typename V, typename FV = void> class Addable :
  public V::template rebind<FV> {
  friend FV operator+(FV lhs, FV rhs) {
    return FV(lhs.val_ + rhs.val_);
  }
  ...
};
```

在最外层装饰器的特化中，最终值类型是装饰器本身：

```cpp
// Example 25
template <typename V> class Addable<V, void> :
  public V::template rebind<FV> {
  friend Addable operator+(Addable lhs,Addable rhs) {
    return Addable(lhs.val_ + rhs.val_);
  }
  ...
};
```

我们必须将此技术应用于每个装饰器模板。现在我们可以以任何顺序组合装饰器，并使用任何可用操作的子集定义值类型：

```cpp
// Example 25
using V = Comparable<Ordered<Addable<Value<int>>>>;
// Addable<Ordered<Comparable<Value<int>>>> also OK
V i, j(5), k(3);
i = j; j = 1;
i == j;         // OK – Comparable
i > j;        // OK – Ordered
i + j == 7 – k;    // OK – Comparable and Addable
i*j;             // Not Multipliable – does not compile
```

到目前为止，我们所有的装饰器都向类中添加了成员或非成员操作符。我们也可以添加成员函数甚至构造函数。后者在需要添加转换时很有用。例如，我们可以添加一个从底层类型（如所写，`Value<T>` 不能隐式地从 `T` 构造）的隐式转换。转换装饰器遵循所有其他装饰器的相同模式，但添加了一个隐式转换构造函数：

```cpp
// Example 25
template <typename V, typename FV = void>
class ImplicitFrom : public V::template rebind<FV> {
  ...
  explicit ImplicitFrom(value_type v) : base_t(v) {}
  ImplicitFrom(basic_type rhs) : base_t(rhs) {}
};
template <typename V> class ImplicitFrom<V, void> :
  public V::template rebind<ImplicitFrom<V>> {
  ...
  explicit ImplicitFrom(value_type v) : base_t(v) {}
  ImplicitFrom(basic_type rhs) : base_t(rhs) {}
};
```

现在我们可以使用隐式转换到我们的值类型，例如，在调用函数时：

```cpp
using V = ImplicitFrom<Ordered<Addable<Value<int>>>>;
void f(V v);
f(3);
```

如果你想要一个到底层类型的隐式转换，你可以使用一个非常相似的适配器，但不是构造函数，而是添加了转换操作符：

```cpp
// Example 25
template <typename V, typename FV = void>
class ImplicitTo : public V::template rebind<FV> {
  ...
  explicit ImplicitTo(value_type v) : base_t(v) {}
  operator basic_type(){ return this->val_; }
  operator const basic_type() const { return this->val_; }
};
template <typename V> class ImplicitTo<V, void> :
  public V::template rebind<ImplicitTo<V>> {
  ...
  explicit ImplicitTo(value_type v) : base_t(v) {}
  operator basic_type(){ return this->val_; }
  operator const basic_type() const { return this->val_; }
};
```

这允许我们进行相反方向的转换：

```cpp
using V = ImplicitTo<Ordered<Addable<Value<int>>>>;
void f(int i);
V i(3);
f(i);
```

这种设计完成了工作，没有特别的问题，除了编写适配器的复杂性：CRTP 的递归应用往往会让你陷入无限递归，直到你习惯了这种类型的模板适配器的思考方式。另一种选择是策略基于的值类型。

## 策略解决方案

我们现在将研究一种与我们在*第十五章**，基于策略的设计*相比略有不同的形式。它并不像后者那样通用，但当它起作用时，它可以提供策略的所有优势，特别是可组合性，而没有一些问题。问题仍然是相同的：创建一个具有我们可以控制的操作集的自定义值类型。这个问题可以通过标准的基于策略的方法来解决：

```cpp
template <typename T, typename AdditionPolicy,
                      typename ComparisonPolicy,
                      typename OrderPolicy,
                      typename AssignmentPolicy, ... >
class Value { ... };
```

这种实现遇到了基于策略设计的所有缺点——策略列表很长，所有策略都必须明确列出，而且没有好的默认值；策略是位置相关的，因此类型声明需要仔细计算逗号的数量，并且随着新策略的添加，策略的任何有意义顺序都消失了。请注意，我们没有提到不同策略集创建不同类型的问题——在这种情况下，这并不是一个缺点，而是设计意图。如果我们想要一个支持加法且类似但不支持加法的类型，这些必须是不相同的类型。

理想情况下，我们只想列出我们想要我们的值拥有的策略——我想有一个基于整数的值类型，支持加法、乘法和赋值，但没有其他功能。毕竟，我们使用适配器模式做到了这一点，所以我们不会满足于任何更少的东西。实际上，有一种方法可以实现这一点。

首先，让我们思考一下这样的策略可能是什么样子。例如，允许加法的策略应该将 `operator+()` 注入类的公共接口（也许还可以注入 `operator+=()`）。使值可赋值的策略应该注入 `operator=()`。我们已经看到了足够多的此类策略，知道它们是如何实现的——它们必须是基类，公开继承，并且需要知道派生类的类型并将其转换为该类型，因此它们必须使用 CRTP：

```cpp
template <
  typename T,    // The base type (like int)
  typename V>    // The derived class
struct Incrementable {
  V operator++() {
    V& v = static_cast<V&>(*this);
    ++v.val_;     // The value inside the derived class
    return v;
  }
};
```

现在，我们需要考虑这些策略在主模板中的使用。首先，我们希望支持任何顺序的未知数量策略。这让我想起了变长模板。然而，为了使用 CRTP，模板参数必须是模板本身。然后，我们希望从每个这些模板的实例化中继承，无论有多少个。我们需要的是一个具有模板模板参数包的变长模板：

```cpp
// Example 26
template <typename T,
          template <typename, typename> class ... Policies>
class Value :
  public Policies<T, Value<T, Policies ... >> ...
{ ... };
```

前面的声明引入了一个名为 `Value` 的类模板，它至少有一个参数是类型，加上零个或多个模板策略，这些策略本身有两个类型参数（在 C++17 中，我们也可以用 `typename ... Policies` 代替 `class ... Policies`）。`Value` 类使用类型 `T` 和自身实例化这些模板，并从它们中公开继承。

`Value` 类模板应该包含我们希望所有值类型共有的接口。其余的将来自策略。让我们使值默认可复制、可赋值和可打印：

```cpp
// Example 26
template <typename T,
          template <typename, typename> class ... Policies>
class Value :
  public Policies<T, Value<T, Policies ... >> ...
{
  public:
  using base_type = T;
  explicit Value() = default;
  explicit Value(T v) : val_(v) {}
  Value(const Value& rhs) : val_(rhs.val_) {}
  Value& operator=(Value rhs) {
    val_ = rhs.val_;
    return *this;
  }
  Value& operator=(T rhs) { val_ = rhs; return *this; }
  friend std::ostream&
  operator<<(std::ostream& out, Value x) {
    out << x.val_; return out;
  }
  friend std::istream&
    operator>>(std::istream& in, Value& x) {
    in >> x.val_; return in;
  }
  private:
  T val_ {};
};
```

再次，我们使用来自*第十二章**，友元工厂*的*友元因子*来生成流操作符。

在我们能够尽情实现所有策略之前，还有一个障碍需要克服。`val_`值在`Value`类中是私有的，我们喜欢这种方式。然而，策略需要访问和修改它。在过去，我们通过将需要此类访问的每个策略都变成友元来解决此问题。这次，我们甚至不知道可能存在的策略名称。在处理完参数包扩展的声明作为一组基类之后，读者可能会合理地期望我们变魔术般地宣布与整个参数包的友谊。不幸的是，标准并没有提供这样的方法。我们能提出的最佳解决方案是提供一组只有策略可以调用的访问器函数，但没有好的方法来强制执行这一点（例如，一个名为`policy_accessor_do_not_call()`的名称可能有助于暗示用户代码应远离它，但程序员的创造力是无限的，这样的提示并不总是得到普遍尊重）：

```cpp
// Example 26
template <typename T,
          template <typename, typename> class ... Policies>
class Value :
  public Policies<T, Value<T, Policies ... >> ...
{
  public:
  ...
  T get() const { return val_; }
  T& get() { return val_; }
  private:
  T val_ {};
};
```

要创建一个具有受限操作集的值类型，我们必须使用我们想要的策略列表实例化此模板，没有其他内容：

```cpp
// Example 26
using V = Value<int, Addable, Incrementable>;
V v1(0), v2(1);
v1++; // Incrementable - OK
V v3(v1 + v2); // Addable - OK
v3 *= 2; // No multiplication policies - won't compile
```

我们可以实现的策略的数量和类型主要受当前需求（或想象力）的限制，但以下是一些示例，展示了如何向类添加不同类型的操作。

首先，我们可以实现前面提到的`Incrementable`策略，它提供了两个`++`运算符，后缀和前缀：

```cpp
// Example 26
template <typename T, typename V> struct Incrementable {
  V operator++() {
    V& v = static_cast<V&>(*this);
    ++(v.get());
    return v;
  }
  V operator++(int) {
    V& v = static_cast<V&>(*this);
    return V(v.get()++);
  }
};
```

我们可以为`--`运算符创建一个单独的`Decrementable`策略，或者如果对我们的类型有意义，可以有一个策略同时处理两者。此外，如果我们想以除了 1 以外的值进行增量，那么我们还需要`+=`运算符：

```cpp
// Example 26
template <typename T, typename V> struct Incrementable {
  V& operator+=(V val) {
    V& v = static_cast<V&>(*this);
    v.get() += val.get();
    return v;
  }
  V& operator+=(T val) {
    V& v = static_cast<V&>(*this);
    v.get() += val;
    return v;
  }
};
```

上述策略提供了`operator+=()`的两个版本 - 一个接受相同`Value`类型的增量，另一个接受基础类型`T`。这并不是一个要求，我们可以根据需要实现其他类型的值增量。我们甚至可以有多种增量策略版本，只要只使用其中一种（编译器会告诉我们是否引入了不兼容的重载）。

我们可以用类似的方式添加`*=`和`/=`运算符。添加如比较运算符或加法和乘法这样的二元运算符略有不同 - 这些运算符必须是非成员函数，以便允许对第一个参数进行类型转换。同样，友元工厂模式在这里很有用。让我们从比较运算符开始：

```cpp
// Example 26
template <typename T, typename V> struct ComparableSelf {
  friend bool operator==(V lhs, V rhs) {
    return lhs.get() == rhs.get();
  }
  friend bool operator!=(V lhs, V rhs) {
    return lhs.get() != rhs.get();
  }
};
```

当实例化时，此模板生成两个非成员非模板函数，即特定`Value`类变量的比较运算符，即实例化的那个。我们可能还希望允许与基础类型（如`int`）进行比较：

```cpp
template <typename T, typename V> struct ComparableValue {
  friend bool operator==(V lhs, T rhs) {
    return lhs.get() == rhs;
  }
  friend bool operator==(T lhs, V rhs) {
    return lhs == rhs.get();
  }
  friend bool operator!=(V lhs, T rhs) {
    return lhs.get() != rhs;
  }
  friend bool operator!=(T lhs, V rhs) {
    return lhs != rhs.get();
  }
};
```

更多的时候，我们可能同时想要这两种类型的比较。我们可以简单地将它们都放入同一个策略中，不必担心分离它们，或者我们可以从已有的两个策略中创建一个组合策略：

```cpp
// Example 26
template <typename T, typename V>
struct Comparable : public ComparableSelf<T, V>,
                    public ComparableValue<T, V> {};
```

在上一节中，我们从一开始就将所有比较组合在一个适配器中。这里，我们使用一种稍微不同的方法，只是为了说明使用策略或适配器（两种解决方案都提供相同的选项）控制类接口的不同选项。加法和乘法运算符是通过类似策略创建的。它们也是非模板非成员函数的朋友。唯一的区别是返回值类型——它们返回对象本身，例如：

```cpp
// Example 26
template <typename T, typename V> struct Addable {
  friend V operator+(V lhs, V rhs) {
    return V(lhs.get() + rhs.get());
  }
  friend V operator+(V lhs, T rhs) {
    return V(lhs.get() + rhs);
  }
  friend V operator+(T lhs, V rhs) {
    return V(lhs + rhs.get());
  }
};
```

如您所见，我们在编写适配器时遇到的返回“最终值类型”的问题在这里不存在：传递给每个策略的派生类本身就是值类型。

向基本类型转换的显式或隐式转换运算符可以同样轻松地添加：

```cpp
// Example 26
template <typename T, typename V>
struct ExplicitConvertible {
  explicit operator T() {
    return static_cast<V*>(this)->get();
  }
  explicit operator const T() const {
    return static_cast<const V*>(this)->get();
  }
};
```

这种方法乍一看似乎解决了传统基于策略的类型的大部分缺点。策略的顺序并不重要——我们只需指定我们想要的策略，不必担心其他的——有什么不喜欢的呢？然而，有两个基本限制。首先，基于策略的类不能通过名称引用任何策略。不再有`DeletionPolicy`或`AdditionPolicy`的位置。没有约定强制的策略接口，例如删除策略必须是可调用的。将策略绑定到单个类型的整个过程是隐式的；它只是接口的叠加。

因此，我们使用这些策略所能做的事情有限——我们可以注入公共成员函数和非成员函数——甚至可以添加私有数据成员——但我们不能为由主要基于策略的类确定和限制的行为方面提供实现。因此，这不是策略模式的实现——我们随意组合接口，以及实现，而不是定制特定的算法（这就是为什么我们将这种替代基于策略的设计模式的演示推迟到本章）。

第二个，与之紧密相关的限制是，没有默认策略。缺失的策略就是缺失。它们的位置上什么也没有。默认行为总是没有任何行为。在传统的基于策略的设计中，每个策略槽都必须被填充。如果有合理的默认值，可以指定它，然后除非用户覆盖它（例如，默认删除策略使用`operator delete`），否则这就是策略。如果没有默认值，编译器不会让我们省略策略——我们必须为模板提供一个参数。

这些限制的后果比你最初想象的要深远。例如，可能会诱使你使用我们在*第十五章**，基于策略的设计*中看到的`enable_if`技术，而不是通过基类注入公共成员函数。然后，我们可以有一个默认行为，如果没有其他选项，则启用。但在这里行不通。我们当然可以创建一个针对`enable_if`使用的策略：

```cpp
template <typename T, typename V> struct Addable {
  constexpr bool adding_enabled = true;
};
```

但使用它是没有可能的——我们无法使用`AdditionPolicy::adding_enabled`，因为没有`AdditionPolicy`——所有策略槽位都是未命名的。另一种选择是使用`Value::adding_enabled`——加法策略是`Value`的基类，因此，其所有数据成员在`Value`类中都是可见的。唯一的问题是这不起作用——在编译器评估此表达式（在定义`Value`类型作为 CRTP 策略的模板参数）时，`Value`是一个不完整类型，我们无法访问其数据成员。如果我们知道策略名称，我们可以评估`policy_name::adding_enabled`。但正是这种知识，我们为了不指定策略的完整列表而放弃了。

虽然严格来说，这不是策略模式的应用，但当策略主要用于控制一组支持的运算时，我们刚刚学到的基于策略的设计的替代方案可能很有吸引力。在讨论基于策略的设计指南时，我们提到，仅仅为了提供受限接口的额外安全性而使用策略槽位通常是不值得的。对于这种情况，这种替代方法应该被记住。

总体来看，我们可以看到这两种模式都有其优点和缺点：适配器依赖于更复杂的 CRTP 形式，而我们刚刚看到的“无槽位”策略要求我们做出妥协（我们必须使用类似我们的`get()`方法之类的机制将值暴露给策略）。

这就是我们作为软件工程师必须解决的问题的本质——一旦问题变得足够复杂，它就可以被解决，通常需要使用多种设计，每种方法都有其自身的优点和局限性。我们无法比较用于创建两种非常不同的设计以解决相同需求的每种模式，至少在有限大小的书中是不可能的。通过展示和分析这些示例，我们希望为读者提供理解和洞察，这将有助于评估类似复杂和多样的设计选项，以解决现实生活中的问题。

# 摘要

我们研究了两种最常用的模式——不仅限于 C++，而且在软件设计的一般领域。适配器模式提供了一种解决广泛设计挑战的方法。这些挑战只有一个最一般的共同属性——给定一个类、一个函数或一个提供特定功能的软件组件，我们必须解决一个特定问题，并为一个不同、相关的问题构建解决方案。在许多方面，装饰器模式是适配器模式的一个子集，它限制于通过添加新行为来增强类或函数的现有接口。

我们已经看到，适配器和装饰器执行的接口转换和修改可以应用于程序生命周期的每个阶段的接口——尽管最常见的用途是修改运行时接口，以便类可以在不同的上下文中使用，但也有编译时适配器用于泛型代码，允许我们将类用作构建块或更大、更复杂类的组件。

适配器模式可以应用于许多非常不同的设计挑战。这些挑战的多样性和模式的普遍性通常意味着可能存在另一种解决方案。这些替代方案通常采用完全不同的方法——一个完全不同的设计模式——但最终提供类似的行为。区别在于设计选择带来的权衡、附加条件和限制，以及以不同方式扩展解决方案的可能性。为此，本章提供了对两种非常不同的设计方法进行比较，包括对两种选项的优缺点评估。

接下来的，倒数第二个章节，介绍了一个庞大、复杂且具有多个相互作用的组件的模式——这是一个适合作为我们的压轴大戏——访问者模式。

# 问题

1.  什么是适配器模式？

1.  装饰器模式是什么，它与适配器模式有何不同？

1.  在 C++ 中，经典的面向对象（OOP）装饰器模式通常不推荐使用。为什么？

1.  在什么情况下，C++ 类装饰器应该使用继承或组合？

1.  在什么情况下，C++ 类适配器应该使用继承或组合？

1.  C++ 提供了一个通用的函数适配器用于柯里化函数参数，`std::bind`。它的局限性是什么？

1.  C++11 提供了模板别名，可以用作适配器。它们的局限性是什么？

1.  适配器和策略模式都可以用来添加或修改类的公共接口。给出一些选择其中一个而不是另一个的理由。
