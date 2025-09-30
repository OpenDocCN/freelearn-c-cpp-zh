

# 第十四章：模板方法模式和伪虚函数

模板方法是经典的*四人帮*设计模式之一，或者更正式地说，是 Erich Gamma、Richard Helm、Ralph Johnson 和 John Vlissides 在《设计模式 - 可复用面向对象软件元素》一书中描述的 24 个模式之一。它是一种行为设计模式，意味着它描述了不同对象之间通信的方式。作为面向对象的语言，C++当然完全支持模板方法模式，尽管本章将阐明一些特定于 C++的实现细节。

本章将涵盖以下主题：

+   模板方法模式是什么，它解决了什么问题？

+   什么是非虚接口？

+   你应该默认将虚函数设置为公有、私有还是保护？

+   你是否应该始终在多态类中将析构函数设置为虚的和公有的？

# 技术要求

本章的示例代码可以在以下 GitHub 链接中找到：[`github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/master/Chapter14`](https://github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/master/Chapter14).

# 模板方法模式

模板方法模式是实现一个算法的常见方式，其整体结构是预先确定的，但实现的一些细节需要定制。如果你正在考虑一个解决方案，类似于这样——首先，我们做*X*，然后*Y*，然后*Z*，但我们如何做*Y*取决于我们处理的数据——你正在考虑模板方法。作为一个允许程序行为动态变化的模式，模板方法在某种程度上类似于策略模式。关键区别在于，策略模式在运行时改变整个算法，而模板方法允许我们定制算法的特定部分。本节处理后者，而我们有专门的*第十六章*，*基于策略的设计*，专门用于前者。

## C++中的模板方法

模板方法模式在任何面向对象的语言中都可以轻松实现。C++实现使用继承和虚函数。请注意，这与泛型编程中的 C++模板无关。这里的*模板*是算法的骨架实现：

```cpp
// Example 01
class Base {
  public:
  bool TheAlgorithm() {
    if (!Step1()) return false; // Step 1 failed
    Step2();
    return true;
  }
};
```

这里的*模板*是算法的结构——所有实现都必须首先执行*步骤 1*，这可能失败。如果发生这种情况，整个算法被认为是失败的，不再进行任何操作。如果*步骤 1*成功，我们必须执行*步骤 2*。按照设计，*步骤 2*不能失败，一旦*步骤 2*完成，整体算法计算被认为是成功的。

注意到`TheAlgorithm()`方法是公开的但不是虚拟的——任何从`Base`派生的类都有它作为其接口的一部分，但不能覆盖它的行为。派生类可以覆盖的是在算法模板限制内的*步骤 1*和*步骤 2*的实现——*步骤 1*可能失败，必须通过返回`false`来表示失败，而*步骤 2*可能不会失败：

```cpp
// Example 01
class Base {
  public:
  ...
  virtual bool Step1() { return true };
  virtual void Step2() = 0;
};
class Derived1 : public Base {
  public:
  void Step2() override { ... do the work ... }
};
class Derived2 : public Base {
  public:
  bool Step1() override { ... check preconditions ... }
  void Step2() override { ... do the work ... }
};
```

在前面的例子中，覆盖可能失败的*步骤 1*是可选的，默认实现很简单；它什么都不做，永远不会失败。*步骤 2*必须由每个派生类实现——没有默认实现，并且它被声明为一个纯虚函数。

你可以看到整体的控制流程——框架——保持不变，但它有*占位符*用于可定制的选项，可能由框架本身提供默认值。这种流程被称为控制反转。在传统的控制流程中，它的具体实现决定了计算的流程和操作的顺序，并调用库函数或其他低级函数来实现必要的通用算法。在模板方法中，是框架在自定义代码中调用特定的实现。

## 模板方法的应用

使用模板方法有许多原因。一般来说，它用于控制可以和不可以被子类化的内容——与通用多态覆盖相反，在通用多态覆盖中，整个虚拟函数可以被替换，这里的基类决定了可以和不可以被覆盖的内容。模板方法的另一个常见用途是避免代码重复，在这种情况下，你可以这样得出使用模板方法的结论。假设你从一个常规的多态开始——一个虚拟函数——并覆盖它。例如，让我们考虑这个玩具设计，为一个游戏的回合制战斗系统：

```cpp
// Example 02
class Character {
  public:
  virtual void CombatTurn() = 0;
  protected:
  int health_;
};
class Swordsman : public Character {
  bool wielded_sword_;
  public:
  void CombatTurn() override {
    if (health_ < 5) { // Critically injured
      Flee();
      return;
    }
    if (!wielded_sword_) {
      Wield();
      return; // Wielding takes a full turn
    }
    Attack();
  }
};
class Wizard : public Character {
  int mana_;
  bool scroll_ready_;
  public:
  void CombatTurn() override {
    if (health_ < 2 ||
        mana_ == 0) { // Critically injured or out of mana
      Flee();
      return;
    }
    if (!scroll_ready_) {
      ReadScroll();
      return; // Reading takes a full turn
    }
    CastSpell();
  }
};
```

注意这个代码是多么的重复——所有角色可能在它们的回合被迫退出战斗，然后他们必须进行一个回合来为战斗做准备，只有在这种情况下，如果他们准备好了并且足够强大，他们才能使用他们的攻击能力。如果你看到这个模式反复出现，这是一个强烈的提示，可能需要调用模板方法。使用模板方法，战斗回合的整体顺序是固定的，但每个角色如何前进到下一步以及到达那里后他们做什么仍然是角色特定的：

```cpp
// Example 03
class Character {
  public:
  void CombatTurn() {
    if (MustFlee()) {
      Flee();
      return;
    }
    if (!Ready()) {
      GetReady();
      return; // Getting ready takes a full turn
    }
    CombatAction();
  }
  virtual bool MustFlee() const = 0;
  virtual bool Ready() const = 0;
  virtual void GetReady() = 0;
  virtual void CombatAction() = 0;
  protected:
  int health_;
};
```

现在每个派生类只需实现这个类独有的代码部分：

```cpp
// Example 03
class Swordsman : public Character {
  bool wielded_sword_;
  public:
  bool MustFlee() const override { return health_ < 5; }
  bool Ready() const override { return wielded_sword_; }
  void GetReady()override { Wield(); }
  void CombatAction()override { Attack(); }
};
class Wizard : public Character {
  int mana_;
  bool scroll_ready_;
  public:
  bool MustFlee() const override { return health_ < 2 ||
                                          mana_ == 0; }
  bool Ready() const override { return scroll_ready_; }
  void GetReady() override { ReadScroll(); }
  void CombatAction() override { CastSpell(); }
};
```

注意这段代码的重复性明显减少。尽管模板方法的优势不仅仅在于外观上的美观。假设在游戏的下一个版本中，我们增加了治疗药水，并且在回合开始时，每个角色都可以喝上一瓶药水。现在，想象一下需要遍历每一个派生类并添加类似 `if (health_ < ... some class-specific value ... && potion_count_ > 0) ...` 的代码。如果设计已经使用了模板方法，那么药水饮用的逻辑只需要编写一次，不同的类实现它们使用药水的特定条件，以及饮用药水的后果。然而，在你读完这一章之前，不要急于实施这个解决方案，因为这并不是你能编写的最好的 C++代码。

## 预条件和后置条件以及动作

模板方法的另一个常见用途是处理预条件和后置条件或动作。在类层次结构中，预条件和后置条件通常验证在执行过程中，接口提供的抽象设计不变量没有被任何特定的实现违反。这种验证自然符合模板方法的设计：

```cpp
// Example 04
class Base {
  public:
  void VerifiedAction() {
    assert(StateIsValid());
    ActionImpl();
    assert(StateIsValid());
  }
  virtual void ActionImpl() = 0;
};
class Derived : public Base {
  public:
  void ActionImpl() override { ... real implementation ...}
};
```

不变量是对象在客户端可访问时必须满足的要求，即在任何成员函数被调用之前或返回之后。成员函数本身通常需要暂时破坏不变量，但它们必须在将控制权返回给调用者之前恢复类的正确状态。让我们假设我们前面例子中的类跟踪执行了多少个动作。每个动作在开始时注册，完成时再次注册，这两个计数必须相同：一旦一个动作被启动，它必须完成，然后才能将控制权返回给调用者。当然，在 `ActionImpl()` 成员函数内部，这个不变量被违反了，因为动作正在进行中：

```cpp
// Example 04
class Base {
  bool StateIsValid() const {
    return actions_started_ == actions_completed_;
  }
  protected:
  size_t actions_started_ = 0;
  size_t actions_completed_ = 0;
  public:
  void VerifiedAction() {
    assert(StateIsValid());
    ActionImpl();
    assert(StateIsValid());
  }
  virtual void ActionImpl() = 0;
};
class Derived : public Base {
  public:
  void ActionImpl() override {
    ++actions_started_;
    ... perform the action ...
    ++actions_completed_;
  }
};
```

当然，任何实际的预条件和后置条件的实现都必须考虑几个额外的因素。首先，一些成员函数可能有额外的不变量，即它们只能在对象处于受限制状态时调用。这样的函数将具有特定的前置条件进行测试。其次，我们没有考虑动作由于错误而中止的可能性（这可能涉及抛出异常）。一个精心设计的错误处理实现必须保证在错误发生后类的不变量没有被违反。在我们的例子中，一个失败的动作可能完全被忽略（在这种情况下，我们需要减少已启动动作的计数）或者我们的不变量可能需要更复杂：所有已启动的动作最终都必须完成或失败，我们需要计算两者：

```cpp
// Example 05
class Base {
  bool StateIsValid() const {
    return actions_started_ ==
      actions_completed_ + actions_failed_;
  }
  protected:
  size_t actions_started_ = 0;
  size_t actions_completed_ = 0;
  size_t actions_failed_ = 0;
  ...
};
class Derived : public Base {
  public:
  void ActionImpl() override {
    ++actions_started_;
    try {
      ... perform the action – may throw ...
      ++actions_completed_;
    } catch (...) {
      ++actions_failed_;
    }
  }
};
```

在实际的程序中，你必须确保失败的交易不仅被正确计数，而且也要得到正确的处理（通常，它必须被撤销）。我们已经在*第五章* *全面审视 RAII*和*第十一章* *ScopeGuard*中进行了详细讨论。最后，在并发程序中，一个对象在成员函数执行期间无法被观察的事实不再成立，类不变性的整个主题变得更加复杂，并且与线程安全保证交织在一起。

当然，在软件设计中，一个人的不变性是另一个人的定制点。有时，主要代码保持不变，但发生的事情取决于具体的应用。在这种情况下，我们可能不会验证任何不变性，而是执行初始和最终操作：

```cpp
// Example 06
class FileWriter {
  public:
  void Write(const char* data) {
    Preamble(data);
    ... write data to a file ...
    Postscript(data);
  }
  virtual void Preamble(const char* data) {}
  virtual void Postscript(const char* data) {}
};
class LoggingFileWriter : public FileWriter {
  public:
  using FileWriter::FileWriter;
  void Preamble(const char* data) override {
    std::cout << "Writing " << data << " to the file" <<
      std::endl;
  }
  void Postscript (const char*) override {
    std::cout << "Writing done" << std::endl;
  }
};
```

当然，没有理由将前置条件和后置条件与打开和关闭操作结合在一起——基类可以在主要实现前后有多个“标准”成员函数调用。

虽然这段代码完成了任务，但它仍然存在一些我们将要揭露的缺陷。

# 非虚接口

动态可定制算法部分的实现通常使用虚函数来完成。对于一般的模板方法模式，这不是必需的，但在 C++中，我们很少需要其他方式。现在，我们将专门关注使用虚函数并改进我们所学的知识。

## 虚函数和访问

让我们从一个问题开始——虚函数应该是公共的还是私有的？教科书中的面向对象设计风格使用公共虚函数，所以我们经常不加思考地使它们成为公共的。在模板方法中，这种做法需要重新评估——公共函数是类接口的一部分。在我们的情况下，类接口包括整个算法，以及我们在基类中设置的框架。这个函数应该是公共的，但它也是非虚的。算法某些部分的定制实现从未打算直接由类层次结构的客户端调用。它们只在一个地方使用——在非虚公共函数中，它们替换了我们放在算法模板中的占位符。

这个想法可能看起来微不足道，但它对许多程序员来说却是一个惊喜。我多次被问到这个问题——*C++甚至允许虚函数不是公共的吗？*事实上，语言本身对虚函数的访问没有限制；它们可以是私有的、受保护的或公共的，就像任何其他类的成员函数一样。这可能需要一些时间来理解；也许一个例子会有所帮助：

```cpp
// Example 07
class Base {
  public:
  void method1() { method2(); method3(); }
  virtual void method2() { ... }
  private:
  virtual void method3() { ... }
};
class Derived : public Base {
  private:
  void method2() override { ... }
  void method3() override { ... }
};
```

在这里，`Derived::method2()` 和 `Derived::method3()` 都是私有的。基类甚至可以调用其派生类的私有方法吗？答案是，它不必这样做——`Base::method1()` 只调用它自己的成员函数（分别是公共和私有）；调用同一类的私有成员函数没有问题。但如果实际类类型是 `Derived`，则在运行时会调用 `method2()` 的虚拟重写。这两个决定，“我是否可以调用” `method2()` 和 “哪个” `method2()`，发生在完全不同的时间——前者发生在包含 `Base` 类的模块编译时（而 `Derived` 类可能甚至还没有被编写），而后者发生在程序执行时（在那个点上，“私有”或“公共”这些词没有任何意义）。此外，请注意，正如前例中的 `method3()` 所示，虚拟函数及其重写可以有不同的访问权限。再次强调，编译时调用的函数（在我们的例子中是 `Base::method3()`）必须在调用点可访问；最终在运行时执行的覆盖函数不必如此（然而，如果我们直接在类外部调用 `Derived::method3()`，我们就会尝试调用该类的私有方法）。

```cpp
// Example 07
Derived* d = new Derived;
Base* b = d;
b->method2();    // OK, calls Derived::method2()
d->method2();    // Does not compile – private function
```

避免公共虚拟函数的另一个更根本的原因是，公共方法构成了类接口的一部分。虚拟函数的重写是对实现的定制。一个公共虚拟函数本质上同时执行这两个任务。同一个实体执行了两个非常不同的功能，这些功能不应该耦合在一起——声明公共接口和提供替代实现。这些功能各自有不同的约束——只要层次不变量保持不变，实现可以被以任何方式更改。但是，接口实际上不能通过虚拟函数来改变（除了返回协变类型，但这实际上并没有改变接口）。所有公共虚拟函数所做的只是重申，是的，公共接口仍然看起来像基类所声明的。这种两种非常不同的角色的混合需要更好的关注点分离。模板方法模式是对该设计问题的回答，在 C++中，它以**非虚拟接口**（NVI）的形式出现。

## C++中的 NVI 习语

公共虚拟函数的两个角色之间的紧张关系，以及由这些函数创建的不必要的定制点暴露，导致我们产生了将实现特定的虚拟函数设为私有的想法。Herb Sutter 在他的文章《虚拟性》([`www.gotw.ca/publications/mill18.htm`](http://www.gotw.ca/publications/mill18.htm))中建议，大多数，如果不是所有，虚拟函数都应该设为私有。

对于模板方法，将虚拟函数从公共部分移动到私有部分不会带来任何后果（除了看到私有虚拟函数时的初始震惊，如果你从未意识到 C++允许它们）：

```cpp
// Example 08 (NVI version of example 01)
class Base {
  public:
  bool TheAlgorithm() {
    if (!Step1()) return false; // Step 1 failed
    Step2();
    return true;
  }
  private:
  virtual bool Step1() { return true };
  virtual void Step2() = 0;
};
class Derived1 : public Base {
  void Step2() override { ... do the work ... }
};
class Derived2 : public Base {
  bool Step1() override { ... check preconditions ... }
  void Step2() override { ... do the work ... }
};
```

这个设计很好地将接口和实现分离开来——客户端接口始终是运行整个算法的一个调用。算法实现部分的可变性并没有在接口中得到体现。因此，仅通过公共接口访问这个类层次结构且不需要扩展层次结构（编写更多派生类）的用户，对这样的实现细节并不知情。为了了解这在实践中是如何工作的，你可以将本章中的每一个示例从公共虚拟函数转换为 NVI；我们将只做其中一个，即示例 06，其余的留给读者作为练习。

```cpp
// Example 09 (NVI version of example 06)
class FileWriter {
  virtual void Preamble(const char* data) {}
  virtual void Postscript(const char* data) {}
  public:
  void Write(const char* data) {
    Preamble(data);
    ... write data to a file ...
    Postscript(data);
  }
};
class LoggingFileWriter : public FileWriter {
  using FileWriter::FileWriter;
  void Preamble(const char* data) override {
    std::cout << "Writing " << data << " to the file" <<
      std::endl;
  }
  void Postscript (const char*) override {
    std::cout << "Writing done" << std::endl;
  }
};
```

NVI（Non-Virtual Interface）将接口的完全控制权交给了基类。派生类只能自定义这个接口的实现。基类可以确定并验证不变性，强加实现的总体结构，并指定哪些部分可以、必须和不能被自定义。NVI 还明确地将接口与实现分离。派生类的实现者不需要担心无意中将实现的一部分暴露给调用者——仅实现私有的方法只能被基类调用。

注意，派生类如`LoggingFileWriter`仍然可以声明自己的非虚拟函数`Write`。这在 C++中被称为“阴影”：在派生类中引入的名称会阴影（或使不可访问）所有具有相同名称的函数，这些函数原本会从基类继承而来。这会导致基类和派生类的接口发生分歧，这是一种非常不好的做法。不幸的是，基类实现者没有好的方法来防止有意阴影。有时，当打算作为虚拟覆盖的函数以略有不同的参数声明时，会发生意外阴影；如果所有覆盖都使用`override`关键字，则可以避免这种情况。

到目前为止，我们已经将所有自定义实现的虚函数设置为私有。然而，这并不是 NVI（Non-Virtual Interface）的主要观点——这个惯用表达式以及更一般的模板方法，关注的是使公共接口非虚。由此延伸，实现特定的覆盖不应是公共的，因为它们不是接口的一部分。但这并不意味着它们应该是私有的。这就留下了*受保护的*。那么，为算法提供自定义的虚函数应该是私有的还是受保护的？模板方法允许两者——层次结构的客户端不能直接调用任何一个，因此算法的框架不受影响。答案取决于派生类是否可能需要调用基类提供的实现。以下是一个后者的例子，考虑一个可以序列化并通过套接字发送到远程机器的类层次结构：

```cpp
// Example 10
class Base {
  public:
  void Send() { // Template Method used here
    ... open connection ...
    SendData(); // Customization point
    ... close connection ...
  }
  protected:
  virtual void SendData() { ... send base class data ... }
  private:
  ... data ...
};
class Derived : public Base {
  protected:
  void SendData() {
    Base::SendData();
    ... send derived class data ...
  }
};
```

在这里，框架由公共非虚方法`Base::Send()`提供，它处理连接协议，并在适当的时候通过网络发送数据。当然，它只能发送基类知道的数据。这就是为什么`SendData`是一个自定义点并且被设置为虚函数。派生类当然必须发送自己的数据，但仍然需要有人发送基类的数据，因此派生类调用基类中的受保护虚函数。

如果这个例子看起来好像缺少了什么，那是有充分理由的。虽然我们提供了发送数据的一般模板以及每个类处理其自身数据的一个自定义点，但还有一个应该由用户可配置的行为方面：*如何*发送数据。这是一个展示模板方法模式和策略模式（有时是隐晦的）之间差异的好地方。

## 模板方法 vs 策略

虽然本章不是关于策略模式，但它有时会与模板方法混淆，所以我们现在将澄清两者的区别。我们可以使用上一节的例子来做这件事。

我们已经使用模板方法为`Base::Send()`中“发送”操作的执行提供了一个整体模板。操作有三个步骤：打开连接、发送数据和关闭连接。发送数据是依赖于对象实际类型的步骤（它实际上是哪个派生类），因此它被明确指定为自定义点。模板的其余部分是固定的。

然而，我们需要另一种类型的自定义：在一般情况下，`Base`类不是定义如何打开和关闭连接的正确地方。派生类也不是：相同的对象可以通过不同类型的连接（套接字、文件、共享内存等）发送。这就是我们可以使用策略模式来定义通信策略的地方。策略由一个单独的类提供：

```cpp
// Example 11
class CommunicationStrategy {
  public:
  virtual void Open() = 0;
  virtual void Close() = 0;
  virtual void Send(int v) = 0;
  virtual void Send(long v) = 0;
  virtual void Send(double v) = 0;
  ... Send other types ...
};
```

模板函数不能是虚函数，这不是很令人沮丧吗？对于这个问题的更好解决方案，你必须等到 *第十五章*，*基于策略的设计*。无论如何，现在我们有了通信策略，我们可以用它来参数化 `Send()` 操作模板：

```cpp
// Example 11
class Base {
  public:
  void Send(CommunicationStrategy* comm) {
    comm->Open();
    SendData(comm);
    comm->Close();
  }
  protected:
  virtual void SendData(CommunicationStrategy* comm) {
    comm->Send(i_);
    ... send all data ...
  }
  private:
  int i_;
  ... other data members ...
};
```

注意，发送数据的模板基本上没有改变，但我们委托了具体步骤的实现给另一个类——策略。这是关键的区别：策略模式允许我们选择（通常在运行时）特定操作应使用哪种实现。公共接口是固定的，但整个实现完全取决于特定的策略。模板方法模式强制执行整体实现流程以及公共接口。只有算法的具体步骤可以定制。

第二个区别在于定制的位置：`Base::Send()` 以两种方式进行了定制。对模板的定制是在派生类中完成的；策略的实现由 `Base` 层次之外的类提供。

正如我们在本节开头所指出的，有很好的理由将所有虚成员函数默认设置为私有（或保护），这不仅仅适用于模板方法模式的应用。然而，有一个特定的成员函数——析构函数——值得单独考虑，因为析构函数的规则有所不同。

## 关于析构函数的说明

对 NVI 的整个讨论是对一个简单指南的详细阐述——使虚函数私有（或保护），并通过非虚基类函数呈现公共接口。这听起来不错，直到它与另一个众所周知的指南正面冲突——如果一个类至少有一个虚函数，那么它的析构函数也必须是虚函数。由于这两个存在冲突，需要一些澄清。

使析构函数为虚函数的原因是，如果对象以多态方式被删除——例如，通过基类指针删除派生类对象——则析构函数必须是虚函数；否则，只有类的基部分将被析构（通常的结果是类的*切片*，部分删除，尽管标准只是简单地声明结果是不确定的）。因此，如果对象是通过基类指针删除的，析构函数必须是虚函数；没有其他选择。但这只是唯一的原因。如果对象总是以正确的派生类型被删除，那么这个原因就不适用。这种情况并不少见：例如，如果派生类对象存储在容器中，它们将按其真实类型被删除。

容器必须知道为对象分配多少内存，因此它不能存储基类和派生类的混合对象，或者将它们作为基类对象删除（请注意，指向基类对象的指针容器是另一种完全不同的结构，通常是为了我们可以以多态方式存储和删除对象而专门创建的）。

现在，如果派生类必须以自身类型被删除，其析构函数不需要是虚函数。然而，如果有人在实际对象为派生类类型时调用基类的析构函数，仍然会发生不好的事情。为了安全地防止这种情况发生，我们可以将非虚基类析构函数声明为受保护的，而不是公共的。当然，如果基类不是抽象的，并且周围有基类和派生类的对象，那么两个析构函数都必须是公共的，更安全的选项是将它们声明为虚函数（可以实施运行时检查来验证基类析构函数没有被用来销毁派生类对象）。

顺便说一下，如果你只需要在基类中编写析构函数来实现多态删除（通过基类指针进行删除），编写`virtual ~Base() = default;`是完全可接受的——析构函数可以同时是`virtual`和`default`。

我们还必须警告读者不要尝试为类析构函数使用模板方法或非虚接口习惯用法。可能会诱使人们做类似这样的事情：

```cpp
// Example 12
class Base {
  public:
  ~Base() { // Non-virtual interface!
    std::cout << "Deleting now" << std::endl;
    clear(); // Employing Template Method here
    std::cout << "Deleting done" << std::endl;
  }
  protected:
  virtual void clear() { ... } // Customizable part
};
class Derived : public Base {
  private:
  void clear() override {
    ...
    Base::clear();
  }
};
```

然而，这不会起作用（如果基类有一个纯虚的`Base::clear()`而不是默认实现，它将以相当壮观的方式失败）。原因在于，在基类析构函数`Base::~Base()`内部，对象的实际、真实和真正类型不再是`Derived`。它是`Base`。没错——当`Derived::~Derived()`析构函数完成其工作并将控制权传递给基类析构函数时，对象的动态类型变为`Base`。

唯一其他以这种方式工作的类成员是构造函数——只要基类构造函数正在运行，对象的类型就是`Base`，然后当派生类构造函数开始运行时，类型变为`Derived`。对于所有其他成员函数，对象的类型始终是其创建时的类型。如果对象是以`Derived`类型创建的，那么这就是其类型，即使调用了基类的方法。那么，如果在先前的例子中，`Base::clear()`是纯虚函数，会发生什么？它仍然会被调用！结果取决于编译器；大多数编译器将生成代码来终止程序，并带有一些诊断信息，指出*调用了纯虚函数*。

# 非虚接口的缺点

在使用 NVI（非虚拟接口）方面，并没有太多缺点。这就是为什么总是将虚函数设为私有，并使用 NVI 来调用它们的指南被广泛接受。然而，在决定模板方法是否是正确的设计模式时，你必须注意一些考虑因素。使用模板模式可能会导致脆弱的层次结构。此外，使用模板模式可以解决的问题和那些更适合使用策略模式或 C++中的策略的问题之间存在一些重叠。我们将在本节中回顾这两个考虑因素。

## 可组合性

考虑一下`LoggingFileWriter`的早期设计。现在，假设我们还想有一个`CountingFileWriter`，它可以计算写入文件中的字符数：

```cpp
class CountingFileWriter : public FileWriter {
  size_t count_ = 0;
  void Preamble(const char* data) {
    count_ += strlen(data);
  }
};
```

这很简单。但是，没有理由计数文件写入器不能也进行日志记录。我们如何实现`CountingLoggingFileWriter`？没问题，我们有技术——将私有虚函数改为受保护的，并从派生类中调用基类版本：

```cpp
class CountingLoggingFileWriter : public LoggingFileWriter {
  size_t count_ = 0;
  void Preamble(const char* data) {
    count_ += strlen(data);
    LoggingFileWriter::Preamble(data);
  }
};
```

或者它应该是从`CountingFileWriter`继承的`LoggingCountingFileWriter`？请注意，无论哪种方式，都会有一些代码重复——在我们的例子中，计数代码同时存在于`CountingLoggingFileWriter`和`CountingFileWriter`中。随着我们添加更多变体，这种重复只会变得更糟。如果你需要可组合的自定义化，模板方法根本就不是正确的模式。为此，你应该阅读*第十五章*，*基于策略的设计*。

## 脆弱基类问题

脆弱基类问题不仅限于模板方法，在一定程度上，它是所有面向对象语言固有的。问题出现在对基类的更改破坏了派生类。为了了解这是如何发生的，特别是当使用非虚拟接口时，让我们回到文件写入器并添加一次性写入多个字符串的能力：

```cpp
class FileWriter {
  public:
  void Write(const char* data) {
    Preamble(data);
    ... write data to a file ...
    Postscript(data);
  }
  void Write(std::vector<const char*> huge_data) {
    Preamble(huge_data);
    for (auto data: huge_data) {
      ... write data to file ...
    }
    Postscript(huge_data);
  }
  private:
  virtual void Preamble(std::vector<const char*> data) {}
  virtual void Postscript(std::vector<const char*> data) {}
  virtual void Preamble(const char* data) {}
  virtual void Postscript(const char* data) {}
};
```

计数写入器会随着更改而保持最新：

```cpp
class CountingFileWriter : public FileWriter {
  size_t count_ = 0;
  void Preamble(std::vector<const char*> huge_data) {
    for (auto data: huge_data) count_ += strlen(data);
  }
  void Preamble(const char* data) {
    count_ += strlen(data);
  }
};
```

到目前为止，一切顺利。后来，一个有良好意图的程序员注意到基类存在一些代码重复，并决定对其进行重构：

```cpp
class FileWriter {
  public:
  void Write(const char* data) { ... no changes here ... }
  void Write(std::vector<const char*> huge_data) {
    Preamble(huge_data);
    for (auto data: huge_data) Write(data); // Code reuse!
    Postscript(huge_data);
  }
  private:
  ... no changes here ...
};
```

现在，派生类被破坏了——当写入字符串向量时，会调用`Write`的两个版本的计数自定义化，数据大小被计算了两次。请注意，我们不是在谈论更基本的脆弱性，即如果基类方法的签名发生变化，派生类中的重写方法可能会停止重写：这种脆弱性在很大程度上可以通过在*第一章*，*继承和多态简介*中推荐使用`override`关键字来避免。

尽管只要使用继承，就没有解决脆弱基类问题的通用方法，但使用模板方法时避免该问题的指南是直接的——当更改基类和算法结构或框架时，避免更改被调用的定制点。具体来说，不要跳过已经调用的任何定制选项，也不要向已经存在的选项中添加新的调用（只要默认实现是合理的，添加新的定制点是允许的）。如果无法避免这种更改，您需要审查每个派生类，以确定它是否依赖于现在已删除或替换的实现覆盖，以及这种更改的后果。

## 关于模板定制点的注意事项

这个简短的章节并不是模板方法的缺点，而是一个关于 C++ 中某个较为晦涩角落的警告。许多最初作为运行时行为（面向对象模式）开发的设计模式，在 C++ 泛型编程中找到了它们的编译时对应物。那么，编译时模板方法是否存在呢？

当然，有一个明显的例子：我们可以有一个函数或类模板，它接受函数参数，或者更普遍地说，接受可调用参数，用于算法的某个固定步骤。标准库中有许多例子，例如 `std::find_if`：

```cpp
std::vector<int> v = ... some data ...
auto it = std::find_if(v.begin(), v.end(),
                       [](int i) { return i & 1; });
if (it != v.end()) { ... } // even value found
```

`std::find_if` 的算法是已知的，无法更改，除非它在检查特定值是否满足调用者的谓词这一步。

如果我们想对类层次结构做同样的事情，我们可以使用成员函数指针（尽管通过 lambda 表达式调用成员函数更容易），但除了使用虚拟函数及其覆盖之外，没有其他方法可以说“*在具有不同名称的类上调用具有相同名称的成员函数*”。在泛型编程中没有与之等效的方法。

不幸的是，有一种情况，模板可能会意外地被定制，通常会产生意料之外的结果。考虑以下示例：

```cpp
// Example 15
void f() { ... }
template <typename T> struct A {
  void f() const { ... }
};
template <typename T> struct B : public A<T> {
  void h() { f(); }
};
B<int> b;
b.h();
```

在 `B<T>::h()` 内部调用的是哪个函数 `f()`？在符合标准的编译器中，应该是独立函数，即 `::f()`，而不是基类的成员函数！这可能会让人感到惊讶：如果 `A` 和 `B` 都是非模板类，那么就会调用基类的方法 `A::f()`。这种行为源于 C++ 解析模板的复杂性（如果你想了解更多关于这个话题的信息，可以搜索“两阶段模板解析”或“两阶段名称查找”，但这个问题远远超出了本书的主题）。

如果全局函数 `f()` 本来就不存在，会发生什么？那么我们不得不调用基类中的那个，不是吗？

```cpp
// Example 15
// No f() here!
template <typename T> struct A {
  void f() const { ... }
};
template <typename T> struct B : public A<T> {
  void h() { f(); } // Should not compile!
};
B<int> b;
b.h();
```

如果你尝试了这段代码并且它调用了`A<T>::f()`，那么你有一个有缺陷的编译器：标准规定这根本不应该编译！但如果你想要调用你自己的基类的成员函数，你应该怎么做？答案是简单但如果你没有编写很多模板代码可能会看起来奇怪：

```cpp
// Example 15
template <typename T> struct A {
  void f() const { ... }
};
template <typename T> struct B : public A<T> {
  void h() { this->f(); } // Definitely A::f()
};
B<int> b;
b.h();
```

没错，你必须显式地调用`this->f()`来确保你正在调用一个成员函数。如果你这样做，无论是否声明了全局的`f()`，都会调用`A<T>::f()`。顺便说一句，如果你打算调用全局函数，明确这样做的方式是`::f()`，或者如果函数在命名空间`NS`中，则是`NS::f()`。

编译器无法找到在基类中明显存在的成员函数的编译错误是 C++中较为令人困惑的错误之一；如果编译器没有报告这个错误，而是“按预期”编译了代码，那就更糟糕了：如果后来有人添加了一个具有相同名称的全局函数（或者它在另一个你包含的头文件中声明），编译器将无警告地切换到那个函数。一般准则是在类模板中对成员函数调用使用`this->`进行限定。

总体而言，模板方法是少数几个在 C++中仍然纯粹面向对象的模式之一：我们看到的`std::find_if`（以及许多其他模板）的模板形式通常属于我们将在下一章研究的基于策略的设计的一般范畴。

# 摘要

在本章中，我们回顾了经典面向对象设计模式之一，即模板方法，以及它如何应用于 C++程序。这个模式在 C++中以及任何其他面向对象的语言中都适用，但 C++也有自己风格的模板方法——非虚接口习语。这种设计模式的优势导致了一个相当广泛的准则——将所有虚函数设为私有或保护。然而，关于多态，要注意析构函数的具体细节。以下是访问（公共与私有）虚函数的一般准则：

1.  更倾向于使用模板方法设计模式将接口设为非虚

1.  更倾向于将虚函数设为私有

1.  只有当派生类需要调用虚拟函数的基类实现时，才将虚拟函数设为保护。

1.  基类析构函数应该是公共和虚的（如果对象通过基类指针被删除）或者保护和非虚的（如果直接删除派生对象）。

我们在本章中已经通过阐明它与模板方法模式的区别来提到了策略模式。策略也是 C++中一个流行的模式，特别是它的泛型编程等价物。这将是下一章的主题。

# 问题

1.  什么是行为设计模式？

1.  模板方法模式是什么？

1.  为什么模板方法被认为是行为模式？

1.  控制反转是什么，它如何应用于模板方法？

1.  非虚接口是什么？

1.  为什么建议在 C++ 中将所有虚函数设置为私有？

1.  应该在何时将虚函数设置为保护？

1.  为什么模板方法不能用于析构函数？

1.  什么是脆弱基类问题，以及我们在使用模板方法时如何避免它？

# 第四部分：高级 C++ 设计模式

本部分继续描述和详细解释 C++ 设计模式，并转向更高级的模式。其中一些模式使用了 C++ 语言的先进特性。其他模式代表了复杂的概念，并解决了更困难的设计问题。还有一些模式实现了非常开放的设计，其中解决方案的一部分可以分解为一种普遍接受的模式，但整个系统必须在非常广泛的范围内可定制。

本部分包含以下章节：

+   *第十五章*，*基于策略的设计*

+   *第十六章*，*适配器和装饰器*

+   *第十七章*，*访问者模式和多重分派*

+   *第十八章*，*并发模式*
