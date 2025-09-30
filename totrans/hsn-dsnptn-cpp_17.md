# 17

# 访问者模式与多态

访问者模式是另一种经典面向对象设计模式，它是 Erich Gamma、Richard Helm、Ralph Johnson 和 John Vlissides 在《设计模式——可重用面向对象软件元素》一书中介绍的 23 个模式之一。在面向对象编程的黄金时代，它是较为流行的模式之一，因为它可以使大型类层次结构更易于维护。近年来，由于大型复杂层次结构变得不那么常见，访问者模式在 C++中的使用有所下降，因为实现访问者模式相对复杂。泛型编程——特别是 C++11 和 C++14 中添加的语言特性——使得实现和维护访问者类变得更加容易，而旧模式的新应用也重新点燃了对它的部分兴趣。

本章将涵盖以下主题：

+   访问者模式

+   C++中访问者模式的实现

+   使用泛型编程简化访问者类

+   使用访问者处理组合对象

+   编译时访问者和反射

技术要求

本章的示例代码可以在以下 GitHub 链接中找到：[`github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/main/Chapter17`](https://github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/main/Chapter17)。

# 访问者模式

访问者模式因其复杂性而与其他经典面向对象模式脱颖而出。一方面，访问者模式的基本结构相当复杂，涉及许多必须协同工作的类，以形成该模式。另一方面，即使是访问者模式的描述也很复杂——有几种非常不同的方式来描述同一个模式。许多模式可以应用于多种类型的问题，但访问者模式超越了这一点——有几种描述其功能的语言，讨论看似无关的问题，总体上没有共同之处。然而，它们都描述了同一个模式。让我们首先考察访问者模式的众多面貌，然后继续探讨其实现。

## 什么是访问者模式？

访问者模式是一种将算法与对象结构分离的模式，这是该算法的数据。使用访问者模式，我们可以在不修改类本身的情况下向类层次结构添加新的操作。访问者模式的使用遵循软件设计的**开/闭原则** - 一个类（或另一个代码单元，如模块）应该对修改封闭；一旦类向其客户端提供了一个接口，客户端就会依赖于这个接口及其提供的功能。这个接口应该保持稳定；不应该需要修改类来维护软件并继续其开发。同时，一个类应该对扩展开放 - 可以添加新功能以满足新的需求。与所有非常通用的原则一样，可以找到一个反例，其中严格应用规则比违反规则更糟。同样，与所有通用原则一样，其价值不在于成为每个情况的绝对规则，而在于一个**默认**规则，一个在没有充分理由不遵循的情况下应该遵循的指南；现实是，大多数日常工作的**不特殊**，如果遵循这个原则，结果会更好。

从这个角度来看，访问者模式允许我们在不修改类的情况下向类或整个类层次结构添加功能。当处理公共 API 时，这个特性尤其有用 - API 的用户可以扩展它以添加额外的操作，而无需修改源代码。

描述访问者模式的一个非常不同、更技术的方法是说它实现了**双分派**。这需要一些解释。让我们从常规的虚函数调用开始：

```cpp
class Base {
  virtual void f() = 0;
};
class Derived1 : public Base {
  void f() override;
};
class Derived2 : public Base {
  void f() override;
};
```

如果我们通过指向`b`基类的指针调用`b->f()`虚函数，调用将根据对象的实际类型分发到`Derived1::f()`或`Derived2::f()`。这是**单分派** - 实际调用的函数由一个单一因素决定，即对象类型。

现在让我们假设函数`f()`还接受一个指向基类的指针作为参数：

```cpp
class Base {
  virtual void f(Base* p) = 0;
};
class Derived1 : public Base {
  void f(Base* p) override;
};
class Derived2 : public Base {
  void f(Base* p) override;
};
```

`*p`对象的实际类型也是派生类之一。现在，`b->f(p)`调用可以有四种不同的版本；`*b`和`*p`对象可以是两种派生类型中的任何一种。在每种情况下都希望实现不同的行为是合理的。这将实现**双分派** - 最终运行的代码由两个独立因素决定。虚函数不提供直接实现双分派的方法，但访问者模式正是如此。

以这种方式呈现时，并不明显地看出双分派访问者模式与操作添加访问者模式有什么关系。然而，它们实际上是同一个模式，这两个要求实际上是相同的。这里有另一种看待它的方法，可能有助于理解——如果我们想向层次结构中的所有类添加一个操作，那么这相当于添加一个虚函数，因此我们有一个因素控制着每个调用的最终处理，即对象类型。但是，如果我们能够有效地添加虚函数，我们可以添加多个——每个操作一个。操作类型是控制分派的第二个因素，类似于我们之前示例中的函数参数。因此，操作添加访问者能够提供双分派。或者，如果我们有实现双分派的方法，我们可以做访问者模式所能做的——为每个我们想要支持的运算添加一个虚拟函数。

现在我们已经知道了访问者模式的作用，合理的疑问是，*为什么*我们要这样做？双分派有什么用？当我们可以直接添加一个*真正的*虚函数时，为什么我们还想*另一种方式*给一个类添加一个虚拟函数的替代品？不考虑公共 API 不可用源代码的情况，为什么我们想要在外部添加一个操作而不是在每一个类中实现它？考虑序列化/反序列化问题。序列化是一个将对象转换为可以存储或传输的格式的操作（例如，写入文件）。反序列化是逆操作——它从其序列化和存储的图像中构建一个新的对象。为了以简单、面向对象的方式支持序列化和反序列化，层次结构中的每个类都需要两个方法，每个操作一个。但如果存在多种存储对象的方式呢？例如，我们可能需要将对象写入内存缓冲区，以便在网络中传输并在另一台机器上反序列化。或者，我们可能需要将对象保存到磁盘，或者我们可能需要将容器中的所有对象转换为 JSON 等标记格式。直接的方法会要求我们为每种序列化机制给每个对象添加序列化和反序列化方法。如果需要新的不同的序列化方法，我们必须遍历整个类层次结构并添加对其的支持。

另一种选择是在一个单独的函数中实现整个序列化/反序列化操作，该函数可以处理所有类。生成的代码是一个循环，它遍历所有对象，并在其中包含一个大的决策树。代码必须查询每个对象并确定其类型，例如，使用动态类型转换。当向层次结构中添加新类时，必须更新所有序列化和反序列化实现以处理新对象。

这两种实现对于大型层次结构来说都难以维护。访问者模式提供了一个解决方案 - 它允许我们在类外部实现一个新操作 - 在我们的情况下，是序列化 - 而不修改它们，但也没有循环中巨大的决策树的缺点（注意，访问者模式不是序列化问题的唯一解决方案；C++还提供了其他可能的方法，但我们在本章中专注于访问者模式）。

正如我们一开始所说的，访问者模式是一个复杂的模式，具有复杂的描述。我们可以通过研究具体示例来最好地处理这个困难的模式，从下一节中的非常简单的示例开始。

## C++中的基本访问者

真正理解访问者模式如何操作的唯一方法是通过一个示例来工作。让我们从一个非常简单的示例开始。首先，我们需要一个类层次结构：

```cpp
// Example 01
class Pet {
  public:
  virtual ~Pet() {}
  Pet(std::string_view color) : color_(color) {}
  const std::string& color() const { return color_; }
  private:
  const std::string color_;
};
class Cat : public Pet {
  public:
  Cat(std::string_view color) : Pet(color) {}
};
class Dog : public Pet {
  public:
  Dog(std::string_view color) : Pet(color) {}
};
```

在这个层次结构中，我们有`Pet`基类和几个派生类，用于不同的宠物动物。现在我们想要给我们的类添加一些操作，比如“喂宠物”或“和宠物玩耍”。实现取决于宠物的类型，所以如果直接添加到每个类中，这些将必须是虚拟函数。对于这样一个简单的类层次结构来说，这不是问题，但我们预计未来需要维护一个更大的系统，其中修改层次结构中的每个类将非常昂贵且耗时。我们需要一种更好的方法，我们首先创建一个新的类，`PetVisitor`，它将被应用于每个`Pet`对象（访问它）并执行我们需要的操作。首先，我们需要声明这个类：

```cpp
// Example 01
class Cat;
class Dog;
class PetVisitor {
  public:
  virtual void visit(Cat* c) = 0;
  virtual void visit(Dog* d) = 0;
};
```

我们必须提前声明`Pet`层次结构类，因为`PetVisitor`必须在具体的`Pet`类之前声明。现在我们需要使`Pet`层次结构可访问，这意味着我们确实需要修改它，但只需修改一次，无论我们以后想添加多少操作。我们需要为每个可访问的类添加一个虚拟函数以接受访问者模式：

```cpp
// Example 01
class Pet {
  public:
  virtual void accept(PetVisitor& v) = 0;
  ...
};
class Cat : public Pet {
  public:
  void accept(PetVisitor& v) override { v.visit(this); }
  ...
};
class Dog : public Pet {
  public:
  void accept(PetVisitor& v) override { v.visit(this); }
  ...
};
```

现在我们的`Pet`层次结构是可访问的，并且我们有一个抽象的`PetVisitor`类。一切准备就绪，为我们的类实现新的操作（注意，到目前为止我们所做的一切都不依赖于我们将要添加的操作；我们已经创建了必须实现一次的访问基础设施）。操作是通过实现从`PetVisitor`派生的具体访问者类来添加的：

```cpp
// Example 01
class FeedingVisitor : public PetVisitor {
  public:
  void visit(Cat* c) override {
    std::cout << "Feed tuna to the " << c->color()
              << " cat" << std::endl;
  }
  void visit(Dog* d) override {
    std::cout << "Feed steak to the " << d->color()
              << " dog" << std::endl;
  }
};
class PlayingVisitor : public PetVisitor {
  public:
  void visit(Cat* c) override {
    std::cout << "Play with a feather with the "
              << c->color() << " cat" << std::endl;
  }
  void visit(Dog* d) override {
    std::cout << "Play fetch with the " << d->color()
              << " dog" << std::endl;
  }
};
```

假设访问者基础设施已经集成到我们的类层次结构中，我们可以通过实现一个派生访问者类来实施新的操作，并覆盖所有`visit()`的虚拟函数。要从我们的类层次结构中的对象上调用新的操作，我们需要创建一个访问者并访问该对象：

```cpp
// Example 01
Cat c("orange");
FeedingVisitor fv;
c.accept(fv); // Feed tuna to the orange cat
```

在调用访问者的最新示例中，有一个重要的方面过于简单——在调用访问者的那一刻，我们知道我们正在访问的对象的确切类型。为了使示例更真实，我们必须以多态的方式访问对象：

```cpp
// Example 02
std::unique_ptr<Pet> p(new Cat("orange"));
...
FeedingVisitor fv;
p->accept(fv);
```

在编译时，我们并不知道`p`所指向的对象的实际类型；在访问者被接受的那一刻，`p`可能来自不同的来源。虽然不太常见，访问者也可以被多态地使用：

```cpp
// Example 03
std::unique_ptr<Pet> p(new Cat("orange"));
std::unique_ptr<PetVisitor> v(new FeedingVisitor);
...
p->accept(*v);
```

当以这种方式编写时，代码突出了访问者模式的二分派特性——对`accept()`的调用最终会根据两个因素分派到特定的`visit()`函数——一个是可访问对象`*p`的类型，另一个是访问者`*v`的类型。如果我们想强调访问者模式的这一特性，我们可以使用辅助函数来调用访问者：

```cpp
// Example 03
void dispatch(Pet& p, PetVisitor& v) { p.accept(v); }
std::unique_ptr<Pet> p = ...;
std::unique_ptr<PetVisitor> v = ...;
dispatch(*p, *v); // Double dispatch
```

现在我们有了经典面向对象访问者模式在 C++中的最简单示例。尽管它很简单，但它包含了所有必要的组件；对于大型现实生活类层次结构和多个访问者操作，代码会更多，但并没有新的代码类型，只是更多我们已经做过的东西。这个示例展示了访问者模式的两个方面；一方面，如果我们关注软件的功能，现在有了访问者基础设施，我们可以添加新的操作而无需对类本身进行任何更改。另一方面，如果我们只看操作调用的方式，即`accept()`调用，我们已经实现了双分派。

我们可以立即看到访问者模式的吸引力，我们可以添加任意数量的新操作，而无需修改层次结构中的每个类。如果向`Pet`层次结构中添加了一个新类，不可能忘记处理它——如果我们对访问者不做任何操作，新类上的`accept()`调用将无法编译，因为没有相应的`visit()`函数可以调用。一旦我们向`PetVisitor`基类添加了新的`visit()`重载，我们也必须将其添加到所有派生类中；否则，编译器会告诉我们有一个没有重载的纯虚函数。后者也是访问者模式的主要缺点之一——如果向层次结构中添加了一个新类，所有访问者都必须更新，无论新类实际上是否需要支持这些操作。因此，有时建议只在*相对稳定*的层次结构上使用访问者，这些层次结构不经常添加新类。还有一个替代的访问者实现，它在某种程度上减轻了这个问题；我们将在本章后面看到它。

本节中的示例非常简单——我们的新操作不接收任何参数也不返回任何结果。我们现在将考虑这些限制是否重要，以及它们如何被消除。

## 访问者泛化和限制

在上一节中，我们的第一个访问者使我们能够有效地为层次结构中的每个类添加一个虚函数。这个虚函数没有参数也没有返回值。前者很容易扩展；我们的`visit()`函数为什么不能有参数，完全没有理由。让我们通过允许我们的宠物拥有小猫和小狗来扩展我们的类层次结构。仅使用访问者模式无法完成这个扩展——我们需要添加不仅新的操作，还有新的数据成员。访问者模式可以用于前者，但后者需要代码更改。如果我们有先见之明提供适当的策略，基于策略的设计可以让我们将这个更改因式分解为现有策略的新实现。我们确实在本书中有一个关于*第十五章*，*基于策略的设计*的独立章节，所以在这里我们将避免混合多个模式，只是添加新的数据成员：

```cpp
// Example 04
class Pet {
  public:
  ..
  void add_child(Pet* p) { children_.push_back(p); }
  virtual void accept(PetVisitor& v, Pet* p = nullptr) = 0;
  private:
  std::vector<Pet*> children_;
};
```

每个父`Pet`对象跟踪其子对象（请注意，容器是一个指针向量，而不是唯一指针向量，因此对象不拥有其子对象，只是可以访问它们）。我们还添加了新的`add_child()`成员函数来向向量中添加对象。我们本来可以用访问者来做这件事，但这个函数是非虚的，所以我们必须只将它添加到基类中，而不是每个派生类中——访问者在这里是不必要的。`accept()`函数已被修改，增加了一个额外的参数，这个参数也必须添加到所有派生类中，它只是简单地转发到`visit()`函数：

```cpp
// Example 04
class Cat : public Pet {
  public:
  Cat(std::string_view color) : Pet(color) {}
  void accept(PetVisitor& v, Pet* p = nullptr) override {
    v.visit(this, p);
  }
};
class Dog : public Pet {
  public:
  Dog(std::string_view color) : Pet(color) {}
  void accept(PetVisitor& v, Pet* p = nullptr) override {
    v.visit(this, p);
  }
};
```

`visit()` 函数也必须修改以接受额外的参数，即使对于不需要它的访问者也是如此。因此，更改 `accept()` 函数的参数是一个昂贵的全局操作，如果可能的话，不应该经常进行，甚至根本不应该进行。请注意，层次结构中相同虚拟函数的所有覆盖版本已经必须具有相同的参数。访问者模式将这种限制扩展到使用相同的基本访问者对象添加的所有操作。对于这个问题的一个常见解决方案是使用聚合（将多个参数组合在一起的类或结构）来传递参数。`visit()` 函数被声明为接受对基本聚合类指针的引用，而每个访问者都接收对派生类的指针，该派生类可能具有额外的字段，并且根据需要使用它们。

现在，我们的额外参数将通过虚拟函数调用的链路传递给访问者，在那里我们可以利用它。让我们创建一个记录宠物出生并作为子对象添加新宠物对象到其父对象中的访问者：

```cpp
// Example 04
class BirthVisitor : public PetVisitor {
  public:
  void visit(Cat* c, Pet* p) override {
    assert(dynamic_cast<Cat*>(p));
    c->add_child(p);
  }
  void visit(Dog* d, Pet* p) override {
    assert(dynamic_cast<Dog*>(p));
    d->add_child(p);
  }
};
```

注意，如果我们想确保我们的家族树中没有生物学上的不可能性，验证必须在运行时进行——在编译时，我们不知道多态对象的实际类型。新的访问者与上一节中的访问者一样容易使用：

```cpp
Pet* parent; // A cat
BirthVisitor bv;
Pet* child(new Cat("calico"));
parent->accept(bv, child);
```

一旦我们建立了亲子关系，我们可能想检查我们的宠物家族。这是我们想要添加的另一个操作，需要另一个访问者：

```cpp
// Example 04
class FamilyTreeVisitor : public PetVisitor {
  public:
  void visit(Cat* c, Pet*) override {
    std::cout << "Kittens: ";
    for (auto k : c->children_) {
      std::cout << k->color() << " ";
    }
    std::cout << std::endl;
  }
  void visit(Dog* d, Pet*) override {
    std::cout << "Puppies: ";
    for (auto p : d->children_) {
      std::cout << p->color() << " ";
    }
    std::cout << std::endl;
  }
};
```

然而，我们遇到了一个小问题，因为按照目前的编写方式，代码将无法编译。原因是 `FamilyTreeVisitor` 类试图访问 `Pet::children_` 私有数据成员。这是访问者模式的一个弱点——从我们的角度来看，访问者向类添加新操作，就像虚拟函数一样，但从编译器的角度来看，它们是完全独立的类，根本不像 `Pet` 类的成员函数，也没有特殊访问权限。访问者模式的通常应用需要放松封装，有两种方式——我们可以允许对数据进行公共访问（直接或通过访问器成员函数）或声明访问者类为友元（这确实需要更改源代码）。在我们的例子中，我们将遵循第二条路线：

```cpp
class Pet {
  ...
  friend class FamilyTreeVisitor;
};
```

现在家族树访问者按预期工作：

```cpp
Pet* parent; // A cat
...
amilyTreeVisitor tv;
parent->accept(tv); // Prints kitten colors
```

与 `BirthVisitor` 不同，`FamilyTreeVisitor` 不需要额外的参数。

现在我们有了使用参数执行操作的访客。那么返回值怎么办呢？技术上讲，`visit()` 和 `accept()` 函数没有必须返回 `void` 的要求。它们可以返回任何其他类型。然而，它们必须返回相同类型的限制通常使得这种能力变得无用。虚函数可以有协变返回类型，其中基类虚函数返回某个类的对象，而派生类覆盖返回的对象是从该类派生出来的，但即使是这也通常过于限制。还有一个更简单、更有效的解决方案——每个访客对象的 `visit()` 函数可以完全访问该对象的数据成员。我们没有理由不能在访客类本身中存储返回值并在稍后访问它。这对于最常见的使用情况非常合适，即每个访客添加不同的操作，并且可能具有唯一的返回类型，但操作本身对于层次结构中的所有类通常具有相同的返回类型。例如，我们可以让我们的 `FamilyTreeVisitor` 计算子代总数并通过访客对象返回该值：

```cpp
// Example 05
class FamilyTreeVisitor : public PetVisitor {
  public:
  FamilyTreeVisitor() : child_count_(0) {}
  void reset() { child_count_ = 0; }
  size_t child_count() const { return child_count_; }
  void visit(Cat* c, Pet*) override {
    visit_impl(c, "Kittens: ");
  }
  void visit(Dog* d, Pet*) override {
    visit_impl(d, "Puppies: ");
  }
  private:
  template <typename T>
  void visit_impl(T* t, const char* s) {
    std::cout << s;
    for (auto p : t->children_) {
      std::cout << p->color() << " ";
        ++child_count_;
      }
      std::cout << std::endl;
  }
  size_t child_count_;
};
FamilyTreeVisitor tv;
parent->accept(tv);
std::cout << tv.child_count() << " kittens total"
          << std::endl;
```

这种方法在多线程程序中会带来一些限制——访客现在不是线程安全的，因为多个线程不能使用同一个访客对象来访问不同的宠物对象。最常见的解决方案是每个线程使用一个访客对象，通常是在调用访客的函数栈上创建的一个局部变量。如果这不可能，还有更复杂的选项可以给访客提供一个线程（线程局部）状态，但分析这些选项超出了本书的范围。另一方面，有时我们想在多次访问中累积结果，在这种情况下，将结果存储在访客对象中的先前技术可以完美工作。此外，请注意，相同的解决方案也可以用来将参数传递到访客操作中，而不是将它们添加到 `visit()` 函数中；我们可以在访客对象内部存储参数，然后我们就不需要任何特殊的东西来从访客那里访问它们。当参数在每次调用访客时都不变，但可能从一个访客对象到另一个访客对象有所变化时，这种技术特别有效。

让我们暂时回顾一下`FamilyTreeVisitor`的实现。请注意，它遍历父对象的子对象，并依次对每个对象调用相同的操作。然而，它并没有处理子对象的子对象——我们的家谱树只有一代。访问包含其他对象的对象的这个问题非常普遍，并且相当常见。我们本章开头提到的动机示例，序列化问题，完美地展示了这种需求——每个复杂对象都是通过逐个序列化其组件来序列化的，然后它们依次以相同的方式序列化，直到我们到达内置类型，如`int`和`double`，我们知道如何读写这些类型。下一节将更全面地处理访问复杂对象的问题。

# 访问复杂对象

在最后一节中，我们看到了访问者模式如何允许我们向现有层次结构中添加新操作。在一个示例中，我们访问了一个包含其他对象指针的复杂对象。访问者以有限的方式遍历这些指针。我们现在将考虑访问由其他对象组成或包含其他对象的对象的普遍问题，并在最后演示一个有效的序列化/反序列化解决方案。

## 访问组合对象

访问复杂对象的一般思想非常直接——在访问对象本身时，我们通常不知道如何处理每个组件或包含对象的详细信息。但是，有一种东西可以做到这一点——针对该对象类型的访问者被专门编写来处理该类，而不会处理其他任何东西。这个观察表明，正确处理组件对象的方法是简单地访问每个对象，并将问题委托给其他人（这是一种在编程和其他方面都普遍有效的技术）。

让我们先以一个简单的容器类为例来演示这个想法，比如`Shelter`类，它可以包含任意数量的宠物对象，代表等待领养的宠物：

```cpp
// Example 06
class Shelter {
  public:
  void add(Pet* p) {
    pets_.emplace_back(p);
  }
  void accept(PetVisitor& v) {
    for (auto& p : pets_) {
      p->accept(v);
    }
  }
  private:
  std::vector<std::unique_ptr<Pet>> pets_;
};
```

这个类本质上是一个适配器，用于使宠物对象的向量可访问（我们已经在同名的章节中详细讨论了适配器模式）。请注意，这个类的对象确实拥有它们包含的宠物对象——当`Shelter`对象被销毁时，向量中的所有`Pet`对象也会被销毁。任何包含唯一指针的容器都是一个拥有其包含对象的容器；这就是如何在`std::vector`等容器中存储多态对象的方式（对于非多态对象，我们可以存储对象本身，但这种情况不适用，因为从`Pet`派生的对象属于不同的类型。）

与我们当前问题相关的代码当然是`Shelter::accept()`，它决定了`Shelter`对象是如何被访问的。正如你所看到的，我们没有在`Shelter`对象本身上调用访问者。相反，我们将访问委托给每个包含的对象。由于我们的访问者已经编写好了来处理宠物对象，所以不需要做更多的事情。当`Shelter`被`FeedingVisitor`等访问者访问时，庇护所中的每只宠物都会被喂食，我们不需要编写任何特殊的代码来实现这一点。

复合对象的访问以类似的方式进行 - 如果一个对象由几个较小的对象组成，我们必须访问这些对象中的每一个。让我们考虑一个代表一个家庭及其两只宠物（狗和猫）的对象（在下面的代码中，照顾宠物的家庭成员没有被包括在内，但我们假设他们也在那里）：

```cpp
// Example 07
class Family {
  public:
  Family(const char* cat_color, const char* dog_color) :
  cat_(cat_color), dog_(dog_color) {}
  void accept(PetVisitor& v) {
    cat_.accept(v);
    dog_.accept(v);
  }
  private: // Other family members not shown for brevity
  Cat cat_;
  Dog dog_;
};
```

再次，使用来自`PetVisitor`层次结构的访问者访问家庭被委托，以便每个`Pet`对象都被访问，访问者已经拥有了处理这些对象所需的一切（当然，`Family`对象也可以接受其他类型的访问者，我们将不得不为它们编写单独的`accept()`方法）。

现在，最后，我们拥有了处理任意对象序列化和反序列化问题所需的所有部件。下一个小节将展示如何使用访问者模式来完成这项工作。

## 使用访问者进行序列化和反序列化

该问题本身在上一节中已详细描述 - 对于序列化，每个对象都需要被转换为一系列位，这些位需要被存储、复制或发送。动作的第一部分取决于对象（每个对象的转换方式不同），但第二部分取决于序列化的具体应用（保存到磁盘与通过网络发送不同）。实现取决于两个因素，因此需要双重分派，这正是访问者模式提供的。此外，如果我们有方法可以序列化某些对象然后反序列化它（从位序列中重建对象），那么当这个对象包含在其他对象中时，我们应该使用相同的方法。

为了演示使用访问者模式进行类层次结构的序列化和反序列化，我们需要一个比我们迄今为止使用的玩具示例更复杂的层次结构。让我们考虑这个二维几何对象的层次结构：

```cpp
// Example 08
class Geometry {
  public:
  virtual ~Geometry() {}
};
class Point : public Geometry {
  public:
  Point() = default;
  Point(double x, double y) : x_(x), y_(y) {}
  private:
  double x_ {};
  double y_ {};
};
class Circle : public Geometry {
  public:
  Circle() = default;
  Circle(Point c, double r) : c_(c), r_(r) {}
  private:
  Point c_;
  double r_ {};
};
class Line : public Geometry {
  public:
  Line() = default;
  Line(Point p1, Point p2) : p1_(p1), p2_(p2) {}
  private:
  Point p1_;
  Point p2_;
};
```

所有对象都从抽象的`Geometry`基类派生，但更复杂的对象包含一个或多个更简单的对象；例如，`Line`由两个`Point`对象定义。请注意，最终，我们所有的对象都是由`double`数字组成的，因此将序列化为一系列数字。关键是知道哪个`double`代表哪个对象的哪个字段；我们需要这个来正确地恢复原始对象。

要使用访问者模式序列化这些对象，我们遵循与上一节相同的流程。首先，我们需要声明基访问者类：

```cpp
// Example 08
class Visitor {
public:
  virtual void visit(double& x) = 0;
  virtual void visit(Point& p) = 0;
  virtual void visit(Circle& c) = 0;
  virtual void visit(Line& l) = 0;
};
```

这里还有一个额外的细节 - 我们也可以访问`double`值；每个访问者都需要适当地处理它们（写入它们、读取它们等）。访问任何几何对象最终都会导致访问它所组成的数字。

我们的基本`Geometry`类及其所有派生类都需要接受这个访问者：

```cpp
// Example 08
class Geometry {
  public:
  virtual ~Geometry() {}
  virtual void accept(Visitor& v) = 0;
};
```

当然，我们无法向`double`添加一个`accept()`成员函数，但我们将不必这样做。派生类的`accept()`成员函数，每个都由一个或多个数字和其他类组成，会按顺序访问每个数据成员：

```cpp
// Example 08
void Point::accept(Visitor& v) {
  v.visit(x_); // double
  v.visit(y_); // double
}
void Circle::accept(Visitor& v) {
  v.visit(c_); // Point
  v.visit(r_); // double
}
void Point::accept(Visitor& v) {
  v.visit(p1_); // Point
  v.visit(p2_); // Point
}
```

具体的访问者类，所有都是基`Visitor`类的派生，负责序列化和反序列化的具体机制。对象分解成其部分的顺序，一直到底层的数字，由每个对象控制，但访问者决定了如何处理这些数字。例如，我们可以使用格式化输入输出将所有对象序列化成一个字符串（类似于我们将数字打印到`cout`时得到的结果）：

```cpp
// Example 08
class StringSerializeVisitor : public Visitor {
public:
  void visit(double& x) override { S << x << " "; }
  void visit(Point& p) override { p.accept(*this); }
  void visit(Circle& c) override { c.accept(*this); }
  void visit(Line& l) override { l.accept(*this); }
  std::string str() const { return S.str(); }
  private:
  std::stringstream S;
};
```

字符串会在`stringstream`中累积，直到所有必要的对象都被序列化：

```cpp
// Example 08
Line l(...);
Circle c(...);
StringSerializeVisitor serializer;
serializer.visit(l);
serializer.visit(c);
std::string s(serializer.str());
```

现在我们已经将对象打印到字符串`s`中，我们可以从这个字符串中恢复它们，也许是在不同的机器上（如果我们安排将字符串发送到那里）。首先，我们需要反序列化的访问者：

```cpp
// Example 08
class StringDeserializeVisitor : public Visitor {
  public:
  StringDeserializeVisitor(const std::string& s) {
    S.str(s);
  }
  void visit(double& x) override { S >> x; }
  void visit(Point& p) override { p.accept(*this); }
  void visit(Circle& c) override { c.accept(*this); }
  void visit(Line& l) override { l.accept(*this); }
  private:
  std::stringstream S;
};
```

这个访问者从字符串中读取数字，并将它们保存为被访问对象提供的变量中。成功反序列化的关键是按照保存时的顺序读取数字 - 例如，如果我们首先写入一个点的 *X* 和 *Y* 坐标，我们应该从读取的前两个数字构建一个点，并将它们用作 *X* 和 *Y* 坐标。如果第一个写入的点是一条线的终点，我们应该使用构建的点作为新线的终点。访问者模式的美妙之处在于，执行实际读取和写入的函数不需要做任何特殊的事情来保持这个顺序 - 顺序由每个对象确定，并且对所有访问者保证是相同的（对象不会区分特定的访问者，甚至不知道它是什么类型的访问者）。我们唯一需要做的就是按照序列化时的顺序访问对象：

```cpp
// Example 08
Line l1;
Circle c1;
// s is the string from a serializer
StringDeserializeVisitor deserializer(s);
deserializer.visit(l1); // Restored Line l
deserializer.visit(c1); // Restored Circle c
```

到目前为止，我们已经知道了哪些对象被序列化以及它们的顺序。因此，我们可以以相同的顺序反序列化相同的对象。更一般的情况是，在反序列化过程中我们不知道期望哪些对象——对象存储在一个可访问的容器中，类似于早期示例中的 `Shelter`，它必须确保对象以相同的顺序进行序列化和反序列化。例如，考虑这个类，它存储一个表示为两个其他几何体交集的几何体：

```cpp
// Example 09
class Intersection : public Geometry {
  public:
  Intersection() = default;
  Intersection(Geometry* g1, Geometry* g2) :
    g1_(g1), g2_(g2) {}
  void accept(Visitor& v) override {
    g1_->accept(v);
    g2_->accept(v);
  }
  private:
  std::unique_ptr<Geometry> g1_;
  std::unique_ptr<Geometry> g2_;
};
```

此对象的序列化很简单——我们按顺序序列化几何体，通过将这些细节委托给这些对象来实现。我们不能直接调用 `v.visit()`，因为我们不知道 `*g1_` 和 `*g2_` 几何体的类型，但我们可以让这些对象根据适当的情况分派调用。但是，按照目前的写法，反序列化将失败——几何指针是 `null`，还没有分配任何对象，我们也不知道应该分配哪种类型的对象。某种方式，我们首先需要在序列化流中编码对象的类型，然后根据这些编码的类型构建它们。还有另一种模式为这个问题提供了标准的解决方案，那就是工厂模式（在构建复杂系统时，通常需要使用多个设计模式）。

有几种方法可以实现这一点，但它们都归结为将类型转换为数字并将这些数字序列化。在我们的情况下，当我们声明基类 `Visitor` 时，我们必须知道完整的几何类型列表，这样我们才能同时定义所有这些类型的枚举：

```cpp
// Example 09
class Geometry {
  public:
  enum type_tag {POINT = 100, CIRCLE, LINE, INTERSECTION};
  virtual type_tag tag() const = 0;
};
class Visitor {
  public:
  static Geometry* make_geometry(Geometry::type_tag tag);
  virtual void visit(Geometry::type_tag& tag) = 0;
  ...
};
```

`enum type_tag` 不一定需要在 `Geometry` 类内部定义，或者 `make_geometry` 工厂构造函数必须是 `Visitor` 类的静态成员函数。它们也可以在任何类外部声明，但返回每个派生几何类型正确标记的虚拟 `tag()` 方法需要按照所示方式声明。必须在每个派生 `Geometry` 类中定义 `tag()` 重写，例如，`Point` 类：

```cpp
// Example 09
class Point : public Geometry {
  public:
  ...
  type_tag tag() const override { return POINT; }
};
```

其他派生类也需要进行类似的修改。

然后，我们需要定义工厂构造函数：

```cpp
// Example 09
Geometry* Visitor::make_geometry(Geometry::type_tag tag) {
  switch (tag) {
    case Geometry::POINT: return new Point;
    case Geometry::CIRCLE: return new Circle;
    case Geometry::LINE: return new Line;
    case Geometry::INTERSECTION: return new Intersection;
  }
}
```

此工厂函数根据指定的类型标记构建正确的派生对象。剩下的只是让 `Intersection` 对象序列化和反序列化构成交集的两个几何体的标记：

```cpp
// Example 09
class Intersection : public Geometry {
  public:
  void accept(Visitor& v) override {
    Geometry::type_tag tag;
    if (g1_) tag = g1_->tag();
    v.visit(tag);
    if (!g1_) g1_.reset(Visitor::make_geometry(tag));
    g1_->accept(v);
    if (g2_) tag = g2_->tag();
    v.visit(tag);
    if (!g2_) g2_.reset(Visitor::make_geometry(tag));
    g2_->accept(v);
  }
  ...
};
```

首先，标记被发送到访问者。序列化访问者应该将标记与数据的其他部分一起写入：

```cpp
// Example 09
class StringSerializeVisitor : public Visitor {
  public:
  void visit(Geometry::type_tag& tag) override {
    S << size_t(tag) << " ";
  }
  ...
};
```

反序列化访问者必须读取标记（实际上，它读取一个 `size_t` 数字并将其转换为标记）：

```cpp
// Example 09
class StringDeserializeVisitor : public Visitor {
  public:
  void visit(Geometry::type_tag& tag) override {
    size_t t;
    S >> t;
    tag = Geometry::type_tag(t);
  }
  ...
};
```

一旦反序列化访问者恢复了标签，`Intersection`对象可以调用工厂构造函数来构建正确的几何对象。现在我们可以从这个流中反序列化这个对象，我们的`Intersection`就被恢复成了与序列化时完全相同的副本。请注意，还有其他方法来封装访问标签和调用工厂构造函数；最佳解决方案取决于系统中不同对象的角色——例如，反序列化访问者可能会根据标签而不是拥有这些几何形状的复合对象来构建对象。然而，需要发生的事件序列仍然是相同的。

到目前为止，我们一直在学习经典的面向对象访问者模式。在我们看到经典模式在 C++中的特定变化之前，我们应该了解另一种类型的访问者，它解决了访问者模式中的一些不便之处。

# 无环访问者

如我们所见，访问者模式到目前为止已经做到了我们想要它做的事情。它将算法的实现与作为算法数据的目标对象分离，并允许我们根据两个运行时因素来选择正确的实现——具体的对象类型和我们要执行的具体操作，这两个因素都从它们各自的类层次结构中选择。然而，这里有一个问题——我们想要减少复杂性并简化代码维护，我们确实做到了，但现在我们必须维护两个并行类层次结构，即可访问对象和访问者，以及两者之间的依赖关系是非平凡的。这些依赖关系中最糟糕的部分是它们形成了一个循环——访问者对象依赖于可访问对象的类型（对于每个可访问类型都有一个`visit()`方法的重载），而基础可访问类型依赖于基础访问者类型。这个依赖关系的前半部分是最糟糕的。每次向层次结构中添加新对象时，每个访问者都必须更新。后半部分对程序员的工作量不大，因为可以在任何时候添加新的访问者，而不需要任何其他更改——这就是访问者模式的核心所在。但仍然存在基础可访问类及其所有派生类对基础访问者类的编译时依赖。访问者的大部分接口和实现都是稳定的，除了一个情况——添加新的可访问类。因此，这个循环在操作中看起来是这样的——向可访问对象的层次结构中添加了一个新类。访问者类需要更新以包含新类型。由于基础访问者类已更改，基础可访问类及其所有依赖于它的代码行都必须重新编译，包括不使用新可访问类的代码，只使用旧的。即使尽可能使用前向声明也无法帮助——如果添加了新的可访问类，所有旧的都必须重新编译。

传统访问者模式的附加问题是必须处理对象类型和访问者类型的所有可能组合。通常情况下，有些组合是没有意义的，某些对象永远不会被某些类型的访问者访问。但我们不能利用这一点，因为每个组合都必须有一个定义好的动作（动作可能非常简单，但仍然，每个访问者类都必须定义完整的`visit()`成员函数集）。

无环访问者模式是访问者模式的一种变体，它专门设计用来打破依赖循环并允许部分访问。无环访问者模式的基础可访问类与常规访问者模式相同：

```cpp
// Example 10
class Pet {
  public:
  virtual ~Pet() {}
  virtual void accept(PetVisitor& v) = 0;
  ...
};
```

然而，相似之处到此为止。基访问者类没有为每个可访问对象提供`visit()`重载。事实上，它根本没有任何`visit()`成员函数：

```cpp
// Example 10
class PetVisitor {
  public:
  virtual ~PetVisitor() {}
};
```

那么，谁来进行访问呢？对于原始层次结构中的每个派生类，我们也声明相应的访问者类，这就是`visit()`函数所在的地方：

```cpp
// Example 10
class Cat;
class CatVisitor {
  public:
  virtual void visit(Cat* c) = 0;
};
class Cat : public Pet {
  public:
  Cat(std::string_view color) : Pet(color) {}
  void accept(PetVisitor& v) override {
    if (CatVisitor* cv = dynamic_cast<CatVisitor*>(&v)) {
      cv->visit(this);
    } else { // Handle error
      assert(false);
    }
  }
};
```

注意，每个访问者只能访问它被设计为访问的类——`CatVisitor`只访问`Cat`对象，`DogVisitor`只访问`Dog`对象，等等。魔法在于新的`accept()`函数——当一个类被要求接受一个访问者时，它首先使用`dynamic_cast`来检查这是否是正确的访问者类型。如果是，一切顺利，访问者被接受。如果不是，我们就有问题，必须处理错误（错误处理的精确机制取决于应用程序；例如，可以抛出异常）。因此，具体的访问者类必须从公共的`PetVisitor`基类以及如`CatVisitor`之类的特定基类派生：

```cpp
// Example 10
class FeedingVisitor : public PetVisitor,
                       public CatVisitor,
                       public DogVisitor {
  public:
  void visit(Cat* c) override {
    std::cout << "Feed tuna to the " << c->color()
              << " cat" << std::endl;
  }
  void visit(Dog* d) override {
    std::cout << "Feed steak to the " << d->color()
              << " dog" << std::endl;
  }
};
```

每个具体的访问者类都从公共访问者基类派生，并且从每个必须由该访问者处理的类型的每个特定类型的访问者基类（例如`CatVisitor`、`DogVisitor`等）派生。另一方面，如果这个访问者没有被设计为访问层次结构中的某些类，我们可以简单地省略相应的访问者基类，然后我们也就不需要实现虚拟函数的重载了：

```cpp
// Example 10
class BathingVisitor : public PetVisitor,
                       public DogVisitor
                       { // But no CatVisitor
  public:
  void visit(Dog* d) override {
    std::cout << "Wash the " << d->color()
              << " dog" << std::endl;
  }
  // No visit(Cat*) here!
};
```

无环访问者模式的调用方式与常规访问者模式完全相同：

```cpp
// Example 10
std::unique_ptr<Pet> c(new Cat("orange"));
std::unique_ptr<Pet> d(new Dog("brown"));
FeedingVisitor fv;
c->accept(fv);
d->accept(fv);
BathingVisitor bv;
//c->accept(bv); // Error
d->accept(bv);
```

如果我们尝试访问特定访问者不支持的对象，错误就会被检测到。因此，我们已经解决了部分访问的问题。那么依赖循环怎么办？这也得到了妥善处理——公共的`PetVisitor`基类不需要列出可访问对象的完整层次结构，具体的可访问类只依赖于它们各自的类访问者，而不是任何其他类型的访问者。因此，当向层次结构中添加另一个可访问对象时，现有的对象不需要重新编译。

Acyclic Visitor 模式看起来如此之好，以至于人们不禁要问，*为什么不总是使用它而不是常规的 Visitor 模式呢？* 有几个原因。首先，Acyclic Visitor 模式使用`dynamic_cast`从一个基类转换到另一个基类（有时称为交叉转换）。这个操作通常比虚函数调用更昂贵，所以 Acyclic Visitor 模式比替代方案更慢。此外，Acyclic Visitor 模式要求为每个可访问类提供一个 Visitor 类，因此类数增加了一倍，并且它使用了多个继承和许多基类。第二个问题对于大多数现代编译器来说不是什么大问题，但许多程序员发现处理多重继承很困难。第一个问题——动态转换的运行时成本——是否是问题取决于应用程序，但这是你需要注意的事情。另一方面，当可访问对象层次结构频繁变化或整个代码库重新编译的成本很高时，Acyclic Visitor 模式确实很出色。

你可能已经注意到 Acyclic Visitor 模式的一个问题——它有很多样板代码。对于每个可访问类，必须复制几行代码。实际上，常规的 Visitor 模式也面临着同样的问题，即实现任何一种 Visitor 都涉及到大量的重复输入。但是 C++有一套特殊的工具来用代码复用来替换代码重复：这正是泛型编程的目的。我们将在下一节中看到 Visitor 模式是如何适应现代 C++的。

# 现代 C++中的 Visitor

正如我们刚才看到的，Visitor 模式促进了关注点的分离；例如，序列化的顺序和序列化的机制被独立出来，每个都由一个单独的类负责。该模式还通过将执行给定任务的代码收集到一个地方来简化代码维护。Visitor 模式不促进的是没有重复的代码复用。但那是现代 C++之前的面向对象的 Visitor 模式。让我们看看我们可以如何利用 C++的泛型能力，从常规的 Visitor 模式开始。

## 泛型 Visitor

我们将尝试减少 Visitor 模式实现中的样板代码。让我们从`accept()`成员函数开始，它必须复制到每个可访问类中；它总是看起来一样：

```cpp
class Cat : public Pet {
  void accept(PetVisitor& v) override { v.visit(this); }
};
```

这个函数不能移动到基类，因为我们需要调用具有实际类型的访问者，而不是基类型——`visit()`接受`Cat*`、`Dog*`等，但不接受`Pet*`。如果我们引入一个中间的模板基类，我们可以得到一个模板来为我们生成这个函数：

```cpp
// Example 11
class Pet { // Same as before
  public:
  virtual ~Pet() {}
  Pet(std::string_view color) : color_(color) {}
  const std::string& color() const { return color_; }
  virtual void accept(PetVisitor& v) = 0;
  private:
  std::string color_;
};
template <typename Derived>
class Visitable : public Pet {
  public:
  using Pet::Pet;
  void accept(PetVisitor& v) override {
    v.visit(static_cast<Derived*>(this));
  }
};
```

模板由派生类参数化。在这方面，它与指向正确派生类指针的 `this` 指针类似。现在我们只需要从模板的正确实例化中派生每个宠物类，我们就会自动获得 `accept()` 函数：

```cpp
// Example 11
class Cat : public Visitable<Cat> {
  using Visitable<Cat>::Visitable;
};
class Dog : public Visitable<Dog> {
  using Visitable<Dog>::Visitable;
};
```

这样就处理了样板代码的一半——派生可访问对象内部的代码。现在只剩下另一半：访问者类内部的代码，在那里我们必须为每个可访问类重复相同的声明。我们对特定访问者无能为力；毕竟，那里才是真正的工作所在，而且，假设我们需要为不同的可访问类做不同的事情（否则为什么要使用双重分派呢？）

然而，如果我们引入这个通用访问者模板，我们可以简化基访问者类的声明：

```cpp
// Example 12
template <typename ... Types> class Visitor;
template <typename T> class Visitor<T> {
  public:
  virtual void visit(T* t) = 0;
};
template <typename T, typename ... Types>
class Visitor<T, Types ...> : public Visitor<Types ...> {
  public:
  using Visitor<Types ...>::visit;
  virtual void visit(T* t) = 0;
};
```

注意，我们只需要实现这个模板一次：不是为每个类层次结构实现一次，而是永远实现一次（或者至少直到我们需要更改 `visit()` 函数的签名，例如，添加参数）。这是一个好的通用库类。一旦我们有了它，声明特定类层次结构的访问者基类就变得如此简单，以至于感觉有些平淡无奇：

```cpp
// Example 12
```

注意到 `class` 关键字有些不寻常的语法——它将模板参数列表与前置声明结合起来，相当于以下内容：

```cpp
class Cat;
class Dog;
using PetVisitor = Visitor<Cat, Dog>;
```

通用访问者基类是如何工作的？它使用变长模板来捕获任意数量的类型参数，但主要模板只声明了，没有定义。其余的是特化。首先，我们有一个只有一个类型参数的特殊情况。我们为该类型声明了纯 `visit()` 虚拟成员函数。然后我们有一个针对多个类型参数的特化，其中第一个参数是显式的，其余的都在参数包中。我们为显式指定的类型生成 `visit()` 函数，并从具有一个较少参数的相同变长模板的实例化中继承其余的。实例化是递归的，直到我们只剩下一个类型参数，然后使用第一个特化。

这段通用且可重用的代码有一个限制：它不能处理深层层次结构。回想一下，每个可访问的类都派生自一个共同的基类：

```cpp
template <typename Derived>
class Visitable : public Pet {...};
class Cat : public Visitable<Cat> {...};
```

如果我们要从 `Cat` 派生另一个类，它也必须从 `Visitable` 派生：

```cpp
class SiameseCat : public Cat,
                   public Visitable<SiameseCat> {...};
```

我们不能仅仅从`Cat`派生出`SiameseCat`，因为它是提供每个派生类`accept()`方法的`Visitable`模板。但我们也不能像之前尝试的那样使用双重继承，因为现在`SiameseCat`类从`Pet`基类和`Visitable`基类继承两次：一次是通过`Cat`基类，一次是通过`Visitable`基类。如果你仍然想使用模板生成`accept()`方法，唯一的解决方案是将层次结构分开，使得每个可访问类（如`Cat`）都从`Visitable`继承，并从具有所有“猫特定”功能（除了访问支持）的相应基类`CatBase`继承。这会使层次结构中的类数量加倍，这是一个主要的缺点。

现在我们有了由模板生成的样板访问者代码，我们也可以使其定义具体的访问者更加简单。

## Lambda 访问者

定义具体访问者的大部分工作是为每个可访问对象必须发生的实际工作编写代码。在特定的访问者类中并没有很多样板代码。但有时我们可能不想声明这个类本身。想想 lambda 表达式——任何可以用 lambda 表达式完成的事情也可以用显式声明的可调用类完成，因为 lambda 是（匿名）可调用类。尽管如此，我们发现 lambda 表达式对于编写一次性可调用对象非常有用。同样，我们可能想要编写一个没有显式命名的访问者——一个 lambda 访问者。我们希望它看起来像这样：

```cpp
auto v(lambda_visitor<PetVisitor>(
  [](Cat* c) { std::cout << "Let the " << c->color()
                         << " cat out" << std::endl;
  },
  [](Dog* d) { std::cout << "Take the " << d->color()
                         << " dog for a walk" << std::endl;
  }
));
pet->accept(v);
```

有两个问题需要解决——如何创建一个处理类型列表及其相应对象（在我们的例子中，是可访问类型和相应的 lambda）的类，以及如何使用 lambda 表达式生成一组重载函数。

前一个问题将需要我们递归地在参数包上实例化一个模板，每次剥掉一个参数。后一个问题与 lambda 表达式的重载集类似，这在类模板章节中已经描述过。我们可以使用那一章中的重载集，但我们可以使用我们需要的递归模板实例化来直接构建函数的重载集。

在这个实现中，我们将面临一个新的挑战——我们必须处理不止一个类型列表。第一个列表包含所有可访问类型；在我们的例子中，是`Cat`和`Dog`。第二个列表包含 lambda 表达式的类型，每个可访问类型一个。我们还没有看到带有两个参数包的变长模板，而且有很好的理由——不能简单地声明`template<typename... A, typename... B>`，因为编译器不知道第一个列表在哪里结束，第二个在哪里开始。技巧是将一个或两个类型列表隐藏在其他模板中。在我们的例子中，我们已经有了一个`Visitor`模板，它在可访问类型的列表上实例化：

```cpp
using PetVisitor = Visitor<class Cat, class Dog>;
```

我们可以从`Visitor`模板中提取这个列表，并将每个类型与其 lambda 表达式匹配。用于同步处理两个参数包的部分特化语法很棘手，所以我们将分步骤进行。首先，我们需要声明我们的`LambdaVisitor`类的一般模板：

```cpp
// Example 13
template <typename Base, typename...>
class LambdaVisitor;
```

注意，这里只有一个通用参数包，加上访问者的基类（在我们的情况下，它将是`PetVisitor`）。这个模板必须被声明，但它永远不会被使用——我们将为每个需要处理的案例提供一个特化。第一个特化用于只有一个可访问类型和一个相应的 lambda 表达式时：

```cpp
// Example 13
template <typename Base, typename T1, typename F1>
class LambdaVisitor<Base, Visitor<T1>, F1> :
  private F1, public Base
{
  public:
  LambdaVisitor(F1&& f1) : F1(std::move(f1)) {}
  LambdaVisitor(const F1& f1) : F1(f1) {}
  using Base::visit;
  void visit(T1* t) override { return F1::operator()(t); }
};
```

这个专业，除了处理我们只有一个可访问类型的情况外，还用作每条递归模板实例化链中的最后一个实例化。由于它始终是`LambdaVisitor`实例化递归层次结构中的第一个基类，因此它是唯一一个直接继承自基`Visitor`类（如`PetVisitor`）的类。请注意，即使只有一个`T1`可访问类型，我们也使用`Visitor`模板作为其包装器。这是为了准备我们将要处理的一般情况，即我们将有一个长度未知的类型列表。两个构造函数将`f1` lambda 表达式存储在`LambdaVisitor`类内部，如果可能，使用移动而不是复制。最后，`visit(T1*)`虚拟函数覆盖简单地转发调用到 lambda 表达式。乍一看，可能看起来从`F1`公开继承并同意使用函数调用语法（换句话说，将所有对`visit()`的调用重命名为对`operator()`的调用）会更简单。但这行不通；我们需要间接引用，因为 lambda 表达式的`operator()`实例本身不能是虚拟函数覆盖。顺便说一句，这里的`override`关键字在检测模板未从正确的基类继承或虚拟函数声明不完全匹配的代码中的错误时非常有价值。

任何数量可访问类型和 lambda 表达式的一般情况由这个部分特化处理，它明确处理两个列表中的第一个类型，然后递归实例化自身以处理其余的列表：

```cpp
// Example 13
template <typename Base,
          typename T1, typename... T,
          typename F1, typename... F>
class LambdaVisitor<Base, Visitor<T1, T...>, F1, F...> :
  private F1,
  public LambdaVisitor<Base, Visitor<T ...>, F ...>
{
  public:
  LambdaVisitor(F1&& f1, F&& ... f) :
    F1(std::move(f1)),
    LambdaVisitor<Base, Visitor<T...>, F...>(
      std::forward<F>(f)...)
  {}
  LambdaVisitor(const F1& f1, F&& ... f) :
    F1(f1),
    LambdaVisitor<Base, Visitor<T...>, F...>(
      std::forward<F>(f) ...)
  {}
  using LambdaVisitor<Base, Visitor<T ...>, F ...>::visit;
  void visit(T1* t) override { return F1::operator()(t); }
};
```

再次，我们有两个构造函数，将第一个 lambda 表达式存储在类中，并将其余的转发到下一个实例化。在递归的每一步都会生成一个虚拟函数覆盖，始终针对剩余的可访问类列表中的第一个类型。然后，该类型从列表中删除，并以相同的方式继续处理，直到我们达到最后一个实例化，即单个可访问类型的实例化。

由于无法显式命名 lambda 表达式的类型，因此我们也不能显式声明 lambda 访问者的类型。相反，lambda 表达式的类型必须通过模板参数推导来推断，因此我们需要一个接受多个 lambda 表达式参数并从所有这些参数中构建`LambdaVisitor`对象的`lambda_visitor()`模板函数：

```cpp
// Example 13
template <typename Base, typename ... F>
auto lambda_visitor(F&& ... f) {
  return LambdaVisitor<Base, Base, F...>(
    std::forward<F>(f) ...);
}
```

在 C++17 中，可以使用推导指南实现相同的功能。现在我们有一个存储任意数量 lambda 表达式并将每个 lambda 表达式绑定到相应的`visit()`重写的类，我们可以像编写 lambda 表达式一样轻松地编写 lambda 访问者：

```cpp
// Example 13
void walk(Pet& p) {
  auto v(lambda_visitor<PetVisitor>(
  [](Cat* c){std::cout << "Let the " << c->color()
                         << " cat out" << std::endl;},
  [](Dog* d){std::cout << "Take the " << d->color()
                       << " dog for a walk" << std::endl;}
  ));
  p.accept(v);
}
```

注意，由于我们在继承相应 lambda 表达式的同一类中声明了`visit()`函数，因此`lambda_visitor()`函数参数列表中 lambda 表达式的顺序必须与`PetVisitor`定义中类型列表中类的顺序相匹配。如果需要，可以通过增加一些实现复杂性的代价来移除这种限制。

在 C++中处理类型列表的另一种常见方法是将它们存储在`std::tuple`中：例如，我们可以使用`std::tuple<Cat, Dog>`来表示由两种类型组成的列表。同样，整个参数包也可以存储在元组中：

```cpp
// Example 14
template <typename Base, typename F1, typename... F>
class LambdaVisitor<Base, std::tuple<F1, F...>> :
  public F1, public LambdaVisitor<Base, std::tuple<F...>>;
```

您可以将示例 13 和 14 进行比较，以了解如何使用`std::tuple`来存储类型列表。

我们已经看到了如何将访问者代码的常见片段转换为可重用的模板，以及这如何反过来让我们创建 lambda 访问者。但我们没有忘记在本章中学到的另一种访问者实现，即非循环访问者模式。让我们看看它如何也能从现代 C++语言特性中受益。

## 泛型非循环访问者

非循环访问者模式不需要具有所有可访问类型列表的基类。然而，它也有自己的样板代码。首先，每个可访问类型都需要一个`accept()`成员函数，并且它比原始访问者模式中的类似函数有更多的代码：

```cpp
// Example 10
class Cat : public Pet {
  public:
  void accept(PetVisitor& v) override {
    if (CatVisitor* cv = dynamic_cast<CatVisitor*>(&v)) {
      cv->visit(this);
    } else { // Handle error
      assert(false);
    }
  }
};
```

假设错误处理是统一的，这个函数会针对不同的访问者类型重复使用，每个访问者类型对应其可访问类型（例如这里的`CatVisitor`）。然后还有每个类型的访问者类本身，例如：

```cpp
class CatVisitor {
  public:
  virtual void visit(Cat* c) = 0;
};
```

再次，这段代码被粘贴到程序的所有地方，只有细微的修改。让我们将这种容易出错的代码复制粘贴转换为易于维护的可重用代码。

我们首先需要创建一些基础设施。非循环访问者模式以其所有访问者的公共基类为基础构建其层次结构，如下所示：

```cpp
class PetVisitor {
  public:
  virtual ~PetVisitor() {}
};
```

注意，这里没有针对`Pet`层次结构的具体内容。通过更好的命名，这个类可以作为任何访问者层次结构的基类：

```cpp
// Example 15
class VisitorBase {
  public:
  virtual ~VisitorBase() {}
};
```

我们还需要一个模板来生成所有这些针对可访问类型的特定`Visitor`基类，以替换几乎相同的`CatVisitor`、`DogVisitor`等。由于这些类所需的所有内容仅仅是纯虚`visit()`方法的声明，我们可以通过可访问类型来参数化模板：

```cpp
// Example 15
template <typename Visitable> class Visitor {
  public:
  virtual void visit(Visitable* p) = 0;
};
```

对于任何类层次结构，其基可访问类现在使用共同的`VisitorBase`基类接受访问者：

```cpp
// Example 15
class Pet {
  ...
  virtual void accept(VisitorBase& v) = 0;
};
```

我们不再直接从`Pet`派生每个可访问类并粘贴`accept()`方法的副本，而是引入一个中间模板基类，它可以生成具有正确类型的此方法：

```cpp
// Example 15
template <typename Visitable>
class PetVisitable : public Pet {
  public:
  using Pet::Pet;
  void accept(VisitorBase& v) override {
    if (Visitor<Visitable>* pv =
        dynamic_cast<Visitor<Visitable>*>(&v)) {
      pv->visit(static_cast<Visitable*>(this));
    } else { // Handle error
      assert(false);
    }
 }
};
```

这是我们需要编写的`accept()`函数的唯一副本，它包含了我们应用程序处理访问者不被基类接受的情况的首选错误处理实现（回想一下，循环访问者允许部分访问，其中某些访问者和可访问类型的组合不受支持）。就像常规访问者一样，中间的 CRTP 基类使得使用深度层次结构变得困难。

具体的可访问类通过中间的`PetVisitable`基类间接继承自共同的`Pet`基类，该基类还为他们提供了可访问接口。`PetVisitable`模板的参数是派生类本身（再次，我们看到 CRTP 的作用）：

```cpp
// Example 15
class Cat : public PetVisitable<Cat> {
  using PetVisitable<Cat>::PetVisitable;
};
class Dog : public PetVisitable<Dog> {
  using PetVisitable<Dog>::PetVisitable;
};
```

当然，对于所有派生类，使用相同的基类构造函数并不是强制性的，因为每个类都可以根据需要定义自定义构造函数。

剩下的唯一事情是实现访问者类。回想一下，在循环访问者模式中，特定的访问者从共同的访问者基类继承，并且每个代表受支持的可访问类型的访问者类。这不会改变，但现在我们有了按需生成这些访问者类的方法：

```cpp
// Example 15
class FeedingVisitor : public VisitorBase,
                       public Visitor<Cat>,
                       public Visitor<Dog>
{
  public:
  void visit(Cat* c) override {
    std::cout << "Feed tuna to the " << c->color()
              << " cat" << std::endl;
  }
  void visit(Dog* d) override {
    std::cout << "Feed steak to the " << d->color()
              << " dog" << std::endl;
  }
};
```

让我们回顾一下我们所做的工作——访问者类的并行层次结构不再需要显式地指定类型；相反，它们按需生成。重复的`accept()`函数减少到单个`PetVisitable`类模板。尽管如此，我们仍然需要为每个新的可访问类层次结构编写这个模板。我们也可以将其泛化，并为所有层次结构创建一个可重用的模板，该模板通过基可访问类进行参数化：

```cpp
// Example 16
template <typename Base, typename Visitable>
class VisitableBase : public Base {
  public:
  using Base::Base;
  void accept(VisitorBase& vb) override {
    if (Visitor<Visitable>* v = 
        dynamic_cast<Visitor<Visitable>*>(&vb)) {
      v->visit(static_cast<Visitable*>(this));
    } else { // Handle error
      assert(false);
    }
  }
};
```

现在，对于每个可访问类层次结构，我们只需要创建一个模板别名：

```cpp
// Example 16
template <typename Visitable>
using PetVisitable = VisitableBase<Pet, Visitable>;
```

我们可以进一步简化，允许程序员将可访问类的列表指定为类型列表，而不是像之前那样从`Visitor<Cat>`、`Visitor<Dog>`等继承。这需要一个变长模板来存储类型列表。实现与之前看到的`LambdaVisitor`实例类似：

```cpp
// Example 17
template <typename ... V> struct Visitors;
template <typename V1>
struct Visitors<V1> : public Visitor<V1> {};
template <typename V1, typename ... V>
struct Visitors<V1, V ...> : public Visitor<V1>,
                             public Visitors<V ...> {};
```

我们可以使用这个包装模板来缩短特定访问者的声明：

```cpp
// Example 17
class FeedingVisitor :
  public VisitorBase, public Visitors<Cat, Dog>
{
  ...
};
```

如果需要，我们甚至可以将`VisitorBase`隐藏在为单个类型参数定义的`Visitors`模板的定义中。

我们现在已经看到了经典面向对象的访问者模式及其可重用的实现，这些实现是由 C++的泛型编程工具实现的。在早期章节中，我们看到了一些模式可以完全在编译时应用。现在让我们考虑是否也可以用访问者模式做到这一点。

# 编译时访问者

在本节中，我们将分析在编译时使用访问者模式的可行性，类似于应用策略模式导致基于策略的设计。

首先，当在模板上下文中使用时，访问者模式的多个分派方面变得非常简单：

```cpp
template <typename T1, typename T2> auto f(T1 t1, T2 t2);
```

模板函数可以轻松地为`T1`和`T2`类型的任何组合运行不同的算法。与使用虚函数实现的运行时多态不同，根据两个或更多类型的不同调用分发并不需要额外的成本（当然，除了编写处理所有组合所需的代码之外）。基于这个观察，我们可以在编译时轻松地模仿经典的访问者模式：

```cpp
// Example 18
class Pet {
  std::string color_;
  public:
  Pet(std::string_view color) : color_(color) {}
  const std::string& color() const { return color_; }
  template <typename Visitable, typename Visitor>
  static void accept(Visitable& p, Visitor& v) {
    v.visit(p);
  }
};
```

`accept()`函数现在是一个模板和静态成员函数 - 第一个参数的实际类型，即从`Pet`类派生的可访问对象，将在编译时推断出来。具体的可访问类以通常的方式从基类派生：

```cpp
// Example 18
class Cat : public Pet {
  public:
  using Pet::Pet;
};
class Dog : public Pet {
  public:
  using Pet::Pet;
};
```

访问者不需要从公共基类派生，因为我们现在在编译时解析类型：

```cpp
// Example 18
class FeedingVisitor {
  public:
  void visit(Cat& c) {
    std::cout << "Feed tuna to the " << c.color()
              << " cat" << std::endl;
  }
  void visit(Dog& d) {
    std::cout << "Feed steak to the " << d.color()
              << " dog" << std::endl;
  }
};
```

可访问的类可以接受任何具有正确接口的访问者，即对层次结构中所有类都有`visit()`重载：

```cpp
// Example 18
Cat c("orange");
Dog d("brown");
FeedingVisitor fv;
Pet::accept(c, fv);
Pet::accept(d, fv);
```

当然，任何接受访问者参数并需要支持多个访问者的函数也必须是一个模板（仅仅有一个公共基类已经不再足够，它只能帮助在运行时确定实际对象类型）。

编译时访问者解决了经典访问者相同的问题，它允许我们有效地向类添加新成员函数，而无需编辑类定义。然而，它看起来比运行时版本要无趣得多。

当我们将访问者模式与组合模式结合使用时，会出现更多有趣的可能。我们在讨论复杂对象的访问问题时已经这样做过一次，尤其是在序列化问题的背景下。这之所以特别有趣，是因为它与 C++中缺失的少数几个“重要特性”之一——反射——有关。在编程中，反射是指程序检查和内省其自身源代码的能力，然后根据这种内省生成新的行为。一些编程语言，如 Delphi 或 Python，具有原生的反射能力，但 C++没有。反射对于解决许多问题很有用：例如，如果我们能够使编译器遍历对象的所有数据成员并递归地序列化每个成员，直到我们达到内置类型，那么序列化问题就可以轻松解决。我们可以使用编译时访问者模式实现类似的功能。

再次，我们将考虑几何对象的层次结构。由于现在所有事情都在编译时发生，我们对类的多态性质不感兴趣（如果需要运行时操作，它们仍然可以使用虚拟函数；我们只是不会在本节中编写或查看它们）。例如，这是`Point`类：

```cpp
// Example 19
class Point {
  public:
  Point() = default;
  Point(double x, double y) : x_(x), y_(y) {}
  template <typename This, typename Visitor>
  static void accept(This& t, Visitor& v) {
    v.visit(t.x_);
    v.visit(t.y_);
  }
  private:
  double x_ {};
  double y_ {};
};
```

访问是通过`accept()`函数提供的，就像之前一样，但现在它是特定于类的。我们只有一个模板参数`This`的原因是为了方便地支持 const 和非 const 操作：`This`可以是`Point`或`const Point`。任何访问这个类的访问者都会被发送去访问定义点的两个值，`x_`和`y_`。访问者必须具有适当的接口，具体来说，就是接受`double`参数的`visit()`成员函数。像大多数 C++模板库一样，包括`Line`类：

```cpp
// Example 19
class Line {
  public:
  Line() = default;
  Line(Point p1, Point p2) : p1_(p1), p2_(p2) {}
  template <typename This, typename Visitor>
  static void accept(This& t, Visitor& v) {
    v.visit(t.p1_);
    v.visit(t.p2_);
  }
  private:
  Point p1_;
  Point p2_;
};
```

`Line`类由两个点组成。在编译时，访问者被引导访问每个点。这就是`Line`类的参与结束；`Point`类将决定如何被访问（正如我们刚才看到的，它也将工作委托给另一个访问者）。由于我们不再使用运行时多态，现在可以容纳不同类型几何形状的容器类现在必须使用模板：

```cpp
// Example 19
template <typename G1, typename G2>
class Intersection {
  public:
  Intersection() = default;
  Intersection(G1 g1, G2 g2) : g1_(g1), g2_(g2) {}
  template <typename This, typename Visitor>
  static void accept(This& t, Visitor& v) {
    v.visit(t.g1_);
    v.visit(t.g2_);
  }
  private:
  G1 g1_;
  G2 g2_;
};
```

现在我们有了可访问的类型。我们可以使用具有此接口的不同类型的访问者，而不仅仅是序列化访问者。然而，我们现在专注于序列化。之前，我们看到了一个将对象转换为 ASCII 字符串的访问者。现在让我们将对象序列化为二进制数据，连续的位流。序列化访问者可以访问一定大小的缓冲区，并将对象写入该缓冲区，每次写入一个`double`值：

```cpp
// Example 19
class BinarySerializeVisitor {
  public:
  BinarySerializeVisitor(char* buffer, size_t size) :
    buf_(buffer), size_(size) {}
  void visit(double x) {
    if (size_ < sizeof(x))
      throw std::runtime_error("Buffer overflow");
    memcpy(buf_, &x, sizeof(x));
    buf_ += sizeof(x);
    size_ -= sizeof(x);
  }
  template <typename T> void visit(const T& t) {
    T::accept(t, *this);
  }
  private:
  char* buf_;
  size_t size_;
};
```

反序列化访问者从缓冲区读取内存并将其复制到它恢复的对象的数据成员中：

```cpp
// Example 19
class BinaryDeserializeVisitor {
  public:
  BinaryDeserializeVisitor(const char* buffer, size_t size)
    : buf_(buffer), size_(size) {}
  void visit(double& x) {
    if (size_ < sizeof(x))
      throw std::runtime_error("Buffer overflow");
    memcpy(&x, buf_, sizeof(x));
    buf_ += sizeof(x);
    size_ -= sizeof(x);
  }
  template <typename T> void visit(T& t) {
    T::accept(t, *this);
  }
  private:
  const char* buf_;
  size_t size_;
};
```

两个访问者都通过将它们复制到缓冲区并从缓冲区复制来直接处理内置类型，同时让更复杂的类型决定如何处理对象。在这两种情况下，如果超出缓冲区大小，访问者都会抛出异常。现在我们可以使用我们的访问者，例如，将对象通过套接字发送到另一台机器：

```cpp
// Example 19
// On the sender machine:
Line l = ...;
Circle c = ...;
Intersection<Circle, Circle> x = ...;
char buffer[1024];
BinarySerializeVisitor serializer(buffer, sizeof(buffer));
serializer.visit(l);
serializer.visit(c);
serializer.visit(x);
... send the buffer to the receiver ...
// On the receiver machine:
Line l;
Circle c;
Intersection<Circle, Circle> x;
BinaryDeserializeVisitor deserializer(buffer, 
  sizeof(buffer));
deserializer.visit(l);
deserializer.visit(c);
deserializer.visit(x);
```

虽然没有语言支持，我们无法实现通用的反射，但我们可以以有限的方式让类反映其内容，例如这种复合访问模式。我们还可以考虑这个主题的一些变体。

首先，通常的做法是将只有一个**重要**成员函数的对象使其可调用；换句话说，不是调用成员函数，而是使用函数调用语法来调用对象本身。这个约定规定`visit()`成员函数应该被命名为`operator()`：

```cpp
// Example 20
class BinarySerializeVisitor {
  public:
  void operator()(double x);
  template <typename T> void operator()(const T& t);
  ...
};
```

可访问的类现在像函数一样调用访问者：

```cpp
// Example 20
class Point {
  public:
  static void accept(This& t, Visitor& v) {
    v(t.x_);
    v(t.y_);
  }
  ...
};
```

实现包装函数以在多个对象上调用访问者可能也很方便：

```cpp
// Example 20
SomeVisitor v;
Object1 x; Object2 y; ...
visitation(v, x, y, z);
```

这可以通过变长模板轻松实现：

```cpp
// Example 20
template <typename V, typename T>
void visitation(V& v, T& t) {
  v(t);
}
template <typename V, typename T, typename... U>
void visitation(V& v, T& t, U&... u) {
  v(t);
  visitation(v, u ...);
}
```

在 C++17 中，我们有折叠表达式，不需要递归模板：

```cpp
// Example 20
template <typename V, typename T, typename... U>
void visitation(V& v, U&... u) {
  (v(u), ...);
}
```

在 C++14 中，我们可以使用基于`std::initializer_list`的技巧来模拟折叠表达式：

```cpp
template <typename V, typename T, typename... U>
void visitation(V& v, U&... u) {
  using fold = int[];
  (void)fold { 0, (v(u), 0)... };
}
```

这可以工作，但它不太可能因为清晰度或可维护性而获奖。

编译时访问者通常更容易实现，因为我们不需要做任何巧妙的事情来获得多态，因为模板已经提供了这种功能。我们只需要想出有趣的应用模式，比如我们刚刚探索的序列化/反序列化问题。

# C++17 中的访问者

C++17 通过在标准库中添加`std::variant`引入了我们对访问者模式使用方式的重大变化。`std::variant`模板本质上是一个“智能联合体：”`std::variant<T1, T2, T3>`与`union { T1 v1; T2 v2; T3 v3; }`类似，因为它们都可以存储指定类型中的一个值，并且一次只能存储一个值。关键区别在于，变体对象知道它包含哪种类型，而联合体则要求程序员完全负责读取与之前写入相同的类型。将联合体作为与初始化时不同的类型访问是不确定的操作：

```cpp
union { int i; double d; std::string s; } u;
u.i = 0;
++u.i;               // OK
std::cout << u.d;     // Undefined behavior
```

相反，`std::variant`提供了一种安全的方式在相同的内存中存储不同类型的值。在运行时很容易检查当前存储在变体中的是哪种备选类型，如果以错误类型访问变体，则会抛出异常：

```cpp
std::variant<int, double, std::string> v;
std::get<int>(v) = 0;     // Initialized as int
std::cout << v.index();     // 0 is the index of int
++std::get<0>(v);     // OK, int is 0th type
std::get<1>(v);          // throws std::bad_variant_access
```

在许多方面，`std::variant` 提供了类似于基于继承的运行时多态的能力：两者都允许我们编写代码，其中相同的变量名在运行时可以引用不同类型的对象。两个主要区别是：首先，`std::variant` 不要求所有类型都来自同一个层次结构（它们甚至不必是类），其次，变体对象只能存储其声明中列出的类型之一，而基类指针可以指向任何派生类。换句话说，向层次结构添加新类型通常不需要重新编译使用基类的代码，而向变体添加新类型则需要更改变体对象的类型，因此所有引用此对象的代码都必须重新编译。

在本节中，我们将重点关注 `std::variant` 的访问使用。这种能力是由名为 `std::visit` 的函数提供的，它接受一个可调用对象和一个变体：

```cpp
std::variant<int, double, std::string> v;
struct Print {
  void operator()(int i) { std::cout << i; }
  void operator()(double d) { std::cout << d; }
  void operator()(const std::string& s) { std::cout << s; }
} print;
std::visit(print, v);
```

要与 `std::visit` 一起使用，可调用对象必须为变体中可以存储的每种类型声明一个 `operator()`（否则调用将无法编译）。当然，如果实现相似，我们可以在函数对象或 lambda 中使用模板 `operator()`：

```cpp
std::variant<int, double, std::string> v;
std::visit([](const auto& x) { std::cout << x;}, v);
```

我们现在将使用 `std::variant` 和 `std::visit` 重新实现我们的宠物访客。首先，`Pet` 类型不再是层次结构的基类，而是变体，包含所有可能的类型替代项：

```cpp
// Example 21
using Pet = 
  std::variant<class Cat, class Dog, class Lorikeet>;
```

类型本身不需要任何访问机制。我们仍然可以使用继承来重用常见的实现代码，但不需要类型属于单个层次结构：

```cpp
// Example 21
class PetBase {
  public:
  PetBase(std::string_view color) : color_(color) {}
  const std::string& color() const { return color_; }
  private:
  const std::string color_;
};
class Cat : private PetBase {
  public:
  using PetBase::PetBase;
  using PetBase::color;
};
class Dog : private PetBase {
  ... similar to Cat ...
};
class Lorikeet {
  public:
  Lorikeet(std::string_view body, std::string_view head) :
    body_(body), head_(head) {}
  std::string color() const {
    return body_ + " and " + head_;
  }
  private:
  const std::string body_;
  const std::string head_;
};
```

现在我们需要实现一些访问者。访问者只是可调用对象，可以用变体中可能存储的任何替代类型来调用：

```cpp
// Example 21
class FeedingVisitor {
  public:
  void operator()(const Cat& c) {
    std::cout << "Feed tuna to the " << c.color()
              << " cat" << std::endl;
  }
  void operator()(const Dog& d) {
    std::cout << "Feed steak to the " << d.color()
              << " dog" << std::endl;
  }
  void operator()(const Lorikeet& l) {
    std::cout << "Feed grain to the " << l.color()
              << " bird" << std::endl;
  }
};
```

要将访问者应用于变体，我们调用 `std::visit`：

```cpp
// Example 21
Pet p = Cat("orange");
FeedingVisitor v;
std::visit(v, p);
```

变体 `p` 可以包含我们在定义 `Pet` 类型时列出的任何类型（在这个例子中，它是一个 `Cat`）。然后我们调用 `std::visit`，产生的动作既取决于访问者本身，也取决于当前存储在变体中的类型。结果看起来很像虚拟函数调用，因此我们可以说 `std::visit` 允许我们向一组类型添加新的多态函数（由于这些类型不必是类，所以称它们为“虚拟函数”可能会有误导性）。

每当我们看到一个具有用户定义的 `operator()` 的可调用对象时，我们必须在考虑 lambdas。然而，与 `std::visit` 一起使用 lambdas 并不简单：我们需要对象能够以变体中可以存储的任何类型进行调用，而 lambda 只有一个 `operator()`。第一个选项是将该操作符做成模板（多态 lambda）并处理所有可能的类型：

```cpp
// Example 22
#define SAME(v, T) \
  std::is_same_v<std::decay_t<decltype(v)>, T>
auto fv = [](const auto& p) {
  if constexpr (SAME(p, Cat)) {
    std::cout << "Feed tuna to the " << p.color()
              << " cat" << std::endl; }
  else if constexpr (SAME(p, Dog)) {
    std::cout << "Feed steak to the " << p.color()
              << " dog" << std::endl; }
  else if constexpr (SAME(p, Lorikeet)) {
    std::cout << "Feed grain to the " << p.color()
              << " bird" << std::endl; }
  else abort();
};
```

在这里，lambda 可以用任何类型的参数调用，并在 lambda 体内部，我们使用`if constexpr`来处理可以存储在变体中的所有类型。这种方法的缺点是我们不再有编译时验证，即所有可能的类型都被访问者处理。然而，另一方面，如果代码现在没有处理所有类型，代码仍然可以编译，并且只要访问者没有被调用带有我们没有定义操作的类型，程序将正常工作。以这种方式，这个版本类似于无环访问者，而之前的实现类似于常规访问者。

还可以使用 lambda 和我们在*第一章*中看到的创建重载集的技术来实现熟悉的重载`operator()`集：

```cpp
// Example 22
template <typename... T> struct overloaded : T... {
  using T::operator()...;
};
template <typename... T>
overloaded( T...)->overloaded<T...>;
auto pv = overloaded {
  [](const Cat& c) {
    std::cout << "Play with feather with the " << c.color()
              << " cat" << std::endl; },
  [](const Dog& d) {
    std::cout << "Play fetch with the " << d.color()
              << " dog" << std::endl; },
  [](const Lorikeet& l) {
    std::cout << "Teach words to the " << l.color()
              << " bird" << std::endl; }
};
```

这个访问者是一个继承自所有 lambda 的类，并暴露它们的`operator()`，从而创建了一组重载。它就像我们明确写出每个`operator()`的访问者一样使用：

```cpp
// Example 22
Pet l = Lorikeet("yellow", "green");
std::visit(pv, l);
```

到目前为止，我们还没有充分利用`std::visit`的潜力：它可以与任意数量的变体参数一起调用。这允许我们执行依赖于超过两个运行时条件的操作：

```cpp
// Example 23
using Pet = std::variant<class Cat, class Dog>;
Pet c1 = Cat("orange");
Pet c2 = Cat("black");
Pet d = Dog("brown");
CareVisitor cv;
std::visit(cv, c1, c2);      // Two cats
std::visit(cv, c1, d);     // Cat and dog
```

访问者必须以处理每个变体中可以存储的所有类型可能组合的方式编写：

```cpp
class CareVisitor {
  public:
  void operator()(const Cat& c1, const Cat& c2) {
    std::cout << "Let the " << c1.color() << " and the "
              << c2.color() << " cats play" << std::endl; }
  void operator()(const Dog& d, const Cat& c) {
    std::cout << "Keep the " << d.color()
              << " dog safe from the vicious " << c.color()
              << " cat" << std::endl; }
  void operator()(const Cat& c, const Dog& d) {
    (*this)(d, c);
  }
  void operator()(const Dog& d1, const Dog& d2) {
    std::cout << "Take the " << d1.color() << " and the "
              << d2.color() << " dogs for a walk"
              << std::endl; }
};
```

在实践中，唯一可行的方式来编写适用于所有可能类型组合的可调用函数是使用模板`operator()`，这仅在访问者操作可以用通用方式编写时才有效。尽管如此，`std::visit`能够进行多重分派的能力是一个潜在的有用特性，它超越了常规访问者模式的双分派能力。

# 摘要

在本章中，我们学习了访问者模式及其在 C++中的不同实现方式。经典的面向对象访问者模式允许我们在不更改类源代码的情况下，有效地向整个类层次结构添加一个新的虚拟函数。层次结构必须可访问，但在此之后，可以添加任意数量的操作，并且它们的实现与对象本身保持分离。在经典访问者模式的实现中，包含被访问层次结构的源代码不需要更改，但在添加新类到层次结构时，需要重新编译。无环访问者模式解决了这个问题，但代价是额外的动态转换。另一方面，无环访问者模式还支持部分访问 - 忽略一些访问者/可访问组合 - 而经典访问者模式要求所有组合至少被声明。

对于所有访问者变体，可扩展性的权衡是需要弱化封装，并且经常授予外部访问者类访问应该为私有数据成员的权限。

访问者模式通常与其他设计模式结合使用，特别是组合模式，以创建复杂的可访问对象。组合对象将访问委托给其包含的对象。这种组合模式特别有用，当对象必须分解为其最小的构建块时；例如，用于序列化。

经典的访问者模式在运行时实现双重分派 - 在执行过程中，程序根据两个因素选择要运行的代码，即访问者和可访问对象类型。该模式也可以在编译时类似地使用，它提供有限的反射能力。

在 C++17 中，可以使用`std::visit`将访问者模式扩展到未绑定到公共层次结构中的类型，甚至实现多重分派。

本章关于访问者模式，原本是这本书的结尾，这本书致力于 C++惯用和设计模式。但是，就像新星一样，新模式的诞生永远不会停止 - 新的前沿和新思想带来了新的挑战要解决，新的解决方案要发明，它们会不断发展和演变，直到编程社区集体达成共识，我们可以自信地说，*这通常是处理那个问题的好方法*。我们将详细阐述每种新方法的优点，考虑其缺点，并给它起一个名字，这样我们就可以简洁地引用关于该问题、其解决方案及其注意事项的全部知识。有了这个，一个新的模式就进入了我们的设计工具集和编程词汇。为了说明这个过程，在下一章和最后一章中，我们收集了一些出现以解决特定于并发程序的问题的模式。

# 问题

1.  访问者模式是什么？

1.  访问者模式解决了什么问题？

1.  双重分派是什么？

1.  无环访问者模式的优势是什么？

1.  访问者模式如何帮助实现序列化？
