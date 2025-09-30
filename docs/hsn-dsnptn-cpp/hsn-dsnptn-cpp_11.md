

# 第十一章：ScopeGuard

本章介绍了一种模式，它可以被视为我们之前研究的 RAII 语法的泛化。在其最初形式中，它是一个古老且成熟的 C++模式，然而，它也是从 C++11、C++14 和 C++17 的语言添加中特别受益的模式。我们将见证随着语言变得更加强大，这个模式的演变。ScopeGuard 模式存在于声明式编程（说明你想要发生什么，而不是你想要如何实现）和错误安全程序（特别是异常安全）的交叉点。在我们完全理解 ScopeGuard 之前，我们需要了解一些关于这两个方面的知识。

本章将涵盖以下主题：

+   我们如何编写错误安全和异常安全的代码？RAII 如何使错误处理更容易？

+   将可组合性应用于错误处理是什么意思？

+   为什么 RAII 在错误处理方面不够强大，以及它是如何泛化的？我们如何在 C++中实现声明式错误处理？

# 技术要求

这里是示例代码：[`github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/master/Chapter11`](https://github.com/PacktPublishing/Hands-On-Design-Patterns-with-CPP-Second-Edition/tree/master/Chapter11)。

你还需要安装和配置 Google Benchmark 库：[`github.com/google/benchmark`](https://github.com/google/benchmark)（参见*第四章*，*交换 – 从简单到微妙*，获取安装说明）。

本章对高级 C++特性有相当大的依赖，所以请将 C++参考手册放在附近（[`en.cppreference.com`](https://en.cppreference.com)，除非你想直接查阅标准本身）。

最后，在 Folly 库中可以找到一个非常详尽和完整的 ScopeGuard 实现：[`github.com/facebook/folly/blob/master/folly/ScopeGuard.h`](https://github.com/facebook/folly/blob/master/folly/ScopeGuard.h)；它包括本书中未涵盖的 C++库编程细节。

# 错误处理和资源获取即初始化

我们首先回顾错误处理的概念，特别是 C++中编写异常安全代码。**资源获取即初始化**（**RAII**）是 C++中错误处理的主要方法之一。我们已经为它专门写了一整章，你在这里需要它来理解我们即将要做的事情。让我们首先认识到我们面临的问题。

## 错误安全和异常安全

在本章的剩余部分，我们将考虑以下问题——假设我们正在实现一个记录数据库。记录存储在磁盘上，但还有一个内存索引，用于快速访问记录。数据库 API 提供了一个将记录插入数据库的方法：

```cpp
class Record { ... };
class Database {
  public:
  void insert(const Record& r);
  ...
};
```

如果插入成功，索引和磁盘存储都会更新，并且彼此一致。如果出现问题，则会抛出异常。

虽然数据库的客户端看起来插入是一个单一的事务，但实现必须处理这样一个事实：它是通过多个步骤完成的——我们需要将记录插入索引并写入磁盘。为了便于这样做，数据库包含两个类，每个类负责其类型的存储：

```cpp
class Database {
  class Storage { ... };    // Disk storage Storage S;
  class Index { ... };    // Memory index Index I;
  public:
  void insert(const Record& r);
  ...
};
```

`insert()`函数的实现必须将记录插入存储和索引中，没有其他方法可以绕过这一点：

```cpp
//Example 01
void Database::insert(const Record& r) {
  S.insert(r);
  I.insert(r);
}
```

不幸的是，这两种操作都可能失败。让我们首先看看如果存储插入失败会发生什么。假设程序中的所有失败都通过抛出异常来表示。如果存储插入失败，存储保持不变，索引插入根本不会尝试，异常会从`Database::insert()`函数中传播出去。这正是我们想要的——插入失败，数据库保持不变，并抛出异常。

那么，如果存储插入成功但索引失败会发生什么？这次情况看起来并不太好——磁盘被成功更改，然后索引插入失败，异常传播到`Database::insert()`的调用者以表示插入失败，但事实是插入并没有完全失败。它也没有完全成功。

数据库被留在一个不一致的状态；磁盘上有一个记录无法从索引中访问。这是处理错误条件失败、异常不安全的代码，这绝对是不行的。

试图改变子操作顺序的冲动尝试并没有帮助：

```cpp
void Database::insert(const Record& r) {
  I.insert(r);
  S.insert(r);
}
```

当然，如果索引失败，现在一切看起来都很正常。但如果存储插入抛出异常，我们仍然会遇到同样的问题——现在索引中有一个条目指向了无意义的位置，因为记录从未被写入磁盘。

显然，我们不能简单地忽略`Index`或`Storage`抛出的异常；我们必须以某种方式处理它们，以保持数据库的一致性。我们知道如何处理异常；这就是`try-catch`块的作用：

```cpp
// Example 02
void Database::insert(const Record& r) {
  S.insert(r);
  try {
    I.insert(r);
  } catch (...) {
    S.undo();
    throw;    // Rethrow
  }
}
```

再次强调，如果存储失败，我们不需要做任何特殊的事情。如果索引失败，我们必须撤销存储上最后的操作（假设它有执行该操作的 API）。现在数据库再次保持一致性，就像插入从未发生一样。尽管我们捕获了索引抛出的异常，我们仍然需要向调用者发出插入失败的信号，所以我们重新抛出异常。到目前为止，一切顺利。

如果我们选择使用错误代码而不是异常，情况并没有太大的不同；让我们考虑所有`insert()`函数在成功时返回`true`，失败时返回`false`的变体：

```cpp
bool Database::insert(const Record& r) {
  if (!S.insert(r)) return false;
  if (!I.insert(r)) {
    S.undo();
    return false;
  }
  return true;
}
```

我们必须检查每个函数的返回值，如果第二个操作失败则撤销第一个动作，并且只有当两个操作都成功时才返回`true`。

到目前为止，一切顺利；我们能够修复最简单的两阶段问题，因此代码是错误安全的。现在，是时候提高复杂性了。假设我们的存储需要在事务结束时进行一些清理，例如，插入的记录只有在调用`Storage::finalize()`方法后才会处于最终状态（也许这是为了使`Storage::undo()`能够工作，并且在插入最终化后不能再撤销）。注意`undo()`和`finalize()`之间的区别；前者只有在想要回滚事务时才必须调用，而后者必须在存储插入成功后调用，无论之后发生什么。

我们的需求通过以下控制流程得到满足：

```cpp
// Example 02a:
void Database::insert(const Record& r) {
  S.insert(r);
  try {
    I.insert(r);
  } catch (...) {
    S.undo();
    S.finalize();
    throw;
  }
  S.finalize();
}
```

或者，在返回错误代码的情况下（在本章的其余部分，我们将使用异常作为所有示例，但转换为错误代码并不困难）。

这已经变得很丑陋了，尤其是关于获取清理代码（在我们的情况下，`S.finalize()`）以在每条执行路径上运行的部分。如果我们有一个更复杂的动作序列，这些动作都必须撤销，除非整个操作成功，那么情况只会变得更糟。以下是三个动作的控制流程，每个动作都有自己的回滚和清理：

```cpp
if (action1() == SUCCESS) {
  if (action2() == SUCCESS) {
    if (action3() == FAIL) {
      rollback2();
      rollback1();
    }
    cleanup2();
  } else {
    rollback1();
  }
  cleanup1();
}
```

明显的问题是对成功进行的显式测试，无论是作为条件还是作为 try-catch 块。更严重的问题是这种错误处理方式是不可组合的。N+1 个动作的解决方案不是在 N 个动作的代码中添加一些位；不，我们必须深入代码并正确地添加这些部分。但我们已经看到了解决这个问题的 C++惯用法。

## 资源获取即初始化

RAII 惯用法将资源绑定到对象上。当获取资源时对象被构造，当对象被销毁时资源被删除。在我们的情况下，我们只对后半部分感兴趣，即销毁。RAII 惯用法的好处是，当控制达到作用域的末尾时，必须调用所有局部对象的析构函数，无论发生什么情况（`return`、`throw`、`break`等）。由于我们已经与清理作斗争，让我们将清理工作交给一个 RAII 对象：

```cpp
// Example 02b:
class StorageFinalizer {
  public:
  StorageFinalizer(Storage& S) : S_(S) {}
  ~StorageFinalizer() { S_.finalize(); }
  private:
  Storage& S_;
};
void Database::insert(const Record& r) {
  S.insert(r);
  StorageFinalizer SF(S);
  try {
    I.insert(r);
  } catch (...) {
    S.undo();
    throw;
  }
}
```

当`StorageFinalizer`对象被构造时，它会绑定到`Storage`对象并在被销毁时调用`finalize()`方法。由于没有方法可以不调用其析构函数就退出定义`StorageFinalizer`对象的范围，所以我们不需要担心控制流，至少对于清理来说是这样；它将会发生。注意，`StorageFinalizer`在存储插入成功后才会被正确构造；如果第一次插入失败，就没有什么可以最终化的。

这段代码可以工作，但它看起来有些半途而废；我们在函数末尾执行了两个操作，第一个操作（清理或`finalize()`）是自动化的，而第二个操作（回滚或`undo()`）则不是。此外，技术本身仍然不可组合；以下是三个操作的流程控制：

```cpp
class Cleanup1() {
  ~Cleanup1() { cleanup1(); }
  ...
};
class Cleanup2() {
  ~Cleanup2() { cleanup2(); }
  ...
};
action1();
Cleanup1 c1;
try {
  action2();
  Cleanup2 c2;
  try {
    action3();
  } catch (...) {
    rollback2();
    throw;
  }
} catch (...) {
  rollback1();
}
```

再次，为了添加另一个操作，我们必须在代码深处添加一个 try-catch 块。另一方面，清理部分本身是完全可以组合的。考虑如果我们不需要执行回滚，之前的代码看起来会是什么样子：

```cpp
action1();
Cleanup1 c1;
action2();
Cleanup2 c2;
```

如果我们需要执行另一个操作，我们只需在函数末尾添加两行代码，清理工作将按正确顺序进行。如果我们能够对回滚也做同样的事情，我们就可以万事大吉了。

我们不能简单地将对`undo()`的调用移动到另一个对象的析构函数中；析构函数总是被调用，但回滚只有在发生错误时才会发生。但我们可以使析构函数有条件地调用回滚：

```cpp
// Example 03:
class StorageGuard {
  public:
  StorageGuard(Storage& S) : S_(S) {}
  ~StorageGuard() {
    if (!commit_) S_.undo();
  }
  void commit() noexcept { commit_ = true; }
  private:
  Storage& S_;
  bool commit_ = false;
};
void Database::insert(const Record& r) {
  S.insert(r);
  StorageFinalizer SF(S);
  StorageGuard SG(S);
  I.insert(r);
  SG.commit();
}
```

现在检查代码；如果存储插入失败，将抛出异常且数据库保持不变。如果成功，将构造两个 RAII 对象。第一个对象将在作用域结束时无条件调用`S.finalize()`。第二个对象将调用`S.undo()`，除非我们首先通过在`StorageGuard`对象上调用`commit()`方法提交更改。除非索引插入失败，否则会发生这种情况，此时将抛出异常，作用域内的其余代码将被跳过，控制直接跳转到作用域的末尾（即关闭的`}`），在那里调用所有局部对象的析构函数。由于我们在此场景中从未调用`commit()`，因此`StorageGuard`仍然处于活动状态并将撤销插入。注意，这里根本没有任何显式的`try-catch`块：以前在`catch`子句中执行的操作现在由析构函数完成。当然，异常最终应该被捕获（在伴随本章的所有示例中，异常都在`main()`中被捕获）。

本地对象的析构函数按反向构造顺序被调用。这很重要；如果我们必须撤销插入，这只能在操作最终化之前完成，因此回滚必须在清理之前发生。因此，我们按正确的顺序构造 RAII 对象——首先，清理（最后执行），然后是回滚保护（如果需要，首先执行）。

代码现在看起来非常好，完全没有 try-catch 块。在某种程度上，它看起来不像常规的 C++。这种编程风格被称为**声明式编程**；它是一种编程范式，其中程序逻辑通过不明确声明控制流（与 C++中更常见的相反，即**命令式编程**，其中程序描述了要执行哪些步骤以及它们的顺序，但不一定说明为什么）来表达。有声明式编程语言（主要例子是 SQL），但 C++不是其中之一。尽管如此，C++非常擅长实现允许在 C++之上创建高级语言的构造，因此我们实现了一种声明式错误处理语言。我们的程序现在表示，在记录被插入存储后，有两个待执行的动作——清理和回滚。

如果整个函数执行成功，则回滚将被解除。代码看起来是线性的，没有显式的控制流，换句话说，是声明式的。

虽然很好，但它也远非完美。明显的问题是，我们必须为程序中的每个动作编写一个保护器或最终化器类。不那么明显的问题是，正确编写这些类并不容易，我们到目前为止做得并不特别出色。在查看这里修复的版本之前，花点时间想一下缺少了什么：

```cpp
class StorageGuard {
  public:
  StorageGuard(Storage& S) : S_(S), commit_(false) {}
  ~StorageGuard() { if (!commit_) S_.undo(); }
  void commit() noexcept { commit_ = true; }
  private:
  Storage& S_;
  bool commit_;
  // Important: really bad things happen if
  // this guard is copied!
  StorageGuard(const StorageGuard&) = delete;
  StorageGuard& operator=(const StorageGuard&) = delete;
};
void Database::insert(const Record& r) {
  S.insert(r);
  StorageFinalizer SF(S);
  StorageGuard SG(S);
  I.insert(r);
  SG.commit();
}
```

我们需要一个通用框架，让我们能够安排在作用域结束时无条件或条件性地执行任意操作。下一节将介绍提供这种框架的模式，即范围保护。

# 范围保护模式

在本节中，我们学习如何编写退出时执行的动作 RAII 类，就像我们在上一节中实现的那样，但不需要所有样板代码。这可以在 C++03 中完成，但在 C++14 中得到了很大的改进，然后在 C++17 中再次改进。

## 范围保护基础

让我们从更困难的问题开始——如何实现一个通用的回滚类，即上一节中`StorageGuard`的通用版本。与清理类之间的唯一区别是，清理始终处于活动状态，但回滚在操作提交后取消。如果我们有条件回滚版本，我们可以总是去掉条件检查，从而得到清理版本，所以我们现在不必担心这个问题。

在我们的例子中，回滚是一个调用`S.undo()`方法的操作。为了简化示例，让我们从一个调用常规函数而不是成员函数的回滚开始：

```cpp
void undo(Storage& S) { S.undo(); }
```

一旦实施完成，程序应该看起来像这样：

```cpp
{
  S.insert(r);
  ScopeGuard SG(undo, S);    // Approximate desired syntax
  ...
  SG.commit();            // Disarm the scope guard
}
```

这段代码以声明式的方式告诉我们，如果插入操作成功，我们将在退出作用域时安排回滚操作。回滚将调用带有参数`S`的`undo()`函数，这反过来将撤销插入操作。如果我们到达了函数的末尾，我们将解除保护并禁用回滚调用，这将提交插入操作并使其永久化。

Andrei Alexandrescu 在 2000 年的 *Dr. Dobbs* 文章中提出了一种更通用且可重用的解决方案（[`www.drdobbs.com/cpp/generic-change-the-way-you-write-excepti/184403758?`](http://www.drdobbs.com/cpp/generic-change-the-way-you-write-excepti/184403758?)）。让我们看看实现并分析它：

```cpp
// Example 04
class ScopeGuardImplBase {
  public:
  ScopeGuardImplBase() = default;
  void commit() const noexcept { commit_ = true; }
  protected:
  ScopeGuardImplBase(const ScopeGuardImplBase& other) :
    commit_(other.commit_) { other.commit(); }
  ~ScopeGuardImplBase() {}
  mutable bool commit_ = false;
};
template <typename Func, typename Arg>
class ScopeGuardImpl : public ScopeGuardImplBase {
  public:
  ScopeGuardImpl(const Func& func, Arg& arg) :
    func_(func), arg_(arg) {}
  ~ScopeGuardImpl() { if (!commit_) func_(arg_); }
  private:
  const Func& func_;
  Arg& arg_;
};
template <typename Func, typename Arg>
ScopeGuardImpl<Func, Arg>
MakeGuard(const Func& func, Arg& arg) {
  return ScopeGuardImpl<Func, Arg>(func, arg);
}
```

从顶部开始，我们有所有作用域保护器的基类，`ScopeGuardImplBase`。基类持有提交标志和操作它的代码；构造函数最初将保护器创建在武装状态，因此延迟操作将在析构函数中发生。对`commit()`的调用将阻止这种情况发生，并使析构函数不执行任何操作。最后，还有一个拷贝构造函数，它创建一个与原始对象状态相同的新保护器，然后解除原始保护器的武装。这样做是为了防止回滚在两个对象的析构函数中发生两次。对象是可拷贝的但不可赋值。我们在这里使用 C++03 的所有功能，包括禁用的赋值运算符。这个实现本质上就是 C++03；少数 C++11 的变化只是锦上添花（这将在下一节中改变）。

`ScopeGuardImplBase` 实现的几个细节可能看起来很奇怪，需要详细说明。首先，析构函数不是虚拟的；这不是一个打字错误或错误，这是故意的，就像我们稍后将要看到的那样。其次，`commit_`标志被声明为`mutable`。当然，这是为了让它可以通过我们声明的`const`方法`commit()`被改变。那么，为什么`commit()`被声明为`const`呢？一个原因是我们可以从拷贝构造函数中调用它，将回滚的责任从另一个对象转移到这个对象。从这个意义上说，拷贝构造函数实际上执行了一个移动操作，稍后将被正式声明为这样的操作。`const`声明的第二个原因稍后就会变得明显（它与非虚拟析构函数有关）。

现在，让我们转向派生类，`ScopeGuardImpl`。这是一个带有两个类型参数的类模板——第一个是我们将要调用的用于回滚的函数或任何其他可调用对象的类型，第二个是参数的类型。目前，我们的回滚函数仅限于只有一个参数。这个函数在`ScopeGuard`对象的析构函数中被调用，除非通过调用`commit()`解除保护。

最后，我们有一个工厂函数模板，`MakeGuard`。这在 C++中是一个非常常见的惯用法；如果你需要从构造函数参数创建一个类模板的实例，请使用一个模板函数，它可以从参数推导出参数类型和返回值类型（在 C++17 中，类模板也可以这样做，我们稍后会看到）。

这些是如何用来创建一个将为我们调用`undo(S)`的守卫对象的？如下所示：

```cpp
void Database::insert(const Record& r) {
  S.insert(r);
  const ScopeGuardImplBase& SG = MakeGuard(undo, S);
  I.insert(r);
  SG.commit();
}
```

`MakeGuard`函数推导出`undo()`函数的类型和参数`S`的类型，并返回相应类型的`ScopeGuard`对象。返回是通过值进行的，因此涉及到一个复制（编译器可能会选择省略复制作为优化，但不是必须的）。返回的对象是一个临时变量，它没有名字，并将其绑定到基类引用`SG`（从派生类到基类的指针和引用的转换是隐式的）。临时变量的生命周期直到创建它的语句的结束分号，正如大家所知。但是，在语句结束后，`SG`引用指向什么？它必须绑定到某个东西，因为引用不能解绑，它们不像`NULL`指针。事实是，“每个人”都知道错了，或者说只是大部分正确——通常，临时变量确实会一直活到语句的结束。然而，将临时变量绑定到`const`引用会将其生命周期延长到与引用本身的生存期一致。换句话说，`MakeGuard`创建的无名`ScopeGuard`对象将不会在`SG`引用超出作用域之前被销毁。在这里，`const`属性很重要，但不必担心，你不会忘记它；语言不允许将非`const`引用绑定到临时变量，因此编译器会告诉你。因此，这就解释了`commit()`方法；它必须是`const`的，因为我们将在`const`引用上调用它（因此，`commit_`标志必须是`mutable`）。

但是，关于析构函数呢？在作用域结束时，`ScopeGuardImplBase`类的析构函数将被调用，因为这是超出作用域的引用类型。基类析构函数本身不执行任何操作，执行我们想要的析构函数的是派生类。一个具有虚拟析构函数的多态类会为我们提供正确的服务，但我们没有走这条路。相反，我们利用了 C++标准中关于`const`引用和临时变量的另一条特殊规则——不仅临时变量的生命周期被延长，而且派生类的析构函数，即实际构造的类，将在作用域结束时被调用。

注意，这条规则仅适用于析构函数；你仍然不能在基类`SG`引用上调用派生类方法。此外，生命周期扩展仅在临时变量直接绑定到`const`引用时才有效。如果，例如，我们从第一个引用初始化另一个`const`引用，则不会生效。这就是为什么我们必须通过值从`MakeGuard`函数返回`ScopeGuard`对象；如果我们尝试通过引用返回它，临时变量将绑定到那个引用，而这个引用将在语句结束时消失。第二个引用`SG`是从第一个引用初始化的，它并没有扩展对象的生命周期。

我们刚才看到的函数实现非常接近原始目标，只是稍微有点冗长（并且提到了`ScopeGuardImplBase`而不是承诺的`ScopeGuard`）。不用担心，最后一步仅仅是语法糖：

```cpp
using ScopeGuard = const ScopeGuardImplBase&;
```

现在，我们可以这样写：

```cpp
// Example 04a
void Database::insert(const Record& r) {
  S.insert(r);
  ScopeGuard SG = MakeGuard(undo, S);
  I.insert(r);
  SG.commit();
}
```

这就是我们迄今为止使用语言工具所能达到的。理想情况下，所需的语法应该是这样的（而且我们并不遥远）：（此处省略具体语法示例）

```cpp
ScopeGuard SG(undo, S);
```

我们可以利用 C++11 的特性来稍微整理一下我们的`ScopeGuard`。首先，我们可以正确地禁用赋值运算符。其次，我们可以停止假装我们的复制构造函数除了移动构造函数之外还有什么：

```cpp
// Example 05
class ScopeGuardImplBase {
  public:
  ScopeGuardImplBase() = default;
  void commit() const noexcept { commit_ = true; }
  protected:
  ScopeGuardImplBase(ScopeGuardImplBase&& other) :
    commit_(other.commit_) { other.commit(); }
  ~ScopeGuardImplBase() {}
  mutable bool commit_ = false;
  private:
  ScopeGuardImplBase& operator=(const ScopeGuardImplBase&)
    = delete;
};
using ScopeGuard = const ScopeGuardImplBase&;
template <typename Func, typename Arg>
class ScopeGuardImpl : public ScopeGuardImplBase {
  public:
  ScopeGuardImpl(const Func& func, Arg& arg) :
    func_(func), arg_(arg) {}
  ~ScopeGuardImpl() { if (!commit_) func_(arg_); }
  ScopeGuardImpl(ScopeGuardImpl&& other) :
    ScopeGuardImplBase(std::move(other)),
    func_(other.func_),
    arg_(other.arg_) {}
  private:
  const Func& func_;
  Arg& arg_;
};
template <typename Func, typename Arg>
ScopeGuardImpl<Func, Arg>
MakeGuard(const Func& func, Arg& arg) {
  return ScopeGuardImpl<Func, Arg>(func, arg);
}
```

转向 C++14，我们可以进一步简化，并推断出`MakeGuard`函数的返回类型：

```cpp
// Example 05a
template <typename Func, typename Arg>
auto MakeGuard(const Func& func, Arg& arg) {
  return ScopeGuardImpl<Func, Arg>(func, arg);
}
```

我们还必须做出一项让步——我们实际上并不需要`undo(S)`函数，我们真正想要的是调用`S.undo()`。这可以通过`ScopeGuard`的成员函数变体轻松实现。事实上，我们之所以从一开始就没有这样做，只是为了使示例更容易理解；成员函数指针语法并不是 C++中最直接的部分：

```cpp
// Example 06
template <typename MemFunc, typename Obj>
class ScopeGuardImpl : public ScopeGuardImplBase {
  public:
  ScopeGuardImpl(const MemFunc& memfunc, Obj& obj) :
    memfunc_(memfunc), obj_(obj) {}
  ~ScopeGuardImpl() { if (!commit_) (obj_.*memfunc_)(); }
  ScopeGuardImpl(ScopeGuardImpl&& other) :
    ScopeGuardImplBase(std::move(other)),
    memfunc_(other.memfunc_),
    obj_(other.obj_) {}
  private:
  const MemFunc& memfunc_; Obj& obj_;
};
template <typename MemFunc, typename Obj>
auto MakeGuard(const MemFunc& memfunc, Obj& obj) {// C++14
  return ScopeGuardImpl<MemFunc, Obj>(memfunc, obj);
}
```

当然，如果在同一个程序中使用了`ScopeGuard`模板的两个版本，我们必须重命名其中一个。此外，我们的函数守卫只能调用只有一个参数的函数，而我们的成员函数守卫只能调用没有参数的成员函数。在 C++03 中，这个问题通过一种繁琐但可靠的方式得到解决——我们必须为具有零、一、二等参数的函数创建实现版本，例如`ScopeGuardImpl0`、`ScopeGuardImp1`、`ScopeGuardImpl2`等。然后，我们为具有零、一、二等参数的成员函数创建`ScopeObjGuardImpl0`、`ScopeObjGuardImpl1`等。如果我们没有创建足够的版本，编译器会告诉我们。所有这些派生类变体都有相同的基类，`ScopeGuard`的`typedef`也是如此。

在 C++11 中，我们有变长模板，旨在解决这个确切的问题，但在这里我们不会看到这样的实现。没有必要；我们可以做得更好，正如你即将看到的。

## 泛型 ScopeGuard

我们现在完全处于 C++11 的领域，你即将看到的任何内容都没有 C++03 的等效物，具有任何实际价值。

到目前为止，我们的`ScopeGuard`允许我们将任意函数作为任何操作的回滚。就像手工制作的守卫对象一样，作用域守卫是可组合的，并保证异常安全。但到目前为止，我们的实现对我们可以调用来实现回滚的功能有限；它必须是一个函数或成员函数。虽然这似乎涵盖了大部分，我们可能还想调用，例如，两个函数来完成单个回滚。我们当然可以为此编写一个包装函数，但这又让我们回到了单用途手工回滚对象的道路上。

实际上，我们的实现中还存在另一个问题。我们决定通过引用捕获函数参数：

```cpp
ScopeGuardImpl(const Func& func, Arg& arg);
```

这通常都有效，除非参数是一个常量或临时变量；那么，我们的代码将无法编译。

C++11 给我们提供了创建任意可调用对象的另一种方法：lambda 表达式。Lambda 实际上是类，但它们的行为像函数，因为它们可以用括号调用。它们可以接受参数，但也可以捕获包含作用域中的任何参数，这通常消除了传递参数给函数调用的需要。我们还可以编写任意代码，并将其打包在 lambda 表达式中。这听起来对作用域守卫来说很理想；我们只需编写一些代码，说“在作用域运行结束时”执行这些代码。

让我们看看 lambda 表达式作用域守卫的样子：

```cpp
// Example 07
class ScopeGuardBase {
  public:
  ScopeGuardBase() = default;
  void commit() noexcept { commit_ = true; }
  protected:
  ScopeGuardBase(ScopeGuardBase&& other) noexcept :
    commit_(other.commit_) { other.commit(); }
  ~ScopeGuardBase() = default;
  bool commit_ = false;
  private:
  ScopeGuardBase& operator=(const ScopeGuardBase&)
    = delete;
};
template <typename Func>
class ScopeGuard : public ScopeGuardBase {
  public:
  ScopeGuard(Func&& func) : func_(std::move(func)) {}
  ScopeGuard(const Func& func) : func_(func) {}
  ~ScopeGuard() { if (!commit_) func_(); }
  ScopeGuard(ScopeGuard&& other) = default;
  private:
  Func func_;
};
template <typename Func>
ScopeGuard<Func> MakeGuard(Func&& func) {
  return ScopeGuard<Func>(std::forward<Func>(func));
}
```

基类基本上与之前相同，只是我们不再使用`const`引用技巧，因此`Impl`后缀已经消失；你所看到的不再是实现辅助，而是守卫类的本身基础；它包含处理`commit_`标志的可重用代码。由于我们不使用`const`引用，我们可以停止假装`commit()`方法是`const`的，并从`commit_`中删除`mutable`声明。

另一方面，派生类有很大的不同。首先，只有一个类用于所有类型的回滚，参数类型已经消失；相反，我们有一个将要成为 lambda 的功能对象，它将包含它需要的所有参数。析构函数与之前相同（除了缺少对可调用`func_`的参数），移动构造函数也是如此。但对象的主体构造函数相当不同；可调用对象按值存储，并从`const`引用或右值引用初始化，编译器会自动选择合适的重载。

`MakeGuard`函数基本上没有变化，我们不需要两个；我们可以使用完美转发（`std::forward`）将任何类型的参数转发到`ScopeGuard`的一个构造函数。

下面是如何使用这个作用域守卫的示例：

```cpp
void Database::insert(const Record& r) {
  S.insert(r);
  auto SG = MakeGuard([&] { S.undo(); });
  I.insert(r);
  SG.commit();
}
```

用作`MakeGuard`参数的标点符号丰富的结构是 lambda 表达式。它创建了一个可调用对象，调用此对象将运行 lambda 体内的代码，在我们的例子中是`S.undo()`。在 lambda 对象本身中没有声明`S`变量，因此它必须从包含的作用域中捕获。所有捕获都是通过引用（`[&]`）完成的。最后，对象被不带参数地调用；括号可以省略，尽管`MakeGuard([&]() { S.undo(); });`也是有效的。该函数不返回任何内容，即返回类型是`void`；不需要显式声明。请注意，到目前为止，我们使用了 C++11 lambda，并没有利用更强大的 C++14 lambda。这通常是 ScopeGuard 的情况，尽管在实践中，您可能会仅为了自动推导的返回类型而使用 C++14，如果其他什么也不做的话。

一直以来，我们有意将常规清理问题放在一边，专注于错误处理和回滚。现在我们有了相当不错的 ScopeGuard，我们可以轻松地解决悬而未决的问题：

```cpp
// Example 07a
void Database::insert(const Record& r) {
  S.insert(r);
  auto SF = MakeGuard([&] { S.finalize(); });
  auto SG = MakeGuard([&] { S.undo(); });
  I.insert(r);
  SG.commit();
}
```

如您所见，我们不需要在我们的框架中添加任何特殊的东西来支持清理。我们只需创建另一个我们永远不会解除武装的 ScopeGuard。

我们还应该指出，在 C++17 中，我们不再需要`MakeGuard`函数，因为编译器可以从构造函数中推导出模板参数：

```cpp
// Example 07b
void Database::insert(const Record& r) {
  S.insert(r);
  ScopeGuard SF = [&] { S.finalize(); };    // C++17
  ScopeGuard SG = [&] { S.undo(); };
  I.insert(r);
  SG.commit();
}
```

既然我们在讨论使 ScopeGuard 使用起来更美观的话题，我们应该考虑一些有用的宏。我们可以轻松地为清理守卫编写一个宏，即总是执行的那个。我们希望生成的语法看起来像这样（如果这还不够声明性，我不知道还有什么）：

```cpp
ON_SCOPE_EXIT { S.finalize(); };
```

实际上，我们可以获取那个非常具体的语法。首先，我们需要为守卫生成一个名称，过去被称为`SF`，并且我们需要它是一个独一无二的名称。从现代 C++的尖端，我们追溯到几十年前的经典 C 及其预处理器技巧，以生成一个匿名变量的唯一名称：

```cpp
#define CONCAT2(x, y) x##y
#define CONCAT(x, y) CONCAT2(x, y)
#ifdef __COUNTER__
#define ANON_VAR(x) CONCAT(x, __COUNTER__)
#else
#define ANON_VAR(x) CONCAT(x, __LINE__)
#endif
```

`__CONCAT__`宏是在预处理器中将两个标记连接起来的方法（是的，你需要两个，这就是预处理器的工作方式）。第一个标记将是一个用户指定的前缀，第二个是一个独一无二的标记。许多编译器支持一个预处理器变量`__COUNTER__`，每次使用时都会递增，所以它永远不会相同。然而，它不在标准中。如果`__COUNTER__`不可用，我们必须使用行号`__LINE__`作为唯一标识符。当然，只有在我们没有在同一行上放置两个守卫的情况下，它才是唯一的，所以不要这样做。

现在我们有了生成匿名变量名的方法，我们可以实现 `ON_SCOPE_EXIT` 宏。实现一个将代码作为宏参数传递的宏是微不足道的，但它不会给我们想要的语法；参数必须放在括号内，所以我们最多只能得到 `ON_SCOPE_EXIT(S.finalize();)`。此外，代码中的逗号会混淆预处理器，因为它将其解释为宏参数之间的分隔符。如果你仔细观察我们请求的语法 `ON_SCOPE_EXIT { S.finalize(); };`，你会意识到这个宏根本没有任何参数，lambda 表达式的主体只是类型化在无参数宏之后。因此，宏展开在可以跟随开括号的东西上结束。以下是它是如何完成的：

```cpp
// Example 08
struct ScopeGuardOnExit {};
template <typename Func>
ScopeGuard<Func> operator+(ScopeGuardOnExit, Func&& func) {
  return ScopeGuard<Func>(std::forward<Func>(func));
}
#define ON_SCOPE_EXIT auto ANON_VAR(SCOPE_EXIT_STATE) = \
  ScopeGuardOnExit() + [&]()
```

宏展开声明了一个以 `SCOPE_EXIT_STATE` 开头的匿名变量，后面跟着一个唯一的数字，并在不完整的 lambda 表达式 `[&]()` 上结束，该表达式由花括号中的代码完成。为了不产生前一个 `MakeGuard` 函数的闭括号（宏无法生成，因为宏在 lambda 体之前展开，所以它不能生成任何代码），我们必须将 `MakeGuard` 函数（或 C++17 中的 `ScopeGuard` 构造函数）替换为一个操作符。操作符的选择并不重要；我们使用 `+`，但也可以使用任何二元操作符。操作符的第一个参数是一个唯一的临时对象，它仅限于之前定义的 `operator+()` 的重载解析（该对象本身根本不使用，我们只需要它的类型）。`operator+()` 本身就是 `MakeGuard` 之前所用的，它推断 lambda 表达式的类型并创建相应的 `ScopeGuard` 对象。这种技术的唯一缺点是，在 `ON_SCOPE_EXIT` 语句的末尾需要一个分号，如果你忘记了，编译器将以最隐晦和模糊的方式提醒你。

我们的程序代码现在可以进一步整理：

```cpp
// Example 08
void Database::insert(const Record& r) {
  S.insert(r);
  ON_SCOPE_EXIT { S.finalize(); };
  auto SG = ScopeGuard([&] { S.undo(); });
  I.insert(r);
  SG.commit();
}
```

很有诱惑力将同样的技术应用到第二个守卫上。不幸的是，这并不简单；我们必须知道这个变量的名字，以便我们可以调用其上的 `commit()` 方法。我们可以定义一个类似的宏，它不使用匿名变量，而是使用用户指定的名字：

```cpp
// Example 08a
#define ON_SCOPE_EXIT_ROLLBACK(NAME) \
  auto NAME = ScopeGuardOnExit() + [&]()
```

我们可以用它来完成我们代码的转换：

```cpp
// Example 08a
void Database::insert(const Record& r) {
  S.insert(r);
  ON_SCOPE_EXIT { S.finalize(); };
  ON_SCOPE_EXIT_ROLLBACK(SG){ S.undo(); };
  I.insert(r);
  SG.commit();
}
```

到目前为止，我们应该重新审视可组合性的问题。对于三个动作，每个都有自己的回滚和清理，我们现在有以下内容：

```cpp
action1();
ON_SCOPE_EXIT { cleanup1; };
ON_SCOPE_EXIT_ROLLBACK(g2){ rollback1(); };
action2();
ON_SCOPE_EXIT { cleanup2; };
ON_SCOPE_EXIT_ROLLBACK(g4){ rollback2(); };
action3();
g2.commit();
g4.commit();
```

可以看到，这种模式可以轻易扩展到任意数量的动作。一个细心的读者可能会怀疑他们是否在代码中发现了错误——回滚保护不应该按照反向构造顺序取消吗？这不是一个错误，尽管所有`commit()`调用的反向顺序也不是错误。原因是`commit()`不能抛出异常，它被声明为`noexcept`，而且确实其实现是这样的，不能抛出异常。这对范围保护模式的工作至关重要；如果`commit()`可以抛出异常，那么就无法保证所有回滚保护都被正确解除。在作用域结束时，一些动作会被回滚，而其他动作则不会，这将使系统处于不一致的状态。

虽然范围保护主要设计用来使编写异常安全代码更容易，但范围保护模式与异常的交互远非简单，我们应该花更多时间来关注它。

# 范围保护与异常

范围保护模式旨在在退出作用域时自动正确执行各种清理和回滚操作，无论退出原因是什么——正常完成作用域的末尾、提前返回或异常。这使得编写错误安全代码变得容易，尤其是异常安全代码；只要我们在每次动作之后都排好正确的保护程序，正确的清理和错误处理就会自动发生。当然，这是假设范围保护在异常存在的情况下本身运行正确。我们将学习如何确保它确实如此，以及如何使用它来使其余代码错误安全。

## 什么不能抛出异常

我们已经看到，用于提交动作并解除回滚保护的`commit()`函数绝对不能抛出异常。幸运的是，这很容易保证，因为这个函数所做的只是设置一个标志。但如果回滚函数也失败了，并抛出了异常怎么办？

```cpp
// Example 09
void Database::insert(const Record& r) {
  S.insert(r);
  auto SF = MakeGuard([&] { S.finalize(); });
  auto SG = MakeGuard([&] { S.undo(); });
             // What if undo() can throw?
  I.insert(r);    // Let's say this fails
  SG.commit();    // Commit never happens
}            // Control jumps here and undo() throws
```

简短的回答是*没有好办法*。一般来说，我们面临一个难题——我们既不能允许动作（在我们的例子中是存储插入）继续进行，也不能撤销它，因为那样也会失败。具体来说，在 C++中，两个异常不能同时传播。因此，析构函数不允许抛出异常；当抛出异常时可能会调用析构函数，如果该析构函数也抛出异常，那么我们现在有两个异常同时传播。如果发生这种情况，程序将立即终止。这与其说是语言的不足，不如说是对一般问题不可解性的反映；我们无法让事情保持原样，但我们也未能成功改变某些东西。已经没有好的选择了。

通常，C++程序有三种处理这种情况的方法。最好的选择是不陷入这个陷阱——如果回滚不能抛出异常，这一切都不会发生。因此，一个编写良好的异常安全程序会竭尽全力提供非抛出回滚和清理。例如，主要动作可以生成新数据并使其就绪，然后只需交换一个指针，就可以简单地将数据提供给调用者，这肯定是一个非抛出操作。回滚只涉及交换指针回原位，可能还需要删除某些东西（正如我们之前所说的，析构函数不应该抛出异常；如果它们这样做，程序的行为是未定义的）。

第二种选择是在回滚中抑制异常。我们尝试撤销操作，但失败了，我们对此无能为力，所以让我们继续前进。这里的危险是程序可能处于未定义的状态，从这一点开始的所有操作都可能是不正确的。然而，这只是一个最坏的情况。在实践中，后果可能不那么严重。例如，对于我们的数据库，我们可能知道如果回滚失败，有一块磁盘空间被记录占用，但无法从索引中访问。调用者将正确地被告知插入失败，但我们浪费了一些磁盘空间。这可能比直接终止程序更好。如果我们希望这样做，我们必须捕获任何可能由 ScopeGuard 操作抛出的异常：

```cpp
// Example 09a
template <typename Func>
class ScopeGuard : public ScopeGuardBase {
  public:
  ...
  ~ScopeGuard() {
    if (!commit_) try { func_(); } catch (...) {}
  }
  ...
};
```

`catch`子句是空的；我们捕获了一切，但什么也没做。这种实现有时被称为*屏蔽的 ScopeGuard*。

最后一种选择是允许程序失败。如果我们只是让两个异常发生，那么这不需要我们做出任何努力就会发生。但我们可以打印一条消息或以其他方式向用户发出即将发生的事情以及原因的信号。如果我们想在程序终止之前插入自己的死亡动作，我们必须编写与之前非常相似的代码：

```cpp
template <typename Func>
class ScopeGuard : public ScopeGuardBase {
  public:
  ...
  ~ScopeGuard() {
    if (!commit_) try { func_(); } catch (...) {
      std::cout << "Rollback failed" << std::endl;
      throw;    // Rethrow
    }
  }
  ...
};
```

关键的区别是没有参数的`throw;`语句。这重新抛出了我们捕获的异常，并允许它继续传播。

最后两个代码片段之间的区别突显了一个我们之前略过的微妙细节，但这个细节在以后会变得很重要。说在 C++中析构函数不应该抛出异常是不准确的。正确的说法是，异常不应该从析构函数中传播出来。只要析构函数也捕获它，它可以抛出任何它想要的东西：

```cpp
class LivingDangerously {
  public:
  ~LivingDangerously() {
    try {
      if (cleanup() != SUCCESS) throw 0;
       more_cleanup();
    } catch (...) {
      std::cout << "Cleanup failed, proceeding anyway" <<
      std::endl;
      // No rethrow - this is critical!
    }
  }
};
```

到目前为止，我们主要将异常视为一种麻烦；如果某个地方抛出了异常，程序必须保持一个良好的定义状态，但除此之外，我们没有使用这些异常；我们只是将它们传递下去。另一方面，我们的代码可以与任何类型的错误处理一起工作，无论是异常还是错误代码。如果我们确信错误总是通过异常来表示，并且任何非抛出异常的函数返回都是成功，我们可以利用这一点来自动检测成功或失败，从而允许根据需要发生提交或回滚。

## 异常驱动的 ScopeGuard

现在，我们将假设如果一个函数没有抛出异常就返回，那么操作已经成功。如果函数抛出异常，显然失败了。现在的目标是取消对`commit()`的显式调用，而是检测 ScopeGuard 的析构函数是因抛出异常而执行，还是因为函数正常返回。

这个实现有两个部分。第一部分是指定我们希望在何时执行操作。清理守卫必须无论以何种方式退出作用域都要执行。回滚守卫仅在失败的情况下执行。为了完整性，我们还可以有一个仅在函数成功时执行的守卫。第二部分是确定实际上发生了什么。

我们将从第二部分开始。我们的 ScopeGuard 现在需要两个额外的参数，这两个参数将告诉我们是否应该在成功时执行，以及是否应该在失败时执行（两者可以同时启用）。只需要修改 ScopeGuard 的析构函数：

```cpp
template <typename Func, bool on_success, bool on_failure>
class ScopeGuard {
  public:
  ...
  ~ScopeGuard() {
    if ((on_success && is_success()) ||
        (on_failure && is_failure())) func_();
  }
  ...
};
```

我们仍然需要弄清楚如何实现伪函数`is_success()`和`is_failure()`。记住，失败意味着抛出了异常。在 C++中，我们有一个函数可以做到这一点：`std::uncaught_exception()`。如果当前正在传播异常，它返回 true，否则返回 false。有了这个知识，我们可以实现我们的守卫：

```cpp
// Example 10
template <typename Func, bool on_success, bool on_failure>
class ScopeGuard {
  public:
  ...
  ~ScopeGuard() {
    if ((on_success && !std::uncaught_exception()) ||
        (on_failure && std::uncaught_exception())) func_();
  }
  ...
};
```

现在，回到第一部分：ScopeGuard 将在条件正确时执行延迟操作，那么我们如何告诉它正确的条件呢？使用我们之前开发的宏方法，我们可以定义三个版本的守卫——`ON_SCOPE_EXIT`总是执行，`ON_SCOPE_SUCCESS`仅在未抛出异常时执行，而`ON_SCOPE_FAILURE`在抛出异常时执行。后者替换了我们的`ON_SCOPE_EXIT_ROLLBACK`宏，但现在它也可以使用匿名变量名，因为没有显式调用`commit()`。这三个宏以非常相似的方式定义，我们只需要三个不同的唯一类型而不是一个`ScopeGuardOnExit`，这样我们就可以决定调用哪个`operator+()`：

```cpp
// Example 10
struct ScopeGuardOnExit {};
template <typename Func>
auto operator+(ScopeGuardOnExit, Func&& func) {
  return
    ScopeGuard<Func, true, true>(std::forward<Func>(func));
}
#define ON_SCOPE_EXIT auto ANON_VAR(SCOPE_EXIT_STATE) = \
  ScopeGuardOnExit() + [&]()
struct ScopeGuardOnSuccess {};
template <typename Func>
auto operator+(ScopeGuardOnSuccess, Func&& func) {
  return
   ScopeGuard<Func, true, false>(std::forward<Func>(func));
}
#define ON_SCOPE_SUCCESS auto ANON_VAR(SCOPE_EXIT_STATE) =\
  ScopeGuardOnSuccess() + [&]()
struct ScopeGuardOnFailure {};
template <typename Func>
auto operator+(ScopeGuardOnFailure, Func&& func) {
  return
   ScopeGuard<Func, false, true>(std::forward<Func>(func));
}
#define ON_SCOPE_FAILURE auto ANON_VAR(SCOPE_EXIT_STATE) =\
  ScopeGuardOnFailure() + [&]()
```

每个 `operator+()` 的重载都会使用不同的布尔参数构建一个 `ScopeGuard` 对象，这些参数控制它何时执行以及何时不执行。每个宏通过指定 `operator+()` 的第一个参数的类型，使用我们为此目的定义的唯一树类型之一来指导 lambda 表达式：`ScopeGuardOnExit`、`ScopeGuardOnSuccess` 和 `ScopeGuardOnFailure`。

此实现可以通过简单的甚至相当复杂的测试，看起来似乎可以工作。不幸的是，它有一个致命的缺陷——它不能正确地检测成功或失败。当然，如果我们的 `Database::insert()` 函数是从正常的控制流中调用的，它可能成功也可能失败，它工作得很好。问题是，我们可能从某个其他对象的析构函数中调用 `Database::insert()`，而这个对象可能被用于一个抛出异常的作用域中：

```cpp
class ComplexOperation {
  Database db_;
  public:
  ...
  ~ComplexOperation() {
    try {
      db_.insert(some_record);
    } catch (...) {}    // Shield any exceptions from insert()
  }
};
{
  ComplexOperation OP;
  throw 1;
}    // OP.~ComplexOperation() runs here
```

现在，`db_.insert()` 在未捕获异常的存在下运行，因此 `std::uncaught_exception()` 将返回 `true`。问题是这并不是我们正在寻找的异常。这个异常并不表明 `insert()` 失败了，但它将被视为失败，并且数据库插入将被撤销。

我们真正需要知道的是当前正在传播多少个异常。这个说法可能听起来有些奇怪，因为 C++ 不允许同时传播多个异常。然而，我们已经看到这是一个过度简化的说法；第二个异常只要它没有逃离析构函数，就可以很好地传播。同样，如果有嵌套析构函数调用，三个或更多异常也可以传播，我们只需及时捕获它们即可。为了正确解决这个问题，我们需要知道在调用 `Database::insert()` 函数时正在传播多少个异常。然后，我们可以将其与函数结束时的异常传播数量进行比较，无论我们如何到达那里。如果这些数字相同，`insert()` 没有抛出任何异常，并且任何现有的异常都不是我们的问题。如果添加了新的异常，`insert()` 失败了，并且退出处理必须相应地更改。

C++17 允许我们实现这种检测；除了之前已弃用的 `std::uncaught_exception()`（在 C++20 中被移除），我们现在有一个新的函数，`std::uncaught_exceptions()`，它返回当前正在传播的异常数量。我们现在可以实现这个 `UncaughtExceptionDetector` 来检测新的异常：

```cpp
// Example 10a
class UncaughtExceptionDetector {
  public:
  UncaughtExceptionDetector() :
    count_(std::uncaught_exceptions()) {}
  operator bool() const noexcept {
    return std::uncaught_exceptions() > count_;
  }
  private:
  const int count_;
};
```

使用这个检测器，我们最终可以实现自动的 `ScopeGuard`：

```cpp
// Example 10a
template <typename Func, bool on_success, bool on_failure>
class ScopeGuard {
  public:
  ...
  ~ScopeGuard() {
  if ((on_success && !detector_) ||
      (on_failure && detector_)) func_();
  }
  ...
  private:
  UncaughtExceptionDetector detector_;
  ...
};
```

需要使用 C++17 可能会在使用此技术在受限于较旧语言版本的程序中遇到（希望是短期）障碍。虽然没有其他符合标准、可移植的解决此问题的方法，但大多数现代编译器都有方法来获取未捕获异常计数器。这就是在 GCC 或 Clang（以 `__` 开头的名称是 GCC 内部类型和函数）中是如何做的：

```cpp
// Example 10b
namespace  cxxabiv1 {
  struct cxa_eh_globals;
  extern "C" cxa_eh_globals* cxa_get_globals() noexcept;
}
class UncaughtExceptionDetector {
  public:
  UncaughtExceptionDetector() :
    count_(uncaught_exceptions()) {}
  operator bool() const noexcept {
    return uncaught_exceptions() > count_;
  }
  private:
  const int count_;
  int uncaught_exceptions() const noexcept {
    return *(reinterpret_cast<int*>(
      static_cast<char*>( static_cast<void*>(
        cxxabiv1::cxa_get_globals())) + sizeof(void*)));
  }
};
```

无论我们使用异常驱动的 ScopeGuard 还是显式命名的 ScopeGuard（可能用于处理错误代码以及异常），我们都已经实现了目标——我们现在可以指定在函数或任何其他作用域结束时必须执行的操作。

在本章末尾，我们将展示另一种可以在网络上找到的 ScopeGuard 实现。这种实现值得考虑，但你也应该意识到它的缺点。

# 类型擦除的 ScopeGuard

如果你在网上搜索 ScopeGuard 示例，可能会偶然发现一个使用`std::function`而不是类模板的实现。这个实现本身相当简单：

```cpp
// Example 11
class ScopeGuard {
  public:
  template <typename Func> ScopeGuard(Func&& func) :
    func_(std::forward<Func>(func)) {}
  ~ScopeGuard() { if (!commit_) func_(); }
  void commit() const noexcept { commit_ = true; }
  ScopeGuard(ScopeGuard&& other) :
    commit_(other.commit_), func_(std::move(other.func_)) {
    other.commit();
  }
  private:
  mutable bool commit_ = false;
  std::function<void()> func_;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
};
```

注意，这个 ScopeGuard 是一个类，而不是类模板。它有模板构造函数，可以接受与另一个守卫相同的 lambda 表达式或另一个可调用对象。但用于存储该表达式的变量无论可调用对象是什么类型，都具有相同的类型。这种类型是`std::function<void()>`，它是任何不接受任何参数且不返回任何内容的函数的包装器。如何将任何类型的值存储在某种固定类型的对象中？这就是类型擦除的魔法，我们有一个专门的章节来介绍它(第六章，*理解类型擦除*)。这个非模板的 ScopeGuard 使得使用它的代码更简单（至少在 C++17 之前），因为没有需要推断的类型：

```cpp
void Database::insert(const Record& r) {
  S.insert(r);
  ScopeGuard SF([&] { S.finalize(); });
  ScopeGuard SG([&] { S.undo(); });
  I.insert(r);
  SG.commit();
}
```

然而，这种方法有一个严重的缺点——类型擦除的对象必须进行相当数量的计算才能实现其魔法。至少，它涉及到间接或虚拟函数调用，并且通常还需要分配和释放一些内存。

通过使用第六章的教训，*理解类型擦除*，我们可以提出一个稍微更有效的类型擦除实现；特别是，我们可以坚持在退出时调用的可调用对象适合守卫的缓冲区：

```cpp
// Example 11
template <size_t S = 16>
class ScopeGuard : public CommitFlag {
  alignas(8) char space_[S];
  using guard_t = void(*)(void*);
  guard_t guard_ = nullptr;
  template<typename Callable>
  static void invoke(void* callable) {
    (*static_cast<Callable*>(callable))();
  }
  mutable bool commit_ = false;
  public:
  template <typename Callable,
            typename D = std::decay_t<Callable>>
    ScopeGuard(Callable&& callable) :
    guard_(invoke<Callable>) {
    static_assert(sizeof(Callable) <= sizeof(space_));
    ::new(static_cast<void*>(space_))
      D(std::forward<Callable>(callable));
  }
  ScopeGuard(ScopeGuard&& other) = default;
  ~ScopeGuard() { if (!commit_) guard_(space_); }
};
```

我们可以使用 Google Benchmark 库比较类型擦除的 ScopeGuard 与模板`ScopeGuard`的运行时成本。结果将取决于我们正在保护的操作：对于长时间计算和昂贵的退出操作，`ScopeGuard`的运行时差异微不足道。如果作用域内的计算和退出作用域的计算很快，差异将会更加明显：

```cpp
void BM_nodelete(benchmark::State& state) {
  for (auto _ : state) {
    int* p = nullptr;
    ScopeGuardTypeErased::ScopeGuard SG([&] { delete p; });
    p = rand() < 0 ? new int(42) : nullptr;
  }
  state.SetItemsProcessed(state.iterations());
}
```

注意，内存永远不会被分配（`rand()`返回非负随机数），指针`p`始终是`null`，因此我们正在基准测试`rand()`调用以及 ScopeGuard 的开销。为了比较，我们可以显式调用`delete`，而不使用守卫。结果显示，守卫的模板版本没有可测量的开销，而两种类型擦除的实现都有一些：

```cpp
Benchmark                              Time
-------------------------------------------
BM_nodelete_explicit                4.48 ns
BM_nodelete_type_erased             6.29 ns
BM_nodelete_type_erased_fast        5.48 ns
BM_nodelete_template                4.50 ns
```

我们自己的类型擦除版本为每个迭代增加了大约 1 纳秒，而基于`std::function`的版本则几乎需要两倍的时间。这种类型的基准测试强烈受到编译器优化的影响，并且可能会对代码的微小变化产生非常不同的结果。例如，让我们将代码改为始终构造新对象：

```cpp
void BM_nodelete(benchmark::State& state) {
  for (auto _ : state) {
    int* p = nullptr;
    ScopeGuardTypeErased::ScopeGuard SG([&] { delete p; });
    p = rand() >= 0 ? new int(42) : nullptr;
  }
  state.SetItemsProcessed(state.iterations());
}
```

现在我们对循环的每次迭代都调用 operator new，因此相应的删除也必须发生。这次，编译器能够比类型擦除版本更好地优化模板`ScopeGuard`：

```cpp
Benchmark                              Time
-------------------------------------------
BM_delete_explicit                  4.54 ns
BM_delete_type_erased               13.4 ns
BM_delete_type_erased_fast          12.7 ns
BM_delete_template                  4.56 ns
```

总体来说，在这里使用类型擦除并没有太大的理由。运行时成本可能微不足道或相当显著，但通常没有什么可以获得的。类型擦除版本的唯一优势是守卫本身总是同一类型。但守卫的类型几乎总是无关紧要：我们创建变量时使用`auto`或构造函数模板参数推导，我们可能需要的唯一显式操作是对守卫进行解除武装。因此，我们永远不需要编写任何依赖于守卫类型的代码。总的来说，基于模板的 ScopeGuard，无论是否有宏，都是自动释放资源并在作用域结束时执行其他操作的优选模式。

# 摘要

在本章中，我们详细研究了编写异常安全和错误安全代码的最佳 C++模式之一。ScopeGuard 模式允许我们在作用域完成后执行任意操作，即 C++代码片段。作用域可能是一个函数，循环体，或者只是插入到程序中以管理局部变量生命周期的作用域。执行到最后的操作可能取决于作用域的成功完成，无论定义如何。ScopeGuard 模式在成功或失败通过返回代码或异常指示时同样有效，尽管在后一种情况下我们可以自动检测失败（对于返回代码，程序员必须明确指定哪些返回值表示成功，哪些不表示成功）。我们已经观察到了 ScopeGuard 模式的演变，随着更现代的语言特性的使用。在其最佳形式中，ScopeGuard 提供了一种简单声明式的方式来指定后置条件和延迟操作，如清理或回滚，这种方式对于任何需要提交或撤销的操作数量都是简单可组合的。

下一章将描述另一种非常 C++特定的模式，即 Friends Factory，它是一种工厂模式，只是在程序执行期间制造对象，而不是在编译期间制造函数。

# 问题

1.  什么是错误安全或异常安全的程序？

1.  我们如何使执行多个相关操作的例程成为错误安全的？

1.  RAII 是如何帮助编写错误安全程序的？

1.  ScopeGuard 模式是如何泛化 RAII 惯用的？

1.  程序如何自动检测函数成功退出和失败的情况？

1.  类型擦除的 ScopeGuard 有什么优点和缺点？
