# 4

# 基础库和工具

LLVM 使用 C++ 语言编写，截至 2022 年 7 月，它使用的是 C++17 版本的 C++ 标准 [6]。LLVM 主动利用 **标准模板库 (STL)** 提供的功能。另一方面，LLVM 包含了许多针对基本容器的内部实现 [13]，主要目的是优化性能。例如，`llvm::SmallVector` 具有与 `std::vector` 类似的接口，但具有内部优化的实现。因此，熟悉这些扩展对于希望与 LLVM 和 Clang 一起工作的人来说是必不可少的。

此外，LLVM 还引入了其他开发工具，如 **TableGen**，这是一个用于结构数据处理 **领域特定语言 (DSL**)，以及 **LIT**（LLVM 集成测试器），LLVM 测试框架。关于这些工具的更多细节将在本章后面讨论。本章将涵盖以下主题：

+   LLVM 编码风格

+   LLVM 基础库

+   Clang 基础库

+   LLVM 支持工具

+   Clang 插件项目

我们计划使用一个简单的示例项目来展示这些工具。该项目将是一个 Clang 插件，用于估计 C++ 类的复杂度。如果一个类的函数数量超过作为参数指定的阈值，则认为该类是复杂的。虽然这种复杂性的定义可能被认为是微不足道的，但我们将在 *第六章* *高级代码分析* 中探讨更复杂的复杂性定义。

## 4.1 技术要求

本章的源代码位于本书 GitHub 仓库的 `chapter4` 文件夹中：[`github.com/PacktPublishing/Clang-Compiler-Frontend-Packt/tree/main/chapter4`](https://github.com/PacktPublishing/Clang-Compiler-Frontend-Packt/tree/main/chapter4)。

## 4.2 LLVM 编码风格

LLVM 遵循特定的代码风格规则 [11]。这些规则的主要目标是促进熟练的 C++ 实践，特别关注性能。如前所述，LLVM 使用 C++17 并倾向于使用 **STL**（即 **标准模板库**）中的数据结构和算法。另一方面，LLVM 提供了许多与 STL 中类似的数据结构的优化版本。例如，`llvm::SmallVector<>` 可以被视为 `std::vector<>` 的优化版本，尤其是在向量大小较小时，这是编译器中使用的数据结构的一个常见特性。

在选择 STL 对象/算法及其对应的 LLVM 版本之间，LLVM 编码标准建议优先选择 LLVM 版本。

其他规则与性能限制相关的问题有关。例如，**运行时类型信息 (RTTI)** 和 C++ 异常都是不允许的。然而，在某些情况下，RTTI 可能会证明是有益的；因此，LLVM 提供了如 `llvm::isa<>` 和其他类似模板辅助函数的替代方案。更多关于此的信息可以在*第 4.3.1 节**，RTTI 替换和转换运算符*中找到。而不是使用 C++ 异常，LLVM 经常使用 C 风格的 `assert`s。

有时，断言信息不足。LLVM 建议向它们添加文本消息以简化调试。以下是从 Clang 代码中的一个典型示例：

```cpp
static bool unionHasUniqueObjectRepresentations(const ASTContext &Context, 

                                                const RecordDecl *RD, 

                                                bool CheckIfTriviallyCopyable) { 

  assert(RD->isUnion() && "Must be union type"); 

  CharUnits UnionSize = Context.getTypeSizeInChars(RD->getTypeForDecl());

```

**图 4.1**：在 clang/lib/AST/ASTContext.cpp 中的 assert() 使用

在代码中，我们检查第二个参数（`RD`）是否为联合类型，如果不是，则抛出一个带有相应信息的断言。

除了性能考虑之外，LLVM 还引入了一些额外的要求。其中之一是关于注释的要求。代码注释非常重要。此外，LLVM 和 Clang 都有从代码生成的全面文档。他们使用 Doxygen ([`www.doxygen.nl/`](https://www.doxygen.nl/)) 来实现这一点。这个工具是 C/C++ 程序注释的事实标准，你很可能之前已经遇到过它。

Clang 和 LLVM 不是单一的大块代码；相反，它们被实现为一组库。这种设计在代码和功能重用方面提供了优势，我们将在*第八章**，IDE 支持和 Clangd*中探讨这些优势。这些库也是 LLVM 代码风格执行的优秀示例。让我们详细检查这些库。

## 4.3 LLVM 基本库

我们将从 LLVM 代码中的 RTTI 替换开始，讨论其实现方式。然后，我们将继续讨论基本容器和智能指针。最后，我们将讨论一些用于表示标记位置的重要类以及 Clang 中诊断的实现方式。稍后，在*第 4.6 节**，Clang 插件项目*中，我们将在我们的测试项目中使用这些类。

### 4.3.1 RTTI 替换和转换运算符

如前所述，LLVM 由于性能考虑而避免使用 RTTI。LLVM 引入了几种辅助函数来替代 RTTI 对应的函数，允许将对象从一个类型转换为另一个类型。基本的有以下几种：

+   `llvm::isa<>` 类似于 Java 的 `java instanceof` 运算符。它根据测试对象的引用是否属于测试的类返回 `true` 或 `false`。

+   `llvm::cast<>`：当你确定对象是特定派生类型时，使用此转换运算符。如果转换失败（即对象不是预期的类型），`llvm::cast` 将终止程序。仅在确信转换不会失败时使用。

+   `llvm``::``dyn_cast``<>`: 这可能是 LLVM 中最常用的类型转换运算符。`llvm``::``dyn_cast` 用于在预期转换通常成功但存在一些不确定性的情况下进行安全的向下转换。如果对象不是指定的派生类型，`llvm``::``dyn_cast``<>` 返回 `nullptr`。

类型转换运算符不接受 `nullptr` 作为输入。然而，有两个特殊的类型转换运算符可以处理空指针：

+   `llvm``::``cast_if_present``<>`: `llvm``::``cast``<>` 的一个变体，接受 `nullptr` 值

+   `llvm``::``dyn_cast_if_present``<>`: `llvm``::``dyn_cast``<>` 的一个变体，接受 `nullptr` 值

这两个运算符都可以处理 `nullptr` 值。如果输入是 `nullptr` 或转换失败，它们将简单地返回 `nullptr`。

重要提示

值得注意的是，类型转换运算符 `llvm``::``cast_if_present``<>` 和 `llvm``:` `:``dyn_cast_if_present``<>` 是最近引入的，具体是在 2022 年。它们作为流行的 `llvm``::``cast_or_null``<>` 和 `llvm``::``dyn_cast_or` `_null``<>` 的替代品，后者最近已被使用。旧版本仍然得到支持，并且现在将调用重定向到新的类型转换运算符。有关更多信息，请参阅关于此更改的讨论：[`discourse.llvm.org/t/psa-swapping-out-or-null-with-if-present/65018`](https://discourse.llvm.org/t/psa-swapping-out-or-null-with-if-present/65018)

.

可能会提出以下问题：如何在没有 RTTI 的情况下执行动态转换操作？这可以通过某些特定的装饰来实现，如下面一个简单的例子所示，该例子受到 *如何为你的类层次结构设置* *LLVM 风格的 RTTI* [14] 的启发。我们将从一个基类 `clangbook``::``Animal` 开始，该类有两个派生类：`clangbook``::``Horse` 和 `clangbook``::``Sheep`。每匹马可以通过其速度（英里/小时）进行分类，而每只羊可以通过其羊毛质量进行分类。以下是它的用法：

```cpp
46 void testAnimal() { 

47   auto AnimalPtr = std::make_unique<clangbook::Horse>(10); 

48   if (llvm::isa<clangbook::Horse>(AnimalPtr)) { 

49     llvm::outs() 

50         << "Animal is a Horse and the horse speed is: " 

51         << llvm::dyn_cast<clangbook::Horse>(AnimalPtr.get())->getSpeed() 

52         << "mph \n"; 

53   } else { 

54     llvm::outs() << "Animal is not a Horse\n"; 

55   } 

56 }
```

**图 4.2**: LLVM `isa``<>` 和 `dyn_cast``<>` 使用示例

代码应生成以下输出：

```cpp
Animal is a Horse and the horse speed is: 10mph
```

图 4.2 中的 *第 48 行* 展示了 `llvm``::``isa``<>` 的用法，而 *第 51 行* 展示了 `llvm``::``dyn_cast``<>` 的用法。在后一个例子中，我们将基类转换为 `clangbook``::``Horse` 并调用该类特定的方法。

让我们来看看类实现，这将提供关于 RTTI 替换如何工作的见解。我们将从基类 `clangbook``::``Animal` 开始：

```cpp
9 class Animal { 

10 public: 

11   enum AnimalKind { AK_Horse, AK_Sheep }; 

12  

13 public: 

14   Animal(AnimalKind K) : Kind(K){}; 

15   AnimalKind getKind() const { return Kind; } 

16  

17 private: 

18   const AnimalKind Kind; 

19 };
```

**图 4.3**: `clangbook``::``Animal` 类

最关键的部分是前面代码中的 *第 11 行*。它指定了不同的 ”种类” 的动物。一个枚举值用于马 (`AK_Horse`)，另一个用于羊 (`AK_Sheep`)。因此，基类对其派生类有一些了解。`clangbook``::``Horse` 和 `clangbook``::``Sheep` 类的实现可以在以下代码中找到：

```cpp
21 class Horse : public Animal { 

22 public: 

23   Horse(int S) : Animal(AK_Horse), Speed(S){}; 

24  

25   static bool classof(const Animal *A) { return A->getKind() == AK_Horse; } 

26  

27   int getSpeed() { return Speed; } 

28  

29 private: 

30   int Speed; 

31 }; 

32  

33 class Sheep : public Animal { 

34 public: 

35   Sheep(int WM) : Animal(AK_Sheep), WoolMass(WM){}; 

36  

37   static bool classof(const Animal *A) { return A->getKind() == AK_Sheep; } 

38  

39   int getWoolMass() { return WoolMass; } 

40  

41 private: 

42   int WoolMass; 

43 };
```

**图 4.4**: `clangbook``::``Horse` 和 `clangbook``::``Sheep` 类

*第 25 行和第 37 行* 特别重要，因为它们包含了 `classof` 静态方法实现。这个方法对于 LLVM 中的类型转换操作至关重要。一个典型的实现可能看起来像以下（简化版本）：

```cpp
1template <typename To, typename From> 

2 bool isa(const From *Val) { 

3   return To::classof(Val); 

4 }
```

**图 4.5**：`llvm::isa<>` 的简化实现

同样的机制可以应用于其他类型转换操作。

我们接下来将讨论各种类型的容器，它们是相应 STL 对应容器的更强大的替代品。

### 4.3.2 容器

LLVM ADT（代表抽象数据类型）库提供了一套容器。虽然其中一些是 LLVM 独有的，但其他一些可以被认为是 STL 容器的替代品。我们将探讨 ADT 提供的一些最受欢迎的类。

#### 字符串操作

在标准 C++ 库中，用于处理字符串的主要类是 `std::string`。尽管这个类被设计成通用的，但它有一些与性能相关的问题。一个重要的问题涉及到复制操作。由于在编译器中复制字符串是一个常见的操作，LLVM 引入了一个专门的类，`llvm::StringRef`，它以高效的方式处理这个操作，而不使用额外的内存。这个类与 C++17 中的 `std::string_view` [20] 和 C++20 中的 `std::span` [21] 相当。

`llvm::StringRef` 类维护对数据的引用，不需要像传统的 C/C++ 字符串那样以空字符终止。它本质上持有指向数据块的指针和块的大小，使得对象的有效大小为 16 字节。由于 `llvm::StringRef` 保留的是引用而不是实际数据，它必须从一个现有的数据源构建。这个类可以从基本字符串对象，如 `const char*`、`std::string` 和 `std::string_view` 实例化。默认构造函数创建一个空对象。`llvm::StringRef` 的典型使用示例在 图 4.6 中展示：

```cpp
1  #include "llvm/ADT/StringRef.h" 

2   ... 

3   llvm::StringRef StrRef("Hello, LLVM!"); 

4   // Efficient substring, no allocations 

5   llvm::StringRef SubStr = StrRef.substr(0, 5); 

6  

7   llvm::outs() << "Original StringRef: " << StrRef.str() << "\n"; 

8   llvm::outs() << "Substring: " << SubStr.str() << "\n";
```

**图 4.6**：`llvm::StringRef` 使用示例

代码的输出如下所示：

```cpp
Original StringRef: Hello, LLVM!
Substring: Hello
```

在 LLVM 中用于字符串操作的另一类是 `llvm::Twine`，它在将多个对象连接成一个对象时特别有用。该类的典型使用示例在 图 4.7 中展示：

```cpp
1   #include "llvm/ADT/Twine.h" 

2    ... 

3    llvm::StringRef Part1("Hello, "); 

4    llvm::StringRef Part2("Twine!"); 

5    llvm::Twine Twine = Part1 + Part2;  // Efficient concatenation 

6  

7    // Convert twine to a string (actual allocation happens here) 

8    std::string TwineStr = Twine.str(); 

9    llvm::outs() << "Twine result: " << TwineStr << "\n";
```

**图 4.7**：`llvm::Twine` 使用示例

代码的输出如下所示：

```cpp
Twine result: Hello, Twine!
```

另一个广泛用于字符串操作的类是 `llvm::SmallString<>`。它表示一个堆栈分配的字符串，大小固定，但也可以超出这个大小，此时它会堆分配内存。这是堆栈分配的空间效率和堆分配的灵活性之间的结合。

`llvm::SmallString<>`的优势在于，在许多场景中，尤其是在编译器任务中，字符串往往很小，可以适应栈分配的空间。这避免了动态内存分配的开销。但在需要更大字符串的情况下，`llvm::SmallString`仍然可以通过切换到堆内存来容纳。一个典型的使用示例显示在图 4.8 中：

```cpp
1   #include "llvm/ADT/SmallString.h" 

2    ... 

3    // Stack allocate space for up to 20 characters. 

4    llvm::SmallString<20> SmallStr; 

5  

6    // No heap allocation happens here. 

7    SmallStr = "Hello, "; 

8    SmallStr += "LLVM!"; 

9  

10    llvm::outs() << "SmallString result: " << SmallStr << "\n";
```

**图 4.8**：`llvm::SmallString<>`使用示例

尽管字符串操作在文本解析等编译器任务中至关重要，但 LLVM 还有许多其他辅助类。我们将接下来探讨其顺序容器。

#### 顺序容器

LLVM 推荐一些针对标准库中的数组和向量的优化替代方案。最显著的是：

+   `llvm::ArrayRef<>`：一个为接受元素顺序列表进行只读访问的接口设计的辅助类。该类类似于`llvm::StringRef<>`，因为它不拥有底层数据，而只是引用它。

+   `llvm::SmallVector<>`：一种针对小尺寸情况的优化向量。它类似于在*第 4.3.2 节**，*字符串操作*中讨论的`llvm::SmallString`。值得注意的是，数组的大小不是固定的，允许存储的元素数量增长。如果元素数量保持在`N`（模板参数）以下，则不需要额外的内存分配。

让我们通过图 4.9 来检查`llvm::SmallVector<>`，以更好地理解这些容器：

```cpp
1    llvm::SmallVector<int, 10> SmallVector; 

2     for (int i = 0; i < 10; i++) { 

3       SmallVector.push_back(i); 

4     } 

5     SmallVector.push_back(10);
```

**图 4.9**：`llvm::SmallVector<>`使用

向量在*行 1*初始化，选择了大小为 10（由第二个模板参数指示）。该容器提供了一个类似于`std::vector<>`的 API，使用熟悉的`push_back`方法添加新元素，如图 4.9，*行 3 和 5*所示。

前十个元素被添加到向量中，而不需要额外的内存分配（参见图 4.9，*行 2-4*）。然而，当第 11 个元素在*行 5*被添加时，数组的大小超过了为 10 个元素预先分配的空间，从而触发了额外的内存分配。这种容器设计有效地最小化了小对象的内存分配。

同时保持灵活性，以便在必要时容纳更大的大小。

#### 类似于映射的容器

标准库提供了几个用于存储键值数据的容器。最常见的是`std::map<>`用于通用映射和`std::unordered_map<>`用于哈希映射。LLVM 为这些标准容器提供了额外的替代方案：

+   `llvm``::``StringMap``<>`: 使用字符串作为键的映射。通常，这比标准的关联容器 `std``::``unordered_map``<``std``::``string``,` `T``>` 性能优化得更好。它常用于字符串键占主导地位且性能至关重要的场景，例如在 LLVM 这样的编译器基础设施中。与 LLVM 中的许多其他数据结构不同，`llvm``::``StringMap``<>` 不存储字符串键的副本。相反，它保留对字符串数据的引用，因此确保字符串数据比映射存在的时间长是防止未定义行为的关键。

+   `llvm``::``DenseMap``<>`: 这个映射在大多数情况下比 `std``::``unordered_map``<>` 更节省内存和时间，尽管它带来了一些额外的限制（例如，键和值具有平凡的析构函数）。当你有简单的键值类型并且需要高性能的查找时，它特别有益。

+   `llvm``::``SmallDenseMap``<>`: 这个映射类似于 `llvm``::``DenseMap``<>`，但针对映射大小通常较小的情况进行了优化。对于小映射，它从栈上分配，只有当映射超过预定义大小时才回退到堆分配。

+   `llvm``::``MapVector``<>`: 这个容器保留了插入顺序，类似于 Python 的 `OrderedDict`。它实现为 `std``::``vector` 和 `llvm``::``DenseMap` 或 `llvm``::``SmallDenseMap` 的混合。

值得注意的是，这些容器使用的是二次探测的哈希表机制。这种方法在解决哈希冲突时非常有效，因为在查找元素时不会重新计算缓存。这对于性能关键的应用程序，如编译器来说至关重要。

### 4.3.3 智能指针

在 LLVM 代码中可以找到不同的智能指针。最受欢迎的是来自标准模板库的：`std``::``unique_ptr``<>` 和 `std``::``shared_ptr``<>`。此外，LLVM 提供了一些辅助类来与智能指针一起使用。其中最突出的是 `llvm``::``IntrusiveRefCntPtr``<>`。这个智能指针旨在与支持侵入式引用计数的对象一起使用。与维护自己的控制块以管理引用计数的 `std``::``shared_ptr` 不同，`IntrusiveRefCntPtr` 预期对象维护自己的引用计数。这种设计可以更节省内存。这里展示了典型的使用示例：

```cpp
1  class MyClass : public llvm::RefCountedBase<MyClass> { 

2   // ... 

3   }; 

4  

5   llvm::IntrusiveRefCntPtr<MyClass> Ptr = new MyClass();
```

**图 4.10**: `llvm``::``IntrusiveRefCntPtr``<>` 使用示例

如我们所见，智能指针显著地使用了前面在 *第 3.3 节* 中提到的 CRTP（Curiously Recurring Template Pattern），即 AST 遍历。CRTP 对于当引用计数降至 0 且对象必须被删除时的 `Release` 操作至关重要。实现如下：

```cpp
1template <class Derived> class RefCountedBase { 

2   // ... 

3   void Release() const { 

4    assert(RefCount > 0 && "Reference count is already zero."); 

5    if (--RefCount == 0) 

6      delete static_cast<const Derived *>(this); 

7   } 

8 }
```

**图 4.11**: 在 `llvm``::``RefCountedBase``<>` 中的 CRTP 使用。代码来源于 `llvm/ADT/IntrusiveRefCntPtr.h` 头文件

由于 图 4.10 中的 `MyClass` 是从 `RefCountedBase` 派生的，我们可以在 图 4.11 的 *第 6 行* 上对其执行类型转换。这种转换是可行的，因为已知要转换的类型，它作为模板参数提供。

我们刚刚完成了 LLVM 基础库。现在是我们转向 Clang 基础库的时候了。Clang 是一个编译器前端，其最重要的操作与诊断相关。诊断需要关于源代码中位置精确的信息。让我们探索 Clang 为这些操作提供的基本类。

## 4.4 Clang 基础库

Clang 是一个编译器前端，其最重要的操作与诊断相关。诊断需要关于源代码中位置精确的信息。让我们探索 Clang 为这些操作提供的基本类。

### 4.4.1 SourceManager 和 SourceLocation

Clang 作为编译器，与文本文件（程序）操作，在程序中定位特定位置是请求最频繁的操作之一。让我们看看典型的 Clang 错误报告。考虑来自 *第三章** 的一个程序，Clang AST，如 图 3.33 所示。Clang 为该程序生成以下错误消息：

```cpp
$ <...>/llvm-project/install/bin/clang -fsyntax-only maxerr.cpp
maxerr.cpp:3:12: error: use of undeclared identifier ’ab’
    return ab;
           ^
1  error generated.
```

**图 4.12**：maxerr.cpp 中报告的错误

正如我们在 图 4.12 中所看到的，显示消息需要以下信息：

+   文件名：在我们的例子中，它是`maxerr.cpp`

+   文件中的行：在我们的例子中，它是`3`

+   文件中的列：在我们的例子中，它是`12`

存储这些信息的应该尽可能紧凑，因为编译器会频繁使用它。Clang 将所需信息存储在 `clang::SourceLocation` 对象中。

这个对象经常被使用，因此它应该体积小且复制速度快。我们可以使用 lldb 检查对象的大小。例如，如果我们以调试器运行 Clang，我们可以确定大小如下：

```cpp
$ lldb <...>/llvm-project/install/clang
...
(lldb) p sizeof(clang::SourceLocation)
(unsigned long) 4
(lldb)
```

**图 4.13**：在调试器下确定 clang::SourceLocation 的大小

也就是说，信息是使用单个 `unsigned` `long` 数字编码的。这是如何可能的？这个数字仅仅作为一个文本文件中位置的标识符。需要一个额外的类来正确提取和表示这些信息，即 `clang::SourceManager`。`SourceManager` 对象包含有关特定位置的所有详细信息。在 Clang 中，由于宏、包含和其他预处理指令的存在，管理源位置可能具有挑战性。因此，有几种方式来解释给定的源位置。主要方式如下：

+   **拼写位置**：指的是在源代码中实际拼写的地方。如果你有一个指向宏体内部的源位置，拼写位置将给出宏内容在源代码中定义的位置。

+   **宏展开位置**：指宏展开的位置。如果你有一个指向宏体内部的源位置，展开位置将给出宏在源代码中被使用（展开）的位置。

让我们来看一个具体的例子：

```cpp
1 #define BAR void bar() 

2 int foo(int x); 

3 BAR;
```

**图 4.14**：测试不同类型源位置的示例程序：functions.hpp

在 图 4.14 中，我们定义了两个函数：*第 2 行* 的 `int foo()` 和 *第 3 行* 的 `void bar()`。对于第一个函数，拼写和展开位置都指向 *第 2 行*。然而，对于第二个函数，拼写位置在 *第 1 行*，而展开位置在 *第 3 行*。

让我们通过一个测试 Clang 工具来检查这个问题。我们将使用 *第 3.4 节* 中的测试项目，递归 AST 访问器，并在此处替换一些代码部分。首先，我们必须将 `clang::ASTContext` 传递给我们的 `Visitor` 实现中。这是必需的，因为 `clang::ASTContext` 提供了对 `clang::SourceManager` 的访问。我们将替换 图 3.8 中的 *第 11 行* 并按如下方式传递 `ASTContext`：

```cpp
10   CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef File) { 

11     return std::make_unique<Consumer>(&CI.getASTContext());
```

`Consumer` 类（参见图 3.9）将接受参数并将其用作 `Visitor` 的参数：

```cpp
8   Consumer(clang::ASTContext *Context) 

9       : V(std::make_unique<Visitor>(Context)) {}
```

主要更改针对 `Visitor` 类，该类大部分已重写。首先，我们将 `clang::ASTContext` 传递给类构造函数，如下所示：

```cpp
5 class Visitor : public clang::RecursiveASTVisitor<Visitor> { 

6 public: 

7   explicit Visitor(clang::ASTContext *C) : Context(C) {}
```

**图 4.15**：Visitor 类实现：构造函数

AST 上下文类存储为我们类的私有成员，如下所示：

```cpp
25 private: 

26   clang::ASTContext *Context;
```

**图 4.16**：Visitor 类实现：私有部分

主要处理逻辑在 `Visitor::VisitFunctionDecl` 方法中，你可以在下面看到：

```cpp
9   bool VisitFunctionDecl(const clang::FunctionDecl *FD) { 

10     clang::SourceManager &SM = Context->getSourceManager(); 

11     clang::SourceLocation Loc = FD->getLocation(); 

12     clang::SourceLocation ExpLoc = SM.getExpansionLoc(Loc); 

13     clang::SourceLocation SpellLoc = SM.getSpellingLoc(Loc); 

14     llvm::StringRef ExpFileName = SM.getFilename(ExpLoc); 

15     llvm::StringRef SpellFileName = SM.getFilename(SpellLoc); 

16     unsigned SpellLine = SM.getSpellingLineNumber(SpellLoc); 

17     unsigned ExpLine = SM.getExpansionLineNumber(ExpLoc); 

18     llvm::outs() << "Spelling : " << FD->getName() << " at " << SpellFileName 

19                  << ":" << SpellLine << "\n"; 

20     llvm::outs() << "Expansion : " << FD->getName() << " at " << ExpFileName 

21                  << ":" << ExpLine << "\n"; 

22     return true; 

23   }
```

**图 4.17**：Visitor 类实现：VisitFunctionDecl 方法

如果我们在 图 4.14 中的测试文件上编译并运行代码，将生成以下输出：

```cpp
Spelling : foo at functions.hpp:2
Expansion : foo at functions.hpp:2
Spelling : bar at functions.hpp:1
Expansion : bar at functions.hpp:3
```

**图 4.18**：recursivevisitor 可执行程序在 functions.hpp 测试文件上的输出

`clang::SourceLocation` 和 `clang::SourceManager` 是非常强大的类。结合其他类，如 `clang::SourceRange`（指定源范围开始和结束的两个源位置），它们为 Clang 中使用的诊断提供了一个很好的基础。

### 4.4.2 诊断支持

Clang 的诊断子系统负责生成和报告警告、错误和其他消息 [8]。涉及的主要类包括：

+   `DiagnosticsEngine`：管理诊断 ID 和选项

+   `DiagnosticConsumer`: 诊断消费者抽象基类

+   `DiagnosticIDs`：处理诊断标志和内部 ID 之间的映射

+   `DiagnosticInfo`：表示单个诊断

这里有一个简单的例子，说明了如何在 Clang 中发出警告：

```cpp
18   // Emit a warning 

19   DiagnosticsEngine.Report(DiagnosticsEngine.getCustomDiagID( 

20       clang::DiagnosticsEngine::Warning, "This is a custom warning."));
```

**图 4.19**：使用 clang::DiagnosticsEngine 发出警告

在我们的例子中，我们将使用一个简单的`DiagnosticConsumer`，即`clang::TextDiagnosticPrinter`，它格式化和打印处理过的诊断消息。

我们示例的主函数的完整代码显示在图 4.20 中：

```cpp
7 int main() { 

8   llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagnosticOptions = 

9       new clang::DiagnosticOptions(); 

10   clang::TextDiagnosticPrinter TextDiagnosticPrinter( 

11       llvm::errs(), DiagnosticOptions.get(), false); 

12  

13   llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagIDs = 

14       new clang::DiagnosticIDs(); 

15   clang::DiagnosticsEngine DiagnosticsEngine(DiagIDs, DiagnosticOptions, 

16                                              &TextDiagnosticPrinter, false); 

17  

18   // Emit a warning 

19   DiagnosticsEngine.Report(DiagnosticsEngine.getCustomDiagID( 

20       clang::DiagnosticsEngine::Warning, "This is a custom warning.")); 

21  

22   return 0; 

23 }
```

**图 4.20**: Clang 诊断示例

代码将产生以下输出

```cpp
warning: This is a custom warning.
```

**图 4.21**: 打印出的诊断信息

在这个例子中，我们首先使用`TextDiagnosticPrinter`作为其`DiagnosticConsumer`来设置`DiagnosticsEngine`。然后我们使用`DiagnosticsEngine`的`Report`方法来发出一个自定义警告。我们将在创建 Clang 插件的测试项目时添加一个更实际的例子，见*第 4.6 节**，Clang 插件* *项目*。

## 4.5 LLVM 支持工具

LLVM 项目有自己的工具支持。最重要的 LLVM 工具是 TableGen 和 LIT（代表 LLVM Integrated Tester）。我们将通过 Clang 代码的例子来探讨它们。这些例子应该有助于我们理解工具的目的以及如何使用它们。

### 4.5.1 TableGen

TableGen 是一种**领域特定语言 (DSL)**和相关的工具，用于 LLVM 项目中描述和生成表格，特别是描述目标架构的表格。这对于编译器基础设施非常有用，因为经常需要以结构化的方式描述诸如指令集、寄存器以及各种其他特定于目标属性。

TableGen 被用于 Clang 编译器的各个部分。它主要用于需要生成大量相似代码的地方。例如，它可以用于支持需要在大类中进行大量枚举声明的类型转换操作，或者在需要生成代码以处理大量相似诊断信息的诊断子系统中。我们将以 TableGen 在诊断系统中的功能为例进行考察。

我们将从描述 Clang 诊断的`Diagnostic.td`文件开始，该文件位于`clang/include/clang/Basic/Diagnostic.td`。让我们看看诊断严重性是如何定义的：

```cpp
16 // Define the diagnostic severities. 

17 class Severity<string N> { 

18   string Name = N; 

19 }
```

**图 4.22**: clang/include/clang/Basic/Diagnostic.td 中的严重性定义

在图 4.22 中，我们定义了一个严重性的类（*第 17-19 行*）。每个严重性都与一个字符串相关联，如下所示：

```cpp
20 def SEV_Ignored : Severity<"Ignored">; 

21 def SEV_Remark  : Severity<"Remark">; 

22 def SEV_Warning : Severity<"Warning">; 

23 def SEV_Error   : Severity<"Error">; 

24 def SEV_Fatal   : Severity<"Fatal">;
```

**图 4.23**: clang/include/clang/Basic/Diagnostic.td 中不同严重类型的定义

图 4.23 包含了不同严重性的定义；例如，`Warning`严重性在*第 22 行*被定义。

严重性随后被用来定义`Diagnostic`类，其中`Warning`诊断被定义为这个类的子类：

```cpp
// All diagnostics emitted by the compiler are an indirect subclass of this. 

class Diagnostic<string summary, DiagClass DC, Severity defaultmapping> { 

  ... 

} 

... 

class Warning<string str>   : Diagnostic<str, CLASS_WARNING, SEV_Warning>;
```

**图 4.24**: clang/include/clang/Basic/Diagnostic.td 中的诊断定义

使用 `Warning` 类定义，可以定义类的不同实例。例如，以下是一个定义位于 `DiagnosticSemaKinds.td` 中的未使用参数警告的实例：

```cpp
def warn_unused_parameter : Warning<"unused parameter %0">, 

  InGroup<UnusedParameter>, DefaultIgnore;
```

**图 4.25**：在 clang/include/clang/Basic/DiagnosticSemaKinds.td 中未使用参数警告的定义

`clang-tblgen` 工具将生成相应的 `DiagnosticSemaKinds.inc` 文件：

```cpp
DIAG(warn_unused_parameter, CLASS_WARNING, (unsigned)diag::Severity::Ignored, "unused parameter %0", 985, SFINAE_Suppress, false, false, true, false, 2)
```

**图 4.26**：在 clang/include/clang/Basic/DiagnosticSemaKinds.inc 中未使用参数警告的定义

此文件保留有关诊断的所有必要信息。这些信息可以通过使用 `DIAG` 宏的不同定义从 Clang 源代码中检索。

例如，以下代码利用 TableGen 生成的代码来提取诊断描述，如 `clang/lib/Basic/DiagnosticIDs.cpp` 中所示：

```cpp
  const StaticDiagInfoDescriptionStringTable StaticDiagInfoDescriptions = { 

#define DIAG(ENUM, CLASS, DEFAULT_SEVERITY, DESC, GROUP, SFINAE, NOWERROR,\ 

            SHOWINSYSHEADER, SHOWINSYSMACRO, DEFERRABLE, CATEGORY)    \ 

  DESC, 

... 

#include "clang/Basic/DiagnosticSemaKinds.inc" 

... 

#undef DIAG 

};
```

**图 4.27**：DIAG 宏定义

C++ 预处理器将扩展为以下内容：

```cpp
  const StaticDiagInfoDescriptionStringTable StaticDiagInfoDescriptions = { 

   ... 

   "unused parameter %0", 

   ... 

  };
```

**图 4.28**：DIAG 宏展开

提供的示例演示了如何使用 TableGen 在 Clang 中生成代码以及它如何简化 Clang 开发。诊断子系统不是 TableGen 被使用的唯一领域；它还在 Clang 的其他部分被广泛使用。例如，在各种类型的 AST 访问者中使用的宏也依赖于 TableGen 生成的代码；参见 *第 3.3.2 节**，访问者* *实现*。

### 4.5.2 LLVM 测试框架

LLVM 使用多个测试框架进行不同类型的测试。主要的是 **LLVM 集成测试器 (LIT)** 和 **Google 测试 (GTest)** [24]。LIT 和 GTest 都在 Clang 的测试基础设施中扮演着重要角色：

+   LIT 主要用于测试 Clang 工具链的整体行为，重点关注其代码编译能力和产生的诊断信息。

+   GTest 用于单元测试，针对代码库中的特定组件，主要是实用库和内部数据结构。

这些测试对于维护 Clang 项目的质量和稳定性至关重要。

重要提示

我们不会深入探讨 GTest，因为这个测试框架在 LLVM 之外被广泛使用，并且不是 LLVM 本身的一部分。有关 GTest 的更多信息，请访问其官方网站：[`github.com/google/googletest`](https://github.com/google/googletest)

我们将重点关注 LIT。LIT 是 LLVM 自有的测试框架，被广泛用于测试 LLVM 中的各种工具和库，包括 Clang 编译器。LIT 被设计为轻量级，并针对编译器测试的需求进行了定制。它通常用于运行本质上为 shell 脚本的测试，通常包含对输出中特定模式的检查。一个典型的 LIT 测试可能包括一个源代码文件以及一组 "RUN" 命令，这些命令指定如何编译、链接或其他方式处理文件，以及预期的输出。

RUN 命令通常使用 FileCheck，LLVM 项目中的另一个实用工具，来检查输出是否符合预期模式。在 Clang 中，LIT 测试通常用于测试前端功能，如解析、语义分析、代码生成和诊断。这些测试通常看起来像源代码文件，其中包含嵌入式注释，指示如何运行测试以及预期结果。

考虑以下来自`clang/test/Sema/attr-unknown.c`的示例：

```cpp
1 // RUN: %clang_cc1 -fsyntax-only -verify -Wattributes %s 

2  

3 int x __attribute__((foobar)); // expected-warning {{unknown attribute ’foobar’ ignored}} 

4 void z(void) __attribute__((bogusattr)); // expected-warning {{unknown attribute ’bogusattr’ ignored}}
```

**图 4.29**: 关于未知属性的 Clang 警告的 LIT 测试

示例是一个典型的 C 源代码文件，它可以被 Clang 处理。LIT 的行为由源文本中的注释控制。第一个注释（在第*1 行*）指定了如何执行测试。如指示，`clang`应使用一些额外的参数启动：`-fsyntax-only`和`-verify`。还有一些以`%`符号开始的替换。其中最重要的是`%s`，它被源文件名替换。LIT 还会检查以`expected-warning`开头的注释，并确保 Clang 输出产生的警告与预期值匹配。

测试可以按以下方式运行：

```cpp
$ ./build/bin/llvm-lit ./clang/test/Sema/attr-unknown.c
...
-- Testing: 1 tests, 1 workers --
PASS: Clang :: Sema/attr-unknown.c (1 of 1)

Testing Time: 0.06s
  Passed: 1
```

**图 4.30**: LIT 测试运行

我们从`build`文件夹运行`llvm-lit`，因为该工具不包括在安装过程中。一旦我们创建了我们的测试 Clang 插件项目并为其配置 LIT 测试，我们就可以获得有关 LIT 设置和其调用的更多详细信息。

## 4.6 Clang 插件项目

测试项目的目标是创建一个 Clang 插件，该插件将估计类复杂性。具体来说，如果一个类的成员方法数量超过某个阈值，则认为该类是复杂的。我们将利用迄今为止所获得的所有知识来完成此项目。这包括使用递归访问者和 Clang 诊断。此外，我们还将为我们的项目创建一个 LIT 测试。开发插件将需要为 LLVM 创建一个独特的构建配置，这是我们最初的步骤。

### 4.6.1 环境设置

插件将被创建为一个共享对象，我们的 LLVM 安装应该支持共享库（参见*第 1.3.1 节**，使用 CMake 进行配置）：

```cpp
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=../install -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_USE_SPLIT_DWARF=ON -DBUILD_SHARED_LIBS=ON ../llvm
```

**图 4.31**: Clang 插件项目使用的 CMake 配置

如所示，我们使用*第 1.4 节**的构建配置，测试项目 –* *使用 Clang 工具进行语法检查*，如图图 1.12 所示。在配置中，我们为安装工件设置了一个文件夹到`../install`，将我们的构建目标限制在`X86`平台，并且只启用`clang`项目。此外，我们为调试符号启用大小优化，并使用共享库而不是静态链接。

下一步涉及构建和安装 clang。这可以通过以下命令实现：

```cpp
$ ninja install
```

一旦我们完成 clang 的构建和安装，我们就可以继续处理我们项目的`CMakeLists.txt`文件。

### 4.6.2 插件的 CMake 构建配置

我们将使用图 3.20 作为插件构建配置的基础。我们将项目名称更改为`classchecker`，`ClassComplexityChecker.cpp`将作为我们的主要源文件。文件的主要部分在图 4.32 中展示。如我们所见，我们将构建一个共享库（*第 18-20 行*），而不是像之前的测试项目那样构建可执行文件。另一个修改是在*第 12 行*，我们为 LLVM 构建文件夹设置了一个配置参数。这个参数是必要的，以便定位 LIT 可执行文件，正如之前在*第 4.5.2 节*中提到的，它不包括在标准安装过程中。需要做一些额外的修改来支持 LIT 测试调用，但我们将稍后在*第 4.6.8 节*中讨论细节，即 LIT 对 clang 插件的测试（参见图 4.44）。

```cpp
8   message(STATUS "$LLVM_HOME found: $ENV{LLVM_HOME}") 

9   set(LLVM_HOME $ENV{LLVM_HOME} CACHE PATH "Root of LLVM installation") 

10   set(LLVM_LIB ${LLVM_HOME}/lib) 

11   set(LLVM_DIR ${LLVM_LIB}/cmake/llvm) 

12   set(LLVM_BUILD $ENV{LLVM_BUILD} CACHE PATH "Root of LLVM build") 

13   find_package(LLVM REQUIRED CONFIG) 

14   include_directories(${LLVM_INCLUDE_DIRS}) 

15   link_directories(${LLVM_LIBRARY_DIRS}) 

16  

17   # Add the plugin’s shared library target 

18   add_library(classchecker MODULE 

19     ClassChecker.cpp 

20   ) 

21   set_target_properties(classchecker PROPERTIES COMPILE_FLAGS "-fno-rtti") 

22   target_link_libraries(classchecker 

23     LLVMSupport 

24     clangAST 

25     clangBasic 

26     clangFrontend 

27     clangTooling 

28   )
```

**图 4.32**: 类复杂度插件的 CMakeLists.txt 文件

完成构建配置后，我们可以开始编写插件的主体代码。我们将创建的第一个组件是一个名为`ClassVisitor`的递归访问者类。

### 4.6.3 递归访问者类

我们的访问者类位于`ClassVisitor.hpp`文件中（参见图 4.33）。这是一个递归访问者，用于处理`clang::CXXRecordDecl`，这是 C++类声明的 AST 节点。我们在*第 13-16 行*计算方法数，如果超过阈值，则在*第 19-25 行*发出诊断。

```cpp
1 #include "clang/AST/ASTContext.h" 

2 #include "clang/AST/RecursiveASTVisitor.h" 

3  

4 namespace clangbook { 

5 namespace classchecker { 

6 class ClassVisitor : public clang::RecursiveASTVisitor<ClassVisitor> { 

7 public: 

8   explicit ClassVisitor(clang::ASTContext *C, int T) 

9       : Context(C), Threshold(T) {} 

10  

11   bool VisitCXXRecordDecl(clang::CXXRecordDecl *Declaration) { 

12     if (Declaration->isThisDeclarationADefinition()) { 

13       int MethodCount = 0; 

14       for (const auto *M : Declaration->methods()) { 

15         MethodCount++; 

16       } 

17  

18       if (MethodCount > Threshold) { 

19         clang::DiagnosticsEngine &D = Context->getDiagnostics(); 

20         unsigned DiagID = 

21             D.getCustomDiagID(clang::DiagnosticsEngine::Warning, 

22                               "class %0 is too complex: method count = %1"); 

23         clang::DiagnosticBuilder DiagBuilder = 

24             D.Report(Declaration->getLocation(), DiagID); 

25         DiagBuilder << Declaration->getName() << MethodCount; 

26       } 

27     } 

28     return true; 

29   }
```

```cpp
30 

31 private: 

32   clang::ASTContext *Context; 

33   int Threshold; 

34 }; 

35 } // namespace classchecker 

36 } // namespace clangbook
```

**图 4.33**: ClassVisitor.hpp 的源代码

值得注意的是诊断调用。诊断消息在第 20-22 行构建。我们的诊断消息接受两个参数：类名和类的方法数。这些参数在第 22 行使用`%1`和`%2`占位符进行编码。这些参数的实际值在第 25 行传递，在那里使用`DiagBuild`对象构建诊断消息。这个对象是`clang::DiagnosticBuilder`类的实例，它实现了**资源获取即初始化（RAII）**模式。它在销毁时发出实际的诊断。

重要提示

在 C++中，RAII 原则是一个常见的惯用语，用于通过将其与对象的生存期相关联来管理资源生存期。当一个对象超出作用域时，其析构函数会自动调用，这为释放对象持有的资源提供了机会。

`ClassVisitor`是在一个 AST 消费者类内部创建的，这将是我们的下一个主题。

### 4.6.4 插件 AST 消费者类

AST 消费者类在`ClassConsumer.hpp`中实现，代表标准的 AST 消费者，正如我们在 AST 访问者测试项目中看到的那样（参见图 3.9）。代码在图 4.35 中展示。

```cpp
1 namespace clangbook { 

2 namespace classchecker { 

3 class ClassConsumer : public clang::ASTConsumer { 

4 public: 

5   explicit ClassConsumer(clang::ASTContext *Context, int Threshold) 

6       : Visitor(Context, Threshold) {} 

7  

8   virtual void HandleTranslationUnit(clang::ASTContext &Context) { 

9     Visitor.TraverseDecl(Context.getTranslationUnitDecl()); 

10   } 

11  

12 private: 

13   ClassVisitor Visitor; 

14 }; 

15 } // namespace classchecker 

16 } // namespace clangbook
```

**图 4.34**: ClassConsumer.hpp 的源代码

代码在第 10 行初始化`Visitor`，并在第 13 行使用 Visitor 类遍历声明，从最顶层开始（翻译单元声明）。消费者必须从一个特殊的 AST 操作类创建，我们将在下一节讨论。

### 4.6.5 插件 AST 操作类

AST 操作的代码如图图 4.35 所示。可以观察到几个重要的部分：

+   *第 7 行*：我们从`clang::PluginASTAction`继承我们的`ClassAction`

+   *第 10-13 行*：我们实例化`ClassConsumer`并使用`MethodCountThreshold`，它是一个可选插件参数的派生

+   *第 15-25 行*：我们处理插件的可选`threshold`参数

```cpp
1 namespace clangbook { 

2 namespace classchecker { 

3 class ClassAction : public clang::PluginASTAction { 

4 protected: 

5   std::unique_ptr<clang::ASTConsumer> 

6   CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef) { 

7     return std::make_unique<ClassConsumer>(&CI.getASTContext(), 

8                                            MethodCountThreshold); 

9   } 

10  

11   bool ParseArgs(const clang::CompilerInstance &CI, 

12                  const std::vector<std::string> &args) { 

13     for (const auto &arg : args) { 

14       if (arg.substr(0, 9) == "threshold") { 

15         auto valueStr = arg.substr(10); // Get the substring after "threshold=" 

16         MethodCountThreshold = std::stoi(valueStr); 

17         return true; 

18       } 

19     } 

20     return true; 

21   } 

22   ActionType getActionType() { return AddAfterMainAction; } 

23  

24 private: 

25   int MethodCountThreshold = 5; // default value 

26 }; 

27 } // namespace classchecker 

28 } // namespace clangbook
```

**图 4.35**: ClassAction.hpp 的源代码

我们几乎完成了，准备初始化我们的插件。

### 4.6.6 插件代码

我们的插件注册是在`ClassChecker.cpp`文件中完成的，如图图 4.36 所示。

```cpp
1 #include "clang/Frontend/FrontendPluginRegistry.h" 

2  

3 #include "ClassAction.hpp" 

4  

5 static clang::FrontendPluginRegistry::Add<clangbook::classchecker::ClassAction> 

6     X("classchecker", "Checks the complexity of C++ classes");
```

**图 4.36**: ClassChecker.cpp 的源代码

如我们所见，大多数初始化都被辅助类隐藏了，我们只需要将我们的实现传递给`lang::FrontendPluginRegistry::Add`。

现在我们已经准备好构建和测试我们的 Clang 插件。

### 4.6.7 构建和运行插件代码

我们需要指定我们的 LLVM 项目的安装文件夹的路径。其余的步骤是标准的，我们之前已经使用过，参见图 3.11：

```cpp
export LLVM_HOME=<...>/llvm-project/install
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ..
ninja classchecker
```

**图 4.37**: Clang 插件的配置和构建命令

构建工件将位于`build`文件夹中。然后我们可以按照以下方式在测试文件上运行我们的插件，其中`<filepath>`是我们想要编译的文件：

```cpp
$ <...>/llvm-project/install/bin/clang -fsyntax-only\
                 -fplugin=./build/libclasschecker.so\
                 <filepath>
```

**图 4.38**: 在测试文件上运行 Clang 插件的方法

例如，如果我们使用一个名为`test.cpp`的测试文件，它定义了一个有三个方法的类（参见图 4.39），我们将不会收到任何警告。

```cpp
1 class Simple { 

2 public: 

3   void func1() {} 

4   void func2() {} 

5   void func3() {} 

6 };
```

**图 4.39**: Clang 插件的测试：test.cpp

然而，如果我们指定一个较小的阈值，我们将为该文件收到一个警告：

```cpp
$ <...>/llvm-project/install/bin/clang -fsyntax-only \
                 -fplugin-arg-classchecker-threshold=2 \
                 -fplugin=./build/libclasschecker.so \
                 test.cpp
test.cpp:1:7: warning: class Simple is too complex: method count = 3
    1 | class Simple {
      |       ^
1  warning generated.
```

**图 4.40**: 在 test.cpp 上运行的 Clang 插件

现在是时候为我们的插件创建一个 LIT 测试了。

### 4.6.8 Clang 插件的 LIT 测试

我们将从一个项目组织的描述开始。我们将采用在 Clang 源代码中使用的常见模式，并将我们的测试放在`test`文件夹中。这个文件夹将包含以下文件：

+   `lit.site.cfg.py.in`：这是主要的配置文件，一个 CMake 配置文件。它将标记为’@...@’的模式替换为 CMake 配置期间定义的相应值。此外，此文件加载`lit.cfg.py`。

+   `lit.cfg.py`：这是 LIT 测试的主要配置文件。

+   `simple_test.cpp`：这是我们的 LIT 测试文件。

基本的工作流程如下：CMake 将`lit.site.cfg.py.in`作为模板，并在`build/test`文件夹中生成相应的`lit.site.cfg.py`。然后，该文件被 LIT 测试用作种子来执行测试。

#### LIT 配置文件

对于 LIT 测试有两个配置文件。第一个显示在图 4.41 中。

```cpp
1 config.ClassComplexityChecker_obj_root = "@CMAKE_CURRENT_BINARY_DIR@" 

2 config.ClassComplexityChecker_src_root = "@CMAKE_CURRENT_SOURCE_DIR@" 

3 config.ClangBinary = "@LLVM_HOME@/bin/clang" 

4 config.FileCheck = "@FILECHECK_COMMAND@" 

5  

6 lit_config.load_config( 

7         config, os.path.join(config.ClassComplexityChecker_src_root, "test/lit.cfg.py"))
```

**图 4.41**：lit.site.cfg.py.in 文件

此文件是一个 CMake 模板，它将被转换为 Python 脚本。最重要的部分显示在*第 6-7 行*，其中加载了主要的 LIT 配置。它来自主源树，并且不会被复制到`build`文件夹中。

后续配置显示在图 4.42。这是一个包含 LIT 测试主要配置的 Python 脚本。

```cpp
1 # lit.cfg.py 

2 import lit.formats 

3  

4 config.name = ’classchecker’ 

5 config.test_format = lit.formats.ShTest(True) 

6 config.suffixes = [’.cpp’] 

7 config.test_source_root = os.path.dirname(__file__) 

8  

9 config.substitutions.append((’%clang-binary’, config.ClangBinary)) 

10 config.substitutions.append((’%path-to-plugin’, os.path.join(config.ClassComplexityChecker_obj_root, ’libclasschecker.so’))) 

11 config.substitutions.append((’%file-check-binary’, config.FileCheck))
```

**图 4.42**：lit.cfg.py 文件

*第 4-7 行*定义了基本配置；例如，*第 6 行*确定哪些文件应该用于测试。`test`文件夹中所有扩展名为`.cpp`的文件都将被用作 LIT 测试。

*第 9-11 行*详细说明了将在 LIT 测试中使用的替换。这包括 clang 二进制文件的路径（*第 9 行*）、带有插件的共享库的路径（*第 10 行*）以及`FileCheck`实用程序的路径（*第 11 行*）。

我们只定义了一个基本的 LIT 测试，`simple_test.cpp`，如图 4.43 所示。

```cpp
1 // RUN: %clang-binary -fplugin=%path-to-plugin -fsyntax-only %s 2>&1 | %file-check-binary %s 

2  

3 class Simple { 

4 public: 

5   void func1() {} 

6   void func2() {} 

7 }; 

8  

9 // CHECK: :[[@LINE+1]]:{{[0-9]+}}: warning: class Complex is too complex: method count = 6 

10 class Complex { 

11 public: 

12   void func1() {} 

13   void func2() {} 

14   void func3() {} 

15   void func4() {} 

16   void func5() {} 

17   void func6() {} 

18 };
```

**图 4.43**：simple_test.cpp 文件

可以在*第 1 行*观察到替换的使用，其中引用了 clang 二进制文件的路径、插件共享库的路径以及`FileCheck`实用程序的路径。在*第 9 行*使用了该实用程序识别的特殊模式。

最后一块拼图是 CMake 配置。这将设置在`lit.site.cfg.py.in`中进行替换所需的变量，并定义一个自定义目标来运行 LIT 测试。

#### LIT 测试的 CMake 配置

`CMakeLists.txt`文件需要一些调整以支持 LIT 测试。必要的更改显示在图 4.44 中。

```cpp
31   find_program(LIT_COMMAND llvm-lit PATH ${LLVM_BUILD}/bin) 

32   find_program(FILECHECK_COMMAND FileCheck ${LLVM_BUILD}/bin) 

33   if(LIT_COMMAND AND FILECHECK_COMMAND) 

34     message(STATUS "$LIT_COMMAND found: ${LIT_COMMAND}") 

35     message(STATUS "$FILECHECK_COMMAND found: ${FILECHECK_COMMAND}") 

36  

37     # Point to our custom lit.cfg.py 

38     set(LIT_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/test/lit.cfg.py") 

39  

40     # Configure lit.site.cfg.py using current settings 

41     configure_file("${CMAKE_CURRENT_SOURCE_DIR}/test/lit.site.cfg.py.in" 

42                    "${CMAKE_CURRENT_BINARY_DIR}/test/lit.site.cfg.py" 

43                    @ONLY) 

44  

45     # Add a custom target to run tests with lit 

46     add_custom_target(check-classchecker 

47                       COMMAND ${LIT_COMMAND} -v ${CMAKE_CURRENT_BINARY_DIR}/test 

48                       COMMENT "Running lit tests for classchecker clang plugin" 

49                       USES_TERMINAL) 

50   else() 

51     message(FATAL_ERROR "It was not possible to find the LIT executables at ${LLVM_BUILD}/bin") 

52   endif()
```

**图 4.44**：CMakeLists.txt 中的 LIT 测试配置

在*第 31 和 32 行*，我们搜索必要的工具，`llvm-lit`和`FileCheck`。值得注意的是，它们依赖于`$LLVM_BUILD`环境变量，我们在配置的第 12 行也进行了验证（参见图 4.32）。*第 41-43 行*中的步骤对于从提供的模板文件`lit.site.cfg.py.in`生成`lit.site.cfg.py`至关重要。最后，我们在*第 46-49 行*中建立了一个自定义目标来执行 LIT 测试。

现在我们已经准备好开始 LIT 测试。

#### 运行 LIT 测试

要启动 LIT 测试，我们必须设置一个环境变量，使其指向构建文件夹，编译项目，然后执行自定义目标`check-classchecker`。以下是这样做的方法：

```cpp
export LLVM_BUILD=<...>/llvm-project/build
export LLVM_HOME=<...>/llvm-project/install
rm -rf build; mkdir build; cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ..
ninja classchecker
ninja check-classchecker
```

**图 4.45**：Clang 插件的配置、构建和检查命令

执行这些命令后，您可能会看到以下输出：

```cpp
...
[2/2] Linking CXX shared module libclasschecker.so
[0/1] Running lit tests for classchecker clang plugin
-- Testing: 1 tests, 1 workers --
PASS: classchecker :: simple_test.cpp (1 of 1)

Testing Time: 0.12s
Passed: 1
```

**图 4.46**：LIT 测试执行

有了这个，我们完成了我们的第一个综合项目，该项目包含一个可以通过补充插件参数定制的实用 clang 插件。此外，它还包括可以执行以验证其功能的相应测试。

## 4.7 概述

在本章中，我们熟悉了 LLVM ADT 库中的基本类。我们了解了 Clang 诊断以及 LLVM 用于各种类型测试的测试框架。利用这些知识，我们创建了一个简单的 Clang 插件，用于检测复杂类并发出关于其复杂性的警告。

本章总结了本书的第一部分，其中我们获得了 Clang 编译器前端的初步知识。我们现在准备探索建立在 Clang 库基础上的各种工具。我们将从 Clang-Tidy 开始，这是一个强大的代码检查框架，用于检测 C++ 代码中的各种问题。

## 4.8 进一步阅读

+   LLVM 编码标准：[`llvm.org/docs/CodingStandards.html`](https://llvm.org/docs/CodingStandards.html)

+   LLVM 程序员手册：[`llvm.org/docs/ProgrammersManual.html`](https://llvm.org/docs/ProgrammersManual.html)

+   “Clang” CFE 内部手册：[`clang.llvm.org/docs/InternalsManual.html`](https://clang.llvm.org/docs/InternalsManual.html)

+   如何为你的类层次结构设置 LLVM 风格的 RTTI：[`llvm.org/docs/HowToSetUpLLVMStyleRTTI.html`](https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html)

+   LIT - LLVM 集成测试器：[`llvm.org/docs/CommandGuide/lit.html`](https://llvm.org/docs/CommandGuide/lit.html)
