

# 使用隐式内存管理编写泛型容器

在上一章中，我们在`Vector<T>`中编写了一个工作（尽管简单）的类似`std::vector<T>`类型的实现，以及在`ForwardList<T>`中编写了一个工作（尽管，再次，简单）的类似`std::forward_list<T>`类型的实现。不错！

在我们`Vector<T>`类型的情况下，经过最初的努力，我们得到了一个工作但有时效率不高的实现，然后我们努力将分配与构造分离，这样做减少了运行时所需的冗余工作量，但代价是更复杂的实现。在这个更复杂的实现中，我们区分了底层存储中已初始化的部分和未初始化的部分，并且当然，对这两部分都进行了适当的操作（将对象视为对象，将原始内存视为原始内存）。例如，我们使用赋值（以及使用赋值运算符的算法）来替换现有对象的内容，但更倾向于使用 placement `new`（以及依赖于此机制的算法）在原始内存中创建对象。

上一章中我们的`Vector<T>`实现是一个用大量源代码表达出的类。这种情况的原因之一是我们所进行的显式内存管理。确实，我们使`Vector<T>`对象负责管理底层内存块以及存储其中的对象，这种双重责任带来了成本。在本章中，我们将通过使内存管理*隐式*来重新审视这种设计，并将讨论这种新方法的影响。希望，亲爱的读者，这将引导你走向可能的简化和对编码实践的改进。

在本章中，我们的目标将是以下内容：

+   为了以这种方式适应手写的容器，如`Vector<T>`，从而显著简化其内存管理责任

+   为了理解我们的设计对源代码复杂性的影响

+   为了理解我们的设计对异常安全性的影响

我们将大部分精力放在重新审视`Vector<T>`容器上，但我们也会重新审视`ForwardList<T>`，看看我们是否可以将同样的推理应用于这两种容器类型。到本章结束时，至少在`Vector<T>`的情况下，我们仍然有一个手写的容器，它能够有效地管理内存并将原始内存与构造对象区分开来，但我们的实现将比我们在*第十二章*中产生的实现简单得多。

注意，关于`Vector<T>`，本章将比较两个版本。一个将被命名为“*天真*版本”，它将是使用底层存储中`T`类型对象的初始实现。另一个将被命名为“*复杂*版本”，它将考虑底层存储由两个（可能为空）的“部分”组成，`T`类型对象位于开始处，原始内存位于末尾。

# 技术要求

您可以在本书的 GitHub 仓库中找到本章的代码文件：[`github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter13`](https://github.com/PacktPublishing/C-Plus-Plus-Memory-Management/tree/main/chapter13)。

关于本章代码摘录的一些说明

本章将主要回顾和修改（希望简化！）*第十二章*中的代码示例，在过程中使用前几章（特别是*第五章*和*第六章*）中的想法。由于用于`Vector<T>`和`ForwardList<T>`的大部分代码不会改变，我们不会重写整个类，以避免不必要的重复。

相反，我们将专注于对那些类先前版本所做的最有意义的修改，有时会比较修改“之前”和“之后”的实现。当然，GitHub 仓库中的代码示例是完整的，可以用来“完善画面”。

# 为什么显式内存管理使我们的实现复杂化

让我们暂时看看`Vector<T>`的一个构造函数，正如在*第十二章*中所写的。为了简单起见，我们将使用接受元素数量和这些元素的初始值的构造函数。如果我们只限于`elems`指向`T`对象序列的简单版本，并暂时不考虑`elems`指向在开始处包含`T`对象和末尾包含原始内存的内存块的更复杂版本，我们就有以下内容：

```cpp
   // naïve version with elems of type T*
   Vector(size_type n, const_reference init)
      : elems{ new value_type[n] }, nelems{ n }, cap{ n } {
      try {
         std::fill(begin(), end(), init);
      } catch (...) {
         delete [] elems;
         throw;
      }
   }
// ...
```

此构造函数分配一个`T`对象的数组，通过一系列赋值初始化它们，处理异常等。`try`块及其相应的`catch`块是我们实现的一部分，但并非因为我们想处理`T`对象构造函数抛出的异常。实际上：如果我们不知道`T`是什么，我们怎么知道它可能会抛出什么异常呢？我们插入这些块是因为如果我们想避免泄漏，我们需要显式地分配和销毁数组。如果我们查看区分分配和构造的更复杂版本，情况会变得更加复杂：

```cpp
   // sophisticated version with elems of type T*
   Vector(size_type n, const_reference init)
      : elems{ static_cast<pointer>(
           std::malloc(n * sizeof(value_type))
        ) }, nelems{ n }, cap{ n } {
      try {
         std::uninitialized_fill(begin(), end(), init);
      } catch (...) {
         std::free(elems);
         throw;
      }
   }
// ...
```

正如我们所见，我们这样做是因为我们决定`Vector<T>`将是那个内存的*所有者*。我们完全有权利这样做！但是，如果我们让其他东西负责我们的内存会怎样呢？

# 使用智能指针的隐式内存管理

在 C++中，将我们的`Vector<T>`实现从手动管理内存更改为隐式管理内存的最简单方法是通过智能指针。这里的想法本质上是将`Vector<T>`的`elems`数据成员的类型从`T*`更改为`std::unique_ptr<T[]>`。我们将从两个角度来探讨这个问题：

+   这种变化如何影响`Vector<T>`的原始版本？作为提醒，我们来自*第十二章*的原始版本没有在底层存储中区分对象和原始内存，因此只存储对象。这导致了一个更简单的实现，但也是一个在许多场合不必要地构建对象，并且对于非平凡构造类型来说比更复杂的实现慢得多的实现。

+   这种变化如何影响避免了在实现上稍微复杂的情况下构建不必要的对象的性能陷阱的`Vector<T>`的复杂版本？

在这两种情况下，我们将检查一些成员函数，这些函数可以表明这种变化的影响。`Vector<T>`的原始和复杂实现的完整实现都可以在本书相关的 GitHub 仓库中查看和使用。

## 对原始`Vector<T>`实现的影响

如果我们的简化工作基于最初的、原始的*第十二章*版本，其中`elems`简单地指向一个连续的`T`对象序列，这将相当简单，因为我们可以改变：

```cpp
// naïve implementation, explicit memory management
// declaration of the data members...
pointer elems{};
size_type nelems{}, cap{};
```

…变为：

```cpp
// naïve implementation, implicit memory management
// declaration of the data members...
std::unique_ptr<value_type[]> elems;
size_type nelems{}, cap{};
```

…然后改变`begin()`成员函数的实现，如下所示：

```cpp
// naïve implementation, explicit memory management
iterator begin() {
   return elems; // raw pointer to the memory block
}
const_iterator begin() const {
   return elems; // raw pointer to the memory block
}
```

…更改为以下内容：

```cpp
// naïve implementation, implicit memory management
iterator begin() {
   return elems.get(); // raw pointer to the beginning
                       // of the underlying memory block
}
const_iterator begin() const {
   return elems.get(); // likewise
}
```

仅做这一点就足以显著简化`Vector<T>`类型的实现，因为释放内存将变得隐式。例如，我们可以通过完全删除异常处理来简化每个构造函数，例如，将以下实现更改为以下内容：

```cpp
// naïve implementation, explicit memory management
   Vector(size_type n, const_reference init)
      : elems{ new value_type[n] }, nelems{ n }, cap{ n } {
      try {
         std:: fill(begin(), end(), init);
      } catch (...) {
         delete [] elems;
         throw;
      }
   }
// ...
```

…变为这个显著更简单的版本：

```cpp
// naïve implementation, implicit memory management
   Vector(size_type n, const_reference init)
      : elems{ new value_type[n] }, nelems{ n }, cap{ n } {
      std:: fill(begin(), end(), init);
   }
// ...
```

这种简化的原因如下：

+   如果`Vector<T>`对象负责分配的内存，那么在调用析构函数时将隐式地删除数组，但是为了调用析构函数，需要有一个要销毁的对象：`Vector<T>`构造函数必须成功！这就解释了为什么我们需要捕获抛出的任何异常，手动删除数组，并重新抛出抛出的任何异常：直到达到析构函数的结束括号，没有`Vector<T>`对象可以销毁，所有资源管理都必须显式完成。

+   另一方面，如果 `elems` 是智能指针，那么一旦智能指针本身被构造，它就负责被指向的对象，这发生在 `Vector<T>` 构造函数的开括号之前。这意味着一旦 `elems` 被构造，如果发生异常导致构造函数退出，它将被销毁，从而释放即将成为 `Vector<T>` 对象的任务，不再需要销毁数组。为了明确：当我们到达 `Vector<T>` 构造函数的开括号时，`*this` 的数据成员已经被构造，因此，即使 `*this` 本身的构造没有完成，如果抛出异常，它们也将被销毁。C++ 的对象模型在这种情况下确实很奇妙。

亲爱的读者，你们中更有洞察力的人可能会注意到，即使你为一家不允许或反对使用异常的公司编写代码，使用智能指针所获得的异常安全性仍然存在。我们（隐式地）编写了异常安全的代码，而没有使用 `try` 或 `catch` 语句。

通过引入隐式内存管理来简化的一些其他示例包括移动操作和 `Vector<T>` 的析构函数，这将从以下内容变为：

```cpp
// naïve implementation, explicit memory management
   Vector(Vector &&other)
      : elems{ std::exchange(other.elems, nullptr) },
        nelems{ std::exchange(other.nelems, 0) },
        cap{ std::exchange(other.cap, 0) } {
   }
   Vector& operator=(Vector &&other) {
      Vector{ other }.swap(*this);
      return *this;
   }
   ~Vector() {
      delete [] elems;
   }
// ...
```

…简化如下：

```cpp
// naïve implementation, implicit memory management
   Vector(Vector&&) = default;
   Vector& operator=(Vector&&) = default;
   ~Vector() = default;
// ...
```

将移动操作设置为 `=default` 之所以有效，是因为类型 `std::unique_ptr` 在移动时“做正确的事情”，并将所有者的所有权从源传递到目的地。

需要注意的事情

通过将移动操作设置为 `=default`，我们在 `Vector<T>` 实现中引起了一点点语义变化。C++ 标准建议移动后的对象处于有效但未指定的状态，但并未详细说明“有效”的含义。我们编写的移动操作将移动后的对象恢复到与默认构造的 `Vector<T>` 对象等效的状态，但“默认”的移动操作将移动后的对象留下一个空的 `elems`，但可能具有非零的大小和容量。只要用户代码在使用移动后的对象之前将其重新赋值，这在实践中仍然有效，但这是一个值得认可的语义变化。

另一种有趣的简化方法将是实现 `resize()` 成员函数。在原始的、天真的 `Vector<T>` 实现中，我们有以下内容：

```cpp
// naïve implementation, explicit memory management
   void resize(size_type new_cap) {
      if(new_cap <= capacity()) return;
      auto p = new T[new_cap];
      if constexpr(std::is_nothrow_move_assignable_v<T>) {
         std::move(begin(), end(), p);
      } else try {
         std::copy(begin(), end(), p);
      } catch (...) {
         delete[] p;
         throw;
      }
      delete[] elems;
      elems = p;
      cap = new_cap;
   }
```

在这里，我们再次面临从 `T` 对象到 `T` 对象的复制赋值操作中抛出异常的可能性，并需要处理异常以避免资源泄露。从显式资源管理到隐式资源管理，我们得到以下内容：

```cpp
// naïve implementation, implicit memory management
   void resize(size_type new_cap) {
      if(new_cap <= capacity()) return;
      auto p = std::make_unique<value_type[]>(new_cap);
      if constexpr(std::is_nothrow_move_assignable_v<T>) {
         std::move(begin(), end(), p.get());
      } else {
         std::copy(begin(), end(), p.get());
      }
      elems.reset(p.release());
      cap = new_cap;
   }
```

如您所见，整个异常处理代码已经消失。对象 `p` 拥有新的数组，并在函数执行结束时销毁它。一旦复制（或移动，取决于类型 `T` 的移动赋值是否标记为 `noexcept`）完成，`elems` 通过 `reset()` 放弃之前拥有的数组（同时销毁它）并通过 `release()` “窃取”由 `p` 释放的数组所有权。请注意，编写 `elems = std::move(p);` 会有类似的效果。

将这个简化过程应用到 `Vector<T>` 中，源代码逐渐减少，对于一个像 `Vector<T>` 的原始版本这样的容器，它只包含对象，没有底层存储末尾的原始内存块，我们可以节省大约 25%的源代码行数（对于这个学术实现，从大约 180 行减少到 140 行）。试试看，看看你自己能发现什么！

## 对复杂的 Vector<T> 实现的影响

将相同的技巧应用到更复杂的 `Vector<T>` 上将需要更多的工作，因为 `std::unique_ptr<T[]>` 类型对象的析构函数的默认行为是将 `operator delete[]` 应用于它拥有的指针。正如我们所知，我们的复杂实现可以概念化为由两个（可能为空）的“部分”组成：一个由 `T` 对象手动放置到原始内存中的初始部分，后面跟着一个未初始化的、没有对象的原始内存部分。因此，我们需要以不同的方式处理每个“部分”。

我们仍然会使用 `std::unique_ptr<T[]>` 对象来管理内存，但我们需要使用一个 `自定义删除器` 对象（在*第五章*和*第六章*中讨论过）来考虑我们实现的特定细节。这个对象需要了解它将伴随的 `Vector<T>` 对象的运行时状态，因为它必须知道底层存储的每个“部分”从哪里开始以及在哪里结束，而这些都是随着代码执行而变化的。

这个实现的一个重要观点，这是一个反复出现但可能我们没有足够坚持的观点，是我们希望我们的实现向客户端代码暴露相同的接口，无论实现有何变化。这有时可能是不可能的或不合理的，但无论如何，这是一个有意义且值得追求的目标。这包括我们选择内部公共类型：例如，我们使用智能指针来管理底层内存的事实并不改变指向元素的指针是一个 `T*` 的事实：

```cpp
// ...
template <class T>
class Vector {
public:
   using value_type = T;
   using size_type = std::size_t;
   using pointer = T*;
   using const_pointer = const T*;
   using reference = T&;
   using const_reference = const T&;
   // ...
```

现在，由于我们希望将 `elems` 定义为一个智能指针，它拥有并管理底层存储，而不是一个原始指针，因此我们需要定义一个将被该智能指针使用的自定义删除器。

这个问题的一个重要方面是，自定义删除器需要知道`Vector<T>`对象的状态，以便知道底层存储中哪些部分持有对象。因此，我们的`std::unique_ptr<T[]>`的自定义删除器将是状态化的，并存储一个名为`source`的`Vector<T>`对象的引用。通过`source`，`deleter`对象的函数调用操作符将能够访问容器中的对象序列（从`source.begin()`到`source.end()`的半开序列）并在释放底层存储之前`destroy()`这些对象：

```cpp
   // ...
private:
   struct deleter {
      Vector& source;
      void operator()(value_type* p) {
         std::destroy(std::begin(source),
                      std::end(source));
         std::free(static_cast<void*>(p));
      }
   };
   std::unique_ptr<value_type[], deleter> elems;
   size_type nelems{},
             cap{};
   // ...
```

`elems`数据成员知道自定义删除器的类型将是`deleter`，但实际上将扮演删除器角色的对象必须知道它将与之交互的`Vector<T>`对象。`Vector<T>`的构造函数将负责提供此信息，并且我们需要小心地实现我们的移动操作，以确保我们不会传递删除器对象的状态并使我们的代码不一致。

如同在简单版本中提到的，我们需要调整`begin()`成员函数，以考虑到`elems`是一个智能指针，但我们的`iterator`接口依赖于原始指针：

```cpp
   // ...
   using iterator = pointer;
   using const_iterator = const_pointer;
   iterator begin() { return elems.get(); }
   const_iterator begin() const { return elems.get(); }
   // ...
```

我们的构造函数需要适应这样一个事实，即我们有一个自定义删除器，它会在发生任何不良情况或程序正常结束时进行清理。以下是`Vector<T>`构造函数的三个示例：

```cpp
   // ...
   constexpr Vector()
      : elems{ nullptr, deleter { *this } } {
   }
   Vector(size_type n, const_reference init)
      : elems{ static_cast<pointer>(
           std::malloc(n * sizeof(value_type))
        ), deleter{ *this }
      } {
      std::uninitialized_fill(begin(), begin() + n, init);
      nelems = cap = n;
   }
   Vector(Vector&& other) noexcept
      : elems{ std::exchange(
           other.elems.release()), deleter{ *this }
        },
        nelems{ std::exchange(other.nelems, 0) },
        cap{ std::exchange(other.cap, 0) } {
   }
   // ...
```

请注意，我们在这里没有使用`=default`来表示移动构造函数，因为我们不希望传递自定义删除器，我们的实现已经将此对象与特定的`Vector<T>`对象关联起来。

在这里需要一个小注解：我们在`deleter`对象的构造函数中传递了`*this`，但是我们在`*this`的构造完成之前就进行了这一操作，所以`deleter`对象在`*this`构造完成（在其构造函数的闭合括号之前）所做的任何操作都值得注意和关注。

在我们的情况下，如果类型`T`的对象的构造函数抛出异常，`deleter`对象将发挥作用。我们需要确保在`deleter`对象可能介入的情况下，`*this`的数据成员的值始终保持一致。

在我们的情况下，由于`begin()`和`end()`成员函数返回定义对象半开范围的迭代器，并且正如我们所知，`std::uninitialized_fill()`调用构造函数（如果抛出异常）则销毁已构造的对象，我们必须确保`nelems==0`直到所有对象都构造完成。请注意，我们定义了从`begin()`到`begin()+n`的范围进行初始化，并在调用`std::uninitialized_fill()`之后改变`nelems`：这样，如果抛出异常，则`begin()==end()`，并且`deleter`对象不会尝试销毁“非对象”。

类`Vector<T>`的其他构造函数同样被简化；我们在这里不会展示它们，所以请将它们视为不那么令人畏惧的“留给读者的练习。”

`Vector<T>`的简化通过一些现在需要我们付出很少或几乎不需要努力的特设成员函数变得明显。在这方面值得注意的是析构函数，现在它可以被默认；如本节前面提到的移动构造函数，我们不默认移动赋值操作，以避免转移自定义删除器的内部状态，如下面的代码片段所示：

```cpp
   // ...
   ~Vector() = default;
   void swap(Vector& other) noexcept {
      using std::swap;
      swap(elems, other.elems);
      swap(nelems, other.nelems);
      swap(cap, other.cap);
   }
   Vector& operator=(const Vector& other) {
      Vector{ other }.swap(*this);
      return *this;
   }
   Vector& operator=(Vector&& other) {
      Vector{ std::move(other) }.swap(*this);
      return *this;
   }
   reference operator[](size_type n) { return elems[n]; }
   const_reference operator[](size_type n) const {
      return elems[n];
   }
```

成员函数`swap()`和`operator[]`已被证明表明`std::unique_ptr<T[]>`在许多方面表现得像`T`对象的“常规”数组。`Vector<T>`的许多其他成员函数保持不变，例如`front()`、`back()`、`operator==()`、`operator!=()`、`grow()`、`push_back()`和`emplace_back()`。请参阅*第十二章*以了解这些函数的详细信息。

通过使用智能指针，`reserve()`和`resize()`函数也可以简化，因为我们可以消除显式的异常管理，同时由于`std::unique_ptr<T[]>`是一个**RAII**类型，它会为我们处理内存，所以我们仍然保持异常安全。

在`reserve()`的情况下，我们现在使用智能指针`p`来持有分配的内存，然后将`elems`中的对象通过`move()`或`copy()`操作移动到`p`。一旦完成这些操作，我们就`destroy()`掉`elems`中剩余的对象，之后`p`放弃其指针并将其转移到`elems`，剩下的唯一事情就是更新容器的容量：

```cpp
   // ...
   void reserve(size_type new_cap) {
      if (new_cap <= capacity()) return;
      std::unique_ptr<value_type[]> p{
         static_cast<pointer>(
            std::malloc(new_cap * sizeof(T))
         )
      };
      if constexpr (std::is_nothrow_move_assignable_v<T>) {
         std::uninitialized_move(begin(), end(), p.get());
      } else {
         std::uninitialized_copy(begin(), end(), p.get());
      }
      std::destroy(begin(), end());
      elems.reset(p.release());
      cap = new_cap;
   }
```

在`resize()`的情况下，我们现在使用智能指针`p`来持有分配的内存，然后将`elems`中的对象通过`move()`或`copy()`操作移动到`p`，并在内存块的剩余部分构造默认的`T`对象。一旦完成这些操作，我们就`destroy()`掉`elems`中剩余的对象，之后`p`放弃其指针并将其转移到`elems`，剩下的唯一事情就是更新容器的容量：

```cpp
   // ...
   void resize(size_type new_cap) {
      if (new_cap <= capacity()) return;
      std::unique_ptr<value_type[]> p =
         static_cast<pointer>(
            std::malloc(new_cap * sizeof(T))
         );
      if constexpr (std::is_nothrow_move_assignable_v<T>) {
         std::uninitialized_move(begin(), end(), p.get());
      } else {
         std::uninitialized_copy(begin(), end(), p.get());
      }
      std::uninitialized_fill(
         p.get() + size(), p.get() + new_cap, value_type{}
      );
      std::destroy(begin(), end());
      elems.reset(p.release());
      nelems = cap = new_cap;
   }
   // ...
```

所有这一切的魔力，或者说，可以这样讲，就是我们的其他成员函数，如`insert()`和`erase()`，是建立在基本抽象如`reserve()`、`begin()`、`end()`等之上的，这意味着它们不需要修改以考虑这种表示变化。

# 重新设计的后果

这种“重新设计”的后果是什么？它们在过程中已经提到，但让我们总结一下：

+   对于用户代码来说，后果基本上没有：`Vector<T>` 类型的对象在内存管理实现中占据相同的空间，与显式内存管理实现（其中自定义删除器是状态化的）几乎占据相同的空间，并且每个都公开了相同的接口。

+   由于在 *第五章* 中解释的原因，基本上没有速度成本：在除基本、专为调试编译的优化级别之外编译的代码中，通过 `std::unique_ptr<T>` 进行操作，由于函数内联调用，将导致与通过 `T*` 进行操作一样高效的代码。

+   实现变得更加简单：指令更少，没有显式的异常处理代码，更多的成员函数可以省略默认值…

+   这个隐式内存管理实现的 重要方面在于，即使在没有显式的 `try` 和 `catch` 块的情况下，它也是异常安全的。这可能在许多情况下都会产生影响：例如，你可能处于不允许异常的情况，但发现自己正在使用一个可能抛出异常的库…或者可以在内存受限的情况下简单地调用 `operator new()`。在我们的隐式内存管理实现中，在这种情况下将是安全的，但一个采用手动内存管理方法且没有异常处理代码的实现则不会这么“幸运”。

在 `Vector<T>` 中实现自定义删除器的努力似乎是一个值得的投资。现在，你可能想知道这种情况是否与基于节点的容器相似，因此我们将通过回顾 *第十二章* 中的原始 `ForwardList<T>` 实现来探索这个问题。

# 推广到 ForwardList<T>？

我们现在知道我们可以调整 `Vector<T>` 的实现，将其从显式内存管理模型转换为隐式模型，并且这样做有很多优点。将同样的方法应用于其他容器很诱人，但在开始这样的冒险之前，分析问题可能更明智。

我们在 *第十二章* 中实现了名为 `ForwardList<T>` 的具有显式内存管理的基于节点的容器。尝试改变这个容器的实现以使其更加隐式会有什么影响？

## 尝试 - 使每个节点对其后续节点负责

在我们探索如何使基于节点的容器中的内存管理更加隐式的方法中，一个可能的方法是改变 `ForwardList<T>::Node` 的定义，使得 `next` 数据成员变为 `std::unique_ptr<Node>` 而不是 `Node*`。

概括来说，我们会得到以下结果：

```cpp
template <class T>
class ForwardList {
public:
   // ...
private:
   struct Node {
      value_type value;
      std::unique_ptr<Node> next; // <--
      Node(const_reference value) : value{ value } {
      }
      Node(value_type&& value) : value{ std::move(value) }{
      }
   };
   Node* head{};
   size_type nelems{};
   // ...
```

初看起来，这似乎是一个改进，因为它将`ForwardList<T>`析构函数简化为以下内容：

```cpp
   // ...
   ~ForwardList() {
      delete head; // <-- lots of work starts here!
   }
   // ...
```

这种简化将引发一种“多米诺效应”：由于节点的`next`数据成员成为列表中其后继节点的所有者，并且这对于链中的每个节点（除了`head`本身）都是真的，因此销毁第一个节点确保了其后继节点以及其后继节点的销毁，依此类推。

这种明显的简化隐藏了一个棘手的事实：在这个实现下调用`delete head;`时，*我们可能会引发堆栈溢出*。确实，我们用一个本质上相当于递归调用的东西替换了逐个节点应用`delete`的循环，这意味着对堆栈使用的影响从固定变为与列表中节点数量成比例。这确实是个不愉快的消息！

在这个阶段，亲爱的读者，你可能正在想：“嗯，我本来只是打算用这个`ForwardList<T>`类型来处理小列表，所以我不担心。”如果这反映了你的思考方式，那么也许我们应该探索一下在`ForwardList<T>`类中这个实现决策的其他潜在影响。

其中一个影响是迭代器会变得稍微复杂一些：我们不希望遍历节点的迭代器成为该节点的唯一所有者。这确实会破坏结构，因为当我们在列表中遍历节点时，节点会被销毁。因此，`ForwardList<T>::Node<U>`（其中`U`是`T`或`const T`）仍然有一个`T*`数据成员，这意味着例如`operator++()`需要获取每个节点中`std::unique_ptr<T>`数据成员的底层指针：

```cpp
   // ...
   template <class U> class Iterator {
   public:
      // ...
   private:
      Node* cur{};
   public:
      // ...
      Iterator& operator++() {
         cur = cur->next.get(); // <--
         return *this;
      }
   // ...
```

这只是略微增加了复杂性，但并不是无法管理的。

在*第十二章*中，我们将大多数`ForwardList<T>`构造函数收敛到更通用的序列构造函数，该构造函数接受一对某种类型`It`的前向迭代器作为参数。这个构造函数将变得部分复杂，因为现在连接节点需要我们知道每个节点内部使用了智能指针，但抛出异常时的清理只需要删除头节点并让上述“多米诺效应”发生：

```cpp
   // ...
   template <std::forward_iterator It>
   ForwardList(It b, It e) {
      try {
         if (b == e) return;
         head = new Node{ *b };
         auto q = head;
         ++nelems;
         for (++b; b != e; ++b) {
            q->next = std::make_unique<Node>(*b); // <--
            q = q->next.get(); // <--
            ++nelems;
         }
      } catch (...) {
         delete head; // <--
         throw;
      }
   }
   // ...
```

`ForwardList<T>`的大多数成员函数将保持不变。例如，`push_front()`这样的操作会有细微的调整：

```cpp
   // ...
   void push_front(const_reference val) {
      auto p = new Node{ val };
      p->next = std::unique_ptr<Node>{ head }; // <--
      head = p;
      ++nelems;
   }
   void push_front(T&& val) {
      auto p = new Node{ std::move(val) };
      p->next = std::unique_ptr<Node>{ head }; // <--
      head = p;
      ++nelems;
   }
   // ...
```

如所见，我们需要区分使用`head`数据成员的代码和使用链中其他节点的代码。类似的调整将适用于任何修改列表结构的成员函数，包括，值得注意的是，插入和删除操作。

一个更有趣、也许更有启发的成员函数将是`insert_after()`成员函数，它在列表中给定迭代器之后插入一个元素。让我们详细看看这个函数：

```cpp
   // ...
   iterator
      insert_after(iterator pos, const_reference value) {
      auto p = std::make_unique<Node>(value); // <-- A
      p->next.reset(pos.cur->next.get()); // <-- B
      pos.cur->next.release(); // <-- C
      pos.cur->next.reset(p.get()); // <-- D
      p.release(); // <-- E
      ++nelems;
      return { pos.cur->next.get() }; // <-- F
   }
   // ...
```

嗯，这是相当多的更新文本！这个函数怎么会变得这么复杂？看看“字母注释”，我们有以下内容：

+   在行 *A* 上，我们为要插入的值创建一个名为 `p` 的 `std::unique_ptr<Node>` 对象。我们知道新创建的节点不会是列表中的第一个节点，因为函数是 `insert_after()`，需要一个指向现有“之前”节点的迭代器（在这里命名为 `pos`），所以这是有意义的。同样地，我们也知道 `pos` 不是 `end()`，根据定义，它不会指向容器中的有效节点。

+   在行 *B* 上，我们做必要的操作，使 `p` 的后继成为 `pos` 的后继。这需要一些小心，因为 `pos.cur->next` 保证是一个 `std::unique_ptr<Node>`（显然它不能是 `head`，因为 `pos.cur` 是“在” `pos.cur->next` 之前），我们使 `p` 成为一个 `std::unique_ptr<Node>`。我们正在将 `pos.cur` 的后继节点的责任转移到 `p->next`，实际上是在 `p` 之后插入 `pos->next`（尽管方式复杂）。

+   在线 *C* 上，我们确保 `pos.cur` 放弃对 `pos.cur->next` 的责任。这是很重要的，因为我们如果不这样做，那么替换那个 `std::unique_ptr<Node>` 将会破坏其指针指向的对象。行 *B* 确保了 `pos.cur->next` 和 `p->next` 将指向同一个对象，如果我们就此停止（两个对象负责同一个指针指向是一个我们不希望出现的语义问题）。

+   一旦 `pos.cur->next` 被断开连接，我们就转到行 *D*，在那里让它指向 `p` 下的原始指针。这又会再次导致对 `Node` 的共享责任，所以我们继续到行 *E*，在那里将 `p` 从其基础指针断开连接。

+   行 *F* 通过返回一个指向原始（因此非所有者）指针的预期迭代器来结束这个函数的工作。

那是…复杂的。这个函数之所以复杂的主要原因是我们在这个函数中的大部分努力都是所有权的转移。毕竟，`std::unique_ptr<T>` 对象代表了对 `T*` 的唯一所有权，在一个链表中，每个插入或删除操作都需要移动指针，从而在节点之间转移所有权。我们通过在类型的大多数操作中增加复杂性来简化偶尔的情况（节点的删除）。那是…悲哀的。

关于意义和责任语义

智能指针都是关于在类型系统中编码意义和责任。简化用户代码很重要，但这不是这些类型的主要目的。在 `ForwardList<T>` 对象中，`T` 对象的真正所有者是 `ForwardList<T>` 对象，而 `ForwardList<T>::Node<U>` 对象（从 `ForwardList<T>` 对象的角度来看）基本上是一个存储设施。尝试改变这一点可以使其工作，但随之而来的复杂性表明有些可疑。

当编写一个类，尤其是容器类时，我们必须清楚地了解每种类型的预期角色。我们知道迭代器本质上是非拥有的（然而，在某些用例中，我们可以设想`shared_ptr<T>`对象与指针共同拥有）。至于容器及其底层表示，重要的是每种类型的责任需要明确，如果我们的设计要可管理。

好吧，所以让节点负责其后续节点并没有奏效。仅仅让`ForwardList<T>`对象的`head`成员负责列表中的其他节点，会让我们过得更好吗？

## 尝试：让头指针负责其他节点

如前节所述，让每个节点负责其后续节点在语义上是不正确的。这会导致复杂、繁琐且容易出错的代码，而通过这种转换简化的实现方面通常被其他地方增加的复杂性所抵消。

也许仅仅让`head`节点成为一个`std::unique_ptr<Node>`对象，并使用一个自定义删除器负责删除整个列表会更有益？嗯，我们可以肯定地尝试这种方法。

作为摘要，我们现在会得到以下内容：

```cpp
template <class T>
class ForwardList {
   // ...
   struct Node {
      value_type value;
      Node* next = nullptr;
      Node(const_reference value) : value{ value } {
      }
      Node(value_type&& value) : value{ std::move(value) }{
      }
   };
   struct deleter { // <--
      void operator()(Node* p) const {
         while (p) {
            Node* q = p->next;
            delete p;
            p = q;
         }
      }
   };
   std::unique_ptr<Node, deleter> head;
ForwardList<T> type that, when an object of that type is destroyed, implicitly ensures that the nodes in the list are destructed. The entire list remains built from raw pointers, such that nodes are not responsible for memory management, which is probably an upgrade from the previous attempt.
With this implementation, we would get a defaulted `ForwardList<T>` destructor, which is a good thing. There would be a tiny complexity increase in `clear()` where we need to distinguish the `head` smart pointer from the underlying pointer:

```

// ...

void clear() noexcept {

for (auto p = head.get(); p; ) { // <--

auto q = p->next;

delete p;

p = q;

}

nelems = 0;

}

// ...

```cpp

 The iterator interface needs to be adapted somewhat since `head` is not a `Node*` anymore, but iterators trade in non-owning resources:

```

// ...

iterator begin() { return { head.get() }; } // <--

const_iterator begin() const {

return { head.get() }; // <--

}

// ...

```cpp

 The `ForwardList<T>` constructor that takes a pair of iterators and towards which most other constructors converge requires slight modifications:

```

// ...

template <std::forward_iterator It>

ForwardList(It b, It e) {

if(b == e) return;

head.reset(new Node{ *b }); // <--

auto q = head.get(); // <--

++nelems;

for(++b; b != e; ++b) {

q->next = new Node{ *b };

q = q->next;

++nelems;

}

}

// ...

```cpp

 The exception handling side of this member function is indeed simplified, being made implicit from the fact that, should any constructor of a `T` object throw an exception, the previously created nodes will be destroyed.
As in the previous version, our `push_front()` member functions will require some adjustment as they interact with the `head` data member:

```

// ...

void push_front(const_reference val) {

auto p = new Node{ val };

p->next = head.get(); // <--

head.release(); // <--

head.reset(p); // <--

++nelems;

}

void push_front(T&& val) {

auto p = new Node{ std::move(val) };

p->next = head.get(); // <--

head.release(); // <--

head.reset(p); // <--

++nelems;

}

// ...

```cpp

 On the upside, no member function that does not interact with the `head` data member requires any modification.
Is this “implicitness” worth it? It probably depends on the way in which you approach writing code. We did gain something of value in implicit exception safety. There is value in separating concerns, and this implementation does free the container from the task of managing memory (for the most part). It is up to you, dear reader, to determine whether the reduced complexity “here” outweighs the added complexity “there.”
Summary
In this chapter, we reexamined containers written in *Chapter 12*, seeking to use implicit memory management tools in such a way as to make our implementations simpler and safer. We did reach an improvement in `Vector<T>` but the results obtained with our node-based `ForwardList<T>` container were… not absent, but arguably less conclusive depending on your perspective.
In the next chapter, we will introduce the idea of allocators, objects that inform containers as to how memory should be obtained or liberated, and examine how they impact the ways in which we write code.

```
