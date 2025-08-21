# 第三章：内存管理和异常安全

C++与 C 编程语言有很高的兼容性。C++保留了指针来表示和访问特定的内存地址，并通过`new`和`delete`运算符提供了手动内存管理原语。您还可以无缝地从 C++访问 C 标准库函数和大多数主要操作系统的 C 系统调用或平台 API。自然地，C++代码经常处理对各种 OS 资源的*句柄*，如堆内存、打开的文件、套接字、线程和共享内存。获取这些资源并未能释放它们可能会对您的程序产生不良后果，表现为隐匿的错误，包括内存泄漏和死锁。

在本章中，我们将探讨使用**智能指针**封装动态分配对象的指针的方法，以确保在不再需要时它们会自动释放。然后我们将这些技术扩展到非内存资源。在这个过程中，我们将理解什么是异常安全的代码，并使用智能指针来编写这样的代码。

这些主题分为以下几个部分：

+   动态内存分配和异常安全

+   智能指针

+   唯一所有权语义

+   共享所有权语义

在本章的某些部分，您将需要使用支持 C++11 的编译器。这将在各个部分中附加说明。

# 动态内存分配和异常安全

想象一下，您需要编写一个程序来旋转图像。您的程序接受文件名和旋转角度作为输入，读取文件的内容，执行处理，并返回输出。以下是一些示例代码。

```cpp
 1 #include <istream>
 2 #include <fstream>
 3 typedef unsigned char byte;
 4 
 5 byte *rotateImage(std::string imgFile, double angle, 
 6                   size_t& sz) {
 7   // open the file for reading
 8   std::ifstream imgStrm(imgFile.c_str(), std::ios::binary);
 9 
10   if (imgStrm) {
11     // determine file size
12     imgStrm.seekg(0, std::ios::end);
13     sz = imgStrm.tellg();
14     imsStrm.seekg(0);        // seek back to start of stream
15
16     byte *img = new byte[sz]; // allocate buffer and read
17     // read the image contents
18     imgStrm.read(reinterpret_cast<char*>(img), sz);
19     // process it
20     byte *rotated = img_rotate(img, sz, angle);
21     // deallocate buffer
22     delete [] img;
23 
24     return rotated;
25   }
26 
27   sz = 0;
28   return 0;
29 }
```

旋转图像的实际工作是由一个名为`img_rotate`的虚构的 C++ API 完成的（第 20 行）。`img_rotate`函数接受三个参数：图像内容作为字节数组，数组的大小以非 const 引用的形式，以及旋转角度。它返回旋转后图像的内容作为动态分配的字节数组。通过作为第三个参数传递的引用返回该数组的大小。这是一个不完美的代码，更像是 C 语言。这样的代码在“野外”中非常常见，这就是为什么了解它的缺陷很重要。因此，让我们来剖析一下问题。

为了读取图像文件的内容，我们首先确定文件的大小（第 12-13 行），然后分配一个足够大的字节数组`img`来容纳文件中的所有数据（第 16 行）。我们读取图像内容（第 18 行），并在通过调用`img_rotate`进行图像旋转后，删除包含原始图像的缓冲区`img`（第 22 行）。最后，我们返回旋转后的图像的字节数组（第 24 行）。为简单起见，我们没有检查读取错误（第 18 行）。

在前面的代码中有两个明显的问题。如果图像旋转失败（第 19 行）并且`img_rotate`抛出异常，那么`rotateImage`函数将在不释放字节缓冲区`img`的情况下返回，这样就会*泄漏*。这是一个明显的例子，说明在面对异常时代码的行为不佳，也就是说，它不是*异常安全*的。此外，即使一切顺利，该函数也会返回旋转后的缓冲区（第 24 行），这本身是动态分配的。因此，我们完全将其释放的责任留给调用者，没有任何保证。我们应该做得更好。

还有一个不太明显的问题。 `img_rotate`函数应该已经记录了它如何分配内存，以便我们知道如何释放它——通过调用数组删除（`delete []`）运算符（第 22 行）。但是，如果开发`img_rotate`找到了更有效的自定义内存管理方案，并希望在下一个版本中使用呢？他们会避免这样做；否则，所有客户端代码都会中断，因为`delete []`运算符可能不再是正确的释放内存的方式。理想情况下，`img_rotate` API 的客户端不应该为此烦恼。

## 异常安全和 RAII

在前面的例子中，我们非正式地看了一下异常安全的概念。我们看到`img_rotate` API 可能抛出的潜在异常可能会在`rotateImage`函数中泄漏资源。事实证明，您可以根据一组标准称为**The Abrahams Exception Safety Guarantees**来推断代码在面对异常时的行为。它们以 Dave Abrahams 的名字命名，他是 Boost 的联合创始人和杰出的 C++标准委员会成员，他在 1996 年正式化了这些保证。此后，它们已经被其他人进一步完善，包括特别是 Herb Sutter，并列在下面：

+   **基本保证**：中途终止的操作保留不变，并且不会泄漏资源

+   **强保证**：中途终止的操作不会产生任何影响，即操作是原子的

+   **无异常保证**：无法失败的操作

不满足这些标准的操作被称为“不安全的异常”或更通俗地说，不安全的异常。操作的适当异常安全级别是程序员的特权，但不安全的异常代码很少被接受。

用于使代码具有异常安全性的最基本和有效的 C++技术是名为**Resource Acquisition is Initialization**（**RAII**）的奇特名称。 RAII 习惯提出了封装需要手动管理的资源的以下模型：

1.  在包装对象的构造函数中封装资源获取。

1.  在包装对象的析构函数中封装资源释放。

1.  此外，为包装对象定义一致的复制和移动语义，或者禁用它们。

如果包装对象是在堆栈上创建的，则其析构函数也会在正常范围退出以及由于异常退出时调用。否则，包装对象本身应该由 RAII 习惯管理。粗略地说，您可以在堆栈上创建对象，也可以使用 RAII 来管理它们。在这一点上，我们需要一些例子，然后我们可以直接回到图像旋转示例并使用 RAII 进行修复：

```cpp
 1 struct ScopeGuard
 2 {
 3   ScopeGuard(byte *buffer) : data_(buffer) {}
 4   ~ScopeGuard() { delete [] data_; }
 5
 6   byte *get() { return data_; }
 7 private:
 8   byte *data_;
 9 };
10 
11 byte *rotateImage(std::string imgFile, double angle, size_t& sz)
12 {
13   // open the file for reading
14   std::ifstream imgStrm(imgFile.c_str(), std::ios::binary);
15 
16   if (imgStrm) {
17     // determine file size
18     imgStrm.seekg(0, std::ios::end);
19     sz = imgStrm.tellg();
20     imgStrm.seekg(0);
21
22     // allocate buffer and read
23     ScopeGuard img(new byte[sz]);
24     // read the image contents
25     imgStrm.read(reinterpret_cast<char*>(img.get()), sz);
26     // process it
27     return img_rotate(img.get(), sz, angle);
28   } // ScopeGuard destructor
29 
30   sz = 0;
31   return 0;
32 }
```

前面的代码是一个谦虚的尝试，使`rotateImage`函数在`img_rotate`函数本身是异常安全的情况下是异常安全的。首先，我们定义了一个名为`ScopeGuard`（第 1-9 行）的`struct`，用于封装由数组`new operator`分配的字符数组。它以分配的数组指针作为其构造函数参数，并将数据成员`data_`设置为该指针（第 3 行）。它的析构函数使用数组`delete`运算符（第 4 行）释放其`data_`成员指向的数组。`get`成员函数（第 6 行）提供了一种从`ScopeGuard`对象获取底层指针的方法。

在`rotateImage`函数内部，我们实例化了一个名为`img`的`ScopeGuard`对象，它包装了使用数组`new`运算符分配的字节数组（第 23 行）。我们调用打开文件流的`read`方法，并将`img`的`get`方法获取的原始字节数组传递给它（第 25 行）。我们假设读取总是成功的，但在生产代码中，我们应该始终进行适当的错误检查。最后，我们调用`img_rotate` API 并返回它返回的旋转图像（第 27 行）。当我们退出作用域时，`ScopeGuard`析构函数被调用，并自动释放封装的字节数组（第 28 行）。即使`img_rotate`抛出异常，`ScopeGuard`析构函数仍将在堆栈展开的过程中被调用。通过使用`ScopeGuard`类的 RAII，我们能够声明`rotateImage`函数永远不会泄漏包含图像数据的缓冲区。

另一方面，由`rotateImage`返回的旋转图像的缓冲区*可能*会泄漏，除非调用者注意将其分配给指针，然后以异常安全的方式释放它。`ScopeGuard`类在其当前形式下并不适用。事实证明，Boost 提供了不同类型的智能指针模板来解决这些问题，值得理解这些智能指针以及它们帮助解决的资源获取模式和异常安全问题。

# 智能指针

智能指针，明确地说，是一个封装指针访问并经常管理与指针相关的内存的类。如果你注意到了，你会注意到智能指针与菠萝的相似之处——智能指针是类，而不是指针，就像菠萝并不是真正的苹果一样。摆脱水果类比，不同类型的智能指针通常具有额外的功能，如边界检查、空指针检查和访问控制等。在 C++中，智能指针通常重载解引用运算符（`operator->`），这允许使用`operator->`在智能指针上调用的任何方法调用都绑定到底层指针上。

Boost 包括四种不同语义的智能指针。此外，由于 C++经常使用指针来标识和操作对象数组，Boost 提供了两种不同的智能数组模板，它们通过指针封装了数组访问。在接下来的章节中，我们将研究 Boost 中不同类别的智能指针及其语义。我们还将看看`std::unique_ptr`，这是一个 C++11 智能指针类，它取代了 Boost 的一个智能指针，并支持 Boost 中不容易获得的语义。

## 独占所有权语义

考虑以下代码片段来实例化一个对象并调用其方法：

```cpp
 1 class Widget;
 2 
 3 // …
 4 
 5 void useWidget()
 6 {
 7   Widget *wgt = new Widget;
 8   wgt->setTitle(...);
 9   wgt->setSize(...);
10   wgt->display(...);
11   delete wgt;
12 }
```

正如我们在前一节中看到的，前面的代码并不具有异常安全性。在动态内存上构造`Widget`对象（第 7 行）之后和销毁`Widget`对象（第 11 行）之前抛出的异常可能导致为`Widget`对象动态分配的内存泄漏。为了解决这个问题，我们需要类似于我们在前一节中编写的`ScopeGuard`类，而 Boost 则提供了`boost::scoped_ptr`模板。

### boost::scoped_ptr

以下是使用`scoped_ptr`修复的前面的示例。`scoped_ptr`模板可以从头文件`boost/scoped_ptr.hpp`中获得。它是一个仅包含头文件的库，你不需要将你的程序链接到任何其他库：

**清单 3.1：使用 scoped_ptr**

```cpp
 1 #include <boost/scoped_ptr.hpp>
 2 #include "Widget.h"  // contains the definition of Widget
 3 
 4 // …
 5 
 6 void useWidget()
 7 {
 8   boost::scoped_ptr<Widget> wgt(new Widget);
 9   wgt->setTitle(...);
10   wgt->setSize(...);
11   wgt->display(...);
12 }
```

在前面的代码中，`wgt`是`scoped_ptr<Widget>`类型的对象，它是`Widget*`指针的替代品。我们用动态分配的`Widget`对象对其进行初始化（第 8 行），并且省略了`delete`的调用。这是使这段代码具有异常安全性所需的唯一两个更改。

像`scoped_ptr`和 Boost 中的其他智能指针一样，在它们的析构函数中调用`delete`来释放封装的指针。当`useWidget`完成或者如果异常中止它，`scoped_ptr`实例`wgt`的析构函数将被调用，并且将销毁`Widget`对象并释放其内存。`scoped_ptr`中的重载解引用运算符（`operator->`）允许通过`wgt`智能指针访问`Widget`成员（第 9-11 行）。

`boost::scoped_ptr`模板的析构函数使用`boost::checked_delete`来释放封装指针指向的动态分配内存。因此，在`boost::scoped_ptr`实例超出范围时，封装指针指向的对象的类型必须在完全定义；否则，代码将无法编译。

`boost::scoped_ptr`是 Boost 智能指针中最简单的一个。它接管传递的动态分配指针，并在自己的析构函数中调用`delete`。这将使底层对象的生命周期绑定到封装`scoped_ptr`操作的范围，因此称为`scoped_ptr`。本质上，它在封装的指针上实现了 RAII。此外，`scoped_ptr`不能被复制。这意味着动态分配的对象在任何给定时间点只能被一个`scoped_ptr`实例包装。因此，`scoped_ptr`被认为具有*唯一所有权语义*。请注意，`scoped_ptr`实例不能存储在标准库容器中，因为它们在 C++11 意义上既不能被复制也不能被移动。

在下面的例子中，我们探索了`scoped_ptr`的一些更多特性：

**清单 3.2：详细介绍 scoped_ptr**

```cpp
 1 #include <boost/scoped_ptr.hpp>
 2 #include <cassert>
 3 #include "Widget.h" // Widget definition
 4 // …
 5 
 6 void useTwoWidgets()
 7 {
 8   // default constructed scoped_ptr 
 9   boost::scoped_ptr<Widget> wgt;
10   assert(!wgt);          // null test - Boolean context
11 
12   wgt.reset(new Widget); // create first widget
13   assert(wgt);          // non-null test – Boolean context
14   wgt->display();        // display first widget
15   wgt.reset(new Widget); // destroy first, create second widget
16   wgt->display();        // display second widget
17   
18   Widget *w1 = wgt.get();  // get the raw pointer
19   Widget& rw1 = *wgt;      // 'dereference' the smart pointer
20   assert(w1 == &rw1);      // same object, so same address
21
22   boost::scoped_ptr<Widget> wgt2(new Widget);
23   Widget *w2 = wgt2.get();
24   wgt.swap(wgt2);
25   assert(wgt.get() == w2);  // effect of swap
26   assert(wgt2.get() == w1); // effect of swap
27 }
```

在这个例子中，我们首先使用默认构造函数（第 9 行）构造了一个`scoped_ptr<Widget>`类型的对象。这创建了一个包含空指针的`scoped_ptr`。任何尝试对这样一个智能指针进行解引用的行为都会导致未定义的行为，通常会导致崩溃。`scoped_ptr`支持隐式转换为布尔值；因此我们可以在布尔上下文中像`wgt`这样使用`scoped_ptr`对象来检查封装的指针是否为空。在这种情况下，我们知道它应该为空，因为它是默认构造的；因此，我们断言`wgt`为空（第 10 行）。

有两种方法可以改变`scoped_ptr`中包含的指针，其中一种是使用`scoped_ptr`的`reset`成员方法。当我们在`scoped_ptr`上调用`reset`时，封装的指针被释放，并且`scoped_ptr`接管新传递的指针。因此，我们可以使用`reset`来改变`scoped_ptr`实例所拥有的指针（第 12 行）。随后，`scoped_ptr`包含一个非空指针，并且我们使用隐式转换`scoped_ptr`为布尔值的能力进行断言（第 13 行）。接下来，我们再次调用`reset`来在`wgt`中存储一个新的指针（第 15 行）。在这种情况下，先前存储的指针被释放，并且在存储新指针之前底层对象被销毁。

我们可以通过调用`scoped_ptr`的`get`成员函数（第 18 行）来获取底层指针。我们还可以通过对智能指针进行解引用（第 19 行）来获取指向的对象的引用。我们断言这个引用和`get`返回的指针都指向同一个对象（第 20 行）。

当然，改变`scoped_ptr`中包含的指针的第二种方法是交换两个`scoped_ptr`对象，它们的封装指针被交换（第 24-26 行）。这是改变动态分配对象的拥有`scoped_ptr`的唯一方法。

总之，我们可以说一旦你用`scoped_ptr`包装了一个对象，它就永远不能从`scoped_ptr`中分离出来。`scoped_ptr`可以销毁对象并接管一个新对象（使用`reset`成员函数），或者它可以与另一个`scoped_ptr`中的指针交换。在这个意义上，`scoped_ptr`表现出独特的、可转移的所有权语义。

#### scoped_ptr 的用途

`scoped_ptr`是一个轻量级且多功能的智能指针，它不仅可以作为作用域保护器，还可以用于其他用途。下面是它在代码中的使用方式。

##### 创建异常安全的作用域

`scoped_ptr`在创建异常安全的作用域时非常有用，当对象在某个作用域中动态分配。C++允许对象在堆栈上创建，通常这是你会采取的创建对象的方式，而不是动态分配它们。但是，在某些情况下，你需要通过调用返回指向动态分配对象的指针的工厂函数来实例化对象。这可能来自某个旧库，`scoped_ptr`可以成为这些指针的方便包装器。在下面的例子中，`makeWidget`就是一个这样的工厂函数，它返回一个动态分配的`Widget`：

```cpp
 1 class Widget { ... };
 2
 3 Widget *makeWidget() // Legacy function
 4 {
 5   return new Widget;
 6 }
 7 
 8 void useWidget()
 9 {
10   boost::scoped_ptr<Widget> wgt(makeWidget());
11   wgt->display();              // widget displayed
12 }   // Widget destroyed on scope exit
```

一般来说，前面形式中的`useWidget`将是异常安全的，只要从`useWidget`中调用的`makeWidget`函数也是异常安全的。

##### 在函数之间转移对象所有权

作为不可复制的对象，`scoped_ptr`对象不能从函数中以值传递或返回。可以将`scoped_ptr`的非 const 引用作为参数传递给函数，这将重置其内容并将新指针放入`scoped_ptr`对象中。

**清单 3.3：使用 scoped_ptr 进行所有权转移**

```cpp
 1 class Widget { ... };
 2
 3 void makeNewWidget(boost::scoped_ptr<Widget>& result)
 4 {
 5   result.reset(new Widget);
 6   result->setProperties(...);
 7 }
 8 
 9 void makeAndUseWidget()
10 {
11   boost::scoped_ptr<Widget> wgt; // null wgt
12   makeNewWidget(wgt);         // wgt set to some Widget object.
13   wgt->display();              // widget #1 displayed
14 
15   makeNewWidget(wgt);        // wgt reset to some other Widget.
16                              // Older wgt released.
17   wgt->display();            // widget #2 displayed
18 }
```

`makeNewWidget`函数使用传递给它的`scoped_ptr<Widget>`引用作为输出参数，用它来返回动态分配的对象（第 5 行）。每次调用`makeNewWidget`（第 12、15 行）都用新的动态分配的`Widget`对象替换其先前的内容，并删除先前的对象。这是一种将在函数内动态分配的对象所有权转移到函数外作用域的方法。这种方法并不经常使用，在 C++11 中使用`std::unique_ptr`有更多成语化的方法来实现相同的效果，这将在下一节中讨论。

##### 作为类成员

在 Boost 的智能指针中，`scoped_ptr`通常只被用作函数中的本地作用域保护，但实际上，它也可以作为类成员来确保异常安全，是一个有用的工具。

考虑以下代码，其中类`DatabaseHandler`为了记录到文件和连接到数据库创建了两个虚构类型`FileLogger`和`DBConnection`的动态分配对象。`FileLogger`和`DBConnection`以及它们的构造函数参数都是用于说明目的的虚构类。

```cpp
// DatabaseHandler.h
 1 #ifndef DATABASEHANDLER_H
 2 #define DATABASEHANDLER_H
 3
 4 class FileLogger;
 5 class DBConnection;
 6
 7 class DatabaseHandler
 8 {
 9 public:
10   DatabaseHandler();
11   ~DatabaseHandler();
12   // other methods here
13
14 private:
15   FileLogger *logger_;
16   DBConnection *dbconn_;
17 };
18
19 #endif /* DATABASEHANDLER_H */
```

前面的代码是`DatabaseHandler`类在头文件`DatabaseHandler.h`中的定义清单。`FileLogger`和`DBConnection`是不完整的类型，只被前向声明过。我们只声明了指向它们的指针，由于指针的大小不依赖于底层类型的大小，编译器不需要知道`FileHandler`和`DBConnection`的定义来确定`DatabaseHandler`类的总大小，而是以其指针成员的总大小来确定。

设计类的这种方式有一个优势。`DatabaseHandler`的客户端包括前面列出的`DatabaseHandler.h`文件，但不依赖于`FileLogger`或`DBConnection`的实际定义。如果它们的定义发生变化，客户端保持不受影响，无需重新编译。这本质上就是 Herb Sutter 所推广的**Pimpl Idiom**。类的实际实现被抽象在一个单独的源文件中：

```cpp
// DatabaseHandler.cpp
 1 #include "DatabaseHandler.h"
 2 
 3 // Dummy concrete implementations
 4 class FileLogger
 5 {
 6 public:
 7   FileLogger(const std::string& logfile) {...}
 8 private:
 9   ...
10 };
11
12 class DBConnection
13 {
14 public:
15   DBConnection(const std::string& dbhost,
16                const std::string& username,
17                const std::string& passwd) {...}
18 private:
19   ...
20 };
21
22 // class methods implementation
23 DatabaseHandler::DatabaseHandler(const std::string& logFile,
24           const std::string& dbHost,
25           const std::string& user, const std::string& passwd)
26         : logger_(new FileLogger(logFile)), 
27           dbconn_(new DBConnection(dbHost, user, passwd))
28 {}
29
30 ~DatabaseHandler()
31 {
32   delete logger_;
33   delete dbconn_;
34 }
35 
36 // Other methods
```

在这个源文件中，我们可以访问`FileLogger`和`DBConnection`的具体定义。即使这些定义和我们的实现的其他部分发生了变化，只要`DatabaseHandler`的公共方法和类布局没有发生变化，`DatabaseHandler`的客户端就不需要改变或重新编译。

但这段代码非常脆弱，可能会泄漏内存和其他资源。考虑一下如果`FileLogger`构造函数抛出异常会发生什么（第 26 行）。为`logger_`指针分配的内存会自动释放，不会造成进一步的损害。异常从`DatabaseHandler`构造函数传播到调用上下文，`DatabaseHandler`的对象不会被实例化；目前为止一切都很好。

现在考虑如果`FileLogger`对象成功构造，然后`DBConnection`构造函数抛出异常（第 27 行）。在这种情况下，异常发生时为`dbconn_`指针分配的内存会自动释放，但为`logger_`指针分配的内存不会被释放。当异常发生时，任何非 POD 类型的完全构造成员的析构函数都会被调用。但`logger_`是一个原始指针，它是一个 POD 类型，因此它没有析构函数。因此，`logger_`指向的内存泄漏了。

一般来说，如果你的类有多个指向动态分配对象的指针，确保异常安全性就变得很具挑战性，大多数围绕使用 try/catch 块的过程性解决方案都不太好扩展。智能指针是解决这类问题的完美工具，只需很少的代码就可以解决。我们在下面使用`scoped_ptr`来修复前面的例子。这是头文件：

**清单 3.4：将 scoped_ptr 用作类成员**

```cpp
// DatabaseHandler.h
 1 #ifndef DATABASEHANDLER_H
 2 #define DATABASEHANDLER_H
 3
 4 #include <boost/scoped_ptr.hpp>
 5
 6 class FileLogger;
 7 class DBConnection;
 8
 9 class DatabaseHandler
10 {
11 public:
12   DatabaseHandler(const std::string& logFile,
13        const std::string& dbHost, const std::string& user,
14        const std::string& passwd);
15   ~DatabaseHandler();
16   // other methods here
17
18 private:
19   boost::scoped_ptr<FileLogger> logger_;
20   boost::scoped_ptr<DBConnection> dbconn_;
21 
22   DatabaseHandler(const DatabaseHandler&);
23   DatabaseHandler& operator=(const DatabaseHandler&);
24 };
25 #endif /* DATABASEHANDLER_H */
```

`logger_`和`dbconn_`现在是`scoped_ptr`实例，而不是原始指针（第 19 行和第 20 行）。另一方面，由于`scoped_ptr`是不可复制的，编译器无法生成默认的复制构造函数和复制赋值运算符。我们可以像这里做的那样禁用它们（第 22 行和第 23 行），或者自己定义它们。一般来说，为`scoped_ptr`定义复制语义只有在封装类型可复制时才有意义。另一方面，使用`scoped_ptr`的`swap`成员函数可能更容易定义移动语义。现在让我们看看源文件的变化：

```cpp
// DatabaseHandler.cpp
 1 #include "DatabaseHandler.h"
 2 
 3 // Dummy concrete implementations
 4 class FileLogger
 5 {
 6 public:
 7   FileLogger(const std::string& logfile) {...}
 8 private:
 9   ...
10 };
11
12 class DBConnection
13 {
14 public:
15   DBConnection(const std::string& dbhost,
16                const std::string& username,
17                const std::string& passwd) {...}
18 private:
19   ...
20 };
21
22 // class methods implementation
23 DatabaseHandler::DatabaseHandler(const std::string& logFile,
24             const std::string& dbHost, const std::string& user,
25             const std::string& passwd)
26         : logger_(new FileLogger(logFileName)),
27           dbconn_(new DBConnection(dbsys, user, passwd))
28 {}
29
30 ~DatabaseHandler()
31 {}
32 
33 // Other methods
```

我们在构造函数初始化列表中初始化了两个`scoped_ptr`实例（第 26 行和第 27 行）。如果`DBConnection`构造函数抛出异常（第 27 行），则会调用`logger_`的析构函数，它会清理动态分配的`FileLogger`对象。

`DatabaseHandler`析构函数为空（第 31 行），因为没有 POD 类型的成员，而`scoped_ptr`成员的析构函数会自动调用。但我们仍然必须定义析构函数。你能猜到为什么吗？如果让编译器生成定义，它会在头文件中的类定义范围内生成析构函数定义。在那个范围内，`FileLogger`和`DBConnection`没有完全定义，`scoped_ptr`的析构函数将无法编译通过，因为它们使用`boost::checked_delete`（第二章，“与 Boost 实用工具的初次接触”）

### boost::scoped_array

`scoped_ptr` 类模板非常适用于单个动态分配的对象。现在，如果您还记得我们的激励示例，即编写图像旋转实用程序，我们需要在我们自定义的 `ScopeGuard` 类中包装一个动态数组，以使 `rotateImage` 函数具有异常安全性。Boost 提供了 `boost::scoped_array` 模板作为 `boost::scoped_ptr` 的数组类似物。`boost::scoped_array` 的语义与 `boost::scoped_ptr` 完全相同，只是它有一个重载的下标运算符 (`operator[]`) 用于访问封装数组的单个元素，并且不提供其他形式的间接操作符的重载 (`operator*` 和 `operator->`)。在这一点上，使用 `scoped_array` 重写 `rotateImage` 函数将是有益的。

**清单 3.5：使用 scoped_array**

```cpp
 1 #include <boost/scoped_array.hpp>
 2
 3 typedef unsigned char byte;
 4
 5 byte *rotateImage(const std::string &imgFile, double angle, 
 6                   size_t& sz) {
 7   // open the file for reading
 8   std::ifstream imgStrm(imgFile, std::ios::binary);
 9 
10   if (imgStrm) {
11     imgStrm.seekg(0, std::ios::end);
12     sz = imgStrm.tellg();            // determine file size
13     imgStrm.seekg(0);
14 
15     // allocate buffer and read
16     boost::scoped_array<byte> img(new byte[sz]);
17     // read the image contents
18     imgStrm.read(reinterpret_cast<char*>(img.get()), sz);
19 
20     byte first = img[0];  // indexed access
21     return img_rotate(img.get(), sz, angle);
22   }
23 
24   sz = 0;
25   return 0;
26 }
```

我们现在使用 `boost::scoped_array` 模板来代替我们的 `ScopeGuard` 类，以包装动态分配的数组（第 16 行）。在作用域退出时，由于正常执行或异常，`scoped_array` 的析构函数将调用包含动态数组的数组删除运算符 (`delete[]`) 并以异常安全的方式释放它。为了突出从 `scoped_array` 接口访问数组元素的能力，我们使用 `scoped_array` 的重载 `operator[]` 来访问第一个字节（第 20 行）。

`scoped_array` 模板主要用于处理大量动态数组的遗留代码。由于重载下标运算符，`scoped_array` 可以直接替换动态分配的数组。因此，将动态数组封装在 `scoped_array` 中是实现异常安全的快速途径。C++ 倡导使用 `std::vector` 而不是动态数组，这可能是你最终的目标。然而，作为几乎没有与向量相比的空间开销的包装器，`scoped_array` 可以帮助更快地过渡到异常安全的代码。

### std::unique_ptr

C++ 11 引入了 `std::unique_ptr` 智能指针模板，它取代了已弃用的 `std::auto_ptr`，支持 `boost::scoped_ptr` 和 `boost::scoped_array` 的功能，并且可以存储在标准库容器中。它在标准头文件 `memory` 中定义，与 C++11 中引入的其他智能指针一起。

`std::unique_ptr` 的成员函数很容易映射到 `boost::scoped_ptr` 的成员函数：

+   默认构造的 `unique_ptr` 包含一个空指针（`nullptr`），就像默认构造的 `scoped_ptr` 一样。

+   您可以调用 `get` 成员函数来访问包含的指针。

+   `reset` 成员函数释放旧指针并接管新指针的所有权（可以是空指针）。

+   `swap` 成员函数交换两个 `unique_ptr` 实例的内容，并且始终成功。

+   您可以使用 `operator*` 对非空的 `unique_ptr` 实例进行解引用，并使用 `operator->` 访问成员。

+   您可以在布尔上下文中使用 `unique_ptr` 实例来检查是否为空，就像 `scoped_ptr` 实例一样。

+   然而，在某些方面，`std::unique_ptr` 比 `boost::scoped_ptr` 更灵活。

+   `unique_ptr` 是可移动的，不像 `scoped_ptr`。因此，它可以存储在 C++11 标准库容器中，并且可以从函数中返回。

+   如果必须，您可以分离 `std::unique_ptr` 拥有的指针并手动管理它。

+   有一个用于动态分配数组的 `unique_ptr` 部分特化。`scoped_ptr` 不支持数组，您必须使用 `boost::scoped_array` 模板来实现这一目的。

#### 使用 unique_ptr 进行所有权转移

`std::unique_ptr` 智能指针可以像 `boost::scoped_ptr` 一样用作作用域保护。与 `boost::scoped_ptr` 不同，`unique_ptr 实例` 不需要绑定到单个作用域，可以从一个作用域移动到另一个作用域。

`std::unique_ptr`智能指针模板不能被复制，但支持移动语义。支持移动语义使得可以将`std::unique_ptr`用作函数返回值，从而在函数之间传递动态分配的对象的所有权。以下是一个这样的例子：

**列表 3.6a：使用 unique_ptr**

```cpp
// Logger.h
 1 #include <memory>
 2
 3 class Logger
 4 {
 5 public:
 6   Logger(const std::string& filename) { ... }
 7   ~Logger() {...}
 8   void log(const std::string& message, ...) { ... }
 9   // other methods
10 };
11
12 std::unique_ptr<Logger> make_logger(
13                       const std::string& filename) {
14   std::unique_ptr<Logger> logger(new Logger(filename));
15   return logger;
16 }
```

`make_logger`函数是一个工厂函数，返回一个包装在`unique_ptr`中的`Logger`的新实例（第 14 行）。一个函数可以这样使用`make_logger`：

**列表 3.6b：使用 unique_ptr**

```cpp
 1 #include "Logger.h"
 2 
 3 void doLogging(const std::string& msg, ...)
 4 {
 5   std::string logfile = "/var/MyApp/log/app.log";
 6   std::unique_ptr<Logger> logger = make_logger(logfile);
 7   logger->log(msg, ...);
 8 }
```

在函数`doLogging`中，局部变量`logger`通过从`make_logger`返回的`unique_ptr`进行移动初始化（第 6 行）。因此，`make_logger`内部创建的`unique_ptr`实例的内容被移动到变量`logger`中。当`logger`超出范围时，即`doLogging`返回时（第 8 行），它的析构函数将销毁底层的`Logger`实例并释放其内存。

#### 在`unique_ptr`中包装数组

为了说明使用`unique_ptr`包装动态数组的用法，我们将再次重写图像旋转示例（列表 3.5），将`scoped_ptr`替换为`unique_ptr`： 

**列表 3.7：使用 unique_ptr 包装数组**

```cpp
 1 #include <memory>
 2
 3 typedef unsigned char byte;
 4
 5 byte *rotateImage(std::string imgFile, double angle, size_t& sz)
 6 {
 7   // open the file for reading
 8   std::ifstream imgStrm(imgFile, std::ios::binary);
 9 
10   if (imgStrm) {
11     imgStrm.seekg(0, std::ios::end);
12     sz = imgStrm.tellg();      // determine file size
13     imgStrm.seekg(0);
14     
15     // allocate buffer and read
16     std::unique_ptr<byte[]> img(new byte[sz]);
17     // read the image contents
18     imgStrm.read(reinterpret_cast<char*>(img.get()),sz);
19     // process it
20     byte first = img[0];  // access first byte
21     return img_rotate(img.get(), sz, angle);
22   }
23 
24   sz = 0;
25   return 0;
26 }
```

除了包含不同的头文件（`memory`代替`boost/scoped_ptr.hpp`）之外，只需要编辑一行代码。在`boost::scoped_array<byte>`的位置，`img`的声明类型更改为`std::unique_ptr<byte[]>`（第 16 行）- 一个明确的替换。重载的`operator[]`仅适用于`unique_ptr`的数组特化，并用于引用数组的元素。

#### 在 C++14 中使用 make_unique

C++14 标准库包含一个函数模板`std::make_unique`，它是一个用于在动态内存上创建对象实例并将其包装在`std::unique_ptr`中的工厂函数。以下示例是对列表 3.6b 的重写，用于说明`make_unique`的用法：

**列表 3.8：使用 make unique**

```cpp
 1 #include "Logger.h"  // Listing 3.6a
 2 
 3 void doLogging(const std::string& msg, ...)
 4 {
 5   std::string filename = "/var/MyApp/log/app.log";
 6   std::unique_ptr<Logger> logger = 
 7                 std::make_unique<Logger>(filename);
 8   logger->log(msg, ...);
 9 }
```

`std::make_unique`函数模板将要构造的基础对象的类型作为模板参数，并将对象的构造函数的参数作为函数参数。我们直接将文件名参数传递给`make_unique`，它将其转发给`Logger`的构造函数（第 7 行）。`make_unique`是一个可变模板；它接受与实例化类型的构造函数参数匹配的变量数量和类型。如果`Logger`有一个两个参数的构造函数，比如一个接受文件名和默认日志级别的构造函数，我们将向`make_unique`传递两个参数：

```cpp
// two argument constructor
Logger::Logger(const std::string& filename, loglevel_t level) {
  ...
}

std::unique_ptr<Logger> logger =
 std::make_unique<Logger>(filename, DEBUG);

```

假设`loglevel_t`描述用于表示日志级别的类型，`DEBUG`描述该类型的一个有效值，前面的片段说明了使用`make_unique`与多个构造函数参数的用法。

如果您已将代码库迁移到 C++11，应优先使用`std::unique_ptr`而不是`boost::scoped_ptr`。

## 共享所有权语义

具有转移所有权能力的独特所有权语义对于大多数您使用智能指针的目的来说已经足够好了。但在一些现实世界的应用中，您需要在多个上下文中共享资源，而这些上下文中没有一个是明确的所有者。这样的资源只有在持有对共享资源的引用的所有上下文释放它们时才能释放。这种释放的时间和地点无法提前确定。

让我们通过一个具体的例子来理解这一点。在单个进程中，两个线程从内存中的同一动态分配区的不同部分读取数据。每个线程对数据进行一些处理，然后再读取更多数据。我们需要确保当最后一个线程终止时，动态分配的内存区域能够被清理释放。任何一个线程都可能在另一个线程之前终止；那么谁来释放缓冲区呢？

通过将缓冲区封装在一个智能包装器中，该包装器可以保持对其的引用计数，并且仅当计数变为零时才释放缓冲区，我们可以完全封装释放逻辑。缓冲区的用户应该切换到使用智能包装器，他们可以自由复制，并且当所有副本超出范围时，引用计数变为零并且缓冲区被释放。

### boost::shared_ptr 和 std::shared_ptr

`boost::shared_ptr`智能指针模板提供了引用计数的共享所有权语义。它使用共享引用计数来跟踪对它的引用次数，该引用计数与包装的动态分配对象一起维护。与我们迄今为止看到的其他智能指针模板一样，它实现了 RAII 习语，负责在其析构函数中销毁和释放包装的对象，但只有当所有对它的引用都被销毁时才这样做，也就是说，引用计数变为零。它是一个仅包含头文件的库，通过包括`boost/shared_ptr.hpp`可用。

`shared_ptr`于 2007 年被包含在 C++标准委员会技术报告（俗称 TR1）中，这是 C++11 标准的前身，并作为`std::tr1::shared_ptr`提供。它现在是 C++11 标准库的一部分，作为`std::shared_ptr`通过标准 C++头文件`memory`提供。如果您将代码库迁移到 C++11，应该使用`std::shared_ptr`。本节中的大部分讨论都适用于两个版本；如果有任何区别，都会被指出。

您创建`shared_ptr`实例来拥有动态分配的对象。与`boost::scoped_ptr`和`std::unique_ptr`不同，您可以复制`shared_ptr`实例。`std::shared_ptr`还支持移动语义。它存储动态分配的指针和共享引用计数对象。每次通过复制构造函数复制`shared_ptr`时，指针和引用计数对象都会被浅复制。复制`shared_ptr`实例会导致引用计数增加。`shared_ptr`实例超出范围会导致引用计数减少。`use_count`成员函数可用于获取当前引用计数。以下是一个展示`shared_ptr`的示例：

**清单 3.9：`shared_ptr`的示例**

```cpp
 1 #include <boost/shared_ptr.hpp>
 2 #include <iostream>
 3 #include <cassert>
 4 
 5 class Foo {
 6 public:
 7   Foo() {}
 8   ~Foo() { std::cout << "~Foo() destructor invoked." << '\n';}
 9 };
10 
11 typedef boost::shared_ptr<Foo> SPFoo;
12   
13 int main()
14 {
15   SPFoo f1(new Foo);
16   // SPFoo f1 = new Foo; // Won't work, explicit ctor
17   assert(f1.use_count() == 1);
18
19   // copy construction
20   SPFoo f2(f1);
21   assert(f1.use_count() == f2.use_count() && 
22          f1.get() == f2.get() && f1.use_count() == 2);
23   std::cout << "f1 use_count: " << f1.use_count() << '\n';
24          
25   SPFoo f3(new Foo);
26   SPFoo f4(f3);
27   assert(f3.use_count() == 2 && f3.get() == f4.get());
28   std::cout << "f3 use_count: " << f3.use_count() << '\n';
29  
30   // copy assignment
31   f4 = f1;
32   assert(f4.use_count() == f1.use_count() && 
33         f1.use_count() == 3 && f1.get() == f4.get());
34   assert(f3.use_count() == 1);
35   std::cout << "f1 use_count: " << f1.use_count() << '\n';
36   std::cout << "f3 use_count: " << f3.use_count() << '\n';
37 }
```

在上述代码中，我们定义了一个带有默认构造函数和打印一些消息的析构函数的`Foo`类（第 5-9 行）。我们包括了`boost/shared_ptr.hpp`（第 1 行），它提供了`boost::shared_ptr`模板。

在主函数中，我们定义了两个`shared_ptr<Foo>`实例`f1`（第 15 行）和`f3`（第 25 行），初始化为`Foo`类的两个不同动态分配的实例。请注意，`shared_ptr`构造函数是显式的，因此您不能使用赋值表达式使用隐式转换来复制初始化`shared_ptr`（第 16 行）。每个`shared_ptr<Foo>`实例在构造后的引用计数为 1（第 17 行和第 25 行）。接下来，我们创建`f2`作为`f1`的副本（第 20 行），并创建`f4`作为`f3`的副本（第 26 行）。复制会导致引用计数增加。`shared_ptr`的`get`成员函数返回封装的指针，`use_count`成员函数返回当前引用计数。使用`use_count`，我们断言`f1`和`f2`具有相同的引用计数，并使用`get`，我们断言它们包含相同的指针（第 21-22 行）。对于`f3`和`f4`也是如此（第 27 行）。

接下来，我们将`f1`复制分配给`f4`（第 31 行）。结果，`f4`现在包含与`f1`和`f2`相同的指针，并且不再与`f3`共享指针。现在，`f1`，`f2`和`f4`是指向相同指针的三个`shared_ptr<Foo>`实例，它们的共享引用计数变为 3（第 32-33 行）。`f3`不再与另一个实例共享其指针，因此其引用计数变为 1（第 34 行）。

运行上述代码，您可以期望以下输出：

```cpp
f1 use_count: 2
f3 use_count: 2
f1 use_count: 3
f3 use_count: 1
~Foo() destructor invoked.
~Foo() destructor invoked.
```

引用计数在`main`函数结束时确实变为零，并且`shared_ptr`析构函数销毁了动态创建的`Foo`实例。

#### `shared_ptr`的用途

在 C++11 之前的代码中，由于其灵活性和易用性，`boost::shared_ptr`或`std::tr1::shared_ptr`往往是智能指针的默认选择，而不是`boost::scoped_ptr`。它用于超出纯共享所有权语义的目的，这使其成为最知名的智能指针模板。在 C++11 中，应该遏制这种普遍使用，而应该优先使用`std::unique_ptr`，`shared_ptr`应该仅用于模拟真正的共享所有权语义。

##### 作为类成员

考虑一个场景，应用程序的多个组件可以共享单个数据库连接以获得更好的性能。只要有一些组件在使用它，就可以在首次请求时创建这样的连接并将其缓存。当所有组件都使用完毕后，连接应该被关闭。这是共享所有权语义的定义，`shared_ptr`在这种情况下非常有用。让我们看看应用程序组件如何使用`shared_ptr`来封装共享数据库连接：

**清单 3.10：将 shared_ptr 用作类成员**

```cpp
 1 class AppComponent
 2 {
 3 public:
 4  AppComponent() : spconn_(new DatabaseConnection(...))
 5  {}
 6 
 7  AppComponent( 
 8         const boost::shared_ptr<DatabaseConnection>& spc)
 9      : spconn_(spc) {}
11 
12  // Other public member
13  ...
14
15  boost::shared_ptr<DatabaseConnection> getConnection() {
16    return spconn_;
17  }
18 
19 private:
20  boost::shared_ptr<DatabaseConnection> spconn_;
21  // other data members
22 };
```

`AppComponent`是应用程序的一个组件，它使用包装在`shared_ptr`（第 20 行）中的数据库连接。默认构造的`AppComponent`创建一个新的数据库连接（第 4 行），但您始终可以通过传递包装在`shared_ptr`（第 7-9 行）中的现有数据库连接来创建`AppComponent`实例。`getConnection`成员函数检索包装在 shared_ptr 中的`DatabaseConnection`对象（第 16 行）。以下是一个例子：

```cpp
 1 AppComponent c1;
 2 AppComponent c2(a.getConnection());
```

在此示例中，我们创建了两个`AppComponent`实例`c1`和`c2`，它们共享相同的数据库连接。第二个实例是使用第一个实例缓存的`shared_ptr`包装的数据库连接通过`getConnection`方法获得的。无论`c1`和`c2`的销毁顺序如何，只有当两者中的最后一个被销毁时，共享连接才会被销毁。

##### 在标准库容器中存储动态分配的对象

标准库容器存储的对象被复制或移动到容器中，并随容器一起销毁。对象也通过复制或移动来检索。在 C++11 之前，没有支持移动语义，复制是在容器中存储对象的唯一机制。标准库容器不支持引用语义。您可以将动态分配对象的指针存储在容器中，但在其生命周期结束时，容器不会尝试通过指针销毁和释放这些对象。

您可以将动态分配的对象包装在`shared_ptr`或`unique_ptr`中并将它们存储在容器中。假设您可以使用 C++11，如果将它们存储在单个容器中就足够了，那么`std::unique_ptr`就足够好了。但是，如果需要在多个容器中存储相同的动态分配对象，`shared_ptr`是包装器的最佳选择。当容器被销毁时，将调用每个`shared_ptr`实例的析构函数，并将该`shared_ptr`的引用计数减少。如果任何`shared_ptr`的引用计数为零，则其中存储的底层动态对象将被释放。以下示例说明了如何将`shared_ptr`中包装的对象存储在多个 STL 容器中：

**清单 3.11：在容器中存储 shared_ptr**

```cpp
 1 class Person;
 2 typedef boost::shared_ptr<Person> PersonPtr;
 3 std::vector<PersonPtr> personList;
 4 std::multimap<std::string, PersonPtr> personNameMap;
 5 ...
 6 
 7 for (auto it = personList.begin(); 
 8      it != personList.end(); ++it) {
 9   personNameMap.insert(std::make_pair((*it)->name(), *it));
10 }
```

在前面的例子中，让我们假设有一个名为`Person`的类（第 1 行）。现在，给定一个类型为`Person`的对象列表，我们想要创建一个将名称映射到`Person`对象的映射。假设`Person`对象不能被复制，因此它们需要以指针的形式存储在容器中。我们为`shared_ptr<Person>`定义了一个类型别名称为`PersonPtr`（第 2 行）。我们还定义了用于存储`Person`对象列表的数据结构（`std::vector<PersonPtr>`（第 3 行））和将`Person`名称映射到`Person`对象的映射（`std::multimap<std::string, PersonPtr>`（第 4 行））。最后，我们从列表构造映射（第 7-9 行）。

`personNameMap`容器中的每个条目都被创建为一个人的名称和`PersonPtr`对象的`std::pair`（使用`std::make_pair`）。每个这样的条目都使用其`insert`成员函数插入到`multimap`中（第 9 行）。我们假设`Person`中有一个名为`name`的成员函数。`PersonPtr`对象作为`shared_ptr`在`vector`和`multimap`容器之间共享。当两个容器中的最后一个被销毁时，`Person`对象也将被销毁。

除了`shared_ptr`，Boost 的指针容器提供了一种在容器中存储动态分配对象的替代方法。我们将在第五章中介绍指针容器，*超越 STL 的有效数据结构*。在第九章中，*文件、目录和 IOStreams*，处理 Boost 线程，我们将看到`shared_ptr`实例如何在线程之间共享。

#### 非拥有别名 - boost::weak_ptr 和 std::weak_ptr

在上一节中，我们看到的一个例子是多个应用程序组件共享的数据库连接。这种使用方式有一定的缺点。在实例化旨在重用打开的数据库连接的应用程序组件时，您需要引用另一个使用连接的现有组件，并将该连接传递给新对象的构造函数。更可扩展的方法是解耦连接创建和应用程序组件创建，以便应用程序组件甚至不知道它们是否获得了新连接或现有可重用连接。但要求仍然是连接必须在所有客户端之间共享，并且在最后一个引用它消失时必须关闭连接。

构建这样一种机制的一种方法是使用数据库连接工厂，它根据调用者传递的连接参数创建到特定数据库实例的连接。然后将连接包装在`shared_ptr`中返回给调用者，并将其存储在可以查找的映射中。当新的客户端请求连接到相同数据库用户的相同实例时，工厂可以简单地从映射中查找现有连接并将其包装在`shared_ptr`中返回。以下是说明此逻辑的代码。它假设连接到数据库实例所需的所有信息都封装在`DBCredentials`对象中：

```cpp
 1 typedef boost::shared_ptr<DatabaseConnection> DBConnectionPtr;
 2
 3 struct DBConnectionFactory
 4 {
 5   typedef std::map<DBCredentials, DBConnectionPtr> 
 6                                             ConnectionMap;
 7
 8   static DBConnectionPtr connect(const DBCredentials& creds)
 9   {
10     auto iter = conn_map_.find(creds);
11
12     if (iter != conn_map_.end()) {
13       return iter->second;
14     } else {
15       DBConnectionPtr dbconn(new DatabaseConnection(creds));
16       conn_map_[creds] = dbconn;
17       return dbconn;
18     }
19   }
20 
21   static ConnectionMap conn_map_;
22 };
23 
24 DBConnectionFactory::ConnectionMap 
25                                DBConnectionFactory::conn_map_;
26 int main()
27 {
28   DBCredentials creds(...);
29   DBConnectionPtr dbconn = DBConnectionFactory::connect(creds);
30   DBConnectionPtr dbconn2 =DBConnectionFactory::connect(creds);
31   assert(dbconn.get() == dbconn2.get() 
32          && dbconn.use_count() == 3);
33 }
```

在前面的代码中，`DBConnectionFactory`提供了一个名为`connect`的静态方法，它接受一个`DBCredentials`对象并返回一个`shared_ptr`包装的`DatabaseConnection`（`DBConnectionPtr`）（第 8-19 行）。我们调用`DBConnectionFactory::connect`两次，传递相同的凭据。第一次调用（第 28 行）应该导致创建一个新的连接（第 15 行），而第二次调用应该只是查找并返回相同的连接（第 10-13 行）。

这段代码存在一个主要问题：`DBConnectionFactory`将连接存储在静态的`std::map` `conn_map_`（第 21 行）中的`shared_ptr`中。结果是，只有在程序结束时`conn_map_`被销毁时，引用计数才会变为 0。否则，即使没有上下文使用连接，引用计数仍保持为 1。我们要求，当所有使用共享连接的上下文退出或过期时，连接应该被销毁。显然这个要求没有得到满足。

在地图中存储原始指针（`DatabaseConnection*`）而不是`shared_ptr`（`DBConnectionPtr`）是不好的，因为我们需要为连接创建更多的`shared_ptr`实例时，能够使用我们分发的第一个`shared_ptr`实例。即使有方法可以解决这个问题（正如我们将在`enable_shared_from_this`中看到的），通过在连接映射中查找原始指针，我们也无法知道它是否仍在使用或已经被释放。

`boost::weak_ptr`模板，也可以在 C++11 中作为`std::weak_ptr`使用，是解决这个问题的正确工具。您可以使用一个或多个`weak_ptr`实例引用`shared_ptr`实例，而不会增加决定其生命周期的引用计数。使用`weak_ptr`实例，您可以安全地确定它所引用的`shared_ptr`是否仍然活动或已过期。如果没有过期，您可以使用`weak_ptr`实例来创建另一个引用相同对象的`shared_ptr`实例。现在我们将使用`weak_ptr`重写前面的示例：

**清单 3.12：使用 weak_ptr**

```cpp
 1 typedef boost::shared_ptr<DatabaseConnection> DBConnectionPtr;
 2 typedef boost::weak_ptr<DatabaseConnection> DBConnectionWkPtr;
 3
 4 struct DBConnectionFactory
 5 {
 6   typedef std::map<DBCredentials, DBConnectionWkPtr> 
 7                                             ConnectionMap;
 8
 9   static DBConnectionPtr connect(const DBCredentials& creds) {
10      ConnectionIter it = conn_map_.find(creds);
11      DBConnectionPtr connptr;
12
13     if (it != conn_map_.end() &&
14         (connptr = it->second.lock())) {
15       return connptr;
16     } else {
17       DBConnectionPtr dbconn(new DatabaseConnection(creds));
18       conn_map_[creds] = dbconn;  // weak_ptr = shared_ptr;
19       return dbconn;
20     }
21   }
22 
23   static ConnectionMap conn_map_;
24 };
25 
26 DBConnectionFactory::ConnectionMap 
27                                DBConnectionFactory::conn_map_;
28 int main()
29 {
30   DBCredentials creds(...);
31   DBConnectionPtr dbconn = DBConnectionFactory::connect(creds);
32   DBConnectionPtr dbconn2 =DBConnectionFactory::connect(creds);
33   assert(dbconn.get() == dbconn2.get() 
34          && dbconn.use_count() == 2);
35 }
```

在这个例子中，我们修改了`ConnectionMap`的定义，将`shared_ptr<DatabaseConnection>`存储为`weak_ptr<DatabaseConnection>`（第 6-7 行）。当调用`DBConnectionFactory::connect`函数时，代码查找条目（第 10 行），失败时，创建一个新的数据库连接，将其包装在`shared_ptr`中（第 17 行），并将其存储为地图中的`weak_ptr`（第 18 行）。请注意，我们使用复制赋值运算符将`shared_ptr`分配给`weak_ptr`。新构造的`shared_ptr`被返回（第 19 行）。如果查找成功，它会尝试从中检索的`weak_ptr`上调用`lock`方法来构造一个`shared_ptr`（第 12 行）。如果由`it->second`表示的检索的`weak_ptr`引用一个有效的`shared_ptr`，`lock`调用将自动返回另一个引用相同对象的`shared_ptr`，并将其分配给`connptr`变量并返回（第 15 行）。否则，`lock`调用将返回一个空的`shared_ptr`，我们将在`else`块中创建一个新的连接，就像之前描述的那样。

如果您只想检查`weak_ptr`实例是否引用有效的`shared_ptr`，而不创建一个新的`shared_ptr`引用对象，只需在`weak_ptr`上调用`expired`方法。只有当至少有一个`shared_ptr`实例仍然存在时，它才会返回`false`。

`weak_ptr`是如何实现这一点的？实际上，`shared_ptr`和`weak_ptr`是设计为相互配合使用的。每个`shared_ptr`实例都有两块内存：它封装的动态分配的对象和一个名为共享计数器的内存块，其中包含两个原子引用计数而不是一个。这两块内存都在所有相关的`shared_ptr`实例之间共享。共享计数器块也与引用这些`shared_ptr`实例的所有`weak_ptr`实例共享。

共享计数器中的第一个引用计数，*使用计数*，保持对`shared_ptr`的引用数量的计数。当此计数变为零时，封装的动态分配对象将被删除，`shared_ptr`将过期。第二个引用计数，*弱引用计数*，是`weak_ptr`引用的数量，加上 1（仅当有`shared_ptr`实例存在时）。只有当弱引用计数变为零时，也就是当所有`shared_ptr`和`weak_ptr`实例都过期时，共享计数块才会被删除。因此，任何剩余的`weak_ptr`实例都可以通过检查使用计数来判断`shared_ptr`是否已过期，并查看它是否为 0。`weak_ptr`的`lock`方法会原子地检查使用计数，并仅在它不为零时递增它，返回一个包装封装指针的有效`shared_ptr`。如果使用计数已经为零，`lock`将返回一个空的`shared_ptr`。

#### 一个 shared_ptr 的批评 - make_shared 和 enable_shared_from_this

`shared_ptr`已被广泛使用，超出了适当使用共享所有权语义的用例。这在一定程度上是由于它作为 C++ **技术报告 1**（**TR1**）发布的一部分而可用，而其他可行的选项，如 Boost 的指针容器（参见第五章，*超出 STL 的有效数据结构*）并不是 TR1 的一部分。但是`shared_ptr`需要额外的分配来存储共享计数，因此构造和销毁比`unique_ptr`和`scoped_ptr`慢。共享计数本身是一个包含两个原子整数的对象。如果您从不需要共享所有权语义但使用`shared_ptr`，则需要为共享计数支付额外的分配，并且需要为原子计数的递增和递减操作付费，这使得复制`shared_ptr`变慢。如果您需要共享所有权语义但不关心`weak_ptr`观察者，那么您需要为弱引用计数占用的额外空间付费，而这是您不需要的。

缓解这个问题的一种方法是以某种方式将两个分配（一个用于对象，一个用于共享计数）合并为一个。`boost::make_shared`函数模板（C++11 中也有`std::make_shared`）是一个可变函数模板，正是这样做的。以下是您将如何使用它的方式：

**清单 3.13：使用 make_shared**

```cpp
 1 #include <boost/make_shared.hpp>
 2
 3 struct Foo {
 4   Foo(const std::string& name, int num);
 5   ...
 6 };
 7
 8 boost::shared_ptr<Foo> spfoo = 
 9             boost::make_shared<Foo>("Foo", 10);
10
```

`boost::make_shared`函数模板将对象的类型作为模板参数，并将对象的构造函数的参数作为函数参数。我们调用`make_shared<Foo>`，将要用来构造`Foo`对象的参数传递给它（第 8-9 行）。然后函数会在内存中分配一个单一的内存块，其中放置对象，并一次性附加两个原子计数。请注意，您需要包含头文件`boost/make_shared.hpp`才能使用`make_shared`。这并不像看起来那么完美，但可能是一个足够好的折衷方案。这并不完美，因为现在它是一个单一的内存块而不是两个，并且在所有`shared_ptr`和`weak_ptr`引用之间共享。

即使所有`shared_ptr`引用都消失并且对象被销毁，只有当最后一个`weak_ptr`消失时，其内存才会被回收。同样，只有在使用持久的`weak_ptr`实例并且对象大小足够大以至于成为一个问题时，这才是一个问题。

我们之前简要讨论过`shared_ptr`的另一个问题。如果我们从同一个原始指针创建两个独立的`shared_ptr`实例，那么它们将有独立的引用计数，并且两者都会尝试在适当的时候删除封装的对象。第一个会成功，但第二个实例的析构函数很可能会崩溃，试图删除一个已经删除的实体。此外，在第一个实例超出范围后，通过第二个`shared_ptr`尝试解引用对象的任何尝试都将同样灾难性。解决这个问题的一般方法是根本不使用`shared_ptr`，而是使用`boost::intrusive_ptr`——这是我们在下一节中探讨的内容。解决这个问题的另一种方法是为包装类的实例方法提供返回`shared_ptr`的能力，使用`this`指针。为此，你的类必须从`boost::enable_shared_from_this`类模板派生。下面是一个例子：

**清单 3.14：使用 enable_shared_from_this**

```cpp
 1 #include <boost/smart_ptr.hpp>
 2 #include <boost/current_function.hpp>
 3 #include <iostream>
 4 #include <cassert>
 5
 6 class CanBeShared
 7        : public boost::enable_shared_from_this<CanBeShared> {
 8 public:
 9   ~CanBeShared() {
10     std::cout << BOOST_CURRENT_FUNCTION << '\n';
11   }
12   
13   boost::shared_ptr<CanBeShared> share()
14   {
15     return shared_from_this();
16   }
17 };
18 
19 typedef boost::shared_ptr<CanBeShared> CanBeSharedPtr;
20 
21 void doWork(CanBeShared& obj)
22 {
23   CanBeSharedPtr sp = obj.share();
24   std::cout << "Usage count in doWork "<<sp.use_count() <<'\n';
25   assert(sp.use_count() == 2);
26   assert(&obj == sp.get());
27 }
28 
29 int main()
30 {
31   CanBeSharedPtr cbs = boost::make_shared<CanBeShared>();
32   doWork(*cbs.get());
33   std::cout << cbs.use_count() << '\n';
34   assert(cbs.use_count() == 1);
35 }
```

在前面的代码中，类`CanBeShared`派生自`boost::enable_shared_from_this<CanBeShared>`（第 7 行）。如果你想知道为什么`CanBeShared`继承自一个类模板实例，该实例以`CanBeShared`本身作为模板参数，那么我建议你查阅一下奇异递归模板模式，这是一个 C++习惯用法，你可以在网上了解更多。现在，`CanBeShared`定义了一个名为`share`的成员函数，它返回包装在`shared_ptr`中的`this`指针（第 13 行）。它使用了它从基类继承的成员函数`shared_from_this`（第 15 行）来实现这一点。

在`main`函数中，我们从类型为`CanBeShared`的动态分配对象（第 31 行）创建了`CanBeSharedPtr`的实例`cbs`（它是`boost::shared_ptr<CanBeShared>`的`typedef`）。接下来，我们调用`doWork`函数，将`cbs`中的原始指针传递给它（第 32 行）。`doWork`函数被传递了一个对`CanBeShared`的引用（`obj`），并调用了它的`share`方法来获取相同对象的`shared_ptr`包装（第 23 行）。这个`shared_ptr`的引用计数现在变成了 2（第 25 行），它包含的指针指向`obj`（第 26 行）。一旦`doWork`返回，`cbs`上的使用计数就会回到 1（第 34 行）。

从对`shared_from_this`的调用返回的`shared_ptr`实例是从`enable_shared_from_this<>`基类中的`weak_ptr`成员实例构造的，并且仅在包装对象的构造函数结束时构造。因此，如果你在类的构造函数中调用`shared_from_this`，你将遇到运行时错误。你还应该避免在尚未包装在`shared_ptr`对象中的原始指针上调用它，或者在开始时就不是动态构造的对象上调用它。C++11 标准将这一功能标准化为`std::enable_shared_from_this`，可以通过标准头文件`memory`使用。我们在编写异步 TCP 服务器时广泛使用`enable_shared_from_this`，详见第十一章 *网络编程使用 Boost Asio*。

如果你有雄辩的眼力，你会注意到我们只包含了一个头文件`boost/smart_ptr.hpp`。这是一个方便的头文件，将所有可用的智能指针功能集成到一个头文件中，这样你就不必记得包含多个头文件。

### 提示

如果你可以使用 C++11，那么在大多数情况下应该使用`std::unique_ptr`，只有在需要共享所有权时才使用`shared_ptr`。如果出于某种原因仍在使用 C++03，你应该尽可能利用`boost::scoped_ptr`，或者使用`boost::shared_ptr`和`boost::make_shared`以获得更好的性能。

### 侵入式智能指针 - boost::intrusive_ptr

考虑一下当你将同一个指针包装在两个不是彼此副本的`shared_ptr`实例中会发生什么。

```cpp
 1 #include <boost/shared_ptr.hpp>
 2 
 3 int main()
 4 {
 5   boost::shared_ptr<Foo> f1 = boost::make_shared<Foo>();
 6   boost::shared_ptr<Foo> f2(f1.get());  // don't try this
 7
 8   assert(f1.use_count() == 1 && f2.use_count() == 1);
 9   assert(f1.get() == f2.get());
10 } // boom!
```

在前面的代码中，我们创建了一个`shared_ptr<Foo>`实例（第 5 行）和第二个独立的`shared_ptr<Foo>`实例，使用与第一个相同的指针（第 6 行）。其结果是两个`shared_ptr<Foo>`实例都具有引用计数为 1（第 8 行的断言），并且都包含相同的指针（第 9 行的断言）。在作用域结束时，`f1`和`f2`的引用计数都变为零，并且都尝试在相同的指针上调用`delete`（第 10 行）。由于双重删除的结果，代码几乎肯定会崩溃。从编译的角度来看，代码是完全合法的，但行为却不好。您需要防范对`shared_ptr<Foo>`的这种使用，但这也指出了`shared_ptr`的一个局限性。这个局限性是由于仅凭借原始指针，无法判断它是否已被某个智能指针引用。共享引用计数在`Foo`对象之外，不是其一部分。`shared_ptr`被称为非侵入式。

另一种方法是将引用计数作为对象本身的一部分进行维护。在某些情况下可能不可行，但在其他情况下将是完全可接受的。甚至可能存在实际维护这种引用计数的现有对象。如果您曾经使用过 Microsoft 的组件对象模型，您就使用过这样的对象。`boost::intrusive_ptr`模板是`shared_ptr`的一种侵入式替代品，它将维护引用计数的责任放在用户身上，并使用用户提供的钩子来增加和减少引用计数。如果用户愿意，引用计数可以成为类布局的一部分。这有两个优点。对象和引用计数在内存中相邻，因此具有更好的缓存性能。其次，所有`boost::intrusive_ptr`实例使用相同的引用计数来管理对象的生命周期。因此，独立的`boost::intrusive_ptr`实例不会造成双重删除的问题。实际上，您可以潜在地同时为同一个对象使用多个不同的智能指针包装器，只要它们使用相同的侵入式引用计数。

#### 使用 intrusive_ptr

要管理类型为`X`的动态分配实例，您创建`boost::intrusive_ptr<X>`实例，就像创建其他智能指针实例一样。您只需要确保有两个全局函数`intrusive_ptr_add_ref(X*)`和`intrusive_ptr_release(X*)`可用，负责增加和减少引用计数，并在引用计数变为零时调用`delete`删除动态分配的对象。如果`X`是命名空间的一部分，那么这两个全局函数最好也应该在相同的命名空间中定义，以便进行参数相关查找。因此，用户控制引用计数和删除机制，并且`boost::intrusive_ptr`提供了一个 RAII 框架，它们被连接到其中。请注意，如何维护引用计数是用户的权利，不正确的实现可能会导致泄漏、崩溃，或者至少是低效的代码。最后，这里是一些使用`boost::intrusive_ptr`的示例代码：

**清单 3.15：使用 intrusive_ptr**

```cpp
 1 #include <boost/intrusive_ptr.hpp>
 2 #include <iostream>
 3 
 4 namespace NS {
 5 class Bar {
 6 public:
 7   Bar() : refcount_(0) {}
 8  ~Bar() { std::cout << "~Bar invoked" << '\n'; }
 9 
10   friend void intrusive_ptr_add_ref(Bar*);
11   friend void intrusive_ptr_release(Bar*);
12 
13 private:
14   unsigned long refcount_;
15 };
16 
17 void intrusive_ptr_add_ref(Bar* b) {
18   b->refcount_++;
19 }
20 
21 void intrusive_ptr_release(Bar* b) {
22   if (--b->refcount_ == 0) {
23     delete b;
24   }
25 }    
26 } // end NS
27 
28 
29 int main()
30 {
31   boost::intrusive_ptr<NS::Bar> pi(new NS::Bar, true);
32   boost::intrusive_ptr<NS::Bar> pi2(pi);
33   assert(pi.get() == pi2.get());
34   std::cout << "pi: " << pi.get() << '\n'
35             << "pi2: " << pi2.get() << '\n';
36 }
```

我们使用`boost::intrusive_ptr`来包装类`Bar`的动态分配对象（第 31 行）。我们还可以将一个`intrusive_ptr<NS::Bar>`实例复制到另一个实例中（第 32 行）。类`Bar`在成员变量`refcount_`中维护其引用计数，类型为`unsigned long`（第 14 行）。`intrusive_ptr_add_ref`和`intrusive_ptr_release`函数被声明为类`Bar`的友元（第 10 和 11 行），并且在与`Bar`相同的命名空间`NS`中（第 3-26 行）。`intrusive_ptr_add_ref`每次调用时都会增加`refcount_`。`intrusive_ptr_release`会减少`refcount_`并在`refcount_`变为零时对其参数指针调用`delete`。

类`Bar`将变量`refcount_`初始化为零。我们将布尔值的第二个参数传递给`intrusive_ptr`构造函数，以便构造函数通过调用`intrusive_ptr_add_ref(NS::Bar*)`来增加`Bar`的`refcount_`（第 31 行）。这是默认行为，`intrusive_ptr`构造函数的布尔值第二个参数默认为`true`，因此我们实际上不需要显式传递它。另一方面，如果我们处理的是一个在初始化时将其引用计数设置为 1 而不是 0 的类，那么我们不希望构造函数再次增加引用计数。在这种情况下，我们应该将第二个参数传递给`intrusive_ptr`构造函数为`false`。复制构造函数总是通过调用`intrusive_ptr_add_ref`增加引用计数。每个`intrusive_ptr`实例的析构函数调用`intrusive_ptr_release`，并将封装的指针传递给它。

虽然前面的示例说明了如何使用`boost::intrusive_ptr`模板，但如果你要管理动态分配的对象，Boost 提供了一些便利。`boost::intrusive_ref_counter`包装了一些通用的样板代码，这样你就不必自己编写那么多了。以下示例说明了这种用法：

**清单 3.16：使用 intrusive_ptr 减少代码**

```cpp
 1 #include <boost/intrusive_ptr.hpp>
 2 #include <boost/smart_ptr/intrusive_ref_counter.hpp>
 3 #include <iostream>
 4 #include <cassert>
 5
 6 namespace NS {
 7 class Bar : public boost::intrusive_ref_counter<Bar> {
 8 public:
 9   Bar() {}
10   ~Bar() { std::cout << "~Bar invoked" << '\n'; }
11 };
12 } // end NS
13
14 int main() {
15   boost::intrusive_ptr<NS::Bar> pi(new NS::Bar);
16   boost::intrusive_ptr<NS::Bar> pi2(pi);
17   assert(pi.get() == pi2.get());
18   std::cout << "pi: " << pi.get() << '\n'
19             << "pi2: " << pi2.get() << '\n';
20   
21   assert(pi->use_count() == pi2->use_count()
22          && pi2->use_count() == 2);
23   std::cout << "pi->use_count() : " << pi->use_count() << '\n'
24          << "pi2->use_count() : " << pi2->use_count() << '\n';
25 }
```

我们不再维护引用计数，并为`intrusive_ptr_add_ref`和`intrusive_ptr_release`提供命名空间级别的重载，而是直接从`boost::intrusive_ref_counter<Bar>`公开继承类`Bar`。这就是我们需要做的全部。这也使得可以轻松地在任何时候获取引用计数，使用从`intrusive_ref_counter<>`继承到`Bar`的`use_count()`公共成员。请注意，`use_count()`不是`intrusive_ptr`本身的成员函数，因此我们必须使用解引用运算符(`operator->`)来调用它（第 21-24 行）。

前面示例中使用的引用计数器不是线程安全的。如果要确保引用计数的线程安全性，请编辑示例，使用`boost::thread_safe_counter`策略类作为`boost::intrusive_ref_counter`的第二个类型参数：

```cpp
 7 class Bar : public boost::intrusive_ref_counter<Bar, 
 8                               boost::thread_safe_counter>
```

有趣的是，`Bar`继承自`boost::intrusive_ref_counter`模板的一个实例化，该模板将`Bar`本身作为模板参数。这再次展示了奇特的递归模板模式的工作原理。

### shared_array

就像`boost::scoped_ptr`有一个专门用于管理动态分配数组的模板一样，有一个名为`boost::shared_array`的模板，可以用来包装动态分配的数组，并使用共享所有权语义来管理它们。与`scoped_array`一样，`boost::shared_array`有一个重载的下标运算符(`operator[]`)。与`boost::shared_ptr`一样，它使用共享引用计数来管理封装数组的生命周期。与`boost::shared_ptr`不同的是，`shared_array`没有`weak_array`。这是一个方便的抽象，可以用作引用计数的向量。我留给你进一步探索。

## 使用智能指针管理非内存资源

到目前为止，我们所见过的所有智能指针类都假设它们的资源是使用 C++的`new`运算符动态分配的，并且需要使用`delete`运算符进行删除。`scoped_array`和`shared_array`类以及`unique_ptr`的数组部分特化都假设它们的资源是动态分配的数组，并使用数组`delete`运算符(`delete[]`)来释放它们。动态内存不是程序需要以异常安全方式管理的唯一资源，智能指针忽视了这种情况。

`shared_ptr`和`std::unique_ptr`模板可以使用替代的用户指定的删除策略。这使它们适用于管理不仅是动态内存，而且几乎任何具有显式创建和删除 API 的资源，例如使用`malloc`和`free`进行 C 风格堆内存分配和释放，打开文件流，Unix 打开文件描述符和套接字，特定于平台的同步原语，Win32 API 句柄到各种资源，甚至用户定义的抽象。以下是一个简短的例子来结束本章：

```cpp
 1 #include <boost/shared_ptr.hpp>
 2 #include <stdio.h>
 3 #include <time.h>
 4 
 5 struct FILEDeleter
 6 {
 7   void operator () (FILE *fp) const {
 8     fprintf(stderr, "Deleter invoked\n");
 9     if (fp) {
10       ::fclose(fp);
11     }
12   }
13 };
14 
15 int main()
16 {
18   boost::shared_ptr<FILE> spfile(::fopen("tmp.txt", "a+"), 
19                                  FILEDeleter());
20   time_t t;
21   time(&t);
22 
23   if (spfile) {
24     fprintf(spfile.get(), "tstamp: %s\n", ctime(&t));
25   }
26 }
```

我们将`fopen`返回的`FILE`指针包装在`shared_ptr<FILE>`对象中（第 18 行）。但是，`shared_ptr`模板对`FILE`指针一无所知，因此我们还必须指定删除策略。为此，我们定义了一个名为`FILEDeleter`的函数对象（第 5 行），它的重载函数调用运算符（`operator()`，第 7 行）接受`FILE`类型的参数，并在其上调用`fclose`（如果不为空）（第 10 行）。临时的`FILEDeleter`实例作为第二个删除器参数（第 19 行）传递给`shared_ptr<FILE>`的构造函数。`shared_ptr<FILE>`的析构函数调用传递的删除器对象上的重载函数调用运算符，将存储的`FILE`指针作为参数传递。在这种情况下，重载的`operator->`几乎没有用处，因此通过使用`get`成员函数访问原始指针执行包装指针的所有操作（第 24 行）。我们还可以在`FILEDeleter`函数对象的位置使用 lambda 表达式。我们在第七章中介绍 Lambda 表达式，*高阶和编译时编程*。

如果您可以访问 C++11，最好始终使用`std::unique_ptr`来实现这些目的。使用`std::unique_ptr`，您必须为删除器的类型指定第二个模板参数。前面的例子将使用一个`std::unique_ptr`，只需进行以下编辑：

```cpp
 1 #include <memory>
...
18   std::unique_ptr<FILE, FILEDeleter> spfile(::fopen("tmp.txt", 
19                                             "a+"), FILEDeleter());
```

我们包括 C++标准头文件`memory`，而不是`boost/shared_ptr.hpp`（第 1 行），并将`fopen`调用返回的`FILE`指针包装在`unique_ptr`实例中（第 18 行），传递给它一个临时的`FILEDeleter`实例（第 19 行）。唯一的额外细节是`unique_ptr`模板的第二个类型参数，指定删除器的类型。我们还可以在 FILEDeleter 函数对象的位置使用 C++ 11 Lambda 表达式。在介绍 Lambda 表达式之后的章节中，我们将看到这种用法。

# 自测问题

对于多项选择题，选择所有适用的选项：

1.  亚伯拉罕的异常安全性保证是什么？

a. 基本的，弱的，和强的

b. 基本的，强的，不抛出异常

c. 弱的，强的，不抛出异常

d. 无，基本的，和强的

1.  `boost::scoped_ptr`和`std::unique_ptr`之间的主要区别是什么？

a. `boost::scoped_ptr`不支持移动语义

b. `std::scoped_ptr`没有数组的部分特化

c. `std::unique_ptr`可以存储在 STL 容器中

d. `std::unique_ptr`支持自定义删除器

1.  为什么`boost::shared_ptr`比其他智能指针更重？

a. 它使用共享引用计数

b. 它支持复制和移动语义

c. 它使用每个封装对象的两个分配

d. 它不比其他智能指针更重

1.  使用`boost::make_shared`创建`shared_ptr`的缺点是什么？

a. 它比直接实例化`boost::shared_ptr`慢

b. 它不是线程安全的

c. 它直到所有`weak_ptr`引用过期才释放对象内存

d. 它在 C++11 标准中不可用

1.  `boost::shared_ptr`和`std::unique_ptr`之间的主要区别是什么？

a. `std::unique_ptr`不支持复制语义

b. `std::unique_ptr`不支持移动语义

c. `boost::shared_ptr`不支持自定义删除器

d. `boost::shared_ptr`不能用于数组

1.  如果您想从类`X`的成员函数中返回包装`this`指针的`shared_ptr<X>`，以下哪个会起作用？

a. `return boost::shared_ptr<X>(this)`

b. `boost::enable_shared_from_this`

c. `boost::make_shared`

d. `boost::enable_shared_from_raw`

# 总结

本章明确了代码异常安全性的要求，然后定义了使用智能指针以异常安全的方式管理动态分配对象的各种方法。我们研究了来自 Boost 和新的 C++11 标准引入的智能指针模板，并理解了不同的所有权语义以及侵入式和非侵入式引用计数。我们还有机会看看如何调整一些智能指针模板来管理非内存资源。

希望您已经理解了各种所有权语义，并能够明智地将本章中的技术应用于这些场景。在智能指针库中有一些我们没有详细介绍的功能，比如`boost::shared_array`和`boost::enable_shared_from_raw`。您应该自行进一步探索它们，重点关注它们的适用性和缺陷。在下一章中，我们将学习使用 Boost 的字符串算法处理文本数据的一些巧妙而有用的技术。

# 参考资料

+   零规则：[`en.cppreference.com/w/cpp/language/rule_of_three`](http://en.cppreference.com/w/cpp/language/rule_of_three)

+   *设计 C++接口-异常安全性*，*Mark Radford*：[`accu.org/index.php/journals/444`](http://accu.org/index.php/journals/444)

+   *异常安全性分析*，*Andrei Alexandrescu 和 David B. Held*：[`erdani.com/publications/cuj-2003-12.pdf`](http://erdani.com/publications/cuj-2003-12.pdf)
