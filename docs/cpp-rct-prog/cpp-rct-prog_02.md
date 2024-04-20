# 第二章：现代 C++及其关键习语之旅

经典的 C++编程语言在 1998 年被标准化，随后在 2003 年进行了一次小的修订（主要是更正）。为了支持高级抽象，开发人员依赖于 Boost ([`www.boost.org`](http://www.boost.org))库和其他公共领域库。由于下一波标准化的到来，语言（从 C++ 11 开始）得到了增强，现在开发人员可以在不依赖外部库的情况下编码大多数其他语言支持的抽象。甚至线程和文件系统接口，原本属于库的范畴，现在已成为标准语言的一部分。现代 C++（代表 C++版本 11/14/17）包含了对语言和其库的出色增强，使得 C++成为编写工业级生产软件的事实选择。本章涵盖的功能是程序员必须了解的最小功能集，以便使用响应式编程构造，特别是 RxCpp。本章的主要目标是介绍语言的最重要的增强功能，使得实现响应式编程构造更加容易，而不需要使用神秘的语言技术。本章将涵盖以下主题：

+   C++编程语言设计的关键问题

+   一些用于编写更好代码的 C++增强功能

+   通过右值引用和移动语义实现更好的内存管理

+   使用增强的智能指针实现更好的对象生命周期管理

+   使用 Lambda 函数和表达式进行行为参数化

+   函数包装器（`std::function`类型）

+   其他功能

+   编写迭代器和观察者（将所有内容整合在一起）

# C++编程语言的关键问题

就开发人员而言，C++编程语言设计者关注的三个关键问题是（现在仍然是）：

+   零成本抽象 - 高级抽象不会带来性能惩罚

+   表现力 - 用户定义类型（UDT）或类应该与内置类型一样具有表现力

+   可替代性 - UDT 可以在期望内置类型的任何地方替代（如通用数据结构和算法）

我们将简要讨论这些内容。

# 零成本抽象

C++编程语言一直帮助开发人员编写利用微处理器的代码（生成的代码运行在微处理器上），并在需要时提高抽象级别。在提高抽象级别的同时，语言的设计者们一直试图最小化（几乎消除）性能开销。这被称为零成本抽象或零开销成本抽象。你所付出的唯一显著代价是间接调用的成本（通过函数指针）来分派虚拟函数。尽管向语言添加了大量功能，设计者们仍然保持了语言从一开始就暗示的“零成本抽象”保证。

# 表现力

C++帮助开发人员编写用户定义类型或类，可以像编程语言的内置类型一样具有表现力。这使得可以编写任意精度算术类（在某些语言中被称为`BigInteger`/`BigFloat`），其中包含了双精度或浮点数的所有特性。为了说明，我们定义了一个`SmartFloat`类，它包装了 IEEE 双精度浮点数，并重载了大多数双精度数据类型可用的运算符。以下代码片段显示，可以编写模仿内置类型（如 int、float 或 double）语义的类型：

```cpp
//---- SmartFloat.cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
class SmartFloat {
     double _value; // underlying store
   public:
      SmartFloat(double value) : _value(value) {}
      SmartFloat() : _value(0) {}
      SmartFloat( const SmartFloat& other ) { _value = other._value; }
      SmartFloat& operator = ( const SmartFloat& other ) {
          if ( this != &other ) { _value = other._value;}
          return *this;
      }
      SmartFloat& operator = (double value )
       { _value = value; return *this;}
      ~SmartFloat(){ }
```

`SmartFloat`类包装了一个 double 值，并定义了一些构造函数和赋值运算符来正确初始化实例。在下面的代码片段中，我们将定义一些操作符来增加值。前缀和后缀变体的操作符都已定义：

```cpp
      SmartFloat& operator ++ () { _value++; return *this; }
      SmartFloat operator ++ (int) { // postfix operator
             SmartFloat nu(*this); ++_value; return nu;
      }
      SmartFloat& operator -- () { _value--; return *this; }
      SmartFloat operator -- (int) {
           SmartFloat nu(*this); --_value; return nu;
      }
```

前面的代码片段实现了增量运算符（前缀和后缀），仅用于演示目的。在真实的类中，我们将检查浮点溢出和下溢，以使代码更加健壮。包装类型的整个目的是编写健壮的代码！

```cpp
     SmartFloat& operator += ( double x ) { _value += x; return *this;}
     SmartFloat& operator -= ( double x ) { _value -= x;return *this; }
     SmartFloat& operator *= ( double x ) { _value *= x; return *this;}
     SmartFloat& operator /= ( double x ) { _value /= x; return *this;}
```

前面的代码片段实现了 C++风格的赋值运算符，再次为了简洁起见，我们没有检查是否存在任何浮点溢出或下溢。我们也没有处理异常，以保持清单的简洁。

```cpp
      bool operator > ( const SmartFloat& other )
        { return _value > other._value; }
      bool operator < ( const SmartFloat& other )
       {return _value < other._value;}
      bool operator == ( const SmartFloat& other )
        { return _value == other._value;}
      bool operator != ( const SmartFloat& other )
        { return _value != other._value;}
      bool operator >= ( const SmartFloat& other )
        { return _value >= other._value;}
      bool operator <= ( const SmartFloat& other )
        { return _value <= other._value;}
```

前面的代码实现了关系运算符，并且大部分与双精度浮点数相关的语义都已经实现如下：

```cpp
      operator int () { return _value; }
      operator double () { return _value;}
};
```

为了完整起见，我们已经实现了到`int`和`double`的转换运算符。我们将编写两个函数来聚合存储在数组中的值。第一个函数期望一个`double`数组作为参数，第二个函数期望一个`SmartFloat`数组作为参数。两个例程中的代码是相同的，只是类型不同。两者将产生相同的结果：

```cpp
double Accumulate( double a[] , int count ){
    double value = 0;
    for( int i=0; i<count; ++i) { value += a[i]; }
    return value;
}
double Accumulate( SmartFloat a[] , int count ){
    SmartFloat value = 0;
    for( int i=0; i<count; ++i) { value += a[i]; }
    return value;
}
int main() {
    // using C++ 1z's initializer list
    double x[] = { 10.0,20.0,30,40 };
    SmartFloat y[] = { 10,20.0,30,40 };
    double res = Accumulate(x,4); // will call the double version
    cout << res << endl;
    res = Accumulate(y,4); // will call the SmartFloat version
    cout << res << endl;
}
```

C++语言帮助我们编写富有表现力的类型，增强基本类型的语义。语言的表现力还帮助我们使用语言支持的多种技术编写良好的值类型和引用类型。通过支持运算符重载、转换运算符、放置 new 和其他相关技术，与其同时代的其他语言相比，该语言已将类设计提升到了一个更高的水平。但是，能力与责任并存，有时语言会给你足够的自由让你自食其果。

# 可替代性

在前面的例子中，我们看到了如何使用用户定义的类型来表达对内置类型进行的所有操作。C++的另一个目标是以一种通用的方式编写代码，其中我们可以替换一个模拟内置类型（如`float`、`double`、`int`等）语义的用户定义类：

```cpp
//------------- from SmartValue.cpp
template <class T>
T Accumulate( T a[] , int count ) {
    T value = 0;
    for( int i=0; i<count; ++i) { value += a[i]; }
    return value;
}
int main(){
    //----- Templated version of SmartFloat
    SmartValue<double> y[] = { 10,20.0,30,40 };
    double res = Accumulate(y,4);
    cout << res << endl;
}
```

C++编程语言支持不同的编程范式，前面概述的三个原则只是其中的一些。该语言支持可以帮助创建健壮类型（特定领域）以编写更好代码的构造。这三个原则确实为我们带来了一个强大而快速的编程语言。现代 C++确实添加了许多新的抽象，以使程序员的生活更加轻松。但是，为了实现这些目标，之前概述的三个设计原则并没有以任何方式被牺牲。这在一定程度上是可能的，因为语言由于模板机制的无意中图灵完备性而具有元编程支持。使用您喜欢的搜索引擎阅读有关**模板元编程**（**TMP**）和图灵完备性的内容。 

# C++增强以编写更好的代码

在过去的十年里，编程语言的世界发生了很大变化，这些变化应该反映在 C++编程语言的新版本中。现代 C++中的大部分创新涉及处理高级抽象，并引入函数式编程构造以支持语言级并发。大多数现代语言都有垃圾收集器，运行时管理这些复杂性。C++编程语言没有自动垃圾收集作为语言标准的一部分。C++编程语言以其隐式的零成本抽象保证（你不用为你不使用的东西付费）和最大的运行时性能，必须依靠大量的编译时技巧和元编程技术来实现 C#、Java 或 Scala 等语言支持的抽象级别。其中一些在以下部分中概述，你可以自行深入研究这些主题。网站[`en.cppreference.com`](http://en.cppreference.com)是提高你对 C++编程语言知识的一个好网站。

# 类型推断和推理

现代 C++语言编译器在程序员指定的表达式和语句中推断类型方面做得非常出色。大多数现代编程语言都支持类型推断，现代 C++也是如此。这是从 Haskell 和 ML 等函数式编程语言借鉴来的习惯用法。类型推断已经在 C#和 Scala 编程语言中可用。我们将编写一个小程序来启动我们的类型推断：

```cpp
//----- AutoFirst.cpp
#include <iostream>
#include <vector>
using namespace std;
int main(){
    vector<string> vt = {"first", "second", "third", "fourth"};
    //--- Explicitly specify the Type ( makes it verbose)
    for (vector<string>::iterator it = vt.begin();
        it != vt.end(); ++it)
    cout << *it << " ";
    //--- Let the compiler infer the type for us
    for (auto it2 = vt.begin(); it2 != vt.end(); ++it2)
        cout << *it2 << " ";
    return 0;
}
```

`auto`关键字指定变量的类型将根据初始化和表达式中指定的函数的返回值由编译器推导出来。在这个特定的例子中，我们并没有获得太多。随着我们的声明变得更加复杂，最好让编译器进行类型推断。我们的代码清单将使用 auto 来简化整本书的代码。现在，让我们编写一个简单的程序来更清楚地阐明这个想法：

```cpp
//----- AutoSecond.cpp
#include <iostream>
#include <vector>
#include <initializer_list>
using namespace std;
int main() {
    vector<double> vtdbl = {0, 3.14, 2.718, 10.00};
    auto vt_dbl2 = vtdbl; // type will be deduced
    auto size = vt_dbl2.size(); // size_t
    auto &rvec = vtdbl; // specify a auto reference
    cout << size << endl;
    // Iterate - Compiler infers the type
    for ( auto it = vtdbl.begin(); it != vtdbl.end(); ++it)
        cout << *it << " ";
    // 'it2' evaluates to iterator to vector of double
    for (auto it2 = vt_dbl2.begin(); it2 != vt_dbl2.end(); ++it2)
        cout << *it2 << " ";
    // This will change the first element of vtdbl vector
    rvec[0] = 100;
    // Now Iterate to reflect the type
    for ( auto it3 = vtdbl.begin(); it3 != vtdbl.end(); ++it3)
        cout << *it3 << " ";
    return 0;
}
```

前面的代码演示了在编写现代 C++代码时使用类型推断。C++编程语言还有一个新关键字，用于查询给定参数的表达式的类型。关键字的一般形式是`decltype(<expr>)`。以下程序有助于演示这个特定关键字的用法：

```cpp
//---- Decltype.cpp
#include <iostream>
using namespace std;
int foo() { return 10; }
char bar() { return 'g'; }
auto fancy() -> decltype(1.0f) { return 1;} //return type is float
int main() {
    // Data type of x is same as return type of foo()
    // and type of y is same as return type of bar()
    decltype(foo()) x;
    decltype(bar()) y;
    //--- in g++, Should print i => int
    cout << typeid(x).name() << endl;
    //--- in g++, Should print c => char 
    cout << typeid(y).name() << endl;
    struct A { double x; };
    const A* a = new A();
    decltype(a->x) z; // type is double
    decltype((a->x)) t= z; // type is const double&
    //--- in g++, Should print  d => double
    cout << typeid(z).name() << endl;
    cout << typeid(t).name() << endl;
    //--- in g++, Should print  f => float
    cout << typeid(decltype(fancy())).name() << endl;
    return 0;
}
```

`decltype`是一个编译时构造，它有助于指定变量的类型（编译器将进行艰苦的工作来找出它），并且还可以帮助我们强制变量的类型（参见前面的`fancy()`函数）。

# 变量的统一初始化

经典 C++对变量的初始化有一些特定的 ad-hoc 语法。现代 C++支持统一初始化（我们已经在类型推断部分看到了示例）。语言为开发人员提供了辅助类，以支持他们自定义类型的统一初始化：

```cpp
//----------------Initialization.cpp
#include <iostream>
#include <vector>
#include <initializer_list>
using namespace std;
template <class T>
struct Vector_Wrapper {
    std::vector<T> vctr;
    Vector_Wrapper(std::initializer_list<T> l) : vctr(l) {}
    void Append(std::initializer_list<T> l)
    { vctr.insert(vctr.end(), l.begin(), l.end());}
};
int main() {
    Vector_Wrapper<int> vcw = {1, 2, 3, 4, 5}; // list-initialization
    vcw.Append({6, 7, 8}); // list-initialization in function call
    for (auto n : vcw.vctr) { std::cout << n << ' '; }
    std::cout << '\n';
}
```

前面的清单显示了如何使程序员创建的自定义类启用初始化列表。

# 可变模板

在 C++ 11 及以上版本中，标准语言支持可变模板。可变模板是一个接受可变数量的模板参数的模板类或模板函数。在经典 C++中，模板实例化发生在固定数量的参数中。可变模板在类级别和函数级别都得到支持。在本节中，我们将处理可变函数，因为它们在编写函数式程序、编译时编程（元编程）和可管道函数中被广泛使用：

```cpp
//Variadic.cpp
#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>
using namespace std;
//--- add given below is a base case for ending compile time
//--- recursion
int add() { return 0; } // end condition
//---- Declare a Variadic function Template
//---- ... is called parameter pack. The compiler
//--- synthesize a function based on the number of arguments
//------ given by the programmer.
//----- decltype(auto) => Compiler will do Type Inference
template<class T0, class ... Ts>
decltype(auto) add(T0 first, Ts ... rest) {
    return first + add(rest ...);
}
int main() { int n = add(0,2,3,4); cout << n << endl; }
```

在上面的代码中，编译器根据传递的参数数量合成一个函数。编译器理解`add`是一个可变参数函数，并通过在编译时递归展开参数来生成代码。编译时递归将在编译器处理完所有参数时停止。基本情况版本是一个提示编译器停止递归的方法。下一个程序展示了可变模板和完美转发如何用于编写接受任意数量参数的函数：

```cpp
//Variadic2.cpp
#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>
using namespace std;
//--------- Print values to the console for basic types
//-------- These are base case versions
void EmitConsole(int value) { cout << "Integer: " << value << endl; }
void EmitConsole(double value) { cout << "Double: " << value << endl; }
void EmitConsole(const string& value){cout << "String: "<<value<< endl; }
```

`EmitConsole` 的三个变体将参数打印到控制台。我们有打印`int`、`double`和`string`的函数。利用这些函数作为基本情况，我们将编写一个使用通用引用和完美转发的函数，以编写接受任意值的函数：

```cpp
template<typename T>
void EmitValues(T&& arg) { EmitConsole(std::forward<T>(arg)); }

template<typename T1, typename... Tn>
void EmitValues(T1&& arg1, Tn&&... args){
    EmitConsole(std::forward<T1>(arg1));
    EmitValues(std::forward<Tn>(args)...);
}

int main() { EmitValues(0,2.0,"Hello World",4); }
```

# 右值引用

如果你长时间在 C++中编程，你可能知道 C++引用可以帮助你给变量取别名，并且可以对引用进行赋值以反映变量别名的变化。C++支持的引用类型称为左值引用（因为它们是引用可以出现在赋值的左侧的变量的引用）。以下代码片段展示了左值引用的用法：

```cpp
//---- Lvalue.cpp
#include <iostream>
using namespace std;
int main() {
  int i=0;
  cout << i << endl; //prints 0
  int& ri = i;
  ri = 20;
  cout << i << endl; // prints 20
}
```

`int&` 是左值引用的一个实例。在现代 C++中，有右值引用的概念。右值被定义为任何不是左值的东西，可以出现在赋值的右侧。在经典的 C++中，没有右值引用的概念。现代 C++引入了它：

```cpp
///---- Rvaluref.cpp
#include <iostream>using namespace std;
int main() {
    int&& j = 42;int x = 3,y=5; int&& z = x + y; cout << z << endl;
    z = 10; cout << z << endl;j=20;cout << j << endl;
}
```

右值引用由两个`&&`表示。以下程序将清楚地演示了在调用函数时使用右值引用：

```cpp
//------- RvaluerefCall.cpp
#include <iostream>
using namespace std;
void TestFunction( int & a ) {cout << a << endl;}
void TestFunction( int && a ){
    cout << "rvalue references" << endl;
    cout << a << endl;
}
int main() {
int&& j = 42;
int x = 3,y=5;
int&& z = x + y;
    TestFunction(x + y ); // Should call rvalue reference function
    TestFunction(j); // Calls Lvalue Refreence function
}
```

右值引用的真正威力在于内存管理方面。C++编程语言具有复制构造函数和赋值运算符的概念。它们大多数情况下是复制源对象的内容。借助右值引用，可以通过交换指针来避免昂贵的复制，因为右值引用是临时的或中间表达式。下一节将演示这一点。

# 移动语义

C++编程语言隐式地为我们设计的每个类提供了一个复制构造函数、赋值运算符和一个析构函数（有时是虚拟的）。这是为了在克隆对象或对现有对象进行赋值时进行资源管理。有时复制对象是非常昂贵的，通过指针的所有权转移有助于编写快速的代码。现代 C++提供了移动构造函数和移动赋值运算符的功能，以帮助开发人员避免复制大对象，在创建新对象或对新对象进行赋值时。右值引用可以作为一个提示，告诉编译器在涉及临时对象时，构造函数的移动版本或赋值的移动版本更适合于上下文：

```cpp
//----- FloatBuffer.cpp
#include <iostream>
#include <vector>
using namespace std;
class FloatBuffer {
    double *bfr; int count;
public:
    FloatBuffer():bfr(nullptr),count(0){}
    FloatBuffer(int pcount):bfr(new double[pcount]),count(pcount){}
        // Copy constructor.
    FloatBuffer(const FloatBuffer& other) : count(other.count)
        , bfr(new double[other.count])
    { std::copy(other.bfr, other.bfr + count, bfr); }
    // Copy assignment operator - source code is obvious
    FloatBuffer& operator=(const FloatBuffer& other) {
        if (this != &other) {
          if ( bfr != nullptr) 
            delete[] bfr; // free memory of the current object
            count = other.count;
            bfr = new double[count]; //re-allocate
            std::copy(other.bfr, other.bfr + count, bfr);
        }
        return *this;
    }
    // Move constructor to enable move semantics
    // The Modern STL containers supports move sementcis
    FloatBuffer(FloatBuffer&& other) : bfr(nullptr) , count(0) {
    cout << "in move constructor" << endl;
    // since it is a move constructor, we are not copying elements from
    // the source object. We just assign the pointers to steal memory
    bfr = other.bfr;
    count = other.count;
    // Now that we have grabbed our memory, we just assign null to
    // source pointer
    other.bfr = nullptr;
    other.count = 0;
    }
// Move assignment operator.
FloatBuffer& operator=(FloatBuffer&& other) {
    if (this != &other)
    {
        // Free the existing resource.
        delete[] bfr;
       // Copy the data pointer and its length from the
       // source object.
       bfr = other.bfr;
       count = other.count;
       // We have stolen the memory, now set the pinter to null
       other.bfr = nullptr;
       other.count = 0;
    }
    return *this;
}

};
int main() {
    // Create a vector object and add a few elements to it.
    // Since STL supports move semantics move methods will be called.
    // in this particular case (Modern Compilers are smart)
    vector<FloatBuffer> v;
    v.push_back(FloatBuffer(25));
    v.push_back(FloatBuffer(75));
}
```

`std::move` 函数可用于指示（在传递参数时）候选对象是可移动的，编译器将调用适当的方法（移动赋值或移动构造函数）来优化与内存管理相关的成本。基本上，`std::move` 是对右值引用的`static_cast`。

# 智能指针

管理对象生命周期一直是 C++编程语言的一个问题。如果开发人员不小心，程序可能会泄漏内存并降低性能。智能指针是围绕原始指针的包装类，其中重载了解引用(*)和引用(->)等操作符。智能指针可以进行对象生命周期管理，充当有限形式的垃圾回收，释放内存等。现代 C++语言具有：

+   `unique_ptr<T>`

+   `shared_ptr<T>`

+   `weak_ptr<T>`

`unique_ptr<T>`是一个具有独占所有权的原始指针的包装器。以下代码片段将演示`<unique_ptr>`的使用：

```cpp
//---- Unique_Ptr.cpp
#include <iostream>
#include <deque>#include <memory>
using namespace std;
int main( int argc , char **argv ) {
    // Define a Smart Pointer for STL deque container...
    unique_ptr< deque<int> > dq(new deque<int>() );
    //------ populate values , leverages -> operator
    dq->push_front(10); dq->push_front(20);
    dq->push_back(23); dq->push_front(16);
    dq->push_back(41);
    auto dqiter = dq->begin();
    while ( dqiter != dq->end())
    { cout << *dqiter << "\n"; dqiter++; }
    //------ SmartPointer will free reference
    //------ and it's dtor will be called here
    return 0;
}
```

`std::shared_ptr`是一个智能指针，它使用引用计数来跟踪对对象实例的引用。当指向它的最后一个`shared_ptr`被销毁或重置时，底层对象将被销毁：

```cpp
//----- Shared_Ptr.cpp
#include <iostream>
#include <memory>
#include <stdio.h>
using namespace std;
////////////////////////////////////////
// Even If you pass shared_ptr<T> instance
// by value, the update is visible to callee
// as shared_ptr<T>'s copy constructor reference
// counts to the orgininal instance
//

void foo_byvalue(std::shared_ptr<int> i) { (*i)++;}

///////////////////////////////////////
// passed by reference,we have not
// created a copy.
//
void foo_byreference(std::shared_ptr<int>& i) { (*i)++; }
int main(int argc, char **argv )
{
    auto sp = std::make_shared<int>(10);
    foo_byvalue(sp);
    foo_byreference(sp);
    //--------- The output should be 12
    std::cout << *sp << std::endl;
}
```

`std:weak_ptr`是一个原始指针的容器。它是作为`shared_ptr`的副本创建的。`weak_ptr`的存在或销毁对`shared_ptr`或其其他副本没有影响。在所有`shared_ptr`的副本被销毁后，所有`weak_ptr`的副本都变为空。以下程序演示了使用`weak_ptr`来检测失效指针的机制：

```cpp
//------- Weak_Ptr.cpp
#include <iostream>
#include <deque>
#include <memory>

using namespace std;
int main( int argc , char **argv )
{
    std::shared_ptr<int> ptr_1(new int(500));
    std::weak_ptr<int> wptr_1 = ptr_1;
    {
        std::shared_ptr<int> ptr_2 = wptr_1.lock();
        if(ptr_2)
        {
            cout << *ptr_2 << endl; // this will be exeucted
        }
    //---- ptr_2 will go out of the scope
    }

    ptr_1.reset(); //Memory is deleted.

    std::shared_ptr<int> ptr_3= wptr_1.lock();
    //-------- Always else part will be executed
    //-------- as ptr_3 is nullptr now 
    if(ptr_3)
        cout << *ptr_3 << endl;
    else
        cout << "Defunct Pointer" << endl;
    return 0;
}
```

经典 C++有一个名为`auto_ptr`的智能指针类型，已从语言标准中删除。需要使用`unique_ptr`代替。

# Lambda 函数

C++语言的一个主要增强是 Lambda 函数和 Lambda 表达式。它们是程序员可以在调用站点定义的匿名函数，用于执行一些逻辑。这简化了逻辑，代码的可读性也以显着的方式增加。

与其定义 Lambda 函数是什么，不如编写一段代码来帮助我们计算`vector<int>`中正数的数量。在这种情况下，我们需要过滤掉负值并计算剩下的值。我们将使用 STL `count_if`来编写代码：

```cpp
//LambdaFirst.cpp
#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>
using namespace std;
int main() {
    auto num_vect =
        vector<int>{ 10, 23, -33, 15, -7, 60, 80};
    //---- Define a Lambda Function to Filter out negatives
    auto filter = [](int const value) {return value > 0; };
    auto cnt= count_if(
        begin(num_vect), end(num_vect),filter);
    cout << cnt << endl;
}
```

在上面的代码片段中，变量 filter 被赋予了一个匿名函数，并且我们在`count_if STL`函数中使用了 filter。现在，让我们编写一个简单的 Lambda 函数，在函数调用时指定。我们将使用 STL accumulate 来聚合向量中的值：

```cpp
//-------------- LambdaSecond.cpp
#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;
int main() {
    auto num_vect =
        vector<int>{ 10, 23, -33, 15, -7, 60, 80};
    //-- Define a BinaryOperation Lambda at the call site
    auto accum = std::accumulate(
        std::begin(num_vect), std::end(num_vect), 0,
        [](auto const s, auto const n) {return s + n;});
    cout << accum << endl;
}
```

# 函数对象和 Lambda

在经典的 C++中，使用 STL 时，我们广泛使用函数对象或函数符号，通过重载函数运算符来编写转换过滤器和对 STL 容器执行减少操作：

```cpp
//----- LambdaThird.cpp
#include <iostream>
#include <numeric>
using namespace std;
//////////////////////////
// Functors to add and multiply two numbers
template <typename T>
struct addition{
    T operator () (const T& init, const T& a ) { return init + a; }
};
template <typename T>
struct multiply {
    T operator () (const T& init, const T& a ) { return init * a; }
};
int main()
{
    double v1[3] = {1.0, 2.0, 4.0}, sum;
    sum = accumulate(v1, v1 + 3, 0.0, addition<double>());
    cout << "sum = " << sum << endl;
    sum = accumulate(v1,v1+3,0.0, [] (const double& a ,const double& b   ) {
        return a +b;
    });
    cout << "sum = " << sum << endl;
    double mul_pi = accumulate(v1, v1 + 3, 1.0, multiply<double>());
    cout << "mul_pi = " << mul_pi << endl;
    mul_pi= accumulate(v1,v1+3,1, [] (const double& a , const double& b ){
        return a *b;
    });
    cout << "mul_pi = " << mul_pi << endl;
}
```

以下程序清楚地演示了通过编写一个玩具排序程序来使用 Lambda。我们将展示如何使用函数对象和 Lambda 来编写等效的代码。该代码以一种通用的方式编写，但假设数字是预期的（`double`，`float`，`integer`或用户定义的等效类型）：

```cpp
/////////////////
//-------- LambdaFourth.cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
//--- Generic functions for Comparison and Swap
template <typename T>
bool Cmp( T& a , T&b ) {return ( a > b ) ? true: false;}
template <typename T>
void Swap( T& a , T&b ) { T c = a;a = b;b = c;}
```

`Cmp`和`Swap`是通用函数，将用于比较相邻元素和交换元素，同时执行排序操作：

```cpp
template <typename T>
void BubbleSortFunctor( T *arr , int length ) {
    for( int i=0; i< length-1; ++i )
        for(int j=i+1; j< length; ++j )
            if ( Cmp( arr[i] , arr[j] ) )
                Swap(arr[i],arr[j] );
}
```

有了 Cmp 和 Swap，编写冒泡排序就变得简单了。我们需要有一个嵌套循环，在其中我们将比较两个元素，如果 Cmp 返回 true，我们将调用 Swap 来交换值：

```cpp
template <typename T>
void BubbleSortLambda( T *arr , int length ) {
    auto CmpLambda = [] (const auto& a , const auto& b )
    { return ( a > b ) ? true: false; };
    auto SwapLambda = [] ( auto& a , auto& b )
    { auto c = a;a = b;b = c;};
    for( int i=0; i< length-1; ++i )
        for(int j=i+1; j< length; ++j )
            if ( CmpLambda( arr[i] , arr[j] ) )
                SwapLambda (arr[i],arr[j] );
}
```

在上面的例程中，我们将比较和交换函数定义为 Lambda。Lambda 函数是一种在调用站点内指定代码或表达式的机制，通常称为匿名函数。定义可以使用 C++语言指定的语法，并且可以赋值给变量，作为参数传递，或者从函数返回。在上面的函数中，变量`CmpLambda`和`SwapLambda`是 Lambda 语法中指定的匿名函数的示例。Lambda 函数的主体与之前的函数版本没有太大的不同。要了解有关 Lambda 函数和表达式的更多信息，可以参考[`en.cppreference.com/w/cpp/language/lambda`](http://en.cppreference.com/w/cpp/language/lambda)页面。

```cpp
template <typename T>
void Print( const T& container){
    for(auto i = container.begin() ; i != container.end(); ++i )
        cout << *i << "\n" ;
}
```

`Print`例程只是循环遍历容器中的元素，并将内容打印到控制台：

```cpp
int main( int argc , char **argv ){
    double ar[4] = {20,10,15,-41};
    BubbleSortFunctor(ar,4);
    vector<double> a(ar,ar+4);
    Print(a);
    cout << "=========================================" << endl;
    ar[0] = 20;ar[1] = 10;ar[2] = 15;ar[3] = -41;
    BubbleSortLambda(ar,4);
    vector<double> a1(ar,ar+4);
    Print(a1);
    cout << "=========================================" << endl;
}
```

# 组合、柯里化和部分函数应用

Lambdas 的一个优点是你可以将两个函数组合在一起，创建函数的组合，就像你在数学中所做的那样（在数学和函数式编程的上下文中阅读有关函数组合的内容，使用喜欢的搜索引擎）。以下程序演示了这个想法。这是一个玩具实现，撰写通用实现超出了本章的范围：

```cpp
//------------ Compose.cpp
//----- g++ -std=c++1z Compose.cpp
#include <iostream>
using namespace std;
//---------- base case compile time recursion
//---------- stops here
template <typename F, typename G>
auto Compose(F&& f, G&& g)
{ return = { return f(g(x)); };}
//----- Performs compile time recursion based
//----- on number of parameters
template <typename F, typename... R>
auto Compose(F&& f, R&&... r){
    return = { return f(Compose(r...)(x)); };
}
```

`Compose`是一个可变模板函数，编译器通过递归扩展`Compose`参数生成代码，直到处理完所有参数。在前面的代码中，我们使用`[=]`指示编译器应该按值捕获 Lambda 体中引用的所有变量。您可以在函数式编程的上下文中学习更多关于闭包和变量捕获的内容。C++语言允许通过值（以及使用`[&]`）或通过显式指定要捕获的变量（如`[&var]`）来灵活地`Capture`变量。

函数式编程范式基于由美国数学家阿隆佐·邱奇发明的一种数学形式主义，称为 Lambda 演算。Lambda 演算仅支持一元函数，柯里化是一种将多参数函数分解为一系列一次接受一个参数的函数评估的技术。

使用 Lambdas 和以特定方式编写函数，我们可以在 C++中模拟柯里化：

```cpp
auto CurriedAdd3(int x) {
    return x { //capture x
        return x, y{ return x + y + z; };
    };
};
```

部分函数应用涉及将具有多个参数的函数转换为固定数量的参数。如果固定数量的参数少于函数的 arity（参数计数），则将返回一个新函数，该函数期望其余的参数。当接收到所有参数时，将调用该函数。我们可以将部分应用视为某种形式的记忆化，其中参数被缓存，直到我们接收到所有参数以调用它们。

在以下代码片段中，我们使用了模板参数包和可变模板。模板参数包是一个接受零个或多个模板参数（非类型、类型或模板）的模板参数。函数参数包是一个接受零个或多个函数参数的函数参数。至少有一个参数包的模板称为可变模板。对参数包和可变模板的良好理解对于理解`sizeof...`构造是必要的。

```cpp
template <typename... Ts>
auto PartialFunctionAdd3(Ts... xs) {
    //---- http://en.cppreference.com/w/cpp/language/parameter_pack
    //---- http://en.cppreference.com/w/cpp/language/sizeof...
    static_assert(sizeof...(xs) <= 3);
    if constexpr (sizeof...(xs) == 3){
        // Base case: evaluate and return the sum.
        return (0 + ... + xs);
    }
    else{
        // Recursive case: bind `xs...` and return another
        return xs...{
            return PartialFunctionAdd3(xs..., ys...);
        };
    }
}
int main() {
    // ------------- Compose two functions together
    //----https://en.wikipedia.org/wiki/Function_composition
    auto val = Compose(
        [](int const a) {return std::to_string(a); },
        [](int const a) {return a * a; })(4); // val = "16"
    cout << val << std::endl; //should print 16
    // ----------------- Invoke the Curried function
    auto p = CurriedAdd3(4)(5)(6);
    cout << p << endl;
    //-------------- Compose a set of function together
    auto func = Compose(
        [](int const n) {return std::to_string(n); },
        [](int const n) {return n * n; },
        [](int const n) {return n + n; },
        [](int const n) {return std::abs(n); });
    cout << func(5) << endl;
    //----------- Invoke Partial Functions giving different arguments
    PartialFunctionAdd3(1, 2, 3);
    PartialFunctionAdd3(1, 2)(3);
    PartialFunctionAdd3(1)(2)(3);
}
```

# 函数包装器

函数包装器是可以包装任何函数、函数对象或 Lambdas 成可复制对象的类。包装器的类型取决于类的函数原型。来自`<functional>`头文件的`std::function(<prototype>)`表示一个函数包装器：

```cpp
//---------------- FuncWrapper.cpp Requires C++ 17 (-std=c++1z )
#include <functional>
#include <iostream>
using namespace std;
//-------------- Simple Function call
void PrintNumber(int val){ cout << val << endl; }
// ------------------ A class which overloads function operator
struct PrintNumber {
    void operator()(int i) const { std::cout << i << '\n';}
};
//------------ To demonstrate the usage of method call
struct FooClass {
    int number;
    FooClass(int pnum) : number(pnum){}
    void PrintNumber(int val) const { std::cout << number + val<< endl; }
};
int main() {
    // ----------------- Ordinary Function Wrapped
    std::function<void(int)> 
    displaynum = PrintNumber;
    displaynum(0xF000);
    std::invoke(displaynum,0xFF00); //call through std::invoke
    //-------------- Lambda Functions Wrapped
    std::function<void()> lambdaprint = []() { PrintNumber(786); };
        lambdaprint();
        std::invoke(lambdaprint);
        // Wrapping member functions of a class
        std::function<void(const FooClass&, int)>
        class display = &FooClass::PrintNumber;
        // creating an instance
        const FooClass fooinstance(100);
        class display (fooinstance,100);
}
```

在接下来的章节中，我们将广泛使用`std::function`，因为它有助于将函数调用作为数据进行处理。

# 使用管道运算符将函数组合在一起

Unix 操作系统的命令行 shell 允许将一个函数的标准输出管道到另一个函数，形成一个过滤器链。后来，这个特性成为大多数操作系统提供的每个命令行 shell 的一部分。在编写函数式风格的代码时，当我们通过函数组合来组合方法时，由于深层嵌套，代码变得难以阅读。现在，使用现代 C++，我们可以重载管道（`|`）运算符，以允许将多个函数链接在一起，就像我们在 Unix shell 或 Windows PowerShell 控制台中执行命令一样。这就是为什么有人重新将 LISP 语言称为许多令人恼火和愚蠢的括号。RxCpp 库广泛使用`|`运算符来组合函数。以下代码帮助我们了解如何创建可管道化的函数。我们将看一下这个原则上如何实现。这里给出的代码仅用于解释目的：

```cpp
//---- PipeFunc2.cpp
//-------- g++ -std=c++1z PipeFunc2.cpp
#include <iostream>
using namespace std;

struct AddOne {
    template<class T>
    auto operator()(T x) const { return x + 1; }
};
struct SumFunction {
    template<class T>
    auto operator()(T x,T y) const { return x + y;} // Binary Operator
};
```

前面的代码创建了一组 Callable 类，并将其用作函数组合链的一部分。现在，我们需要创建一种机制，将任意函数转换为闭包：

```cpp
//-------------- Create a Pipable Closure Function (Unary)
//-------------- Uses Variadic Templates Paramter pack
template<class F>
struct PipableClosure : F{
    template<class... Xs>
    PipableClosure(Xs&&... xs) : // Xs is a universal reference
    F(std::forward<Xs>(xs)...) // perfect forwarding
    {}
};
//---------- A helper function which converts a Function to a Closure
template<class F>
auto MakePipeClosure(F f)
{ return PipableClosure<F>(std::move(f)); }
// ------------ Declare a Closure for Binary
//------------- Functions
//
template<class F>
struct PipableClosureBinary {
    template<class... Ts>
    auto operator()(Ts... xs) const {
        return MakePipeClosure(= -> decltype(auto)
        { return F()(x, xs...);}); }
};
//------- Declare a pipe operator
//------- uses perfect forwarding to invoke the function
template<class T, class F> //---- Declare a pipe operator
decltype(auto) operator|(T&& x, const PipableClosure<F>& pfn)
{ return pfn(std::forward<T>(x)); }

int main() {
    //-------- Declare a Unary Function Closure
    const PipableClosure<AddOne> fnclosure = {};
    int value = 1 | fnclosure| fnclosure;
    std::cout << value << std::endl;
    //--------- Decalre a Binary function closure
    const PipableClosureBinary<SumFunction> sumfunction = {};
    int value1 = 1 | sumfunction(2) | sumfunction(5) | fnclosure;
    std::cout << value1 << std::endl;
}
```

现在，我们可以创建一个带有一元函数作为参数的`PipableClosure`实例，并将一系列调用链接（或组合）到闭包中。前面的代码片段应该在控制台上打印出三。我们还创建了一个`PipableBinaryClosure`实例，以串联一元和二元函数。

# 杂项功能

到目前为止，我们已经介绍了从 C++ 11 标准开始的语言中最重要的语义变化。本章的目的是突出一些可能有助于编写现代 C++程序的关键变化。C++ 17 标准在语言中添加了一些新内容。我们将突出语言的一些其他特性来结束这个讨论。

# 折叠表达式

C++ 17 标准增加了对折叠表达式的支持，以简化可变函数的生成。编译器进行模式匹配，并通过推断程序员的意图生成代码。以下代码片段演示了这个想法：

```cpp
//---------------- Folds.cpp
//--------------- Requires C++ 17 (-std=c++1z )
//--------------- http://en.cppreference.com/w/cpp/language/fold
#include <functional>
#include <iostream>

using namespace std;
template <typename... Ts>
auto AddFoldLeftUn(Ts... args) { return (... + args); }
template <typename... Ts>
auto AddFoldLeftBin(int n,Ts... args){ return (n + ... + args);}
template <typename... Ts>
auto AddFoldRightUn(Ts... args) { return (args + ...); }
template <typename... Ts>
auto AddFoldRightBin(int n,Ts... args) { return (args + ... + n); }
template <typename T,typename... Ts>
auto AddFoldRightBinPoly(T n,Ts... args) { return (args + ... + n); }
template <typename T,typename... Ts>
auto AddFoldLeftBinPoly(T n,Ts... args) { return (n + ... + args); }

int main() {
    auto a = AddFoldLeftUn(1,2,3,4);
    cout << a << endl;
    cout << AddFoldRightBin(a,4,5,6) << endl;
    //---------- Folds from Right
    //---------- should produce "Hello  World C++"
    auto b = AddFoldRightBinPoly("C++ "s,"Hello "s,"World "s );
    cout << b << endl;
    //---------- Folds (Reduce) from Left
    //---------- should produce "Hello World C++"
    auto c = AddFoldLeftBinPoly("Hello "s,"World "s,"C++ "s );
    cout << c << endl;
}
```

控制台上的预期输出如下

```cpp
10
 25
 Hello World C++
 Hello World C++
```

# 变体类型

变体的极客定义将是“类型安全的联合”。在定义变体时，我们可以将一系列类型作为模板参数。在任何给定时间，对象将仅保存模板参数列表中的一种数据类型。如果我们尝试访问不包含当前值的索引，将抛出`std::bad_variant_access`异常。以下代码不处理此异常：

```cpp
//------------ Variant.cpp
//------------- g++ -std=c++1z Variant.cpp
#include <variant>
#include <string>
#include <cassert>
#include <iostream>
using namespace std;

int main(){
    std::variant<int, float,string> v, w;
    v = 12.0f; // v contains now contains float
    cout << std::get<1>(v) << endl;
    w = 20; // assign to int
    cout << std::get<0>(w) << endl;
    w = "hello"s; //assign to string
    cout << std::get<2>(w) << endl;
}
```

# 其他重要主题

现代 C++支持诸如语言级并发、内存保证和异步执行等功能，这些功能将在接下来的两章中介绍。该语言支持可选数据类型和`std::any`类型。其中最重要的功能之一是大多数 STL 算法的并行版本。

# 基于范围的 for 循环和可观察对象

在本节中，我们将实现自己编写的自定义类型上的基于范围的 for 循环，以帮助您了解如何将本章中提到的所有内容组合起来编写支持现代习语的程序。我们将实现一个返回在范围内的一系列数字的类，并将实现基于范围的 for 循环的值的迭代的基础设施支持。首先，我们将利用基于范围的 for 循环编写“Iterable/Iterator”（又名“Enumerable/Enumerable”）版本。经过一些调整，实现将转变为 Observable/Observer（响应式编程的关键接口）模式：此处 Observable/Observer 模式的实现仅用于阐明目的，不应被视为这些模式的工业级实现。

以下的`iterable`类是一个嵌套类：

```cpp
// Iterobservable.cpp
// we can use Range Based For loop as given below (see the main below)
// for (auto l : EnumerableRange<5, 25>()) { std::cout << l << ' '; }
// std::cout << endl;
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <functional>
using namespace std;

template<long START, long END>
class EnumerableRange {
public:

    class iterable : public std::iterator<
        std::input_iterator_tag, // category
        long, // value_type
        long, // difference_type
        const long*, // pointer type
        long> // reference type
        {
            long current_num = START;
            public:
                reference operator*() const { return current_num; }
                explicit iterable(long val = 0) : current_num(val) {}
                iterable& operator++() {
                    current_num = ( END >= START) ? current_num + 1 :
                        current_num - 1;
                return *this;
            }
            iterable operator++(int) {
                iterable retval = *this; ++(*this); return retval;
            }
            bool operator==(iterable other) const
                { return current_num == other.current_num; }
            bool operator!=(iterable other) const
                { return !(*this == other); }
    };
```

前面的代码实现了一个内部类，该类派生自`std::iterator`，以满足类型通过基于范围的 for 循环进行枚举的要求。现在我们将编写两个公共方法（`begin()`和`end()`），以便类的使用者可以使用基于范围的 for 循环：

```cpp
iterable begin() { return iterable(START); }
    iterable end() { return iterable(END >= START ? END + 1 :
        END - 1); }
};
```

现在，我们可以编写代码来使用前面的类：

```cpp
for (long l : EnumerableRange<5, 25>())
    { std::cout << l << ' '; }
```

在上一章中，我们定义了`IEnumerable<T>`接口。这个想法是遵循 Reactive eXtensions 的文档。可迭代类与上一章中的`IEnumerable<T>`实现非常相似。正如在上一章中概述的那样，如果我们稍微调整代码，前面的类可以变为推送型。让我们编写一个包含三个方法的`OBSERVER`类。我们将使用标准库提供的函数包装器来定义这些方法：

```cpp
struct OBSERVER {
    std::function<void(const long&)> ondata;
    std::function<void()> oncompleted;
    std::function<void(const std::exception &)> onexception;
};
```

这里给出的`ObservableRange`类包含一个存储订阅者列表的`vector<T>`。当生成新数字时，事件将通知所有订阅者。如果我们从异步方法中分派通知调用，消费者将与范围流的生产者解耦。我们还没有为以下类实现`IObserver/IObserver<T>`接口，但我们可以通过订阅方法订阅通知：

```cpp
template<long START, long END>
class ObservableRange {
    private:
        //---------- Container to store observers
        std::vector<
            std::pair<const OBSERVER&,int>> _observers;
        int _id = 0;
```

我们将以`std::pair`的形式将订阅者列表存储在`std::vector`中。`std::pair`中的第一个值是对`OBSERVER`的引用，`std::pair`中的第二个值是唯一标识订阅者的整数。消费者应该使用订阅方法返回的 ID 来取消订阅：

```cpp
//---- The following implementation of iterable does
//---- not allow to take address of the pointed value  &(*it)
//---- Eg- &(*iterable.begin()) will be ill-formed
//---- Code is just for demonstrate Obervable/Observer
class iterable : public std::iterator<
    std::input_iterator_tag, // category
    long, // value_type
    long, // difference_type
    const long*, // pointer type
    long> // reference type
    {
        long current_num = START;
    public:
        reference operator*() const { return current_num; }
        explicit iterable(long val = 0) : current_num(val) {}
        iterable& operator++() {
            current_num = ( END >= START) ? current_num + 1 :
                current_num - 1;
            return *this;
        }
        iterable operator++(int) {
            iterable retval = *this; ++(*this); return retval;
        }
        bool operator==(iterable other) const
            { return current_num == other.current_num; }
        bool operator!=(iterable other) const
            { return !(*this == other); }
        };
    iterable begin() { return iterable(START); }
    iterable end() { return iterable(END >= START ? END + 1 : END - 1); }
// generate values between the range
// This is a private method and will be invoked from the generate
// ideally speaking, we should invoke this method with std::asnyc
void generate_async()
{
    auto& subscribers = _observers;
    for( auto l : *this )
        for (const auto& obs : subscribers) {
            const OBSERVER& ob = obs.first;
            ob.ondata(l);
    }
}

//----- The public interface of the call include generate which triggers
//----- the generation of the sequence, subscribe/unsubscribe pair
public:
    //-------- the public interface to trigger generation
    //-------- of thevalues. The generate_async can be executed
    //--------- via std::async to return to the caller
    void generate() { generate_async(); }
    //---------- subscribe method. The clients which
    //----------- expects notification can register here
    int subscribe(const OBSERVER& call) {
        // https://en.cppreference.com/w/cpp/container/vector/emplace_back
        _observers.emplace_back(call, ++_id);
        return _id;
    }
    //------------ has just stubbed unsubscribe to keep
    //------------- the listing small
    void unsubscribe(const int subscription) {}

};

int main() {
    //------ Call the Range based enumerable
    for (long l : EnumerableRange<5, 25>())
        { std::cout << l << ' '; }
    std::cout << endl;
    // instantiate an instance of ObservableRange
    auto j = ObservableRange<10,20>();
    OBSERVER test_handler;
    test_handler.ondata = [=
    {cout << r << endl; };
    //---- subscribe to the notifiactions
    int cnt = j.subscribe(test_handler);
    j.generate(); //trigget events to generate notifications
    return 0;
}
```

# 摘要

在本章中，我们了解了 C++程序员在编写响应式程序或其他类型的程序时应该熟悉的编程语言特性。我们谈到了类型推断、可变模板、右值引用和移动语义、Lambda 函数、基本的函数式编程、可管道化的操作符以及迭代器和观察者的实现。在下一章中，我们将学习 C++编程语言提供的并发编程支持。
