# 第一章：C++17 功能

在本章中，您将学习以下概念：

+   C++17 背景

+   C++17 中有什么新功能？

+   C++17 中弃用或移除的功能是什么？

+   C++17 的关键特性

# C++17 背景

如您所知，C++语言是 Bjarne Stroustrup 的心血结晶，他于 1979 年开发了 C++。C++编程语言由国际标准化组织（ISO）标准化。

最初的标准化于 1998 年发布，通常称为 C++98，接下来的标准化 C++03 于 2003 年发布，这主要是一个修复错误的版本，只有一个语言特性用于值初始化。2011 年 8 月，C++11 标准发布，核心语言增加了一些内容，包括对**标准模板库**（**STL**）的一些重大有趣的更改；C++11 基本上取代了 C++03 标准。C++14 于 2014 年 12 月发布，带有一些新功能，后来，C++17 标准于 2017 年 7 月 31 日发布。

在撰写本书时，C++17 是 C++编程语言的 ISO/IEC 标准的最新修订版。

本章需要支持 C++17 功能的编译器：gcc 版本 7 或更高版本。由于 gcc 版本 7 是撰写本书时的最新版本，因此在本章中我将使用 gcc 版本 7.1.0。

如果您还没有安装支持 C++17 功能的 g++ 7，可以使用以下命令安装：

`sudo add-apt-repository ppa:jonathonf/gcc-7.1

sudo apt-get update

sudo apt-get install gcc-7 g++-7`

# C++17 中有什么新功能？

完整的 C++17 功能列表可以在[`en.cppreference.com/w/cpp/compiler_support#C.2B.2B17_features`](http://en.cppreference.com/w/cpp/compiler_support#C.2B.2B17_features)找到。

为了给出一个高层次的概念，以下是一些新的 C++17 功能：

+   直接列表初始化的新汽车规则

+   没有消息的`static_assert`

+   嵌套命名空间定义

+   内联变量

+   命名空间和枚举器的属性

+   C++异常规范是类型系统的一部分

+   改进的 lambda 功能，可在服务器上提供性能优势

+   NUMA 架构

+   使用属性命名空间

+   用于超对齐数据的动态内存分配

+   类模板的模板参数推导

+   具有自动类型的非类型模板参数

+   保证的拷贝省略

+   继承构造函数的新规范

+   枚举的直接列表初始化

+   更严格的表达式评估顺序

+   `shared_mutex`

+   字符串转换

否则，核心 C++语言中添加了许多有趣的新功能：STL、lambda 等。新功能为 C++带来了面貌更新，从`C++17`开始，作为 C++开发人员，您会感到自己正在使用现代编程语言，如 Java 或 C#。

# C++17 中弃用或移除的功能是什么？

以下功能现在已在 C++17 中移除：

+   `register`关键字在 C++11 中已被弃用，并在 C++17 中被移除

+   `++`运算符对`bool`在 C++98 中已被弃用，并在 C++17 中被移除

+   动态异常规范在 C++11 中已被弃用，并在 C++17 中被移除

# C++17 的关键特性

让我们逐个探讨以下 C++17 的关键功能：

+   更简单的嵌套命名空间

+   从大括号初始化列表中检测类型的新规则

+   简化的`static_assert`

+   `std::invoke`

+   结构化绑定

+   `if`和`switch`局部作用域变量

+   类模板的模板类型自动检测

+   内联变量

# 更简单的嵌套命名空间语法

直到 C++14 标准，C++中支持的嵌套命名空间的语法如下：

```cpp
#include <iostream>
using namespace std;

namespace org {
    namespace tektutor {
        namespace application {
             namespace internals {
                  int x;
             }
        }
    }
}

int main ( ) {
    org::tektutor::application::internals::x = 100;
    cout << "\nValue of x is " << org::tektutor::application::internals::x << endl;

    return 0;
}
```

上述代码可以使用以下命令编译，并且可以查看输出：

```cpp
g++-7 main.cpp -std=c++17
./a.out
```

上述程序的输出如下：

```cpp
Value of x is 100
```

每个命名空间级别都以大括号开始和结束，这使得在大型应用程序中使用嵌套命名空间变得困难。C++17 嵌套命名空间语法真的很酷；只需看看下面的代码，你就会很容易同意我的观点：

```cpp
#include <iostream>
using namespace std;

namespace org::tektutor::application::internals {
    int x;
}

int main ( ) {
    org::tektutor::application::internals::x = 100;
    cout << "\nValue of x is " << org::tektutor::application::internals::x << endl;

    return 0;
}
```

上述代码可以编译，并且可以使用以下命令查看输出：

```cpp
g++-7 main.cpp -std=c++17
./a.out
```

输出与上一个程序相同：

```cpp
Value of x is 100
```

# 来自大括号初始化列表的类型自动检测的新规则

C++17 引入了对初始化列表的自动检测的新规则，这补充了 C++14 的规则。C++17 规则坚持认为，如果声明了`std::initializer_list`的显式或部分特化，则程序是非法的：

```cpp
#include <iostream>
using namespace std;

template <typename T1, typename T2>
class MyClass {
     private:
          T1 t1;
          T2 t2;
     public:
          MyClass( T1 t1 = T1(), T2 t2 = T2() ) { }

          void printSizeOfDataTypes() {
               cout << "\nSize of t1 is " << sizeof ( t1 ) << " bytes." << endl;
               cout << "\nSize of t2 is " << sizeof ( t2 ) << " bytes." << endl;
     }
};

int main ( ) {

    //Until C++14
    MyClass<int, double> obj1;
    obj1.printSizeOfDataTypes( );

    //New syntax in C++17
    MyClass obj2( 1, 10.56 );

    return 0;
}
```

上述代码可以编译，并且可以使用以下命令查看输出：

```cpp
g++-7 main.cpp -std=c++17
./a.out
```

上述程序的输出如下：

```cpp
Values in integer vectors are ...
1 2 3 4 5 

Values in double vectors are ...
1.5 2.5 3.5 
```

# 简化的 static_assert

`static_assert`宏有助于在编译时识别断言失败。这个特性自 C++11 以来就得到了支持；然而，在 C++17 中，`static_assert`宏在之前是需要一个强制的断言失败消息的，现在已经变成了可选的。

以下示例演示了使用`static_assert`的方法，包括消息和不包括消息：

```cpp
#include <iostream>
#include <type_traits>
using namespace std;

int main ( ) {

        const int x = 5, y = 5;

        static_assert ( 1 == 0, "Assertion failed" );
        static_assert ( 1 == 0 );
        static_assert ( x == y );

        return 0;
}
```

上述程序的输出如下：

```cpp
g++-7 staticassert.cpp -std=c++17
staticassert.cpp: In function ‘int main()’:
staticassert.cpp:7:2: error: static assertion failed: Assertion failed
 static_assert ( 1 == 0, "Assertion failed" );

staticassert.cpp:8:2: error: static assertion failed
 static_assert ( 1 == 0 );
```

从上面的输出中，您可以看到消息`Assertion failed`作为编译错误的一部分出现，而在第二次编译中，由于我们没有提供断言失败消息，出现了默认的编译器错误消息。当没有断言失败时，断言错误消息不会出现，如`static_assert(x==y)`所示。这个特性受到了 BOOST C++库中 C++社区的启发。

# `std::invoke()`方法

`std::invoke()`方法可以用相同的语法调用函数、函数指针和成员指针：

```cpp
#include <iostream>
#include <functional>
using namespace std;

void globalFunction( ) {
     cout << "globalFunction ..." << endl;
}

class MyClass {
    public:
        void memberFunction ( int data ) {
             std::cout << "\nMyClass memberFunction ..." << std::endl;
        }

        static void staticFunction ( int data ) {
             std::cout << "MyClass staticFunction ..." << std::endl;
        }
};

int main ( ) {

    MyClass obj;

    std::invoke ( &MyClass::memberFunction, obj, 100 );
    std::invoke ( &MyClass::staticFunction, 200 );
    std::invoke ( globalFunction );

    return 0;
}
```

上述代码可以编译，并且可以使用以下命令查看输出：

```cpp
g++-7 main.cpp -std=c++17
./a.out
```

上述程序的输出如下：

```cpp
MyClass memberFunction ...
MyClass staticFunction ...
globalFunction ...
```

`std::invoke()`方法是一个模板函数，可以帮助您无缝地调用可调用对象，无论是内置的还是用户定义的。

# 结构化绑定

现在您可以使用一个非常酷的语法初始化多个变量，并返回一个值，如下面的示例代码所示：

```cpp
#include <iostream>
#include <tuple>
using namespace std;

int main ( ) {

    tuple<string,int> student("Sriram", 10);
    auto [name, age] = student;

    cout << "\nName of the student is " << name << endl;
    cout << "Age of the student is " << age << endl;

    return 0;
}
```

在上述程序中，用**粗体**突出显示的代码是 C++17 引入的结构化绑定特性。有趣的是，我们没有声明`string name`和`int age`变量。这些都是由 C++编译器自动推断为`string`和`int`，这使得 C++的语法就像任何现代编程语言一样，而不会失去其性能和系统编程的好处。

上述代码可以编译，并且可以使用以下命令查看输出：

```cpp
g++-7 main.cpp -std=c++17
./a.out
```

上述程序的输出如下：

```cpp
Name of the student is Sriram
Age of the student is 10
```

# If 和 Switch 局部作用域变量

有一个有趣的新功能，允许您声明一个绑定到`if`和`switch`语句代码块的局部变量。在`if`和`switch`语句中使用的变量的作用域将在各自的块之外失效。可以通过以下易于理解的示例更好地理解，如下所示：

```cpp
#include <iostream>
using namespace std;

bool isGoodToProceed( ) {
    return true;
}

bool isGood( ) {
     return true;
}

void functionWithSwitchStatement( ) {

     switch ( auto status = isGood( ) ) {
          case true:
                 cout << "\nAll good!" << endl;
          break;

          case false:
                 cout << "\nSomething gone bad" << endl;
          break;
     } 

}

int main ( ) {

    if ( auto flag = isGoodToProceed( ) ) {
         cout << "flag is a local variable and it loses its scope outside the if block" << endl;
    }

     functionWithSwitchStatement();

     return 0;
}
```

上述代码可以编译，并且可以使用以下命令查看输出：

```cpp
g++-7 main.cpp -std=c++17
./a.out
```

上述程序的输出如下：

```cpp
flag is a local variable and it loses its scope outside the if block
All good!
```

# 类模板的模板类型自动推断

我相信你会喜欢你即将在示例代码中看到的内容。虽然模板非常有用，但很多人不喜欢它，因为它的语法很难和奇怪。但是你不用担心了；看看下面的代码片段：

```cpp
#include <iostream>
using namespace std;

template <typename T1, typename T2>
class MyClass {
     private:
          T1 t1;
          T2 t2;
     public:
          MyClass( T1 t1 = T1(), T2 t2 = T2() ) { }

          void printSizeOfDataTypes() {
               cout << "\nSize of t1 is " << sizeof ( t1 ) << " bytes." << endl;
               cout << "\nSize of t2 is " << sizeof ( t2 ) << " bytes." << endl;
     }
};

int main ( ) {

    //Until C++14
    MyClass<int, double> obj1;
    obj1.printSizeOfDataTypes( );

    //New syntax in C++17
    MyClass obj2( 1, 10.56 );

    return 0;
}
```

上述代码可以编译，并且可以使用以下命令查看输出：

```cpp
g++-7 main.cpp -std=c++17
./a.out
```

程序的输出如下：

```cpp
Size of t1 is 4 bytes.
Size of t2 is 8 bytes.
```

# 内联变量

就像 C++中的内联函数一样，现在您可以使用内联变量定义。这对初始化静态变量非常方便，如下面的示例代码所示：

```cpp
#include <iostream>
using namespace std;

class MyClass {
    private:
        static inline int count = 0;
    public:
        MyClass() { 
              ++count;
        }

    public:
         void printCount( ) {
              cout << "\nCount value is " << count << endl;
         } 
};

int main ( ) {

    MyClass obj;

    obj.printCount( ) ;

    return 0;
}
```

上述代码可以编译，并且可以使用以下命令查看输出：

```cpp
g++-7 main.cpp -std=c++17
./a.out
```

上述代码的输出如下：

```cpp
Count value is 1
```

# 总结

在本章中，您了解了 C++17 引入的有趣的新特性。您学会了超级简单的 C++17 嵌套命名空间语法。您还学会了使用大括号初始化列表进行数据类型检测以及 C++17 标准中引入的新规则。

您还注意到，`static_assert`可以在没有断言失败消息的情况下完成。此外，使用`std::invoke()`，您现在可以调用全局函数、函数指针、成员函数和静态类成员函数。并且，使用结构化绑定，您现在可以用返回值初始化多个变量。

您还学到了`if`和`switch`语句可以在`if`条件和`switch`语句之前有一个局部作用域的变量。您了解了类模板的自动类型检测。最后，您使用了`inline`变量。

C++17 有许多更多的特性，但本章试图涵盖大多数开发人员可能需要的最有用的特性。在下一章中，您将学习标准模板库。
