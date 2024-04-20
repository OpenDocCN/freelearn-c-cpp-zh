# 第三章：模板编程

在本章中，我们将涵盖以下主题：

+   泛型编程

+   函数模板

+   类模板

+   重载函数模板

+   泛型类

+   显式类特化

+   部分特化

现在让我们开始学习泛型编程。

# 泛型编程

泛型编程是一种编程风格，可以帮助您开发可重用的代码或通用算法，可以应用于各种数据类型。每当调用通用算法时，数据类型将以特殊的语法作为参数提供。

假设我们想要编写一个`sort()`函数，它接受一个需要按升序排序的输入数组。其次，我们需要`sort()`函数来对`int`、`double`、`char`和`string`数据类型进行排序。有几种方法可以解决这个问题：

+   我们可以为每种数据类型编写四个不同的`sort()`函数

+   我们也可以编写一个单一的宏函数

好吧，这两种方法都有各自的优点和缺点。第一种方法的优点是，由于`int`、`double`、`char`和`string`数据类型都有专门的函数，如果提供了不正确的数据类型，编译器将能够执行类型检查。第一种方法的缺点是，尽管所有函数的逻辑都相同，但我们必须编写四个不同的函数。如果在算法中发现了错误，必须分别在所有四个函数中进行修复；因此，需要进行大量的维护工作。如果我们需要支持另一种数据类型，我们将不得不编写另一个函数，随着需要支持更多的数据类型，这种情况将不断增加。

第二种方法的优点是，我们可以为所有数据类型编写一个宏。然而，一个非常令人沮丧的缺点是，编译器将无法执行类型检查，这种方法更容易出现错误，并可能引发许多意外的麻烦。这种方法与面向对象的编码原则背道而驰。

C++通过模板支持泛型编程，具有以下优点：

+   我们只需要使用模板编写一个函数

+   模板支持静态多态

+   模板提供了前面两种方法的所有优点，没有任何缺点

+   泛型编程实现了代码重用

+   生成的代码是面向对象的

+   C++编译器可以在编译时执行类型检查

+   易于维护

+   支持各种内置和用户定义的数据类型

然而，缺点如下：

+   并不是所有的 C++程序员都感到舒适编写基于模板的代码，但这只是一个初始的阻碍

+   在某些情况下，模板可能会使代码膨胀并增加二进制占用空间，导致性能问题

# 函数模板

函数模板允许您对数据类型进行参数化。之所以称之为泛型编程，是因为单个模板函数将支持许多内置和用户定义的数据类型。模板化函数的工作原理类似于**C 风格的宏**，只是 C++编译器在调用模板函数时会对函数进行类型检查，以确保我们在调用模板函数时提供的数据类型是兼容的。

通过一个简单的例子来更容易理解模板的概念，如下所示：

```cpp
#include <iostream>
#include <algorithm>
#include <iterator>
using namespace std;

template <typename T, int size>
void sort ( T input[] ) {

     for ( int i=0; i<size; ++i) { 
         for (int j=0; j<size; ++j) {
              if ( input[i] < input[j] )
                  swap (input[i], input[j] );
         }
     }

}

int main () {
        int a[10] = { 100, 10, 40, 20, 60, 80, 5, 50, 30, 25 };

        cout << "nValues in the int array before sorting ..." << endl;
        copy ( a, a+10, ostream_iterator<int>( cout, "t" ) );
        cout << endl;

        ::sort<int, 10>( a );

        cout << "nValues in the int array after sorting ..." << endl;
        copy ( a, a+10, ostream_iterator<int>( cout, "t" ) );
        cout << endl;

        double b[5] = { 85.6d, 76.13d, 0.012d, 1.57d, 2.56d };

        cout << "nValues in the double array before sorting ..." << endl;
        copy ( b, b+5, ostream_iterator<double>( cout, "t" ) );
        cout << endl;

        ::sort<double, 5>( b );

        cout << "nValues in the double array after sorting ..." << endl;
        copy ( b, b+5, ostream_iterator<double>( cout, "t" ) );
        cout << endl;

        string names[6] = {
               "Rishi Kumar Sahay",
               "Arun KR",
               "Arun CR",
               "Ninad",
               "Pankaj",
               "Nikita"
        };

        cout << "nNames before sorting ..." << endl;
        copy ( names, names+6, ostream_iterator<string>( cout, "n" ) );
        cout << endl;

        ::sort<string, 6>( names );

        cout << "nNames after sorting ..." << endl;
        copy ( names, names+6, ostream_iterator<string>( cout, "n" ) );
        cout << endl;

        return 0;
}

```

运行以下命令：

```cpp
g++ main.cpp -std=c++17
./a.out
```

上述程序的输出如下：

```cpp
Values in the int array before sorting ...
100  10   40   20   60   80   5   50   30   25

Values in the int array after sorting ...
5    10   20   25   30   40   50   60   80   100

Values in the double array before sorting ...
85.6d 76.13d 0.012d 1.57d 2.56d

Values in the double array after sorting ...
0.012   1.57   2.56   76.13   85.6

Names before sorting ...
Rishi Kumar Sahay
Arun KR
Arun CR
Ninad
Pankaj
Nikita

Names after sorting ...
Arun CR
Arun KR
Nikita
Ninad
Pankaj
Rich Kumar Sahay
```

看到一个模板函数就能完成所有的魔术，是不是很有趣？是的，这就是 C++模板的酷之处！

你是否好奇看到模板实例化的汇编输出？使用命令**`g++ -S main.cpp`**。

# 代码演示

以下代码定义了一个函数模板。关键字`template <typename T, int size>`告诉编译器接下来是一个函数模板：

```cpp
template <typename T, int size>
void sort ( T input[] ) {

 for ( int i=0; i<size; ++i) { 
     for (int j=0; j<size; ++j) {
         if ( input[i] < input[j] )
             swap (input[i], input[j] );
     }
 }

}
```

`void sort ( T input[] )`这一行定义了一个名为`sort`的函数，返回`void`，接收类型为`T`的输入数组。`T`类型不表示任何特定的数据类型。`T`将在编译时实例化函数模板时推导出来。

以下代码用一些未排序的值填充一个整数数组，并将其打印到终端上：

```cpp
 int a[10] = { 100, 10, 40, 20, 60, 80, 5, 50, 30, 25 };
 cout << "nValues in the int array before sorting ..." << endl;
 copy ( a, a+10, ostream_iterator<int>( cout, "t" ) );
 cout << endl;
```

以下行将实例化一个`int`数据类型的函数模板实例。此时，`typename T`被替换，为`int`创建了一个专门的函数。在`sort`前面的作用域解析运算符，即`::sort()`，确保它调用我们在全局命名空间中定义的自定义函数`sort()`；否则，C++编译器将尝试调用`std 命名空间`中定义的`sort()`算法，或者如果存在这样的函数，则从任何其他命名空间中调用。`<int, 10>`变量告诉编译器创建一个函数实例，用`int`替换`typename T`，`10`表示模板函数中使用的数组的大小：

```cpp
::sort<int, 10>( a );
```

以下行将实例化另外两个支持`5`个元素的`double`数组和`6`个元素的`string`数组的实例：

```cpp
::sort<double, 5>( b );
::sort<string, 6>( names );
```

如果您想了解有关 C++编译器如何实例化函数模板以支持`int`、`double`和`string`的更多细节，可以尝试使用 Unix 实用程序`nm`和`c++filt`。`nm` Unix 实用程序将列出符号表中的符号，如下所示：

```cpp
nm ./a.out | grep sort

00000000000017f1 W _Z4sortIdLi5EEvPT_
0000000000001651 W _Z4sortIiLi10EEvPT_
000000000000199b W _Z4sortINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELi6EEvPT_
```

正如您所看到的，二进制文件中有三个不同的重载`sort`函数；然而，我们只定义了一个模板函数。由于 C++编译器对函数重载进行了名称混淆，我们很难解释这三个函数中的哪一个是为`int`、`double`和`string`数据类型设计的。

然而，有一个线索：第一个函数是为`double`设计的，第二个是为`int`设计的，第三个是为`string`设计的。对于`double`，名称混淆的函数为`_Z4sortIdLi5EEvPT_`，对于`int`，为`_Z4sortIiLi10EEvPT_`，对于`string`，为`_Z4sortINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELi6EEvPT_`。还有一个很酷的 Unix 实用程序可以帮助您轻松解释函数签名。检查`c++filt`实用程序的以下输出：

```cpp
c++filt _Z4sortIdLi5EEvPT_
void sort<double, 5>(double*)

c++filt _Z4sortIiLi10EEvPT_
void sort<int, 10>(int*)

c++filt _Z4sortINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELi6EEvPT_
void sort<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, 6>(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >*)
```

希望您在使用 C++模板时会发现这些实用程序有用。我相信这些工具和技术将帮助您调试任何 C++应用程序。

# 重载函数模板

函数模板的重载与 C++中的常规函数重载完全相同。但是，我将帮助您回顾 C++函数重载的基础知识。

C++编译器对函数重载的规则和期望如下：

+   重载的函数名称将是相同的。

+   C++编译器将无法区分仅通过返回值不同的重载函数。

+   重载函数参数的数量、数据类型或它们的顺序应该不同。除了其他规则外，当前项目符号中描述的这些规则中至少应满足一个，但更多的符合也不会有坏处。

+   重载的函数必须在同一个命名空间或同一个类范围内。

如果上述任何规则没有得到满足，C++编译器将不会将它们视为重载函数。如果在区分重载函数时存在任何歧义，C++编译器将立即报告它为编译错误。

现在是时候通过以下程序示例来探索一下了：

```cpp
#include <iostream>
#include <array>
using namespace std;

void sort ( array<int,6> data ) {

     cout << "Non-template sort function invoked ..." << endl;

     int size = data.size();

     for ( int i=0; i<size; ++i ) { 
          for ( int j=0; j<size; ++j ) {
                if ( data[i] < data[j] )
                    swap ( data[i], data[j] );
          }
     }

}

template <typename T, int size>
void sort ( array<T, size> data ) {

     cout << "Template sort function invoked with one argument..." << endl;

     for ( int i=0; i<size; ++i ) {
         for ( int j=0; j<size; ++j ) {
             if ( data[i] < data[j] )
                swap ( data[i], data[j] );
         }
     }

}

template <typename T>
void sort ( T data[], int size ) {
     cout << "Template sort function invoked with two arguments..." << endl;

     for ( int i=0; i<size; ++i ) {
         for ( int j=0; j<size; ++j ) {
             if ( data[i] < data[j] )
                swap ( data[i], data[j] );
         }
     }

}

int main() {

    //Will invoke the non-template sort function
    array<int, 6> a = { 10, 50, 40, 30, 60, 20 };
    ::sort ( a );

    //Will invoke the template function that takes a single argument
    array<float,6> b = { 10.6f, 57.9f, 80.7f, 35.1f, 69.3f, 20.0f };
    ::sort<float,6>( b );

    //Will invoke the template function that takes a single argument
    array<double,6> c = { 10.6d, 57.9d, 80.7d, 35.1d, 69.3d, 20.0d };
    ::sort<double,6> ( c );

    //Will invoke the template function that takes two arguments
    double d[] = { 10.5d, 12.1d, 5.56d, 1.31d, 81.5d, 12.86d };
    ::sort<double> ( d, 6 );

    return 0;

}
```

运行以下命令：

```cpp
g++ main.cpp -std=c++17

./a.out
```

上述程序的输出如下：

```cpp
Non-template sort function invoked ...

Template sort function invoked with one argument...

Template sort function invoked with one argument...

Template sort function invoked with two arguments...
```

# 代码演示

以下代码是我们自定义`sort()`函数的非模板版本：

```cpp
void sort ( array<int,6> data ) { 

     cout << "Non-template sort function invoked ..." << endl;

     int size = data.size();

     for ( int i=0; i<size; ++i ) { 
         for ( int j=0; j<size; ++j ) {
             if ( data[i] < data[j] )
                 swap ( data[i], data[j] );
         }
     }

}
```

非模板函数和模板函数可以共存并参与函数重载。上述函数的一个奇怪行为是数组的大小是硬编码的。

我们的`sort()`函数的第二个版本是一个模板函数，如下面的代码片段所示。有趣的是，我们在第一个非模板`sort()`版本中注意到的奇怪问题在这里得到了解决：

```cpp
template <typename T, int size>
void sort ( array<T, size> data ) {

     cout << "Template sort function invoked with one argument..." << endl;

     for ( int i=0; i<size; ++i ) {
         for ( int j=0; j<size; ++j ) {
             if ( data[i] < data[j] )
                swap ( data[i], data[j] );
         }
     }

}
```

在上述代码中，数据类型和数组的大小都作为模板参数传递，然后传递给函数调用参数。这种方法使函数通用，因为这个函数可以为任何数据类型实例化。

我们自定义的`sort()`函数的第三个版本也是一个模板函数，如下面的代码片段所示：

```cpp
template <typename T>
void sort ( T data[], int size ) {

     cout << "Template sort function invoked with two argument..." << endl;

     for ( int i=0; i<size; ++i ) {
         for ( int j=0; j<size; ++j ) {
             if ( data[i] < data[j] )
                swap ( data[i], data[j] );
         }
     }

}
```

上述模板函数接受 C 风格数组；因此，它也期望用户指示其大小。然而，数组的大小可以在函数内计算，但出于演示目的，我需要一个接受两个参数的函数。前一个函数不推荐使用，因为它使用了 C 风格数组；理想情况下，我们会使用 STL 容器之一。

现在，让我们理解主函数代码。以下代码声明并初始化了 STL 数组容器，其中包含六个值，然后将其传递给我们在默认命名空间中定义的`sort()`函数：

```cpp
 //Will invoke the non-template sort function
 array<int, 6> a = { 10, 50, 40, 30, 60, 20 };
 ::sort ( a );
```

上述代码将调用非模板`sort()`函数。需要注意的重要一点是，每当 C++遇到函数调用时，它首先寻找非模板版本；如果 C++找到匹配的非模板函数版本，它的搜索正确函数定义就在那里结束。如果 C++编译器无法识别与函数调用签名匹配的非模板函数定义，那么它开始寻找任何可以支持函数调用的模板函数，并为所需的数据类型实例化一个专门的函数。

让我们理解以下代码：

```cpp
//Will invoke the template function that takes a single argument
array<float,6> b = { 10.6f, 57.9f, 80.7f, 35.1f, 69.3f, 20.0f };
::sort<float,6>( b );
```

这将调用接收单个参数的模板函数。由于没有接收`array<float,6>`数据类型的非模板`sort()`函数，C++编译器将从我们定义的接收单个参数的`sort()`模板函数中实例化这样的函数。

同样，以下代码触发编译器实例化接收`array<double, 6>`的`double`版本的模板`sort()`函数：

```cpp
  //Will invoke the template function that takes a single argument
 array<double,6> c = { 10.6d, 57.9d, 80.7d, 35.1d, 69.3d, 20.0d };
 ::sort<double,6> ( c );
```

最后，以下代码将实例化一个接收两个参数并调用函数的模板`sort()`的实例：

```cpp
 //Will invoke the template function that takes two arguments
 double d[] = { 10.5d, 12.1d, 5.56d, 1.31d, 81.5d, 12.86d };
 ::sort<double> ( d, 6 );
```

如果您已经走到这一步，我相信您会喜欢迄今为止讨论的 C++模板主题。

# 类模板

C++模板将函数模板概念扩展到类，使我们能够编写面向对象的通用代码。在前面的部分中，您学习了函数模板和重载的用法。在本节中，您将学习编写模板类，这将开启更有趣的通用编程概念。

`class`模板允许您通过模板类型表达式在类级别上对数据类型进行参数化。

让我们通过以下示例理解一个`class`模板：

```cpp
//myalgorithm.h
#include <iostream>
#include <algorithm>
#include <array>
#include <iterator>
using namespace std;

template <typename T, int size>
class MyAlgorithm {

public:
        MyAlgorithm() { } 
        ~MyAlgorithm() { }

        void sort( array<T, size> &data ) {
             for ( int i=0; i<size; ++i ) {
                 for ( int j=0; j<size; ++j ) {
                     if ( data[i] < data[j] )
                         swap ( data[i], data[j] );
                 }
             }
        }

        void sort ( T data[size] );

};

template <typename T, int size>
inline void MyAlgorithm<T, size>::sort ( T data[size] ) {
       for ( int i=0; i<size; ++i ) {
           for ( int j=0; j<size; ++j ) {
               if ( data[i] < data[j] )
                  swap ( data[i], data[j] );
           }
       }
}
```

C++模板函数重载是静态或编译时多态的一种形式。

让我们在以下`main.cpp`程序中使用`myalgorithm.h`如下：

```cpp
#include "myalgorithm.h"

int main() {

    MyAlgorithm<int, 10> algorithm1;

    array<int, 10> a = { 10, 5, 15, 20, 25, 18, 1, 100, 90, 18 };

    cout << "nArray values before sorting ..." << endl;
    copy ( a.begin(), a.end(), ostream_iterator<int>(cout, "t") );
    cout << endl;

    algorithm1.sort ( a );

    cout << "nArray values after sorting ..." << endl;
    copy ( a.begin(), a.end(), ostream_iterator<int>(cout, "t") );
    cout << endl;

    MyAlgorithm<int, 10> algorithm2;
    double d[] = { 100.0, 20.5, 200.5, 300.8, 186.78, 1.1 };

    cout << "nArray values before sorting ..." << endl;
    copy ( d.begin(), d.end(), ostream_iterator<double>(cout, "t") );
    cout << endl;

    algorithm2.sort ( d );

    cout << "nArray values after sorting ..." << endl;
    copy ( d.begin(), d.end(), ostream_iterator<double>(cout, "t") );
    cout << endl;

    return 0;  

}
```

让我们使用以下命令快速编译程序：

```cpp
g++ main.cpp -std=c++17

./a.out
```

输出如下：

```cpp

Array values before sorting ...
10  5   15   20   25   18   1   100   90   18

Array values after sorting ...
1   5   10   15   18   18   20   25   90   100

Array values before sorting ...
100   20.5   200.5   300.8   186.78   1.1

Array values after sorting ...
1.1     20.5   100   186.78  200.5  300.8
```

# 代码演示

以下代码声明了一个类模板。关键字`template <typename T, int size>`可以替换为`<class T, int size>`。这两个关键字可以在函数和类模板中互换使用；然而，作为行业最佳实践，`template<class T>`只能用于类模板，以避免混淆：

```cpp
template <typename T, int size>
class MyAlgorithm 
```

重载的`sort()`方法之一内联定义如下：

```cpp
 void sort( array<T, size> &data ) {
      for ( int i=0; i<size; ++i ) {
          for ( int j=0; j<size; ++j ) {
              if ( data[i] < data[j] )
                 swap ( data[i], data[j] );
          }
      }
 } 
```

第二个重载的`sort()`函数只是在类范围内声明，没有任何定义，如下所示：

```cpp
template <typename T, int size>
class MyAlgorithm {
      public:
           void sort ( T data[size] );
};
```

前面的`sort()`函数是在类范围之外定义的，如下面的代码片段所示。奇怪的是，我们需要为在类模板之外定义的每个成员函数重复模板参数：

```cpp
template <typename T, int size>
inline void MyAlgorithm<T, size>::sort ( T data[size] ) {
       for ( int i=0; i<size; ++i ) {
           for ( int j=0; j<size; ++j ) {
               if ( data[i] < data[j] )
                  swap ( data[i], data[j] );
           }
       }
}
```

否则，类模板的概念与函数模板的概念相同。

您想看看模板的编译器实例化代码吗？使用**`g++ -fdump-tree-original main.cpp -std=c++17`**命令。

# 显式类特化

到目前为止，在本章中，您已经学会了如何使用函数模板和类模板进行通用编程。当您理解类模板时，单个模板类可以支持任何内置和用户定义的数据类型。然而，有时我们需要对某些数据类型进行特殊处理，以便与其他数据类型有所区别。在这种情况下，C++为我们提供了显式类特化支持，以处理具有差异处理的选择性数据类型。

考虑 STL `deque`容器；虽然`deque`看起来适合存储，比如说，`string`、`int`、`double`和`long`，但如果我们决定使用`deque`来存储一堆`boolean`类型，`bool`数据类型至少占用一个字节，而根据编译器供应商的实现可能会有所不同。虽然一个位可以有效地表示真或假，但布尔值至少占用一个字节，即 8 位，剩下的 7 位没有被使用。这可能看起来没问题；但是，如果您必须存储一个非常大的`deque`布尔值，这绝对不是一个有效的想法，对吧？您可能会想，有什么大不了的？我们可以为`bool`编写另一个专门的类或模板类。但这种方法要求最终用户明确为不同的数据类型使用不同的类，这也不是一个好的设计，对吧？这正是 C++的显式类特化派上用场的地方。

显式模板特化也被称为完全模板特化。

如果您还不信服，没关系；下面的例子将帮助您理解显式类特化的必要性以及显式类特化的工作原理。

让我们开发一个`DynamicArray`类来支持任何数据类型的动态数组。让我们从一个类模板开始，如下面的程序所示：

```cpp
#include <iostream>
#include <deque>
#include <algorithm>
#include <iterator>
using namespace std;

template < class T >
class DynamicArray {
      private:
           deque< T > dynamicArray;
           typename deque< T >::iterator pos;

      public:
           DynamicArray() { initialize(); }
           ~DynamicArray() { }

           void initialize() {
                 pos = dynamicArray.begin();
           }

           void appendValue( T element ) {
                 dynamicArray.push_back ( element );
           }

           bool hasNextValue() { 
                 return ( pos != dynamicArray.end() );
           }

           T getValue() {
                 return *pos++;
           }

};
```

前面的`DynamicArray`模板类在内部使用了 STL `deque`类。因此，您可以将`DynamicArray`模板类视为自定义适配器容器。让我们探索如何在`main.cpp`中使用`DynamicArray`模板类，以下是代码片段：

```cpp
#include "dynamicarray.h"
#include "dynamicarrayforbool.h"

int main () {

    DynamicArray<int> intArray;

    intArray.appendValue( 100 );
    intArray.appendValue( 200 );
    intArray.appendValue( 300 );
    intArray.appendValue( 400 );

    intArray.initialize();

    cout << "nInt DynamicArray values are ..." << endl;
    while ( intArray.hasNextValue() )
          cout << intArray.getValue() << "t";
    cout << endl;

    DynamicArray<char> charArray;
    charArray.appendValue( 'H' );
    charArray.appendValue( 'e' );
    charArray.appendValue( 'l' );
    charArray.appendValue( 'l' );
    charArray.appendValue( 'o' );

    charArray.initialize();

    cout << "nChar DynamicArray values are ..." << endl;
    while ( charArray.hasNextValue() )
          cout << charArray.getValue() << "t";
    cout << endl;

    DynamicArray<bool> boolArray;

    boolArray.appendValue ( true );
    boolArray.appendValue ( false );
    boolArray.appendValue ( true );
    boolArray.appendValue ( false );

    boolArray.initialize();

    cout << "nBool DynamicArray values are ..." << endl;
    while ( boolArray.hasNextValue() )
         cout << boolArray.getValue() << "t";
    cout << endl;

    return 0;

}
```

让我们快速使用以下命令编译程序：

```cpp
g++ main.cpp -std=c++17

./a.out
```

输出如下：

```cpp
Int DynamicArray values are ...
100   200   300   400

Char DynamicArray values are ...
H   e   l   l   o

Bool DynamicArray values are ...
1   0   1   0
```

太好了！我们自定义的适配器容器似乎工作正常。

# 代码演示

让我们放大并尝试理解前面的程序是如何工作的。以下代码告诉 C++编译器接下来是一个类模板：

```cpp
template < class T >
class DynamicArray {
      private:
           deque< T > dynamicArray;
           typename deque< T >::iterator pos;
```

正如您所看到的，`DynamicArray`类在内部使用了 STL `deque`，并且为`deque`声明了名为`pos`的迭代器。这个迭代器`pos`被`Dynamic`模板类用于提供高级方法，比如`initialize()`、`appendValue()`、`hasNextValue()`和`getValue()`方法。

`initialize()`方法将`deque`迭代器`pos`初始化为`deque`中存储的第一个数据元素。`appendValue( T element )`方法允许您在`deque`的末尾添加数据元素。`hasNextValue()`方法告诉`DynamicArray`类是否有更多的数据值存储--`true`表示有更多的值，`false`表示`DynamicArray`导航已经到达`deque`的末尾。当需要时，`initialize()`方法可以用来重置`pos`迭代器到起始点。`getValue()`方法返回`pos`迭代器在那一刻指向的数据元素。`getValue()`方法不执行任何验证；因此，在调用`getValue()`之前，必须与`hasNextValue()`结合使用，以安全地访问存储在`DynamicArray`中的值。

现在，让我们理解`main()`函数。以下代码声明了一个存储`int`数据类型的`DynamicArray`类；`DynamicArray<int> intArray`将触发 C++编译器实例化一个专门针对`int`数据类型的`DynamicArray`类：

```cpp
DynamicArray<int> intArray;

intArray.appendValue( 100 );
intArray.appendValue( 200 );
intArray.appendValue( 300 );
intArray.appendValue( 400 );
```

值`100`、`200`、`300`和`400`依次存储在`DynamicArray`类中。以下代码确保`intArray`迭代器指向第一个元素。一旦迭代器初始化，存储在`DynamicArray`类中的值将通过`getValue()`方法打印出来，而`hasNextValue()`确保导航没有到达`DynamicArray`类的末尾：

```cpp
intArray.initialize();
cout << "nInt DynamicArray values are ..." << endl;
while ( intArray.hasNextValue() )
      cout << intArray.getValue() << "t";
cout << endl;
```

在主函数中，创建了一个`char DynamicArray`类，填充了一些数据，并进行了打印。让我们跳过`char` `DynamicArray`，直接转到存储`bool`的`DynamicArray`类。

```cpp
DynamicArray<bool> boolArray;

boolArray.appendValue ( "1010" );

boolArray.initialize();

cout << "nBool DynamicArray values are ..." << endl;

while ( boolArray.hasNextValue() )
      cout << boolArray.getValue() << "t";
cout << endl;
```

从前面的代码片段中，我们可以看到一切都很正常，对吗？是的，前面的代码完全正常；然而，`DynamicArray`的设计方法存在性能问题。虽然`true`可以用`1`表示，`false`可以用`0`表示，只需要 1 位，但前面的`DynamicArray`类却使用了 8 位来表示`1`和 8 位来表示`0`，我们必须解决这个问题，而不强迫最终用户选择一个对`bool`有效率的不同`DynamicArray`类。

让我们通过使用显式类模板特化来解决这个问题，以下是代码：

```cpp
#include <iostream>
#include <bitset>
#include <algorithm>
#include <iterator>
using namespace std;

template <>
class DynamicArray<bool> {
      private:
          deque< bitset<8> *> dynamicArray;
          bitset<8> oneByte;
          typename deque<bitset<8> * >::iterator pos;
          int bitSetIndex;

          int getDequeIndex () {
              return (bitSetIndex) ? (bitSetIndex/8) : 0;
          }
      public:
          DynamicArray() {
              bitSetIndex = 0;
              initialize();
          }

         ~DynamicArray() { }

         void initialize() {
              pos = dynamicArray.begin();
              bitSetIndex = 0;
         }

         void appendValue( bool value) {
              int dequeIndex = getDequeIndex();
              bitset<8> *pBit = NULL;

              if ( ( dynamicArray.size() == 0 ) || ( dequeIndex >= ( dynamicArray.size()) ) ) {
                   pBit = new bitset<8>();
                   pBit->reset();
                   dynamicArray.push_back ( pBit );
              }

              if ( !dynamicArray.empty() )
                   pBit = dynamicArray.at( dequeIndex );

              pBit->set( bitSetIndex % 8, value );
              ++bitSetIndex;
         }

         bool hasNextValue() {
              return (bitSetIndex < (( dynamicArray.size() * 8 ) ));
         }

         bool getValue() {
              int dequeIndex = getDequeIndex();

              bitset<8> *pBit = dynamicArray.at(dequeIndex);
              int index = bitSetIndex % 8;
              ++bitSetIndex;

              return (*pBit)[index] ? true : false;
         }
};
```

你注意到模板类声明了吗？模板类特化的语法是`template <> class DynamicArray<bool> { };`。`class`模板表达式是空的`<>`，对于所有数据类型都适用的`class`模板的名称和适用于`bool`数据类型的类的名称与模板表达式`<bool>`保持一致。

如果你仔细观察，你会发现，专门为`bool`设计的`DynamicArray`类内部使用了`deque<bitset<8>>`，即 8 位的`bitset`的`deque`，在需要时，`deque`会自动分配更多的`bitset<8>`位。`bitset`变量是一个内存高效的 STL 容器，只消耗 1 位来表示`true`或`false`。

让我们来看一下`main`函数：

```cpp
#include "dynamicarray.h"
#include "dynamicarrayforbool.h"

int main () {

    DynamicArray<int> intArray;

    intArray.appendValue( 100 );
    intArray.appendValue( 200 );
    intArray.appendValue( 300 );
    intArray.appendValue( 400 );

    intArray.initialize();

    cout << "nInt DynamicArray values are ..." << endl;

    while ( intArray.hasNextValue() )
          cout << intArray.getValue() << "t";
    cout << endl;

    DynamicArray<char> charArray;

    charArray.appendValue( 'H' );
    charArray.appendValue( 'e' );
    charArray.appendValue( 'l' );
    charArray.appendValue( 'l' );
    charArray.appendValue( 'o' );

    charArray.initialize();

    cout << "nChar DynamicArray values are ..." << endl;
    while ( charArray.hasNextValue() )
          cout << charArray.getValue() << "t";
    cout << endl;

    DynamicArray<bool> boolArray;

    boolArray.appendValue ( true );
    boolArray.appendValue ( false );
    boolArray.appendValue ( true );
    boolArray.appendValue ( false );

    boolArray.appendValue ( true );
    boolArray.appendValue ( false );
    boolArray.appendValue ( true );
    boolArray.appendValue ( false );

    boolArray.appendValue ( true );
    boolArray.appendValue ( true);
    boolArray.appendValue ( false);
    boolArray.appendValue ( false );

    boolArray.appendValue ( true );
    boolArray.appendValue ( true);
    boolArray.appendValue ( false);
    boolArray.appendValue ( false );

    boolArray.initialize();

    cout << "nBool DynamicArray values are ..." << endl;
    while ( boolArray.hasNextValue() )
          cout << boolArray.getValue() ;
    cout << endl;

    return 0;

}
```

有了类模板特化，我们可以从以下代码中观察到，对于`bool`、`char`和`double`，主要代码似乎是相同的，尽管主模板类`DynamicArray`和专门化的`DynamicArray<bool>`类是不同的：

```cpp
DynamicArray<char> charArray;
charArray.appendValue( 'H' );
charArray.appendValue( 'e' );

charArray.initialize();

cout << "nChar DynamicArray values are ..." << endl;
while ( charArray.hasNextValue() )
cout << charArray.getValue() << "t";
cout << endl;

DynamicArray<bool> boolArray;
boolArray.appendValue ( true );
boolArray.appendValue ( false );

boolArray.initialize();

cout << "nBool DynamicArray values are ..." << endl;
while ( boolArray.hasNextValue() )
      cout << boolArray.getValue() ;
cout << endl;
```

我相信你会发现这个 C++模板特化功能非常有用。

# 部分模板特化

与显式模板特化不同，显式模板特化用自己特定数据类型的完整定义替换主模板类，而部分模板特化允许我们专门化主模板类支持的某个子集的模板参数，而其他通用类型可以与主模板类相同。

当部分模板特化与继承结合时，可以做更多的事情，如下例所示：

```cpp
#include <iostream>
using namespace std;

template <typename T1, typename T2, typename T3>
class MyTemplateClass {
public:
     void F1( T1 t1, T2 t2, T3 t3 ) {
          cout << "nPrimary Template Class - Function F1 invoked ..." << endl;
          cout << "Value of t1 is " << t1 << endl;
          cout << "Value of t2 is " << t2 << endl;
          cout << "Value of t3 is " << t3 << endl;
     }

     void F2(T1 t1, T2 t2) {
          cout << "nPrimary Tempalte Class - Function F2 invoked ..." << endl;
          cout << "Value of t1 is " << t1 << endl;
          cout << "Value of t2 is " << 2 * t2 << endl;
     }
};
```

```cpp
template <typename T1, typename T2, typename T3>
class MyTemplateClass< T1, T2*, T3*> : public MyTemplateClass<T1, T2, T3> {
      public:
          void F1( T1 t1, T2* t2, T3* t3 ) {
               cout << "nPartially Specialized Template Class - Function F1 invoked ..." << endl;
               cout << "Value of t1 is " << t1 << endl;
               cout << "Value of t2 is " << *t2 << endl;
               cout << "Value of t3 is " << *t3 << endl;
          }
};
```

`main.cpp`文件将包含以下内容：

```cpp
#include "partiallyspecialized.h"

int main () {
    int x = 10;
    int *y = &x;
    int *z = &x;

    MyTemplateClass<int, int*, int*> obj;
    obj.F1(x, y, z);
    obj.F2(x, x);

    return 0;
}
```

从前面的代码中，你可能已经注意到，主模板类名称和部分特化类名称与完全或显式模板类特化的情况相同。然而，在模板参数表达式中有一些语法变化。在完全模板类特化的情况下，模板参数表达式将为空，而在部分特化的模板类的情况下，列出的表达式会出现，如下所示：

```cpp
template <typename T1, typename T2, typename T3>
class MyTemplateClass< T1, T2*, T3*> : public MyTemplateClass<T1, T2, T3> { };
```

表达式`template<typename T1, typename T2, typename T3>`是主类模板类中使用的模板参数表达式，`MyTemplateClass< T1, T2*, T3*>`是第二类所做的部分特化。正如你所看到的，第二类对`typename T2`和`typename T3`进行了一些特化，因为它们在第二类中被用作指针；然而，`typename T1`在第二类中被直接使用。

除了迄今为止讨论的事实之外，第二类还继承了主模板类，这有助于第二类重用主模板类的公共和受保护的方法。然而，部分模板特化并不会阻止特化类支持其他函数。

虽然主模板类中的`F1`函数被部分特化的模板类替换，但它通过继承重用了主模板类中的`F2`函数。

让我们使用以下命令快速编译程序：

```cpp
g++ main.cpp -std=c++17

./a.out
```

程序的输出如下：

```cpp
Partially Specialized Template Classs - Function F1 invoked ...
Value of t1 is 10
Value of t2 is 10
Value of t3 is 10

Primary Tempalte Class - Function F2 invoked ...
Value of t1 is 10
Value of t2 is 20
```

希望你觉得部分特化的模板类有用。

# 总结

在本章中，你学到了以下内容：

+   你现在知道了使用泛型编程的动机

+   你现在熟悉了函数模板

+   你知道如何重载函数模板

+   你知道类模板

+   你知道何时使用显式模板特化以及何时使用部分特化的模板特化

恭喜！总的来说，你对 C++的模板编程有很好的理解。

在下一章中，你将学习智能指针。
