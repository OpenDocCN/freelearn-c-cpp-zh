# 第五章：理解和设计模板

模板是 C++的一个独特特性，通过它，函数和类能够支持通用数据类型——换句话说，我们可以实现一个与特定数据类型无关的函数或类；例如，客户可能会请求一个`max()`函数来处理不同的数据类型。我们可以通过模板来实现一个`max()`，并将数据类型作为参数传递，而不是通过函数重载来实现和维护许多类似的函数。此外，模板可以与多重继承和运算符重载一起工作，以在 C++中创建强大的通用数据结构和算法，如**标准模板库**（**STL**）。此外，模板还可以应用于编译时计算、编译时和运行时代码优化等。

在本章中，我们将学习函数和类模板的语法，它们的实例化和特化。然后，我们将介绍*可变参数*模板及其应用。接下来，我们将讨论模板参数及用于实例化它们的相应参数。之后，我们将学习如何实现类型*特性*，以及如何利用这种类型的信息来优化算法。最后，我们将介绍在程序执行时可以使用的加速技术，包括编译时计算、编译时代码优化和静态多态性。

本章将涵盖以下主题：

+   探索函数和类模板

+   理解可变参数模板

+   理解模板参数和参数

+   什么是特性？

+   模板元编程及其应用

# 技术要求

本章的代码可以在本书的 GitHub 存储库中找到：[`github.com/PacktPublishing/Expert-CPP`](https://github.com/PacktPublishing/Expert-CPP)。

# 探索函数和类模板

我们将从介绍函数模板的语法及其实例化、推导和特化开始这一部分。然后，我们将转向类模板，并查看类似的概念和示例。

# 动机

到目前为止，当我们定义函数或类时，我们必须提供输入、输出和中间参数。例如，假设我们有一个函数来执行两个 int 类型整数的加法。我们如何扩展它，以便处理所有其他基本数据类型，如 float、double、char 等？一种方法是使用函数重载，手动复制、粘贴和稍微修改每个函数。另一种方法是定义一个宏来执行加法操作。这两种方法都有各自的副作用。

此外，如果我们修复一个 bug 或为一个类型添加一个新功能，这个更新需要在以后的所有其他重载函数和类中完成吗？除了使用这种愚蠢的复制-粘贴-替换方法外，我们有没有更好的方法来处理这种情况？

事实上，这是任何计算机语言都可能面临的一个通用问题。1973 年由通用函数式编程**元语言**（**ML**）首创，ML 允许编写通用函数或类型，这些函数或类型在使用时只在它们操作的类型集合上有所不同，从而减少了重复。后来受到**特许人寿保险师**（**CLU**）提供的参数化模块和 Ada 提供的泛型的启发，C++采用了模板概念，允许函数和类使用通用类型。换句话说，它允许函数或类在不需要重写的情况下处理不同的数据类型。

实际上，从抽象的角度来看，C++函数或类模板（如饼干模具）用作创建其他类似函数或类的模式。这背后的基本思想是创建一个函数或类模板，而无需指定某些或所有变量的确切类型。相反，我们使用占位符类型来定义函数或类模板，称为**模板类型参数**。一旦我们有了函数或类模板，我们可以通过在其他编译器中实现的算法自动生成函数或类。

C++中有三种模板：*函数*模板、*类*模板和*可变参数*模板。我们接下来将看看这些。

# 函数模板

函数模板定义了如何生成一组函数。这里的一组函数指的是行为类似的一组函数。如下图所示，这包括两个阶段：

+   创建函数模板；即编写它的规则。

+   模板实例化；即用于从模板生成函数的规则：

![](img/4ba0b575-0a51-403e-8a1f-1f3b03c37817.png)

函数模板格式

在上图的**part I**中，我们讨论了用于创建通用类型函数模板的格式，但是关于**专门化模板**，我们也称之为**主模板**。然后，在**part II**中，我们介绍了从模板生成函数的三种方式。最后，专门化和重载子节告诉我们如何为特殊类型自定义**主模板**（通过改变其行为）。

# 语法

有两种定义函数模板的方式，如下面的代码所示：

```cpp
template <typename identifier_1, …, typename identifier_n > 
function_declaration;

template <class identifier_1,…, class identifier_n> 
function_declaration;
```

在这里，`identifier_i (i=1,…,n)`是类型或类参数，`function_declaration`声明了函数体部分。在前两个声明中唯一的区别是关键字 - 一个使用`class`，而另一个使用`typename`，但两者的含义和行为都是相同的。由于类型（如基本类型 - int、float、double、enum、struct、union 等）不是类，因此引入了`typename`关键字方法以避免混淆。

例如，经典的查找最大值函数模板`app_max()`可以声明如下：

```cpp
template <class T>
T app_max (T a, T b) {
  return (a>b?a:b);   //note: we use ((a)>(b) ? (a):(b)) in macros  
}                     //it is safe to replace (a) by a, and (b) by b now
```

只要存在可复制构造的类型，其中 *a>b *表达式有效，这个函数模板就可以适用于许多数据类型或类。对于用户定义的类，这意味着必须定义大于号（>）。

请注意，函数模板和模板函数是不同的东西。函数模板指的是一种模板，用于由编译器生成函数，因此编译器不会为其生成任何目标代码。另一方面，模板函数意味着来自函数模板的实例。由于它是一个函数，编译器会生成相应的目标代码。然而，最新的 C++标准文档建议避免使用不精确的术语模板函数。因此，在本书中我们将使用函数模板和成员函数模板。

# 实例化

由于我们可能有无限多种类型和类，函数模板的概念不仅节省了源代码文件中的空间，而且使代码更易于阅读和维护。然而，与为应用程序中使用的不同数据类型编写单独的函数或类相比，它并不会产生更小的目标代码。例如，考虑使用`app_max()`的 float 和 int 版本的程序：

```cpp
cout << app_max<int>(3,5) << endl;
cout << app_max<float>(3.0f,5.0f) << endl;
```

编译器将在目标文件中生成两个新函数，如下所示：

```cpp
int app_max<int> ( int a, int b) {
  return (a>b?a:b);
}

float app_max<float> (float a, float b) {
  return (a>b?a:b);
}
```

从函数模板声明中创建函数的新定义的过程称为**模板实例化**。在这个实例化过程中，编译器确定模板参数，并根据应用程序的需求生成实际的功能代码。通常有三种形式：*显式实例化*，*隐式实例化*和*模板推断*。在接下来的部分，让我们讨论每种形式。

# 显式实例化

许多非常有用的 C++函数模板可以在不使用显式实例化的情况下编写和使用，但我们将在这里描述它们，只是让您知道如果您需要它们，它们确实存在。首先，让我们看一下 C++11 之前显式实例化的语法。有两种形式，如下所示：

```cpp
template return-type 
function_name < template_argument_list > ( function_parameter-list ) ;

template return-type 
function_name ( function_parameter_list ) ;
```

显式实例化定义，也称为**指令**，强制为特定类型的函数模板实例化，无论将来将调用哪个模板函数。显式实例化的位置可以在函数模板的定义之后的任何位置，并且在源代码中对于给定的参数列表只允许出现一次。

自 C++11 以来，显式实例化指令的语法如下。在这里，我们可以看到在`template`关键字之前添加了`extern`关键字：

```cpp
extern template return-type 
function_name < template_argument_list > (function_parameter_list ); 
(since C++11)

extern template return-type 
function_name ( function_parameter_list ); (since C++11)
```

使用`extern`关键字可以防止该函数模板的隐式实例化（有关更多详细信息，请参阅下一节）。

关于之前声明的`app_max()`函数模板，可以使用以下代码进行显式实例化：

```cpp
template double app_max<double>(double, double); 
template int app_max<int>(int, int);
```

也可以使用以下代码进行显式实例化：

```cpp
extern template double app_max<double>(double, double);//(since c++11)
extren template int app_max<int>(int, int);            //(since c++11)
```

这也可以以模板参数推断的方式完成：

```cpp
template double f(double, double);
template int f(int, int);
```

最后，这也可以这样做：

```cpp
extern template double f(double, double); //(since c++11)
extern template int f(int, int);          //(since c++11)
```

此外，显式实例化还有一些其他规则。如果您想了解更多，请参考*进一步阅读*部分[10]以获取更多详细信息。

# 隐式实例化

当调用函数时，该函数的定义需要存在。如果这个函数没有被显式实例化，将会采用隐式实例化的方法，其中模板参数的列表需要被显式提供或从上下文中推断出。以下程序的 A 部分提供了`app_max()`的隐式实例化的一些示例。

```cpp
//ch4_2_func_template_implicit_inst.cpp
#include <iostream>
template <class T>
T app_max (T a, T b) { return (a>b?a:b); }
using namespace std;
int main(){
 //Part A: implicit instantiation in an explicit way 
 cout << app_max<int>(5, 8) << endl;       //line A 
 cout << app_max<float>(5.0, 8.0) << endl; //line B
 cout << app_max<int>(5.0, 8) << endl;     //Line C
 cout << app_max<double>(5.0, 8) << endl;  //Line D

 //Part B: implicit instantiation in an argument deduction way
 cout << app_max(5, 8) << endl;           //line E 
 cout << app_max(5.0f, 8.0f) << endl;     //line F 

 //Part C: implicit instantiation in a confuse way
 //cout<<app_max(5, 8.0)<<endl;          //line G  
 return 0;
}
```

行`A`，`B`，`C`和`D`的隐式实例化分别是`int app_max<int>(int,int)`，`float app_max<float>(float, float>)`，`int app_max<int>(int,int)`和`double app_max<double>(double, double)`。

# 推断

当调用模板函数时，编译器首先需要确定模板参数，即使没有指定每个模板参数。大多数情况下，它会从函数参数中推断出缺失的模板参数。例如，在上一个函数的 B 部分中，当在行`E`中调用`app_max(5, 8)`时，编译器会推断模板参数为 int 类型，即`(int app_max<int>(int,int))`，因为输入参数`5`和`8`都是整数。同样，行`F`将被推断为浮点类型，即`float app_max<float>(float,float)`。

然而，如果在实例化过程中出现混淆会发生什么？例如，在上一个程序中对`G`的注释行中，根据编译器的不同，可能会调用`app_max<double>(double, double)`，`app_max<int>(int, int)`，或者只是给出一个编译错误消息。帮助编译器推断类型的最佳方法是通过显式给出模板参数来调用函数模板。在这种情况下，如果我们调用`app_max<double>(5, 8.0)`，任何混淆都将得到解决。

从编译器的角度来看，有几种方法可以进行模板参数推导——从函数调用中推导，从类型中推导，自动类型推导和非推导上下文[4]。然而，从程序员的角度来看，你不应该编写花哨的代码来滥用函数模板推导的概念，以混淆其他程序员，比如前面示例中的 G 行。

# 专门化和重载

专门化允许我们为给定的模板参数集自定义模板代码。它允许我们为特定的模板参数定义特殊行为。专门化仍然是一个模板；你仍然需要一个实例化来获得真正的代码（由编译器自动完成）。

在下面的示例代码中，主要函数模板`T app_max(T a, T b)`将根据`operator *a>b,*`的返回值返回`a`或`b`，但我们可以将其专门化为`T = std::string`，这样我们只比较`a`和`b`的第 0 个元素；也就是说，`a[0] >b[0]`：

```cpp
//ch4_3_func_template_specialization.cpp
#include <iostream>
#include <string>

//Part A: define a  primary template
template <class T> T app_max (T a, T b) { return (a>b?a:b); }

//Part B: explicit specialization for T=std::string, 
template <> std::string app_max<std::string> (std::string a, std::string b){ 
    return (a[0]>b[0]?a:b);
}

//part C: test function
using namespace std; 
void main(){
 string a = "abc", b="efg";
 cout << app_max(5, 6) << endl; //line A 
 cout << app_max(a, b) << endl; //line B 

 //question: what's the output if un-comment lines C and D?
 //char *x = "abc", *y="efg";     //Line C
 //cout << app_max(x, y) << endl; //line D
}
```

前面的代码首先定义了一个主模板，然后将`T`显式专门化为`std::string`；也就是说，我们只关心`a`和`b`的`a[0]`和`b[0]`（`app_max()`的行为被专门化）。在测试函数中，`行 A`调用`app_max<int>(int,int)`，`行 B`调用专门化版本，因为在推导时没有歧义。如果我们取消注释`C`和`D`行，将调用主函数模板`char* app_max<char > (char*, char*)`，因为`char*`和`std::string`是不同的数据类型。

从某种程度上讲，专门化与函数重载解析有些冲突：编译器需要一种算法来解决这种冲突，找到模板和重载函数中的正确匹配。选择正确函数的算法包括以下两个步骤：

1.  在常规函数和非专门化模板之间进行重载解析。

1.  如果选择了非专门化的模板，请检查是否存在一个更适合它的专门化。

例如，在下面的代码块中，我们声明了主要（`行 0`）和专门化的函数模板（`行 1-4`），以及`f()`的重载函数（`行 5-6`）：

```cpp
template<typename T1, typename T2> void f( T1, T2 );// line 0
template<typename T> void f( T );                   // line 1
template<typename T> void f( T, T );                // line 2
template<typename T> void f( int, T* );             // line 3
template<> void f<int>( int );                      // line 4
void f( int, double );                              // line 5
void f( int );                                      // line 6
```

`f()`将在下面的代码块中被多次调用。根据前面的两步规则，我们可以在注释中显示选择了哪个函数。我们将在此之后解释这样做的原因：

```cpp
int i=0; 
double d=0; 
float x=0;
complex<double> c;
f(i);      //line A: choose f() defined in line 6
f(i,d);    //line B: choose f() defined in line 5
f<int>(i); //line C: choose f() defined in line 4
f(c);      //line D: choose f() defined in line 1
f(i,i);    //line E: choose f() defined in line 2
f(i,x);    //line F: choose f() defined in line 0
f(i, &d);  //line G: choose f() defined in line 3

```

对于`行 A`和`行 B`，由于`行 5`和`行 6`中定义的`f()`是常规函数，它们具有最高的优先级被选择，所以`f(i)`和`f(i,d)`将分别选择它们。对于`行 C`，因为存在专门化的模板，从`行 4`生成的`f()`比从`行 1`生成的更匹配。对于`行 D`，由于`c`是`complex<double>`类型，只有在`行 1`中定义的主要函数模板与之匹配。`行 E`将选择由`行 2`创建的`f()`，因为两个输入变量是相同类型。最后，`行 F`和`行 G`将分别选择`行 0`和`行 3`中的模板创建的函数。

在了解了函数模板之后，我们现在将转向类模板。

# 类模板

类模板定义了一组类，并且通常用于实现容器。例如，C++标准库包含许多类模板，如`std::vector`、`std::map`、`std::deque`等。在 OpenCV 中，`cv::Mat`是一个非常强大的类模板，它可以处理具有内置数据类型的 1D、2D 和 3D 矩阵或图像，如`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`float`、`double`等。

与函数模板类似，如下图所示，类模板的概念包含模板创建语法、其专门化以及其隐式和显式实例化：

![](img/2f784eca-cdaf-490e-9514-942bf80883ac.png)

在前面的图表的**part I**中，使用特定的语法格式，我们可以为通用类型创建一个类模板，也称为主模板，并且可以根据应用的需求为特殊类型定制不同的成员函数和/或变量。一旦有了类模板，在**part II**中，编译器将根据应用的需求显式或隐式地将其实例化为模板类。

现在，让我们看一下创建类模板的语法。

# 语法

创建类模板的语法如下：

```cpp
[export] template <template_parameter_list> class-declaration 
```

在这里，我们有以下内容：

+   `template_parameter-list`（参见*进一步阅读*上下文中的链接[10]）是模板参数的非空逗号分隔列表，每个参数都是非类型参数、类型参数、模板参数或任何这些的参数包。

+   `class-declaration`是用于声明包含类名和其主体的类的部分，用大括号括起来。通过这样做，声明的类名也成为模板名。

例如，我们可以定义一个类模板`V`，使其包含各种 1D 数据类型：

```cpp
template <class T>
class V {
public:
  V( int n = 0) : m_nEle(n), m_buf(0) { creatBuf();}
  ~V(){  deleteBuf();  }
  V& operator = (const V &rhs) { /* ... */}
  V& operator = (const V &rhs) { /* ... */}
  T getMax(){ /* ... */ }
protected:
  void creatBuf() { /* ... */}
  void deleteBuf(){ /* ... */}

public:
  int m_nEle;
  T * m_buf;
};
```

一旦有了这个类模板，编译器就可以在实例化过程中生成类。出于我们在*函数模板*子节中提到的原因，我们将避免在本书中使用不精确的术语`template`类。相反，我们将使用类模板。

# 实例化

考虑到前一节中我们定义的类模板`V`，我们假设后面会出现以下声明：

```cpp
V<char> cV;
V<int>  iV(10);
V<float> fV(5);
```

然后，编译器将创建`V`类的三个实例，如下所示：

```cpp
class V<char>{
public:
  V(int n=0);
 // ...
public:
  int  m_nEle;
  char *m_buf;
};
class V<int>{
public:
  V(int n=0);
 // ...
public:
  int  m_nEle;
  int *m_buf;
};
class V<float>{
public:
  V(int n = 0);
  // ...
public:
  int   m_nEle;
  float *m_buf;
};
```

与函数模板实例化类似，类模板实例化有两种形式 - 显式实例化和隐式实例化。让我们来看看它们。

# 显式实例化

显式实例化的语法如下：

```cpp
template class template_name < argument_list >;
extern template class template_name < argument_list >;//(since C++11)
```

显式实例化定义会强制实例化它们所引用的类、结构或联合体。在 C++0x 标准中，模板特化或其成员的隐式实例化被抑制。与函数模板的显式实例化类似，这种显式实例化的位置可以在其模板定义之后的任何位置，并且在整个程序中只允许定义一次。

此外，自 C++11 以来，显式实例化声明（extern template）将绕过隐式实例化步骤，这可以用于减少编译时间。

回到模板类`V`，我们可以显式实例化它如下：

```cpp
template class V<int>;
template class V<double>;
```

或者，我们可以这样做（自 C++11 以来）：

```cpp
extern template class V<int>;
extern template class V<double>;
```

如果我们显式实例化函数或类模板，但程序中没有相应的定义，编译器将给出错误消息，如下所示：

```cpp
//ch4_4_class_template_explicit.cpp
#include <iostream>
using namespace std;
template <typename T>       //line A
struct A {
  A(T init) : val(init) {}
  virtual T foo();
  T val;
};                         //line B
                           //line C 
template <class T> //T in this line is template parameter
T A<T>::foo() {    //the 1st T refers to function return type,
                   //the T in <> specifies that this function's template
                   //parameter is also the class template parameter
  return val;
}                        //line D

extern template struct A<int>;  //line E
#if 0                           //line F
int A<int>::foo() {  
    return val+1;    
}                    
#endif                         //line G

int main(void) {
  A<double> x(5);
  A<int> y(5);
  cout<<"fD="<<x.foo()<<",fI="<<y.foo()<< endl;
  return 0;        //output: fD=5,fI=6
}
```

在前面的代码块中，我们在 A 行和 B 行之间定义了一个类模板，然后我们从 C 行到 D 行实现了它的成员函数`foo()`。接下来，我们在 E 行明确地为`int`类型实例化了它。由于在 F 行和 G 行之间的代码块被注释掉了（这意味着对于这个显式的`int`类型实例化，没有相应的`foo()`定义），我们会得到一个链接错误。为了解决这个问题，我们需要在 F 行用`#if 1`替换`#if 0`。

最后，显式实例化声明还有一些额外的限制，如下所示：

+   静态：静态类成员可以命名，但静态函数不能在显式实例化声明中允许。

+   内联：在显式实例化声明中，内联函数没有影响，内联函数会被隐式实例化。

+   类及其成员：显式实例化类及其所有成员没有等价物。

# 隐式实例化

当引用模板类时，如果没有显式实例化或显式专门化，编译器将只在需要时从其模板生成代码。这称为**隐式实例化**，其语法如下：

```cpp
class_name<argument list> object_name; //for non-pointer object 
class_name<argument list> *p_object_name; //for pointer object
```

对于非指针对象，模板类被实例化并创建其对象，但只生成此对象使用的成员函数。对于指针对象，除非程序中使用了成员，否则不会实例化。

考虑以下示例，在该示例中，我们在`ch4_5_class_template_implicit_inst.h`文件中定义了一个名为`X`的类模板。

```cpp
//file ch4_5_class_template_implicit_inst.h
#ifndef __CH4_5_H__ 
#define __CH4_5_H__ 
#include <iostream>
template <class T>
class X {
public:
    X() = default;
    ~X() = default;
    void f() { std::cout << "X::f()" << std::endl; };
    void g() { std::cout << "X::g()" << std::endl; };
};
#endif
```

然后，它被以下四个`cpp`文件包含，每个文件中都有`ain()`：

```cpp
//file ch4_5_class_template_implicit_inst_A.cpp
#include "ch4_5_class_template_implicit_inst.h"
void main()
{
    //implicit instantiation generates class X<int>, then create object xi
    X<int>   xi ;  
    //implicit instantiation generates class X<float>, then create object xf
    X<float> xf;
    return 0;  
}
```

在`ch4_5_class_template_implicit_inst_A.cpp`中，编译器将隐式实例化`X<int>`和`X<float>`类，然后创建`xi`和`xf`对象。但由于未使用`X::f()`和`X::g()`，它们不会被实例化。

现在，让我们看一下`ch4_5_class_template_implicit_inst_B.cpp`：

```cpp
//file ch4_5_class_template_implicit_inst_B.cpp
#include "ch4_5_class_template_implicit_inst.h"
void main()
{
    //implicit instantiation generates class X<int>, then create object xi
    X<int> xi;    
    xi.f();      //and generates function X<int>::f(), but not X<int>::g()

    //implicit instantiation generates class X<float>, then create object
    //xf and generates function X<float>::g(), but not X<float>::f()
    X<float> xf;  
    xf.g() ;   
}
```

在这里，编译器将隐式实例化`X<int>`类，创建`xi`对象，然后生成`X<int>::f()`函数，但不会生成`X<int>::g()`。类似地，它将实例化`X<float>`类，创建`xf`对象，并生成`X<float>::g()`函数，但不会生成`X<float>::f()`。

然后，我们有`ch4_5_class_template_implicit_inst_C.cpp`：

```cpp
//file ch4_5_class_template_implicit_inst_C.cpp
#include "ch4_5_class_template_implicit_inst.h"
void main()
{
   //inst. of class X<int> is not required, since p_xi is pointer object
   X<int> *p_xi ;   
   //inst. of class X<float> is not required, since p_xf is pointer object
   X<float> *p_xf ; 
}
```

由于`p_xi`和`p_xf`是指针对象，因此无需通过编译器实例化它们对应的模板类。

最后，我们有`ch4_5_class_template_implicit_inst_D.cpp`：

```cpp
//file ch4_5_class_template_implicit_inst_D.cpp
#include "ch4_5_class_template_implicit_inst.h"
void main()
{
//inst. of class X<int> is not required, since p_xi is pointer object
 X<int> *p_xi; 

 //implicit inst. of X<int> and X<int>::f(), but not X<int>::g()
 p_xi = new X<int>();
 p_xi->f(); 

//inst. of class X<float> is not required, since p_xf is pointer object
 X<float> *p_xf; 
 p_xf = new X<float>();//implicit inst. of X<float> occurs here
 p_xf->f();            //implicit inst. X<float>::f() occurs here
 p_xf->g();            //implicit inst. of X<float>::g() occurs here

 delete p_xi;
 delete p_xf;
}
```

这将隐式实例化`X<int>`和`X<int>::f()`，但不会实例化`X<int>::g()`；同样，对于`X<float>`，将实例化`X<float>::f()`和`X<float>::g()`。

# 专门化

与函数专门化类似，当将特定类型作为模板参数传递时，类模板的显式专门化定义了主模板的不同实现。但是，它仍然是一个类模板，您需要通过实例化来获得真正的代码。

例如，假设我们有一个`struct X`模板，可以存储任何数据类型的一个元素，并且只有一个名为`increase()`的成员函数。但是对于 char 类型数据，我们希望`increase()`有不同的实现，并且需要为其添加一个名为`toUpperCase()`的新成员函数。因此，我们决定为该类型声明一个类模板专门化。我们可以这样做：

1.  声明一个主类模板：

```cpp
template <typename T>
struct X {
  X(T init) : m(init) {}
  T increase() { return ++m; }
  T m;
};
```

这一步声明了一个主类模板，其中它的构造函数初始化了`m`成员变量，`increase()`将`m`加一并返回其值。

1.  接下来，我们需要为 char 类型数据执行专门化：

```cpp
template <>  //Note: no parameters inside <>, it tells compiler 
             //"hi i am a fully specialized template"
struct X<char> { //Note: <char> after X, tells compiler
                 // "Hi, this is specialized only for type char"
  X(char init) : m(init) {}
  char increase() { return (m<127) ? ++m : (m=-128); }
  char toUpperCase() {
    if ((m >= 'a') && (m <= 'z')) m += 'A' - 'a';
    return m;
  }
  char m;
};
```

这一步为 char 类型数据创建了一个专门化（相对于主类模板），并为其添加了一个额外的成员函数`toUpperCase()`。

1.  现在，我们进行测试：

```cpp
int main() {
 X<int> x1(5);         //line A
 std::cout << x1.increase() << std::endl;

 X<char> x2('b');     //line B
 std::cout << x2.toUpperCase() << std::endl;
 return 0;
}
```

最后，我们有一个`main()`函数来测试它。在 A 行，`x1`是一个从主模板`X<T>`隐式实例化的对象。由于`x1.m`的初始值是`5`，所以`x1.increase()`将返回`6`。在 B 行，`x2`是从专门化模板`X<char>`实例化的对象，当它执行时，`x2.m`的值是`b`。在调用`x2.toUpperCase()`之后，`B`将是返回值。

此示例的完整代码可以在`ch4_6_class_template_specialization.cpp`中找到。

总之，在类模板的显式专门化中使用的语法如下：

```cpp
template <> class[struct] class_name<template argument list> { ... }; 
```

在这里，空的模板参数列表`template <>`用于显式声明它为模板专门化，`<template argument list>`是要专门化的类型参数。例如，在`ex4_6_class_template_specialization.cpp`中，我们使用以下内容：

```cpp
template <> struct X<char> { ... };
```

在`X`之后的`<char>`标识了我们要为其声明模板类专门化的类型。

此外，当我们为模板类进行特化时，即使在主模板中相同的成员也必须被定义，因为在模板特化期间没有主模板的继承概念。

接下来，我们将看一下部分特化。这是显式特化的一般陈述。与只有模板参数列表的显式特化格式相比，部分特化需要模板参数列表和参数列表。对于模板实例化，如果用户的模板参数列表与模板参数的子集匹配，编译器将选择部分特化模板，然后编译器将从部分特化模板生成新的类定义。

在下面的示例中，对于主类模板`A`，我们可以为参数列表中的 const `T`进行部分特化。请注意，它们的参数列表相同，即`<typename T>`：

```cpp
//primary class template A
template <typename T>  class A{ /* ... */ }; 

//partial specialization for const T
template <typename T>  class A<const T>{ /* ... */ };  

```

在下面的示例中，主类模板`B`有两个参数：`<typename T1`和`typename T2 >`。我们通过`T1=int`进行部分特化，保持`T2`不变：

```cpp
//primary class template B
template <typename T1, typename T2> class B{ /* ... */ };          

//partial specialization for T1 = int
template <typename T2> class B<int, T2>{ /* ... */};  
```

最后，在下面的示例中，我们可以看到部分特化中的模板参数数量不必与原始主模板中出现的参数数量匹配。然而，模板参数的数量（出现在尖括号中的类名后面）必须与主模板中的参数数量和类型匹配：

```cpp
//primary class template C: template one parameter
template <typename T> struct C { T type; };  

//specialization: two parameters in parameter list 
//but still one argument (<T[N]>) in argument list
template <typename T, int N> struct C<T[N]>          
{T type; };                                 
```

同样，类模板的部分特化仍然是一个类模板。您必须为其成员函数和数量变量分别提供定义。

结束本节，让我们总结一下我们到目前为止学到的内容。在下表中，您可以看到函数和类模板、它们的实例化和特化之间的比较：

| | **函数模板** | **类模板** | **注释** |
| --- | --- | --- | --- |
| 声明 | `template <class T1, class T2>` `void f(T1 a, T2 b) { ... }` | `template <class T1, class T2>` `class X { ... };` | 声明定义了一个函数/类模板，`<class T1, class T2>`称为模板参数。 |
| 显式实例化 | `template void f <int, int >( int, int);`或`extern template`void f <int, int >( int, int);`（自 C++11 起） | `template class X<int, float>;`或`extern template class X<int,float>;`（自 C++11 起） | 实例化后现在有函数/类，但它们被称为模板函数/类。 |
| 隐式实例化 | {...`f(3, 4.5);` `f<char, float>(120, 3.14);`} | {...`X<int,float> obj;` `X<char, char> *p;`} | 当函数调用或类对象/指针声明时，如果没有被显式实例化，则使用隐式实例化方法。 |
| 特化 | `template <>` `void f<int,float>(int a, float b)` | `template <>` `class X <int, float>{ ... };` | 主模板的完全定制版本（无参数列表）仍然需要被实例化。 |
| 部分特化 | `template <class T>` `void f<T, T>(T a, T b)` | `template <class T>` `class X <T, T>` | 主模板的部分定制版本（有参数列表）仍然需要被实例化。 |

这里需要强调五个概念：

+   **声明**：我们需要遵循用于定义函数或类模板的语法。此时，函数或类模板本身不是类型、函数或任何其他实体。换句话说，在源文件中只有模板定义，没有代码可以编译成对象文件。

+   **隐式实例化**：对于任何代码的出现，都必须实例化一个模板。在这个过程中，必须确定模板参数，以便编译器可以生成实际的函数或类。换句话说，它们是按需编译的，这意味着在给定特定模板参数的实例化之前，模板函数或类的代码不会被编译。

+   **显式实例化**：告诉编译器使用给定类型实例化模板，无论它们是否被使用。通常用于提供库。

+   ****完全特化****：这没有参数列表（完全定制）；它只有一个参数列表。模板特化最有用的一点是，您可以为特定类型参数创建特殊模板。

+   **部分特化**：这类似于完全特化，但是部分参数列表（部分定制）和部分参数列表。

# 理解可变模板

在前一节中，我们学习了如何编写具有固定数量类型参数的函数或类模板。但自 C++11 以来，标准通用函数和类模板可以接受可变数量的类型参数。这被称为**可变模板**，它是 C++的扩展，详情请参阅*Further reading* [6]。我们将通过示例学习可变模板的语法和用法。

# 语法

如果一个函数或类模板需要零个或多个参数，可以定义如下：

```cpp
//a class template with zero or more type parameters
template <typename... Args> class X { ... };     

//a function template with zero or more type parameters
template <typename... Args> void foo( function param list) { ...}                                                                      
```

在这里，`<typename ... Args>`声明了一个参数包。请注意，这里的`Args`不是关键字；您可以使用任何有效的变量名。前面的类/函数模板可以接受任意数量的`typename`作为其需要实例化的参数，如下所示：

```cpp
X<> x0;                       //with 0 template type argument
X<int, std::vector<int> > x1; //with 2 template type arguments

//with 4 template type arguments
X<int, std::vector<int>, std::map<std::string, std::vector<int>>> x2; 

//with 2 template type arguments 
foo<float, double>( function argument list ); 

//with 3 template type arguments
foo<float, double, std::vector<int>>( function argument list );
```

如果可变模板需要至少一个类型参数，则使用以下定义：

```cpp
template <typename A, typename... Rest> class Y { ... }; 

template <typename A, typename... Rest> 
void goo( const int a, const float b) { ....};
```

同样，我们可以使用以下代码来实例化它们：

```cpp
Y<int > y1;                                         
Y<int, std::vector<int>, std::map<std::string, std::vector<int>>> y2;
goo<int, float>(  const int a, const float b );                        
goo<int,float, double, std::vector<int>>(  const int a, const float b );      
```

在前面的代码中，我们创建了`y1`和`y2`对象，它们是通过具有一个和三个模板参数的可变类模板`Y`的实例化而得到的。对于可变函数`goo`模板，我们将它实例化为两个模板函数，分别具有两个和三个模板参数。

# 示例

以下可能是最简单的示例，展示了使用可变模板来查找任何输入参数列表的最小值。这个示例使用了递归的概念，直到达到`my_min(double n)`为止：

```cpp
//ch4_7_variadic_my_min.cpp
//Only tested on g++ (Ubuntu/Linaro 7.3.0-27 ubuntu1~18.04)
//It may have compile errors for other platforms
#include <iostream>
#include <math.h> 
double my_min(double n){
  return n;
}
template<typename... Args>
double my_min(double n, Args... args){
  return fmin(n, my_min(args...));
}
int main() {
  double x1 = my_min(2);
  double x2 = my_min(2, 3);
  double x3 = my_min(2, 3, 4, 5, 4.7,5.6, 9.9, 0.1);
  std::cout << "x1="<<x1<<", x2="<<x2<<", x3="<<x3<<std::endl;
  return 0;
}
```

`printf()`可变参数函数可能是 C 或 C++中最有用和强大的函数之一；但是，它不是类型安全的。在下面的代码块中，我们采用了经典的类型安全`printf()`示例来演示可变模板的用处。首先，我们需要定义一个基本函数`void printf_vt(const char *s)`，它结束了递归：

```cpp
//ch4_8_variadic_printf.cpp part A: base function - recursive end
void printf_vt(const char *s)
{
  while (*s){
    if (*s == '%' && *(++s) != '%')
      throw std::runtime_error("invalid format string: missing arguments");
     std::cout << *s++;
  }
}
```

然后，在其可变模板函数`printf_vt()`中，每当遇到`%`时，该值被打印，其余部分被传递给递归，直到达到基本函数：

```cpp
//ch4_8_variadic_printf.cpp part B: recursive function
template<typename T, typename... Rest>
void printf_vt(const char *s, T value, Rest... rest)
{
  while (*s) {
    if (*s == '%' && *(++s) != '%') {
      std::cout << value;
      printf_vt(s, rest...); //called even when *s is 0, 
      return;                //but does nothing in that case
    }
    std::cout << *s++;
  }
}
```

最后，我们可以使用以下代码进行测试和比较传统的`printf()`。

```cpp
//ch4_8_variadic_printf.cpp Part C: testing
int main() {
  int x = 10;
  float y = 3.6;
  std::string s = std::string("Variadic templates");
  const char* msg1 = "%s can accept %i parameters (or %s), x=%d, y=%f\n";
  printf(msg1, s, 100, "more",x,y);  //replace 's' by 's.c_str()' 
                                     //to prevent the output bug
  const char* msg2 = "% can accept % parameters (or %); x=%,y=%\n";
  printf_vt(msg2, s, 100, "more",x,y);
  return 0;
}
```

前面代码的输出如下：

```cpp
p.]ï¿½U can accept 100 parameters (or more), x=10, y=3.600000
Variadic templates can accept 100 parameters (or more); x=10,y=3.6
```

在第一行的开头，我们可以看到一些来自`printf()`的 ASCII 字符，因为`%s`的相应变量类型应该是指向字符的指针，但我们给它一个`std::string`类型。为了解决这个问题，我们需要传递`s.c_str()`。然而，使用可变模板版本的函数，我们就没有这个问题。此外，我们只需要提供`%`，这甚至更好 - 至少对于这个实现来说是这样。

总之，本节简要介绍了可变模板及其应用。可变模板提供了以下好处（自 C++11 以来）：

+   这是模板家族的一个轻量级扩展。

+   它展示了在不使用丑陋的模板和预处理宏的情况下实现大量模板库的能力。因此，实现代码可以被理解和调试，并且还节省了编译时间。

+   它使`printf()`可变参数函数的类型安全实现成为可能。

接下来，我们将探讨模板参数和参数。

# 探索模板参数和参数

在前两节中，我们学习了函数和类模板及其实例化。我们知道，在定义模板时，需要给出其参数列表。而在实例化时，必须提供相应的参数列表。在本节中，我们将进一步研究这两个列表的分类和细节。

# 模板参数

回想一下以下语法，用于定义类/函数模板。在`template`关键字后面有一个`<>`符号，在其中必须给出一个或多个模板参数：

```cpp
//class template declaration
template <*parameter-list*> class-declaration

//function template declaration
template <parameter-list> function-declaration
```

参数列表中的参数可以是以下三种类型之一：

+   `非类型模板参数`：指的是编译时常量值，如整数和指针，引用静态实体。这些通常被称为非类型参数。

+   `类型模板参数`：指的是内置类型名称或用户定义的类。

+   `模板模板参数`：表示参数是其他模板。

我们将在接下来的小节中更详细地讨论这些内容。

# 非类型模板参数

非类型模板参数的语法如下：

```cpp
//for a non-type template parameter with an optional name
type name(optional)

//for a non-type template parameter with an optional name 
//and a default value
type name(optional)=default  

//For a non-type template parameter pack with an optional name
type ... name(optional) (since C++11) 
```

在这里，`type`是以下类型之一 - 整数类型、枚举、对象或函数的指针、对象或函数的`lvalue`引用、成员对象或成员函数的指针，以及`std::nullptr_t`（自 C++11 起）。此外，我们可以在模板声明中放置数组和/或函数类型，但它们会自动替换为数据和/或函数指针。

以下示例显示了一个使用非类型模板参数`int N`的类模板。在`main()`中，我们实例化并创建了一个对象`x`，因此`x.a`有五个初始值为`1`的元素。在将其第四个元素的值设置为`10`后，我们打印输出：

```cpp
//ch4_9_none_type_template_param1.cpp
#include <iostream>
template<int N>
class V {
public:
  V(int init) { 
    for (int i = 0; i<N; ++i) { a[i] = init; } 
  }
  int a[N];
};

int main()
{
  V<5> x(1); //x.a is an array of 5 int, initialized as all 1's 
  x.a[4] = 10;
  for( auto &e : x.a) {
    std::cout << e << std::endl;
  }
}
```

以下是一个使用`const char*`作为非类型模板参数的函数模板示例：

```cpp
//ch4_10_none_type_template_param2.cpp
#include <iostream>
template<const char* msg>
void foo() {
  std::cout << msg << std::endl;
}

// need to have external linkage
extern const char str1[] = "Test 1"; 
constexpr char str2[] = "Test 2";
extern const char* str3 = "Test 3";
int main()
{
  foo<str1>();                   //line 1
  foo<str2>();                   //line 2 
  //foo<str3>();                 //line 3

  const char str4[] = "Test 4";
  constexpr char str5[] = "Test 5";
  //foo<str4>();                 //line 4
  //foo<str5>();                 //line 5
  return 0;
}
```

在`main()`中，我们成功地用`str1`和`str2`实例化了`foo()`，因为它们都是编译时常量值并且具有外部链接。然后，如果我们取消注释第 3-5 行，编译器将报告错误消息。出现这些编译器错误的原因如下：

+   **第 3 行**：`str3`不是一个 const 变量，所以`str3`指向的值不能被改变。然而，`str3`的值可以被改变。

+   **第 4 行**：`str4`不是`const char*`类型的有效模板参数，因为它没有链接。

+   **第 5 行**：`str5`不是`const char*`类型的有效模板参数，因为它没有链接。

非类型参数的最常见用法之一是数组的大小。如果您想了解更多，请访问[`stackoverflow.com/questions/33234979`](https://stackoverflow.com/questions/33234979)。

# 类型模板参数

类型模板参数的语法如下：

```cpp
//A type Template Parameter (TP) with an optional name
typename |class name(optional)               

//A type TP with an optional name and a default
typename[class] name(optional) = default         

//A type TP pack with an optional name
typename[class] ... name(optional) (since C++11) 
```

**注意：**在这里，我们可以互换使用`typename`和`class`关键字。在模板声明的主体内，类型参数的名称是`typedef-name`。当模板被实例化时，它将别名为提供的类型。

现在，让我们看一些例子：

+   没有默认值的类型模板参数：

```cpp
Template<class T>               //with name
class X { /* ... */ };     

Template<class >               //without name
class Y { /* ... */ };
```

+   带有默认值的类型模板参数：

```cpp
Template<class T = void>    //with name 
class X { /* ... */ };     

Template<class = void >     //without name
class Y { /* ... */ };
```

+   类型模板参数包：

```cpp
template<typename... Ts>   //with name
class X { /* ... */ };

template<typename... >   //without name
class Y { /* ... */ };

```

这个模板参数包可以接受零个或多个模板参数，并且仅适用于 C++11 及以后的版本。

# 模板模板参数

模板模板参数的语法如下：

```cpp
//A template template parameter with an optional name
template <parameter-list> class *name*(optional) 

//A template template parameter with an optional name and a default
template <parameter-list> class *name*(optional) = default          

//A template template parameter pack with an optional name
template <parameter-list> class ... *name*(optional) (since C++11)                                                                                               
```

**注意**：在模板模板参数声明中，只能使用`class`关键字；不允许使用`typename`。在模板声明的主体中，参数的名称是`template-name`，我们需要参数来实例化它。

现在，假设您有一个函数，它充当对象列表的流输出运算符：

```cpp
template<typename T>
static inline std::ostream &operator << ( std::ostream &out, 
    std::list<T> const& v)
{ 
    /*...*/ 
}
```

从前面的代码中，您可以看到对于序列容器（如向量，双端队列和多种映射类型），它们是相同的。因此，使用模板模板参数的概念，可以有一个单一的运算符`<<`来控制它们。这种情况的示例可以在`exch4_tp_c.cpp`中找到：

```cpp
/ch4_11_template_template_param.cpp (courtesy: https://stackoverflow.com/questions/213761)
#include <iostream>
#include <vector>
#include <deque>
#include <list>
using namespace std;
template<class T, template<class, class...> class X, class... Args>
std::ostream& operator <<(std::ostream& os, const X<T, Args...>& objs) {
  os << __PRETTY_FUNCTION__ << ":" << endl;
  for (auto const& obj : objs)
    os << obj << ' ';
  return os;
}

int main() {
  vector<float> x{ 3.14f, 4.2f, 7.9f, 8.08f };
  cout << x << endl;

  list<char> y{ 'E', 'F', 'G', 'H', 'I' };
  cout << y << endl;

  deque<int> z{ 10, 11, 303, 404 };
  cout << z << endl;
  return 0;
}
```

前面程序的输出如下：

```cpp
class std::basic_ostream<char,struct std::char_traits<char> > &__cdecl operator
<<<float,class std::vector,class std::allocator<float>>(class std::basic_ostream
<char,struct std::char_traits<char> > &,const class std::vector<float,class std:
:allocator<float> > &):
3.14 4.2 7.9 8.08
class std::basic_ostream<char,struct std::char_traits<char> > &__cdecl operator
<<<char,class std::list,class std::allocator<char>>(class std::basic_ostream<cha
r,struct std::char_traits<char> > &,const class std::list<char,class std::alloca
tor<char> > &):
E F G H I
class std::basic_ostream<char,struct std::char_traits<char> > &__cdecl operator
<<<int,class std::deque,class std::allocator<int>>(class std::basic_ostream<char
,struct std::char_traits<char> > &,const class std::deque<int,class std::allocat
or<int> > &):
10 11 303 404 
```

如预期的那样，每次调用的输出的第一部分是`pretty`格式的模板函数名称，而第二部分输出每个容器的元素值。

# 模板参数

要实例化模板，必须用相应的模板参数替换所有模板参数。参数可以是显式提供的，从初始化程序中推导出（对于类模板），从上下文中推导出（对于函数模板），或者默认值。由于有三种模板参数类别，我们也将有三个相应的模板参数。这些是模板非类型参数，模板类型参数和模板模板参数。除此之外，我们还将讨论默认模板参数。

# 模板非类型参数

请注意，非类型模板参数是指编译时常量值，如整数，指针和对静态实体的引用。在模板参数列表中提供的非类型模板参数必须与这些值中的一个匹配。通常，非类型模板参数用于类初始化或类容器的大小规格。

尽管讨论每种类型（整数和算术类型，指向对象/函数/成员的指针，`lvalue`引用参数等）的详细规则超出了本书的范围，但总体的一般规则是模板非类型参数应转换为相应模板参数的常量表达式。

现在，让我们看下面的例子：

```cpp
//part 1: define template with non-type template parameters
template<const float* p> struct U {}; //float pointer non-type parameter
template<const Y& b> struct V {};     //L-value non-type parameter
template<void (*pf)(int)> struct W {};//function pointer parameter

//part 2: define other related stuff
void g(int,float);   //declare function g() 
void g(int);         //declare an overload function of g() 
struct Y {           //declare structure Y 
    float m1;
    static float m2;
};         
float a[10]; 
Y y; //line a: create a object of Y

//part 3: instantiation template with template non-type arguments
U<a> u1;      //line b: ok: array to pointer conversion
U<&y> u2;     //line c: error: address of Y
U<&y.m1> u3;  //line d: error: address of non-static member
U<&y.m2> u4;  //line e: ok: address of static member
V<y> v;       //line f: ok: no conversion needed
W<&g> w;      //line g: ok: overload resolution selects g(int)
```

在前面的代码中，在`part 1`中，我们定义了具有不同非类型模板参数的三个模板结构。然后，在`part 2`中，我们声明了两个重载函数和`struct Y`。最后，在`part 3`中，我们看了通过不同的非类型参数正确实例化它们的方法。

# 模板类型参数

与模板非类型参数相比，模板类型参数（用于类型模板参数）的规则很简单，要求必须是`typeid`。在这里，`typeid`是一个标准的 C++运算符，它在运行时返回类型识别信息。它基本上返回一个可以与其他`type_info`对象进行比较的`type_info`对象。

现在，让我们看下面的例子：

```cpp
//ch4_12_template_type_argument.cpp
#include <iostream>
#include <typeinfo>
using namespace std;

//part 1: define templates
template<class T> class C  {}; 
template<class T> void f() { cout << "T" << endl; }; 
template<int i>   void f() { cout << i << endl; };     

//part 2: define structures
struct A{};            // incomplete type 
typedef struct {} B; // type alias to an unnamed type

//part 3: main() to test
int main() {
  cout << "Tid1=" << typeid(A).name() << "; "; 
  cout << "Tid2=" << typeid(A*).name() << "; ";    
  cout << "Tid3=" << typeid(B).name()  << "; ";
  cout << "Tid4=" << typeid(int()).name() << endl;

  C<A> x1;    //line A: ok,'A' names a type
  C<A*> x2;   //line B: ok, 'A*' names a type
  C<B> x3;    //line C: ok, 'B' names a type
  f<int()>(); //line D: ok, since int() is considered as a type, 
              //thus calls type template parameter f()
  f<5>();     //line E: ok, this calls non-type template parameter f() 
  return 0;
}
```

在这个例子中，在`part 1`中，我们定义了三个类和函数模板：具有其类型模板参数的类模板 C，具有类型模板参数的两个函数模板，以及一个非类型模板参数。在`part 2`中，我们有一个不完整的`struct A`和一个无名类型`struct B`。最后，在`part 3`中，我们对它们进行了测试。在 Ubuntu 18.04 中四个`typeid()`的输出如下：

```cpp
Tid1=A; Tid2=P1A; Tid3=1B; Tid4=FivE
```

从 x86 MSVC v19.24，我们有以下内容：

```cpp
Tid1=struct A; Tid2=struct A; Tid3=struct B; Tid4=int __cdecl(void)
```

另外，由于`A`，A*，`B`和`int()`具有 typeid，因此从 A 到 D 行的代码段与模板类型类或函数相关联。只有 E 行是从非类型模板参数函数模板实例化的，即`f()`。

# 模板模板参数

对于模板模板参数，其对应的模板参数是类模板或模板别名的名称。在查找与模板模板参数匹配的模板时，只考虑主类模板。

这里，主模板是指正在进行特化的模板。即使它们的参数列表可能匹配，编译器也不会考虑与模板模板参数的部分特化。

以下是模板模板参数的示例：

```cpp
//ch4_13_template_template_argument.cpp
#include <iostream>
#include <typeinfo>
using namespace std;

//primary class template X with template type parameters
template<class T, class U> 
class X {
public:
    T a;
    U b;
};

//partially specialization of class template X
template<class U> 
class X<int, U> {
public:
    int a;  //customized a
    U b;
};

//class template Y with template template parameter
template<template<class T, class U> class V> 
class Y {
public:
    V<int, char> i;
    V<char, char> j;
};

Y<X> c;
int main() {
    cout << typeid(c.i.a).name() << endl; //int
    cout << typeid(c.i.b).name() << endl; //char
    cout << typeid(c.j.a).name() << endl; //char
    cout << typeid(c.j.b).name() << endl; //char
    return 0;
}
```

在这个例子中，我们定义了一个主类模板`X`及其特化，然后是一个带有模板模板参数的类模板`Y`。接下来，我们隐式实例化`Y`，并使用模板模板参数`X`创建一个对象`c`。最后，`main()`输出了四个`typeid()`的名称，结果分别是`int`、`char`、`char`和`char`。

# 默认模板参数

在 C++中，通过传递参数来调用函数，并且函数使用这些参数。如果在调用函数时未传递参数，则使用默认值。与函数参数默认值类似，模板参数可以有默认参数。当我们定义模板时，可以设置其默认参数，如下所示：

```cpp
/ch4_14_default_template_arguments.cpp       //line 0
#include <iostream>                          //line 1  
#include <typeinfo>                          //line 2
template<class T1, class T2 = int> class X;  //line 3
template<class T1 = float, class T2> class X;//line 4
template<class T1, class T2> class X {       //line 5
public:                                      //line 6   
 T1 a;                                       //line 7
 T2 b;                                       //line 8  
};                                           //line 9
using namespace std;
int main() { 
 X<int> x1;          //<int,int>
 X<float>x2;         //<float,int>
 X<>x3;              //<float,int>
 X<double, char> x4; //<double, char>
 cout << typeid(x1.a).name() << ", " << typeid(x1.b).name() << endl;
 cout << typeid(x2.a).name() << ", " << typeid(x2.b).name() << endl;
 cout << typeid(x3.a).name() << ", " << typeid(x3.b).name() << endl;
 cout << typeid(x4.a).name() << ", " << typeid(x4.b).name() << endl;
 return 0
}
```

在设置模板参数的默认参数时，需要遵循一些规则：

+   声明顺序很重要——默认模板参数的声明必须在主模板声明的顶部。例如，在前面的例子中，不能将代码移动到第 3 行和第 4 行之后的第 9 行之后。

+   如果一个参数有默认参数，那么它后面的所有参数也必须有默认参数。例如，以下代码是不正确的：

```cpp
template<class U = char, class V, class W = int> class X { };  //Error 
template<class V, class U = char,  class W = int> class X { }; //OK
```

+   在同一作用域中不能给相同的参数设置默认参数两次。例如，如果使用以下代码，将收到错误消息：

```cpp
template<class T = int> class Y;

//compiling error, to fix it, replace "<class T = int>" by "<class T>"
template<class T = int> class Y { 
    public: T a;  
};
```

在这里，我们讨论了两个列表：`template_parameter_list`和`template_argument_list`。这些分别用于函数或类模板的创建和实例化。

我们还了解了另外两个重要规则：

+   当我们定义类或函数模板时，需要给出其`template_parameter_list`：

```cpp
template <template_parameter_list> 
class X { ... }

template <template_parameter_list> 
void foo( function_argument_list ) { ... } //assume return type is void
```

+   当我们实例化它们时，必须提供相应的`argument_list`：

```cpp
class X<template_argument_list> x
void foo<template_argument_list>( function_argument_list )
```

这两个列表中的参数或参数类型可以分为三类，如下表所示。请注意，尽管顶行是用于类模板，但这些属性也适用于函数模板：

|  | **定义模板时****template** **<template_parameter_list> class X { ... }** | **实例化模板时****class X<template_argument_list> x** |
| --- | --- | --- |

| 非类型 | 此参数列表中的实体可以是以下之一：

+   整数或枚举

+   对象指针或函数指针

+   对对象的`lvalue`引用或对函数的`lvalue`引用

+   成员指针

+   C++11 std `::nullptr_t` C++11 结束

|

+   此列表中的非类型参数是在编译时可以确定其值的表达式。

+   这些参数必须是常量表达式、具有外部链接的函数或对象的地址，或者静态类成员的地址。

+   非类型参数通常用于初始化类或指定类成员的大小。

|

| 类型 | 此参数列表中的实体可以是以下之一：

+   必须以 typename 或 class 开头。

+   在模板声明的主体中，类型参数的名称是`typedef-name`。当模板被实例化时，它将别名为提供的类型。

|

+   参数的类型必须有`typeid`。

+   它不能是局部类型、没有链接的类型、无名类型或由这些类型中的任何一个构成的类型。

|

| 模板 | 此参数列表中的实体可以是以下之一：

+   `template <parameter-list>` class name

+   `template <parameter-list>` class ... name (optional) (自 C++11 起)

| 此列表中的模板参数是类模板的名称。 |
| --- |

在接下来的部分中，我们将探讨如何在 C++中实现特征，并使用它们优化算法。

# 探索特征

泛型编程意味着编写适用于特定要求下的任何数据类型的代码。这是在软件工程行业中提供可重用高质量代码的最有效方式。然而，在泛型编程中有时候泛型并不够好。每当类型之间的差异过于复杂时，一个高效的泛型优化常见实现就会变得非常困难。例如，当实现排序函数模板时，如果我们知道参数类型是链表而不是数组，就会实现不同的策略来优化性能。

尽管模板特化是克服这个问题的一种方法，但它并不能以广泛的方式提供与类型相关的信息。类型特征是一种用于收集有关类型信息的技术。借助它，我们可以做出更明智的决策，开发高质量的优化算法。

在本节中，我们将介绍如何实现类型特征，然后向您展示如何使用类型信息来优化算法。

# 类型特征实现

为了理解类型特征，我们将看一下`boost::is_void`和`boost::is_pointer`的经典实现。

# boost::is_void

首先，让我们来看一下最简单的特征类之一，即由 boost 创建的`is_void`特征类。它定义了一个通用模板，用于实现默认行为；也就是说，接受 void 类型，但其他任何类型都是 void。因此，我们有`is_void::value = false`。

```cpp
//primary class template is_void
template< typename T >
struct is_void{
    static const bool value = false;  //default value=false 
};
```

然后，我们对 void 类型进行了完全特化：

```cpp
//"<>" means a full specialization of template class is_void
template<> 
struct is_void< void >{             //fully specialization for void
    static const bool value = true; //only true for void type
};
```

因此，我们有一个完整的特征类型，可以用来检测任何给定类型`T`是否通过检查以下表达式`is_void`。

```cpp
is_void<T>::value
```

接下来，让我们学习如何在`boost::is_pointer`特征中使用部分特化。

# boost::is_pointer

与`boost::avoid`特征类类似，首先定义了一个主类模板：

```cpp
//primary class template is_pointer
template< typename T > 
struct is_pointer{
    static const bool value = false;
};
```

然后，它对所有指针类型进行了部分特化：

```cpp
//"typename T" in "<>" means partial specialization
template< typename T >   
struct is_pointer< T* >{ //<T*> means partial specialization only for type T* 
  static const bool value = true;  //set value as true
};
```

现在，我们有一个完整的特征类型，可以用来检测任何给定类型`T`是否通过检查以下表达式`is_pointer`。

```cpp
is_pointer<T>::value
```

由于 boost 类型特征功能已经正式引入到 C++ 11 标准库中，我们可以在下面的示例中展示`std::is_void`和`std::is_pointer`的用法，而无需包含前面的源代码：

```cpp
//ch4_15_traits_boost.cpp
#include <iostream>
#include <type_traits>  //since C++11
using namespace std;
struct X {};
int main()
{
 cout << boolalpha; //set the boolalpha format flag for str stream.
 cout << is_void<void>::value << endl;          //true
 cout << is_void<int>::value << endl;           //false
 cout << is_pointer<X *>::value << endl;        //true
 cout << is_pointer<X>::value << endl;          //false
 cout << is_pointer<X &>::value << endl;        //false
 cout << is_pointer<int *>::value << endl;      //true
 cout << is_pointer<int **>::value << endl;     //true
 cout << is_pointer<int[10]>::value << endl;    //false
 cout << is_pointer< nullptr_t>::value << endl; //false
}
```

前面的代码在字符串流的开头设置了`boolalpha`格式标志。通过这样做，所有的布尔值都以它们的文本表示形式提取，即 true 或 false。然后，我们使用几个`std::cout`来打印`is_void<T>::value`和`is_pointer<T>::value`的值。每个值的输出显示在相应的注释行末尾。

# 使用特征优化算法

我们将使用一个经典的优化复制示例来展示类型特征的用法，而不是以一种泛型抽象的方式来讨论这个主题。考虑标准库算法`copy`：

```cpp
template<typename It1, typename It2> 
It2 copy(It1 first, It1 last, It2 out);
```

显然，我们可以为任何迭代器类型编写`copy()`的通用版本，即这里的`It1`和`It2`。然而，正如 boost 库的作者所解释的那样，有些情况下复制操作可以通过`memcpy()`来执行。如果满足以下所有条件，我们可以使用`memcpy()`：

+   `It1`和`It2`这两种迭代器都是指针。

+   `It1`和`It2`必须指向相同的类型，除了 const 和 volatile 限定符

+   `It1`指向的类型必须提供一个平凡的赋值运算符。

这里，平凡的赋值运算符意味着该类型要么是标量类型，要么是以下类型之一：

+   该类型没有用户定义的赋值运算符。

+   该类型内部没有数据成员的引用类型。

+   所有基类和数据成员对象必须定义平凡的赋值运算符。

在这里，标量类型包括算术类型、枚举类型、指针、成员指针，或者这些类型的 const 或 volatile 修饰版本。

现在，让我们看一下原始实现。它包括两部分 - 复制器类模板和用户界面函数，即`copy()`：

```cpp
namespace detail{
//1\. Declare primary class template with a static function template
template <bool b>
struct copier {
    template<typename I1, typename I2>
    static I2 do_copy(I1 first, I1 last, I2 out);
};
//2\. Implementation of the static function template
template <bool b>
template<typename I1, typename I2>
I2 copier<b>::do_copy(I1 first, I1 last, I2 out) {
    while(first != last) {
        *out = *first; 
         ++out;
         ++first;
    }
    return out;
};
//3\. a full specialization of the primary function template
template <>
struct copier<true> {
    template<typename I1, typename I2>
    static I2* do_copy(I1* first, I1* last, I2* out){
        memcpy(out, first, (last-first)*sizeof(I2));
        return out+(last-first);
    }
};
}  //end namespace detail
```

如注释行中所述，前面的复制器类模板有两个静态函数模板 - 一个是主要的，另一个是完全专门化的。主要的函数模板进行逐个元素的硬拷贝，而完全专门化的函数模板通过`memcpy()`一次性复制所有元素：

```cpp
//copy() user interface 
template<typename I1, typename I2>
inline I2 copy(I1 first, I1 last, I2 out) {
    typedef typename boost::remove_cv
    <typename std::iterator_traits<I1>::value_type>::type v1_t;

    typedef typename boost::remove_cv
    <typename std::iterator_traits<I2>::value_type>::type v2_t;

    enum{ can_opt = boost::is_same<v1_t, v2_t>::value
                    && boost::is_pointer<I1>::value
                    && boost::is_pointer<I2>::value
                    && boost::has_trivial_assign<v1_t>::value 
   };
   //if can_opt= true, using memcpy() to copy whole block by one 
   //call(optimized); otherwise, using assignment operator to 
   //do item-by-item copy
   return detail::copier<can_opt>::do_copy(first, last, out);
}
```

为了优化复制操作，前面的用户界面函数定义了两个`remove_cv`模板对象，`v1_t`和`v2_t`，然后评估`can_opt`是否为真。之后，调用`do_copy()`模板函数。通过使用 boost 实用程序库中发布的测试代码（`algo_opt_ examples.cpp`），我们可以看到使用优化实现有显著改进；即对于复制 char 或 int 类型的数据，速度可能提高 8 倍或 3 倍。

最后，让我们用以下要点总结本节：

+   特征除了类型之外还提供额外的信息。它通过模板特化来实现。

+   按照惯例，特征总是作为结构体实现。用于实现特征的结构体称为特征类。

+   Bjarne Stroustrup 说我们应该将特征视为一个小对象，其主要目的是携带另一个对象或算法使用的信息，以确定策略或实现细节。*进一步阅读*上下文[4]

+   Scott Meyers 还总结说我们应该使用特征类来收集有关类型的信息*进一步阅读*上下文[5]。

+   特征可以帮助我们以高效/优化的方式实现通用算法。

接下来，我们将探讨 C++中的模板元编程。

# 探索模板元编程

一种计算机程序具有将其他程序视为其数据的能力的编程技术被称为**元编程**。这意味着程序可以被设计为读取、生成、分析或转换其他程序，甚至在运行时修改自身。一种元编程是编译器，它以文本格式程序作为输入语言（C、Fortran、Java 等），并以另一种二进制机器代码格式程序作为输出语言。

C++ **模板元编程**（**TMP**）意味着使用模板在 C++中生成元程序。它有两个组成部分 - 必须定义一个模板，并且必须实例化已定义的模板。TMP 是图灵完备的，这意味着它至少在原则上有能力计算任何可计算的东西。此外，因为在 TMP 中变量都是不可变的（变量是常量），所以递归而不是迭代用于处理集合的元素。

为什么我们需要 TMP？因为它可以加速程序的执行时间！但在优化世界中并没有免费的午餐，我们为 TMP 付出的代价是更长的编译时间和/或更大的二进制代码大小。此外，并非每个问题都可以用 TMP 解决；它只在我们在编译时计算某些常量时才起作用；例如，找出小于常量整数的所有质数，常量整数的阶乘，展开常量次数的循环或迭代等。

从实际角度来看，模板元编程有能力解决以下三类问题：编译时计算、编译时优化，以及通过在运行时避免虚拟表查找，用静态多态性替换动态多态性。在接下来的小节中，我们将提供每个类别的示例，以演示元编程的工作原理。

# 编译时计算

通常，如果任务的输入和输出在编译时已知，我们可以使用模板元编程来在编译期间进行计算，从而节省任何运行时开销和内存占用。这在实时强度 CPU 利用项目中非常有用。

让我们来看一下计算`*n*!`的阶乘函数。这是小于或等于*n*的所有正整数的乘积，其中根据定义 0!=1。由于递归的概念，我们可以使用一个简单的函数来实现这一点，如下所示：

```cpp
//ch4_17_factorial_recursion.cpp
#include <iostream>
uint32_t f1(const uint32_t n) {
  return (n<=1) ? 1 : n * f1(n - 1);
}

constexpr uint32_t f2(const uint32_t n) {
  return ( n<=1 )? 1 : n * f2(n - 1);
}

int main() {
  uint32_t a1 = f1(10);         //run-time computation 
  uint32_t a2 = f2(10);         //run-time computation 
  const uint32_t a3 = f2(10);   //compile-time computation 
  std::cout << "a1=" << a1 << ", a2=" << a2 << std::endl;
}
```

`f1()`在运行时进行计算，而`f2()`可以根据使用情况在运行时或编译时进行计算。

同样，通过使用带有非类型参数的模板，它的特化和递归概念，这个问题的模板元编程版本如下：

```cpp
//ch4_18_factorial_metaprogramming.cpp
#include <iostream>
//define a primary template with non-type parameters
template <uint32_t n> 
struct fact {
  ***const static uint32_t*** value = n * fact<n - 1>::value;
  //use next line if your compiler does not support declare and initialize
  //a constant static int type member inside the class declaration 
  //enum { value = n * fact<n - 1>::value }; 
};

//fully specialized template for n as 0
template <> 
struct fact<0> { 
    const static uint32_t value = 1;
    //enum { value = 1 };
};
using namespace std;
int main() {
    cout << "fact<0>=" << fact<0>::value << endl;   //fact<0>=1
    cout << "fact<10>=" << fact<10>::value << endl; //fact<10>=3628800

    //Lab: uncomments the following two lines, build and run 
    //     this program, what are you expecting? 
    //uint32_t m=5;
    //std::cout << fact<m>::value << std::endl;
}
```

在这里，我们创建了一个带有非类型参数的类模板，与其他 const 表达式一样，`const static uint32_t`或枚举常量的值在编译时计算。这种编译时评估约束意味着只有 const 变量有意义。此外，由于我们只使用类，静态对象才有意义。

当编译器看到模板的新参数时，它会创建模板的新实例。例如，当编译器看到`fact<10>::value`并尝试使用参数为 10 创建`fact`的实例时，结果是必须创建`fact<9>`。对于`fact<9>`，它需要`fact<8>`等等。最后，编译器使用`fact<0>::value`（即 1），并且在编译时的递归终止。这个过程可以在以下代码块中看到：

```cpp
fact<10>::value = 10* fact<9>::value;
fact<10>::value = 10* 9 * fact<8>::value;
fact<10>::value = 10* 9 * 8 * fact<7>::value;
.
.
.
fact<10>::value = 10* 9 * 8 *7*6*5*4*3*2*fact<1>::value;
fact<10>::value = 10* 9 * 8 *7*6*5*4*3*2*1*fact<0>::value;
...
fact<10>::value = 10* 9 * 8 *7*6*5*4*3*2*1*1;
```

请注意，为了能够以这种方式使用模板，我们必须在模板参数列表中提供一个常量参数。这就是为什么如果取消注释代码的最后两行，编译器会投诉：`fact:template parameter n: m: a variable with non-static storage duration cannot be used as a non-type argument`。

最后，让我们通过简要比较**constexpr 函数**（CF）和 TMP 来结束本小节：

+   **计算时间**：CF 根据使用情况在编译时或运行时执行，但 TMP 只在编译时执行。

+   **参数列表**：CF 只能接受值，但 TMP 可以接受值和类型参数。

+   控制结构：CF 可以使用递归、条件和循环，但 TMP 只能使用递归。

# 编译时代码优化

尽管前面的例子可以在编译时计算常量整数的阶乘，但我们可以使用运行时循环来展开两个-*n*向量的点积（其中*n*在编译时已知）。传统长度-*n*向量的好处是可以展开循环，从而产生非常优化的代码。

例如，传统的点积函数模板可以以以下方式实现：

```cpp
//ch4_19_loop_unoolling_traditional.cpp
#include <iostream>
using namespace std;
template<typename T>
T dotp(int n, const T* a, const T* b)
{
  T ret = 0;
  for (int i = 0; i < n; ++i) {
      ret += a[i] * b[i];
  }
  return ret;
}

int main()
{
  float a[5] = { 1, 2, 3, 4, 5 };
  float b[5] = { 6, 7, 8, 9, 10 };
  cout<<"dot_product(5,a,b)=" << dotp<float>(5, a, b) << '\n'; //130
  cout<<"dot_product(5,a,a)=" << dotp<float>(5, a, a) << '\n'; //55
}
```

**循环展开**意味着如果我们可以优化`dotp()`函数内部的 for 循环为`a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4]`，那么它将节省更多的运行时计算。这正是元编程在以下代码块中所做的：

```cpp
//ch4_20_loop_unroolling_metaprogramming.cpp
#include <iostream>

//primary template declaration
template <int N, typename T>    
class dotp {
public:
  static T result(T* a, T* b) {
    return (*a) * (*b) + dotp<N - 1, T>::result(a + 1, b + 1);
  }
};

//partial specialization for end condition
template <typename T>   
class dotp<1, T> {
public:
  static T result(T* a, T* b) {
    return (*a) * (*b);
  }
};

int main()
{
  float a[5] = { 1, 2, 3, 4, 5 };
  float b[5] = { 6, 7, 8, 9, 10 };
  std::cout << "dot_product(5,a,b) = " 
            << dotp<5, float>::result( a, b) << '\n'; //130
  std::cout << "dot_product(5,a,a) = " 
            << dotp<5,float>::result( a, a) << '\n'; //55
}
```

类似于阶乘元编程示例，在`dotp<5, float>::result(a, b)`语句中，实例化过程递归执行以下计算：

```cpp
dotp<5, float>::result( a, b)
= *a * *b + dotp<4,float>::result(a+1,b+1)
= *a * *b + *(a+1) * *(b+1) + dotp<3,float>::result(a+2,b+2)
= *a * *b + *(a+1) * *(b+1) + *(a+2) * *(b+2) 
  + dotp<2,float>::result(a+3,b+3)
= *a * *b + *(a+1) * *(b+1) + *(a+2) * *(b+2) + *(a+3) * *(b+3) 
  + dotp<1,float>::result(a+4,b+4)
= *a * *b + *(a+1) * *(b+1) + *(a+2) * *(b+2) + *(a+3) * *(b+3) 
  + *(a+4) * *(b+4)
```

由于*N*为 5，它递归调用`dotp<n, float>::results()`模板函数四次，直到达到`dotp<1, float>::results()`。由`dotp<5, float>::result(a, b)`计算的最终表达式显示在前面块的最后两行中。

# 静态多态

多态意味着多个函数具有相同的名称。动态多态允许用户在运行时确定要执行的实际函数方法，而静态多态意味着在编译时已知要调用的实际函数（或者一般来说，要运行的实际代码）。默认情况下，C++通过检查类型和/或参数的数量在编译时匹配函数调用与正确的函数定义。这个过程也被称为静态绑定或重载。然而，通过使用虚函数，编译器也可以在运行时进行动态绑定或覆盖。

例如，在以下代码中，虚函数`alg()`在基类 B 和派生类 D 中都有定义。当我们使用派生对象指针`p`作为基类的实例指针时，`p->alg()`函数调用将调用派生类中定义的`alg()`：

```cpp
//ch4_21_polymorphism_traditional.cpp
#include <iostream>
class B{
public:
    B() = default;
    virtual void alg() { 
        std::cout << "alg() in B"; 
    }
};

class D : public B{
public:
    D() = default; 
    virtual void alg(){
        std::cout << "alg() in D"; 
    }
};

int main()
{
    //derived object pointer p as an instance pointer of the base class
    B *p = new D();
    p->alg();       //outputs "alg() in D"
    delete p;
    return 0;
}
```

然而，在多态行为不变且可以在编译时确定的情况下，可以使用奇异递归模板模式（CRTP）来实现静态多态，模拟静态多态并在编译时解析绑定。因此，程序将在运行时摆脱对虚拟查找表的检查。以下代码以静态多态的方式实现了前面的示例：

```cpp
//ch4_22_polymorphism_metaprogramming.cpp
#include <iostream>
template <class D> struct B {
    void ui() {
        static_cast<D*>(this)->alg();
    }
};

struct D : B<D> {
    void alg() {
        cout << "D::alg()" << endl;
     }
};

int main(){
    B<D> b;
    b.ui();
    return 0;
}
```

总之，模板元编程的一般思想是让编译器在编译时进行一些计算。通过这种方式，可以在一定程度上解决运行时开销的问题。我们之所以能够在编译时计算某些东西，是因为在运行时之前，某些东西是常量。

如进一步阅读中提到的，C++ TMP 是一种非常强大的方法，可以在编译时执行计算任务。第一种方法并不容易，我们必须非常小心处理编译错误，因为模板树是展开的。从实际角度来看，boost 元编程库（MPL）是一个很好的起点。它以通用方式提供了用于算法、序列和元函数的编译时 TMP 框架。此外，C++17 中的新特性 std::variant 和 std::visit 也可以用于静态多态，适用于没有相关类型共享继承接口的情况。

# 总结

在本章中，我们讨论了 C++中与泛型编程相关的主题。从回顾 C 宏和函数重载开始，我们介绍了 C++模板的开发动机。然后，我们介绍了具有固定数量参数的类和函数模板的语法，以及它们的特化和实例化。自 C++11 以来，标准泛型函数和类模板已经接受可变参数模板。基于此，我们进一步将模板参数和参数分为三类：非类型模板参数/参数，类型模板参数/参数和模板模板参数/参数。

我们还学习了特性和模板元编程。作为模板特化的副产品，特性类可以为我们提供有关类型的更多信息。借助类型信息，最终可以实现实现通用算法的优化。类和/或函数模板的另一个应用是通过递归在编译时计算一些常量任务，这被称为模板元编程。它具有执行编译时计算和/或优化的能力，并且可以避免在运行时进行虚拟表查找。

现在，你应该对模板有了深入的了解。你应该能够在应用程序中创建自己的函数和类模板，并练习使用特性来优化你的算法，并使用模板元编程来进行编译时计算以进行额外的优化。

在下一章中，我们将学习有关内存和管理相关主题的内容，例如内存访问、分配和释放技术的概念，以及垃圾收集基础知识。这是 C++最独特的特性，因此每个 C++开发人员都必须了解。

# Questions

1.  宏的副作用是什么？

1.  什么是类/函数模板？什么是模板类/函数？

1.  什么是模板参数列表？什么是模板参数列表？一旦我们有了一个类模板，我们可以显式或隐式地实例化它。在什么样的情况下，显式实例化是必要的？

1.  在 C++中，多态是什么意思？函数重载和函数覆盖之间有什么区别？

1.  什么是类型特征？我们如何实现类型特征？

1.  在`ch4_5_class_template_implicit_inst_B.cpp`文件中，我们说隐式实例化生成了`X<int>`类，然后创建了`xi`对象并生成了`X<int>::f()`函数，但没有生成`X<int>::g()`。如何验证`X<int>::g()`没有生成？

1.  使用模板元编程解决*f(x,n) = x^n*的问题，其中*n*是一个 const，*x*是一个变量。

1.  将`ch4_17_loop_unrolling_metaprogramming.cpp`扩展到 n=10,100,10³,10⁴,10⁶，直到达到系统内存限制。比较编译时间、目标文件大小和运行 CPU 时间。

# Further reading

正如本章中所引用的，查看以下来源，以了解本章涵盖的更多内容：

+   Milner, R., Morris, L., Newey, M. (1975). *A Logic for Computable Functions with Reflexive and Polymorphic Types.* Proceedings of the Conference on Proving and Improving Programs.

+   [`www.research.ed.ac.uk/portal/en/publications/a-logic-for-computable-functions-with-reflexive-and-polymorphic-types(9a69331e-b562-4061-8882-2a89a3c473bb).html`](https://www.research.ed.ac.uk/portal/en/publications/a-logic-for-computable-functions-with-reflexive-and-polymorphic-types(9a69331e-b562-4061-8882-2a89a3c473bb).html)

+   *Curtis, Dorothy (2009-11-06). CLU home page.*Programming Methodology Group, Computer Science and Artificial Intelligence Laboratory. Massachusetts Institute of Technology.

+   [`www.pmg.csail.mit.edu/CLU.html`](http://www.pmg.csail.mit.edu/CLU.html)

+   *Technical Corrigendum for Ada 2012*, published by ISO. Ada Resource Association. 2016-01-29.

+   https://www.adaic.org/2016/01/technical-corrigendum-for-ada-2012-published-by-iso/

+   B. Stroustrup, *C++.*

+   [`dl.acm.org/doi/10.5555/1074100.1074189`](https://dl.acm.org/doi/10.5555/1074100.1074189)

+   *S. Meyers, Effective C++ 55 Specific Ways to Improve Your Programs and Designs (3rd Edition), Chapter 7.*

+   [`www.oreilly.com/library/view/effective-c-55/0321334876/`](https://www.oreilly.com/library/view/effective-c-55/0321334876/)

+   D. Gregor and J. Järvi (February 2008). *Variadic Templates for C++0x.*Journal of Object Technology. pp. 31–51

[`www.jot.fm/issues/issue_2008_02/article2.pdf`](http://www.jot.fm/issues/issue_2008_02/article2.pdf)

+   [`www.boost.org/`](https://www.boost.org/) for type traits, unit testing etc.

+   [`www.ibm.com/support/knowledgecenter/ssw_ibm_i_72/rzarg/templates.htm`](https://www.ibm.com/support/knowledgecenter/ssw_ibm_i_72/rzarg/templates.htm) for generic templates discussions.

+   [`stackoverflow.com/questions/546669/c-code-analysis-tool`](https://stackoverflow.com/questions/546669/c-code-analysis-tool) for code analysis tools.

+   [`en.cppreference.com`](https://en.cppreference.com) for template explicit instantiations.

+   [`www.cplusplus.com`](http://www.cplusplus.com) for library references and usage examples.

+   [`www.drdobbs.com/cpp/c-type-traits/184404270`](http://www.drdobbs.com/cpp/c-type-traits/184404270) for type-traits.

+   [`accu.org/index.php/journals/424`](https://accu.org/index.php/journals/424) for template metaprogramming.

+   [`en.wikipedia.org/wiki/Template_metaprogramming`](https://en.wikipedia.org/wiki/Template_metaprogramming) 用于模板元编程。

+   K. Czarnecki, U. W. Eisenecker, *Generative Programming: Methods, Tools, and Applications*, 第十章。

+   N. Josuttis; D. Gregor 和 D. Vandevoorde, *C++ Templates: The Complete Guide (2nd Edition)*, Addison-Wesley Professional 2017。
