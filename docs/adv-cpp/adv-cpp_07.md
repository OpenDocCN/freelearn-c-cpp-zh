# 第七章：6. 流和 I/O

## 学习目标

在本章结束时，您将能够：

+   使用标准 I/O 库向文件或控制台写入和读取数据

+   使用内存 I/O 接口格式化和解析数据

+   扩展用户定义类型的标准 I/O 流

+   开发使用多个线程的 I/O 标准库的应用程序

在本章中，我们将使用 I/O 标准库开发灵活且易于维护的应用程序，处理流，学习 I/O 库如何在多线程应用程序中使用，并最终学会使用标准库格式化和解析数据。

## 介绍

在上一章中，我们涵盖了 C++中最具挑战性的主题之一 - 并发性。我们研究了主要的多线程概念，并区分了 C++中的同步、异步和线程执行。我们学习了关于同步、数据危害和竞争条件的关键要点。最后，我们研究了在现代 C++中使用线程。在本章中，我们将深入学习如何处理多线程应用中的 I/O。

本章专注于 C++中的`流`和`I/O`。I/O 是输入和输出操作的一般概念。标准库的这一部分的主要目的是提供关于数据输入和输出的清晰接口。但这并不是唯一的目标。有很多情况下，I/O 可以帮助我们的应用程序。很难想象任何一个应用程序不会将错误或异常情况写入日志文件，以便将其发送给开发团队进行分析。在 GUI 应用程序中，我们总是需要格式化显示的信息或解析用户输入。在复杂和大型应用程序中，我们通常需要记录内部数据结构等。在所有这些情况下，我们使用`标准库`的 I/O 部分。

我们将从对标准库的输入/输出部分进行简要介绍开始本章。我们将学习有关 I/O 的概念并探索其主要概念和术语。然后，我们将考虑默认支持哪些类型以及如何将流扩展到用户定义的类型。接下来，我们将研究 I/O 库的结构，并检查可供我们使用的头文件和类。最后，我们将调查如何处理流，读写文件，创建具有输入和输出操作的多线程应用程序，并格式化和解析文本数据。

本章将以一个具有挑战性和令人兴奋的活动结束，我们将改进上一章的`艺术画廊模拟器`项目，并创建一个健壮、清晰、多线程且易于使用的`日志记录器`。我们将开发一个具有清晰接口的类，可以从项目中的任何地方访问。接下来，我们将使其适应多个线程的工作。最后，我们将把我们的健壮的日志记录器整合到艺术画廊模拟器项目中。

让我们从查看 C++标准库的 I/O 部分开始，了解这组工具为我们提供了哪些机会。

### 审查标准库的 I/O 部分

在计算机科学中，I/O 是指程序、设备、计算机等之间的通信。在 C++中，我们使用标准输入和标准输出术语来描述 I/O 过程。标准输入意味着传输到程序中的数据流。要获取这些数据，程序应执行读取操作。标准输出意味着从程序传输到外部设备（如文件、显示器、套接字、打印机等）的数据流。要输出这些数据，程序应执行写操作。标准输入和输出流是从主进程继承的，并且对所有子线程都是通用的。看一下下面的图表，以更好地理解所考虑的术语：

![](img/C14583_06_01.jpg)

###### 图 6.1：设备之间的 I/O 通信

在 C++标准库中，大多数 I/O 类都是通用的类模板。它们在逻辑上分为两类——抽象和实现。我们已经熟悉了抽象类，并且知道我们可以在不重新编译代码的情况下用它们来实现不同的目的。I/O 库也是如此。在这里，我们有六个抽象类，它们是 C++中 I/O 操作的基础。我们不会深入研究这些接口。通常，我们使用更高级的类来进行操作，只有在需要实现自己的派生类时才会使用它们。

**ios_base**抽象类负责管理流状态标志、格式化标志、回调和私有存储。**basic_streambuf**抽象类提供了缓冲输入或输出操作的接口，并提供了对输入源的访问，例如文件、套接字，或输出的接收端，例如字符串或向量。**basic_ios**抽象类实现了与**basic_streambuf**接口派生类的工作设施。**basic_ostream**、**basic_istream**和**basic_iostream**抽象类是**basic_streambuf**接口派生类的包装器，并分别提供了高级的输入/输出接口。让我们简要地考虑它们及其关系，这些关系显示在下面的类图中。您可以看到，除了**ios_base**之外，它们都是模板类。在每个类的名称下面，您可以找到定义该类的文件名：

#### 注

在 UML 符号中，我们使用`<<interface>>`关键字来显示类是一个抽象类。

![图 6.2：I/O 抽象接口的类图](img/C14583_06_02.jpg)

###### 图 6.2：I/O 抽象接口的类图

实现类别在逻辑上分为以下几类：**文件 I/O**，**字符串 I/O**，**同步 I/O**，**I/O 操纵器**和预定义的标准流对象。所有这些类都是从上述抽象类派生而来的。让我们在接下来的部分详细考虑每一个。

### 预定义的标准流对象

我们将从已经熟悉的`<iostream>`头文件中的`std::cout`类开始认识 I/O 标准库。我们用它来将数据输出到终端。您可能也知道`std::cin`类用于读取用户输入——但并不是每个人都知道`std::cout`和`std::cin`是预定义的标准流对象，用于格式化输入和输出到终端。`<iostream>`头文件还包含`std::cerr`和`std::clog`流对象，用于记录错误。通常情况下，它们也有带有前缀“`w`”的宽字符的类似物：`wcout`，`wcin`，`wcerr`和`wclog`。所有这些对象都会在系统启动时自动创建和初始化。虽然从多个线程中使用这些对象是安全的，但输出可能会混合。让我们回顾一下如何使用它们。由于它们只对内置类型进行了重载，我们应该为用户定义的类型编写自己的重载。

`std::cout`流对象经常与`std::endl`操纵器一起使用。它在输出序列中插入换行符并刷新它。这里有一个使用它们的例子：

```cpp
std::string name("Marilyn Monroe");
int age = 18;
std::cout << "Name: " << name << ", age: " << age << std::endl;
```

最初，`std::cin`对象逐个读取所有输入字符序列。但它对于内置类型有重载，并且可以读取诸如`数字`、`字符串`、`字符`等值。在读取字符串时有一个小技巧；`std::cin`会读取字符串直到下一个空格或换行符。因此，如果您需要读取一个字符串，您必须在循环中逐个单词地读取它，或者使用`std::getline()`函数，该函数将`std::cin`对象作为第一个参数，目标字符串作为第二个参数。

#### 注

`std::cin`流对象的右移操作符`>>`只从一行中读取一个单词。使用`std::getline(std::cin, str)`来读取整行。

这是使用`std::cin`与不同类型的示例：

```cpp
std::string name;
std::string sex;
int age;
std::cout << "Enter your name: " << std::endl;
std::getline(std::cin, name);
std::cout << "Enter your age: " << std::endl;
std::cin >> age;
std::cout << "Enter your sex (male, female):" << std::endl;
std::cin >> sex;
std::cout << "Your name is " << name << ", your age is " << age << ", your sex is " << sex << std::endl;
```

如您所见，在这里，我们使用`std::getline()`函数读取名称，因为用户可以输入两个或三个单词。我们还读取年龄，然后使用右移操作符`>>`读取性别，因为我们只需要读取一个单词。然后打印读取的数据，以确保一切顺利。

**std::cerr**和**std::clog**流对象在一个方面有所不同-**std::cerr**立即刷新输出序列，而**std::clog**对其进行缓冲，并且仅在缓冲区满时刷新。在使用时，它们与**std::cout**非常相似。唯一的区别是**std::cerr**和**std::clog**的消息（在大多数 IDE 中）是红色的。

在下面的屏幕截图中，您可以看到这些流对象的输出：

![](img/C14583_06_03.jpg)

###### 图 6.3：来自 std::cerr 和 std::clog 流对象的输出

现在，让我们进行一项练习，巩固我们所学的一切。

### 练习 1：重载左移操作符<<，用于用户定义的类型

在这个练习中，我们将编写一个非常有用的代码部分，您可以在任何地方使用它来输出用户定义的类型。首先，我们将创建一个名为`Track`的类，表示音乐曲目。它将具有以下私有成员：`name`，`singer`，`length`和`date`。然后，我们将重载这个类的左移操作符`<<`。接下来，我们将创建这个类的一个实例，并使用`std::cout`流对象输出它。

执行以下步骤来执行此练习：

1.  包括所需的头文件：`<iostream>` 用于在控制台输出和 `<string>` 用于字符串支持：

```cpp
#include <iostream>
#include <string>
```

1.  声明`Track`类，并添加私有部分变量以保存有关`track`的信息，即`m_Name`，`m_Singer`，`m_Date`和`m_LengthInSeconds`。在公共部分，添加一个带参数的构造函数，初始化所有私有变量。还要为所有类成员添加`public`部分的 getter：

```cpp
class Track
{
public:
     Track(const std::string& name,
           const std::string& singer,
           const std::string& date,
           const unsigned int& lengthInSeconds)
           : m_Name(name)
           , m_Singer(singer)
           , m_Date(date)
           , m_LengthInSeconds(lengthInSeconds)
{
}
     std::string getName() const { return m_Name; }
     std::string getSinger() const { return m_Singer; }
     std::string getDate() const { return m_Date; }
     unsigned int getLength() const { return m_LengthInSeconds; }
private:
     std::string m_Name;
     std::string m_Singer;
     std::string m_Date;
     unsigned int m_LengthInSeconds;
};
```

1.  现在是练习中最困难的部分：为`Track`类型编写重载函数。这是一个具有两个类型参数`charT`和`Traits`的`template`函数：

```cpp
template <typename charT, typename Traits>
```

1.  我们将此函数设置为内联函数，以便让编译器知道我们希望它对此函数进行优化。此函数的返回类型是对`std::basic_ostream<charT, Traits>`类的引用。此函数的名称是`operator <<`。此函数接受两个参数：第一个是对`std::basic_ostream<charT, Traits>`类的引用，第二个是`Track`变量的副本。完整的函数声明如下：

```cpp
template <typename charT, typename Traits>
inline std::basic_ostream<charT, Traits>&
operator<<(std::basic_ostream<charT, Traits>& os, Track trackItem);
```

1.  现在，添加函数定义。使用`os`变量，就像我们使用`std::cout`对象一样，并根据需要格式化输出。然后，从函数返回`os`变量。重载操作符`<<`的完整代码如下：

```cpp
template <typename charT, typename Traits>
inline std::basic_ostream<charT, Traits>&
operator<<(std::basic_ostream<charT, Traits>& os, Track trackItem)
{
      os << "Track information: ["
         << "Name: " << trackItem.getName()
         << ", Singer: " << trackItem.getSinger()
         << ", Date of creation: " << trackItem.getDate()
         << ", Length in seconds: " << trackItem.getLength()
         << "]";
      return os;
}
```

1.  现在，进入`main`函数，并创建并初始化`Track`类型的实例`track_001`。最后，使用`std::cout`打印`track_001`的值：

```cpp
int main()
{
     Track track_001("Summer night city",
                     "ABBA",
                     "1979",
                      213);
     std::cout << track_001 << std::endl;
     return 0;
}
```

1.  编译并执行应用程序。运行它。您将获得以下输出：

![](img/C14583_06_04.jpg)

###### 图 6.4：执行练习 1 的结果

干得好。在这里，我们考虑了使用预定义的标准流对象，并学习了如何为用户定义的类型编写我们自己的重载移位操作符。让我们继续并研究使用 C++标准 IO 库读写文件的部分。

### 文件 I/O 实现类

文件流管理对文件的输入和输出。它们提供了一个接口，实现了输入操作的`basic_ifstream`，输出操作的`basic_ofstream`，同时输入和输出操作的`basic_fstream`，以及用于实现原始文件设备的`basic_filebuf`。它们都在`<fstream>`头文件中定义。标准库还提供了 char 和`wchar_t`类型的 typedefs，即`ifstream`，`fstream`和`ofstream`，以及带有`w`前缀的相同名称，用于宽字符。

我们可以以两种方式创建文件流。第一种方式是在一行中执行此操作，即通过将文件名传递给构造函数来打开文件并将流连接到文件：

```cpp
std::ofstream outFile(filename);
std::ifstream outFile(filename);
std::fstream outFile(filename);
```

另一种方法是创建一个对象，然后调用`open()`函数：

```cpp
std::ofstream outFile;
outFile.open(filename);
```

#### 注意

IO 流具有 bool 变量：**goodbit**，**eofbit**，**failbit**和**badbit**。它们用于在每次操作后检查流的状态，并指示流上发生了哪种错误。

在对象创建后，我们可以通过检查`failbit`或检查与打开文件相关联的流来检查流状态。要检查`failbit`，请在`file`流上调用`fail()`函数：

```cpp
if (outFile.fail())
{
    std::cerr << filename << " file couldn't be opened"<< std::endl;
}
```

要检查流是否与打开的文件相关联，请调用`is_open()`函数：

```cpp
if (!outFile.is_open())
{
    std::cerr << filename << " file couldn't be opened"<< std::endl;
}
```

输入、输出和双向文件流也可以使用标志以不同模式打开。它们声明在`ios_base`命名空间中。除了`ios_base::in`和`ios_base::out`标志之外，我们还有`ios_base::ate`，`ios_base::app`，`ios_base::trunc`和`ios_base::binary`标志。`ios_base::trunc`标志会删除文件的内容。`ios_base::app`标志总是将输出写入文件的末尾。即使您决定更改文件中的位置，也无法这样做。`ios_base::ate`标志将文件描述符的位置设置为文件末尾，但允许您稍后修改位置。最后，`ios_base::binary`标志抑制数据的任何格式化，以便以“原始”格式读取或写入。让我们考虑所有可能的打开模式组合。

默认情况下，`std::ofstream`以`ios_base::out`模式打开，`std::ifstream`以`ios_base::in`模式打开，`std::fstream`以`ios_base::in|ios_base::out`模式打开。`ios_base::out|ios_base::trunc`模式会在文件不存在时创建文件，或者删除现有文件的所有内容。`ios_base::out|ios_base::app`模式会在文件不存在时创建文件，或者打开现有文件，并允许您仅在文件末尾写入。上述两种模式都可以与`ios_base::in`标志结合使用，因此文件将同时以读取和写入模式打开。

以下是使用上述模式打开文件的示例：

```cpp
std::ofstream outFile(filename, std::ios_base::out|std::ios_base::trunc);
```

您还可以执行以下操作：

```cpp
std::ofstream outFile;
outFile.open(filename, std::ios_base::out|std::ios_base::trunc);
```

在我们以所需模式打开文件流之后，我们可以开始读取或写入文件。文件流允许我们更改文件中的位置。让我们考虑如何做到这一点。要获取当前文件的位置，我们可以在`ios_base::out`模式中调用`tellp()`函数，在`ios_base::in`模式中调用`tellg()`函数。稍后可以使用它，以便在需要时返回到此位置。我们还可以使用`seekp()`函数在`ios_base::out`模式中和`seekg()`函数在`ios_base::in`模式中找到文件中的确切位置。它接受两个参数：要移动的字符数以及应从哪个文件位置计数。允许三种位置类型`seek: std::ios_base::beg`，即文件的开头，`std::ios_base::end`，即文件的末尾，以及`std::ios_base::cur`，即当前位置。以下是调用`seekp()`函数的示例：

```cpp
outFile.seekp(-5, std::ios_base::end);
```

如您所见，我们要求将当前文件的位置设置为文件末尾的第五个字符。

要写入文件，我们可以使用重载的左移操作符`<<`进行一般格式化输出，使用`put()`函数写入单个字符，或使用`write()`函数写入一块字符。使用左移操作符是将数据写入文件的最方便的方法，因为可以将任何内置类型作为参数传递：

```cpp
outFile << "This is line No " << 1 << std::endl;
```

`put()`和`write()`函数只能用于字符值。

要从文件中读取，我们可以使用重载的右移操作符`>>`，或使用一组用于读取字符的函数，如`read()`，`get()`和`getline()`。右移操作符已经为所有内置类型进行了重载，我们可以像这样使用它：

```cpp
std::ifstream inFile(filename);		
std::string str;
int num;
float floatNum;
// for data: "book 3 24.5"
inFile >> str >> num >> floatNum;
```

最后，当执行离开可见范围时，文件流将被关闭，因此我们不需要执行任何额外的操作来关闭文件。

#### 注意

在从文件中读取数据时要注意。右移操作符`>>`只会读取字符串直到空格或换行符为止。要读取完整的字符串，可以使用循环或将每个单词读入单独的变量，就像我们在*练习 1*中所做的那样，*重载左移操作符<<，用于用户定义的类型*。

现在，让我们练习使用 C++ IO 标准库将数据读取和写入文件。

### 练习 2：将用户定义的数据类型读写到文件

在这个练习中，我们将为书店编写一段代码。我们需要将有关书籍价格的信息存储在文件中，然后在需要时从文件中读取该信息。为了实现这一点，我们将创建一个代表具有名称、作者、出版年份和价格的书的类。接下来，我们将创建该类的实例并将其写入文件。稍后，我们将从文件中读取有关书籍的信息到书籍类的实例中。执行以下步骤来完成这个练习：

1.  包括所需的头文件：`<iostream>`用于输出到控制台，`<string>`用于字符串支持，`<fstream>`用于 I/O 文件库支持：

```cpp
#include <fstream>
#include <iostream>
#include <string>
```

1.  实现`Book`类，它代表书店中的书。在私有部分，使用不言自明的名称定义四个变量：`m_Name`，`m_Author`，`m_Year`和`m_Price`。在公共部分，定义带参数的构造函数，初始化所有类成员。此外，在`public`部分，为所有类成员定义 getter：

```cpp
class Book
{
public:
      Book(const std::string& name,
           const std::string& author,
           const int year,
           const float price)
     : m_Name(name)
     , m_Author(author)
     , m_Year(year)
     , m_Price(price) {}
     std::string getName() const { return m_Name; }
     std::string getAuthor() const { return m_Author; }
     int getYear() const { return m_Year; }
     float getPrice() const { return m_Price; }
private:
     std::string m_Name;
     std::string m_Author;
     int m_Year;
     float m_Price;
};
```

1.  进入`main`函数并声明`pricesFile`变量，该变量保存文件名：

```cpp
std::string pricesFile("prices.txt");
```

1.  接下来，创建`book`类的实例，并用`book name`，`author name`，`year`和`price`进行初始化：

```cpp
Book book_001("Brave", "Olena Lizina", 2017, 33.57);
```

1.  将此类实例写入文件。创建`std::ofstream`类的实例。使用`pricesFile`变量名打开我们的文件。检查流是否成功打开，如果没有，则打印错误消息：

```cpp
std::ofstream outFile(pricesFile);
if (outFile.fail())
{
      std::cerr << "Failed to open file " << pricesFile << std::endl;
      return 1;
}
```

1.  然后，使用 getter 将有关`book_001`书籍的所有信息写入文件，每个项目之间用空格分隔，并在末尾加上换行符:

```cpp
outFile << book_001.getName() << " "
        << book_001.getAuthor() << " "
        << book_001.getYear() << " "
        << book_001.getPrice() << std::endl;
```

1.  编译并执行应用程序。现在，转到项目文件夹，并找到'**prices.txt**'文件的位置。在下面的屏幕截图中，您可以看到在项目目录中创建的文件的位置:![](img/C14583_06_05.jpg)

###### 图 6.5：创建文件的位置

1.  在**记事本**中打开它。在下面的屏幕截图中，您可以看到文件的输出是什么样子的:![](img/C14583_06_06.jpg)

###### 图 6.6：将用户定义的类型输出到文件的结果

1.  现在，让我们将这些数据读取到变量中。创建`std::ifstream`类的实例。打开名为`pricesFile`的文件。检查流是否成功打开，如果没有，则打印错误消息：

```cpp
std::ifstream inFile(pricesFile);
if (inFile.fail())
{
     std::cerr << "Failed to open file " << pricesFile << std::endl;
     return 1;
}
```

1.  创建将用于从文件输入的本地变量，即`name`，`authorName`，`authorSurname`，`year`和`price`。它们的名称不言自明：

```cpp
std::string name;
std::string authorName;
std::string authorSurname;
int year;
float price;
```

1.  现在，按照文件中的顺序将数据读入变量中：

```cpp
inFile >> name >> authorName >> authorSurname >> year >> price;
```

1.  创建一个名为`book_002`的`Book`实例，并用这些读取的值进行初始化：

```cpp
Book book_002(name, std::string(authorName + " " + authorSurname), year, price);
```

1.  要检查读取操作是否成功执行，请将`book_002`变量打印到控制台：

```cpp
std::cout  << "Book name: " << book_002.getName() << std::endl
           << "Author name: " << book_002.getAuthor() << std::endl
           << "Year: " << book_002.getYear() << std::endl
           << "Price: " << book_002.getPrice() << std::endl;
```

1.  再次编译和执行应用程序。在控制台中，您将看到以下输出：

![](img/C14583_06_07.jpg)

###### 图 6.7：执行练习 2 的结果

正如您所看到的，我们从文件中写入和读取了自定义格式的数据，没有任何困难。我们创建了自己的自定义类型，使用`std::ofstream`类将其写入文件，并检查一切是否都写入成功。然后，我们使用`std::ifstream`类从文件中读取这些数据到我们的自定义变量，将其输出到控制台，并确保一切都被正确读取。通过这样做，我们学会了如何使用 I/O 标准库向文件读写数据。现在，让我们继续学习 I/O 库的内存部分。

### 字符串 I/O 实现

I/O 标准库允许输入和输出 - 不仅可以输出到文件等设备，还可以输出到内存，特别是`std::string`对象。在这种情况下，字符串可以作为输入操作的源，也可以作为输出操作的接收器。在`<sstream>`头文件中，声明了管理输入和输出到字符串的流类。它们，就像文件流一样，还提供了一个实现 RAII 的接口 - 字符串在流创建时打开以供读取或写入，并在销毁时关闭。它们在标准库中由以下类表示：`basic_stringbuf`，它实现了原始字符串接口，`basic_istringstream`用于输入操作，`basic_ostringstream`用于输出操作，`basic_stringstream`用于输入和输出操作。标准库还为`char`和`wchar_t`类型提供了 typedefs：`istringstream`，`ostringstream`和`stringstream`以及带有宽字符的相同名称的前缀为"w"的名称。

要创建`std::istringstream`类的对象，我们应该将初始化字符串作为构造函数参数传递或者稍后使用`str()`函数设置它：

```cpp
std::string track("ABBA 1967 Vule");
std::istringstream iss(track);
```

或者，我们可以这样做：

```cpp
std::string track("ABBA 1967 Vule");
std::istringstream iss;
iss.str(track);
```

接下来，要从流中读取值，请使用重定向运算符`>>`，它对所有内置类型进行了重载：

```cpp
std::string group;
std::string name;
int year;
iss >> group >> year >> name;
```

要创建`std::ostringstream`类的对象，我们只需声明其类型的变量：

```cpp
std::ostringstream oss;
```

接下来，要将数据写入字符串，请使用重定向运算符`<<`，它对所有内置类型进行了重载：

```cpp
std::string group("ABBA");
std::string name("Vule");
int year = 1967;
oss << group << std::endl
    << name << std::endl
    << year << std::endl;
```

要获取结果字符串，请使用`str()`函数：

```cpp
std::cout << oss.str();
```

`std::stringstream`对象是双向的，因此它既有默认构造函数，也有接受字符串的构造函数。我们可以通过声明这种类型的变量来创建默认的`std::stringstream`对象，然后用它进行读写：

```cpp
std::stringstream ss;
ss << "45";
int count;
ss >> count;
```

此外，我们可以使用带有字符串参数的构造函数创建`std::stringstream`。然后，我们可以像往常一样使用它进行读写：

```cpp
std::string employee("Alex Ismailow 26");
std::stringstream ss(employee);
```

或者，我们可以创建一个默认的`std::stringstream`对象，并通过使用`str()`函数设置一个字符串来初始化它：

```cpp
std::string employee("Charlz Buttler 26");
std::stringstream ss;
ss.str(employee);
```

接下来，我们可以使用 ss 对象进行读写：

```cpp
std::string name;
std::string surname;
int age;
ss >> name >> surname >> age;
```

我们还可以为这些类型的流应用打开模式。它们的功能类似于文件流，但有一点不同。在使用字符串流时，`ios_base::binary`是无关紧要的，`ios_base::trunc`会被忽略。因此，我们可以以四种模式打开任何字符串流：`ios_base::app`，`ios_base::ate`，`ios_base::in/ios_base::out`。

现在，让我们练习使用 C++ IO 标准库向字符串读写数据。

### 练习 3：创建一个替换字符串中单词的函数

在这个练习中，我们将实现一个函数，该函数解析给定的字符串，并用其他单词替换给定的单词。要完成这个练习，我们创建一个可调用类，它接受三个参数：原始字符串，要替换的单词和将用于替换的单词。结果应该返回新的字符串。执行以下步骤来完成这个练习：

1.  包括所需的头文件：`<iostream>`用于输出到终端，`<sstream>`用于 I/O 字符串支持：

```cpp
#include <sstream>
#include <iostream>
```

1.  实现名为`Replacer`的可调用类。它只有一个函数 - 重载的括号运算符，即()，它返回一个字符串，并接受三个参数：原始字符串、要替换的单词和用于替换的单词。函数声明如下：

```cpp
std::string operator()(const std::string& originalString,
                       const std::string& wordToBeReplaced,
                       const std::string& wordReplaceBy);
```

1.  接下来，创建`istringstream`对象，即`iss`，并将`originalString`变量设置为输入源：

```cpp
std::istringstream iss(originalString);
```

1.  创建`ostringstream`对象，即`oss`，它将保存转换后的字符串：

```cpp
std::ostringstream oss;
```

1.  然后，在循环中，当可能有输入时，执行对单词变量的读取。检查这个单词是否等于`wordToBeReplaced`变量。如果是，用`wordReplaceBy`变量替换它，并写入`oss`流。如果它们不相等，将原始单词写入`oss`流。在每个单词后，添加一个空格字符，因为`iss`流会截断它们。最后，返回结果。完整的类如下：

```cpp
class Replacer
{
public:
      std::string operator()(const std::string& originalString,
                             const std::string& wordToBeReplaced,
                             const std::string& wordReplaceBy)
     {
           std::istringstream iss(originalString);
           std::ostringstream oss;
           std::string word;
           while (iss >> word)
           {
                if (0 == word.compare(wordToBeReplaced))
                {
                     oss << wordReplaceBy << " ";
                }
                else
                {
                     oss << word << " ";
                }
           }
           return oss.str();
     }
};
```

1.  进入`main`函数。创建一个名为 worker 的`Replacer`类的实例。定义`foodList`变量，并将其初始化为包含食物列表的字符串；一些项目应该重复。定义`changedList`字符串变量，并将其初始化为`worker()`函数的返回值。使用`std::cout`在终端上显示结果：

```cpp
int main()
{
      Replacer worker;
      std::string foodList("coffee tomatoes coffee cucumbers sugar");
      std::string changedList(worker(foodList, "coffee", "chocolate"));
      std::cout << changedList;
      return 0;
}
```

1.  编译、构建并运行练习。结果将如下所示：

![图 6.8：执行练习 3 的结果](img/C14583_06_08.jpg)

###### 图 6.8：执行练习 3 的结果

干得好！在这里，我们学会了如何使用字符串流来格式化输入和输出。我们创建了一个可以轻松替换句子中单词的应用程序，加强了我们的知识，现在我们准备学习 I/O 操作符，以便我们可以提高我们处理线程的技能。

### I/O 操作符

到目前为止，我们已经学习了使用流进行简单的输入和输出，但在许多情况下这是不够的。对于更复杂的 I/O 数据格式化，标准库有一个大量的 I/O 操作符。它们是为了与移位操作符（<<和>>）一起工作而开发的函数，用于控制流的行为。I/O 操作符分为两种类型 - 一种是无需参数调用的，另一种是需要参数的。其中一些既适用于输入又适用于输出。让我们简要地考虑它们的含义和用法。

### 用于更改流的数字基数的 I/O 操作符

在`<ios>`头文件中，声明了用于更改流的数字基数的函数：`std::dec`、`std::hex`和`std::oct`。它们是无需参数调用的，并将流的数字基数分别设置为十进制、十六进制和八进制。在`<iomanip>`头文件中，声明了`std::setbase`函数，它是用以下参数调用的：8、10 和 16。它们是可互换的，并且适用于输入和输出操作。

在`<ios>`头文件中，还有`std::showbase`和`std::noshowbase`函数，它们控制显示流的数字基数。它们只影响十六进制和八进制的整数输出，除了零值和货币输入和输出操作。让我们完成一个练习，学习如何在实践中使用它们。

### 练习 4：以不同的数字基数显示输入的数字

在这个练习中，我们将开发一个应用程序，在无限循环中，要求用户以十进制、十六进制或八进制中的一种输入一个整数。读取输入后，将以其他数字表示形式显示这个整数。要完成这个练习，完成以下步骤：

1.  包括`<iostream>`头文件以支持流。声明名为`BASE`的枚举并定义三个值：`DECIMAL`、`OCTAL`和`HEXADECIMAL`：

```cpp
#include <iostream>
enum BASE
{
      DECIMAL,
      OCTAL,
      HEXADECIMAL
};
```

1.  声明一个名为`displayInBases`的函数，它接受两个参数 - 整数和基数。接下来，定义 switch 语句，测试接收到的数字基数，并以其他两种数字表示显示给定的整数：

```cpp
void displayInBases(const int number, const BASE numberBase)
{
  switch(numberBase)
  {
  case DECIMAL:
    std::cout << "Your input in octal with base: "
          << std::showbase << std::oct << number
          << ", without base: " 
          << std::noshowbase << std::oct << number << std::endl;
    std::cout << "Your input in hexadecimal with base: "
          << std::showbase << std::hex << number
          << ", without base: " 
          << std::noshowbase << std::hex << number << std::endl;
    break;
  case OCTAL:
    std::cout << "Your input in hexadecimal with base: "
          << std::showbase << std::hex << number
          << ", without base: " 
          << std::noshowbase << std::hex << number << std::endl;
    std::cout << "Your input in decimal with base: "
          << std::showbase << std::dec << number
          << ", without base: " 
          << std::noshowbase << std::dec << number << std::endl;
    break;
  case HEXADECIMAL:
    std::cout << "Your input in octal with base: "
          << std::showbase << std::oct << number
          << ", without base: " 
          << std::noshowbase << std::oct << number << std::endl;
    std::cout << "Your input in decimal with base: "
          << std::showbase << std::dec << number
          << ", without base: " 
          << std::noshowbase << std::dec << number << std::endl;
    break;
  }
}
```

1.  进入`main`函数并定义将用于读取用户输入的整数变量：

```cpp
int integer; 
```

1.  创建一个无限循环。在循环内部，要求用户输入一个十进制值。将输入读取为十进制整数。将其传递给`displayInBases`函数。接下来，要求用户输入一个十六进制值。将输入读取为十六进制整数。将其传递给`displayInBases`函数。最后，要求用户输入一个八进制值。将输入读取为八进制整数。将其传递给`displayInBases`函数：

```cpp
int main(int argc, char **argv)
{
  int integer;
  while(true)
  {
    std::cout << "Enter the decimal value: ";
    std::cin >> std::dec >> integer;
    displayInBases(integer, BASE::DECIMAL);
    std::cout << "Enter the hexadecimal value: ";
    std::cin >> std::hex >> integer;
    displayInBases(integer, BASE::HEXADECIMAL);
    std::cout << "Enter the octal value: ";
    std::cin >> std::oct >> integer;
    displayInBases(integer, BASE::OCTAL);
  }
  return 0;
}
```

1.  构建并运行应用程序。跟随输出并输入，例如，在不同的数字表示中输入 12。输出应该如下所示：![图 6.9：执行练习 4，第 1 部分的结果](img/C14583_06_09.jpg)

###### 图 6.9：执行练习 4，第 1 部分的结果

1.  现在，让我们将`std::dec`、`std::oct`和`std::hex`在`std::setbase()`函数中更改，以检查输出是否相同。首先，添加`<iomanip>`头文件以支持`std::setbase()`。接下来，在主函数中的循环中，将`std::dec`替换为`std::setbase(10)`，将`std::hex`替换为`std::setbase(16)`，将`std::oct`替换为`std::setbase(8)`：

```cpp
int main(int argc, char **argv)
{
  int integer;
  while(true)
  {
    std::cout << "Enter the decimal value: ";
    std::cin >> std::setbase(10) >> integer;
    displayInBases(integer, BASE::DECIMAL);
    std::cout << "Enter the hexadecimal value: ";
    std::cin >> std::setbase(16) >> integer;
    displayInBases(integer, BASE::HEXADECIMAL);
    std::cout << "Enter the octal value: ";
    std::cin >> std::setbase(8) >> integer;
    displayInBases(integer, BASE::OCTAL);
  }
  return 0;
}
```

1.  再次构建并运行应用程序。跟随输出并在不同的数字表示中输入相同的整数（12）。输出应该如下所示：

![图 6.10：执行练习 4，第 2 部分的结果](img/C14583_06_10.jpg)

###### 图 6.10：执行练习 4，第 2 部分的结果

现在，比较一下结果。如您所见，输出是相同的。通过这样做，我们确保这些函数是可以互换的。

### 浮点格式的 I/O 操作符

在`<ios>`头文件中，声明了用于更改浮点数位格式的函数：`std::fixed`、`std::scientific`、`std::hexfloat`和`std::defaultfloat`。它们在没有参数的情况下被调用，并将`floatfield`分别设置为固定、科学、固定和科学以及默认值。还有`std::showpoint`和`std::noshowpoint`函数，用于控制显示浮点数位。它们只影响输出。`std::noshowpoint`函数只影响没有小数部分的浮点数位。

在`<iomanip>`头文件中，声明了一个`std::setprecision`函数，它以表示精度的数字调用。当小数点右侧的数字被舍弃时，结果会四舍五入。如果数字太大而无法以正常方式表示，则会忽略精度规范，并以更方便的方式显示数字。您只需要设置一次精度，并且只在需要另一种精度时更改它。当您选择用于存储浮点变量的数据类型时，您应该注意一些技巧。在 C++中，有三种数据类型可以表示浮点值：float、double 和 long double。

浮点数通常是 4 个字节，双精度是 8 个字节，长双精度是 8、12 或 16 个字节。因此，每种类型的精度都是有限的。浮点类型最多可以容纳 6-9 个有效数字，双精度类型最多可以容纳 15-18 个有效数字，长双精度类型最多可以容纳 33-36 个有效数字。如果您希望比较它们之间的差异，请查看以下表格：

![图 6.11：浮点类型的比较表](img/C14583_06_11.jpg)

###### 图 6.11：浮点类型的比较表

#### 注意

当您需要超过六个有效数字的精度时，请优先选择 double，否则您将得到意外的结果。

让我们完成一个练习，学习如何在实践中使用它们。

### 练习 5：以不同格式显示输入的浮点数

在这个练习中，我们将编写一个应用程序，在无限循环中要求用户输入一个浮点数。在读取输入后，它以不同的格式类型显示这个数字。要完成这个练习，完成以下步骤：

1.  包括`<iostream>`头文件以支持流和`<iomanip>`以支持`std::setprecision`：

```cpp
#include <iostream>
#include <iomanip>
```

1.  接下来，声明一个模板`formattingPrint`函数，它有一个名为`FloatingPoint`的模板参数，并接受一个此类型的参数变量。接下来，通过调用`std::cout`对象中的`precision()`函数，将先前的精度存储在一个 auto 变量中。然后，在终端中以不同的格式显示给定的数字：带小数点，不带小数点，以及固定、科学、十六进制浮点和默认浮点格式。接下来，在 for 循环中，从 0 到 22，显示给定的数字的精度和循环计数器的大小。循环退出后，使用我们之前存储的值重新设置精度：

```cpp
template< typename FloatingPoint >
void formattingPrint(const FloatingPoint number)
{
     auto precision = std::cout.precision();
     std::cout << "Default formatting with point: "
               << std::showpoint << number << std::endl
               << "Default formatting without point: "
               << std::noshowpoint << number << std::endl
               << "Fixed formatting: "
               << std::fixed << number << std::endl
               << "Scientific formatting: "
               << std::scientific << number << std::endl
               << "Hexfloat formatting: "
               << std::hexfloat << number << std::endl
               << "Defaultfloat formatting: "
               << std::defaultfloat << number << std::endl;
     for (int i = 0; i < 22; i++)
     {
          std::cout << "Precision: " << i 
                    << ", number: " << std::setprecision(i) 
                    << number << std::endl;
     }
     std::cout << std::setprecision(precision);
}
```

1.  输入`main`函数。声明一个名为`floatNum`的`float`变量，一个名为`doubleNum`的双精度变量，以及一个名为`longDoubleNum`的长双精度变量。然后，在无限循环中，要求用户输入一个浮点数，读取输入到`longDoubleNum`，并将其传递给`formattingPrint`函数。接下来，通过使用`longDoubleNum`的值初始化`doubleNum`并将其传递给`formattingPrint`函数。接下来，通过使用`longDoubleNum`的值初始化`floatNum`并将其传递给`formattingPrint`函数：

```cpp
int main(int argc, char **argv)
{
     float floatNum;
     double doubleNum;
     long double longDoubleNum;
     while(true)
     {
          std::cout << "Enter the floating-point digit: ";
          std::cin >> std::setprecision(36) >> longDoubleNum;
          std::cout << "long double output" << std::endl;
          formattingPrint(longDoubleNum);
          doubleNum = longDoubleNum;
          std::cout << "double output" << std::endl;
          formattingPrint(doubleNum);
          floatNum = longDoubleNum;
          std::cout << "float output" << std::endl;
          formattingPrint(floatNum);
     }
     return 0;
}
```

1.  构建并运行应用程序。跟踪输出并输入具有`22`个有效数字的浮点值，例如`0.2222222222222222222222`。我们将得到一个很长的输出。现在，我们需要将其拆分进行分析。这是长双精度值输出的一部分的屏幕截图：

![图 6.12：执行练习 5，第 1 部分的结果](img/C14583_06_12.jpg)

###### 图 6.12：执行练习 5，第 1 部分的结果

我们可以看到，默认情况下，固定和`defaultfloat`格式只输出六个有效数字。使用科学格式化时，值的输出看起来如预期。当我们调用`setprecision(0)`或`setprecision(1)`时，我们期望小数点后不输出任何数字。但对于小于 1 的数字，setprecision 会在小数点后留下一个数字。通过这样做，我们将看到正确的输出直到 21 精度。这意味着在我们的系统上，长双精度的最大精度是 20 个有效数字。现在，让我们分析双精度值的输出：

![图 6.13：执行练习 5，第 2 部分的结果](img/C14583_06_13.jpg)

###### 图 6.13：执行练习 5，第 2 部分的结果

在这里，我们可以看到相同的格式化结果，但精度不同。不准确的输出从精度 17 开始。这意味着，在我们的系统上，双精度的最大精度是 16 个有效数字。现在，让我们分析浮点值的输出：

![图 6.14：执行练习 5，第 3 部分的结果](img/C14583_06_14.jpg)

###### 图 6.14：执行练习 5，第 3 部分的结果

在这里，我们可以看到相同的格式化结果，但精度不同。不准确的输出从精度 8 开始。这意味着，在我们的系统上，浮点的最大精度是 8 个有效数字。不同系统上的结果可能不同。对它们的分析将帮助您选择正确的数据类型用于您的应用程序。

#### 注意

永远不要使用浮点数据类型来表示货币或汇率；你可能会得到错误的结果。

### 布尔格式化的 I/O 操作符

在`<ios>`头文件中，声明了用于更改布尔格式的函数：`std::boolalpha`和`std::noboolalpha`。它们在没有参数的情况下被调用，并允许我们分别以文本或数字方式显示布尔值。它们用于输入和输出操作。让我们考虑一个使用这些 I/O 操作符进行输出操作的例子。我们将布尔值显示为文本和数字：

```cpp
std::cout << "Default formatting of bool variables: "
          << "true: " << true
          << ", false: " << false << std::endl;
std::cout << "Formatting of bool variables with boolalpha flag is set: "
          << std::boolalpha
          << "true: " << true
          << ", false: " << false << std::endl;
std::cout << "Formatting of bool variables with noboolalpha flag is set: "
          << std::noboolalpha
          << "true: " << true
          << ", false: " << false << std::endl;
```

编译并运行此示例后，您将得到以下输出：

```cpp
Default formatting of bool variables: true: 1, false: 0
Formatting of bool variables with boolalpha flag is set: true: true, false: false
Formatting of bool variables with noboolalpha flag is set: true: 1, false: 0
```

如您所见，布尔变量的默认格式是使用`std::noboolalpha`标志执行的。要在输入操作中使用这些函数，我们需要有一个包含 true/false 单词或 0/1 符号的源字符串。输入操作中的`std::boolalpha`和`std::noboolalpha`函数调用如下：

```cpp
bool trueValue, falseValue;
std::istringstream iss("false true");
iss >> std::boolalpha >> falseValue >> trueValue;
std::istringstream iss("0 1");
iss >> std::noboolalpha >> falseValue >> trueValue;
```

如果您输出这些变量，您会看到它们通过读取布尔值正确初始化。

### 用于字段宽度和填充控制的 I/O 操作符

在标准库中，还有一些函数用于通过输出字段的宽度进行操作，当宽度大于输出数据时应该使用哪些字符，以及这些填充字符应该插入在哪个位置。当您想要将输出对齐到左侧或右侧位置，或者当您想要用其他符号替换空格时，这些函数将非常有用。例如，假设您需要在两列中打印价格。如果您使用标准格式，您将得到以下输出：

```cpp
2.33 3.45
2.2 4.55
3.67 3.02
```

这看起来不太好，很难阅读。如果我们应用格式，输出将如下所示：

```cpp
2.33   3.45
2.2     4.55
3.67   3.02
```

这看起来更好。再次，您可能想要检查用于填充空格的字符以及实际插入在数字之间的空格。例如，让我们将填充字符设置为“*”。您将得到以下输出：

```cpp
2.33* 3.45*
2.2** 4.55*
3.67* 3.02*
```

现在，您可以看到空格被星号填充了。既然我们已经考虑了在哪里可以使用格式化宽度和填充输出，那么让我们考虑如何使用 I/O 操作符进行这样的操作。`std::setw`和`std::setfill`函数声明在`<iomanip>`头文件中。`std::setw`以整数值作为参数，并将流的宽度设置为精确的 n 个字符。有几种情况下，宽度将被设置为 0。它们如下：

+   当调用移位操作符与`std::string`或`char`时

+   当调用`std::put_money()`函数时

+   当调用`std::quoted()`函数时

在`<ios>`头文件中，声明了用于更改填充字符应该插入的位置的函数：`std::internal`、`std::left`和`std::right`。它们仅用于输出操作，仅影响整数、浮点和货币值。

现在，让我们考虑一个同时使用它们的例子。让我们输出正数、负数、浮点数和十六进制值，宽度为 10，并用`#`替换填充字符：

```cpp
std::cout << "Internal fill: " << std::endl
          << std::setfill('#')
          << std::internal
          << std::setw(10) << -2.38 << std::endl
          << std::setw(10) << 2.38 << std::endl
          << std::setw(10) << std::hex << std::showbase << 0x4b << std::endl;
std::cout << "Left fill: " << std::endl
          << std::left
          << std::setw(10) << -2.38 << std::endl
          << std::setw(10) << 2.38 << std::endl
          << std::setw(10) << std::hex << std::showbase << 0x4b << std::endl;
std::cout << "Right fill: " << std::endl
          << std::right
          << std::setw(10) << -2.38 << std::endl
          << std::setw(10) << 2.38 << std::endl
          << std::setw(10) << std::hex << std::showbase << 0x4b << std::endl;
```

构建并运行此示例后，您将得到以下输出：

```cpp
Internal fill: 
-#####2.38
######2.38
0x######4b
Left fill: 
-2.38#####
2.38######
0x4b######
Right fill: 
#####-2.38
######2.38
######0x4b
```

### 其他数字格式的 I/O 操作符

如果您需要输出带有“+”符号的正数值，您可以使用`<ios>`头文件中的另一个 I/O 操作符——`std::showpos`函数。相反的意义操作符也存在——`std::noshowpos`函数。它们都会影响输出。它们的使用非常简单。让我们考虑以下例子：

```cpp
std::cout << "Default formatting: " << 13 << " " << 0 << std::endl;
std::cout << "showpos flag is set: " << std::showpos << 13 << " " << 0 << std::endl;
std::cout << "noshowpos flag is set: " << std::noshowpos << 13 << " " << 0 << std::endl;
```

在这里，我们首先使用默认格式输出，然后使用`std::showpos`标志，最后使用`std::noshowpos`标志。如果您构建并运行这个小例子，您会看到，默认情况下，`std::noshowpos`标志被设置。看一下执行结果：

```cpp
Default formatting: 13 0
showpos flag is set: +13 +0
noshowpos flag is set: 13 0
```

您还希望为浮点或十六进制数字输出大写字符，以便您可以使用`<ios>`头文件中的函数：`std::uppercase`和`std::nouppercase`。它们仅适用于输出。让我们考虑一个小例子：

```cpp
std::cout << "12345.0 in uppercase with precision 4: "
          << std::setprecision(4) << std::uppercase << 12345.0 << std::endl;
std::cout << "12345.0 in no uppercase with precision 4: "
          << std::setprecision(4) << std::nouppercase << 12345.0 << std::endl;
std::cout << "0x2a in uppercase: "
          << std::hex << std::showbase << std::uppercase << 0x2a << std::endl;
std::cout << "0x2a in nouppercase: "
          << std::hex << std::showbase << std::nouppercase << 0x2a << std::endl;
```

在这里，我们输出浮点数和十六进制数字，有时使用`std::uppercase`标志，有时不使用。默认情况下，`std::nouppercase`标志被设置。看一下执行的结果：

```cpp
12345.0 in uppercase with precision 4: 1.234E+004
12345.0 in no uppercase with precision 4: 1.234e+004
0x2a in uppercase: 0X2A
0x2a in nouppercase: 0x2a
```

### 用于处理空白的 I/O 操纵器

在标准库中，有用于处理空白的函数。`<istream>`头文件中的`std::ws`函数只适用于输入流，并丢弃前导空白。`<ios>`头文件中的`std::skipws`和`std::noskipws`函数用于控制读取和写入前导空白。它们适用于输入和输出流。当设置了`std::skipws`标志时，流会忽略字符序列前面的空白。默认情况下，`std::skipws`标志被设置。让我们考虑一下使用这些 I/O 操纵器的例子。首先，我们将用默认格式读取输入并输出我们所读取的内容。接下来，我们将清除我们的字符串，并使用`std::noskipws`标志读取数据：

```cpp
std::string name;
std::string surname;
std::istringstream("Peppy Ping") >> name >> surname;
std::cout << "Your name: " << name << ", your surname: " << surname << std::endl;
name.clear();
surname.clear();
std::istringstream("Peppy Ping") >> std::noskipws >> name >> surname;
std::cout << "Your name: " << name << ", your surname: " << surname << std::endl;
```

构建并运行这个例子后，我们将得到以下输出：

```cpp
Your name: Peppy, your surname: Ping
Your name: Peppy, your surname:
```

从前面的输出中可以看出，如果我们设置了`std::noskipws`标志，我们将读取空白字符。

在`<iomanip>`头文件中，声明了一个不寻常的操纵器：`std::quoted`。当这个函数应用于输入时，它会用转义字符将给定的字符串包装在引号中。如果输入字符串已经包含转义引号，它也会读取它们。为了理解这一点，让我们考虑一个小例子。我们将用一些没有引号的文本初始化一个源字符串，另一个字符串将用带有转义引号的文本初始化。接下来，我们将使用`std::ostringstream`读取它们，没有设置标志，并通过`std::cout`提供输出。看一下下面的例子：

```cpp
std::string str1("String without quotes");
std::string str2("String with quotes \"right here\"");
std::ostringstream ss;
ss << str1;
std::cout << "[" << ss.str() << "]" << std::endl;
ss.str("");
ss << str2;
std::cout << "[" << ss.str() << "]" << std::endl; 
```

结果如下：

```cpp
[String without quotes]
[String with quotes "right here"] 
```

现在，让我们用`std::quoted`调用做同样的输出：

```cpp
std::string str1("String without quotes");
std::string str2("String with quotes \"right here\"");
std::ostringstream ss;
ss << std::quoted(str1);
std::cout << "[" << ss.str() << "]" << std::endl;
ss.str("");
ss << std::quoted(str2);
std::cout << "[" << ss.str() << "]" << std::endl;
```

现在，我们将得到不同的结果：

```cpp
["String without quotes"]
["String with quotes \"right here\""]
```

你注意到第一个字符串被引号包裹，第二个字符串中的子字符串"right here"带有转义字符了吗？

现在，你知道如何将任何字符串包装在引号中了。你甚至可以编写自己的包装器来减少使用`std::quoted()`时的行数。例如，我们将流的工作移到一个单独的函数中：

```cpp
std::string quote(const std::string& str)
{
     std::ostringstream oss;
     oss << std::quoted(str);
     return oss.str();
}
```

然后，当我们需要时，我们调用我们的包装器：

```cpp
std::string str1("String without quotes");
std::string str2("String with quotes \"right here\"");
std::coot << "[" << quote(str1) << "]" << std::endl;
std::cout << "[" << quote(str2) << "]" << std::endl;
```

现在看起来好多了。第一个主题已经结束，让我们复习一下我们刚刚学到的东西。在实践中，我们学习了预定义流对象的使用，内存中的文件 I/O 操作，I/O 格式化，以及用户定义类型的 I/O。现在我们完全了解了如何在 C++中使用 I/O 库，我们将考虑当标准流不够用时该怎么办。

### 创建额外的流

当流的提供的接口不足以解决你的任务时，你可能需要创建一个额外的流，它将重用现有接口之一。你可能需要从特定的外部设备输出或提供输入，或者你可能需要添加调用 I/O 操作的线程的 Id。有几种方法可以做到这一点。你可以创建一个新的类，将现有流作为私有成员聚合起来。它将通过已经存在的流函数实现所有需要的函数，比如移位操作符。另一种方法是继承现有类，并以你需要的方式重写所有虚拟函数。

首先，您必须选择要使用的适当类。您的选择应取决于您想要添加哪种修改。如果您需要修改输入或输出操作，请选择`std::basic_istream`，`std::basic_ostream`和`std::basic_iostream`。如果您想要修改状态信息、控制信息、私有存储等，请选择`std::ios_base`。如果您想要修改与流缓冲区相关的内容，请选择`std::basic_ios`。在选择正确的基类之后，继承上述类之一以创建额外的流。

还有一件事情你必须知道 - 如何正确初始化标准流。在初始化文件或字符串流和基本流类方面，有一些重大区别。让我们来回顾一下。要初始化从文件流类派生的类的对象，您需要传递文件名。要初始化从字符串流类派生的类的对象，您需要调用默认构造函数。它们两者都有自己的流缓冲区，因此在初始化时不需要额外的操作。要初始化从基本流类派生的类的对象，您需要传递一个指向流缓冲区的指针。您可以创建一个缓冲区的变量，或者您可以使用预定义流对象的缓冲区，如`std::cout`或`std::cerr`。

让我们详细回顾一下创建额外流的这两种方法。

### 如何创建一个额外的流 - 组合

组合意味着在类的私有部分声明一些标准流对象作为类成员。当您选择适当的标准流类时，请转到其头文件并注意它有哪些构造函数。然后，您需要在类的构造函数中正确初始化这个成员。要将您的类用作流对象，您需要实现基本函数，如移位运算符、`str()`等。您可能还记得，每个流类都有针对内置类型的重载移位运算符。它们还有针对预定义函数的重载移位运算符，如`std::endl`。您需要能够将您的类用作真正的流对象。我们只需要创建一个模板，而不是声明所有 18 个重载的移位运算符。此外，为了允许使用预定义的操纵器，我们必须声明一个接受函数指针的移位运算符。

这看起来并不是很难，所以让我们尝试实现一个“包装器”来包装`std::ostream`对象。

### 练习 6：在用户定义的类中组合标准流对象

在这个练习中，我们将创建一个自己的流对象，包装`std::ostream`对象并添加额外的功能。我们将创建一个名为`extendedOstream`的类，它将向终端输出数据，并在每个输出的数据前插入以下数据：日期和时间以及线程 ID。要完成这个练习，执行以下步骤：

1.  包括所需的头文件：`<iostream>`用于`std::endl`支持，`<sstream>`用于`std::ostream`支持，`<thread>`用于`std::this_thread::get_id()`支持，`<chrono>`用于`std::chrono::system_clock::now()`，和`<ctime>`用于将时间戳转换为可读表示：

#### 注意

```cpp
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <ctime>
```

1.  接下来，声明`extendedOstream`类。声明名为`m_oss`的`std::ostream`变量和名为`writeAdditionalInfo`的 bool 变量。这个 bool 变量将用于指示是否应该打印扩展数据：

```cpp
class extendedOstream
{
private:
     std::ostream& m_oss;
     bool writeAdditionalInfo;
};
```

1.  接下来，在公共部分，定义一个默认构造函数，并用`std::cout`初始化`m_oss`以将输出重定向到终端。用`true`初始化`writeAdditionalInfo`：

```cpp
extendedOstream()
     : m_oss(std::cout)
     , writeAdditionalInfo(true)
{
}
```

1.  定义一个模板重载的左移操作符`<<`，它返回对`extendedOstream`的引用，并带有名为 value 的模板参数。然后，如果`writeAdditionalInfo`为`true`，输出时间、线程 ID 和给定的值，然后将`writeAdditionalInfo`设置为`false`。如果`writeAdditionalInfo`为`false`，只输出给定的值。这个函数将用于所有内置类型的输出：

```cpp
template<typename T>
extendedOstream& operator<<(const T& value)
{
     if (writeAdditionalInfo)
     {
          std::string time = fTime();
          auto id = threadId();
          m_oss << time << id << value;
          writeAdditionalInfo = false;
     }
     else
     {
          m_oss << value;
     }
     return *this;
}
```

1.  定义另一个重载的左移操作符，它以函数指针作为参数并返回对`std::ostream`的引用。在函数体中，将`writeAdditionalInfo`设置为`true`，调用给定的函数，并将`m_oss`作为参数传递。这个重载的操作符将用于预定义函数，如`std::endl`：

```cpp
extendedOstream&
operator<<(std::ostream& (*pfn)(std::ostream&))
{
     writeAdditionalInfo = true;
     pfn(m_oss);
     return *this;
}
```

1.  在私有部分，定义`fTime`函数，返回 std::string。它获取系统时间。将其格式化为可读表示，并返回它：

```cpp
std::string fTime()
{
     auto now = std::chrono::system_clock::now();
     std::time_t time = std::chrono::system_clock::to_time_t(now);
     std::ostringstream oss;
     std::string strTime(std::ctime(&time));
     strTime.pop_back();
     oss << "[" << strTime << "]";
     return oss.str();
}
```

1.  在私有部分，定义`threadId()`函数，返回一个字符串。获取当前线程的`id`，格式化它，并返回它：

```cpp
std::string threadId()
{
     auto id = std::this_thread::get_id();
     std::ostringstream oss;
     oss << "[" << std::dec << id << "]";
     return oss.str();
}
```

1.  进入`main`函数。为了测试我们的流对象如何工作，创建一个名为`oss`的`extendedOstream`类型的对象。输出不同的数据，例如整数、浮点数、十六进制和布尔值：

```cpp
extendedOstream oss;
oss << "Integer: " << 156 << std::endl;
oss << "Float: " << 156.12 << std::endl;
oss << "Hexadecimal: " << std::hex << std::showbase 
    << std::uppercase << 0x2a << std::endl;
oss << "Bool: " << std::boolalpha << false << std::endl;
```

1.  然后，创建一个线程，用 lambda 函数初始化它，并在 lambda 内部放置相同的输出。不要忘记加入线程：

```cpp
std::thread thr1([]()
     {
          extendedOstream oss;
          oss << "Integer: " << 156 << std::endl;
          oss << "Float: " << 156.12 << std::endl;
          oss << "Hexadecimal: " << std::hex << std::showbase
              << std::uppercase << 0x2a << std::endl;
          oss << "Bool: " << std::boolalpha << false << std::endl;
     });
thr1.join();
```

1.  现在，构建并运行应用程序。你将得到以下输出：

![](img/C14583_06_15.jpg)

###### 图 6.15：执行练习 6 的结果

考虑输出的每一行。你可以看到输出的下一个格式："[日期和时间][线程 ID]输出数据"。确保线程 ID 在不同的线程之间不同。然后，数据以预期的格式输出。所以，正如你所看到的，使用标准流的组合实现自己的 I/O 流对象并不太难。

### 如何创建一个附加流 - 继承

继承意味着你创建自己的流类，并从具有虚拟析构函数的标准流对象中继承它。你的类必须是一个模板类，并且具有模板参数，就像父类一样。要使用你的所有继承函数与你的类的对象，继承应该是公共的。在构造函数中，你应该根据类的类型初始化父类 - 使用文件名、流缓冲区或默认值。接下来，你应该重写那些基本函数，根据你的要求进行更改。

我们需要继承标准流类的最常见情况是当我们想要为新设备（如套接字或打印机）实现 I/O 操作时。所有定义的标准流类都负责格式化输入和输出，并且对字符串、文件和终端进行了重载。只有`std::basic_streambuf`类负责与设备一起工作，因此我们需要继承这个类，编写我们自己的实现，并将其设置为标准类的流缓冲区。`streambuf`类的核心功能是传输字符。它可以使用缓冲区在刷新之间存储字符，也可以在每次调用后立即刷新。这些概念称为缓冲和非缓冲字符传输。

输出操作的缓冲字符传输工作如下：

1.  通过`sputc()`函数调用将字符缓冲到内部缓冲区。

1.  当缓冲区满时，`sputc()`调用受保护的虚拟成员`overflow()`。

1.  `overflow()`函数将所有缓冲区内容传输到外部设备。

1.  调用`pubsync()`函数时，它会调用受保护的虚拟成员`sync()`。

1.  `sync()`函数将所有缓冲区内容传输到外部设备。

输出操作的非缓冲字符传输工作略有不同：

1.  字符传递给`sputc()`函数。

1.  `sputc()`函数立即调用被称为`overflow()`的受保护虚拟成员。

1.  `overflow()`函数将所有缓冲区内容传输到外部设备。

因此，对于输出操作的缓冲和非缓冲字符传输，我们应该重写`overflow()`和`sync()`函数，这些函数执行实际工作。

用于输入操作的缓冲字符传输工作如下：

1.  `sgetc()`函数从内部缓冲区读取字符。

1.  `sgetc()`函数调用`sungetc()`函数，使已消耗的字符再次可用。

1.  如果内部缓冲区为空，`sgetc()`函数会调用`underflow()`函数。

1.  `underflow()`函数从外部设备读取字符到内部缓冲区。

`sgetc()`和`underflow()`函数总是返回相同的字符。为了每次读取不同的字符，我们有另一对函数：`sbumpc()`和`uflow()`。使用它们读取字符的算法是相同的：

1.  `sbumpc()`函数从内部缓冲区读取字符。

1.  `sbumpc()`函数调用`sputbackc()`函数，使下一个字符可用于输入。

1.  如果内部缓冲区为空，`sbumpc()`函数会调用`uflow()`函数。

1.  `uflow()`函数从外部设备读取字符到内部缓冲区。

用于输入操作的非缓冲字符传输工作如下：

1.  `sgetc()`函数调用一个被称为`underflow()`的受保护虚拟成员。

1.  `underflow()`函数从外部设备读取字符到内部缓冲区。

1.  `sbumpc()`函数调用一个被称为`uflow()`的受保护虚拟成员。

1.  `uflow()`函数从外部设备读取字符到内部缓冲区。

在发生任何错误的情况下，会调用被称为`pbackfail()`的受保护虚拟成员，该成员处理错误情况。因此，可以看到，要重写`std::basic_streambuf`类，我们需要重写与外部设备一起工作的虚拟成员。对于输入`streambuf`，我们应该重写`underflow()`、`uflow()`和`pbackfail()`成员。对于输出`streambuf`，我们应该重写`overflow()`和`sync()`成员。

让我们更详细地考虑所有这些步骤。

### 练习 7：继承标准流对象

在这个练习中，我们将创建一个名为`extended_streambuf`的类，它继承自`std::basic_streambuf`。我们将使用`std::cout`流对象的缓冲区，并重写`overflow()`函数，以便我们可以将数据写入外部设备（`stdout`）。接下来，我们将编写一个名为`extended_ostream`的类，它继承自`std::basic_ostream`类，并将流缓冲区设置为`extended_streambuf`。最后，我们将对我们的包装类进行微小的更改，并将`extended_ostream`用作私有流成员。要完成此练习，请执行以下步骤：

1.  包括所需的头文件：`<iostream>`用于支持`std::endl`，`<sstream>`用于支持`std::ostream`和`std::basic_streambuf`，`<thread>`用于支持`std::this_thread::get_id()`，`<chrono>`用于支持`std::chrono::system_clock::now()`，`<ctime>`用于将时间戳转换为可读状态。

1.  创建一个名为`extended_streambuf`的模板类，它继承自`std::basic_streambuf`类。重写一个名为`overflow()`的公共成员，该成员将字符写入输出流并返回 EOF 或已写入的字符：

```cpp
template< class CharT, class Traits = std::char_traits<CharT> >
class extended_streambuf : public std::basic_streambuf< CharT, Traits >
{
public:
    int overflow( int c = EOF ) override
    {
        if (!Traits::eq_int_type(c, EOF))
        {
            return fputc(c, stdout);
        }
        return Traits::not_eof(c);
    }
};
```

1.  接下来，创建一个名为`extended_ostream`的模板类，它是从`std::basic_ostream`类派生而来的。在私有部分，定义一个`extended_streambuf`类的成员，即缓冲区。用缓冲区成员初始化`std::basic_ostream`父类。然后，在构造函数体中，使用缓冲区作为参数调用父类的`init()`函数。还要重载`rdbuf()`函数，该函数返回指向缓冲区变量的指针：

```cpp
template< class CharT, class Traits = std::char_traits<CharT> >
class extended_ostream : public std::basic_ostream< CharT, Traits >
{
public:
    extended_ostream()
        : std::basic_ostream< CharT, Traits >::basic_ostream(&buffer)
        , buffer()
    {
        this->init(&buffer);
    }
    extended_streambuf< CharT, Traits >* rdbuf () const
    {
        return (extended_streambuf< CharT, Traits >*)&buffer;
    }
private:
    extended_streambuf< CharT, Traits > buffer;
};
```

1.  将`extendedOstream`类重命名为 logger，以避免与类似名称的误解。保持现有接口不变，但用我们自己的流替换`std::ostream&`成员，即`object - extended_ostream`。完整的类如下所示：

```cpp
class logger
{
public:
     logger()
          : m_log()
          , writeAdditionalInfo(true)
     {
     }
     template<typename T>
     logger& operator<<(const T& value)
     {
          if (writeAdditionalInfo)
          {
               std::string time = fTime();
               auto id = threadId();
               m_log << time << id << value;
               writeAdditionalInfo = false;
          }
          else
          {
               m_log << value;
          }
          return *this;
     }
     logger&
     operator<<(std::ostream& (*pfn)(std::ostream&))
     {
          writeAdditionalInfo = true;
          pfn(m_log);
          return *this;
     }
private:
     std::string fTime()
     {
          auto now = std::chrono::system_clock::now();
          std::time_t time = std::chrono::system_clock::to_time_t(now);
          std::ostringstream log;
          std::string strTime(std::ctime(&time));
          strTime.pop_back();
          log << "[" << strTime << "]";
          return log.str();
     }
     std::string threadId()
     {
          auto id = std::this_thread::get_id();
          std::ostringstream log;
          log << "[" << std::dec << id << "]";
          return log.str();
     }
private:
     extended_ostream<char> m_log;
     bool writeAdditionalInfo;
};
```

1.  进入`main`函数并将`extendedOstream`对象更改为`logger`对象。将其余代码保持不变。现在，构建并运行练习。您将看到在上一个练习中给出的输出，但在这种情况下，我们使用了自己的流缓冲区，自己的流对象和一个包装类，为输出添加了额外的信息。查看下面截图中显示的执行结果，并将其与先前的结果进行比较。确保它们是相似的。如果是这样，那就意味着我们做得很好，我们的继承类按预期工作：

![图 6.16：执行练习 7 的结果](img/C14583_06_16.jpg)

###### 图 6.16：执行练习 7 的结果

在这个主题中，我们做了很多工作，学会了如何以不同的方式创建额外的流。我们考虑了所有适当的继承类，以及哪个类更适合不同的需求。我们还学会了如何从基本 streambuf 类继承，以实现与外部设备的工作。现在，我们将学习如何以异步方式使用 I/O 流。

### 利用异步 I/O

有很多情况下，I/O 操作可能需要很长时间，例如创建备份文件，搜索大型数据库，读取大文件等。您可以使用线程执行 I/O 操作，而不阻塞应用程序的执行。但对于一些应用程序来说，处理长时间 I/O 的方式并不适合，例如当每秒可能有数千次 I/O 操作时。在这些情况下，C++开发人员使用异步 I/O。它可以节省线程资源，并确保执行线程不会被阻塞。让我们来看看同步和异步 I/O 是什么。

正如您可能还记得第五章《哲学家的晚餐-线程和并发》，同步操作意味着某个线程调用操作并等待其完成。它可以是单线程或多线程应用程序。关键是线程正在等待 I/O 操作完成。

异步执行发生在操作不阻塞工作线程的情况下。执行异步 I/O 操作的线程发送异步请求并继续执行另一个任务。当操作完成时，初始线程将收到完成通知，并可以根据需要处理结果。

从这个角度看，异步 I/O 似乎比同步更好，但这取决于情况。如果需要执行大量快速的 I/O 操作，由于处理内核 I/O 请求和信号的开销，更适合遵循同步方式。因此，在开发应用程序架构时，需要考虑所有可能的情况。

标准库不支持异步 I/O 操作。因此，为了利用异步 I/O，我们需要考虑替代库或编写自己的实现。首先，让我们考虑依赖于平台的实现。然后，我们将看看跨平台库。

### Windows 平台上的异步 I/O

Windows 支持各种设备的 I/O 操作：文件、目录、驱动器、端口、管道、套接字、终端等。一般来说，我们对所有这些设备使用相同的 I/O 接口，但某些设置因设备而异。让我们考虑在 Windows 上对文件进行 I/O 操作。

因此，在 Windows 中，我们需要打开设备并获取其处理程序。不同的设备以不同的方式打开。要打开文件、目录、驱动器或端口，我们使用`<Windows.h>`头文件中的`CreateFile`函数。要打开管道，我们使用`CreateNamedPipe`函数。要打开套接字，我们使用 socket()和 accept()函数。要打开终端，我们使用`CreateConsoleScreenBuffer`和`GetStdHandle`函数。它们都返回一个设备处理程序，该处理程序用于所有与该设备的操作。

`CreateFile`函数接受七个参数，用于管理打开设备的操作。函数声明如下所示：

```cpp
HANDLE CreateFile( PCTSTR pszName, 
                   DWORD  dwDesiredAccess, 
                   DWORD  dwShareMode, 
                   PSECURITY_ATTRIBUTES psa, 
                   DWORD  dwCreationDisposition, 
                   DWORD  dwFlagsAndAttributes, 
                   HANDLE hFileTemplate);
```

第一个参数是`pszName` - 文件的路径。第二个参数调用`dwDesiredAccess`并管理对设备的访问。它可以取以下值之一：

```cpp
0 // only for configuration changing
GENERIC_READ // only reading
GENERIC_WRITE // only for writing
GENERIC_READ | GENERIC_WRITE // both for reading and writing
```

第三个参数`dwShareMode`管理操作系统在文件已经打开时如何处理所有新的`CreateFile`调用。它可以取以下值之一：

```cpp
0 // only one application can open device simultaneously
FILE_SHARE_READ // allows reading by multiple applications simultaneously
FILE_SHARE_WRITE // allows writing by multiple applications simultaneously
FILE_SHARE_READ | FILE_SHARE_WRITE // allows both reading and writing by multiple applications simultaneously
FILE_SHARE_DELETE // allows moving or deleting by multiple applications simultaneously
```

第四个参数`psa`通常设置为`NULL`。第五个参数`dwCreationDisposition`管理文件是打开还是创建。它可以取以下值之一：

```cpp
CREATE_NEW // creates new file or fails if it is existing
CREATE_ALWAYS // creates new file or overrides existing
OPEN_EXISTING // opens file or fails if it is not exists
OPEN_ALWAYS // opens or creates file
TRUNCATE_EXISTING // opens existing file and truncates it or fails if it is not exists
```

第六个参数`dwFlagsAndAttributes`管理缓存或文件的操作。它可以取以下值之一来管理缓存：

```cpp
FILE_FLAG_NO_BUFFERING // do not use cache
FILE_FLAG_SEQUENTIAL_SCAN // tells the OS that you will read the file sequentially
FILE_FLAG_RANDOM_ACCESS // tells the OS that you will not read the file in sequentially
FILE_FLAG_WR1TE_THROUGH // write without cache but read with
```

它可以取以下值之一来管理文件的操作：

```cpp
FILE_FLAG_DELETE_ON_CLOSE // delete file after closing (for temporary files)
FILE_FLAG_BACKUP_SEMANTICS // used for backup and recovery programs
FILE_FLAG_POSIX_SEMANTICS // used to set case sensitive when creating or opening a file
FILE_FLAG_OPEN_REPARSE_POINT // allows to open, read, write, and close files differently
FILE_FLAG_OPEN_NO_RECALL // prevents the system from recovering the contents of the file from archive media
FILE_FLAG_OVERLAPPED // allows to work with the device asynchronously
```

它可以取以下值之一来管理文件属性：

```cpp
FILE_ATTRIBUTE_ARCHIVE // file should be deleted
FILE_ATTRIBUTE_ENCRYPTED // file is encrypted
FILE_ATTRIBUTE_HIDDEN // file is hidden
FILE_ATTRIBUTE_NORMAL // other attributes are not set
FILE_ATTRIBUTE_NOT_CONTENT_ INDEXED // file is being processed by the indexing service
FILE_ATTRIBUTE_OFFLINE // file is transferred to archive media
FILE_ATTRIBUTE_READONLY // only read access
FILE_ATTRIBUTE_SYSTEM // system file
FILE_ATTRIBUTE_TEMPORARY // temporary file
```

最后一个参数`hFileTemplate`接受打开文件的句柄或`NULL`作为参数。如果传递了文件句柄，`CreateFile`函数将忽略所有属性和标志，并使用打开文件的属性和标志。

这就是关于`CreateFile`参数的全部内容。如果无法打开设备，它将返回`INVALID_HANDLE_VALUE`。以下示例演示了如何打开文件进行读取：

```cpp
#include <iostream>
#include <Windows.h>
int main()
{
     HANDLE hFile = CreateFile(TEXT("Test.txt"), GENERIC_READ, 
                                FILE_SHARE_READ | FILE_SHARE_WRITE, 
                                NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
     if (INVALID_HANDLE_VALUE == hFile)
         std::cout << "Failed to open file for reading" << std::endl;
     else
         std::cout << "Successfully opened file for reading" << std::endl;
     CloseHandle(hFile);
     return 0;
}
```

接下来，要执行输入操作，我们使用`ReadFile`函数。它将文件描述符作为第一个参数，源缓冲区作为第二个参数，要读取的最大字节数作为第三个参数，读取字节数作为第四个参数，`NULL`值作为同步执行或者指向有效且唯一的 OVERLAPPED 结构的指针作为最后一个参数。如果操作成功，`ReadFile`返回 true，否则返回 false。以下示例演示了如何从先前打开的文件进行读取输入：

```cpp
BYTE pb[20];
DWORD dwNumBytes;
ReadFile(hFile, pb, 20, &dwNumBytes, NULL);
```

要执行输出操作，我们使用`WriteFile`函数。它与`ReadFile`具有相同的声明，但第三个参数设置要写入的字节数，第五个参数是写入的字节数。以下示例演示了如何向先前打开的文件进行写入输出：

```cpp
BYTE pb[20] = "Some information\0";
DWORD dwNumBytes;
WriteFile(hFile, pb, 20, &dwNumBytes, NULL);
```

要将缓存数据写入设备，使用`FlushFileBuffer`函数。它只有一个参数 - 文件描述符。让我们转向异步 I/O。要让操作系统知道您计划异步地使用设备，需要使用`FILE_FLAG_OVERLAPPED`标志打开它。现在，打开文件进行写入或读取如下所示：

```cpp
#include <iostream>
#include <Windows.h>
int main()
{
     HANDLE hFile = CreateFile(TEXT("Test.txt"), GENERIC_READ, 
                                FILE_SHARE_READ | FILE_SHARE_WRITE, 
                                NULL, OPEN_ALWAYS, FILE_FLAG_OVERLAPPED, NULL);
     if (INVALID_HANDLE_VALUE == hFile)
         std::cout << "Failed to open file for reading" << std::endl;
     else
         std::cout << "Successfully opened file for reading" << std::endl;
     CloseHandle(hFile);
     return 0;
}
```

我们使用相同的操作来执行对文件的读取或写入，即`ReadFile`和`WriteFile`，唯一的区别是读取或写入的字节数设置为 NULL，我们必须传递一个有效且唯一的`OVERLAPPED`对象。让我们考虑一下`OVERLAPPED`对象的结构是什么：

```cpp
typedef struct _OVERLAPPED { 
DWORD  Internal; // for error code 
DWORD  InternalHigh; // for number of read bytes 
DWORD  Offset; 
DWORD  OffsetHigh; 
HANDLE hEvent; // handle to an event 
} OVERLAPPED, *LPOVERLAPPED;
```

内部成员设置为`STATUS_PENDING`，这意味着操作尚未开始。读取或写入的字节数将写入`InternalHigh`成员。在异步操作中，`Offset`和`OffsetHigh`将被忽略。`hEvent`成员用于接收有关异步操作完成的事件。

#### 注意

I/O 操作的顺序不能保证，因此您不能依赖于此。如果您计划在一个地方写入文件，并在另一个地方从文件中读取，您不能依赖于顺序。

在异步模式下使用`ReadFile`和`WriteFile`时有一个不寻常的地方。如果它们以同步方式执行 I/O 请求，则返回一个非零值。如果它们返回`FALSE`，你需要调用`GetLastError`函数来检查为什么返回了`FALSE`。如果错误代码是`ERROR_IO_PENDING`，这意味着 I/O 请求已成功处理，处于挂起状态，并将在以后执行。

你应该记住的最后一件事是，在 I/O 操作完成之前，不能移动或删除`OVERLAPPED`对象或数据缓冲区。对于每个 I/O 操作，你应该创建一个新的 OVERLAPPED 对象。

最后，让我们考虑系统通知我们完成 I/O 操作的方式。有几种这样的机制：释放设备、释放事件、产生警报和使用 I/O 端口。

`WriteFile`和`ReadFile`函数将设备设置为“占用”状态。当 I/O 操作完成时，驱动程序将设备设置为“空闲”状态。我们可以通过调用`WaitForSingleObject`或`WaitForMultipleObject`函数来检查完成的 I/O 操作。以下示例演示了这种方法：

```cpp
#include <Windows.h>
#include <WinError.h>
int main()
{
     HANDLE hFile = CreateFile(TEXT("Test.txt"), GENERIC_READ,
                                     FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                                     OPEN_ALWAYS, FILE_FLAG_OVERLAPPED, NULL);
     BYTE bBuffer[100];
     OVERLAPPED o = { 0 };
     BOOL bResult = ReadFile(hFile, bBuffer, 100, NULL, &o);
     DWORD dwError = GetLastError();
     if (bResult && (dwError == ERROR_IO_PENDING))
     {
          WaitForSingleObject(hFile, INFINITE);
          bResult = TRUE;
     }
     CloseHandle(hFile);
     return 0;
}
```

这是检查 I/O 操作是否已完成的最简单方法。但这种方法使调用线程在`WaitForSingleObject`调用上等待，因此它变成了一个同步调用。此外，你可以为该设备启动几个 I/O 操作，但不能确定线程是否会在需要释放设备时唤醒。

使用`CreateEvent`函数并将其设置为`OVERLAPPED`对象。然后，当 I/O 操作完成时，系统通过调用`SetEvent`函数释放此事件。接下来，当调用线程需要获取正在执行的 I/O 操作的结果时，你调用`WaitForSingleObject`并传递此事件的描述符。以下示例演示了这种方法：

```cpp
#include <Windows.h>
#include <synchapi.h>
int main()
{
     HANDLE hFile = CreateFile(TEXT("Test.txt"), GENERIC_READ, 
                               FILE_SHARE_READ | FILE_SHARE_WRITE,
                               NULL, OPEN_ALWAYS, FILE_FLAG_OVERLAPPED, NULL);
     BYTE bInBuffer[10];
     OVERLAPPED o = { 0 };
     o.hEvent = CreateEvent(NULL,TRUE,FALSE,"IOEvent");
     ReadFile(hFile, bInBuffer, 10, NULL, &o);
     ///// do some work
     HANDLE hEvent = o.hEvent;
     WaitForSingleObject(hEvent, INFINITE);
     CloseHandle(hFile);
     return 0;
}
```

如果你希望通知调用线程 I/O 操作的结束，这是一个相当简单的方法。但这并不是最理想的做法，因为当有很多这样的操作时，你需要为每个操作创建一个事件对象。

`ReadFileEx`和`WriteFileEx`用于输入/输出。它们类似于标准的`ReadFile`和`WriteFile`，但我们不传递存储读取或写入字符数的变量，而是传递回调函数的地址。这个回调函数被称为完成例程，并且具有以下声明：

```cpp
VOID WINAPI 
CompletionRoutine(DWORD dwError,
                  DWORD dwNumBytes,
                  OVERLAPPED* po);
```

`ReadFileEx`和`WriteFileEx`将回调函数的地址传递给设备驱动程序。当设备上的操作完成时，驱动程序将回调函数的地址添加到 APC 队列和 OVERLAPPED 结构的指针。然后，操作系统调用此函数并传递读取或写入的字节数、错误代码和 OVERLAPPED 结构的指针。

这种方法的主要缺点是编写回调函数和使用大量全局变量，因为回调函数在上下文中包含少量信息。不使用这种方法的另一个原因是只有调用线程才能接收有关完成的通知。

现在我们已经讨论了不好的地方，让我们看看处理 I/O 结果的最佳方法 - I/O 端口。I/O 完成端口是为与线程池一起使用而开发的。要创建这样一个端口，我们使用`CreateIoCompletionPort`。该函数的声明如下：

```cpp
HANDLE 
CreateIoCompletionPort(HANDLE hFile,
                       HANDLE hExistingCompletionPort,
                       ULONG_PTR CompletionKey,
                       DWORD dwNumberOfConcurrentThreads);
```

此函数创建一个 I/O 完成端口并将设备与此端口关联。要完成此操作，我们需要调用两次。要创建新的完成端口，我们调用`CreateIoCompletionPort`函数，并将`INVALID_HANDLE_VALUE`作为第一个参数传递，NULL 作为第二个参数，0 作为第三个参数，并传递此端口的线程数。将 0 作为第四个参数将使线程数等于处理器的数量。

#### 注意

对于 I/O 完成端口，建议使用线程数等于处理器数量的两倍。

接下来，我们需要将此端口与输入/输出设备关联起来。因此，我们第二次调用`CreateIoCompletionPort`函数，并传递设备的描述符、创建的完成端口的描述符、将指示对设备进行读取或写入的常量，以及 0 作为线程数。然后，当我们需要获取完成的结果时，我们从我们的端口描述符调用`GetQueuedCompletionStatus`。如果操作完成，函数会立即返回结果。如果没有完成，线程就会等待完成。以下示例演示了这种方法：

```cpp
#include <Windows.h>
#include <synchapi.h>
int main()
{
    HANDLE hFile = CreateFile(TEXT("Test.txt"), GENERIC_READ,
                              FILE_SHARE_READ | FILE_SHARE_WRITE,
                              NULL, OPEN_ALWAYS, FILE_FLAG_OVERLAPPED, NULL);
    HANDLE m_hIOcp = CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 0);
    CreateIoCompletionPort(hFile, m_hIOcp, 1, 0);

    BYTE bInBuffer[10];
    OVERLAPPED o = { 0 };
    ReadFile(hFile, bInBuffer, 10, NULL, &o);

    DWORD dwNumBytes;
    ULONG_PTR completionKey;
    GetQueuedCompletionStatus(m_hIOcp, &dwNumBytes, &completionKey, (OVERLAPPED**) &o, INFINITE);
    CloseHandle(hFile);
    return 0;
}
```

### Linux 平台上的异步 I/O

Linux 上的异步 I/O 支持对不同设备进行输入和输出，如套接字、管道和 TTY，但不包括文件。是的，这很奇怪，但 Linux 开发人员决定文件的 I/O 操作已经足够快了。

要打开 I/O 设备，我们使用 open()函数。它的声明如下：

```cpp
int open (const char *filename, int flags[, mode_t mode])
```

第一个参数是文件名，而第二个参数是一个控制文件应如何打开的位掩码。如果系统无法打开设备，open()返回值为-1。在成功的情况下，它返回一个设备描述符。open 模式的可能标志是`O_RDONLY`、`O_WRONLY`和`O_RDWR`。

为了执行输入/输出操作，我们使用名为`aio`的`POSIX`接口。它们有一组定义好的函数，如`aio_read`、`aio_write`、`aio_fsync`等。它们用于启动异步操作。要获取执行结果，我们可以使用信号通知或实例化线程。或者，我们可以选择不被通知。所有这些都在`<aio.h>`头文件中声明。

几乎所有这些都以`aiocb`结构（异步 IO 控制块）作为参数。它控制 IO 操作。该结构的声明如下：

```cpp
struct aiocb 
{
    int aio_fildes;
    off_t aio_offset;
    volatile void *aio_buf;
    size_t aio_nbytes;
    int aio_reqprio;
    struct sigevent aio_sigevent;
    int aio_lio_opcode;
};
```

`aio_fildes`成员是打开设备的描述符，而`aio_offset`成员是在进行读取或写入操作的设备中的偏移量。`aio_buf`成员是指向要读取或写入的缓冲区的指针。`aio_nbytes`成员是缓冲区的大小。`aio_reqprio`成员是此 IO 操作执行的优先级。`aio_sigevent`成员是一个指出调用线程应如何被通知完成的结构。`aio_lio_opcode`成员是 I/O 操作的类型。以下示例演示了如何初始化`aiocb`结构：

```cpp
std::string fileContent;
constexpr int BUF_SIZE = 20;
fileContent.resize(BUF_SIZE, 0);
aiocb aiocbObj;
aiocbObj.aio_fildes = open("test.txt", O_RDONLY);
if (aiocbObj.aio_fildes == -1)
{
     std::cerr << "Failed to open file" << std::endl;
     return -1;
}
aiocbObj.aio_buf = const_cast<char*>(fileContent.c_str());
aiocbObj.aio_nbytes = BUF_SIZE;
aiocbObj.aio_reqprio = 0;
aiocbObj.aio_offset = 0;
aiocbObj.aio_sigevent.sigev_notify = SIGEV_SIGNAL;
aiocbObj.aio_sigevent.sigev_signo = SIGUSR1;
aiocbObj.aio_sigevent.sigev_value.sival_ptr = &aiocbObj;
```

在这里，我们为读取文件内容创建了一个缓冲区，即`fileContent`。然后，我们创建了一个名为`aiocbObj`的`aiocb`结构。接下来，我们打开了一个文件进行读取，并检查了这个操作是否成功。然后，我们设置了指向缓冲区和缓冲区大小的指针。缓冲区大小告诉驱动程序应该读取或写入多少字节。接下来，我们指出我们将从文件的开头读取，将偏移量设置为 0。然后，我们设置了`SIGEV_SIGNAL`中的通知类型，这意味着我们希望得到有关完成操作的信号通知。然后，我们设置了应触发完成通知的信号号码。在我们的情况下，它是`SIGUSR1` - 用户定义的信号。接下来，我们将`aiocb`结构的指针设置为信号处理程序。

创建和正确初始化`aiocb`结构之后，我们可以执行输入或输出操作。让我们完成一个练习，以了解如何在 Linux 平台上使用异步 I/O。

### 练习 8：在 Linux 上异步读取文件

在这个练习中，我们将开发一个应用程序，以异步方式从文件中读取数据，并将读取的数据输出到控制台。当执行读取操作时，驱动程序使用触发信号通知应用程序。要完成这个练习，执行以下步骤：

1.  包括所有必需的头文件：`<aio.h>`用于异步读写支持，`<signal.h>`用于信号支持，`<fcntl.h>`用于文件操作，`<unistd.h>`用于符号常量支持，`<iostream>`用于输出到终端，`<chrono>`用于时间选项，`<thread>`用于线程支持：

```cpp
#include <aio.h>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <thread>
```

1.  创建一个名为`isDone`的 bool 变量，用于指示操作何时已完成：

```cpp
bool isDone{};
```

1.  定义将作为我们的信号处理程序的函数，即`aioSigHandler`。当异步操作完成时将调用它。信号处理程序应具有以下签名：

```cpp
void name(int number, siginfo_t* si, void* additional)
```

1.  第一个参数是信号编号，第二个参数是一个包含有关信号生成原因的信息的结构，最后一个参数是附加信息。它可以转换为`ucontext_t`结构的指针，以便我们可以接收到被该信号中断的线程上下文。在`aioSigHandler`中，检查异步 I/O 操作相关的信号是否是常量，使用`SI_ASYNCIO`。如果是，输出一条消息。接下来，将`isDone`设置为`true`：

```cpp
void
aioSigHandler(int no, siginfo_t* si, void*)
{
     std::cout << "Signo: " << no << std::endl;
     if (si->si_code == SI_ASYNCIO)
     {
          std::cout << "I/O completion signal received" << std::endl;
     }
     isDone = true;
}
```

1.  定义另一个辅助函数，名为`initSigAct`。它将初始化`sigaction`结构。该结构定义了在 I/O 操作完成时将发送哪个信号以及应调用哪个处理程序。在这里，我们选择了`SIGUSR1` - 一个用户定义的信号。在`sa_flags`中，设置我们希望在操作重新启动或接收到信息时传递此信号：

```cpp
bool 
initSigAct(struct sigaction& item)
{
     item.sa_flags = SA_RESTART | SA_SIGINFO;
     item.sa_sigaction = aioSigHandler;
     if (-1 == sigaction(SIGUSR1, &item, NULL))
     {
          std::cerr << "sigaction usr1 failed" << std::endl;
          return false;
     }
     std::cout << "Successfully set up a async IO handler to SIGUSR1 action" << std::endl;
     return true;
}
```

1.  定义名为`fillAiocb`的辅助函数，它将使用给定的参数填充`aiocb`结构。它将以 aiocb 结构的引用、文件描述符、缓冲区指针和缓冲区大小作为参数。在`sigev_signo`中设置`SIGUSR1`，这是我们之前初始化的：

```cpp
void 
fillAiocb(aiocb& item, const int& fileDescriptor,
          char* buffer, const int& bufSize)
{
     item.aio_fildes = fileDescriptor;
     item.aio_buf = static_cast<void*>(buffer);
     item.aio_nbytes = bufSize;
     item.aio_reqprio = 0;
     item.aio_offset = 0;
     item.aio_sigevent.sigev_notify = SIGEV_SIGNAL;
     item.aio_sigevent.sigev_signo = SIGUSR1;
     item.aio_sigevent.sigev_value.sival_ptr = &item;
}
```

1.  进入`main`函数。定义名为`buf_size`的变量，其中包含缓冲区大小。创建一个该大小的缓冲区：

```cpp
constexpr int bufSize = 100;
char* buffer = new char(bufSize);
if (!buffer)
{
     std::cerr << "Failed to allocate buffer" << std::endl;
     return -1;
}
```

1.  创建一个名为`fileName`的变量，其中包含一个名为"`Test.txt`"的文件。然后，以只读方式打开此文件：

```cpp
const std::string fileName("Test.txt");
int descriptor = open(fileName.c_str(), O_RDONLY);
if (-1 == descriptor)
{
     std::cerr << "Failed to opene file for reading" << std::endl;
     return -1;
}
std::cout << "Successfully opened file for reading" << std::endl;
```

1.  创建一个`sigaction`结构并使用`initSigAct`函数进行初始化：

```cpp
struct sigaction sa;
if (!initSigAct(sa))
{
     std::cerr << "failed registering signal" << std::endl;
     return -1;
}
```

1.  创建一个`aiocb`结构并使用`fillAiocb`函数进行初始化：

```cpp
aiocb aiocbObj;
fillAiocb(aiocbObj, descriptor, buffer, bufSize);
```

1.  使用`aio_read`函数执行`read`操作：

```cpp
if (-1 == aio_read(&aiocbObj))
{
     std::cerr << "aio_read failed" << std::endl;
}
```

1.  接下来，在循环中，评估`isDone`变量。如果它为 false，则使线程休眠`3ms`。通过这样做，我们将等待 I/O 操作完成：

```cpp
while (!isDone)
{
     using namespace std::chrono_literals;
     std::this_thread::sleep_for(3ms);
}
std::cout << "Successfully finished read operation. Buffer: " << std::endl << buffer; 
```

1.  在运行此练习之前，在项目目录中创建一个`Test.txt`文件，并写入不同的符号。例如，我们的文件包含以下数据：

```cpp
a1a"1 a1\a1 a1	a1 
a1a"1 a1\a1 a1	a1 
a1a"1 a1\a1 a1	a1 
a1a"1 a1\a1 a1	a1 
a1a"1 a1\a1 a1	a1 
a1a"1 a1\a1 a1	a1 
a1a"1 a1\a1 a1	a1 
a1a"1 a1\a1 a1	a1 
a1a"1 a1\a1 a1	a1 
a1a"1 a1\a1 a1	a1 
a1a"1 a1\a1 a1	a1 
a1a"1 a1\a1 a1	a1 
a1a"1 a1\a1 a1	a1 
a1a"1 a1\a1 a1	a1
```

这里有字母字符、数字字符、特殊符号、空格、制表符和换行符。

1.  现在，在您的 IDE 中构建并运行此练习。您的输出将类似于以下内容：

![](img/C14583_06_17.jpg)

###### 图 6.17：执行练习 8 的结果

您可以看到文件已成功打开进行读取，并且我们成功设置了`SIGUSR1`信号和其处理程序。然后，我们收到信号编号 30，即`SI_ASYNCIO`信号。最后，我们可以输出我们已读取的内容并将其与文件内容进行比较。通过这样做，我们可以确保所有数据都已正确读取。

这就是 Linux 系统中的异步 I/O 的全部内容。

#### 注意

您可以通过访问 Linux 的 man 页面了解有关 Linux 中异步 IO 的更多信息：http://man7.org/linux/man-pages/man7/aio.7.html。

现在，让我们了解一下我们可以用于跨平台应用的内容。

### 异步跨平台 I/O 库

我们已经考虑了特定于平台的异步 I/O 的决定。现在，要编写一个跨平台应用程序，您可以使用这些特定于平台的方法，并将它们与预处理器指令一起使用；例如：

```cpp
#ifdef WIN
#include <WinAIO.hpp>
#else
#include <LinAIO.hpp>
#endif
```

在这两个头文件中，您可以为特定于平台的实现声明相同的接口。您还可以实现自己的 AIO 库，该库将在单独的线程中使用一些状态机或队列。此外，您可以使用一些实现所需功能的免费库。最流行的库是`Boost.Asio`。它提供了许多用于异步工作的接口，例如以下内容：

+   无需线程的并发

+   线程

+   缓冲区

+   流

+   协程

+   TCP、UDP 和 ICMP

+   套接字

+   SSL

+   定时器

+   串口

让我们简要地考虑一下它的 I/O 操作接口。我们可以使用`Asio`库的接口进行同步和异步操作。所有 I/O 操作都始于`io_service`类，该类提供核心 I/O 功能。它在`<boost/asio/io_service.hpp>`头文件中声明。同步 I/O 调用`io_service`对象的`run()`函数进行单个操作，该操作会阻塞调用线程，直到工作完成。异步 I/O 使用`run()`、`run_one()`、`poll()`和`poll_one()`函数。`run()`函数运行事件循环以处理请求处理程序。`run_one()`函数执行相同的操作，但事件循环只处理一个处理程序。`poll()`函数运行事件循环以执行所有准备好的处理程序。`poll_one()`执行相同的操作，但只针对一个处理程序。以下示例演示了所有这些函数的用法：

```cpp
boost::asio::io_service io_service1;
io_service1.run();
boost::asio::io_service io_service2;
io_service2.run_one();
boost::asio::io_service io_service3;
io_service3.poll();
boost::asio::io_service io_service4;
io_service4.poll_one();
```

在实际进行 I/O 操作之前，可以运行事件处理程序。使用`io_service`类的工作类在代码中实现此功能。工作类保证`run`函数在您决定不会有任何未来的 I/O 操作之前不会返回。例如，您可以将工作类作为另一个类的成员，并在析构函数中将其移除。因此，在您的类的生命周期内，`io_service`将一直运行：

```cpp
boost::asio::io_service io_service1;
boost::asio::io_service::work work(io_service1);
io_service1.run();
boost::asio::io_service io_service2;
boost::asio::io_service::work work(io_service2);
io_service2.poll();
```

接下来，要执行任何 I/O 操作，我们需要确切的 I/O 设备，例如文件、套接字等。有许多类实现了与不同 I/O 设备的工作，例如`<boost/asio/ip/tcp.hpp>`头文件中的`boost::asio::ip::tcp::socket`。接下来，要读取和写入套接字，我们使用`boost::asio::async_read`和`boost::asio::async_write`。它们将套接字、`boost::asio::buffer`和回调函数作为参数。执行异步操作时，将调用回调函数。我们可以将 lambda 函数作为回调函数传递，也可以使用 boost::bind 函数绑定现有函数。`boost::bind`创建一个可调用对象。以下示例演示了如何使用`Boost::Asio`写入套接字：

```cpp
boost::asio::io_service ioService;
tcp::socket socket;
int length = 15;
char* msg = new char(length);
msg = "Hello, world!";
auto postHandler = [=]()
{
     auto writeHandler = =
     {
          if (ec)
          {
               socket_.close();
          }
          else
          {
               // wrote length characters
          }
     };
     boost::asio::async_write(socket, boost::asio::buffer(msg, length), writeHandler);
};
ioService.post(postHandler);
```

在这里，我们使用 lambda 函数作为异步 I/O 操作的回调函数。

#### 注意

`Boost.Asio`在 https://www.boost.org/doc/libs/1_63_0/doc/html/boost_asio.html 上有很好的文档。有许多不同 IO 设备和不同方法的示例。如果您决定在项目中使用`Boost.Asio`，可以参考此文档。

在这里，我们考虑了实现异步 I/O 操作的不同方式。根据您的要求、环境和允许的实用程序，您可以选择适当的方式在应用程序中实现异步 I/O。请记住，如果选择执行许多快速 I/O 操作，最好以同步方式执行，因为它不会占用大量系统资源。现在我们知道如何利用异步 I/O，让我们学习如何在多线程应用程序中使用 I/O。

### 线程和 I/O 的交互

I/O 标准库不是线程安全的。在标准库的文档中，我们可以找到一个解释，说明并发访问流或流缓冲区可能导致数据竞争，从而导致未定义的行为。为了避免这种情况，我们应该使用我们在*第五章*，*哲学家的晚餐-线程和并发性*中学到的技术来同步对流和缓冲区的访问。

让我们稍微谈谈`std::cin`和`std::cout`对象。对它们的每次调用都是线程安全的，但让我们考虑以下例子：

```cpp
std::cout << "Counter: " << counter << std::endl;
```

在这一行中，我们看到`std::cout`只被调用一次，但每次对移位运算符的调用实际上是对`std::cout`对象的不同调用。因此，我们可以将这一行重写如下：

```cpp
std::cout << "Counter: ";
std::cout << counter;
std::cout << std::endl;
```

这段代码与前面的单行代码完全相同，也就是说，如果您从不同的线程调用这个单行代码，您的输出将混合在一起，不清晰。您可以修改它以使其真正线程安全，如下所示：

```cpp
std::stringsream ss;
ss << "Counter: " << counter << std::endl;
std::cout << ss.str();
```

因此，如果您使用第二种方法向终端输出，您的输出将清晰且线程安全。这种行为可能会有所不同，具体取决于编译器或 std 库版本。您还必须知道`std::cout`和`std::cin`在它们之间是同步的。这意味着调用`std::cout`总是刷新`std::cin`流，调用`std::cin`总是刷新`std::cout`流。

最好的方法是将所有 I/O 操作封装在一个保护类中，该类将使用互斥锁控制对流的访问。如果您需要从多个线程使用`std::cout`输出到终端，您可以实现一个非常简单的类，它除了锁定互斥锁并调用`std::cout`之外什么也不做。让我们完成一个练习并创建这样的类。

### 练习 9：为 std::cout 开发一个线程安全的包装器

在这个练习中，我们将开发一个简单的`std::cout`包装器，用于生成线程安全的输出。我们将编写一个小的测试函数来检查它的工作原理。让我们开始并执行以下步骤：

1.  包括所有必需的头文件：

```cpp
#include <iostream> // for std::cout
#include <thread>   // for std::thread
#include <mutex>    // for std::mutex
#include <sstream>  // for std::ostringstream
```

现在，让我们考虑一下我们的包装器。我们可以在某个地方创建这个类的变量，并将其传递给每个创建的线程。然而，这是一个不好的决定，因为在复杂的应用程序中，这将需要大量的工作。我们也可以将其作为单例来做，这样我们就可以从任何地方访问它。接下来，我们必须考虑我们的类的内容。实际上，我们可以使用我们在*练习 7*中创建的类，*继承标准流对象*。在那个练习中，我们重载了`std::basic_streambuf`和`std::basic_ostream`，并将`std::cout`设置为输出设备。我们可以在重载函数中添加一个互斥锁并直接使用它。请注意，我们不需要任何额外的逻辑-只需使用`std::cout`输出数据。为此，我们可以创建一个更简单的类。如果我们没有设置输出设备，应用左移运算符将不会生效，并且将存储要输出的数据到内部缓冲区。太好了！现在，我们需要考虑如何将这个缓冲区输出到`std::cout`。

1.  实现一个诸如`write()`的函数，它将锁定互斥锁并从内部缓冲区输出到`std::cout`。使用这个函数的方式将如下所示：

```cpp
mtcout cout;
cout << msg << std::endl;
cout.write();
```

1.  我们有一个函数将始终自动调用，并且我们可以将写函数的代码放入其中。这是一个析构函数。在这种情况下，我们将创建和销毁合并为一行。这样一个对象的使用将如下所示：

```cpp
mtcout{} << msg << std::endl; 
```

1.  现在，让我们定义我们的`mtcout`（多线程 cout）类。它有一个公共默认构造函数。在私有部分，它有一个静态互斥变量。正如你可能记得的那样，静态变量在类的所有实例之间是共享的。在析构函数中，我们锁定互斥锁并使用 cout 输出。在输出中添加一个前缀-当前线程的 ID 和一个空格字符：

```cpp
class mtcout : public std::ostringstream
{
public:
     mtcout() = default;
     ~mtcout()
     {
     std::lock_guard<std::mutex> lock(m_mux);
          std::cout << std::this_thread::get_id() << " " << this->str();
     }
private:
     static std::mutex m_mux;
};
```

1.  接下来，在类外声明`mutex`变量。我们这样做是因为我们必须在任何源文件中声明一个静态变量：

```cpp
std::mutex mtcout::m_mux; 
```

1.  输入主函数。创建一个名为`func`的 lambda。它将测试我们的`mtcout`类。它以字符串作为参数，并在循环中使用`mtcout`从`0`到`1000`输出这个字符串。使用`std::cout`添加相同的输出并将其注释掉。比较两种情况下的输出：

```cpp
auto func = [](const std::string msg)
{
     using namespace std::chrono_literals;
     for (int i = 0; i < 1000; ++i)
     {
          mtcout{} << msg << std::endl;
//          std::cout << std::this_thread::get_id() << " " << msg << std::endl;
     }
};
```

1.  创建四个线程并将 lambda 函数作为参数传递。将不同的字符串传递给每个线程。最后，加入所有四个线程：

```cpp
std::thread thr1(func, "111111111");
std::thread thr2(func, "222222222");
std::thread thr3(func, "333333333");
std::thread thr4(func, "444444444");
thr1.join();
thr2.join();
thr3.join();
thr4.join();
```

1.  首次构建和运行练习。您将获得以下输出：![图 6.18：执行练习 9，第 1 部分的结果](img/C14583_06_18.jpg)

###### 图 6.18：执行练习 9，第 1 部分的结果

在这里，我们可以看到每个线程都输出自己的消息。这条消息没有被中断，输出看起来很清晰。

1.  现在，取消 lambda 中使用`std::cout`的输出，并注释掉使用`mtcout`的输出。

1.  再次构建和运行应用程序。现在，您将获得一个"脏"的、混合的输出，如下所示：![图 6.19：执行练习 9，第 2 部分的结果](img/C14583_06_19.jpg)

###### 图 6.19：执行练习 9，第 2 部分的结果

您可以看到这种混合输出，因为我们没有输出单个字符串；相反，我们调用`std::cout`四次：

```cpp
std::cout << std::this_thread::get_id();
std::cout << " ";
std::cout << msg;
std::cout << std::endl;
```

当然，我们可以在输出之前格式化字符串，但使用 mtcout 类更方便，不必担心格式。您可以为任何流创建类似的包装器，以便安全地执行 I/O 操作。您可以更改输出并添加任何其他信息，例如当前线程的 ID、时间或您需要的任何其他信息。利用我们在*第五章*中学到的关于同步 I/O 操作、扩展流并使输出对您的需求更有用的东西。

### 使用宏

在本章的活动中，我们将使用宏定义来简化和美化我们的代码，所以让我们回顾一下如何使用它们。宏定义是预处理器指令。宏定义的语法如下：

```cpp
#define [name] [expression]
```

在这里，[name]是任何有意义的名称，[expression]是任何小函数或值。

当预处理器面对宏名称时，它将其替换为表达式。例如，假设您有以下宏：

```cpp
#define MAX_NUMBER 15
```

然后，在代码中的几个地方使用它：

```cpp
if (val < MAX_NUMBER)
while (val < MAX_NUMBER)
```

当预处理器完成其工作时，代码将如下所示：

```cpp
if (val < 15)
while (val < 15)
```

预处理器对函数执行相同的工作。例如，假设您有一个用于获取最大数的宏：

```cpp
#define max(a, b) a < b ? b : a
```

然后，在代码中的几个地方使用它：

```cpp

int res = max (5, 3);

std::cout << (max (a, b));
```

当预处理器完成其工作时，代码将如下所示：

```cpp

int res = 5 < 3 ? 3 : 5;

std::cout << (a < b ? b : a);

```

作为表达式，您可以使用任何有效的表达式，比如函数调用、内联函数、值等。如果您需要在多行中编写表达式，请使用反斜杠运算符"\"。例如，我们可以将 max 定义写成两行，如下所示：

```cpp
#define max(a, b) \
a < b ? b : a
```

#### 注意

宏定义来自 C 语言。最好使用 const 变量或内联函数。然而，仍然有一些情况下使用宏定义更方便，例如在记录器中定义不同的记录级别时。

现在。我们知道完成这个活动所需的一切。所以，让我们总结一下我们在本章学到的东西，并改进我们在*第五章*中编写的项目，*哲学家的晚餐-线程和并发性*。我们将开发一个线程安全的记录器，并将其集成到我们的项目中。

### 活动 1：艺术画廊模拟器的日志系统

在这个活动中，我们将开发一个记录器，它将以格式化的形式输出日志到终端。我们将以以下格式输出日志：

```cpp
[dateTtime][threadId][logLevel][file:line][function] | message
```

我们将为不同的日志级别实现宏定义，这些宏定义将用于替代直接调用。这个记录器将是线程安全的，并且我们将同时从不同线程调用它。最后，我们将把它集成到项目中——美术馆模拟器中。我们将运行模拟并观察漂亮打印的日志。我们将创建一个额外的流，使用并发流，并格式化输出。我们将几乎实现本章中学到的所有内容。我们还将使用上一章的同步技术。

因此，在尝试此活动之前，请确保您已完成本章中的所有先前练习。

在实现此应用程序之前，让我们描述一下我们的类。我们有以下新创建的类：

![图 6.20：应该实现的类的描述](img/C14583_06_20.jpg)

###### 图 6.20：应该实现的类的描述

我们在美术馆模拟器项目中已经实现了以下类：

![](img/C14583_06_21.jpg)

###### 图 6.21：美术馆模拟器项目中已实现的类的表格

在开始实现之前，让我们将新的类添加到类图中。所有描述的类及其关系都组成了以下图表：

![图 6.22：类图](img/C14583_06_22.jpg)

###### 图 6.22：类图

为了以期望的格式接收输出，`LoggerUtils`类应该具有以下`static`函数：

![图 6.23：LoggerUtils 成员函数的描述](img/C14583_06_23.jpg)

###### 图 6.23：LoggerUtils 成员函数的描述

按照以下步骤完成此活动：

1.  定义并实现`LoggerUtils`类，提供输出格式化的接口。它包含将给定数据格式化为所需表示形式的静态变量。

1.  定义并实现`StreamLogger`类，为终端提供线程安全的输出接口。它应该格式化输出如下：

```cpp
[dateTtime][threadId][logLevel][file:line: ][function] | message
```

1.  在一个单独的头文件中，声明不同日志级别的宏定义，返回`StreamLogger`类的临时对象。

1.  将实现的记录器集成到美术馆模拟器的类中。

1.  用适当的宏定义调用替换所有`std::cout`的调用。

在实施了上述步骤之后，您应该在终端上获得有关所有实现类的日志的输出。查看并确保日志以期望的格式输出。预期输出应该如下：

![图 6.24：应用程序执行的结果](img/C14583_06_24.jpg)

###### 图 6.24：应用程序执行的结果

#### 注意

此活动的解决方案可在第 696 页找到。

## 总结

在本章中，我们学习了 C++中的 I/O 操作。我们考虑了 I/O 标准库，它提供了同步 I/O 操作的接口。此外，我们考虑了与平台相关的异步 I/O 的本机工具，以及`Boost.Asio`库用于跨平台异步 I/O 操作。我们还学习了如何在多线程应用程序中使用 I/O 流。

我们首先看了标准库为 I/O 操作提供的基本功能。我们了解了预定义的流对象，如`std::cin`和`std::cout`。在实践中，我们学习了如何使用标准流并重写移位运算符以便轻松读取和写入自定义数据类型。

接下来，我们练习了如何创建额外的流。我们继承了基本流类，实现了自己的流缓冲区类，并练习了它们在练习中的使用。我们了解了最适合继承的流类，并考虑了它们的优缺点。

然后，我们考虑了不同操作系统上异步 I/O 操作的方法。我们简要考虑了使用跨平台 I/O 库 Boost.Asio，该库提供了同步和异步操作的接口。

最后，我们学习了如何在多线程应用程序中执行 I/O 操作。我们将所有这些新技能付诸实践，通过构建一个多线程日志记录器。我们创建了一个日志记录抽象，并在艺术画廊模拟器中使用它。结果，我们创建了一个简单、清晰、健壮的日志记录系统，可以通过日志轻松调试应用程序。总之，我们在本章中运用了我们学到的一切。

在下一章中，我们将更仔细地学习应用程序的测试和调试。我们将首先学习断言和安全网。然后，我们将练习编写接口的单元测试和模拟。之后，我们将在 IDE 中练习调试应用程序：我们将使用断点、观察点和数据可视化。最后，我们将编写一个活动，来掌握我们的代码测试技能。
