# 第十章：诊断和调试

软件是复杂的；无论你设计代码有多好，总有一天你将不得不调试它，无论是在开发代码的正常测试阶段还是在发出错误报告时。最好设计代码，使测试和调试尽可能简单直接。这意味着添加跟踪和报告代码，确定不变式和前后条件，以便你有一个测试代码的起点，并编写具有可理解和有意义的错误代码的函数。

# 准备你的代码

C++和 C 标准库有各种函数，允许你应用跟踪和报告函数，以便你可以测试代码是否以预期的方式处理数据。许多这些功能使用条件编译，以便报告仅在调试构建中发生，但如果你提供有意义的消息，它们将成为你代码的一部分文档。在你可以报告代码的行为之前，你首先必须知道从中期望什么。

# 不变式和条件

类不变式是条件，对象状态，你知道保持不变。在方法调用期间，对象状态将发生变化，可能变为使对象无效的东西，但一旦公共方法完成，对象状态必须保持一致。用户调用类的方法的顺序没有保证，甚至他们是否调用方法，因此对象必须可以使用无论用户调用哪些方法。对象的不变方面适用于方法调用级别：在方法调用之间，对象必须保持一致和可用。

例如，想象一下你有一个代表日期的类：它保存了 1 到 31 之间的日期，1 到 12 之间的月份，以及年份。类不变式是，无论你对日期类的对象做什么，它始终保持有效的日期。这意味着用户可以安全地使用你的日期类的对象。这也意味着类的其他方法（比如，确定两个日期之间有多少天的方法，`operator-`）可以假定日期对象中的值是有效的，因此这些方法不必检查它们所作用的数据的有效性。

然而，一个有效的日期不仅仅是 1 到 31 的范围和 1 到 12 的月份，因为并非每个月都有 31 天。因此，如果你有一个有效的日期，比如 1997 年 4 月 5 日，然后你调用`set_day`方法将日期设置为 31 号，那么类不变条件就被违反了，因为 4 月 31 日不是一个有效的日期。如果你想改变日期对象中的值，唯一安全的方法是同时改变所有的值：日期、月份和年份，因为这是保持类不变性的唯一方法。

一种方法是在调试构建中定义一个私有方法，测试类的不变条件，并确保使用断言（稍后见）维护不变条件。你可以在公开可访问的方法离开之前调用这样的方法，以确保对象保持一致状态。方法还应该有定义的前后条件。前置条件是在调用方法之前你要求为真的条件，后置条件是在方法完成后你保证为真的条件。对于类的方法，类不变式是前置条件（因为在调用方法之前对象的状态应该是一致的），不变式也是后置条件（因为方法完成后对象状态应该是一致的）。

还有一些是方法调用者的先决条件。先决条件是调用者确保的一个已记录的责任。例如，日期类将有一个先决条件，即日期数字在 1 和 31 之间。这简化了类代码，因为接受日期数字的方法可以假定传递的值永远不会超出范围（尽管由于某些月份少于 31 天，值可能仍然无效）。同样，在调试构建中，您可以使用断言来检查这些先决条件是否为真，并且断言中的测试将在发布构建中被编译掉。在方法的末尾将有后置条件，即将保持类不变量（并且对象的状态将有效），并且返回值将有效。

# 条件编译

如第一章中所述，*从 C++开始*，当编译您的 C++程序时，有一个预编译步骤，将 C++源文件中包含的所有文件汇总到一个单个文件中，然后进行编译。预处理器还会展开宏，并根据符号的值包含一些代码和排除其他代码。

在其最简单的形式中，条件编译使用`#ifdef`和`#endif`（可选使用`#else`）将代码括在其中，因此只有在指定的符号已定义时才编译这些指令之间的代码。

```cpp
    #ifdef TEST 
       cout << "TEST defined" << endl;     
    #else 
       cout << "TEST not defined" << endl; 
    #endif
```

您可以确保只编译这些行中的一个，并且至少会编译其中一个。如果定义了符号`TEST`，则将编译第一行，并且在编译器看来，第二行不存在。如果未定义符号`TEST`，则将编译第二行。如果要以相反的顺序输入这些行，可以使用`#ifndef`指令。通过条件编译提供的文本可以是 C++代码，也可以使用当前翻译单元中的其他符号使用`#define`定义，或者使用`#undef`取消定义现有符号。

`#ifdef`指令只是确定符号是否存在：它不测试其值。`#if`指令允许您测试一个表达式。您可以设置一个符号具有一个值，并根据该值编译特定的代码。表达式必须是整数，因此单个`#if`块可以使用`#if`和多个`#elif`指令测试多个值，并且最多一个`#else`：

```cpp
    #if TEST < 0 
       cout << "negative" << endl; 
    #elif TEST > 0 
       cout << "positive" << endl; 
    #else 
       cout << "zero or undefined" << endl; 
    #endif
```

如果未定义符号，则`#if`指令将该符号视为具有值`0`；如果要区分这些情况，可以使用`defined`运算符来测试符号是否已定义。最多只有`#if`/`#endif`块中的一个部分将被编译，如果值不匹配，则不会编译任何代码。表达式可以是宏，此时在测试条件之前将展开该宏。

有三种定义符号的方法。第一种方法是无法控制的：编译器将定义一些符号（通常带有`__`或`_`前缀），这些符号会提供有关编译器和编译过程的信息。其中一些符号将在后面的部分中描述。另外两种方法完全在您的控制之下-您可以在源文件（或头文件）中使用`#define`定义符号，也可以使用`/D`开关在命令行上定义它们：

```cpp
    cl /EHsc prog.cpp /DTEST=1
```

这将使用值为`1`的符号`TEST`编译源代码。

通常，您将使用条件编译来提供不应在生产代码中使用的代码，例如，在调试模式或测试代码时使用的额外跟踪代码。例如，假设您有库代码来从数据库返回数据，但是您怀疑库函数中的 SQL 语句有错误并返回了太多的值。在这种情况下，您可能决定添加代码来记录返回的值的数量：

```cpp
    vector<int> data = get_data(); 
    #if TRACE_LEVEL > 0 
    cout << "number of data items returned: " << data.size() << endl; 
    #endif
```

这样的跟踪消息会污染您的用户界面，并且您希望在生产代码中避免它们。但是，在调试中，它们可以帮助您确定问题发生的位置。

在调试模式下调用的任何代码，条件代码应该是`const`方法（这里是`vector::size`），也就是说，它们不应该影响任何对象或应用程序数据的状态。您必须确保您的代码在调试模式和发布模式下的逻辑*完全*相同。

# 使用 pragma

Pragma 是特定于编译器的，并且通常涉及对象文件中代码部分的技术细节。在调试代码中，有一些 Visual C++ pragma 非常有用。

一般来说，您希望您的代码尽可能少地编译警告。Visual C++编译器的默认警告是`/W1`，这意味着只列出最严重的警告。将值逐渐增加到 2、3 或最高值 4 会增加编译过程中给出的警告数量。使用`/Wall`将会给出级别 4 的警告和默认情况下已禁用的警告。即使对于最简单的代码，最后一个选项也会产生一屏幕的警告。当您有数百个警告时，有用的错误消息将被隐藏在大量不重要的警告之间。由于 C++标准库非常复杂，并且使用了一些几十年前的代码，编译器会对一些构造发出警告。为了防止这些警告污染构建输出，特定文件中的特定警告已被禁用。

如果您支持旧的库代码，您可能会发现代码编译时会出现警告。您可能会想要使用编译器的`/W`开关来降低警告级别，但这将抑制所有高于您启用的警告，并且它同样适用于您的代码和您可能包含到项目中的库代码。`warning` pragma 给了您更多的灵活性。有两种调用方式--您可以重置警告级别以覆盖编译器的`/W`开关，也可以更改特定警告的警告级别或完全禁用警告报告。

例如，在`<iostream>`头文件的顶部是这行：

```cpp
    #pragma warning(push,3)
```

这表示存储当前的警告级别，并在文件的其余部分（或直到更改为止）将警告级别设置为 3。文件底部是这行：

```cpp
    #pragma warning(pop)
```

这将恢复到先前存储的警告级别。

您还可以更改一个或多个警告的报告方式。例如，在`<istream>`的顶部是：

```cpp
    #pragma warning(disable: 4189)
```

这个`pragma`的第一部分是指示报告警告类型（在本例中为 4189）已被禁用的`disable`。如果您愿意，您可以使用警告级别（`1`、`2`、`3`或`4`）作为指示符来更改警告的警告级别。其中一个用途是在您正在处理的一段代码中降低警告级别，然后在代码之后将其恢复到默认级别。例如：

```cpp
    #pragma warning(2: 4333) 
    unsigned shift8(unsigned char c)  
    { 
        return c >> 8;  
    } 
    #pragma warning(default: 4333)
```

这个函数将 char 右移 8 位，这将生成级别 1 警告 4333（*右移位数过大，数据丢失*）。这是一个问题，需要修复，但目前，您希望编译代码时不会收到来自此代码的警告，因此将警告级别更改为级别 2。使用默认警告级别（`/W1`），不会显示警告。但是，如果使用更敏感的警告级别进行编译（例如，`/W2`），则会报告此警告。警告级别的更改仅是临时的，因为最后一行将警告级别重置为其默认值（即 1）。在这种情况下，警告级别增加，这意味着只有在编译器上使用更敏感的警告级别时才会看到它。您还可以降低警告级别，这意味着更有可能报告警告。您甚至可以将警告级别更改为`error`，以便在代码中存在此类型的警告时无法编译。

# 添加信息性消息

在测试和调试代码时，您不可避免地会遇到一些潜在问题，但与您正在处理的问题相比，它的优先级较低。重要的是要记录问题，以便以后可以解决问题。在 Visual C++中，有两种以温和的方式记录问题的方法，还有两种会生成错误的方法。

第一种方法是添加一个`TODO:`注释，如下所示：

```cpp
    // TODO: potential data loss, review use of shift8 function 
    unsigned shift8(unsigned char c)  
    { 
        return c >> 8;  
    }
```

Visual Studio 编辑器有一个名为**任务列表**的工具窗口。这列出了项目中以预定任务之一开头的注释（默认为`TODO`、`HACK`和`UNDONE`）。

如果任务列表窗口不可见，请通过“视图”菜单启用它。Visual Studio 2015 中的默认设置是启用 C++中的任务。对于早期版本来说并非如此，但可以通过“工具”菜单、“选项”对话框，然后“文本编辑器”、“C/C++”、“格式”、“查看”设置“枚举注释任务”为是来启用。任务标签列表可以在“环境”、“任务列表”项目下的“选项”对话框中找到。

任务列表列出了文件和行号的任务，您可以通过双击条目来打开文件并定位注释。

识别需要注意的代码的第二种方法是`message`指示。顾名思义，这只是允许您在代码中放置信息性消息。当编译器遇到此指示时，它只是将消息放在输出流中。考虑以下代码：

```cpp
    #pragma message("review use of shift8 function") 
    unsigned shift8(unsigned char c)  
    { 
        return c >> 8;  
    }
```

如果`test.cpp`文件使用此代码和`/W1`（默认）警告级别进行编译，输出将类似于以下内容：

```cpp
 Microsoft (R) C/C++ Optimizing Compiler Version 19.00.24215.1 for x86
Copyright (C) Microsoft Corporation.  All rights reserved.

test.cpp
review the use of shift8 function
test.cpp(8): warning C4333: '>>': right shift by too large amount, data loss
```

正如您所看到的，字符串是按照编译器看到的方式打印的，并且与警告消息相比，没有文件或行号的指示。有方法可以使用编译器符号来解决这个问题。

如果条件很重要，您会想要发出一个错误，而一种方法是使用`#error`指令。当编译器到达这个指令时，它会发出一个错误。这是一个严重的行为，所以只有在没有其他选择时才会使用它。您很可能会想要将其与条件编译一起使用。典型的用法是只能使用 C++编译器编译的代码：

```cpp
    #ifndef __cplusplus 
    #error C++ compiler required. 
    #endif
```

如果使用`/Tc`开关编译带有此代码的文件以将代码编译为 C，则将不会定义`__cplusplus`预处理符号，并将生成错误。

C++11 添加了一个名为`static_assert`的新指令。这类似于函数调用（*调用*以分号结束），但它不是函数，因为它只在编译时使用。此外，该指令可以在不使用函数调用的地方使用。该指令有两个参数：一个表达式和一个字符串文字。如果表达式为`false`，则字符串文字将在编译时与源文件和行号一起输出，并生成错误。在最简单的级别上，您可以使用它来发出消息：

```cpp
    #ifndef __cplusplus 
    static_assert(false, "Compile with /TP"); 
    #endif 
    #include <iostream> // needs the C++ compiler
```

由于第一个参数是`false`，指令将在编译期间发出错误消息。使用`#error`指令也可以实现相同的效果。`<type_traits>`库具有用于测试类型属性的各种谓词。例如，`is_class`模板类具有一个简单的模板参数，即类型，如果该类型是一个`class`，则`static`成员`value`设置为`true`。如果您有一个应该只对类进行实例化的模板化函数，您可以添加这个`static_assert`：

```cpp
    #include <type_traits> 

    template <class T> 
    void func(T& value) 
    { 
        static_assert(std::is_class<T>::value, "T must be a class"); 
        // other code 
    }
```

在编译时，编译器将尝试实例化函数，并使用`value`在该类型上实例化`is_class`，以确定编译是否应该继续。例如，以下代码：

```cpp
    func(string("hello")); 
    func("hello");
```

第一行将正确编译，因为编译器将实例化一个函数`func<string>`，参数是一个`class`。然而，第二行将无法编译，因为实例化的函数是`func<const char*>`，而`const char*`不是一个`class`。输出是：

```cpp
Microsoft (R) C/C++ Optimizing Compiler Version 19.00.24215.1 for x86
Copyright (C) Microsoft Corporation.  All rights reserved.

test.cpp
test.cpp(25): error C2338: T must be a class
test.cpp(39): note: see reference to function template instantiation 

'void func<const char*>(T)' being compiled
with
[
 T=const char *
]
```

`static_assert`在*第 25 行*，因此会生成错误，即`T 必须是一个 class`。*第 39 行*是对`func<const char*>`的第一次调用，并为错误提供了上下文。

# 调试的编译器开关

为了让调试器能够逐步执行程序，您必须提供信息，以便调试器将机器代码与源代码关联起来。至少，这意味着关闭所有优化，因为在尝试优化代码时，C++编译器会重新排列代码。优化默认关闭（因此使用`/Od`开关是多余的），但显然，为了能够调试进程并逐步执行 C++代码，您需要删除所有`/O`优化开关。

由于 C++标准库使用 C 运行时，您需要编译您的代码以使用后者的调试版本。您使用的开关取决于您是构建进程还是**动态链接库**（**DLL**），以及您是静态链接 C 运行时还是通过 DLL 访问它。如果您正在编译一个进程，您可以使用`/MDd`来获取 DLL 中 C 运行时的调试版本，如果您使用`/MTd`，您将获得静态链接 C 运行时的调试版本。如果您正在编写一个动态链接库，您必须使用`/LDd`，并且还要使用一个 C 运行时开关（`/MTd`是默认值）。这些开关将定义一个名为`_DEBUG`的预处理器符号。

调试器需要知道调试器符号信息--变量的名称和类型，函数的名称和与代码相关联的行号。通过名为**程序数据库**的文件来完成这一点，其扩展名为`pdb`。您可以使用`/Z`开关之一来生成`pdb`文件：`/Zi`或`/ZI`开关将创建两个文件，一个文件名以`VC`开头（例如`VC140.pdb`），其中包含所有`obj`文件的调试信息，另一个文件名为项目名称，其中包含进程的调试信息。如果您编译而不链接（`/c`），则只会创建第一个文件。Visual C++项目向导默认使用`/Od /MDd /ZI`来进行调试构建。`/ZI`开关意味着以一种允许 Visual C++调试器执行`Edit`和`Continue`的格式创建程序数据库，也就是说，您可以更改一些代码并继续逐步执行代码，而无需重新编译。当您为发布构建编译时，向导将使用`/O2 /MD /Zi`开关，这意味着代码经过了速度优化，但仍将创建一个程序数据库（不支持`Edit`和`Continue`）。代码不需要程序数据库来运行（实际上，您不应该将其与代码一起分发），但如果您有崩溃报告并需要在调试器下运行发布构建代码，它将非常有用。

这些`/Z`编译器开关假定链接器使用`/debug`开关运行（如果编译器调用链接器，它将传递此开关）。链接器将从`VC`程序数据库文件中的调试信息创建项目程序数据库。

这引发了一个问题，即为什么发布构建文件需要一个程序数据库。如果在调试器下运行程序并查看调用堆栈，通常会看到操作系统文件中的一长串堆栈帧。这些通常具有由 DLL 名称和一些数字和字符组成的相当无意义的名称。可以安装 Windows 的符号（`pdb`文件），或者如果它们未安装，则指示 Visual C++调试器从网络上的计算机上下载正在使用的库的符号，称为**符号服务器**。这些符号不是库的源代码，但它们确实为您提供了函数的名称和参数的类型，这为您提供了有关在您单步执行时调用堆栈状态的附加信息。

# 预处理器符号

要在代码中使用跟踪、断言和报告功能，必须启用调试运行时库，方法是使用`/MDd`、`/MTd`或`/LDd`编译器开关，这将定义`_DEBUG`预处理器符号。`_DEBUG`预处理器符号启用了许多功能，反之，不定义此符号将有助于优化代码。

```cpp
    #ifdef _DEBUG 
       cout << "debug build" << endl; 
    #else 
       cout << "release built" << endl; 
    #endif
```

C++编译器还将通过一些标准预处理器符号提供信息。其中大多数仅对库编写者有用，但也有一些可能会用到。

ANSI 标准规定，当编译器编译代码为 C++（而不是 C）时，应定义`__cplusplus`符号，并且还指定`__FILE__`符号应包含文件名，`__LINE__`符号将在访问它的地方具有行号。`__func__`符号将具有当前函数名称。这意味着您可以创建以下跟踪代码：

```cpp
    #ifdef _DEBUG 
    #define TRACE cout << __func__ << " (" << __LINE__ << ")" << endl; 
    #else 
    #define TRACE 
    #endif
```

如果此代码编译用于调试（例如，`/MTd`），则`cout`行将在使用`TRACE`时内联；如果代码未编译用于调试，则`TRACE`将不起作用。`__func__`符号只是函数名称，它没有限定，因此如果在类方法中使用它，它将不提供有关类的任何信息。

Visual C++还定义了 Microsoft 特定的符号。`__FUNCSIG__`符号提供完整的签名，包括类名（和任何`namespace`名称）、返回类型和参数。如果只想要完全限定的名称，那么可以使用`__FUNCTION__`符号。在 Windows 头文件中经常看到的一个符号是`_MSC_VER`。它具有当前 C++编译器的版本号，并且与条件编译一起使用，以便只有支持它们的编译器才会编译新的语言特性。

Visual C++项目页面定义了类似`$(ProjectDir)`和`$(Configuration)`的*构建宏*。这些仅由 MSBuild 工具使用，因此在编译过程中源文件中并不自动可用，但是，如果将预处理器符号设置为构建宏的值，则该值将在编译时通过该符号可用。系统环境变量也可用作构建宏，因此可以使用它们来影响构建。例如，在 Windows 上，系统环境变量`USERNAME`具有当前登录用户的名称，因此可以使用它来设置一个符号，然后在编译时访问它。

在 Visual C++项目页面上，您可以在 C/C++预处理器项目页面上添加一个名为**预处理器定义**的定义：

```cpp
    DEVELOPER="$(USERNAME)"
```

然后，在您的代码中，您可以添加一行使用此符号：

```cpp
    cout << "Compiled by " << DEVELOPER << endl;
```

如果你正在使用一个 make 文件，或者只是从命令行调用`cl`，你可以添加一个开关来定义这个符号，就像这样：

```cpp
    /DDEVELOPER="$(USERNAME)"
```

在这里转义双引号很重要，因为没有它们，引号会被编译器吞掉。

之前，你看到了`#pragma message`和`#error`指令如何用于将消息放入编译器的输出流中。在 Visual Studio 中编译代码时，编译器和链接器的输出将显示在输出窗口中。如果消息的形式是：

```cpp
    path_to_source_file(line) message
```

其中`path_to_source_file`是文件的完整路径，`line`是`message`出现的行号。然后，当你在输出窗口中双击这一行时，文件将被加载（如果尚未加载），并且插入点将放在该行上。

`__FILE__`和`__LINE__`符号为你提供了使`#pragma message`和`#error`指令更有用所需的信息。输出`__FILE__`很简单，因为它是一个字符串，C++会连接字符串字面量：

```cpp
    #define AT_FILE(msg) __FILE__ " " msg 

    #pragma message(AT_FILE("this is a message"))
```

该宏作为`#pragma`的一部分被调用以正确格式化消息；然而，你不能从宏中调用`#pragma`，因为`#`有一个特殊的目的（稍后将会用到）。这段代码的结果将类似于：

```cpp
    c:\Beginning_C++Chapter_10test.cpp this is a message
```

通过宏输出`__LINE__`需要更多的工作，因为它保存了一个数字。这个问题在 C 语言中很常见，所以有一个标准的解决方案，使用两个宏和字符串操作符`#`。

```cpp
    #define STRING2(x) #x 
    #define STRING(x) STRING2(x) 
    #define AT_FILE(msg) __FILE__ "(" STRING(__LINE__) ") " msg
```

`STRING`宏用于将`__LINE__`符号扩展为一个数字，`STRING2`宏用于将数字转换为字符串。`AT_FILE`宏以正确的格式格式化整个字符串。

# 生成诊断消息

诊断消息的有效使用是一个广泛的话题，所以本节只会给出基础知识。在设计代码时，你应该让编写诊断消息变得容易，例如，提供机制来转储对象的内容，并提供访问测试类不变量和前后条件的代码。你还应该分析代码，确保适当的消息被记录。例如，在循环中发出诊断消息通常会填满日志文件，使得难以阅读日志文件中的其他消息。然而，循环中一直出现失败可能本身就是一个重要的诊断，尝试执行失败操作的次数也可能是一个重要的诊断，因此你可能希望记录下来。

使用`cout`输出诊断消息的优点是将这些消息与用户输出集成在一起，这样你可以看到中间结果的最终效果。缺点是诊断消息与用户输出集成在一起，而且通常会有大量的诊断消息，这些消息会完全淹没程序的用户输出。

C++有两个流对象，你可以使用它们来代替`cout`。`clog`和`cerr`流对象将字符数据写入标准错误流（C 流指针`stderr`），通常会显示在控制台上，就像使用`cout`一样（它输出到标准输出流，C 流指针`stdout`），但你可以将其重定向到其他地方。`clog`和`cerr`之间的区别在于`clog`使用缓冲输出，这可能比未缓冲的`cerr`性能更好。然而，如果应用程序意外停止而没有刷新缓冲区，数据可能会丢失。

由于`clog`和`cerr`流对象在发布版本和调试版本中都可用，所以你应该只用它们来处理你的最终用户会看到的消息。这使它们不适合用于跟踪消息（稍后将介绍）。相反，你应该用它们来处理用户能够解决的诊断消息（也许找不到文件或者进程没有执行操作的安全访问权限）。

```cpp
    ofstream file; 
    if (!file.open(argv[1], ios::out)) 
    { 
        clog << "cannot open " << argv[1] << endl; 
        return 1; 
    }
```

此代码以两个步骤打开文件（而不是使用构造函数），如果文件无法打开，则`open`方法将返回`false`。代码检查是否成功打开文件，如果失败，它将通过`clog`对象告知用户，然后从包含代码的任何函数返回，因为`file`对象现在无效且无法使用。`clog`对象是有缓冲的，但在这种情况下，我们希望立即通知用户，这是由`endl`操作器执行的，它在流中插入换行然后刷新流。

默认情况下，`clog`和`cerr`流对象将输出到标准错误流，这意味着对于控制台应用程序，您可以通过重定向流来分离输出流和错误流。在命令行上，可以使用`stdin`的值为 0，`stdout`的值为 1，`stderr`的值为 2 以及重定向操作符`>`来重定向标准流。例如，一个名为`app.exe`的应用程序可以在`main`函数中包含以下代码：

```cpp
    clog << "clog" << endl; 
    cerr << "cerrn"; 
    cout << "cout" << endl;
```

`cerr`对象没有缓冲，因此无论您使用`n`还是`endl`来换行都无关紧要。当您在命令行上运行时，您会看到类似以下的内容：

```cpp
C:\Beginning_C++\Chapter_10>app
clog
cerr
cout
```

要将流重定向到文件，请将流句柄（`stdout`为 1，`stderr`为 2）重定向到文件；控制台将打开文件并将流写入文件：

```cpp
C:\Beginning_C++\Chapter_10>app 2>log.txt
cout

C:\Beginning_C++\Chapter_10>type log.txt
clog
cerr
```

正如上一章所示，C++流对象是分层的，因此向流中插入数据的调用将根据流的类型将数据写入底层流对象，有或没有缓冲。可以使用`rdbuf`方法获取和替换此流缓冲区对象。如果要将`clog`对象重定向到应用程序的文件中，可以编写以下代码：

```cpp
    extern void run_code(); 

    int main() 
    { 
        ofstream log_file; 
        if (log_file.open("log.txt")) clog.rdbuf(log_file.rdbuf()); 

        run_code(); 

        clog.flush(); 
        log_file.close(); 
        clog.rdbuf(nullptr); 
        return 0; 
    }
```

在此代码中，应用程序代码将位于`run_code`函数中，其余代码设置了`clog`对象以重定向到文件。

请注意，当`run_code`函数返回时（应用程序已完成），文件将被显式关闭；这并不完全必要，因为`ofstream`析构函数将关闭文件，并且在这种情况下，当`main`函数返回时将会发生这种情况。最后一行很重要。标准流对象在调用`main`函数之前创建，并且它们将在`main`函数返回后的某个时候被销毁，也就是说，在文件对象被销毁之后。为了防止`clog`对象访问已销毁的文件对象，将调用`rdbuf`方法并传递`nullptr`以指示没有缓冲区。

# 使用 C 运行时的跟踪消息

通常，您会希望通过实时运行应用程序并输出*跟踪消息*来测试您的代码，以测试您的算法是否有效。有时，您会希望测试函数的调用顺序（例如，在`switch`语句或`if`语句中正确分支的发生），在其他情况下，您会希望测试中间值，以确保输入数据正确并且对该数据的计算正确。

跟踪消息可能会产生大量数据，因此将这些消息发送到控制台是不明智的。非常重要的是，跟踪消息只在调试构建中生成。如果在产品代码中保留跟踪消息，可能会严重影响应用程序的性能（稍后将进行解释）。此外，跟踪消息不太可能被本地化，也不会被检查以查看它们是否包含可用于反向工程您的算法的信息。在发布构建中跟踪消息的另一个问题是，您的客户会认为您正在提供尚未完全测试的代码。因此，非常重要的是，只有在调试构建中生成跟踪消息时才定义`_DEBUG`符号。

C 运行时提供了一系列以`_RPT`开头的宏，可以在定义`_DEBUG`时用于跟踪消息。这些宏有`char`和宽字符版本，并且有一些版本仅报告跟踪消息，还有一些版本将报告消息和消息的位置（源文件和行号）。最终，这些宏将调用一个名为`_CrtDbgReport`的函数，该函数将使用在其他地方确定的设置生成消息。

`_RPTn`宏（其中`n`为`0`、`1`、`2`、`3`、`4`或`5`）将采用格式字符串和 0 到 5 个参数，这些参数将在报告之前放入字符串中。宏的第一个参数表示要报告的消息类型：`_CRT_WARN`、`_CRT_ERROR`或`_CRT_ASSERT`。这些类别中的最后两个是相同的，并且指的是断言，这将在后面的部分中介绍。报告宏的第二个参数是格式字符串，然后是所需数量的参数。`_RPTFn`宏的格式相同，但还将报告源文件和行号以及格式化的消息。

默认操作是，`_CRT_WARN`消息不会产生任何输出，而`_CRT_ERROR`和`_CRT_ASSERT`消息将生成一个弹出窗口，允许您中止或调试应用程序。您可以通过调用`_CrtSetReportMode`函数并提供类别和指示采取的操作的值来更改对这些消息类别中的任何一个的响应。如果使用`_CRTDBG_MODE_DEBUG`，则消息将被写入调试器输出窗口。如果使用`_CRTDBG_MODE_FILE`，则消息将被写入一个文件，您可以打开并将句柄传递给`_CrtSetReportFile`函数。（您还可以使用`_CRTDBG_FILE_STDERR`或`_CRTDBG_FILE_STDOUT`作为文件句柄，将消息发送到标准输出或错误输出。）如果将`_CRTDBG_MODE_WNDW`用作报告模式，则消息将使用中止/重试/忽略对话框显示。由于这将暂停当前执行线程，因此应仅用于断言消息（默认操作）：

```cpp
    include <crtdbg.h> 

    extern void run_code(); 

    int main() 
    { 
        _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG); 
        _RPTF0(_CRT_WARN, "Application startedn"); 

        run_code(); 

        _RPTF0(_CRT_WARN, "Application endedn"); 
        return 0; 
    }
```

如果在消息中不提供`n`，则下一条消息将附加到您的消息末尾，在大多数情况下，这不是您想要的（尽管您可以为对`_RPTn`宏的一系列调用辩解，其中最后一个以`n`终止）。

在编译项目时，Visual Studio 输出窗口会显示（要在调试时显示，请在“视图”菜单中选择“输出”选项），在顶部是一个名为“显示来自输出”的组合框，通常设置为“生成”。如果将其设置为“调试”，则将在调试会话期间看到生成的调试消息。这些消息将包括有关加载调试符号的消息以及从`_RPTn`宏重定向到输出窗口的消息。

如果希望将消息定向到文件，则需要使用 Win32 的`CreateFile`函数打开文件，并在调用`_CrtSetReportFile`函数时使用该函数的句柄。为此，您需要包含 Windows 头文件：

```cpp
    #define WIN32_LEAN_AND_MEAN 
    #include <Windows.h> 
    #include <crtdbg.h>
```

`WIN32_LEAN_AND_MEAN`宏将减小包含的 Windows 文件的大小。

```cpp
    HANDLE file =  
       CreateFileA("log.txt", GENERIC_WRITE, 0, 0, CREATE_ALWAYS, 0, 0); 
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE); 
    _CrtSetReportFile(_CRT_WARN, file); 
    _RPTF0(_CRT_WARN, "Application startedn"); 

    run_code(); 

    _RPTF0(_CRT_WARN, "Application endedn"); 
    CloseHandle(file);
```

此代码将将警告消息定向到名为`log.txt`的文本文件，每次运行应用程序时都会创建新文件。

# 使用 Windows 跟踪消息

`OutputDebugString`函数用于向调试器发送消息。该函数通过名为`DBWIN_BUFFER`的*共享内存部分*执行此操作。共享内存意味着任何进程都可以访问此内存，因此 Windows 提供了两个称为`DBWIN_BUFFER_READY`和`DBWIN_DATA_READY`的*事件对象*，用于控制对此内存的读取和写入访问。这些事件对象在进程之间共享，并且可以处于已发信号或未发信号状态。调试器将通过发信号`DBWIN_BUFFER_READY`事件来指示它不再使用共享内存，此时`OutputDebugString`函数可以将数据写入共享内存。调试器将等待`DBWIN_DATA_READY`事件，当`OutputDebugString`函数完成写入内存并且可以安全读取缓冲区时，将发出该事件。写入内存部分的数据将是调用`OutputDebugString`函数的进程 ID，后跟长达 4 KB 的数据字符串。

问题在于当您调用`OutputDebugString`函数时，它将等待`DBWIN_BUFFER_READY`事件，这意味着当您使用此函数时，您将应用程序的性能与另一个进程的性能（通常是调试器，但也可能不是）耦合在一起。很容易编写一个进程来访问`DBWIN_BUFFER`共享内存部分并访问相关的事件对象，因此可能会出现您的生产代码在运行具有此类应用程序的机器上。因此，非常重要的是您使用条件编译，以便`OutputDebugString`函数仅在调试构建中使用--这些代码永远不会发布给您的客户：

```cpp
    extern void run_code(); 

    int main() 
    { 
        #ifdef _DEBUG 
            OutputDebugStringA("Application startedn"); 
        #endif 

        run_code(); 

        #ifdef _DEBUG 
           OutputDebugStringA("Application endedn"); 
        #endif 
        return 0; 
    }
```

您需要包含`windows.h`头文件才能编译此代码。至于`_RPT`示例，您将需要在调试器下运行此代码以查看输出，或者运行类似于**DebugView**（可从微软的 Technet 网站获取）的应用程序。

Windows 提供了`DBWinMutex`互斥对象，用作访问此共享内存和事件对象的整体*关键*。顾名思义，当您拥有互斥体的句柄时，您将对资源具有互斥访问权限。问题在于，进程不必拥有此互斥体的句柄即可使用这些资源，因此您无法保证，如果您的应用程序认为它具有独占访问权限，它确实具有独占访问权限。

# 使用断言

断言检查条件是否为真。断言的意思就是：如果条件不为真，则程序不应继续。显然，断言不应在发布代码中调用，因此必须使用条件编译。断言应用于检查永远不应发生的条件：永远不会发生的事件。由于条件不会发生，因此在发布构建中不需要断言。

C 运行时提供了`assert`宏，可通过`<cassert>`头文件使用。该宏及其作为唯一参数传递的表达式中调用的任何函数，除非定义了`NDEBUG`符号，否则将被调用。也就是说，您不必定义`_DEBUG`符号来使用断言，并且应该采取额外的措施来明确阻止调用`assert`。

值得再次强调。即使未定义`_DEBUG`，`assert`宏也已定义，因此断言可能会在发布代码中调用。为防止这种情况发生，您必须在发布构建中定义`NDEBUG`符号。相反，您可以在调试构建中定义`NDEBUG`符号，以便可以使用跟踪，但不必使用断言。

通常，您将在调试构建中使用断言来检查函数中是否满足前置条件和后置条件，以及类不变条件是否得到满足。例如，您可能有一个二进制缓冲区，在第十个字节位置有一个特殊值，因此编写了一个提取该字节的函数：

```cpp
    const int MAGIC=9; 

    char get_data(char *p, size_t size) 
    { 
        assert((p != nullptr)); 
        assert((size >= MAGIC)); 
        return p[MAGIC]; 
    }
```

在这里，对`assert`的调用用于检查指针不是`nullptr`并且缓冲区足够大。如果这些断言为真，则意味着可以通过指针安全地访问第十个字节。

虽然在这段代码中并不是严格必要的，但是断言表达式是用括号括起来的。养成这样做的习惯是很好的，因为`assert`是一个宏，因此表达式中的逗号将被视为宏参数分隔符；括号可以防止这种情况发生。

由于`assert`宏在默认情况下将在发布构建中定义，因此您将需要通过在编译器命令行上定义`NDEBUG`来禁用它们，在您的 make 文件中，或者您可能希望显式使用条件编译：

```cpp
    #ifndef _DEBUG 
    #define NDEBUG 
    #endif
```

如果调用断言并且失败，则会在控制台上打印断言消息以及源文件和行号信息，然后通过调用`abort`终止进程。如果进程是使用发布构建标准库构建的，则进程`abort`是直接的，但是如果使用调试构建，则用户将看到标准的中止/重试/忽略消息框，其中中止和忽略选项将中止进程。重试选项将使用**即时**（**JIT**）调试将注册的调试器附加到进程。

相比之下，只有在定义了`_DEBUG`时，`_ASSERT`和`_ASSERTE`宏才会被定义，因此这些宏在发布构建中将不可用。当表达式为`false`时，这两个宏都会生成一个断言消息。`_ASSERT`宏的消息将包括源文件和行号以及一个说明断言失败的消息。`_ASSERTE`宏的消息类似，但包括失败的表达式。

```cpp
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE); 
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDOUT); 

    int i = 99; 
    _ASSERTE((i > 100));
```

此代码设置了报告模式，以便失败的断言将作为消息打印在控制台上（而不是默认的中止/重试/忽略对话框）。由于变量显然小于 100，断言将失败，因此进程将终止，并在控制台上打印以下消息：

```cpp
    test.cpp(23) : Assertion failed: (i > 100)
```

中止/重试/忽略对话框为测试应用程序的人提供了将调试器附加到进程的选项。如果您决定断言的失败是可恶的，您可以通过调用`_CrtDbgBreak`强制调试器附加到进程。

```cpp
    int i = 99; 
    if (i <= 100) _CrtDbgBreak();
```

您不需要使用条件编译，因为在发布构建中，`_CrtDbgBreak`函数是无操作的。在调试构建中，此代码将触发 JIT 调试，从而使您可以关闭应用程序或启动调试器，如果选择后者，则将启动注册的 JIT 调试器。

# 应用程序终止

`main`函数是应用程序的入口点。但是，操作系统不会直接调用它，因为 C++会在调用`main`之前执行初始化。这包括构造标准库全局对象（`cin`、`cout`、`cerr`、`clog`和宽字符版本），以及为支持 C++库的 C 运行时库执行的一系列初始化。此外，还有您的代码创建的全局和静态对象。当`main`函数返回时，将必须调用全局和静态对象的析构函数，并在 C 运行时上执行清理。

有几种方法可以有意终止进程。最简单的方法是从`main`函数返回，但这假设从代码想要完成进程的地方到`main`函数有一个简单的路径返回。当然，进程终止必须是有序的，您应该避免编写在代码的任何地方正常停止进程的代码。但是，如果您遇到数据损坏且无法恢复的情况，并且任何其他操作都可能损坏更多数据，那么您可能别无选择，只能终止应用程序。

`<cstdlib>`头文件提供了访问允许您终止和处理应用程序终止的函数的头文件。当一个 C++程序正常关闭时，C++基础设施将调用在`main`函数中创建的对象的析构函数（按照它们的构造顺序的相反顺序）和`static`对象的析构函数（可能是在`main`函数之外的函数中创建的）。`atexit`函数允许您注册在`main`函数完成和调用`static`对象析构函数之后将被调用的没有参数和返回值的函数。您可以通过多次调用此函数注册多个函数，并且在终止时，这些函数将按照它们的注册顺序的相反顺序被调用。在调用`atexit`函数注册的函数之后，将调用任何全局对象的析构函数。

还有一个名为`_onexit`的 Microsoft 函数，它也允许您注册在正常终止期间要调用的函数。

`exit`和`_exit`函数执行进程的正常退出，也就是在关闭进程之前清理 C 运行时并刷新任何打开的文件。`exit`函数通过调用任何注册的终止函数来执行额外的工作；`_exit`函数不调用这些终止函数，因此是一个快速退出。这些函数不会调用临时或自动对象的析构函数，因此如果您使用堆栈对象来管理资源，您必须在调用`exit`之前显式调用析构函数代码。但是，静态和全局对象的析构函数将被调用。

`quick_exit`函数导致正常关闭，但不调用任何析构函数，也不刷新任何流，因此没有资源清理。使用`atexit`注册的函数不会被调用，但您可以通过使用`at_quick_exit`函数注册终止函数来调用这些终止函数。在调用这些终止函数之后，`quick_exit`函数调用`_Exit`函数关闭进程。

您还可以调用`terminate`函数来关闭一个没有清理的进程。这个进程将调用一个已经注册了`set_terminate`函数的函数，然后调用`abort`函数。如果程序中发生异常并且没有被捕获，因此传播到`main`函数，C++基础设施将调用`terminate`函数。`abort`函数是终止进程的最严重的机制。这个函数将在不调用对象的析构函数或执行任何其他清理的情况下退出进程。该函数会引发`SIGABORT`信号，因此可以使用`signal`函数注册一个函数，在进程终止之前调用该函数。

# 错误值

有些函数设计为执行一个动作并根据该动作返回一个值，例如，`sqrt`将返回一个数的平方根。其他函数执行更复杂的操作，并使用返回值来指示函数是否成功。关于这种错误值没有共同的约定，因此如果一个函数返回一个简单的整数，就不能保证一个库使用的值与另一个库中的函数返回的值具有相同的含义。这意味着您必须仔细查看您使用的任何库代码的文档。

Windows 提供了常见的错误值，可以在`winerror.h`头文件中找到，Windows 的**软件开发工具包**（**SDK**）中的函数只返回该文件中的值。如果您编写的库代码将专门用于 Windows 应用程序，考虑使用该文件中的错误值，因为您可以使用 Win32 的`FormatMessage`函数来获取错误的描述，如下一节所述。

C 运行时库提供了一个名为`errno`的全局变量（实际上它是一个可以视为变量的宏）。C 函数将返回一个值来指示它们失败了，您可以访问`errno`值来确定错误是什么。`<errno.h>`头文件定义了标准的 POSIX 错误值。`errno`变量不表示成功，它只表示错误，因此只有在函数指示存在错误时才应该访问它。`strerror`函数将返回一个包含您传递的错误值描述的 C 字符串；这些消息根据通过调用`setlocale`函数设置的当前 C 语言环境进行本地化。

# 获取消息描述

要在运行时获取 Win32 错误代码的描述，您可以使用 Win32`FormatMessage`函数。这将获取系统消息或自定义消息的描述（在下一节中描述）。如果要使用自定义消息，则必须加载具有绑定到其上的消息资源的可执行文件（或 DLL），并将`HMODULE`句柄传递给`FormatMessage`函数。如果要获取系统消息的描述，则无需加载模块，因为 Windows 会为您执行此操作。例如，如果调用 Win32`CreateFile`函数打开一个文件，但找不到该文件，函数将返回一个“INVALID_HANDLE_VALUE”的值，表示存在错误。要获取错误的详细信息，您可以调用`GetLastError`函数（它返回一个 32 位无符号值，有时称为`DWORD`或`HRESULT`）。然后，您可以将错误值传递给`FormatMessage`：

```cpp
    HANDLE file = CreateFileA( 
        "does_not_exist", GENERIC_READ, 0, 0, OPEN_EXISTING, 0, 0); 
    if (INVALID_HANDLE_VALUE == file) 
    { 
        DWORD err = GetLastError(); 
        char *str; 
        DWORD ret = FormatMessageA( 
            FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_ALLOCATE_BUFFER, 
            0, err, LANG_USER_DEFAULT, reinterpret_cast<LPSTR>(&str),  
            0, 0); 
        cout << "Error: "<< str << endl; 
        LocalFree(str); 
    } 
    else 
    { 
        CloseHandle(file); 
    }
```

此代码尝试打开一个不存在的文件，并获取与失败相关的错误值（这将是一个`ERROR_FILE_NOT_FOUND`值）。然后，代码调用`FormatMessage`函数获取描述错误的字符串。函数的第一个参数是一个标志，指示函数应该如何工作；在这种情况下，`FORMAT_MESSAGE_FROM_SYSTEM`标志表示错误是系统错误，`FORMAT_MESSAGE_ALLOCATE_BUFFER`标志表示函数应该使用 Win32`LocalAlloc`函数分配足够大的缓冲区来容纳字符串。

如果错误是您定义的自定义值，则应使用`FORMAT_MESSAGE_FROM_HMODULE`标志，使用`LoadLibrary`打开文件，并将结果的`HMODULE`作为通过第二个参数传递的参数。

第三个参数是错误消息编号（来自`GetLastError`），第四个是指示要使用的语言 ID 的`LANGID`（在这种情况下，使用`LANG_USER_DEFAULT`来获取当前登录用户的语言 ID）。 `FormatMessage`函数将为错误值生成一个格式化的字符串，这个字符串可能有替换参数。格式化的字符串返回到一个缓冲区中，您有两个选项：您可以分配一个字符缓冲区，并将指针作为第五个参数传递，并将长度作为第六个参数传递，或者您可以请求函数使用`LocalAlloc`函数分配一个缓冲区，就像这个例子中一样。要访问函数分配的缓冲区，您需要通过第五个参数传递指针变量的*地址*。

请注意，第五个参数用于接受指向用户分配的缓冲区的指针，或者返回系统分配的缓冲区的地址，这就是为什么在这种情况下必须对指针进行转换。

一些格式字符串可能有参数，如果有，这些值将通过第七个参数中的数组传递（在这种情况下，没有传递数组）。前面代码的结果是字符串：

```cpp
    Error: The system cannot find the file specified.
```

使用消息编译器、资源文件和`FormatMessage`，您可以提供一种机制，从您的函数中返回错误值，然后根据当前语言环境将其转换为本地化字符串。

# 使用消息编译器

前面的示例表明，您可以获取 Win32 错误的本地化字符串，但也可以创建自己的错误并提供绑定为资源的本地化字符串到您的进程或库。如果您打算向最终用户报告错误，您必须确保描述已本地化。Windows 提供了一个名为消息编译器（`mc.exe`）的工具，它将获取包含各种语言中消息条目的文本文件，并将它们编译为可以绑定到模块的二进制资源。

例如：

```cpp
    LanguageNames = (British = 0x0409:MSG00409) 
    LanguageNames = (French  = 0x040c:MSG0040C) 

    MessageId       = 1 
    SymbolicName    = IDS_GREETING 
    Language        = English 
    Hello 
    . 
    Language        = British 
    Good day 
    . 
    Language        = French 
    Salut 
    .
```

这为同一消息定义了三个本地化字符串。这里的消息是简单的字符串，但您可以定义带有占位符的格式消息，这些占位符可以在运行时提供。*中性*语言是美国英语，此外我们还为英国英语和法语定义了字符串。语言的名称在文件顶部的`LanguageNames`行中定义。这些条目具有稍后在文件中使用的名称，语言的代码页以及将包含消息资源的二进制资源的名称。

`MessageId`是`FormatMessage`函数将使用的标识符，`SymbolicName`是一个预处理器符号，将在头文件中定义，以便您可以在 C++代码中使用此消息而不是数字。通过将其传递给命令行实用程序`mc.exe`来编译此文件，将创建五个文件：一个具有符号定义的头文件，三个二进制源（`MSG00001.bin`，默认情况下为中性语言创建，`MSG00409.bin`和`MSG0040C.bin`，由于`LanguageNames`行而创建），以及资源编译器文件。对于此示例，资源编译器文件（扩展名为`.rc`）将包含：

```cpp
    LANGUAGE 0xc,0x1 
    1 11 "MSG0040C.bin" 
    LANGUAGE 0x9,0x1 
    1 11 "MSG00001.bin" 
    LANGUAGE 0x9,0x1 
    1 11 "MSG00409.bin"
```

这是一个标准的资源文件，可以由 Windows SDK 资源编译器（`rc.exe`）编译，该编译器将消息资源编译为`.res`文件，该文件可以绑定到可执行文件或 DLL。具有绑定到其的类型`11`资源的进程或 DLL 可以被`FormatMessage`函数用作描述性错误字符串的来源。

通常，您不会使用消息 ID 1，因为它不太可能是唯一的，而且您可能希望利用*facility code*和*severity code*（有关 facility code 的详细信息，请查看`winerror.h`头文件）。此外，为了指示消息不是 Windows，您可以在运行`mc.exe`时使用`/c`开关设置错误代码的客户位。这意味着您的错误代码将不是像 1 这样的简单值，但这应该无所谓，因为您的代码将使用头文件中定义的符号。

# C++异常

异常是指异常情况。它们不是正常情况。它们不是你想要发生的情况，但可能会发生的情况。任何异常情况通常意味着您的数据将处于不一致的状态，因此使用异常意味着您需要以事务性术语思考，即，操作要么成功，要么对象的状态应保持与尝试操作之前相同。当代码块中发生异常时，代码块中发生的所有事情都将无效。如果代码块是更广泛代码块的一部分（比如，一个由另一个函数调用的一系列函数调用的函数），那么另一个代码块中的工作将无效。这意味着异常可能传播到调用堆栈上游的其他代码块，使依赖于操作成功的对象无效。在某个时候，异常情况将是可恢复的，因此您将希望防止异常进一步传播。

# 异常规范

异常规范在 C++11 中已被弃用，但您可能会在早期的代码中看到它们。规范是通过应用于函数声明的`throw`表达式来给出可以从函数中抛出的异常。`throw`规范可以是省略号，这意味着函数可以抛出异常，但类型未指定。如果规范为空，则表示函数不会抛出异常，这与在 C++11 中使用`noexcept`指定符相同。

`noexcept`指定符告诉编译器不需要异常处理，因此如果函数中发生异常，异常将不会从函数中冒出，并且将立即调用`terminate`函数。在这种情况下，不能保证自动对象的析构函数被调用。

# C++异常语法

在 C++中，通过抛出异常对象生成异常情况。该异常对象可以是任何您喜欢的东西：对象、指针或内置类型，但由于异常可能由其他人编写的代码处理，最好标准化用于表示异常的对象。为此，标准库提供了`exception`类，它可以用作基类。

```cpp
    double reciprocal(double d) 
    { 
        if (d == 0)  
        { 
            // throw 0; 
            // throw "divide by zero"; 
            // throw new exception("divide by zero"); 
            throw exception("divide by zero"); 
        } 
        return 1.0 / d; 
    }
```

此代码测试参数，如果为零，则引发异常。给出了四个示例，所有示例都是有效的 C++，但只有最后一个版本是可接受的，因为它使用了一个标准库类（或从标准库类派生的类），并且遵循了异常通过值抛出的约定。

当引发异常时，异常处理基础设施接管。执行将在当前代码块中停止，并且异常将向上传播到调用堆栈。当异常通过代码块传播时，所有自动对象都将被销毁，但在代码块中在堆上创建的对象将不会被销毁。这是一个称为**堆栈展开**的过程，即在异常移动到调用堆栈中的上面的堆栈帧之前，尽可能清理每个堆栈帧。如果异常没有被捕获，它将传播到`main`函数，此时将调用`terminate`函数来处理异常（因此将终止进程）。

您可以保护代码以处理传播的异常。代码受到`try`块的保护，并且通过相关的`catch`块捕获：

```cpp
    try  
    { 
        string s("this is an object"); 
        vector<int> v = { 1, 0, -1}; 
        reciprocal(v[0]); 
        reciprocal(v[1]); 
        reciprocal(v[2]); 
    } 
    catch(exception& e) 
    { 
        cout << e.what() << endl; 
    }
```

与 C++中的其他代码块不同，即使`try`和`catch`块只包含单行代码，括号也是必需的。在前面的代码中，对`reciprocal`函数的第二次调用将引发异常。异常将停止代码块中的任何更多代码的执行，因此不会发生对`reciprocal`函数的第三次调用。相反，异常会传播出代码块。`try`块是大括号之间定义的对象的作用域，这意味着这些对象的析构函数将被调用（`s`和`v`）。然后控制传递到相关的`catch`块，在这种情况下，只有一个处理程序。`catch`块是`try`块的一个单独的块，因此您无法访问在`try`块中定义的任何变量。这是有道理的，因为当生成异常时，整个代码块都是*被污染*的，因此您不能信任在该块中创建的任何对象。此代码使用了被接受的约定，即异常被引用捕获，以便捕获实际的异常对象，而不是副本。

约定是：抛出我的值，通过引用捕获。

标准库提供了一个名为`uncaught_exception`的函数，如果已经抛出异常但尚未处理，则返回`true`。测试这一点似乎有些奇怪，因为当异常发生时，除了异常基础设施之外不会调用任何代码（例如`catch`处理程序），你应该在那里放置异常代码。然而，当异常被抛出时确实会调用其他代码：在堆栈清除期间被销毁的自动对象的析构函数。`uncaught_exception`函数应该在析构函数中使用，以确定对象是否由于异常而被销毁，而不是由于对象超出范围或被删除而进行正常对象销毁。例如：

```cpp
    class test 
    { 
        string str; 
    public: 
        test() : str("") {} 
        test(const string& s) : str(s) {} 
        ~test() 
        { 
            cout << boolalpha << str << " uncaught exception = " 
             << uncaught_exception() << endl; 
        } 
    };
```

这个简单的对象指示它是否因异常堆栈展开而被销毁。可以像这样进行测试：

```cpp
    void f(bool b) 
    { 
        test t("auto f"); 
        cout << (b ? "f throwing exception" : "f running fine")  
            << endl; 
        if (b) throw exception("f failed"); 
    } 

    int main() 
    { 
        test t1("auto main"); 
        try 
        { 
            test t2("in try in main"); 
            f(false); 
            f(true); 
            cout << "this will never be printed"; 
        } 
        catch (exception& e) 
        { 
            cout << e.what() << endl; 
        } 
        return 0; 
    }
```

`f`函数只有在使用`true`值调用时才会抛出异常。`main`函数调用`f`两次，一次使用`false`值（所以在`f`中不会抛出异常），第二次使用`true`。输出为：

```cpp
 f running fine
 auto f uncaught exception = false
 f throwing exception
 auto f uncaught exception = true
 in try in main uncaught exception = true
 f failed
 auto main uncaught exception = false
```

第一次调用`f`时，`test`对象被正常销毁，所以`uncaught_exception`将返回`false`。第二次调用`f`时，函数中的`test`对象在异常被捕获之前被销毁，所以`uncaught_exception`将返回`true`。由于抛出了异常，执行离开`try`块，所以`try`块中的`test`对象被销毁，`uncaught_exception`将返回`true`。最后，当异常被处理并且控制返回到`catch`块后的代码时，`main`函数中堆栈上创建的`test`对象将在`main`函数返回时被销毁，所以`uncaught_exception`将返回`false`。

# 标准异常类

`exception`类是一个简单的 C 字符串容器：字符串作为构造函数参数传递，并通过`what`访问器可用。标准库在`<exception>`库中声明了异常类，并鼓励您从中派生自己的异常类。标准库提供了以下派生类；大多数在`<stdexcept>`中定义。

| **类** | **抛出** |
| --- | --- |
| `bad_alloc` | 当`new`运算符无法分配内存时（在`<new>`中） |
| `bad_array_new_length` | 当`new`运算符被要求创建一个具有无效长度的数组时（在`<new>`中） |
| `bad_cast` | 当`dynamic_cast`到引用类型失败时（在`<typeinfo>`中） |
| `bad_exception` | 发生了意外情况（在`<exception>`中） |
| `bad_function_call` | 调用了空的`function`对象（在`<functional>`中） |
| `bad_typeid` | 当`typeid`的参数为空时（在`<typeinfo>`中） |
| `bad_weak_ptr` | 访问已经销毁的对象的弱指针时（在`<memory>`中） |
| `domain_error` | 尝试在操作定义的域之外执行操作时 |
| `invalid_argument` | 当参数使用了无效值时 |
| `length_error` | 尝试超出对象定义的长度时 |
| `logic_error` | 当存在逻辑错误时，例如类不变量或前置条件 |
| `out_of_range` | 尝试访问对象定义范围之外的元素时 |
| `overflow_error` | 当计算结果大于目标类型时 |
| `range_error` | 当计算结果超出类型范围时 |
| `runtime_error` | 当代码范围外发生错误时 |
| `system_error` | 包装操作系统错误的基类（在`<system_error>`中） |
| `underflow_error` | 当计算结果低于下限时 |

在前面的表中提到的所有类都有一个构造函数，该构造函数接受`const char*`或`const string&`参数，与接受 C 字符串的`exception`类相反（因此，如果通过`string`对象传递描述，则使用`c_str`方法构造基类）。没有宽字符版本，因此如果要从宽字符字符串构造异常描述，必须进行转换。还要注意，标准异常类只有一个构造函数参数，并且可以通过继承的`what`访问器获得。

关于异常可以持有的数据没有绝对规则。您可以从`exception`派生一个类，并使用您想要提供给异常处理程序的任何值来构造它。

# 按类型捕获异常

每个`try`块可以有多个`catch`块，这意味着您可以根据异常类型定制异常处理。`catch`子句中的参数类型将按照它们声明的顺序与异常类型进行测试。异常将由与异常类型匹配的第一个处理程序处理，或者是一个基类。这突出了通过引用捕获异常对象的约定。如果您以基类对象捕获，将会创建一个副本，切割派生类对象。在许多情况下，代码会抛出从`exception`类派生的类型的对象，这意味着`exception`的 catch 处理程序将捕获所有异常。

由于代码可以抛出任何对象，因此可能会有异常传播出处理程序。C++允许您使用`catch`子句中的省略号捕获所有内容。显然，您应该按照从最派生到最不派生的顺序排列`catch`处理程序，并且（如果使用）将省略号处理程序放在最后：

```cpp
    try  
    { 
        call_code(); 
    } 
    catch(invalid_argument& iva) 
    { 
        cout << "invalid argument: " << e.what() << endl; 
    } 
    catch(exception& exc) 
    { 
        cout << typeid(exc).name() << ": " << e.what() << endl; 
    } 
    catch(...) 
    { 
        cout << "some other C++ exception" << endl; 
    }
```

如果受保护的代码没有抛出异常，则`catch`块不会被执行。

当处理程序检查异常时，可能会决定不想抑制异常；这称为重新抛出异常。为此，您可以使用没有操作数的`throw`语句（这仅允许在`catch`处理程序中），它将重新抛出实际捕获的异常对象，而不是副本。

异常是基于线程的，因此很难将异常传播到另一个线程。`exception_ptr`类（在`<exception>`中）为任何类型的异常对象提供了共享所有权语义。您可以通过调用`make_exception_ptr`对象获得异常对象的共享副本，或者甚至可以在`catch`块中使用`current_exception`获得正在处理的异常的共享副本。这两个函数都返回一个`exception_ptr`对象。`exception_ptr`对象可以持有任何类型的异常，而不仅仅是从`exception`类派生的异常，因此从包装的异常获取信息是特定于异常类型的。`exception_ptr`对象对这些细节一无所知，因此您可以将其传递给`rethrow_exception`，在您想要使用共享异常的上下文中（另一个线程），然后捕获适当的异常对象。在下面的代码中，有两个线程在运行。`first_thread`函数在一个线程上运行，`second_thread`函数在另一个线程上运行：

```cpp
    exception_ptr eptr = nullptr; 

    void first_thread() 
    { 
        try  
        { 
            call_code(); 
        } 
        catch (...)  
        { 
            eptr = current_exception();  
        } 
        // some signalling mechanism ... 
    } 

    void second_thread() 
    { 
        // other code 

        // ... some signalling mechanism 
        if (eptr != nullptr)  
        { 
            try 
            { 
                rethrow_exception(eptr); 
            } 
            catch(my_exception& e) 
            { 
                // process this exception 
            } 
            eptr = nullptr; 
        } 
        // other code 
    }
```

前面的代码看起来像是使用`exception_ptr`作为指针。实际上，`eptr`被创建为全局对象，对`nullptr`的赋值使用了复制构造函数来创建一个空对象（其中包装的异常是`nullptr`）。类似地，与`nullptr`的比较实际上测试了包装的异常。

本书不涉及 C++线程，因此我们不会详细介绍两个线程之间的信号传递。这段代码表明，*任何异常*的共享副本可以存储在一个上下文中，然后在另一个上下文中重新抛出和处理。

# 函数 try 块

您可能会决定用`try`块保护整个函数，这种情况下，您可以编写如下代码：

```cpp
    void test(double d) 
    { 
        try 
        { 
            cout << setw(10) << d << setw(10) << reciprocal(d) << endl; 
        } 

        catch (exception& e) 
        { 
            cout << "error: " << e.what() << endl; 
        } 
    }
```

这使用了之前定义的`reciprocal`函数，如果参数为零，它将抛出一个`exception`。这种情况的另一种替代语法是：

```cpp
    void test(double d) 
    try 
    { 
        cout << setw(10) << d << setw(10) << reciprocal(d) << endl; 
    } 
    catch (exception& e) 
    { 
        cout << "error: " << e.what() << endl; 
    }
```

这看起来相当奇怪，因为函数原型后面紧跟着`try... catch`块，而且没有外部的大括号。函数体是`try`块中的代码；当这段代码完成时，函数就会返回。如果函数返回一个值，它必须在`try`块中返回。在大多数情况下，你会发现这种语法会使你的代码变得不太可读，但有一种情况可能会有用——在构造函数的初始化列表中。

```cpp
    class inverse 
    { 
        double recip; 
    public: 
        inverse() = delete; 
        inverse(double d) recip(reciprocal(d)) {} 
        double get_recip() const { return recip; } 
    };
```

在这段代码中，我们包装了一个`double`值，它只是构造函数参数的倒数。通过在初始化列表中调用`reciprocal`函数来初始化数据成员。由于这是在构造函数体之外，发生在这里的异常将直接传递给调用构造函数的代码。如果您想进行一些额外的处理，那么您可以在构造函数体内调用倒数函数：

```cpp
    inverse::inverse(double d)  
    {  
        try { recip = reciprocal(d); } 
        catch(exception& e) { cout << "invalid value " << d << endl; } 
    }
```

重要的是要注意，异常将被自动重新抛出，因为构造函数中的任何异常意味着对象是无效的。然而，这允许您进行一些额外的处理，如果有必要的话。这种解决方案对于在基对象构造函数中抛出的异常是行不通的，因为虽然您可以在派生构造函数体中调用基构造函数，但编译器会自动调用默认构造函数。如果您希望编译器调用除默认构造函数之外的构造函数，您必须在初始化列表中调用它。在`inverse`构造函数中提供异常代码的另一种替代语法是使用函数`try`块：

```cpp
    inverse::inverse(double d)  
    try 
        : recip (reciprocal(d)) {}  
    catch(exception& e) { cout << "invalid value " << d << endl; }
```

这看起来有点凌乱，但构造函数体仍然在初始化列表之后，给`recip`数据成员赋予初始值。对`reciprocal`的调用中的任何异常都将被捕获并在处理后自动重新抛出。初始化列表可以包含对基类和任何数据成员的调用，所有这些都将受到`try`块的保护。

# 系统错误

`<system_error>`库定义了一系列类来封装系统错误。`error_category`类提供了一种将数值错误值转换为本地化描述字符串的机制。通过`<system_error>`中的`generic_category`和`system_category`函数可以获得两个对象，而`<ios>`中有一个名为`isostream_category`的函数；所有这些函数都返回一个`error_category`对象。`error_category`类有一个名为`message`的方法，它返回您传递的错误号的字符串描述。从`generic_category`函数返回的对象将返回 POSIX 错误的描述字符串，因此您可以使用它来获取`errno`值的描述。从`system_category`函数返回的对象将通过 Win32 的`FormatMessage`函数使用`FORMAT_MESSAGE_FROM_SYSTEM`作为标志参数返回错误描述，因此可以用它来获取`string`对象中 Windows 错误消息的描述。

请注意，`message`没有额外的参数来传递值，以获取需要参数的 Win32 错误消息。因此，在这些情况下，您将得到一个带有格式化占位符的消息。

尽管名字是这样，`isostream_category`对象实质上返回的是与`generic_category`对象相同的描述。

`system_error`异常是一个报告由`error_category`对象描述的值的类。例如，这是之前使用`FormatMessage`但使用`system_error`重新编写的示例：

```cpp
    HANDLE file = CreateFileA( 
       "does_not_exist", GENERIC_READ, 0, 0, OPEN_EXISTING, 0, 0); 
    if (INVALID_HANDLE_VALUE == file) 
    { 
        throw system_error(GetLastError(), system_category()); 
    } 
    else 
    { 
        CloseHandle(file); 
    }
```

这里使用的`system_error`构造函数的第一个参数是错误值（从 Win32 函数`GetLastError`返回的`ulong`），第二个参数是用于在调用`system_error::what`方法时将错误值转换为描述性字符串的`system_category`对象。

# 嵌套异常

`catch`块可以通过调用没有操作数的`throw`重新抛出当前异常，并且会进行堆栈展开，直到在调用堆栈中达到下一个`try`块。你还可以将当前异常*嵌套在*另一个异常内部重新抛出。这是通过调用`throw_with_nested`函数（在`<exception>`中）并传递新异常来实现的。该函数调用`current_exception`并将异常对象与参数一起包装在嵌套异常中，然后抛出。调用堆栈上方的`try`块可以捕获此异常，但它只能访问外部异常；它无法直接访问内部异常。相反，可以通过调用`rethrow_if_nested`来抛出内部异常。例如，这是另一个打开文件的代码版本：

```cpp
    void open(const char *filename) 
    { 
        try  
        { 
            ifstream file(filename); 
            file.exceptions(ios_base::failbit); 
            // code if the file exists 
        } 
        catch (exception& e)  
        { 
            throw_with_nested( 
                system_error(ENOENT, system_category(), filename)); 
        } 
    }
```

该代码打开一个文件，如果文件不存在，则设置一个状态位（可以稍后使用`rdstat`方法调用来测试位）。下一行指示应该由抛出异常的类处理的状态位的值，在这种情况下，提供了`ios_base::failbit`。如果构造函数未能打开文件，则将设置此位，因此`exceptions`方法将通过抛出异常来响应。在这个例子中，异常被捕获并包装成嵌套异常。外部异常是一个`system_error`异常，它使用`ENOENT`的错误值（表示文件不存在）和一个`error_category`对象来解释它进行初始化，并传递文件名作为附加信息。

可以这样调用这个函数：

```cpp
    try 
    { 
        open("does_not_exist"); 
    } 
    catch (exception& e) 
    { 
        cout << e.what() << endl; 
    }
```

在这里捕获的异常可以被访问，但它只提供有关外部对象的信息：

```cpp
 does_not_exist: The system cannot find the file specified.
```

这条消息是由`system_error`对象使用传递给它构造函数的附加信息和类别对象的描述构造的。要获取嵌套异常中的内部对象，必须告诉系统使用`rethrow_if_nested`调用抛出内部异常。因此，不是打印外部异常，而是调用这样的函数：

```cpp
    void print_exception(exception& outer) 
    { 
        cout << outer.what() << endl; 
        try { rethrow_if_nested(outer); } 
        catch (exception& inner) { print_exception(inner); } 
    }
```

这将打印外部异常的描述，然后调用`rethrow_if_nested`，只有在有嵌套时才会抛出异常。如果是这样，它会抛出内部异常，然后被捕获并递归调用`print_exception`函数。结果是：

```cpp
    does_not_exist: The system cannot find the file specified. 
    ios_base::failbit set: iostream stream error
```

最后一行是在调用`ifstream::exception`方法时抛出的内部异常。

# 结构化异常处理

Windows 中的本地异常是**结构化异常处理**（**SEH**），Visual C++有一种语言扩展，允许你捕获这些异常。重要的是要理解它们与 C++异常不同，编译器认为 C++异常是*同步*的，也就是说，编译器知道一个方法是否（或者特别地，不会）抛出 C++异常，并且在分析代码时使用这些信息。C++异常也是按类型捕获的。SEH 不是 C++的概念，因此编译器将结构化异常视为*异步*，这意味着它将任何在 SEH 保护块内的代码视为可能引发结构化异常，因此编译器无法执行优化。SEH 异常也是按异常代码捕获的。

SEH 的语言扩展是 Microsoft C/C++的扩展，也就是说，它们可以在 C 和 C++中使用，因此处理基础结构不知道对象析构函数。此外，当你捕获一个 SEH 异常时，不会对堆栈或进程的任何其他部分的状态做出任何假设。

尽管大多数 Windows 函数会以适当的方式捕获内核生成的 SEH 异常，但有些函数故意允许它们传播（例如，**远程过程调用**（**RPC**）函数，或用于内存管理的函数）。对于一些 Windows 函数，您可以明确要求使用 SEH 异常处理错误。例如，`HeapCreate`函数集允许 Windows 应用程序创建私有堆，您可以传递`HEAP_GENERATE_EXCEPTIONS`标志以指示在创建堆以及在私有堆中分配或重新分配内存时出现错误将生成 SEH 异常。这是因为调用这些函数的开发人员可能认为失败是如此严重，以至于无法恢复，因此进程应该终止。由于 SEH 是如此严重的情况，您应该仔细审查是否适当（这并非完全不可能）做更多事情，而不仅仅是报告异常的详细信息并终止进程。

SEH 异常本质上是低级操作系统异常，但熟悉其语法很重要，因为它看起来类似于 C++异常。例如：

```cpp
    char* pPageBuffer; 
    unsigned long curPages = 0; 
    const unsigned long PAGESIZE = 4096; 
    const unsigned long PAGECOUNT = 10; 

    int main() 
    { 
        void* pReserved = VirtualAlloc( 
        nullptr, PAGECOUNT * PAGESIZE, MEM_RESERVE, PAGE_NOACCESS); 
        if (nullptr == pReserved)  
        { 
            cout << "allocation failed" << endl; 
            return 1; 
        } 

        char *pBuffer = static_cast<char*>(pReserved); 
        pPageBuffer = pBuffer; 

        for (int i = 0; i < PAGECOUNT * PAGESIZE; ++i) 
        { 
            __try { pBuffer[i] = 'X'; } __except (exception_filter(GetExceptionCode())) { cout << "Exiting process.n"; ExitProcess(GetLastError()); } 
        } 
        VirtualFree(pReserved, 0, MEM_RELEASE); 
        return 0; 
    }
```

这里突出显示了 SEH 异常代码。此代码使用 Windows 的`VirtualAlloc`函数来保留一定数量的内存页。保留不会分配内存，该操作必须在称为**提交内存**的单独操作中执行。Windows 将以称为**页**的块保留（和提交）内存，在大多数系统上，一页为 4096 字节，如此处所假设的。对`VirtualAlloc`函数的调用指示应保留 4096 字节的十页，稍后将对其进行提交（和使用）。

`VirtualAlloc`的第一个参数指示内存的位置，但由于我们正在保留内存，因此这并不重要，因此传递了`nullptr`。如果保留成功，则会返回指向内存的指针。`for`循环只是逐字节向内存写入数据。突出显示的代码使用结构化异常处理来保护此内存访问。受保护的块以`__try`关键字开始。当引发 SEH 时，执行将传递到`__except`块。这与 C++异常中的`catch`块非常不同。首先，`__except`异常处理程序接收三个值中的一个，以指示它应该如何行为。只有在这是`EXCEPTION_EXECUTE_HANDLER`时，处理程序块中的代码才会运行（在此代码中，以突然关闭进程）。如果值是`EXCEPTION_CONTINUE_SEARCH`，则不会识别异常，并且搜索将继续向上堆栈，*但不会进行 C++堆栈展开*。令人惊讶的值是`EXCEPTION_CONTINUE_EXECUTION`，因为这会解除异常并且`__try`块中的执行将继续。*您无法使用 C++异常做到这一点*。通常，SEH 代码将使用异常过滤器函数来确定`__except`处理程序所需的操作。在此代码中，此过滤器称为`exception_filter`，它通过调用 Windows 函数`GetExceptionCode`获取的异常代码进行传递。此语法很重要，因为此函数只能在`__except`上下文中调用。

第一次循环运行时，不会分配任何内存，因此写入内存的代码将引发异常：页面错误。执行将传递到异常处理程序，然后通过`exception_filter`：

```cpp
    int exception_filter(unsigned int code) 
    { 
        if (code != EXCEPTION_ACCESS_VIOLATION) 
        { 
            cout << "Exception code = " << code << endl; 
            return EXCEPTION_EXECUTE_HANDLER; 
        } 

        if (curPage >= PAGECOUNT) 
        { 
            cout << "Exception: out of pages.n"; 
            return EXCEPTION_EXECUTE_HANDLER; 
        } 

        if (VirtualAlloc(static_cast<void*>(pPageBuffer), PAGESIZE, 
         MEM_COMMIT, PAGE_READWRITE) == nullptr) 
        { 
            cout << "VirtualAlloc failed.n"; 
            return EXCEPTION_EXECUTE_HANDLER; 
        } 

        curPage++; 
        pPageBuffer += PAGESIZE; 
        return EXCEPTION_CONTINUE_EXECUTION; 
    }
```

在 SEH 代码中，重要的是只处理您知道的异常，并且只有在您知道条件已完全解决时才消耗异常。如果访问未提交的 Windows 内存，操作系统会生成一个称为页面错误的异常。在此代码中，测试异常代码以查看是否是页面错误，如果不是，则过滤器返回，告诉异常处理程序运行终止进程的异常处理程序块中的代码。如果异常是页面错误，那么我们可以提交下一页。首先，测试页面编号是否在我们将使用的范围内（如果不是，则关闭进程）。然后，使用另一个调用`VirtualAlloc`来标识要提交的页面和该页面中的字节数，提交下一页。如果函数成功，它将返回指向已提交页面的指针或空值。只有在提交页面成功后，过滤器才会返回`EXCEPTION_CONTINUE_EXECUTION`的值，表示已处理异常并且可以在引发异常的点继续执行。这段代码是使用`VirtualAlloc`的标准方式，因为这意味着只有在需要时才会提交内存页面。

SEH 还有终止处理程序的概念。当执行通过调用`return`离开`__try`代码块，或者通过完成代码块中的所有代码，或者通过调用 Microsoft 扩展`__leave`指令，或者引发 SEH 时，标有`__finally`的终止处理程序代码块将被调用。由于终止处理程序总是被调用，无论`__try`块如何退出，都可以将其用作释放资源的一种方式。然而，由于 SEH 不进行 C++堆栈展开（也不调用析构函数），这意味着您不能在具有 C++对象的函数中使用此代码。实际上，编译器将拒绝编译具有 SEH 并创建 C++对象的函数，无论是在函数堆栈上还是在堆上分配的对象。（但是，您可以使用全局对象或在调用函数中分配并作为参数传递的对象。）`__try`/`__finally`结构看起来很有用，但受到一个限制，即不能与创建 C++对象的代码一起使用。

# 编译器异常开关

在这一点上，值得解释一下为什么要使用`/EHsc`开关编译代码。简单的答案是，如果不使用此开关，编译器将从标准库代码中发出警告，由于标准库使用异常，您必须使用`/EHsc`开关。警告告诉您这样做，所以您就这样做了。

长答案是，`/EH`开关有三个参数，可以影响异常处理方式。使用`s`参数告诉编译器提供同步异常的基础设施，即在`try`块中可能抛出并在`catch`块中处理的 C++异常，并且具有调用自动 C++对象的析构函数的堆栈展开。`c`参数表示`extern C`函数（即所有 Windows SDK 函数）永远不会抛出 C++异常（因此编译器可以进行额外级别的优化）。因此，您可以使用`/EHs`或`/EHsc`编译标准库代码，但后者将生成更多优化的代码。还有一个额外的参数，其中`/EHa`表示代码将使用`try`/`catch`块捕获*同步和异步*异常（SEH）。

# 混合 C++和 SEH 异常处理

`RaiseException` Windows 函数将引发一个 SEH 异常。第一个参数是异常代码，第二个参数指示处理此异常后进程是否可以继续（`0`表示可以）。第三个和第四个参数提供有关异常的附加信息。第四个参数是指向包含这些附加参数的数组的指针，第三个参数给出了参数的数量。

使用`/EHa`，您可以编写如下的代码：

```cpp
    try  
    { 
        RaiseException(1, 0, 0, nullptr); 
    } 
    // legal code, but don't do it 
    catch(...) 
    { 
        cout << "SEH or C++ exception caught" << endl; 
    }
```

这段代码的问题在于它处理了所有 SEH 异常。这是非常危险的，因为一些 SEH 异常可能表明进程状态已经损坏，所以让进程继续运行是危险的。C 运行时库提供了一个名为`_set_se_translator`的函数，它提供了一个机制来指示哪些 SEH 异常由`try`处理。这个函数通过一个具有以下原型的函数传递一个指针：

```cpp
    void func(unsigned int, EXCEPTION_POINTERS*);
```

第一个参数是异常代码（将从`GetExceptionCode`函数返回），第二个参数是`GetExceptionInformation`函数的返回值，并带有与异常相关的任何附加参数（例如，通过`RaiseException`的第三个和第四个参数传递的参数）。您可以使用这些值来抛出 C++异常来代替 SEH。如果您提供了这个函数：

```cpp
    void seh_to_cpp(unsigned int code, EXCEPTION_POINTERS*) 
    { 
        if (code == 1) throw exception("my error"); 
    }
```

现在您可以在处理 SEH 异常之前注册该函数：

```cpp
    _set_se_translator(seh_to_cpp); 
    try  
    { 
        RaiseException(1, 0, 0, nullptr); 
    } 
    catch(exception& e) 
    { 
        cout << e.what() << endl; 
    }
```

在这段代码中，`RaiseException`函数正在引发一个值为 1 的自定义 SEH。这种转换可能并不是最有用的，但它说明了要点。`winnt.h`头文件定义了在 Windows 代码中可以引发的标准 SEH 异常的异常代码。一个更有用的转换函数可能是：

```cpp
    double reciprocal(double d) 
    { 
        return 1.0 / d; 
    } 

    void seh_to_cpp(unsigned int code, EXCEPTION_POINTERS*) 
    { 
        if (STATUS_FLOAT_DIVIDE_BY_ZERO == code || 
            STATUS_INTEGER_DIVIDE_BY_ZERO == code) 
        { 
            throw invalid_argument("divide by zero"); 
        } 
    }
```

这使您可以调用如下的逆函数：

```cpp
    _set_se_translator(seh_to_cpp); 
    try  
    { 
        reciprocal(0.0); 
    } 
    catch(invalid_argument& e) 
    { 
        cout << e.what() << endl; 
    }
```

# 编写异常安全的类

一般来说，当您编写类时，应确保保护类的用户免受异常的影响。异常不是错误传播机制。如果类上的方法失败但是可恢复的（对象状态保持一致），那么应该使用返回值（很可能是错误代码）来指示这一点。异常是用于异常情况的，这些情况已经使数据无效，并且在引发异常的地方，情况是无法恢复的。

当您的代码中发生异常时，您有三个选择。首先，您可以允许异常沿着调用堆栈传播，并将处理异常的责任放在调用代码上。这意味着您调用的代码没有通过`try`块进行保护，即使该代码被记录为可能引发异常。在这种情况下，您必须确保异常对调用代码是有意义的。例如，如果您的类被记录为网络类，并使用临时文件来缓冲从网络接收到的一些数据，如果文件访问代码引发异常，异常对象对调用您的代码的代码来说是没有意义的，因为该客户端代码认为您的类是关于访问网络数据的，而不是文件数据。然而，如果网络代码引发错误，允许这些异常传播到调用代码可能是有意义的，特别是如果它们涉及需要外部操作的错误（比如，网络电缆被拔掉或存在安全问题）。

在这种情况下，您可以应用第二个选项，即使用`try`块保护可能引发异常的代码，捕获已知异常，并抛出更合适的异常，可能嵌套原始异常，以便调用代码可以进行更详细的分析。如果异常对您的调用代码有意义，您可以允许它传播出去，但捕获原始异常允许您在重新引发异常之前采取额外的操作。

利用缓冲网络数据的例子，你可以决定，由于文件缓冲中存在错误，这意味着你无法再读取任何网络数据，因此你的异常处理代码应该以一种优雅的方式关闭网络访问。错误发生在文件代码中，而不是网络代码中，因此突然关闭网络是不合理的，更合理的做法是允许当前的网络操作完成（但忽略数据），以便不会将错误传播回网络代码。

最后的选择是用`try`块保护所有代码，并捕获和消耗异常，以便调用代码在不抛出异常的情况下完成。有两种主要情况适合这样做。首先，错误可能是可恢复的，因此在`catch`子句中，你可以采取措施来解决问题。在缓冲网络数据的例子中，当打开临时文件时，如果出现请求的名称已经存在的文件的错误，你可以简单地使用另一个名称再试一次。你的代码使用者不需要知道发生了这个问题（尽管，在代码测试阶段追踪这个错误可能是有意义的）。如果错误是不可恢复的，可能更合理的做法是使对象的状态无效并返回错误代码。

你的代码应该利用 C++异常基础设施的行为，它保证自动对象被销毁。因此，当你使用内存或其他适当的资源时，你应该尽可能地将它们包装在智能指针中，这样如果抛出异常，资源将由智能指针析构函数释放。使用资源获取即初始化（RAII）的类有`vector`、`string`、`fstream`和`make_shared`函数，因此如果对象的构造（或函数调用）成功，这意味着资源已经被获取，并且你可以通过这些对象使用资源。这些类也是**资源释放销毁**（**RRD**），这意味着当对象被销毁时资源被释放。智能指针类`unique_ptr`和`shared_ptr`不是 RAII，因为它们只是简单地包装资源，资源的分配是由其他代码单独执行的。然而，这些类是 RRD，因此你可以放心，如果抛出异常，资源将被释放。

异常处理可以提供三个级别的异常安全性。在最安全级别的尺度上是*无故障*的方法和函数。这是不会抛出异常并且不允许异常传播的代码。这样的代码将保证类不变量被维护，并且对象状态将是一致的。无故障的代码不是通过简单地捕获所有异常并消耗它们来实现的，相反，你必须保护所有代码并捕获和处理所有异常，以确保对象处于一致的状态。

所有内置的 C++类型都是无故障的。你还有一个保证，所有标准库类型都有无故障的析构函数，但由于容器在实例被销毁时会调用包含对象的析构函数，这意味着你必须确保你写入容器的类型也有无故障的析构函数。

编写无故障类型可能涉及相当详细的代码，所以另一个选择是*强有力的保证*。这样的代码会抛出异常，但它们确保没有内存泄漏，并且当抛出异常时，对象将处于与调用方法时相同的状态。这本质上是一个事务操作：要么对象被修改，要么它保持不变，就好像没有尝试执行操作一样。在大多数情况下的方法中，这将提供基本的异常安全保证。在这种情况下，有一个保证，无论发生什么，都不会有内存泄漏，但当抛出异常时，对象可能会处于不一致的状态，因此调用代码应该通过丢弃对象来处理异常。

文档很重要。如果对象的方法标有`throw`或`noexcept`，那么你就知道它是无故障的。只有在文档中这样说明时，你才应该假设有强有力的保证。否则，你可以假设对象将具有基本的异常安全保证，如果抛出异常，对象将无效。

# 总结

当你编写 C++代码时，你应该始终关注测试和调试代码。防止调试代码的理想方式是编写健壮、设计良好的代码。理想很难实现，所以最好编写易于诊断问题和易于调试的代码。C 运行时和 C++标准库提供了广泛的设施，使你能够跟踪和报告问题，并通过错误代码处理和异常，你有丰富的工具集来报告和处理函数的失败。

阅读完本书后，你应该意识到 C++语言和标准库提供了一种丰富、灵活和强大的编写代码的方式。更重要的是，一旦你知道如何使用语言和其库，C++就是一种乐趣。
