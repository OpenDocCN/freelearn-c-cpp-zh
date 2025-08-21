# 第十章：JIT 编译

LLVM 核心库配备了**ExecutionEngine**组件，允许在内存中编译和执行 IR 代码。使用这个组件，我们可以构建**即时**（**JIT**）编译器，允许直接执行 IR 代码。JIT 编译器更像解释器，因为不需要在辅助存储上存储目标代码。

在本章中，您将了解 JIT 编译器的应用程序，以及 LLVM JIT 编译器的工作原理。您将探索 LLVM 动态编译器和解释器，还将学习如何自己实现 JIT 编译器工具。您还将了解如何在静态编译器中使用 JIT 编译器，以及相关的挑战。

本章将涵盖以下主题：

+   获取 LLVM 的 JIT 实现和用例概述

+   使用 JIT 编译进行直接执行

+   利用 JIT 编译器进行代码评估

在本章结束时，您将了解如何开发 JIT 编译器，无论是使用预配置的类还是符合您需求的定制版本。您还将获得使用静态编译器内部的 JIT 编译器的知识。

# 技术要求

本章的代码文件可以在[`github.com/PacktPublishing/Learn-LLVM-12/tree/master/Chapter10`](https://github.com/PacktPublishing/Learn-LLVM-12/tree/master/Chapter10)找到

您可以在[`bit.ly/3nllhED`](https://bit.ly/3nllhED)找到代码的实际操作视频

# 获取 LLVM 的 JIT 实现和用例概述

到目前为止，我们只看过**提前**（**AOT**）编译器。这些编译器编译整个应用程序。只有在编译完成后，应用程序才能运行。如果在应用程序运行时进行编译，则编译器是 JIT 编译器。JIT 编译器有一些有趣的用例：

+   **虚拟机的实现**：编程语言可以使用 AOT 编译器将其转换为字节码。在运行时，JIT 编译器用于将字节码编译为机器代码。这种方法的优势在于字节码是与硬件无关的，并且由于 JIT 编译器，与 AOT 编译器相比没有性能损失。如今，Java 和 C#使用这种模型，但这个想法实际上很古老：1977 年的 USCD Pascal 编译器已经使用了类似的方法。

+   **表达式评估**：电子表格应用程序可以使用 JIT 编译器编译经常执行的表达式。例如，这可以加速财务模拟。LLVM 调试器 LLDB 使用这种方法在调试时评估源表达式。

+   **数据库查询**：数据库从数据库查询创建执行计划。执行计划描述了对表和列的操作，这导致了查询执行时的结果。JIT 编译器可以用于将执行计划转换为机器代码，从而加速查询的执行。

LLVM 的静态编译模型并不像你想象的那样远离 JIT 模型。LLVM 静态编译器`llc`将 LLVM IR 编译成机器代码，并将结果保存为磁盘上的目标文件。如果目标文件不是存储在磁盘上而是存储在内存中，那么代码是否可以执行？不直接执行，因为对全局函数和全局数据的引用使用重定位而不是绝对地址。

概念上，重定位描述了如何计算地址，例如，作为已知地址的偏移量。如果我们解析重定位为地址，就像链接器和动态加载器所做的那样，那么我们就可以执行目标代码。运行静态编译器将 IR 代码编译成内存中的目标文件，对内存中的目标文件进行链接步骤，然后运行代码，这就给我们了一个 JIT 编译器。LLVM 核心库中的 JIT 实现就是基于这个想法的。

在 LLVM 的开发历史中，有几个不同功能集的 JIT 实现。最新的 JIT API 是**按需编译**（**ORC**）引擎。如果你想知道这个首字母缩略词的含义：这是首席开发人员的意图，在托尔金的宇宙基础上发明另一个首字母缩略词，之前已经有了**ELF**（**可执行和链接格式**）和**DWARF**（**调试标准**）。

ORC 引擎建立在使用静态编译器和动态链接器在内存中的对象文件上的想法之上，并对其进行了扩展。实现采用了*分层*方法。两个基本级别如下：

1.  编译层

1.  链接层

在编译层之上可以放置一个提供对*延迟编译*的支持的层。**转换层**可以堆叠在延迟编译层的上方或下方，允许开发人员添加任意的转换，或者只是在某些事件发生时得到通知。这种分层方法的优势在于 JIT 引擎可以*根据不同的需求进行定制*。例如，高性能虚拟机可能会选择预先编译所有内容，并且不使用延迟编译层。其他虚拟机将强调启动时间和对用户的响应性，并通过延迟编译层的帮助来实现这一点。

较旧的 MCJIT 引擎仍然可用。API 源自一个更早的、已经删除的 JIT 引擎。随着时间的推移，API 变得有点臃肿，并且缺乏 ORC API 的灵活性。目标是删除这个实现，因为 ORC 引擎现在提供了 MCJIT 引擎的所有功能。新的开发应该使用 ORC API。

在下一节中，我们将先看看`lli`，LLVM 解释器和动态编译器，然后再深入实现 JIT 编译器。

# 使用 JIT 编译进行直接执行

直接运行 LLVM IR 是在考虑 JIT 编译器时首先想到的想法。这就是`lli`工具，LLVM 解释器和动态编译器所做的。我们将在下一节中探索`lli`工具，并随后自己实现类似的工具。

## 探索 lli 工具

让我们尝试使用`lli`工具进行一个非常简单的示例。将以下源代码存储为`hello.ll`文件。这相当于一个 C 语言的 hello world 应用程序。它声明了 C 库中`printf()`函数的原型。`hellostr`常量包含要打印的消息。在`main()`函数内部，通过`getelementptr`指令计算出消息的第一个字符的指针，并将该值传递给`printf()`函数。该应用程序始终返回`0`。完整的源代码如下：

```cpp
declare i32 @printf(i8*, ...)
@hellostr = private unnamed_addr constant [13 x i8] c"Hello                                                   world\0A\00"
define i32 @main(i32 %argc, i8** %argv) {
  %res = call i32 (i8*, ...) @printf(                  i8* getelementptr inbounds ([13 x i8],                          [13 x i8]* @hellostr, i64 0, i64 0))
  ret i32 0
}
```

这个 LLVM IR 文件足够通用，适用于所有平台。我们可以直接使用以下命令在`lli`工具中执行 IR：

```cpp
$ lli hello.ll
Hello world
```

这里有趣的一点是如何找到`printf()`函数。IR 代码被编译成机器代码，并触发了对`printf`符号的查找。在 IR 中找不到这个符号，所以当前进程会在其中搜索。`lli`工具动态链接到 C 库，并在那里找到了该符号。

当然，`lli`工具不会链接到您创建的库。为了启用这些函数的使用，`lli`工具支持加载共享库和对象。以下 C 源代码只是打印一个友好的消息：

```cpp
#include <stdio.h>
void greetings() {
  puts("Hi!");
}
```

存储在`greetings.c`文件中，我们将用它来探索使用`lli`工具加载对象。将此源代码编译成共享库。`-fPIC`选项指示 clang 生成位置无关的代码，这对于共享库是必需的。给定`-shared`选项后，编译器将创建`greetings.so`共享库：

```cpp
$ clang –fPIC –shared –o greetings.so greetings.c
```

我们还将文件编译成`greetings.o`对象文件：

```cpp
$ clang –c –o greetings.o greetings.c
```

现在我们有两个文件，`greetings.so`共享库和`greetings.o`对象文件，我们将它们加载到`lli`工具中。

我们还需要一个 LLVM IR 文件，其中调用`greetings()`函数。为此，请创建包含对该函数的单个调用的`main.ll`文件：

```cpp
declare void @greetings(...)
define dso_local i32 @main(i32 %argc, i8** %argv) {
  call void (...) @greetings()
  ret i32 0
}
```

如果尝试像以前一样执行 IR，则`lli`工具无法找到`greetings`符号，将简单崩溃：

```cpp
$ lli main.ll
PLEASE submit a bug report to https://bugs.llvm.org/ and include the crash backtrace.
```

`greetings()`函数在外部文件中定义，为了修复崩溃，我们必须告诉`lli`工具需要加载哪个附加文件。为了使用共享库，您必须使用`–load`选项，该选项以共享库的路径作为参数：

```cpp
$ lli –load ./greetings.so main.ll
Hi!
```

如果包含共享库的目录不在动态加载器的搜索路径中，则重要的是指定共享库的路径。如果省略，则将无法找到库。

或者，我们可以指示`lli`工具使用`–extra-object`选项加载对象文件：

```cpp
$ lli –extra-object greetings.o main.ll
Hi!
```

其他支持的选项是`–extra-archive`，它加载存档，以及`–extra-module`，它加载另一个位代码文件。这两个选项都需要文件的路径作为参数。

现在您知道如何使用`lli`工具直接执行 LLVM IR。在下一节中，我们将实现自己的 JIT 工具。

## 使用 LLJIT 实现我们自己的 JIT 编译器

`lli`工具只是 LLVM API 周围的薄包装器。在第一节中，我们了解到 ORC 引擎使用分层方法。`ExecutionSession`类表示正在运行的 JIT 程序。除其他项目外，此类还保存了使用的`JITDylib`实例。`JITDylib`实例是一个符号表，将符号名称映射到地址。例如，这可以是 LLVM IR 文件中定义的符号，或者是加载的共享库的符号。

要执行 LLVM IR，我们不需要自己创建 JIT 堆栈。实用程序`LLJIT`类提供此功能。当从较旧的 MCJIT 实现迁移时，您也可以使用此类。该类基本上提供了相同的功能。我们将在下一小节中开始实现 JIT 引擎的初始化。

### 初始化用于编译 LLVM IR 的 JIT 引擎

我们首先实现设置 JIT 引擎，编译 LLVM IR 模块并在此模块中执行`main()`函数的函数。稍后，我们将使用此核心功能构建一个小型 JIT 工具。这是`jitmain()`函数：

1.  该函数需要执行 LLVM 模块的 LLVM IR。还需要用于此模块的 LLVM 上下文类，因为上下文类保存重要的类型信息。目标是调用`main()`函数，因此我们还传递通常的`argc`和`argv`参数：

```cpp
Error jitmain(std::unique_ptr<Module> M,
              std::unique_ptr<LLVMContext> Ctx, int 
              argc,
              char *argv[]) {
```

1.  我们使用`LLJITBuilder`类创建`LLJIT`实例。如果发生错误，则返回错误。错误的可能来源是平台尚不支持 JIT 编译：

```cpp
  auto JIT = orc::LLJITBuilder().create();
  if (!JIT)
    return JIT.takeError();
```

1.  然后我们将模块添加到主`JITDylib`实例中。如果配置，则 JIT 编译将利用多个线程。因此，我们需要将模块和上下文包装在`ThreadSafeModule`实例中。如果发生错误，则返回错误：

```cpp
  if (auto Err = (*JIT)->addIRModule(
          orc::ThreadSafeModule(std::move(M),
                                std::move(Ctx))))
    return Err;
```

1.  与`lli`工具一样，我们还支持 C 库中的符号。`DefinitionGenerator`类公开符号，`DynamicLibrarySearchGenerator`子类公开共享库中找到的名称。该类提供了两个工厂方法。`Load()`方法可用于加载共享库，而`GetForCurrentProcess()`方法公开当前进程的符号。我们使用后者功能。符号名称可以具有前缀，取决于平台。我们检索数据布局并将前缀传递给`GetForCurrentprocess()`函数。然后符号名称将以正确的方式处理，我们不需要关心它。通常情况下，如果发生错误，我们会从函数中返回：

```cpp
  const DataLayout &DL = (*JIT)->getDataLayout();
  auto DLSG = orc::DynamicLibrarySearchGenerator::
      GetForCurrentProcess(DL.getGlobalPrefix());
  if (!DLSG)
    return DLSG.takeError();
```

1.  然后我们将生成器添加到主`JITDylib`实例中。如果需要查找符号，则还会搜索加载的共享库中的符号：

```cpp
  (*JIT)->getMainJITDylib().addGenerator(
      std::move(*DLSG));
```

1.  接下来，我们查找`main`符号。该符号必须在命令行给出的 IR 模块中。查找触发了该 IR 模块的编译。如果 IR 模块内引用了其他符号，则使用前一步添加的生成器进行解析。结果是`JITEvaluatedSymbol`类的实例：

```cpp
  auto MainSym = (*JIT)->lookup("main");
  if (!MainSym)
    return MainSym.takeError();
```

1.  我们询问返回的 JIT 符号函数的地址。我们将此地址转换为 C `main()`函数的原型：

```cpp
  auto *Main = (int (*)(
      int, char **))MainSym->getAddress();
```

1.  现在我们可以在 IR 模块中调用`main()`函数，并传递函数期望的`argc`和`argv`参数。我们忽略返回值：

```cpp
  (void)Main(argc, argv);
```

1.  函数执行后报告成功：

```cpp
  return Error::success();
}
```

这演示了使用 JIT 编译是多么容易。除了暴露当前进程或共享库中的符号之外，还有许多其他可能性。`StaticLibraryDefinitionGenerator`类暴露了静态存档中找到的符号，并且可以像`DynamicLibrarySearchGenerator`类一样使用。`LLJIT`类还有一个`addObjectFile()`方法来暴露对象文件的符号。如果现有的实现不符合您的需求，您还可以提供自己的`DefinitionGenerator`实现。在下一小节中，您将把实现扩展为 JIT 编译器。

### 创建 JIT 编译器实用程序

`jitmain()`函数很容易扩展为一个小工具，我们接下来就这样做。源代码保存在`JIT.cpp`文件中，是一个简单的 JIT 编译器：

1.  我们必须包含几个头文件。`LLJIT.h`头文件定义了`LLJIT`类和 ORC API 的核心类。我们包含`IRReader.h`头文件，因为它定义了一个用于读取 LLVM IR 文件的函数。`CommandLine.h`头文件允许我们以 LLVM 风格解析命令行选项。最后，`InitLLVM.h`头文件用于工具的基本初始化，`TargetSelect.h`头文件用于本机目标的初始化：

```cpp
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
```

1.  我们将`llvm`命名空间添加到当前作用域中：

```cpp
using namespace llvm;
```

1.  我们的 JIT 工具在命令行上期望有一个输入文件，我们使用`cl::opt<>`类声明这个文件：

```cpp
static cl::opt<std::string>
    InputFile(cl::Positional, cl::Required,
              cl::desc("<input-file>"));
```

1.  要读取 IR 文件，我们调用`parseIRFile()`函数。文件可以是文本 IR 表示，也可以是位码文件。该函数返回指向创建的模块的指针。错误处理有点不同，因为可以解析文本 IR 文件，这不一定是语法正确的。`SMDiagnostic`实例在语法错误时保存错误信息。错误消息被打印，应用程序退出：

```cpp
std::unique_ptr<Module>
loadModule(StringRef Filename, LLVMContext &Ctx,
           const char *ProgName) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod =
      parseIRFile(Filename, Err, Ctx);
  if (!Mod.get()) {
    Err.print(ProgName, errs());
    exit(-1);
  }
  return std::move(Mod);
}
```

1.  `jitmain()`函数放在这里：

```cpp
Error jitmain(…) { … }
```

1.  然后我们添加`main()`函数，该函数初始化工具和本机目标，并解析命令行：

```cpp
int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();
  cl::ParseCommandLineOptions(argc, argv,
                              "JIT\n");
```

1.  接下来，初始化 LLVM 上下文类：

```cpp
  auto Ctx = std::make_unique<LLVMContext>();
```

1.  然后我们加载命令行上命名的 IR 模块：

```cpp
  std::unique_ptr<Module> M =
      loadModule(InputFile, *Ctx, argv[0]);
```

1.  然后我们可以调用`jitmain()`函数。为了处理错误，我们使用`ExitOnError`实用类。当发生错误时，该类打印错误消息并退出应用程序。我们还设置了一个横幅，显示应用程序的名称，该横幅会在错误消息之前打印：

```cpp
  ExitOnError ExitOnErr(std::string(argv[0]) + ": ");
  ExitOnErr(jitmain(std::move(M), std::move(Ctx),
                    argc, argv));
```

1.  如果控制流到达这一点，那么 IR 已成功执行。我们返回`0`表示成功：

```cpp
  return 0;
}
```

这已经是完整的实现了！我们只需要添加构建描述，这是下一小节的主题。

### 添加 CMake 构建描述

为了编译这个源文件，我们还需要创建一个`CMakeLists.txt`文件，其中包含构建描述，保存在`JIT.cpp`文件旁边：

1.  我们将最小要求的 CMake 版本设置为 LLVM 所需的版本号，并给项目命名为`jit`：

```cpp
cmake_minimum_required (VERSION 3.13.4)
project ("jit")
```

1.  LLVM 包需要被加载，我们将 LLVM 提供的 CMake 模块目录添加到搜索路径中。然后我们包含`ChooseMSVCCRT`模块，以确保与 LLVM 使用相同的 C 运行时：

```cpp
find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})
include(ChooseMSVCCRT)
```

1.  我们还需要添加 LLVM 的定义和包含路径。使用的 LLVM 组件通过函数调用映射到库名称：

```cpp
add_definitions(${LLVM_DEFINITIONS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
llvm_map_components_to_libnames(llvm_libs Core OrcJIT
                                          Support 
                                          native)
```

1.  最后，我们定义可执行文件的名称，要编译的源文件以及要链接的库：

```cpp
add_executable(JIT JIT.cpp)
target_link_libraries(JIT ${llvm_libs})
```

1.  这就是 JIT 工具所需的一切。创建并切换到构建目录，然后运行以下命令来创建和编译应用程序：

```cpp
$ cmake –G Ninja <path to source directory>
$ ninja
```

这将编译`JIT`工具。您可以使用本章开头的`hello.ll`文件检查功能：

```cpp
$ JIT hello.ll
Hello world
```

创建 JIT 编译器非常容易！

示例使用 LLVM IR 作为输入，但这不是必需的。`LLJIT`类使用`IRCompileLayer`类，负责将 IR 编译为机器代码。您可以定义自己的层，接受您需要的输入，例如 Java 字节码。

使用预定义的 LLJIT 类很方便，但限制了我们的灵活性。在下一节中，我们将看看如何使用 ORC API 提供的层来实现 JIT 编译器。

## 从头开始构建 JIT 编译器类

使用 ORC 的分层方法，非常容易构建符合要求的 JIT 编译器。没有一种通用的 JIT 编译器，本章的第一部分给出了一些例子。让我们看看如何设置 JIT 编译器。

ORC API 使用堆叠在一起的层。最低级别是对象链接层，由`llvm::orc::RTDyldObjectLinkingLayer`类表示。它负责链接内存对象并将其转换为可执行代码。此任务所需的内存由`MemoryManager`接口的实例管理。有一个默认实现，但如果需要，我们也可以使用自定义版本。

对象链接层上面是编译层，负责创建内存中的目标文件。`llvm::orc::IRCompileLayer`类以 IR 模块作为输入，并将其编译为目标文件。`IRCompileLayer`类是`IRLayer`类的子类，后者是接受 LLVM IR 的层实现的通用类。

这两个层已经构成了 JIT 编译器的核心。它们将 LLVM IR 模块作为输入，编译并链接到内存中。要添加更多功能，我们可以在这两个层之上添加更多层。例如，`CompileOnDemandLayer`类将模块拆分，以便仅编译请求的函数。这可以用于实现延迟编译。`CompileOnDemandLayer`类也是`IRLayer`类的子类。以非常通用的方式，`IRTransformLayer`类，也是`IRLayer`类的子类，允许我们对模块应用转换。

另一个重要的类是`ExecutionSession`类。这个类表示正在运行的 JIT 程序。基本上，这意味着该类管理`JITDylib`符号表，为符号提供查找功能，并跟踪使用的资源管理器。

JIT 编译器的通用配方如下：

1.  初始化`ExecutionSession`类的一个实例。

1.  初始化层，至少包括`RTDyldObjectLinkingLayer`类和`IRCompileLayer`类。

1.  创建第一个`JITDylib`符号表，通常使用`main`或类似的名称。

使用方法与上一节的`LLJIT`类非常相似：

1.  将 IR 模块添加到符号表中。

1.  查找符号，触发相关函数的编译，可能是整个模块。

1.  执行函数。

在下一小节中，我们将基于通用配方实现一个 JIT 编译器类。

### 创建一个 JIT 编译器类

为了保持 JIT 编译器类的实现简单，我们将所有内容放入`JIT.h`头文件中。类的初始化有点复杂。由于需要处理可能的错误，我们需要一个工厂方法在调用构造函数之前创建一些对象。创建类的步骤如下：

1.  我们首先使用`JIT_H`预处理器定义保护头文件免受多次包含的影响：

```cpp
#ifndef JIT_H
#define JIT_H
```

1.  需要一堆包含文件。其中大多数提供与头文件同名的类。`Core.h`头文件提供了一些基本类，包括`ExecutionSession`类。`ExecutionUtils.h`头文件提供了`DynamicLibrarySearchGenerator`类来搜索库中的符号，我们已经在*使用 LLJIT 实现我们自己的 JIT 编译器*部分中使用过。`CompileUtils.h`头文件提供了`ConcurrentIRCompiler`类：

```cpp
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include     "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include     "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include     "llvm/ExecutionEngine/Orc/TargetProcessControl.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
```

1.  我们的新类是`JIT`类：

```cpp
class JIT {
```

1.  私有数据成员反映了 ORC 层和一个辅助类。 `ExecutionSession`，`ObjectLinkingLayer`，`CompileLayer`，`OptIRLayer`和`MainJITDylib`实例代表了运行中的 JIT 程序，层和符号表，如前所述。 `TargetProcessControl`实例用于与 JIT 目标进程进行交互。这可以是相同的进程，同一台机器上的另一个进程，或者是不同机器上的远程进程，可能具有不同的架构。 `DataLayout`和`MangleAndInterner`类需要以正确的方式操纵符号名称。符号名称是内部化的，这意味着所有相等的名称具有相同的地址。要检查两个符号名称是否相等，只需比较地址，这是一个非常快速的操作：

```cpp
  std::unique_ptr<llvm::orc::TargetProcessControl> 
    TPC;
  std::unique_ptr<llvm::orc::ExecutionSession> ES;
  llvm::DataLayout DL;
  llvm::orc::MangleAndInterner Mangle;
  std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer>
      ObjectLinkingLayer;
  std::unique_ptr<llvm::orc::IRCompileLayer>
      CompileLayer;
  std::unique_ptr<llvm::orc::IRTransformLayer>
      OptIRLayer;
  llvm::orc::JITDylib &MainJITDylib;
```

1.  初始化分为三个部分。在 C++中，构造函数不能返回错误。简单且推荐的解决方案是创建一个静态工厂方法，在构造对象之前进行错误处理。层的初始化更复杂，因此我们也为它们引入了工厂方法。

在`create()`工厂方法中，我们首先创建一个`SymbolStringPool`实例，用于实现字符串内部化，并由几个类共享。为了控制当前进程，我们创建一个`SelfTargetProcessControl`实例。如果我们想要针对不同的进程，则需要更改此实例。

然后，我们构造了一个`JITTargetMachineBuilder`实例，我们需要知道 JIT 进程的目标三元组。接下来，我们查询目标机器生成器以获取数据布局。如果生成器无法根据提供的三元组实例化目标机器，例如，因为对此目标的支持未编译到 LLVM 库中，这一步可能会失败：

```cpp
public:
  static llvm::Expected<std::unique_ptr<JIT>> create() {
    auto SSP =
        std::make_shared<llvm::orc::SymbolStringPool>();
    auto TPC =
        llvm::orc::SelfTargetProcessControl::Create(SSP);
    if (!TPC)
      return TPC.takeError();
    llvm::orc::JITTargetMachineBuilder JTMB(
        (*TPC)->getTargetTriple());
    auto DL = JTMB.getDefaultDataLayoutForTarget();
    if (!DL)
      return DL.takeError();
```

1.  在这一点上，我们已经处理了所有可能失败的调用。我们现在能够初始化`ExecutionSession`实例。最后，调用`JIT`类的构造函数，并将结果返回给调用者：

```cpp
    auto ES =
        std::make_unique<llvm::orc::ExecutionSession>(
            std::move(SSP));
    return std::make_unique<JIT>(
        std::move(*TPC), std::move(ES), 
        std::move(*DL),
        std::move(JTMB));
  }
```

1.  `JIT`类的构造函数将传递的参数移动到私有数据成员。通过调用带有`create`前缀的静态工厂名称构造层对象。每个`layer`工厂方法都需要引用`ExecutionSession`实例，将层连接到运行中的 JIT 会话。除了对象链接层位于层堆栈的底部之外，每个层还需要引用上一个层，说明了堆叠顺序：

```cpp
  JIT(std::unique_ptr<llvm::orc::TargetProcessControl>
          TPCtrl,
      std::unique_ptr<llvm::orc::ExecutionSession> ExeS,
      llvm::DataLayout DataL,
      llvm::orc::JITTargetMachineBuilder JTMB)
      : TPC(std::move(TPCtrl)), ES(std::move(ExeS)),
        DL(std::move(DataL)), Mangle(*ES, DL),
        ObjectLinkingLayer(std::move(
            createObjectLinkingLayer(*ES, JTMB))),
        CompileLayer(std::move(createCompileLayer(
            *ES, *ObjectLinkingLayer, 
             std::move(JTMB)))),
        OptIRLayer(std::move(
            createOptIRLayer(*ES, *CompileLayer))),
        MainJITDylib(ES->createBareJITDylib("<main>")) {
```

1.  在构造函数的主体中，我们添加了生成器来搜索当前进程的符号。`GetForCurrentProcess()`方法是特殊的，因为返回值包装在`Expected<>`模板中，表示也可以返回`Error`对象。但我们知道不会发生错误-当前进程最终会运行！因此，我们使用`cantFail()`函数解包结果，如果发生错误，它将终止应用程序：

```cpp
    MainJITDylib.addGenerator(llvm::cantFail(
        llvm::orc::DynamicLibrarySearchGenerator::
            GetForCurrentProcess(DL.getGlobalPrefix())));
  }
```

1.  要创建对象链接层，我们需要提供一个内存管理器。我们在这里坚持使用默认的`SectionMemoryManager`类，但如果需要，我们也可以提供不同的实现：

```cpp
  static std::unique_ptr<
      llvm::orc::RTDyldObjectLinkingLayer>
  createObjectLinkingLayer(
      llvm::orc::ExecutionSession &ES,
      llvm::orc::JITTargetMachineBuilder &JTMB) {
    auto GetMemoryManager = []() {
      return std::make_unique<
          llvm::SectionMemoryManager>();
    };
    auto OLLayer = std::make_unique<
        llvm::orc::RTDyldObjectLinkingLayer>(
        ES, GetMemoryManager);
```

1.  对于在 Windows 上使用的 COFF 目标文件格式存在一个小复杂性。这种文件格式不允许将函数标记为导出。这随后导致在对象链接层内部的检查失败：存储在符号中的标志与 IR 中的标志进行比较，由于缺少导出标记而导致不匹配。解决方案是仅针对这种文件格式覆盖标志。这完成了对象层的构建，并将对象返回给调用者：

```cpp
    if (JTMB.getTargetTriple().isOSBinFormatCOFF()) {
      OLLayer
         ->setOverrideObjectFlagsWithResponsibilityFlags(
              true);
      OLLayer
         ->setAutoClaimResponsibilityForObjectSymbols(
              true);
    }
    return std::move(OLLayer);
  }
```

1.  要初始化编译器层，需要一个`IRCompiler`实例。`IRCompiler`实例负责将 IR 模块编译成目标文件。如果我们的 JIT 编译器不使用线程，那么我们可以使用`SimpleCompiler`类，它使用给定的目标机器编译 IR 模块。`TargetMachine`类不是线程安全的，同样`SimpleCompiler`类也不是。为了支持多线程编译，我们使用`ConcurrentIRCompiler`类，它为每个要编译的模块创建一个新的`TargetMachine`实例。这种方法解决了多线程的问题：

```cpp
  static std::unique_ptr<llvm::orc::IRCompileLayer>
  createCompileLayer(
      llvm::orc::ExecutionSession &ES,
      llvm::orc::RTDyldObjectLinkingLayer &OLLayer,
      llvm::orc::JITTargetMachineBuilder JTMB) {
    auto IRCompiler = std::make_unique<
        llvm::orc::ConcurrentIRCompiler>(
        std::move(JTMB));
    auto IRCLayer =
        std::make_unique<llvm::orc::IRCompileLayer>(
            ES, OLLayer, std::move(IRCompiler));
    return std::move(IRCLayer);
  }
```

1.  我们不直接将 IR 模块编译成机器代码，而是安装一个优化 IR 的层。这是一个有意的设计决定：我们将我们的 JIT 编译器转变为一个优化的 JIT 编译器，它产生更快的代码，但需要更长的时间来生成，这对用户来说会有延迟。我们不添加延迟编译，所以当查找一个符号时，整个模块都会被编译。这可能会导致用户在看到代码执行之前花费相当长的时间。

```cpp
  static std::unique_ptr<llvm::orc::IRTransformLayer>
  createOptIRLayer(
      llvm::orc::ExecutionSession &ES,
      llvm::orc::IRCompileLayer &CompileLayer) {
    auto OptIRLayer =
        std::make_unique<llvm::orc::IRTransformLayer>(
            ES, CompileLayer,
            optimizeModule);
    return std::move(OptIRLayer);
  }
```

1.  `optimizeModule()`函数是对 IR 模块进行转换的一个示例。该函数以要转换的模块作为参数，并返回转换后的模块。由于 JIT 可能会使用多个线程，IR 模块被包装在一个`ThreadSafeModule`实例中：

```cpp
  static llvm::Expected<llvm::orc::ThreadSafeModule>
  optimizeModule(
      llvm::orc::ThreadSafeModule TSM,
      const llvm::orc::MaterializationResponsibility
          &R) {
```

1.  为了优化 IR，我们回顾一些来自*第八章*的信息，*优化 IR*，在*向编译器添加优化流水线*部分。我们需要一个`PassBuilder`实例来创建一个优化流水线。首先，我们定义了一些分析管理器，并在通行构建器中注册它们。然后，我们使用默认的优化流水线填充了一个`ModulePassManager`实例，用于`O2`级别。这再次是一个设计决定：`O2`级别已经产生了快速的机器代码，但比`O3`级别更快。之后，我们在模块上运行流水线。最后，优化后的模块返回给调用者：

```cpp
    TSM.withModuleDo([](llvm::Module &M) {
      bool DebugPM = false;
      llvm::PassBuilder PB(DebugPM);
      llvm::LoopAnalysisManager LAM(DebugPM);
      llvm::FunctionAnalysisManager FAM(DebugPM);
      llvm::CGSCCAnalysisManager CGAM(DebugPM);
      llvm::ModuleAnalysisManager MAM(DebugPM);
      FAM.registerPass(
          [&] { return PB.buildDefaultAAPipeline(); });
      PB.registerModuleAnalyses(MAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerLoopAnalyses(LAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
      llvm::ModulePassManager MPM =
          PB.buildPerModuleDefaultPipeline(
              llvm::PassBuilder::OptimizationLevel::O2,
              DebugPM);
      MPM.run(M, MAM);
    });
    return std::move(TSM);
  }
```

1.  `JIT`类的客户端需要一种添加 IR 模块的方法，我们使用`addIRModule()`函数提供这种方法。记住我们创建的层栈：我们必须将 IR 模块添加到顶层，否则我们可能会意外地绕过一些层。这将是一个不容易发现的编程错误：如果`OptIRLayer`成员被`CompileLayer`成员替换，那么我们的`JIT`类仍然可以工作，但不作为一个优化的 JIT，因为我们已经绕过了这一层。这在这个小实现中并不值得担心，但在一个大的 JIT 优化中，我们会引入一个函数来返回顶层层次：

```cpp
  llvm::Error addIRModule(
      llvm::orc::ThreadSafeModule TSM,
      llvm::orc::ResourceTrackerSP RT = nullptr) {
    if (!RT)
      RT = MainJITDylib.getDefaultResourceTracker();
    return OptIRLayer->add(RT, std::move(TSM));
  }
```

1.  同样，我们的 JIT 类的客户端需要一种查找符号的方法。我们将这个任务委托给`ExecutionSession`实例，传入主符号表的引用以及所请求符号的 mangled 和 internalized 名称：

```cpp
  llvm::Expected<llvm::JITEvaluatedSymbol>
  lookup(llvm::StringRef Name) {
    return ES->lookup({&MainJITDylib},
                      Mangle(Name.str()));
  }
```

将 JIT 编译器组合在一起相当容易。初始化这个类有点棘手，因为它涉及到`JIT`类的一个工厂方法和构造函数调用，以及每个层的工厂方法。这种分布是由于 C++的限制，尽管代码本身很简单。

在下一小节中，我们将使用我们的新 JIT 编译器类来实现一个命令行实用程序。

### 使用我们的新 JIT 编译器类

我们的新 JIT 编译器类的接口类似于*使用 LLJIT 实现我们自己的 JIT 编译器*部分中使用的`LLJIT`类。为了测试我们的新实现，我们从上一节中复制`LIT.cpp`类，并进行以下更改：

1.  为了能够使用我们的新类，我们包含`JIT.h`头文件。这取代了`llvm/ExecutionEngine/Orc/LLJIT.h`头文件，因为我们不再使用 LLJIT 类，所以它不再需要。

1.  在`jitmain()`函数中，我们用对我们的新`JIT::create()`方法的调用替换了对`orc::LLJITBuilder().create()`的调用。

1.  同样，在`jitmain()`函数中，我们删除了添加`DynamicLibrarySearchGenerator`类的代码。这个生成器已经集成在 JIT 类中。

这已经是需要改变的一切了！我们可以像在上一节中一样编译和运行更改后的应用程序，得到相同的结果。在底层，新类使用了固定的优化级别，因此对于足够大的模块，我们可以注意到启动和运行时的差异。

拥有 JIT 编译器可以激发新的想法。在下一节中，我们将看看如何将 JIT 编译器作为静态编译器的一部分来评估编译时的代码。

# 利用 JIT 编译器进行代码评估

编译器编写者付出了巨大的努力来生成最佳代码。一个简单而有效的优化是用两个常量替换算术运算的结果值。为了能够执行计算，嵌入了一个常量表达式的解释器。为了得到相同的结果，解释器必须实现与生成的机器代码相同的规则！当然，这可能是微妙错误的源泉。

另一种方法是使用相同的代码生成方法将常量表达式编译为 IR，然后让 JIT 编译和执行 IR。这个想法甚至可以进一步发展。在数学中，函数对于相同的输入总是产生相同的结果。对于计算机语言中的函数，这并不成立。一个很好的例子是`rand()`函数，它每次调用都返回一个随机值。在计算机语言中，具有与数学函数相同特性的函数称为**纯函数**。在表达式优化期间，我们可以 JIT 编译和执行只有常量参数的纯函数，并用 JIT 执行返回的结果替换对函数的调用。实际上，我们将函数的执行从运行时移到了编译时！

考虑交叉编译

在静态编译器中使用 JIT 编译器是一个有趣的选择。然而，如果编译器支持交叉编译，那么这种方法应该经过深思熟虑。通常会引起麻烦的候选者是浮点类型。C 语言中`long double`类型的精度通常取决于硬件和操作系统。一些系统使用 128 位浮点数，而其他系统只使用 64 位浮点数。80 位浮点类型仅在 x86 平台上可用，并且通常仅在 Windows 上使用。使用不同精度进行相同的浮点运算可能会导致巨大差异。在这种情况下，无法使用 JIT 编译进行评估。

很难确定一个函数是否是纯函数。常见的解决方案是应用一种启发式方法。如果一个函数既不通过指针也不通过聚合类型间接地读取或写入堆内存，并且只调用其他纯函数，那么它就是一个纯函数。开发人员可以帮助编译器，例如，用特殊的关键字或符号标记纯函数。在语义分析阶段，编译器可以检查违规情况。

在下一小节中，我们将更仔细地看一下在编译时尝试对函数进行 JIT 执行时对语言语义的影响。

## 识别语言语义

困难的部分确实是在语言语义层面决定哪些语言部分适合在编译时进行评估。排除对堆内存的访问是非常限制性的。一般来说，这排除了字符串处理，例如。当分配的内存的生存周期超过 JIT 执行的函数的生存周期时，使用堆内存就会变得棘手。这是一个程序状态，可能会影响其他结果，因此是危险的。另一方面，如果`malloc()`和`free()`函数有匹配的调用，那么内存只用于内部计算。在这种情况下，使用堆内存是安全的。但要证明这种条件并不容易。

在类似的层面上，JIT 执行函数中的无限循环可能会使编译器冻结。艾伦·图灵在 1936 年表明，没有机器可以决定一个函数是否会产生结果，或者它是否陷入无休止的循环。必须采取一些预防措施来避免这种情况，例如，在 JIT 执行的函数被终止之前设置一个运行时限制。

最后，允许更多功能，就必须更多地考虑安全性，因为编译器现在执行的是其他人编写的代码。想象一下，这段代码从互联网下载并运行文件，或者试图擦除硬盘：如果允许 JIT 执行函数有太多状态，我们也需要考虑这样的情况。

这个想法并不新鲜。D 编程语言有一个名为**编译时函数执行**的功能。参考编译器**dmd**通过在 AST 级别解释函数来实现这一功能。基于 LLVM 的 LDC 编译器具有一个试验性的功能，可以使用 LLVM JIT 引擎。您可以在 https://dlang.org/了解更多关于该语言和编译器的信息。

忽略语义上的挑战，实现并不那么困难。在“从头开始构建 JIT 编译器类”部分，我们使用`JIT`类开发了一个 JIT 编译器。我们在类中输入一个 IR 模块，然后可以查找并执行该模块中的函数。通过查看`tinylang`编译器的实现，我们可以清楚地识别对常量的访问，因为 AST 中有一个`ConstantAccess`节点。例如，有如下代码：

```cpp
  if (auto *Const = llvm::dyn_cast<ConstantAccess>(Expr)) {
    // Do something with the constant.
  }
```

与其解释表达式中的操作以推导常量的值，我们可以做如下操作：

1.  创建一个新的 IR 模块。

1.  在模块中创建一个 IR 函数，返回预期类型的值。

1.  使用现有的`emitExpr()`函数为表达式创建 IR，并使用最后一条指令返回计算出的值。

1.  JIT 执行函数以计算值。

这值得实现吗？LLVM 在优化管道中执行常量传播和函数内联。例如，一个简单的表达式如 4 + 5 在 IR 构造过程中已经被替换为结果。像最大公约数的计算这样的小函数会被内联。如果所有参数都是常量值，那么内联的代码会通过常量传播的计算结果被替换。

基于这一观察，这种方法的实现只有在编译时有足够的语言特性可供执行时才有用。如果是这种情况，那么使用给定的草图实现起来是相当容易的。

了解如何使用 LLVM 的 JIT 编译器组件使您能够以全新的方式使用 LLVM。除了实现类似 Java 虚拟机的 JIT 编译器之外，JIT 编译器还可以嵌入到其他应用程序中。这允许创造性的方法，比如在本节中所看到的将其用于静态编译器。

# 总结

在本章中，您学习了如何开发 JIT 编译器。您从 JIT 编译器的可能应用开始，并探索了 LLVM 动态编译器和解释器`lli`。使用预定义的`LLJIT`类，您自己构建了类似于`lli`的工具。为了能够利用 ORC API 的分层结构，您实现了一个优化的`JIT`类。在获得了所有这些知识之后，您探讨了在静态编译器内部使用 JIT 编译器的可能性，这是一些语言可以受益的特性。

在下一章中，您将学习如何为新的 CPU 架构向 LLVM 添加后端。
