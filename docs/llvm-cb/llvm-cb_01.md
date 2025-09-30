# 第一章. LLVM 设计和使用

在本章中，我们将涵盖以下主题：

+   理解模块化设计

+   交叉编译 Clang/LLVM

+   将 C 源代码转换为 LLVM 汇编

+   将 IR 转换为 LLVM 位码

+   将 LLVM 位码转换为目标机器汇编

+   将 LLVM 位码转换回 LLVM 汇编

+   转换 LLVM IR

+   链接 LLVM 位码

+   执行 LLVM 位码

+   使用 C 前端 Clang

+   使用 GO 前端

+   使用 DragonEgg

# 简介

在这个菜谱中，你将了解**LLVM**，它的设计和如何利用它提供的各种工具进行多种用途。你还将了解如何将简单的 C 代码转换为 LLVM 中间表示，以及如何将其转换为各种形式。你还将学习代码如何在 LLVM 源树中组织，以及如何使用它来编写你自己的编译器。

# 理解模块化设计

LLVM 被设计成一套库，与其他编译器（如**GNU 编译器集合**（**GCC**））不同。在这个菜谱中，我们将使用 LLVM 优化器来理解这个设计。由于 LLVM 优化器的设计是基于库的，它允许你以指定的顺序排列要运行的传递。此外，这种设计允许你选择可以运行的优化传递——也就是说，可能有一些优化对于你正在设计的系统类型可能没有用，只有少数优化是特定于该系统的。在查看传统的编译器优化器时，它们被构建成一个紧密相连的代码块，这使得很难将其分解成你可以轻松理解和使用的较小部分。在 LLVM 中，你不需要了解整个系统的工作原理，就可以了解一个特定的优化器。你可以只选择一个优化器并使用它，而无需担心与之相连的其他组件。

在我们继续深入了解这个菜谱之前，我们还必须了解一些关于 LLVM 汇编语言的知识。LLVM 代码以三种形式表示：内存中的编译器**中间表示**（**IR**）、磁盘上的位码表示，以及人类可读的汇编。LLVM 是一个基于**静态单赋值**（**SSA**）的表示，它提供了类型安全、低级操作、灵活性和表示所有高级语言的清洁能力。这种表示在整个 LLVM 编译策略的所有阶段都得到使用。LLVM 表示旨在成为一个通用的 IR，因为它处于足够低的级别，使得高级思想可以干净地映射到它。此外，LLVM 汇编语言是格式良好的。如果你对菜谱中提到的 LLVM 汇编有任何疑问，请参考菜谱末尾的*也见*部分提供的链接。

## 准备工作

我们必须在我们的主机机器上安装 LLVM 工具链。具体来说，我们需要`opt`工具。

## 如何做...

我们将对相同的代码运行两种不同的优化，一个接一个，看看它如何根据我们选择的优化来修改代码。

1.  首先，让我们编写一个可以输入这些优化的代码。这里我们将将其写入一个名为 `testfile.ll:` 的文件中。

    ```cpp
    $ cat testfile.ll
    define i32 @test1(i32 %A) {
     %B = add i32 %A, 0
     ret i32 %B
    }

    define internal i32 @test(i32 %X, i32 %dead) {
     ret i32 %X
    }

    define i32 @caller() {
     %A = call i32 @test(i32 123, i32 456)
     ret i32 %A
    }

    ```

1.  现在，运行 `opt` 工具进行其中一个优化——即，用于合并指令：

    ```cpp
    $ opt –S –instcombine testfile.ll –o output1.ll

    ```

1.  查看输出以了解 `instcombine` 的工作情况：

    ```cpp
    $ cat output1.ll
    ; ModuleID = 'testfile.ll'

    define i32 @test1(i32 %A) {
     ret i32 %A
    }

    define internal i32 @test(i32 %X, i32 %dead) {
     ret i32 %X
    }

    define i32 @caller() {
     %A = call i32 @test(i32 123, i32 456)
     ret i32 %A
    }

    ```

1.  运行 `opt` 命令进行无效参数消除优化：

    ```cpp
    $ opt –S –deadargelim testfile.ll –o output2.ll

    ```

1.  查看输出，以了解 `deadargelim` 的工作情况：

    ```cpp
    $ cat output2.ll
    ; ModuleID = testfile.ll'

    define i32 @test1(i32 %A) {
     %B = add i32 %A, 0
     ret i32 %B
    }

    define internal i32 @test(i32 %X) {
     ret i32 %X
    }

    define i32 @caller() {
     %A = call i32 @test(i32 123)
     ret i32 %A
    }

    ```

## 工作原理...

在前面的例子中，我们可以看到，对于第一个命令，运行了 `instcombine` 过滤器，它合并了指令，因此将 `%B = add i32 %A, 0; ret i32 %B` 优化为 `ret i32 %A`，而不影响代码。

在第二种情况下，当运行 `deadargelim 过滤器` 时，我们可以看到第一个函数没有修改，但上次未修改的代码部分现在被修改了，未使用的函数参数被消除。

LLVM 优化器是提供给用户所有不同过滤步骤的工具。这些过滤步骤都是用类似风格编写的。对于这些过滤步骤中的每一个，都有一个编译后的目标文件。不同过滤步骤的目标文件被存档到一个库中。库中的过滤步骤不是强连接的，LLVM **PassManager** 拥有关于过滤步骤之间依赖关系的信息，它在执行过滤步骤时解决这些依赖关系。以下图像显示了每个过滤步骤如何链接到特定库中的特定目标文件。在以下图中，**PassA** 引用了 **LLVMPasses.a** 中的 **PassA.o**，而自定义过滤步骤则引用了不同的库 **MyPasses.a** 中的 **MyPass.o** 目标文件。

### 小贴士

**下载示例代码**

您可以从您在 [`www.packtpub.com`](http://www.packtpub.com) 的账户中下载您购买的所有 Packt 书籍的示例代码文件。如果您在其他地方购买了这本书，您可以访问 [`www.packtpub.com/support`](http://www.packtpub.com/support) 并注册，以便将文件直接通过电子邮件发送给您。

![如何工作...](img/image00251.jpeg)

## 更多内容...

与优化器类似，LLVM 代码生成器也利用其模块化设计，将代码生成问题分解为单个过滤步骤：指令选择、寄存器分配、调度、代码布局优化和汇编输出。此外，还有许多默认运行的内置过滤步骤。用户可以选择运行哪些过滤步骤。

## 相关链接

+   在接下来的章节中，我们将看到如何编写我们自己的自定义过滤步骤，其中我们可以选择运行哪些优化过滤步骤以及它们的顺序。此外，为了更深入的了解，请参阅 [`www.aosabook.org/en/llvm.html`](http://www.aosabook.org/en/llvm.html)。

+   要了解有关 LLVM 汇编语言的更多信息，请参阅 [`llvm.org/docs/LangRef.html`](http://llvm.org/docs/LangRef.html)。

# 交叉编译 Clang/LLVM

通过交叉编译，我们指的是在一个平台上（例如，x86）构建一个将在另一个平台上（例如，ARM）运行的二进制文件。我们构建二进制文件的机器称为宿主，而生成二进制文件将要运行的机器称为目标。为运行在其上的同一平台构建代码的编译器（宿主和目标平台相同）称为**原生汇编器**，而为宿主平台不同的目标平台构建代码的编译器称为**交叉**编译器。

在这个配方中，将展示为不同于宿主平台的平台交叉编译 LLVM，这样您就可以使用为所需目标平台构建的二进制文件。在这里，将通过一个示例来展示交叉编译，即从宿主平台 x86_64 交叉编译到目标平台 ARM。生成的二进制文件可以在具有 ARM 架构的平台中使用。

## 准备工作

以下软件包需要在您的系统（宿主平台）上安装：

+   `cmake`

+   `ninja-build`（来自 Ubuntu 的 backports）

+   `gcc-4.x-arm-linux-gnueabihf`

+   `gcc-4.x-multilib-arm-linux-gnueabihf`

+   `binutils-arm-linux-gnueabihf`

+   `libgcc1-armhf-cross`

+   `libsfgcc1-armhf-cross`

+   `libstdc++6-armhf-cross`

+   `libstdc++6-4.x-dev-armhf-cross`

+   `在您的宿主平台上安装 llvm`

## 如何操作...

要从宿主架构（即此处为 **X86_64**）编译 ARM 目标，需要执行以下步骤：

1.  将以下 `cmake` 标志添加到 LLVM 的正常 `cmake` 构建中：

    ```cpp
    -DCMAKE_CROSSCOMPILING=True
    -DCMAKE_INSTALL_PREFIX= path-where-you-want-the-toolchain(optional)
    -DLLVM_TABLEGEN=<path-to-host-installed-llvm-toolchain-bin>/llvm-tblgen
    -DCLANG_TABLEGEN=< path-to-host-installed-llvm-toolchain-bin >/clang-tblgen
    -DLLVM_DEFAULT_TARGET_TRIPLE=arm-linux-gnueabihf
    -DLLVM_TARGET_ARCH=ARM
    -DLLVM_TARGETS_TO_BUILD=ARM
    -DCMAKE_CXX_FLAGS='-target armv7a-linux-gnueabihf -mcpu=cortex-a9 -I/usr/arm-linux-gnueabihf/include/c++/4.x.x/arm-linux-gnueabihf/ -I/usr/arm-linux-gnueabihf/include/ -mfloat-abi=hard -ccc-gcc-name arm-linux-gnueabihf-gcc'

    ```

1.  如果使用你的平台编译器，请运行：

    ```cpp
    $ cmake -G Ninja <llvm-source-dir> <options above>

    ```

    如果使用 Clang 作为交叉编译器，请运行：

    ```cpp
    $ CC='clang' CXX='clang++' cmake -G Ninja <source-dir> <options above>

    ```

    如果你的路径上有 clang/Clang++，它应该可以正常工作。

1.  要构建 LLVM，只需输入：

    ```cpp
    $ ninja

    ```

1.  在 LLVM/Clang 构建成功后，使用以下命令安装它：

    ```cpp
    $ ninja install

    ```

如果您已指定 `DCMAKE_INSTALL_PREFIX` 选项，这将创建 `install-dir` 位置的 `sysroot`。

## 它是如何工作的...

`cmake` 软件包通过使用传递给 `cmake` 的选项标志来构建所需平台的工具链，并且使用 `tblgen` 工具将目标描述文件转换为 C++ 代码。因此，通过使用它，可以获得有关目标的信息，例如——目标上可用的指令、寄存器数量等等。

### 注意

如果使用 Clang 作为交叉编译器，LLVM ARM 后端存在一个问题，在**位置无关代码**（**PIC**）上产生绝对重定位，因此作为解决方案，暂时禁用 PIC。

ARM 库在宿主系统上不可用。因此，要么下载它们的副本，要么在您的系统上构建它们。

# 将 C 源代码转换为 LLVM 汇编代码

在这里，我们将使用 C 前端 Clang 将 C 代码转换为 LLVM 的中间表示。

## 准备工作

Clang 必须安装到 PATH 中。

## 如何操作...

1.  让我们在 `multiply.c` 文件中创建一个 C 代码，它看起来可能如下所示：

    ```cpp
    $ cat multiply.c
    int mult() {
    int a =5;
    int b = 3;
    int c = a * b;
    return c;
    }

    ```

1.  使用以下命令从 C 代码生成 LLVM IR：

    ```cpp
    $ clang -emit-llvm -S multiply.c -o multiply.ll

    ```

1.  查看生成的 IR：

    ```cpp
    $ cat multiply.ll
    ; ModuleID = 'multiply.c'
    target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
    target triple = "x86_64-unknown-linux-gnu"

    ; Function Attrs: nounwind uwtable
    define i32 @mult() #0 {
     %a = alloca i32, align 4
     %b = alloca i32, align 4
     %c = alloca i32, align 4
     store i32 5, i32* %a, align 4
     store i32 3, i32* %b, align 4
     %1 = load i32* %a, align 4
     %2 = load i32* %b, align 4
     %3 = mul nsw i32 %1, %2
     store i32 %3, i32* %c, align 4
     %4 = load i32* %c, align 4
     ret i32 %4
    }

    ```

    我们也可以使用 `cc1` 生成 IR：

    ```cpp
    $ clang -cc1 -emit-llvm testfile.c -o testfile.ll

    ```

## 它是如何工作的...

C 代码转换为 IR 的过程始于词法分析过程，其中 C 代码被分解成一个标记流，每个标记代表一个标识符、字面量、运算符等。这个标记流被送入解析器，在语言的帮助下使用**上下文无关文法**（**CFG**）构建一个抽象语法树。之后进行语义分析以检查代码是否语义正确，然后我们生成代码到 IR。

这里我们使用 Clang 前端从 C 代码生成 IR 文件。

## 参见

+   在下一章中，我们将看到词法分析和解析器是如何工作的，以及代码生成是如何进行的。要了解 LLVM IR 的基础知识，可以参考 [`llvm.org/docs/LangRef.html`](http://llvm.org/docs/LangRef.html)。

# 将 IR 转换为 LLVM 位码

在这个菜谱中，你将学习如何从 IR 生成 LLVM 位码。LLVM 位码文件格式（也称为字节码）实际上是两件事：一个位流容器格式和将 LLVM IR 编码到容器格式中的编码。

## 准备工作

`llvm-as` 工具必须安装到 PATH 中。

## 如何做...

执行以下步骤：

1.  首先创建一个将被用作 `llvm-as` 输入的 IR 代码：

    ```cpp
    $ cat test.ll
    define i32 @mult(i32 %a, i32 %b) #0 {
     %1 = mul nsw i32 %a, %b
     ret i32 %1
    }

    ```

1.  要将 `test.ll` 中的 LLVM IR 转换为位码格式，你需要使用以下命令：

    ```cpp
    llvm-as test.ll –o test.bc

    ```

1.  输出生成在 `test.bc` 文件中，该文件为位流格式；因此，当我们想以文本格式查看输出时，我们得到如下截图所示：![如何做...](img/image00252.jpeg)

    由于这是一个位码文件，查看其内容最好的方式是使用 `hexdump` 工具。以下截图显示了 `hexdump` 的输出：

    ![如何做...](img/image00253.jpeg)

## 它是如何工作的...

`llvm-as` 是 LLVM 汇编器。它将 LLVM 汇编文件（即 LLVM IR）转换为 LLVM 位码。在上面的命令中，它以 `test.ll` 文件作为输入和输出，并以 `test.bc` 作为位码文件。

## 更多内容...

为了将 LLVM IR 编码为位码，使用了块和记录的概念。块代表位流的区域，例如——函数体、符号表等。每个块都有一个特定于其内容的 ID（例如，LLVM IR 中的函数体由 ID 12 表示）。记录由一个记录代码和一个整数值组成，它们描述了文件中的实体，如指令、全局变量描述符、类型描述等。

LLVM IR 的位码文件可能被一个简单的包装结构所封装。这个结构包含一个简单的头部，指示嵌入的 BC 文件的偏移量和大小。

## 参见

+   要详细了解 LLVM 位流文件格式，请参阅 [`llvm.org/docs/BitCodeFormat.html#abstract`](http://llvm.org/docs/BitCodeFormat.html#abstract)

# 将 LLVM 位码转换为目标机器汇编

在本食谱中，您将学习如何将 LLVM 位码文件转换为特定目标的汇编代码。

## 准备工作

The LLVM 静态编译器 `llc` 应该是从 LLVM 工具链中安装的。

## 如何操作...

执行以下步骤：

1.  在前面的食谱中创建的位码文件 `test.bc`，可以用作 `llc` 的输入。使用以下命令，我们可以将 LLVM 位码转换为汇编代码：

    ```cpp
    $ llc test.bc –o test.s

    ```

1.  输出文件生成在 `test.s` 文件中，这是汇编代码。要查看它，请使用以下命令行：

    ```cpp
    $ cat test.s
    .text
    .file "test.bc"
    .globl mult
    .align 16, 0x90
    .type mult,@function
    mult:                                   # @mult
    .cfi_startproc
    # BB#0:
    Pushq  %rbp
    .Ltmp0:
    .cfi_def_cfa_offset 16
    .Ltmp1:
    .cfi_offset %rbp, -16
    movq %rsp, %rbp
    .Ltmp2:
    .cfi_def_cfa_register %rbp
    imull %esi, %edi
    movl %edi, %eax
    popq %rbp
    retq
    .Ltmp3:
    .size mult, .Ltmp3-mult
    .cfi_endproc

    ```

1.  您还可以使用 Clang 从位码文件格式中转储汇编代码。通过传递 `–S` 选项给 Clang，当 `test.bc` 文件处于位流文件格式时，我们得到 `test.s` 的汇编格式：

    ```cpp
    $ clang -S test.bc -o test.s –fomit-frame-pointer # using the clang front end

    ```

    输出的 `test.s` 文件与前面的示例相同。我们使用额外的选项 `fomit-frame-pointer`，因为 Clang 默认不消除帧指针，而 `llc` 默认消除它。

## 它是如何工作的...

`llc` 命令将指定架构的 LLVM 输入编译成汇编语言。如果我们没有在前面命令中提及任何架构，汇编代码将为 `llc` 命令正在使用的宿主机器生成。要从汇编文件生成可执行文件，您可以使用汇编器和链接器。

## 还有更多...

通过在前面命令中指定 `-march=architecture` 标志，您可以指定需要生成汇编代码的目标架构。使用 `-mcpu=cpu` 标志设置，您可以指定架构内的 CPU 以生成代码。还可以通过指定 `-regalloc=basic/greedy/fast/pbqp` 来指定要使用的寄存器分配类型。

# 将 LLVM 位码转换回 LLVM 汇编

在本食谱中，您将把 LLVM 位码转换回 LLVM IR。实际上，这是通过使用名为 `llvm-dis` 的 LLVM 反汇编工具来实现的。

## 准备工作

要完成此操作，您需要安装 `llvm-dis` 工具。

## 如何操作...

要查看位码文件如何转换为 IR，请使用在“将 IR 转换为 LLVM Bitcode”食谱中生成的 `test.bc` 文件。`test.bc` 文件作为输入提供给 `llvm-dis` 工具。现在按照以下步骤进行：

1.  使用以下命令显示如何将位码文件转换为我们在 IR 文件中创建的文件：

    ```cpp
    $ llvm-dis test.bc –o test.ll

    ```

1.  以下是如何生成 LLVM IR：

    ```cpp
    | $ cat test.ll
    ; ModuleID = 'test.bc'

    define i32 @mult(i32 %a, i32 %b) #0 {
     %1 = mul nsw i32 %a, %b
     ret i32 %1
    }

    ```

    输出文件 `test.ll` 与我们在“将 IR 转换为 LLVM Bitcode”食谱中创建的文件相同。

## 它是如何工作的...

`llvm-dis` 命令是 LLVM 反汇编器。它接受一个 LLVM 位码文件并将其转换为 LLVM 汇编语言。

在这里，输入文件是 `test.bc`，它通过 `llvm-dis` 转换为 `test.ll`。

如果省略了文件名，`llvm-dis` 将从标准输入读取其输入。

# 转换 LLVM IR

在本配方中，我们将看到如何使用 opt 工具将 IR 从一种形式转换为另一种形式。我们将看到应用于 IR 代码的不同优化。

## 准备工作

您需要安装 opt 工具。

## 如何做到这一点...

`opt`工具按照以下命令运行变换传递：

```cpp
$opt –passname input.ll –o output.ll

```

1.  让我们用一个实际的例子来。我们创建与配方*将 C 源代码转换为 LLVM 汇编*中使用的 C 代码等价的 LLVM IR：

    ```cpp
    $ cat multiply.c
    int mult() {
    int a =5;
    int b = 3;
    int c = a * b;
    return c;
    }

    ```

1.  转换并输出后，我们得到未优化的输出：

    ```cpp
    $ clang -emit-llvm -S multiply.c -o multiply.ll
    $ cat multiply.ll
    ; ModuleID = 'multiply.c'
    target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
    target triple = "x86_64-unknown-linux-gnu"

    ; Function Attrs: nounwind uwtable
    define i32 @mult() #0 {
     %a = alloca i32, align 4
     %b = alloca i32, align 4
     %c = alloca i32, align 4
     store i32 5, i32* %a, align 4
     store i32 3, i32* %b, align 4
     %1 = load i32* %a, align 4
     %2 = load i32* %b, align 4
     %3 = mul nsw i32 %1, %2
     store i32 %3, i32* %c, align 4
     %4 = load i32* %c, align 4
     ret i32 %4
    }

    ```

1.  现在使用 opt 工具将其转换为将内存提升到寄存器的形式：

    ```cpp
    $ opt -mem2reg -S multiply.ll -o multiply1.ll
    $ cat multiply1.ll
    ; ModuleID = 'multiply.ll'
    target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
    target triple = "x86_64-unknown-linux-gnu"

    ; Function Attrs: nounwind uwtable
    define i32 @mult(i32 %a, i32 %b) #0 {
     %1 = mul nsw i32 %a, %b
     ret i32 %1
    }

    ```

## 它是如何工作的...

`opt`、LLVM 优化器和分析工具将`input.ll`文件作为输入，并在其上运行`passname`传递。运行传递后的输出保存在包含变换后 IR 代码的`output.ll`文件中。opt 工具可以传递多个传递。

## 更多...

当传递`-analyze`选项给 opt 时，它对输入源执行各种分析，并将结果通常打印到标准输出或标准错误。此外，当输出要被馈送到另一个程序时，可以将输出重定向到文件。

当没有传递`-analyze`选项给 opt 时，它运行旨在优化输入文件的变换传递。

以下是一些重要的变换，可以作为标志传递给 opt 工具：

+   `adce`：激进死代码消除

+   `bb-vectorize`：基本块向量化

+   `constprop`：简单的常量传播

+   `dce`：删除死代码

+   `deadargelim`：删除死参数

+   `globaldce`：删除死全局变量

+   `globalopt`：全局变量优化器

+   `gvn`：全局值编号

+   `inline`：函数集成/内联

+   `instcombine`：合并冗余指令

+   `licm`：循环不变代码移动

+   `loop`：unswitch：取消切换循环

+   `loweratomic`：将原子内联函数降低为非原子形式

+   `lowerinvoke`：降低调用到调用，用于无回滚代码生成器

+   `lowerswitch`：将 SwitchInsts 降低为分支

+   `mem2reg`：提升内存到寄存器

+   `memcpyopt`：MemCpy 优化

+   `simplifycfg`：简化 CFG

+   `sink`：代码下沉

+   `tailcallelim`：尾调用消除

至少运行前面的一些传递，以了解它们是如何工作的。要到达可能适用于这些传递的适当源代码，请转到`llvm/test/Transforms`目录。对于上述提到的每个传递，您都可以看到测试代码。应用相关的传递，并查看测试代码是如何被修改的。

### 注意

要查看 C 代码如何转换为 IR 的映射，在将 C 代码转换为 IR 后，如在前面的配方*将 C 源代码转换为 LLVM 汇编*中讨论的那样，运行`mem2reg`传递。然后它将帮助您了解 C 指令是如何映射到 IR 指令的。

# 链接 LLVM 位代码

在本节中，您将链接之前生成的`.bc`文件，以获得包含所有所需引用的单个位代码文件。

## 准备工作

要链接 `.bc` 文件，你需要 `llvm-link` 工具。

## 如何操作...

执行以下步骤：

1.  要展示 `llvm-link` 的工作原理，首先在两个不同的文件中编写两个代码，其中一个引用另一个：

    ```cpp
    $ cat test1.c
    int func(int a) {
    a = a*2;
    return a;
    }
    $ cat test2.c
    #include<stdio.h>
    extern int func(int a);
    int main() {
    int num = 5;
    num = func(num);
    printf("number is %d\n", num);
    return num;
    }

    ```

1.  使用以下格式将此 C 代码转换为位流文件格式，首先转换为 `.ll` 文件，然后从 `.ll` 文件转换为 `.bc` 文件：

    ```cpp
    $ clang -emit-llvm -S test1.c -o test1.ll
    $ clang -emit-llvm -S test2.c -o test2.ll
    $ llvm-as test1.ll -o test1.bc
    $ llvm-as test2.ll -o test2.bc

    ```

    我们使用 `test1.bc` 和 `test2.bc`，其中 `test2.bc` 引用了 `test1.bc` 文件中的 `func` 语法。

1.  以以下方式调用 `llvm-link` 命令来链接两个 LLVM 位码文件：

    ```cpp
    $ llvm-link test1.bc test2.bc –o output.bc

    ```

我们向 `llvm-link` 工具提供多个位码文件，它将它们链接在一起以生成单个位码文件。这里，`output.bc` 是生成的输出文件。我们将在下一个步骤 *执行 LLVM 位码* 中执行此位码文件。

## 它是如何工作的...

`llvm-link` 使用链接器的基本功能——也就是说，如果一个文件中引用的函数或变量在另一个文件中定义，那么链接器的任务是解决文件中所有引用和在另一个文件中定义的内容。但请注意，这并不是传统链接器，它将各种目标文件链接起来生成二进制文件。`llvm-link` 工具仅链接位码文件。

在先前的场景中，它是将 `test1.bc` 和 `test2.bc` 文件链接起来生成 `output.bc` 文件，其中引用已解决。

### 注意

在链接位码文件后，我们可以通过给 `llvm-link` 工具提供 `–S` 选项来生成输出作为 IR 文件。

# 执行 LLVM 位码

在本步骤中，你将执行先前步骤中生成的 LLVM 位码。

## 准备工作

要执行 LLVM 位码，你需要 `lli` 工具。

## 如何操作...

在先前的步骤中，我们看到了如何通过将两个 `.bc` 文件链接起来并定义 `func` 来创建单个位流文件。通过以下方式调用 `lli` 命令，我们可以执行生成的 `output.bc` 文件。它将在标准输出上显示输出：

```cpp
| $ lli output.bc
 number is 10

```

`The output.bc` 文件是 `lli` 的输入，它将执行位码文件，并在标准输出上显示任何输出。在这里，输出生成了数字 `10`，这是在先前的步骤中将 `test1.c` 和 `test2.c` 链接形成的 `output.bc` 文件执行的结果。`test2.c` 文件中的主函数以整数 5 作为参数调用 `test1.c` 文件中的 `func` 函数。`func` 函数将输入参数加倍，并将结果返回给主函数，主函数将其输出到标准输出。

## 它是如何工作的...

`lli` 工具命令执行以 LLVM 位码格式存在的程序。它接受以 LLVM 位码格式输入，并使用即时编译器执行它，如果架构有可用的即时编译器，或者使用解释器。

如果 `lli` 正在使用即时编译器，那么它实际上将所有代码生成器选项视为 `llc` 的选项。

## 参见

+   在第三章的*扩展前端和添加 JIT 支持*的*为语言添加 JIT 支持*菜谱中。

# 使用 C 前端 Clang

在这个菜谱中，你将了解如何使用 Clang 前端实现不同的目的。

## 准备工作

你将需要 Clang 工具。

## 如何做...

Clang 可以用作高级编译器驱动程序。让我们用一个例子来展示它：

1.  创建一个`hello world` C 代码，`test.c`：

    ```cpp
    $ cat test.c
    #include<stdio.h>
    int main() {
    printf("hello world\n");
    return 0; }

    ```

1.  使用 Clang 作为编译器驱动程序生成可执行文件`a.out`，执行时给出预期的输出：

    ```cpp
    $ clang test.c
    $ ./a.out
    hello world

    ```

    在这里创建了包含 C 代码的`test.c`文件。使用 Clang 编译它并生成一个可执行文件，执行时给出期望的结果。

1.  可以通过提供`–E`标志来仅使用 Clang 的预处理器模式。在以下示例中，创建一个包含`#define`指令定义 MAX 值的 C 代码，并使用此 MAX 作为你将要创建的数组的大小：

    ```cpp
    $ cat test.c
    #define MAX 100
    void func() {
    int a[MAX];
    }

    ```

1.  使用以下命令运行预处理器，该命令在标准输出上显示输出：

    ```cpp
    $ clang test.c -E
    # 1 "test.c"
    # 1 "<built-in>" 1
    # 1 "<built-in>" 3
    # 308 "<built-in>" 3
    # 1 "<command line>" 1
    # 1 "<built-in>" 2
    # 1 "test.c" 2

    void func() {
    int a[100];
    }

    ```

    在本菜谱的所有后续部分中都将使用的`test.c`文件中，MAX 被定义为`100`，在预处理过程中被替换为`a[MAX]`中的 MAX，变为`a[100]`。

1.  你可以使用以下命令从先前的示例中打印`test.c`文件的 AST，该命令在标准输出上显示输出：

    ```cpp
    | $ clang -cc1 test.c -ast-dump
    TranslationUnitDecl 0x3f72c50 <<invalid sloc>> <invalid sloc>|-TypedefDecl 0x3f73148 <<invalid sloc>> <invalid sloc> implicit __int128_t '__int128'|-TypedefDecl 0x3f731a8 <<invalid sloc>> <invalid sloc> implicit __uint128_t 'unsigned __int128'|-TypedefDecl 0x3f73518 <<invalid sloc>> <invalid sloc> implicit __builtin_va_list '__va_list_tag [1]'`-FunctionDecl 0x3f735b8 <test.c:3:1, line:5:1> line:3:6 func 'void ()'`-CompoundStmt 0x3f73790 <col:13, line:5:1>`-DeclStmt 0x3f73778 <line:4:1, col:11>`-VarDecl 0x3f73718 <col:1, col:10> col:5 a 'int [100]'

    ```

    在这里，`–cc1`选项确保只运行编译器前端，而不是驱动程序，并打印与`test.c`文件代码对应的 AST。

1.  你可以使用以下命令为先前示例中的`test.c`文件生成 LLVM 汇编代码：

    ```cpp
    |$ clang test.c -S -emit-llvm -o -
    |; ModuleID = 'test.c'
    |target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
    |target triple = "x86_64-unknown-linux-gnu"
    |
    |; Function Attrs: nounwind uwtable
    |define void @func() #0 {
    |%a = alloca [100 x i32], align 16
    |ret void
    |}

    ```

    `–S`和`–emit-llvm`标志确保为`test.c`代码生成 LLVM 汇编。

1.  要获取相同`test.c`测试代码的机器代码，将`–S`标志传递给 Clang。由于`–o –`选项，它将在标准输出上生成输出：

    ```cpp
    |$ clang -S test.c -o -
    |	.text
    |	.file	"test.c"
    |	.globl	func
    |	.align	16, 0x90
    |	.type	func,@function
    |func:                                   # @func
    |	.cfi_startproc
    |# BB#0:
    |	pushq	%rbp
    |.Ltmp0:
    |	.cfi_def_cfa_offset 16
    |.Ltmp1:
    |	.cfi_offset %rbp, -16
    |	movq	%rsp, %rbp
    |.Ltmp2:
    |	.cfi_def_cfa_register %rbp
    |	popq	%rbp
    |	retq
    |.Ltmp3:
    |	.size	func, .Ltmp3-func
    |	.cfi_endproc

    ```

当单独使用`–S`标志时，编译器的代码生成过程将生成机器代码。在这里，运行命令时，由于使用了`–o –`选项，机器代码将在标准输出上输出。

## 它是如何工作的...

在先前的示例中，Clang 作为预处理器、编译器驱动程序、前端和代码生成器工作，因此根据给定的输入标志给出期望的输出。

## 参见

+   这是对如何使用 Clang 的基本介绍。还有许多其他可以传递给 Clang 的标志，使其执行不同的操作。要查看列表，使用 Clang `–help`。

# 使用 GO 前端

`llgo`编译器是仅用 Go 语言编写的基于 LLVM 的 Go 语言前端。使用此前端，我们可以从用 Go 编写的程序中生成 LLVM 汇编代码。

## 准备工作

你需要下载`llgo`的二进制文件或从源代码构建`llgo`，并将二进制文件添加到配置的`PATH`文件位置。

## 如何做...

执行以下步骤：

1.  创建一个 Go 源文件，例如，该文件将用于使用 `llgo` 生成 LLVM 汇编。创建 `test.go`：

    ```cpp
    |$ cat test.go
    |package main
    |import "fmt"
    |func main() {
    | fmt.Println("Test Message")
    |}

    ```

1.  现在，使用 `llgo` 获取 LLVM 汇编：

    ```cpp
    $llgo -dump test.go
    ; ModuleID = 'main'
    target datalayout = "e-p:64:64:64..."
    target triple = "x86_64-unknown-linux"
    %0 = type { i8*, i8* }
    ....

    ```

## 工作原理…

`llgo` 编译器是 Go 语言的接口；它将 `test.go` 程序作为其输入并输出 LLVM IR。

## 参见

+   关于如何获取和安装 `llgo` 的信息，请参阅 [`github.com/go-llvm/llgo`](https://github.com/go-llvm/llgo)

# 使用 DragonEgg

Dragonegg 是一个 gcc 插件，允许 gcc 使用 LLVM 优化器和代码生成器，而不是使用 gcc 自身的优化器和代码生成器。

## 准备工作

您需要具备 gcc 4.5 或更高版本，目标机器为 `x86-32/x86-64` 和 ARM 处理器。此外，您还需要下载 Dragonegg 源代码并构建 `dragonegg.so` 文件。

## 如何操作…

执行以下步骤：

1.  创建一个简单的 `hello world` 程序：

    ```cpp
    $ cat testprog.c
    #include<stdio.h>
    int main() {
    printf("hello world");
    }

    ```

1.  使用您的 gcc 编译此程序；这里我们使用 gcc-4.5：

    ```cpp
    $ gcc testprog.c -S -O1 -o -
     .file  " testprog.c"
     .section  .rodata.str1.1,"aMS",@progbits,1
    .LC0:
     .string  "Hello world!"
     .text
    .globl main
     .type  main, @function
    main:
     subq  $8, %rsp
     movl  $.LC0, %edi
     call  puts
     movl  $0, %eax
     addq  $8, %rsp
     ret
     .size  main, .-main

    ```

1.  在 gcc 的命令行中使用 `-fplugin=path/dragonegg.so` 标志使 gcc 使用 LLVM 的优化器和 LLVM 代码生成器：

    ```cpp
    $ gcc testprog.c -S -O1 -o - -fplugin=./dragonegg.so
     .file  " testprog.c"
    # Start of file scope inline assembly
     .ident  "GCC: (GNU) 4.5.0 20090928 (experimental) LLVM: 82450:82981"
    # End of file scope inline assembly

     .text
     .align  16
     .globl  main
     .type  main,@function
    main:
     subq  $8, %rsp
     movl  $.L.str, %edi
     call  puts
     xorl  %eax, %eax
     addq  $8, %rsp
     ret
     .size  main, .-main
     .type  .L.str,@object
     .section  .rodata.str1.1,"aMS",@progbits,1
    .L.str:
     .asciz  "Hello world!"
     .size  .L.str, 13

     .section  .note.GNU-stack,"",@progbits

    ```

## 参见

+   关于如何获取源代码和安装过程的信息，请参阅 [`dragonegg.llvm.org/`](http://dragonegg.llvm.org/)
