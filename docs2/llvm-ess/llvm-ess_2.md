# 第二章：构建 LLVM IR

高级编程语言便于人与目标机器的交互。今天的大多数流行高级语言都有一些基本元素，如变量、循环、if-else 决策语句、块、函数等。变量持有数据类型的价值；基本块给出了变量的作用域的概念。if-else 决策语句有助于选择代码路径。函数使代码块可重用。高级语言可能在类型检查、类型转换、变量声明、复杂数据类型等方面有所不同。然而，几乎每种语言都有本节前面列出的基本构建块。

一种语言可能有自己的解析器，它将语句标记化并提取有意义的信息，如标识符及其数据类型；函数名称、其声明、定义和调用；循环条件等。这些有意义的信息可以存储在数据结构中，以便可以轻松检索代码的流程。**抽象语法树**（**AST**）是源代码的流行树形表示。AST 可以用于进一步的转换和分析。

语言解析器可以用各种方式编写，使用各种工具如 `lex`、`yacc` 等，甚至可以手动编写。编写一个高效的解析器本身就是一门艺术。但本章我们并不打算涵盖这一点。我们更希望关注 LLVM IR 以及如何使用 LLVM 库将解析后的高级语言转换为 LLVM IR。

本章将介绍如何构建基本的工作 LLVM 示例代码，包括以下内容：

+   创建一个 LLVM 模块

+   在模块中发射一个函数

+   向函数中添加一个块

+   发射全局变量

+   发射返回语句

+   发射函数参数

+   在基本块中发射一个简单的算术语句

+   发射 if-else 条件 IR

+   发射循环的 LLVM IR

# 创建一个 LLVM 模块

在上一章中，我们了解到了 LLVM IR 的外观。在 LLVM 中，一个模块代表了一个要一起处理的单个代码单元。LLVM 模块类是所有其他 LLVM IR 对象的最高级容器。LLVM 模块包含全局变量、函数、数据布局、主机三元组等。让我们创建一个简单的 LLVM 模块。

LLVM 提供了 `Module()` 构造函数用于创建模块。第一个参数是模块的名称。第二个参数是 `LLVMContext`。让我们在主函数中获取这些参数并创建一个模块，如下所示：

```cpp
static LLVMContext &Context = getGlobalContext();
static Module *ModuleOb = new Module("my compiler", Context);
```

为了使这些函数正常工作，我们需要包含某些头文件：

```cpp
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
using namespace llvm;
static LLVMContext &Context = getGlobalContext();
static Module *ModuleOb = new Module("my compiler", Context);

int main(int argc, char *argv[]) {
  ModuleOb->dump();
  return 0;
}
```

将此代码放入一个文件中，比如 `toy.cpp`，然后编译它：

```cpp
$ clang++ -O3 toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -o toy
$ ./toy

```

输出将如下所示：

```cpp
; ModuleID = 'my compiler'

```

# 在模块中发射一个函数

现在我们已经创建了一个模块，下一步是输出一个函数。LLVM 有一个 `IRBuilder` 类，用于生成 LLVM IR 并使用模块对象的 `dump` 函数打印它。LLVM 提供了 `llvm::Function` 类来创建函数和 `llvm::FunctionType()` 来为函数关联返回类型。让我们假设我们的 `foo()` 函数返回整数类型。

```cpp
Function *createFunc(IRBuilder<> &Builder, std::string Name) {
  FunctionType *funcType = llvm::FunctionType::get(Builder.getInt32Ty(), false);
  Function *fooFunc = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, Name, ModuleOb);
  return fooFunc;
}
```

最后，在 `fooFunc` 上调用函数 `verifyFunction()`。此函数对生成的代码执行各种一致性检查，以确定我们的编译器是否一切正常。

```cpp
int main(int argc, char *argv[]) {
  static IRBuilder<> Builder(Context);
  Function *fooFunc = createFunc(Builder, "foo");
  verifyFunction(*fooFunc);
  ModuleOb->dump();
  return 0;
}
```

在包含部分添加 `IR/IRBuilder.h`、`IR/DerivedTypes.h` 和 `IR/Verifier.h` 文件。

整体代码如下：

```cpp
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include <vector>
using namespace llvm;

static LLVMContext &Context = getGlobalContext();
static Module *ModuleOb = new Module("my compiler", Context);

Function *createFunc(IRBuilder<> &Builder, std::string Name) {
 FunctionType *funcType = llvm::FunctionType::get(Builder.getInt32Ty(), false);
 Function *fooFunc = llvm::Function::Create(
 funcType, llvm::Function::ExternalLinkage, Name, ModuleOb);
 return fooFunc;
}

int main(int argc, char *argv[]) {
  static IRBuilder<> Builder(Context);
  Function *fooFunc = createFunc(Builder, "foo");
  verifyFunction(*fooFunc);
  ModuleOb->dump();
  return 0;
}
```

使用之前所述的相同选项编译 `toy.cpp`：

```cpp
$ clang++ -O3 toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -o toy

```

输出将如下所示：

```cpp
$ ./toy
; ModuleID = 'my compiler'

declare i32 @foo()

```

# 向函数添加一个块

函数由基本块组成。基本块有一个入口点。基本块由一系列 IR 指令组成，最后一条指令是终止指令。它有一个单一的出口点。LLVM 提供了 `BasicBlock` 类来创建和处理基本块。基本块可能以标签作为入口点，这表示在哪里插入后续指令。我们可以使用 `IRBuilder` 对象来保存这些新的基本块 IR。

```cpp
BasicBlock *createBB(Function *fooFunc, std::string Name) {
  return BasicBlock::Create(Context, Name, fooFunc);
}
```

整体代码如下：

```cpp
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include <vector>
using namespace llvm;

static LLVMContext &Context = getGlobalContext();
static Module *ModuleOb = new Module("my compiler", Context);

Function *createFunc(IRBuilder<> &Builder, std::string Name) {
  FunctionType *funcType = llvm::FunctionType::get(Builder.getInt32Ty(), false);
  Function *fooFunc = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, Name, ModuleOb);
  return fooFunc;
}

BasicBlock *createBB(Function *fooFunc, std::string Name) {
 return BasicBlock::Create(Context, Name, fooFunc);
}

int main(int argc, char *argv[]) {
  static IRBuilder<> Builder(Context);
  Function *fooFunc = createFunc(Builder, "foo");
  BasicBlock *entry = createBB(fooFunc, "entry");
  Builder.SetInsertPoint(entry);
  verifyFunction(*fooFunc);
  ModuleOb->dump();
  return 0;
}
```

编译 `toy.cpp` 文件：

```cpp
$ clang++ -O3 toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -o toy

```

输出将如下所示：

```cpp
; ModuleID = 'my compiler'

define i32 @foo() {
entry:
}
```

# 输出全局变量

全局变量的可见性是给定模块内所有函数的。LLVM 提供了 `GlobalVariable` 类来创建全局变量并设置其属性，如链接类型、对齐等。`Module` 类有 `getOrInsertGlobal()` 方法来创建全局变量。它接受两个参数——第一个是变量的名称，第二个是变量的数据类型。

由于全局变量是模块的一部分，我们在创建模块后创建全局变量。在 `toy.cpp` 中创建模块后立即插入以下代码：

```cpp
GlobalVariable *createGlob(IRBuilder<> &Builder, std::string Name) {
  ModuleOb->getOrInsertGlobal(Name, Builder.getInt32Ty());
  GlobalVariable *gVar = ModuleOb->getNamedGlobal(Name);
  gVar->setLinkage(GlobalValue::CommonLinkage);
  gVar->setAlignment(4);
  return gVar;
}
```

**链接** 决定了相同对象的多个声明是否引用同一个对象，还是不同的对象。LLVM 参考手册引用了以下类型的链接：

| `ExternalLinkage` | 外部可见函数。 |
| --- | --- |
| `AvailableExternallyLinkage` | 可供检查，但不进行输出。 |
| `LinkOnceAnyLinkage` | 链接时（内联）保留函数的一个副本 |
| `LinkOnceODRLinkage` | 相同，但仅替换为等效项。 |
| `WeakAnyLinkage` | 链接时（弱）保留命名函数的一个副本 |
| `WeakODRLinkage` | 相同，但仅替换为等效项。 |
| `AppendingLinkage` | 特殊用途，仅适用于全局数组。 |
| `InternalLinkage` | 链接时重命名冲突（静态函数）。 |
| `PrivateLinkage` | 类似于内部，但省略符号表。 |
| `ExternalWeakLinkage` | `ExternalWeak` 链接描述。 |
| `CommonLinkage` | 暂定定义 |

对齐提供了关于地址对齐的信息。对齐必须是 `2` 的幂。如果没有明确指定，则由目标设置。最大对齐为 `1 << 29`。

整体代码如下：

```cpp
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include <vector>
using namespace llvm;

static LLVMContext &Context = getGlobalContext();
static Module *ModuleOb = new Module("my compiler", Context);

Function *createFunc(IRBuilder<> &Builder, std::string Name) {
  FunctionType *funcType = llvm::FunctionType::get(Builder.getInt32Ty(), false);
  Function *fooFunc = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, Name, ModuleOb);
  return fooFunc;
}

BasicBlock *createBB(Function *fooFunc, std::string Name) {
  return BasicBlock::Create(Context, Name, fooFunc);
}

GlobalVariable *createGlob(IRBuilder<> &Builder, std::string Name) {
 ModuleOb->getOrInsertGlobal(Name, Builder.getInt32Ty());
 GlobalVariable *gVar = ModuleOb->getNamedGlobal(Name);
 gVar->setLinkage(GlobalValue::CommonLinkage);
 gVar->setAlignment(4);
 return gVar;
}

int main(int argc, char *argv[]) {
  static IRBuilder<> Builder(Context);
  GlobalVariable *gVar = createGlob(Builder, "x");
  Function *fooFunc = createFunc(Builder, "foo");
  BasicBlock *entry = createBB(fooFunc, "entry");
  Builder.SetInsertPoint(entry);
  verifyFunction(*fooFunc);
  ModuleOb->dump();
  return 0;
}
```

编译 `toy.cpp`：

```cpp
$ clang++ -O3 toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -o toy

```

输出将如下所示：

```cpp
; ModuleID = 'my compiler'

@x = common global i32, align 4

define i32 @foo() {
entry:
}
```

# 发射返回语句

函数可能返回一个值，也可能返回 void。在我们的例子中，我们定义了我们的函数返回一个整数。让我们假设我们的函数返回 `0`。第一步是获取一个 `0` 值，这可以通过使用 `Constant` 类来完成。

```cpp
Builder.CreateRet(Builder.getInt32(0));
```

整体代码如下：

```cpp
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include <vector>
using namespace llvm;

static LLVMContext &Context = getGlobalContext();
static Module *ModuleOb = new Module("my compiler", Context);

Function *createFunc(IRBuilder<> &Builder, std::string Name) {
  FunctionType *funcType = llvm::FunctionType::get(Builder.getInt32Ty(), false);
  Function *fooFunc = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, Name, ModuleOb);
  return fooFunc;
}

BasicBlock *createBB(Function *fooFunc, std::string Name) {
  return BasicBlock::Create(Context, Name, fooFunc);
}

GlobalVariable *createGlob(IRBuilder<> &Builder, std::string Name) {
  ModuleOb->getOrInsertGlobal(Name, Builder.getInt32Ty());
  GlobalVariable *gVar = ModuleOb->getNamedGlobal(Name);
  gVar->setLinkage(GlobalValue::CommonLinkage);
  gVar->setAlignment(4);
  return gVar;
}

int main(int argc, char *argv[]) {
  static IRBuilder<> Builder(Context);
  GlobalVariable *gVar = createGlob(Builder, "x");
  Function *fooFunc = createFunc(Builder, "foo");
  BasicBlock *entry = createBB(fooFunc, "entry");
  Builder.SetInsertPoint(entry);
 Builder.CreateRet(Builder.getInt32(0));
  verifyFunction(*fooFunc);
  ModuleOb->dump();
  return 0;
}
```

编译 `toy.cpp` 文件

```cpp
$ clang++ -O3 toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -o toy

```

输出将如下所示：

```cpp
; ModuleID = 'my compiler'

@x = common global i32, align 4

define i32 @foo() {
entry:
  ret i32 0
}
```

# 发射函数参数

函数接受具有其自身数据类型的参数。为了简化，假设我们的函数所有参数都是 i32 类型（32 位整数）。

例如，我们将考虑将两个参数 a 和 b 传递给函数。我们将这两个参数存储在一个向量中：

```cpp
 static std::vector <std::string> FunArgs;
 FunArgs.push_back("a");
 FunArgs.push_back("b");

```

下一步是指定函数将有两个参数。这可以通过将整数参数传递给 `functiontype` 来完成。

```cpp
Function *createFunc(IRBuilder<> &Builder, std::string Name) {
 std::vector<Type *> Integers(FunArgs.size(), Type::getInt32Ty(Context));
  FunctionType *funcType =
      llvm::FunctionType::get(Builder.getInt32Ty(), Integers, false);
  Function *fooFunc = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, Name, ModuleOb);
  return fooFunc;
}
```

最后一步是为函数参数设置名称。这可以通过在循环中使用 `Function` 参数迭代器来完成，如下所示：

```cpp
void setFuncArgs(Function *fooFunc, std::vector<std::string> FunArgs) {
 unsigned Idx = 0;
 Function::arg_iterator AI, AE;
 for (AI = fooFunc->arg_begin(), AE = fooFunc->arg_end(); AI != AE;
 ++AI, ++Idx)
 AI->setName(FunArgs[Idx]);
}

```

整体代码如下：

```cpp
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include <vector>
using namespace llvm;

static LLVMContext &Context = getGlobalContext();
static Module *ModuleOb = new Module("my compiler", Context);
static std::vector<std::string> FunArgs;

Function *createFunc(IRBuilder<> &Builder, std::string Name) {
 std::vector<Type *> Integers(FunArgs.size(), Type::getInt32Ty(Context));
  FunctionType *funcType =
      llvm::FunctionType::get(Builder.getInt32Ty(), Integers, false);
  Function *fooFunc = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, Name, ModuleOb);
  return fooFunc;
}

void setFuncArgs(Function *fooFunc, std::vector<std::string> FunArgs) {
 unsigned Idx = 0;
 Function::arg_iterator AI, AE;
 for (AI = fooFunc->arg_begin(), AE = fooFunc->arg_end(); AI != AE;
 ++AI, ++Idx)
 AI->setName(FunArgs[Idx]);
}

BasicBlock *createBB(Function *fooFunc, std::string Name) {
  return BasicBlock::Create(Context, Name, fooFunc);
}

GlobalVariable *createGlob(IRBuilder<> &Builder, std::string Name) {
  ModuleOb->getOrInsertGlobal(Name, Builder.getInt32Ty());
  GlobalVariable *gVar = ModuleOb->getNamedGlobal(Name);
  gVar->setLinkage(GlobalValue::CommonLinkage);
  gVar->setAlignment(4);
  return gVar;
}

int main(int argc, char *argv[]) {
 FunArgs.push_back("a");
 FunArgs.push_back("b");
  static IRBuilder<> Builder(Context);
  GlobalVariable *gVar = createGlob(Builder, "x");
  Function *fooFunc = createFunc(Builder, "foo");
 setFuncArgs(fooFunc, FunArgs);
  BasicBlock *entry = createBB(fooFunc, "entry");
  Builder.SetInsertPoint(entry);
  Builder.CreateRet(Builder.getInt32(0));
  verifyFunction(*fooFunc);
  ModuleOb->dump();
  return 0;
}
```

编译 `toy.cpp` 文件：

```cpp
$ clang++ -O3 toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -o toy

```

输出将如下所示：

```cpp
; ModuleID = 'my compiler'

@x = common global i32, align 4

define i32 @foo(i32 %a, i32 %b) {
entry:
 ret i32 0
}

```

# 在基本块中发射一个简单的算术语句

基本块由一系列指令组成。例如，一个指令可以是一个简单的语句，根据一些简单的算术指令执行任务。我们将看到如何使用 LLVM API 发射算术指令。

例如，如果我们想将第一个参数 a 与整数值 `16` 相乘，我们将使用以下 API 创建一个常量整数值 `16`：

```cpp
Value *constant = Builder.getInt32(16);

```

我们已经从函数参数列表中有了：

```cpp
Value *Arg1 = fooFunc->arg_begin();

```

LLVM 提供了一个丰富的 API 列表来创建二元运算。你可以通过查看 `include/llvm/IR/IRBuild.h` 文件来获取更多关于 API 的详细信息。

```cpp
Value *createArith(IRBuilder<> &Builder, Value *L, Value *R) {
  return Builder.CreateMul(L, R, "multmp");
}
```

### 备注

注意，出于演示目的，前面的函数返回乘法。我们留给读者去使这个函数更灵活，以返回任何二元运算。你可以在 `include/llvm/IR/IRBuild.h` 中探索更多二元运算。

整个代码现在看起来如下：

```cpp
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include <vector>
using namespace llvm;

static LLVMContext &Context = getGlobalContext();
static Module *ModuleOb = new Module("my compiler", Context);
static std::vector<std::string> FunArgs;

Function *createFunc(IRBuilder<> &Builder, std::string Name) {
  std::vector<Type *> Integers(FunArgs.size(), Type::getInt32Ty(Context));
  FunctionType *funcType =
      llvm::FunctionType::get(Builder.getInt32Ty(), Integers, false);
  Function *fooFunc = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, Name, ModuleOb);
  return fooFunc;
}

void setFuncArgs(Function *fooFunc, std::vector<std::string> FunArgs) {

  unsigned Idx = 0;
  Function::arg_iterator AI, AE;
  for (AI = fooFunc->arg_begin(), AE = fooFunc->arg_end(); AI != AE;
       ++AI, ++Idx)
    AI->setName(FunArgs[Idx]);
}

BasicBlock *createBB(Function *fooFunc, std::string Name) {
  return BasicBlock::Create(Context, Name, fooFunc);
}

GlobalVariable *createGlob(IRBuilder<> &Builder, std::string Name) {
  ModuleOb->getOrInsertGlobal(Name, Builder.getInt32Ty());
  GlobalVariable *gVar = ModuleOb->getNamedGlobal(Name);
  gVar->setLinkage(GlobalValue::CommonLinkage);
  gVar->setAlignment(4);
  return gVar;
}

Value *createArith(IRBuilder<> &Builder, Value *L, Value *R) {
 return Builder.CreateMul(L, R, "multmp");
}

int main(int argc, char *argv[]) {
  FunArgs.push_back("a");
  FunArgs.push_back("b");
  static IRBuilder<> Builder(Context);
  GlobalVariable *gVar = createGlob(Builder, "x");
  Function *fooFunc = createFunc(Builder, "foo");
  setFuncArgs(fooFunc, FunArgs);
  BasicBlock *entry = createBB(fooFunc, "entry");
  Builder.SetInsertPoint(entry);
 Value *Arg1 = fooFunc->arg_begin();
 Value *constant = Builder.getInt32(16);
 Value *val = createArith(Builder, Arg1, constant);
  Builder.CreateRet(val);
  verifyFunction(*fooFunc);
  ModuleOb->dump();
  return 0;
}
```

编译以下程序：

```cpp
$ clang++ -O3 toy.cpp `llvm-config --cxxflags --ldflags  --system-libs --libs core` -o toy

```

输出将如下所示：

```cpp
; ModuleID = 'my compiler'

@x = common global i32, align 4

define i32 @foo(i32 %a, i32 %b) {
entry:
  %multmp = mul i32 %a, 16
  ret i32 %multmp
}
```

你注意到返回值了吗？我们返回了乘法而不是常数 0。

# 发射 if-else 条件 IR

**if-else** 语句有一个条件表达式和两个代码路径来执行，取决于条件评估为真或假。条件表达式通常是一个比较语句。让我们在块的开始处发射一个条件语句。例如，让条件为 `a<100`。

```cpp
 Value *val2 = Builder.getInt32(100);
 Value *Compare = Builder.CreateICmpULT(val, val2, "cmptmp"); 
```

在编译时，我们得到以下输出：

```cpp
; ModuleID = 'my compiler'

@x = common global i32, align 4

define i32 @foo(i32 %a, i32 %b) {
entry:
  %multmp = mul i32 %a, 16
  %cmptmp = icmp ult i32 %multmp, 100

  ret i32 %multmp
}
```

下一步是定义`then`和`else`块表达式，这将根据条件表达式"`booltmp`"的结果执行。在这里，**PHI**指令的重要概念出现了。一个 phi 指令接受来自不同基本块的各种值，并根据条件表达式决定分配哪个值。

将创建两个单独的基本块"`ThenBB`"和"`ElseBB`"。假设`then`表达式是'将 a 加 1'，而`else`表达式是'将 a 加 2'。

第三个块将表示合并块，其中包含在`then`和`else`块合并时需要执行的指令。这些块需要推入`foo()`函数中。

为了提高复用性，我们创建如下所示的`BasicBlock`和`Value`容器：

```cpp
typedef SmallVector<BasicBlock *, 16> BBList;
typedef SmallVector<Value *, 16> ValList;
```

### 注意

注意，`SmallVector<>`是 LLVM 为了简化提供的向量容器包装器。

我们还将一些值推入`Value*`列表中，以便在 if-else 块中处理，如下所示：

```cpp
 Value *Condtn = Builder.CreateICmpNE(Compare, Builder.getInt32(0),
 "ifcond");
 ValList VL;
 VL.push_back(Condtn);
 VL.push_back(Arg1);
```

我们创建三个基本块并将它们推入容器中，如下所示：

```cpp
  BasicBlock *ThenBB = createBB(fooFunc, "then");
  BasicBlock *ElseBB = createBB(fooFunc, "else");
  BasicBlock *MergeBB = createBB(fooFunc, "ifcont");
  BBList List;
  List.push_back(ThenBB);
  List.push_back(ElseBB);
  List.push_back(MergeBB);
```

我们最终创建一个函数来生成 if-else 块：

```cpp
Value *createIfElse(IRBuilder<> &Builder, BBList List, ValList VL) {
  Value *Condtn = VL[0];
  Value *Arg1 = VL[1];
  BasicBlock *ThenBB = List[0];
  BasicBlock *ElseBB = List[1];
  BasicBlock *MergeBB = List[2];
  Builder.CreateCondBr(Condtn, ThenBB, ElseBB);

  Builder.SetInsertPoint(ThenBB);
  Value *ThenVal = Builder.CreateAdd(Arg1, Builder.getInt32(1), "thenaddtmp");
  Builder.CreateBr(MergeBB);

  Builder.SetInsertPoint(ElseBB);
  Value *ElseVal = Builder.CreateAdd(Arg1, Builder.getInt32(2), "elseaddtmp");
  Builder.CreateBr(MergeBB);

  unsigned PhiBBSize = List.size() - 1;
  Builder.SetInsertPoint(MergeBB);
  PHINode *Phi = Builder.CreatePHI(Type::getInt32Ty(getGlobalContext()), PhiBBSize, "iftmp");
  Phi->addIncoming(ThenVal, ThenBB);
  Phi->addIncoming(ElseVal, ElseBB);

  return Phi;
}
```

整体代码：

```cpp
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include <vector>
using namespace llvm;

static LLVMContext &Context = getGlobalContext();
static Module *ModuleOb = new Module("my compiler", Context);
static std::vector<std::string> FunArgs;
typedef SmallVector<BasicBlock *, 16> BBList;
typedef SmallVector<Value *, 16> ValList;

Function *createFunc(IRBuilder<> &Builder, std::string Name) {
  std::vector<Type *> Integers(FunArgs.size(), Type::getInt32Ty(Context));
  FunctionType *funcType =
      llvm::FunctionType::get(Builder.getInt32Ty(), Integers, false);
  Function *fooFunc = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, Name, ModuleOb);
  return fooFunc;
}

void setFuncArgs(Function *fooFunc, std::vector<std::string> FunArgs) {

  unsigned Idx = 0;
  Function::arg_iterator AI, AE;
  for (AI = fooFunc->arg_begin(), AE = fooFunc->arg_end(); AI != AE;
       ++AI, ++Idx)
    AI->setName(FunArgs[Idx]);
}

BasicBlock *createBB(Function *fooFunc, std::string Name) {
  return BasicBlock::Create(Context, Name, fooFunc);
}

GlobalVariable *createGlob(IRBuilder<> &Builder, std::string Name) {
  ModuleOb->getOrInsertGlobal(Name, Builder.getInt32Ty());
  GlobalVariable *gVar = ModuleOb->getNamedGlobal(Name);
  gVar->setLinkage(GlobalValue::CommonLinkage);
  gVar->setAlignment(4);
  return gVar;
}

Value *createArith(IRBuilder<> &Builder, Value *L, Value *R) {
  return Builder.CreateMul(L, R, "multmp");
}

Value *createIfElse(IRBuilder<> &Builder, BBList List, ValList VL) {
 Value *Condtn = VL[0];
 Value *Arg1 = VL[1];
 BasicBlock *ThenBB = List[0];
 BasicBlock *ElseBB = List[1];
 BasicBlock *MergeBB = List[2];
 Builder.CreateCondBr(Condtn, ThenBB, ElseBB);

 Builder.SetInsertPoint(ThenBB);
 Value *ThenVal = Builder.CreateAdd(Arg1, Builder.getInt32(1), "thenaddtmp");
 Builder.CreateBr(MergeBB);

 Builder.SetInsertPoint(ElseBB);
 Value *ElseVal = Builder.CreateAdd(Arg1, Builder.getInt32(2), "elseaddtmp");
 Builder.CreateBr(MergeBB);

 unsigned PhiBBSize = List.size() - 1;
 Builder.SetInsertPoint(MergeBB);
 PHINode *Phi = Builder.CreatePHI(Type::getInt32Ty(getGlobalContext()), PhiBBSize, "iftmp");
 PhiBBSize, "iftmp");
 Phi->addIncoming(ThenVal, ThenBB);
 Phi->addIncoming(ElseVal, ElseBB);

 return Phi;
}

int main(int argc, char *argv[]) {
  FunArgs.push_back("a");
  FunArgs.push_back("b");
  static IRBuilder<> Builder(Context);
  GlobalVariable *gVar = createGlob(Builder, "x");
  Function *fooFunc = createFunc(Builder, "foo");
  setFuncArgs(fooFunc, FunArgs);
  BasicBlock *entry = createBB(fooFunc, "entry");
  Builder.SetInsertPoint(entry);
  Value *Arg1 = fooFunc->arg_begin();
  Value *constant = Builder.getInt32(16);
  Value *val = createArith(Builder, Arg1, constant);

 Value *val2 = Builder.getInt32(100);
 Value *Compare = Builder.CreateICmpULT(val, val2, "cmptmp");
 Value *Condtn = Builder.CreateICmpNE(Compare, Builder.getInt32(0), "ifcond");

 ValList VL;
 VL.push_back(Condtn);
 VL.push_back(Arg1);

 BasicBlock *ThenBB = createBB(fooFunc, "then");
 BasicBlock *ElseBB = createBB(fooFunc, "else");
 BasicBlock *MergeBB = createBB(fooFunc, "ifcont");
 BBList List;
 List.push_back(ThenBB);
 List.push_back(ElseBB);
 List.push_back(MergeBB);

 Value *v = createIfElse(Builder, List, VL);

  Builder.CreateRet(v);
  verifyFunction(*fooFunc);
  ModuleOb->dump();
  return 0;
}
```

编译后，输出如下所示：

```cpp
; ModuleID = 'my compiler'

@x = common global i32, align 4

define i32 @foo(i32 %a, i32 %b) {
entry:
  %multmp = mul i32 %a, 16
  %cmptmp = icmp ult i32 %multmp, 100
  %ifcond = icmp ne i1 %cmptmp, i32 0
  br i1 %ifcond, label %then, label %else

then:                                             ; preds = %entry
  %thenaddtmp = add i32 %a, 1
  br label %ifcont

else:                                             ; preds = %entry
  %elseaddtmp = add i32 %a, 2
  br label %ifcont

ifcont:                                           ; preds = %else, %then
  %iftmp = phi i32 [ %thenaddtmp, %then ], [ %elseaddtmp, %else ]
  ret i32 %iftmp
}
```

# 循环的 LLVM IR 生成

与 if-else 语句类似，循环也可以使用 LLVM API 的稍作修改来生成。例如，我们想要以下循环的 LLVM IR：

```cpp
for(i=1; i< b; i++)  {body}
```

循环有一个循环变量`i`，它有一个初始值，在每次迭代后更新。在先前的例子中，循环变量在每次迭代后通过一个步长值更新，该步长值为`1`。然后有一个循环结束条件。在先前的例子中，"`i=1`"是初始值，"`i<b`"是循环的结束条件，"`i++`"是每次循环迭代后循环变量"`i`"增加的步长值。

在编写创建循环的函数之前，需要将一些`Value`和`BasicBlock`推入一个列表中，如下所示：

```cpp
Function::arg_iterator AI = fooFunc->arg_begin();
  Value *Arg1 = AI++;
  Value *Arg2 = AI;
  Value *constant = Builder.getInt32(16);
  Value *val = createArith(Builder, Arg1, constant);
  ValList VL;
  VL.push_back(Arg1);

  BBList List;
  BasicBlock *LoopBB = createBB(fooFunc, "loop");
  BasicBlock *AfterBB = createBB(fooFunc, "afterloop");
  List.push_back(LoopBB);
  List.push_back(AfterBB);

  Value *StartVal = Builder.getInt32(1);
```

让我们创建一个用于生成循环的函数：

```cpp
PHINode *createLoop(IRBuilder<> &Builder, BBList List, ValList VL,
                    Value *StartVal, Value *EndVal) {
  BasicBlock *PreheaderBB = Builder.GetInsertBlock();
  Value *val = VL[0];
  BasicBlock *LoopBB = List[0];
  Builder.CreateBr(LoopBB);
  Builder.SetInsertPoint(LoopBB);
  PHINode *IndVar = Builder.CreatePHI(Type::getInt32Ty(Context), 2, "i");
  IndVar->addIncoming(StartVal, PreheaderBB);
  Builder.CreateAdd(val, Builder.getInt32(5), "addtmp");
  Value *StepVal = Builder.getInt32(1);
  Value *NextVal = Builder.CreateAdd(IndVar, StepVal, "nextval");
  Value *EndCond = Builder.CreateICmpULT(IndVar, EndVal, "endcond");
  EndCond = Builder.CreateICmpNE(EndCond, Builder.getInt32(0), "loopcond");
  BasicBlock *LoopEndBB = Builder.GetInsertBlock();
  BasicBlock *AfterBB = List[1];
  Builder.CreateCondBr(EndCond, LoopBB, AfterBB);
  Builder.SetInsertPoint(AfterBB);
  IndVar->addIncoming(NextVal, LoopEndBB);
  return IndVar;
}
```

考虑以下代码行：

```cpp
IndVar->addIncoming(StartVal, PreheaderBB);…
IndVar->addIncoming(NextVal, LoopEndBB);
```

`IndVar`是一个 PHI 节点，它从两个块中接收两个值——从预头块(`i=1`)的`startval`和从循环结束块(`Nextval`)。

整体代码如下：

```cpp
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include <vector>
using namespace llvm;

typedef SmallVector<BasicBlock *, 16> BBList;
typedef SmallVector<Value *, 16> ValList;

static LLVMContext &Context = getGlobalContext();
static Module *ModuleOb = new Module("my compiler", Context);
static std::vector<std::string> FunArgs;

Function *createFunc(IRBuilder<> &Builder, std::string Name) {
  std::vector<Type *> Integers(FunArgs.size(), Type::getInt32Ty(Context));
  FunctionType *funcType =
      llvm::FunctionType::get(Builder.getInt32Ty(), Integers, false);
  Function *fooFunc = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, Name, ModuleOb);
  return fooFunc;
}

void setFuncArgs(Function *fooFunc, std::vector<std::string> FunArgs) {

  unsigned Idx = 0;
  Function::arg_iterator AI, AE;
  for (AI = fooFunc->arg_begin(), AE = fooFunc->arg_end(); AI != AE;
       ++AI, ++Idx)
    AI->setName(FunArgs[Idx]);
}

BasicBlock *createBB(Function *fooFunc, std::string Name) {
  return BasicBlock::Create(Context, Name, fooFunc);
}

GlobalVariable *createGlob(IRBuilder<> &Builder, std::string Name) {
  ModuleOb->getOrInsertGlobal(Name, Builder.getInt32Ty());
  GlobalVariable *gVar = ModuleOb->getNamedGlobal(Name);
  gVar->setLinkage(GlobalValue::CommonLinkage);
  gVar->setAlignment(4);
  return gVar;
}

Value *createArith(IRBuilder<> &Builder, Value *L, Value *R) {
  return Builder.CreateMul(L, R, "multmp");
}

Value *createLoop(IRBuilder<> &Builder, BBList List, ValList VL,
 Value *StartVal, Value *EndVal) {
 BasicBlock *PreheaderBB = Builder.GetInsertBlock();
 Value *val = VL[0];
 BasicBlock *LoopBB = List[0];
 Builder.CreateBr(LoopBB);
 Builder.SetInsertPoint(LoopBB);
 PHINode *IndVar = Builder.CreatePHI(Type::getInt32Ty(Context), 2, "i");
 IndVar->addIncoming(StartVal, PreheaderBB);
 Value *Add = Builder.CreateAdd(val, Builder.getInt32(5), "addtmp");
 Value *StepVal = Builder.getInt32(1);
 Value *NextVal = Builder.CreateAdd(IndVar, StepVal, "nextval");
 Value *EndCond = Builder.CreateICmpULT(IndVar, EndVal, "endcond");
 EndCond = Builder.CreateICmpNE(EndCond, Builder.getInt32(0), "loopcond");
 BasicBlock *LoopEndBB = Builder.GetInsertBlock();
 BasicBlock *AfterBB = List[1];
 Builder.CreateCondBr(EndCond, LoopBB, AfterBB);
 Builder.SetInsertPoint(AfterBB);
 IndVar->addIncoming(NextVal, LoopEndBB);
 return Add;
}

int main(int argc, char *argv[]) {
  FunArgs.push_back("a");
  FunArgs.push_back("b");
  static IRBuilder<> Builder(Context);
  GlobalVariable *gVar = createGlob(Builder, "x");
  Function *fooFunc = createFunc(Builder, "foo");
  setFuncArgs(fooFunc, FunArgs);
  BasicBlock *entry = createBB(fooFunc, "entry");
  Builder.SetInsertPoint(entry);
 Function::arg_iterator AI = fooFunc->arg_begin();
 Value *Arg1 = AI++;
 Value *Arg2 = AI;
 Value *constant = Builder.getInt32(16);
 Value *val = createArith(Builder, Arg1, constant);
 ValList VL;
 VL.push_back(Arg1);

 BBList List;
 BasicBlock *LoopBB = createBB(fooFunc, "loop");
 BasicBlock *AfterBB = createBB(fooFunc, "afterloop");
 List.push_back(LoopBB);
 List.push_back(AfterBB);

 Value *StartVal = Builder.getInt32(1);
 Value *Res = createLoop(Builder, List, VL, StartVal, Arg2);

  Builder.CreateRet(Res);
  verifyFunction(*fooFunc);
  ModuleOb->dump();
  return 0;
}
```

编译程序后，我们得到以下输出：

```cpp
; ModuleID = 'my compiler'

@x = common global i32, align 4

define i32 @foo(i32 %a, i32 %b) {
entry:
  %multmp = mul i32 %a, 16
  br label %loop

loop:                                             ; preds = %loop, %entry
  %i = phi i32 [ 1, %entry ], [ %nextval, %loop ]
  %addtmp = add i32 %a, 5
  %nextval = add i32 %i, 1
  %endcond = icmp ult i32 %i, %b
  %loopcond = icmp ne i1 %endcond, i32 0
  br i1 %loopcond, label %loop, label %afterloop

afterloop:                                        ; preds = %loop
  ret i32 %addtmp
}
```

# 概述

在本章中，你学习了如何使用 LLVM 提供的丰富库创建简单的 LLVM IR。记住，LLVM IR 是一个中间表示。高级编程语言通过自定义解析器转换为 LLVM IR，该解析器将代码分解为原子元素，如变量、函数、函数返回类型、函数参数、if-else 条件、循环、指针、数组等。这些原子元素可以存储到自定义数据结构中，然后可以使用这些数据结构来生成 LLVM IR，正如本章所演示的那样。

在解析器阶段，可以进行句法分析，而词法分析和类型检查可以在解析后、发射 IR 之前的中级阶段进行。

在实际应用中，几乎不会以本章所示的方式硬编码地发射红外线。相反，一种语言会被解析并表示为抽象语法树。然后，借助 LLVM 库，使用该树发射 LLVM IR，如前所述。LLVM 社区已经提供了一个优秀的教程，用于编写解析器并发射 LLVM IR。您可以访问[`llvm.org/docs/tutorial/`](http://llvm.org/docs/tutorial/)获取相同的信息。

在下一章中，我们将看到如何发射一些复杂的数据结构，如数组、指针。我们还将通过 Clang（C/C++的前端）的一些示例，了解语义分析是如何进行的。
