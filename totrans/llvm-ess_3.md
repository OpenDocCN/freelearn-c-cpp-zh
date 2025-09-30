# 第三章。高级 LLVM IR

LLVM 为高效的编译器转换和分析提供了一种强大的中间表示，同时提供了调试和可视化转换的自然方式。IR 的设计使其可以轻松映射到高级语言。LLVM IR 提供了类型信息，可用于各种优化。

在上一章中，你学习了如何在函数和模块中创建一些简单的 LLVM 指令。从发出二进制操作等简单示例开始，我们在模块中构建了函数，并创建了诸如 if-else 和循环等一些复杂的编程范式。LLVM 提供了一套丰富的指令和内嵌函数，用于发出复杂的 IR。

在本章中，我们将通过一些涉及内存操作的更多 LLVM IR 示例。本章还将涵盖一些高级主题，例如聚合数据类型及其操作。本章涵盖的主题如下：

+   获取元素的地址

+   从内存中读取

+   向内存位置写入

+   将标量插入到向量中

+   从向量中提取标量

# 内存访问操作

内存是几乎所有计算系统的重要组件。内存存储数据，这些数据需要被读取以在计算系统中执行操作。操作的结果将存储回内存中。

第一步是从内存中获取所需元素的地址，并将该特定元素可以找到的地址存储起来。你现在将学习如何计算地址并执行加载/存储操作。

# 获取元素的地址

在 LLVM 中，`getelementptr` 指令用于获取聚合数据结构中元素的地址。它只计算地址，并不访问内存。

`getelementptr` 指令的第一个参数是一个用作计算地址基础的类型。第二个参数是指针或指针的向量，它作为地址的基础 - 在我们的数组情况下将是 `a`。接下来的参数是要访问的元素的索引。

语言参考（[`llvm.org/docs/LangRef.html#getelementptr-instruction`](http://llvm.org/docs/LangRef.html#getelementptr-instruction)）中提到了关于 `getelementptr` 指令的重要注意事项如下：

> 第一个索引始终索引第一个参数给出的指针值，第二个索引索引指向的类型（不一定是直接指向的值，因为第一个索引可能不为零），等等。第一个索引的类型必须是指针值，后续的类型可以是数组、向量和结构体。注意，后续索引的类型不能是指针，因为这需要在继续计算之前加载指针。

这本质上意味着两件重要的事情：

1.  每个指针都有一个索引，第一个索引始终是数组索引。如果它是一个结构体的指针，你必须使用索引 0 来表示（第一个这样的结构体），然后是元素的索引。

1.  第一个类型参数帮助 GEP 识别基结构及其元素的大小，从而轻松计算地址。结果类型（`%a1`）不一定相同。

更详细的解释请参阅 [`llvm.org/docs/GetElementPtr.html`](http://llvm.org/docs/GetElementPtr.html)

假设我们有一个指向两个 32 位整数向量 `<2 x i32>* %a` 的指针，并且我们想要访问向量中的第二个整数。地址将被计算如下

```cpp
%a1 = getelementptr i32, <2 x i32>* %a, i32 1 
```

要发出此指令，可以使用如下所示的 LLVM API：

首先创建一个数组类型，该类型将被作为参数传递给函数。

```cpp
Function *createFunc(IRBuilder<> &Builder, std::string Name) {
  Type *u32Ty = Type::getInt32Ty(Context);
  Type *vecTy = VectorType::get(u32Ty, 2);
  Type *ptrTy = vecTy->getPointerTo(0);
  FunctionType *funcType =
      FunctionType::get(Builder.getInt32Ty(), ptrTy, false);
  Function *fooFunc =
      Function::Create(funcType, Function::ExternalLinkage, Name, ModuleOb);
  return fooFunc;
}

Value *getGEP(IRBuilder<> &Builder, Value *Base, Value *Offset) {
  return Builder.CreateGEP(Builder.getInt32Ty(), Base, Offset, "a1");
}
```

整个代码看起来如下：

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
  Type *u32Ty = Type::getInt32Ty(Context);
  Type *vecTy = VectorType::get(u32Ty, 2);
  Type *ptrTy = vecTy->getPointerTo(0);
  FunctionType *funcType =
      FunctionType::get(Builder.getInt32Ty(), ptrTy, false);
  Function *fooFunc =
      Function::Create(funcType, Function::ExternalLinkage, Name, ModuleOb);
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

Value *getGEP(IRBuilder<> &Builder, Value *Base, Value *Offset) {
  return Builder.CreateGEP(Builder.getInt32Ty(), Base, Offset, "a1");
}

int main(int argc, char *argv[]) {
  FunArgs.push_back("a");
  static IRBuilder<> Builder(Context);
  Function *fooFunc = createFunc(Builder, "foo");
  setFuncArgs(fooFunc, FunArgs);
  Value *Base = fooFunc->arg_begin();
  BasicBlock *entry = createBB(fooFunc, "entry");
  Builder.SetInsertPoint(entry);
  Value *gep = getGEP(Builder, Base, Builder.getInt32(1));
  verifyFunction(*fooFunc);
  ModuleOb->dump();
  return 0;
}
```

编译代码：

```cpp
$ clang++ toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -fno-rtti -o toy
$ ./toy

```

输出：

```cpp
; ModuleID = 'my compiler'

define i32 @foo(<2 x i32>* %a) {
entry:
  %a1 = getelementptr i32, <2 x i32>* %a, i32 1
  ret i32 0
}
```

# 从内存读取

现在，因为我们有了地址，我们准备从该地址读取数据并将读取的值赋给一个变量。

在 LLVM 中，`load` 指令用于从内存位置读取。这个简单的指令或类似指令的组合可以映射到底层汇编中的某些复杂的内存读取指令。

一个 `load` 指令接受一个参数，即从该内存地址读取数据的内存地址。我们在上一节中通过 `getelementptr` 指令在 `a1` 中获得了地址。

`load` 指令看起来如下：

```cpp
%val = load i32, i32* a1
```

这意味着 `load` 将取由 `a1` 指向的数据并将其保存到 `%val` 中。

要发出此，我们可以在函数中使用 LLVM 提供的 API，如下所示：

```cpp
Value *getLoad(IRBuilder<> &Builder, Value *Address) {
  return Builder.CreateLoad(Address, "load");
}
```

让我们也返回加载的值：

```cpp
   builder.CreateRet(val);
```

整个代码如下：

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
  Type *u32Ty = Type::getInt32Ty(Context);
  Type *vecTy = VectorType::get(u32Ty, 2);
  Type *ptrTy = vecTy->getPointerTo(0);
  FunctionType *funcType =
      FunctionType::get(Builder.getInt32Ty(), ptrTy, false);
  Function *fooFunc =
      Function::Create(funcType, Function::ExternalLinkage, Name, ModuleOb);
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

Value *getGEP(IRBuilder<> &Builder, Value *Base, Value *Offset) {
  return Builder.CreateGEP(Builder.getInt32Ty(), Base, Offset, "a1");
}

Value *getLoad(IRBuilder<> &Builder, Value *Address) {
 return Builder.CreateLoad(Address, "load");
}

int main(int argc, char *argv[]) {
  FunArgs.push_back("a");
  static IRBuilder<> Builder(Context);
  Function *fooFunc = createFunc(Builder, "foo");
  setFuncArgs(fooFunc, FunArgs);
  Value *Base = fooFunc->arg_begin();
  BasicBlock *entry = createBB(fooFunc, "entry");
  Builder.SetInsertPoint(entry);
  Value *gep = getGEP(Builder, Base, Builder.getInt32(1));
 Value *load = getLoad(Builder, gep);
  Builder.CreateRet(load);
  verifyFunction(*fooFunc);
  ModuleOb->dump();
  return 0;
}
```

编译以下代码：

```cpp
$ clang++ toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -fno-rtti -o toy
$ ./toy

```

以下是输出：

```cpp
; ModuleID = 'my compiler'

define i32 @foo(<2 x i32>* %a) {
entry:
  %a1 = getelementptr i32, <2 x i32>* %a, i32 1
  %load = load i32, i32* %a1
  ret i32 %load
}
```

# 将数据写入内存位置

LLVM 使用 `store` 指令将数据写入内存位置。`store` 指令有两个参数：要存储的值和存储它的地址。`store` 指令没有返回值。假设我们想要将数据写入两个整数的向量中的第二个元素。`store` 指令看起来像 `store i32 3, i32* %a1`。要发出 `store` 指令，我们可以使用 LLVM 提供的以下 API：

```cpp
void getStore(IRBuilder<> &Builder, Value *Address, Value *V) {
  Builder.CreateStore(V, Address);
}
```

例如，我们将 `<2 x i32>` 向量的第二个元素乘以 `16` 并将其存储在相同的位置。

考虑以下代码：

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
  Type *u32Ty = Type::getInt32Ty(Context);
  Type *vecTy = VectorType::get(u32Ty, 2);
  Type *ptrTy = vecTy->getPointerTo(0);
  FunctionType *funcType =
      FunctionType::get(Builder.getInt32Ty(), ptrTy, false);
  Function *fooFunc =
      Function::Create(funcType, Function::ExternalLinkage, Name, ModuleOb);
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

Value *createArith(IRBuilder<> &Builder, Value *L, Value *R) {
  return Builder.CreateMul(L, R, "multmp");
}

Value *getGEP(IRBuilder<> &Builder, Value *Base, Value *Offset) {
  return Builder.CreateGEP(Builder.getInt32Ty(), Base, Offset, "a1");
}

Value *getLoad(IRBuilder<> &Builder, Value *Address) {
  return Builder.CreateLoad(Address, "load");
}

void getStore(IRBuilder<> &Builder, Value *Address, Value *V) {
 Builder.CreateStore(V, Address);
}

int main(int argc, char *argv[]) {
  FunArgs.push_back("a");
  static IRBuilder<> Builder(Context);
  Function *fooFunc = createFunc(Builder, "foo");
  setFuncArgs(fooFunc, FunArgs);
  Value *Base = fooFunc->arg_begin();
  BasicBlock *entry = createBB(fooFunc, "entry");
  Builder.SetInsertPoint(entry);
  Value *gep = getGEP(Builder, Base, Builder.getInt32(1));
  Value *load = getLoad(Builder, gep);
  Value *constant = Builder.getInt32(16);
  Value *val = createArith(Builder, load, constant);
  getStore(Builder, gep, val);
  Builder.CreateRet(val);
  verifyFunction(*fooFunc);
  ModuleOb->dump();
  return 0;
}
```

编译以下代码：

```cpp
$ clang++ toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -fno-rtti -o toy
$ ./toy

```

生成的输出将如下：

```cpp
; ModuleID = 'my compiler'

define i32 @foo(<2 x i32>* %a) {
entry:
  %a1 = getelementptr i32, <2 x i32>* %a, i32 1
  %load = load i32, i32* %a1
  %multmp = mul i32 %load, 16
  store i32 %multmp, i32* %a1
  ret i32 %multmp
}
```

# 将标量插入到向量中

LLVM 还提供了发出指令的 API，该指令可以将标量插入到向量类型中。请注意，这种向量与数组不同。向量类型是一个简单的派生类型，表示元素向量。当使用 **单指令多数据**（**SIMD**）并行操作多个原始数据时，使用向量类型。向量类型需要一个大小（元素数量）和一个基础原始数据类型。例如，我们有一个 `Vec` 向量，它包含四个 `i32` 类型的整数 `<4 x i32>`。现在，我们想在向量的 0、1、2 和 3 索引处插入值 10、20、30 和 40。

`insertelement` 指令接受三个参数。第一个参数是向量类型的值。第二个操作数是一个标量值，其类型必须等于第一个操作数的元素类型。第三个操作数是一个索引，指示要插入值的位位置。结果值是相同类型的向量。

`insertelement` 指令看起来如下：

```cpp
%vec0 = insertelement <4 x double> Vec, %val0, %idx
```

这可以通过以下要点进一步理解：

+   `Vec` 是向量类型 `< 4 x i32 >`

+   `val0` 是要插入的值

+   `idx` 是要在向量中插入值的索引

考虑以下代码：

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
  Type *u32Ty = Type::getInt32Ty(Context);
  Type *vecTy = VectorType::get(u32Ty, 4);
  FunctionType *funcType =
      FunctionType::get(Builder.getInt32Ty(), vecTy, false);
  Function *fooFunc =
      Function::Create(funcType, Function::ExternalLinkage, Name, ModuleOb);
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

Value *getInsertElement(IRBuilder<> &Builder, Value *Vec, Value *Val,
 Value *Index) {
 return Builder.CreateInsertElement(Vec, Val, Index);
}

int main(int argc, char *argv[]) {
  FunArgs.push_back("a");
  static IRBuilder<> Builder(Context);
  Function *fooFunc = createFunc(Builder, "foo");
  setFuncArgs(fooFunc, FunArgs);

  BasicBlock *entry = createBB(fooFunc, "entry");
  Builder.SetInsertPoint(entry);

  Value *Vec = fooFunc->arg_begin();
  for (unsigned int i = 0; i < 4; i++)
    Value *V = getInsertElement(Builder, Vec,     Builder.getInt32((i + 1) * 10), Builder.getInt32(i));

  Builder.CreateRet(Builder.getInt32(0));
  verifyFunction(*fooFunc);
  ModuleOb->dump();
  return 0;
}
```

编译以下代码：

```cpp
$ clang++ toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -fno-rtti -o toy
$ ./toy 

```

结果输出如下：

```cpp
; ModuleID = 'my compiler'

define i32 @foo(<4 x i32> %a) {
entry:
  %0 = insertelement <4 x i32> %a, i32 10, i32 0
  %1 = insertelement <4 x i32> %a, i32 20, i32 1
  %2 = insertelement <4 x i32> %a, i32 30, i32 2
  %3 = insertelement <4 x i32> %a, i32 40, i32 3
  ret i32 0
}
```

向量 `Vec` 将具有 `<10, 20, 30, 40>` 的值。

# 从向量中提取标量

可以从向量中提取单个标量元素。LLVM 提供了 `extractelement` 指令来完成同样的操作。`extractelement` 指令的第一个操作数是向量类型的值。第二个操作数是一个索引，指示从哪个位置提取元素。

`insertelement` 指令看起来如下：

```cpp
result = extractelement <4 x i32> %vec, i32 %idx
```

这可以通过以下要点进一步理解：

+   `vec` 是一个向量

+   `idx` 是要提取的数据所在的索引

+   `result` 是标量类型，这里为 `i32`

让我们举一个例子，我们想要将给定向量的所有元素相加并返回一个整数。

考虑以下代码：

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
  Type *u32Ty = Type::getInt32Ty(Context);
  Type *vecTy = VectorType::get(u32Ty, 4);
  FunctionType *funcType =
      FunctionType::get(Builder.getInt32Ty(), vecTy, false);
  Function *fooFunc =
      Function::Create(funcType, Function::ExternalLinkage, Name, ModuleOb);
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

Value *createArith(IRBuilder<> &Builder, Value *L, Value *R) {
  return Builder.CreateAdd(L, R, "add");
}

Value *getExtractElement(IRBuilder<> &Builder, Value *Vec, Value *Index) {
 return Builder.CreateExtractElement(Vec, Index);
}

int main(int argc, char *argv[]) {
  FunArgs.push_back("a");
  static IRBuilder<> Builder(Context);
  Function *fooFunc = createFunc(Builder, "foo");
  setFuncArgs(fooFunc, FunArgs);

  BasicBlock *entry = createBB(fooFunc, "entry");
  Builder.SetInsertPoint(entry);

  Value *Vec = fooFunc->arg_begin();
  SmallVector<Value *, 4> V;
  for (unsigned int i = 0; i < 4; i++)
    V[i] = getExtractElement(Builder, Vec, Builder.getInt32(i));

  Value *add1 = createArith(Builder, V[0], V[1]);
  Value *add2 = createArith(Builder, add1, V[2]);
  Value *add = createArith(Builder, add2, V[3]);

  Builder.CreateRet(add);
  verifyFunction(*fooFunc);
  ModuleOb->dump();
  return 0;
}
```

编译以下代码：

```cpp
$ clang++ toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -fno-rtti -o toy
$ ./toy 

```

输出：

```cpp
ModuleID = 'my compiler'

define i32 @foo(<4 x i32> %a) {
entry:
  %0 = extractelement <4 x i32> %a, i32 0
  %1 = extractelement <4 x i32> %a, i32 1
  %2 = extractelement <4 x i32> %a, i32 2
  %3 = extractelement <4 x i32> %a, i32 3
  %add = add i32 %0, %1
  %add1 = add i32 %add, %2
  %add2 = add i32 %add1, %3
  ret i32 %add2
}
```

# 摘要

内存操作对于大多数目标架构来说是一个重要的指令。一些架构具有复杂的指令来在内存中移动数据。一些甚至可以直接在内存操作数上执行二进制操作，而另一些则从内存中加载数据到寄存器，然后对其进行操作（CISC 对比 RISC）。许多加载/存储操作也由 LLVM 内置函数完成。例如，请参阅 [`llvm.org/docs/LangRef.html#masked-vector-load-and-store-intrinsics`](http://llvm.org/docs/LangRef.html#masked-vector-load-and-store-intrinsics)。

LLVM IR 为所有架构提供了一个共同的竞技场。它提供了在内存或聚合数据类型上执行数据操作的基本指令。在将 LLVM IR 降低到特定架构的过程中，架构可能会组合 IR 指令以生成它们特有的指令。在本章中，我们探讨了某些高级 IR 指令，并查看了一些示例。对于详细研究，请参考[`llvm.org/docs/LangRef.html`](http://llvm.org/docs/LangRef.html)，它提供了 LLVM IR 指令的权威资源。

在下一章中，你将学习如何优化 LLVM IR 以减少指令并生成干净的代码。
