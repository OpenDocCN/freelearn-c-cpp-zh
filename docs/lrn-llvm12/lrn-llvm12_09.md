# 第七章：高级 IR 生成

在前几章介绍的**中间表示**（**IR**）生成中，您已经可以实现编译器中所需的大部分功能。在本章中，我们将研究一些通常在实际编译器中出现的高级主题。例如，许多现代语言使用异常处理，我们将看看如何将其转换为**低级虚拟机**（**LLVM**）IR。

为了支持 LLVM 优化器在某些情况下生成更好的代码，我们向 IR 代码添加了额外的类型元数据，并附加调试元数据使编译器的用户能够利用源级调试工具。

在本章中，您将学习以下主题：

+   在*抛出和捕获异常*中，您将学习如何在编译器中实现异常处理。

+   在*为基于类型的别名分析生成元数据*中，您将向 LLVM IR 附加额外的元数据，这有助于 LLVM 更好地优化代码。

+   在*添加调试元数据*中，您将实现所需的支持类，以向生成的 IR 代码添加调试信息。

到本章结束时，您将了解有关异常处理和基于类型的别名分析和调试信息的元数据的知识。

# 技术要求

本章的代码文件可在[`github.com/PacktPublishing/Learn-LLVM-12/tree/master/Chapter07`](https://github.com/PacktPublishing/Learn-LLVM-12/tree/master/Chapter07)找到

您可以在[`bit.ly/3nllhED`](https://bit.ly/3nllhED)找到代码演示视频。

# 抛出和捕获异常

LLVM IR 中的异常处理与平台的支持密切相关。在这里，我们将看到使用`libunwind`进行最常见类型的异常处理。它的全部潜力由 C++使用，因此我们将首先看一个 C++的示例，在该示例中，`bar()`函数可以抛出`int`或`double`值，如下所示：

```cpp
int bar(int x) {
  if (x == 1) throw 1;
  if (x == 2) throw 42.0;
  return x;
}
```

`foo()`函数调用`bar()`，但只处理抛出的`int`值。它还声明它只抛出`int`值，如下所示：

```cpp
int foo(int x) throw(int) {
  int y = 0;
  try {
    y = bar(x);
  }
  catch (int e) {
    y = e;
  }
  return y;
}
```

抛出异常需要两次调用运行时库。首先，使用`__cxa_allocate_exception()`调用分配异常的内存。此函数将要分配的字节数作为参数。然后将异常有效负载（例如示例中的`int`或`double`值）复制到分配的内存中。然后使用`__cxa_throw()`调用引发异常。此函数需要三个参数：指向分配的异常的指针；有关有效负载的类型信息；以及指向析构函数的指针，如果异常有效负载有一个的话。`__cxa_throw()`函数启动堆栈展开过程并且永远不会返回。在 LLVM IR 中，这是针对`int`值完成的，如下所示：

```cpp
%eh = tail call i8* @__cxa_allocate_exception(i64 4)
%payload = bitcast i8* %eh to i32*
store i32 1, i32* %payload
tail call void @__cxa_throw(i8* %eh,
                   i8* bitcast (i8** @_ZTIi to i8*), i8* 
                   null)
unreachable
```

`_ZTIi`是描述`int`类型的类型信息。对于 double 类型，它将是`_ZTId`。对`__cxa_throw()`的调用被标记为尾调用，因为它是该函数中的最终调用，可能使当前堆栈帧得以重用。

到目前为止，还没有做任何特定于 LLVM 的工作。这在`foo()`函数中发生了变化，因为对`bar()`的调用可能会引发异常。如果是`int`类型的异常，则必须将控制流转移到`catch`子句的 IR 代码。为了实现这一点，必须使用`invoke`指令而不是`call`指令，如下面的代码片段所示：

```cpp
%y = invoke i32 @_Z3bari(i32 %x) to label %next
                                 unwind label %lpad
```

两个指令之间的区别在于`invoke`有两个关联的标签。第一个标签是如果被调用的函数正常结束，通常是使用`ret`指令。在前面的代码示例中，这个标签称为`%next`。如果发生异常，则执行将继续在所谓的*着陆垫*上，具有`%lpad`标签。

着陆坪是一个基本的块，必须以`landingpad`指令开始。`landingpad`指令为 LLVM 提供了有关处理的异常类型的信息。对于`foo()`函数，它提供了以下信息：

```cpp
lpad:
%exc = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (i8** @_ZTIi to i8*)
          filter [1 x i8*] [i8* bitcast (i8** @_ZTIi to 
              i8*)]
```

这里有三种可能的操作类型，如下所述：

+   `cleanup`：这表示存在用于清理当前状态的代码。通常，这用于调用局部对象的析构函数。如果存在此标记，则在堆栈展开期间始终调用着陆坪。

+   `catch`：这是一个类型-值对的列表，表示可以处理的异常类型。如果抛出的异常类型在此列表中找到，则调用着陆坪。对于`foo()`函数，该值是指向`int`类型的 C++运行时类型信息的指针，类似于`__cxa_throw()`函数的参数。

+   `filter`：这指定了一个异常类型数组。如果当前异常的异常类型在数组中找不到，则调用着陆坪。这用于实现`throw()`规范。对于`foo()`函数，该数组只有一个成员——`int`类型的类型信息。

`landingpad`指令的结果类型是一个`{ i8*, i32 }`结构。第一个元素是指向抛出的异常的指针，而第二个元素是类型选择器。让我们从结构中提取这两个元素，如下所示：

```cpp
%exc.ptr = extractvalue { i8*, i32 } %exc, 0
%exc.sel = extractvalue { i8*, i32 } %exc, 1
```

*类型选择器*是一个数字，它帮助我们识别*为什么调用着陆坪*的原因。如果当前异常类型与`landingpad`指令的`catch`部分中给定的异常类型之一匹配，则它具有正值。如果当前异常类型与`filter`部分中给定的任何值都不匹配，则该值为负值，如果应调用清理代码，则为`0`。

基本上，类型选择器是偏移量，指向从`landingpad`指令的`catch`和`filter`部分中给定的值构造的类型信息表。在优化期间，多个着陆坪可以合并为一个，这意味着在 IR 级别不知道此表的结构。要检索给定类型的类型选择器，我们需要调用`@llvm.eh.typeid.for`内部函数。我们需要这样做来检查类型选择器的值是否对应于`int`的类型信息，以便能够执行`catch (int e) {}`块中的代码，如下所示：

```cpp
%tid.int = tail call i32 @llvm.eh.typeid.for(
                             i8* bitcast (i8** @_ZTIi to 
                             i8*))
%tst.int = icmp eq i32 %exc.sel, %tid.int
br i1 % tst.int, label %catchint, label %filterorcleanup
```

异常处理由对`__cxa_begin_catch()`和`__cxa_end_catch()`的调用框定。`__cxa_begin_catch()`函数需要一个参数：当前异常。这是`landingpad`指令返回的值之一。它返回指向异常有效负载的指针——在我们的情况下是一个`int`值。`__cxa_end_catch()`函数标记异常处理的结束，并释放使用`__cxa_allocate_exception()`分配的内存。请注意，如果在`catch`块内抛出另一个异常，则运行时行为要复杂得多。处理异常的方式如下：

```cpp
catchint:
%payload = tail call i8* @__cxa_begin_catch(i8* %exc.ptr)
%payload.int = bitcast i8* %payload to i32*
%retval = load i32, i32* %payload.int
tail call void @__cxa_end_catch()
br label %return
```

如果当前异常的类型与`throws()`声明中的列表不匹配，则调用意外异常处理程序。首先，我们需要再次检查类型选择器，如下所示：

```cpp
filterorcleanup:
%tst.blzero = icmp slt i32 %exc.sel, 0
br i1 %tst.blzero, label %filter, label %cleanup
```

如果类型选择器的值小于`0`，则调用处理程序，如下所示：

```cpp
filter:
tail call void @__cxa_call_unexpected(i8* %exc.ptr) #4
unreachable
```

同样，不希望处理程序返回。

在这种情况下不需要清理工作，因此所有清理代码所做的就是恢复堆栈展开器的执行，如下所示：

```cpp
cleanup:
resume { i8*, i32 } %exc
```

还有一部分缺失：`libunwind`驱动堆栈展开，但它与单一语言无关。语言相关的处理在`personality`函数中完成。对于 Linux 上的 C++，`personality`函数称为`__gxx_personality_v0()`。根据平台或编译器的不同，这个名称可能会有所不同。每个需要参与堆栈展开的函数都附有一个`personality`函数。`personality`函数分析函数是否捕获异常，是否有不匹配的过滤列表，或者是否需要清理调用。它将这些信息返回给展开器，展开器会相应地进行操作。在 LLVM IR 中，`personality`函数的指针作为函数定义的一部分给出，如下面的代码片段所示：

```cpp
define i32 @_Z3fooi(i32) personality i8* bitcast
                     (i32 (...)* @__gxx_personality_v0 to 
                      i8*)
```

有了这些，异常处理功能就完成了。

要在编译器中为您的编程语言使用异常处理，最简单的策略是依附于现有的 C++运行时函数。这样做的优势是您的异常与 C++是可互操作的。缺点是您将一些 C++运行时绑定到您的语言运行时中，尤其是内存管理。如果您想避免这一点，那么您需要创建自己的`_cxa_`函数的等价物。但是，您仍然需要使用提供堆栈展开机制的`libunwind`。

1.  让我们看看如何创建这个 IR。我们在*第三章*中创建了`calc`表达式编译器，*编译器的结构*。现在我们将扩展表达式编译器的代码生成器，以便在执行除以`0`时引发和处理异常。生成的 IR 将检查除法的除数是否为`0`。如果为`true`，则会引发异常。我们还将在函数中添加一个着陆块，用于捕获异常，将`Divide by zero!`打印到控制台，并结束计算。在这种简单情况下，使用异常处理并不是真正必要的，但它允许我们集中精力在代码生成上。我们将所有代码添加到`CodeGenerator.cpp`文件中。我们首先添加所需的新字段和一些辅助方法。我们需要存储`__cxa_allocate_exception()`和`__cxa_throw()`函数的 LLVM 声明，包括函数类型和函数本身。需要一个`GlobalVariable`实例来保存类型信息。我们还需要引用包含着陆块的基本块和只包含`unreachable`指令的基本块，如下面的代码片段所示：

```cpp
  GlobalVariable *TypeInfo = nullptr;
  FunctionType *AllocEHFty = nullptr;
  Function *AllocEHFn = nullptr;
  FunctionType *ThrowEHFty = nullptr;
  Function *ThrowEHFn = nullptr;
  BasicBlock *LPadBB = nullptr;
  BasicBlock *UnreachableBB = nullptr;
```

1.  我们还添加了一个新的辅助函数来创建比较两个值的 IR。`createICmpEq()`函数以`Left`和`Right`值作为参数进行比较。它创建一个`compare`指令，测试值的相等性，并创建一个分支指令到两个基本块，用于相等和不相等的情况。两个基本块通过`TrueDest`和`FalseDest`参数的引用返回。新基本块的标签可以在`TrueLabel`和`FalseLabel`参数中给出。代码如下所示：

```cpp
  void createICmpEq(Value *Left, Value *Right,
                    BasicBlock *&TrueDest,
                    BasicBlock *&FalseDest,
                    const Twine &TrueLabel = "",
                    const Twine &FalseLabel = "") {
    Function *Fn =        Builder.GetInsertBlock()->getParent();
    TrueDest = BasicBlock::Create(M->getContext(),                                  TrueLabel, Fn);
    FalseDest = BasicBlock::Create(M->getContext(),                                   FalseLabel, Fn);
    Value *Cmp = Builder.CreateCmp(CmpInst::ICMP_EQ,                                   Left, Right);
    Builder.CreateCondBr(Cmp, TrueDest, FalseDest);
  }
```

1.  使用运行时的函数，我们需要创建几个函数声明。在 LLVM 中，必须构建给出签名的函数类型以及函数本身。我们使用`createFunc()`方法来创建这两个对象。函数需要引用`FunctionType`和`Function`指针，新声明函数的名称和结果类型。参数类型列表是可选的，并且用来指示可变参数列表的标志设置为`false`，表示参数列表中没有可变部分。代码可以在以下片段中看到：

```cpp
  void createFunc(FunctionType *&Fty, Function *&Fn,
                  const Twine &N, Type *Result,
                  ArrayRef<Type *> Params = None,
                  bool IsVarArgs = false) {
    Fty = FunctionType::get(Result, Params, IsVarArgs);
    Fn = Function::Create(
        Fty, GlobalValue::ExternalLinkage, N, M);
  }
```

准备工作完成后，我们继续生成 IR 来引发异常。

## 引发异常

为了生成引发异常的 IR 代码，我们添加了一个`addThrow()`方法。这个新方法需要初始化新字段，然后通过`__cxa_throw`函数生成引发异常的 IR。引发的异常的有效载荷是`int`类型，并且可以设置为任意值。以下是我们需要编写的代码：

1.  新的`addThrow()`方法首先检查`TypeInfo`字段是否已初始化。如果没有，则创建一个`i8*`类型和`_ZTIi`名称的全局外部常量。这代表描述 C++ `int`类型的 C++元数据。代码如下所示：

```cpp
  void addThrow(int PayloadVal) {
    if (!TypeInfo) {
      TypeInfo = new GlobalVariable(
          *M, Int8PtrTy,
          /*isConstant=*/true,
          GlobalValue::ExternalLinkage,
          /*Initializer=*/nullptr, "_ZTIi");
```

1.  初始化继续创建`__cxa_allocate_exception()`和`__cxa_throw`函数的 IR 声明，使用我们的`createFunc()`辅助方法，如下所示：

```cpp
      createFunc(AllocEHFty, AllocEHFn,
                 "__cxa_allocate_exception", 
                 Int8PtrTy,
                 {Int64Ty});
      createFunc(ThrowEHFty, ThrowEHFn, "__cxa_throw",
                 VoidTy,
                 {Int8PtrTy, Int8PtrTy, Int8PtrTy});
```

1.  使用异常处理的函数需要一个`personality`函数，它有助于堆栈展开。我们添加 IR 代码声明来自 C++库的`__gxx_personality_v0()` `personality`函数，并将其设置为当前函数的`personality`例程。当前函数没有存储为字段，但我们可以使用`Builder`实例查询当前基本块，该基本块将函数存储为`parent`字段，如下面的代码片段所示：

```cpp
      FunctionType *PersFty;
      Function *PersFn;
      createFunc(PersFty, PersFn,                 "__gxx_personality_v0", Int32Ty, None,                 true);
      Function *Fn =          Builder.GetInsertBlock()->getParent();
      Fn->setPersonalityFn(PersFn);
```

1.  接下来，我们创建并填充着陆块的基本块。首先，我们需要保存当前基本块的指针。然后，我们创建一个新的基本块，将其设置在构建器内部用作插入指令的基本块，并调用`addLandingPad()`方法。此方法生成处理异常的 IR 代码，并在下一节“捕获异常”中进行描述。以下代码填充了着陆块的基本块：

```cpp
      BasicBlock *SaveBB = Builder.GetInsertBlock();
      LPadBB = BasicBlock::Create(M->getContext(),                                  "lpad", Fn);
      Builder.SetInsertPoint(LPadBB);
      addLandingPad();
```

1.  初始化部分已经完成，创建了一个包含`unreachable`指令的基本块。然后，我们创建一个基本块，并将其设置为构建器的插入点。然后，我们向其中添加一个`unreachable`指令。最后，我们将构建器的插入点设置回保存的`SaveBB`实例，以便后续的 IR 添加到正确的基本块。代码如下所示：

```cpp
      UnreachableBB = BasicBlock::Create(
          M->getContext(), "unreachable", Fn);
      Builder.SetInsertPoint(UnreachableBB);
      Builder.CreateUnreachable();
      Builder.SetInsertPoint(SaveBB);
    }
```

1.  要引发异常，我们需要通过调用`__cxa_allocate_exception()`函数为异常和有效载荷分配内存。我们的有效载荷是 C++ `int`类型，通常大小为 4 字节。我们为大小创建一个常量无符号值，并调用该函数作为参数。函数类型和函数声明已经初始化，所以我们只需要创建一个`call`指令，如下所示：

```cpp
    Constant *PayloadSz =       ConstantInt::get(Int64Ty, 4, false);
    CallInst *EH = Builder.CreateCall(        AllocEHFty, AllocEHFn, {PayloadSz});
```

1.  接下来，我们将`PayloadVal`值存储到分配的内存中。为此，我们需要使用`ConstantInt::get()`函数创建一个 LLVM IR 常量。分配的内存指针是`i8*`类型，但要存储`i32`类型的值，我们需要创建一个`bitcast`指令来转换类型，如下所示：

```cpp
    Value *PayloadPtr =        Builder.CreateBitCast(EH, Int32PtrTy);
    Builder.CreateStore(        ConstantInt::get(Int32Ty, PayloadVal, true),
        PayloadPtr);
```

1.  最后，我们通过调用`__cxa_throw`函数引发异常。因为这个函数实际上引发的异常也在同一个函数中处理，所以我们需要使用`invoke`指令而不是`call`指令。与`call`指令不同，`invoke`指令结束一个基本块，因为它有两个后继基本块。在这里，它们是`UnreachableBB`和`LPadBB`基本块。如果函数没有引发异常，控制流将转移到`UnreachableBB`基本块。由于`__cxa_throw()`函数的设计，这永远不会发生。控制流将转移到`LPadBB`基本块以处理异常。这完成了`addThrow()`方法的实现，如下面的代码片段所示：

```cpp
    Builder.CreateInvoke(
        ThrowEHFty, ThrowEHFn, UnreachableBB, LPadBB,
        {EH, ConstantExpr::getBitCast(TypeInfo, 
         Int8PtrTy),
         ConstantPointerNull::get(Int8PtrTy)});
  }
```

接下来，我们添加生成处理异常的 IR 代码。

## 捕获异常

为了生成捕获异常的 IR 代码，我们添加了一个`addLandingPad()`方法。生成的 IR 从异常中提取类型信息。如果匹配 C++的`int`类型，那么异常将通过向控制台打印`Divide by zero!`并从函数中返回来处理。如果类型不匹配，我们简单地执行一个`resume`指令，将控制转回运行时。因为在调用层次结构中没有其他函数来处理这个异常，运行时将终止应用程序。这些是我们需要采取的步骤来生成捕获异常的 IR：

1.  在生成的 IR 中，我们需要从 C++运行时库中调用`__cxa_begin_catch()`和`_cxa_end_catch()`函数。为了打印错误消息，我们将从 C 运行时库生成一个调用`puts()`函数的调用，并且为了从异常中获取类型信息，我们必须生成一个调用`llvm.eh.typeid.for`指令。我们需要为所有这些都创建`FunctionType`和`Function`实例，并且利用我们的`createFunc()`方法来创建它们，如下所示：

```cpp
  void addLandingPad() {
    FunctionType *TypeIdFty; Function *TypeIdFn;
    createFunc(TypeIdFty, TypeIdFn,
               "llvm.eh.typeid.for", Int32Ty,
               {Int8PtrTy});
    FunctionType *BeginCatchFty; Function 
        *BeginCatchFn;
    createFunc(BeginCatchFty, BeginCatchFn,
               "__cxa_begin_catch", Int8PtrTy,
               {Int8PtrTy});
    FunctionType *EndCatchFty; Function *EndCatchFn;
    createFunc(EndCatchFty, EndCatchFn,
               "__cxa_end_catch", VoidTy);
    FunctionType *PutsFty; Function *PutsFn;
    createFunc(PutsFty, PutsFn, "puts", Int32Ty,
               {Int8PtrTy});
```

1.  `landingpad`指令是我们生成的第一条指令。结果类型是一个包含`i8*`和`i32`类型字段的结构。通过调用`StructType::get()`函数生成这个结构。我们处理 C++ `int`类型的异常，必须将其作为`landingpad`指令的一个子句添加。子句必须是`i8*`类型的常量，因此我们需要生成一个`bitcast`指令将`TypeInfo`值转换为这种类型。我们将指令返回的值存储在`Exc`变量中，以备后用，如下所示：

```cpp
    LandingPadInst *Exc = Builder.CreateLandingPad(
        StructType::get(Int8PtrTy, Int32Ty), 1, "exc");
    Exc->addClause(ConstantExpr::getBitCast(TypeInfo, 
                   Int8PtrTy));
```

1.  接下来，我们从返回值中提取类型选择器。通过调用`llvm.eh.typeid.for`内部函数，我们检索`TypeInfo`字段的类型 ID，表示 C++的`int`类型。有了这个 IR，我们现在已经生成了我们需要比较的两个值，以决定是否可以处理异常，如下面的代码片段所示：

```cpp
    Value *Sel = Builder.CreateExtractValue(Exc, {1},                  "exc.sel");
    CallInst *Id =
        Builder.CreateCall(TypeIdFty, TypeIdFn,
                           {ConstantExpr::getBitCast(
                               TypeInfo, Int8PtrTy)});
```

1.  为了生成比较的 IR，我们调用我们的`createICmpEq()`函数。这个函数还生成了两个基本块，我们将它们存储在`TrueDest`和`FalseDest`变量中，如下面的代码片段所示：

```cpp
    BasicBlock *TrueDest, *FalseDest;
    createICmpEq(Sel, Id, TrueDest, FalseDest, 
                 "match",
                 "resume");
```

1.  如果两个值不匹配，控制流将在`FalseDest`基本块继续。这个基本块只包含一个`resume`指令，将控制返回给 C++运行时。下面的代码片段中有示例：

```cpp
    Builder.SetInsertPoint(FalseDest);
    Builder.CreateResume(Exc);
```

1.  如果两个值相等，控制流将在`TrueDest`基本块继续。我们首先生成 IR 代码，从`landingpad`指令的返回值中提取指向异常的指针，存储在`Exc`变量中。然后，我们生成一个调用`__cxa_begin_catch()`函数的调用，将指向异常的指针作为参数传递。这表示异常开始被运行时处理，如下面的代码片段所示：

```cpp
    Builder.SetInsertPoint(TrueDest);
    Value *Ptr =
        Builder.CreateExtractValue(Exc, {0}, 
            "exc.ptr");
    Builder.CreateCall(BeginCatchFty, BeginCatchFn,
                       {Ptr});
```

1.  我们通过调用`puts()`函数来处理异常，向控制台打印一条消息。为此，我们首先通过调用`CreateGlobalStringPtr()`函数生成一个指向字符串的指针，然后将这个指针作为参数传递给生成的`puts()`函数调用，如下所示：

```cpp
    Value *MsgPtr = Builder.CreateGlobalStringPtr(
        "Divide by zero!", "msg", 0, M);
    Builder.CreateCall(PutsFty, PutsFn, {MsgPtr});
```

1.  这完成了异常处理，并生成了一个调用`__cxa_end_catch()`函数通知运行时的过程。最后，我们使用`ret`指令从函数中返回，如下所示：

```cpp
    Builder.CreateCall(EndCatchFty, EndCatchFn);
    Builder.CreateRet(Int32Zero);
  }
```

通过`addThrow()`和`addLandingPad()`函数，我们可以生成 IR 来引发异常和处理异常。我们仍然需要添加 IR 来检查除数是否为`0`，这是下一节的主题。

## 将异常处理代码集成到应用程序中

除法的 IR 是在`visit(BinaryOp&)`方法中生成的。我们首先生成 IR 来比较除数和`0`，而不仅仅是生成一个`sdiv`指令。如果除数是`0`，那么控制流将继续在一个基本块中引发异常。否则，控制流将在一个包含`sdiv`指令的基本块中继续。借助`createICmpEq()`和`addThrow()`函数，我们可以很容易地编写这个代码。

```cpp
    case BinaryOp::Div:
      BasicBlock *TrueDest, *FalseDest;
      createICmpEq(Right, Int32Zero, TrueDest,
                   FalseDest, "divbyzero", "notzero");
      Builder.SetInsertPoint(TrueDest);
      addThrow(42); // Arbitrary payload value.
      Builder.SetInsertPoint(FalseDest);
      V = Builder.CreateSDiv(Left, Right);
      break;
```

代码生成部分现在已经完成。要构建应用程序，您需要切换到`build`目录并运行`ninja`工具。

```cpp
$ ninja
```

构建完成后，您可以检查生成的 IR，例如使用`with a: 3/a`表达式。

```cpp
$ src/calc "with a: 3/a"
```

您将看到引发和捕获异常所需的额外 IR。

生成的 IR 现在依赖于 C++运行时。链接所需库的最简单方法是使用 clang++编译器。将用于表达式计算器的运行时函数的`rtcalc.c`文件重命名为`rtcalc.cpp`，并在文件中的每个函数前面添加`extern "C"`。然后我们可以使用`llc`工具将生成的 IR 转换为目标文件，并使用 clang++编译器创建可执行文件。

```cpp
$ src/calc "with a: 3/a" | llc -filetype obj -o exp.o
$ clang++ -o exp exp.o ../rtcalc.cpp
```

然后，我们可以使用不同的值运行生成的应用程序，如下所示：

```cpp
$ ./exp
Enter a value for a: 1
The result is: 3
$ ./exp
Enter a value for a: 0
Divide by zero!
```

在第二次运行中，输入为`0`，这引发了一个异常。这符合预期！

我们已经学会了如何引发和捕获异常。生成 IR 的代码可以用作其他编译器的蓝图。当然，所使用的类型信息和`catch`子句的数量取决于编译器的输入，但我们需要生成的 IR 仍然遵循本节中提出的模式。

添加元数据是向 LLVM 提供更多信息的一种方式。在下一节中，我们将添加类型元数据以支持 LLVM 优化器在某些情况下的使用。

# 为基于类型的别名分析生成元数据

两个指针可能指向同一内存单元，然后它们彼此别名。在 LLVM 模型中，内存没有类型，这使得优化器难以确定两个指针是否彼此别名。如果编译器可以证明两个指针不会别名，那么就有可能进行更多的优化。在下一节中，我们将更仔细地研究这个问题，并探讨如何添加额外的元数据将有所帮助，然后再实施这种方法。

## 理解需要额外元数据的原因

为了演示问题，让我们看一下以下函数：

```cpp
void doSomething(int *p, float *q) {
  *p = 42;
  *q = 3.1425;
} 
```

优化器无法确定`p`和`q`指针是否指向同一内存单元。在优化过程中，这是一个重要的分析，称为`p`和`q`指向同一内存单元，那么它们是别名。如果优化器可以证明这两个指针永远不会别名，这将提供额外的优化机会。例如，在`soSomething()`函数中，存储可以重新排序而不改变结果。

这取决于源语言的定义，一个类型的变量是否可以是不同类型的另一个变量的别名。请注意，语言也可能包含打破基于类型的别名假设的表达式，例如不相关类型之间的类型转换。

LLVM 开发人员选择的解决方案是向`load`和`store`指令添加元数据。元数据有两个目的，如下所述：

+   首先，它基于类型层次结构定义了类型层次结构，其中一个类型可能是另一个类型的别名

+   其次，它描述了`load`或`store`指令中的内存访问

让我们来看看 C 中的类型层次结构。每种类型层次结构都以根节点开头，可以是**命名**或**匿名**。LLVM 假设具有相同名称的根节点描述相同类型的层次结构。您可以在相同的 LLVM 模块中使用不同的类型层次结构，LLVM 会安全地假设这些类型可能会别名。在根节点下面，有标量类型的节点。聚合类型的节点不附加到根节点，但它们引用标量类型和其他聚合类型。Clang 为 C 定义了以下层次结构：

+   根节点称为`Simple C/C++ TBAA`。

+   在根节点下面是`char`类型的节点。这是 C 中的特殊类型，因为所有指针都可以转换为指向`char`的指针。

+   在`char`节点下面是其他标量类型的节点和一个名为`any pointer`的所有指针类型。

聚合类型被定义为一系列成员类型和偏移量。

这些元数据定义用于附加到`load`和`store`指令的访问标签。访问标签由三部分组成：基本类型、访问类型和偏移量。根据基本类型，访问标签描述内存访问的方式有两种可能，如下所述：

1.  如果基本类型是聚合类型，则访问标签描述了`struct`成员的内存访问，具有访问类型，并位于给定偏移量处。

1.  如果基本类型是标量类型，则访问类型必须与基本类型相同，偏移量必须为`0`。

有了这些定义，我们现在可以在访问标签上定义一个关系，用于评估两个指针是否可能别名。元组（基本类型，偏移量）的直接父节点由基本类型和偏移量确定，如下所示：

+   如果基本类型是标量类型且偏移量为 0，则直接父节点是（父类型，0），其中父类型是在类型层次结构中定义的父节点的类型。如果偏移量不为 0，则直接父节点未定义。

+   如果基本类型是聚合类型，则元组（基本类型，偏移量）的直接父节点是元组（新类型，新偏移量），其中新类型是在偏移量处的成员的类型。新偏移量是新类型的偏移量，调整为其新的起始位置。

这个关系的传递闭包是父关系。例如，（基本类型 1，访问类型 1，偏移 1）和（基本类型 2，访问类型 2，偏移 2）这两种内存访问类型可能会别名，如果（基本类型 1，偏移 1）和（基本类型 2，偏移 2）或者反之亦然在父关系中相关联。

让我们通过一个例子来说明：

```cpp
struct Point { float x, y; }
void func(struct Point *p, float *x, int *i, char *c) {
  p->x = 0; p->y = 0; *x = 0.0; *i = 0; *c = 0; 
}
```

使用前面对标量类型的内存访问标签定义，参数`i`的访问标签是（`int`，`int`，`0`），参数`c`的访问标签是（`char`，`char`，`0`）。在类型层次结构中，`int`类型的节点的父节点是`char`节点，因此（`int`，`0`）的直接父节点是（`char`，`0`），两个指针可能会别名。对于参数`x`和参数`c`也是如此。但是参数`x`和`i`没有关联，因此它们不会别名。`struct Point`的`y`成员的访问是（`Point`，`float`，`4`），4 是结构体中`y`成员的偏移量。因此（`Point`，`4`）的直接父节点是（`float`，`0`），因此`p->y`和`x`的访问可能会别名，并且根据相同的推理，也会与参数`c`别名。

要创建元数据，我们使用`llvm::MDBuilder`类，该类在`llvm/IR/MDBuilder.h`头文件中声明。数据本身存储在`llvm::MDNode`和`llvm::MDString`类的实例中。使用构建器类可以保护我们免受构造的内部细节的影响。

通过调用`createTBAARoot()`方法创建根节点，该方法需要类型层次结构的名称作为参数，并返回根节点。可以使用`createAnonymousTBAARoot()`方法创建匿名唯一根节点。

使用`createTBAAScalarTypeNode()`方法将标量类型添加到层次结构中，该方法以类型的名称和父节点作为参数。为聚合类型添加类型节点稍微复杂一些。`createTBAAStructTypeNode()`方法以类型的名称和字段列表作为参数。字段作为`std::pair<llvm::MDNode*, uint64_t>`实例给出。第一个元素表示成员的类型，第二个元素表示`struct`类型中的偏移量。

使用`createTBAAStructTagNode()`方法创建访问标签，该方法以基本类型、访问类型和偏移量作为参数。

最后，元数据必须附加到`load`或`store`指令上。`llvm::Instruction`类有一个`setMetadata()`方法，用于添加各种元数据。第一个参数必须是`llvm::LLVMContext::MD_tbaa`，第二个参数必须是访问标签。

掌握了这些知识，我们将在下一节为`tinylang`添加元数据。

## 为 tinylang 添加 TBAA 元数据

为了支持 TBAA，我们添加了一个新的`CGTBAA`类。这个类负责生成元数据节点。我们将它作为`CGModule`类的成员，称之为`TBAA`。每个`load`和`store`指令都可能被注释，我们也在`CGModule`类中放置了一个新的函数来实现这个目的。该函数尝试创建标签访问信息。如果成功，元数据将附加到指令上。这种设计还允许我们在不需要元数据的情况下关闭元数据生成，例如在关闭优化的构建中。代码如下所示：

```cpp
void CGModule::decorateInst(llvm::Instruction *Inst,
                            TypeDenoter *TyDe) {
  if (auto *N = TBAA.getAccessTagInfo(TyDe))
    Inst->setMetadata(llvm::LLVMContext::MD_tbaa, N);
}
```

我们将新的`CGTBAA`类的声明放入`include/tinylang/CodeGen/CGTBAA.h`头文件中，并将定义放入`lib/CodeGen/CGTBAA.cpp`文件中。除了**抽象语法树**（**AST**）定义之外，头文件还需要包括定义元数据节点和构建器的文件，如下面的代码片段所示：

```cpp
#include "tinylang/AST/AST.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
```

`CGTBAA`类需要存储一些数据成员。因此，让我们逐步看看如何做到这一点，如下所示：

1.  首先，我们需要缓存类型层次结构的根，如下所示：

```cpp
 class CGTBAA {
  llvm::MDNode *Root;
```

1.  为了构造元数据节点，我们需要`MDBuilder`类的一个实例，如下所示：

```cpp
  llvm::MDBuilder MDHelper;
```

1.  最后，我们将为类型生成的元数据存储起来以便重用，如下所示：

```cpp
  llvm::DenseMap<TypeDenoter *, llvm::MDNode *> 
    MetadataCache;
// …
};
```

在定义构造所需的变量之后，我们现在添加了创建元数据所需的方法，如下所示：

1.  构造函数初始化数据成员，如下所示：

```cpp
CGTBAA::CGTBAA(llvm::LLVMContext &Ctx)
      : MDHelper(llvm::MDBuilder(Ctx)), Root(nullptr) {}
```

1.  我们懒惰地实例化类型层次结构的根，我们称之为`Simple tinylang TBAA`，如下面的代码片段所示：

```cpp
llvm::MDNode *CGTBAA::getRoot() {
  if (!Root)
    Root = MDHelper.createTBAARoot("Simple tinylang                                    TBAA");
  return Root;
}
```

1.  对于标量类型，我们使用`MDBuilder`类根据类型的名称创建元数据节点。新的元数据节点存储在缓存中，如下面的代码片段所示：

```cpp
llvm::MDNode *
CGTBAA::createScalarTypeNode(TypeDeclaration *Ty,
                             StringRef Name,
                             llvm::MDNode *Parent) {
  llvm::MDNode *N =
      MDHelper.createTBAAScalarTypeNode(Name, Parent);
  return MetadataCache[Ty] = N;
}
```

1.  创建记录的元数据的方法更加复杂，因为我们必须枚举记录的所有字段。代码如下所示：

```cpp
llvm::MDNode *CGTBAA::createStructTypeNode(
    TypeDeclaration *Ty, StringRef Name,
    llvm::ArrayRef<std::pair<llvm::MDNode *, 
        uint64_t>>
        Fields) {
  llvm::MDNode *N =
      MDHelper.createTBAAStructTypeNode(Name, Fields);
  return MetadataCache[Ty] = N;
}
```

1.  为了返回`tinylang`类型的元数据，我们需要创建类型层次结构。由于`tinylang`的类型系统非常受限，我们可以使用简单的方法。每个标量类型都映射到附加到根节点的唯一类型，我们将所有指针映射到单个类型。结构化类型然后引用这些节点。如果我们无法映射类型，我们将返回`nullptr`，如下所示：

```cpp
llvm::MDNode *CGTBAA::getTypeInfo(TypeDeclaration *Ty) {
  if (llvm::MDNode *N = MetadataCache[Ty])
    return N;
  if (auto *Pervasive =
          llvm::dyn_cast<PervasiveTypeDeclaration>(Ty)) {
    StringRef Name = Pervasive->getName();
    return createScalarTypeNode(Pervasive, Name, 
        getRoot());
  }
  if (auto *Pointer =
          llvm::dyn_cast<PointerTypeDeclaration>(Ty)) {
    StringRef Name = "any pointer";
    return createScalarTypeNode(Pointer, Name, 
        getRoot());
  }
  if (auto *Record =
          llvm::dyn_cast<RecordTypeDeclaration>(Ty)) {
    llvm::SmallVector<std::pair<llvm::MDNode *, 
        uint64_t>,
                      4>
        Fields;
    auto *Rec =
        llvm::cast<llvm::StructType>(              CGM.convertType(Record));
    const llvm::StructLayout *Layout =
        CGM.getModule()->getDataLayout()
            .getStructLayout(Rec);
    unsigned Idx = 0;
    for (const auto &F : Record->getFields()) {
      uint64_t Offset = Layout->getElementOffset(Idx);
      Fields.emplace_back(getTypeInfo(F.getType()), 
          Offset);
      ++Idx;
    }
    StringRef Name = CGM.mangleName(Record);
    return createStructTypeNode(Record, Name, Fields);
  }
  return nullptr;
}
```

1.  获取元数据的通用方法是`getAccessTagInfo()`。因为我们只需要查找指针类型，所以我们进行了检查。否则，我们返回`nullptr`，如下面的代码片段所示：

```cpp
llvm::MDNode *CGTBAA::getAccessTagInfo(TypeDenoter *TyDe) 
{
  if (auto *Pointer = llvm::dyn_cast<PointerType>(TyDe)) 
  {
    return getTypeInfo(Pointer->getTyDen());
  }
  return nullptr;
}
```

为了启用 TBAA 元数据的生成，我们现在只需要将元数据附加到我们生成的`load`和`store`指令上。例如，在`CGProcedure::writeVariable()`中，对全局变量的存储，使用`store`指令，如下所示：

```cpp
      Builder.CreateStore(Val, CGM.getGlobal(D));
```

为了装饰指令，我们需要用以下行替换前一行：

```cpp
      auto *Inst = Builder.CreateStore(Val,
                                       CGM.getGlobal(Decl));
      CGM.decorateInst(Inst, V->getTypeDenoter());
```

有了这些变化，我们已经完成了 TBAA 元数据的生成。

在下一节中，我们将讨论一个非常相似的主题：调试元数据的生成。

# 添加调试元数据

为了允许源级调试，我们必须添加调试信息。LLVM 中的调试信息支持使用调试元数据来描述源语言的类型和其他静态信息，并使用内在函数来跟踪变量值。LLVM 核心库在 Unix 系统上生成 DWARF 格式的调试信息，在 Windows 上生成**蛋白质数据银行**（**PDB**）格式。我们将在下一节中看一下一般的结构。

## 理解调试元数据的一般结构

为了描述静态结构，LLVM 使用元数据类似于基于类型的分析的元数据。静态结构描述文件、编译单元、函数、词法块和使用的数据类型。

我们使用的主要类是`llvm::DIBuilder`，我们需要使用`llvm/IR/DIBuilder`包含文件来获取类声明。这个构建器类提供了一个易于使用的接口来创建调试元数据。稍后，元数据要么添加到 LLVM 对象，比如全局变量，要么在调试内部使用。构建器类可以创建的重要元数据在这里列出：

+   `lvm::DIFile`：使用文件名和包含文件的目录的绝对路径来描述文件。您可以使用`createFile()`方法来创建它。一个文件可以包含主编译单元，也可以包含导入的声明。

+   `llvm::DICompileUnit`：用于描述当前编译单元。除其他内容外，您需要指定源语言、特定于编译器的生产者字符串，是否启用优化，以及编译单元所在的`DIFile`。您可以通过调用`createCompileUnit()`来创建它。

+   `llvm::DISubprogram`：描述一个函数。重要信息是作用域（通常是`DICompileUnit`或嵌套函数的`DISubprogram`）、函数的名称、函数的重整名和函数类型。它是通过调用`createFunction()`来创建的。

+   `llvm::DILexicalBlock`：描述了许多高级语言中找到的块作用域的词法块。您可以通过调用`createLexicalBlock()`来创建它。

LLVM 不对编译器翻译的语言做任何假设。因此，它对语言的数据类型没有任何信息。为了支持源级调试，特别是在调试器中显示变量值，也必须添加类型信息。这里列出了重要的构造：

+   `createBasicType()`函数返回一个指向`llvm::DIBasicType`类的指针，用于创建描述`tinylang`中的`INTEGER`或 C++中的`int`等基本类型的元数据。除了类型的名称，所需的参数是位大小和编码，例如，它是有符号还是无符号类型。

+   有几种方法可以构造复合数据类型的元数据，由`llvm::DIComposite`类表示。您可以使用`createArrayType()`、`createStructType()`、`createUnionType()`和`createVectorType()`函数来实例化`array`、`struct`、`union`和`vector`数据类型的元数据。这些函数需要您期望的参数，例如，数组类型的基本类型和订阅数量，或者`struct`类型的字段成员列表。

+   还有支持枚举、模板、类等的方法。

函数列表显示您必须将源语言的每个细节添加到调试信息中。假设您的`llvm::DIBuilder`类的实例称为`DBuilder`。进一步假设您在名为`File.mod`的文件中有一些`tinylang`源码，位于`/home/llvmuser`文件夹中。文件中有一个在*第 5 行*包含在*第 7 行*包含一个`VAR i:INTEGER`本地声明的`Func():INTEGER`函数。让我们从文件的信息开始创建这些元数据。您需要指定文件名和文件所在文件夹的绝对路径，如下面的代码片段所示：

```cpp
llvm::DIFile *DbgFile = DBuilder.createFile("File.mod",
                                            "/home/llvmuser"); 
```

文件是`tinylang`中的一个模块，因此是 LLVM 的编译单元。这携带了大量信息，如下面的代码片段所示：

```cpp
bool IsOptimized = false;
llvm::StringRef CUFlags;
unsigned ObjCRunTimeVersion = 0;
llvm::StringRef SplitName;
llvm::DICompileUnit::DebugEmissionKind EmissionKind =
      llvm::DICompileUnit::DebugEmissionKind::FullDebug;
llvm::DICompileUnit *DbgCU = DBuilder.createCompileUnit(
      llvm::dwarf::DW_LANG_Modula2, DbgFile, „tinylang",
      IsOptimized, CUFlags, ObjCRunTimeVersion, SplitName,
      EmissionKind);
```

调试器需要知道源语言。DWARF 标准定义了一个包含所有常见值的枚举。一个缺点是您不能简单地添加一个新的源语言。要做到这一点，您必须通过 DWARF 委员会创建一个请求。请注意，调试器和其他调试工具也需要支持新语言，仅仅向枚举添加一个新成员是不够的。

在许多情况下，选择一个接近您源语言的语言就足够了。在`tinylang`的情况下，这是 Modula-2，我们使用`DW_LANG_Modula2`进行语言识别。编译单元位于一个文件中，由我们之前创建的`DbgFile`变量标识。调试信息可以携带有关生产者的信息。这可以是编译器的名称和版本信息。在这里，我们只传递一个`tinylang`字符串。如果您不想添加这些信息，那么您可以简单地将一个空字符串作为参数。

下一组信息包括一个`IsOptimized`标志，应指示编译器是否已经打开了优化。通常，此标志是从`-O`命令行开关派生的。您可以使用`CUFlags`参数向调试器传递附加的参数设置。这里没有使用，我们传递一个空字符串。我们不使用 Objective-C，所以我们将`0`作为 Objective-C 运行时版本传递。通常，调试信息嵌入在我们正在创建的目标文件中。如果我们想要将调试信息写入一个单独的文件中，那么`SplitName`参数必须包含此文件的名称；否则，只需传递一个空字符串。最后，您可以定义应该发出的调试信息级别。默认设置是完整的调试信息，通过使用`FullDebug`枚举值表示。如果您只想发出行号，则可以选择`LineTablesOnly`值，或者选择`NoDebug`值以完全不发出调试信息。对于后者，最好一开始就不创建调试信息。

我们的最小化源码只使用`INTEGER`数据类型，这是一个带符号的 32 位值。为此类型创建元数据是直接的，可以在以下代码片段中看到：

```cpp
llvm::DIBasicType *DbgIntTy =
                       DBuilder.createBasicType("INTEGER", 32,
                                  llvm::dwarf::DW_ATE_signed);
```

要为函数创建调试元数据，我们首先必须为签名创建一个类型，然后为函数本身创建元数据。这类似于为函数创建 IR。函数的签名是一个数组，其中包含源顺序中所有参数的类型以及函数的返回类型作为索引`0`处的第一个元素。通常，此数组是动态构建的。在我们的情况下，我们也可以静态构建元数据。这对于内部函数（例如模块初始化）非常有用。通常，这些函数的参数是已知的，并且编译器编写者可以硬编码它们。代码如下所示：

```cpp
llvm::Metadata *DbgSigTy = {DbgIntTy};
llvm::DITypeRefArray DbgParamsTy =
                      DBuilder.getOrCreateTypeArray(DbgSigTy);
llvm::DISubroutineType *DbgFuncTy =
                   DBuilder.createSubroutineType(DbgParamsTy);
```

我们的函数具有`INTEGER`返回类型和没有其他参数，因此`DbgSigTy`数组仅包含指向此类型元数据的指针。这个静态数组被转换成类型数组，然后用于创建函数的类型。

函数本身需要更多的数据，如下所示：

```cpp
unsigned LineNo = 5;
unsigned ScopeLine = 5;
llvm::DISubprogram *DbgFunc = DBuilder.createFunction(
      DbgCU, "Func", "_t4File4Func", DbgFile, LineNo,
      DbgFuncTy, ScopeLine, 
      llvm::DISubprogram::FlagPrivate,
      llvm::DISubprogram::SPFlagLocalToUnit);
```

函数属于编译单元，在我们的案例中存储在`DbgCU`变量中。我们需要在源文件中指定函数的名称，即`Func`，并且搅乱的名称存储在目标文件中。这些信息帮助调试器在以后定位函数的机器代码。根据`tinylang`的规则，搅乱的名称是`_t4File4Func`。我们还需要指定包含函数的文件。

这一开始可能听起来令人惊讶，但想想 C 和 C++中的包含机制：一个函数可以存储在不同的文件中，然后在主编译单元中用`#include`包含。在这里，情况并非如此，我们使用与编译单元相同的文件。接下来，传递函数的行号和函数类型。函数的行号可能不是函数的词法范围开始的行号。在这种情况下，您可以指定不同的`ScopeLine`。函数还有保护，我们在这里用`FlagPrivate`值指定为私有函数。其他可能的值是`FlagPublic`和`FlagProtected`，分别表示公共和受保护的函数。

除了保护级别，这里还可以指定其他标志。例如，`FlagVirtual`表示虚函数，`FlagNoReturn`表示函数不会返回给调用者。您可以在`llvm/include/llvm/IR/DebugInfoFlags.def`的 LLVM 包含文件中找到所有可能的值的完整列表。最后，还可以指定特定于函数的标志。最常用的是`SPFlagLocalToUnit`值，表示该函数是本编译单元的本地函数。还经常使用的是`MainSubprogram`值，表示该函数是应用程序的主函数。您还可以在前面提到的 LLVM 包含文件中找到所有可能的值。

到目前为止，我们只创建了引用静态数据的元数据。变量是动态的，我们将在下一节中探讨如何将静态元数据附加到 IR 代码以访问变量。

## 跟踪变量及其值

要有用，上一节中描述的类型元数据需要与源程序的变量关联起来。对于全局变量，这相当容易。`llvm::DIBuilder`类的`createGlobalVariableExpression()`函数创建了描述全局变量的元数据。这包括源中变量的名称、搅乱的名称、源文件等。LLVM IR 中的全局变量由`GlobalVariable`类的实例表示。该类有一个`addDebugInfo()`方法，它将从`createGlobalVariableExpression()`返回的元数据节点与全局变量关联起来。

对于局部变量，我们需要采取另一种方法。LLVM IR 不知道表示局部变量的类；它只知道值。LLVM 社区开发的解决方案是在函数的 IR 代码中插入对内部函数的调用。内部函数是 LLVM 知道的函数，因此可以对其进行一些魔术操作。在大多数情况下，内部函数不会导致机器级别的子例程调用。在这里，函数调用是一个方便的工具，用于将元数据与值关联起来。

调试元数据最重要的内部函数是`llvm.dbg.declare`和`llvm.dbg.value`。前者用于声明局部变量的地址，而后者在将局部变量设置为新值时调用。

未来的 LLVM 版本将用 llvm.dbg.addr 内部函数替换 llvm.dbg.declare

`llvm.dbg.declare`内部函数做出了一个非常强烈的假设：调用中描述的变量的地址在函数的整个生命周期内都是有效的。这个假设使得在优化期间保留调试元数据变得非常困难，因为真实的存储地址可能会发生变化。为了解决这个问题，设计了一个名为`llvm.dbg.addr`的新内部函数。这个内部函数接受与`llvm.dbg.declare`相同的参数，但语义不那么严格。它仍然描述了局部变量的地址，前端应该生成对它的调用。

在优化期间，传递可以用（可能是多个）对`llvm.dbg.value`和/或`llvm.dbg.addr`的调用来替换这个内部函数，以保留调试信息。

当`llvm.dbg.addr`的工作完成后，`llvm.dbg.declare`内部函数将被弃用并最终移除。

它是如何工作的？LLVM IR 表示和通过`llvm::DIBuilder`类进行编程创建有些不同，因此我们需要同时看两者。继续上一节的例子，我们使用`alloca`指令在`Func`函数内为`i`变量分配局部存储空间，如下所示：

```cpp
@i = alloca i32
```

之后，我们添加一个对`llvm.dbg.declare`内部函数的调用，如下所示：

```cpp
call void @llvm.dbg.declare(metadata i32* %i,
                        metadata !1, metadata 
                        !DIExpression())
```

第一个参数是局部变量的地址。第二个参数是描述局部变量的元数据，由`llvm::DIBuilder`类的`createAutoVariable()`或`createParameterVariable()`调用创建。第三个参数描述一个地址表达式，稍后我会解释。

让我们实现 IR 创建。您可以使用`llvm::IRBuilder<>`类的`CreateAlloca()`方法为`@i`局部变量分配存储空间，如下所示：

```cpp
llvm::Type *IntTy = llvm::Type::getInt32Ty(LLVMCtx);
llvm::Value *Val = Builder.CreateAlloca(IntTy, nullptr, "i");
```

`LLVMCtx`变量是使用的上下文类，`Builder`是`llvm::IRBuilder<>`类的实例。

局部变量也需要用元数据描述，如下所示：

```cpp
llvm::DILocalVariable *DbgLocalVar =
 DBuilder.createAutoVariable(DbgFunc, "i", DbgFile,
                             7, DbgIntTy);
```

使用上一节中的值，我们指定变量是`DbgFunc`函数的一部分，名称为`i`，在由`DbgFile`命名的文件中定义，位于*第 7 行*，类型为`DbgIntTy`。

最后，我们使用`llvm.dbg.declare`内部函数将调试元数据与变量的地址关联起来。使用`llvm::DIBuilder`可以屏蔽掉添加调用的所有细节。代码如下所示：

```cpp
llvm::DILocation *DbgLoc =
                llvm::DILocation::get(LLVMCtx, 7, 5, 
                                      DbgFunc);
DBuilder.insertDeclare(Val, DbgLocalVar,
                       DBuilder.createExpression(), DbgLoc,
                       Val.getParent());
```

同样，我们需要为变量指定源位置。`llvm::DILocation`的实例是一个容器，用于保存与作用域关联的位置的行和列。`insertDeclare()`方法向 LLVM IR 添加对内部函数的调用。作为参数，它需要变量的地址（存储在`Val`中）和变量的调试元数据（存储在`DbgValVar`中）。我们还传递了一个空地址表达式和之前创建的调试位置。与普通指令一样，我们需要指定将调用插入到哪个基本块中。如果我们指定了一个基本块，那么调用将插入到末尾。或者，我们可以指定一个指令，调用将插入到该指令之前。我们有指向`alloca`指令的指针，这是我们插入到基本块中的最后一个指令。因此，我们使用这个基本块，调用将在`alloca`指令之后追加。

如果局部变量的值发生变化，那么必须在 IR 中添加对`llvm.dbg.value`的调用。您可以使用`llvm::DIBuilder`的`insertValue()`方法来实现。对于`llvm.dbg.addr`也是类似的。不同之处在于，现在指定的是变量的新值，而不是变量的地址。

在我们为函数实现 IR 生成时，我们使用了一种先进的算法，主要使用值并避免为局部变量分配存储空间。为了添加调试信息，这意味着我们在 Clang 生成的 IR 中使用`llvm.dbg.value`的频率要比你看到的要高得多。

如果变量没有专用存储空间，而是属于较大的聚合类型，我们可以怎么办？可能出现这种情况的一种情况是使用嵌套函数。为了实现对调用者堆栈帧的访问，您需要将所有使用的变量收集到一个结构中，并将指向此记录的指针传递给被调用的函数。在被调用的函数内部，您可以将调用者的变量视为函数的本地变量。不同的是，这些变量现在是聚合的一部分。

在调用`llvm.dbg.declare`时，如果调试元数据描述了第一个参数指向的整个内存，则使用空表达式。如果它只描述内存的一部分，则需要添加一个表达式，指示元数据适用于内存的哪一部分。在嵌套帧的情况下，需要计算到帧的偏移量。您需要访问`DataLayout`实例，可以从您正在创建 IR 代码的 LLVM 模块中获取。如果`llvm::Module`实例命名为`Mod`，则包含嵌套帧结构的变量命名为`Frame`，类型为`llvm::StructType`，并且您可以访问帧的第三个成员。然后，您可以得到成员的偏移量，如下面的代码片段所示：

```cpp
const llvm::DataLayout &DL = Mod->getDataLayout();
uint64_t Ofs = DL.getStructLayout(Frame)
               ->getElementOffset(3);
```

表达式是从一系列操作中创建的。为了访问帧的第三个成员，调试器需要将偏移量添加到基指针。您需要创建一个数组和这个信息，例如：

```cpp
llvm::SmallVector<int64_t, 2> AddrOps;
AddrOps.push_back(llvm::dwarf::DW_OP_plus_uconst);
AddrOps.push_back(Offset);
```

从这个数组中，您可以创建一个表达式，然后将其传递给`llvm.dbg.declare`，而不是空表达式，如下所示：

```cpp
llvm::DIExpression *Expr = DBuilder.createExpression(AddrOps);
```

您不仅限于此偏移操作。DWARF 知道许多不同的操作符，您可以创建相当复杂的表达式。您可以在`llvm/include/llvm/BinaryFormat/Dwarf.def` LLVM 包含文件中找到操作符的完整列表。

现在，您可以为变量创建调试信息。为了使调试器能够跟踪源代码中的控制流，您还需要提供行号信息，这是下一节的主题。

## 添加行号

调试器允许程序员逐行浏览应用程序。为此，调试器需要知道哪些机器指令属于源代码中的哪一行。LLVM 允许在每条指令中添加源位置。在上一节中，我们创建了`llvm::DILocation`类型的位置信息。调试位置具有比行、列和作用域更多的信息。如果需要，可以指定此行内联的作用域。还可以指示此调试位置属于隐式代码，即前端生成的但不在源代码中的代码。

在将调试位置附加到指令之前，我们必须将调试位置包装在`llvm::DebugLoc`对象中。为此，您只需将从`llvm::DILocation`类获得的位置信息传递给`llvm::DebugLoc`构造函数。通过这种包装，LLVM 可以跟踪位置信息。虽然源代码中的位置显然不会改变，但是源级语句或表达式的生成机器代码可能会在优化期间被丢弃。封装有助于处理这些可能的更改。

将行号信息添加到生成的指令中主要是从 AST 中检索行号信息，并将其添加到生成的指令中。`llvm::Instruction`类有`setDebugLoc()`方法，它将位置信息附加到指令上。

在下一节中，我们将向我们的`tinylang`编译器添加调试信息的生成。

## 为 tinylang 添加调试支持

我们将调试元数据的生成封装在新的`CGDebugInfo`类中。我们将声明放入`tinylang/CodeGen/CGDebugInfo.h`头文件中，将定义放入`tinylang/CodeGen/CGDebugInfo.cpp`文件中。

`CGDebugInfo`类有五个重要成员。我们需要模块的代码生成器`CGM`的引用，因为我们需要将 AST 表示的类型转换为 LLVM 类型。当然，我们还需要`llvm::DIBuilder`类的实例`DBuilder`，就像前面的部分一样。还需要编译单元的指针，并将其存储在名为`CU`的成员中。

为了避免重复创建类型的调试元数据，我们还添加了一个用于缓存这些信息的映射。成员称为`TypeCache`。最后，我们需要一种管理作用域信息的方法，为此我们基于`llvm::SmallVector<>`类创建了一个名为`ScopeStack`的堆栈。因此，我们有以下代码：

```cpp
  CGModule &CGM;
  llvm::DIBuilder DBuilder;
  llvm::DICompileUnit *CU;
  llvm::DenseMap<TypeDeclaration *, llvm::DIType *>
      TypeCache;
  llvm::SmallVector<llvm::DIScope *, 4> ScopeStack;
```

`CGDebugInfo`类的以下方法都使用了这些成员：

1.  首先，我们需要在构造函数中创建编译单元。我们还在这里创建包含编译单元的文件。稍后，我们可以通过`CU`成员引用该文件。构造函数的代码如下所示：

```cpp
CGDebugInfo::CGDebugInfo(CGModule &CGM)
    : CGM(CGM), DBuilder(*CGM.getModule()) {
  llvm::SmallString<128> Path(
      CGM.getASTCtx().getFilename());
  llvm::sys::fs::make_absolute(Path);
  llvm::DIFile *File = DBuilder.createFile(
      llvm::sys::path::filename(Path),
      llvm::sys::path::parent_path(Path));
  bool IsOptimized = false;
  unsigned ObjCRunTimeVersion = 0;
  llvm::DICompileUnit::DebugEmissionKind EmissionKind =
      llvm::DICompileUnit::DebugEmissionKind::FullDebug;
  CU = DBuilder.createCompileUnit(
      llvm::dwarf::DW_LANG_Modula2, File, "tinylang",
      IsOptimized, StringRef(), ObjCRunTimeVersion,
      StringRef(), EmissionKind);
}
```

1.  我们经常需要提供行号。这可以从源管理器位置派生，大多数 AST 节点都可以使用。源管理器可以将其转换为行号，如下所示：

```cpp
unsigned CGDebugInfo::getLineNumber(SMLoc Loc) {
  return CGM.getASTCtx().getSourceMgr().FindLineNumber(
      Loc);
}
```

1.  作用域的信息保存在堆栈上。我们需要方法来打开和关闭作用域，并检索当前作用域。编译单元是全局作用域，我们会自动添加它，如下所示：

```cpp
llvm::DIScope *CGDebugInfo::getScope() {
  if (ScopeStack.empty())
    openScope(CU->getFile());
  return ScopeStack.back();
}
void CGDebugInfo::openScope(llvm::DIScope *Scope) {
  ScopeStack.push_back(Scope);
}
void CGDebugInfo::closeScope() {
  ScopeStack.pop_back();
}
```

1.  我们为需要转换的类型的每个类别创建一个方法。`getPervasiveType()`方法为基本类型创建调试元数据。请注意以下代码片段中对编码参数的使用，声明`INTEGER`类型为有符号类型，`BOOLEAN`类型编码为布尔类型：

```cpp
llvm::DIType *
CGDebugInfo::getPervasiveType(TypeDeclaration *Ty) {
  if (Ty->getName() == "INTEGER") {
    return DBuilder.createBasicType(
        Ty->getName(), 64, llvm::dwarf::DW_ATE_signed);
  }
  if (Ty->getName() == "BOOLEAN") {
    return DBuilder.createBasicType(
        Ty->getName(), 1, 
            llvm::dwarf::DW_ATE_boolean);
  }
  llvm::report_fatal_error(
      "Unsupported pervasive type");
}
```

1.  如果类型名称只是重命名，那么我们将其映射到类型定义。在这里，我们需要首次使用作用域和行号信息，如下所示：

```cpp
llvm::DIType *
CGDebugInfo::getAliasType(AliasTypeDeclaration *Ty) {
  return DBuilder.createTypedef(
      getType(Ty->getType()), Ty->getName(),
      CU->getFile(), getLineNumber(Ty->getLocation()),
      getScope());
}
```

1.  为数组创建调试信息需要指定大小和对齐方式。我们从`DataLayout`类中检索这些数据。我们还需要指定数组的索引范围。我们可以使用以下代码来实现：

```cpp
llvm::DIType *
CGDebugInfo::getArrayType(ArrayTypeDeclaration *Ty) {
  auto *ATy =
      llvm::cast<llvm::ArrayType>(CGM.convertType(Ty));
  const llvm::DataLayout &DL =
      CGM.getModule()->getDataLayout();
  uint64_t NumElements = Ty->getUpperIndex();
  llvm::SmallVector<llvm::Metadata *, 4> Subscripts;
  Subscripts.push_back(
      DBuilder.getOrCreateSubrange(0, NumElements));
  return DBuilder.createArrayType(
      DL.getTypeSizeInBits(ATy) * 8,
      DL.getABITypeAlignment(ATy),
      getType(Ty->getType()),
      DBuilder.getOrCreateArray(Subscripts));
}
```

1.  使用所有这些单个方法，我们创建一个中心方法来为类型创建元数据。这个元数据还负责缓存数据。代码可以在以下代码片段中看到：

```cpp
llvm::DIType *
CGDebugInfo::getType(TypeDeclaration *Ty) {
  if (llvm::DIType *T = TypeCache[Ty])
    return T;
  if (llvm::isa<PervasiveTypeDeclaration>(Ty))
    return TypeCache[Ty] = getPervasiveType(Ty);
  else if (auto *AliasTy =
               llvm::dyn_cast<AliasTypeDeclaration>(Ty))
    return TypeCache[Ty] = getAliasType(AliasTy);
  else if (auto *ArrayTy =
               llvm::dyn_cast<ArrayTypeDeclaration>(Ty))
    return TypeCache[Ty] = getArrayType(ArrayTy);
  else if (auto *RecordTy =
               llvm ::dyn_cast<RecordTypeDeclaration>(
                   Ty))
    return TypeCache[Ty] = getRecordType(RecordTy);
  llvm::report_fatal_error("Unsupported type");
  return nullptr;
}
```

1.  我们还需要添加一个方法来发出全局变量的元数据，如下所示：

```cpp
void CGDebugInfo::emitGlobalVariable(
    VariableDeclaration *Decl,
    llvm::GlobalVariable *V) {
  llvm::DIGlobalVariableExpression *GV =
      DBuilder.createGlobalVariableExpression(
          getScope(), Decl->getName(), V->getName(),
          CU->getFile(),
          getLineNumber(Decl->getLocation()),
          getType(Decl->getType()), false);
  V->addDebugInfo(GV);
}
```

1.  要为过程发出调试信息，我们首先需要为过程类型创建元数据。为此，我们需要参数类型的列表，返回类型是第一个条目。如果过程没有返回类型，则使用一个称为`void`的未指定类型，就像 C 语言一样。如果参数是引用，则需要添加引用类型；否则，我们将类型添加到列表中。代码如下所示：

```cpp
llvm::DISubroutineType *
CGDebugInfo::getType(ProcedureDeclaration *P) {
  llvm::SmallVector<llvm::Metadata *, 4> Types;
  const llvm::DataLayout &DL =
      CGM.getModule()->getDataLayout();
  // Return type at index 0
  if (P->getRetType())
    Types.push_back(getType(P->getRetType()));
  else
    Types.push_back(
        DBuilder.createUnspecifiedType("void"));
  for (const auto *FP : P->getFormalParams()) {
    llvm::DIType *PT = getType(FP->getType());
    if (FP->isVar()) {
      llvm::Type *PTy = CGM.convertType(FP->getType());
      PT = DBuilder.createReferenceType(
          llvm::dwarf::DW_TAG_reference_type, PT,
          DL.getTypeSizeInBits(PTy) * 8,
          DL.getABITypeAlignment(PTy));
    }
    Types.push_back(PT);
  }
  return DBuilder.createSubroutineType(
      DBuilder.getOrCreateTypeArray(Types));
}
```

1.  对于过程本身，我们现在可以使用上一步创建的过程类型创建调试信息。过程还会打开一个新的作用域，因此我们将该过程推送到作用域堆栈上。我们还将 LLVM 函数对象与新的调试信息关联起来，如下所示：

```cpp
void CGDebugInfo::emitProcedure(
    ProcedureDeclaration *Decl, llvm::Function *Fn) {
  llvm::DISubroutineType *SubT = getType(Decl);
  llvm::DISubprogram *Sub = DBuilder.createFunction(
      getScope(), Decl->getName(), Fn->getName(),
      CU->getFile(), getLineNumber(Decl->getLocation()),
      SubT, getLineNumber(Decl->getLocation()),
      llvm::DINode::FlagPrototyped,
      llvm::DISubprogram::SPFlagDefinition);
  openScope(Sub);
  Fn->setSubprogram(Sub);
}
```

1.  当到达过程的结束时，我们必须通知构建器完成该过程的调试信息的构建。我们还需要从作用域堆栈中移除该过程。我们可以使用以下代码来实现：

```cpp
void CGDebugInfo::emitProcedureEnd(
    ProcedureDeclaration *Decl, llvm::Function *Fn) {
  if (Fn && Fn->getSubprogram())
    DBuilder.finalizeSubprogram(Fn->getSubprogram());
  closeScope();
}
```

1.  最后，当我们完成添加调试信息时，我们需要将`finalize()`方法添加到构建器上。然后验证生成的调试信息。这是开发过程中的重要步骤，因为它可以帮助您找到错误生成的元数据。代码可以在以下代码片段中看到：

```cpp
void CGDebugInfo::finalize() { DBuilder.finalize(); }
```

只有在用户请求时才应生成调试信息。我们将需要一个新的命令行开关来实现这一点。我们将把这个开关添加到`CGModule`类的文件中，并且在这个类内部也会使用它，如下所示：

```cpp
static llvm::cl::opt<bool>
    Debug("g", llvm::cl::desc("Generate debug information"),
          llvm::cl::init(false));
```

`CGModule`类持有`std::unique_ptr<CGDebugInfo>`类的实例。指针在构造函数中初始化，关于命令行开关的设置如下：

```cpp
  if (Debug)
    DebugInfo.reset(new CGDebugInfo(*this));
```

在 getter 方法中，我们返回指针，就像这样：

```cpp
CGDebugInfo *getDbgInfo() {
  return DebugInfo.get();
}
```

生成调试元数据时的常见模式是检索指针并检查其是否有效。例如，在创建全局变量后，我们以这种方式添加调试信息：

```cpp
VariableDeclaration *Var = …;
llvm::GlobalVariable *V = …;
if (CGDebugInfo *Dbg = getDbgInfo())
  Dbg->emitGlobalVariable(Var, V);
```

为了添加行号信息，我们需要在`CGDebugInfo`类中添加一个`getDebugLoc()`转换方法，将 AST 中的位置信息转换为调试元数据，如下所示：

```cpp
llvm::DebugLoc CGDebugInfo::getDebugLoc(SMLoc Loc) {
  std::pair<unsigned, unsigned> LineAndCol =
      CGM.getASTCtx().getSourceMgr().getLineAndColumn(Loc);
  llvm::DILocation *DILoc = llvm::DILocation::get(
      CGM.getLLVMCtx(), LineAndCol.first, LineAndCol.second,
      getCU());
  return llvm::DebugLoc(DILoc);
}
```

然后可以调用`CGModule`类中的实用函数来将行号信息添加到指令中，如下所示：

```cpp
void CGModule::applyLocation(llvm::Instruction *Inst,
                             llvm::SMLoc Loc) {
  if (CGDebugInfo *Dbg = getDbgInfo())
    Inst->setDebugLoc(Dbg->getDebugLoc(Loc));
}
```

通过这种方式，您可以为自己的编译器添加调试信息。

# 总结

在本章中，您了解了在 LLVM 中如何抛出和捕获异常，以及需要生成哪些 IR 代码来利用此功能。为了增强 IR 的范围，您学习了如何将各种元数据附加到指令上。基于类型的别名的元数据为 LLVM 优化器提供了额外的信息，并有助于进行某些优化以生成更好的机器代码。用户总是欣赏使用源级调试器的可能性，通过向 IR 代码添加调试信息，您可以提供编译器的这一重要功能。

优化 IR 代码是 LLVM 的核心任务。在下一章中，我们将学习通道管理器的工作原理以及如何影响通道管理器管理的优化流水线。
