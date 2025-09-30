

# 高级 IR 生成

在前几章中介绍了 IR 生成后，你就可以实现编译器所需的大部分功能。在本章中，我们将探讨一些在现实世界编译器中经常出现的高级主题。例如，许多现代语言都使用了异常处理，因此我们将探讨如何将其转换为 LLVM IR。

为了支持 LLVM 优化器，以便它在某些情况下产生更好的代码，我们必须向 IR 代码中添加额外的类型元数据。此外，附加调试元数据使编译器的用户能够利用源级调试工具。

在本章中，我们将涵盖以下主题：

+   *抛出和捕获异常*：在这里，你将学习如何在你的编译器中实现异常处理

+   *为基于类型的别名分析生成元数据*：在这里，你将为 LLVM IR 附加额外的元数据，这有助于 LLVM 更好地优化代码

+   *添加调试元数据*：在这里，你将实现添加到生成的 IR 代码中的调试信息所需的支持类

到本章结束时，你将了解异常处理，以及基于类型的别名分析和调试信息的元数据。

# 抛出和捕获异常

LLVM IR 中的异常处理与平台支持紧密相关。在这里，我们将探讨使用`libunwind`的最常见的异常处理类型。C++使用了它的全部潜力，因此我们将首先查看一个 C++的例子，其中`bar()`函数可以抛出`int`或`double`值：

```cpp

int bar(int x) {
  if (x == 1) throw 1;
  if (x == 2) throw 42.0;
  return x;
}
```

`foo()`函数调用`bar()`，但只处理抛出的`int`。它还声明它只抛出`int`值：

```cpp

int foo(int x) {
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

抛出异常需要调用运行时库两次；这可以在`bar()`函数中看到。首先，通过调用`__cxa_allocate_exception()`为异常分配内存。这个函数接受要分配的字节数作为参数。异常负载（在这个例子中的`int`或`double`值）被复制到分配的内存中。然后，通过调用`__cxa_throw()`来引发异常。这个函数接受三个参数：分配的异常的指针、负载的类型信息以及指向析构函数的指针，以防异常负载有一个。`__cxa_throw()`函数启动堆栈回溯过程，并且永远不会返回。在 LLVM IR 中，这是对`int`值进行的，如下所示：

```cpp

%eh = call ptr @__cxa_allocate_exception(i64 4)
store i32 1, ptr %eh
call void @__cxa_throw(ptr %eh, ptr @_ZTIi, ptr null)
unreachable
```

`_ZTIi`是描述`int`类型的类型信息。对于`double`类型，它将是`_ZTId`。

到目前为止，还没有进行任何特定于 LLVM 的操作。这在前面的`foo()`函数中发生了变化，因为对`bar()`的调用可能会抛出异常。如果它是一个`int`类型的异常，那么控制流必须转移到捕获子句的 IR 代码。为了完成这个任务，必须使用`invoke`指令而不是`call`指令：

```cpp

%y = invoke i32 @_Z3bari(i32 %x) to label %next
                                 unwind label %lpad
```

这两个指令之间的区别在于`invoke`有两个标签相关联。第一个标签是在被调用函数正常结束（通常使用`ret`指令）时继续执行的地方。在示例代码中，这个标签被称为`%next`。如果发生异常，则执行继续在所谓的*着陆点*，标签为`%lpad`。

着陆点是一个必须以`landingpad`指令开始的代码块。`landingpad`指令向 LLVM 提供有关处理异常类型的信息。例如，一个可能的着陆点可能看起来像这样：

```cpp

lpad:
%exc = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIi
          filter [1 x ptr] [ptr @_ZTIi]
```

这里可能有三种操作类型：

+   `cleanup`：这表示存在清理当前状态的代码。通常，这用于调用局部对象的析构函数。如果存在此标记，则在栈回溯期间始终调用着陆点。

+   `catch`：这是一个类型-值对的列表，表示可以处理的异常类型。如果抛出的异常类型在此列表中，则调用着陆点。在`foo()`函数的情况下，值是 C++运行时类型信息指针，类似于`__cxa_throw()`函数的参数。

+   `filter`：这指定了一个异常类型数组。如果当前异常的类型不在数组中，则调用着陆点。这用于实现`throw()`规范。对于`foo()`函数，数组只有一个成员——`int`类型的类型信息。

`landingpad`指令的结果类型是`{ ptr, i32 }`结构。第一个元素是指向抛出异常的指针，而第二个是类型选择器。让我们从结构中提取这两个值：

```cpp

%exc.ptr = extractvalue { ptr, i32 } %exc, 0
%exc.sel = extractvalue { ptr, i32 } %exc, 1
```

*类型选择器*是一个帮助我们识别*为什么着陆点被调用*的原因的数字。如果当前异常类型与`landingpad`指令的`catch`部分中给出的异常类型之一匹配，则该值是正数。如果当前异常类型与`filter`部分中给出的任何值都不匹配，则该值为负数。如果应该调用清理代码，则该值为`0`。

类型选择器是一个类型信息表的偏移量，该表由`landingpad`指令的`catch`和`filter`部分给出的值构建而成。在优化过程中，多个着陆点可以合并为一个，这意味着该表的结构在 IR 级别上是未知的。为了检索给定类型的类型选择器，我们需要调用内建的`@llvm.eh.typeid.for`函数。我们需要这个函数来检查类型选择器值是否对应于`int`类型的类型信息，以便我们可以在`catch (int e) {}`块中执行代码：

```cpp

%tid.int = call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
%tst.int = icmp eq i32 %exc.sel, %tid.int
br i1 %tst.int, label %catchint, label %filterorcleanup
```

异常处理是通过调用 `__cxa_begin_catch()` 和 `__cxa_end_catch()` 来框架化的。`__cxa_begin_catch()` 函数需要一个参数——当前异常，它是 `landingpad` 指令返回的值之一。它返回异常负载的指针——在我们的例子中是一个 `int` 值。

`__cxa_end_catch()` 函数标记了异常处理的结束，并释放了使用 `__cxa_allocate_exception()` 分配的内存。请注意，如果在 `catch` 块内部抛出另一个异常，运行时行为会变得更加复杂。异常处理如下：

```cpp

catchint:
%payload = call ptr @__cxa_begin_catch(ptr %exc.ptr)
%retval = load i32, ptr %payload
call void @__cxa_end_catch()
br label %return
```

如果当前异常的类型与 `throws()` 声明中的列表不匹配，则调用未预期的异常处理器。首先，我们需要再次检查类型选择器：

```cpp

filterorcleanup:
%tst.blzero = icmp slt i32 %exc.sel, 0
br i1 %tst.blzero, label %filter, label %cleanup
```

如果类型选择器的值小于 `0`，则调用处理器：

```cpp

filter:
call void @__cxa_call_unexpected(ptr %exc.ptr) #4
unreachable
```

再次，处理器不应该返回。

在这种情况下不需要进行清理工作，因此所有清理代码所做的只是恢复堆栈回溯器的执行：

```cpp

cleanup:
resume { ptr, i32 } %exc
```

还缺少一部分：`libunwind` 驱动堆栈回溯过程，但它并不绑定到单一的语言。语言相关的处理在个性函数中完成。对于 Linux 上的 C++，个性函数被调用为 `__gxx_personality_v0()`。根据平台或编译器，这个名称可能会有所不同。每个需要参与堆栈回溯的函数都有一个个性函数附加。这个个性函数分析函数是否捕获了异常，是否有不匹配的过滤器列表，或者是否需要清理调用。它将此信息返回给回溯器，回溯器据此采取行动。在 LLVM IR 中，个性函数的指针作为函数定义的一部分给出：

```cpp

define i32 @_Z3fooi(i32) personality ptr @__gxx_personality_v0
```

这样，异常处理功能就完成了。

在编译器中为您的编程语言使用异常处理的最简单策略是利用现有的 C++ 运行时函数。这也具有优势，即您的异常可以与 C++ 兼容。缺点是您将一些 C++ 运行时绑定到您语言的运行时中，最显著的是内存管理。如果您想避免这种情况，那么您需要创建自己的 `_cxa_` 函数等效物。尽管如此，您仍然会想使用 `libunwind`，它提供了堆栈回溯机制：

1.  让我们看看如何创建这个 IR。我们在 *第二章* 中创建了 `calc` 表达式编译器，*编译器的结构*。现在，我们将扩展表达式编译器的代码生成器，以便在执行除以零操作时抛出和处理异常。生成的 IR 将检查除法的除数是否为 `0`。如果是，则抛出异常。我们还将向函数中添加一个 landing pad，它捕获异常并将 `Divide by zero!` 打印到控制台并结束计算。在这个简单的情况下，使用异常处理不是必需的，但它允许我们专注于代码生成过程。我们必须将所有代码添加到 `CodeGen.cpp` 文件中。我们首先添加所需的新字段和一些辅助方法。首先，我们需要存储 `__cxa_allocate_exception()` 和 `__cxa_throw()` 函数的 LLVM 声明，这些声明包括函数类型和函数本身。需要一个 `GlobalVariable` 实例来存储类型信息。我们还需要引用包含 landing pad 的基本块和一个只包含 `unreachable` 指令的基本块：

    ```cpp

      GlobalVariable *TypeInfo = nullptr;
      FunctionType *AllocEHFty = nullptr;
      Function *AllocEHFn = nullptr;
      FunctionType *ThrowEHFty = nullptr;
      Function *ThrowEHFn = nullptr;
      BasicBlock *LPadBB = nullptr;
      BasicBlock *UnreachableBB = nullptr;
    ```

1.  我们还将添加一个新的辅助函数来创建比较两个值的 IR。`createICmpEq()` 函数接受要比较的 `Left` 和 `Right` 值作为参数。它创建一个比较指令来测试值的相等性，并为相等和不相等的情况创建一个分支指令到两个基本块。这两个基本块通过 `TrueDest` 和 `FalseDest` 参数返回。此外，可以在 `TrueLabel` 和 `FalseLabel` 参数中给出新基本块的标签。代码如下：

    ```cpp

      void createICmpEq(Value *Left, Value *Right,
                        BasicBlock *&TrueDest,
                        BasicBlock *&FalseDest,
                        const Twine &TrueLabel = "",
                        const Twine &FalseLabel = "") {
        Function *Fn =
            Builder.GetInsertBlock()->getParent();
        TrueDest = BasicBlock::Create(M->getContext(),
                                      TrueLabel, Fn);
        FalseDest = BasicBlock::Create(M->getContext(),
                                       FalseLabel, Fn);
        Value *Cmp = Builder.CreateCmp(CmpInst::ICMP_EQ,
                                       Left, Right);
        Builder.CreateCondBr(Cmp, TrueDest, FalseDest);
      }
    ```

1.  要使用运行时函数，我们需要创建几个函数声明。在 LLVM 中，函数类型给出签名，而函数本身必须构造。我们使用 `createFunc()` 方法创建这两个对象。函数需要 `FunctionType` 和 `Function` 指针的引用，新声明的函数的名称，以及结果类型。参数类型列表是可选的，表示变量参数列表的标志设置为 `false`，表示参数列表中没有变量部分：

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

在完成这些准备工作后，我们可以生成 IR 来抛出异常。

## 抛出异常

要生成抛出异常的 IR 代码，我们将添加 `addThrow()` 方法。这个新方法需要初始化新的字段，然后通过 `__cxa_throw()` 函数生成抛出异常的 IR。抛出的异常的有效负载是 `int` 类型，可以设置为任意值。以下是我们需要编写的代码：

1.  新的 `addThrow()` 方法首先检查 `TypeInfo` 字段是否已初始化。如果没有初始化，则创建一个名为 `_ZTIi` 的 `i8` 指针类型的全局外部常量。这代表了描述 C++ `int` 类型的 C++ 元数据：

    ```cpp

      void addThrow(int PayloadVal) {
        if (!TypeInfo) {
          TypeInfo = new GlobalVariable(
              *M, Int8PtrTy,
              /*isConstant=*/true,
              GlobalValue::ExternalLinkage,
              /*Initializer=*/nullptr, "_ZTIi");
    ```

1.  初始化继续通过使用我们的辅助`createFunc()`方法创建`__cxa_allocate_exception()`和`__cxa_throw()`函数的 IR 声明：

    ```cpp

          createFunc(AllocEHFty, AllocEHFn,
                     "__cxa_allocate_exception", Int8PtrTy,
                     {Int64Ty});
          createFunc(ThrowEHFty, ThrowEHFn, "__cxa_throw",
                     VoidTy,
                     {Int8PtrTy, Int8PtrTy, Int8PtrTy});
    ```

1.  使用异常处理的函数需要一个个人函数，它有助于栈回溯。我们添加 IR 代码来声明来自 C++库的`__gxx_personality_v0()`个人函数，并将其设置为当前函数的个人例程。当前函数不是作为字段存储的，但我们可以使用`Builder`实例来查询当前基本块，该基本块将函数存储为`Parent`字段：

    ```cpp

          FunctionType *PersFty;
          Function *PersFn;
          createFunc(PersFty, PersFn,
                     "__gxx_personality_v0", Int32Ty, std::nulopt,                  true);
          Function *Fn =
              Builder.GetInsertBlock()->getParent();
          Fn->setPersonalityFn(PersFn);
    ```

1.  接下来，我们必须创建并填充着陆地基本块。首先，我们需要保存当前基本块的指针。然后，我们必须创建一个新的基本块，将其设置在构建器中以便可以使用它来插入指令，并调用`addLandingPad()`方法。这个方法生成处理异常的 IR 代码，并在下一节*捕获异常*中描述。这段代码填充了着陆地基本块：

    ```cpp

          BasicBlock *SaveBB = Builder.GetInsertBlock();
          LPadBB = BasicBlock::Create(M->getContext(),
                                      "lpad", Fn);
          Builder.SetInsertPoint(LPadBB);
          addLandingPad();
    ```

1.  通过创建包含`unreachable`指令的基本块来完成初始化部分。再次，我们创建基本块并将其设置为构建器的插入点。然后，我们可以向其中添加`unreachable`指令。最后，我们可以将构建器的插入点设置回保存的`SaveBB`实例，以便以下 IR 添加到正确的基本块：

    ```cpp

          UnreachableBB = BasicBlock::Create(
              M->getContext(), "unreachable", Fn);
          Builder.SetInsertPoint(UnreachableBB);
          Builder.CreateUnreachable();
          Builder.SetInsertPoint(SaveBB);
        }
    ```

1.  要抛出异常，我们需要通过调用`__cxa_allocate_exception()`函数为异常和有效负载分配内存。我们的有效负载是 C++的`int`类型，通常大小为 4 字节。我们创建一个常量无符号值作为大小，并用它作为参数调用该函数。函数类型和函数声明已经初始化，所以我们只需要创建`call`指令：

    ```cpp

        Constant *PayloadSz =
            ConstantInt::get(Int64Ty, 4, false);
        CallInst *EH = Builder.CreateCall(
            AllocEHFty, AllocEHFn, {PayloadSz});
    ```

1.  接下来，我们将`PayloadVal`值存储在分配的内存中。为此，我们需要创建一个调用`ConstantInt::get()`函数的 LLVM IR 常量。分配的内存的指针是`i8`指针类型；为了存储`i32`类型的值，我们需要创建一个`bitcast`指令来转换类型：

    ```cpp

        Value *PayloadPtr =
            Builder.CreateBitCast(EH, Int32PtrTy);
        Builder.CreateStore(
            ConstantInt::get(Int32Ty, PayloadVal, true),
            PayloadPtr);
    ```

1.  最后，我们必须通过调用`__cxa_throw()`函数来抛出异常。由于这个函数会抛出异常，而这个异常也在同一个函数中处理，因此我们需要使用`invoke`指令而不是`call`指令。与`call`指令不同，`invoke`指令会结束一个基本块，因为它有两个后续的基本块。在这里，这些是`UnreachableBB`和`LPadBB`基本块。如果函数没有抛出异常，控制流将转移到`UnreachableBB`基本块。由于`__cxa_throw()`函数的设计，这种情况永远不会发生，因为控制流会转移到`LPadBB`基本块来处理异常。这完成了`addThrow()`方法的实现：

    ```cpp

        Builder.CreateInvoke(
            ThrowEHFty, ThrowEHFn, UnreachableBB, LPadBB,
            {EH,
             ConstantExpr::getBitCast(TypeInfo, Int8PtrTy),
             ConstantPointerNull::get(Int8PtrTy)});
      }
    ```

接下来，我们将添加生成处理异常的 IR 的代码。

## 捕获异常

要生成捕获异常的 IR 代码，我们必须添加`addLandingPad()`方法。生成的 IR 从异常中提取类型信息。如果它与 C++的`int`类型匹配，则异常通过将`Divide by zero!`打印到控制台并从函数返回来处理。如果类型不匹配，我们只需执行`resume`指令，将控制权转回运行时。由于调用堆栈中没有其他函数来处理这个异常，运行时将终止应用程序。以下步骤描述了生成捕获异常所需的代码：

1.  在生成的 IR 中，我们需要从 C++运行时库中调用`__cxa_begin_catch()`和`__cxa_end_catch()`函数。为了打印错误消息，我们将生成对 C 运行时库中的`puts()`函数的调用。此外，为了从异常中获取类型信息，我们必须生成对`llvm.eh.typeid.for`内建函数的调用。我们还需要所有这些的`FunctionType`和`Function`实例；我们将利用我们的`createFunc()`方法来创建它们：

    ```cpp

      void addLandingPad() {
        FunctionType *TypeIdFty; Function *TypeIdFn;
        createFunc(TypeIdFty, TypeIdFn,
                   "llvm.eh.typeid.for", Int32Ty,
                   {Int8PtrTy});
        FunctionType *BeginCatchFty; Function *BeginCatchFn;
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

1.  `landingpad`指令是我们生成的第一个指令。结果类型是一个包含`i8`指针和`i32`类型字段的结构。我们通过调用`StructType::get()`函数生成此结构。此外，由于我们需要处理 C++ `int`类型的异常，我们还需要将其添加为`landingpad`指令的一个子句，该指令必须是一个`i8`指针类型的常量。这意味着需要生成一个`bitcast`指令来将`TypeInfo`值转换为该类型。之后，我们必须将指令返回的值存储在`Exc`变量中，以供以后使用：

    ```cpp

        LandingPadInst *Exc = Builder.CreateLandingPad(
            StructType::get(Int8PtrTy, Int32Ty), 1, "exc");
        Exc->addClause(
            ConstantExpr::getBitCast(TypeInfo, Int8PtrTy));
    ```

1.  接下来，我们从返回值中提取类型选择器。通过调用`llvm.eh.typeid.for`内建函数，我们检索代表 C++ `int`类型的`TypeInfo`字段的类型 ID。有了这个 IR，我们已经生成了我们需要比较的两个值，以决定我们是否可以处理这个异常：

    ```cpp

        Value *Sel =
            Builder.CreateExtractValue(Exc, {1}, "exc.sel");
        CallInst *Id =
            Builder.CreateCall(TypeIdFty, TypeIdFn,
                               {ConstantExpr::getBitCast(
                                   TypeInfo, Int8PtrTy)});
    ```

1.  要生成比较的 IR，我们必须调用我们的`createICmpEq()`函数。此函数还生成两个基本块，我们将它们存储在`TrueDest`和`FalseDest`变量中：

    ```cpp

        BasicBlock *TrueDest, *FalseDest;
        createICmpEq(Sel, Id, TrueDest, FalseDest, "match",
                     "resume");
    ```

1.  如果两个值不匹配，控制流将继续在`FalseDest`基本块中。此基本块仅包含一个`resume`指令，以将控制权交还给 C++运行时：

    ```cpp

        Builder.SetInsertPoint(FalseDest);
        Builder.CreateResume(Exc);
    ```

1.  如果两个值相等，控制流将继续在`TrueDest`基本块中。首先，我们生成 IR 代码以从`landingpad`指令的返回值中提取指向异常的指针，该返回值存储在`Exc`变量中。然后，我们生成对`__cxa_begin_catch ()`函数的调用，并将指向异常的指针作为参数传递。这表示运行时开始处理异常：

    ```cpp

        Builder.SetInsertPoint(TrueDest);
        Value *Ptr =
            Builder.CreateExtractValue(Exc, {0}, "exc.ptr");
        Builder.CreateCall(BeginCatchFty, BeginCatchFn,
                           {Ptr});
    ```

1.  然后通过调用`puts()`函数来处理异常，打印一条消息到控制台。为此，我们通过调用`CreateGlobalStringPtr()`函数生成一个指向字符串的指针，然后将此指针作为参数传递给生成的`puts()`函数调用：

    ```cpp

        Value *MsgPtr = Builder.CreateGlobalStringPtr(
            "Divide by zero!", "msg", 0, M);
        Builder.CreateCall(PutsFty, PutsFn, {MsgPtr});
    ```

1.  现在我们已经处理了异常，我们必须生成对`__cxa_end_catch()`函数的调用，以通知运行时。最后，我们使用`ret`指令从函数返回：

    ```cpp

        Builder.CreateCall(EndCatchFty, EndCatchFn);
        Builder.CreateRet(Int32Zero);
      }
    ```

使用`addThrow()`和`addLandingPad()`函数，我们可以生成 IR 来抛出异常和处理异常。然而，我们仍然需要添加 IR 来检查除数是否为`0`。我们将在下一节中介绍这一点。

## 将异常处理代码集成到应用程序中

除法的 IR 是在`visit(BinaryOp &)`方法中生成的。我们不仅需要生成一个`sdiv`指令，还必须生成一个 IR 来比较除数与`0`。如果除数为 0，则控制流在基本块中继续，抛出异常。否则，控制流在带有`sdiv`指令的基本块中继续。借助`createICmpEq()`和`addThrow()`函数，我们可以非常容易地实现这一点：

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

代码生成部分现在已完成。要构建应用程序，我们必须切换到构建目录并运行`ninja`工具：

```cpp

$ ninja
```

构建完成后，你可以使用`with a:` `3/a`表达式来检查生成的 IR：

```cpp

$ src/calc "with a: 3/a"
```

你将看到抛出和捕获异常所需的额外 IR。

生成的 IR 现在依赖于 C++运行时。链接所需库的最简单方法是使用`clang++`编译器。将表达式计算器的运行时函数的`rtcalc.c`文件重命名为`rtcalc.cpp`，并在文件中的每个函数前添加`extern "C"`。然后，使用`llc`工具将生成的 IR 转换为对象文件，并使用`clang++`编译器创建可执行文件：

```cpp

$ src/calc "with a: 3/a" | llc -filetype obj -o exp.o
$ clang++ -o exp exp.o ../rtcalc.cpp
```

现在，我们可以使用不同的值运行生成的应用程序：

```cpp

$ ./exp
Enter a value for a: 1
The result is: 3
$ ./exp
Enter a value for a: 0
Divide by zero!
```

在第二次运行中，输入是`0`，这会抛出异常。它按预期工作！

在本节中，我们学习了如何抛出和捕获异常。生成 IR 的代码可以用作其他编译器的蓝图。当然，使用的类型信息和捕获子句的数量取决于编译器的输入，但我们需要生成的 IR 仍然遵循本节中展示的模式。

添加元数据是向 LLVM 提供更多信息的另一种方式。在下一节中，我们将添加类型元数据以支持 LLVM 优化器在特定情况下的工作。

# 为基于类型的别名分析生成元数据

两个指针可能指向同一个内存单元，此时它们相互别名。在 LLVM 模型中，内存没有类型，这使得优化器难以决定两个指针是否相互别名。如果编译器可以证明两个指针不会相互别名，那么可以执行更多的优化。在下一节中，我们将更详细地研究这个问题，并在实现此方法之前探讨添加额外元数据如何有所帮助。

## 理解额外元数据的需求

为了展示问题，让我们看看以下函数：

```cpp

void doSomething(int *p, float *q) {
  *p = 42;
  *q = 3.1425;
}
```

优化器无法决定指针 `p` 和 `q` 是否指向同一个内存单元。在优化过程中，可以进行一个重要的分析，称为 `p` 和 `q` 指向同一个内存单元，那么它们是别名。此外，如果优化器可以证明这两个指针永远不会相互别名，这将使额外的优化机会成为可能。例如，在 `doSomething()` 函数中，存储操作可以被重新排序，而不会改变结果。

此外，一个类型的变量是否可以是另一个不同类型变量的别名，这取决于源语言的定义。请注意，语言也可能包含破坏基于类型的别名假设的表达式——例如，不同类型之间的类型转换。

LLVM 开发者选择的解决方案是在 `load` 和 `store` 指令中添加元数据。添加的元数据有两个目的：

+   首先，它基于哪种类型可以别名另一种类型来定义类型层次结构

+   其次，它描述了 `load` 或 `store` 指令中的内存访问

让我们看看 C 中的类型层次结构。每个类型层次结构都以一个根节点开始，无论是**命名**的还是**匿名**的。LLVM 假设具有相同名称的根节点描述了相同类型的层次结构。你可以在同一个 LLVM 模块中使用不同的类型层次结构，LLVM 假设这些类型可能存在别名。在根节点之下，有标量类型的节点。聚合类型的节点不直接连接到根节点，但它们引用标量类型和其他聚合类型。Clang 将 C 的层次结构定义为如下：

+   根节点被称为 `Simple C/C++ TBAA`。

+   在根节点之下是 `char` 类型的节点。这在 C 中是一个特殊类型，因为所有指针都可以转换为指向 `char` 的指针。

+   在 `char` 节点之下是其他标量类型的节点以及所有指针的类型，称为 `any pointer`。

此外，聚合类型被定义为成员类型和偏移量的序列。

这些元数据定义用于附加到 `load` 和 `store` 指令的访问标签上。一个访问标签由三部分组成：一个基类型、一个访问类型和一个偏移量。根据基类型的不同，访问标签描述内存访问的方式有两种可能：

1.  如果基类型是聚合类型，则访问标签描述了具有必要访问类型的 `struct` 成员的内存访问，并位于给定的偏移量处。

1.  如果基类型是标量类型，则访问类型必须与基类型相同，偏移量必须为 `0`。

使用这些定义，我们现在可以在访问标签上定义一个关系，该关系用于评估两个指针是否可能相互别名。让我们更仔细地看看 `(base type, offset)` 元组的直接父节点的选项：

1.  如果基类型是标量类型且偏移量为 0，则直接父节点是 `(父类型，0)`，其中父类型是父节点在类型层次结构中定义的类型。如果偏移量不为 0，则直接父节点未定义。

1.  如果基类型是聚合类型，则 `(base type, offset)` 元组的直接父节点是 `(new type, new offset)` 元组，其中新类型是偏移量处的成员的类型。新偏移量是新类型的偏移量，调整到其新的起始位置。

这个关系的传递闭包是父关系。两个内存访问，(基类型 1，访问类型 1，偏移量 1) 和 (基类型 2，访问类型 2，偏移量 2)，如果 (基类型 1，偏移量 1) 和 (基类型 2，偏移量 2) 或反之在父关系中相关联，则它们可能相互别名。

让我们用一个例子来说明这一点：

```cpp

struct Point { float x, y; }
void func(struct Point *p, float *x, int *i, char *c) {
  p->x = 0; p->y = 0; *x = 0.0; *i = 0; *c = 0;
}
```

当使用标量类型的内存访问标签定义时，`i` 参数的访问标签是 (`int`，`int`，0)，而 `c` 参数的访问标签是 (`char`，`char`，0)。在类型层次结构中，`int` 类型节点的父节点是 `char` 节点。因此，(`int`，0) 的直接父节点是 (`char`，0)，并且两个指针可以别名。对于 `x` 和 `c` 参数也是如此。然而，`x` 和 `i` 参数不相关，因此它们不会相互别名。`struct Point` 的 `y` 成员的访问是 (`Point`，`float`，4)，其中 4 是 `y` 成员在结构体中的偏移量。(`Point`，4) 的直接父节点是 (`float`，0)，因此对 `p->y` 和 `x` 的访问可能别名，同样地，根据相同的推理，也与 `c` 参数相关。

## 在 LLVM 中创建 TBAA 元数据

要创建元数据，我们必须使用 `llvm::MDBuilder` 类，该类在 `llvm/IR/MDBuilder.h` 头文件中声明。数据本身存储在 `llvm::MDNode` 和 `llvm::MDString` 类的实例中。使用构建器类可以保护我们免受构建内部细节的影响。

通过调用 `createTBAARoot()` 方法创建一个根节点，该方法期望类型层次结构的名称作为参数并返回根节点。可以使用 `createAnonymousTBAARoot()` 方法创建一个匿名、唯一的根节点。

使用 `createTBAAScalarTypeNode()` 方法将标量类型添加到层次结构中，该方法接受类型名称和父节点作为参数。

另一方面，为聚合类型添加类型节点稍微复杂一些。`createTBAAStructTypeNode()`方法接受类型名称和字段列表作为参数。具体来说，字段以`std::pair<llvm::MDNode*, uint64_t>`实例给出，其中第一个元素表示成员的类型，第二个元素表示在`struct`中的偏移量。

使用`createTBAAStructTagNode()`方法创建一个访问标签，该方法接受基类型、访问类型和偏移量作为参数。

最后，元数据必须附加到`load`或`store`指令。`llvm::Instruction`类包含一个名为`setMetadata()`的方法，用于添加基于类型的各种别名分析元数据。第一个参数必须是`llvm::LLVMContext::MD_tbaa`类型，第二个参数必须是访问标签。

带着这些知识，我们必须为`tinylang`添加元数据。

## 添加 TBAA 元数据到 tinylang

为了支持 TBAA，我们必须添加一个新的`CGTBAA`类。这个类负责生成元数据节点。此外，我们将`CGTBAA`类作为`CGModule`类的成员，命名为`TBAA`。

每个加载和存储指令都必须进行注释。在`CGModule`类中为此目的创建了一个新函数，称为`decorateInst()`。此函数尝试创建标签访问信息。如果成功，则将元数据附加到相应的加载或存储指令。此外，这种设计还允许我们在不需要时关闭元数据生成过程，例如在关闭优化的构建中：

```cpp

void CGModule::decorateInst(llvm::Instruction *Inst,
                            TypeDeclaration *Type) {
  if (auto *N = TBAA.getAccessTagInfo(Type))
    Inst->setMetadata(llvm::LLVMContext::MD_tbaa, N);
}
```

我们将新`CGTBAA`类的声明放在`include/tinylang/CodeGen/CGTBAA.h`头文件中，定义放在`lib/CodeGen/CGTBAA.cpp`文件中。除了 AST 定义外，头文件还需要包含定义元数据节点和构建器的文件：

```cpp

#include "tinylang/AST/AST.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
```

`CGTBAA`类需要存储一些数据成员。那么，让我们一步一步地看看如何做这个步骤：

1.  首先，我们需要缓存类型层次结构的根：

    ```cpp

     class CGTBAA {
      llvm::MDNode *Root;
    ```

1.  为了构建元数据节点，我们需要`MDBuilder`类的一个实例：

    ```cpp

      llvm::MDBuilder MDHelper;
    ```

1.  最后，我们必须存储为类型生成的元数据以供重用：

    ```cpp

      llvm::DenseMap<TypeDenoter *, llvm::MDNode *> MetadataCache;
    // …
    };
    ```

现在我们已经定义了构建所需变量，我们必须添加创建元数据所需的方法：

1.  构造函数初始化数据成员：

    ```cpp

    CGTBAA::CGTBAA(CGModule &CGM)
          : CGM(CGM),
            MDHelper(llvm::MDBuilder(CGM.getLLVMCtx())),
            Root(nullptr) {}
    ```

1.  我们必须延迟实例化类型层次结构的根，我们将其命名为`Simple` `tinylang TBAA`：

    ```cpp

    llvm::MDNode *CGTBAA::getRoot() {
      if (!Root)
        Root = MDHelper.createTBAARoot("Simple tinylang TBAA");
      return Root;
    }
    ```

1.  对于标量类型，我们必须使用基于类型的名称通过`MDBuilder`类创建一个元数据节点。新的元数据节点存储在缓存中：

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

1.  创建记录元数据的方法更为复杂，因为我们必须枚举记录的所有字段。类似于标量类型，新的元数据节点存储在缓存中：

    ```cpp

    llvm::MDNode *CGTBAA::createStructTypeNode(
        TypeDeclaration *Ty, StringRef Name,
        llvm::ArrayRef<std::pair<llvm::MDNode *, uint64_t>>
            Fields) {
      llvm::MDNode *N =
          MDHelper.createTBAAStructTypeNode(Name, Fields);
      return MetadataCache[Ty] = N;
    }
    ```

1.  为了返回 `tinylang` 类型的元数据，我们需要创建类型层次结构。由于 `tinylang` 的类型系统非常受限，我们可以使用简单的方法。每个标量类型映射到一个与根节点附加的唯一类型，我们将所有指针映射到单个类型。结构化类型然后引用这些节点。如果我们无法映射类型，则返回 `nullptr`：

    ```cpp

    llvm::MDNode *CGTBAA::getTypeInfo(TypeDeclaration *Ty) {
      if (llvm::MDNode *N = MetadataCache[Ty])
        return N;
      if (auto *Pervasive =
              llvm::dyn_cast<PervasiveTypeDeclaration>(Ty)) {
        StringRef Name = Pervasive->getName();
        return createScalarTypeNode(Pervasive, Name, getRoot());
      }
      if (auto *Pointer =
              llvm::dyn_cast<PointerTypeDeclaration>(Ty)) {
        StringRef Name = "any pointer";
        return createScalarTypeNode(Pointer, Name, getRoot());
      }
      if (auto *Array =
             llvm::dyn_cast<ArrayTypeDeclaration>(Ty)) {
        StringRef Name = Array->getType()->getName();
        return createScalarTypeNode(Array, Name, getRoot());
      }
      if (auto *Record =
              llvm::dyn_cast<RecordTypeDeclaration>(Ty)) {
        llvm::SmallVector<std::pair<llvm::MDNode *, uint64_t>,     4> Fields;
        auto *Rec =
            llvm::cast<llvm::StructType>(CGM.convertType(Record));
        const llvm::StructLayout *Layout =
            CGM.getModule()->getDataLayout().getStructLayout(Rec);
        unsigned Idx = 0;
        for (const auto &F : Record->getFields()) {
          uint64_t Offset = Layout->getElementOffset(Idx);
          Fields.emplace_back(getTypeInfo(F.getType()), Offset);
          ++Idx;
        }
        StringRef Name = CGM.mangleName(Record);
        return createStructTypeNode(Record, Name, Fields);
      }
      return nullptr;
    }
    ```

1.  获取元数据的一般方法是 `getAccessTagInfo()`。要获取 TBAA 访问标签信息，必须添加对 `getTypeInfo()` 函数的调用。该函数期望 `TypeDeclaration` 作为其参数，该参数是从我们想要为生成元数据的指令中检索到的：

    ```cpp

    llvm::MDNode *CGTBAA::getAccessTagInfo(TypeDeclaration *Ty) {
        return getTypeInfo(Ty);
    }
    ```

最后，为了启用 TBAA 元数据的生成，我们只需将元数据附加到我们在 `tinylang` 中生成的所有加载和存储指令上。

例如，在 `CGProcedure::writeVariable()` 中，对一个全局变量的存储使用了一个存储指令：

```cpp

      Builder.CreateStore(Val, CGM.getGlobal(D));
```

为了装饰这个特定的指令，我们需要将这一行替换为以下行，其中 `decorateInst()` 将 TBAA 元数据添加到这个存储指令中：

```cpp

      auto *Inst = Builder.CreateStore(Val, CGM.getGlobal(D));
      // NOTE: V is of the VariableDeclaration class, and
      // the getType() method in this class retrieves the
      // TypeDeclaration that is needed for decorateInst().
      CGM.decorateInst(Inst, V->getType());
```

在这些更改到位后，我们已经完成了 TBAA 元数据的生成。

现在，我们可以将一个示例 `tinylang` 文件编译成 LLVM 中间表示，以查看我们新实现的 TBAA 元数据。例如，考虑以下文件，`Person.mod`：

```cpp

MODULE Person;
TYPE
  Person = RECORD
             Height: INTEGER;
             Age: INTEGER
           END;
PROCEDURE Set(VAR p: Person);
BEGIN
  p.Age := 18;
END Set;
END Person.
```

本章构建目录中构建的 `tinylang` 编译器可以用来为该文件生成中间表示：

```cpp

$ tools/driver/tinylang -emit-llvm ../examples/Person.mod
```

在新生成的 `Person.ll` 文件中，我们可以看到存储指令被装饰了我们本章内生成的 TBAA 元数据，其中元数据反映了最初声明的记录类型的字段：

```cpp

; ModuleID = '../examples/Person.mod'
source_filename = "../examples/Person.mod"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin22.6.0"
define void @_t6Person3Set(ptr nocapture dereferenceable(16) %p) {
entry:
  %0 = getelementptr inbounds ptr, ptr %p, i32 0, i32 1
  store i64 18, ptr %0, align 8, !tbaa !0
  ret void
}
!0 = !{!"_t6Person6Person", !1, i64 0, !1, i64 8}
!1 = !{!"INTEGER", !2, i64 0}
!2 = !{!"Simple tinylang TBAA"}
```

现在我们已经学会了如何生成 TBAA 元数据，我们将在下一节中探索一个非常类似的主题：生成调试元数据。

# 添加调试元数据

为了允许源级调试，我们必须添加调试信息。LLVM 对调试信息支持使用调试元数据来描述源语言类型和其他静态信息，以及内联函数来跟踪变量值。LLVM 核心库在 Unix 系统上使用 *DWARF 格式* 生成调试信息，在 Windows 上使用 *PDB 格式*。我们将在下一节中查看一般结构。

## 理解调试元数据的一般结构

为了描述一般结构，LLVM 使用与基于类型的分析元数据相似的元数据。静态结构描述了文件、编译单元、函数和词法块，以及使用的数据类型。

我们主要使用的是 `llvm::DIBuilder` 类，我们需要使用 `llvm/IR/DIBuilder` 头文件来获取类声明。这个构建器类提供了一个易于使用的接口来创建调试元数据。稍后，这些元数据要么被添加到 LLVM 对象（如全局变量）中，要么用于调用调试内嵌函数。以下是构建器类可以创建的一些重要元数据：

+   `llvm::DIFile`：它使用文件名和包含该文件的目录的绝对路径来描述一个文件。您可以使用 `createFile()` 方法来创建它。一个文件可以包含主编译单元，也可以包含导入的声明。

+   `llvm::DICompileUnit`：它用于描述当前的编译单元。在众多其他信息中，您指定源语言、编译器特定的生产者字符串、是否启用优化，以及当然，`DIFile`，其中包含编译单元。您可以通过调用 `createCompileUnit()` 来创建它。

+   `llvm::DISubprogram`：它描述一个函数。这里最重要的信息是作用域（通常是嵌套函数的 `DICompileUnit` 或 `DISubprogram`），函数的名称、函数的混淆名称和函数类型。它通过调用 `createFunction()` 来创建。

+   `llvm::DILexicalBlock`：它描述一个词法块，并模拟了许多高级语言中发现的块作用域。您可以通过调用 `createLexicalBlock()` 来创建它。

LLVM 对你的编译器所翻译的语言没有任何假设。因此，它没有关于该语言数据类型的信息。为了支持源级调试，特别是显示调试器中的变量值，还必须添加类型信息。以下是一些重要的结构：

+   `createBasicType()` 函数，它返回指向 `llvm::DIBasicType` 类的指针，用于创建描述基本类型（如 `tinylang` 中的 `INTEGER` 或 C++ 中的 `int`）的元数据。除了类型的名称外，所需的参数还包括位大小和编码——例如，如果是有符号或无符号类型。

+   构造复合数据类型的元数据有几种方法，如 `llvm::DIComposite` 类所示。您可以使用 `createArrayType()`、`createStructType()`、`createUnionType()` 和 `createVectorType()` 函数分别实例化数组、结构体、联合和向量数据类型的元数据。这些函数需要您预期的参数，例如数组类型的基类型和订阅数，或结构体类型的字段成员列表。

+   同样存在支持枚举、模板、类等方法。

函数列表显示你必须将源语言的每一个细节都添加到调试信息中。假设你的`llvm::DIBuilder`类实例称为`DBuilder`。假设你有一些`tinylang`源代码位于`/home/llvmuser`文件夹中的`File.mod`文件中。在这个文件中，第 5 行有一个`Func():INTEGER`函数，其中包含在第 7 行的局部`VAR i:INTEGER`声明。让我们为这个创建元数据，从文件的信息开始。你需要指定文件名和文件所在文件夹的绝对路径：

```cpp

llvm::DIFile *DbgFile = DBuilder.createFile("File.mod",
                                            "/home/llvmuser");
```

该文件是`tinylang`中的一个模块，这使得它成为 LLVM 的编译单元。这包含了很多信息：

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

此外，调试器需要知道源语言。DWARF 标准定义了一个包含所有常见值的枚举。其缺点之一是你不能简单地添加一个新的源语言。为此，你必须向 DWARF 委员会提出请求。请注意，调试器和其他调试工具也需要对新语言的支持——仅仅添加枚举的新成员是不够的。

在许多情况下，选择一种接近你的源语言的语言就足够了。对于`tinylang`来说，这是 Modula-2，我们使用`DW_LANG_Modula2`作为语言标识符。编译单元位于一个文件中，该文件由我们之前创建的`DbgFile`变量标识。此外，调试信息可以携带有关生产者的信息，这可以是编译器的名称和版本信息。在这里，我们只是传递了`tinylang`字符串。如果你不想添加这些信息，那么你可以简单地使用一个空字符串作为参数。

下一个信息集包括`IsOptimized`标志，该标志应指示编译器是否已开启优化。通常，此标志是从`–O`命令行开关派生出来的。你可以通过`CUFlags`参数将额外的参数设置传递给调试器。这里我们不使用它，所以我们传递一个空字符串。我们也不使用 Objective-C，所以我们传递`0`作为 Objective-C 运行时版本。

通常，调试信息嵌入在我们创建的对象文件中。如果我们想将调试信息写入一个单独的文件，那么`SplitName`参数必须包含该文件的名称。否则，简单地传递一个空字符串就足够了。最后，你可以定义应该输出的调试信息的级别。默认情况下是完整的调试信息，如使用`FullDebug`枚举值所示，但你也可以选择`LineTablesOnly`值以仅输出行号，或者选择`NoDebug`值以完全不输出调试信息。对于后者，最好从一开始就不创建调试信息。

我们的最简源代码只使用了`INTEGER`数据类型，这是一个有符号的 32 位值。为这个类型创建元数据是直接的：

```cpp

llvm::DIBasicType *DbgIntTy =
                       DBuilder.createBasicType("INTEGER", 32,
                                  llvm::dwarf::DW_ATE_signed);
```

要为函数创建调试元数据，我们首先必须为签名创建一个类型，然后是函数本身的元数据。这与为函数创建 IR 的过程类似。函数的签名是一个数组，包含源顺序中所有参数的类型以及函数的返回类型作为索引`0`的第一个元素。通常，这个数组是动态构建的。在我们的例子中，我们也可以静态构建元数据。这对于内部函数很有用，例如用于模块初始化。通常，这些函数的参数总是已知的，编译器编写者可以将它们硬编码：

```cpp

llvm::Metadata *DbgSigTy = {DbgIntTy};
llvm::DITypeRefArray DbgParamsTy =
                      DBuilder.getOrCreateTypeArray(DbgSigTy);
llvm::DISubroutineType *DbgFuncTy =
                   DBuilder.createSubroutineType(DbgParamsTy);
```

我们的这个函数具有`INTEGER`返回类型，没有其他参数，因此`DbgSigTy`数组只包含此类型的元数据指针。这个静态数组被转换成一个类型数组，然后用于创建函数的类型。

函数本身需要更多的数据：

```cpp

unsigned LineNo = 5;
unsigned ScopeLine = 5;
llvm::DISubprogram *DbgFunc = DBuilder.createFunction(
      DbgCU, "Func", "_t4File4Func", DbgFile, LineNo,
      DbgFuncTy, ScopeLine, llvm::DISubprogram::FlagPrivate,
      llvm::DISubprogram::SPFlagLocalToUnit);
```

一个函数属于一个编译单元，在我们的例子中，它存储在`DbgCU`变量中。我们需要在源文件中指定函数的名称，它是`Func`，而混淆后的名称存储在目标文件中。这些信息有助于调试器定位函数的机器代码。根据`tinylang`的规则，混淆后的名称是`_t4File4Func`。我们还需要指定包含该函数的文件。

这可能一开始听起来令人惊讶，但想想 C 和 C++中的包含机制：一个函数可以存储在不同的文件中，然后在主编译单元中使用`#include`包含它。在这里，情况并非如此，我们使用与编译单元相同的文件。接下来，传递函数的行号和函数类型。函数的行号可能不是函数词法作用域开始的行号。在这种情况下，你可以指定不同的`ScopeLine`。函数还有保护级别，我们使用`FlagPrivate`值来指定私有函数。函数保护的其它可能值是`FlagPublic`和`FlagProtected`，分别表示公共和受保护的函数。

除了保护级别之外，还可以在此处指定其他标志。例如，`FlagVirtual`表示虚函数，而`FlagNoReturn`表示该函数不会返回给调用者。你可以在 LLVM 包含文件中找到可能的完整值列表——即`llvm/include/llvm/IR/DebugInfoFlags.def`。

最后，可以指定特定于函数的标志。最常用的标志是`SPFlagLocalToUnit`值，它表示该函数是此编译单元的局部函数。`MainSubprogram`值也经常使用，表示该函数是应用程序的主函数。前面提到的 LLVM 包含文件还列出了所有与特定于函数的标志相关的可能值。

到目前为止，我们只创建了指向静态数据的元数据。变量是动态的，所以我们将探讨如何在下一节中将静态元数据附加到 IR 代码以访问变量。

## 跟踪变量及其值

为了变得有用，上一节中描述的类型元数据需要与源程序中的变量相关联。对于全局变量，这很简单。`llvm::DIBuilder`类的`createGlobalVariableExpression()`函数创建描述全局变量的元数据。这包括源变量名、混淆名、源文件等。在 LLVM IR 中，全局变量由`GlobalVariable`类的实例表示。这个类有一个名为`addDebugInfo()`的方法，它将`createGlobalVariableExpression()`返回的元数据节点与全局变量关联。

对于局部变量，我们需要采取另一种方法。LLVM IR 不知道表示局部变量的类，因为它只知道值。LLVM 社区开发的解决方案是在函数的 IR 代码中插入内建函数的调用。一个`llvm.dbg.declare`和一个`llvm.dbg.value`。

`llvm.dbg.declare`内建函数提供信息，并且由前端生成一次，用于声明局部变量。本质上，这个内建函数描述了局部变量的地址。在优化过程中，传递可以替换这个内建函数为（可能多个）对`llvm.dbg.value`的调用，以保留调试信息并跟踪局部源变量。优化后，可能存在多个`llvm.dbg.declare`调用，因为它用于描述局部变量在内存中存在的程序点。

另一方面，每当局部变量被设置为新的值时，都会调用`llvm.dbg.value`内建函数。这个内建函数描述了局部变量的值，而不是其地址。

这一切都是如何工作的？LLVM IR 表示和通过`llvm::DIBuilder`类的程序性创建略有不同，所以我们将查看两者。

继续我们上一节中的例子，我们将在`Func`函数内部使用`alloca`指令为`I`变量分配局部存储：

```cpp

@i = alloca i32
```

之后，我们必须添加对`llvm.dbg.declare`内建函数的调用：

```cpp

call void @llvm.dbg.declare(metadata ptr %i,
                        metadata !1, metadata !DIExpression())
```

第一个参数是局部变量的地址。第二个参数是描述局部变量的元数据，它通过调用`createAutoVariable()`为局部变量或`createParameterVariable()`为`llvm::DIBuilder`类的参数创建。最后，第三个参数描述了一个地址表达式，稍后将会解释。

让我们实现 IR 的创建。你可以通过调用`llvm::IRBuilder<>`类的`CreateAlloca()`方法为局部`@i`变量分配存储：

```cpp

llvm::Type *IntTy = llvm::Type::getInt32Ty(LLVMCtx);
llvm::Value *Val = Builder.CreateAlloca(IntTy, nullptr, "i");
```

`LLVMCtx`变量是使用的上下文类，`Builder`是`llvm::IRBuilder<>`类的使用实例。

局部变量也需要由元数据来描述：

```cpp

llvm::DILocalVariable *DbgLocalVar =
 Dbuilder.createAutoVariable(DbgFunc, "i", DbgFile,
                             7, DbgIntTy);
```

使用上一节中的值，我们可以指定变量是`DbgFunc`函数的一部分，被命名为`i`，在`DbgFile`文件的第*7*行定义，并且是`DbgIntTy`类型。

最后，我们使用`llvm.dbg.declare`内省将调试元数据与变量的地址关联起来。使用`llvm::DIBuilder`可以让你免于添加调用的所有细节：

```cpp

llvm::DILocation *DbgLoc =
                llvm::DILocation::get(LLVMCtx, 7, 5, DbgFunc);
DBuilder.insertDeclare(Val, DbgLocalVar,
                       DBuilder.createExpression(), DbgLoc,
                       Val.getParent());
```

同样，我们必须为变量指定一个源位置。`llvm::DILocation`的一个实例是一个容器，它包含与作用域关联的位置的行和列。此外，`insertDeclare()`方法将调用添加到 LLVM IR 的内省函数中。就这个函数的参数而言，它需要存储在`Val`中的变量的地址和存储在`DbgValVar`中的变量的调试元数据。我们还将传递一个空地址表达式和之前创建的调试位置。与正常指令一样，我们需要指定调用插入到哪个基本块中。如果我们指定一个基本块，那么调用将被插入到块的末尾。或者，我们可以指定一个指令，调用将被插入到该指令之前。我们还有`alloca`指令的指针，这是我们最后插入到基本块中的指令。因此，我们可以使用这个基本块，并且调用将在`alloca`指令之后附加。

如果局部变量的值发生了变化，那么必须在 IR 中添加对`llvm.dbg.value`的调用以设置局部变量的新值。可以使用`llvm::DIBuilder`类的`insertValue()`方法来实现这一点。

当我们实现函数的 IR 生成时，我们使用了一个高级算法，该算法主要使用值并避免为局部变量分配存储空间。在添加调试信息方面，这意味着我们比在 clang 生成的 IR 中更频繁地使用`llvm.dbg.value`。

如果变量没有专门的存储空间，而是属于一个更大的聚合类型，我们该怎么办？这种情况可能出现在嵌套函数的使用中。为了实现对调用者栈帧的访问，你必须在一个结构中收集所有使用的变量，并将指向这个记录的指针传递给被调用函数。在被调用函数内部，你可以像引用局部变量一样引用调用者的变量。不同之处在于，这些变量现在成为了聚合的一部分。

在`llvm.dbg.declare`的调用中，如果你使用调试元数据描述了第一个参数指向的整个内存，则使用一个空表达式。然而，如果它只描述内存的一部分，那么你需要添加一个表达式来指示元数据适用于内存的哪一部分。

在嵌套帧的情况下，你需要计算帧中的偏移量。你需要访问一个 `DataLayout` 实例，你可以从创建 IR 代码的 LLVM 模块中获取它。如果 `llvm::Module` 实例命名为 `Mod`，并且持有嵌套帧结构的变量命名为 `Frame` 且为 `llvm::StructType` 类型，你可以以下方式访问帧的第三个成员。这种访问给你成员的偏移量：

```cpp

const llvm::DataLayout &DL = Mod->getDataLayout();
uint64_t Ofs = DL.getStructLayout(Frame)->getElementOffset(3);
```

此外，表达式是由一系列操作创建的。要访问帧的第三个成员，调试器需要将偏移量添加到基指针。例如，你需要创建一个数组以及类似的信息：

```cpp

llvm::SmallVector<int64_t, 2> AddrOps;
AddrOps.push_back(llvm::dwarf::DW_OP_plus_uconst);
AddrOps.push_back(Offset);
```

从这个数组中，你可以创建必须传递给 `llvm.dbg.declare` 的表达式，而不是空表达式：

```cpp

llvm::DIExpression *Expr = DBuilder.createExpression(AddrOps);
```

需要注意的是，你不仅限于这种偏移操作。DWARF 知道许多不同的运算符，你可以创建相当复杂的表达式。你可以在 LLVM 的包含文件中找到运算符的完整列表，该文件名为 `llvm/include/llvm/BinaryFormat/Dwarf.def`。

在这一点上，你可以为变量创建调试信息。为了使调试器能够跟踪源代码中的控制流，你还需要提供行号信息。这是下一节的主题。

## 添加行号

调试器允许程序员逐行执行应用程序。为此，调试器需要知道哪些机器指令属于源代码中的哪一行。LLVM 允许将源位置添加到每个指令。在上一个章节中，我们创建了 `llvm::DILocation` 类型的位置信息。调试位置提供的信息比仅行、列和作用域更多。如果需要，可以指定此行内联的作用域。还可能指示此调试位置属于隐式代码——即前端生成的但不在源代码中的代码。

在此信息可以附加到指令之前，我们必须将调试位置包装在 `llvm::DebugLoc` 对象中。为此，你必须简单地将从 `llvm::DILocation` 类获得的定位信息传递给 `llvm::DebugLoc` 构造函数。通过这种包装，LLVM 可以跟踪位置信息。虽然源代码中的位置没有改变，但在优化过程中，源级语句或表达式的生成机器代码可能会被丢弃。这种封装有助于处理这些可能的变化。

添加行号信息主要归结为从抽象语法树（AST）中检索行号信息并将其添加到生成的指令中。`llvm::Instruction` 类具有 `setDebugLoc()` 方法，该方法将位置信息附加到指令上。

在下一节中，我们将学习如何生成调试信息并将其添加到我们的 `tinylang` 编译器中。

## 为 tinylang 添加调试支持

我们将调试元数据的生成封装在新的`CGDebugInfo`类中。此外，我们将声明放在`tinylang/CodeGen/CGDebugInfo.h`头文件中，定义放在`tinylang/CodeGen/CGDebugInfo.cpp`文件中。

`CGDebugInfo`类有五个重要的成员。我们需要模块的代码生成器的引用，即`CGM`，因为我们需要将 AST 表示的类型转换为 LLVM 类型。当然，我们还需要一个名为`Dbuilder`的`llvm::DIBuilder`类的实例，就像我们在前面的章节中所做的那样。还需要一个指向编译单元实例的指针；我们将其存储在`CU`成员中。

为了避免再次为类型创建调试元数据，我们还必须添加一个映射来缓存这些信息。该成员称为`TypeCache`。最后，我们需要一种管理范围信息的方法，为此我们必须基于`llvm::SmallVector<>`类创建一个名为`ScopeStack`的栈。因此，我们有以下内容：

```cpp

  CGModule &CGM;
  llvm::DIBuilder DBuilder;
  llvm::DICompileUnit *CU;
 llvm::DenseMap<TypeDeclaration *, llvm::DIType *>
      TypeCache;
  llvm::SmallVector<llvm::DIScope *, 4> ScopeStack;
```

`CGDebugInfo`类的以下方法使用了这些成员：

1.  首先，我们需要创建编译单元，这在我们构造函数中完成。我们在这里也创建了包含编译单元的文件。稍后，我们可以通过`CU`成员来引用该文件。构造函数的代码如下：

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
      llvm::StringRef CUFlags;
      unsigned ObjCRunTimeVersion = 0;
      llvm::StringRef SplitName;
      llvm::DICompileUnit::DebugEmissionKind EmissionKind =
          llvm::DICompileUnit::DebugEmissionKind::FullDebug;
      CU = DBuilder.createCompileUnit(
          llvm::dwarf::DW_LANG_Modula2, File, "tinylang",
          IsOptimized, CUFlags, ObjCRunTimeVersion,
          SplitName, EmissionKind);
    }
    ```

1.  通常，我们需要提供一个行号。行号可以从源管理器的位置推导出来，这在大多数 AST 节点中都是可用的。源管理器可以将此转换为行号：

    ```cpp

    unsigned CGDebugInfo::getLineNumber(SMLoc Loc) {
      return CGM.getASTCtx().getSourceMgr().FindLineNumber(
          Loc);
    }
    ```

1.  范围的信息存储在栈上。我们需要方法来打开和关闭范围以及检索当前范围。编译单元是全局范围，我们自动添加：

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

1.  接下来，我们必须为每个需要转换的类型类别创建一个方法。`getPervasiveType()`方法为基本类型创建调试元数据。注意使用编码参数，将`INTEGER`类型声明为有符号类型，将`BOOLEAN`类型编码为布尔值：

    ```cpp

    llvm::DIType *
    CGDebugInfo::getPervasiveType(TypeDeclaration *Ty) {
      if (Ty->getName() == "INTEGER") {
        return DBuilder.createBasicType(
            Ty->getName(), 64, llvm::dwarf::DW_ATE_signed);
      }
      if (Ty->getName() == "BOOLEAN") {
        return DBuilder.createBasicType(
            Ty->getName(), 1, llvm::dwarf::DW_ATE_boolean);
      }
      llvm::report_fatal_error(
          "Unsupported pervasive type");
    }
    ```

1.  如果类型名称只是重命名，那么我们必须将其映射到类型定义。在这里，我们需要使用范围和行号信息：

    ```cpp

    llvm::DIType *
    CGDebugInfo::getAliasType(AliasTypeDeclaration *Ty) {
      return DBuilder.createTypedef(
          getType(Ty->getType()), Ty->getName(),
          CU->getFile(), getLineNumber(Ty->getLocation()),
          getScope());
    }
    ```

1.  创建数组的调试信息需要指定大小和对齐。我们可以从`DataLayout`类中检索这些数据。我们还需要指定数组的索引范围：

    ```cpp

    llvm::DIType *
    CGDebugInfo::getArrayType(ArrayTypeDeclaration *Ty) {
      auto *ATy =
          llvm::cast<llvm::ArrayType>(CGM.convertType(Ty));
      const llvm::DataLayout &DL =
          CGM.getModule()->getDataLayout();
      Expr *Nums = Ty->getNums();
      uint64_t NumElements =
          llvm::cast<IntegerLiteral>(Nums)
              ->getValue()
              .getZExtValue();
      llvm::SmallVector<llvm::Metadata *, 4> Subscripts;
      Subscripts.push_back(
          DBuilder.getOrCreateSubrange(0, NumElements));
      return DBuilder.createArrayType(
          DL.getTypeSizeInBits(ATy) * 8,
          1 << Log2(DL.getABITypeAlign(ATy)),
          getType(Ty->getType()),
          DBuilder.getOrCreateArray(Subscripts));
    }
    ```

1.  使用所有这些单个方法，我们可以创建一个中心方法来创建类型的元数据。这个元数据也负责缓存数据：

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

1.  我们还需要添加一个方法来生成全局变量的元数据：

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

1.  为了生成过程的调试信息，我们需要为过程类型创建元数据。为此，我们需要一个参数类型的列表，其中返回类型是第一个条目。如果过程没有返回类型，那么我们必须使用未指定的类型；这被称为`void`，类似于 C 语言中的用法。如果一个参数是引用类型，那么我们需要添加引用类型；否则，我们必须将类型添加到列表中：

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
              1 << Log2(DL.getABITypeAlign(PTy)));
        }
        Types.push_back(PT);
      }
      return DBuilder.createSubroutineType(
          DBuilder.getOrCreateTypeArray(Types));
    }
    ```

1.  对于实际的过程本身，我们现在可以使用在上一步骤中创建的过程类型来创建调试信息。过程也打开了一个新的作用域，因此我们必须将过程推入作用域栈。我们还必须将 LLVM 函数对象与新的调试信息关联起来：

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

1.  当达到过程的末尾时，我们必须通知构建器完成此过程的调试信息构建。我们还需要从作用域栈中移除过程：

    ```cpp

    void CGDebugInfo::emitProcedureEnd(
        ProcedureDeclaration *Decl, llvm::Function *Fn) {
      if (Fn && Fn->getSubprogram())
        DBuilder.finalizeSubprogram(Fn->getSubprogram());
      closeScope();
    }
    ```

1.  最后，当我们完成添加调试信息后，我们需要在构建器上实现 `finalize()` 方法。生成的调试信息随后被验证。这是开发过程中的一个重要步骤，因为它有助于你找到错误生成的元数据：

    ```cpp

    void CGDebugInfo::finalize() { DBuilder.finalize(); }
    ```

调试信息只有在用户请求时才应生成。这意味着我们需要一个新的命令行开关来实现这一点。我们将将其添加到 `CGModule` 类的文件中，并且我们还将在这个类内部使用它：

```cpp

static llvm::cl::opt<bool>
    Debug("g", llvm::cl::desc("Generate debug information"),
          llvm::cl::init(false));
```

`-g` 选项可以与 `tinylang` 编译器一起使用来生成调试元数据。

此外，`CGModule` 类持有 `std::unique_ptr<CGDebugInfo>` 类的一个实例。该指针在构造函数中初始化，用于设置命令行开关：

```cpp

  if (Debug)
    DebugInfo.reset(new CGDebugInfo(*this));
```

在 `CGModule.h` 中定义的获取方法中，我们简单地返回指针：

```cpp

CGDebugInfo *getDbgInfo() {
  return DebugInfo.get();
}
```

生成调试元数据的常见模式是检索指针并检查其是否有效。例如，在创建全局变量后，我们可以这样添加调试信息：

```cpp

VariableDeclaration *Var = …;
llvm::GlobalVariable *V = …;
if (CGDebugInfo *Dbg = getDbgInfo())
  Dbg->emitGlobalVariable(Var, V);
```

要添加行号信息，我们需要在 `CGDebugInfo` 类中实现一个名为 `getDebugLoc()` 的转换方法，它将 AST 中的位置信息转换为调试元数据：

```cpp

llvm::DebugLoc CGDebugInfo::getDebugLoc(SMLoc Loc) {
  std::pair<unsigned, unsigned> LineAndCol =
      CGM.getASTCtx().getSourceMgr().getLineAndColumn(Loc);
  llvm::DILocation *DILoc = llvm::DILocation::get(
      CGM.getLLVMCtx(), LineAndCol.first, LineAndCol.second,
      getScope());
  return llvm::DebugLoc(DILoc);
}
```

此外，`CGModule` 类中的一个实用函数可以被调用来向指令添加行号信息：

```cpp

void CGModule::applyLocation(llvm::Instruction *Inst,
                             llvm::SMLoc Loc) {
  if (CGDebugInfo *Dbg = getDbgInfo())
    Inst->setDebugLoc(Dbg->getDebugLoc(Loc));
}
```

以这种方式，你可以为你的编译器添加调试信息。

# 摘要

在本章中，你学习了在 LLVM 和 IR 中抛出和捕获异常的工作原理，以及你可以生成以利用此功能的功能。为了扩展 IR 的范围，你学习了如何将各种元数据附加到指令上。基于类型的别名分析元数据为 LLVM 优化器提供了额外的信息，并有助于某些优化以生成更好的机器代码。用户总是欣赏使用源级调试器的可能性，通过向 IR 代码添加调试信息，你可以实现编译器的重要功能。

优化 IR 代码是 LLVM 的核心任务。在下一章中，我们将学习如何工作以及我们如何可以影响由它管理的优化管道。
