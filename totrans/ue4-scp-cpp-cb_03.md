# 内存管理、智能指针和调试

在本章中，我们将介绍以下主题：

+   未管理内存 – 使用 malloc()/free()

+   未管理内存 – 使用 new/delete

+   管理内存 – 使用 NewObject< > 和 ConstructObject< >

+   管理内存 – 释放内存

+   管理内存 – 使用智能指针（TSharedPtr, TWeakPtr, TAutoPtr）来跟踪对象

+   使用 TScopedPointer 来跟踪对象

+   Unreal 的垃圾回收系统和 UPROPERTY()

+   强制垃圾回收

+   断点和单步执行代码

+   寻找错误和使用调用栈

+   使用性能分析器来识别热点

# 简介

内存管理始终是确保您的计算机程序稳定性和代码良好、无错误运行的最重要的事情之一。悬垂指针（*指向已被从内存中移除的指针*）是如果发生则难以追踪的故障示例。

在任何计算机程序中，内存管理都极其重要。UE4 的 `UObject` 引用计数系统是管理从 `UObject` 类派生的演员和类的默认方式。这是您在 UE4 程序中管理内存的默认方式。

如果您编写自己的自定义 C++ 类，这些类不派生自 `UObject`，您可能会发现 `TSharedPtr` / `TWeakPtr` 引用计数类很有用。这些类提供了对象引用计数和自动删除，当它们没有更多引用时。

本章提供了在 UE4 中进行内存管理的配方。它还提供了有关通过 Visual Studio 为我们提供的一些有用功能来调试代码的信息，包括断点和性能分析器。

# 技术要求

本章需要使用 Unreal Engine 4，并使用 Visual Studio 2017 作为 IDE。有关如何安装这两款软件及其要求的信息，请参阅本书的第一章，*UE4 开发工具*。

# 未管理内存 – 使用 malloc( ) / free( )

在 C 语言（在 C++ 中仍然可以使用）中为您的计算机程序分配内存的基本方法是使用 `malloc()` 函数。此函数为您的程序指定计算机系统内存中的一块。一旦您的程序正在使用内存的一个段，则没有其他程序可以使用或访问该内存段。尝试访问未分配给您的程序的内存段将生成一个 **段错误**，并在大多数系统中表示非法操作。

# 如何做...

让我们看看一些示例代码，它分配一个指针变量 `i`，然后使用 `malloc()` 为它分配内存。我们在 `int*` 指针后面分配一个整数。分配后，我们使用解引用运算符 `*` 在 `int` 中存储一个值：

```cpp
// CREATING AND ALLOCATING MEMORY FOR AN INT VARIABLE i 

// Declare a pointer variable i 
int * i; 

// Allocates system memory
i = ( int* )malloc( sizeof( int ) );

// Assign the value 0 into variable i 
*i = 0; 

// Use the variable i, ensuring to 
// use dereferencing operator * during use 
printf( "i contains %d", *i ); 

// RELEASING MEMORY OCCUPIED BY i TO THE SYSTEM 

// When we're done using i, we free the memory 
// allocated for it back to the system. 
free( i ); 

// Set the pointer's reference to address 0 
i = 0;
```

# 它是如何工作的...

以下代码执行的是后续图中所示的操作：

1.  第一行创建了一个`int *`指针变量`i`，它最初是一个悬空指针，指向一个可能对程序无效的内存段。

1.  在图示的第二步中，我们使用`malloc()`调用初始化变量`i`，使其指向一个恰好为`int`变量大小的内存段，这将使程序能够引用。

1.  然后，我们使用命令`*i = 0;`初始化该内存段的值为`0`。请参考以下图示：

![](img/00340e70-51a4-4f85-916c-9e0a483211bf.png)

注意指针变量赋值（`i =`）与指向指针变量所指向内存地址内内容的赋值（`*i =`）之间的区别，前者告诉指针要引用哪个内存地址，而后者则是指向指针变量所指向的内存地址内的内容。

当变量`i`中的内存需要释放回系统时，我们使用`free()`解分配调用来实现，如下面的图所示。`i`随后被分配给指向图中由**电气接地**符号引用的内存地址`0`：![](img/97df6b03-e489-4edc-b53e-53668ce93f7f.png)

![](img/aeff3a49-d8d6-43af-965c-fda71b7d0499.png)

我们将变量`i`设置为指向`NULL`引用的原因是为了清楚地表明变量`i`不指向有效的内存段。

# 未托管内存 – 使用 new/delete

`new`运算符几乎与`malloc`调用相同，不同之处在于它会在内存分配后立即调用对象的构造函数。使用`new`运算符分配的对象应该使用`delete`运算符（而不是`free()`）进行解分配。

# 准备工作

在 C++中，`malloc()`的使用已被`new`运算符作为最佳实践所取代。`malloc()`和`new`运算符功能的主要区别在于`new`会在内存分配后在对象类型上调用构造函数。请参考以下表格：

| `malloc` | 为使用分配一个连续的空间区域 |
| --- | --- |
| `new` | 为使用分配一个连续的空间区域调用构造函数，作为`new`运算符参数的对象类型。 |

# 如何做到这一点...

在以下代码中，我们声明一个简单的`Object`类，然后使用`new`运算符构造它的一个实例：

```cpp
class Object 
{ 
  Object() 
  { 
    puts( "Object constructed" ); 
  } 
  ~Object() 
  { 
    puts( "Object destructed" ); 
  } 
}; 

// Invokes constructor 
Object * object = new Object(); 

// Invokes deconstrctor 
delete object; 

// resets object to a null pointer
object = 0;  
```

# 它是如何工作的...

`new`运算符的工作原理与`malloc()`类似。如果与`new`运算符一起使用的类型是对象类型，则构造函数会自动使用`new`关键字调用，而使用`malloc()`时则不会调用构造函数。

# 还有更多...

应该避免使用带有`new`关键字（或者说是`malloc`）的裸堆分配。在引擎内部，首选使用托管内存，以便跟踪和清理所有内存使用。如果你分配了一个`UObject`派生类，你绝对需要使用`NewObject< >`或`ConstructObject< >`（在后续的菜谱中概述）。

# 托管内存 – 使用 NewObject< >和 ConstructObject< >

**托管内存**指的是由位于 `new`、`delete`、`malloc` 和 `free` 调用之上的某些程序子系统分配和释放的内存。这些子系统通常被创建，以便程序员在分配内存后不会忘记释放它。未释放、占用但未使用的内存块被称为**内存泄漏**，如下所示：

```cpp
// generates memory leaks galore! 
for( int i = 0; i < 100; i++ ) 
{
  int** leak = new int[500];
}
```

在前面的例子中，分配的内存不能被任何变量引用！因此，你既不能在 `for` 循环之后使用分配的内存，也不能释放它。如果你的程序分配了所有可用的系统内存，那么会发生的情况是，你的系统将完全耗尽内存，你的操作系统将标记你的程序，并因为它使用了过多的内存而关闭它。

内存管理可以防止忘记释放内存。在内存管理的程序中，通常通过动态分配的、引用对象指针数量的对象来记住。当没有指针引用对象时，它要么在垃圾收集器下一次运行时自动立即删除，要么被标记为删除。

在 UE4 中使用托管内存是自动的。任何对象的分配都必须使用 `NewObject< >()` 或 `SpawnActor< >()` 函数。

在引擎中使用时必须使用 `NewObject< >()` 或 `SpawnActor< >()` 函数。

对象的释放是通过删除对对象的引用，然后偶尔调用垃圾清理例程（在本章中进一步列出）来完成的。

# 准备工作

当你需要构造任何不是 `Actor` 类派生类的 `UObject` 派生类时，你应该始终使用 `NewObject< >` 函数。只有当对象是 `Actor` 或其派生类时，才应使用 `SpawnActor< >`。

# 如何做到这一点...

假设我们正在尝试构造一个 `UAction` 类型的对象，该对象本身是从 `UObject` 继承的——例如，以下类：

```cpp
UCLASS(BlueprintType, Blueprintable, 
       meta=(ShortTooltip="Base class for any Action type") )
class CHAPTER_03_API UAction : public UObject
{
  GENERATED_BODY()

public:
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category=Properties)
  FString Text;
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category=Properties)
  FKey ShortcutKey;

};
```

要构造 `UAction` 类的实例，我们会这样做：

```cpp
// Create an object
UAction * action = NewObject<UAction>(GetTransientPackage(), 
                                      UAction::StaticClass() 
                                      /* RF_* flags */ ); 
```

# 它是如何工作的...

在这里，`UAction::StaticClass()` 为 `UAction` 对象提供了一个基 `UClass *`。`NewObject< >` 的第一个参数是 `GetTransientPackage()`，它只是检索游戏的临时包。在 UE4 中，包（`UPackage`）只是一个数据集合。我们在这里使用**临时包**来存储我们的堆分配数据。你也可以使用来自 Blueprints 的 `UPROPERTY() TSubclassOf<AActor>` 来选择 `UClass` 实例。

第三个参数（可选）是参数的组合，它指示内存管理系统如何处理 `UObject`。

# 更多内容...

有另一个与 `NewObject< >` 非常相似的功能，称为 `ConstructObject< >`。`ConstructObject< >` 在构造时提供了更多的参数，如果你需要初始化某些属性，可能会发现它很有用。否则，`NewObject` 就足够好了。

你可以在这里了解更多关于 ConstructObject 函数的信息：[`docs.unrealengine.com/en-us/Programming/UnrealArchitecture/Objects/Creation#constructobject`](https://docs.unrealengine.com/en-us/Programming/UnrealArchitecture/Objects/Creation#constructobject)。

# 参见

你可能还想查看 [`docs.unrealengine.com/latest/INT/Programming/UnrealArchitecture/Objects/Creation/index.html#objectflags`](https://docs.unrealengine.com/latest/INT/Programming/UnrealArchitecture/Objects/Creation/index.html#objectflags) 上的 `RF_*` 标志的文档。

# 管理内存 - 释放内存

当没有更多引用指向 `UObject` 实例时，`UObject` 实例会被引用计数和垃圾回收。使用 `ConstructObject<>` 或 `NewObject< >` 在 `UObject` 类派生上分配的内存也可以通过调用 `UObject::ConditionalBeginDestroy()` 成员函数手动（在引用计数降至 0 之前）释放。

# 准备工作

只有当你确定你不再需要在内存中保留 `UObject` 或 `UObject` 类派生实例时，你才会这样做。使用 `ConditionalBeginDestroy()` 函数来释放内存。

# 如何操作...

以下代码演示了 `UObject` 类实例的释放：

```cpp
UObject *o = NewObject< UObject >( ... ); 
o->ConditionalBeginDestroy(); 
```

这个概念也适用于从 `UObject` 类派生的任何类。例如，如果我们想在上一道菜谱中创建的 `UAction` 对象上做这件事，我们会在以下片段中添加粗体文本：

```cpp
// Create an object
UAction * action = NewObject<UAction>(GetTransientPackage(), 
                    UAction::StaticClass() 
                    /* RF_* flags */ ); 

// Destroy an object
action->ConditionalBeginDestroy();
```

# 作用原理...

`ConditionalBeginDestroy()` 命令开始释放过程，调用可重写的 `BeginDestroy()` 和 `FinishDestroy()` 函数。

请注意不要在仍被其他对象的指针引用的任何对象上调用 `UObject::ConditionalBeginDestroy()`。

# 管理内存 - 使用智能指针（TSharedPtr、TWeakPtr、TAutoPtr）跟踪对象

当人们担心他们会忘记他们为标准 C++ 对象创建的 `delete` 调用时，他们通常会使用智能指针来防止内存泄漏。`TSharedPtr` 是一个非常有用的 C++ 类，它将使任何自定义 C++ 对象具有引用计数功能——除了 `UObject` 派生类，它们已经具有引用计数。还提供了一个替代类 `TWeakPtr`，用于指向具有无法阻止删除（因此称为 *弱*）奇怪属性的引用计数对象：

![图片](img/66139d96-b541-46d5-a16e-093e46d853fd.png)

`UObject` 及其派生类（使用 `NewObject` 或 `ConstructObject` 创建的任何内容）不能使用 `TSharedPtr`！

# 准备工作

如果你不想在你的 C++代码中使用原始指针并手动跟踪删除（不使用`UObject`派生类），那么这段代码是使用智能指针（如`TSharedPtr`、`TSharedRef`等）的好候选。当你使用动态分配的对象（使用`new`关键字创建）时，你可以将其包装在一个引用计数指针中，以便自动释放。不同类型的智能指针决定了智能指针的行为和删除调用时间。它们如下：

+   `TSharedPtr`：一个线程安全的（只要你将`ESPMode::ThreadSafe`作为模板的第二个参数提供）引用计数指针类型，表示一个共享对象。当没有更多引用指向它时，共享对象将被释放。

+   `TAutoPtr`：一个非线程安全的共享指针。

# 如何实现...

我们可以使用一个简短的代码段来演示之前提到的四种智能指针的使用。在这段代码中，起始指针可以是原始指针，也可以是另一个智能指针的副本。你所要做的就是将 C++原始指针包装在任何以下构造函数调用中：`TSharedPtr`、`TSharedRef`、`TWeakPtr`或`TAutoPtr`。

例如，看看以下代码片段：

```cpp
// C++ Class NOT deriving from UObject 
class MyClass { }; 
TSharedPtr<MyClass>sharedPtr( new MyClass() ); 
```

# 它是如何工作的...

弱指针和共享指针之间有一些区别。弱指针没有在引用计数降到 0 时保持对象在内存中的能力。

使用弱指针（相对于原始指针）的优势在于，当弱指针下的对象被手动删除（使用`ConditionalBeginDestroy()`）时，弱指针的引用变为`NULL`引用。这使你可以通过检查以下形式的语句来检查指针下的资源是否仍然被正确分配：

```cpp
if( ptr.IsValid() ) // Check to see if the pointer is valid 
{ 
} 
```

# 更多内容...

共享指针是线程安全的。这意味着可以在不同的线程上安全地操作底层对象。

总是记住，你不能在`UObject`或`UObject`派生类中使用`TSharedRef`；你只能在你自定义的 C++类中使用它们。你的`FStructures`可以使用`TSharedPtr`、`TSharedRef`和`TWeakPtr`类来包装原始指针。

如果你想要使用智能指针指向一个对象，你必须使用`TWeakObjectPointer`或`UPROPERTY()`。

如果你不需要`TSharedPtr`的线程安全保证，可以使用`TAutoPtr`。`TAutoPtr`将在引用数降到 0 时自动删除对象。

如果你想了解更多关于 Unreal 智能指针的信息，请查看[`docs.unrealengine.com/en-us/Programming/UnrealArchitecture/SmartPointerLibrary`](https://docs.unrealengine.com/en-us/Programming/UnrealArchitecture/SmartPointerLibrary)。

# 使用 TScopedPointer 跟踪对象

**作用域**指针是一种在声明它的代码块结束时自动删除的指针。回想一下，作用域只是变量存在的一段代码。作用域将持续到第一个出现的闭合括号 `}`。

例如，在以下代码块中，我们有两个作用域。外部作用域声明了一个整数变量 `x`（在整个外部块中有效），而内部作用域声明了一个整数变量 `y`（在声明它的行之后的内部块中有效）：

```cpp
{ 
  int x; 
  { 
    int y; 
  } // scope of y ends 
} // scope of x ends 
```

# 准备工作

当需要保留引用计数的对象（即有越界风险的）在使用的整个过程中保持存在时，作用域指针非常有用。

# 如何做...

要声明作用域指针，我们只需使用以下语法：

```cpp
TScopedPointer<AWarrior> warrior(this ); 
```

这声明了一个引用类型为尖括号内声明的类型的范围指针：`< AWarrior >`。

# 它是如何工作的...

`TScopedPointer` 变量类型会自动为指向的变量添加引用计数。这可以防止在作用域指针的生命周期内对底层对象进行解分配。

# Unreal 的垃圾回收系统和 UPROPERTY( )

当您有一个对象（如 `TArray< >`）作为 `UCLASS()` 的 `UPROPERTY()` 成员时，您需要将该成员声明为 `UPROPERTY()`（即使您不会在蓝图中进行编辑）；否则，`TArray` 将无法正确分配。

# 如何做...

假设我们有一个如下所示的 `UCLASS()` 宏：

```cpp
UCLASS() 
class MYPROJECT_API AWarrior : public AActor 
{ 
  //TArray< FSoundEffect > Greets; // Incorrect 
  UPROPERTY() TArray< FSoundEffect > Greets; // Correct 
}; 
```

您必须将 `TArray` 成员列为 `UPROPERTY()` 才能正确地进行引用计数。如果您不这样做，您将在代码中遇到意外的内存错误类型错误。

# 它是如何工作的...

`UPROPERTY()` 声明告诉 UE4 `TArray` 必须正确地管理内存。没有 `UPROPERTY()` 声明，您的 `TArray` 将无法正常工作。

# 强制垃圾回收

当内存填满，您想要释放其中一些时，可以强制进行垃圾回收。您很少需要这样做，但在您有一个非常大的纹理（或纹理集）需要清除，并且这些纹理是引用计数的这种情况下，您可以这样做。

# 准备工作

您只需在所有想要从内存中解分配的 `UObject` 上调用 `ConditionalBeginDestroy()`，或将它们的引用计数设置为 `0`。

# 如何做...

通过调用以下方法执行垃圾回收：

```cpp
GetWorld()->ForceGarbageCollection( true ); 
```

# 断点和逐行执行代码

**断点**是您暂停 C++ 程序以暂时停止代码运行的方法，并有机会分析和检查程序的操作。您可以查看变量、逐行执行代码，并更改变量值。

# 准备工作

在 Visual Studio 中设置断点很容易。您只需在想要操作暂停的代码行上按 *F9* 键，或者单击想要操作暂停的代码行左侧的灰色边缘。当操作达到指示的行时，代码将暂停。

# 如何做...

1.  在你想暂停执行的那一行按 *F9*。这将向代码添加一个断点，由红色圆点表示，如下所示。单击红色圆点可以切换它：

![图片](img/491cbfc5-5a48-4baf-9391-1023a104c953.png)

1.  将构建配置设置为任何标题中包含 Debug 的配置（DebugGame 编辑器或如果没有使用编辑器启动，则为简单 DebugGame）：

![图片](img/7fb56b04-3359-4b04-bf33-f48dfb485aaa.png)

1.  通过按 *F5*（不按 *Ctrl*）或选择调试 | 开始调试菜单选项来启动你的代码。

1.  当代码到达红色圆点时，代码的执行将暂停。

1.  暂停视图将带您进入**调试模式**下的代码编辑器。在此模式下，窗口可能会重新排列，解决方案资源管理器可能移动到右侧，底部可能出现新的窗口，包括局部变量、监视 1 和调用栈。如果这些窗口没有出现，请在调试 | 窗口子菜单下查找它们。

1.  在局部变量窗口（调试 | 窗口 | 局部变量）下检查你的变量。

1.  按 *F10* 跳过一行代码。

1.  按 *F11* 进入一行代码。

# 它是如何工作的...

调试器是强大的工具，允许你在代码运行时查看有关代码的任何信息，包括变量状态。

跳过一行代码（*F10*）将执行该行代码的整个内容，然后立即在下一行暂停程序。如果该行代码是一个函数调用，那么函数将在函数调用的第一行代码处不暂停执行，如下所示：

```cpp
void f() 
{ 
  // F11 pauses here 
  UE_LOG( LogTemp, Warning, TEXT( "Log message" ) ); 
} 
int main() 
{ 
  f(); // Breakpoint here: F10 runs and skips to next line 
} 
```

跳入一行代码（*F11*）将在运行的下一行代码处暂停执行。

# 寻找错误和使用调用栈

当你的代码中存在导致崩溃的错误、抛出异常等情况时，Visual Studio 将尝试停止代码执行，并允许你检查代码。Visual Studio 停止的地方不一定是错误的精确位置，但它可以非常接近。至少它会在一行代码上，该代码无法正确执行。

# 准备工作

在这个菜谱中，我们将描述**调用栈**以及如何追踪你认为错误可能来自的地方。尝试向你的代码中添加一个错误，或者添加一个你想要暂停检查的有趣位置的断点。

# 如何做到这一点...

1.  通过按 *F5* 或选择调试 | 开始调试菜单选项将代码运行到发生错误的位置。例如，添加以下代码行：

```cpp
UObject *o = 0; // Initialize to an illegal null pointer 
o->GetName(); // Try and get the name of the object (has 
 bug) 
```

1.  代码将在第二行（`o->GetName()`）暂停。

注意，此代码仅在游戏中在编辑器中播放时才会执行（并因此崩溃）。

1.  当代码暂停时，导航到调用栈窗口（调试 | 窗口 | 调用栈）。

# 它是如何工作的...

调用栈是已执行函数调用的列表。当发生错误时，它发生的行将列在调用栈的顶部。参考以下截图：

![图片](img/84f61f30-4a18-4bea-b9fc-5472d2ad68fd.png)

# 使用性能分析器识别热点

C++ 分析器对于查找需要大量处理时间的代码部分非常有用。使用分析器可以帮助您在优化过程中关注代码的特定部分。如果您怀疑某个代码区域运行缓慢，那么实际上如果它在分析器中没有高亮显示，您就可以确认它并不慢。

# 如何操作...

1.  前往调试 | 性能分析器...:

![](img/69e22bc5-0985-4574-ab72-93fee172f5af.png)

1.  在前面截图所示的对话框中，选择您想要显示的分析类型。您可以选择分析 CPU 使用率、GPU 使用率、内存使用率，或者通过性能向导逐步选择您想要查看的内容。

1.  确保在没有编辑器的情况下运行游戏，然后点击对话框底部的开始按钮。

1.  在短时间内（不到一分钟或两分钟）停止代码以停止样本收集。

不要收集过多的样本或分析器，因为那样启动将需要非常长的时间。

1.  检查出现在 `.diagsession` 文件中的结果。务必浏览

    所有的可用选项卡。可用的选项卡将根据执行的分析类型而有所不同。

# 它是如何工作的...

C++ 分析器对运行中的代码进行采样和分析，并向您展示一系列关于代码执行情况的图表和图像。

您可以通过访问[`docs.microsoft.com/en-us/visualstudio/profiling/?view=vs-2017`](https://docs.microsoft.com/en-us/visualstudio/profiling/?view=vs-2017)来获取更多关于性能分析器的信息。
