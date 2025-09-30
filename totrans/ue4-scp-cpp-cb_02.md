# 创建类

本章将涵盖以下食谱：

+   创建 UCLASS – 从 UObject 派生

+   从您的自定义 UCLASS 创建蓝图

+   创建一个可由用户编辑的 UPROPERTY

+   从蓝图访问 UPROPERTY

+   将 UCLASS 指定为 UPROPERTY 的类型

+   实例化从 UObject 派生的类（ConstructObject<> 和 NewObject <>）

+   销毁从 UObject 派生的类

+   创建一个 USTRUCT

+   创建一个 UENUM()

# 简介

本章重点介绍如何创建与 UE4 蓝图编辑器良好集成的 C++ 类和结构体。

我们将在本章中创建的类是常规 C++ 类的毕业版本，并称为 `UCLASS`。

`UCLASS` 只是一个带有大量 UE4 宏装饰的 C++ 类。这些宏生成额外的 C++ 头文件代码，使它与 UE4 编辑器本身集成。

使用 `UCLASS` 是一个很好的实践。如果配置正确，`UCLASS` 宏可以使您的 `UCLASS` 可蓝图化，这可以使您的自定义 C++ 对象能够在 Unreal 的视觉脚本语言蓝图中使用。如果您团队中有设计师，这会非常有用，因为他们可以访问和调整项目的一些方面，而无需深入代码。

我们可以有蓝图的可视化可编辑属性（`UPROPERTY`），例如文本字段、滑块和模型选择框等便捷的 UI 小部件。您还可以有函数（如 `UFUNCTION`），这些函数可以从蓝图图中调用。这两个都在以下屏幕截图中显示：

![图片](img/48b7d5fd-8b17-41f9-b7fd-2d363741cd9e.jpg)

在左侧，两个带有 `UPROPERTY` 装饰的类成员（一个 `UTexture` 引用和一个 `FColor`）在 C++ 类的蓝图中显示出来以供编辑。在右侧，一个标记为 `BlueprintCallable UFUNCTION` 的 C++ `GetName` 函数在蓝图图中显示为可调用的。

由 `UCLASS` 宏生成的代码将位于一个 `ClassName.generated.h` 文件中，该文件将是您的 `UCLASS` 头文件 `ClassName.h` 中所需的最后一个 `#include`。

您将注意到，我们在这个类中创建的示例对象，即使当它们是可蓝图化的，也不会放置在级别中。这是因为，为了放置在级别中，您的 C++ 类必须从 `Actor` 基类派生，或者从其派生的子类。有关更多详细信息，请参阅第四章，*演员和组件*。

一旦您了解了模式，UE4 代码通常很容易编写和管理。我们编写的代码，无论是从另一个 `UCLASS` 派生，还是创建一个 `UPROPERTY` 或 `UFUNCTION` 实例，都是非常一致的。本章提供了围绕基本 `UCLASS` 派生、属性和引用声明、构造、销毁和一般功能等常见 UE4 编码任务的食谱。

# 技术要求

本章需要使用 Unreal Engine 4，并使用 Visual Studio 2017 作为 IDE。有关如何安装这两款软件及其要求的说明，请参阅第一章，*UE4 开发工具*。

# 创建 UCLASS – 从 UObject 派生

当用 C++ 编码时，你可以有自己的代码，这些代码可以作为本地 C++ 代码编译和运行，并适当地调用 `new` 和 `delete` 操作符来创建和销毁你的自定义对象。只要你的 `new` 和 `delete` 调用适当配对，确保代码中没有内存泄漏，本地 C++ 代码在 UE4 项目中是完全可接受的。

然而，你也可以通过使用 `UCLASS` 宏来声明自定义 C++ 类，这些类的行为类似于 UE4 类，通过使用 `UCLASS` 宏声明你的自定义 C++ 对象。`UCLASS` 宏告诉类使用 UE4 的智能指针和内存管理例程进行分配和释放，根据它们的智能指针规则，可以由 UE4 编辑器自动加载和读取，并且可以选择从蓝图访问。

注意，当你使用 `UCLASS` 宏时，你的 `UCLASS` 对象的创建和销毁必须完全由 UE4 管理：你必须使用 `ConstructObject` 函数来创建你的对象实例（而不是 C++ 本地关键字 `new`），并调用 `UObject::ConditionalBeginDestroy()` 函数来销毁对象（而不是 C++ 本地关键字 `delete`）。

# 准备工作

在本食谱中，我们将概述如何编写一个使用 `UCLASS` 宏来启用托管内存分配和释放的 C++ 类，以及允许从 UE4 编辑器和蓝图进行访问。要完成此食谱，你需要一个可以添加新代码的 UE4 项目。

# 如何做到这一点...

要创建自己的 `UObject` 派生类，请按照以下步骤操作：

1.  在 UE4 编辑器中，从你的运行项目中选择文件 | 新 C++ 类。

1.  在出现的添加 C++ 类对话框中，转到窗口的右上角并勾选显示所有类复选框：

![图片](img/4dc61190-34b3-4777-a3b1-dbd680a1c64b.png)

1.  选择 `Object`（层次结构的顶部）作为要继承的父类，然后点击下一步。

注意，尽管在对话框中将写入 `Object`，但在你的 C++ 代码中，你将派生的 C++ 类实际上是带有首字母大写的 `U` 的 `UObject`。这是 UE4 的命名约定。

从 `Actor` 以外的分支派生的 `UCLASS` 必须以首字母 `U` 命名。

从 `Actor` 派生的 `UCLASS` 必须以首字母 `A` 命名（第四章，*演员和组件*）。

不是 `UCLASS` 的 C++ 类（从无派生）没有命名约定，但如果愿意，可以用首字母 `F` 命名（例如，`FAssetData`）。

`UObject` 的直接派生类将不会是层级可放置的，即使它们包含视觉表示元素，如 `UStaticMeshes`。如果你想在 UE4 级别内放置你的对象，你必须至少从 `Actor` 类派生，或者在继承层次结构中位于其下方。有关如何从 `Actor` 类派生以创建层级可放置对象的更多信息，请参阅第四章，*演员和组件*。

本章的示例代码在级别中不可放置，但你可以在 UE4 编辑器中创建和使用基于我们本章中编写的 C++ 类的蓝图。

1.  为你创建的新 `Object` 派生类命名时，应选择适合你创建的对象类型的名称。我将我的命名为 `UserProfile`：

![图片](img/17828513-6e03-4510-8764-731dddf869c3.png)

在 C++ 文件中类的命名中，它表现为 `UUserObject`，以确保遵循 UE4 的约定（在 C++ 中，带有 `UCLASS` 的类名以首字母 `U` 开头）。

1.  点击创建类，文件应该在文件编译完成后创建。之后，Visual Studio 应该打开（否则，通过转到文件 | 打开 Visual Studio 来打开解决方案），并将打开我们刚刚创建的类的 `.cpp` 文件（`UserProfile.cpp`）。打开你的类的头文件规则（`UserProfile.h`），确保你的类文件具有以下形式：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "UserProfile.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_02_API UUserProfile : public UObject
{
  GENERATED_BODY()

};

```

1.  编译并运行你的项目。现在你可以在 Visual Studio 以及 UE4 编辑器中使用你自定义的 `UCLASS` 对象。有关更多详细信息，请参阅以下菜谱，了解你可以用它做什么。

如何创建和销毁你的 `UObject` 派生类将在本章后面的 *实例化 UObject 派生类（ConstructObject <> 和 NewObject <>）* 和 *销毁 UObject 派生类* 菜谱中概述。

# 它是如何工作的...

UE4 为你的自定义 `UCLASS` 生成并管理大量的代码。这些代码是使用 UE4 宏（如 `UPROPERTY`、`UFUNCTION` 和 `UCLASS` 宏本身）的结果。生成的代码放入 `UserProfile.generated.h`。你必须使用 `UCLASSNAME.h` 文件包含 `UCLASSNAME.generated.h` 文件以成功编译，这就是为什么默认情况下编辑器会自动包含它。如果不包含 `UCLASSNAME.generated.h` 文件，编译将失败。

还需要注意的是，`UCLASSNAME.generated.h` 文件必须作为 `UCLASSNAME.h` 文件中 `#include` 列表中的最后一个 `#include` 包含。

这里是一个正确的示例：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"

#include <list> // Newly added include

// CORRECT: generated file is the last file included
#include "UserProfile.generated.h"
```

这里是一个错误的示例：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "UserProfile.generated.h"

// WRONG: NO INCLUDES AFTER .generated.h FILE
#include <list> // Newly added include
```

如果 `UCLASSNAME.generated.h` 文件不是在前面代码示例中显示的 `#include` 语句列表中的最后一个项目，你将得到以下错误：

```cpp
>> #include found after .generated.h file - the .generated.h file 
 should always be the last #include in a header
```

# 还有更多...

在这里，我们想要讨论一些关键字，它们会修改 `UCLASS` 的行为方式。`UCLASS` 可以按以下方式标记：

+   `Blueprintable`：这意味着您希望能够在 UE4 编辑器中的类查看器内构建蓝图（当您右键单击它时，创建蓝图类...变为可用）。如果没有`Blueprintable`关键字，即使您可以在类查看器中找到它并右键单击它，创建蓝图类...选项也不会对您的`UCLASS`可用：

![图片](img/2c569908-51ef-4c1f-b399-4f45af55f714.jpg)

+   仅当您在`UCLASS`宏定义中指定`Blueprintable`时，创建蓝图类...选项才可用。

+   `BlueprintType`：使用此关键字意味着`UCLASS`可以用作来自另一个蓝图中的变量。您可以从任何蓝图的事件图左侧面板的变量组中创建蓝图变量。

+   `NotBlueprintType`：使用此关键字指定您不能将此蓝图变量类型用作蓝图图中的变量。在类查看器中右键单击`UCLASS`名称时，其上下文菜单中不会显示创建蓝图类...：

![图片](img/816ad77a-791c-42fc-b5dd-f517e92db546.jpg)

您可能不确定是否应该将您的 C++类声明为`UCLASS`。一般规则是除非您有充分的理由不这样做，否则请使用`UCLASS`。到这一点为止，Unreal Engine 4 的代码已经写得非常好，并且已经经过彻底的测试。如果您喜欢智能指针，您可能会发现`UCLASS`不仅使代码更安全，而且使整个代码库更加连贯和一致。

# 相关内容

+   要向蓝图图中添加额外的可编程`UPROPERTY`，请参阅*创建用户可编辑的 UPROPERTY*配方

+   有关使用适当的智能指针引用您的`UCLASS`实例的详细信息，请参阅第三章，*内存管理、智能指针和调试*

+   有关`UCLASS`、`UPROPERTY`以及所有其他类似宏的更多信息以及它们如何在 UE4 中使用的详细信息，请查看[`www.unrealengine.com/en-US/blog/unreal-property-system-reflection`](https://www.unrealengine.com/en-US/blog/unreal-property-system-reflection)

# 从您的自定义 UCLASS 创建蓝图

蓝图化只是为您的 C++对象推导出蓝图类的过程。从 UE4 对象创建派生自蓝图类允许您在编辑器中直观地编辑自定义`UPROPERTY`。这避免了将任何资源硬编码到您的 C++代码中。此外，要使您的 C++类能够在级别中放置，它必须首先进行蓝图化。但这仅当蓝图背后的 C++类是`Actor`类派生时才可能。

有一种方法可以使用`FStringAssetReferences`和`StaticLoadObject`加载资源（如纹理）。然而，这些加载资源的方法（通过将路径字符串硬编码到您的 C++代码中）通常是不推荐的。在`UPROPERTY()`中提供一个可编辑的值，并从适当的具体类型资产引用中加载是一种更好的做法。

# 准备工作

你需要有一个构造的`UCLASS`，你想要从中派生一个`Blueprint`类（参见本章前面的*制作 UCLASS – 从 UObject 派生*配方），才能遵循此配方。你还必须在`UCLASS`宏中将你的`UCLASS`标记为`Blueprintable`，以便在引擎内部进行蓝图化。

# 如何做...

1.  要蓝图化你的`UserProfile`类，首先确保`UCLASS`在`UCLASS`宏中有`Blueprintable`标签。它应该看起来像这样：

```cpp
UCLASS( Blueprintable ) 
class CHAPTER2_API UUserProfile : public UObject 
```

1.  编译你的代码。

1.  在类查看器中找到`UserProfile`C++类（窗口 | 开发者工具 | 类查看器）。由于之前创建的`UCLASS`没有从`Actor`派生，为了找到你的自定义`UCLASS`，你必须关闭类查看器中的过滤器 | 仅演员（默认情况下是勾选的）：

![图片](img/1afbf2e4-e1af-4d41-8e0b-a0947e5e6ba6.jpg)

如果你没有这样做，那么你的自定义 C++类可能不会显示！

请记住，你可以使用类查看器内部的小搜索框，通过开始输入来轻松找到`UserProfile`类：

![图片](img/eb3b6c95-62c1-4582-b2fe-2d367fc21356.jpg)

1.  在类查看器中找到你的`UserProfile`类，右键单击它，通过选择创建蓝图...来从它创建蓝图。

1.  命名你的蓝图。有些人喜欢在蓝图类名前加上`BP_`前缀。

    你可以选择遵循此约定；只需确保保持一致。

1.  你将能够编辑为每个创建的`UserProfile`蓝图实例创建的任何字段。

如果蓝图编辑器没有自动打开，你可以通过在内容浏览器中双击文件来打开它。

# 它是如何工作的...

你创建的任何具有`UCLASS`宏中`Blueprintable`标签的 C++类都可以在 UE4 编辑器中进行蓝图化。蓝图允许你在 UE4 的可视 GUI 界面中自定义 C++类的属性。

# 创建一个用户可编辑的 UPROPERTY

你声明的每个`UCLASS`都可以有任意数量的`UPROPERTY`声明，每个`UPROPERTY`都可以是一个可视编辑字段，或者是一个`UCLASS`的可访问数据成员。

我们可以为每个`UPROPERTY`添加许多限定符，这些限定符会改变它在 UE4 编辑器中的行为，例如`EditAnywhere`（指定`UPROPERTY`可以通过代码或编辑器进行更改），以及`BlueprintReadWrite`（指定蓝图可以在任何时间读取和写入变量，除了允许 C++代码这样做之外）。

# 准备工作

要使用此配方，你应该有一个可以添加 C++代码的 C++项目。此外，你应该已经完成了前面的配方，*制作 UCLASS – 从 UObject 派生*。

# 如何做...

1.  首先，我们需要将类标记为`Blueprintable`，然后向你的`UCLASS`声明中添加以下成员，这些成员以粗体显示：

```cpp
/**
 * UCLASS macro options sets this C++ class to be
 * Blueprintable within the UE4 Editor
 */
UCLASS( Blueprintable )
class CHAPTER_02_API UUserProfile : public UObject
{
  GENERATED_BODY()

public:
 UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Stats)
 float Armor;

 UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Stats)
 float HpMax;
};
```

1.  返回 Unreal 编辑器，然后点击编译按钮以更新我们的代码。

1.  一旦更新，如果尚未创建，请创建您的`UObject`类派生版的蓝图。

这可以通过我们之前看到的方式完成，但也可以手动完成，我们现在就是这样做的。

1.  要这样做，请转到内容浏览器选项卡并单击文件夹图标以选择您想要工作的项目部分。从弹出的窗口中，选择内容部分：

![图片](img/cd4114e3-97a9-42b7-abb8-562944beaa03.png)

在内容浏览器中选择内容文件夹

1.  从那里，选择添加新按钮，然后选择蓝图类：

![图片](img/c79a081c-e525-4674-84b1-070f83273e06.png)

从内容浏览器创建蓝图类

1.  从选择父类菜单中，您将看到一些用于常见类的按钮。下面，您将看到所有类选项，有一个箭头可以点击以展开它。从那里，输入您类的名称（在我们的例子中，是`UserProfile`），然后从列表中选择它。之后，单击选择按钮：

![图片](img/818ac6d6-2da2-4908-bea0-7f890a911271.png)

1.  从那里，您将在“内容浏览器”中看到项目出现，您可以将实例重命名为您想要的任何名称；我将其命名为`MyProfile`。

1.  创建后，我们可以通过双击它来在 UE4 编辑器中打开蓝图。

1.  您现在可以为这些新的`UPROPERTY`字段的默认值指定值：

![图片](img/519b9463-1cdd-469b-b1d5-2e8039763b7d.png)

由于蓝图是空的，它可能以仅包含数据而不包含中间和左侧部分的蓝图打开。要查看完整的蓝图菜单，您可能需要点击菜单顶部的“打开完整蓝图编辑器”以使屏幕看起来像之前的截图。然而，变量仍然应该是可见和可修改的。

1.  通过创建蓝图的新实例并编辑放置的对象上的值（通过双击它们）来指定每个实例的值。

# 它是如何工作的...

传递给`UPROPERTY()`宏的参数指定了有关变量的几个重要信息。在先前的示例中，我们指定了以下内容：

+   `EditAnywhere`：这意味着可以从蓝图直接编辑属性，或者在每个放置的`UClass`对象实例上编辑。

    在游戏级别中。与以下内容进行对比：

+   `EditDefaultsOnly`：蓝图值是可编辑的，但不能按实例编辑。

+   `EditInstanceOnly`：这将允许编辑`UClass`对象在游戏级别实例中的属性，而不是在基础蓝图本身上。

+   `BlueprintReadWrite`：这表示属性既可以从蓝图图中读取也可以写入。带有`BlueprintReadWrite`的`UPROPERTY()`必须是公共成员；否则，编译将失败。与以下内容进行对比：

+   `BlueprintReadOnly`：属性必须从 C++设置，并且不能从蓝图更改。

+   `类别`: 你应该始终为你的 `UPROPERTY()` 指定一个 `类别`，因为保持组织有序是一个好习惯。`类别` 决定了 `UPROPERTY()` 将在属性编辑器下的哪个子菜单中显示。所有在 `Category=Stats` 下指定的 `UPROPERTY()` 都将在蓝图编辑器中的相同 `Stats` 区域中显示。如果没有指定类别，`UPROPERTY` 将显示在默认类别 `UserProfile`（或 whatever one called their class）下。

# 还有更多...

理解整个过程很重要，这就是为什么我们在这里详细说明了所有内容，但你也可以通过在内容浏览器的 C++ 类部分右键单击类并选择基于 UserProfile 创建蓝图类来从脚本创建蓝图类。参见图表：

![图片](img/244a29c5-a663-4594-999e-33369eb41c2c.png)

# 参见

+   完整的 `UPROPERTY` 列表位于 [`docs.unrealengine.com/latest/INT/Programming/UnrealArchitecture/Reference/Properties/Specifiers/index.html`](https://docs.unrealengine.com/latest/INT/Programming/UnrealArchitecture/Reference/Properties/Specifiers/index.html)。浏览一下。

# 从蓝图访问 UPROPERTY

从蓝图访问 `UPROPERTY` 相对简单。成员必须作为 `UPROPERTY` 在你想要从蓝图图示中访问的成员变量上暴露。你必须在你宏声明中指定 `UPROPERTY` 为 `BlueprintReadOnly` 或 `BlueprintReadWrite` 以指定你想要变量是否只从蓝图可读（或甚至可写）。

你也可以使用特殊值 `BlueprintDefaultsOnly` 来表示你只想从蓝图编辑器编辑默认值（在游戏开始之前）。`BlueprintDefaultsOnly` 表示数据成员在运行时不能从蓝图编辑。

# 如何操作...

1.  创建一个 `UObject` 派生类，指定 `Blueprintable` 和 `BlueprintType`，如下面的代码所示，使用我们之前创建的相同类：

```cpp
/**
 * UCLASS macro options sets this C++ class to be
 * Blueprintable within the UE4 Editor
 */
UCLASS(Blueprintable, BlueprintType)
class CHAPTER_02_API UUserProfile : public UObject
{
  GENERATED_BODY()

public:
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Stats)
  float Armor;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Stats)
  float HpMax;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Stats)
 FString Name;
};
```

在 `UCLASS` 宏中的 `BlueprintType` 声明是必需的，以便在蓝图图中使用 `UCLASS` 作为类型。

1.  保存并编译你的代码。

1.  在 UE4 编辑器中，如果需要，从 C++ 类派生蓝图类，如前一个食谱或 *从自定义 UCLASS 创建蓝图* 食谱中所示。

1.  双击你的实例并更改名称变量以获得新值，例如，`Billy`。之后，点击编译按钮以保存所有更改：

![图片](img/1472fe26-f9e5-48df-8842-a2e18cc4e7a9.png)

1.  在允许函数调用（如可通过蓝图 | 打开级别蓝图访问的 Level Blueprint）的蓝图图中，我们现在可以尝试使用我们添加的变量。也许我们可以尝试在游戏开始时打印名称属性。

1.  要在游戏开始时发生某些事情，我们需要创建一个 BeginPlay 事件。您可以通过在蓝图图中右键单击并选择“添加事件 | 事件 BeginPlay”来完成此操作：

![图片](img/48f3b87f-6a3d-4555-8329-bf4efb26de61.png)

现在，需要创建类的实例。由于它继承自 `UObject`，我们不能通过拖放来实例化它，但我们可以通过从类蓝图节点创建“构造对象”来创建一个实例。

1.  右键单击刚刚创建的节点右侧，并在搜索栏中输入`construct`，然后从列表中选择“从类创建对象”节点：

![图片](img/8923bed8-7698-4ffa-a1cd-64ab300272ae.png)

1.  接下来，通过将 Event BeginPlay 节点右下角的箭头拖放到 Construct 节点的左侧箭头，并将箭头释放到 Construct 节点的左侧来连接从 Event BeginPlay 节点右侧的线。

在蓝图图中导航图示非常简单。右键单击并拖动以平移蓝图图示，*Alt* + 右键单击 + 拖动，或使用鼠标滚轮进行缩放。您可以左键单击并拖动任何节点以将其定位到您想要的位置。您还可以同时选择多个节点并将它们全部移动。您可以在以下位置找到有关蓝图的信息：[`docs.unrealengine.com/en-US/Engine/Blueprints/BestPractices`](https://docs.unrealengine.com/en-US/Engine/Blueprints/BestPractices)。

1.  在“类”部分，单击下拉菜单并输入您创建的蓝图名称（MyProfile），然后从列表中选择它。

1.  您还需要为将作为对象所有者的“外部”属性选择某个内容。单击并拖动蓝色圆圈，将鼠标移至节点的左侧，然后释放鼠标以创建一个新节点。当菜单弹出时，输入单词`self`，然后选择“获取对自身的引用”选项。如果一切顺利，您的蓝图应该看起来像这样：

![图片](img/bf976ee7-696b-4cbb-8a58-2eda3acc5830.png)

这将使用我们之前创建的 MyProfile 实例的信息创建一个变量。然而，除非我们将其转换为变量，否则我们无法使用它。将其拖放到 Return Value 属性的右侧并选择“提升为变量”。这将自动创建一个名为 `NewVar_0` 的变量并创建一个 SET 节点，但您可以使用菜单左侧的菜单将其重命名为您想要的任何名称。

1.  在 SET 节点右侧，将节点右上角的白色箭头拖放到节点上并创建一个打印文本节点。

1.  我们现在需要打印一些内容，名称属性将非常适合这个用途。在 SET 节点右侧，将蓝色节点拖放到并选择 Variables | Stats | Get Name 节点。

1.  最后，将名称值连接到打印文本节点的 In Text 属性。它将自动创建一个转换节点，将名称字符串转换为它可以理解的文本对象。

![图片](img/223a2a1c-2c4f-4e5c-a49a-bfc510832fc8.png)

最后，整个蓝图应该看起来像这样：

![蓝图示例](img/f4713373-7763-48e8-b6d5-da8ae46aea27.png)

完成的蓝图

如果一切顺利，您应该能够点击编译按钮，然后通过点击菜单顶部的播放按钮来玩游戏：

![蓝图示例](img/eec7e943-df92-4160-8faa-f5435887ca15.png)

当您开始游戏时，应该会看到比利出现在屏幕上，就像我们之前设置的那样！

# 工作原理...

`UPROPERTY`s 会自动为 UE4 类编写 `Get`/`Set` 方法，并可用于访问和分配属性值，正如我们刚才看到的。

# 将 UCLASS 指定为 UPROPERTY 的类型

因此，您已经构建了一些自定义的 `UCLASS`，打算在 UE4 内使用。我们在之前的配方中使用蓝图在编辑器中创建了一个，但如何在 C++ 中实例化它们呢？UE4 中的对象是引用计数和内存管理的对象，因此您不应直接使用 C++ 关键字 `new` 来分配它们。相反，您将不得不使用一个名为 `ConstructObject` 的函数，这样我们就可以实例化您的 `UObject` 派生类。

`ConstructObject` 不仅需要您正在创建的对象的 C++ 类名；它还需要 C++ 类的蓝图类派生（一个 `UClass*` 引用）。`UClass*` 引用只是一个指向蓝图的指针。

我们如何在 C++ 代码中实例化特定蓝图的实例？C++ 代码既不知道也不应该知道具体的 `UCLASS` 名称，因为这些名称是在 UE4 编辑器中创建和编辑的，您只能在编译后才能访问。我们需要一种方法，以某种方式将蓝图类名称返回以供 C++ 代码实例化。

我们这样做是通过让 UE4 程序员从 UE4 编辑器内列出的所有可用的蓝图（从特定的 C++ 类派生而来）的简单下拉菜单中选择要使用的 `UClass`，在 C++ 代码中使用。为此，我们只需提供一个用户可编辑的 `UPROPERTY`，其中包含 `TSubclassOf<C++ClassName>` 类型的变量。或者，您也可以使用 `FStringClassReference` 来实现相同的目标。

应将 `UCLASS` 视为 C++ 代码的资源，其名称永远不应硬编码到代码库中。

# 准备工作

在您的 UE4 代码中，您经常会需要引用项目中的不同 `UCLASS`。例如，假设您需要知道玩家对象的 `UCLASS`，以便您可以在代码中使用 `SpawnObject`。从 C++ 代码中指定 `UCLASS` 非常尴尬，因为 C++ 代码根本不应该知道在蓝图编辑器中创建的派生 `UCLASS` 的具体实例。就像我们不想将特定的资产名称烘焙到 C++ 代码中一样，我们也不希望将派生蓝图类名称硬编码到 C++ 代码中。

因此，我们使用一个 C++ 变量（例如，`UClassOfPlayer`），并在 UE4 编辑器的蓝图对话框中选择它。您可以使用 `TSubclassOf` 成员或 `FStringClassReference` 成员来这样做。

# 如何实现...

1.  导航到你想添加 `UCLASS` 引用成员的 C++ 类。

1.  在 `UCLASS` 内部，使用以下形式的代码来声明一个 `UPROPERTY`，允许选择从 `UObject` 在层次结构中派生的 `UClass`（蓝图类）：

```cpp
UCLASS(Blueprintable, BlueprintType)
class CHAPTER_02_API UUserProfile : public UObject
{
  GENERATED_BODY()

public:
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Stats)
  float Armor;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Stats)
  float HpMax;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Stats)
  FString Name;

 // Displays any UClasses deriving from UObject in a dropdown 
 // menu in Blueprints
 UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Unit)
 TSubclassOf<UObject> UClassOfPlayer; 

 // Displays string names of UCLASSes that derive from
 // the GameMode C++ base class
 UPROPERTY( EditAnywhere, meta=(MetaClass="GameMode"), 
 Category = Unit )
 FStringClassReference UClassGameMode;
};
```

Visual Studio 可能会下划线显示 `UClassOfPlayer` 变量，并指出不允许不完整的类。这是 Visual Studio 错误不正确的情况之一，可以忽略，因为它在 UE4 内部编译时不会有问题。

1.  蓝图 C++ 类，然后打开该蓝图：

![](img/dbb893f3-d100-42be-afc7-4fc7a2076da2.png)

注意，我们现在有了第二个类别，单位，并且它具有我们在脚本中指定的两个属性。

1.  点击 `UClassOfPlayer` 菜单旁边的下拉菜单。

1.  从列出的 `UClass` 的下拉菜单中选择合适的 `UClassOfPlayer` 成员：

![](img/21af16e5-ea46-4084-bd60-4b3f4fa8dac6.png)

# 它是如何工作的...

Unreal Engine 4 提供了多种方式来指定 `UClass` 或期望的类类型。

# TSubclassOf

`TSubclassOf< >` 成员将允许你在 UE4 编辑器内编辑任何具有 `TSubclassOf< >` 成员的蓝图时，使用下拉菜单指定 `UClass` 名称。

# FStringClassReference

`MetaClass` 标签指的是你期望 `UClassName` 从其派生的基 C++ 类。这限制了下拉菜单的内容仅限于从该 C++ 类派生的蓝图。如果你希望显示项目中的所有蓝图，可以省略 `MetaClass` 标签。

# 实例化从 UObject 派生的类（ConstructObject< > 和 NewObject< >）

在 C++ 中创建类实例的传统方法是使用关键字 `new`。然而，UE4 实际上在其内部创建其类的实例，并要求你调用特殊的工厂函数来生成任何你想要实例化的 `UCLASS` 的副本。你生成的是 UE4 蓝图类的实例，而不仅仅是 C++ 类。当你创建 `UObject` 派生的类时，你需要使用特殊的 UE4 引擎函数来实例化它们。

工厂方法允许 UE4 对对象进行一些内存管理，控制对象被删除时发生的情况。此方法允许 UE4 跟踪对象的所有引用，以便在对象销毁时，可以轻松地解除所有对对象的引用。这确保程序中不存在指向已失效内存的悬空指针。这个过程通常被称为 **垃圾回收**。

# 准备工作

实例化不是 `AActor` 类派生的 `UObject` 派生类不使用 `UWorld::SpawnActor< >`。相反，我们必须使用一个特殊的全局函数：`ConstructObject< >` 或 `NewObject< >`。请注意，你不应该使用裸 C++ 关键字 `new` 来分配 UE4 `UObject` 类派生的新实例。

你至少需要两份信息来正确实例化你的 `UCLASS` 实例：

+   一个指向你想要实例化的类类型的 C++ 类型化 `UClass` 引用（蓝图类）

+   蓝图类所继承的原始 C++ 基类

# 如何做到这一点...

在一个全局可访问的对象（如你的 `GameMode` 对象）中，添加一个 `TSubclassOf< YourC++ClassName > UPROPERTY()` 来指定并为你 C++ 代码提供 `UCLASS` 名称。要对 GameMode 执行此操作，请按照以下步骤操作：

1.  从 Visual Studio 中，在 Solution Explorer 中打开 `Chapter02_GameModeBase.h` 文件。从那里，将脚本更新为以下内容：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "UserProfile.h"
#include "Chapter_02GameModeBase.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_02_API AChapter_02GameModeBase : public AGameModeBase
{
  GENERATED_BODY()

public:
 UPROPERTY( EditAnywhere, BlueprintReadWrite, Category = UClassNames ) 
 TSubclassOf<UUserProfile> UPBlueprintClassName; 
};
```

1.  保存并编译你的代码。

1.  从 UE4 编辑器中，从这个类创建一个蓝图。双击它进入蓝图编辑器，然后从下拉菜单中选择你的 `UClass` 名称，以便你可以看到它做了什么。保存并退出编辑器：

![图片](img/1fd16d87-b45d-4366-b178-bc8863d838b8.png)

1.  在你的 C++ 代码中，找到你想要实例化 `UCLASS` 实例的部分。

1.  使用以下公式使用 `ConstructObject< >` 实例化对象：

```cpp
ObjectType* object = ConstructObject< ObjectType >( 
 UClassReference );
```

例如，使用我们在上一个菜谱中指定的 `UserProfile` 对象，我们会得到如下代码：

```cpp
// Get the GameMode object, which has a reference to  
// the UClass name that we should instantiate: 
AChapter2GameMode *gm = Cast<AChapter2GameMode>( 
                                        GetWorld()->GetAuthGameMode()); 
if( gm )
{
  UUserProfile* newobject = NewObject<UUserProfile>(                                         
                                      (UObject*)GetTransientPackage(), 
                                       UUserProfile::StaticClass() );
}
```

你可以在本书示例代码中的 `Chapter_02GameModeBase.cpp` 文件中看到一个使用此功能的例子。

# 它是如何工作的...

使用 `NewObject` 实例化 `UObject` 类很简单。`ConstructObject` 将实例化蓝图类类型的对象，并返回正确类型的 C++ 指针。

很不幸，`NewObject` 有一个讨厌的第一个参数，它要求你在每次调用时传递 `GetTransientPackage()`。

构建你的 UE4 `UObject` 派生类时，不要使用关键字 `new`！它将无法得到适当的内存管理。

想了解更多关于 `NewObject` 和其他对象创建函数的信息，请查看[`docs.unrealengine.com/en-us/Programming/UnrealArchitecture/Objects/Creation`](https://docs.unrealengine.com/en-us/Programming/UnrealArchitecture/Objects/Creation)。

# 还有更多...

`NewObject` 函数是面向对象世界所说的工厂，它是一种常见的设计模式。你要求工厂为你创建对象；你不需要自己构建它。使用工厂模式可以使引擎在对象创建时轻松跟踪对象。

想了解更多关于设计模式的信息，包括工厂模式，请查看 Erich Gamma 所著的 *《设计模式：可复用面向对象软件元素》*。

如果你对游戏开发中的设计模式感兴趣，你可能希望查看 *《游戏开发模式和最佳实践》*，该书也由 Packt Publishing 出版。

# 销毁从 UObject 派生的类

在 UE4 中移除任何`UObject`派生类都很简单。当你准备好删除你的`UObject`派生类时，我们只需在它上面调用一个函数（`ConditionalBeginDestroy()`）来开始拆卸。我们不会在`UObject`派生类上使用原生的 C++ `delete`命令。我们将在下面的配方中展示这一点。

# 准备工作

要完成这个配方，你需要在你的项目中有一个对象（在这个例子中是`objectInstance`）是你希望销毁的。

# 如何做到这一点...

1.  在你的对象实例上调用`objectInstance->ConditionalBeginDestroy()`。

1.  在你的客户端代码中，将所有对`objectInstance`的引用置为空，并且在调用`ConditionalBeginDestroy()`之后不要再使用`objectInstance`：

```cpp
// Destroy object
if(newobject)
{
  newobject->ConditionalBeginDestroy();
  newobject = nullptr;
}
```

# 它是如何工作的...

`ConditionalBeginDestroy()`函数通过移除所有内部引擎链接来开始销毁过程。这从引擎的角度标记对象为销毁。然后，通过销毁其内部属性，稍后实际销毁对象。

在对对象调用`ConditionalBeginDestroy()`之后，你的（客户端）代码必须认为该对象将被销毁，并且必须不再使用它。

实际的内存恢复发生在对对象调用`ConditionalBeginDestroy()`之后的一段时间。有一个垃圾回收例程会在固定的时间间隔内清除游戏程序不再引用的对象的内存。垃圾收集器调用之间的时间间隔列在`C:\Program Files (x86)\Epic Games\Launcher\Engine\Config\BaseEngine.ini`中，默认为每 61.1 秒收集一次：

```cpp
gc.TimeBetweenPurgingPendingKillObjects=61.1
```

如果在多次调用`ConditionalBeginDestroy()`之后内存似乎很低，你可以通过调用`GetWorld()->ForceGarbageCollection(true)`来触发内存清理，以强制进行内部内存清理。

通常，除非你急需清理内存，否则你不需要担心垃圾回收或间隔。不要频繁调用垃圾回收例程，因为这可能会在游戏中引起不必要的延迟。

# 创建一个 USTRUCT

你可能想在 UE4 中构建一个包含多个成员的蓝图可编辑属性。我们将在这个配方中创建的`FColoredTexture`结构将允许你将纹理及其颜色组合在同一结构中，以便在任何其他`UObject`派生类、`Blueprintable`类中包含和指定：

![图片](img/beee3b48-2091-4242-a170-46e6704965ce.png)

`FColoredTexture`结构确实在蓝图外观中具有视觉元素，如前一张截图所示。

这是为了良好的组织和其他`UCLASS UPROPERTIES()`的便利性。

你可能想在游戏中使用`struct`关键字构建一个 C++结构。

# 准备工作

`UObject` 是所有 UE4 类对象的基类，而 `FStruct` 只是任何普通的 C++ 风格的结构体。所有使用引擎内自动内存管理功能的对象都必须从这个类派生。

如果您还记得 C++ 语言，C++ `class` 和 C++ `struct` 之间的唯一区别是 C++ 类默认拥有 `private` 成员，而结构体默认为 `public` 成员。

在像 C# 这样的语言中，情况并非如此。在 C# 中，结构体是值类型，而类是引用类型。

# 如何操作...

我们将在 C++ 代码中创建一个名为 `FColoredTexture` 的结构，以包含一个纹理和一个调节颜色：

1.  从 Visual Studio 中，右键单击 Games/Chapter_02/Source/Chapter_02 文件夹，然后选择 Add | New item.... 从菜单中选择一个头文件 (.h)，然后命名文件为 `ColoredTexture.h`（而不是 `FColoredTexture`）。

1.  在“位置”下，确保您选择与项目中的其他脚本文件相同的文件夹（在我的情况下，`C:\Users\admin\Documents\Unreal Projects\Chapter_02\Source\Chapter_02`），而不是默认设置：

![](img/894cbb06-baf9-4618-bcde-1bd140907c72.png)

1.  创建后，在 `ColoredTexture.h` 中使用以下代码：

```cpp
#pragma once 

#include "ObjectMacros.h"
#include "ColoredTexture.generated.h"

USTRUCT(Blueprintable) 
struct CHAPTER_02_API FColoredTexture 
{
  GENERATED_USTRUCT_BODY()

public: 
  UPROPERTY( EditAnywhere, BlueprintReadWrite, Category = HUD ) 
  UTexture* Texture; 

  UPROPERTY( EditAnywhere, BlueprintReadWrite, Category = HUD ) 
  FLinearColor Color; 
}; 
```

1.  在蓝图可用的 `UCLASS()` 中使用 `ColoredTexture.h` 作为 `UPROPERTY()`，使用如下 `UPROPERTY()` 声明：

```cpp
#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "ColoredTexture.h"
#include "UserProfile.generated.h"

/**
 * UCLASS macro options sets this C++ class to be
 * Blueprintable within the UE4 Editor
 */
UCLASS(Blueprintable, BlueprintType)
class CHAPTER_02_API UUserProfile : public UObject
{
  GENERATED_BODY()

public:
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Stats)
  float Armor;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Stats)
  float HpMax;

  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Stats)
  FString Name;

  // Displays any UClasses deriving from UObject in a dropdown 
  // menu in Blueprints
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Unit)
  TSubclassOf<UObject> UClassOfPlayer; 

  // Displays string names of UCLASSes that derive from
  // the GameMode C++ base class
  UPROPERTY(EditAnywhere, meta=(MetaClass="GameMode"), Category = Unit )
  FStringClassReference UClassGameMode;

 // Custom struct example
 UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = HUD) 
 FColoredTexture Texture; 
};
```

1.  保存您的脚本并编译更改。进入您的对象蓝图后，您应该会注意到新的属性：

![](img/5699f592-65aa-48d5-9511-2035977e0c6f.png)

# 工作原理...

为 `FColoredTexture` 指定的 `UPROPERTY()` 当作为另一个类中的 `UPROPERTY()` 字段包含时，将在编辑器中显示为可编辑字段，如第 3 步所示。

# 还有更多...

创建结构体（即 `USTRUCT()` 而不是普通的 C++ 结构体）的主要原因是为了与 UE4 引擎功能接口。对于快速的小结构，您可以使用普通的 C++ 代码（无需创建 `USTRUCT()` 对象），这些结构不需要引擎直接使用它们。

# 创建一个 UENUM()

C++ `enum` 实例在典型的 C++ 代码中非常有用。UE4 有一种自定义的枚举类型称为 `UENUM()`，它允许您创建一个在您正在编辑的蓝图中的下拉菜单中显示的 `enum`。

# 如何操作...

1.  前往将使用您指定的 `UENUM()` 的头文件，或者创建一个

    文件名为 `EnumName.h`。

1.  使用以下代码：

```cpp
UENUM() 
enum Status 
{ 
  Stopped     UMETA(DisplayName = "Stopped"), 
  Moving      UMETA(DisplayName = "Moving"), 
  Attacking   UMETA(DisplayName = "Attacking"), 
}; 
```

1.  在 `UCLASS()` 中使用您的 `UENUM()`，如下所示：

```cpp
UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = 
 Status) 
TEnumAsByte<Status> status; 
```

# 工作原理...

`UENUM()` 在代码编辑器中作为蓝图编辑器中的下拉菜单出现，您只能从中选择几个值之一：

![](img/ff0d78df-992a-47b1-bd70-9231db4839c7.png)

如您所见，指定的值都在那里！
