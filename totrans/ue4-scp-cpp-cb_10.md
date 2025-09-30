# 集成 C++ 和 Unreal 编辑器：第二部分

在本章中，我们将介绍以下食谱：

+   创建一个新的编辑器模块

+   创建新的工具栏按钮

+   创建新的菜单条目

+   创建一个新的编辑器窗口

+   创建一个新的资产类型

+   为资产创建自定义上下文菜单条目

+   创建新的控制台命令

+   为 Blueprint 创建一个新的图形引脚可视化器

+   使用自定义 Details 面板检查类型

# 简介

在游戏开发中，除了创建游戏，你通常还需要为其他开发者创建工具，这些工具已经根据你正在工作的项目进行了定制。实际上，这通常是 AAA 游戏行业中初级游戏开发者职位中较为常见的一种。在本章中，我们将学习如何实现自定义编辑器窗口和自定义详细面板来检查用户创建的类型。

# 创建一个新的编辑器模块

以下食谱都与编辑器模式特定的代码和引擎模块交互。因此，创建一个仅在引擎以编辑器模式运行时才会加载的新模块被认为是一种良好的实践，这样我们就可以将所有仅用于编辑器的代码放在其中。

# 如何做到这一点...

1.  在文本编辑器（如记事本或 Notepad++）中打开你的项目 `.uproject` 文件。你可以在项目文件夹中找到该文件，它应该看起来与以下截图类似：

![图片](img/b4051c5c-b769-4866-98f4-39af0c8018de.png)

1.  将以下片段中的粗体部分添加到文件中：

```cpp
{
  "FileVersion": 3,
  "EngineAssociation": "4.21",
  "Category": "",
  "Description": "",
  "Modules": [
    {
      "Name": "Chapter_10",
      "Type": "Runtime",
      "LoadingPhase": "Default"
    }, 
 { 
 "Name": "Chapter_10Editor", 
 "Type": "Editor", 
 "LoadingPhase": "PostEngineInit", 
 "AdditionalDependencies": [ 
 "Engine", 
 "CoreUObject" 
 ] 
 } 
  ]
}

```

注意在第二组花括号之前第一个模块后面的逗号。

1.  在你的 `Source` 文件夹中，创建一个与你在 `uproject` 文件中指定的相同名称的新文件夹（在本例中为 `"Chapter_10Editor"`）：

![图片](img/fc3ee6b5-e94d-4330-a120-4b2f707d6edc.png)

1.  打开 `Chapter_10Editor.Target.cs` 文件并将其更新为以下内容：

```cpp
using UnrealBuildTool;
using System.Collections.Generic;

public class Chapter_10EditorTarget : TargetRules
{
  public Chapter_10EditorTarget(TargetInfo Target) : base(Target)
  {
    Type = TargetType.Editor;

    ExtraModuleNames.AddRange( new string[] { "Chapter_10Editor" } );
  }
}
```

1.  在这个新文件夹中，创建一个空的 `.txt` 文件并将其重命名为 `Chapter_10Editor.Build.cs`：

![图片](img/1f0a6cba-aeb7-420e-9d08-cec2aed2029e.png)

1.  将以下内容插入到文件中：

```cpp
using UnrealBuildTool;

public class Chapter_10Editor : ModuleRules
{
    public Chapter_10Editor(ReadOnlyTargetRules Target) : 
    base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] { "Core", 
        "CoreUObject", "Engine", "InputCore", "RHI", "RenderCore", 
        "ShaderCore", "MainFrame", "AssetTools", "AppFramework", 
        "PropertyEditor"});

       PublicDependencyModuleNames.Add("Chapter_10");

        PrivateDependencyModuleNames.AddRange(new string[] { 
        "UnrealEd", "Slate", "SlateCore", "EditorStyle", 
        "GraphEditor", "BlueprintGraph" });

    }
}
```

1.  仍然在 `Chapter10_Editor` 文件夹中，创建一个名为 `Chapter_10Editor.h` 的新文件并添加以下内容：

```cpp
#pragma once

#include "Engine.h"
#include "Modules/ModuleInterface.h"
#include "Modules/ModuleManager.h"
#include "UnrealEd.h"

class FChapter_10EditorModule: public IModuleInterface 
{ 
}; 
```

1.  最后，创建一个名为 `Chapter_10Editor.cpp` 的新源文件。

1.  添加以下代码：

```cpp
#include "Chapter_10Editor.h" 
#include "Modules/ModuleManager.h"
#include "Modules/ModuleInterface.h"

IMPLEMENT_GAME_MODULE(FChapter_10EditorModule, Chapter_10Editor)
```

1.  最后，如果你打开了 Visual Studio，请关闭它。然后，右键单击 `.uproject` 文件并选择“生成 Visual Studio 项目文件”：

![图片](img/71edf948-dcce-4da8-a1a3-e8069a117f83.png)

1.  你应该看到一个小的窗口启动，显示进度条，然后关闭：

![图片](img/766c4e4b-557f-4785-8c01-ae82430cadc7.jpg)

1.  现在，你可以启动 Visual Studio，验证你的新模块是否在 IDE 中可见，并成功编译你的项目：

![图片](img/770bd61f-d16a-49f2-9c81-f2f1d3be25d3.png)

1.  模块现在已准备好进行下一组食谱。

在此编辑器模块中进行的代码更改不会像运行时模块中的代码那样支持热重载。如果你遇到提及生成头文件更改的编译错误，只需关闭编辑器，并在你的 IDE 中重新构建它即可。

# 它是如何工作的...

Unreal 项目使用`.uproject`文件格式来指定有关项目的一些不同信息。

此信息用于通知头文件和构建工具关于构成此项目的模块，并用于代码生成和`makefile`创建。

此文件使用 JSON 风格的格式。

这些包括以下内容：

+   项目应该在其中打开的引擎版本

+   项目中使用的模块列表

+   模块声明的列表

这些模块声明中的每一个都包含以下内容：

+   模块的名称。

+   模块的类型——这是一个编辑器模块（仅在编辑器构建中运行，可以访问仅限编辑器的类）还是一个运行时模块（在编辑器和发布构建中运行）？

+   模块的加载阶段——模块可以在程序启动的不同阶段加载。此值指定模块应该加载的点，例如，如果有其他模块中的依赖项应该先加载。

+   模块的依赖项列表。这些是包含导出函数或类的必要模块，这些函数或类是模块所依赖的。

我们向`uproject`文件中添加了一个新模块。该模块的名称是`Chapter_10Editor`（传统上，`Editor`应该附加到主游戏模块的编辑器模块上）。

此模块被标记为编辑器模块，并设置为在基线引擎之后加载，以便它可以使用在引擎代码中声明的类。

我们模块的依赖项目前保留为默认值。

将`uproject`文件修改为包含我们的新模块后，我们需要为其编写一个构建脚本。

构建脚本是用 C#编写的，名称为`<ModuleName>.Build.cs`。

与 C++不同，C#不使用单独的头文件和实现文件——所有内容都在一个`.cs`文件中。

我们想要访问在`UnrealBuildTool`模块中声明的类，因此我们包含一个`using`语句来指示我们想要访问该命名空间。

我们创建一个与我们的模块同名且继承自`ModuleRules`的`public`类。

在我们的构造函数中，我们向此模块的依赖项中添加了多个模块。这些既有私有依赖项，也有公共依赖项。

根据`ModuleRules`类的代码，公共依赖项是模块的公共头文件所依赖的模块。私有依赖项是私有代码所依赖的模块。任何在公共头文件和私有代码中使用的都应该放入`PublicDependencyModuleNames`数组中。

你会注意到我们的`PublicDependencyModuleNames`数组包含我们的主要游戏模块。这是因为本章的一些食谱将扩展编辑器以更好地支持定义在我们主要游戏模块内的类。

既然我们已经通过项目文件告诉构建系统我们有一个新的模块需要构建，并且我们已经指定了如何使用构建脚本来构建该模块，我们需要创建一个 C++类，即我们的实际模块。

我们创建一个头文件，其中包含引擎头文件、`ModuleManager`头文件和`UnrealEd`头文件。

我们包含`ModuleManager`因为它定义了`IModuleInterface`，这是我们模块将继承的类。

我们还包含`UnrealEd`，因为我们正在编写一个需要访问编辑器功能的编辑器模块。

我们声明的类继承自`IModuleInterface`，并且其名称由通常的前缀`F`加上模块名称组成。

在`.cpp`文件中，我们包含我们的模块头文件，然后使用`IMPLEMENT_GAME_MODULE`宏。

`IMPLEMENT_GAME_MODULE`声明了一个导出的 C 函数`InitializeModule()`，它返回我们新模块类的一个实例。

这意味着 Unreal 可以简单地调用任何导出它的库上的`InitializeModule()`来获取实际模块实现的引用，而无需知道它是什么类。

在添加了我们的新模块后，我们现在需要重新构建我们的 Visual Studio 解决方案，因此我们关闭 Visual Studio，然后使用上下文菜单重新生成项目文件。

重新构建项目后，新的模块将在 Visual Studio 中可见，我们可以像往常一样向其中添加代码。

# 创建新的工具栏按钮

如果你为编辑器内显示的自定义工具或窗口创建了，你可能需要某种方式让用户使其出现。最简单的方法是创建一个工具栏自定义，添加一个新的工具栏按钮，并在点击时显示你的窗口。按照之前的食谱创建一个新的引擎模块，因为我们还需要它来初始化我们的工具栏自定义。

# 如何做到这一点...

1.  在`Chapter_10Editor`文件夹内，创建一个新的头文件，名为`CookbookCommands.h`，并插入以下类声明：

```cpp
#pragma once
#include "Commands.h"
#include "EditorStyleSet.h"

class FCookbookCommands : public TCommands<FCookbookCommands>
{
public:
  FCookbookCommands()
    : TCommands<FCookbookCommands>( 
      FName(TEXT("UE4_Cookbook")), 
      FText::FromString("Cookbook Commands"), 
      NAME_None, 
      FEditorStyle::GetStyleSetName()) 
  {
  };

  virtual void RegisterCommands() override;

  TSharedPtr<FUICommandInfo> MyButton;

  TSharedPtr<FUICommandInfo> MyMenuButton;
};
```

1.  通过在`.cpp`文件中放置以下内容来实现新类：

```cpp
#include "CookbookCommands.h"
#include "Chapter_10Editor.h"
#include "Commands.h"

void FCookbookCommands::RegisterCommands()
{
#define LOCTEXT_NAMESPACE ""
  UI_COMMAND(MyButton, "Cookbook", "Demo Cookbook Toolbar Command", EUserInterfaceActionType::Button, FInputGesture());
  UI_COMMAND(MyMenuButton, "Cookbook", "Demo Cookbook Toolbar Command", EUserInterfaceActionType::Button, FInputGesture());
#undef LOCTEXT_NAMESPACE
}

```

1.  接下来，我们需要更新我们的模块类（`Chapter_10Editor.h`）到以下内容：

```cpp
#pragma once

#include "Engine.h"
#include "Modules/ModuleInterface.h"
#include "Modules/ModuleManager.h"
#include "UnrealEd.h"
#include "CookbookCommands.h"
#include "Editor/MainFrame/Public/Interfaces/IMainFrameModule.h"

class FChapter_10EditorModule: public IModuleInterface 
{ 
 virtual void StartupModule() override;
 virtual void ShutdownModule() override;

 TSharedPtr<FExtender> ToolbarExtender;
 TSharedPtr<const FExtensionBase> Extension;

 void MyButton_Clicked()
 {

 TSharedRef<SWindow> CookbookWindow = SNew(SWindow)
 .Title(FText::FromString(TEXT("Cookbook Window")))
 .ClientSize(FVector2D(800, 400))
 .SupportsMaximize(false)
 .SupportsMinimize(false);

 IMainFrameModule& MainFrameModule =
 FModuleManager::LoadModuleChecked<IMainFrameModule>
 (TEXT("MainFrame"));

 if (MainFrameModule.GetParentWindow().IsValid())
 {
 FSlateApplication::Get().AddWindowAsNativeChild
 (CookbookWindow, MainFrameModule.GetParentWindow()
 .ToSharedRef());
 }
 else
 {
 FSlateApplication::Get().AddWindow(CookbookWindow);
 }

 };

 void AddToolbarExtension(FToolBarBuilder &builder)
 {

 FSlateIcon IconBrush =
 FSlateIcon(FEditorStyle::GetStyleSetName(),
 "LevelEditor.ViewOptions",
 "LevelEditor.ViewOptions.Small"); builder.AddToolBarButton(FCookbookCommands::Get()
 .MyButton, NAME_None, FText::FromString("My Button"),
 FText::FromString("Click me to display a message"),
 IconBrush, NAME_None);

 };
}; 
```

确保也包含你的命令类的头文件。

1.  现在我们需要实现`StartupModule`和`ShutdownModule`：

```cpp
#include "Chapter_10Editor.h" 
#include "Modules/ModuleManager.h"
#include "Modules/ModuleInterface.h"
#include "LevelEditor.h" 
#include "SlateBasics.h" 
#include "MultiBoxExtender.h" 
#include "CookbookCommands.h" 

IMPLEMENT_GAME_MODULE(FChapter_10EditorModule, Chapter_10Editor)

void FChapter_10EditorModule::StartupModule()
{

 FCookbookCommands::Register();

 TSharedPtr<FUICommandList> CommandList = MakeShareable(new FUICommandList());

 CommandList->MapAction(FCookbookCommands::Get().MyButton, FExecuteAction::CreateRaw(this, &FChapter_10EditorModule::MyButton_Clicked), FCanExecuteAction());

 ToolbarExtender = MakeShareable(new FExtender());

 FLevelEditorModule& LevelEditorModule = FModuleManager::LoadModuleChecked<FLevelEditorModule>( "LevelEditor" );

 Extension = ToolbarExtender->AddToolBarExtension("Compile", EExtensionHook::Before, CommandList, FToolBarExtensionDelegate::CreateRaw(this, &FChapter_10EditorModule::AddToolbarExtension)); 

 LevelEditorModule.GetToolBarExtensibilityManager()->AddExtender(ToolbarExtender);

}

void FChapter_10EditorModule::ShutdownModule()
{

 ToolbarExtender->RemoveExtension(Extension.ToSharedRef());

 Extension.Reset();
 ToolbarExtender.Reset();

}
```

1.  如有必要，重新生成你的项目文件，从 Visual Studio 编译你的项目并启动编辑器。

1.  确认主级别编辑器工具栏中有一个新的按钮，可以点击以打开一个新窗口：

![图片](img/6f9e599c-aee8-4438-a084-5350a219f1ea.png)

# 它是如何工作的...

Unreal 的编辑器 UI 基于命令的概念。命令是一种设计模式，它允许 UI 与其需要执行的操作之间有更松散的耦合。

要创建包含一组命令的类，必须从 `TCommands` 继承。

`TCommands` 是一个模板类，它利用了**奇特重复模板模式**（**CRTP**）。CRTP 在 Slate UI 代码中广泛使用，作为创建编译时多态的手段。

在 `FCookbookCommands` 构造函数的初始化列表中，我们调用父类构造函数，传递一系列参数：

+   第一个参数是命令集的名称，它是一个简单的 `FName`。

+   第二个参数是工具提示/人类可读字符串，因此使用 `FText` 以便在必要时支持本地化。

+   如果存在命令的父组，第三个参数包含组的名称。否则，它包含 `NAME_None`。

+   构造函数的最后一个参数是包含命令集将使用的任何命令图标的 Slate 风格集。

`RegisterCommands()` 函数允许 `TCommands` 派生类创建它们所需的任何命令对象。从该函数返回的 `FUICommandInfo` 实例作为 `Commands` 类的成员存储，以便 UI 元素或函数可以绑定到命令。

这就是为什么我们有成员变量 `TSharedPtr<FUICommandInfo> MyButton`。

在类的实现中，我们只需在 `RegisterCommands` 中创建我们的命令。

用于创建 `FUICommandInfo` 实例的 `UI_COMMAND` 宏期望定义一个本地化命名空间，即使它只是一个空的默认命名空间。因此，我们需要用 `#defines` 将我们的 `UI_COMMAND` 调用括起来，以设置 `LOCTEXT_NAMESPACE` 的有效值，即使我们不想使用本地化。

实际的 `UI_COMMAND` 宏接受多个参数：

+   第一个参数是存储 `FUICommandInfo` 的变量

+   第二个参数是命令的人类可读名称

+   第三个参数是命令的描述

+   第四个参数是 `EUserInterfaceActionType`

此枚举基本上指定了正在创建哪种类型的按钮。它支持 `Button`、`ToggleButton`、`RadioButton` 和 `Check` 作为有效类型。

按钮是简单的通用按钮。切换按钮存储开/关状态。单选按钮类似于切换，但与其他单选按钮分组，并且一次只能启用一个。最后，复选框显示与按钮相邻的只读复选框。

`UI_COMMAND` 的最后一个参数是输入和弦，或激活命令所需的键的组合。

此参数主要用于定义与命令相关联的热键的关键组合，而不是按钮。因此，我们使用空的 `InputGesture`。

因此，我们现在有一组命令，但我们还没有告诉引擎我们想要将这组命令添加到显示在工具栏上的命令中。我们也没有设置按钮点击时实际发生的事情。为此，我们需要在我们的模块开始时执行一些初始化，所以我们将在`StartupModule`/`ShutdownModule`函数中放置一些代码。

在`StartupModule`内部，我们调用我们之前定义的命令类上的静态`Register`函数。

然后，我们使用`MakeShareable`函数创建一个命令列表的共享指针。

在命令列表中，我们使用`MapAction`创建一个映射，或关联，将`UICommandInfo`对象（我们将其设置为`FCookbookCommands`的一个成员）与我们要在命令调用时执行的函数关联起来。

你会注意到，我们在这里没有明确设置任何有关可以用来调用命令的内容。

为了执行此映射，我们调用`MapAction`函数。`MapAction`的第一个参数是一个`FUICommandInfo`对象，我们可以通过使用其静态`Get()`方法从`FCookbookCommands`检索实例来获取它。

`FCookbookCommands`实现为一个单例——一个在整个应用程序中存在单个实例的类。你会在大多数地方看到这个模式——在引擎中有一个可用的静态`Get()`方法。

`MapAction`函数的第二个参数是一个委托，当命令执行时，它绑定到要调用的函数。

因为`Chapter_10EditorModule`是一个原始的 C++类而不是`UObject`，并且我们想要调用成员函数而不是`static`函数，所以我们使用`CreateRaw`创建一个新的委托，该委托绑定到一个原始 C++成员函数。

`CreateRaw`期望一个指向对象实例的指针和一个指向要在此指针上调用的函数的函数引用。

`MapAction`的第三个参数是一个委托，用于测试动作是否可以执行。因为我们希望命令始终可执行，我们可以使用一个简单的预定义委托，该委托始终返回`true`。

在我们的命令与其应调用的动作之间创建关联后，我们现在需要实际上告诉扩展系统我们想要将新命令添加到工具栏。

我们可以通过`FExtender`类来实现这一点，该类可以用来扩展菜单、上下文菜单或工具栏。

我们最初创建一个`FExtender`实例作为共享指针，这样当模块关闭时，我们的扩展未初始化。

然后，我们在我们的新扩展器上调用`AddToolBarExtension`，并将结果存储在共享指针中，这样我们就可以在模块卸载时将其删除。

`AddToolBarExtension`的第一个参数是我们想要添加扩展的扩展点名称。

要找到我们想要放置扩展的位置，我们首先需要打开编辑器 UI 中扩展点的显示。

要这样做，请在编辑器中的“编辑”菜单中打开“编辑器首选项”：

![图片](img/733efb6c-4df7-4663-b51c-2629dabb48c2.png)

打开“通用”|“杂项”并选择显示 UI 扩展点：

![截图](img/a7875429-2beb-4350-8952-b13d6b5f419d.png)

重新启动编辑器，你应该会在编辑器 UI 上看到覆盖的绿色文本，如下面的截图所示：

![截图](img/9cd351fc-a334-42c3-8155-24fa0772578c.png)

覆盖编辑器 UI 的绿色文本

绿色文本表示`UIExtensionPoint`，文本的值是我们应该提供给`AddToolBarExtension`函数的字符串。

在这个菜谱中，我们将把我们的扩展添加到`Compile`扩展点，但当然，你也可以使用你想要的任何其他扩展点。

重要的是要注意，将工具栏扩展添加到菜单扩展点将静默失败，反之亦然。

`AddToolBarExtension`的第二个参数是相对于指定扩展点的位置锚点。我们选择了`FExtensionHook::Before`，因此我们的图标将在编译点之前显示。

下一个参数是我们包含映射操作的命令列表。

最后，最后一个参数是一个委托，它负责实际上在扩展点和之前指定的锚点处添加 UI 控件。

委托绑定到一个具有形式 void (`*func`) (`FToolBarBuilder`和`builder`)的函数。在这个例子中，它是一个在模块类中定义的名为`AddToolbarExtension`的函数。

当函数被调用时，在`builder`上调用添加 UI 元素的命令将把那些元素应用到我们在 UI 中指定的位置。

最后，我们需要在这个函数中加载关卡编辑器模块，这样我们就可以将我们的扩展器添加到关卡编辑器中的主工具栏。

如同往常一样，我们可以使用`ModuleManager`来加载一个模块并返回它的引用。

拥有这个引用后，我们可以获取模块的工具栏可扩展性管理器，并告诉它添加我们的扩展器。

虽然一开始这可能看起来有些繁琐，但目的是允许你将相同的工具栏扩展应用到不同模块的多个工具栏上，如果你想在不同的编辑器窗口之间创建一致的 UI 布局的话。

当然，初始化我们的扩展的对应操作是在我们的模块卸载时移除它。为此，我们从扩展器中移除我们的扩展，然后使扩展器和扩展的共享指针都为空，从而回收它们的内存分配。

编辑器模块中的`AddToolBarExtension`函数是负责实际上将 UI 元素添加到工具栏以调用我们的命令的那个函数。

它通过调用作为函数参数传入的`FToolBarBuilder`实例上的函数来实现这一点。

首先，我们使用`FSlateIcon`构造函数检索我们新工具栏按钮的适当图标。然后，在图标加载后，我们在`builder`实例上调用`AddToolBarButton`。

`AddToolbarButton`有几个参数。第一个参数是要绑定的命令——你会注意到它与我们之前绑定操作到命令时访问的相同的`MyButton`成员。第二个参数是之前指定的扩展钩子的覆盖，但我们不想覆盖它，所以我们可以使用`NAME_None`。第三个参数是我们创建的新按钮的标签覆盖。第四个参数是新按钮的提示信息。倒数第二个参数是按钮的图标，最后一个参数是用于引用此按钮元素以便进行高亮显示的名称，如果你希望使用编辑器中的教程框架。

# 创建新的菜单项

创建新菜单项的工作流程几乎与创建新工具栏按钮的工作流程相同，因此这个配方将基于上一个配方，并展示如何将其中创建的命令添加到菜单而不是工具栏中。

# 如何操作...

1.  在`Chapter10_Editor.h`中的`FChapter_10EditorModule`类内部创建一个新函数：

```cpp
void AddMenuExtension(FMenuBuilder &builder) 
{ 
  FSlateIcon IconBrush = 
   FSlateIcon(FEditorStyle::GetStyleSetName(), 
   "LevelEditor.ViewOptions", 
   "LevelEditor.ViewOptions.Small"); 

  builder.AddMenuEntry(FCookbookCommands::Get().MyButton); 
}; 
```

1.  在实现文件（`Chapter_10Editor.cpp`）中，在`StartupModule`函数内找到以下代码：

```cpp
EExtension = ToolbarExtender->AddToolBarExtension("Compile", EExtensionHook::Before, CommandList, FToolBarExtensionDelegate::CreateRaw(this, &FChapter_10EditorModule::AddToolbarExtension)); 
LevelEditorModule.GetToolBarExtensibilityManager()->AddExtender(ToolbarExtender);
```

1.  将前面的代码替换为以下代码：

```cpp
Extension = ToolbarExtender->AddMenuExtension("LevelEditor", EExtensionHook::Before, CommandList, FMenuExtensionDelegate::CreateRaw(this,&FChapter_10EditorModule::AddMenuExtension)); 
LevelEditorModule.GetMenuExtensibilityManager()->AddExtender(ToolbarExtender);
```

1.  编译你的代码并启动编辑器。

1.  确认你现在在“窗口”菜单下有一个菜单项，当点击时会显示“Cookbook”窗口。如果你遵循了前面的配方，你还会看到列出 UI 扩展点的绿色文本，包括我们在这个配方中使用的（LevelEditor）：

![图片](img/23b9b8b0-5d3a-40ba-9ba8-cc176d0f1b58.png)

# 它是如何工作的...

你会注意到`ToolbarExtender`是`FExtender`类型，而不是`FToolbarExtender`或`FMenuExtender`。

通过使用通用的`FExtender`类而不是特定的子类，框架允许你创建一系列命令-函数映射，这些映射可以在菜单或工具栏上使用。实际添加 UI 控件（在这个例子中，`AddMenuExtension`）的委托可以将这些控件链接到`FExtender`中的命令子集。

这样，你不需要为不同类型的扩展有不同的`TCommands`类，你可以将命令放入一个单一的中心类中，无论这些命令是从 UI 的哪个地方调用的。

因此，所需更改如下：

+   将对`AddToolBarExtension`的调用与`AddMenuExtension`交换

+   创建一个可以绑定到`FMenuExtensionDelegate`而不是`FToolbarExtensionDelegate`的函数

+   将扩展器添加到菜单扩展管理器而不是工具栏扩展管理器

# 创建一个新的编辑器窗口

当你有一个带有用户可配置设置的全新工具，或者想要向使用你自定义编辑器的人显示一些信息时，自定义编辑窗口非常有用。在开始之前，请确保你已经按照本章前面的食谱实现了编辑器模块。阅读“创建新的菜单条目”或“创建新的工具栏按钮”的食谱，以便你可以在编辑器中创建一个按钮，该按钮将启动我们的新窗口。

# 如何实现...

1.  在你的命令的绑定函数（在我们的例子中，是 `FChapter_10EditorModule` 类中的 `MyButton_Clicked` 函数，该类位于 `Chapter_10Editor.h` 文件中）内部，添加以下代码：

```cpp
void MyButton_Clicked()
{

    TSharedRef<SWindow> CookbookWindow = SNew(SWindow)
        .Title(FText::FromString(TEXT("Cookbook Window")))
        .ClientSize(FVector2D(800, 400))
        .SupportsMaximize(false)
        .SupportsMinimize(false)
        [
 SNew(SVerticalBox)
 + SVerticalBox::Slot()
 .HAlign(HAlign_Center)
 .VAlign(VAlign_Center)
 [
 SNew(STextBlock)
 .Text(FText::FromString(TEXT("Hello from Slate")))
 ]
 ];

    IMainFrameModule& MainFrameModule =
        FModuleManager::LoadModuleChecked<IMainFrameModule>
        (TEXT("MainFrame"));

    if (MainFrameModule.GetParentWindow().IsValid())
    {
        FSlateApplication::Get().AddWindowAsNativeChild
        (CookbookWindow, MainFrameModule.GetParentWindow()
            .ToSharedRef());
    }
    else
    {
        FSlateApplication::Get().AddWindow(CookbookWindow);
    }

};
```

注意，我们在声明 `.SupportsMinimize(false)` 的行末去掉了分号。

1.  编译你的代码并启动编辑器。

1.  当你激活你创建的命令时，无论是通过选择自定义菜单选项还是添加的工具栏选项，你应该看到窗口已经显示，中间有一些居中的文本：

![图片](img/4e289ce2-a243-402b-9d35-319f0b8f2801.png)

# 它是如何工作的...

你新创建的编辑器窗口不会自动显示，因此，在本食谱的开始部分提到，你应该已经实现了一个自定义菜单或工具栏按钮或控制台命令，我们可以使用它来触发新窗口的显示。

Slate 的所有 widget 通常都以 `TSharedRef< >` 或 `TSharedPtr< >` 的形式进行交互。

`SNew()` 函数返回一个模板化请求的 widget 类的 `TSharedRef`。

如前所述，Slate widget 实现了许多函数，这些函数都返回函数被调用的对象。这允许在创建时使用方法链来配置对象。

这允许使用 `<Widget>.Property(Value).Property(Value)` 的 Slate 语法。

在本食谱中设置的 widget 属性包括窗口标题、窗口大小以及窗口是否可以最大化或最小化。

一旦在 widget 上设置了所有必要的属性，就可以使用括号运算符（`[]`）来指定要放置在 widget 中的内容，例如，按钮内的图片或标签。

`SWindow` 是一个顶级 widget，它只为子 widget 有一个槽位，因此我们不需要自己为它添加槽位。我们通过在括号内创建它来将内容放入该槽位。

我们创建的内容是 `SVerticalBox`，这是一个可以拥有任意数量的槽位以在垂直列表中显示子 widget 的 widget。

对于我们想要放置到垂直列表中的每个 widget，我们需要创建一个**槽位**。

最简单的方法是使用重载的 `+` 运算符和 `SVerticalBox::Slot()` 函数。

`Slot()` 返回一个与任何其他 widget 一样的 widget，因此我们可以像在 `SWindow` 上做的那样设置它的属性。

这个食谱使用 `HAlign` 和 `VAlign` 在水平和垂直轴上居中 Slot 的内容。

`Slot`有一个单独的小部件子代，它是在`[]`运算符内部创建的，就像它们对`SWindow`一样。

在`Slot`内容内部，我们创建了一个包含一些自定义文本的文本块。

我们的新`SWindow`现在已经添加了子小部件，但它还没有被显示，因为它还没有被添加到窗口层次结构中。

主框架模块用于检查我们是否有顶级编辑器窗口，如果有，我们的新窗口就被添加为子窗口。

如果没有顶级窗口可以添加为子窗口，那么我们就使用 Slate 应用程序单例来添加我们的窗口，而不需要父窗口。

如果你想查看我们创建的窗口的层次结构，你可以使用 Slate Widget Reflector，它可以通过 Window | Developer Tools | Widget Reflector 访问。

如果你选择“选择已绘制的小部件”，并将光标悬停在自定义窗口中央的文本上，你将能够看到带有我们添加到其层次结构中的自定义小部件的`SWindow`：

![图片](img/0de07d76-1193-4fc5-ab7c-ed947a065605.png)

# 参见

+   第十一章，*使用 UE4 API*，全部关于 UI，将向你展示如何向你的新自定义窗口添加额外的元素

# 创建一个新的资产类型

在你的项目中的某个时候，你可能需要创建一个新的自定义资产类，例如，一个用于在 RPG 中存储对话数据的资产。为了正确地将这些与内容浏览器集成，你需要创建一个新的资产类型。

# 如何做到这一点...

1.  创建一个基于`UObject`的新 C++类，命名为`MyCustomAsset`：

![图片](img/d4eb0573-b97d-4470-90e8-3627adcf9c0d.png)

1.  打开脚本并更新`.h`文件的代码如下：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "MyCustomAsset.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_10_API UMyCustomAsset : public UObject
{
  GENERATED_BODY()

public:
 UPROPERTY(EditAnywhere, Category = "Custom Asset")
 FString Name;

};
```

1.  接下来，基于`UFactory`创建一个类：

![图片](img/e3e7c79e-3eeb-4cc4-a1e2-cd8e5371e942.png)

1.  给脚本命名为`CustomAssetFactory`并按下“创建类”按钮。

1.  在 Visual Studio 中打开脚本并更新`CustomAssetFactory.h`文件如下：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "Factories/Factory.h"
#include "CustomAssetFactory.generated.h"

UCLASS()
class CHAPTER_10_API UCustomAssetFactory : public UFactory
{
    GENERATED_BODY()

public:
    UCustomAssetFactory();

    virtual UObject* FactoryCreateNew(UClass* InClass,
        UObject* InParent, FName InName, EObjectFlags Flags,
        UObject* Context, FFeedbackContext* Warn, FName
        CallingContext) override;
};

```

1.  然后，切换到`CustomAssetFactory.cpp`文件并实现该类：

```cpp
#include "CustomAssetFactory.h" 
#include "Chapter_10.h"
#include "MyCustomAsset.h" 

UCustomAssetFactory::UCustomAssetFactory()
 :Super()
{
 bCreateNew = true;
 bEditAfterNew = true;
 SupportedClass = UMyCustomAsset::StaticClass();
}

UObject* UCustomAssetFactory::FactoryCreateNew(UClass*
 InClass, UObject* InParent, FName InName, EObjectFlags
 Flags, UObject* Context, FFeedbackContext* Warn, FName
 CallingContext)
{
 auto NewObjectAsset = NewObject<UMyCustomAsset>(InParent,
 InClass, InName, Flags);
 return NewObjectAsset;
}
```

1.  编译你的代码并打开编辑器。

1.  在内容浏览器中右键单击，从内容文件夹中，在创建高级资产部分的 Miscellaneous 选项卡下，你应该能看到你的新类并能够创建你新自定义类型的实例：

![图片](img/df0751b0-3259-47e3-b851-07fd9e1401ba.png)

# 它是如何工作的...

第一个类是实际可以在游戏运行时存在的对象。它是你的纹理、数据文件或曲线数据——无论你需要什么。

为了本菜谱的目的，最简单的例子是一个具有`FString`属性以包含名称的资产。

属性被标记为`UPROPERTY`，以便它保留在内存中，并且还标记为`EditAnywhere`，以便它可以在默认对象及其实例上编辑。

第二个类是`Factory`。Unreal 使用`Factory`设计模式来创建资产实例。

这意味着有一个通用的基类`Factory`，它使用虚拟方法来声明对象创建的接口，然后`Factory`子类负责创建实际的对象。

这种方法的优势在于，如果需要，用户创建的子类可以实例化其自己的子类之一；它将决定创建哪个对象的实现细节隐藏在请求创建的对象之外。

以`UFactory`作为我们的基类，我们包含适当的头文件。

构造函数被重写，因为有一些属性在我们运行默认构造函数之后想要设置给我们的新工厂。

`bCreateNew`表示工厂目前能够从头开始创建所讨论对象的全新实例。

`bEditAfterNew`表示我们希望在创建后立即编辑新创建的对象。

`SupportedClass`变量是包含工厂将要创建的对象类型的反射信息的`UClass`实例。

我们`UFactory`子类最重要的功能是实际的工厂方法——`FactoryCreateNew`。

`FactoryCreateNew`负责确定应该创建的对象类型，并使用`NewObject`来构造该类型的一个实例。它通过一系列参数传递给`NewObject`调用。

`InClass`是将要构造的对象的类。`InParent`是应该包含将要创建的新对象的那个对象。如果没有指定，该对象将被假定为进入临时包，这意味着它不会被自动保存。`Name`是要创建的对象的名称。`Flags`是一个创建标志的位掩码，它控制着诸如使对象在其包含的包外可见等事情。

在`FactoryCreateNew`中，可以做出关于应该实例化哪个子类的决定。也可以执行其他初始化；例如，如果有需要手动实例化或初始化的子对象，它们可以在这里添加。

该函数的引擎代码示例如下：

```cpp
UObject* UCameraAnimFactory::FactoryCreateNew(UClass* 
 Class,UObject* InParent,FName Name,EObjectFlags 
 Flags,UObject* Context,FFeedbackContext* Warn) 
{ 
  UCameraAnim* NewCamAnim = 
   NewObject<UCameraAnim>(InParent, Class, Name, Flags);  NewCamAnim->CameraInterpGroup = 
   NewObject<UInterpGroupCamera>(NewCamAnim); 
  NewCamAnim->CameraInterpGroup->GroupName = Name; 
  return NewCamAnim; 
} 
```

正如我们所看到的，有一个对`NewObject`的第二次调用，用于填充`NewCamAnim`实例的`CameraInterpGroup`成员。

# 参见

+   本章前面提到的*在编辑器中不同位置编辑类属性*的配方为`EditAnywhere`属性指定符提供了更多上下文。

# 为资产创建自定义上下文菜单条目

自定义资产类型通常有您希望能够在它们上执行的特殊功能。例如，将图像转换为精灵是一个您不希望添加到任何其他资产类型的选项。您可以为特定的资产类型创建自定义上下文菜单条目，使用户能够访问这些功能。

# 如何做到这一点...

1.  从`Chapter_10Editor`文件夹中，创建两个新的文件，分别命名为`MyCustomAssetActions.h`和`MyCustomAssetActions.cpp`。

1.  返回到你的项目文件并更新你的 Visual Studio 项目。完成后，在 Visual Studio 中打开项目。

1.  打开`MyCustomAssetActions.h`并使用以下代码：

```cpp
#pragma once
#include "AssetTypeActions_Base.h"
#include "Editor/MainFrame/Public/Interfaces/IMainFrameModule.h"

class CHAPTER_10EDITOR_API FMyCustomAssetActions : public FAssetTypeActions_Base
{
public:

    virtual bool HasActions(const TArray<UObject*>& InObjects)
    const override;

    virtual void GetActions(const TArray<UObject*>& InObjects,
    FMenuBuilder& MenuBuilder) override;

    virtual FText GetName() const override;

    virtual UClass* GetSupportedClass() const override;

    virtual FColor GetTypeColor() const override;

    virtual uint32 GetCategories() override;

    void MyCustomAssetContext_Clicked()
    {
        TSharedRef<SWindow> CookbookWindow = SNew(SWindow)
            .Title(FText::FromString(TEXT("Cookbook Window")))
            .ClientSize(FVector2D(800, 400))
            .SupportsMaximize(false)
            .SupportsMinimize(false);

        IMainFrameModule& MainFrameModule = 
        FModuleManager::LoadModuleChecked<IMainFrameModule>
        (TEXT("MainFrame"));

        if (MainFrameModule.GetParentWindow().IsValid())
        {
            FSlateApplication::Get().AddWindowAsNativeChild(CookbookWindow, 
            MainFrameModule.GetParentWindow().ToSharedRef());
        }
        else
        {
            FSlateApplication::Get().AddWindow(CookbookWindow);
        }

    };
};

```

1.  打开`MyCustomAssetActions.cpp`并添加以下代码：

```cpp
#include "MyCustomAssetActions.h"
#include "Chapter_10Editor.h"
#include "MyCustomAsset.h"

bool FMyCustomAssetActions::HasActions(const TArray<UObject*>& InObjects) const
{
  return true;
}

void FMyCustomAssetActions::GetActions(const TArray<UObject*>& InObjects, FMenuBuilder& MenuBuilder)
{
  MenuBuilder.AddMenuEntry(
    FText::FromString("CustomAssetAction"),
    FText::FromString("Action from Cookbook Recipe"),
    FSlateIcon(FEditorStyle::GetStyleSetName(),
    "LevelEditor.ViewOptions"),
    FUIAction(
      FExecuteAction::CreateRaw(this, 
      &FMyCustomAssetActions::MyCustomAssetContext_Clicked),
      FCanExecuteAction()
      ));
}

uint32 FMyCustomAssetActions::GetCategories()
{
  return EAssetTypeCategories::Misc;
}

FText FMyCustomAssetActions::GetName() const
{
  return FText::FromString(TEXT("My Custom Asset"));
}

UClass* FMyCustomAssetActions::GetSupportedClass() const
{
  return UMyCustomAsset::StaticClass();
}

FColor FMyCustomAssetActions::GetTypeColor() const
{
  return FColor::Emerald;
}

```

1.  打开`Chapter_10Editor.h`文件并将以下属性添加到类中：

```cpp
#pragma once

#include "Engine.h"
#include "Modules/ModuleInterface.h"
#include "Modules/ModuleManager.h"
#include "UnrealEd.h"
#include "CookbookCommands.h"
#include "Editor/MainFrame/Public/Interfaces/IMainFrameModule.h"
#include "Developer/AssetTools/Public/IAssetTypeActions.h"

 class FChapter_10EditorModule: public IModuleInterface 
{ 
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

 TArray< TSharedPtr<IAssetTypeActions> > CreatedAssetTypeActions;

    TSharedPtr<FExtender> ToolbarExtender;
    TSharedPtr<const FExtensionBase> Extension;

```

不要忘记添加`#include`对`IAssetTypeActions.h`。

1.  在你的编辑器模块（`Chapter_10Editor.cpp`）中，将以下代码添加到`StartupModule()`函数中：

```cpp
#include "Developer/AssetTools/Public/IAssetTools.h"
#include "Developer/AssetTools/Public/AssetToolsModule.h"
#include "MyCustomAssetActions.h"
// ...

void FChapter_10EditorModule::StartupModule()
{

    FCookbookCommands::Register();

    TSharedPtr<FUICommandList> CommandList = MakeShareable(new FUICommandList());

    CommandList->MapAction(FCookbookCommands::Get().MyButton, FExecuteAction::CreateRaw(this, &FChapter_10EditorModule::MyButton_Clicked), FCanExecuteAction());

    ToolbarExtender = MakeShareable(new FExtender());

    FLevelEditorModule& LevelEditorModule = FModuleManager::LoadModuleChecked<FLevelEditorModule>("LevelEditor");

 IAssetTools& AssetTools = 
    FModuleManager::LoadModuleChecked<FAssetToolsModule>
    ("AssetTools").Get();

 auto Actions = MakeShareable(new FMyCustomAssetActions);
 AssetTools.RegisterAssetTypeActions(Actions);
 CreatedAssetTypeActions.Add(Actions);

}
```

1.  在模块的`ShutdownModule()`函数内部添加以下代码：

```cpp
void FChapter_10EditorModule::ShutdownModule()
{

    ToolbarExtender->RemoveExtension(Extension.ToSharedRef());

    Extension.Reset();
    ToolbarExtender.Reset();

IAssetTools& AssetTools = FModuleManager::LoadModuleChecked<FAssetToolsModule>("Asset Tools").Get(); 

 for (auto Action : CreatedAssetTypeActions)
 {
 AssetTools.UnregisterAssetTypeActions(Action.ToSharedRef());
 }

}
```

1.  编译你的项目并启动编辑器。

1.  通过在内容浏览器中右键单击并选择 Miscellaneous | My Custom Asset 在内容浏览器中创建你的自定义资产实例。

1.  右键单击你的新资产以在上下文菜单中看到我们的自定义命令：

![图片](img/ae13de6c-8e5e-4f38-8a0e-06ad7da16286.png)

1.  选择 CustomAssetAction 命令以显示一个新的空白编辑器窗口。

# 它是如何工作的...

所有资产类型特定上下文菜单命令的基类是`FAssetTypeActions_Base`，因此我们需要从该类继承。

`FAssetTypeActions_Base`是一个抽象类，它定义了多个虚拟函数，允许我们扩展上下文菜单。包含这些虚拟函数原始信息的接口可以在`IAssetTypeActions.h`中找到。

我们还声明了一个我们绑定到自定义上下文菜单条目的函数。

`IAssetTypeActions::HasActions ( const TArray<UObject*>& InObjects )`是引擎代码调用的函数，用于查看我们的`AssetTypeActions`类是否包含可以应用于所选对象的任何动作。

`IAssetTypeActions::GetActions(const TArray<UObject*>& InObjects, class FMenuBuilder& MenuBuilder)`在`HasActions`函数返回`true`时被调用。它调用`MenuBuilder`上的函数来创建我们提供的动作的菜单选项。

`IAssetTypeActions::GetName()`返回此类的名称。

`IAssetTypeActions::GetSupportedClass()`返回我们的动作类所支持的`UClass`实例。

`IAssetTypeActions::GetTypeColor()`返回与此类和动作关联的颜色。

`IAssetTypeActions::GetCategories()`返回一个适合资产的类别。这用于更改动作在上下文菜单中显示的类别。

我们对`HasActions`的重写实现简单地在所有情况下返回`true`，并依赖于基于`GetSupportedClass`的结果进行过滤。

在`GetActions`的实现内部，我们可以调用作为函数参数给出的`MenuBuilder`对象上的某些函数。`MenuBuilder`是通过引用传递的，因此我们函数所做的任何更改在它返回后都将持续存在。

`AddMenuEntry` 函数有多个参数。第一个参数是动作本身的名称。这个名称将在上下文菜单中可见。名称是一个 `FText`，以便在需要时进行本地化。为了简化，我们从一个字符串字面量构造 `FText`，并且不考虑多语言支持。

第二个参数也是 `FText`，我们通过调用 `FText::FromString` 来构造这个参数。这个参数是当用户将鼠标悬停在命令上超过一小段时间时显示在工具提示中的文本。

下一个参数是命令的 `FSlateIcon`，它由编辑器样式集中的 `LevelEditor.ViewOptions` 图标构造。

这个函数的最后一个参数是一个 `FUIAction` 实例。`FUIAction` 是一个封装了委托绑定的包装器，因此我们使用 `FExecuteAction::CreateRaw` 将命令绑定到 `FMyCustomAssetActions` 的这个实例上的 `MyCustomAsset_Clicked` 函数。

这意味着当菜单项被点击时，我们的 `MyCustomAssetContext_Clicked` 函数将被执行。

我们对 `GetName` 的实现返回我们的资产类型的名称。如果我们没有设置一个，这个字符串将被用于我们的资产缩略图，除了在菜单部分的标题中使用外，我们的自定义资产将被放置在这个菜单部分。

如你所预期，`GetSupportedClass` 的实现返回 `UMyCustomAsset::StaticClass()`，因为我们希望我们的操作作用于这种资产类型。

`GetTypeColor()` 返回在内容浏览器中用于颜色编码的颜色——这个颜色用于资产缩略图底部的条形。我在这里使用了翡翠色，但任何任意颜色都可以工作。

这个菜谱的实际工作马是 `MyCustomAssetContext_Clicked()` 函数。

这个函数首先做的事情是创建一个 `SWindow` 的新实例。

`SWindow` 是 Slate 窗口——来自 Slate UI 框架的一个类。

Slate 小部件是通过 `SNew` 函数创建的，它返回请求的部件实例。

Slate 使用 `builder` 设计模式，这意味着所有在 `SNew` 之后 **链式** 调用的函数都返回正在操作的对象的引用。

在这个函数中，我们创建新的 `SWindow`，然后设置窗口标题、其客户端大小或区域，以及它是否可以被最大化或最小化。

在我们的新窗口准备好后，我们需要获取编辑器的根窗口的引用，以便我们可以将我们的窗口添加到层次结构中并显示它。

我们使用 `IMainFrameModule` 类来完成这项工作。它是一个模块，因此我们使用模块管理器来加载它。

`LoadModuleChecked` 如果我们无法加载模块，将断言，因此我们不需要检查它。

如果模块已加载，我们检查是否有有效的父窗口。如果该窗口有效，则使用 `FSlateApplication::AddWindowAsNativeChild` 将我们的窗口添加为顶级父窗口的子窗口。

如果我们没有顶级父级，函数将使用`AddWindow`来添加新窗口，而不会将其作为另一个窗口的子窗口添加到层次结构中。

因此，我们现在有一个类，它将在我们的自定义资产类型上显示自定义操作，但我们实际上需要做的是告诉引擎它应该询问我们的类来处理该类型的自定义操作。为了做到这一点，我们需要将我们的类与资产工具模块注册。

做这件事的最好方法是，在我们加载编辑器模块时注册我们的类，并在关闭时注销它。

因此，我们将我们的代码放入`StartupModule`和`ShutdownModule`函数中。

在`StartupModule`内部，我们使用模块管理器加载资产工具模块。

在模块加载后，我们创建一个新的共享指针，它引用我们的自定义资产操作类的一个实例。

我们需要做的就是调用`AssetModule.RegisterAssetTypeActions`并传入我们的操作类的一个实例。

我们需要存储对那个`Actions`实例的引用，以便我们可以在以后注销它。

这个配方的示例代码使用所有创建的资产操作的数组，以防我们还想为其他类添加自定义操作。

在`ShutdownModule`中，我们再次检索资产工具模块的一个实例。

使用基于范围的 for 循环，我们遍历我们之前填充的`Actions`实例数组，并调用`UnregisterAssetTypeActions`，传入我们的`Actions`类以便它可以被注销。

在我们的类注册后，编辑器已被指示询问我们的注册类是否可以处理右键单击的资产。

如果资产是自定义资产类，那么它的`StaticClass`将与`GetSupportedClass`返回的相匹配。然后编辑器将调用`GetActions`，并显示我们对该函数实现所做的更改的菜单。

当点击`CustomAssetAction`按钮时，我们的自定义`MyCustomAssetContext_Clicked`函数将通过我们创建的委托被调用。

# 创建新的控制台命令

在开发过程中，控制台命令可以通过允许开发人员或测试人员轻松绕过内容或禁用与当前正在运行的测试无关的机制而非常有帮助。实现这种功能最常见的方式是通过控制台命令，它们可以在运行时调用函数。您可以使用波浪键（`~`）或键盘数字区左上角的对等键访问控制台：

![图片](img/fc5a434d-25b5-4071-8ef7-bbbad9ac670c.jpg)

# 准备工作

如果你还没有遵循*创建新的编辑器模块*配方，请这样做，因为这个配方需要一个初始化和注册控制台命令的地方。

# 如何做到这一点...

1.  打开你的编辑器模块的头文件（`Chapter_10Editor.h`）并添加以下代码：

```cpp
class FChapter_10EditorModule: public IModuleInterface 
{ 
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

    TArray< TSharedPtr<IAssetTypeActions> > CreatedAssetTypeActions;

    TSharedPtr<FExtender> ToolbarExtender;
    TSharedPtr<const FExtensionBase> Extension;

 IConsoleCommand* DisplayTestCommand;
 IConsoleCommand* DisplayUserSpecifiedWindow;
```

1.  在`StartupModule`的实现中添加以下代码：

```cpp
DisplayTestCommand = IConsoleManager::Get().RegisterConsoleCommand(TEXT("DisplayTestCommandWindow"), TEXT("test"), FConsoleCommandDelegate::CreateRaw(this, &FChapter_10EditorModule::DisplayWindow, FString(TEXT("Test Command Window"))), ECVF_Default);

    DisplayUserSpecifiedWindow = IConsoleManager::Get().RegisterConsoleCommand(TEXT("DisplayWindow"), TEXT("test"), FConsoleCommandWithArgsDelegate::CreateLambda(
        &
    {
        FString WindowTitle;
        for (FString Arg : Args)
        {
            WindowTitle += Arg;
            WindowTitle.AppendChar(' ');
        }
        this->DisplayWindow(WindowTitle);
    }

    ), ECVF_Default);
```

1.  在`ShutdownModule`内部添加以下代码：

```cpp
if(DisplayTestCommand)
{
    IConsoleManager::Get().UnregisterConsoleObject(DisplayTestCommand);
    DisplayTestCommand = nullptr;
}

if(DisplayUserSpecifiedWindow)
{
    IConsoleManager::Get().UnregisterConsoleObject(DisplayUserSpecifiedWindow);
    DisplayUserSpecifiedWindow = nullptr;
}
```

1.  在编辑器模块（`Chapter_10Editor.h`）中实现以下函数：

```cpp
void DisplayWindow(FString WindowTitle) 
{ 
  TSharedRef<SWindow> CookbookWindow = SNew(SWindow) 
  .Title(FText::FromString(WindowTitle)) 
  .ClientSize(FVector2D(800, 400)) 
  .SupportsMaximize(false) 
  .SupportsMinimize(false); 
  IMainFrameModule& MainFrameModule = 
   FModuleManager::LoadModuleChecked<IMainFrameModule>
   (TEXT("MainFrame")); 
  if (MainFrameModule.GetParentWindow().IsValid()) 
  { 
    FSlateApplication::Get().AddWindowAsNativeChild
     (CookbookWindow, MainFrameModule.GetParentWindow()
     .ToSharedRef()); 
  } 
  else 
  { 
    FSlateApplication::Get().AddWindow(CookbookWindow); 
  } 
}
```

1.  编译你的代码并启动编辑器。

1.  玩完关卡后，按波浪键打开控制台。

1.  输入 `DisplayTestCommandWindow` 并按 *Enter*：

![图片](img/639eb05b-4ada-4f76-96ed-3e94259015d3.png)

1.  你应该能看到我们的教程窗口打开：

![图片](img/f08671c4-aad4-4102-85ab-e0906f8a10ca.jpg)

# 它是如何工作的...

控制台命令通常由一个模块提供。要使模块在加载时创建命令，最好的方法是将代码放在 `StartupModule` 方法中。

`IConsoleManager` 是包含引擎控制台功能的模块。

由于它是一个核心模块的子模块，我们不需要在构建脚本中添加任何额外的信息来链接额外的模块。

要在控制台管理器中调用函数，我们需要获取引擎正在使用的 `IConsoleManager` 当前实例的引用。为此，我们调用静态 `Get` 函数，它以类似单例的方式返回模块的引用。

`RegisterConsoleCommand` 是我们可以用来添加新的控制台命令并使其在控制台中可用的函数：

```cpp
virtual IConsoleCommand* RegisterConsoleCommand(const 
 TCHAR* Name, const TCHAR* Help, const 
 FConsoleCommandDelegate& Command, uint32 Flags);
```

函数的参数如下：

+   `Name`：用户将输入的实际控制台命令。它不应包含空格。

+   `Help`：当用户在控制台中查看命令时出现的工具提示。如果你的控制台命令接受参数，这是一个向用户显示使用信息的好地方。

+   `Command`：这是当用户输入命令时将被实际执行的函数委托。

+   `Flags`：这些标志控制了在发布构建中命令的可见性，也用于控制台变量。`ECVF_Default` 指定了默认行为，其中命令是可见的，并且在发布构建中没有可用性的限制。

要创建适当委托的实例，我们使用 `FConsoleCommand` 委托类型的 `CreateRaw` 静态函数。这使我们能够将原始 C++ 函数绑定到委托。在函数引用之后提供的额外参数，`FString` `"Test Command Window"`，是一个编译时定义的参数，传递给委托，这样最终用户就不需要指定窗口名称。

第二个控制台命令，`DisplayUserSpecifiedWindow`，展示了如何使用控制台命令的参数。

与此控制台命令的主要区别，除了用户调用它的名称不同之外，还在于特别使用了 `FConsoleCommandWithArgsDelegate` 和其上的 `CreateLambda` 函数。

此函数允许我们将匿名函数绑定到委托。当你想要包装或适配一个函数，使其签名与特定委托匹配时，这特别有用。

在我们的特定用例中，`FConsoleCommandWithArgsDelegate` 的类型指定了该函数应接受一个 `const TArray` 的 `FStrings`。我们的 `DisplayWindow` 函数接受一个 `FString` 参数来指定窗口标题，因此我们需要以某种方式将控制台命令的所有参数连接成一个单一的 `FString`，用作我们的窗口标题。

Lambda 函数允许我们在将 `FString` 传递给实际的 `DisplayWindow` 函数之前完成这一操作。

函数的第一行 `&` 指定这个 lambda 或匿名函数想要通过引用捕获声明函数的上下文，通过在捕获选项中包含 ampersand 实现，`[&]`。

第二部分与正常函数声明相同，指定我们的 lambda 接受一个 `const Tarray`，它包含一个名为 `Args` 的 `FString` 参数。

在 lambda 体内部，我们创建一个新的 `FString` 并将构成我们参数的字符串连接在一起，在它们之间添加一个空格以分隔它们，这样我们就不得到没有空格的标题。

它使用基于范围的 `for` 循环来简化代码，以便遍历所有项并执行连接操作。

一旦它们全部连接起来，我们使用 `this` 指针（由我们之前提到的 `&` 运算符捕获）调用 `DisplayWindow` 并使用我们的新标题。

为了使我们的模块在卸载时移除控制台命令，我们需要保持对控制台命令对象的引用。

为了实现这一点，我们在模块中创建了一个成员变量，类型为 `IConsoleCommand*`，名为 `DisplayTestCommand`。当我们执行 `RegisterConsoleCommand` 函数时，它返回一个指向控制台命令对象的指针，我们可以将其用作后续的句柄。

这允许我们在运行时根据游戏玩法或其他因素启用或禁用控制台命令。

在 `ShutdownModule` 中，我们检查 `DisplayTestCommand` 是否指向一个有效的控制台命令对象。如果是，我们就获取 `IConsoleManager` 对象的引用并调用 `UnregisterConsoleCommand`，传入我们在调用 `RegisterConsoleCommand` 时之前存储的指针。

`UnregisterConsoleCommand` 的调用通过传入的指针删除 `IConsoleCommand` 实例，因此我们不需要自己 `deallocate` 内存 - 我们只需将 `DisplayTestCommand` 重置为 `nullptr`，这样我们就可以确保旧的指针不会悬空。

`DisplayWindow` 函数接受一个 `FString` 参数作为窗口标题。这允许我们使用接受参数来指定标题的控制台命令，或者使用有效载荷参数来为其他命令硬编码标题的控制台命令。

该函数本身使用一个名为 `SNew()` 的函数来分配和创建一个 `SWindow` 对象。

`SWindow` 是一个 Slate 窗口，一个使用 Slate UI 框架的最高级窗口。

Slate 使用 `Builder` 设计模式来允许轻松配置新窗口。

这里使用的`Title`、`ClientSize`、`SupportsMaximize`和`SupportsMinimize`函数都是`SWindow`的成员函数，并返回一个`SWindow`的引用（通常，是方法被调用的同一个对象，但有时会使用新的配置构造一个新对象）。

事实上，所有这些成员方法都返回配置对象的引用，这使得我们可以将这些方法调用串联起来，以正确的配置创建所需的对象。

在`DisplayWindow`中使用的函数创建了一个具有基于函数参数标题的新顶级窗口。它宽 800 x 400 像素，不能最大化或最小化。

在创建我们的新窗口后，我们检索主应用程序框架模块的引用。如果编辑器的顶级窗口存在且有效，我们将我们的新窗口实例添加为该顶级窗口的子窗口。

要做到这一点，我们检索 Slate 接口的引用并调用`AddWindowAsNativeChild`将我们的窗口插入到层次结构中。

如果没有有效的顶级窗口，我们不需要将我们的新窗口作为任何内容的子窗口添加，因此我们可以简单地调用`AddWindow`并传入我们的新窗口实例。

# 参见

+   参考第五章处理事件和委托，了解更多关于委托的信息。它更详细地解释了有效载荷变量。

+   更多关于 Slate 的信息，请参考第十一章使用 UE4 API 工作。

# 为蓝图创建新的图形引脚可视化器

在蓝图系统中，我们可以将我们的`MyCustomAsset`类的实例用作变量，前提是我们将其标记为`BlueprintType`，在它的`UCLASS`宏中。然而，默认情况下，我们的新资产被简单地视为`UObject`，我们无法访问其任何成员：

![图片](img/71c51dd0-1db5-4ab5-9c9e-45a433e1f2fd.jpg)

对于某些类型的资产，我们可能希望启用与`FVector`等类支持的类似方式的内联编辑字面值：

![图片](img/d2f07c2a-77ee-4333-aec6-47facbe0b76f.jpg)

要启用此功能，我们需要使用一个**图形引脚**可视化器。这个菜谱将向您展示如何使用您定义的自定义小部件启用任意类型的内联编辑。

# 如何做到这一点...

1.  首先，我们将更新`MyCustomAsset`类，使其在蓝图编辑器中可编辑，并反映我们将在这个菜谱中执行的操作。前往`MyCustomAsset.h`并更新为以下代码：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "MyCustomAsset.generated.h"

UCLASS(BlueprintType, EditInlineNew)
class CHAPTER_10_API UMyCustomAsset : public UObject
{
  GENERATED_BODY()

public:
    UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "Custom Asset")
    FString ColorName;

};
```

1.  从`Chapter_10Editor`文件夹中，创建一个名为`MyCustomAssetPinFactory.h`的新文件。

1.  在头文件内部，添加以下代码：

```cpp
#pragma once
#include "EdGraphUtilities.h"
#include "MyCustomAsset.h"
#include "SGraphPinCustomAsset.h"

struct CHAPTER_10EDITOR_API FMyCustomAssetPinFactory : public FGraphPanelPinFactory
{
public:
  virtual TSharedPtr<class SGraphPin> CreatePin(class UEdGraphPin* Pin) const override 
  {
    if (Pin->PinType.PinSubCategoryObject == UMyCustomAsset::StaticClass())
    {
      return SNew(SGraphPinCustomAsset, Pin);
    }
    else
    {
      return nullptr;
    }
  };
};

```

1.  创建另一个名为`SGraphPinCustomAsset.h`的头部文件：

```cpp
#pragma once
#include "SGraphPin.h"

class CHAPTER_10EDITOR_API SGraphPinCustomAsset : public SGraphPin
{
  SLATE_BEGIN_ARGS(SGraphPinCustomAsset) {}
  SLATE_END_ARGS()

  void Construct(const FArguments& InArgs, UEdGraphPin* InPin);
protected:
  virtual FSlateColor GetPinColor() const override { return FSlateColor(FColor::Black); };

  virtual TSharedRef<SWidget> GetDefaultValueWidget() override;

  void ColorPicked(FLinearColor SelectedColor);
};

```

1.  通过创建`.cpp`文件实现`SGraphPinCustomAsset`：

```cpp
#include "SGraphPinCustomAsset.h"
#include "Chapter_10Editor.h"
#include "SColorPicker.h"
#include "MyCustomAsset.h"

void SGraphPinCustomAsset::Construct(const FArguments& InArgs, UEdGraphPin* InPin)
{
  SGraphPin::Construct(SGraphPin::FArguments(), InPin);
}

TSharedRef<SWidget> SGraphPinCustomAsset::GetDefaultValueWidget()
{
  return SNew(SColorPicker)
    .OnColorCommitted(this, &SGraphPinCustomAsset::ColorPicked);

}

void SGraphPinCustomAsset::ColorPicked(FLinearColor SelectedColor)
{
  UMyCustomAsset* NewValue = NewObject<UMyCustomAsset>();
  NewValue->ColorName = SelectedColor.ToFColor(false).ToHex();
  GraphPinObj->GetSchema()->TrySetDefaultObject(*GraphPinObj, NewValue);
}
```

1.  重新生成您的 Visual Studio 项目。

1.  将`#include "MyCustomAssetPinFactory.h"`添加到`Chapter_10Editor.h`模块实现文件中。

1.  将以下成员添加到编辑模块类（`FChapter_10EditorModule`）中：

```cpp
TSharedPtr<FMyCustomAssetPinFactory> PinFactory; 
```

1.  打开`Chapter_10Editor.cpp`，然后在`StartupModule()`中添加以下代码：

```cpp
PinFactory = MakeShareable(new FMyCustomAssetPinFactory()); 
FEdGraphUtilities::RegisterVisualPinFactory(PinFactory); 
```

1.  还需要在`ShutdownModule()`中添加以下代码：

```cpp
FEdGraphUtilities::UnregisterVisualPinFactory(PinFactory); PinFactory.Reset(); 
```

1.  编译你的代码并启动编辑器。

1.  在“我的蓝图”面板中点击“函数”旁边的加号符号，在层级蓝图内创建一个新的`Function`：

![图片](img/c9415a2e-d1eb-406e-b064-ae473029c34d.png)

1.  添加一个输入参数：

![图片](img/d9209b56-9fbb-400f-b97a-42fa2ba9a246.png)

1.  将其类型设置为`MyCustomAsset`（对象引用）：

![图片](img/f5d2f5e2-1f18-45ea-b6bd-31ee1db119c9.png)

1.  在层级蓝图的事件图中放置你新函数的一个实例，并验证输入引脚现在是否有一个以颜色选择器形式的自定义可视化器：

![图片](img/779d7694-622f-4cd8-94ec-acdf6fffbc5c.png)

新增的颜色选择器可视化器

# 它是如何工作的...

使用`FGraphPanelPinFactory`类来定制对象在蓝图引脚上作为字面值出现的方式。

这个类定义了一个单独的虚函数：

```cpp
virtual TSharedPtr<class SGraphPin> CreatePin(class 
 UEdGraphPin* Pin) const
```

如其名所示，`CreatePin`函数的功能是创建图引脚的新视觉表示。

它接收一个`UEdGraphPin`实例。`UEdGraphPin`包含关于引脚所表示的对象的信息，以便我们的工厂类可以做出明智的决定，关于我们应该显示哪种视觉表示。

在我们的函数实现中，我们检查引脚的类型是否是我们自定义类。

我们通过查看`PinSubCategoryObject`属性，它包含一个`UClass`，并将其与我们的自定义资产类关联的`UClass`进行比较来做这件事。

如果引脚的类型符合我们的条件，我们返回一个新的共享指针到 Slate Widget，这是我们对象的视觉表示。

如果引脚类型不正确，我们返回一个空指针以指示失败状态。

下一个类`SGraphPinCustomAsset`是 Slate Widget 类，它是我们对象作为字面值的视觉表示。

它继承自`SGraphPin`，这是所有图引脚的基类。

`SGraphPinCustomAsset`类有一个`Construct`函数，当创建小部件时被调用。

它还实现了父类的一些函数：`GetPinColor()`和`GetDefaultValueWidget()`。

最后一个定义的函数是`ColorPicked`，它是当用户在我们的自定义引脚中选择颜色时的处理程序。

在我们自定义类的实现中，我们通过调用`Construct`的默认实现来初始化我们的自定义引脚。

`GetDefaultValueWidget`的作用实际上是创建我们类的自定义表示的控件，并将其返回给引擎代码。

在我们的实现中，它创建了一个新的`SColorPicker`实例——我们希望用户能够选择一种颜色，并将该颜色的十六进制表示存储在我们自定义类的`FString`属性中。

这个`SColorPicker`实例有一个名为`OnColorCommitted`的属性——这是一个可以分配给对象实例上函数的 slate 事件。

在返回我们的新`SColorPicker`之前，我们将`OnColorCommitted`链接到当前对象上的`ColorPicked`函数，以便如果用户选择新的颜色，它将被调用。

`ColorPicked`函数接收所选颜色作为输入参数。

因为当没有对象连接到我们关联的引脚时使用此小部件，所以我们不能简单地将关联对象的属性设置为所需的颜色字符串。

我们需要创建我们自定义资产类的新实例，这是通过使用`NewObject`模板函数来实现的。

此函数的行为类似于我们在其他章节中讨论的`SpawnActor`函数，并在返回指针之前初始化指定类的新实例。

拥有新实例后，我们可以设置其`ColorName`属性。`FLinearColors`可以转换为`FColor`对象，这些对象定义了一个`ToHex()`函数，该函数返回一个包含所选颜色的十六进制表示的`FString`。

最后，我们需要将我们的新对象实例实际放置在图中，以便在图执行时被引用。

要实现这一点，我们需要访问我们代表的图引脚对象，并使用`GetSchema`函数。此函数返回拥有包含我们的引脚的节点的图的 Schema。

模式包含与图引脚对应的实际值，并且在图评估期间是一个关键元素。

现在我们已经可以访问模式，我们可以设置我们的小部件所代表的引脚的默认值。如果引脚未连接到另一个引脚，则此值将在图评估期间使用，并像在 C++函数定义期间提供的默认值一样起作用。

就像我们在本章中做出的所有扩展一样，必须有一些初始化或注册，告诉引擎在使用其默认内置表示之前先使用我们的自定义实现。

要做到这一点，我们需要向我们的编辑器模块添加一个新成员来存储我们的`PinFactory`类实例。

在`StartupModule`期间，我们创建一个新的共享指针，它引用我们的`PinFactory`类的一个实例。

我们将其存储在编辑器模块的成员中，以便以后可以注销。然后，我们调用`FEdGraphUtilities::RegisterVisualPinFactory(PinFactory)`来告诉引擎使用我们的`PinFactory`来创建视觉表示。

在`ShutdownModule`期间，我们使用`UnregisterVisualPinFactory`注销引脚工厂。

最后，我们通过在包含它的共享指针上调用`Reset()`来删除我们的旧的`PinFactory`实例。

# 使用自定义详细面板检查类型

默认情况下，由`UObject`派生的 UAssets 将在通用属性编辑器中打开。它看起来如下：

![](img/1cd21451-0090-40c5-b897-5525655f98ee.png)

然而，有时你可能希望有自定义小部件，以便你可以编辑你类上的属性。为此，Unreal 支持**详细定制**，这是本菜谱的重点。

# 如何做...

1.  从 `Chapter_10Editor` 文件夹中，创建两个名为 `MyCustomAssetDetailsCustomization.h` 和 `MyCustomAssetDetailsCustomization.cpp` 的新文件。

1.  返回您的项目文件并更新您的 Visual Studio 项目。完成后，在 Visual Studio 中打开项目。

1.  将以下 `#pragma` 和 `#includes` 添加到头文件 (`MyCustomAssetDetailsCustomization.h`)：

```cpp
#pragma once

#include "MyCustomAsset.h" 
#include "DetailLayoutBuilder.h" 
#include "IDetailCustomization.h" 
#include "IPropertyTypeCustomization.h" 
```

1.  按照以下方式定义我们的自定义类：

```cpp
class FMyCustomAssetDetailsCustomization : public IDetailCustomization
{

public:
    virtual void CustomizeDetails(IDetailLayoutBuilder& DetailBuilder) override;

    void ColorPicked(FLinearColor SelectedColor);

    static TSharedRef<IDetailCustomization> MakeInstance()
    {
        return MakeShareable(new FMyCustomAssetDetailsCustomization);
    }

    TWeakObjectPtr<class UMyCustomAsset> MyAsset;
};
```

1.  在下面，定义以下附加类：

```cpp
class FMyCustomAssetPropertyDetails : public IPropertyTypeCustomization
{
public:
  void ColorPicked(FLinearColor SelectedColor);
  static TSharedRef<IPropertyTypeCustomization> MakeInstance()
  {
    return MakeShareable(new FMyCustomAssetPropertyDetails);
  }

  UMyCustomAsset* MyAsset;
  virtual void CustomizeChildren(TSharedRef<IPropertyHandle> PropertyHandle, IDetailChildrenBuilder& ChildBuilder, IPropertyTypeCustomizationUtils& CustomizationUtils) override;

  virtual void CustomizeHeader(TSharedRef<IPropertyHandle> PropertyHandle, FDetailWidgetRow& HeaderRow, IPropertyTypeCustomizationUtils& CustomizationUtils) override;

};
```

1.  在实现文件中，在文件顶部添加以下包含：

```cpp
#include "MyCustomAssetDetailsCustomization.h" 
#include "Chapter_10Editor.h" 
#include "IDetailsView.h" 
#include "DetailLayoutBuilder.h" 
#include "DetailCategoryBuilder.h" 
#include "SColorPicker.h" 
#include "SBoxPanel.h" 
#include "DetailWidgetRow.h" 
```

1.  之后，为 `CustomizeDetails` 创建一个实现：

```cpp
void FMyCustomAssetDetailsCustomization::CustomizeDetails(IDetailLayoutBuilder& DetailBuilder)
{
    const TArray< TWeakObjectPtr<UObject> >& SelectedObjects = DetailBuilder.GetDetailsView()->GetSelectedObjects();

    for (int32 ObjectIndex = 0; !MyAsset.IsValid() && ObjectIndex < SelectedObjects.Num(); ++ObjectIndex)
    {
        const TWeakObjectPtr<UObject>& CurrentObject = SelectedObjects[ObjectIndex];
        if (CurrentObject.IsValid())
        {
            MyAsset = Cast<UMyCustomAsset>(CurrentObject.Get());
        }
    }

    DetailBuilder.EditCategory("CustomCategory", FText::GetEmpty(), ECategoryPriority::Important)
.AddCustomRow(FText::GetEmpty())
    [
    SNew(SVerticalBox)
    + SVerticalBox::Slot()
    .VAlign(VAlign_Center)
        [
            SNew(SColorPicker)
            .OnColorCommitted(this, &FMyCustomAssetDetailsCustomization::ColorPicked)
        ]
    ];
}
```

1.  此外，为 `ColorPicked` 创建一个定义：

```cpp
void FMyCustomAssetDetailsCustomization::ColorPicked(FLinearColor SelectedColor)
{
    if (MyAsset.IsValid())
    {
        MyAsset.Get()->ColorName = SelectedColor.ToFColor(false).ToHex();
    }
}
```

1.  在 `MyCustomAssetDetailsCustomization.cpp` 中的所有脚本下方，添加以下代码：

```cpp
void FMyCustomAssetPropertyDetails::CustomizeChildren(TSharedRef<IPropertyHandle> PropertyHandle, IDetailChildrenBuilder& ChildBuilder, IPropertyTypeCustomizationUtils& CustomizationUtils)
{
}

void FMyCustomAssetPropertyDetails::CustomizeHeader(TSharedRef<IPropertyHandle> PropertyHandle, FDetailWidgetRow& HeaderRow, IPropertyTypeCustomizationUtils& CustomizationUtils)
{
    UObject* PropertyValue = nullptr;
    auto GetValueResult = PropertyHandle->GetValue(PropertyValue);

    HeaderRow.NameContent()
        [
            PropertyHandle->CreatePropertyNameWidget()
        ];
    HeaderRow.ValueContent()
        [
            SNew(SVerticalBox)
            + SVerticalBox::Slot()
        .VAlign(VAlign_Center)
        [
            SNew(SColorPicker)
            .OnColorCommitted(this, &FMyCustomAssetPropertyDetails::ColorPicked)
        ]
        ];
}

void FMyCustomAssetPropertyDetails::ColorPicked(FLinearColor SelectedColor)
{
    if (MyAsset)
    {
        MyAsset->ColorName = SelectedColor.ToFColor(false).ToHex();
    }
}
```

1.  在我们的编辑器模块源文件 (`Chapter_10Editor.cpp`) 中，将以下内容添加到 `Chapter_10Editor.cpp` 文件中的 `#includes`：

```cpp
#include "PropertyEditorModule.h" 
#include "MyCustomAssetDetailsCustomization.h"
#include "MyCustomAssetPinFactory.h"
```

1.  将以下内容添加到 `StartupModule` 的实现中：

```cpp
FPropertyEditorModule& PropertyModule = FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");
PropertyModule.RegisterCustomClassLayout(UMyCustomAsset::StaticClass()->GetFName(), FOnGetDetailCustomizationInstance::CreateStatic(&FMyCustomAssetDetailsCustomization::MakeInstance));
PropertyModule.RegisterCustomPropertyTypeLayout(UMyCustomAsset::StaticClass()->GetFName(), FOnGetPropertyTypeCustomizationInstance::CreateStatic(&FMyCustomAssetPropertyDetails::MakeInstance));
```

1.  将以下内容添加到 `ShutdownModule`：

```cpp
FPropertyEditorModule& PropertyModule = FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor");
PropertyModule.UnregisterCustomClassLayout(UMyCustomAsset::StaticClass()->GetFName());
```

1.  编译您的代码并启动编辑器。通过内容浏览器创建 `MyCustomAsset` 的新实例。

1.  双击它以验证现在出现的默认编辑器是否显示您的自定义布局：

![图片](img/27a2b2cf-7c4b-45ba-a188-6e3894e6c8e3.png)

# 它是如何工作的...

通过 `IDetailCustomization` 接口执行细节自定义，开发人员可以在定义自定义显示特定类资产方式的类时继承它。

`IDetailCustomization` 主要使用以下功能来允许此过程发生：

```cpp
virtual void CustomizeDetails(IDetailLayoutBuilder& DetailBuilder) override;
```

在我们此函数的实现中，我们使用 `DetailBuilder` 上作为参数传递的方法来获取所有选中对象的数组。然后循环扫描这些对象以确保至少有一个选中对象是正确的类型。

通过在 `DetailBuilder` 对象上调用方法来自定义类的表示。我们使用 `EditCategory` 函数创建一个新类别以用于我们的细节视图。

`EditCategory` 函数的第一个参数是我们将要操作的类别名称。

第二个参数是可选的，包含一个可能本地化的类别显示名称。

第三个参数是类别的优先级。优先级越高，它显示在列表中的位置越靠前。

`EditCategory` 返回对类别的引用作为 `CategoryBuilder`，允许我们将额外的方法调用链接到 `EditCategory` 的调用上。

因此，我们在 `CategoryBuilder` 上调用 `AddCustomRow()`，这将在类别中添加一个新键值对以显示。

使用 Slate 语法，我们指定该行将包含一个包含单个居中对齐槽的垂直框。

在槽内，我们创建一个颜色选择控件并将其 `OnColorCommitted` 代理绑定到我们本地的 `ColorPicked` 事件处理器。

当然，这需要我们定义和实现 `ColourPicked`。它具有以下签名：

```cpp
void FMyCustomAssetDetailsCustomization::ColorPicked(FLinearColor SelectedColor)
```

在`ColorPicked`的实现中，我们检查是否我们选择的资产中有一个是正确的类型，因为如果至少有一个选定的资产是正确的，那么`MyAsset`将被填充为一个有效的值。

假设我们有一个有效的资产，我们将`ColorName`属性设置为用户所选颜色的十六进制字符串值。
