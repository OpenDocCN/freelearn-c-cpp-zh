# 用户界面 - UI 和 UMG

在本章中，我们将介绍以下食谱：

+   使用画布绘制

+   将 Slate 小部件添加到屏幕上

+   为 UI 创建屏幕尺寸感知缩放

+   在游戏中显示和隐藏 UMG 元素

+   将函数调用附加到 Slate 事件

+   使用数据绑定与 Unreal Motion Graphics

+   使用样式控制小部件外观

+   创建自定义 SWidget/UWidget

# 简介

向玩家显示反馈是游戏设计中最重要的元素之一，这通常涉及游戏中的某种 HUD 或至少是菜单。

在 Unreal 的早期版本中，有简单的 HUD 支持，允许你在屏幕上绘制简单的形状和文本。然而，在美学方面，它有些受限，因此像**Scaleform**这样的解决方案变得很常见，以克服这些限制。Scaleform 利用 Adobe 的 Flash 文件格式来存储矢量图像和 UI 脚本。尽管如此，它对开发者来说也有自己的缺点，尤其是成本——它是一个需要（有时昂贵的）许可证的第三方产品。

因此，Epic 为 Unreal 4 编辑器和游戏内 UI 框架开发了 Slate。Slate 是一组小部件（UI 元素）和框架，它允许编辑器实现跨平台界面。它也可以在游戏中使用，用于绘制小部件，例如滑块和按钮，用于菜单和 HUD。

Slate 使用声明性语法，允许在原生 C++中用 XML 风格的表示来表示用户界面元素及其层次结构。它是通过大量使用宏和运算符重载来实现的。

话虽如此，并不是每个人都想要求程序员设计游戏的 HUD。在 Unreal 3 中使用 Scaleform 的一个显著优势是能够使用 Flash 可视化编辑器开发游戏 UI 的视觉外观，这样视觉设计师就不需要学习编程语言。然后程序员可以分别插入逻辑和数据。例如，这与**Windows Presentation Framework**（**WPF**）所倡导的范式相同。

以类似的方式，Unreal 提供了**Unreal Motion Graphics**（**UMG**）。UMG 是一个用于 Slate 小部件的可视化编辑器，允许你可视化地设计、布局和动画化用户界面。UI 小部件（或控件，如果你来自 Win32 背景）可以通过蓝图代码（在 UMG 窗口的图形视图中编写）或 C++来控制其属性。本章主要涉及显示 UI 元素、创建小部件层次结构和创建可以在 UMG 中样式化和使用的基`SWidget`类。

# 技术要求

本章需要使用 Unreal Engine 4，并使用 Visual Studio 2017 作为 IDE。有关如何安装这两款软件及其要求的信息，请参阅第一章，*UE4 开发工具*。

# 使用画布绘制

**Canvas** 是 Unreal 3 中实现的简单 HUD 的延续。虽然它不太常用于发布的游戏中，大多数情况下被 Slate/UMG 替换，但它使用简单，尤其是在您想要在屏幕上绘制文本或形状时。Canvas 绘图仍然被用于调试和性能分析的控制台命令中，例如 `stat game` 和其他 `stat` 命令。

# 准备中...

如果需要复习使用 C++ 代码向导，请参阅第四章，*演员和组件*。

# 如何做到这一点...

1.  从您的 Visual Studio 项目（文件 | 打开 Visual Studio），打开 `Source\<Module>` 文件夹，然后从那里打开 `<Module>.build.cs` 文件（在我的情况下，将是 `Source\Chapter_14\Chapter_14.build.cs`）。取消注释/添加以下代码行：

```cpp
using UnrealBuildTool;

public class Chapter_14 : ModuleRules
{
  public Chapter_14(ReadOnlyTargetRules Target) : base(Target)
  {
    PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

    PublicDependencyModuleNames.AddRange(new string[] { "Core", 
    "CoreUObject", "Engine", "InputCore" });

    PrivateDependencyModuleNames.AddRange(new string[] { });

    // Uncomment if you are using Slate UI
    PrivateDependencyModuleNames.AddRange(new string[] { "Slate", 
    "SlateCore" });

    // Uncomment if you are using online features
    // PrivateDependencyModuleNames.Add("OnlineSubsystem");

    // To include OnlineSubsystemSteam, add it to the plugins section in your uproject file with the Enabled attribute set to true
  }
}
```

1.  使用编辑器类向导创建一个新的 `GameModeBase`，命名为 `CustomHUDGameMode`。

1.  向类中添加一个构造函数：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "CustomHUDGameMode.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_14_API ACustomHUDGameMode : public AGameModeBase
{
    GENERATED_BODY()

    ACustomHUDGameMode();

};
```

1.  在构造函数实现中添加以下内容：

```cpp
#include "CustomHUDGameMode.h"
#include "CustomHUD.h"

ACustomHUDGameMode::ACustomHUDGameMode() : AGameModeBase()
{
    HUDClass = ACustomHUD::StaticClass();
}
```

到目前为止，您将得到编译错误，因为 `CustomHUD` 类不存在。这正是我们将要创建的。

1.  使用添加 C++ 类向导创建一个新的 `HUD` 子类：

![图片](img/7830c68f-9f06-44ce-aedd-20e29710145d.png)

1.  当被要求输入名称时，输入`CustomHUD`，然后点击创建类按钮：

![图片](img/279fff54-150b-4019-b6c1-fc9d9f07b0a0.png)

1.  在 `CustomHUD.h` 中，添加以下带有 `override` 关键字的功能到类中：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/HUD.h"
#include "CustomHUD.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_14_API ACustomHUD : public AHUD
{
    GENERATED_BODY()

public:
 virtual void DrawHUD() override;

};
```

1.  现在，实现该函数：

```cpp
#include "CustomHUD.h"
#include "Engine/Canvas.h"

void ACustomHUD::DrawHUD()
{
 Super::DrawHUD();
 Canvas->DrawText(GEngine->GetSmallFont(), TEXT("Test string to be printed to screen"), 10, 10); 
 FCanvasBoxItem ProgressBar(FVector2D(5, 25), FVector2D(100, 5));
 Canvas->DrawItem(ProgressBar);
 DrawRect(FLinearColor::Blue, 5, 25, 100, 5);
}
```

1.  编译您的代码并启动编辑器。

1.  在编辑器中，从设置下拉菜单中打开世界设置面板：

![图片](img/d725eff7-9368-40b6-a5da-84695df4ad25.jpg)

1.  在世界设置对话框中，从游戏模式覆盖下选择 `CustomHUDGameMode`：

![图片](img/5ff91259-600d-482f-b002-6497a6fae2ee.jpg)

1.  播放并验证您的自定义 HUD 是否已绘制到屏幕上：

![图片](img/a15e2c53-27d0-40fe-bd70-09c9464d1827.png)

# 它是如何工作的...

这里所有的 UI 菜单都将使用 Slate 进行绘制，因此我们需要在我们的模块和 Slate 框架之间添加一个依赖关系，以便我们可以访问在该模块中声明的类。为游戏 HUD 添加自定义 Canvas 绘制调用的最佳位置是在 `AHUD` 的子类中。

然而，要告诉引擎使用我们的自定义子类，我们需要创建一个新的 `GameMode` 并指定我们自定义类的类型。

在我们自定义游戏模式的构造函数中，我们将我们新的 HUD 类型的 `UClass` 分配给 `HUDClass` 变量。这个 `UClass` 在每个玩家生成时传递给玩家的控制器，控制器随后负责创建 `AHUD` 实例。

由于我们的自定义 `GameMode` 正在加载我们的自定义 HUD，因此我们需要实际创建这个自定义 HUD 类。`AHUD` 定义了一个名为 `DrawHUD()` 的虚拟函数，它在每一帧被调用，以便我们可以将元素绘制到屏幕上。因此，我们重写该函数并在实现中进行绘制。

使用的第一种方法如下：

```cpp
float DrawText(constUFont* InFont, constFString&InText, 
 float X, float Y, float XScale = 1.f, float YScale = 1.f, 
 constFFontRenderInfo&RenderInfo = FFontRenderInfo());
```

`DrawText`需要一个字体来绘制。引擎代码中`stat`和其他 HUD 绘制命令使用的默认字体实际上存储在`GEngine`类中，可以通过使用`GetSmallFont`函数访问，该函数返回一个指向`UFont`实例的指针。

我们正在使用的其余参数是应该渲染的实际文本，以及文本应该绘制的像素偏移量。

`DrawText`是一个允许您直接传递要显示的数据的函数。通用的`DrawItem`函数是一个访问者实现，允许您创建一个封装要绘制对象信息的对象，并在多个绘制调用中重用该对象。

在这个配方中，我们创建了一个可以用来表示进度条的元素。我们将有关我们框的宽度和高度所需的信息封装到一个`FCanvasBoxItem`中，然后将其传递到我们的画布上的`DrawItem`函数。

我们绘制的第三项是一个填充的矩形。此函数使用在 HUD 类中定义的便利方法，而不是在画布本身上。填充的矩形放置在与我们的`FCanvasBox`相同的位置，以便它可以表示进度条中的当前值。

# 参见...

+   请参考第十章，*整合 C++和虚幻编辑器——第二部分*，以及其中的*创建新的控制台命令*配方，了解如何创建您自己的控制台命令

# 向屏幕添加 Slate 小部件

之前的配方使用了`FCanvas` API 来绘制到屏幕上。然而，`FCanvas`存在一些限制，例如，动画难以实现，在屏幕上绘制图形涉及创建纹理或材质。`FCanvas`也没有实现任何形式的控件或窗口控制，这使得数据输入或其他形式的用户输入比必要的更复杂。本配方将向您展示如何开始使用 Slate 在屏幕上创建 HUD 元素，Slate 提供了一些内置控件。

# 准备工作

如果尚未这样做，请将`Slate`和`SlateCore`添加到您的模块依赖项中（参见*使用画布绘制*配方了解如何这样做）。

# 如何做到这一点...

1.  使用添加 C++类向导创建一个新的`PlayerController`子类：

![图片](img/7c52bf2e-a859-4474-8c1d-9cd54069694e.png)

1.  当被要求输入类名时，键入`CustomHUDPlayerController`并按创建类按钮：

![图片](img/1ad35366-16d0-413a-bbb0-a5635ead359a.png)

1.  在您的新子类中重写`BeginPlay`虚拟方法：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/PlayerController.h"
#include "CustomHUDPlayerController.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_14_API ACustomHUDPlayerController : public APlayerController
{
    GENERATED_BODY()

public:
 virtual void BeginPlay() override;

};
```

1.  在子类的实现中添加以下代码以覆盖`BeginPlay()`虚拟方法：

```cpp
#include "CustomHUDPlayerController.h"
#include "SlateBasics.h"

void ACustomHUDPlayerController::BeginPlay()
{
 Super::BeginPlay();
 TSharedRef<SVerticalBox> widget = SNew(SVerticalBox)
 + SVerticalBox::Slot()
 .HAlign(HAlign_Center)
 .VAlign(VAlign_Center)
 [
 SNew(SButton)
 .Content()
 [
 SNew(STextBlock)
 .Text(FText::FromString(TEXT("Test button")))
 ]
 ];
 GEngine->GameViewport->AddViewportWidgetForPlayer(GetLocalPlayer(), widget, 1);
}
```

1.  基于名为`GameModeBase`的新类创建一个名为`SlateHUDGameMode`的新类。

1.  在游戏模式中添加一个构造函数：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "SlateHUDGameMode.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_14_API ASlateHUDGameMode : public AGameModeBase
{
    GENERATED_BODY()

 ASlateHUDGameMode();
};
```

1.  使用以下代码实现构造函数：

```cpp
#include "SlateHUDGameMode.h"
#include "CustomHUDPlayerController.h"

ASlateHUDGameMode::ASlateHUDGameMode() : Super()
{
 PlayerControllerClass = ACustomHUDPlayerController::StaticClass();
}
```

1.  在编辑器中，通过转到设置 | 世界设置，从工具栏打开“世界设置”菜单：

![图片](img/826cba58-111f-40cf-8ff5-4d6c56bf5a45.jpg)

1.  在“世界设置”中，覆盖级别的游戏模式为我们的`SlateHUDGameMode`：

![图片](img/9f73bfe4-1610-4d0c-b44b-b14e786dbbc7.jpg)

1.  播放级别。您将在屏幕上看到您的新 UI：

![图片](img/2347b898-a48c-495d-84ce-d06b195e3021.png)

位于游戏屏幕上的按钮

# 它是如何工作的...

为了在我们的代码中引用 Slate 类或函数，我们的模块必须与`Slate`和`SlateCore`模块链接，因此我们将这些添加到模块依赖项中。

我们需要在游戏运行时加载的类中实例化我们的 UI，因此对于这个配方，我们在`BeginPlay`函数中使用我们的自定义`PlayerController`作为创建 UI 的位置。

在`BeginPlay`实现中，我们使用`SNew`函数创建一个新的`SVerticalBox`。我们向我们的框添加一个用于小部件的槽，并将该槽设置为水平和垂直居中。

在我们使用方括号访问的槽中，我们创建了一个带有`Textblock`的按钮。在`Textblock`中，我们将`Text`属性设置为字符串字面值。

现在 UI 已经创建，我们调用`AddViewportWidgetForPlayer`来在本地玩家的屏幕上显示此小部件。

我们的定制`PlayerController`准备好了，我们现在需要创建一个定制的`GameMode`来指定它应该使用我们新的`PlayerController`。由于自定义`PlayerController`在游戏开始时加载，当调用`BeginPlay`时，我们的 UI 将显示出来。

在这个屏幕尺寸下，UI 非常小。请参考以下配方以获取有关如何根据游戏窗口的分辨率适当缩放的信息。

# 为 UI 创建屏幕尺寸感知的缩放

如果您已经遵循了前面的配方，您会注意到当您使用**在编辑器中播放**时，加载的按钮异常小。

原因是 UI 缩放，这是一个允许您根据屏幕大小缩放用户界面的系统。用户界面元素以像素为单位表示，通常以绝对值（按钮应该高 10 像素）表示。

这个问题在于，如果您使用高分辨率的面板，10 像素可能非常小，因为每个像素的尺寸更小。

# 准备工作

Unreal 中的 UI 缩放系统允许您控制全局缩放修改器，这将根据屏幕分辨率缩放屏幕上的所有控件。根据前面的示例，您可能希望调整按钮的大小，以便在您在较小的屏幕上查看 UI 时，其显示大小保持不变。本配方展示了两种不同的方法来改变缩放率。

# 如何做到这一点...

1.  创建一个自定义的`PlayerController`子类。将其命名为`ScalingUIPlayerController`。

1.  在类内部，覆盖`BeginPlay`：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/PlayerController.h"
#include "ScalingUIPlayerController.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_14_API AScalingUIPlayerController : public APlayerController
{
    GENERATED_BODY()

public:
 virtual void BeginPlay() override;
};

```

1.  在`ScalingUIPlayerController.cpp`函数的实现中添加以下代码：

```cpp
#include "ScalingUIPlayerController.h"
#include "SlateBasics.h"

void AScalingUIPlayerController::BeginPlay()
{
 Super::BeginPlay();
 TSharedRef<SVerticalBox> widget = SNew(SVerticalBox)
 + SVerticalBox::Slot()

 .HAlign(HAlign_Center)
 .VAlign(VAlign_Center)
 [
 SNew(SButton)
 .Content()
 [
 SNew(STextBlock)
 .Text(FText::FromString(TEXT("Test button")))
 ]
 ];
 GEngine->GameViewport->AddViewportWidgetForPlayer(GetLocalPlayer(), widget, 1);
}
```

1.  创建一个新的`GameModeBase`子类，命名为`ScalingUIGameMode`，并给它一个默认构造函数：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "ScalingUIGameMode.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_14_API AScalingUIGameMode : public AGameModeBase
{
    GENERATED_BODY()

 AScalingUIGameMode();
};
```

1.  在默认构造函数中，将默认玩家控制器类设置为`ScalingUIPlayerController`：

```cpp
#include "ScalingUIGameMode.h"
#include "CustomHUDPlayerController.h"

AScalingUIGameMode::AScalingUIGameMode() : AGameModeBase()
{
    PlayerControllerClass = ACustomHUDPlayerController::StaticClass();
}
```

1.  保存并编译你的新类。

1.  在编辑器中，通过转到“设置 | 世界设置”从工具栏打开“世界设置”菜单。

    1.  在世界设置中，覆盖级别的游戏模式为我们的`ScalingUIGameMode`。

    ![图片](img/02f9189c-7fad-4349-8bb7-195579247691.png)

    这应该会给你一个类似于之前菜谱中的用户界面。注意，如果你使用“在编辑器中播放”，UI 会非常小：

    ![图片](img/d136a6b5-4043-4f09-8443-541766ef6a1c.png)

    游戏屏幕上的小按钮

    要改变 UI 缩放上下文的速度，我们需要更改缩放曲线。我们可以通过两种不同的方法来实现。

    # 使用编辑器内方法

    1.  启动 Unreal，然后通过编辑菜单打开项目设置对话框：

    ![图片](img/4bf372f4-6819-4ee5-96f2-662d40adc52c.jpg)

    1.  在“引擎 - 用户界面”部分下，有一个名为 DPI 曲线的曲线，可以用来根据屏幕的短边调整 UI 缩放因子：

    ![图片](img/b3a198bb-bb69-47ea-a3d9-61f52351f475.png)

    1.  点击图表上的第二个点，或关键点。

    1.  将其缩放值更改为`1`。然后，对第一个点也做同样的操作，将其缩放值也设置为`1`：

    ![图片](img/71b43d27-052b-49b2-a2f5-28c1e89ae87c.png)

    1.  返回主编辑器并再次运行游戏。你应该注意到按钮比之前更大了：

    ![图片](img/91c6e745-0bc1-4555-9288-fe5dca318431.png)

    游戏屏幕上更容易看到的按钮

    # 使用配置文件方法

    1.  浏览到你的项目目录，并查看`Config`文件夹：

    ![图片](img/e4843569-d39c-408e-a063-9829f0467739.png)

    1.  打开位于项目`Config`文件夹中的`DefaultEngine.ini`，使用你选择的文本编辑器。

    1.  查找`[/Script/Engine.UserInterfaceSettings]`部分：

    ```cpp
    [/Script/Engine.UserInterfaceSettings]
    UIScaleCurve=(EditorCurveData=(PreInfinityExtrap=RCCE_Constant,PostInfinityExtrap=RCCE_Constant,DefaultValue=340282346638528859811704183484516925440.000000,Keys=((Time=480.000000,Value=1.000000),(Time=720.000000,Value=1.000000),(Time=1080.000000,Value=1.000000),(Time=8640.000000,Value=8.000000))),ExternalCurve=None)
    ```

    1.  在该部分查找名为`UIScaleCurve`的键。

    1.  在该键的值中，你会注意到一些`(Time=x,Value=y)`对。如果尚未设置，编辑第二个对，使其`Time`值为`720.000000`，`Value`为`1.000000`。

    1.  如果你有打开编辑器，请重新启动编辑器。

    1.  启动“在编辑器中播放”预览以确认你的 UI 现在在**PIE**屏幕的分辨率下仍然可读（假设你使用的是 1080p 显示器，因此 PIE 窗口运行在 720p 左右）：

    ![图片](img/288286d3-a546-4e7b-b440-5fc65abe4c8a.jpg)

    1.  你也可以通过使用新编辑器窗口预览你的游戏来查看缩放是如何工作的。

    1.  要这样做，请点击工具栏上“播放”右侧的箭头。

    1.  选择新编辑器窗口。

    1.  在此窗口中，你可以使用控制台命令`r.SetRes widthxheight`来更改分辨率（例如，`r.SetRes 200x200`），并观察这样做产生的变化。

    # 它是如何工作的...

    如同往常，当我们想要使用自定义的 `PlayerController` 时，我们需要一个自定义的 `GameMode` 来指定使用哪个 `PlayerController`。

    我们创建了一个自定义的 `PlayerController` 和 `GameMode`，然后在 `PlayerController` 的 `BeginPlay` 方法中放置一些 `Slate` 代码，以便绘制一些 UI 元素。

    因为在虚幻编辑器中，主游戏视口通常相当小，UI 最初以缩放的方式显示。这是为了允许游戏 UI 在较小分辨率的显示器上占用更少的空间，但如果没有将窗口拉伸以适应全屏，这可能会导致文本非常难以阅读。

    虚幻在配置文件中存储应在会话之间持久存在的配置数据，但不必一定硬编码到可执行文件中。配置文件使用 `.ini` 文件格式的扩展版本，这已经在 Windows 软件中广泛使用。

    配置文件使用以下语法存储数据：

    ```cpp
    [Section Name] 
    Key=Value 
    ```

    虚幻有一个 `UserInterfaceSettings` 类，它上面有一个名为 `UIScaleCurve` 的属性。该 `UPROPERTY` 被标记为配置，因此虚幻将值序列化到 `.ini` 文件中。

    因此，它将 `UIScale` 数据存储在 `DefaultEngine.ini` 文件中的 `Engine.UserInterfaceSettings` 部分。

    数据使用文本格式存储，其中包含一系列关键点。编辑 `Time`、`Value` 对会改变或添加新的关键点到曲线上。

    项目设置对话框是一个简单的界面，可以直接编辑 `.ini` 文件，对于设计师来说，这是一种直观的编辑曲线的方法。然而，以文本形式存储数据允许程序员开发构建工具，修改如 `UIScale` 等属性，而无需重新编译他们的游戏。

    `Time` 指的是输入值。在这种情况下，输入值是屏幕的较窄维度（通常是高度）。

    `Value` 是当屏幕的窄边大约等于 `Time` 字段中的值的高度时，应用于 UI 的通用缩放因子。

    因此，要将 UI 设置为在 1280 x 720 分辨率下保持正常大小，将时间/输入因子设置为 720，将缩放因子设置为 1。

    # 参见

    +   您可以参考 UE4 文档以获取有关配置文件的更多信息：[`docs.unrealengine.com/en-US/Programming/Basics/ConfigurationFiles`](https://docs.unrealengine.com/en-US/Programming/Basics/ConfigurationFiles)。

    # 在游戏中显示和隐藏 UMG 元素的工作表

    我们已经讨论了如何将小部件添加到视口中，这意味着它将在玩家的屏幕上渲染。

    然而，如果我们想根据其他因素切换 UI 元素，比如接近某些 Actor、玩家按下一个键，或者如果我们想创建一个在指定时间后消失的 UI 呢？

    # 如何做到这一点...

    1.  创建一个新的 `GameModeBase` 类，命名为 `ToggleHUDGameMode`：

    ![图片](img/ebab40ac-26b7-46a2-906b-b5eead0b1736.png)

    1.  将以下 `UPROPERTY` 和函数定义添加到 `ToggleHUDGameMode.h` 文件中：

    ```cpp
    #pragma once

    #include "CoreMinimal.h"
    #include "GameFramework/GameModeBase.h"
    #include "SlateBasics.h"
    #include "ToggleHUDGameMode.generated.h"

    UCLASS()
    class CHAPTER_14_API AToggleHUDGameMode : public AGameModeBase
    {
     GENERATED_BODY()

    public:
     UPROPERTY()
     FTimerHandle HUDToggleTimer;

     TSharedPtr<SVerticalBox> widget;

     virtual void BeginPlay() override;
     virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    };
    ```

    1.  在方法体中使用以下代码实现 `BeginPlay`：

    ```cpp
    void AToggleHUDGameMode::BeginPlay()
    {
        Super::BeginPlay();
        widget = SNew(SVerticalBox)
            + SVerticalBox::Slot()
            .HAlign(HAlign_Center)
            .VAlign(VAlign_Center)
            [
                SNew(SButton)
                .Content()
            [
                SNew(STextBlock)
                .Text(FText::FromString(TEXT("Test button")))
            ]
            ];

        auto player = GetWorld()->GetFirstLocalPlayerFromController();

        GEngine->GameViewport->AddViewportWidgetForPlayer(player, widget.ToSharedRef(), 1);

        auto lambda = FTimerDelegate::CreateLambda
        ([this]
        {
            if (this->widget->GetVisibility().IsVisible())
            {
                this->widget->SetVisibility(EVisibility::Hidden);

            }
            else
            {
                this->widget->SetVisibility(EVisibility::Visible);
            }
        });

        GetWorld()->GetTimerManager().SetTimer(HUDToggleTimer, lambda, 5, true);
    }
    ```

    1.  实现 `EndPlay`：

    ```cpp
    void AToggleHUDGameMode::EndPlay(const EEndPlayReason::Type EndPlayReason)
    {
        Super::EndPlay(EndPlayReason);
        GetWorld()->GetTimerManager().ClearTimer(HUDToggleTimer);
    }
    ```

    1.  编译你的代码并启动编辑器。

    1.  在编辑器中，从工具栏打开世界设置：

    ![](img/e117fa02-d7d4-447a-a504-6350baad3827.jpg)

    1.  在世界设置中，覆盖关卡的游戏模式为我们的 `AToggleHUDGameMode`：

    ![](img/3eb58755-c6a1-4830-973d-c23df59487c3.jpg)

    1.  播放关卡并验证 UI 每五秒钟切换其可见性。

    # 它是如何工作的...

    与本章中的大多数其他配方一样，我们使用自定义的 `GameMode` 类来方便地在玩家的视口中显示我们的单人 UI。

    我们覆盖 `BeginPlay` 和 `EndPlay`，以便我们可以正确处理将为我们切换 UI 的计时器。为此，我们需要将计时器的引用存储为 `UPROPERTY`，以确保它不会被垃圾回收。

    在 `BeginPlay` 中，我们使用 `SNew` 宏创建一个新的 `VerticalBox`，并在其第一个槽中放置一个按钮。按钮有 `Content`，它可以是其他小部件，例如 `SImage` 或 `STextBlock`，以在其中托管。

    在这种情况下，我们将一个 `STextBlock` 放入 `Content` 插槽中。文本块的内容无关紧要，也就是说，只要它们足够长，我们就能正确地看到我们的按钮。

    初始化了我们的小部件层次结构后，我们将根小部件添加到玩家的视口中，以便他们可以看到。

    现在，我们设置一个计时器来切换我们小部件的可见性。我们使用计时器来简化这个配方，而不是必须实现用户输入和输入绑定，但原理是相同的。为此，我们获取游戏世界的引用及其相关的计时器管理器。

    拿到计时器管理器后，我们可以创建一个新的计时器。但是，我们需要实际指定计时器到期时运行的代码。一种简单的方法是使用 `lambda` 函数来切换 hud 功能。

    Lambda 是匿名函数。可以将它们视为字面函数。要将 `lambda` 函数链接到计时器，我们需要创建一个 `timer` 代理。

    `FTimerDelegate::CreateLambda` 函数旨在将 `lambda` 函数转换为代理，计时器可以在指定的间隔调用它。

    `lambda` 需要从其包含的对象 `GameMode` 中访问 `this` 指针，以便能够更改我们创建的小部件实例上的属性。为了给它所需的访问权限，我们在 `lambda` 声明中开始使用 `[]` 操作符，这些操作符包围着应该捕获到 `lambda` 中并可在其中访问的变量。然后大括号以与正常函数声明相同的方式包围函数体。

    在函数内部，我们检查我们的小部件是否可见。如果是可见的，则使用 `SWidget::SetVisibility` 隐藏它。如果小部件不可见，则使用相同的函数调用打开它。

    在`SetTimer`的其余调用中，我们指定调用计时器的间隔（以秒为单位），并将计时器设置为循环。

    然而，我们需要小心的一点是，我们的对象在两次计时器调用之间可能被销毁，如果留下对对象的引用悬空，可能会导致崩溃。为了解决这个问题，我们需要删除计时器。

    由于我们在`BeginPlay`期间设置了计时器，因此有道理在`EndPlay`期间清除计时器。`EndPlay`将在`GameMode`结束播放或被销毁时被调用，因此我们可以在其实现期间安全地取消计时器。

    将`GameMode`设置为默认游戏模式，当游戏开始播放时创建 UI，计时器代理每五秒执行一次，在`true`和`false`之间切换小部件的可见性。

    当你关闭游戏时，`EndPlay`清除计时器引用，避免出现任何问题。

    # 将函数调用附加到 Slate 事件

    虽然创建按钮是件好事，但目前，你添加到玩家屏幕上的任何 UI 元素都只是静静地在那里，没有任何反应，即使用户点击它也是如此。目前我们没有将任何事件处理程序附加到 Slate 元素上，因此鼠标点击等事件实际上不会引发任何操作。

    # 准备中

    这个配方展示了如何将这些事件附加到函数，以便在它们发生时运行自定义代码。

    # 如何做到这一点...

    1.  创建一个新的`GameModeBase`子类，命名为`ClickEventGameMode`：

    ![图片](img/ee48dd37-65bb-47a8-aab4-41abb8f4e351.png)

    1.  从`ClickEventGameMode.h`文件中，向类中添加以下函数和`private`成员：

    ```cpp
    #pragma once

    #include "CoreMinimal.h"
    #include "GameFramework/GameModeBase.h"
    #include "SlateBasics.h"
    #include "ClickEventGameMode.generated.h"

    UCLASS()
    class CHAPTER_14_API AClickEventGameMode : public AGameModeBase
    {
        GENERATED_BODY()

    private:
     TSharedPtr<SVerticalBox> Widget;
     TSharedPtr<STextBlock> ButtonLabel;

    public:
     virtual void BeginPlay() override;
     FReply ButtonClicked();
    };
    ```

    1.  在`.cpp`文件中，添加`BeginPlay`的实现：

    ```cpp
    void AClickEventGameMode::BeginPlay()
    {
        Super::BeginPlay();

        Widget = SNew(SVerticalBox)
            + SVerticalBox::Slot()
            .HAlign(HAlign_Center)
            .VAlign(VAlign_Center)
            [
                SNew(SButton)
                .OnClicked(FOnClicked::CreateUObject(this, &AClickEventGameMode::ButtonClicked))
            .Content()
            [
                SAssignNew(ButtonLabel, STextBlock)
                .Text(FText::FromString(TEXT("Click me!")))
            ]
            ];

        auto player = GetWorld()->GetFirstLocalPlayerFromController();

        GEngine->GameViewport->AddViewportWidgetForPlayer(player, Widget.ToSharedRef(), 1);

        GetWorld()->GetFirstPlayerController()->bShowMouseCursor = true;

        auto pc = GEngine->GetFirstLocalPlayerController(GetWorld());

        EMouseLockMode lockMode = EMouseLockMode::DoNotLock;

        auto inputMode = FInputModeUIOnly().SetLockMouseToViewportBehavior(lockMode).SetWidgetToFocus(Widget);

        pc->SetInputMode(inputMode);

    }
    ```

    1.  此外，添加`ButtonClicked()`的实现：

    ```cpp
    FReply AClickEventGameMode::ButtonClicked()
    {
        ButtonLabel->SetText(FString(TEXT("Clicked!")));
        return FReply::Handled();
    }
    ```

    1.  编译你的代码并启动编辑器。

    1.  在世界设置中覆盖游戏模式为`ClickEventGameMode`。

    1.  在编辑器中预览并验证 UI 是否显示了一个按钮，当你使用鼠标光标点击它时，按钮会从“Click Me!”变为“Clicked!”：

    ![图片](img/72e77d50-3b49-4973-b4ac-bddae6bbe8cc.png)

    按钮在被点击后会显示“Clicked!”

    # 它是如何工作的...

    与本章中的大多数配方一样，我们使用`GameMode`创建和显示我们的 UI，以最小化你需要创建的与配方无关的类的数量。

    在我们的新游戏模式中，我们需要保留我们创建的 Slate 小部件的引用，这样我们就可以在它们创建后与之交互。

    因此，我们在`GameMode`中创建了两个共享指针作为成员数据——一个指向我们的 UI 的整体父或根小部件，另一个指向按钮上的标签，因为我们将在稍后运行时更改标签文本。

    我们覆盖`BeginPlay`，因为这是在游戏开始后创建 UI 的一个方便的地方，我们能够获取到有效的玩家控制器引用。

    我们还创建了一个名为 `ButtonClicked` 的函数。它返回 `FReply`，一个表示事件是否被处理的 `struct`。`ButtonClicked` 函数的签名由 `FOnClicked` 的签名决定，这是一个我们将在稍后使用的委托。

    在我们的 `BeginPlay` 实现中，我们首先调用我们正在重写的实现，以确保类被适当地初始化。

    然后，像往常一样，我们使用我们的 `SNew` 函数来创建 `VerticalBox`，并给它添加一个居中的槽位。

    我们在那个槽位中创建一个新的 `Button`，并向按钮包含的 `OnClicked` 属性添加一个值。

    `OnClicked` 是一个委托属性。这意味着当发生特定事件时（如名称所暗示的，在这个例子中，当按钮被点击时），`Button` 将广播 `OnClicked` 委托。

    要订阅或监听委托，并接收它所引用的事件通知，我们需要将委托实例分配给属性。

    我们使用标准的委托函数，如 `CreateUObject`、`CreateStatic` 或 `CreateLambda` 来实现这一点。这些中的任何一个都可以工作——我们可以绑定 `UObject` 成员函数、静态函数、lambda 函数和其他函数。

    查阅第五章，*处理事件和委托*，了解更多关于委托的信息，并查看我们可以绑定到委托的其他函数类型。

    `CreateUObject` 期望一个指向类实例的指针和一个指向在该类中定义的成员函数的指针。该函数必须具有可以转换为委托签名的签名：

    ```cpp
    /** The delegate to execute when the button is clicked */
     FOnClickedOnClicked;
    ```

    如我们所见，`OnClicked` 委托类型是 `FOnClicked` ——这就是为什么我们声明的 `ButtonClicked` 函数具有与 `FOnClicked` 相同的签名。

    通过传递对这个指针和要调用的函数的指针，当按钮被点击时，引擎将在特定的对象实例上调用该函数。

    在设置委托之后，我们使用 `Content()` 函数，它返回对按钮用于包含任何内容的单个槽位的引用。

    我们接着使用 `SAssignNew` 通过 `TextBlock` 小部件创建我们的按钮标签。`SAssignNew` 非常重要，因为它允许我们使用 Slate 的声明性语法，同时将变量分配给层次结构中特定的子小部件。`SAssignNew` 的第一个参数是我们想要存储小部件的变量，第二个参数是小部件的类型。

    由于 `ButtonLabel` 现在指向我们的按钮的 `TextBlock`，我们可以将其 `Text` 属性设置为静态字符串。

    最后，我们使用 `AddViewportWidgetForPlayer` 将小部件添加到玩家的视图中，该函数期望参数为 `LocalPlayer` 以添加小部件，小部件本身，以及一个深度值（值越高越靠前）。

    要获取 `LocalPlayer` 实例，我们假设我们在没有分屏的情况下运行，因此第一个玩家控制器将是唯一的，即玩家的控制器。`GetFirstLocalPlayerFromController` 函数是一个便利函数，它简单地获取第一个玩家的控制器并返回其本地玩家对象。

    我们还需要聚焦小部件，以便玩家可以点击它，并显示光标，以便玩家知道他们的鼠标在屏幕上的位置。

    从上一步我们知道，我们可以假设第一个本地玩家控制器是我们感兴趣的，因此我们可以访问它并更改其 `ShowMouseCursor` 变量为 `true`。这将导致光标在屏幕上渲染。

    `SetInputMode` 允许我们聚焦于一个小部件，以便玩家可以在其他 UI 相关功能（如锁定鼠标到游戏视口）中与之交互。它使用一个 `FInputMode` 对象作为其唯一参数，我们可以使用 `builder` 模式通过特定的元素来构建它，我们希望包含哪些元素。

    `FInputModeUIOnly` 类是 `FInputMode` 的子类，它指定了我们将所有输入事件重定向到 UI 层，而不是玩家控制器和其他输入处理。

    `builder` 模式允许我们在对象实例被作为参数发送到函数之前，通过链式调用方法来自定义我们的对象实例。

    我们链式调用 `SetLockMouseToViewport(false)` 来指定玩家的鼠标可以离开游戏屏幕的边界，通过 `SetWidgetToFocus(Widget)` 指定我们的顶级小部件作为游戏应将玩家输入定向到的小部件。

    最后，我们有 `ButtonClicked` 的实际实现，它是我们的事件处理器。当函数因我们的按钮被点击而运行时，我们更改按钮的标签以指示它已被点击。然后我们需要返回一个 `FReply` 实例给调用者，让 UI 框架知道事件已被处理，并且不要继续将事件向上传播到小部件层次结构。

    `FReply::Handled()` 返回一个设置好的 `FReply`，以指示框架。我们本可以使用 `FReply::Unhandled()`，但这将告诉框架点击事件实际上不是我们感兴趣的，它应该寻找可能对事件感兴趣的其他对象。

    # 使用数据绑定与 Unreal Motion Graphics

    到目前为止，我们一直在将静态值分配给我们的 UI 小部件的属性。然而，如果我们想使小部件内容或如边框颜色等参数更加动态，我们可以使用一个称为数据绑定的原则，将我们的 UI 属性与更广泛程序中的变量动态链接。

    Unreal 使用属性系统来允许我们将属性的值绑定到函数的返回值，例如。这意味着更改这些变量将自动导致 UI 根据我们的意愿进行更改。

    # 如何实现...

    1.  创建一个新的 `GameModeBase` 子类，命名为 `AttributeGameMode`。

    1.  将 `AttributeGameMode.h` 文件更新为以下内容：

    ```cpp
    #pragma once

    #include "CoreMinimal.h"
    #include "GameFramework/GameStateBase.h"
    #include "SlateBasics.h"
    #include "AttributeGameMode.generated.h"

    /**
     * 
     */
    UCLASS()
    class CHAPTER_14_API AAttributeGameMode : public AGameModeBase
    {
     GENERATED_BODY()

     TSharedPtr<SVerticalBox> Widget;
     FText GetButtonLabel() const;

    public:
     virtual void BeginPlay() override;

    };
    ```

    1.  在 `.cpp` 文件中添加 `BeginPlay` 的实现：

    ```cpp
    void AAttributeGameMode::BeginPlay()
    {
        Super::BeginPlay();

        Widget = SNew(SVerticalBox)
            + SVerticalBox::Slot()
            .HAlign(HAlign_Center)
            .VAlign(VAlign_Center)
            [
                SNew(SButton)
                .Content()
            [
                SNew(STextBlock)
                .Text(TAttribute<FText>::Create(TAttribute<FText>::FGetter::CreateUObject(this, &AAttributeGameMode::GetButtonLabel)))
            ]
            ];
        GEngine->GameViewport->AddViewportWidgetForPlayer(GetWorld()->GetFirstLocalPlayerFromController(), Widget.ToSharedRef(), 1);

    }
    ```

    1.  还要添加 `GetButtonLabel()` 的实现：

    ```cpp
    FText AAttributeGameMode::GetButtonLabel() const
    {
        FVector ActorLocation = GetWorld()->GetFirstPlayerController()->GetPawn()->GetActorLocation();
        return FText::FromString(FString::Printf(TEXT("%f, %f, %f"), ActorLocation.X, ActorLocation.Y, ActorLocation.Z));
    }
    ```

    1.  编译你的代码并启动编辑器。

    1.  在世界设置中覆盖游戏模式为 `AAttributeGameMode`。

    1.  注意，在 Play In Editor 会话中，UI 按钮上的值会随着玩家在场景中的移动而改变：

    ![图片](img/2154c05b-1acb-4ebe-b7ec-beea999cb541.png)

    # 它是如何工作的...

    就像本章中几乎所有的其他配方一样，我们首先需要做的是创建一个游戏模式，作为我们 UI 的便捷宿主。我们创建 UI 的方式与其他配方相同，即在游戏模式的 `BeginPlay()` 方法中放置 `Slate` 代码。

    本配方中有趣的功能涉及我们如何设置按钮标签文本的值：

    ```cpp
    .Text( 
     TAttribute<FText>::Create(TAttribute<FText>::FGetter::Creat
     eUObject(this, &AAttributeGameMode::GetButtonLabel)))
    ```

    上述语法非常冗长，但它实际上做的事情相对简单。我们向 `Text` 属性分配了一个值，该属性的类型为 `FText`。我们可以将 `TAttribute<FText>` 分配给此属性，并且每当 UI 想要确保 `Text` 的值是最新的时，都会调用 `TAttribute Get()` 方法。

    要创建 `TAttribute`，我们需要调用静态方法 `TAttribute<VariableType>::Create()`。此函数期望一个描述性的委托。根据传递给 `TAttribute::Create` 的委托类型，`TAttribute::Get()` 将调用不同类型的函数来检索实际值。

    在本配方的代码中，我们调用 `UObject` 的成员函数。这意味着我们知道我们将在某些委托类型上调用 `CreateUObject` 函数。

    我们可以使用 `CreateLambda`、`CreateStatic` 或 `CreateRaw` 分别调用原始 C++ 类上的 `lambda`、`static` 或 `member` 函数。这将给我们当前属性的值。

    但我们想要创建哪种委托类型的实例？因为我们正在将 `TAttribute` 类模板化，以便与属性关联的实际变量类型，我们需要一个委托，其返回值也是模板化的变量类型。

    也就是说，如果我们有 `TAttribute<FText>`，与之连接的委托需要返回一个 `FText`。

    在 `TAttribute` 中，我们有以下代码：

    ```cpp
    template<typenameObjectType>
     classTAttribute
     {
     public:
     /**
     * Attribute 'getter' delegate
     *
     * ObjectTypeGetValue() const
     *
     * @return The attribute's value
     */
     DECLARE_DELEGATE_RetVal(ObjectType, FGetter);
     (...)
     }
    ```

    `FGetter` 委托类型在 `TAttribute` 类内部声明，因此其返回值可以基于 `TAttribute` 模板的 `ObjectType` 参数进行模板化。这意味着 `TAttribute<Typename>::FGetter` 自动定义了一个具有正确返回类型 `Typename` 的委托。因此，我们需要创建一个与 `TAttribute<FText>::FGetter` 类型签名绑定的 UObject 委托。

    一旦我们有了那个委托，我们就可以在委托上调用 `TAttribute::Create` 来将委托的返回值链接到我们的 `TextBlock` 成员变量 `Text`。有了我们的 UI 定义以及 `Text` 属性、`TAttribute<FText>` 和返回 `FText` 的委托之间的绑定，我们现在可以将 UI 添加到玩家的屏幕上，使其可见。

    每一帧，游戏引擎都会检查所有属性，看它们是否链接到 `TAttributes`。如果有连接，则调用 `TAttribute::Get()` 函数，调用委托并返回委托的返回值，以便 Slate 可以将其存储在窗口小部件相应的成员变量中。

    对于我们这个过程的演示，`GetButtonLabel` 获取游戏世界中第一个玩家兵的位置。然后我们使用 `FString::Printf` 将位置数据格式化为可读的字符串，并将其包裹在 `FText` 中，以便它可以作为 `TextBlock` 文本值存储。

    # 使用样式控制小部件外观

    到目前为止，在本章中，我们一直在创建使用默认视觉表示的 UI 元素。这个配方展示了如何在 C++中创建一个样式，该样式可以在整个项目中用作通用的外观和感觉。

    # 如何做...

    1.  通过使用添加 C++类向导并选择无作为父类来为你的项目创建一个新的类：

    ![](img/64c3708b-d1e9-480b-95c7-28fcaf1f9d69.png)

    1.  在名称选项下，使用 `CookbookStyle` 并点击创建类按钮：

    ![](img/ebb36faa-5690-4154-aae9-0569f96d67d4.png)

    1.  将 `CookbookStyle.h` 文件中的代码替换为以下代码：

    ```cpp
    #pragma once
    #include "SlateBasics.h"
    #include "SlateExtras.h"

    class FCookbookStyle
    {
    public:
        static void Initialize();
        static void Shutdown();
        static void ReloadTextures();
        static const ISlateStyle& Get();
        static FName GetStyleSetName();

    private:
        static TSharedRef< class FSlateStyleSet > Create();
    private:
        static TSharedPtr< class FSlateStyleSet > CookbookStyleInstance;
    };
    ```

    1.  打开 `CookbookStyle.cpp` 文件，并使用以下代码：

    ```cpp
    #include "CookbookStyle.h"
    #include "SlateGameResources.h"

    TSharedPtr< FSlateStyleSet > FCookbookStyle::CookbookStyleInstance = NULL;

    void FCookbookStyle::Initialize()
    {
        if (!CookbookStyleInstance.IsValid())
        {
            CookbookStyleInstance = Create();
        FSlateStyleRegistry::RegisterSlateStyle(*CookbookStyleInstance);
        }
    }

    void FCookbookStyle::Shutdown()
    {
        FSlateStyleRegistry::UnRegisterSlateStyle(*CookbookStyleInstance);
        ensure(CookbookStyleInstance.IsUnique());
        CookbookStyleInstance.Reset();
    }

    FName FCookbookStyle::GetStyleSetName()
    {
        static FName StyleSetName(TEXT("CookbookStyle"));
        return StyleSetName;
    }

    ```

    1.  在 `CookbookStyle.cpp` 文件中，在之前创建的脚本下方添加以下内容来描述如何绘制屏幕：

    ```cpp
    #define IMAGE_BRUSH( RelativePath, ... ) FSlateImageBrush( FPaths::GameContentDir() / "Slate"/ RelativePath + TEXT(".png"), __VA_ARGS__ )
    #define BOX_BRUSH( RelativePath, ... ) FSlateBoxBrush( FPaths::GameContentDir() / "Slate"/ RelativePath + TEXT(".png"), __VA_ARGS__ )
    #define BORDER_BRUSH( RelativePath, ... ) FSlateBorderBrush( FPaths::GameContentDir() / "Slate"/ RelativePath + TEXT(".png"), __VA_ARGS__ )
    #define TTF_FONT( RelativePath, ... ) FSlateFontInfo( FPaths::GameContentDir() / "Slate"/ RelativePath + TEXT(".ttf"), __VA_ARGS__ )
    #define OTF_FONT( RelativePath, ... ) FSlateFontInfo( FPaths::GameContentDir() / "Slate"/ RelativePath + TEXT(".otf"), __VA_ARGS__ )

    TSharedRef< FSlateStyleSet > FCookbookStyle::Create()
    {
        TSharedRef<FSlateStyleSet> StyleRef = FSlateGameResources::New(FCookbookStyle::GetStyleSetName(), "/Game/Slate", "/Game/Slate");
        FSlateStyleSet& Style = StyleRef.Get();

        Style.Set("NormalButtonBrush",
            FButtonStyle().
            SetNormal(BOX_BRUSH("Button", FVector2D(54, 54), FMargin(14.0f / 54.0f))));
        Style.Set("NormalButtonText",
            FTextBlockStyle(FTextBlockStyle::GetDefault())
            .SetColorAndOpacity(FSlateColor(FLinearColor(1, 1, 1, 1))));
        return StyleRef;
    }

    #undef IMAGE_BRUSH
    #undef BOX_BRUSH
    #undef BORDER_BRUSH
    #undef TTF_FONT
    #undef OTF_FONT

    void FCookbookStyle::ReloadTextures()
    {
        FSlateApplication::Get().GetRenderer()->ReloadTextureResources();
    }

    const ISlateStyle& FCookbookStyle::Get()
    {
        return *CookbookStyleInstance;
    }
    ```

    1.  创建一个新的 `GameModeBase` 子类，`StyledHUDGameMode`:

    ![](img/a6078305-d562-4c27-b878-7e635d381cfd.png)

    1.  一旦 Visual Studio 打开，向其声明中添加以下代码：

    ```cpp
    #pragma once

    #include "CoreMinimal.h"
    #include "GameFramework/GameModeBase.h"
    #include "SlateBasics.h"
    #include "StyledHUDGameMode.generated.h"

    /**
     * 
     */
    UCLASS()
    class CHAPTER_14_API AStyledHUDGameMode : public AGameModeBase
    {
        GENERATED_BODY()

        TSharedPtr<SVerticalBox> Widget;

    public:
        virtual void BeginPlay() override;
    };
    ```

    1.  同样，实现 `GameMode`:

    ```cpp
    #include "StyledHUDGameMode.h"
    #include "CookbookStyle.h"

    void AStyledHUDGameMode::BeginPlay()
    {
        Super::BeginPlay();

        Widget = SNew(SVerticalBox)
            + SVerticalBox::Slot()
            .HAlign(HAlign_Center)
            .VAlign(VAlign_Center)
            [
                SNew(SButton)
                .ButtonStyle(FCookbookStyle::Get(), "NormalButtonBrush")
            .ContentPadding(FMargin(16))
            .Content()
            [
                SNew(STextBlock)
                .TextStyle(FCookbookStyle::Get(), "NormalButtonText")
            .Text(FText::FromString("Styled Button"))
            ]
            ];
        GEngine->GameViewport->AddViewportWidgetForPlayer(GetWorld()->GetFirstLocalPlayerFromController(), Widget.ToSharedRef(), 1);

    }
    ```

    1.  最后，创建一个带有边框的 54 x 54 像素 PNG 文件，用于我们的按钮：

    ![](img/73922731-36ea-47f4-9e56-48280866be32.png)

    1.  将其保存到 `Content` | `Slate` 文件夹，文件名为 `Button.png`，如果需要则创建文件夹：

    ![](img/293da37b-9933-4c8f-b2e5-d749a70f8302.png)

    1.  你可能会被问是否要将图像导入到你的项目中。请继续并说“是”。

    1.  最后，我们需要设置我们的游戏模块，以便在加载时正确初始化样式。在你的游戏模块实现文件（`Chapter_14.h`）中，确保它看起来像这样：

    ```cpp
    #pragma once

    #include "CoreMinimal.h"
    #include "CookbookStyle.h"

    class Chapter_14Module : public FDefaultGameModuleImpl
    {
     virtual void StartupModule() override
     {
     FCookbookStyle::Initialize();
     };
     virtual void ShutdownModule() override
     {
     FCookbookStyle::Shutdown();
     };
    };
    ```

    1.  然后，转到 `Chapter_14.cpp` 文件并修改代码如下：

    ```cpp
    #include "Chapter_14.h"
    #include "Modules/ModuleManager.h"

    IMPLEMENT_PRIMARY_GAME_MODULE(Chapter_14Module, Chapter_14, "Chapter_14" );
    ```

    1.  编译代码并将你的游戏模式覆盖设置为新的游戏模式，就像我们在本章其他配方中所做的那样。

    1.  当你玩游戏时，你会看到你的自定义边框围绕在按钮周围，而且文本是白色而不是黑色：

    ![](img/0a88391a-3f1f-4343-be79-7a38b5271782.png)

    # 它是如何工作的...

    为了创建可以在多个 Slate 小部件之间共享的样式，我们需要创建一个对象来包含这些样式并保持它们的作用域。

    Epic 提供了`FSlateStyleSet`类来实现这个目的。`FSlateStyleSet`包含许多我们可以在 Slate 声明性语法中访问的样式，用于皮肤小部件。

    然而，在程序中散布多个`StyleSet`对象的副本是不高效的。我们实际上只需要一个这样的对象。

    因为`FSlateStyleSet`本身不是单例，即只能有一个实例的对象，我们需要创建一个类来管理我们的`StyleSet`对象，并确保我们只有一个实例。

    这就是为什么我们有`FCookbookStyle`类。它包含一个`Initialize()`函数，我们将在模块的启动代码中调用它。在`Initialize()`函数中，我们检查我们是否有`StyleSet`的实例。如果没有有效的实例，我们调用私有的`Create()`函数来实例化一个。

    然后我们使用`FSlateStyleRegistry`类注册样式。

    当我们的模块卸载时，我们需要反转这个注册过程，然后删除指针，以防止悬挂。

    现在我们有一个在模块初始化期间通过调用`Create()`创建的类的实例。你会注意到`Create`被多个具有类似形式的宏所包裹。这些宏在函数之前定义，在函数之后未定义。

    这些宏使我们能够通过消除为我们的样式可能想要使用的所有图像资源指定路径和扩展名的需要，简化`Create`函数中所需的代码。

    在`Create`函数内部，我们使用`FSlateGameResources::New()`函数创建一个新的`FSlateStyleSet`对象。`New()`需要一个样式的名称，以及我们想要在这个样式集中搜索的文件夹路径。

    这允许我们声明多个指向不同目录的样式集，但使用相同的图像名称。它还允许我们通过切换到其他基本目录中的一个样式集来简单地皮肤或重设整个 UI。

    `New()`返回一个共享引用对象，因此我们使用`Get()`函数检索实际的`FStyleSet`实例。

    拿着这个参考，我们可以创建我们想要这个集合包含的样式。要向集合添加样式，我们使用`Set()`方法。`Set()`方法期望样式的名称，然后是一个样式对象。样式对象可以使用`builder`模式进行自定义。

    我们首先添加一个名为`"NormalButtonBrush"`的样式。名称可以是任意的。因为我们想使用这个样式来改变按钮的外观，所以我们需要为第二个参数使用`FButtonStyle`。

    为了根据我们的需求自定义样式，我们使用 Slate 构建器语法，链接我们需要设置样式属性的所有方法调用。

    对于这个集合中的第一个样式，我们只是改变按钮在未被点击或处于非默认状态时的视觉外观。这意味着我们想要改变按钮在正常状态时使用的画刷，因此我们使用的函数是`SetNormal()`。

    使用`BOX_BRUSH`宏，我们告诉 Slate 我们想要使用`Button.png`，这是一个 54 x 54 像素大小的图像，并且我们想要保持每个角落的 14 像素不拉伸，用于九宫格缩放。

    对于九宫格缩放功能的更直观解释，请查看引擎源代码中的`SlateBoxBrush.h`。

    对于样式集中的第二个样式，我们创建了一个名为`"NormalButtonText"`的样式。对于这个样式，我们不想从默认样式改变一切；我们只想改变一个属性。因此，我们访问默认的文本样式并使用拷贝构造函数克隆它。

    使用我们默认样式的副本，我们首先将文本颜色更改为白色，首先创建一个线性颜色 R=1 G=1 B=1 A=1，然后将它转换成一个 Slate 颜色对象。

    在我们的样式集配置了两个新样式后，我们可以将其返回给调用函数，即`Initialize`。`Initialize`存储我们的样式集引用，消除了我们需要创建更多实例的需求。

    我们的风格容器类还有一个`Get()`函数，它用于检索用于 Slate 的实际`StyleSet`。因为`Initialize()`已经在模块启动时被调用，所以`Get()`简单地返回在该函数内部创建的`StyleSet`实例。

    在游戏模块内部，我们添加了实际调用`Initialize`和`Shutdown`的代码。这确保了当我们的模块被加载时，我们始终有一个有效的 Slate 样式的引用。

    和往常一样，我们创建一个游戏模式作为我们 UI 的主机，并覆盖`BeginPlay`以便我们可以在游戏开始时创建 UI。

    创建 UI 的语法与我们之前在食谱中使用的完全相同——使用`SNew`创建一个`VerticalBox`，然后使用 Slate 的声明性语法填充其他小部件。

    以下两条以下内容非常重要：

    ```cpp
    .ButtonStyle(FCookbookStyle::Get(), "NormalButtonBrush")
     .TextStyle(FCookbookStyle::Get(), "NormalButtonText")
    ```

    前面的行是我们按钮的声明性语法的一部分，以及构成其标签的文本。当我们使用`<Class>Style()`方法为小部件设置样式时，我们传递两个参数。

    第一个参数是我们实际的样式集，它通过使用`FCookbookStyle::Get()`检索，第二个是一个字符串参数，包含我们想要使用的样式的名称。

    通过这些微小的更改，我们覆盖了小部件的样式以使用我们的自定义样式，这样当我们把小部件添加到玩家的视口时，它们会显示我们的自定义设置。

    # 创建自定义 SWidget/UWidget

    到目前为止，本章中的食谱已经向您展示了如何使用现有的原始小部件创建 UI。

    有时，对于开发者来说，使用组合来收集多个 UI 元素以定义一个按钮类很有用，该按钮类自动具有 `TextBlock` 作为标签，而不是每次声明时都手动指定层次结构，例如。

    此外，如果你在 C++ 中手动指定层次结构，而不是声明由子控件组成的复合对象，则无法使用 UMG 将这些控件作为组实例化。

    # 准备工作

    此配方向您展示了如何创建一个包含一组控件的复合 `SWidget`，并公开新的属性来控制这些子控件的元素。它还将向您展示如何创建一个 `UWidget` 包装器，这样可以将新的复合 `SWidget` 类公开给 UMG，以便设计师可以使用它。

    # 如何做...

    1.  我们需要将 UMG 模块添加到我们模块的依赖项中。

    1.  打开 `<YourModule>.build.cs`，在我们的例子中是 `Chapter_14.Build.cs`，并将 UMG 添加到以下代码中：

    ```cpp
    using UnrealBuildTool;

    public class Chapter_14 : ModuleRules
    {
      public Chapter_14(ReadOnlyTargetRules Target) : base(Target)
      {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore" });

        PrivateDependencyModuleNames.AddRange(new string[] { });

        // Uncomment if you are using Slate UI
        PrivateDependencyModuleNames.AddRange(new string[] { "Slate", 
        "SlateCore", "UMG" });

        // Uncomment if you are using online features
        // PrivateDependencyModuleNames.Add("OnlineSubsystem");

        // To include OnlineSubsystemSteam, add it to the plugins 
        // section in your uproject file with the Enabled attribute
        // set to true
      }
    }
    ```

    1.  基于 Slate Widget 父类（`SCompoundWidget`）创建一个新的类：

    ![图片](img/1a17774f-7605-4d12-ad72-aa538589875a.png)

    1.  当被要求命名时，将其命名为 `CustomButton`。

    1.  创建后，将其声明中的以下代码添加到其中：

    ```cpp
    #pragma once

    #include "CoreMinimal.h"
    #include "Widgets/SCompoundWidget.h"

    class CHAPTER_14_API SCustomButton : public SCompoundWidget
    {
        SLATE_BEGIN_ARGS(SCustomButton)
            : _Label(TEXT("Default Value"))
            , _ButtonClicked()
        {}
        SLATE_ATTRIBUTE(FString, Label)
            SLATE_EVENT(FOnClicked, ButtonClicked)
            SLATE_END_ARGS()

    public:
        void Construct(const FArguments& InArgs);
        TAttribute<FString> Label;
        FOnClicked ButtonClicked;
    };
    ```

    1.  在相应的 `.cpp` 文件中实现该类，如下所示：

    ```cpp
    #include "CustomButton.h"
    #include "SlateOptMacros.h"
    #include "Chapter_14.h"

    void SCustomButton::Construct(const FArguments& InArgs)
    {
        Label = InArgs._Label;
        ButtonClicked = InArgs._ButtonClicked;
        ChildSlot.VAlign(VAlign_Center)
            .HAlign(HAlign_Center)
            [SNew(SButton)
            .OnClicked(ButtonClicked)
            .Content()
            [
                SNew(STextBlock)
                .Text_Lambda([this] {return FText::FromString(Label.Get()); })
            ]
            ];
    }
    ```

    1.  创建第二个类，这次基于 `Widget`：

    ![图片](img/70f5329f-c812-4ee8-9108-2c186e2130c5.png)

    1.  调用这个新的类 `CustomButtonWidget` 并按创建类。

    1.  将以下片段中的粗体代码添加到 `CustomButtonWidget.h` 文件中：

    ```cpp
    #pragma once

    #include "CoreMinimal.h"
    #include "Components/Widget.h"
    #include "CustomButton.h"
    #include "SlateDelegates.h"
    #include "CustomButtonWidget.generated.h"

    DECLARE_DYNAMIC_DELEGATE_RetVal(FString, FGetString);
    DECLARE_DYNAMIC_MULTICAST_DELEGATE(FButtonClicked); 
    UCLASS()
    class CHAPTER_14_API UCustomButtonWidget : public UWidget
    {
        GENERATED_BODY()

    protected:
     TSharedPtr<SCustomButton> MyButton;

     virtual TSharedRef<SWidget> RebuildWidget() override;

    public:
     UCustomButtonWidget();
     //multicast
     UPROPERTY(BlueprintAssignable)
     FButtonClicked ButtonClicked;

     FReply OnButtonClicked();

     UPROPERTY(BlueprintReadWrite, EditAnywhere)
     FString Label;

     //MUST be of the form varnameDelegate
     UPROPERTY()
     FGetString LabelDelegate;

     virtual void SynchronizeProperties() override;
    };
    ```

    1.  现在，为 `UCustomButtonWidget` 创建实现：

    ```cpp
    #include "CustomButtonWidget.h"
    #include "Chapter_14.h"

    TSharedRef<SWidget> UCustomButtonWidget::RebuildWidget()
    {
        MyButton = SNew(SCustomButton)
            .ButtonClicked(BIND_UOBJECT_DELEGATE(FOnClicked, OnButtonClicked));
        return MyButton.ToSharedRef();
    }

    UCustomButtonWidget::UCustomButtonWidget()
        :Label(TEXT("Default Value"))
    {

    }

    FReply UCustomButtonWidget::OnButtonClicked()
    {
        ButtonClicked.Broadcast();
        return FReply::Handled();
    }

    void UCustomButtonWidget::SynchronizeProperties()
    {
        Super::SynchronizeProperties();
        TAttribute<FString> LabelBinding = OPTIONAL_BINDING(FString, Label);
        MyButton->Label = LabelBinding;
    }
    ```

    1.  保存你的脚本并编译你的代码。

    1.  通过在内容浏览器上右键单击并选择用户界面然后选择 Widget 蓝图来创建一个新的 Widget 蓝图：

    ![图片](img/019ade72-7ac5-4639-84ec-59fbf045dcbd.png)

    您可以使用上下文菜单中的鼠标滚轮滚动到用户界面部分。

    1.  通过双击它来打开你的新 Widget 蓝图。

    1.  在 Widget 面板中找到自定义按钮控件：

    ![图片](img/7ec28d12-c477-458d-bb1d-e22f94528e26.png)

    1.  将其实例拖到主区域。

    1.  选择实例后，在详细信息面板中更改标签属性：

    ![图片](img/5941564e-b55b-4f07-b642-a24507181855.png)

    验证你的按钮是否已更改其标签。

    1.  现在，我们将创建一个绑定来演示我们可以将任意蓝图函数链接到我们的小部件的标签属性，这反过来又驱动了 Widget 的文本块标签。

    1.  点击标签属性右侧的绑定，选择创建绑定：

    ![图片](img/e4e52462-a9f1-4ed4-a5fd-8cbe749f1003.png)

    1.  在现在显示的图中，通过在主区域内部右键单击来放置一个获取游戏时间（秒）节点：

    ![图片](img/6f13155b-1c21-4fcc-9bd0-3e49c3ef08fc.png)

    1.  将获取游戏时间节点的返回值链接到函数的返回值引脚：

    ![图片](img/6421b9a8-c550-46aa-a2c4-22482c2699f5.jpg)

    1.  将自动为您插入一个转换浮点数为字符串节点：

    ![图片](img/508844dd-cbe0-48c4-a007-d9ecb4fbafff.jpg)

    1.  编译蓝图以确保其正确运行。

    1.  接下来，通过点击任务栏上的蓝图按钮并选择打开关卡蓝图来打开关卡蓝图：

    ![图片](img/a7e07af8-6327-4b45-9fd2-8b8ee267eba1.png)

    1.  在事件`BeginPlay`节点的右侧放置一个创建小部件节点到图中：

    ![图片](img/287da0f3-30e0-48de-a7fb-7524a6bc0782.png)

    1.  选择要创建的新小部件蓝图（我们刚刚在编辑器中创建的）的小部件类：

    ![图片](img/926ba980-6528-4778-8868-bf71d895ccb8.jpg)

    1.  从创建小部件节点的拥有玩家引脚上点击并拖动，放置一个获取玩家控制器节点：

    ![图片](img/cd0d48b2-dc8f-4b71-805c-639f0510e06e.jpg)

    1.  同样，从创建小部件节点的返回值拖动，放置一个添加到视口节点：

    ![图片](img/e005b240-36a4-4bde-9cb9-fea79fb1ed0e.jpg)

    1.  最后，将`BeginPlay`节点链接到创建小部件节点的执行引脚：

    ![图片](img/29583739-bbd2-4a2f-9750-d150412bd044.png)

    1.  预览你的游戏，并验证屏幕上显示的小部件是否是我们新创建的自定义按钮，其标签绑定自游戏开始以来经过的秒数！![图片](img/6e798818-3319-43c4-a5c3-24c71cc70708.png)

    在关卡中显示经过时间的按钮

    # 它是如何工作的...

    要使用`UWidget`类，我们的模块需要将 UMG 模块作为其依赖项之一包含，因为`UWidget`是在 UMG 模块内部定义的。

    然而，我们需要创建的第一个类实际上是我们的`SWidget`类。

    因为我们想要将两个小部件组合成一个复合结构，所以我们创建了一个新的`CompoundWidget`子类。`CompoundWidget`允许你将小部件层次结构封装为小部件本身。

    在类内部，我们使用`SLATE_BEGIN_ARGS`和`SLATE_END_ARGS`宏在新的`SWidget`上声明一个名为`FArguments`的内部`struct`。在`SLATE_BEGIN_ARGS`和`SLATE_END_ARGS`之间，使用`SLATE_ATTRIBUTE`和`SLATE_EVENT`宏。`SLATE_ATTRIBUTE`为我们提供的类型创建`TAttribute`。在这个类中，我们声明了一个名为`_Label`的`TAttribute`，它更具体地是一个`TAttribute<FString>`。

    `SLATE_EVENT`允许我们创建成员委托，当小部件内部发生某些事件时，我们可以广播这些委托。

    在`SCustomButton`中，我们声明了一个具有`FOnClicked`签名的委托，名为`ButtonClicked`。

    `SLATE_ARGUMENT`是另一个宏（在这个配方中没有使用），它使用你提供的类型和名称创建一个内部变量，并在变量名前添加一个下划线。

    `Construct()`是当小部件实例化时实现自我初始化的函数。你会注意到我们自己也创建了`TAttribute`和`FOnClicked`实例，而没有使用下划线。这些是我们对象的实际属性，我们将把之前声明的参数复制到这些属性中。

    在 `Construct` 的实现中，我们检索传递给我们的 `FArgumentsstruct` 的参数，并将它们存储在我们实际成员变量中，用于此实例。

    我们根据传入的内容分配 `Label` 和 `ButtonClicked`，然后实际上创建我们的小部件层次结构。我们使用通常的语法进行此操作，需要注意的是使用 `Text_Lambda` 来设置内部文本块的文本值。我们使用一个 `lambda` 函数通过 `Get()` 获取我们的 `Label` `TAttribute` 的值，将其转换为 `FText`，并将其存储为文本块的 `Text` 属性。

    现在我们已经声明了 `SWidget`，我们需要创建一个包装 `UWidget` 对象，以便将其暴露给 UMG 系统，这样设计师就可以在 **WYSIWYG** 编辑器中使用该小部件。这个类将被称为 `UCustomButtonWidget`，它继承自 `UWidget` 而不是 `SWidget`。

    `UWidget` 对象需要一个对其拥有的实际 `SWidget` 的引用，因此我们在类中放置一个受保护的成员，将其存储为共享指针。

    声明了一个构造函数，以及一个可以在蓝图设置中的 `ButtonClicked` 代理。我们还镜像了一个标记为 `BlueprintReadWrite` 的 `Label` 属性，以便可以在 UMG 编辑器中设置。

    因为我们希望能够将我们的按钮标签绑定到代理，所以我们添加了我们的最后一个成员变量，即一个返回 `String` 的代理。

    `SynchronizeProperties` 函数将已在我们的 `UWidget` 类中镜像的属性应用到我们关联的 `SWidget` 上。

    `RebuildWidget` 重建与 `UWidget` 关联的本地小部件。它使用 `SNew` 构建我们的 `SCustomButton` 小部件的实例，并使用 Slate 声明性语法将 UWidget 的 `OnButtonClicked` 方法绑定到本地小部件内部的 `ButtonClicked` 代理。这意味着当本地小部件被点击时，`UWidget` 将通过调用 `OnButtonClicked` 来接收通知。

    `OnButtonClicked` 通过 UWidget 的 `ButtonClicked` 代理重新广播来自本地按钮的点击事件。这意味着 UObjects 和 UMG 系统可以在没有本地按钮小部件的引用的情况下通知按钮被点击。我们可以绑定到 `UCustomButtonWidget::ButtonClicked` 以便我们得到通知。

    `OnButtonClicked` 然后返回 `FReply::Handled()` 以指示事件不需要进一步传播。在 `SynchronizeProperties` 中，我们调用父方法以确保父级中的任何属性都得到正确同步。

    我们使用 `OPTIONAL_BINDING` 宏将我们的 `UWidget` 类中的 `LabelDelegate` 代理链接到 `TAttribute`，进而链接到本地按钮的标签。重要的是要注意，`OPTIONAL_BINDING` 宏期望代理根据宏的第二个参数被命名为 `NameDelegate`。

    `OPTIONAL_BINDING`允许通过 UMG 创建的绑定覆盖值，但仅当 UMG 绑定有效时。

    这意味着当`UWidget`被告知更新自身时，例如，因为用户在 UMG 的详情面板中自定义了一个值，它将根据需要重新创建本地的`SWidget`，然后通过`SynchronizeProperties`复制在蓝图/UMG 中设置的值，以确保一切按预期继续工作。
