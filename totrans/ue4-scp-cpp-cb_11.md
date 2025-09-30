# 与 UE4 API 一起工作

**应用程序编程接口**（**API**）是程序员如何指令引擎，以及因此指令 PC 执行的方式。在本章的食谱中，我们将探索一些有趣的 API，如下所示：

+   核心日志 API - 定义自定义日志类别

+   核心日志 API - 使用 FMessageLog 将消息写入消息日志

+   核心数学 API - 使用 FRotator 进行旋转

+   核心数学 API - 使用 FQuat 进行旋转

+   核心数学 API - 使用 FRotationMatrix 进行旋转，使一个对象面向另一个对象

+   地形 API - 使用 Perlin 噪声生成地形

+   Foliage API - 以程序方式将树木添加到你的级别

+   地形和 Foliage API - 使用地形和 Foliage API 进行地图生成

+   GameplayAbilities API - 使用游戏控制触发角色的游戏能力

+   GameplayAbilities API - 使用 AttributeSet 实现属性

+   GameplayAbilities API - 使用 GameplayEffect 实现增益效果

+   GameplayTags API - 将 GameplayTags 附加到角色

+   GameplayTasks API - 使用 GameplayTasks 使事物发生

+   HTTP API - 使用网络请求下载网页

+   HTTP API - 显示下载进度

# 简介

UE4 的所有功能都封装在模块中，包括非常基本和核心的功能。每个模块都有一个 API。要使用 API，有一个非常重要的链接步骤，你必须在一个名为`ProjectName.Build.cs`的文件中列出你将在构建中使用的所有 API，该文件位于你的解决方案资源管理器窗口中。

不要将你的任何 UE4 项目命名为与 UE4 API 名称完全相同！

UE4 引擎内部包含各种 API，它们将功能暴露给其各个基本部分。

UE4 引擎的基础功能，在编辑器中可用，相当广泛。从 C++代码中的功能实际上被分组到称为 API 的小部分中。在 UE4 代码库中，每个重要功能都有一个单独的 API 模块。这样做是为了保持代码库高度组织和模块化。

使用不同的 API 可能需要在你的`Build.cs`文件中进行特殊链接！如果你遇到构建错误，请确保已经存在与正确 API 的链接！

完整的 API 列表位于以下文档中：[`docs.unrealengine.com/latest/INT/API/`](https://docs.unrealengine.com/latest/INT/API/)。

# 技术要求

本章需要使用 Unreal Engine 4，并使用 Visual Studio 2017 作为 IDE。有关如何安装这两款软件及其要求的信息，请参阅本书的第一章，*UE4 开发工具*。

# 核心日志 API - 定义自定义日志类别

UE4 本身定义了几个日志类别，包括例如 `LogActor` 这样的类别，它包含与 `Actor` 类相关的任何日志消息，以及 `LogAnimation`，它记录动画相关的消息。一般来说，UE4 为每个模块定义一个单独的日志类别。这允许开发者将他们的日志消息输出到不同的日志流中。每个日志流的名称都作为前缀添加到输出消息中，如下面的示例日志消息所示：

```cpp
LogContentBrowser: Native class hierarchy updated for 
 'HierarchicalLODOutliner' in 0.0011 seconds. Added 1 classes and 2 
 folders. 
LogLoad: Full Startup: 8.88 seconds (BP compile: 0.07 seconds) 
LogStreaming:Warning: Failed to read file 
 '../../../Engine/Content/Editor/Slate/Common/Selection_16x.png' 
 error. 
LogExternalProfiler: Found external profiler: VSPerf 
```

这些日志消息是来自引擎的样本，每个都带有其日志类别的前缀。警告消息以黄色显示，并在前面添加了“警告”。

你在网上找到的示例代码倾向于使用 `LogTemp` 为 UE4 项目的自身消息，如下所示：

```cpp
UE_LOG( LogTemp, Warning, TEXT( "Message %d" ), 1 ); 
```

实际上，我们可以通过定义自己的自定义 `LogCategory` 来改进这个公式。

# 准备工作

准备一个 UE4 项目，你想要在其中定义一个自定义日志。打开一个将被包含在几乎所有使用此日志的文件中的头文件。

# 如何操作...

1.  打开你的项目的主头文件；例如，如果你的项目名称是 `Chapter_11`，那么你将打开 `Chapter_11.h`。添加以下代码行：

```cpp
#pragma once

#include "CoreMinimal.h"

DECLARE_LOG_CATEGORY_EXTERN(LogCh11, Log, All);
```

如在 `AssertionMacros.h` 中定义的，此声明有三个参数，如下所示：

+   `CategoryName`：这是正在定义的日志类别名称（这里为 `LogCh11`）

+   `DefaultVerbosity`：这是日志消息的默认详细程度

+   `CompileTimeVerbosity`：这是要烘焙到编译代码中的详细程度

1.  在你的项目的主 `.cpp` 文件中（在我们的例子中是 `Chapter_11.cpp`），包含以下代码行：

```cpp
#include "Chapter_11.h"
#include "Modules/ModuleManager.h"

IMPLEMENT_PRIMARY_GAME_MODULE( FDefaultGameModuleImpl, Chapter_11, "Chapter_11" );

DEFINE_LOG_CATEGORY(LogCh11);
```

1.  现在，我们可以在自己的脚本中使用这个日志类别。例如，打开你的项目的 `GameModeBase` 文件（在这种情况下，`Chapter_11GameModeBase.h`）并添加以下函数声明：

```cpp
UCLASS()
class CHAPTER_11_API AChapter_11GameModeBase : public AGameModeBase
{
    GENERATED_BODY()

    void BeginPlay();
};
```

1.  然后，转到实现部分（`Chapter_11GameModeBase.cpp`）并使用以下代码作为各种显示类别的示例：

```cpp
#include "Chapter_11GameModeBase.h"
#include "Chapter_11.h"

void AChapter_11GameModeBase::BeginPlay()
{
 // Traditional Logging
 UE_LOG(LogTemp, Warning, TEXT("Message %d"), 1);

 // Our custom log type
 UE_LOG(LogCh11, Display, TEXT("A display message, log is working" ) ); // shows in gray 
 UE_LOG(LogCh11, Warning, TEXT("A warning message"));
 UE_LOG(LogCh11, Error, TEXT("An error message "));
}
```

1.  编译你的脚本。之后，打开世界设置菜单，将游戏模式覆盖属性设置为 `Chapter_11GameModeBase`，然后运行游戏：

![](img/5e64d7dd-1c39-4e0b-80fe-1556f7b9cf2b.png)

从输出日志窗口中记录的消息的位置

如你所见，我们可以看到我们的自定义日志消息正在显示出来！

# 它是如何工作的...

日志是通过将消息输出到输出日志（窗口 | 开发者工具 | 输出日志）以及一个文件来工作的。所有输出到输出日志的信息也会镜像到一个位于你的项目 `/Saved/Logs` 文件夹中的简单文本文件。日志文件的扩展名为 `.log`，最新的是名为 `YourProjectName.log` 的文件。

# 还有更多...

你可以使用以下控制台命令在编辑器中启用或抑制特定日志通道的日志消息：

```cpp
Log LogName off // Stop LogName from displaying at the output 
Log LogName Log // Turn LogName's output on again 
```

如果你想要编辑某些内置日志类型的输出级别的初始值，你可以使用一个 C++类来为`engine.ini`配置文件创建更改。你可以在`engine.ini`配置文件中更改初始值。

更多详情请见[`wiki.unrealengine.com/Logs,_Printing_Messages_To_Yourself_During_Runtime`](https://wiki.unrealengine.com/Logs,_Printing_Messages_To_Yourself_During_Runtime)。

`UE_LOG`将输出发送到输出窗口。如果你还想使用这个更专业的消息日志窗口，你可以使用`FMessageLog`对象来写入你的输出消息。`FMessageLog`将写入消息日志和输出窗口。有关详细信息，请参阅下一道菜谱。

# 核心日志 API - 使用 FMessageLog 将消息写入消息日志

`FMessageLog`是一个对象，它允许你同时将输出消息写入消息日志（窗口 | 开发者工具 | 消息日志）和输出日志（窗口 | 开发者工具 | 输出日志）。

# 准备工作

准备好你的项目并准备记录到消息日志中的信息。在 UE4 编辑器中显示消息日志（窗口 | 开发者工具 | 消息日志）。

# 如何做到...

1.  在你的主要头文件（`ProjectName.h`）中添加`#define`，将`LOCTEXT_NAMESPACE`定义为你的代码库中独特的东西：

```cpp
#define LOCTEXT_NAMESPACE "Chapter11Namespace"
```

这个`#define`被`LOCTEXT()`宏使用，我们用它来生成`FText`对象，但在输出消息中是不可见的。

1.  在某个非常全局的地方声明你的`FMessageLog`，你可以在`ProjectName.h`文件中使用`extern`。考虑以下代码片段作为示例：

```cpp
#define LOCTEXT_NAMESPACE "Chapter11Namespace"
#define FTEXT(x) LOCTEXT(x, x) 

extern FName LoggerName;

void CreateLog(FName logName);
```

1.  然后，通过在`.cpp`文件中定义它并使用`MessageLogModule`注册它来创建你的`FMessageLog`。确保在构建时给你的记录器一个清晰且独特的名称。这是你的日志类别，它将出现在输出日志中你的日志消息的左侧。例如，`ProjectName.cpp`：

```cpp
#include "Chapter_11.h"
#include "Modules/ModuleManager.h"
#include "MessageLog/Public/MessageLogModule.h"
#include "MessageLog.h"

// ...

FName LoggerName("MessageLogChapter11");

void CreateLog(FName logName)
{
    FMessageLogModule& MessageLogModule = FModuleManager::LoadModuleChecked<FMessageLogModule>("MessageLog");
    FMessageLogInitializationOptions InitOptions;
    InitOptions.bShowPages = true;
    InitOptions.bShowFilters = true;
    FText LogListingName = FTEXT("Chapter 11's Log Listing");
    MessageLogModule.RegisterLogListing(logName, LogListingName, InitOptions);
}
```

1.  然后，回到你的代码中的某个地方，实际创建日志并使用它。例如，我们可以在以下`GameModeBase`类的`BeginPlay`方法中添加以下内容：

```cpp
void AChapter_11GameModeBase::BeginPlay()
{
    // 11-01 - Core/Logging API - Defining a custom log
    // category
    // Traditional Logging
    UE_LOG(LogTemp, Warning, TEXT("Message %d"), 1);

    // Our custom log type
    UE_LOG(LogCh11, Display, TEXT("A display message, log is working" ) ); // shows in gray 
    UE_LOG(LogCh11, Warning, TEXT("A warning message"));
    UE_LOG(LogCh11, Error, TEXT("An error message "));

 // 11-02 - Core/Logging API - FMessageLog to write 
    // messages to the Message Log
 CreateLog(LoggerName);
 // Retrieve the Log by using the LoggerName. 
 FMessageLog logger(LoggerName);
 logger.Warning(FTEXT("A warning message from gamemode"));
}
```

`LOCTEXT`（第一个参数）的`KEY`必须是唯一的，否则你会得到一个之前哈希过的字符串。如果你愿意，你可以包含一个`#define`，重复`LOCTEXT`的参数两次，就像我们之前做的那样：

`#define FTEXT(x) LOCTEXT(x, x)`

1.  使用以下代码记录你的消息：

```cpp
logger.Info( FTEXT( "Info to log" ) ); 
logger.Warning( FTEXT( "Warning text to log" ) ); 
logger.Error( FTEXT( "Error text to log" ) ); 
```

这段代码使用了我们之前定义的`FTEXT()`宏。确保它在你的代码库中。

# 它是如何工作的...

这个菜谱显示了一条消息到消息日志。正如我们之前讨论的，你可以在消息日志（窗口 | 开发者工具 | 消息日志）和输出日志（窗口 | 开发者工具 | 输出日志）中看到记录的信息。

在初始化后再次构建你的消息日志会获取原始消息日志的一个副本。例如，在代码的任何地方，你可以编写以下代码：

`FMessageLog( LoggerName ).Info(FTEXT( "An info message"));`

# 核心数学 API - 使用 FRotator 进行旋转

UE4 中的旋转实现如此完整，以至于选择如何旋转对象可能会很困难。主要有三种方法：`FRotator`、`FQuat`和`FRotationMatrix`。本食谱概述了三种不同旋转方法中第一种——`FRotator`的构建和使用。使用此方法以及以下两个食谱，你可以选择用于旋转对象的方法。

# 准备工作

打开一个 UE4 项目，该项目中有一个你可以通过 C++接口获取的对象。例如，你可以构建一个从`Actor`派生的 C++类`Coin`来测试旋转。通过 C++代码覆盖`Coin::Tick()`方法来应用你的旋转。或者，你也可以在 Blueprints 的`Tick`事件中调用这些旋转函数。

在此示例中，我们将通过使用 Actor 组件以每秒一度速率旋转对象。实际的旋转将是对象创建以来的累积时间。为了获取此值，我们将调用`GetWorld()->TimeSeconds`。

# 如何做到这一点...

1.  在“模式”选项卡下，“放置”部分和“基本”部分下，将一个立方体对象拖放到你的场景中。

1.  在“详细信息”选项卡中，转到“变换”组件并将“移动性”属性更改为“可移动”。

1.  之后，点击“添加组件”按钮并选择“新 C++组件”。

1.  在弹出的菜单中选择“Actor 组件”并选择“下一步”：

![](img/7df035bd-fc84-445a-bc58-40af6b94ac73.png)

1.  从那里，给你的组件命名，例如，`RotateActorComponent`，然后按创建类按钮。

1.  构建你的`FRotator`。`FRotators`可以使用库存的俯仰、偏航和滚转构造函数来构建，如下例所示：

```cpp
FRotator( float InPitch, float InYaw, float InRoll ); 
```

1.  你的`FRotator`将按照以下方式构建：

```cpp
FRotator rotator( 0, GetWorld()->TimeSeconds, 0 ); 
```

1.  UE4 中对象的标准化方向是面向下*+X*轴。右是*+Y*轴，上是*+Z*。

1.  俯仰是关于*Y*轴（横跨）的旋转，偏航是关于*Z*轴（向上）的旋转，滚转是关于*X*轴的旋转。以下三点最能理解这一点：

+   **俯仰**：如果你在 UE4 标准坐标系中想象一架飞机，*Y*轴沿着机翼（俯仰使其前后倾斜）

+   **偏航**：*Z*轴垂直向上和向下（偏航使其左右转动）

+   **滚转**：*X*轴沿着飞机的机身（滚转使其做桶滚）

你应该注意，在其他约定中，*X*轴是俯仰，*Y*轴是偏航，*Z*轴是滚转。

1.  使用`SetActorRotation`成员函数将`FRotator`应用到你的 actor 上，如下所示：

```cpp
// Called every frame
void URotateActorComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

 FRotator rotator(0, GetWorld()->TimeSeconds, 0);
 GetOwner()->SetActorRotation(rotator);
}
```

# 核心数学 API - 使用 FQuat 进行旋转

四元数听起来很吓人，但它们极其容易使用。你可能想通过查看以下视频来回顾它们背后的理论数学：

+   Numberphile 的*奇妙的四元数*：[`www.youtube.com/watch?v=3BR8tK-LuB0`](https://www.youtube.com/watch?v=3BR8tK-LuB0)

+   Jim Van Verth 的《*理解四元数*》：[`gdcvault.com/play/1017653/Math-for-Game-Programmers-Understanding`](http://gdcvault.com/play/1017653/Math-for-Game-Programmers-Understanding)

然而，我们不会在这里介绍数学背景！实际上，你不需要对四元数的数学背景有太多了解就能有效地使用它们。

# 准备工作

准备好一个项目和一个具有覆盖`::Tick()`函数的`Actor`，我们可以将 C++代码输入其中。

# 如何做到这一点...

1.  要构造一个四元数，最好使用的构造函数如下：

```cpp
FQuat( FVector Axis, float AngleRad ); 
```

四元数有四元数加法、四元数减法、标量乘法和标量除法等定义，以及其他函数。它们在以任意角度旋转物体和指向其他物体方面非常有用。

1.  例如，如果你想在`RotateActorComponent.cpp`文件中使用 FQuat，它看起来会类似于这样：

```cpp
void URotateActorComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    // 11-04 - Rotation using FQuat
 FQuat quat = FQuat(FVector(0, 1, 0), GetWorld()->TimeSeconds * PI / 4.f);
 GetOwner()->SetActorRotation(quat);

}
```

编译你的代码并返回游戏后，你应该注意到立方体以恒定速度移动：

![图片](img/ddb86012-77dc-4859-ad2f-ea3689f90875.png)

# 它是如何工作的...

四元数有点奇怪，但使用它们相当简单。如果*v*是旋转的轴，而![图片](img/6edc4c6a-2cc5-4162-9d74-cd70b3b729b3.jpg)是旋转角度的大小，那么我们得到以下四元数分量的方程：

![图片](img/e9dc711a-8f43-48f8-bc68-a56de59ad11f.jpg)

因此，例如，绕![图片](img/ac235d9d-5ef7-46a7-8ce4-0a18ed80d558.jpg)旋转一个角度![图片](img/69cf02ef-9e27-44a6-9a4b-efdc00908cd8.jpg)的四元数成分如下：

![图片](img/d3c4d321-4172-4dbe-9974-d6e0b6c8fe89.jpg)

四元数的四个分量中的三个(*x*，*y*和*z*)定义了旋转的轴（通过旋转角度的一半的正弦进行缩放），而第四个分量(*w*)只有旋转角度的一半的余弦。

# 还有更多...

作为向量本身的四元数也可以旋转。只需提取四元数的(*x*，*y*，*z*)分量，归一化，然后旋转该向量。从新的单位向量构造一个新的四元数，具有所需的旋转角度。

将四元数相乘表示一系列连续发生的旋转。例如，绕*X*轴旋转 45 度，然后绕*Y*轴旋转 45 度，将会组合成以下形式：

```cpp
FQuat( FVector( 1, 0, 0 ), PI/4.f ) * 
FQuat( FVector( 0, 1, 0 ), PI/4.f ); 
```

这将给出一个类似以下的结果：

![图片](img/f4bbdb04-e68b-48f2-abea-f943c4029dfd.png)

# API – 使用 FRotationMatrix 进行旋转以使一个对象面向另一个对象

`FRotationMatrix`提供了一系列`::Make*`例程来构建矩阵。它们易于使用，并且有助于使一个对象面向另一个对象。假设你有两个对象，其中一个对象在跟随另一个对象。我们希望跟随者的旋转始终面向它所跟随的对象。`FRotationMatrix`的构造方法使这变得容易实现。

# 准备工作

在场景中有两个 actor，其中一个应该面向另一个。

# 如何做到...

1.  为跟随者添加一个新的 C++ Actor 组件，称为 `FollowActorComponent`（如果你需要帮助，请参阅 *Core/Math API – 使用 FRotator 进行旋转* 菜单）。 

1.  从 `FollowActorComponent.h` 文件中，我们需要有一个对我们想要跟随的对象的引用，因此添加以下内容：

```cpp
UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class CHAPTER_11_API UFollowActorComponent : public UActorComponent
{
    GENERATED_BODY()

public: 
    // Sets default values for this component's properties
    UFollowActorComponent();

protected:
    // Called when the game starts
    virtual void BeginPlay() override;

public: 
    // Called every frame
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

    UPROPERTY(EditAnywhere)
 AActor * target;
};
```

1.  然后，在 `FollowActorComponent.cpp` 文件中，在 `TickComponent` 函数中，查看 `FRotationMatrix` 类下的可用构造函数。有一系列构造函数可供使用，允许你通过重新定位一个或多个 *X*、*Y* 或 *Z* 轴来指定一个对象（从原生位置）的旋转，这些构造函数以 `FRotationMatrix::Make*()` 模式命名。

1.  假设你为你的 actor 有一个默认的股票方向（其中前进方向朝下沿 *+X*-轴，向上方向朝上沿 *+Z*-轴），找到从跟随者到他们想要跟随的对象的向量，如图中所示：

```cpp
// Called every frame
void UFollowActorComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

 FVector toFollow = target->GetActorLocation() - GetOwner()->GetActorLocation();

 FMatrix rotationMatrix = FRotationMatrix::MakeFromXZ(toFollow, GetOwner()->GetActorUpVector());

 GetOwner()->SetActorRotation(rotationMatrix.Rotator());

}
```

1.  编译你的脚本，并在“跟随者组件”的详细信息选项卡中分配目标属性。这可以通过使用属性右侧的吸管按钮或使用下拉列表来完成：

![](img/37b4eeb3-2998-4d99-b3cf-0d6e219be4d6.png)

如果一切顺利，你应该看到 actor 正确旋转以面对目标：

![](img/c28f774e-1745-4d24-bcd7-8d7986021a0c.png)

# 它是如何工作的...

通过调用正确的函数，根据你的对象的原生方向，使一个对象看向另一个对象，并指定一个期望的向上向量。通常，你想要重新定位 *X*-轴（前进），同时指定 *Y*-轴（右）或 *Z*-轴（上）向量（`FRotationMatrix::MakeFromXY()`）。例如，要使一个 actor 沿着 `lookAlong` 向量看去，其右侧朝向右侧，我们需要为它构造并设置 `FRotationMatrix`，如下所示：

```cpp
FRotationMatrix rotationMatrix = FRotationMatrix::MakeFromXY( 
 lookAlong, right ); 
actor->SetActorRotation( rotationMatrix.Rotator() ); 
```

# GameplayAbilities API – 使用游戏控制触发角色的游戏能力

**GameplayAbilities** API 可以用来将 C++ 函数附加到某些按钮的点击上，触发游戏单位在游戏过程中响应按键事件时展示其能力。在这个菜谱中，我们将向你展示如何做到这一点。

# 准备工作

列出并描述你的游戏角色的能力。你需要知道你的角色在响应按键事件时做什么，以便在这个菜谱中编写代码。

在这里我们需要使用几个对象；它们如下所示：

+   `UGameplayAbility` 类——这是为了从 `UGameplayAbility` 类派生 C++ 类实例所必需的，每个能力都有一个派生类：

+   通过覆盖可用的函数，如 `UGameplayAbility::ActivateAbility`、`UGameplayAbility::InputPressed`、`UGameplayAbility::CheckCost`、`UGameplayAbility::ApplyCost`、`UGameplayAbility::ApplyCooldown` 等，在 `.h` 和 `.cpp` 文件中定义每个能力的作用。

+   `GameplayAbilitiesSet` 是一个 `DataAsset` 派生对象，它包含一系列枚举的命令值，以及定义特定输入命令行为的相应 `UGameplayAbility` 派生类的蓝图。每个 GameplayAbility 都由一个按键或鼠标点击启动，这已在 `DefaultInput.ini` 中设置。

# 如何做到这一点...

在以下代码中，我们将实现一个名为 `UGameplayAbility_Attack` 的 `UGameplayAbility` 派生类，用于 `Warrior` 类对象。我们将把这个游戏功能附加到输入命令字符串 `Ability1` 上，我们将在左鼠标按钮点击时激活它：

1.  打开你的 `.Build.cs` 文件（在我们的例子中，`Chapter_11.Build.cs`），并添加以下依赖项：

```cpp
using UnrealBuildTool;

public class Chapter_11 : ModuleRules
{
    public Chapter_11(ReadOnlyTargetRules Target) : 
    base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] {
        "Core", "CoreUObject", "Engine", "InputCore" });
        PublicDependencyModuleNames.AddRange(new string[] {  
        "GameplayAbilities", "GameplayTags", "GameplayTasks" });

        PrivateDependencyModuleNames.AddRange(new string[] { });
    }
}
```

1.  编译你的代码。

1.  从 Unreal 编辑器，转到 设置 | 插件。

1.  从弹出的菜单中，搜索 `GameplayAbilities` 并勾选它。你会得到一个确认消息。点击是按钮：

![图片](img/fb27ba3b-6389-4a08-9651-614f0d31f7dc.png)

1.  之后，点击立即重启按钮。类应该被正确添加到你的项目中。

1.  现在，通过从内容浏览器添加新 | 新 C++ 类... 选择添加 C++ 类向导，并检查显示所有类选项。从那里，输入 `gameplayability` 并选择基础 GameplayAbility 类以基于我们的新类：

![图片](img/b731ae0b-66a3-40f4-a9d8-dae61e50aafb.png)

1.  给新的游戏能力命名为 `GameplayAbility_Attack` 并按创建类。

1.  至少，你想要重写以下内容：

+   `UGameplayAbility_Attack::CanActivateAbility` 成员函数用于指示当演员被允许调用能力时。

+   `UGameplayAbility_Attack::CheckCost` 函数用于指示玩家是否有能力使用一个能力。这一点非常重要，因为如果这个函数返回 false，能力调用应该失败。

+   `UGameplayAbility_Attack::ActivateAbility` 成员函数用于编写当 `Warrior` 的 `Attack` 能力被激活时 `Warrior` 将要执行的代码。

+   `UGameplayAbility_Attack::InputPressed` 成员函数以及响应分配给能力的按键输入事件：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "Abilities/GameplayAbility.h"
#include "GameplayAbility_Attack.generated.h"

UCLASS()
class CHAPTER_11_API UGameplayAbility_Attack : public UGameplayAbility
{
  GENERATED_BODY()

 /** Returns true if this ability can be activated
        right now. Has no side effects */
 virtual bool CanActivateAbility(const FGameplayAbilitySpecHandle Handle, const FGameplayAbilityActorInfo* ActorInfo, const FGameplayTagContainer* SourceTags = nullptr, const FGameplayTagContainer* TargetTags = nullptr, OUT FGameplayTagContainer* OptionalRelevantTags = nullptr) const {
 UE_LOG(LogTemp, Warning, TEXT("ability_attack 
        CanActivateAbility!"));
 return true;
 }

 /** Checks cost. returns true if we can pay for the
    ability. False if not */
 virtual bool CheckCost(const FGameplayAbilitySpecHandle Handle, 
    const FGameplayAbilityActorInfo* ActorInfo, OUT 
    FGameplayTagContainer* OptionalRelevantTags = nullptr) const {
 UE_LOG(LogTemp, Warning, TEXT("ability_attack CheckCost!"));
 return true;
 //return Super::CheckCost( Handle, ActorInfo, 
        //OptionalRelevantTags );
 }

 virtual void ActivateAbility(const FGameplayAbilitySpecHandle 
    Handle,
 const FGameplayAbilityActorInfo* ActorInfo, const 
        FGameplayAbilityActivationInfo ActivationInfo,
 const FGameplayEventData* TriggerEventData)
 {
 UE_LOG(LogTemp, Warning, TEXT("Activating 
        ugameplayability_attack().. swings weapon!"));
 Super::ActivateAbility(Handle, ActorInfo, ActivationInfo,  
        TriggerEventData);
 }

 /** Input binding stub. */
 virtual void InputPressed(const FGameplayAbilitySpecHandle 
    Handle, const FGameplayAbilityActorInfo* ActorInfo, const 
    FGameplayAbilityActivationInfo ActivationInfo) {
 UE_LOG(LogTemp, Warning, TEXT("ability_attack 
        inputpressed!"));
 Super::InputPressed(Handle, ActorInfo, ActivationInfo);
 }

};
```

1.  在 UE4 编辑器中从你的 `UGameplayAbility_Attack` 对象派生一个蓝图类。

1.  在编辑器内部，导航到内容浏览器，通过以下步骤创建一个 `GameplayAbilitiesSet` 对象：

+   在内容浏览器上右键单击并选择杂项 | 数据资产：

![图片](img/777500fc-befc-4e6b-b1ca-8924737f7ecf.png)

+   在随后的对话框中，选择数据资产类为 `GameplayAbilitySet`：

![图片](img/90023d90-6b67-4cc4-896a-ace1d96ee8c5.png)

实际上，`GameplayAbilitySet` 对象是一个 `UDataAsset` 派生对象。它位于 `GameplayAbilitySet.h` 中，包含一个成员函数 `GameplayAbilitySet::GiveAbilities()`，我强烈建议你不要使用它，原因将在后续步骤中列出。

1.  将您的 `GameplayAbilitySet` 数据资产命名为与 `WarriorAbilitySet` 对象相关的内容，以便我们知道将其放入 `Warrior` 类中（例如，`WarriorAbilitySet`）。

1.  双击打开并编辑新的 `WarriorAbilitySet` 数据资产。通过在其内部的 `TArray` 对象上点击 + 来堆叠一个 `GameplayAbility` 类派生蓝图列表。您的 `UGameplayAbility_Attack` 对象必须出现在下拉菜单中：

![图片](img/01c90d63-775f-4669-8d74-a9e01192e4f7.png)

1.  我们现在需要创建一个从 `Character` 类派生的对象，以便我们可以包含这个能力集。在这个例子中，我们将把这个类命名为 `Warrior`。

1.  在您的 `Warrior` 类中添加一个 `UPROPERTY UGameplayAbilitySet* gameplayAbilitySet` 成员：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "GameplayAbilitySet.h"
#include "AbilitySystemInterface.h"
#include "Warrior.generated.h"

#define FS(x,...) FString::Printf( TEXT( x ), __VA_ARGS__ )

UCLASS()
class CHAPTER_11_API AWarrior : public ACharacter, public IAbilitySystemInterface
{
    GENERATED_BODY()

public:
    // Sets default values for this character's properties
    AWarrior();

protected:
    // Called when the game starts or when spawned
    virtual void BeginPlay() override;

public: 
    // Called every frame
    virtual void Tick(float DeltaTime) override;

    // Called to bind functionality to input
    virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

 // Lists key triggers for various abilities for the 
    // player.
 // Selects an instance of UGameplayAbilitySet (which is a      //    \
    UDataAsset derivative
 // that you construct in the Content Browser).
 UPROPERTY(EditAnywhere, BlueprintReadWrite, Category =
    Stats)
 UGameplayAbilitySet* gameplayAbilitySet;

 // The AbilitySystemComponent itself
 UPROPERTY(EditAnywhere, BlueprintReadWrite, Category =
    Stats)
 UAbilitySystemComponent* AbilitySystemComponent;

 // IAbilitySystemInterface implementation:
 virtual UAbilitySystemComponent* GetAbilitySystemComponent() const { return AbilitySystemComponent; }

};

```

确保您的 `Actor` 类派生也派生自 `UAbilitySystemInterface` 接口。这一点非常重要，以便 `(Cast<IAbilitySystemInterface>(yourActor))->GetAbilitySystemComponent()` 调用成功。

1.  创建 `Warrior` 类的蓝图，并将游戏能力集设置为之前创建的 `Warrior Ability Set`，并将能力系统组件设置为能力系统组件：

![图片](img/16747445-09a2-4b0e-b317-85c07b104892.png)

如果您看不到能力系统组件，请关闭并重新打开蓝图。

1.  完成后，将 `MyWarrior` 分配为您的游戏模式的默认 Pawn 类。

1.  编译、运行并选择在内容浏览器中（在第 5 步到第 7 步创建）的 `WarriorAbilitySet` 作为 `Warrior` 能够使用的技能。

1.  在您的演员构建一段时间后，调用 `gameplayAbilitySet->GiveAbilities( abilitySystemComponent );` 或进入循环，如下一步所示，其中为 `gameplayAbilitySet` 中列出的每个能力调用 `abilitySystemComponent->GiveAbility()`。

1.  为 `AWarrior::SetupPlayerInputComponent( UInputComponent* Input )` 编写一个覆盖，以将输入控制器连接到战士的 `GameplayAbility` 激活。完成后，遍历 `GameplayAbilitySet` 的 `Abilities` 组中列出的每个 `GameplayAbility`：

```cpp
#include "AbilitySystemComponent.h"

// ...

// Called to bind functionality to input
void AWarrior::SetupPlayerInputComponent(UInputComponent* Input)
{
    Super::SetupPlayerInputComponent(Input);

    // Connect the class's AbilitySystemComponent 
    // to the actor's input component 
    AbilitySystemComponent->BindToInputComponent(Input);

    // Go thru each BindInfo in the gameplayAbilitySet. 
    // Give & try and activate each on the 
    // AbilitySystemComponent. 
    for (const FGameplayAbilityBindInfo& BindInfo :
        gameplayAbilitySet->Abilities)
    {

        FGameplayAbilitySpec spec(
            // Gets you an instance of the UClass 
            BindInfo.GameplayAbilityClass->
            GetDefaultObject<UGameplayAbility>(),
            1, (int32)BindInfo.Command);

        // STORE THE ABILITY HANDLE FOR LATER INVOKATION 
        // OF THE ABILITY 
        FGameplayAbilitySpecHandle abilityHandle =
            AbilitySystemComponent->GiveAbility(spec);

        // The integer id that invokes the ability 
        // (ith value in enum listing) 
        int32 AbilityID = (int32)BindInfo.Command;

        // CONSTRUCT the inputBinds object, which will 
        // allow us to wire-up an input event to the 
        // InputPressed() / InputReleased() events of 
        // the GameplayAbility. 
        FGameplayAbilityInputBinds inputBinds(
            // These are supposed to be unique strings that 
            // define what kicks off the ability for the actor        
            // instance. 
            // Using strings of the format 
            // "ConfirmTargetting_Player0_AbilityClass" 
            FS("ConfirmTargetting_%s_%s", *GetName(),
                *BindInfo.GameplayAbilityClass->GetName()),
            FS("CancelTargetting_%s_%s", *GetName(),
                *BindInfo.GameplayAbilityClass->GetName()),
            "EGameplayAbilityInputBinds", // The name of the
            // ENUM that has the abilities listing
            // (GameplayAbilitySet.h). 
            AbilityID, AbilityID
        );
        // MUST BIND EACH ABILITY TO THE INPUTCOMPONENT,
        // OTHERWISE THE ABILITY CANNOT "HEAR" INPUT EVENTS. 
        // Enables triggering of InputPressed() / 
        // InputReleased() events, which you can in-turn use 
        // to call 
        // TryActivateAbility() if you so choose. 
        AbilitySystemComponent->BindAbilityActivationToInputComponent(
            Input, inputBinds
        );

        // Test-kicks the ability to active state. 
        // You can try invoking this manually via your 
        // own hookups to keypresses in this Warrior class 
        // TryActivateAbility() calls ActivateAbility() if 
        // the ability is indeed invokable at this time 
        // according to rules internal to the Ability's class 
        // (such as cooldown is ready and cost is met) 
        AbilitySystemComponent->TryActivateAbility(
            abilityHandle, 1);
    }
}

```

1.  编译您的代码然后玩游戏：

![图片](img/7cad0887-cbad-49f8-966b-83e6fc2d9f0d.png)

# 它是如何工作的...

您必须通过一系列对 `UAbilitySystemComponent::GiveAbility( spec )` 的调用，使用适当构造的 `FGameplayAbilitySpec` 对象，将一组 `UGameplayAbility` 对象子类化并链接到您的演员的 `UAbilitySystemComponent` 对象。这样做会给您的演员配备这批 `GameplayAbilities`。每个 `UGameplayAbility` 的功能，以及其成本、冷却时间和激活，都整齐地包含在您将要构建的 `UGameplayAbility` 类派生类中。

不要使用 `GameplayAbilitySet::GiveAbilities()` 成员函数，因为它不会给您提供访问您实际需要的 `FGameplayAbilitySpecHandle` 对象集合，这些对象稍后用于将能力绑定并调用到输入组件。

# 还有更多...

你还想要仔细编写 `GameplayAbility.h` 头文件中可用的其他函数，包括以下实现的函数：

+   `SendGameplayEvent`：这是一个通知 GameplayAbility 发生了某些通用游戏事件的函数。

+   `CancelAbility`：这是一个在能力使用中途停止并给能力一个中断状态的函数。

+   请记住，在 `UGameplayAbility` 类声明的底部附近有许多现有的 `UPROPERTY` 指示符，它们在添加或删除某些 `GameplayTags` 时会激活或取消能力。有关更多详细信息，请参阅以下 *GameplayTags API – 将 GameplayTags 关联到 Actor* 菜谱。

+   还有更多！探索 API 并实现你在代码中找到的有用的函数。

# 参见

+   `GameplayAbilities` API 是一系列丰富且紧密交织的对象和函数。探索 `GameplayEffects`、`GameplayTags` 和 `GameplayTasks` 以及它们如何与 `UGameplayAbility` 类集成，以全面了解库提供的功能。你可以在此处了解更多关于 API 的信息：[`api.unrealengine.com/INT/API/Plugins/GameplayAbilities/index.html`](https://api.unrealengine.com/INT/API/Plugins/GameplayAbilities/index.html)

# GameplayAbilities API - 使用 UAttributeSet 实现统计数据

`GameplayAbilities` API 允许你将一组属性，即 `UAttributeSet`，关联到一个 Actor 上。`UAttributeSet` 描述了适合该 Actor 游戏内属性的属性，例如 `Hp`（生命值）、`Mana`（法力）、`Speed`（速度）、`Armor`（护甲）、`AttackDamage`（攻击伤害）等等。你可以定义一个通用于所有 Actor 的单个游戏内属性集，或者为不同类别的 Actor 定义几个不同的属性集。

# 准备工作

`AbilitySystemComponent` 是你首先需要添加到你的 Actor 上的，以便它们能够使用 `GameplayAbilities` API 和 `UAttributeSet` 类。为了定义你的自定义 `UAttributeSet`，你只需从 `UAttributeSet` 基类派生，并使用你自己的 `UPROPERTY` 成员扩展基类。之后，你必须将你的自定义 `AttributeSet` 注册到你的 `Actor` 类的 `AbilitySystemComponent`。

# 如何做到...

1.  如果你还没有这样做，请完成 *GameplayAbilities API – 使用游戏控制触发 Actor 的游戏能力* 菜谱的步骤 1-4，以在 `ProjectName.Build.cs` 文件中链接到 `GameplayAbilities` API 并启用其功能。

1.  通过转到内容浏览器并选择添加新项 | 添加 C++ 类来创建一个新的 C++ 类。从添加 C++ 类菜单中，勾选显示所有类选项。从那里，输入 `attr` 并选择 `AttributeSet` 作为你的父类。然后，点击下一步按钮：

![图片](img/98d22785-cb67-4d52-b392-4c4424e5a76e.png)

1.  给这个类命名为 `GameUnitAttributeSet` 并点击创建类：

![图片](img/df7f58db-1f4a-46d1-a5e5-3c5e43478d3d.png)

一旦创建，为每个演员在其属性集中拥有的`UPROPERTY`指定符装饰这个类。

1.  例如，你可能希望声明你的`UAttributeSet`派生类，类似于以下代码片段中给出的内容：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "AttributeSet.h"
#include "GameUnitAttributeSet.generated.h"

/**
 * 
 */
UCLASS(Blueprintable, BlueprintType)
class CHAPTER_11_API UGameUnitAttributeSet : public UAttributeSet
{
    GENERATED_BODY()

public:
 UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = GameUnitAttributes)
 float Hp;

 UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = GameUnitAttributes)
 float Mana;

 UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = GameUnitAttributes)
 float Speed;
};

```

如果你的代码是网络化的，你可能希望在每个`UPROPERTY`指定符上启用复制，并在`UPROPERTY`宏中使用复制声明。

1.  在你的`Actor`类内部将`GameUnitAttributeSet`与`AbilitySystemComponent`连接起来。我们可以通过打开之前创建的`Warrior`类文件并添加以下函数声明来实现这一点：

```cpp
virtual void PostInitializeComponents() override;
```

1.  然后，打开`Warrior.cpp`并添加以下`#include`：

```cpp
#include "GameUnitAttributeSet.h"
```

1.  之后，实现该函数：

```cpp
void AWarrior::PostInitializeComponents()
{
    Super::PostInitializeComponents();

    if(AbilitySystemComponent)
    {
        AbilitySystemComponent->InitStats(UGameUnitAttributeSet::StaticClass(), NULL);
    }
}
```

你可以将这个调用放在`PostInitializeComponents()`中，或者在其之后调用的代码中。

1.  一旦你注册了`UAttributeSet`，你就可以继续下一个菜谱，并将`GameplayEffect`应用于属性集中的某些元素。

1.  确保你的`Actor`类对象通过从它派生来实现`IAbilitySystemInterface`。这非常重要，因为`UAbilitySet`对象将尝试在代码的多个位置将其转换为`IAbilitySystemInterface`以调用其`GetAbilitySystemComponent()`。

# 它是如何工作的...

`UAttributeSets`只是允许你枚举和定义不同演员的属性。`GameplayEffects`将是改变特定演员属性的手段。

# 还有更多...

你可以编写`GameplayEffects`的定义，这些定义将作用于`AbilitySystemComponent`的`AttributeSet`集合。你也可以编写`GameplayTasks`，用于在特定时间或跟随特定事件运行通用函数，或者甚至在响应标签添加时（`GameplayTagResponseTable.cpp`）。你可以定义`GameplayTags`来修改`GameplayAbility`的行为，以及在游戏过程中选择和匹配游戏单位。

# GameplayAbilities API – 使用 GameplayEffect 实现增益效果

增益效果只是引入到游戏单位属性中的临时、永久或重复变化的效果，这些属性来自其`AttributeSet`。增益效果可以是好的或坏的，提供奖励或惩罚。例如，你可能有一个减速增益效果，将单位速度减半，一个增加单位速度 2 倍的翅膀增益效果，或者一个每 5 秒恢复`5 hp`，持续 3 分钟的炽天使增益效果。`GameplayEffect`影响附着在演员`AbilitySystemComponent`的`UAttributeSet`中的单个游戏属性。

# 准备工作

在游戏中，头脑风暴你的游戏单位的效果。确保你已经创建了一个`AttributeSet`，如前一个菜谱中所示，其中包含你想要影响的游戏属性。选择一个要实现的效果，并按照随后的步骤使用你的示例进行操作。

您可能希望通过转到输出日志并输入` `` ``，然后输入`Log LogAbilitySystem All`将`LogAbilitySystem`转换为`VeryVerbose`设置。这将显示输出日志中来自`AbilitySystem`的更多信息，使您更容易看到系统中的情况。

# 如何操作...

在以下步骤中，我们将构建一个快速`GameplayEffect`，为所选单位的`AttributeSet`恢复`50 hp`：

1.  打开我们之前创建的`Warrior.h`文件。在那里，添加以下函数定义：

```cpp
void TestGameplayEffect();
```

1.  之后，打开`Warrior.cpp`并添加以下方法：

```cpp
inline UGameplayEffect* ConstructGameplayEffect(FString name)
{
    return NewObject<UGameplayEffect>(GetTransientPackage(), FName(*name));
}

inline FGameplayModifierInfo& AddModifier(
    UGameplayEffect* Effect, UProperty* Property,
    EGameplayModOp::Type Op,
    const FGameplayEffectModifierMagnitude& Magnitude)
{
    int32 index = Effect->Modifiers.Num();
    Effect->Modifiers.SetNum(index + 1);
    FGameplayModifierInfo& Info = Effect->Modifiers[index];
    Info.ModifierMagnitude = Magnitude;
    Info.ModifierOp = Op;
    Info.Attribute.SetUProperty(Property);
    return Info;
}
```

1.  然后，添加以下代码以实现：

```cpp
void AWarrior::TestGameplayEffect()
{
    // Construct & retrieve UProperty to affect
    UGameplayEffect* RecoverHP = ConstructGameplayEffect("RecoverHP");

    // Compile-time checked retrieval of Hp UPROPERTY()
    // from our UGameUnitAttributeSet class (listed in
    // UGameUnitAttributeSet.h)
    UProperty* hpProperty = FindFieldChecked<UProperty>(
        UGameUnitAttributeSet::StaticClass(),
        GET_MEMBER_NAME_CHECKED(UGameUnitAttributeSet, Hp));

}
```

1.  使用`AddModifier`函数更改`GameUnitAttributeSet`的`Hp`字段，如下所示：

```cpp
  // Command the addition of +5 HP to the hpProperty
  AddModifier(RecoverHP, hpProperty, EGameplayModOp::Additive, FScalableFloat(50.f));
```

1.  填写`GameplayEffect`的其他属性，包括`DurationPolicy`和`ChanceToApplyToTarget`等字段，或您希望修改的任何其他字段，如下所示：

```cpp
// .. for a fixed-duration of 10 seconds ..
RecoverHP->DurationPolicy = EGameplayEffectDurationType::HasDuration;
RecoverHP->DurationMagnitude = FScalableFloat(10.f);

// .. with 100% chance of success ..
RecoverHP->ChanceToApplyToTarget = 1.f;

// .. with recurrency (Period) of 0.5 seconds
RecoverHP->Period = 0.5f;
```

1.  将效果应用于您选择的`AbilitySystemComponent`。底层的`UAttributeSet`将受到您的调用影响并修改，如下面的代码片段所示：

```cpp
FActiveGameplayEffectHandle recoverHpEffectHandle =
    AbilitySystemComponent->ApplyGameplayEffectToTarget(
        RecoverHP, AbilitySystemComponent, 1.f);
```

# 它是如何工作的...

`GameplayEffects`是简单地影响演员`AttributeSet`变化的小对象。`GameplayEffects`可以在`Period`间隔内一次性或重复发生。您可以快速编程效果，并且`GameplayEffect`类的创建旨在内联。

# 还有更多...

一旦`GameplayEffect`激活，您将收到一个`FActiveGameplayEffectHandle`。您可以使用此句柄通过`FActiveGameplayEffectHandle`的`OnRemovedDelegate`成员附加一个函数委托，在效果结束时运行。例如，您可能调用以下代码：

```cpp
FOnActiveGameplayEffectRemoved* ep = AbilitySystemComponent->
    OnGameplayEffectRemovedDelegate(recoverHpEffectHandle);

if (ep) 
{
    ep->AddLambda([]() 
    {
        UE_LOG(LogTemp, Warning, TEXT("Recover effect has been removed."), 1);
    });
}
```

# GameplayTasks API – 使用 GameplayTasks 实现功能

`GameplayTasks`用于将一些游戏功能封装在可重用的对象中。您要使用它们，只需从`UGameplayTask`基类派生并重写一些您希望实现的成员函数。

# 准备工作

如果您尚未这样做，请完成*GameplayAbilities API – 使用游戏控制触发演员的游戏技能*配方中的步骤 1-4，以在您的`ProjectName.Build.cs`文件中链接到`GameplayTasks` API 并启用其功能。

之后，进入 UE4 编辑器，通过转到窗口 | 开发者工具 | 类查看器导航到类查看器。在过滤器下，取消选中仅演员和仅可放置选项。

确保存在`GameplayTask`对象类型：

![](img/3294cedd-2012-4aee-b372-41fee61f3347.png)

# 如何操作...

1.  点击文件 | 添加 C++类....选择从`GameplayTask`派生。为此，您必须首先选中显示所有类，然后在过滤器框中输入`gameplaytask`。点击下一步：

![](img/bcf5cdab-33de-40f8-82db-fc30adcbe46e.png)

1.  为您的 C++类命名（例如，`GameplayTask_TaskName`是惯例），然后将类添加到您的项目中。示例生成一个粒子发射器，并命名为`GameplayTask_CreateParticles`：

![](img/bbd3d5ad-3e83-46d0-abcb-5096ca481530.png)

1.  一旦你的`GameplayTask_CreateParticles.h`和`.cpp`对创建完成，导航到`.h`文件，并更新类如下：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameplayTask.h"
#include "Particles/ParticleSystem.h"
#include "GameplayTask_CreateParticles.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_11_API UGameplayTask_CreateParticles : public UGameplayTask
{
    GENERATED_BODY()

public:
 virtual void Activate();

 // A static constructor for an instance of a 
    // UGameplayTask_CreateEmitter instance,
 // including args of (what class of emitter, where to 
    // create it).
 UFUNCTION(BlueprintCallable, Category = "GameplayTasks", meta = (AdvancedDisplay = "TaskOwner", DefaultToSelf = "TaskOwner", BlueprintInternalUseOnly = "TRUE"))
 static UGameplayTask_CreateParticles* ConstructTask(
 TScriptInterface<IGameplayTaskOwnerInterface> TaskOwner,
 UParticleSystem* particleSystem,
 FVector location);

 UParticleSystem* ParticleSystem;
 FVector Location;

};
```

1.  然后，实现`GameplayTask_CreateParticles.cpp`文件，如下所示：

```cpp
#include "GameplayTask_CreateParticles.h"
#include "Kismet/GameplayStatics.h"

// Like a constructor.
UGameplayTask_CreateParticles* UGameplayTask_CreateParticles::ConstructTask(
 TScriptInterface<IGameplayTaskOwnerInterface> TaskOwner,
 UParticleSystem* particleSystem,
 FVector location)
{
 UGameplayTask_CreateParticles* task = 
    NewTask<UGameplayTask_CreateParticles>(TaskOwner);
 // Fill fields
 if (task)
 {
 task->ParticleSystem = particleSystem;
 task->Location = location;
 }
 return task;
}

void UGameplayTask_CreateParticles::Activate()
{
 Super::Activate();

 UGameplayStatics::SpawnEmitterAtLocation(GetWorld(),
    ParticleSystem, Location);
}
```

1.  打开你的`Warrior.h`文件，并将以下接口添加到类定义中：

```cpp
UCLASS()
class CHAPTER_11_API AWarrior : public ACharacter, public IAbilitySystemInterface, public IGameplayTaskOwnerInterface
{
    GENERATED_BODY()
```

1.  之后，向其中添加以下新的属性：

```cpp
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Stats)
    UGameplayTasksComponent* GameplayTasksComponent;

    // This is the particleSystem that we create with the
    // GameplayTask_CreateParticles object.
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Stats)
    UParticleSystem* particleSystem;
```

1.  在下面，为`GameplayTaskOwnerInterface`添加以下函数定义：

```cpp
    // <GameplayTaskOwnerInterface>

    virtual UGameplayTasksComponent* GetGameplayTasksComponent(const 
    UGameplayTask& Task) const { return GameplayTasksComponent; }

    // This gets called both when task starts and when task gets 
    // resumed.
    // Check Task.GetStatus() if you want to differentiate.
    virtual void OnTaskActivated(UGameplayTask& Task) { }
    virtual void OnTaskDeactivated(UGameplayTask& Task) { }

    virtual AActor* GetOwnerActor(const UGameplayTask* Task) const {
        return Task->GetOwnerActor(); // This will give us the 
    // accurate answer for the Task..
    }
    // </End GameplayTaskOwnerInterface>
```

1.  之后，转到`Warrior.cpp`文件，更新类构造函数如下：

```cpp
AWarrior::AWarrior()
{
    // Set this character to call Tick() every frame. You can 
    // turn this off to improve performance if you don't need 
    // it.
    PrimaryActorTick.bCanEverTick = true;
    AbilitySystemComponent = CreateDefaultSubobject<UAbilitySystemComponent>
    ("UAbilitySystemComponent");
    GameplayTasksComponent = CreateDefaultSubobject<UGameplayTasksComponent>
    ("UGameplayTasksComponent");
}
```

1.  保存你的脚本，返回到 Unreal 编辑器，并编译你的代码。一旦编译完成，打开你的`Actor`类派生（`MyWarrior`），然后滚动到统计部分，将粒子系统属性设置为你想看到的内容，例如，如果你在创建项目时包含了示例内容，可以选择`P_Fire`选项：

![](img/e2540cdb-18c9-4567-9772-0bff4c83ab28.png)

1.  这在完整蓝图编辑器中可用，在蓝图编辑器的组件选项卡下的组件下拉菜单中。将`GameplayTasksComponent`添加到

1.  使用以下代码在你的`Actor`派生（`MyWarrior`）实例中创建并添加一个`GameplayTask`实例：

```cpp
    UGameplayTask_CreateParticles* task =
        UGameplayTask_CreateParticles::ConstructTask(this,
        particleSystem, FVector(200.f, 0.f, 200.f));

    if (GameplayTasksComponent && task)
    {
        GameplayTasksComponent->AddTaskReadyForActivation(*task);
    }
```

此代码可以在你的`Actor`类派生中的任何位置运行，在任何`GameplayTasksComponent`初始化之后（在任何`PostInitializeComponents()`之后）。

1.  编译你的代码。将你的关卡设置为使用`MyWarrior`作为默认的 Pawn 类型，当游戏开始时，你应该会注意到粒子效果播放：

![](img/d976d754-77a1-43dd-9906-1e21151058ec.png)

# 它是如何工作的...

`GameplayTasks`简单地注册到位于你选择的`Actor`类派生中的`GameplayTasksComponent`。你可以在游戏中的任何时间激活任意数量的`GameplayTasks`以触发它们的效果。

`GameplayTasks`也可以启动`GameplayEffects`来改变`AbilitySystemsComponents`的属性，如果你希望的话。

# 还有更多...

你可以为游戏中的任何数量的事件派生`GameplayTasks`。更重要的是，你可以覆盖更多虚拟函数以钩入额外的功能。

# HTTP API – 使用网络请求下载网页

当你维护计分板或其他需要定期通过 HTTP 请求访问服务器的类似事物时，你可以使用 HTTP API 来执行此类网络请求任务。

# 准备工作

拥有一个你可以通过 HTTP 请求数据的服务器。如果你想尝试 HTTP 请求，可以使用任何类型的公共服务器。

# 如何操作...

1.  在你的`ProjectName.Build.cs`文件中链接到`HTTP` API：

```cpp
using UnrealBuildTool;

public class Chapter_11 : ModuleRules
{
    public Chapter_11(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] {
        "Core", "CoreUObject", "Engine", "InputCore" });
        PublicDependencyModuleNames.AddRange(new string[] { 
        "GameplayAbilities", "GameplayTags", "GameplayTasks" });
 PublicDependencyModuleNames.AddRange(new string[] {
        "HTTP" });

        PrivateDependencyModuleNames.AddRange(new string[] { });

        // Uncomment if you are using Slate UI
        // PrivateDependencyModuleNames.AddRange(new string[]
        // { "Slate", "SlateCore" });

        // Uncomment if you are using online features
        // PrivateDependencyModuleNames.Add("OnlineSubsystem");

        // To include OnlineSubsystemSteam, add it to the plugins section in your uproject file with the Enabled attribute set to true
    }
}
```

1.  在你将发送网络请求的文件中（在我的例子中，我将使用`Chapter_11GameModeBase`类），将以下代码片段的新增内容包含进去：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "Runtime/Online/HTTP/Public/HttpManager.h" 
#include "Runtime/Online/HTTP/Public/HttpModule.h" 
#include "Runtime/Online/HTTP/Public/HttpRetrySystem.h" 
#include "Runtime/Online/HTTP/Public/Interfaces/IHttpResponse.h"
using namespace FHttpRetrySystem;
#include "Chapter_11GameModeBase.generated.h"
```

1.  使用以下代码从`FHttpModule`构造一个`IHttpRequest`对象：

```cpp
TSharedRef<IHttpRequest> 
 http=FHttpModule::Get().CreateRequest();
```

`FHttpModule` 是一个单例对象。整个程序中只有一个它的副本，你应该用它来与 `FHttpModule` 类进行所有交互。

1.  将你的函数附加到 `IHttpRequest` 对象的 `FHttpRequestCompleteDelegate`，该委托具有以下签名：

```cpp
void HttpRequestComplete( FHttpRequestPtr request,
 FHttpResponsePtr response, bool success );
```

1.  委托位于 `IHttpRequest` 对象中的 `http->OnProcessRequestComplete()` 内：

```cpp
FHttpRequestCompleteDelegate& delegate = http-
 >OnProcessRequestComplete();
```

有几种方法可以将回调函数附加到委托中。你可以使用以下方法：

+   使用 `delegate.BindLambda()` 的 lambda 表达式：

```cpp
delegate.BindLambda( 
  // Anonymous, inlined code function (aka lambda) 
  []( FHttpRequestPtr request, FHttpResponsePtr response, bool 
   success ) -> void 
{ 
  UE_LOG( LogTemp, Warning, TEXT( "Http Response: %d, %s" ), 
  request->GetResponse()->GetResponseCode(), 
  *request->GetResponse()->GetContentAsString() ); 
}); 
```

1.  指定你想要访问的网站的 URL：

```cpp
http->SetURL( TEXT( "http://unrealengine.com" ) ); 
```

1.  通过调用 `ProcessRequest` 处理请求：

```cpp
http->ProcessRequest();
```

然后，当你运行此代码时，你应该会注意到你指向的 URL 的内容被显示出来：

![](img/b5de988c-e6c7-4509-a9af-9ff2c3a41881.png)

显示 unrealengine.com 的 HTML 内容

当然，在这个例子中，它是一个网页，但你可以轻松地将它指向 CSV 文件、文本文档或任何其他内容，以获取你项目所需的信息！

# 它是如何工作的...

HTTP 对象是发送 HTTP 请求到服务器并获取 HTTP 响应所需的所有内容。你可以使用 HTTP 请求/响应来做任何你想做的事情；例如，将分数提交到高分榜或从服务器检索用于游戏中显示的文本。

它们配备了要访问的 URL 和请求完成时要运行的函数回调。最后，它们通过 `FManager` 发送出去。当 web 服务器响应时，你的回调被调用，并且可以显示你的 HTTP 响应的结果。

# 还有更多...

你还可以通过其他方式将回调函数附加到委托中：

+   使用任何 UObject 的成员函数：

```cpp
delegate.BindUObject(this, &AChapter_11GameModeBase::HttpRequestComplete);
```

你不能直接附加到 `UFunction`，因为 `.BindUFunction()` 命令请求所有都是 `UCLASS`、`USTRUCT` 或 `UENUM` 的参数。

+   对于任何普通的 C++ 对象的成员函数，使用 `.BindRaw`:

```cpp
PlainObject* plainObject = new PlainObject();
delegate.BindRaw( plainObject, &PlainObject::httpHandler );
 // plainObject cannot be DELETED Until httpHandler gets
 // called..
```

你必须确保你的 `plainObject` 在 HTTP 请求完成时指向内存中的有效对象。这意味着你不能在 `plainObject` 上使用 `TAutoPtr`，因为这将导致在声明它的代码块结束时释放 `plainObject`，但这可能发生在 HTTP 请求完成之前。

+   使用全局 C 风格的静态函数：

```cpp
// C-style function for handling the HTTP response: 
void httpHandler( FHttpRequestPtr request,  
FHttpResponsePtr response, bool success ) 
{ 
  Info( "static: Http req handled" ); 
} 
delegate.BindStatic( &httpHandler ); 
```

当使用与对象相关的委托回调时，确保你调用的对象实例至少在 `HttpResponse` 从服务器返回之前仍然存在。处理 `HttpRequest` 需要实际运行时间。毕竟，它是一个网络请求——想象一下等待网页加载。

你必须确保在调用回调函数的对象实例在初始调用和 `HttpHandler` 函数调用之间没有被释放。当 HTTP 请求完成后，回调返回时，对象必须仍然在内存中。

您不能简单地期望在附加回调函数并调用 `ProcessRequest()` 之后立即发生 `HttpResponse` 函数！使用引用计数的 `UObject` 实例来附加 `HttpHandler` 成员函数是一个好主意，以确保对象在 HTTP 请求完成之前保持在内存中。

您可以在本章示例代码中的 `Chapter_11GameModeBase.cpp` 文件中看到所有七种可能的用法示例。

您可以通过以下成员函数设置额外的 HTTP 请求参数：

+   `SetVerb()`，用于更改您在 HTTP 请求中是否使用 `GET` 或 `POST` 方法

+   `SetHeaders()`，用于修改您希望设置的任何通用头部设置

# HTTP API – 显示下载进度

来自 HTTP API 的 `IHttpRequest` 对象将通过 `FHttpRequestProgressDelegate` 上的回调报告 HTTP 下载进度，该委托可以通过 `OnRequestProgress()` 访问。我们可以附加到 `OnRequestProgress()` 委托的函数签名如下：

```cpp
HandleRequestProgress( FHttpRequestPtr request, int32 
 sentBytes, int32 receivedBytes )
```

您可能需要编写的函数的三个参数包括原始的 `IHttpRequest` 对象、已发送的字节数和到目前为止已接收的字节数。此函数会在 `IHttpRequest` 对象完成时定期回调，即当调用 `OnProcessRequestComplete()` 时。您可以使用传递给您的 `HandleRequestProgress` 函数的值来找出您的进度。

# 准备工作

您将需要一个互联网连接来使用此食谱。我们将从公共服务器请求一个文件。如果您愿意，可以使用公共服务器或您自己的私有服务器进行 HTTP 请求。

在本食谱中，我们将绑定一个回调函数到 `OnRequestProgress()` 委托，以显示从服务器下载文件的下载进度。请准备好一个项目，在其中我们可以编写一个执行 `IHttpRequest` 的代码片段，以及一个用于显示百分比进度的良好界面。

# 如何操作...

1.  在您的 `ProjectName.Build.cs` 文件中链接到 `HTTP` API。

1.  使用以下代码构建 `IHttpRequest` 对象：

```cpp
TSharedRef<IHttpRequest> http = 
 HttpModule::Get().CreateRequest();
```

1.  提供一个回调函数，在请求进度时调用，以更新我们的用户：

```cpp
http->OnRequestProgress().BindLambda(
    this
    -> void
    {
        int32 contentLen =
        request->GetResponse()->GetContentLength();
        float percentComplete = 100.f * receivedBytes /
        contentLen;

        UE_LOG(LogTemp, Warning, TEXT("Progress sent=%d bytes /
        received=%d/%d bytes [%.0f%%]"), sentBytes, receivedBytes,
        contentLen, percentComplete);

    });
```

1.  使用 `http->ProcessRequest()` 处理您的请求。

# 它是如何工作的...

`OnRequestProgress()` 回调会定期触发，带有已发送的字节数和已接收的字节数 HTTP 进度。我们将通过计算 `(float)receivedBytes/totalLen` 来计算已完成的下载总百分比，其中 `totalLen` 是 HTTP 响应的总字节数。通过附加到 `OnRequestProgress()` 委托回调的 lambda 函数，我们可以通过文本显示信息。

在上一节 *如何操作...* 的代码基础上，可以创建一个用于进度条的 UMG 小部件，并调用 `.SetPercent()` 成员函数来反映下载进度。
