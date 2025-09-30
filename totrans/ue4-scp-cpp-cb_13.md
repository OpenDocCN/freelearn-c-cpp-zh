# NPC 控制 AI

在本章中，我们将介绍以下食谱：

+   实现简单的跟随行为

+   放置导航网格

+   创建黑板

+   创建行为树

+   将行为树连接到角色

+   创建 BTService

+   创建 BTTask

# 简介

AI 包括游戏 NPC 的许多方面，以及玩家行为。AI 的一般主题包括路径查找和 NPC 行为。通常，我们将 NPC 在游戏内一段时间内所做的事情称为行为。

UE4 中的 AI 得到了很好的支持。存在许多结构，允许在编辑器内进行基本的 AI 编程，但我们将专注于使用 C++ 编程元素，并在需要时涉及引擎方面。

为了使可视化我们的 AI 角色和与玩家的交互更容易，在本章中，我将使用 C++ 第三人称模板：

![图片](img/0293a11c-8c73-4598-b464-097efdc453a5.png)

尽管我很乐意涵盖 Unreal Engine 4 中与 AI 一起工作的所有方面，但这可能需要一本整本书。如果你在阅读本章后对探索 AI 更加感兴趣，我建议你查看*Unreal Engine 4 AI 编程基础*，该书籍也由 Packt Publishing 提供。

# 技术要求

本章需要使用 Unreal Engine 4 和 Visual Studio 2017 作为 IDE。有关如何安装这两款软件及其要求的信息，请参阅第一章，*UE4 开发工具*。

# 实现简单的跟随行为

实现任何类型 AI 的最简单方法就是手动编写。这允许你快速启动并运行，但缺乏使用 Unreal 内置系统给我们带来的优雅和技巧。这个食谱提供了一个超级简单的实现，使一个对象跟随另一个对象。

# 准备工作

准备一个 UE4 项目，其中包含一个简单的景观或地面上的一组几何形状，理想情况下在几何形状中有一个*死胡同*来测试 AI 移动功能。随 C++ 第三人称模板一起提供的`ThirdPersonExampleMap`应该可以正常工作。

# 如何操作...

1.  通过转到“添加新”|“新建 C++ 类”，创建一个新的 C++ 类，该类从`Character`派生。在“添加 C++ 类”菜单下，选择`Character`并点击“下一步”按钮：

![图片](img/39d55e09-2d74-4208-a316-e58723a542d9.png)

1.  在下一屏幕上，将类命名为`FollowingCharacter`并点击“创建类”按钮：

![图片](img/597248e0-ee12-459f-9588-84457a32324a.png)

1.  在`FollowingCharacter.cpp`文件中，将`Tick`函数更新为以下内容：

```cpp
void AFollowingCharacter::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

 // Get current location
 auto startPos = GetActorLocation();

 // Get player's location
 FVector playerPos = GetWorld()->GetFirstPlayerController()->GetPawn()->GetActorLocation();

 // Get the direction to move in
 FVector direction = playerPos - startPos;
 direction.Normalize();

 // Move the player in that direction
 SetActorLocation(startPos + direction);
}
```

如果编译器可以从分配给对象的赋值中推断出对象类型，则可以使用`auto`关键字进行变量声明。

1.  保存你的脚本并编译你的代码。

1.  将跟随角色拖放到你的场景中。目前没有角色的可视化，所以请选择该对象。然后，从详细信息选项卡中，点击添加组件按钮。从那里，选择圆柱形状：

![](img/c8d54ec0-bd6d-4269-8951-2fcc86cf0489.png)

如果一切顺利，你应该能在屏幕上看到对象。

![](img/eaec7727-a31c-429c-a0c9-d53b6b24b012.png)

添加了圆柱形形状的以下角色

1.  运行游戏并四处移动。你应该注意到，无论玩家走到哪里，圆柱体都会跟随玩家！

# 它是如何工作的...

在这个例子中，我们实际上是通过自己进行简单的向量运算来*硬编码*这个敌人跟随玩家角色。虽然技术上可行，但它没有利用 Unreal 内置的 AI 功能。如果没有实际进行路径查找，它会在墙壁处停止 AI，如果你让 AI 赶上玩家，它还会破坏玩家角色。由于碰撞，玩家将无法再移动。

本章的其余部分将使用 Unreal 的实际内置系统，这将创建一个更健壮的实现。

# 放置导航网格

导航网格（也称为**导航网格**）基本上是 AI 控制单位认为可通行的区域（即，“AI 控制”单位被允许进入或穿越的区域）的定义。导航网格不包括如果玩家试图穿过它将会阻挡玩家的几何形状。

# 准备工作

在 UE4 中根据场景的几何形状构建导航网格相当简单。从一个周围有一些障碍物或使用地形的现有项目开始。与 C++第三人称模板一起提供的`ThirdPersonExampleMap`非常适合这个目的。

# 如何做到这一点...

要构建你的导航网格，只需执行以下步骤：

1.  前往 模式 | 体积。

1.  将 导航网格边界体积 选项拖放到你的视图中。

1.  使用缩放工具增加导航网格的大小，使其覆盖使用导航网格的演员应该被允许导航和路径查找的区域。要切换完成后的导航网格的可见性，按*P*键：

![](img/9b025c53-495a-412d-91e5-ae9c9e6b5b92.png)

在导航网格边界体积范围内的导航网格

# 它是如何工作的...

导航网格不会阻止玩家 pawn（或其他实体）踩在某个几何形状上，但它用于指导 AI 控制实体他们可以和不可以去的地方。

想了解更多关于在 UE4 中缩放对象的信息，请查看以下链接：[`docs.unrealengine.com/en-us/Engine/Actors/Transform`](https://docs.unrealengine.com/en-us/Engine/Actors/Transform).

# 创建黑板

**黑板**是用于与行为树一起使用的变量的容器。这些数据用于决策目的，无论是单个 AI 还是一组 AI。我们在这里将创建一个黑板，然后在未来的菜谱中使用它。

# 如何做到这一点...

1.  在`内容`文件夹下的内容浏览器中，选择添加新项 | 人工智能 | 黑板：

![图片](img/ab959cad-1fb0-4c9f-9d53-5425a5ed955c.png)

1.  当被要求提供名称时，提供`EnemyBlackboard`。双击文件以打开黑板编辑器。

1.  在“黑板”选项卡中，点击新建键 | 对象：

![图片](img/bf12d03b-7d23-4239-bdb8-8f1bdc718b8a.png)

1.  当被要求提供对象的名称时，输入`Target`。然后，通过点击名称左侧的箭头打开键类型属性，并将基本类属性设置为`Actor`：

![图片](img/bf4829a3-3d30-4687-a757-43229a9bed7c.png)

1.  添加任何其他你希望访问的属性，然后点击“保存”按钮。

# 它是如何工作的...

在这个菜谱中，我们创建了一个黑板，稍后我们将在代码中使用它来设置和获取我们在行为树中将使用的玩家的值。

# 创建行为树

如果黑板是 AI 的共享内存，那么行为树就是 AI 的处理器，其中将包含 AI 的逻辑。它做出决策，然后根据这些决策采取行动，使 AI 在游戏运行时实际上能够做些什么。在这个菜谱中，我们将创建一个行为树并将其分配给黑板。

# 如何做到这一点...

1.  在`内容`文件夹下的`内容浏览器`中，选择添加新项 | 人工智能 | 行为树：

![图片](img/144f462c-a25f-4e5f-9243-507ee517881a.png)

1.  当被要求提供名称时，提供`EnemyBehaviorTree`。双击文件以打开行为树编辑器。

1.  打开后，在“详细信息”选项卡下，打开 AI | 行为树部分，并验证黑板资产属性是否设置为`EnemyBlackboard`。你应该会注意到我们创建的 Target 属性在“键”下列出。如果没有，请关闭编辑器并重新打开：

![图片](img/40020a2c-e30b-4f4b-b880-8c227ee889d0.png)

行为树编辑器的视图

1.  完成后，点击“保存”按钮。

# 它是如何工作的...

在这个菜谱中，我们创建了一个行为树，这是 AI 系统所必需的，以便它能够完成任务和其他各种功能。在未来的菜谱中，我们将使用它来创建我们自己的自定义角色类。

# 将行为树连接到角色

一个`行为树`在任意时刻选择一个由 AI 控制的单位所表现的行为。行为树相对简单构建，但要使其运行，需要进行大量的设置。你还需要熟悉用于构建你的**行为树**的组件，以便有效地进行操作。

行为树对于定义比简单地朝向对手移动（如前一个菜谱中的`AIMoveTo`所示）更丰富的 NPC 行为非常有用。

# 准备工作

在开始此菜谱之前，请确保你已经完成了以下菜谱：

+   *铺设导航网格*

+   *创建黑板*

+   *创建行为树*

# 如何做到这一点...

1.  打开你的`.Build.cs`文件（在我们的例子中，`Chapter_13.Build.cs`）并添加以下依赖项：

```cpp
using UnrealBuildTool;

public class Chapter_13 : ModuleRules
{
  public Chapter_13(ReadOnlyTargetRules Target) : base(Target)
  {
    PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

    PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "HeadMountedDisplay" });
        PublicDependencyModuleNames.AddRange(new string[] { "AIModule", "GameplayTasks" });

    }
}
```

1.  编译你的代码。

1.  在内容浏览器中，选择添加新 | 新 C++类。在添加 C++类菜单中，检查显示所有类选项，输入`AIController`，然后选择`AIController`类。然后，点击下一步：

![图片](img/4db220e7-1fa7-48a1-9f27-d230b88354d4.png)

1.  当被要求为类命名时，命名为`EnemyAIController`并点击创建类按钮。

1.  打开 Visual Studio 并更新`EnemyAIController.h`文件如下：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "AIController.h"
#include "BehaviorTree/BehaviorTreeComponent.h"
#include "BehaviorTree/BlackboardComponent.h"
#include "EnemyAIController.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_13_API AEnemyAIController : public AAIController
{
    GENERATED_BODY()

private:
 // AI Component references
 UBehaviorTreeComponent* BehaviorComp;
 UBlackboardComponent* BlackboardComp;

public:
 AEnemyAIController();

 // Called when the controller possesses a Pawn/Character
 virtual void Possess(APawn* InPawn) override;

    FBlackboard::FKey TargetKeyID;

};

```

1.  在创建函数声明后，我们需要在`EnemyAIController.cpp`文件中定义它们：

```cpp
#include "EnemyAIController.h"

AEnemyAIController::AEnemyAIController()
{
    //Initialize components
    BehaviorComp = CreateDefaultSubobject<UBehaviorTreeComponent>(TEXT("BehaviorComp"));
    BlackboardComp = CreateDefaultSubobject<UBlackboardComponent>(TEXT("BlackboardComp"));
}

// Called when the controller possesses a Pawn/Character
void AEnemyAIController::Possess(APawn* InPawn)
{
    Super::Possess(InPawn);
}
```

除了 AI 控制器，我们还需要有一个角色。

1.  通过转到添加新 | 新 C++类创建一个新的 C++类，该类从`Character`派生。在添加 C++类菜单下，选择`Character`并点击下一步按钮：

![图片](img/9f10b90b-8e30-4d27-9752-a97c1c011ad5.png)

1.  在下一个屏幕上，将类命名为`EnemyCharacter`并点击创建类按钮。

1.  打开 Visual Studio。在`EnemyCharacter.h`文件下，添加以下属性：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "EnemyCharacter.generated.h"

UCLASS()
class CHAPTER_13_API AEnemyCharacter : public ACharacter
{
    GENERATED_BODY()

public:
    // Sets default values for this character's properties
    AEnemyCharacter();

 UPROPERTY(EditAnywhere, Category = Behavior)
 class UBehaviorTree *EnemyBehaviorTree;

protected:
    // Called when the game starts or when spawned
    virtual void BeginPlay() override;

public: 
    // Called every frame
    virtual void Tick(float DeltaTime) override;

    // Called to bind functionality to input
    virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

};
```

1.  然后，我们可以回到`EnemyAIController.cpp`文件并更新`Possess`函数，因为我们的角色类已经存在：

```cpp
#include "EnemyAIController.h"
#include "EnemyCharacter.h"
#include "BehaviorTree/BehaviorTree.h"

AEnemyAIController::AEnemyAIController()
{
    // Initialize components
    BehaviorComp = CreateDefaultSubobject<UBehaviorTreeComponent>(TEXT("BehaviorComp"));
    BlackboardComp = CreateDefaultSubobject<UBlackboardComponent>(TEXT("BlackboardComp"));
}

// Called when the controller possesses a Pawn/Character
void AEnemyAIController::Possess(APawn* InPawn)
{
    Super::Possess(InPawn);

 // Convert InPawn to EnemyCharacter
 auto Character = Cast<AEnemyCharacter>(InPawn);

 // Check if pointers are valid
 if(Character && Character->EnemyBehaviorTree)
 {
 BlackboardComp->InitializeBlackboard(*Character->EnemyBehaviorTree->BlackboardAsset);

 TargetKeyID = BlackboardComp->GetKeyID("Target");

 BehaviorComp->StartTree(*Character->EnemyBehaviorTree);
 }
}
```

1.  保存你的脚本并编译你的代码。

现在，我们将创建我们刚刚创建的两个类的蓝图版本，并分配我们的变量。

1.  在`C++ Classes/Chapter_13`文件夹下的内容浏览器中，右键单击`EnemyAIController`对象，选择基于`EnemyAIController`创建蓝图类选项。给它一个名字并点击创建蓝图类按钮。

1.  同样，对`EnemyCharacter`对象做同样的事情。

1.  双击你的`MyEnemyCharacter`蓝图，在详细信息选项卡下，将敌对行为树属性设置为`EnemyBehaviorTree`。然后，将 AI 控制器类属性设置为`MyEnemyAIController`：

![图片](img/553deaf5-094e-49be-850d-047a325d2958.png)

分配敌对行为树和 AI 控制器类属性

1.  你可能还想为角色添加一个视觉组件，因此从组件选项卡，点击添加组件按钮并选择立方体。之后，修改缩放为（`0.5, 0.5, 1.5`）。

正如我们之前讨论的，你可能需要点击打开完整蓝图编辑器文本以查看所有可用选项。

16. 然后，编译并保存所有你的资产：

![图片](img/ea359363-aa31-4d0b-b7d0-c8d67aab84e7.png)

完成的敌对角色

有了这个，我们就建立了一个 AI 角色、AI 控制器和行为树之间的连接！

# 它是如何工作的...

我们创建的 AI 控制器类将添加我们在前两个菜谱中创建的行为树和黑板。

行为树连接到 AI 控制器，AI 控制器反过来连接到角色。我们将通过在图中输入任务和服务节点来通过行为树控制`Character`的行为。

行为树托管了六种不同类型的节点，如下所示：

1.  **任务**：任务节点是行为树中的紫色节点，包含要运行的蓝图代码。这是由 AI 控制的单位必须执行的操作（代码方面）。任务必须返回 `true` 或 `false`，具体取决于任务是否成功（通过在末尾提供 `FinishExecution()` 节点）。

1.  **装饰器**：装饰器只是节点执行的布尔条件。它检查一个条件，通常用于选择器或序列块内。

1.  **服务**：当它计时时会运行一些蓝图代码。这些节点的计时间隔是可调整的（例如，它可以比每帧计时慢，比如每 10 秒一次）。你可以使用这些来查询场景更新、新的对手等。黑板可以用来存储查询信息。服务节点在其末尾没有 `FinishExecute()` 调用。

1.  **选择器**：它从左到右运行所有子树，直到遇到成功。当遇到成功时，执行会回溯到树中。

1.  **序列**：它从左到右运行所有子树，直到遇到失败。当遇到失败时，执行会回溯到树中。

1.  **简单并行**：这将在并行子树（灰色）与单个任务（紫色）一起运行。

# 创建 BTService

服务附加到行为树中的节点，并将以它们定义的频率执行；也就是说，只要它们的分支正在执行。类似于其他行为树系统中的并行节点，这些通常用于进行检查和更新黑板，我们将在此配方中使用它来找到我们的玩家对象并将其分配给我们的黑板。

# 准备工作...

完成之前的配方，*将行为树连接到角色*。

# 如何操作...

1.  从内容浏览器中选择添加新内容 | 新 C++ 类。从选择父类菜单中，勾选显示所有类选项，并查找 `BTService` 类。选择它然后点击下一步按钮：

![图片](img/0a41289d-bab4-4f68-a540-4f354a5e08a8.png)

1.  在下一个菜单中，将其名称设置为 `BTService_FindPlayer`，然后点击创建类选项。

1.  从 `BTService_FindPlayer.h` 文件中，使用以下代码：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/BTService.h"
#include "BehaviorTree/BehaviorTreeComponent.h"
#include "BTService_FindPlayer.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_13_API UBTService_FindPlayer : public UBTService
{
    GENERATED_BODY()

public:
 UBTService_FindPlayer();

 /** update next tick interval
 * this function should be considered as const (don't modify state of object) if node is not instanced! */
 virtual void TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds) override;

};
```

1.  从 `BTService_FindPlayer.cpp` 文件中，使用以下代码：

```cpp
#include "BTService_FindPlayer.h"
#include "EnemyAIController.h"
#include "BehaviorTree/Blackboard/BlackboardKeyType_Object.h"

UBTService_FindPlayer::UBTService_FindPlayer()
{
    bCreateNodeInstance = true;
}

void UBTService_FindPlayer::TickNode(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory, float DeltaSeconds)
{
    Super::TickNode(OwnerComp, NodeMemory, DeltaSeconds);

    auto EnemyAIController = Cast<AEnemyAIController>(OwnerComp.GetAIOwner());

    if(EnemyAIController)
    {
        auto PlayerPawn = GetWorld()->GetFirstPlayerController()->GetPawn();
        OwnerComp.GetBlackboardComponent()->SetValue<UBlackboardKeyType_Object>(EnemyAIController->TargetKeyID, PlayerPawn);
        UE_LOG(LogTemp, Warning, TEXT("Target has been set!"));

    }

}
```

1.  保存你的脚本并编译它们。

1.  在内容浏览器中，转到 `Content` 文件夹，其中包含我们之前创建的 `EnemyBehaviorTree`，双击它以打开行为树编辑器。

1.  从那里，从 ROOT 处拖动一条线并选择选择器：

![图片](img/02b17ca7-1f63-459e-812c-f631ef24d0e5.png)

重要的是要注意，你需要从底部的深灰色矩形处拖动。如果你尝试从 ROOT 的中间拖动，你只会移动节点。

1.  右键单击选择器节点并选择添加服务 | 查找玩家：

![图片](img/36058d95-940d-494f-bcd3-9f1d70ff5617.png)

1.  现在，将你的 `MyEnemyCharacter` 对象实例拖放到场景中并运行游戏：

![图片](img/538167bc-799d-4532-a475-eaa0eee25c53.png)

如您所见，值已经被设置！

# 它是如何工作的...

由于没有其他节点可以转换到，我们的行为树将继续调用选择器。

# 创建一个 BTTask

除了服务之外，我们还有任务，它们是行为树的叶节点。这些是实际执行动作的东西。在我们的例子中，我们将让我们的 AI 跟随我们的目标，即玩家。

# 准备中...

完成之前的配方，*创建一个 BTService*。

# 如何做到这一点...

1.  从内容浏览器中选择“添加新内容”|“新 C++类”。在“选择父类”菜单中，勾选“显示所有类”选项，并查找`BTTask_BlackboardBase`类。选择它然后点击“下一步”按钮：

![图片](img/91d4a05e-cee6-4e79-bdc9-18e76094d6d9.png)

1.  在下一个菜单中，将其名称设置为`BTTask_MoveToPlayer`，然后点击“创建类”选项：

![图片](img/00eb28d3-9f0c-4946-89f3-a726f8b2f435.png)

1.  打开 Visual Studio，并将以下函数添加到`BTTask_MoveToPlayer.h`中：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "BehaviorTree/Tasks/BTTask_BlackboardBase.h"
#include "BTTask_MoveToPlayer.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_13_API UBTTask_MoveToPlayer : public UBTTask_BlackboardBase
{
    GENERATED_BODY()

public:
 /** starts this task, should return Succeeded, Failed or InProgress
 * (use FinishLatentTask() when returning InProgress)
 * this function should be considered as const (don't modify state of object) if node is not instanced! */
 virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override;

};
```

1.  然后，打开`BTTask_MoveToPlayer.cpp`文件，并将其更新为以下内容：

```cpp
#include "BTTask_MoveToPlayer.h"
#include "EnemyAIController.h"
#include "GameFramework/Character.h"
#include "BehaviorTree/Blackboard/BlackboardKeyType_Object.h"

EBTNodeResult::Type UBTTask_MoveToPlayer::ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory)
{
 auto EnemyController = Cast<AEnemyAIController>(OwnerComp.GetAIOwner());
 auto Blackboard = OwnerComp.GetBlackboardComponent();

 ACharacter * Target = Cast<ACharacter>(Blackboard->GetValue<UBlackboardKeyType_Object>(EnemyController->TargetKeyID));

 if(Target)
 {
 EnemyController->MoveToActor(Target, 50.0f);
 return EBTNodeResult::Succeeded;
 }

 return EBTNodeResult::Failed;
}
```

1.  保存您的文件并返回到 Unreal 编辑器。编译您的代码。

1.  在内容浏览器中，转到包含我们之前创建的`EnemyBehaviorTree`的`Content`文件夹，并双击它以打开行为树编辑器。

1.  将此操作拖到选择节点下方，并选择任务|移动到玩家：

![图片](img/571d92a5-1006-40e6-b2a4-64a3d13b72fc.png)

1.  保存行为树并返回到 Unreal 编辑器。如果您还没有这样做，将`MyEnemyCharacter`对象拖放到场景中并玩游戏：

![图片](img/dcddb9f0-f8a7-4b70-b9c9-2088fd700a86.png)

如您所见，我们的敌人现在正在跟随我们的玩家，只要导航网格覆盖该区域，这种情况就会发生！

# 它是如何工作的...

此配方将我们迄今为止涵盖的所有材料全部编译在一起。只要行为树处于此状态内部，就会调用`ExecuteTask`方法。此函数要求我们返回一个`EBTNodeResult`，它应该返回`Succeeded`、`Failed`或`InProgress`，以便让行为树知道我们是否可以更改状态。

在我们的例子中，我们首先获取`EnemyController`和`Target`对象，以便我们可以确定我们想要移动谁以及我们想要移动到哪里。只要这些属性有效，我们就可以调用`MoveToActor`函数。

`MoveToActor`函数提供了许多其他属性，这些属性可能很有用，以便您可以自定义您的移动。更多信息，请查看以下链接：[`api.unrealengine.com/INT/API/Runtime/AIModule/AAIController/MoveToActor/index.html`](https://api.unrealengine.com/INT/API/Runtime/AIModule/AAIController/MoveToActor/index.html)。

对于那些对在 UE4 中探索更多 AI 概念的感兴趣的人，我强烈建议查看 Orfeas Eleftheriou 的 UE4 AI 教程：[`orfeasel.com/category/ue_tuts/ai-programming/`](https://orfeasel.com/category/ue_tuts/ai-programming/)。
