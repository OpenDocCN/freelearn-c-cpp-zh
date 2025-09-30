# 类和接口之间的通信：第二部分

本章将涵盖以下菜谱：

+   从本地基类将 UInterface 方法暴露给蓝图

+   在蓝图中实现 UInterface 函数

+   创建 C++ UInterface 函数实现，这些实现可以在蓝图中被覆盖

+   从 C++ 调用蓝图定义的接口函数

# 简介

本章将向您展示如何通过蓝图使用您的 C++ UInterfaces。这可以帮助设计师访问您编写的代码，而无需他们深入研究项目的 C++ 代码。

# 技术要求

本章需要使用 Unreal Engine 4，并使用 Visual Studio 2017 作为 IDE。有关如何安装这两款软件及其要求的信息，请参阅 第一章，*UE4 开发工具*。

# 从本地基类将 UInterface 方法暴露给蓝图

能够在 C++ 中定义 `UInterface` 方法很棒，但它们也应该可以从蓝图访问。否则，使用蓝图的设计师或其他人员将无法与您的 `UInterface` 交互。这个菜谱向您展示了如何在蓝图系统中调用接口中的函数。

# 如何做到这一点...

1.  创建一个名为 `PostBeginPlay` 的 `UInterface`：

![图片](img/7539c4d4-019e-4c79-87e9-1b34d814759e.png)

1.  在 Visual Studio 中打开 `PostBeginPlay.h` 并更新 `UPostBeginPlay` 的 `UINTERFACE`，然后在 `IPostBeginPlay` 中添加以下 `virtual` 方法：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "PostBeginPlay.generated.h"

UINTERFACE(meta = (CannotImplementInterfaceInBlueprint))
class UPostBeginPlay : public UInterface
{
    GENERATED_BODY()
};

/**
 * 
 */
class CHAPTER_08_API IPostBeginPlay
{
    GENERATED_BODY()

    // Add interface functions to this class. This is the
    // class that will be inherited to implement
    // this interface.
public:
 UFUNCTION(BlueprintCallable, Category = Test)
 virtual void OnPostBeginPlay();
};
```

1.  提供函数的实现：

```cpp
#include "PostBeginPlay.h"

// Add default functionality here for any IPostBeginPlay 
// functions that are not pure virtual.
void IPostBeginPlay::OnPostBeginPlay()
{
 GEngine->AddOnScreenDebugMessage(-1, 1, FColor::Red, "PostBeginPlay called");
}
```

1.  创建一个新的名为 `APostBeginPlayTest` 的 `Actor` 类：

![图片](img/c6f5e0ca-46ef-4aa6-8dfc-742573723f0a.png)

1.  修改类声明，使其也继承 `IPostBeginPlay`：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "PostBeginPlay.h"
#include "PostBeginPlayTest.generated.h"

UCLASS()
class CHAPTER_08_API APostBeginPlayTest : public AActor, public IPostBeginPlay
```

1.  编译您的项目。在编辑器中，将 `APostBeginPlayTest` 的实例拖放到您的关卡中：

![图片](img/b3bef66c-220a-4c6a-aafe-cb29f2950e87.png)

1.  在世界大纲中选中实例后，点击蓝 prints | 打开关卡蓝图：

![图片](img/ad526052-f410-46fe-b6e9-99a44dc869d4.jpg)

1.  在关卡蓝图中，右键单击并创建对 PostBeginPlayTest1 的引用：

![图片](img/5f52943e-720b-45de-b7ce-a9aa47bb3a46.png)

注意，您还可以使用我们在上一章的 *从彼此继承 UInterfaces* 菜单中讨论的拖放方法。

1.  从您演员引用的右侧蓝色推针拖动，然后在上下文菜单中搜索 `onpost` 以查看您的新接口函数。点击它以从蓝图插入对本地 `UInterface` 实现的调用：

![图片](img/0a4d5096-348e-4d01-9996-1c1b4738061a.png)

1.  最后，将 `BeginPlay` 节点的执行引脚（白色箭头）连接到 `OnPostBeginPlay` 的执行引脚：

![图片](img/cd60e61c-cce6-4569-9985-af0696055fac.png)

1.  当你播放你的关卡时，你应该在屏幕上看到消息“PostBeginPlay called”短暂可见，这证实蓝图已成功访问并通过你的原生代码实现的`UInterface`调用。

# 它是如何工作的...

`UINTERFACE`/`IInterface`对函数就像在其他菜谱中做的那样，其中`UInterface`包含反射信息和其它数据，而`IInterface`则作为可以继承的实际接口类。

允许`IInterface`内部的函数暴露给蓝图的最重要元素是`UFUNCTION`指定符。`BlueprintCallable`标记此函数为可以从蓝图系统中调用的函数。

任何以任何方式暴露给蓝图的功能都需要一个`Category`值。此`Category`值指定了函数将在上下文菜单中列出的标题下。

函数还必须标记为`virtual`——这是为了让通过原生代码实现接口的类可以覆盖其内部的函数实现。如果没有`virtual`指定符，Unreal 头文件工具会给你一个错误，表明你必须将`virtual`或`BlueprintImplementableEvent`作为`UFUNCTION`指定符添加。

原因是如果没有其中任何一个，接口函数在 C++（由于缺少`virtual`）或蓝图（因为缺少`BlueprintImplementableEvent`）中都不会是可覆盖的。一个不能被覆盖而只能被继承的接口功能有限，因此 Epic 选择不在 UInterfaces 中支持它。

我们随后提供了一个`OnPostBeginPlay`函数的默认实现，该实现使用`GEngine`指针显示一个调试消息，确认函数已被调用。

# 参见

+   参考第八章，*类与接口之间的通信：第二部分*，了解一些菜谱，这些菜谱展示了你如何将你的 C++类与蓝图集成。

# 在蓝图中实现 UInterface 函数

UInterface 在 Unreal 中的一个关键优势是用户可以在编辑器中实现`UInterface`函数。这意味着接口可以完全在蓝图中进行实现，而无需任何 C++代码，这对设计师来说很有帮助。

# 如何做到这一点...

1.  创建一个新的`UInterface`，命名为`AttackAvoider`：

![图片](img/91241cc3-1b98-40c3-8d23-054321e24a96.png)

1.  将以下函数声明添加到头文件中：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "AttackAvoider.generated.h"

// This class does not need to be modified.
UINTERFACE(MinimalAPI)
class UAttackAvoider : public UInterface
{
  GENERATED_BODY()
};

class CHAPTER_08_API IAttackAvoider
{
  GENERATED_BODY()

  // Add interface functions to this class. This is the class
  // that will be inherited to implement this interface.
public:
    UFUNCTION(BlueprintImplementableEvent, BlueprintCallable, 
 Category = AttackAvoider)
 void AttackIncoming(AActor* AttackActor);
};
```

1.  编译你的项目。从内容浏览器中打开内容文件夹，然后在编辑器中选择添加新项目 | 蓝图类来创建一个新的蓝图类：

![图片](img/9cf44734-02d1-4ebb-87e0-f1aa7cbc45f5.jpg)

1.  以`Actor`为基础创建类：

![图片](img/be3e694c-bcd6-4215-b045-1c160d2f7596.png)

1.  将蓝图命名为`AvoiderBlueprint`，然后双击它以打开蓝图编辑器。从那里，打开类设置：

![图片](img/7b83d5a4-c153-433a-826b-a72e0d29a0b2.png)

1.  在详细信息标签页下，点击“实现接口”旁边的下拉菜单，选择“AttackAvoider”：

![图片](img/5019f64b-074e-4432-b9f3-f48a1b693772.jpg)

1.  编译你的蓝图：

![图片](img/8b5c7a09-df49-4876-af79-e9c5ff4b4c9f.jpg)

1.  通过点击事件图标签页并右键点击图中的任意位置，输入`event attack`。在上下文相关菜单中，你应该能看到“事件攻击进入”。选择它以在你的图中放置一个事件节点：

![图片](img/b80d7c45-4c16-4648-b626-9e234aff316f.jpg)

1.  从新节点的执行引脚上拖出，释放。在上下文相关菜单中输入`print string`以添加一个打印字符串节点：

![图片](img/6e86e174-4833-4ccc-aab1-d344fdb4f6c8.png)

选择“打印字符串”节点

你现在已经在蓝图内部实现了一个`UInterface`函数。

1.  要看到事件的实际效果，将事件开始播放事件右侧的引脚拖到右边，并调用一个攻击进入事件：

![图片](img/2a10d7b5-eb26-4f02-970a-af9a5d675ec0.png)

1.  将你的蓝图类的一个实例拖放到关卡中并玩游戏：

![图片](img/8621e029-18b4-4ca3-a875-bccd083bc7d9.png)

如果一切顺利，你应该会看到来自“打印字符串”的默认消息，或者当事件应该发生时你发布的任何消息！

# 它是如何工作的...

`UINTERFACE`/`IInterface`的创建方式与本章其他食谱中看到的方式完全相同。然而，当我们向接口添加函数时，我们使用一个新的`UFUNCTION`指定符，即`BlueprintImplementableEvent`。

`BlueprintImplementableEvent`告诉 Unreal 头文件工具生成代码，创建一个空占位函数，该函数可以被蓝图实现。我们不需要为该函数提供默认的 C++实现。

我们在蓝图内部实现接口，这样它就能以允许我们在蓝图内定义其实现的方式暴露函数。由头文件工具自动生成的代码将调用转发到我们的蓝图实现。

# 参见

“通过蓝图覆盖 C++ UInterface 函数”食谱展示了如何为你的`UInterface`函数在 C++中定义默认实现，然后根据需要可选地在蓝图中进行覆盖。

# 通过蓝图覆盖 C++ UInterface 函数

就像之前的食谱一样，U 接口很有用，但如果没有设计师可以使用其功能，这种实用性就严重受限。

之前的食谱“从本地基类中暴露 UInterface 方法到蓝图”展示了如何从蓝图调用 C++ `UInterface`函数；这个食谱将展示如何用你自己的自定义蓝图专用函数替换`UInterface`函数的实现。

# 如何实现...

1.  创建一个名为`Wearable`的新接口（创建`IWearable`和`UWearable`）：

![图片](img/5c357868-84cf-4094-9a7a-f75ac5064c56.png)

1.  将以下函数添加到`IWearable`类的头部：

```cpp
class CHAPTER_08_API IWearable
{
    GENERATED_BODY()

    // Add interface functions to this class. This is the
    // class that will be inherited to implement
    // this interface.
public:
 UFUNCTION(BlueprintNativeEvent, BlueprintCallable, 
 Category = Wearable)
 int32 GetStrengthRequirement();

 UFUNCTION(BlueprintNativeEvent, BlueprintCallable, 
 Category = Wearable)
 bool CanEquip(APawn* Wearer);

 UFUNCTION(BlueprintNativeEvent, BlueprintCallable, 
 Category = Wearable)
 void OnEquip(APawn* Wearer);
};
```

UE 4.20 及以上版本不允许我们在接口类中定义函数的默认实现，因此我们必须使用 UE 的默认空实现，这将为每个函数提供默认值作为返回值。这是因为，在 C#和其他有接口的语言中，它们不应该有默认实现。

1.  在编辑器中创建一个新的`Actor`类，名为`Boots`：

![图片](img/db097867-0667-461f-ac7a-17516d5381dc.png)

1.  将`#include "Wearable.h"`添加到`Boots`的头文件中，并修改类声明，如下所示：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Wearable.h"
#include "Boots.generated.h"

UCLASS()
class CHAPTER_08_API ABoots : public AActor, public IWearable
```

1.  添加以下由我们的接口创建的纯`virtual`函数的实现：

```cpp
UCLASS()
class CHAPTER_08_API ABoots : public AActor, public IWearable
{
    GENERATED_BODY()

public: 
    // Sets default values for this actor's properties
    ABoots();

protected:
    // Called when the game starts or when spawned
    virtual void BeginPlay() override;

public: 
    // Called every frame
    virtual void Tick(float DeltaTime) override;

 // Implementing the functions needed for IWearable
 virtual void OnEquip_Implementation(APawn* Wearer) override
 {
 GEngine->AddOnScreenDebugMessage(-1, 1, FColor::Red, 
                                         "Item being worn");
 } 
 virtual bool CanEquip_Implementation(APawn* Wearer) override
 {
 return true;
 } 
 virtual int32 GetStrengthRequirement_Implementation() override
 {
 return 0;
 }

};
```

如果您不知道如何执行以下两个步骤，请查看之前的配方，*在蓝图中实现 UInterface 函数*。

1.  编译您的脚本，以便我们可以访问我们创建的新函数。

1.  通过转到内容浏览器，打开`Content`文件夹，然后右键单击并选择蓝图类，创建一个新的基于`Actor`的蓝图类`Gloves`。

1.  在`Class Settings`菜单中，在`Details`选项卡下，滚动到`Implemented Interfaces`属性，并点击`Add`按钮，选择`Wearable`作为`Gloves`演员将实现的接口：

![图片](img/6e0652ae-bd66-450b-9bc7-859d6ff1eae0.png)

添加可穿戴接口

1.  然后，点击编译按钮以应用更改。

1.  打开事件图，右键单击以创建一个新事件。在搜索栏中输入`on equip`，您应该在`Add Event`部分看到我们的事件：

![图片](img/c02f936d-d293-45b9-97f2-60b24aa5a932.png)

1.  这允许我们覆盖默认实现中的`OnEquip`函数，以执行我们想要的任何操作。例如，添加一个`Print String`节点，将`In String`设置为`Gloves being worn`：

![图片](img/aa9216fb-a775-41b3-854f-0bfdfdc6f958.png)

1.  点击编译按钮，然后您可以关闭蓝图。为了测试目的，将`Gloves`和`Boots`的副本拖入您的级别中。

1.  添加后，将以下蓝图代码添加到您的级别中：

![图片](img/7e5b7e41-6f9a-4a02-9065-c8fa4ed3e74f.png)

1.  验证`Boots`执行默认行为，而`Gloves`执行蓝图定义的行为：

![图片](img/95bdce70-174c-418e-98fe-88231ed6b461.png)

# 它是如何工作的...

此配方使用两个`UFUNCTION`指定符一起：`BlueprintNativeEvent`和`BlueprintCallable`。`BlueprintCallable`已在之前的配方中展示，是一种将`UFUNCTION`标记为在蓝图编辑器中可见和可调用的方法。

`BlueprintNativeEvent`表示一个具有默认 C++（本地代码）实现但也可以在蓝图中被覆盖的`UFUNCTION`。它是虚拟函数和`BlueprintImplementableEvent`的组合。

为了使此机制工作，Unreal 头文件工具生成函数的主体，以便如果存在，则调用函数的蓝图版本；否则，它将方法调用调度到本地实现。

`Boots`类实现了`IWearable`，覆盖了默认功能。相比之下，`Gloves`也实现了`IWearable`，但在蓝图中有覆盖的`OnEquip`实现。当我们使用关卡蓝图调用两个角色的`OnEquip`时，可以验证这一点。

# 从 C++中调用蓝图定义的接口函数

尽管之前的配方主要集中在 C++在蓝图中的可用性上，例如在蓝图中调用 C++函数以及用蓝图覆盖 C++函数，但这个配方展示了相反的操作：从 C++中调用蓝图定义的接口函数。

# 如何操作...

1.  创建一个新的`UInterface`名为`Talker`（创建`UTalker`/`ITalker`类）：

![图片](img/246f7357-2422-4d79-a871-d740fc5f04bf.png)

1.  添加以下`UFUNCTION`实现：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "Talker.generated.h"

// This class does not need to be modified.
UINTERFACE(MinimalAPI)
class UTalker : public UInterface
{
    GENERATED_BODY()
};

/**
 * 
 */
class CHAPTER_08_API ITalker
{
    GENERATED_BODY()

    // Add interface functions to this class. This is the
    // class that will be inherited to implement
    // this interface.
public:
 UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category = Talk)
 void StartTalking();

};
```

1.  创建一个新的基于`StaticMeshActor`的 C++类。请记住检查“显示所有类”并以此方式找到该类：

![图片](img/d78ffdd9-291d-4c5e-900b-4afe7bb3b150.png).

1.  点击“下一步”后，将新类命名为`TalkingMesh`：

![图片](img/89e67a4a-0978-4258-ba9c-0ef649ac5c8f.png)

1.  添加`#include`并修改类声明以包含谈话者接口：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "Engine/StaticMeshActor.h"
#include "Talker.h"
#include "TalkingMesh.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_08_API ATalkingMesh : public AStaticMeshActor, public ITalker
```

1.  还要将以下函数添加到类声明中：

```cpp
UCLASS()
class CHAPTER_08_API ATalkingMesh : public AStaticMeshActor, public ITalker
{
    GENERATED_BODY()

public:
 ATalkingMesh();
 void StartTalking_Implementation();
};
```

1.  在实现中，将以下内容添加到`TalkingMesh.cpp`中：

```cpp
#include "TalkingMesh.h"
#include "ConstructorHelpers.h"

ATalkingMesh::ATalkingMesh() : Super()
{
 auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>(TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));

 UStaticMeshComponent * SM = GetStaticMeshComponent();

 if(SM != nullptr)
 {
 if (MeshAsset.Object != nullptr)
 {
 SM->SetStaticMesh(MeshAsset.Object);
 SM->SetGenerateOverlapEvents(true);
 }

 SM->SetMobility(EComponentMobility::Movable);

 }

 SetActorEnableCollision(true);
}

void ATalkingMesh::StartTalking_Implementation()
{
 GEngine->AddOnScreenDebugMessage(-1, 1, FColor::Red, 
 TEXT("Hello there. What is your name?"));
}
```

1.  创建一个新的基于`DefaultPawn`的类来作为我们的玩家角色：

![图片](img/7c9e20bd-ff72-4add-b845-251e8019f6d6.png)

1.  一旦选择“下一步”，给类命名为`TalkingPawn`并选择“创建类”：

![图片](img/459b2a04-31ce-4d3b-8655-29fc0370b550.png)

1.  将以下内容添加到我们的类头文件中：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/DefaultPawn.h"
#include "Components/BoxComponent.h" // UBoxComponent
#include "TalkingPawn.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_08_API ATalkingPawn : public ADefaultPawn
{
  GENERATED_BODY()

public:
 // Sets default values for this character's properties
 ATalkingPawn();

 UPROPERTY()
 UBoxComponent* TalkCollider;

 UFUNCTION()
 void OnTalkOverlap(UPrimitiveComponent* OverlappedComponent, 
 AActor* OtherActor, 
 UPrimitiveComponent* OtherComp, 
 int32 OtherBodyIndex, bool bFromSweep, 
 const FHitResult & SweepResult);

};
```

1.  从`TalkingPawn.cpp`文件中，确保包含以下内容，以便我们能够访问`ITalker`和`UTalker`类：

```cpp
#include "TalkingPawn.h"
#include "Talker.h"
```

1.  之后实现构造函数：

```cpp
ATalkingPawn::ATalkingPawn() : Super()
{
    // Set this character to call Tick() every frame. You can
    // turn this off to improve performance if you
    // don't need it.
    PrimaryActorTick.bCanEverTick = true;

    TalkCollider = CreateDefaultSubobject<UBoxComponent>("TalkCollider");

    TalkCollider->SetBoxExtent(FVector(200, 200, 100));

    TalkCollider->OnComponentBeginOverlap.AddDynamic(this, &ATalkingPawn::OnTalkOverlap);

    TalkCollider->AttachTo(RootComponent);
}
```

1.  实现`OnTalkOverlap`：

```cpp
// Called to bind functionality to input
void ATalkingPawn::OnTalkOverlap(UPrimitiveComponent* OverlappedComponent, 
                                 AActor* OtherActor, 
                                 UPrimitiveComponent* OtherComp, 
                                 int32 OtherBodyIndex, bool bFromSweep, 
                                 const FHitResult & SweepResult)
{
    auto Class = OtherActor->GetClass();
    if (Class->ImplementsInterface(UTalker::StaticClass()))
    {
        ITalker::Execute_StartTalking(OtherActor);
    }
}
```

1.  编译你的脚本。创建一个新的`GameMode`并将`TalkingPawn`设置为玩家的默认角色类。最快的方法是转到`设置 | 世界设置`，然后在`游戏模式覆盖`下点击加号按钮。从那里，展开所选游戏模式选项，在`默认角色类`下选择`TalkingPawn`。参考以下截图：

![图片](img/6aca0461-7c62-4d54-bf82-4b8e96c8b0c9.png)

1.  将你的`ATalkingMesh`类实例拖入关卡。如果你现在玩游戏，你应该能够走近网格并看到它显示消息：

![图片](img/4b6e90fc-40da-46b1-b6b9-98cbbf90fba9.png)

1.  通过在内容浏览器中右键单击`ATalkingMesh`并从上下文菜单中选择适当的选项来创建一个新的蓝图类：

![图片](img/fa84dd2a-1452-4c4c-a716-b9a32a62d245.png)

1.  命名为`MyTalkingMesh`并选择“创建蓝图类”：

![图片](img/d4c82f32-2f65-4bb6-9c3f-29d9235c2bb6.png)

1.  在蓝图编辑器内部，为`StartTalking`创建一个实现。我们可以通过进入事件图并在此图中右键单击来实现这一点。然后，在搜索栏中，我们可以输入`start talking`。在添加事件下，选择`Event Start Talking`选项。

![图片](img/10bf1f2e-cf6a-4c8f-8247-a3b2472aed4d.png)

1.  如果你想要调用事件的上层版本，你可以右键单击事件节点并选择“添加对父函数的调用”选项：

![图片](img/45898ae1-509a-492e-b45b-9f3771054836.png)

1.  之后，你可以连接事件。要执行与原始不同的操作，创建一个`Print String`节点并显示一个新的`In String`消息，例如`I'm the overridden implementation in Blueprint`。示例的最终版本将如下所示：

![图片](img/e67f1e3d-201a-4a7a-ae37-b61e8c2558ca.png)

1.  编译你的蓝图。之后，将你的新蓝图的一个副本拖到你的`ATalkingMesh`实例旁边的级别中。

1.  走到两个演员旁边，验证你的自定义`Pawn`是否正确调用了默认的 C++实现或蓝图实现，具体取决于情况：

![图片](img/ccd4ed7e-9285-4a9f-907d-8a2c011b9402.png)

# 它是如何工作的...

总是，我们创建一个新的接口，然后向`IInterface`类添加一些函数定义。我们使用`BlueprintNativeEvent`指定符来表示我们想要在 C++中声明一个默认实现，然后可以在蓝图中进行覆盖。我们创建一个新的类（为了方便继承自`StaticMeshActor`）并在其上实现接口。

在新类构造函数的实现中，我们加载一个静态网格并设置我们的碰撞，就像平常一样。然后我们添加一个接口函数的实现，它只是将一条消息打印到屏幕上。

如果你在一个完整的项目中使用这个功能，你可以播放动画、播放音频、更改用户界面，以及进行其他必要的操作来与你的`Talker`开始对话。

然而，此时我们还没有任何东西可以在我们的`Talker`上调用`StartTalking`。实现这一点最简单的方法是创建一个新的`Pawn`子类（再次，为了方便继承自`DefaultPawn`），它可以与它碰撞到的任何`Talker`演员开始交谈。

为了使这起作用，我们创建一个新的`BoxComponent`来建立触发对话的半径。像往常一样，它是一个`UPROPERTY`，所以它不会被垃圾回收。我们还为当新的`BoxComponent`与场景中的另一个`Actor`重叠时将被触发的一个函数创建了定义。

我们的`TalkingPawn`构造函数初始化新的`BoxComponent`，并适当地设置其范围。构造函数还绑定`OnTalkOverlap`函数作为事件处理程序来处理与我们的`BoxComponent`的碰撞。它还把盒子组件附加到我们的`RootComponent`上，这样当玩家在级别中移动时，它就会随着玩家角色一起移动。

在`OnTalkOverlap`内部，我们需要检查与我们盒子重叠的其他演员是否实现了`Talker`接口。最可靠的方法是使用`UClass`中的`ImplementsInterface`函数。此函数使用在编译期间由 Unreal Header Tool 生成的类信息，并且正确处理了 C++和 Blueprint 实现的接口。

如果函数返回`true`，我们可以使用包含在我们的`IInterface`中的特殊自动生成函数来调用我们实例上选择的接口方法。这是一个形式为`<IInterface>::Execute_<FunctionName>`的静态方法。在我们的实例中，我们的`IInterface`是`ITalker`，函数是`StartTalking`，所以我们要调用的函数是`ITalker::Execute_StartTalking()`。

我们需要这个函数的原因是，当接口在 Blueprint 中实现时，关系实际上并没有在编译时建立。因此，C++并不知道接口已被实现，所以我们不能将 Blueprint 类强制转换为`IInterface`来直接调用函数。

`Execute_`函数接受实现接口的对象的指针，并调用多个内部方法来调用所需函数的 Blueprint 实现。

当你播放关卡并在周围走动时，自定义的`Pawn`会不断接收到通知，当其`BoxComponent`与其他对象重叠时。如果它们实现了`UTalker`/`ITalker`接口，那么这个`Pawn`就会尝试在相关的`Actor`实例上调用`StartTalking`，然后屏幕上会打印出相应的信息。
