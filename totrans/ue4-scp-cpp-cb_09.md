# 集成 C++ 和 Unreal 编辑器：第一部分

在本章中，我们将介绍以下食谱：

+   使用类或结构体作为蓝图变量

+   创建可以在蓝图中继承的类或结构体

+   创建可以在蓝图中调用的函数

+   创建可以在蓝图中实现的函数

+   将多播委托公开给蓝图

+   创建可以在蓝图中使用的 C++ 枚举

+   在编辑器中的不同位置编辑类属性

+   使属性在蓝图编辑器图中可访问

+   响应来自编辑器的属性更改事件

+   实现原生代码构建脚本

# 简介

Unreal 的主要优势之一是它为程序员提供了创建可以由编辑器中的设计师自定义或使用的 Actor 和其他对象的能力。本章将向您展示如何做到这一点。之后，我们将尝试通过从头创建自定义蓝图和动画节点来自定义编辑器。我们还将实现自定义编辑器窗口和自定义详细信息面板来检查用户创建的类型。

# 技术要求

本章需要使用 Unreal Engine 4，并使用 Visual Studio 2017 作为 IDE。有关如何安装这两款软件及其要求的说明，请参阅第一章，*UE4 开发工具*。

# 使用类或结构体作为蓝图变量

在 C++ 中声明的类型不会自动集成到蓝图中作为变量使用。本食谱向您展示如何使它们可访问，以便您可以使用自定义原生代码类型作为蓝图函数参数。

# 如何做到这一点...

1.  使用编辑器创建一个新类。与前面的章节不同，我们将创建一个基于 `Object` 的类。`Object` 在默认的常用类列表中不可见，因此我们需要在编辑器 UI 中勾选显示所有类按钮，然后选择 Object。之后，点击下一步按钮：

![图片](img/97bdcafe-9d31-4b14-a8e4-afda49cd46d7.png)

1.  将您的新对象子类命名为 `TileType`，然后点击创建类按钮：

![图片](img/bb2c7553-314d-4c9a-af49-5814ac547b50.png)

1.  将以下属性添加到 `TileType` 定义中：

```cpp
UCLASS()
class CHAPTER_09_API UTileType : public UObject
{
    GENERATED_BODY()
public:
 UPROPERTY()
 int32 MovementCost;

 UPROPERTY()
 bool CanBeBuiltOn;

 UPROPERTY()
 FString TileName;
};
```

1.  编译您的代码。

1.  在编辑器内部，基于 `Actor` 创建一个新的蓝图类，命名为 `Tile`：

![图片](img/3a50d41d-8514-4f0f-af7e-6129a0374c88.png)

1.  在 `Tile` 的蓝图编辑器中，通过转到我的蓝图部分，然后移动到变量部分并点击 + 按钮，向蓝图添加一个新变量。屏幕右侧的详细信息面板将填充有关此新变量的信息，包括其类型。检查变量类型属性下可以创建为变量的类型列表，并验证 `TileType` 是否不在其中：

![图片](img/930a3f71-eaba-4f0c-9416-39c8cd3f517c.png)

1.  返回 Visual Studio 并打开 `TileType.h` 文件。将 `BlueprintType` 添加到 `UCLASS` 宏中，如下所示：

```cpp
UCLASS(BlueprintType)
class CHAPTER_09_API UTileType : public UObject
```

1.  保存您的脚本，返回到编辑器并重新编译项目，然后返回到 `Tile` 蓝图（Blueprint）编辑器。

1.  现在，当您向您的演员添加新变量时，您可以将 `TileType` 作为您新变量的类型选择：

![图片](img/f84aea2a-98c2-4c81-8379-494983245449.png)

1.  您现在可以将变量名更改为更好的名称，例如 `MyTileType`。

我们现在在 `Tile` 和 `TileType` 之间建立了一个 *has-a* 关系。现在，`TileType` 是一个可以作为函数参数使用的蓝图（Blueprint）类型。

1.  要这样做，请转到 My Blueprint 部分，并滚动到函数部分。从那里，您可以点击 + 按钮创建一个新函数。将这个新函数命名为 `SetTileType`：

![图片](img/4f058424-18ef-43d7-8b36-2b7664febb9b.png)

1.  函数创建后，详细信息选项卡将显示有关函数本身的信息。在输入部分下，点击 + 按钮添加一个新输入：

![图片](img/a885d752-02c3-406c-8f94-d267d00535c6.png)

1.  一旦选择，您将能够给变量命名并从默认显示为 `Boolean` 的下拉菜单中选择类型。将输入参数的类型设置为 `TileType`：

![图片](img/dca37210-53ef-4160-87e5-6439c6247fa8.png)

一旦这样做，您会看到参数已被添加到蓝图（Blueprint）中的设置瓷砖类型（Set Tile Type）函数的输入中：

![图片](img/6821f604-caf0-4c04-8074-0ae193b002ea.png)

1.  返回到 My Blueprint 部分，将 `MyTileType` 变量拖放到设置瓷砖类型（Set Tile Type）图旁边，紧挨着第一个节点。您可以将您的 `Type` 变量拖放到视图中，并选择设置 MyTileType：

![图片](img/3b1d3861-9389-4e2e-aea7-5d96da0a0e5b.png)

1.  现在我们有了所需的两个节点，将 Exec 输出引脚连接到设置 MyTileType 节点的输入，然后将 `SetTileType` 的参数连接到设置节点：

![图片](img/44e78dd9-47fe-4cc1-bd19-960caa91ba82.png)

# 它是如何工作的...

由于性能原因，Unreal 假设类不需要额外的反射代码，这些代码是使类型在蓝图（Blueprint）中可用的。

我们可以通过在 `UCLASS` 宏中指定 `BlueprintType` 来覆盖此默认设置。

包含指定符后，类型现在作为参数或变量在蓝图（Blueprint）中可用，并且可以像默认类型一样使用。

# 还有更多...

这个配方表明，如果其本地代码声明包含 `BlueprintType`，您可以在蓝图（Blueprint）中使用类型作为函数参数。

然而，目前，我们在 C++ 中定义的属性都无法在蓝图（Blueprint）中访问。

本章中的其他配方涉及使这些属性可访问，以便我们实际上可以用我们的自定义对象做一些有意义的事情。

# 在蓝图（Blueprint）中可以继承的类或结构体

虽然这本书主要关注 C++，但在使用 Unreal 开发时，更标准的流程是在 C++中实现核心游戏功能以及性能关键代码，并将这些功能暴露给蓝图，以便设计师可以原型化游戏玩法，然后程序员可以使用额外的蓝图功能对其进行重构，或者将其推回到 C++层。因此，最常见的任务之一就是以某种方式标记我们的类和结构，使它们对蓝图系统可见。

# 如何做到这一点...

1.  使用编辑器向导创建一个从`Actor`类派生的新的 C++类；命名为`BaseEnemy`：

![](img/ae20069a-adfb-4955-b837-b70770fa206b.png)

1.  将以下`UPROPERTY`添加到类中：

```cpp
UPROPERTY() 
FString WeaponName; 
UPROPERTY() 
int32 MaximumHealth;
```

1.  将以下类指定符添加到`UCLASS`宏中：

```cpp
UCLASS(Blueprintable)
class CHAPTER_09_API ABaseEnemy : public AActor
```

1.  保存并编译脚本。

1.  打开编辑器并创建一个新的蓝图类。展开所有类列表以显示所有类，并将我们的`BaseEnemy`类作为父类选择。之后，点击选择按钮：

![](img/21397387-9ab2-4a72-b976-7f2e71fc8f7b.png)

1.  将新的蓝图命名为`EnemyGoblin`并在蓝图编辑器中打开它。

注意，我们之前创建的`UPROPERTY`宏仍然不存在，因为我们还没有包含适当的标记使它们对蓝图可见。

# 它是如何工作的...

之前的配方演示了将`BlueprintType`用作类指定符的使用。`BlueprintType`允许类型在蓝图编辑器中使用（即，它可以是一个变量或函数输入/返回值）。

然而，我们可能希望根据我们的类型（使用继承）而不是组合（例如，在`Actor`中放置我们的类型的实例）来创建蓝图。

这就是为什么 Epic 提供了`Blueprintable`作为类指定符。`Blueprintable`意味着开发者可以将一个类标记为可以被蓝图类继承。

我们有`BlueprintType`和`Blueprintable`而不是单个组合指定符，因为有时你可能只想暴露部分功能。例如，某些类应该可以作为变量使用，但出于性能原因，不允许在蓝图中创建它们。在这种情况下，你会使用`BlueprintType`而不是两个指定符。

另一方面，也许我们想使用蓝图编辑器来创建新的子类，但又不希望在`Actor`蓝图内部传递对象实例。在这种情况下，建议使用`Blueprintable`，但省略`BlueprintType`。

与之前一样，`Blueprintable`和`BlueprintType`都没有指定我们类中包含的成员函数或成员变量。我们将在后面的配方中使它们可用。

# 创建可以在蓝图中调用的函数

当将类标记为`BlueprintType`或`Blueprintable`时，允许我们在蓝图之间传递类的实例，或者使用蓝图类来子类化类型，但这些指定符实际上并没有说关于成员函数或变量的事情，以及它们是否应该暴露给蓝图。这个食谱展示了如何标记一个函数，以便它可以在蓝图图中被调用。

# 如何做到这一点...

1.  使用编辑器向导创建一个从`StaticMeshActor`类派生的新的 C++类；命名为`SlidingDoor`。

1.  在新类中添加以下粗体文本：

```cpp
class CHAPTER_09_API ASlidingDoor : public AStaticMeshActor
{
    GENERATED_BODY()

public: 
    // Sets default values for this actor's properties
    ASlidingDoor();

protected:
    // Called when the game starts or when spawned
    virtual void BeginPlay() override;

public: 
    // Called every frame
    virtual void Tick(float DeltaTime) override;

 UFUNCTION(BlueprintCallable, Category = Door)
 void Open();

 UPROPERTY()
 bool IsOpen;

 UPROPERTY()
 FVector TargetLocation;
};
```

1.  通过在`.cpp`文件中添加以下粗体文本来创建类实现：

```cpp
#include "SlidingDoor.h"
#include "ConstructorHelpers.h"

// Sets default values
ASlidingDoor::ASlidingDoor()
{
    // Set this actor to call Tick() every frame. You can turn
    // this off to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = true;

    auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>
    (TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));

 UStaticMeshComponent * SM = GetStaticMeshComponent();

 if (SM != nullptr)
 {
 if (MeshAsset.Object != nullptr)
 {
 SM->SetStaticMesh(MeshAsset.Object);
 SM->SetGenerateOverlapEvents(true);
 }

 SM->SetMobility(EComponentMobility::Movable);
 SM->SetWorldScale3D(FVector(0.3, 2, 3));
 }

 SetActorEnableCollision(true);

 IsOpen = false;
 PrimaryActorTick.bStartWithTickEnabled = true;
}

// Called when the game starts or when spawned
void ASlidingDoor::BeginPlay()
{
    Super::BeginPlay();
}

// Called every frame
void ASlidingDoor::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

 if (IsOpen)
 {
 SetActorLocation(FMath::Lerp(GetActorLocation(), 
 TargetLocation, 0.05));
 }
}

void ASlidingDoor::Open()
{
 TargetLocation = ActorToWorld().TransformPositionNoScale( 
 FVector(0, 0, 200));
 IsOpen = true;
}

```

1.  编译你的代码并启动编辑器。

1.  将你的门拖到级别中：

![图片](img/7f1a01da-e630-4133-93f7-0a7381a4b1fd.png)

要让对象*下落*到地面，可以通过使用对象选择器上的 End 键来实现。

1.  确保你已选择你的`SlidingDoor`实例，然后通过转到蓝图 | 打开级别蓝图来打开级别蓝图。在空画布上右键单击并展开“Sliding Door 1”上的调用函数：

![图片](img/0b59687a-998c-4e30-99cb-1112f5ef746e.jpg)

1.  展开门部分，然后选择“打开”函数：

![图片](img/2e0152ed-e212-411e-a05f-a50e2c0c1a0c.jpg)

1.  将执行引脚（白色箭头）从“事件开始播放”链接到“打开”节点上的白色箭头，如图下截图所示：

![图片](img/8d197815-24a8-4c76-a565-e4c8c4a06e09.png)

1.  播放你的级别并验证当在门实例上调用`Open`时，门是否按预期向上移动：

![图片](img/e844796d-e944-4409-9136-09e03f95fa99.jpg)

# 它是如何工作的...

在门的声明中，我们创建了一个新的开门函数，一个布尔值用于跟踪门是否被指示打开，以及一个向量允许我们预先计算门的目标位置。

我们还重写了`Tick`演员函数，以便我们可以在每一帧执行一些行为。

在构造函数中，我们加载立方体网格并将其缩放以表示我们的门。

我们还将`IsOpen`设置为已知的良好值`false`，并使用`bCanEverTick`和`bStartWithTickEnabled`启用演员计时。

这两个布尔值分别控制是否可以为此演员启用计时，以及如果计时以启用状态开始，计时是否启动。

在`Open`函数内部，我们计算相对于门起始位置的目标位置。

我们还将`IsOpen`布尔值从`false`更改为`true`。

现在`IsOpen`布尔值为`true`，在`Tick`函数内部，门尝试使用`SetActorLocation`和`Lerp`来在当前位置和目的地之间进行插值，移动自己到目标位置。

# 参见

+   第五章，*处理事件和委托*，包含与演员生成相关的一些食谱

# 创建可以在蓝图实现的事件

C++可以更紧密地集成蓝图的另一种方式是通过创建具有蓝图实现的函数。这允许程序员指定一个事件并调用它，而无需了解任何有关实现的信息。然后该类可以在蓝图中被子类化，并且制作团队中的另一名成员可以不接触任何 C++代码就实现事件的处理程序。

# 如何做到这一点...

1.  创建一个名为`Spotter`的新`StaticMeshActor`类。记住使用显示所有类按钮来选择`StaticMeshActor`作为父类。

1.  确保以下函数在类头中定义并重写：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "Engine/StaticMeshActor.h"
#include "Spotter.generated.h"

UCLASS()
class CHAPTER_09_API ASpotter : public AStaticMeshActor
{
  GENERATED_BODY()

public:
 // Sets default values for this actor's properties
 ASpotter();

 // Called every frame
 virtual void Tick(float DeltaSeconds) override;

 UFUNCTION(BlueprintImplementableEvent)
 void OnPlayerSpotted(APawn* Player);

};
```

1.  在实现文件（`Spotter.cpp`）中，将代码更新如下：

```cpp
#include "Spotter.h"
#include "ConstructorHelpers.h"
#include "DrawDebugHelpers.h"

// Sets default values
ASpotter::ASpotter()
{
    // Set this actor to call Tick() every frame. You can
    // turn this off to improve performance if
    // you don't need it.
    PrimaryActorTick.bCanEverTick = true;

    // Set up visual aspect of the spotter
    auto MeshAsset = 
    ConstructorHelpers::FObjectFinder<UStaticMesh>
    (TEXT("StaticMesh'/Engine/BasicShapes/Cone.Cone'"));

    UStaticMeshComponent * SM = GetStaticMeshComponent();

    if (SM != nullptr)
    {
        if (MeshAsset.Object != nullptr)
        {
            SM->SetStaticMesh(MeshAsset.Object);
            SM->SetGenerateOverlapEvents(true);
        }

        SM->SetMobility(EComponentMobility::Movable);
        SM->SetRelativeRotation(FRotator(90, 0, 0));
    }

}

// Called every frame
void ASpotter::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    auto EndLocation = GetActorLocation() + 
    ActorToWorld().TransformVector(FVector(0, 0, -200));

    // Check if there is an object in front of us
    FHitResult HitResult;
    GetWorld()->SweepSingleByChannel(HitResult,
    GetActorLocation(), EndLocation, FQuat::Identity, 
    ECC_Camera, FCollisionShape::MakeSphere(25), 
    FCollisionQueryParams("Spot", true, this));

    APawn* SpottedPlayer = Cast<APawn>(HitResult.Actor.Get());

    // If there is call the OnPlayerSpotted function
    if (SpottedPlayer != nullptr)
    {
        OnPlayerSpotted(SpottedPlayer);
    }

    // Displays where we are checking for collision
    DrawDebugLine(GetWorld(), GetActorLocation(), EndLocation, FColor::Red);

}
```

1.  编译并启动编辑器。在内容浏览器中找到你的`Spotter`类，然后左键单击并拖动一个副本到游戏世界中。

1.  当你播放级别时，你会看到表示`Actor`执行的红色线条：

![图片](img/9914c7d6-45da-4f8f-b9f5-1a67276f9a50.png)

1.  然而，如果玩家站在它前面，什么也不会发生，因为我们还没有实现我们的`OnPlayerSpotted`事件。

1.  要实现此事件，我们需要创建一个我们的`Spotter`的蓝图子类。

1.  在内容浏览器中右键单击`Spotter`，并选择基于 Spotter 创建蓝图类。将类命名为`BPSpotter`：

![图片](img/75939870-ef59-44a6-b9fa-38b3e8635a19.png)

基于 Spotter 创建蓝图类

1.  在蓝图编辑器中，点击我的蓝图面板功能区域中的覆盖按钮：

![图片](img/8f457866-8e37-4655-9abc-650ec2e5e8af.png)

1.  选择玩家被看到：

![图片](img/1e41605e-623d-40fb-844c-b47593bb12fe.png)

1.  要查看事件，请点击事件图标签。左键单击它并将其从事件上的白色执行引脚拖离。在出现的上下文菜单中，选择并添加一个`Print String`节点，使其与事件链接：

![图片](img/e38c3014-8e80-4e2a-9787-ceb994895a6e.png)

1.  在级别中删除你的上一个 Spotter 对象，然后拖放一个`BPSpotter`。再次播放级别并验证现在使用`BPSpotter`的跟踪行在玩家走过时会打印字符串到屏幕上：

![图片](img/96f3cfec-578d-40bb-9af1-1e813185972f.png)

# 它是如何工作的...

在我们的`Spotter`对象构造函数中，我们将一个基本原语，一个圆锥体，加载到我们的静态网格组件中作为视觉表示。

我们然后将圆锥体旋转，使其类似于指向演员*x*轴的聚光灯。

在`Tick`函数期间，我们获取演员的位置，然后在其局部*x*轴上找到距离演员 200 个单位的点。我们使用`Super::`调用父类`Tick`的实现，以确保保留任何其他`Tick`功能，尽管我们进行了覆盖。

我们通过首先获取`Actor`的 Actor-to-World 转换，然后使用该转换来转换一个指定位置的向量，将局部位置转换为世界空间位置。

变换基于根组件的朝向，这是我们在构造函数期间旋转的静态网格组件。

由于存在旋转，我们需要旋转我们想要变换的向量。鉴于我们希望向量指向圆锥的底部，我们希望沿着负向上轴有一个距离；也就是说，我们希望有一个形式为（0，0，-d）的向量，其中*d*是实际的距离。

在计算出我们的跟踪终点后，我们实际上使用`SweepSingleByChannel`函数执行跟踪。

一旦执行了扫描，我们尝试将结果击中的`Actor`转换为士兵。

如果转换成功，我们调用我们的可实施事件`OnPlayerSpotted`，并执行用户定义的蓝图代码。

# 将多播委托暴露给蓝图

多播委托是一种向多个可能监听或订阅相关事件的物体广播事件的绝佳方式。如果你有一个生成事件的 C++模块，这些事件可能会被任意演员通知，那么它们尤其有价值。这个配方展示了如何在 C++中创建一个多播委托，以便在运行时通知一组其他演员。

# 如何做到这一点...

1.  创建一个名为`King`的新`StaticMeshActor`类。将以下内容添加到类头文件中：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "Engine/StaticMeshActor.h"
#include "King.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnKingDeathSignature, AKing*, DeadKing);
UCLASS()
class CHAPTER_09_API AKing : public AStaticMeshActor
{
  GENERATED_BODY()
```

1.  我们还希望在屏幕上显示一些内容，因此添加一个构造函数的定义：

```cpp
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnKingDeathSignature, AKing*, DeadKing);
UCLASS()
class CHAPTER_09_API AKing : public AStaticMeshActor
{
    GENERATED_BODY()

 // Sets default values for this actor's properties
 AKing();
};
```

1.  向类中添加一个新的`UFUNCTION`：

```cpp
UFUNCTION(BlueprintCallable, Category = King) 
void Die(); 
```

1.  将我们的多播委托实例添加到类中：

```cpp
UPROPERTY(BlueprintAssignable) 
FOnKingDeathSignature OnKingDeath; 
```

1.  打开`King.cpp`文件，然后添加构造函数的实现以执行我们的网格初始化（记得添加`ConstructionHelpers.h`文件的`#include`）：

```cpp
#include "King.h"
#include "ConstructorHelpers.h"

// Sets default values
AKing::AKing()
{
 // Set this actor to call Tick() every frame. You can turn
    // this off to improve performance if you don't need it.
 PrimaryActorTick.bCanEverTick = true;

 auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>
    (TEXT("StaticMesh'/Engine/BasicShapes/Cone.Cone'"));

 UStaticMeshComponent * SM = GetStaticMeshComponent();

 if (SM != nullptr)
 {
 if (MeshAsset.Object != nullptr)
 {
 SM->SetStaticMesh(MeshAsset.Object);
 SM->SetGenerateOverlapEvents(true);
 }
 SM->SetMobility(EComponentMobility::Movable);
 }
}
```

1.  实现`Die`函数：

```cpp
void AKing :: Die () 
{ 
  OnKingDeath.Broadcast(this); 
} 
```

1.  创建一个名为`Peasant`的新类，也基于`StaticMeshActor`。

1.  在类中声明默认构造函数：

```cpp
APeasant (); 
```

1.  声明以下函数：

```cpp
UFUNCTION(BlueprintCallable, category = Peasant) 
void Flee (AKing * DeadKing); 
```

1.  实现构造函数：

```cpp
#include "Peasant.h"
#include "ConstructorHelpers.h"

APeasant::APeasant()
{
  // Set this actor to call Tick() every frame. You can
  // turn this off to improve performance if 
  // you don't need it.
  PrimaryActorTick.bCanEverTick = true;

  auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>
  (TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));

  UStaticMeshComponent * SM = GetStaticMeshComponent();

  if (SM != nullptr)
  {
    if (MeshAsset.Object != nullptr)
    {
      SM->SetStaticMesh(MeshAsset.Object);
      SM->SetGenerateOverlapEvents(true);
    }
    SM->SetMobility(EComponentMobility::Movable);
  }
}
```

1.  在`.cpp`文件中实现`Flee`函数：

```cpp
void APeasant::Flee(AKing* DeadKing)
{
    // Display message on the screen
    GEngine->AddOnScreenDebugMessage(-1, 2, FColor::Red,
        TEXT("Waily Waily!"));

    // Get the direction away from the dead king
    FVector FleeVector = GetActorLocation() -
        DeadKing->GetActorLocation();

    // Set the magnitude (length) of the vector to 1
    FleeVector.Normalize();

    // Make the vector 500 times longer
    FleeVector *= 500;

    // Set the Actor's new location
    SetActorLocation(GetActorLocation() + FleeVector);
}
```

1.  返回 Unreal 编辑器并编译你的脚本。

1.  之后，基于`APeasant`创建一个蓝图类。你可以通过在内容浏览器中右键单击`Peasant`对象，然后选择基于 Peasant 创建蓝图类来完成此操作。将新类命名为`BPPeasant`。之后，单击创建蓝图类按钮：

![图片](img/d90c146a-d40e-4b97-ad38-1b4c93c2a64f.png)

1.  在蓝图内，单击事件图标签，向上移动到`Event BeingPlay`节点。单击并拖动它远离`BeginPlay`节点的白色（执行）引脚。输入`get all`，你应该看到获取所有类别的演员。选择节点将其放置在图中：

![图片](img/688d8fe5-46f8-4e55-9cbe-814a3c03422c.png)

1.  将紫色（类）节点的值设置为`King`。你可以在搜索栏中输入`king`，以便更容易地在列表中定位该类：

![图片](img/024b2403-c44e-4992-81d0-bf62b0f94269.png)

1.  将蓝色网格（对象数组）节点拖入一个空白区域，并从弹出的动作菜单中输入单词`get`。从可用的选项中选择获取（副本）选项：

![图片](img/4775d89e-250e-4867-879a-5d17ae725240.png)

1.  从获取节点的蓝色输出引脚拖动，放置一个不等式（对象）节点：

![图片](img/f7621e45-e678-460c-a3e5-ef35d2d5ec16.png)

1.  将不等式节点的红色（布尔）引脚连接到`Branch`节点，并将`Branch`的执行引脚连接到我们的`Get All Actors Of Class`节点：

![图片](img/57bbbe4c-7469-4ed1-bd16-419682944815.png)

1.  将分支的 True 引脚连接到绑定事件到 OnKing Death 节点：

![图片](img/165a3781-c973-477d-b8d5-9f92363d1c15.png)

注意，您可能需要在上下文菜单中取消选中上下文敏感选项，以便`Bind Event`节点可见。

1.  然后，将获取节点的输出连接到绑定事件到 OnKingDeath 节点的目标属性：

![图片](img/2b0c31ca-100d-40ba-8f2f-bd1ffbe4b584.png)

将“Get”节点连接到绑定事件到 OnKingDeath 节点的目标属性

如果您双击一个连接，您可以创建一个重路由节点，您可以将其拖动以使其更容易看到节点之间的连接。

1.  将绑定事件到 OnKingDeath 节点的红色引脚拖出，并选择“添加自定义事件....”给您的活动取一个期望的名字：

您可能需要取消选中上下文敏感选项才能看到“添加自定义事件...”选项。

![图片](img/bc779546-1339-4875-b6b8-2137e577236c.png)

连接自定义事件和事件绑定。

1.  将自定义事件的白色执行引脚连接到一个名为`Flee`的新节点，这是我们之前在第 10 步中创建的：

![图片](img/f1f23155-2850-4057-8707-9a47724fd8bf.png)

1.  最后，将自定义事件中的 Dead King 属性拖入 Flee 节点的`Dead King`属性中。

1.  确认您的蓝图看起来像以下截图所示：

![图片](img/fb6eb322-517f-46c1-910c-d2f09ba30a3a.png)

完成的蓝图

1.  将您的`King`类的一个副本拖入级别中，然后在其周围添加几个`BPPeasant`实例形成一个圆圈：

![图片](img/287e0110-9b2b-4353-8d05-18ad8101bde9.png)

1.  打开级别蓝图。在其内部，从`BeginPlay`拖动，并添加一个`Delay`节点。将延迟设置为 5 秒：

![图片](img/b444ee3f-2178-437e-b940-8a63dd56fa2e.jpg)

1.  在级别中选中您的`King`实例后，在图编辑器中右键单击级别蓝图。

1.  在 King 1 上选择调用函数，然后在`King`类别中查找名为`Die`的函数：

![图片](img/287aed93-e6e9-45a3-8ae2-517e3c05e79d.jpg)

1.  选择`Die`，然后将其执行引脚连接到延迟的输出执行引脚：

![图片](img/23c556e5-53bb-4ff5-8dd7-97bb10398bda.png)

1.  当您玩您的级别时，您应该看到国王在 5 秒后死亡：

![图片](img/b527406e-ae24-4ce9-87d5-1d06077f0ebe.png)

之后，您应该看到农民们都在哭泣并直接逃离国王：

![图片](img/9a1cc2a2-35c1-4201-8fb0-a09cb3f069fc.png)

# 它是如何工作的...

我们创建了一个新的演员（基于`StaticMeshActor`以方便起见，因为它可以节省我们为`Actor`视觉表示声明或创建静态网格组件的时间）。

我们使用`DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam`宏声明一个动态多播委托。动态多播委托允许任意数量的对象订阅（监听）和取消订阅（停止监听），以便在委托被广播时得到通知。

宏接受多个参数——正在创建的新委托签名类型的名称，签名参数的类型，然后是签名参数的名称。

我们还向`King`添加了一个函数，允许我们告诉它死亡。因为我们想将此函数暴露给蓝图以进行原型设计，所以我们将其标记为`BlueprintCallable`。

我们之前使用的`DECLARE_DYNAMIC_MULTICAST_DELEGATE`宏只声明了一个类型；它没有声明委托的实例，所以我们现在这样做，引用我们在调用宏时提供的类型名称。

动态多播委托可以在其`UPROPERTY`声明中标记为`BlueprintAssignable`。这表示向 Unreal 表明蓝图系统可以动态地将事件分配给委托，当委托的`Broadcast`函数被调用时，将调用这些委托。

和往常一样，我们给我们的`King`分配一个简单的网格，以便它在游戏场景中有视觉表示。

在`Die`函数内部，我们在自己的委托上调用`Broadcast`。我们指定委托将有一个参数，该参数是指向已故国王的指针，因此我们将此指针作为参数传递给广播函数。

如果你想让国王被摧毁，而不是在它死亡时播放动画或其他效果，你需要更改委托的声明并传入不同的类型。例如，你可以使用`FVector`，并直接传入死去的国王的位置，这样农民仍然可以适当逃跑。

没有这个，你可能会遇到这样的情况：当调用`Broadcast`时，`King`指针是有效的，但在你的绑定函数执行之前，对`Actor::Destroy()`的调用使其无效。

在我们的下一个`StaticMeshActor`子类中，称为`Peasant`，我们像往常一样初始化静态网格组件，使用与`King`不同的形状。

在农民的`Flee`函数的实现中，我们通过在屏幕上打印消息来模拟农民播放声音。

然后，我们计算一个向量，使农民逃跑，首先找到从死去的国王到这个农民位置的一个向量。

我们将向量归一化以检索一个单位向量（长度为 1），指向相同的方向。

缩放归一化向量并将其添加到我们的当前位置，计算出在固定距离和确切方向上的一个位置，这样农民就可以直接逃离死去的国王。

然后使用`SetActorLocation`来实际将农民传送到那个位置。

如果你使用了一个具有 AI 控制器的角色，你可以让`Peasant`路径找到目标位置而不是传送。或者，你可以使用一个在农民的`Tick`期间调用的`Lerp`函数，使他们平滑滑动而不是直接跳到位置。

# 参见

+   请参阅第四章，*演员和组件*，了解更多关于演员和组件的讨论。第五章，*处理事件和委托*，讨论了诸如`Notify`和`ActorOverlap`的事件。

# 创建在蓝图中使用 C++枚举

枚举在 C++中常用作标志或输入到 switch 语句中。然而，如果你想在蓝图中将枚举值传递到或从 C++传递，或者如果你想在蓝图中使用一个使用 C++枚举的`switch`语句，你该如何让蓝图编辑器知道你的枚举应该在编辑器中可访问？这个配方展示了如何在蓝图中使枚举可见。

# 如何做到这一点...

1.  使用编辑器创建一个新的名为`Tree`的`StaticMeshActor`类。

1.  在类声明上方插入以下代码：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "Engine/StaticMeshActor.h"
#include "Tree.generated.h"

UENUM(BlueprintType)
enum TreeType
{
 Tree_Poplar,
 Tree_Spruce,
 Tree_Eucalyptus,
 Tree_Redwood
};

UCLASS()
class CHAPTER_09_API ATree : public AStaticMeshActor
{
```

1.  将以下内容添加到`Tree`类中：

```cpp
UCLASS()
class CHAPTER_09_API ATree : public AStaticMeshActor
{
    GENERATED_BODY()

public:
 // Sets default values for this actor's properties
 ATree();

 UPROPERTY(BlueprintReadWrite)
 TEnumAsByte<TreeType> Type;
};
```

1.  将以下内容添加到`Tree`构造函数中：

```cpp
#include "Tree.h"

#include "ConstructorHelpers.h"

// Sets default values
ATree::ATree()
{
 // Set this actor to call Tick() every frame. You can turn
    // this off to improve performance if you don't need it.
 PrimaryActorTick.bCanEverTick = true;

 auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>
    (TEXT("StaticMesh'/Engine/BasicShapes/Cylinder.Cylinder'"));

 UStaticMeshComponent * SM = GetStaticMeshComponent();

 if (SM != nullptr)
 {
 if (MeshAsset.Object != nullptr)
 {
 SM->SetStaticMesh(MeshAsset.Object);
 SM->SetGenerateOverlapEvents(true);
 }
 SM->SetMobility(EComponentMobility::Movable);
 }
}

```

1.  返回 Unreal 编辑器并编译你的代码。

1.  通过右键单击 Tree 对象并选择基于 Tree 创建蓝图类来创建一个新的名为`MyTree`的蓝图类。一旦菜单出现，点击创建蓝图类按钮。

1.  在`MyTree`的蓝图编辑器中，点击构造脚本选项卡。

1.  在空窗口中右键单击并输入`treetype`。在 TreeType 节点中获取条目数：

![图片](img/8a5f09a0-43b0-4d0e-852f-f28683ca8a5c.png)

1.  然后将它的返回值输出引脚连接到新随机整数节点的最大属性：

![图片](img/aa62d8f0-eac4-4978-903c-33be135516cc.png)

1.  将随机整数的返回值输出连接到一个 ToByte (Integer)节点：

![图片](img/e638984a-344d-4bcb-9d28-dc2bbdcfe361.png)

1.  在 My Blueprint 面板的变量部分，点击+按钮。然后转到详细信息选项卡，将变量类型设置为`Tree Type`。之后，将变量名称设置为`RandomTree`：

![图片](img/43c68378-cfe8-4485-8e5f-1a654392ffb4.png)

1.  将 RandomTree 变量拖入图中，当出现一个小上下文菜单时，选择 Set Random Tree。

1.  将`ToByte`节点的返回值输出连接到 SET Type 节点的输入。你会看到一个额外的转换节点自动出现。

1.  最后，将构造脚本的执行引脚连接到 SET Type 节点的执行引脚。你的蓝图应该看起来如下：

![图片](img/17bd491f-612b-4926-9485-76609cd128da.png)

1.  为了验证蓝图是否正确运行并且随机分配类型到我们的树中，我们将在事件图中添加一些节点。

1.  在 Event BeginPlay 事件节点之后放置一个 `Print String` 节点：

![图片](img/155d1518-6a95-478b-89db-c434fbb4a3ea.png)

1.  放置一个 `Format Text` 节点，并将其输出连接到 `Print String` 节点的输入。系统会为你添加一个转换节点：

![图片](img/825ac029-d5de-496c-87fe-2f17a6ebcd72.png)

1.  在 `Format Text` 节点内部，将 `My Type is {0}!` 添加到格式文本框中：

![图片](img/d1b858b3-6f18-4cba-a7c8-8d54f5137af8.png)

你应该看到它添加了一个新的参数，0，我们现在可以设置它。

1.  将 RandomTree 变量从 My Blueprint 窗口的变量部分拖放到图中，并从菜单中选择 Get：

![图片](img/76fab5b0-2f7d-4c0e-a8d4-0ed39f776c22.png)

1.  将 Enum to Name 节点添加到 `Type` 输出引脚：

![图片](img/d92b7665-d8ed-4b79-b03a-ad3b56eb5f9f.png)

1.  Format Text 节点将不会使用名称，因此我们需要将其转换为文本。向 Enum to Name 输出引脚添加一个 ToText (name) 节点。

1.  将 ToText (name) 节点的返回值输出连接到 Format Text 节点的 0 输入引脚。你的事件图现在应该如下所示：

![图片](img/f79e30f9-f9b9-46cf-8784-a4a7e93d8c40.png)

完成的蓝图图

1.  编译你的蓝图，然后返回到虚幻编辑器。

1.  将几个蓝图副本拖放到级别中并播放。你应该会看到许多树打印有关它们类型的信息，验证蓝图代码正在随机分配类型：

![图片](img/b1534f7b-f726-4577-92d5-380d72ddacf2.png)

# 它是如何工作的...

如同往常，我们使用 `StaticMeshActor` 作为我们的 `Actor` 的基类，这样我们就可以轻松地在级别中为其提供一个视觉表示。

枚举类型通过 `UENUM` 宏暴露给反射系统。

我们使用 `BlueprintType` 指定符将 `enum` 标记为蓝图可用的。

`enum` 声明与我们在任何其他上下文中使用的完全相同。

我们的 `Tree` 需要一个 `TreeType`。因为 *tree has tree-type* 是我们想要体现的关系，我们在 `Tree` 类中包含了一个 `TreeType` 实例。

如同往常，我们需要使用 `UPROPERTY()` 来使成员变量可被反射系统访问。

我们使用 `BlueprintReadWrite` 指定符标记属性在蓝图内既有获取又有设置支持。

枚举类型在使用 `UPROPERTY` 时需要被 `TEnumAsByte` 模板包装，因此我们声明一个 `TEnumAsByte<TreeType>` 的实例作为树的 `Type` 变量。

`Tree` 构造函数的更改仅仅是标准的加载和初始化我们用于其他菜谱中的静态网格组件前缀。

我们创建了一个继承自我们的 `Tree` 类的蓝图，以便我们可以演示 `TreeType enum` 的蓝图可访问性。

为了在创建实例时让蓝图随机为树分配一个类型，我们需要使用构造脚本蓝图。

在构造脚本中，我们计算 `TreeType enum` 中的条目数。

我们生成一个随机数，并使用它作为 `TreeType` 枚举类型的索引来检索一个值作为我们的 `Type` 存储。

然而，随机数节点返回整数。在蓝图中，枚举类型被视为字节，因此我们需要使用 `ToByte` 节点，然后蓝图可以隐式地将它转换为 `enum` 值。

现在我们有了构造脚本，在创建树实例时为它们分配类型，我们需要在运行时显示树的类型。

我们通过在事件图选项卡中的 `BeginPlay` 事件内附加的图来做到这一点。

要在屏幕上显示文本，我们使用 `Print String` 节点。

要执行字符串替换并将我们的类型以人类可读的字符串打印出来，我们使用 `Format Text` 节点。

`Format Text` 节点用于提取花括号内的术语，并允许您通过返回最终字符串来替换这些术语的值。

要将我们的 `Type` 替换到 `Format Text` 节点中，我们需要将我们的变量存储从 `enum` 值转换为实际值名称。

我们可以通过访问我们的 `Type` 变量并使用 `Enum to Name` 节点来实现这一点。

`Names`，或在本地代码中为 `FNames`，是一种可以被蓝图转换为字符串的变量类型，这样我们就可以将我们的 `Name` 连接到 `Format Text` 节点的输入上。

当我们按下播放时，图执行，检索放置在级别中的树实例的类型，并将名称打印到屏幕上。

# 在编辑器中的不同位置编辑类属性

在使用 Unreal 进行开发时，程序员通常会在 C++ 中实现 Actor 或其他对象的属性，并在编辑器中使它们对设计师可见。然而，有时查看属性或使其可编辑，但仅在对象的默认状态下是有意义的。有时，属性应该在运行时使用在 C++ 中指定的默认值进行修改。幸运的是，有一些指定符可以帮助我们限制属性何时可用。

# 如何做到这一点...

1.  在编辑器中创建一个新的 `Actor` 类，命名为 `PropertySpecifierActor`：

![](img/1181d0a4-74a1-4ab3-ac08-773bf590a75a.png)

1.  向类添加以下属性定义：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "PropertySpecifierActor.generated.h"

UCLASS()
class CHAPTER_09_API APropertySpecifierActor : public AActor
{
    GENERATED_BODY()

public: 
    // Sets default values for this actor's properties
    APropertySpecifierActor();

protected:
    // Called when the game starts or when spawned
    virtual void BeginPlay() override;

public: 
    // Called every frame
    virtual void Tick(float DeltaTime) override;

    // Property Specifiers
    UPROPERTY(EditDefaultsOnly)
    bool EditDefaultsOnly;

    UPROPERTY(EditInstanceOnly)
    bool EditInstanceOnly;

    UPROPERTY(EditAnywhere)
    bool EditAnywhere;

    UPROPERTY(VisibleDefaultsOnly)
    bool VisibleDefaultsOnly;

    UPROPERTY(VisibleInstanceOnly)
    bool VisibleInstanceOnly;

    UPROPERTY(VisibleAnywhere)
    bool VisibleAnywhere;
};
```

1.  执行保存，编译您的代码，并启动编辑器。

1.  基于类创建一个新的蓝图。

1.  打开蓝图并查看类默认值部分：

![](img/04d585d0-842c-4de5-b9da-731758e27c2b.jpg)

1.  注意哪些属性在属性指定器演员部分下是可编辑和可见的：

![](img/1d639038-43e2-4ee8-abff-36f3ec2afa6b.png)

属性指定器演员的位置

1.  将实例放置在级别中并查看它们的详细信息面板：

![](img/16d4fe8b-9f98-40b7-96de-36f7d1188c15.png)

1.  注意，一组不同的属性是可编辑的。

# 它是如何工作的...

当指定 `UPROPERTY` 时，我们可以指示我们希望在 Unreal 编辑器内部何处使该值可用。

`Visible*` 前缀表示可以在指示对象的详细面板中查看值。然而，该值不可编辑。

这并不意味着变量是 `const` 标识符；然而，原生代码可以更改其值，例如。

`Edit*` 前缀表示可以在编辑器内的详细面板中更改属性。

作为后缀的 `InstanceOnly` 表示该属性将仅在已放置到游戏中的类的实例的详细面板中显示。例如，它们在蓝图编辑器的“类默认值”部分中不可见。

`DefaultsOnly` 是 `InstanceOnly` 的逆，`UPROPERTY` 只会在类默认值部分显示，并且不能在级别中的单个实例中查看。

前缀 `Anywhere` 是前两个前缀的组合——`UPROPERTY` 将在检查对象默认值或级别中的特定实例的所有详细面板中可见。

如我们之前提到的，如果您想了解更多关于属性指定符的信息，请查看以下链接：[`docs.unrealengine.com/en-us/Programming/UnrealArchitecture/Reference/Properties/Specifiers`](https://docs.unrealengine.com/en-us/Programming/UnrealArchitecture/Reference/Properties/Specifiers)。

# 参见

+   此配方使所讨论的属性在检查器中可见，但不会允许在实际的蓝图事件图中引用该属性。有关如何实现此功能的描述，请参阅以下配方。

# 在蓝图编辑器图中使属性可访问

我们在先前的配方中提到的指定符都很好，但它们只控制 `UPROPERTY` 在详细面板中的可见性。默认情况下，即使使用了这些指定符，`UPROPERTY` 也不会在真正的编辑器图中可见或可访问，以便在 `runtime.Other` 中使用。这些指定符可以与先前的配方中的指定符一起使用，以便您可以在事件图中与属性交互。

# 如何操作...

1.  使用编辑器向导创建一个新的 `Actor` 类，命名为 `BlueprintPropertyActor`。

![](img/40067092-2b56-401c-8825-8bfea89f5019.png)

1.  使用 Visual Studio 将以下 `UPROPERTY` 添加到类中：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BlueprintPropertyActor.generated.h"

UCLASS()
class CHAPTER_09_API ABlueprintPropertyActor : public AActor
{
    GENERATED_BODY()

public: 
    // Sets default values for this actor's properties
    ABlueprintPropertyActor();

protected:
    // Called when the game starts or when spawned
    virtual void BeginPlay() override;

public: 
    // Called every frame
    virtual void Tick(float DeltaTime) override;

 UPROPERTY(BlueprintReadWrite, Category = Cookbook)
 bool ReadWriteProperty;

 UPROPERTY(BlueprintReadOnly, Category = Cookbook)
 bool ReadOnlyProperty;

};
```

1.  执行保存，编译您的项目，并启动编辑器。

1.  基于您的 `BlueprintPropertyActor` 创建一个蓝图类，并打开其图。

1.  在“我的蓝图”面板中，点击搜索栏右侧的图标。从那里，选择显示继承变量：

![](img/04be7b53-3abc-4b96-91df-41feeef6a5b6.png)

1.  在“我的蓝图”面板的变量部分中，验证属性是否在“食谱”类别下可见：

![](img/bba20056-25a1-4ae7-90cf-639c18d06265.jpg)

1.  左键单击并拖动 `ReadWriteProperty` 变量到事件图中。然后选择获取读写属性：

![](img/5ba2bbaf-baa3-48ca-ab9c-c7c14d66d450.png)

1.  重复前面的步骤，但选择设置只读属性。

1.  将只读属性拖入图中，并注意 SET 节点被禁用：

![](img/b43adb75-6bdf-41e5-b6ec-2d103fb65f05.png)

# 如何工作...

作为`UPROPERTY`指定符的`BlueprintReadWrite`表示告诉 Unreal 头文件工具，该属性应该有`Get`和`Set`操作暴露以在蓝图中使用。

如其名所示，`BlueprintReadOnly`是一个指定符，它只允许蓝图检索属性的值；永远不能设置它。

当属性由原生代码设置时，`BlueprintReadOnly`可能很有用，但应在蓝图内可访问。

应注意，`BlueprintReadWrite`和`BlueprintReadOnly`不指定属性在详细信息面板或编辑器的“我的蓝图”部分中的可访问性：这些指定符仅控制为蓝图图生成 getter/setter 节点。

# 响应来自编辑器的属性更改事件

当设计师更改放置在关卡中的`Actor`的属性时，通常很重要立即显示该更改的任何视觉结果，而不仅仅是当关卡模拟或播放时。当使用详细信息面板进行更改时，编辑器会发出一个特殊事件，称为`PostEditChangeProperty`，它给类实例一个机会来响应正在编辑的属性。这个配方展示了如何处理`PostEditChangeProperty`以获得即时的编辑器反馈。

# 如何做到...

1.  创建一个新的基于`StaticMeshActor`的`Actor`，命名为`PostEditChangePropertyActor`：

![](img/45976a36-1621-403c-973b-afa3c859cabe.png)

1.  将以下`UPROPERTY`和函数定义添加到类中：

```cpp
UCLASS()
class CHAPTER_09_API APostEditChangePropertyActor : public 
AStaticMeshActor
{
    GENERATED_BODY()

 // Sets default values for this actor's properties
 APostEditChangePropertyActor();

 UPROPERTY(EditAnywhere)
 bool ShowStaticMesh = true;

 virtual void PostEditChangeProperty(FPropertyChangedEvent& 
                                        PropertyChangedEvent) override;

};
```

1.  通过将以下代码添加到`PostEditChangePropertyActor.cpp`文件中创建类构造函数：

```cpp
#include "PostEditChangePropertyActor.h"
#include "ConstructorHelpers.h"

APostEditChangePropertyActor::APostEditChangePropertyActor()
{
 // Set this actor to call Tick() every frame. You can turn
    // this off to improve performance if you don't need it.
 PrimaryActorTick.bCanEverTick = true;

 auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>
    (TEXT("StaticMesh'/Engine/BasicShapes/Cone.Cone'"));

 UStaticMeshComponent * SM = GetStaticMeshComponent();

 if (SM != nullptr)
 {
 if (MeshAsset.Object != nullptr)
 {
 SM->SetStaticMesh(MeshAsset.Object);
 SM->SetGenerateOverlapEvents(true);
 }
 SM->SetMobility(EComponentMobility::Movable);
 }
}
```

1.  实现`PostEditChangeProperty`：

```cpp
void APostEditChangePropertyActor::PostEditChangeProperty( FPropertyChangedEvent& PropertyChangedEvent)
{
    // Check if property is valid
    if (PropertyChangedEvent.Property != nullptr)
    {
        // Get the name of the changed property
        const FName PropertyName( 
                            PropertyChangedEvent.Property->GetFName());

        // If the changed property is ShowStaticMesh then we
        // will set the visibility of the actor
        if (PropertyName == GET_MEMBER_NAME_CHECKED( 
                         APostEditChangePropertyActor, ShowStaticMesh))
        {
            UStaticMeshComponent * SM = GetStaticMeshComponent();

            if (SM != nullptr)
            {
                SM->SetVisibility(ShowStaticMesh);
            }
        }
    }

    // Then call the parent version of this function
    Super::PostEditChangeProperty(PropertyChangedEvent);
}
```

1.  编译你的代码并启动编辑器。

1.  将你的类的一个实例拖入游戏世界，并验证切换`ShowStaticMesh`的布尔值是否在编辑器视图中切换网格的可见性：

![](img/dba850f4-0119-4dac-85a7-e24f6a9e78f0.png)

显示静态网格属性的定位

然后，如果你将其关闭，你会看到对象消失，如下所示：

![](img/340c8760-1eda-48ab-91fd-99f5034777b0.png)

# 如何工作...

我们基于`StaticMeshActor`创建一个新的`Actor`，以便通过静态网格访问视觉表示。

添加`UPROPERTY`以给我们一个可以更改的属性，这将触发`PostEditChangeProperty`事件。

`PostEditChangeProperty`是一个在`Actor`中定义的虚函数。

因此，我们覆盖了我们类中的函数。

在我们的类构造函数中，我们像往常一样初始化我们的网格，并将我们的`bool`属性的默认状态设置为与它控制的组件的可见性相匹配。

在`PostEditChangeProperty`内部，我们首先检查属性是否有效。

假设它是，我们使用`GetFName()`检索属性的名称。

`FNames`在引擎内部以唯一值表的形式存储。

接下来，我们需要使用`GET_MEMBER_NAME_CHECKED`宏。该宏接受多个参数。

第一个参数是要检查的类的名称，而第二个参数是要检查该类的属性。

该宏将在编译时验证类是否包含由名称指定的成员。

我们将宏返回的类成员名称与我们的属性包含的名称进行比较。

如果它们相同，那么我们验证我们的`StaticMeshComponent`是否正确初始化。

如果是，我们将它的可见性设置为与我们的`ShowStaticMesh`布尔值匹配。

# 实现原生代码构建脚本

在 Blueprint 中，一个构建脚本是一个事件图，它在任何时间对象上更改属性时都会运行——无论是被拖动到编辑器视图中，还是通过细节面板的直接输入进行更改。构建脚本允许相关对象根据其新位置*重建*自己，例如，或者根据用户选择的选项更改其包含的组件。在用 Unreal Engine 以 C++进行编码时，等效的概念是`OnConstruction`函数。

# 如何做到这一点...

1.  基于`StaticMeshActor`创建一个新的名为`OnConstructionActor`的`Actor`：

![](img/e5a597b2-f38a-4fbf-9095-db45235e8c34.png)

1.  将头文件更新为以下内容：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "Engine/StaticMeshActor.h"
#include "OnConstructionActor.generated.h"

UCLASS()
class CHAPTER_09_API AOnConstructionActor : public AStaticMeshActor
{
  GENERATED_BODY()

public:
 AOnConstructionActor();

 virtual void OnConstruction(const FTransform& Transform) override;

 UPROPERTY(EditAnywhere)
 bool ShowStaticMesh;

};
```

1.  前往实现文件（`OnConstructionActor.cpp`）并实现类构造函数：

```cpp
#include "OnConstructionActor.h"
#include "ConstructorHelpers.h"

AOnConstructionActor::AOnConstructionActor()
{ 
 // Set this actor to call Tick() every frame. You can turn
    // this off to improve performance if you don't need it.
 PrimaryActorTick.bCanEverTick = true;

 auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>( 
 TEXT("StaticMesh'/Engine/BasicShapes/Cone.Cone'"));

 UStaticMeshComponent * SM = GetStaticMeshComponent();

 if (SM != nullptr)
 {
 if (MeshAsset.Object != nullptr)
 {
 SM->SetStaticMesh(MeshAsset.Object);
 SM->SetGenerateOverlapEvents(true);
 }
 SM->SetMobility(EComponentMobility::Movable);
 }

 // Default value of property
 ShowStaticMesh = true;
}
```

1.  实现`OnConstruction`：

```cpp
void AOnConstructionActor::OnConstruction(const FTransform& Transform) 
{ 
  GetStaticMeshComponent()->SetVisibility(ShowStaticMesh); 
} 
```

1.  编译你的代码并启动编辑器。

1.  将你的类的一个实例拖入游戏世界，并验证切换`ShowStaticMesh`布尔值是否会在编辑器视图中切换网格的可见性：

![](img/0e5c8d54-687b-45bc-adb7-19163d4c24b0.png)

1.  `OnConstruction`目前不会为放置在关卡中的 C++ actor 运行，如果它们被移动。

1.  要测试这一点，在你的`OnConstruction`函数中放置一个断点，然后移动你的 actor 在关卡中的位置。

要放置断点，将光标放在所需的行上，然后在 Visual Studio 中按*F9*。

1.  你会注意到函数没有被调用，但如果切换`ShowStaticMesh`布尔值，它就会被调用，导致你的断点被触发。

要了解原因，请查看`AActor::PostEditMove`函数的开始部分：

```cpp
void AActor::PostEditMove(bool bFinished)
{
    if ( ReregisterComponentsWhenModified() && !FLevelUtils::IsMovingLevel())
    {
        UBlueprint* Blueprint = Cast<UBlueprint>(GetClass()->ClassGeneratedBy);
        if (bFinished || bRunConstructionScriptOnDrag || (Blueprint && Blueprint->bRunConstructionScriptOnDrag))
        {
            FNavigationLockContext NavLock(GetWorld(), ENavigationLockReason::AllowUnregister);
            RerunConstructionScripts();
        }
    }

    // .... 
```

这里的第一行将当前对象的`UClass`转换为`UBlueprint`，并且只有当类是 Blueprint 时，才会再次运行构建脚本和`OnConstruction`。

# 它是如何工作的...

我们基于`StaticMeshActor`创建一个新的 Actor，以便通过静态网格访问视觉表示。

添加`UPROPERTY`以给我们一个可以更改的属性，这将触发`PostEditChangeProperty`事件。

`OnConstruction`是一个在 Actor 中定义的虚函数。

因此，我们在我们的类中重写了该函数。

在我们的类构造函数中，我们像往常一样初始化我们的网格，并将我们的`bool`属性的默认状态设置为与它所控制的组件的可见性相匹配。

在`OnConstruction`内部，演员使用任何所需的属性来重建自己。

对于这个简单的例子，我们将网格的可见性设置为与我们的`ShowStaticMesh`属性的值相匹配。

这也可以扩展到根据`ShowStaticMesh`变量的值改变其他值。

你会注意到我们并没有像之前的配方中用`PostEditChangeProperty`那样明确地对某个特定属性的改变进行过滤。

`OnConstruction`脚本在对象上每个发生改变的属性上都会完整运行。

它没有方法来测试刚刚编辑的是哪个属性，所以你需要谨慎地将计算密集型代码放置在其中。
