# Actors 和组件

在本章中，我们将介绍以下配方：

+   在 C++ 中创建自定义 Actor

+   使用 SpawnActor 实例化 Actor

+   创建 UFUNCTION

+   使用 Destroy 和计时器销毁 Actor

+   使用 SetLifeSpan 延迟销毁 Actor

+   通过组合实现 Actor 功能

+   使用 FObjectFinder 将资源加载到组件中

+   通过继承实现 Actor 功能

+   通过附加组件来创建层次结构

+   创建自定义 Actor 组件

+   创建自定义 Scene 组件

+   为 RPG 创建 InventoryComponent

+   创建 OrbitingMovement 组件

+   创建一个生成单位的建筑

# 简介

Actors 是在游戏世界中具有存在感的类。通过结合组件，Actors 获得其特殊功能。本章讨论创建自定义 Actors 和组件，它们的作用以及它们是如何协同工作的。

# 技术要求

本章需要使用 Unreal Engine 4，并使用 Visual Studio 2017 作为 IDE。有关如何安装这两款软件及其要求的信息，请参阅本书第一章“UE4 开发工具”。

# 在 C++ 中创建自定义 Actor

虽然 Unreal 默认安装中包含不同类型的 Actor，但在项目开发过程中，您可能需要创建自定义 Actor。这可能发生在您需要向现有类添加功能、组合不在默认子类中存在的组件，或向类添加额外的成员变量时。以下两个配方演示了如何使用组合或继承来定制 Actor。

# 准备工作

确保您已按照第一章中“UE4 开发工具”配方安装了 Visual Studio 和 Unreal 4。您还需要一个现有项目 - 如果没有，可以使用 Unreal 提供的向导创建一个新项目。

# 如何做...

1.  在 Unreal 编辑器中打开您的项目，并点击内容浏览器中的“添加新内容”按钮：

![图片](img/efd3d84e-10b0-4d49-8ea8-c53f92ce86f7.png)

1.  选择“新建 C++ 类...”：

![图片](img/1d0f7ab4-9130-465d-8be6-5a5e8bd5a31c.png)

1.  在打开的对话框中，从列表中选择 Actor：

![图片](img/4524f05e-09f3-4fb8-abfb-d0ad23a869e5.png)

1.  给您的 Actor 起一个名字，例如 `MyFirstActor`，然后点击 OK 以启动 Visual Studio：

按照惯例，`Actor` 子类的类名以字母 `A` 开头。当使用此类创建向导时，请确保不要在您的类名前加上 `A`，因为引擎会自动为您添加前缀。

![图片](img/ef31f6fe-8231-49b3-87e0-6c3edc4007a2.png)

1.  当 Visual Studio 加载时，您应该看到以下列表的类似内容：

```cpp
// MyFirstActor.h

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MyFirstActor.generated.h"

UCLASS()
class CHAPTER_04_API AMyFirstActor : public AActor
{
  GENERATED_BODY()

public: 
  // Sets default values for this actor's properties
  AMyFirstActor();

protected:
  // Called when the game starts or when spawned
  virtual void BeginPlay() override;

public: 
  // Called every frame
  virtual void Tick(float DeltaTime) override;

};

// MyFirstActor.cpp

#include "MyFirstActor.h"

// Sets default values
AMyFirstActor::AMyFirstActor()
{
   // Set this actor to call Tick() every frame. You can turn this off to improve performance if you don't need it.
  PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void AMyFirstActor::BeginPlay()
{
  Super::BeginPlay();

}

// Called every frame
void AMyFirstActor::Tick(float DeltaTime)
{
  Super::Tick(DeltaTime);

}
```

# 它是如何工作的...

随着时间的推移，你会熟悉标准代码，因此你将能够直接从 Visual Studio 创建新类，而无需使用 Unreal 向导。

在 `MyFirstActor.h` 文件中，我们有以下方面需要注意：

+   `#pragma once`: 这个预处理语句或 `pragma` 是 Unreal 实现包含保护代码的预期方法，这些代码防止 `include` 文件因多次引用而引起错误。

+   `#include "CoreMinimal.h"`: 此文件包含了许多常用类的定义，例如 `FString`、`TArray`、`Vector` 等，并且由于这个原因，它默认包含在创建的脚本文件中，尽管在没有它的情况下也可以编译。

+   `#include "GameFramework/Actor.h"`: 我们将要创建一个 `Actor` 子类，因此，自然地，我们需要包含我们继承的类的 `header` 文件，以便了解其内容。

+   `#include "MyFirstActor.generated.h"`: 所有 Actor 类都需要包含它们的 `generated.h` 文件。此文件由 **Unreal Header Tool** （**UHT**）根据在您的文件中检测到的宏自动创建。

+   `UCLASS()`: `UCLASS` 是这样一个宏，它允许我们指示一个类将被暴露给 Unreal 的反射系统。反射允许我们在运行时检查和迭代对象属性，以及管理我们对象的引用以进行垃圾回收。

+   `class CHAPTER_04_API AMyFirstActor : public AActor`: 这是我们的类的实际声明。`CHAPTER_04_API` 宏是由 UHT 创建的，并且对于确保我们的项目模块的类在 DLL 中正确导出是必要的。你还会注意到 `MyFirstActor` 和 `Actor` 都有前缀 `A` – 这是不从 `Actor` 继承的本地类在 Unreal 中所需的命名约定。

注意，在这种情况下，`Chapter_04` 是项目的名称，而你的项目可能具有不同的名称。

+   `GENERATED_BODY()`: `GENERATED_BODY` 是另一个 UHT 宏，它被扩展以包含底层 UE 类型系统所需的自动生成的函数。

在 `MyFirstActor.cpp` 文件中，我们有以下需要注意的方面：

+   `PrimaryActorTick.bCanEverTick = true;`: 在构造函数实现中，此行启用了此 `Actor` 的计时。所有 Actor 都有一个名为 `Tick` 的函数，这个布尔变量意味着 `Actor` 将在每个帧调用该函数一次，使 Actor 能够在每一帧执行必要的操作。作为一个性能优化，这默认是禁用的。

+   `BeginPlay/Tick`: 你还可以看到两个默认方法的实现，`BeginPlay` 和 `Tick`，分别是在对象被实例化后和每一帧存活时调用的。目前，这些方法仅通过 `Super::FunctionName` 调用父类的函数版本。

# 使用 SpawnActor 实例化 Actor

对于这个配方，你需要有一个准备就绪的`Actor`子类以进行实例化。你可以使用内置类，如`StaticMeshActor`，但使用你在上一个配方中创建的自定义`Actor`进行练习会有所帮助。

# 如何实现...

1.  创建一个新的 C++类，就像之前的配方一样。这次，选择游戏模式基类作为你的基类：

![图片](img/f59f3973-6781-406c-be4b-1525e29de708.png)

1.  点击下一步后，为新类命名，例如`UE4CookbookGameModeBase`。

1.  在你的新`GameModeBase`类的`.h`文件中声明一个函数覆盖：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "UECookbookGameModeBase.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_04_API AUECookbookGameModeBase : public AGameModeBase
{
  GENERATED_BODY()

public:
 virtual void BeginPlay() override; 
};
```

1.  在`.cpp`文件中实现`BeginPlay`函数：

```cpp
#include "UECookbookGameModeBase.h"
#include "MyFirstActor.h" // AMyFirstActor

void AUECookbookGameModeBase::BeginPlay()
{
  // Call the parent class version of this function
  Super::BeginPlay();

  // Displays a red message on the screen for 10 seconds
  GEngine->AddOnScreenDebugMessage(-1, 10, FColor::Red, 
                                   TEXT("Actor Spawning")); 

  // Spawn an instance of the AMyFirstActor class at the
  //default location.
  FTransform SpawnLocation;
  GetWorld()->SpawnActor<AMyFirstActor>
                             (AMyFirstActor::StaticClass(), 
                              SpawnLocation);
}
```

1.  通过 Visual Studio 或通过单击 Unreal 编辑器中的编译按钮来编译你的代码：

![图片](img/91a0b957-21e2-4d2f-8f52-81d4bacdbee3.png)

1.  通过单击设置工具栏图标打开当前关卡的“世界设置”面板，然后从下拉菜单中选择“世界设置”：

![图片](img/652d3673-3870-48fc-b748-08d665536339.png)

1.  在游戏模式覆盖部分，将游戏模式更改为你刚刚创建的`GameMode`子类：

![图片](img/e69cfdec-f16a-4902-a53a-e379c485f4e0.png)

设置游戏模式覆盖属性

1.  启动关卡并验证`GameMode`是否在世界上创建了你`Actor`的副本，可以通过查看世界大纲面板来完成。你可以通过查看屏幕上显示的`Actor`生成文本来验证`BeginPlay`函数是否正在运行。如果没有生成，请确保在世界原点没有阻碍`Actor`生成的障碍物。你可以在世界大纲面板顶部的搜索栏中输入以搜索世界中的对象列表。这将过滤显示的实体：

![图片](img/fbbdf8b8-9165-43d8-bf27-706fa64a23af.png)

# 它是如何工作的...

`GameMode`是 Unreal 游戏框架的一部分的特殊类型的 actor。你的地图的`GameMode`在游戏开始时由引擎自动实例化。

通过将一些代码放入我们自定义的`GameMode`的`BeginPlay`方法中，我们可以在游戏开始时自动运行它。

在`BeginPlay`内部，我们创建了一个`FTransform`，它将被`SpawnActor`函数使用。默认情况下，`FTransform`被构造为具有零旋转和位于原点的位置。

我们首先使用`GetWorld`获取当前级别的`UWorld`实例，然后调用其`SpawnActor`函数。我们传入之前创建的`FTransform`，以指定对象应在其位置创建，即原点。

# 创建一个 UFUNCTION

`UFUNCTION()`非常有用，因为它是一个可以从你的 C++客户端代码以及蓝图图中调用的 C++函数。任何 C++函数都可以标记为`UFUNCTION()`。

# 如何实现...

1.  构造一个`UClass`类或派生类（例如`AActor`），它有一个你希望暴露给蓝图图的成员函数。用`UFUNCTION( BlueprintCallable, Category=SomeCategory)`装饰该成员函数，以便可以从蓝图图中调用它。

1.  例如，让我们创建一个名为 `Warrior` 的 `Actor` 类，并为其使用以下脚本：

```cpp
//Warrior.h
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Warrior.generated.h"

UCLASS()
class CHAPTER_04_API AWarrior : public AActor
{
  GENERATED_BODY()

public: 
  // Sets default values for this actor's properties
  AWarrior();

 // Name of the Actor
 UPROPERTY(EditAnywhere, BlueprintReadWrite, 
 Category = Properties) 
 FString Name; 

 // Returns message containing the Name property
 UFUNCTION(BlueprintCallable, Category = Properties) 
 FString ToString(); 

protected:
  // Called when the game starts or when spawned
  virtual void BeginPlay() override;

public: 
  // Called every frame
  virtual void Tick(float DeltaTime) override;

};

// Warrior.cpp

#include "Warrior.h"

// Sets default values
AWarrior::AWarrior()
{
   // Set this actor to call Tick() every frame. You can turn this off to improve performance if you don't need it.
  PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void AWarrior::BeginPlay()
{
  Super::BeginPlay();

}

// Called every frame
void AWarrior::Tick(float DeltaTime)
{
  Super::Tick(DeltaTime);

}

FString AWarrior::ToString() 
{ 
 return FString::Printf(TEXT("An instance of AWarrior: %s"), *Name); 
} 
```

1.  通过内容浏览器打开 `C++ Classes\Chapter_04` 文件夹，创建你的 `Warrior` 类的实例。一旦到达那里，将 `Warrior` 图标拖放到你的游戏世界中，然后释放鼠标。

1.  你应该在“世界大纲”选项卡中看到该项目。通过选择新添加的对象，你应该能够看到我们添加的“名称”属性。在这里输入一个值，例如 `John`：

![](img/4597410e-9600-4936-95cd-1ed87e6ec8ba.png)

1.  从蓝图（蓝图 | 打开水平蓝图）获取你的 `Warrior` 对象的引用。一种方法是将对象从世界大纲中拖放到水平蓝图的事件图中，然后释放。

1.  点击并按住 Warrior1 节点右侧的蓝色圆圈手柄，将其稍微向右拖动。一旦释放鼠标，你会看到你可以选择的一系列操作。

1.  通过单击你的 `Warrior` 实例来调用该 `Warrior` 实例的 `ToString()` 函数。然后，在蓝图图中输入 `ToString`。它应该看起来如下：

![](img/c709de65-ff1b-4ce7-af93-4e71d6a6eaf3.png)

# 它是如何工作的...

`UFUNCTION()` 实际上是一个 C++ 函数，但它带有额外的元数据，这使得它可以通过蓝图访问。这可以非常实用，允许你的设计师访问你编写的函数。

# 使用 `Destroy` 和定时器销毁 Actor

此菜谱将重用上一个菜谱中的 `GameMode`，即使用 `SpawnActor` 实例化 Actor，因此你应该首先完成该菜谱。

# 如何做到这一点...

1.  对 `GameMode` 声明进行以下更改：

```cpp
UCLASS()
class CHAPTER_04_API AUECookbookGameModeBase : public AGameModeBase
{
  GENERATED_BODY()

public:
  virtual void BeginPlay() override; 

 UPROPERTY() 
 AMyFirstActor* SpawnedActor; 

 UFUNCTION() 
 void DestroyActorFunction(); 
};
```

1.  在实现文件的包含语句中添加 `#include "MyFirstActor.h"`。记住，我们需要将其放置在 `.generated` 文件之上：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "MyFirstActor.h"
#include "UECookbookGameModeBase.generated.h"
```

1.  将 `SpawnActor` 的结果分配给新的 `SpawnedActor` 变量：

```cpp
#include "UECookbookGameModeBase.h"
#include "MyFirstActor.h" // AMyFirstActor

void AUECookbookGameModeBase::BeginPlay()
{
  // Call the parent class version of this function
  Super::BeginPlay();

  // Displays a red message on the screen for 10 seconds
  GEngine->AddOnScreenDebugMessage(-1, 10, FColor::Red, 
                   TEXT("Actor Spawning")); 

  // Spawn an instance of the AMyFirstActor class at the
  // default location.
  FTransform SpawnLocation;
 SpawnedActor = GetWorld()->SpawnActor<AMyFirstActor>( 
                             AMyFirstActor::StaticClass(), 
                             SpawnLocation);
}
```

1.  将以下内容添加到 `BeginPlay` 函数的末尾：

```cpp
FTimerHandle Timer; 
GetWorldTimerManager().SetTimer(Timer, this, 
                   &AUECookbookGameModeBase::DestroyActorFunction, 10);
```

1.  最后，实现 `DestroyActorFunction`:

```cpp
void AUECookbookGameModeBase::DestroyActorFunction() 
{ 
  if (SpawnedActor != nullptr) 
  { 
    // Displays a red message on the screen for 10 seconds
    GEngine->AddOnScreenDebugMessage(-1, 10, FColor::Red, 
                                     TEXT("Actor Destroyed")); 
    SpawnedActor->Destroy(); 
  } 
} 
```

1.  加载你在上一个菜谱中创建的水平，并将游戏模式设置为你的自定义类。

1.  播放你的水平，并使用世界大纲验证你的 `SpawnedActor` 在 10 秒后被删除：

![](img/c1492b0f-321d-4cb9-8a0b-1f0bfee5bc05.png)

# 它是如何工作的...

我们声明一个 `UPROPERTY` 来存储我们生成的 `Actor` 实例，并定义一个自定义函数，以便我们可以定时调用 `Destroy()`:

```cpp
UPROPERTY() 
AMyFirstActor* SpawnedActor; 
UFUNCTION() 
void DestroyActorFunction(); 
```

在 `BeginPlay` 中，我们将生成的 `Actor` 分配给我们的新 `UPROPERTY`:

```cpp
SpawnedActor = GetWorld()->SpawnActor<AMyFirstActor> 
 (AMyFirstActor::StaticClass(), SpawnLocation);
```

我们然后声明一个 `TimerHandle` 对象，并将其传递给 `GetWorldTimerManager::SetTimer`。`SetTimer` 在 10 秒后调用指向此指针的对象的 `DestroyActorFunction`。`SetTimer` 返回一个对象——一个句柄——允许我们在必要时取消定时器。`SetTimer` 函数接受 `TimerHandle` 对象作为引用参数，因此我们提前声明它，以便我们可以正确地将它传递到函数中，即使我们不会再次使用它：

```cpp
FTimerHandle Timer; 
GetWorldTimerManager().SetTimer(Timer, this, 
 &AUE4CookbookGameMode::DestroyActorFunction, 10);
```

`DestroyActorFunction`检查我们是否有对已生成`Actor`的有效引用：

```cpp
void AUE4CookbookGameMode::DestroyActorFunction() 
{ 
  if (SpawnedActor != nullptr) 
  {
     // Then we know that SpawnedActor is valid
  }
} 
```

如果我们有，它将在实例上调用`Destroy`，使其被销毁，并最终被垃圾回收：

```cpp
SpawnedActor->Destroy();
```

# 使用`SetLifeSpan`延迟销毁`Actor`

让我们看看如何销毁一个`Actor`。

# 如何操作...

1.  如果你还没有创建，请使用向导创建一个新的 C++类。选择`Actor`作为你的基类。在我们的例子中，我将重用本章先前创建的`AWarrior`类。

1.  在`Actor`的实现中，将以下代码添加到`BeginPlay`函数中：

```cpp
// Called when the game starts or when spawned
void AWarrior::BeginPlay()
{
  Super::BeginPlay();

 // Will destroy this object in 10 seconds
 SetLifeSpan(10); 

}
```

1.  将你的自定义`Actor`的一个副本拖到编辑器中的视图中。

1.  播放你的关卡，查看大纲以验证你的`Actor`实例在 10 秒后消失，因为它已经销毁了自己。

# 它是如何工作的...

我们将代码插入到`BeginPlay`函数中，以便在游戏开始时执行。

`SetLifeSpan`函数允许我们指定秒数，在此之后`Actor`将调用其自己的`Destroy()`方法。

# 通过组合实现`Actor`功能

没有组件的自定义`Actor`没有位置，也不能附加到其他`Actor`上。没有根组件，`Actor`没有基础变换，因此它没有位置。因此，大多数`Actor`至少需要一个组件才有用。

我们可以通过向`Actor`添加多个组件来通过组合创建自定义`Actor`，其中每个组件都提供所需的一些功能。

# 准备工作

这个配方将使用我们在*在 C++中创建自定义`Actor`*配方中创建的`Actor`类。

# 如何操作...

1.  通过在`MyFirstActor.h`文件的`public`部分进行以下更改，在 C++中向你的自定义类添加一个新成员：

```cpp
UPROPERTY() 
UStaticMeshComponent* Mesh; 
```

1.  在`MyFirstActor.cpp`文件的构造函数中添加以下行：

```cpp
// Sets default values
AMyFirstActor::AMyFirstActor()
{
   // Set this actor to call Tick() every frame. You can turn
   // this off to improve performance if you don't need it.
  PrimaryActorTick.bCanEverTick = true;

  // Creates a StaticMeshComponent on this object and assigns
  // Mesh to it
  Mesh = CreateDefaultSubobject<UStaticMeshComponent>
         ("BaseMeshComponent");
}
```

1.  完成后，保存两个文件，并通过编辑器中的编译按钮或使用 Visual Studio 构建项目来编译它们。

1.  一旦编译了这段代码，将你的类的一个实例从内容浏览器拖到游戏环境中。在这里，你将能够验证它现在具有变换和其他属性，例如来自我们添加的`StaticMeshComponent`的静态网格：

![图片](img/ee1cb6d8-1dc7-4bb1-bdd0-c0e6bebf0324.png)

选择实例化的`Actor`

你可以使用详情标签页顶部的搜索栏搜索特定组件，例如静态网格组件。

# 它是如何工作的...

我们添加到类声明的`UPROPERTY 宏`是一个指针，用于持有我们用作`Actor`子对象的组件：

```cpp
UPROPERTY() 
UStaticMeshComponent* Mesh; 
```

使用`UPROPERTY()`宏确保在指针中声明的对象被视为引用，并且不会被垃圾回收（即删除），从而不会留下悬空指针。

我们使用的是静态网格组件，但任何`Actor`组件的子类都可以工作。注意，根据 Epic 的风格指南，星号与变量类型相连。

在构造函数中，我们通过使用`template`函数`template<class TReturnType> TReturnType* CreateDefaultSubobject(FName SubobjectName, bool bTransient = false)`初始化指针到一个已知的有效值。

此函数负责调用引擎代码以适当地初始化组件，并返回新构造的对象的指针，以便我们可以为我们的组件指针提供一个默认值。这很重要，因为它确保指针始终具有有效的值，从而最小化解引用未初始化内存的风险。

该函数基于要创建的对象类型进行模板化，但也接受两个参数——第一个是子对象的名称，理想情况下应该是可读的，第二个是对象是否应该是瞬时的（即不与父对象一起保存）。

# 参见

+   以下配方展示了如何在你的静态网格组件中引用网格资源，以便它可以显示，而无需用户在编辑器中指定网格。

# 使用 FObjectFinder 将资源加载到组件中

在前面的配方中，我们创建了一个静态网格组件，但我们没有尝试为组件加载网格以显示。虽然可以在编辑器中这样做，但有时在 C++中指定默认值是有帮助的。

# 准备工作

完成前面的配方，以便你有一个带有静态网格组件的自定义`Actor`子类。

在你的内容浏览器中，点击查看选项按钮并选择显示引擎内容：

![图片](img/13ca9ea4-400f-4449-9884-8cb62b9519cf.png)

点击显示/隐藏源面板按钮或点击文件夹图标以查看内容浏览器中的文件夹。从那里，浏览到“Engine Content”，然后到“BasicShapes”，以查看我们将在此配方中使用的立方体：

![图片](img/e10cc256-1f8a-40fa-98b8-40c5400f401a.png)

# 如何做到...

1.  将以下代码添加到你的类构造函数中：

```cpp
// Sets default values
AMyFirstActor::AMyFirstActor()
{
   // Set this actor to call Tick() every frame. You can turn this off 
  // to improve performance if you don't need it.
  PrimaryActorTick.bCanEverTick = true;

  // Creates a StaticMeshComponent on this object and assigns Mesh 
  // to it
  Mesh = CreateDefaultSubobject<UStaticMeshComponent>
       ("BaseMeshComponent");

 auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>
                   (TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));  // Check if the MeshAsset is valid before setting it
 if (MeshAsset.Object != nullptr)
 {
 Mesh->SetStaticMesh(MeshAsset.Object); 
 }

} 
```

1.  在编辑器中编译并验证，以确保你的类实例现在有一个网格作为其视觉表示：

![图片](img/4b01adda-3262-411f-a5ef-9600ab836b95.png)

如果在更改之前将演员放置在世界中，网格可能只有在你尝试在视口中移动演员之后才会出现。由于某种原因，它并不总是自动更新。

# 它是如何工作的...

我们创建了一个`FObjectFinder`类的实例，将我们要加载的资源类型作为模板参数传递。

`FObjectFinder`是一个类模板，帮助我们加载资源。当我们构造它时，我们传递一个包含要加载的资源路径的字符串。

字符串的格式为`"{ObjectType}'/Path/To/Asset.Asset'"`。注意字符串中使用了单引号。

要获取编辑器中已存在的资产字符串，你可以在内容浏览器中右键点击该资产并选择复制引用。这将给你一个字符串，以便你可以将其粘贴到你的代码中：

![](img/a080553c-e009-4413-91f1-1347e6efe2b8.png)

我们使用 C++11 中的 `auto` 关键字来避免在声明中输入整个对象类型；编译器为我们推断它。如果没有 `auto`，我们则必须使用以下代码：

```cpp
ConstructorHelpers::FObjectFinder<UStaticMesh> MeshAsset = 
 ConstructorHelpers::FObjectFinder<UStaticMesh>(TEXT("Static
 Mesh'/Engine/BasicShapes/Cube.Cube'"));
```

`FObjectFinder` 类有一个名为 `Object` 的属性，它将指向所需的资产，或者在资产找不到时为 `NULL`。

这意味着我们可以将其与 `nullptr` 进行比较，如果不是空，则可以使用 `SetStaticMesh` 将其分配给 `Mesh`。

# 通过继承实现 Actor 功能

继承是实现自定义 `Actor` 的第二种方式。这通常是为了创建一个新的子类，它向现有的 `Actor` 类添加成员变量、函数或组件。在这个配方中，我们将向自定义的 `GameState` 子类添加一个变量。

# 如何做到这一点...

1.  在 Unreal 编辑器中，点击内容浏览器中的“添加新内容”。然后，在“新建 C++ 类...”中，选择“游戏状态基”作为基类，并为你的新类命名（我将使用默认的 `MyGameStateBase` 通过创建 `AMyGameStateBase` 类）：

![](img/a9707562-42a3-4c40-b578-031460d86dc0.png)

`GameState` 类负责所有玩家共享的信息，并且特定于游戏模式，但不特定于任何单个玩家。假设我们正在开发一个合作游戏，所有玩家都在为总分数共同努力。将这些信息包含在这个类中是有意义的。

1.  将以下代码添加到新类头文件中：

```cpp
UCLASS()
class CHAPTER_04_API AMyGameStateBase : public AGameStateBase
{
    GENERATED_BODY()

public:
 // Constructor to initialize CurrentScore
 AMyGameStateBase(); 

 // Will set the CurrentScore variable
 UFUNCTION() 
 void SetScore(int32 NewScore); 

 // Getter
 UFUNCTION() 
 int32 GetScore(); 

private: 
 UPROPERTY() 
 int32 CurrentScore; 

};
```

1.  将以下代码添加到 `.cpp` 文件中：

```cpp
#include "MyGameStateBase.h"

AMyGameStateBase::AMyGameStateBase()
{
 CurrentScore = 0;
}

int32 AMyGameStateBase::GetScore()
{
 return CurrentScore;
}

void AMyGameStateBase::SetScore(int32 NewScore)
{
 CurrentScore = NewScore;
}
```

1.  确保你的代码看起来像以下列表，并使用 Unreal 编辑器中的编译按钮进行编译：

```cpp
//MyGameStateBase.h 
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameStateBase.h"
#include "MyGameStateBase.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_04_API AMyGameStateBase : public AGameStateBase
{
    GENERATED_BODY()

public:
    // Constructor to initialize CurrentScore
    AMyGameStateBase(); 

    // Will set the CurrentScore variable
    UFUNCTION() 
    void SetScore(int32 NewScore); 

    // Getter
    UFUNCTION() 
    int32 GetScore(); 

private: 
    UPROPERTY() 
    int32 CurrentScore; 

};

//MyGameState.cpp 
#include "MyGameStateBase.h"

AMyGameStateBase::AMyGameStateBase()
{
  CurrentScore = 0;
}

int32 AMyGameStateBase::GetScore()
{
  return CurrentScore;
}

void AMyGameStateBase::SetScore(int32 NewScore)
{
  CurrentScore = NewScore;
}

```

# 它是如何工作的...

首先，我们添加默认构造函数的声明：

```cpp
AMyGameState(); 
```

这允许我们在对象初始化时将我们的新成员变量设置为安全的默认值 `0`：

```cpp
AMyGameState::AMyGameState() 
{ 
  CurrentScore = 0; 
} 
```

我们在声明新变量时使用 `int32` 类型，以确保在 Unreal Engine 支持的各种编译器之间具有可移植性。这个变量将负责在运行时存储当前游戏分数。

如果你希望值只能是正数，你可以使用 `uint32` 类型，它仅用于无符号数。

和往常一样，我们将使用 `UPROPERTY` 标记我们的变量，以便它能够适当地进行垃圾回收。这个变量被标记为 `private`，这样唯一改变值的方法就是通过我们的函数：

```cpp
UPROPERTY() 
int32 CurrentScore; 
```

`GetScore` 函数将检索当前分数并将其返回给调用者。它被实现为一个简单的访问器，它简单地返回底层的成员变量。

第二个函数`SetScore`设置成员变量的值，允许外部对象请求更改分数。将此请求作为一个函数确保`GameState`可以审查此类请求，并且只有在它们有效时才允许它们，以防止作弊。此类检查的细节超出了本配方的范围，但`SetScore`函数是进行此类检查的适当位置。

Cedric 'eXi' Neukirchen 在这里创建了一个关于这个主题的优秀且非常广泛的文档：[`cedric-neukirchen.net/Downloads/Compendium/UE4_Network_Compendium_by_Cedric_eXi_Neukirchen_BW.pdf`](http://cedric-neukirchen.net/Downloads/Compendium/UE4_Network_Compendium_by_Cedric_eXi_Neukirchen_BW.pdf)。

我们的得分函数使用`UFUNCTION`宏声明，出于多个原因。首先，`UFUNCTION`加上一些额外的代码可以被蓝图调用或覆盖。其次，`UFUNCTION`可以被标记为`exec`，这意味着它们可以在游戏会话期间由玩家或开发者作为控制台命令运行，这有助于调试。

# 参见

+   第九章，*整合 C++和 Unreal 编辑器：第二部分*，有一个名为*创建新的控制台命令*的配方，你可以参考它以获取有关`exec`和控制台命令功能的更多信息。

# 通过附加组件来创建层次结构

当从组件创建自定义 actor 时，考虑**附加**的概念非常重要。附加组件创建了一个关系，即应用于父组件的变换也会影响附加到它的组件。

# 如何做...

1.  使用编辑器创建一个从`Actor`类派生的新类，并将其命名为`HierarchyActor`。

1.  在头文件（`HierarchyActor.h`）中添加以下属性到你的新类中：

```cpp
UCLASS()
class CHAPTER_04_API AHierarchyActor : public AActor
{
    GENERATED_BODY()

public: 
    // Sets default values for this actor's properties
    AHierarchyActor();

 UPROPERTY(VisibleAnywhere) 
 USceneComponent* Root; 

 UPROPERTY(VisibleAnywhere) 
 USceneComponent* ChildSceneComponent; 

 UPROPERTY(VisibleAnywhere) 
 UStaticMeshComponent* BoxOne; 

 UPROPERTY(VisibleAnywhere) 
 UStaticMeshComponent* BoxTwo; 

protected:
    // Called when the game starts or when spawned
    virtual void BeginPlay() override;

public: 
    // Called every frame
    virtual void Tick(float DeltaTime) override;

};
```

1.  将以下代码添加到类构造函数中：

```cpp
// Sets default values
AHierarchyActor::AHierarchyActor()
{
    // Set this actor to call Tick() every frame. You can turn this
    // off to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = true;

 // Create four subobjects
 Root = CreateDefaultSubobject<USceneComponent>("Root"); 
 ChildSceneComponent = CreateDefaultSubobject<USceneComponent>
                          ("ChildSceneComponent"); 
 BoxOne = CreateDefaultSubobject<UStaticMeshComponent>("BoxOne"); 
 BoxTwo = CreateDefaultSubobject<UStaticMeshComponent>("BoxTwo");

 // Get a reference to the cube mesh
 auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>
                   (TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));

 // Give both boxes a mesh
 if (MeshAsset.Object != nullptr)
 {
 BoxOne->SetStaticMesh(MeshAsset.Object);
 BoxTwo->SetStaticMesh(MeshAsset.Object);
 }

 RootComponent = Root;

 // Set up the object's hierarchy
 BoxOne->AttachTo(Root);
 BoxTwo->AttachTo(ChildSceneComponent);

 ChildSceneComponent->AttachTo(Root);

 // Offset and scale the child from the root
 ChildSceneComponent->SetRelativeTransform(
 FTransform(FRotator(0, 0, 0), 
 FVector(250, 0, 0), 
 FVector(0.1f))
 );

}
```

1.  编译并启动编辑器。将`HierarchyActor`的一个副本拖入场景：

![图片](img/f978bc11-df51-42b0-b644-c49f5e52f0fb.png)

1.  验证`Actor`是否具有层次结构中的组件，并且第二个框的大小更小：

![图片](img/0b9ec562-6754-455d-b342-43cc0e478f2b.png)

如果你在详细信息选项卡下没有看到根（继承）部分，你可以将鼠标拖到搜索栏上方来扩展它。

# 它是如何工作的...

如同往常，我们为我们的 actor 创建了一些带标签的`UPROPERTY`组件。在这种情况下，我们向标签添加了一个额外的参数，称为`VisibleAnywhere`，这样我们就可以在详细信息选项卡中看到我们的变量。我们创建了两个场景组件和两个静态网格组件。

在构造函数中，我们为每个组件创建了默认的子对象，就像往常一样。

然后我们加载静态网格，如果加载成功，将其分配给两个静态网格组件，以便它们有视觉表示。

我们然后在我们的`Actor`中通过附加组件来构建一个层次结构。

我们将第一个场景组件设置为`Actor`的根组件。此组件将确定应用于层次结构中所有其他组件的变换。

然后我们将第一个盒子附加到我们的新根组件上，并将第二个场景组件作为第一个组件的父组件。

我们将第二个盒子附加到子场景组件上，以演示如何更改该场景组件的变换会影响其子组件，但不会影响对象中的其他组件。

最后，我们设置该场景组件的相对变换，使其从原点移动一定距离，并且缩放为原来的十分之一。

这意味着在编辑器中，你可以看到`BoxTwo`组件已经继承了其父组件`ChildSceneComponent`的平移和缩放。

# 创建自定义演员组件

演员（Actor）组件是实现应在演员之间共享的常见功能的一种简单方式。演员组件不会被渲染，但仍然可以执行诸如订阅事件或与其所在演员的其他组件通信等操作。

# 如何实现...

1.  使用编辑器向导创建一个名为`RandomMovementComponent`的`ActorComponent`：

![图片](img/7e5e56d7-96ca-45c1-8e46-241ffee79da7.png)

1.  在公共部分类头文件中添加以下`UPROPERTY`：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "RandomMovementComponent.generated.h"

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class CHAPTER_04_API URandomMovementComponent : public UActorComponent
{
    GENERATED_BODY()

public: 
    // Sets default values for this component's properties
    URandomMovementComponent();

 UPROPERTY()
 float MovementRadius;

protected:
    // Called when the game starts
    virtual void BeginPlay() override;

public: 
    // Called every frame
    virtual void TickComponent(float DeltaTime, ELevelTick 
    TickType, FActorComponentTickFunction* ThisTickFunction) 
    override;

};

```

1.  将以下代码添加到构造函数的实现中：

```cpp
// Sets default values for this component's properties
URandomMovementComponent::URandomMovementComponent()
{
    // Set this component to be initialized when the game
    // starts, and to be ticked every frame. You can turn
    // these features
    // off to improve performance if you don't need them.
    PrimaryComponentTick.bCanEverTick = true;

    // ...
    MovementRadius = 5;
}
```

1.  最后，将以下代码添加到`TickComponent()`的实现中：

```cpp
// Called every frame
void URandomMovementComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType,
    ThisTickFunction);

    // ...
 AActor* Parent = GetOwner(); 

 if (Parent) 
 { 
 // Find a new position for the object to go to
 auto NewPos = Parent->GetActorLocation() + 
 FVector
 ( 
 FMath::FRandRange(-1, 1) * 
                      MovementRadius, 
 FMath::FRandRange(-1, 1) *
                      MovementRadius, 
 FMath::FRandRange(-1, 1) *
                      MovementRadius
 );
 // Update the object's position
 Parent->SetActorLocation( NewPos ); 
 } 
}
```

1.  编译你的项目。在编辑器中，创建一个空的`Actor`，并将你的随机移动组件添加到其中。例如，从模式选项卡转到基本选项，并将一个立方体拖放到你的关卡中。

1.  之后，确保在详情选项卡中将变换组件的移动性属性设置为可移动：

![图片](img/637d0947-9dfb-451e-998a-729739f42501.png)

1.  然后，选择对象后，在详情面板中单击添加组件，并选择随机移动：

![图片](img/cd367a36-8adb-4a14-9f4d-a247f43044e8.png)

1.  播放你的关卡，并观察演员在每次调用`TickComponent`函数时其位置改变时的随机移动：

![图片](img/7befead9-e431-42e6-9f29-882bb5a0e06b.png)

# 它是如何工作的...

首先，我们在组件声明中使用的`UCLASS`宏中添加了一些指定项。将`BlueprintSpawnableComponent`添加到类的元值中意味着该组件的实例可以被添加到编辑器中的蓝图类中。`ClassGroup`指定项允许我们在类列表中指示我们的组件属于哪个类别：

```cpp
UCLASS( ClassGroup=(Custom), 
 meta=(BlueprintSpawnableComponent) )
```

将`MovementRadius`作为属性添加到新组件中，允许我们指定组件在单个帧内可以随意游荡的距离：

```cpp
UPROPERTY() 
float MovementRadius; 
```

在构造函数中，我们将此属性初始化为安全的默认值：

```cpp
MovementRadius = 5; 
```

`TickComponent`是一个由引擎每帧调用的函数，就像`Tick`对于 actor 一样。在其实现中，我们检索组件拥有者的当前位置，即包含我们的组件的`Actor`，并在世界空间中生成一个偏移量：

```cpp
    AActor* Parent = GetOwner(); 

    if (Parent) 
    { 
        // Find a new position for the object to go to
        auto NewPos = Parent->GetActorLocation() + 
                      FVector
                      ( 
                      FMath::FRandRange(-1, 1) * MovementRadius, 
                      FMath::FRandRange(-1, 1) * MovementRadius, 
                      FMath::FRandRange(-1, 1) * MovementRadius
                      );
        // Update the object's position
        Parent->SetActorLocation( NewPos ); 
    } 
```

我们将随机偏移量添加到当前位置以确定新位置，并将拥有者 actor 移动到那里。这导致 actor 的位置在帧与帧之间随机变化并舞动。

# 创建自定义场景组件

`Scene`组件是具有变换的`Actor`组件的子类，即相对位置、旋转和缩放。就像`Actor`组件一样，`Scene`组件本身不进行渲染，但可以使用其变换进行各种操作，例如在`Actor`的固定偏移量处生成其他对象。

# 如何做到这一点...

1.  创建一个名为`ActorSpawnerComponent`的自定义`SceneComponent`：

![](img/83bd074a-030f-46c2-baa9-9a4b1ccfdde6.png)

1.  对头文件进行以下更改：

```cpp
#include "CoreMinimal.h"
#include "Components/SceneComponent.h"
#include "ActorSpawnerComponent.generated.h"

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class CHAPTER_04_API UActorSpawnerComponent : public USceneComponent
{
  GENERATED_BODY()

public: 
  // Sets default values for this component's properties
  UActorSpawnerComponent();

 // Will spawn actor when called
 UFUNCTION(BlueprintCallable, Category=Cookbook)
 void Spawn();

 UPROPERTY(EditAnywhere)
 TSubclassOf<AActor> ActorToSpawn;

protected:
  // Called when the game starts
  virtual void BeginPlay() override;

public: 
  // Called every frame
  virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

};
```

1.  将以下函数实现添加到`.cpp`文件中：

```cpp
void UActorSpawnerComponent::Spawn()
{
    UWorld* TheWorld = GetWorld();
    if (TheWorld != nullptr)
    {
        FTransform ComponentTransform(this->GetComponentTransform());
        TheWorld->SpawnActor(ActorToSpawn,&ComponentTransform);
    }
}
```

1.  编译并打开你的项目。将一个空`Actor`拖入场景，并将你的`ActorSpawnerComponent`添加到它上面。在详细信息面板中选择你的新组件，并分配一个值给`ActorToSpawn`。

现在，每当在组件的实例上调用`Spawn()`时，它将实例化`ActorToSpawn`中指定的`Actor`类的副本。

# 它是如何工作的...

我们创建`Spawn UFUNCTION`和一个名为`ActorToSpawn`的变量。`ActorToSpawn``UPROPERTY`是`TSubclassOf< >`类型，这是一种模板类型，允许我们将指针限制为基类或其子类。这也意味着在编辑器中，我们将获得一个预先过滤的类列表以供选择，防止我们意外分配无效的值：

![](img/6b7004ee-a06d-4099-b334-bb0d9433f0aa.png)

在`Spawn`函数的实现内部，我们获取对世界的访问权限。从这里，我们检查其有效性。

`SpawnActor`需要一个`FTransform*`来指定新 actor 的生成位置，因此我们创建一个新的栈变量来包含当前组件变换的副本。

如果`TheWorld`有效，我们请求它生成指定的`ActorToSpawn`子类的实例，传入我们刚刚创建的`FTransform`的地址，它现在包含新 actor 期望的位置。

# 相关内容

+   第八章，*整合 C++和虚幻编辑器*，包含了对如何使事物蓝图可访问的更详细的研究

# 为 RPG 创建一个库存组件

一个`InventoryComponent`使包含它的`Actor`能够将其`InventoryActors`存储在它的库存中，并将它们放回游戏世界。

# 准备工作

在继续此配方之前，请确保你已经遵循了第六章，*输入和碰撞*中的*轴映射 - 键盘、鼠标和游戏手柄方向输入用于 FPS 角色*配方，因为它展示了如何创建一个简单的角色。

此外，本章中的*使用 SpawnActor 实例化 Actor*配方展示了如何创建自定义`GameMode`。

# 如何做到这一点...

1.  使用引擎创建一个名为`InventoryComponent`的`ActorComponent`子类：

![图片](img/6a88aa60-de6e-48da-9a0b-df939365bc5f.png)

1.  在`InventoryComponent.h`文件内部，添加以下代码：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "InventoryComponent.generated.h"

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class CHAPTER_04_API UInventoryComponent : public UActorComponent
{
    GENERATED_BODY()

public: 
    // Sets default values for this component's properties
    UInventoryComponent();

 UPROPERTY()
 TArray<AInventoryActor*> CurrentInventory;

 UFUNCTION()
 int32 AddToInventory(AInventoryActor* ActorToAdd);

 UFUNCTION()
 void RemoveFromInventory(AInventoryActor* ActorToRemove);

protected:
    // Called when the game starts
    virtual void BeginPlay() override;

public: 
    // Called every frame
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

};
```

1.  将以下函数实现添加到源文件中：

```cpp
int32 UInventoryComponent::AddToInventory(AInventoryActor* ActorToAdd)
{
    return CurrentInventory.Add(ActorToAdd);
}

void UInventoryComponent::RemoveFromInventory(AInventoryActor* ActorToRemove)
{
    CurrentInventory.Remove(ActorToRemove);
}
```

1.  接下来，创建一个新的`StaticMeshActor`子类，名为`InventoryActor`。记得检查显示所有类以查看`StaticMeshActor`类：

![图片](img/253b7cd0-e29a-4bc8-88b5-15fcd7b599ac.png)

1.  现在我们有了文件，前往`InventoryComponent.h`文件并添加以下包含：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "InventoryActor.h"
#include "InventoryComponent.generated.h"

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class CHAPTER_04_API UInventoryComponent : public UActorComponent
```

1.  返回到`InventoryActor.h`文件并添加以下到其声明中：

```cpp
UCLASS()
class CHAPTER_04_API AInventoryActor : public AStaticMeshActor
{
  GENERATED_BODY()

public:
 virtual void PickUp();
 virtual void PutDown(FTransform TargetLocation);

};
```

1.  在实现文件中实现新函数：

```cpp
void AInventoryActor::PickUp() 
{ 
  SetActorTickEnabled(false); 
  SetActorHiddenInGame(true); 
  SetActorEnableCollision(false); 
} 

void AInventoryActor::PutDown(FTransform TargetLocation) 
{ 
  SetActorTickEnabled(true); 
  SetActorHiddenInGame(false); 
  SetActorEnableCollision(true); 
  SetActorLocation(TargetLocation.GetLocation()); 
} 
```

1.  此外，将构造函数修改如下：

```cpp
AInventoryActor::AInventoryActor()
    :Super()
{
    PrimaryActorTick.bCanEverTick = true;
    auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>(TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));

    if (MeshAsset.Object != nullptr)
    {
        GetStaticMeshComponent()->SetStaticMesh(MeshAsset.Object);
        GetStaticMeshComponent()->SetCollisionProfileName( UCollisionProfile::Pawn_ProfileName);
    }

    GetStaticMeshComponent()->SetMobility(EComponentMobility::Movable);

    SetActorEnableCollision(true);
}
```

1.  之后，我们需要为`InventoryActor.cpp`添加以下`#includes`：

```cpp
#include "InventoryActor.h"
#include "ConstructorHelpers.h"
#include "Engine/CollisionProfile.h"
```

1.  我们需要向我们的角色添加一个`InventoryComponent`，以便我们可以在其中存储物品的库存。创建一个从`Character`类派生的类，名为`InventoryCharacter`：

![图片](img/5a38752a-967c-49e6-a018-7e2316e3a4e4.png)

1.  添加以下到`#includes`：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "InventoryComponent.h"
#include "InventoryActor.h"
#include "InventoryCharacter.generated.h"

UCLASS()
class CHAPTER_04_API AInventoryCharacter : public ACharacter
```

1.  然后，将以下添加到`InventoryCharacter`类的声明中：

```cpp
UCLASS()
class CHAPTER_04_API AInventoryCharacter : public ACharacter
{
    GENERATED_BODY()

public:
    // Sets default values for this character's properties
    AInventoryCharacter();

 UPROPERTY()
 UInventoryComponent* MyInventory;

 UFUNCTION()
 void DropItem();

 UFUNCTION()
 void TakeItem(AInventoryActor* InventoryItem);

 UFUNCTION()
 virtual void NotifyHit(class UPrimitiveComponent* MyComp,
 AActor* Other, class UPrimitiveComponent* OtherComp, 
        bool bSelfMoved, FVector HitLocation, FVector
        HitNormal, FVector NormalImpulse, const FHitResult&
        Hit) override;

 UFUNCTION()
 void MoveForward(float AxisValue);
 void MoveRight(float AxisValue);
 void PitchCamera(float AxisValue);
 void YawCamera(float AxisValue);

protected:
    // Called when the game starts or when spawned
    virtual void BeginPlay() override;

public: 
    // Called every frame
    virtual void Tick(float DeltaTime) override;

    // Called to bind functionality to input
    virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

private:
 FVector MovementInput;
 FVector CameraInput;

};
```

1.  将以下行添加到角色的构造函数实现中：

```cpp
AInventoryCharacter::AInventoryCharacter()
{
    // Set this character to call Tick() every frame. You can turn this 
    // off to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = true;

    MyInventory = CreateDefaultSubobject<UInventoryComponent>("MyInventory");
}
```

1.  将以下代码添加到重写的`SetupPlayerInputComponent`：

```cpp
// Called to bind functionality to input
void AInventoryCharacter::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
    Super::SetupPlayerInputComponent(PlayerInputComponent);

    PlayerInputComponent->BindAction("DropItem", 
                                     EInputEvent::IE_Pressed, this, 
                                     &AInventoryCharacter::DropItem);

    // Movement
    PlayerInputComponent->BindAxis("MoveForward", this, 
                                    &AInventoryCharacter::MoveForward);
    PlayerInputComponent->BindAxis("MoveRight", this, 
                                      &AInventoryCharacter::MoveRight);
    PlayerInputComponent->BindAxis("CameraPitch", this, 
                                    &AInventoryCharacter::PitchCamera);
    PlayerInputComponent->BindAxis("CameraYaw", this, 
                                      &AInventoryCharacter::YawCamera);
}
```

1.  接下来，将`MoveForward`、`MoveRight`、`CameraPitch`和`CameraYaw`轴以及`DropItem`动作添加到`Input`菜单中。如果你不记得如何做，请阅读第六章，*输入和碰撞*，其中我们详细介绍了这一点。以下是我在这个特定示例中使用的设置：

![图片](img/8001e2c7-62e2-434b-8865-931916baf3c8.png)

1.  最后，添加以下函数实现：

```cpp
void AInventoryCharacter::DropItem()
{
    if (MyInventory->CurrentInventory.Num() == 0)
    {
        return;
    }
    AInventoryActor* Item = MyInventory->CurrentInventory.Last();
    MyInventory->RemoveFromInventory(Item);

    FVector ItemOrigin;
    FVector ItemBounds;
    Item->GetActorBounds(false, ItemOrigin, ItemBounds);

    FTransform PutDownLocation = GetTransform() + FTransform(RootComponent->GetForwardVector() * ItemBounds.GetMax());

    Item->PutDown(PutDownLocation);
}

void AInventoryCharacter::NotifyHit(class UPrimitiveComponent* MyComp, AActor* Other, class UPrimitiveComponent* OtherComp, bool bSelfMoved, FVector HitLocation, FVector HitNormal, FVector NormalImpulse, const FHitResult& Hit)
{
    AInventoryActor* InventoryItem = Cast<AInventoryActor>(Other);
    if (InventoryItem != nullptr)
    {
        TakeItem(InventoryItem);
    }

}

void AInventoryCharacter::TakeItem(AInventoryActor* InventoryItem)
{
    InventoryItem->PickUp();
    MyInventory->AddToInventory(InventoryItem);
}

//Movement
void AInventoryCharacter::MoveForward(float AxisValue)
{
    MovementInput.X = FMath::Clamp<float>(AxisValue, -1.0f, 1.0f);
}

void AInventoryCharacter::MoveRight(float AxisValue)
{
    MovementInput.Y = FMath::Clamp<float>(AxisValue, -1.0f, 1.0f);
}

void AInventoryCharacter::PitchCamera(float AxisValue)
{
    CameraInput.Y = AxisValue;
}

void AInventoryCharacter::YawCamera(float AxisValue)
{
    CameraInput.X = AxisValue;
}
```

1.  为了处理移动函数，更新`Tick`函数如下：

```cpp
// Called every frame
void AInventoryCharacter::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    if (!MovementInput.IsZero())
    {
        MovementInput *= 100;

        //Scale our movement input axis values by 100 units
        // per second
        FVector InputVector = FVector(0, 0, 0);
        InputVector += GetActorForwardVector()* MovementInput.X * 
        DeltaTime;
        InputVector += GetActorRightVector()* MovementInput.Y * 
        DeltaTime;
        /* GEngine->AddOnScreenDebugMessage(-1, 1, 
                FColor::Red, 
                FString::Printf(TEXT("x- %f, y - %f, z - %f"), 
             InputVector.X, InputVector.Y, InputVector.Z)); */
    }

    if (!CameraInput.IsNearlyZero())
    {
        FRotator NewRotation = GetActorRotation();
        NewRotation.Pitch += CameraInput.Y;
        NewRotation.Yaw += CameraInput.X;

        APlayerController* MyPlayerController = 
        Cast<APlayerController>(GetController());
        if (MyPlayerController != nullptr)
        {
            MyPlayerController->AddYawInput(CameraInput.X);
            MyPlayerController->AddPitchInput(CameraInput.Y);
        }
        SetActorRotation(NewRotation);
    }
}
```

1.  然后，添加以下`#include`：

```cpp
#include "InventoryCharacter.h"
#include "GameFramework/CharacterMovementComponent.h"
```

1.  编译你的代码并在编辑器中测试它。创建一个新的关卡并将几个`InventoryActor`实例拖到场景中。

1.  如果需要提醒如何覆盖当前游戏模式，请参考*使用 SpawnActor 实例化 Actor*配方。将以下行添加到该配方中 Game Mode 的构造函数中，然后将你的关卡`GameMode`设置为在该配方中创建的那个：

```cpp
#include "Chapter_04GameModeBase.h"
#include "InventoryCharacter.h"

AChapter_04GameModeBase::AChapter_04GameModeBase()
{
 DefaultPawnClass = AInventoryCharacter::StaticClass();
}
```

1.  当然，我们还需要更新 GameMode 的`.h`文件：

```cpp
UCLASS()
class CHAPTER_04_API AChapter_04GameModeBase : public AGameModeBase
{
    GENERATED_BODY()
    AChapter_04GameModeBase();
};
```

1.  编译并启动你的项目。如果一切顺利，你应该能够通过在它们上行走来拾取物体：

![图片](img/d7b159e0-643e-487e-a0e9-d35608810661.png)

1.  然后，你可以随时按分配给`DropItem`的键丢弃物品：

![图片](img/0f75e9f4-9f66-47dd-a18b-e01ec1148af8.png)

# 它是如何工作的...

我们的新组件包含一个演员数组，通过指针存储它们，并声明添加或从数组中删除物品的函数。这些函数是`TArray`添加/删除功能的简单包装，但允许我们选择性地执行诸如在存储物品之前检查数组是否在指定的尺寸限制内等操作。

`InventoryActor`是一个基类，可用于所有可以被玩家取走的物品。

在`PickUp`函数中，当演员被拾起时，我们需要禁用该演员。为此，我们必须执行以下操作：

+   禁用演员计时

+   隐藏演员

+   禁用碰撞

我们使用`SetActorTickEnabled`、`SetActorHiddenInGame`和`SetActorEnableCollision`函数来完成此操作。

`PutDown`函数是此操作的逆操作。我们启用演员计时，取消隐藏演员，然后将其碰撞重新打开，并将演员传输到所需的位置。

我们在我们的新角色中添加了一个`InventoryComponent`以及一个取物品的函数。

在我们角色的构造函数中，我们为我们的`InventoryComponent`创建一个默认的子对象。我们还添加了一个`NotifyHit`重写，这样当角色击中其他演员时，我们会收到通知。

在此函数内部，我们将其他演员强制转换为`InventoryActor`。如果转换成功，那么我们知道我们的`Actor`是一个`InventoryActor`，因此我们可以调用`TakeItem`函数来取它。

在`TakeItem`函数中，我们通知库存物品演员我们想要拿起它，然后将其添加到我们的库存中。

`InventoryCharacter`中的最后一项功能是`DropItem`函数。此函数检查我们库存中是否有任何物品。如果有，我们从库存中移除它，然后使用物品边界计算在我们玩家角色前方的一个安全距离来丢弃物品，以获取其最大边界框尺寸。

我们然后通知物品，我们正在将其放置在所需位置的世界中。

# 相关内容

+   第五章，*处理事件和委托*，详细解释了事件和输入处理如何在引擎内部协同工作，以及我们在此配方中提到的`SimpleCharacter`类的配方

+   第六章，*输入和碰撞*，也包含有关绑定输入动作和轴的配方

# 创建一个绕轨道运动的组件

此组件与`RotatingMovementComponent`类似，因为它旨在使与之关联的组件以特定方式移动。在这种情况下，它将以固定距离围绕一个固定点移动任何附加的组件。

这可以用于例如，一个围绕**动作角色扮演游戏**中的角色旋转的护盾。

# 如何做...

1.  创建一个新的名为 `OrbitingMovementComponent` 的 `SceneComponent` 子类：

![](img/12609215-3391-497d-9a47-7234f7f59bdf.png)

1.  将以下属性添加到类声明中：

```cpp
UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class CHAPTER_04_API UOrbitingMovementComponent : public USceneComponent
{
  GENERATED_BODY()

public: 
  // Sets default values for this component's properties
  UOrbitingMovementComponent();

 UPROPERTY()
 bool RotateToFaceOutwards;

 UPROPERTY()
 float RotationSpeed;

 UPROPERTY()
 float OrbitDistance;

 float CurrentValue;

protected:
  // Called when the game starts
  virtual void BeginPlay() override;

public: 
  // Called every frame
  virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;
```

1.  将以下代码添加到构造函数中：

```cpp
// Sets default values for this component's properties
UOrbitingMovementComponent::UOrbitingMovementComponent()
{
    // Set this component to be initialized when the game
    // starts, and to be ticked every frame. You can turn
    // these features off to improve performance if you
    // don't need them.
    PrimaryComponentTick.bCanEverTick = true;

    // ...
 RotationSpeed = 5; 
 OrbitDistance = 100; 
 CurrentValue = 0; 
 RotateToFaceOutwards = true; 
}
```

1.  将以下代码添加到 `TickComponent` 函数中：

```cpp
// Called every frame
void UOrbitingMovementComponent::TickComponent(float DeltaTime, 
                                               ELevelTick TickType, 
                                         FActorComponentTickFunction* 
                                         ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    // ...
    float CurrentValueInRadians = FMath::DegreesToRadians<float>(
 CurrentValue);

 SetRelativeLocation(
 FVector(OrbitDistance * FMath::Cos(CurrentValueInRadians), 
 OrbitDistance * FMath::Sin(CurrentValueInRadians), 
 RelativeLocation.Z)
 );

 if (RotateToFaceOutwards)
 {
 FVector LookDir = (RelativeLocation).GetSafeNormal();
 FRotator LookAtRot = LookDir.Rotation();
 SetRelativeRotation(LookAtRot);
 }

 CurrentValue = FMath::Fmod(CurrentValue + (RotationSpeed * 
                               DeltaTime), 360);
}
```

1.  你可以通过创建一个简单的 `Actor` 蓝图来测试这个组件。

1.  将 `OrbitingMovement` 组件添加到你的 `Actor` 中，然后使用 `Cube` 组件添加一些网格。通过在组件面板中将它们拖放到它上，将它们设置为 `OrbitingMovement` 组件的父组件。结果层次结构应如下所示：

![](img/06f23a38-ca51-46ec-a3b5-c7b2d49502cc.png)

1.  如果你对这个过程不确定，请参考 *创建自定义 Actor 组件* 菜谱。

1.  播放以查看网格在 `Actor` 的中心周围以圆形模式移动。

# 它是如何工作的...

添加到组件中的属性是我们用来自定义组件圆周运动的基本参数。

`RotateToFaceOutwards` 指定组件是否在每次更新时面向旋转中心的外侧。`RotationSpeed` 是组件每秒旋转的度数。

`OrbitDistance` 表示旋转组件必须从原点移动的距离。`CurrentValue` 是当前旋转位置（以度为单位）。

在我们的构造函数中，我们为我们的新组件设置一些合理的默认值。

在 `TickComponent` 函数中，我们计算组件的位置和旋转。

下一步的公式需要我们的角度以弧度而不是度为单位表示。弧度用 *π* 来描述一个角度。首先，我们使用 `DegreesToRadians` 函数将当前值（以度为单位）转换为弧度。

`SetRelativeLocation` 函数使用圆周运动的通用公式，即，*Pos(θ) = cos(θ in radians), sin(θ in radians)*。我们保留每个对象的 *Z* 轴位置。

下一步是将对象旋转回原点（或直接远离它）。这仅在 `RotateToFaceOutwards` 为 `true` 时计算，涉及获取组件相对于其父组件的偏移量，并基于从父组件指向当前相对偏移量的向量创建一个旋转器。然后我们将相对旋转设置为结果旋转器。

最后，我们增加当前值（以度为单位），使其以每秒 `RotationSpeed` 个单位移动，将结果值限制在 0 到 360 之间，以允许旋转循环。

# 创建一个生成单位的建筑

对于这个菜谱，我们将创建一个建筑，在特定位置以固定的时间间隔生成单位。

# 如何做...

1.  在编辑器中创建一个新的 `Actor` 子类，我们将它命名为 `Barracks`：

![](img/7df75b09-439e-41bc-b8e7-a5c519efe1fd.png)

1.  然后，将以下实现添加到类中：

```cpp
UCLASS()
class CHAPTER_04_API ABarracks : public AActor
{
    GENERATED_BODY()

public: 
    // Sets default values for this actor's properties
    ABarracks();

protected:
    // Called when the game starts or when spawned
    virtual void BeginPlay() override;

public: 
    // Called every frame
    virtual void Tick(float DeltaTime) override;

 UPROPERTY() 
 UStaticMeshComponent* BuildingMesh; 

 UPROPERTY() 
 UParticleSystemComponent* SpawnPoint; 

 UPROPERTY() 
 UClass* UnitToSpawn; 

 UPROPERTY() 
 float SpawnInterval; 

 UFUNCTION() 
 void SpawnUnit(); 

 UFUNCTION() 
 void EndPlay(const EEndPlayReason::Type EndPlayReason) override; 

 UPROPERTY() 
 FTimerHandle SpawnTimerHandle; 

};
```

1.  将以下代码添加到构造函数中：

```cpp
#include "Barracks.h"
#include "Particles/ParticleSystemComponent.h"
#include "BarracksUnit.h"

// Sets default values
ABarracks::ABarracks()
{
    // Set this actor to call Tick() every frame. You can turn
    // this off to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = true;

 BuildingMesh = CreateDefaultSubobject<UStaticMeshComponent>(
 "BuildingMesh");

 SpawnPoint = CreateDefaultSubobject<UParticleSystemComponent>(
 "SpawnPoint"); 

 SpawnInterval = 10; 

 auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>(
 TEXT("Static    
                      Mesh'/Engine/BasicShapes/Cube.Cube'")); 

 if (MeshAsset.Object != nullptr) 
 { 
 BuildingMesh->SetStaticMesh(MeshAsset.Object); 
 } 

 auto ParticleSystem = 
    ConstructorHelpers::FObjectFinder<UParticleSystem>
    (TEXT("ParticleSystem'/Engine/Tutorial/SubEditors/TutorialAssets  
    /TutorialParticleSystem.TutorialParticleSystem'")); 

 if (ParticleSystem.Object != nullptr) 
 { 
 SpawnPoint->SetTemplate(ParticleSystem.Object); 
 } 

 SpawnPoint->SetRelativeScale3D(FVector(0.5, 0.5, 0.5)); 
 UnitToSpawn = ABarracksUnit::StaticClass(); 
}
```

目前，我们还没有创建 `BarracksUnit` 类，所以你会看到 Visual Studio 抱怨。我们将在完成 `Barracks` 类后立即实现它。

1.  将以下代码添加到 `BeginPlay` 函数中：

```cpp
// Called when the game starts or when spawned
void ABarracks::BeginPlay()
{
    Super::BeginPlay();

 RootComponent = BuildingMesh; 
 SpawnPoint->AttachTo(RootComponent); 
 SpawnPoint->SetRelativeLocation(FVector(150, 0, 0)); 
 GetWorld()->GetTimerManager().SetTimer(SpawnTimerHandle, 
 this, &ABarracks::SpawnUnit, SpawnInterval, true);
}
```

1.  为 `SpawnUnit` 函数创建实现：

```cpp
void ABarracks::SpawnUnit() 
{ 
  FVector SpawnLocation = SpawnPoint->GetComponentLocation(); 
  GetWorld()->SpawnActor(UnitToSpawn, &SpawnLocation); 
}
```

1.  实现重写的 `EndPlay` 函数：

```cpp
void ABarracks::EndPlay(const EEndPlayReason::Type 
 EndPlayReason) 
{ 
  Super::EndPlay(EndPlayReason); 
  GetWorld()->GetTimerManager().ClearTimer(SpawnTimerHandle); 
} 
```

1.  接下来，创建一个新的角色子类，`BarracksUnit`，并添加一个属性：

```cpp
UPROPERTY() 
UParticleSystemComponent* VisualRepresentation; 
```

1.  你需要添加以下 #include 以获取对 `UParticleSystemComponent` 类的访问权限：

```cpp
#include "Particles/ParticleSystemComponent.h"
```

1.  在构造函数实现中初始化组件：

```cpp
VisualRepresentation = 
 CreateDefaultSubobject<UParticleSystemComponent>("SpawnPoin
 t");auto ParticleSystem =
 ConstructorHelpers::FObjectFinder<UParticleSystem>(TEXT("Pa
 rticleSystem'/Engine/Tutorial/SubEditors/TutorialAssets/Tut
 orialParticleSystem.TutorialParticleSystem'")); 
if (ParticleSystem.Object != nullptr) 
{ 
  SpawnPoint->SetTemplate(ParticleSystem.Object); 
} 
SpawnPoint->SetRelativeScale3D(FVector(0.5, 0.5, 0.5)); 
SpawnCollisionHandlingMethod = 
 ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
```

1.  将视觉表示附加到根组件：

```cpp
void ABarracksUnit::BeginPlay() 
{ 
  Super::BeginPlay(); 
  SpawnPoint->AttachTo(RootComponent); 
}
```

1.  最后，将以下内容添加到 `Tick` 函数中，以使生成的演员开始移动：

```cpp
SetActorLocation(GetActorLocation() + FVector(10, 0, 0)); 
```

1.  编译你的项目。将营地演员的一个副本放入关卡中。然后你可以观察到它以固定的时间间隔生成角色。

如果一切顺利，你应该能够将 `Barracks` 对象拖放到世界中并玩游戏。之后，你会注意到对象（`BarracksUnit` 对象）从一个单一的位置生成，并持续朝一个方向移动！

![图片](img/ca986044-837e-4148-a6f0-5c8d32c9bbe3.png)

# 它是如何工作的...

首先，我们创建营地演员。我们添加一个粒子系统组件来指示新单位将生成的地方，以及一个静态网格来表示建筑的视觉表示。

在构造函数中，我们初始化组件，然后使用 `FObjectFinder` 设置它们的值。我们还使用 `StaticClass` 函数设置要生成的类，以从类类型检索 `UClass*` 实例。

在营地的 `BeginPlay` 函数中，我们创建一个计时器，以固定的时间间隔调用我们的 `SpawnUnit` 函数。我们将计时器句柄存储在类的成员变量中，这样当我们的实例正在被销毁时，我们可以停止计时器；否则，当计时器再次触发时，我们会遇到对象指针解引用的崩溃。

`SpawnUnit` 函数获取 `SpawnPoint` 对象的世界空间位置，然后要求世界在该位置生成我们单位类的实例。

`BarracksUnit` 在其 `Tick()` 函数中有代码，每帧向前移动 10 个单位，这样每个生成的单位都会移动以腾出空间给下一个单位。

`EndPlay` 函数重写调用父类函数的实现，这在父类中有计时器需要取消或执行反初始化时很重要。然后它使用在 `BeginPlay` 中存储的计时器句柄来取消计时器。
