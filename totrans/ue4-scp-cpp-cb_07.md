# 类和接口之间的通信：第一部分

本章将涵盖以下内容：

+   创建一个 UInterface

+   在对象上实现 UInterface

+   检查类是否实现了 UInterface

+   将类型强制转换为原生代码中实现的 UInterface

+   从 C++ 调用原生 UInterface 函数

+   从一个 UInterface 继承另一个 UInterface

+   在 C++ 中重写 UInterface 函数

+   使用 UInterface 实现一个简单的交互系统

# 简介

本章向您展示如何编写自己的 UInterfaces，并演示如何在 C++ 中利用它们以最小化类耦合并保持代码整洁。

在您的游戏项目中，您有时需要一系列可能不同的对象共享一个共同的功能，但使用继承是不合适的，因为所涉及的不同对象之间没有 *is-a* 关系。像 C++ 这样的语言倾向于使用多重继承来解决这个问题。

然而，在 Unreal 中，如果您想使父类中的函数对 Blueprint 可访问，您需要将它们都设置为 `UCLASS`。这有两个原因。在同一个对象中两次继承 `UClass` 会破坏 `UObject` 应该形成一个整洁的可遍历层次结构的概念。这也意味着对象上有两个 `UClass` 方法的实例，它们必须在代码中明确区分。Unreal 代码库通过借鉴 C# 中的一个概念来解决此问题：显式接口类型。

使用这种方法而不是组合的原因是，组件仅在 Actors 上可用，而不是在一般的 UObjects 上。接口可以应用于任何 `UObject`。此外，这意味着我们不再在对象和组件之间建模 *is-a* 关系；相反，它只能表示 *has-a* 关系。

# 技术要求

本章需要使用 Unreal Engine 4，并使用 Visual Studio 2017 作为 IDE。有关如何安装这两款软件及其要求的信息，请参阅本书第一章，*UE4 开发工具*。

# 创建一个 UInterface

UInterfaces 是成对的类，它们协同工作以使类能够在多个类层次结构中表现出多态行为。本食谱展示了仅通过代码创建 UInterface 的基本步骤。

# 如何操作...

1.  从内容浏览器中，转到添加新 | 新 C++ 类。从弹出的菜单中，向下滚动直到看到 Unreal Interface 选择项并选择它。之后，点击下一步按钮：

![图片](img/6dca5f4f-671b-4279-91c2-4ec127cda935.png)

1.  从那里，验证类的名称是否为 `MyInterface`，然后点击创建类按钮：

![图片](img/c611d729-5476-4d87-8324-0052c4d67e5c.png)

1.  将以下代码添加到头文件中：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "MyInterface.generated.h"

// This class does not need to be modified.
UINTERFACE(MinimalAPI)
class UMyInterface : public UInterface
{
  GENERATED_BODY()
};

class CHAPTER_07_API IMyInterface
{
  GENERATED_BODY()

  // Add interface functions to this class. This is the class that   
  //will be inherited to implement this interface.

public:
    virtual FString GetTestName();
};
```

1.  在 `.cpp` 文件中实现以下代码的类：

```cpp
#include "MyInterface.h"

// Add default functionality here for any IMyInterface functions that are not pure virtual.
FString IMyInterface::GetTestName()
{
 unimplemented();
 return FString();
}
```

1.  编译你的项目以验证代码是否编写无误。

# 它是如何工作的...

UInterfaces 是作为接口头文件中声明的类对的实现。

像往常一样，因为我们正在利用 Unreal 的反射系统，我们需要包含我们的生成头文件。有关更多信息，请参阅 第五章 中的 *Handling events implemented via virtual functions* 菜单，*Handling Events and Delegates*。

就像继承自使用 `UCLASS` 的 `UObject` 的类一样，我们需要使用 `UINTERFACE` 宏来声明我们的新 `UInterface`。传递 `MinimalAPI` 类指定符只会导出类的类型信息供其他模块使用。

想要了解更多关于此以及其他类指定符的信息，请查看：[`docs.unrealengine.com/en-US/Programming/UnrealArchitecture/Reference/Classes/Specifiers`](https://docs.unrealengine.com/en-US/Programming/UnrealArchitecture/Reference/Classes/Specifiers).

该类被标记为 `UE4COOKBOOK_API` 以帮助导出库符号。

接口 `UObject` 部分的基类是 `UInterface`。

就像 `UCLASS` 类型一样，我们需要在类的主体内部放置一个宏，以便将自动生成的代码插入其中。对于 UInterfaces，这个宏是 `GENERATED_BODY()`。这个宏必须放在类主体的开头。

第二个类也被标记为 `UE4COOKBOOK_API`，并且以特定的方式命名。

注意，`UInterface` 派生类和标准类具有相同的名称但不同的前缀。`UInterface` 派生类的前缀是 `U`，而标准类的前缀是 `I`。这一点很重要，因为 Unreal Header Tool 预期类以这种方式命名，以便生成的代码能够正常工作。

纯原生的 Interface 类需要其自己的自动生成内容，我们使用 `GENERATED_BODY()` 宏来包含它。

我们在 `IInterface` 中声明了继承该接口的类应该实现的功能。

在实现文件中，我们实现 `UInterface` 的构造函数，因为它是由 Unreal Header Tool 声明并需要实现的。

我们还为我们自己的 `GetTestName()` 函数创建了一个默认实现。如果没有这个实现，编译过程中的链接阶段将会失败。这个默认实现使用了 `unimplemented()` 宏，当执行到该代码行时会发出调试断言。

想要了解更多关于创建界面的信息，请查看以下链接：[`docs.unrealengine.com/en-us/Programming/UnrealArchitecture/Reference/Interfaces`](https://docs.unrealengine.com/en-us/Programming/UnrealArchitecture/Reference/Interfaces).

# 参见

+   参阅 第五章 中的 *Passing payload data with a delegate binding* 菜单，*Handling Events and Delegates*；特别是第一个菜谱解释了我们在这里应用的一些原则。

# 在对象上实现 UInterface

现在我们已经创建了一个 UInterface，我们可以说一个对象具有定义的所有函数或实现了它们。在这个食谱中，我们将看到如何确切地做到这一点。

# 准备工作

确保你已经遵循了前面的食谱，以便你有一个准备实现的 `UInterface`。

# 如何做...

1.  使用 Unreal Wizard 创建一个新的 `Actor` 类，命名为 `SingleInterfaceActor`：

![图片](img/7e99dfee-3068-4773-a9e3-442b78c69dc1.png)

1.  将 `IInterface`——在这种情况下，`IMyInterface`——添加到我们新的 `Actor` 类的公共继承列表中：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MyInterface.h"
#include "SingleInterfaceActor.generated.h"

UCLASS()
class CHAPTER_07_API ASingleInterfaceActor : public AActor, public IMyInterface
{
  GENERATED_BODY()
```

1.  为我们希望重写的 `IInterface` 函数添加一个 `override` 声明：

```cpp
UCLASS()
class CHAPTER_07_API ASingleInterfaceActor : public AActor, public IMyInterface
{
  GENERATED_BODY()

public: 
  // Sets default values for this actor's properties
  ASingleInterfaceActor();

protected:
  // Called when the game starts or when spawned
  virtual void BeginPlay() override;

public: 
  // Called every frame
  virtual void Tick(float DeltaTime) override;

    FString GetTestName() override;

};
```

1.  在实现文件中通过添加以下代码来实现重写的函数：

```cpp
FString ASingleInterfaceActor::GetTestName()
{
    return IMyInterface::GetTestName();
}
```

# 它是如何工作的...

C++ 在实现接口时使用多重继承，所以我们利用这个机制在这里声明我们的 `SingleInterfaceActor` 类，其中我们添加 `public IMyInterface`。

我们从 `IInterface` 而不是 `UInterface` 继承，以防止 `SingleInterfaceActor` 继承两份 `UObject`。

由于接口声明了一个 `virtual` 函数，如果我们想自己实现它，我们需要使用 `override` 指示符重新声明该函数。

在我们的实现文件中，我们实现我们的重写 `virtual` 函数。

在我们的函数重写中，为了演示目的，我们调用基 `IInterface` 的函数实现。或者，我们也可以编写自己的实现，完全避免调用基类。

我们使用 `IInterface::specifier` 而不是 `Super`，因为 `Super` 指的是我们类父类的 `UClass`，而 `IInterfaces` 不是 UClasses（因此没有 `U` 前缀）。

根据需要，你可以在你的对象上实现第二个或多个 `IInterfaces`。

# 检查类是否实现了 UInterface

当编写 C++ 代码时，确保在使用之前某物存在总是一个好主意。在这个食谱中，我们将看到我们如何检查一个特定的对象是否实现了特定的 `UInterface`。

# 准备工作

按照前两个食谱操作，以便你有一个可以检查的 `UInterface`，以及一个实现该接口并可进行测试的类。

# 如何做...

1.  在你的游戏模式实现中，将以下代码添加到 `BeginPlay` 函数中：

```cpp
void AChapter_07GameModeBase::BeginPlay()
{
    Super::BeginPlay();

    // Spawn a new actor using the ASingleInterfaceActor class at 
    //the default location
    FTransform SpawnLocation;
    ASingleInterfaceActor* SpawnedActor = 
GetWorld()->SpawnActor<ASingleInterfaceActor>(  
                                  ASingleInterfaceActor::StaticClass(), 
                                             SpawnLocation); 

    // Get a reference to the class the actor has
    UClass* ActorClass = SpawnedActor->GetClass();

    // If the class implements the interface, display a message
    if (ActorClass-
    >ImplementsInterface(UMyInterface::StaticClass()))
    {
        GEngine->AddOnScreenDebugMessage(-1, 10, FColor::Red, 
                          TEXT("Spawned actor implements 
                               interface!"));
    }
}
```

1.  由于我们同时引用了 `ASingleInterfaceActor` 和 `IMyInterface`，我们需要在我们的源文件中包含 `MyInterface.h` 和 `SingleInterfaceActor.h`：

```cpp
#include "Chapter_07GameModeBase.h"
#include "MyInterface.h"
#include "SingleInterfaceActor.h"
```

1.  保存你的脚本并编译你的代码。之后，从“世界设置”菜单中，将游戏模式覆盖属性设置为你的 `GameModeBase` 类并玩游戏。如果一切顺利，你应该会看到一个消息表明你已经实现了接口：

![图片](img/422edfd8-62f6-48ff-ab9e-71064b5d6b52.png)

# 它是如何工作的...

在 `BeginPlay` 中，我们创建一个空的 `FTransform` 对象，其所有平移和旋转组件的默认值为 `0`，因此我们不需要显式设置任何组件。

然后，我们使用 `UWorld` 中的 `SpawnActor` 函数来创建我们的 `SingleActorInterface` 实例，并将实例的指针存储到一个临时变量中。

然后，我们使用 `GetClass()` 在我们的实例上获取其关联的 `UClass` 的引用。我们需要 `UClass` 的引用，因为该对象持有对象的全部反射数据。

反射数据包括对象上所有 `UPROPERTY` 的名称和类型，对象的继承层次结构，以及它实现的所有接口列表。

因此，我们可以在 `UClass` 上调用 `ImplementsInterface()`，如果对象实现了相关的 `UInterface`，它将返回 `true`。

如果对象实现了该接口，因此从 `ImplementsInterface` 返回 `true`，我们就会在屏幕上打印一条消息。

# 相关内容

+   第四章，演员和组件，包含许多与演员生成相关的菜谱（例如使用 SpawnActor 实例化演员）

# 将其转换为在本地代码中实现的 UInterface

UInterfaces 作为开发者提供的一个优点是，能够将实现公共接口的异构对象集合视为同一对象的集合，使用 `Cast< >` 来处理转换。

请注意，如果你的类通过蓝图实现接口，则此方法将不起作用。

# 准备工作

你应该准备好一个 `UInterface` 和一个实现该接口的 `Actor`。

使用 Unreal 中的向导创建一个新的游戏模式，或者重用之前菜谱中的项目及 `GameMode`。

# 如何做到...

1.  打开你的游戏模式声明，并向类中添加一个新属性：

```cpp
UCLASS()
class CHAPTER_07_API AChapter_07GameModeBase : public AGameModeBase
{
    GENERATED_BODY()

public:
    virtual void BeginPlay() override;

 TArray<IMyInterface*> MyInterfaceInstances;
};
```

1.  在头文件的包含部分添加 `#include "MyInterface.h"`：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "MyInterface.h"
#include "Chapter_07GameModeBase.generated.h"
```

1.  在游戏模式的 `BeginPlay` 实现中添加以下内容：

```cpp
for (TActorIterator<AActor> It(GetWorld(), AActor::StaticClass()); 
     It; 
     ++It)
{
    AActor* Actor = *It;

    IMyInterface* MyInterfaceInstance = Cast<IMyInterface>(Actor);

    // If the pointer is valid, add it to the list
    if (MyInterfaceInstance)
    {
        MyInterfaceInstances.Add(MyInterfaceInstance);
    }
}

// Print out how many objects implement the interface
FString Message = FString::Printf(TEXT("%d actors implement the 
                              interface"), MyInterfaceInstances.Num());

GEngine->AddOnScreenDebugMessage(-1, 10, FColor::Red, Message);
```

1.  由于我们使用 `TActorIterator` 类，我们需要在 `GameModeBase` 类的实现文件顶部添加以下 `#include`：

```cpp
#include "Chapter_07GameModeBase.h"
#include "MyInterface.h"
#include "SingleInterfaceActor.h"
#include "EngineUtils.h" // TActorIterator
```

1.  如果尚未这样做，请将级别的游戏模式覆盖设置为你的游戏模式，然后将几个自定义接口实现 Actor 的实例拖入级别。

1.  当你播放你的级别时，屏幕上应该会打印一条消息，指示级别中 Actor 实现的接口实例数量：

![图片](img/0a8cdda5-261c-4524-9d5f-bf81e434fb90.png)

# 它是如何工作的...

我们创建一个指向 `MyInterface` 实现的指针数组。

在 `BeginPlay` 中，我们使用 `TActorIterator<AActor>` 来获取我们级别中的所有 `Actor` 实例。

`TActorIterator` 有以下构造函数：

```cpp
explicit TActorIterator( UWorld* InWorld, 
 TSubclassOf<ActorType>InClass = ActorType::StaticClass() ) 
: Super(InWorld, InClass ) 
```

`TActorIterator` 期望一个可以作用的世界，以及一个 `UClass` 实例来指定我们感兴趣的 Actor 类型。

`ActorIterator`是一个类似于 STL 迭代器类型的迭代器。这意味着我们可以编写如下形式的`for`循环：

```cpp
for (iterator-constructor;iterator;++iterator) 
```

在循环内部，我们取消迭代器的引用以获取一个`Actor`指针。

我们随后尝试将其转换为我们的接口；如果它确实实现了该接口，这将返回接口的指针，否则它将返回`nullptr`。

因此，我们可以检查接口指针是否为`null`，如果不是，我们可以将接口指针引用添加到我们的数组中。

最后，一旦我们迭代了`TActorIterator`中的所有演员，我们可以在屏幕上显示一个消息，显示实现了该接口的项目数量。

# 从 C++调用原生 UInterface 函数

我们也可以使用 C++从其他类调用原生的`UInterface`函数。例如，在这个菜谱中，我们将使一个体积在对象实现特定接口的情况下调用该对象上的函数。

# 准备工作

按照之前的菜谱了解如何将`Actor`指针转换为接口指针。

注意，由于这个菜谱依赖于我们在之前的菜谱中使用的转换技术，它只适用于使用 C++而不是 Blueprint 实现接口的对象。这是因为 Blueprint 类在编译时不可用，因此技术上不继承接口。

# 如何做到这一点...

1.  使用编辑器向导创建一个新的`Actor`类。命名为`AntiGravityVolume`：

![截图](img/792cb6e1-aada-45b7-91cb-a91759eec844.png)

1.  更新头文件以向新的`Actor`添加一个`BoxComponent`和两个虚拟函数：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/BoxComponent.h"
#include "AntiGravityVolume.generated.h"

UCLASS()
class CHAPTER_07_API AAntiGravityVolume : public AActor
{
    GENERATED_BODY()

public: 
    // Sets default values for this actor's properties
    AAntiGravityVolume();

protected:
    // Called when the game starts or when spawned
    virtual void BeginPlay() override;

public: 
    // Called every frame
    virtual void Tick(float DeltaTime) override;

 UPROPERTY()
 UBoxComponent* CollisionComponent;

 virtual void NotifyActorBeginOverlap(AActor* OtherActor) override;
 virtual void NotifyActorEndOverlap(AActor* OtherActor) override;

};
```

1.  在你的源文件中创建一个实现，如下所示：

```cpp
void AAntiGravityVolume::NotifyActorBeginOverlap(AActor* OtherActor)
{
    IGravityObject* GravityObject = Cast<IGravityObject>(OtherActor);

    if (GravityObject != nullptr)
    {
        GravityObject->DisableGravity();
    }
}

void AAntiGravityVolume::NotifyActorEndOverlap(AActor* OtherActor)
{
    IGravityObject* GravityObject = Cast<IGravityObject>(OtherActor);

    if (GravityObject != nullptr)
    {
        GravityObject->EnableGravity();
    }
}
```

1.  在你的构造函数中初始化`BoxComponent`：

```cpp
// Sets default values
AAntiGravityVolume::AAntiGravityVolume()
{
    // Set this actor to call Tick() every frame. You can turn this off 
    // to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = true;

    CollisionComponent = 
           CreateDefaultSubobject<UBoxComponent>("CollisionComponent");

    CollisionComponent->SetBoxExtent(FVector(200, 200, 400));
    RootComponent = CollisionComponent;
}
```

脚本将无法编译，因为`GravityObject`不存在。让我们修复这个问题：

1.  创建一个名为`GravityObject`的接口：

![截图](img/f70de395-ca21-4083-ba93-20a4bfc9d5e3.png)

1.  向`IGravityObject`添加以下`virtual`函数：

```cpp
class CHAPTER_07_API IGravityObject
{
    GENERATED_BODY()

    // Add interface functions to this class. This is the class that 
    // will be inherited to implement this interface.
public:
 virtual void EnableGravity();
 virtual void DisableGravity();
};

```

1.  在`IGravityObject`实现文件中创建`virtual`函数的默认实现：

```cpp
#include "GravityObject.h"

// Add default functionality here for any IGravityObject functions that are not pure virtual.
void IGravityObject::EnableGravity()
{
 AActor* ThisAsActor = Cast<AActor>(this);
 if (ThisAsActor != nullptr)
 {
 TArray<UPrimitiveComponent*> PrimitiveComponents;

 ThisAsActor->GetComponents(PrimitiveComponents);

 for (UPrimitiveComponent* Component : PrimitiveComponents)
 {
 Component->SetEnableGravity(true);
 }

 GEngine->AddOnScreenDebugMessage(-1, 1, FColor::Red, 
        TEXT("Enabling Gravity"));
 }
}

void IGravityObject::DisableGravity()
{
 AActor* ThisAsActor = Cast<AActor>(this);
 if (ThisAsActor != nullptr)
 {
 TArray<UPrimitiveComponent*> PrimitiveComponents;

 ThisAsActor->GetComponents(PrimitiveComponents);

 for (UPrimitiveComponent* Component : PrimitiveComponents)
 {
 Component->SetEnableGravity(false);
 }

 GEngine->AddOnScreenDebugMessage(-1, 1, FColor::Red, 
        TEXT("Disabling Gravity"));
 }
}
```

1.  之后，回到`AntiGravityVolume.cpp`文件并添加以下`#include`：

```cpp
#include "AntiGravityVolume.h"
#include "GravityObject.h"
```

到目前为止，我们的代码可以编译，但没有东西使用这个接口。让我们添加一个新的类来使用它。

1.  创建一个名为`PhysicsCube`的`Actor`的子类：

![截图](img/9d05909d-0f15-4f7f-be62-c37bbe5a0822.png)

1.  在头文件中添加一个静态网格属性：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/StaticMeshComponent.h"
#include "PhysicsCube.generated.h"

UCLASS()
class CHAPTER_07_API APhysicsCube : public AActor
{
  GENERATED_BODY()

public: 
  // Sets default values for this actor's properties
  APhysicsCube();

protected:
  // Called when the game starts or when spawned
  virtual void BeginPlay() override;

public: 
  // Called every frame
  virtual void Tick(float DeltaTime) override;

 UPROPERTY()
 UStaticMeshComponent* MyMesh;

};
```

1.  在你的构造函数中初始化组件：

```cpp
#include "PhysicsCube.h"
#include "ConstructorHelpers.h"

// Sets default values
APhysicsCube::APhysicsCube()
{
   // Set this actor to call Tick() every frame. You can turn this 
   //off 
   // to improve performance if you don't need it.
  PrimaryActorTick.bCanEverTick = true;

 MyMesh = CreateDefaultSubobject<UStaticMeshComponent>("MyMesh");

 auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>
    (TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));

 if (MeshAsset.Object != nullptr)
 {
 MyMesh->SetStaticMesh(MeshAsset.Object);
 }

 MyMesh->SetMobility(EComponentMobility::Movable);
 MyMesh->SetSimulatePhysics(true);
 SetActorEnableCollision(true);
}
```

1.  要让`PhysicsCube`实现`GravityObject`，首先在头文件中包含`#include "GravityObject.h"`，然后修改类声明：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/StaticMeshComponent.h"
#include "GravityObject.h"
#include "PhysicsCube.generated.h"

UCLASS()
class CHAPTER_07_API APhysicsCube : public AActor, public IGravityObject
```

1.  编译你的项目。

1.  创建一个新的关卡并将我们的重力体积实例放置在场景中。

1.  在重力体积上方放置一个`PhysicsCube`实例，然后稍微旋转它，使其一个角比其他角低，如下面的截图所示：

![截图](img/104ff00e-4823-4862-b609-48373dbda8a8.png)

1.  验证当对象进入体积时重力被关闭，然后再次打开：

![图片](img/a8a01273-e5e5-42c6-97b6-e47dc0aeeb6a.png)

注意，重力体积不需要了解你的 `PhysicsCube` 演员任何事情，只需要 `GravityObject` 接口。

# 它是如何工作的...

我们创建一个新的 `Actor` 类并添加一个盒子组件，以便让演员拥有可以与角色碰撞的东西。或者，如果你想使用 **二叉空间划分（BSP**） 功能来定义体积的形状（在模式选项卡的“位置”部分下的“几何”中找到），你可以将 `AVolume` 作为子类。

`NotifyActorBeginOverlap` 和 `NotifyActorEndOverlap` 被覆盖，这样我们就可以在对象进入或离开 `AntiGravityVolume` 区域时执行操作。

在 `NotifyActorBeginOverlap` 的实现中，我们尝试将重叠我们的对象强制转换为 `IGravityObject` 指针。这测试了问题中的对象是否实现了接口。如果指针有效，那么该对象确实实现了接口，因此可以使用接口指针在对象上调用接口方法。

由于我们处于 `NotifyActorBeginOverlap` 中，我们希望禁用对象的重力，因此我们调用 `DisableGravity()`。在 `NotifyActorEndOverlap` 中，我们执行相同的检查，但重新启用对象的重力。在 `DisableGravity` 的默认实现中，我们将自己的指针（`this` 指针）强制转换为 `AActor`。这允许我们确认接口仅在 `Actor` 子类中实现，并且可以调用 `AActor` 中定义的方法。

如果指针有效，我们知道我们是一个 `Actor`，因此我们可以使用 `GetComponents<class ComponentType>()` 来从我们自身获取所有特定类型的组件的 `TArray`。`GetComponents` 是一个 `template` 函数。它期望一些模板参数，如下所示：

```cpp
template<class T, class AllocatorType>
 voidGetComponents(TArray<T*, AllocatorType>&OutComponents)
 const
```

自从 2014 年的标准化版本以来，C++ 支持编译时模板参数推导。这意味着如果编译器可以从我们提供的普通函数参数中解析出这些参数，我们就不需要实际指定模板参数来调用函数。

`TArray` 的默认实现是 `template<typename T, typename Allocator = FDefaultAllocator> class TArray;` 这意味着我们默认不需要指定分配器，因此当我们声明数组时，我们只需使用 `TArray<UPrimitiveComponent*>`。

当 `TArray` 被传递到 `GetComponents` 函数时，编译器知道它实际上是 `TArray<UPrimitiveComponent*, FDefaultAllocator>`，并且它能够用 `UPrimitiveComponent` 和 `FDefaultAllocator` 填充模板参数 `T` 和 `AllocatorType`，因此这两个都不是函数调用的模板参数所必需的。

`GetComponents` 遍历 `Actor` 拥有的组件，并将任何从 `typename T` 继承的组件的指针存储在 `PrimitiveComponents` 数组中。

使用基于范围的`for`循环，C++的另一个新特性，我们可以遍历函数放入我们的`TArray`中的组件，而无需使用传统的`for`循环结构。

每个组件都会调用`SetEnableGravity(false)`，这将禁用重力。

同样，`EnableGravity`函数遍历演员中包含的所有原始组件，并使用`SetEnableGravity(true)`启用重力。

# 参考也

+   查看第四章，*演员和组件*，以了解关于演员和组件的详细讨论

+   第五章，*处理事件和委托*，讨论了诸如`NotifyActorOverlap`的事件

# 从彼此继承 UInterfaces

有时，你可能需要创建一个专门于更通用`UInterface`的`UInterface`。这个配方展示了如何使用 UInterfaces 的继承来专门化一个不能通过常规方式被杀死的`Killable`接口。

# 如何做到...

1.  创建一个名为`Killable`的`UInterface`/`IInterface`：

![图片](img/20054dc1-f6a5-4b7e-aec6-a85686630416.png)

1.  在`UInterface`声明中添加`UINTERFACE(meta=(CannotImplementInterfaceInBlueprint))`：

```cpp
// This class does not need to be modified.
UINTERFACE(meta = (CannotImplementInterfaceInBlueprint))
class UKillable : public UInterface
{
  GENERATED_BODY()
};
```

1.  在`IKillable`类下添加以下函数：

```cpp
class CHAPTER_07_API IKillable
{
    GENERATED_BODY()

    // Add interface functions to this class. This is the class that 
    // will be inherited to implement this interface.
public:
 UFUNCTION(BlueprintCallable, Category = Killable)
 virtual bool IsDead();
 UFUNCTION(BlueprintCallable, Category = Killable)
 virtual void Die();
};
```

1.  在实现文件中为接口提供默认实现：

```cpp
#include "Killable.h"

// Add default functionality here for any IKillable functions that are 
// not pure virtual.
bool IKillable::IsDead()
{
 return false;
}

void IKillable::Die()
{
 GEngine->AddOnScreenDebugMessage(-1, 1, FColor::Red, "Arrrgh");

 AActor* Me = Cast<AActor>(this);

 if (Me)
 {
 Me->Destroy();
 }

}
```

1.  创建一个新的`UINTERFACE`/`IInterface`名为`Undead`：

![图片](img/10841280-bffb-4867-a944-d463288ade74.png)

1.  修改它们，使它们从`UKillable`/`IKillable`继承：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "Killable.h"
#include "Undead.generated.h"

// This class does not need to be modified.
UINTERFACE(MinimalAPI)
class UUndead : public UKillable
{
  GENERATED_BODY()
};

/**
 * 
 */
class CHAPTER_07_API IUndead : public IKillable
{
  GENERATED_BODY()

  // Add interface functions to this class. This is the class that will 
  // be inherited to implement this interface.
public:
};
```

确保你包含了定义`Killable`接口的头文件。

1.  向新接口添加一些重写和新方法声明：

```cpp
class CHAPTER_07_API IUndead : public IKillable
{
    GENERATED_BODY()

    // Add interface functions to this class. This is the class that 
    // will be inherited to implement this interface.
public:
 virtual bool IsDead() override;
 virtual void Die() override;
 virtual void Turn();
 virtual void Banish();
};
```

1.  为以下函数创建实现：

```cpp
#include "Undead.h"

// Add default functionality here for any IUndead functions that are 
// not pure virtual.
bool IUndead::IsDead()
{
 return true;
}

void IUndead::Die()
{
 GEngine->AddOnScreenDebugMessage(-1, 1, FColor::Red, "You can't kill what is already dead. Mwahaha");
}

void IUndead::Turn()
{
 GEngine->AddOnScreenDebugMessage(-1, 1, FColor::Red, "I'm fleeing!");

}

void IUndead::Banish()
{
 AActor* Me = Cast<AActor>(this);
 if (Me)
 {
 Me->Destroy();
 }
}
```

1.  在 C++中创建两个新的`Actor`类：一个名为`Snail`，另一个名为`Zombie`。

1.  将`Snail`类设置为实现`IKillable`接口，并添加适当的头文件，`#include`：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Killable.h"
#include "Snail.generated.h"

UCLASS()
class CHAPTER_07_API ASnail : public AActor, public IKillable
```

1.  同样，将`Zombie`类设置为实现`IUndead`，并添加`#include "Undead.h"`：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Undead.h"
#include "Zombie.generated.h"

UCLASS()
class CHAPTER_07_API AZombie : public AActor, public IUndead
```

1.  编译你的项目并将`Zombie`和`Snail`的实例拖入你的关卡中：

![图片](img/131d2208-a1c8-45cd-ab64-d2bb2dfc2f2c.png)

1.  通过转到蓝 prints | 关卡蓝图来打开关卡蓝图。之后，通过从世界大纲拖放并释放每个新创建的对象到关卡蓝图，逐个添加对新创建对象的引用：

![图片](img/266bfc76-5685-4a63-872c-59b39aea216a.png)

1.  之后，对每个引用调用`Die (Interface Call)`：

![图片](img/48189563-55b8-4d89-9964-44ff7b73676d.png)

1.  连接两个消息调用的执行引脚，并将其连接到`Event BeginPlay`。运行游戏，然后验证`Zombie`对你的杀戮努力感到不屑，而`Snail`则会呻吟然后死亡（它将从世界大纲中移除）：

![图片](img/64dc17b9-ed35-4104-bfec-cd739c40da78.jpg)

# 它是如何工作的...

为了使 Level Blueprint 能够测试这个配方，我们需要使接口函数可以通过 Blueprint 调用，因此我们需要在`UFUNCTION`上使用`BlueprintCallable`指定符。

然而，在`UInterface`中，编译器默认期望接口可以通过 C++和 Blueprint 两种方式实现。这与`BlueprintCallable`相冲突，后者仅仅表示函数可以从 Blueprint 中调用，并不意味着可以在其中重写它。

我们可以通过将接口标记为`CannotImplementInterfaceInBlueprint`来解决这个冲突。这使我们可以使用`BlueprintCallable`作为我们的`UFUNCTION`指定符，而不是`BlueprintImplementableEvent`（它由于额外的代码允许通过 Blueprint 重写函数而具有额外的开销）。

我们将`IsDead`和`Die`定义为`virtual`，以便它们可以在继承自这个类的另一个 C++类中被重写。在我们的默认接口实现中，`IsDead`始终返回`false`。`Die`的默认实现会在屏幕上打印一条死亡消息，然后如果实现此接口的是`Actor`，则销毁该对象。

现在，我们可以创建一个名为`Undead`的第二个接口，它继承自`Killable`。我们在类声明中使用`public UKillable`/`public IKillable`来表示这一点。

当然，作为结果，我们需要包含定义`Killable`接口的头文件。我们的新接口覆盖了`Killable`定义的两个函数，为 Undead 提供了更合适的`IsDead`/`Die`定义。我们覆盖的定义在`IsDead`中返回`true`，使 Undead 已经死亡。当对`Undead`调用`Die`时，我们只是打印一条消息，其中 Undead 嘲笑我们再次试图杀死它的无力尝试，并且不采取任何行动。

我们还可以为我们的 Undead 特定函数指定默认实现，即`Turn()`和`Banish()`。当 Undead 被 Turn 时，他们会逃跑，为了演示目的，我们在屏幕上打印一条消息。然而，如果一个 Undead 被 Banished，他们将被消灭并消失无踪。

为了测试我们的实现，我们创建了两个`Actors`，每个`Actors`都继承自这两个接口之一。在我们将每个演员实例添加到我们的关卡中后，我们使用 Level Blueprint 来访问关卡中的`BeginPlay`事件。当关卡开始播放时，我们使用消息调用尝试在我们的实例上调用`Die`函数。

打印出来的消息不同，对应于两个函数实现，这表明僵尸对`Die`的实现不同，并覆盖了蜗牛的实现。

# 在 C++中覆盖 UInterface 函数

UInterfaces 允许在 C++中继承的一个副作用是，我们可以在子类以及 Blueprint 中重写默认实现。这个配方将向您展示如何做到这一点。

# 准备工作

按照在 C++中调用原生 UInterface 函数的配方进行操作，其中已经创建了一个物理立方体，以便您有准备好的类。

# 如何操作...

1.  创建一个名为 `Selectable` 的新接口：

![](img/cf51461a-92cc-42bb-b53b-83914ce680dd.png)

1.  在 `ISelectable` 内部定义以下函数：

```cpp
class CHAPTER_07_API ISelectable
{
  GENERATED_BODY()

  // Add interface functions to this class. This is the class that will 
  // be inherited to implement this interface.
public:
 virtual bool IsSelectable();
 virtual bool TrySelect();
 virtual void Deselect();
};
```

1.  为函数提供默认实现，如下所示：

```cpp
#include "Selectable.h"

// Add default functionality here for any ISelectable functions that are not pure virtual.
bool ISelectable::IsSelectable()
{
    GEngine->AddOnScreenDebugMessage(-1, 1, FColor::Red, "Selectable");
    return true;
}

bool ISelectable::TrySelect()
{
    GEngine->AddOnScreenDebugMessage(-1, 1, FColor::Red, "Accepting Selection");
    return true;
}

void ISelectable::Deselect()
{
    unimplemented();
}
```

1.  通过在内容浏览器中右键单击物理立方体脚本并选择从 `PhysicsCube` 创建派生自 `PhysicsCube` 的 C++ 类来创建一个基于 `APhysicsCube` 的类：

![](img/705a33b1-bc34-4cab-83ef-5eb9f2f739c2.png)

1.  完成后，将新立方体的名称更改为 `SelectableCube` 并单击创建类选项：

![](img/76752a9d-e70c-4fae-a2e7-faa8ab5cb57b.png)

1.  在 `SelectableCube` 类的头文件中添加 `#include "Selectable.h"`。

1.  修改 `ASelectableCube` 的声明，如下所示：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "PhysicsCube.h"
#include "Selectable.h"
#include "SelectableCube.generated.h"

UCLASS()
class CHAPTER_07_API ASelectableCube : public APhysicsCube, public ISelectable
```

1.  在头文件中添加以下函数：

```cpp
UCLASS()
class CHAPTER_07_API ASelectableCube : public APhysicsCube, public ISelectable
{
    GENERATED_BODY()

public:
 ASelectableCube();
 virtual void NotifyHit(class UPrimitiveComponent* MyComp, 
 AActor* Other, 
 class UPrimitiveComponent* OtherComp, 
 bool bSelfMoved, FVector HitLocation, 
 FVector HitNormal, FVector NormalImpulse, 
 const FHitResult& Hit) override;

};

```

1.  实现函数：

```cpp
#include "SelectableCube.h"

ASelectableCube::ASelectableCube() : Super()
{
 MyMesh->SetNotifyRigidBodyCollision(true);
}

void ASelectableCube::NotifyHit(class UPrimitiveComponent* MyComp, 
 AActor* Other, 
 class UPrimitiveComponent* OtherComp, 
 bool bSelfMoved, FVector HitLocation, 
 FVector HitNormal, 
 FVector NormalImpulse, 
 const FHitResult& Hit)
{
 if (ISelectable::IsSelectable())
 {
 TrySelect();
 }
}
```

1.  以与创建 `SelectableCube` 类相同的方式，在相同的位置创建一个名为 `NonSelectableCube` 的新类，它继承自 `SelectableCube`：

![](img/604d9703-5f99-44c1-8cea-36ab7a22d846.png)

1.  `NonSelectableCube` 应该覆盖 `SelectableInterface` 中的函数：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "SelectableCube.h"
#include "NonSelectableCube.generated.h"

UCLASS()
class CHAPTER_07_API ANonSelectableCube : public ASelectableCube
{
    GENERATED_BODY()

public:
 virtual bool IsSelectable() override;
 virtual bool TrySelect() override;
 virtual void Deselect() override;
};

```

1.  实现文件应修改为包含以下内容：

```cpp
#include "NonSelectableCube.h"

bool ANonSelectableCube::IsSelectable()
{
 GEngine->AddOnScreenDebugMessage(-1, 1, FColor::Red, "Not Selectable"); 
 return false;
}

bool ANonSelectableCube::TrySelect()
{
 GEngine->AddOnScreenDebugMessage(-1, 1, FColor::Red, "Refusing Selection");
 return false;
}

void ANonSelectableCube::Deselect()
{
 unimplemented();
}
```

1.  在地面上方一定范围内放置一个 `SelectableCube` 的实例，然后玩游戏。你应该收到确认消息，表明演员是可选择的，并且在立方体触地时已接受选择：

![](img/d514b9e7-96d1-4b54-963e-3a1f72546c67.jpg)

1.  移除 `SelectableCube` 并用 `NonSelectableCube` 的实例替换它，以查看表示此演员不可选择且已拒绝选择的替代消息。

# 它是如何工作的...

我们在 `Selectable` 接口中创建了三个函数。`IsSelectable` 返回一个布尔值，表示对象是否可选择。你可以避免使用它，直接使用 `TrySelect`，因为 `TrySelect` 返回一个布尔值以指示成功，但例如，你可能想知道你的 UI 中的对象是否是一个有效的选择，而无需实际尝试它。

`TrySelect` 实际上尝试选择对象。没有显式的契约强制用户在尝试选择对象时遵守 `IsSelectable`，因此 `TrySelect` 被命名为以传达选择可能不会总是成功。

最后，`Deselect` 是一个函数，它被添加以允许对象处理失去玩家选择的情况。这可能涉及更改 UI 元素、停止声音或其他视觉效果，或者简单地从单位周围移除选择轮廓。

函数的默认实现返回 `true` 对于 `IsSelectable`（默认情况下任何对象都是可选择的），对于 `TrySelect`（选择尝试总是成功），如果调用 `Deselect` 而没有由类实现，则发出调试断言。

如果你想，你也可以将 `Deselect` 实现为一个纯 `virtual` 函数。`SelectableCube` 是一个从 `PhysicsCube` 继承的新类，同时也实现了 `ISelectable` 接口。它还覆盖了 `NotifyHit`，这是一个在 `AActor` 中定义的 `virtual` 函数，当演员发生 **RigidBody** 碰撞时触发。

我们在 `SelectableCube` 的实现中调用 `PhysicsCube` 的构造函数，并在其中使用 `Super()` 构造函数调用。然后我们添加自己的实现，它在我们静态网格实例上调用 `SetNotifyRigidBodyCollision(true)`。这是必要的，因为默认情况下，`RigidBodies`（例如具有碰撞的 `PrimitiveComponents`）不会触发 `Hit` 事件，作为性能优化。因此，我们的覆盖 `NotifyHit` 函数永远不会被调用。

在 `NotifyHit` 的实现中，我们对自己调用了一些 `ISelectable` 接口函数。鉴于我们知道我们是一个继承自 `ISelectable` 的对象，我们不需要将其转换为 `ISelectable*` 来调用它们。

我们检查对象是否可由 `IsSelectable` 选择，如果是，我们尝试使用 `TrySelect` 实际执行选择。`NonSelectableCube` 继承自 `SelectableCube`，因此我们可以强制对象永远不可选择。

我们通过再次覆盖 `ISelectable` 接口函数来实现这一点。在 `ANonSelectableCube::IsSelectable()` 中，我们在屏幕上打印一条消息，以便我们可以验证该函数是否被调用，然后返回 `false` 以指示该对象根本不可选择。

如果用户不尊重 `IsSelectable()`，`ANonSelectableCube::TrySelect()` 总是返回 `false` 以指示选择未成功。

由于 `NonSelectableCube` 不可能被选中，`Deselect()` 调用 `unimplemented()`，这会抛出一个断言警告，表明该函数尚未实现。

现在，当您播放场景时，每次 `SelectableCube`/`NonSelectableCube` 与另一个对象发生碰撞，导致 RigidBody 碰撞时，相关的演员将尝试选择自己，并在屏幕上打印消息。

# 参见

+   请参阅第六章，*输入和碰撞*以及*鼠标 UI 输入处理*食谱，该食谱向您展示了如何从鼠标光标向游戏世界进行 **Raycast** 以确定被点击的对象。这可以用来扩展本食谱，允许玩家点击物品以选择它们。

# 使用 UInterfaces 实现简单的交互系统

本食谱将向您展示如何结合本章中的其他多个食谱来演示一个简单的交互系统，以及一个带有可交互门铃的门，以使门打开。

# 准备工作...

本食谱需要使用动作绑定。如果您不熟悉创建动作映射，请在继续本食谱之前，请参阅第六章，*输入和碰撞*。

# 如何做到这一点...

1.  创建一个新的接口，称为`Interactable`。

1.  将以下函数添加到`IInteractable`类声明中：

```cpp
class CHAPTER_07_API IInteractable
{
    GENERATED_BODY()

    // Add interface functions to this class. This is the class that 
    // will be inherited to implement this interface.
public:
 UFUNCTION(BlueprintNativeEvent, BlueprintCallable, 
 Category = Interactable)
 bool CanInteract(); 
 UFUNCTION(BlueprintNativeEvent, BlueprintCallable, 
 Category = Interactable)
 void PerformInteract();

};
```

1.  创建第二个接口，`Openable`。

1.  将此函数添加到其声明中：

```cpp
class CHAPTER_07_API IOpenable
{
  GENERATED_BODY()

  // Add interface functions to this class. This is the class that 
    // will be inherited to implement this interface.
public:
 UFUNCTION(BlueprintNativeEvent, BlueprintCallable, 
 Category = Openable)
 void Open();
};
```

1.  创建一个新的基于`StaticMeshActor`的类，称为`DoorBell`。

1.  在`DoorBell.h`中添加`#include "Interactable.h"`，并将以下函数添加到类声明中：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "Engine/StaticMeshActor.h"
#include "Interactable.h"
#include "DoorBell.generated.h"

UCLASS()
class CHAPTER_07_API ADoorBell : public AStaticMeshActor, public IInteractable
{
  GENERATED_BODY()

public:
 ADoorBell();

 virtual bool CanInteract_Implementation() override;
 virtual void PerformInteract_Implementation() override;

 UPROPERTY(BlueprintReadWrite, EditAnywhere)
 AActor* DoorToOpen;

private:
 bool HasBeenPushed;
};
```

1.  在`DoorBell.cpp`文件中添加`#include "Openable.h"`。

1.  在构造函数中为我们的`DoorBell`加载一个静态网格：

```cpp
ADoorBell::ADoorBell()
{
    HasBeenPushed = false;

    auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>(TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));

    UStaticMeshComponent * SM = GetStaticMeshComponent();

    if (SM != nullptr)
    {
        if (MeshAsset.Object != nullptr)
        {
            SM->SetStaticMesh(MeshAsset.Object);
            SM->SetGenerateOverlapEvents(true);
        }

        SM->SetMobility(EComponentMobility::Movable);
        SM->SetWorldScale3D(FVector(0.5, 0.5, 0.5));
    }

    SetActorEnableCollision(true);

    SetActorEnableCollision(true);

    DoorToOpen = nullptr;
} 
```

1.  将以下函数实现添加到实现`Interactable`接口的`DoorBell`中：

```cpp
bool ADoorBell::CanInteract_Implementation()
{
    return !HasBeenPushed;
}

void ADoorBell::PerformInteract_Implementation()
{
    HasBeenPushed = true;
    if (DoorToOpen->GetClass()->ImplementsInterface( 
                                             UOpenable::StaticClass()))
    {
        IOpenable::Execute_Open(DoorToOpen);
    }
}
```

1.  现在创建一个基于`StaticMeshActor`的新类，称为`Door`。

1.  在类头文件中包含`Openable`和`Interactable`接口，然后修改`Door`的声明：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "Engine/StaticMeshActor.h"
#include "Interactable.h"
#include "Openable.h"
#include "Door.generated.h"

UCLASS()
class CHAPTER_07_API ADoor : public AStaticMeshActor, public IInteractable, public IOpenable
{
    GENERATED_BODY()

};
```

1.  在`Door`中添加接口函数和构造函数：

```cpp
UCLASS()
class CHAPTER_07_API ADoor : public AStaticMeshActor, public IInteractable, public IOpenable
{
    GENERATED_BODY()

public:
 ADoor();

 UFUNCTION()
 virtual bool CanInteract_Implementation() override;

 UFUNCTION()
 virtual void PerformInteract_Implementation() override;

 UFUNCTION()
 virtual void Open_Implementation() override;
};
```

1.  与`DoorBell`一样，在`Door`的构造函数中，初始化我们的网格组件并在其中加载一个模型：

```cpp
ADoor::ADoor()
{
    auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>(TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));

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
}
```

1.  实现接口函数：

```cpp
bool ADoor::CanInteract_Implementation()
{
    return true;
}

void ADoor::PerformInteract_Implementation()
{
    GEngine->AddOnScreenDebugMessage(-1, 5, FColor::Red, TEXT("The door refuses to budge. Perhaps there is a hidden switch nearby ? ")); 
}

void ADoor::Open_Implementation()
{
    AddActorLocalOffset(FVector(0, 0, 200));
}
```

1.  创建一个基于`DefaultPawn`的新类，称为`InteractingPawn`：

![图片](img/7461c8ae-8fcd-4088-9b81-11ef24f01cad.png)

1.  将以下函数添加到`Pawn`类头文件中：

```cpp
UCLASS()
class CHAPTER_07_API AInteractingPawn : public ADefaultPawn
{
    GENERATED_BODY()

public:
 void TryInteract();

private:
 virtual void SetupPlayerInputComponent( UInputComponent* 
                                            InInputComponent) override;
};
```

1.  在`Pawn`的实现文件中添加`#include "Interactable.h"`，然后提供头文件中两个函数的实现：

```cpp
#include "InteractingPawn.h"
#include "Interactable.h"
#include "Camera/PlayerCameraManager.h"
#include "CollisionQueryParams.h"
#include "WorldCollision.h"

void AInteractingPawn::TryInteract()
{
    APlayerController* MyController = Cast<APlayerController>( 
                                                           Controller);

    if (MyController)
    {
        APlayerCameraManager* MyCameraManager = 
                                     MyController->PlayerCameraManager;

        auto StartLocation = MyCameraManager->GetCameraLocation();
        auto EndLocation = StartLocation + 
                      (MyCameraManager->GetActorForwardVector() * 
                       100);

        FCollisionObjectQueryParams Params;
        FHitResult HitResult;

        GetWorld()->SweepSingleByObjectType(HitResult, StartLocation, 
                                            EndLocation, 
                                            FQuat::Identity,
 FCollisionObjectQueryParams(FCollisionObjectQueryParams::AllObjects), 
                                       FCollisionShape::MakeSphere(25),
              FCollisionQueryParams(FName("Interaction"), true, this));

        if (HitResult.Actor != nullptr)
        {
            auto Class = HitResult.Actor->GetClass();
            if (Class->ImplementsInterface( 
                                         UInteractable::StaticClass()))
            {
                if (IInteractable::Execute_CanInteract( 
                                                HitResult.Actor.Get()))
                {
                    IInteractable::Execute_PerformInteract( 
                                                HitResult.Actor.Get());
                }
            }
        }

    }

}

void AInteractingPawn::SetupPlayerInputComponent(UInputComponent* 
                                                      InInputComponent)
{
    Super::SetupPlayerInputComponent(InInputComponent);
    InInputComponent->BindAction("Interact", IE_Released, this, 
                                       &AInteractingPawn::TryInteract);
}
```

1.  现在，在 C++或蓝图中创建一个新的`GameMode`，并将`InteractingPawn`设置为我们的默认`Pawn`类。

![图片](img/69d62766-4048-48e4-9aeb-2f953a73ba43.png)

1.  将`Door`和`Doorbell`的副本拖入级别中：

![图片](img/7942f20f-0d64-45fd-9d08-35eac82a5dd1.png)

1.  使用门铃的“打开门”旁边的吸管工具，如图所示，然后点击你级别中的门演员实例：

![图片](img/9423df93-c5d2-4666-bf2c-1a97fef82dc0.jpg)

一旦你选择了演员，你应该会看到以下类似的内容：

![图片](img/e2c5d6d5-100e-42b4-87fa-9225736a8849.jpg)

1.  在编辑器中创建一个新的动作绑定，称为`Interact`，并将其绑定到您选择的键：

![图片](img/3e619bc5-fa62-4306-a5b7-44f79c468adc.jpg)

1.  播放你的级别，走到门铃前。看看它，并按你绑定的`Interact`键。验证门移动了一次。参见图示：

![图片](img/3d3015cb-4993-4264-a6aa-5cfa33f45a0a.png)

1.  你也可以直接与门进行交互，以获取一些关于它的信息：

![图片](img/c417b020-3dc8-450b-8f4f-d185d61d75a1.png)

# 它是如何工作的...

与之前的食谱一样，我们将`UFUNCTION`标记为`BlueprintNativeEvent`和`BlueprintCallable`，以允许在本地代码或蓝图实现`UInterface`，并允许使用任一方法调用函数。

我们基于`StaticMeshActor`创建`DoorBell`以方便起见，并让`DoorBell`实现`Interactable`接口。在`DoorBell`的构造函数中，我们将`HasBeenPushed`和`DoorToOpen`初始化为默认的安全值。

在`CanInteract`的实现中，我们返回`HasBeenPushed`的逆值，这样一旦按钮被按下，就不能再与之互动。

在`PerformInteract`内部，我们检查是否有打开门的物体对象的引用。如果我们有一个有效的引用，我们验证门 actor 是否实现了`Openable`接口，然后在我们的大门上调用`Open`函数。在`Door`中，我们实现了`Interactable`和`Openable`接口，并覆盖了每个接口的函数。

我们定义`Door`的`CanInteract`实现与默认值相同。在`PerformInteract`中，我们向用户显示一条消息。在`Open`中，我们使用`AddActorLocalOffset`将门移动一定的距离。通过蓝图中的 Timeline 或线性插值，我们可以使这个过渡更加平滑，而不是瞬移。

最后，我们创建一个新的`Pawn`，以便玩家能够真正地与物体互动。我们创建了一个`TryInteract`函数，并将其绑定到在重写的`SetupPlayerInputComponent`函数中的`Interact`输入动作。

这意味着当玩家执行绑定到`Interact`的输入时，我们的`TryInteract`函数将会运行。`TryInteract`获取对`PlayerController`的引用，将所有 Pawn 都具有的通用控制器引用进行类型转换。

通过`PlayerController`获取`PlayerCameraManager`，这样我们就可以访问玩家摄像机的当前位置和旋转。我们使用摄像机的位置创建起点和终点，然后在摄像机位置前方 100 单位处，将这些传递给`GetWorld::SweepSingleByObjectType`。这个函数接受多个参数。`HitResult`是一个变量，允许函数返回关于被追踪的任何物体的信息。`CollisionObjectQueryParams`允许我们指定我们是否对动态、静态物品或两者都感兴趣。

我们通过使用`MakeSphere`函数传递形状来实现球体追踪。球体追踪允许有轻微的人为错误，因为它定义了一个圆柱来检查物体，而不是直线。鉴于玩家可能不会直接看向你的物体，你可以根据需要调整球体的半径。

最后一个参数`SweepSingleByObjectType`是一个结构体，它为追踪赋予一个名称，允许我们指定我们是否在碰撞复杂的碰撞几何体，并且最重要的是，允许我们指定我们想要忽略触发追踪的对象。

如果在追踪完成后`HitResult`包含一个 actor，我们检查该 actor 是否实现了我们的接口，然后尝试调用其上的`CanInteract`。如果 actor 表示可以互动，那么我们就告诉它实际执行互动。
