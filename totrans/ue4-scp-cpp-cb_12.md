# UE4 中的多人网络

在本章中，我们将涵盖以下主题：

+   同时作为客户端和服务器测试你的游戏

+   在网络上复制属性

+   在网络上复制函数

+   处理 UI 网络事件

# 简介

网络是作为程序员你可以做的更复杂的事情之一。幸运的是，Unreal Engine 自 1998 年最初发布以来就考虑了网络。Unreal 使用客户端-服务器模型在多台计算机之间进行通信。在这种情况下，**服务器**是开始游戏的人，而**客户端**是与第一人称一起玩游戏的人。为了使每个人游戏中发生的事情都能正确工作，我们需要在特定时间调用某些代码为某些人。

例如，当客户端想要射击他的/她的枪时，他们会向服务器发送一条消息，然后服务器将决定他们是否击中任何东西，然后通过复制告诉所有客户端发生了什么。这很重要，因为有些事情，如游戏模式，只存在于服务器上。

有关客户端-服务器模型的更多信息，请查看[`en.wikipedia.org/wiki/Client%E2%80%93server_model`](https://en.wikipedia.org/wiki/Client%E2%80%93server_model)。

由于我们希望在屏幕上看到多个角色，在本章中，我们将使用基于第三人称 C++模板的基础项目：

![图片](img/641e12f8-c70e-4e03-a8df-9670db5bc80a.png)

# 技术要求

本章需要使用 Unreal Engine 4，并使用 Visual Studio 2017 作为 IDE。有关如何安装这两款软件及其要求的信息，请参阅本书的第一章，*UE4 开发工具*。

# 同时作为客户端和服务器测试你的游戏

当你在网络上工作游戏时，经常测试你的项目总是一个好主意。无需使用两台不同的计算机，Unreal 内置了一种简单的方法，可以在同一时间玩多个玩家的游戏。

# 如何做到这一点...

通常，当我们玩游戏时，屏幕上只有一个玩家。我们可以通过播放设置来修改这一点：

1.  在 Unreal 编辑器中，打开`ThirdPersonExampleMap`，点击播放按钮旁边的箭头下拉菜单。在那里，将玩家数量属性设置为`2`：

![图片](img/8ed662ce-096e-4c08-b745-d45fa8545243.png)

1.  之后，点击播放按钮：

![图片](img/b9faaf92-6b73-479c-8729-74858e85eecc.png)

如您所见，现在屏幕上已经增加了两个窗口！

请记住，你可以通过按*Shift* + *F1*将鼠标控制从窗口中返回。

# 它是如何工作的...

除了场景中放置的角色外，世界上还有一个名为`NetworkPlayerStart`的另一个对象，这是网络玩家将被生成的地方：

![图片](img/9cc559e3-4292-4166-9655-c3d333129b7b.png)

如果你将更多的玩家起始对象添加到场景中，默认情况下，对象将随机从可用的玩家起始对象中选择。你可以通过按住*Alt*键并拖动一个对象到新方向来快速创建新的对象。

# 在网络上复制属性

为了确保客户端和服务器上的值相同，我们使用复制的过程。在这个菜谱中，我们将看到这样做是多么简单。

# 如何做到...

对于这个简单的例子，让我们创建一个变量来存储每个玩家在游戏中跳跃的次数：

1.  打开 Visual Studio 并打开你项目中角色的定义（在我的例子中，它是`Chapter_12Character.h`）。将以下属性和函数声明添加到文件中：

```cpp
UPROPERTY(Replicated, EditAnywhere)
uint32 JumpCount;

void Jump() override;
```

1.  然后，转到实现文件并添加以下`#include`：

```cpp
#include "UnrealNetwork.h" // DOREPLIFETIME
```

1.  之后，我们需要告诉`SetupPlayerInputComponent`方法使用我们的`Jump`版本而不是父类的：

```cpp
void AChapter_12Character::SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent)
{
  // Set up gameplay key bindings
  check(PlayerInputComponent);
 PlayerInputComponent->BindAction("Jump", IE_Pressed, this, &AChapter_12Character::Jump);
  PlayerInputComponent->BindAction("Jump", IE_Released, this, &ACharacter::StopJumping);

  ...
```

1.  然后，我们需要添加以下函数：

```cpp
void AChapter_12Character::Jump()
{
    Super::Jump();

    JumpCount++;

    if (Role == ROLE_Authority)
    {
        // Only print function once
        GEngine->AddOnScreenDebugMessage(-1, 5.0f,
            FColor::Green,
            FString::Printf(TEXT("%s called Jump %d times!"),
            *GetName(), JumpCount)
        );
    }

}

void AChapter_12Character::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>&
    OutLifetimeProps) const
{
    Super::GetLifetimeReplicatedProps(OutLifetimeProps);

    // Replicate to every client
    //DOREPLIFETIME(AChapter_12Character, JumpCount);
}
```

1.  保存你的脚本并返回到虚幻编辑器。编译你的脚本并玩你的游戏：

![图片](img/8e4c2455-67e8-4044-8670-e291b73e5eb6.png)

现在，每当任意一个玩家按下*空格键*时，你将看到一个显示他们的名字和它将具有的值的消息。

# 它是如何工作的...

属性复制在理论上很简单。每当变量值发生变化时，网络应该通知所有客户端变化，然后更新变量。这通常用于像健康这样值非常重要的东西。

当你注册这样的变量时，这个变量应该只由服务器修改，然后复制到其他客户端。为了标记要复制的项目，我们在`UPROPERTY`中使用`Replicated`指定符。

在标记某个项目为复制后，我们必须定义一个新的函数，称为`GetLifetimeReplicatedProps`，它不需要在头文件中声明。在这个函数内部，我们使用`DOREPLIFETIME`宏来声明，每当服务器上的`JumpCount`变量发生变化时，所有客户端都需要修改该值。

在`Jump`函数内部，我们添加了一些新的功能，但我们首先检查`Role`变量以确定是否应该发生某些操作。`ROLE_Authority`是最高级别，这意味着你是服务器。这确保了我们的功能只会发生一次，而不是多次。

为了使复制工作，请确保将`bReplicates`变量设置为`true`。这应该在类的构造函数中完成。

# 更多...

对于那些想要在代码中添加一些优化的人来说，你可以使用以下替代我们的当前`DOREPLIFETIME`宏：

```cpp
void AChapter_12Character::GetLifetimeReplicatedProps(TArray<FLifetimeProperty>&
    OutLifetimeProps) const
{
    Super::GetLifetimeReplicatedProps(OutLifetimeProps);

    // Replicate to every client
    //DOREPLIFETIME(AChapter_12Character, JumpCount);

 // Value is already updated locally, so we can skip replicating 
    // the value for the owner
 DOREPLIFETIME_CONDITION(AChapter_12Character, JumpCount, COND_SkipOwner);
}
```

这样做使得值只在其他客户端复制，而不是原始值。

更多关于`DOREPLIFETIME_CONDITION`以及一些关于网络的其他技巧和窍门的信息，请查看[`www.unrealengine.com/en-US/blog/network-tips-and-tricks`](https://www.unrealengine.com/en-US/blog/network-tips-and-tricks)。

# 在网络上复制函数

在这个食谱中，我们将看到一个非平凡的复制示例，它使用了一个简单的拾取对象，我们可能希望玩家跟踪它。

# 如何实现...

创建我们的可收集物品的第一步是实际创建我们将要使用的类：

1.  导航到文件 | 新建 C++类，然后在选择父类窗口中，选择 Actor，然后点击下一步：

![图片](img/8ee5323f-7efd-4090-8af0-c5b49fb19958.png)

1.  在下一个窗口中，将名称属性设置为`CollectibleObject`，然后点击创建类按钮将其添加到项目中并编译基本代码：

![图片](img/80bd2ac8-2cc5-4433-ad39-c66b466c7fe1.png)

1.  一旦 Visual Studio 打开，更新`CollectibleObject.h`到以下内容：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "CollectibleObject.generated.h"

UCLASS()
class CHAPTER_12_API ACollectibleObject : public AActor
{
    GENERATED_BODY()

public: 
    // Sets default values for this actor's properties
    ACollectibleObject();

 // Event called when something starts to overlaps the
 // sphere collider
 // Note: UFUNCTION required for replication callbacks
 UFUNCTION() 
 void OnBeginOverlap(class UPrimitiveComponent*
 HitComp, 
 class AActor* OtherActor,
 class UPrimitiveComponent*
 OtherComp,
 int32 OtherBodyIndex, bool
 bFromSweep,
 const FHitResult& SweepResult);

 // Our server function to update the score.
 UFUNCTION(Reliable, Server, WithValidation)
 void UpdateScore(int32 Amount);

 void UpdateScore_Implementation(int32 Amount);
 bool UpdateScore_Validate(int32 Amount);

protected:
    // Called when the game starts or when spawned
    virtual void BeginPlay() override;

public: 
    // Called every frame
    virtual void Tick(float DeltaTime) override;

};
```

1.  然后，在`CollectibleObject.cpp`中，更新类的构造函数到以下内容：

```cpp
#include "ConstructorHelpers.h"
#include "Components/SphereComponent.h"

// ...

// Sets default values
ACollectibleObject::ACollectibleObject()
{
   // Set this actor to call Tick() every frame. You can turn this off to improve performance if you don't need it.
  PrimaryActorTick.bCanEverTick = true;

    // Must be true for an Actor to replicate anything
 bReplicates = true;

 // Create a sphere collider for players to hit
 USphereComponent * SphereCollider = CreateDefaultSubobject<USphereComponent>(TEXT("SphereComponent"));

 // Sets the root of our object to be the sphere collider
 RootComponent = SphereCollider;

 // Sets the size of our collider to have a radius of
 // 64 units
 SphereCollider->InitSphereRadius(64.0f);

 // Makes it so that OnBeginOverlap will be called
 // whenever something hits this.
 SphereCollider->OnComponentBeginOverlap.AddDynamic(this, &ACollectibleObject::OnBeginOverlap);

 // Create a visual to make it easier to see
 UStaticMeshComponent * SphereVisual = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Static Mesh"));

 // Attach the static mesh to the root
 SphereVisual->SetupAttachment(RootComponent);

 // Get a reference to a sphere mesh
 auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>(TEXT("StaticMesh'/Engine/BasicShapes/Sphere.Sphere'"));

 // Assign the mesh if valid
 if (MeshAsset.Object != nullptr)
 {
 SphereVisual->SetStaticMesh(MeshAsset.Object);
 }

 // Resize to be smaller than the larger sphere collider
 SphereVisual->SetWorldScale3D(FVector(0.5f));

}
```

1.  之后，实现`OnBeginOverlap`函数：

```cpp
// Event called when something starts to overlaps the
// sphere collider
void ACollectibleObject::OnBeginOverlap(
    class UPrimitiveComponent* HitComp,
    class AActor* OtherActor,
    class UPrimitiveComponent* OtherComp,
    int32 OtherBodyIndex,
    bool bFromSweep,
    const FHitResult& SweepResult)
{
    // If I am the server
    if (Role == ROLE_Authority)
    {
        // Then a coin will be gained!
        UpdateScore(1);
        Destroy();
    }
}
```

1.  然后，实现`UpdateScore_Implementation`和`UpdateScore_Validate`方法：

```cpp
// Do something here that modifies game state.
void ACollectibleObject::UpdateScore_Implementation(int32
    Amount)
{
    if (GEngine)
    {
        GEngine->AddOnScreenDebugMessage(-1, 5.0f,
            FColor::Green,
            "Collected!");
    }
}

// Optionally validate the request and return false if the
// function should not be run.
bool ACollectibleObject::UpdateScore_Validate(int32 Amount)
{
    return true;
}
```

1.  保存脚本，然后返回 Unity 编辑器。编译你的脚本，然后将`Collectible Object`类的一个实例拖拽到场景中。保存你的关卡，并使用两个玩家玩游戏，如前一个食谱所示。

1.  收集对象后，你应该在屏幕上看到一条消息显示：

![图片](img/f52e7aba-4dcd-4cb2-b270-ccdc626fffa7.png)

通过这种方式，你可以看到消息是如何从服务器复制到客户端的！

# 它是如何工作的...

在`CollectibleObject`类的构造函数中，我们确保我们的对象将被复制。之后，我们创建一个球体碰撞器，并通过监听器告诉它当它与另一个对象碰撞时调用`OnBeginOverlap`函数。为此，我们使用`OnComponentBeginOverlap`函数。

更多关于`OnComponentBeginOverlap`函数以及需要提供给它的函数的信息，请参阅[`docs.unrealengine.com/latest/INT/API/%20Runtime/Engine/Components/UPrimitiveComponent/%20OnComponentBeginOverlap/index.html`](https://docs.unrealengine.com/latest/INT/API/%20Runtime/Engine/Components/UPrimitiveComponent/%20OnComponentBeginOverlap/index.html)。

在此之后，在我们的`OnBeginOverlap`函数内部，我们首先检查我们是否目前在服务器上。我们不希望事情被多次调用，并且我们希望服务器是告诉其他客户端我们已经增加了我们的分数的那个。

我们还调用了`UpdateScore`函数。此函数已添加以下函数指定符：

+   `Reliable`：该函数将通过网络复制，并确保它能够到达，无论网络错误或带宽问题。它要求我们选择`Client`或`Server`作为额外的指定符。

+   `Server`: 指定该函数只能在服务器上调用。它会在函数末尾添加一个名为 `_Implementation` 的附加函数，这是实现应该发生的地方。自动生成的代码将根据需要使用此函数。

+   `WithValidation`: 添加一个需要以 `_Validate` 结尾实现的附加函数。此函数将接受与给定函数相同的参数，但将返回一个布尔值，指示是否应该调用主函数。

关于其他函数指定符（如 `Unreliable`）的更多信息，请参阅 [`docs.unrealengine.com/en-US/Programming/UnrealArchitecture/Reference/Functions#functionspecifiers`](https://docs.unrealengine.com/en-US/Programming/UnrealArchitecture/Reference/Functions#functionspecifiers)。

调用 `UpdateScore` 将会依次调用我们创建的 `UpdateScore_Implementation` 函数，并显示一条消息，说明我们已经通过打印一些类似我们之前使用的文本收集了对象。

最后，需要 `UpdateScore_Validate` 函数，它只是告诉游戏我们应始终运行 `UpdateScore` 函数的实现。

对于一些关于性能和带宽设置的推荐，这些设置可能对处理大量复制的关卡很有用，请查看以下链接：[`docs.unrealengine.com/en-US/Gameplay/Networking/Actors/ReplicationPerformance`](https://docs.unrealengine.com/en-US/Gameplay/Networking/Actors/ReplicationPerformance)。

# 参见...

如果你想要查看使用网络和复制的另一个示例，请参阅 [`wiki.unrealengine.com/ Networking/Replication`](https://wiki.unrealengine.com/%20Networking/Replication)。

此外，你还可以查看 Unreal Engine 4 中包含的 Shooter Game 示例项目，并阅读文件以了解它在完整示例中的使用方式。要了解更多信息，请参阅 [`docs.unrealengine.com/en-us/Resources/SampleGames/ShooterGame`](https://docs.unrealengine.com/en-us/Resources/SampleGames/ShooterGame)。

# 处理 UI 网络事件

由于每个玩家都有自己的屏幕，因此他们的 UI 只会显示与他们相关的信息是有意义的。在这个配方中，我们将了解如何处理 UI 网络事件。

# 准备工作...

你应该完成本章中关于 *网络上的复制属性* 的配方，以及熟悉创建 HUD，你可以在第十四章 *用户界面 – UI 和 UMG* 中了解更多信息。

# 如何做...

1.  从你的 Visual Studio 项目（文件 | 打开 Visual Studio），打开 `Source\<Module>` 文件夹，然后从那里打开 `<Module>.build.cs` 文件（在我的情况下，将是 `Source\Chapter_12\Chapter_12.build.cs`），并取消注释/添加以下代码行：

```cpp
using UnrealBuildTool;

public class Chapter_12 : ModuleRules
{
  public Chapter_12(ReadOnlyTargetRules Target) : base(Target)
  {
    PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

    PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "HeadMountedDisplay" });

        PrivateDependencyModuleNames.AddRange(new string[] { "Slate", "SlateCore" });
    }
}
```

1.  使用 Add C++ Class 向导创建一个新的 `HUD` 子类：

![](img/7af1e6e1-6019-4ef6-9aa4-93e727b3893c.png)

1.  当被要求输入名称时，输入`NetworkHUD`，然后点击创建类按钮：

![图片](img/7409fcc2-be62-42bd-8a51-b50a2d9970f6.png)

1.  创建完成后，打开您计划使用的`GameMode`（我使用的是`Chapter_12GameMode.cpp`文件），并将以下内容添加到构造函数实现中：

```cpp
#include "Chapter_12GameMode.h"
#include "Chapter_12Character.h"
#include "NetworkHUD.h"
#include "UObject/ConstructorHelpers.h"

AChapter_12GameMode::AChapter_12GameMode()
{
  // set default pawn class to our Blueprinted character
  static ConstructorHelpers::FClassFinder<APawn> PlayerPawnBPClass(TEXT("/Game/ThirdPersonCPP/Blueprints/ThirdPersonCharacter"));
  if (PlayerPawnBPClass.Class != NULL)
  {
    DefaultPawnClass = PlayerPawnBPClass.Class;
  }

 HUDClass = ANetworkHUD::StaticClass();
}
```

1.  在`NetworkHUD.h`中，使用`override`关键字向类中添加以下函数：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/HUD.h"
#include "NetworkHUD.generated.h"

/**
 * 
 */
UCLASS()
class CHAPTER_12_API ANetworkHUD : public AHUD
{
    GENERATED_BODY()

public:
    virtual void DrawHUD() override;
};
```

1.  现在，实现该函数：

```cpp
#include "NetworkHUD.h"
#include "Engine/Canvas.h"
#include "Chapter_12Character.h"

void ANetworkHUD::DrawHUD()
{
 Super::DrawHUD();

 AChapter_12Character* PlayerCharacter = Cast<AChapter_12Character>(GetOwningPawn());

 if(PlayerCharacter)
 {
 Canvas->DrawText(GEngine->GetMediumFont(), FString::Printf(TEXT("Called Jump %d times!"), PlayerCharacter->JumpCount), 10, 10);
 }
}
```

1.  最后，我们可以取消注释原始调试消息，因为我们的 HUD 将为我们处理它：

```cpp
void AChapter_12Character::Jump()
{
    Super::Jump();

    JumpCount++;

 //if (Role == ROLE_Authority)
 //{
 // // Only print function once
 // GEngine->AddOnScreenDebugMessage(-1, 5.0f,
 // FColor::Green,
 // FString::Printf(TEXT("%s called Jump %d times!"), *GetName(), JumpCount));
 //}
}
```

1.  编译您的代码并启动编辑器。

1.  在编辑器中，从“设置”下拉菜单中打开“世界设置”面板：

![图片](img/a304cb34-ba25-4de7-b4d2-b76012b1d7d8.jpg)

1.  在“世界设置”对话框中，从“游戏模式覆盖”下的列表中选择`Chapter_12GameMode`：

![图片](img/10cffe4c-4ab4-4532-90f9-46b7331710a0.png)

1.  播放并验证您的自定义 HUD 是否已绘制到屏幕上，并且每个角色都有自己的跳跃值：

![图片](img/edabde91-00da-49d0-b5a1-a16ce3d0c00f.png)

有了这些概念，我们可以显示任何正在复制的属性！

# 它是如何工作的...

`GetOwningPawn`方法将返回一个指向 HUD 附加的`Pawn`类的指针。我们将它转换为我们的自定义角色派生类，然后可以访问该类具有的属性。在我们的例子中，我们使用的是之前添加了`Replicated`标签的变量，这使得 HUD 能够根据我们使用的屏幕正确更新。

更多信息和复制的使用示例，请查看[`wiki.unrealengine.com/Replication`](https://wiki.unrealengine.com/Replication)。

# 参见...

对于想要了解更多关于使用 Unreal Engine 4 进行网络编程的人来说，Cedric 'eXi' Neukirchen 创建了一个非常好的指南，我推荐阅读。您可以在[`cedric-neukirchen.net/Downloads/Compendium/UE4_Network_Compendium_by_Cedric_eXi_Neukirchen.pdf`](http://cedric-neukirchen.net/Downloads/Compendium/UE4_Network_Compendium_by_Cedric_eXi_Neukirchen.pdf)找到它。
