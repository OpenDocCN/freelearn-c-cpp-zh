# 处理事件和委托

在本章中，我们将涵盖以下示例：

+   处理通过虚拟函数实现的的事件

+   创建一个绑定到 UFUNCTION 的委托

+   取消注册委托

+   创建一个接受输入参数的委托

+   使用委托绑定传递有效负载数据

+   创建一个多播委托

+   创建一个自定义事件

+   创建一天中时间的处理程序

+   为第一人称射击游戏创建一个重生拾取物品

# 简介

Unreal 使用事件以高效的方式通知对象关于游戏世界中发生的事情。事件和委托对于确保这些通知可以以最小化类耦合的方式发布非常有用，并允许任意类订阅以接收通知。

# 技术要求

本章需要使用 Unreal Engine 4，并使用 Visual Studio 2017 作为 IDE。有关如何安装这两款软件及其要求的说明，请参阅本书第一章，*UE4 开发工具*，中的 Chapter 1。

# 处理通过虚拟函数实现的事件

Unreal 提供的一些 `Actor` 和 `Component` 类包含事件处理程序，形式为 `virtual` 函数。这个示例将向你展示如何通过覆盖相关的 `virtual` 函数来自定义这些处理程序。

# 如何实现...

1.  在编辑器中创建一个空的 `Actor`。命名为 `MyTriggerVolume`：

![图片](img/d1d05cdf-0c3c-4ae6-be67-354df71b2b43.png)

1.  将以下代码添加到类头文件中：

```cpp
UPROPERTY() 
UBoxComponent* TriggerZone; 

UFUNCTION() 
virtual void NotifyActorBeginOverlap(AActor* OtherActor) override; 

UFUNCTION() 
virtual void NotifyActorEndOverlap(AActor* OtherActor) override;
```

1.  因为我们在引用一个不属于我们项目的类，所以我们也需要添加一个 `#include`：

```cpp
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/BoxComponent.h"
#include "MyTriggerVolume.generated.h"
```

记得将包含文件放在 `.generated.h` 文件之上。

1.  将以下脚本添加到构造函数中以创建 `BoxComponent`。这将触发我们的事件：

```cpp
// Sets default values
AMyTriggerVolume::AMyTriggerVolume()
{
    // Set this actor to call Tick() every frame. You can turn 
   // this off to improve performance if you don't need it. 

    PrimaryActorTick.bCanEverTick = true;

 // Create a new component for the instance and initialize 
       it
 TriggerZone = CreateDefaultSubobject<UBoxComponent>("TriggerZone");
 TriggerZone->SetBoxExtent(FVector(200, 200, 100));
}
```

1.  将前面添加的额外函数的实现添加到 `.cpp` 文件中：

```cpp
void AMyTriggerVolume::NotifyActorBeginOverlap(AActor* OtherActor) 
{ 
    auto Message = FString::Printf(TEXT("%s entered me"), 
                   *(OtherActor->GetName()));

    GEngine->AddOnScreenDebugMessage(-1, 1, FColor::Red, Message); 
} 

void AMyTriggerVolume::NotifyActorEndOverlap(AActor* OtherActor) 
{ 
    auto Message = FString::Printf(TEXT("%s left me"), 
                   *(OtherActor->GetName()));

    GEngine->AddOnScreenDebugMessage(-1, 1, FColor::Red, Message); 
} 
```

1.  编译你的项目并将 `MyTriggerActor` 的实例放置到关卡中：

![图片](img/e8b4d60a-7163-4710-a5c9-2556e1be3566.png)

在关卡中放置触发体积

你应该能看到围绕对象的线条，指示碰撞发生的位置。请随意移动它，以及/或者根据需要调整属性，以便它们适合你想要的位置。

1.  然后，通过进入体积并查看屏幕上打印的输出，验证重叠/触摸事件是否被处理：

![图片](img/22bfbd2a-71b3-4cd8-be0c-5a2d0de42001.png)

# 它是如何工作的...

和往常一样，我们首先声明一个 `UPROPERTY` 来保存对组件子对象的引用。然后创建两个 `UFUNCTION` 声明。这些被标记为 `virtual` 和 `override`，以便编译器理解我们想要替换父实现，并且我们的函数实现可以被替换。

在对象的构造函数中，我们使用 `CreateDefaultSubobject` 函数创建子对象。之后，我们通过 `SetBoxExtent` 函数设置盒子的范围（大小），使用一个包含我们想要盒子具有的 *X*、*Y* 和 *Z* 大小的 `FVector`。

在函数的实现中，我们使用 `FString::Printf` 函数从一些预设文本创建一个 `FString` 并用一些数据参数替换。请注意，`Actor->GetName()` 函数也返回一个 `FString`，在传递给 `FString::Printf` 之前使用 `*` 运算符进行解引用。不这样做会导致错误。

然后，这个 `FString` 被传递给全局引擎函数 `AddOnScreenDebugMessage`，以便在屏幕上显示此信息。

`-1` 作为第一个参数告诉引擎允许重复字符串，第二个参数是消息应显示的秒数，第三个参数是颜色，第四个是实际要打印的字符串。虽然可以不创建额外的变量，直接在函数调用中放置信息，但这会使代码更难以阅读。

现在，当我们的演员组件与另一个对象重叠时，其 `UpdateOverlaps` 函数将调用 `NotifyActorBeginOverlap`，而虚拟函数调度将调用我们的自定义实现。

# 还有更多...

Unreal 的文档包含了所有内置类中所有变量和函数的信息。例如，`Actor` 类可以在以下链接中找到：[`api.unrealengine.com/INT/API/Runtime/Engine/GameFramework/AActor/index.html`](https://api.unrealengine.com/INT/API/Runtime/Engine/GameFramework/AActor/index.html)，如下面的截图所示：

![图片](img/1790133f-4584-4b4b-8fa6-3fa534aa72b7.png)

如果你滚动到这个配方中我们使用的函数，你可以看到更多关于它们的信息（请随意使用 *Ctrl* + *F* 来查找特定项目）：

![图片](img/9388a24b-1449-414c-bc5a-f8cb197f191f.png)

在最左侧的选项卡上，你会看到一些图标，其中包含蓝色圆圈和 V 的图标表示该函数是 `virtual` 的，可以在你的类中重写。如果你还没有机会，查看所有的事件也是一个好主意，这样你就可以熟悉已经包含的内容了。

# 创建一个绑定到 UFUNCTION 的代理

指针很棒，因为我们能够在运行时分配它们，并且可以更改它们在内存中指向的位置。除了标准类型之外，我们还可以创建指向函数的指针，但这些原始函数指针由于多种原因而不安全使用。委托是函数指针的一个更安全的版本，它为我们提供了在不知道哪个函数被分配直到调用时的灵活性。这种灵活性是相对于静态函数而言选择委托的主要原因之一。本配方展示了如何将一个 `UFUNCTION` 与委托关联，以便在委托执行时调用它。

# 准备工作

确保你已经遵循了之前的配方来创建 `MyTriggerVolume` 类，因为我们将使用它来调用我们的委托。

# 如何做到这一点...

1.  在我们的 Unreal 项目 `GameMode` 头文件中，使用以下宏声明委托，就在类声明之前：

```cpp
DECLARE_DELEGATE(FStandardDelegateSignature)
UCLASS()
class CHAPTER_05_API AChapter_05GameModeBase : public AGameModeBase
```

1.  向我们的游戏模式添加一个新成员：

```cpp
DECLARE_DELEGATE(FStandardDelegateSignature)
UCLASS()
class CHAPTER_05_API AChapter_05GameModeBase : public AGameModeBase
{
  GENERATED_BODY()

public:
 FStandardDelegateSignature MyStandardDelegate;
};
```

1.  创建一个新的 `Actor` 类，称为 `DelegateListener`：

![图片](img/44ea8a24-0a8b-447a-b937-7258cbb81cee.png)

1.  将以下内容添加到该类的声明中：

```cpp
UFUNCTION() 
void EnableLight(); 

UPROPERTY() 
UPointLightComponent* PointLight; 
```

1.  由于脚本不知道 `UPointLightComponent` 类，我们还需要在生成的 `.h` 文件上方添加以下新的 `#include`：

```cpp
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/PointLightComponent.h"
#include "DelegateListener.generated.h"
```

1.  在类的实现文件 `.cpp` 中，将以下粗体代码添加到构造函数中：

```cpp
ADelegateListener::ADelegateListener()
{
  // Set this actor to call Tick() every frame. You can turn this
 // off to improve performance if you don't need it.
 PrimaryActorTick.bCanEverTick = true;

  // Create a point light
 PointLight = CreateDefaultSubobject<UPointLightComponent>("PointLight");
 RootComponent = PointLight;

 // Turn it off at the beginning so we can turn it on later 
 //   through code 
 PointLight->SetVisibility(false);

 // Set the color to blue to make it easier to see
 PointLight->SetLightColor(FLinearColor::Blue);

}
```

1.  在 `ADelegateListener::BeginPlay` 的实现中添加以下代码：

```cpp
void ADelegateListener::BeginPlay()
{
    Super::BeginPlay();

 UWorld* TheWorld = GetWorld();
 if (TheWorld != nullptr)
 {
 AGameModeBase* GameMode = 
 UGameplayStatics::GetGameMode(TheWorld);

 AChapter_05GameModeBase * MyGameMode = 
 Cast<AChapter_05GameModeBase>(GameMode);

 if (MyGameMode != nullptr)
 {
 MyGameMode->MyStandardDelegate.BindUObject(this, 
 &ADelegateListener::EnableLight);
 }
 }

}
```

1.  由于脚本不知道 `UGameplayStatics` 类或我们在 `DelegateListener.cpp` 文件中的 `GameMode`，在文件中添加以下 #includes：

```cpp
#include "DelegateListener.h"
#include "Chapter_05GameModeBase.h"
#include "Kismet/GameplayStatics.h"
```

1.  最后，实现 `EnableLight`：

```cpp
void ADelegateListener::EnableLight() 
{ 
  PointLight->SetVisibility(true); 
} 
```

1.  将以下代码放入我们的 `TriggerVolume` 的 `NotifyActorBeginOverlap` 函数中：

```cpp
void AMyTriggerVolume::NotifyActorBeginOverlap(AActor* OtherActor) 
{ 
    auto Message = FString::Printf(TEXT("%s entered me"), 
                   *(OtherActor->GetName()));

    GEngine->AddOnScreenDebugMessage(-1, 1, FColor::Red, Message); 

    // Call our delegate
 UWorld* TheWorld = GetWorld();

 if (TheWorld != nullptr)
 {
 AGameModeBase* GameMode = 
                               UGameplayStatics::GetGameMode(TheWorld); 
 AChapter_05GameModeBase * MyGameMode = 
                               Cast<AChapter_05GameModeBase>(GameMode);

 if(MyGameMode != nullptr)
 {
 MyGameMode->MyStandardDelegate.ExecuteIfBound();
 }

 }
} 
```

1.  就像我们之前包含的那样，确保你也将以下代码添加到你的 CPP 文件中，这样编译器在我们使用它之前就会知道这个类：

```cpp
#include "MyTriggerVolume.h"
#include "Chapter_05GameModeBase.h"
#include "Kismet/GameplayStatics.h"
```

1.  编译你的游戏。确保你的游戏模式设置为当前关卡（如果你不知道如何设置，请参考第四章，*实例化 Actor 使用 SpawnActor*配方），并将你的 `MyTriggerVolume` 的副本拖到关卡中。同样，将 `DelegateListener` 的副本拖到关卡中，并将其放置在平坦表面上方大约 100 个单位的位置，以便在游戏播放时可以看到光线：

![图片](img/2342fe47-0609-4ba6-a7c8-f06b193ea181.png)

位于地面平面上的委托监听器

1.  当你按下播放并走进触发体积覆盖的区域时，你应该看到我们添加到 `DelegateListener` 中的 `PointLight` 组件被打开：

![图片](img/08d981f9-afb0-4702-93df-d93340ffdbc2.png)

# 它是如何工作的...

在我们的 `GameMode` 头文件中，我们声明了一个不接受任何参数的委托类型，称为 `FStandardDelegateSignature`。然后我们创建了一个委托实例，作为我们的 `GameModeBase` 类的一个成员。

我们在`DelegateListener`内部添加一个`PointLight`组件，以便我们有执行委托的可视表示，我们可以在以后将其打开。在构造函数中，我们初始化我们的`PointLight`，然后将其禁用。为了更容易看到，我们还将其颜色改为蓝色。

我们重写`BeginPlay`。我们首先调用父类的`BeginPlay()`实现。然后，我们获取游戏世界，使用`GetGameMode()`检索`GameMode`类。将结果`AGameMode*`转换为我们的`GameMode`类指针需要使用`Cast`模板函数。然后，我们可以访问`GameMode`的委托实例成员，并将我们的`EnableLight`函数绑定到委托，以便在委托执行时调用。在这种情况下，我们绑定到`UFUNCTION()`，因此使用`BindUObject`。

如果我们想要绑定到一个普通的 C++类函数，我们会使用`BindRaw`。如果我们想要绑定到一个静态函数，我们会使用`BindStatic()`。如果您使用这些，您必须非常小心，并在对象被销毁时手动解除绑定。这就像使用裸露的 C++内存分配一样。当涉及到 UE4 时，一般规则是在可能的情况下使用`UObject`。这可以节省很多麻烦！

当`TriggerVolume`与玩家重叠时，它检索`GameMode`，然后对委托调用`ExecuteIfBound`。`ExecuteIfBound`检查是否有函数绑定到委托，然后为我们调用它。`EnableLight`函数在由委托对象调用时启用`PointLight`组件。

# 相关内容

+   下一个配方，*注销委托*，展示了在`Listener`在委托被调用之前被销毁的情况下如何安全地注销您的委托绑定

要了解更多关于此以及与委托一起工作的其他选项，请查看[`docs.unrealengine.com/en-us/Programming/UnrealArchitecture/Delegates`](https://docs.unrealengine.com/en-us/Programming/UnrealArchitecture/Delegates)。

# 注销委托

有时候，需要移除一个委托绑定。这就像将函数指针设置为`nullptr`，这样它就不再引用已被删除的对象。

# 准备工作

您需要遵循前面的配方，以便您有一个可以注销的委托。

# 如何做...

1.  在`DelegateListener`类中，添加以下重写函数声明：

```cpp
UFUNCTION() 
virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
```

1.  实现函数如下：

```cpp
void ADelegateListener::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    Super::EndPlay(EndPlayReason);
    UWorld* TheWorld = GetWorld();

    if (TheWorld != nullptr)
    {
        AGameModeBase* GameMode = 
                               UGameplayStatics::GetGameMode(TheWorld);

        AChapter_05GameModeBase * MyGameMode = 
                               Cast<AChapter_05GameModeBase>(GameMode);

        if (MyGameMode != nullptr)
        {
            MyGameMode->MyStandardDelegate.Unbind();
        }
    }
}
```

# 它是如何工作的...

这个配方结合了到目前为止本章中的前两个配方。我们重写`EndPlay`，这是一个作为虚拟函数实现的的事件，这样我们就可以在`DelegateListener`离开游戏时执行代码。

在该重写实现中，我们调用委托上的`Unbind()`方法，这将解除成员函数与`DelegateListener`实例的链接。

如果不这样做，代表者就像一个指针一样悬空，当 `DelegateListener` 离开游戏时，将其置于无效状态。使用 `BindUObject()` 有助于避免大多数这些情况，但不幸的时机问题可能会导致对对象的调用被标记为销毁。即使在使用 `BindUObject()` 的情况下，手动解除代表者的绑定也是一个好的做法，因为这些时间错误导致的错误几乎无法追踪。

# 创建一个接受输入参数的代表者

到目前为止，我们使用的代表者没有接受任何输入参数。这个配方展示了如何更改代表者的签名，使其能够接受一些输入。

# 准备工作

确保你遵循了本章开头所示的配方，该配方展示了如何创建 `TriggerVolume` 和我们需要的其他基础设施。

# 如何做到...

1.  在 `GameMode` 中添加一个新的代表者声明：

```cpp
DECLARE_DELEGATE(FStandardDelegateSignature)
DECLARE_DELEGATE_OneParam(FParamDelegateSignature, FLinearColor) 
UCLASS()
class CHAPTER_05_API AChapter_05GameModeBase : public AGameModeBase
```

1.  向 `GameMode` 添加一个新成员：

```cpp
DECLARE_DELEGATE(FStandardDelegateSignature)
DECLARE_DELEGATE_OneParam(FParamDelegateSignature, FLinearColor)

UCLASS()
class CHAPTER_05_API AChapter_05GameModeBase : public AGameModeBase
{
  GENERATED_BODY()

public:
    FStandardDelegateSignature MyStandardDelegate;

    FParamDelegateSignature MyParameterDelegate;

};
```

1.  创建一个新的 `Actor` 类，名为 `ParamDelegateListener`。在声明中添加以下内容：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/PointLightComponent.h"
#include "ParamDelegateListener.generated.h"

UCLASS()
class CHAPTER_05_API AParamDelegateListener : public AActor
{
    GENERATED_BODY()

public: 
    // Sets default values for this actor's properties
    AParamDelegateListener();

 UFUNCTION()
 void SetLightColor(FLinearColor LightColor);

 UPROPERTY()
 UPointLightComponent* PointLight;

protected:
    // Called when the game starts or when spawned
    virtual void BeginPlay() override;

public: 
    // Called every frame
    virtual void Tick(float DeltaTime) override;

};
```

1.  在类实现中，在构造函数中添加以下内容：

```cpp
// Sets default values
AParamDelegateListener::AParamDelegateListener()
{
    // Set this actor to call Tick() every frame. You can turn this off 
    // to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = true;

 PointLight = CreateDefaultSubobject<UPointLightComponent>("PointLight");
 RootComponent = PointLight;

}
```

1.  在 `ParamDelegateListener.cpp` 文件中，添加以下 `#includes`：

```cpp
#include "ParamDelegateListener.h"
#include "Chapter_05GameModeBase.h"
#include "Kismet/GameplayStatics.h"
```

1.  在 `AParamDelegateListener::BeginPlay` 的实现中，添加以下代码：

```cpp
// Called when the game starts or when spawned
void AParamDelegateListener::BeginPlay()
{
    Super::BeginPlay();

    UWorld* TheWorld = GetWorld();

    if (TheWorld != nullptr)
    {
        AGameModeBase* GameMode = 
                               UGameplayStatics::GetGameMode(TheWorld);

        AChapter_05GameModeBase * MyGameMode = 
                               Cast<AChapter_05GameModeBase>(GameMode);

        if (MyGameMode != nullptr)
        {
            MyGameMode->MyParameterDelegate.BindUObject(this, 
                               &AParamDelegateListener::SetLightColor);
        }
    }

}
```

1.  最后，实现 `SetLightColor`：

```cpp
void AParamDelegateListener::SetLightColor(FLinearColor LightColor)
{
    PointLight->SetLightColor(LightColor);
}
```

1.  在我们的 `TriggerVolume` 中，在 `NotifyActorBeginOverlap` 中添加以下新代码：

```cpp
 void AMyTriggerVolume::NotifyActorBeginOverlap(AActor* OtherActor) 
{ 
    auto Message = FString::Printf(TEXT("%s entered me"), 
                   *(OtherActor->GetName()));

    GEngine->AddOnScreenDebugMessage(-1, 1, FColor::Red, Message); 

    // Call our delegate
    UWorld* TheWorld = GetWorld();

    if (TheWorld != nullptr)
    {
        AGameModeBase* GameMode = 
                               UGameplayStatics::GetGameMode(TheWorld); 
        AChapter_05GameModeBase * MyGameMode = 
                               Cast<AChapter_05GameModeBase>(GameMode);

        if(MyGameMode != nullptr)
        {
            MyGameMode->MyStandardDelegate.ExecuteIfBound();

            // Call the function using a parameter
 auto Color = FLinearColor(1, 0, 0, 1);
 MyGameMode->MyParameterDelegate.ExecuteIfBound(Color);
        }

    }
}
```

1.  保存你的脚本，然后回到 Unreal 编辑器并编译你的代码。将 `MyTriggerVolume` 和 `ParamDelegateListener` 对象添加到场景中，然后运行游戏并确认灯光最初是白色的，并且在碰撞后变为红色：

![图片](img/64fa8763-941d-4e56-8d7f-3ca69f499653.png)

# 它是如何工作的...

我们的新代表者签名使用了一个稍微不同的宏来声明。注意 `DECLARE_DELEGATE_OneParam` 结尾的 `_OneParam` 后缀。正如你所期望的，我们还需要指定我们的参数类型。就像我们创建一个没有参数的代表者一样，我们可以将代表者实例化为我们 `GameMode` 类的一个成员。

代表者签名可以是全局或类作用域，但不能是函数体（如文档所述）。你可以在 [`docs.unrealengine.com/en-US/Programming/UnrealArchitecture/Delegates`](https://docs.unrealengine.com/en-US/Programming/UnrealArchitecture/Delegates) 找到更多关于此的信息。

然后，我们创建了一种新的 `DelegateListener` 类型：它期望将参数传递到它绑定到代表者的函数中。当我们为代表者调用 `ExecuteIfBound()` 方法时，我们需要传递将插入函数参数的值。

在我们绑定的函数内部，我们使用参数来设置我们灯光的颜色。这意味着`TriggerVolume`不需要了解`ParamDelegateListener`的任何信息就可以调用它的函数。委托使我们能够最小化两个类之间的耦合。

# 参见

+   *注销委托*的配方展示了在监听器在委托被调用之前被销毁的情况下如何安全地注销你的委托绑定

# 使用委托绑定传递有效载荷数据

只需进行最小更改，参数就可以在创建时传递给委托。本配方展示了如何指定始终作为参数传递给委托调用的数据。数据在绑定创建时计算，并且从那时起不再改变。

# 准备工作

确保你已经遵循了之前的配方。我们将扩展之前配方的功能，将额外的创建时参数传递给我们的绑定委托函数。

# 如何做到这一点...

1.  在你的`AParamDelegateListener::BeginPlay`函数中，将`BindUObject`的调用更改为以下内容：

```cpp
// Called when the game starts or when spawned
void AParamDelegateListener::BeginPlay()
{
    Super::BeginPlay();

    UWorld* TheWorld = GetWorld();

    if (TheWorld != nullptr)
    {
        AGameModeBase* GameMode = 
                               UGameplayStatics::GetGameMode(TheWorld);

        AChapter_05GameModeBase * MyGameMode = 
                               Cast<AChapter_05GameModeBase>
                               (GameMode);

        if (MyGameMode != nullptr)
        {
            MyGameMode->MyParameterDelegate.BindUObject(this, 
                        &AParamDelegateListener::SetLightColor, 
                        false);
        }
    }

}
```

1.  在`ParamDelegateListener.h`文件中，将`SetLightColor`的声明更改为以下内容：

```cpp
UCLASS()
class CHAPTER_05_API AParamDelegateListener : public AActor
{
    GENERATED_BODY()

public: 
    // Sets default values for this actor's properties
    AParamDelegateListener();

    UFUNCTION()
    void SetLightColor(FLinearColor LightColor, bool EnableLight);

    UPROPERTY()
    UPointLightComponent* PointLight;

protected:
    // Called when the game starts or when spawned
    virtual void BeginPlay() override;

public: 
    // Called every frame
    virtual void Tick(float DeltaTime) override;

};
```

1.  按照以下方式修改`SetLightColor`的实现：

```cpp
void AParamDelegateListener::SetLightColor(FLinearColor LightColor, bool EnableLight)
{
    PointLight->SetLightColor(LightColor);
    PointLight->SetVisibility(EnableLight);
}
```

1.  编译并运行你的项目。验证当你走进`TriggerVolume`时，灯光因为你在绑定函数时传入的`false`有效载荷参数而关闭。

# 它是如何工作的...

当我们将函数绑定到委托时，我们指定一些额外的数据（在这种情况下，一个值为`false`的布尔值）。你可以以这种方式传递最多四个“有效载荷”变量。它们在你在`DECLARE_DELEGATE_*`宏中声明的任何参数之后应用于你的函数。

我们更改委托的函数签名，使其能够接受额外的参数。

在函数内部，我们使用额外的参数根据编译时`true`或`false`的值来打开或关闭灯光。

我们不需要更改对`ExecuteIfBound`的调用：委托系统会自动应用通过`ExecuteIfBound`传入的委托参数，然后应用任何有效载荷参数，这些参数总是在调用`BindUObject`时在函数引用之后指定。

# 参见

+   *注销委托*的配方展示了在监听器在委托被调用之前被销毁的情况下如何安全地注销你的委托绑定

# 创建一个多播委托

我们在本章中使用的标准委托本质上是指针函数：它们允许你在特定的对象实例上调用特定的函数。多播委托是一组函数指针，每个指针可能指向不同的对象，当委托被**广播**时，所有这些函数指针都将被调用。

# 准备工作

本食谱假设你已经遵循了本章的初始食谱，即 *通过虚拟函数实现的事件处理*，因为它展示了如何创建 `TriggerVolume`，该 `TriggerVolume` 用于广播多播委托。

# 如何操作...

1.  在 `GameMode` 头文件中添加一个新的委托声明：

```cpp
DECLARE_DELEGATE(FStandardDelegateSignature)
DECLARE_DELEGATE_OneParam(FParamDelegateSignature, FLinearColor)
DECLARE_MULTICAST_DELEGATE(FMulticastDelegateSignature) 
UCLASS()
class CHAPTER_05_API AChapter_05GameModeBase : public AGameModeBase
{
  GENERATED_BODY()

public:
    FStandardDelegateSignature MyStandardDelegate;

    FParamDelegateSignature MyParameterDelegate;

    FMulticastDelegateSignature MyMulticastDelegate;

};
```

1.  创建一个名为 `MulticastDelegateListener` 的新 `Actor` 类。在声明中添加以下内容：

```cpp
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/PointLightComponent.h"
#include "MulticastDelegateListener.generated.h"

UCLASS()
class CHAPTER_05_API AMulticastDelegateListener : public AActor
{
  GENERATED_BODY()

public: 
  // Sets default values for this actor's properties
  AMulticastDelegateListener();

    UFUNCTION()
 void ToggleLight();

 UFUNCTION()
 virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) 
    override;

 UPROPERTY()
 UPointLightComponent* PointLight;

 FDelegateHandle MyDelegateHandle;

protected:
  // Called when the game starts or when spawned
  virtual void BeginPlay() override;

public: 
  // Called every frame
  virtual void Tick(float DeltaTime) override;

};
```

1.  在类实现中，将以下代码添加到构造函数中：

```cpp
// Sets default values
AMulticastDelegateListener::AMulticastDelegateListener()
{
   // Set this actor to call Tick() every frame. You can turn this 
   //off to improve performance if you don't need it.
  PrimaryActorTick.bCanEverTick = true;

 PointLight = CreateDefaultSubobject<UPointLightComponent>
    ("PointLight");
 RootComponent = PointLight;
}
```

1.  在 `MulticastDelegateListener.cpp` 文件中，添加以下 `#includes`：

    ```cpp
    #include "MulticastDelegateListener.h"
    #include "Chapter_05GameModeBase.h"
    #include "Kismet/GameplayStatics.h"
    ```

    1.  在 `MulticastDelegateListener::BeginPlay` 的实现中，添加以下代码：

    ```cpp
    // Called when the game starts or when spawned
    void AMulticastDelegateListener::BeginPlay()
    {
      Super::BeginPlay();

     UWorld* TheWorld = GetWorld();

     if (TheWorld != nullptr)
     {
     AGameModeBase* GameMode = 
     UGameplayStatics::GetGameMode(TheWorld);

     AChapter_05GameModeBase * MyGameMode = 
     Cast<AChapter_05GameModeBase>
                                   (GameMode);

     if (MyGameMode != nullptr)
     {
     MyDelegateHandle = MyGameMode-
                >MyMulticastDelegate.AddUObject(this, 
                &AMulticastDelegateListener::ToggleLight);
     }
     }

    }
    ```

    1.  实现 `ToggleLight`：

    ```cpp
    void AMulticastDelegateListener::ToggleLight() 
    { 
      PointLight->ToggleVisibility(); 
    } 
    ```

    1.  实现我们的 `EndPlay` 覆盖函数：

    ```cpp
    void AMulticastDelegateListener::EndPlay (const EEndPlayReason::Type EndPlayReason)
    {
        Super::EndPlay(EndPlayReason);

        UWorld* TheWorld = GetWorld();

        if (TheWorld != nullptr)
        {
            AGameModeBase* GameMode = 
                                   UGameplayStatics::GetGameMode(TheWorld);

            AChapter_05GameModeBase * MyGameMode = 
                                   Cast<AChapter_05GameModeBase>
                                   (GameMode);

            if (MyGameMode != nullptr)
            {
                MyGameMode-
                >MyMulticastDelegate.Remove(MyDelegateHandle);
            }
        }

    }
    ```

    1.  将以下行添加到 `TriggerVolume::NotifyActorBeginOverlap()`：

    ```cpp
    MyGameMode->MyMulticastDelegate.Broadcast(); 
    ```

    1.  编译并加载你的项目。将你的关卡中的 `GameMode` 设置为我们的食谱游戏模式，然后将四个或五个 `MulticastDelegateListener` 实例拖入场景。

    1.  进入 `TriggerVolume` 以查看所有 `MulticastDelegateListener` 切换它们的光的可见性：

    ![](img/f53635e5-ef24-4ad9-beaa-9c3c07556083.png)

    # 它是如何工作的...

    如你所预期，委托类型需要显式声明为多播委托，而不是标准单绑定委托。我们新的 `Listener` 类与我们原始的 `DelegateListener` 非常相似。主要区别是我们需要在 `FDelegateHandle` 中存储对委托实例的引用。

    当演员被销毁时，我们使用存储的 `FDelegateHandle` 作为参数调用 `Remove()`，安全地从绑定到委托的函数列表中移除自己。

    `Broadcast()` 函数是 `ExecuteIfBound()` 的多播等效函数。与标准委托不同，无需提前或通过调用例如 `ExecuteIfBound` 的方式检查委托是否已绑定。无论绑定多少函数，甚至如果没有绑定任何函数，`Broadcast()` 都是安全的。

    当场景中有多个我们的多播监听器实例时，它们会分别将自己注册到在 `GameMode` 中实现的那个多播委托。然后，当 `TriggerVolume` 与玩家重叠时，它会广播委托，每个监听器都会收到通知，导致它切换其关联点光的可见性。

    多播委托可以像标准委托一样接受参数。

    # 创建自定义事件

    自定义委托非常有用，但它们的局限性之一是它们可以被某些其他第三方类外部广播；也就是说，它们的 `Execute`/`Broadcast` 方法是公开可访问的。

    有时，你可能需要一个可以被其他类外部分配的委托，但只能由包含它们的类广播。这是事件的主要目的。

    # 准备工作

    确保你已经遵循了本章的初始食谱，以便你有 `MyTriggerVolume` 和 `Chapter_05GameModeBase` 的实现。

    # 如何操作...

    1.  将以下事件声明宏添加到 `MyTriggerVolume` 类的头部:

    ```cpp
    DECLARE_EVENT(AMyTriggerVolume, FPlayerEntered) 
    ```

    1.  将声明的事件签名的一个实例添加到类中:

    ```cpp
    FPlayerEntered OnPlayerEntered; 
    ```

    1.  在 `AMyTriggerVolume::NotifyActorBeginOverlap` 中添加以下代码:

    ```cpp
    OnPlayerEntered.Broadcast();
    ```

    1.  创建一个新的 `Actor` 类称为 `TriggerVolEventListener`:

    ![图片](img/e1b64847-e6d5-4921-b15a-d3b256e83a55.png)

    1.  将以下类成员添加到其声明中:

    ```cpp
    #pragma once

    #include "CoreMinimal.h"
    #include "GameFramework/Actor.h"
    #include "Components/PointLightComponent.h"
    #include "MyTriggerVolume.h"
    #include "TriggerVolEventListener.generated.h"

    UCLASS()
    class CHAPTER_05_API ATriggerVolEventListener : public AActor
    {
        GENERATED_BODY()

    public: 
        // Sets default values for this actor's properties
        ATriggerVolEventListener();

        UPROPERTY()
     UPointLightComponent* PointLight;

     UPROPERTY(EditAnywhere)
     AMyTriggerVolume* TriggerEventSource;

     UFUNCTION()
     void OnTriggerEvent();

    protected:
        // Called when the game starts or when spawned
        virtual void BeginPlay() override;

    public: 
        // Called every frame
        virtual void Tick(float DeltaTime) override;

    };
    ```

    1.  在类构造函数中初始化 `PointLight`:

    ```cpp
    // Sets default values
    ATriggerVolEventListener::ATriggerVolEventListener()
    {
        // Set this actor to call Tick() every frame. You can turn this 
        //off to improve performance if you don't need it.
        PrimaryActorTick.bCanEverTick = true;

     PointLight = CreateDefaultSubobject<UPointLightComponent>
        ("PointLight");
     RootComponent = PointLight;
    }
    ```

    1.  在 `BeginPlay` 中添加以下代码:

    ```cpp
    // Called when the game starts or when spawned
    void ATriggerVolEventListener::BeginPlay()
    {
        Super::BeginPlay();

     if (TriggerEventSource != nullptr)
     {
     TriggerEventSource->OnPlayerEntered.AddUObject(this,            
            &ATriggerVolEventListener::OnTriggerEvent);
     }

    }
    ```

    1.  最后，实现 `OnTriggerEvent()`:

    ```cpp
    void ATriggerVolEventListener::OnTriggerEvent() 
    { 
      PointLight->SetLightColor(FLinearColor(0, 1, 0, 1)); 
    }
    ```

    1.  编译你的项目并启动编辑器。创建一个将游戏模式设置为我们的 `Chapter_05GameModeBase` 的关卡，然后将 `ATriggerVolEventListener` 和 `AMyTriggerVolume` 的实例拖放到关卡中。

    1.  选择 `TriggerVolEventListener`，你会在详情面板中看到它作为一个类别列出，属性为触发事件源:

    ![图片](img/d6cf0cd5-d3c1-493a-82ef-d8d32cf5b8ba.png)

    触发体积事件监听器类别

    1.  使用下拉菜单选择你的 `AMyTriggerVolume` 实例，以便监听器知道绑定哪个事件:

    ![图片](img/b26436bd-dbff-45ea-a696-fe24e5d2b01a.png)

    1.  播放你的游戏并进入触发区域的效应区域。验证你的 `EventListener` 的颜色是否变为绿色:

    ![图片](img/018dd943-a7f0-49a1-8002-9f0a09250bc2.png)

    # 它是如何工作的...

    与所有其他类型的委托一样，事件需要它们自己的特殊宏函数。第一个参数是事件将要实现的类。这将将是唯一能够调用 `Broadcast()` 的类，所以请确保它是正确的。第二个参数是我们新事件函数签名的类型名称。我们向我们的类添加这个类型的实例。Unreal 文档建议使用 `On<x>` 作为命名约定。

    当有东西与我们的 `TriggerVolume` 发生重叠时，我们在自己的事件实例上调用 `Broadcast()`。在新的类中，我们创建一个点光源作为触发事件的可视表示。

    我们还创建了一个指向 `TriggerVolume` 的指针来监听事件。我们将 `UPROPERTY` 标记为 `EditAnywhere`，因为这允许我们在编辑器中设置它，而不是必须通过 `GetAllActorsOfClass` 或其他方式程序化地获取引用。

    最后是我们的当有东西进入 `TriggerVolume` 的事件处理器。我们像往常一样在构造函数中创建和初始化我们的点光源。当游戏开始时，监听器会检查我们的 `TriggerVolume` 引用是否有效，然后将我们的 `OnTriggerEvent` 函数绑定到 `TriggerVolume` 事件。在 `OnTriggerEvent` 中，我们改变我们的灯光颜色为绿色。当有东西进入 `TriggerVolume` 时，它会导致 `TriggerVolume` 在其自己的事件上调用广播。然后我们的 `TriggerVolEventListener` 调用其绑定方法，改变我们的灯光颜色。

    # 创建一个“一天中的时间”处理器

    本食谱展示了如何使用在前一个食谱中介绍的概念来创建一个通知其他演员游戏内时间流逝的演员。

    # 如何操作...

    1.  创建一个新的 `Actor` 类，命名为 `TimeOfDayHandler`，如下截图所示：

    ![图片](img/67f4437f-f3c4-481a-88de-3544020d0d69.png)

    1.  在头文件中添加一个多播委托声明：

    ```cpp
    DECLARE_MULTICAST_DELEGATE_TwoParams(FOnTimeChangedSignature, int32, 
                                         int32)
    UCLASS()
    class CHAPTER_05_API ATimeOfDayHandler : public AActor
    ```

    1.  在 `public` 部分将我们的委托实例添加到类声明中：

    ```cpp
    FOnTimeChangedSignature OnTimeChanged; 
    ```

    1.  将以下属性添加到类中：

    ```cpp
    UPROPERTY() 
    int32 TimeScale; 

    UPROPERTY() 
    int32 Hours; 
    UPROPERTY() 
    int32 Minutes; 

    UPROPERTY() 
    float ElapsedSeconds; 
    ```

    1.  将这些属性的初始化添加到构造函数中：

    ```cpp
    // Sets default values
    ATimeOfDayHandler::ATimeOfDayHandler()
    {
        // Set this actor to call Tick() every frame. You can turn this   
       //  off to improve performance if you don't need it.
        PrimaryActorTick.bCanEverTick = true;

     TimeScale = 60;
     Hours = 0;
     Minutes = 0;
     ElapsedSeconds = 0;
    }
    ```

    1.  在 `Tick` 中添加以下代码：

    ```cpp
    // Called every frame
    void ATimeOfDayHandler::Tick(float DeltaTime)
    {
        Super::Tick(DeltaTime);

     ElapsedSeconds += (DeltaTime * TimeScale);

     if(ElapsedSeconds > 60)
     {
     ElapsedSeconds -= 60;
     Minutes++;

     if (Minutes > 60)
     {
     Minutes -= 60;
     Hours++;
     }

     OnTimeChanged.Broadcast(Hours, Minutes);
     }

    }
    ```

    1.  创建一个新的 `Actor` 类，命名为 `Clock`，如下截图所示：

    ![图片](img/45d5f742-3133-4c83-8c99-94081faaf7de.png)

    1.  将以下属性添加到类头中：

    ```cpp
    #pragma once

    #include "CoreMinimal.h"
    #include "GameFramework/Actor.h"
    #include "Clock.generated.h"

    UCLASS()
    class CHAPTER_05_API AClock : public AActor
    {
        GENERATED_BODY()

    public: 
        // Sets default values for this actor's properties
        AClock();

     UPROPERTY()
     USceneComponent* RootSceneComponent;

     UPROPERTY()
     UStaticMeshComponent* ClockFace;

     UPROPERTY()
     USceneComponent* HourHandle;

     UPROPERTY()
     UStaticMeshComponent* HourHand;

     UPROPERTY()
     USceneComponent* MinuteHandle;

     UPROPERTY()
     UStaticMeshComponent* MinuteHand;

     UFUNCTION()
     void TimeChanged(int32 Hours, int32 Minutes);

     FDelegateHandle MyDelegateHandle;

    protected:
        // Called when the game starts or when spawned
        virtual void BeginPlay() override;

    public: 
        // Called every frame
        virtual void Tick(float DeltaTime) override;

    };
    ```

    1.  在构造函数中初始化和变换组件：

    ```cpp
    #include "TimeOfDayHandler.h"
    #include "Kismet/GameplayStatics.h"

    // Sets default values
    AClock::AClock()
    {
        // Set this actor to call Tick() every frame. You can turn this off to improve performance if you don't need it.
        PrimaryActorTick.bCanEverTick = true;

        RootSceneComponent = CreateDefaultSubobject<USceneComponent>("RootSceneComponent");
        ClockFace = CreateDefaultSubobject<UStaticMeshComponent>("ClockFace"); 
        HourHand = CreateDefaultSubobject<UStaticMeshComponent>("HourHand"); 
        MinuteHand = CreateDefaultSubobject<UStaticMeshComponent>("MinuteHand"); 
        HourHandle = CreateDefaultSubobject<USceneComponent>("HourHandle"); 
        MinuteHandle = CreateDefaultSubobject<USceneComponent>("MinuteHandle"); 

        auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>(TEXT("StaticMesh'/Engine/BasicShapes/Cylinder.Cylinder'")); 

        if (MeshAsset.Object != nullptr)
        {
            ClockFace->SetStaticMesh(MeshAsset.Object);
            HourHand->SetStaticMesh(MeshAsset.Object);
            MinuteHand->SetStaticMesh(MeshAsset.Object);
        }

        RootComponent = RootSceneComponent;

        HourHand->AttachToComponent(HourHandle, 
                         FAttachmentTransformRules::KeepRelativeTransform);

        MinuteHand->AttachToComponent(MinuteHandle, 
                         FAttachmentTransformRules::KeepRelativeTransform);

        HourHandle->AttachToComponent(RootSceneComponent, 
                         FAttachmentTransformRules::KeepRelativeTransform);

        MinuteHandle->AttachToComponent(RootSceneComponent, 
                         FAttachmentTransformRules::KeepRelativeTransform);

        ClockFace->AttachToComponent(RootSceneComponent, 
                         FAttachmentTransformRules::KeepRelativeTransform);

        ClockFace->SetRelativeTransform(FTransform(FRotator(90, 0, 0), 
                                        FVector(10, 0, 0), 
                                        FVector(2, 2, 0.1)));

        HourHand->SetRelativeTransform(FTransform(FRotator(0, 0, 0), 
                                       FVector(0, 0, 25), 
                                       FVector(0.1, 0.1, 0.5)));

        MinuteHand->SetRelativeTransform(FTransform(FRotator(0, 0, 0), 
                                         FVector(0, 0, 50), 
                                         FVector(0.1, 0.1, 1)));

    }
    ```

    1.  将以下代码添加到 `BeginPlay`：

    ```cpp
    // Called when the game starts or when spawned
    void AClock::BeginPlay()
    {
        Super::BeginPlay();

     TArray<AActor*> TimeOfDayHandlers; 
     UGameplayStatics::GetAllActorsOfClass(GetWorld(), 
                                          ATimeOfDayHandler::StaticClass(), 
                                              TimeOfDayHandlers); 
     if (TimeOfDayHandlers.Num() != 0)
     {
     auto TimeOfDayHandler = Cast<ATimeOfDayHandler>
            (TimeOfDayHandlers[0]);
     MyDelegateHandle =  
            TimeOfDayHandler->OnTimeChanged.AddUObject(this, 
            &AClock::TimeChanged);
     }

    }
    ```

    1.  最后，实现 `TimeChanged` 作为你的事件处理程序：

    ```cpp
    void AClock::TimeChanged(int32 Hours, int32 Minutes)
    {
        HourHandle->SetRelativeRotation(FRotator(0, 0, 30 * Hours));
        MinuteHandle->SetRelativeRotation(FRotator(0, 0, 6 * Minutes));
    }
    ```

    1.  在你的级别中放置一个 `TimeOfDayHandler` 和 `AClock` 的实例。然后播放它，以查看时钟的指针旋转：

    ![图片](img/5441e608-52a9-4943-b354-09481e9006cb.png)

    # 工作原理...

    `TimeOfDayHandler` 包含一个接受两个参数的委托，因此使用了宏的 `TwoParams` 变体。我们的类包含存储小时、分钟和秒的变量，以及 `TimeScale`，这是一个用于测试目的加速时间的系数。在处理程序的 `Tick` 函数内部，我们根据自上一帧以来经过的时间累积已过秒数。我们检查已过秒数是否超过 60。如果是这样，我们减去 60，并增加 `Minutes`。同样，对于 `Minutes`：如果它们超过 60，我们减去 60，并增加 `Hours`。如果 `Minutes` 或 `Hours` 已更新，我们广播我们的委托，让任何已订阅委托的对象知道时间已更改。

    `Clock` 演员使用一系列场景组件和静态网格来构建一个类似于时钟面的网格层次结构。在 `Clock` 构造函数中，我们将层次结构中的组件设置为父级，并设置它们的初始比例和旋转。在 `BeginPlay` 中，时钟使用 `GetAllActorsOfClass()` 获取级别中所有的一天时间处理程序。如果级别中至少有一个 `TimeOfDayHandler`，则 `Clock` 访问第一个，并订阅其 `TimeChanged` 事件。当 `TimeChanged` 事件触发时，时钟根据当前设置的小时和分钟旋转时针和分针。

    # 为第一人称射击游戏创建一个可重新生成的拾取物品

    本食谱展示了如何创建一个在一定时间后重新生成的拾取物品，适合作为**第一人称射击游戏**（**FPS**）的弹药或其他拾取物品。

    # 如何操作...

    1.  创建一个新的 `Actor` 类，命名为 `Pickup`：

    ![图片](img/228a0cbd-6875-46c2-b5e4-3a5a883b9a4a.png)

    1.  在 `Pickup.h` 中声明以下委托类型：

    ```cpp
    DECLARE_DELEGATE(FPickedupEventSignature) 
    ```

    1.  将以下属性添加到类头中：

    ```cpp
    // Fill out your copyright notice in the Description page of Project Settings.

    #pragma once

    #include "CoreMinimal.h"
    #include "GameFramework/Actor.h"
    #include "GameFramework/RotatingMovementComponent.h"
    #include "Pickup.generated.h"

    DECLARE_DELEGATE(FPickedupEventSignature)
    UCLASS()
    class CHAPTER_05_API APickup : public AActor
    {
        GENERATED_BODY()

    public: 
        // Sets default values for this actor's properties
        APickup();

     virtual void NotifyActorBeginOverlap(AActor* OtherActor) 
        override;

     UPROPERTY()
     UStaticMeshComponent* MyMesh;

     UPROPERTY()
     URotatingMovementComponent* RotatingComponent;

     FPickedupEventSignature OnPickedUp;

    protected:
        // Called when the game starts or when spawned
        virtual void BeginPlay() override;

    public: 
        // Called every frame
        virtual void Tick(float DeltaTime) override;

    };

    ```

    1.  该类将使用`ConstructorHelpers`结构体，因此我们需要将以下`#include`添加到`Pickup.cpp`文件中：

    ```cpp
    #include "ConstructorHelpers.h"
    ```

    1.  将以下代码添加到构造函数中：

    ```cpp
    // Sets default values
    APickup::APickup()
    {
       // Set this actor to call Tick() every frame. You can turn this  
       // off to improve performance if you don't need it.
      PrimaryActorTick.bCanEverTick = true;

        MyMesh = CreateDefaultSubobject<UStaticMeshComponent>("MyMesh");

     RotatingComponent = 
        CreateDefaultSubobject<URotatingMovementComponent>
        ("RotatingComponent");
     RootComponent = MyMesh;

     auto MeshAsset = ConstructorHelpers::FObjectFinder<UStaticMesh>
        (TEXT("StaticMesh'/Engine/BasicShapes/Cube.Cube'"));

     if (MeshAsset.Object != nullptr)
     {
     MyMesh->SetStaticMesh(MeshAsset.Object);
     }

     MyMesh->SetCollisionProfileName(TEXT("OverlapAllDynamic"));
     RotatingComponent->RotationRate = FRotator(10, 0, 10);
    }
    ```

    1.  实现`NotifyActorBeginOverlap`的覆盖版本：

    ```cpp
    void APickup::NotifyActorBeginOverlap(AActor* OtherActor)
    {
        OnPickedUp.ExecuteIfBound();
    }
    ```

    1.  创建一个名为`PickupSpawner`的第二个`Actor`类：

    ![](img/033b1225-911b-43b4-aa47-c543e95f44b4.png)

    1.  将以下代码添加到类头文件中：

    ```cpp
    #pragma once

    #include "CoreMinimal.h"
    #include "GameFramework/Actor.h"
    #include "Pickup.h"
    #include "PickupSpawner.generated.h"

    UCLASS()
    class CHAPTER_05_API APickupSpawner : public AActor
    {
        GENERATED_BODY()

    public: 
        // Sets default values for this actor's properties
        APickupSpawner();

       UPROPERTY() 
     USceneComponent* SpawnLocation; 

     UFUNCTION() 
     void PickupCollected(); 

     UFUNCTION() 
     void SpawnPickup(); 

     UPROPERTY() 
     APickup* CurrentPickup; 

     FTimerHandle MyTimer;

    protected:
        // Called when the game starts or when spawned
        virtual void BeginPlay() override;

    public: 
        // Called every frame
        virtual void Tick(float DeltaTime) override;

    };

    ```

    1.  在构造函数中初始化我们的根组件：

    ```cpp
    SpawnLocation = 
     CreateDefaultSubobject<USceneComponent>("SpawnLocation");
    ```

    1.  在游戏开始时使用`BeginPlay`中的`SpawnPickup`函数生成拾取物：

    ```cpp
    // Called when the game starts or when spawned
    void APickupSpawner::BeginPlay()
    {
        Super::BeginPlay();

        SpawnPickup();

    }
    ```

    1.  实现`PickupCollected`：

    ```cpp
    void APickupSpawner::PickupCollected()
    {
        GetWorld()->GetTimerManager().SetTimer(MyTimer, 
                                               this, 
                                              &APickupSpawner::SpawnPickup, 
                                              10, 
                                              false);

        CurrentPickup->OnPickedUp.Unbind();
        CurrentPickup->Destroy();
    }
    ```

    1.  为`SpawnPickup`创建以下代码：

    ```cpp
    void APickupSpawner::SpawnPickup()
    {
        UWorld* MyWorld = GetWorld();

        if (MyWorld != nullptr) 
        {
            CurrentPickup = MyWorld->SpawnActor<APickup>( 
                                                    APickup::StaticClass(), 
                                                         GetTransform());

            CurrentPickup->OnPickedUp.BindUObject(this, 
                                         &APickupSpawner::PickupCollected);
        }
    }
    ```

    1.  编译并启动编辑器，然后将`PickupSpawner`实例拖放到关卡中。走进由旋转的立方体表示的拾取物，并验证它在 10 秒后再次生成：

    ![](img/b42458a5-3e4d-4621-a7a9-c0b284280dff.png)

    完成配方后的结果

    # 它是如何工作的...

    如同往常一样，我们需要在`Pickup`内部创建一个委托，以便我们的生成器可以订阅，这样它就知道玩家何时收集拾取物。`Pickup`还包含一个静态网格作为视觉表示，以及一个`RotatingMovementComponent`，以便网格会旋转以吸引玩家的注意。在`Pickup`构造函数中，我们加载引擎内置的一个网格作为我们的视觉表示。我们指定网格将与其他对象重叠，然后在*X*和*Z*轴上将网格的旋转速率设置为每秒 10 个单位。当玩家与`Pickup`重叠时，它从第一步触发其`PickedUp`委托。

    使用的`PickupSpawner`：

    一个场景组件来指定拾取物演员的生成位置。它有一个执行此操作的功能，以及一个标记的`UPROPERTY`引用到当前生成的`Pickup`。在`PickupSpawner`构造函数中，我们像往常一样初始化我们的组件。当游戏开始时，生成器运行其`SpawnPickup`函数。此函数生成我们的`Pickup`实例，然后将`APickupSpawner::PickupCollected`绑定到新实例上的`OnPickedUp`函数。它还存储对该当前实例的引用。

    当玩家与`Pickup`重叠后`PickupCollected`运行时，会创建一个计时器，在 10 秒后重新生成拾取物。现有的绑定到已收集拾取物的委托被移除，然后销毁拾取物。10 秒后，计时器触发，再次运行`SpawnActor`，创建一个新的`Pickup`。
