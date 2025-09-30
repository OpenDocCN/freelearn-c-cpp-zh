# 第三章. 探索与战斗

我们已经为我们的游戏设计了游戏界面，并设置了用于游戏的 Unreal 项目。现在是时候深入实际的游戏代码了。

在这一章中，我们将制作一个在世界上移动的游戏角色，定义我们的游戏数据，并为游戏原型设计一个基本的战斗系统。本章将涵盖以下主题：

+   创建玩家角色

+   定义角色、类和敌人

+   跟踪活跃的队伍成员

+   创建一个基本的回合制战斗引擎

+   触发游戏结束屏幕

这一部分是本书中最注重 C++的部分，并为本书的其余部分提供了一个基本框架。由于这一章节提供了我们游戏的后端大部分内容，因此在这一章节中的代码必须完整无误地工作，才能继续阅读本书的其余内容。如果你购买这本书是因为你是一名程序员，正在寻找更多关于创建 RPG 框架的背景知识，那么这一章节就是为你准备的！如果你购买这本书是因为你是一名设计师，更关心在框架上构建而不是从头开始编程，你可能对即将到来的章节更感兴趣，因为那些章节包含更少的 C++和更多的 UMG 和蓝图。无论你是谁，下载前言中提供的源代码都是一个好主意，以防你遇到困难或想根据你的兴趣跳过某些章节。

# 创建玩家角色

我们将要做的第一件事是创建一个新的 Pawn 类。在 Unreal 中，*Pawn*是角色的表示。它处理角色的移动、物理和渲染。

这就是我们的角色 Pawn 将要如何工作。玩家分为两部分：有一个 Pawn，如前所述，负责处理移动、物理和渲染。然后是 Player Controller，负责将玩家的输入转换为让 Pawn 执行玩家想要的动作。

## Pawn

现在，让我们创建实际的 Pawn。

创建一个新的 C++类，并将其父类设置为`Character`。我们将从这个`Character`类中派生这个类，因为`Character`类有很多内置的移动函数，我们可以为我们的场地上玩家使用。将类命名为`RPGCharacter`。打开`RPGCharacter.h`，并使用以下代码更改类定义：

```cpp
UCLASS(config = Game)
class ARPGCharacter : public ACharacter
{
  GENERATED_BODY()

  /** Camera boom positioning the camera behind the character */
UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = Camera, meta = (AllowPrivateAccess = "true")) class USpringArmComponent* CameraBoom;

  /** Follow camera */
UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = Camera, meta = (AllowPrivateAccess = "true")) class UCameraComponent* FollowCamera;
public:
  ARPGCharacter();

  /**Base turn rate, in deg/sec. Other scaling may affect final turn rate.*/
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = Camera)
    float BaseTurnRate;

protected:

  /** Called for forwards/backward input */
  void MoveForward(float Value);

  /** Called for side to side input */
  void MoveRight(float Value);

  /**
  * Called via input to turn at a given rate.
  * @param Rate  This is a normalized rate, i.e. 1.0 means 100% of desired turn rate
  */
  void TurnAtRate(float Rate);

protected:
  // APawn interface
virtual void SetupPlayerInputComponent(class UInputComponent* InputComponent) override;
  // End of APawn interface

public:
  /** Returns CameraBoom subobject **/
FORCEINLINE class USpringArmComponent* GetCameraBoom() const { return CameraBoom; }
  /** Returns FollowCamera subobject **/
FORCEINLINE class UCameraComponent* GetFollowCamera() const { return FollowCamera; }
};
```

接下来，打开`RPGCharacter.cpp`，并用以下代码替换它：

```cpp
#include "RPG.h"
#include "RPGCharacter.h"

ARPGCharacter::ARPGCharacter()
{
  // Set size for collision capsule
  GetCapsuleComponent()->InitCapsuleSize(42.f, 96.0f);

  // set our turn rates for input
  BaseTurnRate = 45.f;

// Don't rotate when the controller rotates. //Let that just affect the camera.
  bUseControllerRotationPitch = false;
  bUseControllerRotationYaw = false;
  bUseControllerRotationRoll = false;

  // Configure character movement
// Character moves in the direction of input...
GetCharacterMovement()->bOrientRotationToMovement = true; // ...at this rotation rate  
GetCharacterMovement()->RotationRate = FRotator(0.0f, 540.0f, 0.0f); 

  // Create a camera boom
CameraBoom = CreateDefaultSubobject<USpringArmComponent>(TEXT("CameraBoom"));
  CameraBoom->SetupAttachment(RootComponent);
// The camera follows at this distance behind the character  CameraBoom->TargetArmLength = 300.0f; 
CameraBoom->RelativeLocation = FVector(0.f, 0.f, 500.f);// Rotate the arm based on the controller
CameraBoom->bUsePawnControlRotation = true; 

// Create a follow camera
FollowCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("FollowCamera"));
FollowCamera->SetupAttachment(CameraBoom, USpringArmComponent::SocketName); // Camera does not rotate relative to arm
FollowCamera->bUsePawnControlRotation = false;  FollowCamera->RelativeRotation = FRotator(-45.f, 0.f, 0.f);

/* Note: The skeletal mesh and anim blueprint references on the Mesh component (inherited from Character) are set in the derived blueprint asset named MyCharacter (to avoid direct content references in C++)*/
}

//////////////////////////////////////////////////////////////////
// Input

void ARPGCharacter::SetupPlayerInputComponent(class UInputComponent* InputComponent)
{
  // Set up gameplay key bindings
  check(InputComponent);

InputComponent->BindAxis("MoveForward", this, &ARPGCharacter::MoveForward);
InputComponent->BindAxis("MoveRight", this, &ARPGCharacter::MoveRight);

/* We have 2 versions of the rotation bindings to handle different kinds of devices differently "turn" handles devices that provide an absolute delta, such as a mouse. "turnrate" is for devices that we choose to treat as a rate of change, such as an analog joystick*/
InputComponent->BindAxis("Turn", this, &APawn::AddControllerYawInput);
InputComponent->BindAxis("TurnRate", this, &ARPGCharacter::TurnAtRate);
}

void ARPGCharacter::TurnAtRate(float Rate)
{
  // calculate delta for this frame from the rate information
AddControllerYawInput(Rate * BaseTurnRate * GetWorld()->GetDeltaSeconds());
}

void ARPGCharacter::MoveForward(float Value)
{
  if ((Controller != NULL) && (Value != 0.0f))
  {
    // find out which way is forward
    const FRotator Rotation = Controller->GetControlRotation();
    const FRotator YawRotation(0, Rotation.Yaw, 0);

    // get forward vector
    const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::X);
    AddMovementInput(Direction, Value);
  }
}

void ARPGCharacter::MoveRight(float Value)
{
  if ((Controller != NULL) && (Value != 0.0f))
  {
    // find out which way is right
    const FRotator Rotation = Controller->GetControlRotation();
    const FRotator YawRotation(0, Rotation.Yaw, 0);

    // get right vector 
    const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::Y);
    // add movement in that direction
    AddMovementInput(Direction, Value);
  }
}
```

如果你曾经创建并使用过 C++的**ThirdPerson**游戏模板，你会注意到我们在这里并没有重新发明轮子。`RPGCharacter`类应该看起来很熟悉，因为它是我们创建 C++ ThirdPerson 模板时提供的`Character`类代码的修改版本，由 Epic Games 提供给我们使用。

由于我们不是在制作快节奏的动作游戏，而是简单地将 Pawn 用作 RPG 角色在战场上移动，因此我们消除了与动作游戏经常相关的机制，例如跳跃。但我们保留了对我们重要的代码，这包括向前、向后、向左和向右移动的能力；Pawn 的旋转行为；一个以等距视角跟随角色的相机；一个用于角色能够与可碰撞对象碰撞的碰撞胶囊；角色移动的配置；以及一个相机吊杆，它允许相机在角色遇到墙壁或其他网格等碰撞时靠近角色，这对于不让玩家视线受阻非常重要。如果您想编辑角色机制，请随意通过遵循代码中的注释来更改某些特定机制的价值，例如将`TargetArmLength`的值更改以改变相机与玩家之间的距离，或者添加跳跃，这可以在与引擎一起提供的 ThirdPerson 角色模板中看到。

由于我们是从`Character`类派生出了`RPGCharacter`类，其默认相机在等距视角下没有旋转；相反，相机的旋转和位置默认为零，设置为 Pawn 的位置。所以我们做的是在`RPGCharacter.cpp`中添加了一个相对位置`CameraBoom`（`CameraBoom->RelativeLocation = FVector(0.f, 0.f, 500.f);`）；这使相机在*Z*轴上向上偏移了 500 个单位。与旋转跟随玩家的相机-45 单位在俯仰（`FollowCamera->RelativeRotation = FRotator(-45.f, 0.f, 0.f);`）一起，我们得到了传统的等距视角。如果您想编辑这些值以进一步自定义相机，建议这样做；例如，如果您仍然认为您的相机离玩家太近，您只需将`CameraBoom`在*Z*轴上的相对位置更改为大于 500 个单位的值，或者调整`TargetArmLength`到一个大于 300 的值。

最后，如果你查看 `MoveForward` 和 `MoveRight` 移动函数，你会注意到，除非传递给 `MoveForward` 或 `MoveRight` 的值不等于 0，否则不会向 pawn 添加任何移动。在本章的后面部分，我们将把键 *W*、*A*、*S* 和 *D* 绑定到这些函数上，并将每个输入设置为传递 1 或 -1 的标量值给相应的移动函数。这个 1 或 -1 的值随后用作 pawn 方向的乘数，这将允许玩家根据其行走速度向特定方向移动。例如，如果我们把 *W* 作为 `MoveForward` 的键绑定并使用标量 1，把 *S* 作为 `MoveFoward` 的键绑定并使用标量 -1，当玩家按下 *W* 时，`MoveFoward` 函数中的值将等于 1，从而使 pawn 向正前方移动。相反，如果玩家按下 *S* 键，-1 就会被传递到 `MoveForward` 函数使用的值中，这将使 pawn 向负前方移动（换句话说，向后）。关于 `MoveRight` 函数的类似逻辑也可以这么说，这就是为什么我们没有 `MoveLeft` 函数——仅仅是因为按下 *A* 键会导致玩家向负右方向移动，这实际上是向左。

## 游戏模式类

现在，为了使用这个 pawn 作为玩家角色，我们需要设置一个新的游戏模式类。这个游戏模式将指定默认的 pawn 和玩家控制器类。我们还将能够创建游戏模式的蓝图并覆盖这些默认设置。

创建一个新的类，并将 `GameMode` 作为父类。将这个新类命名为 `RPGGameMode`（如果 `RPGGameMode` 在你的项目中已经存在，只需导航到你的 C++ 源代码目录，然后继续打开 `RPGGameMode.h`，如下一步所述）。

打开 `RPGGameMode.h` 并使用以下代码更改类定义：

```cpp
UCLASS()
class RPG_API ARPGGameMode : public AGameMode
{
  GENERATED_BODY()

  ARPGGameMode( const class FObjectInitializer& ObjectInitializer );
};
```

就像我们之前做的那样，我们只是在定义一个用于实现 CPP 文件的构造函数。

现在，我们将在这个 `RPGGameMode.cpp` 中实现那个构造函数：

```cpp
#include "RPGCharacter.h"

ARPGGameMode::ARPGGameMode( const class FObjectInitializer& ObjectInitializer )
  : Super( ObjectInitializer )
{

  DefaultPawnClass = ARPGCharacter::StaticClass();
}
```

在这里，我们包含 `RPGCharacter.h` 文件，以便我们可以引用这些类。然后，在构造函数中，我们将该类设置为 Pawn 的默认类。

现在，如果你编译这段代码，你应该能够将你的新游戏模式类作为默认游戏模式。为此，转到 **编辑** | **项目设置**，找到 **默认模式** 框，展开 **默认游戏模式** 下拉菜单，并选择 **RPGGameMode**。

然而，我们并不一定想直接使用这个类。相反，如果我们创建一个蓝图，我们可以暴露游戏模式中可以修改的属性。

因此，让我们在 **内容** | **蓝图** 中创建一个新的蓝图类，将其父类选择为 `RPGGameMode`，并将其命名为 `DefaultRPGGameMode`：

![游戏模式类](img/B04548_03_01.jpg)

如果你打开蓝图并导航到**默认**选项卡，你可以修改游戏模式设置，包括**默认兵类**、**HUD 类**、**玩家控制器类**等更多设置：

![游戏模式类](img/B04548_03_02.jpg)

然而，在我们能够测试我们新的兵之前，我们还需要额外的一步。如果你运行游戏，你将完全看不到兵。实际上，它看起来就像什么都没发生一样。我们需要给我们的兵一个带皮肤的网格，并且让摄像机跟随兵。

## 添加带皮肤的网格

现在，我们只是将要导入与 ThirdPerson 示例一起提供的原型角色。为此，基于 ThirdPerson 示例创建一个新的项目。在**内容** | **ThirdPersonCPP** | **蓝图**中找到 `ThirdPersonCharacter` 蓝图类，通过右键单击 `ThirdPersonCharacter` 蓝图类并导航到**资产操作** | **迁移…**将其迁移到 RPG 项目的**内容**文件夹。此操作应将 `ThirdPersonCharacter` 及其所有资产复制到你的 RPG 项目中：

![添加带皮肤的网格](img/B04548_03_03.jpg)

现在，让我们为我们的兵创建一个新的蓝图。创建一个新的蓝图类，并将**RPGCharacter**作为父类。将其命名为**FieldPlayer**。

当从**组件**选项卡中选择**网格**组件时，在**详细信息**选项卡中展开**网格**，并将**SK_Mannequin**作为兵的骨骼网格选择。接下来，展开**动画**并选择要使用的**ThirdPerson_AnimBP**动画蓝图。你很可能会需要将角色的网格沿 *z* 轴向下移动，以便角色的脚底与碰撞胶囊的底部对齐。同时确保角色网格面向与组件中蓝色箭头相同的方向。你可能还需要在 *z* 轴上旋转角色，以确保角色面向正确的方向：

![添加带皮肤的网格](img/B04548_03_04.jpg)

最后，打开你的游戏模式蓝图，并将兵更改为你的新**FieldPlayer**蓝图。

现在，我们的角色将变得可见，但我们可能还不能移动它，因为我们还没有将任何按键绑定到我们的移动变量上。要做到这一点，请进入**项目设置**并找到**输入**。展开**绑定**然后展开**轴映射**。通过按**+**按钮添加一个轴映射。将第一个轴映射命名为**MoveRight**，它应该与您在本章 earlier 创建的 `MoveRight` 移动变量相匹配。通过按**+**按钮添加两个**MoveRight**的键绑定。让其中一个键是 *A*，缩放为 -1，另一个是 *D*，缩放为 1。为**MoveForward**添加另一个轴映射；这次，有一个 *W* 键绑定，缩放为 1，一个 *S* 键绑定，缩放为 -1：

![添加带皮肤的网格](img/B04548_03_05.jpg)

一旦进行游戏测试，你应该会看到你的角色使用你绑定的 *W*、*A*、*S* 和 *D* 键移动和动画。

当你运行游戏时，摄像机应该以俯视角度跟踪玩家。现在我们有一个可以探索游戏世界的角色，让我们来看看如何定义角色和队伍成员。

# 定义角色和敌人

在上一章中，我们介绍了如何使用数据表导入自定义数据。在此之前，我们决定哪些属性会影响战斗以及如何影响。现在，我们将结合这些属性来定义我们的游戏角色、类别和敌人遭遇。

## 类别

记住，在第一章中，我们确立了我们的角色具有以下属性：

+   生命值

+   最大生命值

+   魔法

+   最大魔法值

+   攻击力

+   防御力

+   幸运值

在这些属性中，我们可以丢弃生命值和魔法值，因为它们在游戏过程中会变化，而其他值是基于角色类别预先定义的。剩余的属性是我们将在数据表中定义的。如第一章中所述，我们还需要存储在 50 级（最大等级）时的值。角色还将有一些初始能力，以及随着等级提升而学习的能力。

我们将在角色类别电子表格中定义这些属性，包括类名。因此，我们的角色类别架构将类似于以下内容：

+   类名（字符串）

+   初始最大生命值（整数）

+   50 级最大生命值（整数）

+   初始最大魔法值（整数）

+   50 级最大魔法值（整数）

+   初始攻击力（整数）

+   50 级攻击力（整数）

+   初始防御力（整数）

+   50 级防御力（整数）

+   初始幸运值（整数）

+   50 级幸运值（整数）

+   初始能力（字符串数组）

+   学到的能力（字符串数组）

+   学到的能力等级（整数数组）

能力字符串数组将包含能力的 ID（UE4 中保留的`name`字段的值）。此外，还有两个单独的单元格用于学习到的能力——一个包含能力 ID，另一个包含学习这些能力时的等级。

在一个生产游戏中，你可能考虑写一个自定义工具来帮助管理这些数据并减少人为错误。然而，编写这样的工具超出了本书的范围。

现在，我们不是为这个创建电子表格，而是首先在 Unreal 中创建类，然后创建数据表。这样做的原因是，在撰写本文时，指定数据表单元格中数组的正确语法并未得到很好的记录。然而，数组仍然可以从 Unreal 编辑器内部进行编辑，因此我们只需在那里创建表格并使用 Unreal 的数组编辑器。

首先，像往常一样，创建一个新的类。这个类将用作你可以从中调用的对象，因此选择`Object`作为父类。将此类命名为`FCharacterClassInfo`，为了组织目的，将新类路径到你的`Source/RPG/Data`文件夹。

打开`FCharacterClassInfo.h`并将类定义替换为以下代码：

```cpp
USTRUCT( BlueprintType )
struct FCharacterClassInfo : public FTableRowBase
{
  GENERATED_USTRUCT_BODY()

  UPROPERTY( BlueprintReadWrite, EditAnywhere, Category = "ClassInfo" )
    FString Class_Name;

  UPROPERTY( BlueprintReadWrite, EditAnywhere, Category = "ClassInfo" )
    int32 StartMHP;

  UPROPERTY( BlueprintReadWrite, EditAnywhere, Category = "ClassInfo" )
    int32 StartMMP;

  UPROPERTY( BlueprintReadWrite, EditAnywhere, Category = "ClassInfo" )
    int32 StartATK;

  UPROPERTY( BlueprintReadWrite, EditAnywhere, Category = "ClassInfo" )
    int32 StartDEF;

  UPROPERTY( BlueprintReadWrite, EditAnywhere, Category = "ClassInfo" )
    int32 StartLuck;

  UPROPERTY( BlueprintReadWrite, EditAnywhere, Category = "ClassInfo" )
    int32 EndMHP;

  UPROPERTY( BlueprintReadWrite, EditAnywhere, Category = "ClassInfo" )
    int32 EndMMP;

  UPROPERTY( BlueprintReadWrite, EditAnywhere, Category = "ClassInfo" )
    int32 EndATK;

  UPROPERTY( BlueprintReadWrite, EditAnywhere, Category = "ClassInfo" )
    int32 EndDEF;

  UPROPERTY( BlueprintReadWrite, EditAnywhere, Category = "ClassInfo" )
    int32 EndLuck;

  UPROPERTY( BlueprintReadWrite, EditAnywhere, Category = "ClassInfo" )
    TArray<FString> StartingAbilities;

  UPROPERTY( BlueprintReadWrite, EditAnywhere, Category = "ClassInfo" )
    TArray<FString> LearnedAbilities;

  UPROPERTY( BlueprintReadWrite, EditAnywhere, Category = "ClassInfo" )
    TArray<int32> LearnedAbilityLevels;
};
```

这段代码中的大部分您可能已经熟悉；然而，您可能不认识最后三个字段。这些都是`TArray`类型，这是 Unreal 提供的一种动态数组类型。本质上，`TArray`可以动态地向其中添加元素并从中移除元素，这与标准 C++数组不同。

编译此代码后，在您的“内容”文件夹内创建一个名为`Data`的新文件夹，以便通过将您创建的数据表保存在`Data`文件夹中保持组织有序。在内容浏览器中导航到**内容** | **数据**，通过右键单击**内容浏览器**并选择**杂项** | **数据表**来创建一个新的数据表。然后，从下拉列表中选择**角色类信息**。将您的数据表命名为**CharacterClasses**，然后双击打开它。

要添加新条目，请点击**+**按钮。然后，在**行名称**字段中输入新条目的名称并按*Enter*键。

添加条目后，您可以在**数据表**面板中选择条目，并在**行编辑器**面板中编辑其属性。

让我们在列表中添加一个士兵类。我们将给它命名为`S1`（我们将用它来引用其他数据表中的角色类）并且它将具有以下属性：

+   **类名**：士兵

+   **开始 MHP**：100

+   **开始 MMP**：100

+   **开始 ATK**：5

+   **开始 DEF**：0

+   **开始幸运**：0

+   **结束 MHP**：800

+   **结束 MMP**：500

+   **结束 ATK**：20

+   **结束 DEF**：20

+   **结束幸运**：10

+   **起始能力**：（目前留空）

+   **学习能力**：（目前留空）

+   **学习能力等级**：（目前留空）

当你完成时，你的数据表应该看起来像这样：

![类](img/B04548_03_06.jpg)

如果您想定义更多的角色类，请继续将它们添加到您的数据表中。

## 角色

在定义了类之后，让我们来看看角色。由于大多数重要的战斗相关数据已经作为角色类的一部分定义，因此角色本身将会相当简单。实际上，目前我们的角色将由以下两点定义：角色的名称和角色的类。

首先，创建一个名为`FCharacterInfo`的新 C++类，其父类为`Object`，并将其路径设置为`Source/RPG/Data`文件夹。现在，将`FCharacterInfo.h`中的类定义替换为以下内容：

```cpp
USTRUCT(BlueprintType)
struct FCharacterInfo : public FTableRowBase
{
  GENERATED_USTRUCT_BODY()

  UPROPERTY( BlueprintReadWrite, EditAnywhere, Category = "CharacterInfo" )
  FString Character_Name;

  UPROPERTY( BlueprintReadOnly, EditAnywhere, Category = "CharacterInfo" )
  FString Class_ID;
};
```

就像我们之前做的那样，我们只是定义了角色的两个字段（角色名称和类 ID）。

编译后，在您之前创建的内容浏览器中的`Data`文件夹内创建一个新的数据表，并选择**CharacterInfo**作为类；命名为`Characters`。添加一个名为`S1`的新条目。您可以给这个角色起任何名字（我们给我们的角色命名为士兵**Kumo**），但请在类 ID 中输入`S1`（因为这是我们之前定义的士兵类的名称）。

## 敌人

至于敌人，我们不会为单独的角色和职业信息定义一个类，而是为这两部分信息创建一个简化的组合表。敌人通常不需要处理经验和升级，因此我们可以省略与此相关的任何数据。此外，敌人不会像玩家那样消耗 MP，因此我们也可以省略这部分数据。

因此，我们的敌人数据将具有以下属性：

+   敌人名称（字符串数组）

+   最大生命值（整数）

+   攻击力（整数）

+   防御力（整数）

+   幸运值（整数）

+   能力（字符串数组）

与之前的数据类创建类似，我们创建一个新的从 `Object` 派生的 C++ 类，但这次我们将它命名为 `FEnemyInfo` 并将其放置在 `Source/RPG/Data` 目录中的其他数据旁边。

在这个阶段，你应该已经了解了如何为这些数据构建类，但无论如何，让我们看一下结构头文件。在 `FEnemyInfo.h` 中，将你的类定义替换为以下内容：

```cpp
USTRUCT( BlueprintType )
struct FEnemyInfo : public FTableRowBase
{
  GENERATED_USTRUCT_BODY()

  UPROPERTY( BlueprintReadWrite, EditAnywhere, Category = "EnemyInfo" )
    FString EnemyName;

  UPROPERTY( BlueprintReadOnly, EditAnywhere, Category = "EnemyInfo" )
    int32 MHP;

  UPROPERTY( BlueprintReadOnly, EditAnywhere, Category = "EnemyInfo" )
    int32 ATK;

  UPROPERTY( BlueprintReadOnly, EditAnywhere, Category = "EnemyInfo" )
    int32 DEF;

  UPROPERTY( BlueprintReadOnly, EditAnywhere, Category = "EnemyInfo" )
    int32 Luck;

  UPROPERTY( BlueprintReadOnly, EditAnywhere, Category = "EnemyInfo" )
    TArray<FString> Abilities;
};
```

编译完成后，创建一个新的数据表，选择 `EnemyInfo` 作为类，并将数据表命名为 `Enemies`。添加一个名为 `S1` 的新条目，并具有以下属性：

+   **敌人名称**: 哥布林

+   **最大生命值**: 20

+   **攻击力**: 5

+   **DEF**: 0

+   **幸运值**: 0

+   **能力**:（目前留空）

到目前为止，我们已经有了角色的数据、角色的职业以及角色要与之战斗的单一敌人。接下来，让我们开始跟踪哪些角色在活动队伍中，以及他们的当前状态。

# 队伍成员

在我们能够跟踪队伍成员之前，我们需要一种方法来跟踪角色的当前状态，比如角色有多少 HP 或者装备了什么。

为了做到这一点，我们将创建一个新的类名为 `GameCharacter`。像往常一样，创建一个新的类并将 `Object` 作为父类。

这个类的头文件看起来像以下代码片段：

```cpp
#pragma once

#include "Data/FCharacterInfo.h"
#include "Data/FCharacterClassInfo.h"

#include "GameCharacter.generated.h"

UCLASS( BlueprintType )
class RPG_API UGameCharacter : public UObject
{
  GENERATED_BODY()

public:
  FCharacterClassInfo* ClassInfo;

  UPROPERTY( EditAnywhere, BlueprintReadWrite, Category = CharacterInfo )
  FString CharacterName;

  UPROPERTY( EditAnywhere, BlueprintReadWrite, Category = CharacterInfo )
  int32 MHP;

  UPROPERTY( EditAnywhere, BlueprintReadWrite, Category = CharacterInfo )
  int32 MMP;

  UPROPERTY( EditAnywhere, BlueprintReadWrite, Category = CharacterInfo )
  int32 HP;

  UPROPERTY( EditAnywhere, BlueprintReadWrite, Category = CharacterInfo )
  int32 MP;

  UPROPERTY( EditAnywhere, BlueprintReadWrite, Category = CharacterInfo )
  int32 ATK;

  UPROPERTY( EditAnywhere, BlueprintReadWrite, Category = CharacterInfo )
  int32 DEF;

  UPROPERTY( EditAnywhere, BlueprintReadWrite, Category = CharacterInfo )
  int32 LUCK;

public:
  static UGameCharacter* CreateGameCharacter( FCharacterInfo* characterInfo, UObject* outer );

public:
  void BeginDestroy() override;
};
```

目前，我们正在跟踪角色的名字、角色的来源职业信息以及角色的当前状态。稍后，我们将使用 `UCLASS` 和 `UPROPERTY` 宏将信息暴露给蓝图。我们将在开发战斗系统时添加其他信息。

对于 `.cpp` 文件，它看起来像这样：

```cpp
#include "RPG.h"
#include "GameCharacter.h"

UGameCharacter* UGameCharacter::CreateGameCharacter( FCharacterInfo* characterInfo, UObject* outer )
{
  UGameCharacter* character = NewObject<UGameCharacter>( outer );

  // locate character classes asset
  UDataTable* characterClasses = Cast<UDataTable>( StaticLoadObject( UDataTable::StaticClass(), NULL, TEXT( "DataTable'/Game/Data/CharacterClasses.CharacterClasses'" ) ) );

  if( characterClasses == NULL )
  {
    UE_LOG( LogTemp, Error, TEXT( "Character classes datatable not found!" ) );
  }
  else
  {
    character->CharacterName = characterInfo->Character_Name;
    FCharacterClassInfo* row = characterClasses->FindRow<FCharacterClassInfo>( *( characterInfo->Class_ID ), TEXT( "LookupCharacterClass" ) );
    character->ClassInfo = row;

    character->MHP = character->ClassInfo->StartMHP;
    character->MMP = character->ClassInfo->StartMMP;
    character->HP = character->MHP;
    character->MP = character->MMP;

    character->ATK = character->ClassInfo->StartATK;
    character->DEF = character->ClassInfo->StartDEF;
    character->LUCK = character->ClassInfo->StartLuck;
  }

  return character;
}

void UGameCharacter::BeginDestroy()
{
  Super::BeginDestroy();
}
```

我们 `UGameCharacter` 类的 `CreateGameCharacter` 工厂函数接受一个指向 `FCharacterInfo` 结构的指针，该结构由数据表返回，并且还接受一个 `Outer` 对象，该对象传递给 `NewObject` 函数。然后它尝试从一个路径中找到角色类数据表，如果结果不为空，它将定位到数据表中的正确行，存储结果，并初始化状态和 `CharacterName` 字段。在前面的代码中，你可以看到角色类数据表所在的路径。你可以通过在内容浏览器中右键单击你的数据表，选择 **复制引用**，然后将结果粘贴到你的代码中。

虽然这目前是一个非常基本的骨架式的人物表示，但暂时可以工作。接下来，我们将存储这些人物的列表作为当前党派。

## 游戏实例类

我们已经创建了一个`GameMode`类，这似乎是跟踪党派成员和库存等信息的一个完美的位置，对吧？

然而，`GameMode`在关卡加载之间不会持久化！这意味着除非你将一些信息保存到磁盘上，否则每次加载新区域时都会丢失所有这些数据。

`GameInstance`类是为了解决这类问题而引入的。`GameInstance`类在整个游戏过程中都保持持久，与`GameMode`不同。我们将创建一个新的`GameInstance`类来跟踪我们的持久数据，例如党派成员和库存。

创建一个新的类，这次选择`GameInstance`作为父类（你可能需要搜索它）。将其命名为`RPGGameInstance`。

在头文件中，我们将添加一个`UGameCharacter`指针的`TArray`，一个表示游戏是否已初始化的标志，以及一个`Init`函数。你的`RPGGameInstance.h`文件应该看起来像这样：

```cpp
#pragma once

#include "Engine/GameInstance.h"
#include "GameCharacter.h"
#include "RPGGameInstance.generated.h"
UCLASS()
class RPG_API URPGGameInstance : public UGameInstance
{
  GENERATED_BODY()

  URPGGameInstance( const class FObjectInitializer& ObjectInitializer );

public:
  TArray<UGameCharacter*> PartyMembers;

protected:
  bool isInitialized;

public:
  void Init();
};
```

在游戏实例的`Init`函数中，我们将添加一个默认的党派成员，并将`isInitialized`标志设置为`true`。你的`RPGGameInstance.cpp`应该看起来像这样：

```cpp
#include "RPG.h"
#include "RPGGameInstance.h"
URPGGameInstance::URPGGameInstance(const class FObjectInitializer& 
ObjectInitializer)
: Super(ObjectInitializer)
{
  isInitialized = false;
}

void URPGGameInstance::Init()
{
  if( this->isInitialized ) return;

  this->isInitialized = true;

  // locate characters asset
  UDataTable* characters = Cast<UDataTable>( StaticLoadObject( UDataTable::StaticClass(), NULL, 
TEXT( "DataTable'/Game/Data/Characters.Characters'" ) ) );
        if( characters == NULL )
  {
    UE_LOG( LogTemp, Error, TEXT( "Characters data table not found!" ) );

    return;
  }

  // locate character
  FCharacterInfo* row = characters->FindRow<FCharacterInfo>( TEXT( "S1" ), TEXT( "LookupCharacterClass" ) );

  if( row == NULL )
  {
    UE_LOG( LogTemp, Error, TEXT( "Character ID 'S1' not found!" ) );
    return;
  }

  // add character to party
  this->PartyMembers.Add( UGameCharacter::CreateGameCharacter( row, this ) );
}
```

如果你尝试编译，可能会遇到链接错误。建议在继续之前保存并关闭所有内容。然后重新启动你的项目。之后，编译项目。

要将此类设置为你的`GameInstance`类，在 Unreal 中，打开**编辑** | **项目设置**，转到**地图与模式**，滚动到**游戏实例**框，并从下拉列表中选择**RPGGameInstance**。最后，从游戏模式中重写`BeginPlay`以调用此`Init`函数。

打开`RPGGameMode.h`，并在你的类末尾添加`virtual void BeginPlay() override;`，这样你的头文件现在看起来像这样：

```cpp
#pragma once

#include "GameFramework/GameMode.h"
#include "RPGGameMode.generated.h"

UCLASS()
class RPG_API ARPGGameMode : public AGameMode
{
  GENERATED_BODY()

  ARPGGameMode(const class FObjectInitializer& ObjectInitializer);
  virtual void BeginPlay() override;
};
```

并且在`RPGGameMode.cpp`中，在`BeginPlay`时将`RPGGameInstance`进行转换，这样`RPGGameMode.cpp`现在看起来像这样：

```cpp
#include "RPG.h"
#include "RPGGameMode.h"
#include "RPGCharacter.h"
#include "RPGGameInstance.h"

ARPGGameMode::ARPGGameMode(const class FObjectInitializer& 
ObjectInitializer)
: Super(ObjectInitializer)
{
  DefaultPawnClass = ARPGCharacter::StaticClass();
   }

void ARPGGameMode::BeginPlay()
{
  Cast<URPGGameInstance>(GetGameInstance())->Init();
}
```

一旦编译代码，你现在就有一个活跃的党派成员列表。现在是时候开始原型设计战斗引擎了。

# 回合制战斗

所以，如第一章中提到的，*在 Unreal 中开始 RPG 设计*，战斗是回合制的。所有角色首先选择要执行的动作；然后，按照顺序执行这些动作。

战斗将分为两个主要阶段：**决策**，在这个阶段，所有角色决定他们的行动方案；和**行动**，在这个阶段，所有角色执行他们选择的行动方案。

让我们创建一个没有父类的类来为我们处理战斗，我们可以将其称为`CombatEngine`，并将其放置在`Source/RPG/Combat`的新目录中，我们可以在这里组织所有与战斗相关的类。将头文件制定如下：

```cpp
#pragma once
#include "RPG.h"
#include "GameCharacter.h"

enum class CombatPhase : uint8
{
  CPHASE_Decision,
  CPHASE_Action,
  CPHASE_Victory,
  CPHASE_GameOver,
};

class RPG_API CombatEngine
{
public:
  TArray<UGameCharacter*> combatantOrder;
  TArray<UGameCharacter*> playerParty;
  TArray<UGameCharacter*> enemyParty;

  CombatPhase phase;

protected:
  UGameCharacter* currentTickTarget;
  int tickTargetIndex;

public:
  CombatEngine( TArray<UGameCharacter*> playerParty, TArray<UGameCharacter*> enemyParty );
  ~CombatEngine();

  bool Tick( float DeltaSeconds );

protected:
  void SetPhase( CombatPhase phase );
  void SelectNextCharacter();
};
```

这里有很多事情在进行，所以我将进行解释。

首先，我们的战斗引擎设计为在遭遇开始时分配，在战斗结束时删除。

`CombatEngine` 实例维护了三个 `TArray`：一个用于战斗顺序（所有参与战斗的成员列表，按照他们轮流行动的顺序），另一个用于玩家列表，第三个用于敌人列表。它还跟踪 `CombatPhase`。战斗有两个主要阶段：`Decision` 和 `Action`。每一轮从 `Decision` 阶段开始；在这个阶段，所有角色都可以选择他们的行动方案。然后，战斗过渡到 `Action` 阶段；在这个阶段，所有角色将执行他们之前选择的行动方案。

当所有敌人死亡或所有玩家死亡时，将过渡到 `GameOver` 和 `Victory` 阶段（这就是为什么玩家和敌人列表被分开保留的原因）。

`CombatEngine` 类定义了一个 `Tick` 函数。这个函数将在战斗未结束的情况下，由每一帧的游戏模式调用，当战斗结束时返回 `true`（否则返回 `false`）。它接受上一帧的持续时间作为参数。

还有 `currentTickTarget` 和 `tickTargetIndex`。在 `Decision` 和 `Action` 阶段，我们将保持对单个角色的指针。例如，在 `Decision` 阶段，这个指针从战斗顺序中的第一个角色开始。在每一帧，都会要求角色做出决策——这将是一个返回 `true` 如果角色已经做出决策，否则返回 `false` 的函数。如果函数返回 `true`，指针将移动到下一个角色，依此类推，直到所有角色都做出决策，此时战斗过渡到 `Action` 阶段。

这个文件的 CPP 代码相当大，所以让我们分块来看。首先，构造函数和析构函数如下：

```cpp
CombatEngine::CombatEngine( TArray<UGameCharacter*> playerParty, TArray<UGameCharacter*> enemyParty )
{
  this->playerParty = playerParty;
  this->enemyParty = enemyParty;

  // first add all players to combat order
  for( int i = 0; i < playerParty.Num(); i++ )
  {
    this->combatantOrder.Add( playerParty[i] );
  }

  // next add all enemies to combat order
  for( int i = 0; i < enemyParty.Num(); i++ )
  {
    this->combatantOrder.Add( enemyParty[i] );
  }

  this->tickTargetIndex = 0;
  this->SetPhase( CombatPhase::CPHASE_Decision );
}

CombatEngine::~CombatEngine()
{
}
```

构造函数首先分配玩家团体和敌人团体字段，然后添加所有玩家，接着添加所有敌人到战斗顺序列表中。最后，它将 `tick` 目标索引设置为 0（战斗顺序中的第一个角色）并将战斗阶段设置为 `Decision`。

接下来，`Tick` 函数如下：

```cpp
bool CombatEngine::Tick( float DeltaSeconds )
{
  switch( phase )
  {
    case CombatPhase::CPHASE_Decision:
      // todo: ask current character to make decision

      // todo: if decision made
      SelectNextCharacter();

      // no next character, switch to action phase
      if( this->tickTargetIndex == -1 )
      {
        this->SetPhase( CombatPhase::CPHASE_Action );
      }
      break;
    case CombatPhase::CPHASE_Action:
      // todo: ask current character to execute decision

      // todo: when action executed
      SelectNextCharacter();

      // no next character, loop back to decision phase
      if( this->tickTargetIndex == -1 )
      {
        this->SetPhase( CombatPhase::CPHASE_Decision );
      }
      break;
    // in case of victory or combat, return true (combat is finished)
    case CombatPhase::CPHASE_GameOver:
    case CombatPhase::CPHASE_Victory:
      return true;
      break;
  }

  // check for game over
  int deadCount = 0;
  for( int i = 0; i < this->playerParty.Num(); i++ )
  {
    if( this->playerParty[ i ]->HP <= 0 ) deadCount++;
  }

  // all players have died, switch to game over phase
  if( deadCount == this->playerParty.Num() )
  {
    this->SetPhase( CombatPhase::CPHASE_GameOver );
    return false;
  }

  // check for victory
  deadCount = 0;
  for( int i = 0; i < this->enemyParty.Num(); i++ )
  {
    if( this->enemyParty[ i ]->HP <= 0 ) deadCount++;
  }

  // all enemies have died, switch to victory phase
  if( deadCount == this->enemyParty.Num() )
  {
    this->SetPhase( CombatPhase::CPHASE_Victory );
    return false;
  }

  // if execution reaches here, combat has not finished - return false
  return false;
}
```

首先，我们切换到当前的战斗阶段。在 `Decision` 的情况下，它目前只是选择下一个角色，如果没有下一个角色，则切换到 `Action` 阶段。对于 `Action` 也是如此——除非没有下一个角色，它会循环回到 `Decision` 阶段。

之后，这将修改为调用角色的函数以做出和执行决策（此外，“选择下一个角色”的代码只有在角色完成决策或执行后才会被调用）。

在`GameOver`或`Victory`的情况下，`Tick`返回`true`表示战斗结束。否则，它首先检查是否所有玩家都已死亡（在这种情况下，游戏结束）或是否所有敌人都已死亡（在这种情况下，玩家赢得战斗）。在这两种情况下，函数将返回`true`，因为战斗已经结束。

函数的最后一部分返回`false`，这意味着战斗尚未结束。

接下来，我们有`SetPhase`函数：

```cpp
void CombatEngine::SetPhase( CombatPhase phase )
{
  this->phase = phase;

  switch( phase )
  {
    case CombatPhase::CPHASE_Action:
    case CombatPhase::CPHASE_Decision:
      // set the active target to the first character in the combat order
      this->tickTargetIndex = 0;
      this->SelectNextCharacter();
      break;
    case CombatPhase::CPHASE_Victory:
      // todo: handle victory
      break;
    case CombatPhase::CPHASE_GameOver:
      // todo: handle game over
      break;
  }
}
```

这个函数设置战斗阶段，在`Action`或`Decision`的情况下，将`tick`目标设置为战斗顺序中的第一个角色。`Victory`和`GameOver`都有处理相应状态的存根。

最后，我们有`SelectNextCharacter`函数：

```cpp
void CombatEngine::SelectNextCharacter()
{
  for( int i = this->tickTargetIndex; i < this->combatantOrder.Num(); i++ )
  {
    GameCharacter* character = this->combatantOrder[ i ];

    if( character->HP > 0 )
    {
      this->tickTargetIndex = i + 1;
      this->currentTickTarget = character;
      return;
    }
  }

  this->tickTargetIndex = -1;
  this->currentTickTarget = nullptr;
}
```

这个函数从当前的`tickTargetIndex`开始，并从那里找到战斗顺序中的第一个非死亡角色。如果找到了一个，它将`tick`目标索引设置为下一个索引，并将`tick`目标设置为找到的角色。否则，它将`tick`目标索引设置为-1，并将`tick`目标设置为空指针（这被解释为战斗顺序中没有剩余的角色）。

在这一点上，缺少了一个非常重要的事情：角色还不能做出或执行决策。

让我们将其添加到`GameCharacter`类中。目前，它们只是存根。

首先，我们将向`GameCharacter.h`添加`testDelayTimer`字段。这只是为了测试目的：

```cpp
protected:
  float testDelayTimer;
```

接下来，我们向类中添加几个公共函数：

```cpp
public:
  void BeginMakeDecision();
  bool MakeDecision( float DeltaSeconds );

  void BeginExecuteAction();
  bool ExecuteAction( float DeltaSeconds );
```

我们将`Decision`和`Action`分成两个函数每个——第一个函数告诉角色开始做出决策或执行动作，第二个函数本质上会查询角色直到决策做出或动作完成。

目前，这两个函数在`GameCharacter.cpp`中的实现只是记录一条消息和 1 秒的延迟：

```cpp
void UGameCharacter::BeginMakeDecision()
{
  UE_LOG( LogTemp, Log, TEXT( "Character %s making decision" ), *this->CharacterName );
  this->testDelayTimer = 1;
}

bool UGameCharacter::MakeDecision( float DeltaSeconds )
{
  this->testDelayTimer -= DeltaSeconds;
  return this->testDelayTimer <= 0;
}

void UGameCharacter::BeginExecuteAction()
{
  UE_LOG( LogTemp, Log, TEXT( "Character %s executing action" ), *this->CharacterName );
  this->testDelayTimer = 1;
}

bool UGameCharacter::ExecuteAction( float DeltaSeconds )
{
  this->testDelayTimer -= DeltaSeconds;
  return this->testDelayTimer <= 0;
}
```

我们还将添加一个指向战斗实例的指针。由于战斗引擎引用角色，而角色引用战斗引擎会产生循环依赖。为了解决这个问题，我们将在`GameCharacter.h`的顶部直接在我们的包含之后添加一个前向声明：

```cpp
class CombatEngine;
```

然后，用于战斗引擎的`include`语句实际上会被放置在`GameCharacter.cpp`文件中，而不是头文件中：

```cpp
#include "Combat/CombatEngine.h"
```

接下来，我们将使战斗引擎调用`Decision`和`Action`函数。首先，我们在`CombatEngine.h`中添加一个受保护的变量：

```cpp
bool waitingForCharacter;
```

这将用于在例如`BeginMakeDecision`和`MakeDecision`之间切换。

接下来，我们将修改`Tick`函数中的`Decision`和`Action`阶段。首先，我们将修改`Decision`的 switch case：

```cpp
case CombatPhase::CPHASE_Decision:
{
  if( !this->waitingForCharacter )
  {
    this->currentTickTarget->BeginMakeDecision();
    this->waitingForCharacter = true;
  }

  bool decisionMade = this->currentTickTarget->MakeDecision( DeltaSeconds );

  if( decisionMade )
  {
    SelectNextCharacter();

    // no next character, switch to action phase
    if( this->tickTargetIndex == -1 )
    {
      this->SetPhase( CombatPhase::CPHASE_Action );
    }
  }
}
break;
```

如果`waitingForCharacter`为`false`，它将调用`BeginMakeDecision`并将`waitingForCharacter`设置为`true`。

记住整个情况语句括号内的内容——如果你不添加这些括号，你将得到关于`decisionMade`初始化被情况语句跳过的编译错误。

接下来，它调用 `MakeDecision` 并传递帧时间。如果此函数返回 `true`，则选择下一个角色，或者如果没有成功，则切换到 `Action` 阶段。

`Action` 阶段看起来与以下内容相同：

```cpp
case CombatPhase::CPHASE_Action:
{
  if( !this->waitingForCharacter )
  {
    this->currentTickTarget->BeginExecuteAction();
    this->waitingForCharacter = true;
  }

  bool actionFinished = this->currentTickTarget->ExecuteAction( DeltaSeconds );

  if( actionFinished )
  {
    SelectNextCharacter();

    // no next character, switch to action phase
    if( this->tickTargetIndex == -1 )
    {
      this->SetPhase( CombatPhase::CPHASE_Decision );
    }
  }
}
break;
```

接下来，我们将修改 `SelectNextCharacter` 以将其 `waitingForCharacter` 设置为 `false`：

```cpp
void CombatEngine::SelectNextCharacter()
{
  this->waitingForCharacter = false;
  for (int i = this->tickTargetIndex; i < this->combatantOrder.
    Num(); i++)
  {
    UGameCharacter* character = this->combatantOrder[i];

    if (character->HP > 0)
    {
      this->tickTargetIndex = i + 1;
      this->currentTickTarget = character;
      return;
    }
  }

  this->tickTargetIndex = -1;
  this->currentTickTarget = nullptr;
}
```

最后，还有一些剩余的细节：我们的战斗引擎应该将所有角色的 `CombatInstance` 指针设置为指向自身，我们将在构造函数中这样做；然后，在析构函数中清除指针，并释放敌人指针。所以首先，在 `GameCharacter.h` 中创建一个指向 `combatInstance` 的指针，在你的 `UProperty` 声明之后和受保护的变量之前：

```cpp
CombatEngine* combatInstance;
```

然后，在 `CombatEngine.cpp` 中，将你的构造函数和析构函数替换为以下内容：

```cpp
CombatEngine::CombatEngine( TArray<UGameCharacter*> playerParty, TArray<UGameCharacter*> enemyParty )
{
  this->playerParty = playerParty;
  this->enemyParty = enemyParty;

  // first add all players to combat order
  for (int i = 0; i < playerParty.Num(); i++)
  {
    this->combatantOrder.Add(playerParty[i]);
  }

  // next add all enemies to combat order
  for (int i = 0; i < enemyParty.Num(); i++)
  {
    this->combatantOrder.Add(enemyParty[i]);
  }

  this->tickTargetIndex = 0;
  this->SetPhase(CombatPhase::CPHASE_Decision);

  for( int i = 0; i < this->combatantOrder.Num(); i++ )
  {
    this->combatantOrder[i]->combatInstance = this;
  }

  this->tickTargetIndex = 0;
  this->SetPhase( CombatPhase::CPHASE_Decision );
}

CombatEngine::~CombatEngine()
{
  // free enemies
  for( int i = 0; i < this->enemyParty.Num(); i++ )
  {
    this->enemyParty[i] = nullptr;
  }

  for( int i = 0; i < this->combatantOrder.Num(); i++ )
  {
    this->combatantOrder[i]->combatInstance = nullptr;
  }
}
```

到目前为止，战斗引擎几乎完全可用。我们仍然需要将其连接到游戏的其他部分，但要以一种可以从游戏模式触发战斗并更新它的方式。

因此，首先在我们的 `RPGGameMode` 类中，我们将添加一个指向当前战斗实例的指针，并重写 `Tick` 函数；此外，跟踪一个敌人角色的列表（用 `UPROPERTY` 装饰，以便敌人可以被垃圾回收）：

```cpp
#pragma once
#include "GameFramework/GameMode.h"
#include "GameCharacter.h"
#include "Combat/CombatEngine.h"
#include "RPGGameMode.generated.h"

UCLASS()
class RPG_API ARPGGameMode : public AGameMode
{
  GENERATED_BODY()

  ARPGGameMode( const class FObjectInitializer& ObjectInitializer );
  virtual void BeginPlay() override;
  virtual void Tick( float DeltaTime ) override;

public:
  CombatEngine* currentCombatInstance;
  TArray<UGameCharacter*> enemyParty;
};
```

接下来，在 `.cpp` 文件中，我们实现 `Tick` 函数：

```cpp
void ARPGGameMode::Tick( float DeltaTime )
{
  if( this->currentCombatInstance != nullptr )
  {
    bool combatOver = this->currentCombatInstance->Tick( DeltaTime );
    if( combatOver )
    {
      if( this->currentCombatInstance->phase == CombatPhase::CPHASE_GameOver )
      {
        UE_LOG( LogTemp, Log, TEXT( "Player loses combat, game over" ) );
                }
      else if( this->currentCombatInstance->phase == CombatPhase::CPHASE_Victory )
      {
        UE_LOG( LogTemp, Log, TEXT( "Player wins combat" ) );
      }

      // enable player actor
      UGameplayStatics::GetPlayerController( GetWorld(), 0 )->SetActorTickEnabled( true );

      delete( this->currentCombatInstance );
      this->currentCombatInstance = nullptr;
      this->enemyParty.Empty();
    }
  }
}
```

目前，这仅仅检查是否当前有战斗实例；如果有，它将调用该实例的 `Tick` 函数。如果它返回 `true`，游戏模式将检查 `Victory` 或 `GameOver`（目前，它只是将消息记录到控制台）。然后，它删除战斗实例，将指针设置为空，并清除敌人队伍列表（这将使敌人有资格进行垃圾回收，因为列表被 `UPROPERTY` 宏装饰）。它还启用了玩家角色的 `Tick`（我们将在战斗开始时禁用 `Tick`，以便玩家角色在战斗期间保持原地不动）。

然而，我们还没有准备好开始战斗遭遇战！玩家没有敌人可以战斗。

我们定义了一个敌人表，但我们的 `GameCharacter` 类不支持从 `EnemyInfo` 初始化。

为了支持这一点，我们将在 `GameCharacter` 类中添加一个新的工厂（确保你也添加了 `EnemyInfo` 类的 `include` 语句）：

```cpp
static UGameCharacter* CreateGameCharacter( FEnemyInfo* enemyInfo, UObject* outer );
```

此外，`GameCharacter.cpp` 中此构造函数重载的实现如下：

```cpp
UGameCharacter* UGameCharacter::CreateGameCharacter( FEnemyInfo* enemyInfo, UObject* outer )
{
  UGameCharacter* character = NewObject<UGameCharacter>( outer );

  character->CharacterName = enemyInfo->EnemyName;
  character->ClassInfo = nullptr;

  character->MHP = enemyInfo->MHP;
  character->MMP = 0;
  character->HP = enemyInfo->MHP;
  character->MP = 0;

  character->ATK = enemyInfo->ATK;
  character->DEF = enemyInfo->DEF;
  character->LUCK = enemyInfo->Luck;

  return character;
}
```

与之相比，这非常简单；只需为 `ClassInfo` 分配名称和空值（因为敌人没有与之关联的类）以及其他统计数据（MMP 和 MP 都设置为零，因为敌人的能力不会消耗 MP）。

为了测试我们的战斗系统，我们将在 `RPGGameMode.h` 中创建一个可以从 Unreal 控制台调用的函数：

```cpp
UFUNCTION(exec)
void TestCombat();
```

`UFUNCTION(exec)` 宏允许此函数作为控制台命令被调用。

这个函数的实现位于 `RPGGameMode.cpp` 中，如下所示：

```cpp
void ARPGGameMode::TestCombat()
{
  // locate enemies asset
  UDataTable* enemyTable = Cast<UDataTable>( StaticLoadObject( UDataTable::StaticClass(), NULL, 
TEXT( "DataTable'/Game/Data/Enemies.Enemies'" ) ) );

  if( enemyTable == NULL )
  {
    UE_LOG( LogTemp, Error, TEXT( "Enemies data table not found!" ) );
    return;
  }

  // locate enemy
  FEnemyInfo* row = enemyTable->FindRow<FEnemyInfo>( TEXT( "S1" ), TEXT( "LookupEnemyInfo" ) );

  if( row == NULL )
  {
    UE_LOG( LogTemp, Error, TEXT( "Enemy ID 'S1' not found!" ) );
    return;
  }

  // disable player actor
  UGameplayStatics::GetPlayerController( GetWorld(), 0 )->SetActorTickEnabled( false );

  // add character to enemy party
  UGameCharacter* enemy = UGameCharacter::CreateGameCharacter( row, this );
  this->enemyParty.Add( enemy );

  URPGGameInstance* gameInstance = Cast<URPGGameInstance>( GetGameInstance() );

  this->currentCombatInstance = new CombatEngine( gameInstance->PartyMembers, this->enemyParty );

  UE_LOG( LogTemp, Log, TEXT( "Combat started" ) );
}
```

它定位敌人数据表，选择 ID 为 `S1` 的敌人，构建一个新的 `GameCharacter`，构建一个敌人列表，添加新的敌人角色，然后创建一个新的 `CombatEngine` 实例，传递玩家团体和敌人列表。它还禁用了玩家演员的 tick，这样当战斗开始时玩家就停止更新。

最后，你应该能够测试战斗引擎。启动游戏并按波浪号 (*~*) 键打开控制台命令文本框。输入 `TestCombat` 并按 *Enter*。

查看输出窗口，你应该看到以下内容：

```cpp
LogTemp: Combat started
LogTemp: Character Kumo making decision
LogTemp: Character Goblin making decision
LogTemp: Character Kumo executing action
LogTemp: Character Goblin executing action
LogTemp: Character Kumo making decision
LogTemp: Character Goblin making decision
LogTemp: Character Kumo executing action
LogTemp: Character Goblin executing action
LogTemp: Character Kumo making decision
LogTemp: Character Goblin making decision
LogTemp: Character Kumo executing action
LogTemp: Character Goblin executing action
LogTemp: Character Kumo making decision

```

这表明战斗引擎按预期工作——首先，所有角色做出决策，执行他们的决策，然后再次做出决策，依此类推。由于实际上没有人真正采取任何行动（更不用说造成任何伤害），目前战斗只是无限期地进行。

这个问题有两个问题：首先，上述问题实际上没有人真正采取任何行动。此外，玩家角色需要有一种不同于敌人的决策方式（玩家角色将需要一个用户界面来选择动作，而敌人应该自动选择动作）。

我们将在处理决策之前解决第一个问题。

## 执行动作

为了允许角色执行动作，我们将所有战斗动作简化为单个通用接口。一个好的开始是让这个接口映射到我们已有的东西——也就是说，角色的 `BeginExecuteAction` 和 `ExecuteAction` 函数。

让我们为这个创建一个新的 `ICombatAction` 接口，它可以从一个没有任何父类的类开始，并放在一个名为 `Source/RPG/Combat/Actions` 的新路径中；`ICombatAction.h` 文件应该看起来像这样：

```cpp
#pragma once

#include "GameCharacter.h"

class UGameCharacter;

class ICombatAction
{
public:
  virtual void BeginExecuteAction( UGameCharacter* character ) = 0;
  virtual bool ExecuteAction( float DeltaSeconds ) = 0;
};
```

`BeginExecuteAction` 接收执行此动作的角色指针。`ExecuteAction` 如前所述，接收上一帧的持续时间（以秒为单位）。

在 `ICombatAction.cpp` 中，移除默认构造函数和析构函数，使文件看起来像这样：

```cpp
#include "RPG.h"
#include "ICombatAction.h"
```

然后，我们可以创建一个新的空 C++ 类来实现这个接口。仅作为一个测试，我们将在一个名为 `TestCombatAction` 的新类中复制角色已经执行的功能（即，绝对不做任何事情），并将其路径设置为 `Source/RPG/Combat/Actions` 文件夹。

首先，头文件将如下所示：

```cpp
#pragma once

#include "ICombatAction.h"

class TestCombatAction : public ICombatAction
{
protected:
  float delayTimer;

public:
  virtual void BeginExecuteAction( UGameCharacter* character ) override;
  virtual bool ExecuteAction( float DeltaSeconds ) override;
};
```

`.cpp` 文件将如下所示：

```cpp
#include "RPG.h"
#include "TestCombatAction.h"

void TestCombatAction::BeginExecuteAction( UGameCharacter* character )
{
  UE_LOG( LogTemp, Log, TEXT( "%s does nothing" ), *character->CharacterName );
  this->delayTimer = 1.0f;
}

bool TestCombatAction::ExecuteAction( float DeltaSeconds )
{
  this->delayTimer -= DeltaSeconds;
  return this->delayTimer <= 0.0f;
}
```

接下来，我们将更改角色，使其能够存储和执行动作。

首先，让我们用战斗动作指针替换测试延迟计时器字段。稍后，我们将使其在 `GameCharacter.h` 中创建决策系统时公开：

```cpp
public:
  ICombatAction* combatAction;
```

还记得在 `GameCharacter.h` 的顶部包含 `ICombatAction`，然后声明 `ICombatAction` 类：

```cpp
#pragma once

#include "Data/FCharacterInfo.h"
#include "Data/FEnemyInfo.h"
#include "Data/FCharacterClassInfo.h"
#include "Combat/Actions/ICombatAction.h"
#include "GameCharacter.generated.h"

class CombatEngine;
class ICombatAction;
```

接下来，我们需要更改我们的决策函数以分配战斗动作，并将动作函数执行此动作在 `GameCharacter.cpp` 中：

```cpp
void UGameCharacter::BeginMakeDecision()
{
  UE_LOG( LogTemp, Log, TEXT( "Character %s making decision" ), *( this->CharacterName ) );
  this->combatAction = new TestCombatAction();
}

bool UGameCharacter::MakeDecision( float DeltaSeconds )
{
  return true;
}

void UGameCharacter::BeginExecuteAction()
{
  this->combatAction->BeginExecuteAction( this );
}

bool UGameCharacter::ExecuteAction( float DeltaSeconds )
{
  bool finishedAction = this->combatAction->ExecuteAction( DeltaSeconds );
  if( finishedAction )
  {
    delete( this->combatAction );
    return true;
  }

  return false;
}
```

还要记得在`GameCharacter.cpp`的顶部使用`include TestCombatAction`：

```cpp
#include "Combat/Actions/TestCombatAction.h"
```

`BeginMakeDecision`现在分配一个新的`TestCombatAction`实例。`MakeDecision`仅返回`true`。`BeginExecuteAction`调用存储的战斗动作上的同名函数，并将角色作为指针传递。最后，`ExecuteAction`调用同名函数，如果结果是`true`，则删除指针并返回`true`；否则返回`false`。

通过运行此代码并测试战斗，你应该得到几乎相同的输出，但现在它显示的是`does nothing`而不是`executing action`。

现在我们有了一种让角色存储和执行动作的方法，我们可以着手为角色开发一个决策系统。

## 做出决策

就像我们对动作所做的那样，我们将创建一个用于决策的接口，其模式与`BeginMakeDecision`和`MakeDecision`函数相似。类似于`ICombatAction`类，我们将创建一个空的`IDecisionMaker`类，并将其放置到新的目录`Source/RPG/Combat/DecisionMakers`中。以下将是`IDecisionMaker.h`的内容：

```cpp
#pragma once

#include "GameCharacter.h"

class UGameCharacter;

class IDecisionMaker
{
public:
  virtual void BeginMakeDecision( UGameCharacter* character ) = 0;
  virtual bool MakeDecision( float DeltaSeconds ) = 0;
};
```

此外，从`IDecisionMaker.cpp`中删除构造函数和析构函数，使其看起来像这样：

```cpp
#include "RPG.h"
#include "IDecisionMaker.h"
```

现在，我们可以创建`TestDecisionMaker` C++类，并将其放置到`Source/RPG/Combat/DecisionMakers`目录中。然后，按照以下方式编程`TestDecisionMaker.h`：

```cpp
#pragma once
#include "IDecisionMaker.h"

class RPG_API TestDecisionMaker : public IDecisionMaker
{
public:
  virtual void BeginMakeDecision( UGameCharacter* character ) override;
  virtual bool MakeDecision( float DeltaSeconds ) override;
};
```

然后，按照以下方式编程`TestDecisionMaker.cpp`：

```cpp
#include "RPG.h"
#include "TestDecisionMaker.h"

#include "../Actions/TestCombatAction.h"

void TestDecisionMaker::BeginMakeDecision( UGameCharacter* character )
{
  character->combatAction = new TestCombatAction();
}

bool TestDecisionMaker::MakeDecision( float DeltaSeconds )
{
  return true;
}
```

接下来，我们将在游戏角色类中添加一个指向`IDecisionMaker`的指针，并修改`BeginMakeDecision`和`MakeDecision`函数以在`GameCharacter.h`中使用决策者：

```cpp
public:
  IDecisionMaker* decisionMaker;
```

还要记得在`GameCharacter.h`的顶部包含`ICombatAction`，然后声明`ICombatAction`类：

```cpp
#pragma once

#include "Data/FCharacterInfo.h"
#include "Data/FEnemyInfo.h"
#include "Data/FCharacterClassInfo.h"
#include "Combat/Actions/ICombatAction.h"
#include "Combat/DecisionMakers/IDecisionMaker.h"
#include "GameCharacter.generated.h"

class CombatEngine;
class ICombatAction;
class IDecisionMaker;
```

接下来，将`GameCharacter.cpp`中的`BeginDestroy`、`BeginMakeDecision`和`MakeDecision`函数替换为以下内容：

```cpp
void UGameCharacter::BeginDestroy()
{
  Super::BeginDestroy();
  delete( this->decisionMaker );
}

void UGameCharacter::BeginMakeDecision()
{
  this->decisionMaker->BeginMakeDecision( this );}

bool UGameCharacter::MakeDecision( float DeltaSeconds )
{
  return this->decisionMaker->MakeDecision( DeltaSeconds );
}
```

注意，我们在析构函数中删除决策者。决策者将在角色创建时分配，因此当角色释放时应该删除。

然后，我们将包含`TestDecisionMaker`实现，以便每个阵营都能做出战斗决策，因此请在类的顶部包含`TestDecisionMaker`：

```cpp
#include "Combat/DecisionMakers/TestDecisionMaker.h"
```

在这里，最终的步骤是为角色的构造函数分配一个决策者。为两个构造函数重载添加以下代码行：`character->decisionMaker = new TestDecisionMaker();`。当你完成时，玩家和敌人角色的构造函数应该看起来像这样：

```cpp
UGameCharacter* UGameCharacter::CreateGameCharacter(
  FCharacterInfo* characterInfo, UObject* outer)
{
  UGameCharacter* character = NewObject<UGameCharacter>(outer);

  // locate character classes asset
  UDataTable* characterClasses = Cast<UDataTable>(
    StaticLoadObject(UDataTable::StaticClass(), NULL, TEXT(
      "DataTable'/Game/Data/CharacterClasses.CharacterClasses'"))
    );

  if (characterClasses == NULL)
  {
    UE_LOG(LogTemp, Error, 
TEXT("Character classes datatable not found!" ) );
  }
  else
  {
    character->CharacterName = characterInfo->Character_Name;
    FCharacterClassInfo* row = 
characterClasses->FindRow<FCharacterClassInfo>
(*(characterInfo->Class_ID), TEXT("LookupCharacterClass"));
    character->ClassInfo = row;

    character->MHP = character->ClassInfo->StartMHP;
    character->MMP = character->ClassInfo->StartMMP;
    character->HP = character->MHP;
    character->MP = character->MMP;

    character->ATK = character->ClassInfo->StartATK;
    character->DEF = character->ClassInfo->StartDEF;
    character->LUCK = character->ClassInfo->StartLuck;

    character->decisionMaker = new TestDecisionMaker();
  }

  return character;
}

UGameCharacter* UGameCharacter::CreateGameCharacter(FEnemyInfo* enemyInfo, UObject* outer)
{
  UGameCharacter* character = NewObject<UGameCharacter>(outer);

  character->CharacterName = enemyInfo->EnemyName;
  character->ClassInfo = nullptr;

  character->MHP = enemyInfo->MHP;
  character->MMP = 0;
  character->HP = enemyInfo->MHP;
  character->MP = 0;

  character->ATK = enemyInfo->ATK;
  character->DEF = enemyInfo->DEF;
  character->LUCK = enemyInfo->Luck;

  character->decisionMaker = new TestDecisionMaker();

  return character;
}
```

运行游戏并再次测试战斗，你应该得到与之前非常相似的输出。然而，最大的不同之处在于现在可以为不同的角色分配不同的决策者实现，并且这些决策者有简单的方法来分配要执行的战斗动作。例如，现在将我们的测试战斗动作处理目标伤害将变得很容易。然而，在我们这样做之前，让我们对`GameCharacter`类做一些小的更改。

## 选择目标

我们将在`GameCharacter`中添加一个字段来标识角色是玩家还是敌人。此外，我们还将添加一个`SelectTarget`函数，该函数从当前的战斗实例的`enemyParty`或`playerParty`中选择第一个活着的角色，具体取决于该角色是玩家还是敌人。

首先，在`GameCharacter.h`中，我们将添加一个公共的`isPlayer`字段：

```cpp
bool isPlayer;
```

然后，我们将添加一个`SelectTarget`函数，如下所示：

```cpp
UGameCharacter* SelectTarget();
```

在`GameCharacter.cpp`中，我们将在构造函数中分配`isPlayer`字段（这很简单，因为我们有玩家和敌人分开的构造函数）：

```cpp
UGameCharacter* UGameCharacter::CreateGameCharacter(
  FCharacterInfo* characterInfo, UObject* outer)
{
  UGameCharacter* character = NewObject<UGameCharacter>(outer);

  // locate character classes asset
  UDataTable* characterClasses = Cast<UDataTable>(
    StaticLoadObject(UDataTable::StaticClass(), NULL, TEXT(
      "DataTable'/Game/Data/CharacterClasses.CharacterClasses'"))
    );

  if (characterClasses == NULL)
  {
    UE_LOG(LogTemp, Error,
      TEXT("Character classes datatable not found!"));
  }
  else
  {
    character->CharacterName = characterInfo->Character_Name;
    FCharacterClassInfo* row =
      characterClasses->FindRow<FCharacterClassInfo>
(*(characterInfo->Class_ID), TEXT("LookupCharacterClass"));
    character->ClassInfo = row;

    character->MHP = character->ClassInfo->StartMHP;
    character->MMP = character->ClassInfo->StartMMP;
    character->HP = character->MHP;
    character->MP = character->MMP;

    character->ATK = character->ClassInfo->StartATK;
    character->DEF = character->ClassInfo->StartDEF;
    character->LUCK = character->ClassInfo->StartLuck;

    character->decisionMaker = new TestDecisionMaker();
  }
  character->isPlayer = true;
  return character;
}

UGameCharacter* UGameCharacter::CreateGameCharacter(FEnemyInfo* enemyInfo, UObject* outer)
{
  UGameCharacter* character = NewObject<UGameCharacter>(outer);

  character->CharacterName = enemyInfo->EnemyName;
  character->ClassInfo = nullptr;

  character->MHP = enemyInfo->MHP;
  character->MMP = 0;
  character->HP = enemyInfo->MHP;
  character->MP = 0;

  character->ATK = enemyInfo->ATK;
  character->DEF = enemyInfo->DEF;
  character->LUCK = enemyInfo->Luck;

  character->decisionMaker = new TestDecisionMaker();
  character->isPlayer = false;
  return character;
}
```

最后，`SelectTarget`函数如下所示：

```cpp
UGameCharacter* UGameCharacter::SelectTarget()
{
  UGameCharacter* target = nullptr;

  TArray<UGameCharacter*> targetList = this->combatInstance->enemyParty;
  if( !this->isPlayer )
  {
    targetList = this->combatInstance->playerParty;
  }

  for( int i = 0; i < targetList.Num(); i++ )
  {
    if( targetList[ i ]->HP > 0 )
    {
      target = targetList[i];
      break;
    }
  }

  if( target->HP <= 0 )
  {
    return nullptr;
  }

  return target;
}
```

这首先确定使用哪个列表（敌人或玩家）作为潜在的目标，然后遍历该列表以找到第一个非死亡目标。如果没有目标，此函数返回一个空指针。

## 造成伤害

现在有了选择目标的方法，让我们让`TestCombatAction`类最终造成一些伤害！

我们将添加一些字段来维护对角色和目标的引用，并添加一个接受目标作为参数的构造函数：

```cpp
protected:
  UGameCharacter* character;
  UGameCharacter* target;

public:
  TestCombatAction( UGameCharacter* target );
```

此外，实现方式是通过在`TestCombatAction.cpp`中创建和更新`BeginExecuteAction`函数，如下所示：

```cpp
void TestCombatAction::BeginExecuteAction( UGameCharacter* character )
{
  this->character = character;

  // target is dead, select another target
  if( this->target->HP <= 0 )
  {
    this->target = this->character->SelectTarget();
  }

  // no target, just return
  if( this->target == nullptr )
  {
    return;
  }

  UE_LOG( LogTemp, Log, TEXT( "%s attacks %s" ), *character->CharacterName, *target->CharacterName );

  target->HP -= 10;

  this->delayTimer = 1.0f;
}
```

然后让类的构造函数设置目标：

```cpp
TestCombatAction::TestCombatAction(UGameCharacter* target)
{
  this->target = target;
}
```

首先，构造函数分配目标指针。然后，`BeginExecuteAction`函数分配角色引用并检查目标是否存活。如果目标是死亡的，它将通过我们刚刚创建的`SelectTarget`函数选择一个新的目标。如果目标指针现在是空，则没有目标，此函数仅返回空。否则，它记录一个类似*[character] attacks [target]*的消息，从目标中减去一些 HP，并设置延迟计时器，就像之前一样。

下一步是将我们的`TestDecisionMaker`更改为选择一个目标并将其传递给`TestCombatAction`构造函数。这在`TestDecisionMaker.cpp`中是一个相对简单的更改：

```cpp
void TestDecisionMaker::BeginMakeDecision( UGameCharacter* character )
{
  // pick a target
  UGameCharacter* target = character->SelectTarget();
  character->combatAction = new TestCombatAction( target );
}
```

到目前为止，你应该能够运行游戏，开始测试遭遇战，并看到以下类似的输出：

```cpp
LogTemp: Combat started
LogTemp: Kumo attacks Goblin
LogTemp: Goblin attacks Kumo
LogTemp: Kumo attacks Goblin
LogTemp: Player wins combat

```

最后，我们有一个战斗系统，其中我们的两个阵营可以互相攻击，一方或另一方可以获胜。

接下来，我们将开始将其连接到用户界面。

## 使用 UMG 的战斗 UI

要开始，我们需要设置我们的项目以正确导入 UMG 和 Slate 相关的类。

首先，打开 `RPG.Build.cs`（或 `[ProjectName].Build.cs`）并将构造函数的第一行代码更改为以下代码：

```cpp
PublicDependencyModuleNames.AddRange( new string[] { "Core", "CoreUObject", "Engine", "InputCore", "UMG", "Slate", "SlateCore" } );
```

这将 `UMG`、`Slate` 和 `SlateCore` 字符串添加到现有的字符串数组中。

接下来，打开 `RPG.h` 并确保以下代码行存在：

```cpp
#include "Runtime/UMG/Public/UMG.h"
#include "Runtime/UMG/Public/UMGStyle.h"
#include "Runtime/UMG/Public/Slate/SObjectWidget.h"
#include "Runtime/UMG/Public/IUMGModule.h"
#include "Runtime/UMG/Public/Blueprint/UserWidget.h"
```

现在编译项目。这可能需要一段时间。

接下来，我们将创建一个用于战斗 UI 的基类。基本上，我们将使用这个基类来允许我们的 C++ 游戏代码通过在头文件中定义蓝图可实现的函数与蓝图 UMG 代码进行通信，这些函数是蓝图可以实现的函数，可以从 C++ 中调用。

创建一个名为 `CombatUIWidget` 的新类，并将其父类选择为 `UserWidget`；然后将其路径设置为 `Source/RPG/UI`。用以下代码替换 `CombatUIWidget.h` 中的内容：

```cpp
#pragma once
#include "GameCharacter.h"

#include "Blueprint/UserWidget.h"
#include "CombatUIWidget.generated.h"

UCLASS()
class RPG_API UCombatUIWidget : public UUserWidget
{
  GENERATED_BODY()

public:
  UFUNCTION( BlueprintImplementableEvent, Category = "Combat UI" )
  void AddPlayerCharacterPanel( UGameCharacter* target );

  UFUNCTION( BlueprintImplementableEvent, Category = "Combat UI" )
  void AddEnemyCharacterPanel( UGameCharacter* target );
};
```

在大多数情况下，我们只是在定义几个函数。`AddPlayerCharacterPanel` 和 `AddEnemyCharacterPanel` 函数将负责接受一个角色指针并为该角色生成一个小部件（以显示角色的当前状态）。

编译代码后，回到编辑器中，在 `Contents/Blueprints` 目录中创建一个名为 `UI` 的新文件夹。在 `Content/Blueprints/UI` 目录中，创建一个名为 `CombatUI` 的新 Widget 蓝图。在创建并打开蓝图后，转到 **文件** | **重新父化蓝图** 并选择 **CombatUIWidget** 作为父类。

在 **设计师** 界面中，创建两个水平框小部件并将它们命名为 `enemyPartyStatus` 和 `playerPartyStatus`。这些将分别持有敌人和玩家的子小部件，以显示每个角色的状态。对于这两个，务必确保启用 **Is Variable** 复选框，这样它们就会作为变量对蓝图可用。保存并编译蓝图。

我们将 `enemyPartyStatus` 水平框定位在画布面板的顶部。首先设置一个顶部水平锚点会有所帮助。

然后将水平框的值设置为以下内容，**偏移左**: 10，**位置 Y**: 10，**偏移右**: 10，**大小 Y**: 200。

以类似的方式定位 `playerPartyStatus` 水平框；唯一的重大区别是我们将框锚定在画布面板的底部，并定位使其跨越屏幕底部：

![带有 UMG 的战斗 UI](img/B04548_03_07.jpg)

接下来，我们将创建用于显示玩家和敌人角色状态的小部件。首先，我们将创建一个基小部件，每个小部件都将从中继承。

创建一个新的 Widget 蓝图并将其命名为 `BaseCharacterCombatPanel`。在这个蓝图里，导航到图表，然后从 **MyBlueprint** 选项卡添加一个新的变量，**CharacterTarget**，并从 **对象引用** 类别中选择 **Game Character** 变量类型。

接下来，我们将为敌人和玩家创建单独的小部件。

创建一个新的 Widget 蓝图并将其命名为 `PlayerCharacterCombatPanel`。将新蓝图的父母设置为 `BaseCharacterCombatPanel`。

在**Designer**界面中添加三个文本小部件。一个标签用于角色的名称，另一个用于角色的 HP，第三个用于角色的 MP。将每个文本块定位在屏幕的左下角，并且完全在`playerPartyStatus`框大小的 200 高像素内，这是我们之前在`CombatUI`小部件中创建的：

![战斗 UI 与 UMG](img/B04548_03_08.jpg)

确保检查每个文本块的**Details**面板中的**Size to Content**，以便文本块可以根据内容调整大小，如果内容不适合文本块参数。

通过选择小部件并点击**Details**面板中**Text**输入旁边的**Bind**来为这些创建新的绑定：

![战斗 UI 与 UMG](img/B04548_03_09.jpg)

这将创建一个新的蓝图函数，该函数将负责生成文本块。

要绑定 HP 文本块，例如，您可以执行以下步骤：

1.  在网格的空白区域右键单击，搜索**Get Character Target**，然后选择它。

1.  将此节点的输出引脚拖动并选择**Variables** | **Character Info**下的**Get HP**。

1.  创建一个新的**Format Text**节点。将文本设置为**HP: {HP}**，然后将**Get HP**的输出连接到**Format Text**节点的**HP**输入。

1.  最后，将**Format Text**节点的输出连接到**Return**节点的**Return**值。

您可以重复类似的步骤为角色名称和 MP 文本块。

在您创建`PlayerCharacterCombatPanel`之后，您可以重复相同的步骤来创建`EnemyCharacterCombatPanel`，除了不需要 MP 文本块（如前所述，敌人不消耗 MP）。唯一的重大区别是`EnemyCharacterCombatPanel`中的文本块需要放置在屏幕顶部，以匹配`CombatUI`小部件中的`enemyPartyStatus`水平框的位置。

显示 MP 的结果图将类似于以下截图：

![战斗 UI 与 UMG](img/B04548_03_10.jpg)

现在我们已经有了玩家和敌人的小部件，让我们在`CombatUI`蓝图实现`AddPlayerCharacterPanel`和`AddEnemyCharacterPanel`函数。

首先，我们将创建一个辅助蓝图函数来生成角色状态小部件。将此新函数命名为`SpawnCharacterWidget`，并将以下参数添加到输入中：

+   **目标角色**（类型为 Game Character Reference）

+   **目标面板**（类型为 Panel Widget Reference）

+   **Class**（类型为 Base Character Combat Panel Class）

此函数将执行以下步骤：

1.  使用**Create Widget**创建给定类的新小部件。

1.  将新小部件投射到`BaseCharacterCombatPanel`类型。

1.  将结果的**Character Target**设置为**TargetCharacter**输入。

1.  将新小部件作为**TargetPanel**输入的子项添加。

在蓝图中的样子如下所示：

![战斗 UI 与 UMG](img/B04548_03_11.jpg)

接下来，在`CombatUI`蓝图的事件图中，右键单击并添加`EventAddPlayerCharacterPanel`和`EventAddEnemyCharacterPanel`事件。将每个事件连接到`SpawnCharacterWidget`节点，将**目标**输出连接到**目标角色**输入，将适当的面板变量连接到**目标面板**输入，如下所示：

![带有 UMG 的战斗 UI](img/B04548_03_12.jpg)

最后，我们可以在战斗开始时从我们的游戏模式中生成这个 UI，并在战斗结束时销毁它。在`RPGGameMode`的头部添加一个指向`UCombatUIWidget`的指针，并添加一个用于生成战斗 UI 的类（这样我们就可以选择继承自我们的`CombatUIWidget`类的 Widget 蓝图）；这些应该是公共的：

```cpp
UPROPERTY()
UCombatUIWidget* CombatUIInstance;

UPROPERTY( EditDefaultsOnly, BlueprintReadOnly, Category = "UI" )
TSubclassOf<class UCombatUIWidget> CombatUIClass;
```

还要确保`RPGGameMode.h`包含`CombatWidget`；在这个时候，`RPGGameMode.h`顶部的内容列表应该看起来像这样：

```cpp
#include "GameFramework/GameMode.h"
#include "GameCharacter.h"
#include "Combat/CombatEngine.h"
#include "UI/CombatUIWidget.h"
#include "RPGGameMode.generated.h"
```

在`RPGGameMode.cpp`中的`TestCombat`函数结束时，我们将生成这个小部件的新实例，如下所示：

```cpp
this->CombatUIInstance = CreateWidget<UCombatUIWidget>( GetGameInstance(), this->CombatUIClass );
this->CombatUIInstance->AddToViewport();

UGameplayStatics::GetPlayerController(GetWorld(), 0)
->bShowMouseCursor = true;

for( int i = 0; i < gameInstance->PartyMembers.Num(); i++ )
  this->CombatUIInstance->AddPlayerCharacterPanel( gameInstance->PartyMembers[i] );

for( int i = 0; i < this->enemyParty.Num(); i++ )
  this->CombatUIInstance->AddEnemyCharacterPanel( this->enemyParty[i] );
```

这将创建小部件，将其添加到视图中，添加鼠标光标，然后分别调用其`AddPlayerCharacterPanel`和`AddEnemyCharacterPanel`函数，为所有玩家和敌人。

战斗结束后，我们将从视图中移除小部件，并将引用设置为 null，以便它可以被垃圾回收；你的`Tick`函数现在应该看起来像这样：

```cpp
void ARPGGameMode::Tick(float DeltaTime)
{
  if (this->currentCombatInstance != nullptr)
  {
    bool combatOver = this->currentCombatInstance->Tick(DeltaTime
    );
    if (combatOver)
    {
      if (this->currentCombatInstance->phase == CombatPhase::
        CPHASE_GameOver)
      {
        UE_LOG(LogTemp, Log, 
        TEXT("Player loses combat, game over" ) );
      }
      else if 
      (this->currentCombatInstance->phase == 
      CombatPhase::  CPHASE_Victory)
      {
        UE_LOG(LogTemp, Log, TEXT("Player wins combat"));
      }
      UGameplayStatics::GetPlayerController(GetWorld(),0)
      ->bShowMouseCursor = false;

      // enable player actor
      UGameplayStatics::GetPlayerController(GetWorld(), 0)->
        SetActorTickEnabled(true);

      this->CombatUIInstance->RemoveFromViewport();
      this->CombatUIInstance = nullptr;

      delete(this->currentCombatInstance);
      this->currentCombatInstance = nullptr;
      this->enemyParty.Empty();
    }
  }
}
```

在这个阶段，你可以编译，但如果测试战斗，游戏将会崩溃。这是因为你需要设置`DefaultRPGGameMode`类的默认值，使用`CombatUI`作为你在`RPGGameMode.h`中创建的`CombatUIClass`。否则，系统将不知道`CombatUIClass`变量应该指向`CombatUI`，这是一个小部件，因此无法创建它。请注意，编辑器在执行此步骤时可能会崩溃。

![带有 UMG 的战斗 UI](img/B04548_03_13.jpg)

现在，如果你运行游戏并开始战斗，你应该能看到哥布林的状态和玩家的状态。两者的生命值都应该会减少，直到哥布林的生命值达到零；在这个时候，用户界面将消失（因为战斗已经结束）。

接下来，我们将进行一些更改，使得玩家角色不再是自动做出决策，而是玩家可以通过用户界面选择他们的行动。

## UI 驱动的决策制定

一个想法是改变决策者分配给玩家的方式——而不是在玩家首次创建时分配，我们可以在战斗开始时让我们的`CombatUIWidget`类实现决策者，并在战斗开始时分配它（在战斗结束时清除指针）。

我们需要对`GameCharacter.cpp`进行一些更改。首先，在`CreateGameCharacter`的玩家重载中，删除以下代码行：

```cpp
character->decisionMaker = new TestDecisionMaker();
```

然后，在`BeginDestroy`函数中，我们将`delete`行包裹在一个`if`语句中：

```cpp
if( !this->isPlayer )
  delete( this->decisionMaker );
```

原因是玩家的决策者将是 UI——我们不希望手动删除 UI（这样做会导致 Unreal 崩溃）。相反，只要没有用`UPROPERTY`装饰的指针指向它，UI 将自动进行垃圾回收。

接下来，在`CombatUIWidget.h`中，我们将使类实现`IDecisionMaker`接口，并添加`BeginMakeDecision`和`MakeDecision`作为公共函数：

```cpp
#pragma once
#include "GameCharacter.h"
#include "Blueprint/UserWidget.h"
#include "CombatUIWidget.generated.h"

UCLASS()
class RPG_API UCombatUIWidget : public UUserWidget, public IDecisionMaker
{
  GENERATED_BODY()

public:
  UFUNCTION(BlueprintImplementableEvent, Category = "Combat UI")
    void AddPlayerCharacterPanel(UGameCharacter* target);

  UFUNCTION(BlueprintImplementableEvent, Category = "Combat UI")
    void AddEnemyCharacterPanel(UGameCharacter* target);

  void BeginMakeDecision(UGameCharacter* target);
  bool MakeDecision(float DeltaSeconds);
};
```

我们还将添加几个辅助函数，这些函数可以在我们的 UI 蓝图图中调用：

```cpp
public:
  UFUNCTION( BlueprintCallable, Category = "Combat UI" )
  TArray<UGameCharacter*> GetCharacterTargets();

  UFUNCTION( BlueprintCallable, Category = "Combat UI" )
  void AttackTarget( UGameCharacter* target );
```

第一个函数检索当前角色的潜在目标列表。第二个函数将为角色提供一个带有指定目标的新的`TestCombatAction`。

此外，我们还将添加一个在蓝图中实现的功能，用于显示当前角色的操作集：

```cpp
UFUNCTION( BlueprintImplementableEvent, Category = "Combat UI" )
void ShowActionsPanel( UGameCharacter* target );
```

我们还将添加一个标志和`currentTarget`的定义，如下所示：

```cpp
protected:
 UGameCharacter* currentTarget;
  bool finishedDecision;
```

这将用于表示已做出决策（并且`MakeDecision`应该返回`true`）。

这些四个函数的实现相当简单，在`CombatUIWidget.cpp`中：

```cpp
#include "RPG.h"
#include "CombatUIWidget.h"
#include "../Combat/CombatEngine.h"
#include "../Combat/Actions/TestCombatAction.h"

void UCombatUIWidget::BeginMakeDecision( UGameCharacter* target )
{
  this->currentTarget = target;
  this->finishedDecision = false;

  ShowActionsPanel( target );
}

bool UCombatUIWidget::MakeDecision( float DeltaSeconds )
{
  return this->finishedDecision;
}

void UCombatUIWidget::AttackTarget( UGameCharacter* target )
{
  TestCombatAction* action = new TestCombatAction( target );
  this->currentTarget->combatAction = action;

  this->finishedDecision = true;
}

TArray<UGameCharacter*> UCombatUIWidget::GetCharacterTargets()
{
  if( this->currentTarget->isPlayer )
  {
    return this->currentTarget->combatInstance->enemyParty;
  }
  else
  {
    return this->currentTarget->combatInstance->playerParty;
  }
}
```

`BeginMakeDecision`设置当前目标，将`finishedDecision`标志设置为`false`，然后调用`ShowActionsPanel`（这将在我们的 UI 蓝图图中处理）。

`MakeDecision`简单地返回`finishedDecision`标志的值。

`AttackTarget`将一个新的`TestCombatAction`分配给角色，并将`finishedDecision`设置为`true`以表示已做出决策。

最后，`GetCharacterTargets`返回一个包含此角色可能对手的数组。

由于 UI 现在实现了`IDecisionMaker`接口，我们可以将其分配为玩家角色的决策者。首先，在`RPGGameMode.cpp`中的`TestCombat`函数，我们将改变遍历角色的循环，使其将 UI 分配为决策者：

```cpp
for( int i = 0; i < gameInstance->PartyMembers.Num(); i++ )
{
  this->CombatUIInstance->AddPlayerCharacterPanel( gameInstance->PartyMembers[i] );
  gameInstance->PartyMembers[i]->decisionMaker = this->CombatUIInstance;
}
```

然后，当战斗结束时，我们将玩家的决策者设置为 null：

```cpp
for( int i = 0; i < this->currentCombatInstance->playerParty.Num(); i++ )
{
  this->currentCombatInstance->playerParty[i]->decisionMaker = nullptr;
}
```

现在，玩家角色将使用 UI 来做出决策。然而，UI 目前什么也不做。我们需要在蓝图编辑器中添加这个功能。

首先，我们将创建一个用于攻击目标选项的小部件。命名为`AttackTargetOption`，添加一个按钮，并在按钮中放置一个文本块。勾选**大小适应内容**，以便按钮可以动态调整大小以适应按钮中的任何文本块。然后将其放置在画布面板的左上角。

在图中添加两个新的变量。一个是战斗 UI 引用类型的`targetUI`。另一个是游戏角色引用类型的`target`。从**设计师**视图，点击你的按钮，然后滚动到**详情**面板并点击**OnClicked**来为按钮创建一个事件。按钮将使用`targetUI`引用来调用**攻击目标**函数，并将`target`引用（即此按钮代表的目标）传递给**攻击目标**函数。

按钮点击事件的图相当简单；只需将执行路由到分配的`targetUI`的**攻击目标**函数，并将`target`引用作为参数传递：

![UI 驱动的决策制定](img/B04548_03_14.jpg)

接下来，我们将为主战斗 UI 添加一个用于角色动作的面板。这是一个包含单个用于**攻击**的按钮子项和用于目标列表的垂直框的画布面板：

![UI 驱动的决策制定](img/B04548_03_15.jpg)

将**攻击**按钮命名为`attackButton`。将垂直框命名为`targets`。将封装这些项的画布面板命名为`characterActions`。这些应该启用**是变量**，以便它们对蓝图可见。

然后，在蓝图图中，我们将实现**显示动作面板**事件。这首先将执行路由到**设置可见性**节点，该节点将启用**动作**面板，然后路由执行到另一个**设置可见性**节点，该节点将隐藏目标列表：

![UI 驱动的决策制定](img/B04548_03_16.jpg)

当点击**攻击**按钮时的蓝图图相当大，所以我们将分块查看它。

首先，通过在**设计师**视图中选择按钮并点击**详情**面板的**事件**部分的**OnClicked**来为你的`attackButton`创建一个`OnClicked`事件。在图中，我们然后使用一个**清除子项**节点在按钮点击时清除可能之前添加的任何目标选项：

![UI 驱动的决策制定](img/B04548_03_17.jpg)

然后，我们使用一个**ForEachLoop**和一个**CompareInt**节点结合使用，遍历由**Get Character Targets**返回的所有 HP > 0（未死亡）的角色。

![UI 驱动的决策制定](img/B04548_03_18.jpg)

从**CompareInt**节点的**>**（大于）引脚，我们创建一个新的**AttackTargetOption**小部件实例，并将其添加到攻击目标列表的垂直框中：

![UI 驱动的决策制定](img/B04548_03_19.jpg)

然后，对于我们刚刚添加的小部件，我们将一个**Self**节点连接到它，以设置其`targetUI`变量，并将**ForEachLoop**的**数组元素**引脚传递给它以设置其`target`变量：

![UI 驱动的决策制定](img/B04548_03_20.jpg)

最后，从**完成**的**ForEachLoop**引脚，我们将目标选项列表的可见性设置为**可见**：

![UI 驱动的决策制定](img/B04548_03_21.jpg)

在完成所有这些之后，我们仍然需要在选择动作时隐藏**动作面板**。我们将在`CombatUI`中添加一个名为**隐藏动作面板**的新函数。这个函数非常简单；它只是将动作面板的可见性设置为**隐藏**：

![UI 驱动的决策制定](img/B04548_03_22.jpg)

此外，在**AttackTargetOption**图中的点击处理程序中，我们将**攻击目标**节点的执行引脚连接到这个**隐藏动作面板**函数：

![UI 驱动的决策制定](img/B04548_03_23.jpg)

最后，你需要将位于 **AttackTargetOption** 小部件中的按钮中的文本块绑定。所以进入 **设计器** 视图并创建一个与本章中之前创建的文本块相同的绑定。现在在图中，将 **目标** 连接到 **角色名称**，并调整文本的格式以显示 `CharacterName` 变量，并将其连接到文本的 **返回** 节点。这个 Blueprint 应该在按钮上显示当前目标的角色名称：

![UI 驱动的决策制定](img/B04548_03_24.jpg)

在完成所有这些之后，你应该能够运行游戏并开始测试遭遇战，在玩家的回合，你会看到一个 **攻击** 按钮允许你选择攻击哥布林。

我们的游戏引擎现在完全功能化。本章的最后一步将是创建一个游戏结束界面，这样当所有团队成员都死亡时，玩家将看到 **游戏结束** 信息。

## 创建游戏结束界面

第一步是创建屏幕本身。创建一个新的 Widget Blueprint，命名为 **GameOverScreen**。我们只需添加一个图像，我们可以将其设置为全屏锚点，并在 **详细信息** 面板中将偏移量设置为 0。你也可以将颜色设置为黑色。还可以添加一个带有文本 **Game Over** 的文本块和一个带有子文本块 **Restart** 的按钮：

![创建游戏结束界面](img/B04548_03_25.jpg)

为 **Restart** 按钮创建一个 `OnClicked` 事件。在 Blueprint 图中，将按钮的事件链接到重启游戏，其目标是 **获取游戏模式**（你可能需要取消选中 **上下文相关** 以找到此节点）：

![创建游戏结束界面](img/B04548_03_26.jpg)

你还需要在这里显示鼠标光标。最好的方法是使用 **事件构造**；链接 **设置显示鼠标光标**，其目标是 **获取玩家控制器**。务必勾选 **显示鼠标光标** 复选框。在 **事件构造** 和 **设置显示鼠标光标** 之间放置一个 0.2 秒的延迟，以确保在战斗结束后移除鼠标后鼠标重新出现：

![创建游戏结束界面](img/B04548_03_27.jpg)

接下来，在 `RPGGameMode.h` 中，我们添加一个用于游戏结束的公共属性来指定小部件类型：

```cpp
UPROPERTY( EditDefaultsOnly, BlueprintReadOnly, Category = "UI" )
TSubclassOf<class UUserWidget> GameOverUIClass;
```

在游戏结束的情况下，我们创建小部件并将其添加到视图中，这可以作为 `void ARPGGameMode::Tick(float DeltaTime)` 中的 `if(combatOver)` 条件嵌套条件添加，该文件位于 `RPGGameMode.cpp`：

```cpp
if( this->currentCombatInstance->phase == CombatPhase::CPHASE_GameOver )
{
  UE_LOG( LogTemp, Log, TEXT( "Player loses combat, game over" ) );

  Cast<URPGGameInstance>( GetGameInstance() )->PrepareReset();

  UUserWidget* GameOverUIInstance = CreateWidget<UUserWidget>( GetGameInstance(), this->GameOverUIClass );
  GameOverUIInstance->AddToViewport();
}
```

如你所见，我们还在游戏实例上调用了一个 `PrepareReset` 函数。这个函数尚未定义，所以我们现在在 `RPGGameInstance.h` 中创建它，作为一个公共函数：

```cpp
public:
  void PrepareReset();
```

然后在 `RPGGameInstance.cpp` 中实现它：

```cpp
cpp.void URPGGameInstance::PrepareReset()
{
  this->isInitialized = false;
  this->PartyMembers.Empty();
}
```

在这种情况下，`PrepareReset`的作用是将`isInitialized`设置为`false`，以便下次调用`Init`时，小组成员将被重新加载。我们还在清空`partyMembers`数组，这样当小组成员被重新添加到数组中时，我们不会将它们附加到我们上次游玩中的小组成员实例（我们不希望带着已死亡的小组成员重置游戏）。

到目前为止，你可以进行编译。但在我们能够测试之前，我们需要设置我们创建的**游戏结束 UIClass**，并将其设置为**GameOverScreen**作为**DefaultRPGGameMode**中的类默认值：

![创建游戏结束屏幕](img/B04548_03_28.jpg)

就像上次你做的那样，编辑器可能会崩溃，但当你回到**DefaultRPGGameMode**时，你应该会看到**GameOverScreen**被正确设置。

为了测试这一点，我们需要给哥布林比玩家更多的生命值。打开敌人表格，给哥布林分配超过 100 HP 的任何数值（例如，200 就足够了）。然后，开始一场遭遇战并玩到主要小组成员的生命值耗尽。此时，你应该会看到一个**游戏结束**屏幕弹出，点击**重新开始**，你将重新开始这一关卡，主要小组成员的生命值将恢复到 100 HP。

# 摘要

在本章中，我们为 RPG 的核心玩法打下了基础。我们有一个可以探索世界地图的角色，一个跟踪小组成员的系统，一个回合制战斗引擎，以及一个游戏结束条件。

在接下来的章节中，我们将通过添加库存系统来扩展这一功能，允许玩家消耗物品，并为他们的小组成员提供装备以提升他们的属性。
