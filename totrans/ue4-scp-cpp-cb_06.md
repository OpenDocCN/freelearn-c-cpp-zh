# 输入和碰撞

本章涵盖了围绕游戏控制输入（键盘、鼠标和游戏手柄）以及与障碍物碰撞的食谱。

本章将涵盖以下食谱：

+   轴映射 – FPS 角色的键盘、鼠标和游戏手柄方向输入

+   轴映射 – 标准化输入

+   动作映射 – FPS 角色的单按钮响应

+   从 C++ 添加轴和动作映射

+   鼠标 UI 输入处理

+   UMG 键盘 UI 快捷键

+   碰撞 – 使用 Ignore 允许对象相互穿过

+   碰撞 – 使用 Overlap 拾取对象

+   碰撞 – 使用 Block 防止相互穿透

# 简介

良好的输入控制对您的游戏至关重要。提供所有键盘、鼠标，尤其是游戏手柄输入将使您的游戏对用户更具吸引力。

![图片](img/79860469-0fa4-4790-bc3c-53efc1f3c422.png) 您可以在 Windows PC 上使用 Xbox 360 和 PlayStation 控制器 – 它们具有 USB 输入。请检查您当地的电子产品商店以找到一些好的 USB 游戏控制器。您还可以使用带有适当接收器的无线控制器连接到您的 PC。

# 技术要求

本章需要使用 Unreal Engine 4，并使用 Visual Studio 2017 作为 IDE。有关如何安装这两款软件及其要求的说明，请参阅本书第一章，*UE4 开发工具*。

# 轴映射 – FPS 角色的键盘、鼠标和游戏手柄方向输入

输入映射有两种类型：**轴映射**和**动作映射**。轴映射是您需要按住一段时间以获得其效果的输入（例如，按住 *W* 键使玩家向前移动），而动作映射是一次性输入（例如，在游戏手柄上按下 *A* 按钮或在键盘上按下空格键使玩家跳跃）。在本食谱中，我们将介绍如何设置键盘、鼠标和游戏手柄的轴映射输入控制，以移动 FPS 角色。

# 准备工作

您必须有一个 UE4 项目，其中包含一个主要角色玩家和一个可以行走的地面平面，为这个食谱做好准备。

# 如何操作...

1.  创建一个 C++ 类，并将 `Character` 作为父类。然后，点击 `Next`：

![图片](img/9341ddb3-a54d-4733-905b-6a49fbfd43bb.png)

1.  在名称属性下，输入 `Warrior`，然后点击创建类：

![图片](img/d443a1e9-8f02-4ee1-9572-4c22c0457ccb.png)

在 UE4 内部进行一些设置后，我们将进行实现。

1.  启动 UE4，右键单击“Warrior”类。然后，选择基于“Warrior”创建蓝图类：

![图片](img/80b8efda-da55-4d8c-a1d1-fa188e117e7f.png)

1.  从弹出的菜单中，将名称设置为 `BP_Warrior`，然后选择创建蓝图类：

![图片](img/a5161b2d-dae5-400d-aec0-9620df861b22.png)

1.  关闭刚刚打开的蓝图菜单。

1.  通过转到设置 | 项目设置 | 地图和模式来创建并选择一个新的 `GameMode` 类的蓝图：

![](img/a5c90fe1-9f45-4ca6-b7ff-b5047325f24f.png)

1.  点击默认 GameMode 下拉菜单旁边的 + 图标，这将创建一个 `GameMode` 类的新蓝图。为它起一个你喜欢的名字（比如 `BP_GameMode`）：

![](img/ef3f7b68-b9a3-4c57-8ef3-41a4b818ae9e.png)

1.  双击你创建的新 `BP_GameMode` 蓝图类进行编辑。它可以在内容浏览器的 `Contents\Blueprints` 文件夹中找到。

    1.  一旦你的 `BP_GameMode` 蓝图打开，选择你的蓝图类 `BP_Warrior` 作为默认的兵种类：

    ![](img/264fec32-0e72-4195-8ea1-b732a7b731d8.png)

    默认兵种类属性的位置。

    1.  要设置驱动玩家的键盘输入，请打开设置 | 项目设置 | 输入。（输入可以在引擎子部分下找到。）在以下步骤中，我们将完成驱动玩家在游戏中前进的过程。

    1.  点击轴映射标题旁边的 + 图标：

    ![](img/f23a8d03-3707-4ca2-9380-432bdcb8ba78.png)

    轴映射支持连续（按钮保持）输入，而动作映射支持一次性事件。

    1.  给轴映射起一个名字。这个第一个例子将展示如何移动玩家前进，所以可以命名为 `Forward`。

    1.  在“转发”下，选择一个键盘键来分配给这个轴映射，例如 *W*。

    1.  点击“前进”旁边的 + 图标，选择一个游戏控制器输入进行映射，以便你可以移动玩家前进（例如游戏手柄左侧摇杆向上）：

    ![](img/eb847eef-39cf-4d97-8023-028dc25d6d52.png)

    1.  使用键盘、游戏手柄，以及可选的鼠标输入绑定，完成向后、向左和向右的轴映射：

    ![](img/58185746-bbc4-45b6-8bc1-8f2241951473.png)

    1.  现在，返回到 `.h` 文件。我们需要添加一些新的函数定义，我们将要编写：

    ```cpp
    #pragma once

    #include "CoreMinimal.h"
    #include "GameFramework/Character.h"
    #include "Warrior.generated.h"

    UCLASS()
    class CHAPTER_06_API AWarrior : public ACharacter
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
        virtual void SetupPlayerInputComponent(class UInputComponent* 
                                            PlayerInputComponent) override;

        // Movement functions
     void Forward(float amount);
     void Back(float amount);
     void Right(float amount);
     void Left(float amount);

    };
    ```

    1.  从你的 C++ 代码中，为 `AWarrior` 类重写 `SetupPlayerInputComponent` 函数，如下所示：

    ```cpp
    #include "Components/InputComponent.h"

    // ...

    // Called to bind functionality to input
    void AWarrior::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
    {
        Super::SetupPlayerInputComponent(PlayerInputComponent);

        check(PlayerInputComponent);
        PlayerInputComponent->BindAxis("Forward", this, 
                                       &AWarrior::Forward);
        PlayerInputComponent->BindAxis("Back", this, &AWarrior::Back);
        PlayerInputComponent->BindAxis("Right", this, &AWarrior::Right);
        PlayerInputComponent->BindAxis("Left", this, &AWarrior::Left);
    }
    ```

    1.  在你的 `AWarrior` 类中提供一个 `Forward` 函数，如下所示：

    ```cpp
    void AWarrior::Forward(float amount)
    {
        // Moves the player forward by an amount in forward 
        // direction
        AddMovementInput(GetActorForwardVector(), amount);
    }
    ```

    1.  为其余的输入方向编写和完成函数，即 `AWarrior::Back`、`AWarrior::Left` 和 `AWarrior::Right`：

    ```cpp
    void AWarrior::Back(float amount)
    {
        AddMovementInput(-GetActorForwardVector(), amount);
    }

    void AWarrior::Right(float amount)
    {
        AddMovementInput(GetActorRightVector(), amount);
    }

    void AWarrior::Left(float amount)
    {
        AddMovementInput(-GetActorRightVector(), amount);
    }
    ```

    1.  返回 Unreal 并编译你的代码。之后，玩游戏并确认你现在可以使用键盘和游戏手柄的左侧摇杆移动：

    ![](img/9ffd2ff0-6c1f-440a-8b80-6ba07b10605b.png)

    # 它是如何工作的...

    UE4 引擎允许将输入事件直接连接到 C++ 函数调用。由输入事件调用的函数是某个类的成员函数。在上面的例子中，我们将 *W* 键的按下和游戏手柄左侧摇杆向上的保持连接到了 `AWarrior::Forward` C++ 函数。调用 `AWarrior::Forward` 的实例是路由控制器输入的实例。这由 `GameMode` 类中设置为玩家角色的对象控制。

    # 参见

    +   你实际上可以在 UE4 编辑器中输入输入轴绑定，而不是用 C++编写代码。我们将在后面的配方中详细描述，*从 C++添加轴和动作映射*。

    # 轴映射 – 归一化输入

    如你所注意到的，1.0 向右和 1.0 向前的输入实际上会相加到总共 2.0 个单位的速度。这意味着你可以以比纯前、后、左或右方向移动更快的速度进行对角移动。我们真正应该做的是，在保持输入指示的方向的同时，限制任何导致速度超过 1.0 单位的输入值。我们可以通过存储先前的输入值并覆盖`::Tick()`函数来实现这一点。

    # 准备工作

    要完成这个配方，你必须完成之前的配方，其中包含我们的`Warrior`类，因为我们将在其基础上添加内容。

    # 如何做到这一点...

    1.  前往你的`Warrior.h`文件，并添加以下属性：

    ```cpp
    protected:
        // Called when the game starts or when spawned
        virtual void BeginPlay() override;

     // The movement from the previous frame
     FVector2D lastInput;
    ```

    1.  然后，我们需要在类构造函数内部初始化变量：

    ```cpp
    // Sets default values
    AWarrior::AWarrior()
    {
        // Set this character to call Tick() every frame. You can turn  
        // this off to improve performance if you don't need it.
        PrimaryActorTick.bCanEverTick = true;

        lastInput = FVector2D::ZeroVector;
    }
    ```

    1.  更新`::Forward`、`::Back`、`::Right`和`::Left`函数，如下所示：

    ```cpp
    void AWarrior::Forward(float amount)
    {
        // We use a += of the amount added so that when the other  
        // function modifying .Y (::Back()) affects lastInput, it won't   
        // overwrite with 0's 
        lastInput.Y += amount;
    }

    void AWarrior::Back(float amount)
    {
        // In this case we are using -= since we are moving backwards
        lastInput.Y -= amount;
    }

    void AWarrior::Right(float amount)
    {
        lastInput.X += amount;
    }

    void AWarrior::Left(float amount)
    {
        lastInput.X -= amount;
    }
    ```

    1.  在`AWarrior::Tick()`函数中，修改输入值，在归一化任何输入向量的超尺寸之后：

    ```cpp
    // Called every frame
    void AWarrior::Tick(float DeltaSeconds)
    {
        Super::Tick(DeltaSeconds);

        float len = lastInput.Size();

     // If the player's input is greater than 1, normalize it
     if (len > 1.f)
     {
     lastInput /= len;
     }

     AddMovementInput(GetActorForwardVector(), lastInput.Y);
     AddMovementInput(GetActorRightVector(), lastInput.X);

     // Zero off last input values
     lastInput = FVector2D(0.f, 0.f);
    }
    ```

    # 它是如何工作的...

    当输入向量的幅度超过 1.0 时，我们对其进行归一化。这限制了最大输入速度为 1.0 单位（例如，当全按向上和向右时，为 2.0 单位）。

    # 动作映射 – FPS 角色的单键响应

    动作映射是用于处理单键按下的（不是按下的按钮）。对于应该按下的按钮，请确保使用轴映射。

    # 准备工作

    准备好一个包含你需要完成的动作的 UE4 项目，例如`Jump`或`ShootGun`。

    # 如何做到这一点...

    1.  打开设置 | 项目设置 | 输入。

    1.  前往动作映射标题，并点击其旁边的+图标：

    ![截图](img/a65fdfd6-0f69-4512-8ba5-82d282b9e325.png)

    1.  开始输入应该映射到按钮按下的动作。例如，为第一个动作输入`Jump`。

    1.  点击动作左侧的箭头以打开菜单，然后选择一个按键来触发该动作，例如空格键。

    1.  如果你希望另一个按键触发相同动作，请点击动作映射名称旁边的+，并选择另一个按键来触发动作。

    1.  如果你希望 Shift、Ctrl、Alt 或 Cmd 键被按下以触发动作，请确保在键选择框右侧的复选框中表明这一点：

    ![截图](img/0a79a026-ca19-4dc7-9c1c-8b274c320eff.png)

    1.  要将你的动作链接到 C++代码函数，你需要覆盖`SetupPlayerInputComponent(UInputControl* control )`函数。在该函数内部输入以下代码：

    ```cpp
    // Called to bind functionality to input
    void AWarrior::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
    {
        Super::SetupPlayerInputComponent(PlayerInputComponent);

        check(PlayerInputComponent);
        PlayerInputComponent->BindAxis("Forward", this, 
                                                       &AWarrior::Forward);
        PlayerInputComponent->BindAxis("Back", this, &AWarrior::Back);
        PlayerInputComponent->BindAxis("Right", this, &AWarrior::Right);
        PlayerInputComponent->BindAxis("Left", this, &AWarrior::Left);

        PlayerInputComponent->BindAction("Jump", IE_Pressed, this, 
                                                          &AWarrior::Jump);
    }
    ```

    1.  编译你的脚本并玩游戏。每次你按下空格键时，你应该看到玩家跳到空中！请参考以下截图：

    ![截图](img/2d487813-cd64-4250-a1cc-8bb083d862f8.png)

    # 它是如何工作的...

    **动作映射**是单按钮点击事件，会触发 C++代码执行以响应它们。您可以在 UE4 编辑器中定义任何数量的动作，但请确保将动作映射与 C++中的实际按键点击关联起来。

    您可能会注意到，当我们使用动作时调用的`Jump`函数，当我们添加对其的引用时已经存在。这是因为`Character`类已经包含了对它的实现。请注意，默认实现并不像常规跳跃那样感觉——它更像是一种上升和漂浮的动作。

    您可以在[`api.unrealengine.com/INT/API/Runtime/Engine/GameFramework/ACharacter/index.html`](https://api.unrealengine.com/INT/API/Runtime/Engine/GameFramework/ACharacter/index.html)上找到有关`Character`类及其预构建函数的更多信息。

    # 参见

    +   您可以从 C++代码中列出您想要映射的动作。请参阅以下配方，*从 C++添加轴和动作映射*。

    # 从 C++添加轴和动作映射

    轴映射和动作映射可以通过 UE4 编辑器添加到您的游戏中，这也是设计师通常会采用的方法，但我们也可以直接从 C++代码中添加它们。由于连接到 C++函数的连接本身就是从 C++代码中进行的，您可能会发现将轴和动作映射定义在 C++中更方便。

    # 准备工作

    您需要一个 UE4 项目，您希望向其中添加一些轴和动作映射。如果您通过 C++代码添加，可以删除在设置 | 项目设置 | 输入中列出的现有轴和动作映射。

    要添加您自定义的轴和动作映射，您需要了解两个 C++函数：`UPlayerInput::AddAxisMapping` 和 `UPlayerInput:: AddActionMapping`。这些是`UPlayerInput`对象上的成员函数。`UPlayerInput`对象位于`PlayerController`对象内部，可以通过以下代码访问：

    ```cpp
    GetWorld()->GetFirstPlayerController()->PlayerInput 
    ```

    如果您不想单独访问玩家控制器，您也可以使用`UPlayerInput`的两个静态成员函数来创建您的轴和动作映射：

    ```cpp
    UPlayerInput::AddEngineDefinedAxisMapping() 
    UPlayerInput::AddEngineDefinedActionMapping() 
    ```

    # 如何操作...

    1.  首先，我们需要定义我们的`FInputAxisKeyMapping`或`FInputActionKeyMapping`对象，具体取决于您是连接轴键映射（用于按下的输入按钮）还是动作键映射（用于一次性事件按钮，按下一次用于输入）。

        1.  要使用以下任一类，我们需要包含以下`.h`文件：

    ```cpp
    #include "GameFramework/PlayerInput.h" 
    ```

    1.  1.  对于轴键映射，我们定义一个`FInputAxisKeyMapping`对象，如下所示：

    ```cpp
    FInputAxisKeyMapping backKey( "Back", EKeys::S, 1.f ); 
    ```

    1.  1.  这将包括动作的字符串名称、要按下的键（使用 EKeys `枚举`），以及是否需要按住*Shift*、*Ctrl*、*Alt*或*cmd*（Mac）来触发事件。

        1.  对于动作键映射，定义`FInputActionKeyMapping`，如下所示：

    ```cpp
    FInputActionKeyMapping jump("Jump", EKeys::SpaceBar, 0, 0, 0, 0); 
    ```

    1.  1.  这将包括动作的字符串名称、要按下的键，以及是否需要按住*Shift*、*Ctrl*、*Alt*或*cmd*（Mac）来触发事件。

    1.  在你的玩家`Pawn`类的`SetupPlayerInputComponent`函数中，将你的轴和动作键映射注册到以下内容：

    1.  1.  连接到特定控制器的`PlayerInput`对象：

    ```cpp
    GetWorld()->GetFirstPlayerController()->PlayerInput
     ->AddAxisMapping( backKey ); // specific to a controller
    ```

    1.  1.  或者，你可以直接注册到`UPlayerInput`对象的静态成员函数：

    ```cpp
    UPlayerInput::AddEngineDefinedActionMapping(jump );
    ```

    确保你使用的是轴映射和动作映射的正确函数！

    1.  使用 C++代码，就像我们在前面的两个菜谱中所做的那样，将你的动作和轴映射注册到 C++函数中，例如：

    ```cpp
    PlayerInputComponent->BindAxis("Back", this, &AWarrior::Back); 
    PlayerInputComponent->BindAction("Jump", IE_Pressed, this, &AWarrior::Jump 
               );
    ```

    # 它是如何工作的...

    动作和轴映射注册函数允许你直接从 C++代码设置输入映射。C++编码的输入映射基本上与在设置 | 项目设置 | 输入对话框中输入输入映射相同。

    # 鼠标 UI 输入处理

    当使用**Unreal 运动图形**（**UMG**）工具包时，你会发现鼠标事件处理非常简单。我们可以将 C++函数注册为在鼠标点击或其他类型的 UMG 组件交互后运行。

    通常，事件注册将通过蓝图进行；然而，在这个菜谱中，我们将概述如何编写和连接 C++函数到 UMG 事件。

    # 准备中

    在你的 UE4 项目中创建一个 UMG 画布。从那里，我们将为`OnClicked`、`OnPressed`和`OnReleased`事件注册事件处理器。

    # 如何做到这一点...

    1.  在你的内容浏览器中右键单击（或点击添加新内容），然后选择用户界面 | 小部件蓝图，如图所示。这将向你的项目添加一个可编辑的小部件蓝图：

    ![图片](img/451daeed-ffb0-4715-9eea-28085c8b1096.png)

    1.  双击你的小部件蓝图进行编辑。

    1.  通过从左侧的调色板拖动来向界面添加一个按钮：

    ![图片](img/4bd1c07c-a8bd-493a-95f6-db057cb0b315.png)

    1.  将按钮的详细信息面板向下滚动，直到找到事件子部分。

    1.  点击你想要处理的事件旁边的+图标：

    ![图片](img/b0d70320-f08a-4d4d-80e5-b026be6eeb6c.png)

    1.  将在蓝图中出现的事件连接到任何具有`BlueprintCallable`标记的 C++ `UFUNCTION()`。例如，在你的`GameModeBase`类派生中，你可以包含如下函数：

    ```cpp
    UCLASS()
    class CHAPTER_06_API AChapter_06GameModeBase : public AGameModeBase
    {
      GENERATED_BODY()

    public:
     UFUNCTION(BlueprintCallable, Category = UIFuncs)
     void ButtonClicked()
     {
     UE_LOG(LogTemp, Warning, TEXT("UI Button Clicked"));
     }
    };
    ```

    1.  通过在蓝图图中的选择事件下路由到它来触发函数调用。例如，我使用了`OnClick`函数。一旦创建，我使用了获取游戏模式节点来获取我们当前的游戏模式，然后`Cast to Chapter06_GameModeBase`来访问`ButtonClicked`函数：

    ![图片](img/8f4f3430-4d15-4359-ad22-e5abb7091743.png)

    注意，为了使这生效，确保你已将级别/项目的游戏模式设置为`Chapter_06GameModeBase`。

    1.  通过在`GameModeBase`（或任何此类主对象）的`BeginPlay`函数中调用`Create Widget`，然后添加到视图中，构建并显示你的 UI。或者，通过级别蓝图进行：

    1.  要通过 C++这样做，你需要转到`Chapter_06.Build.cs`文件并修改以下行：

    ```cpp
    PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "UMG", "Slate", "SlateCore" });
    ```

    1.  之后，将以下属性和函数添加到 `Chapter_06GameModeBase.h` 文件中：

    ```cpp
    public:
        UFUNCTION(BlueprintCallable, Category = UIFuncs)
        void ButtonClicked()
        {
            UE_LOG(LogTemp, Warning, TEXT("UI Button Clicked"));
        }

     void BeginPlay();

     UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "UI")
     TSubclassOf<class UUserWidget> Widget;
    ```

    1.  然后，更新 C++ 文件：

    ```cpp
    #include "Chapter_06GameModeBase.h"
    #include "Blueprint/UserWidget.h"

    void AChapter_06GameModeBase::BeginPlay()
    {
        Super::BeginPlay();

        if(Widget)
        {
            UUserWidget* Menu = CreateWidget<UUserWidget>(GetWorld(), 
            Widget);

            if(Menu)
            {
                Menu->AddToViewport();
           GetWorld()->GetFirstPlayerController()->bShowMouseCursor = 
           true;
            }

        }

    }
    ```

    1.  现在，我们需要将小部件设置到我们创建的菜单中。为此，从内容浏览器中，在 C++ `Classes\Chapter_06` 文件夹中的 `Chapter06_GameModeBase` 上右键单击，并从中创建一个新的蓝图。一旦进入蓝图菜单，转到详细信息选项卡，在 UI 部分，将小部件设置为你要显示的项目：

    ![图片](img/a99bff0f-a8cb-43ef-b2a6-c9cfb004d3ef.png)

    1.  最后，转到设置 | 世界设置。从那里，将游戏模式覆盖更改为你的游戏模式蓝图版本：

    ![图片](img/1a52c677-c0a4-4edd-95a3-ca9712391c4a.png)

    1.  然后，通过转到窗口 | 开发者工具 | 输出日志来打开输出日志，并开始游戏。你应该在屏幕上看到按钮。如果你点击它，你应该在输出日志中看到一个消息显示！

    ![图片](img/60719194-7e50-46ce-961f-af9231f654a5.png)

    # 它是如何工作的...

    你的小部件蓝图按钮事件可以轻松连接到蓝图事件或 C++ 函数，通过创建一个带有 `BlueprintCallable` 标签的 `UFUNCTION()` 来实现：

    更多关于使用 UMG 和构建简单菜单以及使用蓝图显示它们的信息，请查看[`docs.unrealengine.com/en-us/Engine/UMG/HowTo/CreatingWidgets`](https://docs.unrealengine.com/en-us/Engine/UMG/HowTo/CreatingWidgets)。

    # UMG 键盘 UI 快捷键

    每个用户界面都需要与之关联的快捷键。要将这些编程到你的 UMG 界面中，你可以简单地连接某些键组合到一个动作映射中。当动作触发时，只需调用 UI 按钮本身触发的相同蓝图函数。

    # 准备工作

    你应该已经创建了一个 UMG 界面，如前一个菜谱所示。

    # 如何做到这一点...

    1.  在设置 | 项目设置 | 输入中，为你的热键事件定义一个新的动作映射，例如，`HotKey_UIButton_Spell`：

    ![图片](img/70e3025e-174f-4ae7-8be2-139b9ae12850.png)

    1.  在蓝图或 C++ 代码中将事件连接到你的 UI 函数调用。在我们的例子中，我将将其添加到我们之前创建的 `AWarrior` 类中，通过将其添加到 `SetupPlayerInputComponent` 函数中：

    ```cpp
    #include "Chapter_06GameModeBase.h"

    // ...

    // Called to bind functionality to input
    void AWarrior::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
    {
        Super::SetupPlayerInputComponent(PlayerInputComponent);

        check(PlayerInputComponent);
        PlayerInputComponent->BindAxis("Forward", this, 
        &AWarrior::Forward);
        PlayerInputComponent->BindAxis("Back", this, &AWarrior::Back);
        PlayerInputComponent->BindAxis("Right", this, &AWarrior::Right);
        PlayerInputComponent->BindAxis("Left", this, &AWarrior::Left);

        PlayerInputComponent->BindAction("Jump", IE_Pressed, this, 
        &AWarrior::Jump);

        // Example of adding bindings via code instead of the 
        //  editor
        FInputAxisKeyMapping backKey("Back", EKeys::S, 1.f);
        FInputActionKeyMapping jump("Jump", EKeys::SpaceBar, 0, 0, 
        0, 0);

        GetWorld()->GetFirstPlayerController()->PlayerInput-
        >AddAxisMapping(backKey);
        GetWorld()->GetFirstPlayerController()->PlayerInput-
        >AddActionMapping(jump);

     // Calling function for HotKey
     auto GameMode = Cast<AChapter_06GameModeBase>(GetWorld()-
        >GetAuthGameMode());
     auto Func = &AChapter_06GameModeBase::ButtonClicked;

     if(GameMode && Func)
     {
     PlayerInputComponent->BindAction("HotKey_UIButton_Spell", 
                                             IE_Pressed, GameMode,  
                                             Func);
     }

    }
    ```

    1.  编译你的脚本，然后通过转到设置 | 世界设置来打开世界设置。在所选游戏模式下，将默认兵种类设置为 `BP_Warrior`。你现在应该注意到，你可以按你的键或按按钮来执行我们在上一个菜谱中创建的 `ButtonClicked` 函数！

    # 它是如何工作的...

    将动作映射连接到由 UI 调用的函数的短路，这将允许你在游戏程序中优雅地实现热键。

    # 碰撞 - 使用忽略让对象相互穿过

    碰撞设置相对容易入门。碰撞有三个交集类别：

    +   `忽略`：没有任何通知就相互穿过的碰撞。

    +   `Overlap`：触发`OnBeginOverlap`和`OnEndOverlap`事件的碰撞。具有`Overlap`设置的物体之间的相互穿透是允许的。

    +   `Block`：防止所有相互穿透的碰撞，并防止物体相互重叠。

    物体被分类为许多对象类型之一。特定蓝图组件的碰撞设置允许您将对象分类为所选的对象类型，并指定该对象如何与其他所有类型的所有对象碰撞。这在蓝图编辑器的详细信息 | 碰撞部分以表格格式表示。

    例如，以下截图显示了角色的`CapsuleComponent`的碰撞设置：

    ![](img/e187ec4e-1e2f-4603-8cf9-b4eddcf6ae3c.png)

    # 准备工作

    您应该有一个 UE4 项目，其中包含一些您想要编程交叉的对象。

    # 如何操作...

    1.  打开您希望其他对象简单地穿过并忽略的对象的蓝图编辑器。在组件列表下，选择您想要编程设置的组件。

    1.  在您选择了组件后，查看您的详细信息标签页（通常位于右侧）。在碰撞预设下，选择“无碰撞”或“自定义...”预设：

    +   如果您选择“无碰撞”预设，您可以直接保留它，并且所有碰撞都将被忽略。

    +   如果您选择“自定义...”预设，则可以选择以下任一项：

        +   在“碰撞启用”下拉菜单中选择“无碰撞”。

        +   在“碰撞启用”下选择一个涉及查询的碰撞模式，并确保勾选忽略选项。

    +   为您想要忽略碰撞的每个对象类型勾选忽略复选框。

    # 它是如何工作的...

    被忽略的碰撞不会触发任何事件或阻止标记为这样的对象之间的相互穿透。请注意，如果对象 A 被设置为忽略对象 B，则无论对象 B 是否被设置为忽略对象 A，都没有关系。只要其中之一被设置为忽略另一个，它们就会相互忽略。

    # 碰撞 - 使用 Overlap 拾取对象

    有效地进行物品拾取是一项相当重要的事情。在本配方中，我们将概述如何使用 Actor Component 原语的 Overlap 事件来实现物品拾取。

    # 准备工作

    之前的配方，“碰撞 - 使用忽略让对象相互穿过”描述了碰撞的基础。在开始此配方之前，您应该阅读它作为背景知识。我们在这里要创建一个新的对象通道...来识别“物品”类对象，以便它们可以被编程为重叠，但仅与玩家角色的碰撞体积重叠。

    # 如何操作...

    1.  首先，为“物品”对象的碰撞原语创建一个唯一的碰撞通道。这位于设置 | 项目设置 | 碰撞：

    ![](img/1c91758c-6e3b-454b-932d-0fa3f8e59e4a.png)

    1.  到达那里后，通过转到“新建对象通道...”创建一个新的对象通道。

    1.  将新的对象通道命名为“物品”并将默认响应设置为 Overlap。之后，点击接受按钮：

    ![图片](img/aa9ee232-de0e-4b2b-8900-1910ce95c00d.png)

    1.  选择你的`Item`角色，并选择用于与玩家头像拾取相交的原始组件。从详细信息选项卡转到碰撞部分，在碰撞预设下，将选项更改为自定义...之后，将那个原始组件的对象类型设置为`Item`。

    1.  如以下截图所示，勾选`Pawn`类对象类型的重叠复选框：

    ![图片](img/37f20568-fa11-4711-baca-e6b1b0f521ed.png)

    1.  确保已勾选生成重叠事件复选框：

    ![图片](img/21f4abc7-1789-47a8-98ef-99654ed333d0.png)

    生成重叠事件属性的定位。

    1.  选择将拾取物品的玩家角色（例如，本例中的 BP_Warrior）并选择其上用于检测物品的组件。通常，这将是其`CapsuleComponent`。检查与`Item`对象的重叠：

    ![图片](img/c1a57159-25c1-4944-9692-917dd14f30ea.png)

    1.  现在，玩家与物品重叠，物品与玩家角色重叠。我们必须在两个方向上发出重叠信号（`Item`重叠`Pawn`和`Pawn`重叠`Item`），以便它能够正常工作。确保为`Pawn`相交组件也勾选了生成重叠事件。

    1.  接下来，我们必须使用蓝图或 C++代码完成物品或玩家拾取体积的`OnComponentBeginOverlap`事件：

        1.  如果你更喜欢蓝图，在物品的可相交组件的详细信息窗格的事件部分，点击`On Component Begin Overlap`事件旁边的+图标：

    ![图片](img/6dee9413-e593-4c0a-922e-71eab50057a9.png)

    1.  1.  使用出现在你的`Actor`蓝图图中的`OnComponentBeginOverlap`事件，将蓝图代码连接到当与玩家的胶囊体积发生重叠时运行。

        1.  如果你更喜欢 C++，你可以为`CapsuleComponent`编写并附加一个 C++函数。在你的玩家`Character`类（例如，`Warrior.h`文件）中编写一个成员函数，其签名如下：

    ```cpp

    UFUNCTION(BlueprintNativeEvent, Category = Collision)
    void OnOverlapsBegin(UPrimitiveComponent* Comp,
                            AActor* OtherActor, 
                            UPrimitiveComponent* OtherComp,
                            int32 OtherBodyIndex,
                            bool bFromSweep, 
                            const FHitResult&SweepResult);

    UFUNCTION(BlueprintNativeEvent, Category = Collision)
    void OnOverlapsEnd(UPrimitiveComponent* Comp,
                        AActor* OtherActor, 
                        UPrimitiveComponent* OtherComp,
                        int32 OtherBodyIndex);

    virtual void PostInitializeComponents() override;
    ```

    1.  1.  在你的`.cpp`文件中完成`OnOverlapsBegin()`函数的实现，确保函数名以`_Implementation`结尾：

    1.  ```cpp
        void AWarrior::OnOverlapsBegin_Implementation(
            UPrimitiveComponent* Comp,
            AActor* OtherActor, UPrimitiveComponent* OtherComp,
            int32 OtherBodyIndex,
            bool bFromSweep, const FHitResult&SweepResult)
        {
            UE_LOG(LogTemp, Warning, TEXT("Overlaps warrior
            began"));
        }

        void AWarrior::OnOverlapsEnd_Implementation(
            UPrimitiveComponent* Comp,
            AActor* OtherActor, UPrimitiveComponent* OtherComp,
            int32 OtherBodyIndex)
        {
            UE_LOG(LogTemp, Warning, TEXT("Overlaps warrior
            ended"));
        }
        ```

        1.  1.  然后，为你的头像类提供一个`PostInitializeComponents()`重写，将`OnOverlapsBegin()`函数与重叠连接到胶囊，如下所示：

        ```cpp
        #include "Components/CapsuleComponent.h"

        // ...

        void AWarrior::PostInitializeComponents()
        {
            Super::PostInitializeComponents();

            if (RootComponent)
            {
                // Attach contact function to all bounding components. 
                GetCapsuleComponent()->OnComponentBeginOverlap.AddDynamic(this, &AWarrior::OnOverlapsBegin);
                GetCapsuleComponent()->OnComponentEndOverlap.AddDynamic(this, &AWarrior::OnOverlapsEnd);
            }
        }
        ```

        1.  编译你的脚本，然后运行你的项目。当你进入和离开对象时，你应该会看到日志消息！请参考以下截图：

        ![图片](img/98a6d7a0-dcf9-4da6-bc7b-50f130efa029.png)

        # 它是如何工作的...

        引擎引发的 Overlap 事件允许代码在两个 UE4 `Actor`组件重叠时运行，而不阻止对象的穿透。

        # 碰撞 - 使用阻塞防止穿透

        阻塞意味着`Actor`组件在引擎中将被阻止穿透，并且任何两个原始形状之间的碰撞将在发现碰撞后解决，而不是重叠。

        # 准备工作

        从一个具有一些对象且这些对象有附加碰撞原始形状的演员的 UE4 项目开始（`SphereComponents`、`CapsuleComponents` 或 `BoxComponents`）。

        # 如何做到这一点...

        1.  打开你想要阻止另一个演员的演员的蓝图。例如，我们想要玩家演员阻止其他玩家演员实例。

        1.  将你不想与其他组件发生穿透的原始形状标记在演员中，这样就可以在“详细信息”面板中阻止这些组件：

        ![图片](img/15215060-daac-41c6-a9db-c7b95e1be796.png)

        # 它是如何工作的...

        当对象 b

        如果两个对象相互锁定，它们将不允许相互穿透。任何穿透都将自动解决，并且对象将被推开。这是经常引起许多头疼的问题之一。为了实际阻止彼此的对象，它们都必须设置为阻止。

        更多信息，请查看官方 UE4 博客文章：[`www.unrealengine.com/en-US/blog/collision-filtering`](https://www.unrealengine.com/en-US/blog/collision-filtering)。

        # 还有更多...

        你可以覆盖 `OnComponentHit` 函数，以便在两个对象相撞时运行代码。这与 `OnComponentBeginOverlap` 事件不同。
