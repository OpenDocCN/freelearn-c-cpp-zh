# 第二章：Coin Dash

这个第一个项目将引导你完成你的第一个 Godot 引擎项目。你将学习 Godot 编辑器的工作方式，如何构建项目结构，以及如何制作一个小型 2D 游戏。

为什么是 2D？简而言之，3D 游戏比 2D 游戏复杂得多，而你需要了解的许多底层游戏引擎功能是相同的。你应该坚持 2D，直到你对 Godot 的游戏开发过程有很好的理解。到那时，转向 3D 将容易得多。这本书的第五个和最后一个项目将介绍 3D。

重要——即使你不是游戏开发的完全新手，也不要跳过这一章。虽然你可能已经理解了许多底层概念，但这个项目将介绍许多基本的 Godot 功能和设计范式，这些是你今后需要了解的。随着你开发更复杂的项目，你将在此基础上构建这些概念。

本章中的游戏被称为**Coin Dash**。你的角色必须在屏幕上移动，尽可能多地收集硬币，同时与时间赛跑。完成游戏后，游戏将看起来像这样：

![](img/00015.jpeg)

# 项目设置

启动 Godot 并创建一个新项目，确保使用`创建文件夹`按钮来确保此项目的文件将与其他项目分开保存。你可以在此处下载游戏的艺术和声音（统称为*资产*）的 Zip 文件，[`github.com/PacktPublishing/Godot-Game-Engine-Projects/releases`](https://github.com/PacktPublishing/Godot-Game-Engine-Projects/releases)。

将此文件解压到你的新项目文件夹中。

在这个项目中，你将制作三个独立的场景：`Player`、`Coin`和`HUD`，它们都将组合到游戏的`Main`场景中。在一个更大的项目中，创建单独的文件夹来保存每个场景的资源和脚本可能很有用，但在这个相对较小的游戏中，你可以在根文件夹中保存你的场景和脚本，该文件夹被称为`res://`（**res**是**资源**的缩写）。你的项目中的所有资源都将位于`res://`文件夹的相对位置。你可以在左上角的文件系统窗口中查看你的项目文件夹：

![](img/00016.jpeg)

例如，硬币的图片将位于`res://assets/coin/`。

这个游戏将使用竖屏模式，因此你需要调整游戏窗口的大小。点击项目菜单并选择项目设置，如下面的截图所示：

![](img/00017.jpeg)

查找显示/窗口部分，并将宽度设置为`480`，高度设置为`720`。在此部分中，还将拉伸/模式设置为`2D`，并将纵横比设置为`保持`。这将确保如果用户调整游戏窗口的大小，所有内容都将适当地缩放，而不会拉伸或变形。如果你喜欢，你也可以取消选中可调整大小的复选框，以防止窗口完全调整大小。

# 向量和 2D 坐标系

注意：本节是 2D 坐标系的一个非常简短的概述，并没有深入探讨向量数学。它旨在为 Godot 游戏开发提供一个高级概述。向量数学是游戏开发中的基本工具，因此如果你需要对此主题有更广泛的理解，请参阅可汗学院的线性代数系列([`www.khanacademy.org/math/linear-algebra`](https://www.khanacademy.org/math/linear-algebra))。

在 2D 工作中，你将使用笛卡尔坐标系来识别空间中的位置。2D 空间中的特定位置可以表示为一对值，例如`(4,3)`，分别代表沿*x*轴和*y*轴的位置。2D 平面上的任何位置都可以用这种方式描述。

在 2D 空间中，Godot 遵循常见的计算机图形学惯例，将*x*轴向右，*y*轴向下：

![图片 2](img/00018.jpeg)

如果你刚开始接触计算机图形学或游戏开发，可能会觉得正 y 轴向下而不是向上有点奇怪，正如你可能在数学课上所学的。然而，这种方向在计算机图形学应用中非常常见。

# 向量

你也可以将位置`(4, 3)`视为从`(0, 0)`点或**原点**的偏移。想象一支箭从原点指向该点：

![图片 1](img/00019.jpeg)

这支箭是一个**向量**。它代表了许多有用的信息，包括点的位置，*(4, 3)*，其长度，*m*，以及其与*x*-轴的夹角，*θ*。总的来说，这是一个**位置向量**，换句话说，它描述了空间中的位置。向量也可以表示运动、加速度或任何具有*x*和*y*分量的其他量。

在 Godot 中，向量（2D 中的`Vector2`或 3D 中的`Vector3`）被广泛使用，你将在本书构建的项目过程中使用它们。

# 像素渲染

Godot 中的向量坐标是**浮点数**，而不是**整数**。这意味着`Vector2`可以有一个分数值，例如`(1.5, 1.5)`。由于对象不能在半像素处绘制，这可能会为像素艺术游戏带来视觉问题，其中你希望确保所有纹理的像素都被绘制。

为了解决这个问题，打开**项目** *|* **项目设置**，在侧边栏中找到**渲染**/***质量**部分并启用使用像素捕捉，如图下截图所示：

![图片 3](img/00020.jpeg)

如果你在游戏中使用 2D 像素艺术，那么在开始项目时始终启用此设置是个好主意。在 3D 游戏中，此设置没有任何效果。

# 第一部分 – 玩家场景

你将制作的第一个场景定义了玩家对象。创建单独的玩家场景的一个好处是，你可以在创建游戏的其它部分之前独立测试它。随着你的项目规模和复杂性的增长，这种游戏对象的分离将变得越来越有帮助。将单个游戏对象与其他对象保持分离，使它们更容易调试、修改，甚至完全替换而不影响游戏的其它部分。这也使你的玩家可重用——你可以将玩家场景放入一个完全不同的游戏中，它将正常工作。

玩家场景将显示你的角色及其动画，通过响应用户输入相应地移动角色，并检测与游戏中的其他对象的碰撞。

# 创建场景

首先，点击添加/创建新节点按钮并选择一个 `Area2D`。然后，点击其名称并将其更改为 `Player`。点击场景 | 保存场景以保存场景。这是场景的 *根* 或顶级节点。你将通过向此节点添加子节点来为 `Player` 添加更多功能：

![](img/00021.gif)

在添加任何子节点之前，确保你不小心通过点击它们来移动或调整它们的大小。选择 `Player` 节点并点击旁边的锁图标：

![](img/00022.jpeg)

工具提示将显示确保对象的孩子不可选择，如前面的截图所示。

在创建新场景时始终这样做是个好主意。如果一个身体的碰撞形状或精灵偏移或缩放，可能会导致意外的错误并且难以修复。使用此选项，节点及其所有子节点将始终一起移动。

# 精灵动画

使用 `Area2D`，你可以检测其他对象何时与玩家重叠或碰撞，但 `Area2D` 本身没有外观，因此点击 `Player` 节点并添加一个作为子节点的 `AnimatedSprite` 节点。`AnimatedSprite` 将处理玩家的外观和动画。注意，节点旁边有一个警告符号。`AnimatedSprite` 需要一个 `SpriteFrames` 资源，其中包含它可以显示的动画。要创建一个，在检查器中找到 Frame*s* 属性并点击 <null> | 新建 SpriteFrames:

![](img/00023.jpeg)

接下来，在相同的位置，点击 <SpriteFrames> 打开 SpriteFrames 面板：

![](img/00024.gif)

在左侧是一个动画列表。点击默认的动画并将其重命名为 `run`。然后，点击 **添加** 按钮创建第二个名为 `idle` 的动画和第三个名为 `hurt` 的动画。

在左侧的文件系统工具栏中，找到 `run`、`idle` 和 `hurt` 玩家图像并将它们拖入相应的动画中：

![](img/00025.jpeg)

每个动画都有一个默认的每秒 5 帧的速度设置。这有点慢，所以点击每个动画并将速度（FPS）设置更改为 8。在检查器中，勾选 Playing 属性旁边的复选框并选择一个动画来查看动画效果：

![](img/00026.jpeg)

之后，你将编写代码来根据玩家的动作选择这些动画。但首先，你需要完成设置玩家的节点。

# 碰撞形状

当使用`Area2D`或 Godot 中的其他碰撞对象时，它需要定义一个形状，否则无法检测碰撞。碰撞形状定义了对象占据的区域，并用于检测重叠和/或碰撞。形状由`Shape2D`定义，包括矩形、圆形、多边形和其他类型的形状。

为了方便起见，当你需要向区域或物理体添加形状时，你可以添加一个`CollisionShape2D`作为子节点。然后选择你想要的形状类型，你可以在编辑器中编辑其大小。

将一个`CollisionShape2D`作为`Player`的子节点（确保不要将其作为`AnimatedSprite`的子节点）。这将允许你确定玩家的*击打框*，或其碰撞区域的边界。在检查器中，在形状旁边点击<null>并选择 New RectangleShape2D。调整形状的大小以覆盖精灵：

![](img/00027.jpeg)

请注意不要缩放形状的轮廓！仅使用大小手柄（红色）来调整形状！缩放后的碰撞形状将无法正确工作。

你可能已经注意到碰撞形状没有在精灵上居中。这是因为精灵本身在垂直方向上没有居中。我们可以通过向`AnimatedSprite`添加一个小偏移量来修复这个问题。点击节点并在检查器中查找 Offset 属性。将其设置为`(0, -5)`。

当你完成时，你的`Player`场景应该看起来像这样：

![](img/00028.jpeg)

# 编写玩家脚本

现在，你准备好添加脚本了。脚本允许你添加内置节点所不具备的额外功能。点击`Player`节点并点击**添加脚本**按钮：

![](img/00029.jpeg)

在脚本设置窗口中，你可以保留默认设置不变。如果你已经记得保存场景（参见前面的截图），脚本将自动命名为与场景名称匹配。点击创建，你将被带到脚本窗口。你的脚本将包含一些默认注释和提示。你可以删除注释（以`#`开头的行）。参考以下代码片段：

```cpp
extends Area2D

# class member variables go here, for example:
# var a = 2
# var b = "textvar"

func _ready():
 # Called every time the node is added to the scene.
 # Initialization here
 pass

#func _process(delta):
# # Called every frame. Delta is time since last frame.
# # Update game logic here.
# pass
```

每个脚本的第一个行将描述它附加到的节点类型。接下来，你将定义你的类变量：

```cpp
extends Area2D

export (int) var speed
var velocity = Vector2()
var screensize = Vector2(480, 720)
```

在`speed`变量上使用`export`关键字允许你在检查器中设置其值，同时让检查器知道变量应包含的数据类型。这对于你想要能够调整的值非常有用，就像调整节点的内置属性一样。点击**`Player`**节点并将 Speed 属性设置为 350，如图所示：

![](img/00030.jpeg)

`velocity`将包含角色的当前移动速度和方向，而`screensize`将用于设置玩家的移动限制。稍后，游戏的主场景将设置此变量，但现在你将手动设置它以便测试。

# 移动玩家

接下来，你将使用`_process()`函数来定义玩家将做什么。`_process()`函数在每一帧都会被调用，所以你会用它来更新你预期经常更改的游戏元素。你需要玩家做三件事：

+   检查键盘输入

+   按照给定的方向移动

+   播放适当的动画

首先，你需要检查输入。对于这个游戏，你有四个方向输入需要检查（四个箭头键）。输入操作在项目设置下的输入映射标签中定义。在这个标签中，你可以定义自定义事件并将不同的键、鼠标操作或其他输入分配给它们。默认情况下，Godot 已经将事件分配给了键盘箭头，所以你可以使用它们在这个项目中。

你可以使用`Input.is_action_pressed()`检测是否按下了输入，如果按键被按下则返回`true`，如果没有则返回`false`。结合所有四个按钮的状态将给出运动的结果方向。例如，如果你同时按下`right`和`down`，则结果速度向量将是`(1, 1)`。在这种情况下，因为我们正在将水平和垂直运动结合起来，所以玩家会比仅水平移动时移动得更快。

你可以通过*归一化*速度来防止这种情况，这意味着将其长度设置为**1**，然后乘以期望的速度：

```cpp
func get_input():
    velocity = Vector2()
    if Input.is_action_pressed("ui_left"):
        velocity.x -= 1
    if Input.is_action_pressed("ui_right"):
        velocity.x += 1
    if Input.is_action_pressed("ui_up"):
        velocity.y -= 1
    if Input.is_action_pressed("ui_down"):
        velocity.y += 1
    if velocity.length() > 0:
        velocity = velocity.normalized() * speed
```

通过将所有这些代码组合在一个`get_input()`函数中，你可以使后续更改变得更加容易。例如，你可以决定改为使用模拟摇杆或其他类型的控制器。从`_process()`函数中调用此函数，然后通过结果`velocity`更改玩家的`position`。为了防止玩家离开屏幕，你可以使用`clamp()`函数将位置限制在最小和最大值之间：

```cpp
func _process(delta):
    get_input()

    position += velocity * delta
    position.x = clamp(position.x, 0, screensize.x)
    position.y = clamp(position.y, 0, screensize.y)
```

点击“播放编辑的场景”（*F6*）并确认你可以按所有方向移动玩家。

# 关于 delta

`_process()`函数包含一个名为`delta`的参数，然后将其乘以速度。`delta`是什么？

游戏引擎试图以每秒 60 帧的速率一致运行。然而，这可能会因为 Godot 或计算机本身的减速而改变。如果帧率不一致，那么它将影响你的游戏对象的移动。例如，考虑一个设置为每帧移动`10`像素的对象。如果一切运行顺利，这将转化为在一秒内移动`600`像素。然而，如果其中一些帧耗时更长，那么那一秒可能只有 50 帧，所以对象只移动了`500`像素。

Godot，就像大多数游戏引擎和框架一样，通过传递给你 `delta` 来解决这个问题，这是自上一帧以来经过的时间。大多数情况下，这将是大约 `0.016` 秒（或大约 16 毫秒）。如果你然后将你的期望速度（`600` px/s）乘以 delta，你将得到精确的 `10` 像素移动。然而，如果 `delta` 增加到 `0.3`，则对象将被移动 `18` 像素。总的来说，移动速度保持一致，且与帧率无关。

作为一项额外的好处，你可以用 px/s 而不是 px/frame 的单位来表示你的移动，这更容易可视化。

# 选择动画

现在玩家可以移动了，你需要根据玩家是移动还是静止来更改 `AnimatedSprite` 播放的动画。`run` 动画的美术面向右侧，这意味着它应该使用翻转水平属性（使用 `Flip H` 属性）来翻转，以便向左移动。将以下内容添加到你的 `_process()` 函数末尾：

```cpp
    if velocity.length() > 0:
        $AnimatedSprite.animation = "run"
        $AnimatedSprite.flip_h = velocity.x < 0
    else:
        $AnimatedSprite.animation = "idle"
```

注意，这段代码采取了一些捷径。`flip_h` 是一个布尔属性，这意味着它可以设置为 `true` 或 `false`。布尔值也是比较操作（如 `<`）的结果。正因为如此，我们可以将属性设置为比较操作的结果。这一行代码等同于以下这样写：

```cpp
if velocity.x < 0:
    $AnimatedSprite.flip_h = true
else:
    $AnimatedSprite.flip_h = false     
```

再次播放场景并检查每种情况下动画是否正确。确保在 `AnimatedSprite` 中将 `Playing` 设置为 On，以便动画可以播放。

# 开始和结束玩家的移动

当游戏开始时，主场景需要通知玩家游戏已经开始。添加以下 `start()` 函数，主场景将使用它来设置玩家的起始动画和位置：

```cpp
func start(pos):
    set_process(true)
    position = pos
    $AnimatedSprite.animation = "idle"
```

当玩家撞击障碍物或用完时间时，将调用 `die()` 函数：

```cpp
func die():
    $AnimatedSprite.animation = "hurt"
    set_process(false)
```

设置 `set_process(false)` 将导致 `_process()` 函数不再为该节点调用。这样，当玩家死亡时，他们就不能通过按键输入移动了。

# 准备碰撞

玩家应该检测到它撞击硬币或障碍物时，但你还没有让他们这样做。没关系，因为你可以使用 Godot 的 *信号* 功能来实现这一点。信号是节点发送消息的方式，其他节点可以检测并响应这些消息。许多节点都有内置的信号，例如在身体碰撞时或按钮被按下时发出警报。你还可以定义自定义信号以供自己的用途。

通过 *连接* 信号到你想监听和响应的节点，使用信号。这种连接可以在检查器或代码中完成。在项目后期，你将学习如何以这两种方式连接信号。

将以下内容添加到脚本顶部（在 `extends Area2D` 之后）：

```cpp
signal pickup
signal hurt
```

这些定义了玩家在触摸硬币或障碍物时将*发出*（发送）的自定义信号。触摸将由`Area2D`本身检测。选择`Player`节点并点击检查器旁边的节点标签页，以查看玩家可以发出的信号列表：

![图片](img/00031.jpeg)

注意你的自定义信号也在那里。由于其他对象也将是`Area2D`节点，你想要`area_entered()`信号。选择它并点击连接。在连接信号窗口中点击连接 – 你不需要更改任何设置。Godot 将自动在你的脚本中创建一个名为`_on_Player_area_entered()`的新函数。

当连接一个信号时，你不仅可以让 Godot 为你创建一个函数，还可以指定一个现有函数的名称，将其与信号链接。如果你不希望 Godot 为你创建函数，请将“创建函数”开关切换到关闭状态。

将以下代码添加到这个新函数中：

```cpp
func _on_Player_area_entered( area ):
    if area.is_in_group("coins"):
        area.pickup()
        emit_signal("pickup")
    if area.is_in_group("obstacles"):
        emit_signal("hurt")
        die()
```

当检测到另一个`Area2D`时，它将被传递到函数中（使用`area`变量）。硬币对象将有一个`pickup()`函数，该函数定义了捡起硬币时的行为（例如播放动画或声音）。当你创建硬币和障碍物时，你需要将它们分配到适当的*组*，以便可以检测到。

总结一下，以下是到目前为止完整的玩家脚本：

```cpp
extends Area2D

signal pickup
signal hurt

export (int) var speed
var velocity = Vector2()
var screensize = Vector2(480, 720)

func get_input():
    velocity = Vector2()
    if Input.is_action_pressed("ui_left"):
        velocity.x -= 1
    if Input.is_action_pressed("ui_right"):
        velocity.x += 1
    if Input.is_action_pressed("ui_up"):
        velocity.y -= 1
    if Input.is_action_pressed("ui_down"):
        velocity.y += 1
    if velocity.length() > 0:
        velocity = velocity.normalized() * speed

func _process(delta):
    get_input()
    position += velocity * delta
    position.x = clamp(position.x, 0, screensize.x)
    position.y = clamp(position.y, 0, screensize.y)

    if velocity.length() > 0:
        $AnimatedSprite.animation = "run"
        $AnimatedSprite.flip_h = velocity.x < 0
    else:
        $AnimatedSprite.animation = "idle"

func start(pos):
    set_process(true)
    position = pos
    $AnimatedSprite.animation = "idle"

func die():
    $AnimatedSprite.animation = "hurt"
    set_process(false)

func _on_Player_area_entered( area ):
    if area.is_in_group("coins"):
        area.pickup()
        emit_signal("pickup")
    if area.is_in_group("obstacles"):
        emit_signal("hurt")
        die()
```

# 第二部分 – 硬币场景

在这部分，你将为玩家创建可以收集的硬币。这将是一个独立的场景，描述单个硬币的所有属性和行为。一旦保存，主场景将加载硬币场景并创建多个*实例*（即副本）。

# 节点设置

点击“场景”|“新建场景”并添加以下节点。别忘了像处理“玩家”场景那样设置子节点不被选中：

+   `Area2D`（命名为`Coin`）

+   `AnimatedSprite`

+   `CollisionShape2D`

在添加节点后，请确保保存场景。

按照你在玩家场景中设置的方式设置`AnimatedSprite`。这次，你只有一个动画：一个使硬币看起来不那么扁平和无趣的闪光/闪耀效果。添加所有帧并将速度（FPS）设置为`12`。图像有点太大，所以将`AnimatedSprite`的`Scale`设置为（`0.5`，`0.5`）。在`CollisionShape2D`中使用`CircleShape2D`并调整其大小以覆盖硬币图像。别忘了：在调整碰撞形状大小时，永远不要使用缩放手柄。圆形形状有一个单独的手柄，用于调整圆的半径。

# 使用组

组为节点提供了一个标签系统，允许你识别相似的节点。一个节点可以属于任意数量的组。你需要确保所有硬币都将位于一个名为`coins`的组中，以便玩家脚本能够正确响应触摸硬币。选择`Coin`节点，点击节点标签页（与找到信号相同的标签页）并选择组。在框中输入`coins`，然后点击添加，如图所示：

![图片](img/00032.jpeg)

# 脚本

接下来，向`Coin`节点添加一个脚本。如果你在模板设置中选择空，Godot 将创建一个没有注释或建议的空脚本。硬币脚本的代码比玩家脚本的代码要短得多：

```cpp
extends Area2D

func pickup():
    queue_free()
```

`pickup()`函数由玩家脚本调用，告诉硬币在被收集时要做什么。`queue_free()`是 Godot 的节点移除方法。它安全地从树中移除节点，并从内存中删除它及其所有子节点。稍后，你将在这里添加一个视觉效果，但现在硬币消失的效果就足够了。

`queue_free()`不会立即删除对象，而是将其添加到队列中，在当前帧结束时删除。这比立即删除节点更安全，因为游戏中运行的其它代码可能仍然需要该节点存在。通过等待直到帧的结束，Godot 可以确保所有可能访问该节点的代码都已完成，节点可以安全地被移除。

# 第三部分 - 主场景

`主`场景是连接游戏所有部件的关键。它将管理玩家、硬币、计时器以及游戏的其它部件。

# 节点设置

创建一个新的场景并添加一个名为`Main`的节点。要将玩家添加到场景中，点击实例按钮并选择你保存的`Player.tscn`：

![动画](img/00033.gif)

现在，将以下节点作为`Main`的子节点添加，并按以下命名：

+   `纹理矩形`（命名为`Background`）——用于背景图像

+   `节点`（命名为`CoinContainer`）——用于存放所有硬币

+   `二维位置`（命名为`PlayerStart`）——用于标记`玩家`的起始位置

+   `计时器`（命名为`GameTimer`）——用于跟踪时间限制

确保将`Background`作为第一个子节点。节点将按照显示的顺序绘制，所以在这种情况下背景将在玩家后面。通过将`assets`文件夹中的`grass.png`图像拖动到`Background`节点的纹理属性中，向`Background`节点添加一个图像。然后将拉伸模式更改为平铺，然后点击布局|全矩形以将框架大小调整为屏幕大小，如下面的截图所示：

![图片](img/00034.jpeg)

将`PlayerStart`节点的位置设置为（`240`，`350`）。

你的场景布局应该看起来像这样：

![图片](img/00035.jpeg)

# 主脚本

向`Main`节点（使用空模板）添加一个脚本，并添加以下变量：

```cpp
extends Node

export (PackedScene) var Coin
export (int) var playtime

var level
var score
var time_left
var screensize
var playing = false
```

当你点击`主`时，`Coin`和`Playtime`属性将现在出现在检查器中。从文件系统面板中拖动`Coin.tscn`到`Coin`属性，并将其放置在`Coin`属性中。将`Playtime`设置为`30`（这是游戏将持续的时间）。剩余的变量将在代码的后续部分使用。

# 初始化

接下来，添加`_ready()`函数：

```cpp
func _ready():
    randomize()
    screensize = get_viewport().get_visible_rect().size
    $Player.screensize = screensize
    $Player.hide()
```

在 GDScript 中，你可以使用`$`通过名称引用特定的节点。这允许你找到屏幕的大小并将其分配给玩家的`screensize`变量。`hide()`使玩家一开始不可见（你将在游戏真正开始时让它们出现）。

在`$`符号表示法中，节点名称相对于运行脚本的节点。例如，`$Node1/Node2`将指代一个节点（`Node2`），它是`Node1`的子节点，而`Node1`本身又是当前运行脚本的子节点。Godot 的自动完成功能将在你输入时建议树中的节点名称。请注意，如果节点的名称包含空格，你必须将其放在引号内，例如，`$"My Node"`。

如果你想要每次运行场景时“随机”数字序列都不同，你必须使用`randomize()`。从技术上讲，这为随机数生成器选择了一个随机的**种子**。

# 开始新游戏

接下来，`new_game()`函数将为新游戏初始化一切：

```cpp
func new_game():
    playing = true
    level = 1
    score = 0
    time_left = playtime
    $Player.start($PlayerStart.position)
    $Player.show()
    $GameTimer.start()
    spawn_coins()
```

除了将变量设置为起始值外，此函数还调用玩家的`start()`函数以确保它移动到正确的起始位置。游戏计时器开始，这将倒计时剩余的游戏时间。

你还需要一个函数，该函数将根据当前级别创建一定数量的硬币：

```cpp
func spawn_coins():
    for i in range(4 + level):
        var c = Coin.instance()
        $CoinContainer.add_child(c)
        c.screensize = screensize
        c.position = Vector2(rand_range(0, screensize.x),
        rand_range(0, screensize.y))
```

在这个函数中，你创建`Coin`对象（这次是通过代码，而不是点击实例化场景按钮）的多个实例，并将其添加为`CoinContainer`的子节点。每次实例化一个新的节点时，都必须使用`add_child()`将其添加到树中。最后，你为硬币随机选择一个出现的位置。你将在每个级别的开始时调用这个函数，每次生成更多的硬币。

最终，你希望当玩家点击开始按钮时调用`new_game()`。现在，为了测试一切是否正常工作，将`new_game()`添加到你的`_ready()`函数的末尾，并点击**播放项目**（*F5*）。当你被提示选择主场景时，选择`Main.tscn`。现在，每次你播放项目时，`Main`场景将被启动。

到目前为止，你应该在屏幕上看到你的玩家和五个硬币。当玩家触摸一个硬币时，它就会消失。

# 检查剩余的硬币

主脚本需要检测玩家是否已经捡起所有硬币。由于硬币都是`CoinContainer`的子节点，你可以使用此节点的`get_child_count()`来找出剩余多少个硬币。将此放入`_process()`函数中，以便每帧都会进行检查：

```cpp
func _process(delta):
    if playing and $CoinContainer.get_child_count() == 0:
        level += 1
        time_left += 5
        spawn_coins()
```

如果没有更多的硬币剩余，那么玩家将进入下一级。

# 第四部分 – 用户界面

你的游戏需要的最后一部分是一个**用户界面**（**UI**）。这是一个用于在游戏过程中显示玩家需要看到的信息的界面。在游戏中，这也被称为**抬头显示**（**HUD**），因为信息以叠加的形式显示在游戏视图之上。你还会使用这个场景来显示一个开始按钮。

HUD 将显示以下信息：

+   分数

+   剩余时间

+   一条消息，例如游戏结束

+   一个开始按钮

# 节点设置

创建一个新的场景并添加一个名为`HUD`的`CanvasLayer`节点。`CanvasLayer`节点允许你在游戏其他元素之上绘制 UI 元素，这样显示的信息不会被玩家或金币等游戏元素覆盖。

Godot 提供了各种 UI 元素，可用于创建从健康条等指示器到复杂界面如存货界面等任何内容。实际上，你用来制作这个游戏的 Godot 编辑器就是使用这些元素在 Godot 中构建的。UI 元素的基本节点是从`Control`扩展的，在节点列表中以绿色图标显示。要创建你的 UI，你将使用各种`Control`节点来定位、格式化和显示信息。以下是完成后的`HUD`的外观：

![](img/00036.jpeg)

# 锚点和边距

控制节点具有位置和大小，但它们还具有称为**锚点**和**边距**的属性。锚点定义了节点边缘相对于父容器的起点或参考点。边距表示控制节点边缘与其对应锚点之间的距离。当你移动或调整控制节点的大小时，边距会自动更新。

# 消息标签

在场景中添加一个`Label`节点并将其名称更改为`MessageLabel`**。** 这个标签将显示游戏标题，以及游戏结束时显示的 Game Over。这个标签应该在游戏屏幕上居中。你可以用鼠标拖动它，但为了精确放置 UI 元素，你应该使用锚点属性。

选择视图 | 显示辅助工具以显示帮助您看到锚点位置的标记，然后点击布局菜单并选择水平居中宽：

![](img/00037.jpeg)

`MessageLabel`现在横跨屏幕宽度并垂直居中。检查器中的文本属性设置标签显示的文本。将其设置为 Coin Dash!并将对齐和垂直对齐设置为居中。

`Label`节点的默认字体非常小，所以下一步是为其分配一个自定义字体。在检查器中向下滚动到自定义字体部分，并选择新建动态字体，如图下所示：

![](img/00038.jpeg)

现在，点击动态字体，你可以调整字体设置。从文件系统坞中，拖动`Kenney Bold.ttf`字体并将其放入字体数据属性中。将大小设置为**`48`**，如图下所示：

![](img/00039.jpeg)

# 得分和时间显示

`HUD`的顶部将显示玩家的得分和时钟剩余时间。这两个都将使用`Label`节点，分别位于游戏屏幕的两侧。而不是单独定位它们，你将使用`Container`节点来管理它们的位置。

# 容器

UI 容器自动排列其子`Control`节点（包括其他`Containers`）的位置。您可以使用它们在元素周围添加填充、居中元素或按行或列排列元素。每种类型的`Container`都有特殊的属性来控制它们如何排列子元素。您可以在检查器的“自定义常量”部分中查看这些属性。

记住，容器**自动**排列其子元素。如果您移动或调整容器节点内的控件的大小，会发现它自动回到原始位置。您可以手动排列控件**或**使用容器排列控件，但不能同时进行。

要管理分数和时间标签，向`HUD`的**`MarginContainer`**节点添加一个**`MarginContainer`**节点。使用布局菜单设置锚点为顶部宽。在“自定义常量”部分中，将边距右、边距顶和边距左设置为`10`。这将添加一些填充，以便文本不会紧贴屏幕边缘。

由于分数和时间标签将使用与`MessageLabel`相同的字体设置，因此如果您复制它将节省时间。单击`MessageLabel`并按*Ctrl* + *D* (*Cmd* + *D* 在 macOS 上)两次以创建两个副本标签。将它们都拖动并放在`MarginContainer`上，使它们成为其子元素。将一个命名为`ScoreLabel`，另一个命名为`TimeLabel`，并将两者的文本属性都设置为`0`。将`ScoreLabel`的对齐方式设置为左对齐，将**`TimeLabel`**的对齐方式设置为右对齐。

# 通过 GDScript 更新 UI

将脚本添加到`HUD`节点。此脚本将在需要更改属性时更新 UI 元素，例如，每当收集到金币时更新分数文本。参考以下代码：

```cpp
extends CanvasLayer

signal start_game

func update_score(value):
    $MarginContainer/ScoreLabel.text = str(value)

func update_timer(value):
    $MarginContainer/TimeLabel.txt = str(value)
```

`Main`场景的脚本将调用这些函数来更新显示，每当值发生变化时。对于`MessageLabel`，您还需要一个计时器，以便在短时间内消失。添加一个`Timer`节点，并将其名称更改为`MessageTimer`**。**在检查器中，将等待时间设置为`2`秒，并勾选复选框以设置单次触发为开启。这确保了当启动时，计时器只会运行一次，而不是重复。添加以下代码：

```cpp
func show_message(text):
    $MessageLabel.text = text
    $MessageLabel.show()
    $MessageTimer.start()
```

在此函数中，您显示消息并启动计时器。要隐藏消息，连接`MessageTimer`的`timeout()`信号并添加以下代码：

```cpp
func _on_MessageTimer_timeout():
    $MessageLabel.hide()
```

# 使用按钮

添加一个`Button`节点，并将其名称更改为`StartButton`**。**此按钮将在游戏开始前显示，点击后将隐藏自身并向`Main`场景发送信号以开始游戏。将文本属性设置为“开始”，并更改自定义字体，就像您对**`MessageLabel`**所做的那样。在布局菜单中，选择“居中底部”。这将使按钮位于屏幕底部，因此可以通过按*上*箭头键或通过编辑边距并将顶部设置为`-150`、底部设置为`-50`来稍微向上移动它。

当按钮被点击时，会发出一个信号。在`StartButton`的节点标签页中，连接`pressed()`信号：

```cpp
func _on_StartButton_pressed():
    $StartButton.hide()
    $MessageLabel.hide()
    emit_signal("start_game")
```

`HUD`发出`start_game`信号，通知`Main`是时候开始新游戏了。

# 游戏结束

你 UI 的最终任务是响应游戏结束：

```cpp
func show_game_over():
    show_message("Game Over")
    yield($MessageTimer, "timeout")
    $StartButton.show()
    $MessageLabel.text = "Coin Dash!"
    $MessageLabel.show()
```

在这个功能中，你需要游戏结束信息显示两秒钟后消失，这正是`show_message()`所做到的。然而，你希望在信息消失后显示开始按钮。`yield()`函数暂停函数的执行，直到给定的节点（`MessageTimer`）发出给定的信号（`timeout`）。一旦接收到信号，函数继续执行，返回到初始状态，这样你就可以再次玩游戏。

# 将 HUD 添加到 Main

现在，你需要设置`Main`场景和`HUD`之间的通信。将`HUD`场景的实例添加到`Main`场景中。在`Main`场景中，连接`GameTimer`的`timeout()`信号，并添加以下内容：

```cpp
func _on_GameTimer_timeout():
    time_left -= 1
    $HUD.update_timer(time_left)
    if time_left <= 0:
        game_over()
```

每当`GameTimer`超时（每秒一次），剩余时间会减少。

接下来，连接`Player`的`pickup()`和`hurt()`信号：

```cpp
func _on_Player_pickup():
    score += 1
    $HUD.update_score(score)

func _on_Player_hurt():
    game_over()
```

游戏结束时需要发生几件事情，所以添加以下函数：

```cpp
func game_over():
    playing = false
    $GameTimer.stop()
    for coin in $CoinContainer.get_children():
        coin.queue_free()
    $HUD.show_game_over()
    $Player.die()
```

此函数使游戏暂停，并遍历硬币，移除任何剩余的硬币，同时调用 HUD 的`show_game_over()`函数。

最后，`StartButton`需要激活`new_game()`函数。点击`HUD`实例并选择其`new_game()`信号。在信号连接对话框中，点击“Make Function to Off”，在“Method In Node”字段中输入`new_game`。这将连接信号到现有函数而不是创建一个新的函数。请看以下截图：

![图片](img/00040.jpeg)

将`new_game()`从`_ready()`函数中移除，并将以下两行添加到`new_game()`函数中：

```cpp
$HUD.update_score(score)
$HUD.update_timer(time_left)
```

现在，你可以开始玩游戏了！确认所有部分都按预期工作：得分、倒计时、游戏结束和重新开始等。如果你发现某个部分不工作，请返回并检查你创建它的步骤，以及它连接到游戏其他部分的步骤。

# 第五部分 - 完成工作

你已经创建了一个可以工作的游戏，但它仍然可以变得更有趣。游戏开发者使用术语*juice*来描述使游戏感觉好玩的事物。juice 可以包括声音、视觉效果或任何其他增加玩家享受的东西，而无需改变游戏玩法本身。

在本节中，你将添加一些小的*juicy*功能来完成游戏。

# 视觉效果

当你捡起硬币时，它们只是消失，这并不很有吸引力。添加视觉效果将使收集大量硬币变得更加令人满意。

首先，向`Coin`场景添加一个`Tween`节点。

# 什么是 tween？

**tween** 是一种通过特定函数在时间上（从起始值到结束值）逐渐插值（改变）某些值的方法。例如，你可能选择一个稳定改变值的函数，或者一个开始缓慢但逐渐加速的函数。tweening 也被称为 *easing*。

当在 Godot 中使用 `Tween` 节点时，你可以将其分配给改变节点的一个或多个属性。在这种情况下，你将增加硬币的 `Scale` 并使用 Modulate 属性使其淡出。

将此行添加到 `Coin` 的 `_ready()` 函数中：

```cpp
$Tween.interpolate_property($AnimatedSprite, 'scale',
                            $AnimatedSprite.scale,
                            $AnimatedSprite.scale * 3, 0.3,
                            Tween.TRANS_QUAD,
                            Tween.EASE_IN_OUT)
```

`interpolate_property()` 函数会导致 `Tween` 改变节点的属性。有七个参数：

+   影响的节点

+   要改变的属性

+   属性的起始值

+   属性的结束值

+   持续时间（以秒为单位）

+   要使用的函数

+   方向

当玩家捡起硬币时，tween 应该开始播放。在 `pickup()` 函数中替换 `queue_free()`：

```cpp
func pickup():
    monitoring = false
    $Tween.start() 
```

将 `monitoring` 设置为 `false` 确保当玩家在 tween 动画期间触摸硬币时，不会发出 `area_enter()` 信号。

最后，当动画结束时，应该删除硬币，因此连接 `Tween` 节点的 `tween_completed()` 信号：

```cpp
func _on_Tween_tween_completed(object, key):
    queue_free()
```

现在，当你运行游戏时，你应该看到当捡起硬币时，硬币会变大。这是好的，但将 tween 应用到多个属性同时时，效果会更明显。你可以添加另一个 `interpolate_property()`，这次用来改变精灵的不透明度。这是通过改变 `modulate` 属性实现的，它是一个 `Color` 对象，并改变其 alpha 通道从 `1`（不透明）到 `0`（透明）。参考以下代码：

```cpp
$Tween.interpolate_property($AnimatedSprite, 'modulate', 
                            Color(1, 1, 1, 1),
                            Color(1, 1, 1, 0), 0.3,
                            Tween.TRANS_QUAD,
                            Tween.EASE_IN_OUT)
```

# 声音

声音是游戏设计中最重要的但经常被忽视的部分之一。良好的声音设计可以在非常小的努力下给你的游戏增添巨大的活力。声音可以给玩家反馈，将他们与角色情感上联系起来，甚至成为游戏玩法的一部分。

对于这个游戏，你将添加三个音效。在 `Main` 场景中，添加三个 `AudioStreamPlayer` 节点，并分别命名为 `CoinSound`、`LevelSound` 和 `EndSound`。将每个声音从 `audio` 文件夹（你可以在 FileSystem 中的 `assets` 下找到它）拖动到每个节点的相应 Stream 属性中。

要播放声音，你可以在其上调用 `play()` 函数。将 `$CoinSound.play()` 添加到 `_on_Player_pickup()` 函数中，`$EndSound.play()` 添加到 `game_over()` 函数中，以及 `$LevelSound.play()` 添加到 `spawn_coins()` 函数中。

# 提升物品

有很多可能性可以给玩家带来小的优势或提升。在本节中，你将添加一个提升物品，当收集时会给玩家一小段时间奖励。它将偶尔短暂出现，然后消失。

新场景将与你已经创建的 `Coin` 场景非常相似，所以点击你的 `Coin` 场景，选择 Scene | Save Scene As 并将其保存为 `Powerup.tscn`。将根节点名称更改为 Powerup 并通过点击清除脚本按钮移除脚本：![](img/00041.jpeg)。你还应该断开 `area_entered` 信号（你稍后会重新连接它）。在 Groups 选项卡中，通过点击删除按钮（看起来像垃圾桶）将硬币组移除，并将其添加到名为 `powerups` 的新组中。

在 `AnimatedSprite` 中，将硬币的图像更改为 `powerup`，你可以在 `res://assets/pow/` 文件夹中找到它。

点击添加新脚本，并将 `Coin.gd` 脚本中的代码复制过来。将 `_on_Coin_area_entered` 的名称更改为 `_on_Powerup_area_entered` 并再次将 `area_entered` 信号连接到它。记住，这个函数名称将由信号连接窗口自动选择。

接下来，添加一个名为 `Lifetime` 的 `Timer` 节点。这将限制对象在屏幕上停留的时间。将其等待时间设置为 `2`，并将单次触发和自动启动都设置为开启。连接其超时信号，以便在时间周期结束时将其移除：

```cpp
func _on_Lifetime_timeout():
    queue_free()
```

现在，前往你的主场景并添加另一个名为 `PowerupTimer` 的 `Timer` 节点。将其单次触发属性设置为开启。在 `audio` 文件夹中还有一个 `Powerup.wav` 声音，你可以通过另一个 `AudioStreamPlayer` 添加。

连接 `timeout` 信号并添加以下代码以生成 `Powerup`：

```cpp
func _on_PowerupTimer_timeout():
    var p = Powerup.instance()
    add_child(p)
    p.screensize = screensize
    p.position = Vector2(rand_range(0, screensize.x),
                         rand_range(0, screensize.y))
```

`Powerup` 场景需要通过添加变量，然后将场景拖动到检查器中的属性来链接，就像你之前对 `Coin` 场景所做的那样：

```cpp
export (PackedScene) var Powerup
```

能量提升物品应该以不可预测的方式出现，所以 `PowerupTimer` 的等待时间需要在开始新关卡时设置。在 `spawn_coins()` 生成新硬币后，将此代码添加到 `_process()` 函数中：

```cpp
$PowerupTimer.wait_time = rand_range(5, 10)
$PowerupTimer.start()
```

现在你将会有能量提升物品出现，最后一步是在收集到一个时给玩家一些额外的时间。目前，玩家脚本假设它遇到的是硬币或障碍物。将 `Player.gd` 中的代码更改以检查被击中的对象类型：

```cpp
func _on_Player_area_entered( area ):
    if area.is_in_group("coins"):
        area.pickup()
        emit_signal("pickup", "coin")
    if area.is_in_group("powerups"):
        area.pickup()
        emit_signal("pickup", "powerup")
    if area.is_in_group("obstacles"):
        emit_signal("hurt")
        die()
```

注意，现在你正在使用一个额外的参数来命名对象的类型来发射拾取信号。`Main.gd` 中的相应函数现在可以接受该参数，并使用 `match` 语句来决定采取什么行动：

```cpp
func _on_Player_pickup(type):
    match type:
        "coin":
            score += 1
            $CoinSound.play()
            $HUD.update_score(score)
        "powerup":
            time_left += 5
            $PowerupSound.play()
            $HUD.update_timer(time_left)
```

`match` 语句是 `if` 语句的有用替代品，尤其是在你有大量可能值要测试时。

尝试运行游戏并收集能量提升物品。确保声音播放并且计时器增加五秒钟。

# 硬币动画

当你创建了 `Coin` 场景时，你添加了一个 `AnimatedSprite`，但它还没有开始播放。硬币动画显示一个在硬币表面移动的 *闪烁* 效果。如果所有硬币同时显示这个效果，看起来会太规律，所以每个硬币的动画都需要一个小的随机延迟。

首先，点击 `AnimatedSprite`，然后点击 *Frames* 资源。确保 Loop 设置为 Off，并且 Speed 设置为 `12`。

将 `Timer` 节点添加到 `Coin` 场景中，并在 `_ready()` 中添加以下代码：

```cpp
$Timer.wait_time = rand_range(3, 8)
$Timer.start()
```

现在，将 `Timer` 的 `timeout()` 信号连接起来，并添加以下内容：

```cpp
func _on_Timer_timeout():
    $AnimatedSprite.frame = 0
    $AnimatedSprite.play()
```

尝试运行游戏，并观察硬币的动画效果。这需要很少的努力，却是一个很好的视觉效果。你会在专业游戏中注意到很多这样的效果。虽然很微妙，但视觉吸引力使得游戏体验更加愉悦。

前面的 `Powerup` 对象有类似的动画，你可以以相同的方式添加。

# 障碍物

最后，通过引入玩家必须避免的障碍物，可以使游戏更具挑战性。触摸障碍物将结束游戏。

为仙人掌创建一个新的场景，并添加以下节点：

+   `Area2D`（命名为 `Cactus`）

+   `Sprite`

+   `CollisionShape2D`

将仙人掌纹理从 FileSystem 选项卡拖动到 `Sprite` 的 Texture 属性。向碰撞形状添加一个 `RectangleShape2D`，并调整其大小以覆盖图像。记得你之前在玩家脚本中添加了 `if area.is_in_group("obstacles")` 吗？使用节点选项卡（在检查器旁边）将 `Cactus` 身体添加到 `obstacles` 组。

现在，将一个 `Cactus` 实例添加到 `Main` 场景中，并将其移动到屏幕上半部分的一个位置（远离玩家出生点）。玩玩游戏，看看当你撞到仙人掌时会发生什么。

你可能已经发现了一个问题：硬币可能会在仙人掌后面生成，这使得它们无法被捡起。当硬币放置时，如果它检测到与障碍物重叠，它需要移动。连接硬币的 `area_entered()` 信号并添加以下内容：

```cpp
func _on_Coin_area_entered( area ):
    if area.is_in_group("obstacles"):
        position = Vector2(rand_range(0, screensize.x), rand_range(0, screensize.y))
```

如果你已经添加了前面的 `Powerup` 对象，你还需要对其 `area_entered` 信号做同样的处理。

# 概述

在本章中，你通过创建一个基本的 2D 游戏学习了 Godot 引擎的基础知识。你设置了项目并创建了多个场景，处理精灵和动画，捕获用户输入，使用 *signals* 与事件通信，并使用 **Control** 节点创建 UI。在这里学到的技能是你在任何 Godot 项目中都会用到的关键技能。

在进入下一章之前，查看一下项目。你理解每个节点的作用吗？有没有你不理解的代码片段？如果有，请返回并复习该章节的相关部分。

此外，你也可以自由地尝试游戏并改变一些东西。了解游戏不同部分如何工作的最好方法之一就是改变它们并观察会发生什么。

在下一章中，你将探索更多 Godot 的功能，并通过构建一个更复杂的游戏来学习如何使用更多节点类型。
