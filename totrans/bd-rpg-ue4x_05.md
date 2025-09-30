# 第5章。连接角色统计信息

现在我们已经为我们的暂停菜单设置了一个基本框架，我们将现在专注于暂停菜单的编程方面。

在本章中，你将学习如何将角色统计信息链接到暂停菜单，正如在第4章[Chapter 4](ch04.html "Chapter 4. Pause Menu Framework")中讨论的*暂停菜单框架*。到本章结束时，你将能够将任何其他你希望链接到UMG菜单或子菜单的游戏统计信息链接起来。本章我们将涵盖以下主题：

+   获取角色数据

+   获取玩家实例

+   显示状态

# 获取角色数据

到目前为止，暂停菜单已经完全设计完成，并准备好进行数据集成。在第3章[Chapter 3](ch03.html "Chapter 3. Exploration and Combat")中，*探索与战斗*，我们开发了一些方法来显示一些玩家参数，例如玩家的名字、HP和MP，通过将文本块与**Game Character**变量绑定到CombatUI，以便访问**Character Info**中持有的角色状态值。我们将以与上一章非常相似的方式完成这项工作，首先打开**Pause_Main**小部件，然后点击我们将要更新其值的文本块。

在这种情况下，我们已经为所有状态值指定了位置，因此我们将从名为**Editable_Soldier_HP**的HP状态值开始：

![获取角色数据](img/B04548_05_01.jpg)

导航到**Content** | **Text**，然后点击旁边的下拉菜单中的**Bind**下拉菜单。在下拉菜单中点击**Create Binding**：

![获取角色数据](img/B04548_05_02.jpg)

完成此过程后，将创建一个新的函数`Get_Editable_Soldier_HP_Text_0`，你将自动被拉入新函数的图表。与之前的绑定一样，新函数也将自动具有**FunctionEntry**及其标记的返回值：

![获取角色数据](img/B04548_05_03.jpg)

我们现在可以创建一个新的**Game Character**引用变量，我们再次将其命名为**Character Target**：

![获取角色数据](img/B04548_05_04.jpg)

然后，我们将**Character Target**变量拖入`Get_Editable_Soldier_HP_Text_0`图表，并将其设置为**Get**：

![获取角色数据](img/B04548_05_05.jpg)

接下来，我们将创建一个名为**Get HP**的新节点，它位于**Variables** | **Character Info**下，并将它的**Target**引脚链接到**Character Target**变量引脚：

![获取角色数据](img/B04548_05_06.jpg)

最后，将**Get Editable Soldier HP Text 0**节点中的HP状态值链接到**ReturnNode**的**Return Value**引脚。这将自动创建一个**To Text (Int)**转换节点，该节点负责将任何整数转换为字符串。完成之后，你的`Get_Editable_Soldier_HP_Text_0`函数应该看起来像这样：

![获取角色数据](img/B04548_05_07.jpg)

# 获取玩家实例

如果你现在进行测试，你会看到在我们的暂停菜单中创建了一个值，但这个值是**0**。这是不正确的，因为根据角色的当前属性，我们的角色应该从100 HP开始：

![获取玩家实例](img/B04548_05_08.jpg)

问题发生是因为访问暂停菜单的**Field Player**从未将我们的任何角色数据分配给**Character Target**。我们可以在蓝图（Blueprint）中轻松设置正确的角色目标，但如果没有将我们的添加的队伍成员暴露给蓝图，我们将无法分配任何角色数据。因此，我们首先需要进入`RPGGameInstance.h`，并允许我们的当前游戏数据在`UProperty`参数中暴露给蓝图中的**游戏数据**类别：

[PRE0]

你的`RPGGameInstance`.`h`文件现在应该看起来像这样：

[PRE1]

保存并编译你的代码后，你应该能够在蓝图（Blueprint）中正确调用任何创建和添加的队伍成员，因此我们应该通过**Field Player**蓝图获得读取访问权限。

现在，你可以导航回**Field Player**蓝图，并通过创建位于**游戏**下的**Get Game Instance**函数节点来获取**RPGGameInstance**：

![获取玩家实例](img/B04548_05_09.jpg)

将**Get Game Instance**的**返回值**转换为**RPGGameInstance**，它位于**实用工具** | **转换** | **RPGGameInstance**。现在你已经得到了**RPGGameInstance**类的实例，你可以通过导航到在**GameData**下的**变量**中为它创建的类别，让这个实例引用包含所有队伍成员的`TArray`的**队伍成员**：

![获取玩家实例](img/B04548_05_11.jpg)

在这里，我们需要指向包含我们的士兵角色属性的数组元素，这是我们的第一个元素或数组的`0`索引，通过将**队伍成员**数组链接到一个**GET**函数来实现，该函数可以通过导航到**实用工具** | **数组**找到：

![获取玩家实例](img/B04548_05_12.jpg)

### 注意

对于额外的角色，你需要将另一个**GET**函数链接到**队伍成员**，并让**GET**函数指向数组中指向任何其他角色的元素（例如，如果你有一个位于索引1的治疗师，你的第二个**GET**函数只需将其索引列为1而不是0，以从治疗师的属性中提取）。现在，我们只是专注于士兵的属性，但你将想要获取队伍中每个角色的属性。

最后，一旦我们完成了**RPGGameInstance**的施法，我们需要将我们在暂停菜单中创建的**Character Target**设置为我们的**队伍成员**。为此，右键单击你的**事件图**以创建一个新动作，但取消选中**上下文相关**，因为我们正在寻找在另一个类（`Pause_Main`）中声明的变量。如果你导航到**类** | **Pause Main**，你会找到**Set Character Target**：

![获取玩家实例](img/B04548_05_13.jpg)

在这里，只需将 **Character Target** 链接到你的 **GET** 函数的输出引脚：

![获取玩家实例](img/B04548_05_14.jpg)

然后，设置 **Character Target** 以在 **RPGGameInstance** 被调用后触发：

![获取玩家实例](img/B04548_05_15.jpg)

# 显示属性

现在，我们需要选择一个合适的位置来调用 **RPGGameInstance**。最好在创建暂停菜单后调用 **RPGGameInstance**，因此将 **Set Show MouseCursor** 节点的输出引脚链接到 **Cast To RPGGameInstance** 节点的输入引脚。然后，将 **Create Pause_Main Widget** 的 **Return Value** 链接到 **Set Character Target** 的 **Target**。当你完成时，你的 **FieldPlayer** 下的 **EventGraph** 应该看起来像这样：

![显示属性](img/B04548_05_16.jpg)

当你完成时，你会看到士兵的 HP 正确显示为当前 HP：

![显示属性](img/B04548_05_17.jpg)

现在，你可以通过绑定函数并将这些函数返回值（如角色目标的 MP 和名称）添加到暂停菜单中的 **Pause_Main** 中的文本块。当你完成你的士兵角色后，你的 **Pause_Main** 应该看起来像这样：

![显示属性](img/B04548_05_18.jpg)

### 注意

我们还没有等级或经验，我们将在后面的章节中介绍等级和经验。

如果你还有其他角色，请确保你也添加它们。如前所述，如果你的队伍中有其他角色，你需要回到你的 **FieldPlayer** 事件图并创建另一个 **GET** 函数，该函数将获取其他队伍成员的索引并将它们分配给新的 **Character Targets**。

现在我们回到 **Pause_Inventory** 小部件，并将角色属性绑定到相应的文本块。就像在 **Pause_Main** 中一样，选择一个你想要绑定的文本块；在这种情况下，我们将获取 **HP** 右侧的 **Text Block**：

![显示属性](img/B04548_05_19.jpg)

然后，简单地为文本块创建一个绑定，就像你为其他文本块所做的那样。这将当然为一个新的函数创建一个绑定，我们将返回 **Character Target** 的 HP 状态。问题是我们在 **Pause_Main** 中创建的 **Character Target** 是一个局部于 **Pause_Main** 的 **Game Character** 变量，因此我们不得不在 **Pause_Inventory** 中重新创建 **Character Target** 变量。幸运的是，步骤是相同的；我们只需要添加一个新的变量并将其命名为 **Character Target**，然后将其类型设置为指向 **Game Character** 的对象引用：

![显示属性](img/B04548_05_20.jpg)

当你完成时，添加**Character Target**变量作为getter，将**Character Target**变量链接到获取你角色的HP，并将该值链接到你的**ReturnNode**的**Return Value**，就像你之前做的那样。你应该有一个看起来与以下截图非常相似的Event Graph：

![显示统计信息](img/B04548_05_21.jpg)

如果你此时测试库存屏幕，你会看到HP值为0，但不要慌张，现在由于**FieldPlayer**为我们的人物提供了一个通用的框架，你不需要进行太多关键的思考来纠正这个值。如果你记得，当我们创建**FieldPlayer**类中的**Pause_Main**小部件后，我们从游戏实例中拉取了我们添加的团队成员，并将其设置在**Pause_Main**中的**Character Target**。我们需要执行类似的步骤，但不是在**FieldPlayer**中开始检索团队成员，而是在创建**Pause_Inventory**的类中执行，该类是在**Pause_Main**中创建的。所以，导航到**Pause_Main**小部件的事件图：

![显示统计信息](img/B04548_05_22.jpg)

在前面的截图中，我们看到我们通过点击相应的按钮创建了**Pause_Inventory**和**Pause_Equipment**小部件。当屏幕创建完成后，我们移除当前视口。这是一个创建我们的**RPGGameInstance**的完美位置。所以，如前所述，创建一个位于**Game**下的**Get Game Instance**。然后，通过转到**Utilities** | **Casting**将返回值设置为**Cast to RPGGameInstance**，这将引用位于**Variables**下的**Game Data**中的**Party Members**数组。在这里，你将通过转到**Utilities** | **Array**使用**Get**函数，并将其链接到**Party Members**数组，拉取索引0。这就是你应该做的，到目前为止，步骤与你之前在**FieldPlayer**中做的相同：

![显示统计信息](img/B04548_05_23.jpg)

当你设置**Character Target**时，会设置不同的差异。如前所述，我们将设置我们新创建的**Character Target**变量的**Character Target**变量为**Pause_Inventory**：

![显示统计信息](img/B04548_05_24.jpg)

一旦完成，将**Cast To RPGGameInstance**的输出引脚链接到**Set Character Target**的输入引脚。同时，将**Get**链接到**Character Target**：

![显示统计信息](img/B04548_05_25.jpg)

最后，将来自**Pause_Inventory**的**Add to Viewport**的输出引脚链接到**Cast To RPGGameInstance**的输入引脚，以触发角色统计信息的检索，并将**Create Pause_Inventory Widget**的**Return Value**链接到**Set Character Target**的**Target**：

![显示统计信息](img/B04548_05_26.jpg)

在这一点上，如果你测试库存屏幕，你会注意到HP值被正确显示：

![显示统计信息](img/B04548_05_28.jpg)

现在你已经知道了如何从**Pause_Main**创建对党派成员的引用，你可以遵循相同的步骤将每个党派成员设置为**Pause_Inventory**中的角色目标。但首先，我们需要通过在每个属性的相应文本块中创建绑定并设置每个文本块的**返回值**为从**角色目标**检索到的值来完成**Pause_Inventory**中所有属性值显示。

一旦你在**Pause_Inventory**中的士兵上完成操作，你将看到类似这样的东西：

![显示属性](img/B04548_05_29.jpg)

到目前为止，你可以轻松地返回到**Pause_Equipment**，创建一个新的**Character Target**变量，然后在**Pause_Main**中显示**Pause_Equipment**时将**Party Members**设置为**Character Target**变量，就像你在**Pause_Inventory**中做的那样。当你完成时，**Pause_Main**事件图中的**Inventory**和**Equipment**按钮应该看起来像这样：

![显示属性](img/B04548_05_30.jpg)

在**Pause_Equipment**小部件中，我们只能绑定**AP**、**DP**、**Lk**和**Name**文本块，因为我们将在稍后处理武器。如果你以与绑定**Pause_Inventory**文本块完全相同的方式使用新创建的**Character Target**绑定这些文本块，你的**Equipment**屏幕在测试时将看起来像这样：

![显示属性](img/B04548_05_31.jpg)

到目前为止，我们已经完成了将角色属性绑定到我们的暂停菜单屏幕的工作。

# 摘要

在本章中，我们将当前的角色属性添加到了暂停菜单。现在我们已经熟悉了UMG，我们将继续通过对话框与NPC进行通信，并添加一个商店到游戏中。
