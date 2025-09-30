# 第 4 章。与形状和图形一起工作

在本章中，我们开发了一个能够绘制线条、箭头、矩形和椭圆的程序。该应用可以被视为圆应用的更高级版本。类似于圆应用，我们有一个图形列表，并捕获用户的鼠标动作。然而，这里有四种不同的图形：线条、箭头、矩形和椭圆。它们定义在一个类似于但比俄罗斯方块游戏中的层次结构更高级的类层次结构中。此外，我们还引入了剪切、复制、粘贴、光标控制和注册处理：

![与形状和图形一起工作](img/image_04_001.jpg)

用户可以添加新的图形，移动一个或多个图形，通过抓取图形的端点修改图形，通过按鼠标按钮和 *Ctrl* 键标记和取消标记图形，并通过矩形包围多个图形来标记多个图形。当一个图形被标记时，它会被小黑方块标注。用户可以通过抓取其中一个方块来修改图形的形状。用户还可以通过抓取图形的其他部分来移动图形。

# MainWindow 函数

本应用中的 `MainWindow` 函数与 [第 3 章](ch03.html "第 3 章。构建俄罗斯方块应用") 中的相似，*构建俄罗斯方块应用*；它设置应用程序名称并创建主文档窗口：

[PRE0]

# DrawDocument 类

`DrawDocument` 类扩展了 `StandardDocument` 框架，类似于圆应用。它捕获鼠标事件，重写文件方法，实现剪切、复制和粘贴，以及光标处理：

**DrawDocument.h**

[PRE1]

与圆应用类似，我们使用 `OnMouseDown`、`OnMouseMove` 和 `OnMouseUp` 方法捕获鼠标动作。然而，在这个应用中，我们还使用 `OnDoubleClick` 方法捕获双击。当用户双击一个图形时，它将执行单独的操作：

[PRE2]

当窗口的客户区域需要重绘时调用 `OnDraw` 方法。它绘制图形，以及如果用户正在用矩形标记图形，则绘制包围图形的矩形：

[PRE3]

当用户选择 **新建** 菜单项时调用 `ClearDocument` 方法，当用户选择 **打开** 菜单项时调用 `ReadDocumentFromStream` 方法，当用户选择 **保存** 或 **另存为** 菜单项时调用 `WriteDocumentToStream` 方法：

[PRE4]

每个图形都有一个整数标识值，该值由 `WriteDocumentToStream` 方法写入并由 `ReadDocumentFromStream` 方法读取，以决定哪个图形需要被创建。给定标识值，`CreateFigure` 方法创建新的图形：

[PRE5]

在这个应用程序中，我们引入了剪切、复制和粘贴的功能。当用户在 **编辑** 菜单中选择 **剪切** 或 **复制** 菜单项时，调用 `CopyGeneric` 方法；当用户选择 **粘贴** 菜单项时，调用 `PasteGeneric` 方法。在 `StandardDocument` 框架中，还有用于剪切、复制和粘贴 ASCII 和 Unicode 文本的方法。然而，我们在这个应用程序中没有使用它们：

[PRE6]

`CopyEnable` 方法返回 `true` 如果信息已准备好复制。在这种情况下，**剪切**、**复制**和**删除**菜单项被启用。在这个应用程序中，我们没有重写 `PasteEnable` 方法，因为 `StandardDocument` 框架会查找全局剪贴板中是否有适合粘贴的内存缓冲区。当用户选择 **删除** 菜单项时，调用 `OnDelete` 方法：

[PRE7]

与圆形应用程序类似，我们有一组监听器，尽管在这个情况下集合更大。每个监听器都在构造函数中添加到菜单中。与圆形应用程序不同，我们还使用了启用方法：在菜单项变得可见之前被调用的方法。如果方法返回 `false`，菜单项将变为禁用并变灰。如果菜单项连接到加速器，加速器也将变为禁用。我们将 **修改**、**颜色**和**填充**项放在 **修改** 菜单中，将 **线**、**箭头**、**矩形**和**椭圆**项放在 **添加** 菜单中：

[PRE8]

在这个应用程序中，我们还引入了光标控制。`UpdateCursor` 方法根据用户是在创建、修改还是移动图形来设置光标的外观：

[PRE9]

这个应用程序的一个中心点是它的模式：`applicationMode` 方法跟踪用户按下左鼠标按钮时的动作。它保持以下模式：

+   `Idle`: 应用程序等待用户的输入。只要用户没有按下左鼠标按钮，这始终是模式。然而，当用户按下鼠标按钮，直到他们释放它，`applicationMode` 方法保持一个值。用户按下 *Ctrl* 键并点击一个已经标记的图形。图形变为未标记，没有其他操作发生。

+   `ModifySingle`: 用户抓取一个正在修改的单一图形（如果用户点击其端点之一）或移动的图形（如果用户点击图形的任何其他部分）。

+   `ModifyRectangle`: 用户在客户端区域点击而没有击中任何图形，导致绘制了一个矩形。当用户释放鼠标按钮时，矩形完全包围的每个图形都会被标记。

+   `MoveMultiple`: 用户按下 *Ctrl* 键并点击一个未标记的图形。同时修改多个图形是不可能的。

注意，`applicationMode` 方法仅在用户按下左鼠标按钮时相关。一旦他们释放鼠标按钮，`applicationMode` 方法始终是 `Idle`：

[PRE10]

当`applicationMode`方法保持`Idle`模式时，应用程序等待用户进一步的输入。`actionMode`字段定义下一个动作，它可以持有以下值：

+   `Modify`：当用户按下鼠标按钮时，如果他们点击一个图形，则`applicationMode`方法设置为`ModifySingle`模式；如果他们在按下*Ctrl*键的同时点击一个未标记的图形，则设置为`MoveMultiple`模式；如果图形已经被标记，则设置为`Idle`模式；如果他们点击客户端区域而没有击中图形，则设置为`ModifyRectangle`模式。

+   `Add`：当用户按下鼠标左键时，无论该位置是否已有图形，都会在该位置创建一个新的图形。`addFigureId`方法的值决定应该添加哪种类型的图形；它可以持有`LineId`、`ArrowId`、`RectangleId`或`EllipseId`中的任何值。

[PRE11]

在本章的后面部分，我们将遇到**在修改模式**和**在添加模式**之类的表达式，它们指的是`actionMode`变量的值：`Modify`或`Add`。

`nextColor`和`nextFill`字段分别存储下一个图形的颜色和填充状态（在矩形或椭圆的情况下）：

[PRE12]

与圆的应用类似，当用户添加或修改一个图形时，我们需要在`prevMousePoint`方法中存储之前的鼠标位置，以便跟踪鼠标自上次鼠标操作以来移动的距离：

[PRE13]

当`applicationMode`方法保持`ModifySingle`值时，正在修改的图形始终放置在图形指针列表的起始位置（`figurePtrList[0]`），以便它出现在图形之上。当`applicationMode`方法保持`ModifyRectangle`模式时，`insideRectangle`方法跟踪包围图形的矩形：

[PRE14]

`static DrawFormat`常量用于标识要在全局剪贴板中剪切、复制或粘贴的数据。它被任意设置为1000：

[PRE15]

随着用户从绘图添加和删除图形，图形会动态创建和删除；它们的地址存储在`figurePtrList`列表中。`DynamicList`类是一个Small Windows类，它是C++标准类`list`和`vector`的更高级版本。

图形列表的值是指向`DrawFigure`类的指针，这是在本应用中使用的图形层次结构的根类（在[第5章](ch05.html "第5章。图形层次结构")中描述，*图形层次结构*）。与前面章节中的圆和俄罗斯方块应用不同，我们不是直接在列表中存储图形对象，而是它们的指针。这是必要的，因为我们使用具有纯虚方法的类层次结构，这使得`DrawWindow`类成为抽象的，不能直接存储在列表中。这也为了利用类层次结构的动态绑定：

[PRE16]

## 应用程序模式

本节进一步描述了`applicationMode`字段。它与鼠标输入周期紧密相关。当用户没有按下左鼠标按钮时，`applicationMode`方法始终处于`Idle`模式。当用户在修改模式下按下左鼠标按钮时，他们可以选择同时按下**Ctrl**键：

+   如果他们没有按下**Ctrl**键，当点击图形时，`applicationMode`方法会被设置为`ModifySingle`模式。该图形被标记，其他图形变为未标记。

+   如果他们按下**Ctrl**键，当点击一个未标记的图形时，`applicationMode`方法会被设置为`MoveMultiple`模式；如果点击的是一个已标记的图形，则设置为`Idle`模式。图形在被标记时变为未标记，反之亦然。其他图形不受影响。

+   如果他们没有点击图形，无论是否按下**Ctrl**键，`applicationMode`方法都会设置为`ModifyRectangle`模式，并且内部矩形（`insideRectangle`）正在初始化。所有图形都变为未标记。当用户释放左鼠标按钮时，所有完全被矩形包围的图形都会被标记。

当用户在修改模式下按下左鼠标按钮并移动鼠标时，需要考虑`applicationMode`方法的四种可能值：

+   `Idle`：我们不进行任何操作。

+   `ModifySingle`：我们对单个图形调用`Modify`方法。这可能导致用户点击的图形被修改或移动，具体取决于用户点击图形的位置。

+   `MoveMultiple`：我们对所有标记的图形调用`Move`方法。这总是导致标记的图形被移动，而不是被修改。

+   `ModifyRectangle`：我们修改内部矩形。

最后，当用户释放左鼠标按钮时，我们再次查看`applicationMode`方法的四种模式：

+   `Idle`、`ModifySingle`或`MoveMultiple`：由于用户移动鼠标时已经完成了一切，所以我们不做任何操作。标记的图形已经被移动或修改。

+   `ModifyRectangle`：我们标记所有完全被矩形包围的图形。

## 动态列表类

在本章中，我们使用了辅助`DynamicList`类的方法子集。它包含了一组接受回调函数的方法，即作为参数传递给方法并由方法调用的函数：

[PRE17]

`IfFuncPtr`和`DoFuncPtr`是指向回调函数的指针。它们之间的区别在于，`IfFuncPtr`指针旨在用于仅检查列表值的函数。因此，`value`参数是常量。`DoFuncPtr`指针旨在用于修改值的函数。因此，`value`参数不是常量：

[PRE18]

`AnyOf` 方法接受 `ifFuncPtr` 指针并将其应用于数组的每个值。如果至少有一个值满足 `ifFunctPtr` 指针（如果 `ifFuncPtr` 指针对值返回 `true`），则方法返回 `true`。`ifVoidPtr` 参数作为 `ifFuncPtr` 指针的第二个参数发送：

[PRE19]

`FirstOf` 方法如果至少有一个值满足 `ifFuncPtr` 指针，也会返回 `true`。在这种情况下，满足条件的第一个值被复制到 `value` 参数：

[PRE20]

`Apply` 方法调用 `doFunctPtr` 指针到列表中的每个值。`ApplyIf` 方法调用 `doFunctPtr` 指针到所有满足 `ifFuncPtr` 指针的值：

[PRE21]

`CopyIf` 方法将满足 `ifFuncPtr` 指针的值复制到 `copyArray` 方法中。`RemoveIf` 方法移除满足 `ifFuncPtr` 指针的每个值：

[PRE22]

`ApplyRemoveIf` 方法调用 `doFuncPtr` 指针，然后移除满足 `ifFuncPtr` 指针的每个值，这在我们需要从列表中释放和移除指针时非常有用：

[PRE23]

## 初始化

`DrawDocument` 类的构造函数与 `CircleDocument` 类的构造函数类似。我们使用 US 字号大小的 `LogicalWithScroll` 坐标系。文件描述 `Draw Files` 和后缀 `drw` 用于在打开和保存对话框中过滤绘图文件。空指针表示文档没有父窗口，而 `false` 参数表示在 **文件** 菜单中省略了 **打印** 和 **打印预览** 项。最后，包含 `DrawFormat` 参数的初始化列表表示用于标识要复制和粘贴的数据的格式。在这种情况下，我们为复制和粘贴使用相同的格式：

**DrawDocument.cpp**

[PRE24]

由于我们扩展了 `StandardDocument` 框架，窗口具有标准菜单栏，其中 **文件** 菜单包含 **新建**、**打开**、**保存**、**另存为** 和 **退出**（由于构造函数调用中的 `false` 参数，省略了 **打印** 和 **打印预览** 项），**编辑** 菜单包含 **剪切**、**复制**、**粘贴** 和 **删除**，以及 **帮助** 项和 **关于**。

我们还添加了两个特定于应用程序的菜单：**格式** 和 **添加**。**格式** 菜单包含 **修改**、**颜色** 和 **填充** 菜单项。类似于圆形应用程序，我们使用助记符和快捷键标记菜单项。然而，我们还将启用参数；在菜单项可见之前调用 `ModifyEnable`、`ColorEnable` 和 `FillEnable` 方法。如果它们返回 `false`，则菜单项被禁用并变灰：

[PRE25]

**添加** 菜单为要添加的每种图形类型包含一个项：

[PRE26]

最后，我们从**Windows注册表**中读取值，这是Windows系统中我们可以用来在应用程序执行之间存储值的数据库。Small Windows辅助类`Color`、`Font`、`Point`、`Size`和`Rect`都有自己的注册方法。Small Windows的`Registry`类包含用于读取和写入文本以及数值和整数的静态方法：

[PRE27]

析构函数将值写入注册表。在这个应用程序中，不需要提供任何常见的析构函数操作，例如释放内存或关闭文件：

[PRE28]

## 鼠标输入

`IsFigureMarked`、`IsFigureClicked`和`UnmarkFigure`是`DynamicList`方法`AnyOf`、`FirstOf`、`CopyIf`、`ApplyIf`和`ApplyRemoveIf`调用的回调函数。这些方法接受图形的指针和一个可选的void指针，该指针包含附加信息。

`IsFigureMarked`函数如果图形被标记则返回`true`，`IsFigureClicked`函数如果给定的`voidPtr`指针中的鼠标点击图形则返回`true`，如果图形被标记，`IsFigureClicked`函数会取消标记图形。如您所见，`IsFigureMarked`函数被定义为lambda函数，而`IsFigureClicked`函数被定义为常规函数。

这没有合理的理由，除了我想展示定义函数的两种方式：

[PRE29]

在`OnMouseDown`方法中，我们首先检查用户是否按下鼠标左键。如果是这样，我们将鼠标位置保存到`prevMousePoint`字段中，以便我们可以在后续调用`OnMouseMove`方法时计算图形移动的距离：

[PRE30]

如前所述，鼠标点击的结果取决于`actionMode`方法值的差异。在`Modify`方法的情况下，我们在图形指针列表上调用`FirstOf`参数以提取第一个点击的图形。图形可以重叠，点击可能击中多个图形。在这种情况下，我们希望列表开头的最上面的图形。如果至少有一个点击的图形，`FirstOf`方法返回`true`，并将其复制到`topClickedFigurePtr`引用参数中。`mousePoint`方法的地址作为`FirstOf`方法的第二个参数给出，并将其作为第二个参数传递给`IsFigureClicked`函数：

[PRE31]

我们需要考虑两种情况，这取决于用户是否按下*Ctrl*键。如果这样做，如果图形未被标记，则将其标记，反之亦然，并且其他标记的图形将保持标记。

然而，在另一种情况下，当用户没有按下*Ctrl*键时，无论图形是否已经标记，图形都会被标记，所有其他标记的图形都会取消标记，并且应用程序设置为`ModifySingle`模式。图形从列表中移除并插入到列表的开始（前端），以便出现在绘图的最上面：

[PRE32]

如果用户按下 *Ctrl* 键，我们还有另外两种情况。如果点击的图形已经被标记，我们将取消标记它并将 `applicationMode` 方法设置为 `Idle` 模式。如果点击的图形尚未标记，我们将标记它并将 `applicationMode` 方法设置为 `MoveMultiple` 模式。这样，在用户移动鼠标时，`OnMouseMove` 方法中至少有一个标记的图形要移动。请注意，如果用户按下 *Ctrl* 键，一个或多个图形可以移动但不能修改。同时修改多个图形是不合逻辑的：

[PRE33]

如果用户到达一个没有图形的位置（`figurePtrList.FirstOf` 方法返回 `false`），我们将取消所有标记的图形，初始化 `insideRectangle` 方法，并将 `applicationMode` 方法设置为 `ModifyRectangle` 模式。

[PRE34]

在此方法中提到的所有上述情况都发生在 `actionMode` 方法为 `Modify` 时。然而，它也可以是 `Add`，在这种情况下，将在绘图中新添加一个图形。我们使用 `addFigureId` 方法在调用 `CreateFigure` 方法时决定添加哪种类型的图形。我们设置脏标志，因为我们已经添加了一个图形，文档已经被修改。最后，我们将新图形的地址添加到图形列表的开头（这样它就会出现在顶部），并将 `applicationMode` 方法设置为 `ModifySingle` 模式：

[PRE35]

根据操作和模式，窗口和光标可能需要更新：

[PRE36]

`MoveMarkFigure` 方法是一个回调函数，它在 `OnMouseMove` 方法中由 `figurePtrList` 的 `Apply` 方法调用。它移动标记的图形。移动距离的地址在 `voidPtr` 参数中给出：

[PRE37]

在 `OnMouseMove` 方法中，我们首先计算自上次调用 `OnMouseDown` 或 `OnMouseMove` 方法以来的距离。我们还设置 `prevMousePoint` 方法为鼠标位置：

[PRE38]

根据 `applicationMode` 方法的不同，我们执行不同的任务。在单个图形的 `Modify` 方法的情况下，我们在该图形上调用 `MoveOrModify` 方法。由于我们在 `OnMouseDown` 方法中将其放置在那里，该图形位于图形指针列表的开头（`figurePtrList[0]`）。想法是图形本身，根据用户点击的位置，决定它是移动还是修改。图形的状态在用户点击时设置，并取决于他们是否点击了图形的任何端点：

[PRE39]

在多个移动的情况下，我们将每个标记的图形移动到上次鼠标消息的距离。请注意，我们不会像在单个情况下那样在多个情况下修改图形：

[PRE40]

在矩形的情况下，我们设置其右下角并重新绘制它：

[PRE41]

`IsFigureInside`和`MarkFigure`方法是回调函数，在`OnMouseUp`方法中由`DynamicList`方法`CopyIf`、`RemoveIf`和`Apply`在`figurePtrList`上调用。如果图形位于给定的矩形内部，`IsFigureInside`方法返回`true`，而`MarkFigure`方法只是标记图形：

[PRE42]

在`OnMouseUp`方法中，我们只需要考虑`ModifyRectangle`情况。我们需要决定哪些图形完全被矩形包围。为了让它们出现在绘图的最上层，我们首先在`figurePtrList`列表上调用`CopyIf`方法，将完全位于矩形内部的图形临时复制到`insideList`列表中。

然后我们从`figurePtrList`列表中删除图形，并将它们从`insideList`列表中插入到`figurePtrList`列表的开头。这使得它们出现在绘图的最上层。最后，我们通过在`insideList`列表上调用`Apply`来标记矩形内的图形：

[PRE43]

在用户释放左鼠标按钮后，应用程序保持`Idle`模式，只要用户不按下左鼠标按钮，它就会一直保持这种模式：

[PRE44]

当用户双击鼠标按钮时，会调用`OnDoubleClick`方法。双击和连续两次点击之间的区别由Windows系统决定，可以在Windows控制面板中调整。在双击的情况下，在`OnDoubleClick`方法之前会调用`OnMouseDown`和`OnMouseUp`方法。如果有的话，我们提取最顶部的点击图形，并调用`DoubleClick`方法。结果取决于图形的类型：箭头的头部会反转，如果矩形或椭圆未被填充，则填充它们，反之亦然，而直线则不受影响：

[PRE45]

## 绘制

在小窗口中，有三种常见的绘图方法：`OnPaint`、`OnPrint`和`OnDraw`。Windows系统分别间接调用`OnPaint`和`OnPrint`方法来绘制窗口或打印纸张，它们的默认行为是调用`OnDraw`方法。记住，我们不会主动绘制窗口，我们只是等待正确的消息。这个想法是，在需要区分绘制和打印的情况下，我们重写`OnPaint`和`OnPrint`方法，而在不需要这种区分的情况下，我们重写`OnDraw`方法。

在本书稍后讨论的文字处理器中，我们将探讨绘制和打印之间的区别。然而，在这个应用程序中，我们只是重写了`OnDraw`方法。如[第3章](ch03.html "第3章。构建俄罗斯方块应用程序")中所述，*构建俄罗斯方块应用程序*，框架创建了`Graphics`类引用，可以被认为是一个配备了笔刷的工具箱。在这种情况下，我们只需使用`Graphics`引用作为参数调用每个图的`DrawFigure`方法。在`ModifyRectangle`模式下，我们也绘制矩形：

[PRE46]

## 文件菜单

感谢`StandardDocument`类中的框架，文件管理变得相当简单。当用户选择**新建**菜单项时，会调用`ClearDocument`方法，我们只需删除图并清空图列表：

[PRE47]

当用户选择**保存**或**另存为**菜单项时，会调用`WriteDocumentToStream`方法。它首先写入图列表的大小，然后对于每个图，它写入其标识号（这在读取`ReadDocumentFromStream`方法中显示的图时是必要的），然后通过调用其`WriteFigureToStream`方法来写入图本身：

[PRE48]

当用户选择**打开**菜单项时，会调用`ReadDocumentFromStream`方法。它首先读取图列表中的图数量。我们需要读取下一个图的标识号，并调用`CreateFigure`方法以获取创建的图的指针。然后，我们只需调用`ReadFigureFromStream`方法来读取图，并将图的地址添加到图指针列表中：

[PRE49]

`ReadFigureFromStream`和`ReadFigureFromClipboard`方法会调用`CreateFigure`方法，并创建给定类型的图：

[PRE50]

## 剪切、复制和粘贴

与上述文件管理案例类似，框架也负责剪切、复制和粘贴的细节。首先，我们需要决定何时启用剪切和复制菜单项和快捷键。在`Modify`模式下，至少有一个图被标记就足够了。我们使用`DynamicList`方法的`AnyOf`来决定是否至少有一个图被标记。在`Add`模式下，剪切或复制是不允许的。我们不需要重写`CutEnable`方法，因为在`StandardDocument`框架中，它的默认行为是调用`CopyEnable`方法：

[PRE51]

在`StandardDocument`框架中有一个`PasteEnable`方法。然而，在这个应用程序中我们不需要重写它，因为框架决定何时启用粘贴，或者更具体地说，当全局剪贴板上有在`StandardDocument`构造函数中给出的格式代码的数据时，在这种情况下是`DrawFormat`字段。全局剪贴板是一个Windows资源，用于存储已复制信息的短期存储。

`CopyGeneric`方法接受一个字符列表，这些字符打算用应用程序特定的信息填充。我们保存标记图形的数量，并为每个标记的图形，我们写入其身份编号并调用`WriteFigureToClipboard`方法，该方法将图形特定信息写入`infoList`参数：

[PRE52]

`PasteGeneric`方法以类似于前面提到的`ReadDocumentFromStream`方法的方式粘贴图形：

[PRE53]

在`StandardDocument`框架中有一个`DeleteEnable`方法，我们不需要重写它，因为其默认行为是调用`CopyEnable`方法。`OnDelete`方法遍历图形列表，使标记的图形无效并删除它们。我们使用`DynamicList`方法的`ApplyRemoveIf`来删除和删除标记的图形。

我们不能简单地使用`ApplyIf`和`RemoveIf`方法来释放和删除图形，因为这会导致内存错误（悬挂指针）：

[PRE54]

## 修改菜单

**修改**菜单项操作起来非常简单。当应用程序处于`空闲`模式时，它会启用，此时用户没有按下鼠标左键。如果`actionMode`方法设置为`修改`，则也会出现单选按钮，菜单项监听器只需将`actionMode`方法设置为`修改`：

[PRE55]

对于**颜色**和**填充**菜单项，有简单的启用方法，监听器则稍微复杂一些。在`修改`模式下，如果至少有一个图形被标记，则可以更改颜色。在`添加`模式下，始终可以更改颜色：

[PRE56]

`SetFigureColor`方法是一个回调函数，它在`OnColor`方法中由`figurePtrList`列表上的`ApplyIf`方法调用：

[PRE57]

当用户选择**颜色**菜单项时，会调用`OnColor`方法。在`修改`模式下，我们提取标记的图形并选择其最上面的颜色。我们知道至少有一个图形被标记，否则前面的`ColorEnable`方法会返回`false`，**颜色**菜单项将被禁用。如果`ColorDialog`调用返回`true`，我们通过在`figurePtrList`列表上调用`ApplyIf`方法来设置所有标记图形的新颜色：

[PRE58]

如果`actionMode`方法设置为`添加`，我们只需显示一个颜色对话框来设置下一个颜色：

[PRE59]

`IsFigureMarkedAndFilled`方法是一个回调函数，它在`FillCheck`方法中由`figurePtrList`列表上的`AnyOf`方法调用。如果至少有一个图形被标记并填充，则**填充**菜单项会通过单选标记进行检查：

[PRE60]

`IsFigureMarkedAndFillable`方法是一个回调函数，它在`FillEnable`方法中由`figurePtrList`列表上的`AnyOf`方法调用。如果至少有一个可填充的图形（矩形或椭圆）被标记，或者如果用户即将添加矩形或椭圆，则**填充**菜单项将被启用：

[PRE61]

为了测试下一个要添加的图形类型是否可填充，我们创建并删除这样的图形：

[PRE62]

`InverseFill` 方法是一个回调函数，它在用户选择 **填充** 菜单项时，由 `figurePtrList` 列表中的 `OnFill` 方法调用，该方法在 `Modify` 模式下反转所有标记图形的填充状态。在 `Add` 模式下，它仅反转 `nextFill` 的值，表示下一个要添加的图形将具有反转的填充状态：

[PRE63]

## 添加菜单

`Add` 菜单项的监听器相当直接。启用方法很简单，要使菜单项启用，只需 `applicationMode` 方法处于 `Idle` 模式即可：

[PRE64]

在 `Add` 模式下，单选方法返回 `true` 如果要添加的图形与单选方法的图形匹配：

[PRE65]

最后，响应菜单项和快捷键选择的方法将 `actionMode` 设置为 `Add` 并设置要添加的图形：

[PRE66]

## 光标

`Cursor` 类中的 `Set` 方法将光标设置为适当的值。如果应用程序模式是 `Idle` 模式，我们等待用户按下鼠标按钮。在这种情况下，我们使用众所周知的箭头光标图像。如果用户正在用矩形包围图形，我们使用十字准线。如果用户正在移动多个图形，我们使用带有四个箭头的光标（大小全部）。最后，如果他们正在修改单个图形，则该图形（其地址位于 `figurePtrList[0]` 列表中）本身决定使用哪个光标：

[PRE67]

# 概述

在本章中，你开始开发一个能够绘制线条、箭头、矩形和椭圆的绘图程序。在[第5章](ch05.html "第5章。图形层次结构")，*图形层次结构*中，我们将探讨图形层次结构。
