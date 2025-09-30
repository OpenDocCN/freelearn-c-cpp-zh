# 第13章。注册表、剪贴板、标准对话框和打印预览

本章描述了以下内容的实现：

+   **注册表**：一个Windows数据库，用于存储应用程序执行之间的信息。

+   **剪贴板**：一个Windows数据库，用于存储已剪切、复制和粘贴的信息。

+   **标准对话框**：用于保存和打开文档、颜色和字体以及打印。

+   **打印预览**：在`StandardDocument`类中，可以像打印一样在屏幕上查看文档。

# 注册表

`Registry`类中的静态写入、读取和擦除方法在`Integer`、`Double`、`Boolean`和`String`类型的值以及Windows注册表中的内存块上操作。

**Registry.h**：

[PRE0]

**Registry.cpp**：

[PRE1]

全局常量`RegistryFileName`持有Small Windows注册表域的路径：

[PRE2]

`WriteInteger`、`WriteDouble`和`WriteBoolean`函数简单地将值转换为字符串并调用`WriteString`：

[PRE3]

`WriteString`函数调用Win32 API函数`WritePrivateProfileString`，将字符串写入注册表。所有C++ `String`对象都需要通过`c_str`转换为以空字符终止的C字符串（char指针）：

[PRE4]

`WriteBuffer`函数调用Win32 API函数`WritePrivateProfileStruct`，将内存块写入注册表：

[PRE5]

`ReadInteger`、`ReadDouble`和`ReadBoolean`函数将默认值转换为字符串并调用`ReadString`。然后，将`ReadString`的返回值转换并返回；`_tstoi`和`_tstof`是标准C函数`atoi`和`atof`的通用版本：

[PRE6]

`ReadString`函数调用Win32 API函数`GetPrivateProfileString`，将字符串值读取到`text`中并返回读取的字符数。如果读取的字符数大于零，则将文本转换为`string`对象并返回；否则，返回默认文本：

[PRE7]

`ReadBuffer`函数调用Win32 API函数`ReadPrivateProfileStruct`，从注册表中读取内存块。如果它返回零，则表示读取失败，并将默认缓冲区复制到缓冲区：

[PRE8]

当从注册表中删除值时，我们使用空指针而不是字符串调用`WritePrivateProfileString`，从而删除该值：

[PRE9]

# 剪贴板类

`Clipboard`类是对全局Windows剪贴板的接口，这使得在不同类型的应用程序之间剪切、复制和粘贴信息成为可能。剪贴板操作有两种形式：ASCII和Unicode文本以及通用（应用程序特定）信息。

**Clipboard.h**：

[PRE10]

ASCII和Unicode行的格式是预定义的。

[PRE11]

`Open`和`Close`打开和关闭剪贴板。如果成功，它们返回`true`。`Clear`在剪贴板打开时清除剪贴板。更具体地说，它移除任何潜在的信息，并且如果`Available`返回`true`，则表示剪贴板上存储了具有该格式的信息。

不同格式的信息可能存储在剪贴板上。例如，当用户在应用程序中复制文本时，文本可能以ASCII和Unicode文本以及更高级的应用程序特定格式存储在剪贴板上。如果剪贴板上有指定格式的信息，则`Available`返回`true`：

[PRE12]

`WriteText`和`ReadText`函数写入和读取字符串列表，而`WriteGeneric`和`ReadGeneric`函数写入和读取泛型信息：

[PRE13]

**Clipboard.cpp**:

[PRE14]

`Open`、`Close`和`Clear`函数调用Win32 API函数`OpenClipboard`、`CloseClipboard`和`EmptyClipboard`。它们都返回整数值；非零值表示成功：

[PRE15]

`Available`函数通过调用Win32 API函数`FormatAvailable`检查剪贴板上是否有指定格式的数据：

[PRE16]

## ASCII和Unicode行

由于`WriteText`和`ReadText`是模板方法，它们包含在头文件中而不是实现文件中。`WriteText`接受一个泛型字符串列表并将它们以任何格式写入剪贴板；`AsciiFormat`（一个字节/字符）和`UnicodeFormat`（两个字节/字符）是预定义的。

**Clipboard.h**:

[PRE17]

首先，我们需要找到缓冲区大小，我们通过计算行中的字符总数来计算它。我们还要为每一行加一，因为每一行也包含一个终止字符。终止字符是每一行的回车字符（`\r`），除了最后一行，它由一个零字符（`\0`）终止：

[PRE18]

当我们计算出缓冲区大小时，我们可以调用Win32 API的`GlobalAlloc`函数在全局剪贴板上分配缓冲区。我们稍后将将其连接到格式。我们使用模板字符类型的大小作为缓冲区：

[PRE19]

如果分配成功，我们将收到缓冲区的句柄。由于剪贴板及其缓冲区可以同时被多个进程使用，我们需要通过调用Win32 API函数`GlobalLock`来锁定缓冲区。只要缓冲区被锁定，其他进程就无法访问它。当我们锁定缓冲区时，我们收到一个指向它的指针，我们可以用它来向缓冲区写入信息：

[PRE20]

我们将行的字符写入缓冲区，除非它是列表中的最后一行，否则我们添加一个`return`字符：

[PRE21]

我们在缓冲区的末尾添加一个零字符来标记其结束：

[PRE22]

当缓冲区已加载信息后，我们只需解锁缓冲区，以便其他进程可以访问它并将缓冲区与格式关联：

[PRE23]

最后，我们返回`true`以指示操作成功：

[PRE24]

如果我们没有能够为写入行列表分配缓冲区，我们通过返回`false`来指示操作未成功：

[PRE25]

当使用`ReadText`读取行列表时，我们使用`Format`（通常是`AsciiFormat`或`UnicodeFormat`）从剪贴板接收一个句柄，然后我们使用它来锁定缓冲区并接收其指针，这反过来又允许我们从缓冲区中读取：

[PRE26]

注意，我们必须将缓冲区大小除以模板字符类型大小（可能大于1），以找到字符数：

[PRE27]

当我们遇到回车字符（`\r`）时，当前行结束；我们将它添加到行列表中，然后清除它以便为下一行做好准备：

[PRE28]

当我们遇到回车字符（`'\0'`）时，我们也把当前行添加到行列表中。然而，没有必要清除当前行，因为零字符是缓冲区的最后一个字符：

[PRE29]

如果字符既不是回车也不是零字符，我们就将它添加到当前行。注意，我们读取一个`CharType`类型的字符并将其转换为`TCHAR`类型的通用字符：

[PRE30]

最后，我们解锁缓冲区并返回`true`以指示操作成功：

[PRE31]

如果我们没有收到格式的缓冲区，我们返回`false`以指示操作未成功：

[PRE32]

## 通用信息

`WriteGeneric`函数实际上比前面的`WriteText`函数简单，因为它不需要考虑行列表。我们只需锁定剪贴板缓冲区，将`infoList`中的每个字节写入缓冲区，解锁缓冲区，并将其与格式关联：

**Clipboard.cpp**:

[PRE33]

`InfoList`函数中的`ToBuffer`对象将其字节写入缓冲区：

[PRE34]

如果我们没有成功分配全局缓冲区，我们返回`false`以指示操作未成功：

[PRE35]

`ReadGeneric`函数锁定剪贴板缓冲区，将缓冲区中的每个字节写入`infoList`，解锁缓冲区，并返回`true`以指示操作成功：

[PRE36]

如果我们没有收到全局句柄，我们返回`false`以指示操作未成功：

[PRE37]

# 标准对话框

在Windows中，可以定义**对话框**。与窗口不同，对话框的目的是填充控件，如按钮、框和文本字段。一个对话框可能是**模态的**，这意味着在对话框关闭之前，应用程序的其他窗口将变为禁用状态。在下一章中，我们将探讨如何构建我们自己的对话框。

然而，在本节中，我们将探讨Windows**标准**对话框，用于保存和打开文件、选择字体和颜色以及打印。Small Windows通过包装Win32 API函数支持标准对话框，这些函数为我们提供了对话框。

## 保存对话框

`SaveDialog`函数显示标准**保存**对话框。

![保存对话框](img/B05475_13_01.jpg)

`filter` 参数过滤要显示的文件类型。每个文件格式由两部分定义：对话框中显示的文本和默认文件后缀。这两部分由一个零字符分隔，并且过滤器以两个零字符结束。例如，考虑以下：

[PRE38]

`fileSuffixList` 参数指定允许的文件后缀，而 `saveFlags` 包含操作的标志。以下有两个标志可用：

+   `PromptBeforeOverwrite`: 这个标志是一个警告信息，如果文件已经存在，则会显示

+   `PathMustExist`: 如果路径不存在，则会显示一个错误信息

**StandardDialog.h**:

[PRE39]

**StandardDialog.cpp**:

[PRE40]

Win32 API `OPENFILENAME` 结构的 `saveFileName` 被加载了适当的值：`hwndOwner` 设置为窗口句柄，`hInstance` 设置为应用程序实例句柄，`lpstrFilter` 设置为 `filter` 参数，`lpstrFile` 设置为 `pathBuffer`，它反过来又包含 `path` 参数，并且 `Flags` 设置为 `saveFlags` 参数：

[PRE41]

当 `saveFileName` 被加载时，我们调用 Win32 API 函数 `GetSaveFileName`，它显示标准的 **保存** 对话框，如果用户通过点击 **保存** 按钮或按 **回车** 键终止对话框，则返回非零值。在这种情况下，我们将 `path` 参数设置为所选路径，检查路径是否以 `fileSuffixList` 中的后缀之一结尾，如果是以，则返回 `true`。如果路径后缀不在列表中，我们显示一个错误信息，并重新开始保存过程。如果用户取消过程，则返回 `false`。实际上，用户完成过程的唯一方法是选择列表中的文件后缀或取消对话框：

[PRE42]

## 打开对话框

`OpenDialog` 函数显示标准的 **打开** 对话框。

![打开对话框](img/B05475_13_02.jpg)

`filter` 和 `fileSuffixList` 参数与前面的 `SaveDialog` 函数中的方式相同。有三个标志可用：

+   `PromptBeforeCreate`: 如果文件已经存在，则此标志会显示一个警告信息

+   `FileMustExist`: 打开的文件必须存在

+   `HideReadOnly`: 此标志表示在对话框中隐藏只读文件

**OpenDialog.h**:

[PRE43]

`OpenDialog` 的实现与前面的 `SaveDialog` 函数类似。我们使用相同的 `OPENFILENAME` 结构；唯一的区别是我们调用 `GetOpenFileName` 而不是 `GetSaveFileName`。

**OpenDialog.cpp**:

[PRE44]

## 颜色对话框

`ColorDialog` 函数显示标准的 **颜色** 对话框。

![颜色对话框](img/B05475_13_03.jpg)

**StandardDialog.h**:

[PRE45]

静态 `COLORREF` 数组 `customColorArray` 被用户在颜色对话框中使用，以存储所选颜色。由于它是静态的，`customColorArray` 数组在对话框显示会话之间被重用。

`ColorDialog` 函数使用 Win32 API `CHOOSECOLOR` 结构初始化对话框。`hwndOwner` 函数设置为窗口句柄，`rgbResult` 设置为颜色的 `COLORREF` 字段，`lpCustColors` 设置为自定义颜色数组。`CC_RGBINIT` 和 `CC_FULLOPEN` 标志使用给定的颜色初始化对话框，使其完全展开。

**StandardDialog.cpp**:

[PRE46]

Win32 的 `ChooseColor` 函数显示 **颜色** 对话框，如果用户通过点击 **确定** 按钮结束对话框，则返回非零值。在这种情况下，我们设置所选颜色并返回 `true`：

[PRE47]

如果用户取消对话框，我们返回 `false`：

[PRE48]

## 字体对话框

`FontDialog` 函数显示一个标准的 **字体** 对话框。

![字体对话框](img/B05475_13_04.jpg)

**StandardDialog.h**:

[PRE49]

**FontDialog.cpp**:

[PRE50]

Win32 API `CHOOSEFONT` 结构 `chooseFont` 被加载了适当的值。`lpLogFont` 对象设置为字体的 `LOGFONT` 字段，`rgbColors` 设置为颜色的 `COLORREF` 字段：

[PRE51]

Win32 的 `ChooseFont` 函数显示 **字体** 对话框，如果用户点击 **确定** 按钮则返回非零值。在这种情况下，我们设置所选字体和颜色并返回 `true`：

[PRE52]

如果用户取消对话框，我们返回 `false`：

[PRE53]

## 打印对话框

`PrintDialog` 函数显示一个标准的 **打印** 对话框。

![打印对话框](img/B05475_13_05.jpg)

如果用户点击 **打印** 按钮，所选的打印设置将保存在 `PrintDialog` 参数中：

**PrintDialog.h**:

[PRE54]

`PrintDialog` 函数使用适当的值加载 Win32 API `PRINTDLG` 结构 `printDialog`，`nFromPage` 和 `nToPage` 设置为要打印的第一页和最后一页（默认值分别为 1 和页数），`nMaxPage` 设置为页数，`nCopies` 设置为 1（默认值）。

**PrintDialog.cpp**:

[PRE55]

Win32 API 函数 `PrintDlg` 显示标准打印对话框，如果用户通过按下 **打印** 按钮结束对话框，则返回非零值。在这种情况下，打印的第一页和最后一页、副本数量以及是否排序存储在参数中，并创建返回用于打印的 `Graphics` 对象的指针。

如果用户选择了页面间隔，我们使用 `nFromPage` 和 `nToPage` 字段；否则，选择所有页面，并使用 `nMinPage` 和 `nMaxPage` 字段设置要打印的第一页和最后一页：

[PRE56]

如果存在 `PD_COLLATE` 标志，则用户选择了排序页面：

[PRE57]

最后，我们创建并返回一个指向用于打印时绘图的 `Graphics` 对象的指针。

[PRE58]

如果用户通过按下 **取消** 按钮结束对话框，我们返回 null：

[PRE59]

# 打印预览

`PrintPreviewDocument`类显示文档父窗口的页面。`OnKeyDown`方法在用户按下***Esc***键时关闭文档。`OnSize`方法调整页面的物理大小，以确保页面始终适合窗口。`OnVerticalScroll`方法在用户向上或向下滚动时移动页面，而`OnPaint`为每一页调用父文档的`OnPrint`：

**PrintPreviewDocument.h**：

[PRE60]

仅覆盖`OnSize`函数以在`Document`中中和其功能。在`Document`中，`OnSize`修改滚动条，但我们不希望在类中发生这种情况：

[PRE61]

`page`字段存储当前页码，`totalPages`存储总页数：

[PRE62]

**PrintPreviewDocument.cpp**

[PRE63]

构造函数将`page`和`totalPages`字段设置为适当的值。

[PRE64]

水平滚动条始终设置为窗口的宽度，这意味着用户无法更改其设置：

[PRE65]

垂直滚动条设置为与文档的页数相匹配，滚动滑块对应一页：

[PRE66]

标题显示当前页数和总页数：

[PRE67]

## 键盘输入

当用户按下键时，会调用`OnKeyDown`函数。如果他们按下***Esc***键，预览窗口将被关闭并销毁，输入焦点将返回到应用程序的主窗口。如果他们按下***Home***、***End***、***Page Up***、***Page Down***键或上下箭头键，将调用`OnVerticalScroll`以执行适当的操作：

[PRE68]

我们返回`true`以指示已使用键盘输入：

[PRE69]

## 滚动条

当用户滚动垂直条时，会调用`OnVerticalScroll`函数。如果他们点击滚动条本身，在滚动滑块上方或下方，将显示上一页或下一页。如果他们将滑块拖动到新位置，将计算相应的页面。包括`SB_TOP`和`SB_BOTTOM`情况是为了适应前面`OnKeyDown`函数中的***Home***和***End***键，而不是为了适应任何滚动操作；它们将页面设置为第一页或最后一页：

[PRE70]

如果滚动操作导致出现新页面，我们将设置标题和滚动条位置，并使窗口无效并更新：

[PRE71]

`PrintPreviewDocument`中的`OnPaint`函数调用父标准文档窗口中的`OnPaint`以绘制预览窗口的内容：

[PRE72]

# 摘要

在本章中，我们探讨了注册表、剪贴板、标准对话框和打印预览。在[第14章](ch14.html "第14章。对话框、控件和页面设置")中，我们将探讨自定义对话框、控件、转换器和页面设置。
