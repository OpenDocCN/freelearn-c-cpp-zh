# 第七章. 键盘输入和字符计算

在本章中，我们将继续在第六章*构建一个文字处理器*的基础上对文字处理器进行工作。更具体地说，我们将探讨键盘输入和字符计算。键盘处理部分处理常规字符输入和一组相当大的特殊键，例如*Home*、*End*、*Page Up*和*Page Down*、*Return*、*Backspace*和箭头键。

计算部分处理每个字符的计算，包括其字体、段落对齐以及页面设置。最后，我们将计算文档中每个单独字符的位置和大小。

# 键盘处理

首先，我们来看一下常规字符的输入。每当用户按下图形字符（ASCII 值在 32 到 127 之间，包括 127）或*回车*键时，都会调用`OnChar`方法。如果文本的一部分被标记，那么这部分首先会被移除。然后，根据`keyboard`模式，通过`OverwriteChar`类的`InsertChar`方法将字符添加到字符列表中。

```cpp
void WordDocument::OnChar(TCHAR tChar) { 
  if (isprint(tChar) || (tChar == NewLine)) { 
    if (wordMode == WordMark) { 
      OnDelete(); 
    } 

    Paragraph* paragraphPtr = charList[editIndex].ParagraphPtr(); 

    switch (GetKeyboardMode()) { 
      case InsertKeyboard: 
        OnInsertChar(tChar, paragraphPtr); 
        break; 

      case OverwriteKeyboard: 
        OnOverwriteChar(tChar, paragraphPtr); 
        break; 
    } 

    SetDirty(true); 
    GenerateParagraph(paragraphPtr); 
    CalculateDocument(); 

    if (MakeVisible()) { 
      Invalidate(); 
      UpdateWindow(); 
    } 

    UpdateCaret(); 
  } 
} 

```

当插入字符时，我们有三种情况，这与第六章中`UpdateCaret`和`OnFont`方法的处理类似，即*构建一个文字处理器*。如果`nextFont`参数是激活的（如果不等于`SystemFont`），我们则用它来处理新字符。然后，通过`ClearNextFont`方法清除`nextFont`参数。

```cpp
void WordDocument::OnInsertChar(TCHAR tChar, 
                                Paragraph* paragraphPtr) { 
  if (nextFont != SystemFont) { 
    charList.Insert(editIndex++, 
                    CharInfo(paragraphPtr, tChar, nextFont)); 
    ClearNextFont(); 
  } 

```

如果`nextFont`参数未激活且输入不在段落的开始处，我们则使用前一个字符的字体来处理新字符。

```cpp
  else if (charList[editIndex].ParagraphPtr()->First() < 
           editIndex) { 
    Font font = charList[editIndex - 1].CharFont(); 
    charList.Insert(editIndex++, 
                    CharInfo(paragraphPtr, tChar, font)); 
  } 

```

然而，如果输入位于段落的开始处，我们则使用段落中第一个字符的字体。

```cpp
  else { 
    Font font = charList[editIndex].CharFont(); 
    charList.Insert(editIndex++, 
                    CharInfo(paragraphPtr, tChar, font)); 
  } 

```

为了为插入的字符腾出空间，我们增加其段落的最后一个索引。同时，我们也增加后续段落的第一个和最后一个索引。

```cpp
  ++paragraphPtr->Last(); 

  for (int parIndex = paragraphPtr->Index() + 1; 
       parIndex <= paragraphList.Size() - 1; ++parIndex) { 
    ++paragraphList[parIndex]->First(); 
    ++paragraphList[parIndex]->Last(); 
  } 
} 

```

在`overwrite`模式下，我们有两种情况。如果输入位于文档的末尾，我们则插入字符而不是覆盖它；否则，我们覆盖最后一个段落的换行符。然而，我们可以自由地覆盖除最后一个段落外的每个段落的终止换行符，在这种情况下，两个段落将合并为一个。

与`InsertChar`方法类似，如果`nextFont`参数不等于`SystemFont`参数，我们则使用它。如果它等于`SystemFont`参数，我们则使用被覆盖字符的字体，而不是像在`InsertChar`情况中那样使用前一个字符的字体。

```cpp
void WordDocument::OnOverwriteChar(TCHAR tChar, 
                                   Paragraph* paragraphPtr) { 
  if (editIndex == (charList.Size() - 1)) { 
    if (nextFont != SystemFont) { 
      charList.Insert(editIndex++, 
        CharInfo(paragraphPtr, tChar, nextFont)); 
      charList[editIndex] = 
        CharInfo(paragraphPtr, NewLine, nextFont); 
      ClearNextFont(); 
    } 
    else { 
      Font font = charList[editIndex].CharFont(); 
      charList.Insert(editIndex++, 
                      CharInfo(paragraphPtr, tChar, font)); 
    } 

    ++paragraphPtr->Last(); 
  } 
  else { 
    if (nextFont != SystemFont) { 
      charList[editIndex++] = 
        CharInfo(paragraphPtr, tChar, nextFont); 
      ClearNextFont(); 
    } 
    else { 
      Font font = charList[editIndex].CharFont(); 
      charList[editIndex++] = CharInfo(paragraphPtr, tChar, font); 
    } 
  } 
} 

```

`ClearNextFont` 方法通过将其值设置为 `SystemFont` 字体来清除 `nextFont` 参数。它还会重新计算编辑段落和文档，因为移除 `nextFont` 参数可能会导致编辑行（以及因此的编辑段落）降低。行上的字符字体可能都低于 `nextFont` 参数，这会导致移除 `nextFont` 参数后行降低。

```cpp
void WordDocument::ClearNextFont() { 
  if (nextFont != SystemFont) { 
    nextFont = SystemFont; 
    Paragraph* paragraphPtr = charList[editIndex].ParagraphPtr(); 
    GenerateParagraph(paragraphPtr); 
    CalculateDocument(); 
    UpdateWindow(); 
  } 
} 

```

每次用户按下键时，都会调用 `OnKeyDown` 方法。根据键和是否按下 *Shift* 键，`OnKeyDown` 方法会依次调用 `OnShiftKey`、`OnRegularKey` 或 `OnNeutralKey` 方法。*Delete*、*Backspace* 和 *Return* 键在是否按下 *Shift* 键的情况下执行相同的行为。

```cpp
bool WordDocument::OnKeyDown(WORD key, bool shiftPressed, 
                             bool /* controlPressed */) { 
  switch (key) { 
    case KeyLeft: 
    case KeyRight: 
    case KeyUp: 
    case KeyDown: 
    case KeyHome: 
    case KeyEnd: { 

        if (shiftPressed) { 
          OnShiftKey(key); 
        } 
        else { 
          OnRegularKey(key); 
        } 
      } 
      return true; 

    case KeyBackspace: 
    case KeyReturn: 
      OnNeutralKey(key); 
      return true; 
  } 

  return false; 
} 

```

当用户按下图形键时，应用程序将被设置为 `edit` 模式。`EnsureEditStatus` 方法确保这一点。按键可能将光标移动到客户端区域可见部分之外的位置。因此，如果需要，我们调用 `MakeVisible` 方法来移动滚动条，以便光标出现在客户端区域的可见部分。想法是使光标和编辑字符始终在窗口中可见。

```cpp
void WordDocument::OnRegularKey(WORD key) { 
  EnsureEditStatus(); 

  switch (key) { 
    case KeyLeft: 
      OnLeftArrowKey(); 
      break; 

    case KeyRight: 
      OnRightArrowKey(); 
      break; 

    case KeyUp: 
      OnUpArrowKey(); 
      break; 

    case KeyDown: 
      OnDownArrowKey(); 
      break; 

    case KeyHome: 
      OnHomeKey(); 
      break; 

    case KeyEnd: 
      OnEndKey(); 
      break; 
  } 

  if (MakeVisible()) { 
    Invalidate(); 
    UpdateWindow(); 
    UpdateCaret(); 
  } 
} 

```

当用户按下 *Page Up*、*Page Down* 或箭头键之一，而没有按下 *Shift* 键时，我们必须确保应用程序设置为 `edit` 模式。`EnsureEditStatus` 方法负责这一点。`editIndex` 被设置为 `lastMarkIndex`。

```cpp
void WordDocument::EnsureEditStatus() { 
  if (wordMode == WordMark) { 
    wordMode = WordEdit; 
    editIndex = lastMarkIndex; 
    InvalidateBlock(firstMarkIndex, lastMarkIndex); 
    UpdateCaret(); 
    UpdateWindow(); 
  } 
} 

```

## 箭头键

当用户按下左箭头键时，会调用 `OnLeftArrowKey` 方法。它的目的是将光标向左移动一步，这很简单。我们必须确保编辑位置不在文档的开始处。如果我们向左移动位置，我们还需要清除 `nextFont` 参数，因为它只有在用户即将输入新字符时才会激活。

```cpp
void WordDocument::OnLeftArrowKey() { 
  if (editIndex > 0) { 
    ClearNextFont(); 
    --editIndex; 
  } 
} 

```

当用户按下右箭头键时，会调用 `OnRightArrowKey` 方法。如果光标位置不在文档末尾，我们将它向右移动一步。

```cpp
void WordDocument::OnRightArrowKey() { 
  if (editIndex < (charList.Size() - 1)) { 
    ClearNextFont(); 
    ++editIndex; 
  } 
} 

```

当用户按下上箭头键时，我们必须找到编辑行上面的键。我们通过在行稍上方（一个逻辑单位）模拟鼠标点击来实现这一点。请注意，我们必须查找编辑行。仅使用字符矩形是不够的，因为字符的高度和上升（参考下一节）可能不同，我们无法确定字符矩形是该行上最高的矩形。因此，我们查找编辑行的高度。在下面的屏幕截图中，文本被矩形包围以供说明。代码实际上并没有绘制矩形。如果我们使用数字四的矩形，我们就不会达到前面的行，因为数字 **5** 的矩形更高。相反，我们必须使用行 **456** 的行矩形。

![箭头键](img/B05475_07_01.jpg)

```cpp
void WordDocument::OnUpArrowKey() { 
  CharInfo charInfo = charList[editIndex]; 

  Paragraph* paragraphPtr = charInfo.ParagraphPtr(); 
  Point topLeft(0, paragraphPtr->Top()); 

  LineInfo* lineInfoPtr = charInfo.LineInfoPtr(); 
  Rect lineRect = 
    topLeft + Rect(0, lineInfoPtr->Top(), PageInnerWidth(), 
                      lineInfoPtr->Top() + lineInfoPtr->Height()); 

```

我们需要检查编辑字符是否不在文档的第一行。如果编辑字符已经在第一行，则输出不会发生任何变化。

```cpp
  if (lineRect.Top() > 0) { 
    ClearNextFont(); 
    Rect charRect = topLeft + charInfo.CharRect(); 
    editIndex = 
      MousePointToIndex(Point(charRect.Left(), lineRect.Top()-1)); 
  } 
} 

```

当用户按下向下箭头键时，我们通过调用`MousePointToIndexDown`方法来模拟鼠标点击。在调用中，我们使用位于编辑行稍下方的位置（1 个单位）来找到下一行相同水平位置上的字符索引。与前面的`UpArrowKey`情况相比，我们调用`MousePointToIndexDown`方法而不是`MousePointToIndex`方法，因为这可能是在段落的最后一行，并且可能在下一个段落之前有一些空间。在这种情况下，我们希望得到空格后面的字符的索引，这是`MousePointToIndexDown`方法返回的，而`MousePointToIndex`方法返回的是空格前面的字符的索引。

```cpp
void WordDocument::OnDownArrowKey() { 
  CharInfo charInfo = charList[editIndex];  
  Paragraph* paragraphPtr = charInfo.ParagraphPtr(); 
  Point topLeft(0, paragraphPtr->Top());  
  LineInfo* lineInfoPtr = charInfo.LineInfoPtr(); 
  Rect lineRect = 
    topLeft + Rect(0, lineInfoPtr->Top(), PageInnerWidth(), 
                      lineInfoPtr->Top() + lineInfoPtr->Height()); 

```

与前面的`OnUpArrowKey`情况类似，我们需要确保编辑行不是文档中的最后一行。我们通过将其与最后一个段落的底部进行比较来实现这一点。如果是最后一行，则输出不会发生任何变化。

```cpp
  Paragraph* lastParagraphPtr = paragraphList.Back(); 
  int bottom = lastParagraphPtr->Top() + 
               lastParagraphPtr->Height(); 

  if (lineRect.Bottom() < bottom) { 
    ClearNextFont(); 
    Rect charRect = topLeft + charInfo.CharRect(); 
    editIndex = 
      MousePointToIndexDown(Point(charRect.Left(), 
                                  lineRect.Bottom() + 1)); 
  } 
} 

```

`MousePointToIndexDown`方法返回被点击的字符的索引。如果鼠标点在两个段落之间，则返回前一个字符的索引。

```cpp
int WordDocument::MousePointToIndexDown(Point mousePoint) const{ 
  for (int parIndex = 0; parIndex < paragraphList.Size(); 
       ++parIndex) { 
    Paragraph* paragraphPtr = paragraphList[parIndex]; 

    if (mousePoint.Y() <= 
        (paragraphPtr->Top() + paragraphPtr->Height())) { 
      return MousePointToParagraphIndex 
             (paragraphList[parIndex], mousePoint); 
    } 
  } 

```

由于此方法始终找到正确的段落，因此此点永远不会达到，但我们断言在编码错误的情况下，其行为可能会有所不同。

```cpp
  assert(false); 
  return 0; 
} 

```

`OnPageUp`和`OnPageDown`方法查找当前垂直滚动条的高度，以便模拟向上或向下翻一页的鼠标点击。

```cpp
void WordDocument::OnPageUpKey() { 
  CharInfo charInfo = charList[editIndex]; 
  Rect editRect = charInfo.CharRect(); 

  Paragraph* paragraphPtr = charInfo.ParagraphPtr(); 
  Point topLeft(0, paragraphPtr->Top()); 

  int scrollPage = GetVerticalScrollPageHeight(); 
  Point editPoint((editRect.Left() + editRect.Right()) / 2, 
        ((editRect.Top() + editRect.Bottom()) / 2) - scrollPage); 

  editIndex = MousePointToIndex(topLeft + editPoint); 
} 

void WordDocument::OnPageDownKey() { 
  CharInfo charInfo = charList[editIndex]; 
  Rect editRect = charInfo.CharRect(); 

  Paragraph* paragraphPtr = charInfo.ParagraphPtr(); 
  Point topLeft(0, paragraphPtr->Top()); 

  int scrollPage = GetVerticalScrollPageHeight(); 
  Point editPoint((editRect.Left() + editRect.Right()) / 2, 
        ((editRect.Top() + editRect.Bottom()) / 2) + scrollPage); 

  editIndex = MousePointToIndex(topLeft + editPoint); 
} 

```

## Home 和 End

当用户按下*Home*键时调用`OnHomeKey`方法。它通过跟随其段落和行指针来查找编辑行上第一个字符的索引。它使用行中第一个字符的索引。

```cpp
void WordDocument::OnHomeKey() { 
  CharInfo charInfo = charList[editIndex]; 
  int homeCharIndex = charInfo.ParagraphPtr()->First() + 
                      charInfo.LineInfoPtr()->First(); 

```

如果编辑字符尚未位于行的开头，`ClearNextFont`方法会清除`nextFont`参数，更新编辑索引，并更新光标。

```cpp
  if (homeCharIndex < editIndex) { 
    ClearNextFont(); 
    editIndex = homeCharIndex; 
    UpdateCaret(); 
  } 
} 

```

当用户按下*End*键时调用`OnEndKey`方法。它通过跟随其段落和行指针并使用行中最后一个字符的索引来查找编辑行上最后一个字符的索引。

```cpp
void WordDocument::OnEndKey() { 
  CharInfo charInfo = charList[editIndex]; 
  int endCharIndex = charInfo.ParagraphPtr()->First() + 
                     charInfo.LineInfoPtr()->Last(); 

```

如果编辑字符尚未位于行的末尾，`ClearNextFont`方法会清除`nextFont`参数，更新编辑索引，并更新光标。

```cpp
  if (editIndex < endCharIndex) { 
    ClearNextFont(); 
    editIndex = endCharIndex; 
    UpdateCaret(); 
  } 
} 

```

## Shift 箭头键

当用户同时按下*Shift*键和某个键时调用`OnShiftKey`方法：

```cpp
void WordDocument::OnShiftKey(WORD key) { 
  EnsureMarkStatus(); 
  switch (key) { 
    case KeyLeft: 
      OnShiftLeftArrowKey(); 
      break; 

    case KeyRight: 
      OnShiftRightArrowKey(); 
      break; 

    case KeyUp: 
      OnShiftUpArrowKey(); 
      break; 

    case KeyDown: 
      OnShiftDownArrowKey(); 
      break; 

    case KeyPageUp: 
      OnShiftPageUpKey(); 
      break; 

    case KeyPageDown: 
      OnShiftPageDownKey(); 
      break; 

    case KeyHome: 
      OnShiftHomeKey(); 
      break; 

    case KeyEnd: 
      OnShiftEndKey(); 
      break; 
  } 

  if (MakeVisible()) { 
    Invalidate(); 
    UpdateWindow(); 
    UpdateCaret(); 
  } 
} 

```

如果用户同时按下*Shift*键和某个键，我们必须确保应用程序设置为`mark`模式；`EnsureMarkMode`方法处理这个问题。它会清除`nextFont`参数（通过将其设置为`SystemFont`），将应用程序设置为`mark`模式，并将第一个和最后一个标记的索引分配给编辑索引。

```cpp
void WordDocument::EnsureMarkStatus() { 
  if (wordMode == WordEdit) { 
    ClearNextFont(); 
    wordMode = WordMark; 
    firstMarkIndex = editIndex; 
    lastMarkIndex = editIndex; 
    UpdateCaret(); 
  } 
} 

```

`OnShiftLeftArrowKey` 方法减少最后一个标记索引。请注意，我们只使 `lastMarkIndex` 方法的旧值和新值之间的索引无效，以避免闪烁：

```cpp
void WordDocument::OnShiftLeftArrowKey() { 
  if (lastMarkIndex > 0) { 
    InvalidateBlock(lastMarkIndex, --lastMarkIndex); 
  } 
} 

```

`OnShiftRightArrowKey` 方法以类似于 `OnShiftLeftArrowKey` 方式移动最后标记字符的位置。

```cpp
void WordDocument::OnShiftRightArrowKey() { 
  if (lastMarkIndex < charList.Size()) { 
    InvalidateBlock(lastMarkIndex, lastMarkIndex++); 
  } 
} 

```

当用户同时按下上箭头键或下箭头键以及 *Shift* 键时，会调用 `OnShiftUpArrowKey` 和 `OnShiftDownArrowKey` 方法。其任务是向上移动最后一个标记位置。我们以与之前 `OnUpArrowKey` 和 `OnDownArrowKey` 方法相同的方式模拟鼠标点击。

```cpp
void WordDocument::OnShiftUpArrowKey() { 
  CharInfo charInfo = charList[lastMarkIndex]; 

  Paragraph* paragraphPtr = charInfo.ParagraphPtr(); 
  Point topLeft(0, paragraphPtr->Top()); 

  LineInfo* lineInfoPtr = charInfo.LineInfoPtr(); 

  Rect lineRect = 
    topLeft + Rect(0, lineInfoPtr->Top(), PageInnerWidth(), 
                      lineInfoPtr->Top() + lineInfoPtr->Height()); 

  if ((paragraphPtr->Top() + lineRect.Top()) > 0) { 
    Rect charRect = topLeft + charInfo.CharRect(); 
    int newLastMarkIndex = 
      MousePointToIndex(Point(charRect.Left(), lineRect.Top()-1)); 
    InvalidateBlock(lastMarkIndex, newLastMarkIndex); 
    lastMarkIndex = newLastMarkIndex; 
  } 
} 

void WordDocument::OnShiftDownArrowKey() { 
  CharInfo charInfo = charList[lastMarkIndex]; 

  Paragraph* paragraphPtr = charInfo.ParagraphPtr(); 
  Point topLeft(0, paragraphPtr->Top()); 

  LineInfo* lineInfoPtr = charInfo.LineInfoPtr(); 
  Rect lineRect = 
    topLeft + Rect(0, lineInfoPtr->Top(), PageInnerWidth(), 
                      lineInfoPtr->Top() + lineInfoPtr->Height()); 

  Paragraph* lastParagraphPtr = paragraphList.Back(); 
  int bottom = lastParagraphPtr->Top() + 
               lastParagraphPtr->Height(); 

  if (lineRect.Bottom() < bottom) { 
    Rect charRect = topLeft + charInfo.CharRect(); 
    int newLastMarkIndex = 
      MousePointToIndexDown(Point(charRect.Left(), 
                                  lineRect.Bottom() + 1)); 
    InvalidateBlock(lastMarkIndex, newLastMarkIndex); 
    lastMarkIndex = newLastMarkIndex; 
  } 
} 

```

## Shift Page Up 和 Page Down

`OnShiftPageUpKey` 和 `OnShiftPageDown` 方法通过模拟在 *Page Up* 或 *Page Down* 上进行鼠标点击来移动编辑字符索引一个页面高度：

```cpp
void WordDocument::OnShiftPageUpKey() { 
  Rect lastRectMark = charList[lastMarkIndex].CharRect(); 
  int scrollPage = GetVerticalScrollPageHeight(); 
  Point lastPointMark 
    ((lastRectMark.Left() + lastRectMark.Right()) / 2, 
     (lastRectMark.Top()+lastRectMark.Bottom()) / 2 - scrollPage); 

  int newLastMarkIndex = MousePointToIndex(lastPointMark); 
  InvalidateBlock(lastMarkIndex, newLastMarkIndex); 
  lastMarkIndex = newLastMarkIndex; 
} 

void WordDocument::OnShiftPageDownKey() { 
  Rect lastRectMark = charList[lastMarkIndex].CharRect(); 

  int scrollPage = GetVerticalScrollPageHeight(); 
  Point lastPointMark 
    ((lastRectMark.Left() + lastRectMark.Right()) / 2, 
     (lastRectMark.Top()+lastRectMark.Bottom())/2 + scrollPage); 

  int newLastMarkIndex = MousePointToIndexDown(lastPointMark); 
  InvalidateBlock(lastMarkIndex, newLastMarkIndex); 
  lastMarkIndex = newLastMarkIndex; 
} 

```

## Shift Home 和 End

当用户同时按下 *Home* 或 *End* 键以及 *Shift* 键时，会调用 `OnShiftHomeKey` 和 `OnShiftEndKey` 方法。它们的作用是标记从当前位置到行首或行尾的整行：

```cpp
void WordDocument::OnShiftHomeKey() { 
  CharInfo charInfo = charList[editIndex]; 
  int homeCharIndex = charInfo.ParagraphPtr()->First() + 
                      charInfo.LineInfoPtr()->First(); 

  if (homeCharIndex < lastMarkIndex) { 
    InvalidateBlock(lastMarkIndex, homeCharIndex); 
    lastMarkIndex = homeCharIndex; 
  } 
} 

void WordDocument::OnShiftEndKey() { 
  CharInfo charInfo = charList[editIndex]; 
  int endCharIndex = charInfo.ParagraphPtr()->First() + 
                     charInfo.LineInfoPtr()->Last(); 

  if (lastMarkIndex < endCharIndex) { 
    InvalidateBlock(lastMarkIndex, endCharIndex); 
    lastMarkIndex = endCharIndex; 
  } 
} 

```

## Control Home 和 End

`OnControlHomeKey` 和 `OnControlEndKey` 方法将编辑字符位置设置为文档的开始或结束。由于这些方法是监听器，而不是由 `OnRegularKey` 方法调用，因此我们需要调用 `EnsureEditStatus`、`MakeVisible` 和 `UpdateCaret` 方法：

```cpp
void WordDocument::OnControlHomeKey() { 
  EnsureEditStatus(); 

  if (editIndex > 0) { 
    editIndex = 0; 

    if (MakeVisible()) { 
      Invalidate(); 
      UpdateWindow(); 
    } 

    UpdateCaret(); 
  } 
} 

void WordDocument::OnControlEndKey() { 
  EnsureEditStatus(); 

  if (editIndex < (charList.Size() - 1)) { 
    editIndex = charList.Size() - 1; 

    if (MakeVisible()) { 
      Invalidate(); 
      UpdateWindow(); 
    } 

    UpdateCaret(); 
  } 
} 

```

## Shift Control Home 和 End

`OnShiftControlHomeKey` 和 `OnShiftControlEndKey` 方法将最后一个标记索引设置为文档的开始或结束：

```cpp
void WordDocument::OnShiftControlHomeKey() { 
  EnsureMarkStatus(); 
  ClearNextFont(); 

  if (lastMarkIndex > 0) { 
    InvalidateBlock(0, lastMarkIndex); 
    lastMarkIndex = 0; 

    if (MakeVisible()) { 
      Invalidate(); 
      UpdateWindow(); 
    } 

    UpdateCaret(); 
  } 
} 

void WordDocument::OnShiftControlEndKey() { 
  EnsureMarkStatus(); 

  if (lastMarkIndex < (charList.Size() - 1)) { 
    int lastIndex = charList.Size() - 1; 
    InvalidateBlock(lastMarkIndex, lastIndex); 
    lastMarkIndex = lastIndex; 

    if (MakeVisible()) { 
      Invalidate(); 
      UpdateWindow(); 
    } 

    UpdateCaret(); 
  } 
} 

```

## 中性键

*Backspace* 和 *Return* 键是中性键，从意义上讲，我们不在乎用户是否按下了 *Shift* 或 *Ctrl* 键。注意，*Delete* 键不是由 `OnNeutralKey` 方法处理的，因为 **Delete** 菜单项将 *Delete* 键作为其快捷键：

```cpp
void WordDocument::OnNeutralKey(WORD key) { 
  switch (key) { 
    case KeyBackspace: 
      OnBackspaceKey(); 
      break; 
    case KeyReturn: 
      OnReturnKey(); 
      break; 
  } 

  if (MakeVisible()) { 
    Invalidate(); 
    UpdateWindow(); 
    UpdateCaret(); 
  } 
} 

```

`OnBackSpaceKey` 方法所做的是相当简单的——它只是调用 `OnDelete` 方法。在 `edit` 模式下，我们首先向左移动一步，除非编辑位置已经不在文档的开始处。如果是这样，则不执行任何操作。在 `mark` 模式下，*Delete* 键和 *Backspace* 键具有相同的效果——它们都删除标记的文本。

```cpp
void WordDocument::OnBackspaceKey() { 
  switch (wordMode) { 
    case WordEdit: 
      if (editIndex > 0) { 
        OnLeftArrowKey(); 
        OnDelete(); 
      } 
      break; 

    case WordMark: 
      OnDelete(); 
      break; 
  } 
} 

```

当用户按下 *Return* 键时，会调用 `OnReturnKey` 方法。首先，我们使用换行符调用 `OnChar` 方法。`OnChar` 方法在其他任何情况下都不会带换行符调用，因为换行符不是一个图形字符。

```cpp
void WordDocument::OnReturnKey() { 
  OnChar(NewLine); 

```

在字符列表中添加换行后，我们需要将编辑段落分成两部分。`editIndex` 字段已被 `OnChar` 方法更新，现在是换行后的字符索引。第二段从编辑索引开始，到第一段末尾结束。第一段的最后一个索引设置为编辑索引减一。这意味着第一段包含换行符及其之前的所有字符，而第二段包含换行符之后的字符。

```cpp
  Paragraph* firstParagraphPtr = 
    charList[editIndex].ParagraphPtr(); 
  Paragraph* secondParagraphPtr = 
    new Paragraph(editIndex, firstParagraphPtr->Last(), 
                  firstParagraphPtr->AlignmentField(), 
                  firstParagraphPtr->Index() + 1); 
  assert(firstParagraphPtr != nullptr); 
  firstParagraphPtr->Last() = editIndex - 1; 

```

我们在段落列表中插入第二段；我们还需要将第二段中的字符设置为指向第二段。

```cpp
  paragraphList.Insert(firstParagraphPtr->Index() + 1, 
                       secondParagraphPtr); 
  for (int charIndex = secondParagraphPtr->First(); 
       charIndex <= secondParagraphPtr->Last(); ++charIndex) { 
    charList[charIndex].ParagraphPtr() = secondParagraphPtr; 
  } 

```

由于第一段丢失了字符，而第二段是最近创建的，我们需要重新计算第一段和第二段。

```cpp
  GenerateParagraph(firstParagraphPtr); 
  GenerateParagraph(secondParagraphPtr); 

```

由于我们添加了一个段落，我们需要增加后续段落的索引。

```cpp
  for (int parIndex = secondParagraphPtr->Index() + 1; 
       parIndex < paragraphList.Size(); ++parIndex) { 
    ++paragraphList[parIndex]->Index(); 
  } 

  SetDirty(true); 
  CalculateDocument(); 
  UpdateCaret(); 
  UpdateWindow(); 
} 

```

## 可见字符

当用户使用键盘时，编辑中的字符或最后标记的字符始终可见。我们首先找到可见区域；在`编辑`模式下，它是编辑字符的区域。在`标记`模式下，它是最后一个标记索引之前的字符区域，除非它是零，在这种情况下，索引被设置为零。

```cpp
bool WordDocument::MakeVisible() { 
  Rect visibleArea; 

  switch (wordMode) { 
    case WordEdit: { 
        Paragraph* editParagraphPtr = 
          charList[editIndex].ParagraphPtr(); 
        Point topLeft(0, editParagraphPtr->Top()); 
        visibleArea = topLeft + charList[editIndex].CharRect(); 
      } 
      break; 

    case WordMark: { 
        Paragraph* lastParagraphPtr = 
          charList[max(0, lastMarkIndex - 1)].ParagraphPtr(); 
        Point topLeft(0, lastParagraphPtr->Top()); 
        visibleArea = 
          topLeft + charList[max(0,lastMarkIndex - 1)].CharRect(); 
      } 
      break; 
  } 

```

我们测试可见区域是否在当前时刻实际上是可见的。如果不可见，我们调整滚动条以使其可见。

```cpp
  int horiScrollLeft = GetHorizontalScrollPosition(), 
      horiScrollPage = GetHorizontalScrollPageWidth(), 
      vertScrollTop = GetVerticalScrollPosition(), 
      vertScrollPage = GetVerticalScrollPageHeight();  
  int horiScrollRight = horiScrollLeft + horiScrollPage, 
      vertScrollBottom = vertScrollTop + vertScrollPage; 

```

如果可见区域的左边界不可见，我们将水平滚动位置设置为它的左边界。同样，如果可见区域的顶部边界不可见，我们将垂直滚动位置设置为它的顶部边界。

```cpp
  if (visibleArea.Left() < horiScrollLeft) { 
    SetHorizontalScrollPosition(visibleArea.Left()); 
    return true; 
  } 

  if (visibleArea.Top() < vertScrollTop) { 
    SetVerticalScrollPosition(visibleArea.Top()); 
    return true; 
  } 

```

当涉及到可见区域的右边界和底部边界时，事情变得稍微复杂一些。我们首先计算可见区域右边界和右滚动位置（左滚动位置加上水平滚动条的大小）之间的距离，并将水平滚动位置增加该距离。同样，我们计算可见区域右边界和底部滚动位置（顶部滚动位置加上垂直滚动条的大小）之间的距离，并将垂直滚动位置增加该距离。

```cpp
  if (visibleArea.Right() > horiScrollRight) { 
    int horiDifference = visibleArea.Right() - horiScrollRight; 
    SetHorizontalScrollPosition(horiScrollLeft + horiDifference); 
    return true; 
  } 

  if (visibleArea.Bottom() > vertScrollBottom) { 
    int vertDifference = visibleArea.Bottom() - vertScrollBottom; 
    SetVerticalScrollPosition(vertScrollTop + vertDifference); 
    return true; 
  } 

  return false; 
} 

```

# 字符计算

`GenerateParagraph`函数在字符添加或删除、字体或对齐方式更改时，为段落生成字符矩形和行列表。首先，我们通过调用`GenerateSizeAndAscentList`和`GenerateLineList`方法生成每个字符的大小和上升线列表以及行列表。然后，我们遍历行列表，通过调用`GenerateLineRectList`方法生成字符矩形。最后，我们通过将它们与原始矩形列表进行比较来使已更改的字符无效：

```cpp
void WordDocument::GenerateParagraph(Paragraph* paragraphPtr) { 
  if (!charList.Empty()) { 
    DynamicList<Size> sizeList; 
    DynamicList<int> ascentList; 
    DynamicList<CharInfo> prevCharList; 

    charList.Copy(prevCharList, paragraphPtr->First(), 
                  paragraphPtr->Last()); 

    GenerateSizeAndAscentList(paragraphPtr, sizeList, ascentList); 
    GenerateLineList(paragraphPtr, sizeList, ascentList); 

    for (LineInfo* lineInfoPtr : paragraphPtr->LinePtrList()) { 
      if (paragraphPtr->AlignmentField() == Justified) { 
        GenerateJustifiedLineRectList(paragraphPtr, lineInfoPtr, 
                                      sizeList, ascentList); 
      } 
      else {      
        GenerateRegularLineRectList(paragraphPtr, lineInfoPtr, 
                                    sizeList, ascentList); 
      } 
    } 

    GenerateRepaintSet(paragraphPtr, prevCharList); 
  } 
} 

```

## 字符大小和上升线

上升线分隔字母的上部和下部，如下图所示：

![字符大小和上升线](img/B05475_07_02.jpg)

`GenerateSizeAndAscentList`方法将给定的列表填充为段落中每个字符的大小（宽度和高度）和上升线：

```cpp
void WordDocument::GenerateSizeAndAscentList 
         (Paragraph* paragraphPtr, DynamicList<Size>& sizeList, 
             DynamicList<int>& ascentList) { 
  int index = 0; 

  for (int charIndex = paragraphPtr->First(); 
       charIndex <= paragraphPtr->Last(); ++charIndex) { 
    CharInfo charInfo = charList[charIndex]; 
    TCHAR tChar = (charInfo.Char() == NewLine) ? Space 
                                               : charInfo.Char(); 

    int width = GetCharacterWidth(charInfo.CharFont(), tChar), 
        height = GetCharacterHeight(charInfo.CharFont()), 
        ascent = GetCharacterAscent(charInfo.CharFont()); 

    sizeList.PushBack(Size(width, height)); 
    ascentList.PushBack(ascent); 
  } 
} 

```

## 行生成

`GenerateLineList`方法生成行列表。主要点是我们必须决定每行可以容纳多少单词。我们遍历字符并计算每个单词的大小。当下一个单词无法适应行时，我们开始新的一行。我们保存行上第一个和最后一个字符的索引以及其顶部位置。我们还保存其最大高度和上升，即行上最大字符的高度和上升：

```cpp
void WordDocument::GenerateLineList(Paragraph* paragraphPtr, 
                                    DynamicList<Size>& sizeList, 
                                    DynamicList<int>& ascentList){ 
  int maxHeight = 0, maxAscent = 0, lineWidth = 0, 
      spaceLineHeight = 0, spaceLineAscent = 0, 
      startIndex = paragraphPtr->First(), spaceIndex = -1; 

```

我们删除先前存储在行列表中的行。清除行列表和段落高度。将`lineTop`变量设置为零，并在计算每行的顶部位置时使用。

```cpp
  for (LineInfo* lineInfoPtr : paragraphPtr->LinePtrList()) { 
    delete lineInfoPtr; 
  } 

  paragraphPtr->Height() = 0; 
  paragraphPtr->LinePtrList().Clear(); 
  int lineTop = 0; 

  for (int charIndex = paragraphPtr->First(); 
       charIndex <= paragraphPtr->Last(); ++charIndex) { 
    CharInfo charInfo = charList[charIndex]; 

    if (charInfo.Char() != NewLine) { 
      lineWidth += 
        sizeList[charIndex - paragraphPtr->First()].Width(); 
    } 

```

如果`nextFont`参数是活动的（不等于`SystemFont`）并且我们在编辑模式下达到了编辑索引，我们计算`nextFont`参数的高度和上升。在这种情况下，我们只对字体的高度和上升感兴趣，而不需要计算其平均字符的宽度。

```cpp
    if ((nextFont != SystemFont) && (charIndex == editIndex) && 
        (wordMode == WordEdit)) { 
      maxHeight = max(maxHeight, GetCharacterHeight(nextFont)); 
      maxAscent = max(maxAscent, GetCharacterAscent(nextFont)); 
    } 

```

注意，我们必须减去段落的第一个索引，因为每行的索引都是相对于段落开头的。记住，字符列表是文档中所有段落的公共部分。

```cpp
    else { 
      maxHeight = max(maxHeight, 
        sizeList[charIndex - paragraphPtr->First()].Height()); 
      maxAscent = max(maxAscent, 
        ascentList[charIndex - paragraphPtr->First()]); 
    } 

    if (charInfo.Char() == Space) { 
      spaceIndex = charIndex; 

      spaceLineHeight = max(spaceLineHeight, maxHeight); 
      spaceLineAscent = max(spaceLineAscent, maxAscent); 

      maxHeight = 0; 
      maxAscent = 0; 
    } 

```

当我们找到换行符时，我们已经到达段落的末尾。

```cpp
    if (charInfo.Char() == NewLine) { 
      spaceLineHeight = max(spaceLineHeight, maxHeight); 
      spaceLineAscent = max(spaceLineAscent, maxAscent); 

      LineInfo* lineInfoPtr = 
        new LineInfo(startIndex - paragraphPtr->First(), 
                     charIndex - paragraphPtr->First(), 
                     lineTop, spaceLineHeight, spaceLineAscent); 
      assert(lineInfoPtr != nullptr); 

      for (int index = lineInfoPtr->First(); 
           index <= lineInfoPtr->Last(); ++index) { 
        charList[paragraphPtr->First() + index].LineInfoPtr() = 
          lineInfoPtr; 
      } 

      paragraphPtr->Height() += spaceLineHeight; 
      paragraphPtr->LinePtrList().PushBack(lineInfoPtr); 
      break; 
    } 

```

当编辑行的宽度超过页面宽度时，实际上有三种不同的情况：

+   行由至少一个完整的单词组成（空格不等于负一）

+   行由一个太长而无法适应页面的单词组成（空格等于负一且`charIndex`大于`startIndex`）

+   行由一个比页面宽一个字符的单词组成（空格等于负一且`charIndex`等于`startIndex`）

第三种情况不太可能但有可能发生。

```cpp
    if (lineWidth > PageInnerWidth()) { 
      LineInfo* lineInfoPtr = new LineInfo(); 
      assert(lineInfoPtr != nullptr); 
      lineInfoPtr->Top() = lineTop; 
      lineTop += spaceLineHeight; 

```

如果一行由至少一个完整的单词后跟一个空格组成，我们丢弃最后一个空格，并从下一个字符开始新行。

```cpp
      if (spaceIndex != -1) { 
        lineInfoPtr->First() = startIndex - paragraphPtr->First(); 
        lineInfoPtr->Last() = spaceIndex - paragraphPtr->First(); 
        lineInfoPtr->Ascent() = spaceLineAscent; 
        lineInfoPtr->Height() = spaceLineHeight; 
        startIndex = spaceIndex + 1; 
      } 

```

如果一行由一个单独的单词（至少有两个字母）组成，且其宽度不适合在页面上，我们定义该行包含最后一个适合的字符，并从下一个字符开始新行。

```cpp
      else { 
        if (charIndex > startIndex) { 
          lineInfoPtr->First() = 
            startIndex - paragraphPtr->First(); 
          lineInfoPtr->Last() = 
            charIndex - paragraphPtr->First() - 1; 
          startIndex = charIndex; 
        } 

```

最后，在不太可能的情况下，如果单个字符比页面宽，我们只需让该字符构成整个行，并让下一个索引是起始索引。

```cpp
        else { 
          lineInfoPtr->First() =charIndex - paragraphPtr->First(); 
          lineInfoPtr->Last() = charIndex - paragraphPtr->First(); 
          startIndex = charIndex + 1; 
        } 

```

行的高度和上升是最大高度和上升（具有最大高度和上升的字符的高度和上升）。

```cpp
        lineInfoPtr->Height() = maxHeight; 
        lineInfoPtr->Ascent() = maxAscent; 
      } 

```

我们将行上的所有字符设置为指向该行。

```cpp
      for (int index = lineInfoPtr->First(); 
           index <= lineInfoPtr->Last(); ++index) { 
        charList[paragraphPtr->First() + index].LineInfoPtr() = 
          lineInfoPtr; 
      } 

```

段落的高度通过行高增加，并将行指针添加到行指针列表中。

```cpp
      paragraphPtr->Height() += spaceLineHeight; 
      paragraphPtr->LinePtrList().PushBack(lineInfoPtr); 

```

为了准备下一次迭代，清除行宽、最大高度和上升。

```cpp
      lineWidth = 0; 
      maxAscent = 0; 
      maxHeight = 0; 

```

将`charIndex`循环变量设置为最新的空格索引，并将`spaceIndex`设置为`-1`，表示我们尚未在新行上找到空格。

```cpp
      charIndex = startIndex; 
      spaceIndex = -1; 
    } 
  } 
} 

```

## 正规和两端对齐的矩形列表生成

当我们为每个字符决定大小和上升线，并将字符分成行后，就是生成字符矩形的时候了。对于常规（左、居中或右对齐）段落，我们分三步进行。对齐对齐的段落由`GenerateJustifiedLineRectList`方法如下处理：

1.  我们计算每行的宽度。

1.  我们找到最左端的位置。

1.  我们为字符生成矩形。

```cpp
void WordDocument::GenerateRegularLineRectList 
                   (Paragraph* paragraphPtr,LineInfo* lineInfoPtr, 
                    DynamicList<Size>& sizeList, 
                    DynamicList<int>& ascentList) { 

```

我们遍历行的字符并计算其宽度。如果行的最后一个字符之后不是空格或换行符，我们也为其生成矩形。

```cpp
  for (int charIndex = lineInfoPtr->First(); 
       charIndex < lineInfoPtr->Last(); ++charIndex) { 
    if (charList[paragraphPtr->First() + charIndex].Char() != 
        NewLine) { 
      lineWidth += 
        sizeList[charIndex - lineInfoPtr->First()].Width(); 
    } 
  } 

  if ((charList[paragraphPtr->First()+lineInfoPtr->Last()].Char() 
      != Space) && 
      (charList[paragraphPtr->First()+lineInfoPtr->Last()].Char() 
      !=NewLine)) { 
    lineWidth += 
      sizeList[lineInfoPtr->Last()-lineInfoPtr->First()].Width(); 
  } 

```

然后，我们找到行的最左端位置以开始矩形生成。在左对齐的情况下，起始位置始终为零。在居中对齐的情况下，它是页面和文本宽度差的一半。在右对齐的情况下，它是页面和文本宽度之间的整个差值。

```cpp
  int leftPos; 

  switch (paragraphPtr->AlignmentField()) { 
    case Left: 
      leftPos = 0; 
      break; 

    case Center: 
      leftPos = (PageInnerWidth() - lineWidth) / 2; 
      break; 

    case Right: 
      leftPos = PageInnerWidth() - lineWidth; 
      break; 
  } 

```

接下来，我们遍历行并生成每个矩形。如果行的最后一个字符之后是空格，我们也为其生成矩形。

```cpp
  for (int charIndex = lineInfoPtr->First(); 
       charIndex <= lineInfoPtr->Last(); ++charIndex) { 
    Size charSize = sizeList[charIndex]; 
    int ascent = ascentList[charIndex]; 
    int topPos = lineInfoPtr->Top() + 
                 lineInfoPtr->Ascent() - ascent; 
    charList[paragraphPtr->First() + charIndex].CharRect() = 
      Rect(leftPos, topPos, leftPos + charSize.Width(), 
           topPos + charSize.Height()); 
    leftPos += charSize.Width(); 
  } 
} 

```

`GenerateJustifiedLineRectList`方法比`GenerateRegularLineRectList`方法稍微复杂一些。我们遵循之前提到的相同三个步骤。然而，在计算文本宽度时，我们省略了空格的宽度，而是计算空格的数量。

```cpp
void WordDocument::GenerateJustifiedLineRectList 
     (Paragraph* paragraphPtr, LineInfo* lineInfoPtr, 
      DynamicList<Size>& sizeList, DynamicList<int>& ascentList) { 
  int spaceCount = 0, lineWidth = 0; 

  for (int charIndex = lineInfoPtr->First(); 
       charIndex <= lineInfoPtr->Last(); ++charIndex) { 
    CharInfo charInfo = 
      charList[paragraphPtr->First() + charIndex]; 

```

我们将行上的每个字符都包括在`lineWidth`中，除了空格和换行符。

```cpp
    if (charInfo.Char() == Space) { 
      ++spaceCount; 
    } 
    else if (charInfo.Char() != NewLine) { 
      lineWidth += sizeList[charIndex].Width(); 
    } 
  } 

  if ((charList[paragraphPtr->First()+lineInfoPtr->Last()].Char() 
      != Space) && 
      (charList[paragraphPtr->First()+lineInfoPtr->Last()].Char()  
      !=NewLine)) { 
    lineWidth += sizeList[lineInfoPtr->Last()].Width(); 
  } 

```

与之前的左对齐情况类似，对齐对齐的左端位置始终为零。如果行上至少有一个空格，我们通过将页面和文本宽度的差除以空格的数量来计算空格的宽度。我们需要检查空格的数量是否大于零。否则，我们将除以零。另一方面，如果空格的数量为零，我们不需要空格宽度。

```cpp
  int leftPos = 0, spaceWidth; 
  if (spaceCount > 0) { 
    spaceWidth = (PageInnerWidth() - lineWidth) / spaceCount; 
  } 

  for (int charIndex = lineInfoPtr->First(); 
       charIndex <= lineInfoPtr->Last(); ++charIndex) { 
    Size charSize = sizeList[charIndex]; 
    int ascent = ascentList[charIndex], charWidth; 

```

如果字符是空格，我们使用计算出的空格宽度而不是其实际宽度。

```cpp
    if (charList[paragraphPtr->First() + charIndex].Char() == 
        Space) { 
      charWidth = spaceWidth; 
    } 
    else { 
      charWidth = charSize.Width(); 
    } 

    int topPos = 
      lineInfoPtr->Top() + lineInfoPtr->Ascent() - ascent; 
    charList[paragraphPtr->First() + charIndex].CharRect() = 
      Rect(leftPos, topPos, leftPos + charWidth, 
           topPos + charSize.Height()); 
    leftPos += charWidth; 
  } 
} 

```

## 无效矩形集生成

最后，我们需要使已更改的矩形集无效。有两种情况需要考虑。首先，我们有矩形本身。我们遍历字符列表，并对每个字符比较其先前和当前的矩形，如果它们不同（这将导致它们两个区域都被重绘），则使它们两个都无效。记住，无效意味着我们准备在下次窗口更新时重绘的区域。然后我们查看行列表，并在行上如果有，将文本左侧和右侧的区域添加到其中。

```cpp
void WordDocument::GenerateRepaintSet(Paragraph* paragraphPtr, 
                           DynamicList<CharInfo>& prevCharList) { 
  Point topLeft(0, paragraphPtr->Top()); 

  for (int charIndex = paragraphPtr->First(); 
       charIndex <= paragraphPtr->Last(); ++ charIndex) { 
    Rect prevRect = 
      prevCharList[charIndex - paragraphPtr->First()].CharRect(), 
         currRect = charList[charIndex].CharRect(); 

    if (prevRect != currRect) { 
      Invalidate(topLeft + prevRect); 
      Invalidate(topLeft + currRect); 
    } 
  }  
  int pageWidth = PageInnerWidth(); 

  for (LineInfo* lineInfoPtr : paragraphPtr->LinePtrList()) { 
    Rect firstRect = charList[paragraphPtr->First() + 
                              lineInfoPtr->First()].CharRect(); 

    if (firstRect.Left() > 0) { 
      Rect leftRect(0, lineInfoPtr->Top(), firstRect.Left(), 
                    lineInfoPtr->Top() + lineInfoPtr->Height()); 
      Invalidate(topLeft + leftRect); 
    } 

    Rect lastRect = charList[paragraphPtr->First() + 
                             lineInfoPtr->Last()].CharRect(); 

    if (lastRect.Right() < pageWidth) { 
      Rect rightRect(lastRect.Right(), lineInfoPtr->Top(), 
             pageWidth, lineInfoPtr->Top()+lineInfoPtr->Height()); 
      Invalidate(topLeft + rightRect); 
    } 
  } 
} 

```

# 摘要

在本章中，我们通过查看键盘处理和字符计算完成了我们的文字处理器的开发。在第八章《构建电子表格应用程序》中，我们将开始开发电子表格程序。
