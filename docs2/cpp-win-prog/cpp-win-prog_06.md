# 第六章：构建文本处理器

在本章中，我们构建了一个能够处理字符级别文本的文本处理器：也就是说，一个具有自己的字体、颜色、大小和样式的单个字符。我们还介绍了光标处理、打印和打印预览、文件拖放，以及与 ASCII 和 Unicode 文本的剪贴板处理，这意味着我们可以在该应用程序和，例如，文本编辑器之间进行剪切和粘贴。

![构建文本处理器](img/image_06_001.jpg)

# 辅助类

在此应用程序中，文档由页面、段落、行和字符组成。让我尝试解释它们是如何相互关联的：

+   首先，文档由字符列表组成。每个字符都有自己的字体和指向其所属段落和行的指针。字符信息存储在`CharInfo`类的对象中。`WordDocument`类中的`charList`字段是一个`CharInfo`对象列表。

+   字符被分为段落。一个段落不包含自己的字符列表。相反，它包含其第一个和最后一个字符在字符列表中的索引。`WordDocument`中的`paragraphList`字段是一个`Paragraph`对象列表。每个段落的最后一个字符始终是换行符。

+   每个段落被分为一行列表。下面的`Paragraph`类包含一个`Line`对象列表。一行包含其相对于段落开始的第一个和最后一个字符的索引。

+   最后，文档也被分为页面。一个页面尽可能包含尽可能多的完整段落。

每次文档中发生更改时，当前行和段落都会重新计算。页面列表也会重新计算。

让我们继续深入了解`CharInfo`、`LineInfo`和`Paragraph`类。

## 字符信息

`CharInfo`类是一个结构，包含以下内容：

+   字符及其字体

+   它的包围矩形，用于绘制字符

+   指向所属行和段落的指针

**CharInfo.h**

```cpp
class LineInfo; 
class Paragraph; 

class CharInfo { 
  public: 
    CharInfo(Paragraph* paragraphPtr = nullptr, 
             TCHAR tChar = TEXT('\0'), 
             Font font = SystemFont, Rect rect = ZeroRect); 

    CharInfo(const CharInfo& charInfo); 
    CharInfo& operator=(const CharInfo& charInfo); 

    bool WriteCharInfoToStream(ostream& outStream) const; 
    bool ReadCharInfoFromStream(istream& inStream); 

    void WriteCharInfoToClipboard(InfoList& infoList) const; 
    void ReadCharInfoFromClipboard(InfoList& infoList); 

```

该类中的每个私有字段都有自己的获取和设置值的方法。第一组方法是常量方法，返回值本身，这意味着字段的值不能通过这些方法更改。第二组方法是非常量方法，返回字段的引用，这意味着值可以更改。然而，它们不能从常量对象中调用。

```cpp
    TCHAR Char() const {return tChar;} 
    Font CharFont() const {return charFont;} 
    Rect CharRect() const {return charRect;} 
    LineInfo* LineInfoPtr() const {return lineInfoPtr;} 
    Paragraph* ParagraphPtr() const {return paragraphPtr;} 

    TCHAR& Char() {return tChar;} 
    Font& CharFont() {return charFont;} 
    Rect& CharRect() {return charRect;} 
    LineInfo*& LineInfoPtr() {return lineInfoPtr;} 
    Paragraph*& ParagraphPtr() {return paragraphPtr;} 

```

`tChar`和`charFont`字段包含字符本身及其字体，而`charRect`坐标相对于字符所属段落的左上角位置。每个字符属于一个段落以及该段落的一行，`paragraphPtr`和`lineInfoPtr`指向这些位置。

```cpp
  private: 
    TCHAR tChar; 
    Font charFont; 
    Rect charRect; 
    Paragraph* paragraphPtr; 
    LineInfo* lineInfoPtr; 
};
```

**CharInfo.cpp**

```cpp
#include "..\\SmallWindows\\SmallWindows.h"
#include "CharInfo.h"
```

`font`参数的默认值是提供默认字体的系统字体。它通常是 10 磅的 Arial 字体。

```cpp
CharInfo::CharInfo(Paragraph* paragraphPtr /* = nullptr */, 
                   TCHAR tChar /* = TEXT('\0') */, 
                   Font font/* = SystemFont */,  
                   Rect rect /* = ZeroRect */) 
 :lineInfoPtr(nullptr), 
  paragraphPtr(paragraphPtr), 
  tChar(tChar), 
  charFont(font), 
  charRect(rect) { 
  // Empty. 
} 

```

复制构造函数和赋值运算符复制字段。它们在将字符写入和从文件流读取、剪切、复制或粘贴时被调用。

```cpp
CharInfo::CharInfo(const CharInfo& charInfo) 
 :lineInfoPtr(charInfo.lineInfoPtr), 
  paragraphPtr(charInfo.paragraphPtr), 
  tChar(charInfo.tChar), 
  charFont(charInfo.charFont), 
  charRect(charInfo.charRect) { 
  // Empty. 
} 

CharInfo& CharInfo::operator=(const CharInfo& charInfo) { 
  lineInfoPtr = charInfo.lineInfoPtr; 
  paragraphPtr = charInfo.paragraphPtr; 
  tChar = charInfo.tChar; 
  charFont = charInfo.charFont; 
  charRect = charInfo.charRect; 
  return *this; 
} 

```

`WriteCharInfoToStream` 方法写入，而 `ReadCharInfoFromStream` 方法读取类的值到和从文件流和剪贴板。请注意，我们省略了 `paragraphPtr` 和 `lineInfoPtr` 指针，因为将指针地址保存到流中是没有意义的。相反，它们的值在 `WordDocument` 类中调用 `ReadCharInfoFromStream` 方法之后由 `ReadDocumentFromStream` 方法设置。

```cpp
bool CharInfo::WriteCharInfoToStream(ostream& outStream) const { 
  outStream.write((char*) &tChar, sizeof tChar); 
  charFont.WriteFontToStream(outStream); 
  charRect.WriteRectToStream(outStream); 
  return ((bool) outStream); 
} 

bool CharInfo::ReadCharInfoFromStream(istream& inStream) { 
  inStream.read((char*) &tChar, sizeof tChar); 
  charFont.ReadFontFromStream(inStream); 
  charRect.ReadRectFromStream(inStream); 
  return ((bool) inStream); 
} 

```

`WriteCharInfoToClipboard` 方法写入，而 `ReadCharInfoFromClipboard` 方法读取值到和从剪贴板。此外，在这种情况下，我们省略了 `paragraphPtr` 和 `lineInfoPtr` 指针。这些指针在 `WordDocument` 类中调用 `ReadCharInfoFromClipboard` 方法之后由 `PasteGeneric` 方法设置。

```cpp
void CharInfo::WriteCharInfoToClipboard(InfoList& infoList) const{ 
  infoList.AddValue<TCHAR>(tChar); 
  charFont.WriteFontToClipboard(infoList);  
} 

void CharInfo::ReadCharInfoFromClipboard(InfoList& infoList) { 
  infoList.GetValue<TCHAR>(tChar); 
  charFont.ReadFontFromClipboard(infoList); 
} 

```

## 行信息

`LineInfo` 方法是一个小的结构，包含关于段落中行的信息：

+   它的第一个和最后一个字符的整数索引

+   它的高度和上升，即行上最大字符的高度和上升。

+   行相对于其段落顶部位置的顶部位置

**LineInfo.h**

```cpp
class LineInfo { 
  public: 
    LineInfo(); 
    LineInfo(int first, int last, int top, 
             int height, int ascent); 

    bool WriteLineInfoToStream(ostream& outStream) const; 
    bool ReadLineInfoFromStream(istream& inStream); 

```

与前面提到的 `CharInfo` 方法类似，`LineInfo` 方法包含一组用于检查类字段的常量方法，以及一组用于修改它们的非常量方法。

```cpp
    int First() const {return first;} 
    int Last() const {return last;} 
    int Top() const {return top;} 
    int Height() const {return height;} 
    int Ascent() const {return ascent;} 
    int& First() {return first;} 
    int& Last() {return last;} 
    int& Top() {return top;} 
    int& Height() {return height;} 
    int& Ascent() {return ascent;} 

```

该类的字段是四个整数值；`first` 和 `last` 字段分别指代行上的第一个和最后一个字符。`top`、`height` 和 `ascent` 字段是行相对于段落顶部的顶部位置、最大高度和行上升。

```cpp
  private: 
    int first, last, top, height, ascent; 
};
```

**LineInfo.cpp**

```cpp
#include "..\\SmallWindows\\SmallWindows.h"
#include "LineInfo.h"
```

当用户从流中读取文档时调用默认构造函数，而当生成段落的新行时调用第二个构造函数。

```cpp
LineInfo::LineInfo() { 
  // Empty. 
} 

LineInfo::LineInfo(int first, int last, int top, 
                   int height, int ascent) 
 :first(first), 
  last(last), 
  top(top), 
  height(height), 
  ascent(ascent) { 
  // Empty. 
} 

```

`WriteLineInfoToStream` 和 `ReadLineInfoFromStream` 方法简单地写入和读取字段值。请注意，没有相应的剪切、复制和粘贴方法，因为每次粘贴段落时，段落的行列表都会被重新生成。

```cpp
bool LineInfo::WriteLineInfoToStream(ostream& outStream) const { 
  outStream.write((char*) &first, sizeof first); 
  outStream.write((char*) &last, sizeof last); 
  outStream.write((char*) &ascent, sizeof ascent); 
  outStream.write((char*) &top, sizeof top); 
  outStream.write((char*) &height, sizeof height); 
  return ((bool) outStream); 
} 

bool LineInfo::ReadLineInfoFromStream(istream& inStream) { 
  inStream.read((char*) &first, sizeof first); 
  inStream.read((char*) &last, sizeof last); 
  inStream.read((char*) &ascent, sizeof ascent); 
  inStream.read((char*) &top, sizeof top); 
  inStream.read((char*) &height, sizeof height); 
  return ((bool) inStream); 
} 

```

## 段落类

文档由一系列段落组成。`Paragraph` 结构包含以下内容：

+   它的第一个和最后一个字符的索引

+   它相对于文档开头的顶部位置及其高度

+   它在文档段落指针列表中的索引

+   它的对齐方式——段落可以是左对齐、居中对齐、两端对齐或右对齐

+   是否包含分页符，即此段落是否将位于下一页的开头

**Paragraph.h**

```cpp
enum Alignment {Left, Center, Right, Justified}; 
class WordDocument:

class Paragraph { 
  public: 
    Paragraph(); 
    Paragraph(int first, int last, 
              Alignment alignment, int index); 

    bool WriteParagraphToStream(ostream& outStream) const; 
    bool ReadParagraphFromStream(WordDocument* wordDocumentPtr, 
                                 istream& inStream); 

    void WriteParagraphToClipboard(InfoList& infoList) const; 
    void ReadParagraphFromClipboard(InfoList& infoList); 

    int& First() {return first;} 
    int& Last() {return last;} 
    int& Top() {return top;} 
    int& Index() {return index;} 
    int& Height() {return height;} 
    bool& PageBreak() {return pageBreak;} 

```

正如你所见，我们命名`AlignmentField`方法而不是仅仅命名`Alignment`方法。这样做的原因是已经有一个名为`Alignment`的类。我们不能同时给类和方法相同的名称。因此，我们在方法名称中添加了`Field`后缀。

```cpp
    Alignment& AlignmentField() {return alignment;}
    DynamicList<LineInfo*>& LinePtrList() {return linePtrList;}
```

`first`和`last`字段分别是段落中第一个和最后一个字符在文档字符列表中的索引；段落的最后一个字符始终是换行符。`top`字段是段落相对于文档开始的顶部位置，对于文档的第一个段落始终为零，对于其他段落为正值。`height`是段落的高度，`index`指的是段落在文档段落指针列表中的索引。如果`pageBreak`为`true`，则段落将始终位于页面的开头。

```cpp
    int first, last, top, height, index; 
    bool pageBreak; 

```

段落可以左对齐、右对齐、居中对齐和两端对齐。在两端对齐的情况下，为了使单词分布在整个页面宽度上，会扩展空格。

```cpp
    Alignment alignment; 

```

段落至少由一行组成。`linePtrList`列表的索引相对于段落中第一个字符的索引（不是文档），坐标相对于段落的顶部（再次不是文档）。

```cpp
    DynamicList<LineInfo*> linePtrList; 
};
```

**Paragraph.cpp**

```cpp
#include "..\\SmallWindows\\SmallWindows.h"
#include "CharInfo.h"
#include "LineInfo.h"
#include "Paragraph.h"
#include "WordDocument.h"
Paragraph::Paragraph() { /* Empty. */ }
Paragraph::Paragraph(int first, int last, Alignment alignment, int index)
:top(-1), first(first), last(last), index(index), pageBreak(false), alignment(alignment) { /* Empty. */ }
```

这个想法是`WriteParagraphToStream`和`ReadParagraphFromStream`方法分别写入和读取段落的所有信息。记住，所有坐标都是以逻辑单位（毫米的百分之一）给出的，这意味着在不同的分辨率屏幕上保存和打开文件时会有所不同。

```cpp
bool Paragraph::WriteParagraphToStream(ostream& outStream) const { 
  outStream.write((char*) &first, sizeof first); 
  outStream.write((char*) &last, sizeof last); 
  outStream.write((char*) &top, sizeof top); 
  outStream.write((char*) &height, sizeof height); 
  outStream.write((char*) &index, sizeof index); 
  outStream.write((char*) &pageBreak, sizeof pageBreak); 
  outStream.write((char*) &alignment, sizeof alignment); 

  { int linePtrListSize = linePtrList.Size(); 
    outStream.write((char*) &linePtrListSize, 
                    sizeof linePtrListSize); 

    for (const LineInfo* lineInfoPtr : linePtrList) { 
      lineInfoPtr->WriteLineInfoToStream(outStream); 
    } 
  } 

  return ((bool) outStream); 
} 

bool Paragraph::ReadParagraphFromStream 
                (WordDocument* wordDocumentPtr, istream& inStream){
  inStream.read((char*) &first, sizeof first); 
  inStream.read((char*) &last, sizeof last); 
  inStream.read((char*) &top, sizeof top); 
  inStream.read((char*) &height, sizeof height); 
  inStream.read((char*) &index, sizeof index); 
  inStream.read((char*) &pageBreak, sizeof pageBreak); 
  inStream.read((char*) &alignment, sizeof alignment);
```

当我们读取到段落的第一个和最后一个字符的索引时，我们需要设置每个字符的段落指针。

```cpp
  for (int charIndex = first; charIndex <= last; ++charIndex) {
    wordDocumentPtr->CharList()[charIndex].ParagraphPtr() = this;
  }

  { int linePtrListSize = linePtrList.Size(); 
    inStream.read((char*) &linePtrListSize, 
                  sizeof linePtrListSize); 

    for (int count = 0; count < linePtrListSize; ++count) { 
      LineInfo* lineInfoPtr = new LineInfo(); 
      assert(lineInfoPtr != nullptr); 
      lineInfoPtr->ReadLineInfoFromStream(inStream); 
      linePtrList.PushBack(lineInfoPtr);

```

与上面段落指针的情况相同，我们需要设置每个字符的行指针。

```cpp
      for (int charIndex = lineInfoPtr->First();
           charIndex <= lineInfoPtr->Last(); ++charIndex) {
        wordDocumentPtr->CharList()[first + charIndex].
          LineInfoPtr() = lineInfoPtr;
      }
    }
  }

  return ((bool) inStream);
}
```

另一方面，`WriteParagraphToClipboard`和`ReadParagraphFromClipboard`方法分别只写入和读取必要的信息。在读取段落之后，然后调用`CalaulateParagraph`方法，该方法计算字符矩形和段落的行高，并生成其行指针列表。

```cpp
void Paragraph::WriteParagraphToClipboard(InfoList& infoList) const { 
  infoList.AddValue<int>(first); 
  infoList.AddValue<int>(last); 
  infoList.AddValue<int>(top); 
  infoList.AddValue<int>(index); 
  infoList.AddValue<bool>(pageBreak); 
  infoList.AddValue<Alignment>(alignment); 
} 

void Paragraph::ReadParagraphFromClipboard(InfoList& infoList) { 
  infoList.GetValue<int>(first); 
  infoList.GetValue<int>(last); 
  infoList.GetValue<int>(top); 
  infoList.GetValue<int>(index); 
  infoList.GetValue<bool>(pageBreak); 
  infoList.GetValue<Alignment>(alignment); 
} 

```

# MainWindow 类

`MainWindow`类几乎与上一章的版本相同。它将应用程序名称设置为`Word`并返回`WordDocument`实例的地址：

```cpp
#include "..\\SmallWindows\\SmallWindows.h" 
#include "CharInfo.h" 
#include "LineInfo.h" 
#include "Paragraph.h" 
#include "WordDocument.h" 

void MainWindow(vector<String> /* argumentList */, 
                WindowShow windowShow) { 
  Application::ApplicationName() = TEXT("Word"); 
  Application::MainWindowPtr() = new WordDocument(windowShow); 
} 

```

# WordDocument 类

`WordDocument`类是应用程序的主要类。它扩展了`StandardDocument`类并利用了其基于文档的功能。

**WordDocument.h**

```cpp
class WordDocument : public StandardDocument { 
  public: 
    WordDocument(WindowShow windowShow); 

```

`InitDocument`类由构造函数、`ClearDocument`和`Delete`类调用。

```cpp
    void InitDocument(); 

```

每当用户按下 *Insert* 键时，都会调用 `OnKeyboardMode` 方法。`UpdateCaret` 方法将光标设置为 `insert` 模式下的垂直条和 `overwrite` 模式下的块。当用户标记一个或多个字符时，光标会被清除。

```cpp
    void OnKeyboardMode(KeyboardMode keyboardMode); 
    void UpdateCaret(); 

```

当用户按下、移动和释放鼠标时，我们需要找到鼠标位置处的字符索引。`MousePointToIndex` 方法找到段落，而 `MousePointToParagraphIndex` 方法找到段落中的字符。`InvalidateBlock` 方法使从最小索引（包含）到最大索引（不包含）的字符无效。

```cpp
    void OnMouseDown(MouseButton mouseButtons, Point mousePoint, 
                     bool shiftPressed, 
                     bool controlPressed); 
    void OnMouseMove(MouseButton mouseButtons, Point mousePoint, 
                     bool shiftPressed, 
                     bool controlPressed); 
    void OnMouseUp(MouseButton mouseButtons, Point mousePoint, 
                   bool shiftPressed, 
                   bool controlPressed); 
    int MousePointToIndex(Point mousePoint) const; 
    int MousePointToParagraphIndex(Paragraph* paragraphPtr, 
                                   Point mousePoint) const; 
    void InvalidateBlock(int firstIndex, int lastIndex); 

```

当用户双击一个单词时，它将被标记。如果用户确实双击了一个单词（而不是空格、句号、逗号或问号），则 `GetFirstWordIndex` 和 `GetLastWordIndex` 方法分别找到单词的第一个和最后一个索引。

```cpp
    void OnDoubleClick(MouseButton mouseButtons, Point mousePoint, 
                       bool shiftPressed, bool controlPressed); 
    int GetFirstWordIndex(int charIndex) const; 
    int GetLastWordIndex(int charIndex) const; 

```

在这个应用中，我们引入了触摸屏操作。与鼠标点击不同，可以同时触摸屏幕上的多个位置。因此，参数是一个点的列表，而不是一个单独的点。

```cpp
    void OnTouchDown(vector<Point> pointList); 
    void OnTouchMove(vector<Point> pointList); 

```

当用户通过在 **文件** 菜单中选择 **页面设置** 菜单项来更改页面设置时，会调用 `OnPageSetup` 方法。这允许用户修改页面和段落设置。`CalculateDocument` 方法将段落分配到页面上。如果一个段落带有分页标记，或者它没有完全适合当前页的其余部分，它将被放置在下一页的开始处。

```cpp
    void OnPageSetup(PageSetupInfo pageSetupInfo); 
    void CalculateDocument(); 

```

与前几章中的应用不同，我们重写了 `OnPaint` 和 `OnDraw` 方法。当客户端区域需要重新绘制时调用 `OnPaint` 方法。它执行特定的绘制动作，即仅在文档在窗口中绘制时执行的动作，而不是在发送到打印机时执行。更具体地说，我们在客户端区域添加了分页标记，但不在打印机文本中。

然后，`OnPaint` 方法调用执行文档实际绘制的 `OnDraw` 方法。在 `StandardDocument` 类（我们没有重写）中还有一个名为 `OnPrint` 的方法，当打印文档时调用 `OnDraw` 方法。

```cpp
    void OnPaint(Graphics& graphics) const; 
    void OnDraw(Graphics& graphics, DrawMode drawMode) const; 

```

与前几章中的应用类似，当用户在 **文件** 菜单中选择 **新建**、**保存**、**另存为** 或 **打开** 菜单项时，会调用 `ClearDocument`、`WriteDocumentToStream` 和 `ReadDocumentFromStream` 方法。

```cpp
    void ClearDocument(); 
    bool WriteDocumentToStream(String name, ostream& outStream) 
                               const; 
    bool ReadDocumentFromStream(String name, istream& inStream); 

```

当文本准备好复制时，`CopyEnable`方法返回`true`，即当用户标记了文本的一部分时。当用户选择**剪切**或**复制**菜单项并复制标记的文本到字符串列表时，会调用`CopyAscii`和`CopyUnicode`方法。当用户选择**剪切**或**复制**菜单项并将标记的文本以应用程序特定的格式复制时，也会调用`CopyGeneric`方法，这种格式还会复制字符的字体和样式。

```cpp
    bool CopyEnable() const; 
    bool IsCopyAsciiReady() const; 
    bool IsCopyUnicodeReady() const; 
    bool IsCopyGenericReady(int format) const; 

    void CopyAscii(vector<String>& textList) const; 
    void CopyUnicode(vector<String>& textList) const; 
    void CopyGeneric(int format, InfoList& infoList) const; 

```

当用户选择**粘贴**菜单项时，会调用`PasteAscii`、`PasteUnicode`和`PasteGeneric`方法。复制和粘贴之间的一个区别是，在复制时，上述三种方法都会被调用，但在粘贴时，只调用一个方法，其顺序与在`StandardDocument`构造函数调用中给出的格式顺序相同。

```cpp
    void PasteAscii(const vector<String>& textList); 
    void PasteUnicode(const vector<String>& textList); 
    void PasteGeneric(int format, InfoList& infoList); 

```

我们没有重写`CutEnable`或`OnCut`方法，因为`StandardDocument`类中的`CutEnable`方法会调用`CopyEnable`方法，而`OnCut`方法会调用`OnDelete`方法，然后是`OnCopy`方法。

**删除**菜单项处于启用状态，除非输入位置在文档末尾，在这种情况下，没有可以删除的内容。`Delete`方法是一个通用方法，用于删除文本，当用户按下*Delete*或*Backspace*键或正在覆盖标记的文本块时会被调用。

```cpp
    bool DeleteEnable() const; 
    void OnDelete(); 
    void Delete(int firstIndex, int lastIndex); 

```

`OnPageBreak`方法设置编辑段落的分页状态。如果发生分页，段落将被放置在下一页的开头。`OnFont`方法显示标准字体对话框，用于设置下一个要输入的字符或标记块的字体和颜色。

```cpp
    DEFINE_BOOL_LISTENER(WordDocument, PageBreakEnable) 
    DEFINE_VOID_LISTENER(WordDocument, OnPageBreak) 
    DEFINE_VOID_LISTENER(WordDocument, OnFont) 

```

段落可以左对齐、居中对齐、右对齐或两端对齐。如果当前编辑的段落或所有当前标记的段落具有所询问的对齐方式，则会出现单选标记。所有听众都会调用`IsAlignment`和`SetAlignment`方法，分别用于获取编辑段落或所有标记段落的当前对齐方式以及设置对齐方式。

```cpp
    DEFINE_BOOL_LISTENER(WordDocument, LeftRadio) 
    DEFINE_VOID_LISTENER(WordDocument, OnLeft) 
    DEFINE_BOOL_LISTENER(WordDocument, CenterRadio) 
    DEFINE_VOID_LISTENER(WordDocument, OnCenter) 
    DEFINE_BOOL_LISTENER(WordDocument, RightRadio) 
    DEFINE_VOID_LISTENER(WordDocument, OnRight) 
    DEFINE_BOOL_LISTENER(WordDocument, JustifiedRadio) 
    DEFINE_VOID_LISTENER(WordDocument, OnJustified)     

    bool IsAlignment(Alignment alignment) const; 
    void SetAlignment(Alignment alignment); 

```

每当用户按下图形字符时，都会调用`OnChar`方法；它根据键盘是否处于`insert`或`overwrite`模式来调用`InsertChar`或`OverwriteChar`方法。当文本被标记且用户更改字体时，字体会应用于所有标记字符。然而，在编辑文本时，下一个要输入的字符的字体会被设置。

当用户进行除输入下一个字符之外的其他操作时，例如点击鼠标或按下任何箭头键，会调用`ClearNextFont`方法，该方法通过将其设置为`SystemFont`方法来清除下一个字体。

```cpp
    void OnChar(TCHAR tChar); 
    void InsertChar(TCHAR tChar, Paragraph* paragraphPtr); 
    void OverwriteChar(TCHAR tChar, Paragraph* paragraphPtr);   
    void ClearNextFont(); 

```

每当用户按下键时，都会调用`OnKeyDown`方法，例如箭头键、*向上翻页*和*向下翻页*、*Home*和*End*、*Delete*或*Backspace*：

```cpp
    bool OnKeyDown(WORD key, bool shiftPressed, 
                   bool controlPressed); 
    void OnRegularKey(WORD key); 
    void EnsureEditStatus(); 
    void OnLeftArrowKey(); 
    void OnRightArrowKey(); 
    void OnUpArrowKey(); 
    void OnDownArrowKey(); 
    int MousePointToIndexDown(Point mousePoint) const; 
    void OnPageUpKey(); 
    void OnPageDownKey(); 
    void OnHomeKey(); 
    void OnEndKey(); 

```

当用户按下键而没有同时按下 *Shift* 键时，光标会移动。然而，当他们按下 *Shift* 键时，文本的标记会改变。

```cpp
    void OnShiftKey(WORD key); 
    void EnsureMarkStatus(); 
    void OnShiftLeftArrowKey(); 
    void OnShiftRightArrowKey(); 
    void OnShiftUpArrowKey(); 
    void OnShiftDownArrowKey(); 
    void OnShiftPageUpKey(); 
    void OnShiftPageDownKey(); 
    void OnShiftHomeKey(); 
    void OnShiftEndKey(); 

```

当用户同时按下 *Home* 或 *End* 键和 *Ctrl* 键时，光标将被放置在文档的开始或结束位置。如果他们还按下 *Shift* 键，文本将被标记。

我们使用监听器而不是常规方法的原因是，所有涉及 *Ctrl* 键的操作都被 Small Windows 解释为加速器。监听器也被添加到以下构造函数中的菜单中。

```cpp
    DEFINE_VOID_LISTENER(WordDocument, OnControlHomeKey); 
    DEFINE_VOID_LISTENER(WordDocument, OnControlEndKey); 
    DEFINE_VOID_LISTENER(WordDocument, OnShiftControlHomeKey); 
    DEFINE_VOID_LISTENER(WordDocument, OnShiftControlEndKey); 

```

同样存在 *返回*、*退格* 和 *删除* 键，在这种情况下，我们并不关心是否按下了 *Shift* 或 *Ctrl* 键。*删除* 键由 **删除** 菜单项加速器处理。

```cpp
    void OnNeutralKey(WORD key); 
    void OnReturnKey(); 
    void OnBackspaceKey(); 

```

当用户使用键盘移动光标时，编辑字符将可见。`MakeVisible` 方法确保它是可见的，即使这意味着滚动文档。

```cpp
    void MakeVisible(); 

```

当段落发生某些变化（字符被添加或删除，字体或对齐方式改变，或页面设置）时，需要计算字符的位置。`GenerateParagraph` 方法为每个字符计算周围矩形，并通过调用 `GenerateSizeAndAscentList` 方法计算字符的大小和上升线，调用 `GenerateLineList` 方法将段落分成行，调用 `GenerateRegularLineRectList` 方法生成左对齐、居中对齐或右对齐段落的字符矩形，或调用 `GenerateJustifiedLineRectList` 方法为对齐段落生成字符矩形，以及调用 `GenerateRepaintSet` 方法使更改的字符无效。

```cpp
    void GenerateParagraph(Paragraph* paragraphPtr); 
    void GenerateSizeAndAscentList(Paragraph* paragraphPtr, 
                                   DynamicList<Size>& sizeList, 
                                   DynamicList<int>& ascentList); 
    void GenerateLineList(Paragraph* paragraphPtr, 
                          DynamicList<Size>& sizeList, 
                          DynamicList<int>& ascentList); 

    void GenerateRegularLineRectList(Paragraph* paragraphPtr, 
                                     LineInfo* lineInfoPtr, 
                                     DynamicList<Size>& sizeList, 
                                     DynamicList<int>&ascentList); 
    void GenerateJustifiedLineRectList(Paragraph* paragraphPtr, 
                                  LineInfo* lineInfoPtr, 
                                  DynamicList<Size>& sizeList, 
                                  DynamicList<int>& ascentList); 
    void InvalidateRepaintSet(Paragraph* paragraphPtr, 
                            DynamicList<CharInfo>& prevRectList);
    DynamicList<CharInfo>& CharList() {return charList;}
```

本应用的一个核心部分是 `wordMode` 方法。在某个时刻，应用可以被设置为 `编辑` 模式（光标可见），在这种情况下 `wordMode` 是 `WordEdit` 方法，或者 `标记` 模式（文本的一部分被标记），在这种情况下 `wordMode` 是 `WordMark` 方法。在章节的后面，我们将遇到 **编辑模式** 和 **标记模式** 这样的表达式，它们指的是 `wordMode` 的值：`WordEdit` 或 `WordMark`。

我们还会遇到 **插入模式** 和 **覆盖模式** 的表达式，它们指的是键盘的 `input` 模式，即 `InsertKeyboard` 或 `OverwriteKeyboard` 方法，这是 Small Windows 类 `Document` 中的 `GetKeyboardMode` 方法返回的。

`totalPages` 字段包含页数，这在打印和设置垂直滚动条时使用。字符列表存储在 `charList` 列表中，段落指针列表存储在 `paragraphList` 列表中。请注意，段落是动态创建和删除的 `Paragraph` 对象，而字符是静态的 `CharInfo` 对象。此外，请注意，每个段落不包含字符列表。只有一个 `charList`，它是所有段落的公共部分。然而，每个段落都包含它自己的 `Line` 指针列表，这些指针是段落本地的。

在本章中，我们还将遇到诸如 **编辑字符** 这样的表达式，它指的是 `charList` 列表中索引为 `editIndex` 的字符。如本章开头所述，每个字符都有指向其段落和行的指针。**编辑段落** 和 **编辑行** 这些表达式指的是由编辑字符指向的段落和行。

`firstMarkIndex` 和 `lastMarkIndex` 字段包含在 `mark` 模式下第一个和最后一个标记字符的索引。它们也出现在诸如 **第一个标记字符**、**第一个标记段落**、**第一个标记行** 以及 **最后一个标记字符**、**最后一个标记段落** 和 **最后一个标记行** 等表达式中。请注意，这两个字段指的是时间顺序，而不一定是它们的物理顺序。当需要时，我们将定义 `minIndex` 和 `maxIndex` 方法来按物理顺序引用文档中的第一个和最后一个标记。

当用户在 `edit` 模式下设置字体时，它被存储在 `nextFont` 字体中，然后当用户输入下一个字符时使用。光标会考虑 `nextFont` 字体的状态，即如果 `nextFont` 字体不等于 `ZeroFont` 字体，它就会用来设置光标。然而，一旦用户做其他任何事情，`nextFont` 字体就会被清除。

用户可以通过菜单项或触摸屏幕来放大文档。在这种情况下，我们需要 `initZoom` 和 `initDistance` 字段来跟踪缩放。最后，我们需要 `WordFormat` 字段来识别剪切、复制和粘贴的应用程序特定信息。它被赋予任意值 1002。

```cpp
  private: 
    enum {WordEdit, WordMark} wordMode; 

    int totalPages; 
    DynamicList<CharInfo> charList; 
    DynamicList<Paragraph*> paragraphList; 

    int editIndex, firstMarkIndex, lastMarkIndex; 
    Font nextFont; 

    double initZoom, initDistance;  
    static const unsigned int WordFormat = 1002; 
};
```

**WordDocument.cpp**

```cpp
#include "..\\SmallWindows\\SmallWindows.h"
#include "CharInfo.h"
#include "LineInfo.h"
#include "Paragraph.h"
#include "WordDocument.h"
```

`WordDocument` 构造函数调用 `StandardDocument` 构造函数。`UnicodeFormat` 和 `AsciiFormat` 方法是由 Small Windows 定义的通用格式，而 `WordFormat` 方法是特定于这个应用程序的。

```cpp
WordDocument::WordDocument(WindowShow windowShow) 
 :StandardDocument(LogicalWithScroll, USLetterPortrait, 
                   TEXT("Word Files, wrd; Text Files, txt"), 
                   nullptr, OverlappedWindow, windowShow, 
                   {WordFormat, UnicodeFormat, AsciiFormat}, 
                   {WordFormat, UnicodeFormat, AsciiFormat}) { 

```

**格式** 菜单包含 **字体** 和 **分页符** 菜单项。与本书中较早的应用程序不同，我们向 `StandardFileMenu` 发送 `true`。这表示我们希望在 **文件** 菜单中包含 **页面设置**、**打印预览** 和 **打印** 菜单项。

```cpp
  Menu menuBar(this); 
  menuBar.AddMenu(StandardFileMenu(true)); 
  menuBar.AddMenu(StandardEditMenu()); 

  Menu formatMenu(this, TEXT("F&ormat")); 
  formatMenu.AddItem(TEXT("&Font\tCtrl+F"), OnFont); 
  formatMenu.AddItem(TEXT("&Page Break\tCtrl+B"), 
                     OnPageBreak, PageBreakEnable); 
  menuBar.AddMenu(formatMenu); 

```

**对齐** 菜单包含左对齐、居中对齐、右对齐和两端对齐的选项：

```cpp
  Menu alignmentMenu(this, TEXT("&Alignment")); 
  alignmentMenu.AddItem(TEXT("&Left\tCtrl+L"), OnLeft, 
                        nullptr, nullptr, LeftRadio); 
  alignmentMenu.AddItem(TEXT("&Center\tCtrl+E"), OnCenter, 
                        nullptr, nullptr, CenterRadio); 
  alignmentMenu.AddItem(TEXT("&Right\tCtrl+R"), OnRight, 
                        nullptr, nullptr, RightRadio); 
  alignmentMenu.AddItem(TEXT("&Justified\tCtrl+J"), OnJustified, 
                        nullptr, nullptr, JustifiedRadio); 
  menuBar.AddMenu(alignmentMenu); 

  menuBar.AddMenu(StandardHelpMenu()); 
  SetMenuBar(menuBar); 

```

`extraMenu` 菜单仅用于加速器；请注意，我们不会将其添加到菜单栏中。菜单文本或其项目内容也不重要。我们只想允许用户通过按住 *Ctrl* 键并使用 *Home* 或 *End* 键，以及可能使用 *Shift* 键来跳转到文档的开始或结束。

```cpp
  Menu extraMenu(this); 
  extraMenu.AddItem(TEXT("&A\tCtrl+Home"), OnControlHomeKey); 
  extraMenu.AddItem(TEXT("&B\tCtrl+End"), OnControlEndKey); 
  extraMenu.AddItem(TEXT("&C\tShift+Ctrl+Home"), 
                    OnShiftControlHomeKey); 
  extraMenu.AddItem(TEXT("&D\tShift+Ctrl+End"), 
                    OnShiftControlEndKey); 

```

最后，我们调用 `InitDocument` 方法来初始化空文档。`InitDocument` 方法也由 `ClearDocument` 和 `Delete` 类在以下情况下调用，当初始化代码放置在其自己的方法中时。

```cpp
  InitDocument(); 
} 

```

文档始终至少包含一个段落，该段落又至少包含一个换行符。我们创建第一个字符和第一个左对齐的段落。段落和字符被添加到 `paragraphList` 和 `charList` 列表中。

然后，段落通过 `GenerateParagraph` 方法计算，并通过 `CalculateDocument` 方法在文档中分布。最后，通过 `UpdateCaret` 方法更新光标。

```cpp
void WordDocument::InitDocument() { 
  wordMode = WordEdit; 
  editIndex = 0; 
  Paragraph* firstParagraphPtr = new Paragraph(0, 0, Left, 0); 
  assert(firstParagraphPtr != nullptr); 
  Font font(TEXT("Times New Roman"), 36, false, true); 
  charList.PushBack(CharInfo(firstParagraphPtr, NewLine, font)); 
  GenerateParagraph(firstParagraphPtr); 
  paragraphList.PushBack(firstParagraphPtr); 
  CalculateDocument(); 
  UpdateCaret(); 
} 

```

## 光标

由于在本章中我们介绍了文本处理，我们需要跟踪光标：在 `insert` 模式下的闪烁垂直线或块（在 `overwrite` 模式下）指示输入字符的位置。`UpdateCaret` 方法由 `OnKeyboardMode` 方法（当用户按下 *Insert* 键时调用）以及其他方法调用，当输入位置正在修改时。

```cpp
void WordDocument::OnKeyboardMode(KeyboardMode/*=KeyboardMode*/) { 
  UpdateCaret(); 
} 

void WordDocument::UpdateCaret() { 
  switch (wordMode) { 
    case WordEdit: { 
        CharInfo charInfo = charList[editIndex];
        Rect caretRect = charList[editIndex].CharRect();
```

在 `edit` 模式下，光标将可见，我们获取编辑字符所在区域。然而，如果 `nextFont` 字体处于活动状态（不等于 `SystemFont` 字体），用户已更改字体，我们必须考虑这一点。在这种情况下，我们将光标宽度和高度设置为 `nextFont` 字体平均字符的大小。

```cpp
        if (nextFont != SystemFont) { 
          int width = GetCharacterAverageWidth(nextFont), 
              height = GetCharacterHeight(nextFont); 
          caretRect.Right() = caretRect.Left() + width; 
          caretRect.Top() = caretRect.Bottom() - height; 
        } 

```

如果 `nextFont` 字体未处于活动状态，我们检查键盘是否处于 `insert` 模式，并且光标是否不在段落开头。在这种情况下，光标的垂直坐标将反映前一个字符的字体大小，因为下一个要输入的字符将使用该字体。

```cpp
        else if ((GetKeyboardMode() == InsertKeyboard) && 
                 (charInfo.ParagraphPtr()->First() < editIndex)) { 
          Rect prevCharRect = charList[editIndex - 1].CharRect(); 
          caretRect.Top() = caretRect.Bottom() – prevCharRect.Height(); 
        }

```

如果键盘处于 `insert` 模式，无论 `nextFont` 字体是否处于活动状态，光标都将是一条垂直线。它被赋予一个单位宽度（这后来会被四舍五入到物理像素的宽度）。

```cpp
        if (GetKeyboardMode() == InsertKeyboard) { 
          caretRect.Right() = caretRect.Left() + 1; 
        } 

```

光标不会超出页面范围。如果它超出了，其右边界将被设置为页面的边界。

```cpp
        if (caretRect.Right() >= PageInnerWidth()) { 
          caretRect.Right() = PageInnerWidth() - 1; 
        } 

```

最后，我们需要编辑段落的顶部位置，因为光标到目前为止是相对于其顶部位置计算的。

```cpp
        Paragraph* paragraphPtr = 
          charList[editIndex].ParagraphPtr(); 
        Point topLeft = Point(0, paragraphPtr->Top()); 
        SetCaret(topLeft + caretRect); 
      } 
      break; 

```

在 `mark` 模式下，光标将不可见。因此，我们按照以下方式调用 `ClearCaret`：

```cpp
    case WordMark: 
      ClearCaret(); 
      break; 
  } 
} 

```

## 鼠标输入

`OnMouseDown`、`OnMouseMove`、`OnMouseUp` 和 `OnDoubleClick` 方法接收按下的按钮和鼠标坐标。在所有四种情况下，我们检查是否按下了左键鼠标。`OnMouseDown` 方法首先调用 `EnsureEditStatus` 方法以清除任何潜在标记区域。然后它将应用程序设置为 `mark` 模式（这可能会稍后被 `OnMouseUp` 方法更改）并通过调用 `MousePointToIndex` 方法查找指向的字符的索引。通过调用 `ClearNextFont` 方法清除 `nextFont` 字段。我们还调用 `UpdateCaret` 方法，因为当用户拖动鼠标时，光标将被清除。

```cpp
void WordDocument::OnMouseDown(MouseButton mouseButtons, 
                          Point mousePoint, bool shiftPressed, 
                          bool controlPressed) { 
  if (mouseButtons == LeftButton) { 
    EnsureEditStatus(); 
    ClearNextFont(); 
    wordMode = WordMark; 
    firstMarkIndex = lastMarkIndex = 
      MousePointToIndex(mousePoint); 
    UpdateCaret(); 
  } 
} 

```

在 `OnMouseMove` 方法中，我们通过调用 `MousePointToIndex` 方法检索鼠标的段落和字符。如果自上次调用 `OnMouseDown` 或 `OnMouseMove` 方法以来鼠标已移动到新字符，我们通过调用 `InvalidateBlock` 方法并传递当前和新的鼠标位置来更新标记文本，这将使当前和上次鼠标事件之间的文本部分无效。请注意，我们不使整个标记块无效。我们只使前一个和当前鼠标位置之间的块无效，以避免闪烁。

```cpp
void WordDocument::OnMouseMove(MouseButton mouseButtons, 
                          Point mousePoint, bool shiftPressed, 
                          bool controlPressed) { 
  if (mouseButtons == LeftButton) { 
    int newLastMarkIndex = MousePointToIndex(mousePoint); 

    if (lastMarkIndex != newLastMarkIndex) { 
      InvalidateBlock(lastMarkIndex, newLastMarkIndex); 
      lastMarkIndex = newLastMarkIndex; 
    } 
  } 
} 

```

在 `OnMouseUp` 方法中，我们只需检查最后一个位置。如果它与第一个位置相同（用户在同一个字符上按下并释放鼠标），我们将应用程序更改为 `edit` 模式并调用 `UpdateCaret` 方法以使光标可见。

```cpp
void WordDocument::OnMouseUp(MouseButton mouseButtons, 
                             Point mousePoint, bool shiftPressed, 
                             bool controlPressed) { 
  if (mouseButtons == LeftButton) { 
    if (firstMarkIndex == lastMarkIndex) { 
      wordMode = WordEdit; 
      editIndex = min(firstMarkIndex, charList.Size() - 1); 
      UpdateCaret(); 
    } 
  } 
} 

```

`MousePointToIndex` 方法用于找到用户点击的段落，并调用 `MousePointToParagraphIndex` 方法来找到段落中的字符。我们将功能分为两个方法的原因是，第七章中的 `MousePointToIndexDown` 方法，*键盘输入和字符计算*，也调用了 `MousePointToParagraphIndex` 方法，该方法遍历段落列表。如果垂直位置小于段落的顶部位置，则正确的段落是前一个段落。

这种寻找正确段落的略显繁琐的方法是由于段落以这种方式分布在页面上，即当段落无法适应页面的其余部分，或者如果它带有分页符时，它被放置在下一页的开头。这可能会导致文档中没有任何段落的位置。如果用户点击这样的区域，我们希望该区域之前的段落是正确的。同样，如果用户点击文档的最后一个段落下方，它将成为正确的段落。

```cpp
int WordDocument::MousePointToIndex(Point mousePoint) const{ 
  for (int parIndex = 1; parIndex < paragraphList.Size(); 
       ++parIndex) { 
    Paragraph* paragraphPtr = paragraphList[parIndex]; 

    if (mousePoint.Y() < paragraphPtr->Top()) { 
      return MousePointToParagraphIndex 
             (paragraphList[parIndex - 1], mousePoint); 
    } 
  } 

  return MousePointToParagraphIndex 
         (paragraphList[paragraphList.Size() - 1], mousePoint); 
} 

```

`MousePointToParagraphIndex` 方法用于找到段落中点击的字符。首先，我们从鼠标位置中减去段落的顶部位置，因为段落的行坐标是相对于段落顶部位置的。

```cpp
int WordDocument::MousePointToParagraphIndex 
                          (Paragraph* paragraphPtr,Point mousePoint) const{ 
  mousePoint.Y() -= paragraphPtr->Top(); 

```

如前所述，用户可能点击在段落区域下方的一个位置。在这种情况下，我们将鼠标位置设置为它的`-1`高度，这相当于用户点击了段落的最后一行。

```cpp
  if (mousePoint.Y() >= paragraphPtr->Height()) { 
    mousePoint.Y() = paragraphPtr->Height() - 1; 
  } 

```

首先，我们需要在段落中找到正确的行。我们检查每一行，并通过将其与行的顶部位置和高度的加和进行比较来测试鼠标位置是否位于行内。与之前提到的`MousePointToIndex`方法中的段落搜索相比，这个搜索要简单一些，因为段落中的行之间没有空格，而文档中的段落之间可能有空格。

```cpp
  int firstChar = paragraphPtr->First();
  for (LineInfo* lineInfoPtr : paragraphPtr->LinePtrList()) { 
    if (mousePoint.Y() < (lineInfoPtr->Top() + 
                          lineInfoPtr->Height())) { 
      Rect firstRect =
              charList[firstChar +lineInfoPtr->First()].CharRect(),
            lastRect =
              charList[firstChar + lineInfoPtr->Last()].CharRect(); 

```

当我们找到正确的行时，我们需要考虑三种情况：用户可能点击了文本的左侧（如果段落是居中或右对齐的），右侧（如果它是左对齐或居中对齐的），或者文本本身。如果他们点击了行的左侧或右侧，我们返回行的第一个或最后一个字符的索引。请注意，我们添加了段落第一个字符的索引，因为行的索引是相对于段落第一个索引的。

```cpp
      if (mousePoint.X() < firstRect.Left()) { 
        return paragraphPtr->First() + lineInfoPtr->First(); 
      } 
      else if (lastRect.Right() <= mousePoint.X()) { 
        return paragraphPtr->First() + lineInfoPtr->Last(); 
      } 

```

如果用户点击了文本，我们需要找到正确的字符。我们遍历行的字符，并将鼠标位置与字符的右侧边界进行比较。当我们找到正确的字符时，我们需要决定用户是否点击了字符的左侧或右侧边界。在右侧边界的情况下，我们将字符索引加一。

```cpp
      else { 
        for (int charIndex = lineInfoPtr->First(); 
             charIndex <= lineInfoPtr->Last(); ++charIndex) { 
          Rect charRect = charList[charIndex].CharRect(); 

          if (mousePoint.X() < charRect.Right()) { 
            int leftSize = mousePoint.X() - charRect.Left(), 
                rightSide = charRect.Right() - mousePoint.X(); 

            return paragraphPtr->First() + 
              ((leftSize < rightSide) ? charIndex 
                                      : (charIndex + 1)); 
          } 
        } 
      } 
    } 
  } 

```

如前所述，段落中的行与行之间没有空格。因此，我们总能找到正确的行，永远不会达到这个点。然而，为了避免编译器错误，我们仍然必须返回一个值。在这本书中，我们会在少数情况下使用以下符号：

```cpp
  assert(false); 
  return 0; 
} 

void WordDocument::InvalidateBlock(int firstIndex, int lastIndex){ 
  int minIndex = min(firstIndex, lastIndex), 
      maxIndex = min(max(firstIndex, lastIndex).
                     charList.Size() - 1); 

  for (int charIndex = minIndex; charIndex <= maxIndex; 
       ++charIndex) { 
    CharInfo charInfo = charList[charIndex]; 
    Point topLeft(0, charInfo.ParagraphPtr()->Top()); 
    Invalidate(topLeft + charInfo.CharRect()); 
  } 
} 

```

当用户双击鼠标左键时，鼠标击中的单词将被标记。应用已被设置为`编辑`模式，并且`editIndex`方法已经被适当地设置，因为对`OnDoubleClick`方法的调用总是先于对`OnMouseDown`和`OnMouseUp`方法的调用。如果鼠标击中了一个单词，我们将标记该单词并将应用设置为`标记`模式。

我们通过调用`GetFirstWordIndex`和`GetLastWordIndex`方法来找到单词的第一个和最后一个字符的索引。如果第一个索引小于最后一个索引，用户实际上双击了一个单词，我们将它标记。如果第一个索引不小于最后一个索引，用户双击了空格或分隔符，在这种情况下，双击没有效果。

```cpp
void WordDocument::OnDoubleClick(MouseButton mouseButtons, 
                     Point mousePoint, bool shiftPressed, 
                     bool controlPressed) { 
  int firstIndex = GetFirstWordIndex(editIndex), 
      lastIndex = GetLastWordIndex(editIndex); 

  if (firstIndex < lastIndex) { 
    wordMode = WordMark; 
    firstMarkIndex = firstIndex; 
    lastMarkIndex = lastIndex; 

    UpdateCaret(); 
    InvalidateBlock(firstMarkIndex, lastMarkIndex); 
    UpdateWindow(); 
  } 
} 

```

在`GetFirstWordIndex`方法中，我们通过在字符列表中向后移动直到我们到达文档的开始或一个非字母字符来找到单词的第一个字符的索引。

```cpp
int WordDocument::GetFirstWordIndex(int charIndex) const{ 
  while ((charIndex >= 0) && 
         (isalpha(charList[charIndex].Char()))) { 
    --charIndex; 
  } 
  return (charIndex + 1); 
} 

```

在`GetLastWordIndex`方法中，我们不需要检查字符列表的末尾，因为最后一个字符始终是换行符，它不是一个字母。请注意，在这种情况下，我们返回单词最后一个字符之后的字符索引，因为文本标记有效到，但不包括最后一个字符。

```cpp
int WordDocument::GetLastWordIndex(int charIndex) const{ 
  while (isalpha(charList[charIndex].Char())) { 
    ++charIndex; 
  } 
  return charIndex; 
} 

```

## 触摸屏

在触摸屏上，用户可以通过在屏幕上拖动两个手指来缩放文档。当用户触摸屏幕时，会调用`OnTouchDown`方法，当用户移动手指时，会调用`OnTouchMove`方法。与之前提到的鼠标输入方法不同，用户可以同时触摸屏幕上的多个点。这些点存储在`pointList`列表中。

如果列表不包含两个点，我们只需让`Window`类执行默认操作，即将每个触摸动作转换为鼠标动作。

```cpp
void WordDocument::OnTouchDown(vector<Point> pointList) { 
  if (pointList.size() == 2) { 
    initZoom = GetZoom(); 
    Point firstInitPoint = pointList[0], 
          secondInitPoint = pointList[1]; 
    double width = firstInitPoint.X() - secondInitPoint.X(), 
           height = firstInitPoint.Y() - secondInitPoint.Y(), 
    initDistance = sqrt((width * width) + (height * height)); 
  } 
  else { 
    Window::OnTouchDown(pointList); 
  } 
} 

```

当用户在屏幕上移动手指时，会计算手指之间的距离，并根据初始距离设置缩放。缩放的范围允许在 10%（因子 0.1）和 1,000%（因子 10.0）之间：

```cpp
void WordDocument::OnTouchMove(vector<Point> pointList) { 
  if (pointList.size() == 2) { 
    Point firstPoint = pointList[0], secondPoint = pointList[1]; 

    int width = firstPoint.X() - secondPoint.X(), 
        height = firstPoint.Y() - secondPoint.Y(); 
    double distance = sqrt((width * width) + (height * height)); 

    double factor = distance / initDistance; 
    double newZoom = factor * initZoom; 
    SetZoom(min(max(newZoom, 0.1), 10.0)); 

    UpdateCaret(); 
    Invalidate(); 
    UpdateWindow(); 
  } 
  else { 
    Window::OnTouchMove(pointList); 
  } 
} 

```

## 页面设置和计算

当用户在**文件**菜单中选择标准**页面设置**菜单项时，会调用`OnPageSetup`方法。由于页面设置已被更改，我们需要重新计算每个段落以及整个文档。

```cpp
void WordDocument::OnPageSetup(PageSetupInfo pageSetupInfo) { 
  ClearNextFont(); 

  for (Paragraph* paragraphPtr : paragraphList) { 
    GenerateParagraph(paragraphPtr); 
  } 

  CalculateDocument(); 
  UpdateCaret(); 
  UpdateWindow(); 
} 

```

一个小的变化可能会影响整个文档，我们需要计算段落并将它们分配到文档的页面上。

```cpp
void WordDocument::CalculateDocument() { 
  int pageInnerWidth = PageInnerWidth(), 
      pageInnerHeight = PageInnerHeight(), 
      documentHeight = 0, newTotalPages = 1; 

```

我们遍历段落列表，如果当前文档高度与段落的顶部位置不同，我们更新其顶部位置并使其无效。

```cpp
  for (int parIndex = 0; parIndex < paragraphList.Size(); 
       ++parIndex) { 
    Paragraph* paragraphPtr = paragraphList[parIndex]; 

    if (paragraphPtr->Top() != documentHeight) { 
      paragraphPtr->Top() = documentHeight; 
      Invalidate(Rect(0, paragraphPtr->Top(), pageInnerWidth, 
                 paragraphPtr->Top() + paragraphPtr->Height())); 
    } 

```

如果段落被标记为分页，并且它尚未位于页面顶部，则会有一个分页符。

```cpp
    bool pageBreak = paragraphPtr->PageBreak() && 
                     ((paragraphPtr->Top() % pageInnerHeight) != 0); 

```

如果段落的顶部位置加上其高度大于页面高度，则段落无法适应页面的剩余部分。

```cpp
    bool notFitOnPage = 
      (documentHeight > 0) &&
      ((paragraphPtr->Top() + paragraphPtr->Height()) > 
      (newTotalPages * pageInnerHeight)); 

```

如果有分页符，或者如果段落无法适应页面的其余部分，我们需要使页面的其余部分无效，并将段落放置在下一页的顶部。

```cpp
    if (pageBreak || notFitOnPage) { 
      Rect restOfPage(0, documentHeight, pageInnerWidth, 
                      newTotalPages * pageInnerHeight); 
      Invalidate(restOfPage); 
      paragraphPtr->Top() = (newTotalPages++) * pageInnerHeight;
```

由于段落已移动到新位置，我们需要使其新区域无效。

```cpp
      Invalidate(Rect(0, paragraphPtr->Top(), pageInnerWidth,
                 paragraphPtr->Top() + paragraphPtr->Height()));
      documentHeight = paragraphPtr->Top() + 
                       paragraphPtr->Height(); 
    } 

```

如果段落可以适应文档的其余部分，我们只需增加文档高度。

```cpp
    else { 
      documentHeight += paragraphPtr->Height(); 
    } 
  } 

```

在最后一个段落之后，我们需要使最后一页的其余部分无效。

```cpp
  Rect restOfPage(0, documentHeight, pageInnerWidth, 
                  newTotalPages * pageInnerHeight); 
  Invalidate(restOfPage); 

```

如果页数已更改，我们需要使不同的页面无效。

```cpp
  if (totalPages != newTotalPages) { 
    int minTotalPages = min(totalPages, newTotalPages), 
        maxTotalPages = max(totalPages, newTotalPages); 
    Invalidate(Rect(0, minTotalPages * pageInnerHeight, 
                    pageInnerWidth, maxTotalPages * pageInnerHeight));     
    totalPages = newTotalPages; 
    SetVerticalScrollTotalHeight(totalPages * pageInnerHeight);
  }
} 

```

## 绘制和绘图

`OnPaint`方法执行特定于绘制客户端区域的动作，而`OnPrint`方法执行特定于打印的动作。在`StandardDocument`类中，`OnPaint`和`OnPrint`方法的默认行为是调用`OnDraw`方法。

在前几章的应用中，我们只重写了 `OnDraw` 方法，导致无论绘图是在客户端区域发生还是发送到打印机，绘图结果都相同。然而，在这个应用中，我们还重写了 `OnPaint` 方法，该方法用浅灰色填充客户端区域之外的文档部分，并在每对页面之间放置文本 **分页符**，最后调用执行实际文档绘制的 `OnDraw` 方法。

```cpp
void WordDocument::OnPaint(Graphics& graphics) const { 
  int pageInnerWidth = PageInnerWidth(), 
      pageInnerHeight = PageInnerHeight(); 

  int documentInnerHeight = totalPages * pageInnerHeight; 
  Size clientSize = GetClientSize(); 

  if (pageInnerWidth() < clientSize.Width()) {
    int maxHeight = max(documentInnerHeight, clientSize.Height());
    Rect rightRect(pageInnerWidth, 0,
                   clientSize.Width(), maxHeight);
    graphics.FillRectangle(rightRect, LightGray, LightGray); 
  } 

  if (documentInnerHeight() < clientSize.Height()) { 
    Rect bottomRect(0, documentInnerHeight(), 
                    pageInnerWidth(), clientSize.Height()); 
    graphics.FillRectangle(bottomRect, LightGray, LightGray); 
  } 

  OnDraw(graphics, Paint);

  int breakWidth = min(clientSize.Width()), 
      breakHeight = GetCharacterHeight(SystemFont); 
  Size breakSize(breakWidth, breakHeight); 

  for (int pageIndex = 1; pageIndex < totalPages; ++pageIndex) { 
    int line = pageIndex * pageInnerHeight; 
    graphics.DrawLine(Point(0, line), Point(pageInnerWidth, line), 
                      Black); 

    Point topLeft(0, line - (breakHeight / 2)); 
    graphics.DrawText(Rect(topLeft, breakSize), 
                      TEXT("Page Break"), SystemFont,Black,White); 
  } 
} 

```

`OnDraw` 方法绘制 `charList` 列表中的每个字符。当 `OnDraw` 方法由 `OnPaint` 方法调用时，`drawMode` 参数为 `Paint`，而当它由 `OnPrint` 方法调用时，`drawMode` 参数为 `Print`。在先前的应用中，我们忽略了 `drawMode` 方法。然而，在这个应用中，如果由 `OnPaint` 方法调用，我们会在每个带有分页符的段落处绘制一个小方块。

```cpp
void WordDocument::OnDraw(Graphics& graphics, DrawMode drawMode) const { 
  minCharIndex = min(firstMarkIndex, lastMarkIndex), 
  maxCharIndex = max(firstMarkIndex, lastMarkIndex); 

  for (int charIndex = 0; charIndex <= charList.Size() - 1; 
       ++charIndex) { 
    CharInfo charInfo = charList[charIndex]; 
    Point topLeft(0, charInfo.ParagraphPtr()->Top()); 

    Color textColor = charInfo.CharFont().GetColor(); 
    Color backColor = textColor.Inverse(); 

```

如果字符被标记，其文本和背景颜色将被反转。

```cpp
    if ((wordMode == WordMark) && 
        (minCharIndex <= charIndex)&&(charIndex < maxCharIndex)) { 
      swap(textColor, backColor); 
    } 

```

如果字符是换行符，则绘制一个空格代替。

```cpp
    TCHAR tChar = (charInfo.Char() == NewLine)
                  ? Space: charInfo.Char(); 
    TCHAR text[] = {tChar, TEXT('\0')}; 

```

如果字符的矩形位于页面之外，其右边界被设置为页面右边界。

```cpp
    Rect charRect = charList[charIndex].CharRect(); 
    if (charRect.Right() >= pageWidth) { 
      charRect.Right() = pageWidth - 1; 
    } 

```

最后，绘制字符：

```cpp
    graphics.DrawText(topLeft + charRect, text, 
                      charInfo.CharFont(), textColor, backColor); 
  } 

```

实际上，还有一件事：如果 `OnDraw` 方法已经被 `OnPaint` 方法调用，我们会在每个带有分页符的段落的左上角绘制一个小红色方块（2 × 2 毫米）。

```cpp
  if (drawMode == Paint) { 
    for (Paragraph* paragraphPtr : paragraphList) { 
      if (paragraphPtr->PageBreak()) { 
        Point topLeft(0, paragraphPtr->Top()); 
        graphics.FillRectangle(Rect(topLeft, topLeft + 
                                    Size(200, 200)), Red, Red); 
      } 
    } 
  } 
} 

```

## 文件管理

当用户在 **文件** 菜单中选择 **新建** 菜单项时，`StandardDocument` 类会调用 `ClearDocument` 方法；当用户在 **文件** 菜单中选择 **保存** 或 **另存为** 菜单项时，会调用 `WriteDocumentToStream` 方法，而当用户选择 **打开** 菜单项时，会调用 `ReadDocumentFromStream` 方法。

`ClearDocument` 方法通过调用 `DeleteParagraph` 方法删除 `paragraphList` 列表中的每个段落，而 `DeleteParagraph` 方法会删除段落的每一行。实际上，这是我们唯一需要删除的内存，因为它是本应用唯一动态分配的内存。最后，会调用 `InitDocument` 方法，该方法初始化一个空文档。

```cpp
void DeleteParagraph(Paragraph* paragraphPtr) { 
  for (LineInfo* lineInfoPtr : paragraphPtr->LinePtrList()) { 
    delete lineInfoPtr; 
  } 

  delete paragraphPtr; 
} 

void WordDocument::ClearDocument() { 
  nextFont = SystemFont; 

  for (Paragraph* paragraphPtr : paragraphList) { 
    DeleteParagraph(paragraphPtr); 
  } 

  charList.Clear(); 
  paragraphList.Clear(); 
  InitDocument(); 
} 

```

`WriteDocumentToStream` 方法将有关文档的所有信息写入流：`application` 模式（编辑或标记）、编辑字符的索引、第一个和最后一个标记字符的索引、文档中的页数以及下一个字体。想法是文档将以写入时的确切形状打开。

```cpp
bool WordDocument::WriteDocumentToStream(String name, 
                                         ostream& outStream)const{ 
  if (EndsWith(name, TEXT(".wrd")) && 
      WritePageSetupInfoToStream(outStream)){ 
    outStream.write((char*) &wordMode, sizeof wordMode); 
    outStream.write((char*) &editIndex, sizeof editIndex); 

    outStream.write((char*) &firstMarkIndex, 
                    sizeof firstMarkIndex); 
    outStream.write((char*) &lastMarkIndex, sizeof lastMarkIndex); 
    outStream.write((char*) &totalPages, sizeof totalPages); 
    nextFont.WriteFontToStream(outStream); 

    { int charInfoListSize = charList.Size(); 
      outStream.write((char*) &charInfoListSize, 
                      sizeof charInfoListSize); 
      for (CharInfo charInfo : charList) { 
        charInfo.WriteCharInfoToStream(outStream); 
      } 
    } 

    { int paragraphListSize = paragraphList.Size(); 
      outStream.write((char*) &paragraphListSize, 
                      sizeof paragraphListSize); 

      for (const Paragraph* paragraphPtr : paragraphList) { 
        paragraphPtr->WriteParagraphToStream(outStream); 
      } 
    } 
  } 

```

然而，如果文件扩展名是 `.txt`，我们将单词以文本格式保存并丢弃所有格式。

```cpp
  else if (EndsWith(name, TEXT(".txt"))) { 
    for (CharInfo charInfo : charList) { 
      char c = (char) charInfo.Char(); 
      outStream.write(&c, sizeof c); 
    } 
  } 

  return ((bool) outStream); 
} 

```

`ReadDocumentFromStream` 方法读取由 `WriteDocumentToStream` 方法写入的信息。请注意，为了使当前位置可见，在最后会调用 `MakeVisible` 方法。

```cpp
bool WordDocument::ReadDocumentFromStream(String name, 
                                          istream& inStream) { 
  if (EndsWith(name, TEXT(".wrd")) && 
      ReadPageSetupInfoFromStream(inStream)){ 
    inStream.read((char*) &wordMode, sizeof wordMode); 
    inStream.read((char*) &editIndex, sizeof editIndex); 
    inStream.read((char*) &firstMarkIndex, sizeof firstMarkIndex); 
    inStream.read((char*) &lastMarkIndex, sizeof lastMarkIndex); 
    inStream.read((char*) &totalPages, sizeof totalPages); 
    nextFont.ReadFontFromStream(inStream); 

    { charList.Clear(); 
      int charInfoListSize; 
      inStream.read((char*) &charInfoListSize, 
                    sizeof charInfoListSize); 

      for (int count = 0; count < charInfoListSize; ++count) { 
        CharInfo charInfo; 
        charInfo.ReadCharInfoFromStream(inStream); 
        charList.PushBack(charInfo); 
      } 
    } 

    { paragraphList.Clear(); 
      int paragraphListSize; 
      inStream.read((char*) &paragraphListSize, 
                    sizeof paragraphListSize); 

      for (int count = 0; count < paragraphListSize; ++count) { 
        Paragraph* paragraphPtr = new Paragraph(); 
        assert(paragraphPtr != nullptr); 
        paragraphPtr->ReadParagraphFromStream(this, inStream); 
        paragraphList.PushBack(paragraphPtr); 
      } 
    } 
  } 

```

然而，如果文件具有文件扩展名 `.txt`，我们只读取字符，并且所有字符都赋予系统字体。

```cpp
  else if (EndsWith(name, TEXT(".txt"))) { 
    wordMode = WordEdit; 
    editIndex = 0; 
    firstMarkIndex = 0; 
    lastMarkIndex = 0; 
    totalPages = 0; 
    nextFont = SystemFont; 

    Paragraph* paragraphPtr = new Paragraph(0, 0, Left, 0); 
    int charIndex = 0, paragraphIndex = 0; 
    char c; 

    while (inStream >> c) { 
      CharInfo charInfo(paragraphPtr, (TCHAR) c, 
                        SystemFont, ZeroRect); 
      charList.PushBack(charInfo); 

      if (c == '\n') { 
        paragraphPtr->Last() = charIndex;
        for (int index = paragraphPtr->First();
             index <= paragraphPtr->Last(); ++index) {
          charList[index].ParagraphPtr() = paragraphPtr;
        }

        GenerateParagraph(paragraphPtr);
        paragraphList.PushBack(paragraphPtr); 
        Paragraph* paragraphPtr = 
          new Paragraph(charIndex + 1, 0, Left, ++paragraphIndex); 
      } 

      ++charIndex; 
    } 

    paragraphPtr->Last() = charIndex;
    for (int index = paragraphPtr->First();
         index <= paragraphPtr->Last(); ++index) {
      charList[index].ParagraphPtr() = paragraphPtr;
    }

    GenerateParagraph(paragraphPtr); 
    paragraphList.PushBack(paragraphPtr); 
    CalculateDocument(); 
  } 

  MakeVisible(); 
  return ((bool) inStream); 
} 

```

## 剪切、复制和粘贴

**编辑**菜单中的**复制**项在`mark`模式下是启用的：

```cpp
bool WordDocument::CopyEnable() const { 
  return (wordMode == WordMark); 
} 

```

只要之前提到的`CopyEnable`方法返回`true`，我们就始终准备好以每种格式进行复制。因此，我们必须让`IsCopyAsciiReady`、`IsCopyUnicodeReady`和`IsCopyGenericReady`方法返回`true`（如果它们在`StandardDocument`类中返回`false`）。

```cpp
bool WordDocument::IsCopyAsciiReady() const { 
  return true; 
} 

bool WordDocument::IsCopyUnicodeReady() const { 
  return true; 
} 

bool WordDocument::IsCopyGenericReady(int /* format */) const { 
  return true; 
} 

```

`CopyAscii`方法简单地调用`CopyUnicode`方法，因为文本以通用文本格式存储，并在保存到全局剪贴板时转换为 ASCII 和 Unicode。`CopyUnicode`方法遍历标记的段落，并且对于每个标记段落，将存储在段落中的标记文本提取到`textList`参数中。当它遇到换行符时，它将`textList`参数中的当前文本推入。

```cpp
void WordDocument::CopyAscii(vector<String>& textList) { 
  CopyUnicode(textList); 
} 

void WordDocument::CopyUnicode(vector<String>& textList) { 
  int minCharIndex = min(firstMarkIndex, lastMarkIndex), 
      maxCharIndex = max(firstMarkIndex, lastMarkIndex); 

  String text; 
  for (int charIndex = minCharIndex; charIndex < maxCharIndex; 
       ++charIndex) { 
    CharInfo charInfo = charList[charIndex]; 
    text.push_back(charInfo.Char()); 

    if (charInfo.Char() == NewLine) { 
      textList.push_back(text); 
      text.clear(); 
    } 
  } 

  textList.push_back(text); 
} 

```

`CopyGeneric`方法比`CopyUnicode`方法简单。它首先保存要复制的字符数，然后遍历标记的字符（不是段落），然后对每个字符调用`WriteCharInfoToClipboard`方法。这可行，因为`charList`列表中每对段落之间已经通过换行符分隔。我们实际上并不关心格式，因为在这个应用程序中，通用剪切、复制和粘贴操作只有一个格式（`WordFormat`）。

```cpp
void WordDocument::CopyGeneric(int /* format */, 
                               InfoList& infoList) const { 
  int minCharIndex = min(firstMarkIndex, lastMarkIndex), 
      maxCharIndex = max(firstMarkIndex, lastMarkIndex); 
  int copySize = maxCharIndex - minCharIndex; 
  infoList.AddValue<int>(copySize); 

  for (int charIndex = minCharIndex; charIndex < maxCharIndex; 
       ++charIndex) { 
    CharInfo charInfo = charList[charIndex]; 
    charInfo.WriteCharInfoToClipboard(infoList); 
  } 
} 

```

复制和粘贴之间的一个区别在于，当用户选择**剪切**或**复制**时，在先前的`StandardDocument`构造函数中给出的所有三种格式（ASCII、Unicode 和通用）都会被复制。它们的顺序并不重要。另一方面，在粘贴时，`StandardDocument`构造函数会尝试按照构造函数调用中给出的格式顺序粘贴文本。如果它在全局剪贴板中找到一个格式的粘贴信息，它就不会继续检查其他格式。在这个应用程序中，这意味着如果以通用格式（`WordFormat`）复制了文本，那么无论 ASCII 或 Unicode 格式（`AsciiFormat`或`UnicodeFormat`）中是否有文本，都会粘贴该文本。

`PasteAscii`方法调用`PasteUnicode`方法（再次，ASCII 和 Unicode 文本都被转换成通用文本类型），它遍历`textList`参数，并为每个文本插入一个新的段落。请注意，我们没有重写`PasteEnable`方法，因为`StandardDocument`构造函数通过检查是否存在包含在`StandardDocument`构造函数调用中定义的任何格式的剪贴板缓冲区来处理它。

理念是文本列表中的第一和最后一段文本将通过编辑段落的第一个和最后部分合并。潜在剩余的文本将作为段落插入其中。首先，如果存在标记文本，我们确保`edit`模式，并清除`nextFont`参数（将其设置为`SystemFont`）。

```cpp
void WordDocument::PasteUnicode(const vector<String>& textList) { 
  if (wordMode == WordMark) { 
    Delete(firstMarkIndex, lastMarkIndex); 
    EnsureEditStatus(); 
  } 

  else { 
    ClearNextFont(); 
  } 

```

我们从段落列表中移除了编辑段落，这使得稍后插入粘贴的段落更加容易。

```cpp
  Paragraph* paragraphPtr = charList[editIndex].ParagraphPtr(); 
  paragraphList.Erase(paragraphPtr->Index()); 

```

我们为粘贴的字符和段落使用编辑字符的字体和编辑段落的对齐方式。

```cpp
  Alignment alignment = paragraphPtr->AlignmentField(); 
  Font font = charList[editIndex].CharFont(); 

```

我们保存编辑段落剩余字符的数量。我们还保存当前编辑索引，以便计算最终粘贴字符的总数。

```cpp
  int restChars = paragraphPtr->Last() - editIndex, 
      prevEditIndex = editIndex, textListSize = textList.size(); 

```

我们将编辑段落中的每个文本的字符插入。

```cpp
  for (int textIndex = 0; textIndex < textListSize; ++textIndex) { 
    for (TCHAR tChar : textList[textIndex]) { 
      charList.Insert(editIndex++, 
                      CharInfo(paragraphPtr, tChar, font)); 
    } 

```

由于每个文本都将完成一个段落（除了最后一个），我们创建并插入一个新的段落。

```cpp
    if (textIndex < (textListSize - 1)) { 
      charList.Insert(editIndex++, 
                      CharInfo(paragraphPtr, NewLine)); 
      paragraphPtr->Last() = editIndex - 1;
      for (int index = paragraphPtr->First();
           index <= paragraphPtr->Last(); ++index) {
        charList[index].ParagraphPtr() = paragraphPtr;
      }

      GenerateParagraph(paragraphPtr); 
      paragraphList.Insert(paragraphPtr->Index(), paragraphPtr); 
      paragraphPtr = new Paragraph(editIndex, 0, alignment, 
                                   paragraphPtr->Index() + 1); 
    } 

```

对于最后一段文本，我们使用原始编辑段落并更改其最后一个字符索引。

```cpp
    else { 
      paragraphPtr->Last() = editIndex + restChars;
      for (int index = paragraphPtr->First();
           index <= paragraphPtr->Last(); ++index) {
        charList[index].ParagraphPtr() = paragraphPtr;
      }

      GenerateParagraph(paragraphPtr); 
      paragraphList.Insert(paragraphPtr->Index(), paragraphPtr); 
    } 
  } 

```

我们可能还需要更新后续段落的索引，因为可能粘贴了多个段落。由于我们知道至少粘贴了一个字符，我们肯定需要至少修改后续段落的第一和最后一个索引。

```cpp
  int totalAddedChars = editIndex - prevEditIndex; 
  for (int parIndex = paragraphPtr->Index() + 1; 
       parIndex < paragraphList.Size(); ++parIndex) { 
    Paragraph* paragraphPtr = paragraphList[parIndex]; 
    paragraphPtr->Index() = parIndex; 
    paragraphPtr->First() += totalAddedChars; 
    paragraphPtr->Last() += totalAddedChars; 
  } 

  CalculateDocument(); 
  UpdateCaret(); 
  UpdateWindow(); 
} 

```

`PasteGeneric` 方法以类似于先前的 `PasteUnicode` 方法的方式读取并插入存储在剪贴板中的通用段落信息。不同之处在于段落被分隔成换行符，并且每个粘贴的字符都带有自己的字体。

```cpp
void WordDocument::PasteGeneric(int /* format */, 
                                InfoList& infoList) { 
  if (wordMode == WordMark) { 
    Delete(firstMarkIndex, lastMarkIndex); 
    EnsureEditStatus(); 
  } 
  else { 
    ClearNextFont(); 
  } 

```

我们擦除编辑段落以使插入更容易，就像在先前的 `PasteUnicode` 方法中一样。我们使用编辑段落的对齐方式，但不使用编辑字符的字体，因为每个粘贴的字符都有自己的字体。

```cpp
  Paragraph* paragraphPtr = charList[editIndex].ParagraphPtr(); 
  paragraphList.Erase(paragraphPtr->Index()); 
  Alignment alignment = paragraphPtr->AlignmentField(); 

```

我们读取粘贴的大小，即要粘贴的字符数。

```cpp
  int pasteSize, restChars = paragraphPtr->Last() - editIndex; 
  infoList.GetValue<int>(pasteSize); 

```

我们从粘贴缓冲区中读取每个字符并将字符插入到字符列表中。当我们遇到换行符时，我们插入一个新段落。

```cpp
  for (int pasteCount = 0; pasteCount < pasteSize; ++pasteCount) { 
    CharInfo charInfo(paragraphPtr); 
    charInfo.ReadCharInfoFromClipboard(infoList); 
    charList.Insert(editIndex++, charInfo); 

    if (charInfo.Char() == NewLine) { 
      paragraphPtr->Last() = editIndex - 1; 
      GenerateParagraph(paragraphPtr); 
      paragraphList.Insert(paragraphPtr->Index(), paragraphPtr); 
      paragraphPtr = new Paragraph(editIndex, 0, alignment, 
                                   paragraphPtr->Index() + 1); 
      assert(paragraphPtr != nullptr); 
    } 
  } 

  paragraphPtr->Last() = editIndex + restChars; 
  for (int charIndex = editIndex; 
       charIndex <= paragraphPtr->Last(); ++charIndex) { 
    charList[charIndex].ParagraphPtr() = paragraphPtr; 
  } 

```

在插入之前，我们需要计算原始段落。

```cpp
  GenerateParagraph(paragraphPtr); 
  paragraphList.Insert(paragraphPtr->Index(), paragraphPtr); 

```

与前面的 `PasteUnicode` 情况类似，我们可能需要更新后续段落的索引，因为可能粘贴了多个段落。我们还需要修改它们的第一和最后一个索引，因为至少粘贴了一个字符。

```cpp
  for (int parIndex = paragraphPtr->Index() + 1; 
    parIndex < paragraphList.Size(); ++parIndex) { 
    Paragraph* paragraphPtr = paragraphList[parIndex]; 
    paragraphPtr->Index() = parIndex; 
    paragraphPtr->First() += pasteSize; 
    paragraphPtr->Last() += pasteSize; 
  } 

  CalculateDocument(); 
  UpdateCaret(); 
  UpdateWindow(); 
} 

```

## 删除

在 `edit` 模式下，除非字符位于文档的末尾，否则可以删除字符。在 `mark` 模式下，标记的文本始终可以被删除：

```cpp
bool WordDocument::DeleteEnable() const { 
  switch (wordMode) { 
    case WordEdit: 
      return (editIndex < (charList.Size() - 1)); 

    case WordMark: 
      return true; 
  } 

  return false; 
} 

```

在 `edit` 模式下，我们删除编辑字符，在 `mark` 模式下，我们删除标记的文本。在这两种情况下，我们都调用 `Delete` 方法来执行实际的删除操作。

```cpp
void WordDocument::OnDelete() { 
  switch (wordMode) { 
    case WordEdit: 
      ClearNextFont(); 
      Delete(editIndex, editIndex + 1); 
      break; 

    case WordMark: 
      Delete(firstMarkIndex, lastMarkIndex); 
      editIndex = min(firstMarkIndex, lastMarkIndex); 
      wordMode = WordEdit; 
      break; 
  } 

  SetDirty(true); 
  CalculateDocument(); 
  UpdateCaret(); 
  UpdateWindow(); 
} 

```

`Delete` 方法由 `OnDelete`、`EnsureEditStatus`、`PasteUnicode` 和 `PasteGeneric` 方法调用。它删除给定索引之间的字符，这些字符不必按顺序排列。被删除的段落被删除，后续段落被更新。

```cpp
void WordDocument::Delete(int firstIndex, int lastIndex) { 
  int minCharIndex = min(firstIndex, lastIndex), 
      maxCharIndex = max(firstIndex, lastIndex); 

  Paragraph* minParagraphPtr = 
    charList[minCharIndex].ParagraphPtr(); 
  Paragraph* maxParagraphPtr = 
    charList[maxCharIndex].ParagraphPtr(); 

```

被删除的区域至少覆盖两个段落，我们将最大段落的字符设置为指向最小段落，因为它们将被合并。我们还将它们的矩形设置为零，以确保它们将被重新绘制。

```cpp
  if (minParagraphPtr != maxParagraphPtr) {
    for (int charIndex = maxParagraphPtr->First();
         charIndex <= maxParagraphPtr->Last(); ++charIndex) {
      CharInfo& charInfo = charList[charIndex];
      charInfo.ParagraphPtr() = minParagraphPtr;
      charInfo.CharRect() = ZeroRect;
    }
  }
```

字符将从`charList`列表中删除，并且最小段落的最后一个索引被更新。它被设置为最大段落（可能和最小段落相同）的最后一个字符减去要删除的字符数。然后重新生成最小段落。

```cpp
  int deleteChars = maxCharIndex - minCharIndex; 
  minParagraphPtr->Last() = maxParagraphPtr->Last() - deleteChars;
  charList.Remove(minCharIndex, maxCharIndex - 1);
  GenerateParagraph(minParagraphPtr);
```

如果存在，最小和最大段落之间的段落被删除，并且后续段落的索引被设置。我们为每个段落调用`DeleteParagraph`以删除它们的动态分配的内存。

```cpp
  int minParIndex = minParagraphPtr->Index(), 
      maxParIndex = maxParagraphPtr->Index(); 

  if (minParIndex < maxParIndex) {
    for (int parIndex = minParIndex + 1; 
         parIndex <= maxParIndex; ++parIndex) { 
      DeleteParagraph(paragraphList[parIndex]); 
    } 
    paragraphList.Remove(minParIndex + 1, maxParIndex); 
  }

```

最后，我们需要设置后续段落的索引。请注意，无论是否已经删除了任何段落，我们都必须更新第一个和最后一个索引，因为我们至少删除了一个字符。

```cpp
  int deleteParagraphs = maxParIndex - minParIndex;
  for (int parIndex = minParagraphPtr->Index() + 1; 
       parIndex < paragraphList.Size(); ++parIndex) { 
    Paragraph* paragraphPtr = paragraphList[parIndex];
    paragraphPtr->Index() -= deleteParagraphs;
    paragraphPtr->First() -= deleteChars; 
    paragraphPtr->Last() -= deleteChars; 
  } 

```

当删除过程完成后，应用程序设置为`编辑`模式，并且编辑索引被设置为第一个标记的字符。

```cpp
  wordMode = WordEdit; 
  editIndex = minCharIndex;
} 

```

## 分页符

在`编辑`模式下，**分页符**菜单项被启用，并且`OnPageBreak`方法也非常简单。它只是反转编辑段落的分页符状态：

```cpp
bool WordDocument::PageBreakEnable() const { 
  return (wordMode == WordEdit); 
} 

void WordDocument::OnPageBreak() { 
  Paragraph* paragraphPtr = charList[editIndex].ParagraphPtr(); 
  paragraphPtr->PageBreak() = !paragraphPtr->PageBreak(); 
  CalculateDocument(); 
  UpdateCaret(); 
} 

```

## 字体

当用户选择**字体**菜单项并显示字体对话框时，会调用`OnFont`方法。在`编辑`模式下，我们首先需要找到对话框中要使用的默认字体。如果`nextFont`参数是活动的（不等于`SystemFont`），我们使用它。如果不是活动的，我们检查编辑字符是否是段落的第一个字符。如果是第一个字符，我们使用它的字体。如果不是第一个字符，我们使用其前一个字符的字体。这与前面的`UpdateCaret`方法中的相同程序：

```cpp
void WordDocument::OnFont() { 
  switch (wordMode) { 
    case WordEdit: { 
        Font font; 

        if (nextFont != SystemFont) { 
          font = nextFont; 
        } 
        else if (editIndex == 
                 charList[editIndex].ParagraphPtr()->First()) { 
          font = charList[editIndex].CharFont(); 
        } 
        else { 
          font = charList[editIndex - 1].CharFont(); 
        } 

```

如果用户通过选择**确定**来关闭字体对话框，我们将设置`nextFont`参数并重新计算编辑段落。

```cpp
        if (StandardDialog::FontDialog(this, font)) { 
          nextFont = font; 
          Paragraph* paragraphPtr = 
            charList[editIndex].ParagraphPtr(); 
          GenerateParagraph(paragraphPtr); 
          SetDirty(true); 
          CalculateDocument(); 
          UpdateCaret(); 
          UpdateWindow(); 
        } 
      } 
      break; 

```

在`标记`模式下，我们选择具有最低索引的标记字符作为字体对话框中的默认字体。

```cpp
    case WordMark: { 
        int minCharIndex = min(firstMarkIndex, lastMarkIndex), 
            maxCharIndex = max(firstMarkIndex, lastMarkIndex); 
        Font font = charList[minCharIndex].CharFont(); 

```

如果用户选择**确定**，我们将设置每个标记字符的字体并重新计算它们的每个段落。

```cpp
        if (StandardDialog::FontDialog(this, font)) { 
          for (int charIndex = minCharIndex; 
               charIndex < maxCharIndex; ++charIndex) { 
            charList[charIndex].CharFont() = font; 
          } 

          int minParIndex = 
                charList[minCharIndex].ParagraphPtr()->Index(), 
              maxParIndex = 
                charList[maxCharIndex].ParagraphPtr()->Index(); 

          for (int parIndex = minParIndex; 
               parIndex <= maxParIndex; ++parIndex) { 
            Paragraph* paragraphPtr = paragraphList[parIndex]; 
            GenerateParagraph(paragraphPtr); 
          } 

          SetDirty(true); 
          CalculateDocument(); 
          UpdateCaret(); 
          UpdateWindow(); 
        } 
      } 
      break; 
  } 
} 

```

## 对齐

所有单选对齐监听器调用`IsAlignment`方法，所有选择监听器调用`SetAlignment`方法。

```cpp
bool WordDocument::LeftRadio() const { 
  return IsAlignment(Left); 
} 

void WordDocument::OnLeft() { 
  SetAlignment(Left); 
} 

bool WordDocument::CenterRadio() const { 
  return IsAlignment(Center); 
} 

void WordDocument::OnCenter() { 
  SetAlignment(Center); 
} 

bool WordDocument::RightRadio() const { 
  return IsAlignment(Right); 
} 

void WordDocument::OnRight() { 
  SetAlignment(Right); 
} 

bool WordDocument::JustifiedRadio() const { 
  return IsAlignment(Justified); 
} 

void WordDocument::OnJustified() { 
  SetAlignment(Justified); 
} 

```

在`编辑`模式下，`IsAlignment`方法检查编辑段落是否具有给定的对齐方式。在`标记`模式下，它检查所有部分或完全标记的段落是否具有给定的对齐方式。这意味着如果几个段落被标记为不同的对齐方式，则没有对齐菜单项会被标记为单选按钮。

```cpp
bool WordDocument::IsAlignment(Alignment alignment) const { 
  switch (wordMode) { 
    case WordEdit: { 
        Alignment editAlignment = 
          charList[editIndex].ParagraphPtr()->AlignmentField(); 
        return (editAlignment == alignment); 
      } 

    case WordMark: { 
        int minCharIndex = min(firstMarkIndex, lastMarkIndex), 
            maxCharIndex = max(firstMarkIndex, lastMarkIndex); 

        int minParIndex = 
              charList[minCharIndex].ParagraphPtr()->Index(), 
            maxParIndex = 
              charList[maxCharIndex].ParagraphPtr()->Index(); 

        for (int parIndex = minParIndex; parIndex < maxParIndex; 
             ++parIndex) { 
          Alignment markAlignment = 
            paragraphList[parIndex]->AlignmentField(); 

          if (markAlignment != alignment) { 
            return false; 
          } 
        } 

        return true; 
      } 
  } 

  assert(false); 
  return false; 
} 

```

`SetAlignment` 方法设置编辑或标记段落的对齐方式。在 `edit` 模式下，我们只设置编辑段落的对齐方式。请记住，此方法只能在段落具有另一种对齐方式时调用。在 `mark` 模式下，我们遍历标记段落，并设置那些尚未具有所讨论对齐方式的段落的对齐方式。也要记住，此方法只能在至少有一个段落不保持所讨论对齐方式的情况下调用。需要重新计算对齐方式的段落。然而，新的对齐方式不会影响段落的长度，这意味着我们不需要为剩余的段落调用 `CalculateDocument` 方法。

```cpp
void WordDocument::SetAlignment(Alignment alignment) { 
  switch (wordMode) { 
    case WordEdit: { 
        Paragraph* paragraphPtr = 
          charList[editIndex].ParagraphPtr(); 
        paragraphPtr->AlignmentField() = alignment; 
        GenerateParagraph(paragraphPtr); 
        UpdateCaret(); 
      } 
      break; 

    case WordMark: { 
        int minCharIndex = min(firstMarkIndex, lastMarkIndex), 
            maxCharIndex = max(firstMarkIndex, lastMarkIndex); 

        int minParIndex = 
              charList[minCharIndex].ParagraphPtr()->Index(), 
            maxParIndex = 
              charList[maxCharIndex].ParagraphPtr()->Index(); 

        for (int parIndex = minParIndex; parIndex < maxParIndex; 
             ++parIndex) { 
          Paragraph* paragraphPtr = paragraphList[parIndex]; 
          paragraphPtr->AlignmentField() = alignment; 
          GenerateParagraph(paragraphPtr); 
        } 
      } 
      break; 
  } 

  UpdateWindow(); 
} 

```

# 摘要

在本章中，你开始开发一个能够处理单个字符的字处理器。该字处理器支持以下功能：

+   每个字符的独立字体和样式

+   每个段落的左对齐、居中对齐、右对齐和对齐方式

+   分布在页面上的段落

+   滚动和缩放

+   触摸屏

+   使用 ASCII 或 Unicode 文本进行剪切、复制和粘贴，以及应用特定的通用信息

在 第七章，*键盘输入和字符计算* 中，我们将继续讨论键盘输入和字符计算。
