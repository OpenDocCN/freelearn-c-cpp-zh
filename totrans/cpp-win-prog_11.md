# 第十一章。文档

在上一章中，我们探讨了 `Application` 和 `Window` 类的实现，这些类对通用 Windows 应用程序很有用。在本章中，我们将探讨 `Document`、`StandardDocument`、`Menu` 和 `Accelerator` 类的实现，这些类对基于文档的 Windows 应用程序很有用。

# 文档类

在这本书中，**文档**是一个用于通用文档应用程序的窗口，例如这本书的绘图程序、电子表格程序和文字处理程序。`Document` 类实现了之前描述的文档，并且是 `Window` 类的直接子类。它支持光标、脏标志、键盘状态、菜单、快捷键、鼠标滚轮、滚动条和拖放文件。

**Document.h**

```cpp
namespace SmallWindows { 
  extern const Size USLetterPortrait, LineSize; 

```

键盘处于 `insert` 或 `overwrite` 模式之一。

```cpp
  enum KeyboardMode {InsertKeyboard, OverwriteKeyboard}; 

```

与 `Window` 类类似，`Document` 类有一个公共构造函数用于实例化，还有一个受保护的构造函数用于子类。`Document` 类的文档可以接受拖放文件，并且滚动条方法使用行大小：

```cpp
  class Document : public Window { 
    public: 
      Document(CoordinateSystem system, Size pageSize, 
               Window* parentPtr = nullptr, 
               WindowStyle style=OverlappedWindow, 
               WindowShow windowShow = Normal, 
               bool acceptDropFiles = true, 
               Size lineSize = LineSize); 

    protected: 
      Document(String className, CoordinateSystem system, 
               Size pageSize, Window* parentPtr = nullptr, 
               WindowStyle style = OverlappedWindow, 
               WindowShow windowShow = Normal, 
               bool acceptDropFiles = true, 
               Size lineSize = LineSize); 

```

如果窗口已被修改并且需要在关闭前保存（文档已被 *dirty*），则会设置一个脏标志。文档的内容可以根据缩放因子进行缩放；默认值为 1.0。文档的名称通过 `GenerateHeader` 显示在文档标题中，同时显示缩放因子（以百分比表示），如果脏标志为 `true`，则显示一个星号（****）。然而，如果缩放因子为 100%，则不会显示：

```cpp
    public: 
      ~Document(); 

      String GetName() const; 
      void SetName(String name); 
      void SetZoom(double zoom); 
      bool IsDirty() const; 
      void SetDirty(bool dirty); 

    private: 
      void GenerateHeader();  

```

`OnSize` 方法被重写以根据客户端大小修改滚动条的大小。请注意，`OnSize` 的参数是客户端区域的逻辑大小，而不是窗口的大小：

```cpp
    public: 
      virtual void OnSize(Size clientSize); 

```

`OnMouseWheel` 方法被重写，以便每次滚轮点击滚动垂直滚动条一行：

```cpp
      virtual void OnMouseWheel(WheelDirection direction, 
                          bool shiftPressed, bool controlPressed); 

```

`Document` 类支持光标，并且重写了 `OnGainFocus` 和 `OnLoseFocus` 方法以显示或隐藏光标。`SetCaret` 和 `ClearCaret` 方法创建和销毁光标：

```cpp
      void OnGainFocus(); 
      void OnLoseFocus(); 
      void SetCaret(Rect caretLogicalRect); 
      void ClearCaret(); 

```

当光标需要修改时，会调用 `UpdateCaret` 方法，它旨在被重写，并且其默认行为是不做任何事情：

```cpp
      virtual void UpdateCaret() {/* Empty. */} 

```

`SetMenuBar` 方法设置窗口的菜单栏。每次用户选择菜单项或按下快捷键时，都会调用 `OnCommand` 方法，并且在菜单可见之前调用 `CommandInit` 以在菜单项上设置勾选标记或单选按钮，或者启用或禁用它：

```cpp
      void SetMenuBar(Menu& menuBar); 
      void OnCommand(WORD commandId); 
      void OnCommandInit(); 

```

如果构造函数中的 `acceptDropFiles` 参数为 `true`，则文档接受拖放文件。如果用户移动一个或多个文件并将它们拖放到文档窗口中，则会用路径名列表作为参数调用 `OnDropFile`。它旨在被子类重写，并且其默认行为是不做任何事情：

```cpp
      virtual void OnDropFile(vector<String> pathList) 
                             {/* Empty. */} 

```

`GetKeyboardMode`和`SetKeyboardMode`方法设置和获取`keyboard`模式。当`keyboard`模式改变时，会调用`OnKeyboardMode`方法；它旨在被重写，并且默认行为是不做任何事情：

```cpp
      KeyboardMode GetKeyboardMode() const {return keyboardMode;} 
      void SetKeyboardMode(KeyboardMode mode) 
                          {keyboardMode = mode;} 
      virtual void OnKeyboardMode(KeyboardMode mode) 
                                 {/* Empty. */} 

```

`OnHorizontalScroll`和`OnVerticalScroll`方法处理滚动消息。滚动条根据消息设置进行设置：

```cpp
      virtual void OnHorizontalScroll(WORD flags,WORD thumbPos=0); 
      virtual void OnVerticalScroll(WORD flags, WORD thumbPos =0); 

```

`KeyToScroll`方法接受一个键，并根据键以及是否按下***Shift***或***Ctrl***键执行适当的滚动条操作。例如，***Page Up***键将垂直滚动条向上移动一页：

```cpp
      virtual bool KeyToScroll(WORD key, bool shiftPressed, 
                               bool controlPressed); 

```

以下方法设置或获取逻辑位置、行大小、页面大小以及水平和垂直滚动条的总大小：

```cpp
      void SetHorizontalScrollPosition(int scrollPos); 
      int GetHorizontalScrollPosition() const; 
      void SetVerticalScrollPosition(int scrollPos); 
      int GetVerticalScrollPosition() const; 

      void SetHorizontalScrollLineWidth(int lineWidth); 
      int GetHorizontalScrollLineHeight() const; 
      void SetVerticalScrollLineHeight(int lineHeight); 
      int GetVerticalScrollLineHeight() const; 

      void SetHorizontalScrollPageWidth(int pageWidth); 
      int GetHorizontalScrollPageWidth() const; 
      void SetVerticalScrollPageHeight(int pageHeight); 
      int GetVerticalScrollPageHeight() const; 

      void SetHorizontalScrollTotalWidth(int scrollWidth); 
      int GetHorizontalScrollTotalWidth() const; 
      void SetVerticalScrollTotalHeight(int scrollHeight); 
      int GetVerticalScrollTotalHeight() const; 

```

命令映射存储文档的菜单项；对于每个菜单项，存储选择、启用、检查和单选按钮监听器：

```cpp
    public: 
      map<WORD,Command>& CommandMap() {return commandMap;} 

```

加速器集合包含文档的加速器，无论它是常规键还是虚拟键（例如，***F2***，***Home***，或***Delete***）以及是否按下***Ctrl***，***Shift***或***Alt***键。该集合由`Application`中的消息循环使用：

```cpp
      list<ACCEL>& AcceleratorSet() {return acceleratorSet;} 

    private: 
      map<WORD, Command> commandMap; 
      list<ACCEL> acceleratorSet; 

```

`name`字段是显示在窗口顶部的文档名称；当光标可见时，`caretPresent`为`true`：

```cpp
      String name; 
      bool caretPresent = false; 

```

当用户按下箭头键之一时，会调用`OnKeyDown`。然而，如果`OnKeyDown`返回`false`，则滚动条会改变；在这种情况下，我们需要`lineSize`来定义要滚动的行的大小：

```cpp
      Size lineSize; 

```

当用户在未保存的情况下更改文档时，`dirtyFlag`字段为`true`，导致**保存**菜单项被启用，并在关闭窗口或退出应用程序时询问用户是否保存文档：

```cpp
      bool dirtyFlag = false; 

```

`menuBarHandle`方法是处理文档窗口菜单栏的 Win32 API 函数：

```cpp
      HMENU menuBarHandle; 

```

键盘可以保持`insert`或`overwrite`模式，该模式存储在`keyboardMode`中：

```cpp
      KeyboardMode keyboardMode = InsertKeyboard; 
  }; 

```

当文档窗口接收到消息时，会调用`DocumentProc`方法，类似于`Window`类中的`WindowProc`方法：

```cpp
  LRESULT CALLBACK DocumentProc(HWND windowHandle, UINT message, 
                           WPARAM wordParam,LPARAM longParam); 

```

当窗口接收到`WM_DROPFILES`消息时，`ExtractPathList`方法会提取拖放文件的路径：

```cpp
  vector<String> ExtractPathList(WORD wordParam); 
}; 

```

## 初始化

第一个`Document`构造函数接受坐标系、页面大小、父窗口、样式、外观、文档是否接受拖放文件以及行大小作为其参数。在竖直模式（站立）下，美国信函纸张的大小为 215.9 * 279.4 毫米。一行（在滚动行时由`KeyToScroll`使用）在水平和垂直方向上都是 5 毫米。由于逻辑单位是毫米的一百分之一，我们将每个度量乘以一百。

**Document.cpp**

```cpp
#include "SmallWindows.h" 

namespace SmallWindows { 
  const Size USLetterPortrait(21590, 27940), LineSize(500, 500); 

```

第一个构造函数使用名为`Document`的`Windows`类作为第一个参数调用第二个构造函数：

```cpp
  Document::Document(CoordinateSystem system, Size pageSize, 
                     Window* parentPtr /* = nullptr */, 
                     WindowStyle style /* = OverlappedWindow */, 
                     WindowShow windowShow /* = Normal */, 
                     bool acceptDropFiles /* = true */, 
                     Size lineSize /* = LineSize */) 
   :Document::Document(TEXT("document"), system, pageSize, 
                       parentPtr, style, windowShow, 
                       acceptDropFiles, lineSize) { 
    // Empty. 
  } 

```

第二个构造函数与第一个构造函数具有相同的参数，除了它将 `Windows` 类名作为其第一个参数插入：

```cpp
  Document::Document(String className, CoordinateSystem system, 
                     Size pageSize, Window* parentPtr/*=nullptr*/, 
                     WindowStyle style /* = OverlappedWindow */, 
                     WindowShow windowShow /* = Normal */, 
                     bool acceptDropFiles /* = true */, 
                     Size lineSize /* = LineSize */) 
   :Window(className, system, pageSize, parentPtr, 
           style, NoStyle, windowShow), 

```

滚动条的范围和页面大小存储在窗口的滚动条设置中。但是，行的尺寸需要存储在 `lineSize` 中：

```cpp
    lineSize(lineSize) { 

```

标题出现在文档窗口的顶部栏上：

```cpp
    GenerateHeader(); 

```

滚动条的默认位置是 `0`：

```cpp
    SetHorizontalScrollPosition(0); 
    SetVerticalScrollPosition(0); 

```

滚动条的大小是页面的逻辑宽度和高度：

```cpp
    SetHorizontalScrollTotalWidth(pageSize.Width()); 
    SetVerticalScrollTotalHeight(pageSize.Height()); 

```

滚动条的页面大小表示文档的可见部分，即客户端区域的逻辑大小：

```cpp
    Size clientSize = GetClientSize(); 
    SetHorizontalScrollPageWidth(clientSize.Width()); 
    SetVerticalScrollPageHeight(clientSize.Height()); 

```

Win32 API 函数 `DragAcceptFiles` 使窗口接受拖放文件。请注意，我们需要将 C++ 的 `bool` 类型 `acceptDropFiles` 转换为 Win32 API 的 `BOOL` 类型的 `TRUE` 或 `FALSE` 值：

```cpp
    ::DragAcceptFiles(windowHandle, 
                      acceptDropFiles ? TRUE : FALSE); 
  } 

```

析构函数如果存在，会销毁光标：

```cpp
  Document::~Document() { 
    if (caretPresent) { 
      ::DestroyCaret(); 
    } 
  } 

```

## 文档标题

`GetName` 方法简单地返回名称。然而，`SetName` 设置名称并重新生成文档窗口的标题。同样，`SetZoom` 和 `SetDirty`：它们设置缩放因子和脏标志，然后重新生成标题：

```cpp
  String Document::GetName() const { 
    return name; 
  }  

  void Document::SetName(String name) { 
    this->name = name; 
    GenerateHeader(); 
  }  

  void Document::SetZoom(double zoom) { 
    Window::SetZoom(zoom); 
    GenerateHeader(); 
  } 

  bool Document::IsDirty() const { 
    return dirtyFlag; 
  }  

  void Document::SetDirty(bool dirty) { 
    dirtyFlag = dirty; 
    GenerateHeader(); 
  } 

```

文档的标题包括其名称，是否设置了脏标志（由星号表示），以及缩放状态（以百分比表示），除非它是 100%。

```cpp
  void Document::GenerateHeader() { 
    String headerName = name.empty() ? TEXT("[No Name]") : name, 
           dirtyText = dirtyFlag ? TEXT("*") : TEXT(""); 
    int zoomPerCent = (int) (100 * GetZoom()); 

    if (zoomPerCent!= 100) { 
      String zoomText = 
        TEXT(" ") + to_String(zoomPerCent) + TEXT("%"); 
      SetHeader(headerName + dirtyText + zoomText); 
    } 
    else { 
      SetHeader(headerName + dirtyText); 
    } 
  } 

```

`OnSize` 方法根据新的客户端大小修改水平和垂直滚动条的页面大小：

```cpp
  void Document::OnSize(Size clientSize) { 
    SetHorizontalScrollPageWidth(clientSize.Width()); 
    SetVerticalScrollPageHeight(clientSize.Height()); 
  } 

```

## 光标

如第一章中所述，*简介*，光标是表示下一个输入字符位置的标记。它在 `insert` 模式下是一个细长的垂直条，在 `overwrite` 模式下是一个块。`OnGainFocus` 和 `OnLoseFocus` 方法显示和隐藏光标（如果存在）：

```cpp
  void Document::OnGainFocus() { 
    if (caretPresent) { 
      ::ShowCaret(windowHandle); 
    } 
  } 

  void Document::OnLoseFocus() { 
    if (caretPresent) { 
      ::HideCaret(windowHandle); 
    } 
  } 

```

`SetCaret` 方法显示具有给定尺寸的光标。如果已经存在光标，则将其销毁：

```cpp
  void Document::SetCaret(Rect caretLogicalRect) { 
    if (caretPresent) { 
      ::DestroyCaret(); 
    } 

```

光标的大小必须以设备单位给出；存在风险，即 `LogicalToDevice` 调用将宽度四舍五入为零（在垂直条的情况下），在这种情况下，宽度设置为 1：

```cpp
    Rect deviceCaretRect = LogicalToDevice(caretLogicalRect); 
    if (deviceCaretRect.Width() == 0) {       
      deviceCaretRect.Right() = deviceCaretRect.Left() + 1; 
    } 

```

新的光标是通过 Win32 API 函数 `CreateCaret`、`SetCaretPos` 和 `ShowCaret` 创建的：

```cpp
    ::CreateCaret(windowHandle, nullptr, deviceCaretRect.Width(), 
                  deviceCaretRect.Height()); 
    ::SetCaretPos(deviceCaretRect.Left(), deviceCaretRect.Top()); 
    ::ShowCaret(windowHandle); 

    caretPresent = true; 
  } 

```

如果存在，`ClearCaret` 方法会销毁光标：

```cpp
  void Document::ClearCaret() { 
    if (caretPresent) { 
      ::DestroyCaret(); 
    }  
    caretPresent = false; 
  } 

```

## 鼠标滚轮

当用户移动鼠标滚轮时，垂直滚动条向上或向下移动一行（如果他们没有按 *Ctrl* 键）：

```cpp
  void Document::OnMouseWheel(WheelDirection wheelDirection, 
                        bool shiftPressed, bool controlPressed){ 
    if (controlPressed) { 
      switch (wheelDirection) { 
        case WheelUp: 
          OnVerticalScroll(SB_LINEUP); 
          break; 

        case WheelDown: 
          OnVerticalScroll(SB_LINEDOWN); 
          break; 
      } 
    } 

```

如果用户按下 ***Ctrl*** 键，则客户端区域将被缩放。允许的范围是 10% 到 1,000%：

```cpp
    else { 
      switch (wheelDirection) { 
        case WheelUp: 
          SetZoom(min(10.0, 1.11 * GetZoom())); 
          break; 

        case WheelDown: 
          SetZoom(max(0.1, 0.9 * GetZoom())); 
          break; 
      } 
    } 

```

由于垂直滚动条位置已修改，我们需要重新绘制整个客户端区域：

```cpp
    Invalidate(); 
    UpdateWindow(); 
    UpdateCaret(); 
  } 

```

## 菜单栏

文档的菜单栏通过调用 Win32 API 函数 `SetMenu` 来设置，该函数处理文档窗口和菜单栏；`menuBarHandle` 在 `OnCommandInit` 中启用或标记菜单项时使用，如下所示：

```cpp
  void Document::SetMenuBar(Menu& menuBar) { 
    menuBarHandle = menuBar.menuHandle; 
    ::SetMenu(windowHandle, menuBarHandle); 
  } 

```

当用户选择菜单项或加速键时，会调用 `OnCommand` 方法。它查找并调用与给定命令标识符关联的选择监听器：

```cpp
  void Document::OnCommand(WORD commandId) { 
    Command command = commandMap[commandId]; 
    command.Selection()(this); 
  } 

```

在菜单变得可见之前会调用`OnCommandInit`方法。它会遍历每个菜单项，并为每个菜单项决定是否应该用勾选标记或单选按钮进行标注，或者启用或禁用：

```cpp
  void Document::OnCommandInit() { 
    for (pair<WORD,Command> pair : commandMap) { 
      WORD commandId = pair.first; 
      Command command = pair.second; 

```

如果启用监听器不为空，我们调用它并将启用标志设置为`MF_ENABLED`或`MF_GRAYED`（禁用）：

```cpp
      if (command.Enable() != nullptr) { 
        UINT enableFlag = command.Enable()(this) ? 
                          MF_ENABLED : MF_GRAYED; 
        ::EnableMenuItem(menuBarHandle, commandId, 
                         MF_BYCOMMAND | enableFlag); 
      } 

```

如果勾选或单选按钮监听器不为空，我们调用它们并将`checkflag`或`radioFlag`设置为：

```cpp
      { bool checkFlag = false; 
        if (command.Check() != nullptr) { 
          BoolListener checkListener = command.Check(); 
          checkFlag = checkListener(this); 
        } 

        bool radioFlag = false; 
        if (command.Radio() != nullptr) { 
          BoolListener radioListener = command.Radio(); 
          radioFlag = radioListener(this); 
        } 

```

如果`checkFlag`或`radioFlag`中的任何一个为`true`，我们检查菜单项。菜单项是否因此被标注为勾选标记或单选按钮，是在将菜单项添加到菜单时决定的，这在下一节的`Menu`类中描述。`Menu`还指出，至少有一个勾选标记和单选按钮监听器必须为空，因为不可能同时用勾选标记和单选按钮标注菜单项：

```cpp
        UINT checkFlags = (checkFlag | radioFlag) ? 
                          MF_CHECKED : MF_UNCHECKED; 
        ::CheckMenuItem(menuBarHandle, commandId, 
                        MF_BYCOMMAND | checkFlags); 
      } 
    } 
  } 

```

## 滚动条

每次用户通过点击滚动条箭头、滚动条本身或拖动滚动滑块进行滚动时，都会调用`OnHorizontalScroll`和`OnVerticalScroll`方法。

`scrollPos`字段存储当前的滚动条设置。`scrollLine`变量是行的大小，`scrollPage`是页面的大小（表示文档可见部分的逻辑大小，等于客户端区域的大小），而`scrollSize`是滚动条的总大小（表示文档的逻辑大小）：

```cpp
  void Document::OnHorizontalScroll(WORD flags, 
                                    WORD thumbPos /*= 0 */) { 
    int scrollPos = GetHorizontalScrollPosition(), 
        scrollLine = GetHorizontalScrollLineHeight(), 
        scrollPage = GetHorizontalScrollPageWidth(), 
        scrollSize = GetHorizontalScrollTotalWidth(); 

    switch (flags) { 
      case SB_LEFT: 
        SetHorizontalScrollPosition(0); 
        break; 

```

在向左移动的情况下，我们需要验证新的滚动位置是否不低于零：

```cpp
      case SB_LINELEFT: 
        SetHorizontalScrollPosition(max(0, scrollPos - 
                                           scrollLine)); 
        break; 

      case SB_PAGELEFT: 
        SetHorizontalScrollPosition(max(0, scrollPos - 
                                           scrollPage)); 
        break; 

```

在向右移动的情况下，我们需要验证滚动位置是否不超过滚动条大小：

```cpp
      case SB_LINERIGHT: 
        SetHorizontalScrollPosition(min(scrollPos + scrollLine, 
                                        scrollSize - scrollLine)); 
        break; 

      case SB_PAGERIGHT: 
        SetHorizontalScrollPosition(min(scrollPos + scrollLine, 
                                        scrollSize - scrollPage)); 
        break; 

      case SB_RIGHT: 
        SetHorizontalScrollPosition(scrollSize - scrollPage); 
        break; 

```

如果用户拖动滚动条滑块，我们只需设置新的滚动位置。消息之间的区别在于，当用户拖动滑块时，会持续发送`SB_THUMBTRACK`，而`SB_THUMBPOSITION`是在用户释放鼠标按钮时发送的：

```cpp
      case SB_THUMBTRACK: 
      case SB_THUMBPOSITION: 
        SetHorizontalScrollPosition(thumbPos); 
        break; 
    } 
  } 

```

垂直滚动条的运动方式与水平滚动条的运动方式相同：

```cpp
  void Document::OnVerticalScroll(WORD flags, 
                                  WORD thumbPos /* = 0 */) { 
    int scrollPos = GetVerticalScrollPosition(), 
        scrollLine = GetVerticalScrollLineHeight(), 
        scrollPage = GetVerticalScrollPageHeight(), 
        scrollSize = GetVerticalScrollTotalHeight(); 

    switch (flags) { 
      case SB_TOP: 
        SetVerticalScrollPosition(0); 
        break; 

      case SB_LINEUP: 
        SetVerticalScrollPosition(max(0, scrollPos - scrollLine)); 
        break; 

      case SB_PAGEUP: 
        SetVerticalScrollPosition(max(0, scrollPos - scrollPage)); 
        break; 

      case SB_LINEDOWN: 
        SetVerticalScrollPosition(min(scrollPos + scrollLine, 
                                      scrollSize - scrollLine)); 
        break; 

      case SB_PAGEDOWN: 
        SetVerticalScrollPosition(min(scrollPos + scrollLine, 
                                      scrollSize - scrollPage)); 
        break; 

      case SB_BOTTOM: 
        SetVerticalScrollPosition(scrollSize - scrollPage); 
        break; 

      case SB_THUMBTRACK: 
      case SB_THUMBPOSITION: 
        SetVerticalScrollPosition(thumbPos); 
        break; 
    } 
  } 

```

当用户按下键时调用`KeyToScroll`函数。它检查按键，执行适当的滚动操作，如果使用了该键，则返回`true`，表示这一点：

```cpp
  bool Document::KeyToScroll(WORD key, bool shiftPressed, 
                             bool controlPressed) { 
    switch (key) { 
      case KeyUp: 
        OnVerticalScroll(SB_LINEUP); 
        return true; 

      case KeyDown: 
        OnVerticalScroll(SB_LINEDOWN); 
        return true; 

      case KeyPageUp: 
        OnVerticalScroll(SB_PAGEUP); 
        return true; 

      case KeyPageDown: 
        OnVerticalScroll(SB_PAGEDOWN); 
        return true; 

      case KeyLeft: 
        OnHorizontalScroll(SB_LINELEFT); 
        return true; 

      case KeyRight: 
        OnHorizontalScroll(SB_LINERIGHT); 
        return true; 

      case KeyHome: 
        OnHorizontalScroll(SB_LEFT); 
        if (controlPressed) { 
          OnVerticalScroll(SB_TOP); 
        } 
        return true; 

      case KeyEnd: 
        OnHorizontalScroll(SB_RIGHT); 
        if (controlPressed) { 
          OnVerticalScroll(SB_BOTTOM); 
        } 
        return true; 
    } 

    return false; 
  } 

```

如果滚动位置已更改，我们通过调用 Win32 API 函数`SetScrollPos`来设置新的滚动位置，并更新窗口和光标：

```cpp
  void Document::SetHorizontalScrollPosition(int scrollPos) { 
    if (scrollPos != GetHorizontalScrollPosition()) { 
      ::SetScrollPos(windowHandle, SB_HORZ, scrollPos, TRUE); 
      Invalidate(); 
      UpdateWindow(); 
      UpdateCaret(); 
    } 
  } 

```

Win32 API 函数`GetScrollPos`返回当前的滚动条位置：

```cpp
  int Document::GetHorizontalScrollPosition() const { 
    return ::GetScrollPos(windowHandle, SB_HORZ); 
  } 

```

垂直滚动位置的方法与水平滚动条的方法工作方式相同：

```cpp
  void Document::SetVerticalScrollPosition(int scrollPos) { 
    if (scrollPos != GetVerticalScrollPosition()) { 
      ::SetScrollPos(windowHandle, SB_VERT, scrollPos, TRUE); 
      Invalidate(); 
      UpdateWindow(); 
      UpdateCaret(); 
    } 
  }  

  int Document::GetVerticalScrollPosition() const { 
    return ::GetScrollPos(windowHandle, SB_VERT); 
  } 

```

`SetHorizontalScrollLineWidth`、`GetHorizontalScrollLineHeight`、`SetVerticalScrollLineHeight`和`GetVerticalScrollLineHeight`方法没有 Win32 API 的对应方法。相反，我们在`lineSize`字段中存储滚动行的尺寸：

```cpp
  void Document::SetHorizontalScrollLineWidth(int lineWidth) { 
    lineSize.Width() = lineWidth; 
  } 

  int Document::GetHorizontalScrollLineHeight() const { 
    return lineSize.Width(); 
  } 

  void Document::SetVerticalScrollLineHeight(int lineHeight) { 
    lineSize.Height() = lineHeight; 
  } 

  int Document::GetVerticalScrollLineHeight() const { 
    return lineSize.Height(); 
  } 

```

`SetHorizontalScrollPageWidth`、`GetHorizontalScrollPageWidth`、`SetVerticalScrollPageHeight` 和 `GetVerticalScrollPageHeight` 方法没有直接的 Win32 API 对应函数。然而，`GetScrollInfo` 和 `SetScrollInfo` 函数处理一般的滚动信息，我们可以设置和提取页面信息：

```cpp
  void Document::SetHorizontalScrollPageWidth(int pageWidth) { 
    SCROLLINFO scrollInfo = {sizeof(SCROLLINFO), SIF_PAGE}; 
    scrollInfo.nPage = pageWidth; 

    ::SetScrollInfo(windowHandle, SB_HORZ, &scrollInfo, TRUE); 
  } 

  int Document::GetHorizontalScrollPageWidth() const { 
    SCROLLINFO scrollInfo = {sizeof(SCROLLINFO), SIF_PAGE}; 
    ::GetScrollInfo(windowHandle, SB_HORZ, &scrollInfo); 
    return scrollInfo.nPage; 
  } 

  void Document::SetVerticalScrollPageHeight(int pageHeight) { 
    SCROLLINFO scrollInfo = {sizeof(SCROLLINFO), SIF_PAGE}; 
    scrollInfo.nPage = pageHeight; 
    ::SetScrollInfo(windowHandle, SB_VERT, &scrollInfo, TRUE); 
  } 

  int Document::GetVerticalScrollPageHeight() const { 
    SCROLLINFO scrollInfo = {sizeof(SCROLLINFO), SIF_PAGE}; 
    ::GetScrollInfo(windowHandle, SB_VERT, &scrollInfo); 
    return scrollInfo.nPage; 
  } 

```

`SetHorizontalScrollTotalWidth`、`GetHorizontalScrollTotalWidth`、`SetVerticalScrollTotalHeight` 和 `GetVerticalScrollTotalHeight` 方法调用 Win32 API 函数 `SetScrollRange` 和 `GetScrollRange`，这些函数设置和获取滚动值的最小和最大值。然而，我们忽略最小值，因为它始终为 0：

```cpp
  void Document::SetHorizontalScrollTotalWidth(int scrollWidth) { 
    ::SetScrollRange(windowHandle, SB_HORZ, 0, scrollWidth, TRUE); 
  } 

  int Document::GetHorizontalScrollTotalWidth() const { 
    int minRange, maxRange; 
    ::GetScrollRange(windowHandle, SB_HORZ, &minRange, &maxRange); 
    return maxRange; 
  } 

  void Document::SetVerticalScrollTotalHeight(int scrollHeight) { 
    ::SetScrollRange(windowHandle, SB_VERT, 0, scrollHeight,TRUE); 
  } 

  int Document::GetVerticalScrollTotalHeight() const { 
    int minRange, maxRange; 
    ::GetScrollRange(windowHandle, SB_VERT, &minRange, &maxRange); 
    return maxRange; 
  } 

```

## `DocumentProc` 方法

每当文档（`Document` 类的文档）收到消息时，都会调用 `DocumentProc` 方法。如果它使用该消息，则返回 0；否则，调用上一章中描述的 `WindowProc` 方法来进一步处理该消息：

```cpp
LRESULT CALLBACK DocumentProc(HWND windowHandle, UINT message, 
                              WPARAM wordParam, LPARAM longParam){ 

```

我们在 `Window` 类的 `WindowMap` 中查找窗口，并且只有当窗口是 `Document` 对象时才采取行动：

```cpp
    if ((windowHandle != nullptr) &&
        (WindowMap.count(windowHandle) == 1)) {
      Document* documentPtr = 
        dynamic_cast<Document*>(WindowMap[windowHandle]); 

      if (documentPtr != nullptr) { 
        switch (message) { 

```

如果单词参数的第九位被设置，鼠标滚轮的方向向下：

```cpp
          case WM_MOUSEWHEEL: { 
              bool down = (HIWORD(wordParam) & 0x0100) != 0; 
              WheelDirection wheelDirection = 
                down ? WheelDown : WheelUp; 
              bool shiftPressed = (::GetKeyState(VK_SHIFT) < 0); 
              bool controlPressed = (::GetKeyState(VK_CONTROL)<0); 
              documentPtr->OnMouseWheel(wheelDirection, 
                             shiftPressed, controlPressed); 
            } 
            return 0; 

```

按键消息同时检查 ***Insert*** 键并调用 `OnKeyDown` 和 `KeyToScroll`，如果其中一个使用该键则返回 0：

```cpp
          case WM_KEYDOWN: { 
              WORD key = wordParam; 

```

如果用户按下 *Insert* 键，则键盘模式在插入和覆盖模式之间切换。`SetKeyboardMode` 设置键盘模式并调用 `OnKeyboardMode`，该函数旨在被子类覆盖以通知应用程序变化：

```cpp
              if (key == KeyInsert) { 
                switch (documentPtr->GetKeyboardMode()) { 

                  case InsertKeyboard: 
                    documentPtr-> 
                      SetKeyboardMode(OverwriteKeyboard); 
                    documentPtr-> 
                      OnKeyboardMode(OverwriteKeyboard); 
                    break; 

                  case OverwriteKeyboard: 
                    documentPtr->SetKeyboardMode(InsertKeyboard); 
                    documentPtr->OnKeyboardMode(InsertKeyboard); 
                    break; 
                } 

                return 0; 
              } 

```

如果用户没有按下 ***Insert*** 键，我们检查 `OnKeyDown` 是否使用该键（并因此返回 `true`）。如果没有，我们则检查 `KeyToScroll` 是否使用该键。如果 `OnKeyDown` 或 `KeyToScroll` 返回 `true`，则返回 0：

```cpp
              else { 
                bool shiftPressed = (::GetKeyState(VK_SHIFT) < 0); 
                bool controlPressed=(::GetKeyState(VK_CONTROL)<0); 

                if (documentPtr->OnKeyDown(wordParam,shiftPressed, 
                                           controlPressed) || 
                    documentPtr->KeyToScroll(key, shiftPressed, 
                                           controlPressed)) { 
                return 0; 
                } 
              } 
            } 
            break; 

```

当用户选择菜单项时，会发送 `WM_COMMAND` 事件，在菜单可见之前发送 `WM_INITMENUPOPUP` 事件。通过调用 `OnCommand` 来处理消息，该函数执行与菜单项连接的选项监听器，以及调用 `OnCommandInit`，在它们变得可见之前使用复选标记或单选按钮启用或注释菜单项：

```cpp
          case WM_COMMAND: 
            documentPtr->OnCommand(LOWORD(wordParam)); 
            return 0; 

          case WM_INITMENUPOPUP: 
            documentPtr->OnCommandInit(); 
            return 0; 

```

当用户将一组文件拖放到窗口中时，在调用 `OnDropFile` 之前，我们需要提取它们的路径。`ExtractPath` 方法从拖放中提取文件的路径并返回路径列表，该列表被发送到 `OnDropFile`：

```cpp
          case WM_DROPFILES: { 
              vector<String> pathList = 
                ExtractPathList(wordParam); 
              documentPtr->OnDropFile(pathList); 
            } 
            return 0; 

```

通过调用相应的匹配方法来处理 `WM_HSCROLL` 和 `WM_VSCROLL` 消息：

```cpp
          case WM_HSCROLL: { 
              WORD flags = LOWORD(wordParam), 
                   thumbPos = HIWORD(wordParam); 
              documentPtr->OnHorizontalScroll(flags, thumbPos); 
            } 
            return 0; 

          case WM_VSCROLL: { 
              WORD flags = LOWORD(wordParam), 
                   thumbPos = HIWORD(wordParam); 
              documentPtr->OnVerticalScroll(flags, thumbPos); 
            } 
            return 0; 
        } 
      } 
    } 

```

最后，如果消息没有被 `DocumentProc` 捕获，则调用上一章中描述的 `WindowProc` 方法来进一步处理该消息：

```cpp
    return WindowProc(windowHandle, message, 
                      wordParam, longParam); 
  } 

```

`ExtractPathList` 方法通过调用 Win32 API 函数 `DragQueryFile` 提取拖放文件的路径并返回路径列表：

```cpp
  vector<String> ExtractPathList(WORD wordParam) { 
    vector<String> pathList; 
    HDROP dropHandle = (HDROP) wordParam; 

```

`DragQueryFile` 方法在第二个参数为 `0xFFFFFFFF` 时返回文件数量：

```cpp
    int size = 
      ::DragQueryFile(dropHandle, 0xFFFFFFFF, nullptr, 0);  
    for (int index = 0; index < size; ++index) { 

```

当第二个参数是一个基于零的索引，第三个参数为 null 时，`DragQueryFile`方法返回路径字符串的大小：

```cpp
      int bufferSize =  
        ::DragQueryFile(dropHandle, index, nullptr, 0) + 1; 
      TCHAR* path = new TCHAR[bufferSize]; 
      assert(path!= nullptr); 

```

当第三个参数是指向文本缓冲区的指针而不是 null 时，`DragQueryFile`方法会复制路径本身：

```cpp
      assert(::DragQueryFile(dropHandle, index, 
                             path, bufferSize) != 0); 
      pathList.push_back(String(path)); 
      delete [] path; 
    } 

    return pathList; 
  } 
}; 

```

# `Menu`类

`Menu`类处理一个菜单，由菜单项列表、分隔条或子菜单组成。当添加菜单项时，其命令信息存储在文档的命令映射中，以便在接收`WM_COMMAND`和`WM_INITCOMMAND`消息时使用。如果菜单项文本包含一个快捷键，它将被添加到文档的加速器集中。`Command`类是一个辅助类，持有指向菜单项的指针：选择、启用、检查和单选监听器。

**Command.h**

```cpp
namespace SmallWindows { 
  typedef void (*VoidListener)(void* sourcePtr); 
  typedef bool (*BoolListener)(void* sourcePtr);  

  class Command { 
    public: 
      Command(); 
      Command(VoidListener selection, BoolListener enable, 
              BoolListener check, BoolListener radio); 

      VoidListener Selection() const {return selection;} 
      BoolListener Enable() const {return enable;} 
      BoolListener Check() const {return check;} 
      BoolListener Radio() const {return radio;} 

    private: 
      VoidListener selection; 
      BoolListener enable, check, radio; 
  }; 
}; 

```

**Command.cpp**

```cpp
#include "SmallWindows.h"

namespace SmallWindows { 
  Command::Command() 
   :selection(nullptr), 
    enable(nullptr), 
    check(nullptr), 
    radio(nullptr) { 
    // Empty. 
  } 

  Command::Command(VoidListener selection, BoolListener enable, 
                   BoolListener check, BoolListener radio) 
   :selection(selection), 
    enable(enable), 
    check(check), 
    radio(radio) { 
    // Empty. 
  } 
}; 

```

菜单和加速器监听器不是常规方法。它们通过`DECLARE_BOOL_LISTENER`和`DECLARE_VOID_LISTENER`宏声明（它们不需要定义）。这是因为我们无法直接在未知类中调用非静态方法。因此，我们让宏声明一个不带参数的非静态方法，并定义一个带有`void`指针参数的静态方法，该方法调用非静态方法。宏不定义非静态方法。这项任务留给 Small Windows 的用户来完成。

当用户添加一个带有监听器的菜单项时，会创建一个`Command`对象。实际上，这是添加到`Command`对象中的具有`void`指针参数的静态方法。此外，当用户选择一个菜单项时，调用的是静态方法。静态方法反过来调用用户定义的非静态方法。

宏接受当前类和监听器的名称作为参数。请注意，`bool`监听器是常量，而`void`监听器不是常量。这是因为`bool`监听器旨在查找类的字段之一或多个字段的值，而`void`监听器还修改字段。

**Menu.h**

```cpp
#define DEFINE_BOOL_LISTENER(SubClass, Listener)  \ 
  virtual bool Listener() const;                  \ 
  static bool SubClass::Listener(void* voidPtr) { \ 
    return ((SubClass*) voidPtr)->Listener();     \ 
  } 

#define DEFINE_VOID_LISTENER(SubClass, Listener)  \ 
  virtual void Listener();                        \ 
  static void SubClass::Listener(void* voidPtr) { \ 
    ((SubClass*) voidPtr)->Listener();            \ 
  } 

namespace SmallWindows { 
  class Document; 

  class Menu { 
    public: 
      Menu(Document* documentPtr, String text = TEXT("")); 
      Menu(const Menu& menu); 

      void AddMenu(Menu& menu); 
      void AddSeparator(); 
      void AddItem(String text, VoidListener selection, 
                   BoolListener enable = nullptr, 
                   BoolListener check = nullptr, 
                   BoolListener radio = nullptr); 

```

访问文档的命令映射和加速器集时需要文档指针。除了菜单栏之外，每个菜单都有在文档窗口中显示的文本；`menuHandle`是这个类包装的 Win32 API 菜单句柄：

```cpp
    private: 
      Document* documentPtr; 
      String text; 
      HMENU menuHandle; 

      friend class Document; 
      friend class StandardDocument; 
  }; 
}; 

```

**Menu.cpp**

```cpp
#include "SmallWindows.h" 

```

构造函数初始化指针文档和文本。它还通过调用 Win32 API 函数`CreateMenu`创建菜单。由于菜单栏不需要文本，`text`参数默认为空：

```cpp
namespace SmallWindows { 
  Menu::Menu(Document* documentPtr, String text /* = TEXT("") */) 
   :documentPtr(documentPtr), 
    text(text), 
    menuHandle(::CreateMenu()) { 
    // Empty. 
  } 

```

复制构造函数复制菜单的字段。请注意，我们复制`menuHandle`字段而不是创建一个新的菜单句柄。

```cpp
  Menu::Menu(const Menu& menu) 
   :documentPtr(menu.documentPtr), 
    text(menu.text), 
    menuHandle(menu.menuHandle) { 
    // Empty. 
  } 

```

`AddMenu`方法将菜单（不是菜单项）作为子菜单添加到菜单中，而`AddSeparator`将分隔符（水平条）添加到菜单中：

```cpp
  void Menu::AddMenu(Menu& menu) { 
    ::AppendMenu(menuHandle, MF_STRING | MF_POPUP, 
                 (UINT) menu.menuHandle, menu.text.c_str()); 
  } 

  void Menu::AddSeparator() { 
    ::AppendMenu(menuHandle, MF_SEPARATOR, 0, nullptr); 
  } 

```

`AddItem`方法将菜单项（不是菜单）添加到菜单中，带有选择、启用、检查和单选监听器：

```cpp
  void Menu::AddItem(String text, VoidListener selection, 
                     BoolListener enable /* = nullptr */, 
                     BoolListener check /* = nullptr */, 
                     BoolListener radio /* = nullptr */) { 

```

选择监听器不允许为空，并且至少有一个复选框和单选按钮监听器必须为空，因为不可能同时用复选框和单选按钮注释菜单项：

```cpp
    assert((selection != nullptr) && 
           ((check == nullptr) || (radio == nullptr))); 

```

每个菜单项都有一个唯一的标识符，我们通过命令映射的当前大小来获取：

```cpp
    map<WORD,Command>& commandMap = documentPtr->CommandMap(); 
    int itemId = commandMap.size(); 

```

我们使用 Win32 API 函数 `AppendMenu` 将一个 `Command` 对象添加到命令映射中，并添加菜单项，该函数需要菜单句柄、标识符和文本：

```cpp
    commandMap[itemId] = Command(listener, enable, check, radio); 
    ::AppendMenu(menuHandle, MF_STRING, 
                 (UINT) itemId, text.c_str()); 

```

如果单选按钮监听器不为空，我们需要调用 Win32 API 函数 `SetMenuItemInfo` 以使单选按钮与菜单项一起出现：

```cpp
    if (radio != nullptr) { 
      MENUITEMINFO menuItemInfo; 
      menuItemInfo.cbSize = sizeof menuItemInfo; 
      menuItemInfo.fMask = MIIM_FTYPE; 
      menuItemInfo.fType = MFT_RADIOCHECK; 
      ::SetMenuItemInfo(menuHandle, (UINT) itemId, 
                        FALSE, &menuItemInfo); 
    } 

```

最后，我们在 `Accelerator`（在下一节中描述）中调用 `TextToAccelerator`，如果存在，则将其添加到文档的加速器集中，该加速器集由 `Application` 的消息循环使用：

```cpp
    Accelerator::TextToAccelerator(text, itemId, 
                                   documentPtr->AcceleratorSet()); 
  } 
}; 

```

# 加速器类

可以将一个加速器添加到菜单项中。加速器文本前面有一个制表符字符 (`\t`)，文本由可选的前缀 `Ctrl+`、`Shift+` 或 `Alt+` 后跟一个字符（例如，`&Open\tCtrl+O`）或虚拟键的名称（例如，`&Save\tAlt+F2`）组成。

**Accelerator.h**

```cpp
namespace SmallWindows { 

```

Win32 API 包含一组以 `VK_` 开头的虚拟键。在小窗口中，它们被赋予了其他名称，希望更容易理解。可用的虚拟键有：**F1** - **F12**、**Insert**、**Delete**、**Backspace**、**Tab**、**Home**、**End**、**Page Up**、**Page Down**、**Left**、**Right**、**Up**、**Down**、**Space**、**Escape** 和 **Return**：

```cpp
  enum Keys {KeyF1 = VK_F1, KeyF2 = VK_F2, KeyF3 = VK_F3, 
             KeyF4 = VK_F4, KeyF5 = VK_F5, KeyF6 = VK_F6, 
             KeyF7 = VK_F7, KeyF8 = VK_F8, KeyF9 = VK_F9, 
             KeyF10 = VK_F10, KeyF11 = VK_F11, KeyF12 = VK_F12, 
             KeyInsert = VK_INSERT, KeyDelete = VK_DELETE, 
             KeyBackspace = VK_BACK, KeyTabulator = VK_TAB, 
             KeyHome = VK_HOME, KeyEnd = VK_END, 
             KeyPageUp = VK_PRIOR, KeyPageDown = VK_NEXT, 
             KeyLeft = VK_LEFT, KeyRight = VK_RIGHT, 
             KeyUp = VK_UP, KeyDown = VK_DOWN, 
             KeySpace = VK_SPACE, KeyEscape = VK_ESCAPE, 
             KeyReturn = VK_RETURN}; 

```

`Accelerator` 类只包含 `TextToAccelerator` 方法，该方法接受文本，提取加速器，如果存在，则将其添加到加速器集中：

```cpp
  class Accelerator { 
    public: 
      static void TextToAccelerator(String& text, int idemId, 
                                    list<ACCEL>& acceleratorSet); 
  }; 
}; 

```

**Accelerator.cpp**

```cpp
    #include "SmallWindows.h"

```

`TextToVirtualKey` 是一个辅助函数，它接受文本并返回相应的虚拟键。`keyTable` 数组持有文本和可用虚拟键之间的映射：

```cpp
namespace SmallWindows { 
  WORD TextToVirtualKey(String& text) { 
    static const struct { 
      TCHAR* textPtr; 
      WORD key; 
    } keyTable[] = { 
      {TEXT("F1"), KeyF1}, {TEXT("F2"), KeyF2}, 
      {TEXT("F3"), KeyF3}, {TEXT("F4"), KeyF4}, 
      {TEXT("F5"), KeyF5}, {TEXT("F6"), KeyF6}, 
      {TEXT("F7"), KeyF7}, {TEXT("F8"), KeyF8}, 
      {TEXT("F9"), KeyF9}, {TEXT("F10"), KeyF10}, 
      {TEXT("F11"), KeyF11}, {TEXT("F12"), KeyF12}, 
      {TEXT("Insert"), KeyInsert}, {TEXT("Delete"), KeyDelete}, 
      {TEXT("Back"), KeyBackspace}, {TEXT("Tab"), KeyTabulator}, 
      {TEXT("Home"), KeyHome}, {TEXT("End"), KeyEnd}, 
      {TEXT("Page Up"), KeyPageUp}, 
      {TEXT("Page Down"), KeyPageDown}, 
      {TEXT("Left"), KeyLeft}, {TEXT("Right"), KeyRight}, 
      {TEXT("Up"), KeyUp}, {TEXT("Down"), KeyDown}, 
      {TEXT("Space"), KeySpace},  {TEXT("Escape"), KeyEscape}, 
      {TEXT("Return"), KeyReturn}, {nullptr, 0}}; 

```

我们遍历表格，直到找到虚拟键：

```cpp
    for (int index = 0; keyTable[index].textPtr != nullptr; 
         ++index) { 
      if (text == keyTable[index].textPtr) { 
        return keyTable[index].key; 
      } 
    } 

```

如果我们没有找到与文本匹配的键，将发生断言：

```cpp
    assert(false); 
    return 0; 
  } 

```

在 `TextToAccelerator` 中，我们将 **Control**、**Shift**、**Alt** 和虚拟键状态与键一起存储在一个 Win32 API `ACCEL` 结构中：

```cpp
  void Accelerator::TextToAccelerator(String& text, int itemId, 
                                      list<ACCEL>&acceleratorSet){ 

```

首先，我们检查文本是否包含一个 *Tab* 键（**\t**）。如果包含，我们使用 `itemId` 初始化 `ACCEL` 结构，并提取文本的加速器部分：

```cpp
    int tabulatorIndex = text.find(TEXT("\t")); 
    if (tabulatorIndex != -1) { 
      ACCEL accelerator; 
      accelerator.fVirt = 0; 
      accelerator.cmd = itemId; 
      String acceleratorText = text.substr(tabulatorIndex + 1); 

```

如果加速器文本包含前缀 `Ctrl+`、`Alt+` 或 `Shift+`，我们将 `FCONTROL`、`FALT` 或 `FSHIFT` 遮罩到 `fVirt` 字段，并移除前缀：

```cpp
      { String controlText = TEXT("Ctrl+"); 
        int controlIndex = acceleratorText.find(controlText); 

        if (controlIndex != -1) { 
          accelerator.fVirt |= FCONTROL; 
          acceleratorText.erase(controlIndex, 
                                controlText.length()); 
        } 
      } 

      { String altText = TEXT("Alt+"); 
        int altIndex = acceleratorText.find(altText); 

        if (altIndex != -1) { 
          accelerator.fVirt |= FALT; 
          acceleratorText.erase(altIndex, altText.length()); 
        } 
      } 

      { String shiftText = TEXT("Shift+"); 
        int shiftIndex = acceleratorText.find(shiftText); 

        if (shiftIndex != -1) { 
          accelerator.fVirt |= FSHIFT; 
          acceleratorText.erase(shiftIndex, shiftText.length()); 
        } 
      } 

```

在移除 `Ctrl+`、`Shift+` 和 `Alt+` 前缀后，我们查看加速器文本的剩余部分。如果只有一个字符（长度为 1），我们将其保存到 `key` 字段。但是，我们不保存 ASCII 编号。相反，我们保存字母编号，对于 `a` 或 `A` 从 1 开始：

```cpp
      if (acceleratorText.length() == 1) { 
        accelerator.key = 
          (WORD) ((tolower(acceleratorText[0]) - ''a'') + 1); 
      } 

```

如果加速器文本的剩余部分由多个字符组成，我们假设它是一个虚拟键，并调用 `TextToVirtualKey` 来查找它，并将 `FVIRTKEY` 常量屏蔽到 `fVirt` 字段：

```cpp
      else { 
        accelerator.fVirt |= FVIRTKEY; 
        accelerator.key = TextToVirtualKey(acceleratorText); 
      } 

```

如果 `fVirt` 仍然为零，则加速器不包含 `Ctrl+`、`Shift+`、`Alt+` 或虚拟键，这是不允许的：

```cpp
      assert(accelerator.fVirt != 0); 

```

最后，我们将加速器添加到加速器集合中：

```cpp
      acceleratorSet.push_back(accelerator); 
    } 

```

注意，如果文本不包含制表符，则不会向加速器集合中添加任何加速器：

```cpp
  } 
}; 

```

# `StandardDocument` 类

`StandardDocument` 类是 `Document` 的直接子类；它处理 **File**、**Edit** 和 **Help** 菜单，并实现文件处理、剪切、复制和粘贴、拖放文件和打印。此类没有特定的消息函数；所有消息都发送到之前覆盖的 `Document` 部分的 `DocumentProc`。文档名称和脏标志由框架自动更新。`StandardDocument` 还处理页面设置对话框，这在 第十二章 *辅助类* 中有更详细的描述。

**StandardDocument.h**

```cpp
namespace SmallWindows { 
  class StandardDocument : public Document { 
    public: 

```

大多数构造函数参数都发送到 `Document` 构造函数。对于 `StandardDocument` 来说，特定的是文件描述文本和复制粘贴格式列表。文件描述由标准保存和打开对话框使用。复制粘贴列表用于在应用程序和全局剪贴板之间复制粘贴信息：

```cpp
      StandardDocument(CoordinateSystem system, Size pageSize, 
                       String fileDescriptionsText, 
                       Window* parentPtr=nullptr, 
                       WindowStyle style = OverlappedWindow, 
                       WindowShow windowShow = Normal, 
                       initializer_list<unsigned int> 
                         copyFormatList = {}, 
                       initializer_list<unsigned int> 
                         pasteFormatList = {}, 
                       bool acceptDropFiles = true, 
                       Size lineSize = LineSize); 

    private: 
      void InitializeFileFilter(String fileDescription); 

```

`StandardFileMenu`、`StandardEditMenu` 和 `StandardHelpMenu` 方法创建并返回标准菜单。如果 `StandardFileMenu` 中的 `print` 为 `true`，则包括 **Page Setup**、**Print** 和 **Print Preview** 菜单项：

```cpp
    protected: 
      Menu StandardFileMenu(bool print); 
      Menu StandardEditMenu(); 
      Menu StandardHelpMenu(); 

```

当文档不需要保存时（脏标志为 `false`），**Save** 菜单项被禁用。在 **Save** 菜单项可见之前调用 `SaveEnable` 方法，如果脏标志为 `true`，则启用它。

```cpp
    private: 
      DEFINE_VOID_LISTENER(StandardDocument, OnNew); 
      DEFINE_VOID_LISTENER(StandardDocument, OnOpen); 
      DEFINE_BOOL_LISTENER(StandardDocument, SaveEnable); 
      DEFINE_VOID_LISTENER(StandardDocument, OnSave); 
      DEFINE_VOID_LISTENER(StandardDocument, OnSaveAs); 

```

`OnSave` 方法根据文档是否已命名调用 `SaveFileWithName` 或 `SaveFileWidhoutName`。然而，`OnSaveAs` 总是调用 `SaveFileWithoutName`，无论文档是否有名称。

```cpp
   private: 
      void SaveFileWithName(String name); 
      void SaveFileWithoutName(); 

```

当用户选择 **New**、**Save**、**Save As** 或 **Open** 菜单项时，会调用 `ClearDocument`、`WriteDocumentToStream` 和 `ReadDocumentFromStream` 方法，这些方法旨在由子类覆盖以清除、写入和读取文档：

```cpp
    protected: 
      void ClearPageSetupInfo(); 
      bool ReadPageSetupInfoFromStream(istream &inStream); 
      bool WritePageSetupInfoToStream(ostream &outStream) const; 

      virtual void ClearDocument() {/* Empty. */} 
      virtual bool WriteDocumentToStream(String name, 
                        ostream& outStream) const {return true;} 
      virtual bool ReadDocumentFromStream(String name, 
                       istream& inStream) {return true;} 

```

当用户在 **Edit** 菜单中选择相应的菜单项时，会调用 `OnCut`、`OnCopy`、`OnPaste` 和 `OnDelete` 方法。`OnCut` 的默认行为是先调用 `OnCopy`，然后调用 `OnDelete`：

```cpp
      DEFINE_VOID_LISTENER(StandardDocument, OnCut); 
      DEFINE_VOID_LISTENER(StandardDocument, OnCopy); 
      DEFINE_VOID_LISTENER(StandardDocument, OnPaste); 
      DEFINE_VOID_LISTENER(StandardDocument, OnDelete); 

```

`CutEnable`、`CopyEnable`、`PasteEnable` 和 `DeleteEnable` 方法是监听器，用于决定菜单项是否启用。`CutEnable` 和 `DeleteEnable` 的默认行为是调用 `CopyEnable`：

```cpp
      DEFINE_BOOL_LISTENER(StandardDocument, CutEnable); 
      DEFINE_BOOL_LISTENER(StandardDocument, CopyEnable); 
      DEFINE_BOOL_LISTENER(StandardDocument, PasteEnable); 
      DEFINE_BOOL_LISTENER(StandardDocument, DeleteEnable); 

```

`IsCopyAsciiReady`、`IsCopyUnicodeReady` 和 `IsCopyGenericReady` 方法由 `CopyEnable` 调用。它们旨在被重写，并在应用程序准备好以 ASCII、Unicode 或通用格式进行复制时返回 `true`。它们的默认行为是返回 `false`：

```cpp
      virtual bool IsCopyAsciiReady() const {return false;} 
      virtual bool IsCopyUnicodeReady() const {return false;} 
      virtual bool IsCopyGenericReady(int format) 
                                      const {return false;} 

```

当用户选择 **复制** 菜单项时，`OnCopy` 会调用 `CopyAscii`、`CopyUnicode` 和 `CopyGeneric` 方法。它们旨在被子类重写，并按照构造函数中的复制格式列表和复制就绪方法进行调用：

```cpp
      virtual void CopyAscii(vector<String>& textList) const 
                            {/* Empty. */} 
      virtual void CopyUnicode(vector<String>& textList) const 
                              {/* Empty. */} 
      virtual void CopyGeneric(int format, InfoList& infoList) 
                               const {/* Empty. */} 

```

`IsPasteAsciiReady`、`IsPasteUnicodeReady` 和 `IsPasteGenericReady` 方法由 `PasteEnable` 调用，如果至少有一个方法返回 `true`，则返回 `true`。它们旨在被重写，并在应用程序准备好以 ASCII、Unicode 或通用格式粘贴时返回 `true`。它们的默认行为是返回 `true`：

```cpp
      virtual bool IsPasteAsciiReady 
              (const vector<String>&textList) const {return true;} 
      virtual bool IsPasteUnicodeReady 
              (const vector<String>&textList) const {return true;} 
      virtual bool IsPasteGenericReady(int format, 
                        InfoList& infoList) const {return true;} 

```

当用户选择 **粘贴** 菜单项时，`OnPaste` 会调用 `PasteAscii`、`PasteUnicode` 和 `PasteGeneric` 方法。它们旨在被子类重写，并按照构造函数中的粘贴格式列表和粘贴就绪方法进行调用。复制和粘贴之间有一个区别，即复制在所有可用格式中执行，而粘贴仅在第一个可用格式中执行：

```cpp
      virtual void PasteAscii(const vector<String>& textList) 
                             {/* Empty. */} 
      virtual void PasteUnicode(const vector<String>& textList) 
                               {/* Empty. */} 
      virtual void PasteGeneric(int format, InfoList& infoList) 
                               {/* Empty. */} 

```

当用户在窗口的客户区域中拖放一组文件时，会调用 `OnDropFile` 方法。如果路径列表中恰好有一个文件具有构造函数中给出的后缀，则该文件将以与用户在标准打开对话框中选择它相同的方式读取。然而，如果没有文件或列表中有多个具有后缀的文件，则显示错误消息：

```cpp
      void OnDropFile(vector<String> pathList); 

```

`PageOuterSize` 方法根据页面设置设置返回页面在纵向或横向模式下的逻辑大小，不考虑边距，而 `PageInnerSize`、`PageInnerWidth` 和 `PageInnerHeight` 返回减去边距后的页面大小：

```cpp
    private: 
      Size PageOuterSize() const; 
      Size PageInnerSize() const; 

    protected: 
      int PageInnerWidth() const{return PageInnerSize().Width();} 
      int PageInnerHeight()const{return PageInnerSize().Height();} 

```

当用户选择 **页面设置**、**打印** 和 **打印预览** 菜单项时，会调用 `OnPageSetup`、`OnPrintPreview` 和 `OnPrintItem` 方法。它们显示 **页面设置对话框**、**打印预览窗口** 和 **打印对话框**：

```cpp
    public: 
      DEFINE_VOID_LISTENER(StandardDocument, OnPageSetup); 
      DEFINE_VOID_LISTENER(StandardDocument, OnPrintPreview); 
      DEFINE_VOID_LISTENER(StandardDocument, OnPrintItem); 

```

`PrintPage` 方法由 `OnPrintItem` 调用，并打印文档的一页：

```cpp
      bool PrintPage(Graphics* graphicsPtr, int page, 
                     int copy, int totalPages); 

```

当用户选择 **页面设置** 菜单项并更改页面设置信息时，会调用 `OnPageSetup` 方法来通知应用程序。它旨在被子类重写，并且其默认行为是不执行任何操作：

```cpp
      virtual void OnPageSetup(PageSetupInfo info) {/* Empty. */} 

```

`GetTotalPages` 方法返回要打印的页数；默认值为 1。它旨在被子类重写：

```cpp
      virtual int GetTotalPages() const {return 1;} 

```

`OnPrint`方法由`OnPrintItem`对每一页和副本调用一次。它的默认行为是按照**页面设置对话框**中的设置写入页眉和页脚，然后调用`OnDraw`以显示文档的应用程序特定内容：

```cpp
      virtual void OnPrint(Graphics& graphics, int page, 
                           int copy, int totalPages) const; 

```

当用户选择**退出**菜单项并退出应用程序时，会调用`OnExit`方法，如果`TryClose`返回`true`，则应用程序会退出。如果脏标志为`true`，`TryClose`会显示一个消息框，询问用户是否允许关闭窗口：

```cpp
      DEFINE_VOID_LISTENER(StandardDocument, OnExit); 
      virtual bool TryClose(); 

```

`OnAbout`方法显示一个包含应用程序名称的简单消息框：

```cpp
      DEFINE_VOID_LISTENER(StandardDocument, OnAbout); 

```

**文件**字段由**打开**和**保存**标准对话框使用，而`fileSuffixList`用于检查拖放文件的文件后缀：

```cpp
    private: 
      TCHAR fileFilter[MAX_PATH]; 
      vector<String> fileSuffixList; 

```

当用户选择**页面设置**菜单项时，会使用`pageSetupInfo`字段。它存储有关页眉和页脚文本和字体、页面方向（纵向或横向）、页边距以及页面是否被框架包围的信息。请参阅下一章以获取更详细的描述。

```cpp
      PageSetupInfo pageSetupInfo; 

```

`copyFormatList`和`pasteFormatList`字段包含可用于剪切、复制和粘贴的格式：

```cpp
      list<unsigned int> copyFormatList, pasteFormatList; 
  }; 
}; 

```

## 初始化

第一个`StandardDocument`构造函数接受一组大量的参数。坐标系、页面大小、父窗口、样式、外观、文档是否接受拖放文件，以及行大小参数与之前覆盖的`Document`案例相同。

剩余的是文件描述文本，打印菜单是否存在，以及复制和粘贴的格式列表。描述文本包含一个分号分隔的文件描述和允许文件的文件后缀列表，例如，**Calc Files**，*clc*；**Text Files**，*txt*。复制和粘贴的格式列表包含复制和粘贴信息的允许格式。

**StandardDocument.cpp**

```cpp
#include "SmallWindows.h"

```

大多数构造函数参数都发送到`Document`构造函数。然而，复制和粘贴的格式列表存储在`copyFormatList`和`pasteFormatList`中。文件过滤器由`InitializeFileFilter`初始化：

```cpp
namespace SmallWindows { 
  StandardDocument::StandardDocument(CoordinateSystem system, 
                        Size pageSize, 
                        String fileDescriptionsText, 
                        Window* parentPtr /* = nullptr */, 
                        WindowStyle style/* = OverlappedWindow */, 
                        WindowShow windowShow /* = Normal */, 
                        initializer_list<unsigned int> 
                          copyFormatList /* = {} */, 
                        initializer_list<unsigned int> 
                          pasteFormatList /* = {}*/, 
                        bool acceptDropFiles /* = true */, 
                        Size lineSize /* = LineSize */) 
   :Document(TEXT("standarddocument"), system, pageSize, 
             parentPtr, style, windowShow, 
             acceptDropFiles, lineSize), 
    copyFormatList(copyFormatList), 
    pasteFormatList(pasteFormatList) { 
    InitializeFileFilter(fileDescriptionsText); 

```

在`Window`中，我们使用页面大小在逻辑单位和物理单位之间进行转换。在`Document`中，我们使用它来设置滚动页面大小。然而，在`StandardDocument`中，实际上有两种页面大小：外页大小和内页大小。外页大小是不考虑文档边距的页面大小。内页大小是通过从外页大小中减去边距得到的。在`StandardDocument`中，我们使用内页大小来设置滚动条的大小：

```cpp
    SetHorizontalScrollTotalWidth(PageInnerWidth()); 
    SetVerticalScrollTotalHeight(PageInnerHeight()); 
  } 

```

## 标准菜单

以下代码展示了这一点：

```cpp
  void StandardDocument::InitializeFileFilter(String fileListText) 
  { OStringStream filterStream; 
    vector<String> fileList = Split(fileListText, TEXT('';'')); 
    assert(fileList.size() > 0); 

    for (String fileText : fileList) { 
      vector<String> partList = Split(fileText, TEXT('','')); 
      assert(partList.size() == 2); 
      String description = Trim(partList[0]), 
             suffix = Trim(partList[1]); 
      fileSuffixList.push_back(suffix); 
      filterStream << description << TEXT(" (*.") << suffix 
                   << TEXT(")\n") << TEXT("*.") << suffix 
                   << TEXT("\n"); 
    } 

    filterStream << TEXT("\n"); 

    int index = 0; 
    for (TCHAR c : filterStream.str()) { 
      fileFilter[index++] = (c == TEXT(''\n'')) ? TEXT(''\0'') : c; 
    } 
  } 

```

标准的**文件**菜单包含**新建**、**打开**、**保存**、**另存为**和**退出**菜单项，以及（如果`print`为`true`）**页面设置**、**打印预览**和**打印**菜单项：

```cpp
  Menu StandardDocument::StandardFileMenu(bool print) { 
    Menu fileMenu(this, TEXT("&File")); 
    fileMenu.AddItem(TEXT("&New\tCtrl+N"), OnNew); 
    fileMenu.AddItem(TEXT("&Open\tCtrl+O"), OnOpen); 
    fileMenu.AddItem(TEXT("&Save\tCtrl+S"), OnSave, SaveEnable); 
    fileMenu.AddItem(TEXT("Save &As\tCtrl+Shift+S"), OnSaveAs); 

    if (print) { 
      fileMenu.AddSeparator(); 
      fileMenu.AddItem(TEXT("Page Set&up"), OnPageSetup); 
      fileMenu.AddItem(TEXT("Print Pre&view"), OnPrintPreview); 
      fileMenu.AddItem(TEXT("&Print\tCtrl+P"), OnPrintItem); 
    } 

    fileMenu.AddSeparator(); 
    fileMenu.AddItem(TEXT("E&xit\tAlt+X"), OnExit); 
    return fileMenu; 
  } 

```

标准的**编辑**菜单包含**剪切**、**复制**、**粘贴**和**删除**菜单项：

```cpp
  Menu StandardDocument::StandardEditMenu() { 
    Menu editMenu(this, TEXT("&Edit")); 
    editMenu.AddItem(TEXT("C&ut\tCtrl+X"), OnCut, CutEnable); 
    editMenu.AddItem(TEXT("&Copy\tCtrl+C"), OnCopy, CopyEnable); 
    editMenu.AddItem(TEXT("&Paste\tCtrl+V"), OnPaste,PasteEnable); 
    editMenu.AddSeparator(); 
    editMenu.AddItem(TEXT("&Delete\tDelete"), 
                     OnDelete, DeleteEnable); 
    return editMenu; 
  } 

```

标准的**帮助**菜单包含使用应用程序名称的**关于**菜单项：

```cpp
  Menu StandardDocument::StandardHelpMenu() { 
    Menu helpMenu(this, TEXT("&Help")); 
    helpMenu.AddItem(TEXT("About ") + 
                     Application::ApplicationName() + 
                     TEXT(" ..."), OnAbout); 
    return helpMenu; 
  } 

```

## 文件管理

当用户尝试关闭窗口时，`TryClose`方法检查脏标志是否为`true`。如果是`true`，则询问用户在关闭前是否要保存文档。如果他们回答是，则像用户选择了**保存**菜单项一样保存文档。如果之后脏标志设置为`false`，则表示保存操作成功，并返回`true`。如果用户回答否，则返回`true`并关闭窗口而不保存。如果答案是取消，则返回`false`并中止关闭操作：

```cpp
  bool StandardDocument::TryClose() { 
    if (IsDirty()) { 
      switch (MessageBox(TEXT("Do you want to save?"), 
                         TEXT("Unsaved Document"), YesNoCancel)) { 
        case Yes: 
          OnSave(); 
          return !IsDirty(); 

        case No: 
          return true; 

        case Cancel: 
          return false; 
      } 
    } 

    return true; 
  } 

```

`OnExit`方法调用`TryClose`并删除应用程序的主窗口，如果`TryClose`返回`true`，则最终向消息循环发送退出消息以终止应用程序：

```cpp
  void StandardDocument::OnExit() { 
    if (TryClose()) { 
      delete Application::MainWindowPtr(); 
    } 
  } 

```

当用户选择**新建**菜单项时，会调用`OnNew`方法。它尝试通过调用`TryClose`来关闭窗口。如果`TryClose`返回`true`，则清除文档、脏标志和名称，并使窗口无效并更新。`ClearDocument`方法被缩进以供子类覆盖，以清除文档的应用程序特定内容：

```cpp
  void StandardDocument::OnNew() { 
    if (TryClose()) { 
      ClearDocument(); 
      ClearPageSetupInfo(); 
      SetZoom(1.0); 
      SetDirty(false); 
      SetName(TEXT("")); 
      Invalidate(); 
      UpdateWindow(); 
      UpdateCaret(); 
    } 
  } 

```

当用户选择**打开**菜单项时，会调用`OnOpen`方法。它尝试通过调用`TryClose`来关闭窗口，并在成功的情况下显示标准打开对话框以建立文件的路径。如果`OpenDialog`返回`true`且输入流有效，则读取页面设置信息，并调用`ClearDocument`和`ReadDocumentFromStream`方法，这些方法旨在由子类覆盖：

```cpp
  void StandardDocument::OnOpen() { 
    if (TryClose()) { 
      String name = GetName(); 

      if (StandardDialog::OpenDialog(this, name, fileFilter, 
                                     fileSuffixList)) { 
        ClearDocument(); 
        Invalidate(); 
        UpdateWindow(); 
        ifstream inStream(name.c_str()); 

        if (inStream && ReadDocumentFromStream(name, inStream)) { 
          SetName(name); 
        } 

        else { 
          MessageBox(TEXT("Could not open ") + 
                     name + TEXT(".")); 
        } 
      } 
    } 

    SetDirty(false); 
    SetZoom(1.0); 
    Invalidate(); 
    UpdateWindow(); 
    UpdateCaret(); 
  } 

```

如果脏标志为`true`，则**保存**菜单项被启用：

```cpp
  bool StandardDocument::SaveEnable() const { 
    return IsDirty(); 
  } 

```

保存文件时，如果文件已有名称，则调用`SaveFileWithName`。如果文件尚未命名，则调用`SaveFileWithoutName`代替：

```cpp
  void StandardDocument::OnSave() { 
    String name = GetName(); 

    if (!name.empty()) { 
      SaveFileWithName(name); 
    } 
    else { 
      SaveFileWithoutName(); 
    } 
  } 

```

当用户选择**另存为**时，无论文档是否有名称，都会调用`SaveFileWithoutName`并显示**保存**标准对话框：

```cpp
  void StandardDocument::OnSaveAs() { 
    SaveFileWithoutName(); 
  } 

```

`SaveFileWithoutName`方法显示保存对话框。如果用户按下**确定**按钮，则`SaveDialog`调用返回`true`，设置新名称，并调用`SaveFileWithName`以执行文档文件的实际写入：

```cpp
  void StandardDocument::SaveFileWithoutName() { 
    String name = GetName(); 

    if (StandardDialog::SaveDialog(this, name, fileFilter, 
                                   fileSuffixList)) { 
      SaveFileWithName(name); 
    } 
  } 

```

`SaveFileWithName`方法尝试打开文档文件进行写入，并调用`WriteDocumentToStream`，该方法旨在由子类覆盖以执行文档内容的实际写入。如果页面设置信息和文档内容写入成功，则清除脏标志：

```cpp
  void StandardDocument::SaveFileWithName(String name) { 
    ofstream outStream(name.c_str()); 

    if (outStream && WriteDocumentToStream(name, outStream)) { 
      SetName(name); 
      SetDirty(false); 
      SetZoom(1.0); 
    } 
  } 

  void StandardDocument::ClearPageSetupInfo() { 
    pageSetupInfo.ClearPageSetupInfo(); 
  } 

  bool StandardDocument::ReadPageSetupInfoFromStream 
                         (istream &inStream) { 
    pageSetupInfo.ReadPageSetupInfoFromStream(inStream); 
    return ((bool) inStream); 
  } 

  bool StandardDocument::WritePageSetupInfoToStream 
                         (ostream &outStream) const { 
    pageSetupInfo.WritePageSetupInfoToStream(outStream); 
    return ((bool) outStream); 
  } 

```

当用户在**帮助**标准菜单中选择**关于**菜单项时，会显示一个包含应用程序名称的消息框：

```cpp
  void StandardDocument::OnAbout() { 
    String applicationName = Application::ApplicationName(); 
    MessageBox(applicationName + TEXT(", version 1.0"), 
               applicationName, Ok, Information); 
  } 

```

## 剪切、复制和粘贴

`CutEnable`和`DeleteEnable`的默认行为是简单地调用`CopyEnable`，因为它们很可能在相同的条件下被启用：

```cpp
  bool StandardDocument::CutEnable() const { 
    return CopyEnable(); 
  } 

  bool StandardDocument::DeleteEnable() const { 
    return CopyEnable(); 
  } 

```

`OnCut`的默认行为是简单地调用`OnCopy`和`OnDelete`，这是剪切的常见操作：

```cpp
  void StandardDocument::OnCut() { 
    OnCopy(); 
    OnDelete(); 
  } 

```

`OnDelete`方法为空，并打算由子类覆盖：

```cpp
  void StandardDocument::OnDelete() { 
    // Empty. 
  } 

```

`CopyEnable`方法遍历粘贴格式列表，并根据格式调用`IsCopyAsciiReady`、`IsCopyUnicodeReady`或`IsCopyGenericReady`。一旦其中一个方法返回`true`，`CopyEnable`就返回`true`，这意味着只要允许其中一个格式的复制就足够了。当实际复制在`OnCopy`中发生时，准备好的方法会被再次调用：

```cpp
  bool StandardDocument::CopyEnable() const { 
    for (unsigned int format : pasteFormatList) { 
      switch (format) { 
        case AsciiFormat: 
          if (IsCopyAsciiReady()) { 
            return true; 
          } 
          break; 

        case UnicodeFormat: 
          if (IsCopyUnicodeReady()) { 
            return true; 
          } 
          break; 

        default: 
          if (IsCopyGenericReady(format)) { 
            return true; 
          } 
          break; 
      } 
    }  
    return false; 
  } 

```

`OnCopy`方法遍历构造函数中给出的复制格式列表，并根据格式调用适当的方法：

```cpp
  void StandardDocument::OnCopy() { 
    if (Clipboard::Open(this)) { 
      Clipboard::Clear();  
      for (unsigned int format : copyFormatList) { 
        switch (format) { 

```

如果应用了 ASCII 格式，并且`IsCopyAsciiReady`返回`true`，则调用`CopyAscii`，该函数的目的是由子类覆盖以填充`asciiList`中的 ASCII 文本。当列表被复制后，它会被传递给`Clipboard`中的`WriteAscii`，该函数将文本存储在全局剪贴板中：

```cpp
          case AsciiFormat:  
            if (IsCopyAsciiReady()) { 
              vector<String> asciiList; 
              CopyAscii(asciiList); 
              Clipboard::WriteText<AsciiFormat,char>(asciiList); 
            } 
            break; 

```

如果应用了 Unicode 格式，并且`IsCopyUnicodeReady`返回`true`，则调用`CopyUnicode`，该函数的目的是由子类覆盖以填充`unicodeList`中的 Unicode 文本。当列表被复制后，它会被传递给`Clipboard`中的`WriteUnicode`，该函数将文本存储在全局剪贴板中：

```cpp
          case UnicodeFormat: 
            if (IsCopyUnicodeReady()) { 
              vector<String> unicodeList; 
              CopyUnicode(unicodeList); 
              Clipboard::WriteText<UnicodeFormat,wchar_t> 
                                  (unicodeList); 
            } 
            break; 

```

如果既不应用 ASCII 也不应用 Unicode，并且`IsCopyGenericReady`返回`true`，则调用`CopyGeneric`，该函数的目的是由子类覆盖以填充字符列表中的通用信息。在 C++中，`char`类型始终占用一个字节；因此，在没有更通用的字节类型的情况下使用。当信息被复制到`infoList`后，它会被传递给`Clipboard`中的`WriteGeneric`以在全局剪贴板上存储信息：

```cpp
          default: 
            if (IsCopyGenericReady(format)) { 
              InfoList infoList; 
              CopyGeneric(format, infoList); 
              Clipboard::WriteGeneric(format, infoList); 
            } 
            break; 
        } 
      }  
      Clipboard::Close(); 
    } 
  } 

```

`PasteEnable`方法遍历构造函数中给出的粘贴格式列表，如果至少有一个格式在全局剪贴板上可用，则返回`true`：

```cpp
  bool StandardDocument::PasteEnable() const { 
    if (Clipboard::Open(this)) { 
      for (unsigned int format : pasteFormatList) { 
        if (Clipboard::Available(format)) { 
          switch (format) { 
            case AsciiFormat: { 
                vector<String> asciiList; 
                if (Clipboard::ReadText<AsciiFormat,char> 
                                       (asciiList) && 
                    IsPasteAsciiReady(asciiList)) { 
                  Clipboard::Close(); 
                  return true; 
                } 
              } 
              break;  
            case UnicodeFormat: { 
                vector<String> unicodeList; 
                if (Clipboard::ReadText<UnicodeFormat,wchar_t> 
                                       (unicodeList) && 
                    IsPasteUnicodeReady(unicodeList)) { 
                  Clipboard::Close(); 
                  return true; 
                } 
              } 
              break;  
            default: { 
                InfoList infoList; 
                if (Clipboard::ReadGeneric(format, infoList) && 
                    IsPasteGenericReady(format, infoList)) { 
                  Clipboard::Close(); 
                  return true; 
                } 
              } 
          } 
        } 
      } 

      Clipboard::Close(); 
    } 

    return false; 
  } 

```

`OnPaste`方法遍历构造函数中给出的粘贴格式列表，并对每个格式检查它是否在全局剪贴板上可用。如果是，则调用适当的方法。请注意，虽然`OnCopy`遍历整个复制格式列表，但`OnPaste`在剪贴板上的第一个可用格式后就会退出，这使得粘贴格式列表的顺序变得重要：

```cpp
  void StandardDocument::OnPaste() { 
    if (Clipboard::Open(this)) { 
      for (unsigned int format : pasteFormatList) { 
        bool quit = false;  
        if (Clipboard::Available(format)) { 
          switch (format) { 

```

在 ASCII 格式的情况下，`Clipboard`中的`ReadAscii`被调用，它从全局剪贴板读取文本列表，如果`IsPasteAsciiReady`返回`true`，则调用`PasteAscii`，该函数的目的是由子类覆盖以执行实际的应用特定粘贴：

```cpp
            case AsciiFormat: { 
                vector<String> asciiList; 
                if (Clipboard::ReadText<AsciiFormat,char> 
                               (asciiList) && 
                    IsPasteAsciiReady(asciiList)) { 
                  PasteAscii(asciiList); 
                  quit = true; 
                } 
              } 
              break; 

```

在 Unicode 格式的情况下，`Clipboard`中的`ReadUnicode`被调用，它从全局剪贴板读取文本列表，如果`IsPasteUnicodeReady`返回`true`，则调用`PasteUnicode`，该函数的目的是由子类覆盖以执行实际的应用特定粘贴：

```cpp
            case UnicodeFormat: { 
                vector<String> unicodeList; 
                if (Clipboard::ReadText<UnicodeFormat,wchar_t> 
                                     (unicodeList) && 
                    IsPasteUnicodeReady(unicodeList)) { 
                  PasteUnicode(unicodeList); 
                  quit = true; 
                } 
              } 
              break; 

```

如果既不适用 ASCII 也不适用 Unicode，则在`Clipboard`中调用`ReadGeneric`以从全局剪贴板读取通用信息，如果`IsPasteGenericReady`返回`true`，则调用`PasteGeneric`，该函数旨在由子类覆盖以执行实际的粘贴操作。

在通用情况下，复制和粘贴之间的一个区别是`OnCopy`使用字符列表，因为它事先不知道大小（如果我们使用内存块，我们需要两个方法：一个计算块的大小，另一个执行实际的读取，这将很麻烦），而`OnPaste`使用内存块，由于我们不知道大小，因此不能转换为字符列表。只有文档特定的覆盖版本`PasteGeneric`可以决定内存块的大小：

```cpp
            default: { 
                InfoList infoList; 
                if (Clipboard::ReadGeneric(format, infoList) && 
                    IsPasteGenericReady(format, infoList)) { 
                  PasteGeneric(format, infoList); 
                  quit = true; 
                } 
              } 
              break; 
          }  
          if (quit) { 
            break; 
          } 
        } 
      }  
      Clipboard::Close(); 
    } 
  } 

```

## 丢弃文件

当用户在窗口的客户区域拖放一个或多个文件时，我们会检查每个文件名的文件后缀。如果我们找到恰好有一个文件具有文档的文件后缀之一（`fileSuffixList`字段），我们将以与用户使用标准**打开**对话框打开它相同的方式打开它：

```cpp
  void StandardDocument::OnDropFile(vector<String> pathList) { 
    set<String> pathSet; 

```

我们遍历路径列表，并将具有文件后缀的每个路径添加到`pathSet`：

```cpp
    for (String path : pathList) { 
      for (String suffix : fileSuffixList) { 
        if (EndsWith(path, TEXT(".") + suffix)) { 
          pathSet.insert(path); 
          break; 
        } 
      } 
    } 

```

如果`pathSet`为空，则没有带有文件后缀的文件被丢弃。

```cpp
    if (pathSet.empty()) { 
      MessageBox(TEXT("No suitable dropped file."), 
                 TEXT("Drop File"), Ok, Stop); 
    } 

```

如果`pathSet`包含多个文件，则丢弃了太多带有文件后缀的文件：

```cpp
    else if (pathSet.size() > 1) { 
      MessageBox(TEXT("To many suitable dropped files."), 
                 TEXT("Drop File"), Ok, Stop); 
    } 

```

如果`pathSet`恰好包含一个文件，它将以与用户选择**打开**菜单项相同的方式读取：

```cpp
    else { 
      String path = *pathSet.begin(); 

      if (TryClose()) { 
        ClearDocument(); 
        ReadDocumentFromStream(path, ifstream(path)); 
        SetName(path); 
        SetDirty(false); 
        SetZoom(1.0); 
        Invalidate(); 
        UpdateWindow(); 
        UpdateCaret(); 
      } 
    } 
  } 

```

## 页面大小

`PageOuterSize`方法返回不考虑边距的页面大小。根据**页面设置**对话框中的方向，有两种页面大小。构造函数中给出的页面大小指的是`Portrait`方向。在`Landscape`方向的情况下，页面的宽度和高度会互换：

```cpp
  Size StandardDocument::PageOuterSize() const { 
    if (pageSetupInfo.GetOrientation() == Landscape) { 
      return Size(pageSize.Height(), pageSize.Width()); 
    } 

    return pageSize; 
  } 

```

`PageInnerSize`方法返回考虑边距的页面大小。宽度减去左和右边距。高度减去上和下边距。记住，边距是以毫米给出的，逻辑单位是毫米的百分之一。因此，我们将边距乘以 100：

```cpp
  Size StandardDocument::PageInnerSize() const { 
    Size outerSize = PageOuterSize(); 

    int innerWidth = outerSize.Width() - 
                     (100 * (pageSetupInfo.LeftMargin() + 
                     pageSetupInfo.RightMargin())), 
        innerHeight = outerSize.Height() - 
                      (100 * (pageSetupInfo.TopMargin() + 
                      pageSetupInfo.BottomMargin())); 

    return Size(innerWidth, innerHeight); 
  } 

```

`PageInnerWidth`和`PageInnerHeight`方法返回减去边距后的文档宽度和高度。由于边距是以毫米给出的，而一毫米等于一百逻辑单位，因此我们将边距乘以 100 以获得逻辑单位：

```cpp
  int StandardDocument::PageInnerWidth() const { 
    return PageOuterSize().Width() - 
           (100 * (pageSetupInfo.LeftMargin() + 
                   pageSetupInfo.RightMargin())); 
  } 

  int StandardDocument::PageInnerHeight() const { 
    return PageOuterSize().Height() - 
           (100 * (pageSetupInfo.TopMargin() + 
                   pageSetupInfo.BottomMargin())); 
  } 

```

## 页面设置

当用户选择**页面设置**菜单项时，会调用`OnPageSetup`方法。它显示**页面设置**对话框（参考第十二章，*辅助类*）并调用`OnPageSetup`，该函数旨在由子类覆盖，以通知应用程序页面设置信息已更改：

```cpp
  void StandardDocument::OnPageSetup() { 
    PageSetupDialog pageSetupDialog(this, &pageSetupInfo); 

    if (pageSetupDialog.DoModal()) { 
      OnPageSetup(pageSetupInfo); 
    } 
  } 

```

## 打印

当用户选择 **打印预览** 菜单项时，会调用 `OnPrintPreview` 方法。它显示打印预览文档，这在 第十二章 中有更详细的描述，*辅助类*。`GetTotalPages` 方法返回文档中的当前页数：

```cpp
  void StandardDocument::OnPrintPreview() { 
    new PrintPreviewDocument(this, GetTotalPages()); 
  } 

```

当用户选择 **打印** 菜单项时，会调用 `OnPrintItem` 方法。它显示标准的 **打印** 对话框，并根据用户在对话框中指定的页面间隔、顺序和副本数量打印文档的页面：

该方法被命名为 `OnPrintItem`，这样就不会与 `Window` 中的 `OnPrint` 混淆，后者在窗口接收到 `WM_PAINT` 消息时被调用。然而，这两个方法本可以都命名为 `OnPrint`，因为它们有不同的参数列表：

```cpp
void StandardDocument::OnPrintItem() { 
    int totalPages = GetTotalPages(), firstPage, lastPage, copies; 
    bool sorted; 

```

`PrintDialog` 方法创建并返回一个指向 `Graphics` 对象的指针，如果用户按下 **确定** 按钮，或者如果用户按下 **取消** 按钮，则返回一个空指针。`totalPages` 参数指示用户可以选择的最后一个可能的页面（第一个可能的页面是 1）。在按下 **确定** 按钮的情况下，`firstPage`、`lastPage`、`copies` 和 `sorted` 被初始化：`firstPage` 和 `lastPage` 是要打印的页面间隔，`copies` 是要打印的副本数，而 `sorted` 表示（如果多于一个）副本是否将被排序：

```cpp
    Graphics* graphicsPtr = 
      StandardDialog::PrintDialog(this, totalPages, firstPage, 
                                  lastPage, copies, sorted); 

```

Win32 API 函数 `StartDoc` 初始化打印过程。它通过 `Graphics` 对象获取连接到打印机的设备上下文，以及一个 `DOCINFO` 结构，该结构只需要初始化文档名称。如果 `StartDoc` 返回一个大于零的值，我们就可以打印页面。在打印过程中，我们准备设备上下文并禁用窗口：

```cpp
    if (graphicsPtr != nullptr) { 
      static DOCINFO docInfo; 
      docInfo.cbSize = sizeof docInfo; 
      docInfo.lpszDocName = GetName().c_str(); 

      if (::StartDoc(graphicsPtr->GetDeviceContextHandle(), 
                     &docInfo) > 0) { 
        PrepareDeviceContext 
          (graphicsPtr->GetDeviceContextHandle()); 
        EnableWindow(false); 

```

如果 `sorted` 为 `true`，则页面按排序顺序打印。例如，假设 `firstPage` 设置为 1，`lastPage` 设置为 3，`copies` 设置为 2。如果 `sorted` 为 `true`，则页面按顺序 1, 2, 3, 1, 2, 3 打印。如果 `sorted` 为 `false`，则按顺序 1, 1, 2, 2, 3, 3 打印。`PrintPage` 对每个页面进行调用，并且只要它返回 `true`，打印就会继续；`printOk` 跟踪循环是否继续：

```cpp
        if (sorted) { 
          bool printOk = true; 
          for (int copy = 1; (copy <= copies) && printOk; ++copy){ 
            for (int page = firstPage; 
                 (page <= lastPage) && printOk; ++page){ 
              printOk = PrintPage(graphicsPtr, page, 
                                  copy, totalPages); 
            } 
          } 
        } 
        else { 
          bool printOk = true; 
          for (int page = firstPage; 
               (page <= lastPage) && printOk; ++page) { 
            for (int copy = 1; (copy <= copies) && printOk; 
                 ++copy) { 
              printOk = PrintPage(graphicsPtr, page, 
                                  copy, totalPages); 
            } 
          } 
        } 

```

Win32 API 函数 `EndDoc` 用于完成打印：

```cpp
        ::EndDoc(graphicsPtr->GetDeviceContextHandle()); 
      } 
    } 
  } 

```

在打印页面前后，`PrintPage` 方法调用 Win32 API 函数 `StartPage` 和 `EndPage`。如果它们都返回大于零的值，则表示打印成功，返回 `true`，并且可以打印更多页面。调用 `OnPrint`（从 `Window` 中重写）来进行实际打印，`page` 和 `copy` 是当前页和副本，`totalPages` 是文档中的页数：

```cpp
  bool StandardDocument::PrintPage(Graphics* graphicsPtr, 
                              int page, int copy, int totalPages){ 
    if (::StartPage(graphicsPtr->GetDeviceContextHandle()) > 0) { 
      OnPrint(*graphicsPtr, page, copy, totalPages); 
      return (::EndPage(graphicsPtr->GetDeviceContextHandle())>0); 
    } 

    return false; 
  } 

```

`OnPrint`方法通过调用`pageSetupInfo`字段打印提供的信息。然后，通过调用`OnDraw`裁剪并绘制文档内容，如果存在，则绘制包围文档内容的框架：

```cpp
  void StandardDocument::OnPrint(Graphics& graphics, int page, 
                                 int copy, int totalPages) const { 

```

通过绘制白色来清除文档。

```cpp
    graphics.FillRectangle(Rect(0, 0, PageOuterSize().Width(), 
                      PageOuterSize().Height()), White, White); 

    int left = 100 * pageSetupInfo.LeftMargin(), 
        top = 100 * pageSetupInfo.TopMargin(); 
    int right = left + PageInnerWidth(), 
                bottom = top + PageInnerHeight(); 

```

如果当前页面是第一页，除非为空，否则会写入页眉文本：

```cpp
    if (!pageSetupInfo.HeaderText().empty() && 
        !((page == 1) && (!pageSetupInfo.HeaderFirst()))) { 
      Rect headerRect(left, 0, right, top); 
      String headerText = 
        Template(this, pageSetupInfo.HeaderText(), 
                 copy, page, totalPages); 
      Color textColor = pageSetupInfo.HeaderFont().FontColor(); 
      Color backColor = textColor.Inverse(); 
      graphics.DrawText(headerRect, headerText, 
               pageSetupInfo.HeaderFont(), textColor, backColor); 
    } 

```

与页眉文本类似，除非为空，否则会写入页脚文本；如果当前页面是第一页，则不会写入：

```cpp
    if (!pageSetupInfo.FooterText().empty() && 
        !((page == 1) && (!pageSetupInfo.HeaderFirst()))) { 
      Rect footerRect(left, bottom, right, 
                      PageOuterSize().Height()); 
      String footerText = 
        Template(this, pageSetupInfo.FooterText(), 
                 copy, page, totalPages); 
      Color textColor = pageSetupInfo.FooterFont().FontColor(); 
      Color backColor = textColor.Inverse(); 
      graphics.DrawText(footerRect, footerText, 
               pageSetupInfo.FooterFont(), textColor, backColor); 
    } 

```

保存设备上下文当前状态，将原点设置为当前页面的左上角，裁剪当前页面的区域，调用`OnDraw`以绘制当前页面，并最终恢复绘图区域：

```cpp
    int save = graphics.Save(); 
    Point centerPoint(-left, 
                      ((page - 1) * PageInnerHeight()) - top); 
    graphics.SetOrigin(centerPoint); 
    Rect clipRect(0, (page - 1) * PageInnerHeight(), 
                  PageInnerWidth(), page * PageInnerHeight()); 
    graphics.IntersectClip(clipRect); 
    OnDraw(graphics, Print); 
    graphics.Restore(save); 

```

最后，如果页面设置信息中的框架字段为`true`，则页面被矩形包围：

```cpp
    if (pageSetupInfo.Frame()) { 
      graphics.DrawRectangle(Rect(left, top, right, bottom), 
                             Black); 
    } 
  } 
}; 

```

# 摘要

在本章中，我们学习了小窗口的文档类：`Document`、`Menu`、`Accelerator`和`StandardDocument`。在第十二章《辅助类》中，我们继续探讨小窗口的辅助类。
