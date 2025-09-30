# 第五章. 图形层次结构

本章介绍了绘图程序的图形类。每个图形负责决定它是否被鼠标点击击中，或者它是否被矩形包围。它还负责移动或修改，以及绘制以及与文件流和剪贴板进行通信。

绘图图形层次结构由`Draw`、`LineFigure`、`ArrowFigure`、`RectangleFigure`和`EllipseFigure`类组成，如下图所示：

![图形层次结构](img/image_05_001.jpg)

# DrawFigure 类

`Draw`类是层次结构的根类，主要由虚拟方法和纯虚拟方法组成，旨在由子类重写。

虚拟方法和纯虚拟方法之间的区别是，虚拟方法有一个主体，并且可以被子类重写。如果子类重写了该方法，则调用其版本的方法。

如果子类没有重写方法，则调用基类的方法。纯虚拟方法通常没有主体，包含至少一个纯虚拟方法的类成为抽象类。子类可以选择重写其基类的所有纯虚拟方法或自己成为抽象类：

**Draw.h**

```cpp
enum FigureId {LineId, ArrowId, RectangleId, EllipseId}; 
class DrawDocument; 

class Draw { 
  public: 
    Draw(const Window* windowPtr); 

```

每个图形都有自己的唯一编号，由`GetId`方法返回：

```cpp
    virtual FigureId GetId() const = 0; 
    virtual void SetFirstPoint(Point point) = 0; 

```

`IsClick`方法如果鼠标点击图形则返回`True`，如果图形完全被区域包围则`IsInside`方法返回`True`。`DoubleClick`方法给图形一个执行特定于图形的操作的机会：

```cpp
    virtual bool IsClick(Point mousePoint) = 0; 
    virtual bool IsInside(Rect area) = 0; 
    virtual void DoubleClick(Point mousePoint) = 0; 

```

`Modify`和`Move`方法只是简单地移动图形。然而，`Modify`方法执行由`IsClick`方法定义的特定于图形的操作。如果用户点击了图形的一个端点，它将被修改；如果他们点击了图形的任何其他部分，它将被移动：

```cpp
    virtual void Modify(Size distanceSize) = 0; 
    virtual void Move(Size distanceSize) = 0; 

```

`Invalidate`方法通过调用返回图形占据区域的`Area`方法来使图形无效。`Draw`方法使用给定的`Graphics`类引用绘制图形：

```cpp
    virtual Rect Area() const = 0; 
    virtual void Draw(Graphics& graphics) const = 0; 
    void Invalidate() const {windowPtr->Invalidate(Area());} 

```

`IsFillable`、`IsFilled`和`Fill`方法仅由`Rectangle`和`Ellipse`方法重写：

```cpp
    virtual bool IsFillable() const {return false;} 
    virtual bool IsFilled() const {return false;} 
    virtual void Fill(bool fill) {/* Empty. */} 

```

当用户打开或保存文档时调用`WriteFigureToStream`和`ReadFigureFromStream`方法。它们将图形的信息写入和从流中读取：

```cpp
    virtual bool WriteFigureToStream(ostream& outStream) const; 
    virtual bool ReadFigureFromStream(istream& inStream); 

```

当用户复制或粘贴图形时调用`WriteFigureToClipboard`和`ReadFigureFromClipboard`方法。它们将信息写入字符列表并从字符缓冲区读取信息：

```cpp
    virtual void WriteFigureToClipboard(InfoList& infoList) const; 
    virtual void ReadFigureFromClipboard(InfoList& infoList); 

```

`color`和`marked`字段有自己的获取和设置方法：

```cpp
    bool IsMarked() const {return marked;} 
    void Mark(bool mark); 

    Color GetColor() const {return color;} 
    void SetColor(Color color); 

```

`GetCursor`方法返回图形的正确光标：

```cpp
    virtual CursoTyper GetCursor() const = 0; 

```

`MarkRadius`方法表示显示图形被标记的小正方形的大小：

```cpp
    static const Size MarkRadius; 

```

当无效化图形时使用`windowPtr`指针：

```cpp
  private: 
    const Window* windowPtr; 

```

每个图形，无论其类型如何，都有一个颜色，并且被标记或未标记：

```cpp
    Color color; 
    bool marked = false; 
};
```

**Draw.cpp**

```cpp
#include "..\\SmallWindows\\SmallWindows.h"
#include "Draw.h"
```

`MarkRadius`参数设置为 100 * 100 单位，即 1 * 1 毫米：

```cpp
const Size DrawFigure::MarkRadius(100, 100); 

```

当创建图形时，它总是未标记的。

```cpp
DrawFigure::Draw(const Window* windowPtr) 
 :windowPtr(windowPtr) { 
    // Empty. 
} 

```

当用户切换图形的标记状态时，我们会重新绘制。你可能注意到了`if...else`语句中的不同顺序。原因是当我们标记一个图形时，它会变得更大；这就是为什么我们首先将`marked`参数设置为`True`，然后使图形无效以捕获包括标记在内的区域。另一方面，当我们取消标记图形时，它会变得更小；这就是为什么我们首先使图形无效以捕获包括标记在内的区域，然后设置`marked`参数为`False`。

```cpp
void DrawFigure::Mark(bool mark) { 
  if (!marked && mark) { 
    marked = true; 
    Invalidate(); 
  } 
  else if (marked && !mark) { 
    Invalidate(); 
    marked = false; 
  } 
} 

```

颜色是文件处理和与剪贴板通信中唯一写入或读取的字段。`DrawFigure`类的子类调用这些方法，然后写入和读取特定于图形的信息。`WriteFigureToStream`和`ReadFigureFromStream`方法返回流的布尔值，以指示文件操作是否成功。

```cpp
bool DrawFigure::WriteFigureToStream(ostream& outStream) const { 
  color.WriteColorToStream(outStream); 
  return ((bool) outStream); 
} 

bool DrawFigure::ReadFigureFromStream(istream& inStream) { 
  color.ReadColorFromStream(inStream); 
  return ((bool) inStream); 
} 

void DrawFigure::WriteFigureToClipboard(InfoList& infoList) const{ 
  color.WriteColorToClipboard(charList); 
} 

void DrawFigure::ReadFigureFromClipboard(InfoList& infoList) { 
  color.ReadColorFromClipboard(infoList); 
} 

```

# LineFigure 类

两个点之间绘制线条，这些点由`LineFigure`类中的`firstPoint`字段到`lastPoint`字段表示，如下面的图像所示：

![LineFigure 类](img/image_05_002.jpg)

`header`文件覆盖了其`DrawFigure`基类的一些方法。`DoubleClick`方法不做任何事情。在我看来，对于线条的双击并没有真正有意义的响应。然而，我们仍然需要覆盖`DoubleClick`方法，因为它是`DrawFigure`基类中的一个纯虚方法。如果我们不覆盖它，`LineFigure`类将变成抽象的。

**LineFigure.h**

```cpp
class LineFigure : public DrawFigure { 
  public: 
    LineFigure(const Window* windowPtr); 
    virtual FigureId GetId() const {return LineId;} 
    virtual void SetFirstPoint(Point point); 

    virtual bool IsClick(Point mousePoint); 
    virtual bool IsInside(Rect rectangleArea); 
    virtual void DoubleClick(Point mousePoint) {/* Empty. */} 

    virtual void Modify(Size distanceSize); 
    virtual void Move(Size distanceSize); 

    virtual Rect Area() const; 
    virtual void Draw(Graphics& graphics) const; 
    virtual CursorType GetCursor() const; 

    virtual bool WriteFigureToStream(ostream& outStream) const; 
    virtual bool ReadFigureFromStream(istream& inStream); 

    virtual void WriteFigureToClipboard(InfoList& infoList) const; 
    virtual void ReadFigureFromClipboard(InfoList& infoList); 

  protected: 
    enum {CreateLine, FirstPoint, LastPoint, MoveLine} lineMode; 
    Point firstPoint, lastPoint; 
    static bool IsPointInLine(Point firstPoint, Point lastPoint, 
                              Point point); 
};
```

**LineFigure.cpp**

```cpp
#include "..\\SmallWindows\\SmallWindows.h"
#include "Draw.h"
#include "LineFigure.h"

LineFigure::LineFigure(const Window* windowPtr)
 :Draw(windowPtr), lineMode(CreateLine) {
 // Empty. 
}
```

当创建线条时，会调用`SetFirstPoint`方法，并设置第一个和最后一个点。

```cpp
void LineFigure::SetFirstPoint(Point point) { 
  firstPoint = point; 
  lastPoint = point; 
} 

```

`IsClick`方法有两个情况：用户必须击中端点之一或线条本身。我们定义了两个覆盖端点的正方形（`firstSquare`和`lastSquare`），并测试鼠标是否击中其中之一。如果没有，我们通过调用`IsPointInLine`方法测试鼠标是否击中线条本身。

```cpp
bool LineFigure::IsClick(Point mousePoint) { 
  Rect firstSquare(firstPoint - MarkRadius, 
                   firstPoint + MarkRadius); 
  firstSquare.Normalize(); 

  if (firstSquare.PointInside(mousePoint)) { 
    lineMode = FirstPoint; 
    return true; 
  } 

  Rect lastSquare(lastPoint - MarkRadius, lastPoint + MarkRadius); 
  lastSquare.Normalize(); 

  if (lastSquare.PointInside(mousePoint)) { 
    lineMode = LastPoint; 
    return true; 
  } 

  if (IsPointInLine(firstPoint, lastPoint, mousePoint)) { 
    lineMode = MoveLine; 
    return true; 
  } 

  return false; 
} 

```

`IsPointInLine`方法检查点是否位于线上，并有一定的容差。我们使用三角函数来计算点相对于线的位置。然而，如果线完全垂直且点的 x 坐标相同，我们有一个特殊情况。

应用三角函数会导致除以零。相反，我们创建一个围绕线的矩形，并检查点是否位于矩形内，如下面的图像所示：

![LineFigure 类](img/image_05_003.jpg)

```cpp
bool LineFigure::IsPointInLine(Point firstPoint, Point lastPoint, 
                               Point point) { 
  if (firstPoint.X() == lastPoint.X()) { 
    Rect lineRect(firstPoint - MarkRadius, 
                  lastPoint + MarkRadius); 
    lineRect.Normalize(); 
    return lineRect.PointInside(point); 
  } 

```

如果线条不是垂直的，我们首先创建一个包围矩形并测试鼠标点是否在其中。如果是，我们将 `firstPoint` 和 `lastPoint` 字段的左端点等于 `minPoint` 字段，右端点等于 `maxPoint` 字段。然后我们计算包围矩形的宽度（`lineWidth`）和高度（`lineHeight`），以及 `minPoint` 和 `mousePoint` 字段在 x 和 y 方向上的距离（`diffWidth` 和 `diffHeight`），如下面的图像所示：

![The LineFigure class](img/image_05_004.jpg)

由于一致性，如果鼠标点击中线条，以下方程是正确的：

![The LineFigure class](img/image_05_005.jpg)

这意味着：

![The LineFigure class](img/image_05_006.jpg)

这也意味着：

![The LineFigure class](img/image_05_007.jpg)

让我们允许有一点容差；让我们说用户被允许错过线条 1 毫米（100 单位）。这改变了最后一个方程到以下方程：

![The LineFigure class](img/image_05_008.jpg)

```cpp
  else { 
    Point minPoint = Min(firstPoint, lastPoint), 
          maxPoint = Max(firstPoint, lastPoint); 

    if ((minPoint.X() <= point.X()) && 
        (point.X() <= maxPoint.X())) { 
      int lineWidth = maxPoint.X() - minPoint.X(), 
          lineHeight = maxPoint.Y() - minPoint.Y(); 

      int diffWidth = point.X() - minPoint.X(), 
          diffHeight = point.Y() - minPoint.Y(); 

      double delta = fabs(diffHeight - (diffWidth * 
                          ((double) lineHeight) / lineWidth)); 
      return (delta <= 100); 
    } 

    return false; 
  } 
} 

```

`IsInside` 方法比 `IsClick` 方法简单。我们只需检查两个端点是否都被给定的矩形包围。

```cpp
bool LineFigure::IsInside(Rect rect) { 
  return (rect.PointInside(firstPoint) && 
          rect.PointInside(lastPoint)); 
} 

```

在 `Modify` 模式下，我们根据 `IsClick` 方法设置的 `lineMode` 参数的值移动一个端点或线条。如果用户击中了第一个点，我们就移动它。如果他们击中了最后一个点，或者如果线条正在创建过程中，我们就移动最后一个点。如果他们击中了线条，我们就移动线条。也就是说，我们移动第一个和最后一个点。

```cpp
void LineFigure::Modify(Size distanceSize) { 
  Invalidate(); 
  switch (lineMode) { 
    case FirstPoint: 
      firstPoint += distanceSize; 
      break; 

    case CreateLine: 
    case LastPoint: 
      lastPoint += distanceSize; 
      break; 

    case MoveLine: 
      Move(distanceSize); 
      break; 
  } 

  Invalidate(); 
} 

```

`Move` 方法也很简单；我们只需移动两个端点。

```cpp
void LineFigure::Move(Size distanceSize) { 
  Invalidate(); 
  firstPoint += distanceSize; 
  lastPoint += distanceSize; 
  Invalidate(); 
} 

```

在 `Draw` 方法中，我们绘制线条，如果线条被标记，则其两个端点始终是黑色。

```cpp
void LineFigure::Draw(Graphics& graphics) const { 
  graphics.DrawLine(firstPoint, lastPoint, GetColor()); 

  if (IsMarked()) { 
    graphics.FillRectangle(Rect(firstPoint - MarkRadius, 
                           firstPoint + MarkRadius), Black,Black); 
    graphics.FillRectangle(Rect(lastPoint - MarkRadius, 
                           lastPoint + MarkRadius), Black, Black); 
  } 
} 

```

线条占据的区域是一个以端点为顶点的矩形。如果线条被标记，则标记半径被添加。

```cpp
Rect LineFigure::Area() const { 
  Rect lineArea(firstPoint.X(), firstPoint.Y(), 
                lastPoint.X(), lastPoint.Y()); 
  lineArea.Normalize(); 

  if (IsMarked()) { 
    lineArea -= MarkRadius; 
    lineArea += MarkRadius; 
  } 

  return lineArea; 
} 

```

如果正在修改线条，则返回 `Crosshair` 光标。如果正在移动，则返回全选光标（四个指向方位的箭头）。如果没有这些情况，则我们只返回正常的箭头光标。

```cpp
CursorType LineFigure::GetCursor() const { 
  switch (lineMode) { 
    case CreateLine: 
    case FirstPoint: 
    case LastPoint: 
      return Cursor::Crosshair; 

    case MoveLine: 
      return Cursor::SizeAll; 

    default: 
      return Cursor::Normal; 
  } 
} 

```

`WriteFigureToStream`、`ReadFigureFromStream`、`WriteFigureToClipboard` 和 `ReadFigureFromClipboard` 方法在调用 `DrawFigure` 类中的相应方法后，写入和读取线的第一个和最后一个端点。

```cpp
bool LineFigure::WriteFigureToStream(ostream& outStream) const { 
  DrawFigure::WriteFigureToStream(outStream); 
  firstPoint.WritePointToStream(outStream); 
  lastPoint.WritePointToStream(outStream); 
  return ((bool) outStream); 
} 

bool LineFigure::ReadFigureFromStream (istream& inStream) { 
  DrawFigure::ReadFigureFromStream(inStream); 
  firstPoint.ReadPointFromStream(inStream); 
  lastPoint.ReadPointFromStream(inStream); 
  return ((bool) inStream); 
} 

void LineFigure::WriteFigureToClipboard(InfoList& infoList) const{ 
  DrawFigure::WriteFigureToClipboard(charList); 
  firstPoint.WritePointToClipboard(charList); 
  lastPoint.WritePointToClipboard(charList); 
} 

void LineFigure::ReadFigureFromClipboard(InfoList& infoList) { 
  DrawFigure::ReadFigureFromClipboard(infoList); 
  firstPoint.ReadPointFromClipboard(infoList); 
  lastPoint.ReadPointFromClipboard(infoList); 
} 

```

# ArrowFigure 类

`ArrowFigure` 是 `LineFigure` 类的子类，并重用了 `firstPoint` 和 `lastPoint` 字段以及一些功能。箭头端点存储在 `leftPoint` 和 `rightPoint` 字段中，如下面的图像所示。边的长度由 `ArrowLength` 常量定义为 500 单位，即 5 毫米。

![The ArrowFigure class](img/image_05_010.jpg)

`ArrowFigure` 类覆盖了 `LineFigure` 类的一些方法。主要的是，它调用 `LineFigure` 类的方法，然后添加自己的功能。

**ArrowFigure.h**

```cpp
class ArrowFigure : public LineFigure { 
  public: 
    ArrowFigure(const Window* windowPtr); 
    FigureId GetId() const {return ArrowId;}; 

    bool IsClick(Point mousePoint); 
    bool IsInside(Rect area); 
    void DoubleClick(Point mousePoint); 

    void Modify(Size distanceSize); 
    void Move(Size distanceSize); 
    Rect Area() const; 
    void Draw(Graphics& graphics) const; 

    bool WriteFigureToStream(ostream& outStream) const; 
    bool ReadFigureFromStream(istream& inStream); 

    void WriteFigureToClipboard(InfoList& infoList) const; 
    void ReadFigureFromClipboard(InfoList& infoList); 

  private: 
    static const int ArrowLength = 500; 
    Point leftPoint, rightPoint; 
    void CalculateArrowHead(); 
}; 

```

构造函数允许 `LineFigure` 构造函数初始化箭头的端点，然后调用 `CalculateArrowHead` 方法来计算箭头端点。

**ArrowFigure.cpp**

```cpp
#include "..\\SmallWindows\\SmallWindows.h" 
#include "Draw.h" 
#include "LineFigure.h" 
#include "ArrowFigure.h" 

ArrowFigure::ArrowFigure(const Window* windowPtr) 
 :LineFigure(windowPtr) { 
  CalculateArrowHead(); 
} 

```

`IsClick` 方法如果用户点击线或箭头的任何部分，则返回 `True`。

```cpp
bool ArrowFigure::IsClick(Point mousePoint) { 
  return LineFigure::IsClick(mousePoint) || 
         IsPointInLine(firstPoint, leftPoint, mousePoint) || 
         IsPointInLine(firstPoint, rightPoint, mousePoint) || 
         IsPointInLine(leftPoint, rightPoint, mousePoint); 
} 

```

`IsInside` 方法如果线的所有端点和箭头都在区域内，则返回 `True`。

```cpp
bool ArrowFigure::IsInside(Rect area) { 
  return area.PointInside(firstPoint) && 
         area.PointInside(lastPoint) && 
         area.PointInside(leftPoint) && 
         area.PointInside(rightPoint); 
} 

```

`Modify` 方法修改线并重新计算箭头。

```cpp
void ArrowFigure::Modify(Size distanceSize) { 
  LineFigure::Modify(distanceSize); 
  CalculateArrowHead(); 
} 

```

`Move` 方法移动线和箭头。

```cpp
void ArrowFigure::Move(Size distanceSize) { 
  LineFigure::Move(distanceSize); 
  leftPoint += distanceSize; 
  rightPoint += distanceSize; 
} 

```

当用户双击箭头时，其头部和尾部会交换。

```cpp
void ArrowFigure::DoubleClick(Point mousePoint) { 
  if (IsClick(mousePoint)) { 
    Invalidate(); 
    Point tempPoint = firstPoint; 
    firstPoint = lastPoint; 
    lastPoint = tempPoint; 
    CalculateArrowHead(); 
    Invalidate(); 
  } 
} 

```

`Area` 方法计算线和箭头端点的最小和最大值，并返回一个包含其左上角和右下角的区域。如果箭头被标记，则将标记半径添加到区域中。

```cpp
Rect ArrowFigure::Area() const { 
  Point topLeft(min(firstPoint.X(), min(lastPoint.X(), 
                    min(leftPoint.X(), rightPoint.X()))), 
                min(firstPoint.Y(), min(lastPoint.Y(), 
                    min(leftPoint.Y(), rightPoint.Y())))), 
        bottomRight(max(firstPoint.X(), max(lastPoint.X(), 
                        max(leftPoint.X(), rightPoint.X()))), 
                    max(firstPoint.Y(), max(lastPoint.Y(), 
                        max(leftPoint.Y(), rightPoint.Y())))); 

  if (IsMarked()) { 
    topLeft -= MarkRadius; 
    bottomRight += MarkRadius; 
  } 

  return Rect(topLeft, bottomRight); 
} 

```

`Draw` 方法绘制线和箭头。如果箭头被标记，箭头的端点也会用方块标记。

```cpp
void ArrowFigure::Draw(Graphics& graphics) const { 
  LineFigure::Draw(graphics); 

  graphics.DrawLine(lastPoint, leftPoint, GetColor()); 
  graphics.DrawLine(lastPoint, rightPoint, GetColor()); 
  graphics.DrawLine(leftPoint, rightPoint, GetColor());  
  if (IsMarked()) { 
    graphics.FillRectangle(Rect(leftPoint - MarkRadius, 
                           leftPoint + MarkRadius), Black, Black); 
    graphics.FillRectangle(Rect(rightPoint - MarkRadius, 
                           rightPoint + MarkRadius), Black,Black); 
  } 
} 

```

`WriteFigureToStream`、`ReadFigureFromStream`、`WriteFigureToClipboard` 和 `ReadFigureFromClipboard` 方法允许 `LineFigure` 类写入和读取线的端点。然后它写入和读取箭头端点。

```cpp
bool ArrowFigure::WriteFigureToStream(ostream& outStream) const { 
  LineFigure::WriteFigureToStream(outStream); 
  leftPoint.WritePointToStream(outStream); 
  rightPoint.WritePointToStream(outStream); 
  return ((bool) outStream); 
} 

bool ArrowFigure::ReadFigureFromStream(istream& inStream) { 
  LineFigure::ReadFigureFromStream(inStream); 
  leftPoint.ReadPointFromStream(inStream); 
  rightPoint.ReadPointFromStream(inStream); 
  return ((bool) inStream); 
} 

void ArrowFigure::WriteFigureToClipboard(InfoList& infoList)const{ 
  LineFigure::WriteFigureToClipboard(charList); 
  leftPoint.WritePointToClipboard(charList); 
  rightPoint.WritePointToClipboard(charList); 
} 

void ArrowFigure::ReadFigureFromClipboard(InfoList& infoList) { 
  LineFigure::ReadFigureFromClipboard(infoList); 
  leftPoint.ReadPointFromClipboard(infoList); 
  rightPoint.ReadPointFromClipboard(infoList); 
} 

```

`CalculateArrowHead` 方法是一个私有辅助方法，用于计算箭头端点。我们将使用以下关系来计算 `leftPoint` 和 `rightPoint` 字段。

![The ArrowFigure class](img/image_05_011.jpg)

计算分为三个步骤；首先我们计算 `alpha` 和 `beta`。请参见以下插图以了解角度的定义：

![The ArrowFigure class](img/image_05_012.jpg)

然后我们计算 `leftAngle` 和 `rightAngle`，并使用它们的值来计算 `leftPoint` 和 `rightPoint` 的值。线和箭头部分的夹角是 45 度，相当于 Π/4 弧度。因此，为了确定箭头部分的角，我们只需从 `beta` 中减去 Π/4 并将 Π/4 加到 `beta` 上：

![The ArrowFigure class](img/image_05_013.jpg)

然后我们使用以下公式最终确定 `leftPoint` 和 `rightPoint`：

![The ArrowFigure class](img/image_05_014.jpg)

三角函数在 C 标准库中可用。然而，我们需要定义我们的 Π 值。`atan2` 函数计算 `height` 和 `width` 的比例的切线值，并考虑 `width` 可能为零的可能性。

![The ArrowFigure class](img/image_05_018.jpg)

```cpp
void ArrowFigure::CalculateArrowHead() { 
  int height = lastPoint.Y() - firstPoint.Y(); 
  int width = lastPoint.X() - firstPoint.X(); 

  const double Pi = 3.14159265; 
  double alpha = atan2((double) height, (double) width); 
  double beta = alpha + Pi; 

  double leftAngle = beta - (Pi / 4); 
  double rightAngle = beta + (Pi / 4); 

  leftPoint.X() = lastPoint.X() + 
                  (int) (ArrowLength * cos(leftAngle)); 
  leftPoint.Y() = lastPoint.Y() + 
                  (int) (ArrowLength * sin(leftAngle)); 

  rightPoint.X() = lastPoint.X() + 
                   (int) (ArrowLength * cos(rightAngle)); 
  rightPoint.Y() = lastPoint.Y() + 
                   (int) (ArrowLength * sin(rightAngle)); 
} 

```

# RectangleFigure 类

`RectangleFigure` 类包含一个矩形，可以是填充的或不填充的。用户可以通过抓住其四个角之一来修改它。`DrawRectangle` 类覆盖了 `DrawFigure` 类的大部分方法。

与线和箭头的情况相比，一个区别是矩形是二维的，可以是填充的或未填充的。`Fillable` 方法返回 `True`，`IsFilled` 和 `Fill` 方法被覆盖。当用户双击矩形时，它将在填充和未填充状态之间切换。

**RectangleFigure.h**

```cpp
class RectangleFigure : public DrawFigure { 
  public: 
    RectangleFigure(const Window* windowPtr); 

    virtual void SetFirstPoint(Point point); 
    virtual FigureId GetId() const {return RectangleId;} 

    virtual bool IsClick(Point mousePoint); 
    virtual bool IsInside(Rect rectangleArea); 
    virtual void DoubleClick(Point mousePoint); 

    virtual void Modify(Size distanceSize); 
    virtual void Move(Size distanceSize); 

    virtual Rect Area() const; 
    virtual void Draw(Graphics& graphics) const; 
    virtual CursorType GetCursor() const; 

    bool IsFillable() const {return true;} 
    bool IsFilled() const {return filled;} 
    void Fill(bool fill) {filled = fill; Invalidate();} 

    virtual bool WriteFigureToStream(ostream& outStream) const; 
    virtual bool ReadFigureFromStream(istream& inStream); 

    virtual void WriteFigureToClipboard(InfoList& infoList) const; 
    virtual void ReadFigureFromClipboard(InfoList& infoList); 

  private: 
    enum {CreateRectangle, TopLeftPoint, TopRightPoint, 
          BottomRightPoint, BottomLeftPoint, MoveRectangle} 
         rectangleMode; 

  protected: 
    bool filled = false; 
    Point topLeft, bottomRight; 
};
```

**RectangleFigure.cpp**

```cpp
#include "..\\SmallWindows\\SmallWindows.h"
#include "Draw.h"
#include "RectangleFigure.h"
RectangleFigure::RectangleFigure(const Window* windowPtr)
:Draw(windowPtr), rectangleMode(CreateRectangle) { /* Empty. */ }
void RectangleFigure::SetFirstPoint(Point point) { topLeft = point; bottomRight = point; }
```

当用户点击矩形时，他们可能会点击其四个角落之一、矩形的边缘，或者（如果它被填充）其内部。首先，我们检查角落，然后是矩形本身。如果它被填充，我们只需测试鼠标点是否在矩形内。如果矩形未被填充，我们通过构建一个稍微小一点的矩形和一个稍微大一点的矩形来测试是否点击了其任何四个边缘。如果鼠标位置包含在较大的矩形内，但不包含在较小的矩形内，则用户点击了矩形的边缘。

![The RectangleFigure class](img/image_05_019.jpg)

```cpp
bool RectangleFigure::IsClick(Point mousePoint) { 
  Rect topLeftRect(topLeft - MarkRadius, topLeft + MarkRadius); 

  if (topLeftRect.PointInside(mousePoint)) { 
    rectangleMode = TopLeftPoint; 
    return true; 
  } 

  Point topRightPoint(bottomRight.X(), topLeft.Y()); 
  Rect topRectRight(topRightPoint - MarkRadius, 
                    topRightPoint + MarkRadius); 

  if (topRectRight.PointInside(mousePoint)) { 
    rectangleMode = TopRightPoint; 
    return true; 
  } 

  Rect bottomRightRect(bottomRight - MarkRadius, 
                       bottomRight + MarkRadius); 

  if (bottomRightRect.PointInside(mousePoint)) { 
    rectangleMode = BottomRightPoint; 
    return true; 
  } 

  Point bottomLeftPoint(topLeft.X(), bottomRight.Y()); 
  Rect bottomLeftRect(bottomLeftPoint - MarkRadius, 
                      bottomLeftPoint + MarkRadius); 

  if (bottomLeftRect.PointInside(mousePoint)) { 
    rectangleMode = BottomLeftPoint; 
    return true; 
  } 

  Rect areaRect(topLeft, bottomRight); 
  areaRect.Normalize(); 

  if (IsFilled()) { 
    areaRect.PointInside(mousePoint);    

    if (areaRect.PointInside(mousePoint)) { 
      rectangleMode = MoveRectangle; 
      return true; 
    } 
  } 

  else { 
    Rect largeAreaRect(areaRect.TopLeft() - MarkRadius, 
                       areaRect.BottomRight() + MarkRadius), 
         smallAreaRect(areaRect.TopLeft() + MarkRadius, 
                       areaRect.BottomRight() - MarkRadius); 

    if (largeAreaRect.PointInside(mousePoint) && 
        !smallAreaRect.PointInside(mousePoint)) { 
      rectangleMode = MoveRectangle; 
      return true; 
    } 
  } 

  return false; 
} 

```

`IsInside` 方法如果矩形的左上角和右下角被矩形区域包围，则返回 `true`。

```cpp
bool RectangleFigure::IsInside(Rect area) { 
  return area.PointInside(topLeft) && 
         area.PointInside(bottomRight); 
} 

```

`DoubleClick` 方法如果矩形未被填充则填充它，反之亦然。

```cpp
void RectangleFigure::DoubleClick(Point mousePoint) { 
  if (IsClick(mousePoint)) { 
    filled = !filled; 
    Invalidate(); 
  } 
} 

```

`Modify` 方法根据 `IsClick` 方法中 `rectangleMode` 参数的设置修改或移动矩形。

```cpp
void RectangleFigure::Modify(Size distanceSize) { 
  Invalidate(); 

  switch (rectangleMode) { 
    case TopLeftPoint: 
      topLeft += distanceSize; 
      break; 

    case TopRightPoint: 
      topLeft.Y() += distanceSize.Height(); 
      bottomRight.X() += distanceSize.Width(); 
      break; 

    case CreateRectangle: 
    case BottomRightPoint: 
      bottomRight += distanceSize; 
      break; 

    case BottomLeftPoint: 
      topLeft.X() += distanceSize.Width(); 
      bottomRight.Y() += distanceSize.Height(); 
      break; 

    case MoveRectangle: 
      Move(distanceSize); 
      break; 
  } 

  Invalidate(); 
} 

```

`Move` 方法移动矩形的角落。

```cpp
void RectangleFigure::Move(Size distanceSize) { 
  Invalidate(); 
  topLeft += distanceSize; 
  bottomRight += distanceSize; 
  Invalidate(); 
} 

```

矩形的面积简单地说就是矩形的面积。然而，如果它被标记，我们会增加它以包括角落的正方形。

![The RectangleFigure class](img/image_05_020.jpg)

```cpp
Rect RectangleFigure::Area() const {
  Rect areaRect(topLeft, bottomRight); 
  areaRect.Normalize(); 

  if (IsMarked()) { 
    areaRect -= MarkRadius; 
    areaRect += MarkRadius; 
  }  
  return areaRect; 
} 

```

`Draw` 方法绘制或填充矩形。如果它被标记，它还会填充正方形。

```cpp
void RectangleFigure::Draw(Graphics& graphics) const { 
  if (filled) { 
    graphics.FillRectangle(Rect(topLeft, bottomRight), 
                           GetColor(), GetColor()); 
  } 
  else { 
    graphics.DrawRectangle(Rect(topLeft, bottomRight), 
                           GetColor()); 
  } 

  if (IsMarked()) { 
    graphics.FillRectangle(Rect(topLeft - MarkRadius, 
                           topLeft + MarkRadius), Black, Black); 

    Point topRight(bottomRight.X(), topLeft.Y()); 
    graphics.FillRectangle(Rect(topRight - MarkRadius, 
                           topRight + MarkRadius), Black, Black); 

    graphics.FillRectangle(Rect(bottomRight - MarkRadius, 
                           bottomRight + MarkRadius),Black,Black); 

    Point bottomLeft(topLeft.X(), bottomRight.Y()); 
    graphics.FillRectangle(Rect(bottomLeft - MarkRadius, 
                 bottomLeft + MarkRadius.Height()), Black, Black); 
  } 
} 

```

当图形被移动时，矩形的指针是大小全指针（四个方向上的箭头）。它是一个根据抓取的角落而修改时带有箭头的指针：如果是最左上角或最右下角，则使用西北和东南箭头；如果是右上角或左下角，则使用东北和西南箭头。

```cpp
CursorType RectangleFigure::GetCursor() const { 
  switch (rectangleMode) { 
    case TopLeftPoint: 
    case BottomRightPoint: 
      return Cursor::SizeNorthWestSouthEast; 

    case TopRightPoint: 
    case BottomLeftPoint: 
      return Cursor::SizeNorthEastSouthWest; 

    case MoveRectangle: 
      return Cursor::SizeAll; 

    default: 
      return Cursor::Normal; 
  } 
} 

```

`WriteFigureToStream`、`ReadFigureFromStream`、`WriteFigureToClipboard` 和 `ReadFigureFromClipboard` 方法调用 `DrawFigure` 类中的相应方法。然后它们写入和读取矩形的四个角，以及它是否被填充。

```cpp
bool RectangleFigure::WriteFigureToStream(ostream& outStream) 
                                          const { 
  DrawFigure::WriteFigureToStream(outStream); 
  topLeft.WritePointToStream(outStream); 
  bottomRight.WritePointToStream(outStream); 
  outStream.write((char*) &filled, sizeof filled); 
  return ((bool) outStream); 
} 

bool RectangleFigure::ReadFigureFromStream (istream& inStream) { 
  DrawFigure::ReadFigureFromStream(inStream); 
  topLeft.ReadPointFromStream(inStream); 
  bottomRight.ReadPointFromStream(inStream); 
  inStream.read((char*) &filled, sizeof filled); 
  return ((bool) inStream); 
} 

void RectangleFigure::WriteFigureToClipboard(InfoList& infoList) 
                                             const { 
  DrawFigure::WriteFigureToClipboard(infoList); 
  topLeft.WritePointToClipboard(infoList); 
  bottomRight.WritePointToClipboard(infoList); 
  infoList.AddValue<bool>(filled); 
} 

void RectangleFigure::ReadFigureFromClipboard(InfoList& infoList) { 
  DrawFigure::ReadFigureFromClipboard(infoList); 
  topLeft.ReadPointFromClipboard(infoList); 
  bottomRight.ReadPointFromClipboard(infoList); 
  infoList.GetValue<bool>(filled); 
} 

```

# The EllipseFigure 类

`EllipseFigure` 类是 `RectangleFigure` 类的子类。椭圆可以通过水平或垂直角落移动或重塑。`RectangleFigure` 类的大多数方法都没有被 `Ellipse` 类覆盖。

**Ellipse.h**

```cpp
class EllipseFigure : public RectangleFigure { 
  public: 
    EllipseFigure(const Window* windowPtr); 
    FigureId GetId() const {return EllipseId;} 

    bool IsClick(Point mousePoint); 
    void Modify(Size distanceSize); 
    void Draw(Graphics& graphics) const; 
    CursoTyper GetCursor() const; 

  private: 
    enum {CreateEllipse, LeftPoint, TopPoint, RightPoint, 
          BottomPoint, MoveEllipse} ellipseMode; 
};
```

**Ellipse.cpp**

```cpp
#include "..\\SmallWindows\\SmallWindows.h"
#include "Draw.h"
#include "RectangleFigure.h"
#include "EllipseFigure.h"
EllipseFigure::EllipseFigure(const Window* windowPtr) 
 :RectangleFigure(windowPtr), 
  ellipseMode(CreateEllipse) { 
  // Empty. 
} 

```

正如矩形的情况一样，`IsClick` 方法首先决定用户是否点击了四个端点之一；然而，与矩形角落的位置相比，这些位置是不同的。

![The EllipseFigure class](img/image_05_021.jpg)

```cpp
bool EllipseFigure::IsClick(Point mousePoint) { 
  Point leftPoint(topLeft.X(), (topLeft.Y() + bottomRight.Y())/2); 
  Rect leftRect(leftPoint - MarkRadius, leftPoint + MarkRadius); 

  if (leftRect.PointInside(mousePoint)) { 
    ellipseMode = LeftPoint; 
    return true; 
  } 

  Point topPoint((topLeft.X() + bottomRight.X()) / 2,topLeft.Y()); 
  Rect topRect(topPoint - MarkRadius, topPoint + MarkRadius); 

  if (topRect.PointInside(mousePoint)) { 
    ellipseMode = TopPoint; 
    return true; 
  } 

  Point rightPoint(bottomRight.X(), 
                   (topLeft.Y() + bottomRight.Y()) / 2); 
  Rect rightRect(rightPoint - MarkRadius, 
                 rightPoint + MarkRadius); 

  if (rightRect.PointInside(mousePoint)) { 
    ellipseMode = RightPoint; 
    return true; 
  } 
  Point bottomPoint((topLeft.X() + bottomRight.X()) / 2, 
                    bottomRight.Y()); 
  Rect bottomRect(bottomPoint - MarkRadius, 
                  bottomPoint + MarkRadius); 

  if (bottomRect.PointInside(mousePoint)) { 
    ellipseMode = BottomPoint; 
    return true; 
  } 

```

如果用户没有点击任何一个修改位置，我们必须决定用户是否点击了椭圆本身。如果椭圆没有填充，这相当简单。我们使用 Win32 API 函数`CreateEllipticRgn`创建一个椭圆区域，并测试鼠标位置是否在其中。如果椭圆没有填充，我们创建两个区域，一个稍微小一些，一个稍微大一些。如果鼠标位置包含在较大的区域中，但不包含在较小的区域中，则表示发生了点击。

![EllipseFigure 类](img/image_05_022.jpg)

```cpp
  ellipseMode = MoveEllipse; 
  Point minPoint = Min(topLeft, bottomRight), 
        maxPoint = Max(topLeft, bottomRight);  
  if (IsFilled()) { 
    HRGN ellipseRegion = 
      CreateEllipticRgn(minPoint.X(), minPoint.Y(), 
                        maxPoint.X(), maxPoint.Y()); 
    return (PtInRegion(ellipseRegion, mousePoint.X(), 
                       mousePoint.Y()) != 0); 
  } 
  else { 
    HRGN smallRegion = 
      CreateEllipticRgn(minPoint.X() + MarkRadius.Width(), 
                        minPoint.Y() + MarkRadius.Height(), 
                        maxPoint.X() - MarkRadius.Width(), 
                        maxPoint.Y() - MarkRadius.Height()); 
    HRGN largeRegion = 
      CreateEllipticRgn(minPoint.X() - MarkRadius.Width(), 
                        minPoint.Y() - MarkRadius.Height(), 
                        maxPoint.X() + MarkRadius.Width(), 
                        maxPoint.Y() + MarkRadius.Height()); 
    return ((PtInRegion(largeRegion, mousePoint.X(), 
                        mousePoint.Y()) != 0) && 
            (PtInRegion(smallRegion, mousePoint.X(), 
                        mousePoint.Y()) == 0)); 
  } 

  return false; 
} 

```

`Modify`方法根据`IsClick`方法中`ellipseMode`参数的设置移动角落。

```cpp
void EllipseFigure::Modify(Size distanceSize) { 
  Invalidate(); 

  switch (ellipseMode) { 
    case CreateEllipse: 
      bottomRight += distanceSize; 
      break; 

    case LeftPoint: 
      topLeft.X() += distanceSize.Width(); 
      break; 

    case RightPoint: 
      bottomRight.X() += distanceSize.Width(); 
      break; 

    case TopPoint: 
      topLeft.Y() += distanceSize.Height(); 
      break; 

    case BottomPoint: 
      bottomRight.Y() += distanceSize.Height(); 
      break; 

    case MoveEllipse: 
      Move(distanceSize); 
      break; 
  } 

  Invalidate(); 
} 

```

`Draw`方法填充或绘制椭圆，如果椭圆被标记，则绘制四个方块。

```cpp
void EllipseFigure::Draw(Graphics& graphics) const { 
  if (filled) { 
    graphics.FillEllipse(Rect(topLeft, bottomRight), 
                         GetColor(), GetColor()); 
  } 
  else { 
    graphics.DrawEllipse(Rect(topLeft, bottomRight), GetColor()); 
  } 

  if (IsMarked()) { 
    Point leftPoint(topLeft.X(), (topLeft.Y()+bottomRight.Y())/2); 
    graphics.FillRectangle(Rect(leftPoint - MarkRadius,  
                           leftPoint + MarkRadius), Black, Black); 

    Point topPoint((topLeft.X() + bottomRight.X())/2,topLeft.Y()); 
    graphics.FillRectangle(Rect(topPoint - MarkRadius, 
                           topPoint + MarkRadius),Black, Black); 

    Point rightPoint(bottomRight.X(), 
                     (topLeft.Y() + bottomRight.Y()) / 2); 
    graphics.FillRectangle(Rect(rightPoint - MarkRadius, 
                           rightPoint + MarkRadius), Black,Black); 

    Point bottomPoint((topLeft.X() + bottomRight.X()) / 2, 
                      bottomRight.Y()); 
    graphics.FillRectangle(Rect(bottomPoint - MarkRadius, 
                           bottomPoint + MarkRadius),Black,Black); 
  } 
} 

```

最后，关于光标，我们有以下五种不同的情况：

+   当椭圆正在创建时，会返回十字光标

+   当用户抓住椭圆的左端点或右端点时，会返回东西（左右）箭头

+   当用户抓住椭圆的顶部或底部端点时，会返回上下（上下）箭头

+   当用户移动椭圆时，会返回大小箭头（指向左、右、上、下的四个箭头）

+   最后，当用户既不移动也不修改椭圆时，会返回正常的箭头光标

```cpp
CursorType EllipseFigure::GetCursor() const { 
  switch (ellipseMode) { 
    case CreateEllipse: 
      return Cursor::Crosshair; 

    case LeftPoint: 
    case RightPoint: 
      return Cursor::SizeWestEast; 

    case TopPoint: 
    case BottomPoint: 
      return Cursor::SizeNorthSouth; 

    case MoveEllipse: 
      return Cursor::SizeAll; 

    default: 
      return Cursor::Normal; 
  } 
} 

```

# 概述

在本章中，你学习了第四章中绘图程序的图形类层次结构，*处理形状和图形*。你涵盖了以下主题：

+   测试图形是否被鼠标点击击中或是否被矩形包围

+   图形的修改和移动

+   绘制图形和计算图形的面积

+   将图形写入和读取到文件流或剪贴板

+   根据图形的当前状态使用不同光标的游标处理

在第六章中，*构建字处理器*，你将开始开发字处理器。
