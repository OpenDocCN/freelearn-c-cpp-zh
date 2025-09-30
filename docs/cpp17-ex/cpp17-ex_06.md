# 第六章：提升 Qt 图形应用程序

在第五章，*Qt 图形应用程序*中，我们开发了包含模拟时钟、绘图程序和编辑器的图形 Qt 应用程序。在本章中，我们将继续对第五章中提到的三个图形应用程序进行工作，即[第五章](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67)，*Qt 图形应用程序*。然而，我们将进行以下改进：

+   **时钟**: 我们将向时钟表盘添加数字

+   **绘图程序**: 我们将添加移动和修改图形、剪切和粘贴它们以及标记一个或多个图形的能力

+   **编辑器**: 我们将添加更改字体和对齐方式以及标记文本块的能力

在本章中，我们将继续使用 Qt 库：

+   窗口和小部件

+   菜单和工具栏

+   鼠标和键盘事件

# 改进时钟

在本章中，我们将替换时钟表盘标记的版本，使用数字。

# `Clock`类

`Clock`类的定义与[第五章](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67)，*Qt 图形应用程序*中的定义类似。计时器每秒更新窗口 10 次。构造函数初始化时钟，每当窗口需要重绘时，都会调用`paintEvent`。

**Clock.h**: 

```cpp
   #ifndef CLOCK_H 
   #define CLOCK_H 

   #include <QWidget> 
   #include <QTimer> 

   class Clock : public QWidget { 
     Q_OBJECT 

     public: 
       Clock(QWidget *parentWidget = nullptr); 
       void paintEvent(QPaintEvent *eventPtr);

     private: 
     QTimer m_timer; 
   }; 

   #endif // CLOCK_H 
```

**Clock.cpp**: 

```cpp
   #include <QtWidgets> 
   #include "Clock.h" 
```

与[第五章](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67)，*Qt 图形应用程序*类似，构造函数将窗口标题设置为`Clock Advanced`，窗口大小设置为*1000* x *500*像素，初始化计时器以每`100`毫秒发送一个超时消息，并将`timeout`消息连接到`update`方法，这将强制窗口在每次超时时重绘：

```cpp
    Clock::Clock(QWidget *parentWidget /*= nullptr*/) 
    :QWidget(parentWidget) { 
      setWindowTitle(tr("Clock Advanced")); 
      resize(1000, 500); 

      m_timer.setParent(this); 
      connect(&m_timer, SIGNAL(timeout()), this, SLOT(update())); 
      m_timer.start(100); 
    } 
```

每当窗口需要重绘时，都会调用`paintEvent`方法。我们首先计算时钟的边长并获取当前时间：

```cpp
    void Clock::paintEvent(QPaintEvent* /*event*/) { 
      int side = qMin(width(), height()); 
      QTime time = QTime::currentTime(); 
```

我们随后创建并初始化一个`QPainter`对象。我们调用`translate`和`scale`来匹配物理大小（像素）与逻辑大小（*200 x 200*单位）：

```cpp
    QPainter painter(this); 
    painter.setRenderHint(QPainter::Antialiasing); 
    painter.setRenderHint(QPainter::TextAntialiasing); 
    painter.translate(width() / 2, height() / 2); 
    painter.scale(side / 200.0, side / 200.0); 
```

在本章的这个版本中，我们向画家添加了`Times New Roman`字体，`12`点，以写入时钟的数字：

```cpp
    painter.setFont(QFont(tr("Times New Roman"), 12)); 
```

我们将时钟的数字`1`到`12`写入，如下所示：

```cpp
     for (int hour = 1; hour <= 12; ++hour) { 
       QString text; 
       text.setNum(hour); 
```

一个完整的跳跃是 360°，两个连续数字之间的角度是 30°，因为 360 除以 12 等于 30：

```cpp
    double angle = (30.0 * hour) - 90; 
    double radius = 90.0; 
```

数字`x`和`y`坐标通过正弦和余弦函数计算得出。然而，首先我们需要将度数转换为弧度，因为正弦和余弦函数只接受弧度。以下代码展示了这一过程：

```cpp
    double x = radius * qCos(qDegreesToRadians(angle)), 
           y = radius * qSin(qDegreesToRadians(angle)); 
```

`drawText`方法将数字写入，如下所示：

```cpp
     QRect rect(x - 100, y - 100, 200, 200); 
     painter.drawText(rect, Qt::AlignHCenter | 
                            Qt::AlignVCenter, text); 
     } 
```

当数字被写入后，我们将以与[第五章](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67)，*Qt 图形应用程序*中相同的方式绘制`hour`、`minute`和`second`指针：

```cpp
    double hours = time.hour(), minutes = time.minute(), 
      seconds = time.second(), milliseconds = time.msec(); 

    painter.setPen(Qt::black); 
    painter.setBrush(Qt::gray); 

    { static const QPoint hourHand[3] = 
       {QPoint(8, 8), QPoint(-8, 8), QPoint(0, -60)}; 

      painter.save(); 
      double hour = hours + (minutes / 60.0) + (seconds / 3600.0) + 
                  (milliseconds / 3600000.0); 
      painter.rotate(30.0 * hour); 
      painter.drawConvexPolygon(hourHand, 3); 
      painter.restore(); 
    } 

    { static const QPoint minuteHand[3] = 
      {QPoint(6, 8), QPoint(-6, 8), QPoint(0, -70)}; 

      painter.save(); 
      double minute = minutes + (seconds / 60.0) + 
                    (milliseconds / 60000.0); 
      painter.rotate(6.0 * minute); 
      painter.drawConvexPolygon(minuteHand, 3); 
      painter.restore(); 
    } 

    { static const QPoint secondHand[3] = 
       {QPoint(4, 8), QPoint(-4, 8), QPoint(0, -80)}; 

      painter.save(); 
      double second = seconds + (milliseconds / 1000); 
      painter.rotate(6.0 * second); 
      painter.drawConvexPolygon(secondHand, 3); 
      painter.restore(); 
    } 
 }
```

# 主要功能

`main` 函数与 [第五章](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67) 中的类似，*Qt 图形应用程序*。它创建一个应用程序对象，初始化时钟，并执行应用程序。

**Main.cpp:**

```cpp
#include <QApplication> 
#include "Clock.h" 

int main(int argc, char *argv[]) { 
  QApplication application(argc, argv); 
  Clock Clock; 
  Clock.show(); 
  return application.exec(); 
} 
```

**输出**:

![](img/4fc9d538-a689-43c2-9e55-d697cc18342c.png)

# 提高绘图程序

本章的绘图程序是 [第五章](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67) 中 *Qt 图形应用程序* 绘图程序的更高级版本。在这个版本中，可以修改图形，包围一个或多个图形然后改变它们的颜色，以及剪切和粘贴图形。

# 图形类

`Figure` 类与 [第五章](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67) 中的类似，*Qt 图形应用程序*。然而，增加了 `isInside`、`doubleClick`、`modify` 和 `marked`。

**Figure.h:**

```cpp
    #ifndef FIGURE_H 
    #define FIGURE_H 

    enum FigureId {LineId, RectangleId, EllipseId}; 

    #include <QtWidgets> 
    #include <FStream> 
    using namespace std; 

    class Figure { 
      public: 
      Figure(); 
```

在这个版本中，增加了纯虚 `clone` 方法。这是由于剪切和粘贴。当粘贴图形时，我们希望创建它的一个副本，而不必实际知道该对象属于哪个类。我们只能通过复制构造函数来做这件事。这实际上是本节的主要点：如何使用纯虚方法和如何利用动态绑定。我们需要 `clone`，它调用其类的复制构造函数以返回新对象的指针：

```cpp
    virtual Figure* clone() const = 0; 

    virtual FigureId getId() const = 0; 
    virtual void initializePoints(QPoint point) = 0; 
```

在这个版本的绘图程序中，`onClick` 设置字段以指示图形是否应该被修改或移动。如果用户抓住图形的标记点之一（不同类型的图形之间有所不同），则修改图形。否则，应移动图形。当用户抓住图形的一个角时调用 `modify` 方法。在这种情况下，应修改图形而不是移动它：

```cpp
    virtual bool isClick(QPoint mousePoint) = 0; 
    virtual void modify(QSize distance) = 0; 
```

`isInside` 方法返回 `true` 如果图形完全包含在区域内。当用户用鼠标包围图形时调用：

```cpp
    virtual bool isInside(QRect area) = 0; 
```

当用户在图形上双击时调用 `doubleClick` 方法，每个图形执行一些合适的操作：

```cpp
    virtual void doubleClick(QPoint mousePoint) = 0; 

    virtual void move(QSize distance) = 0; 
    virtual void draw(QPainter &painter) const = 0; 

    virtual bool write(ofstream& outStream) const; 
    virtual bool read(ifstream& inStream); 
```

`marked` 方法返回和设置 `m_marked` 字段。当一个图形被标记时，它会被小方块注释：

```cpp
    bool marked() const {return m_marked;} 
    bool& marked() {return m_marked;} 

    const QColor& color() const {return m_color;} 
    QColor& color() {return m_color;} 

    virtual bool filled() const {return m_filled;} 
    virtual bool& filled() {return m_filled;} 

    static const int Tolerance; 

    private: 
    QColor m_color; 
    bool m_marked = false, m_filled = false; 
    }; 

    #endif 
```

**Figure.cpp:**

```cpp
    #include "..\MainWindow\DocumentWidget.h" 
    #include "Figure.h" 

    const int Figure::Tolerance(6); 

    Figure::Figure() { 
       // Empty. 
    } 
```

`write` 和 `read` 方法写入和读取图形的颜色以及它是否被填充。然而，它们不写入或读取标记状态。图形在写入或读取时总是未标记的：

```cpp
    bool Figure::write(ofstream& outStream) const { 
      writeColor(outStream, m_color); 
      outStream.write((char*) &m_filled, sizeof m_filled); 
      return ((bool) outStream); 
    } 

    bool Figure::read(ifstream& inStream) { 
      readColor(inStream, m_color); 
      inStream.read((char*) &m_filled, sizeof m_filled); 
      return ((bool) inStream); 
    } 
```

# 线类

`Line` 类是 `Figure` 的子类。

**Line.h:**

```cpp
    #ifndef LINE_H 
    #define LINE_H 

    #include <FStream> 
    using namespace std; 

    #include "Figure.h" 

    class Line : public Figure { 
      public: 
      Line(); 
      FigureId getId() const {return LineId;} 
      In addition to the  
      Line(const Line& line); 
      Figure* clone() const; 

      void initializePoints(QPoint point); 
```

如前文 `Figure` 部分所述，`isClick` 决定线是否应该被修改或移动。如果用户抓住其端点之一，则仅移动该端点。如果用户抓住端点之间的线，则移动整条线。也就是说，线的两个端点都将移动：

```cpp
    bool isClick(QPoint mousePoint); 
```

`isInside` 方法检查线是否完全被区域包围：

```cpp
    bool isInside(QRect area); 
```

在 `Line` 类中，`doubleClick` 方法不执行任何操作。然而，我们仍然需要定义它，因为它是 `Figure` 中的纯虚函数。如果我们没有定义它，`Line` 将会是抽象的：

```cpp
    void doubleClick(QPoint /* mousePoint */) {/* Empty. */} 
```

`modify` 方法根据前一个 `isClick` 的设置修改线。如果用户抓取了一个端点，则该端点被移动。否则，整个线（包括两个端点）被移动：

```cpp
    void modify(QSize distance); 
    void move(QSize distance); 
```

如果线被标记，`area` 方法返回一个稍微大一点的区域，以便包括标记的正方形：

```cpp
    QRect area() const; 
    void draw(QPainter& painter) const; 

    bool write(ofstream& outStream) const; 
    bool read(ifstream& inStream); 
```

`m_lineMode` 字段跟踪线的移动或修改。当线被创建时，`m_lineMode` 被设置为 `LastPoint`。当用户抓取线的第一个或最后一个端点时，`m_lineMode` 被设置为 `FirstPoint` 或 `LastPoint`。当用户抓取端点之间的线时，`m_lineMode` 被设置为 `MoveLine`：

```cpp
      private: 
         enum {FirstPoint, LastPoint, MoveLine} m_lineMode; 
         QPoint m_firstPoint, m_lastPoint; 
```

`isPointInLine` 方法决定用户是否点击了线，并有一定的容差：

```cpp
    static bool isPointInLine(QPoint m_firstPoint, 
                              QPoint m_lastPoint, QPoint point); 
    }; 

    #endif 
```

**Line.cpp:**

```cpp
    #include "..\MainWindow\DocumentWidget.h" 
    #include "Line.h" 
```

当一条线被创建时，线模式被设置为最后一个点。这意味着当用户移动鼠标时，线的最后一个点将会改变：

```cpp
    Line::Line() 
    :m_lineMode(LastPoint) { 
      // Empty. 
    } 
```

当粘贴线时，会调用 `clone` 方法。调用 `Figure` 的拷贝构造函数来设置图形的颜色。请注意，我们使用 `Line` 对象作为参数调用 `Figure` 构造函数，尽管它接受一个 `Figure` 对象的引用作为参数。我们允许这样做，因为 `Line` 是 `Figure` 的子类，并且在调用过程中 `Line` 对象将被转换成 `Figure` 对象。此外，第一个和最后一个端点被复制。请注意，我们确实需要复制 `m_lineMode` 的值，因为它的值是在用户创建、修改或移动线时设置的：

```cpp
    Line::Line(const Line& line) 
     :Figure(line), 
       m_firstPoint(line.m_firstPoint), 
       m_lastPoint(line.m_lastPoint) { 
      // Empty. 
     } 
```

`clone` 方法使用拷贝构造函数来创建一个新的对象，然后返回：

```cpp
    Figure* Line::clone() const { 
      Line* linePtr = new Line(*this); 
      return linePtr; 
    } 
```

在线被创建后不久，会调用 `initializePoints` 方法。调用这个方法的原因是我们没有直接创建 `Line` 对象。相反，我们通过调用 `clone` 间接创建线。然后我们需要通过调用 `initializePoints` 来初始化端点：

```cpp
    void Line::initializePoints(QPoint point) { 
      m_firstPoint = point; 
      m_lastPoint = point; 
    } 
```

当用户用鼠标点击时，会调用 `isClick` 方法。首先，我们检查他们是否点击了第一个端点。我们使用 `Tolerance` 字段创建一个以第一个端点为中心的小正方形。如果用户点击了这个正方形，`m_lineMode` 被设置为 `FirstPoint` 并返回 `true`：

```cpp
    bool Line::isClick(QPoint mousePoint) { 
      QRect firstSquare(makeRect(m_firstPoint, Tolerance)); 

     if (firstSquare.contains(mousePoint)) { 
       m_lineMode = FirstPoint; 
       return true; 
     } 
```

同样地，我们在最后一个端点的中心创建一个小正方形。如果用户点击这个正方形，`m_lineMode` 被设置为 `LastPoint` 并返回 `true`：

```cpp
    QRect lastSquare(makeRect(m_lastPoint, Tolerance)); 

    if (lastSquare.contains(mousePoint)) { 
      m_lineMode = LastPoint; 
      return true; 
    } 
```

如果用户没有点击任一端点，我们将检查他们是否点击了线本身。如果他们点击了，`m_lineMode` 被设置为 `ModeLine` 并返回 `true`：

```cpp
    if (isPointInLine(m_firstPoint, m_lastPoint, mousePoint)) { 
      m_lineMode = MoveLine; 
      return true; 
    } 
```

最后，如果用户没有点击线的一个端点或线本身，他们完全错过了线，并返回 `false`：

```cpp
    return false; 
    } 
```

`isInside` 方法如果线完全被区域包围则返回 `true`。这相当简单，我们只需检查两个端点是否位于区域内：

```cpp
    bool Line::isInside(QRect area) { 
     return area.contains(m_firstPoint) && 
       area.contains(m_lastPoint); 
    } 
```

`isPointInLine` 方法与 第五章 的版本，*Qt 图形应用程序* 中的 `isClick` 方法相同：

```cpp
    bool Line::isPointInLine(QPoint m_firstPoint, QPoint m_lastPoint,
                         QPoint point) {
  if (m_firstPoint.x() == m_lastPoint.x()) {
    QRect lineRect(m_firstPoint, m_lastPoint);
    lineRect.normalized();
    lineRect += Tolerance;
    return lineRect.contains(point);
  }
  else {
    QPoint leftPoint = (m_firstPoint.x() < m_lastPoint.x())
                       ? m_firstPoint : m_lastPoint,
           rightPoint = (m_firstPoint.x() < m_lastPoint.x())
                       ? m_lastPoint : m_firstPoint;

    if ((leftPoint.x() <= point.x()) &&
        (point.x() <= rightPoint.x())) {
      int lineWidth = rightPoint.x() - leftPoint.x(),
          lineHeight = rightPoint.y() - leftPoint.y();

      int diffWidth = point.x() - leftPoint.x(),
          diffHeight = point.y() - leftPoint.y();

      double delta = fabs(diffHeight -
               (diffWidth * ((double) lineHeight) / lineWidth));
      return (delta <= Tolerance);
    }

    return false;
  }
}
```

`modify` 方法根据前面 `isClick` 方法中 `m_lineMode` 的设置移动第一个或最后一个端点，或两者都移动：

```cpp
    void Line::modify(QSize distance) { 
      switch (m_lineMode) { 
        case FirstPoint: 
        m_firstPoint += distance; 
        break; 

        case LastPoint: 
        m_lastPoint += distance; 
        break; 

        case MoveLine: 
        move(distance); 
        break; 
      } 
    } 
```

`move` 方法简单地移动线的两个端点：

```cpp
    void Line::move(QSize distance) { 
      m_firstPoint += distance; 
      m_lastPoint += distance; 
    } 
```

`draw` 方法绘制线。与 第五章 的版本，*Qt 图形应用程序* 相比，它还绘制了如果被标记的线端点的正方形：

```cpp
    void Line::draw(QPainter& painter) const { 
      painter.setPen(color()); 
      painter.drawLine(m_firstPoint, m_lastPoint); 

      if (marked()) { 
        painter.fillRect(makeRect(m_firstPoint, Tolerance), 
                     Qt::black); 
        painter.fillRect(makeRect(m_lastPoint, Tolerance), 
                     Qt::black); 
      } 
    } 
```

`area` 方法返回覆盖线的区域。如果线被标记，区域会略微扩展以覆盖标记端点的正方形：

```cpp
    QRect Line::area() const { 
      QRect lineArea(m_firstPoint, m_lastPoint); 
      lineArea.normalized(); 

      if (marked()) { 
         lineArea += Tolerance; 
      } 

      return lineArea; 
    } 
```

与 [第五章](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67) 的版本类似，*Qt 图形应用程序*，`write` 和 `read` 调用 `Figure` 中的对应方法，然后写入和读取线的两个端点：

```cpp
    bool Line::write(ofstream& outStream) const { 
      Figure::write(outStream); 
      writePoint(outStream, m_firstPoint); 
      writePoint(outStream, m_lastPoint); 
      return ((bool) outStream); 
    } 

    bool Line::read(ifstream& inStream) { 
      Figure::read(inStream); 
      readPoint(inStream, m_firstPoint); 
      readPoint(inStream, m_lastPoint); 
      return ((bool) inStream); 
    } 
```

# 矩形类

`RectangleX` 是 `Figure` 的子类。它是 [第五章](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67) 的扩展版本，*Qt 图形应用程序*。`isClick` 方法已被修改，增加了 `doubleClick` 和 `modify`：

**Rectangle.h**: 

```cpp
    #ifndef RECTANGLE_H 
    #define RECTANGLE_H 

    #include <FStream> 
    using namespace std; 

    #include "Figure.h" 

    class RectangleX : public Figure { 
      public: 
      RectangleX(); 
      virtual FigureId getId() const {return RectangleId;} 

      RectangleX(const RectangleX& rectangle); 
      Figure* clone() const; 

      virtual void initializePoints(QPoint point); 

      virtual bool isClick(QPoint mousePoint); 
      virtual void modify(QSize distance); 

      virtual bool isInside(QRect area); 
      virtual void doubleClick(QPoint mousePoint); 

      virtual void move(QSize distance); 
      virtual QRect area() const; 
      virtual void draw(QPainter& painter) const; 

      virtual bool write(ofstream& outStream) const; 
      virtual bool read(ifstream& inStream); 

      private: 
        enum {TopLeftPoint, TopRightPoint, BottomRightPoint,  
            BottomLeftPoint, MoveRectangle} m_rectangleMode; 

      protected: 
        QPoint m_topLeft, m_bottomRight; 
    }; 

    #endif 
```

**Rectangle.cpp**: 

```cpp
    #include <CAssert> 
    #include "..\MainWindow\DocumentWidget.h" 
    #include "Rectangle.h" 
```

当用户添加矩形时，其模式是 `BottomRightPoint`。这意味着当用户移动鼠标时，矩形的右下角将被移动：

```cpp
    RectangleX::RectangleX() 
    :m_rectangleMode(BottomRightPoint) { 
      // Empty. 
    } 
```

复制构造函数复制矩形。更具体地说，首先它调用 `Figure` 类的复制构造函数，然后复制左上角和右下角。请注意，它不会复制 `m_rectangleMode` 字段，因为它仅在用户移动鼠标时使用：

```cpp
    RectangleX::RectangleX(const RectangleX& rectangle) 
    :Figure(rectangle), 
      m_topLeft(rectangle.m_topLeft), 
      m_bottomRight(rectangle.m_bottomRight) { 
      // Empty. 
    } 
```

`clone` 方法通过调用复制构造函数创建并返回一个指向新对象的指针：

```cpp
    Figure* RectangleX::clone() const { 
      RectangleX* rectanglePtr = new RectangleX(*this); 
      return rectanglePtr; 
    } 

    void RectangleX::initializePoints(QPoint point) { 
      m_topLeft = point; 
      m_bottomRight = point; 
    } 
```

当用户用鼠标点击时调用 `isClick` 方法。与前面的布尔 `Line` 类似，我们首先检查他们是否点击了任何角落。如果没有，我们检查他们是否点击了矩形边框或矩形内部，这取决于它是否被填充：

我们首先定义一个覆盖左上角的小正方形。如果用户点击它，我们将 `m_rectangleMode` 字段设置为 `TopLeftPoint` 并返回 `true`：

```cpp
    bool RectangleX::isClick(QPoint mousePoint) { 
      QRect topLeftRect(makeRect(m_topLeft, Tolerance)); 

      if (topLeftRect.contains(mousePoint)) { 
        m_rectangleMode = TopLeftPoint; 
        return true; 
      } 
```

我们继续定义一个覆盖右上角的正方形。如果用户点击它，我们将 `m_rectangleMode` 设置为 `TopRightPoint` 并返回 `true`：

```cpp
     QPoint topRightPoint(m_bottomRight.x(), m_topLeft.y()); 
     QRect topRectRight(makeRect(topRightPoint, Tolerance)); 

     if (topRectRight.contains(mousePoint)) { 
       m_rectangleMode = TopRightPoint; 
       return true; 
     } 
```

如果用户点击在覆盖右下角的正方形上，我们将 `m_rectangleMode` 设置为 `BottomRightPoint` 并返回 `true`：

```cpp
     QRect m_bottomRightRect(makeRect(m_bottomRight, Tolerance)); 

     if (m_bottomRightRect.contains(mousePoint)) { 
       m_rectangleMode = BottomRightPoint; 
       return true; 
     } 
```

如果用户点击在覆盖左下角的正方形上，我们将 `m_rectangleMode` 设置为 `BottomLeftPoint` 并返回 `true`：

```cpp
    QPoint bottomLeftPoint(m_topLeft.x(), m_bottomRight.y()); 
    QRect bottomLeftRect(makeRect(bottomLeftPoint, Tolerance)); 

    if (bottomLeftRect.contains(mousePoint)) { 
      m_rectangleMode = BottomLeftPoint; 
      return true; 
    } 
```

如果用户没有点击在矩形的任何一个角落，我们检查矩形本身。如果它是填充的，我们检查鼠标指针是否位于矩形本身内部。如果是，我们将 `m_rectangleMode` 设置为 `MoveRectangle` 并返回 `true`：

```cpp
  QRect areaRect(m_topLeft, m_bottomRight); 

  if (filled()) { 
    if (areaRect.contains(mousePoint)) { 
      m_rectangleMode = MoveRectangle; 
      return true; 
    } 
  } 
```

如果矩形没有被填充，我们定义稍微大一些和稍微小一些的矩形。如果鼠标点击位于较大的矩形内部，但不在较小的矩形内，我们将 `m_rectangleMode` 设置为 `MoveRectangle` 并返回 `true`：

```cpp
     else { 
       QRect largeAreaRect(areaRect), smallAreaRect(areaRect); 

      largeAreaRect += Tolerance; 
      smallAreaRect -= Tolerance; 

      if (largeAreaRect.contains(mousePoint) && 
         !smallAreaRect.contains(mousePoint)) { 
           m_rectangleMode = MoveRectangle; 
           return true; 
        } 
     } 
```

最后，如果用户没有点击在任何一个角落或矩形本身，他们错过了矩形，我们返回 `false`：

```cpp
      return false; 
    } 
```

`isInside` 方法相当简单。我们只需检查左上角和右下角是否位于矩形内部：

```cpp
     bool RectangleX::isInside(QRect area) { 
       return area.contains(m_topLeft) && 
         area.contains(m_bottomRight); 
     } 
```

当用户用鼠标双击时，会调用 `doubleClick` 方法。如果 `onClick` 的调用返回 `true`，则调用 `doubleClick`。在矩形的情况下，填充状态会改变——填充的矩形变为未填充，未填充的矩形变为填充：

```cpp
     void RectangleX::doubleClick(QPoint mousePoint) { 
       if (isClick(mousePoint)) { 
```

第一次调用 `filled` 是调用返回对 `m_filled` 字段引用的版本，这允许我们更改返回的值：

```cpp
     filled() = !filled(); 
    } 
  } 
```

`modify` 方法根据前一个 `isClick` 中设置的 `m_rectangleMode` 字段来修改矩形。如果它设置为四个角落之一，我们修改那个角落。如果不是，我们移动整个矩形：

```cpp
   void RectangleX::modify(QSize distance) { 
     switch (m_rectangleMode) { 
       case TopLeftPoint: 
       m_topLeft += distance; 
       break; 

       case TopRightPoint: 
       m_topLeft.setY(m_topLeft.y() + distance.height()); 
       m_bottomRight.setX(m_bottomRight.x() + distance.width()); 
       break; 

       case BottomRightPoint: 
       m_bottomRight += distance; 
       break; 

       case BottomLeftPoint: 
       m_topLeft.setX(m_topLeft.x() + distance.width()); 
       m_bottomRight.setY(m_bottomRight.y() + distance.height()); 
       break; 

       case MoveRectangle: 
       move(distance); 
       break; 
      } 
   } 
```

`move` 方法相当简单。它只是改变左上角和右下角：

```cpp
    void RectangleX::move(QSize distance) { 
      m_topLeft += distance; 
      m_bottomRight += distance; 
    } 
```

`area` 方法返回覆盖矩形的面积。如果它被标记，我们稍微扩大面积，以便它覆盖标记的正方形：

```cpp
    QRect RectangleX::area() const { 
      QRect areaRect(m_topLeft, m_bottomRight); 
      areaRect.normalized(); 

      if (marked()) { 
        areaRect += Tolerance; 
      } 

      return areaRect; 
    } 
```

`draw` 方法绘制矩形；使用全画笔时填充，使用空心画笔时未填充：

```cpp
    void RectangleX::draw(QPainter& painter) const { 
      painter.setPen(color()); 

      if (filled()) { 
        painter.fillRect(QRect(m_topLeft, m_bottomRight), color()); 
      } 
      else { 
        painter.setBrush(Qt::NoBrush); 
        painter.drawRect(QRect(m_topLeft, m_bottomRight)); 
      } 
```

如果矩形被标记，覆盖矩形四个角落的四个正方形也会被绘制：

```cpp
    if (marked()) { 
      painter.fillRect(makeRect(m_topLeft, Tolerance), Qt::black); 
      QPoint topRight(m_bottomRight.x(), m_topLeft.y()); 
      painter.fillRect(makeRect(topRight, Tolerance), Qt::black); 
      painter.fillRect(makeRect(m_bottomRight, Tolerance), 
                     Qt::black); 
      QPoint bottomLeft(m_topLeft.x(), m_bottomRight.y()); 
      painter.fillRect(makeRect(bottomLeft, Tolerance), Qt::black); 
    } 
  } 
```

`write` 和 `read` 方法首先调用 `Figure` 中的对应方法来写入和读取矩形的颜色。然后写入和读取左上角和右下角：

```cpp
    bool RectangleX::write(ofstream& outStream) const { 
     Figure::write(outStream); 
     writePoint(outStream, m_topLeft); 
     writePoint(outStream, m_bottomRight); 
     return ((bool) outStream); 
    } 

    bool RectangleX::read (ifstream& inStream) { 
      Figure::read(inStream); 
      readPoint(inStream, m_topLeft); 
      readPoint(inStream, m_bottomRight); 
      return ((bool) inStream); 
 } 
```

# Ellipse 类

`EllipseX` 是 `RectangleX` 的直接子类，也是 `Figure` 的间接子类，它绘制一个填充或未填充的椭圆：

**EllipseX.h:** 

```cpp
    #ifndef ELLIPSE_H 
    #define ELLIPSE_H 

    #include "Rectangle.h" 

    class EllipseX : public RectangleX { 
      public: 
      EllipseX(); 
      FigureId getId() const {return EllipseId;} 

      EllipseX(const EllipseX& ellipse); 
      Figure* clone() const; 
```

与前面的矩形情况类似，`isClick` 检查用户是否在椭圆的四个角落之一抓取椭圆，或者椭圆本身是否应该移动：

```cpp
    bool isClick(QPoint mousePoint); 
```

`modify` 方法根据前一个 `isClick` 中设置的 `m_ellipseMode` 的设置来修改椭圆：

```cpp
    void modify(QSize distance); 
    void draw(QPainter& painter) const; 
```

与前面的矩形可以通过其四个角抓取不同，椭圆可以通过其左、上、右和底部点抓取。因此，我们需要添加 `CreateEllipse` 枚举值，它修改覆盖椭圆的区域右下角：

```cpp
     private: 
       enum {CreateEllipse, LeftPoint, TopPoint, RightPoint, 
           BottomPoint, MoveEllipse} m_ellipseMode; 
       }; 

       #endif 
```

**EllipseX.cpp:**

```cpp
    #include <CAssert> 
    #include "..\MainWindow\DocumentWidget.h" 
    #include "Ellipse.h" 
```

与前面的行和矩形情况相比，我们将 `m_ellipseMode` 字段设置为 `CreateEllipse`，这在创建椭圆时是有效的：

```cpp
     EllipseX::EllipseX() 
      :m_ellipseMode(CreateEllipse) { 
      // Empty. 
     } 
```

复制构造函数不需要设置 `m_topLeft` 和 `m_bottomRight` 字段，因为这是由 `RectangleX` 的复制构造函数处理的，而 `RectangleX` 的复制构造函数是由 `EllipseX` 的复制构造函数调用的：

```cpp
    EllipseX::EllipseX(const EllipseX& ellipse) 
    :RectangleX(ellipse) { 
      // Empty. 
    } 

    Figure* EllipseX::clone() const { 
      EllipseX* ellipsePtr = new EllipseX(*this); 
      return ellipsePtr; 
    } 
```

与前面的矩形情况类似，`isClick` 检查用户是否通过椭圆的四个点之一抓取椭圆。然而，在椭圆的情况下，我们不检查矩形的角。相反，我们检查椭圆的左、上、右和底部位置。我们为这些位置中的每一个创建一个小正方形，并检查用户是否点击了它们。如果点击了，我们将 `m_ellipseMode` 字段设置为适当的值并返回 `true`：

```cpp
    bool EllipseX::isClick(QPoint mousePoint) { 
      QPoint leftPoint(m_topLeft.x(), 
                   (m_topLeft.y() + m_bottomRight.y()) / 2); 
      QRect leftRect(makeRect(leftPoint, Tolerance)); 

      if (leftRect.contains(mousePoint)) { 
        m_ellipseMode = LeftPoint; 
        return true; 
      } 

      QPoint topPoint((m_topLeft.x() + m_bottomRight.x()) / 2, 
                  m_topLeft.y()); 
      QRect topRect(makeRect(topPoint, Tolerance)); 

      if (topRect.contains(mousePoint)) { 
        m_ellipseMode = TopPoint; 
        return true; 
      } 

      QPoint rightPoint(m_bottomRight.x(), 
                    (m_topLeft.y() + m_bottomRight.y()) / 2); 
      QRect rightRect(makeRect(rightPoint, Tolerance)); 

      if (rightRect.contains(mousePoint)) { 
        m_ellipseMode = RightPoint; 
        return true; 
      } 

      QPoint bottomPoint((m_topLeft.x() + m_bottomRight.x()) / 2, 
                     m_bottomRight.y()); 
      QRect bottomRect(makeRect(bottomPoint, Tolerance)); 

      if (bottomRect.contains(mousePoint)) { 
        m_ellipseMode = BottomPoint; 
        return true; 
      } 
```

如果用户没有点击四个位置中的任何一个，我们检查他们是否点击了椭圆本身。如果它是填充的，我们使用 Qt 的 `QRegion` 类创建一个椭圆区域，并检查鼠标点是否位于该区域内：

```cpp
    QRect normalRect(m_topLeft, m_bottomRight); 
    normalRect.normalized(); 

    if (filled()) { 
      QRegion normalEllipse(normalRect, QRegion::Ellipse); 

      if (normalEllipse.contains(mousePoint)) { 
        m_ellipseMode = MoveEllipse; 
        return true; 
      } 
    } 
```

如果椭圆未填充，我们创建稍大和稍小的椭圆区域，然后检查鼠标点是否位于较大的区域内，同时也位于较小的区域内：

```cpp
     else { 
       QRect largeRect(normalRect), smallRect(normalRect); 
       largeRect += Tolerance; 
       smallRect -= Tolerance; 

       QRegion largeEllipse(largeRect, QRegion::Ellipse), 
            smallEllipse(smallRect, QRegion::Ellipse); 

       if (largeEllipse.contains(mousePoint) && 
           !smallEllipse.contains(mousePoint)) { 
         m_ellipseMode = MoveEllipse; 
         return true; 
       } 
     } 
```

最后，如果用户没有在任何抓取位置或椭圆本身上点击，我们返回 `false`：

```cpp
      return false; 
   } 
```

`modify` 方法根据 `onClick` 中 `m_ellipseMode` 的设置修改椭圆：

```cpp
    void EllipseX::modify(QSize distance) { 
      switch (m_ellipseMode) { 
        case CreateEllipse: 
        m_bottomRight += distance; 
        break; 

        case LeftPoint: 
        m_topLeft.setX(m_topLeft.x() + distance.width()); 
        break; 

        case RightPoint: 
        m_bottomRight.setX(m_bottomRight.x() + distance.width()); 
        break; 

        case TopPoint: 
        m_topLeft.setY(m_topLeft.y() + distance.height()); 
        break; 

        case BottomPoint: 
        m_bottomRight.setY(m_bottomRight.y() + distance.height()); 
        break; 

        case MoveEllipse: 
        move(distance); 
        break; 
      } 
    } 
```

`draw` 方法根据椭圆是否填充，使用实心画笔绘制椭圆，如果未填充则使用空心画笔：

```cpp
    void EllipseX::draw(QPainter& painter) const { 
      painter.setPen(color()); 

      if (filled()) { 
        painter.setBrush(color()); 
        painter.drawEllipse(QRect(m_topLeft, m_bottomRight)); 
      } 
      else { 
        painter.setBrush(Qt::NoBrush); 
        painter.drawEllipse(QRect(m_topLeft, m_bottomRight)); 
      } 
```

如果椭圆被标记，则绘制覆盖椭圆顶部、左侧、右侧和底部点的四个正方形：

```cpp
   if (marked()) {
    QPoint leftPoint(m_topLeft.x(),
                     (m_topLeft.y() + m_bottomRight.y())/2);
    painter.fillRect(makeRect(leftPoint, Tolerance), Qt::black);

    QPoint topPoint((m_topLeft.x() + m_bottomRight.x()) / 2,
                    m_topLeft.y());
    painter.fillRect(makeRect(topPoint, Tolerance), Qt::black);

    QPoint rightPoint(m_bottomRight.x(),
                      (m_topLeft.y() + m_bottomRight.y()) / 2);
    painter.fillRect(makeRect(rightPoint, Tolerance), Qt::black);

    QPoint bottomPoint((m_topLeft.x() + m_bottomRight.x()) / 2,
                       m_bottomRight.y());
    painter.fillRect(makeRect(bottomPoint, Tolerance), Qt::black);
  }
}
```

# `DrawingWindow` 类

`DrawingWindow` 类与上一章的版本类似。它重写了 `closeEvent` 方法。

**DrawingWindow.h:**

```cpp
    #ifndef DRAWINGWINDOW_H 
    #define DRAWINGWINDOW_H 

    #include <QMainWindow> 
    #include <QActionGroup> 

    #include "..\MainWindow\MainWindow.h" 
    #include "DrawingWidget.h" 

    class DrawingWindow : public MainWindow { 
      Q_OBJECT 

      public: 
      DrawingWindow(QWidget *parentWidget = nullptr); 
      ~DrawingWindow(); 

      public: 
      void closeEvent(QCloseEvent *eventPtr) 
           { m_drawingWidgetPtr->closeEvent(eventPtr); } 

      private: 
       DrawingWidget* m_drawingWidgetPtr; 
       QActionGroup* m_figureGroupPtr; 
    }; 

    #endif // DRAWINGWINDOW_H 
```

**DrawingWindow.cpp:**

```cpp
    #include "..\MainWindow\DocumentWidget.h" 
    #include "DrawingWindow.h" 
```

构造函数将窗口大小初始化为 *1000* x *500* 像素，将绘图小部件放置在窗口中间，添加标准文件和编辑菜单，并添加应用程序特定的格式和图形菜单：

```cpp
DrawingWindow::DrawingWindow(QWidget *parentWidget /*= nullptr*/)
 :MainWindow(parentWidget) {
  resize(1000, 500);

  m_drawingWidgetPtr = new DrawingWidget(this);
  setCentralWidget(m_drawingWidgetPtr);
  addFileMenu();
  addEditMenu();
```

格式菜单包含 `Color`、`Fill` 和 `Modify` 项以及图形子菜单：

```cpp
    { QMenu* formatMenuPtr = menuBar()->addMenu(tr("F&ormat")); 
       connect(formatMenuPtr, SIGNAL(aboutToShow()), 
            this, SLOT(onMenuShow())); 

      addAction(formatMenuPtr, tr("&Color"), 
              SLOT(onColor()), QKeySequence(Qt::ALT + Qt::Key_C), 
              QString(), nullptr, tr("Figure Color")); 

      addAction(formatMenuPtr, tr("&Fill"), 
              SLOT(onFill()), QKeySequence(Qt::CTRL + Qt::Key_F), 
              QString(), nullptr, tr("Figure Fill"), 
              LISTENER(isFillEnabled)); 
```

当用户想要标记或修改现有图形而不是添加新图形时，他们选择修改项：

```cpp
     m_figureGroupPtr = new QActionGroup(this); 
     addAction(formatMenuPtr, tr("&Modify"), 
              SLOT(onModify()), 
              QKeySequence(Qt::CTRL + Qt::Key_M), 
              QString(), nullptr, tr("Modify Figure"), nullptr, 
              LISTENER(isModifyChecked), m_figureGroupPtr); 
```

图形菜单是一个包含 `Line`、`Rectangle` 和 `Ellipse` 项的子菜单。当我们将其添加到格式菜单时，它成为一个子菜单：

```cpp
    { QMenu* figureMenuPtr = 
               formatMenuPtr->addMenu(tr("&Figure")); 
      connect(figureMenuPtr, SIGNAL(aboutToShow()), 
              this, SLOT(onMenuShow())); 

      addAction(figureMenuPtr, tr("&Line"), 
                SLOT(onLine()), 
                QKeySequence(Qt::CTRL + Qt::Key_L), 
                QString(), nullptr, tr("Line Figure"), nullptr, 
                LISTENER(isLineChecked), m_figureGroupPtr); 

      addAction(figureMenuPtr, tr("&Rectangle"), 
                SLOT(onRectangle()), 
                QKeySequence(Qt::CTRL + Qt::Key_R), 
                QString(), nullptr, tr("Rectangle Figure"), 
                nullptr, LISTENER(isRectangleChecked), 
                m_figureGroupPtr); 

      addAction(figureMenuPtr, tr("&Ellipse"), 
                SLOT(onEllipse()), 
                QKeySequence(Qt::CTRL + Qt::Key_E), 
                QString(), nullptr, tr("Ellipse Figure"), nullptr, 
                LISTENER(isEllipseChecked), m_figureGroupPtr); 
    } 
  } 
} 

DrawingWindow::~DrawingWindow() { 
  delete m_figureGroupPtr; 
} 
```

# `DrawingWidget` 类

`DrawingWidget` 类是应用程序的主要类。它捕获鼠标和绘图事件。它还捕获文件、编辑和图形菜单项的选择。

**DrawingWidget.h:**

```cpp
    #ifndef DRAWINGWIDGET_H 
    #define DRAWINGWIDGET_H 

    #include "..\MainWindow\MainWindow.h" 
    #include "..\MainWindow\DocumentWidget.h" 
    #include "Figure.h" 

    class DrawingWidget : public DocumentWidget { 
      Q_OBJECT 

      public: 
      DrawingWidget(QWidget* parentWidget); 
      ~DrawingWidget(); 

      public: 
      void mousePressEvent(QMouseEvent *eventPtr); 
      void mouseMoveEvent(QMouseEvent *eventPtr); 
      void mouseReleaseEvent(QMouseEvent *eventPtr); 
      void mouseDoubleClickEvent(QMouseEvent *eventPtr); 
      void paintEvent(QPaintEvent *eventPtr); 

      private: 
      void newDocument(void); 
      bool writeFile(const QString& filePath); 
      bool readFile(const QString& filePath); 
      Figure* createFigure(FigureId figureId); 
```

与 第五章 的版本（*Qt 图形应用程序*）不同，这个版本覆盖了剪切和复制事件方法：

```cpp
     public slots: 
       bool isCopyEnabled(); 
       void onCopy(void); 
       bool isPasteEnabled(); 
       void onPaste(void); 
       void onDelete(void); 
       void onColor(void); 

       DEFINE_LISTENER(DrawingWidget, isFillEnabled); 
       void onFill(void); 

       DEFINE_LISTENER(DrawingWidget, isModifyChecked); 
       void onModify(void); 

       DEFINE_LISTENER(DrawingWidget, isLineChecked); 
       void onLine(void); 

       DEFINE_LISTENER(DrawingWidget, isRectangleChecked); 
       void onRectangle(void); 

       DEFINE_LISTENER(DrawingWidget, isEllipseChecked); 
       void onEllipse(void); 
```

`m_applicationMode` 字段包含 `Idle`、`ModifySingle` 或 `ModifyRectangle` 的值。当用户没有按鼠标时，`Idle` 模式是活动的。当用户抓住一个图形并修改或移动它（取决于用户抓住图形的哪个部分）时，`ModifySingle` 模式变为活动状态。最后，当用户在窗口中点击而没有点击图形时，`ModifyRectangle` 模式变为活动状态。在这种情况下，显示一个矩形，当用户释放鼠标按钮时，矩形内的每个图形都会被标记。用户可以删除或剪切粘贴标记的图形，或更改其颜色或填充状态。当用户释放鼠标按钮时，`Application` 模式再次变为 `Idle`：

```cpp
  private: 
    enum ApplicationMode {Idle, ModifySingle, ModifyRectangle}; 
    ApplicationMode m_applicationMode = Idle; 
    void setApplicationMode(ApplicationMode mode);
```

`m_actionMode` 字段包含 `Modify` 或 `Add` 的值。在 `Modify` 模式下，当用户用鼠标点击时，`m_applicationMode` 被设置为 `ModifySingle` 或 `ModifyRectangle`，具体取决于是否点击了图形。在 `Add` 模式下，无论用户是否点击了图形，都会添加一个新的图形。要添加的图形类型由 `m_addFigureId` 设置，它包含 `LineId`、`RectangleId` 或 `EllipseId` 的值：

```cpp
    enum ActionMode {Modify, Add}; 
    ActionMode m_actionMode = Add; 
    FigureId m_addFigureId = LineId; 
```

要添加到绘图中的下一个图形的颜色初始化为黑色，填充状态初始化为 false（未填充）。在两种情况下，用户都可以稍后更改它们：

```cpp
    QColor m_nextColor = Qt::black; 
    bool m_nextFilled = false; 
```

我们需要保存最新的鼠标点，以便计算鼠标移动之间的距离：

```cpp
    QPoint m_mousePoint; 
```

绘图图形的指针存储在 `m_figurePtrList` 中。最顶层的图形存储在列表的末尾。当用户剪切或复制一个或多个图形时，图形被复制，复制的指针存储在 `m_copyPtrList` 中：

```cpp
    QList<Figure*> m_figurePtrList, m_copyPtrList; 
```

当 `m_actionMode` 包含 `Modify` 且用户按下鼠标按钮而没有点击图形时，窗口中会出现一个矩形。这个矩形存储在 `m_insideRectangle` 中：

```cpp
     QRect m_insideRectangle; 
   }; 

   #endif // DRAWINGWIDGET_H 
```

**DrawingWidget.cpp:**

```cpp
    #include <CAssert> 
    #include "..\MainWindow\DocumentWidget.h" 

    #include "DrawingWidget.h" 
    #include "Line.h" 
    #include "Rectangle.h" 
    #include "Ellipse.h" 
```

构造函数调用基类 `DocumentWidget` 的构造函数，将窗口标题设置为 `Drawing Advanced`，并将绘图文件的文件后缀设置为 `drw`：

```cpp
    DrawingWidget::DrawingWidget(QWidget* parentWidget) 
    :DocumentWidget(tr("Drawing Advanced"), 
                 tr("Drawing files (*.drw)"), 
                 parentWidget) { 
       // Empty. 
    }  
```

析构函数不执行任何操作，仅为了完整性而包含：

```cpp
    DrawingWidget::~DrawingWidget() { 
       // Empty. 
    } 
```

`setApplicationMode` 方法设置应用程序模式，并在主窗口中调用 `onMenuShow` 以确保工具栏图标正确启用：

```cpp
    void DrawingWidget::setApplicationMode(ApplicationMode mode) { 
      m_applicationMode = mode; 
      ((MainWindow*) parent())->onMenuShow(); 
} 
```

当用户选择 `New` 菜单项时调用 `newDocument` 方法。我们首先在图形和复制指针列表中释放每个图形，然后它们自己清除列表：

```cpp
    void DrawingWidget::newDocument(void) { 
      for (Figure* figurePtr : m_figurePtrList) { 
      delete figurePtr; 
    } 

    for (Figure* copyPtr : m_copyPtrList) { 
      delete copyPtr; 
    } 

    m_figurePtrList.clear(); 
    m_copyPtrList.clear(); 
```

当前颜色和填充状态设置为黑色和 false（未填充）。操作模式设置为 `Add`，添加图形标识符设置为 `LineId`，这意味着当用户按下鼠标按钮时，会在绘图上添加一条黑色线条：

```cpp
      m_nextColor = Qt::black; 
      m_nextFilled = false; 
      m_actionMode = Add; 
      m_addFigureId = LineId; 
    } 
```

当用户选择`保存`或`另存为`菜单项时，会调用`writeFile`方法：

```cpp
    bool DrawingWidget::writeFile(const QString& filePath) { 
      ofstream outStream(filePath.toStdString()); 
```

如果文件成功打开，我们首先写入下一个颜色和填充状态：

```cpp
  if (outStream) { 
    writeColor(outStream, m_nextColor); 
    outStream.write((char*) &m_nextFilled, sizeof m_nextFilled); 
```

然后我们写入绘图中的图形数量，然后写入图形本身：

```cpp
      int size = m_figurePtrList.size(); 
      outStream.write((char*) &size, sizeof size); 
```

对于每个图形，首先我们写入其身份值，然后通过在其指针上调用`write`来写入图形本身。注意，我们不知道图形指针指向哪个类。我们不需要知道这一点，因为`write`是基类`Figure`中的纯虚方法：

```cpp
    for (Figure* figurePtr : m_figurePtrList) { 
      FigureId figureId = figurePtr->getId(); 
      outStream.write((char*) &figureId, sizeof figureId); 
      figurePtr->write(outStream); 
    } 
```

我们返回输出流转换为`bool`的结果，如果写入成功则为`true`：

```cpp
    return ((bool) outStream); 
  } 
```

如果文件未能成功打开，我们返回`false`：

```cpp
   return false; 
  } 
```

当用户选择打开菜单项时，会调用`readFile`方法。我们按照与之前`writeFile`中写入相同的顺序读取文件的部分：

```cpp
    bool DrawingWidget::readFile(const QString& filePath) { 
      ifstream inStream(filePath.toStdString()); 
```

如果文件成功打开，我们首先读取下一个颜色和填充状态：

```cpp
    if (inStream) { 
      readColor(inStream, m_nextColor); 
      inStream.read((char*) &m_nextFilled, sizeof m_nextFilled);
```

然后我们写入绘图中的图形数量，然后写入图形本身：

```cpp
    int size; 
    inStream.read((char*) &size, sizeof size); 
```

对于每个图形，首先我们读取其身份值，然后通过调用`createFigure`创建由身份值指示的图形类。最后，通过在其指针上调用`write`来读取图形本身：

```cpp
    for (int count = 0; count < size; ++count) { 
      FigureId figureId = (FigureId) 0; 
      inStream.read((char*) &figureId, sizeof figureId); 
      Figure* figurePtr = createFigure(figureId); 
      figurePtr->read(inStream); 
      m_figurePtrList.push_back(figurePtr); 
    } 
```

我们返回输入流转换为`bool`的结果，如果读取成功则为`true`：

```cpp
     return ((bool) inStream); 
   } 
```

如果文件未能成功打开，我们返回`false`：

```cpp
  return false; 
} 
```

`createFigure`方法根据`figureId`参数的值动态创建`Line`、`RectangleX`或`EllipseX`类的一个对象：

```cpp
    Figure* DrawingWidget::createFigure(FigureId figureId) { 
      Figure* figurePtr = nullptr; 

      switch (figureId) { 
        case LineId: 
         figurePtr = new Line(); 
         break; 

         case RectangleId: 
         figurePtr = new RectangleX(); 
         break; 

         case EllipseId: 
         figurePtr = new EllipseX(); 
         break; 
      } 

      return figurePtr; 
    } 
```

在编辑菜单可见之前调用`isCopyEnable`方法以启用复制项。框架也会调用它来启用复制工具栏图标。如果至少有一个图形被标记，它返回`true`，此时它就准备好被复制。如果它返回`true`，复制项和工具栏图标将变为启用状态：

```cpp
    bool DrawingWidget::isCopyEnabled() { 
      for (Figure* figurePtr : m_figurePtrList) { 
        if (figurePtr->marked()) { 
          return true; 
        } 
      } 

      return false; 
    } 
```

当用户选择复制菜单项时，会调用`onCopy`方法。首先，它释放复制指针列表中的每个图形并清除列表本身：

```cpp
    void DrawingWidget::onCopy(void) { 
      for (Figure* copyPtr : m_copyPtrList) { 
         delete copyPtr; 
      } 

      m_copyPtrList.clear(); 
```

然后，我们遍历图形指针列表，并将每个标记图形的指针添加到复制指针列表中。我们对每个图形指针调用`clone`以提供复制：

```cpp
   for (Figure* figurePtr : m_figurePtrList) { 
     if (figurePtr->marked()) { 
       m_copyPtrList.push_back(figurePtr->clone()); 
     } 
   } 
} 
```

在编辑菜单可见之前调用`isPasteEnabled`方法以启用粘贴项。框架也会调用它来启用粘贴工具栏图标。如果复制指针列表不为空，它返回`true`，从而启用粘贴项和图像。也就是说，如果有图形准备好粘贴，它返回`true`：

```cpp
   bool DrawingWidget::isPasteEnabled() { 
      return !m_copyPtrList.isEmpty(); 
   } 
```

当用户在编辑菜单中选择粘贴项，或者在选择编辑工具栏中的粘贴图像时，会调用`onPaste`方法。我们遍历复制指针列表，并在将其向下和向右移动 10 像素后，将图形的复制（通过调用`clone`获得）添加到图形指针列表中：

```cpp
    void DrawingWidget::onPaste(void) { 
      for (Figure* copyPtr : m_copyPtrList) { 
        Figure* pastePtr = copyPtr->clone(); 
        pastePtr->move(QSize(10, 10)); 
        m_figurePtrList.push_back(pastePtr); 
      } 
```

最后，当图形被添加到列表中时，我们通过调用`update`强制最终调用`paintEvent`：

```cpp
     update(); 
  } 
```

每次用户选择删除菜单项或工具栏图标时，都会调用`onDelete`方法。我们遍历图形指针列表并删除每个标记的图形：

```cpp
    void DrawingWidget::onDelete(void) { 
       for (Figure* figurePtr : m_figurePtrList) { 
         if (figurePtr->marked()) { 
         m_figurePtrList.removeOne(figurePtr); 
         delete figurePtr; 
       } 
     } 
```

此外，在这种情况下，我们在图形被删除后通过调用`update`方法强制最终调用`paintEvent`。

```cpp
     update(); 
  } 
```

每次用户在格式菜单中选择“颜色”项时，都会调用`onColor`方法。我们首先通过调用 Qt `QColorDialog`类的静态方法`getColor`来获取新的颜色：

```cpp
    void DrawingWidget::onColor(void) { 
      QColor newColor = QColorDialog::getColor(m_nextColor, this); 
```

如果颜色有效，即用户通过按下“确定”按钮而不是“取消”按钮关闭了对话框，并且他们选择了一种新的颜色，我们将下一个颜色设置为新的颜色并设置修改标志。我们还会遍历图形指针列表，并为每个标记的图形设置图形颜色：

```cpp
    if (newColor.isValid() && (m_nextColor != newColor)) { 
      m_nextColor = newColor; 
      setModifiedFlag(true); 

      for (Figure* figurePtr : m_figurePtrList) { 
        if (figurePtr->marked()) { 
          figurePtr->color() = m_nextColor; 
```

如果至少有一个图形被标记，我们将通过调用`update`强制最终调用`paintEvent`：

```cpp
          update(); 
         } 
       } 
     } 
   } 
```

在格式菜单中的“填充”项变得可见之前，会调用`isFillEnabled`方法：

```cpp
    bool DrawingWidget::isFillEnabled(void) { 
      switch (m_actionMode) { 
```

在“修改”模式下，我们遍历图形指针列表。如果至少有一个矩形或椭圆被标记，我们返回`true`并使项目启用：

```cpp
    case Modify: 
      for (Figure* figurePtr : m_figurePtrList) { 
        if (figurePtr->marked() && 
            ((figurePtr->getId() == RectangleId) || 
             (figurePtr->getId() == EllipseId))) { 
          return true; 
        } 
      } 
```

如果没有标记矩形或椭圆，我们返回`false`并使项目禁用：

```cpp
      return false; 
```

在“添加”模式下，如果用户将要添加的下一个图形是矩形或椭圆，我们返回`true`：

```cpp
    case Add: 
      return (m_addFigureId == RectangleId) || 
             (m_addFigureId == EllipseId); 
    } 
```

我们不应该到达这一点。`assert`宏调用仅用于调试目的。然而，我们仍然必须在方法末尾返回一个值：

```cpp
    assert(false); 
    return true; 
   } 
```

当用户在格式菜单中选择“填充”项时，会调用`onFill`方法：

```cpp
    void DrawingWidget::onFill(void) { 
      switch (m_actionMode) { 
```

在“修改”模式下，我们遍历图形指针列表并反转所有标记图形的填充状态。如果至少有一个图形发生变化，我们将通过调用`update`强制最终调用`paintEvent`：

```cpp
    case Modify: 
      for (Figure* figurePtr : m_figurePtrList) { 
        if (figurePtr->marked()) { 
          figurePtr->filled() = !figurePtr->filled(); 
          update(); 
        } 
      } 
```

我们还反转用户将要添加的下一个图形的填充状态：

```cpp
      m_nextFilled = !m_nextFilled; 
      break; 
```

在“添加”模式下，我们反转用户将要添加的下一个图形的填充状态：

```cpp
    case Add: 
       m_nextFilled = !m_nextFilled; 
       break; 
    } 
  } 
```

在格式菜单中的“修改”项变得可见之前，会调用`isModifyChecked`方法。在“修改”模式下，它返回`true`并启用项目：

```cpp
    bool DrawingWidget::isModifyChecked(void) { 
      return (m_actionMode == Modify); 
    } 
```

当用户在格式菜单中选择“修改”项时，会调用`onModify`方法。它将操作模式设置为“修改”：

```cpp
    void DrawingWidget::onModify(void) { 
      m_actionMode = Modify; 
    } 
```

在“添加”子菜单中的“线条”项变得可见之前，会调用`isLineChecked`方法。它返回`true`，并在添加操作模式下，如果下一个要添加的图形是线条，则项目（由于项目属于一组，因此带有单选按钮）变为选中状态：

```cpp
    bool DrawingWidget::isLineChecked(void) { 
      return (m_actionMode == Add) && (m_addFigureId == LineId); 
    } 
```

当用户在“添加”子菜单中选择“线条”项时，会调用`onLine`方法。它将操作模式设置为“添加”，并将用户将要添加到线条中的下一个图形设置为：

```cpp
    void DrawingWidget::onLine(void) { 
      m_actionMode = Add; 
      m_addFigureId = LineId; 
    } 
```

在`Add`子菜单中的`Rectangle`项变得可见之前，会调用`isRectangleChecked`方法。在`Add`动作模式下，如果下一个要添加的图形是矩形，它将返回`true`：

```cpp
    bool DrawingWidget::isRectangleChecked(void) { 
      return (m_actionMode == Add) && (m_addFigureId == RectangleId); 
    } 
```

当用户选择`Rectangle`项时，会调用`onRectangle`方法。它将动作模式设置为`Add`，并将用户将要添加的下一个图形设置为矩形：

```cpp
    void DrawingWidget::onRectangle(void) { 
      m_actionMode = Add; 
      m_addFigureId = RectangleId; 
    } 
```

在`Add`子菜单中的`Ellipse`项变得可见之前，会调用`isEllipseChecked`方法。在`Add`动作模式下，如果下一个要添加的图形是椭圆，它将返回`true`：

```cpp
    bool DrawingWidget::isEllipseEnabled(void) { 
      return !isEllipseChecked(); 
    } 
```

当用户选择`Ellipse`项时，会调用`onEllipse`方法。它将动作模式设置为`Add`，并将用户将要添加的下一个图形设置为椭圆：

```cpp
    void DrawingWidget::onEllipse(void) { 
      m_actionMode = Add; 
      m_addFigureId = EllipseId; 
    }  
```

当用户按下鼠标按钮之一时，会调用`mousePressEvent`方法。我们将鼠标点存储在`m_mousePoint`中，以便在`mouseMoveEvent`中使用，如下所示：

```cpp
    void DrawingWidget::mousePressEvent(QMouseEvent* eventPtr) { 
       if (eventPtr->buttons() == Qt::LeftButton) { 
       m_mousePoint = eventPtr->pos(); 
```

在`Modify`模式下，我们首先遍历图形指针列表，并取消标记每个图形：

```cpp
    switch (m_actionMode) { 
      case Modify: { 
          for (Figure* figurePtr : m_figurePtrList) { 
           figurePtr->marked() = false; 
          } 
```

我们然后再次遍历列表，以查找用户是否击中了图形。由于最顶部的图形被放置在列表的末尾，我们需要从列表的末尾开始遍历。我们通过使用 Qt `QList`类的`reverse_iterator`类型来实现这一点：

```cpp
     m_clickedFigurePtr = nullptr; 
     for (QList<Figure*>::reverse_iterator iterator = 
         m_figurePtrList.rbegin(); 
     iterator != m_figurePtrList.rend(); ++iterator) { 
        Figure* figurePtr = *iterator; 
```

如果我们通过在图形上调用`isClick`发现用户点击了图形，我们将应用程序模式设置为`ModifySingle`并标记该图形。我们还将它从列表中移除，并将其添加到列表的末尾，以便它在绘图中最先显示。最后，我们中断循环，因为我们已经找到了一个图形：

```cpp
    if (figurePtr->isClick(m_mousePoint)) { 
      setApplicationMode(ModifySingle); 
      m_clickedFigurePtr = figurePtr; 
      figurePtr->marked() = true; 
      m_figurePtrList.removeOne(figurePtr); 
      m_figurePtrList.push_back(figurePtr); 
      break; 
    } 
  } 
```

如果我们没有找到图形，我们将应用程序模式设置为`ModifyRectangle`，并将包含矩形的顶部和底部右角初始化为鼠标点：

```cpp
    if (m_clickedFigurePtr == nullptr) { 
      setApplicationMode(ModifyRectangle); 
      m_insideRectangle = QRect(m_mousePoint, m_mousePoint); 
    } 
    } 
    break; 
```

在`Add`动作模式下，我们通过调用`createFigure`并传递用户将要添加的下一个图形的标识符作为参数来创建一个新的图形。然后我们设置新图形的颜色、填充状态，并初始化其端点：

```cpp
      case Add: { 
          Figure* newFigurePtr = createFigure(m_addFigureId); 
          newFigurePtr->color() = m_nextColor; 
          newFigurePtr->filled() = m_nextFilled; 
          newFigurePtr->initializePoints(m_mousePoint); 
```

当新图形被创建并初始化后，我们将它添加到图形指针列表的末尾，并将应用程序模式设置为`ModifySingle`，因为`mouseMoveEvent`方法将继续修改列表中的最后一个图形，就像用户在`Modify`模式下击中了图形一样。我们还设置了修改标志，因为我们已经向绘图添加了一个图形：

```cpp
      m_figurePtrList.push_back(newFigurePtr); 
      setApplicationMode(ModifySingle); 
      setModifiedFlag(true); 
      } 
      break; 
    } 
```

最后，我们通过调用`update`强制调用`paintEvent`：

```cpp
        update(); 
      } 
    } 
```

当用户移动鼠标时，会调用`mouseMoveEvent`方法。如果他们同时按下鼠标左键，我们将鼠标点保存到未来的`mouseMoveEvent`调用中，并计算自上次调用`mousePressEvent`或`mouseMoveEvent`以来的距离：

```cpp
    void DrawingWidget::mouseMoveEvent(QMouseEvent* eventPtr) { 
      if (eventPtr->buttons() == Qt::LeftButton) { 
        QPoint newMousePoint = eventPtr->pos(); 
        QSize distance(newMousePoint.x() - m_mousePoint.x(), 
                       newMousePoint.y() - m_mousePoint.y()); 
        m_mousePoint = newMousePoint; 
```

在`Modify`模式下，我们通过调用`modify`来修改当前图形（位于图形指针列表末尾的图形）。请记住，图形可以是修改的，也可以是移动的，这取决于之前在`onMousePress`中调用`isClick`时的设置。我们还设置了修改标志，因为图形已经被更改：

```cpp
    switch (m_applicationMode) { 
      case ModifySingle: 
        m_figurePtrList.back()->modify(distance); 
        setModifiedFlag(true); 
        break; 
```

在包含矩形的情况下，我们只需更新其右下角。请注意，我们并没有设置修改标志，因为还没有任何图形被更改：

```cpp
    case ModifyRectangle: 
      m_insideRectangle.setBottomRight(m_mousePoint); 
      break; 
    } 
```

最后，通过调用`update`强制进行可能的`paintEvent`调用：

```cpp
        update(); 
      } 
   } 
```

当用户释放鼠标按钮时，会调用`mouseReleaseEvent`方法。如果是左键，我们会检查应用程序模式。我们唯一感兴趣的模式是包含矩形模式：

```cpp
     void DrawingWidget::mouseReleaseEvent(QMouseEvent* eventPtr) { 
       if (eventPtr->buttons() == Qt::LeftButton) { 
         switch (m_applicationMode) { 
           case ModifyRectangle: { 
             QList<Figure*> insidePtrList; 
```

我们遍历图形指针列表，并对每个图形调用`isInside`。每个完全被矩形包含的图形被标记，从列表中移除，并添加到`insidePtrList`中，稍后将其添加到图形指针列表的末尾：

```cpp
    for (Figure* figurePtr : m_figurePtrList) { 
      if (figurePtr->isInside(m_insideRectangle)) { 
        figurePtr->marked() = true; 
        m_figurePtrList.removeOne(figurePtr); 
        insidePtrList.push_back(figurePtr); 
      } 
    } 
```

每个完全被矩形包含的图形都会从图形指针列表中移除：

```cpp
    for (Figure* figurePtr : insidePtrList) { 
      m_figurePtrList.removeOne(figurePtr); 
    } 
```

最后，将所有包含的图形添加到列表的末尾，以便在绘图中最先显示：

```cpp
    m_figurePtrList.append(insidePtrList); 
    } 
    break; 
    } 
```

当用户释放鼠标按钮时，应用程序模式设置为空闲，并通过调用`update`强制进行可能的`paintEvent`调用：

```cpp
       setApplicationMode(Idle); 
       update(); 
      } 
    } 
```

当用户双击按钮之一时，会调用`mouseDoubleClick`方法。然而，`mouseClickEvent`总是在`mouseDoubleClickEvent`之前被调用。如果先前的`mouseClickEvent`调用使`m_clickedFigurePtr`指向被点击的图形，我们会在该图形上调用`doubleClick`。这可能会根据图形的类型引起一些变化：

```cpp
    void DrawingWidget::mouseDoubleClickEvent(QMouseEvent 
       *eventPtr) { 
      if ((eventPtr->buttons() == Qt::LeftButton) && 
          (m_clickedFigurePtr != nullptr)) { 
        m_clickedFigurePtr->doubleClick(eventPtr->pos()); 
        update(); 
      } 
    } 
```

最后，当窗口内容需要重绘时，会调用`paintEvent`。在调用之前，框架会清除窗口：

```cpp
    void DrawingWidget::paintEvent(QPaintEvent* /* 
       eventPtr */) { 
     QPainter painter(this); 
     painter.setRenderHint(QPainter::Antialiasing); 
     painter.setRenderHint(QPainter::TextAntialiasing); 
```

我们遍历图形指针列表，并绘制每个图形。列表中的最后一个图形被放置在列表的末尾，以便在绘图的最上方显示：

```cpp
     for (Figure* figurePtr : m_figurePtrList) { 
       figurePtr->draw(painter); 
     } 
```

在包含矩形模式下，我们用浅灰色边框绘制一个空心矩形：

```cpp
    if (m_applicationMode == ModifyRectangle) { 
      painter.setPen(Qt::lightGray); 
      painter.setBrush(Qt::NoBrush); 
      painter.drawRect(m_insideRectangle); 
    } 
  } 
```

# 主函数

`main`函数与之前应用程序的`main`函数类似——创建一个应用程序，显示绘图窗口，并开始应用程序的执行。

**Main.cpp:**

```cpp
    #include "DrawingWindow.h" 
    #include <QApplication> 

    int main(int argc, char *argv[]) { 
      QApplication application(argc, argv); 
      DrawingWindow drawingWindow; 
      drawingWindow.show(); 
      return application.exec(); 
    } 
```

下面的截图显示了输出：

![](img/e32d6956-fb1d-4513-bb7f-b7b7529a3466.png)

# 改进编辑器

本章的编辑器是第五章中编辑器的更高级版本，*Qt 图形应用程序*。在这个版本中，可以更改文本的字体和对齐方式，标记文本，以及剪切和粘贴文本。

# EditorWindow 类

本章的`EditorWindow`类与[第五章](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67)，*Qt 图形应用程序*中的类相似。它捕获按键事件和窗口关闭事件。

**EditorWindow.h:**

```cpp
    #ifndef EDITORWINDOW_H 
    #define EDITORWINDOW_H 

    #include <QMainWindow> 
    #include <QActionGroup> 
    #include <QPair> 
    #include <QMap> 

    #include "..\MainWindow\MainWindow.h" 
    #include "EditorWidget.h" 

    class EditorWindow : public MainWindow { 
      Q_OBJECT 

      public: 
      EditorWindow(QWidget *parentWidgetPtr = nullptr); 
      ~EditorWindow(); 

      protected: 
      void keyPressEvent(QKeyEvent* eventPtr); 
      void closeEvent(QCloseEvent* eventPtr); 

      private: 
      EditorWidget* m_editorWidgetPtr; 
      QActionGroup* m_alignmentGroupPtr; 
    }; 

    #endif // EDITORWINDOW_H 
```

**EditorWindow.cpp:**

```cpp
#include "EditorWindow.h" 
#include <QtWidgets> 
```

构造函数初始化编辑器窗口。它将窗口大小设置为 *1000 x 500* 像素。它还动态创建了一个编辑器小部件，并添加了标准的文件和编辑菜单：

```cpp
EditorWindow::EditorWindow(QWidget *parentWidgetPtr /*= nullptr*/)
 :MainWindow(parentWidgetPtr) {
  resize(1000, 500);

  m_editorWidgetPtr = new EditorWidget(this);
  setCentralWidget(m_editorWidgetPtr);
  addFileMenu();
  addEditMenu();
```

与第五章，*Qt 图形应用程序*相比，图形单菜单有所不同。我们添加了`字体`项和子菜单对齐，然后我们依次添加了三个项：左对齐、居中对齐和右对齐：

```cpp
      { QMenu* formatMenuPtr = menuBar()->addMenu(tr("F&ormat")); 
       connect(formatMenuPtr, SIGNAL(aboutToShow()), this, 
            SLOT(onMenuShow())); 
       addAction(formatMenuPtr, tr("&Font"), SLOT(onFont()), 
              0, QString(), nullptr, QString(), 
              LISTENER(isFontEnabled)); 

       { QMenu* alignmentMenuPtr = 
             formatMenuPtr->addMenu(tr("&Alignment")); 
        connect(alignmentMenuPtr, SIGNAL(aboutToShow()), 
              this, SLOT(onMenuShow())); 
```

我们还为`Alignment`菜单添加了一个工具栏：

```cpp
       QToolBar* alignmentToolBarPtr = addToolBar(tr("Alignment")); 
       m_alignmentGroupPtr = new QActionGroup(this); 

       addAction(alignmentMenuPtr, tr("&Left"), SLOT(onLeft()), 
                QKeySequence(Qt::ALT + Qt::Key_L), tr("left"), 
                alignmentToolBarPtr, tr("Left-aligned text"), 
                nullptr, LISTENER(isLeftChecked)); 
       addAction(alignmentMenuPtr, tr("&Center"), 
                SLOT(onCenter()), 
                QKeySequence(Qt::ALT + Qt::Key_C), 
                tr("center"), alignmentToolBarPtr, 
                tr("Center-aligned text"), nullptr, 
                LISTENER(isCenterChecked)); 
       addAction(alignmentMenuPtr, tr("&Right"), 
                SLOT(onRight()), 
                QKeySequence(Qt::ALT + Qt::Key_R), 
                tr("right"), alignmentToolBarPtr, 
                tr("Right-aligned text"), nullptr, 
                LISTENER(isRightChecked)); 
       } 
     } 

     m_editorWidgetPtr->setModifiedFlag(false); 
    } 

    EditorWindow::~EditorWindow() { 
     delete m_alignmentGroupPtr; 
    } 
```

关键按键事件和窗口关闭事件被传递给编辑器小部件：

```cpp
    void EditorWindow::keyPressEvent(QKeyEvent* eventPtr) { 
      m_editorWidgetPtr->keyPressEvent(eventPtr); 
    }
```

```cpp

    void EditorWindow::closeEvent(QCloseEvent* eventPtr) { 
      m_editorWidgetPtr->closeEvent(eventPtr); 
    } 
```

# `EditorWidget`类

`EditorWidget`类与第五章，*Qt 图形应用程序*中的版本相似。然而，已经添加了处理字体和对齐的方法和监听器。

**EditorWidget.h:**

```cpp
    #ifndef EDITORWIDGET_H 
    #define EDITORWIDGET_H 

    #include <QWidget> 
    #include <QMap> 
    #include <QMenu> 
    #include <QToolBar> 
    #include <QPair> 
    #include "Caret.h" 

    #include "..\MainWindow\Listener.h" 
    #include "..\MainWindow\DocumentWidget.h" 

    class EditorWidget : public DocumentWidget { 
      Q_OBJECT 

      public: 
      EditorWidget(QWidget* parentWidgetPtr); 
      void keyPressEvent(QKeyEvent* eventPtr); 

      private: 
      void keyEditPressEvent(QKeyEvent* eventPtr); 
      void keyMarkPressEvent(QKeyEvent* eventPtr); 
```

当用户按下鼠标按钮、移动鼠标和释放鼠标按钮时，会调用`mousePressEvent`、`mouseMoveEvent`和`mouseReleaseEvent`：

```cpp
    public: 
      void mousePressEvent(QMouseEvent* eventPtr); 
      void mouseMoveEvent(QMouseEvent* eventPtr); 
      void mouseReleaseEvent(QMouseEvent* eventPtr); 

    private: 
      int mouseToIndex(QPoint point); 

    public: 
      void paintEvent(QPaintEvent* eventPtr); 
      void resizeEvent(QResizeEvent* eventPtr); 
```

当用户选择新建菜单项时，会调用`newDocument`方法，当选择保存或另存为时，会调用`writeFile`，当选择打开菜单项时，会调用`readFile`：

```cpp
    private: 
      void newDocument(void); 
      bool writeFile(const QString& filePath); 
      bool readFile(const QString& filePath); 

    public slots: 
      bool isCopyEnabled(); 
      void onCopy(void); 
      bool isPasteEnabled(); 
      void onPaste(void); 
      void onDelete(void); 

      DEFINE_LISTENER(EditorWidget, isFontEnabled); 
      void onFont(void); 
```

在`Alignment`子菜单可见之前，会调用`isLeftChecked`、`isCenterChecked`和`isRightChecked`方法。然后它们会对选定的对齐方式标注一个单选按钮：

```cpp
    DEFINE_LISTENER(EditorWidget, isLeftChecked); 
    DEFINE_LISTENER(EditorWidget, isCenterChecked); 
    DEFINE_LISTENER(EditorWidget, isRightChecked); 
```

当用户选择对齐子菜单的任一项时，会调用`onLeft`、`onCenter`和`onRight`方法：

```cpp
      void onLeft(void); 
      void onCenter(void); 
      void onRight(void); 

      private: 
       void setCaret(); 
       void simulateMouseClick(int x, int y); 
```

在这个版本的编辑器中，我们有两种模式——编辑和标记。当用户输入文本或使用箭头键移动光标时，编辑标记是活动的，而当用户用鼠标标记代码块时，标记模式是活动的。光标在编辑模式下可见，但在标记模式下不可见：

```cpp
    private: 
      enum Mode {Edit, Mark} m_mode; 
```

文本可以沿左、中、右方向对齐：

```cpp
     enum Alignment {Left, Center, Right} m_alignment; 
```

在编辑模式下，`m_editIndex`持有用户将要输入的下一个字符的索引，这同时也是光标的位置。在标记模式下，`m_firstIndex`和`m_lastIndex`持有第一个和最后一个标记字符的索引：

```cpp
    int m_editIndex, m_firstIndex, m_lastIndex; 
```

`m_caret`对象持有编辑器的光标。光标在编辑模式下可见，但在标记模式下不可见：

```cpp
    Caret m_caret; 
```

`m_editorText`字段持有编辑器的文本，而`m_copyText`持有用户剪切或粘贴的文本：

```cpp
     QString m_editorText, m_copyText; 
```

编辑器的文本被分成行；每行的第一个和最后一个字符的索引存储在`m_lineList`中：

```cpp
    QList<QPair<int,int>> m_lineList; 
```

当前文本的字体存储在`m_textFont`中。当前字体中字符的高度（以像素为单位）存储在`m_fontHeight`中：

```cpp
    QFont m_textFont; 
    int m_fontHeight; 
```

`mousePressEvent` 和 `mouseMoveEvent` 方法存储最后一个鼠标点，以便计算鼠标事件之间的距离：

```cpp
    Qt::MouseButton m_button; 
```

与第五章方法中的方法类似，*Qt 图形应用程序*，`calculate` 是一个辅助方法，用于计算文本中每个字符的包含矩形。然而，本章的版本更复杂，因为它必须考虑文本是左对齐、居中对齐还是右对齐：

```cpp
    void calculate(); 
```

包含矩形存储在 `m_rectList` 中，然后由光标和 `paintEvent` 使用：

```cpp
     QList<QRect> m_rectList; 
    }; 

    #endif // EDITORWIDGET_H 
```

**EditorWidget.cpp:**

```cpp
    #include "EditorWidget.h" 
    #include <QtWidgets> 
    #include <CAssert> 
    using namespace std; 
```

构造函数将窗口标题设置为 `Editor Advanced` 并将文件后缀设置为 `edi`：

```cpp
    EditorWidget::EditorWidget(QWidget* parentWidgetPtr) 
     :DocumentWidget(tr("Editor Advanced"), 
         tr("Editor files (*.edi)"), parentWidgetPtr), 
```

文本字体初始化为 `12` 点 `Times New Roman`。将应用程序模式设置为编辑，将用户下一个要输入的字符的索引设置为零，并从开始将文本设置为左对齐：

```cpp
      m_textFont(tr("Times New Roman"), 12), 
        m_mode(Edit), 
        m_editIndex(0), 
        m_alignment(Left), 
        m_caret(this) { 
```

通过 `calculate` 计算包含字符的矩形，初始化并显示光标，因为应用程序从开始就持有编辑模式：

```cpp
     calculate(); 
     setCaret(); 
     m_caret.show(); 
   } 
```

当用户选择“新建”菜单项时调用 `newDocument` 方法。我们首先将应用程序模式设置为编辑并将编辑索引设置为零。文本字体设置为 `12` 点 Times New Roman。清除编辑器的文本，通过 `calculate` 计算包含字符的矩形，并设置光标：

```cpp
    void EditorWidget::newDocument(void) { 
      m_mode = Edit; 
      m_editIndex = 0; 
      m_textFont = QFont(tr("Times New Roman"), 12); 
      m_editorText.clear(); 
      calculate(); 
      setCaret(); 
    } 
```

当用户选择“保存”或“另存为”菜单项时调用 `writeFile` 方法。文件格式相当简单：我们在第一行写入字体，然后在以下行写入编辑器的文本：

```cpp
    bool EditorWidget::writeFile(const QString& filePath) { 
      QFile file(filePath); 
      if (file.open(QIODevice::WriteOnly | QIODevice::Text)) { 
      QTextStream outStream(&file); 
      outStream << m_textFont.toString() << endl << m_editorText; 
```

我们使用输入流的 `Ok` 字段来决定写入是否成功：

```cpp
     return ((bool) outStream.Ok); 
   } 
```

如果我们无法打开文件进行写入，我们返回 `false`：

```cpp
     return false; 
   } 
```

当用户选择“打开”菜单项时调用 `readFile` 方法。类似于之前的 `writeFile`，我们读取第一行并用文本初始化文本字体。然后读取编辑器文本：

```cpp
    bool EditorWidget::readFile(const QString& filePath) { 
      QFile file(filePath); 

      if (file.open(QIODevice::ReadOnly | QIODevice::Text)) { 
        QTextStream inStream(&file); 
        m_textFont.fromString(inStream.readLine()); 
        m_editorText = inStream.readAll(); 
```

当文本被读取时，我们调用 `calculate` 来计算包含文本字符的矩形。然后设置光标并返回 `true`，因为读取是成功的：

```cpp
      calculate(); 
      setCaret(); 
```

我们使用输入流的 `Ok` 字段来决定读取是否成功：

```cpp
     return ((bool) inStream.Ok); 
    } 
```

如果我们无法打开文件进行读取，我们 `return false`：

```cpp
    return false; 
   } 
```

在编辑菜单可见之前调用 `isCopyEnabled` 方法。框架也会调用它来决定是否启用复制工具栏图标。如果应用程序处于标记模式（意味着用户已标记文本的一部分，可以复制），则返回 `true`（并且项目变为启用状态）：

```cpp
    bool EditorWidget::isCopyEnabled() { 
      return (m_mode == Mark); 
    } 
```

当用户选择“复制”菜单项时调用 `onCopy` 方法。我们将标记的文本复制到 `m_EditorText`：

```cpp
    void EditorWidget::onCopy(void) { 
      int minIndex = qMin(m_firstIndex, m_lastIndex), 
      maxIndex = qMax(m_firstIndex, m_lastIndex); 

      m_copyText = 
        m_editorText.mid(minIndex, maxIndex - minIndex + 1); 
    } 
```

在编辑菜单可见之前也会调用 `isPasteEnabled` 方法。如果复制的文本不为空，它返回 `true`（并且项目变为可见）。也就是说，如果有一个已复制的文本块准备好粘贴：

```cpp
    bool EditorWidget::isPasteEnabled() { 
      return !m_copyText.isEmpty(); 
    } 
```

当用户选择粘贴菜单项时，会调用 `onPaste` 方法。在标记模式下，我们调用 `onDelete`，这将导致标记的文本被删除：

```cpp
    void EditorWidget::onPaste(void) { 
      if (m_mode == Mark) { 
         onDelete(); 
      } 
```

然后，我们将复制的文本插入到编辑器文本中。我们还更新 `m_editIndex`，因为文本被复制后，编辑索引应该是插入文本后的位置：

```cpp
     m_editorText.insert(m_editIndex, m_copyText); 
     m_editIndex += m_copyText.size(); 
```

最后，我们计算包含文本字符的矩形，将光标设置到新索引，设置修改标志，因为文本已被更改，并通过调用 `update` 强制调用 `paintEvent` 以显示新文本：

```cpp
     calculate(); 
     setCaret(); 
     setModifiedFlag(true); 
     update(); 
     } 
```

当用户选择删除菜单项或删除工具栏图标时，会调用 `onDelete` 方法。效果类似于用户按下 *Delete* 键时的事件。因此，我们准备一个带有 *Delete* 键的按键事件，将其用作 `keyPressEvent` 调用的参数：

注意，没有 `isDeleteEnabled` 方法，因为用户始终可以使用删除项。在编辑模式下，下一个字符被删除。在标记模式下，标记的文本被删除：

```cpp
    void EditorWidget::onDelete(void) { 
      QKeyEvent event(QEvent::KeyPress, Qt::Key_Delete, 
                  Qt::NoModifier); 
      keyPressEvent(&event); 
    } 
```

在格式菜单可见之前调用 `isCopyEnabled`。在编辑模式下它返回 `true`，因为当只有一部分字符被标记时，改变所有字符的字体是不合逻辑的：

```cpp
    bool EditorWidget::isFontEnabled() { 
      return (m_mode == Edit); 
    } 
```

当用户选择 `Font` 菜单项时，会调用 `onFont` 方法。我们让用户使用 Qt 的 `QFontDialog` 类选择新的字体：

```cpp
     void EditorWidget::onFont(void) { 
       bool pressedOkButton; 
       QFont newFont = 
         QFontDialog::getFont(&pressedOkButton, m_textFont, this); 
```

如果用户通过按下 Ok 按钮关闭对话框，我们将编辑器（`m_textFont`）字段的字体和修改标志设置：

```cpp
      if (pressedOkButton) { 
        m_textFont = newFont; 
        setModifiedFlag(true); 
```

我们通过调用 `calculate` 计算新包含的矩形，设置光标，并通过调用 `update` 强制调用 `paintEvent`：

```cpp
      calculate(); 
      m_caret.set(m_rectList[m_editIndex]); 
      update(); 
     } 
   } 
```

`isLeftChecked`, `isCenterChecked`, 和 `isRightChecked` 方法在对齐子菜单可见之前被调用。它们返回 `true` 给当前的对齐方式：

```cpp
    bool EditorWidget::isLeftChecked(void) { 
      return (m_alignment == Left); 
    } 

    bool EditorWidget::isCenterChecked(void) { 
      return (m_alignment == Center); 
    } 

    bool EditorWidget::isRightChecked(void) { 
      return (m_alignment == Right); 
    } 
```

当用户选择 `Left`、`Center` 和 `Right` 菜单项时，会调用 `onLeft`、`onCenter` 和 `onRight` 方法。它们设置对齐方式和修改标志。

它们也会计算新的包含矩形，设置光标，并通过调用 `update` 强制调用 `paintEvent`：

```cpp
    void EditorWidget::onLeft(void) { 
      m_alignment = Left; 
      setModifiedFlag(true); 
      calculate(); 
      setCaret(); 
      update(); 
    } 

    void EditorWidget::onCenter(void) { 
      m_alignment = Center; 
      setModifiedFlag(true); 
      calculate(); 
      setCaret(); 
      update(); 
    } 

    void EditorWidget::onRight(void) { 
      m_alignment = Right; 
      setModifiedFlag(true); 
      calculate(); 
      setCaret(); 
      update(); 
    } 
```

当用户按下鼠标按钮之一时，会调用 `mousePressEvent` 方法。我们调用 `mouseToIndex` 来找到用户点击的字符索引。暂时，第一个和最后一个标记索引被设置为鼠标索引。最后一个索引可能稍后通过以下片段中的 `mouseMoveEvent` 调用而改变。最后，模式被设置为标记，光标被隐藏：

```cpp
    void EditorWidget::mousePressEvent(QMouseEvent* eventPtr) { 
      if (eventPtr->buttons() == Qt::LeftButton) { 
         m_firstIndex = m_lastIndex = mouseToIndex(eventPtr->pos()); 
         m_mode = Mark; 
         m_caret.hide(); 
       } 
    } 
```

当用户移动鼠标时，会调用 `mouseMoveEvent` 方法。我们将最后一个标记索引设置为鼠标索引，并通过调用 `update` 强制调用 `paintEvent`：

```cpp
    void EditorWidget::mouseMoveEvent(QMouseEvent* eventPtr) { 
      if (eventPtr->buttons() == Qt::LeftButton) { 
         m_lastIndex = mouseToIndex(eventPtr->pos()); 
         update(); 
      } 
    } 
```

当用户释放鼠标按钮时，会调用 `mouseReleaseEvent` 方法。如果用户将鼠标移动到鼠标移动的原始起始位置，则没有标记要做，并将应用程序设置为编辑模式。在这种情况下，我们将编辑索引设置为第一个标记索引，并设置和显示插入点（因为它应该在编辑模式下可见）。最后，我们通过调用 `update` 强制调用 `paintEvent`：

```cpp
    void EditorWidget::mouseReleaseEvent(QMouseEvent* eventPtr) { 
      if (eventPtr->buttons() == Qt::LeftButton) { 
        if (m_firstIndex == m_lastIndex) { 
          m_mode = Edit; 
          m_editIndex = m_firstIndex; 
          setCaret(); 
          m_caret.show(); 
          update(); 
        } 
      } 
    } 
```

当用户在键盘上按下键时，会调用 `keyPressEvent`。根据应用程序模式（编辑或标记），我们调用 `keyEditPressEvent` 或以下 `keyMarkPressEvent` 以进一步处理按键事件：

```cpp
     void EditorWidget::keyPressEvent(QKeyEvent* eventPtr) { 
       switch (m_mode) { 
         case Edit: 
         keyEditPressEvent(eventPtr); 
         break; 

         case Mark: 
         keyMarkPressEvent(eventPtr); 
         break; 
       } 
     }
```

`keyEditPressEvent` 处理编辑模式下的按键。首先，我们检查键是否是箭头键、页面上下、*删除*、*退格*或回车键：

```cpp
    void EditorWidget::keyEditPressEvent(QKeyEvent* eventPtr) { 
      switch (eventPtr->key()) { 
```

在左箭头键的情况下，我们将编辑索引向后移动一步，除非它已经位于文本的开头：

```cpp
     case Qt::Key_Left: 
       if (m_editIndex > 0) { 
          --m_editIndex; 
       } 
       break; 
```

在右箭头键的情况下，我们将编辑索引向前移动一步，除非它已经位于文本的末尾：

```cpp
    case Qt::Key_Right: 
      if (m_editIndex < m_editorText.size()) { 
        ++m_editIndex; 
      } 
      break; 
```

在上箭头键的情况下，我们计算上一行字符的适当 `x` 和 `y` 位置，除非它已经在文本顶部。然后我们调用 `simulateMouseClick`，这会产生与用户点击行上方的字符相同的效果：

```cpp
     case Qt::Key_Up: { 
       QRect charRect = m_rectList[m_editIndex]; 

       if (charRect.top() > 0) { 
         int x = charRect.left() + (charRect.width() / 2), 
             y = charRect.top() - 1; 
         simulateMouseClick(x, y); 
       } 
     } 
     break; 
```

同样，在下箭头键的情况下，我们将编辑索引向下移动一行，除非它已经位于文本底部。

我们计算下一行字符的适当 `x` 和 `y` 位置，并调用 `simulateMouseClick`，这会产生与用户在点击点相同的效果：

```cpp
     case Qt::Key_Down: { 
       QRect charRect = m_rectList[m_editIndex]; 
       int x = charRect.left() + (charRect.width() / 2), 
           y = charRect.bottom() + 1; 
       simulateMouseClick(x, y); 
     } 
     break; 
```

在 *删除* 键的情况下，我们删除当前键，除非我们位于文本末尾。也就是说，如果我们比最后一个字符多一步：

```cpp
    case Qt::Key_Delete: 
      if (m_editIndex < m_editorText.size()) { 
        m_editorText.remove(m_editIndex, 1); 
        setModifiedFlag(true); 
      } 
      break; 
```

在退格键的情况下，我们将编辑索引向后移动一步，除非它已经位于文本的开头，并调用 `onDelete`。这样，我们就删除了前面的字符，并将编辑索引向后移动一步：

```cpp
     case Qt::Key_Backspace: 
     if (m_editIndex > 0) { 
       --m_editIndex; 
       onDelete(); 
     } 
     break; 
```

在回车键的情况下，我们简单地将在文本中插入新行字符：

```cpp
    case Qt::Key_Return: 
      m_editorText.insert(m_editIndex++, 'n'); 
      setModifiedFlag(true); 
      break; 
```

如果键不是特殊键，我们通过在按键事件指针上调用 `text` 来检查它是否是常规字符。如果文本不为空，则将其第一个字符添加到文本中：

```cpp
     default: { 
        QString text = eventPtr->text(); 

        if (!text.isEmpty()) { 
          m_editorText.insert(m_editIndex++, text[0]); 
          setModifiedFlag(true); 
        } 
     } 
     break; 
    } 
```

最后，我们计算包含的矩形，设置插入点，并通过调用 `update` 强制调用 `paintEvent`：

```cpp
    calculate(); 
    setCaret(); 
    update(); 
  } 
```

当用户在标记模式下按下键时，会调用 `keyMarkPressEvent`：

```cpp
    void EditorWidget::keyMarkPressEvent(QKeyEvent* eventPtr) { 
      switch (eventPtr->key()) { 
```

在左箭头键的情况下，我们将应用程序设置为编辑模式，并将编辑索引设置为第一个和最后一个标记索引的最小值。然而，如果最小索引位于文本开头，我们不做任何操作：

```cpp
    case Qt::Key_Left: { 
        int minIndex = qMin(m_firstIndex, m_lastIndex); 

        if (minIndex > 0) { 
          m_mode = Edit; 
          m_caret.show(); 
          m_editIndex = minIndex; 
        } 
      } 
      break; 
```

另一方面，在右箭头键的情况下，我们将应用程序设置为编辑模式，并将编辑索引设置为第一个和最后一个标记索引的最大值。然而，如果最大索引位于文本末尾，我们不做任何操作：

```cpp
    case Qt::Key_Right: { 
      int maxIndex = qMax(m_firstIndex, m_lastIndex); 

      if (maxIndex < m_editorText.size()) { 
        m_mode = Edit; 
        m_caret.show(); 
        m_editIndex = maxIndex;
```

```cpp
      } 
    } 
    break; 
```

在上下箭头的情况下，我们模拟在当前行上方或下方一行处的鼠标点击，就像在之前的编辑情况中一样：

```cpp
     case Qt::Key_Up: { 
       QRect charRect = m_rectList[m_editIndex]; 

       if (charRect.top() > 0) { 
         int x = charRect.left() + (charRect.width() / 2), 
             y = charRect.top() - 1; 
         simulateMouseClick(x, y); 
       } 
     } 
     break; 

     case Qt::Key_Down: { 
        QRect charRect = m_rectList[m_editIndex]; 
        int x = charRect.left() + (charRect.width() / 2), 
            y = charRect.bottom() + 1; 
        simulateMouseClick(x, y); 
      } 
      break; 
```

在标记模式下，删除键和退格键执行相同的任务——它们删除标记的文本：

```cpp
    case Qt::Key_Delete: 
    case Qt::Key_Backspace: { 
        int minIndex = qMin(m_firstIndex, m_lastIndex), 
            maxIndex = qMax(m_firstIndex, m_lastIndex); 
```

我们从编辑文本中移除标记的文本，设置修改标志，将应用程序设置为编辑模式，将编辑索引设置为第一个和最后一个标记索引的最小值，并显示光标：

```cpp
        m_editorText.remove(minIndex, maxIndex - minIndex); 
        setModifiedFlag(true); 
        m_mode = Edit; 
        m_editIndex = minIndex; 
        m_caret.show(); 
      } 
      break; 
```

回车键的情况与之前的编辑模式情况类似，不同之处在于我们首先删除标记的文本。然后我们在编辑器文本中添加一个新行：

```cpp
     case Qt::Key_Return: 
       onDelete(); 
       m_editorText.insert(m_editIndex++, 'n'); 
       setModifiedFlag(true); 
       break; 
```

如果键不是特殊键，我们通过在键事件指针上调用 `text` 来检查它是否是常规键。如果文本不为空，则用户打印了一个常规键，并且我们在编辑器文本中插入第一个字符：

```cpp
    default: { 
       QString text = eventPtr->text(); 

       if (!text.isEmpty()) { 
         onDelete(); 
         m_editorText.insert(m_editIndex++, text[0]); 
         setModifiedFlag(true); 
       } 
    } 
    break; 
    } 
```

最后，我们计算包围字符的新矩形，设置光标，并通过调用 `update` 强制调用 `paintEvent`：

```cpp
     calculate(); 
     setCaret(); 
     update(); 
    } 
```

当用户移动光标上下时调用 `simulateMouseClick` 方法。它通过调用 `mousePressEvent` 和 `mouseReleaseEvent` 并使用适当准备的事件对象来模拟鼠标点击：

```cpp
    void EditorWidget::simulateMouseClick(int x, int y) { 
      QMouseEvent pressEvent(QEvent::MouseButtonPress, QPointF(x, y), 
          Qt::LeftButton, Qt::NoButton, Qt::NoModifier); 
      mousePressEvent(&pressEvent); 
      QMouseEvent releaseEvent(QEvent::MouseButtonRelease, 
                     QPointF(x, y), Qt::LeftButton, 
                     Qt::NoButton, Qt::NoModifier); 
      mousePressEvent(&releaseEvent); 
    } 
```

`setCaret` 方法在编辑模式下设置光标到适当的大小和位置。首先，我们使用 `m_editIndex` 找到正确字符的矩形。然后我们创建一个只有一像素宽的新矩形，以便光标看起来像一条细的垂直线：

```cpp
    void EditorWidget::setCaret() { 
      QRect charRect = m_rectList[m_editIndex]; 
      QRect caretRect(charRect.left(), charRect.top(), 
                  1, charRect.height()); 
      m_caret.set(caretRect); 
    } 
```

`mouseToIndex` 方法接受一个鼠标点并返回该点的字符索引。与 第五章 的版本 *Qt 图形应用程序* 不同，我们需要考虑文本可能是居中或右对齐的：

```cpp
    int EditorWidget::mouseToIndex(QPoint point) { 
       int x = point.x(), y = point.y(); 
```

如果鼠标指针位于编辑器文本下方，则返回最后一个字符的索引：

```cpp
    if (y > (m_fontHeight * m_lineList.size())) { 
      return m_editorText.size(); 
    } 
```

否则，我们首先找到鼠标指针所在的行，并获取该行第一个和最后一个字符的索引：

```cpp
    else { 
      int lineIndex = y / m_fontHeight; 
      QPair<int,int> lineInfo = m_lineList[lineIndex]; 
      int firstIndex = lineInfo.first, lastIndex = lineInfo.second; 
```

如果鼠标指针位于行上第一个字符的左侧（如果文本是居中或右对齐，则可能如此），我们返回行的第一个字符的索引：

```cpp
     if (x < m_rectList[firstIndex].left()) { 
        return firstIndex; 
     } 
```

如果鼠标指针位于行的右侧，我们返回行中最后一个字符旁边的字符的索引：

```cpp
    else if (x >= m_rectList[lastIndex].right()) { 
      return (lastIndex + 1); 
    } 
```

否则，我们遍历行上的字符，并对每个字符检查鼠标指针是否位于该字符的包围矩形内：

```cpp
     else { 
       for (int charIndex = firstIndex + 1; 
           charIndex <= lastIndex; ++charIndex){ 
           int left = m_rectList[charIndex].left(); 
```

如果鼠标指针位于矩形内，我们检查它是否最接近矩形的左边界或右边界。如果它最接近左边界，我们返回字符的索引。如果它最接近右边界，我们则返回下一个字符的索引：

```cpp
        if (x < left) { 
          int last = m_rectList[charIndex - 1].left(); 
          int leftSize = x - last, rightSize = left - x; 
          return (leftSize < rightSize) ? (charIndex - 1) 
                                        : charIndex; 
          } 
        } 
      } 
    } 
```

我们不应该到达这个点。添加 `assert` 宏仅用于调试目的：

```cpp
      assert(false); 
      return 0; 
   } 
```

当用户调整窗口大小时，会调用`resizeEvent`方法。我们计算包含字符的矩形，因为窗口的宽度可能已更改，这可能导致行包含的字符数量减少或增加：

```cpp
    void EditorWidget::resizeEvent(QResizeEvent* eventPtr) { 
      calculate(); 
      DocumentWidget::resizeEvent(eventPtr); 
    } 
```

`calculate`方法将文本分成行，并计算包含文本中每个字符的矩形。每行的第一个和最后一个字符的索引存储在`m_lineList`中，包围的矩形存储在`m_rectList`中：

```cpp
    void EditorWidget::calculate() { 
      m_lineList.clear(); 
      m_rectList.clear(); 
```

我们使用 Qt `QFontMetrics`类来获取编辑器字体中字符的高度。高度存储在`m_fontHeight`中。`width`方法给出窗口内容的宽度，以像素为单位：

```cpp
      QFontMetrics metrics(m_textFont); 
      m_fontHeight = metrics.height(); 
      QList<int> charWidthList, lineWidthList; 
      int windowWidth = width(); 
```

我们首先遍历编辑器文本，以便将文本分成行：

```cpp
     { int firstIndex = 0, lineWidth = 0; 
        for (int charIndex = 0; charIndex < m_editorText.size(); 
           ++charIndex) { 
          QChar c = m_editorText[charIndex]; 
```

当我们遇到新行时，我们将当前行的第一个和最后一个索引添加到`m_lineList`中：

```cpp
      if (c == 'n') { 
        charWidthList.push_back(1); 
        lineWidthList.push_back(lineWidth); 
        m_lineList.push_back 
                   (QPair<int,int>(firstIndex, charIndex)); 
        firstIndex = charIndex + 1; 
        lineWidth = 0; 
      } 
```

否则，我们调用 Qt `QMetrics`对象的`width`方法来获取字符的宽度，以像素为单位：

```cpp
      else { 
        int charWidth = metrics.width(c); 
        charWidthList.push_back(charWidth); 
```

如果字符使行的宽度超过了窗口内容的宽度，我们将第一个和最后一个索引添加到`m_lineList`中，并开始新的一行。

然而，我们需要考虑两种不同的情况。如果当前字符是行的第一个字符，那么会出现（相当不可能）这种情况，即该字符的宽度超过了窗口内容的宽度。在这种情况下，我们将该字符的索引添加到`m_lineList`中作为第一个和最后一个索引。下一行的第一个索引是那个字符旁边的字符：

```cpp
    if ((lineWidth + charWidth) > windowWidth) { 
       if (firstIndex == charIndex) { 
         lineWidthList.push_back(windowWidth); 
         m_lineList.push_back 
              (QPair<int,int>(firstIndex, charIndex)); 
         firstIndex = charIndex + 1; 
       } 
```

如果当前字符不是行的第一个字符，我们将第一个字符和当前字符之前的字符的索引添加到`m_lineList`中。下一行的索引变为当前字符的索引：

```cpp
       else { 
          lineWidthList.push_back(lineWidth); 
          m_lineList.push_back(QPair<int,int>(firstIndex, 
              charIndex - 1)); 
          firstIndex = charIndex; 
       } 
       lineWidth = 0; 
    } 
```

如果字符没有使行的宽度超过窗口内容的宽度，我们只需将字符的宽度添加到行的宽度中：

```cpp
    else { 
          lineWidth += charWidth; 
         } 
    } 
  } 
```

最后，我们需要将最后一行添加到`m_lineList`中：

```cpp
      m_lineList.push_back(QPair<int,int>(firstIndex, 
           m_editorText.size() - 1)); 
      lineWidthList.push_back(lineWidth); 
    } 
```

当我们将文本分成行后，我们继续计算单个字符的包围矩形。我们首先将`top`设置为零，因为它持有行的顶部位置。对于每一行，它将增加行高：

```cpp
     { int top = 0, left; 
        for (int lineIndex = 0; lineIndex < m_lineList.size(); 
           ++lineIndex) { 
        QPair<int,int> lineInfo = m_lineList[lineIndex]; 
        int lineWidth = lineWidthList[lineIndex]; 
        int firstIndex = lineInfo.first, 
           lastIndex = lineInfo.second; 
```

根据文本的对齐方式，我们需要决定行从哪里开始。在左对齐的情况下，我们将行的左位置设置为零：

```cpp
      switch (m_alignment) { 
        case Left: 
          left = 0; 
          break; 
```

在居中对齐的情况下，我们将左位置设置为窗口内容宽度与行宽度的差的一半。这样，行将出现在窗口的中心：

```cpp
        case Center: 
          left = (windowWidth - lineWidth) / 2; 
          break; 
```

在右对齐的情况下，我们将左位置设置为窗口内容宽度与行宽度的差。这样，行将看起来在窗口的右侧：

```cpp
       case Right: 
          left = windowWidth - lineWidth; 
          break; 
       } 
```

最后，当我们确定了文本行的起始左位置和每个文本字符的宽度后，我们遍历该行并计算每个字符的包围矩形：

```cpp
     for (int charIndex = firstIndex; 
           charIndex <= lastIndex;++charIndex){ 
        int charWidth = charWidthList[charIndex]; 
        QRect charRect(left, top, charWidth, m_fontHeight); 
        m_rectList.push_back(charRect); 
        left += charWidth; 
     } 
```

对于文本的最后一行，我们添加一个包含超出最后一个字符位置的矩形：

```cpp
      if (lastIndex == (m_editorText.size() - 1)) { 
        QRect lastRect(left, top, 1, m_fontHeight); 
        m_rectList.push_back(lastRect); 
      } 
```

每行新行时，顶部字段增加行高：

```cpp
          top += m_fontHeight; 
        } 
      } 
    } 
```

框架在每次需要重绘窗口或通过调用`update`强制重绘时都会调用`paintEvent`方法。在调用`paintEvent`之前，框架会清除窗口的内容：

首先，我们创建一个`QPinter`对象，然后使用它来书写。我们设置了某些渲染和文本字体：

```cpp
     void EditorWidget::paintEvent(QPaintEvent* /* eventPtr */) { 
       QPainter painter(this); 
       painter.setRenderHint(QPainter::Antialiasing); 
       painter.setRenderHint(QPainter::TextAntialiasing); 
       painter.setFont(m_textFont); 
```

我们计算被标记文本的最小和最大索引（即使我们尚不知道应用程序是否处于标记模式）：

```cpp
    int minIndex = qMin(m_firstIndex, m_lastIndex), 
       maxIndex = qMax(m_firstIndex, m_lastIndex); 
```

我们遍历编辑器的文本。我们书写除换行符外的每个字符：

```cpp
     for (int index = 0; index < m_editorText.length(); ++index) { 
       QChar c = m_editorText[index]; 
```

如果字符被标记，我们使用白色文本在黑色背景上书写：

```cpp
    if (c != 'n') { 
      if ((m_mode == Mark) && 
          (index >= minIndex) && (index < maxIndex)) { 
        painter.setPen(Qt::white); 
        painter.setBackground(Qt::black); 
      } 
```

如果字符没有被标记，我们使用黑色文本在白色背景上书写：

```cpp
      else { 
        painter.setPen(Qt::black); 
        painter.setBrush(Qt::white); 
      } 
```

当文本和背景的颜色设置好后，我们查找包含字符的矩形并书写字符本身：

```cpp
      QRect rect = m_rectList[index]; 
      painter.drawText(rect, c); 
    } 
  } 
```

最后，我们还绘制了光标：

```cpp
      m_caret.paint(&painter); 
    } 
```

# 主函数

`main`函数与之前应用程序的`main`函数类似：它创建一个应用程序，显示绘图窗口，并开始应用程序的执行。

**Main.cpp:**

```cpp
#include "EditorWindow.h" 
#include <QApplication> 

int main(int argc, char *argv[]) { 
  QApplication application(argc, argv); 
  EditorWindow editorWindow; 
  editorWindow.show(); 
  return application.exec(); 
} 
```

下面的截图显示了输出：

![截图](img/4191c0b5-fcd6-4a9e-a3e4-540323edcd89.png)

# 摘要

在本章中，我们开发了更高级的模拟时钟、绘图程序和编辑器版本。时钟显示当前的小时、分钟和秒。绘图程序允许用户绘制线条、矩形和椭圆。编辑器允许用户输入和编辑文本。时钟面使用数字而不是线条。在绘图程序中，我们可以标记、修改，以及剪切和粘贴图形，而在编辑器中，我们可以更改字体和段落对齐方式，并标记文本块。

在第七章“游戏”中，我们将开始开发奥赛罗和井字棋游戏。
