# Enhancing the Qt Graphical Applications

In [Chapter 5](411aae8c-9215-4315-8a2e-882bf028834c.xhtml), *Qt Graphical Applications*, we developed graphical Qt applications involving an analog clock, a drawing program, and an editor. In this chapter, we will continue to work on the three graphical applications of  [Chapter 5](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67), *Qt Graphical Applications*. However, we will make the following improvements:

*   **Clock**: We will add digits to the clock dial
*   **The drawing program**: We will add the ability to move and modify figures, to cut and paste them, and to mark one or several figures
*   **The editor**: We will add the ability to change font and alignment as well as to mark a text block

In this chapter, we will continue to work with the Qt libraries:

*   Windows and widgets
*   Menus and toolbars
*   Mouse and keyboard events

# Improving the clock

In this chapter, we will replace the version of clock dial markings with digits.

# The Clock class

The `Clock` class definition is similar to the one in [Chapter 5](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67), *Qt Graphical Applications*. The timer updates the window 10 times each second. The constructor initializes the clock and `paintEvent` is called every time the window needs to be repainted.

**Clock.h:**

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

**Clock.cpp:**

```cpp
   #include <QtWidgets> 
   #include "Clock.h" 
```

Similar to [Chapter 5](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67), *Qt Graphical Applications*, the constructor sets the header of the window to `Clock Advanced`, the window size to *1000* x *500* pixels, initializes the timer to send a timeout message every `100` milliseconds, and connect the `timeout` message to the `update` method, which forces the window to be repainted for each timeout:

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

The `paintEvent` method is called every time the window needs to be repainted. We will start by calculating the side of the clock and obtaining the current time:

```cpp
    void Clock::paintEvent(QPaintEvent* /*event*/) { 
      int side = qMin(width(), height()); 
      QTime time = QTime::currentTime(); 
```

We then create and initialize a `QPainter` object. We call `translate` and `scale` to match the physical size (pixels) to the logical size of *200 x 200* units:

```cpp
    QPainter painter(this); 
    painter.setRenderHint(QPainter::Antialiasing); 
    painter.setRenderHint(QPainter::TextAntialiasing); 
    painter.translate(width() / 2, height() / 2); 
    painter.scale(side / 200.0, side / 200.0); 
```

As we write digits to the clock in this version of the chapter, we add the font `Times New Roman`, `12` points, to the painter:

```cpp
    painter.setFont(QFont(tr("Times New Roman"), 12)); 
```

We write the digits of the clock, `1` to `12`, as shown in the following code:

```cpp
     for (int hour = 1; hour <= 12; ++hour) { 
       QString text; 
       text.setNum(hour); 
```

A whole leap is 360° and the angle between two consecutive digits is 30°, since 360 divided by 12 is 30:

```cpp
    double angle = (30.0 * hour) - 90; 
    double radius = 90.0; 
```

The `x` and `y` coordinates of the digits are calculated by the sine and cosine functions. However, first, we need to transform the degrees to radians since sine and cosine accept radians only. This is shown in the following code:

```cpp
    double x = radius * qCos(qDegreesToRadians(angle)), 
           y = radius * qSin(qDegreesToRadians(angle)); 
```

The `drawText` methods write the digit, as follows:

```cpp
     QRect rect(x - 100, y - 100, 200, 200); 
     painter.drawText(rect, Qt::AlignHCenter | 
                            Qt::AlignVCenter, text); 
     } 
```

When the digits have been written, we draw the `hour`, `minute`, and `second` hands in the same way as in [Chapter 5](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67), *Qt Graphical Applications*:

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

# The main function

The `main` function is similar to the one in [Chapter 5](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67), *Qt Graphical Applications*. It creates an application object, initializes the clock, and executes the application.

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

**Output**:

![](img/4fc9d538-a689-43c2-9e55-d697cc18342c.png)

# Improving the drawing program

The drawing program of this chapter is a more advanced version of the drawing program of [Chapter 5](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67), *Qt Graphical Applications.* In this version, it is possible to modify a figure, to enclose one or more figures and then change their colors, and to cut and paste figures.

# The Figure class

The `Figure` class is rather similar to the one in [Chapter 5](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67), *Qt Graphical Applications*. However, `isInside`, `doubleClick`, `modify`, and `marked` have been added.

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

In this version, the pure virtual `clone` method has been added. That is due to the cut and paste. When pasting a figure we want to create a copy of it, without actually knowing which class the object belongs to. We could not do that with the copy constructor only. This is actually the main point of this section: how to use pure virtual methods and how to take advantage of dynamic binding. We need `clone`, which calls the copy constructor of its class to return a pointer to the new object:

```cpp
    virtual Figure* clone() const = 0; 

    virtual FigureId getId() const = 0; 
    virtual void initializePoints(QPoint point) = 0; 
```

In this version of the drawing program, `onClick` sets fields to indicate whether the figure shall be modified or moved. If the user grabs one of the marked points of the figure (which varies between different kinds of figures), the figure shall be modified. Otherwise, it shall be moved. The `modify` method is called when the user grabs one of the corners of the figure. In that case, the figure shall be modified rather than moved:

```cpp
    virtual bool isClick(QPoint mousePoint) = 0; 
    virtual void modify(QSize distance) = 0; 
```

The `isInside` method returns `true` if the figure is completely enclosed in the area. It is called when the user encloses figures with the mouse:

```cpp
    virtual bool isInside(QRect area) = 0; 
```

The `doubleClick` method is called when the user double-clicks at the figure, each figure performs some suitable action:

```cpp
    virtual void doubleClick(QPoint mousePoint) = 0; 

    virtual void move(QSize distance) = 0; 
    virtual void draw(QPainter &painter) const = 0; 

    virtual bool write(ofstream& outStream) const; 
    virtual bool read(ifstream& inStream); 
```

The `marked` methods return and set the `m_marked` field. When a figure is marked, it is annotated with small squares:

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

The `write` and `read` methods write and read the color of the figure and whether it is filled. However, they do not write or read the marked status. A figure is always unmarked when written or read:

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

# The Line class

The `Line` class is a subclass of `Figure`.

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

As mentioned in the preceding `Figure` section, `isClick` decided whether the line shall be modified or moved. If the user grabs one of its endpoints, only that endpoint shall be moved. If the user grabs the line between the endpoints, the line shall be moved. That is, both the endpoints of the line shall be moved:

```cpp
    bool isClick(QPoint mousePoint); 
```

The `isInside` method checks whether the line is completely enclosed by the area:

```cpp
    bool isInside(QRect area); 
```

The `doubleClick` method does nothing in the `Line` class. However, we still need to define it, since it is pure virtual in `Figure`. If we had not defined it, `Line` would have been abstract:

```cpp
    void doubleClick(QPoint /* mousePoint */) {/* Empty. */} 
```

The `modify` method modifies the line in accordance with the settings of the preceding `isClick`. If the user grabs one of the endpoints, that endpoint is moved. Otherwise, the whole line (both the endpoints) is moved:

```cpp
    void modify(QSize distance); 
    void move(QSize distance); 
```

The `area` method returns a slightly larger area if the line is marked, in order to include the marking squares:

```cpp
    QRect area() const; 
    void draw(QPainter& painter) const; 

    bool write(ofstream& outStream) const; 
    bool read(ifstream& inStream); 
```

The `m_lineMode` field keeps track of the movement or modification of the line. When the line is created, `m_lineMode` is set to `LastPoint`. When the user grabs the first or last endpoint of the line, `m_lineMode` is set to `FirstPoint` or `LastPoint`. When the user grabs the line between the endpoints, `m_lineMode` is set to `MoveLine`:

```cpp
      private: 
         enum {FirstPoint, LastPoint, MoveLine} m_lineMode; 
         QPoint m_firstPoint, m_lastPoint; 
```

The `isPointInLine` method decides whether the user has clicked on the line, with some tolerance:

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

When a line becomes created, the line mode is set to the last point. That means that the last point of the line will be changed when the user moves the mouse:

```cpp
    Line::Line() 
    :m_lineMode(LastPoint) { 
      // Empty. 
    } 
```

The `clone` method is called when a line is being pasted. The copy constructor of `Figure` is called to set the color of the figure. Note that we call the `Figure` constructor with a `Line` object as a parameter, even though it takes a reference to a `Figure` object as a parameter. We are allowed to do this since `Line` is a subclass of `Figure` and the `Line` object will be transformed into a `Figure` object during the call. Moreover, the first and last endpoints are copied. Note that we do need to copy the value `m_lineMode` since its value is set when the user creates, modifies, or moves the line only:

```cpp
    Line::Line(const Line& line) 
     :Figure(line), 
       m_firstPoint(line.m_firstPoint), 
       m_lastPoint(line.m_lastPoint) { 
      // Empty. 
     } 
```

The `clone` method uses the copy constructor to create a new object, which is then returned:

```cpp
    Figure* Line::clone() const { 
      Line* linePtr = new Line(*this); 
      return linePtr; 
    } 
```

The `initializePoints` method is called shortly after the line is being created. The reason for this call is that we do not create a `Line` object directly. Instead, we create the line indirectly by calling `clone`. We then need to initialize the end-points by calling `initializePoints`:

```cpp
    void Line::initializePoints(QPoint point) { 
      m_firstPoint = point; 
      m_lastPoint = point; 
    } 
```

The `isClick` method is called when the user clicks with the mouse. First, we check whether they have clicked at the first endpoint. We use the `Tolerance` field to create a small square, with the first endpoint in its center. If the user clicks on the square, `m_lineMode` is set to `FirstPoint` and `true` is returned:

```cpp
    bool Line::isClick(QPoint mousePoint) { 
      QRect firstSquare(makeRect(m_firstPoint, Tolerance)); 

     if (firstSquare.contains(mousePoint)) { 
       m_lineMode = FirstPoint; 
       return true; 
     } 
```

In the same way, we create a small square with the last endpoint in its center. If the user clicks at the square, `m_lineMode` is set to `LastPoint` and `true` is returned:

```cpp
    QRect lastSquare(makeRect(m_lastPoint, Tolerance)); 

    if (lastSquare.contains(mousePoint)) { 
      m_lineMode = LastPoint; 
      return true; 
    } 
```

If the user does not click on either of the endpoints, we check if they click on the line itself. If they do, `m_lineMode` is set to `ModeLine` and `true` is returned:

```cpp
    if (isPointInLine(m_firstPoint, m_lastPoint, mousePoint)) { 
      m_lineMode = MoveLine; 
      return true; 
    } 
```

Finally, if the user does not click on one of the endpoints or the line itself, they missed the line altogether and `false` is returned:

```cpp
    return false; 
    } 
```

The `isInside` method returns `true` if the line is completely enclosed by the area. It is quite easy, we just check whether the two end-points are located inside the area:

```cpp
    bool Line::isInside(QRect area) { 
     return area.contains(m_firstPoint) && 
       area.contains(m_lastPoint); 
    } 
```

The `isPointInLine` method is identical to `isClick` in the version of [Chapter 5](411aae8c-9215-4315-8a2e-882bf028834c.xhtml), *Qt Graphical Application**s*:

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

The `modify` method moves the first or last endpoint, or both of them, depending on the settings of `m_lineMode` in the preceding `isClick` method:

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

The `move` method simply moves both the end-points of the line:

```cpp
    void Line::move(QSize distance) { 
      m_firstPoint += distance; 
      m_lastPoint += distance; 
    } 
```

The `draw` method draws the line. The difference between this version and the version of [Chapter 5](411aae8c-9215-4315-8a2e-882bf028834c.xhtml), *Qt Graphical Applications*, is that it also draws the squares at the end-points of the line if it is marked:

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

The `area` method returns the area covering the line. If the line is marked, the area is slightly expanded in order to cover the squares marking the endpoints:

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

Similar to the version of [Chapter 5](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67), *Qt Graphical Applications*, `write` and `read` call their counterparts in `Figure` and then write and read the two endpoints of the line:

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

# The Rectangle class

`RectangleX` is a subclass of `Figure`. It is an expanded version of the version of [Chapter 5](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67), *Qt Graphical Applications*. The `isClick` method has been modified, `doubleClick` and `modify` have been added.

**Rectangle.h:**

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

**Rectangle.cpp:**

```cpp
    #include <CAssert> 
    #include "..\MainWindow\DocumentWidget.h" 
    #include "Rectangle.h" 
```

When a rectangle is added by the user, its mode is `BottomRightPoint`. That means that the bottom-right corner of the rectangle will be moved when the user moves the mouse:

```cpp
    RectangleX::RectangleX() 
    :m_rectangleMode(BottomRightPoint) { 
      // Empty. 
    } 
```

The copy constructor copies the rectangle. More specifically, first it calls the copy constructor of the `Figure` class, then it copies the top-left and bottom-right corner. Note that it does not copy the `m_rectangleMode` field, since it is used when the user moves the mouse only:

```cpp
    RectangleX::RectangleX(const RectangleX& rectangle) 
    :Figure(rectangle), 
      m_topLeft(rectangle.m_topLeft), 
      m_bottomRight(rectangle.m_bottomRight) { 
      // Empty. 
    } 
```

The `clone` method creates and returns a pointer to a new object by calling the copy constructor:

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

The `isClick` method is called when the user clicks with the mouse. Similar to the preceding bool `Line`, we start by checking whether they have clicked at any of the corners. If they have not, we check whether they have clicked on the rectangle border or inside the rectangle, depending on whether it is filled.

We start by defining a small square covering the top-left corner. If the user clicks on it, we set the `m_rectangleMode` field to `TopLeftPoint` and return `true`:

```cpp
    bool RectangleX::isClick(QPoint mousePoint) { 
      QRect topLeftRect(makeRect(m_topLeft, Tolerance)); 

      if (topLeftRect.contains(mousePoint)) { 
        m_rectangleMode = TopLeftPoint; 
        return true; 
      } 
```

We continue by defining a square covering the top-right corner. If the user clicks on it, we set `m_rectangleMode` to `TopRightPoint` and return `true`:

```cpp
     QPoint topRightPoint(m_bottomRight.x(), m_topLeft.y()); 
     QRect topRectRight(makeRect(topRightPoint, Tolerance)); 

     if (topRectRight.contains(mousePoint)) { 
       m_rectangleMode = TopRightPoint; 
       return true; 
     } 
```

If the user clicks at the square covering the bottom-right corner, we set `m_rectangleMode` to `BottomRightPoint` and return `true`:

```cpp
     QRect m_bottomRightRect(makeRect(m_bottomRight, Tolerance)); 

     if (m_bottomRightRect.contains(mousePoint)) { 
       m_rectangleMode = BottomRightPoint; 
       return true; 
     } 
```

If the user clicks at the square covering the bottom-left corner, we set `m_rectangleMode` to `BottomLeftPoint` and return `true`:

```cpp
    QPoint bottomLeftPoint(m_topLeft.x(), m_bottomRight.y()); 
    QRect bottomLeftRect(makeRect(bottomLeftPoint, Tolerance)); 

    if (bottomLeftRect.contains(mousePoint)) { 
      m_rectangleMode = BottomLeftPoint; 
      return true; 
    } 
```

If the user does not click at any of the corners of the rectangle, we check the rectangle itself. If it is filled, we check whether the mouse pointer is located inside the rectangle itself. If it is, we set `m_rectangleMode` to `MoveRectangle` and return `true`:

```cpp
  QRect areaRect(m_topLeft, m_bottomRight); 

  if (filled()) { 
    if (areaRect.contains(mousePoint)) { 
      m_rectangleMode = MoveRectangle; 
      return true; 
    } 
  } 
```

If the rectangle is not filled, we define slightly larger and smaller rectangles. If the mouse click is located inside the larger rectangle, but not in the smaller one, we set `m_rectangleMode` to `MoveRectangle` and return `true`:

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

Finally, if the user does not click at one of the corners or the rectangle itself, they missed the rectangle and we return `false`:

```cpp
      return false; 
    } 
```

The `isInside` method is quite simple. We simply check if the top-left and bottom-right corners are located inside the rectangle:

```cpp
     bool RectangleX::isInside(QRect area) { 
       return area.contains(m_topLeft) && 
         area.contains(m_bottomRight); 
     } 
```

The `doubleClick` method is called when the user double-clicks with the mouse. If the call to `onClick` returns `true`, `doubleClick` is called. In the rectangle case, the filled status is changed—a filled rectangle becomes unfilled and an unfilled rectangle becomes filled:

```cpp
     void RectangleX::doubleClick(QPoint mousePoint) { 
       if (isClick(mousePoint)) { 
```

The first call to `filled` is a call to the version that returns a reference to the `m_filled` field, which allows us to change the returned value:

```cpp
     filled() = !filled(); 
    } 
  } 
```

The `modify` method modifies the rectangle in accordance with the `m_rectangleMode` field, which was set by the preceding `isClick`. If it is set to one of the four corners, we modify that corner. If not, we move the whole rectangle:

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

The `move` method is quite simple. It just changes the top-left and bottom-right corners:

```cpp
    void RectangleX::move(QSize distance) { 
      m_topLeft += distance; 
      m_bottomRight += distance; 
    } 
```

The `area` method returns the area covering the rectangle. If it is marked, we slightly expand the area in order for it to cover the marking squares:

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

The `draw` method draws the rectangle; with a full brush it is filled and with a hollow brush if it is unfilled:

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

If the rectangle is marked, the four squares covering the corners of the rectangle are also drawn:

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

The `write` and `read` methods first call their counterparts in `Figure` in order to write and read the color of the rectangle. Then it writes and reads the top-left and bottom-right corners:

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

# The Ellipse class

`EllipseX` is a direct sub class of `RectangleX` and an indirect subclass of `Figure` that draws a filled or unfilled ellipse:

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

Similar to the preceding rectangle case, `isClick` checks whether the user grabs the ellipse in one of its four corners, or if the ellipse itself shall be moved:

```cpp
    bool isClick(QPoint mousePoint); 
```

The `modify` method modifies the ellipse in accordance with the settings of following `m_ellipseMode` in preceding `isClick`:

```cpp
    void modify(QSize distance); 
    void draw(QPainter& painter) const; 
```

While the preceding rectangle could be grabbed by its four corners, the ellipse can be grabbed by its left, top, right, and bottom points. Therefore, we need to add the `CreateEllipse` enumeration value, which modifies the bottom-right corner of the area covering the ellipse:

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

In contrast to the preceding line and rectangle cases, we set the `m_ellipseMode` field to `CreateEllipse`, which is valid when the ellipse is being created only:

```cpp
     EllipseX::EllipseX() 
      :m_ellipseMode(CreateEllipse) { 
      // Empty. 
     } 
```

The copy constructor does not need to set the `m_topLeft` and `m_bottomRight` fields, since it is taken care of by the copy constructor of `RectangleX`, which is being called by the copy constructor of `EllipseX`:

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

Similar to the preceding rectangle case, `isClick` checks whether the user grabs the ellipse by one of its four points. However, in the ellipse case, we do not check the corners of the rectangle. Instead, we check the left, top, right, and bottom position of the ellipse. We create a small square for each of those positions and check whether the user clicks on them. If they do, we set the `m_ellipseMode` field to an appropriate value and return `true`:

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

If the user does not click on any of the four positions, we check whether they click on the ellipse itself. If it is filled, we use the Qt `QRegion` class to create an elliptic region and we check whether the mouse point is located inside the region:

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

If the ellipse is unfilled, we create slightly larger and smaller elliptic regions and then check whether the mouse point is located inside the larger region, and also inside the smaller one:

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

Finally, if the user does not click at any of the grabbing positions or the ellipse itself, we return `false`:

```cpp
      return false; 
   } 
```

The `modify` method modifies the ellipse in accordance with the settings of `m_ellipseMode` in `onClick`:

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

The `draw` method draws the ellipse with a solid brush if it is filled, and with a hollow brush if it is unfilled:

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

If the ellipse is marked, the four squares covering the top, left, right, and bottom points of the ellipse are also drawn:

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

# The DrawingWindow class

The `DrawingWindow` class is similar to the version of the previous chapter. It overrides the `closeEvent` method.

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

The constructor initializes the window size to *1000* x *500* pixels, puts the drawing widget in the middle of the window, adds the standard File and Edit menus, and adds the application-specific Format and Figure menus:

```cpp
DrawingWindow::DrawingWindow(QWidget *parentWidget /*= nullptr*/)
 :MainWindow(parentWidget) {
  resize(1000, 500);

  m_drawingWidgetPtr = new DrawingWidget(this);
  setCentralWidget(m_drawingWidgetPtr);
  addFileMenu();
  addEditMenu();
```

The Format menu holds the `Color`, `Fill`, and `Modify` items as well as the Figure submenu:

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

The user selects the Modify item when they want to mark or modify existing figures instead of adding new figures:

```cpp
     m_figureGroupPtr = new QActionGroup(this); 
     addAction(formatMenuPtr, tr("&Modify"), 
              SLOT(onModify()), 
              QKeySequence(Qt::CTRL + Qt::Key_M), 
              QString(), nullptr, tr("Modify Figure"), nullptr, 
              LISTENER(isModifyChecked), m_figureGroupPtr); 
```

The Figure menu is a submenu holding the `Line`, `Rectangle`, and `Ellipse` items. It becomes a submenu when we add it to the Format menu:

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

# The DrawingWidget class

The `DrawingWidget` class is the main class of the application. It catches the mouse and paint events. It also catches the menu item selections of the File, Edit, and Figure menus.

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

Unlike the version of [Chapter 5](411aae8c-9215-4315-8a2e-882bf028834c.xhtml), *Qt Graphical Applications*, this version overrides the cut and copy event methods:

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

The `m_applicationMode` field holds the values `Idle`, `ModifySingle`, or `ModifyRectangle`. The `Idle` mode is active when the user is not pressing the mouse. The `ModifySingle` mode becomes active when the user grabs a figure and modifies or moves it (depending on which part of the figure the user grabs). Finally, the `ModifyRectangle` mode becomes active when the user clicks at the window without hitting a figure. In that case, a rectangle is shown, and every figure enclosed by the rectangle becomes marked when the user releases the mouse button. The user can delete or cut and paste the marked figure, or change their color or the filled status. When the user releases the mouse button, the `Application` mode again becomes `Idle`:

```cpp
  private: 
    enum ApplicationMode {Idle, ModifySingle, ModifyRectangle}; 
    ApplicationMode m_applicationMode = Idle; 
    void setApplicationMode(ApplicationMode mode);
```

The `m_actionMode` field holds the values `Modify` or `Add`. In `Modify` mode, when the user clicks with the mouse, `m_applicationMode` is set to `ModifySingle` or `ModifyRectangle`, depending on whether they hit a figure. In `Add` mode, a new figure is added, regardless of whether the user hits a figure. The kind of figure to be added is set by `m_addFigureId`, which holds the values `LineId`, `RectangleId`, or `EllipseId`:

```cpp
    enum ActionMode {Modify, Add}; 
    ActionMode m_actionMode = Add; 
    FigureId m_addFigureId = LineId; 
```

The color of the next figure to be added to the drawing is initialized to black, and the filled status is initialized to false (unfilled). In both cases, it can later be changed by the user:

```cpp
    QColor m_nextColor = Qt::black; 
    bool m_nextFilled = false; 
```

We need to save the latest mouse point in order to calculate distances between mouse movements:

```cpp
    QPoint m_mousePoint; 
```

Pointers to the figures of the drawing are stored in `m_figurePtrList`. The top-most figure is stored at the end of the list. When the user cuts or copies one or several figures, the figures are copied and the pointers to the copies are stored in `m_copyPtrList`:

```cpp
    QList<Figure*> m_figurePtrList, m_copyPtrList; 
```

When `m_actionMode` holds `Modify` and the user presses the mouse button without hitting a figure, a rectangle becomes visible in the window. That rectangle is stored in `m_insideRectangle`:

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

The constructor calls the constructor of the base class `DocumentWidget` to set the header of the window to `Drawing Advanced`, and to set the file suffix of the drawing files to `drw`:

```cpp
    DrawingWidget::DrawingWidget(QWidget* parentWidget) 
    :DocumentWidget(tr("Drawing Advanced"), 
                 tr("Drawing files (*.drw)"), 
                 parentWidget) { 
       // Empty. 
    }  
```

The destructor does nothing, it has been included for the sake of completeness only:

```cpp
    DrawingWidget::~DrawingWidget() { 
       // Empty. 
    } 
```

The `setApplicationMode` method sets the application mode and calls `onMenuShow` in the main window for the toolbar icons to be correctly enabled:

```cpp
    void DrawingWidget::setApplicationMode(ApplicationMode mode) { 
      m_applicationMode = mode; 
      ((MainWindow*) parent())->onMenuShow(); 
} 
```

The `newDocument` method is called when the user selects the `New` menu item. We start by deallocating every figure in the figure and copy pointer lists, and they clear the list themselves:

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

The current color and filled status are set to black and false (unfilled). The action mode is set to `Add` and the add figure identity is set to `LineId`, which means that when the user presses the mouse button a black line is added to the drawing:

```cpp
      m_nextColor = Qt::black; 
      m_nextFilled = false; 
      m_actionMode = Add; 
      m_addFigureId = LineId; 
    } 
```

The `writeFile` method is called when the user selects the `Save` or `Save As` menu items:

```cpp
    bool DrawingWidget::writeFile(const QString& filePath) { 
      ofstream outStream(filePath.toStdString()); 
```

If the file was successfully opened, we start by writing the next color and filled status:

```cpp
  if (outStream) { 
    writeColor(outStream, m_nextColor); 
    outStream.write((char*) &m_nextFilled, sizeof m_nextFilled); 
```

We then write the number of figures in the drawing, and then we write the figures themselves:

```cpp
      int size = m_figurePtrList.size(); 
      outStream.write((char*) &size, sizeof size); 
```

For each figure, first we write its identity value, we then write the figure itself by calling `write` on its pointer. Note that we do not know which class the figure pointer points at. We do not need to know that, since `write` is a pure virtual method in the base class `Figure`:

```cpp
    for (Figure* figurePtr : m_figurePtrList) { 
      FigureId figureId = figurePtr->getId(); 
      outStream.write((char*) &figureId, sizeof figureId); 
      figurePtr->write(outStream); 
    } 
```

We return the output stream converted to `bool`, which is true if the writing was successful:

```cpp
    return ((bool) outStream); 
  } 
```

If the file was not successfully opened, we return `false`:

```cpp
   return false; 
  } 
```

The `readFile` method is called when the user selects the Open menu item. We read the parts of the file in the same order as we wrote them in the preceding `writeFile`:

```cpp
    bool DrawingWidget::readFile(const QString& filePath) { 
      ifstream inStream(filePath.toStdString()); 
```

If the file was successfully opened, we start by reading the next color and filled status:

```cpp
    if (inStream) { 
      readColor(inStream, m_nextColor); 
      inStream.read((char*) &m_nextFilled, sizeof m_nextFilled);
```

We then write the number of figures in the drawing, and then we write the figures themselves:

```cpp
    int size; 
    inStream.read((char*) &size, sizeof size); 
```

For each figure, first we read its identity value, we then create a figure of the class indicated by the identity value by calling `createFigure`. Finally, we read the figure itself by calling `write` on its pointer:

```cpp
    for (int count = 0; count < size; ++count) { 
      FigureId figureId = (FigureId) 0; 
      inStream.read((char*) &figureId, sizeof figureId); 
      Figure* figurePtr = createFigure(figureId); 
      figurePtr->read(inStream); 
      m_figurePtrList.push_back(figurePtr); 
    } 
```

We return the input stream converted to `bool`, which is true if the reading was successful:

```cpp
     return ((bool) inStream); 
   } 
```

If the file was not successfully opened, we return `false`:

```cpp
  return false; 
} 
```

The `createFigure` method dynamically creates an object of the `Line`, `RectangleX`, or `EllipseX` class, depending on the value of the `figureId` parameter:

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

The `isCopyEnable` method is called before the Edit menu becomes visible in order to enable the Copy item. It is also called by the framework in order to enable the Copy toolbar icon. It returns `true` if at least one figure is marked, and by then it is ready to be copied. If it returns `true`, the Copy item and toolbar icon become enabled:

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

The `onCopy` method is called when the user selects the Copy menu item. To start with, it deallocates every figure in the copy pointer list and clears the list itself:

```cpp
    void DrawingWidget::onCopy(void) { 
      for (Figure* copyPtr : m_copyPtrList) { 
         delete copyPtr; 
      } 

      m_copyPtrList.clear(); 
```

Then, we iterate through the figure pointer list and add the pointer to a copy of each marked figure to the copy pointer list. We call `clone` on each figure pointer to provide us with the copy:

```cpp
   for (Figure* figurePtr : m_figurePtrList) { 
     if (figurePtr->marked()) { 
       m_copyPtrList.push_back(figurePtr->clone()); 
     } 
   } 
} 
```

The `isPasteEnabled` method is called before the Edit menu becomes visible to enable the Paste item. It is also called by the framework to enable the paste toolbar icon. If the copy pointer list is not empty, it returns `true`, and thereby enables the Paste item and image. That is, it returns `true` if there are figures ready to be pasted:

```cpp
   bool DrawingWidget::isPasteEnabled() { 
      return !m_copyPtrList.isEmpty(); 
   } 
```

The `onPaste` method is called when the user selects the Paste item in the Edit menu, or when they select the paste image in the edit toolbar. We iterate through the copy pointer list and add a copy (which we obtain by calling `clone`) of the figure to the figure pointer list, after we have moved it 10 pixels downwards and to the right:

```cpp
    void DrawingWidget::onPaste(void) { 
      for (Figure* copyPtr : m_copyPtrList) { 
        Figure* pastePtr = copyPtr->clone(); 
        pastePtr->move(QSize(10, 10)); 
        m_figurePtrList.push_back(pastePtr); 
      } 
```

Finally, when the figures have been added to the list, we force an eventual call to the `paintEvent` by calling `update`:

```cpp
     update(); 
  } 
```

The `onDelete` method is called every time the user selects the Delete menu item or toolbar icon. We iterate through the figure pointer list and remove every marked figure:

```cpp
    void DrawingWidget::onDelete(void) { 
       for (Figure* figurePtr : m_figurePtrList) { 
         if (figurePtr->marked()) { 
         m_figurePtrList.removeOne(figurePtr); 
         delete figurePtr; 
       } 
     } 
```

Also, in this case, we force an eventual call to `paintEvent` by calling the `update` method, after the figures have been deleted:

```cpp
     update(); 
  } 
```

The `onColor` method is called every time the user selects the `Color` item in the Format menu. We start by obtaining the new color by calling the static method `getColor` in the Qt `QColorDialog` class:

```cpp
    void DrawingWidget::onColor(void) { 
      QColor newColor = QColorDialog::getColor(m_nextColor, this); 
```

If the color is valid, which it is if the user has closed the dialog by pressing the Ok button rather than the Cancel button, and if they have chosen a new color, we set the next color to the new color and set the modified flag. We also iterate through the figure pointer list and, for each marked figure, set the color of the figure:

```cpp
    if (newColor.isValid() && (m_nextColor != newColor)) { 
      m_nextColor = newColor; 
      setModifiedFlag(true); 

      for (Figure* figurePtr : m_figurePtrList) { 
        if (figurePtr->marked()) { 
          figurePtr->color() = m_nextColor; 
```

If at least one figure is marked, we force an eventual call to `paintEvent` by calling update:

```cpp
          update(); 
         } 
       } 
     } 
   } 
```

The `isFillEnabled` method is called before the `Fill` item in the Format menu becomes visible:

```cpp
    bool DrawingWidget::isFillEnabled(void) { 
      switch (m_actionMode) { 
```

In `Modify` mode, we iterate through the figure pointer list. If at least one rectangle or ellipse is marked, we return `true` and the item becomes enabled:

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

If no rectangle or ellipse is marked, we return `false` and the item becomes disabled:

```cpp
      return false; 
```

In the `Add` mode, we return `true` if the next figure to be added by the user is a rectangle or an ellipse:

```cpp
    case Add: 
      return (m_addFigureId == RectangleId) || 
             (m_addFigureId == EllipseId); 
    } 
```

We are not supposed to reach this point. The `assert` macro call is for debugging purposes only. However, we still must return a value at the end of the method:

```cpp
    assert(false); 
    return true; 
   } 
```

The `onFill` method is called when the user selects the `Fill` item in the Format menu:

```cpp
    void DrawingWidget::onFill(void) { 
      switch (m_actionMode) { 
```

In the `Modify` mode, we iterate through the figure pointer list and invert the filled status of all marked figures. If at least one figure changes, we force an eventual call to `paintEvent` by calling `update`:

```cpp
    case Modify: 
      for (Figure* figurePtr : m_figurePtrList) { 
        if (figurePtr->marked()) { 
          figurePtr->filled() = !figurePtr->filled(); 
          update(); 
        } 
      } 
```

We also invert the filled status of the next figure to be added:

```cpp
      m_nextFilled = !m_nextFilled; 
      break; 
```

In the `Add` mode, we invert the filled status of the next figure to be added by the user:

```cpp
    case Add: 
       m_nextFilled = !m_nextFilled; 
       break; 
    } 
  } 
```

The `isModifyChecked` method is called before the `Modify` item in the Format menu becomes visible. In `Modify` mode, it returns `true` and enables the item:

```cpp
    bool DrawingWidget::isModifyChecked(void) { 
      return (m_actionMode == Modify); 
    } 
```

The `onModify` method is called when the user selects the `Modify` item in the Format menu. It sets the action mode to `Modify`:

```cpp
    void DrawingWidget::onModify(void) { 
      m_actionMode = Modify; 
    } 
```

The `isLineChecked` method is called before the `Line` item in the `Add` submenu becomes visible. It returns `true`, and the item becomes checked (with a radio button, since the item belongs to a group) in case of add action mode, and the next figure to be added is a line:

```cpp
    bool DrawingWidget::isLineChecked(void) { 
      return (m_actionMode == Add) && (m_addFigureId == LineId); 
    } 
```

The `onLine` method is called when the user selects the `Line` item in the `Add` submenu. It set the action mode to `Add` and the next figure to be added by the user to a line:

```cpp
    void DrawingWidget::onLine(void) { 
      m_actionMode = Add; 
      m_addFigureId = LineId; 
    } 
```

The `isRectangleChecked` method is called before the `Rectangle` item in the `Add` submenu becomes visible. It returns `true` in case of `Add` action mode and if the next figure to be added is a rectangle:

```cpp
    bool DrawingWidget::isRectangleChecked(void) { 
      return (m_actionMode == Add) && (m_addFigureId == RectangleId); 
    } 
```

The `onRectangle` method is called when the user selects the `Rectangle` item. It sets the action mode to `Add` and the next figure to be added by the user to a rectangle:

```cpp
    void DrawingWidget::onRectangle(void) { 
      m_actionMode = Add; 
      m_addFigureId = RectangleId; 
    } 
```

The `isEllipseChecked` method is called before the `Ellipse` item in the `Add` submenu becomes visible. It returns `true` in case of `Add` action mode and if the next figure to be added is an ellipse:

```cpp
    bool DrawingWidget::isEllipseEnabled(void) { 
      return !isEllipseChecked(); 
    } 
```

The `onEllipse` method is called when the user selects the `Ellipse` item. It sets the action mode to `Add` and the next figure to be added by the user to an ellipse:

```cpp
    void DrawingWidget::onEllipse(void) { 
      m_actionMode = Add; 
      m_addFigureId = EllipseId; 
    }  
```

The `mousePressEvent` method is called when the user presses one of the mouse buttons. We store the mouse point in `m_mousePoint`, to be used in `mouseMoveEvent` as follows:

```cpp
    void DrawingWidget::mousePressEvent(QMouseEvent* eventPtr) { 
       if (eventPtr->buttons() == Qt::LeftButton) { 
       m_mousePoint = eventPtr->pos(); 
```

In case of `Modify` mode, we first iterate through the figure pointer list and unmark every figure:

```cpp
    switch (m_actionMode) { 
      case Modify: { 
          for (Figure* figurePtr : m_figurePtrList) { 
           figurePtr->marked() = false; 
          } 
```

We then iterate through the list again, to find if the user has hit a figure. Since the top-most figure is placed at the end of the list, we need to iterate through the list backward. We do so by using the `reverse_iterator` type of the Qt `QList` class:

```cpp
     m_clickedFigurePtr = nullptr; 
     for (QList<Figure*>::reverse_iterator iterator = 
         m_figurePtrList.rbegin(); 
     iterator != m_figurePtrList.rend(); ++iterator) { 
        Figure* figurePtr = *iterator; 
```

If we found out (by calling `isClick` on the figure) that a figure has been hit by the user's mouse click, we set the application mode to `ModifySingle` and mark the figure. We also remove it from the list and add it to the end of the list, to make it appear top-most in the drawing. Finally, we break the loop since we have found a figure:

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

If we have not found a figure, we set the application mode to `ModifyRectangle` and initialize the top-most and bottom-right corners of the enclosing rectangle to the mouse point:

```cpp
    if (m_clickedFigurePtr == nullptr) { 
      setApplicationMode(ModifyRectangle); 
      m_insideRectangle = QRect(m_mousePoint, m_mousePoint); 
    } 
    } 
    break; 
```

In case of `Add` action mode, we create a new figure by calling `createFigure` with the identity of the next figure to be added by the user as a parameter. We then set the color, filled status of the new figure, and initialize its endpoints:

```cpp
      case Add: { 
          Figure* newFigurePtr = createFigure(m_addFigureId); 
          newFigurePtr->color() = m_nextColor; 
          newFigurePtr->filled() = m_nextFilled; 
          newFigurePtr->initializePoints(m_mousePoint); 
```

When the new figure has been created and initialized, we add it at the end of the figure pointer list and set the application mode to `ModifySingle`, since the `mouseMoveEvent` method will continue to modify the last figure in the list, just as if the user had hit a figure in the `Modify` mode. We also set the modified flag since we have added a figure to the drawing:

```cpp
      m_figurePtrList.push_back(newFigurePtr); 
      setApplicationMode(ModifySingle); 
      setModifiedFlag(true); 
      } 
      break; 
    } 
```

Finally, we force an eventual call to `paintEvent` by calling `update`:

```cpp
        update(); 
      } 
    } 
```

The `mouseMoveEvent` method is called when the user moves the mouse. If they also press the left mouse button, we save the mouse point to future calls to `mouseMoveEvent` and calculate the distance since the last call to `mousePressEvent` or `mouseMoveEvent`:

```cpp
    void DrawingWidget::mouseMoveEvent(QMouseEvent* eventPtr) { 
      if (eventPtr->buttons() == Qt::LeftButton) { 
        QPoint newMousePoint = eventPtr->pos(); 
        QSize distance(newMousePoint.x() - m_mousePoint.x(), 
                       newMousePoint.y() - m_mousePoint.y()); 
        m_mousePoint = newMousePoint; 
```

In the `Modify` mode, we modify the current figure (the figure placed at the end of the figure pointer list) by calling `modify`. Remember that the figure can be either modified or moved, depending on the settings in the call to `isClick` in `onMousePress` previously. We also set the modified flag since the figure has been altered:

```cpp
    switch (m_applicationMode) { 
      case ModifySingle: 
        m_figurePtrList.back()->modify(distance); 
        setModifiedFlag(true); 
        break; 
```

In case of the enclosing rectangle, we just update its bottom-right corner. Note that we do not set the modified flag since no figure has yet been altered:

```cpp
    case ModifyRectangle: 
      m_insideRectangle.setBottomRight(m_mousePoint); 
      break; 
    } 
```

Finally, we force an eventual call to `paintEvent` by calling `update`:

```cpp
        update(); 
      } 
   } 
```

The `mouseReleaseEvent` method is called when the user releases a mouse button. If it is the left mouse button, we check the application mode. The only mode we actually are interested in is the enclosing rectangle mode:

```cpp
     void DrawingWidget::mouseReleaseEvent(QMouseEvent* eventPtr) { 
       if (eventPtr->buttons() == Qt::LeftButton) { 
         switch (m_applicationMode) { 
           case ModifyRectangle: { 
             QList<Figure*> insidePtrList; 
```

We iterate through the figure pointer list and call `isInside` on each figure. Each figure that is completely enclosed by the rectangle becomes marked, removed from the list, and added to `insidePtrList` to be later added at the end of the figure pointer list:

```cpp
    for (Figure* figurePtr : m_figurePtrList) { 
      if (figurePtr->isInside(m_insideRectangle)) { 
        figurePtr->marked() = true; 
        m_figurePtrList.removeOne(figurePtr); 
        insidePtrList.push_back(figurePtr); 
      } 
    } 
```

Each figure which is completely enclosed by the rectangle is removed from the figure pointer list:

```cpp
    for (Figure* figurePtr : insidePtrList) { 
      m_figurePtrList.removeOne(figurePtr); 
    } 
```

Finally, all enclosed figures are added at the end of the list in order to appear top-most in the drawing:

```cpp
    m_figurePtrList.append(insidePtrList); 
    } 
    break; 
    } 
```

When the user has released the mouse button, the application mode is set to idle, and we force an eventual call to `paintEvent` by calling `update`:

```cpp
       setApplicationMode(Idle); 
       update(); 
      } 
    } 
```

The `mouseDoubleClick` method is called when the user double-clicks one of the buttons. However, `mouseClickEvent` is always called before `mouseDoubleClickEvent`. If the preceding call to `mouseClickEvent` has made `m_clickedFigurePtr` point at the clicked figure, we call `doubleClick` on that figure. This may cause some change in the figure, depending on which kind of figure it is:

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

Finally, `paintEvent` is called when the content of the window needs to be repainted. Before the call, the framework clears the window:

```cpp
    void DrawingWidget::paintEvent(QPaintEvent* /* 
       eventPtr */) { 
     QPainter painter(this); 
     painter.setRenderHint(QPainter::Antialiasing); 
     painter.setRenderHint(QPainter::TextAntialiasing); 
```

We iterate through the figure pointer list and draw every figure. The last figure in the list is placed at the end of the list, to appear at the top of the drawing:

```cpp
     for (Figure* figurePtr : m_figurePtrList) { 
       figurePtr->draw(painter); 
     } 
```

In case of enclosing rectangle mode, we draw a hollow rectangle with a light-gray border:

```cpp
    if (m_applicationMode == ModifyRectangle) { 
      painter.setPen(Qt::lightGray); 
      painter.setBrush(Qt::NoBrush); 
      painter.drawRect(m_insideRectangle); 
    } 
  } 
```

# The main function

The `main` function is similar to the `main` function of the previous applications—it creates an application, shows the drawing window, and starts the execution of the application.

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

The output is shown in the following screenshot:

![](img/e32d6956-fb1d-4513-bb7f-b7b7529a3466.png)

# Improving the editor

The editor of this chapter is a more advanced version of the editor of [Chapter 5](411aae8c-9215-4315-8a2e-882bf028834c.xhtml), *Qt Graphical Applications*. In this version, it is possible to change the font and alignment of the text, to mark text, and to cut and paste text.

# The EditorWindow class

The `EditorWindow` class of this chapter is similar to the class of [Chapter 5](https://cdp.packtpub.com/c___by_example/wp-admin/post.php?post=72&action=edit#post_67), *Qt Graphical Applications*. It catches the key pressing event and the window closing event.

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

The constructor initializes the editor window. It sets the size of the window to *1000 x 500* pixels. It also dynamically creates an editor widget and adds the standard File and Edit menus:

```cpp
EditorWindow::EditorWindow(QWidget *parentWidgetPtr /*= nullptr*/)
 :MainWindow(parentWidgetPtr) {
  resize(1000, 500);

  m_editorWidgetPtr = new EditorWidget(this);
  setCentralWidget(m_editorWidgetPtr);
  addFileMenu();
  addEditMenu();
```

The Figure menu is different, compared to [Chapter 5](411aae8c-9215-4315-8a2e-882bf028834c.xhtml), *Qt Graphical Applications*. We add the item `Font` and the submenu Alignment, to which, in turn, we add the three items: left, center, and right:

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

We also add a toolbar for the `Alignment` menu:

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

The key pressing event and the window closing event are passed on to the editor widget:

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

# The EditorWidget class

The `EditorWidget` class is similar to the version of [Chapter 5](411aae8c-9215-4315-8a2e-882bf028834c.xhtml), *Qt Graphical Applications*. However, methods and listeners to handle the font and alignment have been added.

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

The `mousePresseEvent`, `mouseMoveEvent`, and `mouseReleaseEvent` are called when the user presses a mouse button, moves the mouse, and releases the mouse button:

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

The `newDocument` method is called when the user selects the New menu item, `writeFile` is called when they select Save or Save As, and `readFile` is called when they select the Open menu item:

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

The `isLeftChecked`, `isCenterChecked`, and `isRightChecked` methods are called before the `Alignment` submenu becomes visible. They then annotate a radio button to the selected alignment:

```cpp
    DEFINE_LISTENER(EditorWidget, isLeftChecked); 
    DEFINE_LISTENER(EditorWidget, isCenterChecked); 
    DEFINE_LISTENER(EditorWidget, isRightChecked); 
```

The `onLeft`, `onCenter`, and `onRight` methods are called when the user selects one of the items of the Alignment submenu:

```cpp
      void onLeft(void); 
      void onCenter(void); 
      void onRight(void); 

      private: 
       void setCaret(); 
       void simulateMouseClick(int x, int y); 
```

In this version of the editor, we have two modes—edit and mark. The edit mark is active when the user inputs text or moves the caret with the arrow key, while the mark mode is active when the user has marked a block of the code with the mouse. The caret is visible in edit mode, but not in mark mode:

```cpp
    private: 
      enum Mode {Edit, Mark} m_mode; 
```

The text can be aligned in the left, center, and right direction:

```cpp
     enum Alignment {Left, Center, Right} m_alignment; 
```

In edit mode, `m_editIndex` holds the index to place the next character to be input by the user, which also is the position of the caret. In mark mode, `m_firstIndex` and `m_lastIndex` hold the indexes of the first and last marked character:

```cpp
    int m_editIndex, m_firstIndex, m_lastIndex; 
```

The `m_caret` object holds the caret of the editor. The caret is visible in edit mode, but not in mark mode:

```cpp
    Caret m_caret; 
```

The `m_editorText` field holds the text of the editor, and `m_copyText` holds the text which is cut or pasted by the user:

```cpp
     QString m_editorText, m_copyText; 
```

The text of the editor is divided into lines; the index of the first and last character of each line is stored in `m_lineList`:

```cpp
    QList<QPair<int,int>> m_lineList; 
```

The current font of the text is stored in `m_textFont`. The height in pixels of a character of the current font is stored in `m_fontHeight`:

```cpp
    QFont m_textFont; 
    int m_fontHeight; 
```

The `mousePressEvent` and `mouseMoveEvent` methods store the last mouse point in order to calculate the distance between mouse events:

```cpp
    Qt::MouseButton m_button; 
```

Similar to the method of [Chapter 5](411aae8c-9215-4315-8a2e-882bf028834c.xhtml), *Qt Graphical Applications*, `calculate` is an auxiliary method that calculates the enclosing rectangle of each character of the text. However, the version of this chapter is more complicated since it has to take into consideration whether the text is left, center, or right-aligned:

```cpp
    void calculate(); 
```

The enclosing rectangles are stored in `m_rectList`, and then used by the caret and `paintEvent`:

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

The constructor sets the window header to `Editor Advanced` and the file suffix to `edi`:

```cpp
    EditorWidget::EditorWidget(QWidget* parentWidgetPtr) 
     :DocumentWidget(tr("Editor Advanced"), 
         tr("Editor files (*.edi)"), parentWidgetPtr), 
```

The text font is initialized to `12` point `Times New Roman`. The application mode is set to edit, the index of the next character to be input by the user is set to zero, and the text is left- aligned from the beginning:

```cpp
      m_textFont(tr("Times New Roman"), 12), 
        m_mode(Edit), 
        m_editIndex(0), 
        m_alignment(Left), 
        m_caret(this) { 
```

The rectangles enclosing the characters are calculated by `calculate`, the caret is initialized and shown since the application holds edit mode from the beginning:

```cpp
     calculate(); 
     setCaret(); 
     m_caret.show(); 
   } 
```

The `newDocument` method is called when the user selects the New menu item. We start by setting the application mode to edit and the edit index to zero. The text font is set to `12` point Times New Roman. The text of the editor is cleared, the rectangles enclosing the characters are calculated by `calculate`, and the caret is set:

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

The `writeFile` method is called when the user selects the Save or Save As menu items. The file format is quite simple: we write the font on the first line, and then the text of the editor on the following lines:

```cpp
    bool EditorWidget::writeFile(const QString& filePath) { 
      QFile file(filePath); 
      if (file.open(QIODevice::WriteOnly | QIODevice::Text)) { 
      QTextStream outStream(&file); 
      outStream << m_textFont.toString() << endl << m_editorText; 
```

We use the `Ok` field of the input stream to decide if the writing was successful:

```cpp
     return ((bool) outStream.Ok); 
   } 
```

If we could not open the file for writing, we return `false`:

```cpp
     return false; 
   } 
```

The `readFile` method is called when the user selects the Open menu items. Similar to `writeFile` previously, we read the first line and initialize the text font with the text. We then read the editor text:

```cpp
    bool EditorWidget::readFile(const QString& filePath) { 
      QFile file(filePath); 

      if (file.open(QIODevice::ReadOnly | QIODevice::Text)) { 
        QTextStream inStream(&file); 
        m_textFont.fromString(inStream.readLine()); 
        m_editorText = inStream.readAll(); 
```

When the text is read, we call `calculate` to calculate the rectangles enclosing the characters of the text. We then set the caret and return `true`, since the reading was successful:

```cpp
      calculate(); 
      setCaret(); 
```

We use the `Ok` field of the input stream to decide if the reading was successful:

```cpp
     return ((bool) inStream.Ok); 
    } 
```

If we could not open the file for reading, we `return false`:

```cpp
    return false; 
   } 
```

The `isCopyEnabled` method is called before the Edit menu becomes visible. It is also called by the framework to decide whether the copy toolbar icon shall be enabled. It returns true (and the item becomes enabled) if the application holds mark mode, which means that the user has marked a part of the text, which can be copied:

```cpp
    bool EditorWidget::isCopyEnabled() { 
      return (m_mode == Mark); 
    } 
```

The `onCopy` method is called when the user selects the Copy item. We copy the marked text into `m_EditorText`:

```cpp
    void EditorWidget::onCopy(void) { 
      int minIndex = qMin(m_firstIndex, m_lastIndex), 
      maxIndex = qMax(m_firstIndex, m_lastIndex); 

      m_copyText = 
        m_editorText.mid(minIndex, maxIndex - minIndex + 1); 
    } 
```

The `isPasteEnabled` method is also called before the Edit menu becomes visible. It returns `true` (and the item becomes visible) if the copy text is not empty. That is, if there is a block of text that has been copied and is ready to be pasted:

```cpp
    bool EditorWidget::isPasteEnabled() { 
      return !m_copyText.isEmpty(); 
    } 
```

The `onPaste` method is called when the user selects the Paste menu item. In mark mode, we call `onDelete`, which causes the marked text to be deleted:

```cpp
    void EditorWidget::onPaste(void) { 
      if (m_mode == Mark) { 
         onDelete(); 
      } 
```

We then insert the copied text into the editor text. We also update `m_editIndex`, since the edit index after the text has been copied shall be the position after the inserted text:

```cpp
     m_editorText.insert(m_editIndex, m_copyText); 
     m_editIndex += m_copyText.size(); 
```

Finally, we calculate the rectangles enclosing the characters of the text, set the caret to the new index, set the modified flag since the text has been altered, and call `update` to force an eventual call to `paintEvent` in order to display the new text:

```cpp
     calculate(); 
     setCaret(); 
     setModifiedFlag(true); 
     update(); 
     } 
```

The `onDelete` method is called when the user selects the Delete menu item or the Delete toolbar icon. The effect is similar to the event when the user presses the *Delete* key. Therefore, we prepare a keypress event with the *Delete* key, which we use as a parameter in the call to `keyPressEvent`.

Note that there is no `isDeleteEnabled` method because the user can always use the Delete item. In edit mode, the next character is deleted. In mark mode, the marked text is deleted:

```cpp
    void EditorWidget::onDelete(void) { 
      QKeyEvent event(QEvent::KeyPress, Qt::Key_Delete, 
                  Qt::NoModifier); 
      keyPressEvent(&event); 
    } 
```

`isCopyEnabled` is called before the Format menu becomes visible. It returns `true` in edit mode, since it would be illogical to change the font on all characters when a subset of them is marked:

```cpp
    bool EditorWidget::isFontEnabled() { 
      return (m_mode == Edit); 
    } 
```

The `onFont` method is called when the user selects the `Font` menu item. We let the user select the new font with the Qt `QFontDialog` class:

```cpp
     void EditorWidget::onFont(void) { 
       bool pressedOkButton; 
       QFont newFont = 
         QFontDialog::getFont(&pressedOkButton, m_textFont, this); 
```

If the user closes the dialog by pressing the Ok button, we set the font of the editor (`m_textFont`) field and the modified flag:

```cpp
      if (pressedOkButton) { 
        m_textFont = newFont; 
        setModifiedFlag(true); 
```

We calculate the newly enclosed rectangles by calling `calculate`, set the caret, and force an eventual call to `paintEvent` by calling `update`:

```cpp
      calculate(); 
      m_caret.set(m_rectList[m_editIndex]); 
      update(); 
     } 
   } 
```

The `isLeftChecked`, `isCenterChecked`, and `isRightChecked` methods are called before the alignment submenu becomes visible. They return `true` to the current alignment:

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

The `onLeft`, `onCenter`, and `onRight` methods are called when the user selects the `Left`, `Center`, and `Right` menu item. They set the alignment and the modified flag.

They also calculate the new enclosing rectangles, set the caret, and force an eventual call to `paintEvent` by calling `update`:

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

The `mousePressEvent` method is called when the user presses one of the mouse buttons. We call `mouseToIndex` to find the character index the user clicked on. For the time being, both the first and last mark index is set to the mouse index. The last index may later be changed by a call to `mouseMoveEvent` in the following snippet. Finally, the mode is set to mark, and the caret is hidden:

```cpp
    void EditorWidget::mousePressEvent(QMouseEvent* eventPtr) { 
      if (eventPtr->buttons() == Qt::LeftButton) { 
         m_firstIndex = m_lastIndex = mouseToIndex(eventPtr->pos()); 
         m_mode = Mark; 
         m_caret.hide(); 
       } 
    } 
```

The `mouseMoveEvent` method is called when the user moves the mouse. We set the last mark index to the mouse index and force an eventual call to `paintEvent` by calling `update`:

```cpp
    void EditorWidget::mouseMoveEvent(QMouseEvent* eventPtr) { 
      if (eventPtr->buttons() == Qt::LeftButton) { 
         m_lastIndex = mouseToIndex(eventPtr->pos()); 
         update(); 
      } 
    } 
```

The `mouseReleaseEvent` method is called when the user releases the mouse button. If the user has moved the mouse to the original start position of the mouse movement, there is nothing to mark and we set the application in edit mode. In that case, we set the edit index to the first mark index, and set and show the caret (since it shall be visible in edit mode). Finally, we force an eventual call to `paintEvent` by calling `update`:

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

`keyPressEvent` is called when the user presses a key on the keyboard. Depending on the application mode (edit or mark), we call `keyEditPressEvent` or the following `keyMarkPressEvent` to further process the key event:

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

`keyEditPressEvent` handles the key press in edit mode. First, we check if the key is an arrow key, page up or down, *Delete*, *Backspace*, or return key:

```cpp
    void EditorWidget::keyEditPressEvent(QKeyEvent* eventPtr) { 
      switch (eventPtr->key()) { 
```

In the case of the left-arrow key, we move the edit index one step backward, unless it is already at the beginning of the text:

```cpp
     case Qt::Key_Left: 
       if (m_editIndex > 0) { 
          --m_editIndex; 
       } 
       break; 
```

In the case of the right-arrow key, we mode the edit index one step forward, unless it is already at the end of the text:

```cpp
    case Qt::Key_Right: 
      if (m_editIndex < m_editorText.size()) { 
        ++m_editIndex; 
      } 
      break; 
```

In the case of the up-arrow key, we calculate the appropriate `x` and `y` position for the character on the previous line, unless it is already on top of the text. We then call `simulateMouseClick`, which has the same effect as if the user has clicked on the character above the line:

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

In the same way, in the case of the down-arrow key, we move the edit index one line downwards unless it is already at the bottom of the text.

We calculate the appropriate `x` and `y` position for the character on the line below and call `simulateMouseClick`, which has the same effect as if the user has clicked at the point:

```cpp
     case Qt::Key_Down: { 
       QRect charRect = m_rectList[m_editIndex]; 
       int x = charRect.left() + (charRect.width() / 2), 
           y = charRect.bottom() + 1; 
       simulateMouseClick(x, y); 
     } 
     break; 
```

In the case of the *Delete* key, we remove the current key, unless we are at the end of the text. That is, if we are one step beyond the last character:

```cpp
    case Qt::Key_Delete: 
      if (m_editIndex < m_editorText.size()) { 
        m_editorText.remove(m_editIndex, 1); 
        setModifiedFlag(true); 
      } 
      break; 
```

In the case of the backspace key, we move the edit index one step backward, unless it already is at the beginning of the text, and call `onDelete`. In this way, we remove the previous character and move the edit index one step backward:

```cpp
     case Qt::Key_Backspace: 
     if (m_editIndex > 0) { 
       --m_editIndex; 
       onDelete(); 
     } 
     break; 
```

In the case of the return key, we simply insert the new line character to the text:

```cpp
    case Qt::Key_Return: 
      m_editorText.insert(m_editIndex++, 'n'); 
      setModifiedFlag(true); 
      break; 
```

If the key is not a special key, we check whether it is a regular character by calling `text` on the key event pointer. If the text is not empty, add its first character to the text:

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

Finally, we calculate the enclosing rectangles, set the caret, and force an eventual call to `paintEvent` by calling `update`:

```cpp
    calculate(); 
    setCaret(); 
    update(); 
  } 
```

`keyMarkPressEvent` is called when the user presses a key in mark mode:

```cpp
    void EditorWidget::keyMarkPressEvent(QKeyEvent* eventPtr) { 
      switch (eventPtr->key()) { 
```

In case of the left-arrow key, we set the application to edit mode and the edit index to the minimum of the first and last marked index. However, if the minimum index is located at the beginning of the text, we do nothing:

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

On the other hand, in the case of the right-arrow key, we set the application to edit mode and the edit index to the maximum of the first and last marked index. However, if the maximum index is located at the end of the text, we do nothing:

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

In case of the up and down arrows, we simulate a mouse click one line above or below the current line, just as in the previous edit case:

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

In the mark mode, the delete and backspace keys perform the same task—they delete the marked text:

```cpp
    case Qt::Key_Delete: 
    case Qt::Key_Backspace: { 
        int minIndex = qMin(m_firstIndex, m_lastIndex), 
            maxIndex = qMax(m_firstIndex, m_lastIndex); 
```

We remove the marked text from the edit text, set the modified flag, set the application to edit mode, set the edit index to the minimum of the first and last marked index, and show the caret:

```cpp
        m_editorText.remove(minIndex, maxIndex - minIndex); 
        setModifiedFlag(true); 
        m_mode = Edit; 
        m_editIndex = minIndex; 
        m_caret.show(); 
      } 
      break; 
```

The return key case is similar to the previous edit mode case, with the difference that we first delete the marked text. We then add a new line to the editor text:

```cpp
     case Qt::Key_Return: 
       onDelete(); 
       m_editorText.insert(m_editIndex++, 'n'); 
       setModifiedFlag(true); 
       break; 
```

If the key is not a special key, we check if it is a regular key by calling `text` on the key event pointer. If the text is not empty, the user has printed a regular key, and we insert the first character in the editor text:

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

Finally, we calculate the new rectangles enclosing the characters, set the caret, and force an eventual call to `paintEvent` by calling `update`:

```cpp
     calculate(); 
     setCaret(); 
     update(); 
    } 
```

The `simulateMouseClick` method is called when the user moves the caret up or down. It simulates a mouse click by calling `mousePressEvent` and `mouseReleaseEvent`, with suitably prepared event objects:

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

The `setCaret` method sets the caret to the appropriate size and position in edit mode. Firstly, we use `m_editIndex` to find the rectangle of the correct character. We then create a new rectangle that is of only one-pixel width, in order for the caret to appear as a thin vertical line:

```cpp
    void EditorWidget::setCaret() { 
      QRect charRect = m_rectList[m_editIndex]; 
      QRect caretRect(charRect.left(), charRect.top(), 
                  1, charRect.height()); 
      m_caret.set(caretRect); 
    } 
```

The `mouseToIndex` method takes a mouse point and returns the index of the character at that point. Unlike the version of [Chapter 5](411aae8c-9215-4315-8a2e-882bf028834c.xhtml), *Qt Graphical Applications*, we need to take into consideration that the text may be center or right-aligned:

```cpp
    int EditorWidget::mouseToIndex(QPoint point) { 
       int x = point.x(), y = point.y(); 
```

If the mouse point is below the text of the editor, the index of the last character is returned:

```cpp
    if (y > (m_fontHeight * m_lineList.size())) { 
      return m_editorText.size(); 
    } 
```

Otherwise, we start by finding the line of the mouse point, and obtain the indexes of the first and last character on the line:

```cpp
    else { 
      int lineIndex = y / m_fontHeight; 
      QPair<int,int> lineInfo = m_lineList[lineIndex]; 
      int firstIndex = lineInfo.first, lastIndex = lineInfo.second; 
```

If the mouse point is located to the left of the first character on the line (which it may be if the text is center or right-aligned), we return the index of the first character of the line:

```cpp
     if (x < m_rectList[firstIndex].left()) { 
        return firstIndex; 
     } 
```

If the mouse point, on the other hand, is located to the right of the line, we return the index of the character next to the last character of the line:

```cpp
    else if (x >= m_rectList[lastIndex].right()) { 
      return (lastIndex + 1); 
    } 
```

Otherwise, we iterate through the character on the line and, for each character, we check whether the mouse point is located inside the character's enclosing rectangle:

```cpp
     else { 
       for (int charIndex = firstIndex + 1; 
           charIndex <= lastIndex; ++charIndex){ 
           int left = m_rectList[charIndex].left(); 
```

If the mouse point is located inside the rectangle, we check if it is closest to the left or right border of the rectangle. If it is closest to the left border, we return the index of the character. If it is closest to the right border, we instead return the index of the next character:

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

We are not supposed to reach this point. The `assert` macro is added for debugging purposes only:

```cpp
      assert(false); 
      return 0; 
   } 
```

The `resizeEvent` method is called when the user resizes the window. We calculate the rectangles enclosing the characters, since the width of the window may have changed, which may cause the lines to hold fewer or more characters:

```cpp
    void EditorWidget::resizeEvent(QResizeEvent* eventPtr) { 
      calculate(); 
      DocumentWidget::resizeEvent(eventPtr); 
    } 
```

The `calculate` method divides the text into lines, and calculates the rectangles enclosing every character of the text. The indexes of the first and last character of each line are stored in `m_lineList`, and the enclosing rectangles are stored in `m_rectList`:

```cpp
    void EditorWidget::calculate() { 
      m_lineList.clear(); 
      m_rectList.clear(); 
```

We use the Qt `QFontMetrics` class to obtain the height of a character of the editor font. The height is stored in `m_fontHeight`. The `width` method gives the width of the window content, in pixels:

```cpp
      QFontMetrics metrics(m_textFont); 
      m_fontHeight = metrics.height(); 
      QList<int> charWidthList, lineWidthList; 
      int windowWidth = width(); 
```

We start by iterating through the editor text in order to divide the text into lines:

```cpp
     { int firstIndex = 0, lineWidth = 0; 
        for (int charIndex = 0; charIndex < m_editorText.size(); 
           ++charIndex) { 
          QChar c = m_editorText[charIndex]; 
```

When we encounter a new line, we add the first and last index of the current line to `m_lineList`:

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

Otherwise, we call the `width` method of the Qt `QMetrics` object to obtain the width of the character, in pixels:

```cpp
      else { 
        int charWidth = metrics.width(c); 
        charWidthList.push_back(charWidth); 
```

If the character makes the width of the line exceed the width of the window content, we add the first and last index to `m_lineList` and start a new line.

However, we have two different cases to consider. If the current character is the first character of the line, we have the (rather unlikely) situation that the width of that character exceeds the width of the window content. In that case, we add the index of that character as both the first and last index to `m_lineList`. The first index of the next line is the character next to that character:

```cpp
    if ((lineWidth + charWidth) > windowWidth) { 
       if (firstIndex == charIndex) { 
         lineWidthList.push_back(windowWidth); 
         m_lineList.push_back 
              (QPair<int,int>(firstIndex, charIndex)); 
         firstIndex = charIndex + 1; 
       } 
```

If the current character is not the first character of the line, we add the indexes of the first character and the character preceding the current character to `m_lineList`. The index of the next line becomes the index of the current character:

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

If the character does not make the width of the line exceed the width of the window content, we simply add the width of the character to the width of the line:

```cpp
    else { 
          lineWidth += charWidth; 
         } 
    } 
  } 
```

Finally, we need to add the last line to `m_lineList`:

```cpp
      m_lineList.push_back(QPair<int,int>(firstIndex, 
           m_editorText.size() - 1)); 
      lineWidthList.push_back(lineWidth); 
    } 
```

When we have divided the text into lines, we continue to calculate the enclosing rectangles of the individual characters. We start by setting `top` to zero, since it holds the top position of the line. It will be increased by the line height for each line:

```cpp
     { int top = 0, left; 
        for (int lineIndex = 0; lineIndex < m_lineList.size(); 
           ++lineIndex) { 
        QPair<int,int> lineInfo = m_lineList[lineIndex]; 
        int lineWidth = lineWidthList[lineIndex]; 
        int firstIndex = lineInfo.first, 
           lastIndex = lineInfo.second; 
```

Depending on the alignment of the text, we need to decide where the line starts. In the case of left alignment, we set the left position of the line to zero:

```cpp
      switch (m_alignment) { 
        case Left: 
          left = 0; 
          break; 
```

In case of center alignment, we set the left position to half of the difference between the width of the window content and the line. In this way, the line will appear at the center of the window:

```cpp
        case Center: 
          left = (windowWidth - lineWidth) / 2; 
          break; 
```

In case of right alignment, we set the left position to the difference between the width of the window content and the line. In this way, the line will appear to the right in the window:

```cpp
       case Right: 
          left = windowWidth - lineWidth; 
          break; 
       } 
```

Finally, when we have decided the starting left position of the line and the width of each individual character of the text, we iterate through the line and calculate the enclosing rectangle for each character:

```cpp
     for (int charIndex = firstIndex; 
           charIndex <= lastIndex;++charIndex){ 
        int charWidth = charWidthList[charIndex]; 
        QRect charRect(left, top, charWidth, m_fontHeight); 
        m_rectList.push_back(charRect); 
        left += charWidth; 
     } 
```

For the very last line of the text, we add a rectangle holding the position beyond the last character:

```cpp
      if (lastIndex == (m_editorText.size() - 1)) { 
        QRect lastRect(left, top, 1, m_fontHeight); 
        m_rectList.push_back(lastRect); 
      } 
```

The top field is increased by the height of the line for each new line:

```cpp
          top += m_fontHeight; 
        } 
      } 
    } 
```

The `paintEvent` method is called by the framework every time the window needs to be repainted, or when we force a repainting by calling `update`. The framework clears the content of the window before the call to `paintEvent`:

First, we create a `QPinter` object that we then use to write on. We set some rendering and the font of the text:

```cpp
     void EditorWidget::paintEvent(QPaintEvent* /* eventPtr */) { 
       QPainter painter(this); 
       painter.setRenderHint(QPainter::Antialiasing); 
       painter.setRenderHint(QPainter::TextAntialiasing); 
       painter.setFont(m_textFont); 
```

We calculate the minimum and maximum index of the marked text (even though we do not yet know if the application holds mark mode):

```cpp
    int minIndex = qMin(m_firstIndex, m_lastIndex), 
       maxIndex = qMax(m_firstIndex, m_lastIndex); 
```

We iterate through the text of the editor. We write every character except a new line:

```cpp
     for (int index = 0; index < m_editorText.length(); ++index) { 
       QChar c = m_editorText[index]; 
```

If the character is marked, we write it with white text on a black background:

```cpp
    if (c != 'n') { 
      if ((m_mode == Mark) && 
          (index >= minIndex) && (index < maxIndex)) { 
        painter.setPen(Qt::white); 
        painter.setBackground(Qt::black); 
      } 
```

If the character is not marked, we write it with black text on a white background:

```cpp
      else { 
        painter.setPen(Qt::black); 
        painter.setBrush(Qt::white); 
      } 
```

When the colors of the text and background have been set, we look up the rectangle enclosing the character and write the character itself:

```cpp
      QRect rect = m_rectList[index]; 
      painter.drawText(rect, c); 
    } 
  } 
```

Finally, we also paint the caret:

```cpp
      m_caret.paint(&painter); 
    } 
```

# The main function

The `main` function is similar to the main function of the previous applications: it creates an application, shows the drawing window, and starts the execution of the application.

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

The output is shown in the following screenshot:

![](img/4191c0b5-fcd6-4a9e-a3e4-540323edcd89.png)

# Summary

In this chapter, we have developed more advanced versions of the analog clock, the drawing program, and the editor. The clock shows the current hour, minute, and second. The drawing program, allows the user to draw lines, rectangles, and ellipses. The editor allows the user to input and edit text. The clock face has digits instead of lines. In the drawing program we can mark, modify, and cut and paste figures, and in the editor, we can change font and alignment and mark a text block.

In [Chapter 7](1ce9af28-ea17-439f-945d-2353f6097157.xhtml), *The Games*, we will start developing the games Othello and Nought and Crosses.