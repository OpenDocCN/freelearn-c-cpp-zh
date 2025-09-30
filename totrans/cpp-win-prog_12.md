# 第十二章。辅助类

小型 Windows 包括一组辅助类，如下所示：

+   `Size`、`Point`、`Rect`、`Color`和`Font`：这些封装了 Win32 API 结构`SIZE`、`POINT`、`RECT`、`COLORREF`和`LOGFONT`。它们配备了与文件、剪贴板和注册表通信的方法。注册表是 Windows 系统中的一个数据库，我们可以用它来在应用程序执行之间存储值。

+   `Cursor`：表示 Windows 光标。

+   `DynamicList`：包含具有一组回调函数的动态大小列表。

+   `Tree`：包含递归树结构。

+   `InfoList`：包含可以转换到和从内存缓冲区转换的通用信息列表。

+   此外，还有一些字符串操作函数。

# `Size`类

`Size`类是一个包含宽度和高度的简单类：

**Size.h**

```cpp
namespace SmallWindows { 

```

`ZeroSize`对象是一个其宽度和高度设置为零的对象：

```cpp
  class Size; 
  extern const Size ZeroSize;  
  class Size { 
    public: 

```

默认构造函数将宽度和高度初始化为零。大小可以通过初始化和赋值给另一个大小来初始化。`Size`类使用赋值运算符将大小赋给另一个大小：

```cpp
      Size(); 
      Size(int width, int height); 
      Size(const Size& size); 
      Size& operator=(const Size& size); 

```

`Size`对象可以被初始化并赋值为 Win32 API `SIZE`结构体的值，并且`Size`对象可以被转换为`SIZE`：

```cpp
      Size(const SIZE& size); 
      Size& operator=(const SIZE& size); 
      operator SIZE() const; 

```

比较两个大小时，首先比较宽度。如果它们相等，然后比较高度：

```cpp
      bool operator==(const Size& size) const; 
      bool operator!=(const Size& size) const; 
      bool operator<(const Size& size) const; 
      bool operator<=(const Size& size) const; 
      bool operator>(const Size& size) const; 
      bool operator>=(const Size& size) const;  
      friend Size Min(const Size& left, const Size& right); 
      friend Size Max(const Size& left, const Size& right); 

```

乘法运算符将因子乘以宽度和高度。请注意，尽管因子是双精度浮点数，但得到的宽度和高度始终被四舍五入为整数：

```cpp
      Size operator*=(double factor); 
      friend Size operator*(const Size& size, double factor); 
      friend Size operator*(double factor, const Size& size); 

```

也可以使用一对值来乘以大小，其中第一个值乘以宽度，第二个值乘以高度。此外，在这种情况下，得到的宽度和高度都是整数：

```cpp
      Size operator*=(pair<double,double> factorPair); 
      friend Size operator*(const Size& size, 
                            pair<double,double> factorPair); 
      friend Size operator*(pair<double,double> factorPair, 
                            const Size& size); 

```

第一组加法运算符将距离加到宽度和高度上：

```cpp
      Size operator+=(int distance); 
      Size operator-=(int distance); 
      friend Size operator+(const Size& size, int distance); 
      friend Size operator-(const Size& size, int distance); 

```

第二组加法运算符分别将宽度和高度相加和相减：

```cpp
      Size operator+=(const Size& size); 
      Size operator-=(const Size& size); 
      friend Size operator+(const Size& left, const Size& right); 
      friend Size operator-(const Size& left, const Size& right); 

```

大小可以被写入到文件流、剪贴板和注册表中，也可以从这些地方读取：

```cpp
      bool WriteSizeToStream(ostream& outStream) const;   
      bool ReadSizeFromStream(istream& inStream); 
      void WriteSizeToClipboard(InfoList& infoList) const; 
      void ReadSizeFromClipboard(InfoList& infoList); 
      void WriteSizeToRegistry(String key) const; 
      void ReadSizeFromRegistry(String key, 
                                Size defaultSize = ZeroSize); 

```

宽度和高度通过常量方法进行检查，并通过非常量方法进行修改：

```cpp
      int Width() const {return width;} 
      int Height() const {return height;} 
      int& Width() {return width;} 
      int& Height() {return height;}  

    private: 
      int width, height; 
  }; 
}; 

```

`Size`类的实现相当直接：

**Size.cpp**

```cpp
#include "SmallWindows.h"  
namespace SmallWindows { 
  Size::Size() 
   :width(0), 

    height(0) { 
    // Empty. 
  }  

  Size::Size(int width, int height) 
   :width(width), 
    height(height) { 
    // Empty. 
  } 

  Size::Size(const Size& size) 
   :width(size.width), 
    height(size.height) { 
    // Empty. 
  } 

  Size& Size::operator=(const Size& size) { 
    if (this != &size) { 
      width = size.width; 
      height = size.height; 
    } 
    return *this; 
  } 

  Size::Size(const SIZE& size) 
   :width(size.cx), 
    height(size.cy) { 
    // Empty. 
  } 

  Size& Size::operator=(const SIZE& size) { 
    width = size.cx; 
    height = size.cy; 
    return *this; 
  } 

  Size::operator SIZE() const { 
    SIZE size = {width, height}; 
    return size; 
  } 

  bool Size::operator==(const Size& size) const { 
    return (width == size.width) && (height == size.height); 
  } 

  bool Size::operator!=(const Size& size) const { 
    return !(*this == size); 
  } 

```

如前所述，比较两个大小时，首先比较宽度。如果它们相等，然后比较高度：

```cpp
  bool Size::operator<(const Size& size) const { 
    return (width < size.width) || 
           ((width == size.width) && (height < size.height)); 
  }

  bool Size::operator<=(const Size& size) const { 
    return ((*this < size) || (*this == size)); 
  }

  bool Size::operator>(const Size& size) const { 
    return !(*this <= size); 
  }

  bool Size::operator>=(const Size& size) const { 
    return !(*this < size); 
  } 

```

注意，如果`Min`和`Max`返回的值相等，则返回右侧的值。我们可以让它返回左侧的值。然而，由于在这种情况下`Size`对象持有的`x`和`y`值相同，并且方法返回的是对象而不是对象的引用，所以这并不重要。返回的是相同的值：

```cpp
  Size Min(const Size& left, const Size& right) { 
    return (left < right) ? left : right; 
  }

  Size Max(const Size& left, const Size& right) { 
    return (left > right) ? left : right; 
  } 

```

如前所述，得到的宽度和高度始终被四舍五入为整数，即使因子是双精度浮点数：

```cpp
  Size Size::operator*=(double factor) { 
    width = (int) (factor * width); 
    height = (int) (factor * height); 
    return *this; 
  }  

  Size operator*(const Size& size, double factor) { 
    return Size((int) (size.width * factor), 
                (int) (size.height * factor)); 
  } 

  Size operator*(double factor, const Size& size) { 
    return Size((int) (factor * size.width), 
                (int) (factor * size.height)); 
  } 

  Size Size::operator*=(pair<double,double> factorPair) { 
    width = (int) (factorPair.first * width); 
    height = (int) (factorPair.second * height); 
    return *this; 
  } 

  Size operator*(const Size& size, 
                 pair<double,double> factorPair) { 
    return Size((int) (size.width * factorPair.first), 
                (int) (size.height * factorPair.second)); 
  } 

  Size operator*(pair<double,double> factorPair, 
                 const Size& size) { 
    return Size((int) (factorPair.first * size.width), 
                (int) (factorPair.second * size.height)); 
  } 

  Size Size::operator+=(int distance) { 
    width += distance; 
    height += distance; 
    return *this; 
  } 
  Size Size::operator-=(int distance) { 
    width -= distance; 
    height -= distance; 
    return *this; 
  }  

  Size operator+(const Size& size, int distance) { 
    return Size(size.width + distance, size.height + distance); 
  } 

  Size operator-(const Size& size, int distance) { 
    return Size(size.width - distance, size.height - distance); 
  } 

  Size Size::operator+=(const Size& size) { 
    width += size.width; 
    height += size.height; 
    return *this; 
  } 

  Size Size::operator-=(const Size& size) { 
    width -= size.width; 
    height -= size.height; 
    return *this; 
  } 

  Size operator+(const Size& left, const Size& right) { 
    return Size(left.width + right.width, 
                right.height + right.height); 
  } 

  Size operator-(const Size& left, const Size& right) { 
    return Size(left.width - right.width, 
                right.height - right.height); 
  } 

  bool Size::WriteSizeToStream(ostream& outStream) const { 
    outStream.write((char*) &width, sizeof width); 
    outStream.write((char*) &height, sizeof height); 
    return ((bool) outStream); 
  } 

  bool Size::ReadSizeFromStream(istream& inStream) { 
    inStream.read((char*) &width, sizeof width); 
    inStream.read((char*) &height, sizeof height); 
    return ((bool) inStream); 
  } 

  void Size::WriteSizeToClipboard(InfoList& infoList) const { 
    infoList.AddValue<int>(width); 
    infoList.AddValue<int>(height); 
  } 

  void Size::ReadSizeFromClipboard(InfoList& infoList) { 
    infoList.GetValue<int>(width); 
    infoList.GetValue<int>(height); 
  } 

```

当将大小写入注册表时，我们将大小转换为 `SIZE` 结构，并将其发送到 `Registry` 中的 `WriteBuffer`：

```cpp
  void Size::WriteSizeToRegistry(String key) const { 
    SIZE sizeStruct = (SIZE) *this; 
    Registry::WriteBuffer(key, &sizeStruct, sizeof sizeStruct); 
  } 

```

当从注册表中读取大小，我们将默认大小转换为 `SIZE` 结构，并将其发送到 `Registry` 中的 `ReadBuffer`。然后，结果被转换回 `Size` 对象：

```cpp
  void Size::ReadSizeFromRegistry(String key, 
                                  Size defaultSize /*=ZeroSize*/){ 
    SIZE sizeStruct, defaultSizeStruct = (SIZE) defaultSize; 
    Registry::ReadBuffer(key, &sizeStruct, sizeof sizeStruct, 
                         &defaultSizeStruct); 
    *this = Size(sizeStruct); 
  } 
  const Size ZeroSize(0, 0); 
}; 

```

# 点类

`Point` 类是一个小的类，包含二维点的 *x* 和 *y* 位置：

**Point.h**

```cpp
namespace SmallWindows { 
  class Point { 
    public: 

```

默认构造函数将 *x* 和 *y* 值初始化为零。点可以通过另一个点初始化和赋值：

```cpp
      Point(); 
      Point(int x, int y); 
      Point(const Point& point); 

```

与前面提到的 `Size` 类类似，`Point` 使用赋值运算符：

```cpp
      Point& operator=(const Point& point); 

```

与前一部分中的 `SIZE` 类似，存在一个 `POINT` Win32 API 结构。`Point` 对象可以通过 `POINT` 结构初始化和赋值，并且 `Point` 对象可以转换为 `POINT`：

```cpp
      Point(const POINT& point); 
      Point& operator=(const POINT& point); 
      operator POINT() const; 

```

比较两个点时，首先比较 *x* 值。如果它们相等，然后比较 *y* 值：

```cpp
      bool operator==(const Point& point) const; 
      bool operator!=(const Point& point) const; 
      bool operator<(const Point& point) const; 
      bool operator<=(const Point& point) const; 
      bool operator>(const Point& point) const; 
      bool operator>=(const Point& point) const;  
      friend Point Min(const Point& left, const Point& right); 
      friend Point Max(const Point& left, const Point& right); 

```

与前面提到的 `Size` 类类似，点的 *x* 和 *y* 值可以乘以一个因子。请注意，尽管因子是一个双精度值，但生成的 *x* 和 *y* 值始终四舍五入为整数：

```cpp
      Point& operator*=(double factor); 
      friend Point operator*(const Point& point, double factor); 
      friend Point operator*(double factor, const Point& point); 

```

还可以将点与一对值相乘，其中第一个值乘以 *x* 值，第二个值乘以 *y* 值。在这种情况下，生成的 *x* 和 *y* 值也是整数：

```cpp
      Point& operator*=(pair<double,double> factorPair); 
      friend Point operator*(const Point& point, 
                             pair<double,double> factorPair); 
      friend Point operator*(pair<double,double> factorPair, 
                             const Point& point); 

```

第一组加法运算符将整数距离加到点的 *x* 和 *y* 值上：

```cpp
      Point& operator+=(const int distance); 
      Point& operator-=(const int distance); 
      friend Point operator+(const Point& left, int distance); 
      friend Point operator-(const Point& left, int distance); 

```

第二组加法运算符将大小宽度和高度加到点的 *x* 和 *y* 值上：

```cpp
      Point& operator+=(const Size& size); 
      Point& operator-=(const Size& size); 
      friend Point operator+(const Point& point,const Size& size); 
      friend Point operator-(const Point& point,const Size& size); 

```

第三组加法运算符将点的 *x* 和 *y* 值加和减去：

```cpp
      Point& operator+=(const Point& point); 
      Point& operator-=(const Point& point); 
      friend Point operator+(const Point&left, const Point&right); 
      friend Size operator-(const Point& left, const Point&right); 

```

点可以写入、读取文件流、剪贴板和注册表：

```cpp
      bool WritePointToStream(ostream& outStream) const; 
      bool ReadPointFromStream(istream& inStream); 
      void WritePointToClipboard(InfoList& infoList) const; 
      void ReadPointFromClipboard(InfoList& infoList); 
      void WritePointToRegistry(String key) const;  
      void ReadPointFromRegistry(String key, 
                            Point defaultPoint /* = ZeroPoint */); 

```

点的 *x* 和 *y* 值由常量方法检查并由非常量方法修改：

```cpp
      int X() const {return x;} 
      int Y() const {return y;} 
      int& X() {return x;} 
      int& Y() {return y;} 

    private: 
      int x, y; 
  }; 

  extern const Point ZeroPoint; 
}; 

```

`Point` 类的实现也很直接：

**Point.cpp**

```cpp
#include "SmallWindows.h"

namespace SmallWindows { 
  Point::Point() 
   :x(0), y(0) { 
    // Empty. 
  } 

  Point::Point(int x, int y) 
   :x(x), y(y) { 
    // Empty. 
  } 

  Point::Point(const Point& point) 
   :x(point.x), 
    y(point.y) { 
    // Empty. 
  } 

```

在赋值运算符中，一个好的习惯是验证我们不会分配相同的对象。然而，在这种情况下并不完全必要，因为我们只是分配了 *x* 和 *y* 的整数值：

```cpp
  Point& Point::operator=(const Point& point) { 
    if (this != &point) { 
      x = point.x; 
      y = point.y; 
    } 

    return *this; 
  } 

  Point::Point(const POINT& point) 
   :x(point.x), 
    y(point.y) { 
    // Empty. 
  } 

  Point& Point::operator=(const POINT& point) { 
    x = point.x; 
    y = point.y; 
    return *this; 
  } 

  Point::operator POINT() const { 
    POINT point = {x, y}; 
    return point; 
  } 

  bool Point::operator==(const Point& point) const { 
    return ((x == point.x) && (y == point.y)); 
  } 

  bool Point::operator!=(const Point& point) const { 
    return !(*this == point); 
  } 

  bool Point::operator<(const Point& point) const { 
    return (x < point.x) || ((x == point.x) && (y < point.y)); 
  } 

  bool Point::operator<=(const Point& point) const { 
    return ((*this < point) || (*this == point)); 
  } 

  bool Point::operator>(const Point& point) const { 
    return !(*this <= point); 
  } 

  bool Point::operator>=(const Point& point) const { 
    return !(*this < point); 
  } 

  Point Min(const Point& left, const Point& right) { 
    return (left < right) ? left : right; 
  } 

  Point Max(const Point& left, const Point& right) { 
    return (left > right) ? left : right; 
  } 

  Point& Point::operator*=(double factor) { 
    x = (int) (factor * x); 
    y = (int) (factor * y); 
    return *this; 
  } 

  Point operator*(const Point& point, double factor) { 
    return Point((int) (point.x * factor), 
                 (int) (point.y * factor)); 
  } 

  Point operator*(double factor, const Point& point) { 
    return Point((int) (factor * point.x), 
                 (int) (factor * point.y)); 
  } 

  Point& Point::operator*=(pair<double,double> factorPair) { 
    x = (int) (factorPair.first * x); 
    y = (int) (factorPair.second * y); 
    return *this; 
  } 

  Point operator*(const Point& point, 
                  pair<double,double> factorPair) { 
    return Point((int) (point.x * factorPair.first), 
                 (int) (point.y * factorPair.second)); 
  } 

  Point operator*(pair<double,double> factorPair, 
                  const Point& point) { 
    return Point((int) (factorPair.first * point.x), 
                 (int) (factorPair.second * point.y)); 
  } 

  Point& Point::operator+=(const int distance) { 
    x += distance; 
    y += distance; 
    return *this; 
  } 

  Point& Point::operator-=(const int distance) { 
    x -= distance; 
    y -= distance; 
    return *this; 
  } 

  Point& Point::operator+=(const Size& size) { 
    x += size.Width(); 
    y += size.Height(); 
    return *this; 
  } 

  Point& Point::operator-=(const Size& size) { 
    x -= size.Width(); 
    y -= size.Height(); 
    return *this; 
  } 

  Point& Point::operator+=(const Point& point) { 
    x += point.x; 
    y += point.y; 
    return *this; 
  } 

  Point& Point::operator-=(const Point& point) { 
    x -= point.x; 
    y -= point.y; 
    return *this; 
  } 

  Point operator+(const Point& left, int distance) { 
    return Point(left.x + distance, left.y + distance); 
  } 

  Point operator-(const Point& left, int distance) { 
    return Point(left.x - distance, left.y - distance); 
  } 

  Point operator+(const Point& point, const Size& size) { 
    return Point(point.x + size.Width(), point.y + size.Height()); 
  } 

  Point operator-(const Point& point, const Size& size) { 
    return Point(point.x - size.Width(), point.y - size.Height()); 
  } 

  Point operator+(const Point& left, const Point& right) { 
    return Point(left.x + right.x, left.y + right.y); 
  } 

  Size operator-(const Point& left, const Point& right) { 
    return Size(left.x - right.x, left.y - right.y); 
  } 

  bool Point::WritePointToStream(ostream& outStream) const { 
    outStream.write((char*) &x, sizeof x); 
    outStream.write((char*) &y, sizeof y); 
    return ((bool) outStream); 
  } 

  bool Point::ReadPointFromStream(istream& inStream) { 
    inStream.read((char*) &x, sizeof x); 
    inStream.read((char*) &y, sizeof y); 
    return ((bool) inStream); 
  } 

  void Point::WritePointToClipboard(InfoList& infoList) const { 
    infoList.AddValue<int>(x); 
    infoList.AddValue<int>(y); 
  } 

  void Point::ReadPointFromClipboard(InfoList& infoList) { 
    infoList.GetValue<int>(x); 
    infoList.GetValue<int>(y); 
  } 

  void Point::WritePointToRegistry(String key) const { 
    POINT pointStruct = (POINT) *this; 
    Registry::WriteBuffer(key, &pointStruct, sizeof pointStruct); 
  } 

  void Point::ReadPointFromRegistry(String key, 
                           Point defaultPoint /* = ZeroPoint */) { 
    POINT pointStruct, defaultPointStruct = (POINT) defaultPoint; 
    Registry::ReadBuffer(key, &pointStruct, sizeof pointStruct, 
                         &defaultPointStruct); 
    *this = Point(pointStruct); 
  } 

  const Point ZeroPoint(0, 0); 
}; 

```

# Rect 类

`Rect` 类包含矩形的四个边：左、上、右和下。

**Rect.h**

```cpp
namespace SmallWindows { 
  class Rect; 
  extern const Rect ZeroRect; 

  class Rect { 
    public: 

```

默认构造函数将所有四个边设置为零。矩形可以通过另一个矩形初始化或赋值。也可以使用左上角和右下角以及包含矩形宽度和高度的尺寸初始化矩形：

```cpp
      Rect(); 
      Rect(int left, int top, int right, int bottom); 
      Rect(const Rect& rect); 
      Rect& operator=(const Rect& rect); 
      Rect(Point topLeft, Point bottomRight); 
      Rect(Point topLeft, Size size); 

```

与前几节中的 `SIZE` 和 `POINT` 类似，矩形可以初始化和赋值给 Win32 API `RECT` 结构的值。`Rect` 对象也可以转换为 `RECT`：

```cpp
      Rect(const RECT& rect); 
      Rect& operator=(const RECT& rect); 
      operator RECT() const; 

```

比较运算符首先比较左上角。如果它们相等，然后比较右下角：

```cpp
      bool operator==(const Rect& rect) const; 
      bool operator!=(const Rect& rect) const; 
      bool operator<(const Rect& rect) const; 
      bool operator<=(const Rect& rect) const; 
      bool operator>(const Rect& rect) const; 
      bool operator>=(const Rect& rect) const; 

```

乘法运算符将所有边乘以因子。尽管因子是双精度浮点数，但边框值始终是整数，类似于前几节中的`Size`和`Point`情况：

```cpp
      Rect& operator*=(double factor); 
      friend Rect operator*(const Rect& rect, double factor); 
      friend Rect operator*(double factor, const Rect& rect); 

```

还可以将矩形与一对值相乘，其中第一个值与`left`和`right`相乘，第二个值与`top`和`bottom`相乘。此外，在这种情况下，结果值都是整数：

```cpp
      Rect& operator*=(pair<double,double> factorPair); 
      friend Rect operator*(const Rect& rect, 
                            pair<double,double> factorPair); 
      friend Rect operator*(pair<double,double> factorPair, 
                            const Rect& rect); 

```

以下运算符有点特殊：加法运算符将大小添加到右下角，同时保持左上角不变，而减法运算符从左上角减去大小，同时保持右下角不变：

```cpp
      Rect& operator+=(const Size& size); 
      Rect& operator-=(const Size& size); 

```

然而，以下运算符将大小添加到或从左上角和右下角：

```cpp
      friend Rect operator+(const Rect& rect, const Size& size); 
      friend Rect operator-(const Rect& rect, const Size& size); 

```

以下运算符接受一个点作为参数，并将该点添加到，并从左上角和右下角减去：

```cpp
      Rect& operator+=(const Point& point); 
      Rect& operator-=(const Point& point); 
      friend Rect operator+(const Rect& rect, const Point& point); 
      friend Rect operator+(const Point& point, const Rect& rect); 
      friend Rect operator-(const Rect& rect, const Point& point); 

```

矩形的宽度是左右边框之间的绝对差值，其高度是上下边框之间的绝对差值：

```cpp
      int Width() const {return abs(right - left);} 
      int Height() const {return abs(bottom - top);} 

```

`GetSize`方法返回矩形的宽度和高度。由于存在具有该名称的类，因此无法将其命名为`Size`。然而，仍然可以定义返回`Size`对象的运算符。`Size`和`Point`运算符返回矩形的尺寸和左上角：

```cpp
      Size GetSize() const {return Size(Width(), Height());} 
      operator Size() const {return GetSize();} 
      operator Point() const {return TopLeft();} 

```

左上角和右下角都可以进行检查和修改。由于没有对应于角落的字段，因此不适当定义返回点引用的方法：

```cpp
      Point TopLeft() const {return Point(left, top);} 
      Point BottomRight() const {return Point(right, bottom);} 

      void SetTopLeft(Point topLeft) {left = topLeft.X(); 
                                      right = topLeft.Y();} 
      void SetBottomRight(Point bottomRight) 
                         {right = bottomRight.X(); 
                          bottom = bottomRight.Y();} 

```

`Clear`方法将所有四个角设置为 0，`Normalize`方法如果左右边框和上下边框出现错误顺序，则交换左右边框和上下边框，`PointInside`方法如果点位于矩形内部，则返回`true`，假设它已经被归一化：

```cpp
      void Clear(); 
      void Normalize(); 
      bool PointInside(Point point) const; 

```

矩形可以写入和读取文件流、剪贴板和注册表：

```cpp
      bool WriteRectToStream(ostream& outStream) const; 
      bool ReadRectFromStream(istream& inStream); 
      void WriteRectToClipboard(InfoList& infoList) const; 
      void ReadRectFromClipboard(InfoList& infoList); 
      void WriteRectToRegistry(String key) const; 
      void ReadRectFromRegistry(String key, 
                                Rect defaultRect = ZeroRect); 

```

四个角通过常量方法进行检查，并通过非常量方法进行修改：

```cpp
      int Left() const {return left;} 
      int Right() const {return right;} 
      int Top() const {return top;} 
      int Bottom() const {return bottom;} 

      int& Left() {return left;} 
      int& Right() {return right;} 
      int& Top() {return top;} 
      int& Bottom() {return bottom;} 

    private: 
      int left, top, right, bottom; 
  }; 
}; 

```

与`Size`和`Point`类似，`Rect`的实现相当直接。

**Rect.cpp**

```cpp
#include "SmallWindows.h"

namespace SmallWindows { 
  Rect::Rect() 
   :left(0), top(0), right(0), bottom(0) { 
    // Empty. 
  } 

  Rect::Rect(int left, int top, int right, int bottom) 
   :left(left), 
    top(top), 
    right(right), 
    bottom(bottom) { 
    // Empty. 
  } 

  Rect::Rect(const Rect& rect) 
   :left(rect.left), 
    top(rect.top), 
    right(rect.right), 
    bottom(rect.bottom) { 
    // Empty. 
  } 

  Rect& Rect::operator=(const Rect& rect) { 
    if (this != &rect) { 
      left = rect.left; 
      top = rect.top; 
      right = rect.right; 
      bottom = rect.bottom; 
    } 

    return *this; 
  } 

  Rect::Rect(Point topLeft, Point bottomRight) 
   :left(topLeft.X()), 
    top(topLeft.Y()), 
    right(bottomRight.X()), 
    bottom(bottomRight.Y()) { 
    // Empty. 
  } 

  Rect::Rect(Point topLeft, Size size) 
   :left(topLeft.X()), 
    top(topLeft.Y()), 
    right(topLeft.X() + size.Width()), 
    bottom(topLeft.Y() + size.Height()) { 
    // Empty. 
  } 

  Rect::Rect(const RECT& rect) 
   :left(rect.left), 
    top(rect.top), 
    right(rect.right), 
    bottom(rect.bottom) { 
    // Empty. 
  } 

  Rect& Rect::operator=(const RECT& rect) { 
    left = rect.left; 
    top = rect.top; 
    right = rect.right; 
    bottom = rect.bottom; 
    return *this; 
  } 

  Rect::operator RECT() const { 
    RECT rect = {left, top, right, bottom}; 
    return rect; 
  } 

  bool Rect::operator==(const Rect& rect) const { 
    return (left == rect.left) && (top == rect.top) && 
           (right == rect.right) && (bottom == rect.bottom); 
  } 

  bool Rect::operator!=(const Rect& rect) const { 
    return !(*this == rect); 
  } 

  bool Rect::operator<(const Rect& rect) const { 
    return (TopLeft() < rect.TopLeft()) || 
           ((TopLeft() == rect.TopLeft()) && 
            (BottomRight() < rect.BottomRight())); 
  } 

  bool Rect::operator<=(const Rect& rect) const { 
    return ((*this < rect) || (*this == rect)); 
  } 

  bool Rect::operator>(const Rect& rect) const { 
    return !(*this <= rect); 
  } 

  bool Rect::operator>=(const Rect& rect) const { 
    return !(*this < rect); 
  } 

  Rect& Rect::operator*=(double factor) { 
    left = (int) (factor * left); 
    top = (int) (factor * top); 
    right = (int) (factor * right); 
    bottom = (int) (factor * bottom); 
    return *this; 
  } 

  Rect operator*(const Rect& rect, double factor) { 
    return Rect(rect.TopLeft() * factor, 
                rect.BottomRight() * factor); 
  } 

  Rect operator*(double factor, const Rect& rect) { 
    return Rect(factor * rect.TopLeft(), 
                factor * rect.BottomRight()); 
  } 

  Rect& Rect::operator*=(pair<double,double> factorPair) { 
    left = (int) (factorPair.first * left); 
    top = (int) (factorPair.second * top); 
    right = (int) (factorPair.first * right); 
    bottom = (int) (factorPair.second * bottom); 
    return *this; 
  } 

  Rect operator*(const Rect& rect, 
                 pair<double,double> factorPair) { 
    return Rect(rect.TopLeft() * factorPair, 
                rect.BottomRight() * factorPair); 
  } 

  Rect operator*(pair<double,double> factorPair, 
                 const Rect& rect) { 
    return Rect(factorPair * rect.TopLeft(), 
                factorPair * rect.BottomRight()); 
  } 

  Rect& Rect::operator+=(const Size& size) { 
    right += size.Width(); 
    bottom += size.Height(); 
    return *this; 
  } 

  Rect& Rect::operator-=(const Size& size) { 
    left -= size.Width(); 
    top -= size.Height(); 
    return *this; 
  } 

  Rect operator+(const Rect& rect, const Size& size) { 
    return Rect(rect.left + size.Width(), 
                rect.top + size.Height(), 
                rect.right + size.Width(), 
                rect.bottom + size.Height()); 
  } 

  Rect operator-(const Rect& rect, const Size& size) { 
    return Rect(rect.left - size.Width(), 
                rect.top - size.Height(), 
                rect.right - size.Width(), 
                rect.bottom - size.Height()); 
  } 

  Rect& Rect::operator+=(const Point& point) { 
    left += point.X(); 
    top += point.Y(); 
    right += point.X(); 
    bottom += point.Y(); 
    return *this; 
  } 

  Rect& Rect::operator-=(const Point& point) { 
    left -= point.X(); 
    top -= point.Y(); 
    right -= point.X(); 
    bottom -= point.Y(); 
    return *this; 
  } 

  Rect operator+(const Rect& rect, const Point& point) { 
    return Rect(rect.left + point.X(), rect.top + point.Y(), 
                rect.right + point.X(), rect.bottom + point.Y()); 
  } 

  Rect operator+(const Point& point, const Rect& rect) { 
    return Rect(point.X() + rect.left, point.Y() + rect.top, 
                point.X() + rect.right, point.Y() + rect.bottom); 
  } 

  Rect operator-(const Rect& rect, const Point& point) { 
    return Rect(rect.left - point.X(), rect.top - point.Y(), 
                rect.right - point.X(), rect.bottom - point.Y()); 
  }  

  void Rect::Clear() { 
    left = top = right = bottom = 0; 
  }  

  void Rect::Normalize() { 
    int minX = min(left, right), minY = min(top, bottom), 
        maxX = max(left, right), maxY = max(top, bottom);  
    left = minX; 
    top = minY; 
    right = maxX; 
    bottom = maxY; 
  } 

  bool Rect::PointInside(Point point) const { 
    return ((left <= point.X()) && (point.X() <= right) && 
            (top <= point.Y()) && (point.Y() <= bottom)); 
  } 

  bool Rect::WriteRectToStream(ostream& outStream) const { 
    outStream.write((char*) &left, sizeof left); 
    outStream.write((char*) &top, sizeof top); 
    outStream.write((char*) &right, sizeof right); 
    outStream.write((char*) &bottom, sizeof bottom); 
    return ((bool) outStream); 
  } 

  bool Rect::ReadRectFromStream(istream& inStream) { 
    inStream.read((char*) &left, sizeof left); 
    inStream.read((char*) &top, sizeof top); 
    inStream.read((char*) &right, sizeof right); 
    inStream.read((char*) &bottom, sizeof bottom); 
    return ((bool) inStream); 
  } 

  void Rect::WriteRectToClipboard(InfoList& infoList) const { 
    infoList.AddValue<int>(left); 
    infoList.AddValue<int>(top); 
    infoList.AddValue<int>(right); 
    infoList.AddValue<int>(bottom); 
  }  

  void Rect::ReadRectFromClipboard(InfoList& infoList) { 
    infoList.GetValue<int>(left); 
    infoList.GetValue<int>(top); 
    infoList.GetValue<int>(right); 
    infoList.GetValue<int>(bottom); 
  } 

  void Rect::WriteRectToRegistry(String key) const { 
    RECT pointStruct = (RECT) *this; 
    Registry::WriteBuffer(key, &pointStruct, sizeof pointStruct); 
  } 

  void Rect::ReadRectFromRegistry(String key, 
                             Rect defaultRect /* = ZeroRect */) { 
    RECT rectStruct, defaultRectStruct = (RECT) defaultRect; 
    Registry::ReadBuffer(key, &rectStruct, sizeof rectStruct, 
                         &defaultRectStruct); 
    *this = Rect(rectStruct); 
  } 

  const Rect ZeroRect(0, 0, 0, 0); 
}; 

```

# 颜色类

`Color`类是 Win32 API `COLORREF`结构的包装类，它按照红-绿-蓝（RGB）标准存储颜色。颜色的每个分量都由一个介于 0 到 255 之间的值表示，这给出了理论上的总数 256³ = 16,777,216 种不同的颜色，其中`Color`定义了 142 种标准颜色。

**Color.h**

```cpp
namespace SmallWindows { 
  class Color; 
  extern const Color SystemColor; 

```

默认构造函数将红色、绿色和蓝色值初始化为零，这对应于黑色。颜色对象也可以通过另一个颜色初始化和赋值：

```cpp
  class Color { 
    public: 
      Color(); 
      Color(int red, int green, int blue); 
      Color(const Color& color); 
      Color& operator=(const Color& color); 

```

等价运算符比较红色、绿色和蓝色值：

```cpp
      bool operator==(const Color& color) const; 
      bool operator!=(const Color& color) const; 

```

`Inverse` 函数返回反转的颜色，而 `GrayScale` 返回相应的灰度颜色：

```cpp
      Color Inverse(); 
      void GrayScale(); 

```

颜色可以被写入、从文件流、剪贴板和注册表中读取：

```cpp
      bool WriteColorToStream(ostream& outStream) const; 
      bool ReadColorFromStream(istream& inStream); 
      void WriteColorToClipboard(InfoList& infoList) const; 
      void ReadColorFromClipboard(InfoList& infoList); 
      void WriteColorToRegistry(String key) const; 
      void ReadColorFromRegistry(String key, 
                                 Color defaultColor =SystemColor); 

```

通过常量方法检查包装的 `COLORREF` 结构值，并通过非常量方法进行修改：

```cpp
      COLORREF ColorRef() const {return colorRef;} 
      COLORREF& ColorRef() {return colorRef;} 

    private: 
      COLORREF colorRef; 
  }; 

```

预定义的颜色是常量对象：

```cpp
  extern const Color 
    AliceBlue, AntiqueWhite, Aqua, Aquamarine, 
    Azure, Beige, Bisque, Black, BlanchedAlmond, 
    Blue, BlueViolet, Brown, Burlywood, CadetBlue, 
    Chartreuse, Chocolate, Coral, CornflowerBlue, 
    Cornsilk, Crimson, Cyan, DarkBlue, DarkCyan, 
    DarkGoldenRod, DarkGray, DarkGreen, DarkKhaki, 
    DarkMagenta, DarkOliveGreen, DarkOrange, DarkOrchid, 
    DarkRed, DarkSalmon, DarkSeaGreen, DarkSlateBlue, 
    DarkSlateGray, DarkTurquoise, DarkViolet, DeepPink, 
    DeepSkyBlue, DimGray, DodgerBlue, FireBrick, 
    FloralWhite, ForestGreen, Fuchsia, Gainsboro, 
    GhostWhite, Gold, GoldenRod, Gray, Green, GreenYellow, 
    HoneyDew, HotPink, IndianRed, Indigo, Ivory, Khaki, 
    Lavender, LavenderBlush, Lawngreen, LemonChiffon, 
    LightBlue, LightCoral, LightCyan, LightGoldenRodYellow, 
    LightGreen, LightGray, LightPink, LightSalmon, 
    LightSeaGreen, LightSkyBlue, LightSlateGray, 
    LightSteelBlue, LightYellow, Lime, LimeGreen, Linen, 
    Magenta, Maroon, MediumAquamarine, MediumBlue, 
    MediumOrchid, MediumPurple, MediumSeaGreen, 
    MediumSlateBlue, MediumSpringGreen, MediumTurquoise, 
    MediumVioletRed, MidnightBlue, MintCream, MistyRose, 
    Moccasin, NavajoWhite, Navy, Navyblue, OldLace, Olive, 
    OliveDrab, Orange, OrangeRed, Orchid, PaleGoldenRod, 
    PaleGreen, PaleTurquoise, PaleVioletRed, PapayaWhip, 
    PeachPuff, Peru, Pink, Plum, PowderBlue, Purple, 
    Red, RosyBrown, RoyalBlue, SaddleBrown, Salmon, 
    SandyBrown, SeaGreen, SeaShell, Sienna, Silver, SkyBlue, 
    SlateBlue, SlateGray, Snow, SpringGreen, SteelBlue, 
    SystemColor, Tan, Teal, Thistle, Tomato, Turquoise, 
    Violet, Wheat, White, WhiteSmoke, Yellow, YellowGreen; 
}; 

```

`Color` 的实现相当直接。Win32 的 `RGB` 宏根据三个颜色组件创建一个 `COLORREF` 值。

**Color.cpp**

```cpp
#include "SmallWindows.h" 

namespace SmallWindows { 
  Color::Color() 
   :colorRef(RGB(0, 0, 0)) { 
    // Empty. 
  } 

  Color::Color(COLORREF colorRef) 
   :colorRef(colorRef) { 
    // Empty. 
  } 

  Color::Color(int red, int green, int blue) 
   :colorRef(RGB(red, green, blue)) { 
    // Empty. 
  } 

  Color::Color(const Color& color) 
   :colorRef(color.colorRef) { 
    // Empty. 
  } 

  Color& Color::operator=(const Color& color) { 
    if (this != &color) { 
      colorRef = color.colorRef; 
    } 

    return *this; 
  } 

```

两个颜色相等，如果它们的包装 `COLORREF` 结构相等，并且它们通过 C 标准函数 `memcpy` 进行比较。

```cpp
  bool Color::operator==(const Color& color) const { 
    return (colorRef == color.colorRef); 
  } 

  bool Color::operator!=(const Color& color) const { 
    return !(*this == color); 
  } 

```

`Inverse` 函数返回每个组件从 255 减去的反转颜色，而 `GrayScale` 返回每个组件持有红色、绿色和蓝色组件平均值的相关灰度颜色。`GetRValue`、`GetGValue` 和 `GetBValue` 是 Win32 API 宏，用于提取红色、绿色和蓝色组件：

```cpp
  Color Color::Inverse() { 
    int inverseRed = 255 - GetRValue(colorRef); 
    int inverseGreen = 255 - GetGValue(colorRef); 
    int inverseBlue = 255 - GetBValue(colorRef); 
    return Color(inverseRed, inverseGreen, inverseBlue); 
  } 

  void Color::GrayScale() { 
    int red = GetRValue(colorRef); 
    int green = GetGValue(colorRef); 
    int blue = GetBValue(colorRef); 

    int average = (red + green + blue) / 3; 
    colorRef = RGB(average, average, average); 
  } 

  bool Color::WriteColorToStream(ostream& outStream) const { 
    outStream.write((char*) &colorRef, sizeof colorRef); 
    return ((bool) outStream); 
  } 

  bool Color::ReadColorFromStream(istream& inStream) { 
    inStream.read((char*) &colorRef, sizeof colorRef); 
    return ((bool) inStream); 
  } 

  void Color::WriteColorToClipboard(InfoList& infoList) const { 
    infoList.AddValue<COLORREF>(colorRef); 
  } 

  void Color::ReadColorFromClipboard(InfoList& infoList) { 
    infoList.GetValue<COLORREF>(colorRef); 
  } 

  void Color::WriteColorToRegistry(String key) const { 
    Registry::WriteBuffer(key, &colorRef, sizeof colorRef); 
  } 

  void Color::ReadColorFromRegistry(String key, 
                           Color defaultColor /*=SystemColor */) { 
    Registry::ReadBuffer(key, &colorRef, sizeof colorRef, 
                         &defaultColor.colorRef); 
  } 

```

每个预定义的颜色都调用接受红色、绿色和蓝色组件的构造函数：

```cpp
  const Color 
    AliceBlue(240, 248, 255), AntiqueWhite(250, 235, 215), 
    Aqua(0, 255, 255), Aquamarine(127, 255, 212), 
    Azure(240, 255, 255), Beige(245, 245, 220), 
    Bisque(255, 228, 196), Black(0, 0, 0), 
    BlanchedAlmond(255, 255, 205), Blue(0, 0, 255), 
    BlueViolet(138, 43, 226), Brown(165, 42, 42), 
    Burlywood(222, 184, 135), CadetBlue(95, 158, 160), 
    Chartreuse(127, 255, 0), Chocolate(210, 105, 30), 
    Coral(255, 127, 80), CornflowerBlue(100, 149, 237), 
    Cornsilk(255, 248, 220), Crimson(220, 20, 60), 
    Cyan(0, 255, 255), DarkBlue(0, 0, 139), 
    DarkCyan(0, 139, 139), DarkGoldenRod(184, 134, 11), 
    DarkGray(169, 169, 169), DarkGreen(0, 100, 0), 
    DarkKhaki(189, 183, 107), DarkMagenta(139, 0, 139), 
    DarkOliveGreen(85, 107, 47), DarkOrange(255, 140, 0), 
    DarkOrchid(153, 50, 204), DarkRed(139, 0, 0), 
    DarkSalmon(233, 150, 122), DarkSeaGreen(143, 188, 143), 
    DarkSlateBlue(72, 61, 139), DarkSlateGray(47, 79, 79), 
    DarkTurquoise(0, 206, 209), DarkViolet(148, 0, 211), 
    DeepPink(255, 20, 147), DeepSkyBlue(0, 191, 255), 
    DimGray(105, 105, 105), DodgerBlue(30, 144, 255), 
    FireBrick(178, 34, 34), FloralWhite(255, 250, 240), 
    ForestGreen(34, 139, 34), Fuchsia(255, 0, 255), 
    Gainsboro(220, 220, 220), GhostWhite(248, 248, 255), 
    Gold(255, 215, 0),  GoldenRod(218, 165, 32), 
    Gray(127, 127, 127), Green(0, 128, 0), 
    GreenYellow(173, 255, 47), HoneyDew(240, 255, 240), 
    HotPink(255, 105, 180), IndianRed(205, 92, 92), 
    Indigo(75, 0, 130), Ivory(255, 255, 240), 
    Khaki(240, 230, 140), Lavender(230, 230, 250), 
    LavenderBlush(255, 240, 245), Lawngreen(124, 252, 0), 
    LemonChiffon(255, 250, 205), LightBlue(173, 216, 230), 
    LightCoral(240, 128, 128), LightCyan(224, 255, 255), 
    LightGoldenRodYellow(250, 250, 210), 
    LightGreen(144, 238, 144), LightGray(211, 211, 211), 
    LightPink(255, 182, 193), LightSalmon(255, 160, 122), 
    LightSeaGreen(32, 178, 170), LightSkyBlue(135, 206, 250), 
    LightSlateGray(119, 136, 153), LightSteelBlue(176, 196, 222), 
    LightYellow(255, 255, 224), Lime(0, 255, 0), 
    LimeGreen(50, 205, 50), Linen(250, 240, 230), 
    Magenta(255, 0, 255), Maroon(128, 0, 0), 
    MediumAquamarine(102, 205, 170), MediumBlue(0, 0, 205), 
    MediumOrchid(186, 85, 211), MediumPurple(147, 112, 219), 
    MediumSeaGreen(60, 179, 113), MediumSlateBlue(123, 104, 238), 
    MediumSpringGreen(0, 250, 154), MediumTurquoise(72, 209, 204), 
    MediumVioletRed(199, 21, 133), MidnightBlue(25, 25, 112), 
    MintCream(245, 255, 250), MistyRose(255, 228, 225), 
    Moccasin(255, 228, 181), NavajoWhite(255, 222, 173), 
    Navy(0, 0, 128), Navyblue(159, 175, 223), 
    OldLace(253, 245, 230), Olive(128, 128, 0), 
    OliveDrab(107, 142, 35), Orange(255, 165, 0), 
    OrangeRed(255, 69, 0), Orchid(218, 112, 214), 
    PaleGoldenRod(238, 232, 170), PaleGreen(152, 251, 152), 
    PaleTurquoise(175, 238, 238), PaleVioletRed(219, 112, 147), 
    PapayaWhip(255, 239, 213), PeachPuff(255, 218, 185), 
    Peru(205, 133, 63), Pink(255, 192, 203), 
    Plum(221, 160, 221), PowderBlue(176, 224, 230), 
    Purple(128, 0, 128), Red(255, 0, 0), 
    RosyBrown(188, 143, 143), RoyalBlue(65, 105, 225), 
    SaddleBrown(139, 69, 19), Salmon(250, 128, 114), 
    SandyBrown(244, 164, 96), SeaGreen(46, 139, 87), 
    SeaShell(255, 245, 238), Sienna(160, 82, 45), 
    Silver(192, 192, 192), SkyBlue(135, 206, 235), 
    SlateBlue(106, 90, 205), SlateGray(112, 128, 144), 
    Snow(255, 250, 250), SpringGreen(0, 255, 127), 
    SteelBlue(70, 130, 180), SystemColor(0, 0, 0), 
    Tan(210, 180, 140), Teal(0, 128, 128), 
    Thistle(216, 191, 216), Tomato(255, 99, 71), 
    Turquoise(64, 224, 208), Violet(238, 130, 238), 
    Wheat(245, 222, 179), White(255, 255, 255), 
    WhiteSmoke(245, 245, 245), Yellow(255, 255, 0), 
    YellowGreen(139, 205, 50); 
}; 

```

# `Font` 类

`Font` 类是 Win32 API `LOGFONT` 结构的包装类。该结构包含大量属性；然而，我们只考虑字体名称和大小以及字体是否为斜体、粗体或下划线的字段；其他字段设置为零。系统字体是所有 `LOGFONT` 结构字段都设置为零的字体，这导致系统的标准字体。最后，`Font` 类还包括一个 `Color` 对象。

**Font.h**

```cpp
namespace SmallWindows { 
  class Font; 
  extern const Font SystemFont; 

  class Font { 
    public: 

```

默认构造函数将名称设置为空字符串，并将所有其他值设置为零，从而得到系统字体，通常是 10 点 Arial。字体的大小以排版点给出（1 点 = 1/72 英寸 = 1/72 * 25.4 毫米 ≈ 0.35 毫米）。字体也可以通过另一个字体初始化或赋值：

```cpp
      Font(); 
      Font(String name, int size, 
           bool italic = false, bool bold = false); 
      Font(const Font& Font); 
      Font& operator=(const Font& font); 

```

如果两个字体具有相同的名称和大小，以及相同的斜体、粗体和下划线状态（所有其他字段假定为零），则两个字体相等：

```cpp
      bool operator==(const Font& font) const; 
      bool operator!=(const Font& font) const; 

```

字体可以被写入、从文件流、剪贴板和注册表中读取：

```cpp
      bool WriteFontToStream(ostream& outStream) const; 
      bool ReadFontFromStream(istream& inStream); 
      void WriteFontToClipboard(InfoList& infoList) const; 
      void ReadFontFromClipboard(InfoList& infoList); 
      void WriteFontToRegistry(String key); 
      void ReadFontFromRegistry(String key, 
                                Font defaultFont = SystemFont); 

```

`PointToMeters` 函数将排版点转换为逻辑单位（毫米的百分之一）：

```cpp
      void PointsToLogical(double zoom = 1.0); 

```

通过常量方法检查包装的 `LOGFONT` 结构，并通过非常量方法进行修改：

```cpp
      LOGFONT LogFont() const {return logFont;} 
      LOGFONT& LogFont() {return logFont;} 

```

`color` 字段也可以通过常量方法进行检查，并通过非常量方法进行修改：

```cpp
      Color FontColor() const {return color;} 
      Color& FontColor() {return color;} 

    private: 
      LOGFONT logFont; 
      Color color; 
  }; 
}; 

```

**Font.cpp**

```cpp
#include "SmallWindows.h" 

namespace SmallWindows { 
  Font::Font() { 
    memset(&logFont, 0, sizeof logFont); 
  } 

  Font::Font(String name, int size, bool italic, bool bold) { 
    memset(&logFont, 0, sizeof logFont); 
    wcscpy_s(logFont.lfFaceName, LF_FACESIZE, name.c_str()); 
    logFont.lfHeight = size; 
    logFont.lfItalic = (italic ? TRUE : FALSE); 
    logFont.lfWeight = (bold ? FW_BOLD : FW_NORMAL); 
  } 

  Font::Font(const Font& font) { 
    logFont = font.LogFont(); 
    color = font.color; 
  } 

  Font& Font::operator=(const Font& font) { 
    if (this != &font) { 
      logFont = font.LogFont(); 
      color = font.color; 
    } 

    return *this; 
  } 

```

如果两个字体的包装 `LOGFONT` 结构和它们的 `Color` 字段相等，则两个字体相等：

```cpp
  bool Font::operator==(const Font& font) const { 
    return (::memcmp(&logFont, &font.logFont, 
                     sizeof logFont) == 0) && 
           (color == font.color); 
  } 

  bool Font::operator!=(const Font& font) const { 
    return !(*this == font); 
  } 

```

`write` 和 `read` 方法写入和读取包装的 `LOGFONT` 结构，并调用 `Color` 的写入和读取方法：

```cpp
  bool Font::WriteFontToStream(ostream& outStream) const { 
    outStream.write((char*) &logFont, sizeof logFont); 
    color.WriteColorToStream(outStream); 
    return ((bool) outStream); 
  } 

  bool Font::ReadFontFromStream(istream& inStream) { 
    inStream.read((char*) &logFont, sizeof logFont); 
    color.ReadColorFromStream(inStream); 
    return ((bool) inStream); 
  } 

  void Font::WriteFontToClipboard(InfoList& infoList) const { 
    infoList.AddValue<LOGFONT>(logFont); 
    color.WriteColorToClipboard(infoList); 
  }  

  void Font::ReadFontFromClipboard(InfoList& infoList) { 
    infoList.GetValue<LOGFONT>(logFont); 
    color.ReadColorFromClipboard(infoList); 
  }  

  void Font::WriteFontToRegistry(String key) { 
    Registry::WriteBuffer(key, &logFont, sizeof logFont); 
    color.WriteColorToRegistry(key); 
  }  

  void Font::ReadFontFromRegistry(String key, 
                         Font defaultFont /* = SystemFont */) { 
    Registry::ReadBuffer(key, &logFont, sizeof logFont, 
                         &defaultFont.logFont); 
    color.ReadColorFromRegistry(key); 
  } 

```

一个排版点等于 1/72 英寸，一个英寸等于 25.4 毫米。要将字体排版单位转换为逻辑单位（毫米的百分之一），我们需要将宽度和高度除以 72，乘以 2,540（2,540 逻辑单位等于 25.4 毫米）以及缩放因子：

```cpp
  void Font::PointsToLogical(double zoom /* = 1.0 */) { 
    logFont.lfWidth = 
      (int) (zoom * 2540.0 * logFont.lfWidth / 72.0); 
    logFont.lfHeight = 
      (int) (zoom * 2540.0 * logFont.lfHeight / 72.0); 
  }  

  const Font SystemFont; 
}; 

```

# Cursor 类

Win32 API 中有一组可用的光标，所有这些光标的名称都以`IDC_`开头。在小窗口中，它们被赋予了其他名称，希望这些名称更容易理解。与其他情况不同，我们不能为光标使用枚举，因为它们实际上是零终止的 C++字符串（字符指针）。相反，每个光标都是一个指向零终止字符串的指针。`LPCTSTR`代表**长指针到常量 TChar 字符串**。

光标有自己类的原因，而光标在`Document`类中有方法，是因为光标确实需要一个窗口句柄来设置，而光标则不需要。

**Cursor.h**

```cpp
namespace SmallWindows { 
  typedef LPCTSTR CursorType; 

  class Cursor { 
    public: 
      static const CursorType Normal; 
      static const CursorType Arrow; 
      static const CursorType ArrowHourGlass; 
      static const CursorType Crosshair; 
      static const CursorType Hand; 
      static const CursorType ArrowQuestionMark; 
      static const CursorType IBeam; 
      static const CursorType SlashedCircle; 
      static const CursorType SizeAll; 
      static const CursorType SizeNorthEastSouthWest; 
      static const CursorType SizeNorthSouth; 
      static const CursorType SizeNorthWestSouthEast; 
      static const CursorType SizeWestEast; 
      static const CursorType VerticalArrow; 
      static const CursorType HourGlass; 

      static void Set(CursorType cursor); 
   }; 
}; 

```

**Cursor.cpp**

```cpp
#include "SmallWindows.h" 

namespace SmallWindows { 
  const CursorType Cursor::Normal = IDC_ARROW; 
  const CursorType Cursor::Arrow = IDC_ARROW; 
  const CursorType Cursor::ArrowHourGlass = IDC_APPSTARTING; 
  const CursorType Cursor::Crosshair = IDC_CROSS; 
  const CursorType Cursor::Hand = IDC_HAND; 
  const CursorType Cursor::ArrowQuestionMark = IDC_HELP; 
  const CursorType Cursor::IBeam = IDC_IBEAM; 
  const CursorType Cursor::SlashedCircle = IDC_NO; 
  const CursorType Cursor::SizeAll = IDC_SIZEALL; 
  const CursorType Cursor::SizeNorthEastSouthWest = IDC_SIZENESW; 
  const CursorType Cursor::SizeNorthSouth = IDC_SIZENS; 
  const CursorType Cursor::SizeNorthWestSouthEast = IDC_SIZENWSE; 
  const CursorType Cursor::SizeWestEast = IDC_SIZEWE; 
  const CursorType Cursor::VerticalArrow = IDC_UPARROW; 
  const CursorType Cursor::HourGlass = IDC_WAIT; 

```

`Set`方法通过调用 Win32 API 函数`LoadCursor`和`SetCursor`来设置光标：

```cpp
  void Cursor::Set(CursorType cursor) { 
    ::SetCursor(::LoadCursor(nullptr, cursor)); 
  } 
}; 

```

# DynamicList 类

`DynamicList`类可以被视为 C++标准类`list`和`vector`的更高级版本。它动态地改变其大小：

**DynamicList.h**

```cpp
namespace SmallWindows { 
  template <class Type> 
  class DynamicList { 
    public: 

```

`IfFuncPtr`指针是一个函数原型，用于在测试（不更改）列表中的值时使用。它接受一个常量值和一个`void`指针，并返回一个`Boolean`值。`DoFuncPtr`用于更改列表中的值，并接受一个（非常量）值和一个`void`指针。这些`void`指针由调用方法传递；它们包含额外的信息：

```cpp
      typedef bool (*IfFuncPtr)(const Type& value, void* voidPtr); 
      typedef void (*DoFuncPtr)(Type& value, void* voidPtr); 

```

列表可以通过另一个列表初始化和赋值。默认构造函数创建一个空列表，析构函数则释放列表的内存：

```cpp
      DynamicList(); 
      DynamicList(const DynamicList& list); 
      DynamicList& operator=(const DynamicList& list); 
      ~DynamicList(); 

```

`Empty`函数如果列表为空则返回`true`，`Size`返回列表中的值的数量，`Clear`移除列表中的每个值，而`IndexOf`返回给定值的零基于索引，如果没有这样的值在列表中，则返回负一：

```cpp
      bool Empty() const; 
      int Size() const; 
      void Clear(); 
      int IndexOf(Type& value) const; 

```

`begin`和`end`方法返回列表的开始和结束指针。它们被包含进来是为了使列表可以通过`for`语句迭代：

```cpp
      Type* begin(); 
      const Type* begin() const; 
      Type* end(); 
      const Type* end() const; 

```

索引方法检查或修改列表中给定零基于索引的值：

```cpp
      Type operator[](int index) const; 
      Type& operator[](int index); 

```

`Front`和`Back`方法通过调用之前提到的索引方法来检查和修改列表的第一个和最后一个值：

```cpp
      Type Front() const {return (*this)[0];} 
      Type& Front() {return (*this)[0];} 
      Type Back() const {return (*this)[size - 1];} 
      Type& Back() {return (*this)[size - 1];} 

```

`PushFront`和`PushBack`方法在列表的开始或结束处添加一个值或一个列表，而`Insert`在指定的索引处插入一个值或一个列表：

```cpp
      void PushBack(const Type& value); 
      void PushBack(const DynamicList& list); 
      void PushFront(const Type& value); 
      void PushFront(const DynamicList& list); 
      void Insert(int index, const Type& value); 
      void Insert(int index, const DynamicList& list); 

```

`Erase` 函数删除给定索引处的值，而 `Remove` 删除从 `firstIndex` 到 `lastIndex`（包含）的列表，如果 `lastIndex` 为负一，则删除列表的末尾。如果 `firstIndex` 为零且 `lastIndex` 为负一，则删除整个列表。由于 `Remove` 中的 `lastIndex` 是默认参数，因此方法已被赋予不同的名称。给方法赋予相同的名称将违反重载规则：

```cpp
      void Erase(int deleteIndex); 
      void Remove(int firstIndex = 0, int lastIndex = -1); 

```

`Copy` 函数将 `firstIndex` 到 `lastIndex`（包含）的列表复制到 `copyList` 或 `lastIndex` 为负一时的列表的其余部分。如果 `firstIndex` 为零且 `lastIndex` 为负一，则整个列表被复制：

```cpp
      void Copy(DynamicList& copyList, int firstIndex = 0, 
                int lastIndex = -1) const; 

```

`AnyOf` 函数如果至少有一个值满足 `ifFuncPtr`，则返回 `true`。也就是说，如果 `ifFuncPtr` 在以值作为参数调用时返回 `true`，则 `AllOf` 函数返回 `true` 如果所有值都满足 `ifFuncPtr`：

```cpp
      bool AnyOf(IfFuncPtr ifFuncPtr, void* ifVoidPtr = nullptr) 
                 const; 
      bool AllOf(IfFuncPtr ifFuncPtr, void* ifVoidPtr = nullptr) 
                 const; 

```

`FirstOf` 和 `LastOf` 方法将 `value` 参数设置为满足 `ifFuncPtr` 的第一个和最后一个值；如果没有这样的值，则返回 `false`：

```cpp
      bool FirstOf(IfFuncPtr ifFuncPtr, Type& value, 
                   void* ifVoidPtr = nullptr) const; 
      bool LastOf(IfFuncPtr ifFuncPtr, Type& value, 
                  void* ifVoidPtr = nullptr) const; 

```

`Apply` 方法对列表中的所有值调用 `doFuncPtr`，而 `ApplyIf` 方法对列表中满足 `ifFuncPtr` 的每个值调用 `doFuncPtr`：

```cpp
      void Apply(DoFuncPtr doFuncPtr, void* ifVoidPtr = nullptr); 
      void ApplyIf(IfFuncPtr ifFuncPtr, DoFuncPtr doFuncPtr, 
                   void* ifVoidPtr = nullptr, 
                   void* doVoidPtr = nullptr); 

```

`CopyIf` 方法将列表中满足 `ifFuncPtr` 的每个值复制到 `copyList`。`RemoveIf` 移除满足 `ifFuncPtr` 的值：

```cpp
      void CopyIf(IfFuncPtr ifFuncPtr, DynamicList& copyList, 
                  void* ifVoidPtr = nullptr) const; 
      void RemoveIf(IfFuncPtr ifFuncPtr, 
                    void* ifVoidPtr = nullptr); 

```

`ApplyRemoveIf` 方法对满足 `ifFuncPtr` 的每个值调用 `doFuncPtr`，然后移除它们。将函数应用于要删除的值可能看起来很奇怪。然而，当删除动态分配的值时，这在其中 `doFuncPtr` 在从列表中删除每个值之前释放每个值的内存时非常有用。简单地调用 `ApplyIf` 和 `RemoveIf` 是不起作用的。当值被 `ApplyIf` 删除后，它们不能成为 `RemoveIf` 中 `ifFuncPtr` 调用的参数：

```cpp
      void ApplyRemoveIf(IfFuncPtr ifFuncPtr, DoFuncPtr doFuncPtr, 
                         void* ifVoidPtr=nullptr, 
                         void* doVoidPtr=nullptr); 

```

大小是列表中的值的数量，缓冲区持有这些值本身。缓冲区的大小是动态的，当向列表中添加或从列表中删除值时，它会改变。当列表为空时，缓冲区指针为空：

```cpp
    private: 
      int size; 
      Type* buffer; 
  }; 

  template <class Type> 
  DynamicList<Type>::DynamicList() 
   :size(0), 
    buffer(nullptr) { 
    // Empty. 
  } 

```

默认构造函数和赋值运算符遍历给定的列表并复制每个值。为此，类型必须支持赋值运算符，除了数组之外的所有类型都支持：

```cpp
  template <class Type> 
  DynamicList<Type>::DynamicList(const DynamicList& list) 
   :size(list.size), 
    buffer(new Type[list.size]) { 
    assert(buffer != nullptr);  
    for (int index = 0; index < size; ++index) { 
      buffer[index] = list.buffer[index]; 
    } 
  } 

```

在赋值运算符中，我们首先删除缓冲区，因为它可能包含值。如果列表为空，缓冲区指针为空，删除运算符不执行任何操作：

```cpp
  template <class Type> 
  DynamicList<Type>& DynamicList<Type>::operator= 
                                      (const DynamicList& list) { 
    if (this != &list) { 
      delete[] buffer; 
      size = list.size; 
      assert((buffer = new Type[size]) != nullptr); 

      for (int index = 0; index < size; ++index) { 
        buffer[index] = list.buffer[index]; 
      } 
    } 

    return *this; 
  } 

```

析构函数简单地删除缓冲区。再次强调，如果列表为空，缓冲区指针为空，删除运算符不执行任何操作：

```cpp
  template <class Type> 
  DynamicList<Type>::~DynamicList() { 
    delete[] buffer; 
  } 

  template <class Type> 
  bool DynamicList<Type>::Empty() const { 
    return (size == 0); 
  } 

  template <class Type> 
  int DynamicList<Type>::Size() const { 
    return size; 
  } 

```

`Clear` 方法将大小设置为零并将缓冲区设置为空：

```cpp
  template <class Type> 
  void DynamicList<Type>::Clear() { 
    size = 0; 
    delete[] buffer; 
    buffer = nullptr; 
  } 

```

`IndexOf` 方法遍历列表并返回找到的值的索引，如果没有这样的值，则返回负一：

```cpp
  template <class Type> 
  int DynamicList<Type>::IndexOf(Type& value) const { 
    for (int index = 0; index < size; ++index) { 

      if (buffer[index] == value) { 
        return index; 
      } 
    } 

    return -1; 
  } 

```

`begin` 方法返回列表中第一个值的地址：

```cpp
  template <class Type> 
  Type* DynamicList<Type>::begin() { 
    return &buffer[0]; 
  } 

  template <class Type> 
  const Type* DynamicList<Type>::begin() const { 
    return &buffer[0]; 
  } 

```

`end` 方法返回列表中最后一个值之后的一步地址，这是 C++ 中列表迭代器的惯例：

```cpp
  template <class Type> 
  Type* DynamicList<Type>::end() { 
    return &buffer[size]; 
  } 

  template <class Type> 
  const Type* DynamicList<Type>::end() const { 
    return &buffer[size]; 
  } 

```

如果索引超出列表范围，则发生断言：

```cpp
  template <class Type> 
  Type DynamicList<Type>::operator[](int index) const { 
    assert((index >= 0) && (index < size)); 
    return buffer[index]; 
  } 

  template <class Type> 
  Type& DynamicList<Type>::operator[](int index) { 
    assert((index >= 0) && (index < size)); 
    return buffer[index]; 
  } 

```

当在原始列表的末尾添加值时，我们需要分配一个包含一个额外值的新的列表，并将新值添加到末尾：

```cpp
  template <class Type> 
  void DynamicList<Type>::PushBack(const Type& value) { 
    Type* newBuffer = new Type[size + 1]; 
    assert(newBuffer != nullptr); 

    for (int index = 0; index < size; ++index) { 
      newBuffer[index] = buffer[index]; 
    } 

    newBuffer[size++] = value; 
    delete[] buffer; 
    buffer = newBuffer; 
  } 

```

当在原始列表的末尾添加新列表时，我们需要分配一个大小为原始列表和新列表之和的新列表，并将原始列表中的值复制到新列表中：

```cpp
  template <class Type> 
  void DynamicList<Type>::PushBack(const DynamicList& list) { 
    Type* newBuffer = new Type[size + list.size]; 
    assert(newBuffer != nullptr); 

    for (int index = 0; index < size; ++index) { 
      newBuffer[index] = buffer[index]; 
    } 

    for (int index = 0; index < list.size; ++index) { 
      newBuffer[size + index] = list.buffer[index]; 
    } 

    delete[] buffer; 
    buffer = newBuffer; 
    size += list.size; 
  } 

```

当在列表的开头插入新值时，我们需要将原始列表中的所有值向前移动一步，为新值腾出空间：

```cpp
  template <class Type> 
  void DynamicList<Type>::PushFront(const Type& value) { 
    Type* newBuffer = new Type[size + 1]; 
    assert(newBuffer != nullptr); 
    newBuffer[0] = value; 

    for (int index = 0; index < size; ++index) { 
      newBuffer[index + 1] = buffer[index]; 
    } 

    delete[] buffer; 
    buffer = newBuffer; 
    ++size; 
  } 

```

当在列表的开头插入新列表时，我们需要复制其所有值以及与新列表大小相对应的步数，为新值腾出空间：

```cpp
  template <class Type> 
  void DynamicList<Type>::PushFront(const DynamicList& list) { 
    Type* newBuffer = new Type[size + list.size]; 
    assert(newBuffer != nullptr); 

```

我们移动原始列表的值以腾出空间为新列表：

```cpp
    for (int index = 0; index < list.size; ++index) { 
      newBuffer[index] = list.buffer[index]; 
    } 

```

当我们为新列表腾出空间后，我们将它复制到原始列表的开头：

```cpp
    for (int index = 0; index < size; ++index) { 
      newBuffer[index + list.size] = buffer[index]; 
    } 

    delete[] buffer; 
    buffer = newBuffer; 
    size += list.size; 
  } 

```

`Insert` 方法的工作方式与 `PushFront` 类似。我们需要分配一个新列表，并将原始列表中的值复制到新列表中腾出空间，然后将新值复制到原始列表中：

```cpp
  template <class Type> 
  void DynamicList<Type>::Insert(int insertIndex, 
                                 const Type& value) { 
    assert((insertIndex >= 0) && (insertIndex <= size)); 
    Type* newBuffer = new Type[size + 1]; 
    assert(newBuffer != nullptr); 

    for (int index = 0; index < insertIndex; ++index) { 
      newBuffer[index] = buffer[index]; 
    } 

    newBuffer[insertIndex] = value; 

    for (int index = 0; index < (size - insertIndex); ++index) { 
      newBuffer[insertIndex + index + 1] = 
        buffer[insertIndex + index]; 
    } 

    delete[] buffer; 
    buffer = newBuffer; 
    ++size; 
  } 

  template <class Type> 
  void DynamicList<Type>::Insert(int insertIndex, 
                                 const DynamicList& list){ 
    assert((insertIndex >= 0) && (insertIndex <= size)); 
    Type* newBuffer = new Type[size + list.size]; 
    assert(newBuffer != nullptr); 

    for (int index = 0; index < insertIndex; ++index) { 
      newBuffer[index] = buffer[index]; 
    } 

    for (int index = 0; index < list.size; ++index) { 
      newBuffer[insertIndex + index] = list.buffer[index]; 
    } 

    for (int index = 0; index < (size - insertIndex); ++index) { 
      newBuffer[insertIndex + index + list.size] = 
        buffer[insertIndex + index]; 
    } 

    delete[] buffer; 
    buffer = newBuffer; 
    size += list.size; 
  } 

```

当在列表中删除值时，我们分配一个较小的新的列表，并将剩余的值复制到该列表中：

```cpp
  template <class Type> 
  void DynamicList<Type>::Erase(int eraseIndex) { 
    assert((eraseIndex >= 0) && (eraseIndex < size)); 
    Type* newBuffer = new Type[size - 1]; 
    assert(newBuffer != nullptr); 

```

首先，我们复制删除索引之前的值：

```cpp
    for (int index = 0; index < eraseIndex; ++index) { 
      newBuffer[index] = buffer[index]; 
    } 

```

然后，我们复制删除索引之后的值：

```cpp
    for (int index = 0; index < (size - (eraseIndex + 1)); 
         ++index) { 
      newBuffer[eraseIndex + index] = 
        buffer[eraseIndex + index + 1]; 
    } 

    delete[] buffer; 
    buffer = newBuffer; 
    --size; 
  } 

```

`Remove` 方法的工作方式与 `Delete` 相同；区别在于可以从列表中删除多个值；`removeSize` 保存要删除的值的数量：

```cpp
  template <class Type> 
  void DynamicList<Type>::Remove(int firstIndex /* = 0 */,  
                                 int lastIndex /* = -1 */) { 
    if (lastIndex == -1) { 
      lastIndex = size - 1; 
    } 

    assert((firstIndex >= 0) && (firstIndex < size)); 
    assert((lastIndex >= 0) && (lastIndex < size)); 
    assert(firstIndex <= lastIndex); 

    int removeSize = lastIndex - firstIndex + 1; 
    Type* newBuffer = new Type[size - removeSize]; 
    assert(newBuffer != nullptr);  
    for (int index = 0; index < firstIndex; ++index) { 
      newBuffer[index] = buffer[index]; 
    }  

    for (int index = 0; 
         index < (size - (firstIndex + removeSize)); ++index){ 
      newBuffer[firstIndex + index] = 
        buffer[firstIndex + index + removeSize]; 
    }  

    delete[] buffer; 
    buffer = newBuffer; 
    size -= removeSize; 
  } 

```

`Copy` 方法简单地为要复制的每个值调用 `PushBack`：

```cpp
  template <class Type> 
  void DynamicList<Type>::Copy(DynamicList& copyList,
                               int firstIndex/* =0 */,
                               int lastIndex /* = -1 */) const {
    if (lastIndex == -1) { 
      lastIndex = size - 1; 
    } 

    assert((firstIndex >= 0) && (firstIndex < size)); 
    assert((lastIndex >= 0) && (lastIndex < size)); 
    assert(firstIndex <= lastIndex); 

    for (int index = firstIndex; index <= lastIndex; ++index) { 
      copyList.PushBack(buffer[index]); 
    } 
  } 

```

`AnyOf` 方法遍历列表，如果至少有一个值满足函数，则返回 `true`：

```cpp
  template <class Type> 
  bool DynamicList<Type>::AnyOf(IfFuncPtr ifFuncPtr, 
                          void* ifVoidPtr /* = nullptr */) const { 
    for (int index = 0; index < size; ++index) { 
      if (ifFuncPtr(buffer[index], ifVoidPtr)) { 
        return true; 
      } 
    } 

    return false; 
  } 

```

`AllOf` 方法遍历列表，如果至少有一个值不满足函数，则返回 `false`：

```cpp
  template <class Type> 
  bool DynamicList<Type>::AllOf(IfFuncPtr ifFuncPtr, 
                          void* ifVoidPtr /* = nullptr */) const { 
    for (int index = 0; index < size; ++index) { 
      if (!ifFuncPtr(buffer[index], ifVoidPtr)) { 
        return false; 
      } 
    } 

    return true; 
  } 

```

`FirstOf` 方法以与 `FirstOf` 相同的方式查找列表中满足函数的第一个值，将其复制到值参数中，并返回 `true`。如果没有找到满足函数的任何值，则返回 `false`：

```cpp
  template <class Type> 
  bool DynamicList<Type>::FirstOf(IfFuncPtr ifFuncPtr, 
              Type& value, void* ifVoidPtr /* = nullptr */) const{ 
    for (int index = 0; index < size; ++index) { 
      if (ifFuncPtr(buffer[index], ifVoidPtr)) { 
        value = buffer[index]; 
        return true; 
      } 
    } 

    return false; 
  } 

```

`LastOf` 方法以与 `FirstOf` 相同的方式查找列表中满足函数的最后一个值；区别在于搜索是向后的：

```cpp
  template <class Type> 
  bool DynamicList<Type>::LastOf(IfFuncPtr ifFuncPtr, Type& value, 
                          void* ifVoidPtr /* = nullptr */) const { 
    for (int index = (size - 1); index >= 0; --index) { 
      if (ifFuncPtr(buffer[index], ifVoidPtr)) { 
        value = buffer[index]; 
        return true; 
      } 
    } 

    return false; 
  } 

```

`Apply` 方法遍历列表，并对每个值调用 `doFuncPtr`，值可能会被修改（实际上，`Apply` 的目的就是修改值），因为 `doFuncPtr` 的参数不是常量：

```cpp
  template <class Type> 
  void DynamicList<Type>::Apply(DoFuncPtr doFuncPtr, 
                                void* doVoidPtr /* = nullptr */) { 
    for (int index = 0; index < size; ++index) { 
      doFuncPtr(buffer[index], doVoidPtr); 
    } 
  } 

```

`ApplyIf` 方法遍历列表，并对满足 `ifFuncPtr` 的每个值调用 `doFuncPtr`：

```cpp
  template <class Type> 
  void DynamicList<Type>::ApplyIf(IfFuncPtr ifFuncPtr, 
         DoFuncPtr doFuncPtr, void* ifVoidPtr /* = nullptr */, 
         void* doVoidPtr /* = nullptr */){ 
    for (int index = 0; index < size; ++index) { 
      if (ifFuncPtr(buffer[index], ifVoidPtr)) { 
        doFuncPtr(buffer[index], doVoidPtr); 
      } 
    } 
  } 

```

`CopyIf` 方法通过为每个满足 `ifFuncPtr` 的值调用 `PushBack` 将每个值复制到 `copyList` 中：

```cpp
  template <class Type> 
  void DynamicList<Type>::CopyIf(IfFuncPtr ifFuncPtr, 
                          DynamicList& copyList, 
                          void* ifVoidPtr /* = nullptr */) const { 
    for (int index = 0; index < size; ++index) { 
      if (ifFuncPtr(buffer[index], ifVoidPtr)) { 
        copyList.PushBack(buffer[index]); 
      } 
    } 
  } 

```

`RemoveIf` 方法通过为每个值调用 `Delete` 来删除满足 `ifFuncPtr` 的每个值：

```cpp
  template <class Type> 
  void DynamicList<Type>::RemoveIf(IfFuncPtr ifFuncPtr, 
                                void* ifVoidPtr /* = nullptr */) { 
    for (int index = 0; index < size; ++index) { 
      if (ifFuncPtr(buffer[index], ifVoidPtr)) { 
        Erase(index--); 
      } 
    } 
  } 

```

`ApplyRemoveIf` 方法将 `doFuncPtr` 应用到满足 `ifFuncPtr` 的每个值。我们不能简单地调用 `Apply` 和 `RemoveIf`，因为 `doFuncPtr` 可能会释放 `Apply` 中的值，而 `RemoveIf` 中的 `ifFuncPtr` 在调用已删除的值时不会工作。相反，我们调用 `doFuncPtr` 并在调用后立即调用 `Erase`。这样，在调用 `doFuncPtr` 之后不会访问值：

```cpp
  template <class Type> 
  void DynamicList<Type>::ApplyRemoveIf(IfFuncPtr ifFuncPtr, 
         DoFuncPtr doFuncPtr, void* ifVoidPtr /* = nullptr */, 
         void* doVoidPtr /* = nullptr */) { 
    for (int index = 0; index < size; ++index) { 
      if (ifFuncPtr(buffer[index], ifVoidPtr)) { 
        doFuncPtr(buffer[index], doVoidPtr); 
        Erase(index--); 
      } 
    } 
  } 
}; 

```

# 树类

C++ 标准库包含一组用于数组、列表、向量、集合和映射的容器类。然而，没有用于树结构的类。因此，`Tree` 类被添加到 Small Windows 中。树由一组节点组成，其中之一是根节点。每个节点持有一个（可能为空）的子节点列表：

**Tree.h**

```cpp
namespace SmallWindows { 
  template <class NodeType> 
  class Tree { 
    public: 
      Tree(); 
      Tree(NodeType nodeValue, 
           initializer_list<Tree<NodeType>*> childList = {}); 
      Tree(const Tree& tree); 
      Tree& operator=(const Tree& tree); 
      void Init(const Tree& tree); 
      ~Tree(); 

```

树可以被写入和从文件流或剪贴板读取：

```cpp
      bool WriteTreeToStream(ostream& outStream) const; 
      bool ReadTreeFromStream(istream& inStream); 
      void WriteTreeToClipboard(InfoList& infoList) const; 
      void ReadTreeFromClipboard(InfoList& infoList); 

```

每个树节点持有一个值，该值由常量方法检查并由非常量方法修改：

```cpp
      NodeType NodeValue() const {return nodeValue;} 
      NodeType& NodeValue() {return nodeValue;} 

```

树节点还持有一个子节点列表，该列表由常量方法检查，并由非常量方法修改：

```cpp
      const DynamicList<Tree*>& ChildList() const 
                                            {return childList;} 
      DynamicList<Tree*>& ChildList() {return childList;} 

     private: 
      NodeType nodeValue; 
     DynamicList<Tree*> childList; 
  }; 

  template <class NodeType> 
  Tree<NodeType>::Tree() { 
    // Empty. 
  } 

```

子节点列表是一个树节点的初始化列表；默认情况下它是空的：

```cpp
  template <class NodeType>   
  Tree<NodeType>::Tree(NodeType nodeValue, 
           initializer_list<Tree<NodeType>*> childList /* = {} */) 
   :nodeValue(nodeValue) { 
    for (Tree<NodeType>* childNodePtr : childList) { 
      this->childList.PushBack(childNodePtr); 
    } 
  } 

```

默认构造函数和赋值运算符调用 `Init` 来执行树的实际初始化：

```cpp
  template <class NodeType> 
  Tree<NodeType>::Tree(const Tree& tree) { 
    Init(tree); 
  } 

  template <class NodeType> 
  Tree<NodeType>& Tree<NodeType>::operator=(const Tree& tree) { 
    if (this != &tree) { 
      Init(tree); 
    } 

    return *this; 
  } 

  template <class NodeType> 
  void Tree<NodeType>::Init(const Tree& tree) { 
    nodeValue = tree.nodeValue; 

    for (Tree* childPtr : tree.childList) { 
      Tree* childClonePtr = new Tree(*childPtr); 
      assert(childClonePtr != nullptr); 
      childList.PushBack(childClonePtr); 
    } 
  } 

```

析构函数递归地删除子节点：

```cpp
  template <class NodeType> 
  Tree<NodeType>::~Tree() { 
    for (Tree* childPtr : childList) { 
      delete childPtr; 
    } 
  } 

```

`WriteTreeToStream` 方法将节点值和子节点数量写入流，然后对每个子节点递归调用自身：

```cpp
  template <class NodeType> 
  bool Tree<NodeType>::WriteTreeToStream(ostream& outStream)const{ 
    nodeValue.WriteTreeNodeToStream(outStream);  

    int childListSize = childList.Size(); 
    outStream.write((char*) &childListSize, sizeof childListSize);  

    for (Tree* childPtr : childList) { 
      childPtr->WriteTreeToStream(outStream); 
    } 

    return ((bool) outStream); 
  } 

```

`ReadTreeFromStream` 方法从流中读取节点值和子节点数量，创建子节点，并对每个子节点递归调用自身：

```cpp
  template <class NodeType> 
  bool Tree<NodeType>::ReadTreeFromStream(istream& inStream) { 
    nodeValue.ReadTreeNodeFromStream(inStream);  

    int childListSize; 
    inStream.read((char*) &childListSize, sizeof childListSize); 

    for (int count = 0; count < childListSize; ++count) { 
      Tree* childPtr = new Tree(); 
      assert(childPtr != nullptr); 
      childPtr->ReadTreeFromStream(inStream); 
      childList.PushBack(childPtr); 
    } 

    return ((bool) inStream); 
  } 

```

`WriteTreeToClipboard` 和 `ReadTreeFromClipboard` 方法的工作方式与 `WriteTreeToStream` 和 `ReadTreeFromStream` 类似：

```cpp
  template <class NodeType> 
  void Tree<NodeType>::WriteTreeToClipboard(InfoList& infoList) 
                                            const { 
    nodeValue.WriteTreeNodeToClipboard(infoList); 

    infoList.AddValue<int>( childList.Size()); 

    for (Tree* childPtr : childList) { 
      childPtr->WriteTreeToClipboard(infoList); 
    } 
  } 

  template <class NodeType> 
  void Tree<NodeType>::ReadTreeFromClipboard(InfoList& infoList) { 
    nodeValue.ReadTreeNodeFromClipboard(infoList); 

    int childListSize; 
    infoList.GetValue<int>(childListSize); 

    for (int count = 0; count < childListSize; ++count) { 
      Tree* childPtr = new Tree(); 
      assert(childPtr != nullptr); 
      childPtr->ReadTreeFromClipboard(infoList); 
      childList.PushBack(childPtr); 
    } 
  } 
}; 

```

# InfoList 类

`InfoList` 类是一个具有模板方法的辅助类，它将信息存储在字符列表中；信息可以被添加和提取；或者写入，或从缓冲区读取：

**InfoList** **.h**

```cpp
namespace SmallWindows { 
  class InfoList { 
    public: 
      template <class AlignType> void Align(); 
      template <class ListType> 
        void AddValue(const ListType value); 
      template <class ListType> 
        void PeekValue(ListType& value, int index); 
      template <class ListType> void GetValue(ListType& value); 
      template <class CharType> 
        void AddString(basic_string<CharType> text); 
      template <class CharType> 
        basic_string<CharType> GetString(); 
      void FromBuffer(const void* voidBuffer, int size); 
      void ToBuffer(void* voidBuffer); 
      int Size() const {return list.Size();} 

    private: 
      DynamicList<char> list; 
  }; 

```

`Align` 函数逐字节增加列表，直到对齐类型的大小是列表大小的除数：

```cpp
  template <class AlignType> 
  void InfoList::Align() { 
    int size = sizeof(AlignType); 

    while ((list.Size() % size) > 0) { 
      list.PushBack(0); 
    } 
  } 

```

`AddValue` 函数通过逐字节将模板类型的值添加到列表中来添加值，而 `GetValue` 通过逐字节从列表中提取值来获取列表开头的值：

```cpp
  template <class ListType> 
  void InfoList::AddValue(const ListType value) { 
    int size = sizeof(ListType); 
    const char* buffer = (char*) &value; 

    for (int count = 0; count < size; ++count) { 
      list.PushBack(*(buffer++)); 
    } 
  } 

  template <class ListType> 
  void InfoList::PeekValue(ListType& value, int index) { 
    int size = sizeof(ListType); 
    char* buffer = (char*) &value; 

    for (int count = 0; count < size; ++count) { 
      *(buffer++) = list[index + count]; 
    } 
  } 

  template <class ListType> 
  void InfoList::GetValue(ListType& value) { 
    int size = sizeof(ListType); 
    char* buffer = (char*) &value; 

    for (int count = 0; count < size; ++count) { 
      *(buffer++) = list.Front(); 
      list.Erase(0); 
    } 
  } 

```

`AddString` 函数将文本的字符添加到列表中，并附带一个终止零字符，而 `GetString` 从列表中读取文本，直到遇到终止零字符：

```cpp
  template <class CharType> 
  void InfoList::AddString(basic_string<CharType> text) { 
    for (CharType c : text) { 
      AddValue<CharType>(c); 
    } 

    AddValue<CharType>(0); 
  } 

  template <class CharType> 
  basic_string<CharType> InfoList::GetString() { 
    bacic_string<CharType> text; 

    CharType c, zero = (CharType) 0; 
    while ((c = GetValue<CharType>()) != zero) { 
      text.append(c); 
    } 

    return text; 
  } 
}; 

```

**InfoList.cpp**

```cpp
#include "SmallWindows.h" 

```

`FromBuffer` 函数将缓冲区的每个字节添加到列表中，而 `ToBuffer` 从列表中提取并复制每个字节到缓冲区：

```cpp
void InfoList::FromBuffer(const void* voidBuffer, int size) { 
  const char* charBuffer = (const char*) voidBuffer; 

  for (int count = 0; count < size; ++count) { 
    list.PushBack(*(charBuffer++)); 
  } 
} 

void InfoList::ToBuffer(void* voidBuffer) { 
  char* charBuffer = (char*) voidBuffer; 

  for (char c : list) { 
    *(charBuffer++) = c; 
  } 
} 

```

# 字符串

有少量字符串函数：

+   `CharPtrToGenericString`：这接受文本作为一个 `char` 字符指针，并以一个通用的 `String` 对象返回相同的文本。请记住，`String` 类持有 `TCHAR` 类型的值，其中许多是 `char` 或 `wchar_t`，这取决于系统设置。

+   `Split`：这接受一个字符串并返回一个包含文本空格分隔单词的字符串列表。

+   `IsNumeric`：如果文本包含一个数值，则此函数返回`true`。

+   `Trim`：这会移除文本开头和结尾的空格。

+   `ReplaceAll`：这会将一个字符串替换为另一个字符串。

+   `WriteStringToStream` 和 `ReadStringFromStream`：这些函数将字符串写入和从流中读取。

+   `StartsWith` 和 `EndsWith`：如果文本以子文本开头或结尾，则这些函数返回`true`。

**String.h**

```cpp
namespace SmallWindows { 
  extern String CharPtrToGenericString(char* text); 
  extern vector<String> Split(String text, TCHAR c = TEXT(' ')); 
  extern bool IsNumeric(String text); 
  extern String Trim(String text); 
  void ReplaceAll(String& text, String from, String to); 
  extern bool WriteStringToStream(const String& text, 
                                  ostream& outStream); 
  extern bool ReadStringFromStream(String& text, 
                                   istream& inStream); 
  extern bool StartsWith(String text, String part); 
  extern bool EndsWith(String text, String part); 
}; 

```

**String.cpp**

```cpp
#include "SmallWindows.h" 

namespace SmallWindows { 
  String CharPtrToGenericString(char* text) { 
    String result; 

    for (int index = 0; text[index] != '\0'; ++index) { 
      result += (TCHAR) text[index]; 
    } 

    return result; 
  } 

  vector<String> Split(String text, TCHAR c /* = TEXT(' ') */) { 
    vector<String> list; 
    int spaceIndex = -1, size = text.size(); 

    for (int index = 0; index < size; ++index) { 
      if (text[index] == c) { 
        String word = 
          text.substr(spaceIndex + 1, index - spaceIndex - 1); 
        list.push_back(word); 
        spaceIndex = index; 
      } 
    } 

    String lastWord = text.substr(spaceIndex + 1); 
    list.push_back(lastWord); 
    return list; 
  } 

```

`IsNumeric` 方法使用 `IStringStream` 方法读取字符串的值，并将读取的字符数与文本长度进行比较。如果读取了文本的所有字符，则文本将包含一个数值，并返回 `true`：

```cpp
      bool IsNumeric(String text) { 
    IStringStream stringStream(Trim(text)); 
    double value; 
    stringStream >> value; 
    return stringStream.eof(); 
  } 

  String Trim(String text) { 
    while (!text.empty() && isspace(text[0])) { 
      text.erase(0, 1); 
    } 

    while (!text.empty() && isspace(text[text.length() - 1])) { 
      text.erase(text.length() - 1, 1); 
    } 

    return text; 
  } 

  void ReplaceAll(String& text, String from, String to) { 
    int index, fromSize = from.size(); 

    while ((index = text.find(from)) != -1) { 
      text.erase(index, fromSize); 
      text.insert(index, to); 
    } 
  } 

  bool WriteStringToStream(const String& text,ostream& outStream){ 
    int size = text.size(); 
    outStream.write((char*) &size, sizeof size); 

    for (TCHAR tChar : text) { 
      outStream.write((char*) &tChar, sizeof tChar); 
    } 

    return ((bool) outStream); 
  } 

  bool ReadStringFromStream(String& text, istream& inStream) { 
    int size; 
    inStream.read((char*) &size, sizeof size); 

    for (int count = 0; count < size; ++count) { 
      TCHAR tChar; 
      inStream.read((char*) &tChar, sizeof tChar); 
      text.push_back(tChar); 
    } 

    return ((bool) inStream); 
  } 

  bool StartsWith(String text, String part) { 
    return (text.find(part) == 0); 
  } 

  bool EndsWith(String text, String part) { 
    int index = text.rfind(part), 
        difference = text.length() - part.length(); 
    return ((index != -1) && (index == difference)); 
  } 
}; 

```

# 摘要

在本章中，我们学习了 Small Windows 所使用的辅助类。在第十三章，*剪贴板、标准对话框和打印预览*中，我们将探讨注册表、剪贴板、标准对话框和打印预览。
