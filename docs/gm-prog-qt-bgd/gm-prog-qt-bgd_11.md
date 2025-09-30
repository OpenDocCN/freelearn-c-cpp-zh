# 附录 A. 突击测验答案

# 第三章，Qt GUI 编程

## 突击测验 – 建立信号-槽连接

| Q1 | 一个槽 |
| --- | --- |
| Q2 | `connect(sender, SIGNAL(toggled(bool)), receiver, SLOT(clear()));` 和 `connect(sender, &QPushButton::clicked, receiver, &QLineEdit::clear);` |

## 突击测验 – 使用小部件

| Q1 | `sizeHint` |
| --- | --- |
| Q2 | `QVariant` |
| Q3 | 它代表用户可以在程序中调用的功能。 |

# 第四章，Qt 核心基础

## 突击测验 – Qt 核心基础

| Q1 | `QString` |
| --- | --- |
| Q2 | `((25[0-5]&#124;2[0-4][0-9]&#124;[01]?[0-9][0-9]?)(\.&#124;$)){4}` |
| Q3 | XML |

# 第六章，图形视图

## 突击测验 – 掌握图形视图

| Q1 | 例如，你应该知道有一个 `QGraphicsSimpleTextItem`，你可以用它来绘制简单的文本，在这些情况下你不需要自己处理 `QPainter`。你还应该知道，如果你有一个包含粗体字符的更复杂的文本，你可以使用 `QGraphicsTextItem`，它能够处理富文本。 |
| --- | --- |
| Q2 | 这些问题的正确答案涉及不同系统的原点。 |
| Q3 | 注意，`QObject` 并不仅限于 "小部件" 的世界。你还可以用它与项目一起使用。 |
| Q4 | 正确答案的关键词是视差滚动。 |
| Q5 | 正确答案将考虑你如何控制缓存以及如何影响在请求更新时实际重绘的视图部分。 |

# 第七章，网络

## 突击测验 – 测试你的知识

| Q1 | `QNetworkAccessManager`、`QNetworkRequest` 和 `QNetworkReply`。 |
| --- | --- |
| Q2 | 必须使用 `QNetworkRequest::setRawHeader()` 并设置适当的 HTTP 头字段 "Range"。 |
| Q3 | `QUrlQuery` |
| Q4 | 必须使用 `deleteLater()` 而不是 `delete`。 |
| Q5 | 它们都继承自 `QAbstractSocket`，而 `QAbstractSocket` 继承自 `QIODevice`。`QIODevice` 本身也是 `QFile` 的基类。因此，文件和套接字的处理有很多共同之处。因此，你不必学习第二个（复杂的）API 只是为了与套接字通信。 |
| Q6 | `QUdpSocket` |

# 第八章，脚本

## 突击测验 – 脚本

| Q1 | `QScriptEngine::evaluate()` |
| --- | --- |
| Q2 | `QScriptValue` |
| Q3 | `PyValue` |
| Q4 | 它们包含函数调用中定义的所有变量，因此可以从脚本中修改一组变量，而不会影响全局环境（称为沙盒）。 |

# 第十一章，杂项和高级概念

## 突击测验 – 测试你的知识

| Q1 | 后缀是 `Reading`，例如，`QRotationReading`。 |
| --- | --- |
| Q2 | 命名为 `QSensorGestureRecognizer` 的类。 |
| Q3 | 这是 Qt 定位模块，你可以通过在项目文件中添加 `QT += positioning` 来激活它。 |
| Q4 | 必须重载 `QDebug& operator<<()` |
| Q5 | 只有在程序以调试模式构建时，如果 `condition` 为 `false`，它才会终止程序的执行。 |
