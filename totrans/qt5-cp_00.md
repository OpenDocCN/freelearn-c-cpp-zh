# 前言

Qt 已被开发为一个跨平台框架，并且多年来一直免费提供给公众。它主要用于构建 GUI 应用程序。它还提供了数千个 API，以简化开发。

Qt 5，Qt 的最新主要版本，再次证明是最受欢迎的跨平台工具包。凭借所有这些平台无关的类和函数，您只需编写一次代码，然后就可以让它在任何地方运行。

除了传统的强大 C++之外，Qt Quick 2，一个更加成熟的版本，可以帮助网页开发者开发动态且可靠的应用程序，因为 QML 与 JavaScript 非常相似。

# 本书涵盖内容

第一章, *创建您的第一个 Qt 应用程序*，带您了解 Qt 的基本概念，如信号和槽，并帮助您创建第一个 Qt 和 Qt Quick 应用程序。

第二章, *构建一个漂亮的跨平台时钟*，教您如何读取和写入配置以及处理跨平台开发。

第三章, *使用 Qt Quick 制作 RSS 阅读器*，演示了如何在 QML 中开发一个时尚的 RSS 阅读器，QML 是一种与 JavaScript 非常相似的脚本语言。

第四章, *控制摄像头和拍照*，展示了如何通过 Qt API 访问摄像头设备并利用状态栏和菜单栏。

第五章, *使用插件扩展绘图应用程序*，教您如何通过使用绘图应用程序作为示例来使应用程序可扩展并编写插件。

第六章, *使用进度条利用 Qt 的网络模块以及学习 Qt 中的线程编程*，展示了如何使用进度条来利用 Qt 的网络模块，以及学习 Qt 中的线程编程。

第七章, *解析 JSON 和 XML 文档以使用在线 API*，教您如何在 Qt/C++和 Qt Quick/QML 中解析 JSON 和 XML 文档，这对于从在线 API 获取数据至关重要。

第八章, *使您的 Qt 应用程序支持其他语言*，演示了如何制作国际化应用程序，使用 Qt Linguist 翻译字符串，然后动态加载翻译文件。

第九章, *在其他设备上部署应用程序*，展示了如何打包并使您的应用程序在 Windows、Linux 和 Android 上可重新分发。

第十章, *遇到这些问题时不要慌张*，为您提供了在 Qt 和 Qt Quick 应用程序开发过程中遇到的一些常见问题的解决方案和建议，并展示了如何调试 Qt 和 Qt Quick 应用程序。

# 您需要为本书准备什么

Qt 是跨平台的，这意味着您几乎可以在所有操作系统上使用它，包括 Windows、Linux、BSD 和 Mac OS X。硬件要求如下：

+   一台计算机（PC 或 Macintosh）

+   一台网络摄像头或已连接的摄像头设备

+   可用的互联网连接

不需要安卓手机或平板电脑，但推荐使用，以便您可以在真实的安卓设备上测试应用程序。

本书提到的所有软件，包括 Qt 本身，都是免费的，可以从互联网上下载。

# 本书面向的对象

如果您是一位寻找真正跨平台的 GUI 框架的程序员，希望通过避免不同平台之间不兼容的问题来节省时间，并使用 Qt 5 为多个目标构建应用程序，那么这本书绝对是为您准备的。假设您具有基本的 C++ 编程经验。

# 术语

在本书中，您将找到许多文本样式，用于区分不同类型的信息。以下是一些这些样式的示例及其含义的解释。

文本中的代码单词、数据库表名、文件夹名、文件名、文件扩展名、路径名、虚拟 URL、用户输入和 Twitter 账号都显示如下：“UI 文件位于 `Forms` 目录下。”

代码块设置为如下所示：

```cpp
#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
```

当我们希望将您的注意力引向代码块中的特定部分时，相关的行或项目将以粗体显示：

```cpp
#include <QStyleOption>
#include <QPainter>
#include <QPaintEvent>
#include <QMouseEvent>
#include <QResizeEvent>
#include "canvas.h"

Canvas::Canvas(QWidget *parent) :
  QWidget(parent)
{
}

void Canvas::paintEvent(QPaintEvent *e)
{
  QPainter painter(this);

  QStyleOption opt;
  opt.initFrom(this);
  this->style()->drawPrimitive(QStyle::PE_Widget, &opt, &painter, this);

 painter.drawImage(e->rect().topLeft(), image);
}

void Canvas::updateImage()
{
  QPainter painter(&image);
  painter.setPen(QColor(Qt::black));
  painter.setRenderHint(QPainter::Antialiasing);
  painter.drawPolyline(m_points.data(), m_points.count());
  this->update();
}

void Canvas::mousePressEvent(QMouseEvent *e)
{
  m_points.clear();
  m_points.append(e->localPos());
  updateImage();
}

void Canvas::mouseMoveEvent(QMouseEvent *e)
{
  m_points.append(e->localPos());
  updateImage();
}

void Canvas::mouseReleaseEvent(QMouseEvent *e)
{
  m_points.append(e->localPos());
  updateImage();
}

void Canvas::resizeEvent(QResizeEvent *e)
{
  QImage newImage(e->size(), QImage::Format_RGB32);
  newImage.fill(Qt::white);
  QPainter painter(&newImage);
  painter.drawImage(0, 0, image);
  image = newImage;
  QWidget::resizeEvent(e);
}
```

任何命令行输入或输出都写成如下所示：

```cpp
..\..\bin\binarycreator.exe -c config\config.xml -p packages internationalization_installer.exe
```

**新术语**和**重要单词**以粗体显示。您在屏幕上看到的单词，例如在菜单或对话框中，在文本中显示如下：“导航到**文件** | **新建文件**或**项目**。”

### 注意

警告或重要注意事项以如下所示的框显示。

### 小贴士

小贴士和技巧看起来像这样。

# 读者反馈

我们始终欢迎读者的反馈。告诉我们您对这本书的看法——您喜欢或不喜欢什么。读者反馈对我们来说很重要，因为它帮助我们开发出您真正能从中获得最大价值的标题。

要向我们发送一般反馈，只需发送电子邮件至 `<feedback@packtpub.com>`，并在邮件主题中提及本书的标题。

如果您在某个主题上具有专业知识，并且您有兴趣撰写或为本书做出贡献，请参阅我们的作者指南[www.packtpub.com/authors](http://www.packtpub.com/authors)。

# 客户支持

现在，您已经成为 Packt 书籍的骄傲拥有者，我们有一些东西可以帮助您充分利用您的购买。

## 下载示例代码

您可以从 [`www.packtpub.com`](http://www.packtpub.com) 下载您购买的所有 Packt 出版物的示例代码文件。如果您在其他地方购买了这本书，您可以访问 [`www.packtpub.com/support`](http://www.packtpub.com/support) 并注册，以便将文件直接通过电子邮件发送给您。

## 错误清单

尽管我们已经尽一切努力确保我们内容的准确性，但错误仍然会发生。如果您在我们的某本书中发现错误——可能是文本或代码中的错误——如果您能向我们报告这一点，我们将不胜感激。通过这样做，您可以节省其他读者的挫败感，并帮助我们改进本书的后续版本。如果您发现任何错误清单，请通过访问 [`www.packtpub.com/submit-errata`](http://www.packtpub.com/submit-errata)，选择您的书籍，点击**错误提交表单**链接，并输入您的错误详细信息来报告它们。一旦您的错误清单得到验证，您的提交将被接受，错误清单将被上传到我们的网站或添加到该标题的错误部分下的现有错误清单中。

要查看之前提交的错误清单，请访问 [`www.packtpub.com/books/content/support`](https://www.packtpub.com/books/content/support)，并在搜索字段中输入书籍名称。所需信息将出现在**错误清单**部分。

## 盗版

互联网上对版权材料的盗版是一个跨所有媒体的持续问题。在 Packt，我们非常重视我们版权和许可证的保护。如果您在互联网上发现任何形式的我们作品的非法副本，请立即提供位置地址或网站名称，以便我们可以寻求补救措施。

请通过 `<copyright@packtpub.com>` 联系我们，并提供疑似盗版材料的链接。

我们感谢您在保护我们的作者和我们为您提供有价值内容的能力方面的帮助。

## 询问

如果您对本书的任何方面有问题，您可以通过 `<questions@packtpub.com>` 联系我们，我们将尽力解决问题。
