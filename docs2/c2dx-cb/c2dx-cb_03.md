# 第三章. 使用标签

在本章中，我们将创建标签。要在屏幕上显示标签，您可以使用带有系统字体、真型字体和位图字体的 `Label` 类。本章将涵盖以下主题：

+   创建系统字体标签

+   创建真型字体标签

+   创建位图字体标签

+   创建丰富文本

# 创建系统字体标签

首先，我们将解释如何使用系统字体创建标签。系统字体是已安装在您的设备上的字体。由于它们已经安装，因此无需经过安装过程。因此，我们将跳过此配方中系统字体的安装说明，并直接进入创建标签。

## 如何做到...

这是如何通过指定系统字体创建标签的方法。您可以使用以下代码创建单行标签：

```cpp
auto label = Label::createWithSystemFont("Cocos2d-x", "Arial", 40);
label->setPosition(size/2);
this->addChild(label);
```

![如何做到...](img/B00561_03_01.jpg)

## 它是如何工作的...

您应该使用 `Label` 类通过指定一个字符串、系统字体和字体大小来显示字符串。`Label` 类将显示一个转换为图像的字符串。在创建 `Label` 实例后，您可以像使用 `Sprite` 一样使用它。因为 `Label` 也是一个节点，我们可以使用动作、缩放和透明度函数等属性来操作标签。

### 换行

您还可以在字符串中的任何位置添加新行，只需将换行符代码放入字符串中即可：

```cpp
auto label = Label::createWithSystemFont("Hello\nCocos2d-x", "Arial", 40);
label->setPosition(size/2);
this->addChild(label);
```

![换行](img/B00561_03_02.jpg)

### 文本对齐

您还可以在水平和垂直方向上指定文本对齐。

| 文本对齐类型 | 描述 |
| --- | --- |
| `TextHAlignment::LEFT` | 将文本水平对齐到左侧。这是水平对齐的默认值。 |
| `TextHAlignment::CENTER` | 将文本水平对齐到中心。 |
| `TextHAlignment::RIGHT` | 将文本水平对齐到右侧。 |
| `TextVAlignment::TOP` | 将文本垂直对齐到顶部。这是垂直对齐的默认值。 |
| `TextVAlignment::CENTER` | 将文本垂直对齐到中心。 |
| `TextVAlignment::BOTTOM` | 将文本垂直对齐到底部。 |

以下代码用于将文本水平对齐到中心：

```cpp
label-> setHorizontalAlignment(TextHAlignment::CENTER);
```

![文本对齐](img/B00561_03_03.jpg)

## 还有更多...

您也可以在创建标签后更新字符串。如果您想每秒更新一次字符串，可以通过设置以下计时器来实现：

首先，按照以下方式编辑 `HelloWorld.h`：

```cpp
class HelloWorld : public cocos2d::Layer
{
private:
    int sec;
public:
    …
;
```

接下来，按照以下方式编辑 `HelloWorld.cpp`：

```cpp
sec = 0;
std::string secString = StringUtils::toString(sec);
auto label = Label::createWithSystemFont(secString, "Arial", 40);
label->setPosition(size/2);
this->addChild(label);

this->schedule(= {
	sec++;
	std::string secString = StringUtils::toString(sec);
	label->setString(secString);
}, 1.0f, "myCallbackKey");
```

首先，您必须在头文件中定义一个整型变量。其次，您需要创建一个标签并将其添加到层上。然后，您需要设置调度器每秒执行函数。然后您可以通过使用 `setString` 方法来更新字符串。

### 小贴士

您可以使用 `StringUtils::toString` 方法将整型或浮点值转换为字符串值。

调度器可以在指定的时间间隔执行方法。我们将在第四章中解释调度器的工作原理，*构建场景和层*。请参阅它以获取有关调度器的更多详细信息。

# 创建真型字体标签

在这个食谱中，我们将解释如何使用真型字体创建标签。真型字体是可以安装到项目中的字体。Cocos2d-x 的项目已经包含了两个真型字体，即`arial.ttf`和`Marker Felt.ttf`，它们位于`Resources/fonts`文件夹中。

## 如何操作...

下面是如何通过指定真型字体来创建标签的方法。以下代码可以用来创建一个单行标签，使用真型字体：

```cpp
auto label = Label:: createWithTTF("True Type Font", "fonts/Marker
Felt.ttf", 40.0f);
label->setPosition(size/2);
this->addChild(label);
```

![如何操作...](img/B00561_03_04.jpg)

## 工作原理...

你可以通过指定标签字符串、真型字体的路径和字体大小来创建一个具有真型字体的`Label`。真型字体位于`Resources`文件夹的`font`文件夹中。Cocos2d-x 有两个真型字体，即`arial.ttf`和`Marker Felt.ttf`，它们位于`Resources/fonts`文件夹中。你可以从一个真型字体文件中生成不同字号的`Label`对象。如果你想要添加真型字体，如果你将其添加到`font`文件夹中，你可以使用原始的真型字体。然而，与位图字体相比，在渲染方面，真型字体较慢，并且更改字体样式和大小等属性是一个昂贵的操作。你必须小心不要频繁更新它。

## 更多内容...

如果你想要创建很多具有相同属性的`Label`对象，你可以通过指定`TTFConfig`来创建它们。`TTFConfig`具有真型字体所需的属性。你可以使用以下方式通过`TTFConfig`创建标签：

```cpp
TTFConfig config;
config.fontFilePath = "fonts/Marker Felt.ttf";
config.fontSize = 40.0f;
config.glyphs = GlyphCollection::DYNAMIC;
config.outlineSize = 0;
config.customGlyphs = nullptr;
config.distanceFieldEnabled = false;

auto label = Label::createWithTTF(config, "True Type Font");
label->setPosition(size/2);
this->addChild(label);
```

`TTFConfig`对象允许你设置一些具有相同属性的标签。

如果你想要改变`Label`的颜色，你可以改变它的颜色属性。例如，使用以下代码，你可以将颜色改为`RED`：

```cpp
label->setColor(Color3B::RED);
```

## 相关内容

+   你可以为标签设置效果。请查看本章的最后一个食谱。

# 创建位图字体标签

最后，我们将解释如何创建位图类型的标签。位图字体也是你可以安装到项目中的字体。位图字体本质上是一个包含大量字符和控制文件的图像文件，该控制文件详细说明了图像中每个字符的大小和位置。如果你在游戏中使用位图字体，你会看到位图字体在所有设备上大小相同。

## 准备工作

您必须准备一个位图字体。您可以使用`GlyphDesigner`等工具创建它。我们将在第十章*使用额外功能改进游戏*之后解释这个工具。现在，我们将使用 Cocos2d-x 中的位图字体。它位于`COCOS_ROOT/tests/cpp-tests/Resources/fonts`文件夹中。首先，您必须将以下文件添加到您的项目中`Resources/fonts`文件夹中。

+   `future-48.fnt`

+   `future-48.png`

## 如何操作...

如此通过指定位图字体创建标签。以下代码可以用于使用位图字体创建单行标签：

```cpp
auto label = Label:: createWithBMFont("fonts/futura-48.fnt",
"Bitmap Font");
label->setPosition(size/2);
this->addChild(label);
```

![如何操作...](img/B00561_03_05.jpg)

## 工作原理...

您可以通过指定`label`字符串、真型字体路径和字体大小来创建具有位图字体的`Label`。位图字体中的字符由点阵组成。这种字体渲染速度很快，但不可缩放。这就是为什么它在生成时具有固定字体大小。位图字体需要以下两个文件：一个.fnt 文件和一个`.png`文件。

## 更多...

`Label`中的每个字符都是一个`Sprite`。这意味着每个字符都可以旋转或缩放，并且具有其他可更改的属性：

```cpp
auto sprite1 = label->getLetter(0);
sprite1->setRotation(30.0f);

auto sprite2 = label->getLetter(1);
sprite2->setScale(0.5f);
```

![更多...](img/B00561_03_06.jpg)

# 创建富文本

在屏幕上创建`Label`对象后，您可以在它们上轻松创建一些效果，如阴影和轮廓，而无需创建自己的自定义类。`Label`类可以用于将这些效果应用于这些对象。但是请注意，并非所有标签类型都支持所有效果。

## 如何操作...

### 阴影

如此创建具有阴影效果的`标签`：

```cpp
auto layer = LayerColor::create(Color4B::GRAY);
this->addChild(layer);
auto label = Label::createWithTTF("Drop Shadow", "fonts/Marker
Felt.ttf", 40);
label->setPosition(size/2);
this->addChild(label);
// shadow effect
label->enableShadow();
```

![阴影](img/B00561_03_07.jpg)

### 轮廓

如此创建具有轮廓效果的`标签`：

```cpp
auto label = Label::createWithTTF("Outline", "fonts/Marker
Felt.ttf", 40);
label->setPosition(size/2);
this->addChild(label);
// outline effect
label->enableOutline(Color4B::RED, 5);
```

![轮廓](img/B00561_03_08.jpg)

### 发光

如此创建具有发光效果的`标签`：

```cpp
auto label = Label::createWithTTF("Glow", "fonts/Marker Felt.ttf", 40);
label->setPosition(size/2);
this->addChild(label);
// glow effect label->enableGlow(Color4B::RED);
```

![发光](img/B00561_03_09.jpg)

## 工作原理...

首先，我们生成一个灰色图层并将背景颜色改为灰色，因为否则我们无法看到阴影效果。将效果添加到标签中非常简单。您需要生成一个`Label`实例并执行一个效果方法，例如`enableShadow()`。这可以无参数执行。`enableOutline()`有两个参数，即轮廓颜色和轮廓大小。轮廓大小有一个默认值-1。如果它有负值，则轮廓不会显示。接下来，您必须设置第二个参数。`enableGlow`方法只有一个参数，即发光颜色。

并非所有标签类型都支持所有效果，但所有标签类型都支持阴影效果。`Outline`和`Glow`效果仅适用于真型字体。在之前的版本中，如果我们想在标签上应用效果，我们必须创建自己的自定义字体类。然而，当前版本的 Cocos2d-x，版本 3，支持标签效果，如阴影、轮廓和发光。

## 更多...

您还可以更改阴影颜色和偏移量。第一个参数是阴影颜色，第二个参数是偏移量，第三个参数是模糊半径。然而，不幸的是，在 Cocos2d-x 版本 3.4 中不支持更改模糊半径。

```cpp
auto label = Label::createWithTTF("Shadow", "fonts/Marker
Felt.ttf", 40);
label->setPosition(Vec2(size.width/2, size.height/3*2));
this->addChild(label);
label->enableShadow(Color4B::RED, Size(5,5), 0);
```

同时设置两个或更多这些效果也是可能的。以下代码可以用于设置标签的阴影和轮廓效果：

```cpp
auto label2 = Label::createWithTTF("Shadow & Outline", "fonts/Marker Felt.ttf", 40); label2->setPosition(Vec2(size.width/2, size.height/3)); this->addChild(label2); label2->enableShadow(Color4B::RED, Size(10,-10), 0);
label2->enableOutline(Color4B::BLACK, 5);
```

![还有更多...](img/B00561_03_10.jpg)
