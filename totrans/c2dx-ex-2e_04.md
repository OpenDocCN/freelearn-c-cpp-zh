# 第四章.与精灵的乐趣 - 天空防御

*是时候构建我们的第二个游戏了！这次，你将了解 Cocos2d-x 中动作的力量。我会向你展示如何仅通过运行 Cocos2d-x 中包含的各种动作命令来构建整个游戏，使你的精灵移动、旋转、缩放、淡入淡出、闪烁等。你还可以使用动作通过多个图像来动画化你的精灵，就像在电影中一样。所以，让我们开始吧。*

在本章中，你将学习：

+   如何使用精灵表优化你的游戏开发

+   如何在你的游戏中使用位图字体

+   实现和运行动作是多么简单

+   如何缩放、旋转、摆动、移动和淡出精灵

+   如何加载多个 `.png` 文件并使用它们来动画化精灵

+   如何使用 Cocos2d-x 创建通用游戏

# 游戏 - 天空防御

来认识我们的压力山大的城市……这里填入你选择的名称。这是一个美好的日子，突然天空开始下落。流星正冲向城市，你的任务是保护它安全。

游戏中的玩家可以轻触屏幕开始生长一个炸弹。当炸弹足够大可以激活时，玩家再次轻触屏幕来引爆它。任何附近的流星都会爆炸成百万碎片。炸弹越大，爆炸越剧烈，可以击毁的流星越多。但炸弹越大，生长它所需的时间就越长。

但坏消息不止这些。天空中还会掉落医疗包，如果你允许它们落到地面，你将恢复一些能量。

## 游戏设置

这是一个通用游戏。它为 iPad retina 屏幕设计，并将缩放到适合所有其他屏幕。游戏将以横屏模式进行，且不需要支持多点触控。

## 开始项目

请从本书的支持页面（[www.packtpub.com/support](http://www.packtpub.com/support)）下载文件 `4198_04_START_PROJECT.zip`。当你解压文件时，你会发现基本项目已经设置好，准备好供你工作。

创建此项目涉及的步骤与我们之前游戏中的类似。我使用的命令行是：

```cpp
cocos new SkyDefense -p com.rengelbert.SkyDefense -l cpp -d /Users/rengelbert/Desktop/SkyDefense

```

在 Xcode 中，你必须将 **Deployment Info** 中的 **Devices** 字段设置为 **Universal**，并将 **Device Family** 字段设置为 **Universal**。在 `RootViewController.mm` 中，支持的界面方向设置为 **Landscape**。

我们将要构建的游戏只需要一个类，`GameLayer.cpp`，你会发现这个类的接口已经包含了它所需的所有信息。

此外，一些更简单或过时的逻辑已经在实现文件中就位了。但我会随着我们游戏的发展来讲解这些。

## 为通用应用添加屏幕支持

在上一个游戏中，我们只针对 iPad 尺寸的屏幕。现在事情变得有点复杂，因为我们在通用游戏中增加了对较小屏幕的支持，以及一些最常见的 Android 屏幕尺寸。

因此，打开`AppDelegate.cpp`文件。在`applicationDidFinishLaunching`方法内部，我们现在有以下代码：

```cpp
auto screenSize = glview->getFrameSize();
auto designSize = Size(2048, 1536);
glview->setDesignResolutionSize(designSize.width, designSize.height,  ResolutionPolicy::EXACT_FIT);
std::vector<std::string> searchPaths;
if (screenSize.height > 768) {
   searchPaths.push_back("ipadhd");
   director->setContentScaleFactor(1536/designSize.height);
} else if (screenSize.height > 320) {
   searchPaths.push_back("ipad");
   director->setContentScaleFactor(768/designSize.height);
} else {
   searchPaths.push_back("iphone");
   director->setContentScaleFactor(380/designSize.height);
}
auto fileUtils = FileUtils::getInstance();
fileUtils->setSearchPaths(searchPaths);
```

再次提醒，我们告诉我们的`GLView`对象（我们的 OpenGL 视图）我们为某个屏幕尺寸（iPad 视网膜屏）设计了游戏，并且我们再次希望我们的游戏屏幕调整大小以匹配设备上的屏幕（`ResolutionPolicy::EXACT_FIT`）。

然后我们根据设备的屏幕尺寸确定从哪里加载我们的图像。我们有 iPad 视网膜屏的美术资源，然后是普通 iPad，它由 iPhone 视网膜屏共享，以及普通 iPhone 的美术资源。

我们最后根据设计的目标设置缩放因子。

## 添加背景音乐

仍然在`AppDelegate.cpp`中，我们加载游戏中将使用的声音文件，包括一个`background.mp3`（由 Kevin MacLeod 从[incompetech.com](http://incompetech.com)提供），我们通过以下命令加载它：

```cpp
auto audioEngine = SimpleAudioEngine::getInstance();
audioEngine->preloadBackgroundMusic(fileUtils->fullPathForFilename("background.mp3").c_str());

```

我们最后将音效的音量稍微调低：

```cpp
//lower playback volume for effects
audioEngine->setEffectsVolume(0.4f);
```

对于背景音乐的音量，您必须使用`setBackgroundMusicVolume`。如果您在游戏中创建某种音量控制，这些就是您根据用户的偏好调整音量的调用。

## 初始化游戏

现在回到`GameLayer.cpp`。如果您查看我们的`init`方法，您会看到游戏通过调用三个方法进行初始化：`createGameScreen`、`createPools`和`createActions`。

我们将在第一个方法内部创建所有屏幕元素，然后创建对象池，这样我们就不需要在主循环中实例化任何精灵；我们将在`createActions`方法内部创建游戏中使用的所有主要动作。

游戏初始化完成后，我们开始播放背景音乐，并将其`should loop`参数设置为`true`：

```cpp
SimpleAudioEngine::getInstance()-  >playBackgroundMusic("background.mp3", true);
```

我们再次存储屏幕尺寸以供将来参考，并使用一个`_running`布尔值来表示游戏状态。

如果现在运行游戏，您应该只能看到背景图像：

![初始化游戏](img/00011.jpeg)

# 在 Cocos2d-x 中使用精灵图集

精灵图集是将多个图像组合到一个图像文件中的方法。为了用这些图像之一纹理化精灵，您必须知道该图像在精灵图集中的位置信息（其矩形）。

精灵图集通常分为两个文件：一个是图像文件，另一个是数据文件，它描述了在图像中可以找到单个纹理的位置。

我使用`TexturePacker`为游戏创建了这些文件。您可以在`Resources`文件夹内的`ipad`、`ipadhd`和`iphone`文件夹中找到它们。有一个`sprite_sheet.png`文件用于图像，还有一个`sprite_sheet.plist`文件，它描述了图像中的单个帧。

这就是`sprite_sheet.png`文件的外观：

![在 Cocos2d-x 中使用精灵图集](img/00012.jpeg)

## 批量绘制精灵

在 Cocos2d-x 中，精灵图可以与一个专门的节点一起使用，称为 `SpriteBatchNode`。这个节点可以在你希望使用多个共享相同源图像的精灵时使用。所以你可以有多个 `Sprite` 类的实例，例如使用 `bullet.png` 纹理。如果源图像是精灵图，你可以有多个精灵实例，显示尽可能多的不同纹理，就像你可以在精灵图中打包的那样。

使用 `SpriteBatchNode`，你可以显著减少游戏渲染阶段的调用次数，这有助于针对性能较弱的系统，尽管在更现代的设备上并不明显。

让我展示如何创建一个 `SpriteBatchNode`。

# 行动时间 - 创建 SpriteBatchNode

让我们开始实现 `GameLayer.cpp` 中的 `createGameScreen` 方法。在添加 `bg` 精灵的行下面，我们实例化我们的批处理节点：

```cpp
void GameLayer::createGameScreen() {

  //add bg
  auto bg = Sprite::create("bg.png");
  ...

  SpriteFrameCache::getInstance()->
  addSpriteFramesWithFile("sprite_sheet.plist");

  _gameBatchNode = SpriteBatchNode::create("sprite_sheet.png");
  this->addChild(_gameBatchNode);
```

为了从精灵图中创建批处理节点，我们首先将 `sprite_sheet.plist` 文件中描述的所有帧信息加载到 `SpriteFrameCache` 中。然后我们使用 `sprite_sheet.png` 文件创建批处理节点，这是所有添加到该批处理节点的精灵共享的源纹理。（背景图像不是精灵图的一部分，所以我们将其单独添加，在我们将 `_gameBatchNode` 添加到 GameLayer 之前。）

现在我们可以开始将东西放入 `_gameBatchNode` 中。

1.  首先，是城市：

    ```cpp
    for (int i = 0; i < 2; i++) {
      auto sprite = Sprite::createWithSpriteFrameName ("city_dark.png");
        sprite->setAnchorPoint(Vec2(0.5,0));
      sprite->setPosition(_screenSize.width * (0.25f + i *  0.5f),0));
      _gameBatchNode->addChild(sprite, kMiddleground);

      sprite = Sprite::createWithSpriteFrameName("city_light.png");
      sprite->setAnchorPoint(Vec2(0.5,0));
      sprite->setPosition(Vec2(_screenSize.width * (0.25f + i *  0.5f),
      _screenSize.height * 0.1f));
      _gameBatchNode->addChild(sprite, kBackground);
    }
    ```

1.  然后是树木：

    ```cpp
    //add trees
    for (int i = 0; i < 3; i++) {
      auto sprite = Sprite::createWithSpriteFrameName("trees.png");
      sprite->setAnchorPoint(Vec2(0.5f, 0.0f));
      sprite->setPosition(Vec2(_screenSize.width * (0.2f + i * 0.3f),0));
      _gameBatchNode->addChild(sprite, kForeground);

    }
    ```

    注意，在这里我们通过传递精灵帧名称来创建精灵。这些帧名称的 ID 是通过我们的 `sprite_sheet.plist` 文件加载到 `SpriteFrameCache` 中的。

1.  到目前为止的屏幕由两个实例的 `city_dark.png` 在屏幕底部拼接而成，以及两个实例的 `city_light.png` 也进行拼接。一个需要出现在另一个之上，为此我们使用在 `GameLayer.h` 中声明的枚举值：

    ```cpp
    enum {
      kBackground,
      kMiddleground,
      kForeground
    };
    ```

1.  我们使用 `addChild(Node, zOrder)` 方法通过不同的 `z` 值将精灵层叠在一起。

    例如，当我们后来添加三个显示 `trees.png` 精灵帧的精灵时，我们使用枚举列表中找到的最高 z 值将它们添加到所有之前的精灵之上，这个值是 `kForeground`。

### 注意

为什么费尽心机去拼接图像而不是使用一张大图，或者将其中一些与背景图像结合？因为我想在单个精灵图中包含尽可能多的图像，并使该精灵图尽可能小，以展示所有你可以使用和优化的精灵图方法。在这个特定的游戏中，这并不是必需的。

## *发生了什么？*

我们开始创建游戏的初始屏幕。我们使用 `SpriteBatchNode` 来包含所有使用精灵表图片的精灵。所以 `SpriteBatchNode` 的行为就像任何节点一样——作为一个容器。我们可以通过操作它们的 `z` 排序在批处理节点内层叠单个精灵。

# Cocos2d-x 中的位图字体

Cocos2d-x 的 `Label` 类有一个静态的 `create` 方法，它使用位图图像来显示字符。

我们在这里使用的位图图像是用 GlyphDesigner 程序创建的，本质上，它就像精灵表一样工作。事实上，`Label` 扩展了 `SpriteBatchNode`，所以它的行为就像一个批处理节点。

你需要的所有单个字符的图像都打包在一个 PNG 文件（`font.png`）中，然后是一个描述每个字符位置的描述文件（`font.fnt`）。以下截图显示了我们的游戏中的字体精灵表：

![Cocos2d-x 中的位图字体](img/00013.jpeg)

`Label` 和常规 `SpriteBatchNode` 类之间的区别在于数据文件还向 `Label` 对象提供了如何使用此字体*书写*的信息。换句话说，如何正确地分配字符和行。

我们在游戏中使用的 `Label` 对象是用数据文件名和它们的初始字符串值实例化的：

```cpp
_scoreDisplay = Label::createWithBMFont("font.fnt", "0");
```

标签的值通过 `setString` 方法更改：

```cpp
_scoreDisplay->setString("1000");
```

### 注意事项

就像游戏中每张图片一样，我们在 `Resources` 文件夹中也有不同版本的 `font.fnt` 和 `font.png`，每个屏幕定义一个。`FileUtils` 将再次承担寻找正确文件的重任，以适应正确的屏幕。

现在让我们为我们的游戏创建标签。

# 行动时间 - 创建位图字体标签

创建位图字体与创建批处理节点有些相似。

1.  继续使用 `createGameScreen` 方法，向 `score` 标签添加以下行：

    ```cpp
    _scoreDisplay = Label::createWithBMFont("font.fnt", "0");
    _scoreDisplay->setAnchorPoint(Vec2(1,0.5));
    _scoreDisplay->setPosition(Vec2 (_screenSize.width * 0.8f, _screenSize.height * 0.94f));
    this->addChild(_scoreDisplay);
    ```

    然后添加一个标签来显示能量等级，并将其水平对齐设置为 `Right`：

    ```cpp
    _energyDisplay = Label::createWithBMFont("font.fnt", "100%", TextHAlignment::RIGHT);
    _energyDisplay->setPosition(Vec2 (_screenSize.width * 0.3f, _screenSize.height * 0.94f));
    this->addChild(_energyDisplay);
    ```

1.  为 `_energyDisplay` 标签旁边出现的图标添加以下行：

    ```cpp
    auto icon = Sprite::createWithSpriteFrameName("health_icon.png");
    icon->setPosition( Vec2(_screenSize. width * 0.15f,  _screenSize.height * 0.94f) );
    _gameBatchNode->addChild(icon, kBackground);
    ```

## *刚才发生了什么？*

我们刚刚在 Cocos2d-x 中创建了第一个位图字体对象。现在让我们完成创建游戏精灵。

# 行动时间 - 添加最终屏幕精灵

我们最后需要创建的精灵是云、炸弹和冲击波，以及我们的游戏状态消息。

1.  回到 `createGameScreen` 方法，将云添加到屏幕上：

    ```cpp
    for (int i = 0; i < 4; i++) {
      float cloud_y = i % 2 == 0 ? _screenSize.height * 0.4f : _screenSize.height * 0.5f;
      auto cloud = Sprite::createWithSpriteFrameName("cloud.png");
      cloud->setPosition(Vec2 (_screenSize.width * 0.1f + i * _screenSize.width * 0.3f,  cloud_y));
      _gameBatchNode->addChild(cloud, kBackground);
      _clouds.pushBack(cloud);
    }
    ```

1.  创建 `_bomb` 精灵；玩家在触摸屏幕时会*增长*：

    ```cpp
    _bomb = Sprite::createWithSpriteFrameName("bomb.png");
    _bomb->getTexture()->generateMipmap();
    _bomb->setVisible(false);

    auto size = _bomb->getContentSize();

    //add sparkle inside bomb sprite
    auto sparkle = Sprite::createWithSpriteFrameName("sparkle.png");
    sparkle->setPosition(Vec2(size.width * 0.72f, size.height *  0.72f));
    _bomb->addChild(sparkle, kMiddleground, kSpriteSparkle);

    //add halo inside bomb sprite
    auto halo = Sprite::createWithSpriteFrameName ("halo.png");
    halo->setPosition(Vec2(size.width * 0.4f, size.height *  0.4f));
    _bomb->addChild(halo, kMiddleground, kSpriteHalo);
    _gameBatchNode->addChild(_bomb, kForeground);
    ```

1.  然后创建 `_shockwave` 精灵，它在 `_bomb` 爆炸后出现：

    ```cpp
    _shockWave = Sprite::createWithSpriteFrameName("shockwave.png");
    _shockWave->getTexture()->generateMipmap();
    _shockWave->setVisible(false);
    _gameBatchNode->addChild(_shockWave);
    ```

1.  最后，添加屏幕上出现的两条消息，一条用于我们的 `intro` 状态，另一条用于 `gameover` 状态：

    ```cpp
    _introMessage = Sprite::createWithSpriteFrameName("logo.png");
    _introMessage->setPosition(Vec2 (_screenSize.width * 0.5f, _screenSize.height * 0.6f));
    _introMessage->setVisible(true);
    this->addChild(_introMessage, kForeground);

    _gameOverMessage = Sprite::createWithSpriteFrameName ("gameover.png");
    _gameOverMessage->setPosition(Vec2 (_screenSize.width * 0.5f, _screenSize.height * 0.65f));
    _gameOverMessage->setVisible(false);
    this->addChild(_gameOverMessage, kForeground);
    ```

## *刚才发生了什么？*

在之前的代码中关于精灵有很多新的信息。所以让我们仔细地过一遍：

+   我们首先添加了云。我们将精灵放入一个向量中，以便以后可以移动云。注意，它们也是我们批处理节点的一部分。

+   接下来是炸弹精灵和我们的第一个新调用：

    ```cpp
    _bomb->getTexture()->generateMipmap();
    ```

+   使用这种方法，我们告诉框架创建该纹理的逐级细节（mipmaps）的逐级减小尺寸的抗锯齿副本，因为我们稍后将要将其缩小。当然，这是可选的；精灵可以在不首先生成 mipmaps 的情况下调整大小，但如果您注意到缩放精灵时质量下降，您可以通过为它们的纹理创建 mipmaps 来修复这个问题。

    ### 注意

    纹理必须具有所谓的 POT（2 的幂：2、4、8、16、32、64、128、256、512、1024、2048 等等）的大小值。OpenGL 中的纹理必须始终以这种方式进行尺寸设置；如果不是这样，Cocos2d-x 将执行以下两种操作之一：它将在内存中调整纹理的大小，添加透明像素，直到图像达到 POT 大小，或者停止执行并抛出断言。对于用于 mipmaps 的纹理，框架将停止执行非 POT 纹理。

+   我将`sparkle`和`halo`精灵作为子节点添加到`_bomb`精灵中。这将利用节点的容器特性为我们带来优势。当我放大炸弹时，所有子节点也会随之放大。

+   注意，我还为`halo`和`sparkle`的`addChild`方法使用了第三个参数：

    ```cpp
    bomb->addChild(halo, kMiddleground, kSpriteHalo);
    ```

+   这个第三个参数是从`GameLayer.h`中声明的另一个枚举列表中的整数标签。我可以使用这个标签来检索从精灵中特定的子节点，如下所示：

    ```cpp
    auto halo = (Sprite *)  bomb->getChildByTag(kSpriteHalo);
    ```

我们现在已经设置了游戏屏幕：

![发生了什么？](img/00014.jpeg)

接下来是对象池。

# 是时候行动了——创建我们的对象池

池只是对象的向量。以下是创建它们的步骤：

1.  在`createPools`方法内部，我们首先为流星创建一个池：

    ```cpp
    void GameLayer::createPools() {
      int i;
      _meteorPoolIndex = 0;
      for (i = 0; i < 50; i++) {
      auto sprite = Sprite::createWithSpriteFrameName("meteor.png");
      sprite->setVisible(false);
      _gameBatchNode->addChild(sprite, kMiddleground, kSpriteMeteor);
      _meteorPool.pushBack(sprite);
    }
    ```

1.  然后我们为健康包创建一个对象池：

    ```cpp
    _healthPoolIndex = 0;
    for (i = 0; i < 20; i++) {
      auto sprite = Sprite::createWithSpriteFrameName("health.png");
      sprite->setVisible(false);
      sprite->setAnchorPoint(Vec2(0.5f, 0.8f));
      _gameBatchNode->addChild(sprite, kMiddleground, kSpriteHealth);
      _healthPool.pushBack(sprite);
    }
    ```

1.  随着游戏的进行，我们将使用相应的池索引从向量中检索对象。

## *发生了什么？*

我们现在有一个不可见的流星精灵向量和一个不可见的健康精灵向量。我们将使用它们各自的池索引在需要时从向量中检索这些精灵，正如您一会儿将看到的。但首先我们需要处理动作和动画。

### 注意

使用对象池，我们在主循环中减少了实例化的数量，并且它允许我们永远不会销毁可以重用的任何东西。但如果您需要从节点中移除一个子节点，请使用`->removeChild`或`->removeChildByTag`（如果存在标签）。

# 动作概述

如果您还记得，节点将存储有关位置、缩放、旋转、可见性和不透明度的信息。在 Cocos2d-x 中，有一个`Action`类可以随时间改变这些值中的每一个，从而实现这些转换的动画。

动作通常使用静态方法`create`创建。这些动作中的大多数都是基于时间的，所以通常您需要传递给动作的第一个参数是动作的时间长度。例如：

```cpp
auto fadeout = FadeOut::create(1.0f);
```

这将创建一个`渐隐`动作，它将在一秒钟内完成。您可以在精灵或节点上运行它，如下所示：

```cpp
mySprite->runAction(fadeout);
```

Cocos2d-x 有一个非常灵活的系统，允许我们创建任何组合的动作和变换，以实现我们想要的任何效果。

例如，你可以选择创建一个包含多个动作的动作序列（`Sequence`）；或者你可以对你的动作应用缓动效果（`EaseIn`、`EaseOut` 等）。你可以选择重复一个动作一定次数（`Repeat`）或无限重复（`RepeatForever`）；你还可以添加回调函数，以便在动作完成后调用（通常在 `Sequence` 动作内部）。

# 是时候进行动作了——使用 Cocos2d-x 创建动作

使用 Cocos2d-x 创建动作是一个非常简单的过程：

1.  在我们的 `createActions` 方法中，我们将实例化我们可以在游戏中重复使用的动作。让我们创建我们的第一个动作：

    ```cpp
    void GameLayer::createActions() {
     //swing action for health drops
     auto easeSwing = Sequence::create(
     EaseInOut::create(RotateTo::create(1.2f, -10), 2),
     EaseInOut::create(RotateTo::create(1.2f, 10), 2),
     nullptr);//mark the end of a sequence with a nullptr
     _swingHealth = RepeatForever::create( (ActionInterval *) easeSwing );
     _swingHealth->retain();
    ```

1.  动作可以以许多不同的形式组合。在这里，保留的 `_swingHealth` 动作是一个 `Sequence` 的 `RepeatForever` 动作，它将首先以一个方向旋转健康精灵，然后以另一个方向旋转，`EaseInOut` 包裹着 `RotateTo` 动作。`RotateTo` 需要 `1.2` 秒来首先将精灵旋转到 `-10` 度，然后旋转到 `10` 度。缓动函数的值为 `2`，我建议你尝试一下，以了解它在视觉上的意义。接下来我们再添加三个动作：

    ```cpp
    //action sequence for shockwave: fade out, callback when  //done
    _shockwaveSequence = Sequence::create(
      FadeOut::create(1.0f),
      CallFunc::create(std::bind(&GameLayer::shockwaveDone, this)), nullptr);
    _shockwaveSequence->retain();

    //action to grow bomb
    _growBomb = ScaleTo::create(6.0f, 1.0);
    _growBomb->retain();

    //action to rotate sprites
    auto rotate = RotateBy::create(0.5f ,  -90);
    _rotateSprite = RepeatForever::create( rotate );
    _rotateSprite->retain();
    ```

1.  首先，另一个 `Sequence`。这将使精灵淡出并调用 `shockwaveDone` 函数，该函数已经在类中实现，并在调用时将 `_shockwave` 精灵变为不可见。

1.  最后一个是 `RotateBy` 动作的 `RepeatForever` 动作。在半秒内，执行此动作的精灵将旋转 `-90` 度，并且会不断重复这样做。

## *刚才发生了什么？*

你刚刚看到了如何在 Cocos2d-x 中创建动作，以及框架如何允许进行各种组合以实现任何效果。

起初阅读 `Sequence` 动作并理解其工作原理可能有些困难，但一旦将其分解为各个部分，逻辑就很容易理解了。

但我们还没有完成 `createActions` 方法。接下来是精灵动画。

# 在 Cocos2d-x 中动画精灵

需要记住的关键点是，动画只是另一种类型的动作，它会在一段时间内改变精灵使用的纹理。 

为了创建一个动画动作，你首先需要创建一个 `Animation` 对象。该对象将存储有关你希望在动画中使用的不同精灵帧的所有信息，动画的长度（以秒为单位），以及它是否循环。

使用这个 `Animation` 对象，然后创建一个 `Animate` 动作。让我们看看。

# 是时候进行动作了——创建动画

动画是一种特殊类型的动作，需要一些额外的步骤：

1.  在同一个 `createActions` 方法中，添加游戏中的两个动画的代码行。首先，我们从当流星到达城市时显示爆炸动画的动画开始。我们首先将帧加载到 `Animation` 对象中：

    ```cpp
    auto animation = Animation::create();
    int i;
    for(i = 1; i <= 10; i++) {
      auto name = String::createWithFormat("boom%i.png", i);
      auto frame = SpriteFrameCache::getInstance()->getSpriteFrameByName(name->getCString());
      animation->addSpriteFrame(frame);
    }
    ```

1.  然后，我们在 `Animate` 动作中使用 `Animation` 对象：

    ```cpp
    animation->setDelayPerUnit(1 / 10.0f);
    animation->setRestoreOriginalFrame(true);
    _groundHit = 
      Sequence::create(
        MoveBy::create(0, Vec2(0,_screenSize.height * 0.12f)),
        Animate::create(animation),
       CallFuncN::create(CC_CALLBACK_1(GameLayer::animationDone, this)), nullptr);
    _groundHit->retain();
    ```

1.  相同的步骤被重复使用以创建其他爆炸动画，当玩家击中流星或健康包时使用。

    ```cpp
    animation = Animation::create();
    for(int i = 1; i <= 7; i++) {
     auto name = String::createWithFormat("explosion_small%i.png", i);
     auto frame = SpriteFrameCache::getInstance()->getSpriteFrameByName(name->getCString());
     animation->addSpriteFrame(frame);
    }

    animation->setDelayPerUnit(0.5 / 7.0f);
    animation->setRestoreOriginalFrame(true);
    _explosion = Sequence::create(
         Animate::create(animation),
       CallFuncN::create(CC_CALLBACK_1(GameLayer::animationDone, this)), nullptr);
    _explosion->retain();
    ```

## *发生了什么？*

我们在 Cocos2d-x 中创建了两种非常特殊的行为实例：`Animate`。以下是我们的操作步骤：

+   首先，我们创建了一个 `Animation` 对象。该对象持有动画中使用的所有纹理的引用。帧被命名为，以便它们可以在循环中轻松连接（`boom1`、`boom2`、`boom3` 等等）。第一个动画有 10 帧，第二个有 7 帧。

+   纹理（或帧）是我们从 `SpriteFrameCache` 中获取的 `SpriteFrame` 对象，如您所记得，它包含来自 `sprite_sheet.plist` 数据文件的所有信息。因此，帧在我们的精灵图中。

+   然后当所有帧都就绪后，我们通过将动画所需的总秒数除以总帧数来确定每帧的延迟。

+   `setRestoreOriginalFrame` 方法在这里很重要。如果我们把 `setRestoreOriginalFrame` 设置为 `true`，那么动画结束后精灵将恢复到其原始外观。例如，如果我有一个将在流星精灵上运行的爆炸动画，那么在爆炸动画结束时，精灵将恢复显示流星纹理。

+   是时候进行实际操作了。`Animate` 接收 `Animation` 对象作为其参数。（在第一个动画中，我们在爆炸出现之前移动精灵的位置，因此有一个额外的 `MoveBy` 方法。）

+   在这两种情况下，我调用了类中已经实现的一个 `animationDone` 回调。这使得调用精灵变得不可见：

    ```cpp
    void GameLayer::animationDone (Node* pSender) {
      pSender->setVisible(false);
    }
    ```

    ### 注意

    我们本可以使用相同的方法为两个回调（`animationDone` 和 `shockwaveDone`）进行操作，因为它们完成的是相同的事情。但我想要展示一个接收节点作为参数的回调，这个节点进行了调用，另一个没有。分别是 `CallFuncN` 和 `CallFunc`，它们被用于我们刚刚创建的动作序列中。

# 是时候让我们的游戏开始跳动！

好的，我们已经将主要元素放置到位，并准备好添加运行游戏的最后一点逻辑。但这一切将如何运作呢？

我们将使用倒计时系统来添加新的流星和新的健康包，以及一个逐渐使游戏更难玩的倒计时。

在触摸时，如果游戏未运行，玩家将开始游戏，并在游戏过程中添加炸弹并爆炸它们。爆炸会产生冲击波。

在更新时，我们将检查我们的 `_shockwave` 精灵（如果可见）与所有下落物体之间的碰撞。就这样。Cocos2d-x 将通过我们创建的动作和回调来处理所有其余的事情！

让我们先实现我们的触摸事件。

# 行动时间 - 处理触摸事件

是时候把玩家带到我们的派对上了：

1.  是时候实现我们的`onTouchBegan`方法了。我们首先处理两个游戏状态，`intro`和`game over`：

    ```cpp
    bool GameLayer::onTouchBegan (Touch * touch, Event * event){

      //if game not running, we are seeing either intro or  //gameover
      if (!_running) {
        //if intro, hide intro message
        if (_introMessage->isVisible()) {
          _introMessage->setVisible(false);

          //if game over, hide game over message 
        } else if (_gameOverMessage->isVisible()) {
          SimpleAudioEngine::getInstance()->stopAllEffects();
          _gameOverMessage->setVisible(false);

        }

        this->resetGame();
        return true;
      }
    ```

1.  在这里，我们检查游戏是否没有运行。如果没有运行，我们检查我们的消息是否可见。如果`_introMessage`是可见的，我们就隐藏它。如果`_gameOverMessage`是可见的，我们就停止所有当前音效并隐藏该消息。然后我们调用一个名为`resetGame`的方法，它将所有游戏数据（能量、得分和倒计时）重置为其初始值，并将`_running`设置为`true`。

1.  接下来我们处理触摸。但我们每次只需要处理一个，所以我们使用`Set`上的`->anyObject()`：

    ```cpp
    auto touch = (Touch *)pTouches->anyObject();

    if (touch) {

      //if bomb already growing...
      if (_bomb->isVisible()) {
        //stop all actions on bomb, halo and sparkle
        _bomb->stopAllActions();
        auto child = (Sprite *) _bomb->getChildByTag(kSpriteHalo);
        child->stopAllActions();
        child = (Sprite *) _bomb->getChildByTag(kSpriteSparkle);
        child->stopAllActions();

        //if bomb is the right size, then create shockwave
        if (_bomb->getScale() > 0.3f) {
          _shockWave->setScale(0.1f);
          _shockWave->setPosition(_bomb->getPosition());
          _shockWave->setVisible(true);
          _shockWave->runAction(ScaleTo::create(0.5f, _bomb->getScale() * 2.0f));
          _shockWave->runAction(_shockwaveSequence->clone());
          SimpleAudioEngine::getInstance()->playEffect("bombRelease.wav");

        } else {
          SimpleAudioEngine::getInstance()->playEffect("bombFail.wav");
        }
        _bomb->setVisible(false);
        //reset hits with shockwave, so we can count combo hits
        _shockwaveHits = 0;

     //if no bomb currently on screen, create one
     } else {
        Point tap = touch->getLocation();
        _bomb->stopAllActions();
        _bomb->setScale(0.1f);
        _bomb->setPosition(tap);
        _bomb->setVisible(true);
        _bomb->setOpacity(50);
        _bomb->runAction(_growBomb->clone());

         auto child = (Sprite *) _bomb->getChildByTag(kSpriteHalo);
         child->runAction(_rotateSprite->clone());
         child = (Sprite *) _bomb->getChildByTag(kSpriteSparkle);
         child->runAction(_rotateSprite->clone());
      }
    }
    ```

1.  如果`_bomb`是可见的，这意味着它已经在屏幕上生长了。所以当触摸时，我们在炸弹上使用`stopAllActions()`方法，并且使用`stopAllActions()`方法在其子项上，这些子项是通过我们的标签检索到的：

    ```cpp
    child = (Sprite *) _bomb->getChildByTag(kSpriteHalo);
    child->stopAllActions();
    child = (Sprite *) _bomb->getChildByTag(kSpriteSparkle);
    child->stopAllActions();
    ```

1.  如果`_bomb`的大小正确，我们就开始我们的`_shockwave`。如果它不正确，我们就播放一个炸弹故障音效；没有爆炸，并且`_shockwave`不会被显示出来。

1.  如果我们有爆炸，那么`_shockwave`精灵的缩放设置为`10`%。它放置在炸弹相同的地点，并且我们对它执行几个动作：我们将`_shockwave`精灵的缩放增加到爆炸时炸弹的两倍，并运行我们之前创建的`_shockwaveSequence`的副本。

1.  最后，如果屏幕上没有可见的`_bomb`，我们就创建一个。然后我们在`_bomb`精灵及其子项上运行之前创建的动作的克隆。当`_bomb`生长时，其子项也会生长。但是当子项旋转时，炸弹不会：父项改变其子项，但子项不会改变它们的父项。

## *刚才发生了什么？*

我们刚刚添加了游戏核心逻辑的一部分。玩家通过触摸创建和爆炸炸弹来阻止陨石到达城市。现在我们需要创建我们的下落物体。但首先，让我们设置我们的倒计时和游戏数据。

# 是时候开始和重新开始游戏了

让我们添加开始和重新开始游戏的逻辑。

1.  让我们来编写`resetGame`函数的实现：

    ```cpp
    void GameLayer::resetGame(void) {
        _score = 0;
        _energy = 100;

        //reset timers and "speeds"
        _meteorInterval = 2.5;
        _meteorTimer = _meteorInterval * 0.99f;
        _meteorSpeed = 10;//in seconds to reach ground
        _healthInterval = 20;
        _healthTimer = 0;
        _healthSpeed = 15;//in seconds to reach ground

        _difficultyInterval = 60;
        _difficultyTimer = 0;

        _running = true;

        //reset labels
        _energyDisplay->setString(std::to_string((int) _energy) + "%");
        _scoreDisplay->setString(std::to_string((int) _score));
    }
    ```

1.  接下来，添加`stopGame`的实现：

    ```cpp
    void GameLayer::stopGame() {

        _running = false;

        //stop all actions currently running
        int i;
        int count = (int) _fallingObjects.size();

        for (i = count-1; i >= 0; i--) {
            auto sprite = _fallingObjects.at(i);
            sprite->stopAllActions();
            sprite->setVisible(false);
            _fallingObjects.erase(i);
        }
        if (_bomb->isVisible()) {
            _bomb->stopAllActions();
            _bomb->setVisible(false);
            auto child = _bomb->getChildByTag(kSpriteHalo);
            child->stopAllActions();
            child = _bomb->getChildByTag(kSpriteSparkle);
            child->stopAllActions();
        }
        if (_shockWave->isVisible()) {
            _shockWave->stopAllActions();
            _shockWave->setVisible(false);
        }
        if (_ufo->isVisible()) {
            _ufo->stopAllActions();
            _ufo->setVisible(false);
            auto ray = _ufo->getChildByTag(kSpriteRay);
            ray->stopAllActions();
            ray->setVisible(false);
        }
    }
    ```

## *刚才发生了什么？*

通过这些方法我们控制游戏玩法。我们通过`resetGame()`使用默认值开始游戏，并通过`stopGame()`停止所有动作。

类中已经实现了随着时间推移使游戏更难的方法。如果你看看这个方法（`increaseDifficulty`），你会看到它减少了陨石之间的间隔，并减少了陨石到达地面的时间。

现在我们只需要`update`方法来运行倒计时和检查碰撞。

# 是时候更新游戏了

我们已经在`update`函数内部有了更新倒计时的代码。如果时间到了，我们需要添加陨石或医疗包，我们就这么做。如果时间到了，我们需要让游戏更难玩，我们也这么做。

### 注意

可以使用一个动作来处理这些计时器：一个 `Sequence` 动作和一个 `Delay` 动作对象以及一个回调。但使用这些倒计时有一些优点。重置它们和更改它们更容易，并且我们可以直接将它们带入我们的主循环。

现在是添加我们的主循环的时候了：

1.  我们需要做的是检查碰撞。所以添加以下代码：

    ```cpp
    if (_shockWave->isVisible()) {
     count = (int) _fallingObjects.size();
     for (i = count-1; i >= 0; i--) {
       auto sprite =  _fallingObjects.at(i);
       diffx = _shockWave->getPositionX() - sprite->getPositionX();
       diffy = _shockWave->getPositionY() - sprite->getPositionY();
       if (pow(diffx, 2) + pow(diffy, 2) <= pow(_shockWave->getBoundingBox().size.width * 0.5f, 2)) {
        sprite->stopAllActions();
        sprite->runAction( _explosion->clone());
        SimpleAudioEngine::getInstance()->playEffect("boom.wav");
        if (sprite->getTag() == kSpriteMeteor) {
          _shockwaveHits++;
          _score += _shockwaveHits * 13 + _shockwaveHits * 2;
        }
        //play sound
        _fallingObjects.erase(i);
      }
     }
     _scoreDisplay->setString(std::to_string(_score));
    }
    ```

1.  如果 `_shockwave` 是可见的，我们检查它和 `_fallingObjects` 向量中每个精灵之间的距离。如果我们击中任何流星，我们就增加 `_shockwaveHits` 属性的值，这样我们就可以奖励玩家多次击中。接下来我们移动云朵：

    ```cpp
    //move clouds
    for (auto sprite : _clouds) {
      sprite->setPositionX(sprite->getPositionX() + dt * 20);
      if (sprite->getPositionX() > _screenSize.width + sprite->getBoundingBox().size.width * 0.5f)
        sprite->setPositionX(-sprite->getBoundingBox().size.width * 0.5f);
    }
    ```

1.  我选择不使用 `MoveTo` 动作来展示云朵，以显示可以由简单动作替换的代码量。如果不是因为 Cocos2d-x 动作，我们就必须实现移动、旋转、摆动、缩放和爆炸所有精灵的逻辑！

1.  最后：

    ```cpp
    if (_bomb->isVisible()) {
       if (_bomb->getScale() > 0.3f) {
          if (_bomb->getOpacity() != 255)
             _bomb->setOpacity(255);
       }
    }
    ```

1.  我们通过改变炸弹准备爆炸时的不透明度，给玩家一个额外的视觉提示。

## *刚才发生了什么？*

当你不需要担心更新单个精灵时，主循环相当简单，因为我们的动作会为我们处理这些。我们基本上只需要运行精灵之间的碰撞检测，并确定何时向玩家投掷新的东西。

因此，现在我们唯一要做的就是当计时器结束时从池中抓取流星和健康包。让我们直接开始吧。

# 行动时间 - 从对象池中检索对象

我们只需要使用正确的索引从各自的向量中检索对象：

1.  要检索流星精灵，我们将使用 `resetMeteor` 方法：

    ```cpp
    void GameLayer::resetMeteor(void) {
       //if too many objects on screen, return
        if (_fallingObjects.size() > 30) return;

        auto meteor = _meteorPool.at(_meteorPoolIndex);
          _meteorPoolIndex++;
        if (_meteorPoolIndex == _meteorPool.size()) 
          _meteorPoolIndex = 0;
          int meteor_x = rand() % (int) (_screenSize.width * 0.8f) + _screenSize.width * 0.1f;
       int meteor_target_x = rand() % (int) (_screenSize.width * 0.8f) + _screenSize.width * 0.1f;

        meteor->stopAllActions();
        meteor->setPosition(Vec2(meteor_x, _screenSize.height + meteor->getBoundingBox().size.height * 0.5));

        //create action
        auto  rotate = RotateBy::create(0.5f ,  -90);
        auto  repeatRotate = RepeatForever::create( rotate );
        auto  sequence = Sequence::create (
                   MoveTo::create(_meteorSpeed, Vec2(meteor_target_x, _screenSize.height * 0.15f)),
                   CallFunc::create(std::bind(&GameLayer::fallingObjectDone, this, meteor) ), nullptr);    
      meteor->setVisible ( true );
      meteor->runAction(repeatRotate);
      meteor->runAction(sequence);
     _fallingObjects.pushBack(meteor);
    }
    ```

1.  我们从对象池中获取下一个可用的流星，然后为它的 `MoveTo` 动作随机选择一个起始和结束的 `x` 值。流星从屏幕顶部开始，将移动到底部向城市方向，但每次随机选择 `x` 值。

1.  我们在一个 `RepeatForever` 动作中旋转流星，并使用 `Sequence` 将精灵移动到目标位置，然后在流星达到目标时调用 `fallingObjectDone` 回调。我们通过将我们从池中检索到的新流星添加到 `_fallingObjects` 向量中来完成，这样我们就可以检查与它的碰撞。

1.  检索健康 (`resetHealth`) 精灵的方法基本上是相同的，只是使用 `swingHealth` 动作代替旋转。你会在 `GameLayer.cpp` 中找到该方法已经实现。

## *刚才发生了什么？*

因此，在 `resetGame` 中我们设置计时器，并在 `update` 方法中更新它们。我们使用这些计时器通过从各自的池中获取下一个可用的对象来将流星和健康包添加到屏幕上，然后我们继续运行爆炸炸弹和这些下落对象之间的碰撞。

注意，在 `resetMeteor` 和 `resetHealth` 中，如果屏幕上已经有很多精灵，我们不会添加新的精灵：

```cpp
if (_fallingObjects->size() > 30) return;
```

这样游戏就不会变得荒谬地困难，我们也不会在我们的对象池中耗尽未使用的对象。

我们游戏中最后一点逻辑是我们的`fallingObjectDone`回调，当流星或健康包到达地面时调用，此时它根据玩家是否让精灵通过而奖励或惩罚玩家。

当你查看`GameLayer.cpp`中的那个方法时，你会注意到我们如何使用`->getTag()`来快速确定我们正在处理哪种类型的精灵（调用该方法的那个）：

```cpp
if (pSender->getTag() == kSpriteMeteor) {
```

如果它是一颗流星，我们就减少玩家的能量，播放音效，并运行爆炸动画；我们保留的`_groundHit`动作的一个 autorelease 副本，这样我们就不需要在每次需要运行这个动作时重复所有逻辑。

如果项目是健康包，我们就增加能量或给玩家一些分数，播放一个好听的声音效果，并隐藏精灵。

# 玩这个游戏！

我们疯狂地编码，现在终于到了运行游戏的时候。但首先，别忘了释放我们保留的所有项目。在`GameLayer.cpp`中，添加我们的析构函数方法：

```cpp
GameLayer::~GameLayer () {

    //release all retained actions
    CC_SAFE_RELEASE(_growBomb);
    CC_SAFE_RELEASE(_rotateSprite);
    CC_SAFE_RELEASE(_shockwaveSequence);
    CC_SAFE_RELEASE(_swingHealth);
    CC_SAFE_RELEASE(_groundHit);
    CC_SAFE_RELEASE(_explosion);
    CC_SAFE_RELEASE(_ufoAnimation);
    CC_SAFE_RELEASE(_blinkRay);

    _clouds.clear();
    _meteorPool.clear();
    _healthPool.clear();
    _fallingObjects.clear();
}
```

实际的游戏屏幕现在看起来可能像这样：

![玩这个游戏！](img/00015.jpeg)

再次提醒，如果你在运行代码时遇到任何问题，可以参考`4198_04_FINAL_PROJECT.zip`。

现在，让我们将其带到 Android 上。

# 是时候在 Android 上运行游戏了。

按照以下步骤将游戏部署到 Android：

1.  这次，没有必要修改清单文件，因为默认设置就是我们想要的。所以，导航到`proj.android`，然后到`jni`文件夹，在文本编辑器中打开`Android.mk`文件。

1.  编辑`LOCAL_SRC_FILES`中的行，使其如下所示：

    ```cpp
    LOCAL_SRC_FILES := hellocpp/main.cpp \
                       ../../Classes/AppDelegate.cpp \
                       ../../Classes/GameLayer.cpp 
    ```

1.  按照从`HelloWorld`和`AirHockey`示例中的说明，将游戏导入 Eclipse。

1.  保存并运行你的应用程序。这次，如果你有设备，你可以尝试不同的屏幕尺寸。

## *发生了什么？*

你刚刚在 Android 上运行了一个通用应用。这再简单不过了。

作为奖励，我添加了游戏的另一个版本，增加了额外的敌人类型来处理：一个一心想要电击城市的 UFO！你可以在`4198_04_BONUS_PROJECT.zip`中找到它。

## 突击测验 - 精灵和动作

Q1. `SpriteBatchNode`可以包含哪些类型的元素？

1.  使用来自两个或更多精灵图的纹理的精灵。

1.  使用相同源纹理的精灵。

1.  空精灵。

1.  使用来自一个精灵图和另一个图像的纹理的精灵。

Q2. 为了持续运行一个动作，我需要使用什么？

1.  `RepeatForever`。

1.  `Repeat`。

1.  动作默认的行为是持续运行。

1.  动作不能永远重复。

Q3. 为了使精灵移动到屏幕上的某个点然后淡出，我需要哪些动作？

1.  一个列出`EaseIn`和`EaseOut`动作的`Sequence`。

1.  一个列出`FadeOut`和`MoveTo`动作的`Sequence`。

1.  一个列出`MoveTo`或`MoveBy`和`FadeOut`动作的`Sequence`。

1.  一个列出`RotateBy`和`FadeOut`动作的`Sequence`。

Q4. 要创建一个精灵帧动画，哪些类组是绝对必要的？

1.  `Sprite`, `SpriteBatchNode`, 和 `EaseIn`.

1.  `SpriteFrameCache`, `RotateBy`, 和 `ActionManager`.

1.  `Sprite`, `Layer`, 和 `FadeOut`.

1.  `SpriteFrame`, `Animation`, 和 `Animate`.

# 摘要

在我看来，在节点及其所有派生对象之后，动作是 Cocos2d-x 的第二大优点。它们是节省时间的工具，并且可以快速为任何项目增添专业外观的动画。我希望通过本章中的示例，以及 Cocos2d-x 样本测试项目中的示例，您将能够使用 Cocos2d-x 创建您需要的任何动作。

在下一章中，我将向您介绍另一种简单的方法，您可以用它来为您的游戏增添活力：使用粒子效果！
