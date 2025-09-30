# 第五章. 在线上 – 火箭穿越

*在我们的第三个游戏，火箭穿越中，我们将使用粒子效果来增加一些趣味性，并且我们将使用 DrawNode 在屏幕上绘制自己的 OpenGL 图形。并且请注意，这个游戏使用了相当多的向量数学，但幸运的是，Cocos2d-x 附带了一套甜美的辅助方法来处理这些问题。*

你将学习：

+   如何加载和设置粒子系统

+   如何使用`DrawNode`绘制原语（线条、圆圈等）

+   如何使用 Cocos2d-x 中包含的向量数学辅助方法

# 游戏 – 火箭穿越

在这个经典蛇形游戏引擎的科幻版本中，你控制一艘火箭船，必须在七个行星之间移动，收集微小的超新星。但这里有个问题：你只能通过通过`touch`事件放置的支点旋转来控制火箭。所以，我们为火箭船设定的运动矢量有时是线性的，有时是圆形的。

## 游戏设置

这是一个通用游戏，专为普通 iPad 设计，然后放大和缩小以匹配其他设备的屏幕分辨率。它设置为在纵向模式下播放，并且不支持多点触控。

## 先玩，后工作

从本书的**支持**页面下载`4198_05_START_PROJECT.zip`和`4198_05_FINAL_PROJECT.zip`文件。

你将再次使用**开始项目**选项来工作；这样，你就不需要输入已经在之前章节中覆盖的逻辑或语法。**开始项目**选项包含所有资源文件和所有类声明，以及类实现文件中的所有方法的占位符。我们稍后会介绍这些内容。

你应该运行最终项目版本，以便熟悉游戏。通过按住并拖动你的手指在火箭船上，你可以画一条线。释放触摸，你创建一个支点。船将围绕这个支点旋转，直到你再次按下船来释放它。你的目标是收集明亮的超新星并避开行星。

![先玩，后工作](img/00016.jpeg)

## 开始项目

如果你运行**开始项目**选项，你应该能看到基本的游戏屏幕已经就位。没有必要重复我们在之前的教程中创建批节点和放置所有屏幕精灵的步骤。我们再次有一个`_gameBatchNode`对象和一个`createGameScreen`方法。

但是无论如何，请阅读`createGameScreen`方法内部的代码。这里的关键重要性在于，我们创建的每个行星都存储在`_planets`向量中。我们还在这里创建了我们的`_rocket`对象（`Rocket`类）和我们的`_lineContainer`对象（`LineContainer`类）。关于这些内容，我们稍后会详细介绍。

在**开始项目**选项中，我们还有我们的老朋友`GameSprite`，在这里它扩展了`Sprite`，并增加了一个获取精灵的`radius()`方法。`Rocket`对象和所有行星都是`GameSprite`对象。

## 屏幕设置

所以如果你在 Xcode 中打开了 **Start Project** 选项，让我们回顾一下 `AppDelegate.cpp` 中这个游戏的屏幕设置。在 `applicationDidFinishLaunching` 方法内部，你应该看到以下内容：

```cpp
auto designSize = Size(1536, 2048);

glview->setDesignResolutionSize(designSize.width, designSize.height, ResolutionPolicy::EXACT_FIT);

std::vector<std::string> searchPaths;
if (screenSize.width > 768) {
  searchPaths.push_back("ipadhd");
  director->setContentScaleFactor(1536/designSize.width);
} else if (screenSize.width > 320) {
  searchPaths.push_back("ipad");
  director->setContentScaleFactor(768/designSize.width);
} else {
  searchPaths.push_back("iphone");
  director->setContentScaleFactor(380/designSize.width);
}
auto fileUtils = FileUtils::getInstance();
fileUtils->setSearchPaths(searchPaths);
```

因此，我们基本上是以与上一款游戏相同的方式开始的。这款游戏中的大多数精灵都是圆形的，你可能会在不同的屏幕上注意到一些扭曲；你应该测试相同的配置，但使用不同的 `ResolutionPolicies`，例如 `SHOW_ALL`。

# 那么，粒子是什么？

粒子或粒子系统是向你的应用程序添加特殊效果的一种方式。一般来说，这是通过使用大量的小纹理精灵（粒子）来实现的，这些粒子被动画化并通过一系列变换运行。你可以使用这些系统来创建烟雾、爆炸、火花、闪电、雨、雪以及其他类似效果。

正如我在 第一章 中提到的，*安装 Cocos2d-x*，你应该认真考虑为自己获取一个程序来帮助你设计粒子系统。在这款游戏中，粒子是在 ParticleDesigner 中创建的。

是时候将它们添加到我们的游戏中了！

# 行动时间 - 创建粒子系统

对于粒子，我们只需要描述粒子系统属性的 XML 文件。

1.  所以让我们去 `GameLayer.cpp`。

1.  游戏通过调用 `createGameScreen` 初始化，这已经就位，然后是 `createParticles` 和 `createStarGrid`，这也是已实现的。所以现在让我们来看看 `createParticles` 方法。

1.  前往 `GameLayer.cpp` 中的那个方法，并添加以下代码：

    ```cpp
    _jet = ParticleSystemQuad::create("jet.plist");
    _jet->setSourcePosition(Vec2(-_rocket->getRadius() * 0.8f,0));
    _jet->setAngle(180);
    _jet->stopSystem();
    this->addChild(_jet, kBackground);

    _boom = ParticleSystemQuad::create("boom.plist");
    _boom->stopSystem();
    this->addChild(_boom, kForeground);

    _comet = ParticleSystemQuad::create("comet.plist");
    _comet->stopSystem();
    _comet->setPosition(Vec2(0, _screenSize.height * 0.6f));
    _comet->setVisible(false);
    this->addChild(_comet, kForeground);

    _pickup = ParticleSystemQuad::create("plink.plist");
    _pickup->stopSystem();
    this->addChild(_pickup, kMiddleground);

    _warp = ParticleSystemQuad::create("warp.plist");
    _warp->setPosition(_rocket->getPosition());
    this->addChild(_warp, kBackground);

    _star = ParticleSystemQuad::create("star.plist");
    _star->stopSystem();
    _star->setVisible(false);
    this->addChild(_star, kBackground, kSpriteStar);
    ```

## *刚才发生了什么？*

我们创建了第一个粒子。ParticleDesigner 将粒子系统数据导出为 `.plist` 文件，我们使用它来创建我们的 `ParticleSystemQuad` 对象。你应该在 Xcode 中打开其中一个文件来查看粒子系统中使用的设置数量。从 Cocos2d-x 中，你可以通过 `ParticleSystem` 中的设置器修改这些设置中的任何一个。

我们将在游戏中使用的粒子如下：

+   `_jet`: 这与 `_rocket` 对象相关联，并且它将在 `_rocket` 对象的后面拖尾。我们将系统的角度和源位置参数设置为与 `_rocket` 精灵匹配。

+   `_boom`: 这是在 `_rocket` 爆炸时使用的粒子系统。

+   `_comet`: 这是一个在设定间隔内移动穿过屏幕的粒子系统，并且可以与 `_rocket` 发生碰撞。

+   `_pickup`: 这用于收集星星时。

+   `_warp`: 这标记了火箭的初始位置。

+   `_star`: 这是火箭必须收集的星星所使用的粒子系统。

以下截图显示了这些各种粒子：

![刚才发生了什么？](img/00017.jpeg)

所有粒子系统都作为 `GameLayer` 的子对象添加；它们不能添加到我们的 `SpriteBatchNode` 类中。并且，在创建每个系统时，你必须调用 `stopSystem()`，否则它们一旦被添加到节点中就会立即开始播放。

为了运行系统，你需要调用`resetSystem()`。

### 注意

Cocos2d-x 附带了一些常见的粒子系统，你可以根据需要修改。如果你去`test`文件夹中的`tests/cpp-tests/Classes/ParticleTest`，你会看到这些系统被使用的示例。实际的粒子数据文件位于：`tests/cpp-tests/Resources/Particles`。

# 创建网格

现在我们花些时间来回顾一下游戏中的网格逻辑。这个网格是在`GameLayer.cpp`中的`createStarGrid`方法中创建的。这个方法所做的就是确定屏幕上所有可以放置`_star`粒子系统的可能位置。

我们使用一个名为`_grid`的 C++向量列表来存储可用的位置：

```cpp
std::vector<Point> _grid;
```

`createStarGrid`方法将屏幕划分为多个 32 x 32 像素的单元格，忽略离屏幕边缘太近的区域（`gridFrame`）。然后我们检查每个单元格与存储在向量`_planets`中的星球精灵之间的距离。如果单元格离星球足够远，我们就将其作为`Point`存储在`_grid`向量中。

在以下图中，你可以了解我们想要达到的结果。我们想要所有不与任何星球重叠的白色单元格。

![创建网格](img/00018.jpeg)

我们使用`Log`向控制台输出一条消息，说明我们最终有多少个单元格：

```cpp
CCLOG("POSSIBLE STARS: %i", _grid.size());
```

这个`vector`列表将在每次新游戏中进行洗牌，所以我们最终得到一个可能的星星位置随机序列：

```cpp
std::random_shuffle(_grid.begin(), _grid.end());
```

这样我们就不会在星球上方或离它如此近的地方放置星星，以至于火箭无法到达它而不与星球相撞。

# 在 Cocos2d-x 中绘制原语

游戏中的主要元素之一是`LineContainer.cpp`类。它是一个从`DrawNode`派生出来的类，允许我们在屏幕上绘制线条和圆圈。

`DrawNode`附带了一系列你可以用来绘制线条、点、圆圈、多边形等的绘制方法。

我们将使用的方法是`drawLine`和`drawDot`。

# 是时候动手画一些东西了！

是时候在`LineContainer.cpp`中实现绘制功能了。你会注意到这个类已经实现了大部分方法，所以你可以节省一些输入。一旦我们添加了游戏的主要更新方法，我会解释这些方法代表什么。但基本上`LineContainer`将用于显示玩家在屏幕上绘制的线条，以操纵`_rocket`精灵，以及显示一个充当游戏计时器的能量条：

1.  我们需要在这里更改的是`update`方法。所以这就是你需要在那个方法中输入的内容：

    ```cpp
    _energy -= dt * _energyDecrement;
    if (_energy < 0) _energy = 0;
    clear();

    switch (_lineType) {
      case LINE_NONE:
       break;
      case LINE_TEMP:
       drawLine(_tip, _pivot, Color4F(1.0, 1.0, 1.0, 1.0));
       drawDot(_pivot, 5, Color4F(Color3B::WHITE));
       break;

      case LINE_DASHED:
       drawDot(_pivot, 5, Color4F(Color3B::WHITE));
       int segments = _lineLength / (_dash + _dashSpace);
       float t = 0.0f;
       float x_;
       float y_;

       for (int i = 0; i < segments + 1; i++) {
          x_ = _pivot.x + t * (_tip.x - _pivot.x);
          y_ = _pivot.y + t * (_tip.y - _pivot.y);
          drawDot(Vec2(x_, y_), 5, Color4F(Color3B::WHITE));
          t += (float) 1 / segments;
       }
       break;
    }
    ```

1.  我们通过在同一个`LineContainer`节点上绘制能量条来结束我们的绘制调用：

    ```cpp
    drawLine(Vec2(_energyLineX, _screenSize.height * 0.1f),  Vec2(_energyLineX, _screenSize.height * 0.9f), Color4F(0.0, 0.0, 0.0, 1.0)); 
    drawLine(Vec2(_energyLineX, _screenSize.height * 0.1f),  Vec2(_energyLineX, _screenSize.height * 0.1f + _energy *  _energyHeight ), Color4F(1.0, 0.5, 0.0, 1.0));
    ```

## *刚才发生了什么？*

你刚刚学习了如何在`DrawNode`中绘制。代码中的一条重要行是`clear()`调用。在我们用新状态更新它们之前，它会清除该节点中的所有绘制。

在`LineContainer`中，我们使用`switch`语句来确定如何绘制玩家的线。如果`_lineType`属性设置为`LINE_NONE`，则不绘制任何内容（这实际上会清除玩家所做的任何绘图）。

如果`_lineType`是`LINE_TEMP`，这意味着玩家正在将手指从`_rocket`对象上拖开，我们想显示从`_rocket`当前位置到玩家当前触摸位置的白线。这些点分别称为`tip`和`pivot`。

我们还在`pivot`点上画了一个点。

```cpp
drawLine(_tip, _pivot, Color4F(1.0, 1.0, 1.0, 1.0));
drawDot(_pivot, 5, Color4F(Color3B::WHITE));
```

如果`_lineType`是`LINE_DASHED`，这意味着玩家已经从屏幕上移除了手指，并为`_rocket`设置了一个新的旋转支点。我们用所谓的贝塞尔线性公式画一条白点线，从`_rocket`当前位置和`pivot`点画一系列小圆：

```cpp
for (int i = 0; i < segments + 1; i++) {

    x_ = _pivot.x + t * (_tip.x - _pivot.x);
    y_ = _pivot.y + t * (_tip.y - _pivot.y);

    drawDot(Vec2(x_, y_), 5, Color4F(Color3B::WHITE));
    t += (float) 1 / segments;
}
```

最后，对于能量条，我们在橙色条下方画一条黑色线。当`LineContainer`中的`_energy`值减少时，橙色条会调整大小。黑色线保持不变，它在这里是为了显示对比。你通过`draw`调用的顺序来叠加你的绘图；所以先画的东西会出现在后画的东西下面。

# 火箭精灵

现在是处理游戏中的第二个对象：火箭。

再次强调，我已经放好了对你来说已经是老生常谈的逻辑部分。但请审查已经放在`Rocket.cpp`中的代码。我们有一个方法，每次新游戏开始时重置火箭（`reset`），还有一个方法通过改变其显示纹理来显示火箭的选中状态（`select(bool flag)`）：

```cpp
if (flag) {
    this->setDisplayFrame(SpriteFrameCache::getInstance()->getSpriteFrameByName("rocket_on.png"));
} else {
    this->setDisplayFrame(SpriteFrameCache::getInstance()->getSpriteFrameByName("rocket.png"));
}
```

这将显示火箭周围有光晕，或者不显示。

最后，我们有一个检查与屏幕边缘碰撞的方法（`collidedWithSides`）。如果有碰撞，我们将调整火箭使其远离碰撞的屏幕边缘，并从任何支点位置释放它。

我们真正需要担心的是火箭的`update`方法。这就是我们接下来要添加的。

# 更新我们的火箭精灵的行动时间

游戏的主循环会在每次迭代中调用火箭的`update`方法。

1.  在`Rocket.cpp`中的空`update`方法内，添加以下行：

    ```cpp
    Point position = this->getPosition();
    if (_rotationOrientation == ROTATE_NONE) {
      position.x += _vector.x * dt;
      position.y += _vector.y * dt;
    } else {
      float angle = _angularSpeed * dt;
      Point rotatedPoint = position.rotateByAngle(_pivot, angle);
      position.x = rotatedPoint.x;
      position.y = rotatedPoint.y;
      float rotatedAngle;

      Point diff = position;
      diff.subtract(_pivot);
      Point clockwise = diff.getRPerp();

      if (_rotationOrientation == ROTATE_COUNTER) {
        rotatedAngle = atan2 (-1 * clockwise.y, -1 * clockwise.x);
      } else {
        rotatedAngle = atan2 (clockwise.y, clockwise.x);
      }

      _vector.x = _speed * cos (rotatedAngle);
      _vector.y = _speed * sin (rotatedAngle);
      this->setRotationFromVector();

      if (this->getRotation() > 0) {
        this->setRotation( fmodf(this->getRotation(), 360.0f) );
      } else {
        this->setRotation( fmodf(this->getRotation(), -360.0f) );
      }
    }
    ```

1.  这里我们说的是，如果火箭没有旋转（`_rotationOrientation == ROTATE_NONE`），就根据其当前`_vector`移动它。如果它在旋转，则使用 Cocos2d-x 辅助函数`rotateByAngle`方法找到其绕支点旋转的下一个位置：![更新我们的火箭精灵的行动时间 – 更新我们的火箭精灵](img/00019.jpeg)

1.  该方法将围绕一个支点旋转任意点一定角度。因此，我们使用`Rocket`类的一个属性`_angularSpeed`来旋转火箭的更新位置（由玩家确定），我们稍后会看到它是如何计算的。

1.  根据火箭是顺时针旋转还是逆时针旋转，我们调整其旋转，使火箭与火箭和其支点之间绘制的线条成 90 度角。然后我们根据这个旋转角度改变火箭的运动矢量，并将该角度的值包裹在 0 到 360 度之间。

1.  使用以下行完成 `update` 方法的编写：

    ```cpp
    if (_targetRotation > this->getRotation() + 180) {
      _targetRotation -= 360;
    }
    if (_targetRotation < this->getRotation() - 180) {
      _targetRotation += 360;
    }

    this->setPosition(position);
    _dr = _targetRotation - this->getRotation() ;
    _ar = _dr * _rotationSpring;
    _vr += _ar ;
    _vr *= _rotationDamping;
    float rotationNow = this->getRotation();
    rotationNow += _vr;
    this->setRotation(rotationNow);
    ```

1.  通过这些行我们确定精灵的新目标旋转，并运行一个动画将火箭旋转到目标旋转（带有一点弹性）。

## *刚才发生了什么？*

我们刚刚编写了将火箭在屏幕上移动的逻辑，无论火箭是否旋转。

因此，当玩家为 `_rocket` 精灵选择一个支点时，这个支点被传递给 `Rocket` 和 `LineContainer`。前者将使用它在其周围旋转矢量，后者将使用它来在 `_rocket` 和 `pivot` 点之间绘制虚线。

### 注意

我们不能使用 `Action` 来旋转精灵，因为我们的逻辑中目标旋转更新得太频繁，而 `Action` 需要时间来初始化和运行。

因此，现在是时候编写触摸事件代码，让所有这些逻辑都落到实处。

# 行动时间 - 处理触摸

我们需要实现 `onTouchBegan`、`onTouchMoved` 和 `onTouchEnded`。

1.  现在，在 `GameLayer.cpp` 中，在 `onTouchBegan` 中添加以下行：

    ```cpp
    if (!_running) return true;
    Point tap = touch->getLocation();
    float dx = _rocket->getPositionX() - tap.x;
    float dy = _rocket->getPositionY() - tap.y;
    if (dx * dx + dy * dy <= pow(_rocket->getRadius(), 2) ) {
     _lineContainer->setLineType ( LINE_NONE );
     _rocket->setRotationOrientation ( ROTATE_NONE );
     _drawing = true;
    }

    return true;
    ```

    当触摸开始时，我们只需要确定它是否触摸了飞船。如果是，我们将我们的 `_drawing` 属性设置为 `true`。这将表明我们有一个有效的点（一个从触摸 `_rocket` 精灵开始的点）。

1.  我们通过调用 `setLineType( LINE_NONE )` 清除 `_lineContainer` 中可能正在绘制的任何线条，并确保通过释放 `_rocket (setRotationOrientation ( ROTATE_NONE ))`，我们不会旋转 `_rocket`，直到我们有一个支点，这样它将继续沿着当前的线性轨迹 `(_vector)` 移动。

1.  从这里开始，我们使用下一个 `onTouchMoved` 方法绘制新的线条。在该方法内部，我们添加以下行：

    ```cpp
    if (!_running) return;
      if (_drawing) {
         Point tap = touch->getLocation();
         float dx = _rocket->getPositionX() - tap.x;
         float dy = _rocket->getPositionY() - tap.y;
         if (dx * dx + dy * dy > pow (_minLineLength, 2)) {
           _rocket->select(true);
           _lineContainer->setPivot ( tap );
           _lineContainer->setLineType ( LINE_TEMP );
         } else {
           _rocket->select(false);
           _lineContainer->setLineType ( LINE_NONE );
        }
     }
    ```

1.  我们只处理触摸移动，如果我们正在使用 `_drawing`，这意味着玩家已经按下了飞船，现在正在将手指拖过屏幕。

    一旦手指与 `_rocket` 之间的距离大于我们在游戏 `init` 中规定的 `_` `minLineLength` 距离，我们就通过在 `_rocket` 周围添加发光效果（_rocket->select(true)）向玩家提供一个视觉提示，并在 `_lineContainer` 中通过传递触摸的当前位置并设置线型为 `LINE_TEMP` 来绘制新的线条。如果未达到最小长度，则不显示线条，也不显示玩家已选择。

1.  接下来是 `onTouchEnded`。在我们的 `onTouchEnded` 方法中已经存在处理游戏状态的逻辑。你应该取消注释对 `resetGame` 的调用，并在方法内添加一个新的 `else if` 语句：

    ```cpp
    } else if (_state == kGamePaused) {
      _pauseBtn->setDisplayFrame(SpriteFrameCache::getInstance()->getSpriteFrameByName ("btn_pause_off.png"));
      _paused->setVisible(false);
      _state = kGamePlay;
      _running = true;
      return;
    } 
    ```

1.  如果游戏处于暂停状态，我们通过 `Sprite->setDisplayFrame` 在 `_pauseBtn` 精灵中更改纹理，并重新开始运行游戏。

1.  现在我们开始处理触摸。首先，我们确定它是否落在 `Pause` 按钮上：

    ```cpp
    if (!_running) return;
    if(touch != nullptr) {
      Point tap = touch->getLocation();
      if (_pauseBtn->getBoundingBox().containsPoint(tap)) {
        _paused->setVisible(true);
        _state = kGamePaused;
        _pauseBtn->setDisplayFrame(SpriteFrameCache::getInstance()->getSpriteFrameByName ("btn_pause_on.png"));
        _running = false;
        return;
      }
    }
    ```

1.  如果是这样，我们将游戏状态更改为 `kGamePaused`，在 `_pauseBtn` 精灵上更改纹理（通过从 `SpriteFrameCache` 中检索另一个精灵帧），停止运行游戏（暂停游戏），并从函数中返回。

1.  我们终于可以对火箭飞船做些事情了。所以，继续在之前看到的 `if(touch != nullptr) {` 条件语句中，添加以下行：

    ```cpp
        _drawing = false;
       _rocket->select(false);
       if (_lineContainer->getLineType() == LINE_TEMP) {
          _lineContainer->setPivot (tap);
          _lineContainer->setLineLength ( _rocket->getPosition().distance( tap ) );
          _rocket->setPivot (tap);
    ```

1.  我们首先取消选择 `_rocket` 精灵，然后检查我们是否正在 `_lineContainer` 中显示临时线条。如果我们正在显示，这意味着我们可以继续使用玩家的释放触摸来创建新的支点。我们通过 `setPivot` 方法将此信息传递给 `_lineContainer`，同时传递线条长度。`_rocket` 精灵也会接收到支点信息。

    然后，事情变得复杂！`_rocket` 精灵以像素为基础的速度移动。一旦 `_rocket` 开始旋转，它将通过 `Point.rotateByAngle` 以基于角的速度移动。因此，以下行被添加以将 `_rocket` 当前像素速度转换为角速度：

    ```cpp
    float circle_length = _lineContainer->getLineLength() * 2 * M_PI;
    int iterations = floor(circle_length / _rocket->getSpeed());
    _rocket->setAngularSpeed ( 2 * M_PI / iterations);
    ```

1.  它获取将要被 `_rocket` 描述的圆周长度（`_rocket (line length * 2 * PI)`），然后除以火箭的速度，得到火箭完成该长度所需的迭代次数。然后，将圆的 360 度除以相同的迭代次数（但我们用弧度来计算）以得到火箭在每次迭代中必须旋转的圆周分数：它的角速度。

1.  接下来的是更多的数学计算，使用 Cocos2d-x 中与向量数学相关的非常有帮助的方法（例如 `Point.getRPerp`、`Point.dot`、`Point.subtract` 等），其中一些我们在 `Rocket` 类中已经见过：

    ```cpp
    Vec2 diff = _rocket->getPosition();
    diff.subtract(_rocket->getPivot());
    Point clockwise = diff.getRPerp();
    float dot =clockwise.dot(_rocket->getVector());
    if (dot > 0) {
       _rocket->setAngularSpeed ( _rocket->getAngularSpeed() * -1 );
       _rocket->setRotationOrientation ( ROTATE_CLOCKWISE );
       _rocket->setTargetRotation  ( CC_RADIANS_TO_DEGREES( atan2(clockwise.y, clockwise.x) ) );
    } else {
       _rocket->setRotationOrientation ( ROTATE_COUNTER );
       _rocket->setTargetRotation ( CC_RADIANS_TO_DEGREES  (atan2(-1 * clockwise.y, -1 * clockwise.x) ) );
    }
    _lineContainer->setLineType ( LINE_DASHED );
    ```

1.  他们在这里做的是确定火箭应该旋转的方向：顺时针还是逆时针，基于其当前的运动向量。

1.  玩家刚刚在 `_rocket` 和支点之间绘制的线条，通过减去这两个点（`Point.subtract`）得到，有两个垂直向量：一个向右（顺时针）的向量，通过 `Point.getRPerp` 获取；一个向左（逆时针）的向量，通过 `Point.getPerp` 获取。我们使用其中一个向量的角度作为 `_rocket` 目标旋转，使火箭旋转到与 `LineContainer` 中绘制的线条成 90 度，并通过 `_rocket` 当前向量与其中一个垂直向量的点积（`Point.dot`）找到正确的垂直向量。

## *发生了什么？*

我知道。有很多数学计算，而且都是一次性完成的！幸运的是，Cocos2d-x 使这一切处理起来容易得多。

我们刚刚添加了允许玩家绘制线条并设置 `_rocket` 精灵的新支点的逻辑。

玩家将通过给火箭一个旋转的支点来控制`_rocket`精灵穿越行星。通过释放`_rocket`从支点，玩家将使其再次沿直线移动。所有这些逻辑都由游戏中的触摸事件管理。

并且不用担心数学问题。虽然理解如何处理向量是任何游戏开发者工具箱中非常有用的工具，你应该绝对研究这个话题，但你可以用很少或没有数学知识来构建无数的游戏；所以加油！

# 游戏循环

是时候创建我们熟悉的老式计时器了！主循环将负责碰撞检测，更新`_lineContainer`内的分数，调整`_jet`粒子系统以匹配`_rocket`精灵，以及一些其他事情。

# 行动时间 – 添加主循环

让我们实现我们的主要`update`方法。

1.  在`GameLayer.cpp`中，在`update`方法内部，添加以下行：

    ```cpp
    if (!_running || _state != kGamePlay) return;
    if (_lineContainer->getLineType() != LINE_NONE) {
      _lineContainer->setTip (_rocket->getPosition() );
    }

    if (_rocket->collidedWithSides()) {
      _lineContainer->setLineType ( LINE_NONE );
    }
    _rocket->update(dt);

    //update jet particle so it follows rocket
    if (!_jet->isActive()) _jet->resetSystem();
    _jet->setRotation(_rocket->getRotation());
    _jet->setPosition(_rocket->getPosition());
    ```

    我们检查我们是否不在暂停状态。然后，如果有我们需要在`_lineContainer`中显示的船的线，我们使用`_rocket`的当前位置更新线的`tip`点。

    我们在`_rocket`和屏幕边缘之间运行碰撞检测，更新`_rocket`精灵，并定位和旋转我们的`_jet`粒子系统以与`_rocket`精灵对齐。

1.  接下来我们更新`_comet`（它的倒计时、初始位置、移动和如果`_comet`可见则与`_rocket`的碰撞）：

    ```cpp
    _cometTimer += dt;
    float newY;

    if (_cometTimer > _cometInterval) {
        _cometTimer = 0;
        if (_comet->isVisible() == false) {
            _comet->setPositionX(0);
            newY = (float)rand()/((float)RAND_MAX/_screenSize.height * 0.6f) + _screenSize.height * 0.2f;
            if (newY > _screenSize.height * 0.9f) 
               newY = _screenSize.height * 0.9f;
               _comet->setPositionY(newY);
               _comet->setVisible(true);
               _comet->resetSystem();
        }
    }

    if (_comet->isVisible()) {
        //collision with comet
        if (pow(_comet->getPositionX() - _rocket->getPositionX(), 2) + pow(_comet->getPositionY() - _rocket->getPositionY(), 2) <= pow (_rocket->getRadius() , 2)) {
            if (_rocket->isVisible()) killPlayer();
        }
        _comet->setPositionX(_comet->getPositionX() + 50 * dt);

        if (_comet->getPositionX() > _screenSize.width * 1.5f) {
            _comet->stopSystem();
            _comet->setVisible(false);
        }
    }
    ```

1.  接下来我们更新`_lineContainer`，并逐渐降低`_rocket`精灵的透明度，基于`_lineContainer`中的`_energy`等级：

    ```cpp
    _lineContainer->update(dt);
    _rocket->setOpacity(_lineContainer->getEnergy() * 255);
    ```

    这将为玩家添加一个视觉提示，表明时间正在流逝，因为`_rocket`精灵将逐渐变得不可见。

1.  运行星球的碰撞：

    ```cpp
    for (auto planet : _planets) {
        if (pow(planet->getPositionX() - _rocket->getPositionX(),  2)
        + pow(planet->getPositionY() - _rocket->getPositionY(), 2)  <=   pow (_rocket->getRadius() * 0.8f + planet->getRadius()  * 0.65f, 2)) {

            if (_rocket->isVisible()) killPlayer();
            break;
        }
    }
    ```

1.  并且与星星的碰撞：

    ```cpp
    if (pow(_star->getPositionX() - _rocket->getPositionX(), 2)
        + pow(_star->getPositionY() - _rocket->getPositionY(), 2)  <=
        pow (_rocket->getRadius() * 1.2f, 2)) {

        _pickup->setPosition(_star->getPosition());
        _pickup->resetSystem();
        if (_lineContainer->getEnergy() + 0.25f < 1) {
            _lineContainer->setEnergy(_lineContainer->getEnergy() +  0.25f);
        } else {
            _lineContainer->setEnergy(1.0);
        }
        _rocket->setSpeed(_rocket->getSpeed() + 2);
        if (_rocket->getSpeed() > 70) _rocket->setSpeed(70);
            _lineContainer->setEnergyDecrement(0.0002f);
            SimpleAudioEngine::getInstance()->playEffect("pickup.wav");
            resetStar();

            int points = 100 - _timeBetweenPickups;
            if (points < 0) points = 0;

            _score += points;
            _scoreDisplay->setString(String::createWithFormat("%i", _score)->getCString());
            _timeBetweenPickups = 0;
    }
    ```

    当我们收集到`_star`时，我们在`_star`所在的位置激活`_pickup`粒子系统，填充玩家的能量等级，使游戏稍微困难一些，并立即将`_star`重置到下一个位置以便再次收集。

    分数基于玩家收集`_star`所需的时间。

1.  我们在`update`函数的最后几行记录这个时间，同时检查能量等级：

    ```cpp
    _timeBetweenPickups += dt;
    if (_lineContainer->getEnergy() == 0) {
        if (_rocket->isVisible()) killPlayer();
    }
    ```

## *刚刚发生了什么？*

我们已经将主循环添加到游戏中，并且所有的部件都开始相互通信。但你可能已经注意到，我们调用了一些尚未实现的方法，比如`killPlayer`和`resetStar`。我们将使用这些方法完成游戏逻辑。

# 销毁和重置

又是时候了！是时候杀死我们的玩家并重置游戏了！我们还需要在玩家拾取`_star`时将其精灵移动到新的位置。

# 行动时间 – 添加重置和销毁

我们需要添加逻辑来重新开始游戏，并将我们的拾取星移动到新的位置。但首先，让我们销毁玩家！

1.  在`killPlayer`方法内部，添加以下行：

    ```cpp
    void GameLayer::killPlayer() {

        SimpleAudioEngine::getInstance()->stopBackgroundMusic();
        SimpleAudioEngine::getInstance()->stopAllEffects();
        SimpleAudioEngine::getInstance()->playEffect("shipBoom.wav");

        _boom->setPosition(_rocket->getPosition());
        _boom->resetSystem();
        _rocket->setVisible(false);
        _jet->stopSystem();
        _lineContainer->setLineType ( LINE_NONE );

        _running = false;
        _state = kGameOver;
        _gameOver->setVisible(true);
        _pauseBtn->setVisible(false);
    }
    ```

1.  在`resetStar`内部，添加以下行：

    ```cpp
    void GameLayer::resetStar() {
        Point position = _grid[_gridIndex];
        _gridIndex++;
        if (_gridIndex == _grid.size()) _gridIndex = 0;
        //reset star particles
        _star->setPosition(position);
        _star->setVisible(true);
        _star->resetSystem();
    }
    ```

1.  最后，我们的`resetGame`方法：

    ```cpp
    void GameLayer::resetGame () {

        _rocket->setPosition(Vec2(_screenSize.width * 0.5f,  _screenSize.height * 0.1f));
        _rocket->setOpacity(255);
        _rocket->setVisible(true);
        _rocket->reset();

        _cometInterval = 4;
        _cometTimer = 0;
        _timeBetweenPickups = 0.0;

        _score = 0;
        _scoreDisplay->setString(String::createWithFormat("%i", _score)->getCString());

        _lineContainer->reset();

        //shuffle grid cells

        std::random_shuffle(_grid.begin(), _grid.end());
        _gridIndex = 0;

        resetStar();

        _warp->stopSystem();

        _running = true;

        SimpleAudioEngine::getInstance()->playBackgroundMusic("background.mp3", true);
        SimpleAudioEngine::getInstance()->stopAllEffects();
        SimpleAudioEngine::getInstance()->playEffect("rocket.wav", true);

    }
    ```

## *刚刚发生了什么？*

就这样。我们完成了。这比大多数人舒服的数学要多。但你能告诉我什么呢，我就是喜欢玩弄向量！

现在，让我们继续学习 Android！

# 是时候行动了——在 Android 上运行游戏

按照以下步骤将游戏部署到 Android：

1.  打开清单文件，并将`app`方向设置为`portrait`。

1.  接下来，在文本编辑器中打开`Android.mk`文件。

1.  编辑`LOCAL_SRC_FILES`中的行，使其读取：

    ```cpp
    LOCAL_SRC_FILES := hellocpp/main.cpp \
                       ../../Classes/AppDelegate.cpp \
                       ../../Classes/GameSprite.cpp \
                       ../../Classes/LineContainer.cpp \
                       ../../Classes/Rocket.cpp \
                       ../../Classes/GameLayer.cpp  
    ```

1.  将游戏导入 Eclipse 并构建它。

1.  保存并运行你的应用程序。这次，如果你有设备，你可以尝试不同的屏幕尺寸。

## *刚才发生了什么？*

现在，你的 Rocket Through 已经在 Android 上运行了。

## 大胆尝试

给`resetStar`方法添加逻辑，以确保新选择的位置不要离`_rocket`精灵太近。所以，让这个函数成为一个循环的函数，直到选择了一个合适的位置。

并且使用`warp`粒子系统，目前它并没有做什么，将其用作随机传送场，这样火箭可能会被随机放置的传送门吸入，并远离目标恒星。

# 摘要

恭喜！你现在对 Cocos2d-x 有了足够的信息来制作出色的 2D 游戏。首先是精灵，然后是动作，现在是粒子。

粒子让一切看起来都很闪亮！它们很容易实现，并且是给游戏添加额外动画的好方法。但是很容易过度使用，所以请小心。你不想让你的玩家出现癫痫发作。此外，一次性运行太多粒子可能会让你的游戏停止运行。

在下一章中，我们将看到如何使用 Cocos2d-x 快速测试和开发游戏想法。
