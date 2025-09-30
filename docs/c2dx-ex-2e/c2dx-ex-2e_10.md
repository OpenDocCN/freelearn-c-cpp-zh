# 第十章。介绍 Lua！

*在我们的上一场游戏中，我们将转向新的 Cocos IDE，并使用 Lua 脚本语言开发整个游戏。你将了解并使用 Cocos2d-x API 的 Lua 绑定，这与我们之前在 C++中使用的大致相同；如果有什么不同的话，那就是它更容易！*

这次，你将学习如何：

+   在 Cocos IDE 中创建和发布一个项目

+   使用 Lua 编写整个游戏

+   使用精灵、粒子、标签、菜单和动作，但这次使用 Lua 绑定

+   构建一个三消游戏

# 那 Lua 是什么样的呢？

在 Lua 的核心（在葡萄牙语中意为月亮），你有表。你可以把它想象成类似于 JavaScript 对象，但它远不止如此。它扮演着数组、字典、枚举、结构和类等角色。这使得 Lua 成为管理大量数据的完美语言。你编写一个处理数据的脚本，然后不断地给它提供不同的“东西”。存货或商店系统、互动儿童书——这些类型的项目都可以从 Lua 以表为中心的强大功能中受益，因为它们可以围绕一个固定模板和数据表的核心来构建。

对于不习惯脚本语言的人来说，其语法可能有点奇怪，因为它有 dos、thens 和 ends。但一旦你克服了这个初步的障碍，你会发现 Lua 非常用户友好。以下是它语法中的一些“奇怪之处”：

```cpp
-- a comment
--[[ 
a 
multiline 
comment 
]]
-- a table declared as a local variable
local myTable = {}
-- the length of a table
local len = #myTable
-- looping the table (starting with index 1!)
for i = 1, #myTable do
   local element = myTable[i]
   -- an if elseif else statement
   if (element ~= true ) then
      -- do something
   elseif (element == true) then
      -- do something else
   else
      -- we'll never get here!   
   end
end
```

### 注意

分号是可选的。

表可以被转换成模板以生成其实例，换句话说，就是一个类。必须使用 `:` 符号来访问表的实例方法：

```cpp
myTableClassObject:myMethod()
```

在方法内部，你将类的实例称为 `self`：

```cpp
self.myProperty = 1
self:myOtherMethod()
```

或者，你可以使用点符号调用模板的方法，将模板的实例作为第一个参数传递给它：

```cpp
myTableClassObject.myMethod(myTableClassObject)
```

我承认，这听起来很奇怪，但有时它很有用，因为你在 Lua 中编写的几乎所有方法都可以供代码的其他部分使用——有点像传统面向对象语言中静态方法的使用方式。

## Lua 中的调试 – 说 nil 的骑士

调试 Lua 代码有时可能会让人感到沮丧。但很快你就会学会区分 Lua 运行时错误中的细微差别。编译器会在大约 99.9%的情况下告诉你某个东西是 `nil`（Lua 的 `null`）。这取决于你找出原因。以下是一些主要的原因：

+   你在引用一个对象的属性时没有在前面加上 `self.` 或 `self:`。

+   你正在使用点符号调用实例方法，而没有将实例作为第一个参数传递；例如 `myObject.myMethod()` 而不是 `myObject.myMethod(myObject)`。请使用 `myObject:myMethod()`。

+   你正在引用一个在其作用域之外的地方的变量。例如，一个在 `if` 语句内部声明的局部变量正在条件外部被引用。

+   你在类或模块/表的声明结束时忘记了返回类对象。

+   你尝试访问数组的零索引。

+   你忘记添加一些 dos 和 thens 或 ends。

+   最后，也许你只是碰到了那种日子。一个 `nil` 类似的日子。

Cocos IDE 会用粗体显示错误；它与全局变量使用的相同粗体，有时会让人困惑。但无论如何，它还是有帮助的。只需养成检查代码中粗体文本的习惯即可！

### 小贴士

你可能需要增加 IDE 中的堆内存。完成此操作的最快方法是找到 Cocos IDE 应用程序文件夹中的名为 `eclipse.ini` 的文件。在 Mac 上，这意味着在 Cocos IDE 应用程序包中：右键单击应用程序图标，选择**显示包内容**，然后导航到**Contents/MacOS/eclipse.ini**。

然后找到你读取 `-Xmx256m` 或 `-Xmx512m` 的行，并将其更改为 `-Xmx1024m`。

这可能有助于较慢的计算机。我的笔记本电脑在运行 IDE 时经常崩溃。

# 《游戏 - 石器时代》

这是一个三消游戏。你知道，那种让一些公司赚得盆满钵满，让成千上万家公司克隆这些游戏以赚取一点钱的游戏。是的，就是那个游戏！

你必须匹配三个或更多的宝石。如果你匹配的宝石超过三个，一个随机宝石会爆炸并变成钻石，你可以收集这些钻石以获得更多分数。

游戏有一个计时器，当时间耗尽时，游戏结束。

我基本上使用了这本书中之前游戏相同的结构。但我将其分解成独立的模块，这样你更容易将代码作为参考使用。

我们有一个 `MenuScene` 和一个 `GameScene` 项目。我将几乎所有的 Cocos2d-x 动作放在一个名为 `GridAnimations` 的模块中，大部分交互性放在另一个名为 `GridController` 的模块中。所有对象池都保存在一个名为 `ObjectPools` 的类中。

这是一个网格游戏，非常适合用来展示在 Lua 中使用表格数组，以及它相对于 C++ 的主要优势：在 Lua 中创建和内存管理动态列表（数组）要容易得多。这种灵活性，与 Cocos2d-x 的强大功能相结合，使得原型设计和开发非常快速。实际的游戏将看起来像这样：

![《游戏 - 石器时代》](img/00039.jpeg)

但在你导入起始项目之前，让我先向你展示如何在 Cocos IDE 中创建新项目。

# 行动时间 - 创建或导入项目

没有什么比这更简单了；由于 IDE 基于 Eclipse，你知道它的许多主要功能：

1.  首先，让我们设置 IDE 以使用 Lua 绑定。转到**首选项** | **Cocos** | **Lua**，然后在**Lua 框架**下拉菜单中找到您下载的 Cocos2d-x 框架文件夹：![行动时间 - 创建或导入项目](img/00040.jpeg)

1.  如果该选项已经可用，请选择**文件** | **新建** | **Cocos Lua 项目**，或者选择**文件** | **新建** | **其他** | **Cocos Lua** | **Cocos Lua 项目**。

1.  在**新建 Cocos 项目**向导中，给你的项目命名并点击**下一步**。

1.  在下一个对话框中，您可以选择您项目的方向和设计大小。就这样。点击**完成**。

1.  为了导入项目，点击**文件** | **导入**然后**Cocos** | **导入 Cocos 项目**，并导航到本章节的项目起始文件夹。游戏名为`StoneAge`。（如果您还没有下载，请从本书的网站下载本章节的源文件。这里有一个可以运行和测试的起始项目和最终项目。）

## *发生了什么？*

您学会了如何在 Cocos IDE 中创建和导入项目。由于 IDE 是基于 Eclipse 的程序，这些步骤现在应该对您来说很熟悉。

您可能还希望更改模拟器的设置。为此，只需在您的项目上右键单击并选择**运行为...**或**调试为...**，然后选择**运行**或**调试配置**。

对于**Mac OSX**运行时（如果您在 Mac 上，当然），最好保持默认设置，因为这是最快的选择。但如果您想更改模拟器，这里就是您操作的地方：

![发生了什么？](img/00041.jpeg)

### 注意

在我的机器上，框架的 3.4 版本抛出了编译错误。我不得不添加两个修复才能运行《石器时代》。在`cocos-cocos2d-Cocos2dConstants.lua`中，在最后一个表声明之前，我添加了这一行：

```cpp
cc.AsyncTaskPool = {}
```

同样，在`cocos-ui-GuiConstants.lua`中，我在添加新表到`LayoutComponent`之前添加了`ccui.LayoutComponent = {}`，也接近文件末尾。

如果遇到问题，切换到 3.3 版本，这个版本对 Lua 开发来说更加稳定。

# 是时候动手设置我们的屏幕分辨率了

旧的`AppDelegate`类逻辑现在存在于名为`main.lua`的文件中：

1.  在 IDE 中，打开`src`文件夹内的`main.lua`文件。

1.  在设置动画间隔的行之后，输入以下内容：

    ```cpp
    cc.Director:getInstance():getOpenGLView(): setDesignResolutionSize(640, 960,  cc.ResolutionPolicy.SHOW_ALL)
       local screenSize =  cc.Director:getInstance():getVisibleSize()
       local designSize = cc.size(640, 960)
       if (screenSize.width > 320) then
         cc.Director:getInstance():setContentScaleFactor(640/   designSize.width)       
         cc.FileUtils:getInstance():addSearchPath("res/hd/") 
       else
         cc.Director:getInstance():setContentScaleFactor(320/designSize.width)
         cc.FileUtils:getInstance():addSearchPath("res/sd/")         
       end
    ```

1.  我为 iPhone 视网膜屏设计了这款游戏，并且我们为视网膜和非视网膜手机设置了适当的缩放和资源文件夹。接下来，让我们预加载声音文件：

    ```cpp
           local bgMusicPath =  cc.FileUtils:getInstance():fullPathForFilename("background.mp3") 
           cc.SimpleAudioEngine:getInstance():preloadMusic(bgMusicPath)    
           local effectPath =  cc.FileUtils:getInstance():fullPathForFilename("match.wav")
           cc.SimpleAudioEngine:getInstance():preloadEffect(effectPath)
       effectPath =  cc.FileUtils:getInstance():fullPathForFilename("diamond.wav")
           cc.SimpleAudioEngine:getInstance():preloadEffect(effectPath)
           effectPath =  cc.FileUtils:getInstance():fullPathForFilename("diamond2.wav")
           cc.SimpleAudioEngine:getInstance():preloadEffect(effectPath)
           effectPath =  cc.FileUtils:getInstance():fullPathForFilename("wrong.wav")
       cc.SimpleAudioEngine:getInstance():preloadEffect(effectPath)
    ```

1.  最后，让我们通过创建和运行我们的第一个场景来启动项目：

    ```cpp
    --create scene 
    local scene = require("MenuScene")
    local menuScene = scene.create()
    if cc.Director:getInstance():getRunningScene() then
            cc.Director:getInstance():replaceScene(menuScene)
        else
            cc.Director:getInstance():runWithScene(menuScene)
        end
    ```

## *发生了什么？*

就像我们在几乎每一款游戏中做的那样，我们设置了应用程序的分辨率策略和缩放因子，并预加载了我们将使用的声音。

这次游戏只针对手机设计，并且是以 iPhone 4 屏幕为设计目标，可以调整到旧手机。

但现在不要运行游戏。让我们创建我们的菜单场景。它包含了一些基本元素，这将是一个完美的 Lua Cocos2d-x API 入门介绍。

# 是时候动手创建菜单场景了

让我们创建一个新文件，并将菜单场景添加到我们的游戏中：

1.  右键单击`src`文件夹并选择**新建** | **Lua 文件**；将新文件命名为`MenuScene.lua`。

1.  让我们创建一个扩展场景的类。我们首先加载我们自己的所有游戏常量模块（这个文件在起始项目中已经存在）：

    ```cpp
    local constants = require ("constants")
    ```

1.  然后我们构建我们的类：

    ```cpp
    local MenuScene = class("MenuScene", function()
        return cc.Scene:create()
    end)

    function MenuScene.create()
        local scene = MenuScene.new()
        return scene
    end

    function MenuScene:ctor()
        self.visibleSize =  cc.Director:getInstance():getVisibleSize()
        self.middle = {x = self.visibleSize.width * 0.5,  y = self.visibleSize.height * 0.5}
        self.origin = cc.Director:getInstance():getVisibleOrigin()
        self:init()
    end
    return MenuScene
    ```

    我们将添加方法，包括在类构造函数中调用的`init`方法（总是称为`ctor`），但我想要强调在声明末尾返回类的的重要性。

1.  因此，在构造函数下方，让我们继续构建我们的场景：

    ```cpp
    function MenuScene:init ()
        local bg = cc.Sprite:create("introbg.jpg")
        bg:setPosition(self.middle.x, self.middle.y)
        self:addChild(bg)
        --create pterodactyl animation
       local pterodactyl = cc.Sprite:create("ptero_frame1.png")
       pterodactyl:setPosition(cc.p(self.visibleSize.width + 100,  self.visibleSize.height * 0.8))
       self:addChild(pterodactyl)
       local animation = cc.Animation:create()
       local number, name
       for i = 1, 3 do
         number = i
         name = "ptero_frame"..number..".png"
         animation:addSpriteFrameWithFile(name)
       end
       animation:setDelayPerUnit(0.5 / 3.0)
       animation:setRestoreOriginalFrame(true)
       animation:setLoops(-1)
       local animate = cc.Animate:create(animation)
       pterodactyl:runAction( animate )
       local moveOut = cc.MoveTo:create(0, cc.p(self.visibleSize.width + 100, self.visibleSize.height *  0.8))
       local moveIn = cc.MoveTo:create(4.0, cc.p(-100,  self.visibleSize.height * 0.8))
       local delay = cc.DelayTime:create(2.5)
    pterodactyl:runAction(cc.RepeatForever:create (cc.Sequence:create(moveOut, moveIn, delay) ) )
        local character = cc.Sprite:create("introCharacter.png")
        character:setPosition(self.middle.x, self.middle.y + 110)
        self:addChild(character)
        local frame = cc.Sprite:create("frame.png")
        frame:setPosition(self.middle.x, self.middle.y)
        self:addChild(frame)    
    end
    ```

    通过这种方式，我们添加了一个背景和两个其他精灵，以及一个翼龙在背景中飞行的动画。再一次，调用与 C++中的调用非常相似。

1.  现在，让我们在`init`方法中添加一个带有播放按钮的菜单（所有这些仍然在`init`方法中）：

    ```cpp
    --create play button
    local function playGame()
       local bgMusicPath =    cc.FileUtils:getInstance():fullPathForFilename("background.mp3") 
       cc.SimpleAudioEngine:getInstance():playMusic(bgMusicPath, true)
       local scene = require("GameScene")
       local gameScene = scene.create()
       cc.Director:getInstance():replaceScene(gameScene)
    end

    local btnPlay = cc.MenuItemImage:create("playBtn.png",  "playBtnOver.png")
    btnPlay:setPosition(0,0)
    btnPlay:registerScriptTapHandler(playGame)
    local menu  = cc.Menu:create(btnPlay)
    menu:setPosition(self.middle.x, 80)
    self:addChild(menu)
    ```

在引用回调的同一方法中键入按钮的回调，类似于在 C++中编写一个块或甚至是 lambda 函数。

## *刚才发生了什么？*

你使用 Cocos2d-x 和 Lua 创建了一个场景，其中包括一个菜单、几个精灵和一个动画。很容易看出 Lua 绑定与原始 C++绑定的调用是多么相似。而且，在 IDE 中的代码补全功能使得查找正确的方法变得轻而易举。

现在让我们处理`GameScene`类。

### 注意

Lua 最吸引人的特性之一是所谓的**实时编码**，在 Cocos IDE 中默认开启。为了了解我所说的实时编码是什么意思，这样做：当游戏在模拟器中运行时，更改你的代码中角色精灵的位置并保存它。你应该会在模拟器中看到变化生效。这是一种构建 UI 和游戏场景的绝佳方式。

# 行动时间——创建我们的游戏场景

`GameScene`类已经添加到起始项目中，并且一些代码已经就位。我们首先将专注于构建游戏界面和监听触摸：

1.  让我们专注于`addTouchEvents`方法：

    ```cpp
    function GameScene:addTouchEvents()
        local bg = cc.Sprite:create("background.jpg")
        bg:setPosition(self.middle.x, self.middle.y)
        self:addChild(bg)

        local function onTouchBegan(touch, event)
            self.gridController:onTouchDown(touch:getLocation())
            return true
        end

        local function onTouchMoved(touch, event)
            self.gridController:onTouchMove(touch:getLocation())
        end

        local function onTouchEnded(touch, event)
            self.gridController:onTouchUp(touch:getLocation())
        end

        local listener = cc.EventListenerTouchOneByOne:create()
           listener:registerScriptHandler (onTouchBegan,cc.Handler.EVENT_TOUCH_BEGAN )
       listener:registerScriptHandler (onTouchMoved,cc.Handler.EVENT_TOUCH_MOVED )
        listener:registerScriptHandler (onTouchEnded,cc.Handler.EVENT_TOUCH_ENDED )
        local eventDispatcher = bg:getEventDispatcher()
           eventDispatcher:addEventListenerWithSceneGraphPriority (listener, bg)
    end
    ```

1.  再次，我们使用节点的事件分发器的实例注册事件。实际的触摸由我们的`GridController`对象处理。我们稍后会介绍这些；首先，让我们构建 UI。现在是时候在`init`方法上工作了：

    ```cpp
    function GameScene:init ()
        self.gridController = GridController:create()
        self.gridAnimations = GridAnimations:create()
        self.objectPools = ObjectPools.create()

        self.gridAnimations:setGameLayer(self)
        self.gridController:setGameLayer(self)
        self.objectPools:createPools(self)
    ```

    创建我们的特殊对象，一个用于处理用户交互，另一个用于动画，以及我们熟悉的对象池。

1.  接下来，我们添加几个节点和我们的得分标签：

    ```cpp
    self:addChild( self.gemsContainer )
    self.gemsContainer:setPosition( 25, 80)
    --build interface
    local frame = cc.Sprite:create("frame.png")
    frame:setPosition(self.middle.x, self.middle.y)
    self:addChild(frame)
    local diamondScoreBg = cc.Sprite:create("diamondScore.png")
    diamondScoreBg:setPosition(100, constants.SCREEN_HEIGHT - 30)
    self:addChild(diamondScoreBg)
    local scoreBg = cc.Sprite:create("gemsScore.png")
    scoreBg:setPosition(280, constants.SCREEN_HEIGHT - 30)
    self:addChild(scoreBg)
    local ttfConfig = {}
    ttfConfig.fontFilePath="fonts/myriad-pro.ttf"
    ttfConfig.fontSize=20
    self.diamondScoreLabel = cc.Label:createWithTTF(ttfConfig,  "0", cc.TEXT_ALIGNMENT_RIGHT , 150)    
    self.diamondScoreLabel:setPosition  (140, constants.SCREEN_HEIGHT - 30)
    self:addChild(self.diamondScoreLabel)
    self.scoreLabel = cc.Label:createWithTTF(ttfConfig,  "0", cc.TEXT_ALIGNMENT_RIGHT , 150)    
    self.scoreLabel:setPosition (330, constants.SCREEN_HEIGHT - 30)
    self:addChild(self.scoreLabel) 
    end
    ```

与`Label:createWithTTF`的 C++实现相比，主要的不同之处在于 Lua 中有一个字体配置表。

## *刚才发生了什么？*

这次，我们学习了如何注册触摸事件以及如何创建真类型字体标签。接下来，我们将介绍如何创建一个典型的三消游戏的网格。

# 行动时间——构建宝石

三消游戏基本上有两种类型，一种是在游戏中自动进行匹配选择，另一种是由玩家进行选择。*糖果传奇*是前者的一个好例子，而*钻石冲刺*则是后者。在构建第一种类型的游戏时，你必须添加额外的逻辑来确保你开始游戏时网格中不包含任何匹配项。我们现在就要这样做：

1.  我们从`buildGrid`方法开始：

    ```cpp
    function GameScene:buildGrid ()
       math.randomseed(os.clock())
       self.enabled = false
        local g
        for c = 1, constants.GRID_SIZE_X do
            self.grid[c] = {}
            self.gridGemsColumnMap[c] = {}
            for r = 1, constants.GRID_SIZE_Y do
                if (c < 3) then
                    self.grid[c][r] =  constants.TYPES[ self:getVerticalUnique(c,r) ]
                else
                    self.grid[c][r] =  constants.TYPES[ self:getVerticalHorizontalUnique(c,r) ]
                end
               g = Gem:create()
                g:setType(  self.grid[c][r] )
               g:setPosition ( c * (constants.TILE_SIZE +  constants.GRID_SPACE), 
                    r * (constants.TILE_SIZE +  constants.GRID_SPACE))
               self.gemsContainer:addChild(g)           
                self.gridGemsColumnMap[c][r] = g
                table.insert(self.allGems, g)
           end
        end
        self.gridAnimations:animateIntro()    
    end
    ```

    通过更改 `randomseed` 值，确保每次运行游戏时都生成不同的随机宝石序列。

    当网格正在更改或动画时，`enabled` 属性将阻止用户交互。

    网格是由宝石列组成的二维数组。魔法发生在 `getVerticalUnique` 和 `getVerticalHorizontalUnique` 方法中。

1.  为了确保没有任何宝石会在前两列形成三个宝石的匹配，我们垂直检查它们：

    ```cpp
    function GameScene:getVerticalUnique (col, row)
       local type = math.floor (math.random () *  #constants.TYPES + 1 )
       if (self.grid[col][row-1] == constants.TYPES[type] and  self.grid[col][row-2] ~= nil and self.grid[col][row-2] ==  constants.TYPES[type]) then
            type = type + 1; 
            if (type == #constants.TYPES + 1) then type = 1 end
        end
        return type
    end
    ```

    所有这些代码所做的只是检查一列，看看是否有任何宝石正在形成相同类型的三个相连宝石的字符串。

1.  然后，我们垂直和水平检查，从第三列开始：

    ```cpp
    function GameScene:getVerticalHorizontalUnique (col, row)
       local type = self:getVerticalUnique (col, row)
       if (self.grid[col - 1][row] == constants.TYPES[type] and  self.grid[col - 2][row] ~= nil and self.grid[col - 2][row] ==  constants.TYPES[type]) then
            local unique = false
            while unique == false do
              type = self:getVerticalUnique (col, row)
              if (self.grid[col-1][row] == constants.TYPES[type] and
              self.grid[col - 2 ][row] ~= nil and  self.grid[col -  2 ][row] == constants.TYPES[type]) then
                --do nothing
              else
                 unique = true
              end           
            end
        end
        return type
    end
    ```

此算法正在执行我们之前对列所做的相同操作，但它还在单独的行上进行检查。

## *发生了什么？*

我们创建了一个没有三个宝石匹配的宝石网格。再次强调，如果我们构建了用户必须选择匹配宝石簇以从网格中移除的匹配三游戏（如 *Diamond Dash*），我们根本不需要担心这个逻辑。

接下来，让我们通过宝石交换、识别匹配和网格折叠来操作网格。

# 是时候采取行动了——使用 GridController 改变网格。

`GridController` 对象启动所有网格更改，因为这是我们处理触摸的地方。在游戏中，用户可以拖动宝石与另一个宝石交换位置，或者首先选择他们想要移动的宝石，然后在两指触摸过程中选择他们想要交换位置的宝石。让我们添加处理这种触摸的代码：

1.  在 `GridController` 中，让我们添加 `onTouchDown` 的逻辑：

    ```cpp
    function GridController:onTouchDown (touch)
        if (self.gameLayer.running == false) then
            local scene = require("GameScene")
            local gameScene = scene.create()
            cc.Director:getInstance():replaceScene(gameScene)
            local bgMusicPath =  cc.FileUtils:getInstance():fullPathForFilename("background.mp3") 
            cc.SimpleAudioEngine:getInstance():playMusic(bgMusicPath, true)
            return 
        end
    ```

    如果我们正在显示游戏结束屏幕，则重新启动场景。

1.  接下来，我们找到用户试图选择的宝石：

    ```cpp
      self.touchDown = true
        if (self.enabled == false) then return end
        local touchedGem = self:findGemAtPosition (touch)
        if (touchedGem.gem ~= nil ) then 
            if (self.gameLayer.selectedGem == nil) then
                self:selectStartGem(touchedGem)
            else
                if (self:isValidTarget(touchedGem.x,  touchedGem.y, touch) == true) then 
                    self:selectTargetGem(touchedGem)
                else
                    if (self.gameLayer.selectedGem ~= nil)  then self.gameLayer.selectedGem:deselect() end
                    self.gameLayer.selectedGem = nil
                    self:selectStartGem (touchedGem)
                end
            end
        end
    end
    ```

    我们找到离触摸位置最近的宝石。如果用户尚未选择宝石（`selectedGem = nil`），我们将刚刚触摸到的宝石作为第一个选中的宝石。否则，我们确定第二个选中的宝石是否可以用于交换。只有位于第一个选中宝石上方和下方的宝石，或者位于其左右两侧的宝石可以交换。如果这是有效的，我们就使用第二个宝石作为目标宝石。

1.  在继续到 `onTouchMove` 和 `onTouchUp` 之前，让我们看看我们是如何确定哪个宝石被选中以及哪个宝石是有效目标宝石的。所以让我们处理 `findGemAtPosition` 值。首先确定触摸落在网格容器中的位置：

    ```cpp
    function GridController:findGemAtPosition (position)
        local mx = position.x
        local my = position.y
        local gridWidth = constants.GRID_SIZE_X *  (constants.TILE_SIZE + constants.GRID_SPACE)
        local gridHeight = constants.GRID_SIZE_Y *  (constants.TILE_SIZE + constants.GRID_SPACE)
        mx = mx - self.gameLayer.gemsContainer:getPositionX()
        my = my - self.gameLayer.gemsContainer:getPositionY()
        if (mx < 0) then mx = 0 end
        if (my < 0) then my = 0 end
        if (mx > gridWidth) then mx = gridWidth end
        if (my > gridHeight) then my = gridHeight end
    ```

1.  这里是魔法发生的地方。我们使用网格内触摸的 `x` 和 `y` 位置来确定数组中宝石的索引：

    ```cpp
    local x = math.ceil ((mx - constants.TILE_SIZE * 0.5) /  (constants.TILE_SIZE + constants.GRID_SPACE))
        local y = math.ceil ((my - constants.TILE_SIZE * 0.5) /  (constants.TILE_SIZE + constants.GRID_SPACE))
        if (x < 1) then x = 1 end
        if (y < 1) then y = 1 end
        if (x > constants.GRID_SIZE_X) then x =  constants.GRID_SIZE_X end
        if (y > constants.GRID_SIZE_Y) then y =  constants.GRID_SIZE_Y end
        return {x = x, y = y, gem =  self.gameLayer.gridGemsColumnMap[x][y]}
    end
    ```

    我们最后检查触摸是否超出数组界限。

1.  现在让我们看看确定目标宝石是否为有效目标的逻辑：

    ```cpp
    function GridController:isValidTarget (px, py, touch)
        local offbounds = false
        if (px > self.gameLayer.selectedIndex.x + 1) then 
    offbounds = true end
        if (px < self.gameLayer.selectedIndex.x - 1) then 
    offbounds = true end
        if (py > self.gameLayer.selectedIndex.y + 1) then 
    offbounds = true end
        if (py < self.gameLayer.selectedIndex.y - 1) then 
    offbounds = true end
    ```

    我们首先检查目标宝石是否位于所选宝石的顶部、底部、左侧或右侧：

    ```cpp
    local cell = math.sin (math.atan2  (math.pow( self.gameLayer.selectedIndex.x - px, 2),  math.pow( self.gameLayer.selectedIndex.y- py, 2) ) )
        if (cell ~= 0 and cell ~= 1) then
            offbounds = true
        end
        if (offbounds == true) then
            return false
        end
    ```

    我们接下来使用一点三角学的魔法来确定所选的目标宝石是否与所选宝石对角线：

    ```cpp
       local touchedGem = self.gameLayer.gridGemsColumnMap[px][py]
        if (touchedGem.gem == self.gameLayer.selectedGem or  (px == self.gameLayer.selectedIndex.x and  py == self.gameLayer.selectedIndex.y)) then
            self.gameLayer.targetGem = nil
            return false
        end
        return true
    end
    ```

    我们最后检查目标宝石是否与之前选中的宝石不同。

1.  现在，让我们继续处理`onTouchUp`事件：

    ```cpp
    function GridController:onTouchUp (touch)
        if (self.gameLayer.running == false) then return end
        self.touchDown = false
        if (self.enabled == false) then return end
        if (self.gameLayer.selectedGem ~= nil) then  self.gameLayer:dropSelectedGem() end
    end
    ```

    很简单！我们只是改变了选择宝石的`z`层级，因为我们想确保在交换发生时宝石显示在其他宝石之上。所以当我们释放宝石时，我们将其推回到原始的`z`层级（这就是`dropSelectedGem`方法所做的事情，我们很快就会看到它是如何做到这一点的）。

1.  `onTouchMove`事件处理选择宝石拖动直到它与另一个宝石交换位置：

    ```cpp
    function GridController:onTouchMove (touch)
        if (self.gameLayer.running == false) then return end
        if (self.enabled == false) then return end
        --track to see if we have a valid target
        if (self.gameLayer.selectedGem ~= nil and  self.touchDown == true) then
            self.gameLayer.selectedGem:setPosition(
            touch.x - self.gameLayer.gemsContainer:getPositionX(), 
            touch.y - self.gameLayer.gemsContainer:getPositionY())
            local touchedGem = self:findGemAtPosition (touch)
            if (touchedGem.gem ~= nil and self:isValidTarget(touchedGem.x, touchedGem.y, touch) == true ) then
                self:selectTargetGem(touchedGem)
            end
        end
    end
    ```

    我们运行了与`onTouchDown`相同的逻辑。我们将`selectedGem`对象移动，直到找到一个合适的目标宝石，然后选择第二个作为目标。这就是交换发生的时候。现在让我们来做这件事。

1.  首先，设置我们选择宝石的逻辑：

    ```cpp
    function GridController:selectStartGem (touchedGem)
           if (self.gameLayer.selectedGem == nil) then
            self.gameLayer.selectedGem = touchedGem.gem
            self.gameLayer.targetGem = nil
            self.gameLayer.targetIndex = nil
            touchedGem.gem:setLocalZOrder(constants.Z_SWAP_2)
            self.gameLayer.selectedIndex = {x = touchedGem.x,  y = touchedGem.y}
            self.gameLayer.selectedGemPosition =  {x = touchedGem.gem:getPositionX(),
                                                  y =  touchedGem.gem:getPositionY()}
            self.gameLayer.gridAnimations:animateSelected  (touchedGem.gem)                                              
        end
    end
    ```

    我们开始交换过程；我们有一个选择的宝石但没有目标宝石。我们通过`setLocalZOrder`改变选择宝石的层级。同时，我们也让选择宝石旋转 360 度。

1.  然后，我们准备好选择目标宝石：

    ```cpp
    function GridController:selectTargetGem (touchedGem)
        if (self.gameLayer.targetGem ~= nil) then return end
        self.enabled = false
        self.gameLayer.targetIndex = {x = touchedGem.x,  y = touchedGem.y}
        self.gameLayer.targetGem = touchedGem.gem
        self.gameLayer.targetGem:setLocalZOrder(constants.Z_SWAP_1)
        self.gameLayer:swapGemsToNewPosition()
    end
    ```

现在是我们最终调用`GameScene`类并要求它交换宝石的时候了。

## *刚才发生了什么？*

我们刚刚添加了处理所有用户交互的逻辑。现在，我们剩下要做的就是处理交换，检查匹配项并折叠网格。让我们来做吧！

# 行动时间 - 交换宝石并寻找匹配项

交换逻辑位于`GameScene`中的`swapGemsToNewPosition`方法：

1.  `swapGemsToNewPosition`方法调用一次`GridAnimations`来动画化选择宝石和目标宝石之间的交换。一旦这个动画完成，我们触发一个`onNewSwapComplete`方法。大部分逻辑都发生在这里：

    ```cpp
    function GameScene:swapGemsToNewPosition ()
        local function onMatchedAnimatedOut (sender)
            self:collapseGrid()
        end

        local function onReturnSwapComplete (sender)
            self.gridController.enabled = true
        end

        local function onNewSwapComplete (sender)
           self.gridGemsColumnMap[self.targetIndex.x][self.targetIndex.y]  = self.selectedGem
            self.gridGemsColumnMap[self.selectedIndex.x][self.selectedIndex.y] =  self.targetGem
            self.grid[self.targetIndex.x][self.targetIndex.y] =  self.selectedGem.type
            self.grid[self.selectedIndex.x][self.selectedIndex.y] =  self.targetGem.type
    The call back switches the gems around inside the grid array.
            self.combos = 0
            self.addingCombos = true
    Combos are used to track if we have more than 3 gems matched  after the player's move.        
            --check for new matches
            if (self.gridController:checkGridMatches() == true) then
    ```

1.  如果我们找到了匹配项，我们对匹配的宝石运行动画，如果没有匹配，我们播放一个交换回动画并播放一个音效来表示玩家走错了一步：

    ```cpp
                    --animate matched gems
                if (#self.gridController.matchArray > 3) then  self.combos = self.combos + (#self.gridController.matchArray -  3) end 
                self.gridAnimations:animateMatches  (self.gridController.matchArray, onMatchedAnimatedOut)
                self:showMatchParticle  (self.gridController.matchArray)
                self:setGemsScore(#self.gridController.matchArray *  constants.POINTS)
                self:playFX("match2.wav")
            else
                --no matches, swap gems back
                self.gridAnimations:swapGems (self.targetGem,  self.selectedGem, onReturnSwapComplete)
                self.gridGemsColumnMap[self.targetIndex.x][self.targetIndex.y]  = self.targetGem
                self.gridGemsColumnMap[self.selectedIndex.x][self.selectedIndex.y]  = self.selectedGem
                self.grid[self.targetIndex.x][self.targetIndex.y] =  self.targetGem.type
                self.grid[self.selectedIndex.x][self.selectedIndex.y] =  self.selectedGem.type
                self:playFX("wrong.wav")
            end
    ```

    在每个新动画的末尾，无论是匹配动画还是交换回动画，我们再次运行方法顶部列出的回调。这些回调最重要的作用是在`onMatchedAnimatedOut`回调中匹配宝石动画完成后调用`collapseGrid`：

    ```cpp
            self.selectedGem = nil
            self.targetGem = nil
       end
    ```

    我们通过清除选择的宝石并从一张干净的局面开始来结束回调。

1.  在这里，函数的末尾，我们调用带有`onNewSwapComplete`作为回调的宝石交换动画：

    ```cpp
       self.gridAnimations:swapGems (self.selectedGem, self.targetGem, onNewSwapComplete)
    end
    ```

1.  让我们回到`GridController`并添加`checkGridMatches`方法。这分为三个部分：

    ```cpp
    function GridController:checkGridMatches ()
        self.matchArray = {}
        for c = 1, constants.GRID_SIZE_X do
            for r = 1, constants.GRID_SIZE_Y do
                self:checkTypeMatch(c,r)
            end
        end
        if (#self.matchArray >= 2) then
            self.gameLayer:addToScore()
            return true
        end
        print("no matches")
        return false
    end
    ```

    这个方法通过在每个单元格上运行`checkTypeMatch`来开始检查。

1.  `checkTypeMatch`方法在当前索引周围搜索，寻找索引的上方、下方、左侧和右侧的匹配项：

    ```cpp
    function GridController:checkTypeMatch (c, r)
        local type = self.gameLayer.grid[c][r]
        local stepC = c
        local stepR = r
        local temp_matches = {}
        --check top
        while stepR -1 >= 1 and self.gameLayer.grid[c][stepR-1] ==  type do
            stepR = stepR - 1
            table.insert (temp_matches, {x = c, y = stepR})
        end 
        if (#temp_matches >= 2) then self:addMatches (temp_matches) end
        temp_matches = {}
        --check bottom
        stepR = r
        while stepR + 1 <= constants.GRID_SIZE_Y 
       and self.gameLayer.grid[c][stepR + 1] == type do
            stepR = stepR + 1
            table.insert (temp_matches, {x = c, y= stepR})
        end
        if (#temp_matches >= 2) then self:addMatches (temp_matches) end
        temp_matches = {}
        --check left
        while stepC - 1 >= 1 and self.gameLayer.grid[stepC - 1][r]  == type do
            stepC = stepC - 1
            table.insert (temp_matches, {x = stepC, y= r})
        end
        if (#temp_matches >= 2) then self:addMatches (temp_matches) end
        temp_matches = {}
        --check right
        stepC = c;
        while stepC + 1 <= constants.GRID_SIZE_X and  self.gameLayer.grid[stepC + 1][r] == type do
            stepC = stepC + 1
            table.insert (temp_matches, {x = stepC, y = r})
        end
        if (#temp_matches >= 2) then self:addMatches (temp_matches) end
    end
    ```

    如果找到了任何匹配项，它们将被添加到`matches`数组中。

1.  但首先我们需要确保没有重复项列在那里，所以当我们向`matches`数组添加宝石时，我们检查它是否已经被添加：

    ```cpp
    function GridController:addMatches (matches)
        for key, value in pairs(matches) do
            if (self:find(value, self.matchArray) == false) then
                table.insert(self.matchArray, value)
            end
        end
    end
    ```

1.  以及查找重复项的简单方法：

    ```cpp
    function GridController:find (np, array)
        for key, value in pairs(array) do
            if (value.x == np.x and value.y == np.y) then return true end
        end
        return false
    end
    ```

## *刚才发生了什么？*

寻找匹配项是任何三合一游戏所需逻辑的一半以上。你所需要做的就是尽可能有效地遍历网格，寻找重复的模式。

其余的逻辑涉及网格坍塌。我们将在下一步进行，然后我们就可以发布游戏了。

# 是时候行动了——坍塌网格并重复

因此，游戏的流程是移动部件，寻找匹配项，移除它们，坍塌网格，添加新的宝石，再次寻找匹配项，如果需要，整个流程循环进行：

1.  这是游戏中最长的方法，而且，同样，大部分逻辑都发生在回调中。首先，我们通过将它们的类型数据设置为 `-1` 来标记要移除的宝石。`matchArray` 中的所有宝石都将被移除：

    ```cpp
    function GameScene:collapseGrid ()
        for i = 1, #self.gridController.matchArray do
            self.grid[self.gridController.matchArray[i].x]
            [self.gridController.matchArray[i].y] = -1
        end

        local column = nil
        local newColumn = nil
        local i
    ```

1.  接下来，我们遍历网格的列，重新排列列数组中类型不等于 `-1` 的宝石。本质上，我们在这里更新数据，以便移除的宝石上面的宝石“落下”。实际的变化将在 `animateCollapse` 方法中发生：

    ```cpp
        for c = 1, constants.GRID_SIZE_X do
            column = self.grid[c]
            newColumn = {}
            i = 1
            while #newColumn < #column do
                if (#column > i) then
                    if (column[i] ~= -1) then
                        --move gem
                        table.insert(newColumn, column[i])
                    end
                else
                    --create new gem
                    table.insert(newColumn, 1, column[i])
                end
                i = i+1            
            end
            self.grid[c] = newColumn
        end
        self.gridAnimations:animateCollapse  (onGridCollapseComplete)
    end
    ```

1.  但现在，让我们编写动画回调 `onGridCollapseComplete` 的代码。所以我们在 `collapseGrid` 中已经输入的代码上方添加 `local` 函数：

    ```cpp
    local function onGridCollapseComplete (sender)
       local function onMatchedAnimatedOut (sender)
          self:collapseGrid()
       end
       for i = 1, #self.allGems do
          local gem = self.allGems[i]
          local xIndex = math.ceil ((gem:getPositionX() -  constants.TILE_SIZE * 0.5) / (constants.TILE_SIZE +  constants.GRID_SPACE))
          local yIndex = math.ceil ((gem:getPositionY() -  constants.TILE_SIZE * 0.5) / (constants.TILE_SIZE +  constants.GRID_SPACE))
          self.gridGemsColumnMap[xIndex][yIndex] = gem
          self.grid[xIndex][yIndex] = gem.type
       end
    ```

    首先，我们更新精灵数组，按网格的新 `x` 和 `y` 索引排序。

1.  然后，我们再次检查匹配项。记住，这个回调在网格坍塌动画完成后运行，这意味着已经添加了新的宝石，这些宝石可能创建了新的匹配项（我们很快将查看逻辑）：

    ```cpp
    if (self.gridController:checkGridMatches () == true) then
          --animate matched games
          if (self.addingCombos == true) then
             if (#self.gridController.matchArray > 3) then  self.combos = self.combos + (#self.gridController.matchArray -  3) end
          end
          self.gridAnimations:animateMatches  (self.gridController.matchArray, onMatchedAnimatedOut)
          self:showMatchParticle (self.gridController.matchArray)
          self:setGemsScore(#self.gridController.matchArray *  constants.POINTS)
          self:playFX("match.wav")
    ```

1.  然后，如果我们没有找到更多的匹配项，当组合的价值大于 0（意味着在上一个玩家的移动中我们有多于 3 个宝石匹配）时，我们将一些随机的宝石替换为钻石：

    ```cpp
    else 
       --no more matches, check for combos
       if (self.combos > 0) then
       --now turn random gems into diamonds
           local diamonds = {}
           local removeGems = {}
           local i = 0

           math.randomseed(os.clock())
           while i < self.combos do
             i = i + 1
             local randomGem = nil
             local randomX,randomY = 0
             while randomGem == nil do
               randomX = math.random(1, constants.GRID_SIZE_X)
               randomY = math.random(1, constants.GRID_SIZE_Y)
               randomGem = self.gridGemsColumnMap[randomX][randomY]
               if (randomGem.type == constants.TYPE_GEM_WHITE)  then randomGem = nil end
           end
    ```

1.  我们随机选择宝石作为钻石：

    ```cpp
            local diamond = self.objectPools:getDiamond()
          diamond:setPosition(randomGem:getPositionX(),  randomGem:getPositionY())
          local diamondParticle =  self.objectPools:getDiamondParticle()
          diamondParticle:setPosition(randomGem:getPositionX(),  randomGem:getPositionY())
             table.insert(diamonds, diamond)   
             table.insert(removeGems, {x=randomX, y=randomY}) 
            end
            self:setDiamondScore(#diamonds *  constants.DIAMOND_POINTS)
    ```

    动画收集钻石，并在该动画结束时调用 `onMatchedAnimatedOut` 回调，此时由于宝石“爆裂”成钻石，网格将再次坍塌：

    ```cpp
            self.gridAnimations:animateMatches(removeGems,  onMatchedAnimatedOut)                
         self.gridAnimations:collectDiamonds(diamonds)
         self.combos = 0 
         self:playFX("diamond2.wav")  
        else
         self.gridController.enabled = true
        end
         self.addingCombos = false
       end
    end
    ```

1.  这是整个 `collapseGrid` 方法：

    ```cpp
    function GameScene:collapseGrid ()
        local function onGridCollapseComplete (sender)
           local function onMatchedAnimatedOut (sender)
                self:collapseGrid()
            end
           for i = 1, #self.allGems do
                local gem = self.allGems[i]
                local xIndex = math.ceil ((gem:getPositionX() -  constants.TILE_SIZE * 0.5) / (constants.TILE_SIZE +  constants.GRID_SPACE))
                local yIndex = math.ceil ((gem:getPositionY() -  constants.TILE_SIZE * 0.5) / (constants.TILE_SIZE +  constants.GRID_SPACE))
                self.gridGemsColumnMap[xIndex][yIndex] = gem
                self.grid[xIndex][yIndex] = gem.type
            end
            if (self.gridController:checkGridMatches () == true) then
               --animate matched games
               if (self.addingCombos == true) then
                   if (#self.gridController.matchArray > 3) then  self.combos = self.combos + (#self.gridController.matchArray -  3) end
               end
               self.gridAnimations:animateMatches  (self.gridController.matchArray, onMatchedAnimatedOut)
               self:showMatchParticle  (self.gridController.matchArray)
               self:setGemsScore(#self.gridController.matchArray *  constants.POINTS)
               self:playFX("match.wav")
            else 
                --no more matches, check for combos
                if (self.combos > 0) then
                    --now turn random gems into diamonds
                    local diamonds = {}
                    local removeGems = {}
                    local i = 0
                    math.randomseed(os.clock())
                    while i < self.combos do
                       i = i + 1
                       local randomGem = nil
                        local randomX,randomY = 0
                       while randomGem == nil do
                            randomX =  math.random(1, constants.GRID_SIZE_X)
                            randomY =  math.random(1, constants.GRID_SIZE_Y)
                            randomGem =  self.gridGemsColumnMap[randomX][randomY]
                            if (randomGem.type ==  constants.TYPE_GEM_WHITE) then randomGem = nil end
                        end
                        local diamond =  self.objectPools:getDiamond()
                        diamond:setPosition(randomGem:getPositionX(),  randomGem:getPositionY())
                        local diamondParticle =  self.objectPools:getDiamondParticle()
                        diamondParticle:setPosition(randomGem:getPositionX(),  randomGem:getPositionY())
                        table.insert(diamonds, diamond)
                        table.insert(removeGems, {x=randomX,  y=randomY}) 
                    end
                    self:setDiamondScore(#diamonds *  constants.DIAMOND_POINTS)
                    self.gridAnimations:animateMatches(removeGems,  onMatchedAnimatedOut)                
                    self.gridAnimations:collectDiamonds(diamonds)
                    self.combos = 0 
                    self:playFX("diamond2.wav")                 
                else
                    self.gridController.enabled = true
                end
                self.addingCombos = false
            end
        end
        for i = 1, #self.gridController.matchArray do
            self.grid[self.gridController.matchArray[i].x] [self.gridController.matchArray[i].y] = -1
        end

        local column = nil
        local newColumn = nil
        local i
        for c = 1, constants.GRID_SIZE_X do
            column = self.grid[c]
            newColumn = {}
            i = 1
            while #newColumn < #column do
                if (#column > i) then
                    if (column[i] ~= -1) then
                        --move gem
                        table.insert(newColumn, column[i])
                    end
                else
                    --create new gem
                    table.insert(newColumn, 1, column[i])
                end
                i = i+1            
            end
            self.grid[c] = newColumn
        end
        self.gridAnimations:animateCollapse  (onGridCollapseComplete)
    end
    ```

## *刚才发生了什么？*

`collapseGrid` 方法收集所有受匹配或爆炸成钻石的宝石影响的宝石。结果数组被发送到 `GridAnimations` 以执行适当的动画。

我们将在这些基础上工作，完成我们的游戏。

# 是时候行动了——动画匹配和坍塌

现在是最后一点逻辑：最后的动画：

1.  我们将从简单的开始：

    ```cpp
    function GridAnimations:animateSelected (gem)
        gem:select()
        gem:stopAllActions()
        local rotate = cc.EaseBounceOut:create ( cc.RotateBy:create(0.5, 360) )
        gem:runAction(rotate)
    end
    ```

    这会使宝石旋转；我们使用这个动画来表示宝石首次被选中。

1.  接下来是交换动画：

    ```cpp
    function GridAnimations:swapGems  (gemOrigin, gemTarget, onComplete)
        gemOrigin:deselect()
       local origin = self.gameLayer.selectedGemPosition
        local target = cc.p(gemTarget:getPositionX(),  gemTarget:getPositionY()) 
       local moveSelected =  cc.EaseBackOut:create(cc.MoveTo:create(0.8, target) )   
        local moveTarget =  cc.EaseBackOut:create(cc.MoveTo:create(0.8, origin) )
        local callback = cc.CallFunc:create(onComplete)
       gemOrigin:runAction(moveSelected)
        gemTarget:runAction (cc.Sequence:create(moveTarget, callback))
    end
    ```

    这只是交换第一个选择的宝石和目标宝石的位置。

1.  然后，我们添加运行匹配宝石的动画：

    ```cpp
    function GridAnimations:animateMatches (matches, onComplete)
        local function onCompleteMe (sender)
           self.animatedMatchedGems = self.animatedMatchedGems - 1;
            if (self.animatedMatchedGems == 0) then
                if (onComplete ~= nil) then onComplete() end
            end
    end
        self.animatedMatchedGems = #matches
       local gem = nil
        for i, point in ipairs(matches) do
            gem = self.gameLayer.gridGemsColumnMap[point.x] [point.y]
            gem:stopAllActions()
            local scale = cc.EaseBackOut:create  ( cc.ScaleTo:create(0.3, 0))
            local callback = cc.CallFunc:create(onCompleteMe)
            local action = cc.Sequence:create (scale, callback)
            gem.gemContainer:runAction(action)
        end
    end
    ```

    这将使宝石缩放到无，并且只有当所有宝石完成缩放时才触发最终的回调。

1.  接下来是收集钻石的动画：

    ```cpp
    function GridAnimations:collectDiamonds(diamonds)
        local function removeDiamond (sender)
            sender:setVisible(false)
        end
        for i = 1, #diamonds do
            local delay = cc.DelayTime:create(i * 0.05)
            local moveTo = cc.EaseBackIn:create( cc.MoveTo:create ( 0.8, cc.p(50, constants.SCREEN_HEIGHT - 50) ) )
            local action = cc.Sequence:create  (delay, moveTo, cc.CallFunc:create(removeDiamond))
            diamonds[i]:runAction(action)
        end
    end
    ```

    这将钻石移动到钻石得分标签的位置。

1.  现在，最后，添加网格坍塌：

    ```cpp
    function GridAnimations:animateCollapse ( onComplete )
        self.animatedCollapsedGems = 0
        local gem = nil
        local drop  = 1
       for c = 1, constants.GRID_SIZE_X do 
            drop = 1
            for r = 1, constants.GRID_SIZE_Y do
                gem = self.gameLayer.gridGemsColumnMap[c][r]
                --if this gem has been resized, move it to the top 
                if (gem.gemContainer:getScaleX() ~= 1) then
                    gem:setPositionY((constants.GRID_SIZE_Y +  (drop)) * (constants.TILE_SIZE + constants.GRID_SPACE))
                    self.animatedCollapsedGems =  self.animatedCollapsedGems + 1
                    gem:setType ( self.gameLayer:getNewGem() )
                    gem:setVisible(true)
                    local newY = (constants.GRID_SIZE_Y -  (drop - 1)) * (constants.TILE_SIZE + constants.GRID_SPACE)
                    self:dropGemTo (gem, newY,  0.2, onComplete)
                    drop = drop + 1
                else
                   if (drop > 1) then
                        self.animatedCollapsedGems =  self.animatedCollapsedGems + 1
                        local newY = gem:getPositionY() -  (drop - 1) * (constants.TILE_SIZE + constants.GRID_SPACE)
                        self:dropGemTo (gem, newY, 0, onComplete)
                    end
               end
            end 
        end
    end 
    ```

    我们遍历所有宝石，并识别出那些被缩小的宝石，这意味着它们已经被*移除*。我们将这些宝石移动到列的上方，这样它们就会作为新的宝石落下，并为它们选择一个新的类型：

    ```cpp
    gem:setType ( self.gameLayer:getNewGem() )
    ```

    那些没有被移除的宝石将落到它们的新位置。我们这样做的方式很简单。我们计算有多少宝石被移除，直到我们到达一个没有被移除的宝石。这个计数存储在局部变量 drop 中，每次列重置时都会将其重置为`0`。

    这样，我们就知道了有多少宝石被其他宝石下面的宝石移除。我们使用这个信息来找到新的`y`位置。

1.  `dropGemTo`新位置看起来是这样的：

    ```cpp
    function GridAnimations:dropGemTo (gem, y, delay, onComplete)
          gem:stopAllActions()
        gem:reset()
        local function onCompleteMe  (sender)
            self.animatedCollapsedGems =  self.animatedCollapsedGems - 1;
            if ( self.animatedCollapsedGems == 0 ) then
                if (onComplete ~= nil) then onComplete() end
            end
        end
        local move = cc.EaseBounceOut:create  (cc.MoveTo:create (0.6, cc.p(gem:getPositionX(), y) ) )
        local action = cc.Sequence:create  (cc.DelayTime:create(delay), move,  cc.CallFunc:create(onCompleteMe))
        gem:runAction(action)
    end
    ```

再次强调，我们只在所有宝石都坍塌后才会触发最终的回调。这个最终的回调将运行另一个检查匹配，就像我们之前看到的，然后再次启动整个过程。

## *刚才发生了什么？*

就这样；我们已经拥有了三合一游戏的三个主要部分：交换、匹配和坍塌。

我们还没有介绍的一个动画，它已经包含在本章的代码中，那就是当网格首次创建时，用于介绍动画的列下降动画。但那个并没有什么新意。尽管如此，你可以随意查看它。

现在，是时候发布游戏了。

# 是时候使用 Cocos IDE 发布游戏了。

为了构建和发布游戏，我们需要告诉 IDE 一些信息。我会展示如何为 Android 发布游戏，但步骤对于其他目标也非常相似：

1.  首先，让我们告诉 IDE 在哪里可以找到 Android SDK、NDK 和 ANT，就像我们安装 Cocos2d-x 控制台时做的那样。在 IDE 中，打开**首选项**面板。然后，在**Cocos**下输入三个路径，就像我们之前做的那样（记住，对于 ANT，你需要导航到它的`bin`文件夹）。![使用 Cocos IDE 发布游戏的操作时间](img/00042.jpeg)

1.  现在，为了构建项目，你需要选择 IDE 顶部的第四个按钮（从左侧开始），或者右键点击你的项目并选择**Cocos Tools**。根据你在部署过程中的阶段，你将会有不同的选项可用。![使用 Cocos IDE 发布游戏的操作时间](img/00043.jpeg)

    首先，IDE 需要添加原生代码支持，然后它会在名为 frameworks 的文件夹内构建项目（它将包含 iOS、Mac OS、Windows、Android 和 Linux 版本的你的项目，就像你通过 Cocos 控制台创建它一样）。

1.  然后，你可以选择将应用程序打包成 APK 或 IPA，你可以将其传输到你的手机上。或者，你可以使用 Eclipse 或 Xcode 中的生成项目。

## *刚才发生了什么？*

你刚刚将你的 Lua 游戏构建到了 Android、iOS、Windows、Linux、Mac OS，或者所有这些平台！做得好。

# 概述

就这些。你现在可以选择 C++ 或 Lua 来构建你的游戏。整个 API 都可以通过这两种方式访问。所以，这本书中创建的每个游戏都可以用这两种语言（是的，包括 Box2D API）来完成。

这本书就到这里了。希望你不是太累，可以开始着手自己的想法。并且我希望不久能在附近的 App Store 中看到你的游戏！

# 附录 A. 使用 Cocos2d-x 进行向量计算

本附录将更详细地介绍 第五章 中使用的数学概念，“在线上 – 火箭穿越”。

# 什么是向量？

首先，让我们快速回顾一下向量以及你如何使用 Cocos2d-x 来处理它们。

那么，向量与点的区别是什么？起初，它们看起来很相似。考虑以下点和向量：

+   点 (2, 3.5)

+   Vec2 (2, 3.5)

以下图展示了点和向量：

![什么是向量？](img/00044.jpeg)

在这个图中，它们每个的 *x* 和 *y* 值都相同。那么区别在哪里？

使用向量，你总是有额外的信息。就好像，除了 *x* 和 *y* 这两个值之外，我们还有向量的原点 *x* 和 *y* 的值，在之前的图中我们可以假设它是点 (0, 0)。所以向量是 *移动* 在从点 (0, 0) 到点 (2, 3.5) 描述的方向上。我们可以从向量中推导出的额外信息是方向和长度（通常称为大小）。

就好像向量是一个人的步幅。我们知道每一步有多长，也知道这个人朝哪个方向走。

在游戏开发中，向量可以用来描述运动（速度、方向、加速度、摩擦等）或作用于物体的合力。

## 向量方法

你可以用向量做很多事情，有很多种方法来创建和操作它们。Cocos2d-x 还附带了一些辅助方法，可以帮助你完成大部分计算。以下是一些示例：

+   你有一个向量，并且想要得到它的角度——使用 `getAngle()`

+   你想要一个向量的长度——使用 `getLength()`

+   你想要减去两个向量；例如，为了通过另一个向量减少精灵的移动量——使用 `vector1 - vector2`

+   你想要添加两个向量；例如，为了通过另一个向量增加精灵的移动量——使用 `vector1 + vector2`

+   你想要乘以一个向量；例如，将摩擦值应用到精灵的移动量上——使用 `vector1 * vector2`

+   你想要一个垂直于另一个向量（也称为向量的法线）的向量——使用 `getPerp()` 或 `getRPerp()`

+   最重要的是，对于我们的游戏示例，你想要两个向量的点积——使用 `dot(vector1, vector2)`

现在让我给你展示如何在我们的游戏示例中使用这些方法。

# 使用 ccp 辅助方法

在*Rocket Through*的例子中，我们在第五章中开发的*On the Line – Rocket Through*游戏中使用了向量来描述运动，现在我想向你展示我们用来处理向量操作的一些方法的逻辑以及它们的含义。

## 围绕一个点旋转火箭

让我们以火箭精灵以向量（5, 0）移动为例开始：

![围绕一个点旋转火箭](img/00045.jpeg)

然后，我们从火箭画一条线，比如说从点**A**到点**B**：

![围绕一个点旋转火箭](img/00046.jpeg)

现在我们想让火箭围绕点**B**旋转。那么我们如何改变火箭的向量来实现这一点？使用 Cocos2d-x，我们可以使用辅助点方法`rotateByAngle`来围绕任何其他点旋转一个点。在这种情况下，我们通过一定的角度将火箭的位置点围绕点**B**旋转。

但这里有一个问题——火箭应该朝哪个方向旋转？

![围绕一个点旋转火箭](img/00047.jpeg)

通过观察这个图，你知道火箭应该顺时针旋转，因为它正在向右移动。但程序上，我们如何确定这一点，并且以最简单的方式确定？我们可以通过使用向量和从它们导出的另一个属性：点积来确定这一点。

## 使用向量的点积

两个向量的点积描述了它们的角关系。如果它们的点积大于零，则两个向量形成的角度小于 90 度。如果它小于零，则角度大于 90 度。如果它等于零，则向量是垂直的。看看这个描述性的图：

![使用向量的点积](img/00048.jpeg)

但另一种思考方式是，如果点积是一个正值，那么向量将“指向”同一方向。如果它是负值，它们指向相反的方向。我们如何利用这一点来帮助我们？

向量始终有两个垂线，如图所示：

![使用向量的点积](img/00049.jpeg)

这些垂线通常被称为左右或顺时针和逆时针垂线，并且它们自身也是向量，被称为法线。

现在，如果我们计算火箭的向量与线**AB**上的每个垂线之间的点积，你可以看到我们可以确定火箭应该旋转的方向。如果火箭和向量的右垂线的点积是一个正值，这意味着火箭正在向右移动（顺时针）。如果不是，这意味着火箭正在向左移动（逆时针）。

![使用向量的点积](img/00050.jpeg)

点积非常容易计算。我们甚至不需要担心公式（尽管它很简单），因为我们可以使用`d` `ot(vector1, vector2)`方法。

因此，我们已经有火箭的向量了。我们如何得到法线的向量？首先，我们得到**AB**线的向量。我们为此使用另一个方法——`point1 - point2`。这将减去点**A**和**B**，并返回表示该线的向量。

接下来，我们可以使用`getPerp()`和`getRPerp()`方法分别得到那个线向量的左右垂直线。然而，我们只需要检查其中一个。然后我们用`dot(rocketVector, lineNormal)`得到点积。

如果这是正确的法线，意味着点积的值是正的，我们可以将火箭旋转到指向这个法线方向；因此，当火箭旋转时，它将始终与线保持 90 度角。这很容易，因为我们可以用`getAngle()`方法将法线向量转换为角度。我们只需要将这个角度应用到火箭上。

但火箭应该旋转多快？我们将在下一部分看到如何计算这一点。

## 从基于像素的速度转换为基于角度的速度

当旋转火箭时，我们仍然希望显示它以与直线移动时相同的速度移动，或者尽可能接近。我们如何做到这一点？

![从基于像素的速度转换为基于角度的速度](img/00051.jpeg)

记住，向量正在被用来在每次迭代中更新火箭的位置。在我给出的例子中，(5, 0)向量目前在每次迭代中向火箭的 x 位置添加 5 像素。

现在让我们考虑角速度。如果角速度是 15 度，并且我们保持以那个角度旋转火箭的位置，这意味着火箭将在 24 次迭代内完成一个完整的圆。因为一个完整圆的 360 度除以 15 度等于 24。

但我们还没有正确的角度；我们只有火箭在每次迭代中移动的像素量。但数学可以在这里告诉我们很多。

数学告诉我们，圆的长度是**圆的半径乘以π的两倍**，通常写作**2πr**。

我们知道我们想要火箭描述的圆的半径。它是我们画的线的长度。

![从基于像素的速度转换为基于角度的速度](img/00052.jpeg)

使用那个公式，我们可以得到那个圆的像素长度，也称为其周长。假设线的长度为 100 像素；这意味着火箭即将描述的圆的长度（或周长）为 628.3 像素（2 * π * 100）。

使用向量中描述的速度（5, 0），我们可以确定火箭完成那个像素长度需要多长时间。我们不需要这绝对精确；最后一次迭代最有可能超过那个总长度，但对于我们的目的来说已经足够好了。

![从基于像素的速度转换为基于角度的速度](img/00053.jpeg)

当我们有了完成长度所需的总迭代次数，我们可以将其转换为角度。所以，如果迭代值是 125，角度将是 360 度除以 125；即，2.88 度。这将是在 125 次迭代中描述圆所需的角。

![从基于像素的速度转换为基于角度的速度](img/00054.jpeg)

现在，火箭可以从基于像素的运动转换为基于角度的运动，而视觉变化不大。

# 附录 B. 突击测验答案

# 第四章，与精灵的乐趣 – 天空防御

## 突击测验 – 精灵和动作

| Q1 | 2 |
| --- | --- |
| Q2 | 1 |
| Q3 | 3 |
| Q4 | 4 |

# 第八章，物理化 – Box2D

## 突击测验

| Q1 | 3 |
| --- | --- |
| Q2 | 2 |
| Q3 | 1 |
| Q4 | 3 |
