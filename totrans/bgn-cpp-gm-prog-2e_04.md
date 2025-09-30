# *第三章*：C++ 字符串和 SFML 时间 – 玩家输入和 HUD

在本章中，我们将继续开发 Timber!!游戏。我们将用大约一半的时间学习如何操作文本并在屏幕上显示它，另一半时间将探讨计时以及如何通过视觉时间条让玩家了解剩余时间，并在游戏中创造紧迫感。

我们将涵盖以下主题：

+   暂停和重新启动游戏

+   C++ 字符串

+   SFML 文本和 SFML 字体类

+   为 Timber!!!添加一个 HUD

+   为 Timber!!!添加时间条

# 暂停和重新启动游戏

在接下来的三章中，我们将会对这款游戏进行开发，代码显然会越来越长。因此，现在似乎是提前思考并给我们的代码添加更多结构的好时机。我们将添加这种结构，以便我们可以暂停和重新启动游戏。

我们将添加代码，使得当游戏第一次运行时，它将处于暂停状态。玩家随后可以按下*Enter*键来开始游戏。然后，游戏将继续运行，直到玩家被压扁或用完时间。此时，游戏将暂停并等待玩家按下*Enter*键，以便他们可以重新启动游戏。

让我们一步一步地设置这个变量。

首先，在主游戏循环外部声明一个新的`bool`变量，命名为`paused`，并将其初始化为`true`：

```cpp
// Variables to control time itself
Clock clock;
// Track whether the game is running
bool paused = true;
while (window.isOpen())
{
    /*
    ****************************************
    Handle the players input
    ****************************************
    */
```

现在，每次游戏运行时，我们都有一个`paused`变量，它的初始值是`true`。

接下来，我们将添加另一个`if`语句，其中表达式将检查*Enter*键是否当前被按下。如果是被按下，它将`paused`设置为`false`。在其他的键盘处理代码之后添加以下突出显示的代码：

```cpp
/*
****************************************
Handle the players input
****************************************
*/
if (Keyboard::isKeyPressed(Keyboard::Escape))
{
    window.close();
}
// Start the game
if (Keyboard::isKeyPressed(Keyboard::Return))
{
paused = false; 
}
/*
****************************************
Update the scene
****************************************
*/
```

现在，我们有一个名为`paused`的`bool`变量，它一开始是`true`，但当玩家按下*Enter*键时，它将变为`false`。在这个时候，我们必须让我们的游戏循环根据`paused`的当前值做出适当的响应。

我们将这样进行。我们将整个更新部分的代码，包括我们在上一章中编写的移动蜜蜂和云的代码，包裹在一个`if`语句中。

在下面的代码中，请注意，只有当`paused`不等于`true`时，`if`块才会执行。换句话说，当游戏处于暂停状态时，游戏不会移动/更新。

这正是我们想要的。仔细看看我们添加新`if`语句及其对应的开闭花括号`{...}`的位置。如果它们放在错误的位置，事情将不会按预期工作。

将以下突出显示的代码添加到更新部分的代码中，注意上下文。我在几行代码中添加了`...`来表示隐藏的代码。显然，...不是真正的代码，不应该添加到游戏中。你可以通过周围的未突出显示代码来确定放置新代码（突出显示）的开始和结束位置：

```cpp
/*
****************************************
Update the scene
****************************************
*/
if (!paused)
{
    // Measure time
                ...
        ...
        ...

        // Has the cloud reached the right hand edge of the screen?
        if (spriteCloud3.getPosition().x > 1920)
        {
            // Set it up ready to be a whole new cloud next frame
            cloud3Active = false;
        }
    }
} // End if(!paused)
/*
****************************************
Draw the scene
****************************************
*/
```

注意，当你放置新的 `if` 块的闭合花括号时，Visual Studio 会整洁地调整所有缩进来保持代码整洁。

现在，你可以运行游戏，直到你按下 *Enter* 键，一切都将保持静态。现在，你可以开始添加游戏功能。我们只需要记住，当玩家死亡或用完时间时，我们需要将 `paused` 设置为 `true`。

在上一章中，我们第一次了解了 C++ 字符串。我们需要了解它们更多，以便我们可以实现玩家的 HUD。

# C++ 字符串

在上一章中，我们简要介绍了字符串，并了解到字符串可以存储字母数字数据——从单个字符到整本书。我们没有查看声明、初始化或操作字符串，所以现在让我们来做这件事。

## 声明字符串

声明字符串变量很简单。这与我们在上一章中用于其他变量的过程相同：我们声明类型，然后是名称：

```cpp
String levelName;
String playerName;
```

一旦我们声明了一个字符串，我们就可以给它赋值。

## 为字符串赋值

要为字符串赋值，就像常规变量一样，我们只需写出名称，然后是赋值运算符，最后是值：

```cpp
levelName = "DastardlyCave";
playerName = "John Carmack";
```

注意，值需要用引号括起来。就像常规变量一样，我们也可以在一行中声明和赋值：

```cpp
String score = "Score = 0";
String message = "GAME OVER!!";
```

在下一节中，我们将看到如何更改我们的字符串变量的值。

## 字符串操作

我们可以使用 `#include <sstream>` 指令为我们提供一些额外的字符串操作选项。`sstream` 类允许我们将一些字符串“相加”。当我们把字符串相加时，这被称为**连接**：

```cpp
String part1 = "Hello ";
String part2 = "World";
sstream ss;
ss<< part1 << part2;
// ss now holds "Hello World"
```

此外，通过使用 `sstream` 对象，字符串变量甚至可以与不同类型的变量连接。以下代码开始揭示字符串可能对我们有多有用：

```cpp
String scoreText = "Score = ";
int score = 0;
// Later in the code
score ++;
sstream ss;
ss<<scoreText<< score;
// ss now holds "Score = 1"
```

在前面的代码中，`ss` 用于将 `scoreText` 的内容与 `score` 的值连接起来。请注意，尽管 `score` 保持一个 `int` 值，但 `ss` 最终持有的值仍然是一个包含等效值的字符串；在这种情况下，"1"。

小贴士

`<<` 运算符是位运算符之一。然而，C++ 允许你编写自己的类并覆盖在类上下文中特定运算符的行为。`sstream` 类就是这样做的，以便 `<<` 运算符按这种方式工作。复杂性被隐藏在类中。我们可以使用其功能而不必担心它是如何工作的。如果你感到好奇，你可以阅读有关运算符重载的信息，请参阅 [`www.tutorialspoint.com/cplusplus/cpp_overloading.htm`](http://www.tutorialspoint.com/cplusplus/cpp_overloading.htm)。为了继续项目，你不需要更多的信息。

现在我们已经了解了 C++ 字符串的基础以及如何使用 `sstream`，我们将看看如何使用一些 SFML 类在屏幕上显示它们。

# SFML 的文本和字体类

在我们继续添加游戏代码之前，让我们通过一些假设的代码来讨论`Text`和`Font`类。

在屏幕上绘制文本的第一步是拥有一个字体。在*第一章*，*C++、SFML、Visual Studio 和开始第一个游戏*中，我们将字体文件添加到了项目文件夹中。现在，我们可以将字体加载到 SFML `Font`对象中，使其准备好使用。

实现此目的的代码如下：

```cpp
Font font;
font.loadFromFile("myfont.ttf");
```

在前面的代码中，我们首先声明了`Font`对象，然后加载了一个实际的字体文件。请注意，`myfont.ttf`是一个假设的字体，我们可以在项目文件夹中使用任何字体。

一旦我们加载了一个字体，我们需要一个 SFML `Text`对象：

```cpp
Text myText;
```

现在，我们可以配置我们的`Text`对象。这包括大小、颜色、屏幕上的位置、包含消息的字符串，以及当然，将其与我们的`font`对象关联的操作：

```cpp
// Assign the actual message
myText.setString("Press Enter to start!");
// assign a size
myText.setCharacterSize(75);
// Choose a color
myText.setFillColor(Color::White);
// Set the font to our Text object
myText.setFont(font);
```

现在我们能够创建和操作字符串值，以及分配、声明和初始化 SFML `Text`对象，我们可以继续到下一部分，在那里我们将为 Timber 添加 HUD！！！

# 实现 HUD

现在，我们对字符串、SFML `Text`和 SFML `Font`有了足够的了解，可以着手实现 HUD。**HUD**代表**抬头显示**。它可以像屏幕上的分数和文本消息那样简单，也可以包括更复杂元素，如时间条、小地图或指南针，代表玩家角色面对的方向。

要开始使用 HUD，我们需要在代码文件顶部添加另一个`#include`指令以添加对`sstream`类的访问。正如我们已知的，`sstream`类为将字符串和其他变量类型组合成字符串添加了一些非常实用的功能。

添加以下高亮代码行：

```cpp
#include <sstream>
#include <SFML/Graphics.hpp>
using namespace sf;
int main()
{
```

接下来，我们将设置我们的 SFML `Text`对象：一个用于存储我们将根据游戏状态变化的消息，另一个将存储分数并需要定期更新。

代码声明了`Text`和`Font`对象，加载了字体，将字体分配给`Text`对象，然后添加了字符串消息、颜色和大小。这应该与我们在上一节中的讨论相似。此外，我们添加了一个名为`score`的新`int`变量，我们可以操作它，使其包含玩家的分数。

小贴士

记住，如果你在*第一章**C++、SFML、Visual Studio 和开始第一个游戏*中选择了不同的字体`KOMIKAP_.ttf`，你需要将代码中相应部分更改为与`Visual Studio Stuff/Projects/Timber/fonts`文件夹中的`.ttf`文件匹配。

通过添加以下高亮代码，我们将准备好继续更新 HUD：

```cpp
// Track whether the game is running
bool paused = true;
// Draw some text
int score = 0;
Text messageText;
Text scoreText;
// We need to choose a font
Font font;
font.loadFromFile("fonts/KOMIKAP_.ttf");
// Set the font to our message
messageText.setFont(font);
scoreText.setFont(font);
// Assign the actual message
messageText.setString("Press Enter to start!");
scoreText.setString("Score = 0");
// Make it really big
messageText.setCharacterSize(75);
scoreText.setCharacterSize(100);
// Choose a color
messageText.setFillColor(Color::White);
scoreText.setFillColor(Color::White);
while (window.isOpen())
{
    /*
    ****************************************
    Handle the players input
    ****************************************
    */
```

在前面的代码中，我们实现了以下内容：

+   声明一个变量来保存分数

+   声明了一些 SFML `Text`和`Font`对象

+   通过从文件中加载字体初始化`Font`对象

+   使用字体和一些字符串初始化`Text`对象

+   使用`setCharacterSize`和`setFillColor`函数设置`Text`对象的大小和颜色

以下代码片段可能看起来有些复杂，甚至有些复杂。然而，当你稍微分解它时，它却是直截了当的。检查并添加新的突出显示的代码。我们将在之后讨论它：

```cpp
// Choose a color
messageText.setFillColor(Color::White);
scoreText.setFillColor(Color::White);
// Position the text
FloatRect textRect = messageText.getLocalBounds();
messageText.setOrigin(textRect.left +
    textRect.width / 2.0f,
    textRect.top +
    textRect.height / 2.0f);
messageText.setPosition(1920 / 2.0f,	1080 / 2.0f);
scoreText.setPosition(20, 20);
while (window.isOpen())
{
    /*
    ****************************************
    Handle the players input
    ****************************************
    */
```

我们有两个`Text`类型的对象，我们将在屏幕上显示。我们希望将`scoreText`定位在左上角，并留有一定的填充。这并不构成挑战；我们只需使用`scoreText.setPosition(20, 20)`，这样它就会定位在左上角，水平方向和垂直方向各有 20 像素的填充。

然而，定位`messageText`并不那么简单。我们希望将其定位在屏幕的精确中点。最初，这可能看起来不是问题，但我们必须记住，我们绘制的一切的原点都在左上角。因此，如果我们简单地将屏幕宽度和高度除以二，并在`mesageText.setPosition...`中使用结果，那么文本的左上角就会在屏幕中心，并且它将杂乱无章地向右扩展。

以下是为了方便再次讨论的代码：

```cpp
// Position the text
FloatRect textRect = messageText.getLocalBounds();
messageText.setOrigin(textRect.left +
    textRect.width / 2.0f,
    textRect.top +
    textRect.height / 2.0f);
```

代码所做的就是将`messageText`的*中心*设置为屏幕的中心。我们正在审查的看起来相当复杂的代码片段重新定位了`messageText`的原点到其自身中心。

在前面的代码中，我们首先声明了一个名为`textRect`的新对象，其类型为`FloatRect`。正如其名称所暗示的，`FloatRect`对象包含一个具有浮点坐标的矩形。

然后，代码使用`mesageText.getLocalBounds`函数用包围`messageText`的矩形的坐标初始化`textRect`。

下一行代码，由于它相当长，被分散在四行中，使用了`messageText.setOrigin`函数来改变原点（用于绘制的点）到`textRect`的中心。当然，`textRect`包含一个与`messageText`坐标相匹配的矩形。然后，执行以下代码行：

```cpp
messageText.setPosition(1920 / 2.0f,	1080 / 2.0f);
```

现在，`messageText`将被整齐地定位在屏幕的精确中心。我们将使用此代码每次更改`messageText`的文本，因为更改消息会改变`messageText`的大小，因此需要重新计算其原点。

接下来，我们声明了一个名为`ss`的`stringstream`类型对象。请注意，我们使用了包括命名空间在内的完整名称，即`std::stringstream`。我们可以通过在代码文件顶部添加`using namespace std`来避免这种语法。不过，在这里我们不会这样做，因为我们很少使用它。请查看以下代码并将其添加到游戏中；然后，我们可以更详细地讨论它。由于我们只想在游戏未暂停时执行此代码，请确保将其与其他代码一起添加到`if(!paused)`块中，如下所示：

```cpp
else
    {
        spriteCloud3.setPosition(
            spriteCloud3.getPosition().x +
            (cloud3Speed * dt.asSeconds()),
            spriteCloud3.getPosition().y);
        // Has the cloud reached the right hand edge of the screen?
        if (spriteCloud3.getPosition().x > 1920)
        {
            // Set it up ready to be a whole new cloud next frame
            cloud3Active = false;
        }
    }
    // Update the score text
    std::stringstream ss;
    ss<< "Score = " << score;
    scoreText.setString(ss.str());
}// End if(!paused)
/*
****************************************
Draw the scene
****************************************
*/
```

我们使用`ss`和`<<`运算符提供的特殊功能，该运算符将变量连接到一个`stringstream`中。在这里，`ss << "Score = " << score`的效果是创建一个包含`"Score = "`的字符串。无论`score`的值是多少，都会被连接在一起。例如，当游戏第一次开始时，`score`等于零，所以`ss`将保持`"Score = 0"`的值。如果`score`发生变化，`ss`将适应每一帧。

以下行代码只是将`ss`中包含的字符串设置为`scoreText`：

```cpp
scoreText.setString(ss.str());
```

现在它已经准备好被绘制到屏幕上了。

以下代码绘制了两个`Text`对象（`scoreText`和`messageText`），但绘制`messageText`的代码被包裹在一个`if`语句中。这个`if`语句导致`messageText`只有在游戏暂停时才会被绘制。

添加以下高亮代码：

```cpp
// Now draw the insect
window.draw(spriteBee);
// Draw the score
window.draw(scoreText);
if (paused)
{
    // Draw our message
    window.draw(messageText);
}
// Show everything we just drew
window.display();
```

我们现在可以运行游戏，并看到我们的 HUD 被绘制到屏幕上。你会看到**得分 = 0**和**按回车键开始**的消息。当你按下*Enter*键时，后者将消失：

![图片](img/B14278_03_01.jpg)

如果你想要看到分数更新，请在`while(window.isOpen)`循环中的任何地方添加一个临时行代码，`score ++;`。如果你添加了这个临时行代码，你会看到分数快速上升，非常快！

如果你添加了临时代码，即`score ++;`，在继续之前务必将其删除。

# 添加时间条

由于时间在游戏中是一个关键机制，因此有必要让玩家意识到这一点。他们需要知道他们分配的六秒是否即将用完。当游戏接近结束时，这会给他们一种紧迫感；如果他们表现良好，能够保持或增加剩余时间，这会给他们一种成就感。

在屏幕上绘制剩余秒数不易阅读（当专注于分支时），也不是实现目标的一种特别有趣的方式。

我们需要的是一个时间条。我们的时间条将是一个简单且突出显示在屏幕上的红色矩形。它一开始会很宽，但随着时间的流逝会迅速缩小。当玩家的剩余时间达到零时，时间条将完全消失。

同时添加时间条时，我们还将添加必要的代码来跟踪玩家的剩余时间，并在时间用尽时做出响应。让我们一步一步地完成这个过程。

找到之前声明的`Clock clock;`，在其后添加高亮代码，如下所示：

```cpp
// Variables to control time itself
Clock clock;
// Time bar
RectangleShape timeBar;
float timeBarStartWidth = 400;
float timeBarHeight = 80;
timeBar.setSize(Vector2f(timeBarStartWidth, timeBarHeight));
timeBar.setFillColor(Color::Red);
timeBar.setPosition((1920 / 2) - timeBarStartWidth / 2, 980);
Time gameTimeTotal;
float timeRemaining = 6.0f;
float timeBarWidthPerSecond = timeBarStartWidth / timeRemaining;
// Track whether the game is running
bool paused = true;
```

首先，我们声明一个`RectangleShape`类型的对象，并将其命名为`timeBar`。`RectangleShape`是 SFML 类，非常适合绘制简单的矩形。

接下来，我们将添加几个`float`类型的变量，`timeBarStartWidth`和`timeBarHeight`。我们将它们分别初始化为`400`和`80`。这些变量将帮助我们跟踪在每一帧绘制`timeBar`所需的大小。

接下来，我们使用`timeBar.setSize`函数设置`timeBar`的大小。我们不仅仅传递两个新的`float`变量。首先，我们创建一个新的`Vector2f`类型的对象。然而，这里的不同之处在于我们没有给这个新对象命名。相反，我们直接用两个浮点变量初始化它，并将其直接传递给`setSize`函数。

提示

`Vector2f`是一个包含两个`float`变量的类。它还有一些其他功能，将在本书的其余部分介绍。

之后，我们通过使用`setFillColor`函数将`timeBar`涂成红色。

在之前的代码中我们对`timeBar`做的最后一件事是设置其位置。垂直坐标非常直接，但设置水平坐标的方式稍微复杂一些。这里再次进行计算：

```cpp
(1920 / 2) - timeBarStartWidth / 2
```

首先，代码将 1920 除以 2。然后，它将`timeBarStartWidth`除以 2。最后，它从前者减去后者。

结果使`timeBar`整齐地、水平地位于屏幕中央。

我们正在讨论的最后三行代码声明了一个名为`gameTimeTotal`的新`Time`对象，一个初始化为`6`的新`float`变量`timeRemaining`，以及一个听起来很奇怪的名为`timeBarWidthPerSecond`的`float`变量，我们将在下一节讨论。

`timeBarWidthPerSecond`变量通过将`timeBarStartWidth`除以`timeRemaining`初始化。结果是`timeBar`每秒需要缩小的像素数。当我们在每一帧调整`timeBar`大小时，这将很有用。

显然，每次玩家开始新游戏时，我们都需要重置剩余时间。进行此操作的逻辑位置是在按下*Enter*键时。我们还可以同时将`score`设置回零。现在让我们添加以下突出显示的代码：

```cpp
// Start the game
if (Keyboard::isKeyPressed(Keyboard::Return))
{
    paused = false;
    // Reset the time and the score
    score = 0;
    timeRemaining = 6;
}
```

现在，我们必须通过剩余时间减少每一帧，并相应地调整`timeBar`的大小。将以下突出显示的代码添加到更新部分，如下所示：

```cpp
/*
****************************************
Update the scene
****************************************
*/
if (!paused)
{
    // Measure time
    Time dt = clock.restart();
    // Subtract from the amount of time remaining
    timeRemaining -= dt.asSeconds();
    // size up the time bar
    timeBar.setSize(Vector2f(timeBarWidthPerSecond *
        timeRemaining, timeBarHeight));
    // Set up the bee
    if (!beeActive)
    {
        // How fast is the bee
        srand((int)time(0) * 10);
        beeSpeed = (rand() % 200) + 200;
        // How high is the bee
        srand((int)time(0) * 10);
        float height = (rand() % 1350) + 500;
        spriteBee.setPosition(2000, height);
        beeActive = true;
    }
    else
        // Move the bee
```

首先，我们通过以下代码减去玩家剩余的时间与上一帧执行所需的时间量：

```cpp
timeRemaining -= dt.asSeconds();
```

然后，我们使用以下代码调整了`timeBar`的大小：

```cpp
timeBar.setSize(Vector2f(timeBarWidthPerSecond *
        timeRemaining, timeBarHeight));
```

`Vector2F`的 x 值在乘以`timeRemaining`时初始化为`timebarWidthPerSecond`，这会产生与玩家剩余时间成正比的正确宽度。高度保持不变，`timeBarHeight`未经过任何操作就被使用。

当然，我们必须检测时间是否已耗尽。目前，我们将简单地检测时间是否已耗尽，暂停游戏，并更改`messageText`的文本。稍后，我们将在这里做更多的工作。将以下突出显示的代码添加到之前添加的代码之后。我们稍后会详细讨论它：

```cpp
// Measure time
Time dt = clock.restart();
// Subtract from the amount of time remaining
timeRemaining -= dt.asSeconds();
// resize up the time bar
timeBar.setSize(Vector2f(timeBarWidthPerSecond *
    timeRemaining, timeBarHeight));
if (timeRemaining<= 0.0f) {

    // Pause the game
    paused = true;
    // Change the message shown to the player
    messageText.setString("Out of time!!");
    //Reposition the text based on its new size
    FloatRect textRect = messageText.getLocalBounds();
    messageText.setOrigin(textRect.left +
        textRect.width / 2.0f,
        textRect.top +
        textRect.height / 2.0f);
    messageText.setPosition(1920 / 2.0f, 1080 / 2.0f);
}
// Set up the bee
if (!beeActive)
{
    // How fast is the bee
    srand((int)time(0) * 10);
    beeSpeed = (rand() % 200) + 200;
    // How high is the bee
    srand((int)time(0) * 10);
    float height = (rand() % 1350) + 500;
    spriteBee.setPosition(2000, height);
    beeActive = true;
}
else
    // Move the bee
```

让我们逐步分析之前的代码：

1.  首先，我们使用`if(timeRemaining<= 0.0f)`测试时间是否已耗尽。

1.  然后，我们将 `paused` 设置为 `true`，因此这将是我们代码更新部分的最后一次执行（直到玩家再次按下 *Enter* 键）。

1.  然后，我们更改 `messageText` 的信息，计算其新的中心并将其设置为原点，并将其定位在屏幕中央。

最后，对于这段代码，我们需要绘制 `timeBar`。这段代码中没有我们之前没有多次见过的内容。只需注意，我们在绘制树之后绘制 `timeBar`，这样它就不会被部分遮挡。添加以下高亮代码以绘制时间条：

```cpp
// Draw the score
window.draw(scoreText);
// Draw the timebar
window.draw(timeBar);
if (paused)
{
    // Draw our message
    window.draw(messageText);
}
// Show everything we just drew
window.display();
```

现在，你可以运行游戏，按下 *Enter* 键开始它，并观察时间条平滑地消失到无：

![](img/Image85176.jpg)

游戏随后暂停，屏幕中央将出现 **OUT OF TIME!!** 信息：

![](img/B14278_03_04.jpg)

当然，你可以按下 *Enter* 键重新开始游戏，并从开始观看它运行。

# 摘要

在本章中，我们学习了字符串、SFML `Text` 和 SFML `Font`。它们共同使我们能够在屏幕上绘制文本，为玩家提供了抬头显示（HUD）。我们还使用了 `sstream`，它允许我们将字符串和其他变量连接起来以显示分数。

我们还探讨了 SFML 的 `RectangleShape` 类，它确实如其名称所暗示的那样。我们使用 `RectangleShape` 类型的对象和一些精心策划的变量来绘制一个时间条，它以视觉方式向玩家显示他们剩余的时间。一旦我们实现了可以挤压玩家的砍伐和移动树枝，时间条将提供视觉反馈，从而创造紧张和紧迫感。

在下一章中，我们将学习一系列新的 C++ 功能，包括循环、数组、切换、枚举和函数。这将使我们能够移动树枝，跟踪它们的位置，并挤压玩家。

# 常见问题解答

Q) 我可以预见，有时通过精灵的左上角定位可能会不方便。有没有替代方案？

A) 幸运的是，你可以选择使用精灵的哪个点作为定位/原点像素，就像我们使用 `messageText` 时做的那样，通过使用 `setOrigin` 函数。

Q) 代码变得越来越长，我很难跟踪所有内容的位置。我们该如何解决这个问题？

A) 是的，我同意。在下一章中，我们将探讨几种组织代码和使其更易读的方法之一。我们将在我们学习编写 C++ 函数时探讨这一点。此外，当我们学习 C++ 数组时，我们还将了解一种处理相同类型多个对象/变量的新方法（如云）。
