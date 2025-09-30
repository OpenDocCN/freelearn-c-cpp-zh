# 第十一章。抬头

在本章中，我们将通过添加几乎任何游戏中都会看到的一些功能来对 Space Racer 3D 进行一些收尾工作。其中许多功能与我们在 Robo Racer 2D 游戏中添加的收尾工作类似，尽管现在我们在 3D 中工作有一些特殊考虑。我们将涵盖的主题包括以下内容：

+   **在 3D 世界中实现 2D**：到目前为止，我们学习了如何在 2D 中渲染，以及如何在 3D 中渲染。然而，在 3D 世界中创建 2D 内容有一些特殊考虑。由于我们的用户界面通常是 2D 创建的，我们将学习如何混合这两种类型的渲染。

+   **创建抬头显示（HUD）**：对于第一人称 3D 游戏来说，有一个持续显示与游戏相关的信息的状态是非常典型的。我们将学习如何创建一个基本的抬头显示或 HUD。

+   **更多游戏状态**：正如我们在 Robo Racer 2D 中所做的那样，我们将创建一个基本的状态管理器来处理我们完成的游戏中的各种模式。

+   **计分**：我们需要一种方法来在我们的游戏中计分，并需要设置基本的胜负条件。

+   **游戏结束**：当游戏结束时，我们将通过 3D 的转折点给予一些信用。

# 混合事物

现在我们正在 3D 中渲染，如何渲染 2D 内容并不立即明显。这尤其适用于我们的用户界面，它必须渲染在 3D 场景之上，并且不会随着世界其他部分移动或旋转。

在 3D 世界中创建 2D 界面的技巧是首先渲染 3D 世界，然后在 OpenGL 中切换模式，然后渲染 2D 内容。以下图像表示我们需要渲染的 3D 内容：

![混合事物](img/8199OS_11_01.jpg)

下一个图像表示我们想要渲染的 2D 文本：

![混合事物](img/8199OS_11_02.jpg)

我们希望最终结果是 3D 和 2D 内容的组合，如图所示：

![混合事物](img/8199OS_11_03.jpg)

## 保存状态

状态是游戏编程中用于许多不同方式的一个术语。例如，我们将在本章后面创建一个状态管理器来管理游戏中的不同状态或模式。定义状态的另一种方式是一组条件。例如，当我们设置渲染为 3D 时，这是一组条件或状态。当我们设置渲染为 2D 时，这是另一组条件或状态。

能够在 2D 和 3D 中渲染的技巧是能够设置一个状态，然后切换到另一个状态。OpenGL 通过矩阵保存状态。为了从一个状态切换到另一个状态，我们需要一种方法来保存当前矩阵，设置另一个矩阵，然后在我们完成时返回到先前的矩阵。

## 推送和弹出

OpenGL 提供了两种方法来保存当前状态并在稍后检索它：

+   `glPushMarix()`：此命令通过将其放置在堆栈上保存当前状态。

+   `glPopMatrix()`：此命令通过从堆栈中取出它来检索先前状态。

**栈**是一种结构，允许你将其数据放在顶部（一个**推入**），然后稍后从顶部检索该数据（一个**弹出**）。当你想要按顺序保存数据，然后稍后以相反的顺序检索它时，栈非常有用。

假设我们从一个称为**状态 A**的初始条件集开始：

![推入和弹出](img/8199OS_11_04.jpg)

调用`glPushMatrix()`会将**状态 A**压入栈中：

![推入和弹出](img/8199OS_11_05.jpg)

接下来，我们设置**状态 B**的条件。如果我们想保存这个状态，我们将发出另一个`glPushMatrix()`调用：

![推入和弹出](img/8199OS_11_06.jpg)

现在我们栈中有两个项目，这也应该非常清楚地说明为什么它被称为栈！然后我们可以定义**状态 C**。这个步骤序列可以按需继续，创建一个渲染状态并将其推入栈中。一般来说，我们希望以我们加载的相反顺序卸载栈。这被称为**FILO**栈：先进先出。

我们使用`glPopMatrix()`命令从栈中移除项目：

![推入和弹出](img/8199OS_11_07.jpg)

结果替换了**状态 C**，将渲染设置恢复到**状态 B**：

![推入和弹出](img/8199OS_11_08.jpg)

另一个`glPopMatrix()`调用会清空栈并将渲染设置恢复到**状态 A**：

![推入和弹出](img/8199OS_11_09.jpg)

模型视图允许将 32 个矩阵放入栈中。每个视图都有自己的栈，因此投影视图有一个与模型视图分开的栈。此外，如果你发出`glPopMatrix`，而栈中没有矩阵，你会收到一个错误。换句话说，不要尝试弹出比你推入的更多的东西！

### 小贴士

为了最佳地管理内存，你应该始终弹出你已推入的状态，即使你不需要对它们进行任何操作。这会释放出用于保存你正在保存的状态中数据所占用的内存。

## 双状态渲染

我们现在将设置我们的代码，使其能够在 3D 和 2D 中渲染。打开`SpaceRacer3D.cpp`。我们将把渲染分成两个函数：`Render3D`和`Render2D`。然后，我们将从主`Render`函数中调用这些函数。让我们从`Render3D`开始。在`Render`函数上方添加以下代码（你可以直接从`Render`函数中剪切它）：

```cpp
void Render3D()
{
 if (gameState == GS_Running)
 {
  for (unsigned int i = 0; i < asteroids.size(); i++)
  {
   asteroids[i]->Render();
  }
  ship->Render();
 }
}
```

接下来，我们将创建两个支持函数来打开和关闭 2D 渲染。第一个将是`Enable2D`。在`Render3D`函数上方添加以下函数：

```cpp
void Enable2D()
{
  glColor3f(1.0f, 1.0f, 1.0f);
  glEnable(GL_TEXTURE_2D);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, 0, 1);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glPushAttrib(GL_DEPTH_BUFFER_BIT);
  glDisable(GL_DEPTH_TEST);
}
```

`Enable2D`执行将渲染模式更改为 2D 所必需的任务：

+   `glColor3f`调用设置当前绘图颜色为白色。这需要一些解释。我们总是先渲染 3D，然后切换到 2D。如果我们没有将颜色设置为白色，那么 2D 内容中的所有颜色都会与 3D 渲染最后使用的颜色混合。将渲染颜色设置为白色实际上意味着清除渲染颜色，以便 2D 内容可以准确渲染。将颜色设置为白色并不意味着所有内容都会以白色绘制。这意味着不会向我们在 2D 中渲染的对象添加额外的颜色。

+   如果你想渲染 2D 纹理，`glEnable(GL_TEXTURE_2D)`调用是必需的。如果省略了这个调用，那么任何 2D 纹理都不会正确渲染。

+   接下来的四行代码保存了 3D 投影矩阵，并设置投影矩阵以 2D 模式渲染。`glPushMatrix`将当前投影矩阵推入栈中。然后我们使用`glLoadIdentity`初始化投影矩阵。最后，通过调用`glOrtho`设置正交投影。看看 RoboRacer2D，你会注意到它使用相同的`glOrtho`调用设置 2D 渲染！

+   接下来的三行代码保存了 3D 模型视图矩阵，并为其 2D 绘制初始化。`glPushMatrix`将当前模型视图矩阵推入栈中。然后我们通过调用`glLoadIdentity`初始化模型视图矩阵。

+   最后，我们需要关闭深度缓冲区的检查。深度缓冲区检查仅适用于 3D 渲染，并干扰 2D 渲染。`glPushAttrib`与`glPushMatrix`的工作方式类似，但它只将单个 OpenGL 属性推入栈中。在这种情况下，我们将当前的`GL_DEPTH_BUFFER_BIT`推入属性栈，从而保存之前 3D 渲染的当前状态。接下来，我们使用`glDisable`调用关闭深度检查。

因此，设置 2D 渲染环境涉及四个步骤：

1.  重置渲染颜色并启用 2D 纹理。

1.  保存 3D 投影矩阵并设置 2D 投影矩阵。

1.  保存 3D 模型视图矩阵并初始化 2D 模型视图矩阵。

1.  保存 3D 深度位并关闭 2D 中的深度检查。

现在，我们准备好编写`Disable2D`函数。在刚刚创建的`Enable2D`函数下方创建这个新函数：

```cpp
void Disable2D()
{
  glPopAttrib();

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  glDisable(GL_TEXTURE_2D);
}
```

并不令人意外的是，`Disable2D`执行的动作顺序与我们执行`Enable2D`时的顺序相反：

+   首先，我们通过调用`glPopAttrib()`恢复深度检查，该调用从属性栈中移除最后推入的属性并将其恢复到当前渲染状态。这将恢复到我们开始 2D 渲染之前的深度检查状态。

+   接下来的两行代码将投影矩阵恢复到 3D 状态。同样，`glPopMatrix`调用从栈顶取出项目并将其应用于当前渲染状态。

+   接下来的两行代码弹出模型视图矩阵。

+   最后一行禁用了 2D 纹理。

现在，是时候创建我们的`Render2D`函数了。在`Render3D`函数上方添加以下代码：

```cpp
void Render2D()
{
  Enable2D();
  // Future 2D rendering code here
  Disable2D();
}
```

好玩的是，我们目前还没有任何 2D 内容要渲染！在本章的后面部分，我们将填充这个函数的其余内容。这里需要注意的重要事情是，这个函数将负责通过调用`Enable2D`启用 2D 渲染。然后代码将被添加来渲染我们的 2D 内容。最后，我们将通过调用`Disable2D`关闭 2D 渲染。

现在我们已经有了渲染 2D 和 3D 所需的所有必要支持代码，我们将修改`Render`函数：

```cpp
void Render()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  Render3D();
  Render2D();
  SwapBuffers(hDC);
}
```

你会注意到现在这有多简单：

1.  首先，我们清除颜色缓冲区并重置矩阵。我们在每次渲染每一帧之前都会这样做。

1.  接下来，我们渲染 3D 内容。

1.  然后我们渲染 2D 内容。

1.  最后，我们交换缓冲区，这将渲染我们所有的内容到屏幕上。

如果你现在运行游戏，你应该会注意到没有任何变化。因为我们还没有创建任何 2D 内容来渲染，3D 内容将显示得和之前一样。现在我们准备添加我们的 2D 内容。在这个过程中，我们将完善一些额外的功能，以制作一个更完整的游戏。

# 状态问题

在我们开始实际渲染 2D 项目之前，我们需要在我们的游戏中添加一个状态机。就像我们在 RoboRacer2D 中所做的那样，我们需要能够处理几个不同的游戏状态：显示启动屏幕、加载资源、显示主菜单、运行游戏、暂停游戏和游戏结束。

### 小贴士

不要让“状态”这个词让你困惑，因为它在计算机编程中有多种用法。我们刚刚完成了一个关于渲染状态的章节，学习了如何从 OpenGL 堆栈中推送和弹出这个状态。现在，我们正在谈论游戏状态，你可以将其视为我们的游戏处于的不同模式。处理不同游戏状态的框架被称为**状态机**。

## 添加状态机

幸运的是，我们将能够直接从 RoboRacer2D 中获取一些代码。通过在 SpaceRacer3D 项目中点击**文件**，然后**打开**，并浏览到`RoboRacer2D.cpp`，你可以这样做。这将允许你从`RoboRacer2D.cpp`中复制信息并将其粘贴到 SpaceRacer3D 中。

### 小贴士

打开文件会将它加载到当前项目中，但不会将文件添加到当前项目中。然而，你需要小心，因为如果你修改了文件并保存，原始源文件将被修改。

复制`GameState`枚举，然后将其粘贴到`SpaceRacer3D.cpp`的顶部，紧随头文件之后：

```cpp
enum GameState
{
  GS_Splash,
  GS_Loading,
  GS_Menu,
  GS_Credits,
  GS_Running,
  GS_NextLevel,
  GS_Paused,
  GS_GameOver,
};
```

我们将复制更多代码来自 `RoboRacer2D.cpp`，所以请继续打开它。

接下来，我们需要创建一个全局游戏状态变量。在`SpaceRacer3D.cpp`的全局变量部分添加以下定义：

```cpp
GameState gameState;
```

`gameState`变量将存储当前游戏状态。

## 准备启动

正如我们在 RoboRacer2D 中所做的那样，我们将以启动屏幕开始我们的游戏。启动屏幕将在加载其他资源之前快速加载，并在移动到加载游戏资源和开始游戏之前显示几秒钟。

在`gameState`定义下方，添加以下几行：

```cpp
float splashDisplayTimer;
float splashDisplayThreshold;
```

这两个变量将处理启动屏幕的时间。我们的启动屏幕将是游戏加载的众多 2D 资源之一。让我们继续定义一些用于我们的 2D 资源的变量。将以下代码行添加到`SpaceRacer3D.cpp`的全局变量部分：

```cpp
Sprite* splashScreen;
Sprite* menuScreen;
Sprite* creditsScreen;
Sprite* playButton;
Sprite* creditsButton;
Sprite* exitButton;
Sprite* menuButton;
Sprite* gameOverScreen;
Sprite* replayButton;
```

你会注意到我们所有的 2D 资源都被处理为精灵（Sprites），这是一个我们从 RoboRacer2D 借用的类。

当我们在这里时，让我们也添加以下两行：

```cpp
float uiTimer;
const float UI_THRESHOLD = 0.1f;
```

这两个变量将用于为鼠标点击添加时间缓冲。现在，让我们创建一个加载启动屏幕的函数。将以下函数添加到`SpaceRacer3D.cpp`中，位置在`StartGame`函数之前：

```cpp
void LoadSplash()
{
  gameState = GameState::GS_Splash;
  splashScreen = new Sprite(1);
  splashScreen->SetFrameSize(screenWidth, screenHeight);
  splashScreen->SetNumberOfFrames(1);
  splashScreen->AddTexture("resources/splash.png", false);
  splashScreen->IsActive(true);
  splashScreen->IsVisible(true);
  splashScreen->SetPosition(0.0f, 0.0f);
}
```

这段代码与 RoboRacer2D 中的代码完全相同。实际上，你可以直接从`RoboRacer2D.cpp`中复制并粘贴它。

记住：我们设置了 2D 正交视口，以精确复制我们在 RoboRacer2D 中设置的设置。这允许我们使用完全相同的代码和位置来处理我们的 2D 对象。甚至更好，它允许我们使用 RoboRacer2D 中的`Sprite`类，而无需更改任何代码。

### 小贴士

`LoadSplash`函数从游戏资源文件夹中加载一个名为`splash.png`的文件。你可以从本书的网站下载这个文件以及本章中使用的所有其他 2D 资源。你应该将它们全部放在与游戏源代码相同的文件夹下的`resources`文件夹中。你还得记得通过右键点击**资源文件**，然后选择**添加现有项**，浏览到`resources`文件夹，并将该文件夹中的所有项目添加到**资源文件**文件夹中。

接下来，我们需要修改`StartGame`函数以加载启动屏幕。移动到`StartGame`函数，并添加以下代码：

```cpp
LoadSplash();
uiTimer = 0.0f;
splashDisplayTimer = 0.0f;
splashDisplayThreshold = 5.0f;
```

我们首先调用`LoadSplash`函数，将游戏状态设置为`GS_Splash`，然后加载启动页面。接下来，我们必须更新并渲染启动页面。移动到`Update`函数，并修改它，使其看起来像这样：

```cpp
void Update(const float p_deltaTime)
{
 switch (gameState)
 {
 case GameState::GS_Splash:
 case GameState::GS_Loading:
 {
  splashScreen->Update(p_deltaTime);
 }
 break;
 case GameState::GS_Running:
 {
  inputManager->Update(p_deltaTime);
  ProcessInput(p_deltaTime);
  ship->Update(p_deltaTime);
  ship->SetVelocity(ship->GetVelocity() + ship->GetVelocity()*p_deltaTime/10.0f);
  speed = ship->GetVelocity() * 1000;
  if (maximumSpeed < speed)
  {
   maximumSpeed = speed;
  }
  missionTime = missionTime + p_deltaTime * 100.0f;
  CheckCollisions();
  if (ship->GetPosition().z > 10.0f)
  {
   gameState = GS_GameOver;
   menuButton->IsActive(true);
   gameOverScreen->IsActive(true);
  }
 }
 break;
 case GameState::GS_GameOver:
 {
  gameOverScreen->Update(p_deltaTime);
  replayButton->IsActive(true);
  replayButton->Update(p_deltaTime);
  exitButton->IsActive(true);
  exitButton->Update(p_deltaTime);
  inputManager->Update(p_deltaTime);
  ProcessInput(p_deltaTime);
  ship->Update(p_deltaTime);
  CheckCollisions();
 }
 break;
 }
}
```

唯一真正的变化是我们实现了一部分状态机。你会注意到我们将所有运行游戏的代码移动到了`GS_Running`游戏状态案例下。接下来，我们添加了对启动屏幕游戏状态的更新。我们最终将修改`Update`函数以处理所有游戏状态，但我们还有一些工作要做。

现在，我们已经准备好渲染启动屏幕了。移动到`Render2D`函数，并在`Enable2D`和`Disable2D`调用之间添加以下代码行：

```cpp
splashScreen->Render();
```

在这一点上，如果你运行游戏，你会看到一个启动屏幕被渲染。游戏不会超出启动屏幕，因为我们还没有添加前进的代码。

# 创建用户界面

我们现在准备好定义我们的用户界面，它将包括 2D 屏幕、文本和按钮。这些都将与 RoboRacer2D 中的工作方式完全相同。查看本章前面“准备启动”部分中的提示，以提醒如何将预构建的 2D 资源包含到你的项目中。

## 定义文本系统

2D 文本系统是通过首先创建一个字体框架，然后创建在屏幕上显示文本的函数来构建的。打开`RoboRacer2D.cpp`并复制以下函数。然后将其粘贴到`SpaceRacer3D.cpp`中：

+   `BuildFont`

+   `KillFont`

+   `DrawText`

我们将添加一些新变量来处理我们想要显示的数据。将以下代码行添加到`SpaceRacer3D.cpp`的全局变量部分：

```cpp
int score;
int speed;
int missionTime;
int asteroidsHit;
int maximumSpeed;
```

这些变量将保存游戏使用的统计数据和得分：

+   `score`: 这是当前游戏得分

+   `speed`: 这是飞船的当前速度

+   `missionTime`: 这是自开始任务以来经过的秒数

+   `asteroidsHit`: 这是玩家击中的陨石数量

+   `maximumSpeed`: 这是玩家获得的最大速度

`Score`, `speed`, 和 `missionTime` 都将在玩家驾驶飞船时显示在抬头显示（HUD）上。`Score`, `asteroidsHit`, `missionTime`, 和 `maximumSpeed` 将在游戏结束时显示为统计数据。

让我们转到`StartGame`并初始化这些变量：

```cpp
score = 0;
speed = 1.0f;
maximumSpeed = 0;
asteroidsHit = 0;
missionTime = 0;
```

现在，让我们创建在屏幕上渲染这些项目的函数。将以下两个函数添加到游戏中的某个位置，在`Render2D`函数之上：

```cpp
void DrawUi()
{
 float startY = screenHeight - 50.0f;
 float x1 = 50.0f;
 float x2 = screenWidth / 2.0f - 50.0f;
 float x3 = screenWidth - 250.0f;

 char scoreText[50];
 char speedText[50];
 char missionTimeText[50];

 sprintf_s(scoreText, 50, "Score: %i", score);
 sprintf_s(speedText, 50, "Speed: %i", speed);
 sprintf_s(missionTimeText, 50, "Time: %f", missionTime / 100.0f);
 DrawText(scoreText, x1, startY, 0.0f, 1.0f, 0.0f);
 DrawText(speedText, x2, startY, 0.0f, 1.0f, 0.0f);
 DrawText(missionTimeText, x3, startY, 0.0f, 1.0f, 0.0f);

}

void DrawStats()
{
 float startX = screenWidth - screenWidth / 2.5f;
 float startY = 275.0f;
 float spaceY = 30.0f;

 char asteroidsHitText[50];
 char maximumSpeedText[50];
 char scoreText[50];
 char missionTimeText[50];
 sprintf_s(asteroidsHitText, 50, "Asteroids Hit: %i", asteroidsHit);
 sprintf_s(maximumSpeedText, 50, "Maximum Speed: %i", maximumSpeed);
 sprintf_s(scoreText, 50, "Score: %i", score);
 sprintf_s(missionTimeText, 50, "Time: %f", missionTime / 100.0f);
 DrawText(asteroidsHitText, startX, startY, 0.0f, 1.0f, 0.0f);
 DrawText(maximumSpeedText, startX, startY + spaceY, 0.0f, 1.0f, 0.0f);
 DrawText(scoreText, startX, startY + spaceY * 2.0f, 0.0f, 1.0f, 0.0f);
 DrawText(missionTimeText, startX, startY + spaceY * 3.0f, 0.0f, 1.0f, 0.0f);
}
void DrawCredits()
{
 float startX = screenWidth - screenWidth / 2.5f;
 float startY = 300.0f;
 float spaceY = 30.0f;
 DrawText("Robert Madsen", startX, startY, 0.0f, 1.0f, 0.0f);
 DrawText("Author", startX, startY + spaceY, 0.0f, 1.0f, 0.0f);
}
```

这些函数与 RoboRacer2D 中相应的函数工作方式完全相同。首先，我们使用`sprintf_s`创建一个包含我们想要显示的文本的字符字符串。接下来，我们使用`glRasterPos2f`设置 2D 的渲染位置。然后，我们使用`glCallLists`实际渲染字体。在`DrawCredits`函数中，我们使用`DrawText`辅助函数来渲染文本。

将`CheckCollisions`修改为以下代码：

```cpp
void CheckCollisions()
{
 bool collision = false;
 for (int i = 0; i < asteroids.size(); i++)
 {
  Model* item = asteroids[i];
  collision = ship->CollidedWith(item);
  if (collision)
  {
   item->IsCollideable(false);
   score++;
   asteroidsHit++;
  }
 }
}
```

此代码更新得分和陨石统计数据。

## 定义纹理

现在，是时候加载我们所有的纹理了。将以下函数添加到游戏中：

```cpp
const bool LoadTextures()
{
 menuScreen = new Sprite(1);
 menuScreen->SetFrameSize(screenWidth, screenHeight);
 menuScreen->SetNumberOfFrames(1);
 menuScreen->AddTexture("resources/mainmenu.png", false);
 menuScreen->IsActive(true);
 menuScreen->IsVisible(true);
 menuScreen->SetPosition(0.0f, 0.0f);
 playButton = new Sprite(1);
 playButton->SetFrameSize(75.0f, 38.0f);
 playButton->SetNumberOfFrames(1);
 playButton->SetPosition(690.0f, 300.0f);
 playButton->AddTexture("resources/playButton.png");
 playButton->IsVisible(true);
 playButton->IsActive(false);
 inputManager->AddUiElement(playButton);
 creditsButton = new Sprite(1);
 creditsButton->SetFrameSize(75.0f, 38.0f);
 creditsButton->SetNumberOfFrames(1);
 creditsButton->SetPosition(690.0f, 350.0f);
 creditsButton->AddTexture("resources/creditsButton.png");
 creditsButton->IsVisible(true);
 creditsButton->IsActive(false);
 inputManager->AddUiElement(creditsButton);
 exitButton = new Sprite(1);
 exitButton->SetFrameSize(75.0f, 38.0f);
 exitButton->SetNumberOfFrames(1);
 exitButton->SetPosition(690.0f, 500.0f);
 exitButton->AddTexture("resources/exitButton.png");
 exitButton->IsVisible(true);
 exitButton->IsActive(false);
 inputManager->AddUiElement(exitButton);
 creditsScreen = new Sprite(1);
 creditsScreen->SetFrameSize(screenWidth, screenHeight);
 creditsScreen->SetNumberOfFrames(1);
 creditsScreen->AddTexture("resources/credits.png", false);
 creditsScreen->IsActive(true);
 creditsScreen->IsVisible(true);
 menuButton = new Sprite(1);
 menuButton->SetFrameSize(75.0f, 38.0f);
 menuButton->SetNumberOfFrames(1);
 menuButton->SetPosition(690.0f, 400.0f);
 menuButton->AddTexture("resources/menuButton.png");
 menuButton->IsVisible(true);
 menuButton->IsActive(false);
 inputManager->AddUiElement(menuButton);
 gameOverScreen = new Sprite(1);
 gameOverScreen->SetFrameSize(screenWidth, screenHeight);
 gameOverScreen->SetNumberOfFrames(1);
 gameOverScreen->AddTexture("resources/gameover.png", false);
 gameOverScreen->IsActive(true);
 gameOverScreen->IsVisible(true);
 replayButton = new Sprite(1);
 replayButton->SetFrameSize(75.0f, 38.0f);
 replayButton->SetNumberOfFrames(1);
 replayButton->SetPosition(690.0f, 400.0f);
 replayButton->AddTexture("resources/replayButton.png");
 replayButton->IsVisible(true);
 replayButton->IsActive(false);
 inputManager->AddUiElement(replayButton);
 return true;
}
```

这里没有什么新的！我们只是将所有的 2D 资产作为精灵加载到游戏中。以下是一些关于如何工作的提醒：

+   每个精灵都是从 PNG 文件加载的，指定了帧数。由于这些精灵都没有动画，它们都只有一个帧。

+   我们使用 2D 坐标定位每个精灵。

+   我们设置属性——可见意味着它可以被看到，而激活意味着它可以被点击。

+   如果对象打算是一个按钮，我们将其添加到 UI 系统中。

## 连接渲染、更新和游戏循环

现在我们终于加载了所有的 2D 资产，我们准备完成 `Render2D` 函数：

```cpp
void Render2D()
{
 Enable2D();
 switch (gameState)
 {
 case GameState::GS_Loading:
 {
  splashScreen->Render();
 }
 break;
 case GameState::GS_Menu:
 {
  menuScreen->Render();
  playButton->Render();
  creditsButton->Render();
  exitButton->Render();
 }
 break;
 case GameState::GS_Credits:
 {
  creditsScreen->Render();
  menuButton->Render();
  DrawCredits();
 }
 break;
 case GameState::GS_Running:
 {
  DrawUi();
 }
 break;
 case GameState::GS_Splash:
 {
  splashScreen->Render();
 }
  break;
 case GameState::GS_GameOver:
 {
  gameOverScreen->Render();
  DrawStats();
  menuButton->Render();
 }
 break;
 }
 Disable2D();
}
```

再次强调，这里没有什么是你之前没有见过的。我们只是在实现完整的状态引擎。

现在我们有了可点击的按钮，我们可以实现完整的 `ProcessInput` 函数。将以下行添加到 `switch` 语句中：

```cpp
 case Input::Command::CM_UI:
 {
  if (playButton->IsClicked())
  {
   playButton->IsClicked(false);
   exitButton->IsActive(false);
   playButton->IsActive(false);
   creditsButton->IsActive(false);
   gameState = GameState::GS_Running;
  }
  if (creditsButton->IsClicked())
  {
   creditsButton->IsClicked(false);
   exitButton->IsActive(false);
   playButton->IsActive(false);
   creditsButton->IsActive(false);
   gameState = GameState::GS_Credits;
  }
  if (menuButton->IsClicked())
  {
   menuButton->IsClicked(false);
   exitButton->IsActive(true);
   playButton->IsActive(true);
   menuButton->IsActive(false);
   switch (gameState)
   {
   case GameState::GS_Credits:
   {
    gameState = GameState::GS_Menu;
   }
   break;
   case GameState::GS_GameOver:
   {
    StartGame();
   }
   break;
   }
  }
  if (exitButton->IsClicked())
  {
   playButton->IsClicked(false);
   exitButton->IsActive(false);
   playButton->IsActive(false);
   creditsButton->IsActive(false);
   PostQuitMessage(0);
  }
 }
 break;
 }
```

是的，我们之前已经见过这些了。如果你还记得，`Input` 类为每个可点击的按钮分配了一个命令枚举。这段代码只是简单地处理命令，如果有任何命令，并根据刚刚点击的按钮设置状态。

我们现在实现完整的 `Update` 函数来处理我们新的状态机：

```cpp
void Update(const float p_deltaTime)
{
 switch (gameState)
 {
 case GameState::GS_Splash:
 case GameState::GS_Loading:
 {
  splashScreen->Update(p_deltaTime);
  splashDisplayTimer += p_deltaTime;
  if (splashDisplayTimer > splashDisplayThreshold)
  {
   gameState = GameState::GS_Menu;
  }
 }
 break;
 case GameState::GS_Menu:
 {
  menuScreen->Update(p_deltaTime);
  playButton->IsActive(true);
  creditsButton->IsActive(true);
  exitButton->IsActive(true);
  playButton->Update(p_deltaTime);
  creditsButton->Update(p_deltaTime);
  exitButton->Update(p_deltaTime);
  inputManager->Update(p_deltaTime);
  ProcessInput(p_deltaTime);
 }
 break;
 case GameState::GS_Credits:
 {
  creditsScreen->Update(p_deltaTime);
  menuButton->IsActive(true);
  menuButton->Update(p_deltaTime);
  inputManager->Update(p_deltaTime);
  ProcessInput(p_deltaTime);
 }
 break;
 case GameState::GS_Running:
 {
  inputManager->Update(p_deltaTime);
  ProcessInput(p_deltaTime);
  ship->Update(p_deltaTime);
  ship->SetVelocity(ship->GetVelocity() + ship->GetVelocity()*p_deltaTime/10.0f);
  speed = ship->GetVelocity() * 1000;
  if (maximumSpeed < speed)
  {
   maximumSpeed = speed;
  }
  missionTime = missionTime + p_deltaTime * 100.0f;
  CheckCollisions();
  if (ship->GetPosition().z > 10.0f)
  {
   gameState = GS_GameOver;
   menuButton->IsActive(true);
   gameOverScreen->IsActive(true);
  }
 }
 break;
 case GameState::GS_GameOver:
 {
  gameOverScreen->Update(p_deltaTime);
  replayButton->IsActive(true);
  replayButton->Update(p_deltaTime);
  exitButton->IsActive(true);
  exitButton->Update(p_deltaTime);
  inputManager->Update(p_deltaTime);
  ProcessInput(p_deltaTime);
 }
 break;
 }
}
```

最后，我们需要修改游戏循环，使其支持我们所有的新特性。移动到 `GameLoop` 函数并修改它，使其看起来像以下代码：

```cpp
void GameLoop(const float p_deltatTime)
{
 if (gameState == GameState::GS_Splash)
 {
  BuildFont();
  LoadTextures();
  gameState = GameState::GS_Loading;
 }
 Update(p_deltatTime);
 Render();
}
```

和往常一样，游戏循环调用 `Update` 和 `Render` 函数。我们添加了一个特殊案例来处理启动画面。如果我们处于 `GS_Splash` 游戏状态，那么我们就加载游戏的其他资源，并将游戏状态更改为 `GS_Loading`。

注意，之前提到的几个函数还没有创建！随着我们的继续，我们将添加对声音、字体和纹理的支持。

# 摘要

在本章中，我们讨论了大量的代码。本章的主要课程是学习如何同时渲染 2D 和 3D。然后我们添加了代码来加载所有 2D 资源作为精灵。我们还添加了渲染文本的能力，现在我们可以看到我们的得分、统计数据和信用。

我们为游戏实现了那个状态机，并将其连接到输入、更新、渲染和游戏循环系统中。这包括创建启动画面、加载资源、玩游戏和显示各种游戏屏幕的状态。

现在你已经拥有了一个完整的 3D 游戏。当然，你还可以用它做更多的事情。在下一章和最后一章中，我们将学习一些新技巧，其余的则由你决定！
