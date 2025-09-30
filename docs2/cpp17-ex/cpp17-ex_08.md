# 第八章：计算机走棋

在本章中，我们继续开发 Othello 和井字棋游戏。本章的新内容是计算机与人类对弈；而不是两个人类玩家，计算机与人类对弈。

本章我们将涵盖的主题包括：

+   博弈论推理。在这两个游戏中，人类或计算机可以首先走棋，我们为计算机添加了与人类对弈的代码。

+   在 Othello 中，对于每一步，我们扫描游戏网格并试图找到导致人类标记交换数量最多的走法。

+   在井字棋游戏中，我们试图找到游戏网格中能给我们带来最高行数标记的位置，或者如果人类即将在行中获得五个标记，我们必须放置计算机的标记在防止这种情况发生的位置。

+   随机数生成简介。如果计算机可以在几个等效的走法之间选择，它应随机选择其中一个走法。

+   我们继续使用 C++ 特性，如类、字段和方法。我们还继续使用 Qt 特性，如窗口和小部件。

# Othello

在本章的 Othello 应用程序中，我们重用了上一章的 `MainWindow` 和 `GameWidget` 类。

# OthelloWindow 类

`OthelloWindow` 类与上一章的对应类相当相似。然而，除了菜单和选项外，这个版本的窗口还包含子菜单。子菜单将通过在 `OthelloWindow.cpp` 文件中调用 `addAction` 方法来添加。

**OthelloWindow.h**

```cpp
#ifndef OTHELLOWINDOW_H 
#define OTHELLOWINDOW_H 

#include "..\MainWindow\MainWindow.h" 
#include "OthelloWidget.h" 

class OthelloWindow : public MainWindow { 
  Q_OBJECT 

  public: 
    OthelloWindow(QWidget *parentWidget = nullptr); 
    ~OthelloWindow(); 

    void closeEvent(QCloseEvent *eventPtr)
                   {m_othelloWidgetPtr->closeEvent(eventPtr);} 

  private: 
    OthelloWidget* m_othelloWidgetPtr; 
}; 

#endif // OTHELLOWINDOW_H 
```

`OthelloWindow.cpp` 文件包含 `OthelloWindow` 类的方法定义。

**OthelloWindow.cpp**

```cpp
#include "OthelloWidget.h" 
#include "OthelloWindow.h" 
#include <QtWidgets> 
```

窗口的标题已更改为 `Othello Advanced`：

```cpp
OthelloWindow::OthelloWindow(QWidget *parentWidget /*= nullptr*/) 
 :MainWindow(parentWidget) { 
  setWindowTitle(tr("Othello Advanced")); 
  resize(1000, 500); 

  m_othelloWidgetPtr = new OthelloWidget(this); 
  setCentralWidget(m_othelloWidgetPtr); 

  { QMenu* gameMenuPtr = menuBar()->addMenu(tr("&Game")); 
    connect(gameMenuPtr, SIGNAL(aboutToShow()), 
            this, SLOT(onMenuShow())); 
```

游戏菜单中有两个子菜单，`Computer Starts` 和 `Human Starts`：

```cpp
    { QMenu* computerStartsMenuPtr = 
        gameMenuPtr->addMenu(tr("&Computer Starts")); 
      connect(computerStartsMenuPtr, SIGNAL(aboutToShow()), 
              this, SLOT(onMenuShow())); 
```

`Computer Starts` 子菜单包含两个选项 `Computer Black` 和 `Computer White`：

```cpp
      addAction(computerStartsMenuPtr, tr("Computer &Black"), 
                SLOT(onComputerStartsBlack()), 0, 
                tr("Computer Black"), nullptr, 
                tr("Computer Black"), 
                LISTENER(isComputerStartsBlackEnabled)); 

      addAction(computerStartsMenuPtr, tr("Computer &White"), 
                SLOT(onComputerStartsWhite()), 0, 
                tr("Computer White"), nullptr, 
                tr("Computer White"), 
                LISTENER(isComputerStartsWhiteEnabled)); 
    } 
```

`Human Starts` 子菜单包含两个选项，`Human Black` 和 `Human White`：

```cpp
    { QMenu* humanStartsMenuPtr = 
        gameMenuPtr->addMenu(tr("&Human Starts")); 
      connect(humanStartsMenuPtr, SIGNAL(aboutToShow()), 
              this, SLOT(onMenuShow())); 

      addAction(humanStartsMenuPtr, tr("Human &Black"), 
                SLOT(onHumanStartsBlack()), 0, tr("Human Black"), 
                nullptr, tr("Human Black"), 
                LISTENER(isHumanStartsBlackEnabled)); 

      addAction(humanStartsMenuPtr, tr("Human &White"), 
                SLOT(onHumanStartsWhite()), 0, tr("Human White"), 
                nullptr, tr("Human White"), 
                LISTENER(isHumanStartsWhiteEnabled)); 
    } 

    gameMenuPtr->addSeparator(); 

    addAction(gameMenuPtr, tr("&Quit the Game"), 
              SLOT(onQuit()), 
              QKeySequence(Qt::CTRL + Qt::Key_Q), tr("Quit Game"), 
              nullptr, tr("Quit the Game"), 
              LISTENER(isQuitEnabled)); 

    addAction(gameMenuPtr, tr("E&xit"),i 
              SLOT(onExit()), QKeySequence::Quit); 
  } 
} 

OthelloWindow::~OthelloWindow() { 
  delete m_othelloWidgetPtr; 
} 
```

# OthelloWidget 类

`OthelloWidget` 类包含 Othello 的功能。它允许计算机与人类对弈：

**OthelloWidget.h**

```cpp
#ifndef OTHELLOWIDGET_H 
#define OTHELLOWIDGET_H 

#include "..\MainWindow\GameWidget.h" 

#define ROWS    8 
#define COLUMNS 8 

enum Mark {Empty = 0, Black, White}; 

class OthelloWidget : public GameWidget { 
  Q_OBJECT 

  public: 
    OthelloWidget(QWidget* parentWidget); 

    void mouseMark(int row, int column); 
    void drawMark(QPainter& painter, 
                  const QRect& markRect, int mark); 

  public slots: 
    DEFINE_LISTENER(OthelloWidget, isComputerStartsBlackEnabled); 
    DEFINE_LISTENER(OthelloWidget, isComputerStartsWhiteEnabled); 
    DEFINE_LISTENER(OthelloWidget, isHumanStartsBlackEnabled); 
    DEFINE_LISTENER(OthelloWidget, isHumanStartsWhiteEnabled); 

    void onComputerStartsBlack(); 
    void onComputerStartsWhite(); 
    void onHumanStartsBlack(); 
    void onHumanStartsWhite(); 

  private: 
    bool checkWinner(); 
    void turn(int row, int column, Mark mark); 
    void calculateComputerMove(); 
    void calculateTurns(int row, int column, Mark mark, 
                        QSet<QPair<int,int>>& totalSet, 
                        int& neighbours); 
    Mark m_humanMark, m_computerMark; 
}; 

#endif // OTHELLOWIDGET_H 
```

`OthelloWidget.cpp` 文件包含 `OthelloWidget` 类的方法定义：

**OthelloWidget.cpp**

```cpp
#include "OthelloWidget.h" 
#include "OthelloWindow.h" 

#include <QTime> 
#include <CTime> 
#include <CAssert> 
using namespace std; 

OthelloWidget::OthelloWidget(QWidget* parentWidget) 
 :GameWidget(ROWS, COLUMNS, parentWidget) { 
  // Empty. 
} 
```

在调用 `Computer Starts` 和 `Human Starts` 子菜单之前，会调用 `isComputerStartsBlackEnabled`、`isComputerStartsWhiteEnabled`、`isHumanStartsBlackEnabled` 和 `isHumanStartsWhiteEnabled` 方法。如果没有进行游戏，它们将变为可用状态：

```cpp
bool OthelloWidget::isComputerStartsBlackEnabled() { 
  return !isGameInProgress(); 
} 

bool OthelloWidget::isComputerStartsWhiteEnabled() { 
  return !isGameInProgress(); 
} 

bool OthelloWidget::isHumanStartsBlackEnabled() { 
  return !isGameInProgress(); 
} 

bool OthelloWidget::isHumanStartsWhiteEnabled() { 
  return !isGameInProgress(); 
} 
```

当用户选择 `Computer Starts` 子菜单中的一个选项时，会调用 `onComputerStartsBlack` 和 `onComputerStartsWhite` 方法。它们将计算机标记设置为黑色或白色，通过在游戏网格中间设置标记来开始游戏，并更新窗口：

```cpp
void OthelloWidget::onComputerStartsBlack() { 
  setGameInProgress(true); 
  set(ROWS / 2, COLUMNS / 2, m_computerMark = Black); 
  m_humanMark = White; 
  update(); 
} 

void OthelloWidget::onComputerStartsWhite() { 
  setGameInProgress(true); 
  set(ROWS / 2, COLUMNS / 2, m_computerMark = White); 
  m_humanMark = Black; 
  update(); 
} 
```

当用户在`Human Starts`子菜单中选择一个项目时，会调用`onHumanStartsBlack`和`onHumanStartsWhite`方法。它们将电脑标记设置为黑色或白色并更新窗口。它们在游戏网格中不设置任何标记。相反，人类将首先移动：

```cpp
void OthelloWidget::onHumanStartsBlack() { 
  setGameInProgress(true); 
  m_humanMark = Black; 
  m_computerMark = White; 
  update(); 
} 

void OthelloWidget::onHumanStartsWhite() { 
  setGameInProgress(true); 
  m_humanMark = White; 
  m_computerMark = Black; 
  update(); 
} 
```

当用户在游戏网格中点击一个空位置时，会调用`mouseMark`方法。我们首先设置下一个标记在位置上，然后根据移动翻转标记：

```cpp
void OthelloWidget::mouseMark(int row, int column) { 
  set(row, column, m_humanMark); 
  turn(row, column, m_humanMark); 
  update(); 
```

如果人类的移动没有使游戏网格变满，我们调用`calculateComputerMove`来设置电脑标记到该位置，从而翻转最大数量的相反标记。然后我们更新窗口并再次调用`checkWinner`以决定电脑的移动是否使游戏网格变满：

```cpp
  if (!checkWinner()) { 
    calculateComputerMove(); 
    update(); 
    checkWinner(); 
  } 
} 
```

当游戏网格中的某个位置需要重新绘制时，会调用`drawMark`方法。它以与上一章相同的方式绘制标记：

```cpp
void OthelloWidget::drawMark(QPainter& painter, 
                             const QRect& markRect, int mark) { 
  painter.setPen(Qt::black); 
  painter.drawRect(markRect); 
  painter.fillRect(markRect, Qt::lightGray); 

  switch (mark) { 
    case Black: 
      painter.setPen(Qt::black); 
      painter.setBrush(Qt::black); 
      painter.drawEllipse(markRect); 
      break; 

    case White: 
      painter.setPen(Qt::white); 
      painter.setBrush(Qt::white); 
      painter.drawEllipse(markRect); 
      break; 

    case Empty: 
      break; 
  } 
} 
```

本章的`checkWinner`方法与上一章的对应方法类似。它检查游戏网格是否已满。如果已满，则宣布获胜者，否则为平局：

```cpp
bool OthelloWidget::checkWinner() { 
  int blacks = 0, whites = 0, empties = 0; 

  for (int row = 0; row < ROWS; ++row) { 
    for (int column = 0; column < COLUMNS; ++column) { 
      switch (get(row, column)) { 
        case Black: 
          ++blacks; 
          break; 

        case White: 
          ++whites; 
          break; 

        case Empty: 
          ++empties; 
          break; 
      } 
    } 
  } 

  if (empties == 0) { 
    QMessageBox messageBox(QMessageBox::Information, 
                           tr("Victory"), QString()); 
    QString text; 

    if (blacks > whites) { 
      text.sprintf("The Winner: %s.", (m_computerMark == Black) 
                                      ? "Computer" : "Human"); 
    } 
    else if (whites > blacks) { 
      text.sprintf("The Winner: %s.", (m_computerMark == White) 
                                      ? "Computer" : "Human"); 
    } 
    else { 
      text.sprintf("A Draw."); 
    } 

    messageBox.setText(text); 
    messageBox.setStandardButtons(QMessageBox::Ok); 
    messageBox.exec(); 
    setGameInProgress(false); 
    clearGrid(); 
    update(); 

    return true; 
  } 

  return false; 
} 
```

`calculateComputerMove`方法计算电脑的移动，该移动生成最多的翻转相反标记。我们遍历电脑标记，并对每个标记调用`calculateTurns`以获取如果我们把标记放在那个位置，将翻转的相反标记的最大数量。对于每个标记，我们还获取邻居的数量，这在找不到任何标记翻转时很有价值。

`maxTurnSetSize`和`maxNeighbours`字段保存可翻转标记和邻居的最大数量；`maxTurnSetList`保存可翻转标记位置的最大集合列表，而`maxNeighboursList`保存邻居数量的最大列表：

```cpp
void OthelloWidget::calculateComputerMove() { 
  int maxTurnSetSize = 0, maxNeighbours = 0; 
  QList<QSet<QPair<int,int>>> maxTurnSetList; 
  QList<QPair<int,int>> maxNeighboursList; 
```

我们遍历游戏网格中的所有位置。对于每个空位置，我们获取如果我们把标记放在那个位置，将翻转的相反标记的数量。我们还获取相反邻居的数量：

```cpp
  for (int row = 0; row < ROWS; ++row) { 
    for (int column = 0; column < COLUMNS; ++column) { 
      if (get(row, column) == Empty) { 
        QSet<QPair<int,int>> turnSet; 
        int neighbours = 0; 
        calculateTurns(row, column, m_computerMark, 
                       turnSet, neighbours); 
        int turnSetSize = turnSet.size(); 
```

如果我们发现一个可翻转标记的集合大于当前的最大集合，我们将`maxTurnSetSize`字段设置为新的可翻转集合的大小，将当前位置插入集合中，清除`maxTurnSetList`（因为我们不希望保留其之前的较小集合），并添加新的集合。

我们添加当前集合是为了简化；将其添加到集合中比以其他方式存储它更容易：

```cpp
        if (turnSetSize > maxTurnSetSize) { 
          maxTurnSetSize = turnSetSize; 
          turnSet.insert(QPair<int,int>(row, column)); 
          maxTurnSetList.clear(); 
          maxTurnSetList.append(turnSet); 
        } 
```

如果新集合不为空且与最大集合大小相等，我们只需将其添加到`maxTurnSetList`：

```cpp
        else if ((turnSetSize > 0) && 
                 (turnSetSize == maxTurnSetSize)) { 
          turnSet.insert(QPair<int,int>(row, column)); 
          maxTurnSetList.append(turnSet); 
        } 
```

我们还检查当前位置的邻居数量。我们以与`turnable`集合案例相同的方式进行工作。如果邻居数量超过最大邻居数量，我们清除`maxNeighboursList`并添加新位置：

```cpp
        if (neighbours > maxNeighbours) { 
          maxNeighbours = neighbours; 
          maxNeighboursList.clear(); 
          maxNeighboursList.append(QPair<int,int>(row, column)); 
        } 
```

如果至少有一个邻居，并且邻居数量等于最大邻居数量，我们将它添加到`maxNeighboursList`列表中：

```cpp
        else if ((neighbours > 0) && 
                 (neighbours == maxNeighbours)) { 
          maxNeighboursList.append(QPair<int,int>(row, column)); 
        } 
      } 
    } 
  } 
```

如果至少有一个位置我们将转动至少一个相反标记，我们就选择它。如果有几个位置将转动相同数量的相反标记，我们就随机选择其中一个。我们使用 C 标准函数`srand`、`rand`和`time`来获取一个随机整数。

随机数生成算法接受一个起始值，然后生成一系列随机数。`srand`函数使用起始值初始化生成器，然后`rand`被反复调用以获取新的随机数。为了不在每次调用`srand`时使用相同的起始值（这将导致相同的随机数序列），我们使用调用`time`标准 C 函数的结果来调用`srand`，该函数返回自 1970 年 1 月 1 日以来的秒数。这样，随机数生成器为每场比赛初始化一个新的值，我们通过反复调用`rand`来获得新的随机数序列：

```cpp
  if (maxTurnSetSize > 0) { 
    srand(time(NULL)); 
    int index = rand() % maxTurnSetList.size(); 
    QSet<QPair<int,int>> maxTurnSet = maxTurnSetList[index]; 
```

当我们获得了要转动的位置集合后，我们遍历集合并将电脑标记设置到所有这些位置：

```cpp
    for (QPair<int,int> position : maxTurnSet) { 
      int row = position.first, column = position.second; 
      set(row, column, m_computerMark); 
    } 
  } 
```

如果没有位置会导致相反标记转动，我们就查看邻居。同样地，我们随机选择具有最大邻居数量的位置之一。请注意，我们不需要迭代任何集合；在这种情况下，我们只设置一个标记：

```cpp
  else { 
    assert(!maxNeighboursList.empty()); 
    srand(time(NULL)); 
    int index = rand() % maxNeighboursList.size(); 
    QPair<int,int> position = maxNeighboursList[index]; 
    int row = position.first, column = position.second; 
    set(row, column, m_computerMark); 
  } 
} 
```

当人类玩家移动时调用`turn`方法。它调用`calculateMark`以获取可转动相反标记的集合，然后遍历集合并在游戏网格中设置每个位置：

```cpp
void OthelloWidget::turn(int row, int column, Mark mark) { 
  QSet<QPair<int,int>> turnSet; 
  calculateMark(row, column, mark, turnSet); 

  for (QPair<int,int> pair : turnSet) { 
    int row = pair.first, column = pair.second; 
    set(row, column, mark); 
  } 
} 
```

`calculateTurns`方法计算给定位置的可转动相反标记的集合和邻居的数量：

```cpp
void OthelloWidget::calculateTurns(int row, int column, 
                  Mark playerMark,QSet<QPair<int,int>>& totalSet, 
                  int& neighbours) { 
```

`directionArray`中的每个整数对都指代根据罗盘上升的方向：

```cpp
  QPair<int,int> directionArray[] = 
    {QPair<int,int>(-1, 0),   // North 
     QPair<int,int>(-1, 1),   // Northeast 
     QPair<int,int>(0, 1),    // East 
     QPair<int,int>(1, 1),    // Southeast 
     QPair<int,int>(1, 0),    // South 
     QPair<int,int>(1, -1),   // Southwest 
     QPair<int,int>(0, -1),   // West 
     QPair<int,int>(-1, -1)}; // Northwest 
```

数组的大小可以通过将其总大小（以字节为单位）除以第一个值的尺寸来决定：

```cpp
  int arraySize = 
    (sizeof directionArray) / (sizeof directionArray[0]); 

  neighbours = 0; 
  int opponentMark = (playerMark == Black) ? White : Black; 
```

我们遍历方向，并对每个方向，只要我们找到对手的标记就继续移动：

```cpp
  for (int index = 0; index < arraySize; ++index) { 
    QPair<int,int> pair = directionArray[index]; 
```

`row`和`column`字段在遍历方向时保持当前行和列：

```cpp
    int rowStep = pair.first, columnStep = pair.second, 
        currRow = row, currColumn = column; 
```

首先，我们检查我们是否在最近的位置有一个对手标记的邻居。如果我们没有到达游戏网格的边界之一，并且在该位置有一个对手标记，我们就增加`neighbours`：

```cpp
    if (((row + rowStep) >= 0) && ((row + rowStep) < ROWS) && 
        ((column + rowStep) >= 0) && 
        ((column + columnStep) < COLUMNS) && 
        (get(row + rowStep, column + rowStep) == opponentMark)) { 
      ++neighbours; 
    } 

```

我们在迭代过程中找到的标记收集到`directionSet`中：

```cpp
    QSet<QPair<int,int>> directionSet; 

    while (true) { 
      currRow += rowStep; 
      currColumn += columnStep; 
```

如果我们到达游戏网格的边界之一，或者如果我们找到一个空位，我们就会中断迭代：

```cpp
      if ((currRow < 0) || (currRow == ROWS) || 
          (currColumn < 0) || (currColumn == COLUMNS) || 
          (get(currRow, currColumn) == Empty)) { 
        break; 
      } 
```

如果我们找到了玩家的标记，我们将`directionSet`添加到总集合中，并中断迭代：

```cpp
      else if (get(currRow, currColumn) == playerMark) { 
        totalSet += directionSet; 
        break; 
      } 
```

如果我们确实找到了玩家的标记或空位，我们就找到了对手的标记，并将它的位置添加到方向集合中：

```cpp
      else { 
        directionSet.insert(QPair<int,int>(row, column)); 
      } 
    } 
  } 
} 
```

# 主函数

和往常一样，`main`函数创建一个应用程序，显示窗口，并执行应用程序直到用户关闭窗口或选择退出菜单项。

**Main.cpp**

```cpp
#include "OthelloWidget.h" 
#include "OthelloWindow.h" 
#include <QApplication> 

int main(int argc, char *argv[]) { 
  QApplication application(argc, argv); 
  OthelloWindow othelloWindow; 
  othelloWindow.show(); 
  return application.exec(); 
} 
```

# 奥运十字

本章的 Noughts and Crosses 应用程序基于上一章的版本。不同之处在于，在这个版本中，计算机与人类进行对战：

# `NaCWindow`类

`NaCWindow`类与上一节中的`OthelloWindow`类类似（NaC 是 Noughts and Crosses 的缩写）。它向游戏菜单中添加了两个子菜单，其中计算机或人类先手，并选择零或叉：

**NaCWindow.h**

```cpp
#ifndef NACWINDOW_H 
#define NACWINDOW_H 

#include "..\MainWindow\MainWindow.h" 
#include "NaCWidget.h" 

class NaCWindow : public MainWindow { 
  Q_OBJECT 

  public: 
    NaCWindow(QWidget *parentWidget = nullptr); 
    ~NaCWindow(); 

  public: 
    void closeEvent(QCloseEvent *eventPtr)
                   {m_nacWidgetPtr->closeEvent(eventPtr);} 

  private: 
    NaCWidget* m_nacWidgetPtr; 
}; 

#endif // NACWINDOW_H 
```

`NaCWindow.cpp`文件包含了`NaCWindow`类的方法定义：

**NaCWindow.cpp**

```cpp
#include "NaCWindow.h" 
#include <QtWidgets> 
```

标题已更改为`Noughts and Crosses Advanced`：

```cpp
NaCWindow::NaCWindow(QWidget *parentWidget /*= nullptr*/) 
 :MainWindow(parentWidget) { 
  setWindowTitle(tr("Noughts and Crosses Advanced")); 
  resize(1000, 500); 

  m_nacWidgetPtr = new NaCWidget(this); 
  setCentralWidget(m_nacWidgetPtr); 

  { QMenu* gameMenuPtr = menuBar()->addMenu(tr("&Game")); 
    connect(gameMenuPtr, SIGNAL(aboutToShow()), 
            this, SLOT(onMenuShow())); 

    { QMenu* computerStartsMenuPtr = 
        gameMenuPtr->addMenu(tr("&Computer Starts")); 
      connect(computerStartsMenuPtr, SIGNAL(aboutToShow()), 
              this, SLOT(onMenuShow())); 

      addAction(computerStartsMenuPtr, tr("Computer &Nought"), 
                SLOT(onComputerStartsNought()), 0, 
                tr("Computer Nought"), nullptr, 
                tr("Computer Nought"), 
                LISTENER(isComputerStartsNoughtEnabled)); 

        addAction(computerStartsMenuPtr, tr("Computer &Cross"), 
                  SLOT(onComputerStartsCross()), 0, 
                  tr("Computer Cross"), nullptr, 
                  tr("Computer Cross"), 
                  LISTENER(isComputerStartsCrossEnabled)); 
    } 

    { QMenu* humanStartsMenuPtr = 
        gameMenuPtr->addMenu(tr("&Human Starts")); 
      connect(humanStartsMenuPtr, SIGNAL(aboutToShow()), 
              this, SLOT(onMenuShow())); 

      addAction(humanStartsMenuPtr, tr("Human &Nought"), 
                SLOT(onHumanNought()), 0, tr("Human Nought"), 
                nullptr, tr("Human Nought"), 
                LISTENER(isHumanNoughtEnabled)); 

      addAction(humanStartsMenuPtr, tr("Human &Cross"), 
                SLOT(onHumanCross()), 0, tr("Human Cross"), 
                nullptr, tr("Human Cross"), 
                LISTENER(isHumanCrossEnabled)); 
    } 

    gameMenuPtr->addSeparator(); 

    addAction(gameMenuPtr, tr("&Quit the Game"), 
              SLOT(onQuit()), 
              QKeySequence(Qt::CTRL + Qt::Key_Q), tr("Quit Game"), 
              nullptr, tr("Quit the Game"), 
              LISTENER(isQuitEnabled)); 

    addAction(gameMenuPtr, tr("E&xit"), 
              SLOT(onExit()), QKeySequence::Quit); 
  } 
} 

NaCWindow::~NaCWindow() { 
  delete m_nacWidgetPtr; 
} 
```

# `NaCWidget`类

与上一章中的版本相比，`NaCWidget`类得到了改进。它包含了计算机对抗人类时使用的`calculateComputerMove`和`calculateMarkValue`方法：

**NaCWidget.h**

```cpp
#ifndef NACWIDGET_H 
#define NACWIDGET_H 

#include "..\MainWindow\GameWidget.h" 

#define ROWS    26 
#define COLUMNS 26 

enum Mark {Empty = 0, Nought, Cross}; 

class NaCWidget : public GameWidget { 
  Q_OBJECT 

  public: 
    NaCWidget(QWidget* parentWidget); 

    void mouseMark(int row, int column); 
    void drawMark(QPainter& painter, 
                  const QRect& markRect, int mark); 

  public slots: 
    DEFINE_LISTENER(NaCWidget, isComputerStartsNoughtEnabled); 
    DEFINE_LISTENER(NaCWidget, isComputerStartsCrossEnabled); 
    DEFINE_LISTENER(NaCWidget, isHumanStartsNoughtEnabled); 
    DEFINE_LISTENER(NaCWidget, isHumanStartsCrossEnabled); 

    void onComputerStartsNought(); 
    void onComputerStartsCross(); 
    void onHumanStartsNought(); 
    void onHumanStartsCross(); 

  private: 
    bool checkWinner(int row, int column, Mark mark); 
    int countMarks(int row, int column, int rowStep, 
                   int columnStep, Mark mark); 
    void calculateComputerMove(int& row, int &column); 
    double calculateMarkValue(int row, int column, Mark mark); 

    Mark m_humanMark, m_computerMark; 
}; 

#endif // NACWIDGET_H 
```

`NaCWidget.cpp`文件包含了`NaCWidget`类的方法定义：

**NaCWidget.cpp**

```cpp
#include "NaCWidget.h" 
#include <CTime> 
#include <CAssert> 

NaCWidget::NaCWidget(QWidget* parentWidget) 
 :GameWidget(ROWS, COLUMNS, parentWidget) { 
  // Empty. 
} 
```

`isComputerStartsNoughtEnabled`、`isComputerStartsCrossEnabled`、`isHumanStartsNoughtEnabled`和`isHumanStartsCrossEnabled`方法决定是否启用`计算机零`、`计算机叉`、`人类零`和`人类叉`菜单项。在没有进行游戏时，它们都是启用的：

```cpp
bool NaCWidget::isComputerStartsNoughtEnabled() { 
  return !isGameInProgress(); 
} 

bool NaCWidget::isComputerStartsCrossEnabled() { 
  return !isGameInProgress(); 
} 

bool NaCWidget::isHumanStartsNoughtEnabled() { 
  return !isGameInProgress(); 
} 

bool NaCWidget::isHumanStartsCrossEnabled() { 
  return !isGameInProgress(); 
} 
```

当用户选择`计算机零`、`计算机叉`、`人类零`和`人类叉`菜单项时，会调用`onComputerStartsNought`、`onComputerStartsCross`、`onHumanStartsNought`和`onHumanStartsCross`。它们设置游戏状态，将计算机和人类的标记设置为零和叉，并更新窗口。如果计算机先手，它将被放置在游戏网格的中间，以便尽可能有效地使用游戏网格：

```cpp
void NaCWidget::onComputerStartsNought() { 
  setGameInProgress(true); 
  set(ROWS /2, COLUMNS / 2, m_computerMark = Nought); 
  m_humanMark = Cross; 
  update(); 
} 

void NaCWidget::onComputerStartsCross() { 
  setGameInProgress(true); 
  set(ROWS /2, COLUMNS / 2, m_computerMark = Cross); 
  m_humanMark = Nought; 
  update(); 
} 

void NaCWidget::onHumanStartsNought() { 
  setGameInProgress(true); 
  m_computerMark = Cross; 
  m_humanMark = Nought; 
  update(); 
} 

void NaCWidget::onHumanStartsCross() { 
  setGameInProgress(true); 
  m_computerMark = Nought; 
  m_humanMark = Cross; 
  update(); 
} 
```

当人类玩家在游戏网格中点击一个空位时，会调用`mouseMark`方法。我们首先将标记设置到该位置并更新窗口：

```cpp
void NaCWidget::mouseMark(int row, int column) { 
  set(row, column, m_humanMark); 
  update(); 
```

如果人类的移动没有让他们赢得游戏，我们计算计算机的下一步移动，设置位置，检查移动是否让计算机赢得了游戏，并更新窗口：

```cpp
  if (!checkWinner(row, column, m_humanMark)) { 
    calculateComputerMove(row, column); 
    set(row, column, m_computerMark); 
    checkWinner(row, column, m_computerMark); 
    update(); 
  } 
} 
```

当需要重新绘制位置时，会调用`drawMark`方法。它与上一章中的对应方法类似。它绘制一个零或一个叉：

```cpp
void NaCWidget::drawMark(QPainter& painter, 
                         const QRect& markRect, int mark) { 
  painter.setPen(Qt::black); 

  switch (mark) { 
    case Nought: 
      painter.drawEllipse(markRect); 
      break; 

    case Cross: 
      painter.drawLine(markRect.topLeft(), 
                       markRect.bottomRight()); 
      painter.drawLine(markRect.topRight(), 
                       markRect.bottomLeft()); 
      break; 

    case Empty: 
      break; 
  } 
} 
```

`checkWinner`方法与上一章中的对应方法类似。它决定最新的移动是否导致了五连珠。如果是，则宣布获胜者：

```cpp
bool NaCWidget::checkWinner(int row, int column, Mark mark) { 
  if ((countMarks(row, column, -1, 0, mark) >= 5) || 
      (countMarks(row, column, 0, -1, mark) >= 5) || 
      (countMarks(row, column, -1, 1, mark) >= 5) || 
      (countMarks(row, column, 1, 1, mark) >= 5)) { 
    QMessageBox messageBox(QMessageBox::Information, 
                           tr("Victory"), QString()); 
    QString text; 
    text.sprintf("The Winner: %s.", 
                 (mark == m_computerMark) ? "Computer" : "Human"); 

    messageBox.setText(text); 
    messageBox.setStandardButtons(QMessageBox::Ok); 
    messageBox.exec(); 
    setGameInProgress(false); 
    clearGrid(); 
    update(); 
    return true; 
  } 

  return false; 
} 
```

`countMarks`方法用于计算一行中的标记数量。与上一章中的对应方法相比，它已经得到了改进。在这个版本中，我们还计算了移动可以导致的一行中可能出现的最高标记数量。由于`countMarks`方法是由`calculateComputerMove`调用的，我们需要知道移动可能导致的一行中的标记数量：

```cpp
double NaCWidget::countMarks(int row, int column, int rowStep, 
                        int columnStep, Mark mark) { 
```

`markCount`字段保存了如果我们把标记放在给定位置，我们将得到的连续标记的数量；`freeCount`保存了如果我们继续在那个行添加标记，我们可能得到的连续标记的数量。原因是电脑不会在无法形成连续五个标记的行上添加标记：

```cpp
  double markCount = 0; 
  int freeCount = 0; 
```

我们以给定方向遍历游戏网格：

```cpp
  { bool marked = true; 
    int currentRow = row, currentColumn = column; 

    while ((currentRow >= 0) && (currentRow < ROWS) && 
           (currentColumn >= 0) && (currentColumn < COLUMNS)) { 
```

只要我们找到标记，我们就增加`markCount`和`freeCount`：

```cpp
      if (get(currentRow, currentColumn) == mark) { 
        if (marked) { 
          ++markCount; 
        } 

        ++freeCount; 
      } 
```

如果我们找到一个空位，我们将`0.4`（因为空闲行比封闭行好）加到`markCount`上，并继续增加`freeCount`：

```cpp
      else if (get(currentRow, currentColumn) == Empty) { 
        if (marked) { 
          markCount += 0.4; 
        } 

        marked = false; 
        ++freeCount; 
      } 
```

如果我们既没有找到电脑的标记也没有找到空位，那么我们一定找到了人类的标记，然后我们中断迭代：

```cpp
      else { 
        break; 
      } 
```

在每次迭代的末尾，我们将行和列的步数添加到当前行和列：

```cpp
      currentRow += rowStep; 
      currentColumn += columnStep; 
    } 
  } 
```

我们在相反方向上执行类似的迭代。唯一的区别是在每次迭代的末尾，我们减去行和列的步数，而不是将它们加到上面：

```cpp
  { bool marked = true; 
    int currentRow = row + rowStep, 
        currentColumn = column + columnStep; 

    while ((currentRow >= 0) && (currentRow < ROWS) && 
           (currentColumn >= 0) && (currentColumn < COLUMNS)) { 
      if (get(currentRow, currentColumn) == mark) { 
        if (marked) { 
          ++markCount; 
        } 
      } 
      else if (get(currentRow, currentColumn) == Empty) { 
        if (marked) { 
          markCount += 0.4; 
        } 

        marked = false; 
        ++freeCount; 
      } 
      else { 
        break; 
      } 

      currentRow -= rowStep; 
      currentColumn -= columnStep; 
    } 
  } 
```

如果空闲计数至少为五，我们返回标记计数。如果它少于五，我们返回零，因为我们不能在这个方向上获得连续五个标记：

```cpp
  return (freeCount >= 5) ? markCount : 0; 
} 
```

`calculateComputerMove`方法计算导致最大连续标记数量的电脑移动。我们计算电脑和人类的行，因为我们可能面临需要阻止人类获胜而不是最大化电脑获胜机会的情况。

`maxComputerValue`和`maxHumanValue`字段保存了我们迄今为止找到的连续标记的最大数量。`maxComputerList`和`maxHumanList`保存了导致电脑和人类连续标记最大数量的位置：

```cpp
void NaCWidget::calculateComputerMove(int& maxRow,int &maxColumn){ 
  double maxComputerValue = 0, maxHumanValue = 0; 
  QList<QPair<int,int>> maxComputerList, maxHumanList; 
```

我们遍历游戏网格。对于每个空位，我们尝试设置电脑和人类的标记，并查看这将导致多少个连续标记：

```cpp
  for (int row = 0; row < ROWS; ++row) { 
    for (int column = 0; column < COLUMNS; ++column)  { 
      if (get(row, column) == Empty) { 
        set(row, column, m_computerMark); 
```

我们获得电脑和人类标记的连续标记的最大数量。如果它大于之前的最大数量，我们清除列表并将位置添加到列表中：

```cpp
        { double computerValue = 
            calculateMarkValue(row, column, m_computerMark); 

          if (computerValue > maxComputerValue) { 
            maxComputerValue = computerValue; 
            maxComputerList.clear(); 
            maxComputerList.append(QPair<int,int>(row, column)); 
          } 
```

如果新的连续标记数量大于零或等于最大数量，我们只需添加位置：

```cpp
          else if ((computerValue > 0) && 
                   (computerValue == maxComputerValue)) { 
            maxComputerList.append(QPair<int,int>(row, column)); 
          } 
        } 
```

我们对电脑标记和人类标记做同样的处理：

```cpp
        set(row, column, m_humanMark); 

        { double humanValue = 
            calculateMarkValue(row, column, m_humanMark); 

          if (humanValue > maxHumanValue) { 
            maxHumanValue = humanValue; 
            maxHumanList.clear(); 
            maxHumanList.append(QPair<int,int>(row, column)); 
          } 
          else if ((humanValue > 0) && 
                   (humanValue == maxHumanValue)) { 
            maxHumanList.append(QPair<int,int>(row, column)); 
          } 
        } 
```

最后，我们将位置重置为空值：

```cpp
        set(row, column, Empty); 
      } 
    } 
  } 
```

电脑或人类必须至少有一个在行的位置：

```cpp
  assert(!maxComputerList.empty() && !maxHumanList.empty()); 
```

如果电脑的值至少为两个且大于人类值，或者如果人类值小于四个，我们将随机选择电脑的最大移动之一：

```cpp
  if ((maxComputerValue >= 2) && 
      ((maxComputerValue >= maxHumanValue) || 
       (maxHumanValue < 3.8))) { 
    srand(time(NULL)); 
    QPair<int,int> pair = 
      maxComputerList[rand() % maxComputerList.size()]; 
    maxRow = pair.first; 
    maxColumn = pair.second; 
  } 
```

然而，如果电脑无法连续放置至少两个标记，或者如果人类即将连续放置五个标记，我们将随机选择人类最大移动中的一个：

```cpp
  else { 
    srand(time(NULL)); 
    QPair<int,int> pair = 
      maxHumanList[rand() % maxHumanList.size()]; 
    maxRow = pair.first; 
    maxColumn = pair.second; 
  } 
} 
```

`calculateMarkValue`方法通过计算其四个方向中的较大值来计算给定位置可能导致的连续标记的最大数量：

```cpp
double NaCWidget::calculateMarkValue(int row, int column, 
                                     Mark mark) { 
  return qMax(qMax(countMarks(row, column, -1, 0, mark), 
                   countMarks(row, column, 0, -1, mark)), 
              qMax(countMarks(row, column, -1, 1, mark), 
                   countMarks(row, column, 1, 1, mark))); 
} 
```

# 主函数

最后，`main`函数在 Qt 应用程序中总是这样工作：

**Main.cpp**

```cpp
#include "NaCWidget.h" 
#include "NaCWindow.h" 
#include <QApplication> 

int main(int argc, char *argv[]) { 
  QApplication application(argc, argv); 
  NaCWindow mainWindow; 
  mainWindow.show(); 
  return application.exec(); 
} 

```

# 摘要

在本章中，我们开发了上一章游戏的高级版本。在奥赛罗和井字棋游戏中，我们添加了代码，使计算机能够与人类对弈。在奥赛罗游戏中，我们寻找游戏网格中能够导致对手标记数量变化最大的位置。在井字棋游戏中，我们寻找能够使计算机获得尽可能多的连续标记的走法，最好是连续五条。然而，我们还得寻找对手可能的连续标记数量，并阻止他们的下一步棋，如果这会导致胜利。现在，我建议你在继续到下一章之前，先坐下来和计算机玩几局，享受一下。

在下一章中，我们将开始开发一种**领域特定语言**（**DSL**），这是一种针对特定领域设计的语言。我们将开发一个 DSL 来指定图形对象的绘制，例如线条、矩形、椭圆和文本，以及颜色、字体、笔和刷的样式以及对齐设置。我们还将编写一个查看器来显示图形对象。
