# The Computer Plays

In this chapter, we continue to work on the Othello and Noughts and Crosses games. The new part of this chapter is the computer playing against the human; instead of two human players, the computer plays against a human.

Topics we will cover in this chapter include:

*   Game-theory reasoning. In both games, the human or the computer can make the first move, and we add code for the computer to play against the human.
*   In Othello, for each move, we scan the game grid and try to find the move that causes the highest number of the human's marks to be swapped.
*   In Noughts and Crosses, we try to find the position in the game grid that gives us the highest number of marks in a row, or, if the human is about to get five in row, we have to place the computer’s mark in a position that prevents that.
*   An introduction to random number generation. If the computer can choose between several equivalent moves, it shall randomly select one of the moves.
*   We continue to use C++ features such as classes, fields, and methods. We also continue to use Qt features such as windows and widgets.

# Othello

In the Othello application of this chapter, we reuse the `MainWindow` and `GameWidget` classes of the previous chapter.

# The OthelloWindow class

The `OthelloWindow` class is rather similar to its counterpart in the previous chapter. However, in addition to the menus and items, the window of this version also holds submenus. The submenus will be added by calling the `addAction` method in the `OthelloWindow.cpp` file.

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

The `OthelloWindow.cpp` file holds the definitions of the methods of the `OthelloWindow` class.

**OthelloWindow.cpp**

```cpp
#include "OthelloWidget.h" 
#include "OthelloWindow.h" 
#include <QtWidgets> 
```

The title of the window has been changed to `Othello Advanced`:

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

There are two submenus of the Game menu, `Computer Starts` and `Human Starts`:

```cpp
    { QMenu* computerStartsMenuPtr = 
        gameMenuPtr->addMenu(tr("&Computer Starts")); 
      connect(computerStartsMenuPtr, SIGNAL(aboutToShow()), 
              this, SLOT(onMenuShow())); 
```

The `Computer Starts` submenu holds the two items `Computer Black` and `Computer White`:

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

The `Human Starts` submenu holds two items, `Human Black` and `Human White`:

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

# The OthelloWidget Class

The `OthelloWidget` class holds the functionality of Othello. It allows the computer to play against a human:

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

The `OthelloWidget.cpp` file holds the definitions of the methods of the `OthelloWidget` class:

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

The `isComputerStartsBlackEnabled`, `isComputerStartsWhiteEnabled`, `isHumanStartsBlackEnabled`, and `isHumanStartsWhiteEnabled` methods are called before the `Computer Starts` and `Human Starts` submenus. They become enabled if there is no game in progress:

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

The `onComputerStartsBlack` and `onComputerStartsWhite` methods are called when the user selects one of the items of the `Computer Starts` submenu. They set the computer mark to black or white, start the game by setting the mark in the middle of the game grid, and update the window:

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

The `onHumanStartsBlack` and `onHumanStartsWhite` methods are called when the user selects one of the items of the `Human Starts` submenu. They set the computer mark to black or white and update the window. They do not set any mark in the game grid. Instead, the human is to make the first move:

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

The `mouseMark` method is called when the user clicks one empty position in the game grid. We start by setting the next mark at the position, and turn the marks as a result of the move:

```cpp
void OthelloWidget::mouseMark(int row, int column) { 
  set(row, column, m_humanMark); 
  turn(row, column, m_humanMark); 
  update(); 
```

If the human's move did not cause the game grid to become full, we call to `calculateComputerMove` to set the computer mark to the position, causing the maximum number of opposite marks to be turned. We then update the window and call `checkWinner` again to decide if the computer move caused the game grid to become full:

```cpp
  if (!checkWinner()) { 
    calculateComputerMove(); 
    update(); 
    checkWinner(); 
  } 
} 
```

The `drawMark` method is called when a position in the game grid needs to be repainted. It draws the mark in the same way as in the previous chapter:

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

The `checkWinner` method of this chapter is also similar to its counterpart in the previous chapter. It checks whether the game grid is full. If it is full, the winner is announced, or else it is a draw:

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

The `calculateComputerMove` method calculates the move of the computer that generates the highest number of turned opposite marks. We iterate through the computer marks and, for each mark, call `calculateTurns` to obtain the maximum number of opposite marks that would be turned if we placed the marks at that position. For each mark, we also obtain the number of neighbours, which is valuable if we do not find any marks to turn.

The `maxTurnSetSize` and `maxNeighbours` fields hold the maximum number of turnable marks and neighbours; `maxTurnSetList` holds a list of the maximum sets of positions of turnable marks, and `maxNeighboursList` holds a list of the maximum number of neighbours:

```cpp
void OthelloWidget::calculateComputerMove() { 
  int maxTurnSetSize = 0, maxNeighbours = 0; 
  QList<QSet<QPair<int,int>>> maxTurnSetList; 
  QList<QPair<int,int>> maxNeighboursList; 
```

We iterate through all the positions in the game grid. For each empty position, we obtain the number of opposite marks to be turned if we were to place our mark in that position. We also obtain the number of opposite neighbours:

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

If we find a set of turnable marks that is larger than the current maximum set, we set the `maxTurnSetSize` field to the size of the new turnable set, insert the current position in the set, clear `maxTurnSetList` (since we do not want its previous smaller sets), and add the new set.

We add the current set for the sake of simplicity; it is easier to add it to the set than to store it in any other way:

```cpp
        if (turnSetSize > maxTurnSetSize) { 
          maxTurnSetSize = turnSetSize; 
          turnSet.insert(QPair<int,int>(row, column)); 
          maxTurnSetList.clear(); 
          maxTurnSetList.append(turnSet); 
        } 
```

If the new set is not empty and of equal size to the maximum set, then we simply add it to `maxTurnSetList`:

```cpp
        else if ((turnSetSize > 0) && 
                 (turnSetSize == maxTurnSetSize)) { 
          turnSet.insert(QPair<int,int>(row, column)); 
          maxTurnSetList.append(turnSet); 
        } 
```

We also check the number of neighbours of the current position. We work in the same way as in the `turnable` set case. If the neighbours are more than the maximum number of neighbours, we clear `maxNeighboursList` and add the new position:

```cpp
        if (neighbours > maxNeighbours) { 
          maxNeighbours = neighbours; 
          maxNeighboursList.clear(); 
          maxNeighboursList.append(QPair<int,int>(row, column)); 
        } 
```

If there is at least one neighbour, and the neighbours is equal to the maximum number of neighbours, we add it to the `maxNeighboursList` list:

```cpp
        else if ((neighbours > 0) && 
                 (neighbours == maxNeighbours)) { 
          maxNeighboursList.append(QPair<int,int>(row, column)); 
        } 
      } 
    } 
  } 
```

If there is at least one position where we will turn at least one opposite mark, we choose it. If there are several positions that will turn the same amount of opposite marks, we randomly select one of them. We use the C standard functions `srand`, `rand`, and `time` to obtain a random integer number. 

The random number generator algorithm takes a start value and then generates a sequence of random numbers. The `srand` function initializes the generator with a start value, and then `rand` is called repeatedly in order to obtain new random numbers. In order to not call `srand` with the same start value every time (which would result in the same random number sequence), we call `srand` with the result of a call to the `time` standard C function, which returns the number of seconds since January 1, 1970\. In this way, the random number generator is initialized with a new value for each game, and we obtain a new sequence of random numbers by repeatedly calling `rand`:

```cpp
  if (maxTurnSetSize > 0) { 
    srand(time(NULL)); 
    int index = rand() % maxTurnSetList.size(); 
    QSet<QPair<int,int>> maxTurnSet = maxTurnSetList[index]; 
```

When we have obtained the set of positions to be turned, we iterate through the set and set the computer mark to all its positions:

```cpp
    for (QPair<int,int> position : maxTurnSet) { 
      int row = position.first, column = position.second; 
      set(row, column, m_computerMark); 
    } 
  } 
```

If there is no position that would cause opposite marks to be turned, we look at the neighbours instead. In the same way, we randomly select one of the positions with the maximum number of neighbours. Note that we do not need to iterate through any set; in this case, we only set one mark:

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

The `turn` method is called when the human has made a move. It calls `calculateMark` to obtain a set of turnable opposite marks, and then iterates through the set and sets each position in the game grid:

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

The `calculateTurns` method calculates the set of turnable opposite marks and number of neighbours of the given position:

```cpp
void OthelloWidget::calculateTurns(int row, int column, 
                  Mark playerMark,QSet<QPair<int,int>>& totalSet, 
                  int& neighbours) { 
```

Each integer pair in `directionArray` refers to a direction in accordance with the compass rising:

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

The size of an array can be decided by dividing its total size (in bytes) by the size of its first value:

```cpp
  int arraySize = 
    (sizeof directionArray) / (sizeof directionArray[0]); 

  neighbours = 0; 
  int opponentMark = (playerMark == Black) ? White : Black; 
```

We iterate through the directions and, for each direction, keep moving as long as we find the mark of the opponent:

```cpp
  for (int index = 0; index < arraySize; ++index) { 
    QPair<int,int> pair = directionArray[index]; 
```

The `row` and `column` fields hold the current row and column as long as we iterate through a direction:

```cpp
    int rowStep = pair.first, columnStep = pair.second, 
        currRow = row, currColumn = column; 
```

First, we check if we have a neighbor of the opponent mark in the closest position. If we have not reached one of the borders of the game grid, and if there is an opponent mark in the position, we increase `neighbours`:

```cpp
    if (((row + rowStep) >= 0) && ((row + rowStep) < ROWS) && 
        ((column + rowStep) >= 0) && 
        ((column + columnStep) < COLUMNS) && 
        (get(row + rowStep, column + rowStep) == opponentMark)) { 
      ++neighbours; 
    } 

```

We gather the marks we find during the iteration in `directionSet`:

```cpp
    QSet<QPair<int,int>> directionSet; 

    while (true) { 
      currRow += rowStep; 
      currColumn += columnStep; 
```

If we reach one of the borders of the game grid, or if we find an empty position, we break the iteration:

```cpp
      if ((currRow < 0) || (currRow == ROWS) || 
          (currColumn < 0) || (currColumn == COLUMNS) || 
          (get(currRow, currColumn) == Empty)) { 
        break; 
      } 
```

If we find the player's mark, we add the `directionSet` to the total set and break the iterations:

```cpp
      else if (get(currRow, currColumn) == playerMark) { 
        totalSet += directionSet; 
        break; 
      } 
```

If we do find the player's mark or an empty position, we have found the opponent's mark, and we add its position to the direction set:

```cpp
      else { 
        directionSet.insert(QPair<int,int>(row, column)); 
      } 
    } 
  } 
} 
```

# The main function

As always, the `main` function creates an application, shows the window, and executes the application until the user closes the window or selects the Exit menu item.

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

# Noughts and Crosses

The Noughts and Crosses application of this chapter is based on the version in the previous chapter. The difference is that in this version the computer plays against a human.

# The NaCWindow class

The `NaCWindow` class is similar to the `OthelloWindow` class in the previous section (NaC is an abbreviation for Noughts and Crosses). It adds two submenus to the game menu, where the computer or human makes the first move and selects a nought or cross:

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

The `NaCWindow.cpp` file holds the definitions of the methods of the `NaCWindow` class:

**NaCWindow.cpp**

```cpp
#include "NaCWindow.h" 
#include <QtWidgets> 
```

The title has been changed to `Noughts and Crosses Advanced`:

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

# The NaCWidget class

The `NaCWidget` class has been improved compared to the version in the previous chapter. It holds the `calculateComputerMove` and `calculateMarkValue` methods for the computer to play against the human:

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

The `NaCWidget.cpp` file holds the definitions of the methods of the `NaCWidget` class:

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

The `isComputerStartsNoughtEnabled`, `isComputerStartsCrossEnabled`, `isHumanStartsNoughtEnabled`, and `isHumanStartsCrossEnabled` methods decide whether to enable the `Computer Nought`, `Computer Cross`, `Human Nought`, and `Human cross` menu items. They are all enabled when there is no game in progress:

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

The `onComputerStartsNought`, `onComputerStartsCross`, `onHumanStartsNought`, and `onHumanStartsCross` are called when the user selects the `Computer Noughts`, `Computer Cross`, `Human Noughts`, and `Human Cross` menu items. They set the game in progress, set the computer and human marks to nought and cross, and update the window. In cases where the computer makes the first move, it is placed in the middle of the game grid in order to use the game grid as effectively as possible:

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

The `mouseMark` method is called when the human player clicks an empty position in the game grid. We start by setting the mark to the position and updating the window:

```cpp
void NaCWidget::mouseMark(int row, int column) { 
  set(row, column, m_humanMark); 
  update(); 
```

If the human's move did not cause them to win the game, we calculate the next move of the computer, set the position, check if the move has caused the computer to win the game, and update the window:

```cpp
  if (!checkWinner(row, column, m_humanMark)) { 
    calculateComputerMove(row, column); 
    set(row, column, m_computerMark); 
    checkWinner(row, column, m_computerMark); 
    update(); 
  } 
} 
```

The `drawMark` method is called when a position needs to be repainted. It is similar to its counterpart in the previous chapter. It draws a nought or a cross:

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

The `checkWinner` method is also similar to its counterpart in the previous chapter. It decides if the latest move has caused five marks in a row. If it has, the winner is announced:

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

The `countMarks` method counts the number of marks in a row. It has been improved compared to its counterpart in the previous chapter. In this version, we also count the highest possible number of marks in a row that the move can lead to. Since `countMarks` is called by `calculateComputerMove`, we need to know how many marks in a row the move may lead to:

```cpp
double NaCWidget::countMarks(int row, int column, int rowStep, 
                        int columnStep, Mark mark) { 
```

The `markCount` field holds the number of marks in a row that we would get if we placed our mark at the given position; `freeCount` holds the number of marks in a row we possibly can get if we continue to add marks in that row. The reason is that the computer will not add marks to a row that cannot become five in a row:

```cpp
  double markCount = 0; 
  int freeCount = 0; 
```

We iterate through the game grid in the given direction:

```cpp
  { bool marked = true; 
    int currentRow = row, currentColumn = column; 

    while ((currentRow >= 0) && (currentRow < ROWS) && 
           (currentColumn >= 0) && (currentColumn < COLUMNS)) { 
```

As long as we find the mark, we increase both `markCount` and `freeCount`:

```cpp
      if (get(currentRow, currentColumn) == mark) { 
        if (marked) { 
          ++markCount; 
        } 

        ++freeCount; 
      } 
```

If we find an empty position, we add `0.4` (since a free row is better than a closed row) to the `markCount`, and continue to increase the `freeCount`:

```cpp
      else if (get(currentRow, currentColumn) == Empty) { 
        if (marked) { 
          markCount += 0.4; 
        } 

        marked = false; 
        ++freeCount; 
      } 
```

If we find neither the computer mark nor an empty position, we must have found the human's mark, and we break the iteration:

```cpp
      else { 
        break; 
      } 
```

At the end of each iteration, we add the row and columns steps to the current row and column:

```cpp
      currentRow += rowStep; 
      currentColumn += columnStep; 
    } 
  } 
```

We perform a similar iteration in the opposite direction. The only difference is that we subtract the row and columns steps at the end of each iteration, instead of adding to them:

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

If the free count is at least five, we return the mark count. If it is less than five, we return zero, since we cannot obtain five in a row in this direction:

```cpp
  return (freeCount >= 5) ? markCount : 0; 
} 
```

The `calculateComputerMove` method calculates the computer move that causes the maximum numbers of marks in a row. We count both the computer and human's rows, since we may be facing a situation where we need to stop the human from winning instead of maximizing the computer's chance to win.

The `maxComputerValue` and `maxHumanValue` fields hold the maximum number of marks in a row that we have found so far. The `maxComputerList` and `maxHumanList` hold the position that causes the maximum number of marks in a row for the computer and the human:

```cpp
void NaCWidget::calculateComputerMove(int& maxRow,int &maxColumn){ 
  double maxComputerValue = 0, maxHumanValue = 0; 
  QList<QPair<int,int>> maxComputerList, maxHumanList; 
```

We iterate through the game grid. For each empty position, we try to set the computer and human mark and see how many marks in a row that would cause:

```cpp
  for (int row = 0; row < ROWS; ++row) { 
    for (int column = 0; column < COLUMNS; ++column)  { 
      if (get(row, column) == Empty) { 
        set(row, column, m_computerMark); 
```

We obtain the maximum number of marks in a row for the computer and human mark. If it is larger than the previous maximum number, we clear the list and add the position to the list:

```cpp
        { double computerValue = 
            calculateMarkValue(row, column, m_computerMark); 

          if (computerValue > maxComputerValue) { 
            maxComputerValue = computerValue; 
            maxComputerList.clear(); 
            maxComputerList.append(QPair<int,int>(row, column)); 
          } 
```

If the new number of marks in a row is greater than zero or equals the maximum number, we just add the position:

```cpp
          else if ((computerValue > 0) && 
                   (computerValue == maxComputerValue)) { 
            maxComputerList.append(QPair<int,int>(row, column)); 
          } 
        } 
```

We do the same for the human mark as the computer mark:

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

Finally, we reset the position to the empty value:

```cpp
        set(row, column, Empty); 
      } 
    } 
  } 
```

The computer or human must have at least one in a row for a position:

```cpp
  assert(!maxComputerList.empty() && !maxHumanList.empty()); 
```

If the computer's value is at least two and larger the human value, or if the human value is less the four, we randomly select one of the computer's maximum moves:

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

However, if the computer cannot make at least two in a row, or if the human is about to get five in a row, we randomly select one of the human's maximum moves:

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

The `calculateMarkValue` method calculates the maximum number of marks in a row that the given position may cause by calculating the larger value of its four directions:

```cpp
double NaCWidget::calculateMarkValue(int row, int column, 
                                     Mark mark) { 
  return qMax(qMax(countMarks(row, column, -1, 0, mark), 
                   countMarks(row, column, 0, -1, mark)), 
              qMax(countMarks(row, column, -1, 1, mark), 
                   countMarks(row, column, 1, 1, mark))); 
} 
```

# The main function

Finally, the `main` function works at it always does in the Qt applications:

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

# Summary

In this chapter, we have developed more advanced versions of the games of the previous chapter. In both Othello and Noughts and Crosses, we have added code that lets the computer play against the human. In Othello, we looked for the position in the game grid that would cause the highest number of the opponent’s marks to be changed. In Noughts and Crosses, we searched for the move that gave the computer the highest possible number of marks in a row, preferably five in a row. However, we also had to search for the potential number of marks in a row for the opponent, and prevent their next move if it led to victory. Now, I suggest that you sit back and enjoy a couple of rounds with the computer before moving on to the next chapter.

In the next chapter, we will start developing a **Domain-Specific Language** (**DSL**), which is a language intended for a specific domain. We will develop a DSL for specifying the drawings of graphical objects, such as lines, rectangles, ellipses, and text, as well as the settings for color, font, pen and brush style, and alignment. We will also write a viewer that displays the graphical objects.