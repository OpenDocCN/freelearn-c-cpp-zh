# The Games

In [Chapter 6](f20d7a19-156f-43e8-92a5-46b9068128fc.xhtml), *Enhancing the QT Graphical Applications*, we developed an analog clock, a drawing program, and an editor with the Qt graphical library. In this chapter, we continue by developing the Othello and Noughts and Crosses games with the Qt library. You will find a description of these games after this introduction. We start in this chapter with basic versions, where two players play against each other. In [Chapter 8](ddd1aeb1-7f0c-4a44-b715-860c57771663.xhtml), *The Computer Plays*, we improve the games so that the computer plays against the human.

Topics we will cover in this chapter include:

*   Introduction to game theory. We develop a game grid where the players take turns to add their marks to the game grid.
*   We announce the winner. In Othello, after each move, we calculate how many of the opponent's marks can be changed. When every position of the game grid has been occupied, we declare the winner or a draw.
*   In Noughts and Crosses, we count the number of marks in a row. If there are five marks in a row, we declare the winner.
*   We continue to use C++ features such as classes, fields, and methods. We also continue to use Qt features such as windows and widgets.

# Othello

In Othello, the game grid is empty at the beginning of the game. During the game, two players take turns adding marks, colored in black and white, to the game grid. Each time a player adds a mark, we look at the other marks and see if the new mark causes any of the opponent’s marks to be enclosed. In that case, we swap the color of the opponent’s enclosed marks.

For instance, if the black player adds a black mark in a position where the three marks to the left are white and the fourth mark is black, the three white marks are being enclosed by the two black marks, and they are swapped to black marks. When every position on the game grid has been occupied by white and black marks, we count the marks and the player with the most marks is the winner. If there is an equal number of black and white marks, it is a draw.

Here's what our game should look like:

![](img/c7a9a4b5-274d-46fc-8bde-d9d10a94b08f.png)

# The game widget

First of all, we need a game grid. The `GameWidget` class is common to all the applications of this chapter and of [Chapter 8](ddd1aeb1-7f0c-4a44-b715-860c57771663.xhtml), *The Computer Plays*. In [Chapter 5](411aae8c-9215-4315-8a2e-882bf028834c.xhtml), *Qt Graphical Applications* and [Chapter 6](f20d7a19-156f-43e8-92a5-46b9068128fc.xhtml), *Enhancing the QT Graphical Applications*, we developed the `DocumentWidget` class, since we worked with document-based applications. In this chapter and [Chapter 8](ddd1aeb1-7f0c-4a44-b715-860c57771663.xhtml), *The Computer Plays,* we instead develop the `GameWidget` class.

The `DocumentWidget` class of the two previous chapters and the `GameWidget` class of this chapter and the next chapter have both similarities and differences. They are both subclasses of the Qt class `QWidget`, and they are both intended to be embedded in a window. However, while `DocumentWidget` was intended to hold a document, `GameWidget` is intended to hold a game grid. It draws the grid and catches mouse clicks in the positions of the grid. `GameWidget` is an abstract class that lets it its subclass define methods that are called when the user clicks the mouse or when a mark in one of the positions of the game grid needs to be repainted.

However, we reuse the `MainWindow` class from the previous chapters to hold the main window of the application, with its menu bar.

**GameWidget.h**

```cpp
#ifndef GAMEWIDGET_H 
#define GAMEWIDGET_H 

#include <QPainter> 
#include <QMouseEvent> 
#include <QMessageBox> 

#include "..\MainWindow\MainWindow.h" 

class GameWidget : public QWidget { 
  Q_OBJECT 
```

The constructor initializes the number of rows and columns of the game grid:

```cpp
    public: 
      GameWidget(int rows, int columns, QWidget* parentWidget); 
```

The `clearGrid` method sets every position in the game grid to zero, which is assumed to represent an empty position. Therefore, every class that inherits `GameWidget` shall let the value zero represent an empty position:

```cpp
      void clearGrid(); 
```

The `resizeEvent` method is called when the user changes the size of the window. Since the number of rows and columns is constant, the width and height of each position is changed in accordance with the new size of the window:

```cpp
    void resizeEvent(QResizeEvent *eventPtr); 
```

The `mousePressEvent` is called when the user presses one of the mouse buttons, `paintEvent` is called when the window needs to be repainted, and `closeEvent` is called when the user clicks on the close box at the top-right corner of the window:

```cpp
      void mousePressEvent(QMouseEvent *eventPtr); 
      void paintEvent(QPaintEvent *eventPtr); 
      void closeEvent(QCloseEvent *eventPtr); 
```

The `mouseMark` and `drawMark` methods are pure virtual methods intended to be overridden by subclasses; `mouseMark` is called when the user clicks at a position in the grid, and `drawMark` is called when a position needs to be repainted. They are pure virtual methods, whereas `GameWidget` is abstract, which means that it is only possible to use `GameWidget` as a base class. The subclasses of `GameWidget` must override the methods to become non-abstract:

```cpp
    virtual void mouseMark(int row, int column) = 0; 
    virtual void drawMark(QPainter& painter, 
                          const QRect& markRect, int mark) = 0; 
```

The `isQuitOk` method displays a message box that asks the user if they really want to quit the game:

```cpp
  private: 
    bool isQuitOk(); 
```

The `isQuitEnabled` method is called before the `Game` menu becomes visible. The `Quit` item is enabled when a game is in progress:

```cpp
  public slots: 
    DEFINE_LISTENER(GameWidget, isQuitEnabled); 
```

The `onQuit` and `onExit` methods are called when the user selects the Quit or Exit menu items:

```cpp
    void onQuit(); 
    void onExit(); 
```

The `isGameInProgress` and `setGameInProgress` methods return and set the value of the `m_gameInProgress` field:

```cpp
  protected: 
    bool isGameInProgress() const {return m_gameInProgress;} 
    void setGameInProgress(bool active) 
                          {m_gameInProgress = active;} 
```

The `get` and `set` methods get and set a value at a position in the game grid. The value is an integer; remember that an empty position is assumed to hold the value zero:

```cpp
    protected: 
      int get(int row, int column) const; 
      void set(int row, int column, int value); 
```

The `m_gameInProgress` field is true as long as a game is in progress. The `m_rows` and `m_columns` fields hold the number of rows and columns of the game grid; `m_rowHeight` and `m_columnWidth` hold the height and width in pixels of each position in the game grid. Finally, `m_gameGrid` is a pointer to a buffer holding the values of the positions of the game grid:

```cpp
     private: 
       bool m_gameInProgress = false; 
       int m_rows, m_columns; 
       int m_rowHeight, m_columnWidth; 
       int* m_gameGrid; 
     }; 

     #endif // GAMEWIDGET_H 
```

The `GameWidget.cpp` file holds the definitions of the methods of the `GameWidget` class, the mouse event methods, and the menu methods, as well as the drawings and settings of the marks.

**GameWidget.cpp**

```cpp
#include "GameWidget.h" 
#include <QApplication> 
#include <CAssert> 
```

The constructor initializes the number of rows and columns of the grid, dynamically allocates its memory, and calls `clearGrid` to clear the grid:

```cpp
GameWidget::GameWidget(int rows, int columns, 
                       QWidget* parentWidget) 
 :QWidget(parentWidget), 
  m_rows(rows), 
  m_columns(columns), 
  m_gameGrid(new int[rows * columns]) { 
  assert(rows > 0); 
  assert(columns > 0); 
  clearGrid(); 
} 
```

The `get` method returns the value at the position indicated by the row and column and `set` sets the value. The buffer holding the values is organized in rows. That is, the first part of the buffer holds the first row, and then the second row, and so on:

```cpp
int GameWidget::get(int row, int column) const { 
  return m_gameGrid[(row * m_columns) + column]; 
} 

void GameWidget::set(int row, int column, int value) { 
  m_gameGrid[(row * m_columns) + column] = value; 
} 
```

The `clearGrid` method sets every position to zero, since zero is assumed to represent an empty position:

```cpp
void GameWidget::clearGrid() { 
  for (int row = 0; row < m_rows; ++row) { 
    for (int column = 0; column < m_columns; ++column) { 
      set(row, column, 0); 
    } 
  } 
} 
```

The `Quit` menu item is enabled as long as a game is in progress:

```cpp
bool GameWidget::isQuitEnabled() { 
  return m_gameInProgress; 
} 
```

If a game is in progress when the user selects to quit the game, a message box with a confirmation question is displayed:

```cpp
bool GameWidget::isQuitOk() { 
  if (m_gameInProgress) { 
    QMessageBox messageBox(QMessageBox::Warning, 
                           tr("Quit"), QString()); 
    messageBox.setText(tr("Quit the Game.")); 
    messageBox.setInformativeText 
                  (tr("Do you really want to quit the game?")); 
    messageBox.setStandardButtons(QMessageBox::Yes | 
                                  QMessageBox::No); 
    messageBox.setDefaultButton(QMessageBox::No); 
```

If the user presses the `Yes` button, `true` is returned:

```cpp
    return (messageBox.exec() == QMessageBox::Yes); 
  } 

  return true; 
}

```

The `onQuit` method is called when the user selects the Quit menu item. If the call to `isQuitOk` returns true, `m_gameInProgress` is set to false and update is called, which eventually forces a repaint of the window where the game grid is cleared.

```cpp
void GameWidget::onQuit() { 
  if (isQuitOk()) { 
    m_gameInProgress = false; 
    update(); 
  } 
} 
```

The `onExit` method is called when the user selects the Exit menu item. If the call to `isQuitOk` returns true, the application is exited. This is shown in the following code:

```cpp
void GameWidget::onExit() { 
  if (isQuitOk()) { 
    qApp->exit(0); 
  } 
} 
```

The `resizeEvent` method is called when the user resizes the window. The row height and column width are recalculated since the number of rows and columns is constant regardless of the size of the window. We divide the height and width of the window by the number of rows and columns plus two, since we add extra rows and columns as margins. Consider the following code:

```cpp
void GameWidget::resizeEvent(QResizeEvent* eventPtr) { 
  m_rowHeight = height() / (m_rows + 2); 
  m_columnWidth = width() / (m_columns + 2); 
  QWidget::resizeEvent(eventPtr); 
  update(); 
} 
```

The `mousePressEvent` method is called when the user clicks on the window:

```cpp
    void GameWidget::mousePressEvent(QMouseEvent* eventPtr) { 
       if (m_gameInProgress &&
             (eventPtr->button() == Qt::LeftButton)) { 
       QPoint mousePoint = eventPtr->pos(); 
```

The column width and row height are subtracted from the mouse point, since the game grid is enclosed by margins:

```cpp
    mousePoint.setX(mousePoint.x() - m_columnWidth); 
    mousePoint.setY(mousePoint.y() - m_rowHeight); 
```

If the mouse point is located inside one of the game grid positions, and that position is empty (zero), the pure virtual method `mouseMark` is called, which takes care of the actual action of the mouse click. In the next section, black and white marks are added to the game grid, and in the Noughts and Crosses application later on. Noughts and crosses are added to the game grid:

```cpp
      int row = mousePoint.y() / m_rowHeight, 
         column = mousePoint.x() / m_columnWidth; 
```

If the rows and columns clicked are located in the game grid (rather than in the margins outside the game grid) and the position is empty (zero), we call the `mouseMark`, which is a pure virtual method, with the row and column:

```cpp
    if ((row < m_rows) && (column < m_columns) && 
        (get(row, column) == 0)) { 
      mouseMark(row, column); 
      update(); 
    } 
  } 
} 
```

The `paintEvent` method is called when the window needs to be repainted. If a game is in progress (`m_gameInProgress` is true), the rows and columns are written, and then for each position in the game grid, the pure virtual method `drawMark` is called, which takes care of the actual painting of each position:

```cpp
void GameWidget::paintEvent(QPaintEvent* /*eventPtr*/) { 
  if (m_gameInProgress) { 
    QPainter painter(this); 
    painter.setRenderHint(QPainter::Antialiasing); 
    painter.setRenderHint(QPainter::TextAntialiasing); 
```

First, we iterate through the rows and for each row, we write a letter from `A` to `Z`. There are 26 letters of the alphabet, and we assume there are no more than 26 rows:

```cpp
    for (int row = 0; row < m_rows; ++row) { 
      QString text; 
      text.sprintf("%c", (char) (((int) 'A') + row)); 
      QRect charRect(0, (row + 1) * m_rowHeight, 
                     m_columnWidth, m_rowHeight); 
      painter.drawText(charRect, Qt::AlignCenter | 
                       Qt::AlignHCenter, text); 
    } 
```

Then we iterate through the columns, and for each column, we write its number:

```cpp
    for (int column = 0; column < m_columns; ++column) { 
      QString text; 
      text.sprintf("%i", column); 
      QRect charRect((column + 1) * m_columnWidth, 0, 
                     m_columnWidth, m_rowHeight); 
      painter.drawText(charRect, Qt::AlignCenter | 
                       Qt::AlignHCenter, text); 
    } 

    painter.save(); 
    painter.translate(m_columnWidth, m_rowHeight); 
```

A pure virtual method is a method that is not intended to be defined in the class, only in its subclasses. A class holding at least one pure virtual method becomes abstract, which means that it is not possible to create objects of the class. The class can only be used as a base class in a class hierarchy. A class that inherits an abstract class must define each pure virtual method of the base class, or become abstract itself.

Finally, we iterate through the game grid, and for each position, we call the pure virtual method `drawMark` with the rectangle of the position and its current mark:

```cpp
    for (int row = 0; row < m_rows; ++row) { 
      for (int column = 0; column < m_columns; ++column) { 
        QRect markRect(column * m_columnWidth, row * m_rowHeight, 
                       m_columnWidth, m_rowHeight); 
        painter.setPen(Qt::black); 
        painter.drawRect(markRect); 
        painter.fillRect(markRect, Qt::lightGray); 
        drawMark(painter, markRect, get(row, column)); 
      } 
    } 

    painter.restore(); 
     } 
    } 
```

The `closeEvent` method is called when the user clicks on the close box at the top-right corner of the window. If the call to `isQuitOk` returns true, the window is closed, and the application is exited:

```cpp
void GameWidget::closeEvent(QCloseEvent* eventPtr) { 
  if (isQuitOk()) { 
    eventPtr->accept(); 
    qApp->exit(0); 
  } 
  else { 
    eventPtr->ignore(); 
  } 
} 
```

# The OthelloWindow class

The `Othello` class is a subclass of `MainWindow` from [Chapter 6](f20d7a19-156f-43e8-92a5-46b9068128fc.xhtml), *Enhancing the QT Graphical Applications*. It adds menus to the window and sets the `OthelloWidget` class here, which is a subclass of `GameWidget`, to its central widget.

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
```

The `m_othelloWidgetPtr` field holds a pointer to the widget located in the center of the window. It points at an object of the `OthelloWidget` class. This is shown in the following code:

```cpp
  private: 
    OthelloWidget* m_othelloWidgetPtr; 
}; 

#endif // OTHELLOWINDOW_H 
```

The `OthelloWindow.cpp` file defines the methods of the `OthelloWIndow` class.

**OthelloWindow.cpp**

```cpp
#include "OthelloWidget.h" 
#include "OthelloWindow.h" 
#include <QtWidgets> 
```

The constructor sets the title of the window to `Othello` and the size to *1000* x *500* pixels:

```cpp
OthelloWindow::OthelloWindow(QWidget *parentWidget /*= nullptr*/)
 :MainWindow(parentWidget) {
  setWindowTitle(tr("Othello"));
  resize(1000, 500);
```

An `OthelloWidget` object is dynamically created and placed at the center of the window:

```cpp
    m_othelloWidgetPtr = new OthelloWidget(this); 
    setCentralWidget(m_othelloWidgetPtr); 
```

We add the menu `Game` to the menu bar and connect the `onMenuShow` method to the menu, which causes it to be called before the menu becomes visible:

```cpp
  { QMenu* gameMenuPtr = menuBar()->addMenu(tr("&Game")); 
    connect(gameMenuPtr, SIGNAL(aboutToShow()), 
            this, SLOT(onMenuShow())); 
```

The user can choose the black or white color to make the first move. The `isBlackStartsEnabled` and `isWhiteStartsEnabled` methods are called before the items become visible. The items become disabled when a game is in progress:

```cpp
    addAction(gameMenuPtr, tr("&Black Starts"), 
              SLOT(onBlackStarts()), 0, 
              tr("Black Starts"), nullptr,tr("Black Starts"), 
              LISTENER(isBlackStartsEnabled)); 

    addAction(gameMenuPtr, tr("&White Starts"), 
              SLOT(onWhiteStarts()), 0, 
              tr("White Starts"), nullptr, tr("White Starts"), 
              LISTENER(isWhiteStartsEnabled)); 

    gameMenuPtr->addSeparator(); 
```

When a game is in progress, the user can quit the game. The item becomes disabled when no game is in progress:

```cpp
    addAction(gameMenuPtr, tr("&Quit the Game"), 
              SLOT(onQuit()), 
              QKeySequence(Qt::CTRL + Qt::Key_Q), 
              tr("Quit Game"), nullptr, tr("Quit the Game"), 
              LISTENER(isQuitEnabled)); 
```

The user can exit the application at any time:

```cpp
    addAction(gameMenuPtr, tr("E&xit"), 
              SLOT(onExit()), QKeySequence::Quit); 
  } 
} 
```

The destructor deallocates the `Othello` widget in the center of the window:

```cpp
OthelloWindow::~OthelloWindow() { 
  delete m_othelloWidgetPtr; 
} 
```

# The OthelloWidget class

`OthelloWidget` is a subclass of the `GameWidget` class we defined at the beginning of this chapter. It becomes a non-abstract class by overriding `mouseMark` and `drawMark`, which are called when the user clicks at a position in the game grid and when a position needs to be repainted.

**OthelloWidget.h**

```cpp
#ifndef OTHELLOWIDGET_H 
#define OTHELLOWIDGET_H 

#include "..\MainWindow\GameWidget.h" 

#define ROWS    8 
#define COLUMNS 8 
```

A mark in Othello can be black or white. We use the `Mark` enumeration to store values on the game grid. The `Empty` item holds a value of zero, which is assumed to be `GameWidget` to represent an empty position:

```cpp
enum Mark {Empty = 0, Black, White}; 

class OthelloWidget : public GameWidget { 
  Q_OBJECT 

  public: 
    OthelloWidget(QWidget* parentWidget); 

    void mouseMark(int row, int column); 
    void drawMark(QPainter& painter, 
                  const QRect& markRect, int mark); 
```

The `isBlackStartsEnabled` and `isWhiteStartsEnabled` listeners are called before the `BlackStarts` and `WhiteStarts` menu items become visible in order to enable them. Note that the listeners and methods must be marked as public slots for the menu framework to allow them as listeners:

```cpp
    public slots: 
     DEFINE_LISTENER(OthelloWidget, isBlackStartsEnabled); 
     DEFINE_LISTENER(OthelloWidget, isWhiteStartsEnabled); 
```

The `onBlackStarts` and `onWhiteStarts` methods are called when the `BlackStarts` and `WhiteStarts` menu items are selected by the user:

```cpp
    void onBlackStarts(); 
    void onWhiteStarts(); 
```

The `checkWinner` method checks if every position on the game grid has been occupied by a black or white mark. If it has, the marks are counted, and the winner is announced unless it is a draw:

```cpp
   private: 
     void checkWinner(); 
```

The `turn` method is called when one of the players has made a move. It calculates the positions to be turned as a result of the move:

```cpp
     void turn(int row, int column, Mark mark); 
```

The `calculateMark` method calculates the set of marks to be turned if the player places the mark in the position given by the row and column:

```cpp
    void calculateMark(int row, int column, Mark mark, 
                       QSet<QPair<int,int>>& resultSet); 
```

The `m_nextMark` field is alternatively given the values `Black` and `White` of the preceding `Mark` enumeration, depending on which player is about to do the next move.

It is initialized by `onBlackStarts` or `onWhiteStarts`, as shown in the previous code:

```cpp
    Mark m_nextMark; 
}; 

#endif // OTHELLOWIDGET_H 
```

The `OthelloWidget` class holds the functionality of the game. It allows the player to add black and white marks to the game grid, turn marks, and announce the winner.

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

The `BlackStarts` and `WhiteStarts` menu items are enabled when there is not already a game in progress:

```cpp
bool OthelloWidget::isBlackStartsEnabled() { 
  return !isGameInProgress(); 
} 

bool OthelloWidget::isWhiteStartsEnabled() { 
  return !isGameInProgress(); 
} 
```

The `onBlackStarts` and `onWhiteStarts` methods set a new game in progress, set the mark to make the first move (black or white), clear the grid, and update the window to paint an empty game grid:

```cpp
void OthelloWidget::onBlackStarts() { 
  setGameInProgress(true); 
  m_nextMark = Black; 
  update(); 
} 

void OthelloWidget::onWhiteStarts() { 
  setGameInProgress(true); 
  m_nextMark = White; 
  update(); 
} 
```

The `onMouseMark` is called when the player clicks an empty position on the game grid. We set the position with the next mark, turn every mark that is affected by the move, and update the window to reflect the change:

```cpp
void OthelloWidget::mouseMark(int row, int column) { 
  set(row, column, m_nextMark); 
  turn(row, column, m_nextMark); 
  update(); 
```

We check if the move has caused the game grid to become full and switch the next mark:

```cpp
  checkWinner(); 
  m_nextMark = (m_nextMark == Black) ? White : Black; 
} 
```

The `drawMark` method is called when a position in the game grid needs to be repainted. We draw a black or white ellipse with black borders if the position is not empty. If the position is empty, we do nothing. Note that the framework clears the window before the call to repaint:

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
```

```cpp
  } 
} 
```

The `checkWinner` method counts the number of positions that are occupied by black and white marks or are empty:

```cpp
void OthelloWidget::checkWinner() { 
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
```

If there are no empty positions left, the game is over, and we announce the winner, unless it is a draw. The winner is the player with the most marks in their color:

```cpp
  if (empties == 0) { 
    QMessageBox messageBox(QMessageBox::Information, 
        tr("Victory"), QString()); 
    QString text; 

    if (blacks == whites) { 
      text.sprintf("A Draw."); 
    } 
    else if (blacks > whites) { 
      text.sprintf("The Winner: Black"); 
    } 
    else { 
      text.sprintf("The Winner: White"); 
    } 

    messageBox.setText(text); 
    messageBox.setStandardButtons(QMessageBox::Ok); 
    messageBox.exec(); 
    setGameInProgress(false); 

    clearGrid(); 
    update(); 
  } 
} 
```

The `turn` method calls `calculateMark` to obtain the set of positions where the mark shall be turned. Then each position in the set is set to the mark in question.

In this application, `turn` is the only method that calls `calculateMark`. However, in [Chapter 8](ddd1aeb1-7f0c-4a44-b715-860c57771663.xhtml), *The Computer Plays*, `calculateMark` will also be called to calculate the move of the computer player. Therefore, the functionality of `turn` and `calculateMark` are divided into two methods:

```cpp
void OthelloWidget::turn(int row, int column, Mark mark) { 
  QSet<QPair<int,int>> totalSet; 
  calculateMark(row, column, mark, totalSet); 

  for (QPair<int,int> pair : totalSet) { 
    int row = pair.first, column = pair.second; 
    set(row, column, mark); 
  } 
} 
```

The `calculateMark` method counts the number of marks that will be turned for each position on the game grid, in all eight directions:

```cpp
void OthelloWidget::calculateMark(int row, int column, 
    Mark playerMark, QSet<QPair<int,int>>& totalSet){ 
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
```

We iterate through the directions and, for each direction, keep moving as long as we find the mark of the opponent:

```cpp
  for (int index = 0; index < arraySize; ++index) { 
    QPair<int,int> pair = directionArray[index]; 
```

The `row` and `column` fields hold the current row and column as long as we iterate in that direction:

```cpp
    int rowStep = pair.first, columnStep = pair.second, 
        currRow = row, currColumn = column; 
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

If we find the player's mark, we add the direction set to the total set and break the iteration:

```cpp
else if (get(currRow, currColumn) == playerMark) { 
  totalSet += directionSet; 
  break; 
} 
```

If we do not find the player's mark or an empty position, we have found the opponent's mark, and we add its position to the direction set:

```cpp
      else { 
        directionSet.insert(QPair<int,int>(row, column)); 
      } 
    } 
  } 
} 
```

# The main function

The `main` function works in the same way as in the previous Qt applications. It creates an application, shows the Othello window, and executes the applications. The execution continues until the `exit` method is called, which it is when the user closes the window or selects the Exit menu item.

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

# Noughts and crosses

The Noughts and Crosses application sets up a game grid and allows two players to play each other. In Noughts and Crosses, two players take turns adding noughts and crosses to a game grid. The player that first manages to place five marks in a row wins the game. The marks can be placed horizontally, vertically, or diagonally. While each player tries to place five of their own marks in a row, they must also try to prevent the opponent from placing five marks in a row.

In [Chapter 8](ddd1aeb1-7f0c-4a44-b715-860c57771663.xhtml), *The Computer Plays*, the computer plays against the human.

# The NaCWindow class

We reuse the `GameWidget` from the game widget section. The `NaCWindow` class is similar to `OthelloWindow`. It adds the `Nought Begins` and `Cross Begins` menu items to the window's menu bar.

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
    void closeEvent(QCloseEvent *eventPtr) override 
                   {m_nacWidgetPtr->closeEvent(eventPtr);} 

  private: 
    NaCWidget* m_nacWidgetPtr; 
}; 

#endif // NACWINDOW_H 
```

The `NaCWindow.cpp` file holds the definitions of the methods of the `NacWindow` class.

**NaCWindow.cpp**

```cpp
#include "NaCWindow.h" 
#include <QtWidgets> 

NaCWindow::NaCWindow(QWidget *parentWidget /*= nullptr*/) 
 :MainWindow(parentWidget) { 
  setWindowTitle(tr("Noughts and Crosses")); 
  resize(1000, 500); 

  m_nacWidgetPtr = new NaCWidget(this); 
  setCentralWidget(m_nacWidgetPtr); 

  { QMenu* gameMenuPtr = menuBar()->addMenu(tr("&Game")); 
    connect(gameMenuPtr, SIGNAL(aboutToShow()), 
            this, SLOT(onMenuShow())); 

    addAction(gameMenuPtr, tr("&Nought Starts"), 
              SLOT(onNoughtStarts()), 0, 
              tr("Nought Starts"), nullptr, tr("Nought Starts"), 
              LISTENER(isNoughtStartsEnabled)); 

    addAction(gameMenuPtr, tr("&Cross Starts"), 
              SLOT(onCrossStarts()), 0, 
              tr("Cross Starts"), nullptr, tr("Cross Starts"), 
              LISTENER(isCrossStartsEnabled)); 

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

The `NaCWidget` class handles the functionality of Noughts and Crosses. It allows two players to play each other. In [Chapter 8](ddd1aeb1-7f0c-4a44-b715-860c57771663.xhtml), *The Computer Plays*, we will write a game where the computer plays the human.

**NaCWidget.h**

```cpp
#ifndef NACWIDGET_H 
#define NACWIDGET_H 

#include "..\MainWindow\GameWidget.h" 

#define ROWS    26 
#define COLUMNS 26 
```

Similar to the Othello application, a position in the game grid can hold one of three values:

*   `Empty` (which is zero)
*   `Nought`
*   `Cross`

The `Mark` enumeration corresponds to the `Empty`, `Nought`, and `Cross` values: 

```cpp
enum Mark {Empty = 0, Nought, Cross}; 

class NaCWidget : public GameWidget { 
  Q_OBJECT 

  public: 
    NaCWidget(QWidget* parentWidget); 

    void mouseMark(int row, int column); 
    void drawMark(QPainter& painter, 
                  const QRect& markRect, int mark); 

  public slots: 
    DEFINE_LISTENER(NaCWidget, isNoughtStartsEnabled); 
    void onNoughtStarts(); 

    DEFINE_LISTENER(NaCWidget, isCrossStartsEnabled); 
    void onCrossStarts(); 

  private: 
    void checkWinner(int row, int column, Mark mark); 
    int countMarks(int row, int column, int rowStep, 
                   int columnStep, Mark mark); 

    Mark m_nextMark; 
}; 

#endif // NACWIDGET_H 
```

The `NaCWidget.cpp` file holds the definitions of the methods of the `NaCWidget` class.

**NaCWidget.cpp**

```cpp
#include "NaCWidget.h" 
#include <CTime> 

NaCWidget::NaCWidget(QWidget* parentWidget) 
 :GameWidget(ROWS, COLUMNS, parentWidget) { 
  // Empty. 
} 
```

The `isNoughtStartsEnabled` and `isCrossStartsEnabled` methods are called before the `Game` menu becomes visible. The `Noughts Begins` and `Cross Begins` menu items are enabled if there is no game in progress:

```cpp
bool NaCWidget::isCrossStartsEnabled() { 
  return !isGameInProgress(); 
} 

bool NaCWidget::isNoughtStartsEnabled() { 
  return !isGameInProgress(); 
} 
```

The `onNoughtBegins` and `onCrossBegins` methods are called when the user selects the `Nought Begins` and `Cross Begins` menu items. They set the game in progress, set the first mark to make the first move (`m_nextMark`), and force a repainting of the game grid by calling `update`:

```cpp
void NaCWidget::onNoughtStarts() { 
  setGameInProgress(true); 
  m_nextMark = Nought; 
  update(); 
} 

void NaCWidget::onCrossStarts() { 
  setGameInProgress(true); 
  m_nextMark = Cross; 
  update(); 
} 
```

The `mouseMark` method is called when the players click a position in the game grid. We set the next mark at the position, check if one of the players has won the game, swap the next move, and repaint the window by calling `update`:

```cpp
void NaCWidget::mouseMark(int row, int column) { 
  set(row, column, m_nextMark); 
  checkWinner(row, column, m_nextMark); 
  m_nextMark = (m_nextMark == Nought) ? Cross : Nought; 
  update(); 
} 
```

The `drawMark` method is called when a position in the game grid needs to be repainted:

```cpp
void NaCWidget::drawMark(QPainter& painter, 
    const QRect& markRect, int mark) { 
```

We set the pen color to black, and in the case of a nought, we draw an ellipse, as follows:

```cpp
  painter.setPen(Qt::black); 
  switch (mark) { 
    case Nought: 
      painter.drawEllipse(markRect); 
      break; 
```

In the case of a cross, we draw two lines between the top-left and bottom-right corners and between the top-right and bottom-left corners:

```cpp
    case Cross: 
      painter.drawLine(markRect.topLeft(), 
                       markRect.bottomRight()); 
      painter.drawLine(markRect.topRight(), 
                       markRect.bottomLeft()); 
      break; 
```

In the case of an empty position, we do nothing. Remember that the framework clears the window before the repainting:

```cpp
    case Empty: 
      break; 
  } 
} 
```

When a player has made a move, we check if the move has led to victory. We call `countMarks` in four directions to `checkWinner` and see if the move has caused five marks in a row:

```cpp
    void NaCWidget::checkWinner(int row, int column, Mark mark) { 
```

For the north and south directions, the code would be:

```cpp
  if ((countMarks(row, column, -1, 0, mark) >= 5) || 
```

 For the west and east directions, the code would be:

```cpp
      (countMarks(row, column, 0, -1, mark) >= 5) || 
```

For the northwest and southeast directions, the code would be:

```cpp
      (countMarks(row, column, -1, 1, mark) >=5)|| 
```

For southeast and northwest, it would be:

```cpp
      (countMarks(row, column, 1, 1, mark) >= 5)) { 
```

If the move has caused five marks in a row, we display a message box with the winner (black or white). In Noughts and Crosses, there can be no draw:

```cpp
    QMessageBox messageBox(QMessageBox::Information, 
                           tr("Victory"), QString()); 
    QString text; 
    text.sprintf("The Winner: %s.", 
                 (mark == Nought) ? "Nought" : "Cross"); 

    messageBox.setText(text); 
    messageBox.setStandardButtons(QMessageBox::Ok); 
    messageBox.exec(); 
    setGameInProgress(false); 
```

The game grid is cleared, and is thereby ready for another game:

```cpp
    clearGrid(); 
    update(); 
  } 
} 
```

The `countMarks` method counts the number of marks in a row. We `countMarks` the number of marks in both directions. For instance, if both `rowStep` and `columnStep` are minus one, we decrease the current row and column by one for each iteration. That means that we call `countMarks` in the northeast direction in the first iteration. In the second iteration, we call `countMarks` in the opposite direction, that is, in the southwest direction:

```cpp
int NaCWidget::countMarks(int row, int column, int rowStep, 
                          int columnStep, Mark mark) { 
  int countMarks = 0; 
```

We keep counting until we encounter one of the game grid borders, or we find a mark that is not the mark we are counting, that is, the mark of the opposite player or an empty mark:

```cpp
    { int currentRow = row, currentColumn = column; 
      while ((currentRow >= 0) && (currentRow < ROWS) && 
             (currentColumn >= 0) && (currentColumn < COLUMNS) && 
             (get(currentRow, currentColumn) == mark)) { 
          ++countMarks; 
          currentRow += rowStep; 
          currentColumn += columnStep; 
         } 
    } 
```

In the second iteration, we subtract the row and column steps instead of adding them. In this way, we call `countMarks` in the opposite direction. We also initialize the current rows and columns by adding the steps in order, so we do not `countMarks` the middle mark twice:

```cpp
  { int currentRow = row + rowStep, 
        currentColumn = column + columnStep; 
      while ((currentRow >= 0) && (currentRow < ROWS) && 
           (currentColumn >= 0) && (currentColumn < COLUMNS) && 
           (get(currentRow, currentColumn) == mark)) { 
        ++countMarks; 
        currentRow -= rowStep; 
        currentColumn -= columnStep; 
      } 
     } 

    return countMarks; 
  } 
```

# The main function

The `main` function creates the application, shows the window, and executes the application until the user closes the window or selects the Exit menu item.

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

The output for the preceding code is as follows:

![](img/d0111dd0-efe7-4911-8851-1bfa61a04170.png)

# Summary

In this chapter, we developed the two games, Othello and Noughts and Crosses. We were introduced to game theory, and we developed a game grid where the players take turns to add their marks. In Othello, we developed methods to count the number of marks to change for each move, and in Noughts and Crosses, we developed methods to recognize if one of the players had managed to place five marks in a row—if they had, we declared them the winner.

In [Chapter 8](ddd1aeb1-7f0c-4a44-b715-860c57771663.xhtml), *The Computer Plays*, we will develop more advanced versions of these games, where the computer plays against a human.