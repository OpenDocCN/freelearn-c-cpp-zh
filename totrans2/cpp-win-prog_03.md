# Chapter 3. Building a Tetris Application

In this chapter, we develop a classic Tetris game. We look further into the `Window` class, including text writing and drawing figures that are more complex. We look also into timing, random numbers, and graphical updates such as falling figures and flash effects. An illustration of it is shown next:

![Building a Tetris Application](img/image_03_001.jpg)

# The MainWindow function

The `MainWindow` function is similar to the methods in [Chapter 2](ch02.html "Chapter 2. Hello, Small World!"), *Hello, Small World!*. It sets the application name and returns a pointer to the main window, which, in this case, is an instance of the `TetrisWindow` class. As stated in [Chapter 2](ch02.html "Chapter 2. Hello, Small World!"), *Hello, Small World!* the application name is used when accessing the registry, when opening or saving a file, and by the **About** menu item. However, none of that functionality is used in this application:

**MainWindow.cpp**

[PRE0]

# The Tetris window

In this application, we do not use the `StandardDocument` framework from the [Chapter 2](ch02.html "Chapter 2. Hello, Small World!"), *Hello, Small World!*. Instead, the `TetrisWindow` class extends the Small Windows root class `Window` directly. The reason is simply that we do not need the functionality of the `StandardDocument` framework or its base class `Document`. We do not use menus or accelerators, and we do not save or load files:

**TetrisWindow.h**

[PRE1]

In this application, we ignore the mouse. Instead, we look into keyboard handling. The `OnKeyDown` method is called when the user presses or releases a key:

[PRE2]

Similar to the circle application, the `OnDraw` method is called every time the window's client area needs to be redrawn:

[PRE3]

The `OnGainFocus` and `OnLoseFocus` methods are called when the window gains or loses input focus, respectively. When the window loses input focus, it will not receive any keyboard input and the timer is turned off, preventing the falling figure from moving:

[PRE4]

The `OnTimer` method is called every second the window has focus. It tries to move the falling figure one step downward. It calls the `NewFigure` method if it fails to move the figure downward. The `NewFigure` method tries to introduce a new figure on the game board. If that fails, the `GameOver` method is called, which asks the user if they want a new game. The `NewGame` method is called if the user wants a new game. If the user does not want a new game, it exits the application:

[PRE5]

the `DeleteFullRows` examines each row by calling the `IsRowFull` method and calls the `FlashRow` and `DeleteRow` methods for each full row:

[PRE6]

The `TryClose`  method is called if the user tries to close the window by clicking on the cross in the top-right corner of the window. It displays a message box that asks the user if they really want to quit:

[PRE7]

The `gameGrid` field holds the grid on which the figures are displayed (see the next section). The falling figure (`fallingFigure`) is falling down on the grid, and the next figure to fall down (`nextFigure`) is displayed in the top-right corner. Each time the player fills a row, the score (`currScore`) is increased. The timer identity (`TimerId`) is needed to keep track of the timer and is given the arbitrary value of `1000`. Finally, the figure list (`figureList`) will be filled with seven figures, one of each color. Each time a new figure is needed, a randomly chosen figure from the list will be chosen and copied:

[PRE8]

The `PreviewCoordinate` parameter in the `Window` constructor call indicates that the window's size is fixed, and the second parameter indicates that the size is 100 * 100 units. This means that unlike the circle application, the size of figures and game boards will change when the user changes the window's size:

**TetrisWindow.cpp**

[PRE9]

The upper 20 percent of the client area is reserved for the score and the next figure. The game grid covers the lower 80 percent of the client area (from height unit 20 to 100):

[PRE10]

Since we extend the `Window` class, we need to set the window header manually:

[PRE11]

The timer interval is set to `1000` milliseconds, which means that `OnTimer` will be called every second. The random generator is initialized by calling the C standard functions `srand` and `time`:

[PRE12]

The figure list is initialized with one figure of each color; the falling and next figure are randomly chosen from that list. One of the figures in the list will be copied every time we need a new figure:

[PRE13]

Strictly speaking, it is not necessary to drop the timer when closing the Tetris window. The destructor is included only for the sake of completeness:

[PRE14]

## Keyboard input

The `OnKeyDown` method overrides the method in the `Window` class and is called each time the user presses a key. We try to move the falling figure in accordance with the key pressed. We do not care whether the user has pressed the *Shift* or *Ctrl* key:

[PRE15]

When the user presses the Space key, the falling figure falls with visible speed to create the illusion of falling. We try to move the falling figure one step down every 10 milliseconds by calling the Win32 API function `Sleep`. The `TryMoveDown` method returns `false` when it is no longer possible to move the figure downward:

[PRE16]

## Drawing

The `OnDraw` method starts by drawing the game grid and two lines dividing the client area into three parts. The top-left corner displays the current score, the top-right corner displays the next figure, and the lower part displays the actual game grid:

[PRE17]

Note that we add an offset when drawing the next figure in order to move from the game grid to the top-right corner. The value `25` moves the figure from the middle of the grid to the middle of its right half, and the value `-18` moves from the grid up to the area preceding the grid:

[PRE18]

The score font is set to `Times New Roman`, size `10`. Here, the size does not refer to typographical points, but to logical units. Since the call to the `Window` constructor states we gave the `PreviewCoordinate` coordinate system and the size 100 * 100, the height of the text will be 10 units, which is a tenth of the text client area's height. It is also half the height of the part of the client area where the score is written:

[PRE19]

The final `false` parameter in the call to the `DrawText` method indicates that the size of the text won't be recalculated. In the next chapters, we will display text that maintains the same size, regardless of the window size and the screen resolution. In this chapter, however, the size of the text will be changed when the user changes the size of window:

[PRE20]

## Input focus

The `OnGainFocus` and `OnLoseFocus` methods start and stop the timer, respectively, so that the falling figure does not fall down when the window is out of focus:

[PRE21]

## The timer

The timer is active when it has the input focus. When active, the `TryMoveDown` method will be called every time the `OnTimer` method is called (once every second). When the figure cannot fall down any more (the `TryMoveDown` method returns `false`), the `EndOfFigure` method is called:

[PRE22]

## New figures

When it is not possible for the falling figure to move downward, the `OnTimer` method calls the `NewFigure` method. First, we need to store the falling figure to the game grid by calling the `AddToGrid` method. Then, we let the next figure become the new falling figure and we choose by random the new next figure from the figure list. We invalidate the area of the new falling figure and the area of the top-right corner where the next figure is drawn:

[PRE23]

We delete the possible full rows and update the window:

[PRE24]

If the new falling figure is not valid from the very beginning, the game is over and `GameOver` is called:

[PRE25]

## Game over

The `GameOver` method presents the score and lets the user decide whether they want a new game. If they want a new game, it is initialized by the `NewGame` call. If the user does not want a new game, the call to the Win32 API function `PostQuitMessage` terminates the execution of the application.

Note that we call another version of the `Invalidate` method, without parameters. It invalidates the whole client area:

[PRE26]

The timer is inactive while the message is displayed:

[PRE27]

## New game

The `NewGame` method initializes the randomly chosen new falling and next figures, resets the score, and clears the game grid before activating the timer, as well as invalidates and updates the window, which makes the new falling figure starting to fall and the new game to begin:

[PRE28]

## Deleting and flashing rows

When deleting full rows, we loop through the rows, flashing and removing each full row. We increase the score and update the area of the row. Note that the rows start at the top of the grid. This means that we have to loop from the highest row to the lowest row in order to delete the row in the right order.

Note that if the row becomes flashed and deleted, we do not update the `row` variable since the deleted row will be replaced by the row above, which also needs to be examined:

[PRE29]

A row is considered full if it does not contain a white square:

[PRE30]

The flash effect is executed by redrawing the row in normal and inversed color (the `inverse` method is set) three times with an interval of 50 milliseconds. While doing this, it is especially important that we only invalidate the area of the chosen row. Otherwise, the whole window client area will be flashed:

[PRE31]

When deleting a row, we do not really delete it. Instead, we move each row above the deleted row one step downward and fill the top row with white squares. A complication is that we count rows from the top. This makes the lowest row on the screen the row with the highest index. This gives the appearance that we start from the bottom and remove every full row until we reach the top:

[PRE32]

## Closing the window

Finally, when the user wants to close the window by clicking in the cross on the top-right corner, we need to confirm that they really want to quit. If the `TryClose` method returns `true`, the window is closed:

[PRE33]

# The TetrisFigure class

In this application, there is the root `figure` class and one subclass for each type of falling figure. All figures can be moved sideways or rotated as a response to the user's requests. They are also moved downward by the timer.

There are seven figures, one for each color: red, brown, turquoise, green, yellow, blue, and purple. Each of them also has a unique shape. However, they all contain four squares. They can further be divided into three groups based on their ability to rotate. The red figure is the simplest one. It is a square and does not rotate at all. The brown, turquoise, and green figure can be rotated in vertical and horizontal directions, while the yellow, blue, and purple figures can be rotated in north, east, south, and west directions. For the red figure, it does not really matter since it does not rotate.

The `row` and `col` fields of the `TetrisFigure` class hold the center of the figure, which is marked by a cross in the illustrations of this section. The `color` field holds the color of the figure, and `direction` holds the current direction of the figure.

Finally, the `direction` array holds the relative positions of the three squares surrounding the marked square. There are four directions at most. Each direction holds three squares, which are the three remaining squares that are not the center of the figure. Each square holds two integers: the relative position of the center row and column.

The default constructor is needed to initialize the `fallingFigure` and `nextFigure` methods in the `TetrisWindow` class. The second constructor is protected since it is only called by its sub classes. Each figure has its own `TetrisFigure` subclass. Their constructors take a pointer to the color grid and define its color, start position, and figure patterns:

**TetrisFigure.h**

[PRE34]

The `TryMoveLeft`, `TryMoveRight`, `TryRotateClockwise`, `TryRotateClockwise`, `TryRotateAnticlockwise`, and `TryMoveDown` methods all try to move the figure. They call the `IsFigureValid` method, which checks whether the new location is valid, that is, it is not located outside the game grid or at a location already occupied. The `IsFigureValid` method, in turn, calls the `IsSquareValid` method for each of its four squares:

[PRE35]

There are two versions of the `IsFigureValid` method, where the first version is called by the `TetrisWindow` method and the other version is called by the preceding `try` methods in order to test whether a new location of the falling figure is valid:

[PRE36]

The `AddToGrid` method adds the four squares of the figure to the game grid:

[PRE37]

The `InvalidateFigure` method invalidates the area occupied by the figure, and the `DrawFigure` method draws the figure:

[PRE38]

The `gameGridPtr` field is a pointer to the game grid, which we access when we try to move a figure in order to decide whether its new location is valid. The `color` field is the color of the figure (red, brown, turquoise, green, yellow, blue, or purple). The `row`, `col`, and `direction` fields hold the current location and direction of the figure.

The `figureInfo` field holds the shape of the figure. The figure can hold up to four directions: north, east, south, and west. Remember that `row` and `col` hold the location of the figures. More specifically, they hold the location of the center square of the four squares constituting the figure (marked by a cross in the following illustrations). The other three squares are defined by integer pairs holding their locations relative to the center square.

Technically, `figureInfo` is an array of four pointers (one each for the directions north, east, south, and west). Each pointer points at an array of three integer pairs, holding the locations of the three squares relative to the center square:

[PRE39]

The default constructor is necessary because `fallingFigure` and `nextFigure` are member objects of the `TetrisWindow` class. However, they do not need to be initialized since their values are assigned one of the seven figures in the `figureList` array:

**TetrisFigure.cpp**

[PRE40]

The second constructor is called by the colored figure sub class constructor in order to initialize the figure. It takes a pointer to the main window and the game grid, the color of the figure, its start location and direction, and its location lists in the north, east, south, and west directions. Each of the lists holds three integer pairs representing the location of the squares relative to the center square:

[PRE41]

The assignment operator is necessary because the `fallingFigure` and `nextFigure` methods in the `TetrisWindow` class are copied from the figure list:

[PRE42]

The `TryMoveLeft`, `TryMoveRight`, `TryRotateClockwise`, and `TryRotateAnticlockwise` methods are called when the user presses the arrow keys. They try to move the figure and invalidate its previous and current area if they succeed:

[PRE43]

The `TryMoveDown` method is called by the timer when the player presses the Space key. It is also called by the `OnTimer` method in the `TetrisWindow` class; it returns a `Boolean` value indicating whether the movement succeeded:

[PRE44]

The first version of the `IsFigureValid` method is called by the `TetrisWindow` class and calls the second static version, with the current location and direction of the figure:

[PRE45]

The second version of the `IsFigureValid` method is called by the preceding `try` methods and checks if the figure is valid by calling the `IsSquareValid` method for each square in the figure. In order to do so, it needs to look up the relative positions of the included squares in the `figureInfo` method. The first value of the integer pairs is the row, and the second value is the column:

[PRE46]

The `IsSquareValid` method returns `true` if the given square is located inside the game grid and not already occupied. A square on the game board is considered unoccupied if it is white:

[PRE47]

When the falling figure has reached its final position, it is added to the game grid. It is performed by setting the figure's color to the squares in the game grid at its current location. A falling figure has reached its final position when it cannot fall any longer without colliding with an earlier figure or has reached the game grid's lower bound:

[PRE48]

When a figure has been moved, we need to redraw it. In order to avoid dazzle, we want to invalidate only its area, which is done by the `InvalidateFigure` method. We look up the rows and columns of the figure's four squares and call the `InvalidateSquare` method in the game grid for each of them:

[PRE49]

When drawing the figure, we need to look up the locations of the squares of the figure before we draw them in a way similar to the `InvalidateFigure` method:

[PRE50]

## The red figure

The red figure is one large square, built up by four smaller regular squares. It the simplest figure of the game since it does not change shape when rotating. This implies that we just need to look at one figure, shown as follows:

![The red figure](img/B05475_03_02.jpg)

This also implies that it is enough to define the squares for one direction and this to define the shape of the figure in all four directions:

**RedFigure.h**

[PRE51]

**RedFigure.cpp**

[PRE52]

The first integer pair (`rel row 0`, `rel col 1`) of the generic list represents the square to the right of the marked square, the second integer pair (`rel row 1`, `rel col 0`) represents the square below the marked square, and the third integer pair (`rel row 1`, `rel col 1`) represents the square below and to the right of the marked square. Note that the rows increase downward and the columns increase to the right.

## The brown figure

The brown figure can be oriented in a horizontal or vertical direction. It is initialized to vertical mode, as it can only be rotated into two directions. The north and south arrays are initialized with the vertical array and the east and west arrays are initialized with the horizontal array, as shown in the following image:

![The brown figure](img/B05475_03_03.jpg)

Since the row numbers increase downward and the column numbers increase to the right, the topmost square in the vertical direction (and the leftmost square in the horizontal direction) are represented by negative values:

**BrownFigure.h**

[PRE53]

**BrownFigure.cpp**

[PRE54]

## The turquoise figure

Similar to the brown figure, the turquoise figure can be rotated in a vertical and horizontal direction, as shown in the following figure:

![The turquoise figure](img/B05475_03_04.jpg)

**TurquoiseFigure.h**

[PRE55]

**TurquoiseFigure cpp**

[PRE56]

## The green figure

The green figure is mirrored in relation to the turquoise figure, shown as follows:

![The green figure](img/B05475_03_05.jpg)

**GreenFigure.h**

[PRE57]

**GreenFigure.cpp**

[PRE58]

## The yellow figure

The yellow figure can be rotated in a north, east, south, and west direction. It is initialized to the south, as shown in the following figure:

![The yellow figure](img/B05475_03_06.jpg)

**YellowFigure.h**

[PRE59]

**YellowFigure.cpp**

[PRE60]

## The blue figure

The blue figure can also be directed in all four directions. It is initialized to the south, as shown in the following figure:

![The blue figure](img/B05475_03_07.jpg)

**BlueFigure.h**

[PRE61]

**BlueFigure.cpp**

[PRE62]

## The purple figure

Finally, the purple figure is mirrored in relation to the blue figure and also initialized to the south, as shown in the following image:

![The purple figure](img/B05475_03_08.jpg)

**PurpleFigure.h**

[PRE63]

**PurpleFigure.cpp**

[PRE64]

# The GameGrid class

Finally, the `GameGrid` class is quite simple. It keeps track of the squares on the game board. The `gridArea` field is the portion of the total client area that is occupied by the grid:

**GameGrid.h**

[PRE65]

When called by the `TetrisWindow` constructor, the grid area will be set to (0, 20, 100, 100) units, placing it in the lower 80 percent of the client area of the window:

**GameGrid.cpp**

[PRE66]

When clearing the grid, we actually set every square to white:

[PRE67]

## Invalidating and drawing squares

The `DrawGameGrid` iterates through the squares of the grid. White squares are surrounded by white borders, while squares of every other color are surrounded by black borders. If the `inverseColor` parameter is true, the square color is inversed before drawn. This is useful when flashing rows:

[PRE68]

Note that the `InvalidateSquare` and `DrawSquare` methods add an offset. It is zero in all cases except when invalidating or drawing the next figure in the `TetrisWindow` class. Both methods calculate the size of the rows and columns of the grid and define the area of the square invalidated or drawn:

[PRE69]

# Summary

In this chapter, we developed a Tetris game. You looked into timing and randomization, as well as a new coordinate system, more advanced drawing, how to catch keyboard events, and how to write text.

In [Chapter 4](ch04.html "Chapter 4. Working with Shapes and Figures"), *Working with Shapes and Figures*, we will develop a drawing program capable of drawing lines, arrows, rectangles, and ellipses.