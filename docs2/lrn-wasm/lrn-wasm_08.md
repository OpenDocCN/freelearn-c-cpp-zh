# 使用 Emscripten 端口游戏

如第七章从零开始创建应用程序中所示，WebAssembly 在当前形式下仍然相对有限。Emscripten 提供了强大的 API，用于扩展 WebAssembly 的功能，为你的应用程序添加功能。将代码编译成 WebAssembly 模块和 JavaScript 粘合代码（而不是可执行文件），在某些情况下，可能只需要对现有的 C 或 C++ 源代码进行少量更改。

在本章中，我们将使用一个用 C++ 编写的代码库，该代码库被编译成传统的可执行文件，并更新代码以便它可以编译成 Wasm/JavaScript。我们还将添加一些额外的功能，以便更紧密地与浏览器集成。

到本章结束时，你将知道如何做以下事情：

+   将 C++ 代码库更新为编译成 Wasm 模块/JavaScript 粘合代码（而不是本地可执行文件）

+   使用 Emscripten 的 API 为 C++ 应用程序添加浏览器集成

+   使用适当的 `emcc` 标志构建多文件 C++ 项目

+   使用 `emrun` 在浏览器中运行和测试 C++ 应用程序

# 游戏概述

在本章中，我们将使用一个用 C++ 编写的俄罗斯方块克隆版，并更新代码以集成 Emscripten 并编译成 Wasm/JS。原始形式的代码库编译成可执行文件，使用了 SDL2，并且可以从命令行加载。在本节中，我们将简要回顾俄罗斯方块是什么，如何获取代码（无需从头编写），以及如何运行它。

# 什么是俄罗斯方块？

在俄罗斯方块游戏中，游戏的主要目标是旋转和移动各种形状的块（*Tetriminos*）在游戏区域（*井*或*矩阵*）内，以创建没有间隙的方块行。当创建了一行完整的方块时，它将从游戏区域中删除，并且你的得分增加一分。在我们的游戏版本中，没有胜利条件（尽管添加它很简单）。

理解游戏规则和机制很重要，因为代码使用了碰撞检测和得分等概念的算法。理解函数的目标有助于你理解代码内部。如果你需要复习你的俄罗斯方块技能，我建议你在网上试一试。你可以在[`emulatoronline.com/nes-games/classic-tetris/`](https://emulatoronline.com/nes-games/classic-tetris/)上玩，无需安装 Adobe Flash。它看起来就像原始的任天堂版本：

![](img/60566c17-a951-44f8-a3f4-a1e2bf8c6115.png)

EmulatorOnline.com 上的经典俄罗斯方块

我们将使用的数据版本将不包含方块计数器、等级或分数（我们坚持使用行数），但它将以相同的方式运行。

# 源代码的来源

结果表明，搜索“俄罗斯方块 C++”会提供大量的教程和示例仓库供选择。为了保持我到目前为止使用的格式和命名约定，我将这些资源结合起来，创建了我自己的游戏版本。本章末尾的“进一步阅读”部分包含了这些资源的链接，如果你有兴趣了解更多。移植代码库的概念和过程适用于任何来源。关于这一点，让我们简要地讨论一下移植的一般情况。

# 关于移植的注意事项

将现有的代码库移植到 Emscripten 并不总是简单的事情。在评估一个 C、C++或 Rust 应用程序是否适合转换时，需要考虑几个变量。例如，使用多个第三方库或几个相当复杂的第三方库的游戏可能需要大量的努力。Emscripten 提供了以下常用的库作为默认选项：

+   `asio`：一个网络和底层 I/O 编程库

+   `Bullet`：一个实时碰撞检测和多物理模拟库

+   `Cocos2d`：一套开源、跨平台的游戏开发工具

+   `FreeType`：一个用于渲染字体的库

+   `HarfBuzz`：一个 OpenType 文本形状引擎

+   `libpng`：官方 PNG 参考库

+   `Ogg`：一种多媒体容器格式

+   `SDL2`：一个旨在提供对音频、键盘、鼠标、游戏手柄和图形硬件的低级访问的库

+   `SDL2_image`：一个图像文件加载库

+   `SDL2_mixer`：一个多通道音频混音库示例

+   `SDL2_net`：一个小型的跨平台网络库示例

+   `SDL2_ttf`：一个示例库，允许你在 SDL 应用程序中使用 TrueType 字体

+   `Vorbis`：一种通用的音频和音乐编码格式

+   `zlib`：一个无损数据压缩库

如果库还没有被移植，你需要自己进行移植。这将对社区有益，但需要大量的时间和资源投入。我们的俄罗斯方块示例仅使用 SDL2，这使得移植过程相对简单。

# 获取代码

本章的代码位于`learn-webassembly`仓库的`/chapter-08-tetris`文件夹中。在`/chapter-08-tetris`内部有两个目录：`/output-native`文件夹，其中包含原始（未移植）代码，以及`/output-wasm`文件夹，其中包含移植后的代码。

如果你想要在 VS Code 中使用任务功能进行原生构建步骤，你需要打开 VS Code 中的`/chapter-08-tetris/output-native`文件夹，而不是顶层的`/learn-webassembly`文件夹。

# 构建原生项目

`/cmake`文件夹和位于`/output-native`文件夹中的`CMakeLists.txt`文件是构建项目所必需的。`README.md`文件包含在每个平台上将代码运行起来的说明。构建项目对于完成移植过程不是必需的。安装所需依赖项并使项目在你的平台上成功构建的过程可能既耗时又复杂。如果你仍然希望继续，你可以通过 VS Code 的任务功能构建可执行文件，从菜单中选择**任务** | **运行任务...**，然后在遵循`README.md`文件中的说明后，从列表中选择**构建可执行文件**。

# 游戏运行中的画面

如果你成功构建了项目，你应该可以通过从 VS Code 菜单中选择**任务** | **运行任务...**，然后从列表中选择**启动可执行任务**来运行它。如果一切顺利，你应该会看到如下内容：

![图片](img/34e1026f-bd47-4209-a611-671d81f98ede.png)

本地编译的游戏运行

我们的游戏版本没有失败条件；它只是为每清除一行增加一个 ROWS 计数。如果任何一个 Tetriminos 触碰到板的顶部，游戏结束，板子重置。这是一个基本的游戏实现，但增加的功能会增加复杂性和代码量。让我们更详细地回顾代码库。

# 深入了解代码库

现在你有了代码，你需要熟悉代码库。如果没有很好地理解你想要移植的代码，你将很难成功移植它。在本章中，我们将逐一介绍每个 C++类和头文件，并描述它们在应用程序中的作用。

# 将代码分解为对象

C++的设计是基于面向对象范式，这也是俄罗斯方块代码库用来简化应用程序管理的。代码库由 C++类文件组成

（`.cpp`）和头文件（`.h`），它们代表游戏上下文中的对象。我使用了“什么是俄罗斯方块？”部分的游戏玩法总结来推断我需要哪些对象。

游戏块（Tetriminos）和游戏区域（被称为井或矩阵）是作为类的良好候选。也许不那么直观，但同样有效，还有**游戏**本身。类不一定要像实际对象那样具体——它们非常适合存储共享代码。我是一个大粉丝，喜欢少打字，所以我选择使用`Piece`来表示 Tetrimino，用`Board`来表示游戏区域（尽管单词“井”更短，但它并不完全合适）。我创建了一个头文件来存储全局变量（`constants.h`），一个`Game`类来管理游戏玩法，以及一个`main.cpp`文件，它作为游戏的入口点。以下是`/src`文件夹的内容：

```cpp
├── board.cpp
├── board.h
├── constants.h
├── game.cpp
├── game.h
├── main.cpp
├── piece.cpp
└── piece.h
```

每个文件（除了`main.cpp`和`constants.h`）都有一个类（`.cpp`）和头文件（`.h`）。头文件允许你在多个文件之间重用代码，并防止代码重复。*进一步阅读*部分包含了你想要了解更多关于头文件资源的链接。`constants.h`文件在应用程序中的几乎所有其他文件中都被使用，所以让我们首先回顾一下。

# 常量文件

为了避免在代码库中散布令人困惑的*魔法数字*，我选择了一个包含我们将要使用的常量的头文件（`constants.h`）。此文件的 内容如下所示：

```cpp
#ifndef TETRIS_CONSTANTS_H
#define TETRIS_CONSTANTS_H

namespace Constants {
    const int BoardColumns = 10;
    const int BoardHeight = 720;
    const int BoardRows = 20;
    const int BoardWidth = 360;
    const int Offset = BoardWidth / BoardColumns;
    const int PieceSize = 4;
    const int ScreenHeight = BoardHeight + 50;
}

#endif // TETRIS_CONSTANTS_H
```

文件第一行的`#ifndef`语句是一个`#include`保护，它防止在编译过程中多次包含头文件。这些保护在应用程序的所有头文件中都被使用。每个这些常量的用途将在我们逐步查看每个类时变得清晰。我首先包含它，是为了提供关于各种元素大小及其相互关系的背景信息。

让我们继续到代表游戏各个方面的各种类。`Piece`类代表最低级别的对象，所以我们将从这里开始，逐步向上到`Board`和`Game`类。

# 片件类

片件，或称为*Tetrimino*，是可以在棋盘上移动和旋转的元素。有七种 Tetriminos——每种都由一个字母表示，并对应一种颜色：

![](img/bc6eabd2-b522-4990-9973-6d5432055b3d.png)

Tetrimino 颜色，来自维基百科

我们需要一种方式来定义每个片件在形状、颜色和当前方向上的特性。每个片件有四个不同的方向（以 90 度递增），这导致所有片件共有 28 种总变化。颜色不会改变，所以只需要分配一次。考虑到这一点，让我们首先看一下头文件（`piece.h`）：

```cpp
#ifndef TETRIS_PIECE_H
#define TETRIS_PIECE_H

#include <SDL2/SDL.h>
#include "constants.h"

class Piece {
 public:
  enum Kind { I = 0, J, L, O, S, T, Z };

  explicit Piece(Kind kind);

  void draw(SDL_Renderer *renderer);
  void move(int columnDelta, int rowDelta);
  void rotate();
  bool isBlock(int column, int row) const;
  int getColumn() const;
  int getRow() const;

 private:
  Kind kind_;
  int column_;
  int row_;
  int angle_;
};

#endif // TETRIS_PIECE_H
```

游戏使用 SDL2 来渲染各种图形元素和处理键盘输入，这就是为什么我们在`draw()`函数中传递`SDL_Renderer`的原因。你将在`Game`类中看到 SDL2 的使用，但就现在来说，只需知道它的包含即可。头文件定义了`Piece`类的接口；让我们回顾一下`piece.cpp`中的实现。我们将逐节查看代码并描述其功能。

# 构造函数和 draw()函数

代码的第一部分定义了`Piece`类的构造函数和`draw()`函数：

```cpp
#include "piece.h"

using namespace Constants;

Piece::Piece(Piece::Kind kind) :
    kind_(kind),
    column_(BoardColumns / 2 - PieceSize / 2),
    row_(0),
    angle_(0) {
}

void Piece::draw(SDL_Renderer *renderer) {
    switch (kind_) {
        case I:
            SDL_SetRenderDrawColor(renderer,
                /* Cyan: */ 45, 254, 254, 255);
            break;
        case J:
            SDL_SetRenderDrawColor(renderer,
                /* Blue: */ 11, 36, 251, 255);
            break;
        case L:
            SDL_SetRenderDrawColor(renderer,
                /* Orange: */ 253, 164, 41, 255);
            break;
        case O:
            SDL_SetRenderDrawColor(renderer,
                /* Yellow: */ 255, 253, 56, 255);
            break;
       case S:
            SDL_SetRenderDrawColor(renderer,
                /* Green: */ 41, 253, 47, 255);
            break;
        case T:
            SDL_SetRenderDrawColor(renderer,
                /* Purple: */ 126, 15, 126, 255);
            break;
        case Z:
            SDL_SetRenderDrawColor(renderer,
                /* Red: */ 252, 13, 28, 255);
            break;
        }

        for (int column = 0; column < PieceSize; ++column) {
            for (int row = 0; row < PieceSize; ++row) {
                if (isBlock(column, row)) {
                    SDL_Rect rect{
                        (column + column_) * Offset + 1,
                        (row + row_) * Offset + 1,
                        Offset - 2,
                        Offset - 2
                    };
                SDL_RenderFillRect(renderer, &rect);
            }
        }
    }
}
```

构造函数使用默认值初始化类。`BoardColumns`和`PieceSize`值来自`constants.h`文件。`BoardColumns`表示可以在棋盘上容纳的列数，在这个例子中是`10`。`PieceSize`常量表示一个片件在列中占据的面积或块，是`4`。分配给私有`columns_`变量的初始值代表棋盘的中心。

`draw()` 函数遍历板上的所有可能的行和列，并用与拼图类型相对应的颜色填充任何被拼图填充的单元格。是否由拼图填充单元格的判断是在 `isBlock()` 函数中进行的，我们将在下一节讨论。 

# `move()`、`rotate()` 和 `isBlock()` 函数

第二部分包含移动或旋转拼图以及确定其当前位置的逻辑：

```cpp
void Piece::move(int columnDelta, int rowDelta) {
    column_ += columnDelta;
    row_ += rowDelta;
}

void Piece::rotate() {
    angle_ += 3;
    angle_ %= 4;
}

bool Piece::isBlock(int column, int row) const {
    static const char *Shapes[][4] = {
        // I
        {
            " *  "
            " *  "
            " *  "
            " *  ",
            "    "
            "****"
            "    "
            "    ",
            " *  "
            " *  "
            " *  "
            " *  ",
            "    "
            "****"
            "    "
            "    ",
        },
        // J
        {
            "  * "
            "  * "
            " ** "
            "    ",
            "    "
            "*   "
            "*** "
            "    ",
            " ** "
            " *  "
            " *  "
            "    ",
            "    "
            "    "
            "*** "
            " *  ",
        },
        ...
    };
    return Shapes[kind_][angle_][column + row * PieceSize] == '*';
}

int Piece::getColumn() const {
 return column_;
}
int Piece::getRow() const {
 return row_;
}
```

`move()` 函数更新私有 `column_` 和 `row_` 变量的值，这决定了拼图在板上的位置。`rotate()` 函数将私有 `angle_` 变量的值设置为 `0`、`1`、`2` 或 `3`（这就是为什么使用 `%= 4`）。

判断显示哪种拼图、其位置和旋转的判断是在 `isBlock()` 函数中进行的。我省略了 `Shapes` 多维数组中的除前两个元素之外的所有元素，以避免文件杂乱，但实际代码中仍然存在剩余的五种拼图类型。我必须承认，这并不是最优雅的实现方式，但它完全符合我们的需求。

私有 `kind_` 和 `angle_` 值在 `Shapes` 数组中指定为维度，以选择四个相应的 `char*` 元素。这四个元素代表拼图的四种可能方向。如果你决定通过网上可用的一个俄罗斯方块教程（或查看 GitHub 上众多的俄罗斯方块仓库）进行工作，你会发现有几种不同的方法来计算单元格是否被拼图填充。我选择这种方法，因为它更容易可视化拼图。

# `getColumn()` 和 `getRow()` 函数

代码的最后部分包含获取拼图行和列的函数：

```cpp
int Piece::getColumn() const {
    return column_;
}

int Piece::getRow() const {
    return row_;
}
```

这些函数只是简单地返回私有 `column_` 或 `row_` 变量的值。现在你对 `Piece` 类有了更好的理解，让我们继续到 `Board`。

# `Board` 类

`Board` 包含 `Piece` 类的实例，并需要检测拼图之间的碰撞、行是否填满以及游戏是否结束。让我们从头文件（`board.h`）的内容开始：

```cpp
#ifndef TETRIS_BOARD_H
#define TETRIS_BOARD_H

#include <SDL2/SDL.h>
#include <SDL2/SDL2_ttf.h>
#include "constants.h"
#include "piece.h"

using namespace Constants;

class Board {
 public:
  Board();
  void draw(SDL_Renderer *renderer, TTF_Font *font);
  bool isCollision(const Piece &piece) const;
  void unite(const Piece &piece);

 private:
  bool isRowFull(int row);
  bool areFullRowsPresent();
  void updateOffsetRow(int fullRow);
  void displayScore(SDL_Renderer *renderer, TTF_Font *font);

  bool cells_[BoardColumns][BoardRows];
  int currentScore_;
};

#endif // TETRIS_BOARD_H
```

`Board` 类具有与 `Piece` 类相同的 `draw()` 函数，以及用于管理行和跟踪板上哪些单元格被填充的几个其他函数。`SDL2_ttf` 库用于在窗口底部渲染 ROWS：文本，显示当前得分（清除的行数）。现在，让我们看一下实现文件（`board.cpp`）的每个部分。

# 构造函数和 `draw()` 函数

代码的第一部分定义了 `Board` 类的构造函数和 `draw()` 函数：

```cpp
#include <sstream>
#include "board.h"

using namespace Constants;

Board::Board() : cells_{{ false }}, currentScore_(0) {}

void Board::draw(SDL_Renderer *renderer, TTF_Font *font) {
    displayScore(renderer, font);
    SDL_SetRenderDrawColor(
        renderer,
        /* Light Gray: */ 140, 140, 140, 255);
    for (int column = 0; column < BoardColumns; ++column) {
        for (int row = 0; row < BoardRows; ++row) {
            if (cells_[column][row]) {
                SDL_Rect rect{
                    column * Offset + 1,
                    row * Offset + 1,
                    Offset - 2,
                    Offset - 2
                };
                SDL_RenderFillRect(renderer, &rect);
            }
        }
    }
}
```

`Board` 构造函数将私有 `cells_` 和 `currentScore_` 变量的值初始化为默认值。`cells_` 变量是一个布尔值二维数组，第一维表示列，第二维表示行。如果一个部件占据特定的列和行，则数组中的相应值为 `true`。`draw()` 函数的行为类似于 `Piece` 的 `draw()` 函数，它用颜色填充包含部件的单元格。然而，此函数仅用浅灰色填充占据棋盘底部的部件所在的单元格，而不管部件的类型如何。

# `isCollision()` 函数

代码的第二部分包含检测碰撞的逻辑：

```cpp
bool Board::isCollision(const Piece &piece) const {
    for (int column = 0; column < PieceSize; ++column) {
        for (int row = 0; row < PieceSize; ++row) {
            if (piece.isBlock(column, row)) {
                int columnTarget = piece.getColumn() + column;
                int rowTarget = piece.getRow() + row;
                if (
                    columnTarget < 0
                    || columnTarget >= BoardColumns
                    || rowTarget < 0
                    || rowTarget >= BoardRows
                ) {
                    return true;
                }
                if (cells_[columnTarget][rowTarget]) return true;
            }
        }
    }
    return false;
}
```

`isCollision()` 函数遍历棋盘上的每个单元格，直到它到达由作为参数传递的 `&piece` 占据的一个单元格。如果该部件即将与棋盘的任一侧碰撞，或者它已经到达底部，则该函数返回 `true`，否则返回 `false`。

# `unite()` 函数

代码的第三部分包含在部件到达静止状态时将其与顶部行合并的逻辑：

```cpp
void Board::unite(const Piece &piece) {
    for (int column = 0; column < PieceSize; ++column) {
        for (int row = 0; row < PieceSize; ++row) {
            if (piece.isBlock(column, row)) {
                int columnTarget = piece.getColumn() + column;
                int rowTarget = piece.getRow() + row;
                cells_[columnTarget][rowTarget] = true;
            }
        }
    }

    // Continuously loops through each of the rows until no full rows are
    // detected and ensures the full rows are collapsed and non-full rows
    // are shifted accordingly:
    while (areFullRowsPresent()) {
        for (int row = BoardRows - 1; row >= 0; --row) {
            if (isRowFull(row)) {
                updateOffsetRow(row);
                currentScore_ += 1;
                for (int column = 0; column < BoardColumns; ++column) {
                    cells_[column][0] = false;
                }
            }
        }
    }
}

bool Board::isRowFull(int row) {
    for (int column = 0; column < BoardColumns; ++column) {
        if (!cells_[column][row]) return false;
    }
    return true;
}

bool Board::areFullRowsPresent() {
    for (int row = BoardRows - 1; row >= 0; --row) {
        if (isRowFull(row)) return true;
    }
    return false;
}

void Board::updateOffsetRow(int fullRow) {
    for (int column = 0; column < BoardColumns; ++column) {
        for (int rowOffset = fullRow - 1; rowOffset >= 0; --rowOffset) {
            cells_[column][rowOffset + 1] =
            cells_[column][rowOffset];
        }
    }
}
```

`unite()` 函数以及相应的 `isRowFull()`、`areFullRowsPresent()` 和 `updateOffsetRow()` 函数执行多个操作。它通过将适当的数组位置设置为 `true` 来更新私有 `cells_` 变量，以指定 `&piece` 参数占据的行和列。它还通过将相应的 `cells_` 数组位置设置为 `false` 来清除任何满行（所有列都填满）的行，并增加 `currentScore_`。在清除行之后，`cells_` 数组被更新，以将清除行上方的行向下移动 `1`。

# `displayScore()` 函数

代码的最后一部分在游戏窗口底部显示分数：

```cpp
void Board::displayScore(SDL_Renderer *renderer, TTF_Font *font) {
    std::stringstream message;
    message << "ROWS: " << currentScore_;
    SDL_Color white = { 255, 255, 255 };
    SDL_Surface *surface = TTF_RenderText_Blended(
        font,
        message.str().c_str(),
        white);
    SDL_Texture *texture = SDL_CreateTextureFromSurface(
        renderer,
        surface);
    SDL_Rect messageRect{ 20, BoardHeight + 15, surface->w, surface->h };
    SDL_FreeSurface(surface);
    SDL_RenderCopy(renderer, texture, nullptr, &messageRect);
    SDL_DestroyTexture(texture);
}
```

`displayScore()` 函数使用 `SDL2_ttf` 库在窗口底部（在棋盘下方）显示当前分数。`TTF_Font *font` 参数从 `Game` 类传递进来，以避免每次更新分数时都初始化字体。`stringstream message` 变量用于创建文本值，并将其设置在 `TTF_RenderText_Blended()` 函数中的 C `char*` 中。其余的代码在 `SDL_Rect` 上绘制文本，以确保其正确显示。

关于 `Board` 类的内容就到这里；让我们继续到 `Game` 类，看看所有这些是如何结合在一起的。

# `Game` 类

`Game` 类包含循环函数，允许您通过按键在棋盘上移动部件。以下是头文件（`game.h`）的内容：

```cpp
#ifndef TETRIS_GAME_H
#define TETRIS_GAME_H

#include <SDL2/SDL.h>
#include <SDL2/SDL2_ttf.h>
#include "constants.h"
#include "board.h"
#include "piece.h"

class Game {
 public:
  Game();
  ~Game();
  bool loop();

 private:
  Game(const Game &);
  Game &operator=(const Game &);

  void checkForCollision(const Piece &newPiece);
  void handleKeyEvents(SDL_Event &event);

  SDL_Window *window_;
  SDL_Renderer *renderer_;
  TTF_Font *font_;
  Board board_;
  Piece piece_;
  uint32_t moveTime_;
};

#endif // TETRIS_GAME_H
```

`loop()` 函数包含游戏逻辑，并根据事件管理状态。在 `private:` 标题下的前两行防止创建超过一个游戏实例，这可能导致内存泄漏。私有方法减少了 `loop()` 函数中的代码行数，从而简化了维护和调试。让我们继续到 `game.cpp` 中的实现。

# 构造函数和析构函数

代码的第一部分定义了在类实例加载（构造函数）和卸载（析构函数）时执行的操作：

```cpp
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include "game.h"

using namespace std;
using namespace Constants;

Game::Game() :
    // Create a new random piece:
    piece_{ static_cast<Piece::Kind>(rand() % 7) },
    moveTime_(SDL_GetTicks())
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        throw runtime_error(
            "SDL_Init(SDL_INIT_VIDEO): " + string(SDL_GetError()));
        }
        SDL_CreateWindowAndRenderer(
            BoardWidth,
            ScreenHeight,
            SDL_WINDOW_OPENGL,
            &window_,
            &renderer_);
        SDL_SetWindowPosition(
            window_,
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED);
        SDL_SetWindowTitle(window_, "Tetris");

    if (TTF_Init() != 0) {
        throw runtime_error("TTF_Init():" + string(TTF_GetError()));
    }
    font_ = TTF_OpenFont("PressStart2P.ttf", 18);
    if (font_ == nullptr) {
        throw runtime_error("TTF_OpenFont: " + string(TTF_GetError()));
    }
}

Game::~Game() {
    TTF_CloseFont(font_);
    TTF_Quit();
    SDL_DestroyRenderer(renderer_);
    SDL_DestroyWindow(window_);
    SDL_Quit();
}
```

构造函数代表应用程序的入口点，因此所有必需的资源都在其中分配和初始化。`TTF_OpenFont()` 函数引用的是从 Google Fonts 下载的 TrueType 字体文件，名为 Press Start 2P。您可以在 [`fonts.google.com/specimen/Press+Start+2P`](https://fonts.google.com/specimen/Press+Start+2P) 查看该字体。它在存储库的 `/resources` 文件夹中，并在构建项目时被复制到可执行文件的同一文件夹中。如果在初始化 SDL2 资源时发生任何错误，则会抛出一个带有错误详细信息的 `runtime_error`。析构函数 (`~Game()`) 在应用程序退出之前释放我们为 SDL2 和 `SDL2_ttf` 分配的资源。这样做是为了避免内存泄漏。

# loop() 函数

代码的最后部分表示 `Game::loop`：

```cpp
bool Game::loop() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_KEYDOWN:
                handleKeyEvents(event);
                break;
            case SDL_QUIT:
                return false;
            default:
                return true;
        }
    }

    SDL_SetRenderDrawColor(renderer_, /* Dark Gray: */ 58, 58, 58, 255);
    SDL_RenderClear(renderer_);
    board_.draw(renderer_, font_);
    piece_.draw(renderer_);

    if (SDL_GetTicks() > moveTime_) {
        moveTime_ += 1000;
        Piece newPiece = piece_;
        newPiece.move(0, 1);
        checkForCollision(newPiece);
    }
    SDL_RenderPresent(renderer_);
    return true;
}

void Game::checkForCollision(const Piece &newPiece) {
    if (board_.isCollision(newPiece)) {
        board_.unite(piece_);
        piece_ = Piece{ static_cast<Piece::Kind>(rand() % 7) };
        if (board_.isCollision(piece_)) board_ = Board();
    } else {
        piece_ = newPiece;
    }
}

void Game::handleKeyEvents(SDL_Event &event) {
    Piece newPiece = piece_;
    switch (event.key.keysym.sym) {
        case SDLK_DOWN:
            newPiece.move(0, 1);
            break;
        case SDLK_RIGHT:
            newPiece.move(1, 0);
            break;
        case SDLK_LEFT:
            newPiece.move(-1, 0);
            break;
        case SDLK_UP:
            newPiece.rotate();
            break;
        default:
            break;
     }
     if (!board_.isCollision(newPiece)) piece_ = newPiece;
}
```

`loop()` 函数只要 `SDL_QUIT` 事件没有触发就返回布尔值。每 `1` 秒，执行 `Piece` 和 `Board` 实例的 `draw()` 函数，并相应地更新棋盘上的棋子位置。左、右和下箭头键控制棋子的移动，而上箭头键将棋子旋转 90 度。按键的适当响应在 `handleKeyEvents()` 函数中处理。`checkForCollision()` 函数确定活动棋子的新实例是否与棋盘的任一边碰撞，或者落在其他棋子上面。如果是这样，就会创建一个新的棋子。清除行（通过 `Board` 的 `unite()` 函数）的逻辑也在此函数中处理。我们几乎完成了！让我们继续到 `main.cpp` 文件。

# 主文件

与 `main.cpp` 相关的没有头文件，因为它的唯一目的是作为应用程序的入口点。实际上，该文件只有七行：

```cpp
#include "game.h"

int main() {
    Game game;
    while (game.loop());
    return 0;
}
```

当 `loop()` 函数返回 `false` 时，`while` 语句退出，这发生在 `SDL_QUIT` 事件触发时。这个文件所做的只是创建一个新的 `Game` 实例并启动循环。这就是代码库的全部内容；让我们开始迁移！

# 迁移到 Emscripten

你对代码库有很好的理解，现在是时候开始使用 Emscripten 进行移植了。幸运的是，我们可以利用一些浏览器的功能来简化代码，并完全移除第三方库。在本节中，我们将更新代码以编译为 Wasm 模块和 JavaScript *glue* 文件，并更新一些功能以利用浏览器。

# 准备移植

`/output-wasm` 文件夹包含最终结果，但我建议你创建 `/output-native` 文件夹的副本，这样你就可以跟随移植过程。已经为本地编译和 Emscripten 编译设置了 VS Code 任务。如果你遇到困难，可以始终参考 `/output-wasm` 的内容。确保你在 VS Code 中打开你的副本文件夹（文件 | 打开并选择你的副本文件夹），否则你将无法使用任务功能。

# 有什么变化？

这款游戏是移植的理想候选者，因为它使用了 SDL2，这是一个广泛使用的库，并且已经存在 Emscripten 移植版本。在编译步骤中包含 SDL2 只需要将一个额外的参数传递给 `emcc` 命令。`SDL2_ttf` 库的 Emscripten 移植版本也存在，但将其保留在代码库中并没有太多意义。它的唯一目的是将得分（清除的行数）以文本形式渲染。我们需要将 TTF 文件与应用程序一起包含，从而复杂化构建过程。Emscripten 提供了在 C++ 中使用 JavaScript 代码的方法，因此我们将采取一条更简单的路线：在 DOM 中显示得分。

除了更改现有代码外，我们还需要创建一个 HTML 和 CSS 文件来在浏览器中显示和样式化游戏。我们编写的 JavaScript 代码将非常少——我们只需要加载 Emscripten 模块，所有功能都在 C++ 代码库中处理。我们还需要添加几个 `<div>` 元素并相应地布局它们以显示得分。让我们开始移植！

# 添加网络资源

在你的项目文件夹中创建一个名为 `/public` 的文件夹。在 `/public` 文件夹中添加一个名为 `index.html` 的新文件，并填充以下内容：

```cpp
<!doctype html>
<html lang="en-us">
<head>
  <title>Tetris</title>
  <link rel="stylesheet" type="text/css" href="styles.css" />
</head>
<body>
  <div class="wrapper">
    <h1>Tetris</h1>
    <div>
      <canvas id="canvas"></canvas>
      <div class="scoreWrapper">
        <span>ROWS:</span><span id="score"></span>
      </div>
    </div>
  </div>
  <script type="application/javascript" src="img/index.js"></script>
  <script type="application/javascript">
    Module({ canvas: (() => document.getElementById('canvas'))() })
  </script>
</body>
</html>
```

在第一个 `<script>` 标签中正在加载的 `index.js` 文件尚不存在；它将在编译步骤中生成。让我们给元素添加一些样式。在 `/public` 文件夹中创建一个名为 `styles.css` 的文件，并填充以下内容：

```cpp
@import url("https://fonts.googleapis.com/css?family=Press+Start+2P");

* {
  font-family: "Press Start 2P", sans-serif;
}

body {
  margin: 24px;
}

h1 {
  font-size: 36px;
}

span {
  color: white;
  font-size: 24px;
}

.wrapper {
  display: flex;
  align-items: center;
  flex-direction: column;
}

.titleWrapper {
  display: flex;
  align-items: center;
  justify-content: center;
}

.header {
  font-size: 24px;
  margin-left: 16px;
}

.scoreWrapper {
  background-color: #3A3A3A;
  border-top: 1px solid white;
  padding: 16px 0;
  width: 360px;
}

span:first-child {
  margin-left: 16px;
  margin-right: 8px;
}
```

由于我们使用的 Press Start 2P 字体托管在 Google Fonts 上，我们可以将其导入以在网站上使用。此文件中的 CSS 规则处理简单的布局和样式。这就是我们需要创建的所有与网络相关的文件。现在，是时候更新 C++ 代码了。

# 移植现有代码

为了让 Emscripten 正确工作，我们只需要编辑几个文件。为了简单和紧凑，我们只包括受影响的代码部分（而不是整个文件）。让我们按照上一节相同的顺序处理文件，并从`constants.h`开始。

# 更新常数文件

我们将在 DOM 上显示清除行数，而不是在游戏窗口本身中显示，因此可以删除文件中的`ScreenHeight`常量。我们不再需要额外的空间来容纳分数文本：

```cpp
namespace Constants {
    const int BoardColumns = 10;
    const int BoardHeight = 720;
    const int BoardRows = 20;
    const int BoardWidth = 360;
    const int Offset = BoardWidth / BoardColumns;
    const int PieceSize = 4;
    // const int ScreenHeight = BoardHeight + 50; <----- Delete this line
}
```

不需要对`Piece`类文件（`piece.cpp`/`piece.h`）进行更改。但是，我们需要更新`Board`类。让我们从头文件（`board.h`）开始。从底部开始，逐步向上更新`displayScore()`函数。在`index.html`文件的`<body>`部分，有一个`id="score"`的`<span>`元素。我们将使用`emscripten_run_script`命令更新此元素以显示当前分数。因此，`displayScore()`函数变得更短。以下是前后对比。

这是`Board`类`displayScore()`函数的原始版本：

```cpp
void Board::displayScore(SDL_Renderer *renderer, TTF_Font *font) {
    std::stringstream message;
    message << "ROWS: " << currentScore_;
    SDL_Color white = { 255, 255, 255 };
    SDL_Surface *surface = TTF_RenderText_Blended(
        font,
        message.str().c_str(),
        white);
    SDL_Texture *texture = SDL_CreateTextureFromSurface(
        renderer,
        surface);
    SDL_Rect messageRect{ 20, BoardHeight + 15, surface->w, surface->h };
    SDL_FreeSurface(surface);
    SDL_RenderCopy(renderer, texture, nullptr, &messageRect);
    SDL_DestroyTexture(texture);
 }
```

这是`displayScore()`函数的移植版本：

```cpp
void Board::displayScore(int newScore) {
    std::stringstream action;
    action << "document.getElementById('score').innerHTML =" << newScore;
    emscripten_run_script(action.str().c_str());
 }
```

`emscripten_run_script`动作简单地找到 DOM 中的`<span>`元素，并将`innerHTML`设置为当前分数。在这里我们不能使用`EM_ASM()`函数，因为 Emscripten 不识别`document`对象。由于我们可以在类中访问私有的`currentScore_`变量，我们将把`draw()`函数中的`displayScore()`调用移动到`unite()`函数中。这限制了调用`displayScore()`的次数，确保函数仅在分数实际改变时被调用。我们只需要添加一行代码就能完成这个任务。现在`unite()`函数看起来是这样的：

```cpp
void Board::unite(const Piece &piece) {
    for (int column = 0; column < PieceSize; ++column) {
        for (int row = 0; row < PieceSize; ++row) {
            if (piece.isBlock(column, row)) {
                int columnTarget = piece.getColumn() + column;
                int rowTarget = piece.getRow() + row;
                cells_[columnTarget][rowTarget] = true;
            }
        }
    }

    // Continuously loops through each of the rows until no full rows are
    // detected and ensures the full rows are collapsed and non-full rows
    // are shifted accordingly:
    while (areFullRowsPresent()) {
        for (int row = BoardRows - 1; row >= 0; --row) {
            if (isRowFull(row)) {
                updateOffsetRow(row);
                currentScore_ += 1;
                for (int column = 0; column < BoardColumns; ++column) {
                    cells_[column][0] = false;
                }
            }
        }
        displayScore(currentScore_); // <----- Add this line
    }
}
```

由于我们不再使用`SDL2_ttf`库，我们可以更新`draw()`函数签名并移除`displayScore()`函数调用。以下是更新后的`draw()`函数：

```cpp
void Board::draw(SDL_Renderer *renderer/*, TTF_Font *font */) {
                                        // ^^^^^^^^^^^^^^ <-- Remove this argument
    // displayScore(renderer, font); <----- Delete this line
    SDL_SetRenderDrawColor(
        renderer,
        /* Light Gray: */ 140, 140, 140, 255);
    for (int column = 0; column < BoardColumns; ++column) {
        for (int row = 0; row < BoardRows; ++row) {
            if (cells_[column][row]) {
                SDL_Rect rect{
                    column * Offset + 1,
                    row * Offset + 1,
                    Offset - 2,
                    Offset - 2
                };
                SDL_RenderFillRect(renderer, &rect);
            }
        }
    }
 }
```

`displayScore()`函数调用已被从函数的第一行移除，同时移除了`TTF_Font *font`参数。让我们在构造函数中添加对`displayScore()`的调用，以确保游戏结束时重置并开始新游戏时初始值设置为`0`：

```cpp
Board::Board() : cells_{{ false }}, currentScore_(0) {
    displayScore(0); // <----- Add this line
}
```

这节课的类文件就到这里。由于我们更改了`displayScore()`和`draw()`函数的签名，并移除了对`SDL2_ttf`的依赖，我们需要更新头文件。从`board.h`中移除以下行：

```cpp
#ifndef TETRIS_BOARD_H
#define TETRIS_BOARD_H

#include <SDL2/SDL.h>
// #include <SDL2/SDL2_ttf.h> <----- Delete this line
#include "constants.h"
#include "piece.h"

using namespace Constants;

class Board {
 public:
  Board();
  void draw(SDL_Renderer *renderer /*, TTF_Font *font */);
                                    // ^^^^^^^^^^^^^^ <-- Remove this
  bool isCollision(const Piece &piece) const;
  void unite(const Piece &piece);

 private:
  bool isRowFull(int row);
  bool areFullRowsPresent();
  void updateOffsetRow(int fullRow);
  void displayScore(SDL_Renderer *renderer, TTF_Font *font);
                                         // ^^^^^^^^^^^^^^ <-- Remove this
  bool cells_[BoardColumns][BoardRows];
  int currentScore_;
};

#endif // TETRIS_BOARD_H
```

我们正在稳步前进！我们需要做的最后一个更改也是最大的一个。现有的代码库中有一个 `Game` 类，它管理应用程序逻辑，还有一个 `main.cpp` 文件，在 `main()` 函数中调用 `Game.loop()` 函数。循环机制是一个 while 循环，只要 `SDL_QUIT` 事件没有触发，它就会持续运行。我们需要改变我们的方法以适应 Emscripten。

Emscripten 提供了一个 `emscripten_set_main_loop` 函数，它接受一个 `em_callback_func` 循环函数、`fps` 和一个 `simulate_infinite_loop` 标志。我们不能包含 `Game` 类并将 `Game.loop()` 作为 `em_callback_func` 参数传递，因为构建将会失败。相反，我们将完全删除 `Game` 类并将逻辑移动到 `main.cpp` 文件中。将 `game.cpp` 的内容复制到 `main.cpp` 中（覆盖现有内容）并删除 `Game` 类文件（`game.cpp`/`game.h`）。由于我们没有为 `Game` 声明一个类，所以需要从函数中移除 `Game::` 前缀。构造函数和析构函数不再有效（它们不再是类的一部分），因此我们需要将逻辑移动到不同的位置。我们还需要重新排序文件，以确保我们的调用函数在调用函数之前。最终的结果如下：

```cpp
#include <emscripten/emscripten.h>
#include <SDL2/SDL.h>
#include <stdexcept>
#include "constants.h"
#include "board.h"
#include "piece.h"

using namespace std;
using namespace Constants;

static SDL_Window *window = nullptr;
static SDL_Renderer *renderer = nullptr;
static Piece currentPiece{ static_cast<Piece::Kind>(rand() % 7) };
static Board board;
static int moveTime;

void checkForCollision(const Piece &newPiece) {
    if (board.isCollision(newPiece)) {
        board.unite(currentPiece);
        currentPiece = Piece{ static_cast<Piece::Kind>(rand() % 7) };
        if (board.isCollision(currentPiece)) board = Board();
    } else {
        currentPiece = newPiece;
    }
}

void handleKeyEvents(SDL_Event &event) {
    Piece newPiece = currentPiece;
    switch (event.key.keysym.sym) {
        case SDLK_DOWN:
            newPiece.move(0, 1);
            break;
        case SDLK_RIGHT:
            newPiece.move(1, 0);
            break;
        case SDLK_LEFT:
            newPiece.move(-1, 0);
            break;
        case SDLK_UP:
            newPiece.rotate();
            break;
        default:
            break;
    }
    if (!board.isCollision(newPiece)) currentPiece = newPiece;
}

void loop() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_KEYDOWN:
                handleKeyEvents(event);
                break;
            case SDL_QUIT:
                break;
            default:
                break;
        }
    }

    SDL_SetRenderDrawColor(renderer, /* Dark Gray: */ 58, 58, 58, 255);
    SDL_RenderClear(renderer);
    board.draw(renderer);
    currentPiece.draw(renderer);

    if (SDL_GetTicks() > moveTime) {
        moveTime += 1000;
        Piece newPiece = currentPiece;
        newPiece.move(0, 1);
        checkForCollision(newPiece);
    }
    SDL_RenderPresent(renderer);
}

int main() {
    moveTime = SDL_GetTicks();
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        throw std::runtime_error("SDL_Init(SDL_INIT_VIDEO)");
    }
    SDL_CreateWindowAndRenderer(
        BoardWidth,
        BoardHeight,
        SDL_WINDOW_OPENGL,
        &window,
        &renderer);

    emscripten_set_main_loop(loop, 0, 1);

    SDL_DestroyRenderer(renderer);
    renderer = nullptr;
    SDL_DestroyWindow(window);
    window = nullptr;
    SDL_Quit();
    return 0;
}
```

`handleKeyEvents()` 和 `checkForCollision()` 函数没有变化；我们只是将它们移动到了文件顶部。`loop()` 函数的返回类型从 `bool` 改为了 `void`，这是由 `emscripten_set_main_loop` 所要求的。最后，构造函数和析构函数中的代码被移动到了 `main()` 函数中，并且移除了对 `SDL2_ttf` 的所有引用。我们不再使用调用 `Game` 的 `loop()` 函数的 while 语句，而是使用 `emscripten_set_main_loop(loop, 0, 1)` 调用。我们将文件顶部的 `#include` 语句修改为适应 Emscripten、SDL2 以及我们的 `Board` 和 `Piece` 类。这就是所有的更改——现在该配置构建并测试游戏了。

# 构建和运行游戏

代码更新并准备好所需的网络资源后，是时候构建并测试游戏了。编译步骤与本书中的前几个示例类似，但我们将使用不同的技术来运行游戏。在本节中，我们将配置构建任务以适应 C++ 文件，并使用 Emscripten 提供的功能运行应用程序。

# 使用 VS Code 任务进行构建

我们将以两种方式配置构建：使用 VS Code 任务和 Makefile。如果你更喜欢使用 VS Code 以外的编辑器，Makefiles 会很方便。`/.vscode/tasks.json` 文件已经包含了构建项目所需的任务。Emscripten 构建步骤是默认的（也提供了一套原生构建任务）。让我们逐一查看 `tasks` 数组中的每个任务，并回顾正在发生的事情。第一个任务是在构建之前删除任何现有的编译输出文件：

```cpp
{
  "label": "Remove Existing Web Files",
  "type": "shell",
  "command": "rimraf",
  "options": {
    "cwd": "${workspaceRoot}/public"
  },
  "args": [
    "index.js",
    "index.wasm"
  ]
}
```

第二个任务使用 `emcc` 命令执行构建：

```cpp
{
  "label": "Build WebAssembly",
  "type": "shell",
  "command": "emcc",
  "args": [
    "--bind", "src/board.cpp", "src/piece.cpp", "src/main.cpp",
    "-std=c++14",
    "-O3",
    "-s", "WASM=1",
    "-s", "USE_SDL=2",
    "-s", "MODULARIZE=1",
    "-o", "public/index.js"
  ],
  "group": {
    "kind": "build",
    "isDefault": true
  },
  "problemMatcher": [],
  "dependsOn": ["Remove Existing Web Files"]
}
```

相关参数放在同一行上。`args` 数组中唯一的新增和不熟悉的参数是 `--bind` 参数，对应于 `.cpp` 文件。这告诉 Emscripten 所有在 `--bind` 之后的所有文件都是构建项目所必需的。通过从菜单中选择“任务”|“运行构建任务...”或使用键盘快捷键 *Cmd*/*Ctrl + Shift + B* 来测试构建。构建需要几秒钟，但终端会在编译过程完成后通知您。如果成功，您应该在 `/public` 文件夹中看到 `index.js` 和 `index.wasm` 文件。

# 使用 Makefile 构建

如果您不想使用 VS Code，您可以使用 Makefile 完成与 VS Code 任务相同的目标。在您的项目文件夹中创建一个名为 `Makefile` 的文件，并填充以下内容（确保文件使用的是制表符，而不是空格）：

```cpp
# This allows you to just run the "make" command without specifying
# arguments:
.DEFAULT_GOAL := build

# Specifies which files to compile as part of the project:
CPP_FILES = $(wildcard src/*.cpp)

# Flags to use for Emscripten emcc compile command:
FLAGS = -std=c++14 -O3 -s WASM=1 -s USE_SDL=2 -s MODULARIZE=1 \
        --bind $(CPP_FILES)

# Name of output (the .wasm file is created automatically):
OUTPUT_FILE = public/index.js

# This is the target that compiles our executable
compile: $(CPP_FILES)
    emcc  $(FLAGS) -o $(OUTPUT_FILE)

# Removes the existing index.js and index.wasm files:
clean:
    rimraf $(OUTPUT_FILE)
    rimraf public/index.wasm

# Removes the existing files and builds the project:
build: clean compile
    @echo "Build Complete!"
```

正在执行的操作与 VS Code 任务相同，只是使用了更通用的工具以不同的格式。默认的构建步骤在文件中设置，因此您可以在项目文件夹内运行以下命令来编译项目：

```cpp
make
```

现在您已经有了编译好的 Wasm 文件和 JavaScript 粘合代码，让我们尝试运行游戏。

# 运行游戏

我们将不使用 serve 或 `browser-sync`，而是使用 Emscripten 工具链的一个内置功能，`emrun`。它提供了额外的优势，即捕获 `stdout` 和 `stderr`（如果您将 `--emrun` 链接器标志传递给 `emcc` 命令），并在需要时将它们打印到终端。我们不会使用 `--emrun` 标志，但有一个本地 Web 服务器可用，无需安装任何额外的依赖项，这是一个值得注意的附加功能。在您的项目文件夹内打开一个终端实例，并运行以下命令来启动游戏：

```cpp
emrun --browser chrome --no_emrun_detect public/index.html
```

如果您在开发时使用的是 `firefox` 浏览器，您可以指定 `firefox`。`--no_emrun_detect` 标志会隐藏终端中显示的 HTML 页面不是 `emrun` 兼容的消息。如果您导航到 `http://localhost:6931/index.html`，您应该看到以下内容：

![](img/700df992-90f3-4452-84da-49e770e1a1c7.png)

浏览器中运行的俄罗斯方块

尝试旋转和移动方块以确保一切正常工作。当您成功清除一行时，行数应该增加一。您也可能注意到，如果您离棋盘边缘太近，您将无法旋转某些方块。恭喜您，您已成功将 C++ 游戏移植到 Emscripten！

# 摘要

在本章中，我们将使用 SDL2 编写的 Tetris 克隆移植到 Emscripten，以便在浏览器中使用 WebAssembly 运行。我们涵盖了 Tetris 的规则以及它们如何映射到现有代码库中的逻辑。我们还逐个审查了现有代码库中的每个文件，并确定了为了成功编译为 Wasm 文件和 JavaScript 粘合代码需要做出的更改。更新现有代码后，我们创建了所需的 HTML 和 CSS 文件，然后使用适当的 `emcc` 标志配置了构建步骤。构建完成后，游戏使用 Emscripten 的 `emrun` 命令运行。

在 第九章，*与 Node.js 集成*，我们将讨论如何将 WebAssembly 集成到 Node.js 中以及这种集成带来的好处。

# 问题

1.  Tetris 中的部件叫什么？

1.  为什么选择不将现有的 C++ 代码库移植到 Emscripten 的一个原因是什么？

1.  我们使用了什么工具来本地编译游戏（例如，生成可执行文件）？

1.  `constants.h` 文件的作用是什么？

1.  为什么我们能够消除 SDL2_ttf 库？

1.  我们使用了哪个 Emscripten 函数来开始运行游戏？

1.  我们在 `emcc` 命令中添加了哪个参数来构建游戏，它有什么作用？

1.  `emrun` 相较于 `serve` 和 Browsersync 这样的工具有什么优势？

# 进一步阅读

+   C++ 的头文件：[`www.sitesbay.com/cpp/cpp-header-files`](https://www.sitesbay.com/cpp/cpp-header-files)

+   SDL2 Tetris on GitHub: [`github.com/andwn/sdl2-tetris`](https://github.com/andwn/sdl2-tetris)

+   Tetris on GitHub: [`github.com/abesary/tetris`](https://github.com/abesary/tetris)

+   Tetris - Linux on GitHub: [`github.com/abesary/tetris-linux`](https://github.com/abesary/tetris-linux)
