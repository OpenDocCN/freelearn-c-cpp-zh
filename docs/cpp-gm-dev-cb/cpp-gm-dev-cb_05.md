# 第五章：事件驱动编程-制作你的第一个 2D 游戏

在本章中，将涵盖以下食谱：

+   开始制作 Windows 游戏

+   使用 Windows 类和句柄

+   创建你的第一个窗口

+   添加键盘和鼠标控制以及文本输出

+   使用 GDI 与 Windows 资源

+   使用对话框和控件

+   使用精灵

+   使用动画精灵

# 介绍

Windows 编程是创建适当应用程序的开始。我们需要知道如何将我们的游戏打包成一个可执行文件，以便我们的所有资源，如图像、模型和声音，都能得到适当的加密并打包成一个文件。通过这样做，我们确保文件是安全的，并且在分发时不能被非法复制。然而，应用程序仍然在运行时使用这些文件。

Windows 编程也标志着开始理解 Windows 消息泵。这个系统非常重要，因为所有主要的编程范式都依赖于这个原则，特别是当我们进行事件驱动的编程时。

事件驱动编程的主要原则是，基于事件，我们应该处理一些东西。这里需要理解的概念是我们多久检查一次事件以及我们应该多久处理它们。

# 开始制作 Windows 游戏

在我们开始制作 Windows 游戏之前，首先要了解的是窗口或消息框是如何绘制的。我们需要了解 Windows 提供给我们的众多内置函数以及我们可以使用的不同回调函数。

## 准备工作

通过这个食谱，你需要一台运行 Windows 的机器。你还需要在 Windows 机器上安装一个可用的 Visual Studio 副本。没有其他先决条件。

## 如何做...

在这个食谱中，我们将看到在 Windows 中创建消息框是多么容易。我们可以创建不同类型的消息框，只需要几行代码。按照以下步骤进行：

1.  打开 Visual Studio。

1.  创建一个新的 C++项目。

1.  选择一个 Win32 Windows 应用程序。

1.  添加一个名为`Source.cpp`的源文件。

1.  将以下代码添加到`Source.cpp`中：

```cpp
#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <windowsx.h>

int WINAPI WinMain(HINSTANCE _hInstance,
  HINSTANCE _hPrevInstance,
  LPSTR _lpCmdLine,
  int _iCmdShow)
{
  MessageBox(NULL, L"My first message",
    L"My first Windows Program",
    MB_OK | MB_ICONEXCLAMATION);

  return (0);
}
```

## 它是如何工作的...

`WINMAIN()`是 Windows 程序的入口点。在这个例子中，我们使用了内置函数来创建一个消息框。`windows.h`包含了我们需要调用 Windows API 中的内置函数的所有必要文件。消息框通常用于显示一些内容。我们还可以将消息框与默认的 Windows 声音关联起来。消息框的显示也可以在很大程度上进行控制。我们需要使用正确类型的参数来实现这一点。

我们还可以使用其他类型的消息框：

+   **MB_OK**：一个按钮，带有**OK**消息

+   **MB_OKCANCEL**：两个按钮，带有**OK**，**Cancel**消息

+   **MB_RETRYCANCEL**：两个按钮，带有**Retry**，**Cancel**消息

+   **MB_YESNO**：两个按钮，带有**Yes**，**No**消息

+   **MB_YESNOCANCEL**：三个按钮，带有**Yes**，**No**，**Cancel**消息

+   **MB_ABORTRETRYIGNORE**：三个按钮，带有**Abort**，**Retry**，**Ignore**消息

+   **MB_ICONEXCLAIMATION**：出现一个感叹号图标

+   **MB_ICONINFORMATION**：出现一个信息图标

+   **MB_ICONQUESTION**：出现一个问号图标

+   **MB_ICONSTOP**：出现一个停止标志图标

像所有良好的 Win32 或 Win64 API 函数一样，`MessageBox`返回一个值，让我们知道发生了什么。

# 使用 Windows 类和句柄

为了编写游戏，我们不需要对 Windows 编程了解很多。我们需要知道的是如何打开一个窗口，如何处理消息，以及如何调用主游戏循环。Windows 应用程序的第一个任务是创建一个窗口。在窗口创建后，我们可以做各种其他事情，比如处理事件和处理回调。这些事件最终被游戏框架用来在屏幕上显示精灵，并使它们可移动和交互，以便我们可以玩游戏。

## 准备工作

您需要在 Windows 机器上安装一个可用的 Visual Studio 副本。

## 如何做…

在这个教程中，我们将发现使用 Windows 类和句柄有多么容易。

1.  打开 Visual Studio。

1.  创建一个新的 C++项目。

1.  选择一个 Win32 Windows 应用程序。

1.  添加一个名为`Source.cpp`的源文件。

1.  向其中添加以下代码行：

```cpp
   // This only adds the necessary windows files and not all of them
#define WIN32_LEAN_AND_MEAN

#include <windows.h>   // Include all the windows headers.
#include <windowsx.h>  // Include useful macros.

#define WINDOW_CLASS_NAME L"WINCLASS1"

void GameLoop()
{
  //One frame of game logic occurs here...
}

LRESULT CALLBACK WindowProc(HWND _hwnd,
  UINT _msg,
  WPARAM _wparam,
  LPARAM _lparam)
{
  // This is the main message handler of the system.
  PAINTSTRUCT ps; // Used in WM_PAINT.
  HDC hdc;        // Handle to a device context.

  // What is the message?
  switch (_msg)
  {
  case WM_CREATE:
  {
            // Do initialization stuff here.

            // Return Success.
            return (0);
  }
    break;

  case WM_PAINT:
  {
           // Simply validate the window.
           hdc = BeginPaint(_hwnd, &ps);

           // You would do all your painting here...

           EndPaint(_hwnd, &ps);

           // Return Success.
           return (0);
  }
    break;

  case WM_DESTROY:
  {
             // Kill the application, this sends a WM_QUIT message.
             PostQuitMessage(0);

             // Return success.
             return (0);
  }
    break;

  default:break;
  } // End switch.

  // Process any messages that we did not take care of...

  return (DefWindowProc(_hwnd, _msg, _wparam, _lparam));
}

int WINAPI WinMain(HINSTANCE _hInstance,
  HINSTANCE _hPrevInstance,
  LPSTR _lpCmdLine,
  int _nCmdShow)
{
  WNDCLASSEX winclass; // This will hold the class we create.
  HWND hwnd;           // Generic window handle.
  MSG msg;             // Generic message.

  // First fill in the window class structure.
  winclass.cbSize = sizeof(WNDCLASSEX);
  winclass.style = CS_DBLCLKS | CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
  winclass.lpfnWndProc = WindowProc;
  winclass.cbClsExtra = 0;
  winclass.cbWndExtra = 0;
  winclass.hInstance = _hInstance;
  winclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
  winclass.hCursor = LoadCursor(NULL, IDC_ARROW);
  winclass.hbrBackground =
    static_cast<HBRUSH>(GetStockObject(WHITE_BRUSH));
  winclass.lpszMenuName = NULL;
  winclass.lpszClassName = WINDOW_CLASS_NAME;
  winclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

  // register the window class
  if (!RegisterClassEx(&winclass))
  {
    return (0);
  }

  // create the window
  hwnd = CreateWindowEx(NULL, // Extended style.
    WINDOW_CLASS_NAME,      // Class.
    L"My first Window",   // Title.
    WS_OVERLAPPEDWINDOW | WS_VISIBLE,
    0, 0,                    // Initial x,y.
    400, 400,                // Initial width, height.
    NULL,                   // Handle to parent.
    NULL,                   // Handle to menu.
    _hInstance,             // Instance of this application.
    NULL);                  // Extra creation parameters.

  if (!(hwnd))
  {
    return (0);
  }

  // Enter main event loop
  while (true)
  {
    // Test if there is a message in queue, if so get it.
    if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
    {
      // Test if this is a quit.
      if (msg.message == WM_QUIT)
      {
        break;
      }

      // Translate any accelerator keys.
      TranslateMessage(&msg);
      // Send the message to the window proc.
      DispatchMessage(&msg);
    }

    // Main game processing goes here.
    GameLoop(); //One frame of game logic occurs here...
  }

  // Return to Windows like this...
  return (static_cast<int>(msg.wParam));
}
```

## 它是如何工作的…

整个`typedef`结构`_WNDCLASSEX`的定义如下：

```cpp
{
UINT cbSize;          // Size of this structure.
UINT style;           // Style flags.
WNDPROC lpfnWndProc;  // Function pointer to handler.
int cbClsExtra;       // Extra class info.
int cbWndExtra;       // Extra window info.
HANDLE hInstance;     // The instance of the app.
HICON hIcon;          // The main icon.
HCURSOR hCursor;      // The cursor for the window.
HBRUSH hbrBackground; // The Background brush to paint the window.
LPCTSTR lpszMenuName; // The name of the menu to attach.
LPCTSTR lpszClassName;// The name of the class itself.
HICON hIconSm;        // The handle of the small icon.
} WNDCLASSEX;
```

Windows API 为我们提供了多个 API 回调。我们需要决定拦截哪个消息以及在该消息泵中处理哪些信息。例如，`WM_CREATE`是一个 Windows 创建函数。我们应该在这里执行大部分初始化。同样，`WM_DESTROY`是我们需要销毁已创建对象的地方。我们需要使用 GDI 对象在窗口上绘制框和其他东西。我们还可以在窗口上显示自己的光标和图标。

# 创建您的第一个窗口

创建一个窗口是 Windows 编程的第一步。所有我们的精灵和其他对象都将绘制在这个窗口的顶部。有一个标准的绘制窗口的方法。因此，这部分代码将在所有使用 Windows 编程绘制东西的程序中重复。

## 准备工作

您需要在 Windows 机器上安装一个可用的 Visual Studio 副本。

## 如何做…

在这个教程中，我们将发现创建一个窗口有多么容易：

1.  打开 Visual Studio。

1.  创建一个新的 C++项目。

1.  选择一个 Win32 Windows 应用程序。

1.  添加一个名为`Source.cpp`的源文件。

1.  向其中添加以下代码行：

```cpp
#define WIN32_LEAN_AND_MEAN

#include <windows.h>   // Include all the windows headers.
#include <windowsx.h>  // Include useful macros.
#include "resource.h"

#define WINDOW_CLASS_NAME L"WINCLASS1"

void GameLoop()
{
  //One frame of game logic occurs here...
}

LRESULT CALLBACK WindowProc(HWND _hwnd,
  UINT _msg,
  WPARAM _wparam,
  LPARAM _lparam)
{
  // This is the main message handler of the system.
  PAINTSTRUCT ps; // Used in WM_PAINT.
  HDC hdc;        // Handle to a device context.

  // What is the message?
  switch (_msg)
  {
  case WM_CREATE:
  {
            // Do initialization stuff here.

            // Return Success.
            return (0);
  }
    break;

  case WM_PAINT:
  {
           // Simply validate the window.
           hdc = BeginPaint(_hwnd, &ps);

           // You would do all your painting here...

           EndPaint(_hwnd, &ps);

           // Return Success.
           return (0);
  }
    break;

  case WM_DESTROY:
  {
             // Kill the application, this sends a WM_QUIT message.
             PostQuitMessage(0);

             // Return success.
             return (0);
  }
    break;

  default:break;
  } // End switch.

  // Process any messages that we did not take care of...

  return (DefWindowProc(_hwnd, _msg, _wparam, _lparam));
}

int WINAPI WinMain(HINSTANCE _hInstance,
  HINSTANCE _hPrevInstance,
  LPSTR _lpCmdLine,
  int _nCmdShow)
{
  WNDCLASSEX winclass; // This will hold the class we create.
  HWND hwnd;           // Generic window handle.
  MSG msg;             // Generic message.

  HCURSOR hCrosshair = LoadCursor(_hInstance, MAKEINTRESOURCE(IDC_CURSOR2));

  // First fill in the window class structure.
  winclass.cbSize = sizeof(WNDCLASSEX);
  winclass.style = CS_DBLCLKS | CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
  winclass.lpfnWndProc = WindowProc;
  winclass.cbClsExtra = 0;
  winclass.cbWndExtra = 0;
  winclass.hInstance = _hInstance;
  winclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
  winclass.hCursor = LoadCursor(_hInstance, MAKEINTRESOURCE(IDC_CURSOR2));
  winclass.hbrBackground =
    static_cast<HBRUSH>(GetStockObject(WHITE_BRUSH));
  winclass.lpszMenuName = NULL;
  winclass.lpszClassName = WINDOW_CLASS_NAME;
  winclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

  // register the window class
  if (!RegisterClassEx(&winclass))
  {
    return (0);
  }

  // create the window
  hwnd = CreateWindowEx(NULL, // Extended style.
    WINDOW_CLASS_NAME,      // Class.
    L"Packt Publishing",   // Title.
    WS_OVERLAPPEDWINDOW | WS_VISIBLE,
    0, 0,                    // Initial x,y.
    400, 400,                // Initial width, height.
    NULL,                   // Handle to parent.
    NULL,                   // Handle to menu.
    _hInstance,             // Instance of this application.
    NULL);                  // Extra creation parameters.

  if (!(hwnd))
  {
    return (0);
  }

  // Enter main event loop
  while (true)
  {
    // Test if there is a message in queue, if so get it.
    if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
    {
      // Test if this is a quit.
      if (msg.message == WM_QUIT)
      {
        break;
      }

      // Translate any accelerator keys.
      TranslateMessage(&msg);
      // Send the message to the window proc.
      DispatchMessage(&msg);
    }

    // Main game processing goes here.
    GameLoop(); //One frame of game logic occurs here...
  }

  // Return to Windows like this...
  return (static_cast<int>(msg.wParam));
}
```

## 它是如何工作的…

在这个例子中，我们使用了标准的 Windows API 回调。我们查询传递的消息参数，并根据此拦截并执行适当的操作。我们使用`WM_PAINT`消息为我们绘制窗口，使用`WM_DESTROY`消息销毁当前窗口。要绘制窗口，我们需要一个设备上下文的句柄，然后我们可以适当地使用`BeginPaint`和`EndPaint`。在主结构中，我们需要填充 Windows 结构并指定需要加载的当前光标和图标。在这里，我们可以指定我们将使用什么颜色刷来绘制窗口。最后，指定窗口的大小并注册。之后，我们需要不断地查看消息，将其翻译，并最终将其分派到 Windows 过程中。

# 添加键盘和鼠标控制以及文本输出

在视频游戏中，我们最需要的一个重要的东西是与之交互的人机界面。最常见的界面设备是键盘和鼠标。因此，了解它们的工作原理以及如何检测按键和移动是非常重要的。同样重要的是要知道如何在屏幕上显示特定的文本；这对于调试和 HUD 实现非常有用。

## 准备工作

对于这个教程，您需要一个带有可用的 Visual Studio 副本的 Windows 机器。

## 如何做…

在这个教程中，我们将发现检测键盘和鼠标事件有多么容易：

1.  打开 Visual Studio。

1.  创建一个新的 C++项目。

1.  选择一个 Win32 Windows 应用程序。

1.  添加一个名为`Source.cpp`的源文件。

1.  向其中添加以下代码行：

```cpp
#define WIN32_LEAN_AND_MEAN 
#include <windows.h> //Include all the Windows headers.
#include <windowsx.h> //Include useful macros.
#include <strstream>
#include <string>
#include <cmath>

#include "resource.h"
#include "mmsystem.h"
//also uses winmm.lib

using namespace std;

#define WINDOW_CLASS_NAME "WINCLASS1"

HINSTANCE g_hInstance;
//RECT g_rect;
const RECT* g_prect;

POINT g_pos;
int g_iMouseX;
int g_iMouseY;

bool IS_LEFT_PRESSED  = 0;
bool IS_RIGHT_PRESSED = 0;
bool IS_UP_PRESSED    = 0;
bool IS_DOWN_PRESSED  = 0;

bool IS_LMB_PRESSED = 0;
bool IS_RMB_PRESSED = 0;
bool IS_MMB_PRESSED = 0;

int LAST_KEYPRESS_ASCII = 0;

float ang = 0.0f;

template<typename T>
std::string ToString(const T& _value)
{
  std::strstream theStream;
  theStream << _value << std::ends;
  return (theStream.str());
}

//GameLoop
void GameLoop()
{
  ang += 0.0005f;
  //One frame of game logic goes here
}

//Event handling (window handle, message handle --
LRESULT CALLBACK WindowProc(HWND _hwnd, UINT _msg, WPARAM _wparam, LPARAM _lparam)
{
  //This is the main message handler of the system.
  PAINTSTRUCT ps; //Used in WM_PAINT
  HDC hdc;        // Handle to a device context.

      if ((GetAsyncKeyState(VK_LEFT) & 0x8000) == 0x8000)
      {
        IS_LEFT_PRESSED = TRUE;
      }
      else
      {
        IS_LEFT_PRESSED = FALSE;
      }

      if ((GetAsyncKeyState(VK_RIGHT) & 0x8000) == 0x8000)
      {
        IS_RIGHT_PRESSED = TRUE;
      }
      else
      {
        IS_RIGHT_PRESSED = FALSE;
      }

      if ((GetAsyncKeyState(VK_UP) & 0x8000) == 0x8000)
      {
        IS_UP_PRESSED = TRUE;
      }
      else
      {
        IS_UP_PRESSED = FALSE;
      }

      if ((GetAsyncKeyState(VK_DOWN) & 0x8000) == 0x8000)
      {
        IS_DOWN_PRESSED = TRUE;
      }
      else
      {
        IS_DOWN_PRESSED = FALSE;
      }

  //What is the message?
  switch(_msg)
  {
  case WM_CREATE:
    {
      //Do initialisation stuff here.
      //Return success.
      return(0);
    }
    break;

  case WM_PAINT:
    {
      ////Simply validate the window.
      hdc = BeginPaint(_hwnd, &ps);

      InvalidateRect( _hwnd,
        g_prect,
        FALSE);              

      string temp;
      int iYDrawPos = 15;

      COLORREF red = RGB(255,0,0);

      SetTextColor(hdc, red);

      temp = "MOUSE X: ";
      temp += ToString((g_pos.x));
      while (temp.size() < 14)
      {
        temp += " ";
      }

      TextOut(hdc,30,iYDrawPos,temp.c_str(), static_cast<int>(temp.size()));

      iYDrawPos+= 13;

      temp = "MOUSE Y: ";
      temp += ToString((g_pos.y));
      while (temp.size() < 14)
      {
        temp += " ";
      }

      TextOut(hdc,30,iYDrawPos,temp.c_str(), static_cast<int>(temp.size()));

      iYDrawPos+= 13;

      if (IS_LEFT_PRESSED == TRUE)
      {
        TextOut(hdc,30,iYDrawPos,"LEFT IS PRESSED", 24);
      }
      else
      {
        TextOut(hdc,30,iYDrawPos,"LEFT IS NOT PRESSED ", 20);
      }
      iYDrawPos+= 13;
      if (IS_RIGHT_PRESSED == TRUE)
      {
        TextOut(hdc,30,iYDrawPos,"RIGHT IS PRESSED", 25);
      }
      else
      {
        TextOut(hdc,30,iYDrawPos,"RIGHT IS NOT PRESSED ", 21);
      }
      iYDrawPos+= 13;
      if (IS_DOWN_PRESSED == TRUE)
      {
        TextOut(hdc,30,iYDrawPos,"DOWN IS PRESSED", 24);
      }
      else
      {
        TextOut(hdc,30,iYDrawPos,"DOWN IS NOT PRESSED", 20);
      }
      iYDrawPos+= 13;
      if (IS_UP_PRESSED == TRUE)
      {
        TextOut(hdc,30,iYDrawPos,"UP IS PRESSED", 22);
      }
      else
      {
        TextOut(hdc,30,iYDrawPos,"UP IS NOT PRESSED ", 18);
      }

//      TextOut(hdc, static_cast<int>(200 +(sin(ang)*200)), static_cast<int>(200 +(sin(ang)*200))) , "O", 1);

      EndPaint(_hwnd, &ps);

      //Return success.
      return(0);
    }
    break;

  case WM_DESTROY:
    {
      //Kill the application, this sends a WM_QUIT message.
      PostQuitMessage(0);

      //Return Sucess.
      return(0);
    }
    break;

  case WM_MOUSEMOVE:
    {
      GetCursorPos(&g_pos);
      // here is your coordinates
      //int x=pos.x;
      //int y=pos.y;
      return(0);
    }
  break;

  case WM_COMMAND:
    {

    }

  default:break;
  } // End switch.

  //Process any messages we didn't take care of...

  return(DefWindowProc(_hwnd, _msg, _wparam, _lparam));
}

int WINAPI WinMain(HINSTANCE _hInstance, HINSTANCE _hPrevInstance, LPSTR _lpCmdLine, int _nCmdShow)
{
  WNDCLASSEX winclass; ///This will hold the class we create
  HWND hwnd; //Generic window handle.
  MSG msg; //Generic message.

  g_hInstance = _hInstance;

  //First fill in the window class structure
  winclass.cbSize         = sizeof(WNDCLASSEX);
  winclass.style          = CS_DBLCLKS | CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
  winclass.lpfnWndProc    = WindowProc;
  winclass.cbClsExtra     = 0;
  winclass.cbWndExtra     = 0;
  winclass.hInstance      = _hInstance;
  winclass.hIcon          = LoadIcon(g_hInstance, MAKEINTRESOURCE(IDI_ICON1));
  winclass.hCursor        = NULL;
  winclass.hbrBackground  = static_cast<HBRUSH>(GetStockObject(WHITE_BRUSH));
  winclass.lpszMenuName   = MAKEINTRESOURCE(IDR_MENU1);
  winclass.lpszClassName  = WINDOW_CLASS_NAME;
  winclass.hIconSm        = LoadIcon(g_hInstance, MAKEINTRESOURCE(IDI_ICON1));

  //Register the window class
  if (!RegisterClassEx(&winclass))
  { //perhaps use log manager here
    return(0);
  }

  //Create the window
  if (!(hwnd = CreateWindowEx(NULL, //Extended style.
                WINDOW_CLASS_NAME, //Class
                "Recipe4", //Title
                WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                400,300, //Initial X, Y
                400,400, //Initial width, height.
                NULL, //handle to parent.
                NULL, //handle to menu
                _hInstance, //Instance of this application
                NULL))) //Extra creation parameters
  {
    return (0);
  }

  RECT rect;  
  rect.left = 0;
  rect.right = 400;
  rect.top = 0;
  rect.bottom = 400;
  g_prect = &rect;

  //Enter main event loop
  while (TRUE)
  {
    //Test if there is a message in queue, if so get it.
    if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
    {
      //Test if this is a quit
      if (msg.message == WM_QUIT)
      {
        break;
      }

      //Translate any accelerator keys
      TranslateMessage(&msg);
      //Send the message to the window proc.
      DispatchMessage(&msg);
    }

    //Main game processing goes here.
    GameLoop(); //One frame of game logic goes here...
  }
  //Return to Windows like this...
  return(static_cast<int>(msg.wParam));
}
```

## 它是如何工作的…

创建并注册主窗口。在回调函数中，我们使用一个名为`GetAsyncKeyState(VK_KEYNAME)`的函数来检测按下了哪个键。之后，我们执行按位`AND`操作来检查最后一次按键是否也是相同的键，并且它是否实际被按下。然后，我们有不同的布尔参数来检测按键按下的状态并存储它们。代码可能以更好的方式结构化，但这是理解如何检测按键按下的最简单方式。为了检测鼠标移动坐标，我们在`WM_MOUSEMOVE`中使用一个名为`GetCursorPos`的函数，并相应地获取屏幕上的*x*和*y*坐标。最后，我们需要在屏幕上显示所有这些信息。为此，我们在屏幕上创建一个矩形。在那个矩形中，我们需要使用一个名为`TextOut`的函数来显示该信息。`TextOut`函数使用设备上下文的句柄、*x*和*y*坐标以及要显示的消息。

# 使用 Windows 资源与 GDI

**图形** **设备接口**（**GDI**）允许我们使用位图、图标、光标等进行有趣的事情。如果我们没有实现其他渲染替代方案，如 OpenGL 或 DirectX，GDI 将用作渲染替代方案。

## 准备工作

对于这个教程，您需要一台运行 Windows 的计算机，并安装了可用的 Visual Studio 副本。

## 如何做…

在这个教程中，我们将发现使用 Windows GDI 加载资源有多么容易：

1.  打开 Visual Studio。

1.  创建一个新的 C++项目。

1.  选择一个 Win32 Windows 应用程序。

1.  右键单击**资源文件**，并从**添加资源**子部分添加一个新的光标。

1.  将自动为您创建一个`resource.h`文件。

1.  添加一个名为`Source.cpp`的源文件，并向其中添加以下代码：

```cpp
#define WIN32_LEAN_AND_MEAN

#include <windows.h>   // Include all the windows headers.
#include <windowsx.h>  // Include useful macros.
#include "resource.h"

#define WINDOW_CLASS_NAME L"WINCLASS1"

void GameLoop()
{
  //One frame of game logic occurs here...
}

LRESULT CALLBACK WindowProc(HWND _hwnd,
  UINT _msg,
  WPARAM _wparam,
  LPARAM _lparam)
{
  // This is the main message handler of the system.
  PAINTSTRUCT ps; // Used in WM_PAINT.
  HDC hdc;        // Handle to a device context.

  // What is the message?
  switch (_msg)
  {
  case WM_CREATE:
  {
            // Do initialization stuff here.

            // Return Success.
            return (0);
  }
    break;

  case WM_PAINT:
  {
           // Simply validate the window.
           hdc = BeginPaint(_hwnd, &ps);

           // You would do all your painting here...

           EndPaint(_hwnd, &ps);

           // Return Success.
           return (0);
  }
    break;

  case WM_DESTROY:
  {
             // Kill the application, this sends a WM_QUIT message.
             PostQuitMessage(0);

             // Return success.
             return (0);
  }
    break;

  default:break;
  } // End switch.

  // Process any messages that we did not take care of...

  return (DefWindowProc(_hwnd, _msg, _wparam, _lparam));
}

int WINAPI WinMain(HINSTANCE _hInstance,
  HINSTANCE _hPrevInstance,
  LPSTR _lpCmdLine,
  int _nCmdShow)
{
  WNDCLASSEX winclass; // This will hold the class we create.
  HWND hwnd;           // Generic window handle.
  MSG msg;             // Generic message.

  HCURSOR hCrosshair = LoadCursor(_hInstance, MAKEINTRESOURCE(IDC_CURSOR2));

  // First fill in the window class structure.
  winclass.cbSize = sizeof(WNDCLASSEX);
  winclass.style = CS_DBLCLKS | CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
  winclass.lpfnWndProc = WindowProc;
  winclass.cbClsExtra = 0;
  winclass.cbWndExtra = 0;
  winclass.hInstance = _hInstance;
  winclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
  winclass.hCursor = LoadCursor(_hInstance, MAKEINTRESOURCE(IDC_CURSOR2));
  winclass.hbrBackground =
    static_cast<HBRUSH>(GetStockObject(WHITE_BRUSH));
  winclass.lpszMenuName = NULL;
  winclass.lpszClassName = WINDOW_CLASS_NAME;
  winclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

  // register the window class
  if (!RegisterClassEx(&winclass))
  {
    return (0);
  }

  // create the window
  hwnd = CreateWindowEx(NULL, // Extended style.
    WINDOW_CLASS_NAME,      // Class.
    L"PacktUp Publishing",   // Title.
    WS_OVERLAPPEDWINDOW | WS_VISIBLE,
    0, 0,                    // Initial x,y.
    400, 400,                // Initial width, height.
    NULL,                   // Handle to parent.
    NULL,                   // Handle to menu.
    _hInstance,             // Instance of this application.
    NULL);                  // Extra creation parameters.

  if (!(hwnd))
  {
    return (0);
  }

  // Enter main event loop
  while (true)
  {
    // Test if there is a message in queue, if so get it.
    if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
    {
      // Test if this is a quit.
      if (msg.message == WM_QUIT)
      {
        break;
      }

      // Translate any accelerator keys.
      TranslateMessage(&msg);
      // Send the message to the window proc.
      DispatchMessage(&msg);
    }

    // Main game processing goes here.
    GameLoop(); //One frame of game logic occurs here...
  }

  // Return to Windows like this...
  return (static_cast<int>(msg.wParam));
}
```

## 它是如何工作的…

加载新的光标是最容易实现的任务。我们需要修改以下行：

```cpp
winclass.hCursor = LoadCursor(_hInstance, MAKEINTRESOURCE(IDC_CURSOR2))
```

如果我们在这里指定 null，将加载默认的 Windows 光标。相反，我们可以加载刚刚创建的光标。确保在`resource.h`中指定光标的引用名称为`IDC_CURSOR2`。我们可以随意命名它，但是我们需要从`LoadCursor`函数中调用适当的引用。`MAKEINTRESOURCE`使我们能够从源代码中关联到资源文件。同样，如果需要，我们可以加载多个光标并在运行时切换它们。加载其他资源，如图标和其他位图时，也使用相同的过程。当我们修改资源文件时，相应的`resource.h`文件必须关闭，否则将无法编辑它。同样，如果我们想手动编辑`source.h`文件，我们需要关闭相应的`.rc`或资源文件。

# 使用对话框和控件

对话框是 Windows 编程的强制特性之一。如果我们正在创建一个完整的应用程序，总会有一个阶段需要以某种形式使用对话框。对话框可以是编辑框、单选按钮、复选框等形式。对话框有两种形式：模态和非模态。模态对话框需要立即响应，而非模态对话框更像是浮动框，不需要立即响应。

## 准备工作

要完成这个教程，您需要一台运行 Windows 的计算机。您还需要在 Windows 计算机上安装一个可用的 Visual Studio 副本。不需要其他先决条件。

## 如何做…

在这个教程中，我们将发现创建对话框有多么容易。

1.  打开 Visual Studio。

1.  创建一个新的 C++项目。

1.  选择一个 Win32 窗口应用程序。

1.  创建一个新的资源文件。

1.  选择对话框作为资源的类型。

1.  以您想要的任何方式编辑框。

1.  将创建一个相应的`resource.h`文件。

1.  将以下代码添加到`Source.cpp`文件中：

```cpp
#define WIN32_LEAN_AND_MEAN

#include <windows.h>   // Include all the windows headers.
#include <windowsx.h>  // Include useful macros.
#include "resource.h"
#define WINDOW_CLASS_NAME L"WINCLASS1"

void GameLoop()
{
  //One frame of game logic occurs here...
}

BOOL CALLBACK AboutDlgProc(HWND hDlg, UINT msg, WPARAM wparam, LPARAM lparam)
{
  switch (msg)
  {
    case WM_INITDIALOG:
      break;
    case WM_COMMAND:
      switch (LOWORD(wparam))
      {
      case IDOK:
        EndDialog(
          hDlg, //Handle to the dialog to end.
          0);   //Return code.
        break;
      case IDCANCEL:
        EndDialog(
          hDlg, //Handle to the dialog to end.
          0);   //Return code.
        break;
      default:
        break;
      }

  }

  return true;
}

LRESULT CALLBACK WindowProc(HWND _hwnd,
  UINT _msg,
  WPARAM _wparam,
  LPARAM _lparam)
{
  // This is the main message handler of the system.
  PAINTSTRUCT ps; // Used in WM_PAINT.
  HDC hdc;        // Handle to a device context.

  // What is the message?
  switch (_msg)
  {
  case WM_CREATE:
  {
            // Do initialization stuff here.

            // Return Success.
            return (0);
  }
    break;

  case WM_PAINT:
  {
           // Simply validate the window.
           hdc = BeginPaint(_hwnd, &ps);

           // You would do all your painting here...

           EndPaint(_hwnd, &ps);

           // Return Success.
           return (0);
  }
    break;

  case WM_DESTROY:
  {
             // Kill the application, this sends a WM_QUIT message.
             PostQuitMessage(0);

             // Return success.
             return (0);
  }
    break;

  default:break;
  } // End switch.

  // Process any messages that we did not take care of...

  return (DefWindowProc(_hwnd, _msg, _wparam, _lparam));
}

int WINAPI WinMain(HINSTANCE _hInstance,
  HINSTANCE _hPrevInstance,
  LPSTR _lpCmdLine,
  int _nCmdShow)
{
  WNDCLASSEX winclass; // This will hold the class we create.
  HWND hwnd;           // Generic window handle.
  MSG msg;             // Generic message.

  // First fill in the window class structure.
  winclass.cbSize = sizeof(WNDCLASSEX);
  winclass.style = CS_DBLCLKS | CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
  winclass.lpfnWndProc = WindowProc;
  winclass.cbClsExtra = 0;
  winclass.cbWndExtra = 0;
  winclass.hInstance = _hInstance;
  winclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
  winclass.hCursor = LoadCursor(NULL, IDC_ARROW);
  winclass.hbrBackground =
    static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
  winclass.lpszMenuName = NULL;
  winclass.lpszClassName = WINDOW_CLASS_NAME;
  winclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

  // register the window class
  if (!RegisterClassEx(&winclass))
  {
    return (0);
  }

  // create the window
  hwnd = CreateWindowEx(NULL, // Extended style.
    WINDOW_CLASS_NAME,      // Class.
    L"My first Window",   // Title.
    WS_OVERLAPPEDWINDOW | WS_VISIBLE,
    0, 0,                    // Initial x,y.
    1024, 980,                // Initial width, height.
    NULL,                   // Handle to parent.
    NULL,                   // Handle to menu.
    _hInstance,             // Instance of this application.
    NULL);                  // Extra creation parameters.

  if (!(hwnd))
  {
    return (0);
  }

  DialogBox(_hInstance, MAKEINTRESOURCE(IDD_DIALOG1), hwnd, AboutDlgProc);

  // Enter main event loop
  while (true)
  {
    // Test if there is a message in queue, if so get it.
    if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
    {
      // Test if this is a quit.
      if (msg.message == WM_QUIT)
      {
        break;
      }

      // Translate any accelerator keys.
      TranslateMessage(&msg);
      // Send the message to the window proc.
      DispatchMessage(&msg);
    }

    // Main game processing goes here.
    GameLoop(); //One frame of game logic occurs here...
  }

  // Return to Windows like this...
  return (static_cast<int>(msg.wParam));
}
```

## 它是如何工作的…

在`resource.h`文件自动为我们创建之后，我们可以手动编辑它以适当地命名对话框。创建主窗口后，我们需要获取窗口句柄，然后调用对话框框函数，如下所示：

```cpp
DialogBox(_hInstance, MAKEINTRESOURCE(IDD_DIALOG1), hwnd, AboutDlgProc)
```

与主窗口回调非常相似，对话框框也有自己的回调。我们需要相应地拦截消息并执行我们的操作。`BOOL CALLBACK AboutDlgProc`是我们可以使用的回调。我们有一个类似的初始化消息。对于我们的对话框，大多数拦截将发生在`WM_COMMAND`中。根据`wparam`参数，我们需要进行切换，以便知道我们是否点击了**OK**按钮还是**CANCEL**按钮，并采取适当的步骤。

# 使用精灵

要开发任何 2D 游戏，我们都需要精灵。精灵是计算机图形的元素，可以保持在屏幕上，被操纵和被动画化。GDI 允许我们使用精灵来创建我们的游戏。可能游戏中的所有资源都将是精灵，从 UI 到主要角色等等。

## 准备就绪

对于这个示例，您需要一台运行 Windows 的机器，并且安装了 Visual Studio 的工作副本。

## 如何做...

在这个示例中，我们将了解如何在游戏中使用精灵：

1.  打开 Visual Studio。

1.  创建一个新的 C++项目。

1.  创建一个新的资源类型。

1.  选择**Sprite**选项作为新的资源类型。

1.  添加以下源文件：`backbuffer.h/cpp`，`Clock.h/cpp`，`Game.h/.cpp`，`sprite.h/cpp`和`Utilities.h`。

1.  将以下代码添加到`backbuffer.h`中：

```cpp
#pragma once

#if !defined(__BACKBUFFER_H__)
#define __BACKBUFFER_H__

// Library Includes
#include <Windows.h>

// Local Includes

// Types

// Constants

// Prototypes
class CBackBuffer
{
  // Member Functions
public:
  CBackBuffer();
  ~CBackBuffer();

  bool Initialise(HWND _hWnd, int _iWidth, int _iHeight);

  HDC GetBFDC() const;

  int GetHeight() const;
  int GetWidth() const;

  void Clear();
  void Present();

protected:

private:
  CBackBuffer(const CBackBuffer& _kr);
  CBackBuffer& operator= (const CBackBuffer& _kr);

  // Member Variables
public:

protected:
  HWND m_hWnd;
  HDC m_hDC;
  HBITMAP m_hSurface;
  HBITMAP m_hOldObject;
  int m_iWidth;
  int m_iHeight;

private:

};

#endif    // __BACKBUFFER_H__
```

1.  将以下代码添加到`backbuffer.cpp`中：

```cpp
// Library Includes

// Local Includes

// This include
#include "BackBuffer.h"

// Static Variables

// Static Function Prototypes

// Implementation

CBackBuffer::CBackBuffer()
: m_hWnd(0)
, m_hDC(0)
, m_hSurface(0)
, m_hOldObject(0)
, m_iWidth(0)
, m_iHeight(0)
{

}

CBackBuffer::~CBackBuffer()
{
  SelectObject(m_hDC, m_hOldObject);

  DeleteObject(m_hSurface);
  DeleteObject(m_hDC);
}

bool
CBackBuffer::Initialise(HWND _hWnd, int _iWidth, int _iHeight)
{
  m_hWnd = _hWnd;

  m_iWidth = _iWidth;
  m_iHeight = _iHeight;

  HDC hWindowDC = ::GetDC(m_hWnd);

  m_hDC = CreateCompatibleDC(hWindowDC);

  m_hSurface = CreateCompatibleBitmap(hWindowDC, m_iWidth, m_iHeight);

  ReleaseDC(m_hWnd, hWindowDC);

  m_hOldObject = static_cast<HBITMAP>(SelectObject(m_hDC, m_hSurface));

  HBRUSH brushWhite = static_cast<HBRUSH>(GetStockObject(LTGRAY_BRUSH));
  HBRUSH oldBrush = static_cast<HBRUSH>(SelectObject(m_hDC, brushWhite));

  Rectangle(m_hDC, 0, 0, m_iWidth, m_iHeight);

  SelectObject(m_hDC, oldBrush);

  return (true);
}

void
CBackBuffer::Clear()
{
  HBRUSH hOldBrush = static_cast<HBRUSH>(SelectObject(GetBFDC(), GetStockObject(LTGRAY_BRUSH)));

  Rectangle(GetBFDC(), 0, 0, GetWidth(), GetHeight());

  SelectObject(GetBFDC(), hOldBrush);
}

HDC
CBackBuffer::GetBFDC() const
{
  return (m_hDC);
}

int
CBackBuffer::GetWidth() const
{
  return (m_iWidth);
}

int
CBackBuffer::GetHeight() const
{
  return (m_iHeight);
}

void
CBackBuffer::Present()
{
  HDC hWndDC = ::GetDC(m_hWnd);

  BitBlt(hWndDC, 0, 0, m_iWidth, m_iHeight, m_hDC, 0, 0, SRCCOPY);

  ReleaseDC(m_hWnd, hWndDC);
}
```

1.  将以下代码添加到`Clock.h`中：

```cpp
#pragma once

#if !defined(__CLOCK_H__)
#define __CLOCK_H__

// Library Includes

// Local Includes

// Types

// Constants

// Prototypes
class CClock
{
  // Member Functions
public:
  CClock();
  ~CClock();

  bool Initialise();

  void Process();

  float GetDeltaTick();

protected:

private:
  CClock(const CClock& _kr);
  CClock& operator= (const CClock& _kr);

  // Member Variables
public:

protected:
  float m_fTimeElapsed;
  float m_fDeltaTime;
  float m_fLastTime;
  float m_fCurrentTime;

private:

};

#endif    // __CLOCK_H__
```

1.  将以下代码添加到`Clock.cpp`中：

```cpp
// Library Includes
#include <windows.h>

// Local Includes
#include "Clock.h"

// Static Variables

// Static Function Prototypes

// Implementation

CClock::CClock()
: m_fTimeElapsed(0.0f)
, m_fDeltaTime(0.0f)
, m_fLastTime(0.0f)
, m_fCurrentTime(0.0f)
{

}

CClock::~CClock()
{

}

bool
CClock::Initialise()
{
  return (true);
}

void
CClock::Process()
{
  m_fLastTime = m_fCurrentTime;

  m_fCurrentTime = static_cast<float>(timeGetTime());

  if (m_fLastTime == 0.0f)
  {
    m_fLastTime = m_fCurrentTime;
  }

  m_fDeltaTime = m_fCurrentTime - m_fLastTime;

  m_fTimeElapsed += m_fDeltaTime;
}

float
CClock::GetDeltaTick()
{
  return (m_fDeltaTime / 1000.0f);
}
```

1.  将以下代码添加到`Game.h`中：

```cpp
#pragma once

#if !defined(__GAME_H__)
#define __GAME_H__

// Library Includes
#include <windows.h>

// Local Includes
#include "clock.h"

// Types

// Constants

// Prototypes
class CBackBuffer;

class CGame
{
  // Member Functions
public:
  ~CGame();

  bool Initialise(HINSTANCE _hInstance, HWND _hWnd, int _iWidth, int _iHeight);

  void Draw();
  void Process(float _fDeltaTick);

  void ExecuteOneFrame();

  CBackBuffer* GetBackBuffer();
  HINSTANCE GetAppInstance();
  HWND GetWindow();

  // Singleton Methods
  static CGame& GetInstance();
  static void DestroyInstance();

protected:

private:
  CGame();
  CGame(const CGame& _kr);
  CGame& operator= (const CGame& _kr);

  // Member Variables
public:

protected:
  CClock* m_pClock;

  CBackBuffer* m_pBackBuffer;

  //Application data
  HINSTANCE m_hApplicationInstance;
  HWND m_hMainWindow;

  // Singleton Instance
  static CGame* s_pGame;

private:

};

#endif    // __GAME_H__
```

1.  将以下代码添加到`Game.cpp`中：

```cpp
// Library Includes

// Local Includes
#include "Clock.h"
#include "BackBuffer.h"
#include "Utilities.h"

// This Include
#include "Game.h"

// Static Variables
CGame* CGame::s_pGame = 0;

// Static Function Prototypes

// Implementation

CGame::CGame()
: m_pClock(0)
, m_hApplicationInstance(0)
, m_hMainWindow(0)
, m_pBackBuffer(0)
{

}

CGame::~CGame()
{
  delete m_pBackBuffer;
  m_pBackBuffer = 0;

  delete m_pClock;
  m_pClock = 0;
}

bool
CGame::Initialise(HINSTANCE _hInstance, HWND _hWnd, int _iWidth, int _iHeight)
{
  m_hApplicationInstance = _hInstance;
  m_hMainWindow = _hWnd;

  m_pClock = new CClock();
  VALIDATE(m_pClock->Initialise());
  m_pClock->Process();

  m_pBackBuffer = new CBackBuffer();
  VALIDATE(m_pBackBuffer->Initialise(_hWnd, _iWidth, _iHeight));

  ShowCursor(false);

  return (true);
}

void
CGame::Draw()
{
  m_pBackBuffer->Clear();

  // Do all the game's drawing here...

  m_pBackBuffer->Present();
}

void
CGame::Process(float _fDeltaTick)
{
  // Process all the game's logic here.
}

void
CGame::ExecuteOneFrame()
{
  float fDT = m_pClock->GetDeltaTick();

  Process(fDT);
  Draw();

  m_pClock->Process();

  Sleep(1);
}

CGame&
CGame::GetInstance()
{
  if (s_pGame == 0)
  {
    s_pGame = new CGame();
  }

  return (*s_pGame);
}

void
CGame::DestroyInstance()
{
  delete s_pGame;
  s_pGame = 0;
}

CBackBuffer*
CGame::GetBackBuffer()
{
  return (m_pBackBuffer);
}

HINSTANCE
CGame::GetAppInstance()
{
  return (m_hApplicationInstance);
}

HWND
CGame::GetWindow()
{
  return (m_hMainWindow);
}
```

1.  将以下代码添加到`sprite.h`中：

```cpp
#pragma once

#if !defined(__SPRITE_H__)
#define __SPRITE_H__

// Library Includes
#include "windows.h"

// Local Includes

// Types

// Constants

// Prototypes
class CSprite
{
  // Member Functions
public:
  CSprite();
  ~CSprite();

  bool Initialise(int _iResourceID, int _iMaskResourceID);

  void Draw();
  void Process(float _fDeltaTick);

  int GetWidth() const;
  int GetHeight() const;

  int GetX() const;
  int GetY() const;
  void SetX(int _i);
  void SetY(int _i);

  void TranslateRelative(int _iX, int _iY);
  void TranslateAbsolute(int _iX, int _iY);

protected:

private:
  CSprite(const CSprite& _kr);
  CSprite& operator= (const CSprite& _kr);

  // Member Variables
public:

protected:
  //Center handle
  int m_iX;
  int m_iY;

  HBITMAP m_hSprite;
  HBITMAP m_hMask;

  BITMAP m_bitmapSprite;
  BITMAP m_bitmapMask;

  static HDC s_hSharedSpriteDC;
  static int s_iRefCount;

private:

};

#endif    // __SPRITE_H__
```

1.  将以下代码添加到`sprite.cpp`中：

```cpp
// Library Includes

// Local Includes
#include "resource.h"
#include "Game.h"
#include "BackBuffer.h"
#include "Utilities.h"

// This include
#include "Sprite.h"

// Static Variables
HDC CSprite::s_hSharedSpriteDC = 0;
int CSprite::s_iRefCount = 0;

// Static Function Prototypes

// Implementation

CSprite::CSprite()
: m_iX(0)
, m_iY(0)
{
  ++s_iRefCount;
}

CSprite::~CSprite()
{
  DeleteObject(m_hSprite);
  DeleteObject(m_hMask);

  --s_iRefCount;

  if (s_iRefCount == 0)
  {
    DeleteDC(s_hSharedSpriteDC);
    s_hSharedSpriteDC = 0;
  }
}

bool
CSprite::Initialise(int _iSpriteResourceID, int _iMaskResourceID)
{
  HINSTANCE hInstance = CGame::GetInstance().GetAppInstance();

  if (!s_hSharedSpriteDC)
  {
    s_hSharedSpriteDC = CreateCompatibleDC(NULL);
  }

  m_hSprite = LoadBitmap(hInstance, MAKEINTRESOURCE(_iSpriteResourceID));
  VALIDATE(m_hSprite);
  m_hMask = LoadBitmap(hInstance, MAKEINTRESOURCE(_iMaskResourceID));
  VALIDATE(m_hMask);

  GetObject(m_hSprite, sizeof(BITMAP), &m_bitmapSprite);
  GetObject(m_hMask, sizeof(BITMAP), &m_bitmapMask);

  return (true);
}

void
CSprite::Draw()
{
  int iW = GetWidth();
  int iH = GetHeight();

  int iX = m_iX - (iW / 2);
  int iY = m_iY - (iH / 2);

  CBackBuffer* pBackBuffer = CGame::GetInstance().GetBackBuffer();

  HGDIOBJ hOldObj = SelectObject(s_hSharedSpriteDC, m_hMask);

  BitBlt(pBackBuffer->GetBFDC(), iX, iY, iW, iH, s_hSharedSpriteDC, 0, 0, SRCAND);

  SelectObject(s_hSharedSpriteDC, m_hSprite);

  BitBlt(pBackBuffer->GetBFDC(), iX, iY, iW, iH, s_hSharedSpriteDC, 0, 0, SRCPAINT);

  SelectObject(s_hSharedSpriteDC, hOldObj);
}

void
CSprite::Process(float _fDeltaTick)
{

}

int
CSprite::GetWidth() const
{
  return (m_bitmapSprite.bmWidth);
}

int
CSprite::GetHeight() const
{
  return (m_bitmapSprite.bmHeight);
}

int
CSprite::GetX() const
{
  return (m_iX);
}

int
CSprite::GetY() const
{
  return (m_iY);
}

void
CSprite::SetX(int _i)
{
  m_iX = _i;
}

void
CSprite::SetY(int _i)
{
  m_iY = _i;
}

void
CSprite::TranslateRelative(int _iX, int _iY)
{
  m_iX += _iX;
  m_iY += _iY;
}

void
CSprite::TranslateAbsolute(int _iX, int _iY)
{
  m_iX = _iX;
  m_iY = _iY;
}
```

1.  将以下代码添加到`Utilities.h`中：

```cpp
// Library Includes

// Local Includes
#include "resource.h"
#include "Game.h"
#include "BackBuffer.h"
#include "Utilities.h"

// This include
#include "Sprite.h"

// Static Variables
HDC CSprite::s_hSharedSpriteDC = 0;
int CSprite::s_iRefCount = 0;

// Static Function Prototypes

// Implementation

CSprite::CSprite()
: m_iX(0)
, m_iY(0)
{
  ++s_iRefCount;
}

CSprite::~CSprite()
{
  DeleteObject(m_hSprite);
  DeleteObject(m_hMask);

  --s_iRefCount;

  if (s_iRefCount == 0)
  {
    DeleteDC(s_hSharedSpriteDC);
    s_hSharedSpriteDC = 0;
  }
}

bool
CSprite::Initialise(int _iSpriteResourceID, int _iMaskResourceID)
{
  HINSTANCE hInstance = CGame::GetInstance().GetAppInstance();

  if (!s_hSharedSpriteDC)
  {
    s_hSharedSpriteDC = CreateCompatibleDC(NULL);
  }

  m_hSprite = LoadBitmap(hInstance, MAKEINTRESOURCE(_iSpriteResourceID));
  VALIDATE(m_hSprite);
  m_hMask = LoadBitmap(hInstance, MAKEINTRESOURCE(_iMaskResourceID));
  VALIDATE(m_hMask);

  GetObject(m_hSprite, sizeof(BITMAP), &m_bitmapSprite);
  GetObject(m_hMask, sizeof(BITMAP), &m_bitmapMask);

  return (true);
}

void
CSprite::Draw()
{
  int iW = GetWidth();
  int iH = GetHeight();

  int iX = m_iX - (iW / 2);
  int iY = m_iY - (iH / 2);

  CBackBuffer* pBackBuffer = CGame::GetInstance().GetBackBuffer();

  HGDIOBJ hOldObj = SelectObject(s_hSharedSpriteDC, m_hMask);

  BitBlt(pBackBuffer->GetBFDC(), iX, iY, iW, iH, s_hSharedSpriteDC, 0, 0, SRCAND);

  SelectObject(s_hSharedSpriteDC, m_hSprite);

  BitBlt(pBackBuffer->GetBFDC(), iX, iY, iW, iH, s_hSharedSpriteDC, 0, 0, SRCPAINT);

  SelectObject(s_hSharedSpriteDC, hOldObj);
}

void
CSprite::Process(float _fDeltaTick)
{

}

int
CSprite::GetWidth() const
{
  return (m_bitmapSprite.bmWidth);
}

int
CSprite::GetHeight() const
{
  return (m_bitmapSprite.bmHeight);
}

int
CSprite::GetX() const
{
  return (m_iX);
}

int
CSprite::GetY() const
{
  return (m_iY);
}

void
CSprite::SetX(int _i)
{
  m_iX = _i;
}

void
CSprite::SetY(int _i)
{
  m_iY = _i;
}

void
CSprite::TranslateRelative(int _iX, int _iY)
{
  m_iX += _iX;
  m_iY += _iY;
}

void
CSprite::TranslateAbsolute(int _iX, int _iY)
{
  m_iX = _iX;
  m_iY = _iY;
}
```

## 它是如何工作的...

正如我们所知，后备缓冲用于首先绘制图像，然后我们交换缓冲区以将其呈现到屏幕上。这个过程也被称为*呈现*。我们创建了一个通用的`backbuffer`类，它帮助我们交换缓冲区。`sprite`类用于加载精灵并将它们推送到后备缓冲区，然后可以对它们进行处理并最终绘制到屏幕上。精灵类还提供了一些基本的实用函数，帮助我们获取精灵的宽度和高度。大多数函数只是在 Windows 自己的 API 函数和回调的顶部包装。我们还创建了一个`clock`类，它帮助我们跟踪时间，因为每个时间点都应该实现为时间增量的函数。如果我们不这样做，那么游戏可能会根据执行它的机器而出现波动的行为。`game`类用于将所有内容放在一起。它有一个`backbuffer`的实例，这是一个单例类，处理窗口和其他资源的上下文。

# 使用动画精灵

使用动画精灵是游戏编程的重要部分。除非对精灵应用某种形式的动画，否则它看起来不够真实，整个游戏沉浸感将会丧失。虽然动画可以通过多种方式实现，但我们只会看到精灵带动画的条带动画，因为这是 2D 游戏中最常用的动画形式。

## 准备就绪

要完成这个示例，您需要一台运行 Windows 的机器。您还需要在 Windows 机器上安装 Visual Studio 的工作副本。不需要其他先决条件。

## 如何做...

在这个示例中，我们将发现创建对话框有多么容易。

1.  打开 Visual Studio。

1.  创建一个新的 C++项目。

1.  选择 Win32 Windows 应用程序。

1.  添加一个`AnimatedSprite.cpp`文件。

1.  将以下代码添加到`Source.cpp`中：

```cpp
// This include
#include "AnimatedSprite.h"

// Static Variables

// Static Function Prototypes

// Implementation

CAnimatedSprite::CAnimatedSprite()
: m_fFrameSpeed(0.0f)
, m_fTimeElapsed(0.0f)
, m_iCurrentSprite(0)
{

}

CAnimatedSprite::~CAnimatedSprite()
{
  Deinitialise();
}

bool
CAnimatedSprite::Deinitialise()
{
  return (CSprite::Deinitialise());
}

bool
CAnimatedSprite::Initialise(int _iSpriteResourceID, int _iMaskResourceID)
{
  return (CSprite::Initialise(_iSpriteResourceID, _iMaskResourceID));
}

void
CAnimatedSprite::Draw()
{
  int iTopLeftX = m_vectorFrames[m_iCurrentSprite];
  int iTopLeftY = 0;

  int iW = GetFrameWidth();
  int iH = GetHeight();

  int iX = m_iX - (iW / 2);
  int iY = m_iY - (iH / 2);

  HDC hSpriteDC = hSharedSpriteDC;

  HGDIOBJ hOldObj = SelectObject(hSpriteDC, m_hMask);

  BitBlt(CGame::GetInstance().GetBackBuffer()->GetBFDC(), iX, iY, iW, iH, hSpriteDC, iTopLeftX, iTopLeftY, SRCAND);

  SelectObject(hSpriteDC, m_hSprite);

  BitBlt(CGame::GetInstance().GetBackBuffer()->GetBFDC(), iX, iY, iW, iH, hSpriteDC, iTopLeftX, iTopLeftY, SRCPAINT);

  SelectObject(hSpriteDC, hOldObj);
}

void
CAnimatedSprite::Process(float _fDeltaTick)
{
  m_fTimeElapsed += _fDeltaTick;

  if (m_fTimeElapsed >= m_fFrameSpeed &&
    m_fFrameSpeed != 0.0f)
  {
    m_fTimeElapsed = 0.0f;
    ++m_iCurrentSprite;

    if (m_iCurrentSprite >= m_vectorFrames.size())
    {
      m_iCurrentSprite = 0;
    }
  }

  CSprite::Process(_fDeltaTick);
}

void
CAnimatedSprite::AddFrame(int _iX)
{
  m_vectorFrames.push_back(_iX);
}

void
CAnimatedSprite::SetSpeed(float _fSpeed)
{
  m_fFrameSpeed = _fSpeed;
}

void
CAnimatedSprite::SetWidth(int _iW)
{
  m_iFrameWidth = _iW;
}

int
CAnimatedSprite::GetFrameWidth()
{
  return (m_iFrameWidth);
}
```

## 它是如何工作的...

为了使动画正常工作，我们需要加载一系列图像作为精灵条。图像数量越多，动画就会更流畅。对于相应数量的精灵，我们还需要加载它们的蒙版，以便它们可以一起贴图。我们需要将所有图像存储在一个向量列表中。为了使动画正常工作，所有图像必须等间距分布。在正确存储它们之后，我们可以通过控制在一定时间内要绘制多少帧/精灵来以我们想要的速度快速或缓慢地运行动画。在屏幕上绘制精灵的剩余过程保持不变。
