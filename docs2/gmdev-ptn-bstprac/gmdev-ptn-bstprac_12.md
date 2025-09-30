# 最佳实践

学习编程由于多种原因而困难，但学习游戏编程则更加困难，特别是由于存在许多不同的系统和对象类型需要相互交互。在这本书中，我们已经介绍了一些最重要的设计模式，以使这些交互尽可能简单。每一章都明确地关注一个设计模式，以帮助简化编码。然而，在每个段落和代码示例中，都隐藏着核心思想和技巧，有助于使我们的设计更容易阅读和维护。

这些*最佳实践*有时可以在其他书中找到；然而，编程书籍往往努力教你一门语言的语法，而不是风格、设计和组织。即使是关于设计模式的书籍也可能忽略这些基本技巧。由于它们非常基础，很容易忘记它们并不一定在所有地方都明确讨论。这让你，作为读者，不得不阅读数十本书，并在互联网上搜寻讨论这些基础知识的博客文章。更糟糕的是，你需要花费数小时甚至数十小时编写代码，感觉这些代码可以更好，但你就是不明白为什么它不好。

当然，所有这些事情都会发生。作为程序员的一部分是不断阅读这样的书籍。你应该通过阅读博客来寻找改进的方法，你会在六个月后认为你写的代码是垃圾。我们写这本书的愿望是希望你能尽早而不是更晚地理解和将这些基础知识融入你的程序中。

# 章节概述

在本章中，我们将重点关注那些能够提升你的代码质量和游戏水平的根本思想和技巧。这些思想来源于多年的编程经验以及多年的教学经验。如果这些看起来简单明了，那真是太好了。然而，我们选择这些主题是因为它们是我们，作为作者，早期遇到的难题，或者是我们学生遇到的难题。

# 你的目标

在本章中，我们将讨论多个主题：

+   学习基本的代码质量技巧

+   学习和理解 const 关键字的使用

+   学习迭代如何改进你的游戏和代码设计

+   学习在游戏中何时使用脚本

# 学习基本的代码质量技巧

从初学者成长为专家程序员的进程可能具有挑战性。一开始，你必须学习不仅语言的规则，还要学习如何使用编译器和理解错误信息。此外，你试图解决越来越困难的编程问题，同时遵循可能看似任意的编写*良好*代码的规则。大多数新手程序员专注于解决给定的问题，而不是使代码看起来很漂亮。对许多人来说，花时间使代码看起来整洁似乎毫无价值，因为编写后它几乎肯定会删除。即使是经验丰富的程序员，在匆忙完成作业或项目时也可能忽略代码风格。

这有几个原因不好。首先，写得好的代码更容易阅读和理解。它几乎肯定有更少的错误，并且比随意混合在一起且从未打磨过的代码更有效率。正如我们在前面的章节中讨论的那样，你前期花在确保代码无错误上的时间，是以后你不需要用来调试的时间。你花在确保代码可读和易于维护上的时间，是以后你不需要用来修改或解读旧代码的时间。

其次，良好的编程风格是一种习惯。花时间阅读和调试你的代码一开始会很慢。然而，随着你不断提高代码质量，它变得越来越容易和快速。最终，你会养成习惯，编写高质量的代码将变得自然而然。没有这种习惯，很容易将风格放在一边，以后再担心。然而，正在编写的代码几乎总是草率的，以后很难找到时间回去改进它，因为总是有另一个截止日期在逼近。有了良好的习惯，你甚至在最紧张的时间限制情况下，如面试或即将到来的截止日期，也能写出干净、可读的代码。

最后，在未来的某个时刻，你几乎肯定会与其他程序员一起工作。这可能是一个由两三个程序员组成的小团队，或者可能是在一个拥有遍布全球数十个程序员的跨国公司中。即使你理解你的代码在做什么，也不能保证你的队友会理解。编写难以理解的代码会导致人们错误地使用你的代码。相反，努力使你的代码易于使用且难以破坏。以其他人对你的代码的喜爱为荣，你的队友和上司会感谢你。如果你的队友也这样做，你会感到很感激。在某个时刻，你将需要维护其他程序员离开工作后的代码。如果你离开后他们写的代码质量高，你会发现这要容易得多，所以写上你离开后也容易工作的代码。

在接下来的几页中，我们将介绍一些非常基础但极其重要的代码质量提示。正如我们所说，这些来自多年的阅读编程经验，以及教学。将使用这些技术为每行代码。思考这些技术为每段代码。这样做将帮助你形成良好的习惯。

# 避免使用魔法数字

将数字字面量硬编码到代码中通常被认为是一个坏主意。使用数字字面量而不是命名常量的问题是，读者不知道那个数字的目的。数字在代码中似乎凭空出现。考虑以下代码：

```cpp
M5Object* pUfo = M5ObjectManager::CreateObject(AT_Ufo); 
pUfo->pos.x    = M5Random::GetFloat(-100, 100); 
pUfo->pos.y    = M5Random::GetFloat(-60, 60); 

```

很难知道为什么选择了这四个数字。也很难知道如果修改了这些值，程序将如何改变。如果使用命名常量或变量，这样的代码将更容易阅读和维护：

```cpp
M5Object* pUfo = M5ObjectManager::CreateObject(AT_Ufo); 
pUfo->pos.x    = M5Random::GetFloat(minWorldX, maxWorldX); 
pUfo->pos.y    = M5Random::GetFloat(minWorldY, MaxWorldY); 

```

更改后，更容易理解新的不明飞行物（UFO）的位置是在世界内随机放置的。我们可以理解，如果我们更改这些值，UFO 可能的起始位置将是世界之外，或者被限制在围绕世界中心的更紧密的矩形内。

除了难以阅读和理解之外，使用魔法数字会使代码难以维护和更新。假设我们有一个大小为 256 的数组。每个需要操作数组的循环都必须硬编码值 256。如果数组的大小需要增大或减小，我们就需要更改所有 256 的出现。我们无法简单地进行“查找和替换”，因为 256 在代码中可能用于完全不同的原因。相反，我们必须查看数字的所有出现，并确保我们正确地更改了代码。如果我们错过任何一个，我们可能会创建一个错误。例如，如果我们将数组的大小更改为更小的值，例如 128。任何仍然将数组视为大小为 256 的循环都会导致未定义的行为：

```cpp
int buffer[256]; 

//Some function to give start values 
InitializeBuffer(buffer, 256);  

for(int i = 0; i < 256; ++i) 
std::cout << i " " << std::endl; 

```

如前所述，最好使用命名常量而不是魔法数字。常量更易于阅读和更改，因为它只需要在一个地方更改。它也较少引起错误，因为我们只更改与数组相关的值。我们不会意外地更改不应该更改的值或错过应该更改的值：

```cpp
const int BUFFER_SIZE = 256;  
int buffer[BUFFER_SIZE]; 

//Some function to give start values 
InitializeBuffer(buffer, BUFFER_SIZE);  

for(int i = 0; i < BUFFER_SIZE; ++i) 
std::cout << i " " << std::endl; 

```

我们不想使用魔法数字的另一个重要原因是它们缺乏灵活性。在这本书中，我们试图强调从文件中读取数据的优点。显然，如果你硬编码一个值，它就不能从文件中读取。在先前的例子中，如果`BUFFER_SIZE`需要更改，代码需要重新编译。然而，如果缓冲区的大小在运行时从文件中读取，代码只需要编译一次，程序将适用于所有大小的缓冲区：

```cpp
int bufferSize = GetSizeFromFile(fileName);  

//we can Dynamically allocate our buffer 
int* buffer = new int[bufferSize]; 

//Some function to give start values 
InitializeBuffer(buffer, bufferSize);  

for(int i = 0; i < bufferSize; ++i) 
std::cout << i " " << std::endl; 

delete [] buffer;//We must remember to deallocate 

```

在前面的例子中，我们必须记住释放缓冲区。记住，这很可能不是通常的情况，因为对于数组，我们总是可以使用 STL 向量。更一般的情况是我们从文件中读取整数或浮点数。这些可以用于从屏幕分辨率到玩家速度，甚至到生成敌人的时间间隔等任何东西。

与所有规则一样，有一些例外或特殊情况，可能允许硬编码数字。数字`0`和`1`通常被认为是可接受的。这些可能用作整数或浮点数的初始化值，或者只是数组的起始索引。

你的目标是使你的代码尽可能易于阅读和灵活，因此命名常量几乎总是比硬编码的数字更好。尽你所能确保你的代码可以被他人理解。如果你的变量名为`ZERO`或`TWO`，你的代码并不一定更易于阅读，所以你应该使用你的最佳判断，并在你认为含义不明确时，也许可以询问另一位程序员。

# 空白

当思考高质量代码时，空白往往被忽视。也许这是因为空白不是你编写的代码，而是你代码之间的空白空间。然而，如果你没有正确使用空白，你的代码将难以阅读。当我们提到空白时，我们指的是程序内部的空格、制表符、换行符和空白行。你如何使用这些元素可以决定代码是易于阅读和维护，还是让你做噩梦。以下是一段对空白考虑很少的代码：

```cpp
RECT rect={0}; 
int xStart= 0,yStart = 0; 
rect.right=s_width;rect.bottom=s_height; 
s_isFullScreen = fullScreen; 
if (fullScreen) {DEVMODE settings;  
settings.dmSize = sizeof(settings); 
EnumDisplaySettings(0, ENUM_CURRENT_SETTINGS, &settings); 
settings.dmPelsWidth=(DWORD)s_width;  
settings.dmPelsHeight = (DWORD)s_height; 
settings.dmFields = DM_BITSPERPEL|DM_PELSWIDTH|DM_PELSHEIGHT; 
s_style = FULLSCREEN_STYLE; 
if (ChangeDisplaySettings(&settings 
,CDS_FULLSCREEN) !=DISP_CHANGE_SUCCESSFUL) { 
s_isFullScreen = false;s_style = WINDOWED_STYLE; 
ChangeDisplaySettings(0, 0);M5Debug::MessagePopup( 
"FullScreen is not supported. " 
"You are being switched to Windowed Mode"); }  
} 
 else {ChangeDisplaySettings(0, 0); s_style = WINDOWED_STYLE;} 

```

上述代码对编译器来说是完全可接受的。然而，对于人类来说，上面的代码难以阅读，因为没有行间距、缩进和不一致。当然，这是一个极端的例子，但我们在多年的教学过程中，也看到了对样式和格式考虑很少的代码示例。当代码看起来像上面的例子时，注释和标识符名称的质量并不重要，因为整个块难以阅读。将上述代码与以下版本进行比较，该版本试图使代码对人类可读：

```cpp
/*Set window rect size and start position*/ 
RECT rect   = { 0 }; 
rect.right  = s_width; 
rect.bottom = s_height; 
int xStart  = 0; 
int yStart  = 0; 

/*save input parameter to static var*/ 
s_isFullScreen = fullScreen; 

/*Check if we are going into full screen or not*/ 
if (fullScreen) 
{ 
  /*Get the current display settings*/ 
  DEVMODE settings; 
  settings.dmSize = sizeof(settings); 
  EnumDisplaySettings(0, ENUM_CURRENT_SETTINGS, &settings); 

  /*Change the resolution to the resolution of my window*/ 
  settings.dmPelsWidth  = static_cast<DWORD>(s_width); 
  settings.dmPelsHeight = static_cast<DWORD>(s_height); 
  settings.dmFields     = DM_BITSPERPEL | DM_PELSWIDTH  |  
                          DM_PELSHEIGHT; 

  /*Make sure my window style is full screen*/ 
  s_style = FULLSCREEN_STYLE; 

  /*If we can't change, switch back to desktop mode*/ 
  if ( ChangeDisplaySettings(&settings, CDS_FULLSCREEN) !=  
                                  DISP_CHANGE_SUCCESSFUL ) 
  { 
    s_isFullScreen = false; 
    s_style        = WINDOWED_STYLE; 
    ChangeDisplaySettings(0, 0); 
    M5Debug::MessagePopup("FullScreen is not supported. " 
    "You are being switched to Windowed Mode"); 
  } 
} 
else /*If we are already fullscreen, switch to desktop*/ 
{ 
  /*Make sure I am in windows style*/ 
  s_style = WINDOWED_STYLE; 
  ChangeDisplaySettings(0, 0); 
} 

```

虽然前面的例子绝对不是高质量代码的完美示例，但它确实比第一个例子更易于阅读和维护。当涉及到现实世界的程序和程序员时，没有什么是完美的。每个程序员都有自己的风格，这实际上意味着每个程序员都认为自己的风格是最容易阅读的。然而，随着你阅读更多的代码，你会注意到可读代码有一些共同元素。现在让我们看看这些元素中的一些。

# 缩进

块语句，如循环和条件语句，应该将子语句缩进。这很容易向读者展示程序意图。缩进的空格数量不如缩进本身重要。大多数程序员认为 2 到 4 个空格对于可读性是足够的。最重要的是要保持缩进的一致性。同样，起始大括号的位置并不重要（尽管你可以在网上找到一些有趣的论点），但重要的是要始终将其放置在同一位置：

```cpp
//This shows the purpose of the statement 
if (s_isFullScreen) 
{ 
  s_style = FULLSCREEN_STYLE; 
  SetFullScreen(true); 
} 

//So does this 
if (s_isFullScreen) { 
  s_style = FULLSCREEN_STYLE; 
  SetFullScreen(true); 
} 

//This does not shows the intent of the statement 
if (s_isFullScreen) 
{ 
s_style = FULLSCREEN_STYLE; 
SetFullScreen(true); 
} 

```

需要记住的是，在 C++中，缩进对编译器没有意义。如果没有大括号，循环和条件语句将只执行一个语句。因此，一些程序员无论需要多少个子语句，都会始终使用大括号：

```cpp
/*Single statement in the loop*/ 
while (i++ < 10) 
  printf("The value of i is %d\n", i); 

/*After adding another statement*/ 
while (i++ < 10) 
  printf("The value of i is %d\n", i); 
  printf("i squared is %d\n", i*i); 

```

前面的例子具有误导性，因为只有第一个语句将作为循环的一部分。在编写循环或条件语句时，忘记在添加第二个语句后添加大括号是一种常见的错误。因此，一些程序员即使在单语句循环和条件语句中也会使用大括号。这种想法是代码更易于阅读和维护，因此更不容易出错。

# 空行和空格

正如我们之前所说的，你如何使用空白空间将决定你的代码的可读性。使用缩进来表示代码块是显示程序逻辑结构的一种方法。另一种展示这种结构的好方法是使用空行。就像一篇好的写作被分成段落一样，好的代码也应该以某种逻辑分组的形式被分隔。将逻辑上相关的语句组合在一起。在组之间放置空行以提高可读性：

```cpp
//Save position and scale to variables for readability. 
const float HALF = .5f; 
M5Vec2 halfScale = m_pObj->scale * HALF; 
M5Vec2 pos = m_pObj->pos; 
//Get world extents 
M5Vec2 botLeft; 
M5Vec2 topRight; 
M5Gfx::GetWorldBotLeft(botLeft); 
M5Gfx::GetWorldTopRight(topRight); 
//If object is outside of world, mark as dead 
if (pos.x - halfScale.x > topRight.x || pos.x + 
    halfScale.x < botLeft.x || pos.y - halfScale.y  
    > topRight.y || pos.y + halfScale.y < botLeft.y) 
{ 
  m_pObj->isDead = true; 
} 

```

前面的代码没有空行，所以代码看起来是连续的。看它很难理解代码在做什么，因为你的大脑试图一次性理解所有内容。尽管有注释，但它们并没有真正帮助，因为它们与代码的其他部分混合在一起。if 语句也难以阅读，因为条件是通过它们在行上的适应而不是逻辑对齐来分隔的。在下面的代码中，我们添加了一些空行来分隔语句的逻辑分组：

```cpp
//Save position and scale to variables for readability. 
const float HALF = .5f; 
M5Vec2 halfScale = m_pObj->scale * HALF; 
M5Vec2 pos       = m_pObj->pos; 

//Get world extents 
M5Vec2 botLeft; 
M5Vec2 topRight; 
M5Gfx::GetWorldBotLeft(botLeft); 
M5Gfx::GetWorldTopRight(topRight); 

//If object is outside of world, mark as dead 
if ( pos.x - halfScale.x > topRight.x ||  
     pos.x + halfScale.x < botLeft.x  || 
     pos.y - halfScale.y > topRight.y ||  
     pos.y + halfScale.y < botLeft.y  ) 
{ 
  m_pObj->isDead = true; 
} 

```

通过使用行断开来将相关的语句组合在一起，代码被分隔成易于理解的块。这些块帮助读者理解哪些语句在逻辑上应该放在一起。此外，每个块开头的注释更加突出，并用英语准确说明了块中将要发生的事情。

复杂的条件语句应该根据条件进行分隔和对齐，以便更容易理解。在前面给出的代码中，四个条件都是按照相同的方式进行对齐的。这为读者提供了关于条件将如何执行的解释。使用括号并结合代码对齐进一步增加了可读性：

```cpp
//If object is outside of world, mark as dead 
if (( (pos.x - halfScale.x) > topRight.x ) ||  
    ( (pos.x + halfScale.x) < botLeft.x  ) || 
    ( (pos.y - halfScale.y) > topRight.y ) ||  
    ( (pos.y + halfScale.y) < botLeft.y  )) 
{ 
  m_pObj->isDead = true; 
} 

```

使用括号不仅仅在条件语句中有帮助。所有复杂的表达式都应该用括号括起来。当然，每个人对复杂的定义都不同，所以一个好的通用规则是 `*`、`/` 和 `%` 的执行顺序在 `+` 和 `-` 之前；其他所有情况都使用括号。这不仅会让读者更清晰，还能确保代码的执行方式与你预期的一致。即使你理解了所有 C++ 的优先级和结合性规则，你的队友可能不一定理解。括号并不需要任何成本，但可以提高可读性，所以请尽可能多地使用它们来展示代码的意图。

# 注释和自文档化的代码

注释和文档似乎比它们应有的争议性更大。一方面，许多人认为注释和文档是浪费时间。编写文档实际上会从编写代码中夺走时间，阅读注释则会从阅读代码中夺走时间。此外，有些人认为注释根本不起作用，因为它们可能会过时，而且不能解释源代码中已经存在的内容。注释最糟糕的情况是它们完全错误。在这种情况下，没有注释的代码可能反而更好。

然而，没有什么比调试没有添加注释的代码更令人沮丧的了。即使是你自己几个月前编写的代码，调试起来也可能很困难。最终，编写和更新注释所花费的时间是你和你的队友不需要花费在解读代码上的时间。

虽然注释的使用可能存在争议，但编写干净、高质量的代码对每个人都很重要。正如我们之前已经看到的，合理使用空白空间可以提高可读性。然而，仅仅空白空间本身并不能使代码可读。我们真正希望我们的代码是自文档化的。以下是一个例子，尽管它有适当的空白空间，但仍然难以阅读：

```cpp
void DoStuff(bool x[], int y) 
{   
  for(int i = 0; i < y; ++i) 
    x[i] = true; 

  x[0] = x[1] = false; 

  int b = static_cast<int>(std::sqrt(y)); 

  for(int a = 2; a <= b; ++a) 
  { 
    if(x[a] == false) 
      continue; 

    for(int c = a * 2; c < y; c += a) 
      x[c] = false; 
  } 
} 

```

你能看出这个算法在做什么吗？除非你恰好已经知道这个算法，否则你很可能不会理解函数的意图。注释在这里会有帮助，但更大的问题是标识符的低质量。好的变量名可以提供关于它们将用于什么的线索。想法是，有了好的变量名，你应该能够在不需要注释的情况下理解代码。这就是你使代码自文档化的方式：

```cpp
void CalculateSievePrimes(bool primes[], int arraySize) 
{     
  for(int i = 0; i < arraySize; ++i) 
    primes[i] = true; 

  primes[0] = primes[1] = false; 

  int upperBound = static_cast<int>(std::sqrt(arraySize)); 

  for(int candidate = 2; candidate <= upperBound; ++candidate) 
  { 
    if(primes[candidate] == false) 
      continue; 

    int multiple = candidate * 2; 
    for(; multiple < arraySize; multiple += candidate) 
      primes[multiple] = false; 
  } 
} 

```

即使你不理解前一个代码示例中的每一行，你至少可以使用函数名作为指南。名称`CalculateSievePrimes`是关于函数正在做什么的一个重要线索。从那里，你应该能够拼凑出每一行正在做什么。名称如 candidate、`arraySize`和 multiple 比`a`、`b`和`c`更有意义。自文档代码的最好部分是它永远不会出错，也永远不会过时。当然，代码仍然可能包含错误。只是它不能与文档不同步，因为代码本身就是文档。

如我们之前所说，你可以做一些事情来尝试使代码具有自文档特性。好的变量名是一个开始。变量名应该解释变量的确切目的，并且它们应该只用于那个目的。对于布尔变量，给出一个使真值含义显而易见的名称。例如，`isActive`比仅仅`active`或`activeFlag`要好得多，因为这样的名称给出了关于该变量真值含义的提示。

经常会有一些命名约定来区分类型、局部变量、常量和静态或全局变量。其中一些命名约定，例如使用全部大写字母来表示`const`变量，是非常常见的，并且被大多数程序员所使用。其他命名约定，例如所有静态变量名以`s_`开头，或者在指针名前添加`p`，则不太常见。无论你认为这些风格是否丑陋，都要明白它们的存在是为了帮助提高可读性，并使错误代码看起来更明显。编译器已经可以捕获这些命名约定旨在解决的问题中的一些，但鉴于它们仍然有助于提高可读性，因此值得考虑。

当给方法和函数命名时，也适用类似的规则。给出一个清晰的名称，说明函数的目的。确保函数或方法只有一个目的。通常，名称应该是一个动作。`CalculateSievePrimes`比`SeivePrimes`或仅仅是`Calculate`的名称更清晰。与布尔变量一样，返回布尔值的函数或方法通常带有提示性的名称。名称`IsEmpty`或`IsPowerOfTwo`比`Empty`或`PowerOfTwo`更清晰。

# 注释

如果代码是自文档的，那么我们为什么还需要添加注释呢？这确实是某些程序员的感受。当注释只是简单地重复代码，或者当注释过时且难以更新时，很容易理解他们为什么会这样想。然而，这与好的注释应有的作用正好相反。

好的注释应该解释代码无法表达的内容。例如，版权信息、作者和联系方式等，这些都是代码无法表示但可能对读者有用的信息。此外，好的注释不会简单地重复代码。下面的注释完全无用。它对代码没有任何帮助：

```cpp
//Assign START_VALUE to x 
int x = START_VALUE; 

```

相反，好的注释应该解释代码的意图和目的。即使你理解了一块代码应该做什么，你也不知道作者在编写它时在想什么。了解作者试图实现的目标可以在调试他人代码时节省你很多时间：

```cpp
/****************************************************************/ 
/*! 
Given an array of "arraySize" mark all indices that are prime as true. 

\param [out] primes  
The array to modify and Output. 

\param [in] arraySize  
  The number of elements in the array 

\return  
  None. Indices that are prime will be marked as true 

*/ 
/****************************************************************/ 
void CalculateSievePrimes(bool primes[], int arraySize) 
{  
  /*Ensure array is properly initialized */    
  for(int i = 0; i <size; ++i) 
    primes[i] = true; 

  /*Zero and One are never prime*/  
  primes[0] = primes[1] = false; 

/*Check values up to the square root of the max value*/ 
  int upperBound = static_cast<int>(std::sqrt(arraySize)); 

  /*Check each value, if valid, mark all multiples as false*/ 
  for(int candidate = 2; candidate <= upperBound; ++candidate) 
  { 
    if(primes[candidate] == false) 
      continue; 

    int multiple = candidate * 2; 
    for(; multiple < arraySize; multiple += candidate) 
      primes[multiple] = false; 
  } 
} 

```

上述注释解释了作者在编写代码时的想法。它们不仅仅是重复代码正在做什么。注意，一些注释解释了一行代码，而其他注释总结了整个代码块。关于代码中应该有多少注释并没有硬性规定。一个粗略的建议是，每个代码块都应该有一个注释来解释其目的，对于更复杂的行则可以有额外的注释。

类似于方法顶部的那段注释块最不可能被使用，但它们也能起到重要的作用。就像这本书的章节标题在查找特定内容时很有帮助一样，函数标题在扫描源代码文件查找特定函数时也能提供帮助。

函数标题非常有用，因为它们总结了关于函数的所有信息，而无需查看代码。任何人都可以轻松理解参数的目的、返回值，甚至可能抛出的任何异常。最好的部分是，通过使用像 Doxygen 这样的工具，可以将头文件块提取出来制作外部文档。

查看 Doxygen 工具和文档，请访问[`www.stack.nl/~dimitri/doxygen/`](http://www.stack.nl/~dimitri/doxygen/)。

当然，这些是最难编写和维护的。正是这样的注释块常常变得过时或完全错误。是否使用它们取决于你和你所在的团队。保持它们需要自律，但如果你在团队成员离开团队后处理他们的代码，它们可能就值得了。

# 学习和理解`const`关键字的使用

使用`const`是编程中似乎有些争议的另一个领域。一些程序员认为他们从未遇到过使用`const`就能解决问题的 bug。另一些人则认为，由于你不能保证`const`对象不会被修改，所以它完全无用。事实上，`const`对象是可以被修改的。`const`并非魔法。那么，`const`的正确性是否仍然是一个好东西呢？在我们深入探讨这个问题之前，让我们先看看`const`是什么。

当你创建一个`const`变量时，你必须对其进行初始化。所有`const`变量都会在编译时进行检查，以确保变量不会被赋予新的值。由于这发生在编译时，因此它不会对性能产生影响。以下是一些我们应该考虑的益处。首先，它提高了可读性。通过将变量标记为`const`，你是在告诉读者这个变量不应该改变。你正在分享你对变量的意图，并使你的代码具有自文档化的特性。`const`变量通常也使用全部大写字母命名，这进一步有助于提高可读性。其次，由于变量是在编译时进行检查的，因此用户无法意外地更改其值。如果有人试图修改变量，将会导致编译器错误。如果你预期值保持不变，这对你来说是个好消息。如果修改确实是一个意外，这对用户来说也是个好消息。

应该始终优先考虑编译器错误而不是运行时错误。任何时候我们都可以使用编译器来帮助我们找到问题。这就是为什么许多程序员选择将他们的编译器警告设置为最大，并将这些警告视为错误。花时间修复已知的编译器问题，你就不必花费时间去寻找它可能引起的运行时错误。

此外，应优先使用`const`变量而不是 C 风格的`#define`宏。宏是一个简单的工具。有时它们可能是完成工作的唯一工具，但对于简单的符号常量来说，它们是过度杀伤。宏进行盲目的*查找和替换*。符号常量在源代码中的任何位置都会被其值替换。虽然这些情况可能很少见，但它们也可能令人沮丧。由于值是在预处理阶段被替换的，所以在你试图解决问题时，源代码不会发生变化。

另一方面，`const`变量是语言的一部分。它们遵循所有正常的语言规则，包括类型和运算符。没有神秘的事情发生。它们只是不能重新分配的变量：

```cpp
int i1;            //No initialization, OK 
int i2    = 0;     //Initialization, OK 

const int ci1;     //ERROR: No initialization 
const int ci2 = 0; //Initialization, OK 

i1 = 10;           //Assignment, OK 
i2 += 2;           //Assignment, OK 

ci1 = 10;          //ERROR: Can't Assign 
ci2 += 2;          //ERROR: Can't Assign 

```

# `const`函数参数

将`const`变量作为符号常量创建可以使代码更易读，因为我们避免了使用魔法数字。然而，`const`的正确性不仅仅局限于创建符号常量。理解`const`与函数参数的关系同样重要。

理解这些不同函数签名之间的区别很重要：

```cpp
void Foo(int* a);      //Pass by pointer 
void Foo(int& a);      //Pass by reference 
void Foo(int a);       //Pass by value 
void Foo(const int a); //Pass by const value 
void Foo(const int* a);//Pass by pointer to const 
void Foo(const int& a);//Pass by reference to const 

```

C 和 C++的默认行为是按值传递。这意味着当你将变量传递给函数时，会创建一个副本。对函数参数所做的更改不会修改原始变量。函数作者有自由使用变量的方式，而原始变量所有者可以确信值将保持不变。

这意味着，从原始变量所有者的角度来看，这两个函数签名表现相同。实际上，在考虑函数重载时，编译器不会在这两个之间做出区分：

```cpp
void Foo(int a);       //Pass by value 
void Foo(const int a); //Pass by const value 

```

由于按值传递的变量在传递给函数时无法被修改，许多程序员不会将这些参数标记为`const`。尽管如此，将它们标记为`const`仍然是一个好主意，因为这向读者表明变量的值不应该被改变。然而，这种类型的参数标记为`const`的重要性较低，因为它不能被改变。

当你想将数组传递给函数时怎么办？记住，C 和 C++的一个小特点是数组有时和指针被类似对待。当你将数组传递给函数时，并不会创建数组的副本。相反，传递的是指向第一个元素的指针。这种默认行为的一个副作用是函数现在可以修改原始数据：

```cpp
//A Poorly named function that unexpectedly modifies data 
void PrintArray(int buffer[], int size) 
{ 
  for(int i = 0; i < size; ++i) 
  { 
    buffer[i] = 0; //Whoops!!! 
    std::cout << buffer[i] << " "; 
  } 
  std::cout << std::endl; 
} 
//Example of creating an array and passing it to the function 
int main(void) 
{ 
  const int SIZE  = 5; 
  int array[SIZE] = {1, 2, 3, 4, 5}; 

  PrintArray(array, SIZE); 
  return 0; 
} 

```

上述代码的输出如下：

```cpp
0 0 0 0 0  

```

正如你所见，没有任何东西阻止函数修改原始数据。函数中的`size`变量是主函数中`SIZE`的一个副本。然而，`buffer`变量是一个指向数组的指针。由于`PrintArray`函数很短，所以找到这个错误可能很容易，但在一个可能将指针传递给其他函数的较长的函数中，这个问题可能很难追踪。

如果用户想防止函数修改数据，他们可以将数组标记为 const。然而，他们将无法使用`PrintArray`函数，也无法修改数据：

```cpp
int main(void) 
{ 
  const int SIZE  = 5; 
  const int array[SIZE] = {1, 2, 3, 4, 5};//Marked as const 

  array[0] = 0;//ERROR: Can't modify a const array 

  PrintArray(array, SIZE);//Error: Function doesn't accept const 
return 0; 
} 

```

当然，有时函数的目的是修改数据。在这种情况下，用户必须接受如果他们想使用该函数的话。对于像`PrintArray`这样的名字，用户可能期望在函数调用后数据不会改变。数据修改是有意为之还是意外？用户无法得知。

由于问题出在函数名不清晰，所以修改的责任在于函数的作者。他们可以选择使名称更清晰，比如使用`ClearAndPrintArray`这样的名字，或者修复错误。当然，修复错误并不能防止类似的事情再次发生，也不能明确函数的意图。

一个更好的主意是作者将缓冲区标记为 const 参数。这将允许编译器捕捉到上述类似的事故，并且会向用户表明函数承诺不会修改数据：

```cpp
//Const prevents the function from modifying the data 
void PrintArray(const int buffer[], int size) 
{ 
for(int i = 0; i < size; ++i) 
{ 
  //buffer[i] = 0; //This would be a compiler error 
  std::cout << buffer[i] << " "; 
  } 
std::cout << std::endl; 
} 

int main(void) 
{ 
  const int SIZE  = 5; 
  int array[SIZE] = {1, 2, 3, 4, 5}; 

  array[0] = 0;//Modifying the array is fine 

  PrintArray(array, SIZE);//OK. Can accept non-const 
return 0; 
} 

```

正如我们之前所说的，`size`变量也可以标记为 const。这将更清楚地表明变量不应该改变，但这不是必要的，因为它是一个副本。对大小的任何修改都不会改变主函数中`SIZE`的值。因此，许多程序员，即使是那些追求 const 正确性的程序员，也不会将*按值传递*的参数标记为 const。

# 常量类作为参数

我们已经讨论了将数组传递给函数时的默认行为。编译器会自动传递数组的第一个元素的指针。这对速度和灵活性都有好处。由于只传递了一个指针，编译器不需要花费时间复制一个（可能）大的数组。这也更加灵活，因为函数可以处理所有大小的数组，而不仅仅是特定大小的数组。

不幸的是，当将结构体或类传递给函数时，默认行为是*按值传递*。我们说不幸的是，因为这会自动调用复制构造函数，这可能既昂贵又没有必要，如果函数只是从数据类型中读取数据。遵循的一个好的一般规则是，当将结构体或类传递给函数时，不要按值传递，而是通过指针或引用传递。这避免了可能昂贵的复制数据。当然，这个规则肯定有例外，但 99%的情况下，按值传递是错误的做法：

```cpp
//Simplified GameObject struct 
struct GameObject 
{ 
M5Vec2 pos; 
M5Vec2 vel; 
int    textureID; 
std::list<M5Component*> components; 
std::string name; 
}; 

void DebugPrintGameObject(GameObject& gameObject) 
{ 
//Do printing  
gameObject.textureID = 0;//WHOOPS!!! 
} 

```

我们希望避免在将`GameObjects`传递给函数时调用昂贵的复制构造函数。不幸的是，当我们通过指针或引用传递时，函数可以访问我们的公共数据并修改它。正如之前所做的那样，解决方案是通过指针传递到`const`或通过`const`引用传递：

```cpp
void DebugPrintGameObject(const GameObject& gameObject) 
{ 
//Do printing  
gameObject.textureID = 0;//ERROR: gameObject is const 
} 

```

在编写函数时，如果目的是修改数据，那么你应该通过引用传递。然而，如果目的不是修改数据，那么通过`const`引用传递。这样你将避免昂贵的复制构造函数调用，并且数据将受到意外修改的保护。此外，通过养成通过引用或`const`引用传递的习惯，你的代码将是自文档化的。

# 常量成员函数

在之前的例子中，我们保持了结构体非常简单。由于结构体没有成员函数，我们只需要担心非成员函数何时想要修改数据。然而，面向对象编程建议我们不应该有公共数据。相反，所有数据都应该是私有的，并通过公共成员函数访问。让我们通过一个非常简单的例子来理解这个概念：

```cpp
class Simple 
{ 
public: 
Simple(void) 
{ 
  m_data = 0; 
  } 
void SetData(int data) 
{ 
  m_data = data; 
} 
int GetData(void) 
{ 
  return m_data; 
  } 
private: 
int m_data; 
}; 

int main(void) 
{ 
Simple s; 
const Simple cs; 

s.SetData(10);          //Works as Expected 
int value = s.GetData();//Works as Expected 

cs.SetData(10);         //Error as expected 
value = cs.GetData();   //Error: Not Expected 
return 0; 
} 

```

如预期的那样，当我们的类没有被标记为`const`时，我们可以使用`SetData`和`GetData`成员函数。然而，当我们把我们的类标记为`const`时，我们预期将无法使用`SetData`成员函数，因为它会修改数据。然而，出乎意料的是，即使它根本不会修改数据，我们也无法使用`GetData`成员函数。为了理解发生了什么，我们需要了解成员函数是如何被调用的以及成员函数是如何修改正确数据的。

每次调用非静态成员函数时，第一个参数总是隐藏的 `this` 指针。它是指向调用该函数的实例的指针。这个参数是 `SetData` 和 `GetData` 能够作用于正确数据的方式。`this` 指针在成员函数中是可选的，作为程序员，我们可以选择使用或不使用它：

```cpp
//Example showing the hidden this pointer. This code won't //compile 
Simple::Simple(Simple* this) 
{ 
  this->m_data = 0; 
} 
void Simple::SetData(Simple* this, int data) 
{ 
  this->m_data = data; 
} 
int Simple::GetData(Simple* this) 
{ 
  return this->m_data; 
} 

```

这完全正确。`this` 指针实际上是一个指向 `Simple` 类的 `const` 指针。我们之前没有讨论过 `const` 指针，但这仅仅意味着指针本身不能被修改，但它所指向的数据（即 `Simple` 类）是可以被修改的。这种区别很重要。指针是 `const` 的，但 `Simple` 类不是。实际的隐藏参数看起来可能像这样：

```cpp
//Not Real Code. Will Not Compile 
Simple::Simple(Simple* const this) 
{ 
  this->m_data = 0; 
} 

```

当我们遇到如下调用成员函数的代码：

```cpp
Simple s; 
s.SetData(10); 

```

编译器实际上将其转换成如下代码：

```cpp
Simple s; 
Simple::SetData(&s, 10); 

```

这就是为什么当我们尝试将 `const Simple` 对象传递给成员函数时会出现错误的原因。函数签名是不正确的。该函数不接受 `const Simple` 对象。不幸的是，由于 `this` 指针是隐藏的，我们无法简单地让 `GetData` 函数接受 `const Simple` 指针。相反，我们必须将函数标记为 `const`：

```cpp
//What we would like to do but can't 
int Simple::GetData(const Simple* const this); 

//We must mark the function as const 
   int Simple::GetData(void) const; 

```

我们必须在类内部也将函数标记为 `const`。注意，`SetData` 没有标记为 `const`，因为该函数的目的是修改类，但 `GetData` 被标记为 `const`，因为它只从类中读取数据。所以，我们的代码可能看起来像以下这样。为了节省空间，我们没有再次包含函数定义：

```cpp
class Simple 
{ 
  public: 
  Simple(void); 
  void SetData(int data); 
  int GetData(void) const; 
  private: 
  int m_data; 
}; 

int main(void) 
{ 
  Simple s; 
  const Simple cs; 

  s.SetData(10);          //Works as Expected 
  int value = s.GetData();//Works as Expected 

  cs.SetData(10);         //Error as expected 
  value = cs.GetData();   //Works as Expected 
  return 0; 
} 

```

如你所见，通过将 `GetData` 成员函数标记为 `const`，它可以在变量实例被标记为 `const` 时使用。将成员函数标记为 `const` 允许类与非成员函数正确地工作，这些非成员函数可能正在尝试保持 `const` 正确性。例如，一个非成员函数（可能由另一个程序员编写）试图通过使用 `GetData` 成员函数来显示 `Simple` 对象：

```cpp
//Const correct global function using member functions to access 
//the data 
void DisplaySimple(const Simple& s) 
{ 
  std::cout << s.GetData() << std::end; 
} 

```

由于 `DisplaySimple` 并不打算更改类中的数据，参数应该被标记为 `const`。然而，这段代码只有在 `GetData` 是 `const` 成员函数的情况下才能正常工作。

保持 `const` 正确性需要一点工作，一开始可能看起来有些困难。然而，如果你养成习惯，它最终会成为你编程的自然方式。当你保持 `const` 正确性时，你的代码会更干净、更安全、更具有自解释性，并且更灵活，因为你为 `const` 和非 `const` 实例做好了准备。一般来说，如果你的函数不会修改数据，就将参数标记为 `const`。如果成员函数不会修改类数据，就将成员函数标记为 `const`。

# `const` 相关问题

正如我们之前所说的，const 并非魔法。它并不能使你的代码 100% 安全和受保护。了解和理解与 const 参数和 const 成员函数相关的规则将有助于防止错误。然而，未能理解 const 的规则和行为可能会导致错误。

C++ 中 const 最大的问题是对于位运算 const 和逻辑 const 的误解。这意味着编译器将尝试确保通过该特定变量位和字节不会改变。这并不意味着那些位不会通过另一个变量来改变，这也不意味着你关心的数据不会改变。考虑以下代码：

```cpp
//Example of modifying const bits through different variables. 
int i = 0; 
const int& ci = i; 

ci = 10; //ERROR: can't modify the bits through const variable 
i  = 10; //OK. i is not const 

std::cout << ci << std::endl;//Prints 10 

```

在前面的例子中，`i` 不是 const，但 `ci` 是一个指向 const `int` 的引用。`i` 和 `ci` 都在访问相同的位。由于 `ci` 被标记为 const，我们不能通过该变量更改其值。然而，`i` 不是 const，所以我们有权修改其值。我们可以有多个 const 和非 const 变量指向同一地址，这对 const 成员函数有影响：

```cpp
class Simple 
{ 
public: 
     Simple(void); 
  int GetData(void) const; 
private: 
  int m_data; 
  Simple* m_this; 
}; 
Simple::Simple(void):m_data(0), m_this(this) 
{ 
} 
int Simple::GetData(void) const 
{ 
  m_this->m_data = 10; 
  return m_data; 
} 

int main(void) 
{ 
  const Simple s; 
  std::cout << s.GetData() << std::endl; 
  return 0; 
} 

```

在前面的代码中，我们给 `Simple` 类提供了一个指向自身的指针。这个指针可以在 const 成员函数中用来修改其数据。记住，在 const 成员函数中，`this` 指针被标记为 const，所以不能通过该变量更改数据。然而，正如在这个例子中，数据仍然可以通过另一个变量来更改。即使我们没有使用另一个变量，`const_cast` 的使用也可能允许我们更改数据：

```cpp
int Simple::GetData(void) const 
{ 
  const_cast<Simple*>(this)->m_data = 10; 
   m_data; 
} 

```

非常重要的是要理解，你永远不应该编写这样的代码。尝试使用 `const_cast` 或非 const 指针来修改 const 变量是未定义的行为。原始数据可能被放置在只读内存中，这样的代码可能会导致程序崩溃。也有可能编译器会优化掉不应更改的内存的多次读取。因此，旧值可能会被用于任何未来的计算。使用 `const_cast` 移除 `const` 是为了与旧的 C++ 库保持向后兼容。它*永远*不应该用来修改 const 值。如果有一份数据即使在类是 `const` 的情况下也需要修改，请使用 `mutable` 关键字。

即使避免未定义的行为，位运算的 const 也会让我们在与 const 成员变量打交道时遇到麻烦。考虑一个将包含一些动态内存的简单类。由于它包含动态内存和指针，我们应该添加拷贝构造函数、析构函数以及其他一些东西来防止内存泄漏和内存损坏，但现在我们将省略这些，因为它们对我们讨论 const 的内容不重要：

```cpp
class LeakyArray 
{ 
public: 
LeakyArray(int size) 
{ 
  m_array = new int[size]; 
  } 
void SetValue(int index, int value) 
{ 
  m_array[index] = value; 
} 
int GetValue(int index) const 
{ 
  //function is const so we can't do this 
  //m_array = 0; 

  //but we can do this!!!!!!! 
  m_array[index] = 0; 

  return m_array[index]; 

  } 
private: 
  int* m_array; 
}; 

```

正如你所见，位运算 const 只能阻止我们修改类内部的实际位。这意味着我们不能将`m_array`指向新的位置。然而，它并不能阻止我们修改数组中的数据。在 const 函数中，`GetValue`修改数组没有任何阻碍，因为数组数据不是类的一部分，只有指针是。大多数用户并不关心数据的位置，但他们期望 const 数组保持不变。

正如你所见，保持一致性、正确性并不能保证数据永远不会被修改。如果你勤奋地使用 const，并且理解和避免可能出现的错误，那么这些好处是值得的。

# 学习如何通过迭代改进你的游戏和代码设计

虽然想象它是这样很美好，但游戏并不是完全由设计师/开发者的头脑中产生的。一个游戏是由许多不同的人的不同想法组成的。在过去，人们可以用单一个人的力量开发游戏，但现在，由许多不同学科组成的团队更为常见，团队中的每个游戏开发者都有自己的想法，其中许多很好的想法可以为最终制作的产品做出贡献。但考虑到这一点，你可能想知道，在所有这些不同的变化之后，游戏是如何达到最终阶段的？答案是迭代。

# 游戏开发周期

游戏开发是一个过程，不同的人对这些步骤有不同的名称和/或短语，但大多数人可以同意，对于商业游戏开发，有三个主要阶段：

+   前期制作

+   制作阶段

+   后期制作

这些状态中的每一个都有它们自己的步骤。由于页面限制，我无法详细描述整个过程，但我们将重点关注开发的制作方面，因为这对我们的读者来说是最相关的内容。

如果你想要了解更多关于游戏开发过程的不同方面，请查看[`en.wikipedia.org/wiki/Video_game_development#Development_process`](https://en.wikipedia.org/wiki/Video_game_development#Development_process)。

在游戏开发过程中，你会看到很多公司使用敏捷开发流程，这种流程基于迭代原型设计，通过使用反馈和游戏迭代的精炼，逐渐增加游戏的功能集。许多公司都喜欢这种方法，因为每隔几周就可以玩到游戏的一个版本，并且可以在项目进行中做出调整。如果你听说过 Scrum，它是一种流行的敏捷软件开发方法，也是我在我的学生和游戏行业中使用的方法。

# 制作阶段

进入生产阶段后，我们已经为我们的项目提出了基本想法，并创建了我们的提案和游戏设计文档。现在我们有了这些信息，我们可以开始以下三个步骤：

+   原型制作

+   游戏测试

+   迭代

每个步骤都服务于一个有价值的过程，并且将按照这个顺序完成。我们将反复重复这些步骤，直到发布，因此了解它们是个好主意。

# 原型制作

原型制作就是以快速的方式制作你想法的最简单版本，以证明你的概念是否运作良好。对于一些人来说，他们会通过索引卡、纸张、筹码和板子来完成这个任务，这被称为纸质原型。这可以非常实用，因为你一开始不必考虑代码方面的事情，而是让你能够体验游戏的核心，而不需要所有那些精美的艺术和打磨。一个图形不好的游戏，只有当你添加内容时才会变得有趣。

当然，假设你已经购买了这本书，你很可能已经是一名开发者了，但将其视为一个选项仍然是个好主意。*杰西·谢尔*在他的书《游戏设计艺术：视角之书》中写到了纸质原型，他解释了如何制作《俄罗斯方块》的纸质原型。为此，你可以剪出纸板碎片，然后将它们堆在一起，随机抽取，然后沿着纸板滑动，这将是纸板的一部分。一旦完成了一行，你就可以拿起一把 X-Acto 刀，然后剪下碎片。虽然这不能给你完全相同的感觉，但它足以让你看到你是否使用了正确的形状，以及碎片应该以多快的速度落下。最大的优势是，你可以在 10 到 15 分钟内创建这个原型，而编程可能需要更长的时间。

对于那些没有成功的事情，用 30 分钟来证明比用一整天更有说服力。这同样适用于 3D 游戏，比如第一人称射击游戏，通过创建地图的方式与你在纸上和笔的角色扮演游戏（如海岸巫师的《龙与地下城》）中创建战斗遭遇战的方式相似（作为一个设计师学习如何玩是一个很好的事情，因为你可以了解如何讲述故事和开发有趣的遭遇战）。

原型的任务是证明你的游戏是否运作，以及它具体是如何运作的。不要只投资于一个特定的想法，而是创建多个小型原型，快速制作，不必担心它是否完美或是否是你能做得最好的。

关于构建原型和七天内创建的原型示例，例如关于*Goo 塔*的原型，它是独立游戏《Goo 世界》的原型，你可以查看[`www.gamasutra.com/view/feature/130848/how_to_prototype_a_game_in_under_7_.php?print=1`](http://www.gamasutra.com/view/feature/130848/how_to_prototype_a_game_in_under_7_.php?print=1)了解更多信息。

作为游戏开发者，最重要的技能之一是能够快速创建原型，看看它如何运作，然后对其进行测试。我们称这个过程为游戏想法的测试。

# 游戏测试

一旦我们有了一个原型，我们就可以开始游戏测试过程。在开发过程中尽快进行游戏测试，并且经常进行。一旦你有了一个可玩的东西，就让人来试玩。首先自己玩玩游戏，看看自己的感受如何。然后邀请一些朋友到你家里来，也让他们试玩。

经常发现我的学生在最初进行游戏测试时会有困难，他们可能因为项目尚未*准备好*或担心别人无法理解而犹豫是否展示他们的项目。或者他们知道项目尚未完成，所以认为自己已经知道应该做什么，因此没有必要进行游戏测试。我发现这通常是因为他们害羞，而作为开发者，你需要克服的第一个主要障碍就是能够向世界展示你的想法。

如果你的游戏测试者不是你的亲密朋友和家人，那么很可能会有人对游戏提出负面评价。这是好事。他们还会提到许多你已经知道你的游戏还没有或没有预算去做的事情。这不是你为自己辩护或解释事情为什么是这样的时间，而是一个接受这些观点并记录下来，以便你可以在未来考虑它们的时间。

作为游戏开发者，有一点需要注意，你可能是自己游戏最差的评判者，尤其是在刚开始的时候。很多时候我看到初出茅庐的开发者试图为自己的游戏中的问题辩解，声称那是他们的愿景，人们不理解因为它还没有进入最终游戏。作为游戏开发者，能够获取反馈、接受批评并评估是否值得改变是非常重要的一项技能。

# 进行游戏测试

既然我们知道进行游戏测试是多么有价值，你可能想知道如何进行一次游戏测试。首先，我确实想强调，你在你的项目进行游戏测试时在场至关重要。当他们玩游戏时，你可以看到不仅有人认为什么，还可以看到他们对事物的反应以及他们如何使用你的游戏。这是你发现什么做得好，什么做得不好的时候。如果由于某种原因，你无法亲自到场，让他们记录下自己玩游戏的过程，如果可能的话，包括在 PC 上和通过摄像头。

当有人来电脑上测试你的游戏时，你可能会想告诉他们一些关于你的项目的事情，比如控制、故事、机制，以及其他任何东西，但你应该抵制这些冲动。首先看看玩家在没有提示的情况下会做什么。这将给你一个想法，即玩家在所创造的环境中会自然想要做什么，以及需要解释得更清楚的地方。一旦他们玩了一段时间，并且你已经从那方面获得了所需的信息，然后你可以告诉他们一些事情。

在进行游戏测试时，尽可能从玩家那里获取尽可能多的信息是个好主意。当他们完成游戏后，询问他们喜欢什么，不喜欢什么，是否发现什么令人困惑的地方，他们在哪里卡住了，以及对他们来说最有趣的是什么。请注意，玩家说的话和他们实际做的事情是两回事，所以你必须在场并观察他们。让你的游戏被玩，并观察那些玩家的行为，这是你开始看到设计缺陷的地方，而观察人们的行为将展示他们如何体验你所创造的事物。在进行这项测试时，我看到很多人做了我预期相反的事情，并且没有理解我认为相当简单的东西。然而，在这个问题上，玩家并没有错，是我错了。玩家只能做他们从之前的游戏或游戏中的教学所知道的事情。

在游戏测试期间，你获得的所有信息都很重要。不仅包括他们所说的内容，还包括他们没有说的内容。一旦他们完成游戏，就给他们一份调查问卷填写。我发现使用 Google Sheets 来存储这些信息效果很好，而且设置起来并不困难，你还可以从这些硬数据中做出决策，而不必记住人们说了什么。此外，人们从 1 到 10 选择他们对游戏不同方面的喜爱程度，比要求他们写出对一切的看法要容易得多，而且不需要他们写段落信息（除非他们想在最后的评论部分这样做）。

如果你想看看一个示例测试表单，虽然这个表单是为桌面游戏设计的，但我认为它很好地简化了测试者提供有用信息的流程：[`www.reddit.com/r/boardgames/comments/1ej13y/i_created_a_streamlined_playtesting_feedback_form/`](https://www.reddit.com/r/boardgames/comments/1ej13y/i_created_a_streamlined_playtesting_feedback_form/).

如果你在寻找一些可以提出的问题的想法，*韦斯利·罗克霍兹*提供了一些可能对你有用的提问示例：[`www.gamasutra.com/blogs/WesleyRockholz/20140418/215819/10_Insightful_Playtest_Questions.php`](http://www.gamasutra.com/blogs/WesleyRockholz/20140418/215819/10_Insightful_Playtest_Questions.php).

此外，玩家提供反馈的顺序也很重要，因为它传达了不同事物对他们的重要性。你可能会发现原本打算作为主要机制的东西并不像其他东西那样吸引人/有趣。这是宝贵的反馈，你可能会决定专注于那个次要机制，就像我在多个项目中看到的那样。尽早这样做会更好，这样你就可以尽可能少地浪费时间。

# 迭代

在这一点上，我们已经进行了项目测试并收集了玩家的反馈，如果已经设置好了，我们还从数据和分析中获得了可以继续发展的信息。现在我们需要考虑这些信息，对我们的当前原型进行一些修改，然后再次进行测试。这就是开发中的迭代阶段。

在这个阶段，你需要考虑这个反馈并决定如何将其融入你的设计中。你需要决定应该改变什么，以及不应该改变什么。在这样做的时候，要记住项目的范围，现实地评估做出这些改变需要多长时间，并且愿意砍掉一些功能，即使是你喜欢的，以获得最好的项目。

在再次做出这些决定后，我们将再次创建一个新的原型，然后你将再次进行测试。然后再次迭代。然后构建另一个原型，在那里你将继续测试，移除那些不起作用的原型和项目效果不佳的功能。你还将尝试使用反馈添加新功能，并移除那些不再适合当前游戏状态的功能。你将不断重复这个周期，直到达到最终的发布版本！

如果你等待你的游戏变得*完美*后再发布，你永远不会发布它。游戏永远不会完成，它们只是被放弃了。如果项目已经足够好，你应该发布，因为只有当你发布一个项目时，你才能最终说你已经开发了一个游戏。

如果你想看看这个过程的例子以及它如何有助于一个标题，请查看：[`www.gamasutra.com/blogs/PatrickMorgan/20160217/265915/Gurgamoth_Lessons_in_Iterative_Game_Development.php`](http://www.gamasutra.com/blogs/PatrickMorgan/20160217/265915/Gurgamoth_Lessons_in_Iterative_Game_Development.php)。

# 达成里程碑

当你在进行商业游戏项目时，尤其是当你有发行商时，你通常会有一个需要遵守的日程表和需要达到的里程碑。里程碑是让每个人都知道游戏是否按计划进行的一种方式，因为某些事情需要在它们完成之前完成。未能达到里程碑通常是一件糟糕的事情，因为你的发行商通常只有在里程碑中包含所有商定的内容时才会支付你的团队。没有标准的里程碑时间表，因为每家公司都不同，但其中一些最常见的如下：

+   **First-playable**：这是可以玩的游戏的第一个版本。包含了游戏的主要机制，可以展示它是如何工作的。

+   **Alpha**：当你的游戏的所有功能都齐备时，称为功能完整。功能可以略有变化，并根据反馈和测试进行修订，但在这个阶段，未实现的功能可能会被删除，以确保按时完成标题。

+   **Beta**：游戏已经完成，所有资源和功能都已完善和完成。此时你只是在进行错误测试和修复可能阻止游戏发布的潜在问题。

+   **Gold**：这是游戏的最终版本，你将要么发布它，要么将其发送给发行商，以便在磁盘、卡带或你的设备使用的任何介质上创建副本。

请注意，每家公司都不同，这些里程碑对不同的人可能意味着不同的事情，所以在深入开发之前一定要明确。

# 学习何时在游戏中使用脚本

脚本语言是当你在具有多个学科的团队中工作时，对开发者非常有帮助的东西。但在我们深入探讨它们是什么以及它们是如何工作之前，以及使用脚本语言的优缺点之前，最好先了解一下代码执行的历史。

# 汇编语言简介

在幕后，我们在本书的整个过程中编写的所有代码都是一串零和一，表示我们的计算机处理器应该将哪些开关标记为开启和关闭。低级编程语言，如机器语言，使用这些开关来执行命令。这最初是编程的唯一方式，但我们已经开发出更易于阅读的语言来供我们使用。

从汇编语言开始，低级语言与语言的指令和机器代码的指令之间有着非常紧密的联系。虽然比一串`0`s 和`1`s 更易读，但编写代码仍然相当困难。例如，以下是一些用于在汇编语言中添加两个数字的汇编代码：

```cpp
        push    rbp 
        mov     rbp, rsp 
        mov     DWORD PTR [rbp-20], edi 
        mov     DWORD PTR [rbp-24], esi 
        mov     edx, DWORD PTR [rbp-20] 
        mov     eax, DWORD PTR [rbp-24] 
        add     eax, edx 
        mov     DWORD PTR [rbp-4], eax 
        nop 
        pop     rbp 
        ret 

```

每种计算机架构都有自己的汇编语言，因此使用低级语言编写代码的缺点是不具有可移植性，因为它们依赖于机器。在过去的岁月里，人们必须学习许多不同的语言，以便将你的程序移植到另一个处理器。随着功能需求随着时间的推移而增加，程序结构变得更加复杂，这使得程序员很难实现既高效又足够健壮的程序。

# 转向高级编程语言

作为程序员，我们天生懒惰，因此我们寻求使我们的工作变得更简单，或者更确切地说，找到我们时间的最佳利用方式。考虑到这一点，我们已经开发了其他高级语言，这些语言甚至更容易阅读。当我们说高级时，我们的意思是更接近人类思考的方式，或者更接近我们试图解决的问题。通过从我们的代码中抽象出机器细节，我们简化了编程任务。

# 介绍编译器

一旦我们完成了代码，我们就使用编译器将高级代码翻译成汇编语言，然后汇编语言将被转换成计算机可以执行的机器语言。之后，它将程序转换成用户可以运行的可执行文件。从功能上看，它看起来像这样：

![图片](img/00095.jpeg)

这有几个优点，因为它提供了对硬件细节的抽象。例如，我们不再需要直接与寄存器、内存、地址等打交道。这也使得我们的代码具有可移植性，我们可以使用相同的程序，并由不同的汇编器为使用它的不同机器进行翻译。这正是 C 语言之所以兴起并变得如此受欢迎的原因之一，因为它允许人们编写一次代码，然后它可以在任何地方运行。你可能已经注意到 Unity 在游戏开发中也采用了同样的思考方式，这也是我认为他们之所以成功的原因之一。

与编写汇编语言代码相比，这是一种更有效率的利用时间的方式，因为它允许我们创建更复杂的项目和机器，并且在大多数情况下，现代编译器如微软的编译器都能生成一些非常高效的汇编代码。这正是我们在本书中一直在使用的方法。

尽管在汇编语言中编写代码仍然有其好处。例如，在你用高级语言编写完你的游戏后，你可以开始分析它，看看游戏的哪些方面是瓶颈，然后确定是否将其重写为汇编语言会给你带来速度提升。使用低级语言的目的在于你可以获得一些实质性的速度优势。

对于一个真实生活中的例子，说明如何使用汇编语言来优化游戏引擎，请查看以下来自英特尔的文章：[`software.intel.com/en-us/articles/achieving-performance-an-approach-to-optimizing-a-game-engine/`](https://software.intel.com/en-us/articles/achieving-performance-an-approach-to-optimizing-a-game-engine/).

在运行前需要编译的代码编写中存在的问题之一是，随着项目规模的增加，编译时间也会增加。重新编译整个游戏可能需要几分钟到几小时，这期间你无法工作在项目上，否则你可能需要再次重新编译。这就是脚本语言可能有用的一部分原因。

# 脚本语言的介绍

脚本语言是一种允许为其编写脚本的编程语言。脚本是一种可以在不进行编译的情况下以几种不同方式执行的程序。脚本语言有时也被称为非常高级的编程语言，因为它们在高级别抽象，学习如何编写它们非常快。

脚本语言还有优点，可以处理程序员需要处理的大量事情，例如垃圾回收、内存管理和指针，这些通常会让非开发者感到困惑。即使是像 Unreal 4 的蓝图这样的视觉编辑器，也仍然是脚本语言，因为它完成了与书面语言相同的事情。

大多数游戏以某种形式使用脚本语言，但有些游戏可能使用得更多，例如 GameMaker 使用**Game Maker Language**（**GML**）进行逻辑处理。

# 使用解释器

要使用脚本语言，我们需要能够即时执行新代码。然而，与编译器不同，还有一种将代码转换为机器可以理解的方式，这被称为解释器。解释器不会生成程序，而是存在于程序的执行过程中。这个程序将执行以下操作之一：

+   直接执行源代码

+   将源代码翻译成某种高效的中间表示（代码），然后立即执行它

+   明确执行由解释器系统中的编译器生成的预编译代码

解释器逐行翻译，而编译器则一次性完成所有工作。

从视觉上看，它看起来有点像以下这样：

![图片](img/00096.jpeg)

正如你所见，**解释器**接收**源代码**和任何已接收的**输入**，然后输出其期望的结果。

# 即时编译

运行代码还有另一种方式，使用所谓的**即时编译器**，或简称**JIT**。JIT 缓存了之前已解释为机器代码的指令，并重用这些原生机器代码指令，从而通过不必重新解释已解释的语句来节省时间和资源。

从视觉上看，它看起来类似这样：

![图片](img/00097.jpeg)

现在，Unity 使用即时编译（JIT）和预编译（AOT）编译器将代码转换为机器码，这样机器就可以读取。当函数第一次被调用时，游戏会将该代码转换为机器语言，然后下次调用时，它会直接跳转到翻译后的代码，因此你只需要对正在发生的事情进行转换。由于这是在运行时发生的，这可能会导致你在使用大量新功能时项目出现卡顿。

关于 Unity 游戏引擎内部脚本工作原理的精彩演讲可以在这里找到：[`www.youtube.com/watch?v=WE3PWHLGsX4`](https://www.youtube.com/watch?v=WE3PWHLGsX4)。

# 为什么使用脚本语言？

当涉及到为你的游戏构建工具或处理由技术设计师处理的高级游戏任务时，C++通常过于强大。它确实有一些开发上的优势。具体来说，你不必担心很多底层的事情，因为语言会为你处理这些；程序员由于选项有限，错误也更少。需要的编程知识更少，并且可以根据游戏需求进行定制。这也使得游戏更注重数据驱动，而不是将东西硬编码到游戏引擎中，并允许你在不发送整个项目的情况下修补游戏。

在游戏开发中，游戏逻辑和配置通常可以在脚本文件中找到。这样，非程序员（如设计师）很容易修改和调整脚本，允许他们进行游戏测试和调整游戏玩法，而无需重新编译游戏。

许多游戏也都有一个控制台窗口，它使用脚本语言在运行时执行此类操作。例如，当你按下 Tab 键时，Unreal Engine 会默认打开控制台窗口，而在 Source 引擎中，在暂停菜单中按下~按钮也会打开一个控制台窗口。

脚本语言也常用于具有关卡设计的领域，例如，当进入某些区域时触发器，或控制电影场景。它还允许你的游戏玩家对游戏进行修改，这可能会增加游戏的生命周期并有助于培养你的游戏社区。

# 何时使用 C++

C++是一个很好的语言选择，因为性能是一个关键的第一步。这曾经是游戏引擎的所有方面，但现在主要用于图形和 AI 代码。脚本语言也存在比 C++慢的问题，有时甚至比其他情况慢 10 倍。由于脚本语言自动处理内存管理，有时命令可能会中断或需要一段时间才能完成垃圾回收，导致卡顿和其他问题。

C++也有更好的 IDE 和调试器的优势，这使得你在出错时更容易找到并修复错误。

还有一种可能性，就是你正在处理一个遗留代码库。大多数游戏公司并不是从一张白纸开始。利用 C++的中间件库，如 FMOD 和 AntTweakBar，也可能很有用。

# 编译与脚本

对于某些游戏引擎，游戏引擎本身是用 C++编写的，但游戏逻辑完全是使用脚本语言完成的，例如 Unity 的大多数开发。这允许你更快地迭代游戏玩法，并允许技术设计师和艺术家在不打扰程序员的情况下修改行为。此外，根据语言的不同，它还可以允许人们使用更适合问题域的语言（例如，AI 可能不是在 C++中实现的最容易的事情）。

不同的公司处理与语言合作的方式不同。当我在一间 AAA（发音为三 A）工作室工作时，我们会让设计师为机制原型设计想法，并尽可能好地使用脚本语言实现它们。一旦得到负责人的批准，如果脚本存在性能问题，程序员会以脚本语言代码为基础，然后创建一个超级高效的版本，使用 C++实现，使其在所有级别上都能工作。然而，当我为一个独立游戏项目工作时，所有的代码都是用脚本语言（C#）编写的，因为我们没有访问引擎源代码（Unity）。此外，如果你想要针对内存和处理能力有限的设备（如任天堂 3DS），你可能会更加关注性能，因此使用更优化的代码就更加重要。熟悉这两种选项并且能够舒适地以任何一种方式工作是个好主意。

如果你对你的项目有兴趣使用脚本语言，Lua 在游戏行业中非常广泛使用，因为它非常容易学习，并且相对容易集成到你的引擎中。Lua 最初是一个配置语言。这有一些很好的特性，比如它非常适合创建和配置事物——这正是你在游戏中想要做的。不过，需要注意的是，它不是面向对象的，但使用少量的内存。

使用 Lua 作为脚本语言的游戏列表可以在这里找到：[`en.wikipedia.org/wiki/Category%3aLua-scripted_video_games`](https://en.wikipedia.org/wiki/Category%3aLua-scripted_video_games)。

如果你有兴趣将 Lua 集成到你的项目中或者想看看它是如何工作的，我强烈建议查看[`www.lua.org/start.html`](http://www.lua.org/start.html)。

# 摘要

在本章中，我们涵盖了许多最佳实践信息，希望这将为您在将来构建自己的项目时提供一个良好的基础。我们讨论了为什么硬编码值是个坏主意，并提出了许多其他关于代码质量的建议，以确保您的代码易于理解，也易于在将来需要时进行扩展。

我们还学习了迭代在游戏开发中的有用性，讨论了传统的游戏开发周期，以及关于游戏测试的技巧和窍门，以及它在开发项目时如何非常有用。

我们还探讨了低级和高级编程语言，了解了脚本语言是如何在我们必须将其构建到项目中的另一个程序内部运行的。它们不是编译的，而是解释的，通常比编译语言更容易使用和编写代码，但代价是性能。根据您的游戏复杂程度，坚持使用 C++可能是个好主意，但如果您与设计师合作，给他们提供自己动手的工具可能非常有用。

有了这些，我们就到达了这本书的结尾。我们希望您觉得这些信息既有趣又实用。当您出去构建自己的项目时，请利用我们在过去 12 章中讨论的设计模式和最佳实践，制作出最好的游戏！
