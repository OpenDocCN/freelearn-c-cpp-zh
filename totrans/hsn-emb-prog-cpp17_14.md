# 为混合 SoC/FPGA 系统开发

除了基于标准 CPU 的嵌入式系统之外，一种越来越常见的做法是将 SoC 形式的 CPU 与**现场可编程门阵列**（**FGPAs**）相结合。这允许在系统的 FPGA 部分实现 CPU 密集型算法和处理，包括 DSP 和图像处理，而 CPU 端处理较不密集的任务，如用户交互、存储和网络。

在本章中，我们将涵盖以下主题：

+   如何与混合 FPGA/SoC 系统的 FPGA 端进行通信

+   学习如何在 FPGA 中实现各种算法，并从 SoC 端使用它们

+   如何在混合 FPGA/SoC 系统上实现基本示波器

# 极度并行化

当谈到性能时，在单核处理器上一次执行一条指令基本上是您可以实施算法或其他功能的最慢方式。从这里，您可以通过在单个处理器核心的各个功能单元上同时调度来将这种单一执行流程扩展到多个流程。

提高性能的下一步是添加更多核心，这当然会使调度更加复杂，并引入潜在的延迟问题，因为次要任务阻塞资源，导致关键任务被推迟。对于某些任务，特别是那些令人尴尬的并行任务，使用通用处理器也非常有限。

对于必须使用对集合中每个元素应用相同算法的单个大数据集进行处理的任务，基于**通用图形处理器单元**（**GPGPU**）的处理以及使用**数字信号处理器**（**DSPs**）通过使用专用硬件大幅加快一系列操作的使用变得非常流行。

在这个问题的另一边是任务，这些任务是高度并行的，但涉及在传入数据、内部数据或两者上执行许多不同的操作。这种复杂程度，如果仅用软件在一系列微处理器核心上实现，将非常难以获得任何合理的性能。

使用昂贵的 DSP 硬件可能有所帮助，但即使那样，也不会针对该任务进行优化。传统上，这可能是公司考虑设计并生产定制**集成电路**（**IC**）作为**专用集成电路**（**ASIC**）的点。然而，这种成本非常高，只有在大规模生产中才是现实的，这样它才能与其他选项竞争。

随着时间的推移，发明了不同的解决方案，以使这种定制硬件实现更加现实，其中之一是可编程逻辑芯片的发展。例如，Commodore 64 这样的系统包含一个**PLA**（可编程逻辑阵列的简称，最初是 Signetics 82S100）芯片，这是一个一次性可编程的组合逻辑元件阵列。它允许处理器重新配置地址总线上的板载路由，以改变哪些部分 DRAM 内存芯片、ROM 芯片和其他外围设备处于活动寻址空间中。

编程 PLA 后，它基本上以与大量 74 逻辑芯片（离散逻辑芯片）相同的方式工作，但所需的离散解决方案空间仅为其一小部分。这种方法实际上为 Commodore 提供了他们自己的定制 ASIC，但无需投资设计和生产。相反，他们使用现成的部件，并且可以在 Commodore 64 的整个生命周期内对烧录到 PLA 芯片中的逻辑进行改进。

随着时间的推移，，PLA（也称为 PALs）变得更加先进，发展成为**复杂可编程逻辑器件**（CPLDs），它们基于宏单元，允许实现比简单的组合逻辑更高级的功能。这些最终演变成了 FPGA，它们再次增加了更多高级功能和外围设备。

这些天，FPGA 几乎无处不在，只要有某种高级处理或控制需求。视频和音频处理设备通常与 DSP 一起使用 FPGA，由微控制器（MCU）或片上系统（SoC）处理用户界面和其他低优先级功能。

今天，示波器等设备采用模拟（如果支持，则包括数字）前端，由数字信号处理器（DSPs）进行原始数据转换和该数据的初步处理，然后将数据传递给一个或多个现场可编程门阵列（FPGAs），这些 FPGAs 对数据进行进一步的处理和分析。处理完毕后，这些数据可以存储在缓冲区中（数字存储示波器（DSO）的“数字存储”部分）以及传递到前端，在那里运行在片上系统（SoC）上的软件将在用户界面中渲染它，并允许用户输入命令来操作显示的数据。

在本章中，我们将探讨一个基本的示波器项目，该项目将使用简单的硬件和 VHDL 代码编程的 FPGA 来实现。

# 硬件描述语言

随着过去几十年中**超大规模集成电路**（VLSI）的复杂性增加，找到改进开发过程的方法变得越来越重要，包括验证设计的能力。这导致了**硬件描述语言**（HDLs）的发展，其中 VHDL 和 Verilog 是目前最常用的两种。

HDLs（硬件描述语言）的主要目的是允许开发者轻松描述那些将被集成到 ASICs 中或用于编程 FPGAs 的硬件电路。此外，这些 HDLs 还使得设计模拟和验证其功能正确性成为可能。

在本章中，我们将探讨一个使用 VHDL 实现 FPGA 编程侧的示例。**VHSIC 硬件描述语言**（**VHDL**）作为一种语言首次出现在 1983 年，当时由美国国防部开发。它的目的是作为一种方式来记录供应商提供的与设备一起提供的 ASICs 的行为。

随着时间的推移，有人提出这些文档文件可以用来模拟 ASICs 的行为。这一发展很快就被合成工具的开发所跟随，以创建一个功能硬件实现，可以用来创建 ASICs。

VHDL 语言在很大程度上基于 Ada 编程语言，而 Ada 编程语言本身也起源于美国军事。尽管 VHDL 主要用作 HDL，但它也可以用作通用编程语言，类似于 Ada 及其同类语言。

# FPGA 架构

虽然不是每个 FPGA 的结构都相同，但一般原则是相同的：它们是由逻辑元素组成的阵列，可以配置成特定的电路。因此，这些**逻辑元素**（**LEs**）的复杂性决定了可以形成哪种逻辑电路，这在为特定 FPGA 架构编写 VHDL 代码时必须考虑。

**逻辑元素**（**LEs**）和**逻辑单元**（**LCs**）这两个术语可以互换使用。一个 LE 由一个或多个**查找表**（**LUTs**）组成，一个 LUT 通常有四个到六个输入。无论具体配置如何，每个 LE 都由互连逻辑包围，这允许不同的 LE 相互连接，并且 LE 本身被编程到特定的配置，从而形成预期的电路。

为 FPGA 开发可能存在的潜在问题包括 FPGA 制造商强烈假设 FPGA 将用于带时钟的设计（使用中央时钟源和时钟域），而不是组合逻辑（无时钟）。一般来说，在将目标 FPGA 系统包含在新项目之前熟悉它是一个好主意，以了解它如何支持你所需要的特性。

# 混合 FPGA/SoC 芯片

尽管包含 FPGA 和 SoC 的系统多年来一直非常普遍，但最近的一个新加入的是混合 FPGA/SoC 芯片，这些芯片在同一封装中包含 FPGA 和 SoC（通常是基于 ARM）的晶圆。然后通过总线将它们连接起来，以便两者可以高效地通过内存映射 I/O 和类似方式相互通信。

目前此类 FPGA 的常见例子包括 Altera（现在是英特尔）、Cyclone V SoC 和 Xilinx Zynq。Cyclone V SoC 的官方数据手册中的框图给出了此类系统工作原理的良好概述：

![](img/00dec7ee-3feb-4111-97b6-898f96f9b83c.png)

在这里，我们可以看到**硬处理器系统**（**HPS**）和 FPGA 侧之间有多种通信方式，例如通过共享 SDRAM 控制器、两个点对点链路和多个其他接口。对于 Cyclone V SoC，系统启动时可以是 FPGA 或 SoC 侧的第一个启动侧，从而提供广泛的系统配置选项。

# 示例 – 基本示波器

本例提供了一个基本概述，说明如何在嵌入式项目中使用 FPGA。它使用 FPGA 来采样输入并测量电压或类似内容，就像示波器一样。然后，将生成的 ADC 数据通过串行链路发送到基于 C++/Qt 的应用程序，该程序显示数据。

# 硬件

对于该项目，我们将使用 Fleasystems FleaFPGA Ohm 板([`fleasystems.com/fleaFPGA_Ohm.html`](http://fleasystems.com/fleaFPGA_Ohm.html))。这是一款小型、低于 50 美元、低于 40 欧元的 FPGA 开发板，采用树莓派 Zero 外形：

![](img/0fb4d23c-b3e9-437b-8891-ac36d9abeb84.png)

它具有以下规格：

+   **Lattice ECP5 FPGA**，具有 24K LUT 元素和 112KB 块 RAM。

+   **256-Mbit SDRAM**，16 位宽，167 MHz 时钟。

+   **8-Mbit SPI 闪存**用于 FPGA 配置存储。

+   25 MHz 晶振振荡器。

+   **HDMI 视频输出**（支持高达 1080p30 或 720p60 的屏幕模式）。

+   **μSD 卡槽**。

+   两个微 USB 主机端口，具有备用 PS/2 主机端口功能。

+   29 个用户 GPIO，包括来自（与树莓派兼容的）40 引脚扩展的 4 个中等速度 ADC 输入和 12 个 LVDS 信号对，以及 2 引脚复位引脚。

+   一个微 USB 从端口。为 Ohm 提供+5V 电源馈电，串行控制台/UART 通信，以及访问板上的 JTAG 编程接口（用于配置 ECP5 FPGA）。

+   提供外部 JTAG 编程模块，以便进行实时调试。

我们将电路连接到该板，以便我们可以连接示波器探头：

![](img/be8609a6-c88d-4c76-95aa-3bfdebfdb919.png)

该电路将连接到 Ohm 板 GPIO 引脚的 29 号引脚，对应 GPIO 5。它允许我们测量 0 到 3V 直流信号，以及 1.5V 交流（均方根值），在 1 x 探头测量模式下。带宽略超过 10 MHz。

# VHDL 代码

在本节中，我们将查看 VHDL 项目中的顶层实体，以了解其功能。这从 VHDL 的标准库包含开始，如下所示：

```cpp
library IEEE; 
use IEEE.STD_LOGIC_1164.ALL; 
use IEEE.std_logic_unsigned.ALL; 
use IEEE.numeric_std.all; 

entity FleaFPGA_Ohm_A5 is   
   port( 
   sys_clock         : in        std_logic;  -- 25MHz clock input from external xtal oscillator. 
   sys_reset         : in        std_logic;  -- master reset input from reset header. 
```

这对应于底层 FPGA 的系统时钟和复位线。我们还可以看到端口映射的方式，定义了实体端口的方向和类型。在这里，类型是`std_logic`，它是一个标准的逻辑信号，可以是二进制的一位或零：

```cpp
   n_led1                  : buffer    std_logic; 

   LVDS_Red          : out       std_logic_vector(0 downto 0); 
   LVDS_Green        : out       std_logic_vector(0 downto 0); 
   LVDS_Blue         : out       std_logic_vector(0 downto 0); 
   LVDS_ck                 : out       std_logic_vector(0 downto 0); 

   slave_tx_o        : out       std_logic; 
   slave_rx_i        : in        std_logic; 
   slave_cts_i       : in        std_logic;  -- Receive signal from #RTS pin on FT230x 
```

我们还使用板上的状态 LED，映射 HDMI 的视频引脚（LVDS 信号），以及 UART 接口，该接口使用板上的 FDTI USB-UART 芯片。后者是我们将用于从 FPGA 向 C++应用程序发送数据的方式。

接下来，是 Raspberry Pi 兼容的引脚映射，如下面的代码所示：

```cpp
   GPIO_2                  : inout           std_logic; 
   GPIO_3                  : inout           std_logic; 
   GPIO_4                  : inout           std_logic; 
   -- GPIO_5               : inout           std_logic; 
   GPIO_6                  : inout           std_logic;   
   GPIO_7                  : inout           std_logic;   
   GPIO_8                  : inout           std_logic;   
   GPIO_9                  : inout           std_logic;   
   GPIO_10                 : inout           std_logic; 
   GPIO_11                 : inout           std_logic;   
   GPIO_12                 : inout           std_logic;   
   GPIO_13                 : inout           std_logic;   
   GPIO_14                 : inout           std_logic;   
   GPIO_15                 : inout           std_logic;   
   GPIO_16                 : inout           std_logic;   
   GPIO_17                 : inout           std_logic; 
   GPIO_18                 : inout           std_logic;   
   GPIO_19                 : inout           std_logic;   
   GPIO_20                 : in        std_logic; 
   GPIO_21                 : in        std_logic;   
   GPIO_22                 : inout           std_logic;   
   GPIO_23                 : inout           std_logic; 
   GPIO_24                 : inout           std_logic; 
   GPIO_25                 : inout           std_logic;   
   GPIO_26                 : inout           std_logic;   
   GPIO_27                 : inout           std_logic; 
   GPIO_IDSD         : inout           std_logic; 
   GPIO_IDSC         : inout           std_logic; 
```

GPIO 5 被注释掉的原因是我们想用它来实现 ADC 功能，而不是通用输入/输出。

相反，我们使具有 sigma-delta 功能的 ADC3 外设在该引脚上工作如下：

```cpp
   --ADC0_input      : in        std_logic; 
   --ADC0_error      : buffer    std_logic; 
   --ADC1_input      : in        std_logic; 
   --ADC1_error      : buffer    std_logic; 
   --ADC2_input      : in        std_logic; 
   --ADC2_error      : buffer    std_logic; 
   ADC3_input  : in        std_logic; 
   ADC3_error  : buffer    std_logic; 
```

在这里，我们看到我们还有另外三个 ADC 外设，如果我们想为示波器添加更多通道，可以使用它们，如下面的代码所示：

```cpp
   mmc_dat1          : in        std_logic; 
   mmc_dat2          : in        std_logic; 
   mmc_n_cs          : out       std_logic; 
   mmc_clk           : out       std_logic; 
   mmc_mosi          : out       std_logic; 
   mmc_miso          : in        std_logic; 

   PS2_enable        : out       std_logic; 
   PS2_clk1          : inout           std_logic; 
   PS2_data1         : inout           std_logic; 

   PS2_clk2          : inout           std_logic; 
   PS2_data2         : inout           std_logic 
   ); 
end FleaFPGA_Ohm_A5; 
```

顶层实体定义以 MMC（SD 卡）和 PS2 接口结束。

接下来是模块的架构定义。这部分类似于 C++应用程序的源文件，实体定义的作用类似于头文件，如下所示：

```cpp
architecture arch of FleaFPGA_Ohm_A5 is 
   signal clk_dvi  : std_logic := '0'; 
   signal clk_dvin : std_logic := '0'; 
   signal clk_vga  : std_logic := '0'; 
   signal clk_50  : std_logic := '0'; 
   signal clk_pcs   : std_logic := '0'; 

   signal vga_red     : std_logic_vector(3 downto 0) := (others => '0'); 
   signal vga_green   : std_logic_vector(3 downto 0) := (others => '0'); 
   signal vga_blue    : std_logic_vector(3 downto 0) := (others => '0'); 

   signal ADC_lowspeed_raw     : std_logic_vector(7 downto 0) := (others => '0'); 

   signal red     : std_logic_vector(7 downto 0) := (others => '0'); 
   signal green   : std_logic_vector(7 downto 0) := (others => '0'); 
   signal blue    : std_logic_vector(7 downto 0) := (others => '0'); 
   signal hsync   : std_logic := '0'; 
   signal vsync   : std_logic := '0'; 
   signal blank   : std_logic := '0'; 
```

在这里定义了多个信号。这些信号使我们能够将 VHDL 模块的端口、实体、进程和其他元素相互连接。

我们可以看到，这里定义了一些信号以支持 VGA。这允许与启用 VGA 的 FPGA 板兼容，但其中一部分也与 HDMI（或 DVI）外设兼容，正如我们稍后将看到的。让我们看看下面的代码：

```cpp
begin 
 Dram_CKE <= '0';    -- DRAM Clock disable. 
 Dram_n_cs <= '1';   -- DRAM Chip disable. 
 PS2_enable <= '1'; -- Configures both USB host ports for legacy PS/2 mode. 
 mmc_n_cs <= '1';    -- Micro SD card chip disable. 
```

使用`begin`关键字，我们表明这是我们要开始执行架构定义中命令的点。除非一个指令块被封装在一个`process`（在此代码中未显示）中，否则此关键字之后以及终止关键字（`end architecture`）之后的所有内容都将同时执行。

我们通过写入适当的引脚来禁用多个硬件功能。为了简洁起见，我们在早期的实体定义中省略了 DRAM（外部内存）部分。禁用了 DRAM 和 SD 卡功能，同时启用了 PS2（键盘、鼠标）功能。这允许我们在需要时连接 PS2 输入设备：

```cpp
 user_module1 : entity work.FleaFPGA_DSO 
    port map( 
         rst => not sys_reset, 
         clk => clk_50, 
         ADC_1 => n_led1, 
         ADC_lowspeed_raw => ADC_lowspeed_raw, 
         Sampler_Q => ADC3_error, 
         Sampler_D => ADC3_input, 
         Green_out => vga_green, 
         Red_out => vga_red, 
         Blue_out => vga_blue, 
         VGA_HS => hsync, 
         VGA_VS => vsync, 
         blank => blank, 
         samplerate_adj => GPIO_20, 
         trigger_adj => GPIO_21 
    ); 
```

在这里，我们定义我们将使用 FleaFPGA 数字存储示波器模块的一个实例。尽管该模块可以支持四个通道，但只有第一个通道被映射。这种简化有助于演示操作原理。

DSO 模块负责读取 ADC 的数据，同时它采样我们用探针测量的信号，并将它渲染到本地缓存以在本地（HDMI 或 VGA）显示器上显示，并通过串行接口发送到 UART 模块（在本节末尾显示）。让我们看看下面的代码：

```cpp
   red <= vga_red & "0000"; 
   green <= vga_green & "0000"; 
   blue <= vga_blue & "0000"; 
```

在这里，最终显示输出颜色由 HDMI 输出信号确定：

```cpp
 u0 : entity work.DVI_clkgen 
   port map( 
         CLKI              =>    sys_clock, 
         CLKOP             =>    clk_dvi, 
         CLKOS                   =>  clk_dvin, 
         CLKOS2                  =>  clk_vga, 
         CLKOS3                  =>  clk_50 
         );   

   u100 : entity work.dvid PORT MAP( 
      clk       => clk_dvi, 
      clk_n     => clk_dvin, 
      clk_pixel => clk_vga, 
      red_p     => red, 
      green_p   => green, 
      blue_p    => blue, 
      blank     => blank, 
      hsync     => hsync, 
      vsync     => vsync, 
      -- outputs to TMDS drivers 
      red_s     => LVDS_Red, 
      green_s   => LVDS_Green, 
      blue_s    => LVDS_Blue, 
      clock_s   => LVDS_ck 
   ); 
```

整个这一部分用于输出由 DSO 模块生成的视频信号，使我们也能将 FPGA 板作为独立的示波器单元使用：

```cpp
   myuart : entity work.simple_uart 

         port map( 
               clk => clk_50, 
               reset => sys_reset, -- active low 
               txdata => ADC_lowspeed_raw, 
               --txready => ser_txready, 
               txgo => open, 
               --rxdata => ser_rxdata, 
               --rxint => ser_rxint, 
               txint => open, 
               rxd => slave_rx_i, 
               txd => slave_tx_o 
         ); 
end architecture; 
```

最后，简单的 UART 实现允许 DSO 模块与我们的 C++应用程序通信。

UART 配置为以 19,200 波特率、8 位、1 停止位和无奇偶校验位工作。在构建此 VHDL 项目并用它编程 FPGA 板之后，我们可以通过这个串行连接连接到它。

# C++代码

虽然 VHDL 代码实现了简单的显示输出和基本输入选项，但如果我们要有一个大（高分辨率）的显示，进行信号分析，记录几分钟甚至几小时的数据等，那么在 SBC 上执行这些操作将非常方便。

以下代码是一个 C++/Qt 图形应用程序，它从 FPGA 板接收原始 ADC 数据并在图表中显示。虽然基础，但它为功能齐全、基于 SoC 的系统提供了框架。

首先，展示头文件如下：

```cpp
#include <QMainWindow> 

#include <QSerialPort> 
#include <QChartView> 
#include <QLineSeries> 

namespace Ui { 
    class MainWindow; 
} 

class MainWindow : public QMainWindow { 
    Q_OBJECT 

public: 
    explicit MainWindow(QWidget *parent = nullptr); 
    ~MainWindow(); 

public slots: 
    void connectUart(); 
    void disconnectUart(); 
    void about(); 
    void quit(); 

private: 
    Ui::MainWindow *ui; 

    QSerialPort serialPort; 
    QtCharts::QLineSeries* series; 
    quint64 counter = 0; 

private slots: 
    void uartReady(); 
}; 
```

在这里，我们可以看到我们将使用 Qt 中的串行端口实现以及 QChart 模块来进行可视化部分。

实现如下所示：

```cpp
#include "mainwindow.h" 
#include "ui_mainwindow.h" 

#include <QSerialPortInfo> 
#include <QInputDialog> 
#include <QMessageBox> 

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), 
    ui(new Ui::MainWindow) { 
    ui->setupUi(this); 

    // Menu connections. 
    connect(ui->actionQuit, SIGNAL(triggered()), this, SLOT(quit())); 
    connect(ui->actionConnect, SIGNAL(triggered()), this, SLOT(connectUart())); 
    connect(ui->actionDisconnect, SIGNAL(triggered()), this, SLOT(disconnectUart())); 
    connect(ui->actionInfo, SIGNAL(triggered()), this, SLOT(about())); 

    // Other connections 
    connect(&serialPort, SIGNAL(readyRead()), this, SLOT(uartReady())); 

    // Configure the chart view. 
    QChart* chart = ui->chartView->chart(); 
    chart->setTheme(QChart::ChartThemeBlueIcy); 
    chart->createDefaultAxes(); 
    series = new QtCharts::QLineSeries(chart); 
    chart->setAnimationOptions(QChart::NoAnimation);         
    chart->addSeries(series); 
} 
```

在构造函数中，我们创建与 GUI 中的菜单选项的连接，这些选项允许我们退出应用程序、连接到串行端口、如果我们已连接则断开与串行端口的连接，或获取有关应用程序的信息。

我们将串行端口实例连接到一个槽位，每当有新数据准备好读取时，该槽位会被调用。

最后，我们在 GUI 中配置图表视图，获取 QChartView 小部件内 QChart 实例的引用。在这个引用上，我们为图表设置主题，添加默认轴，并最终添加一个空序列，我们将用从 FPGA 接收到的数据填充它，如下所示：

```cpp
MainWindow::~MainWindow() { 
    delete ui; 
} 

void MainWindow::connectUart() { 
    QList<QSerialPortInfo> comInfo = QSerialPortInfo::availablePorts(); 
    QStringList comNames; 
    for (QSerialPortInfo com: comInfo) { 
        comNames.append(com.portName()); 
    } 

    if (comNames.size() < 1) { 
        QMessageBox::warning(this, tr("No serial port found"), tr("No serial port was found on the system. Please check all connections and try again.")); 
        return; 
    } 

    QString comPort = QInputDialog::getItem(this, tr("Select serial port"), tr("Available ports:"), comNames, 0, false); 

    if (comPort.isEmpty()) { return; } 

    serialPort.setPortName(comPort); 
    if (!serialPort.open(QSerialPort::ReadOnly)) { 
        QMessageBox::critical(this, tr("Error"), tr("Failed to open the serial port.")); 
        return; 
    } 

    serialPort.setBaudRate(19200); 
    serialPort.setParity(QSerialPort::NoParity); 
    serialPort.setStopBits(QSerialPort::OneStop); 
    serialPort.setDataBits(QSerialPort::Data8); 
} 
```

当用户希望通过 UART 连接到 FPGA 时，必须选择连接 FPGA 的串行连接，之后将建立连接，使用我们在项目 VHDL 部分之前设置的 19,200 波特率、8N1 设置。

对于串行端口始终相同的固定配置，当系统启动时可以考虑自动化以下部分：

```cpp
void MainWindow::disconnectUart() { 
    serialPort.close(); 
} 
```

断开与串行端口的连接相当直接：

```cpp
void MainWindow::uartReady() { 
    QByteArray data = serialPort.readAll(); 

    for (qint8 value: data) { 
        series->append(counter++, value); 
    } 
} 
```

当 UART 从 FPGA 板接收到新数据时，这个槽位会被调用。在其中，我们读取 UART 缓冲区中的所有数据，将其附加到我们添加到图形小部件的序列中，从而更新显示的轨迹。计数器变量用于为图表提供递增的时间基准。这里它作为一个简单的时间戳。

在某个时候，我们应该开始从序列中删除数据，以防止它变得太大，同时具备搜索和保存数据的能力。基于计数的时间戳可以报告我们接收信号的实际时间，尽管理想情况下这应该是我们从 FPGA 接收到的数据的一部分：

```cpp
void MainWindow::about() { 
    QMessageBox::aboutQt(this, tr("About")); 
} 

void MainWindow::quit() { 
    exit(0); 
} 
```

我们以几个简单的槽位结束。对于信息对话框，我们简单地显示标准的 Qt 信息对话框。这可以被自定义的帮助或信息对话框所替代。

# 构建项目

使用免费的 Lattice Semiconductor Diamond IDE 软件 ([`www.latticesemi.com/latticediamond`](http://www.latticesemi.com/latticediamond))，可以将 VHDL 项目构建并编程到 Ohm FPGA 板上。编程该板需要安装来自 [`github.com/Basman74/FleaFPGA-Ohm`](https://github.com/Basman74/FleaFPGA-Ohm) 的 FleaFPGA JTAG 工具，以便 Diamond 可以使用它。

按照快速入门指南中描述的 FleaFPGA Ohm 板的说明，应该相对容易使该项目的这部分运行起来。对于 C++ 方面，必须确保 FPGA 板和 SBC（或等效）连接，以便后者可以访问前者的 UART。

在此基础上，只需使用 Qt 框架编译 C++ 项目（直接在 SBC 上或在桌面系统上交叉编译）就足够了。之后，可以在激活了闪存 FPGA 板的情况下运行应用程序，连接到 UART，并观察在应用程序窗口中绘制的跟踪。

# 摘要

在本章中，我们探讨了 FPGA 在嵌入式开发中的作用，以及它们在过去几十年中重要性的变化，以及它们现在的应用。我们查看了一个简单的示波器实现，它使用了一个 FPGA 和基于 SBC 的组件。阅读完本章后，你现在应该知道何时为新的嵌入式项目选择 FPGA，以及如何使用和与这样的设备进行通信。
