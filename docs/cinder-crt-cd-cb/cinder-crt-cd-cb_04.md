# 第四章。使用多媒体内容

在本章中，我们将学习以下内容：

+   加载和显示视频

+   创建一个简单的视频控制器

+   将窗口内容保存为图像

+   将窗口动画保存为视频

+   将窗口内容保存为矢量图形图像

+   使用瓦片渲染器保存高分辨率图像

+   在应用程序之间共享图形

# 简介

大多数有趣的应用都以某种形式使用多媒体内容。在本章中，我们将首先学习如何加载、操作和显示视频。然后，我们将继续将我们的图形保存为图像、图像序列或视频，然后我们将转向录音可视化。

最后，我们将学习如何在应用程序之间共享图形以及如何保存网格数据。

# 加载和显示视频

在这个菜谱中，我们将学习如何使用 Quicktime 和 OpenGL 从文件中加载视频并在屏幕上显示。我们将学习如何将文件作为资源加载，或者通过文件打开对话框由用户选择文件来加载。

## 准备工作

您需要安装 QuickTime，并且还需要一个与 QuickTime 兼容格式的视频文件。

要将视频作为资源加载，需要将其复制到项目中的`resources`文件夹。要了解更多关于资源的信息，请阅读来自第一章的菜谱*在 Windows 上使用资源*和*在 OSX 和 iOS 上使用资源*，*入门*。

## 如何做到这一点…

我们将使用 Cinder 的 QuickTime 包装器来加载和显示视频。

1.  通过在源文件开头添加以下内容来包含包含 Quicktime 和 OpenGL 功能的头文件：

    ```cpp
    #include "cinder/qtime/QuickTime.h"
    #include "cinder/gl/gl.h"
    #include "cinder/gl/Texture.h"
    ```

1.  在您应用程序的类声明中声明一个`ci::qtime::MovieGl`成员。此示例只需要`setup`、`update`和`draw`方法，所以请确保至少声明这些方法：

    ```cpp
    using namespace ci;
    using namespace ci::app;

    class MyApp : public AppBasic {
    public:
      void setup();
      void update();
      void draw();

    qtime::MovieGl mMovie;
    gl::Texture mMovieTexture;
    };
    ```

1.  要将视频作为资源加载，请使用`ci::app::loadResource`方法，将文件名作为`parameter`，并在构造电影对象时传递结果`ci::app::DataSourceRef`。将加载资源放在`trycatch`段中也是一个好习惯，以便捕获任何资源加载错误。请在您的`setup`方法中放置以下代码：

    ```cpp
    try{
    mMovie = qtime::MovieGl( loadResource( "movie.mov" ) );
        } catch( Exception e){
    console() <<e.what()<<std::endl;
        }
    ```

1.  您也可以通过使用文件打开对话框并在构造`mMovie`对象时传递文件路径作为参数来加载视频。您的`setup`方法将具有以下代码：

    ```cpp
    try{
    fs::path path = getOpenFilePath();
    mMovie = qtime::MovieGl( path );
        } catch( Exception e){
    console() <<e.what()<<std::endl;
        }
    ```

1.  要播放视频，请调用电影对象的`play`方法。您可以通过将其放在一个`if`语句中来测试`mMovie`的成功实例化，就像一个普通的指针一样：

    ```cpp
    If( mMovie ){
    mMovie.play();
    }
    ```

1.  在`update`方法中，我们将当前电影帧的纹理复制到我们的`mMovieTexture`中，以便稍后绘制：

    ```cpp
    void MyApp::update(){
    if( mMovie ){
    mMovieTexture = mMovie.getTexture();
    }
    ```

1.  要绘制电影，我们只需使用`gl::draw`方法在屏幕上绘制我们的纹理。我们需要检查纹理是否有效，因为`mMovie`可能需要一段时间才能加载。我们还将创建`ci::Rectf`与纹理大小，并将其居中在屏幕上，以保持绘制的视频居中而不拉伸：

    ```cpp
    gl::clear( Color( 0, 0, 0 ) ); 
    if( mMovieTexture ){
    Rect frect = Rectf( mMovieTexture.getBounds() ).getCenteredFit( getWindowBounds(), true );
    gl::draw( mMovieTexture, rect );
    }
    ```

## 它是如何工作的…

`ci::qtime::MovieGl`类通过封装 QuickTime 框架允许播放和控制电影。电影帧被复制到 OpenGl 纹理中，以便于绘制。要访问电影当前帧的纹理，请使用`ci::qtime::MovieGl::getTexture()`方法，它返回一个`ci::gl::Texture`对象。`ci::qtime::MovieGl`使用的纹理始终绑定到`GL_TEXTURE_RECTANGLE_ARB`目标。

## 还有更多

如果你希望对电影中的像素进行迭代，请考虑使用`ci::qtime::MovieSurface`类。这个类通过封装 QuickTime 框架来播放电影，但将电影帧转换为`ci::Surface`对象。要访问当前帧的表面，请使用`ci::qtime::MovieSurface::getSurface()`方法，它返回一个`ci::Surface`对象。

# 创建一个简单的视频控制器

在这个菜谱中，我们将学习如何使用 Cinder 的内置 GUI 功能创建一个简单的视频控制器。

我们将控制电影播放，包括电影是否循环、播放速度、音量和位置。

## 准备工作

你必须安装 Apple 的 QuickTime，并且有一个与 QuickTime 兼容的电影文件。

要了解如何加载和显示电影，请参考之前的菜谱*加载和显示视频*。

## 如何实现...

我们将创建一个简单的界面，使用 Cinder `params`类来控制视频。

1.  通过在源文件顶部添加以下内容，包含必要的文件以使用 Cinder `params`（QuickTime 和 OpenGl）：

    ```cpp
    #include "cinder/gl/gl.h"
    #include "cinder/gl/Texture.h"
    #include "cinder/qtime/QuickTime.h"
    #include "cinder/params/Params.h"
    #include "cinder/Utilities.h"
    ```

1.  在应用程序类声明之前添加`using`语句，以简化调用 Cinder 命令，如下所示：

    ```cpp
    using namespace ci;
    using namespace ci::app;
    using namespace ci::gl;
    ```

1.  声明一个`ci::qtime::MovieGl`、`ci::gl::Texture`和`ci::params::InterfaceGl`对象，分别用于播放、渲染和控制视频。在你的类声明中添加以下内容：

    ```cpp
    Texture mMovieTexture;
    qtime::MovieGl mMovie;
    params::InterfaceGl mParams;
    ```

1.  通过打开一个打开文件对话框选择视频文件，并使用该路径初始化我们的`mMovie`。以下代码应在`setup`方法中：

    ```cpp
    try{
    fs::path path = getOpenFilePath();
    mMovie = qtime::MovieGl( path );
    }catch( … ){
      console() << "could not open video file" <<std::endl;
    }
    ```

1.  我们还需要一些变量来存储我们将要操作的值。每个可控制的视频参数将有两个变量来表示该参数的当前值和前一个值。现在声明以下变量：

    ```cpp
    float mMoviePosition, mPrevMoviePosition;
    float mMovieRate, mPrevMovieRate;
    float mMovieVolume, mPrevMovieVolume;
    bool mMoviePlay, mPrevMoviePlay;
    bool mMovieLoop, mPrevMovieLoop;
    ```

1.  在`setup`方法中设置默认值：

    ```cpp
    mMoviePosition = 0.0f;
    mPrevMoviePosition = mMoviePosition;
    mMovieRate = 1.0f;
    mPrevMovieRate = mMovieRate;
    mMoviePlay = false;
    mPrevMoviePlay = mMoviePlay;
    mMovieLoop = false;
    mPrevMovieLoop = mMovieLoop;
    mMovieVolume = 1.0f;
    mPrevMovieVolume = mMovieVolume;
    ```

1.  现在让我们初始化`mParams`并为之前定义的每个变量添加一个控件，并在必要时设置`max`、`min`和`step`值。以下代码必须在`setup`方法中：

    ```cpp
    mParams = params::InterfaceGl( "Movie Controller", Vec2i( 200, 300 ) ); 
    if( mMovie ){
    string max = ci::toString( mMovie.getDuration() );
    mParams.addParam( "Position", &mMoviePosition, "min=0.0 max=" + max + " step=0.5" );

    mParams.addParam( "Rate", &mMovieRate, "step=0.01" );

    mParams.addParam( "Play/Pause", &mMoviePlay );

    mParams.addParam( "Loop", &mMovieLoop );

    mParams.addParam( "Volume", &mMovieVolume, "min=0.0 max=1.0 step=0.01" );
    }
    ```

1.  在`update`方法中，我们将检查电影是否有效，并将每个参数与其前一个状态进行比较，以查看它们是否已更改。如果已更改，我们将更新`mMovie`并将参数设置为新的值。以下代码行应放在`update`方法中：

    ```cpp
    if( mMovie ){

    if( mMoviePosition != mPrevMoviePosition ){
    mPrevMoviePosition = mMoviePosition;
    mMovie.seekToTime( mMoviePosition );
            } else {
    mMoviePosition = mMovie.getCurrentTime();
    mPrevMoviePosition = mMoviePosition;
            }
    if( mMovieRate != mPrevMovieRate ){
    mPrevMovieRate = mMovieRate;
    mMovie.setRate( mMovieRate );
            }
    if( mMoviePlay != mPrevMoviePlay ){
    mPrevMoviePlay = mMoviePlay;
    if( mMoviePlay ){
    mMovie.play();
                } else {
    mMovie.stop();
                }
            }
    if( mMovieLoop != mPrevMovieLoop ){
    mPrevMovieLoop = mMovieLoop;
    mMovie.setLoop( mMovieLoop );
            }
    if( mMovieVolume != mPrevMovieVolume ){
    mPrevMovieVolume = mMovieVolume;
    mMovie.setVolume( mMovieVolume );
            }
        }
    ```

1.  在`update`方法中，还需要获取电影纹理并将其复制到之前声明的`mMovieTexture`。在`update`方法中，我们编写：

    ```cpp
    if( mMovie ){
    mMovieTexture = mMovie.getTexture();
    }
    ```

1.  剩下的就是绘制我们的内容了。在 `draw` 方法中，我们将使用黑色清除背景。我们将检查 `mMovieTexture` 的有效性，并在一个适合窗口的矩形内绘制它。我们还调用 `mParams` 的 `draw` 命令来在视频上方绘制控件：

    ```cpp
    gl::clear( Color( 0, 0, 0 ) ); 

    if( mMovieTexture ){
    Rectf rect = Rectf( mMovieTexture.getBounds() ).getCenteredFit( getWindowBounds(), true );
    gl::draw( mMovieTexture, rect );
        }

    mParams.draw();
    ```

1.  绘制它，您将看到具有黑色背景和控件的应用程序窗口。在参数菜单中更改各种参数，您将看到它影响视频：![如何做到这一点…](img/8703OS_4_1.jpg)

## 它是如何工作的…

我们创建了一个 `ci::params::InterfaceGl` 对象，并为每个我们想要操作的参数添加了一个控件。

我们为每个我们想要操作的参数创建了一个变量，并为存储它们的上一个值创建了一个变量。在更新时，我们检查这些值是否不同，这只会发生在用户使用 `mParams` 菜单更改它们的值时。

当参数更改时，我们使用用户设置的值更改 `mMovie` 参数。

一些参数必须保持在特定的范围内。电影位置设置为从 `0` 秒到视频最大持续时间的秒数。音量必须是一个介于 `0` 和 `1` 之间的值，`0` 表示没有音频，而 `1` 是最大音量。

# 将窗口内容保存为图像

在这个例子中，我们将向您展示如何将窗口内容保存到图形文件，以及如何在您的 Cinder 应用程序中实现此功能。这可以用于保存图形算法的输出。

## 如何做到这一点...

我们将在您的应用程序中添加一个窗口内容保存功能：

1.  添加必要的头文件：

    ```cpp
    #include "cinder/ImageIo.h"
    #include "cinder/Utilities.h"
    ```

1.  向您的应用程序主类添加属性：

    ```cpp
    bool mMakeScreenshot;
    ```

1.  在 `setup` 方法中设置默认值：

    ```cpp
    mMakeScreenshot = false;
    ```

1.  按如下方式实现 `keyDown` 方法：

    ```cpp
    void MainApp::keyDown(KeyEvent event)
      {
      if(event.getChar() == 's') {
      mMakeScreenshot = true;
        }
      }
    ```

1.  在 `draw` 方法的末尾添加以下代码：

    ```cpp
    if(mMakeScreenshot) {
    mMakeScreenshot = false;
    writeImage( getDocumentsDirectory() / fs::path("MainApp_screenshot.png"), copyWindowSurface() );
    }
    ```

## 它是如何工作的…

每次将 `mMakeScreenshot` 设置为 `true` 时，您的应用程序的截图将被选中并保存。在这种情况下，应用程序等待按下 *S* 键，然后将标志 `mMakeScreenshot` 设置为 `true`。当前应用程序窗口的截图将被保存在您的文档目录下，文件名为 `MainApp_screenshot.png`。

## 还有更多...

这只是 `writeImage` 函数常见用法的简单示例。还有许多其他实际应用。

### 将窗口动画保存为图像序列

假设您想要记录一系列图像。执行以下步骤来完成此操作：

1.  修改步骤 5 中显示的先前代码片段，以将窗口内容保存如下：

    ```cpp
    if(mMakeScreenshot || mRecordFrames) {
    mMakeScreenshot = false;
    writeImage( getDocumentsDirectory() / fs::path("MainApp_screenshot_" + toString(mFramesCounter) + ".png"), copyWindowSurface() );
    mFramesCounter++;
    }
    ```

1.  您必须将 `mRecordFrames` 和 `mFrameCounter` 定义为您的应用程序主类的属性：

    ```cpp
    bool mRecordFrames;
    int mFramesCounter;
    ```

1.  在 `setup` 方法中设置初始值：

    ```cpp
    mRecordFrames = false;
    mFramesCounter = 1;
    ```

### 录音声音可视化

我们假设您正在使用 `audio` 命名空间中的 `TrackRef` 来播放您的声音。执行以下步骤：

1.  实现之前的步骤以将窗口动画保存为图像序列。

1.  在 `update` 方法的开头输入以下代码行：

    ```cpp
    if(mRecordFrames) {
    mTrack->setTime(mFramesCounter / 30.f);
    }
    ```

我们正在根据经过的帧数计算所需的音频轨道位置。我们这样做是为了使动画与音乐轨道同步。在这种情况下，我们希望产生 `30` fps 的动画，所以我们把 `mFramesCounter` 除以 `30`。

# 将窗口动画保存为视频

在这个菜谱中，我们将从绘制一个简单的动画开始，并学习如何将其导出为视频。我们将创建一个视频，按下任意键将开始或停止录制。

## 准备工作

你必须安装苹果的 QuickTime。确保你知道你想要将视频保存的位置，因为你将不得不在开始时指定其位置。

这可以是使用 OpenGl 绘制的任何东西，但在这个例子中，我们将在窗口中心创建一个黄色的圆圈，其半径会变化。半径是通过自应用程序启动以来经过的秒数的正弦值的绝对值来计算的。我们将此值乘以 `200` 以放大它。现在将以下内容添加到 `draw` 方法中：

```cpp
gl::clear( Color( 0, 0, 0 ) );     
float radius = fabsf( sinf( getElapsedSeconds() ) ) * 200.0f;
Vec2f center = getWindowCenter();
gl::color( Color( 1.0f, 1.0f, 0.0f ) );
gl::drawSolidCircle( center, radius );
```

## 如何操作…

我们将使用 `ci::qtime::MovieWriter` 类来创建我们的渲染视频。

1.  在源文件的开头包含 OpenGl 和 QuickTime 文件，通过添加以下内容：

    ```cpp
    #include "cinder/gl/gl.h"
    #include "cinder/qtime/MovieWriter.h"
    ```

1.  现在让我们声明一个 `ci::qtime::MovieWriter` 对象和一个初始化它的方法。将以下内容添加到你的类声明中：

    ```cpp
    qtime::MovieWriter mMovieWriter;
    void initMovieWriter();
    ```

1.  在 `initMovieWriter` 的实现中，我们首先要求用户使用保存文件对话框指定一个路径，并使用它来初始化电影写入器。电影写入器还需要知道窗口的宽度和高度。这是 `initMovieWriter` 的实现。

    ```cpp
    void MyApp::initMovieWriter(){
    fs::path path = getSaveFilePath();
    if( path.empty() == false ){
    mMovieWriter = qtime::MovieWriter( path, getWindowWidth(), getWindowHeight() );
        }
    }
    ```

1.  让我们通过声明 `keyUp` 方法来声明一个按键事件处理器。

    ```cpp
    void keyUp( KeyEvent event );
    ```

1.  在实现中，我们将通过检查 `mMovieWriter` 的有效性来查看是否已经在录制电影。如果它是一个有效的对象，那么我们必须通过销毁对象来保存当前的电影。我们可以通过调用 `ci::qtime::MovieWriter` 默认构造函数来实现；这将创建一个空实例。如果 `mMovieWriter` 不是一个有效的对象，那么我们通过调用 `initMovieWriter()` 方法来初始化一个新的电影写入器。

    ```cpp
    void MovieWriterApp::keyUp( KeyEvent event ){
    if( mMovieWriter ){
    mMovieWriter = qtime::MovieWriter();
        } else {
    initMovieWriter();
        }
    }
    ```

1.  最后两个步骤是检查 `mMovieWriter` 是否有效，并通过调用带有窗口表面的 `addFrame` 方法来添加一个帧。这个方法必须在 `draw` 方法中调用，在我们的绘图程序之后。这是最终的 `draw` 方法，包括圆圈绘制代码。

    ```cpp
    void MyApp::draw()
    {
      gl::clear( Color( 0, 0, 0 ) ); 

    float radius = fabsf( sinf( getElapsedSeconds() ) ) * 200.0f;
        Vec2f center = getWindowCenter();
    gl::color( Color( 1.0f, 1.0f, 0.0f ) );
    gl::drawSolidCircle( center, radius );

    if( mMovieWriter ){
    mMovieWriter.addFrame( copyWindowSurface() );
        }
    }
    ```

1.  构建并运行应用程序。按下任意键将开始或结束视频录制。每次开始新的录制时，用户将看到一个保存文件对话框，用于设置电影将保存的位置。![如何操作…](img/8703OS_4_2.jpg)

## 它是如何工作的…

`ci::qtime::MovieWriter` 对象允许使用苹果的 QuickTime 容易地写入电影。录制开始于初始化一个 `ci::qtime::MovieWriter` 对象，并在对象被销毁时保存。通过调用 `addFrame` 方法，可以添加新的帧。

## 还有更多...

你还可以通过创建一个`ci::qtime::MovieWriter::Format`对象并将其作为可选参数传递给电影编写器的构造函数来定义视频的格式。如果没有指定格式，电影编写器将使用默认的 PNG 编解码器和每秒 30 帧。

例如，要创建一个使用 H264 编解码器、50%质量和 24 帧每秒的电影编写器，你可以编写以下代码：

```cpp
qtime::MovieWriter::Format format;
format.setCodec( qtime::MovieWriter::CODEC_H264 );
format.setQuality( 0.5f );
format.setDefaultDuration( 1.0f / 24.0f );
qtime::MovieWriter mMovieWriter = ci::Qtime::MovieWriter( "mymovie.mov", getWindowWidth(), getWindowHeight(), format );
```

你可以选择打开一个**设置**窗口，并允许用户通过调用静态方法`qtime::MovieWriter::getUserCompressionSettings`来定义视频设置。此方法将填充一个`qtime::MovieWriter::Format`对象，并在成功时返回`true`，如果用户取消了设置更改则返回`false`。

要使用此方法定义设置并创建一个电影编写器，你可以编写以下代码：

```cpp
qtime::MovieWriter::Format format;
qtime::MovieWriter mMovieWriter;
boolformatDefined = qtime::MovieWriter::getUserCompressionSettings( &format );
if( formatDefined ){
mMovieWriter = qtime::MovieWriter( "mymovie.mov", getWindowWidth(), getWindowHeight(), format );
}
```

还可以启用**多遍**编码。对于 Cinder 的当前版本，它仅通过 H264 编解码器可用。多遍编码将提高电影的质量，但会以性能下降为代价。因此，默认情况下它是禁用的。

要启用多遍编码写入电影，我们可以编写以下代码：

```cpp
qtime::MovieWriter::Format format;
format.setCodec( qtime::MovieWriter::CODEC_H264 );
format.enableMultiPass( true );
qtime::MovieWritermMovieWriter = ci::Qtime::MovieWriter( "mymovie.mov", getWindowWidth(), getWindowHeight(), format );
```

可以使用`ci::qtime::MovieWriter::Format`类设置许多设置和格式，要了解完整的选项列表，请查看该类在[`libcinder.org/docs/v0.8.4/guide__qtime___movie_writer.html`](http://libcinder.org/docs/v0.8.4/guide__qtime___movie_writer.html)的文档。

# 将窗口内容保存为矢量图形图像

在这个菜谱中，我们将学习如何使用 cairo 渲染器在屏幕上绘制 2D 图形并将其保存为矢量图形格式的图像。

矢量图形在创建用于打印的视觉效果时非常有用，因为它们可以缩放而不失真。

Cinder 集成了 cairo 图形库；一个功能强大且功能齐全的 2D 渲染器，能够输出到包括流行的矢量图形格式在内的多种格式。

要了解更多关于 cairo 库的信息，请访问其官方网站：[`www.cairographics.org`](http://www.cairographics.org)

在这个例子中，我们将创建一个应用程序，当用户按下鼠标时，它会绘制一个新的圆。当按下任何键时，应用程序将打开一个保存文件对话框，并将内容以文件扩展名定义的格式保存。

## 准备工作

要绘制使用 cairo 渲染器创建的图形，我们必须将我们的渲染器定义为`Renderer2d`。

在我们的应用程序类的源文件末尾有一个用于初始化应用程序的**宏**，其中第二个参数定义了渲染器。如果你的应用程序名为`MyApp`，你必须将宏更改为以下内容：

```cpp
CINDER_APP_BASIC( MyApp, Renderer2d )
```

cairo 渲染器允许导出 PDF、SVG、EPS 和 PostScript 格式。在指定要保存的文件时，确保你写了一个受支持的扩展名：`pdf`、`svg`、`eps`或`ps`。

在源文件顶部包含以下文件：

```cpp
#include "cinder/Rand.h"
#include "cinder/cairo/Cairo.h"
```

## 如何实现...

我们将使用 Cinder 的 cairo 包装器从我们的渲染中创建矢量格式的图像。

1.  每当用户按下鼠标时创建一个新圆，我们必须首先创建一个 `Circle` 类。这个类将包含位置、半径和颜色参数。它的构造函数将接受 `ci::Vec2f` 来定义其位置，并将生成一个随机半径和颜色。

    在应用程序的类声明之前写入以下代码：

    ```cpp
    class Circle{
    public:
        Circle( const Vec2f&pos ){
    this->pos = pos;
    radius = randFloat( 20.0f, 50.0f );
    color = ColorA( randFloat( 1.0f ), randFloat( 1.0f ), randFloat( 1.0f ), 0.5f );
        }

        Vec2f pos;
    float radius;
    ColorA color;
    };
    ```

1.  我们现在应该声明一个存储创建的圆的 `std::vector` 的圆，并将以下代码添加到类声明中：

    ```cpp
    std::vector< Circle >mCircles;
    ```

1.  让我们创建一个将 `cairo::Context` 作为参数的方法来绘制圆：

    ```cpp
    void renderScene( cairo::Context &context );
    ```

1.  在方法定义中，遍历 `mCircles` 并在上下文中绘制每一个：

    ```cpp
    void MyApp::renderScene( cairo::Context &context ){
    for( std::vector< Circle >::iterator it = mCircles.begin(); it != mCircles.end(); ++it ){
    context.circle( it->pos, it->radius );
    context.setSource( it->color );
    context.fill();
        }
    }
    ```

1.  在这一点上，我们只需要在用户按下鼠标时添加一个圆。为此，我们必须通过在类声明中声明它来实现 `mouseDown` 事件处理程序。

    ```cpp
    void mouseDown( MouseEvent event );
    ```

1.  在其实现中，我们使用鼠标位置将一个 `Circle` 类添加到 `mCircles` 中。

    ```cpp
    void MyApp::mouseDown( MouseEvent event ){
      Circle circle( event.getPos() );
    mCircles.push_back( circle );
    }
    ```

1.  我们现在可以通过创建绑定到窗口表面的 `cairo::Context` 来在窗口上绘制圆。这将让我们可视化我们正在绘制的。以下是 `draw` 方法的实现：

    ```cpp
    void CairoSaveApp::draw()
    {
    cairo::Context context( cairo::createWindowSurface() );
    renderScene( context );
    }
    ```

1.  要将场景保存到图像文件，我们必须创建一个绑定到表示矢量图形格式中文件的表面的上下文。让我们通过声明 `keyUp` 事件处理程序来实现这一点，每当用户释放一个键时执行此操作。

    ```cpp
    void keyUp( KeyEvent event );
    ```

1.  在 `keyUp` 实现中，我们创建 `ci::fs::path` 并通过调用保存文件对话框来填充它。我们还将创建一个空的 `ci::cairo::SurfaceBase`，它是 cairo 渲染器可以绘制到的所有表面的基础。

    ```cpp
    fs::path filePath = getSaveFilePath();
    cairo::SurfaceBase surface;
    ```

1.  我们现在将比较路径的扩展名与支持的格式，并相应地初始化表面。它可以初始化为 `ci::cairo::SurfacePdf`、`ci::cairo::SurfaceSvg`、`ci::cairo::SurfaceEps` 或 `ci::cairo::SurfacePs`。

    ```cpp
    if( filePath.extension() == ".pdf" ){
    surface = cairo::SurfacePdf( filePath, getWindowWidth(), getWindowHeight() );
        } else if( filePath.extension() == ".svg" ){
    surface = cairo::SurfaceSvg( filePath, getWindowWidth(), getWindowHeight() );
        } else if( filePath.extension() == ".eps" ){
    surface = cairo::SurfaceEps( filePath, getWindowWidth(), getWindowHeight() );
        } else if( filePath.extension() == ".ps" ){
    surface = cairo::SurfacePs( filePath, getWindowWidth(), getWindowHeight() );
        }
    ```

1.  现在我们可以创建 `ci::cairo::Context` 并通过调用 `renderScene` 方法并将上下文作为参数传递来将其渲染到场景中。圆将被渲染到上下文中，并将在指定的格式中创建一个文件。以下是最终的 `keyUp` 方法实现：

    ```cpp
    void CairoSaveApp::keyUp( KeyEvent event ){
    fs::path filePath = getSaveFilePath();
    cairo::SurfaceBase surface;
    if( filePath.extension() == ".pdf" ){
    surface = cairo::SurfacePdf( filePath, getWindowWidth(), getWindowHeight() );
        } else if( filePath.extension() == ".svg" ){
    surface = cairo::SurfaceSvg( filePath, getWindowWidth(), getWindowHeight() );
        } else if( filePath.extension() == ".eps" ){
    surface = cairo::SurfaceEps( filePath, getWindowWidth(), getWindowHeight() );
        } else if( filePath.extension() == ".ps" ){
    surface = cairo::SurfacePs( filePath, getWindowWidth(), getWindowHeight() );
        }
    cairo::Context context( surface );
    renderScene( context );
    }
    ```

    ![如何实现…](img/8703OS_4_3.jpg)

## 它是如何工作的...

Cinder 包装并集成了 cairo 2D 矢量渲染器。它允许使用 Cinder 的类型来绘制和与 cairo 交互。

完整的绘图是通过调用 `ci::cairo::Context` 对象的绘图方法来完成的。上下文反过来必须通过传递一个扩展 `ci::cairo::SurfaceBase` 的表面对象来创建。所有绘图都将在这个表面上完成，并根据表面的类型进行光栅化。

以下表面允许以矢量图形格式保存图像：

| 表面类型 | 格式 |
| --- | --- |
| `ci::cairo::SurfacePdf` | PDF |
| `ci::cairo::SurfaceSvg` | SVG |
| `ci::cairo::SurfaceEps` | EPS |
| `ci::cairo::SurfacePs` | PostScript |

## 还有更多...

也可以使用其他渲染器进行绘制。尽管渲染器无法创建矢量图像，但在其他情况下它们可能很有用。

这里是其他可用的表面：

| 表面类型 | 格式 |
| --- | --- |
| `ci::cairo::SurfaceImage` | 基于像素的抗锯齿光栅化器 |
| `ci::cairo::SurfaceQuartz` | 苹果的 Quartz |
| `ci::cairo::SurfaceCgBitmapContext` | 苹果的 CoreGraphics |
| `ci::cairo::SurfaceGdi` | Windows GDI |

# 使用瓦片渲染器保存高分辨率图像

在这个菜谱中，我们将学习如何使用`ci::gl::TileRender`类导出屏幕上绘制的内容的高分辨率图像。这在创建用于打印的图形时非常有用。

我们将首先创建一个简单的场景并在屏幕上绘制它。接下来，我们将编写示例代码，以便每当用户按下任何键时，都会出现一个保存文件对话框，并将高分辨率图像保存到指定的路径。

## 准备中

`TileRender`类可以从使用 OpenGl 调用的任何屏幕绘制内容创建高分辨率图像。

要使用`TileRender`保存图像，我们首先必须在屏幕上绘制一些内容。这可以是任何内容，但为了这个示例，让我们创建一个简单的图案，用圆形填充整个屏幕。

在`draw`方法的实现中，写入以下代码：

```cpp
void MyApp::draw()
{
  gl::clear( Color( 0, 0, 0 ) ); 
gl::color( Color::white() );
for( float i=0; i<getWindowWidth(); i+=10.0f ){
for( float j=0; j<getWindowHeight(); j += 10.0f ){
float radius = j * 0.01f;
gl::drawSolidCircle( Vec2f( i, j ), radius );
        }
    }
}
```

记住，这可以是使用 OpenGl 在屏幕上绘制的任何内容。

![准备中](img/8703OS_4_4.jpg)

## 如何实现...

我们将使用`ci::gl::TileRender`类来生成 OpenGL 渲染的高分辨率图像。

1.  通过在源文件顶部添加以下内容来包含必要的头文件：

    ```cpp
    #include "cinder/gl/TileRender.h"
    #include "cinder/ImageIo.h"
    ```

1.  由于我们将在用户按下任何键时保存高分辨率图像，让我们通过在类声明中声明来实现`keyUp`事件处理器。

    ```cpp
    void keyUp( KeyEvent event );
    ```

1.  在`keyUp`实现中，我们首先创建一个`ci::gl::TileRender`对象，然后设置我们即将创建的图像的宽度和高度。我们将将其设置为应用程序窗口大小的四倍。它可以是你想要的任何大小，但请注意，如果你不尊重窗口的宽高比，图像将会被拉伸。

    ```cpp
    gl::TileRender tileRender( getWindowWidth() * 4, getWindowHeight() * 4 );
    ```

1.  我们必须定义场景的`Modelview`和`Projection`矩阵以匹配我们的窗口。如果我们只使用 2D 图形，我们可以调用`setMatricesWindow`方法，如下所示：

    ```cpp
    tileRender.setMatricesWindow( getWindowWidth(), getWindowHeight() );
    ```

    为了在绘制 3D 内容时定义场景的`Modelview`和`Projection`矩阵以匹配窗口，必须调用`setMatricesWindowPersp`方法：

    ```cpp
    tileRender.setMatricesWindowPersp( getWindowWidth(), getWindowHeight() );
    ```

1.  接下来，我们将使用`nextTile`方法在创建新瓦片时绘制场景。当所有瓦片都创建完毕后，该方法将返回`false`。我们可以通过在`while`循环中重新绘制场景并询问是否有下一个瓦片来创建所有瓦片，如下所示：

    ```cpp
    while( tileRender.nextTile() ){
    draw();
        }
    ```

1.  现在，场景已经完全在 `TileRender` 中渲染，我们必须保存它。让我们通过打开一个保存文件对话框来让用户指定保存位置。必须指定图像文件的扩展名，因为它将用于内部定义图像格式。

    ```cpp
    fs::path filePath = getSaveFilePath();
    ```

1.  我们检查 `filePath` 是否为空，并使用 `writeImage` 方法将标题渲染表面作为图像写入。

    ```cpp
    if( filePath.empty() == false ){
    writeImage( filePath, tileRender.getSurface() );
    }
    ```

1.  保存图像后，需要重新定义窗口的 `Modelview` 和 `Projection` 矩阵。如果在 2D 中绘图，可以通过使用带有窗口尺寸的 `setMatricesWindow` 方法将矩阵设置为默认值，如下所示：

    ```cpp
    gl::setMatricesWindow( getWindowWidth(), getWindowHeight() );
    ```

## 它的工作原理…

`ci::gl::TileRender` 类使得通过将我们的绘图的部分缩放到整个窗口大小并将它们存储为 `ci::Surface`，从而生成我们渲染的高分辨率版本成为可能。在将整个场景存储在各个部分之后，它们被拼接成瓷砖，形成一个单独的高分辨率 `ci::Surface`，然后可以将其保存为图像。

# 在应用程序之间共享图形

在这个菜谱中，我们将向您展示在 Mac OS X 下实时在应用程序之间共享图形的方法。为此，我们将使用 **Syphon** 和其针对 Cinder 的实现。Syphon 是一个开源工具，允许应用程序以静态帧或实时更新的帧序列共享图形。您可以在以下位置了解更多关于 Syphon 的信息：[`syphon.v002.info/`](http://syphon.v002.info/)

## 准备工作

为了测试我们应用程序共享的图形是否可用，我们将使用 **Syphon Recorder**，您可以在以下位置找到它：[`syphon.v002.info/recorder/`](http://syphon.v002.info/recorder/)

## 如何操作…

1.  从 *syphon-implementations* 仓库检出 Syphon CinderBlock [`code.google.com/p/syphon-implementations/`](http://code.google.com/p/syphon-implementations/).

1.  在您的项目树中创建一个新的组，并将其命名为 `Blocks`。

1.  将 Syphon CinderBlock 拖放到你新创建的 `Blocks` 组中。![如何操作…](img/8703OS_4_5.jpg)

1.  确保在 **target** 设置的 **Build Phases** 的 **Copy Files** 部分添加了 **Syphon.framework**。

1.  添加必要的头文件：

    ```cpp
    #include "cinderSyphon.h"
    ```

1.  向您的应用程序主类添加属性：

    ```cpp
    syphonServer mScreenSyphon;
    ```

1.  在 `setup` 方法的末尾，添加以下代码：

    ```cpp
    mScreenSyphon.setName("Cinder Screen");
    gl::clear(Color::white());
    ```

1.  在 `draw` 方法内部添加以下代码：

    ```cpp
    gl::enableAlphaBlending();

    gl::color( ColorA(1.f, 1.f, 1.f, 0.05f) );
    gl::drawSolidRect( getWindowBounds() );

    gl::color( ColorA::black() );
    Vec2f pos = Vec2f( cos(getElapsedSeconds()), sin(getElapsedSeconds())) * 100.f;
    gl::drawSolidCircle(getWindowCenter() + pos, 10.f);

    mScreenSyphon.publishScreen();
    ```

## 它是如何工作的…

应用程序绘制一个简单的旋转动画，并通过 Syphon 库共享整个窗口区域。我们的应用程序窗口如下截图所示：

![它的工作原理…](img/8703OS_4_6.jpg)

要测试图形是否可以被其他应用程序接收，我们将使用 Syphon Recorder。运行 Syphon Recorder 并在“**Cinder Screen – MainApp**”名称下的下拉菜单中找到我们的 Cinder 应用程序。我们在“*如何做...*”部分的步骤 6 中设置了该名称的第一部分，而第二部分是可执行文件名。现在，我们的 Cinder 应用程序的预览应该可用，并且看起来如下截图所示：

![How it works…](img/8703OS_4_7.jpg)

## 还有更多...

Syphon 库非常实用，易于使用，并且适用于其他应用程序和库。

### 接收来自其他应用程序的图形

您还可以接收来自其他应用程序的纹理。为此，您必须使用 `syphonClient` 类，如下步骤所示：

1.  在您的应用程序主类中添加一个属性：

    ```cpp
    syphonClient mClientSyphon;
    ```

1.  在 CIT 方法中初始化 `mClientSyphon`：

    ```cpp
    mClientSyphon.setApplicationName("MainApp Server");
    mClientSyphon.setServerName("");
    mClientSyphon.bind();
    ```

1.  在 `draw` 方法的末尾添加以下行，该行绘制其他应用程序共享的图形：

    ```cpp
    mClientSyphon.draw(Vec2f::zero());
    ```
