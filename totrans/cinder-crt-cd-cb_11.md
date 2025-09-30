# 第十一章. 感知和跟踪来自摄像头的输入

在本章中，我们将学习如何接收和处理来自摄像头或微软 Kinect 传感器等输入设备的数据。

以下菜谱将涵盖：

+   从摄像头捕获

+   基于颜色跟踪对象

+   使用光流跟踪运动

+   对象跟踪

+   读取 QR 码

+   使用 Kinect 构建 UI 导航和手势识别

+   使用 Kinect 构建增强现实

# 从摄像头捕获

在这个菜谱中，我们将学习如何捕获和显示来自摄像头的帧。

## 准备工作

包含必要的文件以从摄像头捕获图像并将它们绘制到 OpenGL 纹理中：

```cpp
#include "cinder/gl/gl.h"
#include "cinder/gl/Texture.h"
#include "cinder/Capture.h"
```

还需要添加以下 `using` 语句：

```cpp
using namespace ci;
using namespace ci::app;
using namespace std;
```

## 如何做…

现在，我们将从摄像头捕获并绘制帧。

1.  在你的应用程序类中声明以下成员：

    ```cpp
        Capture mCamera;
        gl::Texture mTexture;
    ```

1.  在 `setup` 方法中，我们将初始化 `mCamera`：

    ```cpp
        try{
            mCamera = Capture( 640, 480 );
            mCamera.start();
        } catch( ... ){
            console() << "Could not initialize the capture" << endl;
    ```

1.  在 `update` 方法中，我们将检查 `mCamera` 是否已成功初始化。如果还有新的帧可用，将摄像头的图像复制到 `mTexture`：

    ```cpp
        if( mCamera ){
            if( mCamera.checkNewFrame() ){
                mTexture = gl::Texture( mCamera.getSurface() );
            }
        }
    ```

1.  在 `draw` 方法中，我们将简单地清除背景，检查 `mTexture` 是否已初始化，并将其图像绘制到屏幕上：

    ```cpp
      gl::clear( Color( 0, 0, 0 ) ); 
        if( mTexture ){
            gl::draw( mTexture, getWindowBounds() );
        }
    ```

## 它是如何工作的…

`ci::Capture` 是一个类，它在苹果电脑上围绕 Quicktime，在 iOS 平台上围绕 AVFoundation，在 Windows 上围绕 Directshow 进行封装。在底层，它使用这些低级框架来访问和捕获来自网络摄像头的帧。

每当找到新的帧时，它的像素将被复制到 `ci::Surface` 方法中。在前面的代码中，我们在每个 `update` 方法中通过调用 `ci::Capture::checkNewFrame` 方法来检查是否有新的帧，并使用其表面更新我们的纹理。

## 更多…

还可以获取可用捕获设备的列表，并选择你希望开始的设备。

要请求设备列表并打印它们的信息，我们可以编写以下代码：

```cpp
vector<Capture::DeviceRef> devices = Capture::getDevices();
for( vector<Capture::DeviceRef>::iterator it = devices.begin(); 
 it != devices.end(); ++it ){
 Capture::DeviceRef device = *it;
 console() << "Found device:" 
  << device->getName() 
  << " with ID:" << device->getUniqueId() << endl;
}
```

要使用特定设备初始化 `mCapture`，你只需在构造函数中将 `ci::Capture::DeviceRef` 作为第三个参数传递。

例如，如果你想用第一个设备初始化 `mCapture`，你应该编写以下代码：

```cpp
vector<Capture::DeviceRef> devices = Capture::getDevices();
mCapture = Capture( 640, 480 devices[0] );
```

# 基于颜色跟踪对象

在这个菜谱中，我们将展示如何使用 OpenCV 库跟踪指定颜色的对象。

## 准备工作

在这个菜谱中，我们将使用 OpenCV，因此请参阅第三章 *与 OpenCV 集成* 的 *菜谱*，*使用图像处理技术*。我们还需要 InterfaceGl，它包含在第二章 *准备开发* 的 *设置 GUI 以调整参数* 菜谱中。

## 如何做…

我们将创建一个应用程序，该应用程序使用所选颜色跟踪对象。

1.  包含必要的头文件：

    ```cpp
    #include "cinder/gl/gl.h"
    #include "cinder/gl/Texture.h"
    #include "cinder/Surface.h"
    #include "cinder/ImageIo.h"
    #include "cinder/Capture.h"
    #include "cinder/params/Params.h"
    #include  "CinderOpenCV.h"
    ```

1.  添加成员以存储原始和处理的帧：

    ```cpp
    Surface8u   mImage;
    ```

1.  添加成员以存储跟踪对象的坐标：

    ```cpp
    vector<cv::Point2f> mCenters;
    vector<float> mRadius;
    ```

1.  添加成员以存储将传递给跟踪算法的参数：

    ```cpp
    double mApproxEps;
    int mCannyThresh;

    ColorA      mPickedColor;
    cv::Scalar  mColorMin;
    cv::Scalar  mColorMax;
    ```

1.  添加成员以处理捕获设备和帧纹理：

    ```cpp
    Capture     mCapture;
    gl::Texture mCaptureTex;
    ```

1.  在 `setup` 方法中，我们将设置窗口尺寸并初始化捕获设备：

    ```cpp
    setWindowSize(640, 480);

    try {
      mCapture = Capture( 640, 480 );
      mCapture.start();
    }
    catch( ... ) {
      console() <<"Failed to initialize capture"<<std::endl;
    }
    ```

1.  在 `setup` 方法中，我们必须初始化变量并设置 GUI 以预览跟踪的颜色值：

    ```cpp
    mApproxEps = 1.0;
    mCannyThresh = 200;

    mPickedColor = Color8u(255, 0, 0);
    setTrackingHSV();

    // Setup the parameters
    mParams = params::InterfaceGl( "Parameters", Vec2i( 200, 150 ) );
    mParams.addParam( "Picked Color", &mPickedColor, "readonly=1" );
    ```

1.  在 `update` 方法中，检查是否有任何新帧需要处理并将其转换为 `cv::Mat`，这对于进一步的 OpenCV 操作是必要的：

    ```cpp
    if( mCapture&&mCapture.checkNewFrame() ) {
      mImage = mCapture.getSurface();
      mCaptureTex = gl::Texture( mImage );

      cv::Mat inputMat( toOcv( mImage) );
      cv::resize(inputMat, inputMat, cv::Size(320, 240) );

      cv::Mat inputHSVMat, frameTresh;
      cv::cvtColor(inputMat, inputHSVMat, CV_BGR2HSV);
    ```

1.  处理捕获的帧：

    ```cpp
      cv::inRange(inputHSVMat, mColorMin, mColorMax, frameTresh);

      cv::medianBlur(frameTresh, frameTresh, 7);

      cv::Mat cannyMat;
      cv::Canny(frameTresh, cannyMat, mCannyThresh, mCannyThresh*2.f, 3 );
     vector< std::vector<cv::Point> >  contours;
     cv::findContours(cannyMat, contours, CV_RETR_LIST, 
      CV_CHAIN_APPROX_SIMPLE);
     mCenters = vector<cv::Point2f>(contours.size());
     mRadius = vector<float>(contours.size());
     for( int i = 0; i < contours.size(); i++ ) {
      std::vector<cv::Point> approxCurve;
      cv::approxPolyDP(contours[i], approxCurve, 
       mApproxEps, true);
      cv::minEnclosingCircle(approxCurve, mCenters[i], 
       mRadius[i]);
     }
    ```

1.  关闭 `if` 语句的主体。

    ```cpp
    }
    ```

1.  实现方法 `setTrackingHSV`，它设置跟踪颜色的值：

    ```cpp
    void MainApp::setTrackingHSV()
    {
    void MainApp::setTrackingHSV() {
     Color8u col = Color( mPickedColor );
     Vec3f colorHSV = col.get(CM_HSV);
     colorHSV.x *= 179;
     colorHSV.y *= 255;
     colorHSV.z *= 255;
     mColorMin = cv::Scalar(colorHSV.x-5, colorHSV.y -50, 
      colorHSV.z-50);
     mColorMax = cv::Scalar(colorHSV.x+5, 255, 255);
    }
    ```

1.  实现鼠标按下事件处理器：

    ```cpp
    void MainApp::mouseDown(MouseEvent event) {
     if( mImage&&mImage.getBounds().contains( event.getPos() ) ) {
      mPickedColor = mImage.getPixel( event.getPos() );
      setTrackingHSV();
     } 
    }
    ```

1.  按如下方式实现 `draw` 方法：

    ```cpp
    void MainApp::draw()
    {
     gl::clear( Color( 0.1f, 0.1f, 0.1f ) );
     gl::color(Color::white());
     if(mCaptureTex) {
      gl::draw(mCaptureTex);
      gl::color(Color::white());
      for( int i = 0; i <mCenters.size(); i++ )
      {
       Vec2f center = fromOcv(mCenters[i])*2.f;
       gl::begin(GL_POINTS);
       gl::vertex( center );
       gl::end();
       gl::drawStrokedCircle(center, mRadius[i]*2.f );
      }
     }
     params::InterfaceGl::draw();
    }
    ```

## 它是如何工作的…

通过准备捕获帧以进行处理，我们将其转换为 **色调、饱和度和值** （**HSV**） 颜色空间描述方法，这在这种情况下非常有用。这些是描述 HSV 颜色空间中颜色的属性，以更直观的方式用于颜色跟踪。我们可以为检测设置一个固定的色调值，而饱和度和值可以在指定的范围内变化。这可以消除由相机视图中不断变化的光线引起的噪声。看看帧图像处理的第一个步骤；我们使用 `cv::inRange` 函数来获取适合我们跟踪颜色范围的像素掩码。跟踪颜色的范围是从窗口内部点击选择的颜色值计算得出的，这实现在 `mouseDown` 处理器和 `setTrackingHSV` 方法中。

正如您在 `setTrackingHSV` 内部所看到的，我们通过简单地扩大范围来计算 `mColorMin` 和 `mColorMax`。您可能需要根据您的相机噪声和光照条件调整这些计算。

## 参见

+   HSV 在维基百科上的介绍：[`en.wikipedia.org/wiki/HSL_and_HSV`](http://en.wikipedia.org/wiki/HSL_and_HSV)

+   OpenCV 文档：[`opencv.willowgarage.com/documentation/cpp/`](http://opencv.willowgarage.com/documentation/cpp/)

# 使用光流跟踪运动

在这个配方中，我们将学习如何使用 OpenCV 和流行的 Lucas Kanade 光流算法跟踪来自网络摄像头的图像中的运动。

## 准备工作

在这个配方中，我们需要使用 OpenCV，因此请参考第三章（第三章。使用图像处理技术）中的 *与 OpenCV 集成* 配方，*使用图像处理技术*，并将 OpenCV 和 CinderBlock 添加到您的项目中。将以下文件包含到您的源文件中：

```cpp
#include "cinder/Capture.h"
#include "cinder/gl/Texture.h"
#include "CinderOpenCV.h"
```

添加以下 `using` 语句：

```cpp
using namespace ci;
using namespace ci::app;
using namespace std;
```

## 如何做到这一点…

我们将读取来自相机的帧并跟踪运动。

1.  声明 `ci::gl::Texture` 和 `ci::Capture` 对象以显示和从相机捕获。同时，声明一个 `cv::Mat` 对象作为前一个帧，两个 `std::vector<cv::Point2f>` 对象以存储当前和前一个特征，以及一个 `std::vector<uint8_t>` 对象以存储每个特征的状态：

    ```cpp
        gl::Texture mTexture;
        Capture mCamera;
        cv::Mat mPreviousFrame;
        vector< cv::Point2f > mPreviousFeatures, mFeatures;
        vector< uint8_t > mFeatureStatuses;
    ```

1.  在`setup`方法中，我们将初始化`mCamera`：

    ```cpp
    try{
            mCamera = Capture( 640, 480 );
            mCamera.start();
        } catch( ... ){
            console() << "unable to initialize device" << endl;
        }
    ```

1.  在`update`方法中，我们需要检查`mCamera`是否已正确初始化，并且是否有新的帧可用：

    ```cpp
        if( mCamera ){
            if( mCamera.checkNewFrame() ){
    ```

1.  在那些`if`语句之后，我们将获取`mCamera`的`ci::Surface`引用，并将其复制到我们的`mTexture`中以进行绘制：

    ```cpp
                Surface surface = mCamera.getSurface();
                mTexture = gl::Texture( surface );
    ```

1.  现在让我们创建一个`cv::Mat`，包含当前相机帧。我们还将检查`mPreviousFrame`是否包含任何初始化的数据，计算适合跟踪的良好特征，并计算它们从先前的相机帧到当前帧的运动：

    ```cpp
                cv::Mat frame( toOcv( Channel( surface ) ) );
                if( mPreviousFrame.data != NULL ){
                    cv::goodFeaturesToTrack( frame, mFeatures, 300, 0.005f, 3.0f );
                    vector<float> errors;
                    mPreviousFeatures = mFeatures;
                    cv::calcOpticalFlowPyrLK( mPreviousFrame, frame, mPreviousFeatures, mFeatures, mFeatureStatuses, errors );
                }
    ```

1.  现在我们只需将帧复制到`mPreviousFrame`并关闭初始的`if`语句：

    ```cpp
                mPreviousFrame = frame;
            }
        }
    ```

1.  在`draw`方法中，我们将首先用黑色清除背景，并绘制`mTexture`：

    ```cpp
      gl::clear( Color( 0, 0, 0 ) ); 
        if( mTexture ){
            gl::color( Color::white() );
            gl::draw( mTexture, getWindowBounds() );
        }
    ```

1.  接下来，我们将使用`mFeatureStatus`在已跟踪的特征上绘制红色线条，以绘制已匹配的特征：

    ```cpp
        glColor4f( 1.0f, 0.0f, 0.0f, 1.0f );
        for( int i=0; i<mFeatures.size(); i++ ){
            if( (bool)mFeatureStatuses[i] == false ) continue;
            gl::drawSolidCircle( fromOcv( mFeatures[i] ), 5.0f );
        }
    ```

1.  最后，我们将使用`mFeatureStatus`绘制一条线，将先前的特征和当前的特征连接起来，以绘制已匹配的一个特征：

    ```cpp
        for( int i=0; i<mFeatures.size(); i++ ){
            if( (bool)mFeatureStatuses[i] == false ) continue;
            Vec2f pt1 = fromOcv( mFeatures[i] );
            Vec2f pt2 = fromOcv( mPreviousFeatures[i] );
            gl::drawLine( pt1, pt2 );
        }
    ```

    在以下图像中，红色点代表适合跟踪的良好特征：

    ![如何做…](img/8703OS_11_01.jpg)

## 它是如何工作的...

光流算法将对跟踪点从一个帧移动到另一个帧的距离进行估计。

## 还有更多...

在这个菜谱中，我们使用`cv::goodFeaturesToTrack`对象来计算哪些特征最适合跟踪，但也可以手动选择我们希望跟踪的点。我们只需手动将我们希望跟踪的点填充到`mFeatures`中，并将其传递给`cv::calcOpticalFlowPyrLK`对象。

# 对象跟踪

在这个菜谱中，我们将学习如何使用 OpenCV 及其相应的 CinderBlock 在 Webcam 中跟踪特定的平面对象。

## 准备工作

您需要一个描述您希望在相机中跟踪的物理对象的图像。对于这个菜谱，请将此图像放置在`assets`文件夹中，并命名为`object.jpg`。

在这个菜谱中，我们将使用 OpenCV CinderBlock，请参考第三章的*与 OpenCV 集成*菜谱，*使用图像处理技术*，并将 OpenCV 及其 CinderBlock 添加到您的项目中。

如果您使用的是 Mac，您需要自己编译 OpenCV 静态库，因为 OpenCV CinderBlock 在 OSX 上缺少一些必要的库（它将在 Windows 上正常工作）。您可以从以下链接下载正确的版本：[`sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.3/`](http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.3/)。

您需要自己使用提供的`CMake`文件编译静态库。一旦您的库正确添加到项目中，请包含以下文件：

```cpp
#include "cinder/Capture.h"
#include "cinder/gl/Texture.h"
#include "cinder/ImageIo.h"
```

添加以下`using`语句：

```cpp
using namespace ci;
using namespace ci::app;
using namespace std;
```

## 如何做...

我们将根据描述该对象的图像在相机帧中跟踪一个对象

1.  让我们首先创建一个`struct`方法来存储用于特征跟踪和匹配的必要对象。在你的应用程序类声明之前添加以下代码：

    ```cpp
    struct DetectionInfo{
        vector<cv::Point2f> goodPoints;
        vector<cv::KeyPoint> keyPoints;
        cv::Mat image, descriptor;
        gl::Texture texture;
    };
    ```

1.  在你的类声明中添加以下成员对象：

    ```cpp
    DetectionInfo mObjectInfo, mCameraInfo;
        cv::Mat mHomography;
        cv::SurfFeatureDetector mFeatureDetector;
        cv::SurfDescriptorExtractor mDescriptorExtractor;
        cv::FlannBasedMatcher mMatcher;
        vector<cv::Point2f> mCorners;
    ```

1.  在`setup`方法中，让我们首先初始化相机：

    ```cpp
        try{
            mCamera = Capture( 640, 480 );
            mCamera.start();
        } catch( ... ){
            console() << "could not initialize capture" << endl;
        }
    ```

1.  让我们调整`mCorners`的大小，加载我们的物体图像，并计算其`image`、`keyPoints`、`texture`和`descriptor`：

    ```cpp
    mCorners.resize( 4 );
        Surface objectSurface = loadImage( loadAsset( "object.jpg" ) );
        mObjectInfo.texture = gl::Texture( objectSurface );
        mObjectInfo.image = toOcv( Channel( objectSurface ) );
        mFeatureDetector.detect( mObjectInfo.image, mObjectInfo.keyPoints );
        mDescriptorExtractor.compute( mObjectInfo.image, mObjectInfo.keyPoints, mObjectInfo.descriptor );
    ```

1.  在`update`方法中，我们将检查`mCamera`是否已初始化，并且是否有新的帧需要处理：

    ```cpp
        if( mCamera ){
            if( mCamera.checkNewFrame() ){
    ```

1.  现在，让我们获取`mCamera`的表面并初始化`mCameraInfo`的`texture`和`image`对象。我们将从`cameraSurface`创建一个`ci::Channel`对象，该对象将彩色表面转换为灰度通道表面：

    ```cpp
    Surface cameraSurface = mCamera.getSurface();
    mCameraInfo.texture = gl::Texture( cameraSurface );
    mCameraInfo.image = toOcv( Channel( cameraSurface ) );
    ```

1.  让我们计算`mCameraInfo`的`features`和`descriptor`值：

    ```cpp
    mFeatureDetector.detect( mCameraInfo.image, mCameraInfo.keyPoints);
    mDescriptorExtractor.compute( mCameraInfo.image, mCameraInfo.keyPoints, mCameraInfo.descriptor );
    ```

1.  现在，让我们使用`mMatcher`来计算`mObjectInfo`和`mCameraInfo`之间的匹配：

    ```cpp
                vector<cv::DMatch> matches;
                mMatcher.match( mObjectInfo.descriptor, mCameraInfo.descriptor, matches );
    ```

1.  为了进行测试以检查错误匹配，我们将计算匹配之间的最小距离：

    ```cpp
    double minDist = 640.0;
    for( int i=0; i<mObjectInfo.descriptor.rows; i++ ){
                    double dist = matches[i].distance;
                    if( dist < minDist ){
                        minDist = dist;
                    }
                }
    ```

1.  现在，我们将添加所有距离小于`minDist*3.0`的点到`mObjectInfo.goodPoints.clear();`

    ```cpp
    mCameraInfo.goodPoints.clear();
    for( vector<cv::DMatch>::iterator it = matches.begin(); 
     it != matches.end(); ++it ){
     if( it->distance < minDist*3.0 ){
      mObjectInfo.goodPoints.push_back( 
       mObjectInfo.keyPoints[ it->queryIdx ].pt );
      mCameraInfo.goodPoints.push_back( 
       mCameraInfo.keyPoints[ it->trainIdx ].pt );
     }

    ```

1.  `}`在我们所有的点都计算并匹配后，我们需要计算`mObjectInfo`和`mCameraInfo`之间的单应性：

    ```cpp
    mHomography = cv::findHomography( mObjectInfo.goodPoints, mCameraInfo.goodPoints, CV_RANSAC );
    ```

1.  让我们创建一个`vector<cv::Point2f>`，包含我们物体的角点，并执行透视变换来计算相机图像中物体的角点：

    ### 小贴士

    不要忘记关闭我们之前打开的括号。

    ```cpp
      vector<cv::Point2f> objCorners( 4 );
      objCorners[0] = cvPoint( 0.0f, 0.0f );
      objCorners[1] = cvPoint( mObjectInfo.image.cols, 0.0f);
      objCorners[2] = cvPoint( mObjectInfo.image.cols, 
       mObjectInfo.image.rows );
      objCorners[3] = cvPoint( 0.0f, mObjectInfo.image.rows);
      mCorners = vector< cv::Point2f >( 4 );
      cv::perspectiveTransform( objCorners, mCorners, 
       mHomography );
     } 
    }
    ```

1.  让我们转到`draw`方法，首先清除背景并绘制相机和物体纹理：

    ```cpp
      gl::clear( Color( 0, 0, 0 ) );

        gl::color( Color::white() );
        if( mCameraInfo.texture ){
            gl::draw( mCameraInfo.texture, getWindowBounds() );
        }

        if( mObjectInfo.texture ){
            gl::draw( mObjectInfo.texture );
        }
    ```

1.  现在，让我们遍历`mObjectInfo`和`mCameraInfo`中的`goodPoints`值并绘制它们：

    ```cpp
    for( int i=0; i<mObjectInfo.goodPoints.size(); i++ ){
     gl::drawStrokedCircle( fromOcv( mObjectInfo.goodPoints[ i ] ),
      5.0f );
     gl::drawStrokedCircle( fromOcv( mCameraInfo.goodPoints[ i ] ),
      5.0f );
     gl::drawLine( fromOcv( mObjectInfo.goodPoints[ i ] ), 
      fromOcv( mCameraInfo.goodPoints[ i ] ) );
    }
    ```

1.  现在，让我们遍历`mCorners`并绘制找到的物体的角点：

    ```cpp
    gl::color( Color( 1.0f, 0.0f, 0.0f ) );
        gl::begin( GL_LINE_LOOP );
        for( vector<cv::Point2f>::iterator it = mCorners.begin(); it != mCorners.end(); ++it ){
            gl::vertex( it->x, it->y );
        }
        gl::end();
    ```

1.  构建并运行应用程序。拿起你在`object.jpg`图像中描述的物理物体，并将其放在图像前面。程序将尝试在相机图像中跟踪该物体，并在图像中绘制其角点。

## 它是如何工作的…

我们使用**加速鲁棒特征**（**SURF**）特征检测器和描述符来识别特征。在步骤 4 中，我们计算特征和描述符。我们使用一个`cv::SurfFeatureDetect`对象来计算物体上的良好特征以进行跟踪。然后，`cv::SurfDescriptorExtractor`对象使用这些特征来创建我们物体的描述。在步骤 7 中，我们对相机图像做同样的处理。

在步骤 8 中，我们使用一个名为`cv::FlannBasedMatcher`的**快速近似最近邻库**（**FLANN**）来执行操作。这个匹配器从相机帧和我们的物体中获取描述，并计算它们之间的匹配。

在步骤 9 和 10 中，我们使用匹配之间的最小距离来消除可能的错误匹配。结果传递到`mObjectInfo.goodPoints`和`mCameraInfo.goodPoints`。

在步骤 11 中，我们计算图像和摄像头之间的单应性。单应性是使用射影几何从一个空间到另一个空间的投影变换。我们在步骤 12 中使用它来对 `mCorners` 应用透视变换，以识别摄像头图像中的对象角落。

## 更多内容…

要了解更多关于 SURF 是什么以及它是如何工作的信息，请参考以下网页：[`en.wikipedia.org/wiki/SURF`](http://en.wikipedia.org/wiki/SURF).

要了解更多关于 FLANN 的信息，请参考网页 [`en.wikipedia.org/wiki/Nearest_neighbor_search`](http://en.wikipedia.org/wiki/Nearest_neighbor_search).

要了解更多关于单应性的信息，请参考以下网页：

[`en.wikipedia.org/wiki/Homography`](http://en.wikipedia.org/wiki/Homography).

# 读取 QR 码

在本例中，我们将使用 ZXing 库进行 QR 码读取。

## 准备工作

请从 GitHub 下载 Cinder ZXing 模块并将其解压到 `blocks` 文件夹：[`github.com/dawidgorny/Cinder-ZXing`](https://github.com/dawidgorny/Cinder-ZXing)

## 如何操作…

现在我们将创建一个 QR 码读取器：

1.  将头文件搜索路径添加到项目的构建设置中：

    ```cpp
    $(CINDER_PATH)/blocks/zxing/include
    ```

1.  将预编译的 ZXing 库路径添加到项目的构建设置中：`$(CINDER_PATH)/blocks/zxing/lib/macosx/libzxing.a`。对于调试配置，使用 `$(CINDER_PATH)/blocks/zxing/lib/macosx/libzxing_d.a`。

1.  按照以下方式将 Cinder ZXing 模块文件添加到项目结构中：![如何操作…](img/8703OS_11_02.jpg)

1.  将 `libiconv.dylib` 库添加到 `Link Binary With Libraries` 列表中：![如何操作…](img/8703OS_11_03.jpg)

1.  添加必要的头文件：

    ```cpp
    #include "cinder/gl/Texture.h"
    #include "cinder/Surface.h"
    #include "cinder/Capture.h"

    #include <zxing/qrcode/QRCodeReader.h>
    #include <zxing/common/GlobalHistogramBinarizer.h>
    #include <zxing/Exception.h>
    #include <zxing/DecodeHints.h>

    #include "CinderZXing.h"
    ```

1.  将以下成员添加到您的应用程序主类中：

    ```cpp
    Capture     mCapture;
    Surface8u   mCaptureImg;
    gl::Texture mCaptureTex;
    bool        mDetected;
    string      mData;
    ```

1.  在 `setup` 方法中，设置窗口尺寸并从摄像头初始化捕获：

    ```cpp
    setWindowSize(640, 480);

    mDetected = false;

    try {
        mCapture = Capture( 640, 480 );
        mCapture.start();
    }
    catch( ... ) {
        console() <<"Failed to initialize capture"<< std::endl;
    }
    ```

1.  按照以下方式实现 `update` 函数：

    ```cpp
    if( mCapture && mCapture.checkNewFrame() ) {
        mCaptureImg = mCapture.getSurface();
        mCaptureTex = gl::Texture( mCaptureImg );

        mDetected = false;

    try {
            zxing::Ref<zxing::SurfaceBitmapSource> source(new zxing::SurfaceBitmapSource(mCaptureImg));

            zxing::Ref<zxing::Binarizer> binarizer(NULL);
            binarizer = new zxing::GlobalHistogramBinarizer(source);

            zxing::Ref<zxing::BinaryBitmap> image(new zxing::BinaryBitmap(binarizer));
            zxing::qrcode::QRCodeReader reader;
            zxing::DecodeHints hints(zxing::DecodeHints::BARCODEFORMAT_QR_CODE_HINT);

            zxing::Ref<zxing::Result> result( reader.decode(image, hints) );

            console() <<"READ("<< result->count() <<") : "<< result->getText()->getText() << endl;

    if( result->count() ) {
                mDetected = true;
                mData = result->getText()->getText();
            }

        } catch (zxing::Exception& e) {
            cerr <<"Error: "<< e.what() << endl;
        }

    }
    ```

1.  按照以下方式实现 `draw` 函数：

    ```cpp
    gl::clear( Color( 0.1f, 0.1f, 0.1f ) );

    gl::color(Color::white());

    if(mCaptureTex) {
        gl::draw(mCaptureTex);

    }

    if(mDetected) {
        Vec2f pos = Vec2f( getWindowWidth()*0.5f, getWindowHeight()-100.f );
        gl::drawStringCentered(mData, pos);
    }
    ```

## 工作原理…

我们正在使用常规的 ZXing 库方法。由 Cinder ZXing 模块提供的 `SurfaceBitmapSource` 类实现了与 Cinder `Surface` 类型对象的集成。当 QR 码被检测并读取时，`mDetected` 标志被设置为 `true`，读取的数据存储在 `mData` 成员中。

![工作原理…](img/8703OS_11_06.jpg)

# 使用 Kinect 构建 UI 导航和手势识别

在本食谱中，我们将创建由 Kinect 传感器控制的交互式 GUI。

### 小贴士

由于 **Kinect for Windows SDK** 仅适用于 Windows，因此本食谱仅适用于 Windows 用户。

![使用 Kinect 构建 UI 导航和手势识别](img/8703OS_11_04.jpg)

## 准备工作

在本例中，我们使用的是我们在第十章“与用户交互”中的“创建对鼠标响应的交互对象”食谱中介绍的 `InteractiveObject` 类。

从 [`www.microsoft.com/en-us/kinectforwindows/`](http://www.microsoft.com/en-us/kinectforwindows/) 下载并安装 Kinect for Windows SDK。

从 GitHub 下载 KinectSDK CinderBlock [`github.com/BanTheRewind/Cinder-KinectSdk`](https://github.com/BanTheRewind/Cinder-KinectSdk)，并将其解压到`blocks`目录。

## 如何做到这一点…

我们现在将创建一个通过手势控制的手势 Cinder 应用程序。

1.  包含必要的头文件：

    ```cpp
    #include "cinder/Rand.h"
    #include "cinder/gl/Texture.h"
    #include "cinder/Utilities.h"

    #include "Kinect.h"
    #include "InteractiveObject.h";
    ```

1.  使用以下语句添加 Kinect SDK：

    ```cpp
    using namespace KinectSdk;
    ```

1.  按如下方式实现挥手手势识别的类：

    ```cpp
    class WaveHandGesture {
    public:
      enum GestureCheckResult { Fail, Pausing, Suceed };

    private:
      GestureCheckResult checkStateLeft( const Skeleton & skeleton ) {
        // hand above elbow
        if (skeleton.at(JointName::NUI_SKELETON_POSITION_HAND_RIGHT).y > skeleton.at(JointName::NUI_SKELETON_POSITION_ELBOW_RIGHT).y)
        {
          // hand right of elbow
          if (skeleton.at(JointName::NUI_SKELETON_POSITION_HAND_RIGHT).x > skeleton.at(JointName::NUI_SKELETON_POSITION_ELBOW_RIGHT).x)
          {
            return Suceed;
          }
          return Pausing;
        }
        return Fail;
      }
      GestureCheckResult checkStateRight( const Skeleton & skeleton ) {
        // hand above elbow
        if (skeleton.at(JointName::NUI_SKELETON_POSITION_HAND_RIGHT).y > skeleton.at(JointName::NUI_SKELETON_POSITION_ELBOW_RIGHT).y)
        {
          // hand left of elbow
          if (skeleton.at(JointName::NUI_SKELETON_POSITION_HAND_RIGHT).x < skeleton.at(JointName::NUI_SKELETON_POSITION_ELBOW_RIGHT).x)
          {
            return Suceed;
          }
          return Pausing;
        }
        return Fail;
      }

      int currentPhase;

    public:
      WaveHandGesture() {
        currentPhase = 0;
      }

      GestureCheckResult check( const Skeleton & skeleton )
      {
        GestureCheckResult res;
        switch(currentPhase) {
        case0: // start on left
        case2: 
          res = checkStateLeft(skeleton);
          if( res == Suceed ) { currentPhase++; }
          elseif( res == Fail ) { currentPhase = 0; return Fail; }
          return Pausing;
          break;
        case1: // to the right
        case3: 
          res = checkStateRight(skeleton);
          if( res == Suceed ) { currentPhase++; }
          elseif( res == Fail ) { currentPhase = 0; return Fail; }
          return Pausing;
          break;
        case4: // to the left
          res = checkStateLeft(skeleton);
          if( res == Suceed ) { currentPhase = 0; return Suceed; }
          elseif( res == Fail ) { currentPhase = 0; return Fail; }
          return Pausing;
          break;
        }

        return Fail;
      }
    };
    ```

1.  实现`NuiInteractiveObject`类，该类扩展了`InteractiveObject`类：

    ```cpp
    class NuiInteractiveObject: public InteractiveObject {
    public:
      NuiInteractiveObject(const Rectf & rect) : InteractiveObject(rect) {
        mHilight = 0.0f;
      }

      void update(bool activated, const Vec2f & cursorPos) {
        if(activated && rect.contains(cursorPos)) {
          mHilight += 0.08f;
        } else {
          mHilight -= 0.005f;
        }
        mHilight = math<float>::clamp(mHilight);
      }

      virtualvoid draw() {
        gl::color(0.f, 0.f, 1.f, 0.3f+0.7f*mHilight);
        gl::drawSolidRect(rect);
      }

      float mHilight;
    };
    ```

1.  实现`NuiController`类，该类管理活动对象：

    ```cpp
    class NuiController {
    public:
      NuiController() {}

      void registerObject(NuiInteractiveObject *object) {
        objects.push_back( object );
      }

      void unregisterObject(NuiInteractiveObject *object) {
        vector<NuiInteractiveObject*>::iterator it = find(objects.begin(), objects.end(), object);
        objects.erase( it );
      }

      void clear() { objects.clear(); }

      void update(bool activated, const Vec2f & cursorPos) {
        vector<NuiInteractiveObject*>::iterator it;
        for(it = objects.begin(); it != objects.end(); ++it) {
          (*it)->update(activated, cursorPos);
        }
      }

      void draw() {
        vector<NuiInteractiveObject*>::iterator it;
        for(it = objects.begin(); it != objects.end(); ++it) {
          (*it)->draw();
        }
      }

      vector<NuiInteractiveObject*> objects;
    };
    ```

1.  将成员添加到主应用程序类中，用于处理 Kinect 设备和数据：

    ```cpp
    KinectSdk::KinectRef          mKinect;
    vector<KinectSdk::Skeleton>   mSkeletons;
    gl::Texture                   mVideoTexture;
    ```

1.  添加成员以存储计算出的光标位置：

    ```cpp
    Rectf  mPIZ;
    Vec2f  mCursorPos;
    ```

1.  添加我们将用于手势识别和用户激活的成员：

    ```cpp
    vector<WaveHandGesture*>    mGestureControllers;
    bool  mUserActivated;
    int  mActiveUser;
    ```

1.  添加一个处理`NuiController`的成员：

    ```cpp
    NuiController* mNuiController;
    ```

1.  通过实现`prepareSettings`设置窗口设置：

    ```cpp
    void MainApp::prepareSettings(Settings* settings)
    {
      settings->setWindowSize(800, 600);
    }
    ```

1.  在`setup`方法中，为成员设置默认值：

    ```cpp
    mPIZ = Rectf(0.f,0.f, 0.85f,0.5f);
    mCursorPos = Vec2f::zero();

    mUserActivated = false;
    mActiveUser = 0;
    ```

1.  在`setup`方法中，为`10`个用户初始化 Kinect 和手势识别：

    ```cpp
    mKinect = Kinect::create();
    mKinect->enableDepth( false );
    mKinect->enableVideo( false );
    mKinect->start();

    for(int i = 0; i <10; i++) {
        mGestureControllers.push_back( new WaveHandGesture() );
    }
    ```

1.  在`setup`方法中，初始化由`NuiInterativeObject`类型对象组成用户界面：

    ```cpp
    mNuiController = new NuiController();

    float cols = 10.f;
    float rows = 10.f;

    Rectf rect = Rectf(0.f,0.f, getWindowWidth()/cols - 1.f, getWindowHeight()/rows - 1.f);

    or(int ir = 0; ir < rows; ir++) {
     for(int ic = 0; ic < cols; ic++) {
      Vec2f offset = (rect.getSize()+Vec2f::one()) 
       * Vec2f(ic,ir);
      Rectf r = Rectf( offset, offset+rect.getSize() );
      mNuiController->registerObject( 
       new NuiInteractiveObject® );
     } 
    }
    ```

1.  在`update`方法中，我们正在检查 Kinect 设备是否正在捕获、获取跟踪骨骼以及迭代：

    ```cpp
    if ( mKinect->isCapturing() ) {
     if ( mKinect->checkNewSkeletons() ) {
      mSkeletons = mKinect->getSkeletons();
     }
     uint32_t i = 0;
     vector<Skeleton>::const_iterator skeletonIt;
     for (skeletonIt = mSkeletons.cbegin(); 
       skeletonIt != mSkeletons.cend(); ++skeletonIt, i++ ) {
    ```

1.  在循环内部，我们正在检查骨骼是否完整，如果不完整，则停用光标控制：

    ```cpp
      if(mUserActivated && i == mActiveUser 
       && skeletonIt->size() != 
       JointName::NUI_SKELETON_POSITION_COUNT ) {
       mUserActivated = false;
      }
    ```

1.  在循环内部检查骨骼是否有效。注意我们只处理 10 个骨骼。你可以修改这个数字，但请记住在`mGestureControllers`中提供足够的动作控制器数量：

    ```cpp
    if ( skeletonIt->size() == JointName::NUI_SKELETON_POSITION_COUNT && i <10 ) {
    ```

1.  在循环和`if`语句内部，检查完成的激活手势。当骨骼被激活时，我们正在计算人机交互区域：

    ```cpp
    if( !mUserActivated || ( mUserActivated && i != mActiveUser ) ) {
        WaveHandGesture::GestureCheckResult res;
        res = mGestureControllers[i]->check( *skeletonIt );

        if( res == WaveHandGesture::Suceed && ( !mUserActivated || i != mActiveUser ) ) {
            mActiveUser = i;

         float armLen = 0;
            Vec3f handRight = skeletonIt->at(JointName::NUI_SKELETON_POSITION_HAND_RIGHT);
            Vec3f elbowRight = skeletonIt->at(JointName::NUI_SKELETON_POSITION_ELBOW_RIGHT);
            Vec3f shoulderRight = skeletonIt->at(JointName::NUI_SKELETON_POSITION_SHOULDER_RIGHT);

            armLen += handRight.distance( elbowRight );
            armLen += elbowRight.distance( shoulderRight );

            mPIZ.x2 = armLen;
            mPIZ.y2 = mPIZ.getWidth() / getWindowAspectRatio();

            mUserActivated = true;
        }
    }
    ```

1.  在循环和`if`语句内部，我们正在计算活动用户的鼠标位置：

    ```cpp
    if(mUserActivated && i == mActiveUser) {
        Vec3f handPos = skeletonIt->at(JointName::NUI_SKELETON_POSITION_HAND_RIGHT);

        Rectf piz = Rectf(mPIZ);
        piz.offset( skeletonIt->at(JointName::NUI_SKELETON_POSITION_SPINE).xy() );

        mCursorPos = handPos.xy() - piz.getUpperLeft();
        mCursorPos /= piz.getSize();
        mCursorPos.y = (1.f - mCursorPos.y);
        mCursorPos *= getWindowSize();
    }
    ```

1.  关闭打开的`if`语句和`for`循环：

    ```cpp
            }
        }
    }
    ```

1.  在`update`方法的末尾，更新`NuiController`对象：

    ```cpp
    mNuiController->update(mUserActivated, mCursorPos);
    ```

1.  按如下方式实现`draw`方法：

    ```cpp
    void MainApp::draw()
    {
      // Clear window
      gl::setViewport( getWindowBounds() );
      gl::clear( Color::white() );
      gl::setMatricesWindow( getWindowSize() );
      gl::enableAlphaBlending();

      mNuiController->draw();

      if(mUserActivated) {
        gl::color(1.f,0.f,0.5f, 1.f);
        glLineWidth(10.f);
        gl::drawStrokedCircle(mCursorPos, 25.f);
      }
    }
    ```

## 它是如何工作的…

该应用程序正在使用 Kinect SDK 跟踪用户。活动用户的骨骼数据用于根据 Microsoft 提供的 Kinect SDK 文档中的指南计算鼠标位置。激活是通过挥手手势触发的。

这是一个用户通过手势控制的鼠标控制的 UI 响应示例。光标下的网格元素会亮起，并在移出时淡出。

# 使用 Kinect 构建增强现实

在本食谱中，我们将学习如何结合 Kinect 的深度和图像帧来创建增强现实应用程序。

### 小贴士

由于 Kinect for Windows SDK 仅适用于 Windows，因此本食谱仅适用于 Windows 用户。

## 准备工作

从[`www.microsoft.com/en-us/kinectforwindows/`](http://www.microsoft.com/en-us/kinectforwindows/)下载并安装 Kinect for Windows SDK。

从 GitHub 下载 KinectSDK CinderBlock，网址为[`github.com/BanTheRewind/Cinder-KinectSdk`](https://github.com/BanTheRewind/Cinder-KinectSdk)，并将其解压到`blocks`目录中。

在此示例中，我们使用 Cinder 包中提供的示例程序之一中的资源。请将`cinder_0.8.4_mac/samples/Picking3D/resources/`中的`ducky.mshducky.png`、`phong_vert.glsl`和`phong_frag.glsl`文件复制到您的`assets`文件夹中。

## 如何做到这一点…

我们现在将创建一个使用示例 3D 模型的增强现实应用程序。

1.  包含必要的头文件：

    ```cpp
    #include "cinder/app/AppNative.h"
    #include "cinder/gl/Texture.h"
    #include "cinder/gl/GlslProg.h"
    #include "cinder/TriMesh.h"
    #include "cinder/ImageIo.h"
    #include "cinder/MayaCamUI.h"
    #include "cinder/params/Params.h"
    #include "cinder/Utilities.h"

    #include "Kinect.h"
    ```

1.  添加 Kinect SDK 的`using`语句：

    ```cpp
    using namespace KinectSdk;
    ```

1.  向主应用程序类中添加用于处理 Kinect 设备和数据的成员：

    ```cpp
    KinectSdk::KinectRef            mKinect;
    vector<KinectSdk::Skeleton>     mSkeletons;
    gl::Texture                     mVideoTexture;
    ```

1.  向存储 3D 摄像机场景属性的成员中添加成员：

    ```cpp
    CameraPersp        mCam;
    Vec3f              mCamEyePoint;
    float              mCamFov;
    ```

1.  向存储校准设置的成员中添加成员：

    ```cpp
    Vec3f    mPositionScale;
    float    mActivationDist;
    ```

1.  添加将存储几何形状、纹理和着色器程序的 3D 对象的成员：

    ```cpp
    gl::GlslProg  mShader;
    gl::Texture   mTexture;
    TriMesh       mMesh;
    ```

1.  在`setup`方法内部，设置窗口尺寸和初始值：

    ```cpp
    setWindowSize(800, 600);

    mCamEyePoint = Vec3f(0.f,0.f,1.f);
    mCamFov = 33.f;

    mPositionScale = Vec3f(1.f,1.f,-1.f);
    mActivationDist = 0.6f;
    ```

1.  在`setup`方法内部加载 3D 对象的几何形状、纹理和着色器程序：

    ```cpp
    mMesh.read( loadFile( getAssetPath("ducky.msh") ) );

    gl::Texture::Format format;
    format.enableMipmapping(true);
    ImageSourceRef img = loadImage( getAssetPath("ducky.png") );
    if(img) mTexture = gl::Texture( img, format );

    mShader = gl::GlslProg( loadFile(getAssetPath("phong_vert.glsl")), loadFile(getAssetPath("phong_frag.glsl")) );
    ```

1.  在`setup`方法内部，初始化 Kinect 设备并开始捕获：

    ```cpp
    mKinect = Kinect::create();
    mKinect->enableDepth( false );
    mKinect->start();
    ```

1.  在`setup`方法的末尾，创建用于参数调整的 GUI：

    ```cpp
    mParams = params::InterfaceGl( "parameters", Vec2i( 200, 500 ) );
    mParams.addParam("Eye Point", &mCamEyePoint);
    mParams.addParam("Camera FOV", &mCamFov);
    mParams.addParam("Position Scale", &mPositionScale);
    mParams.addParam("Activation Distance", &mActivationDist);
    ```

1.  按如下方式实现`update`方法：

    ```cpp
    void MainApp::update()
    {
      mCam.setPerspective( mCamFov, getWindowAspectRatio(), 0.1, 10000 );
      mCam.setEyePoint(mCamEyePoint);
      mCam.setViewDirection(Vec3f(0.f,0.f, -1.f*mCamEyePoint.z));

      if ( mKinect->isCapturing() ) {
        if ( mKinect->checkNewVideoFrame() ) {
          mVideoTexture = gl::Texture( mKinect->getVideo() );
        }
        if ( mKinect->checkNewSkeletons() ) {
          mSkeletons = mKinect->getSkeletons();
        }
      }
    }
    ```

1.  实现将使用纹理和着色应用来绘制我们的 3D 模型的`drawObject`方法：

    ```cpp
    void MainApp::drawObject()
    {

      mTexture.bind();
      mShader.bind();
      mShader.uniform("tex0", 0);

      gl::color( Color::white() );
      gl::pushModelView();
      gl::scale(0.05f,0.05f,0.05f);
      gl::rotate(Vec3f(0.f,-30.f,0.f));
      gl::draw( mMesh );
      gl::popModelView();

      mShader.unbind();
      mTexture.unbind();
    }
    ```

1.  按如下方式实现`draw`方法：

    ```cpp
    void MainApp::draw()
    {
      gl::setViewport( getWindowBounds() );
      gl::clear( Colorf( 0.1f, 0.1f, 0.1f ) );
      gl::setMatricesWindow( getWindowSize() );

      if ( mKinect->isCapturing() && mVideoTexture ) {
        gl::color( ColorAf::white() );
        gl::draw( mVideoTexture, getWindowBounds() );
        draw3DScene();
      }

      params::InterfaceGl::draw();
    }
    ```

1.  最后缺少的是在`draw`方法内部调用的`draw3DScene`方法。按如下方式实现`draw3DScene`方法：

    ```cpp
    gl::enableDepthRead();
    gl::enableDepthWrite();

    Vec3f mLightDirection = Vec3f( 0, 0, -1 );
    ColorA mColor = ColorA( 0.25f, 0.5f, 1.0f, 1.0f );

    gl::pushMatrices();
    gl::setMatrices( mCam );

    vector<KinectSdk::Skeleton>::const_iterator skelIt;
    for ( skelIt = mSkeletons.cbegin(); skelIt != mSkeletons.cend(); ++skelIt ) {

    if ( skelIt->size() == JointName::NUI_SKELETON_POSITION_COUNT ) {
            KinectSdk::Skeleton skel = *skelIt;

            Vec3f pos, dV;
    float armLen = 0;
            Vec3f handRight = skeletonIt->at(JointName::NUI_SKELETON_POSITION_HAND_RIGHT);
            Vec3f elbowRight = skeletonIt->at(JointName::NUI_SKELETON_POSITION_ELBOW_RIGHT);
            Vec3f shoulderRight = skeletonIt->at(JointName::NUI_SKELETON_POSITION_SHOULDER_RIGHT);

            armLen += handRight.distance( elbowRight );
            armLen += elbowRight.distance( shoulderRight );

            pos = skel[JointName::NUI_SKELETON_POSITION_HAND_RIGHT];
            dV = pos - skel[JointName::NUI_SKELETON_POSITION_SHOULDER_RIGHT];
    if( dV.z < -armLen*mActivationDist ) {
                gl::pushMatrices();
                gl::translate(pos*mPositionScale);
                drawObject();
                gl::popMatrices();
            }
        }
    }

    gl::popMatrices();

    gl::enableDepthRead(false);
    gl::enableDepthWrite(false);
    ```

1.  实现用于在程序终止时停止从 Kinect 捕获的`shutdown`方法：

    ```cpp
    void MainApp::shutdown()
    {
      mKinect->stop();
    }
    ```

## 它是如何工作的…

应用程序正在使用 Kinect SDK 跟踪用户。用户的骨骼数据用于计算从 Cinder 示例程序中获取的 3D 鸭模型的坐标。当用户的手在用户面前时，3D 模型将渲染在用户的右手上方。激活距离是通过`mActivationDist`成员值计算得出的。

![它是如何工作的…](img/8703OS_11_05.jpg)

要正确地将 3D 场景叠加到视频帧上，您必须根据 Kinect 视频摄像机设置相机 FOV。为此，我们使用`Camera FOV`属性。
