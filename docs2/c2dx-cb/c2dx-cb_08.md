# 第八章. 与硬件协同工作

本章将涵盖以下主题：

+   使用原生代码

+   使用平台更改处理

+   使用加速度传感器

+   保持屏幕开启

+   获取 dpi

+   获取最大纹理大小

# 简介

Cocos2d-x 有很多 API。然而，我们没有需要的 API，例如，内购、推送通知等。在这种情况下，我们必须创建原始 API，并且需要为 iOS 编写 Objective-C 代码或为 Android 编写 Java 代码。此外，我们希望获取正在运行的设备信息。当我们想要为每个设备进行调整时，我们必须获取设备信息，例如运行的应用程序版本、设备名称、设备上的 dpi 等。然而，这样做非常困难且令人困惑。在本章中，你可以为 iOS 或 Android 编写原生代码并获取设备信息。

# 使用原生代码

在 Cocos2d-x 中，你可以为跨平台编写一个源文件。然而，你必须为依赖过程（如购买或推送通知）编写 Objective-C 函数或 Java 函数。如果你想从 C++ 调用 Java（Android），你必须使用 **JNI**（**Java 原生接口**）。特别是，JNI 非常复杂。要从 C++ 调用 Java，你必须使用 JNI。在本食谱中，我们将解释如何从 C++ 调用 Objective-C 函数或 Java 函数。

## 准备就绪

在这种情况下，我们将创建一个新的类，名为 `Platform`。你可以通过这个类来获取应用程序的版本。在编写代码之前，你需要在你的项目中创建三个文件，分别命名为 **Platform.h**、**Platform.mm** 和 **Platform.cpp**。

![准备就绪](img/B0561_08_01.jpg)

在 Xcode 中，你很重要的一点是不要将 `Platform.cpp` 添加到 **编译源文件** 中。这就是为什么 `Platform.cpp` 是为 Android 目标而设计的，并且不需要为 iOS 构建的原因。如果你将其添加到 **编译源文件** 中，你必须从那里将其删除。

![准备就绪](img/B0561_08_02.jpg)

## 如何做到这一点...

1.  首先，你必须使用以下代码创建一个头文件，名为 `Platform.h`：

    ```cpp
    class Platform
    {
    public:
        static const char* getAppVersion();
    };
    ```

1.  你必须为 iOS 创建一个名为 `Platform.mm` 的执行文件。此代码是用 Objective-C 编写的。

    ```cpp
    #include "Platform.h"

    const char* Platform::getAppVersion()
    {
        NSDictionary* info = [[NSBundle mainBundle] 
        infoDictionary]; 
        NSString* version = [info 
        objectForKey:(NSString*)kCFBundleVersionKey]; 
        if (version) { 
            return [version UTF8String]; 
        }
        return nullptr;
    }
    ```

1.  你必须为 Android 创建一个名为 `Platform.cpp` 的执行文件。以下代码是用 C++ 编写的，并通过 JNI 使用 Java：

    ```cpp
    #include "Platform.h"
    #include "platform/android/jni/JniHelper.h"
    #define CLASS_NAME "org/cocos2dx/cpp/AppActivity"

    USING_NS_CC;

    const char* Platform::getAppVersion()
    {
        JniMethodInfo t;
        const char* ret = NULL;
        if (JniHelper::getStaticMethodInfo(t, CLASS_NAME, 
        "getAppVersionInJava", "()Ljava/lang/String;")) {
            jstring jstr = (jstring)t.env- 
    >CallStaticObjectMethod(t.classID,t.methodID); 
            std::string sstr = JniHelper::jstring2string(jstr); 
            t.env->DeleteLocalRef(t.classID); 
            t.env->DeleteLocalRef(jstr); 
            ret = sstr.c_str(); 
        }
        return ret;
    }
    ```

1.  当你在项目中添加新的类文件时，你必须编辑 `proj.android/jni/Android.mk` 以构建 Android。

    ```cpp
    LOCAL_SRC_FILES := hellocpp/main.cpp \ 
                       ../../Classes/AppDelegate.cpp \ 
                       ../../Classes/HelloWorldScene.cpp \
                       ../../Classes/Platform.cpp
    ```

1.  接下来，你必须编写 `AppActivity.java` 中的 Java 代码。此文件名为 `pro.android/src/org/cocos2dx/cpp/AppActivity.java`。

    ```cpp
    public class AppActivity extends Cocos2dxActivity { 
        public static String appVersion = "";

        @Override 
        protected void onCreate(Bundle savedInstanceState) { 
            super.onCreate(savedInstanceState); 

            try { 
                PackageInfo packageInfo = 
    getPackageManager().getPackageInfo(getPackageName(), 
    PackageManager.GET_META_DATA); 
                appVersion = packageInfo.versionName; 
            } catch (NameNotFoundException e) { 
            }
        }

        public static String getAppVersionInJava() { return appVersion; 
        }
    }
    ```

1.  最后，你可以通过以下代码获取你游戏的版本：

    ```cpp
    #include "Platform.h" 

    const char* version = Platform::getAppVersion(); 
    CCLOG("application version = %s", version);
    ```

## 它是如何工作的...

1.  首先，我们将从 iOS 开始。你将能够通过在 `Platform.mm` 中使用 Objective-C 来获取你游戏的版本。你可以在 `.mm` 文件中编写 C++ 和 Objective-C。

1.  接下来，我们将寻找 Android。当您在 Android 设备上调用 `Platform::getAppversion` 时，将执行 `Platform.cpp` 中的方法。在这个方法中，您可以通过 JNI 调用 `AppActivity.java` 中的 `getAppVersionInJava` 方法。C++ 可以通过 JNI 连接到 Java。这就是为什么您只能通过 Java 来获取应用程序版本的原因。

1.  在 Java 中，您可以通过使用 `onCreate` 方法来获取您应用程序的版本。您可以将其设置为静态变量，然后从 Java 中的 `getAppVersionInJava` 方法中获取它。

## 还有更多...

您可以通过在 Cocos2d-x 中使用 `JniHelper` 类轻松地使用 JNI。这个类如何从 C++ 中管理错误并创建 C++ 和 Java 之间的桥梁已经解释过了。您可以通过以下代码使用 `JniHelper` 类：

```cpp
JniMethodInfo t; 
JniHelper::getStaticMethodInfo(t, CLASS_NAME, 
"getAppVersionInJava", 
"()Ljava/lang/String;")
```

您可以使用 `JniHelper::getStaticMethodInfo` 来获取 Java 方法的信息。第一个参数是 `JniMethodInfo` 类型的变量。第二个参数是包含您要调用的方法的类的名称。第三个参数是方法名称。最后一个参数是此方法的参数。此参数由返回值和参数决定。括号中的字符是 Java 方法的参数。在这种情况下，此方法没有参数。括号后面的字符是返回值。`Ljava/lang/String` 表示返回值是一个字符串。如果您可以轻松地获取此参数，则应使用名为 `javap` 的命令。使用此命令将生成以下结果。

```cpp
$ cd /path/to/project/pro.android/bin/classes 
$ javap -s org.cocos2dx.cpp.AppActivity 
Compiled from "AppActivity.java" 
public class org.cocos2dx.cpp.AppActivity extends 
org.cocos2dx.lib.Cocos2dxActivity { 
  public static java.lang.String appVersion; 
    descriptor: Ljava/lang/String; 
  public org.cocos2dx.cpp.AppActivity(); 
    descriptor: ()V 

  protected void onCreate(android.os.Bundle); 
    descriptor: (Landroid/os/Bundle;)V 

  public static java.lang.String getAppVersionInJava(); 
    descriptor: ()Ljava/lang/String; 

  static {}; 
    descriptor: ()V 
}
```

从上述结果中，您可以看到 `getAppVersionInJava` 方法的参数为 `()Ljava/lang/String`;

如前所述，您可以将 Java 方法的信息作为 `t` 变量获取。因此，您可以通过这个变量和以下代码来调用 Java 方法：

```cpp
jstring jstr = (jstring)t.env- >CallStaticObjectMethod(t.classID,t.methodID);
```

# 使用平台更改处理方式

您可以使程序在每种操作系统的源代码的特定部分运行。例如，您将根据平台更改文件名、文件路径或图像缩放。在这个菜谱中，我们将介绍在出现问题时根据所选平台进行分支代码的情况。

## 如何做到这一点...

您可以通过以下方式使用预处理器来更改处理方式：

```cpp
#if (CC_TARGET_PLATFORM == CC_PLATFORM_ANDROID) 
    CCLOG("this platform is Android"); 
#elif (CC_TARGET_PLATFORM == CC_PLATFORM_IOS) 
    CCLOG("this platform is iOS"); 
#else 
    CCLOG("this platfomr is others");
#endif
```

## 它是如何工作的...

Cocos2d-x 在 `CCPlatformConfig.h` 中定义了 `CC_TARGET_PLATFORM` 值。如果您的游戏是为 Android 设备编译的，则 `CC_TARGET_PLATFORM` 等于 `CC_PLATFORM_ANDROID`。如果它是为 iOS 设备编译的，则 `CC_TARGET_PLATFORM` 等于 `CC_PLATFORM_IOS`。不用说，除了 Android 和 iOS 之外，还有其他值。请检查 `CCPlatformConfig.h`。

## 还有更多...

在预处理器中使用的代码在编辑器上难以阅读。此外，在编译您的代码之前，您可能无法注意到错误。您应该定义一个可以由预处理器更改的常量值，但您应该尽可能多地使用代码来更改处理方式。您可以使用 Cocos2d-x 中的 `Application` 类来检查平台，如下所示：

```cpp
switch (Application::getInstance()->getTargetPlatform()) { 
        case Application::Platform::OS_ANDROID: 
            CCLOG("this device is Android"); 
            break;
        case Application::Platform::OS_IPHONE: 
            CCLOG("this device is iPhone"); 
            break;
        case Application::Platform::OS_IPAD: 
            CCLOG("this device is iPad"); 
            break;
        default: 
            break;
}
```

你可以使用`Application::getTargetPlatform`方法获取平台值。你将能够检查，不仅仅是 iPhone 或 iPad，还可以通过此方法检查 IOS。

# 使用加速度传感器

通过在设备上使用加速度传感器，我们可以通过使用摇动和倾斜设备等操作使游戏更加吸引人。例如，通过倾斜屏幕移动球，瞄准目标的迷宫游戏，以及试图减肥的瘦熊猫，在这些游戏中玩家需要摇动设备来玩游戏。你可以通过加速度传感器获取设备的倾斜值和移动速度。如果你能使用它，你的游戏就会更加独特。在这个菜谱中，我们学习如何使用加速度传感器。

## 如何操作...

你可以通过以下代码从加速度传感器获取 x、y 和 z 轴的值：

```cpp
Device::setAccelerometerEnabled(true);
auto listener = EventListenerAcceleration::create([](Acceleration* 
acc, Event* event){ 
    CCLOG("x=%f, y=%f, z=%f", acc->x, acc->y, acc->z); 
}); 
this->getEventDispatcher()- 
>addEventListenerWithSceneGraphPriority(listener, this);
```

## 它是如何工作的...

1.  首先，你通过使用`Device::setAccelerometerEnable`方法启用加速度传感器。`Device`类中的方法是静态方法。因此，你可以直接调用方法，而不需要实例，如下所示：

    ```cpp
    Device::setAccelerometerEnable(true);
    ```

1.  你为获取加速度传感器的值设置事件监听器。在这种情况下，你可以通过 lambda 函数获取这些值。

1.  最后，你在事件分发器中设置事件监听器。

1.  如果你在这台真实设备上运行此代码，你可以从加速度传感器获取 x、y 和 z 轴的值。x 轴是斜坡的左右。y 轴是斜坡的前后。z 轴是垂直运动。

## 还有更多...

加速度传感器会消耗更多的电量。当你使用它时，你为事件发生设置一个适当的间隔。以下代码将间隔设置为 1 秒。

```cpp
Device::setAccelerometerInterval(1.0f);
```

### 小贴士

如果间隔较高，我们可能会错过一些倾斜输入。然而，如果我们使用较低的间隔，我们会消耗大量的电量。

# 保持屏幕开启

在玩游戏时，你必须确保设备不会进入睡眠模式。例如，在你的游戏中，玩家可以通过使用加速度传感器来控制游戏并保持游戏进行。问题是，如果玩家在用加速度传感器玩游戏时不触摸屏幕，设备就会进入睡眠模式并进入后台模式。在这个菜谱中，你可以轻松地保持屏幕开启。

## 如何操作...

如果你使用`Device::setKeepScreenOn`方法将其设置为`true`，你可以保持屏幕开启：

```cpp
Device::setKeepScreenOn(true);
```

## 它是如何工作的...

每个平台都有不同的方法来防止设备进入睡眠模式。然而，Cocos2d-x 可以为每个平台做到这一点。你可以使用这种方法而不需要执行平台。在 iOS 平台上，`setKeepScreenOn`方法如下：

```cpp
void Device::setKeepScreenOn(bool value) 
{
    [[UIApplication sharedApplication] 
setIdleTimerDisabled:(BOOL)value]; 
}
```

在 Android 平台上，方法如下：

```cpp
public void setKeepScreenOn(boolean value) { 
    final boolean newValue = value; 
    runOnUiThread(new Runnable() { 
        @Override
        public void run() { 
            mGLSurfaceView.setKeepScreenOn(newValue); 
        }
    });
}
```

# 获取 dpi

每个设备都有许多**dpi**（每英寸点数）的变化。您可以通过分辨率准备几种不同类型的图像。您可能想要根据设备上的 dpi 更改图像。在这个菜谱中，如果您想获取游戏正在运行的 dpi，您需要使用 Cocos2d-x 函数。

## 如何操作...

您可以通过使用`Device::getDPI`方法来获取设备正在执行游戏时的 dpi（每英寸点数），如下所示：

```cpp
int dpi = Device::getDPI(); 
CCLOG("dpi = %d", dpi);
```

## 工作原理...

实际上，我们检查了一些设备的 dpi。为了使用 dpi 信息，您可以进一步调整多屏幕分辨率。

| 设备 | Dpi |
| --- | --- |
| iPhone 6 Plus | 489 |
| iPhone 6 | 326 |
| iPhone 5s | 326 |
| iPhone 4s | 326 |
| iPad Air | 264 |
| iPad 2 | 132 |
| Nexus 5 | 480 |

# 获取最大纹理大小

可使用的最大纹理大小因设备而异。特别是，当您使用纹理图集时，应该小心。这就是为什么包含大量图像的纹理图集体积会变得很大。您不能使用超过最大尺寸的纹理。如果您使用它，您的游戏将会崩溃。在这个菜谱中，您可以获取最大纹理大小。

## 如何操作...

您可以通过以下代码轻松获取最大纹理大小：

```cpp
auto config = Configuration::getInstance();
int texutureSize = config->getMaxTextureSize();
CCLOG("max texture size = %d", texutureSize);
```

## 工作原理...

`Configuration`类是一个单例类。这个类有一些 OpenGL 变量。OpenGL 是一个用于渲染 2D 和 3D 矢量图形的多平台 API。它使用起来相当困难。Cocos2d-x 将其封装起来，使其易于使用。OpenGL 有很多关于图形的信息。最大纹理大小是提供这些信息的一个变量。您可以得到您应用程序正在运行的设备的最大纹理大小。

## 还有更多...

您可以获取其他 OpenGL 变量。如果您想检查`Configuration`拥有的变量，您将使用`Configuration::getInfo`方法。

```cpp
auto config = Configuration::getInstance(); 
std::string info = config->getInfo(); 
CCLOG("%s", info.c_str());
```

iPhone 6 Plus 上的日志结果：

```cpp
{
  gl.supports_vertex_array_object: true  cocos2d.x.version: 
  cocos2d-x 3.5 
  gl.vendor: Apple Inc. 
  gl.supports_PVRTC: true 
  gl.renderer: Apple A8 GPU 
  cocos2d.x.compiled_with_profiler: false 
  gl.max_texture_size: 4096 
  gl.supports_ETC1: false 
  gl.supports_BGRA8888: false 
  cocos2d.x.build_type: RELEASE 
  gl.supports_discard_framebuffer: true 
  gl.supports_NPOT: true 
  gl.supports_ATITC: false 
  gl.max_samples_allowed: 4 
  gl.max_texture_units: 8 
  cocos2d.x.compiled_with_gl_state_cache: true 
  gl.supports_S3TC: false 
  gl.version: OpenGL ES 2.0 Apple A8 GPU - 53.13 
}
```

如果您获取每个变量，并检查`Configuration`类的源代码，您可以轻松理解它们。
