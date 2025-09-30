# 第十一章。利用优势

本章将涵盖以下主题：

+   使用加密的精灵图集

+   使用加密的 zip 文件

+   使用加密的 SQLite 文件

+   创建观察者模式

+   使用 HTTP 进行网络连接

# 简介

到目前为止，我们已经解释了 Cocos2d-x 中的基本技术信息。它支持在智能手机上开发游戏。实际上，您可以使用 Cocos2d-x 的基本功能创建您自己的游戏。然而，如果您的游戏成为热门，作弊者可能会尝试破解代码。因此，在某些情况下，加密是必要的，以防止未经授权访问您的游戏数据。加密是游戏开发中的一个重要方面，因为它可以帮助您保护代码，防止人们破坏游戏的整体体验，并且还可以防止游戏被非法破解。在本章中，您将学习如何加密您的游戏资源。

# 使用加密的精灵图集

对于黑客来说，从应用程序中提取资源文件非常容易。这对版权来说是一个巨大的担忧。精灵图集可以使用`TexturePacker`非常容易地加密。在本食谱中，您将学习如何加密您的精灵以保护它们免受黑客和作弊者的侵害。

## 如何操作...

要使用`TexturePacker`加密精灵图集，您需要在`TexturePacker`的左侧面板上设置它。然后，您需要按照这里写的步骤操作，以成功加密您的精灵。

1.  将纹理格式更改为`zlib compr. PVR(.pvr.ccz, Ver.2)`

1.  点击**内容保护**图标，您将看到一个额外的窗口，用于设置密码。

1.  在以下屏幕截图所示的文本输入区域中输入加密密钥。您可以输入您喜欢的密钥。然而，输入 32 个十六进制数字很难，因此，您只需点击**创建新密钥**按钮。点击后，您会发现它会自动输入**加密密钥**。![如何操作...](img/B0561_11_01.jpg)

1.  记下这个加密密钥。这是您将需要用来解密加密文件的密钥。

1.  最后，您可以发布加密的精灵图集。

## 它是如何工作的...

现在，让我们看看如何使用这些加密的精灵图集。

1.  按照以下图像所示将加密的精灵图集添加到您的项目中：![如何工作...](img/B0561_11_02.jpg)

1.  在`HelloWorld.cpp`中包含`ZipUtils`类以解密。

    ```cpp
    #include "ZipUtils.h"
    ```

1.  设置用于加密的`TexturePacker`的加密密钥。

    ```cpp
    ZipUtils::setPvrEncryptionKey (0x5f2c492e, 0x635eaaf8, 0xe5a4ee49, 0x32ffe0cf);
    ```

1.  最后，使用加密的精灵图集创建精灵。

    ```cpp
    Size visibleSize = Director::getInstance()- >getVisibleSize();
    Vec2 origin = Director::getInstance()->getVisibleOrigin();

    SpriteFrameCache::getInstance()- >addSpriteFramesWithFile("res/encrypted.plist");
    auto sprite = Sprite::createWithSpriteFrameName("run_01.png");
    sprite->setPosition(Vec2(visibleSize/2)+origin);
    this->addChild(sprite);
    ```

## 更多内容…

应用程序通常有很多精灵图集。您可以为每个精灵图集使用不同的加密密钥。但这可能会造成一些混淆。您需要在应用程序中的所有精灵图集中使用相同的密钥。第一次，您需要单击**创建新密钥**按钮来创建加密密钥。然后，您需要单击**另存为全局密钥**按钮将加密密钥保存为全局密钥。下次，当您创建新的加密精灵图集时，您可以通过单击**使用全局密钥**按钮将此加密密钥设置为全局密钥。

现在，我们将继续了解如何检查加密的精灵图集。加密的精灵图集的扩展名是 `.ccz`。

1.  双击具有 `.ccz` 扩展名的加密文件。

1.  启动 Texture Packer，您将看到需要输入解密密钥的窗口，如图所示：![还有更多…](img/B0561_11_03.jpg)

1.  输入解密密钥或单击**使用全局密钥**按钮。如果您已将密钥保存为全局密钥，则单击**确定**按钮。

1.  如果密钥是正确的密钥，您将看到如图所示的精灵图集：

# 使用加密的 zip 文件

在智能手机中，游戏经常从服务器下载 `zip` 文件以更新资源。这些资源通常是黑客的主要目标。他们可以解密这些资源以操纵游戏系统中的信息。因此，这些资源的安全性非常重要。在这种情况下，`zip` 文件被加密以防止作弊者。在本教程中，您将学习如何使用密码解压加密的 `zip` 文件。

## 准备工作

Cocos2d-x 有一个解压库。然而，在这个库中加密/解密是禁用的。这就是为什么我们必须在 `unzip.cpp` 中启用 crypt 选项。此文件的路径是 `cocos2d/external/unzip/unzip.cpp`。您将不得不注释掉 `unzip.cpp` 中的第 71 行以启用 crypt 选项。

```cpp
//#ifndef NOUNCRYPT
//        #define NOUNCRYPT
//#endif
```

当我们尝试在 Cocos2d-x 版本 3.7 中构建时，`unzip.h` 中的第 46 行出现了错误，如下所示：

```cpp
#include "CCPlatformDefine.h"
```

您必须编辑以下代码以移除此错误，如下所示：

```cpp
#include "platform/CCPlatformDefine.h"
```

## 如何操作...

首先，在 `HelloWorld.cpp` 中包含 `unzip.h` 文件以使用解压库，如下所示：

```cpp
#include "external/unzip/unzip.h"
```

接下来，让我们尝试使用密码解压加密的 zip 文件。这可以通过在 `HelloWorld.cpp` 中添加以下代码来完成：

```cpp
#define BUFFER_SIZE    8192
#define MAX_FILENAME   512

bool HelloWorld::uncompress(const char* password)
{
    // Open the zip file
    std::string outFileName = FileUtils::getInstance()- 
    >fullPathForFilename("encrypt.zip"); 
    unzFile zipfile = unzOpen(outFileName.c_str()); 
    int ret = unzOpenCurrentFilePassword(zipfile, password); 
    if (ret!=UNZ_OK) { CCLOG("can not open zip file %s", outFileName.c_str()); 
        return false;
    }

    // Get info about the zip file
    unz_global_info global_info;
    if (unzGetGlobalInfo(zipfile, &global_info) != UNZ_OK) {
        CCLOG("can not read file global info of %s", 
        outFileName.c_str());
        unzClose(zipfile);
        return false;
    }

    CCLOG("start uncompressing");

    // Loop to extract all files.
    uLong i;
    for (i = 0; i < global_info.number_entry; ++i) {
        // Get info about current file.
        unz_file_info fileInfo;
        char fileName[MAX_FILENAME];
        if (unzGetCurrentFileInfo(zipfile, &fileInfo, fileName, 
        MAX_FILENAME, nullptr, 0, nullptr,  0) != UNZ_OK) {
            CCLOG("can not read file info");
            unzClose(zipfile);
            return false;
        }

        CCLOG("filename = %s", fileName);

        unzCloseCurrentFile(zipfile);

        // Goto next entry listed in the zip file.
        if ((i+1) < global_info.number_entry) {
            if (unzGoToNextFile(zipfile) != UNZ_OK) {
                CCLOG("can not read next file");
                unzClose(zipfile);
                return false;
            }
        }
    }

    CCLOG("end uncompressing");
    unzClose(zipfile);

    return true;
}
```

最后，您可以通过指定密码来解压加密的 zip 文件以使用此方法。如果密码是 `cocos2d-x`，您可以使用以下代码解压：

```cpp
this->uncompress("cocos2d-x");
```

## 它是如何工作的...

1.  使用 `unzOpen` 函数打开加密的 zip 文件，如下所示：

    ```cpp
    unzFile zipfile = unzOpen(outFileName.c_str());
    ```

1.  在使用 `unzOpen` 函数打开它之后，再次使用 `unzOpenCurrentFilePassword` 函数打开，如下所示：

    ```cpp
    int ret = unzOpenCurrentFilePassword(zipfile, password);
    if (ret!=UNZ_OK) {
        CCLOG("can not open zip file %s", outFileName.c_str());
        return false;
    }
    ```

1.  之后，您可以继续使用与解压未加密的 zip 文件相同的方式。

# 使用加密的 SQLite 文件

我们经常使用 SQLite 来保存用户数据或游戏数据。SQLite 是一个强大且有用的数据库。然而，在你的游戏沙盒中有一个数据库文件。作弊者会从你的游戏中获取它，并对其进行编辑以作弊。在这个菜谱中，你将学习如何加密你的 SQLite 并防止作弊者编辑它。

## 准备中

我们将使用 `wxSqlite` 库来加密 SQLite。这是一个免费软件。首先，你需要在 Cocos2d-x 中安装 `wxSqlite` 并编辑一些代码，并在 Cocos2d-x 中设置文件。

1.  下载 `wxSqlite3` 项目的 zip 文件。访问以下网址：[`sourceforge.net/projects/wxcode/files/Components/wxSQLite3/wxsqlite3-3.1.1.zip/download`](http://sourceforge.net/projects/wxcode/files/Components/wxSQLite3/wxsqlite3-3.1.1.zip/download)

1.  解压 zip 文件。

1.  在 `cocos2d/external` 下创建一个名为 `wxsqlite` 的新文件夹。

1.  在展开文件夹后，将 `sqlite3/secure/src` 复制到 `cocos2d/external/wxsqlite`，如下截图所示：![准备中](img/B0561_11_04.jpg)

1.  将在第 4 步中添加到 `wxsqlite/src` 的 `sqlite3.h` 和 `sqlite3secure.c` 添加到你的项目中，如下截图所示：![准备中](img/B0561_11_05.jpg)

1.  在 Xcode 的 **构建设置** 中的 **其他 C 标志** 中添加 `-DSQLITE_HAS_CODEC`，如下截图所示：![准备中](img/B0561_11_06.jpg)

1.  在 `cocos2d/external/wxsqlite` 中创建一个名为 `Android.mk` 的新文件，如下代码所示：

    ```cpp
    LOCAL_PATH := $(call my-dir)
    include $(CLEAR_VARS)
    LOCAL_MODULE := wxsqlite3_static
    LOCAL_MODULE_FILENAME := libwxsqlite3
    LOCAL_CFLAGS += -DSQLITE_HAS_CODEC
    LOCAL_SRC_FILES := src/sqlite3secure.c
    LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/src
    LOCAL_C_INCLUDES := $(LOCAL_PATH)/src
    include $(BUILD_STATIC_LIBRARY)
    ```

1.  在 `cocos2d/cocos/storage/local-storage` 中编辑 `Android.mk`，如下代码所示：

    ```cpp
    LOCAL_PATH := $(call my-dir)
    include $(CLEAR_VARS)

    LOCAL_MODULE := cocos_localstorage_static

    LOCAL_MODULE_FILENAME := liblocalstorage

    LOCAL_SRC_FILES := LocalStorage.cpp

    LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/..

    LOCAL_C_INCLUDES := $(LOCAL_PATH)/../..

    LOCAL_CFLAGS += -Wno-psabi
    LOCAL_CFLAGS += -DSQLITE_HAS_CODEC
    LOCAL_EXPORT_CFLAGS += -Wno-psabi

    LOCAL_WHOLE_STATIC_LIBRARIES := cocos2dx_internal_static
    LOCAL_WHOLE_STATIC_LIBRARIES += wxsqlite3_static

    include $(BUILD_STATIC_LIBRARY)

    $(call import-module,.)
    ```

1.  在 `cocos2d/cocos/storage/local-storage` 中编辑 `LocalStorage.cpp`。注释掉第 33 行和第 180 行，如下代码所示。

    `LocalStorage.cpp` 第 33 行：

    ```cpp
    //#if (CC_TARGET_PLATFORM != CC_PLATFORM_ANDROID)
    ```

    `LocalStorage.cpp` 第 180 行：

    ```cpp
    //#endif // #if (CC_TARGET_PLATFORM != CC_PLATFORM_ANDROID)
    ```

1.  在 `proj.andorid/jni` 中编辑 `Android.mk`，如下代码所示：

    ```cpp
    LOCAL_SRC_FILES := hellocpp/main.cpp \
                       ../../Classes/AppDelegate.cpp \
                       ../../Classes/HelloWorldScene.cpp \
                       ../../cocos2d/external/wxsqlite/src/sqlite3secure.c

    LOCAL_C_INCLUDES := $(LOCAL_PATH)/../../Classes
    LOCAL_C_INCLUDES += $(LOCAL_PATH)/../../cocos2d/external/wxsqlite/src/
    LOCAL_CFLAGS += -DSQLITE_HAS_CODEC
    ```

在此之后，SQLite 被加密，可以在你的项目中使用。

## 如何操作...

1.  你必须包含 `sqlite3.h` 以使用 SQLite API。

    ```cpp
    #include "sqlite3.h"
    ```

1.  创建加密的数据库，如下代码所示：

    ```cpp
    std::string dbname = "data.db";
    std::string path = FileUtils::getInstance()->getWritablePath() + dbname;
    CCLOG("%s", path.c_str());

    sqlite3 *database = nullptr;
    if ((sqlite3_open(path.c_str(), &database) != SQLITE_OK)) {
        sqlite3_close(database);
        CCLOG("open error");
    } else {
        const char* key = "pass_phrase";
        sqlite3_key(database, key, (int)strlen(key));

        // sql: create table
        char create_sql[] = "CREATE TABLE sample ( "
        "               id     INTEGER PRIMARY KEY, "
        "               key    TEXT    NOT NULL,    "
        "               value  INTEGER NOT NULL     "
        "             )                             ";

        // create table
        sqlite3_exec(database, create_sql, 0, 0, NULL);

        // insert data
        char insert_sql[] = "INSERT INTO sample ( id, key, value )"
        "            values (%d, '%s', '%d')     ";

        char insert_record[3][256];
        sprintf(insert_record[0],insert_sql,0,"test",300);
        sprintf(insert_record[1],insert_sql,1,"hoge",100);
        sprintf(insert_record[2],insert_sql,2,"foo",200);

        for(int i = 0; i < 3; i++ ) {
            sqlite3_exec(database, insert_record[i], 0, 0, NULL);
        }

        sqlite3_reset(stmt);
        sqlite3_finalize(stmt);
        sqlite3_close(database);
    }
    ```

1.  从加密的数据库中选择数据，如下代码所示：

    ```cpp
    std::string dbname = "data.db";
    std::string path = FileUtils::getInstance()->getWritablePath() + dbname;
    CCLOG("%s", path.c_str());

    sqlite3 *database = nullptr;
    if ((sqlite3_open(path.c_str(), &database) != SQLITE_OK)) {
        sqlite3_close(database);
        CCLOG("open error");
    } else {
        const char* key = "pass_phrase";
        sqlite3_key(database, key, (int)strlen(key));

        // select data
        sqlite3_stmt *stmt = nullptr;

        std::string sql = "SELECT value FROM sample WHERE key='test'";
        if (sqlite3_prepare_v2(database, sql.c_str(), -1, &stmt, NULL) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                int value = sqlite3_column_int(stmt, 0);
                CCLOG("value = %d", value);
            } else {
                CCLOG("error , error=%s", sqlite3_errmsg(database));
            }
        }

        sqlite3_reset(stmt);
        sqlite3_finalize(stmt);
        sqlite3_close(database);
    }
    ```

## 它是如何工作的...

首先，你必须使用 `pass` 语句创建加密的数据库。要创建它，请按照以下三个步骤进行：

1.  正常打开数据库。

1.  接下来，使用 `sqlite3_key` 函数设置密码短语。

    ```cpp
    const char* key = "pass_phrase";
    sqlite3_key(database, key, (int)strlen(key));
    ```

1.  最后，执行 SQL 语句来创建表。

在此之后，你将需要在应用程序中使用加密的数据库文件。你可以从 CCLOG 打印的路径中获取它。

要从中选择数据，使用相同的方法。在打开数据库后，你可以使用相同的密码短语从加密的数据库中获取数据。

## 还有更多...

你可能想知道这个数据库是否真的被加密了。那么让我们检查一下。使用命令行打开数据库并执行以下命令：

```cpp
$ sqlite3 data.db 
SQLite version 3.8.4.3 2014-04-03 16:53:12
Enter ".help" for usage hints.
sqlite> .schema
Error: file is encrypted or is not a database
sqlite>

```

如果数据库被加密，你将无法打开它，并且会弹出错误消息，如下所示：

```cpp
"file is encrypted or is not a database".

```

# 创建观察者模式

事件调度器是一种响应触摸屏幕、键盘事件和自定义事件等事件的机制。你可以使用事件调度器来获取事件。此外，你可以在设计模式中使用它来创建观察者模式。在本教程中，你将学习如何使用事件调度器以及如何在 Cocos2d-x 中创建观察者模式。

## 准备工作

首先，我们将详细介绍观察者模式。观察者模式是一种设计模式。当事件发生时，观察者通知观察者中注册的主题。它主要用于实现分布式事件处理。观察者模式也是 MVC 架构中的关键部分。

![准备工作](img/B0561_11_07.jpg)

## 如何操作...

在本教程中，我们将每秒创建一个计数标签。当触摸屏幕时，计数标签将在此位置创建，然后使用观察者模式每秒进行计数。

1.  创建一个扩展`Label`类的`Count`类，如下面的代码所示：

    ```cpp
    Count.h
    class Count : public cocos2d::Label
    {
    private:
        int _count;
        void countUp(float dt);
    public:
        ~Count();
        virtual bool init();
        CREATE_FUNC(Count);
    };
    Count.cpp
    Count::~Count()
    {
        this->getEventDispatcher()- >removeCustomEventListeners("TimeCount");
    }

    bool Count::init()
    {
        if (!Label::init()) {
            return false;
        }

        _count = 0;

        this->setString("0");
        this->setFontScale(2.0f);

        this->getEventDispatcher()- >addCustomEventListener("TimeCount", = { this->countUp(0); });

        return true;
    }

    void Count::countUp(float dt)
    {
        _count++;
        this->setString(StringUtils::format("%d", _count));
    }
    ```

1.  接下来，当触摸屏幕时，此标签将在触摸位置创建，并使用调度器每秒调用`HelloWorld::countUp`方法，如以下`HelloWorld.cpp`中的代码所示：

    ```cpp
    bool HelloWorld::init()
    {
        if ( !Layer::init() )
        {
            return false;
        }

        auto listener = EventListenerTouchOneByOne::create();
        listener->setSwallowTouches(_swallowsTouches);
        listener->onTouchBegan = C_CALLBACK_2(HelloWorld::onTouchBegan, this);
        this->getEventDispatcher()- >addEventListenerWithSceneGraphPriority(listener, this);

        this->schedule(schedule_selector(HelloWorld::countUp), 1.0f);

        return true;
    }

    bool HelloWorld::onTouchBegan(cocos2d::Touch *touch, cocos2d::Event *unused_event) {
        auto countLabel = Count::create(); this->addChild(countLabel); countLabel->setPosition(touch->getLocation()); 
        return true; }

    void HelloWorld::countUp(float dt)
    {
        this->getEventDispatcher()- >dispatchCustomEvent("TimeCount"); }
    ```

1.  构建并运行此项目后，当你触摸屏幕时，它将在触摸位置创建一个计数标签，然后你会看到标签同时每秒进行计数。

## 它是如何工作的...

1.  添加名为`TimeCount`的自定义事件。如果`TimeCount`事件发生，则调用`Count::countUp`方法。

    ```cpp
    this->getEventDispatcher()- >addCustomEventListener("TimeCount", = {
        this->countUp(0);
    });
    ```

1.  不要忘记，当移除`Count`类的实例时，需要从`EventDispatcher`中移除自定义事件。如果你忘记这样做，那么当事件发生时，`EventDispatcher`将调用`zombie`实例，你的游戏将会崩溃。

    ```cpp
    this->getEventDispatcher()- >removeCustomEventListeners("TimeCount");
    ```

1.  在`HelloWorld.cpp`中，使用调度器调用`HelloWorld::countUp`方法。`HelloWorld::countUp`方法调用名为`TimeOut`的自定义事件。

    ```cpp
    this->getEventDispatcher()- >dispatchCustomEvent("TimeCount");
    ```

    然后，`EventDispatcher`将通知列出的主题。在这种情况下，调用`Count::countUp`方法。

    ```cpp
    void Count::countUp(float dt){
        _count++;
        this->setString(StringUtils::format("%d", _count));
    }
    ```

## 还有更多...

使用`EventDispatcher`，标签同时计数。如果你使用调度器而不是`EventDispatcher`，你会注意到一些不同之处。

按照以下代码更改`Count::init`方法：

```cpp
bool Count::init()
{
    if (!Label::init()) {
        return false;
    }

    _count = 0;

    this->setString("0");
    this->setFontScale(2.0f);
    this->schedule(schedule_selector(Count::countUp), 1.0f); 
    return true;
}
```

在此代码中，通过每秒调用`Count::countUp`方法使用调度器。你可以看到，标签不是同时计数的。每个标签都是每秒进行计数，然而不是同时。使用观察者模式，可以同时调用许多主题。

# 使用 HTTP 进行网络连接

在最近的智能手机游戏中，我们通常使用互联网网络来更新数据、下载资源等。没有网络的游戏是不存在的。在本教程中，你将学习如何使用网络下载资源。

## 准备工作

要使用网络，必须包含`network/HttpClient`的头文件。

```cpp
 #include "network/HttpClient.h"
```

如果你要在 Android 设备上运行它，你需要编辑`proj.android/AndroidManifest.xml`。

```cpp
<user-permission android:name="android.permission.INTERNET" />
```

## 如何操作...

在下面的代码中，我们将从[`google.com/`](http://google.com/)获取响应，然后，将响应数据作为日志打印出来。

```cpp
auto request = new network::HttpRequest();
request->setUrl("http://google.com/ ");
request->setRequestType(network::HttpRequest::Type::GET);
request->setResponseCallback([](network::HttpClient* sender, network::HttpResponse* response){
    if (!response->isSucceed()) {
        CCLOG("error");
        return;
    }

    std::vector<char>* buffer = response->getResponseData();
    for (unsigned int i = 0; i <buffer-> size (); i ++) {
        printf("%c", (* buffer)[i]);
    }
    printf("\n");
});

network::HttpClient::getInstance()->send(request);
request->release();
```

## 它是如何工作的...

1.  首先，创建一个`HttpRequest`实例。`HttpRequest`类没有`create`方法。这就是为什么你使用`new`来创建实例。

    ```cpp
    auto request = new network::HttpRequest();
    ```

1.  指定 URL 和请求类型。在这种情况下，将[`google.com/`](http://google.com/)设置为请求 URL，并将 GET 设置为请求类型。

    ```cpp
    request->setUrl("http://google.com/ "); request->setRequestType(network::HttpRequest::Type::GET);
    ```

1.  设置回调函数以接收来自服务器的数据。你可以使用`HttpResponse::isSucceed`方法检查其成功。然后，你可以使用`HttpResponse::getResponseData`方法获取响应数据。

    ```cpp
    request->setResponseCallback([](network::HttpClient* 
    sender, network::HttpResponse* response){ 
        if (!response->isSucceed()) { 
            CCLOG("error"); 
            return;
        }

        std::vector<char>* buffer = response- >getResponseData(); 
        for (unsigned int i = 0; i <buffer-> size (); i ++) { 
            printf("%c", (* buffer)[i]); 
        }
        printf("\n");
    });
    ```

1.  你可以通过调用指定`HttpRequest`类实例的`HttpClient::send`方法来请求网络。如果你通过网络获取响应，那么调用第 3 步中提到的回调函数。

    ```cpp
    network::HttpClient::getInstance()->send(request);
    ```

1.  最后，你必须释放`HttpRequest`实例。这就是你使用`new`创建它的原因。

    ```cpp
    request->release();
    ```

## 还有更多...

在本节中，你将学习如何使用`HttpRequest`类从网络获取资源。在下面的代码中，从网络获取谷歌日志并显示。

```cpp
auto request = new network::HttpRequest();
request- >setUrl("https://www.google.co.jp/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"); 
request->setRequestType(network::HttpRequest::Type::GET); 
request->setResponseCallback(&{ 
    if (!response->isSucceed()) { 
        CCLOG("error");
        return;
    }

    std::vector<char>* buffer = response->getResponseData(); 
    std::string path = FileUtils::getInstance()->getWritablePath() 
+ "image.png"; 
    FILE* fp = fopen(path.c_str(), "wb");
    fwrite(buffer->data(), 1, buffer->size(), fp);
    fclose(fp);

    auto size = Director::getInstance()->getWinSize();
    auto sprite = Sprite::create(path);
    sprite->setPosition(size/2);
    this->addChild(sprite);
});

network::HttpClient::getInstance()->send(request);
request->release();
```

构建并运行此代码后，你可以看到以下窗口。

![还有更多…](img/B0561_11_08.jpg)

### 小贴士

你必须将原始数据保存在沙盒中。你可以使用`FileUtils::getWritablePath`方法获取沙盒的路径。
