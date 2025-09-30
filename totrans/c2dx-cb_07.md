# 第 7 章. 与资源文件一起工作

游戏有很多资源，如图片和音频文件。Cocos2d-x 有一个资源管理系统。本章将涵盖以下主题：

+   选择资源文件

+   管理资源文件

+   使用 SQLite

+   使用 .xml 文件

+   使用 .plist 文件

+   使用 .json 文件

# 选择资源文件

你的游戏有每个分辨率的图片以支持多分辨率适配。如果你决定为每个分辨率找到一个图片，你的应用程序逻辑将非常复杂。Cocos2d-x 有一个搜索路径机制来解决这个问题。在这个菜谱中，我们将解释这个搜索路径机制。

## 准备工作

如果你想在不同的分辨率之间共享一些资源，那么你可以将所有共享资源放在 `Resources` 文件夹中，并将指定分辨率的资源放在不同的文件夹中，如下面的图片所示。

![准备工作](img/B00561_07_01.jpg)

`CloseNormal.png` 和 `CloseSelected.png` 是不同分辨率之间的共享资源。然而，`HelloWorld.png` 是指定分辨率的资源。

## 如何操作...

你可以按照以下方式设置 Cocos2d-x 搜索资源的优先级：

[PRE0]

## 它是如何工作的...

Cocos2d-x 将在 `Resources/ipad` 中找到 `HelloWorld.png`。Cocos2d-x 将使用此路径中的 `HelloWorld.png`；这就是为什么它可以在 `Resources/ipad` 中找到这个资源。然而，Cocos2d-x 不能在 `Resources/ipad` 中找到 `CloseNormal.png`。它将找到下一个顺序路径的 `Resources` 文件夹。系统可以在 `Resources` 文件夹中找到它并使用它。

你应该在创建第一个场景之前，在 `AppDelegate::applicationDidFinishLaunching` 方法中添加此代码。然后，第一个场景就可以使用这个搜索路径设置了。

## 参见

+   下一个菜谱中称为 *管理资源文件* 的搜索路径机制。

# 管理资源文件

Cocos2d-x 有一个管理资源的扩展，它被称为 `AssetsManagerExtension`。这个扩展是为了资源如图片和音频文件的热更新而设计的。你可以通过这个扩展更新游戏中的资源新版本，而无需更新你的应用程序。

## 准备工作

在使用 `AssetsManagerExtension` 之前，你应该了解它。这个扩展有许多有用的功能来帮助你进行热更新。以下是一些这些功能：

+   支持多线程下载

+   两级进度支持——文件级和字节级进度

+   支持压缩的 ZIP 文件

+   恢复下载

+   详细进度信息和错误信息

+   重试失败资源的可能性

你必须准备一个网络服务器，因此，你的应用程序将下载资源。

## 如何操作...

你需要上传资源和清单文件。在这种情况下，我们将更新 `HelloWorld.png` 和一个名为 `test.zip` 的 `.zip` 文件。这个 `.zip` 文件包含一些新的图片。`AssetsManagerExtension` 将根据清单文件下载资源。清单文件是 `version.manifest` 和 `project.manifest`。

`version.manifest` 文件包含以下代码：

[PRE1]

`project.manifest` 文件包含以下代码：

[PRE2]

然后，你必须上传这些清单文件和新资源。

接下来，你必须为热更新准备你的应用程序。你必须在你的项目中创建 `local.manifest` 文件。本地清单文件应包含以下代码：

[PRE3]

你应该在项目中创建一个管理 `AssetsManagerExtension` 的类。在这里，我们创建了一个名为 `ResourceManager` 的类。首先，你将创建 `ResourceManager` 的头文件。它被称为 `ResourceManager.h`。此文件包含以下代码：

[PRE4]

下一步是创建一个 `ResourceManager.cpp` 文件。此文件包含以下代码：

[PRE5]

最后，要开始更新资源，请使用以下代码：

[PRE6]

## 它是如何工作的...

首先，我们将解释清单文件和 `AssetsManagerExtension` 的机制。清单文件是 JSON 格式。本地清单和版本清单包含以下数据：

| 键 | 描述 |
| --- | --- |
| `packageUrl` | 资源管理器将尝试请求和下载所有资产的 URL。 |
| `remoteVersionUrl` | 允许检查远程版本的远程版本清单文件 URL，以确定是否已将新版本上传到服务器。 |
| `remoteManifestUrl` | 包含所有资产信息的远程清单文件 URL。 |
| `version` | 此清单文件的版本。 |

此外，远程清单中在名为 assets 的键中还有以下数据。

| 键 | 描述 |
| --- | --- |
| `key` | 每个键代表资产的相对路径。 |
| `Md5` | `md5` 字段表示资产的版本信息。 |
| `compressed` | 当压缩字段为 `true` 时，下载的文件将自动解压；此键是可选的。 |

`AssetsManagerExtension` 将按照以下步骤执行热更新：

1.  在应用程序中读取本地清单。

1.  根据本地清单中的远程版本 URL 下载版本清单。

1.  将本地清单中的版本与版本清单中的版本进行比较。

1.  如果两个版本不匹配，`AssetsManagerExtension` 将根据本地清单中的远程清单 URL 下载项目清单。

1.  将远程清单中的 `md5` 值与应用程序中资产的 `md5` 进行比较。

1.  如果两个 `md5` 值不匹配，`AssetsManagerExtension` 将下载此资产。

1.  下次，`AssetsManagerExtension` 将使用下载的版本清单而不是本地清单。

接下来，我们将解释 `ResourceManager` 类。你可以按照以下方式执行热更新：

[PRE7]

您应该通过指定本地清单的路径来调用 `ResourceManager::updateAssets` 方法。`ResourceManager::updateAssets` 将通过指定本地清单的路径和应用程序中存储的路径来创建一个 `AssetsManagerEx` 的实例，这是 `AssetsManagerExtension` 类的名称。

它将创建一个 `EventListenerAssetsManagerEx` 的实例以监听热更新的进度。

如果压缩值为真，`AssetsManagerExtension` 将在下载后解压它。

您可以通过调用 `AssetsManagerEx::update` 方法来更新资产。在更新过程中，您可以获取以下事件：

| 事件 | 描述 |
| --- | --- |
| `ERROR_NO_LOCAL_MANIFEST` | 无法找到本地清单。 |
| `UPDATE_PROGRESSION` | 获取更新的进度。 |
| `ERROR_DOWNLOAD_MANIFEST` | 下载清单文件失败。 |
| `ERROR_PARSE_MANIFEST` | 解析清单文件时出错。 |
| `ALREADY_UP_TO_DATE` | 已在更新资产（本地清单中的版本和版本清单中的版本相等）。 |
| `UPDATE_FINISHED` | 资产更新完成。 |
| `UPDATE_FAILED` | 更新资产时发生错误。在这种情况下，错误的原因可能是连接。您应该尝试再次更新。 |
| `ERROR_UPDATING` | 更新失败。 |
| `ERROR_DECOMPRESS` | 解压时发生错误。 |

当 `ResourceManager` 捕获到名为 `UPDATE_PROGRESSION` 的事件时，它会分发名为 `EVENT_PROGRESS` 的事件。如果您捕获到 `EVENT_PROGRESS`，您应该更新进度标签。

此外，如果它捕获到名为 `UPDATE_FINISHED` 的事件，它还会分发名为 `EVENT_FINISHED` 的事件。如果您捕获到 `EVENT_FINISHED`，您应该刷新所有纹理。这就是为什么我们要移除所有纹理缓存并重新加载场景。

[PRE8]

# 使用SQLite

您可以通过使用游戏中的数据库轻松地保存和加载游戏数据。在智能手机应用程序中，通常使用名为SQLite的数据库。SQLite易于使用。然而，在使用它之前，您必须设置一些事情。在本菜谱中，我们将解释如何在Cocos2d-x中设置和使用SQLite。

## 准备就绪

Cocos2d-x 没有SQLite库。您必须将SQLite的源代码添加到Cocos2d-x中。

您需要从网站 [http://sqlite.org/download.html](http://sqlite.org/download.html) 下载源代码。本书撰写时的最新版本是版本3.8.10。您可以下载此版本的 `.zip` 文件并将其展开。然后，您可以将生成的文件添加到您的项目中，如下面的图像所示：

![准备就绪](img/B00561_07_02.jpg)

在本菜谱中，我们将创建一个名为 `SQLiteManager` 的原始类。因此，您必须将 `SQLiteManager.h` 和 `SQLiteManager.cpp` 文件添加到您的项目中。

然后，如果您为Android构建，您必须按照以下方式编辑 `proj.android/jni/Android.mk`：

[PRE9]

## 如何做到这一点...

首先，您必须按照以下方式编辑 `SQLiteManager.h` 文件：

[PRE10]

接下来，你必须编辑`SQLiteManager.cpp`文件。这段代码有点长。所以，我们将一步一步地解释它。

1.  为单例类添加以下代码：

    [PRE11]

1.  添加打开和关闭数据库的方法：

    [PRE12]

1.  添加向数据库插入数据的方法：

    [PRE13]

1.  添加从数据库选择数据的方法：

    [PRE14]

1.  最后，这是如何使用这个类的方法。要插入数据，请使用以下代码：

    [PRE15]

    要选择数据，请使用以下代码：

    [PRE16]

## 工作原理...

首先，在`SQLiteManager`类的构造方法中，如果该类不存在，则创建一个名为data的表。数据表按以下SQL创建：

[PRE17]

为了使用SQLite，你必须包含`sqlite3.h`并使用sqlite3 API。这个API是用C语言编写的。如果你想学习它，你应该查看网站[http://sqlite.org/cintro.html](http://sqlite.org/cintro.html)。

我们在应用程序的沙盒区域创建了名为`test.sqlite`的数据库。如果你想更改位置或名称，你应该编辑`open`方法。

[PRE18]

你可以通过使用`insert`方法指定键和值来插入数据。

[PRE19]

此外，你可以通过使用`select`方法指定键来选择值。

[PRE20]

## 还有更多...

在本教程中，我们创建了`insert`方法和`select`方法。然而，你也可以执行其他SQL方法，如`delete`和`replace`。此外，你可以使数据库与你的游戏匹配。因此，你可能需要为此类编辑代码。

# 使用.xml文件

XML通常用作API的返回值。Cocos2d-x拥有TinyXML2库，可以解析XML文件。在本教程中，我们将解释如何使用这个库来解析XML文件。

## 准备工作

首先，你需要创建一个XML文件，并将其保存为`test.xml`，位于项目中的`Resources/res`文件夹。在这种情况下，我们将使用以下代码：

[PRE21]

要使用TinyXML-2库，你必须包含它并使用命名空间如下：

[PRE22]

## 如何操作...

你可以使用TinyXML2库来解析XML文件。在以下代码中，我们解析`test.xml`并记录其中的每个元素。

[PRE23]

这个结果是以下日志：

[PRE24]

## 工作原理...

首先，你必须创建`XMLDocument`的一个实例，然后使用`XMLDocument::LoadFile`方法解析`.xml`文件。要获取根元素，你必须使用`XMLDocument::RootElement`方法。基本上，你可以使用`FirstChildElement`方法获取元素。如果它是连续的元素，你可以使用`NextSiblingElement`方法获取下一个元素。如果没有更多元素，`NextSiblingElement`的返回值将是null。

最后，你不应该忘记删除`XMLDocument`的实例。这就是为什么你使用new操作创建它的原因。

# 使用.plist文件

在OS X和iOS中使用的PLIST是一个属性列表。文件扩展名是`.plist`，但实际上，PLIST格式是一个XML格式。我们经常使用`.plist`文件来存储游戏设置等。Cocos2d-x有一个类，通过它可以轻松地使用`.plist`文件。

## 准备工作

首先，您需要创建一个`.plist`文件，并将其保存为`test.plist`到您项目中的`Resources/res`文件夹。在这种情况下，它有两个键，即`foo`和`bar`。`foo`键有一个整数值`1`。`bar`键有一个字符串值`This is string`。请参考以下代码：

[PRE25]

## 如何操作...

您可以使用`FileUtils::getValueMapFromFile`方法来解析`.plist`文件。在以下代码中，我们解析`test.plist`并记录其中的键值。

[PRE26]

## 它是如何工作的...

您可以通过将`.plist`文件的路径指定给`FileUtils::getValueMapFromFile`方法来解析`.plist`文件。这样做之后，您将得到`.plist`文件中的数据作为`ValueMap`值。`ValueMap`类是一个基于`std::unordered_map`的包装类。PLIST的数据容器是`Array`和`Dictionary`。解析`.plist`文件后，`Array`是`std::vector<Value>`，而`Dictionary`是`std::unordered_map<std::string, Value>`。此外，您可以使用`Value::getType`方法来区分值的类型。然后，您可以使用`Value::asInt`、`asFloat`、`asDouble`、`asBool`和`asString`方法来获取值。

## 更多...

您可以从`ValueMap`保存`.plist`文件。这样做，您可以将游戏数据保存到`.plist`文件中。要保存`.plist`文件，请使用以下代码：

[PRE27]

首先，您需要在`ValueMap`中设置键值。在这种情况下，值都是整数类型，但您也可以设置混合类型的值。最后，您需要使用`FileUtils::writeToFile`方法将文件保存为`.plist`文件。

# 使用.json文件

我们可以将JSON格式像XML格式一样用于保存/加载游戏相关数据。JSON比XML格式简单。它比XML文件格式表示相同数据所需的空间更少。此外，今天，它被用作Web API的值。Cocos2d-x有一个名为**RapidJSON**的JSON解析库。在这个菜谱中，我们将解释如何使用RapidJSON。

## 准备工作

RapidJSON通常包含在Cocos2d-x中。然而，您需要按照以下方式包含头文件：

[PRE28]

## 如何操作...

首先，我们将按照以下方式解析一个JSON字符串：

[PRE29]

您可以使用`rapidjson::Document`来解析JSON，如下所示：

[PRE30]

## 它是如何工作的...

您可以通过使用`Document::Parse`方法并指定JSON字符串来解析JSON。当您使用`Document::HasParseError`方法时可能会得到解析错误；您可以通过使用`Document::GetParseError`方法来获取这个错误的描述。此外，您可以通过指定哈希键并使用`Document::GetString`方法来获取一个元素。

## 更多...

在实际应用中，你可以从一个文件中获取一个 JSON 字符串。现在我们将解释如何从文件中获取这个字符串。首先，你需要在项目的 `Resources/res` 文件夹中添加一个名为 `test.json` 的文件，并按照以下方式保存：

[PRE31]

接下来，你可以按照以下方式从一个文件中获取一个 JSON 字符串：

[PRE32]

你可以通过使用 `FileUtils::getStringFromFile` 方法从文件中获取字符串。之后，你可以以相同的方式解析。此外，这个 JSON 字符串可能是一个数组。你可以使用 `Document::IsArray` 方法检查格式是否为数组。然后，你可以使用 for 循环遍历数组中的 JSON 对象。
